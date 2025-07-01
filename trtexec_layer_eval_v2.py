import gc
import os
import random
import subprocess as subp
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn

def main(inp_shape, in_feats, out_feats, dev, csv_file, counter):
    """
    Evaluates latency characteristics of linear layers at different compression levels.
    
    First measures the baseline latency of an uncompressed linear layer, then establishes
    a lower bound with a minimal rank-1 compressed version. Finally, evaluates multiple
    compression ratios and records their normalized performance relative to these bounds.
    """
    file_name = None

    # Uncompressed model
    model = nn.Linear(in_feats, out_feats).to(dev)
    torch2onnx(model, inp_shape, dev, counter)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Evaluating...    Input shape:", inp_shape)
    result, _ = eval_layer("Uncompressed", counter, dev)
    og_lat = float(result.split("\n")[2].split("mean = ")[1].split(" ms")[0])

    # Minimal rank setup
    rank = 1
    model = nn.Sequential(
        nn.Linear(in_feats, rank, bias=False), nn.Linear(rank, out_feats)
    ).to(dev)
    torch2onnx(model, inp_shape, dev, counter)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Layer Compressed to rank 1 ...")
    result, _ = eval_layer(f"Rank: {rank}", counter, dev)
    min_lat = float(result.split("\n")[2].split("mean = ")[1].split(" ms")[0])
    
    # Calculate the equivalent rank for compression
    # This is the rank that would give a similar number of parameters as the original layer
    eq_rank = (in_feats * out_feats) / (in_feats + out_feats)

    # Now, evaluate the different possible ranks
    min_val = random.choice(range(10, 51, 10))
    step = random.choice((5, 10))
    for comp_rate in reversed(range(min_val, 101, step)):
        # # Optionally enforce rank to be a multiple of 8
        # if random.choice((0, 1)):
        #     rank = int(eq_rank * comp_rate / 100)
        # else:
        #     rank = int(np.round((eq_rank * comp_rate / 100) / 8) * 8)

        # Enforce rank to be a multiple of 8
        rank = int(np.round((eq_rank * comp_rate / 100) / 8) * 8)

        model = nn.Sequential(
            nn.Linear(in_feats, rank, bias=False), nn.Linear(rank, out_feats)
        ).to(dev)
        torch2onnx(model, inp_shape, dev, counter)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Layer Compressed to rank: {rank} ({comp_rate}%)...")
        result, _ = eval_layer(f"Rank: {rank} ({comp_rate}%)", counter, dev)
        # print(result.split("\n")[2])
        comp_lat = float(result.split("\n")[2].split("mean = ")[1].split(" ms")[0])

        output = (comp_lat - min_lat) / (og_lat - min_lat)
        with open(csv_file, "a") as f:
            f.write(
                f"{inp_shape[0]},{inp_shape[1]},{inp_shape[2]},{out_feats},{rank},"
                + f"{og_lat},{comp_lat},{min_lat},{output}\n"
            )

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
        gc.collect()

    run_command('echo -e "Evaluation completed.\n"', file_name)
    print(f"Evaluation results saved to {csv_file}")


@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull.
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def torch2onnx(model, input_shape, dev, counter):
    """
    Export a PyTorch model to ONNX format.
    """
    with suppress_stdout_stderr():
        torch.onnx.export(
            model,
            torch.randn(tuple(input_shape), dtype=torch.float32, device=f"cuda:{dev}"),
            f"/workspace/tmp_{counter}.onnx",
            input_names=["inp"],
            output_names=["outp"],
        )


os.openpty


def run_command(command, file_name=None):
    """
    Run a shell command and return its output.
    Args:
        command (str): The shell command to run.
        file_name (str, optional): If provided, the output will be appended to this file.
    Returns:
        tuple: A tuple containing the standard output and standard error of the command.
    """
    if file_name is not None:
        command += f" >> {file_name}"
    result = subp.run(command, shell=True, stdout=subp.PIPE, stderr=subp.PIPE)
    # print(result.stdout.decode("utf-8"))
    return result.stdout.decode("utf-8"), result.stderr.decode("utf-8")
    # Example usage:        stdout, stderr = run_command('ls -l')


def eval_layer(title, counter, dev):
    """
    Evaluate an individual layer saved as ONNX file using trtexec.
    """
    # run_command(f'echo -e "{title}"')
    print(title)
    return run_command(
        f"trtexec --onnx=/workspace/tmp_{counter}.onnx --fp16"
        # f"trtexec --onnx=/workspace/tmp_{counter}.onnx"
        + " --noDataTransfers --useCudaGraph --useSpinWait"
        # + " --iterations=15 --warmUp=500 --duration=5" +
        + f" --device={dev}| tail -n 12"
    )


if __name__ == "__main__":
    batch_size = 1
    dev = torch.cuda.current_device()

    csv_file = "/workspace/layer_wise_latency.csv"
    base_csv_file = csv_file
    counter = 0
    while os.path.exists(csv_file):
        csv_file = f"{base_csv_file.rsplit('.', 1)[0]}({counter}).csv"
        counter += 1

    with open(csv_file, "w") as f:
        f.write(
            "Batch_dim,Patch_dim,In_feats,Out_feats,Rank,"
            + "Og_lat,Latency,Min_lat,Lat_recovery,\n"
        )

    while True:
        a = random.randint(0, 3)
        input_shape = [
            batch_size * 2 ** random.randint(0, 6 - a),  # Random power of 2 up to 64
            random.choice((7, 8, 12, 14, 16, 30, 32))
            ** 2,  # Square power of patch_size
            random.choice((96, 128)) * 2**a,  # Multiple of 96, 128
        ]
        input_shape[1] += random.choice((0, 1))
        if random.choices((0, 1), weights=(0.25, 0.75))[0]:
            in_feats = input_shape[2]
            out_feats = in_feats * random.choice((1, 3, 4))
        else:
            out_feats = input_shape[2]
            in_feats = input_shape[2] = out_feats * 4

        main(input_shape, in_feats, out_feats, dev, csv_file, counter)
