import warnings
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
from torch import nn

# Change relative import to absolute import
from all_utils.flops_counter import calculate_flops

from ..factorization._interface import BaseFactorization
from ._interface import BaseSearch
from .uniform import UNIFORMSearch

class LastFeatureHook:
    def __init__(self, model: nn.Module):
        self.model = model

        self.hooks = []
        self.cp_modules = reversed(
            [
                (name, module_sub)
                for name, module_sub in model.named_modules()
                # if all(omit not in name for omit in name_omit)
                if isinstance(module_sub, nn.Linear)
            ]
        )

    def _hook_fn(self, layer_name):
        def get_feature_extract_hook(module, input, output):
            if "head" in layer_name:
                x = input[0].detach().float()
                if x.dim() > 3:
                    x = x.reshape(x.shape[0], -1, x.shape[-1])
                elif x.dim() == 2:
                    x = x.unsqueeze(0)
                self.model.last_feat = x.clone()
                # self.model.last_feat = output   # if layer: blocks.-1.mlp.fc2

        return get_feature_extract_hook

    def _register_hooks_recursive(self):
        for name, layer in self.cp_modules:
            if layer.out_features < 10:
                continue  # for some head matrix, such as image-text match head

            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)
            # if "head" in name: # continue
            return

    def attach_hooks(self):
        self._register_hooks_recursive()

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()


@dataclass
class ErrorMetrics:
    CE_loss: float = 0.0
    feat_mse: float = 0.0
    l_sr: float = 0.0


@dataclass
class LayerDecomposition:
    input_shape: Tuple[int, ...]
    mat_l: torch.Tensor
    mat_r: torch.Tensor
    eq_rank: int
    next_to_eval: int = field(init=False)
    T_err: ErrorMetrics = field(default_factory=ErrorMetrics)
    Best_err: ErrorMetrics = field(
        default_factory=lambda: ErrorMetrics(
            CE_loss=float("inf"),
            feat_mse=float("inf"),
            l_sr=float("inf"),
        )
    )
    Lo_k: int = 0
    Up_k: int = 0
    best_rank: Optional[int] = field(init=False)
    lat_max_rank: Optional[int] = None

    def __post_init__(self):
        self.best_rank = self.eq_rank
        # only searching increments of 8
        self.Lo_k = 1
        self.Up_k = self.eq_rank // 8
        # set initial rank to be evaluated for binary search
        mid = (self.Lo_k + self.Up_k) // 2
        self.next_to_eval = mid * 8


class FLAR_SVDSearch(BaseSearch):
    # TODO: add option for iterative compression of the reference model.
    def __init__(
        self,
        eval_data,
        name_omit: List[str] = None,
        threshold: float = 0.01,
        target_metric = "params",
        ratio_target: float = 0.5,
        progressive_comp: bool = False,
        stage_name_in_current_model: str = "stages",
        mixup_fn=None,
        latency_predictor_path: str = "/workspace/FLAR-SVD/rf_model_kx8_fp32_80us.pkl",
    ):
        self.eval_data = tuple(data for data in eval_data)  # eval_data
        self.mixup_fn = mixup_fn
        self.criteria = SoftTargetCrossEntropy()
        self.name_omit = name_omit or []
        self.dev = torch.device(torch.cuda.current_device())
        self.threshold = threshold
        self.target_metric = target_metric
        self.ratio_target = ratio_target if threshold is None else None
        self.progressive_comp = progressive_comp
        self.stage_name_in_current_model = stage_name_in_current_model
        if latency_predictor_path:
            print(
                f"Found latency predictor, {latency_predictor_path.split('/')[-1]}. Start loading..."
            )
            self.latency_predictor = joblib.load(latency_predictor_path)
        else:
            self.latency_predictor = None
            print("Latency predictor not found. Latency predictor disabled.")

    def initialize_search(self, lrd_method: BaseFactorization, model: nn.Module):
        self.lrd_method = lrd_method

    def _get_valid_layer_names(
        self, block: nn.Module, stage_idx: int, block_idx: int
    ) -> List[str]:
        layer_names = []
        copied_modules = {
            name: module_sub
            for name, module_sub in block.named_modules()
            if all(omit not in name for omit in self.name_omit)
            and isinstance(module_sub, nn.Linear)
        }
        for name, module_sub in copied_modules.items():
            if module_sub.out_features < 10:
                continue  # for some head matrix, such as image-text match head

            if stage_idx == -1:
                full_name = f"blocks.{block_idx}.{name}"
            else:
                full_name = (
                    f"{self.stage_name_in_current_model}.{stage_idx}."
                    f"blocks.{block_idx}.{name}"
                )
            layer_names.append(full_name)
        return layer_names

    def _initialize_block_search_dict(
        self, block: nn.Module, layer_names: List[str]
    ) -> Dict[str, LayerDecomposition]:
        block_search_dict = {}
        try:
            block_modules = dict(block.named_modules())
        except:
            # this is using arbitrary groups that are easier to use on unknown models.
            block_modules = block
        for layer_name, local_name in layer_names:
            # obtain the original layer
            smodule: nn.Linear = block_modules[local_name]
            # get the input shape
            input_shape = self.lrd_method.input_shapes[layer_name]
            # get the low rank decomposition
            factorized_matrix = self.lrd_method.factorize_matrix(
                smodule.weight, ratio=1.0, name=layer_name
            )
            print(
                f"Layer {layer_name} with input shape {input_shape} and"
                f" rank {factorized_matrix.eq_rank}"
            )
            block_search_dict[layer_name] = LayerDecomposition(
                input_shape=input_shape,
                mat_l=factorized_matrix.mat_l,
                mat_r=factorized_matrix.mat_r,
                eq_rank=factorized_matrix.eq_rank,
            )
        return block_search_dict

    def _compress_layer(
        self,
        layer_name: str,
        model_copy: nn.Module,
        block_dict: Dict[str, LayerDecomposition],
    ) -> Tuple[torch.Tensor, nn.Linear]:
        eval_rank = block_dict[layer_name].next_to_eval
        mat_l = block_dict[layer_name].mat_l[:, :eval_rank].to(self.dev)
        mat_r = block_dict[layer_name].mat_r[:eval_rank, :].to(self.dev)

        # replace original weight with reconstructed representation
        LR_recon = mat_l @ mat_r
        smodule: nn.Linear = dict(model_copy.named_modules())[layer_name]
        og_weight = smodule.weight.data.detach().clone()
        smodule.weight.data.copy_(LR_recon.to(self.dev))
        del mat_l, mat_r, LR_recon
        return og_weight, smodule

    def _eval_current_search_step(
        self, model, block_search_dict: Dict[str, LayerDecomposition]
    ):
        model_copy = deepcopy(model).to(self.dev)
        model = model.eval().to(self.dev)

        # attach hooks to extract a feature from the last layer
        hook_register_model = LastFeatureHook(model)
        hook_register_model_copy = LastFeatureHook(model_copy)
        hook_register_model.attach_hooks()
        hook_register_model_copy.attach_hooks()

        # block with new rank setting.
        for data, target in self.eval_data:
            with torch.cuda.device(self.dev):
                torch.cuda.empty_cache()

            if self.mixup_fn is not None:
                data, target = self.mixup_fn(data, target)

            # evaluate current batch for all layers in the block separately
            for layer_name in block_search_dict:
                # check if search is already done for this layer.
                if (
                    block_search_dict[layer_name].Lo_k
                    >= block_search_dict[layer_name].Up_k
                ):
                    continue

                # compress the layer based on the search step setting
                original_weight, compressed_layer = self._compress_layer(
                    layer_name, model_copy, block_search_dict
                )

                with torch.no_grad():
                    data, target = data.to(self.dev), target.to(self.dev)
                    logit_s = model_copy(data)
                    logit_t = model(data)

                    # Obtain normalized losses for later comparison
                    L_fm = F.mse_loss(model.last_feat, model_copy.last_feat)
                    L_fm = L_fm / torch.mean(model_copy.last_feat**2)
                    L_sr = F.mse_loss(logit_s, logit_t).to(self.dev)
                    L_sr = L_sr / torch.mean(logit_t**2)
                    CE_loss = self.criteria(logit_s, target).to(self.dev)

                # Restore original weight
                compressed_layer.weight.data.copy_(original_weight)

                del logit_s, logit_t, original_weight, compressed_layer

                block_search_dict[layer_name].T_err.CE_loss += CE_loss
                block_search_dict[layer_name].T_err.feat_mse += L_fm
                block_search_dict[layer_name].T_err.l_sr += L_sr
            del data, target
        # calculate average losses for all layers in the block
        n_batch_tensor = torch.tensor(len(self.eval_data), device=self.dev)
        for layer_name in block_search_dict:
            block_search_dict[layer_name].T_err.CE_loss /= n_batch_tensor.item()
            block_search_dict[layer_name].T_err.feat_mse /= n_batch_tensor.item()
            block_search_dict[layer_name].T_err.l_sr /= n_batch_tensor.item()

        hook_register_model.clear_hooks()
        hook_register_model_copy.clear_hooks()

    def _get_latency_pred(self, input_shape, rank):
        input2regressor = input_shape
        input2regressor[4] = rank
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lat_predict = self.latency_predictor.predict([input2regressor])
        return lat_predict

    def _search_step(self, block_dict: Dict[str, LayerDecomposition]):
        for name in block_dict:
            if block_dict[name].Lo_k >= block_dict[name].Up_k:
                continue
            last_evaluated = block_dict[name].next_to_eval

            # check error criteria
            if block_dict[name].T_err.feat_mse <= self.threshold:
                # increase upper bound
                block_dict[name].Up_k = last_evaluated // 8
                # save best
                block_dict[name].best_rank = last_evaluated
                block_dict[name].Best_err = deepcopy(block_dict[name].T_err)
            else:
                # increase lower bound
                block_dict[name].Lo_k = last_evaluated // 8 + 1

            # set next evaluation rank
            mid = (block_dict[name].Up_k + block_dict[name].Lo_k) // 2
            block_dict[name].next_to_eval = mid * 8

            # Clear error for next iteration
            block_dict[name].T_err = ErrorMetrics()

    def _tile_model_in_groups(
        self, model: nn.Module, group_size: int = 5
    ) -> Tuple[List[nn.Module], List[List[str]]]:
        """
        Tiles the model into groups of layers to accelerate searching.
        Note that it would work the same without those groups, but it
        would be slower.
        Each group contains `group_size` layers.
        Returns a list of groups and their corresponding layer names.
        """
        groups = []
        groups_layer_names = []
        current_group = {}
        current_layer_names = []
        idx = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and all(
                omit not in name for omit in self.name_omit
            ):
                current_group[f"{idx}"] = module  # Use idx as short name for the layer
                current_layer_names.append(
                    (name, f"{idx}")
                )  # Store full name and short name
                idx += 1

                if len(current_group) == group_size:
                    groups.append(current_group)
                    groups_layer_names.append(current_layer_names)
                    current_group = {}
                    current_layer_names = []
                    idx = 0

        # Add any remaining layers as a smaller group
        if current_group:
            groups.append(current_group)
            groups_layer_names.append(current_layer_names)

        return groups, groups_layer_names

    def search(self, model: nn.Module) -> Dict[str, int]:
        self._init_threshold(model) if self.ratio_target else None
        compression_dict = {}
        groups, groups_layer_names = self._tile_model_in_groups(model, group_size=4)
        for group, layer_names in zip(groups, groups_layer_names):
            blk_layerwise_compression_dict = self._search_group_wise(
                model=model, group=group, layer_names=layer_names
            )
            compression_dict.update(blk_layerwise_compression_dict)
        if self.ratio_target:
            # Interpolate ranks to ensure target ratio is met
            compression_dict = self._interpolate_ranks(compression_dict, model)
        else:
            compression_dict = self._basic_latency_rank_adjustment(compression_dict, model)

        return compression_dict

    def search_blockwise(self, model: nn.Module, stage_name: str, calib_data=None):
        self._init_threshold(model) if self.ratio_target else None
        compression_dict = {}
        blocks, blocks_layer_names = self.get_model_blocks(model, stage_name)
        for block, layer_names in zip(blocks, blocks_layer_names):
            blk_layerwise_compression_dict = self._search_group_wise(
                model=model, group=block, layer_names=layer_names
            )
            compression_dict.update(blk_layerwise_compression_dict)
        if self.ratio_target:
            # Interpolate ranks to ensure target ratio is met
            compression_dict = self._interpolate_ranks(compression_dict, model)
        else:
            compression_dict = self._basic_latency_rank_adjustment(compression_dict, model)

        return compression_dict

    def _search_group_wise(
        self, model: nn.Module, group: nn.Module, layer_names: list
    ) -> Dict[str, int]:
        # get decomposed representation for all layers
        blk_search_dict = self._initialize_block_search_dict(group, layer_names)
        if len(blk_search_dict) == 0:
            return {}

        # determine search steps based on highest rank within a block
        max_rank = max(decomp_dict.eq_rank for decomp_dict in blk_search_dict.values())
        search_steps = range(int(np.ceil(np.log2(max_rank))))

        # search for optimal rank for that block
        for _ in search_steps:
            # evaluate the model with current rank settings
            self._eval_current_search_step(model, blk_search_dict)
            # perform one step in search based on the results
            self._search_step(blk_search_dict)

        # create layerwise compression dict
        blk_layerwise_compression_dict = {}
        for key in blk_search_dict.keys():
            eq_rank = blk_search_dict[key].eq_rank
            best_rank = blk_search_dict[key].best_rank
            if best_rank == eq_rank:
                print(key, "no compression.")
                blk_layerwise_compression_dict[key] = -1
            else:
                blk_layerwise_compression_dict[key] = best_rank

        if self.progressive_comp:
            for key in blk_layerwise_compression_dict.keys():
                if blk_layerwise_compression_dict[key] == -1:
                    continue
                blk_search_dict[key].next_to_eval = blk_search_dict[key].best_rank
                self._compress_layer(key, model, blk_search_dict)
        return blk_layerwise_compression_dict

    def _init_threshold(self, model: nn.Module):
        '''
        Initializes the threshold for the search initialized by a uniform search to
        determine the expected compression ratio and the expected FLOPS reduction.
        '''
        uniform_search = UNIFORMSearch(
            eval_data=self.eval_data,
            mixup_fn=self.mixup_fn,
            name_omit=self.name_omit,
            ratio_target=self.ratio_target,
            stage_name_in_current_model=self.stage_name_in_current_model,
        )
        layer_compression_dict = uniform_search.search(model)
        model_cp = deepcopy(model)
        self.lrd_method.factorize_model(
            model_cp, layer_compression_dict, name_omit=self.name_omit
        )
        self.full_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.full_flops = float(calculate_flops(model, print_results=False)[0].split(" ")[0])
        self.expected_params = self.full_params * self.ratio_target
        actual_params = sum(p.numel() for p in model_cp.parameters() if p.requires_grad)
        self.actual_flops = float(calculate_flops(model_cp, print_results=False)[0].split(" ")[0])
        self.corrected_ratio = (self.ratio_target*(self.full_params + self.expected_params) - (actual_params - self.expected_params)) / (self.full_params - actual_params)
        self.corrected_ratio = self.ratio_target - (1 - self.ratio_target)*(actual_params - self.expected_params) / (self.full_params - actual_params)
        self.expected_flops = self.actual_flops - (self.full_flops - self.actual_flops)*(self.ratio_target - self.corrected_ratio) / (1 - self.ratio_target)
        print(f"Corrected ratio: {self.corrected_ratio:.3f}")

        hook_reg = LastFeatureHook(model)
        hook_reg_cp = LastFeatureHook(model_cp)
        hook_reg.attach_hooks()
        hook_reg_cp.attach_hooks()
        model = model.eval().to(self.dev)
        model_cp = model_cp.eval().to(self.dev)
        # feat_mse = torch.zeros(1, device=self.dev)
        feat_mse = 0

        # TODO!! evaluate if its better to obtain the error per changed layer
        # or a hollistic one (currently implemented) 
        for data, target in self.eval_data:
            with torch.cuda.device(self.dev):
                torch.cuda.empty_cache()
            if self.mixup_fn is not None:
                data, target = self.mixup_fn(data, target)
            with torch.no_grad():
                data, target = data.to(self.dev), target.to(self.dev)
                model(data)
                model_cp(data)

                # Obtain normalized losses for later comparison
                L_fm = F.mse_loss(model.last_feat, model_cp.last_feat)
                L_fm = L_fm / torch.mean(model_cp.last_feat**2)
                feat_mse += L_fm

        # calculate average losses for all layers in the block
        n_batch_tensor = torch.tensor(len(self.eval_data), device=self.dev)
        feat_mse /= n_batch_tensor.item()
        hook_reg.clear_hooks()
        hook_reg_cp.clear_hooks()

        n_layers = 0
        for name, layer in dict(model.named_modules()).items():
            if any(omit in name for omit in self.name_omit):
                continue
            if isinstance(layer, nn.Linear):
                n_layers += 1
        self.threshold = feat_mse/n_layers if n_layers > 0 else 0
        print(f"Number of layers in model: {n_layers}")
        print(f"Initial error threshold: {self.threshold}\nLast feat error: {feat_mse.item()}")

    def _basic_latency_rank_adjustment(self, layerwise_rank_dict, model):
        print("Adjusting ranks based on latency predictor...")
        for layer_name, rank in layerwise_rank_dict.items():
            if rank != -1:
                # Filter out those ranks that do not comply with our requirements.
                if layer_name in self.lrd_method.input_shapes:
                    input_shape = self.lrd_method.input_shapes[layer_name]
                    lat_predict = self._get_latency_pred(input_shape, layerwise_rank_dict[layer_name])
                    # Check if the predicted latency lower than the uncompressed layer
                    if lat_predict.item() >= 1.0 or lat_predict.item() <= 0:
                        layerwise_rank_dict[layer_name] = -1
        return layerwise_rank_dict

    def _interpolate_ranks(
        self, layerwise_rank_dict: Dict[str, int], model: nn.Module
    ) -> Dict[str, int]:
        """
        Interpolates the ranks for the layers based on the layerwise_rank_dict.
        This is used to ensure that the ranks are evenly distributed across the layers.
        """
        tmp_model = deepcopy(model)
        self.lrd_method.factorize_model(tmp_model, layerwise_rank_dict, name_omit=self.name_omit, verbose=False)
        num_params = sum(p.numel() for p in tmp_model.parameters())
        num_flops = float(calculate_flops(tmp_model, print_results=False)[0].split(" ")[0])

        if self.target_metric == "params":
            curr_rate = self.corrected_ratio - (1 - self.corrected_ratio) * (self.expected_params - num_params) / (self.full_params - self.expected_params)
        else:    # by default match flops
            curr_rate = self.corrected_ratio - (1 - self.corrected_ratio) * (self.expected_flops - num_flops) / (self.full_flops - self.expected_flops)
        print(f"Current rate: {curr_rate}")
        print(f"Correction: {(self.corrected_ratio - curr_rate)*100} %")

        for layer_name, rank in layerwise_rank_dict.items():
            # Adjust rank based on target compression ratio
            if rank != -1:
                adjusted_rank = rank * self.corrected_ratio / curr_rate
                layerwise_rank_dict[layer_name] = int((adjusted_rank // 8) * 8) # ensure divisibility by 8
                
                # Filter out those ranks that do not comply with our requirements.
                if layer_name in self.lrd_method.input_shapes:
                    input_shape = self.lrd_method.input_shapes[layer_name]
                    lat_predict = self._get_latency_pred(input_shape, layerwise_rank_dict[layer_name])
                    # Check if the predicted latency lower than the uncompressed layer
                    if lat_predict.item() >= 1.0 or lat_predict.item() <= 0:
                        layerwise_rank_dict[layer_name] = -1
        
        del tmp_model

        return layerwise_rank_dict
