import torch.nn as nn
import torch
from dataclasses import dataclass
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import List, Union
from utils import ColoredLogger
import os
from typing import Any, Dict

@dataclass
class LLMConfig:
    file_path: str
    model_name: str
    dtype: torch.dtype = torch.bfloat16

@dataclass
class VisualConfig:
    file_path: str
    model_name: str
    dtype: torch.dtype = torch.bfloat16

@dataclass  
class MultiModelConfig:
    projector_hidden_size: int = 4096
    dtype: torch.dtype = torch.bfloat16

class SimVLMModel(nn.Module):
    def __init__(self, llm_config: LLMConfig, visual_config: VisualConfig, multi_model_config: MultiModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_config = llm_config
        self.visual_config = visual_config
        self.multi_model_config = multi_model_config
        self._init_visual_encoder()
        self._init_LLM()
        self._get_visual_feature_size()
        self._get_llm_feature_size()
        self.projector = self._make_full_connected_layer(
            input_size=self.visual_feature_size,
            output_size=self.llm_feature_size,
            hidden_size=self.multi_model_config.projector_hidden_size
        )

    @classmethod
    def from_pretrained(
        cls, 
        projector_weights_path: str, 
        llm_config: LLMConfig, 
        visual_config: VisualConfig, 
        multi_model_config: MultiModelConfig,
        device_map: Union[Dict[str, str], str] = 'cuda',
        *args, 
        **kwargs
    ) -> 'SimVLMModel':
        """
        加载预训练的 VLM 模型（Projector）。
        
        Args:
            projector_weights_path (str): 仅包含 Projector 权重的 .pth 文件路径。
            llm_config, visual_config, multi_model_config: 用于初始化模型结构的配置。
        
        Returns:
            SimVLMModel: 加载了 Projector 权重的模型实例。
        """
        ColoredLogger.info(f"initializing SimVLMModel...")
        
        model = cls(llm_config, visual_config, multi_model_config, *args, **kwargs)
        # find the newest checkpoint file in the projector_weights_path
        if not os.path.exists(projector_weights_path):
            raise FileNotFoundError(f"Projector weights file not found at: {projector_weights_path}")
        
        weight_paths = os.listdir(projector_weights_path)
        if len(weight_paths) == 0:
            ColoredLogger.warning(f"No checkpoint found in {projector_weights_path}. Will from scratch.")
            model.to(device_map)
            return model
        else:
            def find_the_last_ckpt(weight_paths: list) -> str:
                saved_steps = [int(item.split('-')[-1]) for item in weight_paths]
                saved_steps.sort()
                ckpt_name = 'checkpoint-' + str(saved_steps[-1])
                return ckpt_name

            ckpt_name = find_the_last_ckpt(weight_paths)
            projector_weights_path = os.path.join(projector_weights_path, ckpt_name, 'projector_weights.pth')

            ColoredLogger.info(f"Loading projector weights from {projector_weights_path}...")
            state_dict = torch.load(projector_weights_path, map_location=device_map)
            load_info = model.projector.load_state_dict(state_dict, strict=True)
            
            ColoredLogger.info(f"Projector weights successfully loaded.")
            ColoredLogger.info(f"Missing keys (should be empty): {load_info.missing_keys}")
            ColoredLogger.info(f"Unexpected keys (should be empty): {load_info.unexpected_keys}")
            model.to(device_map)

            return model
    
    @classmethod
    def from_specific_checkpoint(
        cls, 
        checkpoint_path: str, 
        llm_config: LLMConfig, 
        visual_config: VisualConfig, 
        multi_model_config: MultiModelConfig,
        device_map: Union[Dict[str, str], str] = 'cuda',
        *args, 
        **kwargs
    ) -> 'SimVLMModel':
        """
        加载指定检查点的 VLM 模型（Projector）。
        
        Args:
            checkpoint_path (str): 包含 Projector 权重的特定检查点路径。
            llm_config, visual_config, multi_model_config: 用于初始化模型结构的配置。

        Returns:
            SimVLMModel: 加载了指定检查点权重的模型实例。
        """
        ColoredLogger.info(f"initializing SimVLMModel from checkpoint...")
        model = cls(llm_config, visual_config, multi_model_config, *args, **kwargs)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

        ColoredLogger.info(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device_map)
        model.load_state_dict(state_dict, strict=False)
        model.to(device_map)

        ColoredLogger.info(f"Checkpoint successfully loaded.")
        return model

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def _init_visual_encoder(self):
        ColoredLogger.info(f'Loading visual model from {self.visual_config.model_name}')
        ColoredLogger.info(f"Visual model torch dtype: {self.visual_config.dtype}")
        self.visual_model = CLIPModel.from_pretrained(
            os.path.join(self.visual_config.file_path, self.visual_config.model_name),
            dtype=self.visual_config.dtype
        )
        self._freeze_module(self.visual_model)
        ColoredLogger.info("Visual model loaded and frozen.")
        self.processor = CLIPProcessor.from_pretrained(os.path.join(self.visual_config.file_path, self.visual_config.model_name), use_fast=True)

    def _init_LLM(self):
        ColoredLogger.info(f'Loading LLM model from {self.llm_config.model_name}')
        ColoredLogger.info(f"LLM torch dtype: {self.llm_config.dtype}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            os.path.join(self.llm_config.file_path, self.llm_config.model_name),
            dtype=self.llm_config.dtype
        )
        self._freeze_module(self.llm)
        ColoredLogger.info("LLM model loaded and frozen.")
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.llm_config.file_path, self.llm_config.model_name))
    
    def _get_visual_feature_size(self):
        self.visual_feature_size = self.visual_model.config.projection_dim

    def _get_llm_feature_size(self):
        self.llm_feature_size = self.llm.get_input_embeddings().embedding_dim

    def _make_full_connected_layer(self, input_size, output_size, hidden_size):
        projector = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=self.multi_model_config.dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, dtype=self.multi_model_config.dtype)
        )
        for m in projector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return projector 
    
    def _fuse_features(self, input_ids, attention_mask, pixel_values) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        替换 <image> token 对应的 embedding 为视觉特征。
        """
        image_features = self.visual_model.get_image_features(pixel_values=pixel_values)
        image_features_aligned = self.projector(image_features)

        # get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        image_token_ids = self.tokenizer.encode("<image>", add_special_tokens=False)

        for b in range(input_ids.size(0)):  
            ids = input_ids[b].tolist()
            
            for start in range(len(ids) - len(image_token_ids) + 1):
                if ids[start:start + len(image_token_ids)] == image_token_ids:
                    mid = start + len(image_token_ids) // 2
                    inputs_embeds[b, mid] = image_features_aligned[b]
                    break

        return inputs_embeds, attention_mask


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, Any]:
        r"""
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            pixel_values: (batch_size, 3, H, W)
            labels(optimal): (batch_size, seq_length)
        Returns:
            outputs: CausalLMOutputWithPast(loss, logits, past_key_values, hidden_states, attentions)
        """
        fused_features, new_attention_mask = self._fuse_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # use fused features to generate output
        outputs = self.llm(inputs_embeds=fused_features, attention_mask=new_attention_mask, labels=labels)
        return outputs

    def named_parameters(self, prefix = "", recurse = True, remove_duplicate = True):
        for name, params in self.projector.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            yield name, params

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor, labels: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        fused_features, new_attention_mask = self._fuse_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # use fused features to generate output
        generated_ids = self.llm.generate(inputs_embeds=fused_features, attention_mask=new_attention_mask, *args, **kwargs)
        return generated_ids