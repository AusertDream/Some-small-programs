import logging
from typing import List, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import os

class ColoredLogger:
    """
    根据不同的日志级别，打印不颜色的日志
    info：绿色
    warning:黄色
    error：红色
    debug：灰色
    """
    #logging日志格式设置
    logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s-%(levelname)s:%(message)s')
    @staticmethod
    def info(message: str):
        #info级别的日志，绿色
        logging.info("\033[0;32m" + message + "\033[0m")
    @staticmethod
    def warning(message: str):
        #warning级别的日志，黄色
        logging.warning("\033[0;33m" + message + "\033[0m")
    @staticmethod
    def error(message: str):
        # error级别的日志，红色
        logging.error("\033[0;31m"+"-" * 120 +"\n| " + message +"\033[0;31m" + "\n" + "L"+"-"* 150)
    @staticmethod
    def debug(message: str):
        #debug级别的日志，灰色
        logging.debug("\033[0;37m" + message +"\033[0m")


def save_model(model: Any, save_path: str, training_step: int):
    r"""
    Save the model to the specified path.
    Args:
        model: SimVLMModel
        save_path: str
        training_step: int
    """
    ckpt_path = os.path.join(save_path, f'checkpoint-{training_step}')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    saved_file_path = os.path.join(ckpt_path, 'projector_weights.pth')
    ColoredLogger.info(f'Saving model at step {training_step} to {saved_file_path}')
    import torch
    torch.save(model.projector.state_dict(), saved_file_path)
    ColoredLogger.info('Model saved successfully.')

class SimVLMDataset(Dataset):
    def __init__(self, qa_file_path: str, image_file_path: str):
        self.qa_file_path = qa_file_path
        self.image_file_path = image_file_path
        ColoredLogger.info(f'Loading dataset from {self.qa_file_path}')
        ColoredLogger.info(f'Image files located at {self.image_file_path}')
        self.data: List[Dict[str, Any]] = []
        with open(self.qa_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                import json
                single_data = json.loads(line)
                self.data.append(single_data)

    def __len__(self) -> int:
        return len(self.data)
    
    def _get_image(self, image_name):
        return Image.open(os.path.join(self.image_file_path, image_name)).convert('RGB')


    def __getitem__(self, idx):
        raw_sample = self.data[idx]
        return {
            "conversations": raw_sample['conversations'],
            "image": raw_sample['image'],
        }

    def get_first_sample(self) -> Dict[str, Any]:
        return self.__getitem__(0)
