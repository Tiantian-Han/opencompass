import argparse
import os
import os.path as osp
import random
import sys
import time
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg, task_abbr_from_cfg)


@TASKS.register_module()
class OpenICLInferTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run inference on a model with a single dataset.
    """

    name_prefix = 'OpenICLInfer'
    log_subdir = 'logs/infer'
    output_subdir = 'predictions'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.num_gpus = self.cfg.get('num_gpus', 0)
        self.model_cfg = self.cfg['model']
        self.dataset_cfg = self.cfg['dataset']

    def get_command(self, cfg_path, template, **kwargs) -> str:
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template for the command.
        """
        script_path = __file__
        if self.num_gpus > 0:
            # torchrun will be used to launch the script
            return template.format(task_cmd=f'{script_path} {cfg_path}')
        return template.format(task_cmd=f'{sys.executable} {script_path} {cfg_path}')

    def run(self, cur_model=None, cur_model_abbr=None):
        self.logger.info(f'Task {self.name} begin')
        if cur_model is not None and self.model_cfg['abbr'] == cur_model_abbr:
            self.model = cur_model
        else:
            self.model = build_model_from_cfg(self.model_cfg)

        self.generation_kwargs = self.model.generation_kwargs
        
        self.dataset = build_dataset_from_cfg(self.dataset_cfg)
        self.inferencer = OpenICLInferencer(
            model=self.model,
            dataset=self.dataset,
            output_json_filepath=self.output_path,
            generation_kwargs=self.generation_kwargs,
            **self.cfg.infer.inferencer.to_dict(),
        )
        self.inferencer.inference()

    def _inference(self):
        self.logger.info(
            f'Start inferencing {task_abbr_from_cfg(self.sub_cfg)}')

        assert hasattr(self.infer_cfg, 'ice_template') or hasattr(self.infer_cfg, 'prompt_template'), \
            'Both ice_template and prompt_template cannot be None simultaneously.'  # noqa: E501
        if hasattr(self.infer_cfg, 'ice_template'):
            ice_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['ice_template'])

        if hasattr(self.infer_cfg, 'prompt_template'):
            prompt_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['prompt_template'])

        retriever_cfg = self.infer_cfg['retriever'].copy()
        retriever_cfg['dataset'] = self.dataset
        retriever = ICL_RETRIEVERS.build(retriever_cfg)

        # set inferencer's default value according to model's config'
        inferencer_cfg = self.infer_cfg['inferencer']
        inferencer_cfg['model'] = self.model
        self._set_default_value(inferencer_cfg, 'max_out_len',
                                self.max_out_len)
        self._set_default_value(inferencer_cfg, 'min_out_len',
                                self.min_out_len)
        self._set_default_value(inferencer_cfg, 'batch_size', self.batch_size)
        inferencer_cfg['max_seq_len'] = self.model_cfg.get('max_seq_len')
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)

        out_path = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        out_dir, out_file = osp.split(out_path)
        mkdir_or_exist(out_dir)

        if hasattr(self.infer_cfg, 'prompt_template') and \
                hasattr(self.infer_cfg, 'ice_template'):
            inferencer.inference(retriever,
                                 ice_template=ice_template,
                                 prompt_template=prompt_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file)
        elif hasattr(self.infer_cfg, 'prompt_template'):
            inferencer.inference(retriever,
                                 prompt_template=prompt_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file)
        else:
            inferencer.inference(retriever,
                                 ice_template=ice_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file)

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        if key not in cfg:
            cfg[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = OpenICLInferTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
