from mmengine.config import read_base
import os
import sys
# 将本地 opencompass 的父目录插入到最前面
local_path = '/lus/lfs1aip2/projects/public/u6er/mingyu/llada/opencompass'
if local_path in sys.path:
    sys.path.remove(local_path)
sys.path.insert(0, local_path)

# 验证导入路径
import opencompass
print(f"Using opencompass from: {opencompass.__file__}")
with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import \
        humaneval_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models
datasets = humaneval_datasets
models = llada_instruct_8b_models
eval_cfg = {'gen_blocksize': 256, 'gen_length': 256, 'gen_steps': 256, 'batch_size_': 1, 'batch_size': 1, 'diff_confidence_eos_eot_inf': False, 'diff_logits_eos_inf': True}
for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,    
        num_split=None,   
        min_task_size=16, 
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)