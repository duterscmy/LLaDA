from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-Instruct-JustGRPO-GSM8K',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
