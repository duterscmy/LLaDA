from opencompass.models import LLaDABaseModel

models = [
    dict(
        type=LLaDABaseModel,
        abbr='llada-8b-base',
        path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Base',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
