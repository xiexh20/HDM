from configs.structured import ProjectConfig
from .model import ConditionalPointCloudDiffusionModel
from .model_coloring import PointCloudColoringModel
from .model_utils import set_requires_grad
from .model_diff_data import ConditionalPCDiffusionSeparateSegm
from .model_hoattn import CrossAttenHODiffusionModel

def get_model(cfg: ProjectConfig):
    if cfg.model.model_name == 'pc2-diff':
        model = ConditionalPointCloudDiffusionModel(**cfg.model)
    elif cfg.model.model_name == 'pc2-diff-ho-sepsegm':
        model = ConditionalPCDiffusionSeparateSegm(**cfg.model)
        print("Using a separate model to predict segmentation label")
    elif cfg.model.model_name == 'diff-ho-attn':
        model = CrossAttenHODiffusionModel(**cfg.model)
        print("Using separate model for human + object with cross attention.")
    else:
        raise NotImplementedError
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model


def get_coloring_model(cfg: ProjectConfig):
    model = PointCloudColoringModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model
