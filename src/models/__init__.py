from src.models.mlp import mlp_model


def get_model(model_cfg, input_shape, seed=None):
    if model_cfg.name == 'mlp':
        return mlp_model(model_cfg, input_shape, seed)
