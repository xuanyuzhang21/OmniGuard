import importlib


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    
    # DictConfig don't support function, so we transform it into a dict
    params = dict(config.get("params", dict()))
    for key in config.get("func_keys", list()):
        params[key] = get_obj_from_str(params[key])
    for key in config.get("cls_keys", list()):
        params[key] = instantiate_from_config(params[key])
    
    return get_obj_from_str(config["target"])(**params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)