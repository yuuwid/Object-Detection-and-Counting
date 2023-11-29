from .config import model_types, trained_urls


def show_downloadeable_models():
    print('\nAvailable ReID models for automatic download')
    print(list(trained_urls.keys()))


def show_supported_models():
    print(model_types)


def get_model_link(model):
    if model in trained_urls:
        return trained_urls[model]
    else:
        None


def is_model_in_factory(model):
    if model in trained_urls:
        return True
    else:
        return False


def is_model_in_model_types(model):
    if model in model_types:
        return True
    else:
        return False


def get_model_type(model):
    for x in model_types:
        if x in model:
            return x
    return None


def is_model_type_in_model_path(model):
    if get_model_type(model) is None:
        return False
    else:
        return True

