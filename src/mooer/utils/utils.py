import logging
import torch


def parse_key_text(input_text):
    result = {}
    with open(input_text, 'r') as r:
        for line in r.readlines():
            line = line.strip()
            if line == '':
                continue
            line = line.split(maxsplit=1)
            if len(line) != 2:
                continue
            key, text = line
            result[key] = text
    return result


def print_module_size(module, module_name, rank: int = 0, info=None) -> None:
    if rank == 0:
        if info:
            logging.info(info)
        logging.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logging.info(f"--> {module_name} has {total_params / 1e6} Million params\n")


# device register/check/create by default
_device_registry = []

def _enable_cuda() -> bool:
    return torch.cuda.is_available()

def _enable_musa() -> bool:
    try:
        import torch_musa
    except:
        return False
    return torch_musa.is_available()

def _create_cuda_device() -> torch.device:
    return torch.device("cuda")

def _create_musa_device() -> torch.device:
    return torch.device("musa")

def _register_device(priority, checker, creator):
    device_elem = (priority, checker, creator)
    _device_registry.append(device_elem)
    _device_registry.sort()

_register_device(10, _enable_musa, _create_musa_device)
_register_device(20, _enable_cuda, _create_cuda_device)

def get_device() -> torch.device:
    for (_, checker, creator) in _device_registry:
        if checker():
            return creator()
    return torch.device("cpu")
