"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/misc_utils.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import os
import pickle
import sys
from typing import List


def batch_iterator(iterator, batch_size):
    if len(iterator) <= batch_size:
        return [iterator]

    output = []
    i = 0

    while (len(iterator) - i) > batch_size:
        output.append(iterator[i : i + batch_size])
        i += batch_size

    output.append(iterator[i:])
    return output


def make_empty_dirs(log_dir):
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, "coordinates"))
    os.makedirs(os.path.join(log_dir, "summary"))


def accelerator_print(string, accelerator):
    if accelerator.is_main_process:
        print(string)


def flatten_dict(dictionary: dict, level: List = []) -> dict:
    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten_dict(val, level + [str(key)]))
        else:
            tmp_dict[".".join(level + [str(key)])] = val
    return tmp_dict


def unflatten_dict(dictionary: dict, to_int: bool = True) -> dict:
    result_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        if to_int:
            parts = [p if not p.isnumeric() else int(p) for p in parts]
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def setup_logger(log_path: str):
    from loguru import logger

    try:
        logger.remove(handler_id=0)  # Remove pre-configured sink to sys.stderror
    except ValueError:
        pass

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        enqueue=True,
        diagnose=True,
    )
    return logger


def pickle_dump(obj: object, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file_path: str) -> object:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def assertion_check(if_statement: bool, failure_message: str = ""):
    assert if_statement, failure_message


class FileHandle:
    def __init__(self, print_fn):
        self.print_fn = print_fn

    def write(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)

    def flush(self, *args, **kwargs):
        pass


def filter_useless_warnings():
    import warnings

    warnings.filterwarnings("ignore", ".*nn\.functional\.upsample is deprecated.*")
    warnings.filterwarnings("ignore", ".*none of the inputs have requires_grad.*")
    warnings.filterwarnings("ignore", ".*with given element none.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true\_divide.*")


def get_esm_model(esm_model_name):
    import esm

    return getattr(esm.pretrained, esm_model_name)()


class Args(object):  # Generic container for arguments
    def __init__(self, kwarg_dict):
        for (k, v) in kwarg_dict.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)
