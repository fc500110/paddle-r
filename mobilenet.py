#!/usr/bin/env python

#from typing import Callable
#from typing import Dict
from typing import List
from typing import Tuple

import functools
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import AnalysisPredictor
from paddle.fluid.core import create_paddle_predictor


def main():
    config: AnalysisConfig = set_config()
    predictor: AnalysisPredictor = create_paddle_predictor(config)

    data, result = parse_data()
    input_names: List[str] = predictor.get_input_names()
    #print(input_names[0])
    #return
    #input_tensor = predictor.get_input_tensor(input_names[0])
    input_tensor = predictor.get_input_tensor("image")
    shape: Tuple[int] = (1, 3, 300, 300)
    #input_data: np.array = data[:-4].reshape(shape).astype(np.float32)
    input_data: np.array = data[:-4].astype(np.float32).reshape(shape)
    input_tensor.copy_from_cpu(input_data)
    #x: np.array = input_tensor.copy_to_cpu()

    predictor.zero_copy_run()

    output_names: str = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data: np.array = output_tensor.copy_to_cpu()
    print(output_names)
    print(output_tensor.shape())
    print(output_tensor.type())
    print(output_data.shape)
    print(output_data.reshape(-1)[:10])
    print(result[:10])
    print(np.argmax(output_data))


def set_config() -> AnalysisConfig:
    config = AnalysisConfig("")
    config.set_model("data/model/__model__", "data/model/__params__")
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_profile()
    config.disable_glog_info()
    #config.delete_pass("conv_bn_fuse_pass")
    #config.pass_builder().delete_pass("conv_bn_fuse_pass")

    return config


def parse_data() -> Tuple[np.array, np.array]:
    with open('data/data.txt', 'r') as fr:
        data = np.array([float(_) for _ in fr.read().split()])

    with open('data/result.txt', 'r') as fr:
        result = np.array([float(_) for _ in fr.read().split()])

    return (data, result)


if __name__ == "__main__":
    main()
