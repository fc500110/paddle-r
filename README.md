# paddle-r
R 调用 paddlepaddle 预测

## install
首先确保已安装Python，路径为`/opt/python3.7`

使用Python安装Paddle
``` bash
/opt/python3.7/bin/python3.7 -m pip install paddlepaddle # CPU
/opt/python3.7/bin/python3.7 -m pip install paddlepaddle-gpu # GPU
```

安装r运行paddle预测所需要的库
``` r
install.packages("reticulate") # 调用Paddle
install.packages("RcppCNPy") # 在R中使用numpy.ndarray
```

## 使用示例
首先在r中引入paddle预测库

``` r
library(reticulate) # call Python library
library(RcppCNPy) # numpy support

use_python("/opt/python3.7/bin/python")

paddle <- import("paddle.fluid.core")
```

创建一个AnalysisConfig，这是预测引擎的配置项

```r
config <- paddle$AnalysisConfig("")
config$switch_use_feed_fetch_ops(FALSE)
config$switch_specify_input_names(TRUE)
```

设置模型路径

``` r
config$set_model("data/model/__model__", "data/model/__params__")
```

其他一些配置选项及说明如下
``` r
config$enable_profile() # 打开预测profile
config$enable_use_gpu(gpu_memory_mb, gpu_id) # 开启GPU预测
config$disable_gpu() # 禁用GPU
config$gpu_device_id() # 返回使用的GPU ID
config$switch_ir_optim(TRUE) # 开启IR优化(默认开启)
config$enable_tensorrt_engine(workspace_size,
                              max_batch_size,
                              min_subgraph_size,
                              paddle$AnalysisConfig$Precision$FLOAT32,
                              use_static,
                              use_calib_mode
                              ) # 开启TensorRT
config$enable_mkldnn() # 开启MKLDNN
config$disable_glog_info() # 禁用预测中的glog日志
config$delete_pass(pass_name) # 预测的时候删除指定的pass

```

创建预测引擎
``` r
predictor <- paddle$create_paddle_predictor(config)
```

获取输入tensor(此处假设只有一个输入)
``` r
input_names <- predictor$get_input_names()
input_tensor <- predictor$get_input_tensor(input_names[1])
```

设置输入tensor中的数据，注意此处需要使用numpy.ndarray传入
``` r
input_shape <- as.integer(c(1, 3, 300, 300)) # shape需要转为int类型
input_data <- np_array(data, dtype="float32")$reshape(input_shape) # 将data转为numpy.ndarray
input_tensor$copy_from_cpu(input_data)
```

运行预测引擎
``` r
predictor$zero_copy_run()
```

获取输出tensor(此处假设只有一个输出)
``` r
output_names <- predictor$get_output_names()
output_tensor <- predictor$get_output_tensor(output_names[1])
```

获取输出tensor中的数据，注意需要转为numpy.ndarray
``` r
output_data <- output_tensor$copy_to_cpu()
output_data <- np_array(output_data)$reshape(as.integer(-1))
```

点击查看完整的[r预测demo](./mobilenet.r)及对应的[python预测demo](./mobilenet.py)
