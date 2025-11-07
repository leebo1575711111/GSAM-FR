# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# Copyright (c) 2025 Baofeng Liao
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import psutil
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from .model_utils import cosine_annealing
import time


class CaptureWeightsCallback(tf.keras.callbacks.Callback):
    """
    Custom TensorFlow callback for capturing and logging model weights during training, with a focus on attention weights.

    This callback is designed to monitor the evolution of model weights, particularly attention weights, across training epochs.
    It facilitates the analysis of training dynamics and model behavior by storing weight snapshots at specified intervals.

    Attributes:
        model (tf.keras.Model): Instance of the TensorFlow model being trained. The model should have a method
                                `get_last_attention_weights()` that this callback can invoke to obtain attention weights.
        attention_weights_history (list): Accumulates the attention weights captured at the end of specified epochs. 
                                          This history facilitates post-training analysis of weight adjustments.

    Methods:
        on_epoch_end(epoch, logs=None): Overrides the base class method to capture attention weights at the end of each epoch.
                                        Weights are captured based on specified criteria, e.g., every 5 epochs.
        get_attention_weights_history(): Provides access to the accumulated history of attention weights captured during training.
    """
    
    def __init__(self, model):
        """
        Initializes the callback with a specific model to monitor its attention weights during training.

        Parameters:
            model (tf.keras.Model): The model whose attention weights are to be monitored and captured.
        """
        super().__init__()
        self.model = model
        self.penultimate_weights = None
        self.attention_weights_history = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch during training to capture and store attention weights if the current
        epoch satisfies the capture criteria (e.g., every 5 epochs).

        Parameters:
            epoch (int): The current epoch number.
            logs (dict): Currently unused. Contains logs from the training epoch.
        """
        if epoch % 5 == 0:  # Perform analysis every 5 epochs
            # Retrieve attention weights from the model
            last_attention_weights = self.model.get_last_attention_weights()
            if last_attention_weights is not None:
                self.attention_weights_history.append(last_attention_weights)
    
    def get_attention_weights_history(self):
        """
        Returns the history of attention weights captured during training.

        Returns:
            A list of attention weights captured at specified intervals during training.
        """
        return self.attention_weights_history




def setup_callbacks(args, checkpoint_path, model):
    """
    Sets up and returns TensorFlow callbacks for use during model training. These callbacks include early stopping,
    learning rate scheduling, model checkpointing, and a custom callback for capturing model weights.

    This function is tailored to support flexible training configurations, allowing for dynamic adjustment of training
    behavior based on model performance and training progress.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments containing training configurations such as patience for early stopping,
                                   total training epochs, and initial learning rate.
        checkpoint_path (str): File path where model checkpoints will be saved. The best model according to validation loss is checkpointed.
        model (tf.keras.Model): The TensorFlow model being trained. Required for initializing the `CaptureWeightsCallback`.
        model_name (str): Name of the model being trained. This can be used to adjust callback behavior for different models.

    Returns:
        tuple: A tuple containing:
               - A list of TensorFlow callbacks configured for the training session.
               - An instance of `CaptureWeightsCallback`, which can be used post-training to access captured weights.

    Raises:
        Exception: If an error occurs in the setup of callbacks, an exception is logged and raised to prevent silent training failures.

    Example:
        >>> callbacks, capture_weights_callback = setup_callbacks(args, './model_checkpoints', model, 'my_model')
        This example demonstrates how to invoke `setup_callbacks` to obtain configured callbacks for training, including a custom
        weight capture callback for post-training analysis.
    """
    try:
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',  
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            verbose=1,
        )

        lr_schedule_callback = LearningRateScheduler(
            lambda epoch: cosine_annealing(epoch, 5, args.learning_rate, 1e-6),
            verbose=1,
        )

        capture_weights_callback = CaptureWeightsCallback(model)

        callbacks = [checkpoint_callback, lr_schedule_callback, capture_weights_callback, early_stop_callback]
        
        return callbacks, capture_weights_callback
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise


class AttentionMonitor(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_attention_support()

    def _check_attention_support(self):
        """检查模型是否支持注意力机制"""
        self.supports_attention = (
                hasattr(self.model, 'use_attention') and
                self.model.use_attention and
                hasattr(self.model, 'get_last_attention_weights')
        )
        if not self.supports_attention:
            print("Info: Attention monitoring disabled - model not configured for attention mechanisms")

    def on_epoch_end(self, epoch, logs=None):
        if not self.supports_attention:
            return

        try:
            weights = self.model.get_last_attention_weights()
            if weights is not None:
                self._log_attention(weights, epoch)
        except Exception as e:
            print(f"Warning: Failed to log attention weights - {str(e)}")
            self.supports_attention = False

    def _log_attention(self, weights, epoch):
        """实际记录注意力权重的逻辑"""
        # 示例：记录权重统计信息
        avg_weight = tf.reduce_mean(weights).numpy()
        logs = {'attention/mean_weight': avg_weight}
        # 可以添加TensorBoard记录或其他自定义逻辑
        print(f"Epoch {epoch}: Mean attention weight = {avg_weight:.4f}")


# class MemoryMonitorCallback(keras.callbacks.Callback):
#     """监控训练过程中内存使用和训练时间的回调"""
#
#     def __init__(self, log_freq=10, monitor_gpu=True):
#         super().__init__()
#         self.log_freq = log_freq
#         self.monitor_gpu = monitor_gpu
#         self.process = psutil.Process(os.getpid())
#         self.memory_history = []
#         self.epoch_memory_peak = []
#         self.batch_times = []  # 新增：存储每个batch的时间
#         self.epoch_times = []  # 新增：存储每个epoch的时间
#         self.epoch_start_time = None  # 新增：记录epoch开始时间
#         self.batch_start_time = None  # 新增：记录batch开始时间
#
#         # 检查GPU是否可用
#         self.gpu_available = tf.config.list_physical_devices('GPU')
#         if self.gpu_available and monitor_gpu:
#             logging.info("GPU内存监控已启用")
#
#     def on_epoch_begin(self, epoch, logs=None):
#         # 清空TensorFlow的GPU缓存以获得准确测量
#         if self.gpu_available and self.monitor_gpu:
#             tf.keras.backend.clear_session()
#             tf.compat.v1.reset_default_graph()
#
#         # 记录epoch开始时的内存
#         self.epoch_start_mem = self._get_memory_usage()
#         self.epoch_start_time = time.time()  # 新增：记录epoch开始时间
#
#     def on_train_batch_begin(self, batch, logs=None):
#         self.batch_start_time = time.time()  # 新增：记录batch开始时间
#
#     def on_train_batch_end(self, batch, logs=None):
#         # 新增：计算batch耗时
#         batch_time = (time.time() - self.batch_start_time) * 1000  # 转换为毫秒
#         self.batch_times.append(batch_time)
#
#         if batch % self.log_freq == 0:
#             memory_usage = self._get_memory_usage()
#             self.memory_history.append({
#                 'epoch': self.params['epochs'],
#                 'batch': batch,
#                 'memory_mb': memory_usage['total_mb'],
#                 'gpu_memory_mb': memory_usage.get('gpu_mb', 0),
#                 'batch_time_ms': batch_time  # 新增：记录batch时间
#             })
#
#     def on_epoch_end(self, epoch, logs=None):
#         # 新增：计算epoch耗时
#         epoch_time = time.time() - self.epoch_start_time
#         self.epoch_times.append(epoch_time)
#
#         epoch_end_mem = self._get_memory_usage()
#         peak_memory = max(
#             [m['memory_mb'] for m in self.memory_history[-self.params['steps']:]] + [epoch_end_mem['total_mb']])
#
#         self.epoch_memory_peak.append({
#             'epoch': epoch,
#             'peak_memory_mb': peak_memory,
#             'gpu_peak_memory_mb': epoch_end_mem.get('gpu_peak_mb', 0),
#             'avg_memory_mb': np.mean([m['memory_mb'] for m in self.memory_history[-self.params['steps']:]]),
#             'epoch_time_s': epoch_time,  # 新增：记录epoch时间
#             'avg_batch_time_ms': np.mean(self.batch_times[-self.params['steps']:]) if self.batch_times else 0
#             # 新增：记录平均batch时间
#         })
#
#         logging.info(f"Epoch {epoch}: Peak Memory = {peak_memory:.2f} MB, "
#                      f"GPU Memory = {epoch_end_mem.get('gpu_mb', 0):.2f} MB, "
#                      f"Epoch Time = {epoch_time:.2f} s, "
#                      f"Avg Batch Time = {np.mean(self.batch_times[-self.params['steps']:]):.2f} ms")
#
#     def _get_memory_usage(self):
#         """获取当前内存使用情况"""
#         memory_info = {}
#
#         # 获取CPU内存
#         mem_info = self.process.memory_info()
#         memory_info['rss_mb'] = mem_info.rss / 1024 ** 2  # 常驻内存
#         memory_info['vms_mb'] = mem_info.vms / 1024 ** 2  # 虚拟内存
#         memory_info['total_mb'] = memory_info['rss_mb']
#
#         # 获取GPU内存（如果可用）
#         if self.gpu_available and self.monitor_gpu:
#             try:
#                 gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
#                 memory_info['gpu_mb'] = gpu_stats['current'] / 1024 ** 2
#                 memory_info['gpu_peak_mb'] = gpu_stats['peak'] / 1024 ** 2
#             except:
#                 # 备用方法：使用nvidia-smi（如果安装）
#                 memory_info['gpu_mb'] = self._get_gpu_memory_nvidia_smi()
#
#         return memory_info
#
#     def _get_gpu_memory_nvidia_smi(self):
#         """使用nvidia-smi获取GPU内存（备用方法）"""
#         try:
#             result = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
#             if result:
#                 return float(result.strip().split('\n')[0])
#         except:
#             pass
#         return 0
#
#     def get_memory_summary(self):
#         """返回内存使用和训练时间摘要"""
#         if not self.memory_history:
#             return None
#
#         peak_memory = max([m['memory_mb'] for m in self.memory_history])
#         avg_memory = np.mean([m['memory_mb'] for m in self.memory_history])
#
#         # 新增：计算平均每迭代时间
#         avg_time_per_iter = np.mean(self.batch_times) if self.batch_times else 0
#         total_training_time = sum(self.epoch_times) if self.epoch_times else 0
#
#         return {
#             'peak_memory_mb': peak_memory,
#             'average_memory_mb': avg_memory,
#             'memory_history': self.memory_history,
#             'epoch_peaks': self.epoch_memory_peak,
#             'avg_time_per_iter_ms': avg_time_per_iter,  # 新增：平均每迭代时间
#             'total_training_time_s': total_training_time  # 新增：总训练时间
#         }