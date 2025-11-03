# MIT License
# 
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

import psutil
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras


class MemoryMonitorCallback(keras.callbacks.Callback):
    """监控训练过程中内存使用的回调"""

    def __init__(self, log_freq=10, monitor_gpu=True):
        super().__init__()
        self.log_freq = log_freq
        self.monitor_gpu = monitor_gpu
        self.process = psutil.Process(os.getpid())
        self.memory_history = []
        self.epoch_memory_peak = []

        # 检查GPU是否可用
        self.gpu_available = tf.config.list_physical_devices('GPU')
        if self.gpu_available and monitor_gpu:
            logging.info("GPU内存监控已启用")

    def on_epoch_begin(self, epoch, logs=None):
        # 清空TensorFlow的GPU缓存以获得准确测量
        if self.gpu_available and self.monitor_gpu:
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()

        # 记录epoch开始时的内存
        self.epoch_start_mem = self._get_memory_usage()

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_freq == 0:
            memory_usage = self._get_memory_usage()
            self.memory_history.append({
                'epoch': self.params['epochs'],
                'batch': batch,
                'memory_mb': memory_usage['total_mb'],
                'gpu_memory_mb': memory_usage.get('gpu_mb', 0)
            })

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_mem = self._get_memory_usage()
        peak_memory = max(
            [m['memory_mb'] for m in self.memory_history[-self.params['steps']:]] + [epoch_end_mem['total_mb']])

        self.epoch_memory_peak.append({
            'epoch': epoch,
            'peak_memory_mb': peak_memory,
            'gpu_peak_memory_mb': epoch_end_mem.get('gpu_peak_mb', 0),
            'avg_memory_mb': np.mean([m['memory_mb'] for m in self.memory_history[-self.params['steps']:]])
        })

        logging.info(f"Epoch {epoch}: Peak Memory = {peak_memory:.2f} MB, "
                     f"GPU Memory = {epoch_end_mem.get('gpu_mb', 0):.2f} MB")

    def _get_memory_usage(self):
        """获取当前内存使用情况"""
        memory_info = {}

        # 获取CPU内存
        mem_info = self.process.memory_info()
        memory_info['rss_mb'] = mem_info.rss / 1024 ** 2  # 常驻内存
        memory_info['vms_mb'] = mem_info.vms / 1024 ** 2  # 虚拟内存
        memory_info['total_mb'] = memory_info['rss_mb']

        # 获取GPU内存（如果可用）
        if self.gpu_available and self.monitor_gpu:
            try:
                gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
                memory_info['gpu_mb'] = gpu_stats['current'] / 1024 ** 2
                memory_info['gpu_peak_mb'] = gpu_stats['peak'] / 1024 ** 2
            except:
                # 备用方法：使用nvidia-smi（如果安装）
                memory_info['gpu_mb'] = self._get_gpu_memory_nvidia_smi()

        return memory_info

    def _get_gpu_memory_nvidia_smi(self):
        """使用nvidia-smi获取GPU内存（备用方法）"""
        try:
            result = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
            if result:
                return float(result.strip().split('\n')[0])
        except:
            pass
        return 0

    def get_memory_summary(self):
        """返回内存使用摘要"""
        if not self.memory_history:
            return None

        peak_memory = max([m['memory_mb'] for m in self.memory_history])
        avg_memory = np.mean([m['memory_mb'] for m in self.memory_history])

        return {
            'peak_memory_mb': peak_memory,
            'average_memory_mb': avg_memory,
            'memory_history': self.memory_history,
            'epoch_peaks': self.epoch_memory_peak
        }