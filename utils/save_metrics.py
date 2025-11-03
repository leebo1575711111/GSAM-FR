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

import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import traceback  # 用于显示完整错误信息
from .data_utils import extract_data, TSFDataLoader
from .model_utils import power_iteration


def save_main_results(history, test_result, args, elapsed_training_time, current_directory):
    """
    Saves the main training and testing results to CSV files.

    Parameters:
        history: Training history object from Keras model training.
        test_result: Results from model evaluation on test data.
        args: Namespace object from argparse containing script arguments.
        elapsed_training_time: Float, total training time elapsed.
        current_directory: String, the path of the current working directory.
    """
    logger = logging.getLogger(__name__)

    # Prepare main results data
    data = {
        'data': [args.data],
        'model': [args.model],
        'batch_size': [args.batch_size],
        'seq_len': [args.seq_len],
        'pred_len': [args.pred_len],
        'lr': [args.learning_rate],
        'lambda_reg': [args.lambda_reg],
        'mse': [test_result[0]],
        'mae': [test_result[1]],
        'val_mse': [min(history.history['val_loss'])],
        'val_mae': [history.history['val_mae'][np.argmin(history.history['val_loss'])]],
        'train_mse': [history.history['loss'][np.argmin(history.history['val_loss'])]],
        'train_mae': [history.history['mae'][np.argmin(history.history['val_loss'])]],
        'training_time': elapsed_training_time,
        'rho': args.rho if args.opt_strategy and (args.model not in ['transformer_random']) else 0.0,
        'alpha': args.alpha if args.opt_strategy == 2 and (args.model not in ['transformer_random']) else 0.0,
    }
    df = pd.DataFrame(data)

    # Ensure the results directory exists
    results_dir = os.path.join(current_directory, 'results')
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    # 根据优化策略生成文件名
    if args.opt_strategy == 0:
        result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{''}.csv")
    elif args.opt_strategy == 1:
        result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{'_sam'}.csv")
    elif args.opt_strategy == 2:
        result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{'_gsam-reg'}.csv")
    # Save to CSV, appending if file exists, else create a new file
    df.to_csv(result_path, mode='a' if os.path.exists(result_path) else 'w', index=False,
              header=not os.path.exists(result_path))
    logger.info(f"Main results saved to {result_path}")



# def save_main_results(history, test_result, args, elapsed_training_time, current_directory, memory_summary=None):
#     """
#     Saves the main training and testing results to CSV files, including memory usage and training time information.
#
#     Parameters:
#         history: Training history object from Keras model training.
#         test_result: Results from model evaluation on test data.
#         args: Namespace object from argparse containing script arguments.
#         elapsed_training_time: Float, total training time elapsed.
#         current_directory: String, the path of the current working directory.
#         memory_summary: Dict containing memory usage and training time statistics (optional).
#     """
#     logger = logging.getLogger(__name__)
#
#     # Prepare main results data
#     data = {
#         'data': [args.data],
#         'model': [args.model],
#         'batch_size': [args.batch_size],
#         'seq_len': [args.seq_len],
#         'pred_len': [args.pred_len],
#         'lambda_reg': [args.lambda_reg],
#         'lr': [args.learning_rate],
#         'mse': [test_result[0]],
#         'mae': [test_result[1]],
#         'val_mse': [min(history.history['val_loss'])],
#         'val_mae': [history.history['val_mae'][np.argmin(history.history['val_loss'])]],
#         'train_mse': [history.history['loss'][np.argmin(history.history['val_loss'])]],
#         'train_mae': [history.history['mae'][np.argmin(history.history['val_loss'])]],
#         'training_time': [elapsed_training_time],
#         'rho': [args.rho if args.opt_strategy and (args.model not in ['transformer_random']) else 0.0],
#         'alpha': [args.alpha if args.opt_strategy == 2 and (args.model not in ['transformer_random']) else 0.0],
#         # 添加内存使用信息
#         'peak_memory_mb': [memory_summary['peak_memory_mb'] if memory_summary else 0],
#         'avg_memory_mb': [memory_summary['average_memory_mb'] if memory_summary else 0],
#         'final_memory_mb': [memory_summary['memory_history'][-1]['memory_mb'] if memory_summary and memory_summary[
#             'memory_history'] else 0],
#         'gpu_peak_memory_mb': [
#             memory_summary['epoch_peaks'][-1]['gpu_peak_memory_mb'] if memory_summary and memory_summary[
#                 'epoch_peaks'] else 0],
#         # 添加训练时间信息
#         'avg_time_per_iter_ms': [memory_summary['avg_time_per_iter_ms'] if memory_summary else 0],
#         'total_training_time_s': [memory_summary['total_training_time_s'] if memory_summary else elapsed_training_time],
#         'use_attention': [args.use_attention],
#         'mixed_precision': [args.mixed_precision],
#         'memory_growth': [args.memory_growth],
#         'gradient_accumulation_steps': [args.gradient_accumulation_steps],
#         'n_block': [args.n_block],
#         'ff_dim': [args.ff_dim],
#         'd_model': [args.d_model],
#         'num_heads': [args.num_heads],
#         'dropout': [args.dropout],
#         'norm_type': [args.norm_type],
#         'activation': [args.activation],
#         'seed': [args.seed]
#     }
#
#     # 如果有GPU内存信息，也添加进去
#     if memory_summary and any('gpu_mb' in m for m in memory_summary.get('memory_history', [])):
#         gpu_memory_values = [m.get('gpu_mb', 0) for m in memory_summary['memory_history'] if 'gpu_mb' in m]
#         if gpu_memory_values:
#             data['gpu_avg_memory_mb'] = [np.mean(gpu_memory_values)]
#             data['gpu_peak_memory_mb'] = [max(gpu_memory_values)]
#
#     # 如果有epoch时间信息，添加每个epoch的平均时间
#     if memory_summary and memory_summary.get('epoch_peaks'):
#         epoch_times = [epoch.get('epoch_time_s', 0) for epoch in memory_summary['epoch_peaks']]
#         if epoch_times:
#             data['avg_epoch_time_s'] = [np.mean(epoch_times)]
#             data['min_epoch_time_s'] = [min(epoch_times)]
#             data['max_epoch_time_s'] = [max(epoch_times)]
#
#     df = pd.DataFrame(data)
#
#     # Ensure the results directory exists
#     results_dir = os.path.join(current_directory, 'results')
#     os.makedirs(results_dir, exist_ok=True)
#
#     # 根据优化策略生成文件名
#     if args.opt_strategy == 0:
#         result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{''}.csv")
#     elif args.opt_strategy == 1:
#         result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{'_sam'}.csv")
#     elif args.opt_strategy == 2:
#         result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{'_gsam-reg'}.csv")
#
#     # 检查文件是否存在以确定是否需要写入表头
#     file_exists = os.path.exists(result_path)
#
#     # 保存到CSV
#     df.to_csv(result_path, mode='a' if file_exists else 'w',
#               index=False, header=not file_exists)
#
#     logger.info(f"Main results saved to {result_path}")
#
#     # 额外保存详细的内存历史记录（可选）
#     if memory_summary and memory_summary.get('memory_history'):
#         memory_df = pd.DataFrame(memory_summary['memory_history'])
#         memory_history_path = os.path.join(results_dir, f"memory_history_{args.model}_{args.data}_{args.seed}.csv")
#         memory_df.to_csv(memory_history_path, index=False)
#         logger.info(f"Detailed memory history saved to {memory_history_path}")
#
#     if memory_summary and memory_summary.get('epoch_peaks'):
#         epoch_memory_df = pd.DataFrame(memory_summary['epoch_peaks'])
#         epoch_memory_path = os.path.join(results_dir, f"epoch_memory_{args.model}_{args.data}_{args.seed}.csv")
#         epoch_memory_df.to_csv(epoch_memory_path, index=False)
#         logger.info(f"Epoch memory peaks saved to {epoch_memory_path}")
#
#     # 额外保存训练时间历史记录（可选）
#     if memory_summary and memory_summary.get('epoch_peaks'):
#         time_df = pd.DataFrame({
#             'epoch': [e.get('epoch', i) for i, e in enumerate(memory_summary['epoch_peaks'])],
#             'epoch_time_s': [e.get('epoch_time_s', 0) for e in memory_summary['epoch_peaks']],
#             'avg_batch_time_ms': [e.get('avg_batch_time_ms', 0) for e in memory_summary['epoch_peaks']]
#         })
#         time_history_path = os.path.join(results_dir, f"training_time_{args.model}_{args.data}_{args.seed}.csv")
#         time_df.to_csv(time_history_path, index=False)
#         logger.info(f"Training time history saved to {time_history_path}")



def save_training_history(history, args, current_directory):
    """
    Saves the training history, including epoch numbers, model name, prediction horizon,
    dataset used, whether SAM (Sharpness-Aware Minimization) was used, training loss, and validation loss,
    to a CSV file. The file is named based on the model, dataset, and whether SAM was employed,
    ensuring unique filenames for different training configurations.

    The function appends the new training history to the existing file if it already exists,
    allowing for cumulative recording of training sessions without overwriting previous data.

    Parameters:
        history (tf.keras.callbacks.History): The training history object returned by the fit method
                                              of a Keras model. It contains loss metrics recorded over
                                              each epoch of training.
        args (argparse.Namespace): A namespace object containing command line arguments. Expected
                                   to include 'model' (the model name), 'pred_len' (prediction horizon),
                                   'data' (the dataset name), and 'opt_strategy' (a boolean indicating if SAM
                                   was used during training).
        current_directory (str): The directory path where the training history CSV file will be saved.
                                 The function constructs a subdirectory named 'results' within this path
                                 to store the file.

    Outputs:
        A CSV file named 'history_[model name]_[dataset]{_sam if SAM was used}.csv' in the
        '[current_directory]/results' directory. The file contains columns for epoch number, model name,
        prediction horizon, dataset, SAM usage, training loss, and validation loss. If the file already exists,
        the function appends the new data to it, preserving any existing data.

    Example Filename:
        'results/history_modelname_datasetname_sam.csv' if SAM was used,
        'results/history_modelname_datasetname.csv' if SAM was not used.
    """
    logger = logging.getLogger(__name__)
    epochs = range(1, len(history.history['loss']) + 1)
    df = pd.DataFrame({
        'Epoch': epochs,
        'Model Name': [args.model] * len(epochs),
        'Horizon': args.pred_len,
        'Dataset': args.data,
        'opt_strategy': args.opt_strategy,
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })
    if args.opt_strategy == 0:
        history_csv_path = f"{current_directory}/results/history_{args.model}_{args.data}.csv"
    if args.opt_strategy == 1:
        history_csv_path = f"{current_directory}/results/history_{args.model}_{args.data}{'_sam'}.csv"
    if args.opt_strategy == 2:
        history_csv_path = f"{current_directory}/results/history_{args.model}_{args.data}{'_gsam-reg'}.csv"
    df.to_csv(history_csv_path, mode='a' if os.path.exists(history_csv_path) else 'w', index=False, header=not os.path.exists(history_csv_path))
    logger.info(f"Training history saved to {history_csv_path}")

def save_additional_metrics(model, args, train_data, current_directory, capture_weights_callback):
    """
    Saves additional metrics including attention weights and sharpness, if the --add_results
    command line argument is specified. Attention weights are calculated on a batch of 32 sequences
    every 5 epochs by default, which is the capture frequency set in the CaptureWeightsCallback.
    The sharpness, measured as the largest eigenvalue of the Hessian matrix, is computed using
    power iteration at the end of the training process.

    Parameters:
        model (tf.keras.Model): The trained model.
        args (argparse.Namespace): Command line arguments specified by the user. Must include
                                   'add_results' to indicate if additional results should be saved.
        train_data (tf.data.Dataset): The training dataset used to compute sharpness.
        current_directory (str): The current working directory where results are saved.
        capture_weights_callback (CaptureWeightsCallback): The callback instance used during training
                                                           to capture attention weights.

    Note:
        - The attention weights are saved as a NumPy array (.npy) file.
        - The sharpness (largest eigenvalue of the Hessian matrix) is saved in a CSV file.
    """
    logger = logging.getLogger(__name__)
    # Check if additional results saving is requested
    if not args.add_results:
        return  # Do nothing if add_results is not True

    # Example of saving attention weights
    if hasattr(model, 'all_attention_weights'):
        attention_weights = capture_weights_callback.get_attention_weights_history()
        attention_weights_path = os.path.join(current_directory, f"results/attention_weights_{args.model}_{args.data}.npy")
        np.save(attention_weights_path, attention_weights)
        logger.info(f"Attention weights saved at {attention_weights_path}")

    # Calculate and save the eigenvalues of the Hessian matrix (sharpness)
    X_input, X_target = extract_data(train_data)
    largest_eigenvalue, delta = power_iteration(model, X_input, X_target)
    eigenvalues_data = {
        'Largest Eigenvalue': [largest_eigenvalue],
        'Delta': [delta]
    }
    eigenvalues_df = pd.DataFrame(eigenvalues_data)
    eigenvalues_path = os.path.join(current_directory, f"results/eigenvalues_{args.model}_{args.data}.csv")
    eigenvalues_df.to_csv(eigenvalues_path, index=False)
    logger.info(f"Eigenvalues (sharpness) recorded at {eigenvalues_path}")

def save_predictions(model, test_data, args, current_directory):
    """保存模型预测结果到NPZ文件"""
    # 创建结果目录
    results_dir = os.path.join(current_directory, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 提取测试数据
    X_input, y_true = [], []
    for batch in test_data:
        x, y = batch
        X_input.append(x.numpy() if hasattr(x, 'numpy') else x)
        y_true.append(y.numpy() if hasattr(y, 'numpy') else y)

    X_input = np.vstack(X_input)
    y_true = np.vstack(y_true)

    logging.info(f"数据提取成功: X_input形状={X_input.shape}, y_true形状={y_true.shape}")

    # 清理GPU内存
    tf.keras.backend.clear_session()

    # 使用更小的批次大小进行预测
    pred_batch_size = max(1, args.batch_size // 4)

    # 分批预测以避免内存不足
    y_pred_parts = []
    n_samples = X_input.shape[0]

    logging.info("开始分批模型预测...")

    # 首先进行一次预测以确定输出类型和形状
    test_batch = X_input[:min(2, n_samples)]
    with tf.device('/CPU:0'):
        test_pred = model.predict(test_batch, verbose=0)

    # 检查模型是否返回多个输出
    if isinstance(test_pred, list):
        logging.info(f"模型返回多个输出，数量: {len(test_pred)}")
        # 假设第一个输出是主要的预测结果
        test_pred = test_pred[0]
        logging.info(f"使用第一个输出作为预测结果，形状: {test_pred.shape}")

    # 确定期望的输出维度（不是形状）
    expected_ndim = test_pred.ndim
    logging.info(f"期望的输出维度数: {expected_ndim}")

    # 统一处理函数：确保所有输出具有相同的维度
    def unify_prediction_shape(pred, expected_ndim):
        """统一预测结果的维度"""
        current_ndim = pred.ndim

        # 如果维度已经一致，直接返回
        if current_ndim == expected_ndim:
            return pred

        # 如果当前是4维但期望是3维，尝试压缩
        if current_ndim == 4 and expected_ndim == 3:
            # 检查第二个维度是否为1（可能是多余的维度）
            if pred.shape[1] == 1:
                return np.squeeze(pred, axis=1)
            else:
                # 如果不是1，取第一个"头"的输出
                logging.warning(f"4D转3D: 从形状{pred.shape}中取第一个头")
                return pred[:, 0, :, :]

        # 如果当前是3维但期望是4维，尝试扩展
        if current_ndim == 3 and expected_ndim == 4:
            logging.warning(f"3D转4D: 为形状{pred.shape}添加维度")
            return np.expand_dims(pred, axis=1)

        # 其他情况记录警告但尝试调整
        logging.warning(f"无法处理的维度转换: {current_ndim}D -> {expected_ndim}D")
        return pred

    for i in range(0, n_samples, pred_batch_size):
        end_idx = min(i + pred_batch_size, n_samples)
        batch_X = X_input[i:end_idx]

        # 使用CPU进行预测以避免GPU内存问题
        with tf.device('/CPU:0'):
            batch_pred = model.predict(batch_X, verbose=0)

        # 处理多输出模型
        if isinstance(batch_pred, list):
            batch_pred = batch_pred[0]
            logging.info(f"使用多输出模型的第一个输出，形状: {batch_pred.shape}")

        # 统一维度
        batch_pred = unify_prediction_shape(batch_pred, expected_ndim)

        # 记录调整后的形状
        if batch_pred.shape[0] != batch_X.shape[0]:
            logging.warning(f"批次{i}预测样本数不匹配: 输入{batch_X.shape[0]}, 输出{batch_pred.shape[0]}")
            # 如果样本数不匹配，尝试截断或跳过
            if batch_pred.shape[0] > batch_X.shape[0]:
                batch_pred = batch_pred[:batch_X.shape[0]]
            else:
                logging.error(f"批次{i}输出样本数少于输入，跳过该批次")
                continue

        y_pred_parts.append(batch_pred)

        # 每完成10%的预测输出一次进度
        progress = (end_idx / n_samples) * 100
        if int(progress) % 10 == 0 and int(progress) > 0:
            logging.info(f"预测进度: {progress:.1f}%")

        # 清理内存
        tf.keras.backend.clear_session()

    # 检查是否有预测结果
    if not y_pred_parts:
        logging.error("没有成功生成任何预测结果")
        return None

    # 检查所有部分的维度是否一致
    ndims = [part.ndim for part in y_pred_parts]
    if len(set(ndims)) > 1:
        logging.warning(f"预测部分维度不一致: {ndims}")
        logging.info("尝试统一所有部分的维度...")

        # 使用最常见的维度作为目标
        from collections import Counter
        dim_counter = Counter(ndims)
        target_ndim = dim_counter.most_common(1)[0][0]
        logging.info(f"目标维度设置为: {target_ndim}D")

        # 统一所有部分的维度
        unified_parts = []
        for i, part in enumerate(y_pred_parts):
            if part.ndim != target_ndim:
                logging.warning(f"统一部分{i}的维度: {part.ndim}D -> {target_ndim}D")
                unified_part = unify_prediction_shape(part, target_ndim)
                unified_parts.append(unified_part)
            else:
                unified_parts.append(part)

        y_pred_parts = unified_parts

    # 合并预测结果
    try:
        # 首先尝试vstack
        y_pred = np.vstack(y_pred_parts)
        logging.info(f"使用vstack成功合并预测结果，形状: {y_pred.shape}")
    except ValueError as e:
        logging.warning(f"vstack失败: {e}，尝试concatenate...")
        try:
            # 尝试concatenate
            y_pred = np.concatenate(y_pred_parts, axis=0)
            logging.info(f"使用concatenate成功合并，形状: {y_pred.shape}")
        except Exception as e2:
            logging.error(f"concatenate也失败: {e2}")
            logging.info("尝试逐个添加...")

            # 最后尝试逐个添加
            y_pred = y_pred_parts[0]
            for i in range(1, len(y_pred_parts)):
                try:
                    y_pred = np.concatenate([y_pred, y_pred_parts[i]], axis=0)
                except Exception as e3:
                    logging.error(f"添加部分{i}失败: {e3}")
                    # 跳过有问题的部分
                    continue

            logging.info(f"最终合并形状: {y_pred.shape}")

    # 确保预测结果与真实值具有相同的样本数量
    if y_pred.shape[0] != y_true.shape[0]:
        logging.warning(f"预测结果样本数 {y_pred.shape[0]} 与真实值样本数 {y_true.shape[0]} 不匹配")
        min_samples = min(y_pred.shape[0], y_true.shape[0])
        y_pred = y_pred[:min_samples]
        y_true = y_true[:min_samples]
        X_input = X_input[:min_samples]
        logging.info(f"已截断为 {min_samples} 个样本")

    # 保存预测结果
    if args.opt_strategy == 0:
        suffix = ''
    elif args.opt_strategy == 1:
        suffix = '_sam'
    else:
        suffix = '_gsam-reg'

    base_name = f"predictions_{args.model}_{args.data}{suffix}"
    npz_path = os.path.join(results_dir, f"{base_name}_{args.seq_len}_{args.pred_len}_{args.lambda_reg}.npz")

    np.savez_compressed(
        npz_path,
        X_input=X_input,
        y_true=y_true,
        y_pred=y_pred,
        model_name=args.model,
        data_name=args.data,
        target_channel=getattr(args, 'target', 0)
    )

    logging.info(f"预测结果已保存到: {npz_path}")

    return npz_path

# def save_predictions(model, test_data, args, current_directory):
#     """保存模型预测结果到NPZ文件"""
#     # 创建结果目录
#     results_dir = os.path.join(current_directory, 'results')
#     os.makedirs(results_dir, exist_ok=True)
#
#     # 提取测试数据
#     X_input, y_true = [], []
#     for batch in test_data:
#         x, y = batch
#         X_input.append(x.numpy() if hasattr(x, 'numpy') else x)
#         y_true.append(y.numpy() if hasattr(y, 'numpy') else y)
#
#     X_input = np.vstack(X_input)
#     y_true = np.vstack(y_true)
#
#     logging.info(f"数据提取成功: X_input形状={X_input.shape}, y_true形状={y_true.shape}")
#
#     # 清理GPU内存
#     tf.keras.backend.clear_session()
#
#     # 使用更小的批次大小进行预测
#     pred_batch_size = max(1, args.batch_size // 4)  # 使用更小的批次大小
#
#     # 分批预测以避免内存不足
#     y_pred_parts = []
#     n_samples = X_input.shape[0]
#
#     logging.info("开始分批模型预测...")
#
#     # 首先进行一次预测以确定输出类型和形状
#     test_batch = X_input[:min(2, n_samples)]
#     with tf.device('/CPU:0'):
#         test_pred = model.predict(test_batch, verbose=0)
#
#     # 检查模型是否返回多个输出
#     if isinstance(test_pred, list):
#         logging.info(f"模型返回多个输出，数量: {len(test_pred)}")
#         # 假设第一个输出是主要的预测结果
#         test_pred = test_pred[0]
#         logging.info(f"使用第一个输出作为预测结果，形状: {test_pred.shape}")
#
#     expected_shape = test_pred.shape[1:]  # 期望的输出形状（去掉批次维度）
#     logging.info(f"期望的输出形状: {expected_shape}")
#
#     for i in range(0, n_samples, pred_batch_size):
#         end_idx = min(i + pred_batch_size, n_samples)
#         batch_X = X_input[i:end_idx]
#
#         # 使用CPU进行预测以避免GPU内存问题
#         with tf.device('/CPU:0'):
#             batch_pred = model.predict(batch_X, verbose=0)
#
#         # 处理多输出模型
#         if isinstance(batch_pred, list):
#             # 假设第一个输出是主要的预测结果
#             batch_pred = batch_pred[0]
#             logging.info(f"使用多输出模型的第一个输出，形状: {batch_pred.shape}")
#
#         # 确保输出形状一致
#         if batch_pred.shape[1:] != expected_shape:
#             logging.warning(
#                 f"批次 {i} 的输出形状 {batch_pred.shape} 与期望形状 {(batch_X.shape[0],) + expected_shape} 不匹配")
#
#             # 尝试调整形状
#             if batch_pred.ndim == 4 and batch_pred.shape[0] == batch_X.shape[0]:
#                 # 如果是4维数组，尝试提取正确的部分
#                 # 假设第一个维度是批次，第二个维度可能是多个输出
#                 # 取第一个输出
#                 if batch_pred.shape[1] > 0:
#                     batch_pred = batch_pred[:, 0, :, :]  # 取第一个输出
#                     logging.info(f"已提取第一个输出，形状: {batch_pred.shape}")
#                 else:
#                     logging.error(f"无法提取输出，跳过该批次")
#                     continue
#             else:
#                 logging.error(f"无法调整形状，跳过该批次")
#                 continue
#
#         y_pred_parts.append(batch_pred)
#
#         # 每完成10%的预测输出一次进度
#         progress = (end_idx / n_samples) * 100
#         if int(progress) % 10 == 0 and int(progress) > 0:
#             logging.info(f"预测进度: {progress:.1f}%")
#
#         # 清理内存
#         tf.keras.backend.clear_session()
#
#     # 检查是否有预测结果
#     if not y_pred_parts:
#         logging.error("没有成功生成任何预测结果")
#         return None
#
#     # 合并预测结果
#     try:
#         y_pred = np.vstack(y_pred_parts)
#         logging.info(f"成功合并预测结果，形状: {y_pred.shape}")
#     except ValueError as e:
#         logging.error(f"合并预测结果时出错: {e}")
#         logging.info("尝试使用concatenate而不是vstack...")
#
#         # 尝试使用concatenate
#         try:
#             y_pred = np.concatenate(y_pred_parts, axis=0)
#             logging.info(f"使用concatenate成功，形状: {y_pred.shape}")
#         except Exception as e2:
#             logging.error(f"使用concatenate也失败: {e2}")
#             return None
#
#     # 确保预测结果与真实值具有相同的样本数量
#     if y_pred.shape[0] != y_true.shape[0]:
#         logging.warning(f"预测结果样本数 {y_pred.shape[0]} 与真实值样本数 {y_true.shape[0]} 不匹配")
#         min_samples = min(y_pred.shape[0], y_true.shape[0])
#         y_pred = y_pred[:min_samples]
#         y_true = y_true[:min_samples]
#         X_input = X_input[:min_samples]
#         logging.info(f"已截断为 {min_samples} 个样本")
#
#     # 保存预测结果
#     if args.opt_strategy == 0:
#         suffix = ''
#     elif args.opt_strategy == 1:
#         suffix = '_sam'
#     else:
#         suffix = '_gsam-reg'
#
#     base_name = f"predictions_{args.model}_{args.data}{suffix}"
#     npz_path = os.path.join(results_dir, f"{base_name}_{args.seq_len}_{args.pred_len}_{args.lambda_reg}.npz")
#
#     np.savez_compressed(
#         npz_path,
#         X_input=X_input,
#         y_true=y_true,
#         y_pred=y_pred,
#         model_name=args.model,
#         data_name=args.data,
#         target_channel=getattr(args, 'target', 0)
#     )
#
#     logging.info(f"预测结果已保存到: {npz_path}")
#
#     return npz_path

# def save_predictions(model, test_data, args, current_directory):
#     logger = logging.getLogger(__name__)
#
#     try:
#         logger.info("开始提取测试数据...")
#         X_input, y_true = extract_data(test_data)
#         logger.info(f"数据提取成功: X_input形状={X_input.shape}, y_true形状={y_true.shape}")
#
#         # 强制使用 CPU 进行预测
#         with tf.device('/CPU:0'):
#             logger.info("开始模型预测（强制使用CPU）...")
#             y_pred = model.predict(X_input, batch_size=8, verbose=0)
#
#         # 处理 RaggedTensor
#         if hasattr(y_pred, 'to_tensor'):
#             logger.info("检测到 RaggedTensor，转换为常规张量")
#             y_pred = y_pred.to_tensor()
#
#         # 转换为numpy数组
#         y_pred = np.array(y_pred)
#         y_true = np.array(y_true)
#         logger.info(f"最终y_pred形状: {y_pred.shape}")
#         logger.info(f"最终y_true形状: {y_true.shape}")
#
#         # 检查数据有效性
#         if y_pred.size == 0:
#             logger.error("预测结果为空，无法保存")
#             return
#
#         # Save directory
#         results_dir = os.path.join(current_directory, 'results')
#         os.makedirs(results_dir, exist_ok=True)
#
#         # Base path
#         if args.opt_strategy == 0:
#             suffix = ''
#         elif args.opt_strategy == 1:
#             suffix = '_sam'
#         else:
#             suffix = '_gsam-reg'
#
#         base_name = f"predictions_{args.model}_{args.data}{suffix}"
#
#         # Save as NPZ - 保存所有数据
#         npz_path = os.path.join(results_dir, f"{base_name}_{args.pred_len}_{args.lambda_reg}.npz")
#         np.savez(npz_path,
#                  X_input=X_input,
#                  y_true=y_true,
#                  y_pred=y_pred,
#                  model_name=args.model,
#                  data_name=args.data,
#                  target_channel=-1)  # 保存目标通道信息
#
#         logger.info(f"预测数据已保存到: {npz_path}")
#         logger.info("可视化图像将在后续步骤中生成")
#
#     except Exception as e:
#         logger.error(f"保存预测结果时出错: {e}")
#         logger.error(f"错误堆栈: {traceback.format_exc()}")
#         raise

# def save_predictions(model, test_data, args, current_directory):
#     logger = logging.getLogger(__name__)
#
#     try:
#         # 清理 GPU 内存
#         import tensorflow as tf
#         tf.keras.backend.clear_session()
#         import gc
#         gc.collect()
#
#         logger.info("开始提取测试数据...")
#         X_input, y_true = extract_data(test_data)
#         logger.info(f"数据提取成功: X_input形状={X_input.shape}, y_true形状={y_true.shape}")
#
#         logger.info("开始模型预测...")
#         y_pred = model.predict(X_input, batch_size=getattr(args, 'batch_size', None), verbose=0)
#         logger.info(f"预测完成: y_pred类型={type(y_pred)}, 形状={getattr(y_pred, 'shape', '无shape属性')}")
#
#         # 处理 RaggedTensor 特殊情况
#         if hasattr(y_pred, 'numpy') and hasattr(y_pred, 'to_tensor'):
#             logger.info("检测到 RaggedTensor，转换为常规张量")
#             y_pred = y_pred.to_tensor()  # 转换为常规张量
#             logger.info(f"转换为常规张量后: 形状={y_pred.shape}")
#
#         # 详细检查预测输出
#         if isinstance(y_pred, (list, tuple)):
#             logger.info(f"输出为列表/元组，长度={len(y_pred)}")
#             for i, item in enumerate(y_pred):
#                 logger.info(f"  第{i}个元素: 类型={type(item)}, 形状={getattr(item, 'shape', '无shape属性')}")
#             y_pred = y_pred[0]  # 通常取第一个元素
#
#         logger.info(f"处理后的y_pred: 类型={type(y_pred)}, 形状={getattr(y_pred, 'shape', '无shape属性')}")
#
#         # 如果是TensorFlow张量，转换为numpy
#         if hasattr(y_pred, 'numpy'):
#             y_pred = y_pred.numpy()
#             logger.info(f"转换为numpy后: 形状={y_pred.shape}")
#
#         # 转换为numpy数组
#         y_pred = np.array(y_pred)
#         y_true = np.array(y_true)
#         logger.info(f"最终y_pred形状: {y_pred.shape}")
#         logger.info(f"最终y_true形状: {y_true.shape}")
#
#         # 检查数据有效性
#         if y_pred.size == 0:
#             logger.error("预测结果为空，无法生成图片")
#             return
#
#         # Save directory
#         results_dir = os.path.join(current_directory, 'results')
#         os.makedirs(results_dir, exist_ok=True)
#
#         # Base path
#         if args.opt_strategy == 0:
#             suffix = ''
#         elif args.opt_strategy == 1:
#             suffix = '_sam'
#         else:
#             suffix = '_gsam-reg'
#
#         base_name = f"predictions_{args.model}_{args.data}{suffix}"
#
#         # Save as NPZ
#         npz_path = os.path.join(results_dir, f"{base_name}.npz")
#         np.savez(npz_path, y_true=y_true, y_pred=y_pred)
#         logger.info(f"Predictions saved to {npz_path}")
#
#         # ==================== 画图代码 ====================
#         logger.info("开始生成可视化图片...")
#
#         # 选择目标通道（OT列，通常是最后一个特征）
#         target_channel = -1  # 假设OT是最后一个特征
#
#         logger.info(f"尝试使用形状: y_pred={y_pred.shape}, y_true={y_true.shape}")
#
#         # 根据实际形状调整索引
#         try:
#             if y_pred.ndim == 4:  # 如果是4维 [batch, None, seq, features]
#                 logger.info("检测到4维y_pred，使用索引模式: [batch, 0, seq, features]")
#                 last_idx = -1
#                 x_hist = X_input[last_idx, :, target_channel]  # 历史数据 [seq_len]
#                 y_true_vec = y_true[last_idx, :, target_channel]  # 真实值 [pred_len]
#                 y_pred_vec = y_pred[last_idx, 0, :, target_channel]  # 预测值 [pred_len]
#
#             elif y_pred.ndim == 3:  # 如果是3维 [batch, seq, features]
#                 logger.info("检测到3维y_pred，使用索引模式: [batch, seq, features]")
#                 last_idx = -1
#                 x_hist = X_input[last_idx, :, target_channel]  # 历史数据 [seq_len]
#                 y_true_vec = y_true[last_idx, :, target_channel]  # 真实值 [pred_len]
#                 y_pred_vec = y_pred[last_idx, :, target_channel]  # 预测值 [pred_len]
#
#             elif y_pred.ndim == 2:  # 如果是2维 [batch, features]（单点预测）
#                 logger.info("检测到2维y_pred，使用单点预测模式")
#                 last_idx = -1
#                 x_hist = X_input[last_idx, :, target_channel]  # 历史数据 [seq_len]
#                 y_true_vec = y_true[last_idx, :, target_channel]  # 真实值 [pred_len]
#                 # 创建与真实值相同长度的预测值数组
#                 y_pred_vec = np.full(y_true_vec.shape, y_pred[last_idx, target_channel])
#
#             elif y_pred.ndim == 1:  # 如果是1维 [features]
#                 logger.info("检测到1维y_pred，使用单特征模式")
#                 last_idx = -1
#                 x_hist = X_input[last_idx, :, target_channel]  # 历史数据 [seq_len]
#
#                 # 对于 y_true，我们需要选择正确的通道
#                 if y_true.ndim == 3:  # [batch, seq, features]
#                     y_true_vec = y_true[last_idx, :, target_channel]  # 真实值 [pred_len]
#                 elif y_true.ndim == 2:  # [batch, seq]
#                     y_true_vec = y_true[last_idx, :]  # 真实值 [pred_len]
#                 else:
#                     logger.error(f"不支持的y_true维度: {y_true.ndim}")
#                     return
#
#                 # 创建与真实值相同长度的预测值数组
#                 y_pred_vec = np.full(y_true_vec.shape, y_pred[target_channel])
#
#             else:
#                 logger.error(f"不支持的y_pred维度: {y_pred.ndim}")
#                 # 保存原始数据用于调试
#                 debug_path = os.path.join(results_dir, f"{base_name}_debug.npz")
#                 np.savez(debug_path, X_input=X_input, y_true=y_true, y_pred=y_pred)
#                 logger.info(f"调试数据已保存到: {debug_path}")
#                 return
#
#         except Exception as e:
#             logger.error(f"数据索引错误: {e}")
#             # 保存原始数据用于调试
#             debug_path = os.path.join(results_dir, f"{base_name}_debug.npz")
#             np.savez(debug_path, X_input=X_input, y_true=y_true, y_pred=y_pred)
#             logger.info(f"调试数据已保存到: {debug_path}")
#             return
#
#         # 简单拼接历史数据和未来预测
#         gt_series = np.concatenate([x_hist, y_true_vec])
#         pd_series = np.concatenate([x_hist, y_pred_vec])
#
#         logger.info(f"画图数据: x_hist长度={len(x_hist)}, y_true长度={len(y_true_vec)}, y_pred长度={len(y_pred_vec)}")
#
#         # 创建和保存图片
#         plt.figure(figsize=(12, 6))
#         plt.plot(gt_series, label='GroundTruth', linewidth=2)
#         plt.plot(pd_series, label='Prediction', linewidth=2)
#         plt.axvline(x=len(x_hist) - 1, color='gray', linestyle='--', linewidth=1)
#         plt.xlabel('time steps')
#         plt.ylabel('value')
#         plt.title(f'Prediction Length = {args.pred_len}')
#         plt.legend()
#         plt.grid(True)
#
#         # 保存图片
#         plot_path = os.path.join(results_dir, f"{base_name}_{args.data}_{args.lambda_reg}+.png")
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()
#
#         logger.info(f"预测可视化图片保存到: {plot_path}")
#
#         # ==================== 画图代码结束 ====================
#
#     except Exception as e:
#         logger.error(f"保存预测结果时出错: {e}")
#         logger.error(f"错误堆栈: {traceback.format_exc()}")
#         raise