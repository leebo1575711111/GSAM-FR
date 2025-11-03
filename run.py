# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
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
import numpy as np
from tensorflow import keras
import argparse
import logging
import os
import sys
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import traceback

from utils import (
    compile_model,
    configure_environment,
    create_optimizer,
    initialize_model,
    load_data,
    log_model_info,
    save_predictions,
    save_additional_metrics,
    save_main_results,
    save_training_history,
    setup_callbacks,
    setup_experiment_id,
    train_model,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    """Parses command line arguments for the training experiment."""
    parser = argparse.ArgumentParser(description="Train models for Time Series Forecasting.")

    parser.add_argument("--model", type=str, default="transformer",
                        choices=["tsmixer", "iTransformer", "transformer", "transformer_random", "spectrans", "linear"],
                        help="Model to train.")

    parser.add_argument("--opt_strategy", type=int, default="2",
                        choices=[0, 1, 2],
                        help="0:None, 1:SAM, 2:GSAMFR")

    parser.add_argument("--data", type=str, default="electricity",
                        choices=["electricity", "exchange_rate", "weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2",
                                 "traffic"],
                        help="Dataset for training.")

    parser.add_argument("--feature_type", type=str, default="M",
                        choices=["S", "M", "MS"],
                        help="Type of forecasting task.")

    parser.add_argument('--use_attention', action='store_true',
                        help='Whether to use attention mechanism (default: False)')

    parser.add_argument('--no_attention', dest='use_attention', action='store_false',
                        help='Disable attention mechanism')
    parser.set_defaults(use_attention=True) 

    parser.add_argument("--target", type=str, default="OT",
                        help="Target feature for S or MS task.")

    parser.add_argument("--seq_len", type=int, default=96,
                        help="Input sequence length.")

    parser.add_argument("--pred_len", type=int, default=96,
                        help="Prediction sequence length.")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")

    parser.add_argument("--train_epochs", type=int, default=300,
                        help="Total number of training epochs.")

    parser.add_argument("--rho", type=float, default=0.4,
                        help="Rho parameter for SAM, if used.")

    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Alpha parameter for GSAMFR, if used.")

    parser.add_argument("--lambda_reg", type=float, default=0.001,
                        help="lambda_reg parameter for GSAMFR, if used.")

    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="Learning rate for optimizer.")

    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping.")

    parser.add_argument("--n_block", type=int, default=2,
                        help="Number of blocks in the model architecture.")

    parser.add_argument("--ff_dim", type=int, default=2048,
                        help="Dimension of feed-forward layers.")

    parser.add_argument("--num_heads", type=int, default=1,
                        help="Number of heads in multi-head attention layers.")

    parser.add_argument("--d_model", type=int, default=16,
                        help="Dimensionality of the model embeddings.")

    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate.")

    parser.add_argument("--norm_type", type=str, default="B", choices=["L", "B"],
                        help="Normalization type: LayerNorm (L) or BatchNorm (B).")

    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"],
                        help="Activation function.")

    parser.add_argument("--seed", type=int, default=1250,
                        help="Random seed for reproducibility.")

    parser.add_argument("--checkpoint_dir", type=str, 
                        default="checkpoints",
                        help="Directory to save model checkpoints.")

    parser.add_argument("--delete_checkpoint", action="store_true",
                        help="Whether to delete model checkpoints after training.")

    parser.add_argument("--result_path", type=str, default="results.csv",
                        help="Path to save the training results.")

    parser.add_argument("--add_results", action="store_true",
                        help="Whether to save additional results.")

    parser.add_argument('--save_predictions', action='store_true',
                        help='Save per-sample predictions and true values after testing')

    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--memory_growth', action='store_true', help='Allow GPU memory dynamic growth')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps for simulating larger batch sizes')

    parser.add_argument('--gpu_id', type=str, default=None,
                        help='Specify GPU ID to use (e.g., "0" or "0,1")')

    return parser.parse_args()


def setup_gpu(gpu_id=None, memory_growth=False):
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logging.warning("No GPU devices detected, will run on CPU")
        return

    # Set visible GPU devices
    if gpu_id is not None:
        try:
            gpu_ids = [int(id) for id in gpu_id.split(',')]
            visible_gpus = [gpus[i] for i in gpu_ids]
            tf.config.set_visible_devices(visible_gpus, 'GPU')
            logging.info(f"Set visible GPU devices: {gpu_ids}")
        except (ValueError, IndexError) as e:
            logging.error(f"Invalid GPU ID setting: {gpu_id}, error: {e}")
            logging.info("Will use all available GPUs")

    # Set memory growth
    if memory_growth:
        for gpu in tf.config.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("Enabled GPU memory growth")

    # Log GPU information
    visible_gpus = tf.config.get_visible_devices('GPU')
    if visible_gpus:
        logging.info(f"Available GPU devices: {visible_gpus}")
    else:
        logging.info("No GPU devices available, will use CPU")

    print("TF version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

def main():

    # Create a named logger
    logger = logging.getLogger(__name__)  # __name__ automatically sets to current module name
    # Parse command line arguments
    args = parse_args()

    # Set up GPU
    setup_gpu(args.gpu_id, args.memory_growth)

    # Set global random seeds for experiment reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        import tensorflow as tf

        # Set all random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

        # Set TensorFlow deterministic operations
        try:
            tf.config.experimental.enable_op_determinism()
            logging.info("Enabled operation determinism")
        except AttributeError:
            logging.warning("Current TensorFlow version does not support enable_op_determinism")

        logging.info(f"Set random seed to: {args.seed}")

    # Set up mixed precision training
    if args.mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Enabled mixed precision training")
            logging.info(f"Compute dtype: {policy.compute_dtype}")
            logging.info(f"Variable dtype: {policy.variable_dtype}")
        except Exception as e:
            logging.error(f"Failed to enable mixed precision: {e}")

    # Check if GPU is available
    gpu_available = tf.config.list_physical_devices('GPU')
    if not gpu_available:
        logging.warning("Warning: No GPU devices detected, training will run on CPU")
    else:
        logging.info(f"Detected {len(gpu_available)} GPU devices")

    # Configure execution environment
    current_directory = configure_environment()

    # Set up experiment identifier
    exp_id = setup_experiment_id(args)

    # Load data
    train_data, val_data, test_data, n_features = load_data(args)

    # Model initialization and logging configuration
    model = initialize_model(args, n_features)
    log_model_info(model, args)

    # Optimizer setup and model compilation
    optimizer = create_optimizer(args)
    compile_model(model, optimizer)

    # Configure model checkpoint callbacks
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{exp_id}_best.h5")

    # Set up callbacks
    callbacks, capture_weights_callback = setup_callbacks(args, checkpoint_path, model)

    # Start training process
    start_training_time = time.time()
    history = train_model(model, train_data, val_data, args, callbacks)
    elapsed_training_time = time.time() - start_training_time
    logging.info(f"Training completed, time elapsed: {elapsed_training_time:.2f} seconds.")

    # Evaluate model on test data
    try:
        # Load best weights
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            logging.info("Loaded best weights for testing")
        else:
            logging.warning(f"Checkpoint file not found: {checkpoint_path}, using final weights for testing")

        # Evaluate model
        test_result = model.evaluate(test_data)
        logging.info(f"Test results: {test_result}")
    except tf.errors.OpError as e:
        logging.error(f"Model evaluation error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during evaluation: {e}")
        sys.exit(1)

    # Clean up checkpoints and log
    if args.delete_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("Checkpoint file deleted.")

    # Save results and training history
    save_main_results(history, test_result, args, elapsed_training_time,
                      current_directory)
    save_training_history(history, args, current_directory)

    # Save additional metrics based on user request
    if args.add_results:
        save_additional_metrics(model, args, train_data, current_directory, capture_weights_callback)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
    main()
