#!/bin/bash
# run_script.sh: Automated Experiment Runner for Time Series Forecasting Models
# Copyright 2025 Baofeng Liao

# Description:
# This script automates the execution of time series forecasting experiments using various models.
# It runs the Python training script (run.py) for specified models and datasets across multiple
# prediction horizons and configurations. The script supports different optimization strategies
# including GSAMFR and dynamic parameter adjustment based on dataset characteristics.

# Usage:
# ./run_script.sh -m <model_name> -d <dataset_name> [-s <sequence_length>] [-o <opt_strategy>] [-a]
#
# Options:
#   -m <model_name>        Model to use (e.g., tsmixer, base_model)
#   -d <dataset_name>      Dataset name (e.g., ETTh1, traffic, weather, electricity, exchange_rate)
#   -s <sequence_length>  Input sequence length (default: 96)
#   -o <opt_strategy>     Optimization strategy: 0=Adam, 1=SAM, 2=GSAMFR (default: 2)
#   -a                     Save additional results

# Examples:
# ./run_script.sh -m tsmixer -d ETTh1 -s 96 -o 2 -a
# ./run_script.sh -m tsmixer -d weather -o 1
# ./run_script.sh -m tsmixer -d electricity

# ----------------------------
# Configuration Section
# ----------------------------

# Initialize variables with default values
model=""
data=""
seq_len=96
opt_strategy=2
add_results_flag=""

# Experiment parameters
pred_lengths=(96 192 336 720)    # Prediction horizons
rhos=(0.4)                        # SAM rho values (default: 0.4)
alphas=(0.1)                      # GSAMFR alpha values (default: 0.1)
lambda_regs=(0.1)               # GSAMFR lambda_reg values (default: 0.1)
num_runs=1                        # Number of runs per configuration

# ----------------------------
# Command Line Argument Parsing
# ----------------------------

# Parse named command line arguments
while getopts "m:d:s:o:a" opt; do
  case ${opt} in
    m ) model=$OPTARG ;;
    d ) data=$OPTARG ;;
    s ) seq_len=$OPTARG ;;
    o ) opt_strategy=$OPTARG ;;
    a ) add_results_flag="--add_results" ;;
    \? ) 
        echo "Error: Invalid option -$OPTARG"
        echo "Usage: $0 -m <model> -d <dataset> [-s <seq_len>] [-o <opt_strategy>] [-a]"
        echo "       -m: Model name (required)"
        echo "       -d: Dataset name (required)"
        echo "       -s: Sequence length (default: 512)"
        echo "       -o: Optimization strategy: 0=None, 1=SAM, 2=GSAMFR (default: 2)"
        echo "       -a: Save additional results"
        exit 1 ;;
  esac
done

shift $((OPTIND -1))

# ----------------------------
# Input Validation
# ----------------------------

# Validate required parameters
if [ -z "$model" ] || [ -z "$data" ]; then
    echo "Error: Model and dataset parameters are required."
    echo "Usage: $0 -m <model> -d <dataset> [-s <seq_len>] [-u] [-a]"
    exit 1
fi

# Validate dataset name
valid_datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "electricity" "traffic" "exchange_rate" "toy")
if [[ ! " ${valid_datasets[@]} " =~ " ${data} " ]]; then
    echo "Warning: Dataset '$data' may not be supported. Continuing anyway..."
fi

# Validate sequence length
if ! [[ "$seq_len" =~ ^[0-9]+$ ]] || [ "$seq_len" -lt 1 ]; then
    echo "Error: Sequence length must be a positive integer."
    exit 1
fi

# ----------------------------
# Experiment Execution
# ----------------------------

echo "========================================"
echo "Starting Experiment Run"
echo "========================================"
echo "Model: $model"
echo "Dataset: $data"
echo "Sequence Length: $seq_len"
echo "Optimization Strategy: $opt_strategy (0=None, 1=SAM, 2=GSAMFR)"
echo "Additional Results: ${add_results_flag:--disabled}"
echo "Prediction Lengths: ${pred_lengths[*]}"
echo "SAM Rho Values: ${rhos[*]}"
echo "GSAMFR Alpha Values: ${alphas[*]}"
echo "GSAMFR Lambda_reg Values: ${lambda_regs[*]}"
echo "Number of Runs: $num_runs"
echo "========================================"

# Loop over prediction lengths and configurations
for pred_len in "${pred_lengths[@]}"
do
    echo ""
    echo "Processing prediction length: $pred_len"
    echo "----------------------------------------"
    
    for rho in "${rhos[@]}"
    do
        for alpha in "${alphas[@]}"
        do
            for lambda_reg in "${lambda_regs[@]}"
            do
                for (( run=1; run<=num_runs; run++ ))
                do
                    echo "Run $run/$num_runs - Rho: $rho, Alpha: $alpha, Lambda_reg: $lambda_reg"
            
            # Dynamic parameter adjustment based on dataset
            if [[ "$data" =~ ^ETT ]]; then
                learning_rate=0.001
                n_block=2
                dropout=0.9
                ff_dim=64
            elif [ "$data" = "weather" ]; then
                learning_rate=0.0001
                n_block=4
                dropout=0.3
                ff_dim=32
            elif [ "$data" = "electricity" ]; then
                learning_rate=0.0001
                n_block=4
                dropout=0.7
                ff_dim=64
            elif [ "$data" = "traffic" ]; then
                learning_rate=0.0001
                n_block=8
                dropout=0.7
                ff_dim=64
            elif [ "$data" = "exchange_rate" ]; then
                learning_rate=0.001
                n_block=8
                dropout=0.7
                ff_dim=64
            elif [ "$data" = "toy" ]; then
                learning_rate=0.001
                n_block=2
                dropout=0.9
                ff_dim=64
            else
                # Default parameters for unknown datasets
                learning_rate=0.001
                n_block=4
                dropout=0.5
                ff_dim=64
                echo "Warning: Using default parameters for unknown dataset: $data"
            fi
            
            # Construct the command with appropriate parameters
            command="python run.py --model $model --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate $learning_rate --n_block $n_block --dropout $dropout --ff_dim $ff_dim --num_heads 1 --d_model 16 --rho $rho --alpha $alpha --lambda_reg $lambda_reg --opt_strategy $opt_strategy ${add_results_flag}"
            
            echo "Executing: $command"
            echo "---"
            
            # Execute the command
            eval $command
            
            # Check exit status
            if [ $? -eq 0 ]; then
                echo "✓ Run completed successfully"
            else
                echo "✗ Run failed with exit code $?"
            fi
            echo ""
                done
            done
        done
    done
done

echo "========================================"
echo "Experiment Run Completed"
echo "========================================"
echo "All configurations have been processed."
echo "Check the output files for results."

# ----------------------------
# End of Script
# ----------------------------
