#!/bin/bash

# Define the parameter name
num_cycle="num_cycle"
train_steps="train_steps"
lr="lr"
mode_loss_weight="mode_loss_weight"
slice_shift_penalty_weigth="slice_shift_penalty_weigth"
global_shift_penalty_weigth="global_shift_penalty_weigth"
rotation_penalty_weigth="rotation_penalty_weigth"

# Define the range for the random float
min_num_cycle=1
max_num_cycle=1
min_train_steps=1000 #50 less than in original paper
max_train_steps=1000 #200 more than I used previously
min_lr=0.00001 #1/3 of what they used in the original paper
max_lr=0.1 #double of what I used previously
min_mode_weight=7.405277111193427e-07
max_mode_weight=7.405277111193427e-07 #double what they used in the original paper
min_slice_shift=10
max_slice_shift=10 #double what they used in the original paper
min_rotation=1
max_rotation=1 #double what they used in the original paper
min_global_shift=0.3
max_global_shift=0.3 #double what they used in the original paper

# Define output file
output_file="Hyperparameter_search/Hyperparameter_learning_rate_ES.csv"

# Clear the output file if it exists
> "$output_file"

# Number of times to run the script
num_runs=200

# Function to round to 3 significant digits
round_to_3_sig() {
    python3 -c "import math, sys; val = float(sys.argv[1]); print(round(val, -int(math.floor(math.log10(abs(val)))) + 2) if val != 0 else 0)" "$1"
}

# Loop for the specified number of runs
for ((i=1; i<=$num_runs; i++))
do
    # Generate random values
    random_num_cycle=$(python -c "import random; print(random.randint($min_num_cycle, $max_num_cycle))")
    random_train_steps=$(python -c "import random; print(random.randint($min_train_steps, $max_train_steps))")
    
    # Generate float values with full precision
    random_lr_full=$(python -c "import random; print(random.uniform($min_lr, $max_lr))")
    random_mode_weight_full=$(python -c "import random; print(random.uniform($min_mode_weight, $max_mode_weight))")
    random_rotation_full=$(python -c "import random; print(random.uniform($min_rotation, $max_rotation))")
    random_global_shift_full=$(python -c "import random; print(random.uniform($min_global_shift, $max_global_shift))")
    
    random_slice_shift=$(python -c "import random; print(random.randint($min_slice_shift, $max_slice_shift))")
    
    # Round float values to 3 significant digits for display and logging
    random_lr=$(round_to_3_sig $random_lr_full)
    random_mode_weight=$(round_to_3_sig $random_mode_weight_full)
    random_rotation=$(round_to_3_sig $random_rotation_full)
    random_global_shift=$(round_to_3_sig $random_global_shift_full)
    
    echo $random_num_cycle
    echo $random_train_steps
    echo $random_lr
    echo $random_mode_weight
    echo $random_slice_shift
    echo $random_rotation
    echo $random_global_shift
    
    # Run Python script with full precision values for better accuracy
    full_output=$(python Hyperparam_Main.py \
        --num_cycle "$random_num_cycle" \
        --training_steps "$random_train_steps" \
        --lr "$random_lr_full" \
        --mode_loss_weight "$random_mode_weight_full" \
        --slice_shift_penalty_weight "$random_slice_shift" \
        --rotation_penalty_weight "$random_rotation_full" \
        --global_shift_penalty_weight "$random_global_shift_full")
    
    # Optional: echo to terminal
    echo "$full_output"
    
    # Extract dice score
    Myo_dice=$(echo "$full_output" | grep "Myocardium Dice" | tail -n 1 | awk '{print $NF}')
    Bp_dice=$(echo "$full_output" | grep "Blood Pool Dice" | tail -n 1 | awk '{print $NF}')

    #ESV extraction
    EDV=$(echo "$full_output" | grep "EDV" | tail -n 1 | awk '{print $NF}')
    ESV=$(echo "$full_output" | grep "ESV" | tail -n 1 | awk '{print $NF}')
    EF=$(echo "$full_output" | grep "EF" | tail -n 1 | awk '{print $NF}')

    
    # Create table row with rounded parameters and dice score for clean output
    table_row=$random_lr,$Myo_dice,$Bp_dice,$EDV,$ESV,$EF
    
    # If this is the first run, create header
    if [ ! -f "$output_file" ]; then
        echo "lr,dice_score,myo_dice,bp_dice,EDV,ESV,EF" > "$output_file"
    fi
    
    # Append the results
    echo "$table_row" >> "$output_file"
    
    echo "Results saved to $output_file"
done

echo "Output has been saved to $output_file"