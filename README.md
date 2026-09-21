# Training First Network in PyTorch

A neural network trained on a real estate pricing dataset to predict house prices per unit, comparing a baseline run against a hyperparameter variation.


## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# 3. Install required dependencies
pip install -r requirements.txt

## Run
# To run the baseline model (Adam Optimizer, lr = 0.01)
python train.py adam  

# To run the variation model (SGD Optimizer, lr = 0.01)
python train.py sgd   

## Example 
Baseline (Adam, lr=0.01): Final Loss = 53.5339
Variation(SGD,lr=0.01): Final Loss = 184.7355

## What changed and why
I only changed optimizer type from Adam to SGD. Everything else remained same
As Expected Adam(Adaptive gradient method) works far superior than SGD on tabular data with small numnber of epochs, SGD requires more fine tuning to get better results.
Final Loss for Adam was 53.5339 and for SGD it spiked to 184.7355
## What happened
SGD progressed more slowly than Adam
