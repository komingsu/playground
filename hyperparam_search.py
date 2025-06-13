import argparse
import torch
from sklearn.model_selection import ParameterGrid
import config
from main import train


def run_search(param_grid, device):
    best_acc = 0.0
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        print(f"\nEvaluating params: {params}")
        config.ARCFACE_PARAMS.update({
            'margin': params['margin'],
            'scale': params['scale']
        })

        args = argparse.Namespace(device=device, subjects=config.SUB)
        metrics = train(args)
        acc = metrics.get('best_acc', 0)
        results.append((params, acc))

        if acc > best_acc:
            best_acc = acc
            best_params = params

    print(f"Best params: {best_params} -> Accuracy: {best_acc}")
    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid = {
        'margin': [0.2, 0.4, 0.5],
        'scale': [15, 30, 64]
    }
    run_search(grid, device)
