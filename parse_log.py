import re

def parse_losses_from_log(log_path):
    bce_losses = []
    ssm_errors = []

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'BCE loss: ([\d.]+), SSM error: ([\d.]+)', line)
            if match:
                bce = float(match.group(1))
                ssm = float(match.group(2))
                bce_losses.append(bce)
                ssm_errors.append(ssm)

    return bce_losses, ssm_errors


def estimate_alpha(bce_losses, ssm_errors):
    ratios = [b / s if s > 0 else 0 for b, s in zip(bce_losses, ssm_errors)]
    return sum(ratios) / len(ratios)

if __name__ == '__main__':
    log_path = "/Users/valerizab/Desktop/masters-diploma/sing-learned-structure/data/logs/training_15_epochs_lca.log"
    bce, ssm = parse_losses_from_log(log_path)
    alpha_estimate = estimate_alpha(bce, ssm)

    print(f"Среднее значение alpha = {alpha_estimate:.5f}")