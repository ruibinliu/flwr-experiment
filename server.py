import argparse
from datetime import datetime
import os
import time

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict

nrounds = 0
start_time = 0
time_str = datetime.now().strftime('%m/%d_%H%M%S')

# Create output folder
output_dir = f'data/output/{time_str}_fl'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def configure_fit(server_round: int) -> Dict:
    print(f'configure_fit, server_round={server_round}')
    """Send round number to client."""
    global start_time
    if server_round == 1:
        print(f":: Starting global execution timer...")
        start_time = time.time()
    print(f":: Starting 'fit' round #{server_round}...")
    return {"server_round": server_round, "last_round": (server_round == nrounds), "output_dir": output_dir}


def configure_evaluate(server_round: int) -> Dict:
    """Send round number to client."""
    print(f":: Starting 'evaluate' round #{server_round}...")
    return {"server_round": server_round, "last_round": (server_round == nrounds), "output_dir": output_dir}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def start_server(nclients, nrounds):
    print(f'Start server is called. nclients={nclients}, nrounds={nrounds}')
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=nclients,  # Never sample less than 3 clients for training
        min_evaluate_clients=nclients,  # Never sample less than 3 clients for evaluation
        min_available_clients=nclients,  # Wait until all 3 clients are available
        on_fit_config_fn=configure_fit,  # Send optional parameters to client before 'fit'
        on_evaluate_config_fn=configure_evaluate,  # Send optional parameters to client before 'evaluate'
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        # initial_parameters=fl.common.ndarrays_to_parameters(init_params)
    )

    fl.server.start_server(
        server_address="0.0.0.0:18080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=nrounds, round_timeout=None)
    )

    # Compute execution time
    end_time = time.time()
    total_time = round(end_time - start_time, 1)
    print(f'Execution total time is {total_time} seconds.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument(
        "-c",
        "--num_clients",
        type=int,
        default=3,
        required=False,
        help="Number of clients",
    )
    parser.add_argument(
        "-r",
        "--num_rounds",
        type=int,
        default=5,
        required=False,
        help="Number of rounds",
    )
    args = parser.parse_args()

    start_server(args.num_clients, args.num_rounds)
