import argparse
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict
import time

nrounds = 0

def configure_fit(server_round: int) -> Dict:
    """Send round number to client."""
    #print(f":: Starting 'fit' round #{server_round}...")
    return {"server_round": server_round, "last_round": (server_round == nrounds)}


def configure_evaluate(server_round: int) -> Dict:
    """Send round number to client."""
    #print(f":: Starting 'evaluate' round #{server_round}...")
    return {"server_round": server_round, "last_round": (server_round == nrounds)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    start_time = time.time()

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
    nclients = args.num_clients
    nrounds = args.num_rounds

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=nclients,           # Never sample less than 3 clients for training
        min_evaluate_clients=nclients,      # Never sample less than 3 clients for evaluation
        min_available_clients=nclients,     # Wait until all 3 clients are available
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
    execution_time = round(end_time - start_time, 1)
    print(f'Execution time of server is {execution_time} seconds.')