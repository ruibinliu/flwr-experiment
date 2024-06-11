import multiprocessing
import time

from client import start_client
from server import start_server


if __name__ == "__main__":
    nround = 5
    nclient = 3

    # for ahead in range(7):
    ahead = 0
    process = multiprocessing.Process(target=start_server, args=(nround, nclient))
    process.start()

    for node_id in [1, 2, 3]:
        time.sleep(1)
        epochs = 1
        client_process = multiprocessing.Process(target=start_client, args=(node_id, epochs, ahead))
        client_process.start()

    process.join()
