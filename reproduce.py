import torch
import ray
from tsp_main import get_network as tsp_get_network_fn
from tsp_main import test as tsp_test_fn
from cvrp_main import get_network as cvrp_get_network_fn
from cvrp_main import test as cvrp_test_fn
from jssp_main import test as jssp_test_fn
from jssp_main import get_network as jssp_get_network_fn
from tsp.config import TSPConfig
from cvrp.config import CVRPConfig
from jssp.config import JSSPConfig

"""
Reproduction script for greedy results to make getting started hopefully easier.
Alter the variable `devices_for_eval_workers` for GPU usage or more workers. 
"""
CUDA_VISIBLE_DEVICES = "0,1"  # Must be set for ray, as it can have difficulties detecting multiple GPUs
devices_for_eval_workers = ["cpu"] * 4  # i-th entry is device that the i-th worker should use for evaluation
beams_with_batch_sizes = {
    1: 4  # Beam width 1 (= greedy) with a batch size of 4 per worker.
}


def reproduce_tsp(network_type: str):
    print("================")
    print(f"TSP greedy results, {network_type} network trained with Gumbeldore")
    print("================")
    tsp_config = TSPConfig()
    tsp_config.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    tsp_config.gumbeldore_eval = False
    if network_type == "bq":
        tsp_config.architecture = "BQ"
        tsp_config.latent_dimension = 128
        tsp_config.num_transformer_blocks = 9
        tsp_config.num_attention_heads = 8
    elif network_type == "lehd":
        tsp_config.architecture = "LEHD"
        tsp_config.latent_dimension = 128
        tsp_config.num_transformer_blocks = 6
        tsp_config.num_attention_heads = 8
    else:
        raise ValueError(f"Unknown network type {network_type}")
    tsp_config.load_checkpoint_from_path = f"./model_checkpoints/tsp/{network_type}/checkpoint.pt"
    tsp_config.num_epochs = 0
    tsp_config.devices_for_eval_workers = devices_for_eval_workers
    tsp_config.beams_with_batch_sizes = beams_with_batch_sizes
    network = tsp_get_network_fn(tsp_config, torch.device("cpu"))
    checkpoint = torch.load(tsp_config.load_checkpoint_from_path)
    network.load_state_dict(checkpoint["model_weights"])
    for num_nodes, test_path in [
        (100, "./data/tsp/tsp_100_10k_seed1234.pickle"),
        (200, "./data/tsp/tsp_200_128_seed777.pickle"),
        (500, "./data/tsp/tsp_500_128_seed777.pickle")
    ]:
        tsp_config.test_set_path = test_path
        print(f"TSP N={num_nodes}")
        print("------------")
        with torch.no_grad():
            test_loggable_dict = tsp_test_fn(tsp_config, network)
            print(test_loggable_dict)


def reproduce_cvrp(network_type: str):
    print("================")
    print(f"CVRP greedy results, {network_type} network trained with Gumbeldore")
    print("================")
    cvrp_config = CVRPConfig()
    cvrp_config.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    cvrp_config.gumbeldore_eval = False
    if network_type == "bq":
        cvrp_config.architecture = "BQ"
        cvrp_config.latent_dimension = 192
        cvrp_config.num_transformer_blocks = 9
        cvrp_config.num_attention_heads = 12
    elif network_type == "lehd":
        cvrp_config.architecture = "LEHD"
        cvrp_config.latent_dimension = 128
        cvrp_config.num_transformer_blocks = 6
        cvrp_config.num_attention_heads = 8
    else:
        raise ValueError(f"Unknown network type {network_type}")
    cvrp_config.load_checkpoint_from_path = f"./model_checkpoints/cvrp/{network_type}/checkpoint.pt"
    cvrp_config.num_epochs = 0
    cvrp_config.devices_for_eval_workers = devices_for_eval_workers
    cvrp_config.beams_with_batch_sizes = beams_with_batch_sizes
    network = cvrp_get_network_fn(cvrp_config, torch.device("cpu"))
    checkpoint = torch.load(cvrp_config.load_checkpoint_from_path)
    network.load_state_dict(checkpoint["model_weights"])
    for num_nodes, test_path in [
        (100, "./data/cvrp/cvrp_100_10k_test.pickle"),
        (200, "./data/cvrp/cvrp_200_128_test.pickle"),
        (500, "./data/cvrp/cvrp_500_128_test.pickle")
    ]:
        cvrp_config.test_set_path = test_path
        print(f"CVRP N={num_nodes}")
        print("------------")
        with torch.no_grad():
            test_loggable_dict = cvrp_test_fn(cvrp_config, network)
            print(test_loggable_dict)


def reproduce_jssp():
    print("================")
    print("JSSP greedy results, trained with Gumbeldore")
    print("================")
    jssp_config = JSSPConfig()
    jssp_config.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    jssp_config.gumbeldore_eval = False
    jssp_config.load_checkpoint_from_path = "./model_checkpoints/jssp/checkpoint.pt"
    jssp_config.num_epochs = 0
    jssp_config.devices_for_eval_workers = devices_for_eval_workers
    jssp_config.beams_with_batch_sizes = beams_with_batch_sizes
    network = jssp_get_network_fn(jssp_config, torch.device("cpu"))
    checkpoint = torch.load(jssp_config.load_checkpoint_from_path)
    network.load_state_dict(checkpoint["model_weights"])
    for taillard_size, test_path in [
        ("15x15", "./data/jssp/literature/jssp_taillard_15_15.pickle"),
        ("20x15", "./data/jssp/literature/jssp_taillard_20_15.pickle"),
        ("20x20", "./data/jssp/literature/jssp_taillard_20_20.pickle"),
        ("30x15", "./data/jssp/literature/jssp_taillard_30_15.pickle"),
        ("30x20", "./data/jssp/literature/jssp_taillard_30_20.pickle"),
        ("50x15", "./data/jssp/literature/jssp_taillard_50_15.pickle"),
        ("50x20", "./data/jssp/literature/jssp_taillard_50_20.pickle"),
        ("100x20", "./data/jssp/literature/jssp_taillard_100_20.pickle")
    ]:
        jssp_config.test_set_path = test_path
        print(f"Taillard {taillard_size}")
        print("------------")
        with torch.no_grad():
            test_loggable_dict = jssp_test_fn(jssp_config, network)
            print(test_loggable_dict)


if __name__ == '__main__':
    num_gpus = len(set([d for d in devices_for_eval_workers if d != "cpu"]))
    ray.init(num_gpus=num_gpus, logging_level="info")
    reproduce_tsp("bq")
    reproduce_tsp("lehd")
    reproduce_cvrp("bq")
    reproduce_cvrp("lehd")
    reproduce_jssp()
    ray.shutdown()