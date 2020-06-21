import json
import logging
import pickle

from CacheManager import CacheManager
from TqdmLoggingHandler import TqdmLoggingHandler

__author__ = "Pierre Monnin"


def load_configuration_parameters(conf_file_path):
    """
    Load configuration parameters from a JSON file whose path is given
    :param conf_file_path: path to the JSON configuration file
    :return: a dict with the contents of the JSON configuration file
    """

    with open(conf_file_path, 'r') as configuration_file:
        configuration_parameters = json.load(configuration_file, encoding="utf-8")

    return configuration_parameters


def get_logger(debug=False):
    """
    Get a logger (not breaking tqdm progress bars)
    :param debug: if the logger level is set to DEBUG (otherwise set to INFO)
    :return: a logger object
    """

    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)

    return logger


def get_blacklists(configuration_parameters):
    """
    Return blacklists from the configuration parameters
    :param configuration_parameters: configuration parameters as in the JSON configuration file
    :return: a dict with 3 keys (path_predicates, types and types_expansion), each associated with a blacklist
    """

    return {
        "path_predicates": configuration_parameters["path_predicates_blacklist"],
        "types": configuration_parameters["types_blacklist"],
        "types_expansion": configuration_parameters["types_expansion_blacklist"]
    }


def load_graph(graph_dir):
    """
    Load graph from the given directory
    :param graph_dir: directory where canonical graph files are stored
    :return: a dict containing data of the canonical graph (mapping canonical <-> rdf, and adjacencies)
    """

    if graph_dir[-1] != "/":
        graph_dir += "/"

    return {
        "rdf_to_canonical_index": pickle.load(open(graph_dir + "rdf_to_canonical_index", "rb")),
        "canonical_to_rdf_index": pickle.load(open(graph_dir + "canonical_to_rdf_index", "rb")),
        "adjacency": pickle.load(open(graph_dir + "canonical_graph_adjacency", "rb")),
        "inv_adjacency": pickle.load(open(graph_dir + "canonical_graph_inv_adjacency", "rb"))
    }


def load_cache_managers(cache_managers_dir):
    """
    Load cache managers from the given directory
    :param cache_managers_dir: directory where cache managers files are stored
    :return: a dict with 2 keys representing the 2 cache managers (rdf_nodes and predicates)
    """

    if cache_managers_dir[-1] != "/":
        cache_managers_dir += "/"

    cache_managers = {
        "rdf_nodes": CacheManager(),
        "predicates": CacheManager()
    }
    cache_managers["predicates"].load_from_csv(cache_managers_dir + "predicates_cache_manager.csv")
    cache_managers["rdf_nodes"].load_from_csv(cache_managers_dir + "rdf_nodes_cache_manager.csv")

    return cache_managers


def load_dataset(dataset_file):
    """
    Load dataset from the given file
    :param dataset_file: the path to the dataset file
    :return: a dict associating each URI representing a seed node of the dataset with its class (0 or 1)
    """

    classes = dict()

    with open(dataset_file, "r") as dataset:
        for line in dataset:
            tline = line.rstrip().split(",")
            uri = tline[0]
            classes[uri] = tline[1]

    return classes
