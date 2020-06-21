import argparse
import datetime
import pickle
import queue

import tqdm

from CacheManager import CacheManager
from ServerManager import ServerManager
from load_utils import load_configuration_parameters, get_logger


def get_rdf_graph_adjacency(server_manager, cache_managers, logger):
    """
    Query and return the RDF graph adjacency (only considering object predicates)
    :param server_manager: server manager
    :param cache_managers: dict with key ``predicates'' (CacheManager for predicates) and ``rdf_nodes'' (CacheManager
    for RDF nodes)
    :param logger: configured logger
    :return: dict with an entry per (object) predicate, associated with a dict (node -> set of nodes) corresponding to
    adjacency lists
    """

    rdf_graph_adjacency = {}

    # We do not need to add isIRI(?s) as RDF triples only accept URIs as subjects
    for p_uri in server_manager.query_elements("?s ?e ?o . FILTER(isIRI(?o))", verbose=True):
        p_index = cache_managers["predicates"].get_element_index(p_uri)

        rdf_graph_adjacency[p_index] = {}

        logger.info("Processing %s edges" % p_uri)

        edges = server_manager.query_two_elements("?e1 <{}> ?e2 . FILTER(isIRI(?e2))".format(p_uri), verbose=True)
        for (uri_1, uri_2) in tqdm.tqdm(edges):
            index_1 = cache_managers["rdf_nodes"].get_element_index(uri_1)
            index_2 = cache_managers["rdf_nodes"].get_element_index(uri_2)

            if index_1 not in rdf_graph_adjacency[p_index]:
                rdf_graph_adjacency[p_index][index_1] = set()

            rdf_graph_adjacency[p_index][index_1].add(index_2)

    return rdf_graph_adjacency


def same_as_adjacency_completion(rdf_graph_adjacency, predicates_cache_manager):
    """
    Return the RDF graph adjacency completed with owl:sameAs symmetry and transitive closure
    :param rdf_graph_adjacency: RDF graph adjacency (dict (predicate) -> dict (node) -> set (neighboring nodes))
    :param predicates_cache_manager: cache manager for predicates
    :return: RDF graph adjacency completed with owl:sameAs symmetry and transitive closure
    """

    same_as_index = predicates_cache_manager.get_element_index("http://www.w3.org/2002/07/owl#sameAs")
    if same_as_index not in rdf_graph_adjacency:
        rdf_graph_adjacency[same_as_index] = dict()

    # Add empty adjacency list when needed
    nodes_to_add_in_adj = set()
    for n_index in tqdm.tqdm(rdf_graph_adjacency[same_as_index]):
        for n2_index in rdf_graph_adjacency[same_as_index][n_index]:
            if n2_index not in rdf_graph_adjacency[same_as_index]:
                nodes_to_add_in_adj.add(n2_index)

    while len(nodes_to_add_in_adj) != 0:
        n = nodes_to_add_in_adj.pop()
        rdf_graph_adjacency[same_as_index][n] = set()

    # Symmetry
    for n_index in tqdm.tqdm(rdf_graph_adjacency[same_as_index]):
        for n2_index in rdf_graph_adjacency[same_as_index][n_index]:
            if n_index not in rdf_graph_adjacency[same_as_index][n2_index]:
                rdf_graph_adjacency[same_as_index][n2_index].add(n_index)

    # Transitivity (add connected component minus itself for each node)
    to_compute = set(rdf_graph_adjacency[same_as_index].keys())

    with tqdm.tqdm(total=len(to_compute)) as pbar:
        while len(to_compute) != 0:
            current_n_index = to_compute.pop()

            # Compute owl:sameAs connected component
            connected_component = {current_n_index}
            q = queue.Queue()
            q.put(current_n_index)

            while not q.empty():
                n_index = q.get()

                for n2_index in rdf_graph_adjacency[same_as_index][n_index]:
                    if n2_index not in connected_component:
                        connected_component.add(n2_index)
                        q.put(n2_index)

            # Add connected component minus itself for each node in the component
            for n_index in connected_component:
                rdf_graph_adjacency[same_as_index][n_index] = connected_component - {n_index}
                to_compute -= {n_index}
                pbar.update(1)

    return rdf_graph_adjacency


def get_canonical_graph(rdf_graph_adjacency, predicates_cache_manager):
    """
    Canonicalize RDF graph by transforming nodes connected by owl:sameAs into a unique node
    :param rdf_graph_adjacency: original RDF graph adjacency
    :param predicates_cache_manager: cache manager for predicates
    :return: dict with keys
    ``rdf_to_canonical_index'' (mapping from RDF node indices to canonical indices, int -> int),
    ``canonical_to_rdf_index'' (mapping from canonical node indices to RDF indices, int -> set),
    ``adjacency``, ``inv_adjacency`` (adjacencies for the canonical graph)
    """

    same_as_index = predicates_cache_manager.get_element_index("http://www.w3.org/2002/07/owl#sameAs")
    canonical_graph = dict()

    # Mapping from RDF indices to canonical indices
    rdf_to_canonical_index = {}  # RDF graph index -> canonical graph index
    canonical_to_rdf_index = []  # Canonical graph index -> set(RDF graph index)

    # Detect node to canonicalize
    nodes_to_canonicalize = set()

    for p_index in tqdm.tqdm(rdf_graph_adjacency):
        for n1_index in rdf_graph_adjacency[p_index]:
            nodes_to_canonicalize.add(n1_index)

            for n2_index in rdf_graph_adjacency[p_index][n1_index]:
                nodes_to_canonicalize.add(n2_index)

    # Canonicalize nodes
    with tqdm.tqdm(total=len(nodes_to_canonicalize)) as pbar:
        while len(nodes_to_canonicalize) != 0:
            n_index = nodes_to_canonicalize.pop()

            # If n_index is not already canonical
            if n_index not in rdf_to_canonical_index:
                same_as_component = {n_index}

                if n_index in rdf_graph_adjacency[same_as_index]:
                    same_as_component |= rdf_graph_adjacency[same_as_index][n_index]

                for n1_index in same_as_component:
                    rdf_to_canonical_index[n1_index] = len(canonical_to_rdf_index)

                canonical_to_rdf_index.append(set(same_as_component))

            pbar.update(1)

    canonical_graph["rdf_to_canonical_index"] = rdf_to_canonical_index
    canonical_graph["canonical_to_rdf_index"] = canonical_to_rdf_index

    # Canonicalize adjacencies
    canonical_graph_adjacency = {}
    for p_index in tqdm.tqdm(rdf_graph_adjacency):
        if p_index != same_as_index:  # owl:sameAs adjacency is useless in canonical graph so it is removed
            canonical_graph_adjacency[p_index] = {}

            for n1_index in rdf_graph_adjacency[p_index]:
                n1_canonical_index = rdf_to_canonical_index[n1_index]

                if n1_canonical_index not in canonical_graph_adjacency[p_index]:
                    canonical_graph_adjacency[p_index][n1_canonical_index] = set()

                for n2_index in rdf_graph_adjacency[p_index][n1_index]:
                    n2_canonical_index = rdf_to_canonical_index[n2_index]
                    canonical_graph_adjacency[p_index][n1_canonical_index].add(n2_canonical_index)

    canonical_graph["adjacency"] = canonical_graph_adjacency

    # Compute inverse adjacencies
    canonical_graph_inv_adjacency = {}

    for p_index in tqdm.tqdm(canonical_graph_adjacency):
        canonical_graph_inv_adjacency[p_index] = {}

        for n1_canonical_index in canonical_graph_adjacency[p_index]:
            for n2_canonical_index in canonical_graph_adjacency[p_index][n1_canonical_index]:
                if n2_canonical_index not in canonical_graph_inv_adjacency[p_index]:
                    canonical_graph_inv_adjacency[p_index][n2_canonical_index] = set()

                canonical_graph_inv_adjacency[p_index][n2_canonical_index].add(n1_canonical_index)

    canonical_graph["inv_adjacency"] = canonical_graph_inv_adjacency

    return canonical_graph


def get_number_of_edges(adjacency):
    """
    Count the number of edges in the given adjacency list
    :param adjacency: graph adjacency list
    :return: the number of edges in the given adjacency
    """
    edges_count = 0

    for p in tqdm.tqdm(adjacency):
        for n in adjacency[p]:
            edges_count += len(adjacency[p][n])

    return edges_count


def print_debug_output(canonical_graph, cache_managers, logger):
    logger.debug("--------- DEBUG OUTPUT BEGIN ---------")

    logger.debug("----- CANONICAL ADJACENCY BEGIN ------")

    for p_index in canonical_graph["adjacency"]:
        logger.debug("Predicate {} / {}".format(p_index,
                                                cache_managers["predicates"].get_element_from_index(p_index)))

        for n1_canonical_index in canonical_graph["adjacency"][p_index]:
            for n2_canonical_index in canonical_graph["adjacency"][p_index][n1_canonical_index]:
                logger.debug("{} -> {}".format(n1_canonical_index, n2_canonical_index))

    logger.debug("------ CANONICAL ADJACENCY END -------")

    logger.debug("------- CANONICAL NODES BEGIN --------")
    for n_canonical_index in range(0, len(canonical_graph["canonical_to_rdf_index"])):
        logger.debug("{} <-> {}".format(n_canonical_index,
                                        [
                                            cache_managers["rdf_nodes"].get_element_from_index(n_index)
                                            for n_index in canonical_graph["canonical_to_rdf_index"][n_canonical_index]
                                        ]))
    logger.debug("-------- CANONICAL NODES END ---------")

    logger.debug("---------- DEBUG OUTPUT END ----------")


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (triplestore, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--max-rows", dest="max_rows", help="Value of the parameter ResultSetMaxRows in virtuoso.ini",
                        required=True, type=int, default=10000)
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    parser.add_argument("--self-signed-ssl", dest="self_signed_ssl",
                        help="Enable querying a server with self-signed SSL certificate", action="store_true",
                        default=False)
    parser.add_argument("--debug", dest="debug", help="Enable debug output", action="store_true")
    args = parser.parse_args()

    # Logging parameters
    logger = get_logger(args.debug)

    # Configuration parameters
    configuration_parameters = load_configuration_parameters(args.conf_file_path)

    # Server manager
    server_manager = ServerManager(configuration_parameters, args.max_rows, args.self_signed_ssl)

    # CacheManagers
    cache_managers = {
        "predicates": CacheManager(),  # CacheManager for predicates
        "rdf_nodes": CacheManager()  # CacheManager for RDF nodes
    }

    # Get RDF graph adjacency
    logger.info("Get RDF graph adjacency")
    rdf_graph_adjacency = get_rdf_graph_adjacency(server_manager, cache_managers, logger)

    logger.info("Compute statistics")
    rdf_edges_before_completion = get_number_of_edges(rdf_graph_adjacency)

    # Complete RDF graph adjacency with owl:sameAs symmetry and transitivity
    logger.info("Complete RDF graph adjacency with owl:sameAs symmetry and transitivity")
    rdf_graph_adjacency = same_as_adjacency_completion(rdf_graph_adjacency, cache_managers["predicates"])

    # Canonicalize RDF graph by transforming nodes in owl:sameAs connected components into one node
    logger.info("Get canonical RDF graph")
    canonical_graph = get_canonical_graph(rdf_graph_adjacency, cache_managers["predicates"])

    # Saving data
    logger.info("Saving data")

    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    pickle.dump(canonical_graph["rdf_to_canonical_index"], open(output_dir + "rdf_to_canonical_index", "wb"))
    pickle.dump(canonical_graph["canonical_to_rdf_index"], open(output_dir + "canonical_to_rdf_index", "wb"))
    pickle.dump(canonical_graph["adjacency"], open(output_dir + "canonical_graph_adjacency", "wb"))
    pickle.dump(canonical_graph["inv_adjacency"], open(output_dir + "canonical_graph_inv_adjacency", "wb"))

    cache_managers["rdf_nodes"].save_to_csv(output_dir + "rdf_nodes_cache_manager.csv")
    cache_managers["predicates"].save_to_csv(output_dir + "predicates_cache_manager.csv")

    # Computing and saving statistics
    logger.info("Compute and save final statistics on RDF graph and canonical graph")
    with open(output_dir + "graphs_statistics.md", "w") as file:
        file.write("# Graphs statistics\n")
        file.write("Generated on {}\n".format(datetime.datetime.now()))

        file.write("## RDF graph\n")
        file.write("* Number of nodes: {}\n".format(len(canonical_graph["rdf_to_canonical_index"])))
        file.write("* Number of edges (before sameAs completion): {}\n".format(rdf_edges_before_completion))
        file.write("* Number of edges (after sameAs completion): {}\n".format(get_number_of_edges(rdf_graph_adjacency)))
        file.write("* Number of types of edges: {}\n".format(len(rdf_graph_adjacency)))

        file.write("## Canonical graph\n")
        file.write("* Number of nodes: {}\n".format(len(canonical_graph["canonical_to_rdf_index"])))
        file.write("* Number of edges: {}\n".format(get_number_of_edges(canonical_graph["adjacency"])))
        file.write("* Number of types of edges: {}\n".format(len(canonical_graph["adjacency"])))

    logger.info("Done")

    if args.debug:
        print_debug_output(canonical_graph, cache_managers, logger)


if __name__ == '__main__':
    main()
