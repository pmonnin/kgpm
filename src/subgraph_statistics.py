import argparse
import csv

import tqdm

from graph_utils import add_thing_class_type_subclassof, get_nodes_degree, get_nodes_ancestors, get_nodes_full_type, \
    is_blacklisted, canonical_index_to_uris, is_node_blacklisted
from load_utils import get_logger, get_blacklists, load_configuration_parameters, load_graph, load_cache_managers, \
    load_dataset

__author__ = "Pierre Monnin"


def is_type_blacklisted(t_index, canonical_to_rdf_index, rdf_nodes_cache_manager, types_blacklist):
    """
    Test if a type is blacklisted w.r.t. the given types blacklist
    :param t_index: canonical index of the type to test
    :param canonical_to_rdf_index: mapping from canonical indices to RDF indices
    :param rdf_nodes_cache_manager: cache manager for RDF nodes
    :param types_blacklist: blacklist for types
    :return: True if t_index represents a blacklisted type, False otherwise
    """

    t_uris = canonical_index_to_uris(t_index, canonical_to_rdf_index, rdf_nodes_cache_manager)

    return any(is_blacklisted(t_uri, types_blacklist) for t_uri in t_uris)


def get_types_ancestors_per_level(canonical_graph, cache_managers, types_blacklist):
    """
    Compute (not blacklisted) ancestors at each level t for all (not blacklisted) classes in canonical graph.
    All classes (except owl:Thing) have at least owl:Thing as their ancestor at each level (to be consistent with
    the behavior for owl:Thing in extract_features.py).
    Nodes that are not classes are not in the returned dict
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param types_blacklist: types blacklist
    :return: dict associating each (not blacklisted) canonical node index representing a class with the set of
    indices of its (not blacklisted) ancestors at each level t
    """

    # Get rdf:type and rdfs:subClassOf predicate indices
    type_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    subclass_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/2000/01/rdf-schema#subClassOf")

    # Get owl:Thing canonical index
    top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
    top_canonical_index = canonical_graph["rdf_to_canonical_index"][top_rdf_index]

    # Compute blacklisted types
    blacklisted_types = set()
    for current_n in tqdm.tqdm(range(0, len(canonical_graph["canonical_to_rdf_index"])), desc="blacklisted types"):
        # If n is subclass of another node, or superclass of another node or type of another node: it is a type
        if current_n in canonical_graph["adjacency"][subclass_p_index] or \
                current_n in canonical_graph["inv_adjacency"][subclass_p_index] or \
                current_n in canonical_graph["inv_adjacency"][type_p_index]:
            if is_type_blacklisted(current_n, canonical_graph["canonical_to_rdf_index"], cache_managers["rdf_nodes"],
                                   types_blacklist):
                blacklisted_types.add(current_n)

    # Compute non-blacklisted ancestors for non-blacklisted types
    ancestors = dict()
    for current_n in tqdm.tqdm(range(0, len(canonical_graph["canonical_to_rdf_index"])), desc="ancestors"):

        # If n is subclass of another node, or superclass of another node or type of another node: it can have ancestors
        if current_n in canonical_graph["adjacency"][subclass_p_index] or \
                current_n in canonical_graph["inv_adjacency"][subclass_p_index] or \
                current_n in canonical_graph["inv_adjacency"][type_p_index]:

            # current_n is not owl:Thing (which does not have ancestors) and current_n is a non-blacklisted type
            if current_n != top_canonical_index and current_n not in blacklisted_types:
                ancestors[current_n] = dict()

                # t = 1
                t = 1
                ancestors[current_n][t] = set()
                if current_n in canonical_graph["adjacency"][subclass_p_index]:
                    ancestors[current_n][t] = set(canonical_graph["adjacency"][subclass_p_index][current_n])
                    ancestors[current_n][t] -= blacklisted_types
                ancestors[current_n][t].add(top_canonical_index)

                # t = 2 (and next...)
                to_expand = set(ancestors[current_n][t] - {top_canonical_index})
                t += 1
                while len(to_expand) != 0:
                    ancestors[current_n][t] = set(ancestors[current_n][t - 1])

                    for t_index in to_expand:
                        if t_index in canonical_graph["adjacency"][subclass_p_index]:
                            ancestors[current_n][t] |= set(canonical_graph["adjacency"][subclass_p_index][t_index])

                    ancestors[current_n][t] -= blacklisted_types
                    ancestors[current_n][t].add(top_canonical_index)

                    to_expand = ancestors[current_n][t] - ancestors[current_n][t - 1]
                    t += 1

    return ancestors


def compute_full_neighborhoord(seed_nodes, canonical_graph, cache_managers, blacklists, d, undirected):
    """
    Compute estimated full neighborhood of given seed nodes w.r.t. blacklists, d and undirected parameters
    :param seed_nodes: seed nodes whose neighborhood is computed
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param blacklists: blacklists (used to control neighborhood expansion)
    :param d: Max degree (included) to allow expansion of nodes (-1 to disable)
    :param undirected: If the graph is considered as undirected
    :return: the set of neighborhing nodes for the given seed nodes (excluding themselves, that's why it is
    an estimated neighborhood)
    """

    neighborhood = set(seed_nodes)
    to_expand = set(seed_nodes)

    while len(to_expand) != 0:
        new_neighborhood = set(neighborhood)

        for n in tqdm.tqdm(to_expand, desc="n. exp."):
            if d == -1 or canonical_graph["nodes_degree"][n] <= d:  # Check degree constraint
                if n in canonical_graph["uniq_adjacency"]:
                    for n2 in canonical_graph["uniq_adjacency"][n]:
                        if not is_node_blacklisted(n2, canonical_graph, cache_managers, blacklists):
                            new_neighborhood.add(n2)

                if undirected:  # If canonical graph is considered as undirected
                    if n in canonical_graph["uniq_inv_adjacency"]:
                        for n2 in canonical_graph["uniq_inv_adjacency"][n]:
                            if not is_node_blacklisted(n2, canonical_graph, cache_managers, blacklists):
                                new_neighborhood.add(n2)

        to_expand = set(new_neighborhood - neighborhood)
        neighborhood = set(new_neighborhood)

    return neighborhood - seed_nodes


def compute_full_type_neighborhood(seed_nodes, canonical_graph, cache_managers, blacklists):
    """
    Compute the set of all types associated with the given seed nodes
    :param seed_nodes: seed nodes whose types are computed
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param blacklists: blacklists
    :return: the set of all types associated with the given seed nodes. Each returned type is associated either
    directly or indirectly with at least one of the seed nodes
    """

    type_neighborhood = set()

    for n in tqdm.tqdm(seed_nodes, desc="n. type exp."):
        if n in canonical_graph["nodes_full_type"]:
            type_neighborhood |= set(canonical_graph["nodes_full_type"][n])

    type_neighborhood = {
        t_index for t_index in tqdm.tqdm(type_neighborhood, desc="t. blacklist")
        if not is_type_blacklisted(t_index, canonical_graph["canonical_to_rdf_index"], cache_managers["rdf_nodes"],
                                   blacklists["types"])
    }

    # Ensure owl:Thing is always in the type neighborhood
    top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
    top_canonical_index = canonical_graph["rdf_to_canonical_index"][top_rdf_index]
    type_neighborhood.add(top_canonical_index)

    return type_neighborhood


def compute_node_neighborhoods(seed_node, canonical_graph, cache_managers, blacklists, d, undirected, logger):
    """
    Compute neighboring nodes at each level k for the given seed node (until max is reached)
    :param seed_node: node whose neighborhood is computed
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param blacklists: blacklists
    :param d: maximal degree (included) to allow the expansion of a node (-1 to disable)
    :param undirected: if the graph is considered directed or undirected
    :param logger: logger
    :return: a dict associating each level k (int) with the corresponding neighboring nodes of the given seed_node
    """

    neighborhoods = dict()
    to_expand = {seed_node}

    k = 1
    while len(to_expand) != 0:
        # Initialize current neighborhood with previous one
        if k == 1:
            neighborhoods[k] = set()
        else:
            neighborhoods[k] = set(neighborhoods[k - 1])

        # Expansion
        logger.info("k = {}, # nodes to expand = {}".format(k, len(to_expand)))
        for n in tqdm.tqdm(to_expand, desc="k={}".format(k)):
            if d == -1 or canonical_graph["nodes_degree"][n] <= d:  # Check degree constraint
                if n in canonical_graph["uniq_adjacency"]:
                    for n2 in canonical_graph["uniq_adjacency"][n]:
                        if not is_node_blacklisted(n2, canonical_graph, cache_managers, blacklists) and n2 != seed_node:
                            neighborhoods[k].add(n2)

                if undirected:  # If graph is considered undirected
                    if n in canonical_graph["uniq_inv_adjacency"]:
                        for n2 in canonical_graph["uniq_inv_adjacency"][n]:
                            if not is_node_blacklisted(n2, canonical_graph, cache_managers,
                                                       blacklists) and n2 != seed_node:
                                neighborhoods[k].add(n2)

        if k == 1:
            to_expand = set(neighborhoods[k])
        else:
            to_expand = set(neighborhoods[k] - neighborhoods[k - 1])

        k += 1

    return neighborhoods


def compute_subgraph_detailed_statistics(seed_nodes, canonical_graph, cache_managers, d, undirected, blacklists,
                                         logger):
    """
    Compute detailed statistics for the subgraph made from the neighboring nodes of seed nodes:
    - number of neighboring nodes accessible at each k (until max is reached)
    - number of types associated at each t with each level k of neighborhood (until max is reached)
    :param seed_nodes: seed nodes whose neighboring subgraph is studied
    :param canonical_graph: canonical graph
    :param cache_managers: cache managers
    :param d: maximal degree (included) to allow the expansion of a node (-1 to disable)
    :param undirected: if the graph is considered directed or undirected
    :param blacklists: blacklists
    :param logger: logger
    :return: a dict with statistics associated with each level k of neighborhood
    """

    subgraph_statistics = dict()

    # Neighborhood expansion for each seed node
    logger.info("Neighborhood expansion for each seed node")
    neighborhoods = dict()

    # Copy neighborhoods from each seed node
    for s in tqdm.tqdm(seed_nodes, desc="seed nodes"):
        s_neighborhoods = compute_node_neighborhoods(s, canonical_graph, cache_managers, blacklists, d, undirected,
                                                     logger)

        for k in s_neighborhoods:
            if k not in neighborhoods:
                neighborhoods[k] = set(s_neighborhoods[k])
            else:
                neighborhoods[k] |= set(s_neighborhoods[k])

    # Ensure each level of neighborhood is based on the previous level
    for k in range(2, max(neighborhoods.keys()) + 1):
        neighborhoods[k] |= neighborhoods[k - 1]

    # Add number of nodes per level in statistics and compute number of types at each level t
    type_p_index = cache_managers["predicates"].get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    top_rdf_index = cache_managers["rdf_nodes"].get_element_index("http://www.w3.org/2002/07/owl#Thing")
    top_canonical_index = canonical_graph["rdf_to_canonical_index"][top_rdf_index]

    for k in tqdm.trange(1, max(neighborhoods.keys()) + 1):
        subgraph_statistics[k] = {"# nodes": len(neighborhoods[k]), "# types": dict()}

        logger.info("Types expansion for neighborhood at k = {}".format(k))
        type_neighborhood = dict()
        type_neighborhood[1] = {top_canonical_index}

        # t = 1 is the type adjacency (minus blacklisted types)
        for n in tqdm.tqdm(neighborhoods[k], desc="t=1"):
            if n in canonical_graph["adjacency"][type_p_index]:
                for t_index in canonical_graph["adjacency"][type_p_index][n]:
                    if not is_type_blacklisted(t_index, canonical_graph["canonical_to_rdf_index"],
                                               cache_managers["rdf_nodes"], blacklists["types"]):
                        type_neighborhood[1].add(t_index)

        # t = 2 and next is the copy and shift + 1 of ancestors of types at t = 1
        for t_index in tqdm.tqdm(type_neighborhood[1], desc="t. exp."):
            if t_index in canonical_graph["types_ancestors_per_level"]:
                for t, ancestors in canonical_graph["types_ancestors_per_level"][t_index].items():
                    if t + 1 not in type_neighborhood:
                        type_neighborhood[t + 1] = set()
                    type_neighborhood[t + 1] |= ancestors

        # Ensure each level of types is based on the previous one
        for t in range(2, max(type_neighborhood.keys()) + 1):
            type_neighborhood[t] |= type_neighborhood[t - 1]

        # Add # types in statistics
        for t in range(1, max(type_neighborhood.keys()) + 1):
            subgraph_statistics[k]["# types"][t] = len(type_neighborhood[t])

    return subgraph_statistics


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (blacklists, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--graph", dest="graph_dir", help="Base directory for input graph files", required=True)
    parser.add_argument("--dataset-csv", dest="dataset_file", help="CSV file with the seed nodes URIs", required=True)
    parser.add_argument("--dataset-name", dest="dataset_name", help="Name of the data set", required=True)
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    parser.add_argument("-d", dest="d", help="Max degree (included) to allow expansion of nodes (-1 to disable)",
                        default=-1, type=int)
    parser.add_argument("--undirected", dest="undirected", help="If the graph is considered as undirected",
                        default=False, action="store_true")
    parser.add_argument("--detailed", dest="detailed", help="Output the repartition of # nodes and types w.r.t. k / t",
                        default=False, action="store_true")
    args = parser.parse_args()

    # Logging parameters
    logger = get_logger(False)

    # Configuration parameters
    logger.info("Load configuration")
    configuration_parameters = load_configuration_parameters(args.conf_file_path)

    # Blacklists
    blacklists = get_blacklists(configuration_parameters)

    # Load graph
    logger.info("Load graph from " + args.graph_dir)
    canonical_graph = load_graph(args.graph_dir)

    # Load cache managers
    logger.info("Load cache managers from " + args.graph_dir)
    cache_managers = load_cache_managers(args.graph_dir)

    # Load dataset
    logger.info("Load dataset from " + args.dataset_file)
    classes = load_dataset(args.dataset_file)

    # Configuring output
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"
    logger.info("Output configured in " + output_dir)

    # Ensure owl:Thing, owl:Class, rdf:type and rdfs:subClassOf are in canonical graph
    canonical_graph = add_thing_class_type_subclassof(canonical_graph, cache_managers)

    # Compute node degree
    logger.info("Compute degree for nodes")
    canonical_graph["nodes_degree"] = get_nodes_degree(
        canonical_graph,
        cache_managers["predicates"],
        blacklists["path_predicates"],
        args.undirected
    )

    # Compute ancestors
    logger.info("Compute ancestors of classes (transitive closure of rdfs:subClassOf)")
    canonical_graph["nodes_ancestors"] = get_nodes_ancestors(canonical_graph, cache_managers)

    # Compute full type
    logger.info("Compute full type of nodes (transitive closure of rdf:type and rdfs:subClassOf)")
    canonical_graph["nodes_full_type"] = get_nodes_full_type(canonical_graph, cache_managers)

    # Compute unique adjacency and inverse unique adjacency for neighborhood expansion
    logger.info("Compute unique adjacency and inverse unique adjacency for neighborhood expansion")

    canonical_graph["uniq_adjacency"] = dict()
    for p_index in tqdm.tqdm(canonical_graph["adjacency"]):
        if not is_blacklisted(cache_managers["predicates"].get_element_from_index(p_index),
                              blacklists["path_predicates"]):
            for n in canonical_graph["adjacency"][p_index]:
                if n not in canonical_graph["uniq_adjacency"]:
                    canonical_graph["uniq_adjacency"][n] = set()

                canonical_graph["uniq_adjacency"][n] |= canonical_graph["adjacency"][p_index][n]

    canonical_graph["uniq_inv_adjacency"] = dict()
    for p_index in tqdm.tqdm(canonical_graph["inv_adjacency"]):
        if not is_blacklisted(cache_managers["predicates"].get_element_from_index(p_index),
                              blacklists["path_predicates"]):
            for n in canonical_graph["inv_adjacency"][p_index]:
                if n not in canonical_graph["uniq_inv_adjacency"]:
                    canonical_graph["uniq_inv_adjacency"][n] = set()

                canonical_graph["uniq_inv_adjacency"][n] |= canonical_graph["inv_adjacency"][p_index][n]

    # From seed URIs to canonical node indices
    seed_nodes = set()
    for seed_uri in tqdm.tqdm(classes.keys(), desc="URIs to canonical nodes"):
        seed_rdf_index = cache_managers["rdf_nodes"].get_element_index(seed_uri)

        if seed_rdf_index in canonical_graph["rdf_to_canonical_index"]:
            seed_nodes.add(canonical_graph["rdf_to_canonical_index"][seed_rdf_index])

    # Compute full neighborhood
    logger.info("Compute full neighborhood")
    neighborhood = compute_full_neighborhoord(seed_nodes, canonical_graph, cache_managers, blacklists, args.d,
                                              args.undirected)
    logger.info("Compute full type neighborhood")
    type_neighborhood = compute_full_type_neighborhood(neighborhood, canonical_graph, cache_managers, blacklists)

    with open(output_dir + "{}_subgraph_statistics_d{}_undirected{}.md".format(args.dataset_name, args.d,
                                                                                    args.undirected), "w") as f_output:
        f_output.write("* # nodes (estimated): {} (+/- {})\n".format(len(neighborhood), len(seed_nodes)))
        f_output.write("* # types (estimated): {}\n".format(len(type_neighborhood)))

    if args.detailed:
        # Compute each level of ancestors for each class
        logger.info("Compute levels of ancestors for classes")
        canonical_graph["types_ancestors_per_level"] = get_types_ancestors_per_level(canonical_graph, cache_managers,
                                                                                     blacklists["types"])

        # Compute subgraph statistics
        logger.info("Compute subgraph detailed statistics")
        subgraph_statistics = compute_subgraph_detailed_statistics(seed_nodes, canonical_graph, cache_managers, args.d,
                                                                   args.undirected, blacklists, logger)

        # Detect max t
        max_t = max({t for k in subgraph_statistics for t in subgraph_statistics[k]["# types"]})
        logger.info("max k = {}, max t = {}".format(max(subgraph_statistics.keys()), max_t))

        # Saving subgraph detailed statistics
        logger.info("Saving subgraph detailed statistics")
        with open(output_dir +
                  "{}_subgraph_detailed_statistics_d{}_undirected{}.csv".format(args.dataset_name, args.d,
                                                                                args.undirected), "w") as csvfile:
            writer = csv.writer(csvfile)

            # Prepare headers
            headers = ["k", "# nodes"]
            for t in range(1, max_t + 1):
                headers.append("# types at t={}".format(t))
            writer.writerow(headers)

            # One line per k
            for k in sorted(subgraph_statistics.keys()):
                row = [k, subgraph_statistics[k]["# nodes"]]

                for t in range(1, max_t + 1):
                    if t in subgraph_statistics[k]["# types"]:
                        row.append(subgraph_statistics[k]["# types"][t])
                    else:
                        row.append("")

                writer.writerow(row)


if __name__ == '__main__':
    main()
