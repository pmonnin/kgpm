# kgpm

Mining neighbors, paths, and path patterns from a Knowledge Graph and a set of seed nodes

## ``query_graph.py``

Python script to query a knowledge graph and perform its canonicalization.
The script outputs:
* Files representing the canonical knowledge graph (in ``rdf_to_canonical_index``,
    ``canonical_to_rdf_index``, ``canonical_graph_adjacency``,
    ``canonical_graph_inv_adjacency``, ``rdf_nodes_cache_manager.csv``,
    ``predicates_cache_manager.csv``)
* Statistics about the knowledge graph before and after canonicalization (in ``graphs_statistics.md``)

Parameters:
* ``--configuration``: path of the JSON configuration file
* ``--max-rows``: max number of rows the SPARQL endpoint can return
* ``--output``: base directory for output files
* ``--self-signed-ssl``: enable self signed SSL certificates
* ``--debug``: print debug statements

## ``extract_features.py``

Python script to mine neighbors, paths, and path patterns from a canonical knowledge graph and a set of seed nodes.

Parameters:
* ``--configuration``: path of the JSON configuration file
* ``--graph``: base directory for the input graph files
* ``--dataset-csv``: CSV file with the seed nodes URIs (column 0) and class labels (column 1)
* ``--dataset-name``: name of the data set
* ``--output``: base directory for output files (statistics, scipy matrice of nodes x features, column name file, and a numpy vector of class labels)
* ``-d``: maximum degree to allow expansion (disabled with ``d = -1``)
* ``--lmin``: minimum support for features
* ``--lmax``: maximum support for features
* ``--kmin``: minimum k to test (i.e., number of traversed edges, size of paths and path patterns)
* ``--kmax``: maximum k to test
* ``--tmin``: minimum t to test (i.e., level for generalization in class hierarchies); ``t = -1`` disables type generalization, ``t = 0`` only allows to generalize with ``owl:Thing``
* ``--tmax``: maximum t to test
* ``--undirected``: whether only out arcs (``false``) or all arcs (``true``) are traversed
* ``--meaningful``: biomedical additional filtering strategies:
  * ``p``: only select features containing a pathway
  * ``g``: only select features containing a gene or a GO class
  * ``m``: only select features containing a MeSH class
  * ``pg``: disjunction of ``p`` and ``g``
  * ``pgm``: disjunction of ``p``, ``g``, and ``m``
  * ``all``: test all previous filters (thus, 5 outputs)
  * ``no_check``: disable the additional filtering
* ``--debug``: print debug statements

## ``subgraph_statistics.py``

Python script to compute the statistics about the subgraph accessible from a set of seed nodes in a canonical knowledge graph.
It outputs a markdown file containing the number of neighbors and types reachable from the seed nodes.

Parameters:
* ``--configuration``: path of the JSON configuration file
* ``--graph``: base directory for the input graph files
* ``--dataset-csv``: CSV file with the seed nodes URIs (column 0) and class labels (column 1)
* ``--dataset-name``: name of the data set
* ``--output``: base directory for output files (Markdown files)
* ``-d``: maximum degree to allow expansion (disabled with ``d = -1``)
* ``--undirected``: whether only out arcs (``false``) or all arcs (``true``) are traversed
* ``--detailed``: enable detailed statistics, i.e., number of neighbors and types accessible w.r.t. k and t until full neighborhood is reached. By default, only the max numbers of reachable neighbors and types in the full neighborhood are output (k and t are not given).

## Configuration

An example of a JSON configuration file is given in [configuration.json.example](configuration.json.example).
Keys are:

* _server-address_: address of the SPARQL endpoint to query
* _url-json-conf-attribute_: URL attribute to use to get JSON results
* _url-json-conf-value_: value of the _url-json-conf-attribute_ to get JSON results
* _url-default-graph-attribute_: URL attribute to use to define the default graph
* _url-default-graph-value_: value of _url-default-graph-attribute_ to define the default graph
* _url-query-attribute_: URL attribute to use to define the query
* _timeout_: timeout value for HTTP requests
* _username_: username to use if HTTP authentication is required (empty otherwise)
* _password_: password to use if HTTP authentication is required (empty otherwise)
* _path_predicates_blacklist_: blacklist of URIs or prefixes of predicates not to traverse
* _types_blacklist_: blacklist of URIs or prefixes of types not to use in path generalization
* _types_expansion_blacklist_: blacklist of URIs or prefixes of types whose instances cannot be traversed

## Dependencies

* tqdm
* numpy
* bitarray
* scipy
