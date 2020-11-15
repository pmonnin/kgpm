import json
import logging
import socket
import sys

import requests
import requests.auth
import tqdm


class ServerManager:
    """
    Class handling queries to the triplestore
    """

    def __init__(self, configuration_parameters, max_rows, self_signed_ssl):
        """
        Configure the ServerManager object
        :param configuration_parameters: all configuration parameters loaded from the JSON configuration file
        :param max_rows: max number of rows to use when querying server
        :param self_signed_ssl: enable querying a server with self-signed SSL certificate
        """

        self.server_address = configuration_parameters["server-address"]
        self.json_conf_attribute = configuration_parameters["url-json-conf-attribute"]
        self.json_conf_value = configuration_parameters["url-json-conf-value"]
        self.default_graph_attribute = configuration_parameters["url-default-graph-attribute"]
        self.default_graph_value = configuration_parameters["url-default-graph-value"]
        self.query_attribute = configuration_parameters["url-query-attribute"]
        self.self_signed_ssl = self_signed_ssl
        socket.setdefaulttimeout(configuration_parameters["timeout"])

        if "username" in configuration_parameters:
            self.username = configuration_parameters["username"]
            self.password = configuration_parameters["password"]
        else:
            self.username = ""
            self.password = ""

        self.max_rows = max_rows
        self.prefixes = "PREFIX pgxo:<http://pgxo.loria.fr/> " + \
                        "PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#> " + \
                        "PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#> " + \
                        "PREFIX owl:<http://www.w3.org/2002/07/owl#> " + \
                        "PREFIX obo:<http://purl.obolibrary.org/obo/> "
        self._logger = logging.getLogger()

    def query_server(self, query):
        """
        Query the server with the given SPARQL query. This function DOES NOT handle pagination
        with LIMIT and OFFSET depending on the parameters max_rows
        :param query: a SPARQL query to execute
        :return: JSON content returned by server parsed as a dict
        """

        done = False

        content = {}

        while not done:
            done = True

            query_parameters = {
                self.json_conf_attribute: self.json_conf_value,
                self.default_graph_attribute: self.default_graph_value,
                self.query_attribute: query
            }

            if self.username != "" and self.password != "":
                content = requests.get(self.server_address, query_parameters,
                                       auth=requests.auth.HTTPBasicAuth(self.username, self.password),
                                       verify=not self.self_signed_ssl)
            else:
                content = requests.get(self.server_address, query_parameters, verify=not self.self_signed_ssl)

            if content.status_code == 404:
                done = False
                self._logger.critical("404 error. New try.")

            elif content.status_code != 200:
                self._logger.critical(content.content)
                sys.exit(-1)

        return json.loads(content.text)

    def query_count_elements(self, where_clause):
        """
        Query the server with a SPARQL query counting number of elements ?e validating the given WHERE clause
        :param where_clause: WHERE clause to use in a query select count(distinct ?e) as ?count where { ... }
        :return: Number of elements ?e satisfying the given WHERE clause
        """

        results_json = self.query_server(self.prefixes + " select count(distinct ?e) as ?count where { " +
                                         where_clause + " }")

        return int(results_json["results"]["bindings"][0]["count"]["value"])

    def query_count_two_elements(self, where_clause):
        """
        Query the server with a SPARQL query counting number of elements ?e1 ?e2 validating the given WHERE clause
        :param where_clause: WHERE clause to use in a query
        select count(*) as ?count where { select distinct ?e1 ?e2 where { ... } }
        :return: Number of elements ?e1 ?e2 satisfying the given WHERE clause
        """

        results_json = self.query_server(self.prefixes + " select count(*) as ?count where { "
                                         "select distinct ?e1 ?e2 where {" + where_clause + " } }")
        return int(results_json["results"]["bindings"][0]["count"]["value"])

    def query_elements(self, where_clause, verbose=False):
        """
        Query the server to select elements ?e validating the given WHERE clause. This function handles pagination
        with LIMIT and OFFSET depending on the parameter max_rows
        :param where_clause: WHERE clause to use in a query select distinct ?e where { ... }
        :param verbose: activate tqdm progress bar while querying different pages
        :return: a list of elements ?e satisfying the given WHERE clause
        """

        ret_val = []
        elements_count = self.query_count_elements(where_clause)

        if verbose and elements_count != 0:
            pbar = tqdm.tqdm(total=elements_count)

        while len(ret_val) != elements_count:
            ret_val = []
            offset = 0
            while offset <= elements_count:
                results_json = self.query_server(self.prefixes + " select distinct ?e where { " + where_clause
                                                 + " } LIMIT " + str(self.max_rows) + " OFFSET " + str(offset))

                for result in results_json["results"]["bindings"]:
                    ret_val.append(str(result["e"]["value"]))

                    if verbose:
                        pbar.update(1)

                offset += self.max_rows

            if len(ret_val) != elements_count:
                self._logger.critical("Number of elements different from count, retry...")

                if verbose:
                    pbar.close()
                    pbar = tqdm.tqdm(total=elements_count)

        if verbose and elements_count != 0:
            pbar.close()

        return ret_val

    def query_two_elements(self, where_clause, verbose=False):
        """
        Query the server to select elements ?e1 ?e2 validating the given WHERE clause. This function handles
        pagination with LIMIT and OFFSET depending on the parameter max_rows
        :param where_clause: WHERE clause to use in a query select distinct ?e1 ?e2 where { ... }
        :param verbose: activate tqdm progress bar while querying different pages
        :return: a list of tuples (?e1, ?e2) satisfying the given WHERE clause
        """

        ret_val = []
        elements_count = self.query_count_two_elements(where_clause)

        if verbose and elements_count != 0:
            pbar = tqdm.tqdm(total=elements_count)

        while len(ret_val) != elements_count:
            ret_val = []
            offset = 0

            while offset <= elements_count:
                results_json = self.query_server(self.prefixes + " select distinct ?e1 ?e2 where { " + where_clause +
                                                 " } LIMIT " + str(self.max_rows) + " OFFSET " + str(offset))

                for result in results_json["results"]["bindings"]:
                    ret_val.append((str(result["e1"]["value"]), str(result["e2"]["value"])))

                    if verbose:
                        pbar.update(1)

                offset += self.max_rows

            if len(ret_val) != elements_count:
                self._logger.critical("Number of elements different from count, retry...")

                if verbose:
                    pbar.close()
                    pbar = tqdm.tqdm(total=elements_count)

        if verbose and elements_count != 0:
            pbar.close()

        return ret_val
