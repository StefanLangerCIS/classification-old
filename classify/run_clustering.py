"""
Evaluate any of the classifier, print a confusion matrix and create further evalution metrics
"""
import argparse
from datetime import datetime
import json
import os
from sklearn_clustering import SklearnClustering

import time

def configure_filters(filter_args_string: str) -> dict:
    """
    Configure the filters from the runtime argument string
    :param filter_args_string: The args string
    :return: Filters as dictionary
    """
    filters = {}
    if filter_args_string is not None and ':' in filter_args_string:
        filter_strings = filter_args_string.split(',')
        for filter_string in filter_strings:
            (field, value) = filter_string.split(':')
            filters[field] = value
    return filters


def filter_match(record: dict, filters: dict)-> bool:
    """
    Check whether filters match for data record
    :param record: record to check
    :param filters: filter(s) to match
    :return: True is all filters pass, else False
    """
    # Each filter must match
    for field in filters:
        if not field in record:
            return False
        if record[field] != filters[field]:
            return False
    # All filters passed (or no filters were present)
    return True

def main():
    parser = argparse.ArgumentParser(description='Evaluate one or several text classifiers')

    # All available clustering algorithm types
    algorithm_types = SklearnClustering.supported_algos

    parser.add_argument('--input',
                    default= r"D:\ProjectData\Uni\ltrs\classifier\clustering_data_full_letters.json",
                    help='Data for clustering')

    parser.add_argument('--output',
                        default= r"D:\ProjectData\Uni\clustering-results",
                        help='Folder where to write the clustering evaluation results')

    parser.add_argument('--algorithm',
                    choices=algorithm_types + ["all"],
                    #default="KMeans",
                    default="AgglomerativeClustering_ward",
                    help="The clustering algorithm to use. If 'all' iterate through all available algorithms" )

    parser.add_argument('--n_clusters',
                    type=int,
                    default="5",
                        help="The clustering algorithm to use. If 'all' iterate through all available algorithms" )

    parser.add_argument('--text_label',
                    default="text",
                    help='Label/field in the json data contains the text to cluster')

    parser.add_argument('--filter',
                    default="lang:de",
                    help='Label/field in the json data contains the text to cluster')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Provide verbose output')

    args = parser.parse_args()

    # Run all clustering algorithms
    if args.algorithm == "all":
        algorithms = algorithm_types
    else:
        algorithms = [args.algorithm]

    print("INFO: Running algorithm(s) {0}".format(algorithms))

    filters = configure_filters(args.filter)

    # read data
    records = []
    with open(args.input, encoding='utf-8') as input_fp:
        for line in input_fp:
            record = json.loads(line)
            if filter_match(record, filters):
                records.append(record)

    # Iterate over the clustering algorithms

    for algorithm in algorithms:
        clustering = SklearnClustering(algorithm, args.n_clusters)
        clustering.verbose = args.verbose
        print("INFO: Running clustering with algorithm {0} for {1} records".format(algorithm, len(records)))
        clustering_time = time.time()
        clustering_results = clustering.cluster(records, args.text_label)
        clustering_time = int(time.time()-clustering_time)
        print("INFO: Clustering completed for algorithm {0} in {1} seconds".format(algorithm, clustering_time))

        outfile_name = os.path.join(args.output, "results_{0}.txt".format(algorithm))
        print("INFO: Writing results for algorithm {0} to file {1}".format(algorithm, outfile_name))
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write("Clustering algorithm: {0}\n".format(algorithm))
            outfile.write("\n#Counts:\n")
            outfile.write("Number of data_records: {0}\n".format(len(records)))
            for cluster in clustering_results:
                outfile.write("###\n")
                outfile.write("Cluster: {0}, n_records: {1}, desc {2}\n".format(cluster.cluster_id, len(cluster.data_points),cluster.cluster_decription))
                for record in cluster.data_points[0:30]:
                    outfile.write("{0}\n".format(json.dumps(record, ensure_ascii=False)))
                outfile.write("###\n")
            outfile.write("\n#Performance:\n")
            outfile.write("Seconds used for clustering: {0}\n".format(clustering_time))

        # Also store confusion matrix as image
        imagefile_name = os.path.join(args.output, "results_{0}.jpg".format(algorithm))


if __name__ == "__main__":
    main()