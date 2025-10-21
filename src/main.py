import pathlib
import logging
import sys
import os
import csv
import json
import numpy as np
import argparse
import yaml
from rdflib import URIRef
from rdflib_hdt import HDTDocument
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import torch

# idx is a global to keep track of point ids for the Qdrant DB
# inserts they need to be unique and not reset when performing
# chunking of data
# if adding data to an existing Qdrant collection,this number
# must be set to the integer point id in the collection - it will
# be incremented by 1 when it is used.
idx = 0

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


# create embeddings and write to a tsv file
def saveToTSV(model, input_file, sentences_to_embed_dict):
    tsv_file = ""
    embedding_list = []
    try:
        # create an embedding for each unique subject
        # and write to the tsv file
        tsv_file = os.path.splitext(os.path.basename(input_file))[0] + ".tsv"
        with open(tsv_file, "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['iri', 'label', 'embedding'])
            for k, v in sentences_to_embed_dict.items():
                embedding = model.encode(v)
                writer.writerow([k, v, np.array2string(embedding, separator=', ').replace('\n', '')])
        f.close()
        logger.info(f"Saved embeddings to {tsv_file}")
    except Exception as e:
        logger.error(f"An error occurred while save embedded data to the tsv file:{tsv_file} {e}")


# create embeddings and write to a json file that is in
# a format suitable for uploading into a vector database
def saveToJSON(model, input_file, sentences_to_embed_dict):
    global idx
    graph_name = os.path.splitext(os.path.basename(input_file))[0]
    json_file = os.path.splitext(os.path.basename(input_file))[0] + ".json"
    dict_list = []

    try:
        # create an embedding for each unique subject
        # and write to the json file
        for k, v in sentences_to_embed_dict.items():
            idx += 1
            embedding = model.encode(v)
            embedded_dict = {"id": idx, "vector": embedding.tolist(), "payload": {"graph": graph_name, "iri": k, "label": v}}
            dict_list.append(embedded_dict)

        with open(json_file, 'w') as f:
            json.dump({'points': dict_list}, f, indent=3)

        f.close()
        logger.info(f"Saved embeddings to {json_file}")
    except Exception as e:
        logger.error(f"An error occurred while saving embedded data to the json file:{json_file} {e}")

# create an embedding for each unique subject
# and write to a qdrant collection
def saveToQdrant(model, url, collection_name, input_file, sentences_to_embed_dict):
    # connect to Qdrant client
    global idx
    try:
        client = QdrantClient(url=url, timeout=60)
        # create graph name to add to point payload
        # use .hdt input file name
        graph_name = os.path.splitext(os.path.basename(input_file))[0]

        if not client.collection_exists(collection_name):
            # collection does not exist - create and populate
            logger.debug(f"Collection '{collection_name}' does not exist - creating.")
            client.create_collection(collection_name=collection_name,
                                     vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        points = []

        for k, v in sentences_to_embed_dict.items():
            idx += 1
            embedding = model.encode(v)
            points.append(
                models.PointStruct(
                    id=idx,  # Unique ID for each point
                    vector=embedding,
                    payload={"graph": graph_name, "iri": k, "label": v}  # Add metadata (payload)
                )
            )
            # batch upserts to avoid timeouts
            if idx % 1000 == 0:
                client.upsert(collection_name=collection_name, points=points)
                points = []

        # final upsert
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Saved embeddings to Qdrant collection '{collection_name}'.")
        logger.info(f"last point id was {points[-1].id}")

    except Exception as e:
        logger.error(f"An error occurred uploading data to Qdrant:{url}: {e}")

def getIriKeyFromValue(iri_types, iri_type_value):

    for iris in iri_types:
        for key, value in iris.items():
            if isinstance(value, list) and iri_type_value in value:
                return key

    return None


# get list of iri types from config file
def getIriTypes(conf_file):
    iri_types = []

    try:
        with open(conf_file, 'r') as conf:
            conf_yaml = yaml.safe_load(conf)
            iri_types = conf_yaml['irisToEmbed']

        conf.close()

    except Exception as e:
        logger.error(f"An error occurred while parsing file {conf_file}: {e}")

    return iri_types

def getAllIriTypeValues(iri_types):
    iri_values = []

    for type in iri_types:
        value_list_obj = type.values()
        value_list = list(value_list_obj)[0]
        iri_values += value_list

    return iri_values


# counts how many strings in the sentence list match the target string
def countDuplicates(string_list, target):

    match_count = sum(1 for s in string_list if s.lower() == target.lower())

    return match_count


# this function takes a list of dicts and creates
# another dict that take the format of:
# <triple subject>: <string - sentence to embed>
# the string to embed looks something like this:
# 'HELLO; also known as  ERS-1_BYU_L3_OW_SIGMA0_ENHANCED; description: WHEEEEE!; subject: HELLO; subject: ERS-1 Gridded Level 3 Enhanced Resolution Sigma-0 from BYU'
# note the labels are a special case and not preceded by a <key>: before the value
def createSentences(embed_list):

    sentence_dict = {}
    sentences_to_embed_dict = {}

    for item in embed_list:
        subject = item.get('subject')
        obj = item.get('object')
        key = item.get('config_key')
        if subject is not None and key is not None:
            if subject not in sentence_dict:
                sentence_dict[subject] = []
            sentence_dict[subject].append({key: obj})

    # now create sentence to embed for each subject (s)
    also_str = "also known as"
    for key, value in sentence_dict.items():
        sentence = ""

        # special case - don't add the key for labels
        # use following for testing
        # val = [{'label': 'HELLO'},{'description': 'WHEEEEE!'},{'subject': 'HELLO'},{'subject': 'ERS-1 Gridded Level 3 Enhanced Resolution Sigma-0 from BYU'},{'label': 'ERS-1_BYU_L3_OW_SIGMA0_ENHANCED'}]
        labels = [d["label"] for d in value if "label" in d]
        if labels is not None and len(labels) > 0:
            sentence = labels[0]
            if len(labels) > 1:
                # only precede with "also known as" if the reminder of
                # the labels in the list are not all duplicates
                if countDuplicates(labels, sentence) != len(labels):
                    sentence += f"; {also_str}"
                    for label in labels[1:]:
                        if label not in sentence:  # issue 9 - only add labels that don't match what is already in the sentence
                            sentence += f" {label}"

        # now append the rest of the predicates
        for value_dict in value:
            for k, v in value_dict.items():
                if k != "label":
                    sentence += f"; {k}: {v}"

        sentences_to_embed_dict[key] = sentence

    return sentences_to_embed_dict


# create embeddings from rdf triples
def main(input_file: pathlib.Path, config_file: pathlib.Path, tsv_output, json_output, qdrant_url, collection_name):
    logger.info(f"input: {input_file}  config: {config_file}")

    doc = HDTDocument(str(input_file), indexed=False)
    logger.info(f"subjects: {doc.nb_subjects}  predicates: {doc.nb_predicates}  objects: {doc.nb_objects}")

    iri_types = getIriTypes(config_file)
    iri_values = getAllIriTypeValues(iri_types)
    sentences_to_embed_dict = {}
    embed_list = []
    model = None

    try:
        # load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("loaded model")
        # Check if CUDA is available
        # print("CUDA available:", torch.cuda.is_available())
        # Check if model parameters are on GPU
        # print("Model is on GPU:", next(model.parameters()).is_cuda)
    except Exception as e:
        logger.error(f"An error occurred while loading the embedding model: {e}")

    # in order to use the least amount of memory, we will process the
    # triples grouped by subject - thus avoiding loading all the triples
    # into memory at once
    # first store all the unique subjects
    all_subjects = set()
    triples, cardinality = doc.search((None, None, None))
    for s, p, o in triples:
        all_subjects.add(s)
    # convert the set to a list for chunking
    subject_list = list(all_subjects)

    # iterate through the subject list in chunks
    subject_chunk_size = 1000
    for i in range(0, len(subject_list), subject_chunk_size):
        try:
            subject_chunk = subject_list[i:i + subject_chunk_size]

            # reset the sentences to embed dict and embed_list for each chunk
            sentences_to_embed_dict = {}
            embed_list = []
            # Process each subject in the current chunk
            for subject in subject_chunk:
                # Get all triples for the current subject
                triples, cardinality = doc.search((subject, None, None))

                # collect all the triples desired - as defined
                # by the predicates listed in the config file
                for s, p, o in triples:
                    if str(p) in iri_values and isinstance(p, URIRef):
                        # print(f"Found a match: {s}  {p}  {o}")
                        key = getIriKeyFromValue(iri_types, str(p))
                        embed_dict = {"config_key": key, "subject": str(s), "predicate": str(p), "object": str(o)}
                        embed_list.append(embed_dict)
        except Exception as e:
            logger.error(f"An error occurred while parsing HDT file {input_file}: {e}")

        try:
            # now create a list of sentences to embed
            sentences_to_embed_dict = createSentences(embed_list)
        except Exception as e:
                logger.error(f"An error occurred creating embedding sentences: {e}")

        # now see if we have data, and how we want to embed and save this data
        try:
            if sentences_to_embed_dict is not None:
                if tsv_output:
                    saveToTSV(model, input_file, sentences_to_embed_dict)

                if json_output:
                    saveToJSON(model, input_file, sentences_to_embed_dict)

                if qdrant_url is not None:
                    saveToQdrant(model, qdrant_url, collection_name, input_file, sentences_to_embed_dict)
        except Exception as e:
            logger.error(f"An error occurred saving the embedded sentences: {e}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    parser.add_argument('-q', '--qdrant_url', required=False, help='The url for the Qdrant client')
    parser.add_argument('-n', '--collection_name', required=False, help='The name of the Qdrant collection')
    parser.add_argument('--tsv', action='store_const', const=True, help='Write the output to a tsv file')
    parser.add_argument('--json', action='store_const', const=True, help='Write the output to a json file')
    args = parser.parse_args()
    if not (args.tsv or args.json or args.qdrant_url):
        parser.error("At least one of --tsv, --json or --qdrant_url is required.")
    if args.qdrant_url and args.collection_name is None:
        parser.error("--collection_name is required when using --qdrant_url.")

    main(args.input, args.conf, args.tsv, args.json, args.qdrant_url, args.collection_name)
