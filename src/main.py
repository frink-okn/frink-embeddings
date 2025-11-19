#How to Run

# Original behavior (no types):  
"""python src/main.py \
  -i dreamkg.hdt \
  -c config.yaml \
  --json"""


# For new behavior (Type-aware): 
""" python /Users/maryammubarak/Desktop/frink-embeddings-main-typeAware/src/main.py \ 
  -i dreamkg.hdt \
  -c /Users/maryammubarak/Desktop/frink-embeddings-main-typeAware/conf/config.yaml \
  --json \
  --type-aware \
  --max-types 2 """


import pathlib
import logging
import sys
import os
import csv
import json
from logging import NullHandler

import numpy as np
import argparse
import yaml
from typing import Dict, List, Optional, Iterable

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


# RDF type IRI for pulling types directly from HDT
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def iri_tail(iri: str) -> str:
    """Return a readable tail from an IRI/URI for fallback labeling."""
    if not iri:
        return ""
    tail = iri.rsplit("/", 1)[-1]
    tail = tail.rsplit("#", 1)[-1]
    return tail


def prettify_from_iri(iri: str) -> str:
    """
    Derive a human-ish label from an IRI tail.
    Example: 'StainlessSteelProcessingCapability' -> 'Stainless Steel Processing Capability'
             'HP_0004321' -> 'HP 0004321'
    """
    tail = iri_tail(iri)
    if not tail:
        return iri or ""

    out = []
    token = ""
    for ch in tail:
        if ch in {"_", "-"}:
            if token:
                out.append(token)
                token = ""
        elif ch.isupper() and token and not token[-1].isupper():
            out.append(token)
            token = ch
        else:
            token += ch
    if token:
        out.append(token)

    label = " ".join(out).strip()
    return label or iri


def normalize_label(s: Optional[str]) -> str:
    return (s or "").strip()


def english_with_types(
    label_text: str,
    type_terms: Iterable[str],
    max_types: Optional[int] = None
) -> str:
    """
    Single, English-style phrasing for type-aware context (per review):
      "<label>, an instance of <type>"
      "<label>, an instance of <type1>; <type2>; ..."
    Here we follow Jim's example and use the raw type IRIs in the phrase.
    """
    label_text = normalize_label(label_text)

    # Clean and keep only non-empty type strings
    terms = [
        (t or "").strip()
        for t in (type_terms or [])
        if (t or "").strip()
    ]
    if not terms:
        return label_text

    if max_types is not None and max_types >= 0:
        terms = terms[:max_types]

    if len(terms) == 1:
        return f"{label_text}, an instance of {terms[0]}"

    return f"{label_text}, an instance of " + "; ".join(terms)



# create embeddings and write to a tsv file
def saveToTSV(model, input_file, sentences_to_embed_dict, file):
    tsv_file = os.path.splitext(os.path.basename(input_file))[0] + ".tsv"

    try:
        # create an embedding for each unique subject
        # and write to the tsv file
        # open the tsv output file if this is the first time through
        if file is None:
            file = open(tsv_file, 'w')
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['iri', 'label', 'embedding'])
        for k, v in sentences_to_embed_dict.items():
            embedding = model.encode(v)
            writer.writerow([k, v, np.array2string(embedding, separator=', ').replace('\n', '')])

        logger.info(f"Saved {len(sentences_to_embed_dict)} embeddings to {tsv_file}")
    except Exception as e:
        logger.error(f"An error occurred while save embedded data to the tsv file:{tsv_file} {e}")

    return file

# create embeddings and write to a json file that is in
# a format suitable for uploading into a vector database
def saveToJSON(model, input_file, sentences_to_embed_dict, file):
    global idx
    graph_name = os.path.splitext(os.path.basename(input_file))[0]
    json_file = os.path.splitext(os.path.basename(input_file))[0] + ".json"
    dict_list = []

    try:
        for k, v in sentences_to_embed_dict.items():
            idx += 1
            embedding = model.encode(v)
            embedded_dict = {
                "id": idx,
                "vector": embedding.tolist(),
                "payload": {"graph": graph_name, "iri": k, "label": v}
            }
            dict_list.append(embedded_dict)

        # open the json output file if this is the first time through
        if file is None:
            file = open(json_file, 'w')
        json.dump({'points': dict_list}, file, indent=3)

        logger.info(f"Saved {len(dict_list)} embeddings to {json_file}")
    except Exception as e:
        logger.error(f"An error occurred while saving embedded data to the json file:{json_file} {e}")

    return file

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
    for t in iri_types:
        value_list = list(t.values())[0]
        iri_values += value_list
    return iri_values


# counts how many strings in the sentence list match the target string
def countDuplicates(string_list, target):
    return sum(1 for s in string_list if s.lower() == target.lower())


def createSentences(
    embed_list: List[Dict[str, str]],
    *,
    type_aware: bool = False,
    types_by_subject: Optional[Dict[str, List[str]]] = None,
    max_types: Optional[int] = None
) -> Dict[str, str]:
    """
    Build a dict: subject IRI -> text to embed.

    Base (original) format:
      "<label>; also known as <alias1> <alias2> ...; description: ...; subject: ...; ..."

    Type-aware extension (per review) — applied exactly once here:
      If --type-aware and types exist for subject:
        "<label>, an instance of <type1>; <type2>; ...; also known as ...; description: ...; ..."

    If no label is present, we fall back to a prettified label from the subject IRI tail.
    """
    sentence_dict: Dict[str, List[Dict[str, str]]] = {}
    sentences_to_embed_dict: Dict[str, str] = {}
    types_by_subject = types_by_subject or {}

    # Collect by subject
    for item in embed_list:
        subject = item.get('subject')
        obj = item.get('object')
        key = item.get('config_key')  # subject IRI (same as `subject`, but preserving your naming)
        if subject is not None and key is not None:
            if subject not in sentence_dict:
                sentence_dict[subject] = []
            sentence_dict[subject].append({key: obj})

    also_str = "also known as"

    # Build sentence per subject
    for subj, kv_list in sentence_dict.items():
        # Gather labels first
        labels = [d["label"] for d in kv_list if "label" in d]
        label_main = normalize_label(labels[0]) if (labels and labels[0]) else prettify_from_iri(subj)

        sentence = label_main

        # Append types (English phrasing) before other predicates
        if type_aware:
            type_terms = types_by_subject.get(subj, [])
            sentence = english_with_types(
                label_text=sentence,
                type_terms=type_terms,
                max_types=max_types
            )

        # Handle additional labels as "also known as"
        if labels and len(labels) > 1:
            # only add "also known as" if not all duplicates of the main label
            if countDuplicates(labels, label_main) != len(labels):
                sentence += f"; {also_str}"
                for label in labels[1:]:
                    lab = normalize_label(label)
                    if lab and lab not in sentence:
                        sentence += f" {lab}"

        # Append the rest of the predicates (non-label keys).
        # IMPORTANT: when type-aware is enabled, we *skip* "type"
        # here so it is not duplicated (it's already in "an instance of ...").
        for value_dict in kv_list:
            for k, v in value_dict.items():
                if k == "label":
                    continue
                if type_aware and k == "type":
                    # avoid a second "type: ..." clause when we're already
                    # including types via english_with_types
                    continue
                sentence += f"; {k}: {v}"

        sentences_to_embed_dict[subj] = sentence

    return sentences_to_embed_dict

def isURI(term):
    return isinstance(term, URIRef) and not term.startswith("_:")


# create embeddings from rdf triples
def main(input_file: pathlib.Path, config_file: pathlib.Path, tsv_output, json_output, qdrant_url, collection_name,
         type_aware: bool = False, types_json: Optional[pathlib.Path] = None, max_types: Optional[int] = None):
    logger.info(f"input: {input_file}  config: {config_file}")

    tsvfile = None
    jsonfile = None

    doc = HDTDocument(str(input_file), indexed=False)
    logger.info(f"subjects: {doc.nb_subjects}  predicates: {doc.nb_predicates}  objects: {doc.nb_objects}")

    iri_types = getIriTypes(config_file)
    iri_values = getAllIriTypeValues(iri_types)

    embed_list: List[Dict[str, str]] = []
    sentences_to_embed_dict: Dict[str, str] = {}
    model = None

    # collect rdf:type triples from HDT into a dict
 
    types_by_subject: Dict[str, List[str]] = {}
    try:
        triples_type, _ = doc.search((None, URIRef(RDF_TYPE), None))
        for s, p, o in triples_type:
            subj = str(s)
            typ = str(o)
            types_by_subject.setdefault(subj, []).append(typ)
        logger.info(f"Collected rdf:type for {len(types_by_subject)} subjects from HDT.")
    except Exception as e:
        logger.error(f"Error collecting rdf:type from HDT: {e}")

    # Optional: merge user-provided types JSON on top
    if types_json is not None and types_json.exists():
        try:
            user_types = json.loads(types_json.read_text(encoding="utf-8"))
            merged_count = 0
            for subj, tlist in user_types.items():
                if not tlist:
                    continue
                types_by_subject.setdefault(subj, [])
                existing = set(types_by_subject[subj])
                for t in tlist:
                    if t not in existing:
                        types_by_subject[subj].append(t)
                        existing.add(t)
                        merged_count += 1
            logger.info(f"Merged types from {types_json} into HDT-derived types (added {merged_count} entries).")
        except Exception as e:
            logger.error(f"Failed to read types JSON {types_json}: {e}")

    # Collect triples according to config predicates
    try:
        triples, cardinality = doc.search((None, None, None))
        for s, p, o in triples:
            if str(p) in iri_values and isinstance(p, URIRef):
                key = getIriKeyFromValue(iri_types, str(p))
                embed_dict = {"config_key": key, "subject": str(s), "predicate": str(p), "object": str(o)}
                embed_list.append(embed_dict)

        # Build sentences (single point where type-aware is applied)
        sentences_to_embed_dict = createSentences(
            embed_list,
            type_aware=type_aware,
            types_by_subject=types_by_subject,
            max_types=max_types
        )

    except Exception as e:
        logger.error(f"An error occurred while parsing file {input_file}: {e}")

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("loaded model")
        # Check if CUDA is available
        # print("CUDA available:", torch.cuda.is_available())
        # Check if model parameters are on GPU
        # print("Model is on GPU:", next(model.parameters()).is_cuda)
    except Exception as e:
        logger.error(f"An error occurred while loading the embedding model: {e}")

        # now see how we want to embed and save this data
    try:
        if tsv_output:
            tsvfile = saveToTSV(model, input_file, sentences_to_embed_dict, tsvfile)

        if json_output:
            jsonfile = saveToJSON(model, input_file, sentences_to_embed_dict, jsonfile)

        if qdrant_url is not None:
            saveToQdrant(model, qdrant_url, collection_name, input_file, sentences_to_embed_dict)
    except Exception as e:
        logger.error(f"An error occurred saving the embedded sentences: {e}")

    # close up any open files
    if tsvfile is not None:
        tsvfile.close()
    if jsonfile is not None:
        jsonfile.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    parser.add_argument('-q', '--qdrant_url', required=False, help='The url for the Qdrant client')
    parser.add_argument('-n', '--collection_name', required=False, help='The name of the Qdrant collection')
    parser.add_argument('--tsv', action='store_const', const=True, help='Write the output to a tsv file')
    parser.add_argument('--json', action='store_const', const=True, help='Write the output to a json file')

    # New, minimal flags for type-aware behavior (per review)
    parser.add_argument('--type-aware', action='store_const', const=True,
                        help='Include type context in the English sentence (no mode; single phrasing).')
    parser.add_argument('--types-json', type=pathlib.Path, required=False,
                        help='Optional JSON file: { subject_iri: [type1, type2, ...], ... }')
    parser.add_argument('--max-types', type=int, required=False, default=None,
                        help='Optional cap on number of types to append.')

    args = parser.parse_args()
    if not (args.tsv or args.json or args.qdrant_url):
        parser.error("At least one of --tsv, --json or --qdrant_url is required.")
    if args.qdrant_url and args.collection_name is None:
        parser.error("--collection_name is required when using --qdrant_url.")

    main(
        args.input,
        args.conf,
        args.tsv,
        args.json,
        args.qdrant_url,
        args.collection_name,
        type_aware=bool(args.type_aware),
        types_json=args.types_json,
        max_types=args.max_types
    )