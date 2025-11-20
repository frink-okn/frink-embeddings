#How to Run

# Original behavior (no types):  
"""python src/main.py \
  -i dreamkg.hdt \
  -c config.yaml \
  --json"""


# For new behavior (Type-aware): 
""" python /Users/maryammubarak/Desktop/frink-embeddings-main-typeAware/src/main.py 
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
import itertools
from logging import NullHandler

import numpy as np
import argparse
import yaml
from typing import Dict, Iterator, List, Optional, Iterable, Tuple

from rdflib import RDF, RDFS, OWL, URIRef, Literal
from rdflib.term import Identifier
from rdflib_hdt import HDTDocument
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


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
    parent_terms: Iterable[str],
    max_types: Optional[int] = None
) -> str:
    """
    Single, English-style phrasing for type-aware context (per review):
      "<label>, an instance of <type>"
      "<label>, an instance of <type1>; <type2>; ..."
    """
    label_text = normalize_label(label_text)

    # Clean and keep only non-empty type strings
    types = [
        (t or "").strip()
        for t in (type_terms or [])
        if (t or "").strip()
    ]
    parents = [
        (p or "").strip()
        for p in (parent_terms or [])
        if (p or "").strip()
    ]
    if not types and not parents:
        return label_text
    if max_types is not None and max_types >= 0:
        types = types[:max_types]
    type_text = f". An instance of {'; '.join(types)}" if types else ""
    parent_text = f". A kind of {'; '.join(parents)}" if parents else ""
    return f"{label_text}{type_text}{parent_text}"


def write_tsv(records: Iterator[Tuple[URIRef, str, torch.Tensor]], output: str) -> None:
     with open(output, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['iri', 'representation', 'embedding'])
        for subj, repr, embedding in records:
            writer.writerow([subj.replace('\n', ' ').replace('\r', ' '), repr, np.array2string(embedding, separator=', ').replace('\n', '')])


def write_json(records: Iterator[Tuple[URIRef, str, torch.Tensor]], output: str, graph_name: str) -> None:
     idx = 0
     with open(output, 'w') as f:
        f.write('{ "points": [\n')
        for subj, repr, embedding in records:
            idx += 1
            if idx > 1:
                f.write(",\n")
            embedded_dict = {
                "id": idx,
                "vector": embedding.tolist(),
                "payload": {"graph": graph_name, "iri": subj, "label": repr}
            }
            json.dump(embedded_dict, f, indent=3)
        f.write(']}')


def write_qdrant(records: Iterator[Tuple[URIRef, str, torch.Tensor]], url, collection_name, graph_name):
    # idx is to keep track of point ids for the Qdrant DB
    # inserts they need to be unique and not reset when performing
    # chunking of data
    # if adding data to an existing Qdrant collection, this number
    # must be set to the integer point id in the collection - it will
    # be incremented by 1 when it is used.
    idx = 0
    # connect to Qdrant client
    client = QdrantClient(url=url, timeout=60)
    if not client.collection_exists(collection_name):
        # collection does not exist - create and populate
        logger.debug(f"Collection '{collection_name}' does not exist - creating.")
        client.create_collection(collection_name=collection_name,
                                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    for chunk in itertools.batched(records, 1000):
        points = []
        for subj, repr, embedding in chunk:
            idx += 1
            points.append(
            models.PointStruct(
                id=idx,  # Unique ID for each point
                vector=embedding,
                payload={"graph": graph_name, "iri": subj, "label": repr}  # Add metadata (payload)
            )
        )
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Saved embeddings to Qdrant collection '{collection_name}'.")
        logger.info(f"last point id was {points[-1].id}")


def is_uri(term):
    return isinstance(term, URIRef) and not term.startswith("_:")


@lru_cache(maxsize=50000)
def lookup_label(uri: URIRef) -> str:
    """
    Given a URI (typically an rdf:type object),
    try several label predicates until one returns a value.
    Requires predicates and doc to be set within main method, due to cache.
    Returns: string label or None.
    """
    for pred in lookup_label.predicates: # type: ignore
        triples, _ = lookup_label.doc.search((uri, pred, None)) # type: ignore
        other_langs = []
        for (_, _, o) in triples:
            if isinstance(o, Literal):
                if o.language and (o.language == "en" or o.language.startswith("en-")):
                    return o  # return first label encountered
                else:
                    other_langs.append(o)
        for o in other_langs:
            return o
    return prettify_from_iri(uri)

def label_from_values(subject: URIRef, data: dict[URIRef, list[Identifier]]) -> str:
    for pred in lookup_label.predicates: # type: ignore
        values = data.get(pred) or []
        other_langs = []
        for value in values:
            if isinstance(value, Literal):
                if value.language and (value.language == "en" or value.language.startswith("en-")):
                    return value  # return first label encountered
                else:
                    other_langs.append(value)
            return value  # return first label encountered
        for value in other_langs:
            return value
    return prettify_from_iri(subject)


def representation_for_subject(subject: URIRef, data: dict[URIRef, list[Identifier]], doc: HDTDocument, conf: dict[str, set[str]]) -> str:
    types = data.get(RDF.type, [])
    labeled_types = [ (t, lookup_label(t)) for t in types ]
    parents = data.get(RDFS.subClassOf, [])
    labeled_parents = [ (p, lookup_label(p)) for p in parents ]
    subject_label = label_from_values(subject, data)
    label_and_types = english_with_types(subject_label, [v for (k, v) in labeled_types], [v for (k, v) in labeled_parents])
    extra_labels = []
    descriptions = []
    for p, objs in data.items():
        container = None
        if str(p) in conf['label']:
            container = extra_labels
        elif str(p) in conf['description']:
            container = descriptions
        else:
            container = None
        if container is not None:
            for o in objs:
                if o != subject_label:
                    container.append(str(o))
    aka_text = f". Also known as {'; '.join(extra_labels)}." if extra_labels else ''
    desc_text = '. ' + '. '.join(descriptions) if descriptions else ''
    return f"{label_and_types}{aka_text}{desc_text}"


def stream_by_subject(doc: HDTDocument, conf: dict[str, list[str]]) -> Iterator[Tuple[URIRef, Dict[URIRef, list[Identifier]]]]:
    """
    Yields subjects in order with only the wanted predicates retained.
    """
    triples, _ = doc.search((None, None, None)) # SPO ordered
    keep_predicates = { URIRef(v) for lst in conf.values() for v in lst }
    keep_predicates.update([ RDF.type, RDFS.subClassOf ])
    # group consecutive triples with same subject
    for subj, group in itertools.groupby(triples, key=lambda t: t[0]):
        if not is_uri(subj):
            continue
        collected = {}
        for (_, p, o) in group:
            if p in keep_predicates and not o.startswith("_:") and not o == subj and not o.startswith(str(RDF)) and not o.startswith(str(RDFS)) and not o.startswith(str(OWL)):
                collected.setdefault(p, []).append(o)
        yield subj, collected


def main(input_file: pathlib.Path, config_file: pathlib.Path, mode: str, output: str, collection_name: str, graph_name: str,
         type_aware: bool = False, types_json: Optional[pathlib.Path] = None, max_types: Optional[int] = None):
    logger.info(f"input: {input_file}  config: {config_file}")
    conf = {}
    with open(config_file, 'r') as f:
        conf = yaml.safe_load(f)
    conf_sets = { k: set(v) for k, v in conf.items() }
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.debug(f"Loaded model {model}")
    doc = HDTDocument(str(input_file), indexed = False)
    lookup_label.doc = doc # type: ignore
    lookup_label.predicates = [ URIRef(p) for p in conf['label'] ] # type: ignore
    def gen():
        for subj, data in stream_by_subject(doc, conf):
            repr = representation_for_subject(subj, data, doc, conf_sets)
            embedding = model.encode(repr)
            yield subj, repr, embedding
    if mode == 'tsv':
        write_tsv(gen(), output)
    elif mode == 'json':
        write_json(gen(), output, graph_name)
    else:
        write_qdrant(gen(), output, collection_name, graph_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    parser.add_argument('-o', '--output', required=True, help='The ourput file path or URL for the Qdrant endpoint')
    parser.add_argument('-n', '--collection-name', required=False, help='The name of the Qdrant collection')
    parser.add_argument('-g', '--graph-name', required=False, help='The name of the graph')
    parser.add_argument('-m', '--mode', type=lambda s: s.lower(), choices=["tsv", "json", "qdrant"], required=True, help='Output mode: TSV, JSON, or Qdrant')

    # New, minimal flags for type-aware behavior (per review)
    parser.add_argument('--type-aware', action='store_const', const=True,
                        help='Include type context in the English sentence (no mode; single phrasing).')
    parser.add_argument('--types-json', type=pathlib.Path, required=False,
                        help='Optional JSON file: { subject_iri: [type1, type2, ...], ... }')
    parser.add_argument('--max-types', type=int, required=False, default=None,
                        help='Optional cap on number of types to append.')

    args = parser.parse_args()
    if args.mode.lower == "qdrant" and args.collection_name is None:
        parser.error("--collection-name is required when outputting to Qdrant.")
    if args.mode.lower in {"json", "qdrant"} and args.graph_name is None:
        parser.error("--graph-name is required when outputting to JSON or Qdrant.")

    main(
        args.input,
        args.conf,
        args.mode,
        args.output,
        args.collection_name,
        args.graph_name,
        type_aware=bool(args.type_aware),
        types_json=args.types_json,
        max_types=args.max_types
    )
