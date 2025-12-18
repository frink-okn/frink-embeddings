# frink-embeddings

This project is built and run using [uv](https://docs.astral.sh/uv/).

### Example usage
```
usage: main.py [-h] -i INPUT -c CONF -o OUTPUT [-n COLLECTION_NAME] [-g GRAPH_NAME] -m {tsv,json,qdrant} [--type-aware] [--types-json TYPES_JSON]
               [--max-types MAX_TYPES] [--batch-size BATCH_SIZE]

frink-embeddings

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        An hdt file from an rdf graph
  -c CONF, --conf CONF  The yaml file for configuration
  -o OUTPUT, --output OUTPUT
                        The ourput file path or URL for the Qdrant endpoint
  -n COLLECTION_NAME, --collection-name COLLECTION_NAME
                        The name of the Qdrant collection
  -g GRAPH_NAME, --graph-name GRAPH_NAME
                        The name of the graph
  -m {tsv,json,qdrant}, --mode {tsv,json,qdrant}
                        Output mode: TSV, JSON, or Qdrant
  --type-aware          Include type context in the English sentence (no mode; single phrasing).
  --types-json TYPES_JSON
                        Optional JSON file: { subject_iri: [type1, type2, ...], ... }
  --max-types MAX_TYPES
                        Optional cap on number of types to append.
  --batch-size BATCH_SIZE
                        Batch size for processing embeddings (default: 10000). Use around 500-1000 for running on a laptop.
```
