# frink-embeddings

### Example usage
```
usage: python main.py [-h] -i INPUT -c CONF [-q QDRANT_URL] [--csv] [--json]

frink-embeddings

options:
  -h, --help                               Show this help message and exit
  -i INPUT, --input INPUT                  An hdt file from an rdf graph
  -c CONF, --conf CONF                     The yaml file for configuration
  -q QDRANT_URL, --qdrant_url QDRANT_URL   The url for the Qdrant client
  -n COLLECTION_NAME, --collection_name COLLECTION_NAME
                                           The name of the Qdrant collection (required if using Qdrant)
  --tsv                                    Write the output to a tsv file
  --json                                   Write the output to a json file formatted for Qdrant upload
```

### IMPORTANT NOTE for adding data to an existing Qdrant collection
This application uses sequential integers for Qdrant data point ids.
By default, the Qdrant client will start adding data to a collection
with point id = 0.
The code can be modified to use a different starting point id, if you
are trying to add data to an existing collection with the same scheme of
using sequential integers for point ids. 
There is a global variable to keep track of point ids for the Qdrant DB
inserts, named **idx**. This is set to 0 by default.
The Qdrant data point ids need to be unique.
If adding data to an existing Qdrant DB, the src/main.py file must be
edited to set the **idx** variable to the highest point id in the existing
Qdrant DB collection. It will be incremented by 1 when it is used. For instance,
setting idx = 4895902, will start adding data with point id = 4895903.
### FAILURE TO MANUALLY UPDATE THE IDX VARIABLE WILL OVERWRITE EXISTING DATA ###


#### Another Note:
The graph name, which is stored as metadata for the Qdrant collection
points, is derived from the input hdt file name.