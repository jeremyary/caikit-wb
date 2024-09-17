# Working with Caikit Embedding / Reranking

## Purpose of project

Extend use cases that the Red Hat AI BU has made available [here](https://github.com/rh-aiservices-bu/llm-on-openshift) to showcase 
use of Caikit Standalone [ServingRuntime](https://kserve.github.io/website/0.8/modelserving/servingruntimes/) for both
Embedding (bi-encoder) & Rerank (cross-encoder).

## Origin

This repo builds on demo use cases that the BU has [supplied](https://github.com/rh-aiservices-bu/llm-on-openshift/tree/main/examples/notebooks/langchain), therefore several files found within are not currently relevant to the described processed entailed below.

## Prerequisites

OpenShift cluster with Red Hat Openshift AI deployed & configured to utilize [Single-Model serving stack](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2-latest/html/serving_models/serving-large-models_serving-large-models). At the 
time of writing, Caikit standalone doesn't work with multi-model serving stack. Note that once a model has been served, your cluster will lock into single/multi mode that was selected,
so preexisting clusters configured for multi-model won't work for this purpose. Note that Prerequisites for the included Langchain demo use case requires additional setup / config, 
see [this section](#langchain-prerequisites) before continuing.

Please note that at the time of writing, the `Caikit Standalone Runtime` **ONLY** supports HTTP protocol, gRPC has not yet been implemented.

## Proof of Concept Example Use Case
The following notebooks represent straight-forward invocation of the Embedding & Reranking endpoints of a Caikit Standalone Runtime serving the `all-MiniLM-L12-v2`
model converted to Caikit format using [this process](https://github.com/opendatahub-io/caikit-tgis-serving/blob/main/demo/kserve/built-tip.md#bootstrap-process).

### Embedding Calls
[call_SR_from_caikit_client.ipynb](call_SR_from_caikit_client.ipynb)
- This notebook showcases using [caikit-nlp](https://github.com/caikit/caikit-nlp) & the corresponding [client](https://github.com/opendatahub-io/caikit-nlp-client/) 
(albeit with some [modifications](/caikit_nlp_client) on top of what the ODH community has done).
```python
# Define some lightweight documents to use for embedding examples
input_docs = [
    "Green is the color of envy, grass, and Kermit the frog.",
    "Red is the color of the planet Mars & the commonly associated with both Valentine's Day & Christmas holidays.",
    "Blue is the color of sadness & jazz music.",
    "Purple is the color you receive when mixing red & blue"
]

# Bring in the needed ServingRuntime endpoint & Bearer token to config the HttpClient
model_endpoint_url = "https://caikit-nomic5-jary-wb.apps.jary-intel-opea.51ty.p1.openshiftapps.com"
embedding_model_name = "all-MiniLM-L12-v2-caikit"

http_client = HttpClient(model_endpoint_url, verify=False)
token = '''\
Bearer <token>
'''.replace('\n', '')

# Call the embedding_tasks endpoint to embed the multiple docs we've supplied 
embedded_vectors = http_client.embedding_tasks(
  token=token,
  model_id=embedding_model_name,
  texts=input_docs,
)
```
The resulting output includes a list of the supplied documents + their vectorized representation (truncated here for space):
```json
Response: <Response [200]>
[
    {
        "Green is the color of envy, grass, and Kermit the frog.": {
            "data": {
                "values": [
                    0.029085179790854454,
                    0.06175018101930618,
                    0.09134913980960846,
                    -0.036822956055402756,
                    .......
                    -0.030950335785746574
                ]
            }
            ......
        }
    }
]
```

### Rerank Calls

With a slightly different document format, we can also showcase the Reranking inference endpoint:
```python
input_docs = [
        {
            "text":  "Green is the color of envy, grass, and Kermit the frog.",
            "title": "first title"
        },
        {
            "text": "Red is the color of the planet Mars & the commonly associated with both Valentine's Day & Christmas holidays.",
            "more": "more attributes here"
        },
        {
            "text": "Blue is the color of sadness & jazz music.",
            "meta": {"foo": "bar", "i": 999, "f": 12.34}
        },
        {
            "text": "Purple is the color you receive when mixing red & blue",
            "meta": "blah"
        }
    ]

reranked_vectors = http_client.rerank(
  model_id=embedding_model_name,
  documents=input_docs,
  token=token,
  query="Tell me about the color red"
)

print(json.dumps(reranked_vectors, sort_keys=True, indent=4))
```

The resulting output includes the list of supplied documents, ordered by relevance & including attached scores:
```json
{
    "input_token_count": 75,
    "producer_id": {
        "name": "EmbeddingModule",
        "version": "0.0.1"
    },
    "result": {
        "query": "Tell me about the color red",
        "scores": [
            {
                "document": {
                    "more": "more attributes here",
                    "text": "Red is the color of the planet Mars & the commonly associated with both Valentine's Day & Christmas holidays."
                },
                "index": 1,
                "score": 0.5381495356559753,
                "text": "Red is the color of the planet Mars & the commonly associated with both Valentine's Day & Christmas holidays."
            },
            .........
            {
                "document": {
                    "meta": {
                        "f": 12.34,
                        "foo": "bar",
                        "i": 999
                    },
                    "text": "Blue is the color of sadness & jazz music."
                },
                "index": 2,
                "score": 0.3349013924598694,
                "text": "Blue is the color of sadness & jazz music."
            }
        ]
    }
}
```

## Langchain Example Use Case

I've included a worksheet copying one of the other BU-provided demo use cases in order to showcase utiliziation of langchain for invoking Embedding/Rerank endpoints.
This use case should build on the previously showcased functionality above, but demonstrate a workflow that would much more likely encountered in actual business use cases.

### Langchain Prerequisites

If you wish to follow this more complex example of making inference calls via Langchain & persisting vectors to a vector DB 
like [Milvus](https://milvus.io/), then you will need further project initialization following the Requirements, Model Serving, & Vector Store sections of 
[this guide](https://ai-on-openshift.io/demos/llm-chat-doc/llm-chat-doc/).

### A few issues I encountered when following this process:

- The Milvus installation utilized YAML configuration files coming from a ConfigMap that are copied into the Milvus image to override config & provide the needed 
endpoint to utilize. I had to add a third YAML config file called `milvus.yaml` as the provided `default.yaml`, when placed within the image, was not overriding or 
adding to the default `milvus.yaml` config and this it needed replacement.
- Ensure during your OpenShift AI installation that it is set to `Managed` for ALL components, including those which utilize Authorino. If you follow the steps within the guide linked above, this will largely be called out & handled as you go.
- If you wish for any added custom ServingRuntime to show in the RHOAI UI as supporting any Accelerator Profiles, you'll need to add an annotation to your runtime YAML that can be found & copied from
some of the other out-of-the-box profiles:
```yaml
metadata:
  annotations:
    opendatahub.io/recommended-accelerators: '["nvidia.com/gpu"]'
```

### Setup

The demo use case set up within [this notebook](Langchain-Milvus-Ingest-Caikit.ipynb) is a bit more complex than previous examples, but also more representative of workflows likely to be encountered in the wild.

- Top portions of the notebook deal with configuration info for Milvus, as well as preparing several RH product PDF's & websites into document chunks that can be
vectorized and persisted/utilized. I'll leave these as a reading exercise given that the logic is not complex nor relevant to the topic herein.

  
- Initial code setup somewhat resembles the above examples in that we're providing configuration info for our endpoint calls, however, we aren't cofiguring the HttpClient directly here. Rather, we
pass the info on to a LangchainEmbeddings [object](caikit_nlp_client/langchain_embeddings.py) which was created to bridge the gap between Langchain workflow & our Caikit standlone runtime.
Note that there's no direct invocation here, but rather, an association of our intended Embeddings object is passed to Milvus for function assignment (think 'handler' pattern).
```python
model = "all-MiniLM-L12-v2-caikit"
model_endpoint_url = "https://caikit-nomic5-jary-wb.apps.jary-intel-opea.51ty.p1.openshiftapps.com"
token = '''\
Bearer <token>
'''.replace('\n', '')

embeddings = LangchainEmbeddings(
    token=token,
    endpoint=model_endpoint_url,
    model=model
)

db = Milvus(
    embedding_function=embeddings,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    auto_id=True,
    drop_old=True
    )
```

- Invocation, as a result, looks different as all we need to do is use Langchain to "store" the docs to the db, from there invocations occur behind the scenes:
```python
db.add_documents(all_splits)
```
- And likewise, querying also utilizes Langchain that will lean on the underlying embeddings for augmented generation:
```python
query = "How can I work with GPU and taints in OpenShift AI?"
docs_with_score = db.similarity_search_with_score(query)
```
- Accompanying results:
```python
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

--------------------------------------------------------------------------------
Score:  1.0047132968902588
Red Hat OpenShift AI Self-Managed

2.12
Getting started with Red Hat OpenShift AI
Self-Managed
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
Score:  1.0182170867919922
Self-Managed
Learn how to work in an OpenShift AI environment
Last Updated: 2024-08-27
--------------------------------------------------------------------------------
......
```