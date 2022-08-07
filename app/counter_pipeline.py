import requests
import json
from neo4j import GraphDatabase

with open("./creds.json", "r") as f:
    config = json.load(f)

from neo4j import GraphDatabase

class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None

        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))

        except Exception as e:
            print("Failed to create driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None

        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query))

        except Exception as e:
            print("Query failed:", e)

        finally:
            if session is not None:
                session.close()

            return response

def identify_stance(topic, claim):
    # Note: COPIED CODE
    payload = {
        "topic": topic,
        "text": claim,
        "predictStance": True,
        "userID": config["argumentext_user"],
        "apiKey": config["argumentext_api_key"],
    }

    is_timed_out = True
    timed_out_ctr = 1
    while is_timed_out == True:
        try:
            json_dict = requests.post("https://api.argumentsearch.com/en/classify", timeout=300, data=json.dumps(payload),
                                      headers={'Content-Type': 'application/json'}).json()
            is_timed_out = False
        except requests.exceptions.Timeout or ConnectionError or json.decoder.JSONDecodeError:
            print("Timed out for {0} times.".format(str(timed_out_ctr)))
        except Exception as e:
            print(e)

    return json_dict

def identify_aspects(topic, claim):
    # Note: COPIED CODE
    payload = {
        "query": topic,
        "arguments": claim,
        "userID": config["argumentext_user"],
        "apiKey": config["argumentext_api_key"],
    }

    is_timed_out = True
    timed_out_ctr = 1
    while is_timed_out == True:
        try:
            json_dict = requests.post("https://api.argumentsearch.com/en/get_aspects", timeout=300, data=json.dumps(payload),
                                      headers={'Content-Type': 'application/json'}).json()
            is_timed_out = False
        except requests.exceptions.Timeout or ConnectionError or json.decoder.JSONDecodeError:
            print("Timed out for {0} times.".format(str(timed_out_ctr)))
        except Exception as e:
            print(e)

    return json_dict

from transformers import pipeline
import numpy as np
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def semantic_entailment(claim, statements, classifier=classifier):
    entails = classifier(claim, statements)

    idx_min = np.argmin(entails["scores"])
    idx_max = np.argmax(entails["scores"])

    entails_ = statements[idx_max]
    counter = statements[idx_min]

    return counter

def counter(topic, claim):
    # Construct Argument Unit
    # arg_unit = {
    #     "claim": claim,
    #     "topic": topic,
    #     "stance": identify_stance(topic, claim),
    #     "aspect": identify_aspects(topic, {"1": claim}),
    # }

    uri = "bolt://127.0.0.1:7687"
    user = "neo4j"
    pwd = "testing"

    graphdp = GraphDatabase.driver(uri=uri, auth=(user, pwd))
    session = graphdp.session()

    topic = [topic]
    results = []

    ### KB Query ###
    for i in topic:
        query_string_2 = "MATCH (n:CausalConcept {concept:$i})-[c]-(b)-[e]-(f:CausalConcept)" \
                         "MATCH (b)-[]-(h)" \
                         "RETURN n, b, e, c, f, h"
        nodes = session.run(query_string_2, i=i)
        results.append((i, nodes))

    query, nodes_ = results[0]
    results_ = [i for i in nodes_.data()]

    ### Cache Cause-Effect Triples ###
    concepts = set()

    for i in range(0, len(results_)):
        type, relation, concept = results_[i]["e"]
        concept = concept["concept"].replace("_", "")

        concepts.add(f"{concept} {relation} {query}")

    evidence = dict((k, []) for k in concepts)

    for i in range(0, len(results_)):
        type, relation, concept = results_[i]["e"]
        concept = concept["concept"].replace("_", "")

        concept_ = f"{concept} {relation} {query}"

        if results_[i]["h"]:
            ev = next(iter(results_[i]["h"].values()))
            evidence[concept_].append(ev)

        else:
            continue

    triples = [str(i) for i in evidence.keys()]
    triples

    entailment = semantic_entailment(claim, triples)
    counter_evidence_ = evidence[entailment]

    ### Argument Phrase ###
    counter_argument = ""
    entailment_ = entailment.split(" ")

    if entailment_[1] == "effect":
        counter_argument = f"{entailment_[2]} can lead to {entailment_[0]}"
    else:
        counter_argument = f"{entailment_[0]} can lead to {entailment_[2]}"

    return {
        "argument": claim,
        "counters": f"Did you know that {counter_argument}?",
        "evidence": counter_evidence_}


# from sentence_transformers import SentenceTransformer, util
# import torch

# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# def semantic_rank(claim, counters):
#     # Corpus with example sentences
#     corpus = [i for i in counters]
#     corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
#
#     # Query sentences:
#     query = [claim]
#     #arg_topics = dict((k, []) for k in queries)
#
#     # Find the closest sentences of the corpus for each query sentence based on cosine similarity
#     top_k = min(5, len(corpus))
#
#     for query in query:
#         query_embedding = embedder.encode(query, convert_to_tensor=True)
#
#         # We use cosine-similarity and torch.topk to find the highest 5 scores
#         cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
#         top_results = torch.topk(cos_scores, k=top_k)
#
#         print("\n\n======================\n\n")
#         print("Query:", query)
#         print("\nTop similar sentences in corpus:")
#
#         for score, idx in zip(top_results[0], top_results[1]):
#             print(corpus[idx], "(Score: {:.4f})".format(score))
#             counter = str(corpus[idx])
#
#     return counter

