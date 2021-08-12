import copy
import csv
import json
import os
import random
from parrot import Parrot
import warnings

from sentence_transformers import SentenceTransformer
from scipy.spatial import distance


def vector_distance(v1, v2):
    return distance.cdist(v1, v2, 'euclidean')


def centroid(question_cluster: list, ref_dict: dict):
    center = []
    for dim in range(len(ref_dict[question_cluster[0]])):
        total = 0
        for q in range(len(question_cluster)):
            total += ref_dict[question_cluster[q]][dim]
        center.append(total / len(ref_dict[question_cluster[0]]))

    return center


def dbscan(qv: dict, eps: float, min_samples: int):
    qv_copy = copy.deepcopy(qv)
    clusters = []
    representatives = []
    while len(qv) > 0:
        random_key = random.choice(list(qv.keys()))
        cluster = [random_key]
        core = {random_key: qv[random_key]}
        qv.pop(random_key)
        for remaining_key in qv.keys():
            dist = vector_distance(core[random_key], qv[remaining_key])
            actual_dist = dist[0][0]
            if actual_dist <= eps:
                cluster.append(remaining_key)
        if len(cluster) >= min_samples:
            clusters.append(cluster)
            center = centroid(cluster, qv_copy)
            rep = ""
            rep_val = -1
            for question in cluster:
                dist_from_center = vector_distance(center, qv_copy[question])[0][0]
                if rep_val == -1 or dist_from_center < rep_val:
                    rep = question
                    rep_val = dist_from_center
            representatives.append(rep)
        for k in range(1, len(cluster)):
            qv.pop(cluster[k])

    return clusters, representatives


warnings.filterwarnings("ignore")
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
data_dir = []
laptop_models = []
vector_totals = []
question_totals = []
model = SentenceTransformer('paraphrase-mpnet-base-v2')
i = 0

for file in os.listdir('data'):
    data_dir.append(file)

for file in data_dir:
    i += 1
    with open('./data/' + file) as json_file:
        data = json.load(json_file)

    qa_raw = data["qAndA"]
    qa = []
    vectors = []

    for pair in qa_raw:
        qa.append(pair['question'])

    sentence_embeddings = model.encode(qa)

    for embedding in zip(sentence_embeddings):
        vectors.append(embedding)

    question_totals.append(qa)
    vector_totals.append(vectors)
    laptop_models.append(file[:len(file) - 9])
    print("Embedding: {0}/{1}".format(i + 1, len(data_dir)))

for i in range(len(laptop_models)):
    out_file = csv.writer(open("./clustered/" + laptop_models[i] + ".csv", "w+"))
    question_to_vec = {}
    for j in range(len(question_totals[i])):
        question_to_vec[question_totals[i][j]] = vector_totals[i][j]
    final_clusters, final_representatives = dbscan(question_to_vec, 2.5, 5)
    print()
    for r in range(len(final_representatives)):
        para_phrases = parrot.augment(input_phrase=final_representatives[r], use_gpu=False, max_return_phrases=1)
        if para_phrases is not None:
            for p in para_phrases:
                final_representatives[r] = p[0]
    for j in range(len(final_representatives)):
        out_file.writerow([j + 1, final_representatives[j], final_clusters[j]])
    print("Writing: {0}/{1}".format(i + 1, len(laptop_models)))
