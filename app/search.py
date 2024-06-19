import numpy as np
import pickle

def get_video_token(address):
    token = ""
    check = False
    for i in range(len(address) - 1, -1, -1):
        if address[i] == "/":
            check = False
        if check:
            token += address[i]
        if address[i] == '-':
            check = True
    return token[::-1]


def check_similarity(music_cap_collection, prompt_embeddings, size=128, pct=0.2, given_index=None):
    prompt_embeddings = prompt_embeddings.reshape(-1)
    prompt_embeddings = prompt_embeddings[:size]
    cosine_sim_list = []
    if size == 128:
        extracted_index = music_cap_collection.find()
    else:
        given_index = given_index.tolist()
        query = {"index": {"$in": given_index}}
        extracted_index = music_cap_collection.find(query)
    for embeddings in extracted_index:
        index = embeddings["index"]
        embed_vector = pickle.loads(embeddings["embeddings"])
        slices_embed = embed_vector[:size]
        cosine_sim_list.append([np.dot(slices_embed, prompt_embeddings.T), index])
    cosine_sim_list.sort(reverse=True)
    length = len(cosine_sim_list)
    top_k_ele = np.array(cosine_sim_list[:int(pct*length) + 1], dtype=int)
    return top_k_ele[:, 1]

def get_music_address(music_cap_collection, index):
    given_index = index.tolist()
    query = {"index": {"$in": given_index}}
    extracted_index = music_cap_collection.find(query)
    address_list = []
    for address in extracted_index:
        address_list.append(get_video_token(address["address"]))
    return address_list
