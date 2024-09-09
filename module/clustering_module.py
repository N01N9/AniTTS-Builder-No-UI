import torch
import numpy as np
from sklearn.cluster import DBSCAN
import os
import json
import shutil
from speechbrain.inference import EncoderClassifier
import torchaudio

def clustering(distance_matrix_path, embedding_path, json_path, directory, destination_folder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cosine_similarity_matrix = torch.load(distance_matrix_path).cpu().numpy()
    cosine_distance_matrix = (1-(np.clip(cosine_similarity_matrix, -1.0, 1.0)))/2

    embeddings = torch.load(embedding_path).cpu()

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-8).to(device)

    with open(json_path, 'r') as file:
        data1 = json.load(file)
        data2 = dict(map(reversed,data1.items()))

    ecapa_tdnn = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device})
    


    db = DBSCAN(eps=0.1, min_samples=2, metric='precomputed').fit(cosine_distance_matrix)

    labels = db.labels_

    core_sample_mask = np.zeros_like(labels, dtype=bool)
    core_sample_mask[db.core_sample_indices_] = True

    clusters_core_indices = {}
    for label in set(labels):
        if label != -1:
            core_indices = np.where((labels == label) & core_sample_mask)[0]
            clusters_core_indices[label] = core_indices

    clusters_all_indices = {}
    for label in set(labels):
        if label != -1:
            all_indices = np.where(labels == label)[0]
            clusters_all_indices[label] = all_indices



    def cos_distance(a, b):
        a,b = a.cpu(), b.cpu()
        return (1-(np.clip(cosine_similarity(a,b), -1.0, 1.0)))/2

    def mean(embeddings):
        waveforms = []
        for filename in [data2[x[0]] for x in embeddings]:
            filename = filename.replace('.pt','.wav')
            filepath = os.path.join(directory, filename)
            # 스테레오 오디오 파일 로드
            waveform, sample_rate = torchaudio.load(filepath)

                    # 스테레오를 모노로 변환 (좌, 우 채널의 평균)
            if waveform.shape[0] == 2:  # 2개의 채널이면
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 샘플링 레이트를 16kHz로 변환 (필요한 경우)
            if sample_rate != 16000:
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = transform(waveform)

            waveforms.append(waveform)

        combined_waveform = torch.cat(waveforms, dim=1)

        embedding = ecapa_tdnn.encode_batch(combined_waveform)

        return embedding.squeeze()

    def kmeans_with_noise(X, X_start, k, max_distance, max_iters=100):

        centroids = X_start

        noise_points = np.full(len(list(X)), False)

        for num in range(max_iters):
            clusters = [[] for _ in range(k)]
            noise_points.fill(False)

            for idx, x in X:

                distances = [cos_distance(x, centroid) for centroid in centroids]
                cluster_idx = np.argmin(distances)
                
                if distances[cluster_idx] > max_distance:
                    noise_points[idx] = True
                else:
                    clusters[cluster_idx].append([idx,x])

            new_centroids = []
            for i in range(k):
                if clusters[i]: 
                    new_centroids.append(mean(clusters[i]))
                else:
                    new_centroids.append(centroids[i])
            new_centroids = torch.stack(new_centroids).to(device=device)

            if torch.equal(centroids, new_centroids):
                break
            else:
                centroids = new_centroids
        
        idx_clusters = []
        for cluster in clusters:
            cluster_ = []
            for idx, x in cluster:
                cluster_.append(idx)
            idx_clusters.append(cluster_)

        return centroids, idx_clusters, noise_points

    X = [[idx,x] for idx,x in enumerate(embeddings.to(device=device))]
    X_start = embeddings[[indices[0] for indices in clusters_core_indices.values()]].to(device=device)
    k = len(clusters_core_indices.keys())

    max_distance = 0.2

    centroids_result, clusters_result, noise_points_result = kmeans_with_noise(X, X_start, k, max_distance, max_iters=1000)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    for label in range(len(clusters_result)):
        label_folder = os.path.join(destination_folder, f'clustering_{label}')
        os.makedirs(label_folder, exist_ok=True)
  
        for file_idx in clusters_result[label]:
            file_name =  data2[file_idx]
            source_file = os.path.join(directory, file_name)
            destination_file = os.path.join(label_folder, file_name)
            
            if os.path.exists(source_file):
                shutil.copy(source_file, destination_file)
                os.remove(source_file)
            else:
                print(f"File not found: {source_file}")



