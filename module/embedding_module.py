import torch
from speechbrain.inference import EncoderClassifier
import os
import torchaudio
import json
from tqdm import tqdm

def compute_cosine_similarity_batch(device, embeddings, cosine_similarity, batch_size=32):
    num_embeddings = embeddings.size(0)
    result_tensor = torch.zeros((num_embeddings, num_embeddings), device=device)

    for i in range(0, num_embeddings, batch_size):
        batch_embeddings = embeddings[i:i+batch_size]
        similarity_batch = cosine_similarity(batch_embeddings.unsqueeze(1), embeddings.unsqueeze(0)).squeeze()
        result_tensor[i:i+batch_size] = similarity_batch

    return result_tensor

def embeddings(directory, all_embeddings_path, outputfile_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ecapa_tdnn = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device})
    
    embedding_dict = {}
    all_embeddings = []

    batch_size = 32  # 배치 크기를 설정, 필요에 따라 조절 가능
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]

    for i in tqdm(range(0, len(wav_files), batch_size)):
        batch_files = wav_files[i:i + batch_size]
        batch_embeddings = []

        for filename in batch_files:
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

            # 임베딩 추출
            embedding = ecapa_tdnn.encode_batch(waveform)
            batch_embeddings.append(embedding)
            embedding_dict[filename] = len(all_embeddings) + len(batch_embeddings) - 1

        all_embeddings.extend(batch_embeddings)

    all_embeddings_tensor = torch.stack(all_embeddings).squeeze()

    torch.save(all_embeddings_tensor, os.path.join(all_embeddings_path, "all_embeddings.pt"))

    with open(os.path.join(all_embeddings_path, "embedding_map.json"), 'w') as json_file:
        json.dump(embedding_dict, json_file)

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-8).to(device)
    result_tensor = compute_cosine_similarity_batch(device, all_embeddings_tensor, cosine_similarity, batch_size=32)

    torch.save(result_tensor, outputfile_path)
