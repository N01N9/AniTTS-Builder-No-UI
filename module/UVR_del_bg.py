import numpy as np
import librosa
import soundfile as sf
from audio_separator.separator import Separator
import os
import shutil

def UVR(model, stem, input_file, output_dir):

    separator = Separator(output_dir=output_dir, output_single_stem=stem)
    separator.load_model(model_filename=model)
    output_filename = separator.separate(input_file)[0]
    return os.path.join(output_dir, output_filename)

def load_wav_files(file_paths, sr=None):

    specs = []
    for file_path in file_paths:
        audio, sr = librosa.load(file_path, sr=sr)
        spec = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        specs.append(spec)
    return specs, sr

def max_spec_ensemble(spectrograms):
 
    max_spectrogram = np.max(np.abs(spectrograms), axis=0)
    
    max_phase = np.angle(spectrograms[0])
    max_spectrogram_complex = max_spectrogram * np.exp(1.j * max_phase)
    
    return max_spectrogram_complex

def save_wav_from_spec(spectrogram, sr, output_path):

    # ISTFT를 통해 시간 도메인으로 변환
    audio_output = librosa.istft(spectrogram, hop_length=512, win_length=2048)
    # 결과를 WAV 파일로 저장
    sf.write(output_path, audio_output, sr)

def UVR_ensemble(base_UVR_model_list, input_dir, output_dir, ensemble_output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(ensemble_output_dir):
        os.makedirs(ensemble_output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_dir, filename)
            
            # 모델별로 분리 결과를 저장할 리스트
            model_outputs = []
            
            # 모든 UVR 모델에 대해 처리
            for model_info in base_UVR_model_list:
                model, stem = model_info 
                output_wav_path = UVR(model, stem, file_path, output_dir)
                model_outputs.append(output_wav_path)
            
            # 모델별로 생성된 wav 파일들을 스펙트로그램으로 변환
            spectrograms, sr = load_wav_files(model_outputs)
            
            # Max Spec Ensemble 수행
            combined_spec = max_spec_ensemble(spectrograms)
            
            # 결합된 결과를 저장할 파일 경로
            ensemble_output_path = os.path.join(ensemble_output_dir, filename)
            
            # 결합된 결과를 시간 도메인으로 변환하여 wav 파일로 저장
            save_wav_from_spec(combined_spec, sr, ensemble_output_path)

            shutil.rmtree(output_dir)