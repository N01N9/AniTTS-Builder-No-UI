import os
import json
from pydub import AudioSegment  
import chardet

def slice_audio_from_subtitles(audio_file_path, json_file_path, output_folder_path, filename, infojson_path, type):
    # JSON 파일 읽기

    with open(json_file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

    with open(json_file_path, 'r',encoding=encoding) as json_file:
        subtitles_dict = json.load(json_file)

    if type=="vocal":
        # 파일이 있는지 확인
        if not os.path.exists(infojson_path):
            # 파일이 없으면 빈 파일 생성
            with open(infojson_path, 'w',encoding=encoding) as file:
                json.dump({}, file, ensure_ascii=False)  # 빈 JSON 객체를 작성하여 초기화

        if os.path.getsize(json_file_path) == 0:
            print(f"Warning: {json_file_path} is empty.")
            wavinfo = {}  # 빈 JSON 파일일 경우 기본값 설정
        else:
            # 파일 열기 (읽기 모드로)
            with open(infojson_path, 'r',encoding=encoding) as file:
                wavinfo = json.load(file)
    
    # 오디오 파일 읽기
    audio = AudioSegment.from_wav(audio_file_path)
    
    
    # 자막에 따라 오디오 파일 자르기 및 저장
    for idx, subtitle in subtitles_dict.items():
        start_time = parse_time_to_milliseconds(subtitle["start"])
        end_time = parse_time_to_milliseconds(subtitle["end"])
        audio_slice = audio[start_time:end_time]
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        output_file_path = os.path.join(output_folder_path, f"{filename}-{idx}.wav")
        if type=="vocal":
            wavinfo[f"{filename}-{idx}.wav"] = {'subtitles':subtitle["text"]}
        audio_slice.export(output_file_path, format="wav")


    if type=="vocal":
        with open(infojson_path, 'w',encoding=encoding) as file:
            json.dump(wavinfo, file, ensure_ascii=False,indent=4)

def parse_time_to_milliseconds(time_str):
    # 시간 형식이 "H:MM:SS.sss"로 주어졌을 때 밀리초로 변환
    h, m, s = map(float, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * 1000)

def find_matching_json(wav_folder, json_folder, output_folder, infojson_file_path, type):
    # wav_folder의 모든 파일명을 가져옴
    wav_files = os.listdir(wav_folder)
    
    for wav_file in wav_files:
        if wav_file.endswith('.wav'):
            # .wav 확장자를 제거하여 파일명만 추출
            file_name_without_ext = os.path.splitext(wav_file)[0]
            
            # .json 파일의 예상 경로
            json_file_path = os.path.join(json_folder, file_name_without_ext + '.json')
            
            # 해당 경로에 .json 파일이 있는지 확인
            if os.path.exists(json_file_path):
                audio_file_path = os.path.join(wav_folder, wav_file)
                slice_audio_from_subtitles(audio_file_path, json_file_path, output_folder, file_name_without_ext, infojson_file_path, type)

                os.remove(audio_file_path)

            else:
                print(f"WAV 파일: {os.path.join(wav_folder, wav_file)}")
                print("매칭되는 JSON 파일 없음\n")

