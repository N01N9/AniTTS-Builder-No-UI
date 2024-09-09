import os
from moviepy.editor import VideoFileClip
import json
import ass
import chardet
import re

def convert_mp4_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')
            
            video = VideoFileClip(input_path)
            audio = video.audio
            audio.write_audiofile(output_path)

def convert_ass_to_json(style, ass_folder_path, output_folder_path):

    def extract_specific_style_subtitles(ass_file_path, target_style, output_file_path):

        with open(ass_file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(ass_file_path, 'r', encoding=encoding) as file:
            doc = ass.parse(file)
        
        # 특정 스타일의 자막만 추출
        subtitles = []
        
        for event in doc.events:
            if event.style == target_style or target_style == None:
                subtitle = {
                    "start": str(event.start),
                    "end": str(event.end),
                    "text": re.sub(r'[\u202A-\u202E]', '', event.text.replace("\\N", " ").replace("\n", " ").strip())
                }
                subtitles.append(subtitle)
        
        # 추출된 자막을 사전 형식으로 저장
        subtitles_dict = {i: subtitle for i, subtitle in enumerate(subtitles)}
        
        # 결과를 JSON 파일로 저장
        with open(output_file_path, 'w', encoding=encoding) as json_file:
            json.dump(subtitles_dict, json_file, ensure_ascii=False, indent=4)

    target_style = style #ass파일 중 대사 style만 추출

    # 폴더 내 모든 .ass 파일에 대해 처리
    for file_name in os.listdir(ass_folder_path):
        if file_name.endswith('.ass'):
            ass_file_path = os.path.join(ass_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, os.path.splitext(file_name)[0] + ".json")
            extract_specific_style_subtitles(ass_file_path, target_style, output_file_path)
            
