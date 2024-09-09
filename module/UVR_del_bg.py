from audio_separator.separator import Separator
import os

def UVR(model, stem, input_dir, output_dir):

    separator = Separator(output_dir = output_dir, output_single_stem = stem)

    separator.load_model(model_filename=model)

    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):  # .wav 파일만 처리
            file_path = os.path.join(input_dir, filename)
            output_filename = separator.separate(file_path)[0]
            output_file_path = os.path.join(output_dir, output_filename)
            rename_file_path = os.path.join(output_dir, filename)

            os.rename(output_file_path, rename_file_path)
