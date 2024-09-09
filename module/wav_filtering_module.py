from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import os
import json
import librosa
import librosa.display
import shutil

def array_to_frequency_dict(array):

    flat_array = array.flatten()

    frequency_dict = dict(Counter(flat_array))

    return frequency_dict

def extract_and_save_lightness(image_path):

    image = Image.open(image_path)
    image = image.convert("RGB")

    image_array = np.array(image)

    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness = image_lab[:, :, 0]

    return lightness

def check_and_create_json(json_path):

    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)
    else:
        pass

def process_function(png_file_path):
    
    lightness_values = extract_and_save_lightness(png_file_path)
    counter = dict(sorted(array_to_frequency_dict(lightness_values).items()))
    per = float(sum([key*value for key,value in counter.items()])/sum(counter.values()))
    max = float(np.max(lightness_values))

    return list((per,max))

def process_png_files(directory, json_path, process_function):

    check_and_create_json(json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            file_base_name = os.path.splitext(filename)[0]

            result = process_function(os.path.join(directory, filename))

            data[file_base_name] = result

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)



def compute_global_min_max(file_list, sr=22050, n_mels=128, hop_length=512, fmax=8000):
    global_min = np.inf
    global_max = -np.inf

    for file in file_list:
        y, sr = librosa.load(file, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        
        if np.min(S) < global_min:
            global_min = np.min(S)
        if np.max(S) > global_max:
            global_max = np.max(S)
    
    return global_min, global_max

def spectogram(audio_path, output_folder, img_filename, global_min, global_max):

    y, sr = librosa.load(audio_path, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    DB = librosa.power_to_db(S, ref = global_min)

    plt.figure(figsize=(12, 8), dpi=100)
    librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log', cmap= 'inferno', vmin=0, vmax=10*(np.log10(global_max/global_min)))

    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])
    plt.xlabel('') 
    plt.ylabel('')
    plt.title('')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.gca().patch.set_alpha(0)
    plt.axis('off')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    temp_output_path = os.path.join(output_folder, f"{img_filename}_spectrogram_temp.png")
    plt.savefig(temp_output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close() 

    final_output_path = os.path.join(output_folder, f"{img_filename}.png")
    trim_white_border(temp_output_path, final_output_path)

    os.remove(temp_output_path)

def trim_white_border(image_path, output_path):
    image = Image.open(image_path)

    image = image.convert("RGBA")
    bbox = image.getbbox()
    trimmed_image = image.crop(bbox)
    
    trimmed_image.save(output_path)



def load_json(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def find_and_filtering_files_based_on_json(json_path_list, source_dir):

    json_path_1 = json_path_list[0]
    json_path_2 = json_path_list[1]
    
    data_1 = load_json(json_path_1)
    data_2 = load_json(json_path_2)

    for key in data_1.keys():
        source_file_path = os.path.join(source_dir, key)
        value_1 = data_1[key]
        value_2 = data_2[key]
        if os.path.exists(source_file_path):
            if value_1[0] <= 1.3 and value_1[1]<=90 and value_2[0]<0.05:
                pass
            else:
                os.remove(source_file_path)
            
        else:
            pass


def spectrogram_json(input_folder, output_folder, output_json_path):

    global_min, global_max = compute_global_min_max([os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.wav')])
    if not global_min:
        global_min = 10**-3

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_path}")
            spectogram(file_path, output_folder, file_name, global_min, global_max)

    process_png_files(output_folder, output_json_path, process_function)

    shutil.rmtree(output_folder)
