from module import converter, UVR_del_bg, wav_slice_module, wav_filtering_module, embedding_module, clustering_module
import os
import shutil
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

substyle = None

gitkeep_files = glob.glob(os.path.join(".", "**", ".gitkeep"), recursive=True)

for file in gitkeep_files:
    if os.path.isfile(file):
        os.remove(file)

base_UVR_model_list = ["MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
                  "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                  ]
inst_UVR_model_list = ["UVR-BVE-4B_SN-44100-1.pth",
                       "MDX23C-8KFFT-InstVoc_HQ_2.ckpt"
                       ]

converter.convert_mp4_to_wav("./input/mp4", "./save/rawwav")
converter.convert_ass_to_json(substyle, "./input/ass", "./save/assjson")


input_folder = "./save/rawwav"

for model in base_UVR_model_list:
    folder_name = os.path.splitext(model)[0]
    model_folder_path = os.path.join("./save/uvrwav/base_uvr", folder_name)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path, exist_ok=True)

    stem = "Vocals"

    UVR_del_bg.UVR(model, stem, input_folder, model_folder_path)
    
    input_folder = model_folder_path

for filename in os.listdir(input_folder):
    src_file_path = os.path.join(input_folder, filename)

    if filename.endswith('.wav'):
        dst_file_path = os.path.join("./save/uvrwav/base_uvr", filename)
        shutil.copy2(src_file_path, dst_file_path) 

for item in os.listdir("./save/uvrwav/base_uvr"):
    item_path = os.path.join("./save/uvrwav/base_uvr", item)
    
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)


input_folder = "./save/uvrwav/base_uvr"

for model in inst_UVR_model_list:
    idx = inst_UVR_model_list.index(model)+1
    folder_name = f"inst_uvr_{idx}"
    model_folder_path = os.path.join("./save/uvrwav/inst_uvr", folder_name)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path, exist_ok=True)

    if idx == 2:
        stem = "Instrumental"
    elif idx == 1:
        stem = "Vocals"

    UVR_del_bg.UVR(model, stem, input_folder, model_folder_path)


wav_slice_module.find_matching_json("./save/uvrwav/base_uvr", "./save/assjson", "./save/slicewav/vocals", "./save/info/wav_info.json",'vocal')

for dir in os.listdir("./save/uvrwav/inst_uvr"):
    dir_path = os.path.join("./save/uvrwav/inst_uvr", dir)
    wav_slice_module.find_matching_json(dir_path, "./save/assjson", f"./save/slicewav/{dir}", "./save/info/wav_info.json",'inst')

for dir in os.listdir("./save/slicewav"):
    if dir == "vocals":
        pass
    else:
        input_path = os.path.join("./save/slicewav", dir)
        output_path = os.path.join("./save/spectrogram", dir)
        output_json_path = os.path.join("./save/info", f'spectrogram_{dir}.json')
        if dir == "inst_uvr_1":
            condition = [0.05, 10**6]
        elif dir == "inst_uvr_2":
            condition = [1.3, 90]
        wav_filtering_module.spectrogram_json(input_path, output_path, output_json_path, './save/slicewav/vocals', condition)

files = os.listdir("./save/info")
spectrogram_files = sorted([file for file in files if file.startswith("spectrogram") and os.path.isfile(os.path.join("./save/info", file))])
wav_filtering_module.find_and_filtering_files_based_on_json(spectrogram_files, "./save/slicewav/vocals")


output_pt_path = os.path.join("./save/info", "cosine_distance.pt")
embedding_module.embeddings("./save/slicewav/vocals","./save/info", output_pt_path)

embedding_path = os.path.join("./save/info", "all_embeddings.pt")
json_path = os.path.join("./save/info", "embedding_map.json")
clustering_module.clustering(output_pt_path, embedding_path, json_path, "./save/slicewav/vocals", "./output")

for folder_name in os.listdir("./output"):
    folder_path = os.path.join("./output", folder_name)
    if os.path.isdir(folder_path):
        if folder_name.startswith('clustering_'):

            new_name = folder_name.replace('clustering_', 'speaker_')

            old_path = os.path.join("./output", folder_name)
            new_path = os.path.join("./output", new_name)

            os.rename(old_path, new_path)

            wav_files = [f for f in os.listdir(new_path) if f.endswith('.wav')]

            for idx, wav_file in enumerate(sorted(wav_files), start=1):
                old_file_path = os.path.join(new_path, wav_file)
                new_file_name = f"{idx}.wav"
                new_file_path = os.path.join(new_path, new_file_name)
                
                os.rename(old_file_path, new_file_path)