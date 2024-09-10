from module import converter, UVR_del_bg, wav_slice_module, wav_filtering_module, embedding_module, clustering_module
import os
import shutil
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

substyle = "JPN TOP"

gitkeep_files = glob.glob(os.path.join(".", "**", ".gitkeep"), recursive=True)

for file in gitkeep_files:
    if os.path.isfile(file):
        os.remove(file)

base_UVR_model_list = [("MDX23C-8KFFT-InstVoc_HQ_2.ckpt","Vocals"),
                  ("model_bs_roformer_ep_317_sdr_12.9755.ckpt","Vocals"),
                  ("6_HP-Karaoke-UVR.pth","Vocals"),
                  ("UVR_MDXNET_KARA_2.onnx","Vocals")
                  ]
inst_UVR_model_list = [("MDX23C-8KFFT-InstVoc_HQ_2.ckpt","Instrumental"),
                  ("model_bs_roformer_ep_317_sdr_12.9755.ckpt","Instrumental"),
                  ("6_HP-Karaoke-UVR.pth","Instrumental"),
                  ("UVR_MDXNET_KARA_2.onnx","Instrumental"),
                       ("UVR-BVE-4B_SN-44100-1.pth","Vocals")
                       ]

converter.convert_mp4_to_wav("./input/mp4", "./save/rawwav")
converter.convert_ass_to_json(substyle, "./input/ass", "./save/assjson")

input_folder = "./save/rawwav"
output_dir = "./save/uvrwav/models"
ensemble_output_dir = "./save/uvrwav/base_uvr"

UVR_del_bg.UVR_ensemble(base_UVR_model_list, input_folder, output_dir, ensemble_output_dir)
    
input_folder = "./save/uvrwav/base_uvr"
output_dir = "./save/uvrwav/models"
ensemble_output_dir = "./save/uvrwav/inst_uvr"

UVR_del_bg.UVR_ensemble(inst_UVR_model_list, input_folder, output_dir, ensemble_output_dir)


wav_slice_module.find_matching_json("./save/uvrwav/base_uvr", "./save/assjson", "./save/slicewav/vocals", "./save/info/wav_info.json",'vocal')
wav_slice_module.find_matching_json("./save/uvrwav/inst_uvr", "./save/assjson", f"./save/slicewav/inst", "./save/info/wav_info.json",'inst')

for dir in os.listdir("./save/slicewav"):
    if dir == "vocals":
        pass
    else:
        input_path = os.path.join("./save/slicewav", dir)
        output_path = os.path.join("./save/spectrogram", dir)
        output_json_path = os.path.join("./save/info", f'spectrogram_{dir}.json')
        condition = [0.05, 10**6]
        wav_filtering_module.spectrogram_json(input_path, output_path, output_json_path, './save/slicewav/vocals', condition)

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