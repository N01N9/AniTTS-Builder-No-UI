# AniTTS-Builder

## About

- Summary
  
  This program processes anime videos and subtitles to create Text-to-Speech (TTS) datasets. It extracts and cleans the audio by removing background noise, then slices it into smaller segments. Finally, the program clusters the audio by speaker for easy organization, streamlining the creation of speaker-specific TTS datasets.

  This program operates based on the models from Audio-separator and Speechbrain.

- Developer
  - N01N9

## Installation

This project is developed for a Windows environment. FFmpeg must be installed. Please install CUDA 12.X + CUDNN 9.X versions. If you use a lower version of CUDA, some models may not be able to utilize CUDA.

1. Install Anaconda, FFmpeg, and CUDA (version 11.8 or 12.1) along with CUDNN, and ensure the environment variables are set up correctly.
   
2. Clone the repositories using the following command.
   
3. Import the conda environment that matches the installed CUDA version.
   
4. Open the project folder using conda to verify that the environment has been successfully imported and the project has been cloned properly.

## Usage

To run this program, you will need an .mp4 file of an anime featuring the character with the voice you desire, as well as a .ass subtitle file that is synced with the .mp4 file. To gather sufficient data, you will need at least one season of anime (approximately 12 episodes, 20 minutes each).

1. Match the filenames of the anime .mp4 file and the .ass subtitle file for each episode. (e.g., animename_epX.mp4 and animename_epX.ass)

2. Place the anime .mp4 files in the ./input/mp4 folder and the anime .ass files in the ./input/ass folder.

3. The .ass file is used to check the timeline of the character's dialogue. Therefore, open the .ass file and copy the subtitle style corresponding to the character's dialogue. Most anime subtitles distinguish between different languages or between background music and dialogue using specific styles.
