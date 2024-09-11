# AniTTS-Builder

## About

- Summary
  
  This program processes anime videos and subtitles to create Text-to-Speech (TTS) datasets. It extracts and cleans the audio by removing background noise, then slices it into smaller segments. Finally, the program clusters the audio by speaker for easy organization, streamlining the creation of speaker-specific TTS datasets.

  This program operates based on the models from Audio-separator and Speechbrain.

- Developer
  - N01N9

## Installation

This project is developed for a Windows environment. FFmpeg must be installed. Please install CUDA 12.X + CUDNN 9.X versions. If you use a lower version of CUDA, some models may not be able to utilize CUDA.
