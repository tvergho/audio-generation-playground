import os
from pydub import AudioSegment

def convert_mp3_to_wav(directory):
    print(f"Converting directory {directory}")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            convert_mp3_to_wav(filepath)
        elif filename.endswith('.mp3'):
            try:
                mp3_audio = AudioSegment.from_file(filepath, format='mp3')
                wav_filename = filename.split('.')[0] + '.wav'
                mp3_audio.export("./wav/" + wav_filename, format='wav')
            except:
                pass
    print(f"Converted directory {directory}")

# Example usage: convert all mp3 files under the 'music' directory
convert_mp3_to_wav(os.path.join(os.getcwd(), './fma_small'))