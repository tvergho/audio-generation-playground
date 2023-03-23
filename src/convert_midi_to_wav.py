import os
import multiprocessing
from midi2audio import FluidSynth

def convert_midi_to_wav(midi_file_path, output_dir):
    # Load the MIDI file and initialize the FluidSynth converter
    fluidsynth = FluidSynth()
    # Convert the MIDI file to a WAV file
    wav_file_path = os.path.splitext(os.path.basename(midi_file_path))[0] + '.wav'
    wav_file_path = os.path.join(output_dir, wav_file_path)
    fluidsynth.midi_to_audio(midi_file_path, wav_file_path)

def convert_all_midi_to_wav(input_dir, output_dir):
    # Iterate through all the files in the input directory
    processes = []
    for filename in os.listdir(input_dir):
        input_filepath = os.path.join(input_dir, filename)
        # If the file is a MIDI file, spawn a new process to convert it to a WAV file
        if os.path.isfile(input_filepath) and input_filepath.lower().endswith('.mid'):
            print(f"Converting {filename}...")
            p = multiprocessing.Process(target=convert_midi_to_wav, args=(input_filepath, output_dir))
            p.start()
            processes.append(p)
        # If the file is a directory, recurse into it
        elif os.path.isdir(input_filepath):
            sub_output_dir = os.path.join(output_dir, filename)
            os.makedirs(sub_output_dir, exist_ok=True)
            convert_all_midi_to_wav(input_filepath, sub_output_dir)
    # Wait for all the processes to finish
    for p in processes:
        p.join()

convert_all_midi_to_wav('/workspace/adl-piano-midi/World', '/workspace/adl-data')