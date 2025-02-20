import os
import librosa
import numpy as np
from mytypes import classes

from pydub import AudioSegment
from pydub.utils import which

# Set the FFmpeg path
AudioSegment.converter = which("ffmpeg")


def convert_m4a_to_wav(m4a_file_path, wav_file_path):
    # Load the m4a file
    audio = AudioSegment.from_file(m4a_file_path)
    # Export the audio as a wav file
    audio.export(wav_file_path, format="wav")

# Example usage:
# convert_m4a_to_wav("input.m4a", "output.wav")


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf 
model = tf.keras.models.load_model('model\Trained_model.h5')
print ("Data Preprocessing is ready")

def load_and_preprocess(file_name, data_dir, target_shape=(150, 150)):
  new_file_name = file_name.split('.')[0] + '.wav'
  print("file_name: ", file_name)
  print("new_file_name: ", new_file_name)
  print(os.path.join(data_dir, file_name), os.path.join(data_dir, new_file_name))

  convert_m4a_to_wav("./audio_files/record_audio.m4a", os.path.join(data_dir, new_file_name))
  data=[]
  if(True):
    file_path = os.path.join(data_dir, new_file_name)
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 4
    overlap_duration = 2

    overlap_sample_size = overlap_duration*sample_rate
    chunk_sample_size = chunk_duration*sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_sample_size) / (chunk_sample_size - overlap_sample_size))) + 1
    for i in range(num_chunks):
      start = i * (chunk_sample_size-overlap_sample_size)
      end = start+chunk_sample_size

      chunk = audio_data[start:end]

      mel_spectrogram = librosa.feature.melspectrogram(y=chunk,sr=sample_rate)

      mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis = -1), target_shape)

      data.append(mel_spectrogram)
      
  return np.array(data)

def get_output_label(data_dir, file_name):
  model = tf.keras.models.load_model('model\Trained_model.h5')
  # try:
  data = load_and_preprocess(file_name=file_name, data_dir= data_dir)
  # except Exception as e:
  #   print(f"Error loading and preprocessing the data: {str(e)}")
  #   return e
  try:
    y_pred = model.predict(data)
  except Exception as e:
    print(f"Error predicting the data: {str(e)}")
    return "Error processing the prediction"
  try:
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elemnts, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    output_label = unique_elemnts[counts==max_count][0]
    return classes[output_label]
  except Exception as e:
    print(f"Error processing the prediction: {str(e)}")
    return "Error processing the prediction"