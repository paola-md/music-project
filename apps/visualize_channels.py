import sys
sys.path.append("../")

import streamlit as st

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

st.title("Visualizing Behavior of Channels in JukeBox Encoder")


available_models = {
    "5b",
    "1b_lyrics",
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache()
def setup_dist(dummy=False):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi()
    

@st.cache(ttl=None, allow_output_mutation=True, max_entries=3)
def load_pretrained_model(model_name):
    from jukebox.make_models import make_vqvae, MODELS
    from jukebox.hparams import setup_hparams
    
    vqvae, *priors = MODELS[model_name]
    hparams = setup_hparams(vqvae, dict(sample_length = 1048576))
    vqvae = make_vqvae(hparams, DEVICE)
    vqvae = vqvae.eval()
    
    # from lucent.optvis.render import hook_model
    # hook = hook_model(model, None)
    return vqvae


chosen_model_name = st.sidebar.selectbox(
    "Choose the JukeBox VQVAE Model",
    list(available_models)
)
setup_dist()
chosen_model = load_pretrained_model(chosen_model_name)
chosen_model = chosen_model.to(DEVICE)


import scipy
from lucent.optvis.render import hook_model
from lucent.modelzoo.util import get_model_layers
import matplotlib.pyplot as plt


from lucent.modelzoo.util import get_model_layers

available_encoders = ['8x', '32x', '128x']
encoder_type = st.sidebar.radio("Choose encoder resolution", available_encoders)

encoder_model = chosen_model.encoders[available_encoders.index(encoder_type)]
encoder_model.eval()

model_layers = get_model_layers(encoder_model)
hook, _ = hook_model(encoder_model, None)


sample_rate = int(st.sidebar.text_input('Sample Rate', value='44100'))
num_seconds = int(st.sidebar.text_input('Number of seconds', value='10'))
sweep_type = st.sidebar.radio("Choose frequency sweep type", ["Linear", "Chromatic", "Major"])

if sweep_type == 'Linear':
    t = np.linspace(0, num_seconds, sample_rate * num_seconds)
    f_start = int(st.sidebar.text_input('Frequency Sweep f_start', value='80'))
    f_end = int(st.sidebar.text_input('Frequency Sweep f_end', value='4000'))
    freqs = np.linspace(f_start, f_end, num_seconds * sample_rate)
    wave = scipy.signal.chirp(t, f_start, num_seconds, f_end) # Frequency sweep
elif sweep_type == 'Chromatic':
    f_start = int(st.sidebar.text_input('Frequency Sweep f_start', value='80'))
    num_octaves = int(st.sidebar.text_input('Number of octaves', value='3'))
    num_half_steps = 12 * num_octaves + 1
    wave = []
    N = num_seconds * sample_rate // num_half_steps
    delta_t = num_seconds / num_half_steps
    for i in range(num_half_steps):
        f = f_start * 2 ** (i / 12)
        k = int(delta_t * f)
        t = np.linspace(0, delta_t, N)
        t[t > k / f] = 0
        w = np.sin(2 * f * np.pi * t)
        wave.append(w)
        
    wave = np.concatenate(wave, axis=0)
elif sweep_type == 'Major':
    f_start = int(st.sidebar.text_input('Frequency Sweep f_start', value='80'))
    num_octaves = int(st.sidebar.text_input('Number of octaves', value='3'))
    num_half_steps = 12 * num_octaves + 1
    num_notes = 7 * num_octaves + 1
    wave = []
    N = num_seconds * sample_rate // num_notes
    delta_t = num_seconds / num_half_steps
    
    major_intervals = [2, 2, 1, 2, 2, 2, 1]
    
    for i in range(num_notes):
        j = i % 7
        h = sum(major_intervals[:j]) + 12 * (i // 7)
        f = f_start * 2 ** (h / 12)
        k = int(delta_t * f)
        t = np.linspace(0, delta_t, N)
        t[t > k / f] = 0
        w = np.sin(2 * f * np.pi * t)
        wave.append(w)
        
    wave = np.concatenate(wave, axis=0)
    
else:
    pass
    
    

with torch.no_grad():
    x = torch.from_numpy(wave).to(DEVICE).float()
    x = x[None, None, :]
    out = encoder_model(x)
    
available_layers = list(get_model_layers(encoder_model))
chosen_layer = st.sidebar.selectbox(
    "Choose the layer you want to visualize",
    available_layers
)
activations = hook(chosen_layer)
b, c, T = activations.shape


channel_idx = int(st.sidebar.text_input('Channel Index', value='0'))
channel_activation = activations[0, channel_idx, :]
channel_activation = channel_activation.detach().cpu().numpy()

log_frequency = st.sidebar.checkbox("Log frequency axis")

fig, ax = plt.subplots()

if sweep_type == 'Linear':
    freqs = np.linspace(f_start, f_end, T)
else:
    freqs = np.linspace(f_start, f_start * 2 ** (num_half_steps / 12), T)



channel_sample_rate = len(channel_activation) // num_seconds
ax.plot(freqs, channel_activation, label='Channel Activation')
layer_idx = available_layers.index(chosen_layer)
ax.set_title(f"Layer {layer_idx}, Channel {channel_idx}")

ax.set_ylabel("Activation")
ax.set_xlabel("Frequency")
if log_frequency:
    ax.set_xscale('log')
    
ax.vlines([channel_sample_rate / 2], colors=['red'], label='Nyquist Frequency', 
          ymin=channel_activation.min(), ymax=channel_activation.max())
plt.legend()
st.pyplot(fig)

import soundfile
import nussl
import io
channel_activation = (channel_activation - channel_activation.min())
channel_activation = channel_activation / channel_activation.max()
# channel_activation = channel_activation * 2 - 1
input_signal = nussl.AudioSignal(audio_data_array=wave, sample_rate=sample_rate)
output_signal = nussl.AudioSignal(audio_data_array=channel_activation, sample_rate=channel_sample_rate)

fig = plt.figure(figsize=(10, 5))
plt.title("Input Signal Mel Spectogram")
nussl.utils.visualize_spectrogram(input_signal, y_axis='log')
st.pyplot(fig)

fig = plt.figure(figsize=(10, 5))
plt.title("Channel Activation Mel Spectogram")
nussl.utils.visualize_spectrogram(output_signal, y_axis='log')
st.pyplot(fig)

print(channel_sample_rate, output_signal.sample_rate, output_signal.signal_duration)

# byte_io = io.BytesIO()

# signal = nussl.AudioSignal(audio_data_array=channel_activation, sample_rate=channel_sample_rate)
# signal.write_audio_to_file(byte_io)
# st.audio(byte_io)

# sub = 'PCM_32'  # could be 'PCM_32' or 'FLOAT'
# soundfile.write(byte_io, channel_activation, channel_sample_rate, subtype=sub, format='WAV')


# st.audio(channel_activation, format='audio/wave')

    
