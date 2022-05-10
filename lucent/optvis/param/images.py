# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""High-level wrapper for paramaterizing images."""

from __future__ import absolute_import, division, print_function

from lucent.optvis.param.spatial import pixel_image, fft_image, waveform_audio, stft_audio
from lucent.optvis.param.color import to_valid_rgb, to_waveform


import julius


def image(w, h=None, sd=None, batch=None, decorrelate=True,
          fft=True, channels=None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output


def audio(sample_length, sd=None, batch=None, stft=False, freqs=None):
    batch = batch or 1
    
    if stft:
        T = sample_length // freqs + 1
        shape = [batch, 1, freqs, T]
        # shape = [batch, 1, sample_length]
        params, audio_f = stft_audio(shape, sd=sd)
    else:
        shape = [batch, 1, sample_length]
        params, audio_f = waveform_audio(shape, sd=sd)
    output = to_waveform(audio_f)
    
    
    return params, output