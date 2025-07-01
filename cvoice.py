import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('/home/system/CosyVoice')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice('/home/system/CosyVoice/pretrained_models/CosyVoice-300M', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('trump.wav', 16000)

for i, j in enumerate(cosyvoice.inference_zero_shot('Hey babe how are you doing, so can you like umm... help me write a birthday message for my sister? I\'m terrible with words and want to say something meaningful. but i keep drawing a blank.', 'slowly', prompt_speech_16k, stream=False)):
    torchaudio.save('cvoice_zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

"""
for i, j in enumerate(cosyvoice.inference_cross_lingual('Hey babe how are you doing, so can you like umm... help me write a birthday message for my sister? I\'m terrible with words and want to say something meaningful. but i keep drawing a blank.', prompt_speech_16k, stream=False)):
    torchaudio.save('cvoice_cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

source_speech_16k = load_wav('girl.wav', 16000)
for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
    torchaudio.save('cvoice_vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('../CosyVoice/pretrained_models/CosyVoice-300M-Instruct', load_jit=False, load_trt=False, fp16=False)
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct('<strong>Hey babe</strong> how are you doing, so can you like umm... [laughter] help me write a birthday message for my sister? I\'m terrible with words and want to say something meaningful. but i keep drawing a blank.', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
    torchaudio.save('cvoice_instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
"""