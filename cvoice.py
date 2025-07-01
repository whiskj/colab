import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=False, fp16=True)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('/run/media/mesh/Music/Asmongold reacted to my video..short_1.wav', 16000)

for i, j in enumerate(cosyvoice.inference_cross_lingual('Hey babe how are you doing, so can you like umm... help me write a birthday message for my sister? I\'m terrible with words and want to say something meaningful. but i keep drawing a blank.', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_0.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
