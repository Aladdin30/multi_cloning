import whisperx
import gc
import torch
from pydub import AudioSegment
from TTS.api import TTS
from googletrans import Translator
print(torch.cuda.is_available())  # to check if cuda available or not; will return True if available
model = whisperx.load_model("large-v3", "cuda", compute_type="float16")# TO LOAD YOUR VERSION MODEL FROM WHISPERX
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
language={
            "english" :'en',
            "spanish" :'es',
            "french" :'fr',
            "german": 'de',
            "italian" :'it',
            "portuguese":'pt',
            "polish": 'pl',
            "turkish" :'tr',
            "russian" :"ru",
            "dutch": "nl",
            "czech": "cs",
            "arabic" :'ar',
            "chinese": 'cn',
            "japanese" :"ja",
            "hungarian": 'hu',
            "korean" :'ko',
            "hindi": 'hi'
        }
def check_lang(input_lang):
    lang = None  # Default value
    for key, value in language.items():
        if input_lang.lower() == key:
            lang = value
            break
    return lang
def WhisperX(audio_file):# TO TRANSCRIBE AUDIO FILE
    batch_size = 4
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print(result)
    return result, audio
def diarizer(result,audio,numper_of_speaker,target_lang):
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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",device=device)
    diarize_segments = diarize_model(audio, min_speakers=numper_of_speaker, max_speakers=numper_of_speaker)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    unique_speakers_array=diarize_segments.speaker.unique()
    speaker_name = unique_speakers_array.tolist()
    # for multi language use tis code
    speaker_text_pairs=[]
    for segment in result["segments"]:
        translator = Translator()
        translated_text = translator.translate(segment["text"], dest=target_lang)
        speaker_text_pair = (segment["speaker"], translated_text.text)
        speaker_text_pairs.append(speaker_text_pair)
    # Print the result
    print(speaker_text_pairs)
    #speaker_text_pairs = [(segment["speaker"], segment["text"]) for segment in result["segments"]]
    return speaker_text_pairs,speaker_name
def get_speaker_input(speaker_name):
    num_users=len(speaker_name)
    speakers = {}
    for i in range(num_users):
        speaker_num = str(i+1-1).zfill(2)  # Convert to 2-digit string
        speaker_name = f"SPEAKER_{speaker_num}"
        user_input = input(f"Enter input for {speaker_name}: ")
        speakers[speaker_name] = user_input
        i+=1
    return speakers
def multi_user(speaker_text_pairs,speakers,target_lang):
    combined_audio = AudioSegment.silent(duration=0)
    data=speaker_text_pairs
    speakers=speakers
    for speaker, text in data:
        print(text)
        print(speaker)
        tts.tts_to_file(text=text, speaker_wav=speakers[speaker], language=target_lang, file_path="output.wav")
        audio = AudioSegment.from_file("output.wav")
        combined_audio += audio
    All_Audio=combined_audio.export("combined_output.wav", format="wav")
    return All_Audio
def all(audio_file,numper_of_speaker,target_lang):
    lang=check_lang(target_lang)
    result,audio=WhisperX(audio_file)
    speaker_text_pairs,speaker_name=diarizer(result,audio,numper_of_speaker,lang)
    speakers=get_speaker_input(speaker_name)
    All_Audio=multi_user(speaker_text_pairs,speakers,lang)
    return All_Audio
audio_file = "shafic.wav"
numper_of_speaker=1
target_lang="english"
x=all(audio_file,numper_of_speaker,target_lang)
print(x)
