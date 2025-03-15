import pydub, os, re
import speech_recognition as sr

audio_dir = "store/audios"
chunk_dir = "store/chunks"

def convert_AudioToText(
                        src_path
                        ):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(src_path) as source:
            audio_text = r.record(source)
            text = r.recognize_google(audio_text)
            # print('Converting audio transcripts into text ...')
            return text
        
    except Exception as e:
        # print("Error in convert_AudioToText : ", e)
        return ' '
    
def match_target_amplitude(aChunk, target_dBFS):
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def detect_scilences_inAudio(
                            src_path,
                            dest_dir,
                            min_silence_len = 500
                            ):
    
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

        try:
            audio = pydub.AudioSegment.from_mp3(src_path)

            chunks = pydub.silence.split_on_silence (
                                                    audio,
                                                    min_silence_len = min_silence_len,
                                                    silence_thresh = audio.dBFS-14
                                                    )
            full_text = ""
            for i, chunk in enumerate(chunks):
                silence_chunk = pydub.AudioSegment.silent(duration=500)       # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
                audio_chunk = silence_chunk + chunk + silence_chunk           # Add the padding chunk to beginning and end of the entire chunk.
                normalized_chunk = match_target_amplitude(audio_chunk, -20.0) # Normalize the entire chunk.

                print(f"Exporting chunk{i}.wav")                    # Export the audio chunk with new bitrate.
                normalized_chunk.export(
                                        f"{dest_dir}/chunk{i}.wav",
                                        bitrate = "192k",
                                        format = "wav"
                                        )
                text = convert_AudioToText(f"{dest_dir}/chunk{i}.wav")
                full_text += text + ". "
                
            print("Completed the Separates !")
            return full_text

        except Exception as e:
            print("Error in detect_scilences_inAudio : ", e)
            return None
        
    else:
        print("Document already Chunked !")
        full_text = ""
        for i, file in enumerate(os.listdir(dest_dir)):
            text = convert_AudioToText(f"{dest_dir}/{file}")
            full_text += text + ". "
        return full_text
        
def end_to_end_audio_to_text(audio_file):
    audio_file = audio_file.replace("\\", "/")
    audio_name = audio_file.split("/")[-1].split(".")[0]
    dest_dir = os.path.join(chunk_dir, audio_name)
    dest_dir = dest_dir.replace("\\", "/")
    full_text = detect_scilences_inAudio(audio_file, dest_dir)

    if full_text is None:
        print("Silence Detection Failed !")
        return False
    
    full_text = re.sub(' +', ' ', full_text)
    full_text = re.sub('\.   +', '.', full_text)
    full_text = re.sub('\.  +', '.', full_text)
    full_text = re.sub('\. +', '.', full_text)
    full_text = re.sub('\.+', '.', full_text)
    full_text = re.sub('\.', ' . ', full_text)
    return full_text