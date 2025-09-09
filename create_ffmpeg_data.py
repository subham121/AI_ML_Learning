import pandas as pd
import re

def is_supported(value, supported_list):
    value = str(value).lower()
    for supported in supported_list:
        # Use word boundary or substring match
        if re.search(r'\b' + re.escape(supported.lower()) + r'\b', value) or supported.lower() in value:
            return True
    return False

def check_support(row):
    container = str(row['Container List']).lower() if pd.notnull(row['Container List']) else ''
    video = str(row['Video Codec']).lower() if pd.notnull(row['Video Codec']) else ''
    audio = str(row['Audio Codec']).lower() if pd.notnull(row['Audio Codec']) else ''
    comments = []
    supported = True
    if not is_supported(container, supported_containers):
        supported = False
        comments.append(f"container '{container}' not supported")
    if not is_supported(video, supported_video_codecs):
        supported = False
        comments.append(f"video codec '{video}' not supported")
    if not is_supported(audio, supported_audio_codecs):
        supported = False
        comments.append(f"audio codec '{audio}' not supported")
    if supported:
        return "Can be supported (supported container/video codec/audio codec)"
    else:
        return f"Cannot be supported ({', '.join(comments)})"
# Load your table (assuming columns: Container, Video Codec, Audio Codec, ...)
df = pd.read_excel('FFMPEG_Data.xlsx')  # or pd.read_csv('ffmpeg_test.csv')

supported_containers = [
    'aac', 'ac3', 'aiff', 'amr', 'ape', 'asf', 'ass', 'avi', 'bink', 'caf', 'dts', 'dtshd', 'dv', 'dxa', 'eac3', 'flac', 'flv', 'gsm', 'h261', 'h263', 'h264', 'hls', 'ircam', 'mjpeg', 'mov', 'mp3', 'mpeg2ps', 'mpeg2ts', 'mpeg4bs', 'ogg', 'rm', 'srt', 'swf', 'vc1', 'wav', 'webm', 'wtv', 'dash', 'smoothstream'
]
supported_video_codecs = [
    'h264', 'vc1', 'mpeg2', 'mpeg4', 'theora', 'vp8', 'vp9', 'hevc', 'dolbyvision', 'av1'
]
supported_audio_codecs = [
    'aac', 'mp3', 'pcm', 'vorbis', 'flac', 'amr_nb', 'amr_wb', 'pcm_mulaw', 'gsm_ms', 'pcm_s16be', 'pcm_s24be', 'opus', 'eac3', 'pcm_alaw', 'alac', 'ac3', 'mpeghaudio', 'dts', 'dtsx', 'dtse', 'ac4', 'iamf'
]
print(len(supported_containers))
print(len(supported_audio_codecs))
print(len(supported_video_codecs))
df['Support Status'] = df.apply(check_support, axis=1)
df.to_excel('ffmpeg_test_with_support_status.xlsx', index=False)