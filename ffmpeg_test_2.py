import pandas as pd

df = pd.read_excel('FFMPEG_Data.xlsx')

containers = set(df['Container List'].dropna().apply(lambda x: str(x).replace('.', '').lower()))
audio_codecs = set(df['Audio Codec'].dropna().apply(lambda x: str(x).split()[0].lower()))
video_codecs = set(df['Video Codec'].dropna().apply(lambda x: str(x).split()[0].lower()))
supported_containers = [
    'aac', 'ac3', 'aiff', 'amr', 'ape', 'asf', 'ass', 'avi', 'bink', 'caf', 'dts', 'dtshd', 'dv', 'dxa', 'eac3', 'flac', 'flv', 'gsm', 'h261', 'h263', 'h264', 'hls', 'ircam', 'mjpeg', 'mov', 'mp3', 'mpeg2ps', 'mpeg2ts', 'mpeg4bs', 'ogg', 'rm', 'srt', 'swf', 'vc1', 'wav', 'webm', 'wtv', 'dash', 'smoothstream'
]
supported_video_codecs = [
    'h264', 'vc1', 'mpeg2', 'mpeg4', 'theora', 'vp8', 'vp9', 'hevc', 'dolbyvision', 'av1'
]
supported_audio_codecs = [
    'aac', 'mp3', 'pcm', 'vorbis', 'flac', 'amr_nb', 'amr_wb', 'pcm_mulaw', 'gsm_ms', 'pcm_s16be', 'pcm_s24be', 'opus', 'eac3', 'pcm_alaw', 'alac', 'ac3', 'mpeghaudio', 'dts', 'dtsx', 'dtse', 'ac4', 'iamf'
]
print("Containers in file:", len(containers), containers)
print("Audio codecs in file:", audio_codecs)
print("Video codecs in file:", video_codecs)


unsupported_containers = [c for c in containers if c not in supported_containers]
unsupported_video_codecs = [v for v in video_codecs if v not in supported_video_codecs]
unsupported_audio_codecs = [a for a in audio_codecs if a not in supported_audio_codecs]
print('\n')
print('\n')
print("Unsupported Containers:", len(unsupported_containers),unsupported_containers)
print("Unsupported Video Codecs:", unsupported_video_codecs)
print("Unsupported Audio Codecs:", unsupported_audio_codecs)

print('\n')
print('\n')
containers_to_do = [c for c in containers if c in supported_containers]
video_codecs_to_do = [v for v in video_codecs if v in supported_video_codecs]
audio_codecs_to_do = [a for a in audio_codecs if a in supported_audio_codecs]

print("Supported Containers:", len(containers_to_do),containers_to_do)
print("Supported Video Codecs:", video_codecs_to_do)
print("Supported Audio Codecs:", audio_codecs_to_do)