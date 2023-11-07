from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("AppliedForce.mp4", 20, 30, targetname="video1.mp4")