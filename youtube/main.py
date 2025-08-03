import os
import re
from datetime import timedelta
import yt_dlp
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
from moviepy.video.fx.all import resize, crop, mirror_x
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import streamlit as st
import tempfile

# Configure ImageMagick path (adjust if necessary for Streamlit Cloud)
IMAGEMAGICK_BINARY = os.getenv("IMAGEMAGICK_BINARY", "/usr/bin/convert")
if os.path.exists(IMAGEMAGICK_BINARY):
    change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})

class YouTubeShortGenerator:
    def __init__(self):
        self.video_url = ""
        self.video_id = ""
        self.video_title = "YouTube Video"
        self.video_length = 0
        self.output_folder = tempfile.mkdtemp()  # Use temporary directory for Streamlit Cloud
        self.max_short_length = 60  # Maximum duration for each short
        self.min_short_length = 30  # Minimum duration for each short
        self.engagement_data = []
        self.short_ratio = (9, 16)  # Width, Height for Shorts format
        self.transcript = []
        self.language = 'en'  # Default language (will be auto-detected)
        self.font_mapping = {
            'en': 'Arial-Bold',
            'hi': 'Mangal'  # Hindi font (should be installed on system)
        }
        # Load face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_video_info(self) -> bool:
        """Get video info using yt-dlp"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.video_url, download=False)
                self.video_title = info.get('title', 'YouTube Video')
                self.video_length = info.get('duration', 0)
                self.video_id = info.get('id', '')
                
                # Check if video is long enough
                if self.video_length < self.min_short_length:
                    st.error(f"Video is too short (must be at least {self.min_short_length} seconds)")
                    return False
                return True
        except Exception as e:
            st.error(f"Error getting video info: {e}")
            return False

    def download_video(self) -> str:
        """Download the video using yt-dlp"""
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            temp_path = os.path.join(self.output_folder, f"temp_{self.video_id}.mp4")
            
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': temp_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.video_url])
            return temp_path
        except Exception as e:
            st.error(f"Error downloading video: {e}")
            return ""

    def get_transcript(self) -> bool:
        """Fetch transcript with auto language detection"""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                self.language = 'en'
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript(['hi'])
                    self.language = 'hi'
                except NoTranscriptFound:
                    available = transcript_list._generated_transcripts or transcript_list._manually_created_transcripts
                    if available:
                        lang = list(available.keys())[0]
                        transcript = transcript_list.find_generated_transcript([lang])
                        self.language = lang
                    else:
                        st.warning("No transcript available for this video")
                        return False
            
            self.transcript = transcript.fetch()
            if self.transcript and not isinstance(self.transcript[0], dict):
                self.transcript = [{
                    'text': s.text,
                    'start': s.start,
                    'duration': s.duration
                } for s in self.transcript]
            
            st.info(f"Auto-detected transcript language: {self.language}")
            return True
        except TranscriptsDisabled:
            st.warning("Transcripts are disabled for this video")
            return False
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
            return False

    def analyze_engagement(self, video_path: str):
        """Analyze video to find most engaging segments of at least 30 seconds"""
        try:
            if not self.transcript:
                st.warning("No transcript available - creating segments of at least 30 seconds")
                segment_count = min(3, int(self.video_length // self.min_short_length))
                for i in range(segment_count):
                    start = i * (self.video_length / segment_count)
                    end = start + self.min_short_length
                    if end > self.video_length:
                        end = self.video_length
                        start = max(0, end - self.min_short_length)
                    self.engagement_data.append({
                        'start': start,
                        'end': end,
                        'score': 80
                    })
                return True

            segments = []
            current_segment = None
            
            for entry in self.transcript:
                start = entry['start']
                end = start + entry['duration']
                text = entry['text']
                word_count = len(text.split())
                density = word_count / entry['duration'] if entry['duration'] > 0 else 0
                
                if current_segment is None:
                    current_segment = {
                        'start': start,
                        'end': end,
                        'text': text,
                        'score': density * 10
                    }
                else:
                    potential_end = end
                    if potential_end - current_segment['start'] <= self.max_short_length:
                        current_segment['end'] = potential_end
                        current_segment['text'] += " " + text
                        current_segment['score'] += density * 10
                    else:
                        if current_segment['end'] - current_segment['start'] >= self.min_short_length:
                            segments.append(current_segment)
                        current_segment = {
                            'start': start,
                            'end': end,
                            'text': text,
                            'score': density * 10
                        }
            
            if current_segment and current_segment['end'] - current_segment['start'] >= self.min_short_length:
                segments.append(current_segment)

            segments.sort(key=lambda x: x['score'], reverse=True)
            self.engagement_data = segments[:3]
            
            if not self.engagement_data:
                st.warning("No valid segments from transcript - using default segments")
                return self.analyze_engagement(video_path)
            return True
        except Exception as e:
            st.error(f"Error analyzing engagement: {e}")
            return False

    def create_even_segments(self):
        """Create evenly spaced segments of 30 seconds each"""
        try:
            self.engagement_data = []
            segment_count = int(self.video_length // self.min_short_length)
            for i in range(segment_count):
                start = i * self.min_short_length
                end = start + self.min_short_length
                if end > self.video_length:
                    end = self.video_length
                    start = max(0, end - self.min_short_length)
                self.engagement_data.append({
                    'start': start,
                    'end': end,
                    'score': 80
                })
            return True
        except Exception as e:
            st.error(f"Error creating even segments: {e}")
            return False

    def convert_to_shorts_format(self, clip):
        """Convert the clip to 9:16 vertical format"""
        try:
            width, height = clip.size
            target_ratio = self.short_ratio[0] / self.short_ratio[1]
            if width/height > target_ratio:
                new_width = height * target_ratio
                x_center = width / 2
                return crop(clip, x1=x_center - new_width/2, x2=x_center + new_width/2)
            else:
                new_height = width / target_ratio
                y_center = height / 2
                return crop(clip, y1=y_center - new_height/2, y2=y_center + new_height/2)
        except Exception as e:
            st.error(f"Error converting to shorts format: {e}")
            return clip

    def detect_faces_in_frame(self, frame):
        """Detect faces in a video frame and return their positions"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            st.error(f"Error detecting faces: {e}")
            return []

    def should_flip(self, left_face, right_face):
        """Determine if we should flip based on face sizes"""
        return right_face[2] * right_face[3] > left_face[2] * left_face[3]

    def apply_two_face_flip(self, clip, start_time, duration):
        """Check for two faces and flip horizontally if needed"""
        try:
            sample_time = start_time + duration / 2
            frame = clip.get_frame(sample_time)
            faces = self.detect_faces_in_frame(frame)
            if len(faces) == 2:
                centers = [(x + w/2, (x, y, w, h)) for (x, y, w, h) in faces]
                centers.sort(key=lambda c: c[0])
                left_face = centers[0][1]
                right_face = centers[1][1]
                if self.should_flip(left_face, right_face):
                    st.info("Two faces detected - flipping video horizontally")
                    return mirror_x(clip)
            return clip
        except Exception as e:
            st.error(f"Error in two-face flip logic: {e}")
            return clip

    def get_safe_text_position(self, clip, start_time, duration):
        """Determine safe position for text overlay based on face detection"""
        try:
            sample_times = np.linspace(start_time, start_time + duration, num=3)
            face_positions = []
            for t in sample_times:
                frame = clip.get_frame(t)
                faces = self.detect_faces_in_frame(frame)
                for (x, y, w, h) in faces:
                    rel_x = (x + w/2) / clip.size[0]
                    rel_y = (y + h/2) / clip.size[1]
                    face_positions.append((rel_x, rel_y))
            if not face_positions:
                return 0.83
            avg_y = sum(pos[1] for pos in face_positions) / len(face_positions)
            text_y = max(0.7, min(0.9, avg_y - 0.15))
            return text_y
        except Exception as e:
            st.error(f"Error determining text position: {e}")
            return 0.83

    def add_transcript_overlay(self, clip, start_time: float):
        """Add real-time transcript overlay to the clip with Vizard.ai-style enhancements"""
        try:
            if not self.transcript:
                return clip
            clip_end = start_time + clip.duration
            relevant_segments = [
                seg for seg in self.transcript
                if seg['start'] < clip_end and (seg['start'] + seg['duration']) > start_time
            ]
            font = self.font_mapping.get(self.language, 'Arial-Bold')
            text_y_position = self.get_safe_text_position(clip, start_time, clip.duration)
            text_y_position = max(text_y_position, 0.75)
            subtitles = []
            for seg in relevant_segments:
                seg_start = max(0, seg['start'] - start_time)
                seg_end = min(clip.duration, (seg['start'] + seg['duration']) - start_time)
                words = seg['text'].split()
                word_duration = (seg_end - seg_start) / len(words) if words else 0
                for idx, word in enumerate(words):
                    word_start = seg_start + idx * word_duration
                    word_end = word_start + word_duration
                    try:
                        txt_clip = TextClip(
                            word,
                            fontsize=36,
                            color='yellow',
                            bg_color='rgba(0,0,0,0.5)',
                            size=(clip.size[0], None),
                            method='caption',
                            font=font,
                            align='center',
                            stroke_color='black',
                            stroke_width=1.5,
                            kerning=1
                        ).set_position(('center', text_y_position), relative=True
                        ).set_start(word_start
                        ).set_duration(word_end - word_start
                        ).fadein(0.2).fadeout(0.2)
                        subtitles.append(txt_clip)
                    except Exception as e:
                        st.error(f"Error creating text clip for word: {e}")
                        continue
            if subtitles:
                return CompositeVideoClip([clip] + subtitles)
            return clip
        except Exception as e:
            st.error(f"Error adding transcript overlay: {e}")
            return clip

    def generate_short(self, start_time: float, end_time: float, clip_num: int) -> str:
        """Generate a short from specified time range with transcript overlay"""
        try:
            video_path = os.path.join(self.output_folder, f"temp_{self.video_id}.mp4")
            if not os.path.exists(video_path):
                video_path = self.download_video()
                if not video_path:
                    return ""
            actual_start = max(0, start_time)
            actual_end = min(end_time, self.video_length)
            if actual_end - actual_start < self.min_short_length:
                actual_end = min(actual_start + self.min_short_length, self.video_length)
                if actual_end - actual_start < self.min_short_length:
                    actual_start = max(0, actual_end - self.min_short_length)
            st.info(f"Creating short from {actual_start:.1f}s to {actual_end:.1f}s (duration: {actual_end-actual_start:.1f}s)")
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclip(actual_start, actual_end)
                vertical_clip = self.convert_to_shorts_format(subclip)
                duration = actual_end - actual_start
                vertical_clip = self.apply_two_face_flip(vertical_clip, actual_start, duration)
                final_clip = self.add_transcript_overlay(vertical_clip, actual_start)
                output_path = os.path.join(self.output_folder, f"short_{self.video_id}_{clip_num}.mp4")
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    threads=8,
                    fps=30,
                    preset='ultrafast',
                    logger=None
                )
                return output_path
        except Exception as e:
            st.error(f"Error generating short: {e}")
            return ""

def main():
    st.title("YouTube Short Generator")
    st.markdown("Generate engaging shorts from YouTube videos with Vizard.ai-style transcript overlays. Supports auto language detection with a minimum 30-second duration.")
    
    generator = YouTubeShortGenerator()
    url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    choice = st.radio("Create shorts from:", 
                      ["Most engaging parts (requires transcript)", "Evenly divided 30-second segments"])
    
    if st.button("Generate Shorts"):
        if not url:
            st.error("Please enter a valid YouTube URL")
            return
        
        generator.video_url = url
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        if match:
            generator.video_id = match.group(1)
        else:
            st.error("Invalid YouTube URL")
            return
        
        if not generator.get_video_info():
            return
        
        st.markdown(f"**Processing:** {generator.video_title}")
        st.markdown(f"**Duration:** {timedelta(seconds=generator.video_length)}")
        
        generator.get_transcript()
        video_path = generator.download_video()
        if not video_path:
            return
        
        if choice == "Most engaging parts (requires transcript)":
            if not generator.analyze_engagement(video_path):
                return
        else:
            if not generator.create_even_segments():
                return
        
        success_count = 0
        with st.spinner("Generating shorts..."):
            for i, segment in enumerate(generator.engagement_data[:3], 1):
                st.markdown(f"Generating short {i} from {segment['start']:.1f}s to {segment['end']:.1f}s")
                output_path = generator.generate_short(segment['start'], segment['end'], i)
                if output_path:
                    st.success(f"Successfully created short {i}")
                    with open(output_path, "rb") as file:
                        st.download_button(f"Download Short {i}", file, file_name=f"short_{i}.mp4")
                    success_count += 1
        
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {e}")
        
        st.markdown(f"**Successfully created {success_count} shorts with Vizard.ai-style overlays**")

if __name__ == "__main__":
    main()
