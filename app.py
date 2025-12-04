import gradio as gr
import whisper
from pydub import AudioSegment
from deep_translator import GoogleTranslator
import os
import tempfile
import textwrap

# --- Setup Global ---

# Memuat model Whisper (di luar fungsi untuk caching)
# Menggunakan 'base' untuk keseimbangan. Ganti ke 'small' atau 'medium' jika perlu akurasi lebih tinggi
MODEL = whisper.load_model("base") 

# --- Helper Functions ---

# Mengubah detik menjadi format SRT (HH:MM:SS,MMM)
def format_time(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

# Membuat konten SRT Terjemahan
def create_translated_srt(segments):
    srt_content = ""
    translator = GoogleTranslator(source='auto', target='id')
    
    for i, segment in enumerate(segments):
        start = format_time(segment['start'])
        end = format_time(segment['end'])
        
        # Menerjemahkan per segmen
        translated_segment_text = translator.translate(segment['text'])
        
        srt_content += f"{i + 1}\n"
        srt_content += f"{start} --> {end}\n"
        srt_content += textwrap.fill(translated_segment_text, width=45) + "\n\n"
        
    return srt_content

# Fungsi Utama Pemrosesan Audio/Video
def process_file(uploaded_file):
    if uploaded_file is None:
        return "Error: Harap unggah file MP4 atau MP3.", None, None

    file_path = uploaded_file
    file_extension = file_path.split('.')[-1].lower()
    
    temp_audio_path = None
    
    try:
        # 1. Ekstraksi Audio
        if file_extension == 'mp4':
            # Menggunakan pydub untuk ekstrak audio dari MP4
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio.mp3")
            audio = AudioSegment.from_file(file_path, format="mp4")
            audio.export(temp_audio_path, format="mp3")
            audio_to_process = temp_audio_path
        elif file_extension == 'mp3':
            audio_to_process = file_path
        else:
            return "Format file tidak didukung. Mohon upload MP4 atau MP3.", None, None

        # 2. Transkripsi (Whisper)
        result = MODEL.transcribe(audio_to_process)
        original_text = result["text"]
        detected_lang = result["language"]
        
        # 3. Terjemahan & SRT
        srt_content = create_translated_srt(result['segments'])
        
        # Output utama (Teks Penuh Terjemahan)
        translator_full = GoogleTranslator(source=detected_lang, target='id')
        translated_full_text = translator_full.translate(original_text)
        
        output_text = f"Bahasa Asli Terdeteksi: {detected_lang}\n\n--- Teks Asli ---\n{original_text}\n\n--- Terjemahan Penuh ---\n{translated_full_text}"

        # Menyimpan SRT ke file sementara untuk diunduh
        srt_file_path = os.path.join(tempfile.gettempdir(), "subtitle_terjemahan.srt")
        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        return output_text, srt_file_path, f"Subtitle Terjemahan (.srt) Berhasil Dibuat!"

    except Exception as e:
        return f"Terjadi Kesalahan Fatal pada AI: {e}", None, None
    finally:
        # Membersihkan file audio sementara (jika ada)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- Gradio Interface ---

# Input: File Upload (MP4 atau MP3)
file_input = gr.File(label="Upload File Video (MP4) atau Audio (MP3)", file_types=["video", "audio"])

# Output: 1. Teks Hasil, 2. Link Download SRT, 3. Status
text_output = gr.Textbox(label="Hasil Transkripsi dan Terjemahan Penuh")
file_output = gr.File(label="Download File Subtitle (.SRT)")
status_message = gr.Markdown("Status: Siap menerima file.")

# Membuat antarmuka Gradio
iface = gr.Interface(
    fn=process_file,
    inputs=file_input,
    outputs=[text_output, file_output, status_message],
    title="Open Source AI Subtitle Translator (Hugging Face Edition)",
    description="Unggah file MP4/MP3. AI Whisper akan mentranskripsi dan menerjemahkannya ke Bahasa Indonesia, lalu menyediakan file SRT untuk diunduh. (Gratis & Menggunakan AI terbaik)",
    allow_flagging="never"
)

# Jalankan server
if __name__ == "__main__":
    iface.launch()
