import streamlit as st
import whisper
from moviepy.editor import VideoFileClip
from deep_translator import GoogleTranslator
import os
import tempfile
import textwrap

# --- Helper Functions ---

# Mengubah detik menjadi format SRT (HH:MM:SS,MMM)
def format_time(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

# Membuat konten SRT
def create_srt(segments, translated_text):
    srt_content = ""
    
    # Karena GoogleTranslator mengembalikan string tunggal, 
    # kita perlu memecahnya kembali berdasarkan segmen. 
    # Namun, karena ini demo, kita akan membuat SRT dari teks asli 
    # dan menyediakan terjemahan penuh.
    # UNTUK SRT TERJEMAHAN: Kami akan menerjemahkan setiap segmen
    
    for i, segment in enumerate(segments):
        start = format_time(segment['start'])
        end = format_time(segment['end'])
        
        # Terjemahkan per segmen untuk akurasi timing (ini memerlukan waktu lebih lama)
        translator = GoogleTranslator(source='auto', target='id')
        translated_segment_text = translator.translate(segment['text'])
        
        srt_content += f"{i + 1}\n"
        srt_content += f"{start} --> {end}\n"
        # Memastikan teks tidak terlalu panjang per baris
        srt_content += textwrap.fill(translated_segment_text, width=40) + "\n\n"
        
    return srt_content

# Fungsi untuk ekstrak audio dari video
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    # Gunakan tempfile untuk nama file yang unik
    audio_path = os.path.join(tempfile.gettempdir(), f"audio_{os.path.basename(video_path)}.mp3")
    video.audio.write_audiofile(audio_path, logger=None)
    return audio_path

# Memuat model Whisper (Menggunakan cache agar hanya dimuat sekali)
@st.cache_resource
def load_model():
    # Model 'base' cepat. Untuk akurasi bahasa Asia yang lebih baik, coba 'small' atau 'medium'.
    return whisper.load_model("base") 

# --- Streamlit UI ---

st.set_page_config(page_title="AI Subtitle Generator", page_icon="📝")

st.title("📝 Generator Subtitle AI (Open Source)")
st.markdown("Kami menggunakan **OpenAI Whisper** (transkripsi) dan **Google Translator** (terjemahan) untuk solusi multibahasa gratis.")

model = load_model()

uploaded_file = st.file_uploader("Pilih file MP4 atau MP3 (Maksimal 200MB di Streamlit Cloud)", type=["mp4", "mp3"])

if uploaded_file is not None:
    # Simpan file yang diunggah ke lokasi sementara
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        file_path = tfile.name
    
    st.video(uploaded_file) if "mp4" in uploaded_file.type else st.audio(uploaded_file)
    
    if st.button("Mulai Proses Transkripsi & Terjemahan"):
        with st.spinner('Sedang memproses... Ini akan memakan waktu 1-2 menit per 1 menit video.'):
            try:
                # 1. Ekstrak Audio
                st.info("Mengekstrak audio...")
                if uploaded_file.name.endswith('.mp4'):
                    audio_path = extract_audio(file_path)
                else:
                    audio_path = file_path

                # 2. Transkripsi
                st.info("Mentranskripsi audio (AI Whisper)...")
                result = model.transcribe(audio_path)
                detected_lang = result["language"]
                
                st.success(f"Transkripsi Selesai! Bahasa Asli: **{detected_lang}**")

                # 3. Generate Translated SRT
                st.info("Menerjemahkan per segmen & membuat file SRT...")
                
                # Gunakan st.progress() untuk visual feedback, karena ini proses terlama
                srt_content = create_srt(result['segments'], None) 
                
                st.success("File Subtitle Terjemahan Selesai Dibuat!")
                
                # --- TAMPILKAN HASIL & DOWNLOAD ---
                st.subheader("✅ Hasil Subtitle Terjemahan (Indonesia)")
                st.code(srt_content[:1000] + "...", language="text") 
                
                st.download_button(
                    '⬇️ Download Subtitle Terjemahan (.srt)',
                    srt_content,
                    file_name=f'subtitle_terjemahan_{os.path.basename(uploaded_file.name).split(".")[0]}.srt'
                )
                
                st.subheader("Teks Penuh Asli")
                st.text_area("Teks Asli", result["text"], height=150)


            except Exception as e:
                st.error(f"Terjadi kesalahan. Pastikan file tidak terlalu besar atau formatnya benar. Error: {e}")
            
            finally:
                # Membersihkan file temporary
                if os.path.exists(file_path):
                    os.remove(file_path)
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)
