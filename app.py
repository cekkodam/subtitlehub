import streamlit as st
import whisper
from pydub import AudioSegment # Pustaka yang diubah: Pydub
from deep_translator import GoogleTranslator
import os
import tempfile
import textwrap

# Konfigurasi Halaman
st.set_page_config(page_title="AI Video/Audio Translator (Pydub)", page_icon="📝")

st.title("📝 Generator Subtitle AI (Versi Stabil)")
st.markdown("Menggunakan **OpenAI Whisper** dan **Pydub** untuk pemrosesan video/audio yang stabil.")

# --- Helper Functions ---

# Mengubah detik menjadi format SRT (HH:MM:SS,MMM)
def format_time(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

# Fungsi untuk ekstrak audio dari video (menggunakan pydub)
def extract_audio(video_path):
    st.info("Mengekstrak audio dari video menggunakan pydub...")
    
    # Menentukan format input
    input_format = video_path.split('.')[-1]
    
    # Load file
    audio = AudioSegment.from_file(video_path, format=input_format) 

    # Membuat path sementara untuk output MP3
    audio_path = os.path.join(tempfile.gettempdir(), f"audio_pydub_{os.path.basename(video_path)}.mp3")
    
    # Export ke format MP3
    audio.export(audio_path, format="mp3")
    return audio_path

# Membuat konten SRT Terjemahan
def create_srt(segments):
    srt_content = ""
    translator = GoogleTranslator(source='auto', target='id')
    
    for i, segment in enumerate(segments):
        start = format_time(segment['start'])
        end = format_time(segment['end'])
        
        # Menerjemahkan per segmen
        translated_segment_text = translator.translate(segment['text'])
        
        srt_content += f"{i + 1}\n"
        srt_content += f"{start} --> {end}\n"
        # Memastikan teks tidak terlalu panjang per baris
        srt_content += textwrap.fill(translated_segment_text, width=45) + "\n\n"
        
    return srt_content

# Memuat model Whisper
@st.cache_resource
def load_model():
    # Menggunakan model 'base' untuk keseimbangan kecepatan dan akurasi
    return whisper.load_model("base") 

model = load_model()

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Pilih file video (MP4) atau audio (MP3)", type=["mp4", "mp3"])

if uploaded_file is not None:
    # Simpan file yang diunggah ke lokasi sementara
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tfile:
        tfile.write(uploaded_file.read())
        file_path = tfile.name
    
    st.video(uploaded_file) if file_extension == 'mp4' else st.audio(uploaded_file)
    
    if st.button("Mulai Proses Transkripsi & Terjemahan"):
        # Variabel audio_path harus didefinisikan di luar try/except agar bisa diakses di finally
        audio_path = None
        
        with st.spinner('Sedang memproses... Harap tunggu (tergantung durasi file)'):
            try:
                # 1. Cek tipe file & Ekstrak Audio jika perlu
                
                if file_extension == 'mp4':
                    audio_path = extract_audio(file_path)
                elif file_extension == 'mp3':
                    # Jika sudah MP3, gunakan path file aslinya
                    audio_path = file_path
                else:
                    st.error("Format file tidak didukung. Mohon upload MP4 atau MP3.")
                    # Mengganti 'return' di sini dengan raise error untuk menghindari SyntaxError
                    raise ValueError("Unsupported file type") 

                # 2. Transkripsi
                st.info("Mentranskripsi audio (AI Whisper)...")
                result = model.transcribe(audio_path)
                detected_lang = result["language"]
                
                st.success(f"Transkripsi Selesai! Bahasa Asli: **{detected_lang}**")

                # 3. Generate Translated SRT
                st.info("Menerjemahkan per segmen & membuat file SRT...")
                srt_content = create_srt(result['segments']) 
                
                st.success("File Subtitle Terjemahan Selesai Dibuat!")
                
                # --- TAMPILKAN HASIL & DOWNLOAD ---
                st.subheader("✅ Hasil Subtitle Terjemahan (Indonesia)")
                st.code(srt_content[:1500] + "\n...", language="text") 
                
                st.download_button(
                    '⬇️ Download Subtitle Terjemahan (.srt)',
                    srt_content,
                    file_name=f'subtitle_terjemahan_{os.path.basename(uploaded_file.name).split(".")[0]}.srt'
                )
                
                st.subheader("Teks Penuh Asli")
                st.text_area("Teks Asli", result["text"], height=150)


            except ValueError as ve:
                # Menangkap error jika format file tidak didukung
                if str(ve) != "Unsupported file type":
                     st.error(f"Terjadi kesalahan pemrosesan: {ve}")
            except Exception as e:
                # Menangkap error umum lainnya
                st.error(f"Terjadi kesalahan: {e}. Pastikan file MP4/MP3 yang diunggah valid.")
            
            finally:
                # Membersihkan file temporary
                if os.path.exists(file_path):
                    os.remove(file_path)
                # Membersihkan audio_path jika itu adalah file sementara hasil ekstrak
                if audio_path and os.path.exists(audio_path) and audio_path != file_path:
                    os.remove(audio_path)
