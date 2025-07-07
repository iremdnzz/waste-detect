import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("best.pt")  # Eğittiğiniz modelin yolu

# Global değişkenler
uploaded_image = None
input_image_label_photo = None
output_image_label_photo = None
webcam_image_label_photo = None
is_webcam_active = False  # Webcam aktif mi?

# Orta frame boyutları
MIDDLE_FRAME_WIDTH = 640
MIDDLE_FRAME_HEIGHT = 500

def select_image():
    """Resim Seçme Fonksiyonu"""
    global uploaded_image, input_image_label_photo
    is_webcam_active = False  # Webcam'i kapat
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if file_path:
        uploaded_image = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Resmi orta frame'e sığacak şekilde boyutlandır
        img_resized = img_pil.resize((MIDDLE_FRAME_WIDTH // 2, MIDDLE_FRAME_HEIGHT // 2), Image.Resampling.LANCZOS)
        input_image_label_photo = ImageTk.PhotoImage(img_resized)

        input_image_label.config(image=input_image_label_photo)
        input_image_label.image = input_image_label_photo  # Referansı tut
        result_label.config(text="Resim başarıyla yüklendi.")

def detect_objects():
    """Resmi Tespit Etme Fonksiyonu"""
    global uploaded_image, output_image_label_photo
    if uploaded_image is None:
        result_label.config(text="Lütfen önce bir resim yükleyin.")
        return

    results = model.predict(uploaded_image, conf=0.4)
    result_image = results[0].plot()
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(result_image_rgb)

    # Tespit edilen resmi orta frame'e sığacak şekilde boyutlandır
    img_resized = img_pil.resize((MIDDLE_FRAME_WIDTH // 2, MIDDLE_FRAME_HEIGHT // 2), Image.Resampling.LANCZOS)
    output_image_label_photo = ImageTk.PhotoImage(img_resized)

    output_image_label.config(image=output_image_label_photo)
    output_image_label.image = output_image_label_photo  # Referansı tut
    result_label.config(text="Tespit tamamlandı!")

def play_webcam():
    """Web kamerası ile canlı tespit fonksiyonu"""
    global is_webcam_active, webcam_image_label_photo
    is_webcam_active = True
    cap = cv2.VideoCapture(0)

    def show_frame():
        if not is_webcam_active:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            return

        results = model.predict(frame, conf=0.4)
        result_frame = results[0].plot()

        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(result_frame_rgb)

        img_resized = img_pil.resize((MIDDLE_FRAME_WIDTH, MIDDLE_FRAME_HEIGHT), Image.Resampling.LANCZOS)
        webcam_image_label_photo = ImageTk.PhotoImage(img_resized)

        webcam_image_label.config(image=webcam_image_label_photo)
        webcam_image_label.image = webcam_image_label_photo

        root.after(10, show_frame)

    show_frame()

def stop_webcam():
    """Webcam'i durdurma fonksiyonu"""
    global is_webcam_active
    is_webcam_active = False

def toggle_ui():
    """Image/Webcam seçimine göre UI güncelle"""
    if selected_source.get() == "Image":
        browse_button.pack(pady=10)
        detect_button.pack(pady=10)
        webcam_button.pack_forget()
        stop_webcam_button.pack_forget()

        input_image_label.pack(side="left", padx=10, pady=10)
        output_image_label.pack(side="left", padx=10, pady=10)
        webcam_image_label.pack_forget()
    elif selected_source.get() == "Webcam":
        browse_button.pack_forget()
        detect_button.pack_forget()
        webcam_button.pack(pady=10)
        stop_webcam_button.pack(pady=10)

        input_image_label.pack_forget()
        output_image_label.pack_forget()
        webcam_image_label.pack(side="left", padx=10, pady=10)

# Ana pencereyi oluştur
root = tk.Tk()
root.title("Waste Classification using YOLOv8")
root.geometry("1000x600")
root.resizable(False, False)

# Renkler ve fontlar
bg_color = "#2b2b2b"
frame_color = "#404040"
text_color = "#ffffff"
btn_color = "#4caf50"

root.configure(bg=bg_color)

# Başlık etiketi
title_label = tk.Label(
    root, text="Waste Classification", font=("Arial", 20, "bold"), fg=text_color, bg=bg_color
)
title_label.pack(pady=10)

# Sol çerçeve
left_frame = tk.Frame(root, bg=frame_color, width=300, height=500)
left_frame.place(x=20, y=70)

# Orta çerçeve
middle_frame = tk.Frame(root, bg=frame_color, width=MIDDLE_FRAME_WIDTH, height=MIDDLE_FRAME_HEIGHT)
middle_frame.place(x=230, y=70)

# Sol Frame İçeriği
source_label = tk.Label(
    left_frame, text="Source Selection", font=("Arial", 14, "bold"), bg=frame_color, fg=text_color
)
source_label.pack(pady=20)

source_options = [("Image", "Image"), ("Webcam", "Webcam")]
selected_source = tk.StringVar(value="Image")

for text, value in source_options:
    tk.Radiobutton(
        left_frame,
        text=text,
        variable=selected_source,
        value=value,
        font=("Arial", 12),
        bg=frame_color,
        fg=text_color,
        selectcolor=frame_color,
        activeforeground=text_color,
        activebackground=frame_color,
        command=toggle_ui,
    ).pack(anchor="w", padx=20)

browse_button = tk.Button(
    left_frame, text="Browse Files", command=select_image, bg=btn_color, fg=text_color, font=("Arial", 12), relief="raised"
)
detect_button = tk.Button(
    left_frame, text="Detect Objects", command=detect_objects, bg=btn_color, fg=text_color, font=("Arial", 12), relief="raised"
)
webcam_button = tk.Button(
    left_frame, text="Start Camera", command=play_webcam, bg=btn_color, fg=text_color, font=("Arial", 12), relief="raised"
)
stop_webcam_button = tk.Button(
    left_frame, text="Stop Camera", command=stop_webcam, bg=btn_color, fg=text_color, font=("Arial", 12), relief="raised"
)

input_image_label = tk.Label(middle_frame, bg=frame_color)
output_image_label = tk.Label(middle_frame, bg=frame_color)
webcam_image_label = tk.Label(middle_frame, bg=frame_color)

toggle_ui()

result_label = tk.Label(
    root, text="Detection Results will appear here.", font=("Arial", 12), fg=text_color, bg=bg_color
)
result_label.pack(pady=10)

root.mainloop()
