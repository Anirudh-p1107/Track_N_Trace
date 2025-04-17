from tkinter import Tk, Label, Button, Frame, filedialog
from PIL import Image, ImageTk, ImageOps
import os
import numpy as np
import librosa
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms

audio_model = tf.keras.models.load_model("best_audio.keras")
audio_labels = {0:'bear',1:'cat',2:'cow',3:'dog',4:'donkey',5:'elephant',6:'horse',7:'lion',8:'monkey',9:'sheep'}

safe_audio = {'cat','cow','dog','donkey','horse','sheep'}
unsafe_audio = {'bear','elephant','lion','monkey'}

foot_labels = {0:'bear',1:'bobcat',2:'deer',3:'fox',4:'horse',5:'lion',6:'mouse',7:'racoon',8:'squirrel',9:'wolf'}

safe_foot = {'deer','horse','mouse','squirrel','racoon'}
unsafe_foot = {'bear','bobcat','fox','lion','wolf'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*16*16, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, len(foot_labels))
        )

    def forward(self, x): return self.net(x)

foot_model = CNN().to(device)
foot_model.load_state_dict(torch.load("best_footprint.pth", map_location=device))
foot_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def extract_audio(path):
    try:
        y, sr = librosa.load(path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=256, fmax=8000)
        mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
        mel_norm = (mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db) + 1e-9)
        padded = np.zeros((128, 174))
        padded[:, :mel_norm.shape[1]] = mel_norm[:, :174]
        return np.expand_dims(padded, axis=(0, -1))
    except:
        return None

def show_img(lbl, folder, animal):
    name = "unknown" if animal.lower() == "unknown" else animal
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(folder, f"{name}{ext}")
        if os.path.exists(path):
            try:
                img = Image.open(path).resize((280, 230))
                photo = ImageTk.PhotoImage(img)
                lbl.config(image=photo, text="")
                lbl.image = photo
                return
            except:
                break
    lbl.config(text="Image not found", image="")

def update_ui(label, img_lbl, animal, cat, conf, folder):
    color = {"Safe": "#20eb0e", "Unsafe": "#eb1d0e", "Unknown": "#fffb1c"}[cat]
    label.config(text=f"Detected: {animal}\n{cat} Animal\nConfidence: {conf:.2f}%", bg=color, fg='black')
    img_lbl.config(bg=color)
    show_img(img_lbl, folder, animal)

def predict_audio():
    path = filedialog.askopenfilename(filetypes=[["Audio", "*.wav *.mp3"]])
    if not path: return
    feat = extract_audio(path)
    if feat is None:
        audio_result.config(text="Error processing audio.")
        return
    pred = audio_model.predict(feat)[0]
    idx = np.argmax(pred)
    conf = pred[idx] * 100
    if conf < 60:
        update_ui(audio_result, audio_img, "Unknown", "Unknown", conf, "images")
    else:
        animal = audio_labels[idx]
        cat = "Safe" if animal in safe_audio else "Unsafe"
        update_ui(audio_result, audio_img, animal, cat, conf, "images")

def predict_foot():
    path = filedialog.askopenfilename(filetypes=[["Images", "*.jpg *.png"]])
    if not path: return
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        foot_result.config(text="Error reading image.")
        return
    img = cv2.resize(img, (128, 128))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = foot_model(tensor)
        probs = F.softmax(out, dim=1)
        idx = torch.argmax(probs).item()
        conf = probs[0][idx].item() * 100
        if conf < 60:
            update_ui(foot_result, foot_img, "Unknown", "Unknown", conf, "animals")
        else:
            animal = foot_labels[idx]
            cat = "Safe" if animal in safe_foot else "Unsafe"
            update_ui(foot_result, foot_img, animal, cat, conf, "animals")

# GUI setup
root = Tk()
root.title("TRACK & TRACE")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{w}x{h}")

# Set background
try:
    bg_img = Image.open("background.png").resize((w, h))
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_lbl = Label(root, image=bg_photo)
    bg_lbl.image = bg_photo
    bg_lbl.place(x=0, y=0, relwidth=1, relheight=1)
    bg_lbl.lower()
except:
    print("Background not found.")

# Font-styled widgets
def lbl(p, t, s=12, w="normal", pady=10):
    return Label(p, text=t, font=("Comic Sans MS", s, w), bg="white", pady=pady)

def btn(p, t, cmd):
    return Button(p, text=t, font=("Comic Sans MS", 12), bg="black", fg="white", command=cmd, padx=10, pady=5)

def show(f):
    for fr in [home, audio, foot]:
        fr.pack_forget()
    f.pack(pady=100)

home = Frame(root, bg="white")
lbl(home, "*** TRACK & TRACE ***", 40, "bold").pack(pady=10)
lbl(home, "Select Input Type",25).pack()
btn(home, "Footprint Image", lambda: show(foot)).pack(pady=20)
btn(home, "Animal Sound", lambda: show(audio)).pack(pady=20)
# Load and display the logo on home frame
try:
    
    logo_img = Image.open("logo.png").resize((200, 200), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = Label(home, image=logo_photo, bg="white")
    logo_label.image = logo_photo  # Keep a reference!
    logo_label.pack(pady=10)
except Exception as e:
    print("Logo not found or failed to load:", e)

home.pack(pady=100)


audio = Frame(root, bg="white")
lbl(audio, "Identify Animal Sound", 40).pack()
btn(audio, "Upload Audio", predict_audio).pack(pady=20)
btn(audio, "Home", lambda: show(home)).pack(pady=20)
audio_result = lbl(audio, "", 18)
audio_result.pack()
audio_img = Label(audio, bg="white")
audio_img.pack(pady=10)

foot = Frame(root, bg="white")
lbl(foot, "Identify Animal Footprint", 40).pack()
btn(foot, "Upload Image", predict_foot).pack(pady=20)
btn(foot, "Home", lambda: show(home)).pack(pady=20)
foot_result = lbl(foot, "", 18)
foot_result.pack()
foot_img = Label(foot, bg="white")
foot_img.pack(pady=10)

root.mainloop()
