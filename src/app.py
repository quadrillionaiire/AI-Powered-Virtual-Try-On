import tkinter as tk
from tkinter import filedialog
from src.model import overlay_clothing

def browse_files():
    file_path = filedialog.askopenfilename()
    return file_path

def apply_clothing():
    pose_image = browse_files()
    clothing_image = browse_files()
    output_path = "data/processed/overlay_result.png"
    overlay_clothing(pose_image, clothing_image, output_path)
    print("Clothing applied.")

app = tk.Tk()
app.title("Virtual Try-On")
app.geometry("400x300")

button = tk.Button(app, text="Apply Clothing", command=apply_clothing)
button.pack()

app.mainloop()

