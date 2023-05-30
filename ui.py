import tkinter
import customtkinter
import torch
from torchvision.utils import save_image
from PIL import Image, ImageTk
from models import Generator, AGenerator

image_channels = [3,3,1]
noise_channels = [128,256,256]
gen_features = [32,64,64]


def generate_image():
    if (radio_var.get() == 1):
        generator = AGenerator(128, 3)
        checkpoint = torch.load('modelCheckPoints/Art_G.pth', map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint)
        noise = torch.randn(32, 128, 1, 1)

    if (radio_var.get() == 2):
        generator = Generator(256, 3, 64)
        checkpoint = torch.load('modelCheckPoints/Landscape_G.pth', map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint)
        noise = torch.randn(32, 256, 1, 1)

    if (radio_var.get() == 3):
        generator = Generator(noise_channels[2], image_channels[2], gen_features[2])
        checkpoint = torch.load('modelCheckPoints/MNIST_G.pt', map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint['gen_model_state_dict'])
        noise = torch.randn(32, 256, 1, 1)

    generated_image = generator(noise)

    # Save the generated image
    save_image(generated_image, 'generated_image.png')  # Specify the desired output filename and format

    # Display the generated image in the canvas
    image = Image.open('generated_image.png')
    image = image.resize((500, 500))  # Adjust the size of the displayed image as needed
    photo = ImageTk.PhotoImage(image)
    canvas.image = photo
    canvas.create_image(0, 0, anchor='nw', image=photo)


def radiobutton_event():
    print("radiobutton toggled, current value:", radio_var.get())


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("800x750")
app.title("Image Generator")

title = customtkinter.CTkLabel(app, text="Generate original images", font=('Terminal', 21))
title.pack(padx=20, pady=20)

message = customtkinter.CTkLabel(app, text="Choose the type of images you want to generate : ", font=('Terminal', 16))
message.pack(padx=20, pady=30)

radio_var = tkinter.IntVar(value=0)
radiobutton_1 = customtkinter.CTkRadioButton(app, font=('Terminal', 10), text="Art",
                                             command=radiobutton_event, variable=radio_var, value=1)
radiobutton_1.place(x=150, y=190)
radiobutton_2 = customtkinter.CTkRadioButton(app, font=('Terminal', 10), text="Landscapes",
                                             command=radiobutton_event, variable=radio_var, value=2)
radiobutton_2.place(x=330, y=190)
radiobutton_3 = customtkinter.CTkRadioButton(app, font=('Terminal', 10), text="Fashion clothing",
                                             command=radiobutton_event, variable=radio_var, value=3)
radiobutton_3.place(x=510, y=190)

generate_button = customtkinter.CTkButton(app, font=('Terminal', 10), text="Generate ! ", command=generate_image)
generate_button.place(x=330, y=260)

canvas = tkinter.Canvas(app, width=500, height=500)
canvas.pack(side='bottom', pady=20)

app.mainloop()