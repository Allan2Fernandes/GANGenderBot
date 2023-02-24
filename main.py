
from discord.ext import commands
import discord
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

intents = discord.Intents.all()
classification_model_path = "C:/Users/allan/PycharmProjects/FacialAttributeClassifier/EyeGlassesModels/Epoch20.h5"
generator_model_path = "C:/Users/allan/PycharmProjects/GANGradientAscent/Models/Epoch340/Generator.h5"
noise_dimensions = 64
bot = commands.Bot(command_prefix='!', intents=intents)
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def center_crop(img):
    #Get the 2 dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Get the smaller dim
    smaller_dim = min(dim1, dim2)

    #Get the center for both dimensions
    dim1_center = dim1 // 2
    dim2_center = dim2 // 2

    #Crop from center for both dimensions
    dim1_start = dim1_center-smaller_dim//2
    dim1_end = dim1_center+smaller_dim//2

    dim2_start = dim2_center-smaller_dim//2
    dim2_end = dim2_center+smaller_dim//2

    img = img[dim1_start:dim1_end, dim2_start:dim2_end, :]
    return img

def get_img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def plot_image(tensor):
    plt.imshow(tensor)
    plt.show()

def preprocess_tensor(image_file):
    image_array = np.array(image_file)
    image_array = image_array[:,:,:3]
    processed_tensor = tf.convert_to_tensor(image_array)
    processed_tensor = tf.cast(processed_tensor, tf.float32)
    processed_tensor = (processed_tensor) / 255
    return processed_tensor

def preprocess_input_image(tensor):
    # Center crop
    tensor = center_crop(tensor)
    #Resize to correct input size
    image_tensor = tf.image.resize(tensor, (128, 128), method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=True)
    #Expand dims
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

def get_prediction_CAM(processed_tensor):
    with tf.GradientTape() as tape:
        input_tensor = preprocess_input_image(processed_tensor)
        tape.watch(input_tensor)
        saved_image_tensor = input_tensor[0]
        actual_prediction = classification_model(input_tensor)
        if actual_prediction[0] > 0.5:
            classification_prediction = "Glasses with certainty = {0}".format(actual_prediction[0][0])
            expected_prediction = tf.ones_like(actual_prediction)
        else:
            classification_prediction = "Without glasses with certainty = {0}".format(1 - actual_prediction[0][0])
            expected_prediction = tf.zeros_like(actual_prediction)
            pass
        loss = loss_function(y_true=expected_prediction, y_pred=actual_prediction)
        gradient = tape.gradient(loss, input_tensor)
    return gradient, saved_image_tensor, classification_prediction

def post_process_gradients_into_image(saved_image_tensor, gradient, alpha):
    # Remove the unneeded dimension
    gradient = tf.squeeze(gradient)
    # Collapse the 3 channels
    gradient = tf.abs(gradient)
    gradient = tf.reduce_sum(gradient, axis=-1)
    # Scale it from 0 to 1
    grayscale_tensor = (gradient - tf.reduce_min(gradient)) / (tf.reduce_max(gradient) - tf.reduce_min(gradient))
    # Give the grayscale tensor a channel to be comparable to the input image
    grayscale_tensor = tf.expand_dims(grayscale_tensor, axis=-1)

    combined_image = alpha * grayscale_tensor + (1 - alpha) * saved_image_tensor
    return combined_image

def post_process_tensor_for_saving(tensor):
    tensor = np.array(tensor)
    #Convert to UInt8
    tensor = tensor*255
    tensor = tensor.astype(np.uint8)
    return tensor

@bot.event
async def on_ready():
    global classification_model
    global generator_model
    classification_model = tf.keras.models.load_model(classification_model_path)
    generator_model = tf.keras.models.load_model(generator_model_path)
    print("Ready")
    pass

@bot.command(name='classify')
async def classify(ctx):
    #Get the url of the image attachment
    url = ctx.message.attachments[0].url
    #Get the image file from the url
    image_file = get_img_from_url(url)
    processed_tensor = preprocess_tensor(image_file)
    gradient, saved_image_tensor, classification_prediction = get_prediction_CAM(processed_tensor)
    combined_image = post_process_gradients_into_image(saved_image_tensor=saved_image_tensor, gradient=gradient, alpha = 0.5)
    combined_image = post_process_tensor_for_saving(combined_image)
    resized = cv2.resize(combined_image, (512,512), interpolation=cv2.INTER_AREA)
    combined_image_file = Image.fromarray(resized)
    combined_image_file.save("combined_image.jpeg")

    await ctx.send(classification_prediction, file = discord.File("combined_image.jpeg"))
    pass

@bot.command(name='gen')
async def generate_image(ctx):
    noise = tf.random.normal([1, noise_dimensions])
    generated_image = generator_model(noise)
    generated_image = generated_image*0.5 + 0.5
    generated_image = tf.squeeze(generated_image)
    image = post_process_tensor_for_saving(generated_image)
    resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    generated_image_file = Image.fromarray(resized)
    generated_image_file.save("generated_image.jpeg")
    await ctx.send("Generated Image", file=discord.File("generated_image.jpeg"))

Token = "Token"

bot.run(token=Token)