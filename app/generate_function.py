import torch
from diffusers import DPMSolverMultistepScheduler
import streamlit as st
from PIL import Image
import re
import ast
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

album = []
def generate(prompt, pipeline):
    with torch.inference_mode():
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=8.0, height=512, width=768).images[0]
        print(type(image))
    return image

def string_to_dict(input_string):
    scenes = {}
    scene_lines = input_string.split('\n')

    for line in scene_lines:
        line = line.strip()
        if line:
            scene_parts = line.split(':', 1)
            if len(scene_parts) == 2:
                scene_num = scene_parts[0].strip()
                description = scene_parts[1].strip()
                scenes[scene_num] = description
    return scenes

def string_to_list(input_string):

    # Find the list using regular expression
    match = re.search(r'\[.*?\]', input_string, re.DOTALL)

    if match:
        extracted_list_str = match.group(0)
    
        # Convert the extracted string to a list using ast.literal_eval
        extracted_list = ast.literal_eval(extracted_list_str)
    
        return extracted_list
    else:
        print("No list found in the input string.")

def llm_create_story(llm, suggestion):
    template = """
    The context of your movie is {suggestion}
    Return a list of important events in your movie the format of a Python list like this ["Detailed description of Event 1", "Detailed description of Event 2", "Detailed description of Event 3", "Detailed description of Event 4"]
    Only return your Python list of important events, do not add any unnecessary information of your movie
     """

    prompt = PromptTemplate(input_variables=['suggestion'], template=template)
    with torch.inference_mode():
        llm_chain = LLMChain(
            prompt=prompt,
            llm=llm
        )
        story = llm_chain.run(suggestion)
    return story

def create_image_album(input_string, pipeline):
    scenes = string_to_dict(input_string)
    for scene_num, description in scenes.items():
        image_filename = f"{scene_num}.png"
        img = generate(f"{description}, high resolution, realistic", pipeline)
        print(f"{image_filename} finished")
        print(type(img))
        st.image(img, use_column_width=True, caption=description)
        album.append(img)
def create_image_album_llm(input_string, pipeline):
    scenes = string_to_list(input_string)
    for i in range(len(scenes)):
        image_filename = f"{i}.png"
        img = generate(f"{scenes[i]}, high resolution, realistic", pipeline)
        print(f"{image_filename} finished")
        print(type(img))
        st.image(img, use_column_width=True, caption=scenes[i])
        album.append(img)
        

    
        
        

    