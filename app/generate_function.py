import torch
from diffusers import DPMSolverMultistepScheduler




def generate(prompt, pipeline):
    with torch.inference_mode():
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=8.0, height=512, width=768).images[0]
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

def create_image_album(input_string, pipeline, image_folder):
    scenes = string_to_dict(input_string)
    for scene_num, description in scenes.items():
        image_filename = f"{scene_num}.png"
        generate(f"{description}, high resolution, realistic, perfect face", pipeline).save(f"{image_folder}/{image_filename}")
        print(f"{image_filename} finished")