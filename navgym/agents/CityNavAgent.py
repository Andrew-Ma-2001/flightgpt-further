import base64
import os
import time
from mimetypes import guess_type
from openai import AzureOpenAI
from pydantic import BaseModel
from navgym.models.NavGym import action_dict, action_list
from openai import OpenAI
from PIL import Image


def get_prompt(instruction, cur_pose):
    prompt = f"""
[Mission Objective]  
Your mission is to locate a specific target described via natural language instructions.

[Details of the Target]  
{instruction}

[Environmental Perception]  
- The UAV's current position is indicated by the starting point of an arrow in the image, with its orientation represented by the arrow's direction.  
- The yellow box outlines the UAV's current field of view, centered at pixel coordinates: cur_pose = {cur_pose}.  
- Street-related landmark regions are visually marked using red masks.

[Operational Guidance]  
- The target is always positioned near a red-masked street landmark.  
- Use both the instruction and the visual scene to identify the most relevant red-masked landmark region.  
- Reason about the likely relative position of the target with respect to that landmark.

[Output Format Specification]  
- Present your reasoning within `<think>` and `</think>` tags.  
  For example, your reasoning may include the following elements:  
  - A semantic interpretation of the instruction.  
  - Identification of the correct landmark region.  
  - The bounding box of that region in the format:  
    `{{"landmark_bbox": [x1, y1, x2, y2]}}`  

- Then provide your final answer within `<answer>` and `</answer>` tags as:  
  `{{"target_location": [x, y]}}`
"""
     
    return prompt


class GPTInfo(BaseModel):
    api_key: str
    api_version: str
    api_base: str
    model: str


class GPTAgent:
    def __init__(
            self, api_key, api_version, api_base, model, 
            system_prompt, target_description, drone_see_shape,
            scale, top_left, compress_images=True
        ):
        self.gpt_info = GPTInfo(
            api_key = api_key,
            api_base = api_base,
            api_version =api_version,
            model = model
        )
        self.system_prompt = system_prompt
        self.target_description = target_description
        self.drone_see_shape = drone_see_shape
        self.scale = scale
        self.top_left = top_left
        self.compress_images = compress_images
        self.image_scale_factor = 1.0  # Track image resize scale


    def act(self, cur_whole_map, cur_rgb_drone, cur_position):
        # Optionally resize images to reduce VLLM throughput pressure
        if self.compress_images:
            start_time = time.time()
            cur_whole_map, map_scale = self._check_and_resize_image(cur_whole_map, "map")
            cur_rgb_drone, drone_scale = self._check_and_resize_image(cur_rgb_drone, "drone")
            resize_time = time.time() - start_time
        else:
            map_scale = (1.0, 1.0)
            resize_time = 0.0
        
        # Store the map scale factor for coordinate transformation
        self.image_scale_factor = map_scale
        scale_x, scale_y = map_scale
        
        # Scale the current position to match resized image coordinates
        scaled_position = [int(cur_position[0] * scale_x), int(cur_position[1] * scale_y)]
        
        if resize_time > 1.0:
            print(f"  [Image Resize] took {resize_time:.2f}s")
        
        if cur_position != scaled_position:
            print(f"  [Coordinate] cur_pose scaled: {cur_position} -> {scaled_position}")
        
        result = self._gpt4o_imagefile(
            map_file=cur_whole_map, 
            view_file=cur_rgb_drone, 
            system_prompt=self.system_prompt, 
            prompt=get_prompt(
                instruction=self.target_description, cur_pose=scaled_position
            )
        )
        response = result.choices[0].message.content
        return response
    
    def scale_coordinates_to_original(self, coordinates):
        """
        Scale coordinates from resized image back to original image dimensions
        Args:
            coordinates: Can be [x, y] for point or [x1, y1, x2, y2] for bbox
        Returns:
            Scaled coordinates in original image space
        """
        scale_x, scale_y = self.image_scale_factor
        
        if len(coordinates) == 2:
            # Point coordinates [x, y]
            x, y = coordinates
            original_x = int(x / scale_x)
            original_y = int(y / scale_y)
            return [original_x, original_y]
        elif len(coordinates) == 4:
            # Bounding box [x1, y1, x2, y2]
            x1, y1, x2, y2 = coordinates
            original_x1 = int(x1 / scale_x)
            original_y1 = int(y1 / scale_y)
            original_x2 = int(x2 / scale_x)
            original_y2 = int(y2 / scale_y)
            return [original_x1, original_y1, original_x2, original_y2]
        else:
            print(f"  [WARNING] Unexpected coordinate format: {coordinates}")
            return coordinates
    


    @staticmethod
    def _check_and_resize_image(image_path, image_type="image", max_size=2048):
        """
        Check image size and resize if too large
        Returns: (resized_image_path, scale_factor)
        scale_factor = resized_size / original_size (for coordinate transformation)
        """
        try:
            # Check file size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            
            # Load image
            img = Image.open(image_path)
            original_size = img.size
            
            # Check if resizing is needed
            if max(img.size) > max_size or file_size_mb > 5:
                print(f"  [{image_type}] Original: {img.size} ({file_size_mb:.2f}MB) - Resizing to max {max_size}px")
                
                # Calculate scale factor before resizing
                original_width, original_height = img.size
                
                # Resize maintaining aspect ratio
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                new_width, new_height = img.size
                
                # Calculate scale factors (resized / original)
                scale_x = new_width / original_width
                scale_y = new_height / original_height
                
                # Save resized image
                resized_path = image_path.replace('.jpg', '_resized.jpg').replace('.png', '_resized.png')
                if resized_path == image_path:  # fallback if no extension
                    resized_path = image_path + '_resized.jpg'
                
                # Save with quality optimization
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(resized_path, 'JPEG', quality=85, optimize=True)
                
                new_size_mb = os.path.getsize(resized_path) / (1024 * 1024)
                print(f"  [{image_type}] Resized: {img.size} ({new_size_mb:.2f}MB)")
                print(f"  [{image_type}] Scale factor: x={scale_x:.4f}, y={scale_y:.4f}")
                
                return resized_path, (scale_x, scale_y)
            else:
                print(f"  [{image_type}] Size OK: {img.size} ({file_size_mb:.2f}MB)")
                return image_path, (1.0, 1.0)
                
        except Exception as e:
            print(f"  [WARNING] Failed to check/resize {image_type}: {e}")
            return image_path, (1.0, 1.0)

    @staticmethod
    def _local_image_to_data_url(image_path):
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(image_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def _gpt4o_imagefile(self, map_file, view_file, system_prompt, prompt,
                         timeout=120.0):
        """
        Gpt-4o model with timeout, fail fast on error
        """
        client = OpenAI(
            base_url=self.gpt_info.api_base,
            api_key=self.gpt_info.api_key,
            timeout=timeout,
        )

        # Time the base64 encoding
        encode_start = time.time()
        map_data_url = self._local_image_to_data_url(map_file)
        encode_time = time.time() - encode_start
        
        if encode_time > 1.0:
            print(f"  [Base64 Encode] took {encode_time:.2f}s")

        api_start = time.time()
        response = client.chat.completions.create(
            model=self.gpt_info.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": map_data_url},
                        }
                    ],
                },
            ],
            max_tokens=2000,
            temperature=0.0,
            timeout=timeout,
        )
        api_time = time.time() - api_start
        print(f"  [API Call] took {api_time:.2f}s")
        return response