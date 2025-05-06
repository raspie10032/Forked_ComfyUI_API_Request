import os
import zipfile
import aiohttp
import asyncio
import json
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, List, Union, Dict, Any

ucPreset_list = [
    "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts",
    "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, signature",
    ""
]


async def post_novelai(url, data, header, proxy):
    async with aiohttp.ClientSession(headers=header) as session:
        try:
            async with session.post(url, json=data, proxy=proxy) as response:
                if response.status == 429:
                    resp = await response.json()
                    message = resp.get("message", "Too many requests, retrying...")
                    await asyncio.sleep(5)
                    return await post_novelai(url, data, header, proxy)
                elif response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Request failed with status {response.status}")
        except Exception as e:
            print(f"Error during request: {e}")
            raise e


class CharacterPrompt:
    def __init__(self, prompt: str, uc: str, x: float, y: float):
        self.prompt = prompt
        self.uc = uc
        self.x = x
        self.y = y
        self.center = {"x": self.x, "y": self.y}


class BaseRequest:
    def __init__(
            self,
            prompt: Optional[str] = "",
            negative_prompt: Optional[str] = "",
            seed: int = -1,
            sampler_name: str = "Euler a",
            batch_size: int = 1,
            n_iter: int = 1,
            steps: int = 28,
            cfg_scale: float = 7,
            width: int = 832,
            height: int = 1216,
            denoising_strength: float = 1,
            scheduler: str = "Automatic",
            send_images: bool = True,
            save_images: bool = True,
            override_settings: Dict[str, Any] = {},
            override_settings_restore_afterwards: bool = False
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.sampler_name = sampler_name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.width = width
        self.height = height
        self.denoising_strength = denoising_strength
        self.scheduler = scheduler
        self.send_images = send_images
        self.save_images = save_images
        self.override_settings = override_settings
        self.override_settings_restore_afterwards = override_settings_restore_afterwards


class NovelAITXT2IMGPayload(BaseRequest):
    def __init__(
            self, ucPreset: int,
            cfg_rescale: int, characterPrompts: list[CharacterPrompt]|list,
            prefer_brownian: bool, base_request: BaseRequest, model,
            use_coords: bool = True,  # 추가된 매개변수
            use_order: bool = True    # 추가된 매개변수
    ):
        super().__init__(**vars(base_request))

        self.scale = base_request.cfg_scale
        self.sampler = base_request.sampler_name
        self.n_samples = base_request.n_iter
        self.ucPreset = ucPreset
        self.cfg_rescale = cfg_rescale
        self.noise_schedule = base_request.scheduler
        self.characterPrompts = characterPrompts
        self.model = model
        self.use_coords = use_coords  # 추가된 속성
        self.use_order = use_order    # 추가된 속성
        
        characterPrompts_list = []

        for cp in self.characterPrompts:
            cp = dict(vars(cp))
            del cp['x']
            del cp['y']
            characterPrompts_list.append(cp)

        self.characterPrompts = characterPrompts_list
        char_captions = []
        for cp in self.characterPrompts:
            char_captions.append({"char_caption": cp["prompt"], "centers": [cp["center"]]})
        self.v4_prompt = {
            "caption": {
                "base_caption": self.prompt,
                "char_captions": char_captions
            },
            "use_coords": self.use_coords,  # 여기에 추가
            "use_order": self.use_order     # 여기에 추가
        }
        char_captions = []
        for cp in self.characterPrompts:
            char_captions.append({"char_caption": cp["uc"], "centers": [cp["center"]]})
        self.v4_negative_prompt = {
            "caption": {
                "base_caption": self.negative_prompt,
                "char_captions": char_captions
            },
            "use_coords": False,  # 부정 프롬프트에는 좌표 사용 안함
            "use_order": False    # 부정 프롬프트에는 순서 사용 안함
        }
        self.prefer_brownian = prefer_brownian


def tensor_to_pil(img_tensor, batch_index=0):
    # Takes an image in a batch in the form of a tensor of shape [batch_size, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels

    # Take the image in the batch given by batch_index
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


class NovelAIRequest:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "payload": ("NovelAITXT2IMGPayload",),
            },
            "optional": {
                "token": ("STRING", {"default": "ey...."}),
                "proxy": ("STRING", {"default": "http://127.0.0.1:7890"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "novelai_generate_image"
    OUTPUT_NODE = True
    CATEGORY = "NovelAI"  # 카테고리 변경

    def novelai_generate_image(self, payload, token, proxy):

        save_path = "./temp/api_request"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # NOVELAI_TOKEN에서 NAI_ACCESS_TOKEN으로 환경 변수 이름 변경
        token = token if token else os.getenv("NAI_ACCESS_TOKEN", "")
        header = {
            "authorization": "Bearer " + token,
            ":authority": "https://api.novelai.net",
            ":path": "/ai/generate-image",
            "content-type": "application/json",
            "referer": "https://novelai.net",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
        }

        payload = {
            "input": payload.prompt,
            "model": payload.model,
            "parameters": dict(vars(payload))

        }

        response_data = asyncio.run(post_novelai("https://image.novelai.net/ai/generate-image", payload, header, proxy))

        with zipfile.ZipFile(BytesIO(response_data)) as z:
            z.extractall(save_path)

        image_bytes = []

        def images_to_base64(save_path):

            for filename in os.listdir(save_path):
                if filename.endswith('.png'):
                    file_path = os.path.join(save_path, filename)
                    with open(file_path, "rb") as image_file:
                        image_data = image_file.read()
                        image_bytes.append(image_data)

        images_to_base64(save_path)

        def pil_and_tensor(img_data: bytes):
            img = Image.open(BytesIO(img_data))
            tensor_img = pil_to_tensor(img)

            return tensor_img

        tensor_images = [pil_and_tensor(img_data) for img_data in image_bytes]

        tensor = torch.cat(tensor_images, dim=0)

        return (tensor,)


class CharacterPromptSelect:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character1": ("STRING", {"default": "firefly_(honkai:_star_rail)"}),
                "character1_uc": ("STRING", {"default": "negative_prompt"}),
                "character1_x": ("FLOAT", {"default": 0.5}),
                "character1_y": ("FLOAT", {"default": 0.5}),
            },
            "optional": {
                "character2_enable": ("BOOLEAN", {"default": True}),
                "character2": ("STRING", {"default": "kafka__(honkai:_star_rail)"}),
                "character2_uc": ("STRING", {"default": "negative_prompt"}),
                "character2_x": ("FLOAT", {"default": 0.1}),
                "character2_y": ("FLOAT", {"default": 0.5}),
                "character3_enable": ("BOOLEAN", {"default": False}),
                "character3": ("STRING", {"default": ""}),
                "character3_uc": ("STRING", {"default": "negative_prompt"}),
                "character3_x": ("FLOAT", {"default": 0.1}),
                "character3_y": ("FLOAT", {"default": 0.5}),
                "character4_enable": ("BOOLEAN", {"default": False}),
                "character4": ("STRING", {"default": ""}),
                "character4_uc": ("STRING", {"default": "negative_prompt"}),
                "character4_x": ("FLOAT", {"default": 0.1}),
                "character4_y": ("FLOAT", {"default": 0.5}),
                "character5_enable": ("BOOLEAN", {"default": False}),
                "character5": ("STRING", {"default": ""}),
                "character5_uc": ("STRING", {"default": "negative_prompt"}),
                "character5_x": ("FLOAT", {"default": 0.1}),
                "character5_y": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("CharacterPrompt",)
    FUNCTION = "build_character_prompt"
    OUTPUT_NODE = False
    CATEGORY = "NovelAI/Character"  # 카테고리 변경 및 세분화

    def build_character_prompt(
        self, character1, character1_uc, character1_x, character1_y,
        character2_enable, character2, character2_uc, character2_x, character2_y, 
        character3_enable, character3, character3_uc, character3_x, character3_y, 
        character4_enable, character4, character4_uc, character4_x, character4_y, 
        character5_enable, character5, character5_uc, character5_x, character5_y
   ):
        character1_pos = {"x": character1_x, "y": character1_y}
        character2_pos = {"x": character2_x, "y": character2_y}
        character3_pos = {"x": character3_x, "y": character3_y}
        character4_pos = {"x": character4_x, "y": character4_y}
        character5_pos = {"x": character5_x, "y": character5_y}

        character_prompts = []

        character_prompts.append(CharacterPrompt(character1, character1_uc, character1_pos["x"], character1_pos["y"]))

        if character2_enable:
            character_prompts.append(CharacterPrompt(character2, character2_uc, character2_pos["x"], character2_pos["y"]))
        if character3_enable:
            character_prompts.append(CharacterPrompt(character3, character3_uc, character3_pos["x"], character3_pos["y"]))
        if character4_enable:
            character_prompts.append(CharacterPrompt(character4, character4_uc, character4_pos["x"], character4_pos["y"]))
        if character5_enable:
            character_prompts.append(CharacterPrompt(character5, character5_uc, character5_pos["x"], character5_pos["y"]))

        return (character_prompts,)


class NovelAIRequestPayload:

    @classmethod
    def INPUT_TYPES(cls):
        sampler_list = [
            "k_dpmpp_2m", "k_dpmpp_sde", "k_dpmpp_2m_sde", "k_dpmpp_2s_ancestral",
            "k_euler_ancestral", "k_euler", "ddim_v3"
        ]
        scheduler_list = ["karras"]

        # 표시할 모델 이름 리스트 (이 이름이 UI에 표시됨)
        model_display_list = [
            "NAI Diffusion V4.5 Curated",
            "NAI Diffusion V4 Full",
            "NAI Diffusion V4 Curated Preview",
            "NAI Diffusion V3",
            "NAI Diffusion Furry V3",
            "NAI Diffusion V2"
        ]

        # 각 표시 이름에 대응하는 실제 API 모델 ID
        model_ids = [
            "nai-diffusion-4-5-curated",
            "nai-diffusion-4-full",
            "nai-diffusion-4-curated-preview",
            "nai-diffusion-3",
            "nai-diffusion-furry-3",
            "nai-diffusion-2",
        ]

        # 표시 이름과 API ID를 딕셔너리로 매핑
        model_name_to_id = dict(zip(model_display_list, model_ids))

        return {
            "required": {
                "prompt": ("STRING", {"default": "prompt here"})
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "negative prompt here"}),
                "seed": ("INT", {"default": -1}),
                "sampler": (sampler_list,),
                "n_iter": ("INT", {"default": 1}),
                "steps": ("INT", {"default": 28}),
                "cfg_scale": ("FLOAT", {"default": 6}),
                "width": ("INT", {"default": 832}),
                "height": ("INT", {"default": 1216}),
                "scheduler": (scheduler_list,),
                "ucPreset": (ucPreset_list,),
                "cfg_rescale": ("INT", {"default": 0}),
                "characterPrompts": ("LIST",),
                "prefer_brownian": ("BOOLEAN", {"default": False}),
                "use_coords": ("BOOLEAN", {"default": True}),
                "use_order": ("BOOLEAN", {"default": True}),
                "model": (model_display_list, {"default": model_display_list[0]})  # 표시 이름 사용
            }
        }

    RETURN_TYPES = ("NovelAITXT2IMGPayload", "DICT")
    RETURN_NAMES = ("payload", "payload_dict")
    FUNCTION = "build_payload"
    CATEGORY = "NovelAI/Payload"  # 카테고리 변경 및 세분화

    def build_payload(
            self,
            prompt,
            negative_prompt,
            seed,
            sampler,
            n_iter,
            steps,
            cfg_scale,
            width,
            height,
            scheduler,
            ucPreset,
            cfg_rescale,
            prefer_brownian,
            model,  # 이제 이 값은 표시 이름
            use_coords=True,
            use_order=True,
            characterPrompts=[],
    ):
        # 표시 이름에서 API 모델 ID로 변환
        model_ids = {
            "NAI Diffusion V4.5 Curated": "nai-diffusion-4-5-curated",
            "NAI Diffusion V4 Full": "nai-diffusion-4-full",
            "NAI Diffusion V4 Curated Preview": "nai-diffusion-4-curated-preview",
            "NAI Diffusion V3": "nai-diffusion-3",
            "NAI Diffusion Furry V3": "nai-diffusion-furry-3",
            "NAI Diffusion V2": "nai-diffusion-2"
        }
        model_id = model_ids.get(model, "nai-diffusion-4-5-curated")  # 기본값 설정

        negative_prompt = negative_prompt + "," + ucPreset
        instance_ = BaseRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            sampler_name=sampler,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            scheduler=scheduler,
        )

        ucPreset = ucPreset_list.index(ucPreset)

        instance_ = NovelAITXT2IMGPayload(
            ucPreset,
            cfg_rescale,
            characterPrompts,
            prefer_brownian,
            instance_,
            model_id,  # API 모델 ID 사용
            use_coords,
            use_order
        )

        return instance_, dict(vars(instance_))


NODE_CLASS_MAPPINGS = {
    "NovelAI_Request": NovelAIRequest,
    "Character_Prompt_Select": CharacterPromptSelect,
    "NovelAI_Request_Payload": NovelAIRequestPayload
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NovelAI_Request": "NovelAIRequest",
    "Character_Prompt_Select": "CharacterPromptSelect",
    "NovelAI_Request_Payload": "NovelAIRequestPayload"
}

# 모델 ID와 표시 이름 매핑을 클래스 외부에서도 사용할 수 있도록 정의
MODEL_DISPLAY_NAMES = {
    "nai-diffusion-4-5-curated": "NAI Diffusion V4.5 Curated",
    "nai-diffusion-4-full": "NAI Diffusion V4 Full",
    "nai-diffusion-4-curated-preview": "NAI Diffusion V4 Curated Preview",
    "nai-diffusion-3": "NAI Diffusion V3",
    "nai-diffusion-furry-3": "NAI Diffusion Furry V3",
    "nai-diffusion-2": "NAI Diffusion V2"
}