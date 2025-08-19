# RotBench


Authors: [Tianyi Niu](https://www.linkedin.com/in/tianyi-niu/)¹, [Jaemin Cho](https://j-min.io/)¹, [Elias Stengel-Eskin](https://esteng.github.io)¹, and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)¹

¹UNC Chapel Hill <br>

***RotBench data will be released soon***

# Abstract
We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0°, 90°, 180°, and 270°. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0°) images, while certain models are able to identify upside-down (180°) images. None can reliably distinguish between 90° and 270°. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90° and 270° rotations, despite substantially improving the identification of 180° images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation.

# RotBench Pipeline

<img src='./assets/imgrot_main_v2.png'>

For each image in RotBench, we rotate the image 0°, 90°, 180°, and 270° counter-clockwise. We represent the rotation estimation problem as a multiple-choice question answering problem, and separately measure accuracy on each image orientation. We optionally provide different forms of auxiliary information to aid the model in identifying image rotation. We emphasize that all forms of auxiliary information are separately extracted for each rotation; the ground truth rotation is not marked.

# Code Setup

### VLLM Servers

We evaluate Llama-3.2-11B-Vision (meta-llama/Llama-3.2-11B-Vision) and Qwen2.5-VL-7B-Instruct (Qwen/Qwen2.5-VL-7B-Instruct) on RotBench. We also use Qwen--8B (Qwen/Qwen3-8B) to evalute all model responses. Our code assumes access to these models through the following ports on localhost: 

| Model         | Port  |
|---------------|-------|
| Qwen 3        | 7471  |
| Qwen 2.5 VL   | 7472  |
| Llama 3.2     | 7473  |

### Model Nicknames

Moreover, the codebase makes frequent use of the follwoing model nicknames:

| Nickname      | Model Name                                       |
|---------------|--------------------------------------------------|
| Llama11       | meta-llama/Llama-3.2-11B-Vision-Instruct         |
| Qwen7         | Qwen/Qwen2.5-VL-7B-Instruct                      |
| Qwen32        | Qwen/Qwen2.5-VL-32B-Instruct                     |
| GPT4o         | gpt-4o                                           |
| GPT41         | gpt-4.1                                          |
| GPT5          | gpt-5                                            |
| o3            | o3                                               |
| Gemini2       | gemini-2.0-flash                                 |
| Gemini25      | gemini-2.5-flash                                 |
| Gemini25pro   | gemini-2.5-pro                                   |

### API Keys

To run call proprietary models, ensure you replace the placeholder API keys in `.env`

## Directory Setup 

The directory should follow this structure:

<pre>
RotBench
├── Get_Additional_Info
│   ├── get_bounding_boxes.py
│   ├── get_captions.py
│   ├── get_depth_maps.py
│   ├── get_image_subjects.py
│   ├── get_rotated_grid.py
│   ├── get_scene_graph.py
│   └── get_seg_mask.py
├── RotBench-Data
│   ├── RotBench_large
│   └── RotBench_small
├── README.md
├── .env
├── common.py
├── create_datasets.py
├── evaluate.py
├── inference.py
└── README.md
</pre>


## Obtaining Rotated Datasets

First, rotate RotBench-Small and RotBench-Large, creating four separate datasets (each containing images rotated 0, 90, 180, and 270 degrees counter-clockwise). The resulting dataests are saved in `./Small` and `./Large`.

`python create_datasets.py`.

## Creating Auxiliary Information

`./Get_Additional_Info` includes all scripts necessary for extracting the various forms of auxiliary information. Ensure `get_image_subject.py` is first executed as multiple files require a list of image subjects as a prerequisite. 

Extract image subjects: 
`python Get_Additional_Info/get_image_subjects.py -ds small`.


## Get Model Predictions on RotBench

To get model predictions on RotBench-small using GPT4o with a seed of 0: 
`python inference.py -nick GPT4o -workers 3 -ds small --run 0 -dm`.

## Evaluate Model Predictions
To evaluate the run: 
`python evaluate.py -nick GPT4o -ds small --runs 0 -dm`.

## Clockwise vs. Counter-clockwise

To evaluate GPT-4o or Qwen on the binary clockwise vs. counter-clockwise classification task: Modify the `nick` and `max_workers` on line 60 and run `python CW_CCW_inference.py`.

To evaluate the results, modify the `nick` and `max_workers` parameters on line 98 and run `python CW_CCW_inference.py -r 0 -w 3 -fp QwenFT`.


## In-context Learning

To evaluate Qwen with `10` in-context examples: 
`python ICL.py -n 10 -w 3`.


## Fine-tuning Qwen

MS COCO should be downloaded and formatted as following: 
<pre>
RotBench
├── Coco
│   ├── 0_imgs
│   ├── 90_imgs
│   ├── 180_imgs
│   └── 270_imgs
</pre>

To fine-tune the model: 
`python FT_training.py -r 0 -fp QwenFT`.

To evaluate the resulting model: 
`python FT_inference.py -r 0 -w 3 -fp QwenFT`.

## Normalized Voting 

To evaluate model using Normalized Rotation Voting: 
`python FT_training.py -r 0 -fp QwenFT`.

Note running this file first requires obtaining zero-shot data for the model of choice. In other words, first run `inference.py` with no auxiliary information.  
