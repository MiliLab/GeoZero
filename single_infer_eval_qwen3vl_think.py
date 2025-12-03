import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import random
import logging
import math
import re
from typing import Optional, Union, Tuple, List, Any, Dict


system_prompt = """When answering, first describe your reasoning or visual observations in natural language, and then provide the final answer enclosed in <answer></answer>."""


MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8

MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
logger = logging.getLogger(__name__)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def reverse_from_qwen3vl_format(pred_bbox, orig_height, orig_width, image_patch_size = 14):
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    resized_height, resized_width = smart_resize(orig_height, orig_width, patch_factor)
    scale_w = orig_width / resized_width
    scale_h = orig_height / resized_height

    print('###########', pred_bbox)
    x1, y1, x2, y2 = pred_bbox
    x1_orig = round(x1 * scale_w)
    y1_orig = round(y1 * scale_h)
    x2_orig = round(x2 * scale_w)
    y2_orig = round(y2 * scale_h)

    return x1_orig, y1_orig, x2_orig, y2_orig
'''

[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>human instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      },
    {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "images": [
      "image path (required)"
    ]
  },
  ....
]
'''


import argparse
parser = argparse.ArgumentParser(description="Qwen-2.5-VL Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default="Qwen/Qwen-2.5-VL-Chat",
    help="Path to the pretrained model.",
)
parser.add_argument(
    "--json_path",
    type=str,
    default="data/scene_cls.json",
    help="Path to the dataset.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="data/scene_cls.json",
    help="Path to the output dataset.",
)
parser.add_argument(
    "--task",
    type=str,
    default="scene_cls",
    choices=["scene_cls", "vqa", "caption", "grounding"],
    help="Task type.",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=1,
    help="Batch size for inference.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of workers for data loading.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum number of new tokens to generate.",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=1,
    help="GPU device ID to use.",
)
parser.add_argument(
    "--system",
    type=str,
    default="False",
    help="Whether to use system prompt.",
)
args = parser.parse_args()

out_dir = os.path.dirname(args.output_path)
out_name = os.path.basename(args.output_path).split('.')[0]

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

logger_name = "main-logger"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(out_dir, f'{args.task}_{out_name}_log.txt'), mode='w')
log_format = '%(asctime)s %(message)s'
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)

handler = logging.StreamHandler()
fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)

# Load the model

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=4, sci_mode=False)

# Set the device

# random_list = list(range(5000))

# random.shuffle(random_list)

multi_ref = (args.task == 'caption')

class VisionLanguageInfDataset(Dataset):
    def __init__(self, args, processor):

        assert args.json_path[-4:]=='json'

        with open(args.json_path, 'rb') as file:
            self.data_list = json.load(file)

        #self.data_list = [self.data_list[i] for i in random_list[:188]]

        self.processor = processor
        self.multi_ref = (args.task == 'caption')  # 仅 caption 多参考

        self.args = args

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image_paths = data['images']
        conversations = data['conversations']

        # transform to qwen2.5-vl-template
        qwen_message = []

        if self.args.system == 'True':
        
            system = system_prompt
            system_content = {"role": "system", "content": system}
            qwen_message.append(system_content)

            #print('####################### Using system prompt:', system)

        if self.multi_ref:
            turn_num = len(conversations)  # 只有提问
        else:
            turn_num = len(conversations) - 1 # 最后一轮是回答

        k = 0
        for i in range(turn_num): # skip the last one
            conv = conversations[i]
            role = 'user' if conv['from'] == 'human' else 'assistant'
            if role == 'user':
                if '<image>' in conv['value']:
                    content = [
                    {"type": "image", "image": image_paths[k]},
                    {"type": "text", "text": conv['value'].split('<image>')[1].strip()}
                    ]
                    k += 1
                else:
                    content = [{"type": "text", "text": conv['value'].strip()}]
            else:
                content = conv['value'].strip()

            qwen_message.append({"role": role, "content": content})
            

        # ★ 仅 caption 返回多参考；其他任务返回单标签
        if self.multi_ref:
            if 'refs' in data and isinstance(data['refs'], list) and len(data['refs']) >= 1:
                target = data['refs']                       # List[str]
            elif conversations and conversations[-1]['from'] == 'gpt':
                target = [conversations[-1]['value']]       # 兼容：至少有1条
            else:
                target = [""]
        else:
            # 非 caption：仍然取单字符串标签
            if conversations and conversations[-1]['from'] == 'gpt':
                target = conversations[-1]['value']         # str
            else:
                target = ""

        return qwen_message, target, image_paths[0]#, question
    
def collate_fn(batch, processor):
    qwen_messages = []
    ground_truths = []
    image_paths = []
    #questions = []

    for item in batch:
        qwen_message, ground_truth, image_path = item
        qwen_messages.append(qwen_message)
        ground_truths.append(ground_truth) # caption: List[str]；其他：str
        image_paths.append(image_path)
        #questions.append(question)

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
              for msg in qwen_messages
              ]
    image_inputs, _ = process_vision_info(qwen_messages)

    inputs = processor(
        text=prompts,
        images=image_inputs,
        padding=True,
        padding_side = "left",
        return_tensors="pt",
    )

    return inputs, ground_truths, image_paths#, questions

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto", device_map=device)
model.to(device)
processor  = AutoProcessor.from_pretrained(args.model_path)
vldataset = VisionLanguageInfDataset(args, processor)
infer_dataloader = torch.utils.data.DataLoader(vldataset, batch_size=args.batchsize, shuffle=True, collate_fn=lambda x: collate_fn(x, processor), num_workers=args.workers)



# def build_generate_kwargs_thinking(task, max_new_tokens):
#     pad_id = processor.tokenizer.pad_token_id
#     caps = {
#         "caption":    dict(num_beams=1, max_new_tokens=max(max_new_tokens, 512)),
#         "vqa":        dict(num_beams=1, max_new_tokens=max(max_new_tokens, 128)),
#         "scene_cls":  dict(num_beams=1, max_new_tokens=max(max_new_tokens, 128)),
#         "grounding":  dict(num_beams=1, max_new_tokens=max(max_new_tokens, 192)),
#     }
#     kwargs = caps.get(task, {"num_beams": 1, "max_new_tokens": max_new_tokens})
#     return {"do_sample": False, "early_stopping": False, 'temperature': None, "pad_token_id": pad_id, **kwargs}

# def build_generate_kwargs_thinking(task, max_new_tokens):
#     pad_id = processor.tokenizer.pad_token_id
#     caps = {
#         "caption":    dict(num_beams=5, length_penalty=0.8, early_stopping=True, max_new_tokens=max_new_tokens),
#         "vqa":        dict(num_beams=1, max_new_tokens=min(max_new_tokens, 16)),
#         "scene_cls":  dict(num_beams=1, max_new_tokens=min(max_new_tokens, 8)),
#         "grounding":  dict(num_beams=1, max_new_tokens=min(max_new_tokens, 64)),
#     }
#     kwargs = caps.get(task, {"num_beams": 1, "max_new_tokens": max_new_tokens})
#     return {"do_sample": False, "pad_token_id": pad_id, **kwargs}

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

def extract_answer_block(text: str) -> str:
    """抽取 <answer>...</answer> 内的内容；若无该块则回退用原文本。"""
    if not text:
        return ""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return text.strip()
    # 取第一个非空块；都为空则取第一个
    for seg in matches:
        seg = seg.strip()
        if seg:
            return seg
    return matches[0].strip()

predicts = []
labels = []
records = []

model.eval()

with torch.no_grad():
    for i, (batch_inputs, batch_gts, batch_imps) in enumerate(infer_dataloader):

        batch_inputs.to(device)

        # # 动态生成参数
        generate_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'pad_token_id': processor.tokenizer.pad_token_id
        }

        # # 分类任务使用贪婪解码
        if args.task == 'scene_cls':
            generate_kwargs.update({'do_sample': False, 'temperature': None})
        elif args.task == 'caption':
            generate_kwargs.update({
                'do_sample': False,
                'num_beams': 5,           # 可调：3~5
                'length_penalty': 0.8,    # 可调
                'early_stopping': True
            })
        else:
            generate_kwargs.update({
            'do_sample': True,
            'temperature': 0.9,
            'top_p': 0.95,
            'top_k': 50
            })
        
        #generate_kwargs = build_generate_kwargs_thinking(args.task, args.max_new_tokens)
        generated_ids = model.generate(**batch_inputs, **generate_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 注意 single_label 的类型：caption -> List[str]；其他 -> str
        for single_output_text, single_label, single_imp in zip(output_texts, batch_gts, batch_imps):
            
            pred = extract_answer_block(single_output_text)

            records.append({
                'raw_output': single_output_text.strip(),
                'pred': pred,
                'label': single_label,
                'image_path': single_imp
                #'question': single_quest
            })

            predicts.append(pred)
            labels.append(single_label) #### caption任务的话labels 现在是 List[List[str]]

        print(f'inferencing sample batch: [{i}/{len(infer_dataloader)}]')

        #if args.task == 'scene_cls':
        logger.info(f'Batch {i}, predicts: {output_texts}, labels: {batch_gts}')


# Save the results
with open(args.output_path, 'w') as f:
    json.dump(records, f, indent=4)


# 匹配所有非字母数字和空白的字符（即标点）
_punct_re = re.compile(r"[^\w\s]")
_space_re = re.compile(r"\s+")

def normalize_caption(s: str) -> str:
    s = s.strip().lower()       # 去首尾空格并小写化
    s = _punct_re.sub("", s)    # 去掉标点符号
    s = _space_re.sub(" ", s)   # 多个空白合并成一个空格
    return s

if args.task == 'scene_cls':
    acc = np.mean([1 if normalize_caption(pred) == normalize_caption(label) else 0 for pred, label in zip(predicts, labels)])
    acc = acc * 100
    logger.info(f'Task: {str(args.task)}, Accuracy of dataset {str(args.json_path).split("/")[-1]}: {acc:.2f}%')

elif args.task == 'vqa':
    def loose_match(pred, label):
        p, l = normalize_caption(pred), normalize_caption(label)
        return p == l or p in l or l in p

    acc_list = [1 if loose_match(pred, label) else 0 
                for pred, label in zip(predicts, labels)]
    acc = np.mean(acc_list) * 100
    logger.info(f'Task: {str(args.task)}, Accuracy of dataset {str(args.json_path).split("/")[-1]}: {acc:.2f}%')

elif args.task == 'caption':
    # 计算评估指标
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    # 1) 规范化
    norm_preds = [normalize_caption(x) for x in predicts]
    norm_labels = [[normalize_caption(r) for r in refs] for refs in labels]  # List[List[str]]

    #nltk.download('wordnet')

    '''
    多参考：label为包含多个字符串的list；predicts为单字符串列表
    '''

    # 2) BLEU（多参考）
    references_tok = [[r.split() for r in ref_list] for ref_list in norm_labels]  # List[List[List[str]]]
    candidates_tok = [p.split() for p in norm_preds]
    
    metrics = {}
    
    # BLEU 计算（优化权重定义）
    weights = {
        1: (1.0, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (1/3, 1/3, 1/3, 0),
        4: (0.25, 0.25, 0.25, 0.25)
    }

    smooth = SmoothingFunction().method1  # 使用更标准的平滑方法


    for n in range(1, 5):
        metrics[f'BLEU-{n}'] = corpus_bleu(
            list_of_references=references_tok,
            hypotheses=candidates_tok,
            weights=weights[n],
            smoothing_function=smooth
        )

    # 3) CIDEr（多参考）
    #   gts/res 结构：{i: [ref1, ref2, ...]} / {i: [hyp]}
    gts = {i: ref_list for i, ref_list in enumerate(norm_labels)}
    res = {i: [norm_preds[i]] for i in range(len(norm_preds))}
    cider = Cider()
    metrics['CIDEr'], _ = cider.compute_score(gts, res)
    

    # 4) ROUGE（按样本取平均；与 pycocoevalcap 版本可能略有差异）

    from rouge import Rouge

    rouge = Rouge()
    r1_sum, rl_sum = 0.0, 0.0
    valid_cnt = 0      # 真正参与均值计算的样本数
    skipped_cnt = 0    # 因异常/无有效参考而跳过的样本数
    total_cnt = len(norm_preds)

    for hyp, ref_list in zip(norm_preds, norm_labels):
        hyp = hyp.strip()
        # 情况A：模型真输出空串 -> 计0分，计入分母
        if not hyp:
            r1_sum += 0.0
            rl_sum += 0.0
            valid_cnt += 1
            continue

        # 正常计算：对每个参考求分
        scores = []
        for ref in ref_list:
            ref = (ref or "").strip()
            if not ref:
                continue  # 跳过空参考
            try:
                s = rouge.get_scores(hyp, ref)[0]  # {'rouge-1':{'f':...}, 'rouge-l':{'f':...}}
                scores.append(s)
            except Exception:
                # rouge 在极端短串或奇异字符时可能报错：跳过该参考
                continue

        # 情况B：该样本没有任何有效得分 -> 这是技术性问题，跳过样本，不计入分母
        if not scores:
            skipped_cnt += 1
            continue

        # 每个指标分别取 max(F1)
        best_r1 = max(sc['rouge-1']['f'] for sc in scores)
        best_rl = max(sc['rouge-l']['f'] for sc in scores)

        r1_sum += best_r1
        rl_sum += best_rl
        valid_cnt += 1

    # 避免除0
    den = max(valid_cnt, 1)
    metrics['ROUGE-1'] = r1_sum / den
    metrics['ROUGE-L'] = rl_sum / den

    # 额外记录覆盖率，方便检查被跳过的比例
    metrics['ROUGE_valid_coverage'] = valid_cnt / total_cnt
    metrics['ROUGE_skipped_samples'] = skipped_cnt
    
    # 5) METEOR（nltk 版本；集群上 wordnet 不稳定的话可跳过或换 pycocoevalcap 的 METEOR）

    meteor_list = []
    for ref_list, hyp in zip(norm_labels, norm_preds):
        # 如果你想自己控制分词，可用 r.split() / hyp.split()
        scores = [meteor_score([r.split()], hyp.split()) for r in ref_list]
        meteor_list.append(max(scores))
    metrics['METEOR'] = sum(meteor_list) / len(meteor_list)

    '''
    单参考：labels为单字符串列表；predicts为单字符串列表
    '''

    # metrics = {}

    # # 1) 规范化
    # norm_preds = [normalize_caption(x) for x in predicts]
    # norm_labels = [[normalize_caption(r) for r in refs] for refs in labels]  # List[List[str]]

    # ############ BLEU 计算（单参考支持）

    # references = [[caption.split()] for caption in labels] # 每个样本对应1个参考
    # candidates = [pred.split() for pred in predicts]
    
    # weights = {
    #     1: (1.0, 0, 0, 0),
    #     2: (0.5, 0.5, 0, 0),
    #     3: (0.333, 0.333, 0.333, 0),
    #     4: (0.25, 0.25, 0.25, 0.25)
    # }
    # smooth = SmoothingFunction().method1

    # for n in range(1, 5):
    #     metrics[f'BLEU-{n}'] = corpus_bleu(
    #         list_of_references=references,  # 现在每个样本对应5个参考
    #         hypotheses=candidates,
    #         weights=weights[n],
    #         smoothing_function=smooth
    #     )

    # ############ CIDER 计算
    # ref_dict = {i: [label] for i, label in enumerate(labels)}  # 直接使用原始标签列表
    # cand_dict = {i: [pred] for i, pred in enumerate(predicts)}               # 保持候选格式不变

    # cider = Cider()
    # metrics['CIDER'], _ = cider.compute_score(ref_dict, cand_dict)
    
    # ############ ROUGE 计算
    # rouge = Rouge()

    # rouge_1_scores = []
    # rouge_l_scores = []

    # for hyp, ref in zip(predicts, labels):
    #     score = rouge.get_scores(hyp, ref)[0] 
    #     rouge_1_scores.append(score["rouge-1"]["f"])
    #     rouge_l_scores.append(score["rouge-l"]["f"])

    # # 取所有样本的平均
    # metrics['ROUGE-1'] = sum(rouge_1_scores) / len(rouge_1_scores)
    # metrics['ROUGE-L'] = sum(rouge_l_scores) / len(rouge_l_scores)

    # ############ METEOR 计算

    # meteor_scores = []

    # for ref, hyp in zip(labels, predicts):
        
    #     # 计算候选与每个参考的 METEOR 得分
    #     score = meteor_score([ref.split()], hyp.split())
    #     meteor_scores.append(score)

    # # 全局平均
    # metrics['METEOR'] = sum(meteor_scores) / len(meteor_scores)

    logger.info(f'Task: {str(args.task)}, Accuracy of dataset {str(args.json_path).split("/")[-1]}')
    logger.info(metrics)

# borrow from https://github.com/fitzpchao/RSEvalKit/blob/master/model_eval_mp.py
elif args.task == 'grounding':
    import re
    EXTRACT_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
    def extract_bbox(text):
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1:
            answer_numbers = EXTRACT_NUMBER_PATTERN.findall(text[start_index:end_index+1])
            return [float(number) for number in answer_numbers]
        else:
            return None

    def intersection_geo(box1, box2):
        # 解包两个矩形框的坐标
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        # 计算交集的坐标
        x_min_int = max(x_min1, x_min2)
        y_min_int = max(y_min1, y_min2)
        x_max_int = min(x_max1, x_max2)
        y_max_int = min(y_max1, y_max2)

        return x_min_int, y_min_int, x_max_int, y_max_int

    def calculate_area(box):
        x_min1, y_min1, x_max1, y_max1 = box
        area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        return area_box1

    def calculate_iou(box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2
        x_min_int, y_min_int, x_max_int, y_max_int = intersection_geo(box1, box2)

        # 如果没有交集，直接返回0
        if x_min_int >= x_max_int or y_min_int >= y_max_int:
            return 0.0

        # 计算交集的面积
        area_int = (x_max_int - x_min_int) * (y_max_int - y_min_int)

        area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)
        iou = area_int / (area_box1 + area_box2 - area_int)
        return iou
    
    def is_valid_bbox(bbox):
        return (
            isinstance(bbox, list)          # 必须是列表类型
            and len(bbox) == 4              # 必须包含4个元素
            and all(isinstance(x, (int, float)) for x in bbox)  # 所有元素必须是数字
        )
    
    AREA_LEVEL = (32**2, 96**2, float('inf'))
    LEVEL_NAME = ('S', 'M', 'L')
    level_count = np.zeros([len(AREA_LEVEL)])
    level_correct_count = np.zeros([len(AREA_LEVEL)])

    cnt=0
    for pred, label, record in zip(predicts, labels, records):
        answer_bbox_ori = extract_bbox(label)
        pred_bbox_ori = extract_bbox(pred)

        image_path = record['image_path']
        img = Image.open(image_path)
        w, h = img.size

        img.close()

        if answer_bbox_ori is not None and is_valid_bbox(answer_bbox_ori):

            l = 0

            while calculate_area(answer_bbox_ori) > AREA_LEVEL[l]:
                l += 1
            level_count[l] += 1

            answer_bbox = [
                float(answer_bbox_ori[0] / w),
                float(answer_bbox_ori[1] / h),
                float(answer_bbox_ori[2] / w),
                float(answer_bbox_ori[3] / h),
            ]

            if pred_bbox_ori is not None and is_valid_bbox(pred_bbox_ori):

                # Convert the predicted bounding box to the original image size
                x1_orig, y1_orig, x2_orig, y2_orig = reverse_from_qwen3vl_format(pred_bbox_ori, h, w)
                pred_bbox_ori = [x1_orig, y1_orig, x2_orig, y2_orig]

                pred_bbox = [
                    float(pred_bbox_ori[0] /w),
                    float(pred_bbox_ori[1]/ h),
                    float(pred_bbox_ori[2]/ w),
                    float(pred_bbox_ori[3]/ h),
                ]
                iou = calculate_iou(answer_bbox, pred_bbox)
                logger.info(f'[{cnt}/{len(records)}] answer_bbox:{answer_bbox}, pred_bbox:{pred_bbox}, iou:{iou}')
                cnt += 1

                if iou >= 0.5:
                    level_correct_count[l] += 1
            else:
                pass

        else:
            pass

    precision = np.sum(level_correct_count) / sum(level_count)

    level_precision = level_correct_count / level_count

    logger.info(f'Task: {str(args.task)}, Accuracy of dataset {str(args.json_path).split("/")[-1]}')

    for i, name in enumerate(LEVEL_NAME):
        tmp_level_precision = level_precision[i] * 100
        logger.info(f'Level {name} box count: {level_count[i]}, precision:{tmp_level_precision:.2f}%')

    overall_precision = precision*100
    logger.info(f'Overall precision: {overall_precision:.2f}')

    