import json
import re
import os

# ========== 配置 ========   'swap_manipulation': "B. Face swap and text swap.",      # ✅ 同时有
describe_temple = "The following are multiple choice questions about fake news detection. \n\nThe caption of news is: "
describe_ques_latter = ". The identity and emotion of the face, and the semantic and sentiment of the text should not be manipulated. Question: Is there any face swap/attribute or text_swap in the news?\nA. No.\nB. Only face swap.\nC. Only face attribute.\nD. Only text swap.\nE. Face swap and text swap.\nF. Face attribute and text swap.\nThe options is:"
face_text_locate = "If there is manipulation of a face, locate the most likely manipulated face in the image and append the results to your selected option. If there is text_swap, list all swapped words in the caption.\nThe answer is:"

# ✅ 修正：选项必须严格匹配题干
describles_answ = {
    'orig': "A. No.",
    'swap_manipulation': "E. Face swap and text swap.",      # ✅ 同时有 face swap + text swap → 选 E
    'attribute_manipulation': "F. Face attribute and text swap."  # ✅ 同时有 attribute + text swap → 选 F
}

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(r"\s{2,}", ' ', caption).rstrip('\n').strip()
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

def extract_swapped_words(caption, fake_text_pos):
    words = caption.split()
    swapped_words = [words[i] for i in fake_text_pos if 0 <= i < len(words)]
    return f" Swapped words: {', '.join(swapped_words)}" if swapped_words else ""

def denormalize_fake_image_box_xyxy(fake_image_box, image_width, image_height):
    """将归一化 [cx, cy, w, h] 转为绝对像素 [x1, y1, x2, y2]"""
    cx, cy, w, h = fake_image_box
    x1 = (cx - w / 2) * image_width
    y1 = (cy - h / 2) * image_height
    x2 = (cx + w / 2) * image_width
    y2 = (cy + h / 2) * image_height
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

def get_bbox(bbox):
    """假设 bbox 是 [x, y, w, h] 像素坐标，转为 [x1, y1, x2, y2]"""
    xmin, ymin, w, h = bbox
    return [int(xmin), int(ymin), int(xmin + w), int(ymin + h)]

# ========== 主函数 ==========
def generate_qwen_vl_jsonl_from_list(
    data_list: list,
    output_jsonl_path: str = "train.json",
    base_image_dir: str = "./"  # 图像根目录
):
    samples = []
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in data_list:
            # ✅ 修正：简化路径处理，避免 unicode_escape 破坏路径
            img_path = os.path.join(base_image_dir, item['image'])
            # 只做基础清理，避免破坏合法路径
            img_path = img_path.replace("'", "").replace(",_", "_")

            caption = pre_caption(item['text'], 30)
            question = '<image>\n' + describe_temple + caption + describe_ques_latter + face_text_locate

            label = item['fake_cls']
            answer = describles_answ.get(label, "A. No.")  # 默认安全值

            # ✅ 修正：只在有篡改时添加 bbox 和 swapped words
            if label in ['swap_manipulation', 'attribute_manipulation']:
                # ✅ 重要：判断 bbox 是归一化还是像素，选择处理方式
                fake_image_box = item.get('fake_image_box', [0,0,0,0])
                x1, y1, x2, y2 = get_bbox(fake_image_box)
                # ✅ 修正：用自然语言描述 bbox，而不是 JSON 格式
                answer += f"\nManipulated face bbox: [{x1}, {y1}, {x2}, {y2}]"
            # 添加被替换的词
            fake_text_pos = item.get('fake_text_pos', [])
            if fake_text_pos:
                answer += extract_swapped_words(caption, fake_text_pos)
            sample = {
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer}
                ]
            }
            samples.append(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 已生成 {output_jsonl_path}，共 {len(data_list)} 条数据")


# ========== 执行 ==========
if __name__ == "__main__":
    train_js = '/scratch-shared/khe/qwen/SAMM/SAMM-with-CAP/train.json'
    with open(train_js, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    generate_qwen_vl_jsonl_from_list(train_data, "train.jsonl", "/scratch-shared/khe/qwen/SAMM/")
