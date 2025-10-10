import json
import os

# ========== 配置 ==========
describe_temple = "You are a fake news detector. The news caption is: "
describe_instruction = """
Analyze the image and caption for manipulations. Only consider:
- Face swap (full identity replacement)
- Face attribute manipulation (e.g., emotion, age, gender)
- Text swap in the caption (words replaced with misleading ones)

Choose one option:
A. No manipulation.
B. Face swap and text swap.
C. Face attribute manipulation and text swap.

If B or C, also:
- Give the bounding box of the most likely manipulated face as (x1,y1,x2,y2)
- List all swapped words in the caption as [word1, word2, ...]

Respond strictly in this format:
Option: X
Face: (x1,y1,x2,y2) or None
Text: [word1, word2, ...] or []

Do not explain.
"""

label_to_option = {
    'orig': 'A',
    'swap_manipulation': 'B',
    'attribute_manipulation': 'C'
}

def generate_qwen_vl_jsonl_from_list(
    data_list: list,
    output_jsonl_path: str = "train.jsonl",
    base_image_dir: str = "./images"
):
    with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for item in data_list:
            img_name = item['image']
            img_path = os.path.join(base_image_dir, img_name).replace("\\", "/")

            caption = item['text']
            user_content = f"{describe_temple}\"{caption}\"{describe_instruction}"

            label = item.get('fake_cls', 'orig')
            option = label_to_option.get(label, 'A')

            if option == 'A':
                assistant_content = "Option: A\nFace: None\nText: []"
            else:
                # 假设这里有逻辑获取或计算出正确的face和text信息
                bbox_xyxy = "(0,0,0,0)"  # 示例占位符
                swapped_words = "[]"  # 示例占位符
                
                assistant_content = f"Option: {option}\nFace: {bbox_xyxy}\nText: {swapped_words}"

            sample = {
                "image": img_path,
                "conversations": [
                    {"from": "human", "value": f"{user_content}<image>"},
                    {"from": "gpt", "value": assistant_content}
                ]
            }
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 已生成 {output_jsonl_path}，共 {len(data_list)} 条数据")


# ========== 执行 ==========
if __name__ == "__main__":
    train_js = '/scratch-shared/npu/qwen/SAMM_data/SAMM-with-CAP/test.json'
    with open(train_js, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    generate_qwen_vl_jsonl_from_list(
        train_data,
        output_jsonl_path="test.jsonl",
        base_image_dir="/scratch-shared/npu/qwen/SAMM_data/"  # 替换为实际图像目录
    )
