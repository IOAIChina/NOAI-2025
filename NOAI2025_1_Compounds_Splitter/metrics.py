import os
import json

dataset_dir = './'

def get_segments(word, labels):
    """
    根据单词和对应的标签数组生成分词列表。
    每个分词用 (start, end) 表示，区间为 [start, end)，其中 end 表示分词结束位置（该位置包含的字符在分词中）。
    """
    segments = []
    start = 0
    for i, label in enumerate(labels):
        if label == 1:
            segments.append((start, i + 1))
            start = i + 1
    # 如果最后还有剩余部分（理论上应该不会出现这种情况，除非最后一个标签不是1）
    if start < len(word):
        segments.append((start, len(word)))
    return segments

def evaluate_f1(original_file, predicted_file):
    """
    比较原始 JSON 文件与预测 JSON 文件，计算各词的分词 F1 得分，最后返回 F1 均值。
    
    JSON 文件格式：
    {
        "word1": [labels...],
        "word2": [labels...],
        ...
    }
    其中 labels 数组的长度应与 word 的字符数一致，1 表示对应位置为分词结尾，0 表示非结尾。
    
    若预测标签序列长度与真实标签序列长度不一致，则直接判定该样本 F1 为 0。
    """
    with open(original_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    with open(predicted_file, "r", encoding="utf-8") as f:
        predicted_data = json.load(f)

    f1_list = []
    total_samples = len(original_data)
    
    for word, gold_labels in original_data.items():
        pred_labels = predicted_data.get(word, None)
        # 若预测结果不存在或长度不一致，则该样本 F1 直接为 0
        if pred_labels is None or len(pred_labels) != len(gold_labels):
            f1_list.append(0)
            continue

        # 根据标签还原出分词边界
        gold_segments = get_segments(word, gold_labels)
        pred_segments = get_segments(word, pred_labels)

        # 只有完全匹配分词边界才算作正确的分词
        gold_set = set(gold_segments)
        pred_set = set(pred_segments)
        tp = len(gold_set & pred_set)

        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gold_set) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_list.append(f1)

    f1_avg = sum(f1_list) / total_samples if total_samples > 0 else 0
    print(f"Average F1: {f1_avg * 100:.2f}%")
    return f1_avg

if __name__ == '__main__':
    # 计算验证集的 F1 均值
    f1_val = evaluate_f1('valans.json', 'submissionval.json')
    print(f"Validation F1: {f1_val:.4f}")
    # 计算测试集的 F1 均值
    f1_test = evaluate_f1('testans.json', 'submissiontest.json')
    print(f"Test F1: {f1_test:.4f}")
    
    score = {
        "public_a": f1_val,
        "private_b": f1_test,
    }
    
    ret_json = {
        "status": True,
        "score": score,
        "msg": "Success!",
    }
    with open(os.path.join(dataset_dir, 'score.json'), 'w', encoding="utf-8") as f:
        f.write(json.dumps(ret_json, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    scoreval = evaluate_accuracy('valans.json', 'submissionval.json')
    print(scoreval)
    scoretest =  evaluate_accuracy('testans.json', 'submissiontest.json')
    print(scoretest)
    score = {
        "public_a": scoreval,
        "private_b": scoretest,
    }
    #print(score)
    ret_json = {
        "status": True,
        "score": score,
        "msg": "Success!",
    }
    with open(os.path.join(dataset_dir,'score.json'), 'w') as f:
        f.write(json.dumps(ret_json))   
