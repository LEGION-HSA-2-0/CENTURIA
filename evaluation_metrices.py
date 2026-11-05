import re
from collections import Counter

import jiwer
from cotescore import spacer, spacer_micro


def clean_text_for_evaluation(text):

    if not isinstance(text, str):
        text = str(text)
    
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def character_error_rate(predictions, references):
    return jiwer.cer(references, predictions)

def word_error_rate(predictions, references):
    return jiwer.wer(references, predictions)

def bag_of_words_f1(predictions, references):

    pred_tokens = re.findall(r'\w+', str(predictions).lower())
    ref_tokens = re.findall(r'\w+', str(references).lower())

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    common = pred_counts & ref_counts # intersections
    num_common = sum(common.values())

    precision = num_common / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = num_common / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_field(predictions, references):

    results = {}
    def extract_rd(text):
        match = re.search(r"RD\s*[:=]\s*([\d,\./]+)", text, re.IGNORECASE)
        return match.group(1).replace(",", ".") 
    
    ref_rd = extract_rd(references)
    pred_rd = extract_rd(predictions)

    results["rd_found"] = pred_rd is not None
    results["rd_correct"] = (pred_rd == ref_rd) 

    types = ["krug", "teller", "schüssel", "schale", "deckel"]
    ref_types = [t for t in types if t in references.lower()]
    pred_types = [t for t in types if t in predictions.lower()]

    results["types_recall"] = len(set(ref_types) & set(pred_types)) / len(ref_types)
    
    return results

def get_iou(box1, box2):

    x_left = max(box1["x"], box2["x"])
    y_top = max(box1["y"], box2["y"])
    x_right = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y_bottom = min(box1["y"] + box1["h"], box2["y"] + box2["h"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1["w"] * box1["h"]
    box2_area = box2["w"] * box2["h"]
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def calculate_macro_spacer(clean_gt_text, clean_pred_text):

    if not clean_gt_text and not clean_pred_text:
        return 0.0
        
    gt_counter_macro = Counter(clean_gt_text)
    pred_counter_macro = Counter(clean_pred_text)
    
    return spacer(gt_counter_macro, pred_counter_macro)


def calculate_micro_spacer(gt_boxes, pred_boxes):

    if not gt_boxes and not pred_boxes:
        return 0.0, [], []

    gt_list = []
    pred_list = []
    
    unmatched_preds = pred_boxes.copy()
    
    # Greedy IoU Matching
    for gt in gt_boxes:
        best_iou = 0
        best_pred = None
        
        for pred in unmatched_preds:
            iou = get_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
                
        # IoU > 0
        if best_pred and best_iou > 0.0:
            clean_gt = clean_text_for_evaluation(gt.get("bbox_text", ""))
            clean_pred = clean_text_for_evaluation(best_pred.get("bbox_text", ""))
            
            gt_list.append(Counter(clean_gt))
            pred_list.append(Counter(clean_pred))
            unmatched_preds.remove(best_pred)
        else:
            clean_gt = clean_text_for_evaluation(gt.get("bbox_text", ""))
            gt_list.append(Counter(clean_gt))
            pred_list.append(Counter()) 
            
    # Insertions/Halluzinationen
    for pred in unmatched_preds:
        clean_pred = clean_text_for_evaluation(pred.get("bbox_text", ""))
        gt_list.append(Counter()) 
        pred_list.append(Counter(clean_pred))
        

    micro_val = spacer_micro(gt_list, pred_list) 
    
    return micro_val, gt_list, pred_list