import json
import numpy as np
import os
from tqdm import tqdm

def get_confmat(pred_list, threshold=0.1):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = (gt[0] - (gt[2] / 2.), gt[1] - (gt[3] / 2.), gt[0] + (gt[2] / 2.), gt[1] + (gt[3] / 2.))
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False

    # tp, tn, fp, fn
    conf_mat = np.zeros((4))
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp = False
            new_preds = []
            for pred_box in pred_boxes:
                if true_positive(gt_box, pred_box):
                    add_tp = True
                else:
                    new_preds.append(pred_box)
            pred_boxes = new_preds
            if add_tp:
                out_array[0] += 1
            else:
                out_array[3] += 1
        out_array[2] = len(pred_boxes)
        conf_mat += out_array
    return conf_mat

def calc_froc_threshold(pred_data, fps_req=[0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 1.5, 2, 3, 4], num_thresh=1000):
    num_images = len(pred_data)
    thresholds = np.linspace(0, 1, num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    for i, thresh_val in enumerate(tqdm(thresholds)):
        conf_mat = get_confmat(pred_data, thresh_val)
        conf_mat_thresh[i] = conf_mat

    sensitivity = np.zeros((num_thresh))  # recall
    specificity = np.zeros((num_thresh))  # precision
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if (conf_mat[0] + conf_mat[3]) == 0:
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0] / (conf_mat[0] + conf_mat[3])
        if (conf_mat[0] + conf_mat[2]) == 0:
            specificity[i] = 0
        else:
            specificity[i] = conf_mat[0] / (conf_mat[0] + conf_mat[2])

    senses_req = []
    froc_thresh = []
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2]
            if f / num_images < fp_req:
                senses_req.append(sensitivity[i - 1])
                froc_thresh.append(thresholds[i])
                break
    return fps_req, senses_req, froc_thresh, specificity, senses_req

def save_predictions_at_fpi(pred_data, fpi_value, output_dir):
    # Calculate FROC curve
    fps_req, senses_req, froc_thresh, specificity, _ = calc_froc_threshold(pred_data)

    # Find the threshold corresponding to the desired FPI value
    threshold_index = np.argmin(np.abs(np.array(fps_req) - fpi_value))
    threshold = froc_thresh[threshold_index]

    # Filter predictions using the threshold
    filtered_predictions = []
    for data_item in pred_data:
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        filtered_pred = {
            'image': data_item['image'],
            'masks': data_item['masks'],
            'target': data_item['target'],
            'pred': {
                'boxes': pred['boxes'][select_mask],
                'scores': pred['scores'][select_mask],
                'labels': pred['labels'][select_mask]
            },
            'image_id': data_item['image_id']
        }
        filtered_predictions.append(filtered_pred)

    # Save filtered predictions to a JSON file
    output_path = os.path.join(output_dir, f'predictions_at_fpi_{fpi_value}.json')
    with open(output_path, 'w') as f:
        json.dump(filtered_predictions, f, indent=4)

    print(f"Predictions saved at: {output_path}")

# Usage in eval_only function
def eval_only(model, device):
    if args.distributed:
        Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    test_loader = build_dataloader(args, args.target_dataset, 'target', 'test', val_trans)

    pred_list_test = evaluate_froc(
        model=model,
        criterion=criterion,
        data_loader_val=test_loader,
        output_result_labels=True,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )

    combined_dict = {}
    for key in pred_list_test[0].keys():
        if key == 'images':
            combined_values = [item for sublist in [d[key] for d in pred_list_test] for item in sublist]
        else:
            combined_values = [item for sublist in [d[key] for d in pred_list_test] for item in sublist]
        combined_dict[key] = combined_values

    new_dict = []
    image_length = len(combined_dict['images'])
    for i in range(image_length):
        new_dict.append({
            'image': combined_dict['images'][i],
            'masks': combined_dict['masks'][i],
            'target': combined_dict['target'][i],
            'pred': combined_dict['pred'][i],
            'image_id': combined_dict['image_id'][i]
        })

    # Save predictions at a specific FPI value
    desired_fpi = 0.3  # Change this to your desired FPI value
    save_predictions_at_fpi(new_dict, desired_fpi, args.output_dir)

    # Calculate FROC and other metrics
    test_froc, test_fpi, test_recall, test_pres = calc_froc(new_dict)
    print(test_froc, test_fpi)