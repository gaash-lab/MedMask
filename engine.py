import time
import datetime
import json
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator

from models.criterion import post_process, get_pseudo_labels, get_pred_dict
from utils.distributed_utils import is_main_process
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List
from tqdm import tqdm
import csv
import torch.nn.functional as F

def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# def contrastive_loss(embeddings1, embeddings2, temperature=0.1):
#     # Step 1: Align Sequence Lengths
#     min_seq_len = min(embeddings1.size(1), embeddings2.size(1))
#     embeddings1 = embeddings1[:, :min_seq_len, :]
#     embeddings2 = embeddings2[:, :min_seq_len, :]

#     # Step 2: Match Feature Dimensions
#     # Project embeddings to the same feature dimension (e.g., 256)
#     target_dim = 256
#     projection1 = nn.Linear(embeddings1.size(-1), target_dim).to(embeddings1.device)
#     projection2 = nn.Linear(embeddings2.size(-1), target_dim).to(embeddings2.device)

#     embeddings1 = projection1(embeddings1)
#     embeddings2 = projection2(embeddings2)

#     # Flatten the embeddings
#     embeddings1 = embeddings1.view(embeddings1.size(0), -1)
#     embeddings2 = embeddings2.view(embeddings2.size(0), -1)

#     # Step 3: Normalize the embeddings
#     embeddings1 = F.normalize(embeddings1, p=2, dim=1)
#     embeddings2 = F.normalize(embeddings2, p=2, dim=1)

#     # Step 4: Compute the similarity matrix
#     similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / temperature

#     # Step 5: Compute the contrastive loss
#     labels = torch.arange(embeddings1.size(0)).to(embeddings1.device)
#     loss = F.cross_entropy(similarity_matrix, labels)
    
#     return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ntxent_loss(emb1, emb2, temperature=0.1):
    """
    NT-Xent (InfoNCE) loss.
    Computes a similarity matrix between emb1 and emb2 and applies cross entropy loss.
    """
    # similarity matrix: (batch, batch)
    sim_matrix = torch.matmul(emb1, emb2.T) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def compute_triplet_loss(emb1, emb2, margin=1.0):
    """
    Triplet loss using hard negative mining from the batch.
    For each anchor (from emb1) and positive (its corresponding emb2), the hardest negative
    is chosen as the most similar non-matching emb2.
    """
    batch_size = emb1.size(0)
    # Compute pairwise Euclidean distances between emb1 (anchors) and emb2 (others)
    distances = torch.cdist(emb1, emb2, p=2)  # shape: (batch, batch)
    # Positive distances are on the diagonal.
    pos_dist = distances.diag()
    # For negatives, mask out the diagonal.
    mask = torch.eye(batch_size, dtype=torch.bool, device=emb1.device)
    distances.masked_fill_(mask, float('inf'))
    # Hardest negative: the minimum distance among non-matching pairs.
    neg_dist, _ = distances.min(dim=1)
    loss = F.relu(pos_dist - neg_dist + margin).mean()
    return loss

def compute_margin_contrastive_loss(emb1, emb2, margin=1.0):
    """
    Margin-based contrastive loss.
    For a positive pair (matching indices), the loss is the squared distance.
    For negative pairs (all off-diagonals), the loss is the squared hinge:
        max(0, margin - distance)^2.
    """
    batch_size = emb1.size(0)
    # Compute pairwise distances between emb1 and emb2.
    distances = torch.cdist(emb1, emb2, p=2)  # shape: (batch, batch)
    # Positive pairs (diagonal)
    pos_dist = distances.diag()
    loss_pos = pos_dist ** 2

    # Negative pairs (off-diagonals)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=emb1.device)
    neg_dist = distances[mask]
    loss_neg = F.relu(margin - neg_dist) ** 2

    # Combine and average the losses.
    loss = torch.cat([loss_pos, loss_neg]).mean()
    return loss

def compute_cosine_embedding_loss(emb1, emb2, margin=0.0):
    """
    Cosine embedding loss.
    Expects a target of 1 for similar pairs.
    """
    batch_size = emb1.size(0)
    target = torch.ones(batch_size, device=emb1.device)
    loss = F.cosine_embedding_loss(emb1, emb2, target, margin=margin)
    return loss

def contrastive_loss(embeddings1, embeddings2, loss_type='margin', temperature=0.1, margin=1.0):
    """
    Computes a contrastive loss between two sets of embeddings.
    
    Arguments:
        embeddings1, embeddings2: Tensors of shape (batch, seq_len, feature_dim).
        loss_type: One of 'NTXent', 'triplet', 'margin', or 'cosine'.
        temperature: Temperature scaling for NTXent.
        margin: Margin parameter for triplet and margin losses.
    
    Returns:
        A scalar loss value.
    """
    # Step 1: Align Sequence Lengths.
    min_seq_len = min(embeddings1.size(1), embeddings2.size(1))
    embeddings1 = embeddings1[:, :min_seq_len, :]
    embeddings2 = embeddings2[:, :min_seq_len, :]

    # Step 2: Project to a common target dimension (e.g., 256).
    target_dim = 256
    projection1 = nn.Linear(embeddings1.size(-1), target_dim).to(embeddings1.device)
    projection2 = nn.Linear(embeddings2.size(-1), target_dim).to(embeddings2.device)
    embeddings1 = projection1(embeddings1)
    embeddings2 = projection2(embeddings2)

    # Flatten the sequence dimension so that each sample is represented as a single vector.
    embeddings1 = embeddings1.view(embeddings1.size(0), -1)
    embeddings2 = embeddings2.view(embeddings2.size(0), -1)

    # Step 3: Normalize the embeddings.
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Step 4: Compute the selected contrastive loss.
    if loss_type == 'NTXent':
        loss = compute_ntxent_loss(embeddings1, embeddings2, temperature=temperature)
    elif loss_type == 'triplet':
        loss = compute_triplet_loss(embeddings1, embeddings2, margin=margin)
    elif loss_type == 'margin':
        loss = compute_margin_contrastive_loss(embeddings1, embeddings2, margin=margin)
    elif loss_type == 'cosine':
        loss = compute_cosine_embedding_loss(embeddings1, embeddings2, margin=margin)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def train_one_epoch_with_mae(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_mae: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             mae_loader: DataLoader,
                             coef_target: float,
                             mask_ratio: float,
                             optimizer: torch.optim.Optimizer,
                             optimizer_mr: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True,
                             embeddings_dir: str = None,
                             train_annotations_path: str = None):
    start_time = time.time()
    model.train()
    criterion.train()
    criterion_mae.train()
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    mae_fetcher = DataPreFetcher(mae_loader, device=device)
    source_images, source_masks, source_annotations = source_fetcher.next()
    target_images, target_masks, _ = target_fetcher.next()
    mae_images, mae_masks, _ = mae_fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        # Source forward
        out = model(source_images, source_masks)
        # Target forward
        out_mae = model(mae_images, mae_masks, enable_mae=True, mask_ratio=mask_ratio)
        
        # Get embeddings from Deformable DETR
        def_embeddings = out['embeddings']
        
        # Load corresponding Biomed embeddings
        biomed_embeddings_list = []

        # Load the annotations
        with open(train_annotations_path, 'r') as f:
            coco_annotations = json.load(f)

        image_id_to_filename = {item['id']: item['file_name'] for item in coco_annotations['images']}

        # Assuming source_annotations is a list of dictionaries
        for annotation in source_annotations:
            # Get image_id from the dictionary
            image_id = annotation['image_id'].item()  # Change this key if your structure is different
            # Now, check if this image_id is in the dictionary
            if image_id in image_id_to_filename:
                image_name = image_id_to_filename[image_id]
                
                # Get the base filename without extension
                filename = os.path.splitext(image_name)[0]
                
                # Construct the embedding path
                embedding_path = os.path.join(embeddings_dir, filename + '_embedding.npy')

                # Load the embedding if it exists
                if os.path.exists(embedding_path):
                    biomed_embedding = np.load(embedding_path)
                    biomed_embedding = torch.from_numpy(biomed_embedding).float().to(device)
                    biomed_embeddings_list.append(biomed_embedding)
                else:
                    print(f"Warning: Embedding not found for {filename}")
                    biomed_embeddings_list.append(torch.zeros_like(def_embeddings[0]))
            else:
                print(f"Warning: Image ID {image_id} not found in annotations.")


        # Stack to create a batch tensor
        biomed_embeddings = torch.stack(biomed_embeddings_list)
        
        # Compute contrastive loss
        contrastive_loss_value = contrastive_loss(def_embeddings, biomed_embeddings)
        lambda_contrastive = 1
        
        # Loss
        loss, loss_dict = criterion(out, source_annotations)
        loss_mae, loss_dict_mae = criterion_mae(out_mae, enable_mae=True)
        loss += loss_mae * coef_target + lambda_contrastive * contrastive_loss_value  # Add contrastive loss to the total loss
        loss_dict['loss_mae'] = loss_dict_mae['loss_mae']
        loss_dict['contrastive_loss'] = contrastive_loss_value.item()
        # Backward
        optimizer.zero_grad()
        optimizer_mr.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                epoch_loss_dict[k] += v.detach().cpu().item()
            else:
                epoch_loss_dict[k] += v  # if v is already a float

        # Data pre-fetch
        source_images, source_masks, source_annotations = source_fetcher.next()

        target_images, target_masks, _ = target_fetcher.next()
        mae_images, mae_masks, _ = mae_fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Cross-domain MAE training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' +
                  str(total_iters) + ' ] ' + 'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= total_iters
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Cross-domain MAE training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


# def train_one_epoch_with_mae(model: torch.nn.Module,
#                              criterion: torch.nn.Module,
#                              criterion_mae: torch.nn.Module,
#                              source_loader: DataLoader,
#                              target_loader: DataLoader,
#                              mae_loader:DataLoader,
#                              coef_target: float,
#                              mask_ratio: float,
#                              optimizer: torch.optim.Optimizer,
#                              optimizer_mr: torch.optim.Optimizer,
#                              device: torch.device,
#                              epoch: int,
#                              clip_max_norm: float = 0.0,
#                              print_freq: int = 20,
#                              flush: bool = True):
#     start_time = time.time()
#     model.train()
#     criterion.train()
#     criterion_mae.train()
#     source_fetcher = DataPreFetcher(source_loader, device=device)
#     target_fetcher = DataPreFetcher(target_loader, device=device)
#     mae_fetcher    = DataPreFetcher(mae_loader, device=device)
#     source_images, source_masks, source_annotations = source_fetcher.next()
#     target_images, target_masks, _ = target_fetcher.next()
#     mae_images, mae_masks, _ = mae_fetcher.next()
#     # Training statistics
#     epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
#     epoch_loss_dict = defaultdict(float)
#     total_iters = min(len(source_loader), len(target_loader))
#     for i in range(total_iters):
#         # Source forward
#         out = model(source_images, source_masks)
#         # Target forward
#         out_mae = model(mae_images, mae_masks, enable_mae=True, mask_ratio=mask_ratio)
        
#         # Embedding forward

#         def_embeddings = out['embeddings']

#         # get two embedings one from features from def_detr.py and biomed parse saved embidings
#         # compute the contrastive loss bw two embeddings (same size)

#         # Loss
#         loss, loss_dict = criterion(out, source_annotations)
#         loss_mae, loss_dict_mae = criterion_mae(out_mae, enable_mae=True)
#         loss += loss_mae * coef_target
#         loss_dict['loss_mae'] = loss_dict_mae['loss_mae']
#         # Backward
#         optimizer.zero_grad()
#         optimizer_mr.zero_grad()
#         loss.backward()
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
#         optimizer.step()
#         # Record loss
#         epoch_loss += loss
#         for k, v in loss_dict.items():
#             epoch_loss_dict[k] += v.detach().cpu().item()
#         # Data pre-fetch
#         source_images, source_masks, source_annotations = source_fetcher.next()
#         target_images, target_masks, _ = target_fetcher.next()
#         mae_images, mae_masks, _ = mae_fetcher.next()
#         # Log
#         if is_main_process() and (i + 1) % print_freq == 0:
#             print('Cross-domain MAE training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' +
#                   str(total_iters) + ' ] ' + 'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
#     # Final process of training statistic
#     epoch_loss /= total_iters
#     for k, v in epoch_loss_dict.items():
#         epoch_loss_dict[k] /= total_iters
#     end_time = time.time()
#     total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
#     print('Cross-domain MAE training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
#           ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
#     return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching(student_model: torch.nn.Module,
                             teacher_model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_pseudo: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             thresholds: List[float],
                             coef_target: float,
                             mask_ratio: float,
                             alpha_ema: float,
                             device: torch.device,
                             epoch: int,
                             enable_mae: bool = False,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion.train()
    criterion_pseudo.train()
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    source_images, source_masks, source_annotations = source_fetcher.next()
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # Training data statistics
    epoch_source_loss_dict = defaultdict(float)
    epoch_target_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        # Source forward
        source_out = student_model(source_images, source_masks)
        source_loss, source_loss_dict = criterion(source_out, source_annotations, domain_label=0)
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logits_all'][-1], teacher_out['boxes_all'][-1], thresholds)
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks, enable_mae, mask_ratio)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels, 1, enable_mae)
        # Backward
        optimizer.zero_grad()
        loss = source_loss + coef_target * target_loss
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()
        # Record epoch losses
        epoch_loss += loss
        # update loss_dict
        for k, v in source_loss_dict.items():
            epoch_source_loss_dict[k] += v.detach().cpu().item()
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()
        # EMA update teacher
        with torch.no_grad():
            state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
            for key, value in state_dict.items():
                state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
            teacher_model.load_state_dict(state_dict)
        # Data pre-fetch
        source_images, source_masks, source_annotations = source_fetcher.next()
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_source_loss_dict.items():
        epoch_source_loss_dict[k] /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_source_loss_dict, epoch_target_loss_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False, 
             conf_threshold: float = 0.1):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        image_ids = [img['id'] for img in coco_data['images']]
        image_id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
        dataset_annotations = [[] for _ in range(len(coco_data['images']))]
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logits_all, boxes_all = out['logits_all'], out['boxes_all']
        # Get pseudo labels
        if output_result_labels:
            # results = get_pseudo_labels(logits_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
            results = get_pseudo_labels(logits_all[-1], boxes_all[-1], [conf_threshold for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    # import pdb; pdb.set_trace()
                    image_id = anno['image_id'].item()
                    index = image_id_to_index[image_id]
                    dataset_annotations[index].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logits_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        for image_anno in dataset_annotations:
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)




@torch.no_grad()
def evaluate_csv(model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 data_loader_val: DataLoader,
                 device: torch.device,
                 print_freq: int,
                 output_result_labels: bool = False,
                 flush: bool = False):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        dataset_annotations = [[] for _ in range(len(coco_data['images']))]
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    results_to_save = []
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logits_all, boxes_all,  = out['logits_all'], out['boxes_all']
        # Get pseudo labels
        # Thresholds at 0.3 FPi
        # Thres = 
        if output_result_labels:
            results = get_pseudo_labels(logits_all[-1], boxes_all[-1], [0.001 for _ in range(2)])
            for anno, res in zip(annotations, results):

                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = box_cxcywh_to_xyxy(res['boxes'] * scale_fct)
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    if label in [0, 1]:
                        pseudo_anno = {
                            'id': 0,
                            'image_id': image_id,
                            'category_id': label,
                            'iscrowd': 0,
                            'area': box[-2] * box[-1],
                            'bbox': box
                        }
                        dataset_annotations[image_id].append(pseudo_anno)
                        # Save results for CSV
                        results_to_save.append({
                        'image_name': image_id,  # Assuming image_id is the image name
                        'confidence_score': res['scores'].detach().cpu().numpy().max(),  # Confidence score of highest box
                        'bounding_box': np.array(box),  # Convert bounding box to NumPy array
                    })
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logits_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    
    # Apply non-maximum suppression (NMS) to get only one box per image
    results_to_save_nms = []
    for result in results_to_save:
        if result['image_name'] not in [res['image_name'] for res in results_to_save_nms]:
            results_to_save_nms.append(result)
    
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    
    # Save results to CSV
    if output_result_labels:
        csv_filename = './outputs/outputs.csv'
        with open(csv_filename, mode='w', newline='') as csv_file:
            fieldnames = ['image_name', 'confidence_score', 'bounding_box']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results_to_save_nms:
                writer.writerow(result)
        print("Saved outputs to csv at : ", csv_filename)
    return aps, epoch_loss / len(data_loader_val)


@torch.no_grad()
def evaluate_froc(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False):
    
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        dataset_annotations = [[] for _ in range(len(coco_data['images']))]
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    preds = []
    
    # Wrap the data_loader with tqdm to create a progress bar
    for i, (images, masks, annotations) in tqdm(enumerate(data_loader_val), total=len(data_loader_val)):
        # To CUDA
        item_info = {}
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.cpu() for k, v in t.items()} for t in annotations]

        # import pdb; pdb.set_trace()
        # Forward
        out = model(images, masks)
        logits_all, boxes_all = out['logits_all'], out['boxes_all']
        pred = get_pred_dict(logits_all[-1], boxes_all[-1], [0.000000000000000000000000001 for _ in range(2)])
        item_info['images'] = images.cpu()
        item_info['masks'] = masks.cpu()
        item_info['target'] = annotations
        item_info['image_id'] = annotations
        item_info['pred'] = pred
        
        preds.append(item_info)

    return preds