import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion import create_diffusion
import argparse
import pandas as pd
from models import DiT_models
import json
import numpy as np
from tqdm import tqdm

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data_info, 'r') as file:
        data_info = json.load(file)
        image_data_info = data_info['Image Data']
        condition_info = data_info['Condition Data']

    model = DiT_models[args.model](condition_info=condition_info).to(device)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()  # important!
    model.class_dropout_prob = -1
    diffusion = create_diffusion(str(args.num_sampling_steps))

    sample_data = pd.read_csv(args.sample_file)
    n = len(sample_data)
    average_num = args.average_num
    eid_labels = sample_data['eid'].astype(int).tolist()
    age_labels = sample_data['21003-2.0'].astype(float).tolist()
    sex_labels = sample_data['31-0.0'].astype(int).tolist()

    batch_size = args.batch_size
    save_dir = os.path.join(args.save_dir,f"epoch_{args.ckpt_epoch}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Begin sampling...")
    for i in range(0, n, batch_size):  # (0, n, batch_size)

        for sample_num in range(average_num):

            eid_batch = eid_labels[i:i + batch_size]
            age_batch = age_labels[i:i + batch_size]
            sex_batch = sex_labels[i:i + batch_size]

            real_sub_num = len(eid_batch)
            modality_batch = [m for m in range(args.modality_num)] * real_sub_num

            z = torch.randn(real_sub_num * args.modality_num, 1, args.image_size, device=device)
            y1 = torch.tensor(age_batch, dtype=torch.float32, device=device).repeat_interleave(args.modality_num)
            y2 = torch.tensor(sex_batch, dtype=torch.int32, device=device).repeat_interleave(args.modality_num)
            y3 = torch.tensor(modality_batch, dtype=torch.int32, device=device)

            z = torch.cat([z, z], 0)  # Uncomment if intentionally doubling z

            real_batch_size = real_sub_num * args.modality_num

            y1_null = torch.tensor([0] * real_batch_size, dtype=torch.int32, device=device)
            y1 = torch.cat([y1, y1_null], 0)

            y2_null = torch.tensor([2] * real_batch_size, dtype=torch.int32, device=device)
            y2 = torch.cat([y2, y2_null], 0)

            y3_null = torch.tensor([args.modality_num] * real_batch_size, dtype=torch.int32, device=device)
            y3 = torch.cat([y3, y3_null], 0)
            # Assert tensor shapes

            assert y1.shape == y2.shape == y3.shape, "Condition tensors must have the same shape"

            model_kwargs = dict(y1=y1, y2=y2, y3=y3, cfg_scale=args.cfg_scale)

            with torch.no_grad():
                # Perform sampling with the diffusion model
                if args.use_ddim:
                    samples = diffusion.ddim_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
                else:
                    samples = diffusion.p_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )

            samples, _ = samples.chunk(2, dim=0)

            save_batch_data(eid_batch, samples.cpu(), save_dir, image_data_info, args.modality_num, sample_num)

            print(f"Batch {i // batch_size + 1} , sample num {sample_num}, saved successfully.")
        
        get_average_samples(eid_batch,save_dir,average_num)

    compute_correlation(sample_data,save_dir,average_num)

    compute_similarity(save_dir)

    print(f"All generated images and metadata saved in {save_dir}")




def get_average_samples(eid_batch, save_dir, average_num):
    
    
    gen_eids = eid_batch

    modalities = [  'dti_FA_2mm', 'dti_L1_2mm', 'dti_L2_2mm', 'dti_L3_2mm', 'dti_MD_2mm',
                    'dti_MO_2mm','NODDI_ICVF_2mm','NODDI_ISOVF_2mm','NODDI_OD_2mm','QSM_2mm','SWI_2mm',
                    'T1_brain_nonlinear_2mm','T1_warp_Jac_2mm','T2_FLAIR_brain_to_MNI','T2star_2mm', 'VBM_2mm',
                    'zstat1s','zstat2s','zstat5s','DU15_1','DU15_2','DU15_3','DU15_4','DU15_5',
                    'DU15_6','DU15_7','DU15_8','DU15_9','DU15_10','DU15_11','DU15_12',
                    'DU15_13','DU15_14','DU15_15']

    for eid in gen_eids:
        addr = os.path.join(save_dir, f"generated_{eid}")
        for modality in modalities:
            modality_data = []
            for sample_num in range(average_num):
                data_path = os.path.join(addr,f"{modality}_sample_{sample_num}.npy")
                data = np.load(data_path)
                modality_data.append(data)
            
            modality_data = np.stack(modality_data)
            modality_data = np.mean(modality_data,axis=0)
            save_path = os.path.join(addr,f"{modality}_average_{average_num}.npy")
            np.save(save_path, modality_data)
            


def BWAS_correlation(fMRI_2D_1, fMRI_2D_2):
    fMRI_2D_1 = (fMRI_2D_1 - fMRI_2D_1.mean(axis=0)) / fMRI_2D_1.std(axis=0)
    fMRI_2D_2 = (fMRI_2D_2 - fMRI_2D_2.mean(axis=0)) / fMRI_2D_2.std(axis=0)
    r = np.dot(np.transpose(fMRI_2D_1), fMRI_2D_2) / fMRI_2D_1.shape[0]
    return r

def compute_correlation(gen_with_labels,save_dir,average_num):

    gen_eids = gen_with_labels['eid'].values
    gen_age_labels = gen_with_labels['21003-2.0'].values
    gen_sex_labels = gen_with_labels['31-0.0'].values

    modalities = [  'dti_FA_2mm', 'dti_L1_2mm', 'dti_L2_2mm', 'dti_L3_2mm', 'dti_MD_2mm',
                    'dti_MO_2mm','NODDI_ICVF_2mm','NODDI_ISOVF_2mm','NODDI_OD_2mm','QSM_2mm','SWI_2mm',
                    'T1_brain_nonlinear_2mm','T1_warp_Jac_2mm','T2_FLAIR_brain_to_MNI','T2star_2mm', 'VBM_2mm',
                    'zstat1s','zstat2s','zstat5s','DU15_1','DU15_2','DU15_3','DU15_4','DU15_5',
                    'DU15_6','DU15_7','DU15_8','DU15_9','DU15_10','DU15_11','DU15_12',
                    'DU15_13','DU15_14','DU15_15']

    for modality in modalities:
        gen_data = []
        for eid in gen_eids:
            data = np.load(os.path.join(save_dir,f"generated_{eid}",f'{modality}_average_{average_num}.npy'))
            gen_data.append(data)
        gen_data = np.array(gen_data).reshape(len(gen_data),-1)
        gen_age_coef = []
        gen_sex_coef = []
        for i in range(228453):
            gen_age_coef.append(BWAS_correlation(gen_data[:,i],gen_age_labels))
            gen_sex_coef.append(BWAS_correlation(gen_data[:,i],gen_sex_labels))
            
        gen_age_coef= np.array(gen_age_coef)
        gen_sex_coef= np.array(gen_sex_coef)
        
        np.save(os.path.join(save_dir,f'gen_{modality}_age.npy'),gen_age_coef)
        np.save(os.path.join(save_dir,f'gen_{modality}_sex.npy'),gen_sex_coef)


def compute_similarity(save_dir):

    modalities = [  'dti_FA_2mm', 'dti_L1_2mm', 'dti_L2_2mm', 'dti_L3_2mm', 'dti_MD_2mm',
                    'dti_MO_2mm','NODDI_ICVF_2mm','NODDI_ISOVF_2mm','NODDI_OD_2mm','QSM_2mm','SWI_2mm',
                    'T1_brain_nonlinear_2mm','T1_warp_Jac_2mm','T2_FLAIR_brain_to_MNI','T2star_2mm', 'VBM_2mm',
                    'zstat1s','zstat2s','zstat5s','DU15_1','DU15_2','DU15_3','DU15_4','DU15_5',
                    'DU15_6','DU15_7','DU15_8','DU15_9','DU15_10','DU15_11','DU15_12',
                    'DU15_13','DU15_14','DU15_15']

    results = []

    for modality in modalities:
        
        # age
        train_age = np.load(os.path.join('Path_to_UKB_Reference_Cohort_Pattern', f'train_{modality}_age.npy'))
        gen_age = np.load(os.path.join(save_dir, f'gen_{modality}_age.npy'))
        age_similarity = cosine_similarity(train_age, gen_age)
        
        # sex 
        train_sex = np.load(os.path.join('Path_to_UKB_Reference_Cohort_Pattern', f'train_{modality}_sex.npy'))
        gen_sex = np.load(os.path.join(save_dir, f'gen_{modality}_sex.npy'))
        sex_similarity = cosine_similarity(train_sex, gen_sex)
        
        print(modality, age_similarity, sex_similarity)
        
        results.append({ 'modality': modality, 'age_similarity': age_similarity,'sex_similarity': sex_similarity})

        df = pd.DataFrame(results)
        
        df.to_csv(os.path.join(save_dir,'similarities.csv'), index=False)
        print("similarities.csv is saved...")


def cosine_similarity(vec1, vec2):
    vec1 = np.nan_to_num(vec1, nan=0.0)
    vec2 = np.nan_to_num(vec2, nan=0.0)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return cosine_sim


def save_batch_data(eid_batch, data, save_dir, image_data_info, modality_num, sample_num):
    for i, eid in enumerate(eid_batch):
        addr = os.path.join(save_dir, f"generated_{eid}")
        os.makedirs(addr, exist_ok=True)
        # Correct slicing with colon
        eid_data = data[i * modality_num:(i + 1) * modality_num]
        for index, x in enumerate(eid_data):
            modality = image_data_info[str(index)]['modality']
            mean = image_data_info[str(index)]['mean']
            std = image_data_info[str(index)]['std']
            x = (x * std + mean).numpy()
            # Provide the array to save
            np.save(os.path.join(addr, f"{modality}_sample_{sample_num}.npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-UKB-L/256")
    parser.add_argument("--cfg-scale", type=float, default=1.2)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=228453)
    parser.add_argument("--ckpt", type=str, default='Path_to_GenBrain_ckpt/0000200.pt')
    parser.add_argument("--ckpt_epoch", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default='Path_to_Evaluation/avg_evaluations_model')
    parser.add_argument("--data_info", type=str, default="../labels/data_info.json")
    parser.add_argument("--sample_file", type=str, default="Path_to_Evaluation_Files")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of samples per batch")
    parser.add_argument("--modality_num", type=int, default=34, help="Number of modalities per sample")
    parser.add_argument("--average_num", type=int, default=5, help="Average number per sample")
    parser.add_argument("--use_ddim", default=True)
    args = parser.parse_args()
    main(args)
