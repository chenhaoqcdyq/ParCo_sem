import os
import time

from tqdm import tqdm
import clip
import numpy
import numpy as np
import torch
from scipy import linalg

# import visualize.plot_3d_global as plot_3d
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visualize.plot_3d_global as plot_3d
# from dataset.dataset_VQ_bodypart import parts2whole
from utils.motion_process import recover_from_ric


def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)


@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, best_mpjpe=1e9) : 
    """
    Evaluate the VQVAE, used in train and test.
    Compute the FID, DIV, and R-Precision.
    """
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:

        # Get motion and parts. We use parts to represent parts' motion.
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, parts = batch

        motion = motion.cuda()

        if torch.isnan(motion).sum() > 0 or torch.isinf(motion).sum() > 0:
            print('Detected NaN or Inf in raw motion data')
            print('NaN elem numbers:', torch.isnan(motion).sum())
            print('Inf elem numbers:', torch.isinf(motion).sum())
            print('motion:', motion)

        # (text, motion) ==> (text_emb, motion_emb)
        #   motion is normalized.
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)

        if torch.isnan(em).sum() > 0:
            print('Detected NaN in em (embedding of motion), replace NaN with 0.0')
            print('NaN elem numbers:', torch.isnan(em).sum())
            print('em:', em)
            em = torch.nan_to_num(em)  # use default param to replace nan with 0.0. Require pytorch >= 1.8.0


        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):

            # [gt_motion] (augmented representation) ==[de-norm, convert]==> [gt_xyz] (xyz representation)
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            # Preprocess parts: get single sample from the batch
            single_parts = []
            for p in parts:
                single_parts.append(p[i:i+1, :m_length[i]].cuda())

            # (parts, GT) ==> (reconstruct_parts)
            #   parts is normalized.
            try:
                result = net(single_parts, caption[i:i+1])
            except:
                result = net(single_parts)
            pred_parts = result[0]

            # pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            # pred_parts = outputs['recon_parts']
            # pred_parts ==> whole_motion
            #   todo: support different shared_joint_rec_mode in the parts2whole function
            pred_pose = val_loader.dataset.parts2whole(pred_parts, mode=val_loader.dataset.dataset_name)

            # de-normalize reconstructed motion
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            # Convert to xyz representation
            #   todo: maybe we should support the recover_from_rot, not only ric.
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose
            
            mpjpe += torch.sum(calculate_mpjpe(pose_xyz, pred_xyz))
            num_poses += pose_xyz.shape[0]
            
            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        # pred_pose_eval is normalized
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    mpjpe = mpjpe / num_poses
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, mpjpe. {mpjpe:.4f}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))
            
    if mpjpe < best_mpjpe:
        msg = f"--> --> \t mpjpe Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
        logger.info(msg)
        best_mpjpe = mpjpe
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))
    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe



@torch.no_grad()
def evaluation_transformer_batch(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False, semantic_flag=False) :
    """
    This is used for evaluate GPT at training stage.
    It excludes the multi-modality evaluation by simply set a circle only at 1 time.
    """
    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in tqdm(val_loader):

            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, parts = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()  # (B, 512)
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()


            # [Text-to-motion Generation] get generated parts' token sequence
            # get parts_index_motion given the feat_clip_text
            batch_parts_index_motion = trans.sample_batch(feat_clip_text, False)  # List: [(B, seq_len), ..., (B, seq_len)]
            if semantic_flag:
                batch_parts_index_motion = [batch_parts_index_motion[i][:, trans.semantic_len:] for i in range(len(batch_parts_index_motion))]

            max_motion_seq_len = batch_parts_index_motion[0].shape[1]
            if isinstance(batch_parts_index_motion, torch.Tensor):
                batch_parts_index_motion = [batch_parts_index_motion[:,i,:] for i in range(batch_parts_index_motion.shape[1])]
            for k in range(bs):

                min_motion_seq_len = max_motion_seq_len
                parts_index_motion = []
                for part_index, name in zip(batch_parts_index_motion, ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):

                    # get one sample
                    part_index = part_index[k:k+1]  # (1, seq_len)

                    # find the earliest end token position
                    idx = torch.nonzero(part_index == trans.num_vq)

                    # # Debug
                    # print('part_index:', part_index)
                    # print('nonzero_idx', idx)

                    if idx.numel() == 0:
                        motion_seq_len = max_motion_seq_len
                    else:
                        min_end_idx = idx[:,1].min()
                        motion_seq_len = min_end_idx

                    if motion_seq_len < min_motion_seq_len:
                        min_motion_seq_len = motion_seq_len

                    parts_index_motion.append(part_index)

                # Truncate
                for j in range(len(parts_index_motion)):
                    if min_motion_seq_len == 0:
                        # assign a nonsense motion index to handle length is 0 issue.
                        parts_index_motion[j] = torch.ones(1,4).cuda().long()  # (B, seq_len) B==1, seq_len==1
                    elif min_motion_seq_len < 4:
                        parts_index_motion[j] = torch.cat([parts_index_motion[j], torch.ones(1, 4 - min_motion_seq_len).cuda().long()], dim=1)  # (B, seq_len) B==1, seq_len==1
                    else:
                        parts_index_motion[j] = parts_index_motion[j][:,:min_motion_seq_len]



                '''
                index_motion: (B, nframes). Here: B == 1, nframes == predicted_length
                '''

                # [Token-to-RawMotion with VQ-VAE decoder] get each parts' raw motion
                parts_pred_pose = net.forward_decoder(parts_index_motion)  # (B, pred_nframes, parts_sk_dim)
                #   todo: support different shared_joint_rec_mode in the parts2whole function
                pred_pose = val_loader.dataset.parts2whole(parts_pred_pose, mode=val_loader.dataset.dataset_name)  # (B, pred_nframes, raw_motion_dim)

                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)  # save the min len

                # It's actually should use pred_len[k] to replace cur_len and seq for understanding convenience
                #   Below code seems equal to use pred_len[k].
                #   But should not change it to keep the same test code with T2M-GPT.
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    if i == 0 and k < 4:
                        pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                        pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])


            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)


    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)


        if nb_iter % 10000 == 0 :
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)

        if nb_iter % 10000 == 0 :
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)


    if fid < best_fid :
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching :
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) :
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 :
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 :
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3 :
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger



@torch.no_grad()
def evaluation_transformer_test_batch(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, mmod_gen_times=30, skip_mmod=False):

    trans.eval()

    if skip_mmod:
        mmod_gen_times = 1

    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for batch in tqdm(val_loader):

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, sample_name, parts = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(mmod_gen_times):  # mmod_gen_times default: 30
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()


            # [Text-to-motion Generation] get generated parts' token sequence
            # get parts_index_motion given the feat_clip_text
            batch_parts_index_motion = trans.sample_batch(feat_clip_text, True)  # List: [(B, seq_len), ..., (B, seq_len)]

            max_motion_seq_len = batch_parts_index_motion[0].shape[1]


            for k in range(bs):

                min_motion_seq_len = max_motion_seq_len
                parts_index_motion = []
                for part_index, name in zip(batch_parts_index_motion, ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):

                    # get one sample
                    part_index = part_index[k:k+1]  # (1, seq_len)

                    # find the earliest end token position
                    idx = torch.nonzero(part_index == trans.parts_code_nb[name])

                    # # Debug
                    # print('part_index:', part_index)
                    # print('nonzero_idx', idx)

                    if idx.numel() == 0:
                        motion_seq_len = max_motion_seq_len
                    else:
                        min_end_idx = idx[:,1].min()
                        motion_seq_len = min_end_idx

                    if motion_seq_len < min_motion_seq_len:
                        min_motion_seq_len = motion_seq_len

                    parts_index_motion.append(part_index)

                # Truncate
                for j in range(len(parts_index_motion)):
                    if min_motion_seq_len == 0:
                        # assign a nonsense motion index to handle length is 0 issue.
                        parts_index_motion[j] = torch.ones(1,1).cuda().long()  # (B, seq_len) B==1, seq_len==1
                    else:
                        parts_index_motion[j] = parts_index_motion[j][:,:min_motion_seq_len]



                # [Token-to-RawMotion with VQ-VAE decoder] get each parts' raw motion
                parts_pred_pose = net.forward_decoder(parts_index_motion)  # (B, pred_nframes, parts_sk_dim)
                #   todo: support different shared_joint_rec_mode in the parts2whole function
                pred_pose = val_loader.dataset.parts2whole(parts_pred_pose, mode=val_loader.dataset.dataset_name)  # (B, pred_nframes, raw_motion_dim)

                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, sample_name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(sample_name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, sample_name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    if not skip_mmod:
        print('Calculate multimodality...')
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)


    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)

            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist


@torch.no_grad()        
def evaluation_magvit2(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, best_mpjpe=1e9) : 
    """
    Evaluate the VQVAE, used in train and test.
    Compute the FID, DIV, and R-Precision.
    """
    from demo import Ours_MAGVIT2
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []

    magvit2 = Ours_MAGVIT2()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in tqdm(val_loader):

        # Get motion and parts. We use parts to represent parts' motion.
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, parts = batch

        motion = motion.cuda()

        if torch.isnan(motion).sum() > 0 or torch.isinf(motion).sum() > 0:
            print('Detected NaN or Inf in raw motion data')
            print('NaN elem numbers:', torch.isnan(motion).sum())
            print('Inf elem numbers:', torch.isinf(motion).sum())
            print('motion:', motion)

        # (text, motion) ==> (text_emb, motion_emb)
        #   motion is normalized.
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)

        if torch.isnan(em).sum() > 0:
            print('Detected NaN in em (embedding of motion), replace NaN with 0.0')
            print('NaN elem numbers:', torch.isnan(em).sum())
            print('em:', em)
            em = torch.nan_to_num(em)  # use default param to replace nan with 0.0. Require pytorch >= 1.8.0


        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()
        pred_len = torch.zeros((bs)).cuda()
        result_list = magvit2.sample(caption)
        for i in range(bs):

            # [gt_motion] (augmented representation) ==[de-norm, convert]==> [gt_xyz] (xyz representation)
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            # Preprocess parts: get single sample from the batch
            single_parts = []
            for p in parts:
                single_parts.append(p[i:i+1, :m_length[i]].cuda())

            # (parts, GT) ==> (reconstruct_parts)
            #   parts is normalized.
            # try:
            #     result = net(single_parts, caption[i:i+1])
            # except:
            #     result = net(single_parts)
            
            result = result_list[i]
            # pred_parts_code = result
            idx = torch.nonzero(result == 512)
            if idx.numel() > 0:
                m_length_pred = idx[:,1].min()
                
            else:
                # pred_parts = torch.zeros((1, 0, 50, 50)).cuda()
                m_length_pred = 196
            pred_parts_code = result[:,:m_length_pred]
            pred_len[i] = m_length_pred
            pred_parts = net.decode(pred_parts_code)
            # pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            # pred_parts = outputs['recon_parts']
            # pred_parts ==> whole_motion
            #   todo: support different shared_joint_rec_mode in the parts2whole function
            pred_pose = val_loader.dataset.parts2whole(pred_parts, mode=val_loader.dataset.dataset_name)

            # de-normalize reconstructed motion
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            # Convert to xyz representation
            #   todo: maybe we should support the recover_from_rot, not only ric.
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:pred_len[i].long(),:] = pred_pose
            min_seq_len = min(m_length[i], pred_len[i]).long()
            mpjpe += torch.sum(calculate_mpjpe(pose_xyz[:,:min_seq_len], pred_xyz[:,:min_seq_len]))
            num_poses += pose_xyz.shape[0]
            
            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        # pred_pose_eval is normalized
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    mpjpe = mpjpe / num_poses
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, mpjpe. {mpjpe:.4f}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))
            
    if mpjpe < best_mpjpe:
        msg = f"--> --> \t mpjpe Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
        logger.info(msg)
        best_mpjpe = mpjpe
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))
    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe

def calculate_motion_metrics(code_indices, eval_wrapper, top_k=3, save_path=None):
    """
    计算预测动作的评估指标
    
    Args:
        code_indices: 编码索引字典，每个元素是一个动作序列
        eval_wrapper: 评估包装器，用于计算动作嵌入
        top_k: 计算top-k准确率时的k值
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    # 初始化指标
    metrics = {
        'diversity': 0.0,
        'top_k_accuracy': [0.0] * top_k,
        'matching_score': 0.0,
        'fid': 0.0
    }
    
    # 将预测动作转换为numpy数组
    pred_motions = [code_indices[name]['decoded_motion'] for name in code_indices]
    word_embeddings = [code_indices[name]['word_embeddings'] for name in code_indices]
    pos_one_hots = [code_indices[name]['pos_one_hots'] for name in code_indices]
    caption = [code_indices[name]['caption'] for name in code_indices]
    sent_len = [code_indices[name]['sent_len'] for name in code_indices]
    m_length = [code_indices[name]['m_length'] for name in code_indices]
    gt_motions = [code_indices[name]['gt'] for name in code_indices]
    mean = torch.from_numpy(np.load('checkpoints/t2m/Decomp_SP001_SM001_H512/meta/mean.npy')).cuda()
    std = torch.from_numpy(np.load('checkpoints/t2m/Decomp_SP001_SM001_H512/meta/std.npy')).cuda()
    # 计算动作嵌入
    pred_embeddings = []
    et_pred_list = []
    em_pred_list = []
    
    for i in range(len(pred_motions)):
        # 确保m_length是一个标量值，而不是张量
        # m_length = pred_motions[i].shape[1]
            
        # 计算动作嵌入
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings[i].unsqueeze(0), 
            pos_one_hots[i].unsqueeze(0), 
            torch.tensor([sent_len[i]], dtype=torch.int64).to(pred_motions[i].device), 
            pred_motions[i], 
            torch.tensor([m_length[i]], dtype=torch.int64).to(pred_motions[i].device))
        pred_embeddings.append(em_pred)
        et_pred_list.append(et_pred)
        em_pred_list.append(em_pred)
    et_pred = torch.cat(et_pred_list, dim=0).cpu()
    em_pred = torch.cat(em_pred_list, dim=0).cpu()
    pred_embeddings = torch.cat(pred_embeddings, dim=0).cpu().numpy()
    # 计算多样性
    metrics['diversity'] = calculate_diversity(pred_embeddings, 300 if len(pred_embeddings) > 300 else 100)
    # 计算top-k准确率和匹配分数
    # 这里我们使用预测动作之间的相似度来计算
    # dist_mat = euclidean_distance_matrix(pred_embeddings, pred_embeddings)
    batch = 32
    R_precision_list = []
    matching_score_pred_list = []
    for i in range(len(pred_motions)//batch):
        R_precision_tmp, matching_score_pred_tmp = calculate_R_precision(et_pred[i*batch:(i+1)*batch].cpu().numpy(), em_pred[i*batch:(i+1)*batch].cpu().numpy(), top_k=3, sum_all=True)
        # print(R_precision, matching_score_pred)
        R_precision_list.append(R_precision_tmp)
        matching_score_pred_list.append(matching_score_pred_tmp)
    R_precision = sum(R_precision_list) / len(R_precision_list)
    matching_score_pred = sum(matching_score_pred_list) / len(matching_score_pred_list)
    # R_precision += temp_R
    # matching_score_pred += temp_match
    # matching_score = dist_mat.trace()
    # argmax = np.argsort(dist_mat, axis=1)
    # top_k_mat = calculate_top_k(argmax, top_k)
    metrics['top_k_accuracy'] = R_precision / batch
    metrics['matching_score'] = matching_score_pred / batch
    et_gt_list = []
    em_gt_list = []
    for i in range(len(gt_motions)):
        # 计算动作嵌入
        et_gt, em_gt = eval_wrapper.get_co_embeddings(
            word_embeddings[i].unsqueeze(0), 
            pos_one_hots[i].unsqueeze(0), 
            torch.tensor([sent_len[i]], dtype=torch.int64).to(gt_motions[i].device), 
            gt_motions[i].unsqueeze(0), 
            torch.tensor([m_length[i]], dtype=torch.int64).to(gt_motions[i].device))
        em_gt_list.append(em_gt)
    em_gt = torch.cat(em_gt_list, dim=0).cpu().numpy()
    
    gt_mu, gt_cov  = calculate_activation_statistics(em_gt)
    mu, cov= calculate_activation_statistics(em_pred.numpy())
    metrics['fid'] = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    # # 计算MPJPE (Mean Per Joint Position Error)
    # # 这里我们计算相邻帧之间的MPJPE
    # total_mpjpe = 0
    # count = 0
    # for i in range(len(pred_motions)):
    #     if pred_motions[i].shape[0] > 1:  # 确保有足够的帧数
    #         # 计算相邻帧之间的MPJPE
    #         mpjpe = calculate_mpjpe(pred_motions[i][:-1], pred_motions[i][1:])
    #         total_mpjpe += mpjpe.sum().item()
    #         count += mpjpe.shape[0]
    
    # if count > 0:
    #     metrics['mpjpe'] = total_mpjpe / count
    print(metrics)
    if save_path is not None:
        infer_out_dir = os.path.join(save_path, 'infer_out')
        os.makedirs(infer_out_dir, exist_ok=True)
        for i in range(len(pred_motions)):  
            pred_xyz = recover_from_ric((pred_motions[i]*std + mean).float(), 22)
            xyz = pred_xyz.reshape(1, -1, 22, 3)
            npy_save_dir = os.path.join(infer_out_dir, f'{i}_motion.npy')
            gif_save_dir = os.path.join(infer_out_dir, f'{i}_skeleton_viz.gif')
            np.save(npy_save_dir, xyz.detach().cpu().numpy())
            pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), [caption[i]], [gif_save_dir])
            print(caption[i], "save in ", gif_save_dir)
        # metrics['mpjpe'] += calculate_mpjpe(xyz, pred_motions[i]) 
    
    return metrics

class CodeIndexEvaluator:
    """
    用于读取编码索引文件，解码动作并评估的类
    """
    def __init__(self, net, eval_wrapper, val_loader, dataset_name='t2m', device='cuda', path=False):
        """
        初始化评估器
        
        Args:
            net: 预训练的VQ-VAE网络
            eval_wrapper: 评估包装器，用于计算动作嵌入
            dataset_name: 数据集名称，默认为't2m'
            device: 计算设备，默认为'cuda'
        """
        self.net = net
        self.eval_wrapper = eval_wrapper
        self.dataset_name = dataset_name
        self.device = device
        self.net.to(device)
        self.net.eval()
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        self.val_loader = val_loader
        if path:
            self.have_path = True
        else:
            self.have_path = False
        
    def load_code_indices(self, code_dir):
        """
        加载指定目录中的所有编码索引文件
        
        Args:
            code_dir: 编码索引文件所在的目录
            
        Returns:
            dict: 文件名到编码索引的映射
        """
        code_indices = {}
        for filename in tqdm(os.listdir(code_dir)):
            if filename.endswith('.npy') or filename.endswith('.npz'):
                code_tmp = {}
                file_path = os.path.join(code_dir, filename)
                name = os.path.splitext(filename)[0]
                if filename.endswith('.npy'):
                    code_index = np.load(file_path)
                    if len(code_index.shape) == 3:
                        code_index = code_index[1,...]
                elif filename.endswith('.npz'):
                    # code_index = np.load(file_path)['arr_0']
                    data = np.load(file_path)
                    code_index = [data[part] for part in self.parts_name]
                    code_index = np.concatenate(code_index, axis=0)
                else:
                    continue
                # code_indices[name] = code_index
                code_tmp['code_index'] = code_index
                # code_tmp['name'] = data['motion']
                # code_tmp['text'] = data['text']
                code_indices[name] = code_tmp
        
        for batch in self.val_loader:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, names, parts = batch
            for i in range(len(names)):
                name = names[i]
                if name in code_indices.keys():
                    code_indices[name]['word_embeddings'] = word_embeddings[i]
                    code_indices[name]['pos_one_hots'] = pos_one_hots[i]
                    code_indices[name]['caption'] = caption[i]
                    code_indices[name]['sent_len'] = sent_len[i]
                    code_indices[name]['motion'] = motion[i]
                    code_indices[name]['m_length'] = m_length[i]
                    code_indices[name]['token'] = token[i]
                    code_indices[name]['gt'] = motion[i]
                    
        new_code_indices = {}
        for name in code_indices:
            if 'word_embeddings' in code_indices[name].keys():
                new_code_indices[name] = code_indices[name]
        del code_indices
        return new_code_indices
    
    def decode_motions(self, code_indices):
        """
        使用网络解码编码索引为动作
        
        Args:
            code_indices: 编码索引字典
            
        Returns:
            dict: 文件名到解码后动作的映射
        """
        # decoded_motions = []
        # text_list = []
        for name in code_indices:
            with torch.no_grad():
                # 解码动作
                decoded_parts = self.net.decode(code_indices[name]['code_index'])
                # 将部分动作转换为完整动作
                decoded_motion = self._parts2whole(decoded_parts)
                code_indices[name]['decoded_motion'] = decoded_motion
                # text_list.append(code_indices[name]['text'])
        return code_indices
    
    def _parts2whole(self, parts):
        """
        将部分动作转换为完整动作
        
        Args:
            parts: 部分动作列表
            
        Returns:
            torch.Tensor: 完整动作
        """
        # 使用数据集中的parts2whole函数
        from dataset.dataset_VQ_bodypart_text_mask_196 import parts2whole
        return parts2whole(parts, mode=self.dataset_name)
    
    def evaluate_motions(self, code_indices):
        """
        评估解码后的动作
        
        Args:
            decoded_motions: 解码后的动作字典
            
        Returns:
            dict: 评估指标
        """
        # 将动作转换为列表
        # motion_list = [code_indices[name]['decoded_motion'] for name in code_indices]
        
        # 计算评估指标
        if self.path is not None:
            metrics = calculate_motion_metrics(code_indices, self.eval_wrapper, save_path=self.path)
        else:
            metrics = calculate_motion_metrics(code_indices, self.eval_wrapper)
        
        return metrics
    
    def evaluate_code_directory(self, code_dir):
        """
        评估指定目录中的所有编码索引文件
        
        Args:
            code_dir: 编码索引文件所在的目录
            
        Returns:
            dict: 评估指标
        """
        if self.have_path:
            self.path = code_dir
        else:
            self.path = None
        # 加载编码索引
        code_indices = self.load_code_indices(code_dir)
        
        # 解码动作
        code_indices = self.decode_motions(code_indices)
        
        # 评估动作
        metrics = self.evaluate_motions(code_indices)
        
        return metrics

if __name__ == "__main__":
    import os
    import json
    import argparse

    import torch
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    import models.rvqvae_bodypart as vqvae
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from dataset import dataset_TM_eval_bodypart

    import options.option_vq_bodypart as option_vq
    from options.get_eval_option import get_opt

    import utils.utils_model as utils_model
    import utils.eval_bodypart as eval_bodypart
    from utils.word_vectorizer import WordVectorizer
    from utils.misc import EasyDict
    test_args = option_vq.get_vavae_test_args_parser()
    test_args.select_vqvae_ckpt = 'fid'
    test_args.vqvae_train_dir = 'output/00417-t2m-v11/VQVAE-v11-t2m-default'
    # test_args.vqvae_train_dir = 'output/00889-t2m-v24_dual3_downlayer1/VQVAE-v24_dual3_downlayer1-t2m-default'
    select_ckpt = test_args.select_vqvae_ckpt
    assert select_ckpt in [
        'last',  # last  saved ckpt
        'fid',  # best FID ckpt
        'div',  # best diversity ckpt
        'top1',  # best top-1 R-precision
        'matching',  # MM-Dist: Multimodal Distance
    ]


    vqvae_train_dir = test_args.vqvae_train_dir

    # Checkpoint path
    if select_ckpt == 'last':
        test_args.ckpt_path = os.path.join(vqvae_train_dir, 'net_' + select_ckpt + '.pth')
    else:
        test_args.ckpt_path = os.path.join(vqvae_train_dir, 'net_best_' + select_ckpt + '.pth')

    # Prepare testing directory
    test_args.test_dir = os.path.join(vqvae_train_dir, 'test_vqvae-' + select_ckpt)
    test_args.test_npy_save_dir = os.path.join(test_args.test_dir, 'saved_npy')
    os.makedirs(test_args.test_dir, exist_ok=True)
    os.makedirs(test_args.test_npy_save_dir, exist_ok=True)

    # Load the config of vqvae training
    print('\nLoading training argument...\n')
    test_args.training_options_path = os.path.join(vqvae_train_dir, 'train_config.json')
    with open(test_args.training_options_path, 'r') as f:
        train_args_dict = json.load(f)  # dict
    train_args = EasyDict(train_args_dict)  # convert dict to easydict for convenience
    test_args.train_args = train_args  # save train_args into test_args for logging convenience
    args = train_args

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(test_args.test_dir )
    writer = SummaryWriter(test_args.test_dir )
    logger.info(json.dumps(vars(test_args), indent=4, sort_keys=True))


    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if train_args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD01/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


    ##### ---- Dataloader ---- #####

    val_loader = dataset_TM_eval_bodypart.DATALoader(
        train_args.dataname, False, 32, w_vectorizer, unit_length=2**train_args.down_t)


    ##### ---- Network ---- #####
    print('\n\n===> Constructing network...')
    net = getattr(vqvae, f'HumanVQVAETransformerV{args.vision}')(args,  # use args to define different parameters in different quantizers
            parts_code_nb=args.vqvae_arch_cfg['parts_code_nb'],
            parts_code_dim=args.vqvae_arch_cfg['parts_code_dim'],
            parts_output_dim=args.vqvae_arch_cfg['parts_output_dim'],
            parts_hidden_dim=args.vqvae_arch_cfg['parts_hidden_dim'],
            down_t=args.down_t,
            stride_t=args.stride_t,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation=args.vq_act,
            norm=args.vq_norm
        )    


    #### Loading weights #####
    print('\n\n===> Loading weights...')
    if test_args.ckpt_path:
        logger.info('loading checkpoint from {}'.format(test_args.ckpt_path))
        ckpt = torch.load(test_args.ckpt_path, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    else:
        raise Exception('You need to specify the ckpt path!')

    net.cuda()
    net.eval()

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    repeat_time = 20
    # 初始化评估器
    evaluator = CodeIndexEvaluator(net, eval_wrapper, val_loader, path=True)
    # code_dir = "/workspace/motion_diffusion/ParCo/dataset/HumanML3D/gen/0416_Val_lg0"
    # code_dir = "/workspace/motion_diffusion/ParCo/dataset/HumanML3D/vqvae_code11_lg0_val"
    # code_dir = "/workspace/motion_diffusion/ParCo/dataset/HumanML3D/vqvae_code24_lg7_val"
    code_dir = "/workspace/motion_diffusion/ParCo/dataset/HumanML3D/Sampler_v2"
    
    # 评估指定目录中的所有编码索引文件
    metrics = evaluator.evaluate_code_directory(code_dir)
    print(metrics)