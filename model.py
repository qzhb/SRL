import torch 
from os.path import join

from networks import base_srl

def get_model(args):

    if args.modality != 'fusion' :
        model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, args.feat_in, args.hidden, args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout, args.clc_dropout)
        print(args.model_name)

    elif args.modality == 'fusion':
        print(args.model_name)
        rgb_model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, 1024, 1024, args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout, args.clc_dropout)
        flow_model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, 1024, 1024, args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout, args.clc_dropout)
        obj_model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, 352, 352, args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout, args.clc_dropout)
        
        exp_name = args.exp_name.replace('_fusion', '_rgb')
        rgb_chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, args.resume_timestamp, exp_name + '_best.pth.tar'))['state_dict']
        
        exp_name = args.exp_name.replace('_fusion', '_flow')
        flow_chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, args.resume_timestamp, exp_name + '_best.pth.tar'))['state_dict']
       
        exp_name = args.exp_name.replace('_fusion', '_obj')
        obj_chk = torch.load(join(args.path_to_results, args.dataset, args.model_name, args.resume_timestamp, exp_name + '_best.pth.tar'))['state_dict']
        
        rgb_model.load_state_dict(rgb_chk)
        flow_model.load_state_dict(flow_chk)
        obj_model.load_state_dict(obj_chk)

        model = [rgb_model, flow_model, obj_model]

    return model
