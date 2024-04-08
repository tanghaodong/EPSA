import numpy as np
import torch
import torch.nn as nn

from align import model_util
from align import alignment
from align.model_htrans_layer import Multi_Htrans_Layer


class Align_Htrans(nn.Module):
    def __init__(self, kgs_data, config):
        super(Align_Htrans, self).__init__()
        # #self.myprint = config.myprint
        # #self.is_cuda = config.is_cuda
        # # if config.is_cuda:
        # #     self.device = config.device
        # self.metric = 'L1'
        # # self.l_beta = config.l_beta
        # self.gamma_rel = config.gamma_rel
        # self.neg_k = config.neg_k
        # self.top_k = config.top_k

        # Super Parameter
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)

        self.kg_E = kgs_data.kg_E
        self.kg_R = kgs_data.kg_R
        self.e_dim = 300
        # entityname embed ##############
        self.kg_name_embed = kgs_data.kg_entity_embed
        self.kg_name_model = nn.Linear(300, 300)

        # Multi_Htrans_Layer
        self.model_rels = Multi_Htrans_Layer(kgs_data.ent_neigh_dict, kgs_data.kg_E, kgs_data.kg_R, config)
        # if 'path' in config.model_type:
        self.isPath = True
        print('new kg_path:' + str(kgs_data.kg_path))
        self.model_paths = Multi_Htrans_Layer(kgs_data.path_neigh_dict, kgs_data.kg_E, kgs_data.kg_path, config)
        # else:
        #     self.isPath = False

        # if self.is_cuda:
        #     self.kg_name_embed = self.kg_name_embed.cuda(self.device)
        #     self.model_rels = self.model_rels.cuda(self.device)
        #     if self.isPath:
        #         self.model_paths = self.model_paths.cuda(self.device)

        ## Parameters ##############
        params = list(self.parameters())
        self.model_params = [{'params': params}]
        params_list = [param for param in self.state_dict()]
        print('model Parameters:{}\n{}'.format(str(len(params_list)), params_list.__str__()))

        ########################


    def resetPath(self, path_neigh_dict, kg_path):
        add_rid = kg_path
        for eid in range(self.kg_E):
            if len(path_neigh_dict[eid]) <1:
                path_neigh_dict[eid].append((eid, add_rid))

        return path_neigh_dict, kg_path+1

    # 2 rel_gat
    def forward(self):
        #1 model_name
        end_embed_in = self.kg_name_model(self.kg_name_embed)

        # 2 rel model
        rel_embed, skip_w = self.model_rels(self.kg_name_embed, end_embed_in)
        #3 path model
        if self.isPath:
            path_embed, p_skip_w = self.model_paths(self.kg_name_embed, end_embed_in)
            skip_w_all = skip_w + p_skip_w
        else:
            path_embed = None
            skip_w_all = skip_w

        self.rel_embed, self.path_embed = rel_embed, path_embed
        skip_w_all = [round(w, 4) for w in skip_w_all]

        return path_embed


