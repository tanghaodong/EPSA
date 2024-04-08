import math
import sys
import time
import torch
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter

from autil import alignment2, fileUtil
from align import align_model2
from align import pre_relScore
from align.model_htrans import Align_Htrans
#from align import fileUtil
class load_data(object):
    def __init__(self, config, model_type='E'):
        # Load data, entity sets, and relationship sets
        with open(config.datasetPath + 'pre/kgs_num', 'r') as ff:
            self.kg_E = int(ff.readline()[:-1].split(':')[1])
            self.kg_R = int(ff.readline()[:-1].split(':')[1])
            self.kg_M = int(ff.readline()[:-1].split(':')[1])
            self.kg_V = int(ff.readline()[:-1].split(':')[1])

        # relation triples
        self.rel_triples = fileUtil.load_triples_list(config.datasetPath + 'pre/rel_triples_id')
        self.r_head, self.r_tail, self.eer_adj_index, self.eer_adj_data = self.get_r_adj(self.rel_triples, self.kg_E, self.kg_R)
        # e_adj
        self.e_adj_index, self.e_adj_data = self.get_e_adj(self.rel_triples, self.kg_E)
        # entity embedding
        kg_entity_embed = fileUtil.loadpickle(config.datasetPath + 'pre/entity_embedding.out')
        self.kg_entity_embed = torch.FloatTensor(kg_entity_embed)
        # attribute triples
        if 'M' in model_type:
            attr_triples_id = fileUtil.load_triples_list(config.datasetPath + 'pre/attr_triples_id')
            KGs_attr_triple = np.array(attr_triples_id)
            KGs_attr_triple = KGs_attr_triple.astype(np.int)

            # attribute value embedding
            kg_value_embed = fileUtil.loadpickle(config.datasetPath + 'pre/value_embedding.out')
            if self.kg_V < self.kg_E:
                value_embed_mat2 = np.zeros((self.kg_E - self.kg_V, 300), dtype=np.float32)
                kg_value_embed = np.vstack((kg_value_embed, value_embed_mat2))
                self.kg_V = self.kg_E
            self.kg_value_embed = torch.FloatTensor(kg_value_embed)
            # neighbourhood
            self.m_head2e, self.m_tail2v, self.emv_adj_index, self.emv_adj_data \
                = self.get_m_adj(KGs_attr_triple, self.kg_E, self.kg_M, self.kg_V)
            # self.em_adj_dense: (M+E-M,E)
            # self.Hm: Count the reciprocal of the number of entities associated with each attribute


        # train、Valid、Test
        train_links_id = fileUtil.load_link_list(config.datasetPath + config.dataset_division + 'train_links_id')
        self.train_links = np.array(train_links_id)  # (3000, 2)
        self.train_links_tensor = torch.LongTensor(self.train_links)

        test_links_id = fileUtil.load_link_list(config.datasetPath + config.dataset_division + 'test_links_id')
        test_links = np.array(test_links_id)
        self.test_links_tensor = torch.LongTensor(test_links)

        # path
        self.pre_path(config)
        self.pre_relation(config)

        if '100K' not in config.datasetPath:
            valid_links_id = fileUtil.load_link_list(config.datasetPath + config.dataset_division + 'valid_links_id')
            self.valid_links = np.array(valid_links_id)
            self.valid_links_tensor = torch.LongTensor(self.valid_links)

        if config.cuda:
            # Relation triples
            self.r_head = self.r_head.cuda()
            self.r_tail = self.r_tail.cuda()
            self.eer_adj_index = self.eer_adj_index.cuda()
            self.eer_adj_data = self.eer_adj_data.cuda()
            # name embedding
            self.kg_entity_embed = self.kg_entity_embed.cuda()

            # GCN+highway
            self.e_adj_index = self.e_adj_index.cuda()
            self.e_adj_data = self.e_adj_data.cuda()

            # attribute triple
            if 'M' in model_type:
                self.m_head2e = self.m_head2e.cuda()
                self.m_tail2v = self.m_tail2v.cuda()
                self.emv_adj_index = self.emv_adj_index.cuda()
                self.emv_adj_data = self.emv_adj_data.cuda()
                # value embedding
                self.kg_value_embed = self.kg_value_embed.cuda()

            # train、Valid、Test
            self.train_links_tensor = self.train_links_tensor.cuda()
            self.valid_links_tensor = self.valid_links_tensor.cuda()
            self.test_links_tensor = self.test_links_tensor.cuda()


    def pre_relation(self, myconfig):
        print("\n=== pre_relation ==")
        ent_neigh_dict = dict()

        # self
        rel_self = self.kg_R
        self.kg_R += 1
        for eid in range(self.kg_E):
            ent_neigh_dict[eid] = [(eid, rel_self)]
        rel_triple_num = 0
        for (h, r, t) in self.rel_triples:  # 无方向
            ent_neigh_dict[h].append((t, r))
            rel_triple_num += 1
            if h != t:
                ent_neigh_dict[t].append((h, r))
                rel_triple_num += 1
        self.ent_neigh_dict = ent_neigh_dict
        print('ent_neigh_dict:' + str(len(self.ent_neigh_dict)))
        print('rel_triple_num:' + str(rel_triple_num))


    def pre_path(self,config):
        print("\n=== pre_path ==")
        # path_neigh_dict: Path and its associated head and tail entities
        path_neigh_dict = dict()
        with open(config.datasetPath + 'pre/path_neigh_dict', 'r', encoding='utf-8') as fr:
            for line in fr:
                entid, rtlist = eval(line[:-1])
                path_neigh_dict[entid] = rtlist

        # rpath_sort_dict: Paths and their frequency numbers
        pathid2rr = dict()
        with open(config.datasetPath + 'pre/rpath_sort_dict', 'r', encoding='utf-8') as fr:
            for line in fr:
                rpath, pathid = eval(line[:-1])
                pathid2rr[pathid] = rpath

        # 2、path_pair_dict
        path_len = len(pathid2rr)
        path_pair_dict, temp_path_list, temp_notpath_list = pre_relScore.get_align_path(self.kg_entity_embed, self.train_links, path_neigh_dict,
                                                                                self.kg_E, path_len)
        print("Number of path_pair_dict:" + str(len(path_pair_dict)))

        # 5、pathid
        kg1_id2rel = fileUtil.load_ids2dict(config.datasetPath + 'pre/kg1_rel_dict', read_kv='kv')  # name:id
        kg2_id2rel = fileUtil.load_ids2dict(config.datasetPath + 'pre/kg2_rel_dict', read_kv='kv')
        kg1_id2rel.update(kg2_id2rel)

        oldpath2newpath = dict()
        path_save_list = []
        for path_newid, (path1, path2) in enumerate(path_pair_dict.items()):
            oldpath2newpath[path1] = oldpath2newpath[path2] = path_newid
            path1_r1, path1_r2 = pathid2rr[path1]
            path2_r1, path2_r2 = pathid2rr[path2]
            path_save_list.append((path_newid, (path1, pathid2rr[path1]), (path2, pathid2rr[path2]),
                                   (path1, kg1_id2rel[path1_r1], kg1_id2rel[path1_r2]),
                                   (path2, kg1_id2rel[path2_r1], kg1_id2rel[path2_r2])))

        path_triple_num = 0
        path_triple_old_num = 0
        # self
        path_self_id = path_newid + 1
        for h, trlist in path_neigh_dict.items():
            path_triple_old_num += len(trlist)
            trlist_new = [(t, oldpath2newpath[path_oldid]) for (t, path_oldid) in trlist if path_oldid in oldpath2newpath.keys()]
            if len(trlist_new) <= 1:
                trlist_new.append((h, path_self_id))
            path_neigh_dict[h] = trlist_new
            path_triple_num += len(trlist_new)

        self.kg_path = path_self_id + 1
        self.path_neigh_dict = path_neigh_dict
        print("Number of path_newid:" + str(self.kg_path))
        print("Number of all path_triple:" + str(path_triple_old_num))
        print("Number of path_triple by aligned:" + str(path_triple_num))

    def get_r_adj(self, KG, e_num, r_num):
        r_head_array = np.zeros((r_num, e_num))  # (R,E)  array[r,h]= sum(hr)
        r_tail_array = np.zeros((r_num, e_num))
        r_mat_row = []
        r_mat_col = []
        r_mat_data = []
        for (h, r, t) in KG:
            r_head_array[r][h] += 1
            r_tail_array[r][t] += 1

            r_mat_row.append(h)
            r_mat_col.append(t)
            r_mat_data.append(r)

            r_mat_row.append(t)
            r_mat_col.append(h)
            r_mat_data.append(r)

        r_head = torch.FloatTensor(r_head_array)
        r_tail = torch.FloatTensor(r_tail_array)

        eer_adj_index = np.vstack((r_mat_row, r_mat_col))  # (2,D)
        eer_adj_index = torch.LongTensor(eer_adj_index)
        eer_adj_data = torch.LongTensor(r_mat_data)

        return r_head, r_tail, eer_adj_index, eer_adj_data


    def get_e_adj(self, KG, e_num):
        ''' GCN+highway Neighbor matrix '''
        du = [1] * e_num  # # du[e] is the number of occurrences of entity e in the triples
        for (h, r, t) in KG:
            if h != t:
                du[h] += 1
                du[t] += 1

        e_mat_row = []
        e_mat_col = []
        e_mat_data = []
        for (h, r, t) in KG:
            e_mat_row.append(h)
            e_mat_col.append(t)
            e_mat_data.append(1 / math.sqrt(du[h]) / math.sqrt(du[t]))

        e_mat_row = np.array(e_mat_row)
        e_mat_col = np.array(e_mat_col)
        e_mat_data = np.array(e_mat_data)

        e_adj_index = np.vstack((e_mat_row, e_mat_col))  # (2,D)
        e_adj_index = torch.LongTensor(e_adj_index)
        e_adj_data = torch.FloatTensor(e_mat_data)

        return e_adj_index, e_adj_data  # e_adj =>A+I


    def get_m_adj(self, KG_attr, e_num, m_num, v_num):
        m_head2e_array = np.zeros((m_num, e_num))  # (R,E)  array[r,h]= sum(hr)
        m_tail2v_array = np.zeros((m_num, v_num))
        ee_index = []
        mm_index = []
        vv_index = []
        for (h, a, v) in KG_attr:  # (e,a,v)
            m_head2e_array[a][h] += 1
            m_tail2v_array[a][v] += 1

            ee_index.append(h)
            mm_index.append(a)
            vv_index.append(v)

        m_head2e = torch.FloatTensor(m_head2e_array)
        m_tail2v = torch.FloatTensor(m_tail2v_array)

        mm_index = np.array(mm_index)
        ee_index = np.array(ee_index)
        vv_index = np.array(vv_index)

        emv_adj_index = np.vstack((ee_index, vv_index))
        emv_adj_index = torch.LongTensor(emv_adj_index)
        emv_adj_data = torch.LongTensor(mm_index)

        return m_head2e, m_tail2v, emv_adj_index, emv_adj_data

class modelClass2():
    def __init__(self, config, myprint_fun):
        super(modelClass2, self).__init__()

        self.myprint = myprint_fun  # Printing and logging
        self.config = config
        self.best_mode_pkl_title = config.output + time.strftime('%Y-%m-%d_%H_%M_%S-', time.localtime(time.time()))

        # Load data
        input_data = load_data(config)
        # train、Valid、Test
        self.train_links = input_data.train_links
        self.valid_links = input_data.valid_links
        # self.test_links = input_data.test_links
        self.train_links_tensor = input_data.train_links_tensor
        self.valid_links_tensor = input_data.valid_links_tensor
        self.test_links_tensor = input_data.test_links_tensor

        # Model and optimizer
        self.mymodel = align_model2.HET_align2(input_data, config)

        if config.cuda:
            self.mymodel.cuda()
        # [TensorboardX]Summary_Writer
        self.board_writer = SummaryWriter(log_dir=self.best_mode_pkl_title + '-E/', comment='HET_align')

        # optimizer
        self.parameters = filter(lambda p: p.requires_grad, self.mymodel.parameters())
        self.myprint('All parameter names in the model:' + str(len(self.mymodel.state_dict())))
        for i in self.mymodel.state_dict():
            self.myprint(i)
        if config.optim_type == 'Adagrad':
            self.optimizer = optim.Adam(self.parameters, lr=config.learning_rate,
                            weight_decay=config.weight_decay)  # weight_decay =5e-4
        else:
            self.optimizer = optim.SGD(self.parameters, lr=config.learning_rate,
                            weight_decay=config.weight_decay)
        # Weight initialization Weight initialization
        self.mymodel.init_weights()

    ## model train
    def model_train(self, epochs_beg=0):
        self.myprint("model training start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        t_begin = time.time()

        bad_counter = 0
        best_hits1 = 0  # best
        best_epochs = 0
        min_eval_loss = sys.maxsize
        min_bad_conter = 0
        for epochs_i in range(epochs_beg, self.config.train_epochs):  # epochs=1001
            epochs_i_t = time.time()
            #  Forward pass
            self.mymodel.train()
            self.optimizer.zero_grad()
            e_out_embed = self.mymodel()

            #   generate negative samples
            self.regen_neg(epochs_i, e_out_embed)

            # loss、acc
            loss_train = self.mymodel.get_loss(e_out_embed, self.train_neg_pairs)  # loss
            # Backward and optimize
            loss_train.backward()
            self.optimizer.step()

            if epochs_i % 5 != 0:
                self.myprint('Epoch-{:04d}: train_loss-{:.4f}, cost time-{:.4f}s'.format(
                    epochs_i, loss_train.data.item(), time.time() - epochs_i_t))
            else:
                # accuracy: hits, mr, mrr
                result_train = alignment2.my_accuracy(e_out_embed, self.train_links_tensor,
                                                      metric=self.config.metric, top_k=self.config.top_k)

                # [TensorboardX]
                self.board_writer.add_scalar('train_loss', loss_train.data.item(), epochs_i)
                self.board_writer.add_scalar('train_hits1', result_train[0][0], epochs_i)

                self.myprint('Epoch-{:04d}: train_loss-{:.4f}, cost time-{:.4f}s'.format(
                    epochs_i, loss_train.data.item(), time.time() - epochs_i_t))
                self.print_result('Train', result_train)

            # # ********************no early stop********************************************
            if epochs_i >= self.config.start_valid and epochs_i % self.config.eval_freq == 0:
                epochs_i_t = time.time()
                # valid
                self.mymodel.eval()  # self.train(False)
                e_out_embed = self.mymodel()

                loss_val = self.mymodel.get_loss(e_out_embed, self.valid_neg_pairs)
                result_val = alignment2.my_accuracy(e_out_embed, self.valid_links_tensor,
                                                    metric=self.config.metric, top_k=self.config.top_k)
                loss_val_value = loss_val.data.item()
                # [TensorboardX]
                self.board_writer.add_scalar('valid_loss', loss_val_value, epochs_i)
                self.board_writer.add_scalar('valid_hits1', result_val[0][0], epochs_i)

                self.myprint('Epoch-{:04d}: valid_loss-{:.4f}, cost time-{:.4f}s'.format(
                       epochs_i, loss_val_value, time.time() - epochs_i_t))
                self.print_result('Valid', result_val)

                # save best model in valid
                if result_val[0][0] >= best_hits1:
                    best_hits1 = result_val[0][0]
                    best_epochs = epochs_i
                    bad_counter = 0
                    self.myprint('Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, best_hits1))
                    self.save_model(epochs_i, 'best-epochs')
                else:
                    # no best, but save model every 50 epochs
                    if epochs_i % self.config.eval_save_freq == 0:
                        self.save_model(epochs_i, 'eval-epochs')
                    # bad model, stop train
                    bad_counter += 1
                    self.myprint('bad_counter++:' + str(bad_counter))
                    if bad_counter == self.config.patience:  # patience=20
                        self.myprint('Epoch-{:04d},bad_counter.'.format(epochs_i))
                        break

                # Verification set loss continuous decline also stop training!
                if loss_val_value < min_eval_loss:
                    min_eval_loss = loss_val_value
                    min_bad_conter = 0
                    self.myprint('Epoch-{:04d}, min eval loss, min_eval_loss:{:.4f}..'.format(epochs_i, loss_val_value))
                else:
                    min_bad_conter += 1
                    if min_bad_conter == self.config.patience_val:  # patience_val=10
                        self.myprint('Epoch-{:04d}, min_bad_conter.'.format(epochs_i))
                        break

        self.save_model(epochs_i, 'last-epochs')  # save last epochs
        self.myprint("Optimization Finished!")
        self.myprint('Best epoch-{:04d}:'.format(best_epochs))
        self.myprint('Last epoch-{:04d}:'.format(epochs_i))
        self.myprint("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

        return best_epochs, epochs_i


    # get negative samples
    def regen_neg(self, epochs_i, ent_embed):
        if epochs_i % self.config.sample_neg_freq == 0:  # sample negative pairs every 20 epochs
            with torch.no_grad():
                # Negative sample sampling-training pair (positive sample and negative sample)
                self.train_neg_pairs = alignment2.gen_neg(ent_embed, self.train_links, self.config.metric, self.config.neg_k)
                self.valid_neg_pairs = alignment2.gen_neg(ent_embed, self.valid_links, self.config.metric, self.config.neg_k)


    def compute_test(self, epochs_i, name_epochs):
        ''' run best model '''
        model_savefile = '{}-epochs{}-{}.pkl'.format(self.best_mode_pkl_title + name_epochs, epochs_i, self.config.model_param)
        self.myprint('\nLoading {} - {}th epoch'.format(name_epochs, epochs_i))
        self.re_test(model_savefile)


    def re_test(self, model_savefile):
        ''' restart run best model '''
        # Restore best model
        self.myprint('Loading file: ' + model_savefile)
        self.mymodel.load_state_dict(torch.load(model_savefile))  # load_state_dict()
        self.mymodel.eval()  # self.train(False)
        e_out_embed_test = self.mymodel()

        # From left
        result_test = alignment2.my_accuracy(e_out_embed_test, self.test_links_tensor, top_k=self.config.top_k,
                                                    metric=self.config.metric)
        result_str1 = self.print_result('Test From Left', result_test)
        # From right
        result_test = alignment2.my_accuracy(e_out_embed_test, self.test_links_tensor, top_k=self.config.top_k,
                                                    metric=self.config.metric, fromLeft=False)
        result_str2 = self.print_result('Test From right', result_test)

        model_result_file = '{}_Result-{}.txt'.format(self.best_mode_pkl_title, self.config.model_param)
        with open(model_result_file, "a") as ff:
            ff.write(result_str1)
            ff.write('\n')
            ff.write(result_str2)
            ff.write('\n')


    def save_model(self, better_epochs_i, epochs_name):  # best-epochs
        # save model to file
        model_savefile = self.best_mode_pkl_title + epochs_name + \
                         str(better_epochs_i) + '-' + self.config.model_param + '.pkl'
        torch.save(self.mymodel.state_dict(), model_savefile)


    def print_result(self, pt_type, run_re):
        ''' Output result '''
        hits, mr, mrr = run_re[0], run_re[1], run_re[2]
        result = pt_type
        result += "==results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}".format(self.config.top_k, hits, mr, mrr)
        self.myprint(result)
        return result



