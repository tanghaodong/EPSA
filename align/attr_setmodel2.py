import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from autil.sparse_tensor import SpecialSpmm
from tensorboardX import SummaryWriter

from autil import alignment2
from align import attr_align_model2, align_setmodel2
from align.model_htrans import Align_Htrans

class modelClass2():
    def __init__(self, config, myprint_fun):
        super(modelClass2, self).__init__()

        self.myprint = myprint_fun
        self.config = config
        self.best_mode_pkl_title = config.output + time.strftime('%Y-%m-%d_%H_%M_%S-', time.localtime(time.time()))
        self.dropout = nn.Dropout(config.dropout)
        self.special_spmm = SpecialSpmm()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.e_dim = self.v_dim = 300
        self.dim = self.e_dim
        self.Linear = nn.Linear(2 * self.dim, self.dim)
        self.leakyrelu = nn.LeakyReLU(config.LeakyReLU_alpha)  # alpha: leakyrelu
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.special_spmm = SpecialSpmm()
        self.dropout = nn.Dropout(config.dropout)
        self.top_k = config.top_k
        self.neg_k = config.neg_k
        self.metric = config.metric
        self.alpha3 = config.alpha3  # alpha = 0.1
        self.gamma = config.gamma  # gamma = 1.0  margin based loss
        self.l_bata = config.l_bata
        # Load data

        input_data = align_setmodel2.load_data(config, model_type='M-path')

        # train、Valid、Test
        self.train_links = input_data.train_links
        self.valid_links = input_data.valid_links
        # self.test_links = input_data.test_links
        self.train_links_tensor = input_data.train_links_tensor
        self.valid_links_tensor = input_data.valid_links_tensor
        self.test_links_tensor = input_data.test_links_tensor

        # Model and optimizer  【attr_align_model】
        self.mymodel = attr_align_model2.HET_attr_align2(input_data, config)  # E+R+M+V

        self.align_model = Align_Htrans(input_data, config)
        if config.cuda:
            self.mymodel.cuda()
        # [TensorboardX]Summary_Writer
        self.board_writer = SummaryWriter(log_dir=self.best_mode_pkl_title + '-M/', comment='HET_attr_align')

        self.att = MultiHeadAttention(self.dim,6)

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

        # Weight initialization
        self.mymodel.init_weights()


    #### model train
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
            self.mymodel.train() ## ？？修正！
            self.align_model.train()
            self.optimizer.zero_grad()
            # path
            path_embed = self.align_model()
            es_embed, ec_embed = self.mymodel()
            # Semantic aggregates
            e_out_embeds = torch.cat((es_embed, ec_embed, path_embed), -1)
            e_out_embeds = self.relu(e_out_embeds)
            # get neg
            self.regen_neg(epochs_i, es_embed, ec_embed, path_embed, e_out_embeds)

            # loss、acc
            loss_train, e_out_embed = self.mymodel.get_loss(es_embed, ec_embed, path_embed, e_out_embeds, self.train_neg_pairs_es,
                                                            self.train_neg_pairs_ec, self.train_neg_pairs_path, self.train_neg_pairs_e_out)  # 求解loss:
            # Backward and optimize
            loss_train.backward()
            self.optimizer.step()

            if epochs_i % 5 != 0:
                self.myprint('Epoch-{:04d}: train_loss-{:.4f}, cost time-{:.4f}s'.format(
                    epochs_i, loss_train.data.item(), time.time() - epochs_i_t))
            else:
                # accuracy: hits, mr, mrr
                result_train = alignment2.my_accuracy(e_out_embed, self.train_links_tensor, top_k=self.config.top_k,
                                                    metric=self.config.metric)

                # [TensorboardX]
                self.board_writer.add_scalar('train_loss', loss_train.data.item(), epochs_i)
                self.board_writer.add_scalar('train_hits1', result_train[0][0], epochs_i)

                self.myprint('Epoch-{:04d}: train_loss-{:.4f}, cost time-{:.4f}s'.format(
                    epochs_i, loss_train.data.item(), time.time() - epochs_i_t))
                self.print_result('Train', result_train)

            # test
            if epochs_i > 0 and epochs_i % 100 == 0: #
                self.myprint('===Temp TEST 5000 ===')
                # From left
                result_test = alignment2.my_accuracy(e_out_embed, self.test_links_tensor, top_k=self.config.top_k,
                                                     metric=self.config.metric)
                self.print_result('Temp Test From Left', result_test)

            # ********************no early stop********************************************
            if epochs_i >= self.config.start_valid and epochs_i % self.config.eval_freq == 0:
                epochs_i_t = time.time()
                self.mymodel.eval()  # self.train(False)
                self.align_model.eval()
                es_embed, ec_embed = self.mymodel()
                path_embed = self.align_model()
                e_out_embeds = torch.cat((es_embed, ec_embed, path_embed), -1)
                e_out_embeds = self.relu(e_out_embeds)

                loss_val, e_out_embed = self.mymodel.get_loss(es_embed, ec_embed, path_embed, e_out_embeds, self.valid_neg_pairs_es, self.valid_neg_pairs_ec, self.valid_neg_pairs_path, self.valid_neg_pairs_e_out)
                result_val = alignment2.my_accuracy(e_out_embed, self.valid_links_tensor, top_k=self.config.top_k, metric=self.config.metric)
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
                    # no best, but save model every 20 epochs
                    if epochs_i % self.config.eval_save_freq == 0:
                        self.save_model(epochs_i, 'eval-epochs')
                    # bad model, stop train
                    bad_counter += 1
                    self.myprint('bad_counter++:' + str(bad_counter))
                    if bad_counter == self.config.patience:  # patience=15
                        self.myprint('Epoch-{:04d}, bad_counter.'.format(epochs_i))
                        break

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
    def regen_neg(self, epochs_i, es_embed, ec_embed, path_embed, e_out_embeds):
        if epochs_i % self.config.sample_neg_freq == 0:  # sample negative pairs every 20 epochs
            with torch.no_grad():
                # Negative sample sampling-training pair (positive sample and negative sample)

                self.train_neg_pairs_es = alignment2.gen_neg(es_embed, self.train_links, self.config.metric, self.config.neg_k)
                self.train_neg_pairs_ec = alignment2.gen_neg(ec_embed, self.train_links, self.config.metric, self.config.neg_k)
                self.train_neg_pairs_path = alignment2.gen_neg(path_embed, self.train_links, self.config.metric,
                                                                self.config.neg_k)
                self.train_neg_pairs_e_out = alignment2.gen_neg(e_out_embeds, self.train_links, self.config.metric,
                                                                self.config.neg_k)


                self.valid_neg_pairs_es = alignment2.gen_neg(es_embed, self.valid_links, self.config.metric, self.config.neg_k)
                self.valid_neg_pairs_ec = alignment2.gen_neg(ec_embed, self.valid_links, self.config.metric, self.config.neg_k)
                self.valid_neg_pairs_e_out = alignment2.gen_neg(e_out_embeds, self.valid_links, self.config.metric, self.config.neg_k)
                self.valid_neg_pairs_path = alignment2.gen_neg(path_embed, self.valid_links, self.config.metric, self.config.neg_k)

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
        es_embed, ec_embed = self.mymodel()
        e_out_embed_test = torch.cat((es_embed, ec_embed), dim=1)

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

        return es_embed, ec_embed


    def re_train(self, epochs_beg, model_savefile):
        ''' restart run best model '''
        # Restore best model
        es_embed, ec_embed = self.re_test(model_savefile)

        self.myprint('==== begin retrain ====')
        #   generate negative samples
        self.regen_neg(0, es_embed, ec_embed)
        self.model_train(epochs_beg)



    def save_model(self, better_epochs_i, epochs_name):
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

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.W_q = nn.Linear(input_size, input_size)
        self.W_k = nn.Linear(input_size, input_size)
        self.W_v = nn.Linear(input_size, input_size)

        self.fc_out = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_len = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)

        x = x.permute(0, 1).contiguous().view(batch_size, seq_len, -1)
        #x = self.fc_out(x)

        return x