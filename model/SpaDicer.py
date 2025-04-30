import torch
import torch.nn as nn
class SpaDicer(nn.Module):
    def __init__(self, in_f, n_class):
        super(SpaDicer, self).__init__()
        self.code_size_l = 512
        self.code_size = 256
        self.code_size_r = 128
        self.in_f = in_f
        self.n_class = n_class
        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('fc_pse1', nn.Linear(in_features=self.in_f, out_features=self.code_size_l))
        self.source_encoder_fc.add_module('ac_pse1', nn.ReLU(True))
        self.source_encoder_fc.add_module('fc_pse2', nn.Linear(in_features=self.code_size_l, out_features=self.code_size))
        self.source_encoder_fc.add_module('ac_pse2', nn.ReLU(True))
        #########################################
        # private target encoder
        #########################################

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('fc_pte1', nn.Linear(in_features=in_f, out_features=self.code_size_l))
        self.target_encoder_fc.add_module('ac_pte1', nn.ReLU(True))
        self.target_encoder_fc.add_module('fc_pte2', nn.Linear(in_features=self.code_size_l, out_features=self.code_size))
        self.target_encoder_fc.add_module('ac_pte2', nn.ReLU(True))

        ################################
        # shared encoder
        ################################

        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('fc_se1', nn.Linear(in_features=in_f, out_features=self.code_size_l))
        self.shared_encoder_fc.add_module('ac_se2', nn.ReLU(True))
        self.shared_encoder_fc.add_module('fc_se3', nn.Linear(in_features=self.code_size_l, out_features=self.code_size))


        # Ps
        self.shared_encoder_pred_class_source = nn.Sequential()
        self.shared_encoder_pred_class_source.add_module('fc_se4', nn.Linear(in_features=self.code_size, out_features=self.code_size_r))
        self.shared_encoder_pred_class_source.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class_source.add_module('fc_se6', nn.Linear(in_features=self.code_size_r, out_features=2))

        # Pt
        self.shared_encoder_pred_class_target = nn.Sequential()
        self.shared_encoder_pred_class_target.add_module('fc_se_t1', nn.Linear(in_features=self.code_size, out_features=self.code_size_r))
        self.shared_encoder_pred_class_target.add_module('relu_se_t1', nn.ReLU(True))
        self.shared_encoder_pred_class_target.add_module('fc_se_t3', nn.Linear(in_features=self.code_size_r, out_features=n_class))

        ######################################
        # Decoder
        ######################################

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=self.code_size*2, out_features=self.code_size_l))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))
        self.shared_decoder_fc.add_module('fc_sd2', nn.Linear(in_features=self.code_size_l, out_features=in_f))

    def forward(self, input_data, mode, rec_scheme='all'):

        result = []

        if mode == 'source':

            # source private encoder
            private_code = self.source_encoder_fc(input_data)

        elif mode == 'target':

            # target private encoder
            private_code = self.target_encoder_fc(input_data)

        result.append(private_code)

        # shared encoder
        shared_code = self.shared_encoder_fc(input_data)
        result.append(shared_code)


        if mode == 'source':
            class_label = self.shared_encoder_pred_class_source(shared_code)
            result.append(class_label)

        if mode == 'target':
            class_label = self.shared_encoder_pred_class_target(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = torch.cat((private_code, shared_code), dim=1)

        elif rec_scheme == 'private':
            union_code = private_code
        rec_vec = self.shared_decoder_fc(union_code)

        result.append(rec_vec)

        return result