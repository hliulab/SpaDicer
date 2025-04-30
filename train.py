import numpy as np
import torch
from torch import optim, nn
from torch.optim import lr_scheduler
import utils
from dataset import trainDataset
import config
from model.SpaDicer import SpaDicer
from model.loss import MSE, SIMSE, OrthogonalLoss
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
def train():
    #######################
    # load data           #
    #######################
    print("data loader...")
    target_set = trainDataset(data_path=config.sc_data,
                                    label_path=config.sc_meta,
                                    dataset_type='target')
    source_set = trainDataset(data_path=config.st_data,
                                    label_path=config.st_meta,
                                    dataset_type='source')

    dataloader_source = torch.utils.data.DataLoader(
            dataset=source_set,
            batch_size=config.batchsize,
            shuffle=True,
            num_workers=1
        )
    dataloader_target = torch.utils.data.DataLoader(
            dataset=target_set,
            batch_size=config.batchsize,
            shuffle=True,
            num_workers=1
        )
    print("data loader finished!")

    #####################
    #  load model       #
    #####################

    model = SpaDicer(config.inf, config.input_class)

    #####################
    # setup optimizer   #
    #####################


    def exp_lr_scheduler(optimizer, step, init_lr=config.lr, lr_decay_step=config.lr_decay_step, step_decay_weight=config.step_decay_weight):
        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))
        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer
    '''Loss Function and Optimization Function'''
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weiht_decay)
    loss_classification = torch.nn.CrossEntropyLoss()
    loss_recon1 = MSE()
    loss_fn = nn.MSELoss(reduction='mean')
    loss_recon2 = SIMSE()
    loss_orth = OrthogonalLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if config.cuda:
        model = model.cuda()
        loss_classification = loss_classification.cuda()
        loss_recon1 = loss_recon1.cuda()
        loss_recon2 = loss_recon2.cuda()
        loss_orth = loss_orth.cuda()

    for p in model.parameters():
        p.requires_grad = True

        '''training network'''


        len_dataloader = min(len(dataloader_source), len(dataloader_target))

        current_step = 0


        model.train()
        for epoch in range(config.n_epoch):
            print("train epoch is:", epoch)
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)
            i = 0
            train_label = []
            train_predicted = []
            while i < len_dataloader:
                '''target data training '''


                data_target = next(data_target_iter)
                t_img, t_label = data_target

                optimizer.zero_grad()
                batch_size = len(t_label)

                input_img = torch.FloatTensor(batch_size, config.inf)
                class_label = torch.LongTensor(batch_size)

                if config.cuda:
                    t_img = t_img.cuda()
                    t_label = t_label.cuda()
                    input_img = input_img.cuda()
                    class_label = class_label.cuda()

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)
                target_inputv_img = Variable(input_img)
                target_classv_label = Variable(class_label)

                target_inputv_img = target_inputv_img.to(torch.float32)
                result = model(input_data=target_inputv_img, mode='target', rec_scheme='all')
                target_privte_code, target_share_code, target_class_label, target_rec_code = result

                target_classification = loss_classification(target_class_label, target_classv_label)
                prob_softmax = F.softmax(target_class_label, dim=1)
                pred = torch.max(prob_softmax, dim=1)[1]

                sum_train = torch.eq(pred, target_classv_label.to(config.device)).sum()

                target_mse = loss_recon1(target_rec_code, target_inputv_img)

                target_simse = loss_recon2(target_rec_code, target_inputv_img)

                target_orth = loss_orth(target_privte_code, target_share_code)

                '''source data training '''
                data_source = next(data_source_iter)
                s_img, s_label = data_source

                batch_size = len(s_label)

                input_img = torch.FloatTensor(batch_size, config.inf)
                class_label = torch.FloatTensor(batch_size)

                if config.cuda:
                    s_img = s_img.cuda()
                    s_label = s_label.cuda()
                    input_img = input_img.cuda()
                    class_label = class_label.cuda()

                input_img.resize_as_(input_img).copy_(s_img)
                class_label.resize_as_(s_label).copy_(s_label)
                source_inputv_img = Variable(input_img)
                source_classv_label = Variable(class_label)

                result = model(input_data=source_inputv_img, mode='source', rec_scheme='all')
                source_privte_code, source_share_code, source_class_label, source_rec_code = result

                source_classification = loss_fn(source_class_label, source_classv_label.float())

                source_orth = loss_orth(source_privte_code, source_share_code)

                source_mse = loss_recon1(source_rec_code, source_inputv_img)
                source_simse = loss_recon2(source_rec_code, source_inputv_img)
                '''Gradient Scaling Strategy for Updating Loss Weights'''
                model_parameters = list(model.parameters())

                target_classification_grad = torch.autograd.grad(target_classification, model_parameters,
                                                                 create_graph=True,
                                                                 allow_unused=True)
                target_orth_grad = torch.autograd.grad(target_orth, model_parameters, create_graph=True,
                                                       allow_unused=True)
                target_mse_grad = torch.autograd.grad(target_mse, model_parameters, create_graph=True,
                                                      allow_unused=True)
                target_simse_grad = torch.autograd.grad(target_simse, model_parameters, create_graph=True,
                                                        allow_unused=True)

                source_classification_grad = torch.autograd.grad(source_classification, model_parameters,
                                                                 create_graph=True,
                                                                 allow_unused=True)
                source_orth_grad = torch.autograd.grad(source_orth, model_parameters, create_graph=True,
                                                       allow_unused=True)
                source_mse_grad = torch.autograd.grad(source_mse, model_parameters, create_graph=True,
                                                      allow_unused=True)
                source_simse_grad = torch.autograd.grad(source_simse, model_parameters, create_graph=True,
                                                        allow_unused=True)
                # Calculate the Norm of Each Loss
                target_classification_norm = torch.norm(
                    torch.stack([torch.norm(g.detach(), 2) for g in target_classification_grad]), 2)
                target_orth_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in target_orth_grad]), 2)
                target_mse_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in target_mse_grad]), 2)
                target_simse_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in target_simse_grad]), 2)
                source_classification_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in source_classification_grad]), 2)
                source_orth_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in source_orth_grad]), 2)
                source_mse_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in source_mse_grad]), 2)
                source_simse_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in source_simse_grad]), 2)

                # Calculate the Weight of Each Loss
                total_norm = target_classification_norm + target_orth_norm + target_mse_norm + target_simse_norm + \
                             source_classification_norm + source_orth_norm + source_mse_norm + source_simse_norm
                w1 = target_classification_norm / total_norm
                w2 = target_orth_norm / total_norm
                w3 = target_mse_norm / total_norm
                w4 = target_simse_norm / total_norm
                w5 = source_classification_norm / total_norm
                w6 = source_orth_norm / total_norm
                w7 = source_mse_norm / total_norm
                w8 = source_simse_norm / total_norm

                loss = w1 * target_classification + w2 * target_orth + w3 * target_mse + w4 * target_simse + w5 * source_classification + w6 * source_orth + w7 * source_mse + w8 * source_simse
                # loss = target_classification + target_diff + target_mse + target_simse + source_classification + source_diff + source_mse + source_simse
                loss.backward()
                optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
                optimizer.step()

                i += 1
                current_step += 1

            acc_train = sum_train / len(dataloader_target)
            print("train_acc:", acc_train)
            scheduler.step()
            auc_train = utils.cal_roc_auc_score(np.array(train_label), np.array(train_predicted), config.input_class)
            utils.plot_auc(config.label_dict, train_label, train_predicted, config.result_path)
            print("train_auc:", auc_train)

            print('source_classification: %f, source_orth: %f, ' \
                  'source_mse: %f, source_simse: %f,target_orth: %f, ' \
                  'target_mse: %f, target_simse: %f, target_classification: %f' \
                  % (source_classification.data.cpu().numpy(),
                     source_orth.data.cpu().numpy(),
                     source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(),
                     target_orth.data.cpu().numpy(), target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy(),
                     target_classification.data.cpu().numpy()))
            if ((epoch + 1) % 10 == 0):
                torch.save(model.state_dict(), config.model_root + str(epoch + 1) + '.pth')