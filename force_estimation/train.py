import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import yaml
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim, nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

np.set_printoptions(suppress=True)
import argparse
from data import DTactDataset
from model import *
import force_estimation
import cv2

# use_wandb = True
use_wandb = False

wrench_range = [[-3, 3], [-3, 3], [-12, 0], [-0.2, 0.2], [-0.2, 0.2], [-0.05, 0.05]]
min_wrench = np.array([.0, .0, .0, .0, .0, .0])
max_wrench = np.array([.0, .0, .0, .0, .0, .0])
for i in range(len(wrench_range)):
    min_wrench[i] = wrench_range[i][0]
    max_wrench[i] = wrench_range[i][1]
wrench_span = max_wrench - min_wrench


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calculate_accuracy(predict_wrench, label_wrench):
    accuracy = torch.sum(torch.abs(predict_wrench - label_wrench) / torch.abs(label_wrench), dim=0)
    # torch.sum(torch.abs(predict_forces - label_forces), dim=0)
    return accuracy


class ForceEstimation:
    def __init__(self, cfg):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default=None, type=str)
        parser.add_argument("--model_layer", default=None, type=str)
        parser.add_argument("--optimizer", default=None, type=str)
        parser.add_argument("--lrs", default=None, type=str2bool)
        parser.add_argument("--image_type", default=None, type=str)
        parser.add_argument("--cuda_index", default=None, type=int)
        parser.add_argument("--train_mode", default=None, type=str2bool)
        parser.add_argument("--test_object", default=None, type=str2bool)
        parser.add_argument("--mixed_image", default=None, type=str2bool)
        parser.add_argument("--pretrained", default=True, type=str2bool)
        parser.add_argument("--batch_size", default=None, type=int)
        parser.add_argument("--num_epoch", default=None, type=int)
        parser.add_argument("--learning_rate", default=None, type=float)
        parser.add_argument("--weight_decay", default=None, type=float)
        parser.add_argument("--end_factor", default=0.4, type=float)
        parser.add_argument("--total_iters", default=20, type=int)
        args = parser.parse_args()

        # parameters
        model_name = args.model_name if args.model_name is not None else cfg['model_list'][cfg['model_choice']][0]
        model_layer = args.model_layer if args.model_name is not None else cfg['model_list'][cfg['model_choice']][1]
        model_type = model_name + '-' + str(model_layer)
        optimizer = args.optimizer if args.optimizer is not None else 'Adam'
        self.lrs = args.lrs if args.lrs is not None else True
        image_type = args.image_type if args.image_type is not None else cfg['image_type']
        cuda_index = args.cuda_index if args.cuda_index is not None else cfg['cuda_index']
        train_mode = args.train_mode if args.train_mode is not None else cfg['train_mode']
        test_object = args.test_object if args.test_object is not None else cfg['test_object']
        mixed_image = args.mixed_image if args.mixed_image is not None else cfg['mixed_image']
        pretrained = args.pretrained & train_mode
        batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
        eval_batch = batch_size if train_mode else cfg['eval_batch']

        # model configuration
        self.device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
        if model_name == 'Resnet':
            self.model = Resnet(layer=int(model_layer), pretrained=pretrained).to(self.device)
        elif model_name == 'Densenet':
            self.model = Densenet(layer=int(model_layer), pretrained=pretrained).to(self.device)
        self.lossFunction = nn.L1Loss(reduction='sum')

        self.min_wrench = torch.tensor(min_wrench, dtype=torch.float).to(self.device)
        self.wrench_span = torch.tensor(wrench_span, dtype=torch.float).to(self.device)

        # test dataset
        test_dataset = DTactDataset(mode='test', root_path=cfg['data_dir'], image_type=image_type,
                                    test_object=test_object, mixed_image=mixed_image)
        self.test_img_num = len(test_dataset)
        self.test_dataLoader = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False)

        if train_mode:
            self.num_epoch = args.num_epoch if args.num_epoch is not None else cfg['num_epoch']
            learning_rate = args.learning_rate if args.learning_rate is not None else cfg['learning_rate']
            regularization = cfg['regularization']
            weight_decay = .0
            if args.weight_decay is not None:
                regularization = True
                weight_decay = args.weight_decay
            elif cfg['regularization']:
                weight_decay = cfg['weight_decay']

            # create directories
            time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + model_type + time_suffix
            self.log_dir = self.save_dir + '/log'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # save basic information in the txt file
            self.write_basic_information(model_type, batch_size, self.num_epoch, learning_rate, regularization,
                                         weight_decay)

            # training dataset
            train_dataset = DTactDataset(mode='train', root_path=cfg['data_dir'],
                                         image_type=image_type, test_object=test_object, mixed_image=mixed_image)
            self.train_img_num = len(train_dataset)
            self.train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # optimizer
            if optimizer == 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
                                                   end_factor=args.end_factor, total_iters=args.total_iters)

            # wandb
            wandb_config = {
                "model_type": model_type,
                "image_type": image_type,
                "optimizer": optimizer,
                "test_object": test_object,
                "mixed_image": mixed_image,
                "pretrained": pretrained,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epoch": self.num_epoch,
                "weight_decay": weight_decay,
                "lrs": self.lrs,
                "end_factor": args.end_factor,
                "total_iters": args.total_iters
            }
            run_name = model_type + '_' + optimizer + '_lr-' + f'{learning_rate:.4f}' + '_OBJ-' + \
                       str(bool(test_object))[0] + '_MIX-' + str(bool(mixed_image))[0]
            if use_wandb:
                self.run_log = force_estimation.init(project="Force_Estimation", config=wandb_config, name=run_name)
            self.train()
        else:
            self.save_dir = cfg['save_dir'] + '/' + model_type + str(cfg['weights'][cfg['model_choice']][0])
            weights_path = self.save_dir + '/epoch_' + str(cfg['weights'][cfg['model_choice']][1]) + '.pt'
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.test_record(cfg['weights'][cfg['model_choice']][1])

    def write_basic_information(self, model_type, batch_size, num_epoch, learning_rate, regularization,
                                weight_decay):
        basic_txt_path = self.save_dir + "/basic.txt"
        basic_txt_file = open(basic_txt_path, 'a')
        basic_txt_file.write(f"model: {model_type}\n")
        basic_txt_file.write(f"batch_size: {batch_size}\n")
        basic_txt_file.write(f"epoch: {num_epoch}\n")
        basic_txt_file.write(f"learning_rate: {learning_rate}\n")
        basic_txt_file.write(f"regularization: {regularization}\n")
        basic_txt_file.write(f"weight_decay: {weight_decay}\n")
        basic_txt_file.close()

    def train(self):
        start_time = time.time()
        test_best_loss = float('inf')
        iter_num = 0
        writer = SummaryWriter(log_dir=self.log_dir)
        train_txt_path = self.save_dir + "/train.txt"
        train_txt_file = open(train_txt_path, 'a')
        best_epoch = 0
        for epoch in range(self.num_epoch):
            train_loss = 0.0
            train_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            force_train_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            self.model.train()
            for batch_idx, dataset in enumerate(self.train_dataLoader):
                self.optimizer.zero_grad()
                images, label_forces = dataset
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                label_forces = label_forces.type(torch.FloatTensor)
                label_forces = label_forces.to(self.device)
                predict_forces = self.model(images)
                # restrict the range
                predict_forces[predict_forces < 0] = 0
                predict_forces[predict_forces > 1] = 1
                loss = self.lossFunction(predict_forces, label_forces)
                loss.backward()
                self.optimizer.step()
                train_error += calculate_accuracy(predict_forces, label_forces)
                train_loss += loss.item()
                force_train_error += torch.sum(torch.abs(predict_forces - label_forces), dim=0)
                iter_num += 1
            # learning rate schedule
            lr = self.optimizer.param_groups[0]["lr"]
            if self.lrs:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]

            train_loss /= self.train_img_num
            train_error = torch.div(train_error, self.train_img_num)
            force_train_error = torch.div(force_train_error, self.train_img_num)

            # test the model
            time_dif = get_time_dif(start_time)
            test_loss, force_test_error, force_test_error_span, test_error = self.test()
            if test_loss < test_best_loss:
                test_best_loss = test_loss
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = '******'
                best_epoch = epoch + 1
            else:
                improve = ''
            msg = '[Epoch: {}/{}, iter: {}], Lr: {:.4f},Train Loss: {:.3f}, Train Error: {}, {} ' \
                  'Test Loss: {:.3f}, Test Error: {}, {}, Time: {} {}'
            print(msg.format(epoch + 1, self.num_epoch, iter_num, lr, train_loss,
                             np.round(force_train_error.detach().cpu().numpy(), 4),
                             np.round(train_error.detach().cpu().numpy(), 4),
                             test_loss, np.round(force_test_error.detach().cpu().numpy(), 4),
                             np.round(force_test_error_span.detach().cpu().numpy(), 3), time_dif / 60, improve))
            writer.add_scalar("loss/train", train_loss, iter_num)
            writer.add_scalar("loss/dev", test_loss, iter_num)
            train_txt_file.write(
                msg.format(epoch + 1, self.num_epoch, iter_num, lr, train_loss,
                           np.around(force_train_error.detach().cpu().numpy(), decimals=2),
                           np.round(train_error.detach().cpu().numpy(), 4),
                           test_loss, np.around(force_test_error.detach().cpu().numpy(), decimals=2),
                           np.round(force_test_error_span.detach().cpu().numpy(), 3), time_dif / 60, improve) + "\n")
            if use_wandb:
                self.run_log.log({"train_loss": train_loss, "test_loss": test_loss})
            self.model.train()
        writer.close()
        if use_wandb:
            self.run_log.finish()
        self.test_record(best_epoch)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            force_test_error = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
            for batch_idx, dataset in enumerate(self.test_dataLoader):
                images, label_forces, image_names = dataset
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                label_forces = label_forces.type(torch.FloatTensor)
                label_forces = label_forces.to(self.device)
                predict_forces = self.model(images)
                loss = self.lossFunction(predict_forces, label_forces)
                test_loss += loss.item()
                test_error += calculate_accuracy(predict_forces, label_forces)
                force_test_error += torch.sum(torch.abs(predict_forces - label_forces), dim=0)
            test_loss /= self.test_img_num
            test_error = torch.div(test_error, self.test_img_num)
            force_test_error = torch.div(force_test_error, self.test_img_num)
            force_test_error_span = torch.mul(force_test_error, self.wrench_span)
        return test_loss, force_test_error, force_test_error_span, test_error

    def test_record(self, epoch):
        self.model.eval()
        test_txt_path = self.save_dir + "/test_" + str(epoch) + '.txt'
        if os.path.exists(test_txt_path):
            os.remove(test_txt_path)
        test_txt_file = open(test_txt_path, 'a')
        test_image_dir = self.save_dir + "/image_" + str(epoch)
        if not os.path.exists(test_image_dir):
            os.makedirs(test_image_dir)
        image_name_list = []
        mean_error_list = []
        truth_list = []
        predict_list = []
        error_list = []
        img_number = 0
        with torch.no_grad():
            start_time = time.time()
            for batch_idx, dataset in enumerate(self.test_dataLoader):
                images, label_forces, image_paths = dataset
                # print(images[0])
                images = images.type(torch.FloatTensor)
                images = images.to(self.device)
                label_forces = label_forces.type(torch.FloatTensor)
                label_forces = label_forces.to(self.device)
                predict_forces = self.model(images)
                print(predict_forces)
                predict_forces[predict_forces < 0] = 0
                predict_forces[predict_forces > 1] = 1
                force_truth = torch.add(torch.mul(label_forces, self.wrench_span), self.min_wrench)
                force_predict = torch.add(torch.mul(predict_forces, self.wrench_span), self.min_wrench)
                force_error = torch.abs(force_truth - force_predict)
                # print(force_error)

                for i in range(images.size(0)):
                    msg = "{}: {} Mean: {:.4f} Tru: {} Pre: {} Error: {}" \
                        .format(1 + i + img_number * batch_idx, image_paths[i].split('image')[-1],
                                force_error[i].mean(),
                                np.round(force_truth[i].detach().cpu().numpy(), 4),
                                np.round(force_predict[i].detach().cpu().numpy(), 4),
                                np.round(force_predict[i].detach().cpu().numpy(), 4))
                    test_txt_file.write(msg + "\n")
                    image_name_list.append(image_paths[i])
                    mean_error_list.append(np.round(force_error[i].mean().item(), 4))
                    truth_list.append(np.round(force_truth[i].detach().cpu().numpy(), 4))
                    predict_list.append(np.round(force_predict[i].detach().cpu().numpy(), 4))
                    error_list.append(np.round(force_error[i].detach().cpu().numpy(), 4))
                img_number = images.size(0)
            test_txt_file.write("--------------------------------------------\n")
            time_dif = get_time_dif(start_time)
            final_error = np.mean(error_list, axis=0)
            msg_time = "Final Error: {}, Time: {:2f}, FPS: {:2f} \n"\
                .format(final_error, time_dif, self.test_img_num / time_dif)
            print(msg_time)
            test_txt_file.write(msg_time)
            test_txt_file.write("--------------------------------------------\n")
            sorted_index = np.argsort(np.array(mean_error_list))
            for j in range(len(error_list)):
                msg_sort = "{}: {} Tru: {} Pre: {}  Err: {}" \
                    .format(1 + j, image_name_list[sorted_index[j]].split('image')[-1],
                            truth_list[sorted_index[j]],
                            predict_list[sorted_index[j]],
                            error_list[sorted_index[j]])
                test_txt_file.write(msg_sort + "\n")
                saved_img = cv2.imread(image_name_list[sorted_index[j]])
                cv2.imwrite(test_image_dir + '/' + str(j + 1) + '.png', saved_img)

            # error_span_list = np.array(error_span_list)
            truth_list = np.array(truth_list)
            predict_list = np.array(predict_list)
            print(truth_list.shape)
            print(predict_list.shape)
            np.save(self.save_dir + '/image_' + str(epoch) + '/truth_list.npy', truth_list)
            np.save(self.save_dir + '/image_' + str(epoch) + '/predict_list.npy', predict_list)


if __name__ == '__main__':
    f = open("force_config.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    force_estimation = ForceEstimation(config)
