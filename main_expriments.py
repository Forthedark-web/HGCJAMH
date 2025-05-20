import torch
from prepareData import prepare_data
import numpy as np
import pandas as pd
from torch import optim
from param import parameter_parser
from Module import HGCLAMIR
from utils import get_L2reg, Myloss
from Calculate_Metrics import Metric_fun
from trainData import Dataset
import ConstructHW
import plot2
from scipy import interpolate
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0")




def train_epoch(model, train_data, optim, opt,vaLi):
    model.train()
    regression_crit = Myloss()

    one_index = train_data[2][0].to(device).t().tolist()
    zero_index = train_data[2][1].to(device).t().tolist()


    dis_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)

    concat_miRNA = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA)
    concat_mi_tensor = concat_mi_tensor.to(device)

    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_mi_Km = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[5])
    G_mi_Kn = G_mi_Kn.to(device)
    G_mi_Km = G_mi_Km.to(device)

    concat_dis = np.hstack(
        [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)


    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[13],is_probH=False)
    G_dis_Km = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[5])
    G_dis_Kn = G_dis_Kn.to(device)
    G_dis_Km = G_dis_Km.to(device)



    for epoch in range(1, opt.epoch + 1):
        score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
                                               G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)

        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        reg_loss = get_L2reg(model.parameters())

        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00002 * reg_loss
        optim.zero_grad()
        tol_loss.backward()
        optim.step()

    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor,
                                                                          concat_dis_tensor,
                                                                          G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,vaLi)


    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,vaLi):
    model.eval()
    score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
                        G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
    #计算得分矩阵
    score_np = score.detach().cpu().numpy()  # 转换为 NumPy 数组
    pd.DataFrame(score_np).to_csv(f'prediction_fold_{vaLi}.csv', index=False, header=False)
    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()

    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]


    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]

    return true_one, true_zero, pre_one, pre_zero


def evaluate(true_one, true_zero, pre_one, pre_zero):
    Metric = Metric_fun()
    metrics_tensor = np.zeros((1, 7))

    for seed in range(10):
        test_po_num = true_one.shape[0]
        test_index = np.array(np.where(true_zero == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        test_ne_index = tuple(test_index[:, :test_po_num])

        eval_true_zero = true_zero[test_ne_index]
        eval_true_data = torch.cat([true_one, eval_true_zero])
        # print(f"第{seed+1}次抽样：正样本数量 = {true_one.shape[0]}, 负样本数量 = {eval_true_zero.shape[0]}")

        eval_pre_zero = pre_zero[test_ne_index]
        eval_pre_data = torch.cat([pre_one, eval_pre_zero])
        # print(f"第{seed+1}次抽样：测试总数量 = {eval_true_data.shape[0]}, 预测总数量 = {eval_pre_data.shape[0]}")
        metrics_tensor = metrics_tensor + Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)

    metrics_tensor_avg = metrics_tensor / 10

    return metrics_tensor_avg


def main(opt):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)

    metrics_cross = np.zeros((1, 7))




    for i in range(opt.validation):
        hidden_list = [256, 256]
        num_proj_hidden = 64

        model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(model, train_data[i],optimizer,
                                                                                     opt,i)

        metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one,
                                 pre_score_zero)

        print("--------- epoch num:------------", i + 1)
        print('AUC:', metrics_value[0][0])
        print('AUPR:', metrics_value[0][1])
        print('F1:', metrics_value[0][2])
        print('Acc:', metrics_value[0][3])
        print('Recall:', metrics_value[0][4])
        print('Spe:', metrics_value[0][5])
        print('Precision:', metrics_value[0][6])

        metrics_cross = metrics_cross + metrics_value

    metrics_cross_avg = metrics_cross / opt.validation
    print("---------5 fold epoch:------------")
    print('AUC:', metrics_cross_avg[0][0])
    print('AUPR:', metrics_cross_avg[0][1])
    print('F1:', metrics_cross_avg[0][2])
    print('Acc:', metrics_cross_avg[0][3])
    print('Recall:', metrics_cross_avg[0][4])
    print('Spe:', metrics_cross_avg[0][5])
    print('Precision:', metrics_cross_avg[0][6])




if __name__ == '__main__':
    args = parameter_parser()
    main(args)