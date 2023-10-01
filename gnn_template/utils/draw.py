import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import json
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax

def unsorted_segment_mean(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    data = torch.tensor(data).reshape(-1,1)
    segment_ids = torch.tensor(segment_ids)
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def simpleConvolve1d(signal:np.array, kernel:np.array, stride=1,method='mean')->np.array:
    # method: mean, sum
    # signal:[N], kernel:[M]. output: [(N-M+1)/stride]
    N, M = len(signal), len(kernel)
    # output = np.zeros( int((N-M+1)/stride) )
    if method=='mean':
        denom = M
    elif method=='sum':
        denom = 1
    else:
        print('error: method should be "mean" or "sum" ')
        exit()

    output = [(signal[M-1::-1] * kernel).sum()/denom]
    for isignal in range(stride,N,stride):
        sigSlice = signal[isignal+M-1:isignal-1:-1]
        if len(sigSlice)<len(kernel):
            continue
        output.append((sigSlice*kernel).sum()/denom)

    return np.array(output)

def draw_loss_acc(res, logDir):
    trainSliceId = np.unique(res['train_slice'])
    testSliceId = np.unique(res['test_slice'])
    # Draw loss with data set
    trainloss = res['train_loss']
    testloss = res['test_loss']
    # loss limit
    ymax = max(np.max(trainloss), np.max(testloss)) * 1.2

    # plt.figure(dpi=500)
    # plt.plot(range(1,len(trainloss)+1), trainloss, label="train loss")
    # plt.xlabel('set')
    # plt.ylim(0, ymax)
    # plt.legend()
    # plt.savefig(logDir+"/loss_to_sets_train.png")
    
    # trainloss = res['train_loss']
    # testloss = res['test_loss']
    # plt.figure(dpi=500)
    # plt.plot(range(1,len(testloss)+1), testloss, label="test loss")
    # plt.xlabel('set')
    # plt.ylim(0, ymax)
    # plt.legend()
    # plt.savefig(logDir+"/loss_to_sets_test.png")

    # Draw loss with epoch
    trainloss = simpleConvolve1d(trainloss, np.ones(len(trainSliceId)), stride=len(trainSliceId), method='mean')
    testloss = simpleConvolve1d(testloss, np.ones(len(testSliceId)), stride=len(testSliceId), method='mean')
    plt.figure(figsize=(5, 4), dpi=500, constrained_layout=True)
    plt.plot(range(1,len(trainloss)+1), trainloss, label="train loss")
    plt.plot(range(1,len(trainloss)+1), testloss, label="test loss")
    plt.xlabel('epoch')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/loss_to_epoch.pdf")

    # Draw acc with data set
    trainacc = res['train_acc']
    testacc = res['test_acc']
    # Acc limit
    ymax = max(np.max(trainacc), np.max(testacc)) * 1.2
    ymin = min(np.min(trainacc), np.min(testacc)) * 0.8

    # plt.figure(dpi=500)
    # plt.plot(range(1,len(trainacc)+1), trainacc, label="train acc")
    # plt.xlabel('set')
    # plt.ylim(ymin, ymax)
    # plt.legend()
    # plt.savefig(logDir+"/acc_to_sets_train.png")
    
    # trainacc = res['train_acc']
    # testacc = res['test_acc']
    # plt.figure(dpi=500)
    # plt.plot(range(1,len(testacc)+1), testacc, label="test acc")
    # plt.xlabel('set')
    # plt.ylim(ymin, ymax)
    # plt.legend()
    # plt.savefig(logDir+"/acc_to_sets_test.png")

    # Draw acc with epoch
    trainacc = simpleConvolve1d(trainacc, np.ones(len(trainSliceId)), stride=len(trainSliceId), method='mean')
    testacc = simpleConvolve1d(testacc, np.ones(len(testSliceId)), stride=len(testSliceId), method='mean')
    plt.figure(figsize=(5, 4), dpi=500, constrained_layout=True)
    plt.plot(range(1,len(trainacc)+1), trainacc, label="train acc")
    plt.plot(range(1,len(trainacc)+1), testacc, label="test acc")
    plt.xlabel('epoch')
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(logDir+"/acc_to_epoch.pdf")
    plt.cla()
    plt.close('all')


def draw_roc_score(truth_label, score, weights, logDir, raw_score):
    # Get roc curve
    sig_idx = (truth_label==1)
    signal_scores, background_scores = score[sig_idx], score[~sig_idx]
    # weights = weights.clip(0)
    sig_weight, bkg_weight = weights[sig_idx], weights[~sig_idx]

    fpr, tpr, _ = roc_curve(truth_label, score, sample_weight=weights)
    # roc_auc = auc(fpr, tpr)
    roc_auc = np.trapz(tpr, fpr)
    print(f"auc: {roc_auc}.", flush=True)

    # Draw roc curve
    plt.figure(figsize=(5, 4), dpi=500, constrained_layout=True)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(logDir+"roc.pdf")

    # Draw score
    signal_scores, background_scores = score[sig_idx], score[~sig_idx]
    bins = 50
    # score = raw_score
    # bins = np.linspace(-3, 3, 100)
    # signal_scores, background_scores = score[sig_idx], score[~sig_idx]
    plt.figure(figsize=(5, 4), dpi=500, constrained_layout=True)
    plt.hist(signal_scores, bins=bins, weights=sig_weight*1000, label='Signal x 1e3', color='red', histtype='step', linewidth=2)
    plt.hist(background_scores, bins=bins, weights=bkg_weight, label='Background', color='blue', histtype='step', linewidth=2)
    plt.xlabel('Score')
    plt.ylabel('Counts')
    # plt.yscale('log')
    plt.title('Signal vs. Background Score')

    plt.legend(loc='upper center')
    plt.savefig(logDir+"score.pdf")
    plt.cla()


def draw_significance(truth_label, score, weights, logDir):
    """
    Significance Curve
    """
    sig_idx = (truth_label==1)
    sig_score, bkg_score = score[sig_idx], score[~sig_idx]
    sig_weight, bkg_weight = weights[sig_idx], weights[~sig_idx]
    # sig_weight, bkg_weight = np.ones(len(sig_score)), np.ones(len(bkg_score))
    # sig_weight = sig_weight * 0.01

    bins = np.linspace(0, 1, num=200, endpoint=True)
    hist_sig, _ = np.histogram(sig_score, bins=bins, weights=sig_weight)
    hist_bkg, _ = np.histogram(bkg_score, bins=bins, weights=bkg_weight)
    s = np.cumsum(hist_sig[::-1])[::-1]
    b = np.cumsum(hist_bkg[::-1])[::-1]


    sig_err = np.sqrt(np.histogram(sig_score, bins=bins, weights=sig_weight ** 2)[0])
    bkg_err = np.sqrt(np.histogram(bkg_score, bins=bins, weights=bkg_weight ** 2)[0])
    s_err = np.sqrt(np.cumsum(sig_err[::-1] ** 2)[::-1])
    b_err = np.sqrt(np.cumsum(bkg_err[::-1] ** 2)[::-1])

    significance = (s / np.sqrt(s + b))
    significance[np.isnan(significance)] = 0

    def sig_unc(s, b, ds, db):
        t1 = ((np.sqrt(s + b) - s / (2 * np.sqrt(s + b))) / (s + b) * ds) ** 2
        t2 = (-(s * 1. / (2 * np.sqrt(s + b)) / (s + b)) * db) ** 2
        return np.sqrt(t1 + t2)

    significance_err = sig_unc(s, b, s_err, b_err)
    significance_err[np.isnan(significance_err)] = 0
    significance_with_min_bkg = max([(y, x) for x, y in enumerate(significance) if b[x] > 1.0])

    significance_value = significance_with_min_bkg[0]
    mva_score = bins[1 + significance_with_min_bkg[1]]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=500, constrained_layout=True)
    plt.plot(bins[1:], significance, color='#3776ab')
    plt.fill_between(
        bins[1:], significance - significance_err, significance + significance_err, alpha=0.35,
        edgecolor='#3776ab', facecolor='#3776ab', hatch='///', linewidth=0, interpolate=True
    )

    plt.vlines(
        x=mva_score, ymin=0, ymax=significance_value,
        colors='purple',
        label=f'max Sig. = {significance_value:.3f} at {mva_score:.2f}'
    )

    plt.ylabel('Significance')
    plt.xlabel('MVA score')
    plt.legend(loc=3)
    fig.savefig(logDir+"significance.pdf")

def draw_all(log_dir, out_dir, if_apply):
    with open(log_dir+"/train-result.json",'r') as f:
        res = json.load(f)
    
    draw_loss_acc(res, out_dir)

    if if_apply:
        out = np.load(log_dir + "outApply_GPU0.npy")
        weights = np.load(log_dir + "sampleweightApply.npy")
    else:
        out = np.load(log_dir + "outTest_GPU0.npy")
        weights = np.load(log_dir + "sampleweightTest.npy")
        
    truth_label = out[:,0]
    raw_score = out[:,2]
    raw_score = out[:,2] - out[:,1]
    score = softmax(out[:,1:], axis=1)[:,1]
    draw_roc_score(truth_label=truth_label, score=score, weights=weights, logDir=out_dir, raw_score=raw_score)

    draw_significance(truth_label=truth_label, score=score, weights=weights, logDir=out_dir)



if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description="Draw loss and acc by Cen Mo")
    parser.add_argument('--dir', '-d', type=str, default="./", help='Directory of trainning result. Defualt=./')
    parser.add_argument('--apply', '-a', type=int, default=0, help='Use apply result. Defualt=0')
    # parser.add_argument('--res', '-r', type=str, default="./train-result.json", help='Trainning result. Defualt=./train-result.json')
    # parser.add_argument('--scores', '-s', type=str, default="./outTest_GPU0.npy", help='Trainning scores. Defualt=./outTest_GPU0.npy')
    parser.add_argument('--out_dir', '-o', type=str, default="./images/", help='Output directory. Default=./images/')
    args = parser.parse_args()

    draw_all(args.dir, args.out_dir, args.apply)