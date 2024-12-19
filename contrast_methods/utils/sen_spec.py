import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def sen_spec(y_true, y_pre):
    """
    1. sensitivity是一种局部性的指标，表达  正确识别正类个数 / 正类总个数
          - Sensitivity/TPR = TP / (TP + FN)
    2. specificity同理，不同之处为，正确识别负类个数 / 负类总个数
          - Specificity/TNR = TN / (TN + FP)
    """
    TP = [(y_pre[i] == 1 and y_true[i] == 1) for i in range(len(y_true))].count(1)
    TN = [(y_pre[i] == 0 and y_true[i] == 0) for i in range(len(y_true))].count(1)
    FP = [(y_pre[i] == 1 and y_true[i] == 0) for i in range(len(y_true))].count(1)
    FN = [(y_pre[i] == 0 and y_true[i] == 1) for i in range(len(y_true))].count(1)

    return TP, TN, FP, FN


def curve_plot(train_loss, val_loss, val_auc, train_acc, val_acc, args, ii):
    # plt.plot(train_loss, label='train_loss')
    # plt.plot(val_loss, label='val_loss')
    plt.plot(val_auc, label='val_auc')
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    # plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(args.curve_title + f'_fold_{ii}')
    plt.savefig(os.path.join(args.eval_path, f'curve_fold_{ii}.png'))
    # plt.show()
    plt.close()

    with open(os.path.join(args.eval_path, f'record_fold_{ii}.txt'), 'w') as f:
        for idx, (tr_loss, v_loss, v_auc, tr_acc, v_acc) in enumerate(zip(train_loss, val_loss, val_auc, train_acc, val_acc)):
            txt = f'Epoch {idx:03d}\t   train_loss={tr_loss:.4f}\t train_acc={tr_acc:.4f}\t ' \
                  f'val_acc={v_acc:.4f}\t val_auc={v_auc:.4f}\t val_loss={v_loss:.4f} \n'
            f.write(txt)

def curve_plot_cox(train_loss, val_loss, train_acc, val_acc, args, ii):
    # plt.plot(train_loss, label='train_loss')
    # plt.plot(val_loss, label='val_loss')
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    # plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(args.curve_title + f'_fold_{ii}')
    plt.savefig(os.path.join(args.eval_path, f'curve_fold_{ii}.png'))
    # plt.show()
    plt.close()

    with open(os.path.join(args.eval_path, f'record_fold_{ii}.txt'), 'w') as f:
        for idx, (tr_loss, v_loss, tr_acc, v_acc) in enumerate(zip(train_loss, val_loss, train_acc, val_acc)):
            txt = f'Epoch {idx:03d}\t   train_loss={tr_loss:.4f}\t train_acc={tr_acc:.4f}\t ' \
                  f'val_acc={v_acc:.4f}\t val_loss={v_loss:.4f} \n'
            f.write(txt)

