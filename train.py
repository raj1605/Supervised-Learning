import  torch

import load
import time
import torch.nn.functional as F
import numpy
import random
import logging
import torch.optim as optim

from ssl_lib.consistency.builder import gen_consistency
from ssl_lib.algs.builder import gen_ssl_alg
from ssl_lib.models.builder import gen_model
from ssl_lib.misc.meter import Meter
from ssl_lib.param_scheduler import scheduler
from ssl_lib.models import utils as model_utils



def evaluation(raw_model, eval_model, loader, device):
    raw_model.eval()
    eval_model.eval()
    sum_raw_acc = sum_acc = sum_loss = 0
    with torch.no_grad():
        for (data, labels) in loader:
            data, labels = data.to(device), labels.to(device)

            #forward pass
            preds = eval_model(data)
            raw_preds = raw_model(data)
            #softmax comes with cross entropy loss for numerical stability in the pytorch
            loss = F.cross_entropy(preds, labels)
            sum_loss += loss.item()
            #get max predicton over axis one which means we take the max over the columns, returns maximum value in each row:
            #the outout of max is the index of the maximum prediction (second element)
            # and the prediction value the (first element), output : ( max value of presiction, index)
            # we need the index which coresponds to the class
            acc = (preds.max(1)[1] == labels).float().mean()
            raw_acc = (raw_preds.max(1)[1] == labels).float().mean()
            #updatae count
            sum_acc += acc.item()
            sum_raw_acc += raw_acc.item()
    mean_raw_acc = sum_raw_acc / len(loader)
    mean_acc = sum_acc / len(loader)
    mean_loss = sum_loss / len(loader)
    raw_model.train()
    eval_model.train()
    return mean_raw_acc, mean_acc, mean_loss

'''
:param:labeled: is the labele of labeled data not psudolabels
avarage_model: will be the same as eval model if we do not use exponential moving average for evaluaton
'''


def param_update(
    cfg,
    cur_iteration,
    model,
    teacher_model,
    optimizer,
    ssl_alg,
    consistency,
    labeled_data,
    ul_weak_data,
    ul_strong_data,
    labels,
    average_model
):

    #measure the time of one iteration
    start_time = time.time()


    #concatenate all labeled data and unlabeled data
    all_data = torch.cat([labeled_data, ul_weak_data], 0)

    forward_func = model.forward
    stu_logits = forward_func(all_data)
    model.logits_with_feature()
    features = []

    # get prediction for labeled data
    labeled_preds = stu_logits[:labeled_data.shape[0]]

    #get prediction for unlabled data
    stu_unlabeled_weak_logits = stu_logits[labels.shape[0]:]

    L_supervised = F.cross_entropy(labeled_preds, labels)

    if cfg.coef > 0:

        # calc consistency loss
        model.update_batch_stats(False)
# ssl_alg  return ConsistencyRegularization and ConsistencyRegularization reurns stu_preds, adjusted targets, mask for psuldo labeling
        y, targets, mask = ssl_alg(
            stu_preds= stu_unlabeled_weak_logits,
            #if there is no teacher the tea_logit is the model(student) logits
            tea_logits=stu_unlabeled_weak_logits.detach(),
            #in the original code the ul_strong_data can be the same as ul_weak_data
            #data=ul_strong_data,
            data = ul_weak_data,
            stu_forward=stu_logits,
            #if there is no teacher model, the t_forward_func is the same as forward_func
            tea_forward=stu_logits
            )
        model.update_batch_stats(True)
        #returns the loss from for example CrossEntropy class which returns
        #consistency is consistency type
        L_consistency = consistency(y, targets, mask, weak_prediction=stu_unlabeled_weak_logits.softmax(1))

   #supervised learning
    else:
        L_consistency = torch.zeros_like(L_supervised)
        mask = None

   #schaduler for coef of unsupervised loss
    coef = scheduler.linear_warmup(cfg.coef, cfg.warmup_iter, cur_iteration + 1)
   # calc total loss
    loss = L_supervised + coef * L_consistency


    if cfg.entropy_minimization > 0:
        loss -= cfg.entropy_minimization * \
                    (stu_unlabeled_weak_logits.softmax(1) * F.log_softmax(stu_unlabeled_weak_logits, 1)).sum(1).mean()

    # update parameters
    #get access to current learning rate
    cur_lr = optimizer.param_groups[0]["lr"]
    #zero the parameter gradients
    optimizer.zero_grad()
    #take the deravitives(gradients) and backward
    loss.backward()
    #if we have weight regularization in the loss
    #weight decay factor 0.2 after 400,000 iterations
    if cfg.weight_decay > 0:
        decay_coeff = cfg.weight_decay * cur_lr
        model_utils.apply_weight_decay(model.modules(), decay_coeff)
    #update the parameters
    optimizer.step()

    # update evaluation model's parameters by exponential moving average
    if cfg.weight_average:
        model_utils.ema_update(
            average_model, model, cfg.wa_ema_factor,
            cfg.weight_decay * cur_lr if cfg.wa_apply_wd else None)

    # calculate accuracy for labeled data
    acc = (labeled_preds.max(1)[1] == labels).float().mean()


    return {
            "acc": acc,
            "loss": loss.item(),
            "sup loss": L_supervised.item(),
            "ssl loss": L_consistency.item(),
            "mask": mask.float().mean().item() if mask is not None else 1,
            "coef": coef,
            "sec/iter": (time.time() - start_time)
    },features


def main(cfg, logger):

    # set seed
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    #all the model parameters and the input data should be on the same gpu or RAM
    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benckmark = True
        print("running on GPU!")
    else:
        logger.info("CUDA is NOT available")
        device = "cpu"
        print("running on CPU")

    # build data loader
    logger.info("load dataset")
    data_loaders = load.get_dataloaders(root= cfg.root, data=cfg.dataset, n_labels=cfg.n_labels, n_unlabels=cfg.n_unlabels, n_valid=cfg.n_valid,
                                        l_batch_size=cfg.l_batch_size, ul_batch_size=cfg.ul_batch_size,
                                        test_batch_size=cfg.test_batch_size, iterations=cfg.iteration,
                                        n_class=cfg.n_class, ratio=cfg.ratio, unlabeled_aug=cfg.unlabeled_aug, cfg=cfg)
    label_loader = data_loaders['labeled']
    unlabel_loader = data_loaders['unlabeled']
    test_loader = data_loaders['test']
    val_loader = data_loaders['valid']
    num_classes = cfg.n_class
    img_size = cfg.img_size
    print("data is loaded!")


    # set consistency type: consistency type (cross entropy, mean squre)
    consistency = gen_consistency(cfg.consis, cfg)
    # set ssl algorithm
    ssl_alg = gen_ssl_alg(cfg.alg, cfg)
    # build student model
    model = gen_model(cfg.arch, num_classes, img_size).to(device)
    # build teacher model
    if cfg.ema_teacher:
        teacher_model = gen_model(cfg.arch, num_classes, img_size).to(device)
        teacher_model.load_state_dict(model.state_dict())
    else:
        teacher_model = None
    # for evaluation
    if cfg.weight_average:
        average_model = gen_model(cfg.arch, num_classes, img_size).to(device)
        average_model.load_state_dict(model.state_dict())
    else:
        average_model = None
# sets the model in training mode (it does not train the model)
    model.train()

    logger.info(model)

    # build optimizer
    if cfg.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), cfg.lr, cfg.momentum, weight_decay=0, nesterov=True
        )
    elif cfg.optimizer == "adam":
        optimizer = optim.AdamW(
            model.parameters(), cfg.lr, (cfg.momentum, 0.999), weight_decay=0
        )
    else:
        raise NotImplementedError
    # set lr scheduler
    if cfg.lr_decay == "cos":
        lr_scheduler = scheduler.CosineAnnealingLR(optimizer, cfg.iteration)
    elif cfg.lr_decay == "step":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], cfg.lr_decay_rate)
    else:
        raise NotImplementedError

    # init meter
    metric_meter = Meter()
    maximum_val_acc = 0
    logger.info("training")

    feature_vectors_mapping = {}
    for i,(l_data, ul_data) in enumerate(zip(label_loader, unlabel_loader)):

        l_aug, labels = l_data
        ul_w_aug, ul_s_aug, _ = ul_data

        params, features = param_update(
            cfg, i, model, teacher_model, optimizer, ssl_alg,
            consistency, l_aug.to(device), ul_w_aug.to(device),
            ul_s_aug.to(device), labels.to(device),
            average_model
        )

        batchSize = len(labels)
        for j in range(batchSize):
            labelIdx = labels[j].item()
            if(labelIdx in feature_vectors_mapping.keys()):
                feature_vectors_mapping[labelIdx][0] += features
                feature_vectors_mapping[labelIdx][1] += 1
            else:
                feature_vectors_mapping[labelIdx] = [features,1]
            # moving average for reporting losses and accuracy
        metric_meter.add(params, ignores=["coef"])

            # display losses every cfg.disp iterations
        if ((i+1) % cfg.disp) == 0:
            state = metric_meter.state(
                    header = f'[{i+1}/{cfg.iteration}]',
                    footer = f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                )
            logger.info(state)

        lr_scheduler.step()
        # validation
        if ((i + 1) % cfg.checkpoint) == 0 or (i + 1) == cfg.iteration:
            with torch.no_grad():
                if cfg.weight_average:
                    eval_model = average_model
                else:
                    eval_model = model
                logger.info("validation")
                mean_raw_acc, mean_val_acc, mean_val_loss = evaluation(model, eval_model, val_loader, device)
                logger.info("validation loss %f | validation acc. %f | raw acc. %f", mean_val_loss, mean_val_acc,
                                mean_raw_acc)

                # test
                if not cfg.only_validation and mean_val_acc > maximum_val_acc:
                    torch.save(eval_model.state_dict(), os.path.join(cfg.out_dir, "best_model.pth"))
                    maximum_val_acc = mean_val_acc
                    logger.info("test")
                    mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                    logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                mean_raw_acc)
                    logger.info("test accuracy %f", mean_test_acc)

            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
            torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))


if __name__ == "__main__":
    import os, sys
    from parser import get_args
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # setup logger
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    f_handler = logging.FileHandler(os.path.join(args.out_dir, "console.log"))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False

    logger.info(args)

    main(args, logger)





