from captum.attr import Saliency, IntegratedGradients, GuidedGradCam, GuidedBackprop, InputXGradient
import torch


TEST_MODEL = None
def build_maps(X, graphs, y, subject_idx, map_kind, test_model):
    global TEST_MODEL
    train_X_ = torch.tensor(X[subject_idx]).float()
    train_X_= train_X_.reshape(1, train_X_.shape[0], train_X_.shape[1], train_X_.shape[2])
    attr = torch.tensor(graphs[subject_idx]).float()
    attr = attr.reshape(1, attr.shape[0], attr.shape[1], attr.shape[2])
    out = test_model(train_X_, attr)
    y = int(y[subject_idx][0])

    if map_kind == "node":
        TEST_MODEL = test_model
        train_X_.requires_grad=True
        attr.requires_grad=False
        sal = Saliency(model_forward1)
        mask = sal.attribute(train_X_, target=y, additional_forward_args=(attr))

    elif map_kind == "edge":
        TEST_MODEL = test_model
        train_X_.requires_grad=False
        attr.requires_grad=True
        sal = Saliency(model_forward2)
        mask = sal.attribute(attr, target=y, additional_forward_args=(train_X_))
    
    mask = mask.squeeze().detach().numpy()
    
    return mask

def model_forward1(X, attr):
    out = TEST_MODEL(X, attr)
    return out  

def model_forward2(attr, X): 
    out = TEST_MODEL(X, attr)
    return out