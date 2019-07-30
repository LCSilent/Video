from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable
import torch
from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()

    with torch.no_grad():
        inputs = to_torch(inputs)
        inputs = Variable(inputs.cuda())
        # print(inputs.shape)
        if modules is None:

            # b = inputs.shape[1]
            # if b < 4:
            #     outputs = model(inputs)
            # else:
            #     features = torch.cuda.FloatTensor()
            #     output_list = []
            #     iternum = b // 4
            #     modnum = b % 4
            #     for i in range(iternum):
            #         features = torch.cat((features,model(inputs[:,i*4:i*4+4])),0)
            #     #     output_list.append(model(inputs[:,i*32:i*32+32]))
            #     # output_list.append(model(inputs[:,b-32:b]))
            #     if modnum != 0:
            #         features = torch.cat((features, model(inputs[:,b-4:b])), 0)
            #     outputs = torch.mean(features,dim=0,keepdim=True)
            # print(outputs.shape)
            outputs = model(inputs)
            outputs = outputs.data.cpu()
            return outputs
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None
            def func(m, i, o): outputs[id(m)] = o.data.cpu()
            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
        return list(outputs.values())
