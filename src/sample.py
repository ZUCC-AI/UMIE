import torch
import torch.nn as nn
import torch.nn.functional as F
class Downsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool2d(output_size)

    def downsample_inputs(self, inputs):
        # import pdb
        # pdb.set_trace()
        # print(inputs.shape)
        B, L, dim = inputs.shape

        inputs = inputs.permute(0, 2, 1) # (2B, dim, L/2)

        # restriction: L**0.5 must to be integer
        sqrt_L = int(L ** 0.5)

        inputs = inputs.reshape(B, dim, sqrt_L, sqrt_L)

        inputs = self.pool(inputs) # (B, dim, self.output_size[0], self.output_size[1])
        inputs = inputs.reshape(B, dim, -1)

        inputs = inputs.permute(0, 2, 1) # (2B, self.output_size[0]**2, dim)

        return inputs

    def forward(self, inputs_tuple):
        # inputs (B, L, dim)
        # import pdb
        # pdb.set_trace()
        
        
        if len(inputs_tuple) == 4: # (NLVR)
            inputs, boxes, img_order_ids, obj_order_ids = inputs_tuple

            inputs = torch.cat(torch.chunk(inputs, 2, 1), 0) # (2B, L/2, dim)
            inputs = self.downsample_inputs(inputs)
            inputs = torch.cat(torch.chunk(inputs, 2, 0), 1) # (B, L, dim)

            boxes = torch.cat(torch.chunk(boxes, 2, 1), 0)
            boxes = boxes[:, :inputs.shape[1]//2]
            boxes = torch.cat(torch.chunk(boxes, 2, 0), 1)

            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 1), 0)
            img_order_ids = img_order_ids[:, :inputs.shape[1]//2]
            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 0), 1)

            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 1), 0)
            obj_order_ids = obj_order_ids[:, :inputs.shape[1]//2]
            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 0), 1)

            outputs_tuple = (inputs, boxes, img_order_ids, obj_order_ids)
        else:
            inputs, boxes = inputs_tuple
            
            inputs = self.downsample_inputs(inputs)
            boxes = boxes[:, :inputs.shape[1]] # Get the first few data because the element are all zeros

            outputs_tuple = (inputs, boxes)

        return outputs_tuple


class OneDDownsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool1d(output_size)

    def downsample_inputs(self, inputs):
        B, L, dim = inputs.shape

        inputs = inputs.permute(0, 2, 1) # (2B, dim, L/2)

        inputs = self.pool(inputs) # (B, dim, self.output_size[0], self.output_size[1])
        inputs = inputs.reshape(B, dim, -1)

        inputs = inputs.permute(0, 2, 1) # (2B, self.output_size[0]**2, dim)

        return inputs

    def forward(self, inputs_tuple):
        # inputs (B, L, dim)
        # import pdb
        # pdb.set_trace()
        
        if len(inputs_tuple) == 4: # (NLVR)
            inputs, boxes, img_order_ids, obj_order_ids = inputs_tuple

            inputs = torch.cat(torch.chunk(inputs, 2, 1), 0) # (2B, L/2, dim)
            inputs = self.downsample_inputs(inputs)
            inputs = torch.cat(torch.chunk(inputs, 2, 0), 1) # (B, L, dim)

            boxes = torch.cat(torch.chunk(boxes, 2, 1), 0)
            boxes = boxes[:, :inputs.shape[1]//2]
            boxes = torch.cat(torch.chunk(boxes, 2, 0), 1)

            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 1), 0)
            img_order_ids = img_order_ids[:, :inputs.shape[1]//2]
            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 0), 1)

            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 1), 0)
            obj_order_ids = obj_order_ids[:, :inputs.shape[1]//2]
            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 0), 1)

            outputs_tuple = (inputs, boxes, img_order_ids, obj_order_ids)
        else:
            inputs, boxes = inputs_tuple
            
            inputs = self.downsample_inputs(inputs)
            boxes = boxes[:, :inputs.shape[1]] # Get the first few data because the element are all zeros

            outputs_tuple = (inputs, boxes)

        return outputs_tuple


class SparseSample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size

    def forward(self, inputs):
        if self.training:
            B, L, _ = inputs.shape

            x = torch.rand(B, L)

            indices = torch.argsort(torch.rand(*x.shape), dim=-1)

            indices = indices[:, :self.output_size]

            indices = torch.sort(indices)[0]

            return inputs[torch.arange(B).unsqueeze(1), indices]
        else:
            return inputs