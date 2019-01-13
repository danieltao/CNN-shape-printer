import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def calculate_model_output(model, input_shape):
    """
    calculate the output shape of the model
    :param model: the pytorch model
    :param input_shape: the input image shape
    :return: the shape (4 dimension tensor)
    """
    dummy_tensor = generate_dummy(input_shape)
    with torch.no_grad():
        output = model(dummy_tensor)
        output = np.array(output)
        print("The output of the model is: ")
        print([None] + list(output.shape[1:]))
    return [None] + list(output.shape[1:])


def calculate_layer_output(model, end_layer, input_shape):
    """
    calculate the output shape of until the target layer
    :param model: pytorch model
    :param end_layer: the target layer name that one is aiming at
    :param input_shape: the input image shape (list of two)
    :return: output shape
    """
    x = generate_dummy(input_shape)
    keys, index = list(model.state_dict().keys()), 0
    with torch.no_grad():
        for layer in model.children():
            x = layer(x)
            if keys[index].split(".")[0] == end_layer:
                break
            index += 1
        print([None] + list(x.shape[1:]))
    return [None] + list(x.shape[1:])


def generate_dummy(input_shape):
    """
    generate a dummy input tensor according to the input shape
    :param input_shape: the shape of input
    :return: a 4 dimensional tensor
    """
    if len(input_shape) == 2:
        input_shape = (1, 3, input_shape[0], input_shape[1])
    elif type(input_shape) == int:
        input_shape = (1, 3, input_shape, input_shape)
    dummy_input = np.zeros(shape=input_shape)
    dummy_tensor = torch.from_numpy(dummy_input).float()
    return dummy_tensor


def generate_architect_image(model, input_shape=None):
    """
    Generate paper-quality image depicting the architecture of the model.
    :param model: the pytorch model
    :param input_shape: the shape of the input (if any)
    :return: None. A png file is saved.
    """
    


class Patch(nn.Module):
    def __init__(self):
        super(Patch, self).__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_pool = nn.MaxPool2d(kernel_size=2, stride=2)

 #       self.dropout_1 = nn.Dropout()
 #       self.dense_1 = nn.Linear(512, 5)


    def forward(self, x):
        out = F.relu(self.block1_conv1(x), inplace=True)
        out = F.relu(self.block1_conv2(out), inplace=True)
        out = self.block1_pool(out)
        out = F.relu(self.block2_conv1(out), inplace=True)
        out = F.relu(self.block2_conv2(out), inplace=True)
        out = self.block2_pool(out)
        out = F.relu(self.block3_conv1(out), inplace=True)
        out = F.relu(self.block3_conv2(out), inplace=True)
        out = F.relu(self.block3_conv3(out), inplace=True)
        out = self.block3_pool(out)
        out = F.relu(self.block4_conv1(out), inplace=True)
        out = F.relu(self.block4_conv2(out), inplace=True)
        out = F.relu(self.block4_conv3(out), inplace=True)
        out = self.block4_pool(out)
        out = F.relu(self.block5_conv1(out), inplace=True)
        out = F.relu(self.block5_conv2(out), inplace=True)
        out = F.relu(self.block5_conv3(out), inplace=True)
        x = self.block5_pool(out)
        return x


test = Patch()
my = calculate_layer_output(test, "block3_conv3", (1152, 896))
