import os, cv2
import torch, torchvision
import torch.nn as nn
from models.predrnn import RNN, Standard_RNN
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ----------------------SpatioTemporal LSTM ------------------------------------------

#   Architecture: PredRNN with combined Static-Semantic objects black and white images
class Network_combinedStaticSemantic(nn.Module):
    def __init__(self, args):
        super(Network_combinedStaticSemantic, self).__init__()
        self.args = args
        self.RNN = RNN(args)
        
    def forward(self, seq_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = seq_tensor.shape 
        seq_tensor =  seq_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )
        next_frames = self.RNN(seq_tensor, numiterations)
        return next_frames 

#   Architecture: PredRNN with separate Static and Semantic objects black and white images
class Network_static_Semantic(nn.Module):
    def __init__(self, args):
        super(Network_static_Semantic, self).__init__()
        self.args = args
        self.RNN_static = RNN(args)
        self.RNN_Semantic = RNN(args)
        
    def forward(self, staticgrid_tensor, Semanticgrid_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        # static grid 
        batch, seqlen, imght, imgwd, imgch = staticgrid_tensor.shape 
        staticgrid_tensor =  staticgrid_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )

        next_frames_static = self.RNN_static(staticgrid_tensor, numiterations)

        # Semantic grid 
        batch, seqlen, imght, imgwd, imgch = Semanticgrid_tensor.shape 
        Semanticgrid_tensor =  Semanticgrid_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )

        next_frames_Semantic = self.RNN_Semantic(Semanticgrid_tensor, numiterations)

        return next_frames_static, next_frames_Semantic 

#   Architecture: PredRNN with separate Static and Full scenes as input and predicting separate Static and Semantic objects black and white images
class Network_static_full(nn.Module):
    def __init__(self, args):
        super(Network_static_full, self).__init__()
        self.args = args
        self.RNN_static = RNN(args)
        self.RNN_full = RNN(args)
        
    def forward(self, staticgrid_tensor, fullgrid_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = staticgrid_tensor.shape 
        staticgrid_tensor =  staticgrid_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )

        next_frames_static = self.RNN_static(staticgrid_tensor, numiterations)

        # full grid 
        batch, seqlen, imght, imgwd, imgch = fullgrid_tensor.shape 
        fullgrid_tensor =  fullgrid_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )

        next_frames_full = self.RNN_full(fullgrid_tensor, numiterations)

        return next_frames_static, next_frames_full 


# ------------------- STANDARD ConvLSTM --------------------------------------------------------------

#   Architecture: Standard RNN with separate Static and Full scenes as input and predicting separate Static and Semantic objects black and white images
class Network_standard_static_full(nn.Module):
    def __init__(self, args):
        super(Network_standard_static_full, self).__init__()
        self.args = args
        self.RNN_static = Standard_RNN(args)
        self.RNN_full = Standard_RNN(args)
        
    def forward(self, staticgrid_tensor, fullgrid_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = staticgrid_tensor.shape 
        staticgrid_tensor =  staticgrid_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )

        next_frames_static = self.RNN_static(staticgrid_tensor, numiterations)

        # full grid 
        batch, seqlen, imght, imgwd, imgch = fullgrid_tensor.shape 
        fullgrid_tensor =  fullgrid_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )
        next_frames_full = self.RNN_full(fullgrid_tensor, numiterations)

        return next_frames_static, next_frames_full 

#   Architecture: Standard RNN with combined Static-Semantic objects black and white images
class Network_ConvLSTM_combinedStaticSemantic(nn.Module):
    def __init__(self, args):
        super(Network_ConvLSTM_combinedStaticSemantic, self).__init__()
        self.args = args
        self.RNN = Standard_RNN(args)
        
    def forward(self, seq_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = seq_tensor.shape 
        seq_tensor =  seq_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd )
        next_frames = self.RNN(seq_tensor, numiterations)
        return next_frames 