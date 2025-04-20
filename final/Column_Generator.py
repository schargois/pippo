from Blocks import *
from ProgNet import ProgColumnGenerator
import torch.nn.functional as F



# we define a class that generates an LSTM based columns for us
class Column_generator_LSTM(ProgColumnGenerator):
    def __init__(self,input_size,hidden_size,num_of_classes,num_LSTM_layer,num_dens_Layer,dropout = 0.2):
        self.ids = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_of_classes = num_of_classes
        self.num_LSTM_layer = num_LSTM_layer
        self.num_dens_Layer = num_dens_Layer
        self.dropout = dropout
    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

    def create_column(self,device):
        columns = []
        columns.append(ProgLSTMBlock(inSize=self.input_size, outSize=self.hidden_size, lateralsType='LSTM',
                                     numLaterals=0, drop_out=self.dropout))
        if self.num_LSTM_layer == 2:
            columns.append(ProgLSTMBlock(inSize=self.hidden_size, outSize=self.hidden_size, lateralsType='LSTM',
                                         numLaterals=0, drop_out=self.dropout))

        activation = F.relu
        if self.num_dens_Layer == 0:
            columns.append(ProgDenseBlock(inSize=self.hidden_size, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = self.dropout))
        elif self.num_dens_Layer == 1:  # adding an extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize=self.hidden_size, outSize=32
                                          , numLaterals=0,drop_out = self.dropout))
            columns.append(ProgDenseBlock(inSize=32, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = self.ropout))
        elif self.num_dens_Layer == 2:  # adding two extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize=self.hidden_size, outSize=64
                                          , numLaterals=0,drop_out = self.dropout))
            columns.append(ProgDenseBlock(inSize=64, outSize=32
                                          , activation=activation, numLaterals=0,drop_out = self.dropout))
            columns.append(ProgDenseBlock(inSize=32, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = self.dropout))

        return ProgColumn(self.__genID(), columns, parentCols=[]).to(device)
    def generateColumn(self,device,parent_cols): #generates column with its parents connections
        new_column = self.create_column(device = device)
        # setting connections to previous columns
        for i in range(1,len(new_column.blocks)):
            for j in range(len(parent_cols)):
                new_column.blocks[i].laterals.append(nn.Linear(new_column.blocks[i].inSize, new_column.blocks[i].outSize))
        new_column.freeze(unfreeze = True)
        return new_column