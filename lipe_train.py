import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
from rdkit import Chem
import torch.nn.functional as F
from itertools import chain
import matplotlib.pyplot as plt

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

WORD_KEY = ['#', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'y']

def manual_seed(seed):
    np.random.seed(seed) #1
    random.seed(seed) #2
    torch.manual_seed(seed) #3
    torch.cuda.manual_seed(seed) #4.1
    torch.cuda.manual_seed_all(seed) #4.2
    torch.backends.cudnn.benchmark = False #5 
    torch.backends.cudnn.deterministic = True #6
    

#######################################SMILES Tokenizer#######################################
class SMILES_Tokenizer():
    def __init__(self, max_length):
        self.txt2idx = {}
        self.idx2txt = {}
        self.max_length = max_length
    
    def fit(self, SMILES_list): # SMILES_list에서 모든
        unique_char = set()
        for smiles in SMILES_list:
            for char in smiles:
                unique_char.add(char)
        unique_char = sorted(list(unique_char))
        for i, char in enumerate(unique_char): # 0: pad, 1: start,  from 2: ~ , voca_size+1 : end
            self.txt2idx[char]=i+2
            self.idx2txt[i+2]=char
            
        self.voca_size = len(unique_char)
        
    def txt2seq(self, texts):
        seqs = []
        for text in texts:
            seq = [0]*self.max_length
            seq[0] = 1
            for i, t in enumerate(text):
                if i == self.max_length-1:
                    break
                try:
                    seq[i+1] = self.txt2idx[t]
                except:
                    seq[i+1] = 0
            seq[i+2] = 2 + self.voca_size
            seqs.append(seq)
        return np.array(seqs)
    
    def seq2txt(self, seqs):
        return ''.join([self.idx2txt.get(idx, '') for idx in seqs if idx != 0])
        

###################################SMILES DATASET##################################################
class SMILES_dataset(Dataset):
    
    TRANSFROM = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ##  mean, std
    ])
    
    def __init__(self, data_path, tokenizer, train=True, max_length = 167, feature=False):
        
        self.train = train
        if train:
            file_name = 'train_100.csv'
        else:
            file_name = 'val_100.csv'
    
        data = pd.read_csv(os.path.join(data_path, file_name))
        data = data.dropna(subset=['pIC50','logP'])
        self.tokenizer = tokenizer
        
        
        self.feature = feature
        if feature:
            self.imgs = np.array([(data_path+'/feature/'+f) for f in data['file_name']])
            for i in range(len(self.imgs)):
                self.imgs[i] = self.imgs[i].replace('.png', '.pt')
        else:    
            self.imgs = np.array([(data_path+'/feature/'+f) for f in data['file_name']])
        self.smiles = self.tokenizer.txt2seq(data.SMILES)
        self.labels = data[['pIC50', 'logP']].to_numpy()
        # self.pICs = data[['pIC50']].to_numpy()
        # self.logPs = data[['logP']].to_numpy()
        
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        if self.feature:
            img = torch.load(self.imgs[i], map_location='cpu')
        else:
            img = Image.open(self.imgs[i])
            img = self.TRANSFROM(img)
        
        return {
            'img' : img,
            'smile' : torch.tensor(self.smiles[i], dtype=torch.long),
            'label' : torch.tensor(self.labels[i], dtype=torch.float32)
            # 'pIC' : torch.tensor(self.pICs[i], dtype=torch.float32),
            # 'logP' : torch.tensor(self.logPs[i], dtype=torch.float32),
        }

###################################MODEL######################################################

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width,kernel_size=3,stride= stride, padding=dilation , groups=groups, dilation = dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class feature_extract(nn.Module):
    def __init__(self,dim=2048,in_size=7,out_size=10, rate=0.3):
        super(feature_extract, self).__init__()
        self.conbvn = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout2d(rate),
            nn.Conv2d(dim,dim,3,1,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.dilation = 1
        self.inplanes = 2048
        self.groups = 1
        self.base_width = 64
        
        self.conbvn = self._make_layer(Bottleneck, 512, 3, stride=1, dilate=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((10,10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self,x):
        add_x = x
        
        x = add_x + self.conbvn(x)
        # x = self.avgpool(x)
        return x


# CNN Encoder Module
class CNN_Encoder(nn.Module):
    def __init__(self, encoder_dim, rate, feature=False, max_len=102):
        super(CNN_Encoder, self).__init__()
        self.feature = feature
        if feature:
            self.backbone = feature_extract()
        else:
            model = torchvision.models.resnet101(pretrained=MODEL_PRETRAIN) # UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
            modules = list(model.children())[:-2]
            self.backbone = nn.Sequential(*modules)
        
        self.dropout1 = nn.Dropout(rate)
        self.fc = nn.Linear(2048, encoder_dim)
        
    def forward(self, x):
        x = self.backbone(x) # [batch, 2048, 7, 7]
        x = x.view(x.size(0), 2048, -1) # [64, 2048, 49]
        x = x.permute(0,2,1)
        # x = x.mean(dim=1) # [batch, 2048, 1]
        x = self.fc(x) # [batch, 2048, embedding_dim]
        x = self.dropout1(x)
        return x 
    
    
class Attention(nn.Module):
    """
    Attention network for calculate attention value
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, encoder_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        
        score = torch.relu(att1+att2.unsqueeze(1))
        
        att = F.softmax(self.full_att(score).squeeze(-1), dim=1) # (batch, pixel size)
        
        context_vector = att.unsqueeze(-1) * encoder_out # (batch, pixel, encoder_dim)
        context_vector = context_vector.sum(dim=1) # (batch, encoder_dim)

        return context_vector, att

# RNN Decoder Module
class RNN_Decoder(nn.Module):
    """_summary_
    LSTM input : ( N, L, H ) in when batch_first=True
        output : ( N, L, D*Hout ) when batch_first=True
    N=BATCH
    L=MAX_LENGTH
    Hout = voca_size
    
    """
    def __init__(self, voca_size, embedding_dim, max_len, num_layers, rate, encoder_dim=2048):
        super(RNN_Decoder, self).__init__()
        self.embedding = nn.Embedding(voca_size, embedding_dim)
        self.max_len = max_len
        self.voca_size = voca_size
        self.lstm = nn.LSTM(embedding_dim+encoder_dim, embedding_dim, num_layers, batch_first=True,dropout=rate, bidirectional=False)
        
        self.init_h = nn.Linear(encoder_dim, embedding_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, embedding_dim)  # linear layer to find initial cell state of LSTMCell
        
        self.dropout = nn.Dropout(rate)
        
        self.emb2voca = nn.Linear(embedding_dim, voca_size) # linear layer to voca size from embedding_dim
        
        self.attention = Attention(encoder_dim, embedding_dim, embedding_dim)
        

    def forward(self, enc_out, dec_in, epoch, training=True): # [batch, embedding], [batch, max_len]
        hidden = self.init_h(enc_out.mean(dim=1)) # [batch, embedding_dim]
        cell_state = self.init_c(enc_out.mean(dim=1)) # [batch, embedding_dim]
        
        embedding = self.embedding(dec_in) # [batch, max_len, embedding_dim]
        input = embedding[:,0,:]
        
        out_list = []
        atts = []
        for i in range(self.max_len):
                
            context_vector, att = self.attention(enc_out, hidden) # [batch, embedding_dim] # [batch, pixel size]
            # input = context_vector.unsqueeze(1) + input
            # context_vector: [batch, encoder_dim], input: [batch, embedding_dim]
            input = torch.cat([context_vector, input], dim=1) # [batch, embedding_dim+embedding_dim]
            output, (hidden, cell_state) = self.lstm(input.unsqueeze(1), (hidden.unsqueeze(0),cell_state.unsqueeze(0))) # hidden in: (batch, dim)
            hidden = hidden.squeeze(0)
            cell_state = cell_state.squeeze(0)
            
            output = self.dropout(output)
            output = self.emb2voca(output) # hiiden [batch, 1, voca_size]
            #input = self.embedding(output.argmax(dim=-1))
            
            if training == True:
                # next input 
                if random.random() > 0.5:
                    input = self.embedding(torch.argmax(output.squeeze(1), -1))
                else:
                    input = embedding[:,i,:]
            else:
                input = self.embedding(torch.argmax(output.squeeze(1), -1))

            out_list.append(output.squeeze(1))
            atts.append(att)
            
        outputs = torch.stack(out_list,dim=1)
        atts = torch.stack(atts,dim=1)
        return outputs, atts
        

class LinearRegression(nn.Module):
    def __init__(self,enc_dim = 512, max_len=102, pixel_size=49, hidden_dim=256, out_dim = 2):
        super(LinearRegression, self).__init__()
        import math
        self.device = device
        self.avg_size = math.sqrt(pixel_size)
        assert self.avg_size.is_integer() , "mismatch pixel_size in LinearRegression Model" 
        self.avg_size = int(self.avg_size)
        
        self.enc_dim = enc_dim
        self.max_len = max_len
        
        self.pooling = nn.AvgPool2d((self.avg_size, self.avg_size))
        
        self.fc1 = nn.Linear(enc_dim+ max_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, out_dim)
    
        self.apply(self.initialize_weights)
    
    def initialize_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        
    
    def forward(self, enc_out, att):
        # enc_out (batch, pixel, enc_dim)
        # attention (batch, maxlen, pixel*pixel)
        x = torch.cat([enc_out.permute(0,2,1), att], dim=1)
        
        # x (batch, enc_dim + maxlen, pixel, pixel)
        x = x.view(-1, self.enc_dim+ self.max_len , self.avg_size, self.avg_size)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        
        # x (batch, enc_dim+maxlen)
        x = F.relu(nn.Dropout()(self.fc1(x)))
        x = F.relu(nn.Dropout()(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
def is_smiles(sequence):
    """
    check the sequence matches with the SMILES format
    :param sequence: decoded sequence
    :return: True or False
    """
    m = Chem.MolFromSmiles(sequence)
    return False if m == None else True

if __name__ == "__main__":
    
    manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    BATCH_SIZE = 64
    MODEL_PRETRAIN = True
    EPOCHS = 50
    NUM_LAYERS = 1
    dropout_rate = 0.3
    EMBEDDING_DIM = 512
    ENCODER_DIM = 512
    LR = 1e-4
    vision_pretrain = True
    DATA_PATH = "./kaggle_data"
    #DATA_PATH = "./kaggle_data"
    TRAIN_CSV_PATH = os.path.join(DATA_PATH, "train_100.csv")
    MAX_LEN = 102
    SAVE_PATH = f'./models/best_model_coin_teacher_05_22.pt'
    
    FEATURE = True
    
    
    ### make tokenizer
    train_pd = pd.read_csv(TRAIN_CSV_PATH)
    tokenizer = SMILES_Tokenizer(MAX_LEN)
    tokenizer.fit(WORD_KEY)
    # tokenizer.fit(train_pd.SMILES)
    print(tokenizer.txt2idx.keys())
    voca_size = len(tokenizer.idx2txt)+3
    print(f"voca_size : {voca_size}")
    
    # dataset
    train_dataset = SMILES_dataset(DATA_PATH, tokenizer, train=True, max_length=MAX_LEN, feature=FEATURE)
    print(f"Train data size : {len(train_dataset)}")
    val_dataset = SMILES_dataset(DATA_PATH, tokenizer, train=False, max_length=MAX_LEN, feature=FEATURE)
    print(f"Validation data size : {len(val_dataset)}")
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle=False, drop_last=True)
    
    # model
    encoder = CNN_Encoder(encoder_dim=ENCODER_DIM, rate=dropout_rate, feature=FEATURE, max_len=MAX_LEN)
    decoder = RNN_Decoder(voca_size=voca_size, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN, num_layers=NUM_LAYERS, rate=dropout_rate, encoder_dim=ENCODER_DIM)
    load_state_dict = torch.load('checkpoint/best.pth', map_location='cpu')
    
    encoder.load_state_dict(load_state_dict['encoder_state_dict'])
    decoder.load_state_dict(load_state_dict['decoder_state_dict'])
    
    # freeze
    # for p in encoder.parameters():
    #     p.requires_grad = False
        
    # for p in decoder.parameters():
        # p.requires_grad = False
    
    regression_model = LinearRegression(enc_dim= ENCODER_DIM, pixel_size=49, hidden_dim=EMBEDDING_DIM)
    
    
    
    # # optimizer and loss        
    # optim_group = []
    combined_parameters = chain(encoder.parameters(), decoder.parameters())
    optimizer = torch.optim.Adam(combined_parameters, lr=LR*0.1)
                
    optimizer.add_param_group({'params':regression_model.parameters(), 'lr':LR})
    # optimizer = torch.optim.Adam(regression_model.parameters(), lr = LR*0.1)
    smiles_criterion = nn.CrossEntropyLoss()
    labels_criterion = nn.MSELoss()
    
    # train
    encoder.to(device)
    decoder.to(device)
    regression_model.to(device)
    
    best_loss = np.inf
    train_epoch_loss = []
    train_step_loss = []
    save_train_acc = []
    save_val_acc = []
    save_lipe_acc = []
    save_logp_acc = []
    save_pic_acc = []
    for epoch in tqdm(range(EPOCHS), desc="Epoch", position=0, leave=False, total=EPOCHS):
        ## train
        encoder.train()
        decoder.train()
        
        smiles_loss = 0.
        train_count = 0
        epoch_loss = 0.
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train step", position=1, leave=False)
        for step, batch in progress_bar:
            imgs = batch['img'].to(device)
                
            smiles = batch['smile'].to(device)
            labels = batch['label'].to(device)
            
            dec_in = smiles
            
            # image to text
            enc_out = encoder(imgs)
            smiles_outs, alphas  = decoder(enc_out, dec_in, epoch, training=True)
            
            # image and text attention + smiles_outs to pIC, logP
            preds_smiles = torch.argmax(smiles_outs, dim=-1)
            
            # preds_labels = regression_model(enc_out, smiles, alphas)
            preds_labels = regression_model(enc_out, alphas)
            
            smiles_loss = smiles_criterion(smiles_outs.view(-1, voca_size), smiles.view(-1))
            
            labels_loss = labels_criterion(labels, preds_labels)
            
            loss = smiles_loss + labels_loss
            # Add doubly stochastic attention regularization
            # loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
            epoch_loss += loss.detach().item()
            train_step_loss.append(loss.detach().item())
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            for i in range(BATCH_SIZE):
                if torch.equal(preds_smiles[i], smiles[i]):
                    train_count +=1
                        
            progress_bar.set_postfix(
                {
                "loss": f'{loss.detach().item():.4f}', 
                "smiles": f'{smiles_loss.detach().item():.4f}', 
                "labels": f'{labels_loss.detach().item():.4f}', 
                }
            )
            
        train_epoch_loss.append(epoch_loss/len(train_loader))  
        train_acc = train_count / len(train_dataset) * 100
        save_train_acc.append(train_acc)
        print("\n")
        print(f"Train Smiles acc : {train_acc:.2f}% , {train_count}")
        ## evalutate
        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            # val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Val step", position=1, leave=False)
            smiles_count = 0
            total_loss = 0.
            lipe_count = 0
            pIC_count = 0
            logP_count = 0
            for batch in tqdm(val_loader, leave=False):
                imgs = batch['img'].to(device)
                smiles = batch['smile'].to(device)
                labels = batch['label'].to(device)
                
                dec_in = torch.ones((BATCH_SIZE,MAX_LEN), dtype=torch.long).to(device)
                
                enc_out = encoder(imgs)
                smiles_outs, alphas  = decoder(enc_out, dec_in, epoch, training=False)
                preds_smiles = torch.argmax(smiles_outs, dim=-1)
                
                preds_labels = regression_model(enc_out, alphas)
                
                smiles_loss = smiles_criterion(smiles_outs.view(-1, voca_size), smiles.view(-1))
                labels_loss = labels_criterion(labels, preds_labels)
                
                loss = smiles_loss + labels_loss
                # loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
                
                total_loss+=loss.detach().item()
                
                for i in range(BATCH_SIZE):
                    if torch.equal(preds_smiles[i], smiles[i]):
                        smiles_count+=1
                
                Lipe = (labels[:,0] - labels[:,1]) > 6
                pred_Lipe = (preds_labels[:,0] - preds_labels[:,1]) > 6 
                
                lipe_count += (Lipe == pred_Lipe).sum().item()
            
                pIC = (labels[:,0]) > 8
                pred_pIC = preds_labels[:,0] > 8
                
                pIC_count += (pIC == pred_pIC).sum().item()
                
                logP = abs(labels[:,1] - preds_labels[:,1]) < 3
                logP_count += logP.sum().item()
                
            smiles_acc = smiles_count / len(val_dataset) * 100
            lipe_acc = lipe_count / len(val_dataset) * 100
            pIC_acc = pIC_count / len(val_dataset) * 100
            logP_acc = logP_count / len(val_dataset) * 100
            
            
            save_val_acc.append(smiles_acc)
            save_lipe_acc.append(lipe_acc)
            save_pic_acc.append(pIC_acc)
            save_logp_acc.append(logP_acc)
           
            
            print(f"Validation Smiles acc : {smiles_acc:.2f}% , lipe acc : {lipe_acc:.2f}%, pIC acc : {pIC_acc:.2f}%, logP acc : {logP_acc:.2f}")
            # best model save
            if not os.path.exists('./checkpoint'):
                os.mkdir('./checkpoint')
            if best_loss > total_loss:
                best_loss = total_loss
                save_checkpoint = {
                        'encoder' : encoder,
                        'encoder_state_dict' : encoder.state_dict(),
                        'decoder' : decoder,
                        'decoder_state_dict' : decoder.state_dict(),
                        'linear' : regression_model,
                        'linear_state_dict' : regression_model.state_dict(),
                    }
                torch.save(save_checkpoint, f'./checkpoint/best_lipe.pth')
                print(f"save best model.")
            
    # last model save
    save_checkpoint = {
            'encoder' : encoder,
            'encoder_state_dict' : encoder.state_dict(),
            'decoder' : decoder,
            'decoder_state_dict' : decoder.state_dict(),
            'linear' : regression_model,
            'linear_state_dict' : regression_model.state_dict(),
        }
    torch.save(save_checkpoint, f'./checkpoint/last_lipe.pth')
    print(f"save last model.")  
    
    save_data = {
        'epoch' : list(range(EPOCHS)),
        'train_acc' : save_train_acc,
        'val_acc' : save_val_acc,
        'train_loss' : train_epoch_loss,
        'val_lipe_acc' : save_lipe_acc,
        'val_pIC_acc' : save_pic_acc,
        'val_logP_acc' : save_logp_acc,
    }
    
    df = pd.DataFrame(save_data)
    df.to_csv("result/save_lipe.csv")
    
    # with open('result/save_lipe.csv', 'a') as file:
    #     for i in range(len(save_train_acc)):
    #         file.write(f'{i},{save_train_acc[i]},{save_val_acc[i]},{train_epoch_loss[i]},{save_lipe_acc},{save_pic_acc},{save_logp_acc}\n')