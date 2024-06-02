import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F

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
        # self.conbvn = nn.Sequential(
        #     nn.Conv2d(dim,dim,3,1,1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(),
        #     nn.Dropout2d(rate),
        #     nn.Conv2d(dim,dim,3,1,1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(),
        # )
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
class CNNEncoder(nn.Module):
    def __init__(self, encoder_dim, rate, feature=False, max_len=102):
        super(CNNEncoder, self).__init__()
        self.feature = feature
        if feature:
            print("Using Wide-ResNet101 feature")
            self.backbone = feature_extract()
        else:
            print("Using ResNet101 model for training")
            model = torchvision.models.resnet101(pretrained=True) # UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
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

class RNNDecoderWithATT(nn.Module):
    """_summary_
    LSTM input : ( N, L, H ) in when batch_first=True
        output : ( N, L, D*Hout ) when batch_first=True
    N=BATCH
    L=MAX_LENGTH
    Hout = voca_size
    
    """
    def __init__(self, voca_size, embedding_dim, max_len, num_layers, rate, encoder_dim=2048, use_teacher=False, use_coin=False):
        super(RNNDecoderWithATT, self).__init__()
        self.embedding = nn.Embedding(voca_size, embedding_dim)
        self.max_len = max_len
        self.voca_size = voca_size
        self.lstm = nn.LSTM(embedding_dim+encoder_dim, embedding_dim, num_layers, batch_first=True,dropout=rate, bidirectional=False)
        
        self.init_h = nn.Linear(encoder_dim, embedding_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, embedding_dim)  # linear layer to find initial cell state of LSTMCell
        
        self.dropout = nn.Dropout(rate)
        
        self.emb2voca = nn.Linear(embedding_dim, voca_size) # linear layer to voca size from embedding_dim
        
        self.attention = Attention(encoder_dim, embedding_dim, embedding_dim)
        
        self.use_teacher = use_teacher
        self.use_coin = use_coin

    def forward(self, enc_out, dec_in, training=True): # [batch, embedding], [batch, max_len]
        hidden = self.init_h(enc_out.mean(dim=1)) # [batch, embedding_dim]
        cell_state = self.init_c(enc_out.mean(dim=1)) # [batch, embedding_dim]
        
        embedding = self.embedding(dec_in) # [batch, max_len, embedding_dim]
        input = embedding[:,0,:]
        
        out_list = []
        atts = []
        for i in range(self.max_len):
            context_vector, att = self.attention(enc_out, hidden) # [batch, embedding_dim] # [batch, pixel size]
                
            input = torch.cat([context_vector, input], dim=1) # [batch,  encoder+dim+embedding_dim]
            
            output, (hidden, cell_state) = self.lstm(input.unsqueeze(1), (hidden.unsqueeze(0),cell_state.unsqueeze(0))) # hidden in: (batch, dim)
            hidden = hidden.squeeze(0)
            cell_state = cell_state.squeeze(0)
            
            
            output = self.dropout(output)
            output = self.emb2voca(output) # hiiden [batch, 1, voca_size]

            if training == True:
                # next input 
                if self.use_teacher:
                    if self.use_coin and random.random() > 0.5:
                        input = self.embedding(torch.argmax(output.squeeze(1), -1))
                    else:
                        input = embedding[:,i,:]
                else:
                    input = self.embedding(torch.argmax(output.squeeze(1), -1))
                # input = embedding[:,i,:]
            else:
                input = self.embedding(torch.argmax(output.squeeze(1), -1))
                
                
            out_list.append(output.squeeze(1))
            atts.append(att)
            
        outputs = torch.stack(out_list,dim=1)
        atts = torch.stack(atts,dim=1)
        return outputs, att
    
    
class RNNDecoder(nn.Module):
    """_summary_
    LSTM input : ( N, L, H ) in when batch_first=True
        output : ( N, L, D*Hout ) when batch_first=True
    N=BATCH
    L=MAX_LENGTH
    Hout = voca_size
    
    """
    def __init__(self, voca_size, embedding_dim, max_len, num_layers, rate, encoder_dim=2048, use_teacher=False, use_coin=False):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(voca_size, embedding_dim)
        self.max_len = max_len
        self.voca_size = voca_size
        self.lstm = nn.LSTM(embedding_dim+encoder_dim, embedding_dim, num_layers, batch_first=True,dropout=rate, bidirectional=False)
        
        self.init_h = nn.Linear(encoder_dim, embedding_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, embedding_dim)  # linear layer to find initial cell state of LSTMCell
        
        self.dropout = nn.Dropout(rate)
        
        self.emb2voca = nn.Linear(embedding_dim, voca_size) # linear layer to voca size from embedding_dim
        
        self.attention = Attention(encoder_dim, embedding_dim, embedding_dim)
        
        self.use_teacher = use_teacher
        self.use_coin = use_coin

    def forward(self, enc_out, dec_in, training=True): # [batch, embedding], [batch, max_len]
        hidden = self.init_h(enc_out.mean(dim=1)) # [batch, embedding_dim]
        cell_state = self.init_c(enc_out.mean(dim=1)) # [batch, embedding_dim]
        
        embedding = self.embedding(dec_in) # [batch, max_len, embedding_dim]
        input = embedding[:,0,:]
        
        context_vector = enc_out.mean(dim=1)
        
        out_list = []
        for i in range(self.max_len):
                
            input = torch.cat([context_vector, input], dim=1) # [batch,  encoder+dim+embedding_dim]
            
            output, (hidden, cell_state) = self.lstm(input.unsqueeze(1), (hidden.unsqueeze(0),cell_state.unsqueeze(0))) # hidden in: (batch, dim)
            hidden = hidden.squeeze(0)
            cell_state = cell_state.squeeze(0)
            
            output = self.dropout(output)
            output = self.emb2voca(output) # hiiden [batch, 1, voca_size]

            if training == True:
                # next input 
                if self.use_teacher:
                    if self.use_coin and random.random() > 0.5:
                        input = self.embedding(torch.argmax(output.squeeze(1), -1))
                    else:
                        input = embedding[:,i,:]
                else:
                    input = self.embedding(torch.argmax(output.squeeze(1), -1))
                # input = embedding[:,i,:]
            else:
                input = self.embedding(torch.argmax(output.squeeze(1), -1))
                
                
            out_list.append(output.squeeze(1))
            
        outputs = torch.stack(out_list,dim=1)
        
        return outputs, torch.ones((len(outputs),1), device=outputs.device)
        