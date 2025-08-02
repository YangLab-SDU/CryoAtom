import torch
from torch import nn
from einops.layers.torch import Rearrange
import math
import einops
from CryoAtom.CryNet.sequence_attention import SequenceAttention
from CryoAtom.CryNet.common_modules import Rope
class NodeAttention(nn.Module):
    def __init__(
            self,
            in_features:int,
            num_neighbours:int,
            attention_heads:int,
            attention_features:int
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.ifz = in_features
        self.kz = num_neighbours
        self.ahz = attention_heads
        self.afz = attention_features
        self.attention_scale = math.sqrt(self.afz)
        self.short_add = nn.Identity()
        self.q = nn.Sequential(
            nn.Linear(self.ifz,self.ahz * self.afz,bias=False),
            Rearrange("N (ahz afz) -> N ahz afz",ahz=self.ahz,afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.bias = nn.Sequential(nn.LayerNorm(in_features),
                                  nn.Linear(in_features,attention_heads,bias=False))
        self.gated = nn.Sequential(
            nn.Linear(self.ifz,self.ahz * self.afz,bias=True),
            Rearrange("N (ahz afz) -> N ahz afz",ahz=self.ahz,afz=self.afz),
            nn.Sigmoid()
        )
        self.back = nn.Sequential(
            Rearrange("N ahz afz -> N (ahz afz)",ahz=self.ahz,afz=self.afz),
            nn.Linear(self.ahz*self.afz,self.ifz)
        )
    def forward(self,x_1,x_2,pos_emb,edge_index):
        x_11 = x_1
        y_1 = x_11[edge_index] # N kz ifz
        query = self.q(x_11) # N ahz afz
        key = self.k(y_1) # N kz ahz afz
        query,key = Rope(query,key,pos_emb,edge_index)
        value = self.v(y_1) # N kz ahz afz
        bias_2 = self.bias(x_2)
        gate = self.gated(x_11)
        attention_scores = (torch.einsum('nai,nkai->nka',query,key)/self.attention_scale)+bias_2
        attention_weights = torch.softmax(attention_scores,dim=1)
        out = gate*torch.einsum('nka,nkai->nai',attention_weights,value)
        out = self.norm1(self.back(out) + math.sqrt(2)*self.short_add(x_1))
        return out
class Transition(nn.Module):
    def __init__(self,in_features:int,norm:nn.Module,n:int=3):
        super().__init__()
        self.norm = norm(in_features)
        self.w1 = nn.Linear(in_features,in_features*n,bias=False)
        self.w2 = nn.Linear(in_features,in_features*n,bias=False)
        self.w3 = nn.Linear(in_features*n,in_features,bias=False)
        self.short = nn.Identity()
    def forward(self,x):
        y = self.w3(nn.functional.silu(self.w1(x))*self.w2(x))
        y = self.norm(y + math.sqrt(2)*self.short(x))
        return y
class OutProductMean(nn.Module):
    def __init__(self,in_features:int,c:int = 32):
        super().__init__()
        self.line = nn.Linear(in_features,c*2)
        self.c = c
        self.line2 = nn.Linear(c**2,in_features)
    def forward(self,x_1,edge_index):
        y_1,y_2 = self.line(x_1).chunk(2,dim=-1)
        z_1 = y_2[edge_index]
        out = torch.einsum('ni,nkj->nkij',y_1,z_1)
        out = einops.rearrange(out,"N k i j -> N k (i j)",i=self.c,j=self.c)
        out = self.line2(out)
        return out

class EdgeAttention(nn.Module):
    def __init__(self, in_features: int, num_neighbours: int = 20, attention_heads: int = 4,
                 attention_features: int = 32):
        super().__init__()
        self.ifz = in_features
        self.kz = num_neighbours
        self.ahz = attention_heads
        self.afz = attention_features
        self.norm = nn.LayerNorm(in_features)
        self.attention_scale = math.sqrt(self.afz)
        self.short_add = nn.Identity()
        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.bia = nn.Linear(in_features, attention_heads, bias=False)
        self.gated = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=True),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz),
            nn.Sigmoid()
        )
        self.back = nn.Sequential(
            Rearrange("N kz ahz afz -> N kz (ahz afz)", ahz=self.ahz, afz=self.afz),
            nn.Linear(self.ahz * self.afz, self.ifz)
        )
    def forward(self,x_2,edge_index):
        y_2 = x_2
        with torch.no_grad():
            edge_temp = torch.ones((x_2.size(0),x_2.size(1)),dtype=torch.long)*torch.arange(len(x_2),dtype=torch.long).unsqueeze(1)
        query = self.q(y_2)  # N kz ahz afz
        key = self.k(y_2)  # N kz ahz afz
        value = self.v(y_2)  # N kz ahz afz
        key = torch.cat((key[edge_temp],key[edge_index]),dim=2) # N kz 2kz ahz afz
        value = torch.cat((value[edge_temp],value[edge_index]),dim=2) # N kz 2kz ahz afz
        bias = self.bia(y_2)
        bias = torch.cat((bias[edge_temp],bias[edge_index]),dim=2) # N kz 2kz ahz
        gate = self.gated(y_2)
        attention_scores = (torch.einsum('nkac,nkjac->nkja', query, key) / self.attention_scale) + bias
        attention_weights = torch.softmax(attention_scores, dim=2)
        out = gate * torch.einsum('nkja,nkjac->nkac', attention_weights, value)
        out = self.norm(self.back(out) + math.sqrt(2)*self.short_add(x_2))
        return out
class TraingleAttention(nn.Module):
    def __init__(self,in_features:int,num_neighbours:int=20,attention_heads:int=4,attention_features:int=32):
        super().__init__()
        self.ifz = in_features
        self.kz = num_neighbours
        self.ahz = attention_heads
        self.afz = attention_features
        self.norm = nn.LayerNorm(in_features)
        self.attention_scale = math.sqrt(self.afz)
        self.short_add = nn.Identity()
        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.bia = nn.Sequential(nn.Linear(in_features, self.kz*attention_heads, bias=False),
                                 Rearrange("N kz (nz ahz) -> N kz nz ahz",nz=self.kz,ahz=self.ahz))
        self.gated = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=True),
            Rearrange("N kz (ahz afz) -> N kz ahz afz", ahz=self.ahz, afz=self.afz),
            nn.Sigmoid()
        )
        self.back = nn.Sequential(
            Rearrange("N kz ahz afz -> N kz (ahz afz)", ahz=self.ahz, afz=self.afz),
            nn.Linear(self.ahz * self.afz, self.ifz)
        )
    def forward(self,x_2):
        y_2 = self.norm(x_2)
        query = self.q(y_2)  # N kz ahz afz
        key = self.k(y_2)  # N kz ahz afz
        value = self.v(y_2)  # N kz ahz afz
        bias = self.bia(y_2)
        gate = self.gated(y_2)
        attention_scores = (torch.einsum('niac,njac->nija', query, key) / self.attention_scale) + bias
        attention_weights = torch.softmax(attention_scores, dim=2)
        out = gate * torch.einsum('nija,njac->niac', attention_weights, value)
        out = self.back(out) + self.short_add(x_2)
        return out
class Cryformer(nn.Module):
    def __init__(
            self,
            in_features:int,
            attention_heads: int,
            sequence_features:int =1280,
            attention_features: int = 48,
            num_neighbours : int = 20,
            activation_class:nn.Module = nn.ReLU
    ):
        super().__init__()
        self.nodeattention = NodeAttention(in_features,num_neighbours,attention_heads,attention_features)
        self.sequence_attention = SequenceAttention(sequence_features=sequence_features,
                                                    in_features=in_features,
                                                    attention_features=attention_features,
                                                    attention_heads=attention_heads,
                                                    activation_class=activation_class,
                                                    checkpoint=False)
        self.transition1 = Transition(in_features,nn.LayerNorm)
        self.outer = OutProductMean(in_features)
        self.norm = nn.LayerNorm(in_features)
        self.edge_deliver = EdgeAttention(in_features,num_neighbours,attention_heads,attention_features)
        # self.traingle_attention = TraingleAttention(in_features,num_neighbours,attention_heads,attention_features)
        self.transition2 = Transition(in_features,nn.LayerNorm)
    def forward(
            self,
            x_1,
            x_2,
            pos_emb,
            edge_index,
            packed_sequence_emb,
            packed_sequence_mask,
            batch=None,
            attention_batch_size=200,
            **kwargs,
    ):
        y_1,attention_scores = self.sequence_attention(x_1,packed_sequence_emb,packed_sequence_mask,batch,attention_batch_size)
        y_1 = self.nodeattention(y_1,x_2,pos_emb,edge_index)
        y_1 = self.transition1(y_1)
        y_2 = self.norm(math.sqrt(2)*x_2 + self.outer(y_1,edge_index))
        y_2 = self.edge_deliver(y_2,edge_index)
        # y_2 = self.traingle_attention(y_2)
        y_2 = self.transition2(y_2)
        return y_1,y_2,attention_scores
class Cryformer_no_seq(nn.Module):
    def __init__(
            self,
            in_features:int,
            attention_heads: int,
            attention_features: int = 48,
            num_neighbours : int = 20,
            activation_class:nn.Module = nn.ReLU
    ):
        super().__init__()
        self.nodeattention = NodeAttention(in_features,num_neighbours,attention_heads,attention_features)
        self.transition1 = Transition(in_features,nn.LayerNorm)
        self.outer = OutProductMean(in_features)
        self.norm = nn.LayerNorm(in_features)
        self.edge_deliver = EdgeAttention(in_features,num_neighbours,attention_heads,attention_features)
        self.transition2 = Transition(in_features,nn.LayerNorm)
    def forward(
            self,
            x_1,
            x_2,
            pos_emb,
            edge_index,
            **kwargs,
    ):
        y_1 = self.nodeattention(x_1,x_2,pos_emb,edge_index)
        y_1 = self.transition1(y_1)
        y_2 = self.norm(math.sqrt(2)*x_2 + self.outer(y_1,edge_index))
        y_2 = self.edge_deliver(y_2,edge_index)
        y_2 = self.transition2(y_2)
        return y_1,y_2