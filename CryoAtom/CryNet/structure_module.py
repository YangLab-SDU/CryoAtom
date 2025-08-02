import torch
from torch import nn
from einops.layers.torch import Rearrange
import math
import einops

from CryoAtom.utils.affine_utils import affine_mul_vecs, invert_affine
from CryoAtom.CryNet.common_modules import Rope

class InvariantPointAttention(nn.Module):
    def __init__(self,in_features:int,attention_heads:int=12,c:int=48,query_points:int=4,point_values:int=8):
        super().__init__()
        self.ifz = in_features
        self.ahz = attention_heads
        self.afz = c
        self.attention_scale = math.sqrt(self.afz)
        self.qpz = query_points
        self.vpz = point_values
        self.gamma = nn.Parameter(1.1*torch.ones((1,1,attention_heads)))
        self.q1 = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N (ahz afz) -> N ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.k1 = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N (ahz afz) -> N ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v1 = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("N (ahz afz) -> N ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.ifz,self.ahz * self.qpz * 3, bias=False),
            Rearrange("N (kz ahz qpz d) -> N kz ahz qpz d", kz=1,ahz=self.ahz, qpz=self.qpz,d=3)
        )
        self.k2 = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.qpz * 3, bias=False),
            Rearrange("N (ahz qpz d) -> N ahz qpz d", ahz=self.ahz, qpz=self.qpz, d=3)
        )
        self.v2 = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.vpz * 3, bias=False),
            Rearrange("N (ahz vpz d) -> N ahz vpz d", ahz=self.ahz, vpz=self.vpz, d=3)
        )
        self.bia = nn.Linear(in_features//2,attention_heads,bias=False)
        self.back = nn.Linear(self.ahz*self.afz+self.ahz*(self.ifz//2)+self.ahz*self.vpz*4,self.ifz)
        self.short = nn.Identity()
        self.norm = nn.LayerNorm(in_features)
    def forward(self,x1,x2,affines,pos_emb,edge_index):
        query1 = self.q1(x1) # N ahz afz
        key1 = self.k1(x1)[edge_index]# N kz ahz afz
        query1,key1 = Rope(query1,key1,pos_emb,edge_index)
        value1 = self.v1(x1)[edge_index]# N kz ahz afz
        query2 = self.q2(x1) # N 1 ahz qpz 3
        key2 = self.k2(x1)[edge_index]# N kz ahz qpz 3
        value2 = self.v2(x1)# N ahz vpz 3
        bias = self.bia(x2)
        wc = math.sqrt(2/(9*self.qpz))
        wl = math.sqrt(1/3)
        attention_score1 = (torch.einsum('nai,nkai->nka',query1,key1)/self.attention_scale)+bias
        attention_score2 = -torch.sum(torch.square(
            affine_mul_vecs(affines,query2)-affine_mul_vecs(affines,key2)),
        dim=-1).sum(dim=-1) # N kz ahz
        attention_scores = wl*(attention_score1+0.1*wc*self.gamma*attention_score2)# N kz ahz
        attention_weights = torch.softmax(attention_scores,dim=1)
        output1 = torch.einsum('nka,nkai->nai',attention_weights,value1)# N ahz afz
        output2 = torch.einsum('nka,nki->nai',attention_weights,x2)# N ahz ifz
        output_ipa = torch.einsum('nka,nkavd->navd',attention_weights,affine_mul_vecs(affines,value2)[edge_index])# N ahz vpz 3
        output_ipa = affine_mul_vecs(invert_affine(affines),output_ipa)# N ahz vpz 3
        output_ipa_norm = torch.norm(output_ipa,dim=-1,p=2)
        out = self.back(torch.cat((output1.flatten(1),output2.flatten(1),output_ipa.flatten(1),output_ipa_norm.flatten(1)),dim=1))
        out = self.norm(math.sqrt(2)*self.short(x1) + out)
        return out

class Transition(nn.Module):
    def __init__(self,in_features:int,n:int=2):
        super().__init__()
        self.transition = nn.Sequential(nn.Linear(in_features,in_features*n),
                                        nn.ReLU(),
                                        nn.Linear(in_features*n,in_features*n),
                                        nn.ReLU(),
                                        nn.Linear(in_features*n,in_features))
        self.short = nn.Identity()
        self.norm = nn.LayerNorm(in_features)
    def forward(self,x):
        y = self.transition(x)
        y = self.norm(y + math.sqrt(2)*self.short(x))
        return y
class LinearWithShortcut(nn.Module):
    def __init__(self,in_features:int,hidden_features:int,out_features:int,activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.ipa_linear1 = nn.Linear(in_features,hidden_features)
        self.ipa_linear2 = nn.Linear(in_features,hidden_features)
        self.lin1 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.lin2 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.lin3 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, out_features)
        )
    def forward(self,x1,x2):
        x = self.ipa_linear1(x1)+self.ipa_linear2(x2)
        x = x + self.lin1(x)
        x = x + self.lin2(x)
        x = self.lin3(x)
        return x

class LinearWithSeq(nn.Module):
    def __init__(self,in_features:int,hidden_features:int,out_features:int,activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.ipa_linear1 = nn.Linear(in_features,hidden_features)
        self.ipa_linear2 = nn.Linear(in_features,hidden_features)
        self.lin1 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.lin2 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, out_features)
        )
    def forward(self,x1,x2):
        x = self.ipa_linear1(x1)+self.ipa_linear2(x2)
        x = x + self.lin1(x)
        x = self.lin2(x)
        return x

class LinearWithEdge(nn.Module):
    def __init__(self,in_features:int,hidden_features:int,neibour_dim:int,out_features:int,activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.ipa_linear1 = nn.Linear(in_features+neibour_dim,hidden_features)
        self.ipa_linear2 = nn.Linear(in_features,hidden_features)
        self.lin1 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.lin2 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.lin3 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, out_features)
        )
    def forward(self,x1,x2):
        x = self.ipa_linear1(x1)+self.ipa_linear2(x2)
        x = x + self.lin1(x)
        x = x + self.lin2(x)
        x = self.lin3(x)
        return x