import torch
import torch.nn as nn
from transformers import AutoModel
from src.utility import cond_no_grad

from lightning import LightningModule


class TimeSeqMoSEMERTModel(LightningModule):
    def __init__(
        self,
        mert,
        num_class=10,
        config=None,
    ):
        super(TimeSeqMoSEMERTModel, self).__init__()

        self.mert = mert
        self.num_class = num_class
        self.num_aux = 1 if config["mose_aux"] > 0 else 0

        # Model architecture
        self.encoder = MERTEncoder(mert)
        
        self.branchs = nn.ModuleList()
        for _ in range(self.num_aux + 1):
            self.branchs.append(
                AttMLPHead(
                    feature_num=len(self.encoder.extract_layer_ids),
                    in_dim=self.encoder.out_dim, 
                    h_dim=config["representation_dim"], 
                    out_dim=num_class, 
                )
            )
        
        self.phase = "train"
        self.phase_set = ("att", "rep", "test", "predict", "train", "val")
        self.freeze1 = False
        self.freeze2 = True

    def train_aux(self):
        self.freeze1, self.freeze2 = True, False
    
    def train_main(self):
        self.freeze1, self.freeze2 = False, True
    
    def set_forward(self, phase):
        assert phase in self.phase_set
        self.phase = phase

    def forward(self, batch):
        if self.phase not in self.phase_set:
            print("phase not set to test or rep, force set to test")
            self.phase = "test"
        
        branch_out = self._forward(batch[0])

        if self.phase == "att":
            return [out[2] for out in branch_out]
        
        elif self.phase == "rep":
            return [out[0] for out in branch_out]
        
        elif self.phase in ("predict", "test"):
            return [out[1] for out in branch_out]
        else:
            return {
                    "y_pred": branch_out[0][1][-1], 
                    "y_aux_pred": branch_out[1][1][-1], 
                    "att": branch_out[0][2][-1],
                    "att_aux": branch_out[1][2][-1]
                    }
    
    def _forward_branch(self, x: torch.Tensor, branch_idx: int):
        logit, x_rep, att = self.branchs[branch_idx](x)
        return logit, x_rep, att
    
    def _forward(self, x: torch.Tensor):
        
        x_reps, atts, logits = [], [], []
        x_aux_reps, att_auxs, logit_auxs,  = [], [], []
        
        x = self.encoder(x)
        # mert_feature (batch_size, time_seq, 768 feature_dim)
        
        with cond_no_grad(cond=self.freeze1):
            logit, x_rep, att = self._forward_branch(x, 0)
            
        atts.append(att)
        x_reps.append(x_rep)
        logits.append(logit)
    
        if self.num_aux > 0:
            with cond_no_grad(cond=self.freeze2):
                aux_logit, aux_rep, aux_att = self._forward_branch(x, 1)
            
            att_auxs.append(aux_att)
            x_aux_reps.append(aux_rep)
            logit_auxs.append(aux_logit)

            return [
                (torch.stack(x_reps), torch.stack(logits), torch.stack(atts)),
                (torch.stack(x_aux_reps), torch.stack(logit_auxs), torch.stack(att_auxs)),
            ]
        return [
            (torch.stack(x_reps), torch.stack(logits), torch.stack(atts)),
            (None, None, None)
        ]
    
    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)


class MERTEncoder(nn.Module):
    def __init__(self, mert, mert_feature_layer='all'):
        super(MERTEncoder, self).__init__()
        
        if "95M" in mert:
            self.out_dim = 768
        elif "330M" in mert:
            self.out_dim = 1024
        else:
            raise ValueError(f"Invalid MERT model: {mert}")
        
        self.mert = AutoModel.from_pretrained(mert, trust_remote_code=True)
        
        for param in self.mert.parameters():
            param.requires_grad = False
        
        if mert_feature_layer == "all":
            self.extract_layer_ids = list(range(13))
        else:
            assert isinstance(mert_feature_layer, list)
            self.extract_layer_ids = mert_feature_layer
    
    def forward(self, x: torch.Tensor):
        
        with torch.no_grad():
            outputs = self.mert(x, output_hidden_states=True)
        # outputs [13 layer, batch_size, time_seq, 768 feature_dim]

        mert_feature = torch.stack(
            [outputs.hidden_states[i] for i in self.extract_layer_ids], dim=1
        )
        # mert_feature (batch_size, # layer, time_seq, 768 feature_dim)

        return mert_feature


class AttMLPHead(nn.Module):
    def __init__(self, feature_num, in_dim, h_dim, out_dim):
        super(AttMLPHead, self).__init__()
        self.aggregator = Aggregater(feature_num=feature_num)
        self.mose_block = AvgPoolAtt1D()
        self.classifier = TimeSeqLinearClassifier(
                    in_dim=in_dim, h_dim=h_dim, out_dim=out_dim)
    def forward(self, x: torch.Tensor):
        # x: mert_feature (batch_size, time_seq, 768 feature_dim)
        
        x = self.aggregator(x)
        z, att = self.mose_block(x)
        # z (batch_size, time_seq, 768 feature_dim)
        # att (batch_size, num_feature, 1)

        x_rep, logit = self.classifier(z)
        # rep (batch, hidden_dim)
        # logit (batch, num_class)
        return logit, x_rep, att


class Aggregater(nn.Module):
    def __init__(self, feature_num):
        super(Aggregater, self).__init__()
        self.drop_before = nn.Dropout(0.2)
        self.drop_after = nn.Dropout(0.2)
        if feature_num > 1:
            self.aggregator = nn.Conv2d(
                in_channels=feature_num, out_channels=1, kernel_size=1
            )
        else:
            self.aggregator = Passing()
    
    def forward(self, x: torch.Tensor):
        # x: MERTEncoder output (batch_size, # layer, time_seq, 768 feature_dim)
        x = self.drop_before(x)
        mert_feature = self.aggregator(x).squeeze(1)
        # mert_feature (batch_size, time_seq, 768 feature_dim)
        mert_feature = self.drop_after(mert_feature)
        
        return mert_feature


class AvgPoolAtt1D(nn.Module):
    def __init__(self):
        super(AvgPoolAtt1D, self).__init__()

        # tensor_input (batch_size, num_feature, d)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # (batch_size, num_feature, 1)

        self.attention = nn.Sequential( # (batch_size, 1, num_feature, 1)
            nn.Conv2d(1, 16, kernel_size=1, stride=1), # (batch_size, 16, num_feature, 1)
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=1, stride=1), # (batch_size, 1, num_feature, 1)
            nn.Flatten(start_dim=1, end_dim=2), # (batch_size, num_feature, 1)
            nn.Softmax(dim=-2)
        )

    def forward(self, x: torch.Tensor):
        # x (batch_size, num_feature, 768 feature_dim)

        feature = self.avg_pool(x)
        feature = torch.unsqueeze(feature, dim=1) # (batch_size, 1, num_feature, 1)
        att = self.attention(feature) # (batch_size, num_feature, 1)
        
        z = torch.mul(x, x.shape[1] * att)
        # z (batch_size, num_feature, 768 feature_dim)
        return z, att


class TimeSeqLinearClassifier(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super(TimeSeqLinearClassifier, self).__init__()
        self.drop = nn.Dropout(0.2)
        
        # self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        self.output_layer = nn.Linear(h_dim, out_dim)
    
    def forward(self, x: torch.Tensor):
        # x (batch_size, time_seq, 768 feature_dim)
        x = torch.mean(x, dim=1, keepdim=False)
        x = self.drop(x)

        x_rep = self.hidden_layer(x) # (batch, hidden_dim)
        logit = self.output_layer(x_rep)  # (batch, num_class)
        return x_rep, logit


class Passing(nn.Module):
    def forward(self, x: torch.Tensor):
        return x
