import torch.nn as nn
import torchvision.models as models

class DigitalClockClassifier(nn.Module):
    """
    ResNet18 backbone with 3 independent classification heads:
      - hour_head:   24 classes  (0–23)
      - minute_head: 60 classes  (0–59)
      - second_head: 60 classes  (0–59)

    Why classification instead of regression?
    Regression treats 12:59 and 13:00 as "close" in loss space,
    even though they are very different clock positions.
    Classification treats each digit combination as an independent class,
    so the model can learn crisp decision boundaries.
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = base.fc.in_features          # 512 for ResNet18
        base.fc = nn.Identity()             # Remove the original head
        self.backbone = base

        # Shared bottleneck (helps all 3 heads)
        self.bottleneck = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hour_head   = nn.Linear(256, 24)
        self.minute_head = nn.Linear(256, 60)
        self.second_head = nn.Linear(256, 60)

    def forward(self, x):
        f = self.backbone(x)
        f = self.bottleneck(f)
        return self.hour_head(f), self.minute_head(f), self.second_head(f)

    def predict_time(self, x):
        """Returns integer (h, m, s) — use this at inference time."""
        h_logits, m_logits, s_logits = self.forward(x)
        h = h_logits.argmax(dim=1)
        m = m_logits.argmax(dim=1)
        s = s_logits.argmax(dim=1)
        return h, m, s