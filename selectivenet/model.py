import torch
import torch.nn.functional as F

class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.  
    """
    def __init__(self, features, dim_features:int, num_classes:int, init_weights=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
        """
        super(SelectiveNet, self).__init__()
        self.features = features
        self.dim_features = dim_features
        self.num_classes = num_classes
        
        # represented as f() in the original paper
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes),
        )

        # represented as g() in the original paper
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.aux_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes),
        )

        self.activation = torch.nn.Softmax()

        # initialize weights of heads
        init_weights = None
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.selector)
            self._initialize_weights(self.aux_classifier)

    def forward_logit(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        
        #prediction_out = self.classifier(x)
        #selection_out  = self.selector(x)
        auxiliary_out  = self.aux_classifier(x)

        return auxiliary_out #prediction_out, selection_out, auxiliary_out

    def forward(self, x):
        x = self.forward_logit(x)
        return self.activation(x, dim=1)

    # you coudl redefine the above to forward_logit and then have a forward with the softma
    # it could be, then, that your scoring is wrong, but there's probably more

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
