# cocktail efficacy function
from discrete_mixed_bo.problems.base import DiscreteTestProblem
from botorch.test_functions.base import ConstrainedBaseTestProblem
import torch


key_2_val = lambda key: [int(x) for x in key]
key_idx = lambda p: ''.join([str(int(x)) for x in p])
# fitness function with memory
class scorer_obj():
    def __init__(self, score_func):
        self.score_func = score_func
        self.score_dict = {}
    def score(self, x):
        try:
            return self.score_dict[key_idx(x)]
        except:
            if sum(x)==0:
                val = 0.0
            else: 
                val = self.score_func(x)
            self.score_dict[key_idx(x)] = val
            return val

class cocktail_efficacy(DiscreteTestProblem,ConstrainedBaseTestProblem):
    num_constraints = 2
    def __init__(self, dim: int,
        objective_func, 
        negate: bool = False,
    ) -> None:

        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(
            negate=negate, integer_indices=list(range(self.dim))
        )
        self.scorer = scorer_obj(objective_func)

    def evaluate_true(self, X):
        '''
        X --> batch tensor
        '''
        
        Y = torch.tensor(
            [self.scorer.score(x.numpy()) for x in X.view(-1, self.dim).cpu() ],
            dtype=X.dtype,
            device=X.device,
        ).view(X.shape[:-1])
        
        return Y

    def evaluate_slack_true(self,X):

        slack = torch.tensor([[3-x.numpy().sum(), x.numpy().sum()-3 ] for x in  X.view(-1, self.dim).cpu()],
            dtype=X.dtype,
            device=X.device,
            ).view(-1,self.num_constraints)

        
        return slack


