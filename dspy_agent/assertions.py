from typing import Any

class Assert:  #Minimal assertion for now

    def __call__(self, preds: Any, **kwargs):  
        return self.forward(preds, **kwargs)
    def forward(self, preds: Any, **kwargs) -> bool:  
        raise NotImplementedError


class RefAssertion(Assert): 

    def forward(self, preds, **kwargs):  
        return all(ref in preds.solution for ref in preds.references)
