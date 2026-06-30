import os
import torch
from alignscore import AlignScore

class AlignScoreEvaluator:
    def __init__(self, ckpt_path: str = "./checkpoints/AlignScore-base.ckpt", model_name: str = "roberta-base", use_gpu: bool = False):
        """
        Initializes the AlignScore model framework once to optimize inference loops.
        """
        self.ckpt_path = ckpt_path
        self.device = "cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.scorer = None
        
        # Check for checkpoint file to prevent catastrophic initialization failure
        if not os.path.exists(self.ckpt_path):
            print(f"Warning: AlignScore checkpoint missing at {self.ckpt_path}. "
                  f"AlignScore evaluation will safely default to return 0.0.")
        else:
            # Set up the core framework class
            self.scorer = AlignScore(
                model=model_name, 
                batch_size=8,              # Standard micro-batching for local memory stability
                device=self.device, 
                ckpt_path=self.ckpt_path,
                evaluation_mode='nli_sp'   # Splitting model mode for optimal document processing
            )

    def evaluate(self, generated_summary: str, source_context: str) -> float:
        """
        Computes the alignment score for a single summary against its source document.
        
        Parameters:
        -----------
        generated_summary : str
            The summary/claim text being evaluated.
        source_context : str
            The original source document context.
            
        Returns:
        --------
        float
            The factual alignment/consistency score.
        """
        if self.scorer is None:
            return 0.0
            
        # The underlying library native method expects arrays/lists
        raw_scores = self.scorer.score(contexts=[source_context], claims=[generated_summary])
        
        # Extract the scalar metric value out of the single-element results collection
        return round(float(raw_scores[0]), 4) if raw_scores else 0.0