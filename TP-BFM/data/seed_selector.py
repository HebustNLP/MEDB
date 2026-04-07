

import random
from typing import Dict, List, Tuple


class SeedSelector:
  

    def __init__(self, class_samples: Dict[str, List[str]], seed: int = 42):
  
        self.class_samples = class_samples
        self.seed = seed
        random.seed(seed)

    def select_seeds(
        self, class_name: str, k: int = 5, exclude_indices: List[int] = None
    ) -> List[str]:

        samples = self.class_samples.get(class_name, [])
        if not samples:
            return []

        available_indices = list(range(len(samples)))
        if exclude_indices:
            available_indices = [i for i in available_indices if i not in exclude_indices]

        k = min(k, len(available_indices))
        selected_indices = random.sample(available_indices, k)

        return [samples[i] for i in selected_indices]

    def select_seeds_for_all_classes(self, k: int = 5) -> Dict[str, List[str]]:
   
        seeds = {}
        for class_name in self.class_samples:
            seeds[class_name] = self.select_seeds(class_name, k)
        return seeds