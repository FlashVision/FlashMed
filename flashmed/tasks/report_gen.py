"""Radiology report generation task."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import TASKS


@TASKS.register("report_gen")
class ReportGenerationTask:
    """Automated radiology report generation from medical images.

    Generates structured reports with findings, impressions, and recommendations
    using a vision-language model architecture.

    Args:
        vocab_size: Vocabulary size for tokenizer
        max_length: Maximum report length in tokens
        beam_size: Beam search width for generation
        length_penalty: Penalty for longer sequences during beam search
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_length: int = 256,
        beam_size: int = 4,
        length_penalty: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for report generation.

        Args:
            logits: Model output [B, T, vocab_size]
            target_ids: Target token IDs [B, T]
        """
        B, T, V = logits.shape
        logits_flat = logits[:, :-1].contiguous().view(-1, V)
        targets_flat = target_ids[:, 1:].contiguous().view(-1)
        return self.criterion(logits_flat, targets_flat)

    @torch.no_grad()
    def generate_beam(
        self,
        model: nn.Module,
        images: torch.Tensor,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
    ) -> List[List[int]]:
        """Generate reports using beam search.

        Args:
            model: The VLM model
            images: Input images [B, C, H, W]
            bos_token_id: Start token
            eos_token_id: End token

        Returns:
            List of token ID sequences (one per image)
        """
        model.eval()
        B = images.shape[0]
        device = images.device

        visual_features = model.encode_image(images)
        results = []

        for b in range(B):
            vis_feat = visual_features[b:b + 1].expand(self.beam_size, -1, -1)

            beam_seqs = torch.full((self.beam_size, 1), bos_token_id, dtype=torch.long, device=device)
            beam_scores = torch.zeros(self.beam_size, device=device)
            beam_scores[1:] = -1e9

            finished = []

            for step in range(self.max_length):
                logits = model.text_decoder(beam_seqs, vis_feat)
                next_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_logits, dim=-1)

                scores = beam_scores.unsqueeze(1) + log_probs
                scores_flat = scores.view(-1)

                top_scores, top_indices = scores_flat.topk(self.beam_size * 2)
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size

                new_seqs = []
                new_scores = []

                for score, beam_idx, token_idx in zip(top_scores, beam_indices, token_indices):
                    seq = torch.cat([beam_seqs[beam_idx], token_idx.unsqueeze(0).unsqueeze(0).squeeze(0)], dim=0)

                    if token_idx.item() == eos_token_id:
                        length_norm = ((5 + len(seq)) / 6) ** self.length_penalty
                        finished.append((score.item() / length_norm, seq.tolist()))
                    else:
                        new_seqs.append(seq)
                        new_scores.append(score)

                    if len(new_seqs) >= self.beam_size:
                        break

                if not new_seqs or len(finished) >= self.beam_size:
                    break

                beam_seqs = torch.stack(new_seqs)
                beam_scores = torch.tensor(new_scores, device=device)

            if finished:
                finished.sort(key=lambda x: x[0], reverse=True)
                results.append(finished[0][1])
            else:
                results.append(beam_seqs[0].tolist())

        return results

    def compute_metrics(self, generated: List[str], references: List[str]) -> Dict[str, float]:
        """Compute NLG metrics (BLEU, ROUGE-L, CIDEr approximation).

        Args:
            generated: List of generated report strings
            references: List of reference report strings

        Returns:
            Dict of metric names and scores
        """
        bleu_scores = []
        rouge_l_scores = []

        for gen, ref in zip(generated, references):
            gen_tokens = gen.lower().split()
            ref_tokens = ref.lower().split()

            bleu_scores.append(self._compute_bleu(gen_tokens, ref_tokens))
            rouge_l_scores.append(self._compute_rouge_l(gen_tokens, ref_tokens))

        return {
            "bleu-4": np.mean(bleu_scores),
            "rouge-l": np.mean(rouge_l_scores),
        }

    @staticmethod
    def _compute_bleu(hypothesis: List[str], reference: List[str], max_n: int = 4) -> float:
        """Compute BLEU-N score."""
        from collections import Counter

        if len(hypothesis) == 0:
            return 0.0

        scores = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(zip(*[hypothesis[i:] for i in range(n)]))
            ref_ngrams = Counter(zip(*[reference[i:] for i in range(n)]))
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = max(len(hypothesis) - n + 1, 1)
            scores.append(matches / total if total > 0 else 0.0)

        if any(s == 0 for s in scores):
            return 0.0

        import math
        log_avg = sum(math.log(s + 1e-10) for s in scores) / len(scores)

        bp = 1.0
        if len(hypothesis) < len(reference):
            bp = math.exp(1 - len(reference) / max(len(hypothesis), 1))

        return bp * math.exp(log_avg)

    @staticmethod
    def _compute_rouge_l(hypothesis: List[str], reference: List[str]) -> float:
        """Compute ROUGE-L F1 score."""
        if not hypothesis or not reference:
            return 0.0

        m, n = len(reference), len(hypothesis)
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    lcs[i][j] = lcs[i - 1][j - 1] + 1
                else:
                    lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

        lcs_len = lcs[m][n]
        precision = lcs_len / n if n > 0 else 0
        recall = lcs_len / m if m > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
