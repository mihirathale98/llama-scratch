from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs, Transformer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    
    @staticmethod
    def build(checkpoint_path: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()

        if load_model:
            checkpoints = sorted(Path(checkpoint_path).glob("*.pth"))
            assert len(checkpoints) > 0, "no checkpoint found"
            chk_path = checkpoints[0]
            print(f"loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location=device)
            print(f"took {time.time() - prev_time}")
            prev_time = time.time()
        
        with open(Path(checkpoint_path) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Storage)
        
        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint)
            print(f"took {time.time() - prev_time}")

        return LLaMA(model, tokenizer, model_args)
    
    def _sample_top_p(self, probs, top_p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
        

    
    def text_completion(self,
                        prompts: list[str],
                        temperature: float = 0.6,
                        top_p: float=0.9,
                        max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        prompt_tokens = [self.tokenizer.encode(p, out_type=int, add_bos=True, add_eos=False) for p in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)

        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.Tensor([False] * batch_size).to(self.args.device) 
        prompt_tokens_mask = tokens != pad_id

        for cur_pos in tqdm(range(1, total_len)):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1]/ temperature, dim=-1)
                next_token =self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break
    
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_index = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_index]

            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return out_tokens, out_text

if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"
    model = LLaMA.build(
        checkpoint_path="Llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=2048,
        max_batch_size=1,
        device=device
    )

    out_tokens, out_text = model.text_completion(["The quick brown fox jumps over the lazy dog."], max_gen_len=128)

    print(out_text)