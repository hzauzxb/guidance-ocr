from queue import Queue

import torch
from transformers import AutoTokenizer


SPLIT_CHAR_LIST: list[str] = [':', '-']


def split_text_func(texts: list[str]) -> list[str]:
    """ split text
    for example:
        origin ocr: "abcd:efg"
        result: "abcdefg", "abcd:", "efg"

    Parameters
    ----------
    texts : list[str]

    Returns
    -------
    list[str]
        _description_
    """
    for split_char in SPLIT_CHAR_LIST:
        q = Queue()
        [q.put(text) for text in texts]
        new_texts: list[str] = []
        while not q.empty():
            text: str = q.get()
            if not text:
                continue
            new_texts.append(text)
            if split_char in text:
                idx = text.index(split_char)
                new_texts.append(text[:idx + 1])
                q.put(text[idx + 1: ])
        texts = list(set(new_texts))
    return texts


class TreeNode:
    def __init__(self, char: str = None):
        self.childrens: dict[str, "TreeNode"] = dict()
        self.char = char
        self.maybe_last = True

    def __contains__(self, char: str) -> bool:
        return char in self.childrens

    def __getitem__(self, char: str) -> "TreeNode":
        return self.childrens[char]

    def add_node(self, char: str, node: "TreeNode") -> "TreeNode":
        self.childrens[char] = node
        return node
    
    def check_and_search(self, text: str, processor) -> tuple[bool, list['TreeNode']]:
        cur_node = self
        for char in text:
            if char not in cur_node:
                return False, []
            cur_node = cur_node[char]
        
        candidates = [cur_node]
        if cur_node.is_leaf or cur_node.maybe_last:
            candidates.extend(processor.all_nodes)
        return True, list(set(candidates))

    @property
    def is_leaf(self):
        return len(self.childrens) == 0

    @classmethod
    def build_from_ocr(cls, texts: list[str], tokenizer: AutoTokenizer) -> "TreeNode":
        root = TreeNode()
        cur_root = root
        for text in texts:
            tokens: list[int] = tokenizer.encode(text, add_special_tokens=False)
            for token in tokens:
                if token in cur_root:
                    cur_root = cur_root[token]
                else:
                    cur_root = cur_root.add_node(token=token, node=TreeNode(token=token))
            cur_root = root
        return root
    
    @classmethod
    def get_all_nodes(cls, root: "TreeNode") -> list['TreeNode']:
        all_nodes = list()
        q = Queue()
        q.put(root)
        while not q.empty():
            node: "TreeNode" = q.get()
            all_nodes.append(node)
            for child in node.childrens.values():
                q.put(child)
        return all_nodes
    
    @classmethod
    def build_from_ocr(cls, ocr_texts: list[str]) -> "TreeNode":
        root = TreeNode()
        cur_root = root
        for text in ocr_texts:
            for char in text:
                if char in cur_root:
                    cur_root = cur_root[char]
                else:
                    cur_root = cur_root.add_node(char, TreeNode(char=char))
            cur_root.maybe_last = True
            cur_root = root
        return root


class OCRLogitProcessor:
    def __init__(
        self,
        ocr_tree: TreeNode,
        tokenizer: AutoTokenizer,
        filter_value: int = -float("Inf"),
        special_tokens: list[int] = None,
        topk: int = 15
    ):
        # origin pointer
        self.ocr_tree = ocr_tree
        self.all_nodes: list[TreeNode] = TreeNode.get_all_nodes(root=ocr_tree)
        self.cur_ocr_trees: list[TreeNode] = self.all_nodes
        self.filter_value = filter_value
        self.special_tokens = special_tokens
        self.topk = topk
        
        self.is_valid = True
        self.tokenizer = tokenizer
    
    def clear(self):
        self.is_valid = True
        self.cur_ocr_trees = self.all_nodes
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """process logit by ocr informations

        Parameters
        ----------
        input_ids : torch.LongTensor
            model input
        scores : torch.FloatTensor
            input_ids logits

        Returns
        -------
        torch.FloatTensor
            input_ids logits
        """
        if not self.ocr_tree or not self.is_valid or input_ids.size()[0] != 1:
            return scores

        vocab_size: int = input_ids.size()[-1]
        dim: int = scores.dim()
        
        order_indexes = scores.argsort(dim=-1, descending=True)[..., :self.topk]
        order_indexes = order_indexes[0]
        
        valid_token: int | None = None
        
        for token in order_indexes:
            token = int(token)
            valid_next_nodes: list[TreeNode] = []
            text: str = self.tokenizer.decode(token)
            
            for node in self.cur_ocr_trees:
                can, valid_nodes = node.check_and_search(text=text, processor=self)
                if not can:
                    continue
                valid_next_nodes.extend(valid_nodes)
            
            if len(valid_next_nodes):
                self.cur_ocr_trees = list(set(valid_next_nodes))
                valid_token = token
            elif token in self.special_tokens:
                self.cur_ocr_trees = self.all_nodes
                valid_token = token
            else:
                continue
            break
        if not valid_token:
            self.is_valid = False
            return scores
        
        mask: torch.BoolTensor = torch.ones((vocab_size), device=scores.device, dtype=torch.bool)
        mask[valid_token] = 0
        for _ in range(dim - 1):
            mask = mask[None, ...]
        scores_processed: torch.FloatTensor = scores.masked_fill(mask, self.filter_value)
        return scores


def get_ocr_logit_processor(
    texts: list[str],
    tokenizer: AutoTokenizer,
    special_chars: list[str] = [],
    split_text: bool = False,
    topk: int = 15
):
    if split_text:
        texts: list[str] = split_text_func(texts=texts)
    
    ocr_tree: TreeNode = TreeNode.build_from_ocr(ocr_texts=texts)
    special_tokens: list[int] = []
    for text in special_chars:
        special_tokens.extend(tokenizer.encode(text))
    special_tokens = list(set(special_tokens))
    ocr_logit_processor: OCRLogitProcessor = OCRLogitProcessor(
        ocr_tree=ocr_tree,
        special_tokens=special_tokens,
        tokenizer=tokenizer,
        topk=topk
    )
    return ocr_logit_processor
        

