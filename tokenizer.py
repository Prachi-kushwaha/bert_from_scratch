from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece


class BertTokenizer:
    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        is_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        strip_accents: bool = True,
        **kwargs,
    ):

        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }

        self.vocab = vocab
        self.is_lower_case = is_lower_case
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.strip_accents = strip_accents

        # tokenizer model
        self._tokenizer = Tokenizer(
            WordPiece(self.vocab, unk_token=str(self.unk_token))
        )

        # normalizer
        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            lowercase=self.is_lower_case,
            strip_accents=self.strip_accents,
        )

        # pre-tokenizer
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # post-processor
        cls_token_id = self.vocab[str(self.cls_token)]
        sep_token_id = self.vocab[str(self.sep_token)]

        self._tokenizer.post_processor = processors.BertProcessing(
            (str(self.sep_token), sep_token_id),
            (str(self.cls_token), cls_token_id),
        )

        # decoder
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")