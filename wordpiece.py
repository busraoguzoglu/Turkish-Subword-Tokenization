import argparse
import glob

from tokenizers import BertWordPieceTokenizer

def main():

    tokenizer = BertWordPieceTokenizer(clean_text=True, strip_accents=True, lowercase=True,)

    # And then train
    tokenizer.train(
        files=["UD_Turkish-Penn/tr_penn-ud-train.txt"],
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    encoding = tokenizer.encode("Merhaba bu bir kelime")
    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))

    vocab = tokenizer.get_vocab()
    print(vocab)

if __name__ == '__main__':
    main()