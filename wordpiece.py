from tokenizers import BertWordPieceTokenizer
from unigram_lm import unigram, perplexity

def main():

    tokenizer = BertWordPieceTokenizer(clean_text=True, strip_accents=True, lowercase=True)
    tokenizer.train(files=["UD_Turkish-Penn/tr_penn-ud-train.txt", "UD_Turkish-Penn/tr_penn-ud-dev.txt"], min_frequency=2, vocab_size=25000, show_progress=True)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    encoding = tokenizer.encode("Merhaba bu bir kelime")
    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))

    vocab = tokenizer.get_vocab()

    # Perplexity:
    # Tokenize training and test corpus:
    with open("UD_Turkish-Penn/tr_penn-ud-train.txt", encoding="utf-8") as f:
        train_contents = f.read()
    with open("UD_Turkish-Penn/tr_penn-ud-test.txt", encoding="utf-8") as f:
        test_contents = f.read()

    train_tokenized = tokenizer.encode(train_contents)
    test_tokenized = tokenizer.encode(test_contents)

    # Get unigram model to calculate perplexity
    model = unigram(train_tokenized.tokens)
    print('Perplexity:', perplexity(test_tokenized.tokens, model))

if __name__ == '__main__':
    main()