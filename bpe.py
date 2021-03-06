from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from unigram_lm import unigram, perplexity


def main():

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=25000, show_progress=True, initial_alphabet=ByteLevel.alphabet())
    tokenizer.train(files=["UD_Turkish-Penn/tr_penn-ud-train.txt", "UD_Turkish-Penn/tr_penn-ud-dev.txt"], trainer=trainer)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    # If we want to save the trained model
    tokenizer.model.save('.')

    # If we want to use the saved model
    tokenizer.model = BPE('vocab.json', 'merges.txt')

    # Test
    encoding = tokenizer.encode("Merhaba bu bir kelime")
    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))

    # Get the subword vocabulary:
    vocab = tokenizer.get_vocab()

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