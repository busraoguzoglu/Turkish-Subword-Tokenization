from tokenizers import Tokenizer, models, trainers
from tokenizers.models import Unigram

def main():

    tokenizer = Tokenizer(models.Unigram())
    trainer = trainers.UnigramTrainer(
        unk_token="<unk>",
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    )
    tokenizer.train(files=["UD_Turkish-Penn/tr_penn-ud-train.txt", "UD_Turkish-Penn/tr_penn-ud-dev.txt"], trainer=trainer)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    encoding = tokenizer.encode("Merhaba bu bir kelime")
    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))

    vocab = tokenizer.get_vocab()
    print(vocab)

if __name__ == '__main__':
    main()