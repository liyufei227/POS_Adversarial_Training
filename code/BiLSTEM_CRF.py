import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, char_embedding_dim, char_hidden_dim):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim

        # Word Embedding Layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Character Embedding Layer
        self.char_embeddings = nn.Embedding(vocab_size, char_embedding_dim)
        # Character-Level BiLSTM Layer
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, bidirectional=True)
        # Main BiLSTM Layer
        self.lstm = nn.LSTM(embedding_dim + (2 * char_hidden_dim), hidden_dim, bidirectional=True)

        # Linear Layer to map BiLSTM output to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

        # CRF Layer
        self.crf = CRF(tagset_size)

    def forward(self, sentence, char_sentence):
        # Get the word and character embeddings for the sentence
        word_embeddings = self.word_embeddings(sentence)
        char_embeddings = self.char_embeddings(char_sentence)

        # Pass the character embeddings through the Character-Level BiLSTM
        char_lstm_out, _ = self.char_lstm(char_embeddings.view(len(char_sentence), 1, -1))

        # Concatenate word embeddings and Character-Level BiLSTM output
        embeddings = torch.cat((word_embeddings, char_lstm_out.view(len(sentence), -1)), 1)

        # Pass the concatenated embeddings through the Main BiLSTM
        lstm_out, _ = self.lstm(embeddings.view(len(sentence), 1, -1))

        # Map BiLSTM output to tag space
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        
         # CRF layer
        return self.crf.decode(tag_space)
