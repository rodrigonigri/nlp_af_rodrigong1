import openai
import streamlit as st
import torch
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


def get_vocabulary(text : str,
                   expr: str=r"\b\w+\b",
                   case_sensitive : bool=False,
                   ) -> dict:
    if case_sensitive == False:
        text = text.lower()  
    vocabulary = set(re.findall(expr, text))
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for i, token in enumerate(vocabulary):
        if token not in vocab:
            vocab[token] = i+2 # We start from 2 because 0 and 1 are reserved for <UNK> and <PAD>
            inverse_vocab.append(token)
    return vocab, inverse_vocab

def tokenize_words(text : str,
             vocab : dict,
             expr : str= r"\b\w+\b",
             sentence_length : int = 10,
             case_sensitive : bool = False) -> list:
    if case_sensitive == False:
        text = text.lower()
    words = re.findall(expr, text)
    tokens = []
    for i, w in enumerate(words):
        if i == sentence_length:
            break
        if w in vocab:
            tokens.append(vocab[w])
        else:
            tokens.append(vocab["<UNK>"])


    if len(tokens) < sentence_length:
        n_pad = sentence_length - len(tokens)
        pad = [vocab["<PAD>"]] * n_pad
        tokens = pad + tokens
    return tokens

def detokenize_words(tokens : list,
                    invert_vocab : list) -> str:
    text = " ".join([invert_vocab[token] for token in tokens])
    return text    

class MyTokenizer:
    def __init__(self, sentence_length, case_sensitive=False, vocab=None, inverse_vocab=None):
        self.sentence_length = sentence_length
        self.case_sensitive = case_sensitive
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        if vocab is not None:
            self.vocab_size = len(vocab)

    def fit(self, phrases : list, expr : str=r"\b\w+\b"):
        self.vocab, self.inverse_vocab = get_vocabulary(" ".join(phrases),
                                                        expr=expr,
                                                        case_sensitive=self.case_sensitive)
        self.vocab_size = len(self.vocab)
        
    def __call__(self, x):
        return tokenize_words(x,
                              self.vocab,
                              sentence_length=self.sentence_length,
                              case_sensitive=self.case_sensitive)

df = pd.read_csv("./data/docs.csv", sep=";")

# Função para carregar os vetores GloVe
def load_glove_vectors(glove_file):
    glove_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            glove_vectors[word] = vector
    return glove_vectors

# Função para gerar o vocabulário a partir dos vetores GloVe
def get_vocabulary_from_glove(glove_vectors):
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for word, vector in glove_vectors.items():
        vocab[word] = len(inverse_vocab)
        inverse_vocab.append(word)
    return vocab, inverse_vocab

# Verificando e carregando GloVe e o vocabulário apenas uma vez
if "glove_vectors" not in st.session_state:
    # Exibir um aviso temporário durante o carregamento
    with st.spinner("Carregando embeddings GloVe, por favor aguarde..."):
        glove_file = "glove.6B/glove.6B.300d.txt"  # Ajuste o caminho conforme necessário
        st.session_state["glove_vectors"] = load_glove_vectors(glove_file)
        st.session_state["vocab"], st.session_state["inverse_vocab"] = get_vocabulary_from_glove(st.session_state["glove_vectors"])
    st.success("Embeddings GloVe carregados com sucesso!")  # Mensagem de sucesso quando o carregamento termina



tokenizer = MyTokenizer(sentence_length=10, case_sensitive=False, vocab=st.session_state["vocab"], inverse_vocab=st.session_state["inverse_vocab"])
tokenizer.fit(df["Resposta"])
df['Tokens'] = df["Resposta"].apply(tokenizer)


# Definindo a função que calcula o embedding médio
def get_average_embedding(tokens, embedding):
    # Converte tokens em embeddings e calcula a média
    embeddings = torch.stack([embedding(torch.tensor([token])) for token in tokens if token != tokenizer.vocab["<PAD>"]])
    return torch.mean(embeddings, dim=0) if embeddings.shape[0] > 0 else torch.zeros(embedding.embedding_dim)

embedding_dim = 300
vocab_size = len(st.session_state["glove_vectors"]) + 2
embedding = torch.nn.Embedding(vocab_size, embedding_dim)

df["Embedding"] = df["Tokens"].apply(lambda tokens: get_average_embedding(tokens, embedding))



def get_closest_documents(query_embedding, df, top_n=5, threshold=9.5):
    # Calcular a similaridade coseno entre o embedding da query e os embeddings dos documentos
    similaridades = df['Embedding'].apply(
        lambda x: cosine_similarity(
            query_embedding.detach().numpy().reshape(1, -1),  # Garantindo 2D
            x.detach().numpy().reshape(1, -1)                # Garantindo 2D
        )[0][0]
    )
    
    # Encontrar os índices dos documentos mais similares
    indices_mais_similares = similaridades.nlargest(top_n).index
    
    # Filtrar documentos com similaridade acima do threshold
    documentos_relevantes = df.iloc[indices_mais_similares]
    documentos_relevantes = documentos_relevantes[similaridades[indices_mais_similares] >= threshold]
    
    return documentos_relevantes, similaridades[indices_mais_similares]

# Código Streamlit para o chatbot
st.title("ChatBot Insper")

# Inicializando o cliente OpenAI com a chave da API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Definindo o modelo padrão para uso
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Armazenando mensagens na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibindo mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quando o usuário envia uma nova mensagem
if prompt := st.chat_input("Faça perguntas sobre a faculdade?"):
    prompt_header = "Chat, você está sendo usado em um RAG e deve se comportar como o chatbot da faculdade insper. Abaixo estão os documentos que contém as perguntas e respostas que você deve basear sua resposta. Não diga nenhuma informação extra! Somente o escrito nos documentos."

    # o prompt eh a mensagem em si. entao vai precisar tokenizar a mensagem, pegar o embedding e ver qual/is documento(s)
    # tem o embedding mais proximo.
    tokens = tokenizer(prompt)

    query_embedding = get_average_embedding(tokens, embedding)

    prompt_documents, similaridade = get_closest_documents(query_embedding, df, threshold=0.05)
    
    # concatenar o prompt_header, prompt_documents["Pergunta"] e prompt_documents["Resposta"] e o prompt em si
    
    prompt_final = "Pergunta do usuário:" + prompt + "\n\n" + "Instruções:" + prompt_header + "Documentos para se basear na resposta:" + "\n".join(prompt_documents["Pergunta"].tolist()) + "\n" + "\n".join(prompt_documents["Resposta"].tolist())
    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inicializando a variável para acumular a resposta
    response_content = ""
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder para exibir a resposta completa depois
        response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[{"role": "user", "content": prompt_final}],
            stream=True,
        )
        
        # Acumulando a resposta completa
        for chunk in response:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            response_content += content
            response_placeholder.markdown(response_content)  # Atualizando o conteúdo acumulado

    # Armazenando a resposta final no estado da sessão
    st.session_state.messages.append({"role": "assistant", "content": response_content})
