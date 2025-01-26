# Tradução de Texto com Interface Gráfica

Este projeto é uma aplicação que permite traduzir textos entre diferentes idiomas utilizando uma interface gráfica desenvolvida em Python com a biblioteca `tkinter`. Ele também utiliza um modelo de sequência para sequência (seq2seq) para realizar a tradução do texto fornecido.

## Funcionalidades

- **Interface Gráfica (GUI):** Permite ao usuário inserir o texto de entrada, selecionar o idioma e visualizar o texto traduzido.
- **Modelo Seq2Seq:** Utiliza um modelo de redes neurais para realizar a tradução automática de texto.
- **Treinamento e Decodificação:** Inclui funções para treinar o modelo e decodificar as sequências traduzidas.

## Requisitos

- Python 3.8+
- Bibliotecas necessárias:
  - `tkinter` (interface gráfica)
  - `numpy` (operações matemáticas e vetoriais)
  - `tensorflow` (treinamento e execução do modelo)

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Instale as dependências:**
   Certifique-se de que você tem o Python e o `pip` instalados. Em seguida, execute:
   ```bash
   pip install -r requirements.txt
   ```

   Caso o arquivo `requirements.txt` não esteja presente, instale manualmente as dependências:
   ```bash
   pip install numpy tensorflow
   ```

3. **Execute a aplicação:**
   ```bash
   python app.py
   ```

## Estrutura do Projeto

```plaintext
.
├── app.py               # Arquivo principal que executa a interface gráfica
├── model.py             # Arquivo contendo a lógica do modelo seq2seq
├── README.md            # Documentação do projeto
├── requirements.txt     # Dependências do projeto
└── data/                # (Opcional) Dados para treinamento do modelo
```

## Uso

1. **Abra a aplicação:** Execute o arquivo `app.py` para abrir a interface.
2. **Insira o texto:** No campo de entrada, digite o texto que deseja traduzir.
3. **Selecione o idioma:** Escolha o idioma de destino para a tradução.
4. **Veja o resultado:** Clique no botão para traduzir e visualize o texto traduzido no campo de saída.

## Exemplo de Uso do Modelo Seq2Seq

### Função de Decodificação

A função `decode_sentence` é responsável por processar a sequência de texto traduzido:

```python
def decode_sentence(input_seq):
    # Gera a sequência traduzida com base no modelo seq2seq
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence
```

Essa função pode ser personalizada para atender às necessidades específicas do projeto.

## Contribuição

Contribuições são bem-vindas! Siga os passos abaixo:

1. Faça um fork do repositório.
2. Crie um branch para sua feature/bugfix:
   ```bash
   git checkout -b minha-feature
   ```
3. Faça commit das suas alterações:
   ```bash
   git commit -m "Minha nova feature"
   ```
4. Envie para o branch principal:
   ```bash
   git push origin minha-feature
   ```
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.

## Autor

Desenvolvido por [iameosada0](https://github.com/iamrosada).

