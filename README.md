# Transformer Decoder From Scratch

Projeto desenvolvido para o **Laboratório 3 da disciplina Tópicos em Inteligência Artificial**.

O objetivo deste trabalho é implementar partes importantes do **Decoder do Transformer** utilizando apenas **Python e NumPy**, sem usar bibliotecas de deep learning.

---

# Estrutura do projeto

Arquivos principais:

- `main.py` → executa o programa
- `mask.py` → implementação da máscara causal (look-ahead mask)
- `cross_attention.py` → implementação da cross attention
- `utils.py` → funções auxiliares (se utilizadas)

---

# O que foi implementado

Este projeto implementa três componentes principais do Decoder:

### 1. Causal Mask

Criação de uma máscara que impede o modelo de olhar para palavras futuras durante a geração.

### 2. Cross Attention

Implementação da atenção entre o **Encoder e o Decoder**, permitindo que o Decoder consulte a informação produzida pelo Encoder.

### 3. Loop de geração auto-regressiva

Simulação da geração de texto token por token até que o modelo produza o token `<EOS>`.

---

# Como executar o projeto

## 1. Instalar dependências

pip install numpy


---

## 2. Rodar o programa

### No Mac ou Linux

python3 main.py

### No Windows

python main.py


---

# Saída esperada

O programa deve mostrar algo parecido com:

Causal Mask:
[[-0.e+00 -1.e+09 -1.e+09 -1.e+09 -1.e+09]
[-0.e+00 -0.e+00 -1.e+09 -1.e+09 -1.e+09]
[-0.e+00 -0.e+00 -0.e+00 -1.e+09 -1.e+09]
[-0.e+00 -0.e+00 -0.e+00 -0.e+00 -1.e+09]
[-0.e+00 -0.e+00 -0.e+00 -0.e+00 -0.e+00]]
Cross Attention Output Shape:
(1, 4, 512)
Simulando geração de texto:
['<START>', 'O']
['<START>', 'O', '<EOS>']

Essa saída mostra que:

- A máscara causal foi criada corretamente
- A cross attention foi executada
- O loop de geração auto-regressiva gerou uma sequência até o token `<EOS>`





