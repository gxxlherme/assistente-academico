# Caderno — assistente acadêmico com RAG e ChromaDB

Protótipo individual para consultar PDFs e arquivos TXT de disciplinas e documentos institucionais. Interface Streamlit, armazenamento vetorial persistente no ChromaDB e embeddings/geração via Ollama local. Não é um serviço institucional oficial.

## Executar no seu computador

Requer Python 3.11 ou 3.12 e Ollama instalado em https://ollama.com/. Reserve alguns GB para os modelos; o tempo de resposta depende da máquina.

1. Abra esta pasta no terminal e crie um ambiente virtual:

```sh
python -m venv .venv
```

2. Ative o ambiente:

Windows PowerShell: `.venv\Scripts\Activate.ps1`

Linux/macOS: `source .venv/bin/activate`

3. Instale as dependências:

```sh
python -m pip install -r requirements.txt
```

4. Com Ollama aberto, baixe os modelos:

```sh
ollama pull embeddinggemma
ollama pull qwen3:4b
```

Se necessário, inicie o serviço em outro terminal com `ollama serve`. Não precisa executá-lo duas vezes se o aplicativo já o iniciou.

5. Abra o protótipo:

```sh
streamlit run app.py
```

O terminal mostrará o endereço local. Não hospede apenas os arquivos HTML: este projeto precisa de Python, ChromaDB e Ollama em execução.

## Demonstração em sala

1. Importe `exemplos/regulamento-demo.txt` em Institucional e `exemplos/disciplina-demo.txt` em Disciplinas. São dados fictícios, não regras de uma faculdade real.
2. Pergunte: “Segundo o regulamento de demonstração, qual é o prazo de entrega do projeto?” A resposta deve citar 20 de novembro de 2026 e mostrar o documento como fonte.
3. Pergunte: “O que é terceira forma normal segundo o material?” Confira a resposta na fonte.
4. Escolha apenas Disciplinas e pergunte pelo prazo institucional. O filtro deve impedir recuperar o regulamento.
5. Pergunte algo ausente: “Qual é o valor da mensalidade?” Confira se o sistema reconhece a ausência; o comportamento do modelo exige avaliação, não é garantia.
6. Abra “Fontes e trechos recuperados” e explique o caminho pergunta → embedding → busca ChromaDB → contexto → resposta.

## Organização

- `app.py`: interface, upload, biblioteca, conversa e demonstração do RAG.
- `rag.py`: extração por página, divisão em trechos, embeddings, ChromaDB e geração com fontes.
- `data/chroma/`: base local criada automaticamente. Não a envie para o repositório.
- `tests/test_rag.py`: testes do pipeline com vetores determinísticos; não requer Ollama.

Execute os testes com `python -m unittest discover -s tests -v`.

## Decisões e limites

Os trechos possuem até 1.000 caracteres com sobreposição de 180. A busca usa distância cosseno, até cinco resultados e um corte inicial de similaridade de 0,35. Esse corte é uma heurística a calibrar, não uma probabilidade de acerto. A coleção é separada por nome do modelo de embedding; trocas de modelo exigem nova indexação. Não substitua o conteúdo de uma tag de modelo existente sem reindexar.

O histórico é enviado ao gerador, mas a recuperação usa a pergunta atual; repita o assunto nas perguntas de continuação. Referências numéricas inválidas são rejeitadas, mas o suporte factual de uma citação não é validado automaticamente. Respostas sem referências recebem aviso. Prompts reduzem, mas não eliminam, alucinações e injeções de instruções em documentos.

Sem OCR: páginas escaneadas não são compreendidas. Tabelas e colunas complexas podem perder estrutura. A numeração corresponde à página física do PDF, não necessariamente à impressa. TXT usa página 1. Não há autenticação, controle por usuário, backups ou suporte garantido a múltiplos usuários. Use localmente com materiais autorizados. Para remover completamente os dados, pare o app e exclua `data/`.

As respostas são geradas em tempo real: nenhum conteúdo de resposta foi pré-programado. Os arquivos de exemplo são explicitamente fictícios. Downloads iniciais de dependências e modelos precisam de internet; depois disso, o pipeline usa o Ollama local. Para testar qualidade real, use perguntas com respostas conhecidas, perguntas sem resposta e documentos conflitantes, registrando recuperação, citações, fidelidade e latência.

## Validação neste ambiente

Os testes automatizados usam ChromaDB real e um embedder determinístico para verificar persistência, filtros, duplicação e remoção. O teste do gerador usa uma resposta simulada apenas para conferir rejeição de citações inválidas. Isso não valida a qualidade dos modelos reais. A execução completa com Ollama depende de sua instalação e download dos modelos no computador de apresentação.
