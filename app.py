import os
import time
import requests
import streamlit as st
from rag import Ollama, AcademicRAG

st.set_page_config(page_title="Caderno — Seu espaço de estudo", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {color-scheme:light}
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif!important}
.stApp,[data-testid="stAppViewContainer"]{background:#fafbf9;color:#202722}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#f2f5f0;border-right:1px solid #e4e9e2}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.65rem}
.block-container{max-width:1040px;padding-top:2.1rem;padding-bottom:5rem}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.045em;color:#202923}
h1{font-weight:800!important;font-size:2.35rem!important;line-height:1.2!important}
h2{font-size:1.45rem!important;font-weight:750!important}
h3{font-size:1.07rem!important;font-weight:700!important}
p,li,label{color:#4b574e}
[data-testid="stCaptionContainer"]{color:#7c8980!important}
[data-testid="stChatMessage"]{background:#fff;border:1px solid #e7ece5;border-radius:18px;padding:1.1rem 1.3rem;margin-bottom:12px;box-shadow:0 3px 16px rgba(30,50,30,.025)}
[data-testid="stChatInput"]{border-radius:16px!important;border:1px solid #dce5da!important;background:#fff!important}
.stButton>button[kind="primary"]{background:#2f644b;border-color:#2f644b;color:white}
.stButton>button{border-radius:11px!important;font-weight:650!important;min-height:2.65rem;border-color:#dbe4da;background:white}
.stButton>button:hover{border-color:#8fae9b;color:#245c40}
[data-baseweb="select"]>div,[data-testid="stFileUploader"] section{border-radius:12px!important;border-color:#dce5da!important;background:#fff}
[data-testid="stTabs"] button{font-weight:650!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:#367457}
hr{border-color:#e2e9e0!important}
[data-testid="stExpander"]{background:#fff;border:1px solid #e5ebe3;border-radius:12px}
.hero{background:linear-gradient(120deg,#eaf3e9,#f6f8f0 75%);border:1px solid #e1eade;border-radius:22px;padding:32px 35px;margin:6px 0 27px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:#397657;margin-bottom:13px}
.hero-title{font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-.05em;font-size:35px;line-height:1.16;color:#1e3327;margin:0 0 12px}
.hero-desc{font-size:15px;line-height:1.7;color:#667568;max-width:570px;margin:0}
.brand{font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;letter-spacing:-.055em;color:#244a36;margin:8px 0 0}
.brand-sub{font-size:12px;color:#7b897e;margin-bottom:22px}
.section-kicker{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#829185;font-weight:800;margin:15px 0 7px}
.empty{padding:33px 24px;border:1px dashed #cbd9cc;background:#fff;border-radius:17px;text-align:center;color:#718174;margin-top:14px}
.doc-icon{display:inline-block;background:#e8f2e9;color:#387554;padding:7px 11px;border-radius:10px;font-size:13px;font-weight:800}
.small-note{font-size:12px;color:#87958a}
@media(max-width:700px){.hero{padding:24px 20px}.hero-title{font-size:27px}.block-container{padding-top:1rem}}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def build_rag(url, embedding, model):
    return AcademicRAG(Ollama(url, embedding, model))

with st.sidebar:
    st.markdown('<div class="brand">◈ Caderno</div><div class="brand-sub">Seu espaço inteligente de estudos</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Sua biblioteca</div>', unsafe_allow_html=True)
    category = st.selectbox('Categoria dos novos arquivos', ['Disciplinas', 'Institucional'])
    files = st.file_uploader('Adicionar documentos', type=['pdf','txt'], accept_multiple_files=True, help='PDFs escaneados precisam de OCR.')
    with st.expander('Configurações da inteligência artificial'):
        url = st.text_input('Servidor Ollama', os.getenv('OLLAMA_URL','http://localhost:11434'))
        embedding = st.text_input('Modelo de embeddings', os.getenv('EMBEDDING_MODEL','embeddinggemma'))
        model = st.text_input('Modelo de respostas', os.getenv('CHAT_MODEL','qwen3:4b'))
        min_similarity = st.slider('Similaridade mínima', 0.0, 0.9, 0.35, 0.05, help='Filtro experimental; não é uma probabilidade de acerto.')
    rag = build_rag(url, embedding, model)
    if st.button('＋  Adicionar à biblioteca', disabled=not files, use_container_width=True, type='primary'):
        for file in files:
            try:
                with st.spinner(f'Processando {file.name}…'):
                    count, skipped = rag.ingest(file.name, file.getvalue(), category)
                st.success(f'{file.name}: {count} trechos adicionados.' if count else f'{file.name}: já estava na biblioteca.')
                if skipped:
                    st.warning(f'{skipped} página(s) sem texto não foram indexadas.')
            except (requests.RequestException, ValueError, KeyError) as exc:
                st.error(f'Erro ao adicionar {file.name}: {exc}')
            except Exception:
                st.error(f'Não foi possível ler {file.name}. Verifique o arquivo.')
    st.divider()
    st.caption('🔒 Seus arquivos e modelos ficam neste computador. Sem login ou sincronização em nuvem.')

st.markdown('''<div class="hero"><div class="eyebrow">✦ Seu assistente acadêmico</div><div class="hero-title">Aprender fica mais simples<br>com o material certo.</div><p class="hero-desc">Converse com seus documentos, encontre explicações e estude com respostas baseadas nas suas próprias fontes.</p></div>''', unsafe_allow_html=True)

chat_tab, library_tab, about_tab = st.tabs(['✦  Assistente', '▤  Biblioteca', 'ⓘ  Sobre'])

with library_tab:
    st.subheader('Seus materiais')
    st.caption('Gerencie os documentos que o assistente utiliza nas respostas.')
    docs = rag.documents()
    if not docs:
        st.markdown('<div class="empty">📄<br><br><b>Sua biblioteca está vazia</b><br>Adicione um PDF ou TXT usando a barra lateral.</div>', unsafe_allow_html=True)
    for doc in docs:
        left, right = st.columns([5,1], vertical_alignment='center')
        with left:
            st.markdown(f"**📄 {doc['source']}**")
            st.caption(f"{doc['category']} · {doc['chunks']} trechos indexados")
        with right:
            if st.button('Remover', key=doc['document_id']):
                rag.delete_document(doc['document_id'])
                st.rerun()
        st.divider()

with about_tab:
    st.subheader('Como o Caderno funciona')
    st.write('O Caderno usa **RAG (geração aumentada por recuperação)** para consultar os seus documentos antes de produzir uma resposta.')
    st.markdown('**01 · Preparação** — Extrai o texto dos PDFs e TXT e divide em trechos.\n\n**02 · Indexação** — O Ollama cria embeddings e o ChromaDB guarda os trechos e suas origens.\n\n**03 · Busca** — Sua pergunta encontra até cinco trechos relevantes da categoria selecionada.\n\n**04 · Resposta** — O modelo gera uma explicação com referências ao material.')
    st.info('A IA pode cometer erros. Sempre confira as fontes, especialmente em assuntos importantes.')
    st.caption('Limitações: sem OCR para PDFs digitalizados; tabelas complexas podem não ser lidas corretamente. Perguntas de continuação devem mencionar novamente o assunto, pois a busca usa a pergunta atual.')

with chat_tab:
    left, right = st.columns([4,1], vertical_alignment='bottom')
    with left:
        scope = st.selectbox('Pesquisar em', ['Ambos', 'Disciplinas', 'Institucional'])
    with right:
        if st.button('↻ Nova conversa', use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    st.session_state.setdefault('messages', [])
    if not st.session_state.messages:
        st.markdown('<div class="empty">✦<br><br><b>Por onde vamos começar?</b><br>Envie seus materiais e faça uma pergunta sobre eles.<br><span class="small-note">Ex.: “Resuma o capítulo 2” ou “Explique normalização com base na apostila”.</span></div>', unsafe_allow_html=True)

    def show_sources(hits):
        if hits:
            with st.expander(f'📚 Ver fontes consultadas ({len(hits)})'):
                for i, hit in enumerate(hits, 1):
                    st.markdown(f"**[{i}] {hit['source']} · página {hit['page']}**")
                    st.caption(f"{hit['category']} · Similaridade: {hit['similarity']:.3f} (não é confiança)")
                    st.text(hit['text'])

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            show_sources(message.get('sources', []))
    question = st.chat_input('Pergunte alguma coisa sobre seus materiais…')
    if question:
        history = st.session_state.messages.copy()
        st.session_state.messages.append({'role':'user','content':question})
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            hits = []
            answer = ''
            start = time.monotonic()
            try:
                with st.status('Consultando a biblioteca...', expanded=True) as status:
                    st.write('Etapa 1/2: procurando trechos no ChromaDB e gerando o embedding da pergunta...')
                    hits = rag.retrieve(question, scope, min_similarity=min_similarity)
                    st.write(f'Busca concluída: {len(hits)} trecho(s) relevante(s) em {time.monotonic()-start:.1f}s.')
                    if not hits:
                        answer = 'Não encontrei trechos suficientes nos materiais selecionados. Confira sua biblioteca, tente outra categoria ou reformule a pergunta.'
                        status.update(label='Busca concluída: nenhum trecho relevante', state='complete', expanded=False)
                    else:
                        st.write('Etapa 2/2: aguardando a resposta do Ollama...')
                        answer = rag.ollama.generate(question, hits, history)
                        status.update(label='Resposta gerada', state='complete', expanded=False)
                st.markdown(answer)
                st.caption(f'Resposta em {time.monotonic()-start:.1f}s')
            except Exception as exc:
                answer = 'Ocorreu um erro ao consultar os materiais ou a IA local.'
                st.error(answer)
                st.caption(f'Falha após {time.monotonic()-start:.1f}s')
                with st.expander('Detalhes técnicos do erro'):
                    st.code(f'{type(exc).__name__}: {exc}')
            show_sources(hits)
            st.session_state.messages.append({'role':'assistant','content':answer,'sources':hits})
