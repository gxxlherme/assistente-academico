import tempfile
import unittest
from rag import AcademicRAG, chunk_text, Ollama

class FakeEmbedder:
    embedding = 'deterministic-test'
    def embed(self, texts):
        return [[1., 0., 0.] if 'matrícula' in t.lower() else [0., 1., 0.] for t in texts]

class RAGTests(unittest.TestCase):
    def test_chunking_terminates_and_preserves_text(self):
        text = 'abcdefghijklmnop' * 300
        chunks = chunk_text(text, 100, 20)
        self.assertEqual(chunks[0], text[:100])
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        reconstructed = chunks[0] + ''.join(c[20:] for c in chunks[1:])
        self.assertEqual(reconstructed, text)
        with self.assertRaises(ValueError):
            chunk_text(text, 100, 100)

    def test_persistence_filter_duplicate_and_delete(self):
        with tempfile.TemporaryDirectory() as path:
            rag = AcademicRAG(FakeEmbedder(), path)
            data = 'A matrícula precisa ser solicitada na secretaria.'.encode()
            self.assertEqual(rag.ingest('regras.txt', data, 'Institucional')[0], 1)
            self.assertEqual(rag.ingest('regras.txt', data, 'Institucional')[0], 0)
            rag.ingest('aula.txt', 'Banco de dados e normalização.'.encode(), 'Disciplinas')
            reopened = AcademicRAG(FakeEmbedder(), path)
            hits = reopened.retrieve('matrícula', 'Institucional')
            self.assertEqual(hits[0]['source'], 'regras.txt')
            self.assertEqual(hits[0]['page'], 1)
            self.assertEqual(reopened.retrieve('matrícula', 'Disciplinas'), [])
            self.assertEqual(len(reopened.documents()), 2)
            reopened.delete_document(hits[0]['document_id'])
            self.assertEqual(reopened.retrieve('matrícula', 'Institucional'), [])

    def test_invalid_citation_is_rejected(self):
        class Response:
            def raise_for_status(self): pass
            def json(self): return {'message': {'content': 'O prazo é amanhã [99].'}}
        class Session:
            def post(self, *args, **kwargs): return Response()
        model = Ollama()
        model.session = Session()
        hits = [{'source': 'demo.txt', 'page': 1, 'category': 'Institucional', 'text': 'Texto.'}]
        self.assertIn('referências inválidas', model.generate('Prazo?', hits))

if __name__ == '__main__':
    unittest.main()
