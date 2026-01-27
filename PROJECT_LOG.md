# Project Log — Agente  Educacional

## 🎯 Objetivo Atual
Construir um agente educacional que atue como **monitor inteligente de uma disciplina**, capaz de:
- Utilizar exclusivamente os materiais fornecidos pelo professor;
- Responder apenas dentro do domínio de determinado pelos documentos;
- **Não fornecer respostas diretas**, atuando de forma pedagógica e orientativa.

# ➡️ Log Entries

## 📅 Data
- **2026-01-27**

### 📌 Status do Projeto
Primeira versão funcional de um sistema RAG educacional baseada em PDFs, com pipeline completo de ingestão, recuperação e geração controlada de respostas.  
A implementação atual é considerada **dummy**, porém já valida a arquitetura geral pretendida para o projeto.

### ⚠️ Limitações Conhecidas

- Não há:
  - Persistência de embeddings;
  - Metadados estruturais (capítulos, seções, tipo de documento);
  - Diferenciação entre PDFs textuais e PDFs de slides;
  - Mecanismo de avaliação da qualidade da resposta pedagógica.
- O campo documents no estado ainda não é explorado de forma significativa.
- Recuperação baseada apenas em similaridade textual simples.
- Respostas do modelo são repetitivas
- O agente funciona apenas para SysADL, se trocar os PDFs ele buga

### 🔜 Próximos Passos

- Evoluir o processo de ingestão:
  - Classificação de tipo de documento (artigo, slide, capítulo);
  - Enriquecimento de metadados por chunk.
- Refinar o controle pedagógico:
  - Melhor separação entre decisão de recuperação e decisão pedagógica.
- Persistência do vector store.
- Avaliação do comportamento do agente com alunos reais ou cenários simulados.