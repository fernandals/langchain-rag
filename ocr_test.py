import fitz
import re

# ========= CONFIG =========
PDF_PATH = "pdfs/SAIA-Chapter12.pdf"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ========= 1. ABRIR PDF =========
doc = fitz.open(PDF_PATH)

print(doc.name)
print(doc.page_count)
print(doc.metadata)
print(doc.chapter_count)
print(doc.is_pdf)
print(doc.language)

print(type(doc[0]))
print(doc[0].artbox)
print(doc[0].mediabox)
print(doc[0].mediabox_size)


full_text = ""
page_offsets = []  # vai guardar onde cada página começa no texto global

for page_number, page in enumerate(doc):
    text = page.get_text("text")
    page_offsets.append({
        "page": page_number + 1,
        "start": len(full_text)
    })
    full_text += text + "\n"

print("Total de páginas:", len(doc))
print("Total de caracteres:", len(full_text))


# ========= 2. DETECÇÃO DE SEÇÕES =========

SECTION_REGEX = re.compile(
    r"""
    ^(?P<number>\d+(?:\.\d+)*)      # número 12 ou 12.3
    \s{1,3}
    (?P<title>[A-Z][A-Za-z0-9\s\-\(\),]{3,80})  # título controlado
    $
    """,
    re.MULTILINE | re.VERBOSE
)

sections = []

for match in SECTION_REGEX.finditer(full_text):
    number = match.group("number")
    title = match.group("title").strip()

    # ===== FILTROS INTELIGENTES =====

    # ignora se for muito longo
    if len(title.split()) > 12:
        continue

    # ignora se terminar com hífen (linha quebrada)
    if title.endswith("-"):
        continue

    # ignora se parecer frase normal
    if title.endswith("."):
        continue

    level = number.count(".") + 1

    sections.append({
        "number": number,
        "title": title,
        "level": level,
        "start_index": match.start()
    })

print("Seções detectadas:", len(sections))


# ========= 3. FUNÇÃO AUXILIAR PARA ACHAR PÁGINA =========

def find_page_for_index(index):
    for i in range(len(page_offsets)):
        if i + 1 < len(page_offsets):
            if page_offsets[i]["start"] <= index < page_offsets[i+1]["start"]:
                return page_offsets[i]["page"]
        else:
            return page_offsets[i]["page"]
    return None


# ========= 4. CHUNKING COM METADATA ESTRUTURAL =========

chunks = []
current_section = None
section_pointer = 0
current_chapter = None

i = 0
while i < len(full_text):

    chunk_text = full_text[i:i+CHUNK_SIZE]

    while (
        section_pointer < len(sections) and
        sections[section_pointer]["start_index"] <= i
    ):
        s = sections[section_pointer]

        if s["level"] == 1:
            current_chapter = f'{s["number"]} {s["title"]}'
            current_section = None

        elif s["level"] == 2:
            current_section = f'{s["number"]} {s["title"]}'

        section_pointer += 1

    page_number = find_page_for_index(i)

    chunk = {
        "text": chunk_text,
        "metadata": {
            "chapter": current_chapter,
            "section": current_section,
            "page": page_number
        }
    }

    chunks.append(chunk)

    i += CHUNK_SIZE - CHUNK_OVERLAP


# ========= 5. INSPEÇÃO =========

print("\nExemplo de chunk 1:\n")
print("Capítulo:", chunks[0]["metadata"]["chapter"])
print("Seção:", chunks[0]["metadata"]["section"])
print("Página:", chunks[0]["metadata"]["page"])
print("Texto:")
print(chunks[0]["text"])
print("\nTotal de chunks:", len(chunks))

print("\nExemplo de chunk 2:\n")
print("Capítulo:", chunks[1]["metadata"]["chapter"])
print("Seção:", chunks[1]["metadata"]["section"])
print("Página:", chunks[1]["metadata"]["page"])
print("Texto:")
print(chunks[1]["text"])
print("\nTotal de chunks:", len(chunks))

print("\nExemplo de chunk 3:\n")
print("Capítulo:", chunks[8]["metadata"]["chapter"])
print("Seção:", chunks[8]["metadata"]["section"])
print("Página:", chunks[8]["metadata"]["page"])
print("Texto:")
print(chunks[8]["text"])
print("\nTotal de chunks:", len(chunks))
