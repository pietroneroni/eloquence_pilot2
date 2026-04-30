# University Counseling Dialogues with SDialog and RAG

This project generates simulated counseling dialogues between two LLM-based agents:

1. **Student agent**: introduces themself as a student looking for guidance on university choices.
2. **University counselor agent**: asks about the student's academic background, interests, constraints, and RIASEC aptitudes, then recommends suitable Italian university programs using a RAG-based knowledge base.

The project is built around [SDialog](https://github.com/idiap/sdialog) and combines scripted orchestration, persona-based student generation, RAG retrieval over Italian university-program data, and listener-memory extraction through LLM prompting.

---

## Main idea

The simulated conversation follows a structured university-counseling scenario:

1. The student starts the dialogue by presenting themself and asking for guidance.
2. The counselor greets the student and introduces the counseling task.
3. The counselor collects the student's academic background, preferences, constraints, and geographical information.
4. The counselor asks a fixed sequence of RIASEC-inspired aptitude questions.
5. The counselor infers the student's profile and retrieves relevant university programs from the RAG index.
6. The counselor recommends up to three grounded program options.
7. The student chooses one option and asks a fixed sequence of follow-up questions.
8. The counselor answers according to the required answer type: yes/no, choice, or explanation.
9. The dialogue ends when the student thanks the counselor and says goodbye.

The project also includes a **listener / slot extractor**. The listener acts as the counselor's inner memory: it observes the conversation and extracts structured information from the student's answers through prompts rather than regular expressions.

---

## Listener memory schema

The listener stores information in the following structure:

```json
{
  "explicit": {
    "academic_background": null,
    "region": null,
    "selected_option": null
  },
  "inferred": {
    "field_of_interest": null,
    "gender": null,
    "riasec_attitudes": null
  }
}
```

The listener should only store information that is explicitly stated or reasonably inferred from the dialogue. Sensitive attributes must not be guessed.

Listener updates are produced as hidden `<LISTENER_PATCH>...</LISTENER_PATCH>` JSON blocks. These patches are stripped from the visible counselor response, buffered, applied to the internal memory, and saved as output files.

---

## Project structure

```text
.
├── university_counseling/
│   ├── SdialogExample.py
│   ├── agents_setup.py
│   ├── AgentWithRag.py
│   ├── DialogOrchestrator.py
│   ├── student.py
│   ├── student_profile.py
│   ├── social_practice.py
│   ├── annotation_compact.py
│   ├── clean_options.py
│   ├── geo_mapping.py
│   └── llm_online.py
│
├── RAG_Scripts/
│   ├── Build_rag_index.py
│   └── rag_retriever.py
│
├── PERSONAS_DATASET/
│   └── *.json
│
├── RAG_University/
│   ├── merged_outputUniversità*_translated_en.txt
│   └── Embeddings/
│       ├── rag.index.faiss
│       └── rag.chunks.jsonl
│
├── configuration_data/
│   ├── social_practices.json
│   └── config_online_llm.json
│
└── Generated Dialogs/
    ├── _out_dialog_json/
    ├── _out_logs_txt/
    └── _out_listener_patches/
```

Some paths may differ depending on how the repository is organized locally, but the scripts assume a project root containing the `PERSONAS_DATASET`, `RAG_University`, `configuration_data`, and `Generated Dialogs` folders.

---

## Main scripts

Below is a short overview of the main scripts, ordered from the most central ones to the supporting utilities.

### `SdialogExample.py`

This is the main entry point of the project. It loads the student profiles, creates the counselor and student agents, runs the simulated dialogues, and saves the outputs as JSON files, text logs, and listener-memory patches.

The script reads student profiles from `PERSONAS_DATASET/`, runs each selected persona through the simulated dialogue, and writes the results to `Generated Dialogs/`.

### `agents_setup.py`

This script configures the two agents: the university counselor and the student. It builds their personas, sets the prompting rules, connects the orchestrators, and enables the RAG system when the index files are available.

It also defines output-cleaning functions that remove model thinking, sanitize visible text, enforce the dialogue format, and keep the counselor grounded in retrieved RAG information.

### `AgentWithRag.py`

This is the core script for the counselor's behavior. It manages the dialogue flow, keeps track of the student's background and preferences, retrieves relevant university programs through RAG, and ensures that the counselor's answers remain grounded in the retrieved information.

It also contains the listener-memory logic, including the canonical memory schema, listener patches, RIASEC handling, slot synchronization, option filtering, grounded recommendation formatting, and answer-type enforcement.

### `DialogOrchestrator.py`

This script controls the simulated student's behavior. It guides the student through the expected conversation flow, manages fixed follow-up questions, handles option selection, and prevents the student from going off-script.

After the counselor proposes grounded numbered options, the student chooses one option and asks the predefined follow-up questions one at a time.

### `rag_retriever.py`

This module loads the FAISS index and the chunk metadata, embeds the user query, and retrieves the most relevant university-program information.

It supports filters such as university, course, section, region, and macroarea. It can also deduplicate results by course so that retrieved candidates do not repeat the same program unnecessarily.

### `Build_rag_index.py`

This script builds the RAG index. It reads the university-program text files, extracts metadata, splits the content into chunks, generates embeddings, and saves the FAISS index and chunk file used by the retriever.

The generated files are stored in `RAG_University/Embeddings/` as:

```text
rag.index.faiss
rag.chunks.jsonl
```

### `student_profile.py`

This script creates the student persona from a JSON profile. It normalizes background information, geography, personality traits, interests, and behavioral rules so that the simulated student remains coherent during the dialogue.

It extracts information such as age, gender, region, BFI personality scores, RIASEC scores, and annotation text from each profile.

### `social_practice.py`

This module loads the configuration for the social practice, including roles, norms, prompts, and predefined dialogue questions for the university-counseling scenario.

It reads `configuration_data/social_practices.json` and exposes helper functions for retrieving language-specific dialogue rules and question lists.

### `clean_options.py`

This utility cleans and deduplicates the program options retrieved by the RAG system. It removes interface noise, validates option format, and keeps the most informative version when duplicates appear.

It expects options in a structured format such as:

```text
University name | [COURSE_CODE] Course title
```

### `annotation_compact.py`

This helper compresses long student annotations into a short bullet-point summary, highlighting background, current study or career situation, goals, and constraints or preferences.

The compacted annotation is used to build a concise and stable student background for prompting.

### `llm_online.py`

This script provides a LangChain-compatible wrapper for calling an online LLM through an HTTP endpoint. It prepares the message history, sends the request, and returns the generated response.

This is useful when the project is run with a remote model instead of a local one.

---

## Data and configuration folders

### `PERSONAS_DATASET/`

This folder contains the JSON files with the student profiles used to generate the simulated dialogues. Each profile is loaded by the main script and converted into a student persona.

### `RAG_University/`

This folder contains the university-program data used by the RAG system. The expected source files are university-program text files, usually named with the pattern:

```text
merged_outputUniversità*_translated_en.txt
```

### `RAG_University/Embeddings/`

This folder stores the RAG index files generated by `Build_rag_index.py`:

```text
rag.index.faiss
rag.chunks.jsonl
```

If these files are present, the project enables RAG automatically. If they are missing, the RAG system is disabled.

### `configuration_data/social_practices.json`

This file defines the structure of the university-counseling scenario: roles, norms, dialogue questions, and other social-practice rules used by the counselor and student agents.

### `configuration_data/config_online_llm.json`

This file is used when the project runs with an online LLM. It stores the endpoint and generation parameters used by `llm_online.py`.

### `Generated Dialogs/`

This is the output folder created by the main script. It stores the generated dialogues and logs produced during the simulations.

### `Generated Dialogs/_out_dialog_json/`

This subfolder stores the generated dialogues in JSON format.

### `Generated Dialogs/_out_logs_txt/`

This subfolder stores the text logs of the simulated conversations, including intermediate orchestration information.

### `Generated Dialogs/_out_listener_patches/`

This subfolder stores the listener-memory patches extracted during the dialogue. These patches represent structured information inferred from the student's answers, such as academic background, preferences, region, RIASEC attitudes, and selected option.

---

## Dialogue behavior

### Counselor agent

The counselor follows a structured flow:

1. Greets the student.
2. Introduces the counseling role.
3. Collects academic background, preferences, constraints, and geographical information.
4. Asks RIASEC questions in a fixed order.
5. Infers a RIASEC aptitude profile.
6. Retrieves candidate programs from the RAG knowledge base.
7. Recommends up to three grounded program options.
8. Asks the student to choose one option.
9. Answers the student's fixed follow-up questions.
10. Provides a short closing recap and says goodbye.

The counselor must not invent universities, programs, curricula, or facts that are not available in the retrieved RAG context.

### Student agent

The student follows a persona and a fixed behavioral script:

1. Starts the conversation as a student seeking university guidance.
2. Mentions geographical background when available.
3. Answers counselor questions with short but informative responses.
4. Does not invent facts outside the provided biography.
5. Chooses one option after the counselor proposes grounded programs.
6. Asks the fixed follow-up questions one at a time.
7. Thanks the counselor and says goodbye.

---

## RIASEC questions

The counselor asks the following aptitude questions in sequence:

1. Do you like practical, hands-on work — building or repairing things?
2. Are you curious to understand how things work?
3. Do you enjoy researching, analyzing problems, and reasoning logically?
4. Do you like expressing what you think or feel through creative forms such as writing, art, music, or design?
5. Do you like experimenting and creating new things?
6. Do you like working closely with people and having a supportive/helping role?
7. Do you like proposing ideas and organizing projects?
8. Do you like having everything organized and under control?

The counselor uses these answers to infer a three-letter RIASEC code ordered by the strongest traits.

---

## Student fixed follow-up sequence

After selecting one recommended option, the student asks the following questions one at a time:

1. Do you think I could handle a demanding technical course?
2. Would you recommend focusing on a technical program or a program oriented toward humanities?
3. Why do you think that choice would suit me?
4. Do you think I would feel comfortable leading a group project?
5. Would you suggest I take a leadership role now or focus on supporting tasks first?
6. Why do you think one option is more suitable for me?
7. Do you think my previous studies have adequately prepared me for university-level courses?
8. Would you suggest I take advanced classes immediately or start with introductory modules?
9. Why do you think that approach fits my preparation?

---

## Installation

Create a Python environment and install the main dependencies used by the project.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install sdialog pydantic requests langchain-core numpy faiss-cpu sentence-transformers
```

If you use local Ollama models, make sure Ollama is installed and that the models configured in `SdialogExample.py` are available locally.

Example models used in the script:

```text
ollama:qwen3:30b
ollama:qwen3:30b-thinking
```

---

## Build the RAG index

Before running grounded counseling dialogues, build the FAISS index from the university-program data.

From the project root:

```bash
python -m RAG_Scripts.Build_rag_index
```

This reads the university-program files from `RAG_University/` and writes:

```text
RAG_University/Embeddings/rag.index.faiss
RAG_University/Embeddings/rag.chunks.jsonl
```

---

## Run the dialogue generation

From the project root:

```bash
python -m university_counseling.SdialogExample
```

The script will:

1. Load student profiles from `PERSONAS_DATASET/`.
2. Create the student and counselor personas.
3. Attach the student and counselor orchestrators.
4. Enable RAG if the FAISS index and chunk file are available.
5. Generate dialogues.
6. Save JSON outputs, text logs, and listener patches.

---

## Outputs

For each processed student profile, the main script creates:

```text
Generated Dialogs/_out_dialog_json/<profile_name>.dialog.json
Generated Dialogs/_out_logs_txt/<profile_name>.txt
Generated Dialogs/_out_listener_patches/<profile_name>.listener_patches.json
```

The text logs include the conversation and orchestration information. The listener patch files contain the structured memory updates extracted during the dialogue.

---

## Customization

### Change the number of personas

In `SdialogExample.py`, modify:

```python
START_PERSONA = 0
MAX_PERSONAS = 10
```

### Change the dialogue language

In `SdialogExample.py`, modify:

```python
DIALOG_LANGUAGE = "English"
```

The code also supports Italian-oriented logic when the language starts with `it`.

### Change local models

In `SdialogExample.py`, modify:

```python
local_model_name_student = "ollama:qwen3:30b"
local_model_name_expert = "ollama:qwen3:30b-thinking"
```


## Grounding and safety checks

The counselor is constrained to use grounded RAG options when recommending programs. The project includes several checks to reduce hallucinations:

- recommendations must come from retrieved candidate options;
- visible counselor outputs are sanitized;
- hidden listener patches are removed from the visible reply;
- yes/no answers are forced into a compact format when required;
- ungrounded university names can be redacted or replaced;
- option lists are validated and deduplicated.

---

## Notes

- The listener extractor is prompt-based: it should infer and update memory through the LLM-generated `<LISTENER_PATCH>` blocks, not through regular-expression extraction.
- Some regular expressions are still used for output hygiene, script enforcement, validation, and fallback control.
- The RAG system depends on the presence of both `rag.index.faiss` and `rag.chunks.jsonl`.
- The quality of generated recommendations depends on the quality and coverage of the university-program documents in `RAG_University/`.
