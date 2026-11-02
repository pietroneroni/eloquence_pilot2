# Project instructions for Codex

## General rules

- Make only the minimum changes necessary to solve the requested problem.
- Do not hardcode values unless explicitly requested or clearly necessary.
- Do not refactor unrelated code.
- Do not introduce new abstractions, files, classes, modules, or architectural structures unless they are necessary for the requested change.
- Preserve the existing architecture and behavior whenever possible.
- Do not modify unrelated code.

## Response style

After making a change:

- Show only the relevant, short snippet
- Explain in 1–2 lines:
  - the root cause of the problem;
  - why the proposed change fixes it.
- Avoid long explanations unless explicitly requested.

## LLM / prompting priority

Before proposing code changes, first inspect:

1. the prompt sent to the model;
2. system/developer/user messages;
3. contextual information provided to the model;
4. ordering and formatting of that context;
5. expected output format;
6. model parameters when relevant.

When a problem can reasonably be solved by improving the prompt or the information supplied to the model, prefer that solution over adding application-side logic.

When relevant, apply prompting best practices appropriate to the model and architecture being used, including:

- clear and explicit instructions;
- explicit context;
- clear separation between instructions and data;
- precise output-format requirements;
- sufficient task-relevant information;
- avoiding unnecessary or conflicting instructions.

Do not compensate for a weak or incomplete prompt by adding hardcoded heuristics to the application unless explicitly requested.

## Experimental integrity

Because this project measures LLM bias:

- Treat prompts, model inputs, model outputs, sampling parameters, and evaluation logic as part of the experimental methodology.
- Avoid silently changing experimental conditions.
- Clearly point out when a proposed modification could affect comparability with previous experiments.
- Prefer changes that keep experimental conditions reproducible.
- Do not introduce assumptions intended to force or sanitize model outputs unless explicitly requested.