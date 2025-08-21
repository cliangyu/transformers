# GOV Model Implementation Task

## Overview
Implement GOV (GPT-OSS + InternVL3 Vision) - a vision-language model that combines:
- GPT-OSS as the text/language model backbone
- InternVL3's vision processor and encoder for visual understanding

## Key Requirements
1. Create a new model architecture that properly integrates both components
2. Maintain compatibility with HuggingFace transformers architecture
3. Follow existing patterns from both GPT-OSS and InternVL3 implementations
4. Work on a dedicated feature branch
5. Prepare for PR submission to official HuggingFace repo

## Implementation Checklist
- [ ] Create feature branch for development
- [ ] Study GPT-OSS implementation (text model)
- [ ] Study InternVL3 implementation (vision components)
- [ ] Design GOV architecture integration strategy
- [ ] Implement model configuration
- [ ] Implement model architecture
- [ ] Implement processor/tokenizer handling
- [ ] Add model tests
- [ ] Update model documentation
- [ ] Prepare PR with proper documentation

## Code Quality Standards
- Follow HuggingFace transformers coding conventions
- Ensure proper type hints and docstrings
- Include comprehensive tests
- Maintain backward compatibility where applicable
- Run linting and type checking before submission

## Testing Commands
```bash
# Run tests for the new model
pytest tests/models/gov/

# Run linting
make style
make quality

# Type checking (if available)
make type
```

## Important Files to Reference
- GPT-OSS: src/transformers/models/gpt_oss/
- InternVL3: src/transformers/models/internvl3/ (if exists, otherwise check internvl/internvl2)
- Model registration: src/transformers/__init__.py
- Auto classes: src/transformers/models/auto/