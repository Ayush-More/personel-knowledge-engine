from knowledge_engine.context_builder import build_context
from knowledge_engine.models import DocumentChunk , GeneratedAnswer , SourceCitation

class MockAnswerGeneration:
     """Test answer-generation flow without an external LLM."""
     def generation(
               question: str,
               result: list[tuple[DocumentChunk , float]]
     )-> GeneratedAnswer:
          if not question.strip():
               raise ValueError("Question must not be empty")

          if not result:
               GeneratedAnswer(
                    question=question,
                    answer="I dont have enough information in the available ",
                    citations=[],
                    insufficient_evidence=True
               )
          context = build_context(result)

          answer = (
            "I found relevant information in your notes. "
            "Review the cited source chunks for the evidence."
          )

          return GeneratedAnswer(
               question=question,
               answer=answer,
               citations=context,
               insufficient_evidence=not bool(context.strip())
          )

        

