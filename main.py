"""Command-line interface for CourseAssistant."""
import argparse
from src.pipeline import CourseAssistant


def main():
    parser = argparse.ArgumentParser(
        description="Course Assistant - Part I CFG Parser + Part II LLM/RAG"
    )
    parser.add_argument("--ingest", action="store_true", help="Build/rebuild the local retrieval index")
    parser.add_argument("--ask", type=str, help="Ask a question about the course")
    args = parser.parse_args()

    if not args.ingest and args.ask is None:
        parser.print_help()
        return

    assistant = CourseAssistant()
    if args.ingest:
        count = assistant.ingest()
        print(f"Đã nạp knowledge base: {count} chunks.")

    if args.ask is not None:
        result = assistant.ask(args.ask)
        print("\n===== SEMANTIC REPRESENTATION =====")
        print(result["semantic"])
        print("\n===== RETRIEVAL =====")
        for item in result["retrieved"]:
            print(
                f"{item['source']} | chunk={item['chunk']} | "
                f"similarity={item['similarity']:.3f}"
            )
        print("\n===== ANSWER =====")
        print(result["answer"])
        print(f"\nGeneration status: {result['generation_status']}")
        print("Đã ghi output/retrieval.txt và output/answer.txt")


if __name__ == "__main__":
    main()
