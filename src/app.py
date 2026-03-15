from rag_chain import build_rag_chain


def main():

    print("RAG Document Assistant Started")
    print("Type 'exit' to quit\n")

    qa_chain = build_rag_chain()

    while True:

        question = input("User: ")

        if question.lower() == "exit":
            break

        result = qa_chain.invoke({
            "question": question
        })

        answer = result["answer"]

        print("\nAssistant:", answer)
        print()


if __name__ == "__main__":
    main()

    