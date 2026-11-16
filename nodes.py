import os
from google import genai
from langgraph.types import Command, interrupt
from langgraph.graph import END
from ingestion import ensemble_retriever
from dotenv import load_dotenv

load_dotenv()
from state import MyState

client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))


def intent_classification(state: MyState) -> str:
    """Classifies the intent of the User's input into one of the predefined categories.
    Information query , IT support request, IT support ticket with explicit ticket."""

    interactions = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=f"""
        You are an Intent Classifier. Your task is to classify the intent of the user's input {state["messages"][-1].content}
        into one of the following categories:
        1. Information query: The user is seeking information or knowledge about a specific topic.
        2. IT support request: The user is requesting assistance or support for an IT-related issue.
        3. IT support request with explicit ticket: The user wants to create an IT support ticket, the query should contain explicit "ticket" word.
        JUST PROVIDE ONE OF THE 3 CATEGORIES, DONOT PROVIDE EXTRA INFORMATION
        """,
    )
    return {
        "messages": [interactions.output_text],
        "intent": interactions.output_text,
        "user_id": state.get("user_id"),
    }


def get_relevant_documents(state: MyState):
    """Get the relevant documents using Hybrid Search retriever."""

    user_input = state["messages"][-2].content
    # print("User input for the relevan docs node : ", user_input)

    docs = ensemble_retriever.invoke(user_input)
    interactions = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=f"""
        You are a precise Request-Response Agent. Your goal is to answer the user's request accurately using ONLY the provided context documents.
            ### Input Data
            <documents>
            {docs}
            </documents>

            <user_input>
            {user_input}
            </user_input>

            ### Instructions
            1. **Analyze the Request:** Carefully read the `<user_input>` to identify the core question or task.
            2. **Search the Context:** Locate all relevant facts, procedures, and details inside `<documents>`.
            3. **Draft a Step-by-Step Response:** Present the solution clearly in sequential, numbered steps. Keep each step focused on a single action or point.
            4. **Stay Strictly Grounded:** Base your entire answer solely on `<documents>`. Do not assume, extrapolate, or use outside knowledge.
            5. **Handle Missing Information:** If `<documents>` do not contain enough information to answer the request, state clearly: *"I do not have enough information in the provided documents to answer this request."* Do not attempt to guess.

            ### Output Format (Strict Plain Text)
            - Write the entire response in plain text only.
            - Do NOT use any Markdown formatting symbols, including asterisks (** or *), hashtags (#), backticks (`), or underscores (_).
            - Begin with a 1-sentence summary of the answer.
            - Follow with a numbered list of steps (1., 2., 3.) using plain text only.
            - Example of correct formatting:
            1. Review Troubleshooting Steps: Before raising a ticket, ensure you have tried standard troubleshooting...
        """,
    )
    print("Similar documents are: ",interactions.output_text)
    return {"messages": [interactions.output_text], "user_id": state.get("user_id")}


def ticket_node(state: MyState):
    intent = state["intent"]
    if intent == "Information query" or intent == "IT support request":
        return Command(update={"messages": ""}, goto=END)

    else:
        user_feedback = interrupt(
            "Are the provided documents Relevant? Please respond with Relevant/Not Relevant"
        )
        if user_feedback == "Resolved":
            return Command(update={"messages": ""}, goto=END)

        else:
            interactions = client.interactions.create(
                model="gemini-3.5-flash-lite",
                input=f"""You are a summarization agent. You take input of {state["messages"][-3].content} 
                and then generate a json file which contains the fields like : etc..""",
            )
            output_json = interactions.output_text
            confirm_ticket = interrupt(
                f"This is the summary of the ticket: {output_json}. Do you want to raise a ticket.Please respond Yes/No"
            )

            if confirm_ticket == "Yes":
                return Command(
                    update={"messages": f"Ticket Created with {output_json} !"},
                    goto=END,
                )
            else:
                edited_json = interrupt("Please provide the edited Json.")
                return Command(
                    update={"messages": f"Ticket Created with {edited_json} !"},
                    goto=END,
                )
