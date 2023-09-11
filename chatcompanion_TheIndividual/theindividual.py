# The Individual AI Assistant: (theindividual.py)
#
# an LLM chat bot for facilitating solo-person conversation over custom corpus +
# OpenAI corpus (c2021), with local corrective fine-tuning.
#
# The purpose of this bot is to facilitate the universal piece process and make
# the operator's world piece computer less shitty.
#
# License: The Human Imperative...
#
#          ...use this to maintain the universal piece and satisfy The Human Imperative.
#             Failure to do so with mal-intent will result in legal action
#             to protect the service-marked time machine for peace social
#             invention program, under the purview of the federal government
#             for The United States of America. Ask The Individual if you need
#             additional clarification.
#
# Operation:
#
# . put corpus into data directory, and hardcode those links if I haven't implemented the
#   iterative form of inject_data.py
# . get OpenAI API key, charge account, and add to environment
# . run injest_data.py
# . run theindividual.py
# . start entering prompts...your name is 'You:'
# ! NOTE if index error occurs on first prompt run, then rerun inject_data.py
# ! NOTE also, sometimes the thing just breaks for no apparent reason. take a deep breath.
#          
# BEGINS #################################################################################

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

# import local stuff
from constants import *
from utilities import *

# this is for tuning, determining which similarity search results to filter out
CONTEXT_THRESHOLD = 1
CONTEXT_COUNT = 10
FEEDBACK_THRESHOLD = 0.4
FEEDBACK_COUNT = 10

# because streaming is a callback, we do not need to print responses
# ...for cost savings, I need to figure out how to specify the ada text model
#    because it may be good enough compared to gpt 3.5 or davinci
chat_gpt_callback = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0,
    model_name="gpt-4"
)

# templates of this resolution are not possible without a window
# --four seems to be a sweetspot
# NB: vectorstore memory is pretty shitty for working memory
#     ...but it seems that for feedback/fine tuning, vector store works great.
#
# NnotB: the above conclusion may be shitty
memory = ConversationBufferWindowMemory(
    k=4,
    ai_prefix="The Individual",
)

# specify the embedding function for decoding vector stores
openAI_embedding = OpenAIEmbeddings()

# this store is the primary corpus ... it may be reinstantiated at any time
contextdb = Chroma(persist_directory="./.contextdb", embedding_function=openAI_embedding)

# this store is for fine tuning
feedbackdb = Chroma(persist_directory="./.feedbackdb", embedding_function=openAI_embedding)

print(feedbackdb.get())
print(text.INTRO)

feedback = []
feedback_scores = []
while True:

    # input prompt now
    print("\n< enter: 'menu' == get topic menu, 'wild' == get wildcard, 'hint' == get suggestion >")
    print("You: ")
    prompt = input()

    # validate ... check for '' input (breaks chain)
    if prompt == "":
        print("You must provide a prompt. Try again.\n")
        continue

    # 'menu' is the token to get list of all relevant topics.
    elif prompt == "menu":
        print(text.MENU)
        continue

    # $ is the token to indicate intent to inject feedback
    elif prompt == "$":
        utilities.add_feedback(feedbackdb)
        continue

    # X is the token to indicate intent to remove feedback for debug purposes.
    # ...this should be replaced with a password for production use
    elif prompt == "X":
        utilities.remove_embedding(
            vectordb=feedbackdb,
            injection=feedback_injection,
            scores=feedback_scores
        )
        continue

    elif prompt == "wild":

        continue

    # get context to inject into template
    context_injection, context_scores = utilities.generate_injection(
        vectordb=contextdb,
        prompt=prompt,
        COUNT=CONTEXT_COUNT,
        THRESHOLD=CONTEXT_THRESHOLD
    )
    print("context scores: " + str(context_scores))

    # get feedback to inject into template
    feedback_injection, feedback_scores = utilities.generate_injection(
        vectordb=feedbackdb,
        prompt=prompt,
        COUNT=FEEDBACK_COUNT,
        THRESHOLD=FEEDBACK_THRESHOLD
    )
    print("feedback scores: " + str(feedback_scores) + "\n")

    # here we inject text to construct the final prompt for cases where prompt contains keyword
    template_main = utilities.inject_main(context_injection, feedback_injection)
    PROMPT = PromptTemplate(
        input_variables=["history", "input"],
        template=template_main
    )

    # because template changes with each prompt (to inject feedback embeddings)
    # we must reconstruct the chain object for each new prompt
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=chat_gpt_callback,
        verbose=False,
        memory=memory
    )

    # again, no print needed because streaming is callback
    response = conversation(prompt)

    # log current call and response with injection data and scores
    utilities.log_exchange(
        prompt=prompt,
        response=response,
        context_scores=context_scores,
        context=context_injection,
        feedback_scores=feedback_scores,
        feedback=feedback_injection,
        tokens=utilities.num_tokens(template_main, "cl100k_base")
    )

    # we want a log so that we can correlate responses with prompts, contexts, and feedbacks

# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW
