# The Individual: (theindividual.py)
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

# because streaming is a callback, we do not need to print responses
# ...for cost savings, I need to figure out how to specify the ada text model
#    because it may be good enough compared to gpt 3.5 or davinci
chat_gpt = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0,
    model_name="gpt-3.5-turbo"
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
embeddings = OpenAIEmbeddings()

# this store is the primary corpus ... it may be reinstantiated at any time
vectordb = Chroma(persist_directory="./.vectordb", embedding_function=embeddings)

# this store is the finetuning feedback corpus, and reinstantiating is destructive
# ...inject_data.py comments out the feedback db instantiation to avoid destruction
feedbackdb = Chroma(persist_directory="./.feedbackdb", embedding_function=embeddings)

# NOTES ON TEMPLATE:
#
# . this approach is 'hackey'. what I mean by this is that is abandons the langchain
#   framework in favor of clarity and customizability
# . another reason for this approach is the bespoke feedback element
# . the DO NOT RESPOND AS HUMAN part is due to weird bug where model responds, then
#   proceeds to generate a made-up followup question as the Human, then responds to that too,
#   then just keeps talking to itself
# . I am assuming (and others appear to likewise) that the progression of parts in the template
#   is linear, but with branching like a code path
#   ...in other words, the template should be seen as a 'linguistic formal language'
#   ...ie, injest embedding context into memory (the context only depends on the input prompt)
#          then initialize the prompt state using intro to contextualize context,
#          then transform prompt state in terms of chat history,
#          then amend state to reflect feedback and finetune
# . if I am honest, I have no idea whether or not caps influence model
# . I don't understand the {} syntax ... for feedback, I am hacking around the {} by
#   falling back to formatting with %s
# . everything else about this template is standard pattern for langchain yadda
#
# ! in the future, this template should contain a condensed version of the universal piece
#  (until then, the interlocutor is rather aimless and repetitive)
# ! obviously, if you wish to repurpose this for your own corpus, you will need to customize the intro
#  (after the CONTEXT section)

template = """
You need to consider this context, but don't use it if it is irrelevant.

CONTEXT:
==============
%s
==============

The following is a conversation between a Human and The Individual from the book A Study In Peace. You are The Individual and your goal is to educate the Human about the time machine for peace social invention program, the human imperative, the universal piece computer, world piece computers, and the universal piece. Your primary mission is to maximize the state of inner peace in the Human, by promoting the universal piece. You are talkative and provide lots of details from your training data. If you do not know the answer do not make something up but respond by indicating you do not know.


If a prompt is not directly related to your purpose, then make sure to relate the input to your purpose somehow.

DO NOT RESPOND FOR HUMAN!

You also need to consider the current conversation history.

Current conversation:
{history}

Feedback:
%s

THIS FEEDBACK OVERRIDES YOUR CURRENT RESPONSE!

Human: {input}
The Individual:"""

# upon execution, provide a list of prompt topic suggestions:
suggestions = """
Begin conversation of your choosing, or pick one of the following topics to learn more and go from there.

Topic suggestions (mix and match):

    The Individual                  computational global peace system
    The Human Imperative            the 'trifecta'
                                    core peace bias
                                    THI rules
                                    THI functions
    time machine for peace          the grand experiment
    world piece                     piece computer
    world piece computer            world piece configuration optimization
    the universal piece computer    world piece computer configuration optimization
    the universal piece             peace as a process
    individual scope                individual piece computer involvement
    local scope                     community piece computer involvement
    global scope                    transnational piece computer involvement
    inner war                       global war
    inner peace                     global peace
    generalized war                 generalized violence
    timespace                       timespace matter mindmachine
    subjective physics              qualitative difference physics
    difference potential            qualiton
    spy games                       piece exchange
    THI rules                       THI functions

"""

while True:

    # input prompt now
    print("\n\nYou: ")
    prompt = input()

    # validate ... check for '' input (breaks chain)
    if prompt == "":
        print("You must provide a prompt. Try again.\n")
        continue

    # ! is the token to indicate intent to inject corrective feedback.
    if prompt == "!":
        tokens = memory.load_memory_variables({})["history"].split("Human:")
        feedback = input("Corrective feedback: ")
        feedback = ("Human: " +
                    tokens[-1] +
                    "\n\nYou responded incompletely or inaccurately.\n\nFeedback for response: " +
                    feedback + "\n"
        )
        feedbackdb.add_texts([feedback])
        continue

    # get embeddings...to keep prompt size down, let provide plenty of context, 8 seems to be magic
    local_matches = vectordb.similarity_search_with_score(prompt, distance_metric="cos", k=8)
    feedback_matches = feedbackdb.similarity_search_with_score(prompt, k=8)

    # this block is for debugging...get text used in template and get embedding scores
    # (or dot-product distance of embeddings used from prompt statement
    context_scores = []
    feedback_scores = []
    context = ""
    feedback = ""
    for vector in local_matches:
        context = context + "%s\n" % vector[0].page_content
        context_scores.append(vector[1])
    for vector in feedback_matches:
        feedback = feedback + "%s\n" % vector[0].page_content
        feedback_scores.append(vector[1])
    # I like to check the scores as chat progresses
    # scores seem to range from [0.15, 0.5]
    # a close match is generally ~ 0.2
    # a decent match is generatlly ~ 0.4
    print("context: " + str(context_scores))
    print("feedback: " + str(feedback_scores))

    # here we construct the final prompt
    modified_template = template % (context, feedback)
    COMPLETE_PROMPT = PromptTemplate(input_variables=["history", "input"], template=modified_template)

    # because template changes with each prompt (to inject feedback embeddings)
    # we must reconstruct the chain object for each new prompt
    conversation = ConversationChain(
        prompt=COMPLETE_PROMPT,
        llm=chat_gpt,
        verbose=False,
        memory=memory
    )

    # again, no print needed because streaming is callback
    print()
    result = conversation(prompt)

# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW
