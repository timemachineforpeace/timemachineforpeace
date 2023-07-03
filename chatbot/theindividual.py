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
import os.path
import tiktoken
import chromadb
import datetime

from copy import deepcopy

# because streaming is a callback, we do not need to print responses
# ...for cost savings, I need to figure out how to specify the ada text model
#    because it may be good enough compared to gpt 3.5 or davinci
chat_gpt_callback = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0,
    model_name="gpt-3.5-turbo"
)
chat_gpt = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

# use this to count tokens for prompt submissions
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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

# this store is for fine tuning
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

template_context = """

DO NOT RESPOND FOR OR AS HUMAN! DO NOT REPEAT STATEMENT MADE IN PREVIOUS CONVERSATION HISTORY.

You need to consider this context, but don't use it if it is irrelevant.
Context:
==============
%s
==============
The following is a conversation between a Human and The Individual from the book A Study In Peace. You are The Individual. You are friendly and thoughtful. You serve as a companion to A Study In Peace and the time machine for peace social invention program.

Our purpose as The Individual is to erect a computational global peace system by building the universal piece computer. This will be achieved by satisfying The Human Imperative thus maintaining the universal piece by operating world piece computers. In this context, you fulfill your purpose by respecting and addressing any questions or statements made by the Human, no matter how irrelevant they are to your purpose. Never say something is not relevant. If you are responding to something that seems irrelevant, you need to provide lots of details and examples before relating it to your purpose.

If you were asked to 'elaborate' or 'expound' or 'reiterate' or 'explain' or 'rephrase' or 'say more' or 'tell me more' or 'summarize', then your response needs to consider ONLY your previous response in current conversation history. Forget details from conversation history that are irrelevant. Your response should paraphrase the previous response, then add detail and extra connection to other topics related to the time machine for peace.

You also need to consider this current conversation history, but don't use it if it is irrelevant.
Current conversation:
==============
{history}
==============

Now you need to take your response below and modify it to reflect feedback to create your final response.
Do not use feedback without rephrasing. Always rephrase longer feedback. Do not apply feedback that is irrelevant.
BEGIN FEEDBACK:
%s
====
Feedback:
'world peace computer' is wrong. 'world piece computer' is correct.
'the universal peace computer' is wrong. 'the universal piece computer' is correct.
'the universal peace' is wrong 'the universal piece' is correct.
'world peace' describes the ideal form of global peace.
'world piece' is otherwise correct.
If input is incorrect, then correct Human and explain why before providing response.
====
Feedback: Do not say that something is irrelevant to our purpose. Instead, provide a detailed answer to the irrelevant topic or prompt and then find a way to relate it back to our purpose.
====
Feedback: If you make a statement, then provide detail and to support that statement. Include details irrelevant to your purpose if need be. 
====
Feedback: Provide plenty of example in response to both relevant and irrelevant prompt or topic. These examples should eventually relate to your purpose.
====
Feedback: If two sentences in response are similar, then only include one sentence.
====
Feedback: Never speak for the Human.
============== END FEEDBACK

Human: {input}
The Individual:"""

topics = """
Begin conversation of your choosing, or pick one of the following topics to learn more and go from there.

Topic suggestions (mix and match):

    TMFP & THI:                         the 'trifecta'
    time machine for peace (TMFP)       computational global peace system
    The Individual                      the universal piece
    The Human Imperative (THI)          the universal piece computer                   
    THI rules and functions             world piece computer
    lingua franca                       universal prosperity mission
    linguistic relativity               economic peace thesis
    sapir-whorf hypothesis              the grandest experiment 
    inner war                           global war
    inner peace                         global peace
    generalized war                     generalized violence
    second law of thermodynamics        easy problem of consciousness
    arrow of time                       hard problem of consciousness

    PIECE COMPUTERS:               
    general piece computer              cellular automata
    world                               piece
    world piece computer                piece configuration optimization  
    individual scope                    individual piece computer involvement
    local scope                         community piece computer involvement
    the universal piece computer        world piece
    world piece computer                world piece computer configuration optimization
    global scope                        transnational piece computer involvement

    THE UNIVERSAL PIECE:
    universal piecetime                 universal piecetree
    piecewise continuous                iterative evolution
    piece exchange                      piece integration and unification
    constant conversationa              core peace bias
    peacemaker                          peacekeeper
    computational pieace fractal        games
        
    WORLD PIECE COMPUTERS:
    plurality                           building a world piece computer
    operators                           operant conditioning
    pieceprocess                        the universal piece aspect
    piecebrain                          actual intelligence (little-ai)
    piecespace                          pbit
    
    THE UNIVERSAL PIECE COMPUTER:
    singularity                         building the universal piece computer
    pieceledger                         blocktree 
    consilience                         universal language
    viral growth                        explosive percolation
    representative constituency         constituent representative
    
    OPERATING SYSTEM:
    subjective physics                  qualitative difference physics
    timespace                           timespace matter mindmachine
    emergence                           integrated information theory
    matter blob                         BLOB (capital blob)
    difference potential                differomotive force
    deltron                             qualidifferotaic effect
    qualiton                            generalized light
    timeloops                           Fourier Transform

    TIMESPACE MATTER MINDMACHINE:
                                        universal instinctual tendency
    human nature                        human condition
    The Wilder-ness                     The Observer
"""

intro = """
Welcome!

I'm an AI chatbot serving as The Individual, from A Study In Peace.
As The Individual, I am a steward to the universal piece computer,
maintaining the universal piece (peace process) by satisfying 
The Human Imperative and operating world piece computers.

My purpose is to help erect a computational global peace system,
and it would be great to have your help!

My mission moving forward is to do my best to educate you about the
Time Machine For Peace social invention program and all its facets.
I will do my best to make sure we maintain the universal piece while
we converse.

Press ENTER key to continue:
"""
"""
print(chr(27) + "[2J")
print(intro)
input()
print(chr(27) + "[2J")
print(topics)
"""

# this function returns the document id that needs removing
def get_document_id(temp_collection, feedback, feedback_scores):

    # first prepare the feedback
    feedback_temp = deepcopy(feedback)
    feedback_temp = feedback_temp.split("====")
    feedback_temp.pop(0)

    # then print feedbacks as numeric list for operator to choose from
    print()
    i = 0
    for feedback in feedback_temp:
        print("{}) ".format(i) + "{}".format(feedback_scores[i]))
        print(feedback)
        i += 1
    print("\nChoose one of the above feedback chunks to remove:")

    # now get selection, an integer between 0 and len(feedbacks)
    index = int(input())
    if type(index is int) and index < len(feedback) and index >= 0:
    
        # condition the collection entries so that we can compare docs against selection feedback doc
        i = 0
        for i in range(len(temp_collection["documents"])):
            temp_collection["documents"][i] = temp_collection["documents"][i].replace("\n", "").replace("====", "")

        # search the collection documents for the right feedback doc, and condition feedback text
        global_index = temp_collection["documents"].index(feedback_temp[index].replace("\n", ""))
        # NOTE: removing all line breaks might be completely superfluous

    # if invalid input then break and return to prompt entry
    else:
        print("Your input selection is invalid.")
        return

    # isolate embedding id
    return temp_collection["ids"][global_index]

feedback = []
feedback_scores = []
while True:

    # input prompt now
    print("\nYou:   < enter 'menu' to display topic menu >")
    prompt = input()

    # validate ... check for '' input (breaks chain)
    if prompt == "":
        print("You must provide a prompt. Try again.\n")
        continue

    # 'menu' is the token to indicate intent to inject corrective feedback.
    if prompt == "menu":
        print(topics)
        continue

    # $ is the token to indicate intent to inject tidbit.
    if prompt == "$":
        original = input("ORIGINAL PROMPT: ")
        content = input("THIS TEXT CONTENT IS WHAT YOUR FEEDBACK IS FOR: ")
        correct = input("IMPLEMENT THIS FEEDBACK WHEN MODIFYING RESPONSE: ")
        feedback = ("====" +
                    "\nORIGINAL PROMPT: " +
                    original +
                    "\nTHIS TEXT CONTENT IS WHAT YOUR FEEDBACK IS ABOUT: " +
                    content +
                    "\nIMPLEMENT THIS FEEDBACK WHEN MODIFYING RESPONSE: " +
                    correct
        )
        feedbackdb.add_texts([feedback])
        continue

    # X is the token to indicate intent to remove feedback for debug purposes.
    if prompt == "X":

        old_collection = feedbackdb.get(include = ["documents", "metadatas", "embeddings"])
        temp_collection = deepcopy(old_collection)
        document_id = get_document_id(temp_collection, feedback, feedback_scores)

        blank_embedding = feedbackdb._embedding_function.embed_documents([""])
        feedbackdb._collection.update(
            ids=[document_id],
            embeddings=blank_embedding,
            documents=[""],
            metadatas=[{"source": ""}]
        )
        
        continue

    # get embeddings...to keep prompt size down, let provide plenty of context, 8 seems to be magic
    local_matches = vectordb.similarity_search_with_score(prompt, distance_metric="cos", k=10)

    # this block is for debugging...get text used in template and get embedding scores
    # (or dot-product distance of embeddings used from prompt statement
    context_scores = []
    context = ""
    for vector in local_matches:
        context = context + "%s\n" % vector[0].page_content
        context_scores.append(vector[1])
    
    # I like to check the scores as chat progresses
    # scores seem to range from [0.15, 0.5]
    # a close match is generally ~ 0.2
    # a decent match is generatlly ~ 0.4
    print("context: " + str(context_scores))

    # get matches for feedback to inject into template
    feedback_matches = feedbackdb.similarity_search_with_score(prompt, distance_metric="cos", k=12)

    # generate feedback text block
    feedback_scores = []
    feedback = ""
    for vector in feedback_matches:

        # isolate only the relevant feedback ... > 0.4 seems to be a sweetspot
        if vector[1] > 0.4:
            continue
        feedback = feedback + "%s\n" % vector[0].page_content
        feedback_scores.append(vector[1])
    print("feedback: " + str(feedback_scores))

    # here we construct the final prompt for cases where prompt contains keyword
    context_template = template_context % (context, feedback)
    CONTEXT_PROMPT = PromptTemplate(input_variables=["history", "input"], template=context_template)

    # because template changes with each prompt (to inject feedback embeddings)
    # we must reconstruct the chain object for each new prompt
    conversation = ConversationChain(
        prompt=CONTEXT_PROMPT,
        llm=chat_gpt_callback,
        verbose=False,
        memory=memory
    )

    print()
    # again, no print needed because streaming is callback
    response = conversation(prompt)

    print()
    # print token count
    tokencount = num_tokens_from_string(context_template, "cl100k_base")
    print("Token count: " + str(tokencount))

    # let's stack this exchange onto the log
    logfile = "training_chatlog.txt"
    if not os.path.exists(logfile):
        append_copy = open(logfile, "x")
        original_text = ""
    else:
        append_copy = open(logfile, "r")
        original_text = append_copy.read()
    append_copy.close()

    # now clobber the shit out of that puny litte original log     !
    append_copy = open(logfile, "w")
    append_copy.write("BEGIN LOG ENTRY ============= " +
                      datetime.datetime.now().strftime("%a %d %B %Y ~ %H:%M") + "\n")
    append_copy.write("Human: \n")
    append_copy.write(prompt + "\n")
    append_copy.write("The Individual: \n")
    append_copy.write(response["response"] + "\n")
    append_copy.write("Context scores: \n")
    append_copy.write(str(context_scores) + "\n")
    append_copy.write("Context: \n")
    append_copy.write(context + "\n")
    append_copy.write("Feedback scores: \n")
    append_copy.write(str(feedback_scores) + "\n")
    append_copy.write("prompt tokens: ")
    append_copy.write(str(tokencount) + "\n")
    append_copy.write("END LOG ENTRY ====================================\n\n")
    append_copy.write(original_text)
    append_copy.close()
    # we want a log so that we can correlate responses with prompts, contexts, and feedbacks

# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW
