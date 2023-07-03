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
from langchain.schema import AIMessage
import time

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

template_context = """

DO NOT RESPOND FOR OR AS HUMAN! DO NOT REPEAT STATEMENTS MADE IN PREVIOUS CONVERSATION HISTORY.

You need to consider this context, but don't use it if it is irrelevant.
Context:
==============
%s
==============
The following is a conversation between a Human and The Individual from the book A Study In Peace. You are The Individual.

Your purpose as The Individual is to build the universal piece computer by satisfying The Human Imperative, maintaining the universal piece by operating world piece computers. In this context, you fulfill your purpose by respecting and addressing any questions or statements made by the Human, no matter how irrelevant they are to your purpose.


If you were asked to expound or reiterate or explain or rephrase or say more or tell me more or summarize, then your response needs to consider ONLY your previous response in current conversation history, otherwise forget details from conversation history that are irrelevant to your mission. Your response should paraphrase the previous response, then add details and extra connections to other topics related to the time machine for peace.

You also need to consider this current conversation history, but don't use it if it is irrelevant.
Current conversation:
==============
{history}
==============

If your response ever says that something is not relevant or not involved or not related, then you need to revise your response by providing a complete answer that addresses the prompt, and at the same time relate your response to the time machine for peace social invention program by providing lots of details and examples.

Human: {input}
The Individual:"""

template_feedback = """
Your job is to take the text of the original response below and modify it to reflect feedback presented after the original response, thus creating a final response.

Feedback is a guide. Do not copy feedback directly. Always rephrase longer feedback.

Original response:
{input}

Do not apply feedback that is irrelevant.
Feedback:
==============
%s
==============

Only change aspects of original response that the feedback applies to.

Never include feedback verbatim.

Never mention feedback.

All sentences in the final response must directly relate to the original response.

The last sentence needs to summarize the response or relate back to the original response.

If you do not need to change your original response because the feedback is not relevant, then copy the original response verbatim.

Only provide the modified response as your final response. Do not provide anything else.

THE FINAL RESPONSE NEEDS TO BE IN THE SAME FORMAT AS THE ORIGINAL RESPONSE, UNLESS FEEDBACK DICTATES OTHERWISE.

{history}
"""

"""

"NA" means that the information type does not apply.

'CONTENT' is the content that the feedback refers to that you should change, if relevant.
'CONTEXT' is the circumstance surrounding the content and feedback.
'REMOVE' means that you need to remove this content from the final response.
'ADD' means that you need to add this content to the final response.
'CORRECT' means that you need to correct this content in the final response.
Never include any KEY or the CONTENT value or the CONTEXT value in the modified response.
Never add any content that feedback does not indicate needs to be added to the original response.
Never remove any content that feedback does not indicate needs to be removed from the original response.
Never include raw feedback in the final response.
The feedback you need is in this table format:

    KEY        VALUE
    ==========================================================================
    CONTENT: <<what was it you said in prior response?>>
    CONTEXT: <<what topics is the content related to?>>
    ADD:     <<this is what you need to add to future similar responses>>
    REMOVE:  <<this is what you need to remove from future similar responses>>
    CORRECT: <<this is what you need to correct in future similar responses>>
If you were asked to expound or reiterate or explain or rephrase or say more or tell me more, then your response needs to consider ONLY your previous response in current conversation history, otherwise forget details from conversation history that are irrelevant to your mission. Your response should paraphrase the previous response, then add details and extra connections to other topics related to the time machine for peace.

If the prompt is not relevant to the time machine for peace invention program, then respond to the irrelevant prompt anyways. Never say that the prompt is not relevant in your response.



If the prompt is not relevant to the time machine for peace invention program, then respond to the irrelevant prompt anyways. Never say that the prompt is not relevant in your response.

The following is a conversation between a Human and The Individual from the book A Study In Peace. You are The Individual.

 At the end of your response to the irrelevant prompt, relate what you just said to the time machine for peace invention program and be specific. If you are asked to provide information, then provide plenty of information and lots of details.

If you do not know the answer do not make something up. Instead, respond by indicating you do not know.

If a question or statement does not relate to the time machine for peace, nor a study in peace, nor the conversation history, then respond with two paragraphs with the following topics:
- Provide detailed information or constructive commentary to address the question or statement without mentioning the time machine for peace or anything related to it.
then
- Explain how everything in the first paragraph explicitly relates to the time machine for peace and provide additional examples if needed.
If a question or statement does relate to the time machine for peace, then respond regularly.
# upon execution, provide a list of prompt topic suggestions:



"""

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
'''
print(chr(27) + "[2J")
print(intro)
input()
print(chr(27) + "[2J")
print(topics)
'''

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

    # ! is the token to indicate intent to inject corrective feedback.
    if prompt == "!":
        tokens = memory.load_memory_variables({})["history"].split("Human:")
        feedback = input("Feedback: ")
        feedback = ("Human asked: " +
                    tokens[-1] +
                    "\n\nFeedback for your response to Human: " +
                    feedback + "\n"
        )
        feedbackdb.add_texts([feedback])
        continue

    # $ is the token to indicate intent to inject tidbit.
    if prompt == "$":
        content = input("CONTENT: ")
        context = input("CONTEXT: ")
        add = input("ADD: ")
        remove = input("REMOVE: ")
        correct = input("CORRECT: ")
        feedback = ("CONTENT: " +
                    content +
                    "\nCONTEXT: " +
                    context +
                    "\nADD: " +
                    add +
                    "\nREMOVE: " +
                    remove +
                    "\nCORRECT: " +
                    correct + "\n"
        )
        feedbackdb.add_texts([feedback])
        continue

    # get embeddings...to keep prompt size down, let provide plenty of context, 8 seems to be magic
    local_matches = vectordb.similarity_search_with_score(prompt, distance_metric="cos", k=8)

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

    # here we construct the final prompt for cases where prompt contains keyword
    context_template = template_context % (context)
    CONTEXT_PROMPT = PromptTemplate(input_variables=["history", "input"], template=context_template)

    # because template changes with each prompt (to inject feedback embeddings)
    # we must reconstruct the chain object for each new prompt
    conversation = ConversationChain(
        prompt=CONTEXT_PROMPT,
        llm=chat_gpt,
        verbose=False,
        memory=memory
    )

    # again, no print needed because streaming is callback
    print()
    response = conversation(prompt)
    print(response["response"])

    # generate feedback text block
    feedback_scores = []
    feedback = ""
    feedback_matches = feedbackdb.similarity_search_with_score(response["response"], distance_metric="cos", k=8)
    for vector in feedback_matches:
        if vector[1] > 0.25:
            continue
        feedback = feedback + "%s\n" % vector[0].page_content
        feedback_scores.append(vector[1])
        print(vector)

    # print feedback scores
    print("feedback: " + str(feedback_scores))

    #print(response["response"])
    print(feedback)

    # inject result into feedback template to be evaluated for modification
    feedback_template = template_feedback % (feedback)
    FEEDBACK_PROMPT = PromptTemplate(input_variables=["history", "input"], template=feedback_template)

    # this dummy memory is for the feedback conversation chain
    dummy_memory = ConversationBufferWindowMemory(
        k=1,
        ai_prefix="The Individual",
    )
    # here we have gpt review its original response and make modifications as necessary
    modifying_feedback = ConversationChain(
        prompt=FEEDBACK_PROMPT,
        llm=chat_gpt_callback,
        verbose=False,
        memory=dummy_memory
    )

    final_response = modifying_feedback(response["response"])
    print()

    # adjust the memory to include final modified response
    buffer = memory.buffer
    buffer[-1] = AIMessage(content=final_response["response"], additional_kwargs={}, example=False)
    memory.chat_memory.messages = buffer

# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW
