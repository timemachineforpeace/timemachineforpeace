from flask import Flask, render_template, request, jsonify
import openai
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
openAI_embedding = OpenAIEmbeddings()

# this store is the primary corpus ... it may be reinstantiated at any time
contextdb = Chroma(persist_directory="./.contextdb", embedding_function=openAI_embedding)

# this store is for fine tuning
feedbackdb = Chroma(persist_directory="./.feedbackdb", embedding_function=openAI_embedding)  
  
app = Flask(__name__)
  
# OpenAI API Key
openai.api_key = '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'
  
def get_completion(prompt):
    # validate ... check for '' input (breaks chain)
    if prompt == "":
        print("You must provide a prompt. Try again.\n")

    # 'menu' is the token to get list of all relevant topics.
    elif prompt == "menu":
        print(text.MENU)


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
    return response["response"]
  
@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        print('step1')
        prompt = request.form['prompt']
        response = get_completion(prompt)
        print(response)
  
        return response
    return render_template('index.html')
  
  
if __name__ == "__main__":
    app.run(debug=True)
