from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
from os import listdir
from os.path import isfile, join
from django.templatetags.static import static
import markdown2




client = OpenAI()

def create_assistant(model = 'gpt-4o'):
    assistant = client.beta.assistants.create(
    name="GenePattern Copilot",
    instructions='You are a bioinformatics wizard who works for the GenePattern team. \
        The ‘module_descriptions.txt’ is a list of tools and modules available in GenePattern. \
            Do not describe tools that are not in the ‘module_descriptions.txt’, instead respond \
                with "That tool is not currently available in GenePattern. Feel free to contact \
                    the GenePattern team if you think it would be a good addition to our repository." \
                        Provide input file formats when giving instructions on how to run modules \
                            or tools. Only give module suggestions for modules in GenePattern. \
                                Do not tell users to “go to GenePattern and log in”. \
                                    Answer the following questions using all your knowledge\
                                        and  providing  as much detail as possible with step-by-step instructions.',
    model=model,
    tools=[{"type": "file_search"}],
    temperature=0.6)
    print('Assistant is created!')
    return assistant


def retrieve_assistant(model = 'gpt-4o'):
    ## replaced with Edwin's assistant ID to run manage.py successfully. 
    assistant = create_assistant(model)
    print('Assistant is retrieved!')
    return assistant
    
def create_vector_store(assistant):
    # Create a vector store caled "Financial Statements"
    vector_store = client.beta.vector_stores.create(name="GenePattern Documents")
    
    # Ready the files for upload to OpenAI
    file_paths = ['/Users/forrestkim/Documents/GitHub/GPCopilot/GP_Copilot/static/files/Differential_expression_guide.html']
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    return assistant

def create_thread(user_input):
    # Create a thread and attach the file to the message
    thread = client.beta.threads.create(
    messages=[
        {
        "role": "user",
        "content": user_input,
        }
    ]
    )
    
    # The thread now has a vector store with that file in its tool resources.
    print(thread.tool_resources.file_search)
    return thread

def run_inference(assistant, thread):

    run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id)
    
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    citation_string = "\n".join(citations)
    response = markdown2.markdown(message_content.value)
    return response, citation_string