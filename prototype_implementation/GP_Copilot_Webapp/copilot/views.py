from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from GP_Copilot.settings import BASE_DIR
from .langchain_utilities import *
from .models import conversations
import datetime, base64, markdown2, os, numpy as np, csv, sqlite3, os, pandas as pd , json
from django.views.decorators.csrf import csrf_exempt
from .langgraph_main import * 
from bs4 import BeautifulSoup

con = sqlite3.connect('db.sqlite3',check_same_thread=False)
cur = con.cursor()

#########################################################
# Path to the retriever. 
# When deploying, replace chroma --> /chroma as it is mounted at /chroma in the container. 

try:
    if os.environ['DJANGO_ENV'] == 'production':
        RETRIEVER_PATH = '/chroma' 
    elif os.environ['DJANGO_ENV'] == 'testing':
        RETRIEVER_PATH = '/chroma'
    else:
        RETRIEVER_PATH = 'chroma'
except:
    RETRIEVER_PATH = 'chroma'
#########################################################


def index(request):
    """
    Displays the page that displays the GPT-4o model 
    """
    print(BASE_DIR)
    conversation_id = np.random.randint(low = 0, high = 9223372036854775807, size = 1)[0]
    print(conversation_id)

    return render(request, template_name = 'gpt4o.html', 
                  context = {'conversation_id' : conversation_id, 'assistant' : "gpt-4o"})

def llama_page(request):
    """
    Displays the page for the llama page. 
    """
    print(BASE_DIR)
    conversation_id = np.random.randint(low = 0, high = 9223372036854775807, size = 1)[0]
    print(conversation_id)
    
    # chain = get_chain()

    return render(request, template_name = 'llama.html', 
                  context = {'conversation_id' : conversation_id, 
                             'assistant' : "llama-3.3"})
    
def haiku_page(request):
    """
    Displays the page for the haiku model
    """
    
    print(BASE_DIR)
    conversation_id = np.random.randint(low = 0, high = 9223372036854775807, size = 1)[0]
    print(conversation_id)
    
    # chain = get_chain()

    return render(request, template_name = 'haiku.html', 
                  context = {'conversation_id' : conversation_id,
                             'assistant' : "haiku"})


def prompt_and_vecstore_experimenter_API(request):
    """
    Displays the page for internal users to 
    log in and experiment with different documents in vector stores.
    """
    password = "TESTER"  # Set your desired password
    authenticated = request.session.get('authenticated', False)
    # Generate a session ID if one doesn't exist
    session_id = request.session.session_key
    if not session_id:
        request.session.create()
        session_id = request.session.session_key
        
    if request.method == 'POST':
        user_password = request.POST.get('password')
        if user_password == password:
            request.session['authenticated'] = True  # Mark session as authenticated
            authenticated = True
        else:
            return render(request, template_name="experiment.html", context={"authenticated": False, "error": "Invalid password"})

    return render(request, template_name="experiment.html", context={"authenticated": authenticated, 'session_id' : session_id})

UPLOADS_LOG_PATH = os.path.join('uploads', 'uploads_log.txt')

@csrf_exempt
def upload_files_API(request):
    """
    API Endpoint to handle file uploads, 
    logs metadata, and saves files to the uploads/vectorstore_rawdocuments directory.
    Returns all files currently in the vectorstore directory.
    """
    if request.method == 'POST':
        session_id = request.POST.get('session_id', 'unknown_session')
        uploaded_files = request.FILES.getlist('files')

        if uploaded_files:
            # Define the target directory
            target_dir = os.path.join('uploads', 'vectorstore_rawdocuments', session_id)
            os.makedirs(target_dir, exist_ok=True)

            # Log each uploaded file and save to the directory
            log_entries = []
            for file in uploaded_files:
                # Generate the full file path
                file_path = os.path.join(target_dir, file.name)
                
                # Write the file to the target directory
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                
                # Log the file metadata
                log_entry = f"{datetime.datetime.now()}, Session ID: {session_id}, File: {file.name}\n"
                log_entries.append(log_entry)

            # Append log entries to the log file
            with open(UPLOADS_LOG_PATH, 'a') as log_file:
                log_file.writelines(log_entries)

            # Add newly uploaded files into vectorstore
            vector_store = get_vector_store(path=RETRIEVER_PATH, session_id=session_id)
            load_files_into_vector_store(vector_store, [os.path.join(target_dir, file.name) for file in uploaded_files])

            # Get all files in the directory to display
            all_files = os.listdir(target_dir)

            # Return all files
            return JsonResponse({'files': all_files})
        
        return JsonResponse({'error': 'No files uploaded.'}, status=400)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


def experiment_view_API(request):
    """
    API Endpoint to handle the experiment page, 
    retrieves system/user prompts, and logs LLM response.
    Stores results in a 'validation_files' directory as a JSON file.
    """
    if request.method == "POST":
        # Retrieve system prompt
        system_prompt = request.POST.get("input1", "")

        # Retrieve user prompt
        user_prompt = request.POST.get("input3", "")

        # Retrieve session ID
        session_id = request.POST.get("session_id", "")

        # Retrieve assistant type
        assistant = request.POST.get("modelSelect", "")

        # Retrieve pasted image
        img = request.POST.get("pastedImage", "")

        # Retrieve pasted links and process them
        links = request.POST.get("inputLinks", "").split('\n')
        download_html_links(links, 'uploads/vectorstore_rawdocuments', session_id)

        # Process image if provided
        desc = get_image_description(user_input=user_prompt, convo_id=session_id, assistant=assistant, aws=True, image_data=img)

        # Get LLM response
        chain = get_chain(model_type=assistant, aws=True, retriever_path=RETRIEVER_PATH, custom_system_prompt=system_prompt, session_id=session_id)
        llm_response = chain.invoke({'question': user_prompt, 'image_description': desc})

        # Prepare document store version info (modify as needed)
        document_store_version = ["manifest", "readme", "documentation", "wrapper scripts"]  # Example, modify accordingly

        # Create results directory if it doesn't exist
        validation_dir = "validation_files"
        os.makedirs(validation_dir, exist_ok=True)

        # Generate JSON filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{validation_dir}/validation_{session_id}_{timestamp}.json"

        # Create JSON structure
        result_data = {
            "DateTime": datetime.now().isoformat(),
            "ModelType": assistant,
            "SystemPrompt": system_prompt,
            "DocumentStoreVersion": document_store_version,
            'ImageDescription': desc,
            "UserPrompt": user_prompt,
            "Response": llm_response
        }

        # Save JSON file
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(result_data, json_file, indent=4)

        # Return JSON response to frontend
        return JsonResponse({"response": markdown2.markdown(llm_response), "file_saved": filename})

    return JsonResponse({"response": "No data received"})
VECTORSTORE_DIR = os.path.join('uploads', 'vectorstore_rawdocuments')
@csrf_exempt  # Remove this if CSRF is handled in the frontend
def list_uploaded_files_API(request):
    """
    API endpoint that lists all files in the uploads/vectorstore_rawdocuments directory for a given session.
    Only supports POST requests.
    """
    if request.method == 'POST':
        session_id = request.POST.get('session_id')
        print(session_id)
        if not session_id:
            return JsonResponse({'error': 'Session ID is required.'}, status=400)

        session_dir = os.path.join(VECTORSTORE_DIR, session_id)

        if not os.path.exists(session_dir):
            return JsonResponse({'files': []})  # Return an empty list if no files exist

        files = os.listdir(session_dir)
        return JsonResponse({'files': files}, status=200)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def delete_uploaded_file_API(request):
    """
    API Endpoint that deletes a specified file from the uploads/vectorstore_rawdocuments/**session_id** directory.
    """
    if request.method == 'POST':
        file_name = request.POST.get('file_name')
        session_id = request.POST.get('session_id', 'unknown_session')
        ## chromadb stuff
        client = chromadb.PersistentClient(path=RETRIEVER_PATH)
        collection = client.get_or_create_collection(name=session_id)
        
        
        if file_name:
            file_path = os.path.join(VECTORSTORE_DIR, session_id, file_name)
            # Check if the file exists
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    collection.delete(where = {'file_name': file_name})
                    return JsonResponse({'success': True, 'message': f'{file_name} removed successfully.'})
                except Exception as e:
                    return JsonResponse({'success': False, 'message': f'Error removing file: {str(e)}'}, status=500)
            else:
                return JsonResponse({'success': False, 'message': 'File does not exist.'}, status=404)
        
        return JsonResponse({'success': False, 'message': 'No file name provided.'}, status=400)
    
    return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)

def get_image_description(user_input, convo_id, assistant, aws, image_data):
    """
    Get a description of an image via LLM inference
    """
        # Remove the 'data:image/png;base64,' prefix if present
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    # Decode the image data
    
    if image_data == "":
        return 'No image provided'
    
    try:
        
        image_binary = base64.b64decode(image_data)
        # Save the image to a file (e.g., in the 'uploads/' directory)
        image_filename = f"uploads/screenshot_{convo_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        with open(image_filename, 'wb') as image_file:
            image_file.write(image_binary)
        print(f"Screenshot saved as {image_filename}")
        with open(image_filename, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        image_file.close()
        image_description = build_image_message(user_input, image_filename, model_type=assistant, aws = aws)

    except Exception as e:
        print(f"Error decoding image: {e}")
        image_description = "No image provided"
    return image_description

def langgraph_API(request):
    """
    API Endpoint method for the langgraph endpoint to post
    
    Main inference API
    """
    aws = True
    if request.method == 'POST':
        user_input = request.POST.get('message')
        convo_id = request.POST.get('convo_id')
        assistant = request.POST.get('assistant')
        image_data = request.POST.get('image_data')  # Get the Base64 image data
        response_id = request.POST.get('response_id')
        # Process image if it exists
        if image_data:
            image_description = get_image_description(user_input, convo_id, assistant, aws, image_data)
            print(image_description)
        else:
            image_description = "No image provided"
            
        if image_description != "No image provided":
            user_input += f"I've also uploaded an image, here's the description: {image_description}"
        ## get the graph:
        graph = get_graph()
        initial_state = {
            "session_id": "1",  # Assign a session ID (string format)
            "messages": [{"role": "user", "content": user_input}],  # Ensure messages are a list
            'user_query' : 'Message Zero',
            "query_type": None,  # This will be populated by the `agent` node
            "query_info": {},  # This will be populated by `detect_modules`
            "model_type": assistant,  # Default model type
            "retriever_path": RETRIEVER_PATH,  # Path for retriever
            "custom_system_prompt": "",  # Custom prompt if any
            "verbose": True,  # Debugging flag
            "extra_field": 0,  # Placeholder extra field
            "action": "",  # Placeholder, modify if needed
        }

        config = {
            "configurable": {
                "model_type": assistant,
                "retriever_path": RETRIEVER_PATH,
                "custom_system_prompt": "",
                "thread_id": convo_id
            }
        }
        
        invoked = graph.invoke(initial_state, config)
        # print('ACTIONS TAKEN:')
        # print(invoked['action'])
        response = invoked['messages'][-1].content
        # response = "peoeoeoeoepwoepweopoweopwoeopwopeopweopwopeoweopweoweopwepoweowe \n\n\n\n ieowieonwe ##headerheader https://cloud.genepattern.org"
        actions = invoked['action']
        conversation = conversations(conversation_id = convo_id,
                                     user_prompt = user_input, 
                                     response = response,
                                     date = datetime.now(),
                                     user_score = 0,
                                     actions_taken = actions,
                                     response_id = response_id)
        conversation.save()

        html = markdown2.markdown(response)
        soup = BeautifulSoup(html, 'html.parser')

        for link in soup.find_all('a'):
            link['target'] = '_blank'

        html_with_targets = str(soup)

        return JsonResponse({'response': html_with_targets})


def response_API(request):
    """
    API endpoint to record
    the response by updating the user score for the 
    conversation with the matching response ID.
    """
    if request.method == 'POST':
        number = request.POST.get('number')
        response_id = request.POST.get('response_id')
        
        print("Number:", number, "Response ID:", response_id)
        
        query = f'''
        UPDATE copilot_conversations 
        SET user_score = {number}
        WHERE response_id = '{response_id}';
        '''
        cur.execute(query)
        con.commit()
        return JsonResponse({'response' : 'success'})


def download_html_links(links, target_dir, session_id="default_session"):
    """
    Downloads HTML content from a list of links and saves them to a specified directory.

    Parameters:
        links (list): List of HTML links to download.
        target_dir (str): Path to the directory where files should be saved.
        session_id (str): Optional session ID to include in filenames for uniqueness.

    Returns:
        list: List of filenames that were successfully downloaded.
    """
    if not links:
        raise ValueError("No links provided.")

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    downloaded_files = []

    for link in links:
        try:
            # Download the content of the link
            response = requests.get(link)
            if response.status_code == 200:
                # Generate a unique filename based on the session ID and link basename
                filename = f"{session_id}_{os.path.basename(link).split('?')[0]}.html"
                file_path = os.path.join(target_dir, filename)

                # Save the content to the target directory
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(response.text)

                downloaded_files.append(filename)
            else:
                print(f"Failed to download {link}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading {link}: {e}")

    return downloaded_files

def get_documents_count_API(request):
    """
    Returns the number of documents in the vectorstore directory.
    """
    if request.method == "POST":
        session_id = request.POST.get("session_id")
        retriever = get_vector_store(path=RETRIEVER_PATH, session_id=session_id)
        documents_count = len(retriever.get())
        print(documents_count)
        
        return JsonResponse({"documents_count": documents_count})
    return JsonResponse({"error": "Invalid request method."}, status=400)


# CSV_FILE_PATH = os.path.join(BASE_DIR, "validation_files", "validation_results_full.csv")
CSV_FILE_PATH = os.path.join("validation_files", "validation_results_full.csv")

def validation_results(request):
    """
    Display the initial validation results page with default filters and groupings.
    """
    if not os.path.exists(CSV_FILE_PATH):
        return render(request, "validation_results.html", {"error": "CSV file not found"})

    df = pd.read_csv(CSV_FILE_PATH)

    system_prompt_filters = {str(k): v for k, v in enumerate(df.SystemPrompt.unique())}

    doc_filter = 'empty'
    document_store_versions = list(df.DocumentStoreVersion.unique())
    system_prompt_options = list(system_prompt_filters.keys())

    return render(request, "validation_results.html", {
        "document_store_versions": document_store_versions,
        "system_prompt_options": system_prompt_options,
    })
    
@csrf_exempt
def get_grouped_data_API(request):
    """
    API endpoint to return grouped data based on selected DocumentStore Version and System Prompt.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            doc_filter = data.get("document_store_version", "empty")

            system_prompt_key = data.get("system_prompt", "0")

            if not os.path.exists(CSV_FILE_PATH):
                return JsonResponse({"error": "CSV file not found"}, status=404)

            df = pd.read_csv(CSV_FILE_PATH)

            # Ensure valid key exists in dictionary
            system_prompt_filters = {str(k): v for k, v in enumerate(df.SystemPrompt.unique())}
            systemprompt_filter = system_prompt_filters.get(system_prompt_key, system_prompt_filters["0"])
            

            df_filtered = df[(df["DocumentStoreVersion"] == doc_filter) & (df["SystemPrompt"] == systemprompt_filter)]
            
            grouped = df_filtered.groupby(["UserPrompt", 'real_answer'])
            
            grouped_data = []
            for (user_prompt, real_answer), group in grouped:
                grouped_data.append({
                    "UserPrompt": markdown2.markdown(user_prompt),
                    "RealAnswer": markdown2.markdown(real_answer),
                    "responses": [
                        {
                            "ID": response["ID"],
                            "Response": markdown2.markdown(response["Response"]),
                            "ModelType": markdown2.markdown(response["ModelType"])
                        }
                        for response in group[['ID', "Response", 'ModelType']].to_dict(orient="records")
                    ]
                })

            print(len(grouped_data))

            return JsonResponse({"grouped_data": grouped_data})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)

