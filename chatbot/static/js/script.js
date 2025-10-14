function getCookie(name) {
    // First try to get from cookie
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }

    // Fallback to meta tag only if cookie not found
    if (!cookieValue && name === 'csrftoken') {
        const csrfMeta = document.querySelector('meta[name="csrf-token"]');
        if (csrfMeta) {
            return csrfMeta.getAttribute('content');
        }
    }

    return cookieValue;
}
window.getCookie = getCookie; // Make it globally accessible

document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const methodSelect = document.getElementById('llm-methods');
    const modelSelect = document.getElementById('llm-models');
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

    // File attachment elements
    const fileInput = document.getElementById('file-input');
    const attachmentButton = document.getElementById('attachment-button');
    const attachedFilesContainer = document.getElementById('attached-files');
    const fileSizeErrorModal = new bootstrap.Modal(document.getElementById('fileSizeErrorModal'));
    const MAX_FILE_SIZE = 325 * 1024; // 325 KB in bytes
    let attachedFiles = [];

    // Sidebar elements
    const conversationsList = document.getElementById('conversations-list');
    const sidebarContainer = document.getElementById('conversations-sidebar-container');
    const newConversationForm = document.querySelector('.conversation-form');

    // Main chat column for responsive layout adjustments based on auth state
    const mainChatCol = document.getElementById('main-chat-col');
    function setChatMainAuthLayout(isAuthenticated) {
        if (!mainChatCol) return;
        // Reset classes we control
        mainChatCol.classList.remove('col-md-9', 'col-lg-9', 'col-md-8', 'col-lg-6', 'mx-auto');
        if (isAuthenticated) {
            mainChatCol.classList.add('col-md-9', 'col-lg-9');
        } else {
            mainChatCol.classList.add('col-md-8', 'col-lg-6', 'mx-auto');
        }
        if (typeof refreshLayoutHeight === 'function') refreshLayoutHeight();
    }
    window.setChatMainAuthLayout = setChatMainAuthLayout;

    // Layout height calculation between fixed navbar and footer
    function refreshLayoutHeight() {
        const row = document.getElementById('content-row');
        if (!row) return;
        const footer = document.querySelector('footer.fixed-bottom');
        const footerHeight = footer ? footer.offsetHeight : 0;
        const rect = row.getBoundingClientRect();
        const available = Math.max(window.innerHeight - footerHeight - rect.top - 20, 200);
        row.style.height = available + 'px';
        row.style.minHeight = available + 'px';
    }
    window.refreshLayoutHeight = refreshLayoutHeight;
    // Initial call and listeners
    window.addEventListener('resize', refreshLayoutHeight);
    window.addEventListener('orientationchange', refreshLayoutHeight);
    setTimeout(refreshLayoutHeight, 0);

    // Show LLM controls by default; toggle with Cmd+i / Ctrl+i
    if (methodSelect) methodSelect.classList.remove('d-none');
    if (modelSelect) modelSelect.classList.remove('d-none');

    document.addEventListener('keydown', (e) => {
        const isCtrl = e.ctrlKey;
        if (isCtrl && String(e.key).toLowerCase() === 'i') {
            e.preventDefault();
            [methodSelect, modelSelect].forEach(el => el && el.classList.toggle('d-none'));
        }
    });

    let currentConversationId = null; // Store the conversation ID
    let selectedLlmModel = null; // Store the selected LLM model

    function addWaitMessage() {
        // Load status messages
        const statusMessages = [
            "🧠 Thinking real hard...",
            "☕ Making coffee and contemplating life...",
            "📚 Looking through documents...",
            "🪄 Summoning GPT spirits...",
            "📰 Reading the latest findings...",
            "🐢 Waiting for the AI hamster to catch up...",
            "🔍 Scanning retrieved data...",
            "🧬 Analyzing modules...",
            "⚙️ Generating response...",
            "🧙‍♂️ Consulting the ancient scrolls of Stack Overflow...",
            "🦾 Arguing with other robots...",
            "🎨 Sketching the answer in ASCII art...",
            "🪐 Traveling through the embeddings universe...",
            "💅 Applying semantic lip gloss...",
            "🤹‍♀️ Juggling tokens...",
            "🧩 Solving wordle to warm up...",
            "🐸 Asking Kermit for advice...",
            "🥸 Pretending to be smarter than it is...",
            "🚿 Shower thoughts incoming...",
            "🚧 Building the response brick by brick...",
            "🍕 Bribing the model with virtual pizza...",
            "💾 Loading witty comeback...",
            "🫠 Melting under pressure...",
            "🤖 Beep boop beep... translating human nonsense...",
        ];
        let statusIndex = 0; // alternate: RANDOM START: Math.floor(Math.random() * (statusMessages.length + 1));

        // Create wait box
        const messageBox = addMessage('bot', statusMessages[statusIndex], null);
        messageBox.querySelector('p');

        // Start cycling through waiting messages
        const statusInterval = setInterval(() => {
            statusIndex = (statusIndex + 1) % statusMessages.length;
            messageBox.querySelector('p').innerHTML = `<span class="loader"></span><em> ${statusMessages[statusIndex]}</em>`;
        }, 5000);

        return {
            statusInterval: statusInterval,
            messageBox: messageBox
        }
    }

    function clearWaitMessage(waitMessage) {
        clearInterval(waitMessage.statusInterval);
        waitMessage.messageBox.remove();
    }

    function addMessage(sender, text, queryId = null, rating = 0) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender); // sender is 'user' or 'bot' or 'error'

        const messageParagraph = document.createElement('p');
        messageParagraph.innerHTML = text;
        messageDiv.appendChild(messageParagraph);

        // Add rating buttons for bot messages
        if (sender === 'bot' && queryId) {
            const ratingDiv = document.createElement('div');
            ratingDiv.classList.add('rating-buttons');

            const thumbUpButton = document.createElement('button');
            thumbUpButton.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>'; // Thumbs Up
            thumbUpButton.classList.add('thumb-up');
            thumbUpButton.dataset.queryId = queryId;
            thumbUpButton.dataset.rating = 1; // 1 for thumbs up

            const thumbDownButton = document.createElement('button');
            thumbDownButton.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>'; // Thumbs Down
            thumbDownButton.classList.add('thumb-down');
            thumbDownButton.dataset.queryId = queryId;
            thumbDownButton.dataset.rating = -1; // -1 for thumbs down

            // Set initial rated state
            if (rating === 1) thumbUpButton.classList.add('rated-up');
            if (rating === -1) thumbDownButton.classList.add('rated-down');

             // Disable buttons if already rated
            if (rating !== 0) {
                 thumbUpButton.disabled = true;
                 thumbDownButton.disabled = true;
            }

            ratingDiv.appendChild(thumbUpButton);
            ratingDiv.appendChild(thumbDownButton);
            messageDiv.appendChild(ratingDiv);

            // Add event listeners for rating
            thumbUpButton.addEventListener('click', handleRatingClick);
            thumbDownButton.addEventListener('click', handleRatingClick);
        }

        chatBox.appendChild(messageDiv);

        // Scroll to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;

        return messageDiv;
    }

    function handleApiError(error, context) {
        console.error(`API Error (${context}):`, error);
        let errorMessage = `⚠️ Error: ${error.message || 'Could not reach server.'}`;
        if (error.response) {
            // Try to get more specific error from API response body
            error.response.json().then(data => {
                errorMessage = `⚠️ API Error (${error.response.status}): ${data.error || data.detail || JSON.stringify(data)}`;
                addMessage('error', errorMessage);
            }).catch(() => {
                 addMessage('error', `⚠️ API Error (${error.response.status}): Could not parse error response.`);
            });
        } else {
             addMessage('error', errorMessage);
        }

    }

    function showSidebar() {
        if (!sidebarContainer) return;
        // Show on all appropriate viewports: remove d-none and add d-md-block
        sidebarContainer.classList.remove('d-none');
        if (!sidebarContainer.classList.contains('d-md-block')) sidebarContainer.classList.add('d-md-block');
        if (typeof refreshLayoutHeight === 'function') refreshLayoutHeight();
    }

    function hideSidebar() {
        if (!sidebarContainer) return;
        // Hide for all viewports
        sidebarContainer.classList.add('d-none');
        // Ensure it doesn't reappear on md+ breakpoints
        sidebarContainer.classList.remove('d-md-block');
        if (conversationsList) conversationsList.innerHTML = '';
        if (typeof refreshLayoutHeight === 'function') refreshLayoutHeight();
    }

    async function fetchConversations() {
        const response = await fetch('/api/conversations/', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    }

    function renderConversations(conversations, activeId = null) {
        if (!conversationsList) return;
        conversationsList.innerHTML = '';
        conversations.forEach(conv => {
            const a = document.createElement('a');
            a.href = '#';
            a.className = 'list-group-item list-group-item-action conversation-item text-light text-truncate';
            a.dataset.conversationId = conv.id;
            a.title = conv.title || 'Conversation';
            a.textContent = conv.title || 'Conversation';
            if (activeId && conv.id === activeId) a.classList.add('active');
            a.addEventListener('click', (e) => {
                e.preventDefault();
                loadConversation(conv.id);
            });
            conversationsList.appendChild(a);
        });
    }

    async function refreshConversations(activeId = null) {
        try {
            const conversations = await fetchConversations();
            renderConversations(conversations, activeId || currentConversationId);
        } catch (e) {
            console.warn('Could not load conversations (maybe not logged in).');
        }
    }

    async function loadConversation(conversationId) {
        try {
            const response = await fetch(`/api/conversations/${conversationId}/`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();

            // Clear chat and render history
            chatBox.innerHTML = '';
            (data.queries || []).forEach(q => {
                if (q.query) addMessage('user', q.query);
                if (q.response) addMessage('bot', q.response, q.id, q.rating);
            });

            currentConversationId = data.id;
            renderConversations(await fetchConversations(), currentConversationId);
        } catch (e) {
            handleApiError(e, 'loading conversation');
        }
    }

    // Expose helpers for login/logout handler
    window.refreshConversations = refreshConversations;
    window.showConversationsSidebar = showSidebar;
    window.hideConversationsSidebar = hideSidebar;

    async function sendMessage() {
        const queryText = userInput.value.trim();
        if (!queryText && attachedFiles.length === 0) return;

        // Display user message with file indicators if files are attached
        let displayMessage = queryText;
        if (attachedFiles.length > 0) {
            const fileNames = attachedFiles.map(f => f.name).join(', ');
            displayMessage += `\n<br><small>📎 Attached: ${fileNames}</small>`;
        }
        addMessage('user', displayMessage);

        const waitMessage = addWaitMessage();
        userInput.value = ''; // Clear input field
        sendButton.disabled = true; // Disable button while processing
        modelSelect.disabled = true; // Disable model selection once a conversation is started
        methodSelect.disabled = true; // Disable method selection once a conversation is started

        // Prepare request - use FormData if files are attached, otherwise JSON
        let fetchOptions;

        if (attachedFiles.length > 0) {
            // Use FormData for file uploads
            const formData = new FormData();
            formData.append('query', queryText);
            if (currentConversationId) formData.append('conversation_id', currentConversationId);
            if (modelSelect.value) formData.append('model_id', modelSelect.value);
            if (methodSelect.value) formData.append('method_id', methodSelect.value);
            formData.append('html', 'true');
            if (localStorage.getItem('gp_api_key')) {
                formData.append('api_key', localStorage.getItem('gp_api_key'));
            }

            // Attach files
            attachedFiles.forEach(file => {
                formData.append('files', file);
            });

            fetchOptions = {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: formData
            };
        } else {
            // Use JSON for text-only messages
            const payload = {
                query: queryText,
                conversation_id: currentConversationId,
                model_id: modelSelect.value || null,
                method_id: methodSelect.value || null,
                html: true
            };

            if (localStorage.getItem('gp_api_key')) {
                payload.api_key = localStorage.getItem('gp_api_key');
            }

            fetchOptions = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify(payload)
            };
        }

        try {
            const response = await fetch('/api/chat/', fetchOptions);
            clearWaitMessage(waitMessage); // Clear wait message

            if (!response.ok) {
                 // Throw an error object that includes the response for detailed handling
                 const error = new Error(`⚠️ HTTP error! status: ${response.status}`);
                 error.response = response;
                 throw error;
            }

            const data = await response.json();

            // Update conversation ID if this was the first message or a new conv started
            if (data.conversation && currentConversationId !== data.conversation) {
                 currentConversationId = data.conversation;
                 console.log("Started/Using Conversation ID:", currentConversationId);
                 // Refresh sidebar on first message of new conversation
                 if (window.IS_AUTHENTICATED) {
                    try { await refreshConversations(currentConversationId); } catch(e) {}
                 }
            }

            // Add bot response
            addMessage('bot', data.response, data.id, data.rating); // Pass queryId and initial rating

            // Clear attached files after successful send
            attachedFiles = [];
            updateAttachedFilesUI();

        } catch (error) {
             handleApiError(error, 'sending message');
        } finally {
            sendButton.disabled = false; // Re-enable button
            userInput.focus();
        }
    }

    async function handleRatingClick(event) {
        const button = event.currentTarget;
        const queryId = button.dataset.queryId;
        const rating = parseInt(button.dataset.rating, 10);

        // Prevent re-rating for now (could implement changing rating later)
        if (button.disabled) return;

        const payload = {
            rating: rating
        };

         // Disable both buttons for this message immediately
        const parentRatingDiv = button.parentElement;
        const buttons = parentRatingDiv.querySelectorAll('button');
        buttons.forEach(btn => btn.disabled = true);

        try {
             const response = await fetch(`/api/rate/${queryId}/`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                 const error = new Error(`HTTP error! status: ${response.status}`);
                 error.response = response;
                 throw error;
            }

            const data = await response.json();
            console.log('Rating successful:', data);

            // Replace buttons with message
            parentRatingDiv.classList.add('alert', 'alert-sm', 'alert-success');
            parentRatingDiv.innerHTML = `<span class="alert-text">${data.response}</span>`;

            // Hide after 3 seconds
            setTimeout(() => parentRatingDiv.style.display = 'none', 3000);
        }
        catch (error) {
             handleApiError(error, 'submitting rating');

             // Re-enable buttons if rating failed
              buttons.forEach(btn => btn.disabled = false);

        }
    }

    // File attachment handling
    function calculateTotalFileSize() {
        return attachedFiles.reduce((total, file) => total + file.size, 0);
    }

    function formatFileSize(bytes) {
        return (bytes / 1024).toFixed(2) + ' KB';
    }

    function updateAttachedFilesUI() {
        attachedFilesContainer.innerHTML = '';

        attachedFiles.forEach((file, index) => {
            const fileTag = document.createElement('div');
            fileTag.className = 'file-tag';
            fileTag.innerHTML = `
                <span class="file-tag-name" title="${file.name}">${file.name}</span>
                <button class="file-tag-remove" data-index="${index}" title="Remove file">
                    <i class="fa fa-times"></i>
                </button>
            `;
            attachedFilesContainer.appendChild(fileTag);
        });
    }

    function addFiles(files) {
        const newFiles = Array.from(files);
        const tempFiles = [...attachedFiles, ...newFiles];
        const totalSize = tempFiles.reduce((sum, file) => sum + file.size, 0);

        if (totalSize > MAX_FILE_SIZE) {
            const errorMsg = `The total size of attached files (${formatFileSize(totalSize)}) exceeds the maximum limit of 325 KB.`;
            document.getElementById('fileSizeErrorMessage').textContent = errorMsg;
            fileSizeErrorModal.show();
            return;
        }

        attachedFiles = tempFiles;
        updateAttachedFilesUI();
    }

    // Attachment button click handler
    attachmentButton.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            addFiles(e.target.files);
            fileInput.value = ''; // Reset input
        }
    });

    // Remove file handler (event delegation)
    attachedFilesContainer.addEventListener('click', (e) => {
        const removeBtn = e.target.closest('.file-tag-remove');
        if (removeBtn) {
            const index = parseInt(removeBtn.dataset.index);
            attachedFiles.splice(index, 1);
            updateAttachedFilesUI();
        }
    });

    // Drag and drop functionality
    let dragCounter = 0;

    chatBox.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        chatBox.classList.add('drag-over');
    });

    chatBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter === 0) {
            chatBox.classList.remove('drag-over');
        }
    });

    chatBox.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    chatBox.addEventListener('drop', (e) => {
        e.preventDefault();
        dragCounter = 0;
        chatBox.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            addFiles(files);
        }
    });

    // Send button event listener
    sendButton.addEventListener('click', () => {
        sendMessage();
    });

    // Enter key event listener
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Load LLM models
    async function loadModels(default_model = 'us.anthropic.claude-3-5-haiku-20241022-v1:0') {
        const models = await fetch(`/api/models/`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'},
        }).then(r => r.json());
        for (const model of models) {
            if (model.disabled) continue;
            const option = document.createElement('option');
            option.value = model.model_id;
            option.textContent = model.label;
            modelSelect.appendChild(option);
        }
        modelSelect.value = default_model
        selectedLlmModel = default_model;

        modelSelect.addEventListener('change', async (event) => {
            selectedLlmModel = event.target.value;
        });
    }
    loadModels();
    if (window.IS_AUTHENTICATED) {
        try { refreshConversations(); } catch (e) {}
    }
    if (newConversationForm) {
        newConversationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            currentConversationId = null;
            chatBox.innerHTML = '';
            if (methodSelect) methodSelect.disabled = false; // Re-enable method select
            if (modelSelect) modelSelect.disabled = false;  // Re-enable model select
            try { refreshConversations(); } catch (e) {}
        });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
    const logoutBtn = document.getElementById('logoutBtn');

    // Handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const spinner = document.getElementById('loginSpinner');
            const submitBtn = document.getElementById('loginSubmitBtn');
            const errorBox = document.getElementById('loginError');
            if (errorBox) errorBox.classList.add('d-none');

            if (spinner) spinner.classList.remove('d-none');
            if (submitBtn) submitBtn.disabled = true;

            const formData = new FormData();
            formData.append('username', document.getElementById('username').value);
            formData.append('password', document.getElementById('password').value);

            try {
                const response = await fetch('/api/login/', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': getCookie('csrftoken')
                    }
                });

                const data = await response.json();

                if (data.success) {
                    loginModal.hide();
                    window.IS_AUTHENTICATED = true;
                    updateUIForLoggedInUser(data.username, data.is_staff);
                    if (typeof window.setChatMainAuthLayout === 'function') window.setChatMainAuthLayout(true);
                    if (data.api_key) localStorage.setItem('gp_api_key', data.api_key); // Store API key in localStorage
                } else {
                    if (errorBox) {
                        errorBox.textContent = data.error || 'Invalid credentials';
                        errorBox.classList.remove('d-none');
                    }
                }
            } catch (error) {
                console.error('Login error:', error);
                if (errorBox) {
                    errorBox.textContent = 'Login failed. Please try again.';
                    errorBox.classList.remove('d-none');
                }
            } finally {
                if (spinner) spinner.classList.add('d-none');
                if (submitBtn) submitBtn.disabled = false;
            }
        });
    }

    // Handle logout
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            await do_logout();
        });
    }

    function updateUIForLoggedInUser(username, isStaff) {
        const navbarNav = document.querySelector('.navbar-nav.ms-auto');
        const adminLink = isStaff ? '<li><a class="dropdown-item" target="_blank" href="/admin/">Admin</a></li>' : '';

        navbarNav.innerHTML = `
            <div class="nav-item dropdown">
                <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown">
                    ${username}
                </a>
                <ul class="dropdown-menu dropdown-menu-end">
                    ${adminLink}
                    <li><a class="dropdown-item" href="#" id="logoutBtn">Sign out</a></li>
                </ul>
            </div>
        `;

        // Re-attach logout event listener to the new element
        const newLogoutBtn = document.getElementById('logoutBtn');
        if (newLogoutBtn) {
            newLogoutBtn.addEventListener('click', async function(e) {
                e.preventDefault();
                await do_logout()
            });
        }

        // Show sidebar and refresh conversations after login
        if (window.showConversationsSidebar) window.showConversationsSidebar();
        if (window.refreshConversations) window.refreshConversations();
    }

    function updateUIForLoggedOutUser() {
        const navbarNav = document.querySelector('.navbar-nav.ms-auto');
        navbarNav.innerHTML = `
            <button type="button" class="btn btn-outline-primary text-nowrap" data-bs-toggle="modal" data-bs-target="#loginModal">
                Sign in
            </button>
        `;
        window.IS_AUTHENTICATED = false;
        if (typeof window.setChatMainAuthLayout === 'function') window.setChatMainAuthLayout(false);
        if (window.hideConversationsSidebar) window.hideConversationsSidebar();
    }

    async function do_logout() {
        try {
            const response = await fetch('/api/logout/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                }
            });

            const data = await response.json();
            if (data.success) {
                updateUIForLoggedOutUser();
                localStorage.removeItem('gp_api_key'); // Clear API key from localStorage
            }
        } catch (error) {
            console.error('Logout error:', error);
        }
    }
});
