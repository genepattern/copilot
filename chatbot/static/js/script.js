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
    const modelSelect = document.getElementById('llm-models');
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

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

    async function sendMessage() {
        const queryText = userInput.value.trim();
        if (!queryText) return;

        addMessage('user', queryText);
        const waitMessage = addWaitMessage();
        userInput.value = ''; // Clear input field
        sendButton.disabled = true; // Disable button while processing
        modelSelect.disabled = true; // Disable model selection once a conversation is started

        const payload = {
            query: queryText,
            conversation_id: currentConversationId, // Send null if it's the first message
            model_id: modelSelect.value || null,    // Use selected model
            html: true
        };

        // Include API key if available
        if (localStorage.getItem('gp_api_key')) {
            payload.api_key = localStorage.getItem('gp_api_key');
        }

        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify(payload)
            });
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
            }

            // Add bot response
            addMessage('bot', data.response, data.id, data.rating); // Pass queryId and initial rating

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

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        // Send message on Enter key, unless Shift+Enter is pressed
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default newline insertion
            sendMessage();
        }
    });

    // Load LLM models
    async function loadModels(default_model = 'us.meta.llama3-3-70b-instruct-v1:0') {
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
});

document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
    const logoutBtn = document.getElementById('logoutBtn');

    // Handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();

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
                    updateUIForLoggedInUser(data.username, data.is_staff);
                    if (data.api_key) localStorage.setItem('gp_api_key', data.api_key); // Store API key in localStorage
                } else {
                    document.getElementById('loginError').textContent = data.error;
                    document.getElementById('loginError').classList.remove('d-none');
                }
            } catch (error) {
                console.error('Login error:', error);
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
    }

    function updateUIForLoggedOutUser() {
        const navbarNav = document.querySelector('.navbar-nav.ms-auto');
        navbarNav.innerHTML = `
            <button type="button" class="btn btn-outline-primary text-nowrap" data-bs-toggle="modal" data-bs-target="#loginModal">
                Sign in
            </button>
        `;
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