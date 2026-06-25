const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

// Add initial greeting and suggestions if the chat window is empty
function initChat() {
    chatWindow.innerHTML = `
        <div class="welcome-card">
            <div class="welcome-icon">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #6366f1;">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <h3>Welcome to PulseDesk Support!</h3>
            <p>I am your AI assistant, grounded in our Knowledge Base. How can I help you today?</p>
            <div class="suggestions-container">
                <button class="suggestion-btn" onclick="sendSuggestion('How do I reset my password?')">Reset Password</button>
                <button class="suggestion-btn" onclick="sendSuggestion('What pricing plans does PulseDesk offer?')">Pricing Plans</button>
                <button class="suggestion-btn" onclick="sendSuggestion('How do I integrate PulseDesk with Slack?')">Slack Integration</button>
            </div>
        </div>
    `;
}

window.sendSuggestion = function(text) {
    userInput.value = text;
    chatForm.dispatchEvent(new Event('submit'));
};

function appendMessage(text, sender, sources = null) {
    // Remove welcome card on first user message
    const welcomeCard = chatWindow.querySelector(".welcome-card");
    if (welcomeCard && (sender === "user" || sender === "bot-loading")) {
        welcomeCard.remove();
    }

    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    if (sender === "user") {
        avatar.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
        `;
    } else {
        avatar.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
        `;
    }

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const msgDiv = document.createElement("div");
    msgDiv.className = `message-bubble`;
    
    if (sender === "bot-loading") {
        msgDiv.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
    } else {
        msgDiv.textContent = text;
    }

    contentDiv.appendChild(msgDiv);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "sources-container";
        
        const header = document.createElement("div");
        header.className = "sources-header";
        header.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px; display: inline-block; vertical-align: middle;">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
            </svg>
            <span>Grounded Sources (${sources.length})</span>
        `;
        sourcesDiv.appendChild(header);

        const list = document.createElement("div");
        list.className = "sources-list";
        
        sources.forEach(s => {
            const item = document.createElement("div");
            item.className = "source-item";
            
            const scorePercentage = Math.round(s.score * 100);
            
            item.innerHTML = `
                <div class="source-question">${s.question}</div>
                <div class="source-meta">
                    <div class="source-bar-wrapper">
                        <div class="source-bar" style="width: ${scorePercentage}%"></div>
                    </div>
                    <span class="source-score">Confidence: ${scorePercentage}%</span>
                </div>
            `;
            list.appendChild(item);
        });

        sourcesDiv.appendChild(list);
        contentDiv.appendChild(sourcesDiv);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(contentDiv);
    chatWindow.appendChild(wrapper);

    chatWindow.scrollTop = chatWindow.scrollHeight;
    return wrapper;
}

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage(message, "user");
    userInput.value = "";

    // Append loading placeholder
    const loadingWrapper = appendMessage("", "bot-loading");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });
        const data = await response.json();

        // remove the loading placeholder
        if (loadingWrapper && loadingWrapper.parentNode) {
            chatWindow.removeChild(loadingWrapper);
        }

        appendMessage(data.answer, "bot", data.sources);
    } catch (err) {
        // remove the loading placeholder
        if (loadingWrapper && loadingWrapper.parentNode) {
            chatWindow.removeChild(loadingWrapper);
        }
        appendMessage("Something went wrong. Please try again.", "bot");
        console.error(err);
    }
});

// Initialize welcome card
initChat();
