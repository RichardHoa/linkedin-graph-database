// Replace the existing sendMessage function
async function sendMessage() {
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const userMsg = input.value.trim();
    
    if (!userMsg) return;

    // Display user message
    chatBox.innerHTML += `<div class="message user">${userMsg}</div>`;
    input.value = '';

    // Add loading indicator
    const loadingId = 'loading-' + Date.now();
    chatBox.innerHTML += `<div id="${loadingId}" class="message ai">Please wait for the response</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMsg })
        });
        
        const data = await response.json();
        
        // Replace loading indicator with AI response
        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();
        
        chatBox.innerHTML += `<div class="message ai">${data.reply}</div>`;
    } catch (error) {
        const loader = document.getElementById(loadingId);
        if (loader) loader.innerText = "System error. Please check server logs.";
    }
    
    chatBox.scrollTop = chatBox.scrollHeight;
}