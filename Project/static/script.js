async function sendMessage() {
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    if (!input.value.trim()) return;

    // Display user message
    chatBox.innerHTML += `<div class="message user">${input.value}</div>`;
    const userMsg = input.value;
    input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    // Fetch from Flask backend
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg })
    });
    
    const data = await response.json();
    chatBox.innerHTML += `<div class="message ai">${data.reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
}