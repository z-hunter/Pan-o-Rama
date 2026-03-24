/**
 * Common API fetch wrapper for Pan-o-Rama
 */
async function apiFetch(url, opts = {}) {
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    
    if (res.status === 401 && !window.location.pathname.includes('/login')) {
        window.location.href = '/login';
        return null;
    }
    
    return { res, data };
}

/**
 * Logout the current user and redirect to login
 */
async function logout() {
    await apiFetch('/auth/logout', { method: 'POST' });
    window.location.href = '/login';
}

/**
 * Utility for showing notifications (simple for now)
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Check if container exists
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 500);
    }, 4000);
}
