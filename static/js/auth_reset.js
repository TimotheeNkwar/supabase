// Password reset helpers and wiring for forgot_password.html

async function requestPasswordReset(email) {
  if (!email) throw new Error('Email requis');
  const res = await fetch('/api/auth/request-reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: String(email).trim().toLowerCase() })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Erreur envoi du code');
  return data;
}

async function resetPassword(email, code, newPassword) {
  if (!email || !code || !newPassword) throw new Error('Champs requis');
  if (!/^\d{6}$/.test(code)) throw new Error('Code invalide (6 chiffres)');
  if (newPassword.length < 8) throw new Error('Mot de passe trop court');
  const res = await fetch('/api/auth/reset-password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: String(email).trim().toLowerCase(),
      code: String(code).trim(),
      new_password: newPassword
    })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Reset error');
  return data;
}

async function verifyResetCode(email, code) {
  if (!email || !code) throw new Error('Champs requis');
  if (!/^\d{6}$/.test(code)) throw new Error('Code invalide (6 chiffres)');
  const res = await fetch('/api/auth/verify-reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: String(email).trim().toLowerCase(), code: String(code).trim() })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Verification failed');
  return data; // { valid: true }
}

// Auto-wire if the page contains the expected forms/fields
document.addEventListener('DOMContentLoaded', () => {
  const emailInput = document.getElementById('email');
  const requestForm = document.getElementById('request-form');
  const requestMsg = document.getElementById('request-msg');
  const resetForm = document.getElementById('reset-form');
  const resetMsg = document.getElementById('reset-msg');
  const step2Section = resetForm ? resetForm.closest('section') : null;
  const verifyBlock = document.getElementById('verify-block');
  const verifyBtn = document.getElementById('verify-btn');
  const verifyMsg = document.getElementById('verify-msg');
  const codeStep1 = document.getElementById('code_step1');

  // Hide Step 2 by default; reveal only after verification
  if (step2Section) step2Section.style.display = 'none';

  if (requestForm && emailInput && requestMsg) {
    requestForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      requestMsg.textContent = '';
      try {
        const data = await requestPasswordReset(emailInput.value);
        requestMsg.textContent = data.message || 'Code sent (if account exists)';
      } catch (err) {
        requestMsg.textContent = err.message || 'Erreur réseau';
        return;
      }
      // Show inline verification block
      if (verifyBlock) verifyBlock.style.display = '';
    });
  }

  if (resetForm && emailInput && resetMsg) {
    resetForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      resetMsg.textContent = '';
      const codeEl = document.getElementById('code');
      const passEl = document.getElementById('new_password');
      try {
        const effectiveEmail = (sessionStorage.getItem('resetEmail') || emailInput.value);
        const data = await resetPassword(effectiveEmail, codeEl.value, passEl.value);
        resetMsg.textContent = data.message || 'Password updated';
      } catch (err) {
        resetMsg.textContent = err.message || 'Erreur réseau';
      }
    });
  }

  if (verifyBtn && codeStep1 && verifyMsg) {
    verifyBtn.addEventListener('click', async () => {
      verifyMsg.textContent = '';
      try {
        await verifyResetCode(emailInput.value, codeStep1.value);
        try { sessionStorage.setItem('resetEmail', String(emailInput.value).trim().toLowerCase()); } catch (_) {}
        if (step2Section) step2Section.style.display = '';
        verifyMsg.textContent = 'Code validé. Vous pouvez définir un nouveau mot de passe.';
      } catch (err) {
        verifyMsg.textContent = err.message || 'Code invalide ou expiré';
      }
    });
  }
});


