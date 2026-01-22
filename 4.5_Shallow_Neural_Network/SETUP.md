# Szybki setup - komendy do skopiowania

## Windows CMD (Command Prompt) - GŁÓWNE INSTRUKCJE

### 1. Przejdź do folderu projektu:
```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.5_Shallow_Neural_Network
```

### 2. Stwórz środowisko wirtualne:
```cmd
python -m venv venv
```

### 3. Aktywuj środowisko wirtualne:
```cmd
venv\Scripts\activate.bat
```

**Po aktywacji powinieneś zobaczyć `(venv)` na początku linii komend.**

### 4. Zainstaluj zależności:
```cmd
pip install -r requirements.txt
```

### 5. Uruchom aplikację:
```cmd
streamlit run gui.py --server.port 9001
```

---

## Windows PowerShell (alternatywa)

### 1. Przejdź do folderu projektu:
```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.5_Shallow_Neural_Network
```

### 2. Stwórz środowisko wirtualne:
```cmd
python -m venv venv
```

### 3. Aktywuj środowisko wirtualne:
```cmd
venv\Scripts\activate.bat
```

### 4. Zainstaluj zależności:
```cmd
pip install -r requirements.txt
```

### 5. Uruchom aplikację:
```cmd
streamlit run gui.py --server.port 9001
```

---

## Wszystko w jednej linii (CMD):

```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.5_Shallow_Neural_Network && python -m venv venv && venv\Scripts\activate.bat && pip install -r requirements.txt && streamlit run gui.py --server.port 9001
```

**Uwaga:** W CMD użyj `&&` zamiast `;` do łączenia komend.

---

## Sprawdzenie czy działa:

Po uruchomieniu aplikacji powinieneś zobaczyć w terminalu:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:9001
```

Aplikacja automatycznie otworzy się w przeglądarce.

---

## Deaktywacja venv (gdy skończysz):

```powershell
deactivate
```

---

## Jeśli coś nie działa:

1. **Sprawdź czy masz Python:**
   ```powershell
   python --version
   ```
   Powinno pokazać Python 3.7 lub wyższy.

2. **Sprawdź czy pip działa:**
   ```powershell
   pip --version
   ```

3. **Jeśli pip nie działa, użyj:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Jeśli streamlit nie działa:**
   ```powershell
   python -m streamlit run gui.py --server.port 9001
   ```
