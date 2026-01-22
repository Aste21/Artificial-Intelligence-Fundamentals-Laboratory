# Setup Instructions - Convolutional Neural Network

## Krok po kroku (Windows CMD)

### 1. Otwórz CMD (Command Prompt) i przejdź do folderu projektu:

```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.6_Convolutional_Neural_Network
```

### 2. Stwórz środowisko wirtualne (venv):

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

Lub jeśli pip nie działa:

```cmd
python -m pip install -r requirements.txt
```

### 5. Uruchom trening sieci:

```cmd
python train_cnn.py
```

---

## Krok po kroku (Windows PowerShell)

### 1. Otwórz PowerShell i przejdź do folderu projektu:

```powershell
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.6_Convolutional_Neural_Network
```

### 2. Stwórz środowisko wirtualne (venv):

```powershell
python -m venv venv
```

### 3. Aktywuj środowisko wirtualne:

```powershell
.\venv\Scripts\Activate.ps1
```

**Jeśli dostaniesz błąd o ExecutionPolicy, uruchom:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Po aktywacji powinieneś zobaczyć `(venv)` na początku linii komend.**

### 4. Zainstaluj zależności:

```powershell
pip install -r requirements.txt
```

### 5. Uruchom trening sieci:

```powershell
python train_cnn.py
```

---

## Szybkie komendy (kopiuj-wklej dla CMD):

```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.6_Convolutional_Neural_Network
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python train_cnn.py
```

**Uwaga:** Po każdej komendzie naciśnij Enter. Po aktywacji venv zobaczysz `(venv)` na początku linii.

---

## Szybkie komendy (kopiuj-wklej dla PowerShell):

```powershell
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.6_Convolutional_Neural_Network
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_cnn.py
```

---

## Testowanie komponentów

### Test custom convolution operator:

```cmd
python convolution.py
```

---

## Trening sieci

### Pełny trening (oba modele):

```cmd
python train_cnn.py
```

To uruchomi:
1. Trening prostej CNN (Conv → ReLU → FC)
2. Trening CNN z poolingiem (Conv → ReLU → MaxPool → FC → Softmax)

**Uwaga:** Pierwsze uruchomienie pobierze zbiór MNIST (może zająć kilka minut).

---

## Deaktywacja środowiska wirtualnego:

Gdy skończysz pracę, możesz deaktywować venv:

```cmd
deactivate
```

---

## Rozwiązywanie problemów

### Problem: "ModuleNotFoundError: No module named 'scipy'"

**Rozwiązanie:** Upewnij się, że venv jest aktywowany i zainstaluj zależności:
```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Problem: "ExecutionPolicy" w PowerShell

**Rozwiązanie:** Uruchom:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: "Failed to download MNIST dataset"

**Rozwiązanie:** 
- Sprawdź połączenie z internetem
- Upewnij się, że `kagglehub` jest zainstalowany: `pip install kagglehub`
- Dataset zostanie zapisany w cache po pierwszym pobraniu

### Problem: Trening trwa zbyt długo

**Rozwiązanie:** W pliku `train_cnn.py` możesz zmniejszyć liczbę próbek:
```python
n_samples = 10000  # Zamiast pełnego zbioru
```

---

## Struktura projektu

```
4.6_Convolutional_Neural_Network/
├── venv/                    # Środowisko wirtualne (po utworzeniu)
├── convolution.py           # Custom convolution operator
├── cnn_layers.py            # Warstwy konwolucyjne i pooling
├── cnn.py                   # Implementacja CNN
├── mnist_loader_2d.py       # Ładowanie MNIST (obrazy 2D)
├── train_cnn.py             # Skrypt treningowy
├── requirements.txt         # Zależności
├── README.md                # Dokumentacja
└── SETUP.md                 # Ten plik
```
