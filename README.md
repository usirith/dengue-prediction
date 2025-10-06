# Dengue Prediction Dashboard

## How to Run

1. **Create an venv**

-   Linux
    ```bash
    python3 -m venv .venv
    ```
-   Windows (CMD)
    ```bash
    python -m venv .venv
    ```

2. **Activate venv**

-   Linux
    ```bash
    source ./.venv/bin/activate
    ```
-   Windows (CMD)
    ```cmd
    .venv\Scripts\activate
    ```
-   Windows (Powershell)
    ```cmd
    .venv\Scripts\Activate.ps1
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the training script first:**

    ```bash
    python train.py
    ```

5. **Launch the dashboard:**

    ```bash
    streamlit run app.py --server.port 8000
    ```

6. **Navigate to Dashboard**
    ```
    Goto localhost:8000
    ```
