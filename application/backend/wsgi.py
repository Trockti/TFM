from api import app
import os

if __name__ == "__main__":
    # Para desarrollo local
    app.run(host='0.0.0.0', port=3232)
else:
    # Asegurar que las rutas relativas funcionen en producción
    # Gunicorn ya configurará esto cuando se ejecute con --chdir
    pass