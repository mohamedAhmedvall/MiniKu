import reflex as rx

config = rx.Config(
    app_name="miniku",
    # Backend accessible depuis le devcontainer / Docker
    backend_host="0.0.0.0",
    frontend_port=3000,
    backend_port=8000,
)
