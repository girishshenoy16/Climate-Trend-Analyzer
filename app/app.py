from flask import Flask
from routes import register_routes


def create_app():

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    app.config["SECRET_KEY"] = "climate-trend-analyzer"

    register_routes(app)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )