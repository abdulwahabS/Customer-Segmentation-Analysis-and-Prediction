from flask import Blueprint, render_template, request, flash
from .model import predict, meta

ui = Blueprint("ui", __name__)

@ui.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            cluster = predict(request.form.to_dict())
            return render_template("result.html",
                                   cluster=cluster,
                                   meta=meta())
        except Exception as e:
            flash(f"⛔ {e}")
    return render_template("index.html", meta=meta())
