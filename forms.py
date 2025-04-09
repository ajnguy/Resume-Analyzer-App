from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import ValidationError, DataRequired
from models import Todo

class TodoForm(FlaskForm):
    todo = StringField("Todo")
    submit = SubmitField("Add Todo")