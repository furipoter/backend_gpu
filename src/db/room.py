from src.db import db


class Room(db.Model):
    id = db.Column(db.String(64), primary_key=True)
    number = db.Column(db.Integer, nullable=False)
