import base64
from typing import Optional, TypedDict
from datetime import datetime
import marshmallow as ma

class KeyModel(TypedDict):
    id: int
    file: str
    chat_id: int


class ErrorType(TypedDict):
    section: str
    message: str

class PostKeySchema(ma.Schema):
    id = ma.fields.Str(required=True)
    file = ma.fields.Str(required=True)
    chat_id = ma.fields.Int(required=True)

class PostContextSchema(ma.Schema):
    context = ma.fields.Str(required=True)
    chat_id = ma.fields.Int(required=True)

class PostKeyBodySchema(ma.Schema):
    file = ma.fields.Str(required=True)
    chat_id = ma.fields.Int(required=True)

class PostContextBodySchema(ma.Schema):
    context = ma.fields.Str(required=True)
    model = ma.fields.Str(required=True)
    chat_id = ma.fields.Int(required=True)

class GetClientSchema(ma.Schema):
    keys = ma.fields.Str(required=True)
    client_specs = ma.fields.Str(required=True)

class PostBodySecretSchema(ma.Schema):
    context = ma.fields.Str(required=True)

class PostBodyModelSchema(ma.Schema):
    context = ma.fields.Str(required=True)

class PostSecretSchema(ma.Schema):
    secret_key = ma.fields.Str(required=True)

class PostModelSchema(ma.Schema):
    model = ma.fields.Str(required=True)

class ErrorTypeSchema(ma.Schema):
    section = ma.fields.Str(required=True)
    message = ma.fields.Str(required=True)

class EncryptedBodyMessageSchema(ma.Schema):
    chat_id = ma.fields.Int(required=True)
    sender_id = ma.fields.Int(required=True)
    sender_name = ma.fields.Str(required=True)
    receiver_id = ma.fields.Int(required=True)
    content = ma.fields.Str(required=False)
    image = ma.fields.Str(required=False)
    iv = ma.fields.Str(required=False)
    image_to_classify = ma.fields.Str(required=False)
    type = ma.fields.Str(required=True)
    is_typing = ma.fields.Bool(required=False)
    timestamp = ma.fields.DateTime(required=True)
    classification_result = ma.fields.String(required=False)

class EncryptedMessageSchema(ma.Schema):
    chat_id = ma.fields.Int(required=True)
    sender_id = ma.fields.Int(required=True)
    sender_name = ma.fields.Str(required=True)
    receiver_id = ma.fields.Int(required=True)
    content = ma.fields.Str(required=True)
    iv = ma.fields.Str(required=False)
    image = ma.fields.Str(required=False)
    image_to_classify = ma.fields.Str(required=False)
    type = ma.fields.Str(required=True)
    is_typing = ma.fields.Bool(required=False)
    timestamp = ma.fields.DateTime(required=True)
    classification_result = ma.fields.String(required=True)

