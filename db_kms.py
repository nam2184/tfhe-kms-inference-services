from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime
from typing import Optional, List


class KeyModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    file: str
    chat_id: int

class ClientHEModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    client_specs: str
    file: str
    chat_id: int = Field(unique=True) 

class KeysDBService:
    def __init__(self, db_path: str = "sqlite:///keys.db"):
        self.engine = create_engine(db_path, echo=False)
        SQLModel.metadata.create_all(self.engine)

    # CRUD for KeyModel
    def insert_key(self, key: KeyModel) -> KeyModel:
        with Session(self.engine) as session:
            session.add(key)
            session.commit()
            session.refresh(key)
            return key

    def get_key_by_chat_id(self, chat_id: int) -> KeyModel | None:
        with Session(self.engine) as session:
            stmt = select(KeyModel).where(KeyModel.chat_id == chat_id)
            return session.exec(stmt).first()

    def delete_key_by_chat_id(self, chat_id: int) -> bool:
        with Session(self.engine) as session:
            stmt = select(KeyModel).where(KeyModel.chat_id == chat_id)
            key = session.scalars(stmt).first()
            if key:
                session.delete(key)
                session.commit()
                return True
            return False

    # CRUD for ClientHEModel
    def insert_heclient_key(self, key: ClientHEModel) -> ClientHEModel:
        with Session(self.engine) as session:
            session.add(key)
            session.commit()
            session.refresh(key)
            return key

    def get_heclient_key_by_chat_id(self, chat_id: int) -> ClientHEModel | None:
        with Session(self.engine) as session:
            stmt = select(ClientHEModel).where(ClientHEModel.chat_id == chat_id)
            return session.exec(stmt).first()

    def delete_heclient_key_by_chat_id(self, chat_id: int) -> bool:
        with Session(self.engine) as session:
            stmt = select(ClientHEModel).where(ClientHEModel.chat_id == chat_id)
            key = session.scalars(stmt).first()
            if key:
                session.delete(key)
                session.commit()
                return True
            return False
