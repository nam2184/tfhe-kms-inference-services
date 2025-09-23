from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime
from typing import Optional, List


class KeyModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    file: str
    chat_id: int

class ClientHEModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    file: str
    chat_id: int = Field(unique=True) 

class HomomorphicKeyModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    file: str
    chat_id: int = Field(unique=True) 

class DBService:
    def __init__(self, db_path: str = "sqlite:///keys.db"):
        self.engine = create_engine(db_path, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def insert_key(self, key : KeyModel) -> KeyModel:
        with Session(self.engine) as session:
            session.add(key)
            session.commit()
            session.refresh(key)
            return key

    def get_key_by_chat_id(self, chat_id: int) -> KeyModel | None:
         with Session(self.engine) as session:
            stmt = select(KeyModel).where(KeyModel.chat_id == chat_id)
            return session.exec(stmt).first()

    def delete_key_by_id(self, key_id: int) -> bool:
        with Session(self.engine) as session:
            key = session.get(KeyModel, key_id)
            if key:
                session.delete(key)
                session.commit()
                return True
            return False


    from sqlalchemy import select

    def delete_key_by_chat_id(self, chat_id: int) -> bool:
        with Session(self.engine) as session:
            stmt = select(KeyModel).where(KeyModel.chat_id == chat_id)
            key = session.scalars(stmt).first()
            if key:
                session.delete(key)
                session.commit()
                return True
            return False
    
    def insert_homomorphic_key(self, key : HomomorphicKeyModel) -> HomomorphicKeyModel:
        with Session(self.engine) as session:
            session.add(key)
            session.commit()
            session.refresh(key)
            return key

    def get_homomorphic_key_by_chat_id(self, chat_id: int) -> HomomorphicKeyModel | None:
         with Session(self.engine) as session:
            stmt = select(HomomorphicKeyModel).where(HomomorphicKeyModel.chat_id == chat_id)
            return session.exec(stmt).first()

    def delete_homomorphic_key_by_id(self, key_id: int) -> bool:
        with Session(self.engine) as session:
            key = session.get(HomomorphicKeyModel, key_id)
            if key:
                session.delete(key)
                session.commit()
                return True
            return False


    from sqlalchemy import select

    def delete_homomorphic_key_by_chat_id(self, chat_id: int) -> bool:
        with Session(self.engine) as session:
            stmt = select(HomomorphicKeyModel).where(HomomorphicKeyModel.chat_id == chat_id)
            key = session.scalars(stmt).first()
            if key:
                session.delete(key)
                session.commit()
                return True
            return False
    
    def insert_heclient_key(self, key : ClientHEModel) -> ClientHEModel:
        with Session(self.engine) as session:
            session.add(key)
            session.commit()
            session.refresh(key)
            return key

    def get_heclient_key_by_chat_id(self, chat_id: int) -> ClientHEModel | None:
         with Session(self.engine) as session:
            stmt = select(ClientHEModel).where(ClientHEModel.chat_id == chat_id)
            return session.exec(stmt).first()

    def delete_heclient_key_by_id(self, key_id: int) -> bool:
        with Session(self.engine) as session:
            key = session.get(ClientHEModel, key_id)
            if key:
                session.delete(key)
                session.commit()
                return True
            return False


    from sqlalchemy import select

    def delete_heclient_key_by_chat_id(self, chat_id: int) -> bool:
        with Session(self.engine) as session:
            stmt = select(ClientHEModel).where(ClientHEModel.chat_id == chat_id)
            key = session.scalars(stmt).first()
            if key:
                session.delete(key)
                session.commit()
                return True
            return False
    
    
