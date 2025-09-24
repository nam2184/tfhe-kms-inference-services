# db_homomorphic.py
from sqlmodel import SQLModel, Field,create_engine, Session, select


class HomomorphicKeyModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    file: str
    chat_id: int = Field(unique=True) 


class HomomorphicDBService:
    def __init__(self, db_path: str = "sqlite:///homomorphic_keys.db"):
        self.engine = create_engine(db_path, echo=False)
        SQLModel.metadata.create_all(self.engine)

    def insert_homomorphic_key(self, key: HomomorphicKeyModel) -> HomomorphicKeyModel:
        with Session(self.engine) as session:
            session.add(key)
            session.commit()
            session.refresh(key)
            return key

    def get_homomorphic_key_by_chat_id(self, chat_id: int) -> HomomorphicKeyModel | None:
        with Session(self.engine) as session:
            stmt = select(HomomorphicKeyModel).where(HomomorphicKeyModel.chat_id == chat_id)
            return session.exec(stmt).first()

    def delete_homomorphic_key_by_chat_id(self, chat_id: int) -> bool:
        with Session(self.engine) as session:
            stmt = select(HomomorphicKeyModel).where(HomomorphicKeyModel.chat_id == chat_id)
            key = session.scalars(stmt).first()
            if key:
                session.delete(key)
                session.commit()
                return True
            return False
