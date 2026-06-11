from sqlalchemy import Integer, String, BigInteger
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class Detection(Base):
    """SQLAlchemy model for the detections table."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_name: Mapped[str] = mapped_column(String(50), nullable=False)
    num_persone: Mapped[int] = mapped_column(Integer, nullable=False)
    emotion: Mapped[str | None] = mapped_column(String(20), nullable=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
