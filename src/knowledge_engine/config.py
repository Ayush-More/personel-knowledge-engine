from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Personal Knowledge Engine"
    envireonment: str = "development"
    data_dir: str = "data"
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
     env_prefix="PKE_",
     env_file=".env",
     extra="ignore"   
    )

settings = Settings() 