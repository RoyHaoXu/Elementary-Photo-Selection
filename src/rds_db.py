import logging

import pandas as pd
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)
Base = declarative_base()


class PhotoStyleFeatures(Base):
    """Data model for the database to be set up for capturing photo style features。"""

    __tablename__ = 'style_features'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=False, nullable=False)
    contrast = Column(Float, unique=False, nullable=False)
    B_shadow = Column(Float, unique=False, nullable=False)
    B_dark = Column(Float, unique=False, nullable=False)
    B_light = Column(Float, unique=False, nullable=False)
    B_highlight = Column(Float, unique=False, nullable=False)
    G_shadow = Column(Float, unique=False, nullable=False)
    G_dark = Column(Float, unique=False, nullable=False)
    G_light = Column(Float, unique=False, nullable=False)
    G_highlight = Column(Float, unique=False, nullable=False)
    R_shadow = Column(Float, unique=False, nullable=False)
    R_dark = Column(Float, unique=False, nullable=False)
    R_light = Column(Float, unique=False, nullable=False)
    R_highlight = Column(Float, unique=False, nullable=False)
    B_average = Column(Float, unique=False, nullable=False)
    G_average = Column(Float, unique=False, nullable=False)
    R_average = Column(Float, unique=False, nullable=False)
    sharpness = Column(Float, unique=False, nullable=False)
    red_average = Column(Float, unique=False, nullable=False)
    orange_average = Column(Float, unique=False, nullable=False)
    yellow_average = Column(Float, unique=False, nullable=False)
    green_average = Column(Float, unique=False, nullable=False)
    cyan_average = Column(Float, unique=False, nullable=False)
    blue_average = Column(Float, unique=False, nullable=False)
    purple_average = Column(Float, unique=False, nullable=False)


class PhotoObjectFeatures(Base):
    """Data model for the database to be set up for capturing photo object features。"""

    __tablename__ = 'object_features'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=False, nullable=False)
    pc1 = Column(Float, unique=False, nullable=False)
    pc2 = Column(Float, unique=False, nullable=False)
    pc3 = Column(Float, unique=False, nullable=False)
    pc4 = Column(Float, unique=False, nullable=False)
    pc5 = Column(Float, unique=False, nullable=False)
    pc6 = Column(Float, unique=False, nullable=False)
    pc7 = Column(Float, unique=False, nullable=False)
    pc8 = Column(Float, unique=False, nullable=False)
    pc9 = Column(Float, unique=False, nullable=False)
    pc10 = Column(Float, unique=False, nullable=False)
    pc11 = Column(Float, unique=False, nullable=False)
    pc12 = Column(Float, unique=False, nullable=False)
    pc13 = Column(Float, unique=False, nullable=False)
    pc14 = Column(Float, unique=False, nullable=False)
    pc15 = Column(Float, unique=False, nullable=False)
    pc16 = Column(Float, unique=False, nullable=False)
    pc17 = Column(Float, unique=False, nullable=False)
    pc18 = Column(Float, unique=False, nullable=False)
    pc19 = Column(Float, unique=False, nullable=False)
    pc20 = Column(Float, unique=False, nullable=False)
    pc21 = Column(Float, unique=False, nullable=False)
    pc22 = Column(Float, unique=False, nullable=False)
    pc23 = Column(Float, unique=False, nullable=False)
    pc24 = Column(Float, unique=False, nullable=False)
    pc25 = Column(Float, unique=False, nullable=False)
    pc26 = Column(Float, unique=False, nullable=False)
    pc27 = Column(Float, unique=False, nullable=False)
    pc28 = Column(Float, unique=False, nullable=False)
    pc29 = Column(Float, unique=False, nullable=False)
    pc30 = Column(Float, unique=False, nullable=False)


class PhotoClusters(Base):
    """Data model for the database to be set up for capturing photo clusters features。"""

    __tablename__ = 'photo_clusters'

    id = Column(Integer, primary_key=True)
    images_names = Column(String(500), unique=False, nullable=False)
    contrast = Column(Float, unique=False, nullable=False)
    B_shadow = Column(Float, unique=False, nullable=False)
    B_dark = Column(Float, unique=False, nullable=False)
    B_light = Column(Float, unique=False, nullable=False)
    B_highlight = Column(Float, unique=False, nullable=False)
    G_shadow = Column(Float, unique=False, nullable=False)
    G_dark = Column(Float, unique=False, nullable=False)
    G_light = Column(Float, unique=False, nullable=False)
    G_highlight = Column(Float, unique=False, nullable=False)
    R_shadow = Column(Float, unique=False, nullable=False)
    R_dark = Column(Float, unique=False, nullable=False)
    R_light = Column(Float, unique=False, nullable=False)
    R_highlight = Column(Float, unique=False, nullable=False)
    B_average = Column(Float, unique=False, nullable=False)
    G_average = Column(Float, unique=False, nullable=False)
    R_average = Column(Float, unique=False, nullable=False)
    sharpness = Column(Float, unique=False, nullable=False)
    red_average = Column(Float, unique=False, nullable=False)
    orange_average = Column(Float, unique=False, nullable=False)
    yellow_average = Column(Float, unique=False, nullable=False)
    green_average = Column(Float, unique=False, nullable=False)
    cyan_average = Column(Float, unique=False, nullable=False)
    blue_average = Column(Float, unique=False, nullable=False)
    purple_average = Column(Float, unique=False, nullable=False)
    pc1 = Column(Float, unique=False, nullable=False)
    pc2 = Column(Float, unique=False, nullable=False)
    pc3 = Column(Float, unique=False, nullable=False)
    pc4 = Column(Float, unique=False, nullable=False)
    pc5 = Column(Float, unique=False, nullable=False)
    pc6 = Column(Float, unique=False, nullable=False)
    pc7 = Column(Float, unique=False, nullable=False)
    pc8 = Column(Float, unique=False, nullable=False)
    pc9 = Column(Float, unique=False, nullable=False)
    pc10 = Column(Float, unique=False, nullable=False)
    pc11 = Column(Float, unique=False, nullable=False)
    pc12 = Column(Float, unique=False, nullable=False)
    pc13 = Column(Float, unique=False, nullable=False)
    pc14 = Column(Float, unique=False, nullable=False)
    pc15 = Column(Float, unique=False, nullable=False)
    pc16 = Column(Float, unique=False, nullable=False)
    pc17 = Column(Float, unique=False, nullable=False)
    pc18 = Column(Float, unique=False, nullable=False)
    pc19 = Column(Float, unique=False, nullable=False)
    pc20 = Column(Float, unique=False, nullable=False)
    pc21 = Column(Float, unique=False, nullable=False)
    pc22 = Column(Float, unique=False, nullable=False)
    pc23 = Column(Float, unique=False, nullable=False)
    pc24 = Column(Float, unique=False, nullable=False)
    pc25 = Column(Float, unique=False, nullable=False)
    pc26 = Column(Float, unique=False, nullable=False)
    pc27 = Column(Float, unique=False, nullable=False)
    pc28 = Column(Float, unique=False, nullable=False)
    pc29 = Column(Float, unique=False, nullable=False)
    pc30 = Column(Float, unique=False, nullable=False)


def create_db(engine_string):
    """Create database in RDS or local with feature tables.

    Args:
        engine_string (str): engine string for database's creation.

    Returns:
        None
    """

    # Connect to RDS
    engine = sqlalchemy.create_engine(engine_string)

    # Create schema
    PhotoStyleFeatures.metadata.create_all(engine)
    PhotoObjectFeatures.metadata.create_all(engine)
    PhotoClusters.metadata.create_all(engine)
    logger.info("Database created.")


class PhotoManager:

    def __init__(self, app=None, engine_string=None):
        """
        Args:
            app Flask: Flask app
            engine_string (str): Engine string
        """
        if app:
            self.db = SQLAlchemy(app)
            self.session = self.db.session
        elif engine_string:
            engine = sqlalchemy.create_engine(engine_string)
            Session = sessionmaker(bind=engine)
            self.session = Session()
        else:
            raise ValueError("Need either an engine string or a Flask app to initialize")

    def close(self):
        """Closes session
        Returns: None
        """
        self.session.close()

    def add_style_feature_row(self, name, contrast, B_shadow, B_dark, B_light, B_highlight,
                              G_shadow, G_dark, G_light, G_highlight, R_shadow, R_dark,
                              R_light, R_highlight, B_average, G_average, R_average, sharpness,
                              red_average, orange_average, yellow_average, green_average, cyan_average,
                              blue_average, purple_average):
        """Add new row to style_features table."""

        session = self.session
        row = PhotoStyleFeatures(name=name, contrast=contrast, B_shadow=B_shadow, B_dark=B_dark,
                                 B_light=B_light, B_highlight=B_highlight, G_shadow=G_shadow,
                                 G_dark=G_dark, G_light=G_light, G_highlight=G_highlight,
                                 R_shadow=R_shadow, R_dark=R_dark, R_light=R_light, R_highlight=R_highlight,
                                 B_average=B_average, G_average=G_average, R_average=R_average,
                                 sharpness=sharpness, red_average=red_average, orange_average=orange_average,
                                 yellow_average=yellow_average, green_average=green_average, cyan_average=cyan_average,
                                 blue_average=blue_average, purple_average=purple_average)

        session.add(row)
        session.commit()
        logger.info('Photo style features added to database.')

    def add_object_feature_row(self, name, pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9,
                               pc10, pc11, pc12, pc13, pc14, pc15, pc16, pc17, pc18, pc19, pc20,
                               pc21, pc22, pc23, pc24, pc25, pc26, pc27, pc28, pc29, pc30):
        """Add new row to object_features table."""

        session = self.session
        row = PhotoObjectFeatures(name=name, pc1=pc1, pc2=pc2, pc3=pc3, pc4=pc4, pc5=pc5, pc6=pc6,
                                  pc7=pc7, pc8=pc8, pc9=pc9, pc10=pc10, pc11=pc11, pc12=pc12, pc13=pc13,
                                  pc14=pc14, pc15=pc15, pc16=pc16, pc17=pc17, pc18=pc18, pc19=pc19, pc20=pc20,
                                  pc21=pc21, pc22=pc22, pc23=pc23, pc24=pc24, pc25=pc25, pc26=pc26, pc27=pc27,
                                  pc28=pc28, pc29=pc29, pc30=pc30)

        session.add(row)
        session.commit()
        logger.info('Photo style features added to database.')

    def add_offline_df(self, input_path, table_name, truncate=0):
        """
        Add offline dataframes to RDS.

        Args:
            input_path(str): path to the file that need to be uploaded
            table_name(str): name of the table that the data will be injected into
            truncate(int): whether truncate existing tables

        Returns:
            None
        """
        session = self.session

        if truncate == 1:
            session.execute(f'DELETE FROM {table_name}')
            logger.info(f"Truncated {table_name} table.")

        data = pd.read_csv(input_path, index_col=0)

        records = data.T.to_dict()
        rows = []

        for record in records:
            if table_name == 'object_features':
                rows.append(PhotoObjectFeatures(name=record, **records[record]))
            elif table_name == 'style_features':
                rows.append(PhotoStyleFeatures(name=record, **records[record]))
            elif table_name == 'photo_clusters':
                rows.append(PhotoClusters(**records[record]))

        session.add_all(rows)
        session.commit()
        logger.info(f'{len(rows)} records from {table_name} were added to the table')
