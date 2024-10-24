import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier

from app.core.config import settings


class MineralClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model_group_nedra = None
        self.model_name_gbz_tbz = None
        self.model_normalized_name = None
        self.model_measurement_unit = None
        self.model_measurement_unit_alt = None
        self.load_models()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def load_models(self):
        df = pd.read_excel(settings.DATA_PATH, sheet_name=0, header=3)
        df = df.iloc[:, 2:]

        # Rename columns using index-based approach
        new_column_names = [
            "pi_variants",
            "normalized_name_for_display",
            "pi_name_gbz_tbz",
            "pi_group_is_nedra",
            "pi_measurement_unit",
            "pi_measurement_unit_alternative",
        ]
        df.columns = new_column_names

        # Normalize column names (convert to lowercase and replace spaces with underscores)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Drop duplicate rows based on the 'pi_variants' column, keeping the first occurrence.
        df = df.drop_duplicates(subset=['pi_variants'], keep='first')

        # Keep only rows where 'pi_variants', 'normalized_name_for_display', 'pi_name_gbz_tbz', 'pi_group_is_nedra' are not null
        df = df.dropna(subset=['pi_variants', 'normalized_name_for_display', 'pi_name_gbz_tbz', 'pi_group_is_nedra'])

        # Обработка NaN и "-" значений для единиц измерения
        df['pi_measurement_unit'] = df['pi_measurement_unit'].replace('-', '')
        df['pi_measurement_unit'] = df['pi_measurement_unit'].fillna('')
        df['pi_measurement_unit_alternative'] = df['pi_measurement_unit_alternative'].replace('-', '')
        df['pi_measurement_unit_alternative'] = df['pi_measurement_unit_alternative'].fillna('')

        df['pi_variants'] = df['pi_variants'].apply(self.clean_text)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['pi_variants'])

        # Обучение модели для pi_group_is_nedra
        model_group_nedra = LogisticRegression()
        y_group_nedra = df['pi_group_is_nedra']
        X_train, X_test, y_train_nedra, y_test_nedra = train_test_split(X, y_group_nedra, test_size=0.2,
                                                                        random_state=42)
        model_group_nedra.fit(X_train, y_train_nedra)

        # Обучение модели для pi_name_gbz_tbz
        model_name_gbz_tbz = LogisticRegression()
        y_name_gbz_tbz = df['pi_name_gbz_tbz']
        X_train, X_test, y_train_gbz_tbz, y_test_gbz_tbz = train_test_split(X, y_name_gbz_tbz, test_size=0.2,
                                                                            random_state=42)
        model_name_gbz_tbz.fit(X_train, y_train_gbz_tbz)

        # Обучение модели для normalized_name_for_display
        # Для этой задачи лучше использовать модель для предсказания текста, например, RidgeClassifier
        model_normalized_name = RidgeClassifier()
        y_normalized_name = df['normalized_name_for_display']
        X_train, X_test, y_train_normalized_name, y_test_normalized_name = train_test_split(X, y_normalized_name,
                                                                                            test_size=0.2,
                                                                                            random_state=42)
        model_normalized_name.fit(X_train, y_train_normalized_name)

        # Добавляем новые модели для единиц измерения
        model_measurement_unit = LogisticRegression()
        y_measurement_unit = df['pi_measurement_unit']
        X_train, X_test, y_train_measurement, y_test_measurement = train_test_split(
            X, y_measurement_unit, test_size=0.2, random_state=42
        )
        model_measurement_unit.fit(X_train, y_train_measurement)

        model_measurement_unit_alt = LogisticRegression()
        y_measurement_unit_alt = df['pi_measurement_unit_alternative']

        # Обрабатываем NaN значения, заменяя их на пустую строку или другое значение по умолчанию
        y_measurement_unit_alt = y_measurement_unit_alt.fillna('')
        X_train, X_test, y_train_measurement_alt, y_test_measurement_alt = train_test_split(
            X, y_measurement_unit_alt, test_size=0.2, random_state=42
        )
        model_measurement_unit_alt.fit(X_train, y_train_measurement_alt)

    def classify(self, mineral_name: str) -> dict:
        cleaned_name = self.clean_text(mineral_name)
        vectorized_name = self.vectorizer.transform([cleaned_name])

        return {
            "mineral_name": mineral_name,
            "pi_group_is_nedra": self.model_group_nedra.predict(vectorized_name)[0],
            "pi_name_gbz_tbz": self.model_name_gbz_tbz.predict(vectorized_name)[0],
            "normalized_name_for_display": self.model_normalized_name.predict(vectorized_name)[0],
            "pi_measurement_unit": self.model_measurement_unit.predict(vectorized_name)[0],
            "pi_measurement_unit_alternative": self.model_measurement_unit_alt.predict(vectorized_name)[0]
        }


classifier = MineralClassifier()
