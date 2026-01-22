import sys
import os
import torch
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Импорт VAE
class StudentVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=1):
        super(StudentVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class StudentAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.init_ui()
        self.load_model()
        
    def init_ui(self):
        self.setWindowTitle("Анализ лояльности студентов")
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Заголовок
        title_label = QLabel("Ввод данных студента")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(title_label)
        
         # Поля ввода (от 1 до 5)
        self.feature_inputs = []
        feature_descriptions = [
            "Активность на платформе (1-5)",
            "Успеваемость (1-5)",
            "Посещаемость занятий (1-5)",
            "Вовлеченность в сообщество (1-5)",
            "Использование доп. материалов (1-5)",
            "Качество выполненных работ (1-5)",
            "Продолжительность обучения (1-5)"
        ]
        
        for desc in feature_descriptions:
            label = QLabel(desc)
            label.setWordWrap(True)
            left_layout.addWidget(label)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(1.0, 5.0)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(1)
            spinbox.setValue(3.0)  # Среднее значение по умолчанию
            spinbox.setStyleSheet("padding: 5px;")
            left_layout.addWidget(spinbox)
            self.feature_inputs.append(spinbox)
            
            left_layout.addSpacing(5)
        
        # Поле для имени студента
        left_layout.addWidget(QLabel("Имя студента:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Введите имя студента")
        self.name_input.setStyleSheet("padding: 5px;")
        left_layout.addWidget(self.name_input)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        analyze_btn = QPushButton("Анализировать")
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        analyze_btn.clicked.connect(self.analyze_student)
        
        button_layout.addWidget(analyze_btn)
        button_layout.addWidget(save_btn)
        left_layout.addLayout(button_layout)
        
        # Групповой анализ
        group_btn = QPushButton("Анализ группы (5 студентов)")
        group_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        group_btn.clicked.connect(self.analyze_group)
        left_layout.addWidget(group_btn)
        
        # Статус
        self.status_label = QLabel("Анализ")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #666; padding: 10px;")
        left_layout.addWidget(self.status_label)
        
        # Правая панель
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        right_layout = QVBoxLayout(right_panel)
        
        result_title = QLabel("Результаты анализа")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        right_layout.addWidget(result_title)
        
        # Текстовое поле для результатов
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: Consolas, Monaco, monospace;
                font-size: 11px;
            }
        """)
        right_layout.addWidget(self.result_text)
        group_btn.clicked.connect(self.analyze_group)
        left_layout.addWidget(group_btn)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Правая панель
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        right_layout = QVBoxLayout(right_panel)
        
        result_title = QLabel("Результаты анализа")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        right_layout.addWidget(result_title)
        
        # Текстовое поле для результатов
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 10px;
                font-family: Consolas, Monaco, monospace;
                font-size: 11px;
            }
        """)
        right_layout.addWidget(self.result_text)
        
        # Добавляем панели в главный layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Инициализируем список результатов
        self.results = []
        
    def load_model(self):
        try:
            self.model = StudentVAE(input_dim=9, hidden_dim=16, latent_dim=1)  # Исправлено: input_dim=9
            checkpoint = torch.load('student_model_vae_best.pth', map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.status_label.setText("Модель загружена успешно")
            self.status_label.setStyleSheet("color: #4CAF50; padding: 10px;")
        except Exception as e:
            self.status_label.setText(f"Ошибка загрузки модели: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336; padding: 10px;")
    
    def create_student_from_inputs(self):
        # Получаем 7 основных признаков от пользователя
        main_features = []
        for spinbox in self.feature_inputs:
            main_features.append(spinbox.value())
        
        # Дополняем до 9 признаков
        enrollment_change = 1.0  # Среднее изменение набора
        assessment_rate = 0.8   # Средняя доля оцененных работ
        
        full_features = np.concatenate([main_features, [enrollment_change, assessment_rate]])
        return np.array(full_features, dtype=np.float32)
        return np.array(features, dtype=np.float32)
    
    def analyze_student_profile(self, student_data):
        avg_score = np.mean(student_data)
        min_score = np.min(student_data)
        weak_features = np.sum(student_data < -1.0)
        strong_features = np.sum(student_data > 1.0)
        
        if weak_features >= 4:
            return "Критический"
        elif avg_score < -0.8:
            return "Низкий"
        elif avg_score < -0.3:
            return "Ниже среднего"
        elif avg_score < 0.3:
            return "Средний"
        elif avg_score < 0.8:
            return "Выше среднего"
        else:
            return "Высокий"
    
    def predict_student_segment(self, student_data, student_name):
        if self.model is None:
            return None
        
        self.model.eval()
        with torch.no_grad():
            student_tensor = torch.FloatTensor(student_data).unsqueeze(0)
            recon, mu, logvar = self.model(student_tensor)
            latent_value = mu.item()
            
            # Анализ профиля
            avg_score = np.mean(student_data)
            weak_features = np.sum(student_data < -1.0)
            min_score = np.min(student_data)
            strong_features = np.sum(student_data > 1.0)
            
            if weak_features >= 4 or min_score < -1.5:
                cluster = 0
                segment_name = "Проблемные студенты"
                risk_level = "Высокий риск"
                base_loyalty = 82.0
                description = "Студенты с критически низкими показателями по ключевым метрикам"
                actions = [
                    "- Срочные персональные консультации",
                    "- Индивидуальный план восстановления",
                    "- Повышенное внимание куратора",
                    "- Специальные мотивационные программы"
                ]
            elif avg_score < -0.5:
                cluster = 3  
                segment_name = "Рискованные студенты"
                risk_level = "Средний риск"
                base_loyalty = 86.0
                description = "Студенты с нестабильными или низкими показателями"
                actions = [
                    "- Групповые занятия и воркшопы",
                    "- Регулярный мониторинг прогресса",
                    "- Дополнительные учебные материалы",
                    "- Мотивационные рассылки"
                ]
            elif avg_score > 0.8 and strong_features >= 3 and min_score > -0.3:
                cluster = 2
                segment_name = "Перспективные студенты" 
                risk_level = "Очень низкий риск"
                base_loyalty = 94.0
                description = "Студенты со стабильно высокими показателями"
                actions = [
                    "- Премиальные программы обучения",
                    "- Реферальные бонусы",
                    "- Участие в менторских программах",
                    "- Приоритетный доступ к новым курсам"
                ]
            else:
                cluster = 1
                segment_name = "Обычные студенты"
                risk_level = "Низкий риск"
                base_loyalty = 90.0
                description = "Студенты со стабильными средними показателями"
                actions = [
                    "- Стандартные программы обучения",
                    "- Групповые активности",
                    "- Регулярные опросы удовлетворенности",
                    "- Доступ ко всем базовым ресурсам"
                ]
            
            # Корректировка лояльности на основе латентного значения
            loyalty_adjustment = latent_value * 1.5  
            predicted_loyalty = max(75, min(98, base_loyalty + loyalty_adjustment))
            
            # Анализ признаков
            feature_analysis = self.analyze_features_detailed(student_data)
            
            result = {
                'student_name': student_name,
                'latent_value': latent_value,
                'cluster': cluster,
                'segment_name': segment_name,
                'risk_level': risk_level,
                'predicted_loyalty': round(predicted_loyalty, 1),
                'description': description,
                'recommended_actions': actions,
                'avg_score': avg_score,
                'weak_features': weak_features,
                'strong_features': strong_features,
                'profile': self.analyze_student_profile(student_data),
                'feature_analysis': feature_analysis,
                'student_data': student_data.copy()
            }
            
            return result
    
    def analyze_features_detailed(self, student_data):
        """Детальный анализ признаков"""
        feature_names = [
            "Активность на платформе",
            "Успеваемость",
            "Посещаемость занятий",
            "Вовлеченность в сообщество",
            "Использование доп. материалов",
            "Качество выполненных работ",
            "Продолжительность обучения"
        ]
        
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'neutral': [],
            'recommendations': []
        }
        
        for i, value in enumerate(student_data):
            feature_name = feature_names[i]
            
            if value > 1.0:
                analysis['strengths'].append(f"{feature_name}: {value:.2f} (сильно выше среднего)")
            elif value > 0.5:
                analysis['strengths'].append(f"{feature_name}: {value:.2f} (выше среднего)")
            elif value < -1.0:
                analysis['weaknesses'].append(f"{feature_name}: {value:.2f} (критически низкий)")
            elif value < -0.5:
                analysis['weaknesses'].append(f"{feature_name}: {value:.2f} (ниже среднего)")
            else:
                analysis['neutral'].append(f"{feature_name}: {value:.2f} (в норме)")
        
        # Рекомендации
        if len(analysis['weaknesses']) >= 3:
            analysis['recommendations'].append("Требуется комплексная программа поддержки")
        if any("Активность" in strength for strength in analysis['strengths']):
            analysis['recommendations'].append("Можно привлекать к менторской деятельности")
        if any("Успеваемость" in weakness for weakness in analysis['weaknesses']):
            analysis['recommendations'].append("Рекомендуются дополнительные занятия")
        
        return analysis
    
    def analyze_student(self):
        student_name = self.name_input.text().strip()
        if not student_name:
            student_name = "Анонимный студент"
        
        student_data = self.create_student_from_inputs()
        result = self.predict_student_segment(student_data, student_name)
        
        if result:
            self.results = [result]  # Сохраняем для возможного экспорта
            self.display_result(result)
            self.status_label.setText(f"Анализ завершен для {student_name}")
            self.status_label.setStyleSheet("color: #4CAF50; padding: 10px;")
    
    def analyze_group(self):
        # Создаем тестовых студентов для теста группы
        test_students = [
            ([-1.8, -2.0, -1.5, -1.2, -0.8, -1.0, -0.5], "Алексей Петров (проблемный)"),
            ([0.3, 0.5, 0.2, 0.1, 0.4, 0.3, 0.6], "Мария Сидорова (стандартная)"),
            ([1.8, 2.1, 1.5, 1.2, 1.4, 2.0, 1.3], "Дмитрий Иванов (перспективный)"),
            ([-1.2, 0.7, -0.9, -0.7, 0.2, -0.6, -0.8], "Анна Козлова (рискованная)"),
            ([0.8, 1.2, 0.9, 0.6, 0.7, 1.1, 1.0], "Елена Смирнова (стабильная)")
        ]
        
        group_results = []
        result_text = "="*60 + "\nАнализ группы студентов\n" + "="*60 + "\n\n"
        
        for student_data, student_name in test_students:
            result = self.predict_student_segment(np.array(student_data), student_name)
            if result:
                group_results.append(result)
                result_text += self.format_result_text(result)
                result_text += "\n" + "-"*60 + "\n\n"
        
        self.results = group_results
        self.result_text.setText(result_text)
        
        # Статистика
        stats_text = self.calculate_group_statistics(group_results)
        result_text += "\n" + "="*60 + "\nСТАТИСТИКА ГРУППЫ\n" + "="*60 + "\n\n"
        result_text += stats_text
        
        self.result_text.setText(result_text)
        self.status_label.setText("Анализ группы из 5 студентов завершен")
        self.status_label.setStyleSheet("color: #4CAF50; padding: 10px;")
    
    def format_result_text(self, result):
        text = f"{result['student_name']}\n"
        text += "="*len(result['student_name']) + "\n\n"
        
        text += f"Сегмент: {result['segment_name']}\n"
        text += f"Латентный показатель: {result['latent_value']:.3f}\n"
        text += f"Уровень риска: {result['risk_level']}\n"
        text += f"Прогнозируемая лояльность: {result['predicted_loyalty']}%\n"
        text += f"{result['description']}\n\n"
        
        text += f"Общий профиль: {result['profile']}\n"
        text += f"Средний балл: {result['avg_score']:.2f}\n"
        text += f"Сильных сторон: {result['strong_features']}\n"
        text += f"Слабых сторон: {result['weak_features']}\n\n"
        
        text += "Рекомендации для повышения лояльности:\n"
        for action in result['recommended_actions']:
            text += f"{action}\n"
        text += "\n"
        
        # Анализ признаков
        analysis = result['feature_analysis']
        
        if analysis['weaknesses']:
            text += "Слабые стороны:\n"
            for weakness in analysis['weaknesses']:
                text += f" - {weakness}\n"
            text += "\n"
        
        if analysis['strengths']:
            text += "Сильные стороны:\n"
            for strength in analysis['strengths']:
                text += f" - {strength}\n"
            text += "\n"
        
        if analysis['neutral']:
            text += "Нормальные показатели:\n"
            for neutral in analysis['neutral']:
                text += f" - {neutral}\n"
            text += "\n"
        
        if analysis['recommendations']:
            text += "Дополнительные рекомендации:\n"
            for rec in analysis['recommendations']:
                text += f" - {rec}\n"
        
        return text
    
    def calculate_group_statistics(self, results):
        if not results:
            return "Нет данных для статистики"
        
        segments = {}
        total_loyalty = 0
        total_students = len(results)
        
        for result in results:
            segment = result['segment_name']
            if segment not in segments:
                segments[segment] = 0
            segments[segment] += 1
            total_loyalty += result['predicted_loyalty']
        
        avg_loyalty = total_loyalty / total_students
        
        stats = f"Всего студентов: {total_students}\n"
        stats += f"Средняя лояльность: {avg_loyalty:.1f}%\n\n"
        stats += "Распределение по сегментам:\n"
        
        for segment, count in segments.items():
            percentage = (count / total_students) * 100
            stats += f"  {segment}: {count} студентов ({percentage:.1f}%)\n"
        
        # Определение общего риска группы
        if "Проблемные студенты" in segments:
            overall_risk = "Высокий"
        elif "Рискованные студенты" in segments:
            overall_risk = "Средний"
        else:
            overall_risk = "Низкий"
        
        stats += f"\nОбщий уровень риска группы: {overall_risk}\n"
        
        return stats
    
    def display_result(self, result):
        """Отображение результата в текстовом поле"""
        result_text = self.format_result_text(result)
        self.result_text.setText(result_text)
    
    def save_results(self):
        """Сохранение результатов в файл"""
        if not self.results:
            QMessageBox.warning(self, "Внимание", "Нет результатов для сохранения")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "result.txt", "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    if len(self.results) == 1:
                        # Сохраняем одного студента
                        result = self.results[0]
                        f.write(self.format_result_text(result))
                    else:
                        # Сохраняем группу
                        f.write("" + "\n")
                        f.write("Анализ лояльности студентов\n")
                        f.write("" + "\n\n")
                        
                        for result in self.results:
                            f.write(self.format_result_text(result))
                            f.write("\n" + "-"*60 + "\n\n")
                        
                        # Статистика
                        stats = self.calculate_group_statistics(self.results)
                        f.write("\n" + "="*60 + "\n")
                        f.write("Статистика\n")
                        f.write("="*60 + "\n\n")
                        f.write(stats)
                
                self.status_label.setText(f"Результаты сохранены в {filename}")
                self.status_label.setStyleSheet("color: #4CAF50; padding: 10px;")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = StudentAnalyzerApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()