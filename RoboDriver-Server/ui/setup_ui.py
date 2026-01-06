import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
                             QGroupBox, QMessageBox, QTimeEdit)
from PyQt5.QtCore import Qt, QTime
from ruamel.yaml import YAML  # 替换 pyyaml，保留原格式

class YamlConfigEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_path = None  # 配置文件路径
        self.yaml = YAML()  # 初始化 ruamel.yaml 对象（保留格式关键）
        self.yaml.preserve_quotes = True  # 保留字符串引号
        self.yaml.indent(mapping=2, sequence=4, offset=2)  # 保持原有缩进风格
        self.yaml_data = None  # 加载的 YAML 数据
        # 自定义字段中文名（可根据需求修改）
        self.field_names = {
            "is_upload": "是否启用上传",
            "upload_immadiately_gpu": "GPU 即时上传",
            "is_update_machine_information": "自动更新设备信息",
            "is_collect_upload_at_sametime": "采集与上传同步执行",
            "upload_time": "批量上传时间"
        }
        self.init_ui()
        # 初始化时自动填充默认架构（x86）的路径
        self.on_arch_change(self.arch_combo.currentText())

    def init_ui(self):
        self.setWindowTitle("配置文件编辑器")
        self.setGeometry(100, 100, 800, 600)

        # 中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # 1. 架构选择区域
        arch_group = QGroupBox("架构选择")
        arch_layout = QHBoxLayout(arch_group)
        arch_layout.setSpacing(20)

        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["x86", "ARM"])
        self.arch_combo.currentTextChanged.connect(self.on_arch_change)
        arch_layout.addWidget(QLabel("架构类型："))
        arch_layout.addWidget(self.arch_combo)
        arch_layout.addStretch(1)
        main_layout.addWidget(arch_group)

        # 2. 配置文件路径区域（仅展示路径，不可编辑）
        path_group = QGroupBox("配置文件定位（自动匹配架构）")
        path_layout = QHBoxLayout(path_group)
        path_layout.setSpacing(15)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("配置文件路径（.yaml）")
        self.path_edit.setReadOnly(True)  # 路径不可手动编辑
        self.path_edit.setStyleSheet("background-color: #f0f0f0;")  # 灰色背景提示不可编辑
        path_layout.addWidget(self.path_edit)

        self.load_btn = QPushButton("加载配置")
        self.load_btn.clicked.connect(self.load_config)
        path_layout.addWidget(self.load_btn)
        main_layout.addWidget(path_group)

        # 3. 可配置字段区域（初始禁用，加载配置后启用）
        self.config_group = QGroupBox("核心配置项")
        config_layout = QVBoxLayout(self.config_group)
        config_layout.setSpacing(15)
        self.config_group.setEnabled(False)

        # 布尔型字段（CheckBox）
        self.checkbox_fields = {}
        for field in ["is_upload", "upload_immadiately_gpu", "is_update_machine_information", "is_collect_upload_at_sametime"]:
            h_layout = QHBoxLayout()
            checkbox = QCheckBox(self.field_names[field])
            self.checkbox_fields[field] = checkbox
            h_layout.addWidget(checkbox)
            h_layout.addStretch(1)
            config_layout.addLayout(h_layout)

        # 时间字段（TimeEdit）
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel(self.field_names["upload_time"] + "："))
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm")
        time_layout.addWidget(self.time_edit)
        time_layout.addStretch(1)
        config_layout.addLayout(time_layout)

        main_layout.addWidget(self.config_group)

        # 4. 操作按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_config)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch(1)
        main_layout.addLayout(btn_layout)

    def on_arch_change(self, arch):
        """架构切换时，自动填充对应路径（无需用户选择）"""
        default_paths = {
            "x86": "../x86/setup.yaml",  # x86 架构默认路径（可修改）
            "ARM": "../arm/setup.yaml"     # ARM 架构默认路径（可修改）
        }
        self.config_path = default_paths.get(arch, "")  # 自动赋值路径
        self.path_edit.setText(self.config_path)  # 展示路径给用户

    def load_config(self):
        """加载自动匹配路径的 YAML 配置文件（保留原格式）"""
        if not self.config_path:
            QMessageBox.warning(self, "警告", "未获取到对应架构的配置文件路径！")
            return

        try:
            # 用 ruamel.yaml 读取，保留原格式（空行、注释、缩进）
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.yaml_data = self.yaml.load(f)

            # 验证配置文件结构（确保包含所有需要的字段）
            required_fields = ["is_upload", "upload_immadiately_gpu", "is_update_machine_information",
                              "is_collect_upload_at_sametime", "upload_time"]
            missing_fields = [f for f in required_fields if f not in self.yaml_data]
            if missing_fields:
                QMessageBox.warning(self, "警告", f"配置文件缺少必要字段：{', '.join(missing_fields)}")
                return

            # 填充 UI 控件
            for field, checkbox in self.checkbox_fields.items():
                checkbox.setChecked(self.yaml_data[field])

            # 解析时间（HH:mm 格式）
            upload_time = self.yaml_data["upload_time"].strip()
            try:
                hour, minute = map(int, upload_time.split(":"))
                self.time_edit.setTime(QTime(hour, minute))
            except:
                QMessageBox.warning(self, "警告", "上传时间格式错误，默认设置为 20:00")
                self.time_edit.setTime(QTime(20, 0))

            # 启用配置编辑和保存按钮
            self.config_group.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            QMessageBox.information(self, "成功", f"已加载 {self.arch_combo.currentText()} 架构配置文件！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置失败：{str(e)}")

    def save_config(self):
        """保存修改后的配置（保留原文件格式，仅更新目标字段）"""
        if not self.yaml_data or not self.config_path:
            QMessageBox.warning(self, "警告", "请先加载配置文件！")
            return

        try:
            # 更新 YAML 数据中的目标字段（不改动其他字段和格式）
            for field, checkbox in self.checkbox_fields.items():
                self.yaml_data[field] = checkbox.isChecked()

            # 获取时间并格式化（保持原字符串格式）
            upload_time = self.time_edit.time().toString("HH:mm")
            self.yaml_data["upload_time"] = upload_time

            # 用 ruamel.yaml 写入，保留原格式（空行、注释、缩进都不变）
            with open(self.config_path, "w", encoding="utf-8") as f:
                self.yaml.dump(self.yaml_data, f)

            QMessageBox.information(self, "成功", "配置文件保存成功！原格式已保留～")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败：{str(e)}")

    def reset_config(self):
        """重置 UI 到加载时的配置状态"""
        if not self.yaml_data:
            return

        for field, checkbox in self.checkbox_fields.items():
            checkbox.setChecked(self.yaml_data[field])

        upload_time = self.yaml_data["upload_time"].strip()
        try:
            hour, minute = map(int, upload_time.split(":"))
            self.time_edit.setTime(QTime(hour, minute))
        except:
            self.time_edit.setTime(QTime(20, 0))

        QMessageBox.information(self, "成功", "配置已重置为加载时状态！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YamlConfigEditor()
    window.show()
    sys.exit(app.exec_())