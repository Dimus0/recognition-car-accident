import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
)


class AccidentClassifier:
    """
    ResNet50-based класифікатор ДТП / не ДТП.

    Приклад інтеграції в систему:
    ─────────────────────────────
    clf = AccidentClassifier.load('best_accident_classifier.pth')

    # Зображення з диску:
    label, probs = clf.predict('/path/to/frame.jpg')

    # Кадр з відео (np.ndarray BGR від OpenCV):
    label, probs = clf.predict(cv2_frame)

    # Отримати впевненість:
    is_accident = label == 'Accident'
    confidence  = probs[clf.class_names.index('Accident')]
    """

    # ── ImageNet нормалізація ──────────────────────────────────────────
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    _INFERENCE_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, model: nn.Module, class_names: list,
                 device: str = 'cpu'):
        self.model       = model
        self.class_names = class_names
        self.device      = device
        self.model.to(device)

    # ── Конструктори ──────────────────────────────────────────────────
    @classmethod
    def build(cls, class_names: list,
              dropout: float = 0.4,
              device: str = 'cpu') -> 'AccidentClassifier':
        """
        Створює модель з pretrained ImageNet вагами.
        Всі шари одразу розморожені для повного навчання.
        """
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Розморожуємо ВСІ шари — навчання без фаз
        for param in backbone.parameters():
            param.requires_grad = True

        # Замінюємо FC-голову
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(128, len(class_names)),
        )

        total     = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f'Параметрів: {total:,}  (trainable: {trainable:,})')

        return cls(backbone, class_names, device)

    @classmethod
    def load(cls, path: str,
             device: str = None) -> 'AccidentClassifier':
        """
        Завантажує збережену модель з .pth файлу.
        Використовується для інтеграції в систему.

        clf = AccidentClassifier.load('best.pth')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint   = torch.load(path, map_location=device)
        class_names  = checkpoint['class_names']
        dropout      = checkpoint.get('dropout', 0.4)

        # Будуємо архітектуру (без завантаження ImageNet ваг)
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(128, len(class_names)),
        )
        backbone.load_state_dict(checkpoint['model_state_dict'])

        obj = cls(backbone, class_names, device)
        obj.model.eval()
        print(f'✅ Модель завантажена: {path}')
        print(f'   Класи  : {class_names}')
        print(f'   Device : {device}')
        if 'best_val_acc' in checkpoint:
            print(f'   Val Acc: {checkpoint["best_val_acc"]:.4f}')
        return obj

    # ── Збереження ────────────────────────────────────────────────────
    def save(self, path: str, best_val_acc: float = 0.0,
             dropout: float = 0.4):
        """
        Зберігає ваги + метадані (class_names, dropout, val_acc).
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names'     : self.class_names,
            'dropout'         : dropout,
            'best_val_acc'    : best_val_acc,
        }, path)

    # ── Навчання ──────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 50,
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
        patience:     int   = 8,
        min_delta:    float = 0.001,
        save_path:    str   = None,
        plot_dir:     str   = None,
        class_weights: torch.Tensor = None,
    ) -> dict:
        """
        Єдина фаза навчання — всі шари відразу.

        Повертає словник з історією:
          {'train_loss': [...], 'val_loss': [...],
           'train_acc':  [...], 'val_acc':  [...]}
        """
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Диференційовані LR: backbone з меншим, голова — з повним
        fc_params       = list(self.model.fc.parameters())
        fc_ids          = {id(p) for p in fc_params}
        backbone_params = [p for p in self.model.parameters()
                           if id(p) not in fc_ids]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},   # backbone × 0.1
            {'params': fc_params,       'lr': lr},          # FC голова
        ], weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        history = {'train_loss': [], 'val_loss': [],
                   'train_acc' : [], 'val_acc' : []}

        best_val_loss = float('inf')
        best_val_acc  = 0.0
        patience_cnt  = 0

        print(f"\n{'='*65}")
        print(f"  Навчання ResNet50 — {epochs} епох | device={self.device}")
        print(f"  LR backbone={lr*0.1:.2e}  LR head={lr:.2e}")
        print(f"{'='*65}")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, criterion, optimizer)
            vl_loss, vl_acc = self._run_epoch(val_loader,   criterion)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(vl_acc)

            saved = ''
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                if save_path:
                    self.save(save_path, best_val_acc)
                saved = '  ✅'

            print(f'Epoch {epoch:3d}/{epochs} | '
                  f'Train {tr_loss:.4f}/{tr_acc:.4f} | '
                  f'Val {vl_loss:.4f}/{vl_acc:.4f}{saved}')

            # Early stopping
            if vl_loss < best_val_loss - min_delta:
                best_val_loss = vl_loss
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f'\n⛔ Early stopping — епоха {epoch}')
                    break

        print(f'\n🏆 Найкраща Val Acc: {best_val_acc:.4f}')

        if plot_dir:
            self._plot_history(history, plot_dir)

        return history

    # ── Оцінка ────────────────────────────────────────────────────────
    def evaluate(
        self,
        loader:    DataLoader,
        weights_path: str = None,
        plot_dir:  str   = None,
    ) -> dict:
        """
        Повна оцінка: accuracy, balanced_accuracy, classification_report,
        confusion matrix, аналіз критичних помилок.

        Якщо weights_path вказано — завантажує найкращі ваги перед оцінкою.
        """
        if weights_path:
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f'Завантажено ваги: {weights_path}')

        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for images, labels in loader:
                outputs = self.model(images.to(self.device))
                probs   = torch.softmax(outputs, dim=1)
                preds   = outputs.argmax(dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        acc     = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        print(f"\n{'='*50}")
        print(f'  Accuracy          : {acc:.4f}')
        print(f'  Balanced Accuracy : {bal_acc:.4f}')
        print(f"{'='*50}")

        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        print(report_df.to_string())

        if plot_dir:
            report_df.to_csv(os.path.join(plot_dir, 'metrics_report.csv'))
            self._plot_confusion_matrix(y_true, y_pred, plot_dir)

        # Критичні помилки
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        acc_idx    = 0  # Accident — завжди перший клас

        fn = int(np.sum((y_true_arr == acc_idx) & (y_pred_arr != acc_idx)))
        fp = int(np.sum((y_true_arr != acc_idx) & (y_pred_arr == acc_idx)))
        print(f'\n--- АНАЛІЗ БЕЗПЕКИ ---')
        print(f'❗ Критичні пропуски (ДТП → Чисто) : {fn}')
        print(f'⚠️  Хибні тривоги   (Чисто → ДТП) : {fp}')

        return {
            'accuracy': acc, 'balanced_accuracy': bal_acc,
            'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
            'false_negatives': fn, 'false_positives': fp,
        }

    # ── Інференс ──────────────────────────────────────────────────────
    def predict(
        self,
        source,                     # str | Path | PIL.Image | np.ndarray
        threshold: float = 0.5,     # поріг для класу accident (index=0)
    ) -> tuple:
        """
        Передбачення для одного зображення.

        source може бути:
          - str або Path  → шлях до файлу
          - PIL.Image
          - np.ndarray    → BGR кадр від OpenCV

        Повертає:
          (label: str, probs: np.ndarray)

        Приклад:
          label, probs = clf.predict(cv2_frame)
          is_accident  = label == 'Accident'
          confidence   = probs[0]  # якщо Accident — перший клас
        """
        img = self._load_image(source)
        inp = self._INFERENCE_TRANSFORM(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(inp), dim=1)[0].cpu().numpy()

        # Якщо є клас Accident — використовуємо поріг
        acc_names = [n for n in self.class_names
                     if 'accident' in n.lower() and 'non' not in n.lower()]
        if acc_names:
            acc_idx      = self.class_names.index(acc_names[0])
            label = acc_names[0] if probs[acc_idx] >= threshold \
                    else self.class_names[1 - acc_idx]
        else:
            label = self.class_names[int(np.argmax(probs))]

        return label, probs

    def predict_batch(
        self,
        sources: list,
        threshold: float = 0.5,
    ) -> list:
        """
        Батч інференс.
        sources: список str/Path/PIL/np.ndarray
        Повертає: [(label, probs), ...]
        """
        imgs = [self._INFERENCE_TRANSFORM(self._load_image(s))
                for s in sources]
        batch = torch.stack(imgs).to(self.device)

        self.model.eval()
        with torch.no_grad():
            all_probs = torch.softmax(self.model(batch), dim=1).cpu().numpy()

        results = []
        for probs in all_probs:
            label = self.class_names[int(np.argmax(probs))]
            results.append((label, probs))
        return results

    # ── Grad-CAM ──────────────────────────────────────────────────────
    def gradcam(
        self,
        source,
        class_idx: int = None,
    ) -> np.ndarray:
        """
        Генерує Grad-CAM теплову карту для зображення.
        Повертає overlay (H×W×3 float32 в [0,1]).
        """
        img     = self._load_image(source)
        img_np  = np.array(img.resize((224, 224))) / 255.0
        inp     = self._INFERENCE_TRANSFORM(img).unsqueeze(0).to(self.device)

        gradients  = []
        activations = []

        def fwd_hook(_, __, out): activations.append(out.detach())
        def bwd_hook(_, __, g):   gradients.append(g[0].detach())

        handle_f = self.model.layer4[-1].register_forward_hook(fwd_hook)
        handle_b = self.model.layer4[-1].register_full_backward_hook(bwd_hook)

        self.model.eval()
        output = self.model(inp)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        handle_f.remove(); handle_b.remove()

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * activations[0]).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.resize(cam, (224, 224))
        overlay = 0.5 * img_np + 0.5 * plt.cm.jet(heatmap)[..., :3]
        return np.clip(overlay, 0, 1)

    # ── Приватні методи ───────────────────────────────────────────────
    def _run_epoch(
        self,
        loader:    DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer = None,
    ) -> tuple:
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if is_train else torch.no_grad()

        with ctx:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss    = criterion(outputs, labels)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                correct    += outputs.argmax(1).eq(labels).sum().item()
                total      += labels.size(0)

        return total_loss / len(loader), correct / total

    def _load_image(self, source) -> Image.Image:
        """Уніфіковане завантаження: str/Path/PIL/np.ndarray → PIL RGB."""
        if isinstance(source, (str, Path)):
            return Image.open(source).convert('RGB')
        if isinstance(source, np.ndarray):
            # OpenCV BGR → RGB
            return Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        if isinstance(source, Image.Image):
            return source.convert('RGB')
        raise TypeError(f'Непідтримуваний тип: {type(source)}')

    def _plot_history(self, history: dict, save_dir: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric, title in zip(
            axes,
            [('train_loss','val_loss'), ('train_acc','val_acc')],
            ['Loss', 'Accuracy']
        ):
            ax.plot(history[metric[0]], label='Train')
            ax.plot(history[metric[1]], label='Val')
            ax.set_title(title); ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.show(); plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, save_dir: str):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix'); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.show(); plt.close()


print('✅ Клас AccidentClassifier визначено')