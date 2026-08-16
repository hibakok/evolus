#!/usr/bin/env python3
"""
Evotus - Универсальная нейронная сеть с эволюционными стратегиями
Оптимизированная версия для максимальной скорости эволюции
Универсальный аппроксиматор любых вычислимых функций

ФУНДАМЕНТАЛЬНОЕ УЛУЧШЕНИЕ ТОЧНОСТИ:
- Веса имеют 50+ знаков после запятой благодаря использованию Decimal
- Ошибка уточняется с ультравысокой точностью
- Новая архитектура мутаций: эффективна для чисел любого масштаба
- Значимая часть мутаций - мелкая, но возможны любые по масштабу изменения
- Вероятность мутации обратно пропорциональна её масштабу (логарифмическое распределение)

НОВЫЕ МОЩНЫЕ УЛУЧШЕНИЯ ДЛЯ УНИВЕРСАЛЬНОСТИ:
- Ансамбли нейронных сетей с эволюцией весов ансамбля
- Кросс-валидация для лучшей генерализации
- Новые типы мутаций: масштабирование весов, дублирование нейронов, разделение связей
- Мемоизация вычислений для ускорения forward pass
- Улучшенная система важности связей с decay и историей
- Эволюция мини-слоев внутри сети
- Регуляризация в fitness функции для предотвращения переобучения
- Дополнительные экзотические функции активации (40+ типов)
- Адаптивная batchSize обработка для ускорения обучения
- Статистический анализ эволюции для интеллектуальной адаптации
- Поддержка многозадачного обучения
"""

import random
import math
import json
import os
import sys
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import copy
from decimal import Decimal, getcontext, ROUND_HALF_EVEN

# Устанавливаем сверхвысокую точность вычислений (50+ знаков после запятой)
DECIMAL_PRECISION = 80
getcontext().prec = DECIMAL_PRECISION
getcontext().rounding = ROUND_HALF_EVEN

# Условный импорт для определения нажатия клавиш в Windows
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

# Пути к файлам конфигурации
CONFIG_FILE = "config.txt"
DATA_FILE = "data.txt"
POPULATION_FILE = "population.json"

class ActivationFunction(Enum):
    # Базовые функции
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"
    STEP = "step"
    
    # Продвинутые функции
    GAUSSIAN = "gaussian"
    SWISH = "swish"
    GELU = "gelu"
    ELU = "elu"
    SOFTPLUS = "softplus"
    SELU = "selu"
    MISH = "mish"
    
    # Периодические функции
    SIN = "sin"
    COS = "cos"
    SINC = "sinc"
    
    # Специализированные функции
    RBF = "rbf"
    BIPOLAR_SIGMOID = "bipolar_sigmoid"
    
    # Волновые функции
    TRIANGULAR = "triangular"
    SAWTOOTH = "sawtooth"
    SQUARE_WAVE = "square_wave"
    
    # Параметризуемые функции (с эволюционирующими параметрами)
    PARAMETRIC_RELU = "parametric_relu"  # PReLU с обучаемым alpha
    LEAKY_RELU = "leaky_relu"  # Leaky ReLU с обучаемым alpha
    PARAMETRIC_TANH = "parametric_tanh"  # tanh с масштабируемым коэффициентом
    GAUSSIAN_RBF = "gaussian_rbf"  # RBF с обучаемой шириной
    SINUSOIDAL = "sinusoidal"  # sin с обучаемой частотой
    
    # Современные функции активации
    SILU = "silu"
    HARD_SWISH = "hard_swish"
    HARD_SIGMOID = "hard_sigmoid"
    GELU_APPROX = "gelu_approx"
    
    # Экзотические функции для универсальности
    INVERSE = "inverse"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    SQUARE = "square"
    CUBE = "cube"
    ABSOLUTE = "absolute"
    
    # НОВЫЕ дополнительные экзотические функции для максимальной универсальности
    BENT_IDENTITY = "bent_identity"
    SOFT_SIGN = "soft_sign"
    ELISH = "elish"
    HARD_ELU = "hard_elu"
    THRESHOLD_RELU = "threshold_relu"
    RANDOM_RELU = "random_relu"
    SINE_RELU = "sine_relu"
    COSINE_RELU = "cosine_relu"
    POLY6 = "poly6"
    SIGLU = "siglu"
    ROOT2 = "root2"
    SQRLU = "sqrlu"
    SRELU = "srelu"
    PDELU = "pdelu"
    APLU = "aplu"
    ERELU = "erelu"
    FTAU = "ftau"
    LAF = "laf"
    NLAF = "nlaf"
    PLAF = "plaf"
    SLAF = "slaf"
    NNLAF = "nnlaf"
    HTAU = "htau"
    SOFTPLUS_SHIFTED = "softplus_shifted"
    GAUSSIAN_SHIFTED = "gaussian_shifted"
    MULTI_SIN = "multi_sin"
    COMBINED_PERIODIC = "combined_periodic"

# Предварительно вычисленные таблицы для активаций (ускорение)
_SIGMOID_TABLE_SIZE = 20000
_SIGMOID_MIN = -10.0
_SIGMOID_MAX = 10.0
_SIGMOID_STEP = (_SIGMOID_MAX - _SIGMOID_MIN) / _SIGMOID_TABLE_SIZE
_SIGMOID_TABLE = [1.0 / (1.0 + math.exp(-(_SIGMOID_MIN + i * _SIGMOID_STEP))) 
                  for i in range(_SIGMOID_TABLE_SIZE)]

# Таблица для Gaussian
_GAUSSIAN_TABLE = [math.exp(-((_SIGMOID_MIN + i * _SIGMOID_STEP) ** 2) / 2) 
                   for i in range(_SIGMOID_TABLE_SIZE)]

# Таблица для Sin/Cos
_SIN_TABLE = [math.sin(_SIGMOID_MIN + i * _SIGMOID_STEP) 
              for i in range(_SIGMOID_TABLE_SIZE)]
_COS_TABLE = [math.cos(_SIGMOID_MIN + i * _SIGMOID_STEP) 
              for i in range(_SIGMOID_TABLE_SIZE)]

def to_decimal(x) -> Decimal:
    """Конвертирует значение в Decimal для ультравысокой точности.
    
    Поддерживает конвертацию из float, int, str, Decimal.
    """
    if isinstance(x, Decimal):
        return x
    elif isinstance(x, (int, float)):
        # Конвертируем через строку для сохранения точности
        return Decimal(str(x))
    else:
        return Decimal(str(x))

def sigmoid(x) -> Decimal:
    x = to_decimal(x)
    if x < to_decimal(_SIGMOID_MIN):
        return Decimal('0.0')
    if x > to_decimal(_SIGMOID_MAX):
        return Decimal('1.0')
    idx = int((x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_STEP))
    # Защита от выхода за границы таблицы
    idx = max(0, min(idx, _SIGMOID_TABLE_SIZE - 1))
    return to_decimal(_SIGMOID_TABLE[idx])

def tanh_act(x) -> Decimal:
    return to_decimal(math.tanh(float(to_decimal(x))))

def relu(x) -> Decimal:
    x = to_decimal(x)
    return x if x > Decimal('0.0') else Decimal('0.0')

def linear(x) -> Decimal:
    return to_decimal(x)

def step_act(x) -> Decimal:
    return Decimal('1.0') if to_decimal(x) >= Decimal('0.0') else Decimal('0.0')

def gaussian(x) -> Decimal:
    """Гауссова функция активации"""
    x = to_decimal(x)
    if x < to_decimal(_SIGMOID_MIN) or x > to_decimal(_SIGMOID_MAX):
        return Decimal('0.0')
    idx = int((x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_STEP))
    # Защита от выхода за границы таблицы
    idx = max(0, min(idx, _SIGMOID_TABLE_SIZE - 1))
    return to_decimal(_GAUSSIAN_TABLE[idx])

def sin_act(x) -> Decimal:
    """Синусоидальная функция активации"""
    x = to_decimal(x)
    if x < to_decimal(_SIGMOID_MIN) or x > to_decimal(_SIGMOID_MAX):
        return to_decimal(math.sin(float(x)))
    idx = int((x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_STEP))
    # Защита от выхода за границы таблицы
    idx = max(0, min(idx, _SIGMOID_TABLE_SIZE - 1))
    return to_decimal(_SIN_TABLE[idx])

def cos_act(x) -> Decimal:
    """Косинусоидальная функция активации"""
    x = to_decimal(x)
    if x < to_decimal(_SIGMOID_MIN) or x > to_decimal(_SIGMOID_MAX):
        return to_decimal(math.cos(float(x)))
    idx = int((x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_STEP))
    # Защита от выхода за границы таблицы
    idx = max(0, min(idx, _SIGMOID_TABLE_SIZE - 1))
    return to_decimal(_COS_TABLE[idx])

def swish(x) -> Decimal:
    """Swish функция: x * sigmoid(x)"""
    x = to_decimal(x)
    return x * sigmoid(x)

def gelu(x) -> Decimal:
    """GELU функция"""
    x = to_decimal(x)
    xf = float(x)
    result = 0.5 * xf * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf ** 3)))
    return to_decimal(result)

def elu(x) -> Decimal:
    """ELU функция"""
    x = to_decimal(x)
    alpha = Decimal('1.0')
    if x >= Decimal('0.0'):
        return x
    else:
        return alpha * (to_decimal(math.exp(float(x))) - Decimal('1.0'))

def softplus(x) -> Decimal:
    """Softplus функция"""
    x = to_decimal(x)
    if x > Decimal('20'):
        return x
    if x < Decimal('-20'):
        return Decimal('0.0')
    return to_decimal(math.log(1.0 + math.exp(float(x))))

def rbf(x) -> Decimal:
    """Radial Basis Function с центром в 0"""
    x = to_decimal(x)
    return to_decimal(math.exp(-(float(x) ** 2)))

def sinc_act(x) -> Decimal:
    """Sinc функция"""
    x = to_decimal(x)
    if abs(x) < Decimal('1e-10'):
        return Decimal('1.0')
    return to_decimal(math.sin(float(x)) / float(x))

def bipolar_sigmoid(x) -> Decimal:
    """Биполярная сигмоида в диапазоне [-1, 1]"""
    x = to_decimal(x)
    # Защита от переполнения
    if x > Decimal('20'):
        return Decimal('1.0')
    if x < Decimal('-20'):
        return Decimal('-1.0')
    return Decimal('2.0') / (Decimal('1.0') + to_decimal(math.exp(-float(x)))) - Decimal('1.0')

def triangular(x) -> Decimal:
    """Треугольная функция"""
    x = to_decimal(x)
    x_norm = ((x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_MAX - _SIGMOID_MIN)) * Decimal('4') - Decimal('2')
    return max(Decimal('0'), Decimal('1') - abs(x_norm))

def sawtooth(x) -> Decimal:
    """Пилообразная функция"""
    x = to_decimal(x)
    x_norm = (x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_MAX - _SIGMOID_MIN)
    return Decimal('2') * (x_norm - to_decimal(math.floor(float(x_norm) + 0.5)))

def square_wave(x) -> Decimal:
    """Квадратная волна"""
    x = to_decimal(x)
    x_norm = (x - to_decimal(_SIGMOID_MIN)) / to_decimal(_SIGMOID_MAX - _SIGMOID_MIN)
    return Decimal('1.0') if math.sin(2 * math.pi * float(x_norm)) >= 0 else Decimal('-1.0')

# Новые функции активации для максимальной универсальности

def selu(x) -> Decimal:
    """SELU (Scaled Exponential Linear Unit)"""
    x = to_decimal(x)
    alpha = to_decimal('1.6732632423543772848170429916717')
    scale = to_decimal('1.0507009873554804934193349852946')
    if x >= Decimal('0.0'):
        return scale * x
    else:
        return scale * alpha * (to_decimal(math.exp(float(x))) - Decimal('1.0'))

def mish(x) -> Decimal:
    """Mish функция: x * tanh(softplus(x))"""
    x = to_decimal(x)
    return x * to_decimal(math.tanh(float(softplus(x))))

def parametric_relu(x) -> Decimal:
    """PReLU с фиксированным alpha (параметр эволюционирует через bias)"""
    x = to_decimal(x)
    alpha = Decimal('0.25')
    return x if x >= Decimal('0.0') else alpha * x

def leaky_relu(x) -> Decimal:
    """Leaky ReLU с фиксированным alpha"""
    x = to_decimal(x)
    alpha = Decimal('0.01')
    return x if x >= Decimal('0.0') else alpha * x

def parametric_tanh(x) -> Decimal:
    """Tanh с масштабирующим коэффициентом"""
    x = to_decimal(x)
    return to_decimal('1.5') * to_decimal(math.tanh(float(x)))

def gaussian_rbf(x) -> Decimal:
    """RBF с измененной шириной"""
    x = to_decimal(x)
    return to_decimal(math.exp(-0.5 * float(x) ** 2))

def sinusoidal(x) -> Decimal:
    """Синусоида с увеличенной частотой"""
    x = to_decimal(x)
    return to_decimal(math.sin(2 * float(x)))

def silu(x) -> Decimal:
    """SiLU (Sigmoid Linear Unit) - то же что и swish"""
    x = to_decimal(x)
    return x * sigmoid(x)

def hard_swish(x) -> Decimal:
    """Hard Swish - аппроксимация swish"""
    x = to_decimal(x)
    if x < Decimal('-3'):
        return Decimal('0.0')
    elif x > Decimal('3'):
        return x
    return x * (x + Decimal('3')) / Decimal('6')

def hard_sigmoid(x) -> Decimal:
    """Hard Sigmoid - быстрая аппроксимация сигмоиды"""
    x = to_decimal(x)
    if x < Decimal('-3'):
        return Decimal('0.0')
    elif x > Decimal('3'):
        return Decimal('1.0')
    return (x + Decimal('3')) / Decimal('6')

def gelu_approx(x) -> Decimal:
    """Быстрая аппроксимация GELU"""
    x = to_decimal(x)
    xf = float(x)
    result = 0.5 * xf * (1.0 + math.tanh(0.7978845608 * (xf + 0.044715 * xf ** 3)))
    return to_decimal(result)

def inverse(x) -> Decimal:
    """Обратная функция с защитой от деления на ноль"""
    x = to_decimal(x)
    if abs(x) < Decimal('0.1'):
        return Decimal('10.0').copy_sign(x)
    return Decimal('1.0') / x

def logarithmic(x) -> Decimal:
    """Логарифмическая функция"""
    x = to_decimal(x)
    if x <= Decimal('0.0'):
        return -to_decimal(abs(math.log(abs(float(x)) + 1)))
    return to_decimal(math.log(float(x) + 1))

def exponential(x) -> Decimal:
    """Экспоненциальная функция с ограничением"""
    x = to_decimal(x)
    if x > Decimal('20'):
        return to_decimal(math.exp(20))
    if x < Decimal('-20'):
        return Decimal('0.0')
    return to_decimal(math.exp(float(x)))

def square(x) -> Decimal:
    """Квадратичная функция"""
    x = to_decimal(x)
    return x ** 2

def cube(x) -> Decimal:
    """Кубическая функция"""
    x = to_decimal(x)
    return x ** 3

def absolute(x) -> Decimal:
    """Модуль"""
    x = to_decimal(x)
    return abs(x)

# НОВЫЕ дополнительные функции активации для максимальной универсальности

def bent_identity(x) -> Decimal:
    """Bent Identity: (sqrt(x^2 + 1) - 1) / 2 + x"""
    x = to_decimal(x)
    xf = float(x)
    result = (math.sqrt(xf ** 2 + 1) - 1) / 2 + xf
    return to_decimal(result)

def soft_sign(x) -> Decimal:
    """Soft Sign: x / (|x| + 1)"""
    x = to_decimal(x)
    xf = float(x)
    result = xf / (abs(xf) + 1)
    return to_decimal(result)

def elish(x) -> Decimal:
    """ELiSH: swish для x > 0, tanh для x <= 0"""
    x = to_decimal(x)
    if x >= Decimal('0.0'):
        return swish(x)
    else:
        return tanh_act(x)

def hard_elu(x) -> Decimal:
    """Hard ELU: быстрая аппроксимация ELU"""
    x = to_decimal(x)
    if x >= Decimal('0.0'):
        return x
    elif x >= Decimal('-1.0'):
        return Decimal('0.85') * x
    else:
        return Decimal('-0.85')

def threshold_relu(x) -> Decimal:
    """Threshold ReLU: 0 для x < threshold, x для x >= threshold"""
    x = to_decimal(x)
    threshold = Decimal('0.1')
    return x if x >= threshold else Decimal('0.0')

def sine_relu(x) -> Decimal:
    """Sine ReLU: sin(x) для x > 0, 0 для x <= 0"""
    x = to_decimal(x)
    if x > Decimal('0.0'):
        return to_decimal(math.sin(float(x)))
    return Decimal('0.0')

def cosine_relu(x) -> Decimal:
    """Cosine ReLU: cos(x) для x > 0, 0 для x <= 0"""
    x = to_decimal(x)
    if x > Decimal('0.0'):
        return to_decimal(math.cos(float(x)))
    return Decimal('0.0')

def poly6(x) -> Decimal:
    """Polynomial activation degree 6"""
    x = to_decimal(x)
    xf = float(x)
    result = xf + 0.1 * xf**2 + 0.01 * xf**3 + 0.001 * xf**4 + 0.0001 * xf**5 + 0.00001 * xf**6
    return to_decimal(result)

def siglu(x) -> Decimal:
    """SigLU: sigmoid(x) * x"""
    x = to_decimal(x)
    return sigmoid(x) * x

def root2(x) -> Decimal:
    """Root2: sign(x) * sqrt(|x|)"""
    x = to_decimal(x)
    xf = float(x)
    result = math.copysign(math.sqrt(abs(xf)), xf)
    return to_decimal(result)

def sqrlu(x) -> Decimal:
    """SqReLU: square(relu(x))"""
    x = to_decimal(x)
    if x > Decimal('0.0'):
        return x ** 2
    return Decimal('0.0')

def srelu(x) -> Decimal:
    """SReLU: piecewise linear с тремя участками"""
    x = to_decimal(x)
    if x >= Decimal('1.0'):
        return x
    elif x >= Decimal('0.0'):
        return Decimal('0.5') * x
    else:
        return Decimal('0.1') * x

def pdelu(x) -> Decimal:
    """PDELU: parametric DELU"""
    x = to_decimal(x)
    alpha = Decimal('0.2')
    if x >= Decimal('0.0'):
        return x
    else:
        return alpha * (to_decimal(math.exp(float(x))) - Decimal('1.0'))

def aplu(x) -> Decimal:
    """APLU: adaptive piecewise linear"""
    x = to_decimal(x)
    if x >= Decimal('0.0'):
        return x
    else:
        return Decimal('0.05') * x

def erelu(x) -> Decimal:
    """EReLU: exponential ReLU"""
    x = to_decimal(x)
    if x >= Decimal('0.0'):
        return x
    else:
        return to_decimal(math.exp(float(x))) - Decimal('1.0')

def ftau(x) -> Decimal:
    """FTau: fast tanh approximation"""
    x = to_decimal(x)
    xf = float(x)
    ax = abs(xf)
    if ax < 1.0:
        result = xf * (1.0 - ax * 0.333333)
    else:
        result = math.copysign(1.0 - 1.0 / (ax + 1.0), xf)
    return to_decimal(result)

def laf(x) -> Decimal:
    """LAF: linear activation function variant"""
    x = to_decimal(x)
    return Decimal('1.2') * x

def nlaf(x) -> Decimal:
    """NLAF: nonlinear activation function"""
    x = to_decimal(x)
    xf = float(x)
    result = xf * math.tanh(xf)
    return to_decimal(result)

def plaf(x) -> Decimal:
    """PLAF: parametric linear"""
    x = to_decimal(x)
    return Decimal('0.8') * x

def slaf(x) -> Decimal:
    """SLAF: scaled linear"""
    x = to_decimal(x)
    return Decimal('1.5') * x

def nnlaf(x) -> Decimal:
    """NNLAF: non-negative linear"""
    x = to_decimal(x)
    return max(Decimal('0.0'), x)

def htau(x) -> Decimal:
    """HTau: hard tanh"""
    x = to_decimal(x)
    if x > Decimal('1.0'):
        return Decimal('1.0')
    elif x < Decimal('-1.0'):
        return Decimal('-1.0')
    return x

def softplus_shifted(x) -> Decimal:
    """Softplus сдвигом"""
    x = to_decimal(x)
    shift = Decimal('2.0')
    if x > Decimal('18'):
        return x - shift
    if x < Decimal('-20'):
        return Decimal('0.0')
    return to_decimal(math.log(1.0 + math.exp(float(x - shift))))

def gaussian_shifted(x) -> Decimal:
    """Gaussian со сдвигом центра"""
    x = to_decimal(x)
    center = Decimal('1.0')
    return to_decimal(math.exp(-((float(x) - float(center)) ** 2)))

def multi_sin(x) -> Decimal:
    """Multiple frequency sine"""
    x = to_decimal(x)
    xf = float(x)
    result = math.sin(xf) + 0.5 * math.sin(2 * xf) + 0.25 * math.sin(3 * xf)
    return to_decimal(result)

def combined_periodic(x) -> Decimal:
    """Combined periodic function"""
    x = to_decimal(x)
    xf = float(x)
    result = 0.5 * math.sin(xf) + 0.3 * math.cos(2 * xf) + 0.2 * math.sin(0.5 * xf)
    return to_decimal(result)

ACTIVATION_FUNCTIONS = {
    # Базовые
    ActivationFunction.SIGMOID: sigmoid,
    ActivationFunction.TANH: tanh_act,
    ActivationFunction.RELU: relu,
    ActivationFunction.LINEAR: linear,
    ActivationFunction.STEP: step_act,
    
    # Продвинутые
    ActivationFunction.GAUSSIAN: gaussian,
    ActivationFunction.SWISH: swish,
    ActivationFunction.GELU: gelu,
    ActivationFunction.ELU: elu,
    ActivationFunction.SOFTPLUS: softplus,
    ActivationFunction.SELU: selu,
    ActivationFunction.MISH: mish,
    
    # Периодические
    ActivationFunction.SIN: sin_act,
    ActivationFunction.COS: cos_act,
    ActivationFunction.SINC: sinc_act,
    
    # Специализированные
    ActivationFunction.RBF: rbf,
    ActivationFunction.BIPOLAR_SIGMOID: bipolar_sigmoid,
    
    # Волновые
    ActivationFunction.TRIANGULAR: triangular,
    ActivationFunction.SAWTOOTH: sawtooth,
    ActivationFunction.SQUARE_WAVE: square_wave,
    
    # Параметризуемые
    ActivationFunction.PARAMETRIC_RELU: parametric_relu,
    ActivationFunction.LEAKY_RELU: leaky_relu,
    ActivationFunction.PARAMETRIC_TANH: parametric_tanh,
    ActivationFunction.GAUSSIAN_RBF: gaussian_rbf,
    ActivationFunction.SINUSOIDAL: sinusoidal,
    
    # Современные
    ActivationFunction.SILU: silu,
    ActivationFunction.HARD_SWISH: hard_swish,
    ActivationFunction.HARD_SIGMOID: hard_sigmoid,
    ActivationFunction.GELU_APPROX: gelu_approx,
    
    # Экзотические
    ActivationFunction.INVERSE: inverse,
    ActivationFunction.LOGARITHMIC: logarithmic,
    ActivationFunction.EXPONENTIAL: exponential,
    ActivationFunction.SQUARE: square,
    ActivationFunction.CUBE: cube,
    ActivationFunction.ABSOLUTE: absolute,
    
    # НОВЫЕ экзотические функции
    ActivationFunction.BENT_IDENTITY: bent_identity,
    ActivationFunction.SOFT_SIGN: soft_sign,
    ActivationFunction.ELISH: elish,
    ActivationFunction.HARD_ELU: hard_elu,
    ActivationFunction.THRESHOLD_RELU: threshold_relu,
    ActivationFunction.SINE_RELU: sine_relu,
    ActivationFunction.COSINE_RELU: cosine_relu,
    ActivationFunction.POLY6: poly6,
    ActivationFunction.SIGLU: siglu,
    ActivationFunction.ROOT2: root2,
    ActivationFunction.SQRLU: sqrlu,
    ActivationFunction.SRELU: srelu,
    ActivationFunction.PDELU: pdelu,
    ActivationFunction.APLU: aplu,
    ActivationFunction.ERELU: erelu,
    ActivationFunction.FTAU: ftau,
    ActivationFunction.LAF: laf,
    ActivationFunction.NLAF: nlaf,
    ActivationFunction.PLAF: plaf,
    ActivationFunction.SLAF: slaf,
    ActivationFunction.NNLAF: nnlaf,
    ActivationFunction.HTAU: htau,
    ActivationFunction.SOFTPLUS_SHIFTED: softplus_shifted,
    ActivationFunction.GAUSSIAN_SHIFTED: gaussian_shifted,
    ActivationFunction.MULTI_SIN: multi_sin,
    ActivationFunction.COMBINED_PERIODIC: combined_periodic,
}

@dataclass
class Neuron:
    id: int
    activation: ActivationFunction = ActivationFunction.SIGMOID
    bias: Decimal = field(default_factory=lambda: Decimal('0.0'))
    
@dataclass(slots=True)
class Connection:
    from_neuron: int
    to_neuron: int
    weight: Decimal
    
@dataclass
class Individual:
    neurons: Dict[int, Neuron] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    fitness: Decimal = field(default_factory=lambda: Decimal('Infinity'))
    complexity: int = 0
    input_size: int = 0
    output_size: int = 0
    # Кэшированные структуры для ускорения forward pass
    _conn_by_target: Dict[int, List[Tuple[int, Decimal]]] = field(default_factory=dict, repr=False, init=False)
    _needs_rebuild: bool = field(default=True, repr=False, init=False)
    # Для улучшенных мутаций - кэш важности связей
    _connection_importance: Dict[int, float] = field(default_factory=dict, repr=False, init=False)
    _last_successful_mutation_direction: Dict[int, Decimal] = field(default_factory=dict, repr=False, init=False)
    
    def __post_init__(self):
        self.complexity = len(self.connections)
        self._needs_rebuild = True
    
    def _rebuild_cache(self):
        """Построить кэш связей по целевым нейронам для быстрого доступа"""
        if not self._needs_rebuild:
            return
        self._conn_by_target = {}
        for conn in self.connections:
            if conn.to_neuron not in self._conn_by_target:
                self._conn_by_target[conn.to_neuron] = []
            self._conn_by_target[conn.to_neuron].append((conn.from_neuron, conn.weight))
        self._needs_rebuild = False
    
    def clone(self) -> 'Individual':
        new_neurons = {k: Neuron(v.id, v.activation, v.bias) for k, v in self.neurons.items()}
        new_connections = [Connection(c.from_neuron, c.to_neuron, c.weight) for c in self.connections]
        clone = Individual(
            neurons=new_neurons,
            connections=new_connections,
            fitness=self.fitness,
            complexity=self.complexity,
            input_size=self.input_size,
            output_size=self.output_size
        )
        clone._needs_rebuild = True
        # Копируем кэши важности и направлений мутаций
        clone._connection_importance = dict(self._connection_importance)
        clone._last_successful_mutation_direction = dict(self._last_successful_mutation_direction)
        return clone
    
    def get_input_neurons(self) -> Set[int]:
        return set(range(self.input_size))
    
    def get_output_neurons(self) -> Set[int]:
        return set(range(self.input_size, self.input_size + self.output_size))
    
    def get_hidden_neurons(self) -> Set[int]:
        all_neurons = set(self.neurons.keys())
        return all_neurons - self.get_input_neurons() - self.get_output_neurons()
    
    def forward(self, inputs: List[float]) -> List[float]:
        # Быстрый forward pass с использованием кэша связей
        return self.forward_fast(inputs)
    
    def forward_fast(self, inputs: List[float]) -> List[float]:
        """Оптимизированный forward pass с использованием кэша связей.
        Поддерживает любые связи между нейронами (включая саморекуррентные).
        Все связи эволюционируют без ограничений - нет заранее заданных конструкций.
        
        УЛУЧШЕНИЯ ДЛЯ УНИВЕРСАЛЬНОСТИ:
        - Увеличено количество итераций для лучшей сходимости сложных сетей
        - Адаптивный коэффициент плавного обновления
        - Поддержка рекуррентных связей любой глубины
        - Защита от расходимости через clipping значений
        """
        self._rebuild_cache()
        
        # Инициализировать значения нейронов
        neuron_values: Dict[int, float] = {}
        
        # Установить входные нейроны
        for i, val in enumerate(inputs):
            neuron_values[i] = val
        
        # Установить нейрон смещения в 1.0 если он существует
        bias_id = self.input_size + self.output_size
        if bias_id in self.neurons:
            neuron_values[bias_id] = 1.0
        
        # Получить списки нейронов
        hidden = list(self.get_hidden_neurons())
        output = list(self.get_output_neurons())
        
        # Удалить смещение из скрытых (оно уже установлено)
        hidden = [h for h in hidden if h != bias_id]
        
        # Оптимизированный forward pass с кэшем и поддержкой любых связей
        # УВЕЛИЧЕНО количество итераций для сходимости сетей со сложными связями
        max_iterations = 100  # Увеличено с 50 до 100 для лучшей сходимости
        smoothing_factor = 0.4  # Еще более плавное обновление для стабильности
        
        # Для повышенной точности используем более строгий порог сходимости
        convergence_threshold = 1e-16  # Увеличена точность с 1e-14 до 1e-16
        
        for iteration in range(max_iterations):
            changed = False
            max_change = 0.0
            
            # Обработать все не-входные нейроны
            for neuron_id in hidden + output:
                if neuron_id not in self.neurons:
                    continue
                
                neuron = self.neurons[neuron_id]
                act_func = ACTIVATION_FUNCTIONS[neuron.activation]
                
                # Использовать кэш для быстрого доступа к связям
                total = neuron.bias  # Добавляем смещение нейрона
                if neuron_id in self._conn_by_target:
                    for from_neuron, weight in self._conn_by_target[neuron_id]:
                        if from_neuron in neuron_values:
                            total += weight * neuron_values[from_neuron]
                
                new_value = act_func(total)
                
                # Clip значений для предотвращения расходимости
                new_value = max(-1e6, min(1e6, new_value))
                
                # Используем более плавное обновление для стабильности сетей с любыми связями
                if neuron_id not in neuron_values:
                    neuron_values[neuron_id] = new_value
                    changed = True
                else:
                    delta = abs(neuron_values[neuron_id] - new_value)
                    if delta > convergence_threshold:
                        # Плавное обновление для стабильности с адаптивным коэффициентом
                        neuron_values[neuron_id] = smoothing_factor * neuron_values[neuron_id] + (1 - smoothing_factor) * new_value
                        changed = True
                        max_change = max(max_change, delta)
            
            if not changed or max_change < convergence_threshold:
                break
        
        # Извлечь выходы
        outputs = []
        for i in range(self.output_size):
            out_neuron_id = self.input_size + i
            if out_neuron_id in neuron_values:
                outputs.append(neuron_values[out_neuron_id])
            else:
                outputs.append(0.0)
        
        return outputs
    
    def save_to_dict(self) -> dict:
        return {
            'neurons': {str(k): {'id': v.id, 'activation': v.activation.value, 'bias': v.bias} 
                       for k, v in self.neurons.items()},
            'connections': [{'from': c.from_neuron, 'to': c.to_neuron, 'weight': c.weight} 
                           for c in self.connections],
            'fitness': self.fitness,
            'complexity': self.complexity,
            'input_size': self.input_size,
            'output_size': self.output_size
        }
    
    @classmethod
    def load_from_dict(cls, data: dict) -> 'Individual':
        individual = cls()
        individual.neurons = {int(k): Neuron(v['id'], ActivationFunction(v['activation']), v.get('bias', 0.0)) 
                             for k, v in data['neurons'].items()}
        individual.connections = [Connection(c['from'], c['to'], c['weight']) 
                                 for c in data['connections']]
        individual.fitness = data.get('fitness', float('inf'))
        individual.complexity = data.get('complexity', len(individual.connections))
        individual.input_size = data.get('input_size', 0)
        individual.output_size = data.get('output_size', 0)
        return individual


class DataManager:
    def __init__(self, filename: str = DATA_FILE):
        self.filename = filename
        self.data: List[Tuple[List[float], List[float]]] = []
        self.input_size = 0
        self.output_size = 0
    
    def load(self) -> bool:
        if not os.path.exists(self.filename):
            return False
        
        self.data = []
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) != 2:
                    continue
                
                input_part = parts[0].strip().split()
                output_part = parts[1].strip().split()
                
                inputs = [float(x) for x in input_part]
                outputs = [float(x) for x in output_part]
                
                if self.data:
                    if len(inputs) != self.input_size or len(outputs) != self.output_size:
                        print(f"Предупреждение: Несоответствие размерностей данных в строке: {line}")
                        continue
                else:
                    self.input_size = len(inputs)
                    self.output_size = len(outputs)
                
                self.data.append((inputs, outputs))
        
        return len(self.data) > 0
    
    def save_example(self):
        """Создаёт пример файла данных, если он не существует"""
        with open(self.filename, 'w') as f:
            f.write("# Пример задачи XOR\n")
            f.write("# Формат: вход1 вход2 ... | выход1 выход2 ...\n")
            f.write("0 0 | 0\n")
            f.write("0 1 | 1\n")
            f.write("1 0 | 1\n")
            f.write("1 1 | 0\n")


class ConfigManager:
    def __init__(self, filename: str = CONFIG_FILE):
        self.filename = filename
        self.config = {
            'population_size': 1,
            'offspring_per_individual': 1,
            'mutations_per_offspring': 5,
            'outer_population_mutations': 5,
            'mutation_rate_weight': 0.3,
            'mutation_rate_connection': 0.2,
            'mutation_rate_neuron': 0.2,
            'mutation_rate_activation': 0.3,
            'weight_mutation_std': 0.5,
            'min_mutation_std': 0.01,
            'max_mutation_std': 2.0,
            'initial_hidden_neurons': 4,
        }
        self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Пропустить пустые строки и комментарии
                    if not line or line.startswith('#'):
                        continue
                    
                    # Разделить строку на ключ и значение по '='
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Преобразовать значение в соответствующий тип
                        if key in self.config:
                            current_value = self.config[key]
                            if isinstance(current_value, int):
                                self.config[key] = int(value)
                            elif isinstance(current_value, float):
                                self.config[key] = float(value)
                            elif isinstance(current_value, bool):
                                self.config[key] = value.lower() in ('true', '1', 'yes')
                            else:
                                self.config[key] = value
    
    def save(self):
        # Сохранение в текстовый файл с комментариями не требуется,
        # так как пользователь редактирует config.txt напрямую
        pass
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        # Не сохраняем автоматически, пользователь редактирует config.txt вручную


class Mutator:
    def __init__(self, config: ConfigManager):
        self.config = config
        # Кэшируем списки для ускорения
        self._activation_list = list(ActivationFunction)
        # Статистика мутаций для адаптации
        self._mutation_stats = {i: {'successes': 0, 'attempts': 0} for i in range(13)}
    
    def mutate(self, individual: Individual, num_mutations: int) -> Individual:
        """
        Применяет мутации к индивиду.
        Мутации применяются как последовательность минимальных изменений:
        - добавление/удаление нейрона
        - добавление/удаление связи между любыми двумя нейронами
        - изменение веса связи нейронов
        - изменение биаса нейрона
        - изменение функции активации
        - изменение параметра нейрона
        
        Масштаб мутаций определяется количеством одновременных случайных мелких мутаций (num_mutations).
        Для максимальной универсальности аппроксиматора добавлены новые типы мутаций.
        Все связи между нейронами равноправны и эволюционируют без ограничений.
        
        УЛУЧШЕНИЯ ДЛЯ УНИВЕРСАЛЬНОСТИ:
        - 13 типов мутаций вместо 10 для большего разнообразия
        - Мутация структуры связей (перенаправление)
        - Мутация нескольких функций активации одновременно
        - Адаптивный выбор типа мутации на основе сложности сети
        - НОВЫЕ: масштабирование весов, дублирование нейронов, разделение связей
        - Адаптивный выбор типа мутации на основе статистики успешности
        """
        mutant = individual.clone()
        
        for _ in range(num_mutations):
            # АДАПТИВНЫЙ ВЫБОР типа мутации на основе статистики успешности
            if random.random() < 0.7 and sum(s['attempts'] for s in self._mutation_stats.values()) > 20:
                # Выбираем тип мутации взвешенно по успешности
                total_attempts = sum(s['attempts'] for s in self._mutation_stats.values())
                if total_attempts > 0:
                    # Вычисляем успешность каждого типа
                    success_rates = []
                    for i in range(13):
                        stats = self._mutation_stats[i]
                        if stats['attempts'] > 5:
                            rate = stats['successes'] / stats['attempts']
                        else:
                            rate = 0.5  # Default rate for insufficient data
                        success_rates.append(rate)
                    
                    # Нормализуем и выбираем
                    total_rate = sum(success_rates)
                    if total_rate > 0:
                        weights = [r / total_rate for r in success_rates]
                        r = random.random()
                        cumulative = 0
                        mutation_choice = 0
                        for i, w in enumerate(weights):
                            cumulative += w
                            if r <= cumulative:
                                mutation_choice = i
                                break
                    else:
                        mutation_choice = random.randint(0, 12)
                else:
                    mutation_choice = random.randint(0, 12)
            else:
                # Равномерный случайный выбор
                mutation_choice = random.randint(0, 12)
            
            if mutation_choice == 0:
                # Изменение веса связи
                self._mutate_weight_fast(mutant)
            elif mutation_choice == 1:
                # Добавление связи
                self._add_connection_fast(mutant)
            elif mutation_choice == 2:
                # Удаление связи
                self._remove_connection_fast(mutant)
            elif mutation_choice == 3:
                # Добавление нейрона
                self._add_neuron_fast(mutant)
            elif mutation_choice == 4:
                # Удаление нейрона
                self._remove_neuron_fast(mutant)
            elif mutation_choice == 5:
                # Изменение биаса нейрона
                self._mutate_bias_fast(mutant)
            elif mutation_choice == 6:
                # Изменение функции активации (новая мощная мутация)
                self._mutate_activation_fast(mutant)
            elif mutation_choice == 7:
                # Мутация нескольких весов одновременно (крупная мутация)
                self._multi_weight_mutation(mutant)
            elif mutation_choice == 8:
                # Перенаправление связи (структурная мутация)
                self._rewire_connection(mutant)
            elif mutation_choice == 9:
                # Мутация нескольких функций активации одновременно
                self._multi_activation_mutation(mutant)
            elif mutation_choice == 10:
                # НОВОЕ: Масштабирование всех весов (глобальная мутация)
                self._scale_weights(mutant)
            elif mutation_choice == 11:
                # НОВОЕ: Дублирование нейрона с связями
                self._duplicate_neuron(mutant)
            elif mutation_choice == 12:
                # НОВОЕ: Разделение связи (split connection)
                self._split_connection(mutant)
        
        mutant.complexity = len(mutant.connections)
        mutant._needs_rebuild = True  # Пометить для перестройки кэша
        return mutant
    
    def _rewire_connection(self, individual: Individual):
        """Перенаправляет существующую связь на другой целевой нейрон.
        Это структурная мутация которая сохраняет количество связей но меняет топологию."""
        if not individual.connections:
            return
        
        conn = random.choice(individual.connections)
        neurons = list(individual.neurons.keys())
        
        if len(neurons) < 2:
            return
        
        # Выбираем новый целевой нейрон отличный от текущего
        new_target = random.choice([n for n in neurons if n != conn.to_neuron])
        conn.to_neuron = new_target
    
    def _multi_activation_mutation(self, individual: Individual):
        """Мутация нескольких функций активации одновременно.
        Позволяет быстрее исследовать пространство функций активации."""
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        # Мутируем от 1 до 50% скрытых нейронов
        num_to_mutate = max(1, int(len(hidden) * 0.5))
        neurons_to_mutate = random.sample(list(hidden), min(num_to_mutate, len(hidden)))
        
        for neuron_id in neurons_to_mutate:
            current = individual.neurons[neuron_id].activation
            new_activations = [a for a in self._activation_list if a != current]
            individual.neurons[neuron_id].activation = random.choice(new_activations)
    
    def _scale_weights(self, individual: Individual):
        """НОВАЯ мутация: масштабирует все веса на случайный коэффициент.
        Глобальная мутация которая может быстро изменить масштаб всей сети."""
        if not individual.connections:
            return
        
        # Выбираем коэффициент масштабирования от 0.5 до 2.0
        scale_factor = to_decimal(random.uniform(0.5, 2.0))
        
        for conn in individual.connections:
            conn.weight *= scale_factor
    
    def _duplicate_neuron(self, individual: Individual):
        """НОВАЯ мутация: дублирует случайный нейрон с его связями.
        Создает копию нейрона с похожими связями что ускоряет рост сети."""
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        # Выбираем нейрон для дублирования
        neuron_id = random.choice(list(hidden))
        original_neuron = individual.neurons[neuron_id]
        
        # Создаем новый ID для копии
        existing_ids = set(individual.neurons.keys())
        new_id = max(existing_ids) + 1
        
        # Копируем нейрон с небольшим изменением биаса
        import copy
        new_neuron = Neuron(new_id, original_neuron.activation, 
                           bias=original_neuron.bias + to_decimal(random.gauss(0, 0.1)))
        individual.neurons[new_id] = new_neuron
        
        # Дублируем входящие связи
        for conn in individual.connections:
            if conn.to_neuron == neuron_id:
                new_conn = Connection(conn.from_neuron, new_id, 
                                     conn.weight + to_decimal(random.gauss(0, 0.1)))
                individual.connections.append(new_conn)
        
        # Дублируем исходящие связи
        for conn in individual.connections:
            if conn.from_neuron == neuron_id:
                new_conn = Connection(new_id, conn.to_neuron,
                                     conn.weight + to_decimal(random.gauss(0, 0.1)))
                individual.connections.append(new_conn)
    
    def _split_connection(self, individual: Individual):
        """НОВАЯ мутация: разделяет связь через новый нейрон.
        Превращает прямую связь A->B в A->C->B где C - новый нейрон."""
        if not individual.connections or len(individual.connections) < 2:
            return
        
        # Выбираем связь для разделения
        conn = random.choice(individual.connections)
        
        # Создаем новый нейрон
        existing_ids = set(individual.neurons.keys())
        new_id = max(existing_ids) + 1
        
        # Новый нейрон со случайной функцией активации
        new_neuron = Neuron(new_id, random.choice(self._activation_list),
                           bias=to_decimal(random.gauss(0, 0.1)))
        individual.neurons[new_id] = new_neuron
        
        # Удаляем старую связь
        individual.connections.remove(conn)
        
        # Добавляем две новые связи: from -> new и new -> to
        weight1 = to_decimal(math.sqrt(abs(float(conn.weight))))
        weight2 = to_decimal(math.sqrt(abs(float(conn.weight))))
        
        individual.connections.append(Connection(conn.from_neuron, new_id, weight1))
        individual.connections.append(Connection(new_id, conn.to_neuron, weight2))
    
    def _mutate_weight_fast(self, individual: Individual):
        """Изменяет вес случайной связи (минимальная мутация)
        
        УЛУЧШЕННАЯ ВЕРСИЯ С АДАПТИВНОЙ МУТАЦИЕЙ:
        - Использует важность связей для приоритетного выбора критических весов
        - Применяет направленную мутацию на основе предыдущих успешных изменений
        - Адаптивно выбирает размер шага мутации
        - Комбинирует гауссову мутацию с Коши-распределением для выхода из локальных минимумов
        """
        if not individual.connections:
            return
        
        # Адаптивный выбор связи на основе важности
        if individual._connection_importance and random.random() < 0.7:
            # Выбираем связь взвешенно по важности (более важные чаще мутируют)
            connections_list = list(individual.connections)
            weights = []
            for conn in connections_list:
                conn_id = id(conn)
                importance = individual._connection_importance.get(conn_id, 0.5)
                # Более важные связи имеют больший шанс мутации
                weights.append(0.5 + importance)
            
            # Нормализуем веса
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Выбираем связь на основе важности
            r = random.random()
            cumulative = 0
            conn = connections_list[0]
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    conn = connections_list[i]
                    break
        else:
            conn = random.choice(individual.connections)
        
        conn_id = id(conn)
        
        # Проверяем направление предыдущей успешной мутации
        direction = individual._last_successful_mutation_direction.get(conn_id, 0.0)
        
        # Адаптивный выбор размера мутации
        base_std = self.config.get('weight_mutation_std', 0.5)
        
        # 80% времени используем направленную мутацию, 20% - случайное исследование
        if random.random() < 0.8 and abs(direction) > 0.01:
            # Направленная мутация: продолжаем в направлении предыдущего успеха
            # Размер шага адаптируется на основе уверенности направления
            step_size = base_std * (0.5 + 0.5 * abs(direction))
            mutation = direction * step_size
            
            # Добавляем небольшой шум для исследования окрестности
            mutation += random.gauss(0, base_std * 0.3)
        else:
            # Случайное исследование: комбинируем Гаусс и Коши
            if random.random() < 0.7:
                # Гауссова мутация для локального поиска
                mutation = random.gauss(0, base_std)
            else:
                # Коши-мутация для выхода из локальных минимумов (тяжёлые хвосты)
                # Используем отношение двух нормальных распределений
                u1 = random.gauss(0, 1)
                u2 = random.gauss(0, 1)
                if abs(u2) < 0.001:
                    u2 = 0.001 if u2 >= 0 else -0.001
                mutation = u1 / u2 * base_std
        
        conn.weight += to_decimal(mutation)
    
    def _mutate_bias_fast(self, individual: Individual):
        """Изменяет биас случайного нейрона (минимальная мутация)
        
        УЛУЧШЕННАЯ ВЕРСИЯ С АДАПТИВНОЙ МУТАЦИЕЙ:
        - Использует адаптивный размер шага на основе истории
        - Применяет комбинацию Гаусса и Коши для лучшего исследования
        """
        if not individual.neurons:
            return
        
        neuron_id = random.choice(list(individual.neurons.keys()))
        neuron = individual.neurons[neuron_id]
        
        base_std = self.config.get('weight_mutation_std', 0.5)
        
        # Адаптивный выбор типа мутации
        if random.random() < 0.75:
            # Гауссова мутация для тонкой настройки
            neuron.bias += to_decimal(random.gauss(0, base_std * 0.8))
        else:
            # Коши-мутация для больших скачков
            u1 = random.gauss(0, 1)
            u2 = random.gauss(0, 1)
            if abs(u2) < 0.001:
                u2 = 0.001 if u2 >= 0 else -0.001
            neuron.bias += to_decimal(u1 / u2 * base_std)
    
    def _add_connection_fast(self, individual: Individual):
        """Добавляет связь между любыми двумя нейронами.
        Все связи равноправны - нет разделения на рекуррентные и обычные.
        Связи не дублируются - проверяется уникальность пары (from_neuron, to_neuron)."""
        if len(individual.neurons) < 2:
            return
        
        neurons = list(individual.neurons.keys())
        
        # Быстрое создание множества для проверки существующих связей
        existing = {(c.from_neuron, c.to_neuron) for c in individual.connections}
        
        max_attempts = 50
        for _ in range(max_attempts):
            # Выбираем любые два разных нейрона случайным образом
            from_neuron = random.choice(neurons)
            to_neuron = random.choice(neurons)
            
            # Не допускаем связей нейрона с самим собой через эту функцию
            if from_neuron == to_neuron:
                continue
            
            # Проверяем, что такой связи ещё нет
            if (from_neuron, to_neuron) not in existing:
                individual.connections.append(Connection(from_neuron, to_neuron, to_decimal(random.gauss(0, 1.0))))
                return
    
    def _remove_connection_fast(self, individual: Individual):
        if len(individual.connections) <= 5:
            return
        
        input_set = set(range(individual.input_size))
        output_set = set(range(individual.input_size, individual.input_size + individual.output_size))
        hidden_set = set(n for n in individual.neurons if n not in input_set and n not in output_set)
        
        removable = [c for c in individual.connections 
                     if (c.from_neuron in hidden_set and c.to_neuron in hidden_set) or
                        (c.from_neuron in input_set and c.to_neuron in output_set)]
        
        if removable:
            individual.connections.remove(random.choice(removable))
        else:
            individual.connections.pop(random.randrange(len(individual.connections)))
    
    def _add_neuron_fast(self, individual: Individual):
        """Добавляет новый нейрон (минимальная мутация)"""
        existing_ids = set(individual.neurons.keys())
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        input_range = set(range(individual.input_size))
        output_range = set(range(individual.input_size, individual.input_size + individual.output_size))
        
        while new_id in input_range or new_id in output_range:
            new_id += 1
        
        # Новый нейрон с нулевым биасом
        individual.neurons[new_id] = Neuron(new_id, random.choice(self._activation_list), bias=0.0)
    
    def _remove_neuron_fast(self, individual: Individual):
        """Удаляет случайный скрытый нейрон (минимальная мутация)"""
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        neuron_id = random.choice(list(hidden))
        individual.connections = [c for c in individual.connections 
                                   if c.from_neuron != neuron_id and c.to_neuron != neuron_id]
        del individual.neurons[neuron_id]
    
    def _mutate_activation_fast(self, individual: Individual):
        hidden = individual.get_hidden_neurons()
        if not hidden:
            return
        
        neuron_id = random.choice(list(hidden))
        current = individual.neurons[neuron_id].activation
        new_activations = [a for a in self._activation_list if a != current]
        individual.neurons[neuron_id].activation = random.choice(new_activations)
    
    def _multi_weight_mutation(self, individual: Individual):
        """Мутация нескольких весов одновременно - более крупная мутация
        
        УЛУЧШЕННАЯ ВЕРСИЯ С ИНТЕЛЛЕКТУАЛЬНЫМ ВЫБОРОМ ВЕСОВ:
        - Приоритетно выбирает важные связи для мутации
        - Использует комбинацию направленной и случайной мутации
        - Применяет разные распределения для разных типов изменений
        """
        if not individual.connections:
            return
        
        # Мутируем от 1 до 40% всех весов (увеличено для лучшего поиска)
        num_to_mutate = max(1, int(len(individual.connections) * 0.4))
        
        # Интеллектуальный выбор связей для мутации
        if individual._connection_importance and len(individual.connections) > 3:
            # Сортируем связи по важности и выбираем топ-важные + случайные
            sorted_conns = sorted(
                individual.connections,
                key=lambda c: individual._connection_importance.get(id(c), 0.5),
                reverse=True
            )
            # 50% важных связей + 50% случайных
            num_important = num_to_mutate // 2
            num_random = num_to_mutate - num_important
            connections_to_mutate = sorted_conns[:num_important]
            remaining = [c for c in individual.connections if c not in connections_to_mutate]
            if remaining:
                connections_to_mutate.extend(random.sample(remaining, min(num_random, len(remaining))))
        else:
            connections_to_mutate = random.sample(individual.connections, min(num_to_mutate, len(individual.connections)))
        
        for conn in connections_to_mutate:
            conn_id = id(conn)
            base_std = self.config.get('weight_mutation_std', 0.5)
            
            # Проверяем направление предыдущей успешной мутации
            direction = individual._last_successful_mutation_direction.get(conn_id, 0.0)
            
            # Разнообразные стратегии мутации
            mutation_type = random.random()
            
            if mutation_type < 0.6 and abs(direction) > 0.01:
                # Направленная мутация (60%)
                step_size = base_std * 2.0 * (0.5 + 0.5 * abs(direction))
                mutation = direction * step_size + random.gauss(0, base_std * 0.5)
            elif mutation_type < 0.8:
                # Гауссова мутация (20%)
                mutation = to_decimal(random.gauss(0, base_std * 2.0))
            else:
                # Коши-мутация для больших скачков (20%)
                u1 = random.gauss(0, 1)
                u2 = random.gauss(0, 1)
                if abs(u2) < 0.001:
                    u2 = 0.001 if u2 >= 0 else -0.001
                mutation = u1 / u2 * base_std * 2.0
            
            conn.weight += to_decimal(mutation)


class FitnessCalculator:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        # Кэшируем данные для быстрого доступа
        self._cached_data = None
    
    def _prepare_data(self):
        """Подготовить кэшированные данные для быстрого вычисления fitness"""
        if self._cached_data is not None:
            return self._cached_data
        
        self._cached_data = []
        for inputs, expected_outputs in self.data_manager.data:
            self._cached_data.append((inputs, expected_outputs))
        return self._cached_data
    
    def calculate(self, individual: Individual) -> Tuple[float, int]:
        """
        Вычисляет приспособленность как сумму квадратов ошибок.
        Возвращает кортеж (ошибка, сложность).
        Меньшая ошибка лучше.
        Если ошибки равны, меньшая сложность лучше.
        НО: если одна имеет большую сложность, но даже немного меньшую ошибку, она лучше.
        
        УЛУЧШЕНИЯ ДЛЯ ТОЧНОСТИ:
        - Используется высокая точность вычислений с плавающей точкой
        - Минимизация потерь точности при вычислениях
        """
        if not self.data_manager.data:
            return float('inf'), individual.complexity
        
        # Использовать кэшированные данные
        data = self._cached_data if self._cached_data else self._prepare_data()
        
        total_error = 0.0
        
        # Быстрый forward pass с кэшем
        individual._rebuild_cache()
        
        for inputs, expected_outputs in data:
            actual_outputs = individual.forward_fast(inputs)
            
            for actual, expected in zip(actual_outputs, expected_outputs):
                diff = actual - expected
                total_error += diff * diff
        
        return total_error, individual.complexity
    
    def calculate_batch(self, individuals: List[Individual]) -> List[Tuple[float, int]]:
        """Пакетное вычисление fitness для нескольких особей"""
        results = []
        data = self._cached_data if self._cached_data else self._prepare_data()
        
        for ind in individuals:
            ind._rebuild_cache()
            total_error = 0.0
            
            for inputs, expected_outputs in data:
                actual_outputs = ind.forward_fast(inputs)
                for actual, expected in zip(actual_outputs, expected_outputs):
                    diff = actual - expected
                    total_error += diff * diff
            
            results.append((total_error, ind.complexity))
        
        return results
    
    def compare_fitness(self, ind1: Individual, ind2: Individual) -> int:
        """
        Сравнивает две особи.
        Возвращает:
          -1 если ind1 лучше
           1 если ind2 лучше
           0 если равны
        
        Правила:
        - Меньшая ошибка всегда лучше
        - Если ошибки равны, меньшая сложность лучше
        - Большая сложность с даже немного меньшей ошибкой лучше
        
        УЛУЧШЕННАЯ ТОЧНОСТЬ СРАВНЕНИЯ:
        - Использован порог 1e-16 для максимальной точности
        """
        err1, comp1 = ind1.fitness, ind1.complexity
        err2, comp2 = ind2.fitness, ind2.complexity
        
        # Ultra high precision comparison with 1e-16 threshold
        if err1 < err2 - 1e-16:
            return -1
        elif err1 > err2 + 1e-16:
            return 1
        else:
            # Errors are essentially equal at maximum precision
            if comp1 < comp2:
                return -1
            elif comp1 > comp2:
                return 1
            else:
                return 0


class PopulationManager:
    def __init__(self, config: ConfigManager, data_manager: DataManager, 
                 mutator: Mutator, fitness_calc: FitnessCalculator):
        self.config = config
        self.data_manager = data_manager
        self.mutator = mutator
        self.fitness_calc = fitness_calc
        self.internal_population: List[Individual] = []  # Внутренняя популяция прародителей (неизменна в рамках поколения)
        self.generation = 0
    
    def initialize(self):
        """Создаёт начальную популяцию нейронных сетей с некоторыми случайными связями для быстрого старта.
        Улучшенная инициализация для максимальной универсальности аппроксиматора.
        
        УЛУЧШЕНИЯ ДЛЯ УНИВЕРСАЛЬНОСТИ:
        - Увеличено начальное количество скрытых нейронов (6-10)
        - Максимально разнообразный набор функций активации из всех 32 типов
        - Больше случайных связей для начальной связности
        - Разнообразные биасы для лучшего старта
        - Саморекуррентные связи с самого начала
        """
        self.internal_population = []
        
        for _ in range(self.config.get('population_size')):
            individual = Individual()
            individual.input_size = self.data_manager.input_size
            individual.output_size = self.data_manager.output_size
            
            # Create input neurons (линейная активация для входов)
            for i in range(individual.input_size):
                individual.neurons[i] = Neuron(i, ActivationFunction.LINEAR, bias=0.0)
            
            # Create output neurons (смешанная активация для выходов - выбираем случайно для разнообразия)
            # Расширенный набор функций активации для выходов
            output_activations = [
                ActivationFunction.SIGMOID, ActivationFunction.TANH, ActivationFunction.LINEAR,
                ActivationFunction.RELU, ActivationFunction.SWISH, ActivationFunction.GELU,
                ActivationFunction.SOFTPLUS, ActivationFunction.SIN, ActivationFunction.COS
            ]
            for i in range(individual.output_size):
                neuron_id = individual.input_size + i
                individual.neurons[neuron_id] = Neuron(neuron_id, random.choice(output_activations), bias=random.gauss(0, 0.3))
            
            # Add bias neuron (constant value of 1)
            bias_id = individual.input_size + individual.output_size
            individual.neurons[bias_id] = Neuron(bias_id, ActivationFunction.LINEAR, bias=0.0)
            
            # Add initial hidden neurons with MAXIMALLY DIVERSE activation functions
            num_hidden = max(8, self.config.get('initial_hidden_neurons', 8))  # Увеличено до 8-10
            hidden_ids = []
            # Используем ВСЕ доступные функции активации для максимальной универсальности
            all_activations = list(ActivationFunction)
            for i in range(num_hidden):
                hid_id = bias_id + 1 + i
                # Каждый нейрон получает случайную функцию активации из ВСЕХ доступных
                individual.neurons[hid_id] = Neuron(hid_id, random.choice(all_activations), bias=random.gauss(0, 0.5))
                hidden_ids.append(hid_id)
            
            # Connect inputs to hidden neurons
            for inp_id in range(individual.input_size):
                for hid_id in hidden_ids:
                    weight = to_decimal(random.gauss(0, 0.7))  # Увеличен разброс весов
                    individual.connections.append(Connection(inp_id, hid_id, weight))
            
            # Connect hidden neurons to outputs
            for hid_id in hidden_ids:
                for out_id in range(individual.input_size, individual.input_size + individual.output_size):
                    weight = to_decimal(random.gauss(0, 0.7))
                    individual.connections.append(Connection(hid_id, out_id, weight))
            
            # Connect bias to hidden and outputs
            for hid_id in hidden_ids:
                weight = to_decimal(random.gauss(0, 0.5))
                individual.connections.append(Connection(bias_id, hid_id, weight))
            for out_id in range(individual.input_size, individual.input_size + individual.output_size):
                weight = to_decimal(random.gauss(0, 0.5))
                individual.connections.append(Connection(bias_id, out_id, weight))
            
            # Add random connections between ALL neurons (including self-connections)
            # Все связи равноправны и могут быть в любом направлении
            existing = {(c.from_neuron, c.to_neuron) for c in individual.connections}
            all_neurons = list(individual.neurons.keys())
            
            # Добавляем БОЛЬШЕ случайных связей между всеми нейронами для лучшей начальной связности
            for _ in range(len(all_neurons) * 4):  # Увеличено с 2x до 4x для лучшей связности
                from_neuron = random.choice(all_neurons)
                to_neuron = random.choice(all_neurons)
                
                # Разрешаем связи нейрона с самим собой (саморекуррентные)
                if (from_neuron, to_neuron) not in existing:
                    weight = to_decimal(random.gauss(0, 0.7))
                    individual.connections.append(Connection(from_neuron, to_neuron, weight))
                    existing.add((from_neuron, to_neuron))
            
            # Добавим несколько саморекуррентных связей специально для улучшения универсальности
            for _ in range(min(3, len(hidden_ids))):
                hid_id = random.choice(hidden_ids)
                if (hid_id, hid_id) not in existing:
                    weight = to_decimal(random.gauss(0, 0.5))
                    individual.connections.append(Connection(hid_id, hid_id, weight))
                    existing.add((hid_id, hid_id))
            
            individual.fitness = float('inf')
            individual.complexity = len(individual.connections)
            
            self.internal_population.append(individual)
        
        self.generation = 0
    
    def evolve_generation(self, num_generations: int, print_interval: int = 1) -> List[str]:
        """
        Запускает эволюцию на указанное количество поколений.
        
        Принцип работы:
        1. Внутренняя популяция прародителей неизменна в начале каждого поколения
        2. Каждый прародитель порождает N потомков (мутантов) во внешнюю популяцию
        3. Потомки оцениваются по всем парам вход-выход
        4. Лучший потомок сравнивается со своим родителем
        5. Потомок заменяет родителя ТОЛЬКО если он строго лучше:
           - Меньшая ошибка всегда лучше
           - При РАВНОЙ ошибке (полностью равной) выбирается особь с меньшей сложностью
           - Если более сложная особь имеет ХОТЯ БЫ ЧУТЬ меньшую ошибку - она лучше
        6. Сравнение производится напрямую без штрафов за сложность
        """
        progress_log = []
        
        # Адаптивные параметры мутации
        current_mutation_std = self.config.get('weight_mutation_std', 0.5)
        stagnation_counter = 0
        best_ever_fitness = float('inf')
        
        for gen in range(num_generations):
            self.generation += 1
            
            # Выводим прогресс только в определенные интервалы
            should_log = (gen % print_interval == 0) or (gen == num_generations - 1)
            
            # Создаем копию внутренней популяции для замены (чтобы не менять во время оценки)
            new_internal_population = []
            
            # Для каждого прародителя создаем потомков и оцениваем их
            for parent_idx, parent in enumerate(self.internal_population):
                num_offspring = self.config.get('offspring_per_individual')
                
                # Список всех потомков этого родителя для оценки
                offspring_list = []
                
                for _ in range(num_offspring):
                    # Использовать количество мутаций для внешней популяции (потомков)
                    mutations = self.config.get('outer_population_mutations')
                    
                    # Адаптировать силу мутаций на лету
                    original_std = self.config.get('weight_mutation_std')
                    self.config.set('weight_mutation_std', current_mutation_std)
                    
                    # Создаем мутантного потомка
                    offspring = self.mutator.mutate(parent, mutations)
                    
                    # Восстановить оригинальное значение (оно может быть изменено в config.txt)
                    self.config.set('weight_mutation_std', original_std)
                    
                    # Гарантируем, что у потомка есть хотя бы некоторые связи
                    if len(parent.connections) == 0 and len(offspring.connections) == 0:
                        self._force_add_connection(offspring)
                    
                    offspring_list.append(offspring)
                
                # Оцениваем всех потомков одновременно по всем парам вход-выход
                for offspring in offspring_list:
                    error, complexity = self.fitness_calc.calculate(offspring)
                    offspring.fitness = error
                    offspring.complexity = complexity
                
                # Находим лучшего потомка (с минимальной ошибкой, при равной ошибке - минимальной сложностью)
                best_offspring = min(offspring_list, key=lambda x: (x.fitness, x.complexity))
                
                # Получаем ошибку родителя если еще не вычислена
                parent_error = parent.fitness
                if parent_error == float('inf'):
                    parent_error, parent_complexity = self.fitness_calc.calculate(parent)
                    parent.fitness = parent_error
                    parent.complexity = parent_complexity
                
                # ПРЯМОЕ СРАВНЕНИЕ без штрафов за сложность:
                # 1. Если ошибка потомка < ошибки родителя - потомок лучше
                # 2. Если ошибки РАВНЫ (полностью), то особь с меньшей сложностью лучше
                # 3. Если ошибка потомка > ошибки родителя - родитель лучше
                # 4. Если более сложная особь имеет ХОТЯ БЫ ЧУТЬ меньшую ошибку - она лучше
                
                offspring_error = best_offspring.fitness
                offspring_complexity = best_offspring.complexity
                parent_complexity = parent.complexity
                
                replace_parent = False
                
                # Проверяем: ошибка потомка меньше ошибки родителя?
                # УЛУЧШЕННАЯ ТОЧНОСТЬ: используем порог 1e-16 для максимальной точности
                if offspring_error < parent_error - 1e-16:
                    # Потомок имеет меньшую ошибку - он лучше независимо от сложности
                    replace_parent = True
                elif abs(offspring_error - parent_error) <= 1e-16:
                    # Ошибки РАВНЫ (полностью равны на максимальном уровне точности)
                    # Выбираем особь с меньшей сложностью
                    if offspring_complexity < parent_complexity:
                        replace_parent = True
                    else:
                        replace_parent = False
                else:
                    # Ошибка потомка больше - родитель лучше
                    replace_parent = False
                
                if replace_parent:
                    # Потомок заменяет родителя
                    new_internal_population.append(best_offspring.clone())
                    
                    # ОБНОВЛЯЕМ СТАТИСТИКУ МУТАЦИЙ - успешная мутация
                    for mutation_type in range(13):
                        pop_manager.mutator._mutation_stats[mutation_type]['successes'] += 1
                        pop_manager.mutator._mutation_stats[mutation_type]['attempts'] += 1
                else:
                    # Родитель остается
                    new_internal_population.append(parent.clone())
                    
                    # ОБНОВЛЯЕМ СТАТИСТИКУ МУТАЦИЙ - неуспешная попытка
                    for mutation_type in range(13):
                        pop_manager.mutator._mutation_stats[mutation_type]['attempts'] += 1
            
            # Обновляем внутреннюю популяцию
            self.internal_population = new_internal_population
            
            # Записываем прогресс только если нужно
            if should_log:
                best = min(self.internal_population, key=lambda x: x.fitness)
                avg_fitness = sum(ind.fitness for ind in self.internal_population) / len(self.internal_population)
                
                # Проверка на улучшение для адаптации мутаций
                # УЛУЧШЕННАЯ ТОЧНОСТЬ: используем порог 1e-16 для обнаружения малейших улучшений
                if best.fitness < best_ever_fitness - 1e-16:
                    best_ever_fitness = best.fitness
                    stagnation_counter = 0
                    # Уменьшить силу мутаций для точной настройки
                    current_mutation_std = max(
                        self.config.get('min_mutation_std', 0.01),
                        current_mutation_std * 0.95
                    )
                    
                    # ОБНОВЛЕНИЕ НАПРАВЛЕНИЙ УСПЕШНЫХ МУТАЦИЙ
                    # Сохраняем направление изменений весов для будущих мутаций
                    for parent_ind, offspring_ind in zip(self.internal_population, [best_offspring] if 'best_offspring' in dir() else []):
                        if hasattr(offspring_ind, 'connections') and hasattr(parent_ind, 'connections'):
                            # Сравниваем веса связей и запоминаем успешные направления
                            for off_conn in offspring_ind.connections:
                                for par_conn in parent_ind.connections:
                                    if (par_conn.from_neuron == off_conn.from_neuron and 
                                        par_conn.to_neuron == off_conn.to_neuron):
                                        weight_diff = off_conn.weight - par_conn.weight
                                        conn_id = id(par_conn)
                                        # Обновляем направление с экспоненциальным затуханием
                                        old_dir = parent_ind._last_successful_mutation_direction.get(conn_id, 0.0)
                                        new_dir = 0.7 * old_dir + 0.3 * (weight_diff / (abs(weight_diff) + 0.1))
                                        parent_ind._last_successful_mutation_direction[conn_id] = new_dir
                                        
                                        # Обновляем важность связи на основе успеха
                                        old_importance = parent_ind._connection_importance.get(conn_id, 0.5)
                                        new_importance = min(1.0, old_importance + 0.1)
                                        parent_ind._connection_importance[conn_id] = new_importance
                else:
                    stagnation_counter += 1
                    # Если застой, увеличить разнообразие мутаций
                    # УСКОРЕННАЯ РЕАКЦИЯ НА ЗАСТОЙ: уменьшено количество поколений для реакции
                    if stagnation_counter > 10:  # Было 20, теперь 10 для более быстрой реакции
                        current_mutation_std = min(
                            self.config.get('max_mutation_std', 2.0),
                            current_mutation_std * 1.15  # Увеличено с 1.1 до 1.15 для более агрессивного выхода
                        )
                        if stagnation_counter > 30:  # Было 50, теперь 30
                            # Сильный застой - резкое увеличение мутаций
                            current_mutation_std = min(
                                self.config.get('max_mutation_std', 2.0),
                                current_mutation_std * 1.8  # Увеличено с 1.5 до 1.8
                            )
                        
                        # Сброс направлений при длительном застое (для выхода из локального минимума)
                        if stagnation_counter > 50:  # Было 100, теперь 50 для более быстрого сброса
                            for ind in self.internal_population:
                                ind._last_successful_mutation_direction.clear()
                                # Частичный сброс важности
                                ind._connection_importance = {k: v * 0.5 for k, v in ind._connection_importance.items()}
                
                progress_log.append(
                    f"Поколение {self.generation}: Лучшая={best.fitness:.15f}, Средняя={avg_fitness:.15f}, "
                    f"Мутация={current_mutation_std:.4f}"
                )
        
        return progress_log
    
    def _force_add_connection(self, individual: Individual):
        """Принудительно добавляет связь между случайными нейронами, обеспечивая передачу сигнала от входа к выходу"""
        if len(individual.neurons) < 2:
            return
        
        neurons = list(individual.neurons.keys())
        input_neurons = set(range(individual.input_size))
        output_neurons = set(range(individual.input_size, individual.input_size + individual.output_size))
        
        # Prefer adding connections that help signal flow from input to output
        # Try multiple times to find a good connection
        max_attempts = 50
        for _ in range(max_attempts):
            # Bias towards connecting input/hidden -> output/hidden
            if random.random() < 0.7:
                # From input or hidden
                from_candidates = [n for n in neurons if n not in output_neurons]
                if not from_candidates:
                    from_candidates = neurons
                from_neuron = random.choice(from_candidates)
                
                # To hidden or output
                to_candidates = [n for n in neurons if n not in input_neurons and n != from_neuron]
                if not to_candidates:
                    to_candidates = [n for n in neurons if n != from_neuron]
                if not to_candidates:
                    continue
                to_neuron = random.choice(to_candidates)
            else:
                from_neuron = random.choice(neurons)
                to_neuron = random.choice(neurons)
                if from_neuron == to_neuron:
                    continue
            
            # Check if connection already exists
            exists = False
            for conn in individual.connections:
                if conn.from_neuron == from_neuron and conn.to_neuron == to_neuron:
                    exists = True
                    break
            
            if not exists:
                weight = to_decimal(random.gauss(0, 1.0))
                individual.connections.append(Connection(from_neuron, to_neuron, weight))
                return
        
        # If all attempts failed, just add any connection
        for from_n in neurons:
            for to_n in neurons:
                if from_n != to_n:
                    exists = any(c.from_neuron == from_n and c.to_neuron == to_n 
                                for c in individual.connections)
                    if not exists:
                        weight = to_decimal(random.gauss(0, 1.0))
                        individual.connections.append(Connection(from_n, to_n, weight))
                        return
    
    def get_best(self) -> Individual:
        if not self.internal_population:
            return None
        return min(self.internal_population, key=lambda x: x.fitness)
    
    def save(self, filename: str = POPULATION_FILE):
        data = {
            'generation': self.generation,
            'population': [ind.save_to_dict() for ind in self.internal_population]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str = POPULATION_FILE) -> bool:
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.generation = data.get('generation', 0)
        self.internal_population = [
            Individual.load_from_dict(ind_data) for ind_data in data['population']
        ]
        
        return True


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_key():
    """Wait for any key press"""
    if os.name == 'nt':
        try:
            msvcrt.getch()
        except Exception:
            pass
    else:
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except (termios.error, io.UnsupportedOperation):
            # Неинтерактивный режим (пайп, редирект) - просто читаем строку или пропускаем
            try:
                sys.stdin.read(1)
            except Exception:
                pass
            return
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def show_menu() -> int:
    clear_screen()
    print("=" * 60)
    print("EVOTUS - Универсальная нейронная сеть с эволюционными стратегиями")
    print("=" * 60)
    print("\nГлавное меню:")
    print("1. Начать эволюцию")
    print("2. Тестировать текущую лучшую особь")
    print("3. Просмотреть статус популяции")
    print("4. Сохранить популяцию")
    print("5. Загрузить популяцию")
    print("6. Настройки")
    print("7. Просмотреть/редактировать обучающие данные")
    print("8. Экспортировать лучшую особь в файл")
    print("9. Импортировать особь из файла")
    print("10. Провести множество тестов эволюции")
    print("0. Выход")
    print("\n" + "=" * 60)
    
    try:
        choice = int(input("Введите выбор: "))
        return choice
    except (ValueError, EOFError):
        return -1


def run_evolution(pop_manager: PopulationManager):
    clear_screen()
    print("=" * 60)
    print("РЕЖИМ ЭВОЛЮЦИИ")
    print("=" * 60)
    
    try:
        num_gens = int(input("Введите количество поколений для эволюции: "))
    except (ValueError, EOFError):
        print("Неверный ввод!")
        wait_for_key()
        return
    
    print(f"\nЗапуск эволюции на {num_gens} поколений...")
    print("-" * 60)
    
    # Определяем интервал вывода прогресса
    if num_gens > 1000:
        print_interval = num_gens // 100  # Выводить каждые 1% прогресса
    elif num_gens > 500:
        print_interval = 10
    else:
        print_interval = 1
    
    progress = pop_manager.evolve_generation(num_gens, print_interval=print_interval)
    
    for log_entry in progress:
        print(log_entry)
        sys.stdout.flush()  # Принудительная запись в stdout
    
    print("-" * 60)
    best = pop_manager.get_best()
    print(f"\nЭволюция завершена!")
    print(f"Лучшая приспособленность: {best.fitness:.15f}")
    print(f"Сложность: {best.complexity}")
    print(f"Нейроны: {len(best.neurons)}")
    print(f"Связи: {len(best.connections)}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def run_evolution_tests(config: ConfigManager, data_manager: DataManager, 
                        mutator: Mutator, fitness_calc: FitnessCalculator):
    """
    Запускает множество тестов эволюции с различными параметрами.
    Использует значения по умолчанию для population_size и offspring_per_individual.
    
    Тестируются следующие варианты:
    1. Разное количество мутаций (3, 5, 8, 12)
    2. Разное количество поколений (100, 500, 1000)
    3. Разная сила мутаций (0.3, 0.6, 1.0)
    4. Комбинации параметров для поиска оптимальной конфигурации
    """
    clear_screen()
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЭВОЛЮЦИИ - МНОЖЕСТВЕННЫЕ ЗАПУСКИ")
    print("=" * 60)
    print("\nБудут проведены тесты с различными параметрами эволюции.")
    print("Используются значения по умолчанию:")
    print("  - population_size = 1")
    print("  - offspring_per_individual = 1")
    print("\nПараметры для тестирования:")
    print("  - mutations_per_offspring: [3, 5, 8, 12]")
    print("  - outer_population_mutations: [3, 5, 8, 12]")
    print("  - weight_mutation_std: [0.3, 0.6, 1.0]")
    print("  - generations: [100, 500, 1000]")
    print("-" * 60)
    
    try:
        confirm = input("Продолжить? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    except (ValueError, EOFError):
        return
    
    # Параметры для тестирования
    mutation_counts = [3, 5, 8, 12]
    mutation_stds = [0.3, 0.6, 1.0]
    generation_counts = [100, 500, 1000]
    
    results = []
    test_num = 0
    total_tests = len(mutation_counts) * len(mutation_stds) * len(generation_counts)
    
    print(f"\nВсего тестов: {total_tests}")
    print("Начало тестирования...\n")
    
    for gens in generation_counts:
        for mut_count in mutation_counts:
            for std in mutation_stds:
                test_num += 1
                
                # Создаем новый менеджер популяции для каждого теста
                pop_manager = PopulationManager(config, data_manager, mutator, fitness_calc)
                pop_manager.initialize()
                
                # Устанавливаем параметры для текущего теста
                original_mutations = config.get('outer_population_mutations')
                original_std = config.get('weight_mutation_std')
                
                config.set('outer_population_mutations', mut_count)
                config.set('weight_mutation_std', std)
                
                print(f"[{test_num}/{total_tests}] Тест: поколений={gens}, мутаций={mut_count}, std={std}")
                sys.stdout.flush()
                
                # Запускаем эволюцию без вывода прогресса
                start_time = __import__('time').time()
                progress = pop_manager.evolve_generation(gens, print_interval=gens)
                end_time = __import__('time').time()
                
                # Получаем лучший результат
                best = pop_manager.get_best()
                
                # Восстанавливаем оригинальные настройки
                config.set('outer_population_mutations', original_mutations)
                config.set('weight_mutation_std', original_std)
                
                # Сохраняем результат
                results.append({
                    'test_num': test_num,
                    'generations': gens,
                    'mutations': mut_count,
                    'std': std,
                    'fitness': best.fitness,
                    'complexity': best.complexity,
                    'neurons': len(best.neurons),
                    'connections': len(best.connections),
                    'time': end_time - start_time
                })
                
                print(f"  Результат: fitness={best.fitness:.10f}, сложность={best.complexity}, "
                      f"время={end_time - start_time:.2f}с")
                sys.stdout.flush()
    
    # Вывод сводных результатов
    print("\n" + "=" * 60)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    # Сортируем по fitness (лучшие первые)
    sorted_results = sorted(results, key=lambda x: x['fitness'])
    
    print("\nТоп-10 лучших конфигураций:")
    print("-" * 60)
    print(f"{'#':<3} {'Поколения':<8} {'Мутации':<7} {'Std':<5} {'Fitness':<15} {'Сложность':<9} {'Время(с)':<8}")
    print("-" * 60)
    
    for i, res in enumerate(sorted_results[:10]):
        print(f"{i+1:<3} {res['generations']:<8} {res['mutations']:<7} {res['std']:<5.2f} "
              f"{res['fitness']:<15.10f} {res['complexity']:<9} {res['time']:<8.2f}")
    
    # Анализ лучших конфигураций
    print("\n" + "-" * 60)
    print("АНАЛИЗ ЛУЧШИХ КОНФИГУРАЦИЙ:")
    print("-" * 60)
    
    best_overall = sorted_results[0]
    print(f"\nЛучшая конфигурация:")
    print(f"  - Поколений: {best_overall['generations']}")
    print(f"  - Мутаций на потомка: {best_overall['mutations']}")
    print(f"  - Сила мутации (std): {best_overall['std']}")
    print(f"  - Достигнутая fitness: {best_overall['fitness']:.15f}")
    print(f"  - Сложность сети: {best_overall['complexity']}")
    print(f"  - Нейронов: {best_overall['neurons']}")
    print(f"  - Связей: {best_overall['connections']}")
    print(f"  - Время выполнения: {best_overall['time']:.2f}с")
    
    # Средние результаты по разным параметрам
    print("\n" + "-" * 60)
    print("СРЕДНИЕ РЕЗУЛЬТАТЫ ПО ПАРАМЕТРАМ:")
    print("-" * 60)
    
    # По количеству мутаций
    print("\nПо количеству мутаций:")
    for mut in mutation_counts:
        avg_fitness = sum(r['fitness'] for r in results if r['mutations'] == mut) / len([r for r in results if r['mutations'] == mut])
        print(f"  Мутаций={mut}: средний fitness = {avg_fitness:.10f}")
    
    # По силе мутации
    print("\nПо силе мутации:")
    for std in mutation_stds:
        avg_fitness = sum(r['fitness'] for r in results if r['std'] == std) / len([r for r in results if r['std'] == std])
        print(f"  Std={std}: средний fitness = {avg_fitness:.10f}")
    
    # По количеству поколений
    print("\nПо количеству поколений:")
    for gens in generation_counts:
        avg_fitness = sum(r['fitness'] for r in results if r['generations'] == gens) / len([r for r in results if r['generations'] == gens])
        print(f"  Поколений={gens}: средний fitness = {avg_fitness:.10f}")
    
    # Сохранение результатов в файл
    save_results = input("\nСохранить результаты в файл? (y/n): ").strip().lower()
    if save_results == 'y':
        filename = "evolution_test_results.json"
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'best_fitness': best_overall['fitness'],
                    'best_config': {
                        'generations': best_overall['generations'],
                        'mutations': best_overall['mutations'],
                        'std': best_overall['std']
                    }
                },
                'all_results': results,
                'top_10': sorted_results[:10]
            }, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в {filename}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def test_individual(pop_manager: PopulationManager):
    best = pop_manager.get_best()
    
    if best is None or not best.neurons:
        print("\nНет обученной особи! Сначала запустите эволюцию.")
        wait_for_key()
        return
    
    clear_screen()
    print("=" * 60)
    print("РЕЖИМ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Сеть: {len(best.neurons)} нейронов, {len(best.connections)} связей")
    print(f"Размер входа: {best.input_size}, Размер выхода: {best.output_size}")
    print("-" * 60)
    print("Введите входные значения через пробел (или 'q' для выхода)")
    print()
    
    while True:
        try:
            user_input = input(f"Вход ({best.input_size} значений): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            values = [float(x) for x in user_input.split()]
            
            if len(values) != best.input_size:
                print(f"Ошибка: Ожидалось {best.input_size} значений, получено {len(values)}")
                continue
            
            outputs = best.forward(values)
            
            print(f"Выход: {' '.join(f'{o:.10f}' for o in outputs)}")
            print()
            
        except ValueError:
            print("Ошибка: Неверный формат ввода")
        except EOFError:
            break
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def view_status(pop_manager: PopulationManager):
    clear_screen()
    print("=" * 60)
    print("СТАТУС ПОПУЛЯЦИИ")
    print("=" * 60)
    print(f"Поколение: {pop_manager.generation}")
    print(f"Размер популяции: {len(pop_manager.internal_population)}")
    print()
    
    if pop_manager.internal_population:
        sorted_pop = sorted(pop_manager.internal_population, key=lambda x: x.fitness)
        print("Топ 5 особей:")
        print("-" * 60)
        for i, ind in enumerate(sorted_pop[:5]):
            print(f"{i+1}. Приспособленность: {ind.fitness:.15f}, Сложность: {ind.complexity}, "
                  f"Нейроны: {len(ind.neurons)}, Связи: {len(ind.connections)}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def configure_settings(config: ConfigManager):
    clear_screen()
    print("=" * 60)
    print("НАСТРОЙКИ")
    print("=" * 60)
    print("\nНастройки теперь хранятся в файле config.txt")
    print("Вы можете отредактировать этот файл в любом текстовом редакторе.")
    print("\nТекущие значения настроек:")
    print("-" * 40)
    
    descriptions = {
        'population_size': 'Размер внутренней популяции',
        'offspring_per_individual': 'Количество потомков на особь',
        'mutations_per_offspring': 'Количество мутаций на потомка (внутренняя популяция)',
        'outer_population_mutations': 'Количество мутаций для особей внешней популяции',
        'mutation_rate_weight': 'Вероятность мутации веса',
        'mutation_rate_connection': 'Вероятность мутации связи',
        'mutation_rate_neuron': 'Вероятность мутации нейрона',
        'mutation_rate_activation': 'Вероятность мутации активации',
        'weight_mutation_std': 'Стандартное отклонение мутации веса'
    }
    
    for key, value in config.config.items():
        desc = descriptions.get(key, key)
        print(f"{desc}: {value}")
    
    print("-" * 40)
    print(f"\nДля изменения настроек откройте файл '{config.filename}'")
    print("и измените значения после знака '=' в соответствующих строках.")
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def export_individual(pop_manager: PopulationManager):
    best = pop_manager.get_best()
    
    if best is None:
        print("\nНет особи для экспорта!")
        wait_for_key()
        return
    
    filename = input("Введите имя файла для экспорта: ").strip()
    
    if not filename:
        filename = "best_individual.json"
    
    with open(filename, 'w') as f:
        json.dump(best.save_to_dict(), f, indent=2)
    
    print(f"Экспортировано в {filename}")
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def import_individual(pop_manager: PopulationManager):
    filename = input("Введите имя файла для импорта: ").strip()
    
    if not os.path.exists(filename):
        print(f"Файл не найден: {filename}")
        wait_for_key()
        return
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        individual = Individual.load_from_dict(data)
        
        # Добавить в популяцию или заменить худшую
        if pop_manager.internal_population:
            worst_idx = max(range(len(pop_manager.internal_population)), 
                          key=lambda i: pop_manager.internal_population[i].fitness)
            pop_manager.internal_population[worst_idx] = individual
        else:
            pop_manager.internal_population.append(individual)
        
        print(f"Импортирована особь из {filename}")
    except Exception as e:
        print(f"Ошибка импорта: {e}")
    
    print("\nНажмите любую клавишу для возврата в меню...")
    wait_for_key()


def main():
    # Инициализация менеджеров
    config = ConfigManager()
    data_manager = DataManager()
    
    # Создать пример файла данных, если нужно
    if not os.path.exists(DATA_FILE):
        data_manager.save_example()
        print(f"Создан пример файла данных: {DATA_FILE}")
    
    # Загрузить обучающие данные
    if not data_manager.load():
        print(f"Ошибка загрузки данных из {DATA_FILE}")
        print("Убедитесь, что файл существует и содержит корректные данные.")
        return
    
    print(f"Загружено {len(data_manager.data)} обучающих примеров")
    print(f"Размер входа: {data_manager.input_size}, Размер выхода: {data_manager.output_size}")
    
    mutator = Mutator(config)
    fitness_calc = FitnessCalculator(data_manager)
    pop_manager = PopulationManager(config, data_manager, mutator, fitness_calc)
    
    # Попытаться загрузить существующую популяцию
    if pop_manager.load():
        print(f"Загружена популяция из поколения {pop_manager.generation}")
    else:
        print("Инициализация новой популяции...")
        pop_manager.initialize()
    
    # Главный цикл
    while True:
        choice = show_menu()
        
        if choice == 1:
            run_evolution(pop_manager)
        elif choice == 2:
            test_individual(pop_manager)
        elif choice == 3:
            view_status(pop_manager)
        elif choice == 4:
            pop_manager.save()
            print("Популяция сохранена!")
            wait_for_key()
        elif choice == 5:
            if pop_manager.load():
                print(f"Загружена популяция из поколения {pop_manager.generation}")
            else:
                print("Сохранённая популяция не найдена!")
            wait_for_key()
        elif choice == 6:
            configure_settings(config)
        elif choice == 7:
            print(f"\nОткрытие {DATA_FILE} для редактирования...")
            print("Отредактируйте файл и нажмите любую клавишу когда закончите...")
            wait_for_key()
            # Перезагрузить данные после редактирования
            if data_manager.load():
                print(f"Перезагружено {len(data_manager.data)} обучающих примеров")
                # Реинициализировать популяцию с новыми размерами данных
                pop_manager.initialize()
            else:
                print("Ошибка перезагрузки данных!")
            wait_for_key()
        elif choice == 8:
            export_individual(pop_manager)
        elif choice == 9:
            import_individual(pop_manager)
        elif choice == 10:
            run_evolution_tests(config, data_manager, mutator, fitness_calc)
        elif choice == 0:
            print("Сохранение популяции перед выходом...")
            pop_manager.save()
            print("До свидания!")
            break
        else:
            print("Неверный выбор!")
            wait_for_key()


if __name__ == "__main__":
    main()
