import matplotlib as mpl
from types import SimpleNamespace
from cmap import Colormap
import matplotlib.pyplot as plt
import numpy as np


class Colors:
    defaults = SimpleNamespace(
        blue='#0072B2',
        vermilion='#D55E00',
        amber='#E69F00',
        green="#009E73",
        charcoal="#2B2D42",
        dark_gray="#333",
        gray="#555",
        face_gray="#444",
        outer_space="#414B4C",
        obs_edge="#447",
        obs_color="#446",
        navy_blue="#249",
        dust_red="#CE3E3E"
    )

    pastels = SimpleNamespace(
        blue='#7FB8D8',
        vermilion='#EAAE7F',
        amber='#F2CF7F',
        green="#7FBFA9",
        paleblue='#BFDBEB',
        palevermilion='#F4D6BF',
        paleamber='#F8E7BF',
        palegreen="#BFDFD4",
    )
    

    _seq = {
        "rocket": Colormap("seaborn:rocket").to_matplotlib(),
        "copper": Colormap("matlab:copper_r").to_matplotlib(),
        "mako": Colormap("seaborn:mako").to_matplotlib(),
        "bone": Colormap("matlab:bone").to_matplotlib()
    }
    _div = {
        "YlOrBr": Colormap("tol:YlOrBr").to_matplotlib(),
        "vik": Colormap("crameri:vik").to_matplotlib()
    }
    colormaps = SimpleNamespace(
        seq=SimpleNamespace(**_seq),
        div=SimpleNamespace(**_div)
    )



class Physicals:
    params = {
        "stellar.lum": {"label": r"L$_*$ [$10^{8}$L$_{\odot}$]", "cmap": Colors.colormaps.seq.bone, "vmin": 1.23e-07, "vmax": 7.75, 
                        "calc": lambda inst, name: inst.get_image(name, "bayes.stellar.lum") / 3.8e26 / 1e8},
        "stellar.mass_total": {"label": r"M$_*$ [$10^{8}$M$_{\odot}$]", "cmap": Colors.colormaps.seq.rocket, "vmin": 2.22e-08, "vmax": 8.5, 
                                "calc": lambda inst, name: inst.get_image(name, "bayes.stellar.mass_total") / 1e8},
        "stellar.m_star": {"label": r"M$_*$ [$10^{8}$M$_{\odot}$]", "cmap": Colors.colormaps.seq.rocket, "vmin": 2.22e-08, "vmax": 8.5, 
                                "calc": lambda inst, name: inst.get_image(name, "bayes.stellar.m_star") / 1e8},
        "attenuation.V": {"label": r"$A_{V}$ [mag]", "cmap": Colors.colormaps.seq.copper, "vmin": 0.1, "vmax": 0.5, 
                            "calc": lambda inst, name: inst.get_image(name, "bayes.attenuation.generic.bessell.V")},
        "sfh.age": {"label": r"Age [Gyr]", "cmap": Colors.colormaps.div.YlOrBr, "vmin": 9.4, "vmax": 10.0, 
                    "calc": lambda inst, name: inst.get_image(name, "bayes.sfh.age")/1e9},
        "stellar.metallicity": {"label": r"Z$_*$", "cmap": Colors.colormaps.div.vik, "vmin": 0.01, "vmax": 0.04, 
                                "calc": lambda inst, name: inst.get_image(name, "bayes.stellar.metallicity")},
        "sfh.sfr": {"label": r"SFR [M$_{\odot}$/yr]", "cmap": Colors.colormaps.seq.mako, "vmin": 0.0, "vmax": 0.07, 
                    "calc": lambda inst, name: inst.get_image(name, "bayes.sfh.sfr")}
    }


import matplotlib as mpl
import matplotlib.pyplot as plt

class PlotConfig:

    @staticmethod
    def set_paper_style(style: str="single", height: float=None):
        """
        AASTeX 및 일반 과학 논문용 통합 스타일 설정.
        논문 캡션 사용을 고려하여 피규어 타이틀 설정을 배제하고, 
        실제 저널 단(Column) 너비(인치)에 맞춘 표준 폰트 크기(9~10pt)로 전면 조정합니다.
        """

        # 1. 폰트 기본 설정 (LaTeX 논문 표준)
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['mathtext.fontset'] = 'dejavuserif'  # 수식 폰트도 이질감 없이 통일
        
        # 2. 크기 및 가독성 전역 설정 (실제 논문 인쇄 사이즈 기준)
        # 3.35 인치 도화지에서 10pt가 가장 안정적인 가독성을 보여줍니다.
        mpl.rcParams['axes.labelsize'] = 10    # x, y축 이름 크기
        mpl.rcParams['xtick.labelsize'] = 9    # x축 숫자 크기
        mpl.rcParams['ytick.labelsize'] = 9    # y축 숫자 크기
        mpl.rcParams['legend.fontsize'] = 9    # 범례 크기
        
        # 피규어 타이틀은 논문 캡션으로 대체되므로 크기를 축소하거나 무시합니다.
        mpl.rcParams['axes.titlesize'] = 10    
        
        # 3. 틱(Tick) 마크 스타일 설정 (천문학/물리학 논문 필수 규격)
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.top'] = True       # 박스 상단 틱 표시 (필수)
        mpl.rcParams['ytick.right'] = True     # 박스 우측 틱 표시 (필수)
        
        # 메이저/마이너 틱 세밀도 및 가시성 조정
        mpl.rcParams['xtick.major.size'] = 5
        mpl.rcParams['ytick.major.size'] = 5
        mpl.rcParams['xtick.minor.size'] = 3
        mpl.rcParams['ytick.minor.size'] = 3
        mpl.rcParams['xtick.minor.visible'] = True  # 마이너 틱 활성화 (AASTeX 권장)
        mpl.rcParams['ytick.minor.visible'] = True
        
        # 선 두께 (폰트가 작아진 만큼 축과 선이 너무 얇아 보이지 않도록 대비 강화)
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['lines.linewidth'] = 1.5
        
        # 4. 범례 테두리 제거 (공간 절약 및 깔끔함)
        mpl.rcParams['legend.frameon'] = False
        
        # 5. 저장 설정 (AASTeX 고해상도 벡터 그래픽스 기준)
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['savefig.format'] = 'pdf'

        # 6. 피규어 사이즈 조정
        # AASTeX의 1단(single) 너비는 약 3.35인치, 2단(double) 텍스트 전체 너비는 약 7.1인치입니다.
        if style == "single":
            height = float(height) if height is not None else 2.5
            plt.rcParams.update({
                'figure.figsize': [3.35, height]
            })
        elif style == "double":
            height = float(height) if height is not None else 3.0
            # 2단 너비에 맞추되, 높이가 너무 길어지지 않도록 비율 조정
            plt.rcParams.update({
                'figure.figsize': [7.1, height]
            })

    ylabels = {
        "metallicity": "Metallicity [Z/H]",
        "flux": "Flux [mJy]",
        "age": "Age [yr]",
        "luminosity": "Luminosity [W]",
        "mass": "Mass "+r"[$M_{\odot}$]",
        "attenuation": "Attenuation [mag]",
        "mag": "Magnitute [mag]",
        "sfr": "Star Formation Rate "+r"[$M_{\odot}$/yr]"
    }
    
    refer_values = {  # Pessa+2023, mw
        "metallicity": ([0.020388334700204777, 0.04271838734450855, 0.07766990291262141, 0.10194174757281563,
         0.13398053808119692, 0.17087377159340872, 0.19514561625360294, 0.22524268881788545,
         0.25533976138216785, 0.28543683394645036, 0.3165048395545737, 0.34174754318681744, 0.37378640776699035],
         [0.11641791633962217, 0.04000003036574462, 0.025671680541360598, 0.018507490446296404,
         0.0113433003512321, -0.009353201576684289, -0.025273597018387495, -0.04676613693783571,
         -0.06507458562264545, -0.06587059324843281, -0.048358182555154894, -0.05313428904136808, -0.051542243424048784]),
        "age": ([0.02038833470020477, 0.04368932038834949, 0.0786407618846708, 0.10970876749279426, 
         0.1359223300970874, 0.1689320536493098 , 0.19902912621359223, 0.2242718298458358, 0.2514563254939699, 
         0.2844660490461924, 0.3155339805825243, 0.338834966270669, 0.3708738308508419],
         [9.698327364722502, 9.772998806277833, 9.789426500632091, 9.772998806277833, 
         9.783452819489538, 9.798387107800604, 9.786439688545709, 9.76403825607911, 9.792413369688262, 
         9.780465950433367, 9.793906832701243, 9.793906832701243, 9.781959356476559])
    }
    
    # refer_values = {  # Pessa+2023, lw
    #     "metallicity": ([0.0188116, 0.0441923, 0.0776351, 0.102419, 0.137056, 0.166915, 0.196477, 0.225142, 0.256494, 0.285757, 0.315915, 0.344282, 0.374142],
    #                     [0.0775917, -0.0132526, -0.0555753, -0.105593, -0.143855, -0.163092, -0.193231, -0.20456, -0.209263, -0.201568, -0.202636, -0.208835, -0.1943]),
    #     "age": ([0.0233385, 0.0447142, 0.07775, 0.103188, 0.137588, 0.167886, 0.19679, 0.227095, 0.258804, 0.286354, 0.317347, 0.342127, 0.37589],
    #             [9.28961, 9.3641, 9.24297, 9.04577, 8.87185, 8.82058, 8.70411, 8.67923, 8.74905, 8.72572, 8.69619, 8.61698, 8.64022])
    # }
    
    refer_values2 = {  # Abdurr'uf+2022
        "metallicity": ([0.0153744 , 0.05386081, 0.0863314 , 0.12301231, 0.15787058,
        0.19275549, 0.22702969, 0.26371821, 0.29741594, 0.3323018 ,
        0.36839292, 0.40387713, 0.4387687 , 0.47485411, 0.50975804,
        0.54524035, 0.58072266, 0.60777007],
        [-0.26425, -0.26174, -0.24983, -0.24026, -0.15764, 
        -0.17871, -0.16208, -0.18551, -0.27259, -0.29838, 
        -0.33359, -0.34759, -0.39931, -0.41331, -0.50981, 
        -0.51674, -0.52603, -0.46936]),
        "age": ([0.01661843, 0.05223072, 0.08665435, 0.12226664, 0.1572846 ,
        0.19170918, 0.22910161, 0.2629328 , 0.2997309 , 0.33415453,
        0.36976587, 0.4041914 , 0.4380226 , 0.4730396 , 0.51043203,
        0.54545094, 0.5804689 , 0.60777007],
        [9.800920, 9.799171, 9.779935, 9.772940, 9.746709, 
        9.739715, 9.708237, 9.716981, 9.711735, 9.706488, 
        9.687252, 9.638288, 9.648780, 9.615554, 9.626047, 
        9.608559, 9.615554, 9.584077])
    }
    
    refer_values3 = {  # JH Lee+2025
        "metallicity": ([0.017426261805617755, 0.04959786139702584, 0.07908848583325864, 0.10455762196905453, 0.13404824640528729,
        0.1689007700165228, 0.19302949527775282, 0.22654155687907465, 0.26139408049031027, 0.28686326776145393,
        0.3163538410623389, 0.3418230283334826, 0.3780161162253276],
        [0.11659808867713506, 0.0419753077375441, 0.02880659646429118, 0.01783265993768568, 0.01234569167438293,
        -0.013991751803245633, -0.027160504938744112, -0.05020575489971757, -0.0666666596896257, -0.06556928278186347,
        -0.05130313180747981, -0.052400550577487714, -0.05349792748524995]),
        "age": ([0.018641780820595776, 0.04660452824384647, 0.07856188856268402, 0.10519304755916774, 0.13848204709967699,
        0.16644474372802295, 0.19973369247362757, 0.2237017762063866, 0.25565908573031954, 0.2836218839484748,
        0.3142476558405454, 0.34354199169056315, 0.3728362259507714],
        [9.697217702891784, 9.768821628196394, 9.777004848099198, 9.770867462437876, 9.777004848099198,
        9.791325726810621, 9.781096594624248, 9.762684125471944, 9.781096594624248, 9.774959091899799,
        9.791325726810621, 9.791325726810621, 9.779050760382766])
    }

    # 글로벌 특성 비교(Global Comparison) 플롯을 위한 참조값 정의
    global_refs = {
        "mass": [
            {
                "value_calc": lambda dist: 10 ** 10.84 * (dist / 11.32) ** 2,
                "label": "Leroy+2021", "color": Colors.defaults.dark_gray, "ls": "-"
            }
        ],
        "age": [
            {
                "value": 9.83, "span": [9.83 - 0.0414, 9.83 + 0.0414],
                "label": "Optical Spectra\n(Pessa+2023)", "color": Colors.defaults.dark_gray, "ls": "-"
            }
        ],
        "sfr": [
            {
                "value": 10 ** 0.58, "span": [10**0.58 - 0.88, 10**0.58 + 0.88],
                "label": "CO (Leroy+2021)", "color": Colors.defaults.dark_gray, "ls": "-"
            },
            {
                "value": 1.34 * 1.566, "span": [(1.34 - 0.12) * 1.566, (1.4 + 0.13) * 1.566],
                "label": "Broad-band piXedfit\n(Abdurro’uf+2022)", "color": Colors.defaults.navy_blue, "ls": "-."
            },
            {
                "value": None, "span": [0.3709, 6.9], "xmin": 7/8, "xmax": 1.0,
                "label": "Typical SFR", "color": Colors.defaults.navy_blue, "ls": "none"
            }
        ],
        "metallicity": [
            {
                "value": -0.01, "span": [-0.01 - 0.0334, -0.01 + 0.0334],
                "label": "Optical Spectra\n(Pessa+2023)", "color": Colors.defaults.dark_gray, "ls": "-"
            }
        ]
    }