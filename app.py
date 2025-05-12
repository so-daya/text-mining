# app.py
import streamlit as st
import MeCab
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import re
import os
import numpy as np
import japanize_matplotlib # Matplotlibの日本語化のため (requirements.txt にも追加)
from itertools import combinations # 共起ネットワークで使用
from IPython.core.display import HTML # PyvisのHTML表示に使う場合があるが、Streamlitではst.components.v1.html

# --- 定数定義 ---
# Streamlit Cloud環境でのMeCabと辞書、フォントの標準的なパス
MECABRC_PATH = "/etc/mecabrc"
DICTIONARY_PATH = "/var/lib/mecab/dic/ipadic-utf8"
# packages.txtでfonts-ipafont-gothicをインストールした場合のIPA Pゴシックのパス
FONT_PATH_PRIMARY = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf' 
TAGGER_OPTIONS = f"-r {MECABRC_PATH} -d {DICTIONARY_PATH}"

# --- MeCab Taggerの初期化 ---
@st.cache_resource # Taggerの初期化はリソースを消費するのでキャッシュする
def initialize_mecab_tagger():
    try:
        tagger_obj = MeCab.Tagger(TAGGER_OPTIONS)
        tagger_obj.parse('') # 初期化のおまじない (BOM対策にもなる)
        st.session_state['mecab_tagger_initialized'] = True
        return tagger_obj
    except Exception as e_init:
        st.error(f"MeCab Taggerの初期化に失敗しました: {e_init}")
        st.error("リポジトリに `packages.txt` が正しく設定され、MeCab関連パッケージ (mecab, mecab-ipadic-utf8, libmecab-dev) がインストールされるか確認してください。")
        st.session_state['mecab_tagger_initialized'] = False
        return None

tagger = initialize_mecab_tagger()

# app.py のフォントパス決定部分の修正案

# ... (MECABRC_PATH, DICTIONARY_PATH, TAGGER_OPTIONS, initialize_mecab_tagger() はそのまま) ...
tagger = initialize_mecab_tagger() # 先にMeCabを初期化

# --- フォントパスの最終決定 ---
FONT_PATH_FINAL = None
FONT_PATH_PRIMARY = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'

if os.path.exists(FONT_PATH_PRIMARY):
    FONT_PATH_FINAL = FONT_PATH_PRIMARY
    st.info(f"日本語フォントとして '{FONT_PATH_FINAL}' を使用します。")
else:
    st.warning(f"指定されたIPAフォント '{FONT_PATH_PRIMARY}' が見つかりません。japanize_matplotlibのフォントで代替を試みます。")
    try:
        # japanize_matplotlib のインポートとフォントパス取得をここで行う
        import japanize_matplotlib 
        alt_font_path = japanize_matplotlib.get_font_path()
        if os.path.exists(alt_font_path):
            FONT_PATH_FINAL = alt_font_path
            st.info(f"japanize_matplotlibによる代替フォントとして '{FONT_PATH_FINAL}' を使用します。")
        else:
            st.error("japanize_matplotlibの代替フォントも見つかりませんでした。")
    except ImportError as e_import_jm: # japanize_matplotlib のインポート自体が失敗する場合
        st.error(f"japanize_matplotlibのインポートに失敗しました: {e_import_jm}")
        st.error("`requirements.txt` に `japanize-matplotlib` が正しく記述されているか確認してください。")
    except Exception as e_font: # get_font_path() で他のエラーが出る場合
        st.error(f"japanize_matplotlibからのフォントパス取得中にエラーが発生しました: {e_font}")

if FONT_PATH_FINAL is None:
    st.error("有効な日本語フォントが見つからないため、ワードクラウドやグラフの日本語表示に問題が出る可能性があります。")
else:
    # matplotlibの日本語設定 (japanize_matplotlibをインポートしただけでは適用されない場合があるため明示的に)
    # ただし、japanize_matplotlibをインポートするだけでrcParamsが更新されるはずなので、
    # ここでの plt.rcParams の設定は不要な場合も多い。
    # import japanize_matplotlib を実行した時点で日本語化は試みられる。
    pass

# ... (以降の分析関数の定義やStreamlit UIの定義は続く) ...
# WordCloudやmatplotlibの描画関数に渡すfont_pathは FONT_PATH_FINAL を使う

# --- 分析関数の定義 ---

def perform_morphological_analysis(text_input, tagger_instance):
    if tagger_instance is None or not text_input: 
        return []
    all_morphemes = []
    node = tagger_instance.parseToNode(text_input)
    while node:
        if node.surface: # BOS/EOSノードはsurfaceが空なので除外
            features = node.feature.split(',')
            all_morphemes.append({
                '表層形': node.surface,
                '原形': features[6] if features[6] != '*' else node.surface,
                '品詞': features[0],
                '品詞細分類1': features[1],
                '品詞細分類2': features[2],
                '品詞細分類3': features[3],
                '活用型': features[4],
                '活用形': features[5],
                '読み': features[7] if len(features) > 7 and features[7] != '*' else '',
                '発音': features[8] if len(features) > 8 and features[8] != '*' else ''
            })
        node = node.next
    return all_morphemes

def generate_word_report(all_morphemes, target_pos_list, stop_words_set):
    if not all_morphemes: 
        return pd.DataFrame()
    
    report_target_morphemes = []
    for m in all_morphemes:
        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:
            # 名詞の場合の追加フィルタリング
            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['非自立', '数', '代名詞', '接尾', 'サ変接続', '副詞可能']:
                continue
            report_target_morphemes.append(m)

    if not report_target_morphemes: 
        return pd.DataFrame()
        
    word_counts = Counter(m['原形'] for m in report_target_morphemes)
    report_data = []
    
    # 原形ごとに最初の出現時の詳細情報を代表として保持
    representative_info_for_report = {}
    for m in reversed(report_target_morphemes): # 後ろから見て最初に見つかったものを採用（特に意味はないが一貫性のため）
        if m['原形'] not in representative_info_for_report:
            representative_info_for_report[m['原形']] = m
            
    total_all_morphemes_count_for_freq = len(all_morphemes) # 頻度計算の母数は全形態素数

    for rank, (word, count) in enumerate(word_counts.most_common(), 1):
        info = representative_info_for_report.get(word, {}) # 原形に対応する代表情報を取得
        frequency = (count / total_all_morphemes_count_for_freq) * 100 if total_all_morphemes_count_for_freq > 0 else 0
        report_data.append({
            '順位': rank,
            '単語 (原形)': word,
            '出現数': count,
            '出現頻度 (%)': round(frequency, 3), # 小数点3桁に
            '品詞': info.get('品詞', ''),
            '品詞細分類1': info.get('品詞細分類1', ''),
            '代表的な表層形': info.get('表層形', ''), # 代表的な表層形も追加
            '代表的な読み': info.get('読み', '')   # 代表的な読みも追加
        })
    return pd.DataFrame(report_data)

def generate_wordcloud_image(all_morphemes, font_path_wc, target_pos_list, stop_words_set):
    if not all_morphemes:
        st.info("ワードクラウド生成のための形態素データがありません。")
        return None
    if font_path_wc is None or not os.path.exists(font_path_wc):
        st.error(f"ワードクラウド生成に必要な日本語フォントパス '{font_path_wc}' が見つかりません。")
        return None
    
    wordcloud_words = []
    for m in all_morphemes:
        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:
            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['数', '非自立', '代名詞', '接尾']:
                continue
            wordcloud_words.append(m['原形']) # ワードクラウドは原形ベースが一般的
    
    wordcloud_text_input_str = " ".join(wordcloud_words)
    if not wordcloud_text_input_str.strip():
        st.info("ワードクラウド表示対象の単語が見つかりませんでした（フィルタリング後）。")
        return None

    try:
        wc = WordCloud(
            font_path=font_path_wc,
            background_color="white",
            width=800, height=400,
            max_words=200,
            collocations=False, 
            random_state=42 
        ).generate(wordcloud_text_input_str)
        
        fig, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        return fig
    except Exception as e_wc:
        st.error(f"ワードクラウド画像生成中にエラーが発生しました: {e_wc}")
        return None

def generate_cooccurrence_network_html(all_morphemes, text_input_co, tagger_instance, font_path_co, target_pos_list, stop_words_set, node_min_freq, edge_min_freq):
    if not all_morphemes or tagger_instance is None or not text_input_co.strip():
        st.info("共起ネットワーク生成に必要なデータが不足しています。")
        return None
    if font_path_co is None or not os.path.exists(font_path_co):
        st.error(f"共起ネットワークのラベル表示に必要な日本語フォントパス '{font_path_co}' が見つかりません。")
        return None

    temp_words_for_nodes = []
    for m in all_morphemes:
        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:
            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['非自立', '数', '代名詞', '接尾', 'サ変接続', '副詞可能']:
                continue
            if len(m['原形']) < 2 and m['品詞'] != '名詞': # 名詞以外の1文字の原形は除外
                continue
            temp_words_for_nodes.append(m['原形'])

    word_counts = Counter(temp_words_for_nodes)
    node_candidates = {word: count for word, count in word_counts.items() if count >= node_min_freq}

    if len(node_candidates) < 2:
        st.info(f"共起ネットワークのノードとなる単語（フィルタ後）が2つ未満です。")
        return None

    sentences = re.split(r'[。\n！？]+', text_input_co)
    sentences = [s.strip() for s in sentences if s.strip()]
    cooccurrence_counts_map = Counter()
    for sentence in sentences:
        node_s = tagger_instance.parseToNode(sentence)
        words_in_sentence = []
        while node_s:
            if node_s.surface:
                features_s = node_s.feature.split(',')
                original_form_s = features_s[6] if features_s[6] != '*' else node_s.surface
                if original_form_s in node_candidates:
                    words_in_sentence.append(original_form_s)
            node_s = node_s.next
        for pair in combinations(sorted(list(set(words_in_sentence))), 2):
            cooccurrence_counts_map[pair] += 1
    
    if not cooccurrence_counts_map:
        st.info("共起ペアが見つかりませんでした。")
        return None

    font_name_for_pyvis_graph = os.path.splitext(os.path.basename(font_path_co))[0]
    if font_name_for_pyvis_graph.lower() == 'ipagp': font_name_for_pyvis_graph = 'IPAPGothic'
    elif font_name_for_pyvis_graph.lower() == 'ipamp': font_name_for_pyvis_graph = 'IPAPMincho'
    
    net_graph = Network(notebook=True, height="750px", width="100%", directed=False, bgcolor="#F5F5F5", font_color="#333333")
    for word, count in node_candidates.items():
        node_s = int(np.sqrt(count) * 10 + 10)
        net_graph.add_node(word, label=word, size=node_s, title=f"{word} (出現数: {count})",
                           font={'face': font_name_for_pyvis_graph, 'size': 14, 'color': '#333333'},
                           borderWidth=1, color={'border': '#666666', 'background': '#D2E5FF'})
    
    added_edge_num = 0
    for pair_nodes, freq_cooc in cooccurrence_counts_map.items():
        if freq_cooc >= edge_min_freq:
            edge_w = float(np.log1p(freq_cooc) * 1.5 + 0.5)
            net_graph.add_edge(pair_nodes[0], pair_nodes[1], value=edge_w, title=f"共起: {freq_cooc}回",
                               color={'color': '#cccccc', 'highlight': '#848484', 'opacity':0.6})
            added_edge_num +=1
    
    if added_edge_num == 0:
        st.info(f"表示対象の共起ペア（共起回数 {edge_min_freq} 回以上）がありませんでした。")
        return None

    options_js = """ var options = {"interaction": {"navigationButtons": false, "keyboard": {"enabled": false}}, "manipulation": {"enabled": false}, "configure": {"enabled": false}, "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -30000, "centralGravity": 0.1, "springLength": 150, "springConstant": 0.03, "damping": 0.09, "avoidOverlap": 0.5}, "solver": "barnesHut", "stabilization": {"iterations": 500}}}; """
    try:
        net_graph.set_options(options_js)
    except Exception as e_set_opt:
        st.warning(f"Pyvisオプション設定で軽微なエラー: {e_set_opt}")
    net_graph.show_buttons(filter_=False)
    
    # HTML文字列を直接取得
    html_content = net_graph.generate_html(name="temp_cooc_net_streamlit.html", notebook=True)
    return html_content


def perform_kwic_search(all_morphemes, keyword_str, search_key_type_str, window_int):
    if not keyword_str.strip() or not all_morphemes: 
        return []
    kwic_results_data = []
    for i, morpheme_item in enumerate(all_morphemes):
        if morpheme_item[search_key_type_str] == keyword_str:
            left_start_idx = max(0, i - window_int)
            left_ctx_str = "".join(m['表層形'] for m in all_morphemes[left_start_idx:i])
            kw_surface = morpheme_item['表層形']
            right_end_idx = min(len(all_morphemes), i + 1 + window_int)
            right_ctx_str = "".join(m['表層形'] for m in all_morphemes[i+1:right_end_idx])
            kwic_results_data.append({'左文脈': left_ctx_str, 'キーワード': kw_surface, '右文脈': right_ctx_str})
    return kwic_results_data

# --- Streamlit UIの定義 ---
st.set_page_config(layout="wide", page_title="簡易テキストマイニングツール")
st.title("テキストマイニングツール (Streamlit版)")
st.markdown("日本語テキストを入力して、形態素解析、単語レポート、ワードクラウド、共起ネットワーク、KWIC検索を実行します。")

# --- デフォルトストップワード ---
DEFAULT_STOP_WORDS_SET = {
    "する", "ある", "いる", "なる", "いう", "できる", "思う", "やる", "ない", "よい", "良い",
    "大きい", "小さい", "高い", "低い", "嬉しい", "楽しい", "悲しい", "同じ", "様々",
    "こと", "もの", "ため", "よう", "的", "的だ", "とき", "ところ", "ほう", "なか", "うち",
    "私", "あなた", "彼", "彼女", "これ", "それ", "あれ", "ここ", "そこ", "あそこ", "方", "為",
    "非常", "大変", "少し", "かなり", "色々", "いつも", "いただく", "/", ":", "れる", "\""
}

# --- サイドバー: オプション設定 ---
st.sidebar.header("⚙️ 分析オプション")
st.sidebar.markdown("**全般設定**")
report_target_pos = st.sidebar.multiselect("単語レポート: 対象品詞", ['名詞', '動詞', '形容詞', '副詞', '感動詞', '連体詞', '接続詞', '助詞', '助動詞', '記号', 'その他'], default=['名詞', '動詞', '形容詞', '副詞'])
wc_target_pos = st.sidebar.multiselect("ワードクラウド: 対象品詞", ['名詞', '動詞', '形容詞', '副詞', '感動詞'], default=['名詞', '動詞', '形容詞'])
net_target_pos = st.sidebar.multiselect("共起Net: 対象品詞", ['名詞', '動詞', '形容詞', '副詞'], default=['名詞', '動詞', '形容詞'])

custom_stopwords_str = st.sidebar.text_area("共通ストップワード (原形をカンマや改行区切りで追加):", 
                                         value="サンプル, デモ, テキスト\nマイニング, 解析, 機能") # 改行も区切り文字として処理

# ストップワードの最終決定
final_stop_words = DEFAULT_STOP_WORDS_SET.copy()
if custom_stopwords_str.strip():
    custom_list = [word.strip() for word in re.split(r'[,\n]', custom_stopwords_str) if word.strip()]
    final_stop_words.update(custom_list)
st.sidebar.caption(f"適用される総ストップワード数: {len(final_stop_words)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**共起ネットワーク設定**")
network_node_min_freq = st.sidebar.slider("ノード最低出現数:", 1, 10, 2, key="net_node_freq_slider")
network_edge_min_freq = st.sidebar.slider("エッジ最低共起数:", 1, 10, 2, key="net_edge_freq_slider")


# --- メイン画面: テキスト入力と実行ボタン ---
main_text_input = st.text_area("📝 分析したい日本語テキストを入力してください:", height=250, 
                             value="これはStreamlitを使用して作成したテキスト分析ツールです。日本語の形態素解析を行い、単語の出現頻度レポート、ワードクラウド、共起ネットワーク、そしてKWIC（文脈付きキーワード検索）などを試すことができます。様々な文章で分析を実行してみてください。")

analyze_button = st.button("分析実行", type="primary")

# --- 分析結果表示エリア ---
if analyze_button:
    if not main_text_input.strip():
        st.warning("分析するテキストを入力してください。")
    elif tagger is None or not st.session_state.get('mecab_tagger_initialized', False):
        st.error("MeCab Taggerが利用できません。ページを再読み込みするか、管理者にお問い合わせください。")
    else:
        with st.spinner("形態素解析を実行中... しばらくお待ちください。"):
            morphemes_result = perform_morphological_analysis(main_text_input, tagger)
        
        if not morphemes_result:
            st.error("形態素解析に失敗したか、結果が空です。入力テキストを確認してください。")
        else:
            st.success(f"形態素解析が完了しました。総形態素数: {len(morphemes_result)}")
            st.markdown("---")

            # タブで各分析結果を表示
            tab_report, tab_wc, tab_network, tab_kwic = st.tabs(["📊 単語出現レポート", "☁️ ワードクラウド", "🕸️ 共起ネットワーク", "🔍 KWIC検索"])

            with tab_report:
                st.subheader("単語出現レポート")
                with st.spinner("単語出現レポートを作成中..."):
                    df_report_result = generate_word_report(morphemes_result, report_target_pos, final_stop_words)
                    if not df_report_result.empty:
                        st.dataframe(df_report_result.style.bar(subset=['出現数'], align='left', color='#90EE90')
                                     .format({'出現頻度 (%)': "{:.3f}%"}))
                    else:
                        st.info("レポート対象の単語が見つかりませんでした（フィルタリング後）。")
            
            with tab_wc:
                st.subheader("ワードクラウド")
                if FONT_PATH_FINAL: # FONT_PATH_FINAL が None でないことを確認
                    with st.spinner("ワードクラウド生成中..."):
                        fig_wc_result = generate_wordcloud_image(morphemes_result, FONT_PATH_FINAL, wc_target_pos, final_stop_words)
                        if fig_wc_result:
                            st.pyplot(fig_wc_result)
                        # else: # 関数内でst.infoを出しているのでここでは不要
                        #    st.info("ワードクラウドは生成されませんでした。")
                    st.caption(f"使用フォント: {os.path.basename(FONT_PATH_FINAL) if FONT_PATH_FINAL else '未設定'}")
                else:
                    st.error("日本語フォントの準備ができていません。ワードクラウドは生成できません。")
            
            with tab_network:
                st.subheader("共起ネットワーク")
                if FONT_PATH_FINAL:
                    with st.spinner("共起ネットワーク生成中..."):
                        html_cooc_result = generate_cooccurrence_network_html(
                            morphemes_result, main_text_input, tagger, FONT_PATH_FINAL,
                            net_target_pos, final_stop_words,
                            network_node_min_freq, network_edge_min_freq
                        )
                        if html_cooc_result:
                            st.components.v1.html(html_cooc_result, height=750, scrolling=True)
                        # else: # 関数内でst.infoを出しているのでここでは不要
                        #    st.info("共起ネットワークは生成されませんでした。")
                    st.caption(f"使用フォント (ノードラベル): {os.path.basename(FONT_PATH_FINAL) if FONT_PATH_FINAL else '未設定'}")
                else:
                     st.error("日本語フォントの準備ができていません。共起ネットワークは生成できません。")
            
            with tab_kwic:
                st.subheader("KWIC検索 (文脈付きキーワード検索)")
                kwic_keyword_input = st.text_input("KWIC検索キーワード:", placeholder="検索したい単語(原形推奨)...", key="kwic_keyword_input")
                # 検索モードの選択肢を修正
                kwic_search_mode_options = ("原形一致", "表層形一致") 
                kwic_search_mode_selected = st.radio("KWIC検索モード:", kwic_search_mode_options, index=0, key="kwic_mode_radio")
                kwic_window_val = st.slider("KWIC表示文脈の形態素数 (前後各):", 1, 10, 5, key="kwic_window_slider")

                if kwic_keyword_input.strip():
                    search_key_type_for_kwic = '原形' if kwic_search_mode_selected == "原形一致" else '表層形'
                    with st.spinner(f"「{kwic_keyword_input}」を検索中..."):
                        results_kwic_list = perform_kwic_search(morphemes_result, kwic_keyword_input.strip(), search_key_type_for_kwic, kwic_window_val)
                    if results_kwic_list:
                        st.write(f"「{kwic_keyword_input}」の検索結果 ({len(results_kwic_list)}件):")
                        df_kwic_to_display = pd.DataFrame(results_kwic_list)
                        st.dataframe(df_kwic_to_display)
                    else:
                        st.info(f"「{kwic_keyword_input}」は見つかりませんでした（現在の検索モードにおいて）。")

# --- フッター情報 (オプション) ---
st.sidebar.markdown("---")
st.sidebar.info("このWebアプリはStreamlitと各種Pythonライブラリを使用して作成されました。")
