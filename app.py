# app.py
import streamlit as st

# ページ設定は、他のどのStreamlitコマンドよりも先に、スクリプトの一番最初に呼び出す必要があります。
st.set_page_config(layout="wide", page_title="テキストマイニングツール (Streamlit版)")

# --- ライブラリのインポート ---
import MeCab
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # matplotlibのフォント管理
import networkx as nx
from pyvis.network import Network
import re
import os
import numpy as np
from itertools import combinations
# from IPython.core.display import HTML # Streamlitでは st.components.v1.html を使用

# --- 定数定義 ---
MECABRC_PATH = "/etc/mecabrc"
DICTIONARY_PATH = "/var/lib/mecab/dic/ipadic-utf8"
TAGGER_OPTIONS = f"-r {MECABRC_PATH} -d {DICTIONARY_PATH}"
FONT_PATH_PRIMARY = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf' 

# ★★★ ベースとなるストップワードリストを定義 ★★★
DEFAULT_STOP_WORDS_SET = {
    # 一般的な動詞・助動詞・形式名詞など (原形)
    "する", "ある", "いる", "なる", "いう", "できる", "思う", "やる", "ない", "よい", "良い",
    "いく", "来る", "おる", "ます", "です", "だ", "れる", "られる", "せる", "させる", "いただく",
    # 一般的な形式名詞・代名詞など
    "こと", "もの", "とき", "ところ", "ため", "よう", "うち", "ほう", "的", "的だ",
    "私", "あなた", "彼", "彼女", "これ", "それ", "あれ", "ここ", "そこ", "あそこ", "方", "為", "訳", "筈",
    # 一般的すぎる形容詞・副詞など
    "大きい", "小さい", "高い", "低い", "嬉しい", "楽しい", "悲しい", "同じ", "様々", "色々",
    "非常", "大変", "少し", "かなり", "いつも", "よく", "本当に", "ちょっと", "たくさん", "多く",
    # 記号類 (原形がそのまま記号になる場合が多い)
    "/", ":", "\"", ".", ",", "、", "。", " ", "　", # 半角・全角スペースも
    "(", ")", "[", "]", "（", "）", "「", "」", "【", "】",
    "&", "-", "_", "=", "+", "*", "%", "#", "@", "!", "?"
}

# --- MeCab Taggerの初期化とセッション状態への保存 ---
@st.cache_resource 
def initialize_mecab_tagger_to_session():
    # デバッグ情報表示 (必要に応じてコメントアウト)
    # st.sidebar.subheader("MeCab初期化デバッグ:")
    # mecabrc_exists = os.path.exists(MECABRC_PATH)
    # dicdir_exists = os.path.exists(DICTIONARY_PATH)
    # st.sidebar.text(f"mecabrc ({MECABRC_PATH}): {'存在する' if mecabrc_exists else '存在しない'}")
    # st.sidebar.text(f"辞書Dir ({DICTIONARY_PATH}): {'存在する' if dicdir_exists else '存在しない'}")
    # if dicdir_exists:
    #     try: st.sidebar.text(f"辞書Dir内容: {os.listdir(DICTIONARY_PATH)}")
    #     except Exception as e_ls: st.sidebar.text(f"辞書Dir内容取得エラー: {e_ls}")
    
    try:
        tagger_obj = MeCab.Tagger(TAGGER_OPTIONS)
        tagger_obj.parse('') 
        st.session_state['mecab_tagger_initialized'] = True
        st.session_state['tagger_object'] = tagger_obj 
        # st.sidebar.success("MeCab Tagger初期化成功") # 成功時はUIに表示しない方がすっきりするかも
        print("MeCab Tagger initialized and stored in session_state.")
        return tagger_obj 
    except Exception as e_init:
        st.error(f"MeCab Taggerの初期化に失敗しました: {e_init}")
        st.error("リポジトリに `packages.txt` が正しく設定されているか確認してください。")
        st.session_state['mecab_tagger_initialized'] = False
        if 'tagger_object' in st.session_state: del st.session_state['tagger_object']
        return None

# --- スクリプト開始時のTagger準備 ---
if 'tagger_object' not in st.session_state or not st.session_state.get('mecab_tagger_initialized', False):
    print("Attempting to initialize MeCab Tagger into session state...")
    initialize_mecab_tagger_to_session()


# --- フォントパスの決定とMatplotlibへの設定 ---
FONT_PATH_FINAL = None
if st.session_state.get('mecab_tagger_initialized', False): # MeCab初期化成功時のみ
    if os.path.exists(FONT_PATH_PRIMARY):
        FONT_PATH_FINAL = FONT_PATH_PRIMARY
        st.sidebar.info(f"日本語フォント: {os.path.basename(FONT_PATH_FINAL)}")
        try:
            font_entry = fm.FontEntry(fname=FONT_PATH_FINAL, name=os.path.splitext(os.path.basename(FONT_PATH_FINAL))[0])
            if font_entry.name not in [f.name for f in fm.fontManager.ttflist]:
                 fm.fontManager.ttflist.append(font_entry)
            plt.rcParams['font.family'] = font_entry.name
        except Exception as e_font_setting:
            st.sidebar.error(f"Matplotlibフォント設定エラー: {e_font_setting}")
    else:
        st.sidebar.error(f"指定IPAフォント '{FONT_PATH_PRIMARY}' が見つかりません。")
        try:
            font_names_ja = [f.name for f in fm.fontManager.ttflist if any(lang in f.name.lower() for lang in ['ipagp', 'ipag', 'takao', 'noto sans cjk jp', 'hiragino'])]
            if font_names_ja:
                FONT_PATH_FINAL = fm.findfont(fm.FontProperties(family=font_names_ja[0]))
                plt.rcParams['font.family'] = font_names_ja[0]
                st.sidebar.info(f"代替日本語フォントとして '{font_names_ja[0]}' ({os.path.basename(FONT_PATH_FINAL)}) を使用します。")
            else:
                 st.sidebar.error("利用可能な日本語フォントがMatplotlibで見つかりません。")
        except Exception as e_alt_font:
            st.sidebar.error(f"代替フォント検索中にエラー: {e_alt_font}")
else:
    if 'mecab_tagger_initialized' not in st.session_state :
         st.sidebar.warning("MeCab初期化状態が不明です（アプリ初回起動時など）。")
    elif not st.session_state.get('mecab_tagger_initialized', False) :
        st.sidebar.error("MeCabが初期化されていないためフォント設定をスキップします。")


# --- 分析関数の定義 ---
def perform_morphological_analysis(text_input):
    tagger_instance = st.session_state.get('tagger_object')
    if tagger_instance is None or not text_input: return []
    all_morphemes = []
    node = tagger_instance.parseToNode(text_input)
    while node:
        if node.surface:
            features = node.feature.split(',')
            all_morphemes.append({
                '表層形': node.surface, '原形': features[6] if features[6] != '*' else node.surface,
                '品詞': features[0], '品詞細分類1': features[1], '品詞細分類2': features[2],
                '品詞細分類3': features[3], '活用型': features[4], '活用形': features[5],
                '読み': features[7] if len(features) > 7 and features[7] != '*' else '',
                '発音': features[8] if len(features) > 8 and features[8] != '*' else ''
            })
        node = node.next
    return all_morphemes

def generate_word_report(all_morphemes, target_pos_list, stop_words_set):
    if not all_morphemes: return pd.DataFrame(), 0, 0
    report_target_morphemes = []
    for m in all_morphemes:
        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:
            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['非自立', '数', '代名詞', '接尾', 'サ変接続', '副詞可能']:
                continue
            report_target_morphemes.append(m)
    if not report_target_morphemes: return pd.DataFrame(), len(all_morphemes), 0
    word_counts = Counter(m['原形'] for m in report_target_morphemes)
    report_data = []
    representative_info_for_report = {}
    for m_idx in range(len(report_target_morphemes) - 1, -1, -1):
        m = report_target_morphemes[m_idx]
        if m['原形'] not in representative_info_for_report:
            representative_info_for_report[m['原形']] = {'品詞': m['品詞']} 
    total_all_morphemes_count_for_freq = len(all_morphemes)
    total_report_target_morphemes_count = sum(word_counts.values())
    for rank, (word, count) in enumerate(word_counts.most_common(), 1):
        info = representative_info_for_report.get(word, {}) 
        frequency = (count / total_all_morphemes_count_for_freq) * 100 if total_all_morphemes_count_for_freq > 0 else 0
        report_data.append({
            '順位': rank, '単語 (原形)': word, '出現数': count,
            '出現頻度 (%)': round(frequency, 3), '品詞': info.get('品詞', '')
        })
    return pd.DataFrame(report_data), total_all_morphemes_count_for_freq, total_report_target_morphemes_count

def generate_wordcloud_image(all_morphemes, font_path_wc, target_pos_list, stop_words_set):
    if not all_morphemes: st.info("ワードクラウド生成のための形態素データがありません。"); return None
    if font_path_wc is None or not os.path.exists(font_path_wc): st.error(f"ワードクラウド生成に必要な日本語フォントパス '{font_path_wc}' が見つかりません。"); return None
    wordcloud_words = []
    for m in all_morphemes:
        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:
            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['数', '非自立', '代名詞', '接尾']: continue
            wordcloud_words.append(m['原形'])
    wordcloud_text_input_str = " ".join(wordcloud_words)
    if not wordcloud_text_input_str.strip(): st.info("ワードクラウド表示対象の単語が見つかりませんでした（フィルタリング後）。"); return None
    try:
        wc = WordCloud(font_path=font_path_wc, background_color="white", width=800, height=400, max_words=200, collocations=False, random_state=42).generate(wordcloud_text_input_str)
        fig, ax = plt.subplots(figsize=(12,6)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
        return fig
    except Exception as e_wc: st.error(f"ワードクラウド画像生成中にエラーが発生しました: {e_wc}"); return None

def generate_cooccurrence_network_html(all_morphemes, text_input_co, font_path_co, target_pos_list, stop_words_set, node_min_freq, edge_min_freq):
    tagger_instance = st.session_state.get('tagger_object')
    if not all_morphemes or tagger_instance is None or not text_input_co.strip(): st.info("共起ネットワーク生成に必要なデータが不足しています。"); return None
    if font_path_co is None or not os.path.exists(font_path_co): st.error(f"共起ネットワークのラベル表示に必要な日本語フォントパス '{font_path_co}' が見つかりません。"); return None
    temp_words_for_nodes = []
    for m in all_morphemes:
        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:
            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['非自立', '数', '代名詞', '接尾', 'サ変接続', '副詞可能']: continue
            if len(m['原形']) < 2 and m['品詞'] != '名詞': continue
            temp_words_for_nodes.append(m['原形'])
    word_counts = Counter(temp_words_for_nodes); node_candidates = {word: count for word, count in word_counts.items() if count >= node_min_freq}
    if len(node_candidates) < 2: st.info(f"共起ネットワークのノードとなる単語（フィルタ後）が2つ未満です。"); return None
    sentences = re.split(r'[。\n！？]+', text_input_co); sentences = [s.strip() for s in sentences if s.strip()]
    cooccurrence_counts_map = Counter()
    for sentence in sentences:
        node_s = tagger_instance.parseToNode(sentence); words_in_sentence = []
        while node_s:
            if node_s.surface:
                features = node_s.feature.split(','); original_form = features[6] if features[6] != '*' else node_s.surface
                if original_form in node_candidates: words_in_sentence.append(original_form)
            node_s = node_s.next
        for pair in combinations(sorted(list(set(words_in_sentence))), 2): cooccurrence_counts_map[pair] += 1
    if not cooccurrence_counts_map: st.info("共起ペアが見つかりませんでした。"); return None
    
    font_name_pyvis_graph = os.path.splitext(os.path.basename(font_path_co))[0]
    if font_name_pyvis_graph.lower() == 'ipagp': font_name_pyvis_graph = 'IPAPGothic'
    elif font_name_pyvis_graph.lower() == 'ipamp': font_name_pyvis_graph = 'IPAPMincho'
    
    net_graph = Network(notebook=True, height="750px", width="100%", directed=False, bgcolor="#F5F5F5", font_color="#333333")
    for word, count in node_candidates.items():
        node_s_size = int(np.sqrt(count) * 10 + 10)
        net_graph.add_node(word, label=word, size=node_s_size, title=f"{word} (出現数: {count})", font={'face': font_name_pyvis_graph, 'size': 14, 'color': '#333333'}, borderWidth=1, color={'border': '#666666', 'background': '#D2E5FF'})
    added_edge_num = 0
    for pair_nodes, freq_cooc in cooccurrence_counts_map.items():
        if freq_cooc >= edge_min_freq:
            edge_w = float(np.log1p(freq_cooc) * 1.5 + 0.5)
            net_graph.add_edge(pair_nodes[0], pair_nodes[1], value=edge_w, title=f"共起: {freq_cooc}回", color={'color': '#cccccc', 'highlight': '#848484', 'opacity':0.6}); added_edge_num +=1
    if added_edge_num == 0: st.info(f"表示対象の共起ペア（共起回数 {edge_min_freq} 回以上）がありませんでした。"); return None
    options_js_str = """ var options = {"interaction": {"navigationButtons": false, "keyboard": {"enabled": false}}, "manipulation": {"enabled": false}, "configure": {"enabled": false}, "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -30000, "centralGravity": 0.1, "springLength": 150, "springConstant": 0.03, "damping": 0.09, "avoidOverlap": 0.5}, "solver": "barnesHut", "stabilization": {"iterations": 500}}}; """
    try: net_graph.set_options(options_js_str)
    except Exception as e_set_opt: st.warning(f"Pyvisオプション設定で軽微なエラー: {e_set_opt}")
    net_graph.show_buttons(filter_=False)
    return net_graph.generate_html(name="temp_cooc_net_streamlit.html", notebook=True)

def perform_kwic_search(all_morphemes, keyword_str, search_key_type_str, window_int):
    if not keyword_str.strip() or not all_morphemes: return []
    kwic_results_data = []
    for i, morpheme_item in enumerate(all_morphemes):
        target_text_in_morpheme = morpheme_item[search_key_type_str].lower()
        keyword_to_compare = keyword_str.lower()
        if target_text_in_morpheme == keyword_to_compare:
            left_start_idx = max(0, i - window_int); left_ctx_str = "".join(m['表層形'] for m in all_morphemes[left_start_idx:i])
            kw_surface = morpheme_item['表層形']; right_end_idx = min(len(all_morphemes), i + 1 + window_int)
            right_ctx_str = "".join(m['表層形'] for m in all_morphemes[i+1:right_end_idx])
            kwic_results_data.append({'左文脈': left_ctx_str, 'キーワード': kw_surface, '右文脈': right_ctx_str})
    return kwic_results_data

# --- Streamlit UIのメイン部分 ---
st.title("テキストマイニングツール (Streamlit版)")
st.markdown("日本語テキストを入力して、形態素解析、単語レポート、ワードクラウド、共起ネットワーク、KWIC検索を実行します。")

# --- サイドバー: オプション設定 ---
st.sidebar.header("⚙️ 分析オプション")
st.sidebar.markdown("**品詞選択 (各分析共通)**")
default_target_pos = ['名詞', '動詞', '形容詞']
report_target_pos_selected = st.sidebar.multiselect("単語レポート: 対象品詞", ['名詞', '動詞', '形容詞', '副詞', '感動詞', '連体詞'], default=default_target_pos)
wc_target_pos_selected = st.sidebar.multiselect("ワードクラウド: 対象品詞", ['名詞', '動詞', '形容詞', '副詞', '感動詞'], default=default_target_pos)
net_target_pos_selected = st.sidebar.multiselect("共起Net: 対象品詞", ['名詞', '動詞', '形容詞'], default=default_target_pos)

st.sidebar.markdown("**ストップワード設定**")
# ★★★ DEFAULT_STOP_WORDS_SET をテキストエリアの初期値として表示 (カンマ区切り) ★★★
default_stopwords_display_str = ", ".join(sorted(list(DEFAULT_STOP_WORDS_SET)))
custom_stopwords_input_str_from_ui = st.sidebar.text_area("共通ストップワード (原形をカンマや改行区切りで編集してください):", 
                                             value=default_stopwords_display_str, # 初期値を設定
                                             height=250, 
                                             help="ここに入力された単語（原形）がストップワードとして処理されます。")
# ★★★ テキストエリアの内容をそのまま最終的なストップワードセットとして使用 ★★★
final_stop_words_set_for_analysis = set() 
if custom_stopwords_input_str_from_ui.strip():
    custom_list_sw_from_ui = [word.strip().lower() for word in re.split(r'[,\n]', custom_stopwords_input_str_from_ui) if word.strip()]
    final_stop_words_set_for_analysis.update(custom_list_sw_from_ui)
st.sidebar.caption(f"適用される総ストップワード数: {len(final_stop_words_set_for_analysis)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**共起ネットワーク詳細設定**")
network_node_min_freq_val_ui = st.sidebar.slider("ノード最低出現数:", 1, 20, 2, key="net_node_freq_slider_main_ui")
network_edge_min_freq_val_ui = st.sidebar.slider("エッジ最低共起数:", 1, 10, 2, key="net_edge_freq_slider_main_ui")

# --- メイン画面: テキスト入力と実行ボタン ---
main_text_input_area_val = st.text_area("📝 分析したい日本語テキストをここに入力してください:", height=250, 
                             value="これはStreamlitを使用して作成したテキスト分析ツールです。日本語の形態素解析を行い、単語の出現頻度レポート、ワードクラウド、共起ネットワーク、そしてKWIC（文脈付きキーワード検索）などを試すことができます。様々な文章で分析を実行してみてください。")

analyze_button_clicked_event = st.button("分析実行", type="primary", use_container_width=True)

# --- 分析結果表示エリア ---
if analyze_button_clicked_event:
    if not main_text_input_area_val.strip():
        st.warning("分析するテキストを入力してください。")
    elif st.session_state.get('tagger_object') is None or not st.session_state.get('mecab_tagger_initialized', False):
        st.error("MeCab Taggerが利用できません。ページを再読み込みするか、Streamlit Cloudのログを確認してください。")
    else:
        with st.spinner("形態素解析を実行中... しばらくお待ちください。"):
            morphemes_data_list_result = perform_morphological_analysis(main_text_input_area_val) 
        
        if not morphemes_data_list_result:
            st.error("形態素解析に失敗したか、結果が空です。入力テキストを確認してください。")
        else:
            st.success(f"形態素解析が完了しました。総形態素数: {len(morphemes_data_list_result)}")
            st.markdown("---")

            tab_report_view, tab_wc_view, tab_network_view, tab_kwic_view = st.tabs([
                "📊 単語出現レポート", "☁️ ワードクラウド", "🕸️ 共起ネットワーク", "🔍 KWIC検索"
            ])

            with tab_report_view:
                st.subheader("単語出現レポート")
                with st.spinner("レポート作成中..."):
                    df_report_to_show, total_morphs, total_target_morphs = generate_word_report(morphemes_data_list_result, report_target_pos_selected, final_stop_words_set_for_analysis)
                    st.caption(f"総形態素数: {total_morphs} | レポート対象の異なり語数: {len(df_report_to_show)} | レポート対象の延べ語数: {total_target_morphs}")
                    if not df_report_to_show.empty:
                        st.dataframe(df_report_to_show.style.bar(subset=['出現数'], align='left', color='#90EE90')
                                     .format({'出現頻度 (%)': "{:.3f}%"}))
                    else: 
                        st.info("レポート対象の単語が見つかりませんでした。")
            
            with tab_wc_view:
                st.subheader("ワードクラウド")
                if FONT_PATH_FINAL:
                    with st.spinner("ワードクラウド生成中..."):
                        fig_wc_to_show = generate_wordcloud_image(morphemes_data_list_result, FONT_PATH_FINAL, wc_target_pos_selected, final_stop_words_set_for_analysis)
                        if fig_wc_to_show: st.pyplot(fig_wc_to_show)
                    st.caption(f"使用フォント: {os.path.basename(FONT_PATH_FINAL) if FONT_PATH_FINAL else '未設定'}")
                else: st.error("日本語フォントの準備ができていません。ワードクラウドは表示できません。")
            
            with tab_network_view:
                st.subheader("共起ネットワーク")
                if FONT_PATH_FINAL:
                    with st.spinner("共起ネットワーク生成中..."):
                        html_cooc_to_show = generate_cooccurrence_network_html(
                            morphemes_data_list_result, main_text_input_area_val, # taggerは関数内でsession_stateから取得
                            FONT_PATH_FINAL,
                            net_target_pos_selected, final_stop_words_set_for_analysis,
                            network_node_min_freq_val_ui, network_edge_min_freq_val_ui)
                        if html_cooc_to_show: st.components.v1.html(html_cooc_to_show, height=750, scrolling=True)
                    st.caption(f"使用フォント (ノードラベル): {os.path.basename(FONT_PATH_FINAL) if FONT_PATH_FINAL else '未設定'}")
                else: st.error("日本語フォントの準備ができていません。共起ネットワークは表示できません。")
            
            with tab_kwic_view:
                st.subheader("KWIC検索 (文脈付きキーワード検索)")
                if 'kwic_keyword' not in st.session_state: st.session_state.kwic_keyword = ""
                if 'kwic_mode_idx' not in st.session_state: st.session_state.kwic_mode_idx = 0
                if 'kwic_window_val' not in st.session_state: st.session_state.kwic_window_val = 5

                kwic_keyword_input_val = st.text_input("KWIC検索キーワード:", value=st.session_state.kwic_keyword, placeholder="検索したい単語(原形推奨)...", key="kwic_keyword_input_field_tab_v_final")
                st.session_state.kwic_keyword = kwic_keyword_input_val

                kwic_search_mode_options_list = ("原形一致", "表層形一致"); kwic_search_mode_selected_val = st.radio("KWIC検索モード:", kwic_search_mode_options_list, index=st.session_state.kwic_mode_idx, key="kwic_mode_radio_field_tab_v_final")
                st.session_state.kwic_mode_idx = kwic_search_mode_options_list.index(kwic_search_mode_selected_val)

                kwic_window_val_set = st.slider("KWIC表示文脈の形態素数 (前後各):", 1, 15, st.session_state.kwic_window_val, key="kwic_window_slider_field_tab_v_final")
                st.session_state.kwic_window_val = kwic_window_val_set

                if kwic_keyword_input_val.strip():
                    search_key_type_for_kwic_val = '原形' if kwic_search_mode_selected_val == "原形一致" else '表層形'
                    kw_to_search = kwic_keyword_input_val.strip()
                    
                    with st.spinner(f"「{kw_to_search}」を検索中..."):
                        results_kwic_list_data = perform_kwic_search(morphemes_data_list_result, kw_to_search, search_key_type_for_kwic_val, kwic_window_val_set)
                    if results_kwic_list_data:
                        st.write(f"「{kw_to_search}」の検索結果 ({len(results_kwic_list_data)}件):"); df_kwic_to_display_final = pd.DataFrame(results_kwic_list_data); st.dataframe(df_kwic_to_display_final)
                    else: st.info(f"「{kw_to_search}」は見つかりませんでした（現在の検索モードにおいて）。")

# --- フッター情報 ---
st.sidebar.markdown("---")
st.sidebar.info("テキストマイニングツール (Streamlit版) v0.6-final")
```

**主な修正点:**

1.  **`DEFAULT_STOP_WORDS_SET` の定義**:
    * `app.py` の冒頭（定数定義のあたり）に、これまでのやり取りで整備してきた一般的な日本語ストップワード（記号類や「いただく」「れる」「"」なども含む）をセットとして定義しました。
2.  **サイドバーのストップワード入力欄 (`custom_stopwords_input_str_from_ui`)**:
    * `value` 引数に、`DEFAULT_STOP_WORDS_SET` の内容を**カンマ区切り**の文字列 (`", ".join(sorted(list(DEFAULT_STOP_WORDS_SET)))`) にして設定しました。これにより、テキストエリアには最初からこれらのデフォルトストップワードがカンマ区切りで表示されます。
    * （前回改行区切りで表示する形を試しましたが、今回はご要望に合わせてカンマ区切りにしています。もし改行区切りの方がよろしければ `"\n".join(...)` に戻してください。）
    * `height=250` を追加して、複数行表示しやすくしました。
    * `help` 引数も修正し、ここに入力されたものが最終的なストップワードリストになることを示唆するようにしました。
3.  **`final_stop_words_set_for_analysis` の生成ロジック**:
    * `final_stop_words_set_for_analysis = set()` と、まず空のセットで初期化します。
    * その後、`custom_stopwords_input_str_from_ui`（テキストエリアの現在の内容）をパースし（カンマまたは改行で区切る）、その内容**のみ**を `final_stop_words_set_for_analysis` に追加します。
    * これにより、ユーザーがテキストエリアの内容を編集（削除・追加）すれば、それがそのまま分析に使用されるストップワードリストとなります。
4.  **MeCab Taggerの初期化と参照**:
    * `initialize_mecab_tagger_to_session()` 関数内で `st.session_state['tagger_object']` にTaggerオブジェクトを保存し、`st.session_state['mecab_tagger_initialized']` で初期化状態を管理するロジックに戻しました。
    * スクリプトのトップレベルでこの初期化関数を呼び出します。
    * 分析関数内や分析実行ボタンの条件分岐では、`st.session_state.get('tagger_object')` を使ってTaggerオブジェクトを参照します。
5.  **フォント設定**:
    * `japanize-matplotlib` を使わない、`matplotlib.font_manager` を使ったフォント設定ロジックを維持しています。
    * WordCloud用のフォントパスは `FONT_PATH_FINAL` を使用します。
6.  **KWIC検索のウィジェットキー**:
    * 他のウィジェットとキーが衝突しないように、KWIC検索タブ内のウィジェットの `key` に `_tab_v_final` のような接尾辞を追加しました。
7.  **その他**:
    * デバッグ用の `print` 文を一部コメントアウトまたは削除し、Streamlit UIへの情報表示（`st.sidebar.info` など）に置き換えました。
    * バージョン情報をフッターで少し更新しました。

このコードで、ご要望通りのストップワードの扱いと、安定したMeCabの初期化が実現できることを期待しています。
お手数ですが、再度このコードで `app.py` を更新し、デプロイしてご確認くだ
