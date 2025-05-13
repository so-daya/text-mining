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



# --- MeCab Taggerの初期化 (キャッシュ利用) ---

@st.cache_resource

def initialize_mecab_tagger():

    try:

        tagger_obj = MeCab.Tagger(TAGGER_OPTIONS)

        tagger_obj.parse('') 

        st.session_state['mecab_tagger_initialized'] = True

        print("MeCab Tagger initialized successfully via cache.")

        return tagger_obj

    except Exception as e_init:

        st.error(f"MeCab Taggerの初期化に失敗しました: {e_init}")

        st.error("リポジトリに `packages.txt` が正しく設定され、MeCab関連パッケージがインストールされるか確認してください。")

        st.session_state['mecab_tagger_initialized'] = False

        return None



tagger = initialize_mecab_tagger()



# --- フォントパスの決定とMatplotlibへの設定 ---

FONT_PATH_FINAL = None

if 'mecab_tagger_initialized' in st.session_state and st.session_state['mecab_tagger_initialized']:

    if os.path.exists(FONT_PATH_PRIMARY):

        FONT_PATH_FINAL = FONT_PATH_PRIMARY

        st.sidebar.info(f"日本語フォント: {os.path.basename(FONT_PATH_FINAL)}")

        try:

            font_entry = fm.FontEntry(fname=FONT_PATH_FINAL, name=os.path.splitext(os.path.basename(FONT_PATH_FINAL))[0])

            if font_entry.name not in [f.name for f in fm.fontManager.ttflist]:

                 fm.fontManager.ttflist.append(font_entry)

            plt.rcParams['font.family'] = font_entry.name

            print(f"Matplotlibのフォントとして {font_entry.name} を設定しました。")

        except Exception as e_font_setting:

            st.sidebar.error(f"Matplotlibフォント設定エラー: {e_font_setting}")

    else:

        st.sidebar.error(f"指定IPAフォント '{FONT_PATH_PRIMARY}' が見つかりません。")

        try:

            font_names_ja = [f.name for f in fm.fontManager.ttflist if any(lang in f.name.lower() for lang in ['ipagp', 'ipag', 'takao', 'noto sans cjk jp', 'hiragino'])]

            if font_names_ja:

                FONT_PATH_FINAL = fm.findfont(fm.FontProperties(family=font_names_ja[0]))

                plt.rcParams['font.family'] = font_names_ja[0]

                st.sidebar.info(f"代替日本語フォントとして '{font_names_ja[0]}' ({FONT_PATH_FINAL}) を使用します。")

                print(f"Matplotlibの代替フォントとして {font_names_ja[0]} を設定しました。")

            else:

                 st.sidebar.error("利用可能な日本語フォントがMatplotlibで見つかりません。")

        except Exception as e_alt_font:

            st.sidebar.error(f"代替フォント検索中にエラー: {e_alt_font}")

else:

    if 'mecab_tagger_initialized' in st.session_state and not st.session_state.get('mecab_tagger_initialized', False) :

        st.sidebar.error("MeCabが初期化されていないためフォント設定をスキップします。")





# --- 分析関数の定義 ---

def perform_morphological_analysis(text_input, tagger_instance):

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

    if not all_morphemes: 

        return pd.DataFrame(), 0, 0

    

    report_target_morphemes = []

    for m in all_morphemes:

        if m['品詞'] in target_pos_list and m['原形'] not in stop_words_set:

            if m['品詞'] == '名詞' and m['品詞細分類1'] in ['非自立', '数', '代名詞', '接尾', 'サ変接続', '副詞可能']:

                continue

            report_target_morphemes.append(m)



    if not report_target_morphemes: 

        return pd.DataFrame(), len(all_morphemes), 0

        

    word_counts = Counter(m['原形'] for m in report_target_morphemes)

    report_data = []

    

    representative_info_for_report = {}

    for m in reversed(report_target_morphemes): 

        if m['原形'] not in representative_info_for_report:

            representative_info_for_report[m['原形']] = {'品詞': m['品詞']} 

            

    total_all_morphemes_count_for_freq = len(all_morphemes)

    total_report_target_morphemes_count = sum(word_counts.values())



    for rank, (word, count) in enumerate(word_counts.most_common(), 1):

        info = representative_info_for_report.get(word, {}) 

        frequency = (count / total_all_morphemes_count_for_freq) * 100 if total_all_morphemes_count_for_freq > 0 else 0

        report_data.append({

            '順位': rank,

            '単語 (原形)': word,

            '出現数': count,

            '出現頻度 (%)': round(frequency, 3),

            '品詞': info.get('品詞', '')

            # '品詞細分類1', '代表的な表層形', '代表的な読み' は削除済み

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



def generate_cooccurrence_network_html(all_morphemes, text_input_co, tagger_instance, font_path_co, target_pos_list, stop_words_set, node_min_freq, edge_min_freq):

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

        # 検索キーワードと形態素の原形/表層形を比較する前に、両方を小文字化して大文字小文字を区別しないようにする

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



# ★★★ デフォルトストップワードを空のセットに ★★★

DEFAULT_STOP_WORDS_SET = set()



st.sidebar.header("⚙️ 分析オプション")

st.sidebar.markdown("**品詞選択 (各分析共通)**")

default_target_pos = ['名詞', '動詞', '形容詞']

report_target_pos_selected = st.sidebar.multiselect("単語レポート: 対象品詞", ['名詞', '動詞', '形容詞', '副詞', '感動詞', '連体詞'], default=default_target_pos)

wc_target_pos_selected = st.sidebar.multiselect("ワードクラウド: 対象品詞", ['名詞', '動詞', '形容詞', '副詞', '感動詞'], default=default_target_pos)

net_target_pos_selected = st.sidebar.multiselect("共起Net: 対象品詞", ['名詞', '動詞', '形容詞'], default=default_target_pos)



st.sidebar.markdown("**ストップワード設定**")

# ★★★ カスタムストップワード入力のデフォルト値を空に ★★★

custom_stopwords_input_str = st.sidebar.text_area("共通ストップワード (原形をカンマや改行区切りで入力):", 

                                             value="", # デフォルト値を空文字列に変更

                                             help="ここに入力した単語（原形）がストップワードとして処理されます。")

final_stop_words_set = DEFAULT_STOP_WORDS_SET.copy() # デフォルトが空なので、実質ユーザー入力のみ

if custom_stopwords_input_str.strip():

    custom_list_sw = [word.strip().lower() for word in re.split(r'[,\n]', custom_stopwords_input_str) if word.strip()]

    final_stop_words_set.update(custom_list_sw)

st.sidebar.caption(f"適用される総ストップワード数: {len(final_stop_words_set)}")



st.sidebar.markdown("---")

st.sidebar.markdown("**共起ネットワーク詳細設定**")

network_node_min_freq_val = st.sidebar.slider("ノード最低出現数:", 1, 20, 2, key="net_node_freq_slider_main")

network_edge_min_freq_val = st.sidebar.slider("エッジ最低共起数:", 1, 10, 2, key="net_edge_freq_slider_main")



main_text_input_area = st.text_area("📝 分析したい日本語テキストをここに入力してください:", height=250, 

                             value="これはStreamlitを使用して作成したテキスト分析ツールです。日本語の形態素解析を行い、単語の出現頻度レポート、ワードクラウド、共起ネットワーク、そしてKWIC（文脈付きキーワード検索）などを試すことができます。様々な文章で分析を実行してみてください。")



analyze_button_clicked = st.button("分析実行", type="primary", use_container_width=True)



if analyze_button_clicked:

    if not main_text_input_area.strip():

        st.warning("分析するテキストを入力してください。")

    elif tagger is None or not st.session_state.get('mecab_tagger_initialized', False):

        st.error("MeCab Taggerが利用できません。ページを再読み込みするか、Streamlit Cloudのログを確認してください。")

    else:

        with st.spinner("形態素解析を実行中... しばらくお待ちください。"):

            morphemes_data_list = perform_morphological_analysis(main_text_input_area, tagger)

        

        if not morphemes_data_list:

            st.error("形態素解析に失敗したか、結果が空です。入力テキストを確認してください。")

        else:

            st.success(f"形態素解析が完了しました。総形態素数: {len(morphemes_data_list)}")

            st.markdown("---")



            # ★★★ 感情分析タブを削除 ★★★

            tab_report_view, tab_wc_view, tab_network_view, tab_kwic_view = st.tabs([

                "📊 単語出現レポート", "☁️ ワードクラウド", "🕸️ 共起ネットワーク", "🔍 KWIC検索"

            ])



            with tab_report_view:

                st.subheader("単語出現レポート")

                with st.spinner("レポート作成中..."):

                    df_report_to_show, total_morphs, total_target_morphs = generate_word_report(morphemes_data_list, report_target_pos_selected, final_stop_words_set)

                    st.caption(f"総形態素数: {total_morphs} | レポート対象の異なり語数: {len(df_report_to_show)} | レポート対象の延べ語数: {total_target_morphs}")

                    if not df_report_to_show.empty:

                        # ★★★ 出現数の列にミニグラフを適用 (以前のコードで正しく実装済みのはず) ★★★

                        st.dataframe(df_report_to_show.style.bar(subset=['出現数'], align='left', color='#90EE90')

                                     .format({'出現頻度 (%)': "{:.3f}%"}))

                    else: 

                        st.info("レポート対象の単語が見つかりませんでした。")

            

            with tab_wc_view:

                st.subheader("ワードクラウド")

                if FONT_PATH_FINAL:

                    with st.spinner("ワードクラウド生成中..."):

                        fig_wc_to_show = generate_wordcloud_image(morphemes_data_list, FONT_PATH_FINAL, wc_target_pos_selected, final_stop_words_set)

                        if fig_wc_to_show: st.pyplot(fig_wc_to_show)

                    st.caption(f"使用フォント: {os.path.basename(FONT_PATH_FINAL) if FONT_PATH_FINAL else '未設定'}")

                else: st.error("日本語フォントの準備ができていません。ワードクラウドは表示できません。")

            

            with tab_network_view:

                st.subheader("共起ネットワーク")

                if FONT_PATH_FINAL:

                    with st.spinner("共起ネットワーク生成中..."):

                        html_cooc_to_show = generate_cooccurrence_network_html(

                            morphemes_data_list, main_text_input_area, tagger, FONT_PATH_FINAL,

                            net_target_pos_selected, final_stop_words_set,

                            network_node_min_freq_val, network_edge_min_freq_val)

                        if html_cooc_to_show: st.components.v1.html(html_cooc_to_show, height=750, scrolling=True)

                    st.caption(f"使用フォント (ノードラベル): {os.path.basename(FONT_PATH_FINAL) if FONT_PATH_FINAL else '未設定'}")

                else: st.error("日本語フォントの準備ができていません。共起ネットワークは表示できません。")

            

            with tab_kwic_view:

                st.subheader("KWIC検索 (文脈付きキーワード検索)")

                if 'kwic_keyword' not in st.session_state: st.session_state.kwic_keyword = ""

                if 'kwic_mode_idx' not in st.session_state: st.session_state.kwic_mode_idx = 0

                if 'kwic_window_val' not in st.session_state: st.session_state.kwic_window_val = 5



                kwic_keyword_input_val = st.text_input("KWIC検索キーワード:", value=st.session_state.kwic_keyword, placeholder="検索したい単語(原形推奨)...", key="kwic_keyword_input_field_tab")

                st.session_state.kwic_keyword = kwic_keyword_input_val



                kwic_search_mode_options_list = ("原形一致", "表層形一致"); kwic_search_mode_selected_val = st.radio("KWIC検索モード:", kwic_search_mode_options_list, index=st.session_state.kwic_mode_idx, key="kwic_mode_radio_field_tab")

                st.session_state.kwic_mode_idx = kwic_search_mode_options_list.index(kwic_search_mode_selected_val)



                kwic_window_val_set = st.slider("KWIC表示文脈の形態素数 (前後各):", 1, 15, st.session_state.kwic_window_val, key="kwic_window_slider_field_tab")

                st.session_state.kwic_window_val = kwic_window_val_set



                if kwic_keyword_input_val.strip():

                    search_key_type_for_kwic_val = '原形' if kwic_search_mode_selected_val == "原形一致" else '表層形'

                    kw_to_search = kwic_keyword_input_val.strip()

                    

                    with st.spinner(f"「{kw_to_search}」を検索中..."):

                        results_kwic_list_data = perform_kwic_search(morphemes_data_list, kw_to_search, search_key_type_for_kwic_val, kwic_window_val_set)

                    if results_kwic_list_data:

                        st.write(f"「{kw_to_search}」の検索結果 ({len(results_kwic_list_data)}件):"); df_kwic_to_display_final = pd.DataFrame(results_kwic_list_data); st.dataframe(df_kwic_to_display_final)

                    else: st.info(f"「{kw_to_search}」は見つかりませんでした（現在の検索モードにおいて）。")



# --- フッター情報 ---

st.sidebar.markdown("---")

st.sidebar.info("テキストマイニングツール (Streamlit版) v0.3") # バージョンを少し上げました
