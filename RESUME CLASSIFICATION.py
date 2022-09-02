{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "dff2a0cc",
   "metadata": {},
   "source": [
    "### IMPORT NECESSARY LIBRARIES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "969607e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import docx\n",
    "import glob\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import spacy\n",
    "import pickle\n",
    "import random\n",
    "\n",
    "from spacy import displacy\n",
    "import docx\n",
    "import spacy\n",
    "from spacy import schemas\n",
    "from spacy import Dict\n",
    "from spacy.lang.en.stop_words import STOP_WORDS\n",
    "import string\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "\n",
    "import textract\n",
    "import antiword\n",
    "import re\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "stop = stopwords.words('english')\n",
    "#nltk.download('punkt')\n",
    "#nltk.download('averaged_perceptron_tagger')\n",
    "#nltk.download('maxent_ne_chunker')\n",
    "#nltk.download('words')\n",
    "from spacy.matcher import Matcher\n",
    "#nltk.download('stopwords')\n",
    "#nltk.download('wordnet')\n",
    "\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "nlp = spacy.load(\"en_core_web_sm\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6d0b210",
   "metadata": {},
   "source": [
    "### IMPORT DATASETS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1d2c3b90",
   "metadata": {},
   "outputs": [],
   "source": [
    "data1 = pd.read_csv(\"D:\\Data Science\\Resume csv\\intership_resumes.csv\")\n",
    "data2 = pd.read_csv(\"D:\\Data Science\\Resume csv\\Peoplesoft_Resumes.csv\")\n",
    "data3 = pd.read_csv(\"D:\\Data Science\\Resume csv\\React_Developer_resumes.csv\")\n",
    "data4 = pd.read_csv(\"D:\\Data Science\\Resume csv\\SQLDeveloperLightning_Resumes.csv\")\n",
    "data5 = pd.read_csv(\"D:\\Data Science\\Resume csv\\workday_resumes.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8c193b61",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Label</th>\n",
       "      <th>CV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Internship</td>\n",
       "      <td>Name: Ravali P Curriculum Vitae Specialization...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Internship</td>\n",
       "      <td>SUSOVAN BAG Seeking a challenging position in ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Peoplesoft</td>\n",
       "      <td>Anubhav Kumar Singh To work in a globally comp...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Peoplesoft</td>\n",
       "      <td>Profile Summary: 7+ years of experience in imp...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Peoplesoft</td>\n",
       "      <td>PeopleSoft Database Administrator Gangareddy P...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74</th>\n",
       "      <td>workdayResumes</td>\n",
       "      <td>Workday Integration Consultant Name : Sri Kris...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75</th>\n",
       "      <td>workdayResumes</td>\n",
       "      <td>SRIKANTH (WORKDAY HCM CONSULTANT) Seeking suit...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>76</th>\n",
       "      <td>workdayResumes</td>\n",
       "      <td>WORKDAY | HCM | FCM Name : Kumar S.S Role : Wo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77</th>\n",
       "      <td>workdayResumes</td>\n",
       "      <td>Venkateswarlu.B Workday Consultant Having 5.3 ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>78</th>\n",
       "      <td>workdayResumes</td>\n",
       "      <td>Vinay kumar .v Workday Functional Consultant E...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>79 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             Label                                                 CV\n",
       "0       Internship  Name: Ravali P Curriculum Vitae Specialization...\n",
       "1       Internship  SUSOVAN BAG Seeking a challenging position in ...\n",
       "2       Peoplesoft  Anubhav Kumar Singh To work in a globally comp...\n",
       "3       Peoplesoft  Profile Summary: 7+ years of experience in imp...\n",
       "4       Peoplesoft  PeopleSoft Database Administrator Gangareddy P...\n",
       "..             ...                                                ...\n",
       "74  workdayResumes  Workday Integration Consultant Name : Sri Kris...\n",
       "75  workdayResumes  SRIKANTH (WORKDAY HCM CONSULTANT) Seeking suit...\n",
       "76  workdayResumes  WORKDAY | HCM | FCM Name : Kumar S.S Role : Wo...\n",
       "77  workdayResumes  Venkateswarlu.B Workday Consultant Having 5.3 ...\n",
       "78  workdayResumes  Vinay kumar .v Workday Functional Consultant E...\n",
       "\n",
       "[79 rows x 2 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Resume = pd.concat([data1,data2,data3,data4,data5],axis=0)\n",
    "Resume = Resume.reset_index()\n",
    "Resume = Resume.drop(columns='Number',axis=0)\n",
    "Resume = Resume.drop(columns='index',axis=0)\n",
    "Resume"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af177081",
   "metadata": {},
   "source": [
    "### Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "18c4de3f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 79 entries, 0 to 78\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype \n",
      "---  ------  --------------  ----- \n",
      " 0   Label   79 non-null     object\n",
      " 1   CV      79 non-null     object\n",
      "dtypes: object(2)\n",
      "memory usage: 1.4+ KB\n"
     ]
    }
   ],
   "source": [
    "Resume.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59734f95",
   "metadata": {},
   "source": [
    "### Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbe5f824",
   "metadata": {},
   "source": [
    "##### We will perform label encoding to convert category variable from string datatype to float datatype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e4c7bb86",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Label</th>\n",
       "      <th>CV</th>\n",
       "      <th>Encoded_Skill</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Internship</td>\n",
       "      <td>Name: Ravali P Curriculum Vitae Specialization...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Internship</td>\n",
       "      <td>SUSOVAN BAG Seeking a challenging position in ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Peoplesoft</td>\n",
       "      <td>Anubhav Kumar Singh To work in a globally comp...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Peoplesoft</td>\n",
       "      <td>Profile Summary: 7+ years of experience in imp...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Peoplesoft</td>\n",
       "      <td>PeopleSoft Database Administrator Gangareddy P...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Label                                                 CV  \\\n",
       "0  Internship  Name: Ravali P Curriculum Vitae Specialization...   \n",
       "1  Internship  SUSOVAN BAG Seeking a challenging position in ...   \n",
       "2  Peoplesoft  Anubhav Kumar Singh To work in a globally comp...   \n",
       "3  Peoplesoft  Profile Summary: 7+ years of experience in imp...   \n",
       "4  Peoplesoft  PeopleSoft Database Administrator Gangareddy P...   \n",
       "\n",
       "   Encoded_Skill  \n",
       "0              0  \n",
       "1              0  \n",
       "2              1  \n",
       "3              1  \n",
       "4              1  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "le_encoder = LabelEncoder()\n",
    "Resume[\"Encoded_Skill\"] = le_encoder.fit_transform(Resume[\"Label\"])\n",
    "Resume.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "69c74546",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ReactDeveloper    22\n",
       "workdayResumes    21\n",
       "Peoplesoft        20\n",
       "SQLDeveloper      14\n",
       "Internship         2\n",
       "Name: Label, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Resume.Label.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e501b5d8",
   "metadata": {},
   "source": [
    "### Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "87532f7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re #REGULAR EXPRESSION\n",
    "import string\n",
    "\n",
    "def clean_text(CV):\n",
    "    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''\n",
    "    CV = CV.lower()\n",
    "    CV = re.sub('\\[.*?\\]', '', CV)\n",
    "    CV = re.sub('[%s]' % re.escape(string.punctuation), '', CV)\n",
    "    CV = re.sub('\\w*\\d\\w*', '', CV)\n",
    "    CV = re.sub(\"[0-9\" \"]+\",\" \",CV)\n",
    "    CV = re.sub('[‘’“”…]', '', CV)\n",
    "    return CV\n",
    "\n",
    "clean = lambda x: clean_text(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "cd398200",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     name ravali p curriculum vitae specialization ...\n",
       "1     susovan bag seeking a challenging position in ...\n",
       "2     anubhav kumar singh to work in a globally comp...\n",
       "3     profile summary  years of experience in implem...\n",
       "4     peoplesoft database administrator gangareddy p...\n",
       "                            ...                        \n",
       "74    workday integration consultant name  sri krish...\n",
       "75    srikanth workday hcm consultant seeking suitab...\n",
       "76    workday  hcm  fcm name  kumar ss role  workday...\n",
       "77    venkateswarlub workday consultant having  year...\n",
       "78    vinay kumar v workday functional consultant ex...\n",
       "Name: CV, Length: 79, dtype: object"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Resume['CV'] = Resume.CV.apply(clean)\n",
    "Resume.CV"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d3a8f46",
   "metadata": {},
   "source": [
    "### Word frequency BEFORE removal of STOPWORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "14816305",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "and            2696\n",
       "the            1329\n",
       "in             1244\n",
       "to             1048\n",
       "of              961\n",
       "for             636\n",
       "on              625\n",
       "experience      572\n",
       "with            410\n",
       "as              391\n",
       "peoplesoft      386\n",
       "application     378\n",
       "using           375\n",
       "workday         368\n",
       "server          317\n",
       "a               307\n",
       "from            296\n",
       "reports         295\n",
       "data            285\n",
       "project         265\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Word Frequency\n",
    "frequency = pd.Series(' '.join(Resume['CV']).split()).value_counts()[:20] #For top 20\n",
    "frequency"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2247c16e",
   "metadata": {},
   "source": [
    "### Removing STOPWORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fd765777",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "stop = stopwords.words('english')\n",
    "Resume['CV'] = Resume['CV'].apply(lambda x: \" \".join(x for x in x.split() if x not in stop))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f01c3b4a",
   "metadata": {},
   "source": [
    "### Word frequency AFTER removal of STOPWORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ef489a11",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "experience      572\n",
       "peoplesoft      386\n",
       "application     378\n",
       "using           375\n",
       "workday         368\n",
       "server          317\n",
       "reports         295\n",
       "data            285\n",
       "project         265\n",
       "business        250\n",
       "process         220\n",
       "database        217\n",
       "web             214\n",
       "knowledge       201\n",
       "sql             196\n",
       "worked          195\n",
       "involved        184\n",
       "integrations    175\n",
       "like            169\n",
       "integration     167\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "frequency_Sw = pd.Series(' '.join(Resume['CV']).split()).value_counts()[:20] # for top 20\n",
    "frequency_Sw"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7dd797db",
   "metadata": {},
   "source": [
    "### Performing A NER (Using Spacy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7b06d860",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">name ravali p curriculum vitae specialization computer science engg utilize technical skills achieving target developing best performance organization manual testing skills strong knowledge sdlc concepts extensive knowledge white box testing good knowledge functional testing integration testing extreme knowledge system testing good knowledge adhoc testing reliability testing good knowledge exploratory testing good knowledge stlc concepts good knowledge test cases test scenarios good knowledge globalization testing compatibility testing knowledge regression testing good knowledge test plan agile methdology good knowledge scrum methodology expertise sprint planning meeting good knowledge scrum meeting extreme knowledge sprint retrospective meeting good knowledge product backlog meeting bug triage meeting extreme knowledge normalization java skills good knowledge method overloading method overriding good understanding static nonstatic good understanding variables good knowledge constructor good knowledge abstraction good knowledge encapsulation good knowledge inheritance good knowledge collections training courses industrial exposure achievements im certified cyber security training sjbit bengaluru im certified volleyball olympics distict level assignements identified functional test cases flipkartcom identified integration test cases whatsapp identified integration test cases \n",
       "<mark class=\"entity\" style=\"background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    amazoncom\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">ORG</span>\n",
       "</mark>\n",
       " found defects \n",
       "<mark class=\"entity\" style=\"background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    ft usability\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">ORG</span>\n",
       "</mark>\n",
       " camaptibility globalization testing strengths date birth gender female father name fasala reddy n languages known \n",
       "<mark class=\"entity\" style=\"background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    english\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">NORP</span>\n",
       "</mark>\n",
       " telugukannadahindi nationality indian address thirumaladevarahallivparthihallip \n",
       "<mark class=\"entity\" style=\"background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    kodigenahallihmadhugirittumkurd\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
       "</mark>\n",
       " state \n",
       "<mark class=\"entity\" style=\"background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    karnataka hereby declare\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">ORG</span>\n",
       "</mark>\n",
       " abovementioned information true best knowledge yoursincerely ravali p place bangalore</div></span>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "nlp = spacy.load(\"en_core_web_sm\")\n",
    "text=nlp(Resume[\"CV\"][0])\n",
    "displacy.render(text, style = \"ent\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "233b1021",
   "metadata": {},
   "source": [
    "### VISUALIZATION OF DATASET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6c4e492d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "plt.style.use('seaborn-dark-palette')\n",
    "plt.figure(figsize=(10,4))\n",
    "plt.title(\"The distinct categories of resumes\")\n",
    "plt.xticks(rotation=90)\n",
    "sns.countplot(y=\"Label\", data=Resume,palette=(\"Set2\"))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f56bd91b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x1080 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib.gridspec import GridSpec\n",
    "targetCounts = Resume.Label.value_counts()\n",
    "targetLabels  = Resume.Label.unique()\n",
    "# Make square figures and axes\n",
    "plt.figure(1, figsize=(15,15))\n",
    "the_grid = GridSpec(2, 2)\n",
    "\n",
    "\n",
    "cmap = plt.get_cmap('plasma')\n",
    "colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']\n",
    "plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')\n",
    "\n",
    "source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3343205",
   "metadata": {},
   "source": [
    "#### First take a look at the number of Characters present in each sentence. This can give us a rough idea about the resume length"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6fd299a9",
   "metadata": {},
   "source": [
    "##### Calculating each Characterstic in dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1c5ffae0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     1784\n",
       "1     1078\n",
       "2     5804\n",
       "3     5931\n",
       "4     4795\n",
       "      ... \n",
       "74    5516\n",
       "75    5146\n",
       "76    6046\n",
       "77    2154\n",
       "78    4131\n",
       "Name: CV, Length: 79, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "characters=Resume[\"CV\"].apply(len)\n",
    "characters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b75b61b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Number of characters dataset: 295942\n",
      "Mean of each characters in datset: 3746.1012658227846\n",
      "Median of characters in dataset: 3015.0\n",
      "Standard Deviation of characters in dataset: 2487.3184489600117\n",
      "skew of characters dataset: 1.7826848176246577\n"
     ]
    }
   ],
   "source": [
    "print('Total Number of characters dataset:',characters.sum())\n",
    "print('Mean of each characters in datset:',characters.mean())\n",
    "print('Median of characters in dataset:',characters.median())\n",
    "print('Standard Deviation of characters in dataset:',characters.std())\n",
    "print('skew of characters dataset:',characters.skew())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "43cf424e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Density'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "sns.distplot(x = characters)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90d352c8",
   "metadata": {},
   "source": [
    "#### Calculating each Word Characterstic in dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "25e664b0",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     207\n",
       "1     130\n",
       "2     685\n",
       "3     707\n",
       "4     553\n",
       "     ... \n",
       "74    658\n",
       "75    644\n",
       "76    768\n",
       "77    259\n",
       "78    500\n",
       "Name: CV, Length: 79, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "words = Resume['CV'].apply(lambda x: len(str(x).split(' ')))\n",
    "words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "aebb9cc3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Number of Word in dataset: 36185\n",
      "Mean of each Word in datset: 458.0379746835443\n",
      "Median of Word in dataset: 380.0\n",
      "Standard Deviation of Word in dataset: 295.5711308826187\n",
      "skew of Word dataset: 1.6889132345848146\n"
     ]
    }
   ],
   "source": [
    "print('Total Number of Word in dataset:',words.sum())\n",
    "print('Mean of each Word in datset:',words.mean())\n",
    "print('Median of Word in dataset:',words.median())\n",
    "print('Standard Deviation of Word in dataset:',words.std())\n",
    "print('skew of Word dataset:',words.skew())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "36054929",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Density'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(x = words)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72d2cb95",
   "metadata": {},
   "source": [
    "#### Feature Extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "6910fdd6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import Counter\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6b0dc452",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "words =['using','Workday','Experience','PeopleSoft',\n",
    " 'experience','SQL','Application','data','Server',\n",
    " 'business','Project','reports','like','HCM','Worked',\n",
    " 'knowledge','Involved','various','Good', 'Reports','React','EIB','integrations','Web','system','creating','issues',\n",
    " 'Created', 'Responsibilities','Process','process','support', \n",
    " 'application','new','People','I','team','working', \n",
    " 'Database','database','Integration','Domains','client', \n",
    " 'requirements','Core',  'Business', \n",
    "'Oracle','Report', 'Developer', 'Data']\n",
    "indices = np.random.zipf(1.6, size=500).astype(np.int) % len(words)\n",
    "tw = np.array(words)[indices]\n",
    "\n",
    "tf = Counter(tw)\n",
    "\n",
    "y = [count for tag, count in tf.most_common(50)]\n",
    "x = [tag for tag, count in tf.most_common(50)]\n",
    "plt.style.use('seaborn-dark-palette')\n",
    "plt.figure(figsize=(12,5))\n",
    "plt.bar(x, y, color=['gold','lightcoral', 'lightskyblue'])\n",
    "plt.title(\"Word frequencies in Resume Data in Log Scale\")\n",
    "plt.ylabel(\"Frequency (log scale)\")\n",
    "plt.yscale('symlog') # optionally set a log scale for the y-axis\n",
    "plt.xticks(rotation=90)\n",
    "for i, (tag, count) in enumerate(tf.most_common(50)):\n",
    "    plt.text(i, count, f' {count} ', rotation=90,\n",
    "             ha='center', va='top' if i < 10 else 'bottom', color='white' if i < 10 else 'black')\n",
    "plt.xlim(-0.6, len(x)-0.4) # optionally set tighter x lims\n",
    "plt.tight_layout() # change the whitespace such that all labels fit nicely\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b7375dd0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def wordBarGraphFunction_1(df,column,title):\n",
    "    topic_words = [ z.lower() for y in\n",
    "                       [ x.split() for x in df[column] if isinstance(x, str)]\n",
    "                       for z in y]\n",
    "    word_count_dict = dict(Counter(topic_words))\n",
    "    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)\n",
    "    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words(\"english\")]\n",
    "    plt.style.use('fivethirtyeight')\n",
    "    sns.barplot(x=np.arange(20),y= [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])])\n",
    "    plt.xticks([x + 0.5 for x in range(20)], reversed(popular_words_nonstop[0:20]),rotation=90)\n",
    "    plt.title(title)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "da17d58c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "wordBarGraphFunction_1(Resume,\"CV\",\"Most frequent Words \")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca899ff1",
   "metadata": {},
   "source": [
    "#### WORDCLOUD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3ed74854",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import packages\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "from wordcloud import WordCloud, STOPWORDS\n",
    "# Define a function to plot word cloud\n",
    "def plot_cloud(wordcloud):\n",
    "    # Set figure size\n",
    "    plt.figure(figsize=(15,5))\n",
    "    # Display image\n",
    "    plt.imshow(wordcloud) \n",
    "    # No axis details\n",
    "    plt.axis(\"off\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "42e932da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Generate wordcloud\n",
    "stopwords = STOPWORDS\n",
    "stopwords.add('will')\n",
    "wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(str(Resume))\n",
    "# Plot\n",
    "plot_cloud(wordcloud)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1451e661",
   "metadata": {},
   "source": [
    "#### Bag Of Words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "85af2dc5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'ravali': 3137,\n",
       " 'p': 2706,\n",
       " 'curriculum': 914,\n",
       " 'vitae': 4200,\n",
       " 'specialization': 3623,\n",
       " 'computer': 722,\n",
       " 'science': 3417,\n",
       " 'engg': 1270,\n",
       " 'utilize': 4118,\n",
       " 'technical': 3870,\n",
       " 'skills': 3570,\n",
       " 'achieving': 39,\n",
       " 'target': 3844,\n",
       " 'developing': 1075,\n",
       " 'best': 394,\n",
       " 'performance': 2795,\n",
       " 'organization': 2667,\n",
       " 'manual': 2292,\n",
       " 'testing': 3909,\n",
       " 'strong': 3738,\n",
       " 'knowledge': 2107,\n",
       " 'sdlc': 3435,\n",
       " 'concepts': 727,\n",
       " 'extensive': 1386,\n",
       " 'white': 4262,\n",
       " 'box': 448,\n",
       " 'good': 1609,\n",
       " 'functional': 1537,\n",
       " 'integration': 1952,\n",
       " 'extreme': 1393,\n",
       " 'adhoc': 72,\n",
       " 'reliability': 3225,\n",
       " 'exploratory': 1375,\n",
       " 'stlc': 3716,\n",
       " 'test': 3905,\n",
       " 'cases': 545,\n",
       " 'scenarios': 3401,\n",
       " 'globalization': 1595,\n",
       " 'compatibility': 700,\n",
       " 'regression': 3211,\n",
       " 'plan': 2846,\n",
       " 'agile': 108,\n",
       " 'methdology': 2374,\n",
       " 'scrum': 3434,\n",
       " 'methodology': 2378,\n",
       " 'expertise': 1371,\n",
       " 'sprint': 3645,\n",
       " 'planning': 2848,\n",
       " 'meeting': 2356,\n",
       " 'retrospective': 3312,\n",
       " 'product': 2983,\n",
       " 'backlog': 340,\n",
       " 'bug': 479,\n",
       " 'triage': 4018,\n",
       " 'normalization': 2562,\n",
       " 'java': 2034,\n",
       " 'method': 2375,\n",
       " 'overloading': 2696,\n",
       " 'overriding': 2697,\n",
       " 'understanding': 4061,\n",
       " 'static': 3707,\n",
       " 'nonstatic': 2560,\n",
       " 'variables': 4141,\n",
       " 'constructor': 776,\n",
       " 'abstraction': 12,\n",
       " 'encapsulation': 1256,\n",
       " 'inheritance': 1912,\n",
       " 'collections': 662,\n",
       " 'training': 3986,\n",
       " 'courses': 854,\n",
       " 'industrial': 1899,\n",
       " 'exposure': 1381,\n",
       " 'achievements': 37,\n",
       " 'im': 1833,\n",
       " 'certified': 581,\n",
       " 'cyber': 931,\n",
       " 'security': 3447,\n",
       " 'sjbit': 3566,\n",
       " 'bengaluru': 393,\n",
       " 'volleyball': 4211,\n",
       " 'olympics': 2620,\n",
       " 'distict': 1128,\n",
       " 'level': 2163,\n",
       " 'assignements': 263,\n",
       " 'identified': 1816,\n",
       " 'flipkartcom': 1478,\n",
       " 'whatsapp': 4260,\n",
       " 'amazoncom': 148,\n",
       " 'defects': 992,\n",
       " 'ft': 1528,\n",
       " 'usability': 4099,\n",
       " 'camaptibility': 512,\n",
       " 'strengths': 3734,\n",
       " 'date': 952,\n",
       " 'birth': 415,\n",
       " 'gender': 1571,\n",
       " 'female': 1439,\n",
       " 'father': 1422,\n",
       " 'fasala': 1419,\n",
       " 'reddy': 3186,\n",
       " 'n': 2489,\n",
       " 'languages': 2129,\n",
       " 'known': 2109,\n",
       " 'english': 1279,\n",
       " 'telugukannadahindi': 3889,\n",
       " 'nationality': 2508,\n",
       " 'indian': 1890,\n",
       " 'address': 67,\n",
       " 'thirumaladevarahallivparthihallip': 3924,\n",
       " 'kodigenahallihmadhugirittumkurd': 2110,\n",
       " 'state': 3702,\n",
       " 'karnataka': 2085,\n",
       " 'declare': 984,\n",
       " 'abovementioned': 6,\n",
       " 'information': 1905,\n",
       " 'true': 4026,\n",
       " 'yoursincerely': 4338,\n",
       " 'place': 2842,\n",
       " 'bangalore': 354,\n",
       " 'susovan': 3806,\n",
       " 'bag': 345,\n",
       " 'seeking': 3449,\n",
       " 'challenging': 588,\n",
       " 'position': 2881,\n",
       " 'field': 1446,\n",
       " 'technology': 3880,\n",
       " 'individual': 1893,\n",
       " 'growth': 1651,\n",
       " 'enhance': 1281,\n",
       " 'academic': 13,\n",
       " 'learning': 2152,\n",
       " 'ccna': 561,\n",
       " 'routing': 3357,\n",
       " 'switching': 3812,\n",
       " 'subnetting': 3759,\n",
       " 'programming': 3005,\n",
       " 'c': 500,\n",
       " 'cjava': 617,\n",
       " 'htmlcss': 1783,\n",
       " 'sql': 3647,\n",
       " 'oops': 2630,\n",
       " 'algorithms': 133,\n",
       " 'data': 943,\n",
       " 'structuresdbms': 3742,\n",
       " 'networking': 2535,\n",
       " 'os': 2679,\n",
       " 'linux': 2192,\n",
       " 'administration': 81,\n",
       " 'troubleshooting': 4025,\n",
       " 'soft': 3591,\n",
       " 'leadership': 2145,\n",
       " 'collaboration': 659,\n",
       " 'communication': 689,\n",
       " 'customer': 918,\n",
       " 'handling': 1679,\n",
       " 'englishfluent': 1280,\n",
       " 'hindifluent': 1735,\n",
       " 'bengalinative': 392,\n",
       " 'telugu': 3887,\n",
       " 'projects': 3008,\n",
       " 'smart': 3582,\n",
       " 'agriculture': 113,\n",
       " 'built': 488,\n",
       " 'farmers': 1418,\n",
       " 'using': 4111,\n",
       " 'iot': 2007,\n",
       " 'solution': 3599,\n",
       " 'automatic': 303,\n",
       " 'water': 4233,\n",
       " 'motor': 2447,\n",
       " 'controller': 807,\n",
       " 'android': 178,\n",
       " 'api': 194,\n",
       " 'technologies': 3879,\n",
       " 'combined': 672,\n",
       " 'automate': 301,\n",
       " 'work': 4281,\n",
       " 'controllers': 808,\n",
       " 'sensors': 3468,\n",
       " 'bookstore': 437,\n",
       " 'management': 2275,\n",
       " 'interface': 1971,\n",
       " 'pp': 2895,\n",
       " 'manage': 2273,\n",
       " 'purchase': 3065,\n",
       " 'return': 3313,\n",
       " 'books': 436,\n",
       " 'store': 3721,\n",
       " 'hotel': 1762,\n",
       " 'systemdec': 3824,\n",
       " 'designed': 1048,\n",
       " 'end': 1260,\n",
       " 'module': 2421,\n",
       " 'website': 4251,\n",
       " 'online': 2624,\n",
       " 'movie': 2455,\n",
       " 'ticket': 3933,\n",
       " 'booking': 435,\n",
       " 'fully': 1534,\n",
       " 'functioning': 1540,\n",
       " 'htmlcssjavascript': 1785,\n",
       " 'education': 1202,\n",
       " 'background': 339,\n",
       " 'lovely': 2236,\n",
       " 'professional': 2989,\n",
       " 'university': 4071,\n",
       " 'punjab': 3062,\n",
       " 'india': 1889,\n",
       " 'btech': 475,\n",
       " 'engineering': 1273,\n",
       " 'gpa': 1621,\n",
       " 'hobbies': 1742,\n",
       " 'web': 4241,\n",
       " 'surfing': 3803,\n",
       " 'cricketcarromchess': 879,\n",
       " 'anubhav': 191,\n",
       " 'kumar': 2118,\n",
       " 'singh': 3556,\n",
       " 'globally': 1596,\n",
       " 'competitive': 705,\n",
       " 'environment': 1302,\n",
       " 'assignments': 266,\n",
       " 'shall': 3509,\n",
       " 'yield': 4332,\n",
       " 'twin': 4035,\n",
       " 'benefits': 390,\n",
       " 'job': 2045,\n",
       " 'satisfaction': 3389,\n",
       " 'steadypaced': 3711,\n",
       " 'experience': 1366,\n",
       " 'current': 911,\n",
       " 'hcl': 1695,\n",
       " 'role': 3342,\n",
       " 'admin': 74,\n",
       " 'offshore': 2616,\n",
       " 'shell': 3516,\n",
       " 'scripting': 3431,\n",
       " 'peoplesoft': 2790,\n",
       " 'github': 1585,\n",
       " 'managing': 2284,\n",
       " 'hcm': 1696,\n",
       " 'fscm': 1525,\n",
       " 'production': 2984,\n",
       " 'environments': 1304,\n",
       " 'support': 3796,\n",
       " 'installed': 1936,\n",
       " 'windows': 4268,\n",
       " 'involved': 2000,\n",
       " 'day': 956,\n",
       " 'activities': 53,\n",
       " 'project': 3007,\n",
       " 'migration': 2390,\n",
       " 'database': 944,\n",
       " 'refresh': 3201,\n",
       " 'changes': 591,\n",
       " 'tax': 3854,\n",
       " 'updates': 4085,\n",
       " 'various': 4144,\n",
       " 'servers': 3482,\n",
       " 'like': 2180,\n",
       " 'application': 201,\n",
       " 'process': 2971,\n",
       " 'scheduler': 3405,\n",
       " 'applying': 208,\n",
       " 'tuxedo': 4032,\n",
       " 'weblogic': 4243,\n",
       " 'middleware': 2387,\n",
       " 'cpu': 858,\n",
       " 'patches': 2755,\n",
       " 'applications': 202,\n",
       " 'working': 4296,\n",
       " 'used': 4103,\n",
       " 'exchange': 1346,\n",
       " 'files': 1449,\n",
       " 'external': 1388,\n",
       " 'systems': 3825,\n",
       " 'installation': 1932,\n",
       " 'setup': 3498,\n",
       " 'requirement': 3261,\n",
       " 'reviewing': 3321,\n",
       " 'vulnerabilities': 4220,\n",
       " 'reported': 3247,\n",
       " 'teams': 3863,\n",
       " 'renewal': 3238,\n",
       " 'ssl': 3672,\n",
       " 'vulnerability': 4221,\n",
       " 'remediation': 3231,\n",
       " 'report': 3246,\n",
       " 'rmis': 3336,\n",
       " 'team': 3859,\n",
       " 'worked': 4289,\n",
       " 'pum': 3056,\n",
       " 'update': 4082,\n",
       " 'manager': 2278,\n",
       " 'dpk': 1157,\n",
       " 'ansible': 188,\n",
       " 'docker': 1136,\n",
       " 'new': 2537,\n",
       " 'scripts': 3433,\n",
       " 'script': 3430,\n",
       " 'failures': 1407,\n",
       " 'techmahindra': 3868,\n",
       " 'roleproject': 3344,\n",
       " 'automationdevops': 307,\n",
       " 'tools': 3960,\n",
       " 'jenkins': 2038,\n",
       " 'people': 2787,\n",
       " 'upgrade': 4089,\n",
       " 'ntt': 2579,\n",
       " 'cloud': 639,\n",
       " 'onpremises': 2626,\n",
       " 'aws': 326,\n",
       " 'automated': 302,\n",
       " 'startstop': 3698,\n",
       " 'basic': 370,\n",
       " 'certificates': 578,\n",
       " 'released': 3220,\n",
       " 'elastic': 1220,\n",
       " 'search': 3437,\n",
       " 'configuration': 747,\n",
       " 'gained': 1553,\n",
       " 'resources': 3281,\n",
       " 'unix': 4074,\n",
       " 'architecturecommand': 235,\n",
       " 'trouble': 4022,\n",
       " 'shooting': 3522,\n",
       " 'unixlinux': 4075,\n",
       " 'platform': 2853,\n",
       " 'efficient': 1211,\n",
       " 'deployment': 1037,\n",
       " 'tool': 3959,\n",
       " 'scheduling': 3408,\n",
       " 'crontab': 883,\n",
       " 'ibm': 1805,\n",
       " 'tivoli': 3954,\n",
       " 'workload': 4298,\n",
       " 'tws': 4037,\n",
       " 'automation': 306,\n",
       " 'continuous': 798,\n",
       " 'installing': 1938,\n",
       " 'configuring': 751,\n",
       " 'responsible': 3287,\n",
       " 'writing': 4312,\n",
       " 'playbook': 2856,\n",
       " 'perform': 2794,\n",
       " 'task': 3846,\n",
       " 'managed': 2274,\n",
       " 'tasks': 3847,\n",
       " 'related': 3214,\n",
       " 'issue': 2015,\n",
       " 'certification': 579,\n",
       " 'products': 2987,\n",
       " 'platforms': 2854,\n",
       " 'browsers': 471,\n",
       " 'peopletools': 2792,\n",
       " 'server': 3479,\n",
       " 'amazon': 147,\n",
       " 'service': 3485,\n",
       " 'components': 717,\n",
       " 'logic': 2216,\n",
       " 'release': 3219,\n",
       " 'latest': 2134,\n",
       " 'patch': 2754,\n",
       " 'change': 589,\n",
       " 'assistant': 270,\n",
       " 'passes': 2745,\n",
       " 'creating': 866,\n",
       " 'running': 3367,\n",
       " 'setting': 3496,\n",
       " 'monitor': 2429,\n",
       " 'app': 196,\n",
       " 'domains': 1150,\n",
       " 'post': 2887,\n",
       " 'severs': 3504,\n",
       " 'common': 685,\n",
       " 'domain': 1148,\n",
       " 'boot': 440,\n",
       " 'problems': 2967,\n",
       " 'identifying': 1818,\n",
       " 'source': 3609,\n",
       " 'databases': 945,\n",
       " 'ca': 501,\n",
       " 'images': 1835,\n",
       " 'packages': 2709,\n",
       " 'sourcetarget': 3611,\n",
       " 'srdt': 3656,\n",
       " 'pvt': 3078,\n",
       " 'srm': 3661,\n",
       " 'group': 1642,\n",
       " 'designer': 1050,\n",
       " 'campus': 514,\n",
       " 'maintaining': 2260,\n",
       " 'supporting': 3799,\n",
       " 'oracle': 2658,\n",
       " 'possess': 2885,\n",
       " 'architecture': 234,\n",
       " 'administering': 79,\n",
       " 'pia': 2831,\n",
       " 'internet': 1981,\n",
       " 'broker': 469,\n",
       " 'nodes': 2553,\n",
       " 'issues': 2017,\n",
       " 'migrations': 2391,\n",
       " 'ps': 3036,\n",
       " 'bundle': 491,\n",
       " 'refreshed': 3202,\n",
       " 'dev': 1068,\n",
       " 'preprod': 2921,\n",
       " 'prod': 2979,\n",
       " 'experienced': 1367,\n",
       " 'providing': 3029,\n",
       " 'development': 1077,\n",
       " 'hrms': 1776,\n",
       " 'cs': 894,\n",
       " 'configured': 750,\n",
       " 'ses': 3492,\n",
       " 'secure': 3445,\n",
       " 'enterprise': 1294,\n",
       " 'instance': 1939,\n",
       " 'policy': 2871,\n",
       " 'modelling': 2412,\n",
       " 'opa': 2632,\n",
       " 'existing': 1358,\n",
       " 'implemented': 1846,\n",
       " 'single': 3558,\n",
       " 'sign': 3537,\n",
       " 'interaction': 1965,\n",
       " 'hub': 1793,\n",
       " 'integrate': 1948,\n",
       " 'content': 793,\n",
       " 'upgraded': 4090,\n",
       " 'finance': 1457,\n",
       " 'image': 1834,\n",
       " 'bugs': 480,\n",
       " 'tailored': 3834,\n",
       " 'dbua': 963,\n",
       " 'created': 863,\n",
       " 'instances': 1940,\n",
       " 'provide': 3025,\n",
       " 'prepared': 2919,\n",
       " 'status': 3710,\n",
       " 'reports': 3249,\n",
       " 'sheets': 3515,\n",
       " 'coordinated': 826,\n",
       " 'provided': 3026,\n",
       " 'imported': 1851,\n",
       " 'self': 3453,\n",
       " 'signed': 3541,\n",
       " 'certificate': 577,\n",
       " 'port': 2879,\n",
       " 'access': 19,\n",
       " 'set': 3494,\n",
       " 'terminal': 3898,\n",
       " 'central': 573,\n",
       " 'technicaldevelopers': 3871,\n",
       " 'tickets': 3935,\n",
       " 'followed': 1486,\n",
       " 'resolution': 3273,\n",
       " 'error': 1312,\n",
       " 'occurred': 2596,\n",
       " 'client': 630,\n",
       " 'drdo': 1161,\n",
       " 'description': 1043,\n",
       " 'signon': 3544,\n",
       " 'sso': 3675,\n",
       " 'property': 3015,\n",
       " 'control': 805,\n",
       " 'multiple': 2472,\n",
       " 'independent': 1882,\n",
       " 'software': 3593,\n",
       " 'user': 4105,\n",
       " 'logs': 2223,\n",
       " 'id': 1810,\n",
       " 'password': 2749,\n",
       " 'gain': 1552,\n",
       " 'connected': 758,\n",
       " 'different': 1093,\n",
       " 'usernames': 4108,\n",
       " 'passwords': 2750,\n",
       " 'netapp': 2532,\n",
       " 'maintenance': 2262,\n",
       " 'peopletool': 2791,\n",
       " 'handled': 1676,\n",
       " 'ib': 1804,\n",
       " 'clear': 625,\n",
       " 'cache': 502,\n",
       " 'weekly': 4255,\n",
       " 'psadmin': 3037,\n",
       " 'acs': 45,\n",
       " 'personal': 2809,\n",
       " 'details': 1063,\n",
       " 'profile': 2995,\n",
       " 'summary': 3781,\n",
       " 'years': 4330,\n",
       " 'implementing': 1848,\n",
       " 'upgrading': 4092,\n",
       " 'including': 1873,\n",
       " 'human': 1794,\n",
       " 'capital': 525,\n",
       " 'financials': 1459,\n",
       " 'solutions': 3600,\n",
       " 'portal': 2880,\n",
       " 'ihub': 1826,\n",
       " 'indepth': 1884,\n",
       " 'analysis': 162,\n",
       " 'implementation': 1843,\n",
       " 'stages': 3683,\n",
       " 'load': 2201,\n",
       " 'quality': 3094,\n",
       " 'assurance': 276,\n",
       " 'tuning': 4029,\n",
       " 'deploying': 1036,\n",
       " 'skilled': 3569,\n",
       " 'capability': 520,\n",
       " 'analyse': 159,\n",
       " 'interpret': 1985,\n",
       " 'unique': 4066,\n",
       " 'combination': 671,\n",
       " 'logical': 2217,\n",
       " 'thinking': 3922,\n",
       " 'right': 3327,\n",
       " 'core': 834,\n",
       " 'competencies': 703,\n",
       " 'install': 1931,\n",
       " 'configure': 749,\n",
       " 'upgrades': 4091,\n",
       " 'refreshes': 3203,\n",
       " 'cloning': 633,\n",
       " 'workflow': 4292,\n",
       " 'users': 4109,\n",
       " 'monitoring': 2431,\n",
       " 'log': 2212,\n",
       " 'bottleneck': 446,\n",
       " 'resetting': 3270,\n",
       " 'lockingunlocking': 2211,\n",
       " 'profiles': 2998,\n",
       " 'middle': 2386,\n",
       " 'tier': 3936,\n",
       " 'quarterly': 3098,\n",
       " 'apply': 207,\n",
       " 'fixes': 1471,\n",
       " 'sets': 3495,\n",
       " 'infrastructureiaas': 1911,\n",
       " 'managerlift': 2280,\n",
       " 'shift': 3517,\n",
       " 'idc': 1811,\n",
       " 'sol': 3594,\n",
       " 'clients': 632,\n",
       " 'texas': 3911,\n",
       " 'department': 1028,\n",
       " 'transportationtxdot': 4009,\n",
       " 'duration': 1177,\n",
       " 'aug': 293,\n",
       " 'till': 3940,\n",
       " 'dba': 960,\n",
       " 'responsibilities': 3285,\n",
       " 'performing': 2797,\n",
       " 'phire': 2822,\n",
       " 'patchingjava': 2757,\n",
       " 'wls': 4276,\n",
       " 'jdk': 2037,\n",
       " 'posting': 2889,\n",
       " 'resolved': 3276,\n",
       " 'developer': 1073,\n",
       " 'examining': 1338,\n",
       " 'clearing': 627,\n",
       " 'monthly': 2434,\n",
       " 'maintained': 2259,\n",
       " 'documentation': 1140,\n",
       " 'non': 2554,\n",
       " 'safalta': 3371,\n",
       " 'infotech': 1908,\n",
       " 'nov': 2570,\n",
       " 'uts': 4124,\n",
       " 'papa': 2721,\n",
       " 'jones': 2053,\n",
       " 'complete': 708,\n",
       " 'life': 2175,\n",
       " 'cycle': 934,\n",
       " 'scratch': 3426,\n",
       " 'golive': 1608,\n",
       " 'executing': 1353,\n",
       " 'demo': 1018,\n",
       " 'utility': 4116,\n",
       " 'administer': 78,\n",
       " 'create': 862,\n",
       " 'webserver': 4247,\n",
       " 'performed': 2796,\n",
       " 'resolutions': 3274,\n",
       " 'analysed': 160,\n",
       " 'defining': 995,\n",
       " 'gateway': 1563,\n",
       " 'entire': 1297,\n",
       " 'audit': 291,\n",
       " 'reviews': 3322,\n",
       " 'sys': 3821,\n",
       " 'ddd': 966,\n",
       " 'inconsistency': 1875,\n",
       " 'moves': 2454,\n",
       " 'balancing': 349,\n",
       " 'tiers': 3937,\n",
       " 'hands': 1680,\n",
       " 'pumdpk': 3057,\n",
       " 'download': 1153,\n",
       " 'executed': 1352,\n",
       " 'nprod': 2574,\n",
       " 'perfumed': 2798,\n",
       " 'verification': 4164,\n",
       " 'tests': 3910,\n",
       " 'patching': 2756,\n",
       " 'extracting': 1391,\n",
       " 'daytona': 958,\n",
       " 'dsc': 1172,\n",
       " 'algonquin': 132,\n",
       " 'college': 664,\n",
       " 'canada': 515,\n",
       " 'acceptance': 18,\n",
       " 'hcmfscmcspihub': 1699,\n",
       " 'batch': 372,\n",
       " 'connect': 757,\n",
       " 'purposes': 3071,\n",
       " 'stored': 3722,\n",
       " 'installconfigure': 1935,\n",
       " 'bundles': 492,\n",
       " 'maintain': 2258,\n",
       " 'customization': 922,\n",
       " 'enable': 1254,\n",
       " 'tracing': 3974,\n",
       " 'page': 2714,\n",
       " 'pre': 2908,\n",
       " 'verity': 4166,\n",
       " 'db': 959,\n",
       " 'nt': 2578,\n",
       " 'admindba': 75,\n",
       " 'multitasking': 2474,\n",
       " 'effective': 1207,\n",
       " 'player': 2858,\n",
       " 'customers': 919,\n",
       " 'selfmotivated': 3454,\n",
       " 'quick': 3107,\n",
       " 'learner': 2151,\n",
       " 'bachelors': 337,\n",
       " 'anil': 180,\n",
       " 'neerukonda': 2522,\n",
       " 'institute': 1943,\n",
       " 'sciences': 3418,\n",
       " 'andhra': 175,\n",
       " 'awards': 320,\n",
       " 'delight': 1006,\n",
       " 'award': 318,\n",
       " 'sport': 3641,\n",
       " 'csat': 895,\n",
       " 'score': 3423,\n",
       " 'fathers': 1423,\n",
       " 'g': 1549,\n",
       " 'ananda': 171,\n",
       " 'rayudu': 3140,\n",
       " 'marital': 2310,\n",
       " 'pan': 2718,\n",
       " 'passport': 2748,\n",
       " 'administrator': 83,\n",
       " 'gangareddy': 1558,\n",
       " 'objective': 2589,\n",
       " 'utilizing': 4121,\n",
       " 'talent': 3838,\n",
       " 'keeping': 2094,\n",
       " 'abreast': 7,\n",
       " 'advancement': 93,\n",
       " 'derive': 1040,\n",
       " 'utmost': 4122,\n",
       " 'successful': 3770,\n",
       " 'separate': 3472,\n",
       " 'host': 1756,\n",
       " 'strategy': 3728,\n",
       " 'ensured': 1291,\n",
       " 'availability': 312,\n",
       " 'failover': 1405,\n",
       " 'spreading': 3643,\n",
       " 'hosts': 1759,\n",
       " 'processing': 2975,\n",
       " 'objects': 2593,\n",
       " 'packs': 2713,\n",
       " 'sqr': 3652,\n",
       " 'engine': 1271,\n",
       " 'index': 1885,\n",
       " 'creation': 868,\n",
       " 'refreshing': 3204,\n",
       " 'qa': 3083,\n",
       " 'migrating': 2389,\n",
       " 'evaluating': 1329,\n",
       " 'required': 3260,\n",
       " 'timely': 3945,\n",
       " 'basis': 371,\n",
       " 'jobs': 2046,\n",
       " 'taking': 3837,\n",
       " 'scheduled': 3404,\n",
       " 'backup': 342,\n",
       " 'rman': 3333,\n",
       " 'regular': 3212,\n",
       " 'offline': 2615,\n",
       " 'backups': 344,\n",
       " 'exp': 1360,\n",
       " 'imp': 1839,\n",
       " 'datapump': 949,\n",
       " 'successfully': 3771,\n",
       " 'applied': 205,\n",
       " 'erp': 1311,\n",
       " 'package': 2707,\n",
       " 'hrmsfscmcrmcshcmportal': 1778,\n",
       " 'versions': 4174,\n",
       " 'bea': 381,\n",
       " 'mssql': 2465,\n",
       " 'operating': 2638,\n",
       " 'rhel': 3325,\n",
       " 'oel': 2605,\n",
       " 'emergtech': 1241,\n",
       " 'business': 494,\n",
       " 'amerit': 155,\n",
       " 'fleet': 1475,\n",
       " 'sep': 3471,\n",
       " 'administratordba': 84,\n",
       " 'crm': 882,\n",
       " 'elm': 1231,\n",
       " 'pt': 3044,\n",
       " 'kc': 2089,\n",
       " 'services': 3487,\n",
       " 'daily': 936,\n",
       " 'proactive': 2964,\n",
       " 'members': 2360,\n",
       " 'managerpum': 2281,\n",
       " 'purpose': 3070,\n",
       " 'updated': 4083,\n",
       " 'updatesfixespatches': 4086,\n",
       " 'kept': 2096,\n",
       " 'record': 3173,\n",
       " 'dbs': 962,\n",
       " 'webservers': 4249,\n",
       " 'extensively': 1387,\n",
       " 'delete': 1002,\n",
       " 'monitored': 2430,\n",
       " 'queue': 3105,\n",
       " 'changed': 590,\n",
       " 'psappsrvcfgpsprcscfg': 3039,\n",
       " 'file': 1448,\n",
       " 'specific': 3627,\n",
       " 'checked': 599,\n",
       " 'cleared': 626,\n",
       " 'trace': 3972,\n",
       " 'levels': 2164,\n",
       " 'logfence': 2213,\n",
       " 'parameter': 2726,\n",
       " 'section': 3442,\n",
       " 'appsrvcfg': 222,\n",
       " 'code': 647,\n",
       " 'ae': 95,\n",
       " 'psappservcfg': 3038,\n",
       " 'psprcscfg': 3042,\n",
       " 'care': 533,\n",
       " 'adding': 61,\n",
       " 'additional': 62,\n",
       " 'groups': 1646,\n",
       " 'mapping': 2299,\n",
       " 'dr': 1159,\n",
       " 'snap': 3585,\n",
       " 'shots': 3527,\n",
       " 'mutli': 2483,\n",
       " 'factor': 1400,\n",
       " 'authentication': 296,\n",
       " 'processes': 2973,\n",
       " 'helping': 1718,\n",
       " 'terms': 3903,\n",
       " 'gaining': 1554,\n",
       " 'executions': 1355,\n",
       " 'document': 1139,\n",
       " 'preparation': 2916,\n",
       " 'steps': 3713,\n",
       " 'murali': 2479,\n",
       " 'infrastructure': 1910,\n",
       " 'manually': 2293,\n",
       " 'dpks': 1158,\n",
       " 'installations': 1933,\n",
       " 'capi': 524,\n",
       " 'stat': 3700,\n",
       " 'ssoimplementation': 3676,\n",
       " 'compare': 698,\n",
       " 'workstation': 4303,\n",
       " 'developers': 1074,\n",
       " 'testers': 3908,\n",
       " 'modules': 2422,\n",
       " 'sending': 3462,\n",
       " 'messages': 2372,\n",
       " 'dddaudit': 967,\n",
       " 'sysaudit': 3822,\n",
       " 'integrity': 1957,\n",
       " 'checks': 602,\n",
       " 'ren': 3235,\n",
       " 'settings': 3497,\n",
       " 'career': 534,\n",
       " 'sembcorp': 3460,\n",
       " 'brazil': 458,\n",
       " 'active': 51,\n",
       " 'fields': 1447,\n",
       " 'additionally': 63,\n",
       " 'enhancement': 1282,\n",
       " 'responsibility': 3286,\n",
       " 'assigning': 264,\n",
       " 'roles': 3345,\n",
       " 'privileges': 2959,\n",
       " 'debugging': 976,\n",
       " 'resolving': 3277,\n",
       " 'serverweb': 3483,\n",
       " 'serverprocess': 3481,\n",
       " 'weeklymonthly': 4256,\n",
       " 'maintains': 2261,\n",
       " 'wipro': 4270,\n",
       " 'ind': 1880,\n",
       " 'modifying': 2419,\n",
       " 'mover': 2453,\n",
       " 'reporting': 3248,\n",
       " 'interacting': 1964,\n",
       " 'title': 3952,\n",
       " 'asg': 253,\n",
       " 'usa': 4098,\n",
       " 'aix': 125,\n",
       " 'statcapi': 3701,\n",
       " 'hyderabad': 1798,\n",
       " 'overall': 2695,\n",
       " 'hrmsfscm': 1777,\n",
       " 'deterministic': 1067,\n",
       " 'approach': 216,\n",
       " 'problem': 2966,\n",
       " 'solving': 3604,\n",
       " 'proficient': 2994,\n",
       " 'graduated': 1627,\n",
       " 'electronics': 1226,\n",
       " 'mvgr': 2487,\n",
       " 'vizianagaramjntuk': 4202,\n",
       " 'aggregate': 105,\n",
       " 'achieved': 35,\n",
       " 'marks': 2315,\n",
       " 'standard': 3690,\n",
       " 'scored': 3424,\n",
       " 'awarded': 319,\n",
       " 'bravo': 457,\n",
       " 'pat': 2753,\n",
       " 'techahindra': 3866,\n",
       " 'associate': 273,\n",
       " 'month': 2433,\n",
       " 'innovator': 1921,\n",
       " 'time': 3941,\n",
       " 'spot': 3642,\n",
       " 'capgemini': 523,\n",
       " 'respective': 3282,\n",
       " 'axa': 329,\n",
       " 'consultant': 779,\n",
       " 'cognizant': 654,\n",
       " 'technol': 3877,\n",
       " 'ogy': 2617,\n",
       " 'ut': 4114,\n",
       " 'ions': 2005,\n",
       " 'augus': 294,\n",
       " 'april': 224,\n",
       " 'voya': 4215,\n",
       " 'financial': 1458,\n",
       " 'insurance': 1945,\n",
       " 'deals': 973,\n",
       " 'tech': 3865,\n",
       " 'mahindra': 2253,\n",
       " 'limit': 2182,\n",
       " 'ed': 1196,\n",
       " 'july': 2068,\n",
       " 'hr': 1771,\n",
       " 'fin': 1454,\n",
       " 'includes': 1872,\n",
       " 'interfaces': 1972,\n",
       " 'thirdparty': 3923,\n",
       " 'live': 2197,\n",
       " 'cio': 610,\n",
       " 'engineer': 1272,\n",
       " 'solaris': 3596,\n",
       " 'administrative': 82,\n",
       " 'supports': 3800,\n",
       " 'version': 4172,\n",
       " 'financialsscm': 1460,\n",
       " 'indexes': 1887,\n",
       " 'tables': 3828,\n",
       " 'master': 2322,\n",
       " 'disk': 1124,\n",
       " 'bouncing': 447,\n",
       " 'schedulers': 3406,\n",
       " 'prerefresh': 2922,\n",
       " 'activity': 54,\n",
       " 'recompilation': 3171,\n",
       " 'cobol': 646,\n",
       " 'codes': 649,\n",
       " 'handson': 1681,\n",
       " 'aware': 321,\n",
       " 'udm': 4049,\n",
       " 'transfer': 3995,\n",
       " 'gnupg': 1600,\n",
       " 'keys': 2099,\n",
       " 'toad': 3956,\n",
       " 'sqldeveloper': 3648,\n",
       " 'microsoft': 2385,\n",
       " 'studio': 3744,\n",
       " 'filezilla': 1450,\n",
       " 'winscp': 4269,\n",
       " 'pcomm': 2775,\n",
       " 'silva': 3547,\n",
       " 'certifications': 580,\n",
       " 'architect': 232,\n",
       " 'varkala': 4145,\n",
       " 'vikas': 4183,\n",
       " 'total': 3963,\n",
       " 'hope': 1752,\n",
       " 'skill': 3568,\n",
       " 'value': 4134,\n",
       " 'aid': 119,\n",
       " 'companys': 697,\n",
       " 'objectives': 2590,\n",
       " 'anticipating': 189,\n",
       " 'needs': 2520,\n",
       " 'interests': 1970,\n",
       " 'motivations': 2444,\n",
       " 'deliver': 1007,\n",
       " 'budget': 478,\n",
       " 'delivering': 1011,\n",
       " 'improving': 1861,\n",
       " 'agility': 110,\n",
       " 'driving': 1168,\n",
       " 'hardware': 1686,\n",
       " 'disaster': 1114,\n",
       " 'recovery': 3176,\n",
       " 'https': 1789,\n",
       " 'health': 1707,\n",
       " 'check': 598,\n",
       " 'depth': 1039,\n",
       " 'socket': 3590,\n",
       " 'layer': 2139,\n",
       " 'proficiency': 2993,\n",
       " 'reconfiguration': 3172,\n",
       " 'generating': 1576,\n",
       " 'precompare': 2909,\n",
       " 'station': 3708,\n",
       " 'locking': 2210,\n",
       " 'unlocking': 4078,\n",
       " 'accounts': 32,\n",
       " 'taxupdates': 3855,\n",
       " 'internal': 1978,\n",
       " 'consistency': 766,\n",
       " 'alteraudit': 144,\n",
       " 'periodically': 2802,\n",
       " 'regularly': 3213,\n",
       " 'compilation': 706,\n",
       " 'guard': 1656,\n",
       " 'proven': 3024,\n",
       " 'contributor': 804,\n",
       " 'area': 240,\n",
       " 'educational': 1203,\n",
       " 'qualification': 3090,\n",
       " 'bsc': 472,\n",
       " 'osmania': 2682,\n",
       " 'progile': 3003,\n",
       " 'hartford': 1692,\n",
       " 'ct': 900,\n",
       " 'adminpeoplesoft': 85,\n",
       " 'daytoday': 957,\n",
       " 'build': 482,\n",
       " 'proper': 3013,\n",
       " 'object': 2588,\n",
       " 'involving': 2003,\n",
       " 'building': 486,\n",
       " 'permission': 2804,\n",
       " 'lists': 2196,\n",
       " 'granting': 1628,\n",
       " 'vms': 4204,\n",
       " 'accessing': 22,\n",
       " 'administrating': 80,\n",
       " 'raised': 3117,\n",
       " 'node': 2551,\n",
       " 'configurations': 748,\n",
       " 'downloading': 1154,\n",
       " 'customizations': 923,\n",
       " 'appling': 206,\n",
       " 'balancer': 347,\n",
       " 'clustering': 643,\n",
       " 'setups': 3499,\n",
       " 'high': 1723,\n",
       " 'pump': 3058,\n",
       " 'utilities': 4115,\n",
       " 'synchronisation': 3815,\n",
       " 'documenting': 1143,\n",
       " 'feedback': 1436,\n",
       " 'vivekanand': 4201,\n",
       " 'sayana': 3392,\n",
       " 'valid': 4127,\n",
       " 'epm': 1307,\n",
       " 'implementations': 1845,\n",
       " 'deploy': 1034,\n",
       " 'interpersonal': 1983,\n",
       " ...}"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "requiredText = Resume[\"CV\"]\n",
    "requiredTarget = Resume[\"Encoded_Skill\"].values\n",
    "Countvectorizer=CountVectorizer(analyzer='word',token_pattern=r'\\w{1,}',stop_words = 'english')\n",
    "bag = Countvectorizer.fit_transform(requiredText)\n",
    "Countvectorizer.vocabulary_"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b60215d3",
   "metadata": {},
   "source": [
    "#### VECTORIZATION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e609488",
   "metadata": {},
   "source": [
    "#### COUNT VECTORIZER tells the frequency of a word."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7e37fd69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>term</th>\n",
       "      <th>occurrences</th>\n",
       "      <th>frequency</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>aa</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>abdul</td>\n",
       "      <td>2</td>\n",
       "      <td>0.000057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>abilities</td>\n",
       "      <td>4</td>\n",
       "      <td>0.000114</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>abilitiescommunication</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ability</td>\n",
       "      <td>37</td>\n",
       "      <td>0.001053</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4428</th>\n",
       "      <td>òpaper</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4429</th>\n",
       "      <td>òposter</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4430</th>\n",
       "      <td>ôbroadband</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4431</th>\n",
       "      <td>þnding</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4432</th>\n",
       "      <td>þts</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000028</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>4433 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                        term  occurrences  frequency\n",
       "0                         aa            1   0.000028\n",
       "1                      abdul            2   0.000057\n",
       "2                  abilities            4   0.000114\n",
       "3     abilitiescommunication            1   0.000028\n",
       "4                    ability           37   0.001053\n",
       "...                      ...          ...        ...\n",
       "4428                  òpaper            1   0.000028\n",
       "4429                 òposter            1   0.000028\n",
       "4430              ôbroadband            1   0.000028\n",
       "4431                  þnding            1   0.000028\n",
       "4432                     þts            1   0.000028\n",
       "\n",
       "[4433 rows x 3 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vectorizer1 = CountVectorizer(min_df = 1, max_df = 0.9)\n",
    "count_vect = vectorizer1.fit_transform(Resume[\"CV\"])\n",
    "word_freq_df = pd.DataFrame({'term': vectorizer1.get_feature_names(), 'occurrences':np.asarray(count_vect.sum(axis=0)).ravel().tolist()})\n",
    "word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])\n",
    "word_freq_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1eb490d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Density'>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(x =[word_freq_df['frequency']])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6749a54b",
   "metadata": {},
   "source": [
    "#### TFIDF - Term frequency inverse Document Frequency"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "fe937dad",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from scipy.sparse import hstack\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "0d7785ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',max_features=1500)\n",
    "word_vectorizer.fit(requiredText)\n",
    "WordFeatures = word_vectorizer.transform(requiredText)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b55f1e6",
   "metadata": {},
   "source": [
    "### Model Building || Model Training || Model Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72ee0242",
   "metadata": {},
   "source": [
    "#### DATA PREPARATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "a38b1290",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X Train shape: (63, 1500)\n",
      "Y Train shape: (63,)\n",
      "x Test shape: (16, 1500)\n",
      "y Test shape: (16,)\n"
     ]
    }
   ],
   "source": [
    "x_train,x_test,y_train,y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)\n",
    "print(\"X Train shape:\",x_train.shape)\n",
    "print(\"Y Train shape:\",y_train.shape)\n",
    "print(\"x Test shape:\",x_test.shape)\n",
    "print(\"y Test shape:\",y_test.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "823011f0",
   "metadata": {},
   "source": [
    "### 1. LOGISTIC REGRESSION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d2818abf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPORTING NECESSARY LIBRARIES FOR LOGISTIC REGRESSION\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "fe8a769d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ACCURACY OF TRAIN DATA: 0.9682539682539683\n",
      "ACCURACY OF TEST DATA: 0.9375\n",
      "CLASSIFICATION REPORT:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           1       0.50      1.00      0.67         1\n",
      "           2       1.00      1.00      1.00         4\n",
      "           3       1.00      0.80      0.89         5\n",
      "           4       1.00      1.00      1.00         6\n",
      "\n",
      "    accuracy                           0.94        16\n",
      "   macro avg       0.88      0.95      0.89        16\n",
      "weighted avg       0.97      0.94      0.94        16\n",
      "\n"
     ]
    }
   ],
   "source": [
    "logistic_classifier = LogisticRegression()\n",
    "logistic_classifier.fit(x_train,y_train)\n",
    "\n",
    "#Predicting on Training Data\n",
    "pred_train_log = logistic_classifier.predict(x_train)\n",
    "#Accuracy On Train Data\n",
    "train_acc_log = np.mean(pred_train_log==y_train)\n",
    "print(\"ACCURACY OF TRAIN DATA:\", train_acc_log)\n",
    "\n",
    "#Predicting on Test Data\n",
    "pred_test_log = logistic_classifier.predict(x_test)\n",
    "#Accuracy On Test Data\n",
    "test_acc_log = np.mean(pred_test_log==y_test)\n",
    "print(\"ACCURACY OF TEST DATA:\",test_acc_log )\n",
    "\n",
    "#Confusion Matrix\n",
    "logistic_cm = confusion_matrix(y_test,pred_test_log)\n",
    "\n",
    "#Classification Report\n",
    "print(\"CLASSIFICATION REPORT:\\n\", classification_report(y_test,pred_test_log))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1c5e442",
   "metadata": {},
   "source": [
    "### 2. DECISION TREE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "3bcc12ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPORTING NECESSARY LIBRARIES FOR DECISION TREE\n",
    "from sklearn import tree\n",
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "d2b0f3dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ACCURACY OF TRAIN DATA: 0.8253968253968254\n",
      "ACCURACY OF TEST DATA: 0.6875\n",
      "CLASSIFICATION REPORT:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           1       1.00      1.00      1.00         1\n",
      "           2       0.44      1.00      0.62         4\n",
      "           3       0.00      0.00      0.00         5\n",
      "           4       1.00      1.00      1.00         6\n",
      "\n",
      "    accuracy                           0.69        16\n",
      "   macro avg       0.61      0.75      0.65        16\n",
      "weighted avg       0.55      0.69      0.59        16\n",
      "\n"
     ]
    }
   ],
   "source": [
    "DT = DecisionTreeClassifier()\n",
    "DT_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=2)\n",
    "DT_classifier.fit(x_train,y_train)\n",
    "\n",
    "#Predicting on Train Data\n",
    "pred_train_dt = DT_classifier.predict(x_train)\n",
    "#Accuracy On Train Data\n",
    "train_acc_dt = np.mean(pred_train_dt==y_train)\n",
    "print(\"ACCURACY OF TRAIN DATA:\",train_acc_dt )\n",
    "\n",
    "#Predicting on Test Data\n",
    "pred_test_dt = DT_classifier.predict(x_test)\n",
    "#Accuracy on Test Data\n",
    "test_acc_dt = np.mean(pred_test_dt==y_test)\n",
    "print(\"ACCURACY OF TEST DATA:\",test_acc_dt )\n",
    "\n",
    "#Confusion Matrix\n",
    "dt_cm = confusion_matrix(y_test,pred_test_dt)\n",
    "\n",
    "#Classification Report\n",
    "print(\"CLASSIFICATION REPORT:\\n\", classification_report(y_test,pred_test_dt))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d6d58de7",
   "metadata": {},
   "source": [
    "### 3. RANDOM FOREST "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1153c9bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPORTING NECESSARY LIBRARIES FOR RANDOM FOREST\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "3826949b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ACCURACY OF TRAIN DATA: 1.0\n",
      "ACCURACY OF TEST DATA: 1.0\n",
      "CLASSIFICATION REPORT:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           1       1.00      1.00      1.00         1\n",
      "           2       1.00      1.00      1.00         4\n",
      "           3       1.00      1.00      1.00         5\n",
      "           4       1.00      1.00      1.00         6\n",
      "\n",
      "    accuracy                           1.00        16\n",
      "   macro avg       1.00      1.00      1.00        16\n",
      "weighted avg       1.00      1.00      1.00        16\n",
      "\n"
     ]
    }
   ],
   "source": [
    "RF = {'n_estimators':15,'class_weight': \"balanced\",'n_jobs':-1,'random_state':42}\n",
    "RF_classifier = RandomForestClassifier(**RF)\n",
    "RF_classifier.fit(x_train,y_train)\n",
    "\n",
    "#Predicting on Train Data\n",
    "pred_train_rf = RF_classifier.predict(x_train)\n",
    "#Accuracy On Train Data\n",
    "train_acc_rf = np.mean(pred_train_rf==y_train)\n",
    "print(\"ACCURACY OF TRAIN DATA:\",train_acc_rf)\n",
    "\n",
    "#Predicting on Test Data\n",
    "pred_test_rf = RF_classifier.predict(x_test)\n",
    "#Accuracy On Test Data\n",
    "test_acc_rf = np.mean(pred_test_rf==y_test)\n",
    "print(\"ACCURACY OF TEST DATA:\",test_acc_rf )\n",
    "\n",
    "#Confusion Matrix\n",
    "rf_cm = confusion_matrix(y_test,pred_test_rf)\n",
    "\n",
    "#Classification Report\n",
    "print(\"CLASSIFICATION REPORT:\\n\", classification_report(y_test,pred_test_rf))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f50088e0",
   "metadata": {},
   "source": [
    "### 4. MULTINOMIAL NAVIE BAYES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "3e7b3d4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPORTING NECESSARY LIBRARIES FOR MULTINOMIAL NAVIE BAYES\n",
    "from sklearn.naive_bayes import MultinomialNB as MB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "7c36c3db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ACCURACY OF TRAIN DATA: 0.9682539682539683\n",
      "ACCURACY OF TEST DATA: 0.875\n",
      "CLASSIFICATION REPORT:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           1       0.50      1.00      0.67         1\n",
      "           2       0.80      1.00      0.89         4\n",
      "           3       1.00      0.60      0.75         5\n",
      "           4       1.00      1.00      1.00         6\n",
      "\n",
      "    accuracy                           0.88        16\n",
      "   macro avg       0.82      0.90      0.83        16\n",
      "weighted avg       0.92      0.88      0.87        16\n",
      "\n"
     ]
    }
   ],
   "source": [
    "classifier_mb = MB()\n",
    "classifier_mb.fit(x_train,y_train)\n",
    "\n",
    "#Predicting On Train Data\n",
    "pred_train_mb = classifier_mb.predict(x_train)\n",
    "#Accuracy On Train Data\n",
    "train_acc_mb = np.mean(pred_train_mb==y_train)\n",
    "print(\"ACCURACY OF TRAIN DATA:\", train_acc_mb)\n",
    "\n",
    "#Predicting On Test Data\n",
    "pred_test_mb = classifier_mb.predict(x_test)\n",
    "#Accuracy On Test Data\n",
    "test_acc_mb = np.mean(pred_test_mb==y_test)\n",
    "print(\"ACCURACY OF TEST DATA:\", test_acc_mb)\n",
    "\n",
    "#Confusion Matrix\n",
    "mb_cm = confusion_matrix(y_test,pred_test_mb)\n",
    "\n",
    "#Classification Report\n",
    "print(\"CLASSIFICATION REPORT:\\n\", classification_report(y_test,pred_test_mb))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79451de4",
   "metadata": {},
   "source": [
    "### 5. SUPPORT VECTOR MACHINE "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a42b689f",
   "metadata": {},
   "outputs": [],
   "source": [
    "##IMPORTING NECESSARY LIBRARIES FOR SUPPORT VECTOR MACHINE\n",
    "from sklearn import svm\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "da25299d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ACCURACY OF TRAIN DATA: 1.0\n",
      "ACCURACY OF TEST DATA: 1.0\n",
      "CLASSIFICATION REPORT:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           1       1.00      1.00      1.00         1\n",
      "           2       1.00      1.00      1.00         4\n",
      "           3       1.00      1.00      1.00         5\n",
      "           4       1.00      1.00      1.00         6\n",
      "\n",
      "    accuracy                           1.00        16\n",
      "   macro avg       1.00      1.00      1.00        16\n",
      "weighted avg       1.00      1.00      1.00        16\n",
      "\n"
     ]
    }
   ],
   "source": [
    "svm_classifier = (SVC(kernel='linear'))\n",
    "svm_classifier.fit(x_train,y_train)\n",
    "\n",
    "#Predicting On Train Data\n",
    "pred_train_svm = svm_classifier.predict(x_train)\n",
    "#Accuracy On Train Data\n",
    "train_acc_svm = np.mean(pred_train_svm==y_train)\n",
    "print(\"ACCURACY OF TRAIN DATA:\",train_acc_svm )\n",
    "\n",
    "#Prediciting On Test Data\n",
    "pred_test_svm = svm_classifier.predict(x_test)\n",
    "#Accuracy On Test Data\n",
    "test_acc_svm = np.mean(pred_test_svm==y_test)\n",
    "print(\"ACCURACY OF TEST DATA:\",test_acc_svm)\n",
    "\n",
    "#Confusion Matrix\n",
    "svm_cm = confusion_matrix(y_test,pred_test_svm)\n",
    "\n",
    "#Classification Report\n",
    "print(\"CLASSIFICATION REPORT:\\n\", classification_report(y_test,pred_test_svm))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "8614f593",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x504 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,7))\n",
    "\n",
    "plt.suptitle(\"Confusion Matrixes\", fontsize=12)\n",
    "\n",
    "plt.subplot(2,3,1)\n",
    "plt.title(\"LOGISTIC REGRESSION\")\n",
    "sns.heatmap(logistic_cm, cbar=False, annot=True, cmap=\"YlGnBu\",  fmt=\"d\")\n",
    "\n",
    "plt.subplot(2,3,2)\n",
    "plt.title(\"DECISION TREE\")\n",
    "sns.heatmap(dt_cm, cbar=False, annot=True, cmap=\"YlGnBu\", fmt=\"d\")\n",
    "\n",
    "plt.subplot(2,3,3)\n",
    "plt.title(\"RANDOM FOREST CLASSIFICATION\")\n",
    "sns.heatmap(rf_cm, cbar=False, annot=True, cmap=\"YlGnBu\", fmt=\"d\")\n",
    "\n",
    "plt.subplot(2,3,4)\n",
    "plt.title(\"NaiveBayes Classification\")\n",
    "sns.heatmap(mb_cm, cbar=False, annot=True, cmap=\"YlGnBu\", fmt=\"d\")\n",
    "\n",
    "plt.subplot(2,3,5)\n",
    "plt.title(\"SVM Classification\")\n",
    "sns.heatmap(svm_cm, cbar=False, annot=True, cmap=\"YlGnBu\",  fmt=\"d\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "737d1076",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>TN</th>\n",
       "      <th>FP</th>\n",
       "      <th>FN</th>\n",
       "      <th>TP</th>\n",
       "      <th>Accuracy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Logistic Regression</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0.9375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Decision Tree</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0.6875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Multinomial NaiveBayes</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0.8750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Support Vector Machine</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1.0000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                    Model  TN  FP  FN  TP  Accuracy\n",
       "0     Logistic Regression   1   0   0   4    0.9375\n",
       "1           Decision Tree   1   0   0   4    0.6875\n",
       "2           Random Forest   1   0   0   4    1.0000\n",
       "3  Multinomial NaiveBayes   1   0   0   4    0.8750\n",
       "4  Support Vector Machine   1   0   0   4    1.0000"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "TN = [logistic_cm[0,0], dt_cm[0,0], rf_cm[0,0], mb_cm[0,0], svm_cm[0,0]]\n",
    "FP = [logistic_cm[0,1], dt_cm[0,1], rf_cm[0,1], mb_cm[0,1], svm_cm[0,1]]\n",
    "FN = [logistic_cm[1,0], dt_cm[1,0], rf_cm[1,0], mb_cm[1,0], svm_cm[1,0]]\n",
    "TP = [logistic_cm[1,1], dt_cm[1,1], rf_cm[1,1], mb_cm[1,1], svm_cm[1,1]]\n",
    "Accuracy = [test_acc_log, test_acc_dt, test_acc_rf, test_acc_mb, test_acc_svm]\n",
    "#MSE = [lr_mse, knn_mse, svm_mse, nb_mse, dt_mse, rf_mse]\n",
    "Classification = [\"Logistic Regression\", \"Decision Tree\", \"Random Forest\", \"Multinomial NaiveBayes\", \n",
    "                  \"Support Vector Machine\"]\n",
    "list_matrix = [Classification, TN, FP, FN, TP, Accuracy]\n",
    "list_headers = [\"Model\", \"TN\", \"FP\", \"FN\", \"TP\", \"Accuracy\"]\n",
    "zipped = list(zip(list_headers, list_matrix))\n",
    "data_dict = dict(zipped)\n",
    "df_1=pd.DataFrame(data_dict)\n",
    "df_1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "ca4df616",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Accuracy\n",
    "plt.figure(figsize=(10,4))\n",
    "ax= sns.barplot(x=df_1.Model, y=df_1.Accuracy, palette =sns.color_palette(\"Set2\") )\n",
    "ax.set_xticklabels(ax.get_xticklabels(),rotation=30)\n",
    "plt.xlabel('Classification Models')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Accuracy Scores of Classification Models')\n",
    "for i in ax.patches:\n",
    "    ax.text(i.get_x()+.19, i.get_height()-0.3, \\\n",
    "            str(round((i.get_height()), 4)), fontsize=15, color='b')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ac020f7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
