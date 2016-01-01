
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#include <libgen.h>
// PATH_MAX
#include <limits.h>
#ifdef PATH_MAX
  #define MAX_STRING PATH_MAX // this version is portable to different platforms. http://stackoverflow.com/questions/4109638/what-is-the-safe-alternative-to-realpath 
#else
  #define MAX_STRING 1000
#endif

#define MAX_LANGS 20
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENT_LEN 20000
#define MAX_WORD_PER_SENT 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

// training structure, useful when training embeddings for multiple languages
struct pair_params **all_pairs; //malloc'd in main
struct lang_params **all_langs; //malloc'd in main
char **language_indices;
struct pair_params *pair; //current pair being used by TrainModelThread

//struct for all info related to a language: vocab, output file, vectors
struct lang_params {
  char lang_name[MAX_STRING];
  char output_file[MAX_STRING];
  char vocab_file[MAX_STRING];
  char config_file[MAX_STRING];
  struct vocab_word *vocab;
  struct vocab_word *backup_vocab;
  int *vocab_hash;

  // syn0: input embeddings (exist for both hierarchical softmax and negative sampling)
  // syn1: output embeddings (hierarchical softmax)
  // syn1neg: output embeddings (negative sampling)
  // table, vocab_size corresponds to the output side.
  long long vocab_max_size, vocab_size, total_words;
  real *syn0, *syn1, *syn1neg;
  int *table;
  int full_vocab; //set to 1 once all training files have been read and vocab is complete

  long long unk_id; // index of the <unk> word

  //pointers to file-related structs
  int num_files;
  struct file_params **files; //file structs for lang
};

//struct for training file info
struct file_params {
  char train_file[MAX_STRING];
  
  long long file_size;

  struct lang_params *lang;

  long long num_lines; //number of lines
  long long *line_blocks; //offsets in each file for each thread (indexed by thread #)
  long long train_words; //number of tokens in training file
  long long word_count_actual; //current progress in training file
};

//struct for grouping specific language pairs, alignment info
struct pair_params {
  struct file_params *src;
  struct file_params *tgt;

  // alignment info
  char align_file[MAX_STRING];
  long long align_num_lines;
  long long *align_line_blocks;
};

//looping over languages
int ll1;

//looping over language pairs
int lp1;

int binary = 0, debug_mode = 2, min_count = 5, num_threads = 1, min_reduce = 1;
long long layer1_size = 100;
long long classes = 0;

clock_t start;
char prefix[MAX_STRING];
char output_prefix[MAX_STRING]; // output_prefix.lang: stores embeddings
int eval_opt = 0; // evaluation option
long long specified_train_words = 0;

// cbow or skipgram
int cbow = 1, window = 5;

// hierarchical softmax or negative sampling
int hs = 0, negative = 5;
real *expTable;
const int table_size = 1e8;

// training epoch & learning rate
int num_train_iters = 1, cur_iter = 0, start_iter = 0; // run multiple iterations
real alpha = 0.025, starting_alpha;

// monolingual embeddings
real sample = 1e-4;

/** For bilingual embeddings **/
int align_debug = 0;
int align_opt = 0;

real bi_weight = 4.0; // how much we weight the crosslingual predictions. 4 by default, according to Luong et al. 2015
real bi_alpha; // learning rate for crosslingual predictions, set to alpha * bi_weight;
/** End For bilingual embeddings **/


/** For multilingual embeddings **/

int is_multi = 0;
int num_languages = 0;
int num_pairs = 0;
real samples[MAX_LANGS];

//bi weight also applies to multilingual case
/** End For multilingual embeddings**/

int global_debug_flag = 0;

/** Debugging code **/
// print stat of a real array
void print_real_array(real* a_syn, long long num_elements, char* name){
  float min = 1000000;
  float max = -1000000;
  float avg = 0;
  long long i;
  for(i=0; i<num_elements; ++i){
    if (a_syn[i]>max) max = a_syn[i];
    if (a_syn[i]<min) min = a_syn[i];
    avg += a_syn[i];
  }
  avg /= num_elements;
  printf("%s: min=%f, max=%f, avg=%f\n", name, min, max, avg);
}

// print stats of input and output embeddings
void print_model_stat(struct lang_params *params){
  printf("# model stats:\n");
  print_real_array(params->syn0, params->vocab_size * layer1_size, (char*) "  syn0");
  if (hs) print_real_array(params->syn1, params->vocab_size * layer1_size, (char*) "  syn1");
  if (negative) print_real_array(params->syn1neg, params->vocab_size * layer1_size, (char*) "  syn1neg");
}

// print a sent
void print_sent(long long* sent, int sent_len, struct vocab_word* vocab, char* name){
  int i;
  char buf[MAX_SENT_LEN];
  char token[MAX_STRING];
  sprintf(buf, "%s ", name);
  for(i=0; i<sent_len; i++) {
    if(i<(sent_len-1)) {
      sprintf(token, "%s ", vocab[sent[i]].word);
      strcat(buf, token);
    } else {
      sprintf(token, "%s\n", vocab[sent[i]].word);
      strcat(buf, token);
    }
  }
  printf("%s", buf);
  fflush(stdout);
}


void BackupVocab(struct lang_params * params) {
  int a;
  printf("Vocab size is %lld\n", params->vocab_size);
  params->backup_vocab = (struct vocab_word *)calloc(params->vocab_max_size, sizeof(struct vocab_word));
  for (a = 0; a < params->vocab_size; a++) {
    memcpy(&params->backup_vocab[a], &params->vocab[a], sizeof(struct vocab_word));
    printf("Vocab word %s == %s        \r", params->vocab[a].word, params->backup_vocab[a].word);
    if (strcmp(params->vocab[a].word, params->backup_vocab[a].word)) {
      printf("Words %s and %s not equal at position %d\n", params->vocab[a].word, params->backup_vocab[a].word, a);
      return;
    }
  }
  printf("\nFinished backing up %s vocab. Press enter to continue: \n", params->lang_name);
}

void CompareBackupVocab(struct lang_params * params) {
  int a;
  printf("Vocab size is %lld\n", params->vocab_size);
  for (a = 0; a < params->vocab_size; a++) {
    if (strcmp(params->vocab[a].word, params->backup_vocab[a].word)) {
      printf("Words %s and %s not equal at position %d\n", params->vocab[a].word, params->backup_vocab[a].word, a);
      return;
    }
  }
  printf("\nFinished comparing %s vocab to backup; all words are equal. Press enter to continue: \n", params->lang_name);
}  

void CheckVocab(struct lang_params * params) {
  int a;
  printf("Vocab size is %lld\n", params->vocab_size);
  for (a = 0; a < params->vocab_size; a++) {
    printf("Vocab word #%d, ", a);
    printf("word is %s, ", params->vocab[a].word);
    printf("count (cn) is %lld, ", params->vocab[a].cn);
    printf("point (relevant to sorting) is %d, ", *(params->vocab[a].point));
    printf("code is %s, ", params->vocab[a].code);
    printf("codelen is %d. ", params->vocab[a].codelen);
    printf("\r");
    fflush(stdout);
    printf("\r");
    fflush(stdout);
  }
  printf("\nFinished checking %s vocab. Press enter to continue: \n", params->lang_name);
}

/** End Debugging code **/

/** Evaluation code **/
void execute(char* command){
  //fprintf(stderr, "# Executing: %s\n", command);
  system(command);
}

void eval_mono(char* emb_file, char* lang, int iter) {
  char command[MAX_STRING];

  /** WordSim **/
  chdir("wordsim/code");
  fprintf(stderr, "# eval %d %s %s", iter, lang, "wordSim");
  sprintf(command, "./run_wordSim.sh %s 1 %s", emb_file, lang);
  execute(command);
  chdir("../..");

  /** Analogy **/
  if((iter+1)%5==0 && strcmp(lang, "en")==0){
    chdir("analogy/code");
    fprintf(stderr, "# eval %d %s %s", iter, "en", "analogy");
    sprintf(command, "./run_analogy.sh %s 1", emb_file);
    execute(command);
    chdir("../..");
  }
}

// cross-lingual document classification
void cldc(char* outPrefix, int iter) {
  char command[MAX_STRING];

  /* de2en */
  // prepare data
  chdir("cldc/scripts/de2en");
  sprintf(command, "./prepare-data-klement-4cat-1000-my-embeddings.ch %s", outPrefix); execute(command);

  // run perceptron
  fprintf(stderr, "# eval %d %s %s", iter, "de2en", "cldc");
  sprintf(command, "./run-perceptron-1000-my-embeddings.ch %s > %s.eval%d", outPrefix, output_prefix, iter);
  execute(command);

  /** en2de **/
  // prepare data
  chdir("../en2de");
  sprintf(command, "./prepare-data-klement-4cat-1000-my-embeddings.ch %s", outPrefix); execute(command);

  // run perceptron
  fprintf(stderr, "# eval %d %s %s", iter, "en2de", "cldc");
  sprintf(command, "./run-perceptron-1000-my-embeddings.ch %s > %s.eval%d", outPrefix, output_prefix, iter);
  execute(command);
  chdir("../../..");
}
/** End Evaluation code **/


void InitUnigramTable(struct lang_params *params) {
  printf("# Init unigram table\n");
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  long long vocab_size = params->vocab_size;
  struct vocab_word *vocab = params->vocab;
  params->table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    params->table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// Return word length
int ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return 4;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;

  return a;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}


// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word, const struct vocab_word *vocab, const int *vocab_hash) {
  // puts("     Searching vocab...");
  unsigned int hash = GetWordHash(word);
  int original_hash = hash;
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (global_debug_flag) {
      printf("Hash is now %d (%s), vs the original hash of %d", hash, vocab[vocab_hash[hash]].word, original_hash);
    }
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) {
      return vocab_hash[hash];
    }
    hash = (hash + 1) % vocab_hash_size;
  }
  // puts("     Done searching vocab.");
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, const struct vocab_word *vocab, const int *vocab_hash) {
  char word[MAX_STRING];
  int word_len = ReadWord(word, fin);
  if(word_len >= MAX_STRING - 2) printf("! long word: %s\n", word);

  if (feof(fin)) return -1;
  return SearchVocab(word, vocab, vocab_hash);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, struct lang_params *params) {
  // puts("Adding word to vocab");
  unsigned int hash, length = strlen(word) + 1;
  long long vocab_size = params->vocab_size;
  long long vocab_max_size = params->vocab_max_size;
  struct vocab_word *vocab = params->vocab;
  int *vocab_hash = params->vocab_hash;

  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) {
    hash = (hash + 1) % vocab_hash_size;
  }
  vocab_hash[hash] = vocab_size - 1;
  params->vocab_size = vocab_size;
  params->vocab_max_size = vocab_max_size;
  params->vocab = vocab;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct lang_params *params) {
  int a, size;
  unsigned int hash;
  int *vocab_hash = params->vocab_hash;
  struct vocab_word *vocab = params->vocab;
  long long vocab_size = params->vocab_size;

  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  params->total_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)){ // a=0 is </s> and we want to keep it.
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      params->total_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }

  params->vocab = vocab;
  params->vocab_size = vocab_size;
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct lang_params *params) {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < params->vocab_size; a++) if (params->vocab[a].cn > min_reduce) {
    params->vocab[b].cn = params->vocab[a].cn;
    params->vocab[b].word = params->vocab[a].word;
    b++;
  } else free(params->vocab[a].word);
  params->vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  for (a = 0; a < params->vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(params->vocab[a].word);
    while (params->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    params->vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(struct lang_params *params) {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(params->vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(params->vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(params->vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < params->vocab_size; a++) count[a] = params->vocab[a].cn;
  for (a = params->vocab_size; a < params->vocab_size * 2; a++) count[a] = 1e15;
  pos1 = params->vocab_size - 1;
  pos2 = params->vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < params->vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[params->vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = params->vocab_size + a;
    parent_node[min2i] = params->vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < params->vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == params->vocab_size * 2 - 2) break;
    }
    params->vocab[a].codelen = i;
    params->vocab[a].point[0] = params->vocab_size - 2;
    for (b = 0; b < i; b++) {
      params->vocab[a].code[i - b - 1] = code[b];
      params->vocab[a].point[i - b] = point[b] - params->vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void CountWordsFromTrainFile(struct file_params *params) {
  char word[MAX_STRING];
  FILE *fin;

  if (debug_mode > 0) printf("# Count words from %s\n", params->train_file);

  fin = fopen(params->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  while (1) {
    //puts("Reading word");
    //printf("Debug mode: %d\n", debug_mode);
    //printf("File size: %lld\n", params->file_size);
    ReadWord(word, fin);
    if (feof(fin)) break;
    params->train_words++;
    if ((debug_mode > 1) && (params->train_words % 100000 == 0)) {
      printf("%lldK%c", params->train_words / 1000, 13);
      fflush(stdout);
    }
  }
  if (debug_mode > 0) {
    printf("  Words in train file: %lld\n", params->train_words);
  }
  params->file_size = ftell(fin);
  fclose(fin);
}


void LearnVocabFromTrainFiles(struct lang_params *params) {
  //this does not remove any previous vocab-related info in params, but updates it if the new file
  // has new information
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;

  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  puts("Vocabulary set to -1");
  for (ll1=0; ll1 < (params->num_files); ll1++) {
    struct file_params *file = params->files[ll1];
    if (debug_mode > 0) {
      printf("# Learn vocab for %s from %s (file %d of %d)\n", params->lang_name, file->train_file, ll1+1, params->num_files);
    }
    fin = fopen(file->train_file, "rb");
    if (fin == NULL) {
      printf("ERROR: training data file not found!\n");
      exit(1);
    }

    if (ll1 == 0) {
      params->vocab_size = 0;
      AddWordToVocab((char *)"</s>", params);
    }
    while (1) {
      ReadWord(word, fin);
      if (feof(fin)) break;
      file->train_words++;
      if ((debug_mode > 1) && (file->train_words % 100000 == 0)) {
        printf("%lldK%c", file->train_words / 1000, 13);
        fflush(stdout);
      }
      i = SearchVocab(word, params->vocab, params->vocab_hash);

      if (i == -1) {
        a = AddWordToVocab(word, params);
        params->vocab[a].cn = 1;
      } else params->vocab[i].cn++;
      if (params->vocab_size > vocab_hash_size * 0.7) ReduceVocab(params);
    }
    //used to have SortVocab here - revert if broken
    if (debug_mode > 0) {
      printf("\n");
      printf("  Vocab size: %lld\n", params->vocab_size);
      printf("  Words in train file: %lld\n", file->train_words);
    }
    file->file_size = ftell(fin);
    fclose(fin);
  }
  SortVocab(params);
  printf("Finished learning vocab for language %s\n", params->lang_name);
  params->full_vocab = 1;
}

void SaveVocab(struct lang_params *params) {
  long long i;
  FILE *fo = fopen(params->vocab_file, "wb");
  for (i = 0; i < params->vocab_size; i++) fprintf(fo, "%s %lld\n", params->vocab[i].word, params->vocab[i].cn);
  fclose(fo);
}

void ReadVocab(struct lang_params *params) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(params->vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) params->vocab_hash[a] = -1;
  params->vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word, params);
    fscanf(fin, "%lld%c", &params->vocab[a].cn, &c);
    i++;
  }
  SortVocab(params);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", params->vocab_size);
    // printf("Words in train file: %lld\n", params->train_words);
  }

  //this doesn't seem to have anything to do wth the vocab, should be moved
  // fin = fopen(params->train_files[0], "rb");
  // if (fin == NULL) {
  //   printf("ERROR: training data file not found!\n");
  //   exit(1);
  // }
  // fseek(fin, 0, SEEK_END);
  // params->file_size = ftell(fin);
  // fclose(fin);
  printf("Finished learning vocab for language %s\n", params->lang_name);
  params->full_vocab = 1;
}

// To find split points in a file, so that later each thread can handle one chunk of the data
void ComputeBlockStartPoints(char* file_name, int num_blocks, long long **blocks, long long *num_lines) {
  printf("# ComputeBlockStartPoints %s, num_blocks=%d\n", file_name, num_blocks);
  long long block_size;
  int line_count = 0;
  int curr_block = 0;
  char line[MAX_SENT_LEN];
  FILE *file;

  *num_lines = 0;
  file = fopen(file_name, "r");
  while (1) {
    fgets(line, MAX_SENT_LEN, file);
    if (feof(file)) break;
    ++(*num_lines);
  }
  printf("  num_lines=%lld, eof position %lld\n", *num_lines, (long long) ftell(file));

  fseek(file, 0, SEEK_SET);
  block_size = (*num_lines - 1) / num_blocks + 1;
  printf("  block_size=%lld lines\n  blocks = [0", block_size);

  *blocks = malloc((num_blocks+1) * sizeof(long long));
  (*blocks)[0] = 0;
  curr_block = 0;
  long long int cur_size = 0;
  while (1) {
    fgets(line, MAX_SENT_LEN, file);
    line_count++;
    cur_size++;

    // done with a block or reach eof
    if (cur_size == block_size || line_count==(*num_lines)) {
      curr_block++;
      (*blocks)[curr_block] = (long long)ftell(file);
      printf(" %lld", (*blocks)[curr_block]);
      if (line_count==(*num_lines)) { // eof
        break;
      }

      // reset
      cur_size = 0;
    }
  }
  printf("]\n");
  assert(curr_block==num_blocks);
  assert(line_count==(*num_lines));

  fclose(file);
}


// neu1: avg context embedding
// syn0: input embeddings (both hs and negative)
// syn1: output node embeddings (hs)
// syn1neg: output embeddings (negative)
// neu1: hidden vector
// neu1e: hidden vector error
void ProcessSkipPair(long long in_word, long long out_word, unsigned long long *next_random,
    struct lang_params *in_params, struct lang_params *out_params, real *neu1e, real skip_alpha) {
  long long d;
  long long l1, l2, c, target, label;
  real f, g;
  
#ifdef DEBUG
  //printf("  skip %s -> %s\n", in_params->vocab[in_word].word, out_params->vocab[out_word].word); fflush(stdout);
#endif

  l1 = in_word * layer1_size;
  for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

  // HIERARCHICAL SOFTMAX
  if (hs) for (d = 0; d < out_params->vocab[out_word].codelen; d++) {
    f = 0;
    l2 = out_params->vocab[out_word].point[d] * layer1_size;
    // Propagate hidden -> output
    for (c = 0; c < layer1_size; c++) f += in_params->syn0[c + l1] * out_params->syn1[c + l2];
    if (f <= -MAX_EXP) continue;
    else if (f >= MAX_EXP) continue;
    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    // 'g' is the gradient multiplied by the learning rate
    g = (1 - out_params->vocab[out_word].code[d] - f) * skip_alpha;
    // Propagate errors output -> hidden
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * out_params->syn1[c + l2];
    // Learn weights hidden -> output
    for (c = 0; c < layer1_size; c++) out_params->syn1[c + l2] += g * in_params->syn0[c + l1];
  }
  // NEGATIVE SAMPLING
  if (negative > 0) for (d = 0; d < negative + 1; d++) {
    if (d == 0) {
      target = out_word;
      label = 1;
    } else {
      *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
      target = out_params->table[((*next_random) >> 16) % table_size];
      if (target == 0) target = (*next_random) % (out_params->vocab_size - 1) + 1;
      if (target == out_word) continue;
      label = 0;
    }
    l2 = target * layer1_size;
    f = 0;
    for (c = 0; c < layer1_size; c++) f += in_params->syn0[c + l1] * out_params->syn1neg[c + l2];
    if (f > MAX_EXP) g = (label - 1) * skip_alpha;
    else if (f < -MAX_EXP) g = (label - 0) * skip_alpha;
    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * skip_alpha;
    for (c = 0; c < layer1_size; c++) neu1e[c] += g * out_params->syn1neg[c + l2];
    for (c = 0; c < layer1_size; c++) out_params->syn1neg[c + l2] += g * in_params->syn0[c + l1];
  }
  // Learn weights input -> hidden
  //TODO remove, index should be 'c'
  for (c = 0; c < layer1_size; c++) in_params->syn0[c + l1] += neu1e[c];
}

/** Monolingual predictions **/
// side = 0 ---> src
// side = 1 ---> tgt
// neu1: cbow, hidden vectors
// neu1e: skipgram
// syn0: input embeddings (both hs and negative)
// syn1: output embeddings (hs)
// syn1neg: output embeddings (negative)
void ProcessSentence(int sentence_length, long long *sen, struct lang_params *src, unsigned long long *next_random, real *neu1, real *neu1e) {
  int a, b, c, sentence_position;
  long long out_word, in_word;

  for (sentence_position = 0; sentence_position < sentence_length; ++sentence_position) {
    out_word = sen[sentence_position];
    if (out_word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
    b = (*next_random) % window;
    if (cbow) {  //train the cbow architecture
      (void)0;
      // ProcessCbow(sentence_position, sentence_length, sen, out_word, b, next_random, src, src, neu1, neu1e);
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a; // sentence - (window - b) -> sentence + (window - b)
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        in_word = sen[c];
        if (in_word == -1) continue;

        ProcessSkipPair(in_word, out_word, next_random, src, src, neu1e, alpha);
      } // for a (skipgram)
    } // end if cbow
  } // sentence
}

/** Crosslingual predictions **/
void ProcessSentenceAlign(struct lang_params *src, long long src_word, int src_pos, //int *tgt_id_map,
                          struct lang_params *tgt, long long* tgt_sent, int tgt_len, int tgt_pos,
                          unsigned long long *next_random, real *neu1, real *neu1e) {
  int neighbor_pos, a;
  //int neighbor_pos, neighbor_count;
  real b;

  // get the range
  (*next_random) = (*next_random) * (unsigned long long)25214903917 + 11;
  b = (*next_random) % window;

#ifdef DEBUG
  long long tgt_word = tgt_sent[tgt_pos];
  printf(" align %s (%d) - %s (%d)\n", src->vocab[src_word].word, src_pos, tgt->vocab[tgt_word].word, tgt_pos);
  fflush(stdout);
#endif

  if (cbow) {  // cbow
    // tgt -> src
    (void)0;
    // ProcessCbow(tgt_pos, tgt_len, tgt_sent, src_word, b, next_random, tgt, src, neu1, neu1e);
  } else {  // skip-gram
    for (a = b; a < window * 2 + 1 - b; ++a) if (a != window) {
      // src -> tgt neighbor
      neighbor_pos = tgt_pos -window + a;
      if (neighbor_pos >= 0 && neighbor_pos < tgt_len) {
        ProcessSkipPair(src_word, tgt_sent[neighbor_pos], next_random, src, tgt, neu1e, bi_alpha);
      }
    }
  } // end for if (cbow)
}


void *TrainModelThread(void *id) {
#ifdef DEBUG
  long long src_sen_orig[MAX_WORD_PER_SENT + 1], tgt_sen_orig[MAX_WORD_PER_SENT + 1];
#endif
  puts("Start TrainModelThread");
  long long word;
  int src_sentence_length = 0;
  int tgt_sentence_length = 0;
  long long src_sen[MAX_WORD_PER_SENT + 1];
  long long tgt_sen[MAX_WORD_PER_SENT + 1];
  unsigned long long next_random = (long long)id;
  clock_t now;

  //possibly replaceable by per-filepair arrays
  long long src_word_count = 0, src_last_word_count = 0, tgt_word_count = 0;
  FILE *src_fi = NULL, *tgt_fi = NULL, *align_fi=NULL;
  long long int sent_id = 0;
  //
  
  struct file_params *src_train;
  struct file_params *tgt_train;

  struct lang_params *src_lang;
  struct lang_params *tgt_lang;

  //index current pair in global list of file pairs
  int finished_pairs = 0;
  int current_pair = 0;

  //check at the beginning of every file switch, only loop over files where you haven't finished this thread's block
  int finished[num_pairs];

  //sent_ids - optional, seem to be only a debugging/printing thing, could be universal rather than file-pair specific

  //to keep track of progress through each file (pair)
  long long src_word_counts[num_pairs];
  long long tgt_word_counts[num_pairs];
  long long src_last_word_counts[num_pairs];

  //pointers to open files
  FILE *src_fps[num_pairs], *tgt_fps[num_pairs], *align_fps[num_pairs];

  //have to replace all references to point at these arrays
  //create global structure to track file pair set
  //loop over file pairs
  //alter EOF behavior from breaking out of thread to continuing on to next file pair and setting this one to "finished"
  //

  // for alignment
  int src_sentence_orig_length=0, tgt_sentence_orig_length=0;
  int src_id_map[MAX_WORD_PER_SENT + 1], tgt_id_map[MAX_WORD_PER_SENT + 1]; // map from original indices to new indices if id_map[j]==0, word j is deleted
  int src_align_map[MAX_WORD_PER_SENT + 1]; // map from src positions to tgt positions and vice versa
  int count; //for unsupervised alignment gaps
  int src_pos, tgt_pos;
  char ch;

  //temporary storage for a single word vector (layer1_size real numbers)
  real *neu1 = (real *)calloc(layer1_size, sizeof(real)); // cbow
  real *neu1e = (real *)calloc(layer1_size, sizeof(real)); // skipgram

  long long all_tgt_words = 0; //debugging-related only
  long long all_src_words = 0;
  long long prev_all_src_words = 0;
  long long total_all_tgt_words = 0;
  long long total_all_src_words = 0;

  //I'm fairly confident about the pointer juggling here but if something goes wrong look here first
  for (current_pair=0; current_pair<num_pairs; current_pair++) {
    pair = all_pairs[current_pair];
    src_train = pair->src;
    tgt_train = pair->tgt;

    total_all_src_words = total_all_src_words + src_train->train_words;
    total_all_tgt_words = total_all_tgt_words + tgt_train->train_words;
    
    src_fi = fopen(src_train->train_file, "rb");
    fseek(src_fi, src_train->line_blocks[(long long)id], SEEK_SET);
    src_fps[current_pair] = src_fi;
    // tgt
    tgt_fi = fopen(tgt_train->train_file, "rb");
    fseek(tgt_fi, tgt_train->line_blocks[(long long)id], SEEK_SET);
    tgt_fps[current_pair] = tgt_fi;
    // align
    if(align_opt){
      align_fi = fopen(pair->align_file, "rb");
      fseek(align_fi, pair->align_line_blocks[(long long)id], SEEK_SET);
      tgt_fps[current_pair] = tgt_fi;
    }
    src_word_counts[current_pair] = 0;
    src_last_word_counts[current_pair] = 0;
    tgt_word_counts[current_pair] = 0;

    finished[current_pair] = 0;
  }

  printf("Total src words: %lld \n", total_all_src_words);
  printf("Total tgt words: %lld \n", total_all_tgt_words);
  
  current_pair = 0;
  while (finished_pairs < num_pairs) {
    //within while loop, iterate over different src/tgt pairs? but would have to keep track of separate src/tgt_word_counts, sent_ids, last_word_counts, etc.
    //and also different behavior when finishing a file block - don't drop entire thread, but continue with other file pairs?
    current_pair = current_pair % num_pairs;
    int loops = 0;
    while (finished[current_pair] == 1) {
      current_pair++;
      if (current_pair >= num_pairs) {
	current_pair = 0;
	loops++;
      }
      if (loops > 3*num_pairs) {
	printf("Stuck in infinite loop");
	break;
      }
    }

#ifdef DEBUG
    printf("Continuing with current_pair %d (num_pairs %d) with languages %s, %s\n", current_pair, num_pairs, all_pairs[current_pair]->src->lang->lang_name, all_pairs[current_pair]->tgt->lang->lang_name);
#endif
    //switch to values for this pair
    pair = all_pairs[current_pair];
    src_word_count = src_word_counts[current_pair];
    src_last_word_count = src_last_word_counts[current_pair];
    tgt_word_count = tgt_word_counts[current_pair];
    src_fi = src_fps[current_pair];
    tgt_fi = tgt_fps[current_pair];
    align_fi=align_fps[current_pair];
  
    src_train = pair->src;
    tgt_train = pair->tgt;
    src_lang = src_train->lang;
    tgt_lang = tgt_train->lang;

#ifdef DEBUG
    printf("# Load sentence %lld, src_word_count %lld, src_last_word_count %lld\n", sent_id, src_word_count, src_last_word_count); fflush(stdout);
    printf("  src, sample=%g, dropping words:", sample); fflush(stdout);
#endif

    if (src_word_count > src_train->train_words) {
      printf("BREAKPOINT: Out of source (swc/st->tw) words\n");
    }
    if (all_src_words > total_all_src_words) {
      printf("BREAKPOINT: Out of source (all/total) words\n");
    }

    if (all_src_words - prev_all_src_words > 2000) {
      src_train->word_count_actual += src_word_count - src_last_word_count;
      prev_all_src_words = all_src_words;
      src_last_word_count = src_word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f, bi_alpha: %f,  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, bi_alpha,
	       //(src_train->word_count_actual)/ (real)(num_threads * src_train->train_words + 1) * 100,
	       // src_train->word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
	       (all_src_words)/ (real)(num_threads * total_all_src_words + 1) * 100,
	        all_src_words / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }

      alpha = starting_alpha * (1 - (cur_iter * src_train->train_words + src_train->word_count_actual) / (real)(num_train_iters * src_train->train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      bi_alpha = alpha*bi_weight;
    }

    // load src sentence
    src_sentence_length = 0;
    src_sentence_orig_length = 0;
    while (1) {
      word = ReadWordIndex(src_fi, src_lang->vocab, src_lang->vocab_hash);
      all_src_words++;
      if (feof(src_fi) || word == 0) break; // end of file or sentence
      if(src_sentence_orig_length>=MAX_WORD_PER_SENT) continue; // read enough

      // keep the orig src
#ifdef DEBUG
      if (word==-1) src_sen_orig[src_sentence_orig_length] = src_lang->unk_id;
      else src_sen_orig[src_sentence_orig_length] = word;
#endif
      src_sentence_orig_length++;

      // unknown token. IMPORTANT: this line needs to be after the one where we store src_sen_orig (for bilingual models to work)
      if (word == -1) {
        src_id_map[src_sentence_orig_length-1] = -1;
        continue;
      }
      src_word_count++;

      // The subsampling randomly discards frequent words while keeping the ranking same
      if (sample > 0) {
        // larger sample means larger ran, which means discard less frequent
        // [ sqrt(freq) / sqrt(sample * N) + 1 ] * (sample * N / freq) = sqrt(sample * N / freq) + (sample * N / freq)
        real ran = (sqrt(src_lang->vocab[word].cn / (sample * src_train->train_words)) + 1) * (sample * src_train->train_words) / src_lang->vocab[word].cn;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) { // discard

#ifdef DEBUG
          //printf("dropped: %s\n", src_lang->vocab[word].word);
#endif

          src_id_map[src_sentence_orig_length-1] = -1;
          continue;
        } else {
#ifdef DEBUG
          //printf("kept: %s\n", src_lang->vocab[word].word);
#endif
	  src_id_map[src_sentence_orig_length-1] = src_sentence_length;
        }
      }

      src_sen[src_sentence_length] = word;
      src_sentence_length++;
    }

#ifdef DEBUG
    sprintf(prefix, "\n  src orig %lld, len %d:", sent_id, src_sentence_orig_length);
    print_sent(src_sen_orig, src_sentence_orig_length, src_lang->vocab, prefix);
    sprintf(prefix, "  src %lld, len %d:", sent_id, src_sentence_length);
    print_sent(src_sen, src_sentence_length, src_lang->vocab, prefix);
    //printf("Press enter to continue:");
    //getchar();
#endif

    ProcessSentence(src_sentence_length, src_sen, src_lang, &next_random, neu1, neu1e);
    
    // load tgt sentence
    tgt_sentence_length = 0;
    tgt_sentence_orig_length = 0;
#ifdef DEBUG
    printf("  tgt, sample=%g, dropping words:", sample); fflush(stdout);
#endif
    while (1) {

      word = ReadWordIndex(tgt_fi, tgt_lang->vocab, tgt_lang->vocab_hash);
      all_tgt_words++;
      if (feof(tgt_fi) || word == 0) break; // end of file or sentence
      if(tgt_sentence_orig_length>=MAX_WORD_PER_SENT) continue; // read enough

      // keep the orig tgt
#ifdef DEBUG
      if (word==-1) tgt_sen_orig[tgt_sentence_orig_length] = tgt_lang->unk_id;
      else tgt_sen_orig[tgt_sentence_orig_length] = word;
#endif
      tgt_sentence_orig_length++;

      // unknown token. IMPORTANT: this line needs to be after the one where we store sen_orig for bilingual models to work
      if (word == -1) {
        tgt_id_map[tgt_sentence_orig_length-1] = -1;
        continue;
      }
      tgt_word_count++;

      // The subsampling randomly discards frequent words while keeping the ranking same
      if (sample > 0) {
        real ran = (sqrt(tgt_lang->vocab[word].cn / (sample * tgt_train->train_words)) + 1) * (sample * tgt_train->train_words) / tgt_lang->vocab[word].cn;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) {

#ifdef DEBUG
          //printf("dropped: %s\n", tgt_lang->vocab[word].word); fflush(stdout);
#endif

          tgt_id_map[tgt_sentence_orig_length-1] = -1;
          continue;
        } else {
#ifdef DEBUG
          //printf("kept: %s\n", tgt_lang->vocab[word].word); fflush(stdout);
#endif
	  tgt_id_map[tgt_sentence_orig_length-1] = tgt_sentence_length;
        }
      }
      tgt_sen[tgt_sentence_length] = word;
      tgt_sentence_length++;
    }

#ifdef DEBUG 
    sprintf(prefix, "\n  tgt orig %lld, len %d:", sent_id, tgt_sentence_orig_length);
    print_sent(tgt_sen_orig, tgt_sentence_orig_length, tgt_lang->vocab, prefix);
    sprintf(prefix, "  tgt %lld, len %d:", sent_id, tgt_sentence_length);
    print_sent(tgt_sen, tgt_sentence_length, tgt_lang->vocab, prefix);
    //printf("Press enter to continue:");
    //getchar();
#endif
    
    ProcessSentence(tgt_sentence_length, tgt_sen, tgt_lang, &next_random, neu1, neu1e);
    
    // align
    if (tgt_sentence_length) { //tgt sentence is not empty
      if (align_opt) { // use unsupervised alignments (UnsupAlign)
#ifdef DEBUG
	printf("Using unsupervised alignments.\n");
#endif
	for (src_pos = 0; src_pos < src_sentence_orig_length; ++src_pos) src_align_map[src_pos] = -1;

	while (fscanf(align_fi, "%d %d%c", &src_pos, &tgt_pos, &ch)) {
	  src_align_map[src_pos] = tgt_pos;
	  if (ch == '\n') break;
	}
	
	for (src_pos = 0; src_pos < src_sentence_orig_length; ++src_pos) {
	  if(src_id_map[src_pos]==-1) continue;
	  
	  // get tgt_pos
	  if(src_align_map[src_pos]==-1){ // no alignment, try to infer
	    count = 0;
	    tgt_pos = 0;
	    if(src_pos>0 && src_align_map[src_pos-1]!=-1){ // previous link
	      tgt_pos += src_align_map[src_pos-1];
	      count++;
	    }
	    if(src_pos<(src_sentence_orig_length-1) && src_align_map[src_pos+1]!=-1){ // next link
	      tgt_pos += src_align_map[src_pos+1];
	      count++;
	    }
	    if (count>0) tgt_pos = tgt_pos / count;
	  } else {
	    tgt_pos = src_align_map[src_pos];
	    count = 1;
	  }
	  
	  if (count>0 && tgt_id_map[tgt_pos]>=0){
	    // src, src_word, src_pos, tgt, tgt_sent, tgt_len, tgt_pos
	    ProcessSentenceAlign(src_lang, src_sen[src_id_map[src_pos]], src_id_map[src_pos],
				 tgt_lang, tgt_sen, tgt_sentence_length, tgt_id_map[tgt_pos],
				 &next_random, neu1, neu1e);
	    ProcessSentenceAlign(tgt_lang, tgt_sen[tgt_id_map[tgt_pos]], tgt_id_map[tgt_pos],
				 src_lang, src_sen, src_sentence_length, src_id_map[src_pos],
				 &next_random, neu1, neu1e);
	  }
	}
      } else { // uniform alignments (MonoAlign)
#ifdef DEBUG
	printf("Using uniform alignments.\n");
	printf("src_sentence_length %d, src_sentence_orig_length %d, tgt_sentence_length %d, tgt_sentence_orig_length %d\n", src_sentence_length, src_sentence_orig_length, tgt_sentence_length, tgt_sentence_orig_length);
	int sl = 0;
	for(sl=0; sl<src_sentence_orig_length; sl++) {
	  printf(" %d", src_id_map[sl]);
	}
	printf("\n");
	
	for(sl=0; sl<tgt_sentence_orig_length; sl++) {
	  printf(" %d", tgt_id_map[sl]);
	}
	printf("\n");
	//printf("Press enter to continue:");
	//getchar();
#endif
	for (src_pos = 0; src_pos < src_sentence_orig_length; ++src_pos) {
	  tgt_pos = src_pos * tgt_sentence_orig_length / src_sentence_orig_length;
	  if(src_id_map[src_pos]>=0 && tgt_id_map[tgt_pos]>=0){
	    ProcessSentenceAlign(src_lang, src_sen[src_id_map[src_pos]], src_id_map[src_pos],
				 tgt_lang, tgt_sen, tgt_sentence_length, tgt_id_map[tgt_pos],
				 &next_random, neu1, neu1e);
	    ProcessSentenceAlign(tgt_lang, tgt_sen[tgt_id_map[tgt_pos]], tgt_id_map[tgt_pos],
				 src_lang, src_sen, src_sentence_length, src_id_map[src_pos],
				 &next_random, neu1, neu1e);
	  }
	}
      }
    }

#ifdef DEBUG
    if ((sent_id % 1000) == 0) printf("Done processing sentence pair %lld\n", sent_id);
#endif

    sent_id++;

    if (feof(tgt_fi)) {
      printf("End of target file for file pair %d (%s-%s)\n", current_pair, src_lang->lang_name, tgt_lang->lang_name);
      finished[current_pair] = 1;
    }

    if (tgt_word_count > tgt_train->train_words / num_threads) {
      printf("Exceeded target words per thread for file pair %d (%s-%s): %lld / %lld with %d threads\n", current_pair, src_lang->lang_name, tgt_lang->lang_name, tgt_word_count, tgt_train->train_words, num_threads);
      finished[current_pair] = 1;
    }

    if (feof(src_fi)) {
      printf("End of source file for file pair %d (%s-%s)\n", current_pair, src_lang->lang_name, tgt_lang->lang_name);
      finished[current_pair] = 1;
    }
    
    if (src_word_count > src_train->train_words / num_threads) {
      printf("Exceeded source words per thread for file pair %d (%s-%s): %lld / %lld with %d threads\n", current_pair, src_lang->lang_name, tgt_lang->lang_name, src_word_count, src_train->train_words, num_threads);
      finished[current_pair] = 1;
    }

    if (finished[current_pair]) {
      fclose(src_fi); fclose(tgt_fi); if (align_opt) fclose(align_fi);
      finished_pairs++;
      continue;
    }

    //save values and switch to new file pair
    src_word_counts[current_pair] = src_word_count;
    src_last_word_counts[current_pair] = src_last_word_count;
    tgt_word_counts[current_pair] = tgt_word_count;
    //printf("Incrementing current_pair from %d to %d\n", current_pair, current_pair+1);
    current_pair++;
  } //end while(1)

  free(neu1);
  free(neu1e);
  
  printf("Target words read: %lld/%lld \n", all_tgt_words, total_all_tgt_words);
  printf("Source words read: %lld/%lld \n", all_src_words, total_all_src_words);
  printf("End of thread\n");

  pthread_exit(NULL);
}

// opt 1: save avg vecs, 2: save out vecs
void SaveVector(char* output_prefix, char* lang, struct lang_params *params, int opt){
  long a, b;
  long long vocab_size = params->vocab_size;
  struct vocab_word *vocab = params->vocab;
  real sum;
  int save_out_vecs = 0, save_avg_vecs = 0;
  if (opt==1) save_avg_vecs = 1;
  if (opt==2) save_out_vecs = 1;

  char output_file[MAX_STRING];
  sprintf(output_file, "%s.%s", output_prefix, lang);

  // Save the word vectors
  real *syn0 = params->syn0;
  FILE* fo = fopen(output_file, "wb");
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);

  // Save sum out vecs or sum of in and out vecs
  FILE* fo_sum = NULL;
  FILE* fo_out = NULL;
  real *syn1neg = params->syn1neg; // negative sampling
  if(hs==0) { // only for negative sampling, we have the notion of output vectors
    if (save_avg_vecs){
      char sum_vector_file[MAX_STRING];
      sprintf(sum_vector_file, "%s.sumvec.%s", output_prefix, lang);
      fo_sum = fopen(sum_vector_file, "wb");
      fprintf(fo_sum, "%lld %lld\n", vocab_size, layer1_size);
    }

    if (save_out_vecs){
      char out_vector_file[MAX_STRING];
      sprintf(out_vector_file, "%s.outvec.%s", output_prefix, lang);
      fo_out = fopen(out_vector_file, "wb");
      fprintf(fo_out, "%lld %lld\n", vocab_size, layer1_size);
    }
  }

  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if(hs==0) {
      if (save_avg_vecs) fprintf(fo_sum, "%s ", vocab[a].word);
      if (save_out_vecs) fprintf(fo_out, "%s ", vocab[a].word);
    }

    if (binary) { // binary
      for (b = 0; b < layer1_size; b++) {
        fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);

        if(hs==0) {
          if (save_avg_vecs) {
            sum = syn0[a * layer1_size + b] + syn1neg[a * layer1_size + b];
            fwrite(&sum, sizeof(real), 1, fo_sum);
          }
          if (save_out_vecs) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo_out);
        }

      }
    } else { // text
      for (b = 0; b < layer1_size; b++) {
        fprintf(fo, "%lf ", syn0[a * layer1_size + b]);

        if(hs==0) {
          if (save_avg_vecs) {
            sum = syn0[a * layer1_size + b] + syn1neg[a * layer1_size + b];
            fprintf(fo_sum, "%lf ", sum);
          }
          if (save_out_vecs) fprintf(fo_out, "%lf ", syn1neg[a * layer1_size + b]);
        }
      }
    }
    fprintf(fo, "\n");

    if(hs==0) {
      if (save_avg_vecs) fprintf(fo_sum, "\n");
      if (save_out_vecs) fprintf(fo_out, "\n");
    }
  }
  fclose(fo);
  
  if(hs==0) {
    if (save_avg_vecs) fclose(fo_sum);
    if (save_out_vecs) fclose(fo_out);
  }
}

// init vocab, unk_id, vector table for each language
void LanguageInit(struct lang_params *params){
  puts("Calling LanguageInit");
  /* initialize full vocabulary by reading vocab file or all training files */
  if (access(params->vocab_file, F_OK) != -1) { // vocab file exists
    printf("# Vocab file (%s) exists. Loading ...\n", params->vocab_file);
    ReadVocab(params);
  } else { // vocab file doesn't exist
    printf("# Vocab file (%s) doesn't exist. Deriving ...\n", params->vocab_file);
    LearnVocabFromTrainFiles(params);
    printf("Vocab learnt...\n");
    SaveVocab(params);
    printf("Vocab saved.\n");
  }

  /* set unk_id from vocab */
  params->unk_id = params->vocab_hash[GetWordHash("<unk>")];
  if (params->unk_id<0){
    fprintf(stderr, "! Can't find <unk> in the vocab file %s\n", params->vocab_file);
    exit(1);
  } else {
    fprintf(stderr, "  <unk> id in %s = %lld\n", params->vocab_file, params->unk_id);
  }

  /* set output filename based on output prefix and language name */
  sprintf(params->output_file, "%s.%s", output_prefix, params->lang_name);

  /* initializes space for the embeddings arrays based on vocab_size */
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&params->syn0, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
  if (params->syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    // this is because the number of nodes in a tree is approximately the number of words.
    a = posix_memalign((void **)&params->syn1, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
    if (params->syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < params->vocab_size; a++) for (b = 0; b < layer1_size; b++)
     params->syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&params->syn1neg, 128, (long long)params->vocab_size * layer1_size * sizeof(real));
    if (params->syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < params->vocab_size; a++) for (b = 0; b < layer1_size; b++)
     params->syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < params->vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    params->syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree(params);

  if (negative > 0) InitUnigramTable(params);

#ifdef DEBUG
    printf("  MonoInit Vocab size: %lld\n", params->vocab_size);
    //printf("  MonoInit Words in train file: %lld\n", params->train_words);
#endif
  puts("Exiting LanguageInit");
}

void MonoInit(struct file_params *params) {
  puts("Calling MonoInit");
  if (params->lang->full_vocab == 0) {
    LanguageInit(params->lang);
  }
  //get params->train_words in case vocab was already known
  CountWordsFromTrainFile(params);
  ComputeBlockStartPoints(params->train_file, num_threads, &params->line_blocks, &params->num_lines);
  puts("Exiting MonoInit");
}


void TrainAllLanguagePairs() {
  long a;
  int current_pair;
  struct file_params *src;
  struct file_params *tgt;

  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  starting_alpha = alpha;
  if (output_prefix[0] == 0) {
    printf("Output prefix is empty, exiting");
    return;
  }
  // init all language/file pairs
  for (current_pair=0; current_pair<num_pairs; current_pair++) {
    pair = all_pairs[current_pair];
    src = pair->src;
    tgt = pair->tgt;
    printf("Initializing src-file %s and tgt-file %s\n", src->train_file, tgt->train_file);

    MonoInit(src);
    MonoInit(tgt);

    puts("Finished MonoInit");
    assert(src->num_lines==tgt->num_lines);

    if (align_opt > 0) {
      ComputeBlockStartPoints(pair->align_file, num_threads, &pair->align_line_blocks, &pair->align_num_lines);
      assert(src->num_lines==pair->align_num_lines);
    }
  }  
  int save_opt = 1;
  //char sum_vector_file[MAX_STRING];
  //char sum_vector_prefix[MAX_STRING];
  for(cur_iter=start_iter; cur_iter<num_train_iters; cur_iter++){
    puts("Starting new training iter");
    start = clock();
    for (current_pair=0; current_pair<num_pairs; current_pair++) {
      pair = all_pairs[current_pair];
      pair->src->word_count_actual = pair->tgt->word_count_actual = 0;
    }
    // Train Model
    fprintf(stderr, "\n## Start iter %d, alpha=%f ... ", cur_iter, alpha); execute("date"); fflush(stderr);
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    fprintf(stderr, "\n# Done iter %d, alpha=%f, ", cur_iter, alpha); execute("date"); fflush(stderr);
    for (current_pair=0; current_pair<num_pairs; current_pair++) {
      pair = all_pairs[current_pair];
      src = pair->src;
      tgt = pair->tgt;
      print_model_stat(src->lang);
      print_model_stat(tgt->lang);

      // Save
      SaveVector(output_prefix, src->lang->lang_name, src->lang, save_opt);
      SaveVector(output_prefix, tgt->lang->lang_name, tgt->lang, save_opt);
    }  
    /* Eval
    if (eval_opt) {
      fprintf(stderr, "\n# eval %d, ", cur_iter); execute("date"); fflush(stderr);
      eval_mono(src->lang->output_file, src->lang->lang_name, cur_iter);

      SaveVector(output_prefix, tgt->lang->lang_name, tgt->lang, save_opt);
      eval_mono(tgt->lang->output_file, tgt->lang->lang_name, cur_iter);
      // cldc
      cldc(output_prefix, cur_iter);

      // sum vector for negative sampling
      if (save_opt==1 && hs==0){
        fprintf(stderr, "\n# Eval on sum vector file %s\n", sum_vector_file);
        sprintf(sum_vector_file, "%s.sumvec.%s", output_prefix, src->lang->lang_name);
        eval_mono(sum_vector_file, src->lang->lang_name, cur_iter);

        sprintf(sum_vector_file, "%s.sumvec.%s", output_prefix, tgt->lang->lang_name);
        eval_mono(sum_vector_file, tgt->lang->lang_name, cur_iter);

        // cldc
        sprintf(sum_vector_prefix, "%s.sumvec", output_prefix);
        cldc(sum_vector_prefix, cur_iter);
      }

      fflush(stderr);
      } */ //end if eval_opt
  } // for cur_iter
}


int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

struct lang_params *InitLangParams(char * lang) {
  //DOES NOT INIT: ouput_file, vocab_file, unk_id
  // syn0, syn1, syn1neg, table
  struct lang_params *params = malloc(sizeof(struct lang_params));
  strcpy(params->lang_name, lang);

  // printf("Calling InitLangParams\n");

  params->vocab_size = 0;
  params->vocab_max_size = 1000;
  params->vocab = (struct vocab_word *)calloc(params->vocab_max_size, sizeof(struct vocab_word));
  params->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  params->full_vocab = 0;

  params->num_files = 0;
  params->files = (struct file_params **)malloc(num_languages*sizeof(struct file_params *));
  // printf("Exiting InitLangParams\n");
  return params;
}

struct pair_params *InitPairParams(struct file_params *src_params, struct file_params *tgt_params, char *align) {
  // printf("Calling InitPairParams\n");

  struct pair_params *params = malloc(sizeof(struct pair_params));
  //DOES NOT INIT: src_line_blocks, tgt_line_blocks
  strcpy(params->align_file, align);

  params->src = src_params;
  params->tgt = tgt_params;
  params->align_num_lines = 0;

  // printf("Exiting InitPairParams\n");
  return params;
}

struct file_params *InitFileParams(char *filename) {
  // printf("Calling InitFileParams\n");

  struct file_params *params = malloc(sizeof(struct file_params));
  //DOES NOT INIT: lang, line_blocks
  strcpy(params->train_file, filename);

  params->file_size = 0;
  params->num_lines = 0;
  params->train_words = 0;
  params->word_count_actual = 0;

  // printf("Exiting InitFileParams\n");
  return params;
}

//strcmp with pointer addition to get last characters
int FileLangCmp(char * lang, char * filename) {
  return strcmp(filename + strlen(filename) - strlen(lang), lang);
}

void LinkFilesToLangParams() {
  struct pair_params *cur_pair;
  struct lang_params *cur_lang;
  for (lp1=0; lp1<num_pairs; lp1++) {
    cur_pair = all_pairs[lp1];
    char *src_lang_name = basename(strdup(all_pairs[lp1]->src->train_file));
    char *tgt_lang_name = basename(strdup(all_pairs[lp1]->tgt->train_file));
    for (ll1=0; ll1<num_languages; ll1++) {
      cur_lang = all_langs[ll1];
      if (!FileLangCmp(cur_lang->lang_name, src_lang_name)) {
        cur_pair->src->lang = cur_lang;
        cur_lang->files[cur_lang->num_files] = cur_pair->src;
        cur_lang->num_files++;
        printf("src for pair %d is at index %d of all_langs\n",lp1, ll1);
      }
      if (!FileLangCmp(cur_lang->lang_name, tgt_lang_name)) {
        cur_pair->tgt->lang = cur_lang;
        cur_lang->files[cur_lang->num_files] = cur_pair->tgt;
        cur_lang->num_files++;
        printf("tgt for pair %d is at index %d of all_langs\n",lp1, ll1);
      }
    }
  }
}

int main(int argc, char **argv) {
  // srand(21260063);
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train_dir <path>\n");
    printf("\t\tUse text data from subfolders of <path> to train the model\n");
    printf("\t-output <path>\n");
    printf("\t\tUse <path> to save the resulting word vectors / word clusters (split by language)\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    return 0;
  }

  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) {
    layer1_size = atoi(argv[i + 1]);
    printf("# layer1_size (emb dim)=%lld\n", layer1_size);
  }

  /* multilingual arguments - create these arg strings with python because c sucks*/
  if ((i = ArgPos((char *)"-num_languages", argc, argv)) > 0) {
    num_languages = atoi(argv[i+1]);
    printf("num_languages: %d\n", num_languages);
    all_langs = malloc(num_languages*sizeof(struct lang_params *));
    language_indices = malloc(num_languages*sizeof(char *));
    for (ll1=0; ll1<num_languages; ll1++) {
      language_indices[ll1] = malloc(MAX_STRING*sizeof(char));
    }
  }
  if ((i = ArgPos((char *)"-num_pairs", argc, argv)) > 0) {
    num_pairs = atoi(argv[i+1]);
    printf("num_pairs: %d\n", num_pairs);
    all_pairs = malloc(num_pairs*sizeof(struct pair_params *));
  }

  if ((i = ArgPos((char *)"-language_names", argc, argv)) > 0) {
    for (ll1 = 0; ll1 < num_languages; ll1++) {
      struct lang_params *lparams = InitLangParams(argv[i+1+ll1]);
      printf("lparams lang: %s\n", lparams->lang_name);
      all_langs[ll1] = lparams;
      strcpy(language_indices[ll1], argv[i+1+ll1]);
      printf("in language indices: %.*s\n", 10, language_indices[ll1]);
    }
  }
  if ((i = ArgPos((char *)"-pair_filenames", argc, argv)) > 0) {
    for (lp1 = 0; lp1 < num_pairs; lp1++) {
      struct file_params *src = InitFileParams(argv[i+1+lp1*3]);
      struct file_params *tgt = InitFileParams(argv[i+1+lp1*3+1]);
      char *align_file = argv[i+1+lp1*3+2];
      all_pairs[lp1] = InitPairParams(src, tgt, align_file);
    }
  }


  if ((i = ArgPos((char *)"-align-opt", argc, argv)) > 0) align_opt = atoi(argv[i + 1]);

  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.025;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_prefix, argv[i + 1]);
    printf("# output_prefix=%s\n", output_prefix);
  }
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  // evaluation
  if ((i = ArgPos((char *)"-eval", argc, argv)) > 0) eval_opt = atoi(argv[i + 1]);

  // number of iterations
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) num_train_iters = atoi(argv[i + 1]);

  // bi_weight
  if ((i = ArgPos((char *)"-bi-weight", argc, argv)) > 0) bi_weight = atof(argv[i + 1]);

  printf("# MAX_STRING=%d\n", MAX_STRING);

  // get absolute path for output_prefix
  char actual_path[MAX_STRING];
  printf("# output prefix=%s\n", output_prefix);
  realpath(output_prefix, actual_path); 
  strcpy(output_prefix, actual_path);
  printf("# absolute path=%s\n", output_prefix);

  // vocab files
  for (ll1=0; ll1 < num_languages; ll1++) {
    sprintf(all_langs[ll1]->vocab_file, "%s.vocab.min%d", all_langs[ll1]->lang_name, min_count);
    if (specified_train_words>0) printf("# specified_train_words=%lld\n", specified_train_words);
  }

  LinkFilesToLangParams();

  // assertions and debugging
  for (lp1=0; lp1<num_pairs; lp1++) {
    struct pair_params *cur_pair = all_pairs[lp1];
    if (strcmp(cur_pair->align_file, "")==1) { // align_file is specified
      assert(align_opt>0);
    }
    printf("align: %s\n", cur_pair->align_file);
    printf("src: %s\n", cur_pair->src->lang->lang_name);
    printf("tgt: %s\n", cur_pair->tgt->lang->lang_name);
  }

  // compute exp table
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }

  TrainAllLanguagePairs();
  return 0;
}
