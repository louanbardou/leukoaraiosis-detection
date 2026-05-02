# Rapport Technique — Détection de Leucoaraïose par Deep Learning (ABCD Study)
**Date :** Mai 2026  
**Auteur :** Louan Bardou  
**Cluster :** CHPC-UCSF (nœuds H100 NVL)

---

## 1. Contexte scientifique — Pourquoi ce projet ?

### 1.1 Qu'est-ce que la leucoaraïose ?

La **leucoaraïose** (White Matter Abnormalities, WMA) désigne des lésions diffuses de la substance blanche cérébrale visibles en IRM. Chez l'adulte, elle est associée au risque vasculaire, aux démences, et aux troubles cognitifs. Chez l'adolescent, sa prévalence est rare (~3–5%) mais sa présence pourrait constituer un marqueur précoce de trajectoires neurodéveloppementales pathologiques.

### 1.2 Pourquoi la cohorte ABCD ?

L'**Adolescent Brain Cognitive Development (ABCD) Study** est la plus grande étude longitudinale du développement cérébral aux États-Unis : ~11 900 enfants suivis depuis 9–10 ans sur 7 visites annuelles, avec IRM T1w et T2w systématiques. C'est la seule cohorte pédiatrique de cette taille avec un suivi longitudinal aussi dense.

**L'opportunité scientifique :** en détectant la leucoaraïose à chaque session pour chaque sujet, on peut modéliser sa **trajectoire longitudinale** et identifier à quel âge elle apparaît, comment elle évolue, et quels facteurs environnementaux ou génétiques la prédisent.

### 1.3 Objectif à long terme

1. **(Ce projet)** Construire un classifieur binaire T1w+T2w → leucoaraïose présente/absente pour chaque scan ABCD
2. **(Phase suivante)** Alimenter un modèle spatio-temporel avec des triplets longitudinaux pour prédire la trajectoire de la substance blanche

---

## 2. Défis techniques fondamentaux

### Défi 1 — Pas d'IRM FLAIR

Le standard clinique pour détecter les hypersignaux de la substance blanche est l'IRM **FLAIR**, qui supprime le signal du LCR pour faire ressortir les lésions. ABCD n'a pas de FLAIR — uniquement T1w et T2w.

**Conséquence :** Sur T2w, lésions et LCR apparaissent tous les deux en hypersignal (difficile à distinguer). Sur T1w, les lésions apparaissent sombres (confondues avec la matière grise). Le modèle doit apprendre le contraste FLAIR-équivalent implicitement en combinant les deux modalités.

### Défi 2 — Pas de labels voxel-à-voxel

ABCD ne fournit que des labels cliniques au niveau de l'image entière (score MRIF ou présence/absence). Il n'y a aucun masque de segmentation manuelle.

**Conséquence :** Approche de **supervision faible** obligatoire. On entraîne un classifieur image-niveau, puis on extrait des heatmaps spatiales (Grad-CAM) depuis les activations internes du modèle pour localiser les lésions a posteriori.

### Défi 3 — Déséquilibre de classes extrême

La leucoaraïose est rare chez l'adolescent. Dans la cohorte ABCD, ~11% des scans sont positifs. Avec une telle disproportion :
- La **cross-entropy** converge vers "tout négatif" (haute accuracy, zero recall)
- L'**AUROC** est biaisée car triviallement haute quand le modèle prédit tout négatif
- La métrique correcte est l'**AUPREC** (aire sous la courbe Précision-Rappel), insensible à la taille de la classe négative

---

## 3. Architecture du modèle

### 3.1 Pourquoi Swin UNETR ?

Les CNNs classiques utilisent des noyaux convolutifs locaux (3×3×3), ce qui limite leur capacité à modéliser des relations spatiales à longue distance. Or les lésions de la substance blanche sont diffuses — elles couvrent de larges territoires across les faisceaux de matière blanche.

Le **Swin Transformer** calcule un self-attention multi-têtes dans des fenêtres locales, puis décale ces fenêtres entre les couches, propageant l'information globalement avec une complexité **linéaire** (et non quadratique) en nombre de voxels.

**Swin UNETR** est la version 3D médicale du Swin Transformer, pré-entraînée sur des données IRM cérébrales, ce qui fournit un point de départ pertinent pour notre tâche.

### 3.2 Adaptation pour la classification faiblement supervisée

Le Swin UNETR original est un modèle de segmentation (encodeur + décodeur). On modifie l'architecture :

```
Input (B, 2, 96, 96, 96)   ← T1w + T2w en 2 canaux
       ↓
Encodeur Swin UNETR         ← 4 stages, self-attention hiérarchique
       ↓
hidden_states[-1]           ← (B, 768, 3, 3, 3) — carte de features la plus profonde
       ↓
Global Average Pool         ← (B, 768) — effondrement spatial en un vecteur
       ↓
MLP head                    ← LayerNorm → Dropout → Linear(768,256) → GELU → Linear(256,1)
       ↓
Logit brut (B, 1)           ← sigmoid appliqué par la loss
```

**Pourquoi le Global Average Pool ?** En forçant le réseau à résumer le volume entier en un seul vecteur avant la décision, on l'oblige à activer fortement les régions discriminatives (les lésions). Lors de l'inférence Grad-CAM (Phase 4), supprimer le GAP révèle ces régions comme une heatmap spatiale 3D.

**Dimensions cachées :**  
- `feature_size=48` → Stage 1: 48, Stage 2: 96, Stage 3: 192, Stage 4: 384, Bottleneck: 768  
- `hidden_dim = feature_size × 16 = 768` *(bug corrigé : était × 32, donnait 1536 → crash à l'initialisation)*

**Paramètres totaux :** 62.4M (recommandé pour entraînement sur A100/H100 avec `use_checkpoint=True`)

### 3.3 Gradient checkpointing

`use_checkpoint=True` : au lieu de stocker toutes les activations intermédiaires pendant le forward pass, elles sont recalculées pendant le backward pass. Réduit l'utilisation mémoire GPU de ~40%, nécessaire pour faire rentrer le modèle + batch de 4 volumes 3D dans un seul GPU.

---

## 4. Stratégie d'entraînement

### 4.1 Loss : APLoss (LibAUC)

L'AUPREC n'est pas directement différentiable car elle repose sur des opérations de tri. `APLoss` de LibAUC calcule un surrogate lisse via une relaxation hinge carrée de l'objectif de ranking par paires.

Pour chaque paire (positif, négatif) dans le batch, la loss pénalise les cas où le score positif n'est pas supérieur au score négatif d'une marge de 1.0. Le paramètre `gamma=0.9` contrôle une moyenne mobile qui stabilise l'estimée du gradient entre les batchs.

```python
loss_fn = APLoss(data_len=len(train_df), margin=1.0, gamma=0.9)
```

### 4.2 Optimiseur : SOAP (LibAUC)

SOAP est l'optimiseur compagnon d'APLoss. Il maintient une variable duale interne couplée à l'objectif APLoss et prend en compte la structure non-i.i.d. de l'objectif de ranking par paires. Utiliser Adam+APLoss est sous-optimal car Adam ignore cette structure.

```python
optimizer = SOAP(model.parameters(), lr=1e-5, epoch_decay=1e-6, weight_decay=1e-5)
```

**Bug corrigé :** `epoch_decay=0` cassait l'optimiseur (model_ref=None dans update_regularizer). Valeur minimale : `1e-6`.

**Bug corrigé :** L'appel à `optimizer.update_regularizer(decay_factor=10)` à chaque epoch tuait le LR de façon exponentielle (LR → 0 en quelques epochs). Supprimé.

### 4.3 Sampler : DualSampler

`APLoss` requiert au moins un exemple positif par batch (`pos_mask.sum() > 0`). Avec seulement 11% de positifs, un batch de taille 4 a 65% de chance d'être entièrement négatif.

`DualSampler` garantit au moins `num_pos=1` positif par batch :

```python
train_sampler = DualSampler(train_dataset, batch_size=4, num_pos=1, sampling_rate=None)
```

Le dataset doit exposer un attribut `.targets` (liste des labels) pour que DualSampler identifie les classes.

### 4.4 Validation croisée : StratifiedGroupKFold

5 folds, groupes = subject_id. La contrainte "group" garantit que toutes les sessions d'un même sujet tombent entièrement dans train ou val, jamais dans les deux. Sans cette contrainte, le modèle pourrait apprendre l'anatomie individuelle plutôt que les caractéristiques de lésion, gonflant artificiellement les métriques de validation.

### 4.5 Hyperparamètres

| Paramètre | Valeur | Pourquoi |
|-----------|--------|----------|
| LR | 1e-5 | 1e-4 causait une divergence à epoch 6 |
| Batch size | 4 | Limite GPU (volumes 3D lourds) |
| Epochs | 50 | 100 avec 4466 samples dépasse 48h |
| feature_size | 48 | Standard Swin UNETR, ~62M params |
| Fold | 0 | Fold initial — à répéter sur 0–4 |

---

## 5. Pipeline de données

### 5.1 Sources de labels

Trois sources de labels ont été identifiées et fusionnées :

| Source | Sujets | Contenu | Priorité |
|--------|--------|---------|----------|
| `labels.csv` | 6871 | Labels manuellement curés (WMA + sains) | Haute |
| `all_labels_merged.csv` | 4442 | mrif_score ≥ 3 → label=1 | Moyenne |
| `Baseline_healthy.csv` | 7202 | Sujets sains confirmés, label=0 | Basse |

**Découverte clé :** `Baseline_healthy.csv` ne couvrait que `baseline_year_1_arm_1` (ses-00A), mais ces sujets ont des données pour d'autres sessions sur le disque. La correction : traiter ces sujets comme label=0 pour **toutes leurs sessions** sur disque (pas seulement ses-00A).

### 5.2 Approche disk-first

Le pipeline initial (`phase1_2_build_manifest.py`) partait des labels CSV pour trouver les fichiers → trouvait seulement 421 sujets sur les 2717 présents sur disque.

**Cause :** `labels_all.csv` couvrait 11787 sujets mais ne listait que ses-00A pour les sujets de `Baseline_healthy`. Les sessions ses-02A, ses-04A, ses-06A de ces sujets n'étaient pas cherchées.

**Solution :** `scripts/build_manifest_from_disk.py` — approche inversée :
1. Scanner le disque pour tous les `sub-*/ses-*/anat/` ayant T1w + T2w
2. Pour chaque paire trouvée, chercher le label dans les sources
3. Baseline_healthy = label=0 pour toutes les sessions du sujet

**Résultat :** 4466 lignes (506 WMA, 3960 sains, 2708 sujets uniques)

### 5.3 Distribution par session

| Session | Total | WMA | Sains | Remarque |
|---------|-------|-----|-------|----------|
| ses-00A | 2696 | 145 (5.4%) | 2551 | Baseline, majorité sains |
| ses-02A | 179 | 153 (85%) | 26 | Enrichi en WMA — biais de sélection |
| ses-04A | 1481 | 117 (7.9%) | 1364 | Suivis 4 ans |
| ses-06A | 110 | 91 (82.7%) | 19 | Très enrichi en WMA |

**Note :** Les sessions ses-02A et ses-06A sont très enrichies en WMA. Cela reflète un biais de sélection : les sujets ayant un suivi long sont plus souvent ceux suivis pour une raison médicale. Le modèle devra être évalué par session pour identifier si ce biais affecte les performances.

### 5.4 Preprocessing MONAI

```
LoadImaged(keys=["t1w","t2w"])
EnsureChannelFirstd
Orientationd(axcodes="RAS")
Spacingd(pixdim=(1.0,1.0,1.0), mode=("bilinear","bilinear"))
CropForegroundd(source_key="t1w")
Resized(spatial_size=(96,96,96))
NormalizeIntensityd(nonzero=True, channel_wise=True)
ConcatItemsd(keys=["t1w","t2w"], name="image")
```

Train : augmentation (flips, rotations aléatoires, shifts d'intensité)  
Val : transforms déterministes uniquement

---

## 6. Problèmes rencontrés et solutions

### P1 — API LibAUC 1.3.0 (multiples erreurs)

**Contexte :** LibAUC 1.3.0 a changé son API par rapport aux versions antérieures utilisées dans les exemples en ligne.

| Erreur | Cause | Fix |
|--------|-------|-----|
| `pos_len not in APLoss` | Renommé en `data_len` | `APLoss(data_len=len(train_df), ...)` |
| `num_labels not in APLoss` | Supprimé | Retiré |
| `APLoss.forward missing index` | Nouveau paramètre requis | Retourner `idx` depuis `__getitem__`, passer à la loss |
| `DualSampler assertion` | `sampling_rate` requis | `sampling_rate=None` |
| `DualSampler needs .targets` | Attribut manquant | `self.targets = self.df["label"].astype(int).tolist()` |
| `SOAP doesn't accept loss_fn` | Paramètre supprimé | Retirer `loss_fn=loss_fn` |
| Val loop 3-tuple unpack | `__getitem__` retourne 3 items | `for images, batch_labels, _ in val_loader` |

**Leçon :** Avant tout déploiement sur cluster, tester l'API LibAUC localement avec `inspect.signature()` et un mock dataset.

### P2 — LR collapse (epoch 4–6)

**Symptôme :** AUPREC stagnait à 0.11 (= taux de positifs, = random) dès epoch 4. LR tracé = 0.

**Cause :** `optimizer.update_regularizer(decay_factor=10)` appelé chaque epoch multipliait un terme de régularisation par 10, ce qui annulait effectivement les gradients. Dans LibAUC 1.3.0, cet appel n'est plus recommandé.

**Fix :** Supprimer `update_regularizer`. Réduire LR de 1e-4 à 1e-5.

### P3 — hidden_dim incorrect

**Symptôme :** Crash à l'initialisation : `size mismatch for head.2.weight: 1536 vs 768`.

**Cause :** `hidden_dim = feature_size * 32` — calcul erroné. Le Swin UNETR produit `feature_size × 16` au bottleneck (pas × 32).

**Fix :** `hidden_dim = feature_size * 16  # 768 pour feature_size=48`

### P4 — Seulement 421 sujets au lieu de 2717

**Symptôme :** Build manifest trouve 421 sujets, alors que 2717 ont T1w+T2w sur disque.

**Cause :** Triple cause :
1. `labels_all.csv` ne couvrait que ses-00A pour `Baseline_healthy`
2. Approche labels-first manquait les sessions non listées
3. `ABCD_IMAGING` pointait parfois sur scratch (421 sujets) au lieu de fac (2717)

**Fix :** Approche disk-first + Baseline_healthy couvre toutes les sessions.

### P5 — ABCD_IMAGING pointant sur le mauvais répertoire

**Symptôme :** Training job trouve 421 sujets au lieu de 4466.

**Cause :** `activate_env.sh` exportait `ABCD_IMAGING="/mnt/scratch/user/lbardou/abcd_leuko"` (copie partielle) au lieu de `/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc` (source complète).

**Fix :** Corriger `activate_env.sh`. Confirmer que fac est accessible depuis les nœuds GPU.

### P6 — I/O bottleneck NFS : 15h pour epoch 1

**Symptôme :** GPU-Util = 0% après 4h. 415/4466 fichiers cache construits. Estimation : 40h pour epoch 1.

**Cause :** Le fac storage est un NFS réseau. Lire ~37MB par fichier NIfTI pour ~3556 samples/epoch depuis NFS est très lent. Le GPU attend les données.

**Fix :** Cache de tenseurs pré-transformés sur scratch :
- Epoch 1 : lit NIfTI (fac, lent) + applique transforms + sauvegarde `.pt` sur scratch
- Epochs 2–50 : charge directement `.pt` depuis scratch (rapide)

```python
# Cache par sujet/session (partageable train/val)
cache_path = cache_dir / f"{subject_id}_{session}.pt"
if cache_path.exists():
    image = torch.load(cache_path, weights_only=False)
else:
    image = transform({"t1w": t1w_path, "t2w": t2w_path})["image"]
    torch.save(image, cache_path)
```

**Bug secondaire :** `weights_only=True` refusait les MetaTensors MONAI contenant des arrays numpy. Fix : `weights_only=False` (fichiers trusted).

### P7 — Job interrompu à 15h sans screen

**Symptôme :** rsync et jobs perdus à la fermeture du terminal.

**Fix :** Toujours utiliser `screen -S nom` ou `nohup ... &` avant tout processus long.

---

## 7. Résultats intermédiaires (Fold 0, 421 sujets)

*Run initial avec seulement 421 sujets (avant fix des données)*

| Métrique | Valeur | Référence baseline |
|----------|--------|-------------------|
| Val AUPREC | **0.3613** | 0.12 (taux positifs = random) |
| Val AUROC | **0.70** | 0.50 (random) |
| Epoch meilleure | 94 | — |

**Interprétation :**
- AUPREC = 3× le baseline → le modèle détecte bien les cas positifs
- AUROC = 0.70 → bonne discrimination globale
- Marge de progression significative (target : AUPREC > 0.50)

**Limite :** N=421 sujets très faible pour 62M paramètres. Avec 4466 samples (run en cours), les métriques devraient s'améliorer significativement.

---

## 8. État actuel

| Composant | Statut |
|-----------|--------|
| Architecture `LeukoBinaryClassifier` | ✅ Complet |
| Pipeline MONAI (transforms) | ✅ Complet |
| Boucle d'entraînement APLoss+SOAP | ✅ Complet |
| Manifest 4466 samples (2708 sujets) | ✅ Complet |
| Cache NIfTI→.pt sur scratch | ✅ Implémenté, en cours de construction |
| Training Fold 0 (4466 samples) | 🔄 En cours — epoch 1 (cache building) |
| Folds 1–4 | ⏳ À faire |
| Grad-CAM Phase 4 | ⏳ À faire |
| Segmentation Phase 5 | ⏳ Conditionnel (AUPREC > 0.60) |

---

## 9. Prochaines étapes

1. **Attendre fin epoch 1** (cache en cours de construction) → monitorer `ls /mnt/scratch/user/lbardou/leuko_cache/ | wc -l`
2. **Évaluer résultats Fold 0** avec 4466 samples → comparer avec 0.3613 obtenu sur 421
3. **Lancer folds 1–4** pour obtenir des métriques robustes (5-fold cross-validation)
4. **Analyser le biais de session** — ses-02A et ses-06A très enrichis en WMA, évaluer séparément
5. **Grad-CAM Phase 4** — si AUPREC > 0.50
6. **Considérer** weighted sampling par session pour corriger le biais

---

## 10. Infrastructure

| Élément | Détail |
|---------|--------|
| Cluster | CHPC-UCSF |
| GPU | NVIDIA H100 NVL (95GB VRAM) |
| Données source | `/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc` |
| Cache tenseurs | `/mnt/scratch/user/lbardou/leuko_cache` |
| Env Python | `/mnt/home/lbardou/leuko_env` (Python 3.9, PyTorch 2.5.1+cu124) |
| SLURM | 48h, 8 CPUs, 64GB RAM, 1 GPU H100 |
| Repo | github.com/louanbardou/leukoaraiosis-detection |
