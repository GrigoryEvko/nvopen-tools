// Function: sub_125DC90
// Address: 0x125dc90
//
void sub_125DC90()
{
  _QWORD *v0; // r13
  _QWORD *v1; // r13
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  int *v5; // r13
  int *v6; // r14
  unsigned __int64 v7; // r15
  const void *v8; // rdi
  signed __int64 v9; // rax
  unsigned __int64 v10; // r13
  const void *v11; // rsi
  int v12; // eax
  signed __int64 v13; // rcx
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // r14
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  int *v23; // r15
  int *v24; // r14
  unsigned __int64 v25; // r12
  const void *v26; // rdi
  signed __int64 v27; // rax
  int *v28; // r15
  unsigned __int64 v29; // rcx
  const void *v30; // rsi
  int v31; // eax
  signed __int64 v32; // r8
  _QWORD *v33; // rax
  _QWORD *v34; // rax
  _QWORD *v35; // rax
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r14
  __int64 v40; // rax
  __m128i si128; // xmm0
  _QWORD *v42; // rax
  _QWORD *v43; // rax
  _QWORD *v44; // r14
  __int64 v45; // rax
  __m128i v46; // xmm0
  _QWORD *v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rax
  __m128i v50; // xmm0
  __int64 v51; // r13
  unsigned __int64 v52; // r12
  _QWORD *v53; // r15
  int *v54; // rbx
  unsigned __int64 v55; // r14
  size_t v56; // rdx
  int v57; // eax
  unsigned __int64 v58; // rcx
  size_t v59; // rdx
  int v60; // eax
  __int64 v61; // rax
  _QWORD *v62; // rax
  _QWORD *v63; // rax
  _QWORD *v64; // r14
  __int64 v65; // rax
  __m128i v66; // xmm0
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  _QWORD *v69; // rax
  _QWORD *v70; // rax
  _QWORD *v71; // r14
  __int64 v72; // rax
  __m128i v73; // xmm0
  _QWORD *v74; // r14
  _QWORD *v75; // rax
  _QWORD *v76; // r14
  __int64 v77; // rax
  __m128i v78; // xmm0
  _QWORD *v79; // rax
  _QWORD *v80; // r14
  __int64 v81; // rax
  __m128i v82; // xmm0
  _QWORD *v83; // rax
  __int64 v84; // rax
  __m128i v85; // xmm0
  __int64 v86; // r13
  unsigned __int64 v87; // r12
  _QWORD *v88; // r15
  int *v89; // rbx
  unsigned __int64 v90; // r14
  size_t v91; // rdx
  int v92; // eax
  unsigned __int64 v93; // rcx
  size_t v94; // rdx
  int v95; // eax
  __int64 v96; // rax
  _QWORD *v97; // rax
  _QWORD *v98; // r14
  __m128i *v99; // rax
  __m128i v100; // xmm0
  _QWORD *v101; // rax
  _QWORD *v102; // r14
  __int64 v103; // rax
  __m128i v104; // xmm0
  _QWORD *v105; // r14
  _QWORD *v106; // rax
  _QWORD *v107; // r14
  __int64 v108; // rax
  __m128i v109; // xmm0
  _QWORD *v110; // rax
  _QWORD *v111; // r14
  __int64 v112; // rax
  __m128i v113; // xmm0
  _QWORD *v114; // rax
  __int64 v115; // rax
  __m128i v116; // xmm0
  _QWORD *v117; // r15
  __int64 v118; // r12
  unsigned __int64 v119; // r13
  int *v120; // rbx
  unsigned __int64 v121; // r14
  size_t v122; // rdx
  int v123; // eax
  unsigned __int64 v124; // rcx
  size_t v125; // rdx
  int v126; // eax
  __int64 v127; // rax
  _QWORD *v128; // r14
  __int64 v129; // rax
  __m128i v130; // xmm0
  _QWORD *v131; // rax
  _QWORD *v132; // rax
  _QWORD *v133; // rax
  _QWORD *v134; // rax
  _QWORD *v135; // rax
  _QWORD *v136; // r14
  __int64 v137; // rax
  __m128i v138; // xmm0
  _QWORD *v139; // rax
  _QWORD *v140; // r14
  __int64 v141; // rax
  __m128i v142; // xmm0
  _QWORD *v143; // rax
  _QWORD *v144; // r14
  __int64 v145; // rax
  __m128i v146; // xmm0
  _QWORD *v147; // rax
  __int64 v148; // rax
  _QWORD *v149; // rdi
  __int64 v150; // rax
  _QWORD *v151; // rdi
  unsigned __int64 v152; // [rsp-88h] [rbp-88h]
  unsigned __int64 v153; // [rsp-88h] [rbp-88h]
  __int64 v154; // [rsp-80h] [rbp-80h]
  _QWORD *v155; // [rsp-80h] [rbp-80h]
  _QWORD *v156; // [rsp-80h] [rbp-80h]
  _QWORD *v157; // [rsp-80h] [rbp-80h]
  int v158; // [rsp-80h] [rbp-80h]
  _QWORD *v159; // [rsp-78h] [rbp-78h]
  _QWORD *v160; // [rsp-78h] [rbp-78h]
  int *v161; // [rsp-78h] [rbp-78h]
  int *v162; // [rsp-78h] [rbp-78h]
  int *v163; // [rsp-78h] [rbp-78h]
  unsigned __int64 v164; // [rsp-70h] [rbp-70h]
  __int64 v165; // [rsp-60h] [rbp-60h] BYREF
  __m128i v166; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v167[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( qword_4F92C48 )
    return;
  v0 = (_QWORD *)sub_22077B0(16);
  if ( v0 )
  {
    v0[1] = 0;
    *v0 = "--m32";
  }
  strcpy((char *)v167, "-m32");
  v166.m128i_i64[0] = (__int64)v167;
  v166.m128i_i64[1] = 4;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v0;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v1 = (_QWORD *)sub_22077B0(16);
  if ( v1 )
  {
    v1[1] = 0;
    *v1 = "--m64";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-m64");
  v166.m128i_i64[1] = 4;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v1;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v2 = (_QWORD *)sub_22077B0(16);
  if ( v2 )
  {
    *v2 = 0;
    v2[1] = "-fast-math";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-fast-math");
  v166.m128i_i64[1] = 10;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v2;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v3 = (_QWORD *)sub_22077B0(16);
  if ( v3 )
  {
    *v3 = 0;
    v3[1] = "-ftz=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-ftz=1");
  v166.m128i_i64[1] = 6;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v3;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v4 = (_QWORD *)sub_22077B0(16);
  v159 = v4;
  if ( v4 )
  {
    *v4 = 0;
    v4[1] = "-ftz=0";
  }
  v5 = (int *)qword_4F92C30;
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-ftz=0");
  v6 = &dword_4F92C28;
  v166.m128i_i64[1] = 6;
  if ( !qword_4F92C30 )
    goto LABEL_296;
  do
  {
    while ( 1 )
    {
      v7 = *((_QWORD *)v5 + 5);
      v8 = (const void *)*((_QWORD *)v5 + 4);
      if ( v7 > 6 )
        break;
      LODWORD(v9) = -6;
      if ( v7 )
      {
        LODWORD(v9) = memcmp(v8, v167, *((_QWORD *)v5 + 5));
        if ( !(_DWORD)v9 )
          LODWORD(v9) = v7 - 6;
      }
LABEL_25:
      if ( (int)v9 >= 0 )
        goto LABEL_26;
LABEL_22:
      v5 = (int *)*((_QWORD *)v5 + 3);
      if ( !v5 )
        goto LABEL_27;
    }
    LODWORD(v9) = memcmp(v8, v167, 6u);
    if ( (_DWORD)v9 )
      goto LABEL_25;
    v9 = v7 - 6;
    if ( (__int64)(v7 - 6) < 0x80000000LL )
    {
      if ( v9 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_22;
      goto LABEL_25;
    }
LABEL_26:
    v6 = v5;
    v5 = (int *)*((_QWORD *)v5 + 2);
  }
  while ( v5 );
LABEL_27:
  if ( v6 == &dword_4F92C28 )
    goto LABEL_296;
  v10 = *((_QWORD *)v6 + 5);
  v11 = (const void *)*((_QWORD *)v6 + 4);
  if ( v10 <= 5 )
  {
    LODWORD(v13) = 6;
    if ( !v10 || (v12 = memcmp(v167, v11, *((_QWORD *)v6 + 5)), LODWORD(v13) = 6 - v10, !v12) )
LABEL_32:
      v12 = v13;
LABEL_33:
    if ( v12 < 0 )
      goto LABEL_296;
    goto LABEL_34;
  }
  v12 = memcmp(v167, v11, 6u);
  if ( v12 )
    goto LABEL_33;
  v13 = 6 - v10;
  if ( (__int64)(6 - v10) > 0x7FFFFFFF )
  {
LABEL_34:
    *((_QWORD *)v6 + 8) = v159;
    goto LABEL_35;
  }
  if ( v13 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_32;
LABEL_296:
  v165 = (__int64)&v166;
  v148 = sub_125D9E0(&qword_4F92C20, v6, (__m128i **)&v165);
  v149 = (_QWORD *)v166.m128i_i64[0];
  *(_QWORD *)(v148 + 64) = v159;
  if ( v149 != v167 )
    j_j___libc_free_0(v149, v167[0] + 1LL);
LABEL_35:
  v14 = (_QWORD *)sub_22077B0(16);
  if ( v14 )
  {
    *v14 = 0;
    v14[1] = "-prec-sqrt=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-prec_sqrt=1");
  v166.m128i_i64[1] = 12;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v14;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v15 = (_QWORD *)sub_22077B0(16);
  if ( v15 )
  {
    *v15 = 0;
    v15[1] = "-prec-sqrt=0";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-prec_sqrt=0");
  v166.m128i_i64[1] = 12;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v15;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v16 = (_QWORD *)sub_22077B0(16);
  v17 = v16;
  if ( v16 )
  {
    *v16 = 0;
    v16[1] = "-disable-allopts";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 16;
  v166.m128i_i64[0] = sub_22409D0(&v166, &v165, 0);
  v167[0] = v165;
  *(__m128i *)v166.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3E9F880);
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v17;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v18 = (_QWORD *)sub_22077B0(16);
  if ( v18 )
  {
    *v18 = 0;
    v18[1] = "-prec-div=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-prec_div=1");
  v166.m128i_i64[1] = 11;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v18;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v19 = (_QWORD *)sub_22077B0(16);
  if ( v19 )
  {
    *v19 = 0;
    v19[1] = "-prec-div=0";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-prec_div=0");
  v166.m128i_i64[1] = 11;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v19;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v20 = (_QWORD *)sub_22077B0(16);
  if ( v20 )
  {
    *v20 = 0;
    v20[1] = "-fma=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-fmad=1");
  v166.m128i_i64[1] = 7;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v20;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v21 = (_QWORD *)sub_22077B0(16);
  if ( v21 )
  {
    *v21 = 0;
    v21[1] = "-fma=0";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-fmad=0");
  v166.m128i_i64[1] = 7;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v21;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v22 = (_QWORD *)sub_22077B0(16);
  v160 = v22;
  if ( v22 )
  {
    *v22 = "--device-O=0";
    v22[1] = "-opt=0";
  }
  v23 = (int *)qword_4F92C30;
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-O0");
  v166.m128i_i64[1] = 3;
  if ( !qword_4F92C30 )
  {
    v28 = &dword_4F92C28;
    goto LABEL_305;
  }
  v24 = &dword_4F92C28;
  while ( 2 )
  {
    while ( 2 )
    {
      v25 = *((_QWORD *)v23 + 5);
      v26 = (const void *)*((_QWORD *)v23 + 4);
      if ( v25 <= 3 )
      {
        LODWORD(v27) = -3;
        if ( v25 )
        {
          LODWORD(v27) = memcmp(v26, v167, *((_QWORD *)v23 + 5));
          if ( !(_DWORD)v27 )
            LODWORD(v27) = v25 - 3;
        }
        goto LABEL_70;
      }
      LODWORD(v27) = memcmp(v26, v167, 3u);
      if ( (_DWORD)v27 )
      {
LABEL_70:
        if ( (int)v27 >= 0 )
          break;
        goto LABEL_67;
      }
      v27 = v25 - 3;
      if ( (__int64)(v25 - 3) < 0x80000000LL )
      {
        if ( v27 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_70;
LABEL_67:
        v23 = (int *)*((_QWORD *)v23 + 3);
        if ( !v23 )
          goto LABEL_72;
        continue;
      }
      break;
    }
    v24 = v23;
    v23 = (int *)*((_QWORD *)v23 + 2);
    if ( v23 )
      continue;
    break;
  }
LABEL_72:
  v28 = v24;
  if ( v24 == &dword_4F92C28 )
    goto LABEL_305;
  v29 = *((_QWORD *)v24 + 5);
  v30 = (const void *)*((_QWORD *)v24 + 4);
  if ( v29 <= 2 )
  {
    LODWORD(v32) = 3;
    if ( v29 )
    {
      v158 = *((_QWORD *)v24 + 5);
      v31 = memcmp(v167, v30, *((_QWORD *)v24 + 5));
      if ( v31 )
      {
LABEL_78:
        if ( v31 < 0 )
          goto LABEL_305;
        goto LABEL_79;
      }
      LODWORD(v32) = 3 - v158;
    }
LABEL_77:
    v31 = v32;
    goto LABEL_78;
  }
  v154 = *((_QWORD *)v24 + 5);
  v31 = memcmp(v167, v30, 3u);
  if ( v31 )
    goto LABEL_78;
  v32 = 3 - v154;
  if ( 3 - v154 > 0x7FFFFFFF )
  {
LABEL_79:
    *((_QWORD *)v24 + 8) = v160;
    goto LABEL_80;
  }
  if ( v32 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_77;
LABEL_305:
  v165 = (__int64)&v166;
  v150 = sub_125D9E0(&qword_4F92C20, v28, (__m128i **)&v165);
  v151 = (_QWORD *)v166.m128i_i64[0];
  *(_QWORD *)(v150 + 64) = v160;
  if ( v151 != v167 )
    j_j___libc_free_0(v151, v167[0] + 1LL);
LABEL_80:
  v33 = (_QWORD *)sub_22077B0(16);
  if ( v33 )
  {
    *v33 = "--device-O=1";
    v33[1] = "-opt=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-O1");
  v166.m128i_i64[1] = 3;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v33;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v34 = (_QWORD *)sub_22077B0(16);
  if ( v34 )
  {
    *v34 = "--device-O=2";
    v34[1] = "-opt=2";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-O2");
  v166.m128i_i64[1] = 3;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v34;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v35 = (_QWORD *)sub_22077B0(16);
  if ( v35 )
  {
    *v35 = "--device-O=3";
    v35[1] = "-opt=3";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-O3");
  v166.m128i_i64[1] = 3;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v35;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v36 = (_QWORD *)sub_22077B0(16);
  if ( v36 )
  {
    *v36 = 0;
    v36[1] = "-Osize";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-Osize");
  v166.m128i_i64[1] = 6;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v36;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v37 = (_QWORD *)sub_22077B0(16);
  if ( v37 )
  {
    *v37 = 0;
    v37[1] = "-Om";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-Om");
  v166.m128i_i64[1] = 3;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v37;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v38 = (_QWORD *)sub_22077B0(16);
  v39 = v38;
  if ( v38 )
  {
    *v38 = 0;
    v38[1] = "-Ofast-compile=max";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 19;
  v40 = sub_22409D0(&v166, &v165, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v166.m128i_i64[0] = v40;
  v167[0] = v165;
  *(_WORD *)(v40 + 16) = 24941;
  *(_BYTE *)(v40 + 18) = 120;
  *(__m128i *)v40 = si128;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v39;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v42 = (_QWORD *)sub_22077B0(16);
  if ( v42 )
  {
    *v42 = 0;
    v42[1] = "-Ofast-compile=max";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-Ofc=max");
  v166.m128i_i64[1] = 8;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v42;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v43 = (_QWORD *)sub_22077B0(16);
  v44 = v43;
  if ( v43 )
  {
    *v43 = 0;
    v43[1] = "-Ofast-compile=mid";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 19;
  v45 = sub_22409D0(&v166, &v165, 0);
  v46 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v166.m128i_i64[0] = v45;
  v167[0] = v165;
  *(_WORD *)(v45 + 16) = 26989;
  *(_BYTE *)(v45 + 18) = 100;
  *(__m128i *)v45 = v46;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v44;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v47 = (_QWORD *)sub_22077B0(16);
  if ( v47 )
  {
    *v47 = 0;
    v47[1] = "-Ofast-compile=mid";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-Ofc=mid");
  v166.m128i_i64[1] = 8;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v47;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v48 = (_QWORD *)sub_22077B0(16);
  v155 = v48;
  if ( v48 )
  {
    *v48 = 0;
    v48[1] = "-Ofast-compile=min";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 19;
  v49 = sub_22409D0(&v166, &v165, 0);
  v50 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v166.m128i_i64[0] = v49;
  v167[0] = v165;
  *(_WORD *)(v49 + 16) = 26989;
  *(_BYTE *)(v49 + 18) = 110;
  *(__m128i *)v49 = v50;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  if ( !qword_4F92C30 )
  {
    v161 = &dword_4F92C28;
    goto LABEL_139;
  }
  v51 = qword_4F92C30;
  v52 = v166.m128i_u64[1];
  v53 = (_QWORD *)v166.m128i_i64[0];
  v54 = &dword_4F92C28;
  while ( 2 )
  {
    while ( 2 )
    {
      v55 = *(_QWORD *)(v51 + 40);
      v56 = v52;
      if ( v55 <= v52 )
        v56 = *(_QWORD *)(v51 + 40);
      if ( !v56 || (v57 = memcmp(*(const void **)(v51 + 32), v53, v56)) == 0 )
      {
        if ( (__int64)(v55 - v52) >= 0x80000000LL )
          goto LABEL_129;
        if ( (__int64)(v55 - v52) > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v57 = v55 - v52;
          break;
        }
LABEL_120:
        v51 = *(_QWORD *)(v51 + 24);
        if ( !v51 )
          goto LABEL_130;
        continue;
      }
      break;
    }
    if ( v57 < 0 )
      goto LABEL_120;
LABEL_129:
    v54 = (int *)v51;
    v51 = *(_QWORD *)(v51 + 16);
    if ( v51 )
      continue;
    break;
  }
LABEL_130:
  v161 = v54;
  if ( v54 == &dword_4F92C28 )
    goto LABEL_139;
  v58 = *((_QWORD *)v54 + 5);
  v59 = v52;
  if ( v58 <= v52 )
    v59 = *((_QWORD *)v54 + 5);
  if ( v59 && (v152 = *((_QWORD *)v54 + 5), v60 = memcmp(v53, *((const void **)v54 + 4), v59), v58 = v152, v60) )
  {
LABEL_138:
    if ( v60 < 0 )
      goto LABEL_139;
  }
  else if ( (__int64)(v52 - v58) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v52 - v58) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v60 = v52 - v58;
      goto LABEL_138;
    }
LABEL_139:
    v165 = (__int64)&v166;
    v61 = sub_125D9E0(&qword_4F92C20, v161, (__m128i **)&v165);
    v53 = (_QWORD *)v166.m128i_i64[0];
    v161 = (int *)v61;
  }
  *((_QWORD *)v161 + 8) = v155;
  if ( v53 != v167 )
    j_j___libc_free_0(v53, v167[0] + 1LL);
  v62 = (_QWORD *)sub_22077B0(16);
  if ( v62 )
  {
    *v62 = 0;
    v62[1] = "-Ofast-compile=min";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-Ofc=min");
  v166.m128i_i64[1] = 8;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v62;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v63 = (_QWORD *)sub_22077B0(16);
  v64 = v63;
  if ( v63 )
  {
    *v63 = 0;
    v63[1] = 0;
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 17;
  v65 = sub_22409D0(&v166, &v165, 0);
  v66 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v166.m128i_i64[0] = v65;
  v167[0] = v165;
  *(_BYTE *)(v65 + 16) = 48;
  *(__m128i *)v65 = v66;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v64;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v67 = (_QWORD *)sub_22077B0(16);
  if ( v67 )
  {
    *v67 = 0;
    v67[1] = 0;
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-Ofc=0");
  v166.m128i_i64[1] = 6;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v67;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v68 = (_QWORD *)sub_22077B0(16);
  if ( v68 )
  {
    *v68 = "--device-debug";
    v68[1] = "-g";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-g");
  v166.m128i_i64[1] = 2;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v68;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v69 = (_QWORD *)sub_22077B0(16);
  if ( v69 )
  {
    *v69 = 0;
    v69[1] = "-show-src";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-show-src");
  v166.m128i_i64[1] = 9;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v69;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v70 = (_QWORD *)sub_22077B0(16);
  v71 = v70;
  if ( v70 )
  {
    *v70 = 0;
    v70[1] = "disable-llc-opts";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 17;
  v72 = sub_22409D0(&v166, &v165, 0);
  v73 = _mm_load_si128((const __m128i *)&xmmword_3E9F8A0);
  v166.m128i_i64[0] = v72;
  v167[0] = v165;
  *(_BYTE *)(v72 + 16) = 115;
  *(__m128i *)v72 = v73;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v71;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v74 = (_QWORD *)sub_22077B0(16);
  if ( v74 )
  {
    *v74 = "-w";
    v74[1] = "-w";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-w");
  v166.m128i_i64[1] = 2;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v74;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v75 = (_QWORD *)sub_22077B0(16);
  v76 = v75;
  if ( v75 )
  {
    *v75 = 0;
    v75[1] = "-Wno-memory-space";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 17;
  v77 = sub_22409D0(&v166, &v165, 0);
  v78 = _mm_load_si128((const __m128i *)&xmmword_3E9F8B0);
  v166.m128i_i64[0] = v77;
  v167[0] = v165;
  *(_BYTE *)(v77 + 16) = 101;
  *(__m128i *)v77 = v78;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v76;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v79 = (_QWORD *)sub_22077B0(16);
  v80 = v79;
  if ( v79 )
  {
    *v79 = 0;
    v79[1] = "-disable-inlining";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 17;
  v81 = sub_22409D0(&v166, &v165, 0);
  v82 = _mm_load_si128((const __m128i *)&xmmword_3E9F8C0);
  v166.m128i_i64[0] = v81;
  v167[0] = v165;
  *(_BYTE *)(v81 + 16) = 103;
  *(__m128i *)v81 = v82;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v80;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v83 = (_QWORD *)sub_22077B0(16);
  v156 = v83;
  if ( v83 )
  {
    *v83 = 0;
    v83[1] = "-aggressive-inline";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 18;
  v84 = sub_22409D0(&v166, &v165, 0);
  v85 = _mm_load_si128((const __m128i *)&xmmword_3E9F8D0);
  v166.m128i_i64[0] = v84;
  v167[0] = v165;
  *(_WORD *)(v84 + 16) = 25966;
  *(__m128i *)v84 = v85;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  if ( !qword_4F92C30 )
  {
    v162 = &dword_4F92C28;
    goto LABEL_201;
  }
  v86 = qword_4F92C30;
  v87 = v166.m128i_u64[1];
  v88 = (_QWORD *)v166.m128i_i64[0];
  v89 = &dword_4F92C28;
  while ( 2 )
  {
    while ( 2 )
    {
      v90 = *(_QWORD *)(v86 + 40);
      v91 = v87;
      if ( v90 <= v87 )
        v91 = *(_QWORD *)(v86 + 40);
      if ( !v91 || (v92 = memcmp(*(const void **)(v86 + 32), v88, v91)) == 0 )
      {
        if ( (__int64)(v90 - v87) >= 0x80000000LL )
          goto LABEL_191;
        if ( (__int64)(v90 - v87) > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v92 = v90 - v87;
          break;
        }
LABEL_182:
        v86 = *(_QWORD *)(v86 + 24);
        if ( !v86 )
          goto LABEL_192;
        continue;
      }
      break;
    }
    if ( v92 < 0 )
      goto LABEL_182;
LABEL_191:
    v89 = (int *)v86;
    v86 = *(_QWORD *)(v86 + 16);
    if ( v86 )
      continue;
    break;
  }
LABEL_192:
  v162 = v89;
  if ( v89 == &dword_4F92C28 )
    goto LABEL_201;
  v93 = *((_QWORD *)v89 + 5);
  v94 = v87;
  if ( v93 <= v87 )
    v94 = *((_QWORD *)v89 + 5);
  if ( v94 && (v153 = *((_QWORD *)v89 + 5), v95 = memcmp(v88, *((const void **)v89 + 4), v94), v93 = v153, v95) )
  {
LABEL_200:
    if ( v95 < 0 )
      goto LABEL_201;
  }
  else if ( (__int64)(v87 - v93) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v87 - v93) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v95 = v87 - v93;
      goto LABEL_200;
    }
LABEL_201:
    v165 = (__int64)&v166;
    v96 = sub_125D9E0(&qword_4F92C20, v162, (__m128i **)&v165);
    v88 = (_QWORD *)v166.m128i_i64[0];
    v162 = (int *)v96;
  }
  *((_QWORD *)v162 + 8) = v156;
  if ( v88 != v167 )
    j_j___libc_free_0(v88, v167[0] + 1LL);
  v97 = (_QWORD *)sub_22077B0(16);
  v98 = v97;
  if ( v97 )
  {
    *v97 = "--kernel-params-are-restrict";
    v97[1] = "-restrict";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 27;
  v99 = (__m128i *)sub_22409D0(&v166, &v165, 0);
  v100 = _mm_load_si128((const __m128i *)&xmmword_3E9F8E0);
  v166.m128i_i64[0] = (__int64)v99;
  v167[0] = v165;
  qmemcpy(&v99[1], "re-restrict", 11);
  *v99 = v100;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v98;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v101 = (_QWORD *)sub_22077B0(16);
  v102 = v101;
  if ( v101 )
  {
    *v101 = 0;
    v101[1] = "-allow-restrict-in-struct";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 24;
  v103 = sub_22409D0(&v166, &v165, 0);
  v104 = _mm_load_si128((const __m128i *)&xmmword_3E9F8F0);
  v166.m128i_i64[0] = v103;
  v167[0] = v165;
  *(_QWORD *)(v103 + 16) = 0x7463757274732D6ELL;
  *(__m128i *)v103 = v104;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v102;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v105 = (_QWORD *)sub_22077B0(16);
  if ( v105 )
  {
    *v105 = "--device-c";
    v105[1] = "--device-c";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "--device-c");
  v166.m128i_i64[1] = 10;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v105;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v106 = (_QWORD *)sub_22077B0(16);
  v107 = v106;
  if ( v106 )
  {
    *v106 = "--generate-line-info";
    v106[1] = "-generate-line-info";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 19;
  v108 = sub_22409D0(&v166, &v165, 0);
  v109 = _mm_load_si128((const __m128i *)&xmmword_3E9F900);
  v166.m128i_i64[0] = v108;
  v167[0] = v165;
  *(_WORD *)(v108 + 16) = 26222;
  *(_BYTE *)(v108 + 18) = 111;
  *(__m128i *)v108 = v109;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v107;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v110 = (_QWORD *)sub_22077B0(16);
  v111 = v110;
  if ( v110 )
  {
    *v110 = "--enable-opt-byval";
    v110[1] = "-enable-opt-byval";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 17;
  v112 = sub_22409D0(&v166, &v165, 0);
  v113 = _mm_load_si128((const __m128i *)&xmmword_3E9F910);
  v166.m128i_i64[0] = v112;
  v167[0] = v165;
  *(_BYTE *)(v112 + 16) = 108;
  *(__m128i *)v112 = v113;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v111;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v114 = (_QWORD *)sub_22077B0(16);
  v157 = v114;
  if ( v114 )
  {
    *v114 = 0;
    v114[1] = "-no-lineinfo-inlined-at";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 23;
  v115 = sub_22409D0(&v166, &v165, 0);
  v116 = _mm_load_si128((const __m128i *)&xmmword_3E9F920);
  v166.m128i_i64[0] = v115;
  v167[0] = v165;
  *(_DWORD *)(v115 + 16) = 1684369001;
  *(_WORD *)(v115 + 20) = 24877;
  *(_BYTE *)(v115 + 22) = 116;
  *(__m128i *)v115 = v116;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  if ( !qword_4F92C30 )
  {
    v163 = &dword_4F92C28;
    goto LABEL_247;
  }
  v117 = (_QWORD *)v166.m128i_i64[0];
  v118 = qword_4F92C30;
  v119 = v166.m128i_u64[1];
  v120 = &dword_4F92C28;
  while ( 2 )
  {
    while ( 2 )
    {
      v121 = *(_QWORD *)(v118 + 40);
      v122 = v119;
      if ( v121 <= v119 )
        v122 = *(_QWORD *)(v118 + 40);
      if ( !v122 || (v123 = memcmp(*(const void **)(v118 + 32), v117, v122)) == 0 )
      {
        if ( (__int64)(v121 - v119) >= 0x80000000LL )
          goto LABEL_237;
        if ( (__int64)(v121 - v119) > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v123 = v121 - v119;
          break;
        }
LABEL_228:
        v118 = *(_QWORD *)(v118 + 24);
        if ( !v118 )
          goto LABEL_238;
        continue;
      }
      break;
    }
    if ( v123 < 0 )
      goto LABEL_228;
LABEL_237:
    v120 = (int *)v118;
    v118 = *(_QWORD *)(v118 + 16);
    if ( v118 )
      continue;
    break;
  }
LABEL_238:
  v163 = v120;
  if ( v120 == &dword_4F92C28 )
    goto LABEL_247;
  v124 = *((_QWORD *)v120 + 5);
  v125 = v119;
  if ( v124 <= v119 )
    v125 = *((_QWORD *)v120 + 5);
  if ( v125 && (v164 = *((_QWORD *)v120 + 5), v126 = memcmp(v117, *((const void **)v120 + 4), v125), v124 = v164, v126) )
  {
LABEL_246:
    if ( v126 < 0 )
      goto LABEL_247;
  }
  else if ( (__int64)(v119 - v124) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v119 - v124) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v126 = v119 - v124;
      goto LABEL_246;
    }
LABEL_247:
    v165 = (__int64)&v166;
    v127 = sub_125D9E0(&qword_4F92C20, v163, (__m128i **)&v165);
    v117 = (_QWORD *)v166.m128i_i64[0];
    v163 = (int *)v127;
  }
  *((_QWORD *)v163 + 8) = v157;
  if ( v117 != v167 )
    j_j___libc_free_0(v117, v167[0] + 1LL);
  v128 = (_QWORD *)sub_22077B0(16);
  if ( v128 )
  {
    v128[1] = 0;
    *v128 = "--keep-device-functions";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 23;
  v129 = sub_22409D0(&v166, &v165, 0);
  v130 = _mm_load_si128((const __m128i *)&xmmword_3E9F930);
  v166.m128i_i64[0] = v129;
  v167[0] = v165;
  *(_DWORD *)(v129 + 16) = 1769235310;
  *(_WORD *)(v129 + 20) = 28271;
  *(_BYTE *)(v129 + 22) = 115;
  *(__m128i *)v129 = v130;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v128;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v131 = (_QWORD *)sub_22077B0(16);
  if ( v131 )
  {
    *v131 = "--emit-lifetime-intrinsics";
    v131[1] = "--emit-optix-ir";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "--emit-optix-ir");
  v166.m128i_i64[1] = 15;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v131;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v132 = (_QWORD *)sub_22077B0(16);
  if ( v132 )
  {
    *v132 = 0;
    v132[1] = "-opt-fdiv=0";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-opt-fdiv=0");
  v166.m128i_i64[1] = 11;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v132;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v133 = (_QWORD *)sub_22077B0(16);
  if ( v133 )
  {
    *v133 = 0;
    v133[1] = "-opt-fdiv=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-opt-fdiv=1");
  v166.m128i_i64[1] = 11;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v133;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v134 = (_QWORD *)sub_22077B0(16);
  if ( v134 )
  {
    *v134 = 0;
    v134[1] = "-new-nvvm-remat";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-new-nvvm-remat");
  v166.m128i_i64[1] = 15;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v134;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v135 = (_QWORD *)sub_22077B0(16);
  v136 = v135;
  if ( v135 )
  {
    *v135 = 0;
    v135[1] = "-disable-new-nvvm-remat";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 23;
  v137 = sub_22409D0(&v166, &v165, 0);
  v138 = _mm_load_si128((const __m128i *)&xmmword_3E9F940);
  v166.m128i_i64[0] = v137;
  v167[0] = v165;
  *(_DWORD *)(v137 + 16) = 1701981549;
  *(_WORD *)(v137 + 20) = 24941;
  *(_BYTE *)(v137 + 22) = 116;
  *(__m128i *)v137 = v138;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v136;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v139 = (_QWORD *)sub_22077B0(16);
  v140 = v139;
  if ( v139 )
  {
    *v139 = 0;
    v139[1] = "-disable-nvvm-remat";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 19;
  v141 = sub_22409D0(&v166, &v165, 0);
  v142 = _mm_load_si128((const __m128i *)&xmmword_3E9F950);
  v166.m128i_i64[0] = v141;
  v167[0] = v165;
  *(_WORD *)(v141 + 16) = 24941;
  *(_BYTE *)(v141 + 18) = 116;
  *(__m128i *)v141 = v142;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v140;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v143 = (_QWORD *)sub_22077B0(16);
  v144 = v143;
  if ( v143 )
  {
    *v143 = "--discard_value_names=1";
    v143[1] = "-discard-value-names=1";
  }
  v166.m128i_i64[0] = (__int64)v167;
  v165 = 20;
  v145 = sub_22409D0(&v166, &v165, 0);
  v146 = _mm_load_si128((const __m128i *)&xmmword_3E9F960);
  v166.m128i_i64[0] = v145;
  v167[0] = v165;
  *(_DWORD *)(v145 + 16) = 1936026977;
  *(__m128i *)v145 = v146;
  v166.m128i_i64[1] = v165;
  *(_BYTE *)(v166.m128i_i64[0] + v165) = 0;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v144;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
  v147 = (_QWORD *)sub_22077B0(16);
  if ( v147 )
  {
    *v147 = 0;
    v147[1] = "-gen-opt-lto";
  }
  v166.m128i_i64[0] = (__int64)v167;
  strcpy((char *)v167, "-gen-opt-lto");
  v166.m128i_i64[1] = 12;
  *(_QWORD *)sub_125DB60(&qword_4F92C20, &v166) = v147;
  if ( (_QWORD *)v166.m128i_i64[0] != v167 )
    j_j___libc_free_0(v166.m128i_i64[0], v167[0] + 1LL);
}
