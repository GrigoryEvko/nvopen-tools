// Function: sub_8FE280
// Address: 0x8fe280
//
void sub_8FE280()
{
  _QWORD *v0; // r13
  _QWORD *v1; // r13
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  int *v12; // r15
  int *v13; // r14
  unsigned __int64 v14; // r12
  const void *v15; // rdi
  signed __int64 v16; // rax
  int *v17; // r15
  unsigned __int64 v18; // rcx
  const void *v19; // rsi
  int v20; // eax
  signed __int64 v21; // r8
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  _QWORD *v30; // r14
  __int64 v31; // rax
  __m128i si128; // xmm0
  _QWORD *v33; // rax
  _QWORD *v34; // rax
  _QWORD *v35; // r14
  __int64 v36; // rax
  __m128i v37; // xmm0
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  _QWORD *v40; // r14
  __int64 v41; // rax
  __m128i v42; // xmm0
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  _QWORD *v45; // r14
  __int64 v46; // rax
  __m128i v47; // xmm0
  _QWORD *v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rax
  _QWORD *v52; // r14
  __int64 v53; // rax
  __m128i v54; // xmm0
  _QWORD *v55; // r14
  _QWORD *v56; // rax
  _QWORD *v57; // r14
  __int64 v58; // rax
  __m128i v59; // xmm0
  _QWORD *v60; // rax
  _QWORD *v61; // r14
  __int64 v62; // rax
  __m128i v63; // xmm0
  _QWORD *v64; // rax
  _QWORD *v65; // r14
  __int64 v66; // rax
  __m128i v67; // xmm0
  _QWORD *v68; // rax
  _QWORD *v69; // r14
  __m128i *v70; // rax
  __m128i v71; // xmm0
  _QWORD *v72; // rax
  __int64 v73; // rax
  __m128i v74; // xmm0
  __int64 v75; // r13
  unsigned __int64 v76; // r12
  _QWORD *v77; // r15
  int *v78; // rbx
  unsigned __int64 v79; // r14
  size_t v80; // rdx
  int v81; // eax
  unsigned __int64 v82; // rcx
  size_t v83; // rdx
  int v84; // eax
  __int64 v85; // rax
  _QWORD *v86; // r14
  _QWORD *v87; // rax
  _QWORD *v88; // r14
  __int64 v89; // rax
  __m128i v90; // xmm0
  _QWORD *v91; // rax
  __int64 v92; // rax
  __m128i v93; // xmm0
  __int64 v94; // r13
  unsigned __int64 v95; // r12
  _QWORD *v96; // r15
  int *v97; // rbx
  unsigned __int64 v98; // r14
  size_t v99; // rdx
  int v100; // eax
  unsigned __int64 v101; // rcx
  size_t v102; // rdx
  int v103; // eax
  __int64 v104; // rax
  _QWORD *v105; // rax
  __int64 v106; // rax
  __m128i v107; // xmm0
  _QWORD *v108; // r15
  __int64 v109; // r12
  unsigned __int64 v110; // r13
  int *v111; // rbx
  unsigned __int64 v112; // r14
  size_t v113; // rdx
  int v114; // eax
  unsigned __int64 v115; // rcx
  size_t v116; // rdx
  int v117; // eax
  __int64 v118; // rax
  _QWORD *v119; // r14
  __int64 v120; // rax
  __m128i v121; // xmm0
  _QWORD *v122; // rax
  _QWORD *v123; // rax
  _QWORD *v124; // rax
  _QWORD *v125; // rax
  _QWORD *v126; // rax
  _QWORD *v127; // r14
  __int64 v128; // rax
  __m128i v129; // xmm0
  _QWORD *v130; // rax
  _QWORD *v131; // r14
  __int64 v132; // rax
  __m128i v133; // xmm0
  _QWORD *v134; // rax
  _QWORD *v135; // r14
  __int64 v136; // rax
  __m128i v137; // xmm0
  _QWORD *v138; // rax
  int *v139; // r15
  int *v140; // r14
  unsigned __int64 v141; // r12
  const void *v142; // rdi
  signed __int64 v143; // rax
  int *v144; // r15
  unsigned __int64 v145; // rcx
  const void *v146; // rsi
  int v147; // eax
  signed __int64 v148; // r8
  __int64 v149; // rax
  _QWORD *v150; // rdi
  __int64 v151; // rax
  _QWORD *v152; // rdi
  unsigned __int64 v153; // [rsp-88h] [rbp-88h]
  unsigned __int64 v154; // [rsp-88h] [rbp-88h]
  unsigned __int64 v155; // [rsp-88h] [rbp-88h]
  __int64 v156; // [rsp-80h] [rbp-80h]
  _QWORD *v157; // [rsp-80h] [rbp-80h]
  _QWORD *v158; // [rsp-80h] [rbp-80h]
  _QWORD *v159; // [rsp-80h] [rbp-80h]
  int v160; // [rsp-80h] [rbp-80h]
  _QWORD *v161; // [rsp-78h] [rbp-78h]
  int *v162; // [rsp-78h] [rbp-78h]
  int *v163; // [rsp-78h] [rbp-78h]
  int *v164; // [rsp-78h] [rbp-78h]
  _QWORD *v165; // [rsp-78h] [rbp-78h]
  __int64 v166; // [rsp-70h] [rbp-70h]
  int v167; // [rsp-70h] [rbp-70h]
  __int64 v168; // [rsp-60h] [rbp-60h] BYREF
  __m128i v169; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v170[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( qword_4F6D2C8 )
    return;
  v0 = (_QWORD *)sub_22077B0(16);
  if ( v0 )
  {
    v0[1] = 0;
    *v0 = "--m32";
  }
  strcpy((char *)v170, "-m32");
  v169.m128i_i64[0] = (__int64)v170;
  v169.m128i_i64[1] = 4;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v0;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v1 = (_QWORD *)sub_22077B0(16);
  if ( v1 )
  {
    v1[1] = 0;
    *v1 = "--m64";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-m64");
  v169.m128i_i64[1] = 4;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v1;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v2 = (_QWORD *)sub_22077B0(16);
  if ( v2 )
  {
    *v2 = 0;
    v2[1] = "-fast-math";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-fast-math");
  v169.m128i_i64[1] = 10;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v2;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v3 = (_QWORD *)sub_22077B0(16);
  if ( v3 )
  {
    *v3 = 0;
    v3[1] = "-ftz=1";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-ftz=1");
  v169.m128i_i64[1] = 6;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v3;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v4 = (_QWORD *)sub_22077B0(16);
  if ( v4 )
  {
    *v4 = 0;
    v4[1] = "-ftz=0";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-ftz=0");
  v169.m128i_i64[1] = 6;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v4;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v5 = (_QWORD *)sub_22077B0(16);
  if ( v5 )
  {
    *v5 = 0;
    v5[1] = "-prec-sqrt=1";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-prec_sqrt=1");
  v169.m128i_i64[1] = 12;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v5;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v6 = (_QWORD *)sub_22077B0(16);
  if ( v6 )
  {
    *v6 = 0;
    v6[1] = "-prec-sqrt=0";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-prec_sqrt=0");
  v169.m128i_i64[1] = 12;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v6;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v7 = (_QWORD *)sub_22077B0(16);
  v8 = v7;
  if ( v7 )
  {
    *v7 = 0;
    v7[1] = "-disable-allopts";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 16;
  v169.m128i_i64[0] = sub_22409D0(&v169, &v168, 0);
  v170[0] = v168;
  *(__m128i *)v169.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3E9F880);
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v8;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v9 = (_QWORD *)sub_22077B0(16);
  if ( v9 )
  {
    *v9 = 0;
    v9[1] = "-prec-div=1";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-prec_div=1");
  v169.m128i_i64[1] = 11;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v9;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v10 = (_QWORD *)sub_22077B0(16);
  if ( v10 )
  {
    *v10 = 0;
    v10[1] = "-prec-div=0";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-prec_div=0");
  v169.m128i_i64[1] = 11;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v10;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v11 = (_QWORD *)sub_22077B0(16);
  v161 = v11;
  if ( v11 )
  {
    *v11 = 0;
    v11[1] = "-fma=1";
  }
  v12 = (int *)qword_4F6D2B0;
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-fmad=1");
  v169.m128i_i64[1] = 7;
  if ( !qword_4F6D2B0 )
  {
    v17 = &dword_4F6D2A8;
    goto LABEL_306;
  }
  v13 = &dword_4F6D2A8;
  do
  {
    while ( 1 )
    {
      v14 = *((_QWORD *)v12 + 5);
      v15 = (const void *)*((_QWORD *)v12 + 4);
      if ( v14 > 7 )
        break;
      LODWORD(v16) = -7;
      if ( v14 )
      {
        LODWORD(v16) = memcmp(v15, v170, *((_QWORD *)v12 + 5));
        if ( !(_DWORD)v16 )
          LODWORD(v16) = v14 - 7;
      }
LABEL_49:
      if ( (int)v16 >= 0 )
        goto LABEL_50;
LABEL_46:
      v12 = (int *)*((_QWORD *)v12 + 3);
      if ( !v12 )
        goto LABEL_51;
    }
    LODWORD(v16) = memcmp(v15, v170, 7u);
    if ( (_DWORD)v16 )
      goto LABEL_49;
    v16 = v14 - 7;
    if ( (__int64)(v14 - 7) < 0x80000000LL )
    {
      if ( v16 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_46;
      goto LABEL_49;
    }
LABEL_50:
    v13 = v12;
    v12 = (int *)*((_QWORD *)v12 + 2);
  }
  while ( v12 );
LABEL_51:
  v17 = v13;
  if ( v13 == &dword_4F6D2A8 )
    goto LABEL_306;
  v18 = *((_QWORD *)v13 + 5);
  v19 = (const void *)*((_QWORD *)v13 + 4);
  if ( v18 <= 6 )
  {
    LODWORD(v21) = 7;
    if ( !v18
      || (v160 = *((_QWORD *)v13 + 5), v20 = memcmp(v170, v19, *((_QWORD *)v13 + 5)), LODWORD(v21) = 7 - v160, !v20) )
    {
LABEL_56:
      v20 = v21;
    }
LABEL_57:
    if ( v20 < 0 )
      goto LABEL_306;
    goto LABEL_58;
  }
  v156 = *((_QWORD *)v13 + 5);
  v20 = memcmp(v170, v19, 7u);
  if ( v20 )
    goto LABEL_57;
  v21 = 7 - v156;
  if ( 7 - v156 > 0x7FFFFFFF )
  {
LABEL_58:
    *((_QWORD *)v13 + 8) = v161;
    goto LABEL_59;
  }
  if ( v21 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_56;
LABEL_306:
  v168 = (__int64)&v169;
  v151 = sub_8FDFD0(&qword_4F6D2A0, v17, (__m128i **)&v168);
  v152 = (_QWORD *)v169.m128i_i64[0];
  *(_QWORD *)(v151 + 64) = v161;
  if ( v152 != v170 )
    j_j___libc_free_0(v152, v170[0] + 1LL);
LABEL_59:
  v22 = (_QWORD *)sub_22077B0(16);
  if ( v22 )
  {
    *v22 = 0;
    v22[1] = "-fma=0";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-fmad=0");
  v169.m128i_i64[1] = 7;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v22;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v23 = (_QWORD *)sub_22077B0(16);
  if ( v23 )
  {
    *v23 = "--device-O=0";
    v23[1] = "-opt=0";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-O0");
  v169.m128i_i64[1] = 3;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v23;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v24 = (_QWORD *)sub_22077B0(16);
  if ( v24 )
  {
    *v24 = "--device-O=1";
    v24[1] = "-opt=1";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-O1");
  v169.m128i_i64[1] = 3;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v24;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v25 = (_QWORD *)sub_22077B0(16);
  if ( v25 )
  {
    *v25 = "--device-O=2";
    v25[1] = "-opt=2";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-O2");
  v169.m128i_i64[1] = 3;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v25;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v26 = (_QWORD *)sub_22077B0(16);
  if ( v26 )
  {
    *v26 = "--device-O=3";
    v26[1] = "-opt=3";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-O3");
  v169.m128i_i64[1] = 3;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v26;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v27 = (_QWORD *)sub_22077B0(16);
  if ( v27 )
  {
    *v27 = 0;
    v27[1] = "-Osize";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-Osize");
  v169.m128i_i64[1] = 6;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v27;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v28 = (_QWORD *)sub_22077B0(16);
  if ( v28 )
  {
    *v28 = 0;
    v28[1] = "-Om";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-Om");
  v169.m128i_i64[1] = 3;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v28;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v29 = (_QWORD *)sub_22077B0(16);
  v30 = v29;
  if ( v29 )
  {
    *v29 = 0;
    v29[1] = "-Ofast-compile=max";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 19;
  v31 = sub_22409D0(&v169, &v168, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v169.m128i_i64[0] = v31;
  v170[0] = v168;
  *(_WORD *)(v31 + 16) = 24941;
  *(_BYTE *)(v31 + 18) = 120;
  *(__m128i *)v31 = si128;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v30;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v33 = (_QWORD *)sub_22077B0(16);
  if ( v33 )
  {
    *v33 = 0;
    v33[1] = "-Ofast-compile=max";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-Ofc=max");
  v169.m128i_i64[1] = 8;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v33;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v34 = (_QWORD *)sub_22077B0(16);
  v35 = v34;
  if ( v34 )
  {
    *v34 = 0;
    v34[1] = "-Ofast-compile=mid";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 19;
  v36 = sub_22409D0(&v169, &v168, 0);
  v37 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v169.m128i_i64[0] = v36;
  v170[0] = v168;
  *(_WORD *)(v36 + 16) = 26989;
  *(_BYTE *)(v36 + 18) = 100;
  *(__m128i *)v36 = v37;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v35;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v38 = (_QWORD *)sub_22077B0(16);
  if ( v38 )
  {
    *v38 = 0;
    v38[1] = "-Ofast-compile=mid";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-Ofc=mid");
  v169.m128i_i64[1] = 8;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v38;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v39 = (_QWORD *)sub_22077B0(16);
  v40 = v39;
  if ( v39 )
  {
    *v39 = 0;
    v39[1] = "-Ofast-compile=min";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 19;
  v41 = sub_22409D0(&v169, &v168, 0);
  v42 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v169.m128i_i64[0] = v41;
  v170[0] = v168;
  *(_WORD *)(v41 + 16) = 26989;
  *(_BYTE *)(v41 + 18) = 110;
  *(__m128i *)v41 = v42;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v40;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v43 = (_QWORD *)sub_22077B0(16);
  if ( v43 )
  {
    *v43 = 0;
    v43[1] = "-Ofast-compile=min";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-Ofc=min");
  v169.m128i_i64[1] = 8;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v43;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v44 = (_QWORD *)sub_22077B0(16);
  v45 = v44;
  if ( v44 )
  {
    *v44 = 0;
    v44[1] = 0;
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 17;
  v46 = sub_22409D0(&v169, &v168, 0);
  v47 = _mm_load_si128((const __m128i *)&xmmword_3E9F890);
  v169.m128i_i64[0] = v46;
  v170[0] = v168;
  *(_BYTE *)(v46 + 16) = 48;
  *(__m128i *)v46 = v47;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v45;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v48 = (_QWORD *)sub_22077B0(16);
  if ( v48 )
  {
    *v48 = 0;
    v48[1] = 0;
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-Ofc=0");
  v169.m128i_i64[1] = 6;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v48;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v49 = (_QWORD *)sub_22077B0(16);
  if ( v49 )
  {
    *v49 = "--device-debug";
    v49[1] = "-g";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-g");
  v169.m128i_i64[1] = 2;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v49;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v50 = (_QWORD *)sub_22077B0(16);
  if ( v50 )
  {
    *v50 = 0;
    v50[1] = "-show-src";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-show-src");
  v169.m128i_i64[1] = 9;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v50;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v51 = (_QWORD *)sub_22077B0(16);
  v52 = v51;
  if ( v51 )
  {
    *v51 = 0;
    v51[1] = "disable-llc-opts";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 17;
  v53 = sub_22409D0(&v169, &v168, 0);
  v54 = _mm_load_si128((const __m128i *)&xmmword_3E9F8A0);
  v169.m128i_i64[0] = v53;
  v170[0] = v168;
  *(_BYTE *)(v53 + 16) = 115;
  *(__m128i *)v53 = v54;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v52;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v55 = (_QWORD *)sub_22077B0(16);
  if ( v55 )
  {
    *v55 = "-w";
    v55[1] = "-w";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-w");
  v169.m128i_i64[1] = 2;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v55;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v56 = (_QWORD *)sub_22077B0(16);
  v57 = v56;
  if ( v56 )
  {
    *v56 = 0;
    v56[1] = "-Wno-memory-space";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 17;
  v58 = sub_22409D0(&v169, &v168, 0);
  v59 = _mm_load_si128((const __m128i *)&xmmword_3E9F8B0);
  v169.m128i_i64[0] = v58;
  v170[0] = v168;
  *(_BYTE *)(v58 + 16) = 101;
  *(__m128i *)v58 = v59;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v57;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v60 = (_QWORD *)sub_22077B0(16);
  v61 = v60;
  if ( v60 )
  {
    *v60 = 0;
    v60[1] = "-disable-inlining";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 17;
  v62 = sub_22409D0(&v169, &v168, 0);
  v63 = _mm_load_si128((const __m128i *)&xmmword_3E9F8C0);
  v169.m128i_i64[0] = v62;
  v170[0] = v168;
  *(_BYTE *)(v62 + 16) = 103;
  *(__m128i *)v62 = v63;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v61;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v64 = (_QWORD *)sub_22077B0(16);
  v65 = v64;
  if ( v64 )
  {
    *v64 = 0;
    v64[1] = "-aggressive-inline";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 18;
  v66 = sub_22409D0(&v169, &v168, 0);
  v67 = _mm_load_si128((const __m128i *)&xmmword_3E9F8D0);
  v169.m128i_i64[0] = v66;
  v170[0] = v168;
  *(_WORD *)(v66 + 16) = 25966;
  *(__m128i *)v66 = v67;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v65;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v68 = (_QWORD *)sub_22077B0(16);
  v69 = v68;
  if ( v68 )
  {
    *v68 = "--kernel-params-are-restrict";
    v68[1] = "-restrict";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 27;
  v70 = (__m128i *)sub_22409D0(&v169, &v168, 0);
  v71 = _mm_load_si128((const __m128i *)&xmmword_3E9F8E0);
  v169.m128i_i64[0] = (__int64)v70;
  v170[0] = v168;
  qmemcpy(&v70[1], "re-restrict", 11);
  *v70 = v71;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v69;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v72 = (_QWORD *)sub_22077B0(16);
  v157 = v72;
  if ( v72 )
  {
    *v72 = 0;
    v72[1] = "-allow-restrict-in-struct";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 24;
  v73 = sub_22409D0(&v169, &v168, 0);
  v74 = _mm_load_si128((const __m128i *)&xmmword_3E9F8F0);
  v169.m128i_i64[0] = v73;
  v170[0] = v168;
  *(_QWORD *)(v73 + 16) = 0x7463757274732D6ELL;
  *(__m128i *)v73 = v74;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  if ( !qword_4F6D2B0 )
  {
    v162 = &dword_4F6D2A8;
    goto LABEL_174;
  }
  v75 = qword_4F6D2B0;
  v76 = v169.m128i_u64[1];
  v77 = (_QWORD *)v169.m128i_i64[0];
  v78 = &dword_4F6D2A8;
  while ( 2 )
  {
    while ( 2 )
    {
      v79 = *(_QWORD *)(v75 + 40);
      v80 = v76;
      if ( v79 <= v76 )
        v80 = *(_QWORD *)(v75 + 40);
      if ( !v80 || (v81 = memcmp(*(const void **)(v75 + 32), v77, v80)) == 0 )
      {
        if ( (__int64)(v79 - v76) >= 0x80000000LL )
          goto LABEL_164;
        if ( (__int64)(v79 - v76) > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v81 = v79 - v76;
          break;
        }
LABEL_155:
        v75 = *(_QWORD *)(v75 + 24);
        if ( !v75 )
          goto LABEL_165;
        continue;
      }
      break;
    }
    if ( v81 < 0 )
      goto LABEL_155;
LABEL_164:
    v78 = (int *)v75;
    v75 = *(_QWORD *)(v75 + 16);
    if ( v75 )
      continue;
    break;
  }
LABEL_165:
  v162 = v78;
  if ( v78 == &dword_4F6D2A8 )
    goto LABEL_174;
  v82 = *((_QWORD *)v78 + 5);
  v83 = v76;
  if ( v82 <= v76 )
    v83 = *((_QWORD *)v78 + 5);
  if ( v83 && (v153 = *((_QWORD *)v78 + 5), v84 = memcmp(v77, *((const void **)v78 + 4), v83), v82 = v153, v84) )
  {
LABEL_173:
    if ( v84 < 0 )
      goto LABEL_174;
  }
  else if ( (__int64)(v76 - v82) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v76 - v82) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v84 = v76 - v82;
      goto LABEL_173;
    }
LABEL_174:
    v168 = (__int64)&v169;
    v85 = sub_8FDFD0(&qword_4F6D2A0, v162, (__m128i **)&v168);
    v77 = (_QWORD *)v169.m128i_i64[0];
    v162 = (int *)v85;
  }
  *((_QWORD *)v162 + 8) = v157;
  if ( v77 != v170 )
    j_j___libc_free_0(v77, v170[0] + 1LL);
  v86 = (_QWORD *)sub_22077B0(16);
  if ( v86 )
  {
    *v86 = "--device-c";
    v86[1] = "--device-c";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "--device-c");
  v169.m128i_i64[1] = 10;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v86;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v87 = (_QWORD *)sub_22077B0(16);
  v88 = v87;
  if ( v87 )
  {
    *v87 = "--generate-line-info";
    v87[1] = "-generate-line-info";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 19;
  v89 = sub_22409D0(&v169, &v168, 0);
  v90 = _mm_load_si128((const __m128i *)&xmmword_3E9F900);
  v169.m128i_i64[0] = v89;
  v170[0] = v168;
  *(_WORD *)(v89 + 16) = 26222;
  *(_BYTE *)(v89 + 18) = 111;
  *(__m128i *)v89 = v90;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v88;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v91 = (_QWORD *)sub_22077B0(16);
  v158 = v91;
  if ( v91 )
  {
    *v91 = "--enable-opt-byval";
    v91[1] = "-enable-opt-byval";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 17;
  v92 = sub_22409D0(&v169, &v168, 0);
  v93 = _mm_load_si128((const __m128i *)&xmmword_3E9F910);
  v169.m128i_i64[0] = v92;
  v170[0] = v168;
  *(_BYTE *)(v92 + 16) = 108;
  *(__m128i *)v92 = v93;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  if ( !qword_4F6D2B0 )
  {
    v163 = &dword_4F6D2A8;
    goto LABEL_208;
  }
  v94 = qword_4F6D2B0;
  v95 = v169.m128i_u64[1];
  v96 = (_QWORD *)v169.m128i_i64[0];
  v97 = &dword_4F6D2A8;
  while ( 2 )
  {
    while ( 2 )
    {
      v98 = *(_QWORD *)(v94 + 40);
      v99 = v95;
      if ( v98 <= v95 )
        v99 = *(_QWORD *)(v94 + 40);
      if ( !v99 || (v100 = memcmp(*(const void **)(v94 + 32), v96, v99)) == 0 )
      {
        if ( (__int64)(v98 - v95) >= 0x80000000LL )
          goto LABEL_198;
        if ( (__int64)(v98 - v95) > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v100 = v98 - v95;
          break;
        }
LABEL_189:
        v94 = *(_QWORD *)(v94 + 24);
        if ( !v94 )
          goto LABEL_199;
        continue;
      }
      break;
    }
    if ( v100 < 0 )
      goto LABEL_189;
LABEL_198:
    v97 = (int *)v94;
    v94 = *(_QWORD *)(v94 + 16);
    if ( v94 )
      continue;
    break;
  }
LABEL_199:
  v163 = v97;
  if ( v97 == &dword_4F6D2A8 )
    goto LABEL_208;
  v101 = *((_QWORD *)v97 + 5);
  v102 = v95;
  if ( v101 <= v95 )
    v102 = *((_QWORD *)v97 + 5);
  if ( v102 && (v154 = *((_QWORD *)v97 + 5), v103 = memcmp(v96, *((const void **)v97 + 4), v102), v101 = v154, v103) )
  {
LABEL_207:
    if ( v103 < 0 )
      goto LABEL_208;
  }
  else if ( (__int64)(v95 - v101) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v95 - v101) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v103 = v95 - v101;
      goto LABEL_207;
    }
LABEL_208:
    v168 = (__int64)&v169;
    v104 = sub_8FDFD0(&qword_4F6D2A0, v163, (__m128i **)&v168);
    v96 = (_QWORD *)v169.m128i_i64[0];
    v163 = (int *)v104;
  }
  *((_QWORD *)v163 + 8) = v158;
  if ( v96 != v170 )
    j_j___libc_free_0(v96, v170[0] + 1LL);
  v105 = (_QWORD *)sub_22077B0(16);
  v159 = v105;
  if ( v105 )
  {
    *v105 = 0;
    v105[1] = "-no-lineinfo-inlined-at";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 23;
  v106 = sub_22409D0(&v169, &v168, 0);
  v107 = _mm_load_si128((const __m128i *)&xmmword_3E9F920);
  v169.m128i_i64[0] = v106;
  v170[0] = v168;
  *(_DWORD *)(v106 + 16) = 1684369001;
  *(_WORD *)(v106 + 20) = 24877;
  *(_BYTE *)(v106 + 22) = 116;
  *(__m128i *)v106 = v107;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  if ( !qword_4F6D2B0 )
  {
    v164 = &dword_4F6D2A8;
    goto LABEL_234;
  }
  v108 = (_QWORD *)v169.m128i_i64[0];
  v109 = qword_4F6D2B0;
  v110 = v169.m128i_u64[1];
  v111 = &dword_4F6D2A8;
  while ( 2 )
  {
    while ( 2 )
    {
      v112 = *(_QWORD *)(v109 + 40);
      v113 = v110;
      if ( v112 <= v110 )
        v113 = *(_QWORD *)(v109 + 40);
      if ( !v113 || (v114 = memcmp(*(const void **)(v109 + 32), v108, v113)) == 0 )
      {
        if ( (__int64)(v112 - v110) >= 0x80000000LL )
          goto LABEL_224;
        if ( (__int64)(v112 - v110) > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v114 = v112 - v110;
          break;
        }
LABEL_215:
        v109 = *(_QWORD *)(v109 + 24);
        if ( !v109 )
          goto LABEL_225;
        continue;
      }
      break;
    }
    if ( v114 < 0 )
      goto LABEL_215;
LABEL_224:
    v111 = (int *)v109;
    v109 = *(_QWORD *)(v109 + 16);
    if ( v109 )
      continue;
    break;
  }
LABEL_225:
  v164 = v111;
  if ( v111 == &dword_4F6D2A8 )
    goto LABEL_234;
  v115 = *((_QWORD *)v111 + 5);
  v116 = v110;
  if ( v115 <= v110 )
    v116 = *((_QWORD *)v111 + 5);
  if ( v116 && (v155 = *((_QWORD *)v111 + 5), v117 = memcmp(v108, *((const void **)v111 + 4), v116), v115 = v155, v117) )
  {
LABEL_233:
    if ( v117 < 0 )
      goto LABEL_234;
  }
  else if ( (__int64)(v110 - v115) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v110 - v115) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v117 = v110 - v115;
      goto LABEL_233;
    }
LABEL_234:
    v168 = (__int64)&v169;
    v118 = sub_8FDFD0(&qword_4F6D2A0, v164, (__m128i **)&v168);
    v108 = (_QWORD *)v169.m128i_i64[0];
    v164 = (int *)v118;
  }
  *((_QWORD *)v164 + 8) = v159;
  if ( v108 != v170 )
    j_j___libc_free_0(v108, v170[0] + 1LL);
  v119 = (_QWORD *)sub_22077B0(16);
  if ( v119 )
  {
    v119[1] = 0;
    *v119 = "--keep-device-functions";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 23;
  v120 = sub_22409D0(&v169, &v168, 0);
  v121 = _mm_load_si128((const __m128i *)&xmmword_3E9F930);
  v169.m128i_i64[0] = v120;
  v170[0] = v168;
  *(_DWORD *)(v120 + 16) = 1769235310;
  *(_WORD *)(v120 + 20) = 28271;
  *(_BYTE *)(v120 + 22) = 115;
  *(__m128i *)v120 = v121;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v119;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v122 = (_QWORD *)sub_22077B0(16);
  if ( v122 )
  {
    *v122 = "--emit-lifetime-intrinsics";
    v122[1] = "--emit-optix-ir";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "--emit-optix-ir");
  v169.m128i_i64[1] = 15;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v122;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v123 = (_QWORD *)sub_22077B0(16);
  if ( v123 )
  {
    *v123 = 0;
    v123[1] = "-opt-fdiv=0";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-opt-fdiv=0");
  v169.m128i_i64[1] = 11;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v123;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v124 = (_QWORD *)sub_22077B0(16);
  if ( v124 )
  {
    *v124 = 0;
    v124[1] = "-opt-fdiv=1";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-opt-fdiv=1");
  v169.m128i_i64[1] = 11;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v124;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v125 = (_QWORD *)sub_22077B0(16);
  if ( v125 )
  {
    *v125 = 0;
    v125[1] = "-new-nvvm-remat";
  }
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-new-nvvm-remat");
  v169.m128i_i64[1] = 15;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v125;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v126 = (_QWORD *)sub_22077B0(16);
  v127 = v126;
  if ( v126 )
  {
    *v126 = 0;
    v126[1] = "-disable-new-nvvm-remat";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 23;
  v128 = sub_22409D0(&v169, &v168, 0);
  v129 = _mm_load_si128((const __m128i *)&xmmword_3E9F940);
  v169.m128i_i64[0] = v128;
  v170[0] = v168;
  *(_DWORD *)(v128 + 16) = 1701981549;
  *(_WORD *)(v128 + 20) = 24941;
  *(_BYTE *)(v128 + 22) = 116;
  *(__m128i *)v128 = v129;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v127;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v130 = (_QWORD *)sub_22077B0(16);
  v131 = v130;
  if ( v130 )
  {
    *v130 = 0;
    v130[1] = "-disable-nvvm-remat";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 19;
  v132 = sub_22409D0(&v169, &v168, 0);
  v133 = _mm_load_si128((const __m128i *)&xmmword_3E9F950);
  v169.m128i_i64[0] = v132;
  v170[0] = v168;
  *(_WORD *)(v132 + 16) = 24941;
  *(_BYTE *)(v132 + 18) = 116;
  *(__m128i *)v132 = v133;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v131;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v134 = (_QWORD *)sub_22077B0(16);
  v135 = v134;
  if ( v134 )
  {
    *v134 = "--discard_value_names=1";
    v134[1] = "-discard-value-names=1";
  }
  v169.m128i_i64[0] = (__int64)v170;
  v168 = 20;
  v136 = sub_22409D0(&v169, &v168, 0);
  v137 = _mm_load_si128((const __m128i *)&xmmword_3E9F960);
  v169.m128i_i64[0] = v136;
  v170[0] = v168;
  *(_DWORD *)(v136 + 16) = 1936026977;
  *(__m128i *)v136 = v137;
  v169.m128i_i64[1] = v168;
  *(_BYTE *)(v169.m128i_i64[0] + v168) = 0;
  *(_QWORD *)sub_8FE150(&qword_4F6D2A0, &v169) = v135;
  if ( (_QWORD *)v169.m128i_i64[0] != v170 )
    j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
  v138 = (_QWORD *)sub_22077B0(16);
  v165 = v138;
  if ( v138 )
  {
    *v138 = 0;
    v138[1] = "-gen-opt-lto";
  }
  v139 = (int *)qword_4F6D2B0;
  v169.m128i_i64[0] = (__int64)v170;
  strcpy((char *)v170, "-gen-opt-lto");
  v169.m128i_i64[1] = 12;
  if ( !qword_4F6D2B0 )
  {
    v144 = &dword_4F6D2A8;
    goto LABEL_303;
  }
  v140 = &dword_4F6D2A8;
  while ( 2 )
  {
    while ( 2 )
    {
      v141 = *((_QWORD *)v139 + 5);
      v142 = (const void *)*((_QWORD *)v139 + 4);
      if ( v141 <= 0xC )
      {
        LODWORD(v143) = -12;
        if ( v141 )
        {
          LODWORD(v143) = memcmp(v142, v170, *((_QWORD *)v139 + 5));
          if ( !(_DWORD)v143 )
            LODWORD(v143) = v141 - 12;
        }
        goto LABEL_276;
      }
      LODWORD(v143) = memcmp(v142, v170, 0xCu);
      if ( (_DWORD)v143 )
      {
LABEL_276:
        if ( (int)v143 >= 0 )
          break;
        goto LABEL_273;
      }
      v143 = v141 - 12;
      if ( (__int64)(v141 - 12) < 0x80000000LL )
      {
        if ( v143 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_276;
LABEL_273:
        v139 = (int *)*((_QWORD *)v139 + 3);
        if ( !v139 )
          goto LABEL_278;
        continue;
      }
      break;
    }
    v140 = v139;
    v139 = (int *)*((_QWORD *)v139 + 2);
    if ( v139 )
      continue;
    break;
  }
LABEL_278:
  v144 = v140;
  if ( v140 == &dword_4F6D2A8 )
    goto LABEL_303;
  v145 = *((_QWORD *)v140 + 5);
  v146 = (const void *)*((_QWORD *)v140 + 4);
  if ( v145 <= 0xB )
  {
    LODWORD(v148) = 12;
    if ( !v145
      || (v167 = *((_QWORD *)v140 + 5),
          v147 = memcmp(v170, v146, *((_QWORD *)v140 + 5)),
          LODWORD(v148) = 12 - v167,
          !v147) )
    {
LABEL_283:
      v147 = v148;
    }
LABEL_284:
    if ( v147 < 0 )
      goto LABEL_303;
    goto LABEL_285;
  }
  v166 = *((_QWORD *)v140 + 5);
  v147 = memcmp(v170, v146, 0xCu);
  if ( v147 )
    goto LABEL_284;
  v148 = 12 - v166;
  if ( 12 - v166 > 0x7FFFFFFF )
  {
LABEL_285:
    *((_QWORD *)v140 + 8) = v165;
    return;
  }
  if ( v148 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_283;
LABEL_303:
  v168 = (__int64)&v169;
  v149 = sub_8FDFD0(&qword_4F6D2A0, v144, (__m128i **)&v168);
  v150 = (_QWORD *)v169.m128i_i64[0];
  *(_QWORD *)(v149 + 64) = v165;
  if ( v150 != v170 )
    j_j___libc_free_0(v150, v170[0] + 1LL);
}
