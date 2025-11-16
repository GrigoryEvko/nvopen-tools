// Function: sub_E95100
// Address: 0xe95100
//
__int64 *__fastcall sub_E95100(__int64 *a1, char *a2, char *a3, void **a4, void **a5, int *a6, _BYTE *a7, _DWORD *a8)
{
  size_t *p_s1; // rsi
  size_t v11; // r14
  void *v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  char *i; // r15
  __int64 v16; // r9
  char *v17; // rbx
  bool v18; // zf
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // r12
  unsigned int v22; // ebx
  __m128i *v23; // rax
  __m128i v24; // xmm0
  __m128i v25; // xmm0
  unsigned int v26; // eax
  __int64 v27; // rdx
  __m128i *v28; // rax
  __m128i v29; // xmm0
  __m128i v30; // xmm0
  __m128i v31; // xmm0
  unsigned int v33; // eax
  __int64 v34; // rdx
  __m128i *v35; // rax
  __m128i si128; // xmm0
  _QWORD *v37; // r13
  size_t v38; // r15
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rax
  size_t v42; // rdx
  size_t v43; // rax
  bool v44; // cc
  size_t v45; // rax
  _QWORD *v46; // r12
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rcx
  __int64 v49; // rdx
  size_t v50; // r12
  unsigned __int64 v51; // rax
  size_t v52; // rdx
  size_t v53; // rax
  size_t v54; // rax
  _QWORD *v55; // r12
  size_t v56; // r15
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rcx
  unsigned __int64 v59; // rax
  size_t v60; // rdx
  size_t v61; // rax
  _QWORD *v62; // r15
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rcx
  __int64 v65; // rdx
  size_t v66; // r15
  size_t v67; // rax
  size_t v68; // rax
  _QWORD *v69; // r15
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rdx
  size_t v73; // r15
  unsigned __int64 v74; // rax
  size_t v75; // rdx
  size_t v76; // rax
  size_t v77; // rax
  __int64 *v78; // rbx
  char *j; // r12
  size_t v80; // r15
  char *v81; // r14
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rcx
  __int64 v84; // rdx
  size_t v85; // rdx
  size_t v86; // rdx
  size_t v87; // r15
  unsigned __int64 v88; // rax
  unsigned __int64 v89; // rcx
  __int64 v90; // rdx
  size_t v91; // rdx
  size_t v92; // rdx
  size_t v93; // r15
  unsigned __int64 v94; // rax
  unsigned __int64 v95; // rcx
  __int64 v96; // rdx
  size_t v97; // rdx
  size_t v98; // rdx
  size_t v99; // r14
  unsigned __int64 v100; // rax
  unsigned __int64 v101; // rcx
  size_t v102; // rdx
  size_t v103; // rdx
  int v104; // eax
  unsigned __int64 v105; // rax
  unsigned __int64 v106; // rdx
  __int64 v107; // rcx
  size_t v108; // r8
  unsigned __int64 v109; // rax
  size_t v110; // rdx
  size_t v111; // rax
  size_t v112; // rax
  size_t v113; // r14
  char *v114; // r15
  unsigned __int64 v115; // rax
  unsigned __int64 v116; // rcx
  __int64 v117; // rdx
  unsigned __int64 v118; // rax
  size_t v119; // rdx
  size_t v120; // rax
  size_t v121; // rax
  size_t v122; // r14
  unsigned __int64 v123; // rax
  unsigned __int64 v124; // rcx
  unsigned __int64 v125; // rax
  size_t v126; // rdx
  size_t v127; // rax
  size_t v128; // rax
  unsigned int v129; // eax
  __int64 v130; // rdx
  __int64 v131; // r12
  unsigned int v132; // ebx
  __m128i *v133; // rax
  __m128i v134; // xmm0
  void *v135; // rdi
  unsigned int v136; // eax
  __int64 v137; // rdx
  __m128i *v138; // rax
  __m128i v139; // xmm0
  __m128i v140; // xmm0
  unsigned int v141; // eax
  __int64 v142; // rdx
  __int64 v143; // r12
  unsigned int v144; // ebx
  __m128i *v145; // rax
  __m128i v146; // xmm0
  __m128i v147; // xmm0
  __m128i v148; // xmm0
  unsigned int v149; // eax
  __int64 v150; // rdx
  __m128i *v151; // rax
  __m128i v152; // xmm0
  __m128i v153; // xmm0
  unsigned int v154; // eax
  __int64 v155; // rdx
  __m128i *v156; // rax
  __m128i v157; // xmm0
  size_t v158; // [rsp+0h] [rbp-140h]
  __int64 *v159; // [rsp+10h] [rbp-130h]
  void *v160; // [rsp+18h] [rbp-128h]
  __int64 v161; // [rsp+20h] [rbp-120h]
  char *v164[3]; // [rsp+40h] [rbp-100h] BYREF
  size_t v165; // [rsp+58h] [rbp-E8h] BYREF
  char *v166; // [rsp+60h] [rbp-E0h] BYREF
  size_t v167; // [rsp+68h] [rbp-D8h]
  size_t v168; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v169; // [rsp+78h] [rbp-C8h]
  _BYTE v170[16]; // [rsp+80h] [rbp-C0h] BYREF
  void *s1; // [rsp+90h] [rbp-B0h] BYREF
  size_t n; // [rsp+98h] [rbp-A8h]
  _QWORD v173[2]; // [rsp+A0h] [rbp-A0h] BYREF
  _QWORD *v174; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v175; // [rsp+B8h] [rbp-88h]
  _BYTE v176[128]; // [rsp+C0h] [rbp-80h] BYREF

  v164[0] = a2;
  p_s1 = (size_t *)&v174;
  *a7 = 0;
  v164[1] = a3;
  v174 = v176;
  v175 = 0x500000000LL;
  sub_C93960(v164, (__int64)&v174, 44, -1, 1, (__int64)a6);
  if ( !(_DWORD)v175 )
  {
    *a4 = 0;
    v11 = 0;
    a4[1] = 0;
    v12 = 0;
    *a5 = 0;
    a5[1] = 0;
LABEL_3:
    v166 = 0;
    v167 = 0;
LABEL_4:
    v161 = 0;
    v160 = 0;
    goto LABEL_5;
  }
  v37 = v174;
  v38 = 0;
  v39 = sub_C935B0(v174, byte_3F15413, 6, 0);
  v40 = v37[1];
  if ( v39 < v40 )
  {
    v38 = v40 - v39;
    v40 = v39;
  }
  p_s1 = (size_t *)byte_3F15413;
  s1 = (void *)(*v37 + v40);
  n = v38;
  v41 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v42 = n;
  v43 = v41 + 1;
  v44 = v43 <= n;
  *a4 = s1;
  if ( !v44 )
    v43 = v42;
  v45 = v42 - v38 + v43;
  if ( v45 > v42 )
    v45 = v42;
  v44 = (unsigned int)v175 <= 1;
  a4[1] = (void *)v45;
  if ( v44 )
  {
    *a5 = 0;
    a5[1] = 0;
LABEL_153:
    v11 = 0;
    v12 = 0;
    goto LABEL_3;
  }
  v46 = v174;
  v47 = sub_C935B0(v174 + 2, byte_3F15413, 6, 0);
  v48 = v46[3];
  v49 = v46[2];
  v50 = 0;
  if ( v47 < v48 )
  {
    v50 = v48 - v47;
    v48 = v47;
  }
  p_s1 = (size_t *)byte_3F15413;
  s1 = (void *)(v49 + v48);
  n = v50;
  v51 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v52 = n;
  v53 = v51 + 1;
  v44 = v53 <= n;
  *a5 = s1;
  if ( !v44 )
    v53 = v52;
  v54 = v52 - v50 + v53;
  if ( v54 > v52 )
    v54 = v52;
  v44 = (unsigned int)v175 <= 2;
  a5[1] = (void *)v54;
  if ( v44 )
    goto LABEL_153;
  v55 = v174;
  v56 = 0;
  v57 = sub_C935B0(v174 + 4, byte_3F15413, 6, 0);
  v58 = v55[5];
  if ( v57 < v58 )
  {
    v56 = v58 - v57;
    v58 = v57;
  }
  p_s1 = (size_t *)byte_3F15413;
  s1 = (void *)(v55[4] + v58);
  n = v56;
  v59 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v60 = n;
  v12 = s1;
  v61 = v59 + 1;
  if ( v61 > n )
    v61 = n;
  if ( n - v56 + v61 <= n )
    v60 = n - v56 + v61;
  v11 = v60;
  if ( (unsigned int)v175 <= 3 )
    goto LABEL_3;
  v62 = v174;
  v63 = sub_C935B0(v174 + 6, byte_3F15413, 6, 0);
  v64 = v62[7];
  v65 = v62[6];
  v66 = 0;
  if ( v63 < v64 )
  {
    v66 = v64 - v63;
    v64 = v63;
  }
  p_s1 = (size_t *)byte_3F15413;
  s1 = (void *)(v65 + v64);
  n = v66;
  v67 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
  v166 = (char *)s1;
  if ( v67 > n )
    v67 = n;
  v68 = n - v66 + v67;
  if ( v68 > n )
    v68 = n;
  v167 = v68;
  if ( (unsigned int)v175 <= 4 )
    goto LABEL_4;
  v69 = v174;
  v70 = sub_C935B0(v174 + 8, byte_3F15413, 6, 0);
  v71 = v69[9];
  v72 = v69[8];
  v73 = 0;
  if ( v70 < v71 )
  {
    v73 = v71 - v70;
    v71 = v70;
  }
  s1 = (void *)(v72 + v71);
  n = v73;
  v74 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
  v75 = n;
  p_s1 = (size_t *)s1;
  v76 = v74 + 1;
  v160 = s1;
  if ( v76 > n )
    v76 = n;
  v77 = n - v73 + v76;
  if ( v77 <= n )
    v75 = v77;
  v161 = v75;
LABEL_5:
  v13 = (unsigned __int64)a5[1];
  if ( !v13 )
  {
    v33 = sub_C63BB0();
    v168 = 76;
    v21 = v34;
    s1 = v173;
    v22 = v33;
    v35 = (__m128i *)sub_22409D0(&s1, &v168, 0);
    s1 = v35;
    v173[0] = v168;
    *v35 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F81B70);
    qmemcpy(&v35[4], "d by a comma", 12);
    v35[1] = si128;
    v35[2] = _mm_load_si128((const __m128i *)&xmmword_3F81B80);
    v35[3] = _mm_load_si128((const __m128i *)&xmmword_3F81B90);
LABEL_30:
    p_s1 = (size_t *)&s1;
    n = v168;
    *((_BYTE *)s1 + v168) = 0;
    sub_C63F00(a1, (__int64)&s1, v22, v21);
    if ( s1 != v173 )
    {
      p_s1 = (size_t *)(v173[0] + 1LL);
      j_j___libc_free_0(s1, v173[0] + 1LL);
    }
    goto LABEL_32;
  }
  if ( v13 > 0x10 )
  {
    v26 = sub_C63BB0();
    v168 = 87;
    v21 = v27;
    s1 = v173;
    v22 = v26;
    v28 = (__m128i *)sub_22409D0(&s1, &v168, 0);
    s1 = v28;
    v173[0] = v168;
    *v28 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    v29 = _mm_load_si128((const __m128i *)&xmmword_3F81B70);
    v28[5].m128i_i32[0] = 1952670066;
    v28[1] = v29;
    v30 = _mm_load_si128((const __m128i *)&xmmword_3F81BA0);
    v28[5].m128i_i16[2] = 29285;
    v28[2] = v30;
    v31 = _mm_load_si128((const __m128i *)&xmmword_3F81BB0);
    v28[5].m128i_i8[6] = 115;
    v28[3] = v31;
    v28[4] = _mm_load_si128((const __m128i *)&xmmword_3F81BC0);
    goto LABEL_30;
  }
  *a6 = 0;
  *a8 = 0;
  if ( !v11 )
  {
LABEL_71:
    *a1 = 1;
    goto LABEL_32;
  }
  p_s1 = (size_t *)"regular";
  v14 = 7;
  for ( i = (char *)&off_497A9A0; ; i += 128 )
  {
    if ( v11 == v14 && !memcmp(v12, p_s1, v11) )
      goto LABEL_25;
    if ( v11 == *((_QWORD *)i + 5) )
    {
      p_s1 = (size_t *)*((_QWORD *)i + 4);
      if ( !memcmp(v12, p_s1, v11) )
      {
        i += 32;
        goto LABEL_25;
      }
    }
    if ( v11 == *((_QWORD *)i + 9) )
    {
      p_s1 = (size_t *)*((_QWORD *)i + 8);
      if ( !memcmp(v12, p_s1, v11) )
      {
        i += 64;
        goto LABEL_25;
      }
    }
    if ( v11 == *((_QWORD *)i + 13) )
    {
      p_s1 = (size_t *)*((_QWORD *)i + 12);
      if ( !memcmp(v12, p_s1, v11) )
        break;
    }
    if ( i + 128 == (char *)&unk_497AC20 )
    {
      if ( v11 == *((_QWORD *)i + 17) )
      {
        p_s1 = (size_t *)*((_QWORD *)i + 16);
        if ( !memcmp(v12, p_s1, v11) )
        {
          i = (char *)&unk_497AC20;
          goto LABEL_26;
        }
      }
      v17 = i + 160;
      if ( v11 == *((_QWORD *)i + 21) && (p_s1 = (size_t *)*((_QWORD *)i + 20), !memcmp(v12, p_s1, v11))
        || (v17 = i + 192, v11 == *((_QWORD *)i + 25))
        && (p_s1 = (size_t *)*((_QWORD *)i + 24), !memcmp(v12, p_s1, v11)) )
      {
        i = v17;
        goto LABEL_25;
      }
LABEL_154:
      v136 = sub_C63BB0();
      v168 = 53;
      v21 = v137;
      s1 = v173;
      v22 = v136;
      v138 = (__m128i *)sub_22409D0(&s1, &v168, 0);
      s1 = v138;
      v173[0] = v168;
      *v138 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
      v139 = _mm_load_si128((const __m128i *)&xmmword_3F81BD0);
      v138[3].m128i_i32[0] = 1887007776;
      v138[1] = v139;
      v140 = _mm_load_si128((const __m128i *)&xmmword_3F81BE0);
      v138[3].m128i_i8[4] = 101;
      v138[2] = v140;
      goto LABEL_30;
    }
    v14 = *((_QWORD *)i + 17);
    p_s1 = (size_t *)*((_QWORD *)i + 16);
  }
  i += 96;
LABEL_25:
  if ( i == (char *)&unk_497AC80 )
    goto LABEL_154;
LABEL_26:
  v18 = v167 == 0;
  *a6 = (i - (char *)&off_497A9A0) >> 5;
  *a7 = 1;
  if ( v18 )
  {
    if ( *a6 != 8 )
      goto LABEL_71;
    v19 = sub_C63BB0();
    v168 = 73;
    v21 = v20;
    s1 = v173;
    v22 = v19;
    v23 = (__m128i *)sub_22409D0(&s1, &v168, 0);
    s1 = v23;
    v173[0] = v168;
    *v23 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    v24 = _mm_load_si128((const __m128i *)&xmmword_3F81BF0);
    v23[4].m128i_i64[0] = 0x6569666963657073LL;
    v23[1] = v24;
    v25 = _mm_load_si128((const __m128i *)&xmmword_3F81C00);
    v23[4].m128i_i8[8] = 114;
    v23[2] = v25;
    v23[3] = _mm_load_si128((const __m128i *)&xmmword_3F81C10);
    goto LABEL_30;
  }
  p_s1 = &v168;
  v168 = (size_t)v170;
  v169 = 0x100000000LL;
  sub_C93960(&v166, (__int64)&v168, 43, -1, 0, v16);
  v78 = (__int64 *)v168;
  v159 = (__int64 *)(v168 + 16LL * (unsigned int)v169);
  if ( v159 == (__int64 *)v168 )
  {
    v104 = *a6;
  }
  else
  {
    while ( 2 )
    {
      for ( j = (char *)&unk_497A7E0; ; j += 160 )
      {
        v99 = 0;
        v100 = sub_C935B0(v78, byte_3F15413, 6, 0);
        v101 = v78[1];
        if ( v100 < v101 )
        {
          v99 = v101 - v100;
          v101 = v100;
        }
        s1 = (void *)(*v78 + v101);
        n = v99;
        v102 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
        if ( v102 > n )
          v102 = n;
        v103 = n - v99 + v102;
        if ( v103 > n )
          v103 = n;
        if ( *((_QWORD *)j + 2) == v103 && (!v103 || !memcmp(s1, *((const void **)j + 1), v103)) )
          goto LABEL_112;
        v80 = 0;
        v81 = j + 40;
        v82 = sub_C935B0(v78, byte_3F15413, 6, 0);
        v83 = v78[1];
        v84 = *v78;
        if ( v82 < v83 )
        {
          v80 = v83 - v82;
          v83 = v82;
        }
        n = v80;
        s1 = (void *)(v83 + v84);
        v85 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
        if ( v85 > n )
          v85 = n;
        v86 = n - v80 + v85;
        if ( v86 > n )
          v86 = n;
        if ( v86 == *((_QWORD *)j + 7) && (!v86 || !memcmp(s1, *((const void **)j + 6), v86)) )
          goto LABEL_117;
        v87 = 0;
        v81 = j + 80;
        v88 = sub_C935B0(v78, byte_3F15413, 6, 0);
        v89 = v78[1];
        v90 = *v78;
        if ( v88 < v89 )
        {
          v87 = v89 - v88;
          v89 = v88;
        }
        n = v87;
        s1 = (void *)(v89 + v90);
        v91 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
        if ( v91 > n )
          v91 = n;
        v92 = n - v87 + v91;
        if ( v92 > n )
          v92 = n;
        if ( *((_QWORD *)j + 12) == v92 && (!v92 || !memcmp(s1, *((const void **)j + 11), v92)) )
          goto LABEL_117;
        v93 = 0;
        v81 = j + 120;
        v94 = sub_C935B0(v78, byte_3F15413, 6, 0);
        v95 = v78[1];
        v96 = *v78;
        if ( v94 < v95 )
        {
          v93 = v95 - v94;
          v95 = v94;
        }
        n = v93;
        s1 = (void *)(v95 + v96);
        v97 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
        if ( v97 > n )
          v97 = n;
        v98 = n - v93 + v97;
        if ( v98 > n )
          v98 = n;
        if ( v98 == *((_QWORD *)j + 17) && (!v98 || !memcmp(s1, *((const void **)j + 16), v98)) )
        {
LABEL_117:
          j = v81;
          goto LABEL_112;
        }
        if ( j + 160 == (char *)&unk_497A920 )
          break;
      }
      v105 = sub_C935B0(v78, byte_3F15413, 6, 0);
      v106 = v78[1];
      v107 = *v78;
      v108 = 0;
      if ( v105 < v106 )
      {
        v108 = v106 - v105;
        v106 = v105;
      }
      n = v108;
      s1 = (void *)(v107 + v106);
      v158 = v108;
      v109 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v110 = n;
      v111 = v109 + 1;
      if ( v111 > n )
        v111 = n;
      v112 = n - v158 + v111;
      if ( v112 <= n )
        v110 = v112;
      if ( *((_QWORD *)j + 22) == v110 && (!v110 || !memcmp(s1, *((const void **)j + 21), v110)) )
      {
        j += 160;
        goto LABEL_113;
      }
      v113 = 0;
      v114 = j + 200;
      v115 = sub_C935B0(v78, byte_3F15413, 6, 0);
      v116 = v78[1];
      v117 = *v78;
      if ( v115 < v116 )
      {
        v113 = v116 - v115;
        v116 = v115;
      }
      n = v113;
      s1 = (void *)(v116 + v117);
      v118 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v119 = n;
      v120 = v118 + 1;
      if ( v120 > n )
        v120 = n;
      v121 = n - v113 + v120;
      if ( v121 <= n )
        v119 = v121;
      if ( *((_QWORD *)j + 27) != v119 || v119 && memcmp(s1, *((const void **)j + 26), v119) )
      {
        v122 = 0;
        v114 = j + 240;
        v123 = sub_C935B0(v78, byte_3F15413, 6, 0);
        v124 = v78[1];
        if ( v123 < v124 )
        {
          v122 = v124 - v123;
          v124 = v123;
        }
        s1 = (void *)(*v78 + v124);
        n = v122;
        v125 = sub_C93740((__int64 *)&s1, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
        v126 = n;
        v127 = v125 + 1;
        if ( v127 > n )
          v127 = n;
        v128 = n - v122 + v127;
        if ( v128 <= n )
          v126 = v128;
        if ( *((_QWORD *)j + 32) != v126 || v126 && memcmp(s1, *((const void **)j + 31), v126) )
        {
LABEL_147:
          v129 = sub_C63BB0();
          v131 = v130;
          s1 = v173;
          v132 = v129;
          v165 = 46;
          v133 = (__m128i *)sub_22409D0(&s1, &v165, 0);
          v134 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
          s1 = v133;
          v173[0] = v165;
          qmemcpy(&v133[2], "alid attribute", 14);
          *v133 = v134;
          v133[1] = _mm_load_si128((const __m128i *)&xmmword_3F81C20);
          goto LABEL_148;
        }
      }
      j = v114;
LABEL_112:
      if ( j == (char *)&unk_497A998 )
        goto LABEL_147;
LABEL_113:
      p_s1 = (size_t *)a6;
      v78 += 2;
      v104 = *(_DWORD *)j | *a6;
      *a6 = v104;
      if ( v159 != v78 )
        continue;
      break;
    }
  }
  if ( v161 )
  {
    if ( (_BYTE)v104 == 8 )
    {
      p_s1 = (size_t *)v161;
      if ( !sub_C93C90((__int64)v160, v161, 0, (unsigned __int64 *)&s1) && s1 == (void *)(unsigned int)s1 )
      {
        *a8 = (_DWORD)s1;
        goto LABEL_116;
      }
      v154 = sub_C63BB0();
      v131 = v155;
      s1 = v173;
      v132 = v154;
      v165 = 50;
      v156 = (__m128i *)sub_22409D0(&s1, &v165, 0);
      s1 = v156;
      v173[0] = v165;
      *v156 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
      v157 = _mm_load_si128((const __m128i *)&xmmword_3F81C80);
      v156[3].m128i_i16[0] = 25978;
      v156[1] = v157;
      v156[2] = _mm_load_si128((const __m128i *)&xmmword_3F81C90);
LABEL_148:
      p_s1 = (size_t *)&s1;
      n = v165;
      *((_BYTE *)s1 + v165) = 0;
      sub_C63F00(a1, (__int64)&s1, v132, v131);
      v135 = s1;
      if ( s1 == v173 )
        goto LABEL_150;
      goto LABEL_149;
    }
    v141 = sub_C63BB0();
    v165 = 103;
    v143 = v142;
    s1 = v173;
    v144 = v141;
    v145 = (__m128i *)sub_22409D0(&s1, &v165, 0);
    s1 = v145;
    v173[0] = v165;
    *v145 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    v146 = _mm_load_si128((const __m128i *)&xmmword_3F81C30);
    v145[6].m128i_i32[0] = 1970565983;
    v145[1] = v146;
    v147 = _mm_load_si128((const __m128i *)&xmmword_3F81C40);
    v145[6].m128i_i16[2] = 29538;
    v145[2] = v147;
    v148 = _mm_load_si128((const __m128i *)&xmmword_3F81C50);
    v145[6].m128i_i8[6] = 39;
    v145[3] = v148;
    v145[4] = _mm_load_si128((const __m128i *)&xmmword_3F81C60);
    v145[5] = _mm_load_si128((const __m128i *)&xmmword_3F81C70);
  }
  else
  {
    if ( v104 != 8 )
    {
LABEL_116:
      *a1 = 1;
      goto LABEL_150;
    }
    v149 = sub_C63BB0();
    v165 = 73;
    v143 = v150;
    s1 = v173;
    v144 = v149;
    v151 = (__m128i *)sub_22409D0(&s1, &v165, 0);
    s1 = v151;
    v173[0] = v165;
    *v151 = _mm_load_si128((const __m128i *)&xmmword_3F81B60);
    v152 = _mm_load_si128((const __m128i *)&xmmword_3F81BF0);
    v151[4].m128i_i64[0] = 0x6569666963657073LL;
    v151[1] = v152;
    v153 = _mm_load_si128((const __m128i *)&xmmword_3F81C00);
    v151[4].m128i_i8[8] = 114;
    v151[2] = v153;
    v151[3] = _mm_load_si128((const __m128i *)&xmmword_3F81C10);
  }
  p_s1 = (size_t *)&s1;
  n = v165;
  *((_BYTE *)s1 + v165) = 0;
  sub_C63F00(a1, (__int64)&s1, v144, v143);
  v135 = s1;
  if ( s1 != v173 )
  {
LABEL_149:
    p_s1 = (size_t *)(v173[0] + 1LL);
    j_j___libc_free_0(v135, v173[0] + 1LL);
  }
LABEL_150:
  if ( (_BYTE *)v168 != v170 )
    _libc_free(v168, p_s1);
LABEL_32:
  if ( v174 != (_QWORD *)v176 )
    _libc_free(v174, p_s1);
  return a1;
}
