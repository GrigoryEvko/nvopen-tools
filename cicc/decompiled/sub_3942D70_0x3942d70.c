// Function: sub_3942D70
// Address: 0x3942d70
//
__int64 __fastcall sub_3942D70(_QWORD *a1, __int64 **a2, unsigned __int8 a3, unsigned int a4)
{
  _QWORD *v4; // r15
  __int64 v5; // r8
  unsigned __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // r11
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  unsigned __int64 v14; // r12
  _QWORD *v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // rdi
  __m128i si128; // xmm0
  unsigned __int64 *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _WORD *v24; // rdx
  __int64 v25; // rdi
  unsigned int v26; // ebx
  __int64 v28; // rbx
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r9
  __int64 v31; // r11
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // r9
  unsigned __int64 v36; // rbx
  __int64 v37; // r9
  unsigned __int64 v38; // r8
  unsigned int v39; // ebx
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // rax
  size_t v44; // rdx
  __int64 v45; // rcx
  unsigned __int8 v46; // bl
  bool v47; // zf
  size_t v48; // r8
  unsigned __int8 v49; // r12
  _QWORD *v50; // r13
  __int64 v51; // rdi
  unsigned __int64 v52; // rdx
  __int64 v53; // rsi
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rdx
  unsigned int v57; // r15d
  __int64 v58; // rsi
  unsigned __int64 v59; // rax
  int v60; // ebx
  unsigned int v61; // eax
  int v62; // eax
  int v63; // r14d
  unsigned int v64; // ebx
  int v65; // r15d
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __int64 v68; // rdi
  unsigned __int64 v69; // rsi
  unsigned __int64 v70; // rcx
  __int64 v71; // r8
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rcx
  unsigned __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // rsi
  unsigned __int64 v77; // r9
  __int64 v78; // rsi
  unsigned __int64 v79; // rdi
  __int64 v80; // rsi
  unsigned __int8 *v81; // r11
  unsigned __int64 v82; // rsi
  size_t v83; // rax
  unsigned __int64 v84; // r9
  _QWORD *v85; // rax
  __m128i *v86; // rdx
  __int64 v87; // rdi
  __m128i v88; // xmm0
  __int64 v89; // rax
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  _WORD *v93; // rdx
  __int64 v94; // rdi
  __int64 v95; // rsi
  void *v96; // rdi
  unsigned int v97; // r12d
  unsigned int v98; // r13d
  __int64 v99; // rbx
  unsigned __int64 *v100; // rdx
  unsigned __int64 v101; // rax
  __int64 v102; // rax
  _QWORD *v103; // r13
  void *v104; // r15
  size_t v105; // r14
  __int64 v106; // rbx
  size_t v107; // r12
  size_t v108; // rdx
  int v109; // eax
  __int64 v110; // r12
  size_t v111; // rbx
  unsigned __int64 *v112; // r14
  size_t v113; // r12
  int v114; // eax
  __int64 v115; // rbx
  __int64 v116; // rax
  char *v117; // r15
  char *v118; // rbx
  __int64 v119; // r12
  unsigned __int64 v120; // r9
  __int64 v121; // r8
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rdi
  char **v125; // r12
  int v126; // ebx
  __int64 v127; // rax
  unsigned __int64 v128; // rdx
  __int64 v129; // rcx
  unsigned __int64 v130; // rax
  char *v131; // rdx
  unsigned int v132; // r11d
  unsigned int v133; // eax
  __int64 v134; // rax
  unsigned __int64 *v135; // rdi
  _QWORD *v136; // rax
  char *v137; // rdx
  __int64 v138; // r8
  size_t v139; // [rsp-10h] [rbp-120h]
  __int64 v140; // [rsp-8h] [rbp-118h]
  unsigned int v141; // [rsp+8h] [rbp-108h]
  size_t v142; // [rsp+8h] [rbp-108h]
  unsigned __int64 v143; // [rsp+8h] [rbp-108h]
  int v144; // [rsp+10h] [rbp-100h]
  unsigned int v145; // [rsp+14h] [rbp-FCh]
  int v146; // [rsp+20h] [rbp-F0h]
  _QWORD *v147; // [rsp+20h] [rbp-F0h]
  unsigned __int64 n; // [rsp+38h] [rbp-D8h]
  _QWORD *na; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *src; // [rsp+40h] [rbp-D0h]
  unsigned int srca; // [rsp+40h] [rbp-D0h]
  unsigned __int8 **v152; // [rsp+48h] [rbp-C8h]
  __int64 v153; // [rsp+48h] [rbp-C8h]
  unsigned int v154; // [rsp+50h] [rbp-C0h]
  unsigned __int8 v155; // [rsp+56h] [rbp-BAh]
  unsigned int v158; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v159; // [rsp+68h] [rbp-A8h] BYREF
  void **p_s2; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v161; // [rsp+78h] [rbp-98h] BYREF
  void *s2; // [rsp+80h] [rbp-90h] BYREF
  __int64 v163; // [rsp+88h] [rbp-88h]
  unsigned __int64 v164; // [rsp+90h] [rbp-80h] BYREF
  char v165[120]; // [rsp+98h] [rbp-78h] BYREF

  v4 = a1;
  v5 = a1[9];
  v6 = a1[10];
  if ( *((_DWORD *)a2 + 2) )
  {
    v14 = 0;
  }
  else
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = v6 + 4;
    v9 = *(_QWORD *)(v5 + 16) - v7;
    if ( v6 + 4 > v9
      || (v4[10] = v8, v10 = *(_QWORD *)(v5 + 8), v11 = v6 + 8, v12 = *(_QWORD *)(v5 + 16) - v10, v12 < v6 + 8) )
    {
LABEL_10:
      v15 = sub_16E8CB0();
      v16 = (__m128i *)v15[3];
      v17 = (__int64)v15;
      if ( v15[2] - (_QWORD)v16 <= 0x20u )
      {
        v17 = sub_16E7EE0((__int64)v15, "Unexpected end of memory buffer: ", 0x21u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
        v16[2].m128i_i8[0] = 32;
        *v16 = si128;
        v16[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        v15[3] += 33LL;
      }
      v19 = (unsigned __int64 *)(v4[10] + 4LL);
LABEL_13:
      v20 = sub_16E7A90(v17, (__int64)v19);
      v24 = *(_WORD **)(v20 + 24);
      v25 = v20;
      if ( *(_QWORD *)(v20 + 16) - (_QWORD)v24 <= 1u )
      {
        v19 = (unsigned __int64 *)".\n";
        sub_16E7EE0(v20, ".\n", 2u);
      }
      else
      {
        *v24 = 2606;
        *(_QWORD *)(v20 + 24) += 2LL;
      }
LABEL_15:
      sub_393D180(v25, (__int64)v19, (__int64)v24, v21, v22, v23);
      return 4;
    }
    if ( v6 > v9 )
      v6 = v9;
    if ( v12 > v8 )
      v12 = v8;
    v13 = *(unsigned int *)(v7 + v6);
    v4[10] = v11;
    v14 = v13 | ((unsigned __int64)*(unsigned int *)(v10 + v12) << 32);
    v6 = v11;
  }
  v28 = *(_QWORD *)(v5 + 8);
  v29 = v6 + 4;
  v30 = *(_QWORD *)(v5 + 16) - v28;
  if ( v30 < v6 + 4 )
    goto LABEL_10;
  v4[10] = v29;
  v31 = *(_QWORD *)(v5 + 8);
  v32 = v6 + 8;
  v33 = *(_QWORD *)(v5 + 16) - v31;
  if ( v33 < v6 + 8 )
    goto LABEL_10;
  if ( v30 > v6 )
    v30 = v6;
  v34 = v6 + 12;
  v35 = v4[11] + 32LL * *(unsigned int *)(v28 + v30);
  src = *(unsigned __int8 **)v35;
  v36 = *(_QWORD *)(v35 + 8);
  v4[10] = v32;
  v37 = *(_QWORD *)(v5 + 8);
  n = v36;
  v38 = *(_QWORD *)(v5 + 16) - v37;
  if ( v38 < v34 )
    goto LABEL_10;
  if ( v29 > v33 )
    v29 = v33;
  if ( v32 > v38 )
    v32 = v38;
  v39 = *(_DWORD *)(v31 + v29);
  v4[10] = v34;
  v154 = v39;
  v145 = *(_DWORD *)(v37 + v32);
  v40 = *((unsigned int *)a2 + 2);
  if ( !(_DWORD)v40 )
  {
    v41 = sub_3940400((__int64)(v4 + 1), src, n);
    v33 = 1;
    v42 = *(_QWORD *)v41;
    v152 = (unsigned __int8 **)(*(_QWORD *)v41 + 8LL);
    v43 = sub_393FEE0(v14, 1u, *(_QWORD *)(*(_QWORD *)v41 + 32LL), (bool *)&s2);
    v46 = a3;
    v47 = *(_QWORD *)(v42 + 24) == 0;
    *(_QWORD *)(v42 + 32) = v43;
    if ( !v47 )
      v46 = 0;
    a3 = v46;
    goto LABEL_31;
  }
  v97 = (unsigned __int16)a4;
  v98 = HIWORD(a4);
  v99 = **a2;
  if ( src )
  {
    s2 = &v164;
    v161 = n;
    if ( n > 0xF )
    {
      s2 = (void *)sub_22409D0((__int64)&s2, &v161, 0);
      v135 = (unsigned __int64 *)s2;
      v164 = v161;
    }
    else
    {
      if ( n == 1 )
      {
        v100 = &v164;
        LOBYTE(v164) = *src;
        v101 = 1;
LABEL_77:
        v163 = v101;
        *((_BYTE *)v100 + v101) = 0;
        goto LABEL_78;
      }
      if ( !n )
      {
        v100 = &v164;
        v101 = 0;
        goto LABEL_77;
      }
      v135 = &v164;
    }
    v33 = (unsigned __int64)src;
    memcpy(v135, src, n);
    v101 = v161;
    v100 = (unsigned __int64 *)s2;
    goto LABEL_77;
  }
  LOBYTE(v164) = 0;
  s2 = &v164;
  v163 = 0;
LABEL_78:
  v161 = __PAIR64__(v97, v98);
  v102 = *(_QWORD *)(v99 + 96);
  if ( v102 )
  {
    v45 = v99 + 88;
    do
    {
      if ( v98 > *(_DWORD *)(v102 + 32) || v98 == *(_DWORD *)(v102 + 32) && v97 > *(_DWORD *)(v102 + 36) )
      {
        v102 = *(_QWORD *)(v102 + 24);
      }
      else
      {
        v45 = v102;
        v102 = *(_QWORD *)(v102 + 16);
      }
    }
    while ( v102 );
    v153 = v45;
    if ( v99 + 88 != v45
      && v98 >= *(_DWORD *)(v45 + 32)
      && (v98 != *(_DWORD *)(v45 + 32) || v97 >= *(_DWORD *)(v45 + 36)) )
    {
      goto LABEL_91;
    }
  }
  else
  {
    v153 = v99 + 88;
  }
  p_s2 = (void **)&v161;
  v33 = v153;
  v153 = sub_3941C40((_QWORD *)(v99 + 80), v153, (__int64 **)&p_s2);
LABEL_91:
  v103 = (_QWORD *)(v153 + 48);
  if ( !*(_QWORD *)(v153 + 56) )
  {
    v103 = (_QWORD *)(v153 + 48);
    goto LABEL_113;
  }
  v147 = v4;
  v104 = s2;
  v105 = v163;
  v106 = *(_QWORD *)(v153 + 56);
  do
  {
    v107 = *(_QWORD *)(v106 + 40);
    v108 = v105;
    if ( v107 <= v105 )
      v108 = *(_QWORD *)(v106 + 40);
    if ( !v108 || (v33 = (unsigned __int64)v104, (v109 = memcmp(*(const void **)(v106 + 32), v104, v108)) == 0) )
    {
      v110 = v107 - v105;
      if ( v110 >= 0x80000000LL )
        goto LABEL_103;
      if ( v110 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_93;
      v109 = v110;
    }
    if ( v109 < 0 )
    {
LABEL_93:
      v106 = *(_QWORD *)(v106 + 24);
      continue;
    }
LABEL_103:
    v103 = (_QWORD *)v106;
    v106 = *(_QWORD *)(v106 + 16);
  }
  while ( v106 );
  v111 = v105;
  v112 = (unsigned __int64 *)v104;
  v4 = v147;
  if ( (_QWORD *)(v153 + 48) == v103 )
    goto LABEL_113;
  v113 = v103[5];
  v44 = v111;
  if ( v113 <= v111 )
    v44 = v103[5];
  if ( v44 && (v33 = v103[4], (v114 = memcmp(v112, (const void *)v33, v44)) != 0) )
  {
LABEL_112:
    if ( v114 < 0 )
      goto LABEL_113;
  }
  else
  {
    v115 = v111 - v113;
    if ( v115 <= 0x7FFFFFFF )
    {
      if ( v115 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v114 = v115;
        goto LABEL_112;
      }
LABEL_113:
      v33 = (unsigned __int64)v103;
      p_s2 = &s2;
      v116 = sub_3942120((_QWORD *)(v153 + 40), v103, (__m128i **)&p_s2);
      v112 = (unsigned __int64 *)s2;
      v103 = (_QWORD *)v116;
    }
  }
  v152 = (unsigned __int8 **)(v103 + 8);
  if ( v112 != &v164 )
  {
    v33 = v164 + 1;
    j_j___libc_free_0((unsigned __int64)v112);
  }
LABEL_31:
  v48 = v154;
  *v152 = src;
  v152[1] = (unsigned __int8 *)n;
  if ( v154 )
  {
    v49 = a3;
    v50 = v4;
    v146 = 0;
    na = v4 + 9;
    while ( 1 )
    {
      v51 = v50[9];
      v52 = v50[10];
      v53 = *(_QWORD *)(v51 + 8);
      v54 = v52 + 4;
      v55 = *(_QWORD *)(v51 + 16) - v53;
      if ( v55 < v52 + 4 )
        goto LABEL_9;
      v50[10] = v54;
      if ( v55 > v52 )
        v55 = v52;
      v56 = v52 + 8;
      v57 = *(_DWORD *)(v53 + v55);
      v58 = *(_QWORD *)(v51 + 8);
      v59 = *(_QWORD *)(v51 + 16) - v58;
      if ( v59 < v56 )
      {
LABEL_9:
        v4 = v50;
        goto LABEL_10;
      }
      v50[10] = v56;
      v25 = (__int64)na;
      if ( v54 > v59 )
        v54 = v59;
      v60 = *(_DWORD *)(v58 + v54);
      v19 = &v159;
      if ( !(unsigned __int8)sub_393EC10(na, &v159) )
        goto LABEL_15;
      v61 = v57;
      v33 = (unsigned __int64)v165;
      v57 = (unsigned __int16)v57;
      srca = HIWORD(v61);
      s2 = &v164;
      v164 = (unsigned __int64)v152;
      v163 = 0xA00000001LL;
      sub_393D800((__int64)&s2, v165, (char *)*a2, (char *)&(*a2)[*((unsigned int *)a2 + 2)]);
      if ( v49 )
      {
        if ( (char *)s2 + 8 * (unsigned int)v163 != s2 )
        {
          v155 = v49;
          v141 = (unsigned __int16)v57;
          v117 = (char *)s2 + 8 * (unsigned int)v163;
          v144 = v60;
          v118 = (char *)s2;
          do
          {
            v119 = *(_QWORD *)v118;
            v118 += 8;
            *(_QWORD *)(v119 + 16) = sub_393FEE0(v159, 1u, *(_QWORD *)(v119 + 16), (bool *)&v161);
          }
          while ( v117 != v118 );
          v57 = v141;
          v60 = v144;
          v49 = v155;
        }
        v120 = v159;
        v121 = (__int64)(v152 + 5);
        v161 = __PAIR64__(v57, srca);
        v122 = (__int64)v152[6];
        if ( !v122 )
          goto LABEL_151;
        do
        {
          if ( srca > *(_DWORD *)(v122 + 32) || srca == *(_DWORD *)(v122 + 32) && v57 > *(_DWORD *)(v122 + 36) )
          {
            v122 = *(_QWORD *)(v122 + 24);
          }
          else
          {
            v121 = v122;
            v122 = *(_QWORD *)(v122 + 16);
          }
        }
        while ( v122 );
        if ( v152 + 5 == (unsigned __int8 **)v121
          || srca < *(_DWORD *)(v121 + 32)
          || srca == *(_DWORD *)(v121 + 32) && v57 < *(_DWORD *)(v121 + 36) )
        {
LABEL_151:
          v143 = v159;
          p_s2 = (void **)&v161;
          v134 = sub_39416E0(v152 + 4, v121, (__int64 **)&p_s2);
          v120 = v143;
          v121 = v134;
        }
        v33 = 1;
        v142 = v121;
        v123 = sub_393FEE0(v120, 1u, *(_QWORD *)(v121 + 40), (bool *)&p_s2);
        v48 = v142;
        *(_QWORD *)(v142 + 40) = v123;
      }
      if ( v60 )
        break;
LABEL_135:
      if ( s2 != &v164 )
        _libc_free((unsigned __int64)s2);
      if ( ++v146 == v154 )
      {
        v4 = v50;
        goto LABEL_139;
      }
    }
    v62 = v60;
    v63 = 0;
    v64 = v57;
    v65 = v62;
    while ( 1 )
    {
      v66 = v50[9];
      v67 = v50[10];
      v68 = *(_QWORD *)(v66 + 8);
      v69 = v67 + 4;
      v70 = *(_QWORD *)(v66 + 16) - v68;
      if ( v70 < v67 + 4 )
        break;
      v50[10] = v69;
      if ( v70 > v67 )
        v70 = v67;
      if ( *(_DWORD *)(v68 + v70) != 7 )
      {
        v26 = 5;
        sub_393D180(v68, v69, v67, v70, v48, v40);
        goto LABEL_68;
      }
      v71 = *(_QWORD *)(v66 + 8);
      v72 = v67 + 8;
      v73 = *(_QWORD *)(v66 + 16) - v71;
      if ( v73 < v67 + 8 )
        break;
      v50[10] = v72;
      if ( v69 > v73 )
        v69 = v73;
      v74 = v67 + 12;
      v75 = *(unsigned int *)(v71 + v69);
      v76 = *(_QWORD *)(v66 + 8);
      v77 = *(_QWORD *)(v66 + 16) - v76;
      if ( v77 < v67 + 12 )
        break;
      v50[10] = v74;
      if ( v72 > v77 )
        v72 = v77;
      v40 = *(_QWORD *)(v66 + 8);
      v78 = *(unsigned int *)(v76 + v72);
      v79 = v67 + 16;
      v80 = v50[11] + 32 * (v75 | (v78 << 32));
      v81 = *(unsigned __int8 **)v80;
      v48 = *(_QWORD *)(v80 + 8);
      v82 = *(_QWORD *)(v66 + 16) - v40;
      if ( v82 < v67 + 16 )
        break;
      v50[10] = v79;
      if ( v74 > v82 )
        v74 = v82;
      v44 = v67 + 20;
      v33 = *(unsigned int *)(v40 + v74);
      v45 = *(_QWORD *)(v66 + 8);
      v83 = *(_QWORD *)(v66 + 16) - v45;
      if ( v83 < v44 )
        break;
      v50[10] = v44;
      if ( v49 )
      {
        if ( v79 <= v83 )
          v83 = v79;
        v84 = v33 | ((unsigned __int64)*(unsigned int *)(v45 + v83) << 32);
        v33 = srca;
        sub_39417C0((__int64)v152, srca, v64, v81, v48, v84, 1u);
        v44 = v139;
        v45 = v140;
      }
      if ( ++v63 == v65 )
        goto LABEL_135;
    }
    v85 = sub_16E8CB0();
    v86 = (__m128i *)v85[3];
    v87 = (__int64)v85;
    if ( v85[2] - (_QWORD)v86 <= 0x20u )
    {
      v87 = sub_16E7EE0((__int64)v85, "Unexpected end of memory buffer: ", 0x21u);
    }
    else
    {
      v88 = _mm_load_si128((const __m128i *)&xmmword_4530950);
      v86[2].m128i_i8[0] = 32;
      *v86 = v88;
      v86[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      v85[3] += 33LL;
    }
    v89 = sub_16E7A90(v87, v50[10] + 4LL);
    v93 = *(_WORD **)(v89 + 24);
    v94 = v89;
    if ( *(_QWORD *)(v89 + 16) - (_QWORD)v93 <= 1u )
    {
      v95 = (__int64)".\n";
      sub_16E7EE0(v89, ".\n", 2u);
    }
    else
    {
      v95 = 2606;
      *v93 = 2606;
      *(_QWORD *)(v89 + 24) += 2LL;
    }
    v26 = 4;
    sub_393D180(v94, v95, (__int64)v93, v90, v91, v92);
LABEL_68:
    v96 = s2;
    if ( s2 != &v164 )
LABEL_69:
      _libc_free((unsigned __int64)v96);
  }
  else
  {
LABEL_139:
    v124 = v145;
    if ( v145 )
    {
      v125 = (char **)a2;
      v126 = 0;
      while ( 1 )
      {
        v127 = v4[9];
        v128 = v4[10];
        v129 = *(_QWORD *)(v127 + 8);
        v130 = *(_QWORD *)(v127 + 16) - v129;
        if ( v130 < v128 + 4 )
        {
          v136 = sub_16E8CB0();
          v137 = (char *)v136[3];
          v138 = (__int64)v136;
          if ( v136[2] - (_QWORD)v137 <= 0x20u )
          {
            v138 = sub_16E7EE0((__int64)v136, "Unexpected end of memory buffer: ", 0x21u);
          }
          else
          {
            qmemcpy(v137, "Unexpected end of memory buffer: ", 0x21u);
            v136[3] += 33LL;
          }
          v17 = v138;
          v19 = (unsigned __int64 *)(v4[10] + 4LL);
          goto LABEL_13;
        }
        v4[10] = v128 + 4;
        if ( v130 > v128 )
          v130 = v128;
        v131 = *v125;
        v132 = *(_DWORD *)(v129 + v130);
        s2 = &v164;
        v164 = (unsigned __int64)v152;
        v163 = 0xA00000001LL;
        v158 = v132;
        sub_393D800((__int64)&s2, v165, v131, &v131[8 * *((unsigned int *)v125 + 2)]);
        v33 = (unsigned __int64)&s2;
        v133 = sub_3942D70(v4, &s2, a3, v158);
        if ( v133 )
          break;
        v124 = (__int64)s2;
        if ( s2 != &v164 )
          _libc_free((unsigned __int64)s2);
        if ( ++v126 == v145 )
          goto LABEL_148;
      }
      v96 = s2;
      v26 = v133;
      if ( s2 != &v164 )
        goto LABEL_69;
    }
    else
    {
LABEL_148:
      sub_393D180(v124, v33, v44, v45, v48, v40);
      return 0;
    }
  }
  return v26;
}
