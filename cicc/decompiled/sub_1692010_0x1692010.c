// Function: sub_1692010
// Address: 0x1692010
//
__int64 __fastcall sub_1692010(__int64 *a1, __int64 a2, const char *a3, const char *a4, int a5, int a6, char a7)
{
  __int64 v7; // r15
  void *v10; // rdx
  __int64 v11; // r13
  size_t v12; // rax
  _BYTE *v13; // rdi
  size_t v14; // rbx
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  size_t v20; // rax
  __m128i *v21; // rdi
  size_t v22; // r14
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  _BYTE *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  void *v29; // r14
  __int64 v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  int v33; // r15d
  __int64 v34; // r13
  __int64 v35; // rax
  int v36; // ebx
  char *v37; // r12
  unsigned __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // r15
  const char **v41; // rax
  size_t v42; // rax
  const char *v43; // rsi
  size_t v44; // rdx
  __int64 v45; // rcx
  __m128i si128; // xmm0
  __int64 v47; // rbx
  _BYTE *v48; // rdx
  unsigned __int64 v49; // rax
  const char *v50; // r13
  size_t v51; // rbx
  size_t v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdx
  int *v55; // r15
  int *v56; // r12
  size_t v57; // r14
  void *v58; // r13
  size_t v59; // rbx
  size_t v60; // rdx
  int v61; // eax
  __int64 v62; // rbx
  const void *v63; // r8
  size_t v64; // r13
  size_t v65; // rbx
  size_t v66; // rdx
  int v67; // eax
  __int64 v68; // r15
  size_t v69; // rbx
  void *v70; // rcx
  size_t v71; // rdx
  void *v72; // r8
  const void *v73; // r11
  int v74; // eax
  signed __int64 v75; // rax
  int v76; // eax
  __int64 v77; // rcx
  __int64 v78; // rax
  size_t v79; // rdx
  unsigned __int64 v80; // r11
  int *v81; // r9
  signed __int64 v82; // rax
  __int64 v83; // rax
  int *v84; // rdx
  bool v85; // al
  __int64 v86; // rdi
  int *v87; // rdx
  size_t v88; // rax
  __m128i *v89; // rcx
  __m128i *v90; // rdi
  __m128i v91; // xmm3
  __int64 v92; // r12
  const void **v93; // rax
  size_t v94; // r13
  _BYTE *v95; // r14
  const void *v96; // rbx
  unsigned __int64 v97; // rax
  __int64 v98; // rdi
  __int64 v99; // r8
  unsigned __int64 v100; // rcx
  _QWORD *v101; // rax
  unsigned int v102; // r13d
  __int64 v103; // rbx
  int v104; // r14d
  __int64 v105; // rax
  __int64 v106; // r8
  _BYTE *v107; // rdi
  __int64 v108; // rax
  const char *v109; // rsi
  size_t v110; // rdx
  unsigned __int64 v111; // rax
  __int64 v113; // rax
  _BYTE *v114; // rax
  _BYTE *v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rbx
  _BYTE *v118; // rdx
  unsigned __int64 v119; // rax
  const __m128i *v120; // r13
  signed __int64 v121; // r15
  unsigned __int64 v122; // rdi
  unsigned __int64 v123; // rax
  bool v124; // cf
  unsigned __int64 v125; // rax
  __int64 v126; // rdx
  __int64 v127; // rax
  __int64 m128i_i64; // rdx
  __m128i *v129; // rax
  __int64 v130; // rsi
  __m128i v131; // xmm4
  const __m128i *v132; // r12
  __m128i *v133; // r13
  const __m128i *v134; // rbx
  __m128i v135; // xmm1
  const __m128i *v136; // rdi
  const __m128i *v137; // rcx
  __m128i *v138; // r15
  int v139; // esi
  size_t v140; // rdx
  signed __int64 v141; // r9
  signed __int64 v142; // rax
  __int64 v143; // rax
  size_t v144; // rdx
  size_t v145; // r11
  int *v146; // r9
  signed __int64 v147; // rax
  int v148; // r13d
  int v149; // r12d
  size_t v150; // r13
  size_t v151; // rdx
  unsigned int v152; // edi
  __int64 v153; // rbx
  void *v154; // [rsp+8h] [rbp-138h]
  size_t nb; // [rsp+10h] [rbp-130h]
  const void *n; // [rsp+10h] [rbp-130h]
  size_t na; // [rsp+10h] [rbp-130h]
  size_t nc; // [rsp+10h] [rbp-130h]
  void *s1a; // [rsp+18h] [rbp-128h]
  void *s1b; // [rsp+18h] [rbp-128h]
  void *s1c; // [rsp+18h] [rbp-128h]
  __m128i *s1d; // [rsp+18h] [rbp-128h]
  const __m128i *s1; // [rsp+18h] [rbp-128h]
  void *s1e; // [rsp+18h] [rbp-128h]
  int *s1f; // [rsp+18h] [rbp-128h]
  void *v167; // [rsp+28h] [rbp-118h]
  void *v168; // [rsp+28h] [rbp-118h]
  void *v169; // [rsp+28h] [rbp-118h]
  int *v170; // [rsp+28h] [rbp-118h]
  char *v171; // [rsp+28h] [rbp-118h]
  char *v172; // [rsp+28h] [rbp-118h]
  void *v173; // [rsp+28h] [rbp-118h]
  void *v174; // [rsp+28h] [rbp-118h]
  char *v175; // [rsp+40h] [rbp-100h]
  char *v176; // [rsp+40h] [rbp-100h]
  char *s; // [rsp+58h] [rbp-E8h]
  char *sb; // [rsp+58h] [rbp-E8h]
  char *sa; // [rsp+58h] [rbp-E8h]
  unsigned __int64 v182; // [rsp+60h] [rbp-E0h]
  __int64 v183; // [rsp+60h] [rbp-E0h]
  __int64 v185; // [rsp+68h] [rbp-D8h]
  size_t v186; // [rsp+68h] [rbp-D8h]
  _BYTE *v187; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v188; // [rsp+78h] [rbp-C8h]
  _QWORD v189[2]; // [rsp+80h] [rbp-C0h] BYREF
  void *s2; // [rsp+90h] [rbp-B0h] BYREF
  size_t v191; // [rsp+98h] [rbp-A8h]
  __m128i v192; // [rsp+A0h] [rbp-A0h] BYREF
  char v193[8]; // [rsp+B0h] [rbp-90h] BYREF
  int v194; // [rsp+B8h] [rbp-88h] BYREF
  int *v195; // [rsp+C0h] [rbp-80h]
  int *v196; // [rsp+C8h] [rbp-78h]
  int *v197; // [rsp+D0h] [rbp-70h]
  __int64 v198; // [rsp+D8h] [rbp-68h]
  __m128i *v199; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v200; // [rsp+E8h] [rbp-58h]
  __m128i v201; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v202; // [rsp+100h] [rbp-40h] BYREF

  v7 = a2;
  v10 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v10 <= 9u )
  {
    v11 = sub_16E7EE0(a2, "OVERVIEW: ", 10);
  }
  else
  {
    v11 = a2;
    qmemcpy(v10, "OVERVIEW: ", 10);
    *(_QWORD *)(a2 + 24) += 10LL;
  }
  if ( !a4 )
    goto LABEL_37;
  v12 = strlen(a4);
  v13 = *(_BYTE **)(v11 + 24);
  v14 = v12;
  v15 = *(_BYTE **)(v11 + 16);
  if ( v14 > v15 - v13 )
  {
    v11 = sub_16E7EE0(v11, a4, v14);
LABEL_37:
    v15 = *(_BYTE **)(v11 + 16);
    v13 = *(_BYTE **)(v11 + 24);
    goto LABEL_38;
  }
  if ( v14 )
  {
    memcpy(v13, a4, v14);
    v16 = *(_BYTE **)(v11 + 16);
    v13 = (_BYTE *)(*(_QWORD *)(v11 + 24) + v14);
    *(_QWORD *)(v11 + 24) = v13;
    if ( v13 == v16 )
      goto LABEL_7;
LABEL_39:
    *v13 = 10;
    ++*(_QWORD *)(v11 + 24);
    v17 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v17 < *(_QWORD *)(a2 + 16) )
      goto LABEL_8;
    goto LABEL_40;
  }
LABEL_38:
  if ( v13 != v15 )
    goto LABEL_39;
LABEL_7:
  sub_16E7EE0(v11, "\n", 1);
  v17 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v17 < *(_QWORD *)(a2 + 16) )
  {
LABEL_8:
    *(_QWORD *)(a2 + 24) = v17 + 1;
    *v17 = 10;
    goto LABEL_9;
  }
LABEL_40:
  sub_16E7DE0(a2, 10);
LABEL_9:
  v18 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v18) <= 6 )
  {
    v19 = sub_16E7EE0(a2, "USAGE: ", 7);
  }
  else
  {
    *(_DWORD *)v18 = 1195463509;
    v19 = a2;
    *(_WORD *)(v18 + 4) = 14917;
    *(_BYTE *)(v18 + 6) = 32;
    *(_QWORD *)(a2 + 24) += 7LL;
  }
  if ( !a3 )
    goto LABEL_42;
  v20 = strlen(a3);
  v21 = *(__m128i **)(v19 + 24);
  v22 = v20;
  v23 = *(_QWORD *)(v19 + 16) - (_QWORD)v21;
  if ( v22 > v23 )
  {
    v19 = sub_16E7EE0(v19, a3, v22);
LABEL_42:
    v21 = *(__m128i **)(v19 + 24);
    v23 = *(_QWORD *)(v19 + 16) - (_QWORD)v21;
    goto LABEL_43;
  }
  if ( v22 )
  {
    memcpy(v21, a3, v22);
    v21 = (__m128i *)(v22 + *(_QWORD *)(v19 + 24));
    v24 = *(_QWORD *)(v19 + 16) - (_QWORD)v21;
    *(_QWORD *)(v19 + 24) = v21;
    if ( v24 <= 0x13 )
      goto LABEL_15;
LABEL_44:
    si128 = _mm_load_si128((const __m128i *)&xmmword_42AE530);
    v21[1].m128i_i32[0] = 171864948;
    *v21 = si128;
    *(_QWORD *)(v19 + 24) += 20LL;
    v25 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v25 < *(_QWORD *)(a2 + 16) )
      goto LABEL_16;
LABEL_45:
    sub_16E7DE0(a2, 10);
    goto LABEL_17;
  }
LABEL_43:
  if ( v23 > 0x13 )
    goto LABEL_44;
LABEL_15:
  sub_16E7EE0(v19, " [options] <inputs>\n", 20);
  v25 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v25 >= *(_QWORD *)(a2 + 16) )
    goto LABEL_45;
LABEL_16:
  *(_QWORD *)(a2 + 24) = v25 + 1;
  *v25 = 10;
LABEL_17:
  v195 = 0;
  v196 = &v194;
  v197 = &v194;
  v194 = 0;
  v26 = *a1;
  v27 = a1[1];
  v198 = 0;
  v28 = (v27 - v26) >> 6;
  if ( !(_DWORD)v28 )
    goto LABEL_150;
  v29 = 0;
  v182 = (unsigned int)v28;
  while ( 1 )
  {
    v33 = (_DWORD)v29 + 1;
    v34 = (_QWORD)v29 << 6;
    v35 = v26 + ((_QWORD)v29 << 6);
    if ( !*(_BYTE *)(v35 + 36) )
      goto LABEL_23;
    v36 = *(unsigned __int16 *)(v35 + 38);
    if ( a5 )
    {
      if ( (v36 & a5) == 0 )
        goto LABEL_23;
    }
    if ( (a6 & v36) != 0 )
      goto LABEL_23;
    s = *(char **)(v35 + 16);
    if ( s )
      break;
    if ( a7 )
    {
      v30 = sub_1691920(a1, v33);
      v32 = sub_1691920(v31, *(unsigned __int16 *)(v30 + 42));
      if ( !v32 )
        goto LABEL_23;
      v26 = *a1;
      s = *(char **)(*a1 + ((unsigned __int64)(unsigned int)(*(_DWORD *)(v32 + 32) - 1) << 6) + 16);
    }
    if ( s )
      break;
LABEL_23:
    v29 = (char *)v29 + 1;
    if ( (void *)v182 == v29 )
      goto LABEL_119;
LABEL_24:
    v26 = *a1;
  }
  v37 = "OPTIONS";
  if ( *(_WORD *)(v26 + v34 + 40) )
  {
    v38 = v26 + ((unsigned __int64)((unsigned int)*(unsigned __int16 *)(v26 + v34 + 40) - 1) << 6);
    v37 = *(char **)(v38 + 16);
    if ( !v37 )
    {
      v139 = *(unsigned __int16 *)(v38 + 40);
      v37 = "OPTIONS";
      if ( *(_WORD *)(v38 + 40) )
      {
        v37 = *(char **)(v26 + ((unsigned __int64)(unsigned int)(v139 - 1) << 6) + 16);
        if ( !v37 )
          v37 = (char *)sub_1691370(a1, v139);
      }
    }
  }
  v40 = sub_1691920(a1, v33);
  v41 = *(const char ***)v40;
  if ( !**(_QWORD **)v40 )
  {
    LOBYTE(v189[0]) = 0;
    v188 = 0;
    v43 = *(const char **)(v40 + 8);
    v187 = v189;
    if ( v43 )
      goto LABEL_34;
    goto LABEL_161;
  }
  v175 = (char *)*v41;
  v42 = strlen(*v41);
  v187 = v189;
  sub_16910A0((__int64 *)&v187, v175, (__int64)&v175[v42]);
  v43 = *(const char **)(v40 + 8);
  if ( !v43 )
  {
LABEL_161:
    v44 = 0;
    goto LABEL_35;
  }
LABEL_34:
  v44 = strlen(v43);
  if ( 0x3FFFFFFFFFFFFFFFLL - v188 < v44 )
    goto LABEL_168;
LABEL_35:
  sub_2241490(&v187, v43, v44, v39);
  switch ( *(_BYTE *)(v40 + 36) )
  {
    case 4:
    case 9:
    case 0xC:
      goto LABEL_51;
    case 6:
    case 7:
    case 8:
    case 0xB:
      v47 = v188;
      v48 = v187;
      v49 = 15;
      if ( v187 != (_BYTE *)v189 )
        v49 = v189[0];
      if ( v188 + 1 > v49 )
      {
        sub_2240BB0(&v187, v188, 0, 0, 1);
        v48 = v187;
      }
      v48[v47] = 32;
      v188 = v47 + 1;
      v187[v47 + 1] = 0;
LABEL_51:
      v50 = *(const char **)(*a1 + v34 + 24);
      v51 = 0x3FFFFFFFFFFFFFFFLL - v188;
      if ( v50 )
      {
        v52 = strlen(v50);
        if ( v52 > v51 )
          goto LABEL_168;
        goto LABEL_53;
      }
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v188) <= 6 )
        goto LABEL_168;
      sub_2241490(&v187, "<value>", 7, v45);
      break;
    case 0xA:
      v50 = *(const char **)(*a1 + v34 + 24);
      if ( v50 )
      {
        v117 = v188;
        v118 = v187;
        v119 = 15;
        if ( v187 != (_BYTE *)v189 )
          v119 = v189[0];
        if ( v188 + 1 > v119 )
        {
          sub_2240BB0(&v187, v188, 0, 0, 1);
          v118 = v187;
        }
        v118[v117] = 32;
        v188 = v117 + 1;
        v187[v117 + 1] = 0;
        v52 = strlen(v50);
        if ( v52 > 0x3FFFFFFFFFFFFFFFLL - v188 )
LABEL_168:
          sub_4262D8((__int64)"basic_string::append");
LABEL_53:
        sub_2241490(&v187, v50, v52, v53);
      }
      else
      {
        v148 = *(unsigned __int8 *)(v40 + 37);
        if ( *(_BYTE *)(v40 + 37) )
        {
          v176 = v37;
          v149 = 0;
          do
          {
            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v188) <= 7 )
              goto LABEL_168;
            ++v149;
            sub_2241490(&v187, " <value>", 8, v45);
          }
          while ( v148 != v149 );
          v37 = v176;
        }
      }
      break;
    default:
      break;
  }
  v54 = -1;
  s2 = &v192;
  if ( v37 )
    v54 = (__int64)&v37[strlen(v37)];
  sub_16910A0((__int64 *)&s2, v37, v54);
  v55 = v195;
  if ( !v195 )
  {
    v56 = &v194;
    goto LABEL_77;
  }
  v167 = v29;
  v56 = &v194;
  v57 = v191;
  v58 = s2;
  while ( 2 )
  {
    while ( 2 )
    {
      v59 = *((_QWORD *)v55 + 5);
      v60 = v57;
      if ( v59 <= v57 )
        v60 = *((_QWORD *)v55 + 5);
      if ( !v60 || (v61 = memcmp(*((const void **)v55 + 4), v58, v60)) == 0 )
      {
        v62 = v59 - v57;
        if ( v62 >= 0x80000000LL )
          goto LABEL_67;
        if ( v62 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v61 = v62;
          break;
        }
LABEL_58:
        v55 = (int *)*((_QWORD *)v55 + 3);
        if ( !v55 )
          goto LABEL_68;
        continue;
      }
      break;
    }
    if ( v61 < 0 )
      goto LABEL_58;
LABEL_67:
    v56 = v55;
    v55 = (int *)*((_QWORD *)v55 + 2);
    if ( v55 )
      continue;
    break;
  }
LABEL_68:
  v63 = v58;
  v64 = v57;
  v29 = v167;
  if ( v56 == &v194 )
    goto LABEL_77;
  v65 = *((_QWORD *)v56 + 5);
  v66 = v64;
  if ( v65 <= v64 )
    v66 = *((_QWORD *)v56 + 5);
  if ( v66 )
  {
    v67 = memcmp(v63, *((const void **)v56 + 4), v66);
    if ( v67 )
    {
LABEL_76:
      if ( v67 < 0 )
        goto LABEL_77;
      goto LABEL_107;
    }
  }
  if ( (__int64)(v64 - v65) >= 0x80000000LL )
    goto LABEL_107;
  if ( (__int64)(v64 - v65) > (__int64)0xFFFFFFFF7FFFFFFFLL )
  {
    v67 = v64 - v65;
    goto LABEL_76;
  }
LABEL_77:
  v68 = sub_22077B0(88);
  *(_QWORD *)(v68 + 32) = v68 + 48;
  if ( s2 == &v192 )
  {
    *(__m128i *)(v68 + 48) = _mm_load_si128(&v192);
  }
  else
  {
    *(_QWORD *)(v68 + 32) = s2;
    *(_QWORD *)(v68 + 48) = v192.m128i_i64[0];
  }
  v69 = v191;
  v191 = 0;
  v192.m128i_i8[0] = 0;
  *(_QWORD *)(v68 + 40) = v69;
  s2 = &v192;
  *(_QWORD *)(v68 + 64) = 0;
  *(_QWORD *)(v68 + 72) = 0;
  *(_QWORD *)(v68 + 80) = 0;
  if ( v56 == &v194 )
  {
    if ( v198 )
    {
      v56 = v197;
      v140 = v69;
      v141 = *((_QWORD *)v197 + 5);
      if ( v141 <= v69 )
        v140 = *((_QWORD *)v197 + 5);
      if ( v140
        && (v173 = (void *)*((_QWORD *)v197 + 5),
            LODWORD(v142) = memcmp(*((const void **)v197 + 4), *(const void **)(v68 + 32), v140),
            v141 = (signed __int64)v173,
            (_DWORD)v142) )
      {
LABEL_209:
        if ( (int)v142 < 0 )
          goto LABEL_210;
      }
      else
      {
        v142 = v141 - v69;
        if ( (__int64)(v141 - v69) < 0x80000000LL )
        {
          if ( v142 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_209;
LABEL_210:
          v85 = 0;
LABEL_103:
          if ( v56 == &v194 || v85 )
          {
LABEL_105:
            v86 = 1;
LABEL_106:
            v87 = v56;
            v56 = (int *)v68;
            sub_220F040(v86, v68, v87, &v194);
            ++v198;
            goto LABEL_107;
          }
          v150 = *((_QWORD *)v56 + 5);
          v151 = v150;
          if ( v69 <= v150 )
            v151 = v69;
          if ( !v151 || (v152 = memcmp(*(const void **)(v68 + 32), *((const void **)v56 + 4), v151)) == 0 )
          {
            v153 = v69 - v150;
            v86 = 0;
            if ( v153 >= 0x80000000LL )
              goto LABEL_106;
            if ( v153 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_105;
            v152 = v153;
          }
          v86 = v152 >> 31;
          goto LABEL_106;
        }
      }
    }
LABEL_101:
    v83 = sub_1691200((__int64)v193, v68 + 32);
    v56 = v84;
    if ( !v84 )
    {
      v72 = *(void **)(v68 + 32);
      v56 = (int *)v83;
      goto LABEL_228;
    }
    v85 = v83 != 0;
    goto LABEL_103;
  }
  v70 = (void *)*((_QWORD *)v56 + 5);
  v71 = v69;
  v72 = *(void **)(v68 + 32);
  v73 = (const void *)*((_QWORD *)v56 + 4);
  if ( (unsigned __int64)v70 <= v69 )
    v71 = *((_QWORD *)v56 + 5);
  if ( !v71 )
  {
    v75 = v69 - (_QWORD)v70;
    if ( (__int64)(v69 - (_QWORD)v70) < 0x80000000LL )
      goto LABEL_85;
    goto LABEL_89;
  }
  v154 = (void *)*((_QWORD *)v56 + 5);
  nb = v71;
  s1a = (void *)*((_QWORD *)v56 + 4);
  v168 = *(void **)(v68 + 32);
  v74 = memcmp(v168, s1a, v71);
  v72 = v168;
  v73 = s1a;
  v71 = nb;
  v70 = v154;
  if ( v74 )
  {
    if ( v74 >= 0 )
    {
LABEL_88:
      s1b = v70;
      v169 = v72;
      v76 = memcmp(v73, v72, v71);
      v72 = v169;
      v70 = s1b;
      if ( !v76 )
        goto LABEL_89;
LABEL_92:
      if ( v76 >= 0 )
        goto LABEL_228;
LABEL_93:
      n = v72;
      if ( v197 == v56 )
        goto LABEL_210;
      v78 = sub_220EEE0(v56);
      v79 = v69;
      v80 = *(_QWORD *)(v78 + 40);
      v81 = (int *)v78;
      if ( v80 <= v69 )
        v79 = *(_QWORD *)(v78 + 40);
      if ( v79 )
      {
        s1c = *(void **)(v78 + 40);
        v170 = (int *)v78;
        LODWORD(v82) = memcmp(n, *(const void **)(v78 + 32), v79);
        v81 = v170;
        v80 = (unsigned __int64)s1c;
        if ( (_DWORD)v82 )
          goto LABEL_100;
      }
      v82 = v69 - v80;
      if ( (__int64)(v69 - v80) >= 0x80000000LL )
        goto LABEL_101;
      if ( v82 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_100:
        if ( (int)v82 >= 0 )
          goto LABEL_101;
      }
      if ( !*((_QWORD *)v56 + 3) )
        goto LABEL_210;
      v56 = v81;
    }
    else
    {
LABEL_215:
      v174 = v72;
      if ( v196 != v56 )
      {
        v143 = sub_220EF80(v56);
        v144 = v69;
        v145 = *(_QWORD *)(v143 + 40);
        v146 = (int *)v143;
        if ( v145 <= v69 )
          v144 = *(_QWORD *)(v143 + 40);
        if ( v144 )
        {
          nc = *(_QWORD *)(v143 + 40);
          s1f = (int *)v143;
          LODWORD(v147) = memcmp(*(const void **)(v143 + 32), v174, v144);
          v146 = s1f;
          v145 = nc;
          if ( (_DWORD)v147 )
            goto LABEL_222;
        }
        v147 = v145 - v69;
        if ( (__int64)(v145 - v69) >= 0x80000000LL )
          goto LABEL_101;
        if ( v147 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
LABEL_222:
          if ( (int)v147 >= 0 )
            goto LABEL_101;
        }
        if ( !*((_QWORD *)v146 + 3) )
          v56 = v146;
        v85 = *((_QWORD *)v146 + 3) != 0;
        goto LABEL_103;
      }
    }
    v85 = 1;
    goto LABEL_103;
  }
  v75 = v69 - (_QWORD)v154;
  if ( (__int64)(v69 - (_QWORD)v154) >= 0x80000000LL )
    goto LABEL_88;
LABEL_85:
  if ( v75 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v75 < 0 )
    goto LABEL_215;
  if ( v71 )
    goto LABEL_88;
LABEL_89:
  v77 = (__int64)v70 - v69;
  if ( v77 < 0x80000000LL )
  {
    if ( v77 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
      v76 = v77;
      goto LABEL_92;
    }
    goto LABEL_93;
  }
LABEL_228:
  if ( (void *)(v68 + 48) != v72 )
    j_j___libc_free_0(v72, *(_QWORD *)(v68 + 48) + 1LL);
  j_j___libc_free_0(v68, 88);
LABEL_107:
  v199 = &v201;
  sub_1690FF0((__int64 *)&v199, v187, (__int64)&v187[v188]);
  v202.m128i_i64[0] = (__int64)s;
  v88 = strlen(s);
  v89 = (__m128i *)*((_QWORD *)v56 + 9);
  v202.m128i_i64[1] = v88;
  if ( v89 != *((__m128i **)v56 + 10) )
  {
    v90 = v199;
    if ( v89 )
    {
      v89->m128i_i64[0] = (__int64)v89[1].m128i_i64;
      if ( v199 == &v201 )
      {
        v89[1] = _mm_load_si128(&v201);
      }
      else
      {
        v89->m128i_i64[0] = (__int64)v199;
        v89[1].m128i_i64[0] = v201.m128i_i64[0];
      }
      v199 = &v201;
      v90 = &v201;
      v89->m128i_i64[1] = v200;
      v91 = _mm_load_si128(&v202);
      v200 = 0;
      v201.m128i_i8[0] = 0;
      v89[2] = v91;
      v89 = (__m128i *)*((_QWORD *)v56 + 9);
    }
    *((_QWORD *)v56 + 9) = v89 + 3;
    goto LABEL_113;
  }
  v120 = (const __m128i *)*((_QWORD *)v56 + 8);
  v121 = (char *)v89 - (char *)v120;
  v122 = 0xAAAAAAAAAAAAAAABLL * (v89 - v120);
  if ( v122 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v123 = 1;
  if ( v122 )
    v123 = 0xAAAAAAAAAAAAAAABLL * (v89 - v120);
  v124 = __CFADD__(v122, v123);
  v125 = v122 + v123;
  if ( v124 )
  {
    v126 = 0x7FFFFFFFFFFFFFE0LL;
    goto LABEL_177;
  }
  if ( v125 )
  {
    if ( v125 > 0x2AAAAAAAAAAAAAALL )
      v125 = 0x2AAAAAAAAAAAAAALL;
    v126 = 48 * v125;
LABEL_177:
    s1d = v89;
    v171 = (char *)v126;
    v127 = sub_22077B0(v126);
    v89 = s1d;
    sa = (char *)v127;
    v172 = &v171[v127];
    m128i_i64 = v127 + 48;
  }
  else
  {
    v172 = 0;
    m128i_i64 = 48;
    sa = 0;
  }
  v129 = (__m128i *)&sa[v121];
  if ( &sa[v121] )
  {
    v129->m128i_i64[0] = (__int64)v129[1].m128i_i64;
    if ( v199 == &v201 )
    {
      v129[1] = _mm_load_si128(&v201);
    }
    else
    {
      v129->m128i_i64[0] = (__int64)v199;
      v129[1].m128i_i64[0] = v201.m128i_i64[0];
    }
    v130 = v200;
    v131 = _mm_load_si128(&v202);
    v199 = &v201;
    v200 = 0;
    v129->m128i_i64[1] = v130;
    v201.m128i_i8[0] = 0;
    v129[2] = v131;
  }
  if ( v89 != v120 )
  {
    s1 = v120;
    na = (size_t)v56;
    v132 = v120 + 1;
    v133 = (__m128i *)sa;
    v134 = v89;
    while ( 1 )
    {
      if ( v133 )
      {
        v133->m128i_i64[0] = (__int64)v133[1].m128i_i64;
        v137 = (const __m128i *)v132[-1].m128i_i64[0];
        if ( v132 == v137 )
        {
          v133[1] = _mm_loadu_si128(v132);
        }
        else
        {
          v133->m128i_i64[0] = (__int64)v137;
          v133[1].m128i_i64[0] = v132->m128i_i64[0];
        }
        v133->m128i_i64[1] = v132[-1].m128i_i64[1];
        v135 = _mm_loadu_si128(v132 + 1);
        v132[-1].m128i_i64[0] = (__int64)v132;
        v132[-1].m128i_i64[1] = 0;
        v132->m128i_i8[0] = 0;
        v133[2] = v135;
      }
      v136 = (const __m128i *)v132[-1].m128i_i64[0];
      if ( v132 != v136 )
        j_j___libc_free_0(v136, v132->m128i_i64[0] + 1);
      if ( v134 == &v132[2] )
        break;
      v132 += 3;
      v133 += 3;
    }
    v56 = (int *)na;
    v138 = v133;
    v120 = s1;
    m128i_i64 = (__int64)v138[6].m128i_i64;
  }
  if ( v120 )
  {
    s1e = (void *)m128i_i64;
    j_j___libc_free_0(v120, *((_QWORD *)v56 + 10) - (_QWORD)v120);
    m128i_i64 = (__int64)s1e;
  }
  v90 = v199;
  *((_QWORD *)v56 + 9) = m128i_i64;
  *((_QWORD *)v56 + 8) = sa;
  *((_QWORD *)v56 + 10) = v172;
LABEL_113:
  if ( v90 != &v201 )
    j_j___libc_free_0(v90, v201.m128i_i64[0] + 1);
  if ( s2 != &v192 )
    j_j___libc_free_0(s2, v192.m128i_i64[0] + 1);
  if ( v187 == (_BYTE *)v189 )
    goto LABEL_23;
  v29 = (char *)v29 + 1;
  j_j___libc_free_0(v187, v189[0] + 1LL);
  if ( (void *)v182 != v29 )
    goto LABEL_24;
LABEL_119:
  v7 = a2;
  v92 = (__int64)v196;
  if ( v196 != &v194 )
  {
    v93 = (const void **)v196;
    v94 = *((_QWORD *)v196 + 5);
    v95 = *(_BYTE **)(a2 + 24);
LABEL_144:
    v96 = *(const void **)(v92 + 32);
    if ( !v94 )
      goto LABEL_148;
    if ( !memcmp(*(const void **)(v92 + 32), v93[4], v94) )
    {
      if ( v94 <= *(_QWORD *)(a2 + 16) - (_QWORD)v95 )
      {
LABEL_147:
        memcpy(v95, v96, v94);
        v95 = (_BYTE *)(v94 + *(_QWORD *)(a2 + 24));
        *(_QWORD *)(a2 + 24) = v95;
LABEL_148:
        v98 = a2;
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v95 > 1u )
          goto LABEL_126;
LABEL_149:
        sub_16E7EE0(v98, ":\n", 2);
        goto LABEL_127;
      }
    }
    else
    {
      while ( 1 )
      {
        if ( *(_BYTE **)(a2 + 16) == v95 )
        {
          sub_16E7EE0(a2, "\n", 1);
          v95 = *(_BYTE **)(a2 + 24);
        }
        else
        {
          *v95 = 10;
          v95 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 1LL);
          *(_QWORD *)(a2 + 24) = v95;
        }
        v94 = *(_QWORD *)(v92 + 40);
        v96 = *(const void **)(v92 + 32);
        v97 = *(_QWORD *)(a2 + 16) - (_QWORD)v95;
        if ( v97 < v94 )
          break;
        v98 = a2;
        if ( v94 )
          goto LABEL_147;
LABEL_125:
        if ( v97 <= 1 )
          goto LABEL_149;
LABEL_126:
        *(_WORD *)v95 = 2618;
        *(_QWORD *)(v98 + 24) += 2LL;
LABEL_127:
        v99 = *(_QWORD *)(v92 + 64);
        v100 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(v92 + 72) - v99) >> 4);
        if ( (_DWORD)v100 )
        {
          v101 = (_QWORD *)(v99 + 8);
          v102 = 0;
          do
          {
            if ( (unsigned int)*v101 <= 0x17 && v102 < (unsigned int)*v101 )
              v102 = *v101;
            v101 += 6;
          }
          while ( (_QWORD *)(v99 + 48LL * (unsigned int)(v100 - 1) + 56) != v101 );
          v103 = 0;
          v183 = 48LL * (unsigned int)v100;
          while ( 1 )
          {
            v185 = v103 + v99;
            v104 = v102 - *(_DWORD *)(v103 + v99 + 8);
            v105 = sub_16E8750(a2, 2);
            sub_16E7EE0(v105, *(const char **)v185, *(_QWORD *)(v185 + 8));
            if ( v104 < 0 )
            {
              v114 = *(_BYTE **)(a2 + 24);
              if ( *(_BYTE **)(a2 + 16) == v114 )
              {
                sub_16E7EE0(a2, "\n", 1);
              }
              else
              {
                *v114 = 10;
                ++*(_QWORD *)(a2 + 24);
              }
              v104 = v102 + 2;
            }
            v106 = sub_16E8750(a2, (unsigned int)(v104 + 1));
            v107 = *(_BYTE **)(v106 + 24);
            v108 = v103 + *(_QWORD *)(v92 + 64);
            v109 = *(const char **)(v108 + 32);
            v110 = *(_QWORD *)(v108 + 40);
            v111 = *(_QWORD *)(v106 + 16);
            if ( v110 > v111 - (unsigned __int64)v107 )
            {
              v113 = sub_16E7EE0(v106, v109);
              v107 = *(_BYTE **)(v113 + 24);
              v106 = v113;
              v111 = *(_QWORD *)(v113 + 16);
            }
            else if ( v110 )
            {
              sb = (char *)v106;
              v186 = v110;
              memcpy(v107, v109, v110);
              v106 = (__int64)sb;
              v115 = (_BYTE *)(*((_QWORD *)sb + 3) + v186);
              v111 = *((_QWORD *)sb + 2);
              *((_QWORD *)sb + 3) = v115;
              v107 = v115;
            }
            if ( (unsigned __int64)v107 < v111 )
            {
              v103 += 48;
              *(_QWORD *)(v106 + 24) = v107 + 1;
              *v107 = 10;
              if ( v103 == v183 )
                break;
            }
            else
            {
              v103 += 48;
              sub_16E7DE0(v106, 10);
              if ( v103 == v183 )
                break;
            }
            v99 = *(_QWORD *)(v92 + 64);
          }
        }
        v92 = sub_220EEE0(v92);
        if ( (int *)v92 == &v194 )
          goto LABEL_150;
        v93 = (const void **)v196;
        v94 = *(_QWORD *)(v92 + 40);
        v95 = *(_BYTE **)(a2 + 24);
        if ( v94 == *((_QWORD *)v196 + 5) )
          goto LABEL_144;
      }
    }
    v116 = sub_16E7EE0(a2, (const char *)v96, v94);
    v95 = *(_BYTE **)(v116 + 24);
    v98 = v116;
    v97 = *(_QWORD *)(v116 + 16) - (_QWORD)v95;
    goto LABEL_125;
  }
LABEL_150:
  if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v7 + 8) )
    sub_16E7BA0(v7);
  return sub_1691150(v195);
}
