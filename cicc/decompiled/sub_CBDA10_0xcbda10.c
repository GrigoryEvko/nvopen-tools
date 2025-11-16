// Function: sub_CBDA10
// Address: 0xcbda10
//
__int64 __fastcall sub_CBDA10(__int64 a1, const char *a2, char *a3, __m128i *a4, char a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // r15
  int v10; // eax
  bool v11; // zf
  char *v12; // rax
  char *v13; // rbx
  char *v14; // rcx
  const void *v15; // r14
  char v16; // r15
  signed __int64 v17; // rdx
  __int64 v18; // r12
  int v19; // ebx
  unsigned __int64 k; // rcx
  int v21; // r13d
  int v22; // eax
  int v23; // edi
  int v24; // eax
  char *i; // rsi
  char *v27; // rax
  __int64 v28; // r8
  const char *v29; // rbx
  __int64 v30; // rsi
  char *v31; // rcx
  const void *v32; // r14
  char v33; // r15
  signed __int64 v34; // rdx
  size_t v35; // r13
  __int64 v36; // rax
  void *v37; // r12
  __int64 v38; // rax
  void *v39; // rcx
  _BYTE *v40; // rax
  char *v41; // r12
  int v42; // r8d
  int v43; // r13d
  int v44; // eax
  int v45; // r13d
  char *v46; // rax
  int v47; // ebx
  unsigned __int8 *v48; // rax
  int v49; // r13d
  __int64 v50; // rcx
  __int64 v51; // rdi
  int v52; // r12d
  __int64 v53; // rax
  int v54; // r10d
  int v55; // ebx
  int v56; // r10d
  int v57; // r8d
  int v58; // eax
  int v59; // eax
  unsigned __int8 *m; // r12
  char *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  unsigned __int8 *v66; // r10
  __int64 v67; // r12
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rax
  unsigned __int64 v72; // r10
  unsigned __int64 v73; // r13
  unsigned __int8 *v74; // r12
  unsigned __int8 *v75; // rax
  __int64 v76; // r8
  __int64 v77; // rcx
  unsigned __int64 v78; // rax
  __m128i *v79; // rdx
  char *v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rax
  __int64 v83; // rax
  char *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  char *v89; // r12
  __int64 v90; // r13
  unsigned __int64 v91; // rax
  unsigned __int64 v92; // rdx
  __int64 v93; // rax
  char *v94; // rsi
  char *v95; // rax
  __int64 j; // rax
  __int64 v97; // rdi
  __int64 v98; // rcx
  unsigned __int64 v99; // rax
  __m128i *v100; // rdx
  int v101; // r13d
  int v102; // eax
  __int64 v103; // rax
  __int64 v105; // [rsp+10h] [rbp-110h]
  unsigned int v106; // [rsp+18h] [rbp-108h]
  int v107; // [rsp+24h] [rbp-FCh]
  unsigned __int64 v108; // [rsp+28h] [rbp-F8h]
  char *v109; // [rsp+30h] [rbp-F0h]
  char *v110; // [rsp+30h] [rbp-F0h]
  __int64 v111; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v112; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v113; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v114; // [rsp+40h] [rbp-E0h]
  int v115; // [rsp+40h] [rbp-E0h]
  int v116; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v117; // [rsp+40h] [rbp-E0h]
  char *v118; // [rsp+48h] [rbp-D8h]
  char *v119; // [rsp+50h] [rbp-D0h]
  char *v120; // [rsp+50h] [rbp-D0h]
  char *v121; // [rsp+58h] [rbp-C8h]
  int v122; // [rsp+58h] [rbp-C8h]
  int v123; // [rsp+58h] [rbp-C8h]
  int v124; // [rsp+58h] [rbp-C8h]
  _DWORD *v125; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *dest; // [rsp+60h] [rbp-C0h]
  void *desta; // [rsp+60h] [rbp-C0h]
  __int64 v128; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v129; // [rsp+68h] [rbp-B8h]
  __int64 v130; // [rsp+68h] [rbp-B8h]
  unsigned __int8 *v131; // [rsp+68h] [rbp-B8h]
  char *s1; // [rsp+70h] [rbp-B0h]
  const char *s1a; // [rsp+70h] [rbp-B0h]
  _QWORD *s1b; // [rsp+70h] [rbp-B0h]
  __int64 v135; // [rsp+78h] [rbp-A8h]
  int v136; // [rsp+78h] [rbp-A8h]
  __int64 v137; // [rsp+78h] [rbp-A8h]
  int v138; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v139; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v140; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v141; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v142; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v143; // [rsp+78h] [rbp-A8h]
  __int64 v144; // [rsp+80h] [rbp-A0h] BYREF
  int v145; // [rsp+88h] [rbp-98h]
  __int64 v146; // [rsp+90h] [rbp-90h]
  const char *v147; // [rsp+98h] [rbp-88h]
  char *v148; // [rsp+A0h] [rbp-80h]
  char *v149; // [rsp+A8h] [rbp-78h]
  char *v150; // [rsp+B0h] [rbp-70h]
  __int64 v151; // [rsp+B8h] [rbp-68h]
  __int64 v152; // [rsp+C0h] [rbp-60h]
  __int64 v153; // [rsp+C8h] [rbp-58h]
  void *v154; // [rsp+D0h] [rbp-50h]
  void *v155; // [rsp+D8h] [rbp-48h]
  char *v156; // [rsp+E0h] [rbp-40h]
  char *v157; // [rsp+E8h] [rbp-38h]

  v106 = 2;
  if ( *(_DWORD *)a1 != 62053 )
    return v106;
  v6 = *(_QWORD *)(a1 + 24);
  v111 = v6;
  if ( *(_DWORD *)v6 != 53829 )
    return v106;
  v106 = *(_DWORD *)(v6 + 72) & 4;
  if ( v106 )
    return 2;
  v7 = *(_QWORD *)(v6 + 64);
  v107 = a5 & 7;
  v9 = *(_QWORD *)(v6 + 56) + 1LL;
  v10 = *(_DWORD *)(v6 + 40) & 4;
  if ( *(__int64 *)(*(_QWORD *)(a1 + 24) + 48LL) > 64 )
  {
    v11 = v10 == 0;
    v27 = 0;
    if ( v11 )
      v27 = a3;
    v28 = a5 & 4;
    v110 = v27;
    if ( (_DWORD)v28 )
    {
      s1a = &a2[a4->m128i_i64[0]];
      v118 = (char *)&a2[a4->m128i_i64[1]];
    }
    else
    {
      s1a = a2;
      v118 = (char *)&a2[strlen(a2)];
    }
    v29 = s1a;
    if ( s1a > v118 )
      return 16;
    v30 = v111;
    v31 = *(char **)(v111 + 96);
    if ( v31 )
    {
      if ( s1a >= v118 )
      {
        v29 = s1a;
      }
      else
      {
        v137 = v9;
        v130 = v7;
        v32 = *(const void **)(v111 + 96);
        v33 = *v31;
        while ( 1 )
        {
          if ( *v29 == v33 )
          {
            v34 = *(int *)(v111 + 104);
            if ( v118 - v29 >= v34 )
            {
              v30 = (__int64)v32;
              if ( !memcmp(v29, v32, v34) )
                break;
            }
          }
          if ( v118 == ++v29 )
            return 1;
        }
        v9 = v137;
        v7 = v130;
      }
      if ( v118 == v29 )
        return 1;
    }
    v147 = a2;
    v146 = 0;
    v145 = v107;
    v144 = v111;
    v148 = (char *)s1a;
    v151 = 0;
    v149 = v118;
    v35 = *(_QWORD *)(v111 + 48);
    v36 = malloc(4 * v35, v30, a3, v31, v28, a6);
    v153 = v36;
    v37 = (void *)v36;
    if ( !v36 )
      return 12;
    v154 = (void *)v36;
    desta = (void *)(v35 + v36);
    v155 = (void *)(v35 + v36);
    v120 = (char *)(v35 + v35 + v36);
    v156 = v120;
    v152 = 4;
    v157 = &v120[v35];
    memset(&v120[v35], 0, v35);
    v38 = v111;
    v39 = v37;
    v138 = 128;
LABEL_66:
    v40 = memset(v39, 0, *(_QWORD *)(v38 + 48));
    v40[v9] = 1;
    v41 = (char *)sub_CBB420(v144, v9, v7, (__int64)v40, 132, (__int64)v40);
    memmove(desta, v41, *(_QWORD *)(v144 + 48));
    v131 = 0;
    while ( 1 )
    {
      v47 = 128;
      if ( v149 != s1a )
        v47 = *s1a;
      i = (char *)desta;
      v125 = (_DWORD *)v144;
      v11 = memcmp(v41, desta, *(_QWORD *)(v144 + 48)) == 0;
      v48 = v131;
      if ( v11 )
        v48 = (unsigned __int8 *)s1a;
      v131 = v48;
      if ( v138 == 10 )
        break;
      if ( v138 != 128 )
      {
        if ( v47 == 10 )
        {
          v42 = v125[10] & 8;
          if ( v42 )
            goto LABEL_94;
        }
        else
        {
          if ( v47 == 128 && (v145 & 2) == 0 )
            goto LABEL_94;
          v42 = 0;
        }
        goto LABEL_71;
      }
      if ( (v145 & 1) == 0 )
        goto LABEL_191;
      if ( v47 == 10 )
      {
        if ( (v125[10] & 8) == 0 )
          goto LABEL_83;
LABEL_94:
        v49 = v125[20];
        v42 = 130;
        if ( v49 > 0 )
        {
LABEL_95:
          v50 = (__int64)v41;
          v51 = (__int64)v125;
          v52 = v42;
          goto LABEL_97;
        }
LABEL_71:
        if ( v138 == 128 )
          goto LABEL_83;
LABEL_72:
        v43 = (unsigned __int8)v138;
        v122 = v42;
        v44 = isalnum((unsigned __int8)v138);
        v42 = v122;
        if ( !v44 )
        {
          if ( v138 != 95 )
          {
            if ( v47 != 128 )
              goto LABEL_75;
LABEL_80:
            v124 = v42;
            v44 = isalnum(v43);
            v42 = v124;
          }
          if ( !v44 && v138 != 95 )
            goto LABEL_83;
        }
        v101 = 134;
        if ( v42 != 130 )
        {
          if ( v47 == 128 || (v102 = isalnum((unsigned __int8)v47), v47 == 95) || v102 )
          {
LABEL_83:
            v46 = &v41[v7];
            if ( v41[v7] )
              goto LABEL_200;
            goto LABEL_84;
          }
        }
LABEL_226:
        i = (char *)v9;
        v41 = (char *)sub_CBB420(v144, v9, v7, (__int64)v41, v101, (__int64)v41);
        goto LABEL_83;
      }
      if ( v47 != 128 )
        goto LABEL_83;
      if ( (v145 & 2) == 0 )
        goto LABEL_94;
      v46 = &v41[v7];
      if ( v41[v7] )
      {
LABEL_200:
        v150 = (char *)v131;
        if ( !*v46 )
        {
          _libc_free(v146, i);
          _libc_free(v151, i);
          _libc_free(v153, i);
          return 1;
        }
        if ( !v110 && !*(_DWORD *)(v111 + 120) )
        {
          v97 = v146;
          goto LABEL_258;
        }
        for ( i = (char *)v131; ; i = ++v150 )
        {
          v84 = sub_CBD060(&v144, i, v118, v9, v7);
          if ( v84 )
            break;
        }
        v89 = v84;
        if ( v110 == (char *)1 && !*(_DWORD *)(v111 + 120) )
        {
          v97 = v146;
          a4->m128i_i64[0] = v150 - v147;
          a4->m128i_i64[1] = v84 - v147;
LABEL_258:
          if ( v97 )
            _libc_free(v97, i);
          if ( v151 )
            _libc_free(v151, i);
          v81 = v153;
          goto LABEL_174;
        }
        v90 = *(_QWORD *)(v144 + 112);
        if ( !v146 )
        {
          v146 = malloc(16 * (v90 + 1), i, v85, v86, v87, v88);
          if ( !v146 )
          {
LABEL_265:
            _libc_free(v153, i);
            return 12;
          }
        }
        v91 = 1;
        if ( v90 )
        {
          do
          {
            v92 = v91++;
            v92 *= 16LL;
            *(_QWORD *)(v146 + v92 + 8) = -1;
            v86 = v146;
            *(_QWORD *)(v146 + v92) = -1;
            v85 = v144;
          }
          while ( v91 <= *(_QWORD *)(v144 + 112) );
        }
        if ( *(_DWORD *)(v111 + 120) | v145 & 0x400 )
        {
          v93 = *(_QWORD *)(v111 + 128);
          if ( v93 <= 0 )
            goto LABEL_216;
          if ( v151 )
            goto LABEL_216;
          v103 = malloc(8 * v93 + 8, i, v85, v86, v87, v88);
          v151 = v103;
          if ( v103 )
            goto LABEL_216;
          _libc_free(v146, i);
          goto LABEL_265;
        }
        for ( j = (__int64)sub_CBD5A0(&v144, v150, v89, v9, v7);
              ;
              j = sub_CBC8D0(&v144, (unsigned __int8 *)v150, (unsigned __int8 *)v89, v9, v7, 0, 0) )
        {
          if ( j )
            goto LABEL_218;
          v94 = v150;
          if ( v150 >= v89 )
            break;
          v95 = sub_CBD060(&v144, v150, v89 - 1, v9, v7);
          v94 = v150;
          v89 = v95;
          if ( !v95 )
            break;
LABEL_216:
          ;
        }
        if ( v118 == v94 )
        {
LABEL_218:
          i = v110;
          v97 = v146;
          if ( v110 )
          {
            a4->m128i_i64[0] = v150 - v147;
            a4->m128i_i64[1] = v89 - v147;
            if ( v110 != (char *)1 )
            {
              v98 = v144;
              v99 = 1;
              v100 = a4 + 1;
              do
              {
                if ( *(_QWORD *)(v98 + 112) >= v99 )
                {
                  *v100 = _mm_loadu_si128((const __m128i *)(v97 + 16 * v99));
                }
                else
                {
                  v100->m128i_i64[0] = -1;
                  v100->m128i_i64[1] = -1;
                }
                ++v99;
                ++v100;
              }
              while ( v110 != (char *)v99 );
            }
          }
          goto LABEL_258;
        }
        v39 = v154;
        s1a = v94 + 1;
        desta = v155;
        v120 = v156;
        if ( v94 + 1 == v148 )
          v138 = 128;
        else
          v138 = *v94;
        v38 = v144;
        goto LABEL_66;
      }
LABEL_84:
      if ( v118 == s1a )
        goto LABEL_200;
      memmove(v120, v41, *(_QWORD *)(v144 + 48));
      memmove(v41, desta, *(_QWORD *)(v144 + 48));
      ++s1a;
      v138 = v47;
      v41 = (char *)sub_CBB420(v144, v9, v7, (__int64)v120, v47, (__int64)v41);
    }
    v42 = v125[10] & 8;
    if ( !v42 )
    {
      if ( v47 == 128 && (v145 & 2) == 0 )
        goto LABEL_94;
      goto LABEL_72;
    }
LABEL_191:
    v49 = v125[19];
    if ( v47 == 10 )
    {
      if ( (v125[10] & 8) == 0 )
      {
        v42 = 129;
        if ( v49 > 0 )
          goto LABEL_95;
LABEL_75:
        v123 = v42;
        v45 = isalnum((unsigned __int8)v47);
        if ( v45 || v47 == 95 )
        {
          if ( v138 == 128 || !isalnum((unsigned __int8)v138) && v138 != 95 || v47 == 95 || v45 )
            v101 = 133;
          else
            v101 = 134;
          goto LABEL_226;
        }
        v42 = v123;
LABEL_78:
        if ( v138 == 128 )
          goto LABEL_83;
        v43 = (unsigned __int8)v138;
        goto LABEL_80;
      }
    }
    else
    {
      if ( v47 != 128 )
      {
        if ( v49 <= 0 )
          goto LABEL_99;
        goto LABEL_194;
      }
      if ( (v145 & 2) != 0 )
      {
        if ( v49 <= 0 )
        {
LABEL_242:
          v42 = 129;
          goto LABEL_78;
        }
LABEL_194:
        v50 = (__int64)v41;
        v51 = (__int64)v125;
        v52 = 129;
LABEL_97:
        while ( 1 )
        {
          i = (char *)v9;
          v53 = sub_CBB420(v51, v9, v7, v50, v52, v50);
          v50 = v53;
          if ( !--v49 )
            break;
          v51 = v144;
        }
        v42 = v52;
        v41 = (char *)v53;
        if ( v42 != 129 )
          goto LABEL_71;
LABEL_99:
        if ( v47 != 128 )
        {
          v42 = 129;
          goto LABEL_75;
        }
        goto LABEL_242;
      }
    }
    v42 = 131;
    v49 += v125[20];
    if ( v49 > 0 )
      goto LABEL_95;
    goto LABEL_71;
  }
  v11 = v10 == 0;
  v12 = 0;
  if ( v11 )
    v12 = a3;
  v109 = v12;
  if ( (a5 & 4) != 0 )
  {
    s1 = (char *)&a2[a4->m128i_i64[0]];
    v121 = (char *)&a2[a4->m128i_i64[1]];
  }
  else
  {
    s1 = (char *)a2;
    v121 = (char *)&a2[strlen(a2)];
  }
  v13 = s1;
  if ( s1 > v121 )
    return 16;
  v14 = *(char **)(v111 + 96);
  if ( v14 )
  {
    if ( s1 >= v121 )
    {
      v13 = s1;
    }
    else
    {
      v135 = v9;
      v128 = v7;
      v15 = *(const void **)(v111 + 96);
      v16 = *v14;
      while ( 1 )
      {
        if ( *v13 == v16 )
        {
          v17 = *(int *)(v111 + 104);
          if ( v121 - v13 >= v17 && !memcmp(v13, v15, v17) )
            break;
        }
        if ( v121 == ++v13 )
          return 1;
      }
      v9 = v135;
      v7 = v128;
    }
    if ( v121 == v13 )
      return 1;
  }
  v146 = 0;
  v151 = 0;
  v145 = v107;
  v144 = v111;
  v148 = s1;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v105 = 0;
  v147 = a2;
  v18 = v111;
  v149 = v121;
  v119 = v121;
  v19 = 128;
  while ( 2 )
  {
    dest = 0;
    v129 = sub_CBAFE0(v18, v9, v7, 1LL << v9, 132, 1LL << v9);
    for ( k = v129; ; k = sub_CBAFE0(v18, v9, v7, k, v19, v129) )
    {
      i = v119;
      if ( s1 != v119 )
      {
        v136 = *s1;
        i = (char *)dest;
        if ( v129 == k )
          i = s1;
        dest = (unsigned __int8 *)i;
LABEL_43:
        if ( v19 == 10 )
        {
          if ( (*(_BYTE *)(v18 + 40) & 8) != 0 )
            goto LABEL_106;
          if ( v136 != 128 )
            goto LABEL_46;
          goto LABEL_182;
        }
        if ( v19 != 128 )
        {
          if ( v136 == 10 )
            goto LABEL_129;
          v21 = 0;
          if ( v136 != 128 )
            goto LABEL_24;
LABEL_104:
          i = (char *)(v107 & 2);
          if ( (v107 & 2) != 0 )
          {
            v136 = 128;
            v21 = 0;
            goto LABEL_24;
          }
          goto LABEL_118;
        }
        goto LABEL_114;
      }
      v136 = 128;
      if ( v129 != k )
        goto LABEL_43;
      if ( v19 == 10 )
      {
        if ( (*(_BYTE *)(v18 + 40) & 8) == 0 )
        {
          dest = (unsigned __int8 *)s1;
LABEL_182:
          if ( (v107 & 2) != 0 )
          {
LABEL_46:
            v21 = 0;
            goto LABEL_25;
          }
          goto LABEL_118;
        }
        v54 = *(_DWORD *)(v18 + 76);
        dest = (unsigned __int8 *)s1;
        goto LABEL_136;
      }
      dest = (unsigned __int8 *)s1;
      if ( v19 != 128 )
        goto LABEL_104;
LABEL_114:
      if ( (v107 & 1) == 0 )
      {
LABEL_106:
        v54 = *(_DWORD *)(v18 + 76);
        if ( v136 == 10 )
        {
          if ( (*(_BYTE *)(v18 + 40) & 8) == 0 )
          {
            if ( v54 <= 0 )
              goto LABEL_113;
            goto LABEL_109;
          }
          goto LABEL_140;
        }
        if ( v136 != 128 )
        {
          if ( v54 <= 0 )
            goto LABEL_112;
LABEL_109:
          v115 = v19;
          v21 = 129;
          v55 = v54;
          goto LABEL_110;
        }
LABEL_136:
        if ( (v107 & 2) != 0 )
        {
          if ( v54 <= 0 )
            goto LABEL_138;
          goto LABEL_109;
        }
LABEL_140:
        v56 = *(_DWORD *)(v18 + 80) + v54;
        v21 = 131;
        if ( v56 <= 0 )
          goto LABEL_24;
LABEL_120:
        v115 = v19;
        v55 = v56;
        do
        {
LABEL_110:
          i = (char *)v9;
          k = sub_CBAFE0(v18, v9, v7, k, v21, k);
          --v55;
        }
        while ( v55 );
        v19 = v115;
        if ( v21 == 129 )
        {
LABEL_112:
          if ( v136 == 128 )
          {
LABEL_138:
            v136 = 128;
            v21 = 129;
          }
          else
          {
LABEL_113:
            v21 = 129;
LABEL_28:
            v113 = k;
            v24 = isalnum((unsigned __int8)v136);
            k = v113;
            if ( v136 == 95 || v24 )
            {
              if ( v19 == 128
                || (v108 = v113, v116 = v24, v58 = isalnum((unsigned __int8)v19), k = v108, !v58) && v19 != 95
                || v136 == 95
                || (v57 = 134, v116) )
              {
                v57 = 133;
              }
LABEL_122:
              i = (char *)v9;
              k = sub_CBAFE0(v18, v9, v7, k, v57, k);
LABEL_35:
              v19 = v136;
              goto LABEL_36;
            }
          }
          if ( v19 == 128 )
            goto LABEL_35;
          v23 = (unsigned __int8)v19;
LABEL_32:
          v114 = k;
          v22 = isalnum(v23);
          k = v114;
          goto LABEL_33;
        }
        goto LABEL_24;
      }
      if ( v136 == 10 )
      {
LABEL_129:
        if ( (*(_BYTE *)(v18 + 40) & 8) == 0 )
        {
          v136 = 10;
          v21 = 0;
          goto LABEL_24;
        }
        v136 = 10;
        v56 = *(_DWORD *)(v18 + 80);
        goto LABEL_119;
      }
      if ( v136 != 128 || (v107 & 2) != 0 )
        goto LABEL_35;
LABEL_118:
      v136 = 128;
      v56 = *(_DWORD *)(v18 + 80);
LABEL_119:
      v21 = 130;
      if ( v56 > 0 )
        goto LABEL_120;
LABEL_24:
      if ( v19 == 128 )
        goto LABEL_35;
LABEL_25:
      v112 = k;
      v22 = isalnum((unsigned __int8)v19);
      k = v112;
      if ( v22 )
        goto LABEL_121;
      if ( v19 != 95 )
      {
        v23 = (unsigned __int8)v19;
        if ( v136 != 128 )
          goto LABEL_28;
        goto LABEL_32;
      }
LABEL_33:
      if ( !v22 && v19 != 95 )
        goto LABEL_35;
LABEL_121:
      v57 = 134;
      if ( v21 == 130 )
        goto LABEL_122;
      v19 = 128;
      if ( v136 != 128 )
      {
        v117 = k;
        v59 = isalnum((unsigned __int8)v136);
        k = v117;
        if ( !v59 )
        {
          v57 = 134;
          if ( v136 != 95 )
            goto LABEL_122;
        }
        goto LABEL_35;
      }
LABEL_36:
      if ( (k & (1LL << v7)) != 0 )
        break;
      if ( v121 == s1 )
      {
        _libc_free(v105, i);
        _libc_free(v151, i);
        return 1;
      }
      ++s1;
    }
    v150 = (char *)dest;
    if ( v109 || *(_DWORD *)(v111 + 120) )
    {
      s1b = (_QWORD *)v18;
      for ( m = dest; ; v150 = (char *)m )
      {
        i = (char *)m;
        v139 = m++;
        v61 = sub_CBB820((__int64)&v144, i, v121, v9, v7);
        if ( v61 )
          break;
      }
      v66 = (unsigned __int8 *)v61;
      if ( v109 == (char *)1 && !*(_DWORD *)(v111 + 120) )
      {
        v80 = (char *)(v61 - v147);
        a4->m128i_i64[0] = v139 - (unsigned __int8 *)v147;
        a4->m128i_i64[1] = (__int64)v80;
        break;
      }
      v67 = s1b[14];
      if ( !v105 )
      {
        v142 = (unsigned __int8 *)v61;
        v83 = malloc(16 * (v67 + 1), i, v62, v63, v64, v65);
        v66 = v142;
        v146 = v83;
        if ( !v83 )
          return 12;
      }
      v68 = 1;
      if ( v67 )
      {
        do
        {
          v69 = v68++;
          v69 *= 16LL;
          *(_QWORD *)(v146 + v69 + 8) = -1;
          v63 = v146;
          *(_QWORD *)(v146 + v69) = -1;
          v62 = v144;
        }
        while ( v68 <= *(_QWORD *)(v144 + 112) );
        LOWORD(v107) = v145;
      }
      if ( *(_DWORD *)(v111 + 120) | v107 & 0x400 )
      {
        v70 = *(_QWORD *)(v111 + 128);
        if ( v70 > 0 && !v151 )
        {
          v141 = v66;
          v82 = malloc(8 * v70 + 8, i, v62, v63, v64, v65);
          v66 = v141;
          v151 = v82;
          if ( !v82 )
          {
            _libc_free(v146, i);
            return 12;
          }
        }
        v140 = v66;
        v71 = sub_CBC140(&v144, (unsigned __int8 *)v150, v66, v9, v7, 0, 0);
        v72 = (unsigned __int64)v140;
      }
      else
      {
        v143 = v66;
        v71 = (__int64)sub_CBBCD0(&v144, v150, (char *)v66, v9, v7);
        v72 = (unsigned __int64)v143;
      }
      if ( !v71 )
      {
        v73 = v72;
        while ( 1 )
        {
          v74 = (unsigned __int8 *)v150;
          if ( (unsigned __int64)v150 >= v73 )
            break;
          v75 = (unsigned __int8 *)sub_CBB820((__int64)&v144, v150, (char *)(v73 - 1), v9, v7);
          v73 = (unsigned __int64)v75;
          if ( !v75 )
            break;
          if ( sub_CBC140(&v144, v74, v75, v9, v7, 0, 0) )
          {
            v72 = v73;
            goto LABEL_162;
          }
        }
        v72 = v73;
        if ( v121 != (char *)v74 )
        {
          v19 = 128;
          s1 = (char *)(v74 + 1);
          if ( v74 + 1 != (unsigned __int8 *)v148 )
            v19 = (char)*v74;
          v18 = v144;
          v105 = v146;
          v119 = v149;
          LOWORD(v107) = v145;
          continue;
        }
      }
LABEL_162:
      v76 = v146;
      i = v109;
      v105 = v146;
      if ( v109 )
      {
        a4->m128i_i64[0] = v150 - v147;
        a4->m128i_i64[1] = v72 - (_QWORD)v147;
        if ( v109 != (char *)1 )
        {
          v77 = v144;
          v78 = 1;
          v79 = a4 + 1;
          do
          {
            if ( *(_QWORD *)(v77 + 112) >= v78 )
            {
              *v79 = _mm_loadu_si128((const __m128i *)(v76 + 16 * v78));
            }
            else
            {
              v79->m128i_i64[0] = -1;
              v79->m128i_i64[1] = -1;
            }
            ++v78;
            ++v79;
          }
          while ( v109 != (char *)v78 );
        }
      }
    }
    break;
  }
  if ( v105 )
    _libc_free(v105, i);
  v81 = v151;
  if ( v151 )
LABEL_174:
    _libc_free(v81, i);
  return v106;
}
