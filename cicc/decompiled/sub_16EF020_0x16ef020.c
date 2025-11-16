// Function: sub_16EF020
// Address: 0x16ef020
//
__int64 __fastcall sub_16EF020(__int64 a1, const char *a2, __int64 a3, __m128i *a4, char a5)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  int v8; // eax
  bool v9; // zf
  __int64 v10; // rax
  char *v11; // rbx
  char *v12; // rcx
  const void *v13; // r14
  char v14; // r15
  signed __int64 v15; // rdx
  __int64 v16; // r12
  int v17; // ebx
  unsigned __int64 k; // rcx
  int v19; // r13d
  int v20; // eax
  int v21; // edi
  int v22; // eax
  unsigned __int8 *v23; // rsi
  __int64 v25; // rax
  const char *v26; // rbx
  char *v27; // rcx
  const void *v28; // r14
  char v29; // r15
  signed __int64 v30; // rdx
  size_t v31; // r13
  __int64 v32; // rax
  void *v33; // r12
  __int64 v34; // rax
  void *v35; // rcx
  _BYTE *v36; // rax
  char *v37; // r12
  int v38; // r8d
  int v39; // r13d
  int v40; // eax
  int v41; // r13d
  char *v42; // rax
  int v43; // ebx
  unsigned __int8 *v44; // rax
  int v45; // r13d
  __int64 v46; // rcx
  __int64 v47; // rdi
  int v48; // r12d
  __int64 v49; // rax
  int v50; // r10d
  int v51; // ebx
  int v52; // r10d
  int v53; // r8d
  int v54; // eax
  int v55; // eax
  unsigned __int8 *m; // r12
  char *v57; // rsi
  char *v58; // rax
  unsigned __int8 *v59; // r10
  __int64 v60; // r12
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned __int64 v65; // r10
  unsigned __int64 v66; // r13
  unsigned __int8 *v67; // r12
  unsigned __int8 *v68; // rax
  unsigned __int64 v69; // r8
  __int64 v70; // rcx
  unsigned __int64 v71; // rax
  __m128i *v72; // rdx
  char *v73; // rax
  unsigned __int64 v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rax
  char *i; // rsi
  char *v78; // rax
  char *v79; // r12
  __int64 v80; // r13
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  char *v84; // rsi
  char *v85; // rax
  __int64 j; // rax
  unsigned __int64 v87; // rdi
  __int64 v88; // rcx
  unsigned __int64 v89; // rax
  __m128i *v90; // rdx
  int v91; // r13d
  int v92; // eax
  __int64 v93; // rax
  unsigned __int64 v95; // [rsp+10h] [rbp-110h]
  unsigned int v96; // [rsp+18h] [rbp-108h]
  int v97; // [rsp+24h] [rbp-FCh]
  unsigned __int64 v98; // [rsp+28h] [rbp-F8h]
  __int64 v99; // [rsp+30h] [rbp-F0h]
  __int64 v100; // [rsp+30h] [rbp-F0h]
  __int64 v101; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v102; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v103; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v104; // [rsp+40h] [rbp-E0h]
  int v105; // [rsp+40h] [rbp-E0h]
  int v106; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v107; // [rsp+40h] [rbp-E0h]
  char *v108; // [rsp+48h] [rbp-D8h]
  char *v109; // [rsp+50h] [rbp-D0h]
  char *v110; // [rsp+50h] [rbp-D0h]
  char *v111; // [rsp+58h] [rbp-C8h]
  int v112; // [rsp+58h] [rbp-C8h]
  int v113; // [rsp+58h] [rbp-C8h]
  int v114; // [rsp+58h] [rbp-C8h]
  _DWORD *v115; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *dest; // [rsp+60h] [rbp-C0h]
  void *desta; // [rsp+60h] [rbp-C0h]
  __int64 v118; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v119; // [rsp+68h] [rbp-B8h]
  __int64 v120; // [rsp+68h] [rbp-B8h]
  unsigned __int8 *v121; // [rsp+68h] [rbp-B8h]
  char *s1; // [rsp+70h] [rbp-B0h]
  const char *s1a; // [rsp+70h] [rbp-B0h]
  _QWORD *s1b; // [rsp+70h] [rbp-B0h]
  __int64 v125; // [rsp+78h] [rbp-A8h]
  int v126; // [rsp+78h] [rbp-A8h]
  __int64 v127; // [rsp+78h] [rbp-A8h]
  int v128; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v129; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v130; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v131; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v132; // [rsp+78h] [rbp-A8h]
  unsigned __int8 *v133; // [rsp+78h] [rbp-A8h]
  __int64 v134; // [rsp+80h] [rbp-A0h] BYREF
  int v135; // [rsp+88h] [rbp-98h]
  unsigned __int64 v136; // [rsp+90h] [rbp-90h]
  const char *v137; // [rsp+98h] [rbp-88h]
  char *v138; // [rsp+A0h] [rbp-80h]
  char *v139; // [rsp+A8h] [rbp-78h]
  char *v140; // [rsp+B0h] [rbp-70h]
  unsigned __int64 v141; // [rsp+B8h] [rbp-68h]
  __int64 v142; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v143; // [rsp+C8h] [rbp-58h]
  void *v144; // [rsp+D0h] [rbp-50h]
  void *v145; // [rsp+D8h] [rbp-48h]
  char *v146; // [rsp+E0h] [rbp-40h]
  char *v147; // [rsp+E8h] [rbp-38h]

  v96 = 2;
  if ( *(_DWORD *)a1 != 62053 )
    return v96;
  v5 = *(_QWORD *)(a1 + 24);
  v101 = v5;
  if ( *(_DWORD *)v5 != 53829 )
    return v96;
  v96 = *(_DWORD *)(v5 + 72) & 4;
  if ( v96 )
    return 2;
  v6 = *(_QWORD *)(v5 + 64);
  v97 = a5 & 7;
  v7 = *(_QWORD *)(v5 + 56) + 1LL;
  v8 = *(_DWORD *)(v5 + 40) & 4;
  if ( *(__int64 *)(*(_QWORD *)(a1 + 24) + 48LL) > 64 )
  {
    v9 = v8 == 0;
    v25 = 0;
    if ( v9 )
      v25 = a3;
    v100 = v25;
    if ( (a5 & 4) != 0 )
    {
      s1a = &a2[a4->m128i_i64[0]];
      v108 = (char *)&a2[a4->m128i_i64[1]];
    }
    else
    {
      s1a = a2;
      v108 = (char *)&a2[strlen(a2)];
    }
    v26 = s1a;
    if ( s1a > v108 )
      return 16;
    v27 = *(char **)(v101 + 96);
    if ( v27 )
    {
      if ( s1a >= v108 )
      {
        v26 = s1a;
      }
      else
      {
        v127 = v7;
        v120 = v6;
        v28 = *(const void **)(v101 + 96);
        v29 = *v27;
        while ( 1 )
        {
          if ( *v26 == v29 )
          {
            v30 = *(int *)(v101 + 104);
            if ( v108 - v26 >= v30 && !memcmp(v26, v28, v30) )
              break;
          }
          if ( v108 == ++v26 )
            return 1;
        }
        v7 = v127;
        v6 = v120;
      }
      if ( v108 == v26 )
        return 1;
    }
    v137 = a2;
    v136 = 0;
    v135 = v97;
    v134 = v101;
    v138 = (char *)s1a;
    v141 = 0;
    v139 = v108;
    v31 = *(_QWORD *)(v101 + 48);
    v32 = malloc(4 * v31);
    v143 = v32;
    v33 = (void *)v32;
    if ( !v32 )
      return 12;
    v144 = (void *)v32;
    desta = (void *)(v31 + v32);
    v145 = (void *)(v31 + v32);
    v110 = (char *)(v31 + v31 + v32);
    v146 = v110;
    v142 = 4;
    v147 = &v110[v31];
    memset(&v110[v31], 0, v31);
    v34 = v101;
    v35 = v33;
    v128 = 128;
LABEL_66:
    v36 = memset(v35, 0, *(_QWORD *)(v34 + 48));
    v36[v7] = 1;
    v37 = (char *)sub_16ECBD0(v134, v7, v6, (__int64)v36, 132, (__int64)v36);
    memmove(desta, v37, *(_QWORD *)(v134 + 48));
    v121 = 0;
    while ( 1 )
    {
      v43 = 128;
      if ( v139 != s1a )
        v43 = *s1a;
      v115 = (_DWORD *)v134;
      v9 = memcmp(v37, desta, *(_QWORD *)(v134 + 48)) == 0;
      v44 = v121;
      if ( v9 )
        v44 = (unsigned __int8 *)s1a;
      v121 = v44;
      if ( v128 == 10 )
        break;
      if ( v128 != 128 )
      {
        if ( v43 == 10 )
        {
          v38 = v115[10] & 8;
          if ( v38 )
            goto LABEL_94;
        }
        else
        {
          if ( v43 == 128 && (v135 & 2) == 0 )
            goto LABEL_94;
          v38 = 0;
        }
        goto LABEL_71;
      }
      if ( (v135 & 1) == 0 )
        goto LABEL_191;
      if ( v43 == 10 )
      {
        if ( (v115[10] & 8) == 0 )
          goto LABEL_83;
LABEL_94:
        v45 = v115[20];
        v38 = 130;
        if ( v45 > 0 )
        {
LABEL_95:
          v46 = (__int64)v37;
          v47 = (__int64)v115;
          v48 = v38;
          goto LABEL_97;
        }
LABEL_71:
        if ( v128 == 128 )
          goto LABEL_83;
LABEL_72:
        v39 = (unsigned __int8)v128;
        v112 = v38;
        v40 = isalnum((unsigned __int8)v128);
        v38 = v112;
        if ( !v40 )
        {
          if ( v128 != 95 )
          {
            if ( v43 != 128 )
              goto LABEL_75;
LABEL_80:
            v114 = v38;
            v40 = isalnum(v39);
            v38 = v114;
          }
          if ( !v40 && v128 != 95 )
            goto LABEL_83;
        }
        v91 = 134;
        if ( v38 != 130 )
        {
          if ( v43 == 128 || (v92 = isalnum((unsigned __int8)v43), v43 == 95) || v92 )
          {
LABEL_83:
            v42 = &v37[v6];
            if ( v37[v6] )
              goto LABEL_200;
            goto LABEL_84;
          }
        }
LABEL_226:
        v37 = (char *)sub_16ECBD0(v134, v7, v6, (__int64)v37, v91, (__int64)v37);
        goto LABEL_83;
      }
      if ( v43 != 128 )
        goto LABEL_83;
      if ( (v135 & 2) == 0 )
        goto LABEL_94;
      v42 = &v37[v6];
      if ( v37[v6] )
      {
LABEL_200:
        v140 = (char *)v121;
        if ( !*v42 )
        {
          _libc_free(v136);
          _libc_free(v141);
          _libc_free(v143);
          return 1;
        }
        if ( !v100 && !*(_DWORD *)(v101 + 120) )
        {
          v87 = v136;
          goto LABEL_258;
        }
        for ( i = (char *)v121; ; i = ++v140 )
        {
          v78 = sub_16EE780(&v134, i, v108, v7, v6);
          if ( v78 )
            break;
        }
        v79 = v78;
        if ( v100 == 1 && !*(_DWORD *)(v101 + 120) )
        {
          v87 = v136;
          a4->m128i_i64[0] = v140 - v137;
          a4->m128i_i64[1] = v78 - v137;
LABEL_258:
          if ( v87 )
            _libc_free(v87);
          if ( v141 )
            _libc_free(v141);
          v74 = v143;
          goto LABEL_174;
        }
        v80 = *(_QWORD *)(v134 + 112);
        if ( !v136 )
        {
          v136 = malloc(16 * (v80 + 1));
          if ( !v136 )
          {
LABEL_265:
            _libc_free(v143);
            return 12;
          }
        }
        v81 = 1;
        if ( v80 )
        {
          do
          {
            v82 = v81++;
            v82 *= 16LL;
            *(_QWORD *)(v136 + v82 + 8) = -1;
            *(_QWORD *)(v136 + v82) = -1;
          }
          while ( v81 <= *(_QWORD *)(v134 + 112) );
        }
        if ( *(_DWORD *)(v101 + 120) | v135 & 0x400 )
        {
          v83 = *(_QWORD *)(v101 + 128);
          if ( v83 <= 0 )
            goto LABEL_216;
          if ( v141 )
            goto LABEL_216;
          v93 = malloc(8 * v83 + 8);
          v141 = v93;
          if ( v93 )
            goto LABEL_216;
          _libc_free(v136);
          goto LABEL_265;
        }
        for ( j = (__int64)sub_16EEC10(&v134, v140, v79, v7, v6);
              ;
              j = sub_16EDFF0(&v134, (unsigned __int8 *)v140, (unsigned __int8 *)v79, v7, v6, 0, 0) )
        {
          if ( j )
            goto LABEL_218;
          v84 = v140;
          if ( v140 >= v79 )
            break;
          v85 = sub_16EE780(&v134, v140, v79 - 1, v7, v6);
          v84 = v140;
          v79 = v85;
          if ( !v85 )
            break;
LABEL_216:
          ;
        }
        if ( v108 == v84 )
        {
LABEL_218:
          v87 = v136;
          if ( v100 )
          {
            a4->m128i_i64[0] = v140 - v137;
            a4->m128i_i64[1] = v79 - v137;
            if ( v100 != 1 )
            {
              v88 = v134;
              v89 = 1;
              v90 = a4 + 1;
              do
              {
                if ( *(_QWORD *)(v88 + 112) >= v89 )
                {
                  *v90 = _mm_loadu_si128((const __m128i *)(v87 + 16 * v89));
                }
                else
                {
                  v90->m128i_i64[0] = -1;
                  v90->m128i_i64[1] = -1;
                }
                ++v89;
                ++v90;
              }
              while ( v100 != v89 );
            }
          }
          goto LABEL_258;
        }
        v35 = v144;
        s1a = v84 + 1;
        desta = v145;
        v110 = v146;
        if ( v84 + 1 == v138 )
          v128 = 128;
        else
          v128 = *v84;
        v34 = v134;
        goto LABEL_66;
      }
LABEL_84:
      if ( v108 == s1a )
        goto LABEL_200;
      memmove(v110, v37, *(_QWORD *)(v134 + 48));
      memmove(v37, desta, *(_QWORD *)(v134 + 48));
      ++s1a;
      v128 = v43;
      v37 = (char *)sub_16ECBD0(v134, v7, v6, (__int64)v110, v43, (__int64)v37);
    }
    v38 = v115[10] & 8;
    if ( !v38 )
    {
      if ( v43 == 128 && (v135 & 2) == 0 )
        goto LABEL_94;
      goto LABEL_72;
    }
LABEL_191:
    v45 = v115[19];
    if ( v43 == 10 )
    {
      if ( (v115[10] & 8) == 0 )
      {
        v38 = 129;
        if ( v45 > 0 )
          goto LABEL_95;
LABEL_75:
        v113 = v38;
        v41 = isalnum((unsigned __int8)v43);
        if ( v41 || v43 == 95 )
        {
          if ( v128 == 128 || !isalnum((unsigned __int8)v128) && v128 != 95 || v43 == 95 || v41 )
            v91 = 133;
          else
            v91 = 134;
          goto LABEL_226;
        }
        v38 = v113;
LABEL_78:
        if ( v128 == 128 )
          goto LABEL_83;
        v39 = (unsigned __int8)v128;
        goto LABEL_80;
      }
    }
    else
    {
      if ( v43 != 128 )
      {
        if ( v45 <= 0 )
          goto LABEL_99;
        goto LABEL_194;
      }
      if ( (v135 & 2) != 0 )
      {
        if ( v45 <= 0 )
        {
LABEL_242:
          v38 = 129;
          goto LABEL_78;
        }
LABEL_194:
        v46 = (__int64)v37;
        v47 = (__int64)v115;
        v48 = 129;
LABEL_97:
        while ( 1 )
        {
          v49 = sub_16ECBD0(v47, v7, v6, v46, v48, v46);
          v46 = v49;
          if ( !--v45 )
            break;
          v47 = v134;
        }
        v38 = v48;
        v37 = (char *)v49;
        if ( v38 != 129 )
          goto LABEL_71;
LABEL_99:
        if ( v43 != 128 )
        {
          v38 = 129;
          goto LABEL_75;
        }
        goto LABEL_242;
      }
    }
    v38 = 131;
    v45 += v115[20];
    if ( v45 > 0 )
      goto LABEL_95;
    goto LABEL_71;
  }
  v9 = v8 == 0;
  v10 = 0;
  if ( v9 )
    v10 = a3;
  v99 = v10;
  if ( (a5 & 4) != 0 )
  {
    s1 = (char *)&a2[a4->m128i_i64[0]];
    v111 = (char *)&a2[a4->m128i_i64[1]];
  }
  else
  {
    s1 = (char *)a2;
    v111 = (char *)&a2[strlen(a2)];
  }
  v11 = s1;
  if ( s1 > v111 )
    return 16;
  v12 = *(char **)(v101 + 96);
  if ( v12 )
  {
    if ( s1 >= v111 )
    {
      v11 = s1;
    }
    else
    {
      v125 = v7;
      v118 = v6;
      v13 = *(const void **)(v101 + 96);
      v14 = *v12;
      while ( 1 )
      {
        if ( *v11 == v14 )
        {
          v15 = *(int *)(v101 + 104);
          if ( v111 - v11 >= v15 && !memcmp(v11, v13, v15) )
            break;
        }
        if ( v111 == ++v11 )
          return 1;
      }
      v7 = v125;
      v6 = v118;
    }
    if ( v111 == v11 )
      return 1;
  }
  v136 = 0;
  v141 = 0;
  v135 = v97;
  v134 = v101;
  v138 = s1;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v95 = 0;
  v137 = a2;
  v16 = v101;
  v139 = v111;
  v109 = v111;
  v17 = 128;
  while ( 2 )
  {
    dest = 0;
    v119 = sub_16EC790(v16, v7, v6, 1LL << v7, 132, 1LL << v7);
    for ( k = v119; ; k = sub_16EC790(v16, v7, v6, k, v17, v119) )
    {
      if ( s1 != v109 )
      {
        v126 = *s1;
        v23 = dest;
        if ( v119 == k )
          v23 = (unsigned __int8 *)s1;
        dest = v23;
LABEL_43:
        if ( v17 == 10 )
        {
          if ( (*(_BYTE *)(v16 + 40) & 8) != 0 )
            goto LABEL_106;
          if ( v126 != 128 )
            goto LABEL_46;
          goto LABEL_182;
        }
        if ( v17 != 128 )
        {
          if ( v126 == 10 )
            goto LABEL_129;
          v19 = 0;
          if ( v126 != 128 )
            goto LABEL_24;
LABEL_104:
          if ( (v97 & 2) != 0 )
          {
            v126 = 128;
            v19 = 0;
            goto LABEL_24;
          }
          goto LABEL_118;
        }
        goto LABEL_114;
      }
      v126 = 128;
      if ( v119 != k )
        goto LABEL_43;
      if ( v17 == 10 )
      {
        if ( (*(_BYTE *)(v16 + 40) & 8) == 0 )
        {
          dest = (unsigned __int8 *)s1;
LABEL_182:
          if ( (v97 & 2) != 0 )
          {
LABEL_46:
            v19 = 0;
            goto LABEL_25;
          }
          goto LABEL_118;
        }
        v50 = *(_DWORD *)(v16 + 76);
        dest = (unsigned __int8 *)s1;
        goto LABEL_136;
      }
      dest = (unsigned __int8 *)s1;
      if ( v17 != 128 )
        goto LABEL_104;
LABEL_114:
      if ( (v97 & 1) == 0 )
      {
LABEL_106:
        v50 = *(_DWORD *)(v16 + 76);
        if ( v126 == 10 )
        {
          if ( (*(_BYTE *)(v16 + 40) & 8) == 0 )
          {
            if ( v50 <= 0 )
              goto LABEL_113;
            goto LABEL_109;
          }
          goto LABEL_140;
        }
        if ( v126 != 128 )
        {
          if ( v50 <= 0 )
            goto LABEL_112;
LABEL_109:
          v105 = v17;
          v19 = 129;
          v51 = v50;
          goto LABEL_110;
        }
LABEL_136:
        if ( (v97 & 2) != 0 )
        {
          if ( v50 <= 0 )
            goto LABEL_138;
          goto LABEL_109;
        }
LABEL_140:
        v52 = *(_DWORD *)(v16 + 80) + v50;
        v19 = 131;
        if ( v52 <= 0 )
          goto LABEL_24;
LABEL_120:
        v105 = v17;
        v51 = v52;
        do
        {
LABEL_110:
          k = sub_16EC790(v16, v7, v6, k, v19, k);
          --v51;
        }
        while ( v51 );
        v17 = v105;
        if ( v19 == 129 )
        {
LABEL_112:
          if ( v126 == 128 )
          {
LABEL_138:
            v126 = 128;
            v19 = 129;
          }
          else
          {
LABEL_113:
            v19 = 129;
LABEL_28:
            v103 = k;
            v22 = isalnum((unsigned __int8)v126);
            k = v103;
            if ( v126 == 95 || v22 )
            {
              if ( v17 == 128
                || (v98 = v103, v106 = v22, v54 = isalnum((unsigned __int8)v17), k = v98, !v54) && v17 != 95
                || v126 == 95
                || (v53 = 134, v106) )
              {
                v53 = 133;
              }
LABEL_122:
              k = sub_16EC790(v16, v7, v6, k, v53, k);
LABEL_35:
              v17 = v126;
              goto LABEL_36;
            }
          }
          if ( v17 == 128 )
            goto LABEL_35;
          v21 = (unsigned __int8)v17;
LABEL_32:
          v104 = k;
          v20 = isalnum(v21);
          k = v104;
          goto LABEL_33;
        }
        goto LABEL_24;
      }
      if ( v126 == 10 )
      {
LABEL_129:
        if ( (*(_BYTE *)(v16 + 40) & 8) == 0 )
        {
          v126 = 10;
          v19 = 0;
          goto LABEL_24;
        }
        v126 = 10;
        v52 = *(_DWORD *)(v16 + 80);
        goto LABEL_119;
      }
      if ( v126 != 128 || (v97 & 2) != 0 )
        goto LABEL_35;
LABEL_118:
      v126 = 128;
      v52 = *(_DWORD *)(v16 + 80);
LABEL_119:
      v19 = 130;
      if ( v52 > 0 )
        goto LABEL_120;
LABEL_24:
      if ( v17 == 128 )
        goto LABEL_35;
LABEL_25:
      v102 = k;
      v20 = isalnum((unsigned __int8)v17);
      k = v102;
      if ( v20 )
        goto LABEL_121;
      if ( v17 != 95 )
      {
        v21 = (unsigned __int8)v17;
        if ( v126 != 128 )
          goto LABEL_28;
        goto LABEL_32;
      }
LABEL_33:
      if ( !v20 && v17 != 95 )
        goto LABEL_35;
LABEL_121:
      v53 = 134;
      if ( v19 == 130 )
        goto LABEL_122;
      v17 = 128;
      if ( v126 != 128 )
      {
        v107 = k;
        v55 = isalnum((unsigned __int8)v126);
        k = v107;
        if ( !v55 )
        {
          v53 = 134;
          if ( v126 != 95 )
            goto LABEL_122;
        }
        goto LABEL_35;
      }
LABEL_36:
      if ( (k & (1LL << v6)) != 0 )
        break;
      if ( v111 == s1 )
      {
        _libc_free(v95);
        _libc_free(v141);
        return 1;
      }
      ++s1;
    }
    v140 = (char *)dest;
    if ( v99 || *(_DWORD *)(v101 + 120) )
    {
      s1b = (_QWORD *)v16;
      for ( m = dest; ; v140 = (char *)m )
      {
        v57 = (char *)m;
        v129 = m++;
        v58 = sub_16ECFD0((__int64)&v134, v57, v111, v7, v6);
        if ( v58 )
          break;
      }
      v59 = (unsigned __int8 *)v58;
      if ( v99 == 1 && !*(_DWORD *)(v101 + 120) )
      {
        v73 = (char *)(v58 - v137);
        a4->m128i_i64[0] = v129 - (unsigned __int8 *)v137;
        a4->m128i_i64[1] = (__int64)v73;
        break;
      }
      v60 = s1b[14];
      if ( !v95 )
      {
        v132 = (unsigned __int8 *)v58;
        v76 = malloc(16 * (v60 + 1));
        v59 = v132;
        v136 = v76;
        if ( !v76 )
          return 12;
      }
      v61 = 1;
      if ( v60 )
      {
        do
        {
          v62 = v61++;
          v62 *= 16LL;
          *(_QWORD *)(v136 + v62 + 8) = -1;
          *(_QWORD *)(v136 + v62) = -1;
        }
        while ( v61 <= *(_QWORD *)(v134 + 112) );
        LOWORD(v97) = v135;
      }
      if ( *(_DWORD *)(v101 + 120) | v97 & 0x400 )
      {
        v63 = *(_QWORD *)(v101 + 128);
        if ( v63 > 0 && !v141 )
        {
          v131 = v59;
          v75 = malloc(8 * v63 + 8);
          v59 = v131;
          v141 = v75;
          if ( !v75 )
          {
            _libc_free(v136);
            return 12;
          }
        }
        v130 = v59;
        v64 = sub_16ED860(&v134, (unsigned __int8 *)v140, v59, v7, v6, 0, 0);
        v65 = (unsigned __int64)v130;
      }
      else
      {
        v133 = v59;
        v64 = (__int64)sub_16ED430(&v134, v140, (char *)v59, v7, v6);
        v65 = (unsigned __int64)v133;
      }
      if ( !v64 )
      {
        v66 = v65;
        while ( 1 )
        {
          v67 = (unsigned __int8 *)v140;
          if ( (unsigned __int64)v140 >= v66 )
            break;
          v68 = (unsigned __int8 *)sub_16ECFD0((__int64)&v134, v140, (char *)(v66 - 1), v7, v6);
          v66 = (unsigned __int64)v68;
          if ( !v68 )
            break;
          if ( sub_16ED860(&v134, v67, v68, v7, v6, 0, 0) )
          {
            v65 = v66;
            goto LABEL_162;
          }
        }
        v65 = v66;
        if ( v111 != (char *)v67 )
        {
          v17 = 128;
          s1 = (char *)(v67 + 1);
          if ( v67 + 1 != (unsigned __int8 *)v138 )
            v17 = (char)*v67;
          v16 = v134;
          v95 = v136;
          v109 = v139;
          LOWORD(v97) = v135;
          continue;
        }
      }
LABEL_162:
      v69 = v136;
      v95 = v136;
      if ( v99 )
      {
        a4->m128i_i64[0] = v140 - v137;
        a4->m128i_i64[1] = v65 - (_QWORD)v137;
        if ( v99 != 1 )
        {
          v70 = v134;
          v71 = 1;
          v72 = a4 + 1;
          do
          {
            if ( *(_QWORD *)(v70 + 112) >= v71 )
            {
              *v72 = _mm_loadu_si128((const __m128i *)(v69 + 16 * v71));
            }
            else
            {
              v72->m128i_i64[0] = -1;
              v72->m128i_i64[1] = -1;
            }
            ++v71;
            ++v72;
          }
          while ( v99 != v71 );
        }
      }
    }
    break;
  }
  if ( v95 )
    _libc_free(v95);
  v74 = v141;
  if ( v141 )
LABEL_174:
    _libc_free(v74);
  return v96;
}
