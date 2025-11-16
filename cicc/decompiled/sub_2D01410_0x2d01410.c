// Function: sub_2D01410
// Address: 0x2d01410
//
__int64 __fastcall sub_2D01410(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int16 v16; // ax
  const __m128i *v17; // r15
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  bool v20; // r8
  __int64 v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  char v24; // di
  unsigned __int64 v25; // r8
  int v26; // eax
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // r13
  unsigned __int16 v30; // ax
  int v31; // edx
  unsigned __int64 v32; // rsi
  __int64 v33; // r15
  __int64 j; // r15
  char v35; // r13
  __int64 v36; // rax
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rdi
  _QWORD *v41; // r15
  __int64 v42; // r14
  char *v43; // r15
  char v44; // r9
  const __m128i *v45; // rax
  const __m128i *v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rdx
  _QWORD *v49; // r9
  _QWORD *k; // r15
  unsigned int v51; // r14d
  int v53; // eax
  int v54; // edx
  _QWORD *v55; // r13
  char v56; // al
  _QWORD *v57; // r12
  __int64 v58; // rax
  int v59; // eax
  unsigned __int64 v60; // rbx
  unsigned __int64 v61; // rdx
  __int64 v62; // rax
  unsigned __int16 v63; // ax
  __int64 *v64; // rax
  __int64 *v65; // rax
  __int64 v66; // rax
  __int64 v67; // rbx
  __int64 *v68; // rax
  _QWORD *v69; // r14
  __int64 i; // rbx
  char v71; // al
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  unsigned __int64 v74; // rcx
  __int64 v75; // rax
  int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  unsigned __int16 v80; // ax
  __int64 v81; // rdx
  unsigned __int8 *v82; // rax
  __int64 v83; // rax
  char v84; // r14
  __int64 *v85; // rax
  __int64 *v86; // rax
  __int64 v87; // rax
  __int64 v88; // r14
  __int64 *v89; // rax
  __int64 v90; // r8
  unsigned __int16 v91; // ax
  unsigned __int64 v92; // rdx
  char v93; // r14
  __int64 *v94; // rax
  __int64 *v95; // rax
  __int64 v96; // rax
  __int64 *v97; // rax
  __int64 v98; // [rsp+8h] [rbp-78h]
  unsigned __int64 v99; // [rsp+8h] [rbp-78h]
  __int64 v100; // [rsp+10h] [rbp-70h]
  _QWORD *v102; // [rsp+18h] [rbp-68h]
  __int64 v103; // [rsp+18h] [rbp-68h]
  const __m128i *v104; // [rsp+20h] [rbp-60h]
  unsigned __int64 v105; // [rsp+20h] [rbp-60h]
  _QWORD *v106; // [rsp+28h] [rbp-58h]
  __int64 v107; // [rsp+28h] [rbp-58h]
  _BYTE *v108; // [rsp+28h] [rbp-58h]
  __int64 *v109; // [rsp+28h] [rbp-58h]
  __int64 v110; // [rsp+28h] [rbp-58h]
  __int64 v111; // [rsp+30h] [rbp-50h]
  unsigned __int64 v112; // [rsp+30h] [rbp-50h]
  char v113; // [rsp+30h] [rbp-50h]
  __int64 v114; // [rsp+30h] [rbp-50h]
  _QWORD *v115; // [rsp+30h] [rbp-50h]
  char v116; // [rsp+30h] [rbp-50h]
  bool v117; // [rsp+30h] [rbp-50h]
  __int64 v118; // [rsp+30h] [rbp-50h]
  unsigned __int64 v119; // [rsp+30h] [rbp-50h]
  __int64 v120; // [rsp+38h] [rbp-48h]
  unsigned int *v121; // [rsp+38h] [rbp-48h]
  unsigned __int64 v122; // [rsp+40h] [rbp-40h] BYREF
  int v123; // [rsp+48h] [rbp-38h]

  v3 = (__int64)a1;
  v4 = a1 + 8;
  v5 = (__int64)(a1 + 7);
  v100 = a2;
  v98 = (__int64)(a1 + 1);
  sub_2D00360(a1[3]);
  a1[3] = 0;
  v6 = a1[27];
  *(_QWORD *)(v3 + 32) = v3 + 16;
  *(_QWORD *)(v3 + 40) = v3 + 16;
  *(_QWORD *)(v3 + 48) = 0;
  v106 = (_QWORD *)(v3 + 16);
  sub_2CFFFC0(v6);
  v7 = *(_QWORD *)(v3 + 120);
  *(_QWORD *)(v3 + 216) = 0;
  *(_QWORD *)(v3 + 224) = v3 + 208;
  *(_QWORD *)(v3 + 232) = v3 + 208;
  *(_QWORD *)(v3 + 240) = 0;
  sub_2D00190(v7);
  v8 = *(_QWORD *)(v3 + 72);
  *(_QWORD *)(v3 + 120) = 0;
  *(_QWORD *)(v3 + 128) = v3 + 112;
  *(_QWORD *)(v3 + 136) = v3 + 112;
  *(_QWORD *)(v3 + 144) = 0;
  v120 = v3 + 112;
  sub_2D00190(v8);
  *(_QWORD *)(v3 + 80) = v4;
  v9 = *(_QWORD *)(v3 + 168);
  *(_QWORD *)(v3 + 72) = 0;
  *(_QWORD *)(v3 + 88) = v4;
  *(_QWORD *)(v3 + 96) = 0;
  v111 = v3 + 152;
  v10 = *(_QWORD *)(a2 + 40) + 8LL;
  sub_2D00360(v9);
  *(_QWORD *)(v3 + 168) = 0;
  *(_QWORD *)(v3 + 176) = v3 + 160;
  *(_QWORD *)(v3 + 184) = v3 + 160;
  *(_QWORD *)(v3 + 192) = 0;
  v104 = (const __m128i *)(v3 + 160);
  if ( *(_QWORD *)(v10 + 8) == v10 )
  {
    *(_QWORD *)(v3 + 248) = *(_QWORD *)(a2 + 40) + 312LL;
    *(_QWORD *)(v3 + 256) = a3;
  }
  else
  {
    v13 = v3;
    v14 = *(_QWORD *)(v10 + 8);
    do
    {
      if ( !v14 )
        BUG();
      LODWORD(v15) = 0;
      v16 = (*(_WORD *)(v14 - 22) >> 1) & 0x3F;
      if ( v16 )
        v15 = 1LL << ((unsigned __int8)v16 - 1);
      v122 = v14 - 56;
      a2 = (__int64)&v122;
      v123 = v15;
      sub_2D00A20(v111, &v122);
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( v14 != v10 );
    v3 = v13;
    v17 = *(const __m128i **)(v13 + 176);
    *(_QWORD *)(v13 + 248) = *(_QWORD *)(v100 + 40) + 312LL;
    for ( *(_QWORD *)(v13 + 256) = a3; v17 != v104; v17 = (const __m128i *)sub_220EEE0((__int64)v17) )
    {
      v25 = v17[2].m128i_u64[0];
      v26 = *(_DWORD *)v13;
      v122 = v25;
      if ( v17[2].m128i_i32[2] == v26 )
      {
        a2 = v25;
        sub_2D00AD0((_QWORD *)v13, v25, *(_DWORD *)(v13 + 4));
      }
      else
      {
        v112 = v25;
        v18 = sub_23FDE00(v5, &v122);
        if ( v19 )
        {
          v20 = v18 || v4 == v19 || v112 < v19[4];
          v102 = v19;
          v113 = v20;
          v21 = sub_22077B0(0x28u);
          *(_QWORD *)(v21 + 32) = v122;
          sub_220F040(v113, v21, v102, v4);
          ++*(_QWORD *)(v13 + 96);
        }
        v114 = sub_22077B0(0x30u);
        *(__m128i *)(v114 + 32) = _mm_loadu_si128(v17 + 2);
        v22 = sub_2CBB810(v98, (unsigned __int64 *)(v114 + 32));
        if ( v23 )
        {
          v24 = v22 || v106 == v23 || *(_QWORD *)(v114 + 32) < v23[4];
          a2 = v114;
          sub_220F040(v24, v114, v23, v106);
          ++*(_QWORD *)(v13 + 48);
        }
        else
        {
          a2 = 48;
          j_j___libc_free_0(v114);
        }
      }
    }
  }
  if ( (*(_BYTE *)(v100 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v100, a2, v11, v12);
    v27 = *(_QWORD *)(v100 + 96);
    if ( (*(_BYTE *)(v100 + 2) & 1) != 0 )
      sub_B2C6D0(v100, a2, v78, v79);
    v28 = *(_QWORD *)(v100 + 96);
  }
  else
  {
    v27 = *(_QWORD *)(v100 + 96);
    v28 = v27;
  }
  v29 = v28 + 40LL * *(_QWORD *)(v100 + 104);
  while ( v27 != v29 )
  {
    while ( *(_BYTE *)(*(_QWORD *)(v27 + 8) + 8LL) != 14 )
    {
      v27 += 40;
      if ( v27 == v29 )
        goto LABEL_34;
    }
    v30 = sub_B2BD00(v27);
    v31 = 1;
    if ( HIBYTE(v30) )
    {
      v31 = 1LL << v30;
      if ( !v31 )
        v31 = *(_DWORD *)(v3 + 4);
    }
    v32 = v27;
    v27 += 40;
    sub_2D00AD0((_QWORD *)v3, v32, v31);
  }
LABEL_34:
  v33 = *(_QWORD *)(v100 + 80);
  v103 = v100 + 72;
  if ( v100 + 72 == v33 )
    goto LABEL_40;
  if ( !v33 )
    BUG();
  while ( *(_QWORD *)(v33 + 32) == v33 + 24 )
  {
    v33 = *(_QWORD *)(v33 + 8);
    if ( v103 == v33 )
      goto LABEL_40;
    if ( !v33 )
      BUG();
  }
  v118 = v5;
  v69 = (_QWORD *)v3;
  i = *(_QWORD *)(v33 + 32);
  while ( v33 != v103 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(*(_QWORD *)(i - 16) + 8LL) == 14 )
    {
      LODWORD(v73) = *(_DWORD *)v69;
      if ( *(_BYTE *)(i - 24) == 60 )
      {
        _BitScanReverse64(&v74, 1LL << *(_WORD *)(i - 22));
        v73 = 0x8000000000000000LL >> ((unsigned __int8)v74 ^ 0x3Fu);
      }
      sub_2D00AD0(v69, i - 24, v73);
      v71 = *(_BYTE *)(i - 24);
      if ( v71 != 85 )
        goto LABEL_136;
    }
    else
    {
      v71 = *(_BYTE *)(i - 24);
      if ( v71 != 85 )
        goto LABEL_136;
    }
    v75 = *(_QWORD *)(i - 56);
    if ( !v75 || *(_BYTE *)v75 || *(_QWORD *)(v75 + 24) != *(_QWORD *)(i + 56) || (*(_BYTE *)(v75 + 33) & 0x20) == 0 )
      goto LABEL_138;
    v76 = *(_DWORD *)(v75 + 36);
    if ( v76 == 243 )
    {
      sub_2D00AD0(v69, *(_QWORD *)(i - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF) - 24), *(_DWORD *)v69);
      v71 = *(_BYTE *)(i - 24);
    }
    else
    {
      if ( v76 != 241 && v76 != 238 )
        goto LABEL_138;
      v77 = *(_DWORD *)(i - 20) & 0x7FFFFFF;
      v99 = *(_QWORD *)(i + 32 * (1 - v77) - 24);
      sub_2D00AD0(v69, *(_QWORD *)(i - 32 * v77 - 24), *(_DWORD *)v69);
      sub_2D00AD0(v69, v99, *(_DWORD *)v69);
      v71 = *(_BYTE *)(i - 24);
    }
LABEL_136:
    if ( v71 == 61 || v71 == 62 )
      sub_2D00AD0(v69, *(_QWORD *)(i - 56), *(_DWORD *)v69);
LABEL_138:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v33 + 32) )
    {
      v72 = v33 - 24;
      if ( !v33 )
        v72 = 0;
      if ( i != v72 + 48 )
        break;
      v33 = *(_QWORD *)(v33 + 8);
      if ( v103 == v33 )
        break;
      if ( !v33 )
        BUG();
    }
  }
  v3 = (__int64)v69;
  v5 = v118;
  do
  {
LABEL_40:
    for ( j = *(_QWORD *)(v3 + 128); j != v120; j = sub_220EF30(j) )
    {
      v37 = sub_23FDE00(v5, (unsigned __int64 *)(j + 32));
      if ( v38 )
      {
        v35 = v37 || v4 == v38 || *(_QWORD *)(j + 32) < v38[4];
        v115 = v38;
        v36 = sub_22077B0(0x28u);
        *(_QWORD *)(v36 + 32) = *(_QWORD *)(j + 32);
        sub_220F040(v35, v36, v115, v4);
        ++*(_QWORD *)(v3 + 96);
      }
    }
    v39 = *(_QWORD *)(v3 + 120);
    while ( v39 )
    {
      sub_2D00190(*(_QWORD *)(v39 + 24));
      v40 = v39;
      v39 = *(_QWORD *)(v39 + 16);
      j_j___libc_free_0(v40);
    }
    v41 = *(_QWORD **)(v3 + 80);
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 144) = 0;
    *(_QWORD *)(v3 + 128) = v120;
    *(_QWORD *)(v3 + 136) = v120;
    if ( v41 == v4 )
      break;
    v116 = 0;
    v107 = v5;
    v42 = (__int64)v41;
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                v43 = *(char **)(v42 + 32);
                if ( (unsigned int)sub_2D00C30((unsigned int *)v3, v43) == *(_DWORD *)(v3 + 4) )
                  goto LABEL_65;
                v44 = *v43;
                if ( (unsigned __int8)*v43 <= 0x1Cu )
                  break;
                if ( v44 != 60 )
                {
                  v45 = *(const __m128i **)(v3 + 168);
                  if ( !v45 )
                    goto LABEL_63;
                  goto LABEL_57;
                }
LABEL_65:
                v42 = sub_220EF30(v42);
                if ( (_QWORD *)v42 == v4 )
                  goto LABEL_66;
              }
              if ( v44 == 22 )
                goto LABEL_65;
              v45 = *(const __m128i **)(v3 + 168);
              if ( v45 )
                break;
LABEL_78:
              if ( v44 == 5 )
              {
                v116 |= sub_2D011D0(v3, (unsigned __int64)v43);
                v42 = sub_220EF30(v42);
                if ( (_QWORD *)v42 == v4 )
                  goto LABEL_66;
              }
              else
              {
                v53 = sub_2D00C30((unsigned int *)v3, v43);
                v54 = *(_DWORD *)(v3 + 4);
                if ( v54 == v53 )
                  goto LABEL_65;
                sub_2D00AD0((_QWORD *)v3, (unsigned __int64)v43, v54);
                v116 = 1;
                v42 = sub_220EF30(v42);
                if ( (_QWORD *)v42 == v4 )
                  goto LABEL_66;
              }
            }
LABEL_57:
            v46 = v104;
            do
            {
              while ( 1 )
              {
                v47 = v45[1].m128i_i64[0];
                v48 = v45[1].m128i_i64[1];
                if ( v45[2].m128i_i64[0] >= (unsigned __int64)v43 )
                  break;
                v45 = (const __m128i *)v45[1].m128i_i64[1];
                if ( !v48 )
                  goto LABEL_61;
              }
              v46 = v45;
              v45 = (const __m128i *)v45[1].m128i_i64[0];
            }
            while ( v47 );
LABEL_61:
            if ( v46 != v104 && v46[2].m128i_i64[0] <= (unsigned __int64)v43 )
              goto LABEL_65;
LABEL_63:
            if ( (unsigned __int8)(v44 - 67) <= 0xCu )
            {
              v116 |= sub_2D00DF0((unsigned int *)v3, v43);
              goto LABEL_65;
            }
            if ( v44 != 63 )
              break;
            v116 |= sub_2D010F0(v3, (unsigned __int64)v43);
            v42 = sub_220EF30(v42);
            if ( (_QWORD *)v42 == v4 )
              goto LABEL_66;
          }
          if ( v44 != 84 )
            break;
          v116 |= sub_2D00E90((unsigned int *)v3, (unsigned __int64)v43);
          v42 = sub_220EF30(v42);
          if ( (_QWORD *)v42 == v4 )
            goto LABEL_66;
        }
        if ( v44 == 86 )
          break;
        if ( v44 != 85 )
          goto LABEL_78;
        v116 |= sub_2D00FF0((unsigned int *)v3, (unsigned __int64)v43);
        v42 = sub_220EF30(v42);
        if ( (_QWORD *)v42 == v4 )
          goto LABEL_66;
      }
      v116 |= sub_2D00F70((unsigned int *)v3, v43);
      v42 = sub_220EF30(v42);
    }
    while ( (_QWORD *)v42 != v4 );
LABEL_66:
    v5 = v107;
  }
  while ( *(_QWORD *)(v3 + 144) || v116 );
  v49 = *(_QWORD **)(v100 + 80);
  if ( (_QWORD *)v103 == v49 )
    return 0;
  if ( !v49 )
    BUG();
  while ( 1 )
  {
    k = (_QWORD *)v49[4];
    if ( k != v49 + 3 )
      break;
    v49 = (_QWORD *)v49[1];
    if ( (_QWORD *)v103 == v49 )
      return 0;
    if ( !v49 )
      BUG();
  }
  if ( (_QWORD *)v103 == v49 )
  {
    return 0;
  }
  else
  {
    v121 = (unsigned int *)v3;
    v51 = 0;
    v55 = v49;
    do
    {
      if ( !k )
        BUG();
      v56 = *((_BYTE *)k - 24);
      v57 = k - 3;
      switch ( v56 )
      {
        case '>':
          v51 |= sub_2D00CB0(v121, (__int64)(k - 3));
          break;
        case '=':
          v51 |= sub_2D00D50(v121, (__int64)(k - 3));
          break;
        case 'U':
          v58 = *(k - 7);
          if ( v58 )
          {
            if ( !*(_BYTE *)v58 && *(_QWORD *)(v58 + 24) == k[7] && (*(_BYTE *)(v58 + 33) & 0x20) != 0 )
            {
              v59 = *(_DWORD *)(v58 + 36);
              v117 = v59 == 238 || v59 == 241;
              if ( v117 )
              {
                v108 = (_BYTE *)v57[-4 * (*((_DWORD *)k - 5) & 0x7FFFFFF)];
                v60 = (unsigned int)sub_2D00C30(v121, (_BYTE *)v57[4 * (1LL - (*((_DWORD *)k - 5) & 0x7FFFFFF))]);
                v61 = (unsigned int)sub_2D00C30(v121, v108);
                if ( *((_BYTE *)k - 24) == 85 )
                {
                  v62 = *(k - 7);
                  if ( v62 )
                  {
                    if ( !*(_BYTE *)v62
                      && *(_QWORD *)(v62 + 24) == k[7]
                      && (*(_BYTE *)(v62 + 33) & 0x20) != 0
                      && ((*(_DWORD *)(v62 + 36) - 238) & 0xFFFFFFFD) == 0
                      || !*(_BYTE *)v62
                      && *(_QWORD *)(v62 + 24) == k[7]
                      && (*(_BYTE *)(v62 + 33) & 0x20) != 0
                      && *(_DWORD *)(v62 + 36) == 241 )
                    {
                      if ( v61 )
                      {
                        v105 = v61;
                        v91 = sub_A74840(k + 6, 0);
                        if ( !HIBYTE(v91) || v105 > 1LL << v91 )
                        {
                          _BitScanReverse64(&v92, v105);
                          v93 = 63 - (v92 ^ 0x3F);
                          v94 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                          k[6] = sub_A7B980(k + 6, v94, 1, 86);
                          v95 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                          v96 = sub_A77A40(v95, v93);
                          LODWORD(v122) = 0;
                          v110 = v96;
                          v97 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                          v51 = v117;
                          v57[9] = sub_A7B660(k + 6, v97, &v122, 1, v110);
                        }
                      }
                      if ( v60 )
                      {
                        v109 = k + 6;
                        v63 = sub_A74840(k + 6, 1);
                        if ( !HIBYTE(v63) || v60 > 1LL << v63 )
                        {
                          _BitScanReverse64(&v60, v60);
                          v64 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                          k[6] = sub_A7B980(v109, v64, 2, 86);
                          v65 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                          v66 = sub_A77A40(v65, 63 - ((unsigned __int8)v60 ^ 0x3Fu));
                          LODWORD(v122) = 1;
                          v67 = v66;
                          v68 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                          v51 = v117;
                          v57[9] = sub_A7B660(v109, v68, &v122, 1, v67);
                        }
                      }
                    }
                  }
                }
              }
              else if ( v59 == 243 )
              {
                v80 = sub_A74840(k + 6, 0);
                v81 = 1;
                if ( HIBYTE(v80) )
                  v81 = 1LL << v80;
                v119 = v81;
                v82 = sub_BD3990((unsigned __int8 *)v57[-4 * (*((_DWORD *)k - 5) & 0x7FFFFFF)], 0);
                LODWORD(v83) = sub_2D00C30(v121, v82);
                if ( (unsigned int)v83 > v119 )
                {
                  v84 = -1;
                  if ( (_DWORD)v83 )
                  {
                    _BitScanReverse64((unsigned __int64 *)&v83, (unsigned int)v83);
                    v84 = 63 - (v83 ^ 0x3F);
                  }
                  v85 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                  k[6] = sub_A7B980(k + 6, v85, 1, 86);
                  v86 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                  v87 = sub_A77A40(v86, v84);
                  LODWORD(v122) = 0;
                  v88 = v87;
                  v89 = (__int64 *)sub_BD5C60((__int64)(k - 3));
                  v90 = v88;
                  v51 = 1;
                  v57[9] = sub_A7B660(k + 6, v89, &v122, 1, v90);
                }
              }
            }
          }
          break;
      }
      for ( k = (_QWORD *)k[1]; k == v55 + 3; k = (_QWORD *)v55[4] )
      {
        v55 = (_QWORD *)v55[1];
        if ( (_QWORD *)v103 == v55 )
          return v51;
        if ( !v55 )
          BUG();
      }
    }
    while ( (_QWORD *)v103 != v55 );
  }
  return v51;
}
