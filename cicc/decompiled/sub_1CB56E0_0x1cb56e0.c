// Function: sub_1CB56E0
// Address: 0x1cb56e0
//
__int64 __fastcall sub_1CB56E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  _QWORD *v3; // rsi
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 i; // r13
  int v9; // ebx
  __int64 v11; // r12
  __int64 v12; // r13
  char *v13; // r14
  __int64 v14; // r15
  size_t v15; // rdx
  char v16; // al
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned __int8 v22; // r14
  const char *v23; // rax
  size_t v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rsi
  _BYTE *v33; // rsi
  __int64 *v34; // rax
  __int64 v35; // rax
  char *v36; // rsi
  size_t v37; // rdx
  __int64 v38; // rcx
  size_t v39; // rax
  int v40; // eax
  __int64 v41; // r12
  __int64 v42; // rax
  const char *v43; // rax
  __m128i *v44; // rdi
  char *v45; // rsi
  size_t v46; // rdx
  size_t v47; // r13
  unsigned __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rbx
  __int64 v51; // r12
  __int64 j; // r13
  _QWORD *v53; // rax
  __int64 v54; // r14
  __int64 v55; // rax
  const char *v56; // rax
  size_t v57; // rdx
  __int64 *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rsi
  _QWORD *v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 *v67; // rax
  __int64 v68; // rax
  size_t v69; // rdx
  char *v70; // rsi
  __int64 v71; // r8
  size_t v72; // rdx
  char *v73; // r14
  size_t v74; // rax
  _QWORD *v75; // rax
  __m128i *v76; // rdx
  __int64 v77; // r12
  __m128i si128; // xmm0
  const char *v79; // rax
  size_t v80; // rdx
  _BYTE *v81; // rdi
  char *v82; // rsi
  unsigned __int64 v83; // rax
  __int64 *v84; // r13
  __int64 *v85; // rbx
  int v86; // r14d
  __int64 v87; // rdi
  __int64 v88; // rax
  _QWORD *v89; // rax
  __int64 v90; // rdx
  const char *v91; // rax
  size_t v92; // rdx
  __m128i v93; // xmm0
  __int64 v94; // rax
  __int64 v95; // r14
  char *v96; // rsi
  size_t v97; // rdx
  const char *v98; // rax
  size_t v99; // rdx
  size_t v100; // r14
  __int64 *v101; // rax
  __int64 v102; // rax
  __int64 v103; // rsi
  _QWORD *v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  _QWORD *v107; // rax
  __int64 v108; // rsi
  __int64 v109; // r14
  __int64 *v110; // rax
  __int64 v111; // rax
  char *v112; // rsi
  size_t v113; // rdx
  __int64 v114; // r14
  __int64 v115; // rcx
  unsigned int v116; // eax
  __int64 v117; // r13
  __int64 v118; // r12
  unsigned int v119; // r14d
  __int64 v120; // r15
  char *v121; // rsi
  size_t v122; // rdx
  __int64 v123; // rax
  __int64 v124; // r14
  char *v125; // rsi
  size_t v126; // rdx
  const char *v127; // rax
  size_t v128; // rdx
  size_t v129; // r14
  __int64 *v130; // rax
  __int64 v131; // rax
  __int64 v132; // rsi
  _QWORD *v133; // rax
  __int64 v134; // rax
  __int64 v135; // rax
  _QWORD *v136; // rax
  __int64 v137; // rsi
  __int64 v138; // r14
  __int64 *v139; // rax
  __int64 v140; // rax
  char *v141; // r14
  size_t v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // [rsp+0h] [rbp-110h]
  char *v146; // [rsp+8h] [rbp-108h]
  __int64 v147; // [rsp+8h] [rbp-108h]
  __int64 v148; // [rsp+8h] [rbp-108h]
  __int64 v149; // [rsp+10h] [rbp-100h]
  size_t v150; // [rsp+18h] [rbp-F8h]
  __int64 *v151; // [rsp+18h] [rbp-F8h]
  char *v152; // [rsp+18h] [rbp-F8h]
  char *v153; // [rsp+18h] [rbp-F8h]
  __int64 v154; // [rsp+18h] [rbp-F8h]
  __int64 v155; // [rsp+18h] [rbp-F8h]
  char *v156; // [rsp+18h] [rbp-F8h]
  __int64 *v157; // [rsp+18h] [rbp-F8h]
  char *v158; // [rsp+18h] [rbp-F8h]
  __int64 *v159; // [rsp+18h] [rbp-F8h]
  __int64 v160; // [rsp+18h] [rbp-F8h]
  unsigned __int8 v161; // [rsp+26h] [rbp-EAh]
  unsigned __int8 v162; // [rsp+27h] [rbp-E9h]
  __int64 v163; // [rsp+28h] [rbp-E8h]
  __int64 v165; // [rsp+38h] [rbp-D8h]
  size_t v166; // [rsp+38h] [rbp-D8h]
  __int64 *v167; // [rsp+38h] [rbp-D8h]
  __int64 v168; // [rsp+38h] [rbp-D8h]
  size_t v169; // [rsp+38h] [rbp-D8h]
  _QWORD *v170; // [rsp+48h] [rbp-C8h] BYREF
  __int64 *v171; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v172; // [rsp+58h] [rbp-B8h]
  _QWORD v173[22]; // [rsp+60h] [rbp-B0h] BYREF

  v2 = a2;
  if ( byte_4FBE8C0 )
  {
    v75 = sub_16E8CB0();
    v76 = (__m128i *)v75[3];
    v77 = (__int64)v75;
    if ( v75[2] - (_QWORD)v76 <= 0x2Cu )
    {
      v77 = sub_16E7EE0((__int64)v75, "Processing __restrict__ keyword for function ", 0x2Du);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42E00F0);
      qmemcpy(&v76[2], "for function ", 13);
      *v76 = si128;
      v76[1] = _mm_load_si128((const __m128i *)&xmmword_42E0100);
      v75[3] += 45LL;
    }
    v79 = sub_1649960(a2);
    v81 = *(_BYTE **)(v77 + 24);
    v82 = (char *)v79;
    v83 = *(_QWORD *)(v77 + 16) - (_QWORD)v81;
    if ( v83 < v80 )
    {
      v143 = sub_16E7EE0(v77, v82, v80);
      v81 = *(_BYTE **)(v143 + 24);
      v77 = v143;
      if ( *(_QWORD *)(v143 + 16) - (_QWORD)v81 > 4u )
      {
LABEL_93:
        *(_DWORD *)v81 = 774778400;
        v81[4] = 10;
        *(_QWORD *)(v77 + 24) += 5LL;
        goto LABEL_2;
      }
    }
    else
    {
      if ( v80 )
      {
        v169 = v80;
        memcpy(v81, v82, v80);
        v94 = *(_QWORD *)(v77 + 16);
        v81 = (_BYTE *)(v169 + *(_QWORD *)(v77 + 24));
        *(_QWORD *)(v77 + 24) = v81;
        v83 = v94 - (_QWORD)v81;
      }
      if ( v83 > 4 )
        goto LABEL_93;
    }
    sub_16E7EE0(v77, " ...\n", 5u);
  }
LABEL_2:
  v3 = (_QWORD *)v2;
  if ( (unsigned __int8)sub_1CCAAC0(1, v2) || (v3 = (_QWORD *)v2, (v162 = sub_1CCAAC0(2, v2)) != 0) )
  {
    v6 = *(_QWORD *)(v2 + 80);
    v7 = v2 + 72;
    v171 = v173;
    v172 = 0x1000000000LL;
    if ( v2 + 72 == v6 )
    {
LABEL_10:
      v162 = 0;
    }
    else
    {
      if ( !v6 )
        BUG();
      while ( 1 )
      {
        i = *(_QWORD *)(v6 + 24);
        if ( i != v6 + 16 )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( v7 == v6 )
          goto LABEL_10;
        if ( !v6 )
          BUG();
      }
      while ( v7 != v6 )
      {
        if ( !i )
          BUG();
        if ( !*(_QWORD *)(i - 16) )
        {
          v88 = (unsigned int)v172;
          if ( (unsigned int)v172 >= HIDWORD(v172) )
          {
            v3 = v173;
            sub_16CD150((__int64)&v171, v173, 0, 8, v4, v5);
            v88 = (unsigned int)v172;
          }
          v171[v88] = i - 24;
          LODWORD(v172) = v172 + 1;
        }
        for ( i = *(_QWORD *)(i + 8); i == v6 - 24 + 40; i = *(_QWORD *)(v6 + 24) )
        {
          v6 = *(_QWORD *)(v6 + 8);
          if ( v7 == v6 )
            goto LABEL_103;
          if ( !v6 )
            BUG();
        }
      }
LABEL_103:
      v84 = v171;
      v85 = &v171[(unsigned int)v172];
      if ( v85 == v171 )
      {
        v162 = 0;
      }
      else
      {
        v86 = 0;
        do
        {
          v87 = *v84;
          v3 = 0;
          ++v84;
          v86 |= sub_1AEB370(v87, 0);
        }
        while ( v85 != v84 );
        v162 = v86;
        v84 = v171;
      }
      if ( v84 != v173 )
        _libc_free((unsigned __int64)v84);
    }
  }
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( !byte_4FBEB60 )
      goto LABEL_13;
  }
  else
  {
    if ( (*(_BYTE *)(v2 + 18) & 1) != 0 )
    {
      sub_15E08E0(v2, (__int64)v3);
      v50 = *(_QWORD *)(v2 + 88);
      v51 = v50 + 40LL * *(_QWORD *)(v2 + 96);
      if ( (*(_BYTE *)(v2 + 18) & 1) != 0 )
      {
        sub_15E08E0(v2, (__int64)v3);
        v50 = *(_QWORD *)(v2 + 88);
      }
    }
    else
    {
      v50 = *(_QWORD *)(v2 + 88);
      v51 = v50 + 40LL * *(_QWORD *)(v2 + 96);
    }
    for ( ; v51 != v50; v50 += 40 )
    {
      if ( (unsigned __int8)sub_15E04B0(v50) )
      {
        for ( j = *(_QWORD *)(v50 + 8); j; j = *(_QWORD *)(j + 8) )
        {
          v53 = sub_1648700(j);
          v54 = (__int64)v53;
          if ( *((_BYTE *)v53 + 16) == 55 )
          {
            v55 = *(v53 - 6);
            if ( v55 )
            {
              if ( v55 == v50 )
              {
                v171 = 0;
                v172 = 0;
                v173[0] = 0;
                v56 = sub_1649960(v2);
                v166 = v57;
                v153 = (char *)v56;
                v58 = (__int64 *)sub_16498A0(v54);
                v59 = sub_161FF10(v58, v153, v166);
                v60 = v172;
                v170 = (_QWORD *)v59;
                if ( v172 == v173[0] )
                {
                  sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
                }
                else
                {
                  if ( v172 )
                  {
                    *(_QWORD *)v172 = v59;
                    v60 = v172;
                  }
                  v172 = v60 + 8;
                }
                v61 = (_QWORD *)sub_16498A0(v54);
                v62 = sub_1643350(v61);
                v63 = sub_159C470(v62, 0, 0);
                v64 = sub_1624210(v63);
                v65 = v172;
                v170 = v64;
                if ( v172 == v173[0] )
                {
                  sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
                  v66 = v172;
                }
                else
                {
                  if ( v172 )
                  {
                    *(_QWORD *)v172 = v64;
                    v65 = v172;
                  }
                  v66 = v65 + 8;
                  v172 = v65 + 8;
                }
                v154 = v66;
                v167 = v171;
                v67 = (__int64 *)sub_16498A0(v54);
                v68 = sub_1627350(v67, v167, (__int64 *)((v154 - (__int64)v167) >> 3), 0, 1);
                v69 = 0;
                v168 = v68;
                v70 = off_4CD4978[0];
                if ( off_4CD4978[0] )
                {
                  v70 = off_4CD4978[0];
                  v69 = strlen(off_4CD4978[0]);
                }
                sub_1626100(v54, v70, v69, v168);
                v71 = *(_QWORD *)(v54 - 24);
                if ( *(_BYTE *)(v71 + 16) > 0x17u )
                {
                  v72 = 0;
                  v73 = off_4CD4978[0];
                  if ( off_4CD4978[0] )
                  {
                    v155 = v71;
                    v74 = strlen(off_4CD4978[0]);
                    v71 = v155;
                    v72 = v74;
                  }
                  sub_1626100(v71, v73, v72, v168);
                }
                sub_1CCAB50(1, v2);
                sub_1CCABF0(3, v2);
                sub_1CCABF0(2, v2);
                if ( v171 )
                  j_j___libc_free_0(v171, v173[0] - (_QWORD)v171);
              }
            }
          }
        }
      }
    }
    if ( !byte_4FBEB60 || !*(_BYTE *)(a1 + 16) )
    {
LABEL_13:
      v9 = sub_1CB4F40(a1, (_QWORD *)v2);
      if ( !(_BYTE)v9 || !byte_4FBE8C0 )
        return v9 | (unsigned int)v162;
      v89 = sub_16E8CB0();
      v90 = v89[3];
      v41 = (__int64)v89;
      if ( (unsigned __int64)(v89[2] - v90) <= 8 )
      {
        v41 = sub_16E7EE0((__int64)v89, "Function ", 9u);
      }
      else
      {
        *(_BYTE *)(v90 + 8) = 32;
        *(_QWORD *)v90 = 0x6E6F6974636E7546LL;
        v89[3] += 9LL;
      }
      v91 = sub_1649960(v2);
      v44 = *(__m128i **)(v41 + 24);
      v45 = (char *)v91;
      v47 = v92;
      v48 = *(_QWORD *)(v41 + 16) - (_QWORD)v44;
      if ( v92 <= v48 )
        goto LABEL_115;
      goto LABEL_58;
    }
  }
  v161 = 0;
  v149 = v2 + 72;
  v165 = *(_QWORD *)(v2 + 80);
  if ( v165 != v2 + 72 )
  {
    v163 = v2;
    while ( 1 )
    {
      if ( !v165 )
        BUG();
      v11 = v165 + 16;
      if ( *(_QWORD *)(v165 + 24) != v165 + 16 )
        break;
LABEL_51:
      v165 = *(_QWORD *)(v165 + 8);
      if ( v149 == v165 )
      {
        v2 = v163;
        goto LABEL_53;
      }
    }
    v12 = *(_QWORD *)(v165 + 24);
    while ( 1 )
    {
      v13 = off_4CD4978[0];
      v14 = v12 - 24;
      if ( !v12 )
        v14 = 0;
      v15 = 0;
      if ( off_4CD4978[0] )
        v15 = strlen(off_4CD4978[0]);
      if ( (*(_QWORD *)(v14 + 48) || *(__int16 *)(v14 + 18) < 0) && sub_1625940(v14, v13, v15) )
        goto LABEL_24;
      v16 = *(_BYTE *)(v14 + 16);
      if ( v16 != 86 )
        break;
      v17 = *(__int64 **)(v14 - 24);
      v18 = *v17;
      if ( *(_BYTE *)(*v17 + 8) == 13 && (*(_BYTE *)(v18 + 9) & 4) == 0 )
      {
        v19 = sub_1643640(v18);
        v21 = v19;
        if ( v20 > 6 && *(_DWORD *)v19 == 1970435187 && *(_WORD *)(v19 + 4) == 29795 && *(_BYTE *)(v19 + 6) == 46 )
        {
          v20 -= 7LL;
          v21 = v19 + 7;
        }
        if ( *(_DWORD *)(v14 + 64) == 1 )
        {
          v22 = sub_1CCACD0(*(_QWORD *)(v163 + 40), v21, v20, **(unsigned int **)(v14 + 56));
          if ( v22 )
          {
            v171 = 0;
            v172 = 0;
            v173[0] = 0;
            v23 = sub_1649960(v163);
            v150 = v24;
            v146 = (char *)v23;
            v25 = (__int64 *)sub_16498A0(v14);
            v26 = sub_161FF10(v25, v146, v150);
            v27 = v172;
            v170 = (_QWORD *)v26;
            if ( v172 == v173[0] )
            {
              sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
            }
            else
            {
              if ( v172 )
              {
                *(_QWORD *)v172 = v26;
                v27 = v172;
              }
              v172 = v27 + 8;
            }
            v28 = (_QWORD *)sub_16498A0(v14);
            v29 = sub_1643350(v28);
            v30 = sub_159C470(v29, 0, 0);
            v31 = sub_1624210(v30);
            v32 = v172;
            v170 = v31;
            if ( v172 == v173[0] )
            {
              sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
              v33 = (_BYTE *)v172;
            }
            else
            {
              if ( v172 )
              {
                *(_QWORD *)v172 = v31;
                v32 = v172;
              }
              v33 = (_BYTE *)(v32 + 8);
              v172 = (__int64)v33;
            }
            v151 = v171;
            v34 = (__int64 *)sub_16498A0(v14);
            v35 = sub_1627350(v34, v151, (__int64 *)((v33 - (_BYTE *)v151) >> 3), 0, 1);
            v36 = off_4CD4978[0];
            v37 = 0;
            v38 = v35;
            if ( off_4CD4978[0] )
            {
              v152 = off_4CD4978[0];
              v147 = v35;
              v39 = strlen(off_4CD4978[0]);
              v38 = v147;
              v36 = v152;
              v37 = v39;
            }
            sub_1626100(v14, v36, v37, v38);
            sub_1CCAB50(1, v163);
            sub_1CCABF0(3, v163);
            sub_1CCABF0(2, v163);
            if ( v171 )
              j_j___libc_free_0(v171, v173[0] - (_QWORD)v171);
            v161 = v22;
          }
        }
      }
LABEL_24:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == v11 )
        goto LABEL_51;
    }
    switch ( v16 )
    {
      case 'H':
        v95 = *(_QWORD *)(v14 - 24);
        if ( *(_BYTE *)(v95 + 16) <= 0x17u )
          goto LABEL_24;
        v96 = off_4CD4978[0];
        v97 = 0;
        if ( off_4CD4978[0] )
        {
          v96 = off_4CD4978[0];
          v97 = strlen(off_4CD4978[0]);
        }
        if ( !*(_QWORD *)(v95 + 48) && *(__int16 *)(v95 + 18) >= 0 || !sub_1625940(v95, v96, v97) )
          goto LABEL_24;
        break;
      case 'M':
        v116 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
        if ( v116 )
        {
          v148 = v12;
          v145 = v11;
          v117 = 0;
          v118 = v14;
          v119 = 0;
          while ( 1 )
          {
            if ( (*(_BYTE *)(v118 + 23) & 0x40) != 0 )
            {
              v120 = *(_QWORD *)(*(_QWORD *)(v118 - 8) + v117);
              if ( *(_BYTE *)(v120 + 16) <= 0x17u )
                goto LABEL_151;
            }
            else
            {
              v120 = *(_QWORD *)(v118 - 24LL * v116 + v117);
              if ( *(_BYTE *)(v120 + 16) <= 0x17u )
              {
LABEL_151:
                v12 = v148;
                v11 = v145;
                goto LABEL_24;
              }
            }
            v121 = off_4CD4978[0];
            v122 = 0;
            if ( off_4CD4978[0] )
            {
              v121 = off_4CD4978[0];
              v122 = strlen(off_4CD4978[0]);
            }
            if ( !*(_QWORD *)(v120 + 48) && *(__int16 *)(v120 + 18) >= 0 )
              break;
            if ( !sub_1625940(v120, v121, v122) )
              goto LABEL_151;
            ++v119;
            v117 += 24;
            v116 = *(_DWORD *)(v118 + 20) & 0xFFFFFFF;
            if ( v116 <= v119 )
            {
              v14 = v118;
              v12 = v148;
              v11 = v145;
              goto LABEL_125;
            }
          }
          v12 = v148;
          v11 = v145;
          goto LABEL_24;
        }
        break;
      case 'N':
        v123 = *(_QWORD *)(v14 - 24);
        if ( *(_BYTE *)(v123 + 16) )
          goto LABEL_24;
        if ( (*(_BYTE *)(v123 + 33) & 0x20) == 0 )
          goto LABEL_24;
        if ( *(_DWORD *)(v123 + 36) != 3660 )
          goto LABEL_24;
        v124 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v124 + 16) <= 0x17u )
          goto LABEL_24;
        v125 = off_4CD4978[0];
        v126 = 0;
        if ( off_4CD4978[0] )
        {
          v125 = off_4CD4978[0];
          v126 = strlen(off_4CD4978[0]);
        }
        if ( !*(_QWORD *)(v124 + 48) && *(__int16 *)(v124 + 18) >= 0 || !sub_1625940(v124, v125, v126) )
          goto LABEL_24;
        v171 = 0;
        v172 = 0;
        v173[0] = 0;
        v127 = sub_1649960(v163);
        v129 = v128;
        v158 = (char *)v127;
        v130 = (__int64 *)sub_16498A0(v14);
        v131 = sub_161FF10(v130, v158, v129);
        v132 = v172;
        v170 = (_QWORD *)v131;
        if ( v172 == v173[0] )
        {
          sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
        }
        else
        {
          if ( v172 )
          {
            *(_QWORD *)v172 = v131;
            v132 = v172;
          }
          v172 = v132 + 8;
        }
        v133 = (_QWORD *)sub_16498A0(v14);
        v134 = sub_1643350(v133);
        v135 = sub_159C470(v134, 0, 0);
        v136 = sub_1624210(v135);
        v137 = v172;
        v170 = v136;
        if ( v172 == v173[0] )
        {
          sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
          v138 = v172;
        }
        else
        {
          if ( v172 )
          {
            *(_QWORD *)v172 = v136;
            v137 = v172;
          }
          v138 = v137 + 8;
          v172 = v137 + 8;
        }
        v159 = v171;
        v139 = (__int64 *)sub_16498A0(v14);
        v140 = sub_1627350(v139, v159, (__int64 *)((v138 - (__int64)v159) >> 3), 0, 1);
        v141 = off_4CD4978[0];
        v113 = 0;
        v115 = v140;
        if ( off_4CD4978[0] )
        {
          v160 = v140;
          v142 = strlen(off_4CD4978[0]);
          v115 = v160;
          v113 = v142;
        }
        v112 = v141;
        goto LABEL_136;
      default:
        goto LABEL_24;
    }
LABEL_125:
    v171 = 0;
    v172 = 0;
    v173[0] = 0;
    v98 = sub_1649960(v163);
    v100 = v99;
    v156 = (char *)v98;
    v101 = (__int64 *)sub_16498A0(v14);
    v102 = sub_161FF10(v101, v156, v100);
    v103 = v172;
    v170 = (_QWORD *)v102;
    if ( v172 == v173[0] )
    {
      sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
    }
    else
    {
      if ( v172 )
      {
        *(_QWORD *)v172 = v102;
        v103 = v172;
      }
      v172 = v103 + 8;
    }
    v104 = (_QWORD *)sub_16498A0(v14);
    v105 = sub_1643350(v104);
    v106 = sub_159C470(v105, 0, 0);
    v107 = sub_1624210(v106);
    v108 = v172;
    v170 = v107;
    if ( v172 == v173[0] )
    {
      sub_1273E00((__int64)&v171, (_BYTE *)v172, &v170);
      v109 = v172;
    }
    else
    {
      if ( v172 )
      {
        *(_QWORD *)v172 = v107;
        v108 = v172;
      }
      v109 = v108 + 8;
      v172 = v108 + 8;
    }
    v157 = v171;
    v110 = (__int64 *)sub_16498A0(v14);
    v111 = sub_1627350(v110, v157, (__int64 *)((v109 - (__int64)v157) >> 3), 0, 1);
    v112 = off_4CD4978[0];
    v113 = 0;
    v114 = v111;
    if ( off_4CD4978[0] )
    {
      v112 = off_4CD4978[0];
      v113 = strlen(off_4CD4978[0]);
    }
    v115 = v114;
LABEL_136:
    sub_1626100(v14, v112, v113, v115);
    sub_1CCAB50(1, v163);
    sub_1CCABF0(3, v163);
    sub_1CCABF0(2, v163);
    if ( v171 )
      j_j___libc_free_0(v171, v173[0] - (_QWORD)v171);
    v161 = 1;
    goto LABEL_24;
  }
LABEL_53:
  v40 = sub_1CB4F40(a1, (_QWORD *)v2);
  v9 = v40 | v161;
  if ( (_BYTE)v40 && byte_4FBE8C0 )
  {
    v41 = (__int64)sub_16E8CB0();
    v42 = *(_QWORD *)(v41 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v41 + 16) - v42) <= 8 )
    {
      v41 = sub_16E7EE0(v41, "Function ", 9u);
    }
    else
    {
      *(_BYTE *)(v42 + 8) = 32;
      *(_QWORD *)v42 = 0x6E6F6974636E7546LL;
      *(_QWORD *)(v41 + 24) += 9LL;
    }
    v43 = sub_1649960(v2);
    v44 = *(__m128i **)(v41 + 24);
    v45 = (char *)v43;
    v47 = v46;
    v48 = *(_QWORD *)(v41 + 16) - (_QWORD)v44;
    if ( v48 >= v46 )
    {
LABEL_115:
      if ( v47 )
      {
        memcpy(v44, v45, v47);
        v144 = *(_QWORD *)(v41 + 16);
        v44 = (__m128i *)(v47 + *(_QWORD *)(v41 + 24));
        *(_QWORD *)(v41 + 24) = v44;
        v48 = v144 - (_QWORD)v44;
      }
      goto LABEL_117;
    }
LABEL_58:
    v49 = sub_16E7EE0(v41, v45, v47);
    v44 = *(__m128i **)(v49 + 24);
    v41 = v49;
    v48 = *(_QWORD *)(v49 + 16) - (_QWORD)v44;
LABEL_117:
    if ( v48 <= 0x21 )
    {
      sub_16E7EE0(v41, ": __restrict__ keyword processed.\n", 0x22u);
    }
    else
    {
      v93 = _mm_load_si128((const __m128i *)&xmmword_42E0110);
      v44[2].m128i_i16[0] = 2606;
      *v44 = v93;
      v44[1] = _mm_load_si128((const __m128i *)&xmmword_42E0120);
      *(_QWORD *)(v41 + 24) += 34LL;
    }
  }
  return v9 | (unsigned int)v162;
}
