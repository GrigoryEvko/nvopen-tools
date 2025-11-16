// Function: sub_79B7D0
// Address: 0x79b7d0
//
__int64 __fastcall sub_79B7D0(__int64 a1, const __m128i *a2, FILE *a3, __int64 a4, const __m128i *a5, unsigned int a6)
{
  __int64 v6; // r12
  const __m128i *v7; // r15
  unsigned int v8; // r13d
  const __m128i *v10; // rsi
  __m128i *v11; // rdx
  __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // edi
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 v18; // rsi
  __m128i *v19; // r9
  const __m128i *v20; // rcx
  const __m128i **v21; // rdx
  _BYTE *v23; // r14
  _QWORD *v24; // r9
  char v25; // al
  char v26; // dl
  int v27; // eax
  size_t v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // r13d
  unsigned int v31; // r13d
  char *v32; // r8
  char *v33; // r8
  unsigned int v34; // eax
  const __m128i *v35; // r8
  const __m128i *v36; // rsi
  __int64 *v37; // rbx
  __int64 v38; // r15
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // r13
  char j; // cl
  unsigned int k; // edx
  __int64 v43; // rax
  __m128i *v44; // r14
  char v45; // cl
  __int64 v46; // rsi
  __int32 v47; // eax
  __int64 v48; // rax
  int v49; // eax
  __int64 v50; // rax
  __int8 v51; // al
  _BYTE *v52; // r14
  __int64 v53; // rax
  char i; // dl
  int v55; // eax
  char v56; // al
  __int8 v57; // al
  int v58; // eax
  unsigned __int64 v59; // r8
  __int64 v60; // rsi
  __int32 v61; // eax
  __int64 v62; // rax
  __int32 v63; // eax
  __int64 v64; // rax
  __int64 v65; // rdi
  char v66; // r9
  int v67; // r11d
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // rsi
  int v71; // eax
  __m128i *v72; // r10
  char v73; // al
  char v74; // al
  _QWORD **v75; // rsi
  unsigned int v76; // eax
  unsigned __int64 v77; // r13
  unsigned int m; // edx
  __int64 v79; // rax
  unsigned int v80; // edi
  int v81; // esi
  unsigned int v82; // edx
  __int64 *v83; // rax
  __int64 v84; // r8
  _QWORD *v85; // rax
  __m128i v86; // xmm3
  unsigned __int64 v87; // rsi
  __int64 v88; // r13
  __m128i v89; // xmm4
  __m128i v90; // xmm5
  __m128i v91; // xmm6
  __m128i v92; // xmm7
  __m128i v93; // xmm0
  __m128i v94; // xmm2
  __int64 v95; // rax
  char v96; // al
  __int64 v97; // r10
  bool v98; // zf
  __int64 v99; // rax
  __m128i v100; // xmm5
  __int64 v101; // r14
  __int64 v102; // r12
  const __m128i *v103; // r15
  int v104; // r11d
  int v105; // eax
  unsigned int v106; // r13d
  __int64 v107; // rax
  __int64 v108; // r8
  __int64 v109; // rax
  __int64 v110; // rdi
  char v111; // al
  __int8 v112; // al
  int v113; // edx
  char v114; // al
  unsigned int v115; // eax
  const __m128i *v116; // rsi
  unsigned __int64 v117; // [rsp+0h] [rbp-140h]
  unsigned __int64 v118; // [rsp+0h] [rbp-140h]
  int v119; // [rsp+8h] [rbp-138h]
  unsigned int v120; // [rsp+Ch] [rbp-134h]
  __int64 v121; // [rsp+10h] [rbp-130h]
  const __m128i *v122; // [rsp+10h] [rbp-130h]
  __int64 v123; // [rsp+18h] [rbp-128h]
  _BYTE *v124; // [rsp+18h] [rbp-128h]
  __int64 v125; // [rsp+20h] [rbp-120h]
  size_t v126; // [rsp+20h] [rbp-120h]
  size_t v127; // [rsp+20h] [rbp-120h]
  int v128; // [rsp+20h] [rbp-120h]
  const __m128i *v129; // [rsp+20h] [rbp-120h]
  __m128i *v130; // [rsp+20h] [rbp-120h]
  __int64 v131; // [rsp+28h] [rbp-118h]
  const __m128i *v132; // [rsp+28h] [rbp-118h]
  __int64 v133; // [rsp+28h] [rbp-118h]
  __int64 v134; // [rsp+28h] [rbp-118h]
  _QWORD *v135; // [rsp+30h] [rbp-110h]
  char *v136; // [rsp+30h] [rbp-110h]
  __m128i *v137; // [rsp+30h] [rbp-110h]
  _QWORD *v138; // [rsp+30h] [rbp-110h]
  _QWORD *v139; // [rsp+30h] [rbp-110h]
  _QWORD *v140; // [rsp+30h] [rbp-110h]
  int v141; // [rsp+40h] [rbp-100h]
  char *v142; // [rsp+40h] [rbp-100h]
  int v145; // [rsp+6Ch] [rbp-D4h] BYREF
  __m256i v146; // [rsp+70h] [rbp-D0h] BYREF
  __m128i v147; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v148; // [rsp+A0h] [rbp-A0h]
  __m128i v149; // [rsp+B0h] [rbp-90h]
  __m128i v150; // [rsp+C0h] [rbp-80h]
  __m128i v151; // [rsp+D0h] [rbp-70h]
  __m128i v152; // [rsp+E0h] [rbp-60h]
  __m128i v153; // [rsp+F0h] [rbp-50h]
  __m128i v154; // [rsp+100h] [rbp-40h]

  v6 = a1;
  v7 = a2;
  switch ( a2[3].m128i_i8[0] )
  {
    case 0:
    case 1:
      goto LABEL_5;
    case 2:
    case 6:
      v10 = (const __m128i *)a2[3].m128i_i64[1];
      v11 = *(__m128i **)a4;
      v12 = v10[10].m128i_i8[13];
      if ( v12 == 1 )
      {
        if ( (v10[10].m128i_i8[8] & 8) != 0 )
          goto LABEL_75;
        if ( (v10[10].m128i_i8[11] & 4) != 0 )
          return 0;
        *v11 = _mm_loadu_si128(v10 + 11);
LABEL_5:
        if ( (v7[3].m128i_i8[3] & 8) != 0 )
          goto LABEL_69;
      }
      else
      {
        if ( v12 != 3 )
          goto LABEL_75;
        if ( (v10[10].m128i_i8[11] & 4) != 0 )
          return 0;
        *v11 = _mm_loadu_si128(v10 + 11);
        if ( (v7[3].m128i_i8[3] & 8) != 0 )
          goto LABEL_69;
      }
      return 1;
    case 3:
    case 4:
      v13 = *(_QWORD *)(a1 + 72);
      v14 = a2[3].m128i_i64[1];
      if ( !v13
        || *(_BYTE *)(v14 + 24) != 1
        || (unsigned __int8)(*(_BYTE *)(v14 + 56) - 105) > 4u
        || (*(_BYTE *)(v14 + 25) & 3) != 0
        || *(_DWORD *)(v13 + 44) )
      {
        v8 = sub_786210(a1, (_QWORD **)v14, *(_QWORD *)a4, *(char **)(a4 + 24));
      }
      else
      {
        *(_DWORD *)(v13 + 44) = *(_DWORD *)(a4 + 12) + 1;
        v8 = sub_786210(a1, (_QWORD **)v14, *(_QWORD *)a4, *(char **)(a4 + 24));
        *(_DWORD *)(*(_QWORD *)(a1 + 72) + 44LL) = 0;
      }
      goto LABEL_15;
    case 5:
      if ( (a2[4].m128i_i8[8] & 0x10) == 0 )
      {
        v8 = sub_799B70(a1, (__int64)a2, a3, (const __m128i *)a4, a5, a6);
        goto LABEL_15;
      }
      v145 = 1;
      v23 = (_BYTE *)a2[4].m128i_i64[0];
      v24 = *(_QWORD **)v23;
      if ( *(_BYTE *)(*(_QWORD *)v23 + 140LL) == 12 )
      {
        do
          v24 = (_QWORD *)v24[20];
        while ( *((_BYTE *)v24 + 140) == 12 );
      }
      v25 = v23[25];
      if ( (v25 & 3) != 0 )
      {
        v141 = 0;
        v26 = *((_BYTE *)v24 + 140);
        v27 = 32;
        goto LABEL_27;
      }
      v56 = v25 | 1;
      v23[25] = v56;
      v26 = *((_BYTE *)v24 + 140);
      if ( (v56 & 3) != 0 )
      {
        v141 = 1;
        v27 = 32;
        goto LABEL_27;
      }
      v27 = 16;
      v141 = 1;
      if ( (unsigned __int8)(v26 - 2) <= 1u )
      {
LABEL_27:
        if ( (unsigned __int8)(v26 - 8) > 3u )
        {
LABEL_28:
          v28 = 8;
          v29 = 16;
          v30 = 16;
          goto LABEL_29;
        }
        v30 = ((unsigned int)(v27 + 7) >> 3) + 9;
        v113 = ((unsigned __int8)((unsigned int)(v27 + 7) >> 3) + 9) & 7;
        goto LABEL_208;
      }
      v138 = v24;
      v27 = sub_7764B0(a1, (unsigned __int64)v24, &v145);
      v8 = v145;
      v24 = v138;
      if ( v145 )
      {
        if ( (unsigned __int8)(*((_BYTE *)v138 + 140) - 8) > 3u )
          goto LABEL_28;
        v30 = ((unsigned int)(v27 + 7) >> 3) + 9;
        v113 = ((unsigned __int8)((unsigned int)(v27 + 7) >> 3) + 9) & 7;
        if ( (((unsigned __int8)((unsigned int)(v27 + 7) >> 3) + 9) & 7) == 0 )
        {
LABEL_209:
          v29 = v30;
          v28 = v30 - 8LL;
LABEL_29:
          v31 = v27 + v30;
          if ( v31 > 0x400 )
          {
            v126 = v28;
            v133 = v29;
            v106 = v31 + 16;
            v139 = v24;
            v107 = sub_822B10(v106);
            v24 = v139;
            v29 = v133;
            v108 = v107;
            v109 = *(_QWORD *)(a1 + 32);
            v28 = v126;
            *(_DWORD *)(v108 + 8) = v106;
            *(_QWORD *)v108 = v109;
            *(_DWORD *)(v108 + 12) = *(_DWORD *)(a1 + 40);
            *(_QWORD *)(a1 + 32) = v108;
            v32 = (char *)(v108 + 16);
          }
          else
          {
            v32 = *(char **)(a1 + 16);
            if ( (v31 & 7) != 0 )
              v31 = v31 + 8 - (v31 & 7);
            if ( 0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24)) < v31 )
            {
              v127 = v28;
              v134 = v29;
              v140 = v24;
              sub_772E70((_QWORD *)(a1 + 16));
              v32 = *(char **)(a1 + 16);
              v28 = v127;
              v29 = v134;
              v24 = v140;
            }
            *(_QWORD *)(a1 + 16) = &v32[v31];
          }
          v135 = v24;
          v33 = (char *)memset(v32, 0, v28) + v29;
          *((_QWORD *)v33 - 1) = v135;
          if ( (unsigned __int8)(*((_BYTE *)v135 + 140) - 9) <= 2u )
            *(_QWORD *)v33 = 0;
          v131 = (__int64)v135;
          v136 = v33;
          v34 = sub_786210(a1, (_QWORD **)v23, (unsigned __int64)v33, v33);
          v35 = (const __m128i *)v136;
          v8 = v34;
          if ( !v34 )
          {
            v145 = 0;
            goto LABEL_39;
          }
          if ( (v136[8] & 1) != 0 )
          {
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_6855B0(0xA8Du, a3, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
            v145 = 0;
            v8 = 0;
            goto LABEL_39;
          }
          v86 = _mm_loadu_si128(a2 + 1);
          v87 = v131;
          v88 = 1;
          v89 = _mm_loadu_si128(v7 + 2);
          v90 = _mm_loadu_si128(v7 + 3);
          v147 = _mm_loadu_si128(v7);
          v91 = _mm_loadu_si128(v7 + 4);
          v92 = _mm_loadu_si128(v7 + 5);
          v93 = _mm_loadu_si128(v7 + 6);
          v94 = _mm_loadu_si128(v7 + 7);
          v150 = v90;
          v151 = v91;
          v148 = v86;
          v149 = v89;
          v152 = v92;
          v153 = v93;
          v154 = v94;
          v151.m128i_i64[0] = *((_QWORD *)v23 + 2);
          v151.m128i_i8[8] = v91.m128i_i8[8] & 0xEE | 1;
          do
          {
            v95 = *(_QWORD *)(v87 + 176);
            if ( !v95 && (*(_BYTE *)(v87 + 169) & 0x20) == 0 )
            {
              if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
              {
                sub_686CA0(0xAA6u, a1 + 112, v131, (_QWORD *)(a1 + 96));
                sub_770D30(a1);
              }
              goto LABEL_179;
            }
            v88 *= v95;
            do
            {
              v87 = *(_QWORD *)(v87 + 160);
              v96 = *(_BYTE *)(v87 + 140);
            }
            while ( v96 == 12 );
          }
          while ( v96 == 8 );
          v97 = 16;
          if ( (unsigned __int8)(v96 - 2) > 1u )
          {
            v115 = sub_7764B0(a1, v87, &v145);
            v35 = (const __m128i *)v136;
            v97 = v115;
          }
          if ( v145 )
          {
            v98 = (v35->m128i_i8[8] & 4) == 0;
            v35->m128i_i32[2] = ((_DWORD)v88 << 8) | v35->m128i_i8[8] | 8;
            v99 = v35->m128i_i64[0];
            if ( v98 )
              v35[1].m128i_i64[0] = v99;
            else
              *(_QWORD *)(v35[1].m128i_i64[0] + 24) = v99;
            v100 = _mm_loadu_si128((const __m128i *)(a4 + 16));
            *(__m128i *)v146.m256i_i8 = _mm_loadu_si128((const __m128i *)a4);
            *(__m128i *)&v146.m256i_u64[2] = v100;
            if ( !v88 )
            {
LABEL_231:
              v8 = v145;
LABEL_39:
              if ( v141 )
                v23[25] &= ~1u;
              goto LABEL_15;
            }
            v124 = v23;
            v101 = 0;
            v102 = v97;
            v122 = v7;
            v103 = v35;
            while ( (unsigned int)sub_799B70(a1, (__int64)&v147, a3, (const __m128i *)&v146, v103, a6) )
            {
              v103->m128i_i64[0] += v102;
              ++v101;
              v146.m256i_i64[0] += v102;
              if ( v88 == v101 )
              {
                v23 = v124;
                v7 = v122;
                v6 = a1;
                goto LABEL_231;
              }
            }
            v23 = v124;
            v7 = v122;
            v6 = a1;
LABEL_179:
            v145 = 0;
          }
          v8 = 0;
          goto LABEL_39;
        }
LABEL_208:
        v30 = v30 + 8 - v113;
        goto LABEL_209;
      }
      v23[25] &= ~1u;
      if ( (a2[3].m128i_i8[3] & 8) != 0 )
        return 0;
      return v8;
    case 7:
      v52 = (_BYTE *)a2[3].m128i_i64[1];
      if ( !v52 )
        goto LABEL_239;
      v53 = *(_QWORD *)v52;
      for ( i = *(_BYTE *)(*(_QWORD *)v52 + 140LL); i == 12; i = *(_BYTE *)(v53 + 140) )
        v53 = *(_QWORD *)(v53 + 160);
      if ( i == 8 && (v74 = v52[25], (v74 & 1) != 0) )
      {
        v75 = (_QWORD **)a2[3].m128i_i64[1];
        v52[25] = v74 & 0xFE;
        v76 = sub_786210(a1, v75, *(_QWORD *)a4, *(char **)(a4 + 24));
        v52[25] |= 1u;
        v8 = v76;
      }
      else
      {
        v8 = sub_786210(a1, (_QWORD **)a2[3].m128i_i64[1], *(_QWORD *)a4, *(char **)(a4 + 24));
      }
      goto LABEL_15;
    case 8:
      v8 = dword_4D041E0;
      if ( dword_4D041E0 )
      {
        v36 = (const __m128i *)a2[3].m128i_i64[1];
        v145 = 1;
        v142 = *(char **)(a4 + 24);
        v137 = *(__m128i **)a4;
        if ( (v7[4].m128i_i8[8] & 1) != 0 )
        {
          v37 = *(__int64 **)v7[4].m128i_i64[0];
          if ( !v36[11].m128i_i64[0] || !v37 )
            goto LABEL_63;
          v132 = v7;
          v38 = v36[11].m128i_i64[0];
          v121 = a1 + 72;
          v120 = (unsigned __int64)(a1 + 72) >> 3;
          while ( 1 )
          {
            v39 = v37[3];
            v40 = *(_QWORD *)(v39 + 120);
            for ( j = *(_BYTE *)(v40 + 140); j == 12; j = *(_BYTE *)(v40 + 140) )
              v40 = *(_QWORD *)(v40 + 160);
            for ( k = qword_4F08388 & (v39 >> 3); ; k = qword_4F08388 & (k + 1) )
            {
              v43 = qword_4F08380 + 16LL * k;
              if ( v39 == *(_QWORD *)v43 )
              {
                v44 = (__m128i *)((char *)v137 + *(unsigned int *)(v43 + 8));
                goto LABEL_53;
              }
              if ( !*(_QWORD *)v43 )
                break;
            }
            v44 = v137;
LABEL_53:
            if ( (unsigned __int8)(j - 9) <= 2u )
              v44->m128i_i64[0] = 0;
            v45 = *((_BYTE *)v37 + 32);
            if ( (v45 & 1) != 0 )
            {
              v46 = v37[1];
              if ( *(_BYTE *)(v46 + 48) <= 1u )
              {
                sub_7790A0(v6, v44, v40, (__int64)v142);
                v49 = v145;
              }
              else
              {
                v47 = *(_DWORD *)(v6 + 40);
                v147.m128i_i64[0] = (__int64)v44;
                v147.m128i_i32[2] = 0;
                v147.m128i_i32[3] = v47;
                v148.m128i_i64[1] = (__int64)v142;
                if ( !(unsigned int)sub_79B7D0(v6, v46, a3, &v147, 0, 0) )
                  goto LABEL_105;
                v48 = -(((unsigned int)((_DWORD)v44 - (_DWORD)v142) >> 3) + 10);
                v142[v48] |= 1 << (((_BYTE)v44 - (_BYTE)v142) & 7);
                v49 = v145;
              }
              goto LABEL_59;
            }
            if ( (v45 & 2) == 0 )
            {
              v59 = v37[1];
              if ( v59 )
              {
                if ( !v37[2] || (*(_BYTE *)(v59 + 172) & 1) != 0 )
                  break;
              }
            }
            v60 = *(_QWORD *)(v38 + 176);
            if ( *(_BYTE *)(v60 + 48) == 7 && !*(_QWORD *)(v60 + 56) )
            {
              v77 = v37[2];
              for ( m = qword_4F08388 & (v77 >> 3); ; m = qword_4F08388 & (m + 1) )
              {
                v79 = qword_4F08380 + 16LL * m;
                if ( v77 == *(_QWORD *)v79 )
                {
                  v80 = *(_DWORD *)(v79 + 8);
                  goto LABEL_144;
                }
                if ( !*(_QWORD *)v79 )
                  break;
              }
              v80 = 0;
LABEL_144:
              v81 = *(_DWORD *)(v6 + 8);
              v82 = v81 & v120;
              v83 = (__int64 *)(*(_QWORD *)v6 + 16LL * (v81 & v120));
              v84 = *v83;
              if ( v121 != *v83 )
              {
                while ( v84 )
                {
                  v82 = v81 & (v82 + 1);
                  v83 = (__int64 *)(*(_QWORD *)v6 + 16LL * v82);
                  v84 = *v83;
                  if ( v121 == *v83 )
                    goto LABEL_150;
                }
LABEL_148:
                v145 = 0;
                v7 = v132;
                if ( (*(_BYTE *)(v6 + 132) & 0x20) == 0 )
                {
                  sub_6855B0(0xAA1u, a3, (_QWORD *)(v6 + 96));
                  sub_770D30(v6);
                }
LABEL_63:
                v50 = -(((unsigned int)((_DWORD)v137 - (_DWORD)v142) >> 3) + 10);
                v142[v50] |= 1 << (((_BYTE)v137 - (_BYTE)v142) & 7);
                v8 = v145;
                goto LABEL_15;
              }
LABEL_150:
              v85 = (_QWORD *)v83[1];
              if ( !v85 )
                goto LABEL_148;
              if ( !(unsigned int)sub_778F10(
                                    v6,
                                    *(_QWORD *)(v77 + 120),
                                    a3,
                                    (_QWORD *)(*v85 + v80),
                                    v85[3],
                                    v44,
                                    (__int64)v142) )
              {
LABEL_105:
                v7 = v132;
                goto LABEL_106;
              }
              if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v77 + 120) + 140LL) - 9) <= 2u )
              {
LABEL_102:
                v44->m128i_i64[0] = 0;
                v49 = v145;
                goto LABEL_59;
              }
            }
            else
            {
              v61 = *(_DWORD *)(v6 + 40);
              v147.m128i_i64[0] = (__int64)v44;
              v148.m128i_i64[1] = (__int64)v142;
              v147.m128i_i32[2] = 0;
              v147.m128i_i32[3] = v61;
              if ( !(unsigned int)sub_79B7D0(v6, v60, a3, &v147, 0, 0) )
                goto LABEL_105;
              v62 = -(((unsigned int)((_DWORD)v44 - (_DWORD)v142) >> 3) + 10);
              v142[v62] |= 1 << (((_BYTE)v44 - (_BYTE)v142) & 7);
              if ( (unsigned __int8)(*(_BYTE *)(v40 + 140) - 9) <= 2u )
                goto LABEL_102;
            }
LABEL_111:
            v49 = v145;
LABEL_59:
            v37 = (__int64 *)*v37;
            v38 = *(_QWORD *)(v38 + 120);
            if ( !v49 || !v37 || !v38 )
            {
LABEL_62:
              v7 = v132;
              goto LABEL_63;
            }
          }
          v65 = *(_QWORD *)(v59 + 120);
          v66 = *(_BYTE *)(v65 + 140);
          v125 = v65;
          if ( v66 == 12 )
          {
            do
            {
              v65 = *(_QWORD *)(v65 + 160);
              v66 = *(_BYTE *)(v65 + 140);
            }
            while ( v66 == 12 );
          }
          else
          {
            v65 = *(_QWORD *)(v59 + 120);
          }
          v67 = *(_DWORD *)(v6 + 8);
          v68 = v67 & (v59 >> 3);
          v123 = *(_QWORD *)(v38 + 176);
          v69 = (__int64 *)(*(_QWORD *)v6 + 16LL * v68);
          v70 = *v69;
          if ( *v69 == v59 )
          {
LABEL_135:
            v72 = (__m128i *)v69[1];
            if ( v66 != 6 )
              goto LABEL_136;
            if ( (*(_BYTE *)(v65 + 168) & 1) != 0 )
            {
              v125 = *(_QWORD *)(v65 + 160);
              if ( v72 )
              {
                if ( !v145 )
                  goto LABEL_62;
                goto LABEL_187;
              }
              goto LABEL_221;
            }
            if ( (*(_BYTE *)(v59 + 172) & 1) == 0 )
            {
LABEL_136:
              if ( v72 )
              {
                if ( !v145 )
                  goto LABEL_62;
                goto LABEL_125;
              }
              goto LABEL_181;
            }
            if ( (v45 & 8) != 0 )
            {
              v71 = v145;
              if ( v72 )
              {
                if ( !v145 )
                  goto LABEL_62;
                goto LABEL_125;
              }
              goto LABEL_123;
            }
            v125 = *(_QWORD *)(v65 + 160);
            if ( v72 )
            {
              if ( !v145 )
                goto LABEL_62;
              goto LABEL_188;
            }
          }
          else
          {
            while ( v70 )
            {
              v68 = v67 & (v68 + 1);
              v69 = (__int64 *)(*(_QWORD *)v6 + 16LL * v68);
              v70 = *v69;
              if ( v59 == *v69 )
                goto LABEL_135;
            }
            if ( v66 != 6 )
              goto LABEL_181;
            if ( (*(_BYTE *)(v65 + 168) & 1) != 0 )
            {
              v125 = *(_QWORD *)(v65 + 160);
LABEL_221:
              v104 = 1;
LABEL_182:
              if ( (v37[4] & 8) != 0 )
              {
                v105 = v145;
                v72 = 0;
                goto LABEL_184;
              }
LABEL_197:
              v110 = v125;
              if ( (*(_BYTE *)(v125 + 140) & 0xFB) == 8 )
              {
                v117 = v37[1];
                v128 = v104;
                v111 = sub_8D4C10(v110, dword_4F077C4 != 2);
                v104 = v128;
                v59 = v117;
                if ( (v111 & 2) != 0 )
                {
                  v7 = v132;
                  if ( (*(_BYTE *)(v6 + 132) & 0x20) == 0 )
                  {
                    sub_6855B0(0xAC0u, a3, (_QWORD *)(v6 + 96));
                    sub_770D30(v6);
                  }
                  goto LABEL_106;
                }
              }
              v118 = v59;
              v119 = v104;
              v129 = (const __m128i *)sub_6EA7C0(v59);
              if ( !v129 )
              {
                v7 = v132;
                v114 = *(_BYTE *)(v6 + 132) & 0x20;
                if ( (*(_BYTE *)(v118 + 172) & 1) != 0 )
                {
                  if ( !v114 )
                  {
                    sub_6855B0(0xACFu, a3, (_QWORD *)(v6 + 96));
                    sub_770D30(v6);
                  }
                  goto LABEL_106;
                }
                if ( *(_QWORD *)v118 )
                {
                  if ( !v114 )
                  {
                    sub_686E10(0xA81u, a3, *(_QWORD *)v118, (_QWORD *)(v6 + 96));
                    sub_770D30(v6);
                  }
                  goto LABEL_106;
                }
                if ( !v114 )
                {
LABEL_236:
                  sub_6855B0(0xA8Du, a3, (_QWORD *)(v6 + 96));
                  sub_770D30(v6);
                }
LABEL_106:
                v145 = 0;
                goto LABEL_63;
              }
              v72 = (__m128i *)sub_77A250(v6, v118, &v145);
              v104 = v119;
              if ( !v145 )
                goto LABEL_62;
              v112 = v129[10].m128i_i8[13];
              if ( v112 == 1 )
              {
                if ( (v129[10].m128i_i8[8] & 8) == 0 )
                {
                  if ( (v129[10].m128i_i8[11] & 4) != 0 )
                    goto LABEL_105;
                  goto LABEL_204;
                }
              }
              else if ( v112 == 3 )
              {
                if ( (v129[10].m128i_i8[11] & 4) != 0 )
                  goto LABEL_105;
LABEL_204:
                *v72 = _mm_loadu_si128(v129 + 11);
                v145 = 1;
LABEL_185:
                if ( !v104 )
                  goto LABEL_125;
                v45 = *((_BYTE *)v37 + 32);
LABEL_187:
                if ( (v45 & 8) == 0 )
                {
LABEL_188:
                  *(__m128i *)v146.m256i_i8 = _mm_loadu_si128(v72);
                  *(__m128i *)&v146.m256i_u64[2] = _mm_loadu_si128(v72 + 1);
                  if ( (v146.m256i_i8[8] & 1) != 0 )
                  {
                    v7 = v132;
                    if ( (*(_BYTE *)(v6 + 132) & 0x20) == 0 )
                      goto LABEL_236;
                    goto LABEL_106;
                  }
                  v72 = (__m128i *)v146.m256i_i64[0];
                  if ( (v146.m256i_i8[8] & 0x20) == 0
                    && (*(_BYTE *)(v146.m256i_i64[3] - 9) & 1) == 0
                    && ((unsigned __int8)(1 << ((v146.m256i_i8[0] - v146.m256i_i8[24]) & 7))
                      & *(_BYTE *)(v146.m256i_i64[3]
                                 + -(((unsigned int)(v146.m256i_i32[0] - v146.m256i_i32[6]) >> 3) + 10))) == 0 )
                  {
                    v7 = v132;
                    sub_770DD0(0xABFu, a3, v6);
                    v145 = 0;
                    goto LABEL_63;
                  }
LABEL_126:
                  v73 = *(_BYTE *)(v123 + 48);
                  switch ( v73 )
                  {
                    case 7:
                      if ( !(unsigned int)sub_778F10(v6, v40, a3, v72, v146.m256i_i64[3], v44, (__int64)v142) )
                        goto LABEL_105;
                      break;
                    case 5:
                      v63 = *(_DWORD *)(v6 + 40);
                      v147.m128i_i64[0] = (__int64)v44;
                      v147.m128i_i32[3] = v63;
                      v147.m128i_i32[2] = 0;
                      v148.m128i_i64[1] = (__int64)v142;
                      if ( !(unsigned int)sub_799B70(v6, v123, a3, &v147, (const __m128i *)&v146, 0) )
                        goto LABEL_105;
                      break;
                    case 3:
                      if ( !(unsigned int)sub_786210(v6, *(_QWORD ***)(v123 + 56), (unsigned __int64)v44, v142) )
                        goto LABEL_105;
                      break;
                    default:
                      v7 = v132;
                      if ( v73 && (v73 != 2 || *(_BYTE *)(*(_QWORD *)(v123 + 56) + 173LL)) )
LABEL_239:
                        sub_721090();
                      *(_BYTE *)(v6 + 132) |= 0x40u;
                      v145 = 0;
                      goto LABEL_63;
                  }
                  if ( !v145 )
                    goto LABEL_62;
                  v64 = -(((unsigned int)((_DWORD)v44 - (_DWORD)v142) >> 3) + 10);
                  v142[v64] |= 1 << (((_BYTE)v44 - (_BYTE)v142) & 7);
                  if ( (unsigned __int8)(*(_BYTE *)(v40 + 140) - 9) <= 2u )
                    goto LABEL_102;
                  goto LABEL_111;
                }
LABEL_125:
                v146.m256i_i64[0] = (__int64)v72;
                v146.m256i_i64[3] = (__int64)v72;
                *(_OWORD *)&v146.m256i_u64[1] = 0;
                goto LABEL_126;
              }
              v116 = v129;
              v130 = v72;
              v105 = sub_79CCD0(v6, v116, v72, v72, 0);
              v72 = v130;
              v104 = v119;
              v145 = v105;
LABEL_184:
              if ( !v105 )
                goto LABEL_62;
              goto LABEL_185;
            }
            if ( (*(_BYTE *)(v59 + 172) & 1) == 0 )
            {
LABEL_181:
              v104 = 0;
              goto LABEL_182;
            }
            if ( (v37[4] & 8) != 0 )
            {
              v71 = v145;
LABEL_123:
              if ( !v71 )
                goto LABEL_62;
              v72 = 0;
              goto LABEL_125;
            }
            v125 = *(_QWORD *)(v65 + 160);
          }
          v104 = 1;
          goto LABEL_197;
        }
        v57 = v36[10].m128i_i8[13];
        if ( v57 == 1 )
        {
          if ( (v36[10].m128i_i8[8] & 8) == 0 )
          {
            v58 = 0;
            if ( (v36[10].m128i_i8[11] & 4) != 0 )
              goto LABEL_93;
            goto LABEL_92;
          }
        }
        else if ( v57 == 3 )
        {
          v58 = 0;
          if ( (v36[10].m128i_i8[11] & 4) != 0 )
          {
LABEL_93:
            v145 = v58;
            goto LABEL_63;
          }
LABEL_92:
          *v137 = _mm_loadu_si128(v36 + 11);
          v58 = 1;
          goto LABEL_93;
        }
        v58 = sub_79CCD0(a1, v36, v137, v142, 0);
        goto LABEL_93;
      }
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        return 0;
      sub_6855B0(0xB26u, a3, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v8;
    case 9:
      v10 = (const __m128i *)a2[3].m128i_i64[1];
      if ( !v10 )
        return 0;
      v11 = *(__m128i **)a4;
      v51 = v10[10].m128i_i8[13];
      if ( v51 == 1 )
      {
        if ( (v10[10].m128i_i8[8] & 8) == 0 )
        {
          if ( (v10[10].m128i_i8[11] & 4) == 0 )
          {
            *v11 = _mm_loadu_si128(v10 + 11);
            if ( (v7[3].m128i_i8[3] & 8) != 0 )
              goto LABEL_69;
            return 1;
          }
          return 0;
        }
      }
      else if ( v51 == 3 )
      {
        if ( (v10[10].m128i_i8[11] & 4) == 0 )
        {
          *v11 = _mm_loadu_si128(v10 + 11);
          if ( (v7[3].m128i_i8[3] & 8) != 0 )
          {
LABEL_69:
            v8 = 1;
            goto LABEL_17;
          }
          return 1;
        }
        return 0;
      }
LABEL_75:
      v8 = sub_79CCD0(a1, v10, v11, *(_QWORD *)(a4 + 24), a5);
LABEL_15:
      if ( (v7[3].m128i_i8[3] & 8) == 0 )
        return v8;
      if ( !v8 )
        return 0;
LABEL_17:
      v15 = *(_DWORD *)(v6 + 8);
      v16 = *(_QWORD *)v6;
      v17 = v15 & ((unsigned __int64)v7 >> 3);
      v18 = v17;
      v19 = (__m128i *)(*(_QWORD *)v6 + 16LL * v17);
      v20 = (const __m128i *)v19->m128i_i64[0];
      if ( v19->m128i_i64[0] )
      {
        do
        {
          if ( v20 == v7 )
          {
            *(_QWORD *)(v16 + 16 * v18 + 8) = *(_QWORD *)a4;
            return v8;
          }
          v17 = v15 & (v17 + 1);
          v18 = v17;
          v21 = (const __m128i **)(v16 + 16LL * v17);
          v20 = *v21;
        }
        while ( *v21 );
        *(__m128i *)v21 = _mm_loadu_si128(v19);
        *v21 = v7;
        v21[1] = *(const __m128i **)a4;
      }
      else
      {
        v19->m128i_i64[0] = (__int64)v7;
        v19->m128i_i64[1] = *(_QWORD *)a4;
      }
      v55 = *(_DWORD *)(v6 + 12) + 1;
      *(_DWORD *)(v6 + 12) = v55;
      if ( 2 * v55 > v15 )
        sub_7704A0(v6);
      return v8;
    default:
      goto LABEL_239;
  }
}
