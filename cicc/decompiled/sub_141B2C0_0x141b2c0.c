// Function: sub_141B2C0
// Address: 0x141b2c0
//
__int64 __fastcall sub_141B2C0(
        __int64 a1,
        __m128i *a2,
        unsigned __int8 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        unsigned __int8 a8)
{
  __int64 v12; // rax
  char v13; // di
  int v14; // edi
  __int64 v15; // r8
  int v16; // esi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r10
  _QWORD *v20; // rbx
  _QWORD *v21; // r15
  __int64 v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // r13
  char v25; // al
  __int64 v26; // rcx
  unsigned __int8 v28; // al
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdi
  _QWORD *v32; // rdi
  __m128i v33; // xmm0
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 *v40; // rdi
  char v42; // al
  _BYTE *v43; // rdx
  __int64 v44; // rcx
  char v45; // al
  char v46; // al
  char v47; // al
  __int64 v48; // rax
  unsigned __int16 v49; // ax
  __int64 v50; // rdx
  __int64 v51; // rcx
  char v52; // al
  char v53; // al
  char v54; // al
  __int64 *v55; // r15
  __int64 *v56; // r12
  _QWORD *v57; // rdi
  int v58; // esi
  unsigned int v59; // edx
  _QWORD *v60; // rax
  __int64 v61; // r8
  __int64 v62; // rbx
  __int64 v63; // r14
  char v64; // cl
  unsigned int v65; // esi
  unsigned int v66; // edx
  int v67; // edi
  unsigned int v68; // r8d
  unsigned int v69; // edx
  int v70; // r8d
  unsigned int v71; // r10d
  __int64 v72; // rdi
  __int64 *v73; // rdx
  char v74; // al
  char v75; // al
  unsigned __int16 v76; // dx
  int *v77; // rax
  int v78; // edi
  int v79; // eax
  int v80; // r10d
  _QWORD *v81; // r9
  int v82; // ecx
  __int64 v83; // rsi
  int v84; // ecx
  unsigned int v85; // edx
  __int64 v86; // rdi
  int v87; // ecx
  __int64 v88; // rsi
  int v89; // ecx
  unsigned int v90; // edx
  __int64 v91; // rdi
  int v92; // r9d
  _QWORD *v93; // r8
  __int64 v94; // rdx
  int v95; // ecx
  __int64 *v96; // rbx
  char v97; // al
  char v98; // al
  int v99; // r9d
  __int64 v100; // [rsp+10h] [rbp-360h]
  unsigned __int8 v102; // [rsp+2Bh] [rbp-345h]
  bool v103; // [rsp+2Ch] [rbp-344h]
  char v105; // [rsp+2Eh] [rbp-342h]
  char v106; // [rsp+2Fh] [rbp-341h]
  __m128i v109; // [rsp+40h] [rbp-330h] BYREF
  __m128i v110; // [rsp+50h] [rbp-320h]
  __int64 v111; // [rsp+60h] [rbp-310h]
  __m128i v112; // [rsp+70h] [rbp-300h] BYREF
  __m128i v113; // [rsp+80h] [rbp-2F0h]
  __int64 v114; // [rsp+90h] [rbp-2E0h]
  _QWORD *v115; // [rsp+A0h] [rbp-2D0h] BYREF
  __int64 v116; // [rsp+A8h] [rbp-2C8h]
  _QWORD *v117; // [rsp+B0h] [rbp-2C0h]
  __int64 v118; // [rsp+B8h] [rbp-2B8h]
  __int64 *v119; // [rsp+C0h] [rbp-2B0h]
  __int64 v120; // [rsp+C8h] [rbp-2A8h]
  _BYTE v121[64]; // [rsp+D0h] [rbp-2A0h] BYREF
  int v122; // [rsp+110h] [rbp-260h] BYREF
  char v123; // [rsp+118h] [rbp-258h]
  __int64 v124; // [rsp+120h] [rbp-250h]

  if ( !a7 )
  {
    v77 = (int *)sub_16D40F0(qword_4FBB410);
    if ( v77 )
      v78 = *v77;
    else
      v78 = qword_4FBB410[2];
    v79 = 32 * dword_4F994A0;
    if ( v78 <= 2 )
      v79 = 5 * dword_4F994A0;
    v122 = v79;
    return sub_141B2C0(a1, (_DWORD)a2, a3, (_DWORD)a4, a5, a6, (__int64)&v122, a8);
  }
  v103 = 0;
  v102 = a3 & (a6 != 0);
  if ( v102 && *(_BYTE *)(a6 + 16) == 54 && (*(_QWORD *)(a6 + 48) || *(__int16 *)(a6 + 18) < 0) )
    v103 = sub_1625790(a6, 6) != 0;
  v12 = sub_157EB90(a5);
  v100 = sub_1632FA0(v12);
  sub_143ACA0(&v122, a5);
  v13 = *(_BYTE *)(a1 + 472);
  v112.m128i_i64[0] = a5;
  v106 = a8 & byte_4F99660;
  v14 = v13 & 1;
  if ( v14 )
  {
    v15 = a1 + 480;
    v16 = 3;
  }
  else
  {
    v35 = *(_DWORD *)(a1 + 488);
    v15 = *(_QWORD *)(a1 + 480);
    if ( !v35 )
    {
      v69 = *(_DWORD *)(a1 + 472);
      ++*(_QWORD *)(a1 + 464);
      v18 = 0;
      v70 = (v69 >> 1) + 1;
LABEL_118:
      v71 = 3 * v35;
      goto LABEL_119;
    }
    v16 = v35 - 1;
  }
  v17 = v16 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
  v18 = (__int64 *)(v15 + 88LL * v17);
  v19 = *v18;
  if ( a5 == *v18 )
    goto LABEL_9;
  v95 = 1;
  v96 = 0;
  while ( v19 != -8 )
  {
    if ( !v96 && v19 == -16 )
      v96 = v18;
    v17 = v16 & (v95 + v17);
    v18 = (__int64 *)(v15 + 88LL * v17);
    v19 = *v18;
    if ( a5 == *v18 )
      goto LABEL_9;
    ++v95;
  }
  v69 = *(_DWORD *)(a1 + 472);
  v71 = 12;
  v35 = 4;
  if ( v96 )
    v18 = v96;
  ++*(_QWORD *)(a1 + 464);
  v70 = (v69 >> 1) + 1;
  if ( !(_BYTE)v14 )
  {
    v35 = *(_DWORD *)(a1 + 488);
    goto LABEL_118;
  }
LABEL_119:
  if ( v71 <= 4 * v70 )
  {
    v35 *= 2;
    goto LABEL_176;
  }
  v72 = a5;
  if ( v35 - *(_DWORD *)(a1 + 476) - v70 <= v35 >> 3 )
  {
LABEL_176:
    sub_141AF30(a1 + 464, v35);
    sub_1414980(a1 + 464, v112.m128i_i64, &v115);
    v18 = v115;
    v72 = v112.m128i_i64[0];
    v69 = *(_DWORD *)(a1 + 472);
  }
  *(_DWORD *)(a1 + 472) = (2 * (v69 >> 1) + 2) | v69 & 1;
  if ( *v18 != -8 )
    --*(_DWORD *)(a1 + 476);
  *v18 = v72;
  v73 = v18 + 3;
  v18[1] = 0;
  v18[2] = 1;
  do
  {
    if ( v73 )
      *v73 = -8;
    v73 += 2;
  }
  while ( v18 + 11 != v73 );
LABEL_9:
  v116 = a5;
  v20 = a4;
  v115 = v18 + 1;
  v119 = (__int64 *)v121;
  v118 = 0;
  v120 = 0x800000000LL;
  v105 = 0;
  v117 = a4;
  v21 = a4;
  v22 = a1;
  do
  {
    while ( 1 )
    {
      do
      {
LABEL_10:
        v23 = *(_QWORD **)(a5 + 48);
        if ( v106 )
        {
          if ( v117 == v23 )
            goto LABEL_178;
          v24 = *(_QWORD *)(sub_1416A50((__int64)&v115) + 16);
          if ( v24 )
            v24 -= 24;
          if ( byte_4F99580 )
          {
            do
              v21 = (_QWORD *)(*v21 & 0xFFFFFFFFFFFFFFF8LL);
            while ( v21 != v117 && v21 != *(_QWORD **)(a5 + 48) );
          }
          v25 = *(_BYTE *)(v24 + 16);
          if ( v25 == 77 )
            goto LABEL_44;
        }
        else
        {
          if ( v20 == v23 )
          {
            v23 = v117;
LABEL_178:
            v94 = *(_QWORD *)(*(_QWORD *)(a5 + 56) + 80LL);
            if ( !v94 || (v39 = 0x4000000000000003LL, a5 != v94 - 24) )
              v39 = 0x2000000000000003LL;
            goto LABEL_35;
          }
          v20 = (_QWORD *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
          if ( !v20 )
            BUG();
          v25 = *((_BYTE *)v20 - 8);
          v24 = (__int64)(v20 - 3);
          if ( v25 == 77 )
          {
LABEL_44:
            v105 = 1;
            break;
          }
        }
        if ( v25 != 78 )
          break;
        v48 = *(_QWORD *)(v24 - 24);
        if ( *(_BYTE *)(v48 + 16) )
          break;
      }
      while ( (*(_BYTE *)(v48 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v48 + 36) - 35) <= 3 );
      v26 = (__int64)a7;
      if ( (*a7)-- == 1 )
      {
        v23 = v117;
        v39 = 0x6000000000000003LL;
        goto LABEL_35;
      }
      v28 = *(_BYTE *)(v24 + 16);
      if ( v28 <= 0x17u )
        break;
      switch ( v28 )
      {
        case 'N':
          v36 = *(_QWORD *)(v24 - 24);
          if ( *(_BYTE *)(v36 + 16) || (*(_BYTE *)(v36 + 33) & 0x20) == 0 || *(_DWORD *)(v36 + 36) != 117 )
          {
LABEL_20:
            if ( !(unsigned __int8)sub_140AFC0(v24, *(_QWORD **)(v22 + 272), 0) )
              goto LABEL_23;
            goto LABEL_21;
          }
          v37 = *(_QWORD *)(v22 + 256);
          v38 = *(_QWORD *)(v24 + 24 * (1LL - (*(_DWORD *)(v24 + 20) & 0xFFFFFFF)));
          v112.m128i_i64[1] = -1;
          v113 = 0u;
          v112.m128i_i64[0] = v38;
          v114 = 0;
          if ( (unsigned __int8)sub_134CB50(v37, (__int64)&v112, (__int64)a2) == 3 )
            goto LABEL_33;
          break;
        case '6':
          if ( (*(_BYTE *)(v24 + 18) & 1) != 0 )
          {
            if ( !a6 )
              goto LABEL_69;
            v42 = *(_BYTE *)(a6 + 16);
            if ( ((unsigned __int8)(v42 - 54) <= 1u || v42 == 58) && (*(_BYTE *)(a6 + 18) & 1) != 0 )
              goto LABEL_69;
          }
          if ( !(unsigned __int8)sub_15F32D0(v24) )
            goto LABEL_61;
          v43 = byte_42880A0;
          if ( !byte_42880A0[8 * ((*(unsigned __int16 *)(v24 + 18) >> 7) & 7) + 1] )
            goto LABEL_61;
          v44 = a6;
          if ( !a6 )
            goto LABEL_69;
          v45 = *(_BYTE *)(a6 + 16);
          if ( v45 == 54 )
          {
            if ( (unsigned __int8)sub_15F32D0(a6) )
              goto LABEL_69;
            v44 = a6;
            v46 = *(_WORD *)(a6 + 18) & 1;
          }
          else
          {
            if ( v45 != 55 )
              goto LABEL_59;
            if ( (unsigned __int8)sub_15F32D0(a6) )
              goto LABEL_69;
            v46 = *(_WORD *)(a6 + 18) & 1;
          }
          if ( v46 )
            goto LABEL_69;
          v45 = *(_BYTE *)(a6 + 16);
LABEL_59:
          if ( (unsigned __int8)(v45 - 54) > 1u
            && ((unsigned __int8)sub_15F2ED0(a6) || (unsigned __int8)sub_15F3040(a6))
            || ((*(unsigned __int16 *)(v24 + 18) >> 7) & 7) != 2 )
          {
            goto LABEL_69;
          }
LABEL_61:
          sub_141EB40(&v112, v24, v43, v44, &v112);
          v47 = sub_134CB50(*(_QWORD *)(v22 + 256), (__int64)&v112, (__int64)a2);
          if ( a3 )
          {
            if ( v47 )
            {
              if ( v47 == 3 )
                goto LABEL_33;
              if ( (((unsigned __int8)v105 ^ 1) & v102) != 0
                && v47 == 2
                && *(_BYTE *)(*(_QWORD *)v24 + 8LL) == 13
                && *(_BYTE *)(a6 + 16) == 54
                && *(_BYTE *)(*(_QWORD *)a6 + 8LL) != 13 )
              {
                goto LABEL_69;
              }
            }
          }
          else if ( v47 && !(unsigned __int8)sub_134CBB0(*(_QWORD *)(v22 + 256), (__int64)&v112, 0) )
          {
LABEL_33:
            v39 = v24 | 2;
            goto LABEL_34;
          }
          break;
        case '7':
          v49 = *(_WORD *)(v24 + 18);
          if ( ((v49 >> 7) & 6) != 0 || (v49 & 1) != 0 )
          {
            if ( !(unsigned __int8)sub_15F32D0(v24) )
            {
              if ( (*(_BYTE *)(v24 + 18) & 1) == 0 )
                goto LABEL_84;
              if ( !a6 )
                goto LABEL_69;
LABEL_92:
              v53 = *(_BYTE *)(a6 + 16);
              if ( v53 == 54 )
              {
                if ( (unsigned __int8)sub_15F32D0(a6) )
                  goto LABEL_69;
                v54 = *(_WORD *)(a6 + 18) & 1;
                goto LABEL_95;
              }
              if ( v53 == 55 )
              {
                if ( (unsigned __int8)sub_15F32D0(a6) )
                  goto LABEL_69;
                v54 = *(_WORD *)(a6 + 18) & 1;
LABEL_95:
                if ( v54 )
                  goto LABEL_69;
                v53 = *(_BYTE *)(a6 + 16);
              }
              if ( (unsigned __int8)(v53 - 54) > 1u
                && ((unsigned __int8)sub_15F2ED0(a6) || (unsigned __int8)sub_15F3040(a6)) )
              {
                goto LABEL_69;
              }
              goto LABEL_84;
            }
            if ( !a6 )
              goto LABEL_69;
            v74 = *(_BYTE *)(a6 + 16);
            if ( v74 == 54 )
            {
              if ( (unsigned __int8)sub_15F32D0(a6) )
                goto LABEL_69;
              v75 = *(_WORD *)(a6 + 18) & 1;
              goto LABEL_132;
            }
            if ( v74 == 55 )
            {
              if ( (unsigned __int8)sub_15F32D0(a6) )
                goto LABEL_69;
              v75 = *(_WORD *)(a6 + 18) & 1;
LABEL_132:
              if ( v75 )
                goto LABEL_69;
              v74 = *(_BYTE *)(a6 + 16);
            }
            if ( (unsigned __int8)(v74 - 54) > 1u
              && ((unsigned __int8)sub_15F2ED0(a6) || (unsigned __int8)sub_15F3040(a6)) )
            {
              goto LABEL_69;
            }
            v76 = *(_WORD *)(v24 + 18);
            if ( ((v76 >> 7) & 7) != 2 )
              goto LABEL_69;
            if ( (v76 & 1) == 0 )
              goto LABEL_84;
            goto LABEL_92;
          }
LABEL_84:
          if ( (sub_134D0E0(*(_QWORD *)(v22 + 256), v24, a2, v26) & 3) != 0 )
          {
            sub_141EDF0(&v112, v24, v50, v51, &v112);
            v52 = sub_134CB50(*(_QWORD *)(v22 + 256), (__int64)&v112, (__int64)a2);
            if ( v52 )
            {
              if ( v52 == 3 )
                goto LABEL_33;
              if ( !v103 )
              {
LABEL_69:
                v39 = v24 | 1;
                goto LABEL_34;
              }
            }
          }
          break;
        default:
          goto LABEL_19;
      }
    }
LABEL_19:
    if ( v28 != 53 )
      goto LABEL_20;
LABEL_21:
    v30 = sub_14AD280(a2->m128i_i64[0], v100, 6);
    if ( v30 == v24 )
      goto LABEL_33;
    v31 = *(_QWORD *)(v22 + 256);
    v112.m128i_i64[0] = v30;
    v112.m128i_i64[1] = 1;
    v113 = 0u;
    v114 = 0;
    v109.m128i_i64[0] = v24;
    v109.m128i_i64[1] = 1;
    v110 = 0u;
    v111 = 0;
    if ( (unsigned __int8)sub_134CB50(v31, (__int64)&v109, (__int64)&v112) == 3 )
      goto LABEL_33;
LABEL_23:
    ;
  }
  while ( v103 || *(_BYTE *)(v24 + 16) == 57 && a3 && ((*(unsigned __int16 *)(v24 + 18) >> 1) & 0x7FFFBFFF) == 5 );
  v32 = *(_QWORD **)(v22 + 256);
  v33 = _mm_loadu_si128(a2 + 1);
  v34 = a2[2].m128i_i64[0];
  v112 = _mm_loadu_si128(a2);
  v114 = v34;
  v111 = v34;
  v113 = v33;
  v109 = v112;
  v110 = v33;
  switch ( *(_BYTE *)(v24 + 16) )
  {
    case 0x1D:
      v97 = sub_134F0E0(v32, v24 & 0xFFFFFFFFFFFFFFFBLL, (__int64)&v109);
      goto LABEL_196;
    case 0x21:
      v97 = sub_134D290((__int64)v32, v24, &v109);
      goto LABEL_196;
    case 0x36:
      v97 = sub_134D040((__int64)v32, v24, &v109, v29);
      goto LABEL_196;
    case 0x37:
      v97 = sub_134D0E0((__int64)v32, v24, &v109, v29);
      goto LABEL_196;
    case 0x39:
      v97 = sub_134D190((__int64)v32, v24, &v109);
      goto LABEL_196;
    case 0x3A:
      v97 = sub_134D2D0((__int64)v32, v24, &v109);
      goto LABEL_196;
    case 0x3B:
      v97 = sub_134D360((__int64)v32, v24, &v109);
      goto LABEL_196;
    case 0x4A:
      v97 = sub_134D250((__int64)v32, v24, &v109);
      goto LABEL_196;
    case 0x4E:
      v97 = sub_134F0E0(v32, v24 | 4, (__int64)&v109);
      goto LABEL_196;
    case 0x52:
      v97 = sub_134D1D0((__int64)v32, v24, &v109);
LABEL_196:
      if ( (v97 & 3) == 3 )
        v97 = sub_13510F0(*(_QWORD *)(v22 + 256), v24, a2, *(_QWORD *)(v22 + 280), (__int64)&v122);
      v98 = v97 | 4;
      switch ( v98 )
      {
        case 5:
          if ( !a3 )
            goto LABEL_69;
          goto LABEL_10;
        case 6:
          goto LABEL_69;
        case 4:
          goto LABEL_10;
      }
      v39 = v24 | 1;
LABEL_34:
      v23 = v117;
LABEL_35:
      v40 = v119;
      if ( *(_QWORD **)(v116 + 48) == v23 )
      {
        v55 = &v119[(unsigned int)v120];
        if ( v119 != v55 )
        {
          v56 = v119;
          while ( 1 )
          {
            v62 = (__int64)v115;
            v63 = *v56;
            v64 = v115[1] & 1;
            if ( v64 )
            {
              v57 = v115 + 2;
              v58 = 3;
            }
            else
            {
              v65 = *((_DWORD *)v115 + 6);
              v57 = (_QWORD *)v115[2];
              if ( !v65 )
              {
                v66 = *((_DWORD *)v115 + 2);
                ++*v115;
                v60 = 0;
                v67 = (v66 >> 1) + 1;
                goto LABEL_110;
              }
              v58 = v65 - 1;
            }
            v59 = v58 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v60 = &v57[2 * v59];
            v61 = *v60;
            if ( *v60 != v63 )
              break;
LABEL_105:
            ++v56;
            v60[1] = 0;
            if ( v55 == v56 )
            {
              v40 = v119;
              goto LABEL_36;
            }
          }
          v80 = 1;
          v81 = 0;
          while ( v61 != -8 )
          {
            if ( !v81 && v61 == -16 )
              v81 = v60;
            v59 = v58 & (v80 + v59);
            v60 = &v57[2 * v59];
            v61 = *v60;
            if ( v63 == *v60 )
              goto LABEL_105;
            ++v80;
          }
          v66 = *((_DWORD *)v115 + 2);
          v68 = 12;
          v65 = 4;
          if ( v81 )
            v60 = v81;
          ++*v115;
          v67 = (v66 >> 1) + 1;
          if ( !v64 )
          {
            v65 = *(_DWORD *)(v62 + 24);
LABEL_110:
            v68 = 3 * v65;
          }
          if ( 4 * v67 >= v68 )
          {
            sub_14163A0(v62, 2 * v65);
            if ( (*(_BYTE *)(v62 + 8) & 1) != 0 )
            {
              v83 = v62 + 16;
              v84 = 3;
            }
            else
            {
              v82 = *(_DWORD *)(v62 + 24);
              v83 = *(_QWORD *)(v62 + 16);
              if ( !v82 )
                goto LABEL_239;
              v84 = v82 - 1;
            }
            v85 = v84 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v60 = (_QWORD *)(v83 + 16LL * v85);
            v86 = *v60;
            if ( *v60 != v63 )
            {
              v99 = 1;
              v93 = 0;
              while ( v86 != -8 )
              {
                if ( !v93 && v86 == -16 )
                  v93 = v60;
                v85 = v84 & (v99 + v85);
                v60 = (_QWORD *)(v83 + 16LL * v85);
                v86 = *v60;
                if ( v63 == *v60 )
                  goto LABEL_161;
                ++v99;
              }
LABEL_170:
              if ( v93 )
                v60 = v93;
            }
          }
          else
          {
            if ( v65 - *(_DWORD *)(v62 + 12) - v67 > v65 >> 3 )
            {
LABEL_113:
              *(_DWORD *)(v62 + 8) = (2 * (v66 >> 1) + 2) | v66 & 1;
              if ( *v60 != -8 )
                --*(_DWORD *)(v62 + 12);
              *v60 = v63;
              v60[1] = 0;
              goto LABEL_105;
            }
            sub_14163A0(v62, v65);
            if ( (*(_BYTE *)(v62 + 8) & 1) != 0 )
            {
              v88 = v62 + 16;
              v89 = 3;
            }
            else
            {
              v87 = *(_DWORD *)(v62 + 24);
              v88 = *(_QWORD *)(v62 + 16);
              if ( !v87 )
              {
LABEL_239:
                *(_DWORD *)(v62 + 8) = (2 * (*(_DWORD *)(v62 + 8) >> 1) + 2) | *(_DWORD *)(v62 + 8) & 1;
                BUG();
              }
              v89 = v87 - 1;
            }
            v90 = v89 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v60 = (_QWORD *)(v88 + 16LL * v90);
            v91 = *v60;
            if ( v63 != *v60 )
            {
              v92 = 1;
              v93 = 0;
              while ( v91 != -8 )
              {
                if ( v91 == -16 && !v93 )
                  v93 = v60;
                v90 = v89 & (v92 + v90);
                v60 = (_QWORD *)(v88 + 16LL * v90);
                v91 = *v60;
                if ( v63 == *v60 )
                  goto LABEL_161;
                ++v92;
              }
              goto LABEL_170;
            }
          }
LABEL_161:
          v66 = *(_DWORD *)(v62 + 8);
          goto LABEL_113;
        }
      }
LABEL_36:
      if ( v40 != (__int64 *)v121 )
        _libc_free((unsigned __int64)v40);
      if ( (v123 & 1) == 0 )
        j___libc_free_0(v124);
      return v39;
    default:
      goto LABEL_10;
  }
}
