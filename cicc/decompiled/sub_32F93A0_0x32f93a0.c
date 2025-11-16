// Function: sub_32F93A0
// Address: 0x32f93a0
//
__int64 __fastcall sub_32F93A0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int16 *v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // r13
  __m128i v8; // xmm1
  int v9; // eax
  __int64 v10; // rcx
  int v11; // r13d
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r13
  __int64 v23; // rdi
  bool v24; // zf
  __m128i v25; // xmm3
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  int v32; // eax
  const __m128i *v33; // rax
  unsigned int v34; // eax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int16 v38; // si
  __int64 v39; // rax
  unsigned int v40; // r8d
  unsigned __int16 *v41; // rdx
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rax
  int v45; // r9d
  unsigned __int64 v46; // rax
  __int64 v47; // rcx
  unsigned int v48; // edx
  int v49; // r9d
  __int64 v50; // r9
  __int64 v51; // r8
  char v52; // al
  const __m128i *v53; // rax
  int v54; // r9d
  __int64 v55; // rcx
  unsigned int v56; // edx
  const __m128i *v57; // rax
  unsigned int v58; // r9d
  int v59; // r8d
  bool v60; // al
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned __int64 v63; // rsi
  int v64; // eax
  char v65; // al
  __int64 v66; // rax
  __int64 v67; // rax
  char v68; // al
  __int64 v69; // rdx
  __int128 v70; // rax
  __int64 v71; // r14
  __int64 v72; // rdi
  __int64 v73; // rax
  __int64 v74; // r14
  __int64 v75; // rdi
  int v76; // eax
  int v77; // edx
  __int64 v78; // r14
  __int64 v79; // rdx
  __m128i *v80; // rdx
  __int64 v81; // rdi
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // r8
  int v85; // eax
  unsigned int v86; // eax
  int v87; // r9d
  __int64 v88; // rdx
  bool v89; // cc
  _QWORD *v90; // rdx
  __int64 v91; // rax
  __int64 v92; // rdx
  __int128 v93; // rax
  int v94; // r9d
  __int64 v95; // r8
  const __m128i *v96; // roff
  int v97; // ebx
  __int64 v98; // rdx
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  int v103; // r9d
  int v104; // r8d
  bool v105; // al
  __int128 v106; // rax
  int v107; // r9d
  __int128 v108; // rax
  __int128 v109; // [rsp+0h] [rbp-170h]
  unsigned int v110; // [rsp+10h] [rbp-160h]
  unsigned int v111; // [rsp+10h] [rbp-160h]
  __int64 v112; // [rsp+10h] [rbp-160h]
  int v113; // [rsp+10h] [rbp-160h]
  __int64 v114; // [rsp+18h] [rbp-158h]
  int v115; // [rsp+18h] [rbp-158h]
  unsigned int v116; // [rsp+18h] [rbp-158h]
  unsigned int v117; // [rsp+18h] [rbp-158h]
  unsigned int v118; // [rsp+18h] [rbp-158h]
  int v119; // [rsp+18h] [rbp-158h]
  unsigned __int64 *v120; // [rsp+18h] [rbp-158h]
  unsigned int v121; // [rsp+20h] [rbp-150h]
  __int64 v122; // [rsp+20h] [rbp-150h]
  char v123; // [rsp+20h] [rbp-150h]
  __int128 v124; // [rsp+20h] [rbp-150h]
  __int128 v125; // [rsp+20h] [rbp-150h]
  __int64 v126; // [rsp+20h] [rbp-150h]
  unsigned __int64 v127; // [rsp+20h] [rbp-150h]
  __int64 v128; // [rsp+30h] [rbp-140h]
  __int64 v129; // [rsp+40h] [rbp-130h]
  unsigned __int16 v130; // [rsp+40h] [rbp-130h]
  unsigned __int64 v131; // [rsp+40h] [rbp-130h]
  unsigned __int16 v132; // [rsp+40h] [rbp-130h]
  int v133; // [rsp+40h] [rbp-130h]
  __m128i v134; // [rsp+50h] [rbp-120h] BYREF
  __int64 v135; // [rsp+60h] [rbp-110h]
  __int64 v136; // [rsp+68h] [rbp-108h]
  __m128i v137; // [rsp+70h] [rbp-100h] BYREF
  unsigned int v138; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v139; // [rsp+88h] [rbp-E8h]
  __int64 v140; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v141; // [rsp+98h] [rbp-D8h]
  __int64 v142; // [rsp+A0h] [rbp-D0h] BYREF
  int v143; // [rsp+A8h] [rbp-C8h]
  __int64 v144; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v145; // [rsp+B8h] [rbp-B8h]
  __int64 v146; // [rsp+C0h] [rbp-B0h]
  __int64 v147; // [rsp+C8h] [rbp-A8h]
  __int64 v148; // [rsp+D0h] [rbp-A0h]
  __int64 v149; // [rsp+D8h] [rbp-98h]
  __m128i v150; // [rsp+E0h] [rbp-90h] BYREF
  __m128i v151; // [rsp+F0h] [rbp-80h]
  __m128i v152; // [rsp+100h] [rbp-70h]
  __m128i v153; // [rsp+110h] [rbp-60h]
  __m128i v154; // [rsp+120h] [rbp-50h]
  __m128i v155; // [rsp+130h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *(_QWORD *)(v4 + 40);
  v7 = *((_QWORD *)v5 + 1);
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v9 = *v5;
  v10 = *(_QWORD *)(v6 + 104);
  v137 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  LOWORD(v6) = *(_WORD *)(v6 + 96);
  v139 = v7;
  LOWORD(v138) = v9;
  LOWORD(v140) = v6;
  v141 = v10;
  v134 = v8;
  if ( !(_WORD)v9 )
  {
    if ( !sub_30070B0((__int64)&v138) )
    {
      v150.m128i_i64[1] = v7;
      v150.m128i_i16[0] = 0;
      goto LABEL_11;
    }
    LOWORD(v9) = sub_3009970((__int64)&v138, a2, v29, v30, v31);
LABEL_10:
    v150.m128i_i16[0] = v9;
    v150.m128i_i64[1] = v16;
    if ( (_WORD)v9 )
      goto LABEL_4;
LABEL_11:
    v17 = sub_3007260((__int64)&v150);
    v11 = (unsigned __int16)v140;
    v146 = v17;
    v147 = v18;
    LODWORD(v129) = v17;
    if ( !(_WORD)v140 )
      goto LABEL_7;
    goto LABEL_12;
  }
  if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
  {
    LOWORD(v9) = word_4456580[v9 - 1];
    v16 = 0;
    goto LABEL_10;
  }
  v150.m128i_i16[0] = v9;
  v150.m128i_i64[1] = v7;
LABEL_4:
  if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    goto LABEL_173;
  v11 = (unsigned __int16)v140;
  v129 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v9 - 16];
  if ( !(_WORD)v140 )
  {
LABEL_7:
    if ( sub_30070B0((__int64)&v140) )
    {
      LOWORD(v11) = sub_3009970((__int64)&v140, a2, v12, v13, v14);
      goto LABEL_14;
    }
    goto LABEL_13;
  }
LABEL_12:
  if ( (unsigned __int16)(v11 - 17) > 0xD3u )
  {
LABEL_13:
    v15 = v141;
    goto LABEL_14;
  }
  v15 = 0;
  LOWORD(v11) = word_4456580[v11 - 1];
LABEL_14:
  LOWORD(v144) = v11;
  v145 = v15;
  if ( !(_WORD)v11 )
  {
    v148 = sub_3007260((__int64)&v144);
    LODWORD(v19) = v148;
    v149 = v20;
    goto LABEL_16;
  }
  if ( (_WORD)v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
LABEL_173:
    BUG();
  v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v11 - 16];
LABEL_16:
  v21 = *(_QWORD *)(a2 + 80);
  v142 = v21;
  if ( v21 )
    sub_B96E90((__int64)&v142, v21, 1);
  v22 = v137.m128i_i64[0];
  v23 = *a1;
  v24 = *(_DWORD *)(v137.m128i_i64[0] + 24) == 51;
  v143 = *(_DWORD *)(a2 + 72);
  if ( v24 )
  {
    v27 = sub_3400BD0(v23, 0, (unsigned int)&v142, v138, v139, 0, 0);
    goto LABEL_21;
  }
  v25 = _mm_load_si128(&v134);
  v150 = _mm_loadu_si128(&v137);
  v151 = v25;
  v26 = sub_3402EA0(v23, 222, (unsigned int)&v142, v138, v139, 0, (__int64)&v150, 2);
  if ( v26 )
  {
LABEL_20:
    v27 = v26;
    goto LABEL_21;
  }
  if ( (unsigned int)sub_33DF530(*a1, v137.m128i_i64[0], v137.m128i_i64[1], 0) <= (unsigned int)v19 )
  {
    v27 = v137.m128i_i64[0];
    goto LABEL_21;
  }
  v32 = *(_DWORD *)(v22 + 24);
  if ( v32 == 222 )
  {
    v122 = *(_QWORD *)(v22 + 40);
    if ( sub_3280B30(
           (__int64)&v140,
           *(unsigned __int16 *)(*(_QWORD *)(v122 + 40) + 96LL),
           *(_QWORD *)(*(_QWORD *)(v122 + 40) + 104LL)) )
    {
      v67 = sub_3406EB0(*a1, 222, (unsigned int)&v142, v138, v139, v49, *(_OWORD *)v122, *(_OWORD *)&v134);
      goto LABEL_118;
    }
LABEL_59:
    v50 = *a1;
    v51 = 1LL << ((unsigned __int8)v19 - 1);
    v150.m128i_i32[2] = v129;
    if ( (unsigned int)v129 > 0x40 )
    {
      v126 = v50;
      sub_C43690((__int64)&v150, 0, 0);
      v50 = v126;
      v51 = 1LL << ((unsigned __int8)v19 - 1);
      if ( v150.m128i_i32[2] > 0x40u )
      {
        *(_QWORD *)(v150.m128i_i64[0] + 8LL * ((unsigned int)(v19 - 1) >> 6)) |= 1LL << ((unsigned __int8)v19 - 1);
        goto LABEL_62;
      }
    }
    else
    {
      v150.m128i_i64[0] = 0;
    }
    v150.m128i_i64[0] |= v51;
LABEL_62:
    v52 = sub_33DD210(v50, v137.m128i_i64[0], v137.m128i_i64[1], &v150, 0);
    if ( v150.m128i_i32[2] > 0x40u && v150.m128i_i64[0] )
    {
      v123 = v52;
      j_j___libc_free_0_0(v150.m128i_u64[0]);
      v52 = v123;
    }
    if ( v52 )
    {
      v26 = sub_34070B0(*a1, v137.m128i_i64[0], v137.m128i_i64[1], &v142, (unsigned int)v140, v141);
      goto LABEL_20;
    }
    if ( (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
    {
      v27 = a2;
      goto LABEL_21;
    }
    v63 = a2;
    v26 = sub_32B3F40(a1, a2);
    if ( v26 )
      goto LABEL_20;
    if ( *(_DWORD *)(v22 + 24) != 192 )
      goto LABEL_99;
    v84 = *(_QWORD *)(*(_QWORD *)(v22 + 40) + 40LL);
    v85 = *(_DWORD *)(v84 + 24);
    if ( v85 != 11 && v85 != 35 )
      goto LABEL_111;
    v120 = *(unsigned __int64 **)(v22 + 40);
    v63 = (unsigned int)(v129 - v19);
    v127 = v120[5];
    v131 = v63;
    if ( sub_AAD8D0(*(_QWORD *)(v84 + 96) + 24LL, v63) )
      goto LABEL_111;
    v63 = *v120;
    v86 = sub_33D4D80(*a1, *v120, v120[1], 0);
    v88 = *(_QWORD *)(v127 + 96);
    v89 = *(_DWORD *)(v88 + 32) <= 0x40u;
    v90 = *(_QWORD **)(v88 + 24);
    if ( !v89 )
      v90 = (_QWORD *)*v90;
    if ( v131 - (unsigned __int64)v90 >= v86 )
    {
LABEL_99:
      v64 = *(_DWORD *)(v22 + 24);
      switch ( v64 )
      {
        case 298:
          v65 = (*(_BYTE *)(v22 + 33) >> 2) & 3;
          if ( v65 != 1 )
          {
            if ( v65 != 3
              || (*(_WORD *)(v22 + 32) & 0x380) != 0
              || !(unsigned __int8)sub_3286E00(&v137)
              || *(_WORD *)(v22 + 96) != (_WORD)v140
              || !(_WORD)v140 && *(_QWORD *)(v22 + 104) != v141 )
            {
              break;
            }
            v130 = v140;
            if ( *((_BYTE *)a1 + 33)
              || !(unsigned __int8)sub_3287C60(v22)
              || !v130
              || !(_WORD)v138
              || (*(_BYTE *)(a1[1] + 2 * (v130 + 274LL * (unsigned __int16)v138 + 71704) + 7) & 0xF) != 0 )
            {
              break;
            }
            *(_QWORD *)&v108 = sub_33F1B30(
                                 *a1,
                                 2,
                                 (unsigned int)&v142,
                                 v138,
                                 v139,
                                 *(_QWORD *)(v22 + 112),
                                 **(_QWORD **)(v22 + 40),
                                 *(_QWORD *)(*(_QWORD *)(v22 + 40) + 8LL),
                                 *(_QWORD *)(*(_QWORD *)(v22 + 40) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(v22 + 40) + 48LL),
                                 v140,
                                 v141);
            v150 = (__m128i)v108;
            v134.m128i_i64[0] = *((_QWORD *)&v108 + 1);
            v71 = v108;
LABEL_128:
            sub_32EB790((__int64)a1, a2, v150.m128i_i64, 1, 1);
            v72 = (__int64)a1;
            v27 = a2;
            sub_32EFDE0(v72, v22, v71, v134.m128i_i64[0], v71, 1, 1);
            goto LABEL_21;
          }
          if ( (*(_WORD *)(v22 + 32) & 0x380) != 0 )
            break;
          v132 = v140;
          if ( *(_WORD *)(v22 + 96) != (_WORD)v140 )
            break;
          if ( (_WORD)v140 )
          {
            if ( *((_BYTE *)a1 + 33) || !(unsigned __int8)sub_3287C60(v22) )
              goto LABEL_165;
          }
          else if ( *(_QWORD *)(v22 + 104) != v141 || *((_BYTE *)a1 + 33) || !(unsigned __int8)sub_3287C60(v22) )
          {
            break;
          }
          if ( (unsigned __int8)sub_3286E00(&v137) )
          {
LABEL_151:
            v91 = sub_33F1B30(
                    *a1,
                    2,
                    (unsigned int)&v142,
                    v138,
                    v139,
                    *(_QWORD *)(v22 + 112),
                    **(_QWORD **)(v22 + 40),
                    *(_QWORD *)(*(_QWORD *)(v22 + 40) + 8LL),
                    *(_QWORD *)(*(_QWORD *)(v22 + 40) + 40LL),
                    *(_QWORD *)(*(_QWORD *)(v22 + 40) + 48LL),
                    v140,
                    v141);
            v150.m128i_i64[1] = v92;
            v78 = v91;
            v134.m128i_i64[0] = v92;
            v80 = &v150;
            v150.m128i_i64[0] = v91;
            goto LABEL_135;
          }
LABEL_165:
          if ( !(_WORD)v138 )
            break;
          v63 = v132;
          if ( !v132 || (*(_BYTE *)(a1[1] + 2 * (v132 + 274LL * (unsigned __int16)v138 + 71704) + 7) & 0xF) != 0 )
            break;
          goto LABEL_151;
        case 362:
          if ( (_WORD)v140 != *(_WORD *)(v22 + 96) )
            goto LABEL_136;
          if ( !(_WORD)v140 && *(_QWORD *)(v22 + 104) != v141 )
            goto LABEL_116;
          v134.m128i_i32[0] = (unsigned __int16)v140;
          if ( !(unsigned __int8)sub_3286E00(&v137) || (v68 = *(_BYTE *)(v22 + 33), (v68 & 0xC) == 0) )
          {
LABEL_136:
            if ( (unsigned int)v19 > 0x10 )
              goto LABEL_116;
            goto LABEL_113;
          }
          if ( !v134.m128i_i16[0]
            || !(_WORD)v138
            || (*(_BYTE *)(a1[1] + 2 * (v134.m128i_u16[0] + 274LL * (unsigned __int16)v138 + 71704) + 7) & 0xF) != 0 )
          {
            goto LABEL_116;
          }
          v69 = *(_QWORD *)(v22 + 40);
          *(_QWORD *)&v70 = sub_33E8F60(
                              *a1,
                              v138,
                              v139,
                              (unsigned int)&v142,
                              *(_QWORD *)v69,
                              *(_QWORD *)(v69 + 8),
                              *(_QWORD *)(v69 + 40),
                              *(_QWORD *)(v69 + 48),
                              *(_OWORD *)(v69 + 80),
                              *(_OWORD *)(v69 + 120),
                              *(_OWORD *)(v69 + 160),
                              v140,
                              v141,
                              *(_QWORD *)(v22 + 112),
                              (*(_WORD *)(v22 + 32) >> 7) & 7,
                              2,
                              (v68 & 0x10) != 0);
          v150 = (__m128i)v70;
          v134.m128i_i64[0] = *((_QWORD *)&v70 + 1);
          v71 = v70;
          goto LABEL_128;
        case 364:
          v150.m128i_i64[0] = v22;
          v150.m128i_i32[2] = 0;
          if ( (unsigned __int8)sub_3286E00(&v150) )
          {
            if ( (_WORD)v140 == *(_WORD *)(v22 + 96) && ((_WORD)v140 || *(_QWORD *)(v22 + 104) == v141) )
            {
              v63 = v22;
              if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)a1[1] + 1584LL))(
                     a1[1],
                     v22,
                     0) )
              {
                v73 = *(_QWORD *)(v22 + 40);
                v74 = *a1;
                v75 = *a1;
                v150 = _mm_loadu_si128((const __m128i *)v73);
                v151 = _mm_loadu_si128((const __m128i *)(v73 + 40));
                v152 = _mm_loadu_si128((const __m128i *)(v73 + 80));
                v153 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v22 + 40) + 120LL));
                v154 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v22 + 40) + 160LL));
                v155 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v22 + 40) + 200LL));
                v128 = *(_QWORD *)(v22 + 112);
                v134.m128i_i32[0] = (*(_WORD *)(v22 + 32) >> 7) & 7;
                v76 = sub_33E5110(v75, v138, v139, 1, 0);
                v144 = sub_33E8420(
                         v74,
                         v76,
                         v77,
                         v140,
                         v141,
                         (unsigned int)&v142,
                         (__int64)&v150,
                         6,
                         v128,
                         v134.m128i_i16[0],
                         2,
                         v75);
                v78 = v144;
                v145 = v79;
                v134.m128i_i64[0] = v79;
                v80 = (__m128i *)&v144;
LABEL_135:
                sub_32EB790((__int64)a1, a2, v80->m128i_i64, 1, 1);
                sub_32EFDE0((__int64)a1, v22, v78, v134.m128i_i64[0], v78, 1, 1);
                v81 = (__int64)a1;
                v27 = a2;
                sub_32B3E80(v81, v78, 1, 0, v82, v83);
                goto LABEL_21;
              }
            }
          }
          break;
      }
LABEL_111:
      if ( (unsigned int)v19 <= 0x10 && *(_DWORD *)(v22 + 24) == 187 )
      {
        v63 = v22;
        *(_QWORD *)&v93 = sub_3283260(a1, v22, **(_QWORD **)(v22 + 40), *(_QWORD *)(*(_QWORD *)(v22 + 40) + 40LL), 0);
        if ( (_QWORD)v93 )
        {
          v67 = sub_3406EB0(*a1, 222, (unsigned int)&v142, v138, v139, v94, v93, *(_OWORD *)&v134);
          goto LABEL_118;
        }
      }
LABEL_113:
      if ( *(_DWORD *)(v22 + 24) == 161 )
      {
        if ( (unsigned __int8)sub_3286E00(&v137) )
        {
          v66 = **(_QWORD **)(v22 + 40);
          if ( (unsigned int)(*(_DWORD *)(v66 + 24) - 213) <= 2 )
          {
            v95 = *(_QWORD *)(*(_QWORD *)(v66 + 48) + 8LL);
            v96 = *(const __m128i **)(v66 + 40);
            v97 = **(unsigned __int16 **)(v66 + 48);
            v98 = v96->m128i_i64[0];
            v99 = v96->m128i_u32[2];
            v133 = v95;
            v134 = _mm_loadu_si128(v96);
            v100 = *(_QWORD *)(v98 + 48) + 16 * v99;
            LOWORD(v98) = *(_WORD *)v100;
            v101 = *(_QWORD *)(v100 + 8);
            v150.m128i_i16[0] = v98;
            v150.m128i_i64[1] = v101;
            v102 = sub_32844A0((unsigned __int16 *)&v150, v63);
            v104 = v133;
            if ( v102 == (unsigned int)v19 )
            {
              if ( !*((_BYTE *)a1 + 33) || (v105 = sub_328D6E0(a1[1], 0xD5u, v97), v104 = v133, v105) )
              {
                *(_QWORD *)&v106 = sub_33FAF80(*a1, 213, (unsigned int)&v142, v97, v104, v103, *(_OWORD *)&v134);
                v27 = sub_3406EB0(
                        *a1,
                        161,
                        (unsigned int)&v142,
                        v138,
                        v139,
                        v107,
                        v106,
                        *(_OWORD *)(*(_QWORD *)(v22 + 40) + 40LL));
                goto LABEL_21;
              }
            }
          }
        }
      }
LABEL_116:
      v27 = 0;
      goto LABEL_21;
    }
    v67 = sub_3406EB0(
            *a1,
            191,
            (unsigned int)&v142,
            v138,
            v139,
            v87,
            *(_OWORD *)*(_QWORD *)(v22 + 40),
            *(_OWORD *)(*(_QWORD *)(v22 + 40) + 40LL));
LABEL_118:
    v27 = v67;
    goto LABEL_21;
  }
  if ( (v32 & 0xFFFFFFFD) == 0xD5 )
  {
    v53 = *(const __m128i **)(v22 + 40);
    v124 = (__int128)_mm_loadu_si128(v53);
    if ( (unsigned int)v19 >= (unsigned int)sub_3263630(v53->m128i_i64[0], v53->m128i_u32[2])
      || (unsigned int)sub_33DF530(*a1, v124, *((_QWORD *)&v124 + 1), 0) <= (unsigned int)v19 )
    {
      if ( !*((_BYTE *)a1 + 33)
        || ((v55 = a1[1], v56 = 1, (_WORD)v138 == 1)
         || (_WORD)v138 && (v56 = (unsigned __int16)v138, *(_QWORD *)(v55 + 8LL * (unsigned __int16)v138 + 112)))
        && !*(_BYTE *)(v55 + 500LL * v56 + 6627) )
      {
        v61 = sub_33FAF80(*a1, 213, (unsigned int)&v142, v138, v139, v54, v124);
        goto LABEL_84;
      }
    }
    v32 = *(_DWORD *)(v22 + 24);
  }
  if ( (unsigned int)(v32 - 223) > 2 )
  {
LABEL_79:
    if ( v32 == 214 )
    {
      v57 = *(const __m128i **)(v22 + 40);
      v125 = (__int128)_mm_loadu_si128(v57);
      if ( sub_3263630(v57->m128i_i64[0], v57->m128i_u32[2]) == (unsigned int)v19 )
      {
        v58 = v138;
        v59 = v139;
        if ( !*((_BYTE *)a1 + 33)
          || (v111 = v138, v115 = v139, v60 = sub_328D6E0(a1[1], 0xD5u, v138), v59 = v115, v58 = v111, v60) )
        {
          v61 = sub_33FAF80(*a1, 213, (unsigned int)&v142, v58, v59, v58, v125);
LABEL_84:
          v27 = v61;
          goto LABEL_21;
        }
      }
    }
    goto LABEL_59;
  }
  v33 = *(const __m128i **)(v22 + 40);
  v110 = v33->m128i_u32[2];
  v114 = v33->m128i_i64[0];
  v109 = (__int128)_mm_loadu_si128(v33);
  v34 = sub_3263630(v33->m128i_i64[0], v110);
  v35 = v114;
  v36 = v110;
  v121 = v34;
  v37 = *(_QWORD *)(v22 + 48) + 16LL * v137.m128i_u32[2];
  v38 = *(_WORD *)v37;
  v39 = *(_QWORD *)(v37 + 8);
  v150.m128i_i16[0] = v38;
  v150.m128i_i64[1] = v39;
  if ( v38 )
  {
    v40 = word_4456340[v38 - 1];
  }
  else
  {
    v112 = v114;
    v117 = v36;
    v62 = sub_3007240((__int64)&v150);
    v35 = v112;
    v36 = v117;
    v136 = v62;
    v40 = v62;
  }
  v41 = (unsigned __int16 *)(*(_QWORD *)(v35 + 48) + 16 * v36);
  v42 = *v41;
  v43 = *((_QWORD *)v41 + 1);
  v150.m128i_i16[0] = v42;
  v150.m128i_i64[1] = v43;
  if ( (_WORD)v42 )
  {
    LODWORD(v44) = word_4456340[v42 - 1];
  }
  else
  {
    v116 = v40;
    v44 = sub_3007240((__int64)&v150);
    v40 = v116;
    v135 = v44;
  }
  v45 = *(_DWORD *)(v22 + 24);
  v150.m128i_i32[2] = v44;
  if ( (unsigned int)v44 > 0x40 )
  {
    v113 = v45;
    v118 = v40;
    sub_C43690((__int64)&v150, 0, 0);
    v45 = v113;
    v40 = v118;
  }
  else
  {
    v150.m128i_i64[0] = 0;
  }
  if ( v40 )
  {
    if ( v40 > 0x40 )
    {
      v119 = v45;
      sub_C43C90(&v150, 0, v40);
      v45 = v119;
    }
    else
    {
      v46 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v40);
      if ( v150.m128i_i32[2] > 0x40u )
        *(_QWORD *)v150.m128i_i64[0] |= v46;
      else
        v150.m128i_i64[0] |= v46;
    }
  }
  if ( (_DWORD)v19 != v121
    && (v45 == 225
     || (unsigned int)v19 <= v121 && (unsigned int)sub_33DF530(*a1, v109, *((_QWORD *)&v109 + 1), 0) > (unsigned int)v19)
    || *((_BYTE *)a1 + 33)
    && ((v47 = a1[1], v48 = 1, (_WORD)v138 != 1)
     && (!(_WORD)v138 || (v48 = (unsigned __int16)v138, !*(_QWORD *)(v47 + 8LL * (unsigned __int16)v138 + 112)))
     || *(_BYTE *)(v47 + 500LL * v48 + 6638)) )
  {
    if ( v150.m128i_i32[2] > 0x40u && v150.m128i_i64[0] )
      j_j___libc_free_0_0(v150.m128i_u64[0]);
    v32 = *(_DWORD *)(v22 + 24);
    goto LABEL_79;
  }
  v27 = sub_33FAF80(*a1, 224, (unsigned int)&v142, v138, v139, v45, v109);
  if ( v150.m128i_i32[2] > 0x40u && v150.m128i_i64[0] )
    j_j___libc_free_0_0(v150.m128i_u64[0]);
LABEL_21:
  if ( v142 )
    sub_B91220((__int64)&v142, v142);
  return v27;
}
