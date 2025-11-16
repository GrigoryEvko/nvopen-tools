// Function: sub_713ED0
// Address: 0x713ed0
//
__int64 __fastcall sub_713ED0(
        __int64 a1,
        const __m128i *a2,
        const __m128i *a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        int a7,
        int *a8,
        _DWORD *a9,
        _DWORD *a10,
        _DWORD *a11)
{
  unsigned __int8 v11; // r15
  const __m128i *v12; // r14
  const __m128i *v14; // r12
  char v16; // cl
  __int8 v17; // si
  __int8 v18; // al
  __int64 result; // rax
  __int8 v20; // r11
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdi
  int v30; // eax
  __int64 v31; // rax
  unsigned __int8 v32; // bl
  __m128i *v33; // rax
  __int64 v34; // rax
  __m128i v35; // xmm3
  __int64 v36; // rax
  __m128i v37; // xmm2
  __int64 v38; // rax
  __m128i v39; // xmm1
  int v40; // ecx
  const __m128i *v41; // rdx
  const __m128i *v42; // rdi
  __int64 nn; // rax
  unsigned __int8 v44; // bl
  __int32 v45; // r15d
  __int64 i1; // rax
  unsigned __int8 v47; // r10
  int v48; // eax
  int v49; // r15d
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  unsigned __int8 v53; // al
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  int v57; // ebx
  __int64 i2; // rax
  unsigned __int8 v59; // di
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rcx
  unsigned __int8 v65; // al
  __int64 i3; // rax
  unsigned __int8 v67; // di
  const __m128i *v68; // rax
  const __m128i *v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // r9
  unsigned __int64 v72; // r8
  int v73; // r14d
  __int64 v74; // r12
  __int64 v75; // rdx
  __int64 v76; // rdi
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rsi
  __int64 v83; // r15
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // rax
  __m128i v88; // xmm4
  __int64 v89; // rdx
  __int64 kk; // rax
  unsigned __int8 v91; // r15
  const __m128i *v92; // rax
  const __m128i *v93; // rax
  __m128i *v94; // rsi
  __m128i *v95; // rsi
  _OWORD *v96; // rcx
  int v97; // ebx
  int v98; // ebx
  __int64 n; // rax
  unsigned __int8 v100; // bl
  __int64 jj; // rax
  unsigned __int8 v102; // bl
  __int64 k; // rax
  unsigned __int8 v104; // bl
  __int32 v105; // ebx
  __int64 v106; // rax
  __m128i v107; // xmm5
  __int64 mm; // rax
  unsigned __int8 v109; // bl
  __int64 j; // rax
  unsigned __int8 v111; // bl
  __int64 v112; // rsi
  __int64 m; // rax
  unsigned __int8 v114; // bl
  __int64 v115; // rsi
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 ii; // rax
  bool v120; // zf
  unsigned __int8 v121; // bl
  __int64 i4; // rax
  int v123; // eax
  __int64 v124; // rdi
  __int8 v125; // al
  __int64 v126; // rcx
  const __m128i *v127; // rsi
  __int8 v128; // al
  __int64 v129; // rcx
  _DWORD *v130; // rdi
  const __m128i *v131; // rsi
  signed int v132; // eax
  unsigned int v133; // ebx
  __int64 v134; // rdx
  __int64 v135; // rax
  _BOOL4 v136; // ebx
  __int64 v137; // rdi
  unsigned __int8 v138; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v139; // [rsp+10h] [rbp-F0h]
  __int64 i; // [rsp+18h] [rbp-E8h]
  int v141; // [rsp+20h] [rbp-E0h]
  int v142; // [rsp+28h] [rbp-D8h]
  int v143; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v144; // [rsp+30h] [rbp-D0h]
  int v145; // [rsp+30h] [rbp-D0h]
  int v146; // [rsp+30h] [rbp-D0h]
  int v147; // [rsp+30h] [rbp-D0h]
  int v148; // [rsp+30h] [rbp-D0h]
  int v149; // [rsp+30h] [rbp-D0h]
  int v150; // [rsp+30h] [rbp-D0h]
  int v151; // [rsp+30h] [rbp-D0h]
  int v152; // [rsp+38h] [rbp-C8h]
  char v153; // [rsp+40h] [rbp-C0h]
  unsigned __int8 v154; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v155; // [rsp+40h] [rbp-C0h]
  int v156; // [rsp+40h] [rbp-C0h]
  char v157; // [rsp+40h] [rbp-C0h]
  unsigned __int8 v158; // [rsp+40h] [rbp-C0h]
  char v160; // [rsp+5Fh] [rbp-A1h] BYREF
  unsigned int v161; // [rsp+60h] [rbp-A0h] BYREF
  __int32 v162; // [rsp+64h] [rbp-9Ch] BYREF
  int v163; // [rsp+68h] [rbp-98h] BYREF
  unsigned int v164; // [rsp+6Ch] [rbp-94h] BYREF
  __m128i v165; // [rsp+70h] [rbp-90h] BYREF
  __m128i v166; // [rsp+80h] [rbp-80h] BYREF
  __m128i v167; // [rsp+90h] [rbp-70h] BYREF
  __m128i v168; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v169; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v170[4]; // [rsp+C0h] [rbp-40h] BYREF

  v11 = a1;
  v12 = a2;
  v14 = a3;
  v16 = a1;
  v162 = 0;
  *a8 = 0;
  *a9 = 0;
  if ( a10 )
    *a10 = 0;
  v17 = a2[10].m128i_i8[13];
  v160 = 5;
  v161 = 0;
  if ( !v17 )
    return sub_72C970(a5);
  v18 = a3[10].m128i_i8[13];
  if ( !v18 )
    return sub_72C970(a5);
  if ( dword_4F077C4 == 2 )
  {
    if ( v18 == 12 || v17 == 12 )
      goto LABEL_35;
    if ( dword_4D03F94 )
      goto LABEL_8;
    if ( dword_4F07588 )
    {
      if ( dword_4F04C44 == -1 )
      {
        v89 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v89 + 6) & 6) == 0 && *(_BYTE *)(v89 + 4) != 12 )
          goto LABEL_8;
      }
    }
    if ( v17 == 10 )
    {
      if ( (unsigned int)sub_8DBE70(v12[8].m128i_i64[0]) )
        goto LABEL_35;
      v18 = v14[10].m128i_i8[13];
      v16 = a1;
    }
    if ( v18 == 10 )
    {
      v157 = v16;
      if ( (unsigned int)sub_8DBE70(v14[8].m128i_i64[0]) )
        goto LABEL_35;
      v16 = v157;
    }
    v153 = v16;
    if ( !(unsigned int)sub_8DBE70(a4) )
    {
      v16 = v153;
      v17 = v12[10].m128i_i8[13];
      goto LABEL_8;
    }
LABEL_35:
    *a8 = 1;
    *a9 = 1;
    return (__int64)a9;
  }
LABEL_8:
  if ( v17 == 8 || (v20 = v14[10].m128i_i8[13], v20 == 8) )
  {
LABEL_25:
    *a8 = 1;
    return (__int64)a8;
  }
  if ( *(_BYTE *)(a4 + 140) == 15 )
  {
    v70 = *(_QWORD *)(a4 + 160);
    v166.m128i_i32[0] = 0;
    v71 = *(_QWORD *)(a4 + 128);
    for ( i = v70; *(_BYTE *)(v70 + 140) == 12; v70 = *(_QWORD *)(v70 + 160) )
      ;
    v72 = *(_QWORD *)(v70 + 128);
    v144 = v71 / v72;
    switch ( (char)a1 )
    {
      case 'A':
        v16 = 58;
        break;
      case 'B':
        v16 = 59;
        break;
      case 'C':
        v16 = 60;
        break;
      case 'D':
        v16 = 61;
        break;
      case 'E':
        v16 = 62;
        break;
      case 'F':
        v16 = 63;
        break;
      case 'Y':
        v16 = 87;
        break;
      case 'Z':
        v16 = 88;
        break;
      default:
        break;
    }
    v152 = 0;
    if ( v17 == 10 )
    {
      v152 = 1;
      v12 = (const __m128i *)v12[11].m128i_i64[0];
    }
    v167.m128i_i64[0] = (__int64)v12;
    v73 = 0;
    if ( v20 == 10 )
    {
      v14 = (const __m128i *)v14[11].m128i_i64[0];
      v73 = 1;
    }
    v169.m128i_i64[0] = (__int64)v14;
    v74 = 0;
    v139 = v72;
    v155 = v71;
    v138 = v16;
    sub_724C70(a5, 10);
    *(_QWORD *)(a5 + 128) = a4;
    v142 = 0;
    v141 = 0;
    if ( v155 < v139 )
    {
LABEL_164:
      result = v166.m128i_u32[0];
      *a8 = v166.m128i_i32[0];
      return result;
    }
    v156 = v73;
    while ( 1 )
    {
      if ( v166.m128i_i32[0] )
      {
LABEL_160:
        if ( v141 )
          sub_724E30(&v167);
        if ( v142 )
          sub_724E30(&v169);
        goto LABEL_164;
      }
      v76 = 0;
      v77 = sub_724D50(0);
      v82 = v167.m128i_i64[0];
      v83 = v77;
      if ( v167.m128i_i64[0] )
        break;
      v76 = i;
      v82 = sub_724DC0(0, 0, v78, v79, v80, v81);
      v167.m128i_i64[0] = v82;
      sub_72BB40(i, v82);
      v75 = v169.m128i_i64[0];
      if ( !v169.m128i_i64[0] )
      {
        v152 = 0;
        v141 = 1;
LABEL_159:
        v169.m128i_i64[0] = sub_724DC0(v76, v82, v75, v79, v80, v81);
        sub_72BB40(i, v169.m128i_i64[0]);
        LODWORD(v75) = v169.m128i_i32[0];
        LODWORD(v82) = v167.m128i_i32[0];
        v156 = 0;
        v142 = 1;
LABEL_142:
        sub_713ED0(v138, v82, v75, i, v83, a6, a7, (__int64)&v166, (__int64)a9, (__int64)a10, (__int64)a11);
        sub_72A690(v83, a5, 0, 0);
        if ( v152 )
          v167.m128i_i64[0] = *(_QWORD *)(v167.m128i_i64[0] + 120);
        goto LABEL_144;
      }
      sub_713ED0(
        v138,
        v167.m128i_i32[0],
        v169.m128i_i32[0],
        i,
        v83,
        a6,
        a7,
        (__int64)&v166,
        (__int64)a9,
        (__int64)a10,
        (__int64)a11);
      sub_72A690(v83, a5, 0, 0);
      v152 = 0;
      v141 = 1;
LABEL_144:
      if ( v156 )
        v169.m128i_i64[0] = *(_QWORD *)(v169.m128i_i64[0] + 120);
      if ( v144 <= ++v74 )
        goto LABEL_160;
    }
    v75 = v169.m128i_i64[0];
    if ( !v169.m128i_i64[0] )
      goto LABEL_159;
    goto LABEL_142;
  }
  sub_724C70(a5, 0);
  *(_QWORD *)(a5 + 128) = a4;
  if ( (v12[10].m128i_i64[1] & 0xFF0000000008LL) != 0x60000000008LL || !(unsigned int)sub_8D2930(v12[8].m128i_i64[0]) )
  {
    if ( (v14[10].m128i_i64[1] & 0xFF0000000008LL) == 0x60000000008LL && (unsigned int)sub_8D2930(v14[8].m128i_i64[0]) )
    {
      if ( (_BYTE)a1 == 39 )
      {
        if ( v12[10].m128i_i8[13] == 1 )
        {
          sub_70F370(v14, 39, v12, (__m128i *)a5, a8, &v161, &v160);
          v29 = v161;
          goto LABEL_19;
        }
      }
      else if ( HIDWORD(qword_4F077B4) && (_BYTE)a1 == 55 && (unsigned int)sub_72A2A0(v12, 0, v26, v27, v28) )
      {
        v106 = v12[11].m128i_i64[1];
        v169.m128i_i64[0] = v12[11].m128i_i64[0];
        v169.m128i_i64[1] = v106;
        sub_6213D0((__int64)&v169, (__int64)v12[11].m128i_i64);
        sub_724A80(a5, 1);
        v107 = _mm_loadu_si128(&v169);
        *(_BYTE *)(a5 + 169) |= 1u;
        v29 = v161;
        *(__m128i *)(a5 + 176) = v107;
        goto LABEL_19;
      }
LABEL_18:
      v29 = v161;
      *a8 = 1;
      goto LABEL_19;
    }
    v21 = v12[8].m128i_i64[0];
    a1 = (unsigned __int8)a1;
    v22 = (unsigned int)sub_730040((unsigned __int8)a1, v21, v14[8].m128i_i64[0]);
    switch ( (char)a1 )
    {
      case '\'':
        if ( (unsigned __int8)v22 <= 4u )
        {
          if ( (unsigned __int8)v22 <= 2u )
          {
            if ( (_BYTE)v22 == 2 )
            {
              v161 = 0;
              v160 = 5;
              v169 = _mm_loadu_si128(v12 + 11);
              v49 = sub_620E90((__int64)v12);
              sub_621270((unsigned __int16 *)&v169, v14[11].m128i_i16, v49, (_BOOL4 *)v167.m128i_i32);
              if ( !v167.m128i_i32[0] )
                goto LABEL_78;
LABEL_91:
              if ( v49 )
              {
                v53 = 5;
                v161 = 61;
                if ( dword_4D04964 )
                  v53 = byte_4F07472[0];
                v160 = v53;
              }
              goto LABEL_78;
            }
LABEL_36:
            sub_721090(a1);
          }
          for ( j = v12[8].m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v111 = *(_BYTE *)(j + 160);
          v112 = v12[10].m128i_u8[13];
          v161 = 0;
          v160 = 5;
          sub_724A80(a5, v112);
          sub_70B8D0(v111, v12 + 11, v14 + 11, (_OWORD *)(a5 + 176), &v169, &v162);
          if ( !v169.m128i_i32[0] )
            goto LABEL_61;
LABEL_239:
          v161 = 222;
          v160 = 8;
          goto LABEL_61;
        }
        if ( (_BYTE)v22 != 5 )
          goto LABEL_36;
        for ( k = v12[8].m128i_i64[0]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        v104 = *(_BYTE *)(k + 160);
        sub_70DCF0((__int64)v12, &v167);
        sub_70DCF0((__int64)v14, &v169);
        v161 = 0;
        v160 = 5;
        sub_724A80(a5, 4);
        sub_70BF90(v104, &v167, &v169, *(_OWORD **)(a5 + 176), &v166, &v162);
        if ( !v166.m128i_i32[0] )
          goto LABEL_61;
        goto LABEL_209;
      case '(':
        if ( (unsigned __int8)v22 <= 4u )
        {
          if ( (unsigned __int8)v22 > 2u )
          {
            for ( m = v12[8].m128i_i64[0]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
              ;
            v114 = *(_BYTE *)(m + 160);
            v115 = v12[10].m128i_u8[13];
            v161 = 0;
            v160 = 5;
            sub_724A80(a5, v115);
            sub_70B9E0(v114, v12 + 11, v14 + 11, (_OWORD *)(a5 + 176), &v169, &v162);
            if ( !v169.m128i_i32[0] )
              goto LABEL_61;
            goto LABEL_239;
          }
          if ( (_BYTE)v22 != 2 )
            goto LABEL_36;
          v161 = 0;
          v160 = 5;
          v169 = _mm_loadu_si128(v12 + 11);
          v49 = sub_620E90((__int64)v12);
          sub_6215F0((unsigned __int16 *)&v169, v14[11].m128i_i16, v49, (_BOOL4 *)v167.m128i_i32);
          if ( v167.m128i_i32[0] )
            goto LABEL_91;
          goto LABEL_78;
        }
        if ( (_BYTE)v22 != 5 )
          goto LABEL_36;
        for ( n = v12[8].m128i_i64[0]; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
          ;
        v100 = *(_BYTE *)(n + 160);
        sub_70DCF0((__int64)v12, &v167);
        sub_70DCF0((__int64)v14, &v169);
        v161 = 0;
        v160 = 5;
        sub_724A80(a5, 4);
        sub_70C020(v100, &v167, &v169, *(_OWORD **)(a5 + 176), &v166, &v162);
        if ( !v166.m128i_i32[0] )
          goto LABEL_61;
        goto LABEL_209;
      case ')':
        if ( (unsigned __int8)v22 <= 4u )
        {
          if ( (unsigned __int8)v22 > 2u )
          {
            for ( ii = v12[8].m128i_i64[0]; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
              ;
            v120 = v12[10].m128i_i8[13] == 5;
            v121 = *(_BYTE *)(ii + 160);
            v161 = 0;
            v160 = 5;
            sub_724A80(a5, (unsigned __int8)(2 * (v120 != (v14[10].m128i_i8[13] == 5)) + 3));
            sub_70BBE0(v121, v12 + 11, v14 + 11, (_OWORD *)(a5 + 176), &v169, &v162);
            if ( !v169.m128i_i32[0] )
              goto LABEL_61;
            goto LABEL_239;
          }
          if ( (_BYTE)v22 != 2 )
            goto LABEL_36;
          v161 = 0;
          v160 = 5;
          v169 = _mm_loadu_si128(v12 + 11);
          v49 = sub_620E90((__int64)v12);
          sub_621F20(&v169, v14 + 11, v49, (_BOOL4 *)v167.m128i_i32);
          if ( v167.m128i_i32[0] )
            goto LABEL_91;
LABEL_78:
          sub_70FF50(&v169, a5, v49, 0, &v161, (unsigned __int8 *)&v160);
          v29 = v161;
          goto LABEL_19;
        }
        if ( (_BYTE)v22 != 5 )
          goto LABEL_36;
        for ( jj = v12[8].m128i_i64[0]; *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
          ;
        v102 = *(_BYTE *)(jj + 160);
        sub_70DCF0((__int64)v12, &v167);
        sub_70DCF0((__int64)v14, &v169);
        v161 = 0;
        v160 = 5;
        sub_724A80(a5, 4);
        sub_70C130(v102, &v167, &v169, *(_OWORD **)(a5 + 176), v166.m128i_i32, (unsigned int *)&v162);
        if ( !v166.m128i_i32[0] )
          goto LABEL_61;
        goto LABEL_209;
      case '*':
        if ( (unsigned __int8)v22 > 4u )
        {
          if ( (_BYTE)v22 != 5 )
            goto LABEL_36;
          for ( kk = v12[8].m128i_i64[0]; *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
            ;
          v91 = *(_BYTE *)(kk + 160);
          v92 = (const __m128i *)v12[11].m128i_i64[0];
          if ( v12[10].m128i_i8[13] == 4 )
          {
            v167 = _mm_loadu_si128(v92);
            v168 = _mm_loadu_si128(v92 + 1);
          }
          else
          {
            v167 = _mm_loadu_si128(v92 + 11);
            v168 = _mm_loadu_si128((const __m128i *)(v92[7].m128i_i64[1] + 176));
          }
          v93 = (const __m128i *)v14[11].m128i_i64[0];
          if ( v14[10].m128i_i8[13] == 4 )
          {
            v169 = _mm_loadu_si128(v93);
            v170[0] = _mm_loadu_si128(v93 + 1);
          }
          else
          {
            v169 = _mm_loadu_si128(v93 + 11);
            v170[0] = _mm_loadu_si128((const __m128i *)(v93[7].m128i_i64[1] + 176));
          }
          v161 = 0;
          v160 = 5;
          sub_724A80(a5, 4);
          sub_70BBE0(v91, &v169, &v169, &v165, &v163, &v164);
          v145 = v163;
          v162 = v164;
          sub_70BBE0(v91, v170, v170, &v166, &v163, &v164);
          v162 |= v164;
          v146 = v163 | v145;
          sub_70B8D0(v91, &v165, &v166, &v165, &v163, &v164);
          v147 = v163 | v146;
          v162 |= v164;
          if ( unk_4D04248 || !(unsigned int)sub_70B8A0(v91, &v165) )
          {
            sub_70BBE0(v91, &v167, &v169, *(_OWORD **)(a5 + 176), &v163, &v164);
            v162 |= v164;
            v148 = v163 | v147;
            sub_70BBE0(v91, &v168, v170, &v166, &v163, &v164);
            v94 = *(__m128i **)(a5 + 176);
            v162 |= v164;
            v149 = v163 | v148;
            sub_70B8D0(v91, v94, &v166, v94, &v163, &v164);
            v95 = *(__m128i **)(a5 + 176);
            v162 |= v164;
            v150 = v163 | v149;
            sub_70BCF0(v91, v95, &v165, v95, &v163, &v164);
            v151 = v163 | v150;
            v96 = (_OWORD *)(*(_QWORD *)(a5 + 176) + 16LL);
            v162 |= v164;
            sub_70BBE0(v91, &v167, v170, v96, &v163, &v164);
            v162 |= v164;
            v143 = v163 | v151;
            sub_70BBE0(v91, &v168, &v169, &v166, &v163, &v164);
            v97 = v163 | v143;
            v162 |= v164;
            sub_70B9E0(
              v91,
              &v166,
              (const __m128i *)(*(_QWORD *)(a5 + 176) + 16LL),
              (_OWORD *)(*(_QWORD *)(a5 + 176) + 16LL),
              &v163,
              &v164);
            v162 |= v164;
            v98 = v163 | v97;
            sub_70BCF0(
              v91,
              (const __m128i *)(*(_QWORD *)(a5 + 176) + 16LL),
              &v165,
              (_OWORD *)(*(_QWORD *)(a5 + 176) + 16LL),
              &v163,
              &v164);
            v162 |= v164;
            if ( !(v163 | v98) )
              goto LABEL_61;
LABEL_209:
            v161 = 1047;
            v160 = 8;
            goto LABEL_61;
          }
        }
        else
        {
          if ( (unsigned __int8)v22 <= 2u )
          {
            if ( (_BYTE)v22 != 2 )
              goto LABEL_36;
            v161 = 0;
            v160 = 5;
            v169 = _mm_loadu_si128(v12 + 11);
            v49 = sub_620E90((__int64)v12);
            sub_6220A0(&v169, v14 + 11, v49, (_BOOL4 *)v167.m128i_i32);
            if ( v167.m128i_i32[0] )
            {
              if ( (unsigned int)sub_6210B0((__int64)v14, 0) )
                goto LABEL_91;
              v161 = 39;
              v160 = 8;
            }
            goto LABEL_78;
          }
          for ( mm = v12[8].m128i_i64[0]; *(_BYTE *)(mm + 140) == 12; mm = *(_QWORD *)(mm + 160) )
            ;
          v109 = *(_BYTE *)(mm + 160);
          v161 = 0;
          v160 = 5;
          if ( unk_4D04248 || !(unsigned int)sub_70B8A0(v109, v14 + 11) )
          {
            sub_724A80(a5, (unsigned __int8)(2 * ((v12[10].m128i_i8[13] == 5) != (v14[10].m128i_i8[13] == 5)) + 3));
            sub_70BCF0(v109, v12 + 11, v14 + 11, (_OWORD *)(a5 + 176), &v169, &v162);
            if ( !v169.m128i_i32[0] )
              goto LABEL_61;
            goto LABEL_239;
          }
        }
LABEL_76:
        v161 = 39;
        v160 = 8;
LABEL_61:
        v29 = v161;
        goto LABEL_19;
      case '+':
        v161 = 0;
        v160 = 5;
        v169 = _mm_loadu_si128(v12 + 11);
        v49 = sub_620E90((__int64)v12);
        sub_6220C0(&v169, (__m128i *)&v14[11], v49, (_BOOL4 *)v167.m128i_i32);
        if ( !v167.m128i_i32[0] )
          goto LABEL_78;
        if ( (unsigned int)sub_6210B0((__int64)v14, 0) )
          goto LABEL_91;
        v161 = 179;
        v160 = 8;
        goto LABEL_78;
      case ',':
        for ( nn = v12[8].m128i_i64[0]; *(_BYTE *)(nn + 140) == 12; nn = *(_QWORD *)(nn + 160) )
          ;
        v44 = *(_BYTE *)(nn + 160);
        v161 = 0;
        v160 = 5;
        sub_724A80(a5, 3);
        sub_70BBE0(v44, v12 + 11, v14 + 11, (_OWORD *)(a5 + 176), &v167, &v169);
        v45 = v167.m128i_i32[0];
        v162 = v169.m128i_i32[0];
        sub_70BAF0(v44, (const __m128i *)(a5 + 176), (_OWORD *)(a5 + 176), &v167, &v169);
        v162 |= v169.m128i_i32[0];
        if ( !(v167.m128i_i32[0] | v45) )
          goto LABEL_61;
        goto LABEL_45;
      case '-':
        for ( i1 = v12[8].m128i_i64[0]; *(_BYTE *)(i1 + 140) == 12; i1 = *(_QWORD *)(i1 + 160) )
          ;
        v47 = *(_BYTE *)(i1 + 160);
        v161 = 0;
        v160 = 5;
        v162 = 0;
        if ( !unk_4D04248 )
        {
          v154 = v47;
          v48 = sub_70B8A0(v47, v14 + 11);
          v47 = v154;
          if ( v48 )
            goto LABEL_76;
        }
        v158 = v47;
        sub_724A80(a5, 5);
        sub_70BCF0(v158, v12 + 11, v14 + 11, (_OWORD *)(a5 + 176), &v167, &v169);
        v105 = v167.m128i_i32[0];
        v162 = v169.m128i_i32[0];
        sub_70BAF0(v158, (const __m128i *)(a5 + 176), (_OWORD *)(a5 + 176), &v167, &v169);
        v162 |= v169.m128i_i32[0];
        if ( v167.m128i_i32[0] | v105 )
          goto LABEL_209;
        goto LABEL_61;
      case '.':
      case '/':
      case '0':
      case '1':
        v31 = v12[8].m128i_i64[0];
        for ( v169.m128i_i32[0] = 0; *(_BYTE *)(v31 + 140) == 12; v31 = *(_QWORD *)(v31 + 160) )
          ;
        v32 = *(_BYTE *)(v31 + 160);
        v161 = 0;
        v160 = 5;
        v162 = 0;
        sub_724A80(a5, 4);
        v33 = *(__m128i **)(a5 + 176);
        switch ( (_BYTE)a1 )
        {
          case '0':
            *v33 = _mm_loadu_si128(v12 + 11);
            sub_70BAF0(v32, v14 + 11, (_OWORD *)(*(_QWORD *)(a5 + 176) + 16LL), &v169, &v162);
            break;
          case '1':
            v33[1] = _mm_loadu_si128(v12 + 11);
            sub_70BAF0(v32, v14 + 11, *(_OWORD **)(a5 + 176), &v169, &v162);
            break;
          case '/':
            v33[1] = _mm_loadu_si128(v12 + 11);
            *(__m128i *)*(_QWORD *)(a5 + 176) = _mm_loadu_si128(v14 + 11);
            break;
          default:
            *v33 = _mm_loadu_si128(v12 + 11);
            *(__m128i *)(*(_QWORD *)(a5 + 176) + 16LL) = _mm_loadu_si128(v14 + 11);
            break;
        }
        if ( !v169.m128i_i32[0] )
          goto LABEL_61;
LABEL_45:
        v161 = 1047;
        v29 = 1047;
        v160 = 8;
        goto LABEL_46;
      case '2':
        if ( (unsigned int)sub_8D2E30(v14[8].m128i_i64[0]) )
        {
          v41 = v12;
          v42 = v14;
        }
        else
        {
          v41 = v14;
          v42 = v12;
        }
        sub_70F370(v42, 50, v41, (__m128i *)a5, a8, &v161, &v160);
        v29 = v161;
        goto LABEL_19;
      case '3':
        sub_70F370(v12, 51, v14, (__m128i *)a5, a8, &v161, &v160);
        v29 = v161;
        goto LABEL_19;
      case '4':
        sub_713640(v12, v14, (_QWORD *)a5, (unsigned int *)a8, &v161, (unsigned __int8 *)&v160);
        v29 = v161;
        goto LABEL_19;
      case '5':
        v40 = 0;
        goto LABEL_60;
      case '6':
        v40 = 1;
LABEL_60:
        sub_7132C0(v12, (__int64)v14, a5, v40, &v161, (unsigned __int8 *)&v160);
        goto LABEL_61;
      case '7':
        v38 = v12[11].m128i_i64[1];
        v169.m128i_i64[0] = v12[11].m128i_i64[0];
        v169.m128i_i64[1] = v38;
        sub_6213D0((__int64)&v169, (__int64)v14[11].m128i_i64);
        sub_724A80(a5, 1);
        v39 = _mm_loadu_si128(&v169);
        *(_BYTE *)(a5 + 169) |= 1u;
        v29 = v161;
        *(__m128i *)(a5 + 176) = v39;
        goto LABEL_19;
      case '8':
        v36 = v12[11].m128i_i64[1];
        v169.m128i_i64[0] = v12[11].m128i_i64[0];
        v169.m128i_i64[1] = v36;
        sub_6213B0((__int64)&v169, (__int64)v14[11].m128i_i64);
        sub_724A80(a5, 1);
        v37 = _mm_loadu_si128(&v169);
        *(_BYTE *)(a5 + 169) |= 1u;
        v29 = v161;
        *(__m128i *)(a5 + 176) = v37;
        goto LABEL_19;
      case '9':
        v34 = v12[11].m128i_i64[1];
        v169.m128i_i64[0] = v12[11].m128i_i64[0];
        v169.m128i_i64[1] = v34;
        sub_6213F0((__int64)&v169, (__int64)v14[11].m128i_i64);
        sub_724A80(a5, 1);
        v35 = _mm_loadu_si128(&v169);
        *(_BYTE *)(a5 + 169) |= 1u;
        v29 = v161;
        *(__m128i *)(a5 + 176) = v35;
        goto LABEL_19;
      case ':':
      case ';':
      case '<':
      case '=':
      case '>':
      case '?':
        switch ( (char)v22 )
        {
          case 2:
          case 19:
            sub_70CCB0((__int64)v12, a1, (__int64)v14, a5);
            v29 = v161;
            goto LABEL_19;
          case 3:
            for ( i2 = v12[8].m128i_i64[0]; *(_BYTE *)(i2 + 140) == 12; i2 = *(_QWORD *)(i2 + 160) )
              ;
            v59 = *(_BYTE *)(i2 + 160);
            v162 = 0;
            v57 = sub_70BE30(v59, v12 + 11, v14 + 11, &v169);
            if ( v169.m128i_i32[0] )
            {
              v162 = 1;
              v57 = v11 == 59;
            }
            else
            {
              switch ( v11 )
              {
                case ';':
                  v57 = v57 != 0;
                  goto LABEL_98;
                case '<':
                  v57 = v57 > 0;
                  goto LABEL_98;
                case '=':
                  goto LABEL_291;
                case '>':
                  v57 = ~v57;
LABEL_291:
                  v57 = (unsigned int)v57 >> 31;
                  break;
                case '?':
                  v57 = v57 <= 0;
                  break;
                default:
                  goto LABEL_130;
              }
            }
            goto LABEL_98;
          case 5:
            for ( i3 = v12[8].m128i_i64[0]; *(_BYTE *)(i3 + 140) == 12; i3 = *(_QWORD *)(i3 + 160) )
              ;
            v67 = *(_BYTE *)(i3 + 160);
            v68 = (const __m128i *)v12[11].m128i_i64[0];
            if ( v12[10].m128i_i8[13] == 4 )
            {
              v167 = _mm_loadu_si128(v68);
              v168 = _mm_loadu_si128(v68 + 1);
            }
            else
            {
              v167 = _mm_loadu_si128(v68 + 11);
              v168 = _mm_loadu_si128((const __m128i *)(v68[7].m128i_i64[1] + 176));
            }
            v69 = (const __m128i *)v14[11].m128i_i64[0];
            if ( v14[10].m128i_i8[13] == 4 )
            {
              v169 = _mm_loadu_si128(v69);
              v170[0] = _mm_loadu_si128(v69 + 1);
            }
            else
            {
              v169 = _mm_loadu_si128(v69 + 11);
              v170[0] = _mm_loadu_si128((const __m128i *)(v69[7].m128i_i64[1] + 176));
            }
            v57 = sub_70C550(v67, &v167, &v169);
            if ( v11 == 59 )
LABEL_130:
              v57 = v57 == 0;
            goto LABEL_98;
          case 6:
            v167.m128i_i64[0] = sub_724DC0((unsigned __int8)a1, v21, v22, v23, v24, v25);
            a1 = (__int64)v12;
            v169.m128i_i64[0] = sub_724DC0(v11, v21, v60, v61, v62, v63);
            v161 = 0;
            v160 = 5;
            if ( sub_70E2C0((__int64)v12, (__int64)v14, &v166, v64) )
            {
              v125 = v12[10].m128i_i8[13];
              if ( v125 == 1 )
              {
                v126 = 52;
                a1 = v167.m128i_i64[0];
                v127 = v12;
                while ( v126 )
                {
                  *(_DWORD *)a1 = v127->m128i_i32[0];
                  v127 = (const __m128i *)((char *)v127 + 4);
                  a1 += 4;
                  --v126;
                }
              }
              else
              {
                if ( v125 != 6 )
                  goto LABEL_36;
                a1 = v167.m128i_i64[0];
                sub_72BAF0(v167.m128i_i64[0], v12[12].m128i_i64[0], unk_4F06A60);
              }
              v128 = v14[10].m128i_i8[13];
              if ( v128 == 1 )
              {
                v129 = 52;
                v130 = (_DWORD *)v169.m128i_i64[0];
                v131 = v14;
                while ( v129 )
                {
                  *v130 = v131->m128i_i32[0];
                  v131 = (const __m128i *)((char *)v131 + 4);
                  ++v130;
                  --v129;
                }
              }
              else
              {
                if ( v128 != 6 )
                  goto LABEL_36;
                sub_72BAF0(v169.m128i_i64[0], v14[12].m128i_i64[0], unk_4F06A60);
              }
              v132 = sub_621060(v167.m128i_i64[0], v169.m128i_i64[0]);
              v133 = v132;
              switch ( v11 )
              {
                case ';':
                  v136 = v132 != 0;
                  break;
                case '<':
                  v136 = v132 > 0;
                  break;
                case '=':
                  goto LABEL_314;
                case '>':
                  v133 = ~v132;
LABEL_314:
                  v136 = v133 >> 31;
                  break;
                case '?':
                  v136 = v132 <= 0;
                  break;
                default:
                  v136 = v132 == 0;
                  break;
              }
              sub_724A80(a5, 1);
              sub_620D80((_WORD *)(a5 + 176), v136);
              goto LABEL_122;
            }
            if ( v166.m128i_i32[0] )
              goto LABEL_122;
            v65 = v11 - 58;
            if ( word_4D04898 )
            {
              if ( v65 > 1u )
              {
                v166.m128i_i32[0] = 1;
                goto LABEL_122;
              }
LABEL_289:
              sub_724A80(a5, 1);
              sub_620D80((_WORD *)(a5 + 176), v11 == 59);
              goto LABEL_122;
            }
            v166.m128i_i32[0] = 1;
            if ( v65 > 1u || dword_4D04964 )
              goto LABEL_122;
            if ( (unsigned int)sub_710600((__int64)v14) && v12[10].m128i_i8[13] == 6 && v12[11].m128i_i8[0] == 1 )
            {
              v137 = v12[11].m128i_i64[1];
            }
            else
            {
              if ( !(unsigned int)sub_710600((__int64)v12) || v14[10].m128i_i8[13] != 6 || v14[11].m128i_i8[0] != 1 )
              {
LABEL_280:
                if ( (unsigned int)sub_710600((__int64)v14) && v12[10].m128i_i8[13] == 6 && !v12[11].m128i_i8[0] )
                {
                  v124 = v12[11].m128i_i64[1];
                }
                else
                {
                  if ( !(unsigned int)sub_710600((__int64)v12) || v14[10].m128i_i8[13] != 6 || v14[11].m128i_i8[0] )
                    goto LABEL_122;
                  v124 = v14[11].m128i_i64[1];
                }
                if ( v124 && sub_70FCD0(v124) )
                  goto LABEL_288;
LABEL_122:
                sub_724E30(&v167);
                sub_724E30(&v169);
                v29 = v161;
                *a8 = v166.m128i_i32[0];
                goto LABEL_19;
              }
              v137 = v14[11].m128i_i64[1];
            }
            if ( v137 && sub_70FCC0(v137) )
            {
LABEL_288:
              v166.m128i_i32[0] = 0;
              goto LABEL_289;
            }
            goto LABEL_280;
          case 13:
            v57 = 0;
            if ( v12[11].m128i_i64[0] == v14[11].m128i_i64[0] && ((v14[12].m128i_i8[0] ^ v12[12].m128i_i8[0]) & 2) == 0 )
            {
              v134 = v12[12].m128i_i64[1];
              v135 = v14[12].m128i_i64[1];
              if ( (v12[12].m128i_i8[0] & 2) != 0 )
              {
                v57 = v134 == v135;
              }
              else
              {
                if ( v134 == v135 )
                {
                  LOBYTE(v57) = 1;
                }
                else
                {
                  LOBYTE(v57) = v135 != 0 && v134 != 0;
                  if ( (_BYTE)v57 )
                  {
                    LOBYTE(v57) = 0;
                    if ( *(_QWORD *)(v134 + 128) == *(_QWORD *)(v135 + 128) )
                      LOBYTE(v57) = *(_BYTE *)(v134 + 136) == *(_BYTE *)(v135 + 136);
                  }
                }
                v57 = (unsigned __int8)v57;
              }
            }
            if ( (_BYTE)a1 == 59 )
              v57 ^= 1u;
            goto LABEL_98;
          default:
            goto LABEL_36;
        }
      case 'G':
      case 'H':
        if ( (_BYTE)v22 != 3 )
        {
          if ( (_BYTE)v22 == 6 )
            goto LABEL_18;
          if ( (_BYTE)v22 != 2 )
            goto LABEL_36;
          if ( (int)sub_621060((__int64)v12, (__int64)v14) <= 0 )
          {
            if ( (_BYTE)a1 == 71 )
              goto LABEL_53;
          }
          else if ( (_BYTE)a1 != 71 )
          {
LABEL_53:
            sub_72A510(v12, a5);
            v29 = v161;
            goto LABEL_19;
          }
LABEL_55:
          sub_72A510(v14, a5);
          v29 = v161;
          goto LABEL_19;
        }
        for ( i4 = v12[8].m128i_i64[0]; *(_BYTE *)(i4 + 140) == 12; i4 = *(_QWORD *)(i4 + 160) )
          ;
        v123 = sub_70BE30(*(_BYTE *)(i4 + 160), v12 + 11, v14 + 11, &v169);
        if ( (_BYTE)a1 == 71 )
        {
          if ( v169.m128i_i32[0] || v123 >= 0 )
            goto LABEL_262;
        }
        else if ( v169.m128i_i32[0] || v123 <= 0 )
        {
LABEL_262:
          sub_72A510(v14, a5);
          goto LABEL_61;
        }
        sub_72A510(v12, a5);
        goto LABEL_61;
      case 'W':
        *a8 = 0;
        if ( !sub_70FCE0((__int64)v12) )
          goto LABEL_86;
        v57 = 0;
        if ( (unsigned int)sub_711520((__int64)v12, v21, v54, v55, v56) )
          goto LABEL_97;
        goto LABEL_249;
      case 'X':
        *a8 = 0;
        if ( !sub_70FCE0((__int64)v12) )
          goto LABEL_86;
        v57 = 1;
        if ( !(unsigned int)sub_711520((__int64)v12, v21, v50, v51, v52) )
          goto LABEL_97;
LABEL_249:
        if ( !sub_70FCE0((__int64)v14) )
          goto LABEL_86;
        v57 = sub_711520((__int64)v14, v21, v116, v117, v118) == 0;
LABEL_97:
        if ( *a8 )
          goto LABEL_61;
LABEL_98:
        sub_724A80(a5, 1);
        sub_620D80((_WORD *)(a5 + 176), v57);
        v29 = v161;
        goto LABEL_19;
      case '[':
        goto LABEL_55;
      default:
        goto LABEL_36;
    }
  }
  if ( (unsigned __int8)(a1 - 39) <= 1u && v14[10].m128i_i8[13] == 1 )
  {
    sub_70F370(v12, (unsigned __int8)a1, v14, (__m128i *)a5, a8, &v161, &v160);
    v29 = v161;
    goto LABEL_19;
  }
  if ( !dword_4F077C0 && ((v86 = dword_4F077BC) == 0 || qword_4F077A8 > 0x9C3Fu) || (_BYTE)a1 != 40 )
  {
    if ( HIDWORD(qword_4F077B4) && (_BYTE)a1 == 55 && (unsigned int)sub_72A2A0(v14, 0, v84, v85, v86) )
    {
      v87 = v14[11].m128i_i64[1];
      v169.m128i_i64[0] = v14[11].m128i_i64[0];
      v169.m128i_i64[1] = v87;
      sub_6213D0((__int64)&v169, (__int64)v14[11].m128i_i64);
      sub_724A80(a5, 1);
      v88 = _mm_loadu_si128(&v169);
      *(_BYTE *)(a5 + 169) |= 1u;
      v29 = v161;
      *(__m128i *)(a5 + 176) = v88;
      goto LABEL_19;
    }
    goto LABEL_86;
  }
  if ( (v14[10].m128i_i64[1] & 0xFF0000000008LL) != 0x60000000008LL || !(unsigned int)sub_8D2930(v14[8].m128i_i64[0]) )
  {
LABEL_86:
    *a8 = 1;
    goto LABEL_61;
  }
  sub_713640(v12, v14, (_QWORD *)a5, (unsigned int *)a8, &v161, (unsigned __int8 *)&v160);
  v29 = v161;
  if ( !*a8 && *(_BYTE *)(a5 + 173) == 8 )
    *(_QWORD *)(a5 + 128) = a4;
LABEL_19:
  if ( (_DWORD)v29 )
  {
LABEL_46:
    sub_70CE90((int *)v29, v160, a6, a7, a8, a10, a11, a5);
    if ( v160 == 8 )
      v162 = 0;
  }
  if ( dword_4F077C4 == 2
    && (unk_4F07778 > 201102 || dword_4F07774)
    && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0xEA5Fu)
    || (v12[10].m128i_i64[1] & 0xFF0000000400LL) != 0x10000000000LL
    || (v12[10].m128i_i8[8] & 8) != 0 && !(unsigned int)sub_8D2660(v12[8].m128i_i64[0])
    || (v14[10].m128i_i64[1] & 0xFF0000000400LL) != 0x10000000000LL )
  {
    v30 = 1;
  }
  else
  {
    v30 = 0;
    if ( (v14[10].m128i_i8[8] & 8) != 0 )
    {
      v30 = sub_8D2660(v14[8].m128i_i64[0]);
      LOBYTE(v30) = v30 == 0;
    }
  }
  result = (4 * v30) | *(_BYTE *)(a5 + 169) & 0xFBu;
  *(_BYTE *)(a5 + 169) = result;
  if ( v162 )
  {
    result = a6;
    if ( !a6 )
      goto LABEL_25;
  }
  return result;
}
