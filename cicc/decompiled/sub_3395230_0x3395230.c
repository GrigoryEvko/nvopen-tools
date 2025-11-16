// Function: sub_3395230
// Address: 0x3395230
//
void __fastcall sub_3395230(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 *v4; // rdi
  int v5; // eax
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // r15d
  _BYTE *v10; // rax
  _BYTE *v11; // rcx
  _BYTE *i; // rdx
  __int64 *v13; // rdx
  __int64 v14; // r14
  unsigned int v15; // edx
  unsigned int v16; // r13d
  __int64 v17; // rdx
  unsigned int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r13
  __int16 v22; // ax
  int v23; // edx
  __int64 v24; // rdx
  char v25; // al
  int v26; // edx
  int v27; // ecx
  int v28; // ecx
  int v29; // edi
  int v30; // ecx
  int v31; // esi
  __int64 v32; // r8
  int v33; // eax
  const __m128i *v34; // rax
  const __m128i *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r12
  _BYTE *v38; // r13
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v40; // rdx
  bool v41; // r12
  __int64 v42; // rsi
  unsigned __int64 v43; // rax
  __int64 v44; // r9
  int v45; // edx
  __int64 v46; // r8
  unsigned int v47; // eax
  unsigned int v48; // r12d
  __int64 v49; // r13
  unsigned __int64 v50; // r8
  int v51; // edi
  unsigned __int64 v52; // rax
  __int64 v53; // r8
  unsigned __int64 v54; // r9
  unsigned __int64 v55; // rdx
  __int64 *v56; // r8
  __int64 v57; // rdx
  unsigned __int64 v58; // r8
  __int64 *v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r10
  __int64 v62; // rax
  __int64 v63; // r11
  __int64 v64; // r8
  __m128i *v65; // rax
  __int64 v66; // rsi
  __int64 v67; // rax
  int v68; // edx
  int v69; // edi
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 *v72; // r10
  __int64 v73; // r9
  _QWORD *v74; // rdi
  __int64 v75; // r15
  unsigned __int64 v76; // r12
  __int64 v77; // r13
  int v78; // eax
  int v79; // edx
  int v80; // r9d
  int v81; // ecx
  int v82; // r8d
  __m128i *v83; // rax
  __int64 v84; // rsi
  __int64 v85; // r12
  int v86; // edx
  int v87; // r13d
  _QWORD *v88; // rax
  __int64 v89; // rax
  unsigned int v90; // edx
  unsigned int v91; // r12d
  __int64 v92; // r15
  int v93; // edx
  __m128i *v94; // rax
  __int64 v95; // rsi
  unsigned __int16 *v96; // rax
  unsigned int v97; // r13d
  __int64 v98; // rax
  int v99; // edx
  int v100; // edi
  __int64 v101; // rdx
  unsigned __int64 v102; // rax
  __int64 v103; // rax
  int v104; // edx
  int v105; // edi
  __int64 v106; // rdx
  unsigned __int64 v107; // rax
  unsigned int v108; // ecx
  unsigned __int16 v109; // ax
  char v110; // dl
  __int64 v111; // rdx
  __int64 v112; // rdx
  unsigned __int16 v113; // ax
  __int64 v114; // rax
  unsigned int v115; // edx
  __int64 v116; // rax
  int v117; // edx
  unsigned __int16 v118; // ax
  unsigned int v119; // r8d
  unsigned __int16 v120; // ax
  unsigned int v121; // r8d
  char v122; // cl
  char v123; // cl
  __int128 v124; // [rsp-10h] [rbp-240h]
  size_t n; // [rsp+10h] [rbp-220h]
  size_t na; // [rsp+10h] [rbp-220h]
  __int64 v127; // [rsp+18h] [rbp-218h]
  int srcb; // [rsp+28h] [rbp-208h]
  __int64 *srcc; // [rsp+28h] [rbp-208h]
  void *srca; // [rsp+28h] [rbp-208h]
  void *src; // [rsp+28h] [rbp-208h]
  __int16 v132; // [rsp+3Ah] [rbp-1F6h]
  int v133; // [rsp+48h] [rbp-1E8h]
  __int64 v134; // [rsp+48h] [rbp-1E8h]
  __int64 v135; // [rsp+48h] [rbp-1E8h]
  __int64 v136; // [rsp+48h] [rbp-1E8h]
  unsigned int v137; // [rsp+54h] [rbp-1DCh]
  int v138; // [rsp+54h] [rbp-1DCh]
  __int64 v139; // [rsp+58h] [rbp-1D8h]
  int v140; // [rsp+60h] [rbp-1D0h]
  int v141; // [rsp+64h] [rbp-1CCh]
  unsigned __int64 v142; // [rsp+68h] [rbp-1C8h]
  char v143; // [rsp+68h] [rbp-1C8h]
  unsigned __int64 v144; // [rsp+70h] [rbp-1C0h]
  __int64 v145; // [rsp+70h] [rbp-1C0h]
  int v146; // [rsp+78h] [rbp-1B8h]
  int v147; // [rsp+78h] [rbp-1B8h]
  int v148; // [rsp+78h] [rbp-1B8h]
  int v149; // [rsp+78h] [rbp-1B8h]
  unsigned __int64 v150; // [rsp+78h] [rbp-1B8h]
  unsigned __int64 v151; // [rsp+78h] [rbp-1B8h]
  int v152; // [rsp+78h] [rbp-1B8h]
  __int64 v153; // [rsp+78h] [rbp-1B8h]
  __int64 v154; // [rsp+80h] [rbp-1B0h]
  int v155; // [rsp+80h] [rbp-1B0h]
  __int64 v156; // [rsp+80h] [rbp-1B0h]
  int v158; // [rsp+90h] [rbp-1A0h]
  unsigned int v159; // [rsp+90h] [rbp-1A0h]
  unsigned int v160; // [rsp+90h] [rbp-1A0h]
  __int64 v161; // [rsp+90h] [rbp-1A0h]
  __int64 v162; // [rsp+90h] [rbp-1A0h]
  unsigned __int64 v163; // [rsp+98h] [rbp-198h]
  __int64 v164; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v165; // [rsp+E8h] [rbp-148h] BYREF
  __m128i v166; // [rsp+F0h] [rbp-140h] BYREF
  __int64 *v167; // [rsp+100h] [rbp-130h]
  __int64 v168; // [rsp+108h] [rbp-128h]
  __int64 v169; // [rsp+110h] [rbp-120h] BYREF
  unsigned int v170; // [rsp+118h] [rbp-118h]
  _QWORD *v171; // [rsp+120h] [rbp-110h] BYREF
  __int64 v172; // [rsp+128h] [rbp-108h]
  _QWORD v173[6]; // [rsp+130h] [rbp-100h] BYREF
  const __m128i *v174; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v175; // [rsp+168h] [rbp-C8h]
  _BYTE v176[64]; // [rsp+170h] [rbp-C0h] BYREF
  _BYTE *v177; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v178; // [rsp+1B8h] [rbp-78h]
  _BYTE v179[112]; // [rsp+1C0h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a2 + 8);
  v174 = (const __m128i *)v176;
  v4 = *(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL);
  v175 = 0x400000000LL;
  v5 = sub_2E79000(v4);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v177 = 0;
  LOBYTE(v178) = 0;
  sub_34B8C80(v6, v5, v3, (unsigned int)&v174, 0, 0, __PAIR128__(v178, 0));
  v9 = v175;
  if ( !(_DWORD)v175 )
    goto LABEL_2;
  v10 = v179;
  v178 = 0x400000000LL;
  v11 = v179;
  v177 = v179;
  if ( (unsigned int)v175 > 4 )
  {
    sub_C8D5F0((__int64)&v177, v179, (unsigned int)v175, 0x10u, v7, v8);
    v11 = v177;
    v10 = &v177[16 * (unsigned int)v178];
  }
  for ( i = &v11[16 * v9]; i != v10; v10 += 16 )
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *((_DWORD *)v10 + 2) = 0;
    }
  }
  LODWORD(v178) = v9;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v13 = *(__int64 **)(a2 - 8);
  else
    v13 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v14 = sub_338B750(a1, *v13);
  v16 = v15;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v17 = *(_QWORD *)(a2 - 8);
  else
    v17 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v154 = sub_338B750(a1, *(_QWORD *)(v17 + 32));
  v137 = v18;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v19 = *(_QWORD *)(a2 - 8);
  else
    v19 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v20 = sub_338B750(a1, *(_QWORD *)(v19 + 64));
  v170 = v16;
  v139 = v20;
  v21 = *(_QWORD *)(v14 + 48) + 16LL * v16;
  v167 = &v169;
  v169 = v14;
  v168 = 0x100000001LL;
  v22 = *(_WORD *)v21;
  v146 = v23;
  v24 = *(_QWORD *)(v21 + 8);
  LOWORD(v171) = v22;
  v172 = v24;
  if ( v22 )
  {
    if ( (unsigned __int16)(v22 - 17) > 0xD3u )
      v140 = 205;
    else
      v140 = 206;
  }
  else
  {
    v140 = 205 - (!sub_30070B0((__int64)&v171) - 1);
  }
  v141 = 0;
  if ( (unsigned __int8)sub_920620(a2) )
  {
    v25 = *(_BYTE *)(a2 + 1) >> 1;
    v26 = (16 * v25) & 0x20 | 0x40;
    if ( (v25 & 4) == 0 )
      v26 = (16 * v25) & 0x20;
    v27 = v26;
    LOBYTE(v26) = v26 | 0x80;
    if ( (v25 & 8) == 0 )
      v26 = v27;
    v28 = v26;
    BYTE1(v26) |= 1u;
    if ( (v25 & 0x10) == 0 )
      v26 = v28;
    v29 = v26;
    BYTE1(v26) |= 2u;
    if ( (v25 & 0x20) == 0 )
      v26 = v29;
    v30 = v26;
    BYTE1(v26) |= 4u;
    if ( (v25 & 0x40) == 0 )
      v26 = v30;
    v31 = v26;
    BYTE1(v26) |= 8u;
    if ( (*(_BYTE *)(a2 + 1) & 2) == 0 )
      v26 = v31;
    v141 = v26;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v32 = sub_B91C10(a2, 15);
    v33 = v141;
    BYTE1(v33) = BYTE1(v141) | 0x20;
    if ( !v32 )
      v33 = v141;
    v141 = v33;
  }
  v34 = &v174[(unsigned int)v175];
  if ( v174 != v34 )
  {
    v35 = v174 + 1;
    if ( v34 != &v174[1] )
    {
      while ( v35[-1].m128i_i16[0] == v35->m128i_i16[0] )
      {
        if ( !v35->m128i_i16[0] && v35[-1].m128i_i64[1] != v35->m128i_i64[1] )
          goto LABEL_91;
        if ( v34 == ++v35 )
          goto LABEL_45;
      }
      v47 = v168;
      goto LABEL_56;
    }
  }
LABEL_45:
  v36 = *(_QWORD *)(a1 + 864);
  v166 = _mm_loadu_si128(v174);
  v37 = *(_QWORD *)(v36 + 64);
  v38 = *(_BYTE **)(v36 + 16);
  while ( 1 )
  {
    sub_2FE6CC0((__int64)&v171, (__int64)v38, v37, v166.m128i_i64[0], v166.m128i_i64[1]);
    if ( !(_BYTE)v171 )
      break;
    v39 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v38 + 592LL);
    if ( v39 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v171, (__int64)v38, v37, v166.m128i_i64[0], v166.m128i_i64[1]);
      v166.m128i_i16[0] = v172;
      v166.m128i_i64[1] = v173[0];
    }
    else
    {
      v166.m128i_i32[0] = v39((__int64)v38, v37, v166.m128i_u32[0], v166.m128i_i64[1]);
      v166.m128i_i64[1] = v40;
    }
  }
  if ( v166.m128i_i16[0] )
  {
    v41 = 0;
    if ( (unsigned __int16)(v166.m128i_i16[0] - 17) <= 0xD3u )
    {
      v41 = 1;
      if ( *(_QWORD *)&v38[8 * v166.m128i_u16[0] + 112] )
        v41 = (v38[500 * v166.m128i_u16[0] + 6620] & 0xFB) != 0;
    }
  }
  else
  {
    v41 = sub_30070B0((__int64)&v166);
  }
  v42 = (__int64)&v164;
  v164 = 0;
  v165 = 0;
  v43 = sub_99AEC0((_BYTE *)a2, &v164, &v165, 0, 0);
  v44 = v164;
  v171 = (_QWORD *)v43;
  LODWORD(v172) = v45;
  v46 = v165;
  switch ( (int)v43 )
  {
    case 1:
      v108 = 180;
      v109 = v166.m128i_i16[0];
      goto LABEL_105;
    case 2:
      v108 = 182;
      v109 = v166.m128i_i16[0];
      goto LABEL_105;
    case 3:
      v108 = 181;
      v109 = v166.m128i_i16[0];
      goto LABEL_105;
    case 4:
      v108 = 183;
      v109 = v166.m128i_i16[0];
      goto LABEL_105;
    case 5:
      if ( HIDWORD(v43) != 3 )
      {
        if ( HIDWORD(v43) > 3 )
          goto LABEL_91;
        if ( HIDWORD(v43) )
        {
          if ( HIDWORD(v43) == 2 )
          {
            v108 = 279;
            v109 = v166.m128i_i16[0];
            goto LABEL_105;
          }
          goto LABEL_91;
        }
LABEL_171:
        BUG();
      }
      v109 = v166.m128i_i16[0];
      if ( v166.m128i_i16[0] == 1 )
      {
        v122 = v38[7193];
        if ( !v122 )
        {
          v108 = 279;
          goto LABEL_133;
        }
      }
      else
      {
        if ( !v166.m128i_i16[0] || (v111 = v166.m128i_u16[0], !*(_QWORD *)&v38[8 * v166.m128i_u16[0] + 112]) )
        {
LABEL_152:
          if ( !v41 )
            goto LABEL_91;
          v136 = v165;
          v162 = v164;
          v120 = sub_3281100((unsigned __int16 *)&v166, (__int64)&v164);
          v42 = 279;
          if ( !(unsigned __int8)sub_3365F70((__int64)v38, 0x117u, v120, 0, v121) )
            goto LABEL_109;
          v109 = v166.m128i_i16[0];
          v44 = v162;
          v46 = v136;
LABEL_155:
          v108 = 279;
          goto LABEL_105;
        }
        v122 = v38[500 * v166.m128i_u16[0] + 6693];
        if ( !v122 )
        {
          v108 = 279;
          goto LABEL_126;
        }
      }
      if ( v122 == 4 )
        goto LABEL_155;
      goto LABEL_152;
    case 6:
      if ( HIDWORD(v43) == 3 )
      {
        v109 = v166.m128i_i16[0];
        if ( v166.m128i_i16[0] == 1 )
        {
          v123 = v38[7194];
          if ( !v123 )
          {
            v108 = 280;
            goto LABEL_133;
          }
        }
        else
        {
          if ( !v166.m128i_i16[0] || (v111 = v166.m128i_u16[0], !*(_QWORD *)&v38[8 * v166.m128i_u16[0] + 112]) )
          {
LABEL_145:
            if ( !v41 )
              goto LABEL_91;
            v135 = v165;
            v161 = v164;
            v118 = sub_3281100((unsigned __int16 *)&v166, (__int64)&v164);
            v42 = 280;
            if ( !(unsigned __int8)sub_3365F70((__int64)v38, 0x118u, v118, 0, v119) )
              goto LABEL_109;
            v109 = v166.m128i_i16[0];
            v44 = v161;
            v46 = v135;
LABEL_148:
            v108 = 280;
            goto LABEL_105;
          }
          v123 = v38[500 * v166.m128i_u16[0] + 6694];
          if ( !v123 )
          {
            v108 = 280;
            goto LABEL_126;
          }
        }
        if ( v123 == 4 )
          goto LABEL_148;
        goto LABEL_145;
      }
      if ( HIDWORD(v43) > 3 )
        goto LABEL_91;
      if ( !HIDWORD(v43) )
        goto LABEL_171;
      if ( HIDWORD(v43) == 2 )
      {
        v108 = 280;
        v109 = v166.m128i_i16[0];
LABEL_105:
        if ( v109 == 1 )
        {
          v110 = v38[v108 + 6914];
LABEL_107:
          if ( (v110 & 0xFB) == 0 )
            goto LABEL_133;
          LODWORD(v111) = v109;
          if ( !v41 )
          {
LABEL_109:
            v47 = v168;
            goto LABEL_56;
          }
          goto LABEL_138;
        }
        if ( !v109 )
        {
          if ( !v41 )
            goto LABEL_109;
          src = (void *)v46;
          v134 = v44;
          v159 = v108;
          if ( !sub_30070B0((__int64)&v166) )
            goto LABEL_109;
          v113 = sub_3009970((__int64)&v166, v42, v112, v159, (__int64)src);
          v46 = (__int64)src;
          v44 = v134;
          v108 = v159;
          LODWORD(v111) = v113;
          goto LABEL_130;
        }
        v111 = v109;
        if ( !*(_QWORD *)&v38[8 * v109 + 112] )
        {
          if ( !v41 )
            goto LABEL_109;
LABEL_138:
          if ( (unsigned __int16)(v109 - 17) > 0xD3u )
          {
            if ( v109 == 1 )
              goto LABEL_132;
            goto LABEL_140;
          }
          LODWORD(v111) = (unsigned __int16)word_4456580[(int)v111 - 1];
LABEL_130:
          if ( (_WORD)v111 == 1 )
          {
            LODWORD(v111) = 1;
            goto LABEL_132;
          }
          if ( !(_WORD)v111 )
            goto LABEL_109;
LABEL_140:
          if ( !*(_QWORD *)&v38[8 * (int)v111 + 112] )
            goto LABEL_109;
LABEL_132:
          if ( (v38[500 * (unsigned int)v111 + 6414 + v108] & 0xFB) != 0 )
            goto LABEL_109;
LABEL_133:
          v114 = *(_QWORD *)(*(_QWORD *)(a2 - 96) + 16LL);
          if ( !v114 )
          {
LABEL_136:
            v160 = v108;
            v153 = v46;
            v154 = sub_338B750(a1, v44);
            v137 = v115;
            v116 = sub_338B750(a1, v153);
            LODWORD(v168) = 0;
            v139 = v116;
            v47 = 0;
            v146 = v117;
            v140 = v160;
            goto LABEL_56;
          }
          while ( **(_BYTE **)(v114 + 24) == 86 )
          {
            v114 = *(_QWORD *)(v114 + 8);
            if ( !v114 )
              goto LABEL_136;
          }
          goto LABEL_109;
        }
LABEL_126:
        v110 = v38[500 * v111 + 6414 + v108];
        goto LABEL_107;
      }
LABEL_91:
      v47 = v168;
LABEL_56:
      v48 = v137;
      v49 = 0;
      v50 = v47;
      v51 = v146 - v137;
      v138 = v137 + v9;
      HIWORD(v9) = v132;
      v133 = v51;
      while ( 1 )
      {
        v72 = v167;
        v73 = 16 * v50;
        v171 = v173;
        v172 = 0x300000000LL;
        if ( v50 <= 3 )
        {
          if ( !v73 )
            goto LABEL_58;
          v74 = v173;
        }
        else
        {
          na = 16 * v50;
          srcc = v167;
          v148 = v50;
          sub_C8D5F0((__int64)&v171, v173, v50, 0x10u, v50, v73);
          LODWORD(v50) = v148;
          v72 = srcc;
          v73 = na;
          v74 = &v171[2 * (unsigned int)v172];
        }
        v149 = v50;
        memcpy(v74, v72, v73);
        LODWORD(v73) = v172;
        LODWORD(v50) = v149;
LABEL_58:
        v52 = v48;
        LODWORD(v172) = v73 + v50;
        v53 = (unsigned int)(v73 + v50);
        v54 = v48 | v142 & 0xFFFFFFFF00000000LL;
        v55 = (unsigned int)v53 + 1LL;
        v142 = v54;
        if ( v55 > HIDWORD(v172) )
        {
          v151 = v54;
          sub_C8D5F0((__int64)&v171, v173, v55, 0x10u, v53, v54);
          v53 = (unsigned int)v172;
          v52 = v48;
          v54 = v151;
        }
        v56 = &v171[2 * v53];
        v56[1] = v54;
        *v56 = v154;
        v58 = v144 & 0xFFFFFFFF00000000LL | (v133 + v48);
        LODWORD(v172) = v172 + 1;
        v57 = (unsigned int)v172;
        v144 = v58;
        if ( (unsigned __int64)(unsigned int)v172 + 1 > HIDWORD(v172) )
        {
          srca = (void *)v52;
          v150 = v58;
          sub_C8D5F0((__int64)&v171, v173, (unsigned int)v172 + 1LL, 0x10u, v58, (unsigned int)v172 + 1LL);
          v57 = (unsigned int)v172;
          v52 = (unsigned __int64)srca;
          v58 = v150;
        }
        v59 = &v171[2 * v57];
        v59[1] = v58;
        *v59 = v139;
        v60 = *(_QWORD *)(a1 + 864);
        v61 = (__int64)v171;
        LODWORD(v172) = v172 + 1;
        v62 = *(_QWORD *)(v154 + 48) + 16 * v52;
        v63 = (unsigned int)v172;
        v64 = *(_QWORD *)(v62 + 8);
        LOWORD(v9) = *(_WORD *)v62;
        v147 = v60;
        LODWORD(v59) = *(_DWORD *)(a1 + 848);
        v65 = *(__m128i **)a1;
        v166.m128i_i64[0] = 0;
        v166.m128i_i32[2] = (int)v59;
        if ( v65 )
        {
          if ( &v166 != &v65[3] )
          {
            v66 = v65[3].m128i_i64[0];
            v166.m128i_i64[0] = v66;
            if ( v66 )
            {
              n = (size_t)v171;
              v127 = (unsigned int)v172;
              srcb = v64;
              sub_B96E90((__int64)&v166, v66, 1);
              v61 = n;
              v63 = v127;
              LODWORD(v64) = srcb;
            }
          }
        }
        v67 = sub_33FBA10(v147, v140, (unsigned int)&v166, v9, v64, v141, v61, v63);
        v69 = v68;
        v70 = v67;
        v71 = (unsigned __int64)v177;
        *(_QWORD *)&v177[v49] = v70;
        *(_DWORD *)(v71 + v49 + 8) = v69;
        if ( v166.m128i_i64[0] )
          sub_B91220((__int64)&v166, v166.m128i_i64[0]);
        if ( v171 != v173 )
          _libc_free((unsigned __int64)v171);
        v49 += 16;
        if ( ++v48 == v138 )
          break;
        v50 = (unsigned int)v168;
      }
LABEL_76:
      v75 = *(_QWORD *)(a1 + 864);
      v76 = (unsigned __int64)v177;
      v77 = (unsigned int)v178;
      v78 = sub_33E5830(v75, v174);
      v171 = 0;
      v81 = v78;
      v82 = v79;
      v83 = *(__m128i **)a1;
      LODWORD(v172) = *(_DWORD *)(a1 + 848);
      if ( v83 )
      {
        if ( &v171 != (_QWORD **)&v83[3] )
        {
          v84 = v83[3].m128i_i64[0];
          v171 = (_QWORD *)v84;
          if ( v84 )
          {
            v155 = v79;
            v158 = v81;
            sub_B96E90((__int64)&v171, v84, 1);
            v82 = v155;
            v81 = v158;
          }
        }
      }
      *((_QWORD *)&v124 + 1) = v77;
      *(_QWORD *)&v124 = v76;
      v85 = sub_3411630(v75, 55, (unsigned int)&v171, v81, v82, v80, v124);
      v87 = v86;
      v166.m128i_i64[0] = a2;
      v88 = sub_337DC20(a1 + 8, v166.m128i_i64);
      *v88 = v85;
      *((_DWORD *)v88 + 2) = v87;
      if ( v171 )
        sub_B91220((__int64)&v171, (__int64)v171);
      if ( v167 != &v169 )
        _libc_free((unsigned __int64)v167);
      if ( v177 != v179 )
        _libc_free((unsigned __int64)v177);
LABEL_2:
      if ( v174 != (const __m128i *)v176 )
        _libc_free((unsigned __int64)v174);
      return;
    case 7:
      v143 = 0;
      goto LABEL_93;
    case 8:
      v143 = 1;
LABEL_93:
      v89 = sub_338B750(a1, v164);
      LODWORD(v168) = 0;
      v145 = v89;
      LODWORD(v89) = v9 + v90;
      v91 = v90;
      v92 = 0;
      v152 = v89;
      do
      {
        v93 = *(_DWORD *)(a1 + 848);
        v94 = *(__m128i **)a1;
        v171 = 0;
        LODWORD(v172) = v93;
        if ( v94 )
        {
          if ( &v171 != (_QWORD **)&v94[3] )
          {
            v95 = v94[3].m128i_i64[0];
            v171 = (_QWORD *)v95;
            if ( v95 )
              sub_B96E90((__int64)&v171, v95, 1);
          }
        }
        v96 = (unsigned __int16 *)(*(_QWORD *)(v145 + 48) + 16LL * v91);
        v97 = *v96;
        v156 = *((_QWORD *)v96 + 1);
        v163 = v91 | v163 & 0xFFFFFFFF00000000LL;
        v98 = sub_33FAF80(*(_QWORD *)(a1 + 864), 189, (unsigned int)&v171, v97, v156, v156);
        v100 = v99;
        v101 = v98;
        v102 = (unsigned __int64)v177;
        *(_QWORD *)&v177[v92] = v101;
        *(_DWORD *)(v102 + v92 + 8) = v100;
        if ( v143 )
        {
          v103 = sub_3407430(*(_QWORD *)(a1 + 864), *(_QWORD *)&v177[v92], *(_QWORD *)&v177[v92 + 8], &v171, v97, v156);
          v105 = v104;
          v106 = v103;
          v107 = (unsigned __int64)v177;
          *(_QWORD *)&v177[v92] = v106;
          *(_DWORD *)(v107 + v92 + 8) = v105;
        }
        if ( v171 )
          sub_B91220((__int64)&v171, (__int64)v171);
        v92 += 16;
        ++v91;
      }
      while ( v152 != v91 );
      goto LABEL_76;
    default:
      goto LABEL_91;
  }
}
