// Function: sub_1024C90
// Address: 0x1024c90
//
__int64 __fastcall sub_1024C90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 p_src; // r14
  __int64 v11; // r12
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int8 v21; // al
  unsigned __int8 *v22; // r8
  void *v23; // r9
  __int64 v24; // rax
  __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 *v28; // rax
  _BYTE *v29; // r8
  __int64 *v30; // r13
  __int64 v31; // r13
  __int64 v32; // rdx
  char *v33; // rax
  char *v34; // rcx
  __int64 v35; // rdx
  char *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 v39; // rbx
  __int64 v40; // r8
  __int64 v41; // r9
  int v42; // ebx
  unsigned __int64 v43; // rax
  int v44; // eax
  __int64 v45; // rdx
  unsigned __int64 v46; // rax
  __int64 v47; // rbx
  _QWORD *v48; // rax
  unsigned int v49; // eax
  unsigned __int8 **v50; // rdx
  __int64 v51; // rcx
  unsigned __int8 *v52; // r13
  unsigned __int8 **v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rax
  unsigned __int8 *v56; // rbx
  unsigned __int8 *v57; // r13
  _BYTE *v58; // r15
  _QWORD *v59; // rax
  _QWORD *v60; // rdx
  _QWORD *v61; // rax
  _QWORD *v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rax
  __int64 v67; // rax
  bool v68; // zf
  __int64 v69; // r9
  __int64 v70; // rax
  bool v71; // r13
  char v72; // al
  unsigned __int8 v73; // dl
  unsigned __int8 v74; // cl
  __int64 v75; // rcx
  __m128i v76; // xmm0
  char v77; // al
  int v78; // eax
  int v79; // edx
  _BYTE *v80; // rdi
  __int64 v81; // rcx
  __int64 v82; // r13
  __int64 v83; // r15
  __int64 v84; // r12
  __int64 *v85; // rax
  __int64 *v86; // rdx
  __int64 *v87; // rax
  char v88; // al
  char v89; // al
  int v90; // r13d
  __int64 v91; // rax
  void *v92; // r9
  unsigned __int64 v93; // rdx
  __int64 v94; // r8
  int v95; // r13d
  __int64 v96; // rax
  unsigned __int64 v97; // rdx
  __int64 v98; // rdx
  _QWORD *v99; // rax
  __int64 v100; // rdx
  _QWORD *v101; // rdx
  __int64 v102; // rax
  _QWORD *v103; // rdx
  char v104; // dl
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  __int64 *v108; // rcx
  __int64 *v109; // rax
  __int64 *v110; // rcx
  __int64 v111; // rax
  unsigned __int64 v112; // rdx
  __int64 v113; // rax
  unsigned __int64 v114; // rdx
  unsigned __int8 *v115; // rdi
  __int64 v116; // rdx
  unsigned __int8 *v117; // rsi
  unsigned int v118; // r13d
  unsigned __int64 v119; // rax
  signed int v120; // r13d
  _QWORD *v121; // rax
  __int64 *v122; // rdx
  __int64 *v123; // rcx
  unsigned int v124; // eax
  _BYTE *v125; // rax
  signed __int64 v126; // rdx
  _QWORD *v127; // rdx
  __int64 v128; // rax
  int v129; // eax
  int v130; // eax
  unsigned int v131; // ebx
  void *v132; // rax
  __int64 v133; // rdx
  unsigned __int64 v134; // rax
  __int64 v135; // [rsp+8h] [rbp-368h]
  __int64 v136; // [rsp+8h] [rbp-368h]
  unsigned int v137; // [rsp+10h] [rbp-360h]
  void *v138; // [rsp+10h] [rbp-360h]
  void *v139; // [rsp+10h] [rbp-360h]
  _BYTE *v140; // [rsp+18h] [rbp-358h]
  _BYTE *v141; // [rsp+18h] [rbp-358h]
  unsigned __int8 v142; // [rsp+23h] [rbp-34Dh]
  unsigned int v143; // [rsp+24h] [rbp-34Ch]
  __int64 v144; // [rsp+28h] [rbp-348h]
  char v145; // [rsp+28h] [rbp-348h]
  unsigned __int8 v146; // [rsp+28h] [rbp-348h]
  __int64 v147; // [rsp+30h] [rbp-340h]
  char v148; // [rsp+40h] [rbp-330h]
  __int64 v149; // [rsp+48h] [rbp-328h]
  __int64 v150; // [rsp+50h] [rbp-320h]
  __int64 v151; // [rsp+58h] [rbp-318h]
  __int64 v152; // [rsp+60h] [rbp-310h]
  unsigned __int8 *v153; // [rsp+60h] [rbp-310h]
  _BYTE *v154; // [rsp+68h] [rbp-308h]
  __int64 v155; // [rsp+68h] [rbp-308h]
  unsigned int v156; // [rsp+68h] [rbp-308h]
  char v158; // [rsp+78h] [rbp-2F8h]
  int v160; // [rsp+88h] [rbp-2E8h]
  unsigned int v161; // [rsp+88h] [rbp-2E8h]
  char v163; // [rsp+8Ch] [rbp-2E4h]
  __m128i v164; // [rsp+90h] [rbp-2E0h] BYREF
  __m128i v165; // [rsp+A0h] [rbp-2D0h] BYREF
  __m128i v166; // [rsp+B0h] [rbp-2C0h] BYREF
  __m128i v167; // [rsp+C0h] [rbp-2B0h]
  _QWORD v168[4]; // [rsp+D0h] [rbp-2A0h] BYREF
  char v169[32]; // [rsp+F0h] [rbp-280h] BYREF
  char v170[32]; // [rsp+110h] [rbp-260h] BYREF
  char v171[32]; // [rsp+130h] [rbp-240h] BYREF
  __int64 v172; // [rsp+150h] [rbp-220h] BYREF
  __int64 *v173; // [rsp+158h] [rbp-218h]
  __int64 v174; // [rsp+160h] [rbp-210h]
  int v175; // [rsp+168h] [rbp-208h]
  char v176; // [rsp+16Ch] [rbp-204h]
  char v177; // [rsp+170h] [rbp-200h] BYREF
  _BYTE *v178; // [rsp+190h] [rbp-1E0h] BYREF
  __int64 v179; // [rsp+198h] [rbp-1D8h]
  _BYTE v180[64]; // [rsp+1A0h] [rbp-1D0h] BYREF
  void *v181; // [rsp+1E0h] [rbp-190h] BYREF
  __int64 v182; // [rsp+1E8h] [rbp-188h]
  _QWORD v183[8]; // [rsp+1F0h] [rbp-180h] BYREF
  __int64 v184; // [rsp+230h] [rbp-140h] BYREF
  __int64 *v185; // [rsp+238h] [rbp-138h]
  __int64 v186; // [rsp+240h] [rbp-130h]
  int v187; // [rsp+248h] [rbp-128h]
  char v188; // [rsp+24Ch] [rbp-124h]
  char v189; // [rsp+250h] [rbp-120h] BYREF
  void *src; // [rsp+290h] [rbp-E0h] BYREF
  __int64 v191; // [rsp+298h] [rbp-D8h] BYREF
  __int64 v192; // [rsp+2A0h] [rbp-D0h] BYREF
  __int64 v193; // [rsp+2A8h] [rbp-C8h]
  __int64 v194; // [rsp+2B0h] [rbp-C0h] BYREF
  int v195; // [rsp+2B8h] [rbp-B8h]
  int v196; // [rsp+2BCh] [rbp-B4h]
  __int64 v197; // [rsp+2C0h] [rbp-B0h]
  __int64 v198; // [rsp+2C8h] [rbp-A8h]
  __int16 v199; // [rsp+2D0h] [rbp-A0h]
  char v200[8]; // [rsp+2D8h] [rbp-98h] BYREF
  __int64 v201; // [rsp+2E0h] [rbp-90h]
  char v202; // [rsp+2F4h] [rbp-7Ch]
  int v203; // [rsp+338h] [rbp-38h]

  LODWORD(p_src) = 0;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return (unsigned int)p_src;
  v11 = a3;
  if ( *(_QWORD *)(a1 + 40) != **(_QWORD **)(a3 + 32) )
    return (unsigned int)p_src;
  v13 = a2;
  v14 = sub_D4B130(a3);
  v15 = *(_QWORD *)(a1 - 8);
  v16 = v14;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
  {
    v17 = 0;
    a2 = v15 + 32LL * *(unsigned int *)(a1 + 72);
    while ( v16 != *(_QWORD *)(a2 + 8 * v17) )
    {
      if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == (_DWORD)++v17 )
        goto LABEL_109;
    }
    v18 = 32 * v17;
  }
  else
  {
LABEL_109:
    v18 = 0x1FFFFFFFE0LL;
  }
  v19 = *(_QWORD *)(v15 + v18);
  v20 = *(_QWORD *)(a1 + 8);
  v167 = 0u;
  v149 = v19;
  v173 = (__int64 *)&v177;
  v185 = (__int64 *)&v189;
  v178 = v180;
  v174 = 4;
  v175 = 0;
  v176 = 1;
  v186 = 8;
  v187 = 0;
  v188 = 1;
  v179 = 0x800000000LL;
  v21 = *(_BYTE *)(v20 + 8);
  v166 = 0u;
  v151 = v20;
  v172 = 0;
  v184 = 0;
  if ( v21 <= 3u || v21 == 5 || (v21 & 0xFD) == 4 )
  {
    v154 = (_BYTE *)a1;
    if ( !(unsigned __int8)sub_1021EE0(v13) )
      goto LABEL_116;
    goto LABEL_11;
  }
  LODWORD(p_src) = 0;
  if ( v21 != 12 )
    goto LABEL_107;
  if ( !sub_1021EC0(v13) )
    goto LABEL_116;
  v154 = (_BYTE *)a1;
  if ( v13 - 6 <= 3 )
    goto LABEL_11;
  if ( v13 - 12 <= 3 )
    goto LABEL_11;
  v70 = *(_QWORD *)(a1 + 16);
  if ( !v70 || *(_QWORD *)(v70 + 8) )
    goto LABEL_11;
  v154 = *(_BYTE **)(v70 + 24);
  if ( *v154 == 57 && **((_BYTE **)v154 - 8) > 0x1Cu )
  {
    v115 = (unsigned __int8 *)*((_QWORD *)v154 - 4);
    v116 = *v115;
    v117 = v115 + 24;
    if ( (_BYTE)v116 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v115 + 1) + 8LL) - 17 > 1 )
        goto LABEL_132;
      if ( (unsigned __int8)v116 > 0x15u )
        goto LABEL_132;
      v125 = sub_AD7630((__int64)v115, 0, v116);
      if ( !v125 || *v125 != 17 )
        goto LABEL_132;
      v117 = v125 + 24;
    }
    LODWORD(v182) = *((_DWORD *)v117 + 2);
    if ( (unsigned int)v182 > 0x40 )
      sub_C43780((__int64)&v181, (const void **)v117);
    else
      v181 = *(void **)v117;
    sub_C46A40((__int64)&v181, 1);
    v118 = v182;
    v22 = (unsigned __int8 *)v181;
    LODWORD(v182) = 0;
    LODWORD(v191) = v118;
    src = v181;
    if ( v118 > 0x40 )
    {
      p_src = (__int64)&src;
      v153 = (unsigned __int8 *)v181;
      v129 = sub_C44630((__int64)&src);
      v22 = v153;
      if ( v129 == 1 )
      {
        v130 = sub_C444A0((__int64)&src);
        v22 = v153;
        v120 = v118 - 1 - v130;
      }
      else
      {
        v120 = -1;
      }
      if ( v22 )
        j_j___libc_free_0_0(v22);
LABEL_310:
      if ( (unsigned int)v182 > 0x40 && v181 )
        j_j___libc_free_0_0(v181);
      if ( v120 > 0 )
      {
        v121 = (_QWORD *)sub_BD5C60(a1);
        v151 = sub_BCCE00(v121, v120);
        if ( !v188 )
          goto LABEL_398;
        v122 = v185;
        v123 = &v185[HIDWORD(v186)];
        while ( v123 != v122 )
        {
          if ( a1 == *v122 )
            goto LABEL_321;
          ++v122;
        }
        if ( HIDWORD(v186) < (unsigned int)v186 )
        {
          ++HIDWORD(v186);
          *v123 = a1;
          ++v184;
        }
        else
        {
LABEL_398:
          sub_C8CC70((__int64)&v184, a1, (__int64)v122, (__int64)v123, (__int64)v22, (__int64)v23);
        }
LABEL_321:
        if ( !v176 )
          goto LABEL_399;
        v122 = v173;
        v123 = &v173[HIDWORD(v174)];
        while ( v123 != v122 )
        {
          if ( v154 == (_BYTE *)*v122 )
            goto LABEL_11;
          ++v122;
        }
        if ( HIDWORD(v174) < (unsigned int)v174 )
        {
          ++HIDWORD(v174);
          *v123 = (__int64)v154;
          ++v172;
        }
        else
        {
LABEL_399:
          sub_C8CC70((__int64)&v172, (__int64)v154, (__int64)v122, (__int64)v123, (__int64)v22, (__int64)v23);
        }
        goto LABEL_11;
      }
      goto LABEL_132;
    }
    if ( v181 && ((unsigned __int64)v181 & ((unsigned __int64)v181 - 1)) == 0 )
    {
      _BitScanReverse64(&v119, (unsigned __int64)v181);
      v120 = 63 - (v119 ^ 0x3F);
      goto LABEL_310;
    }
  }
LABEL_132:
  v154 = (_BYTE *)a1;
LABEL_11:
  v24 = (unsigned int)v179;
  v25 = HIDWORD(v179);
  v26 = (unsigned int)v179 + 1LL;
  if ( v26 > HIDWORD(v179) )
  {
    sub_C8D5F0((__int64)&v178, v180, v26, 8u, (__int64)v22, (__int64)v23);
    v24 = (unsigned int)v179;
  }
  *(_QWORD *)&v178[8 * v24] = v154;
  v27 = (unsigned int)(v179 + 1);
  LODWORD(v179) = v179 + 1;
  if ( !v188 )
    goto LABEL_110;
  v28 = v185;
  a2 = HIDWORD(v186);
  v25 = (__int64)&v185[HIDWORD(v186)];
  if ( v185 != (__int64 *)v25 )
  {
    while ( v154 != (_BYTE *)*v28 )
    {
      if ( (__int64 *)v25 == ++v28 )
        goto LABEL_221;
    }
    goto LABEL_18;
  }
LABEL_221:
  if ( HIDWORD(v186) < (unsigned int)v186 )
  {
    a2 = (unsigned int)++HIDWORD(v186);
    *(_QWORD *)v25 = v154;
    LODWORD(v27) = v179;
    ++v184;
  }
  else
  {
LABEL_110:
    a2 = (__int64)v154;
    sub_C8CC70((__int64)&v184, (__int64)v154, v27, v25, (__int64)v22, (__int64)v23);
    LODWORD(v27) = v179;
  }
LABEL_18:
  if ( !(_DWORD)v27 )
  {
    if ( v13 - 6 > 3 )
    {
      v148 = 0;
      v147 = 0;
      v142 = 0;
      v143 = -1;
      v150 = 0;
      v152 = 0;
      goto LABEL_291;
    }
    if ( v13 - 17 <= 1 )
      goto LABEL_116;
    v147 = 0;
    v152 = 0;
    v142 = 0;
    v143 = -1;
    v148 = 0;
    v150 = 0;
    goto LABEL_113;
  }
  v143 = -1;
  v147 = 0;
  v160 = 0;
  v148 = 0;
  v142 = 0;
  v150 = 0;
  v152 = 0;
  do
  {
    v29 = v178;
    p_src = *(_QWORD *)&v178[8 * (unsigned int)v27 - 8];
    LODWORD(v179) = v27 - 1;
    if ( *(_BYTE *)p_src == 62 )
    {
      if ( !a9 )
      {
LABEL_125:
        LODWORD(p_src) = 0;
        goto LABEL_117;
      }
      v30 = sub_DD8400(a9, *(_QWORD *)(p_src - 32));
      if ( !v150 || (a2 = *(_QWORD *)(v150 - 32), v30 == sub_DD8400(a9, a2)) )
      {
        a2 = (__int64)v30;
        if ( sub_DADE90(a9, (__int64)v30, v11) )
        {
          v150 = p_src;
          goto LABEL_26;
        }
      }
LABEL_124:
      v29 = v178;
      goto LABEL_125;
    }
    if ( !*(_QWORD *)(p_src + 16) )
      goto LABEL_125;
    v71 = a1 != p_src && *(_BYTE *)p_src == 84;
    if ( v71 )
    {
      if ( *(_QWORD *)(p_src + 40) == *(_QWORD *)(a1 + 40) )
        goto LABEL_125;
      v145 = 0;
      if ( v154 == (_BYTE *)p_src )
        goto LABEL_123;
      goto LABEL_138;
    }
    v146 = *(_BYTE *)p_src;
    v140 = v178;
    v72 = sub_B46D50((unsigned __int8 *)p_src);
    v73 = v146;
    v74 = v146 - 82;
    v145 = v146 != 84;
    LOBYTE(a2) = v74 > 2u;
    if ( v73 == 86 || v74 <= 2u )
      goto LABEL_137;
    v29 = v140;
    if ( v72 == 1 )
      goto LABEL_137;
    if ( (*(_BYTE *)(p_src + 7) & 0x40) != 0 )
      v108 = *(__int64 **)(p_src - 8);
    else
      v108 = (__int64 *)(p_src - 32LL * (*(_DWORD *)(p_src + 4) & 0x7FFFFFF));
    a2 = *v108;
    if ( *(_BYTE *)*v108 <= 0x1Cu )
      a2 = 0;
    if ( !v188 )
    {
      if ( !sub_C8CA60((__int64)&v184, a2) )
        goto LABEL_124;
LABEL_137:
      if ( v154 == (_BYTE *)p_src )
      {
LABEL_149:
        if ( *(_BYTE *)p_src == 86 )
        {
          if ( v13 - 10 <= 1 )
          {
            a2 = (__int64)&v184;
            if ( (unsigned __int8)sub_10228B0(p_src, (__int64)&v184, 2u) )
              goto LABEL_124;
            if ( !v71 )
              goto LABEL_160;
            if ( !(unsigned __int8)sub_1021E10(p_src, (__int64)&v184) )
              goto LABEL_124;
            goto LABEL_156;
          }
        }
        else
        {
          if ( !v145 )
            goto LABEL_244;
          if ( v13 - 6 <= 3 )
          {
            if ( !v71 )
            {
              v73 = *(_BYTE *)p_src;
              goto LABEL_154;
            }
            a2 = (__int64)&v184;
            v145 = sub_1021E10(p_src, (__int64)&v184);
            if ( !v145 )
              goto LABEL_124;
            goto LABEL_247;
          }
LABEL_272:
          if ( v13 - 12 > 3 )
          {
            v145 = 1;
            if ( v13 - 17 > 1 )
            {
              a2 = (__int64)&v184;
              if ( (unsigned __int8)sub_10228B0(p_src, (__int64)&v184, 1u) )
                goto LABEL_124;
            }
          }
          else
          {
            v145 = 1;
          }
        }
LABEL_244:
        if ( v71 )
        {
LABEL_123:
          a2 = (__int64)&v184;
          if ( !(unsigned __int8)sub_1021E10(p_src, (__int64)&v184) )
            goto LABEL_124;
        }
LABEL_245:
        if ( v13 - 6 > 3 && v13 != 17 )
        {
LABEL_156:
          if ( (v13 - 12 <= 3 || v13 == 18) && (*(_BYTE *)p_src == 83 || *(_BYTE *)p_src == 86) )
            ++v160;
          goto LABEL_160;
        }
LABEL_247:
        if ( (*(_BYTE *)p_src & 0xFB) == 0x52 )
LABEL_155:
          ++v160;
        goto LABEL_156;
      }
LABEL_138:
      a2 = v11;
      sub_1024900((__int64)&v164, v11, (_BYTE *)a1, p_src, v13, (__int64)&v166, a4, a9);
      v75 = v147;
      v76 = _mm_loadu_si128(&v164);
      v167 = _mm_loadu_si128(&v165);
      if ( !v147 )
        v75 = v167.m128i_i64[1];
      v166 = v76;
      v147 = v75;
      if ( !v164.m128i_i8[0] )
        goto LABEL_124;
      v77 = sub_920620(v166.m128i_i64[1]);
      if ( v145 && v77 )
      {
        v141 = (_BYTE *)v166.m128i_i64[1];
        v78 = sub_B45210(v166.m128i_i64[1]);
        v79 = v78;
        if ( *v141 == 86 )
        {
          v80 = (_BYTE *)*((_QWORD *)v141 - 12);
          if ( *v80 == 83 )
            v79 = sub_B45210((__int64)v80) | v78;
        }
        v143 &= v79;
      }
      if ( v167.m128i_i32[0] )
        v13 = v167.m128i_i32[0];
      goto LABEL_149;
    }
    v109 = v185;
    v110 = &v185[HIDWORD(v186)];
    if ( v185 == v110 )
      goto LABEL_125;
    while ( a2 != *v109 )
    {
      if ( v110 == ++v109 )
        goto LABEL_125;
    }
    if ( v154 != (_BYTE *)p_src )
      goto LABEL_138;
    if ( v73 == 84 )
    {
      v145 = 0;
      goto LABEL_245;
    }
    if ( v13 - 6 > 3 )
      goto LABEL_272;
LABEL_154:
    v145 = 1;
    if ( (v73 & 0xFB) == 0x52 )
      goto LABEL_155;
LABEL_160:
    v81 = (__int64)&v192;
    src = &v192;
    v181 = v183;
    v182 = 0x800000000LL;
    v191 = 0x800000000LL;
    v82 = *(_QWORD *)(p_src + 16);
    if ( !v82 )
      goto LABEL_183;
    v137 = v13;
    v83 = v11;
    do
    {
      v84 = *(_QWORD *)(v82 + 24);
      if ( *(_BYTE *)v84 == 85 )
      {
        v102 = *(_QWORD *)(v84 - 32);
        if ( v102 )
        {
          if ( !*(_BYTE *)v102 )
          {
            v81 = *(_QWORD *)(v84 + 80);
            if ( *(_QWORD *)(v102 + 24) == v81 && (*(_BYTE *)(v102 + 33) & 0x20) != 0 && *(_DWORD *)(v102 + 36) == 174 )
            {
              v103 = (*(_BYTE *)(v84 + 7) & 0x40) != 0
                   ? *(_QWORD **)(v84 - 8)
                   : (_QWORD *)(v84 - 32LL * (*(_DWORD *)(v84 + 4) & 0x7FFFFFF));
              if ( p_src == *v103 || p_src == v103[4] )
              {
LABEL_216:
                if ( src != &v192 )
                  _libc_free(src, a2);
                if ( v181 != v183 )
                {
                  _libc_free(v181, a2);
                  v29 = v178;
                  goto LABEL_125;
                }
                goto LABEL_124;
              }
            }
          }
        }
      }
      a2 = *(_QWORD *)(v84 + 40);
      if ( *(_BYTE *)(v83 + 84) )
      {
        v85 = *(__int64 **)(v83 + 64);
        v86 = &v85[*(unsigned int *)(v83 + 76)];
        if ( v85 == v86 )
          goto LABEL_197;
        while ( a2 != *v85 )
        {
          if ( v86 == ++v85 )
            goto LABEL_197;
        }
LABEL_168:
        memset(v168, 0, sizeof(v168));
        if ( !v188 )
          goto LABEL_234;
        v87 = v185;
        v81 = HIDWORD(v186);
        v86 = &v185[HIDWORD(v186)];
        if ( v185 == v86 )
        {
LABEL_255:
          if ( HIDWORD(v186) < (unsigned int)v186 )
          {
            ++HIDWORD(v186);
            *v86 = v84;
            ++v184;
            v88 = *(_BYTE *)v84;
            goto LABEL_235;
          }
LABEL_234:
          a2 = v84;
          sub_C8CC70((__int64)&v184, v84, (__int64)v86, v81, (__int64)v29, v69);
          v88 = *(_BYTE *)v84;
          if ( v104 )
          {
LABEL_235:
            if ( v88 == 84 )
            {
              v111 = (unsigned int)v191;
              v112 = (unsigned int)v191 + 1LL;
              if ( v112 > HIDWORD(v191) )
              {
                a2 = (__int64)&v192;
                sub_C8D5F0((__int64)&src, &v192, v112, 8u, (__int64)v29, v69);
                v111 = (unsigned int)v191;
              }
              *((_QWORD *)src + v111) = v84;
              LODWORD(v191) = v191 + 1;
            }
            else
            {
              if ( v88 == 62 )
              {
                v105 = *(_QWORD *)(v84 - 32);
                if ( p_src == v105 )
                {
                  if ( v105 )
                    goto LABEL_216;
                }
              }
              v106 = (unsigned int)v182;
              v107 = (unsigned int)v182 + 1LL;
              if ( v107 > HIDWORD(v182) )
              {
                a2 = (__int64)v183;
                sub_C8D5F0((__int64)&v181, v183, v107, 8u, (__int64)v29, v69);
                v106 = (unsigned int)v182;
              }
              *((_QWORD *)v181 + v106) = v84;
              LODWORD(v182) = v182 + 1;
            }
LABEL_178:
            v89 = v148;
            v81 = 1;
            if ( a1 == v84 )
              v89 = 1;
            v148 = v89;
            goto LABEL_181;
          }
        }
        else
        {
          while ( v84 != *v87 )
          {
            if ( v86 == ++v87 )
              goto LABEL_255;
          }
          v88 = *(_BYTE *)v84;
        }
        if ( v88 != 84 )
        {
          if ( (unsigned __int8)(v88 - 82) > 1u && v88 != 86 )
            goto LABEL_216;
          a2 = v137;
          sub_1022600((__int64)v169, v137, (_BYTE *)v84);
          if ( !v169[0] )
          {
            a2 = v83;
            sub_1021F00((__int64)v170, v83, (_BYTE *)a1, v84, (__int64)v168);
            v29 = v168;
            if ( !v170[0] )
            {
              a2 = v84;
              sub_1024090((__int64)v171, v84, v137, (__int64)v168);
              if ( !v171[0] )
                goto LABEL_216;
            }
          }
        }
        goto LABEL_178;
      }
      if ( sub_C8CA60(v83 + 56, a2) )
        goto LABEL_168;
LABEL_197:
      if ( p_src == v152 )
        goto LABEL_181;
      if ( v152 || a1 == p_src )
        goto LABEL_216;
      v98 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v99 = *(_QWORD **)(a1 - 8);
        a2 = (__int64)&v99[(unsigned __int64)v98 / 8];
      }
      else
      {
        a2 = a1;
        v99 = (_QWORD *)(a1 - v98);
      }
      v81 = v98 >> 5;
      v100 = v98 >> 7;
      if ( v100 )
      {
        v101 = &v99[16 * v100];
        while ( 1 )
        {
          if ( p_src == *v99 )
            goto LABEL_209;
          if ( p_src == v99[4] )
          {
            v99 += 4;
            goto LABEL_209;
          }
          if ( p_src == v99[8] )
          {
            v99 += 8;
            goto LABEL_209;
          }
          if ( p_src == v99[12] )
            break;
          v99 += 16;
          if ( v99 == v101 )
          {
            v81 = (a2 - (__int64)v99) >> 5;
            goto LABEL_212;
          }
        }
        v99 += 12;
        goto LABEL_209;
      }
LABEL_212:
      if ( v81 == 2 )
        goto LABEL_276;
      if ( v81 != 3 )
      {
        if ( v81 != 1 )
          goto LABEL_216;
LABEL_215:
        if ( p_src != *v99 )
          goto LABEL_216;
        goto LABEL_209;
      }
      if ( p_src != *v99 )
      {
        v99 += 4;
LABEL_276:
        if ( p_src != *v99 )
        {
          v99 += 4;
          goto LABEL_215;
        }
      }
LABEL_209:
      if ( v99 == (_QWORD *)a2 )
        goto LABEL_216;
      v152 = p_src;
LABEL_181:
      v82 = *(_QWORD *)(v82 + 8);
    }
    while ( v82 );
    v11 = v83;
    v13 = v137;
LABEL_183:
    v90 = v191;
    v91 = (unsigned int)v179;
    v92 = src;
    v93 = (unsigned int)v191 + (unsigned __int64)(unsigned int)v179;
    v94 = 8LL * (unsigned int)v191;
    if ( v93 > HIDWORD(v179) )
    {
      a2 = (__int64)v180;
      v136 = 8LL * (unsigned int)v191;
      v139 = src;
      sub_C8D5F0((__int64)&v178, v180, v93, 8u, v94, (__int64)src);
      v91 = (unsigned int)v179;
      v94 = v136;
      v92 = v139;
    }
    if ( v94 )
    {
      a2 = (__int64)v92;
      memcpy(&v178[8 * v91], v92, v94);
      LODWORD(v91) = v179;
    }
    LODWORD(v96) = v90 + v91;
    v95 = v182;
    LODWORD(v179) = v96;
    v96 = (unsigned int)v96;
    v23 = v181;
    v97 = (unsigned int)v182 + (unsigned __int64)(unsigned int)v96;
    v22 = (unsigned __int8 *)(8LL * (unsigned int)v182);
    if ( v97 > HIDWORD(v179) )
    {
      a2 = (__int64)v180;
      v135 = 8LL * (unsigned int)v182;
      v138 = v181;
      sub_C8D5F0((__int64)&v178, v180, v97, 8u, (__int64)v22, (__int64)v181);
      v96 = (unsigned int)v179;
      v22 = (unsigned __int8 *)v135;
      v23 = v138;
    }
    if ( v22 )
    {
      a2 = (__int64)v23;
      memcpy(&v178[8 * v96], v23, (size_t)v22);
    }
    LODWORD(v179) = v95 + v179;
    if ( src != &v192 )
      _libc_free(src, a2);
    if ( v181 != v183 )
      _libc_free(v181, a2);
    LODWORD(p_src) = (v154 != (_BYTE *)p_src) & (unsigned __int8)v145;
    v142 |= p_src;
LABEL_26:
    LODWORD(v27) = v179;
  }
  while ( (_DWORD)v179 );
  LODWORD(v27) = v160;
  if ( v13 - 6 <= 3 )
    goto LABEL_28;
LABEL_291:
  if ( v13 - 12 > 3 )
  {
LABEL_29:
    LOBYTE(p_src) = (_DWORD)v27 != 1 && v13 - 17 <= 1;
    if ( (_BYTE)p_src )
      goto LABEL_116;
    if ( v150 )
    {
      v31 = *(_QWORD *)(v150 - 64);
      v32 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v33 = (char *)(a1 - v32);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v33 = *(char **)(a1 - 8);
      v34 = &v33[v32];
      v35 = v32 >> 7;
      if ( v35 )
      {
        v36 = &v33[128 * v35];
        while ( v31 != *(_QWORD *)v33 )
        {
          if ( v31 == *((_QWORD *)v33 + 4) )
          {
            v33 += 32;
            break;
          }
          if ( v31 == *((_QWORD *)v33 + 8) )
          {
            v33 += 64;
            break;
          }
          if ( v31 == *((_QWORD *)v33 + 12) )
          {
            v33 += 96;
            break;
          }
          v33 += 128;
          if ( v33 == v36 )
            goto LABEL_344;
        }
LABEL_40:
        if ( v34 == v33 )
          goto LABEL_116;
        if ( v152 )
        {
          if ( v31 != v152 || !v148 || !v142 )
            goto LABEL_116;
          goto LABEL_45;
        }
LABEL_114:
        if ( (v142 & (unsigned __int8)v148) != 1 || !v31 )
          goto LABEL_116;
LABEL_45:
        if ( v13 == 10 )
        {
          if ( *(_BYTE *)v31 != 43 )
            goto LABEL_49;
        }
        else
        {
          if ( v13 != 16 )
            goto LABEL_49;
          if ( *(_BYTE *)v31 != 85 )
            goto LABEL_49;
          v128 = *(_QWORD *)(v31 - 32);
          if ( !v128
            || *(_BYTE *)v128
            || *(_QWORD *)(v128 + 24) != *(_QWORD *)(v31 + 80)
            || (*(_BYTE *)(v128 + 33) & 0x20) == 0
            || *(_DWORD *)(v128 + 36) != 174 )
          {
            goto LABEL_49;
          }
        }
        if ( v147 == v31 )
        {
          a2 = 3;
          if ( !(unsigned __int8)sub_BD3660(v147, 3) )
          {
            if ( (*(_BYTE *)(v147 + 7) & 0x40) != 0 )
              v127 = *(_QWORD **)(v147 - 8);
            else
              v127 = (_QWORD *)(v147 - 32LL * (*(_DWORD *)(v147 + 4) & 0x7FFFFFF));
            if ( *v127 == a1 || v13 != 10 )
            {
              v163 = 1;
              if ( v13 == 16 )
                v163 = v127[8] == a1;
            }
            else
            {
              v163 = v127[4] == a1;
            }
            goto LABEL_50;
          }
        }
LABEL_49:
        v163 = 0;
LABEL_50:
        if ( v154 == (_BYTE *)a1 )
        {
          v158 = 0;
        }
        else
        {
          v155 = sub_B43CC0(v31);
          src = (void *)sub_9208B0(v155, *(_QWORD *)(v31 + 8));
          v191 = v37;
          v39 = sub_CA1930(&src);
          if ( a6 )
          {
            sub_D19730((__int64)&src, a6, v31, v38, v40, v41);
            v42 = v191;
            if ( (unsigned int)v191 > 0x40 )
            {
              v39 = v42 - (unsigned int)sub_C444A0((__int64)&src);
              if ( src )
                j_j___libc_free_0_0(src);
            }
            else
            {
              _BitScanReverse64(&v43, (unsigned __int64)src);
              v44 = v43 ^ 0x3F;
              if ( !src )
                v44 = 64;
              v39 = (unsigned int)(64 - v44);
            }
          }
          src = (void *)sub_9208B0(v155, *(_QWORD *)(v31 + 8));
          v191 = v45;
          if ( sub_CA1930(&src) == v39 && (v158 = a8 != 0 && a7 != 0) != 0 )
          {
            v131 = sub_9AF8B0(v31, v155, 0, a7, 0, a8, 1);
            v132 = (void *)sub_9208B0(v155, *(_QWORD *)(v31 + 8));
            v182 = v133;
            v181 = v132;
            v39 = sub_CA1930(&v181) - v131;
            sub_9AC3E0((__int64)&src, v31, v155, 0, 0, 0, 0, 1);
            if ( (unsigned int)v191 > 0x40 )
              v134 = *((_QWORD *)src + ((unsigned int)(v191 - 1) >> 6));
            else
              v134 = (unsigned __int64)src;
            if ( (v134 & (1LL << ((unsigned __int8)v191 - 1))) != 0 )
              v158 = 0;
            else
              ++v39;
            if ( (unsigned int)v193 > 0x40 && v192 )
              j_j___libc_free_0_0(v192);
            if ( (unsigned int)v191 > 0x40 && src )
              j_j___libc_free_0_0(src);
          }
          else
          {
            v158 = 0;
          }
          if ( v39 <= 1 )
          {
            LODWORD(v47) = 1;
          }
          else
          {
            _BitScanReverse64(&v46, v39 - 1);
            v47 = 1LL << (64 - ((unsigned __int8)v46 ^ 0x3Fu));
          }
          v48 = (_QWORD *)sub_BD5C60(v31);
          a2 = (unsigned int)v47;
          if ( v151 != sub_BCD140(v48, v47) )
            goto LABEL_116;
        }
        src = 0;
        v181 = v183;
        v192 = 8;
        LODWORD(v193) = 0;
        BYTE4(v193) = 1;
        v183[0] = v31;
        v156 = -1;
        v144 = v31;
        v161 = v13;
        v191 = (__int64)&v194;
        v182 = 0x800000001LL;
        v49 = 1;
        while ( 1 )
        {
          v50 = (unsigned __int8 **)v181;
          v51 = v49;
          v52 = (unsigned __int8 *)*((_QWORD *)v181 + v49 - 1);
          LODWORD(v182) = v49 - 1;
          if ( !BYTE4(v193) )
            break;
          v53 = (unsigned __int8 **)v191;
          v51 = HIDWORD(v192);
          v50 = (unsigned __int8 **)(v191 + 8LL * HIDWORD(v192));
          if ( (unsigned __int8 **)v191 == v50 )
          {
LABEL_300:
            if ( HIDWORD(v192) >= (unsigned int)v192 )
              break;
            v51 = (unsigned int)++HIDWORD(v192);
            *v50 = v52;
            src = (char *)src + 1;
          }
          else
          {
            while ( v52 != *v53 )
            {
              if ( v50 == ++v53 )
                goto LABEL_300;
            }
          }
LABEL_67:
          if ( (unsigned int)*v52 - 67 > 0xC )
            goto LABEL_70;
          v54 = *(_QWORD *)(*((_QWORD *)v52 - 4) + 8LL);
          if ( v54 == v151 )
          {
            if ( !v176 )
              goto LABEL_353;
            v50 = (unsigned __int8 **)v173;
            v51 = (__int64)&v173[HIDWORD(v174)];
            while ( (unsigned __int8 **)v51 != v50 )
            {
              if ( v52 == *v50 )
                goto LABEL_85;
              ++v50;
            }
            if ( HIDWORD(v174) < (unsigned int)v174 )
            {
              ++HIDWORD(v174);
              *(_QWORD *)v51 = v52;
              ++v172;
            }
            else
            {
LABEL_353:
              a2 = (__int64)v52;
              sub_C8CC70((__int64)&v172, (__int64)v52, (__int64)v50, v51, (__int64)v22, (__int64)v23);
            }
          }
          else
          {
            if ( *((_QWORD *)v52 + 1) != v151 )
            {
LABEL_70:
              v55 = 32LL * (*((_DWORD *)v52 + 1) & 0x7FFFFFF);
              v22 = &v52[-v55];
              if ( (v52[7] & 0x40) != 0 )
                v22 = (unsigned __int8 *)*((_QWORD *)v52 - 1);
              v56 = &v22[v55];
              v57 = v22;
              if ( &v22[v55] != v22 )
              {
                while ( 1 )
                {
                  v58 = *(_BYTE **)v57;
                  if ( **(_BYTE **)v57 > 0x1Cu )
                  {
                    a2 = *((_QWORD *)v58 + 5);
                    if ( *(_BYTE *)(v11 + 84) )
                    {
                      v59 = *(_QWORD **)(v11 + 64);
                      v60 = &v59[*(unsigned int *)(v11 + 76)];
                      if ( v59 == v60 )
                        goto LABEL_84;
                      while ( a2 != *v59 )
                      {
                        if ( v60 == ++v59 )
                          goto LABEL_84;
                      }
                    }
                    else if ( !sub_C8CA60(v11 + 56, a2) )
                    {
                      goto LABEL_84;
                    }
                    if ( BYTE4(v193) )
                    {
                      v61 = (_QWORD *)v191;
                      v62 = (_QWORD *)(v191 + 8LL * HIDWORD(v192));
                      if ( (_QWORD *)v191 != v62 )
                      {
                        while ( v58 != (_BYTE *)*v61 )
                        {
                          if ( v62 == ++v61 )
                            goto LABEL_297;
                        }
                        goto LABEL_84;
                      }
LABEL_297:
                      v113 = (unsigned int)v182;
                      v114 = (unsigned int)v182 + 1LL;
                      if ( v114 > HIDWORD(v182) )
                      {
                        a2 = (__int64)v183;
                        sub_C8D5F0((__int64)&v181, v183, v114, 8u, (__int64)v22, (__int64)v23);
                        v113 = (unsigned int)v182;
                      }
                      *((_QWORD *)v181 + v113) = v58;
                      LODWORD(v182) = v182 + 1;
                      goto LABEL_84;
                    }
                    a2 = (__int64)v58;
                    if ( !sub_C8CA60((__int64)&src, (__int64)v58) )
                      goto LABEL_297;
                  }
LABEL_84:
                  v57 += 32;
                  if ( v56 == v57 )
                    goto LABEL_85;
                }
              }
              goto LABEL_85;
            }
            v124 = sub_BCB060(v54);
            if ( v156 <= v124 )
              v124 = v156;
            v156 = v124;
          }
LABEL_85:
          v49 = v182;
          if ( !(_DWORD)v182 )
          {
            if ( !BYTE4(v193) )
              _libc_free(v191, a2);
            if ( v181 != v183 )
              _libc_free(v181, a2);
            sub_1021C30((__int64)&src, v149, v144, v150, v161, v143, v147, v151, v158, v163, (__int64)&v172, v156);
            *(_QWORD *)a5 = src;
            v66 = *(_QWORD *)(a5 + 24);
            if ( v66 != v193 )
            {
              if ( v66 != -4096 && v66 != 0 && v66 != -8192 )
                sub_BD60C0((_QWORD *)(a5 + 8));
              v67 = v193;
              v68 = v193 == 0;
              *(_QWORD *)(a5 + 24) = v193;
              LOBYTE(v63) = !v68;
              if ( v67 != -4096 && !v68 && v67 != -8192 )
                sub_BD6050((unsigned __int64 *)(a5 + 8), v191 & 0xFFFFFFFFFFFFFFF8LL);
            }
            *(_QWORD *)(a5 + 32) = v194;
            a2 = a5 + 104;
            *(_DWORD *)(a5 + 40) = v195;
            *(_DWORD *)(a5 + 44) = v196;
            *(_QWORD *)(a5 + 48) = v197;
            *(_QWORD *)(a5 + 56) = v198;
            *(_WORD *)(a5 + 64) = v199;
            sub_C8CE00(a5 + 72, a5 + 104, (__int64)v200, v63, v64, v65);
            v68 = v202 == 0;
            *(_DWORD *)(a5 + 168) = v203;
            if ( v68 )
              _libc_free(v201, a2);
            if ( v193 != -4096 && v193 != 0 && v193 != -8192 )
              sub_BD60C0(&v191);
            LODWORD(p_src) = 1;
            goto LABEL_103;
          }
        }
        a2 = (__int64)v52;
        sub_C8CC70((__int64)&src, (__int64)v52, (__int64)v50, v51, (__int64)v22, (__int64)v23);
        goto LABEL_67;
      }
LABEL_344:
      v126 = v34 - v33;
      if ( v34 - v33 != 64 )
      {
        if ( v126 != 96 )
        {
          if ( v126 != 32 )
            goto LABEL_103;
          goto LABEL_347;
        }
        if ( v31 == *(_QWORD *)v33 )
          goto LABEL_40;
        v33 += 32;
      }
      if ( v31 == *(_QWORD *)v33 )
        goto LABEL_40;
      v33 += 32;
LABEL_347:
      if ( v31 == *(_QWORD *)v33 )
        goto LABEL_40;
LABEL_103:
      v29 = v178;
      goto LABEL_117;
    }
LABEL_113:
    v31 = v152;
    goto LABEL_114;
  }
LABEL_28:
  if ( (v27 & 0xFFFFFFFD) == 0 )
    goto LABEL_29;
LABEL_116:
  v29 = v178;
  LODWORD(p_src) = 0;
LABEL_117:
  if ( v29 != v180 )
    _libc_free(v29, a2);
  if ( !v188 )
    _libc_free(v185, a2);
LABEL_107:
  if ( !v176 )
    _libc_free(v173, a2);
  return (unsigned int)p_src;
}
