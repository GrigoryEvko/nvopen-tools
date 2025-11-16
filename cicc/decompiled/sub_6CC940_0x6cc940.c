// Function: sub_6CC940
// Address: 0x6cc940
//
_DWORD *__fastcall sub_6CC940(
        __int64 *a1,
        __int16 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BOOL4 a7,
        __int64 a8,
        _DWORD *a9)
{
  __int64 v12; // r13
  char v13; // al
  __int64 v14; // rdi
  char v15; // cl
  __int64 v16; // rsi
  char v17; // al
  char v18; // al
  __int64 v19; // rdi
  _QWORD *v20; // rdx
  __int64 v21; // rcx
  _DWORD *v22; // r8
  __int64 v23; // r9
  unsigned int *v24; // rsi
  __int64 v25; // r11
  _BYTE *v26; // rax
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  __m128i v29; // xmm3
  bool v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 m; // rdx
  __int64 v35; // rcx
  __int64 v36; // rax
  _DWORD *result; // rax
  __int64 v38; // rdx
  char v39; // dl
  unsigned __int64 v40; // xmm4_8
  __m128i v41; // xmm5
  __m128i v42; // xmm6
  __m128i v43; // xmm7
  __int64 v44; // r13
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rcx
  __int64 i; // rax
  __int64 j; // rax
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rax
  bool v55; // zf
  unsigned __int64 v56; // r10
  __int64 v57; // rax
  char k; // dl
  __int64 v59; // rax
  int v60; // eax
  __int64 ii; // rax
  __int64 v62; // rax
  __int64 v63; // rdi
  int v64; // eax
  __int64 v65; // rdi
  __int64 v66; // r8
  __int64 v67; // rax
  __int64 v68; // rdi
  char v69; // al
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // r10
  __int64 v73; // r10
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 v76; // rcx
  __int64 v77; // rbx
  __int64 v78; // r13
  _QWORD *v79; // r14
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rax
  __int16 v85; // ax
  _DWORD *v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  char v90; // al
  __int64 v91; // rax
  unsigned int v92; // eax
  __int64 *v93; // rdi
  __int64 v94; // rbx
  unsigned int v95; // eax
  __int64 v96; // rdi
  int v97; // eax
  __int64 v98; // rbx
  int v99; // eax
  __int16 v100; // ax
  _BOOL4 v101; // eax
  __int64 v102; // rax
  __int64 v103; // rax
  _QWORD *v104; // r13
  __int64 v105; // rbx
  __int64 v106; // rax
  int v107; // eax
  unsigned __int64 v108; // rsi
  unsigned __int64 jj; // rdx
  unsigned int v110; // edx
  _QWORD *v111; // rax
  int v112; // eax
  __int64 v113; // rax
  __int64 v114; // r13
  int v115; // eax
  __int64 v116; // rax
  __int64 *v117; // rdi
  int v118; // eax
  __int64 v119; // r13
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rax
  char v123; // al
  __int64 v124; // rdi
  __int64 v125; // rax
  int v126; // eax
  int v127; // eax
  int v128; // eax
  __int64 v129; // rdi
  __int64 v130; // rbx
  __int64 v131; // rax
  __int64 v132; // r10
  __int64 v133; // rdi
  char v134; // al
  unsigned __int64 v135; // r10
  __int64 v136; // rax
  int v137; // eax
  __int16 v138; // ax
  __int64 v139; // rdx
  __int64 v140; // rcx
  int v141; // eax
  __int64 v142; // rax
  __int64 n; // rdi
  char v144; // al
  int v145; // eax
  char v146; // al
  __int64 v147; // rax
  __int64 v148; // rax
  char v149; // al
  __int64 v150; // rax
  int v151; // eax
  __int16 v152; // ax
  __int64 v153; // rax
  __int64 v154; // rax
  unsigned int v155; // eax
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v158; // [rsp+8h] [rbp-398h]
  __int64 v159; // [rsp+10h] [rbp-390h]
  unsigned int v160; // [rsp+10h] [rbp-390h]
  int v161; // [rsp+10h] [rbp-390h]
  __int64 v162; // [rsp+10h] [rbp-390h]
  __int64 v163; // [rsp+10h] [rbp-390h]
  __int64 v164; // [rsp+10h] [rbp-390h]
  unsigned int v165; // [rsp+10h] [rbp-390h]
  int v166; // [rsp+18h] [rbp-388h]
  bool v167; // [rsp+18h] [rbp-388h]
  __int64 v168; // [rsp+18h] [rbp-388h]
  __int64 v169; // [rsp+18h] [rbp-388h]
  __int64 v170; // [rsp+18h] [rbp-388h]
  __int64 v171; // [rsp+18h] [rbp-388h]
  unsigned int v172; // [rsp+18h] [rbp-388h]
  int v174; // [rsp+20h] [rbp-380h]
  __int16 v175; // [rsp+28h] [rbp-378h]
  int v176; // [rsp+28h] [rbp-378h]
  __int64 v177; // [rsp+28h] [rbp-378h]
  unsigned __int64 v179; // [rsp+30h] [rbp-370h]
  unsigned __int64 v180; // [rsp+30h] [rbp-370h]
  unsigned __int64 v181; // [rsp+30h] [rbp-370h]
  __int64 v182; // [rsp+30h] [rbp-370h]
  int v183; // [rsp+30h] [rbp-370h]
  __int64 v184; // [rsp+30h] [rbp-370h]
  int v185; // [rsp+30h] [rbp-370h]
  __int64 v186; // [rsp+30h] [rbp-370h]
  __int64 v187; // [rsp+30h] [rbp-370h]
  __int64 v188; // [rsp+30h] [rbp-370h]
  __int64 v189; // [rsp+30h] [rbp-370h]
  __int64 v190; // [rsp+30h] [rbp-370h]
  __int64 v191; // [rsp+30h] [rbp-370h]
  __int64 v192; // [rsp+30h] [rbp-370h]
  unsigned int v193; // [rsp+38h] [rbp-368h]
  unsigned int v194; // [rsp+38h] [rbp-368h]
  unsigned int v195; // [rsp+38h] [rbp-368h]
  int v196; // [rsp+3Ch] [rbp-364h]
  unsigned __int8 v197; // [rsp+47h] [rbp-359h] BYREF
  int v198; // [rsp+48h] [rbp-358h] BYREF
  unsigned int v199; // [rsp+4Ch] [rbp-354h] BYREF
  int v200; // [rsp+50h] [rbp-350h] BYREF
  int v201; // [rsp+54h] [rbp-34Ch] BYREF
  __int64 v202; // [rsp+58h] [rbp-348h] BYREF
  __int64 v203; // [rsp+60h] [rbp-340h] BYREF
  __int64 v204; // [rsp+68h] [rbp-338h] BYREF
  __m128i v205; // [rsp+70h] [rbp-330h] BYREF
  __m128i v206; // [rsp+80h] [rbp-320h]
  __m128i v207; // [rsp+90h] [rbp-310h]
  __m128i v208; // [rsp+A0h] [rbp-300h]
  _QWORD v209[44]; // [rsp+B0h] [rbp-2F0h] BYREF
  __int64 *v210[50]; // [rsp+210h] [rbp-190h] BYREF

  v198 = 0;
  if ( !a4 )
  {
    v203 = *(_QWORD *)&dword_4F063F8;
    v16 = 0;
    v14 = (__int64)v210;
    v12 = sub_7BF130(16386 - ((unsigned int)(dword_4D04808 == 0) - 1), 11, &v198);
    v27 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
    v28 = _mm_loadu_si128(&xmmword_4D04A20);
    v29 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    v205 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    v206 = v27;
    v204 = qword_4F063F0;
    v207 = v28;
    v208 = v29;
    v175 = sub_7BE840(v210, 0);
    v30 = v175 == 27;
    LOBYTE(v31) = v175 == 72;
    v196 = v175 == 27;
    a7 = v175 == 72;
    if ( unk_4D047C8 )
    {
      if ( unk_4F04C48 != -1
        && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0
        && dword_4F04C44 == -1 )
      {
        v52 = 776LL * dword_4F04C64 + qword_4F04C68[0];
        if ( (*(_BYTE *)(v52 + 6) & 6) == 0 && *(_BYTE *)(v52 + 4) != 12 )
        {
          if ( unk_4D047D8 )
          {
            v17 = v206.m128i_i8[0];
            if ( (v206.m128i_i8[0] & 1) == 0 )
            {
              if ( v175 != 27 && v175 != 72 )
                goto LABEL_9;
              v14 = LODWORD(v210[0]);
              if ( v175 == 72 )
              {
                v172 = (unsigned int)v210[0];
                v155 = sub_8269E0(LODWORD(v210[0]), 0, v31);
                v14 = v172;
                v30 = 0;
                if ( v155 )
                  v14 = v155;
              }
              v16 = 0;
              v167 = v30;
              v53 = sub_878D80(v14, 0, v31);
              v30 = v167;
              if ( v53 )
              {
                v54 = *(_QWORD *)(v53 + 32);
                if ( v54 && (v206.m128i_i8[1] & 0x40) == 0 )
                {
                  v55 = *(_BYTE *)(v54 + 80) == 16;
                  v206.m128i_i64[1] = v54;
                  v12 = v54;
                  if ( v55 )
                    v12 = **(_QWORD **)(v54 + 88);
                  if ( *(_BYTE *)(v12 + 80) == 24 )
                    v12 = *(_QWORD *)(v12 + 88);
                }
              }
              else if ( dword_4F077BC && qword_4F077A8 > 0x76BFu )
              {
                if ( qword_4F077A8 > 0x9CA3u )
                {
                  if ( v12 )
                    goto LABEL_32;
                  if ( qword_4F077A8 > 0x9EFBu )
                    goto LABEL_33;
                }
                if ( (v206.m128i_i8[1] & 0x40) == 0 )
                {
                  v206.m128i_i8[0] &= ~0x80u;
                  v206.m128i_i64[1] = 0;
                }
                v14 = (__int64)&v205;
                v16 = 6291456;
                v153 = sub_7D5DD0(&v205, 6291456);
                v30 = v167;
                if ( v153 && *(_BYTE *)(v153 + 80) != 19 )
                  v12 = v153;
              }
            }
          }
        }
      }
    }
    if ( !v12 )
      goto LABEL_33;
LABEL_32:
    if ( !v30 )
      goto LABEL_33;
    v14 = v12;
    v39 = sub_877F80(v12);
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 <= 0x9E33u )
          goto LABEL_63;
LABEL_76:
        if ( (unk_4D04A12 & 2) != 0 && *(_BYTE *)(v12 + 80) == 3 && *(_BYTE *)(v12 + 104) )
        {
          v48 = *(_QWORD *)(v12 + 88);
          for ( i = v48; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 8LL) )
          {
            for ( j = xmmword_4D04A20.m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            if ( v48 == j
              || (v16 = (__int64)&dword_4F07588, dword_4F07588)
              && (v51 = *(_QWORD *)(j + 32), *(_QWORD *)(v48 + 32) == v51)
              && v51 )
            {
LABEL_64:
              if ( (unk_4D04A10 & 1) != 0 )
              {
                v16 = (__int64)&v203;
                v14 = 2966;
                v12 = 0;
                sub_6851C0(0xB96u, &v203);
                v40 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
                v41 = _mm_loadu_si128(&xmmword_4F06660[1]);
                v42 = _mm_loadu_si128(&xmmword_4F06660[2]);
                v43 = _mm_loadu_si128(&xmmword_4F06660[3]);
                v206 = v41;
                v205.m128i_i64[0] = v40;
                v207 = v42;
                v206.m128i_i8[1] = v41.m128i_i8[1] | 0x20;
                v205.m128i_i64[1] = *(_QWORD *)dword_4F07508;
                v17 = v41.m128i_i8[0];
                v208 = v43;
                goto LABEL_9;
              }
LABEL_33:
              if ( v206.m128i_i8[0] >= 0 )
                goto LABEL_10;
LABEL_34:
              v16 = 406;
              v14 = unk_4F07470;
              if ( (unsigned int)sub_6E53E0(unk_4F07470, 406, dword_4F07508) )
              {
                v14 = unk_4F07470;
                v16 = 406;
                sub_685440(unk_4F07470, 0x196u, v206.m128i_i64[1]);
              }
              goto LABEL_10;
            }
          }
        }
LABEL_63:
        if ( v39 != 1 )
          goto LABEL_33;
        goto LABEL_64;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_63;
    }
    if ( qword_4F077A0 <= 0xC34Fu )
      goto LABEL_63;
    goto LABEL_76;
  }
  v12 = a5;
  v203 = *(_QWORD *)(a6 + 68);
  v204 = *(_QWORD *)(a6 + 76);
  v196 = *(_BYTE *)(a6 + 19) >> 7;
  sub_878710(a5, &v205);
  v13 = *(_BYTE *)(v12 + 80);
  if ( v13 != 16 )
  {
    if ( v13 != 24 )
      goto LABEL_4;
LABEL_56:
    v14 = *(unsigned int *)(a6 + 112);
    v12 = *(_QWORD *)(v12 + 88);
    if ( !(_DWORD)v14 )
      goto LABEL_5;
LABEL_26:
    v205.m128i_i64[1] = *(_QWORD *)(a6 + 112);
    goto LABEL_6;
  }
  v12 = **(_QWORD **)(v12 + 88);
  if ( *(_BYTE *)(v12 + 80) == 24 )
    goto LABEL_56;
LABEL_4:
  v14 = *(unsigned int *)(a6 + 112);
  if ( (_DWORD)v14 )
    goto LABEL_26;
LABEL_5:
  v205.m128i_i64[1] = v203;
LABEL_6:
  v15 = *(_BYTE *)(a6 + 19);
  v16 = a7;
  v206.m128i_i16[0] = v206.m128i_i16[0] & 0xFEFE | ((*(_BYTE *)(a6 + 18) & 0x40) != 0) | ((v15 & 1) << 8);
  if ( a7 )
  {
    v206.m128i_i8[2] |= 1u;
    a7 = 0;
    v175 = 0;
    v207.m128i_i64[1] = a8;
  }
  else
  {
    v175 = 0;
    v206.m128i_i8[2] = v206.m128i_i8[2] & 0xFE | ((v15 & 8) != 0);
    v207.m128i_i64[1] = *(_QWORD *)(a6 + 104);
  }
  v17 = v206.m128i_i8[0];
LABEL_9:
  if ( v17 < 0 )
    goto LABEL_34;
LABEL_10:
  if ( !v12 )
  {
    if ( (v206.m128i_i8[1] & 0x20) == 0 )
    {
      v44 = sub_87EBB0(13, v205.m128i_i64[0]);
      v45 = sub_6E50B0(v44, &v203);
      sub_6E72F0(v44, v45, &v205, a1);
      goto LABEL_38;
    }
    goto LABEL_37;
  }
  v18 = *(_BYTE *)(v12 + 80);
  if ( v18 == 17 )
  {
LABEL_53:
    v202 = 0;
    v166 = 0;
    goto LABEL_14;
  }
  if ( (*(_BYTE *)(v12 + 84) & 4) != 0 )
    goto LABEL_13;
  v38 = v12;
  if ( v18 == 16 )
  {
    v38 = **(_QWORD **)(v12 + 88);
    v18 = *(_BYTE *)(v38 + 80);
  }
  if ( v18 == 24 )
    v18 = *(_BYTE *)(*(_QWORD *)(v38 + 88) + 80LL);
  if ( (unsigned __int8)(v18 - 10) <= 1u && (unsigned int)sub_82EE30(v14, v16, v38) )
    goto LABEL_13;
  if ( !(a7 | v196) || !unk_4D047D8 || (v206.m128i_i8[0] & 1) != 0 )
    goto LABEL_52;
  v69 = *(_BYTE *)(v12 + 80);
  if ( v69 == 11 )
  {
    if ( !unk_4D041F8 )
      goto LABEL_13;
    v106 = *(_QWORD *)(v12 + 88);
    if ( *(_BYTE *)(v106 + 174) || !*(_WORD *)(v106 + 176) )
      goto LABEL_13;
    goto LABEL_52;
  }
  if ( v69 != 13 && v69 != 20 )
  {
LABEL_52:
    if ( !a4 )
    {
      v166 = 0;
      v202 = sub_6E50B0(v12, &v205.m128i_u64[1]);
      goto LABEL_14;
    }
    goto LABEL_53;
  }
LABEL_13:
  v202 = 0;
  v166 = 1;
LABEL_14:
  v19 = (__int64)&v205;
  sub_6E62A0(&v205);
  if ( (v206.m128i_i8[1] & 0x20) != 0 )
  {
LABEL_55:
    sub_6E6260(a1);
    sub_6E5930(v202);
    v202 = 0;
    goto LABEL_38;
  }
  v200 = 0;
  v201 = 0;
  v24 = (unsigned int *)dword_4D04348;
  if ( dword_4D04348 && (*(_BYTE *)(v12 + 83) & 0x10) != 0 && (v206.m128i_i8[0] & 1) == 0 )
  {
    v24 = &v205.m128i_u32[2];
    v19 = v12;
    sub_861C20(v12, &v205.m128i_u64[1]);
  }
  if ( a4 )
  {
    if ( v207.m128i_i64[1] )
    {
      v24 = (unsigned int *)a4;
      v19 = (__int64)&v205;
      sub_691040((__int64)&v205, a4, v166);
      if ( *(_BYTE *)(a4 + 56) )
      {
LABEL_71:
        sub_6E6260(a1);
        goto LABEL_42;
      }
    }
  }
  v25 = v206.m128i_i64[1];
  if ( dword_4F04C64 != -1 )
  {
    v20 = qword_4F04C68;
    v26 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    if ( (v26[7] & 1) != 0 )
    {
      v22 = &dword_4F04C44;
      if ( dword_4F04C44 != -1 || (v26[6] & 6) != 0 || v26[4] == 12 )
      {
        v24 = &v205.m128i_u32[2];
        v19 = v12;
        v159 = v206.m128i_i64[1];
        sub_867130(v12, &v205.m128i_u64[1], 0, 0);
        v25 = v159;
      }
    }
  }
  switch ( *(_BYTE *)(v12 + 80) )
  {
    case 2:
      if ( (a2 & 4) != 0 )
        sub_6E83C0(*(_QWORD *)(v12 + 88), 1);
      sub_6F35D0(v12, a1);
      v32 = (__int64)&v205;
      sub_6E46C0(a1, &v205);
      if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 && !(unsigned int)sub_8D3D40(*a1) )
        sub_6E9350(a1);
      goto LABEL_127;
    case 3:
    case 4:
    case 5:
    case 6:
      goto LABEL_92;
    case 7:
      v56 = *(_QWORD *)(v12 + 88);
      if ( (*(_BYTE *)(v56 + 176) & 0x40) != 0 && (v63 = *(_QWORD *)(v56 + 120), v63 == *(_QWORD *)&dword_4D03B80) )
      {
        if ( dword_4F04C44 != -1 || (v70 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v70 + 6) & 6) != 0) )
        {
          if ( (*(_DWORD *)(v56 + 168) & 0xD01000) != 0x100000 )
          {
            v32 = (__int64)&v205.m128i_i64[1];
            goto LABEL_117;
          }
        }
        else
        {
          if ( *(_BYTE *)(v70 + 4) != 12 )
            goto LABEL_37;
          v32 = (__int64)&v205.m128i_i64[1];
          if ( (*(_DWORD *)(v56 + 168) & 0xD01000) != 0x100000 )
            goto LABEL_199;
        }
      }
      else
      {
        if ( (*(_DWORD *)(v56 + 168) & 0xD01000) != 0x100000 )
        {
LABEL_115:
          v32 = (__int64)&v205.m128i_i64[1];
          goto LABEL_116;
        }
        v63 = *(_QWORD *)(v56 + 120);
      }
      v180 = *(_QWORD *)(v12 + 88);
      v64 = sub_8D2600(v63);
      v56 = v180;
      v32 = (__int64)&v205.m128i_i64[1];
      if ( v64 )
      {
        sub_686890(0xD5Bu, &v205.m128i_i32[2], v12, *(_QWORD *)(v180 + 120));
        sub_6E6260(a1);
        v56 = v180;
        v32 = (__int64)&v205.m128i_i64[1];
      }
LABEL_116:
      if ( dword_4F04C44 != -1 )
        goto LABEL_117;
      v70 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v70 + 6) & 6) != 0 )
        goto LABEL_117;
LABEL_199:
      if ( *(_BYTE *)(v70 + 4) != 12 && (*(_BYTE *)(v56 + 170) & 0x10) != 0 )
      {
        v181 = v56;
        v71 = sub_8D23E0(*(_QWORD *)(v56 + 120));
        v56 = v181;
        if ( v71 )
        {
          sub_8AC4A0(v181, &v205.m128i_u64[1]);
          v56 = v181;
        }
      }
LABEL_117:
      v179 = v56;
      if ( (unsigned int)sub_69D830(v12, &v205.m128i_i32[2], (__int64)a1, &v202, v210, &v200, &v201) )
        goto LABEL_122;
      if ( v210[0] )
      {
        v32 = (__int64)&v203;
        sub_6F8E70(v179, &v203, &v204, a1, v202);
        if ( (*((_BYTE *)v210[0] + 33) & 4) != 0 && !(unsigned int)sub_8D2310(*a1) )
        {
          v103 = sub_73C570(*a1, 1, -1);
          v104 = (_QWORD *)a1[18];
          v32 = 1;
          *a1 = v103;
          *v104 = sub_73C570(*v104, 1, -1);
        }
        *((_BYTE *)a1 + 20) |= 0x10u;
        if ( (unsigned int)sub_8DD010(*a1) )
          sub_6F9060(a1);
        goto LABEL_122;
      }
      if ( dword_4F077C4 == 2 )
      {
        if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001
          && (v133 = *(_QWORD *)(v179 + 120), (*(_BYTE *)(v133 + 140) & 0xFB) == 8) )
        {
          v134 = sub_8D4C10(v133, 0);
          v135 = v179;
          if ( (v134 & 1) != 0 )
          {
            v154 = sub_6EA7C0(v179, 0);
            v135 = v179;
            if ( !v154 )
            {
              v32 = (__int64)a1;
              sub_6E6890(259, a1);
LABEL_122:
              if ( !*((_BYTE *)a1 + 16) )
                goto LABEL_126;
              v57 = *a1;
              for ( k = *(_BYTE *)(*a1 + 140); k == 12; k = *(_BYTE *)(v57 + 140) )
                v57 = *(_QWORD *)(v57 + 160);
              if ( k )
              {
                v32 = (__int64)&v205;
                sub_6E46C0(a1, &v205);
              }
              else
              {
LABEL_126:
                sub_6E5930(v202);
                v202 = 0;
              }
LABEL_127:
              v33 = *a1;
              *((_WORD *)a1 + 9) = *((_WORD *)a1 + 9) & 0xFEF7 | ((v206.m128i_i8[1] & 1) << 8) | 8;
              if ( (unsigned int)sub_8D2780(v33) )
                goto LABEL_40;
              v33 = *a1;
              if ( (unsigned int)sub_8D3D40(*a1) || !*((_BYTE *)a1 + 16) )
                goto LABEL_40;
              v59 = *a1;
              for ( m = *(unsigned __int8 *)(*a1 + 140); (_BYTE)m == 12; m = *(unsigned __int8 *)(v59 + 140) )
                v59 = *(_QWORD *)(v59 + 160);
              if ( !(_BYTE)m )
                goto LABEL_40;
              goto LABEL_39;
            }
          }
          if ( !sub_68B790(v135) )
          {
            if ( dword_4F077C4 == 2 )
              goto LABEL_207;
            goto LABEL_205;
          }
        }
        else if ( !sub_68B790(v179) )
        {
          goto LABEL_207;
        }
      }
      else if ( !sub_68B790(v179) )
      {
LABEL_205:
        if ( unk_4F07778 > 199900 )
        {
          v190 = v72;
          sub_68B6B0(v12);
          v72 = v190;
        }
LABEL_207:
        if ( (*(_BYTE *)(v72 + 170) & 0x30) == 0x10
          && *(_BYTE *)(v12 + 80) == 9
          && dword_4F077BC
          && (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0 )
        {
          v187 = v72;
          sub_5EB3F0((_QWORD *)v72);
          v72 = v187;
        }
        if ( !v201 )
          goto LABEL_209;
        v188 = v72;
        v112 = sub_8D32E0(*(_QWORD *)(v72 + 120));
        v72 = v188;
        if ( v112 )
        {
          if ( dword_4F077BC && !(_DWORD)qword_4F077B4 )
            goto LABEL_301;
          v113 = sub_8D46C0(*(_QWORD *)(v188 + 120));
          v72 = v188;
          v114 = v113;
          if ( (*(_BYTE *)(v113 + 140) & 0xFB) == 8 )
          {
            v149 = sub_8D4C10(v113, dword_4F077C4 != 2);
            v72 = v188;
            if ( (v149 & 1) != 0 )
              goto LABEL_301;
          }
          v188 = v72;
          v115 = sub_8D2310(v114);
          v72 = v188;
          if ( v115 )
          {
LABEL_301:
            v32 = (__int64)&v203;
            sub_6F8E70(v72, &v203, &v204, a1, v202);
            v73 = v188;
LABEL_210:
            if ( v200 )
            {
              v74 = sub_6EA7C0(v73, &v203);
              v75 = v74;
              if ( v74 && (unsigned int)sub_8D2FB0(*(_QWORD *)(v74 + 128)) )
              {
                v147 = sub_73A720(v75);
                v148 = sub_73DDB0(v147);
                v32 = (__int64)a1;
                sub_6E7150(v148, a1);
              }
              else
              {
                if ( (unsigned int)sub_8D3410(*a1) )
                  sub_6FB570(a1);
                else
                  sub_6FA3A0(a1);
                v32 = 0;
                sub_6E6B60(a1, 0);
              }
            }
            goto LABEL_122;
          }
          v130 = *(_QWORD *)(v188 + 120);
          v156 = sub_73C570(v114, 1, -1);
          v157 = sub_72D750(v156, v130);
          v132 = v188;
          *(_QWORD *)(v188 + 120) = v157;
        }
        else
        {
          v129 = *(_QWORD *)(v188 + 120);
          if ( (*(_BYTE *)(v129 + 140) & 0xFB) == 8 )
          {
            v146 = sub_8D4C10(v129, dword_4F077C4 != 2);
            v72 = v188;
            if ( (v146 & 1) != 0 )
            {
LABEL_209:
              v32 = (__int64)&v203;
              v182 = v72;
              sub_6F8E70(v72, &v203, &v204, a1, v202);
              v73 = v182;
              goto LABEL_210;
            }
            v129 = *(_QWORD *)(v188 + 120);
          }
          v191 = v72;
          v130 = v129;
          v131 = sub_73C570(v129, 1, -1);
          v132 = v191;
          *(_QWORD *)(v191 + 120) = v131;
        }
        v32 = (__int64)&v203;
        v192 = v132;
        sub_6F8E70(v132, &v203, &v204, a1, v202);
        v73 = v192;
        if ( v130 )
          *(_QWORD *)(v192 + 120) = v130;
        goto LABEL_210;
      }
      v32 = (__int64)a1;
      sub_6E6890(1586, a1);
      goto LABEL_122;
    case 8:
      v65 = *(_QWORD *)(v12 + 88);
      v66 = 0;
      if ( *(char *)(v65 + 144) >= 0 )
        goto LABEL_166;
      v177 = v25;
      v94 = sub_5F77E0((__int64 *)v65, &v205.m128i_i64[1]);
      if ( !v94 )
        goto LABEL_55;
      v95 = sub_85E8D0();
      v96 = sub_830A00(v95);
      *(_QWORD *)(v96 + 28) = *(_QWORD *)&dword_4F063F8;
      sub_6E70E0(v96, v209);
      v25 = v177;
      v66 = 1;
      v12 = **(_QWORD **)(v94 + 24);
      v206.m128i_i64[1] = v12;
LABEL_166:
      v176 = a2 & 0x8000;
      if ( a2 < 0 )
      {
        sub_6E7350(v12, v206.m128i_i8[0] & 1, v202, a1, v66);
        goto LABEL_38;
      }
      v67 = qword_4D03C50;
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
        goto LABEL_168;
      v160 = v66;
      v168 = v25;
      v158 = *(_QWORD *)(v12 + 88);
      v99 = sub_693A90();
      v25 = v168;
      v66 = v160;
      if ( !v99 )
        goto LABEL_257;
      v193 = v160;
      v162 = v168;
      v170 = *(_QWORD *)(v158 + 120);
      v126 = sub_8D3A70(v170);
      v25 = v162;
      LODWORD(v66) = v193;
      if ( v126
        || (v127 = sub_8D2FB0(v170), v25 = v162, LODWORD(v66) = v193, v127)
        && (v136 = sub_8D46C0(v170), v137 = sub_8D3A70(v136), v25 = v162, LODWORD(v66) = v193, v137) )
      {
        v195 = v66;
        v164 = v25;
        v138 = sub_7BE840(0, 0);
        v25 = v164;
        v66 = v195;
        if ( v138 == 29 )
          goto LABEL_337;
      }
      v194 = v66;
      v163 = v25;
      v128 = sub_8D2E30(v170);
      v25 = v163;
      v66 = v194;
      if ( !v128 )
        goto LABEL_319;
      v150 = sub_8D46C0(v170);
      v151 = sub_8D3A70(v150);
      v25 = v163;
      v66 = v194;
      if ( !v151 )
        goto LABEL_319;
      v152 = sub_7BE840(0, 0);
      v25 = v163;
      v66 = v194;
      if ( v152 != 30 )
        goto LABEL_257;
LABEL_337:
      if ( dword_4F077C4 != 2 )
        goto LABEL_257;
      v165 = v66;
      v171 = v25;
      v141 = sub_830940(0, v210, v139, v140);
      v25 = v171;
      v66 = v165;
      if ( v141 )
      {
        v142 = sub_8D46C0(v210[0]);
        v25 = v171;
        v66 = v165;
        for ( n = v142; ; n = *(_QWORD *)(n + 160) )
        {
          v144 = *(_BYTE *)(n + 140);
          if ( v144 != 12 )
            break;
        }
        if ( (unsigned __int8)(v144 - 9) <= 2u && (v145 = sub_8D5DF0(n), v25 = v171, v66 = v165, (v176 = v145) != 0) )
        {
          sub_6E25A0(&v197, &v199);
          v67 = qword_4D03C50;
          v176 = 1;
          v25 = v171;
          v66 = v165;
        }
        else
        {
LABEL_257:
          v67 = qword_4D03C50;
        }
      }
      else
      {
LABEL_319:
        v176 = 0;
        v67 = qword_4D03C50;
      }
LABEL_168:
      if ( (*(_DWORD *)(v67 + 16) & 0x400000FF) == 0x40000001 )
        goto LABEL_249;
      if ( !*(_QWORD *)(v12 + 96) )
        goto LABEL_173;
      v68 = v12;
      while ( *(_BYTE *)(v68 + 80) == 8 )
      {
        v68 = *(_QWORD *)(v68 + 96);
        if ( !v68 )
        {
LABEL_173:
          if ( (a2 & 0x40) != 0 && (v206.m128i_i8[0] & 1) != 0 )
          {
            if ( a4
              || (v169 = v25, v161 = v66, v100 = sub_7BE840(0, 0), v101 = sub_688800(v100, a3, a2), v25 = v169, v101) )
            {
              sub_6E7350(v25, 1, v202, a1, v66);
              sub_6E4620(a1);
              goto LABEL_177;
            }
            v67 = qword_4D03C50;
            LODWORD(v66) = v161;
          }
          if ( (*(_BYTE *)(v67 + 17) & 4) == 0
            || (v174 = v66, v189 = v25, v118 = sub_830950(v12, 0), v25 = v189, LODWORD(v66) = v174, v118) )
          {
            if ( (_DWORD)v66 || (unsigned int)sub_830D50(v12, v25, &v205.m128i_u64[1], v206.m128i_i8[1] & 1, v209) )
            {
              v102 = sub_8D46C0(v209[0]);
              sub_68D540(v209, v102, 1, 1u, (__int64)&v205, (__int64)&v203, (__int64)&v204, v202, (const __m128i *)a1);
            }
            else
            {
              sub_6E6260(a1);
            }
            goto LABEL_177;
          }
          v119 = sub_72D2E0(*(_QWORD *)(v12 + 64), 0);
          sub_6E7080(v209, 0);
          sub_6FC3F0(v119, v209, 1);
          v120 = qword_4D03C50;
          v121 = v205.m128i_i64[1];
          *(_BYTE *)(qword_4D03C50 + 17LL) |= 8u;
          *(_QWORD *)(v120 + 88) = v121;
          if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
            goto LABEL_313;
          if ( HIDWORD(qword_4F077B4) )
          {
            if ( qword_4F077A8 > 0x9DCFu )
            {
LABEL_313:
              v122 = sub_8D46C0(v209[0]);
              sub_68D540(v209, v122, 1, 1u, (__int64)&v205, (__int64)&v203, (__int64)&v204, v202, (const __m128i *)a1);
              v123 = *((_BYTE *)a1 + 16);
              if ( v123 == 1 )
              {
                v124 = a1[18];
              }
              else
              {
                v124 = 0;
                if ( v123 == 2 )
                {
                  v124 = a1[36];
                  if ( !v124 && *((_BYTE *)a1 + 317) == 12 && *((_BYTE *)a1 + 320) == 1 )
                    v124 = sub_72E9A0(a1 + 18);
                }
              }
              v125 = sub_6E36E0(v124, 1);
              *(_BYTE *)(v125 + 26) |= 2u;
              goto LABEL_177;
            }
            if ( !dword_4D04964 )
            {
              sub_6E5C80(7, 245, &v205.m128i_u64[1]);
              goto LABEL_313;
            }
          }
          else if ( !dword_4D04964 )
          {
            goto LABEL_313;
          }
          sub_6E5C80(unk_4F07471, 245, &v205.m128i_u64[1]);
          goto LABEL_313;
        }
      }
      if ( (*(_BYTE *)(v67 + 19) & 0x40) != 0 )
      {
LABEL_249:
        sub_6E6890(28, a1);
        sub_6E5930(v202);
        v202 = 0;
      }
      else if ( !(unsigned int)sub_69D830(v68, &v205.m128i_i32[2], (__int64)a1, &v202, 0, 0, 0) )
      {
        v98 = v202;
        v184 = *(_QWORD *)(v68 + 88);
        sub_6F8E70(v184, &dword_4F077C8, &dword_4F077C8, v210, 0);
        sub_68D540(
          v210,
          *(_QWORD *)(v184 + 120),
          0,
          1u,
          (__int64)&v205,
          (__int64)&v203,
          (__int64)&v204,
          v98,
          (const __m128i *)a1);
        *(__int64 *)((char *)a1 + 68) = v203;
      }
LABEL_177:
      if ( v176 )
        sub_6E25E0(v197, v199);
      goto LABEL_38;
    case 9:
      v56 = *(_QWORD *)(v12 + 88);
      if ( (*(_BYTE *)(v56 + 176) & 1) != 0 && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
        *(_BYTE *)(qword_4D03C50 + 21LL) |= 2u;
      goto LABEL_115;
    case 0xA:
      if ( v166 )
        goto LABEL_72;
      if ( (*(_BYTE *)(v12 + 104) & 1) != 0 )
      {
        v19 = v12;
        v186 = v25;
        v60 = sub_8796F0(v12);
        v25 = v186;
      }
      else
      {
        v60 = (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 208LL) & 4) != 0;
      }
      if ( v60 )
      {
        if ( (unsigned int)sub_6E5430(v19, v24, v20, v21, v22, v23) )
          sub_6854C0(0xC40u, (FILE *)&v205.m128i_u64[1], v12);
        goto LABEL_37;
      }
      for ( ii = *(_QWORD *)(*(_QWORD *)(v12 + 88) + 152LL); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(ii + 168) + 40LL) )
      {
        if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 )
        {
          sub_6E6890(28, a1);
          sub_6E5930(v202);
          v202 = 0;
        }
        else
        {
          sub_6E7350(v25, v206.m128i_i8[0] & 1, v202, a1, v22);
          sub_6E4620(a1);
        }
        goto LABEL_38;
      }
LABEL_135:
      if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 )
      {
        if ( !unk_4D041F8 || (v183 = v25, v97 = sub_7176C0(*(_QWORD *)(v12 + 88), 0), LODWORD(v25) = v183, !v97) )
        {
          sub_6E6890(157, a1);
          sub_6E5930(v202);
          v202 = 0;
          goto LABEL_38;
        }
      }
      if ( unk_4D04374 )
      {
        v185 = v25;
        sub_68B6B0(v12);
        LODWORD(v25) = v185;
      }
      sub_6EAB60(v25, v206.m128i_i8[0] & 1, 0, (unsigned int)&v205.m128i_u32[2], (unsigned int)&v204, v202, (__int64)a1);
      sub_6E4620(a1);
      *((_WORD *)a1 + 9) = *((_WORD *)a1 + 9) & 0xFEF7 | ((v206.m128i_i8[1] & 1) << 8) | 8;
      goto LABEL_39;
    case 0xB:
      if ( !v166 )
        goto LABEL_135;
      goto LABEL_72;
    case 0xD:
      if ( !a4 )
        goto LABEL_37;
      if ( *(char *)(a6 + 19) >= 0 )
      {
        *(_BYTE *)(a4 + 56) = 1;
        goto LABEL_37;
      }
      sub_6E72F0(v12, 0, &v205, a1);
      goto LABEL_38;
    case 0x11:
    case 0x14:
LABEL_72:
      if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) == 0x40000001 )
        sub_6E6890(28, a1);
      else
        sub_6E7190(v25, &v205, a1);
      goto LABEL_38;
    case 0x12:
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) != 0 )
        goto LABEL_188;
      if ( !(unsigned int)sub_693580() )
        goto LABEL_219;
      v107 = sub_85E8D0();
      if ( v107 == -1 )
        goto LABEL_381;
      v108 = *(_QWORD *)(qword_4F04C68[0] + 776LL * v107 + 216);
      for ( jj = v108 >> 3; ; LODWORD(jj) = v110 + 1 )
      {
        v110 = qword_4F04C10[1] & jj;
        v111 = (_QWORD *)(*qword_4F04C10 + 16LL * v110);
        if ( v108 == *v111 )
          break;
        if ( !*v111 )
LABEL_381:
          BUG();
      }
      if ( (*(_BYTE *)(*(_QWORD *)(**(_QWORD **)(v111[1] + 8LL) + 96LL) + 181LL) & 1) != 0 )
        goto LABEL_188;
LABEL_219:
      if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x20200) == 0x200 )
      {
LABEL_188:
        sub_6E6890(411, a1);
        sub_6E5930(v202);
        v202 = 0;
      }
      else if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 2) != 0
             && *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 1 )
      {
        v116 = *(_QWORD *)(v12 + 88);
        v117 = *(__int64 **)(v116 + 56);
        if ( !v117 )
        {
          v117 = (__int64 *)sub_735FB0(*(_QWORD *)(v116 + 16), 3, 0xFFFFFFFFLL, v76);
          *v117 = v12;
          *(_QWORD *)(*(_QWORD *)(v12 + 88) + 56LL) = v117;
        }
        sub_6F8E70(v117, &v203, &v204, a1, v202);
        if ( !a4 )
          sub_87E130(0, a1[18], v12, (char *)a1 + 68);
      }
      else
      {
        sub_688B20((__int64)a1, v12);
      }
      goto LABEL_38;
    case 0x13:
      if ( (v206.m128i_i8[2] & 1) != 0 )
      {
        if ( dword_4F04C44 != -1
          || (v62 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v62 + 6) & 6) != 0)
          || *(_BYTE *)(v62 + 4) == 12 )
        {
          sub_6F3AD0(v25, 1, v207.m128i_i64[1], v206.m128i_i8[0] & 1, a1);
          if ( (a2 & 4) != 0 && *((_BYTE *)a1 + 16) == 2 )
            sub_6E83C0(a1 + 18, 1);
          sub_6F7FE0(a1, 0);
          goto LABEL_38;
        }
LABEL_94:
        sub_6E6890(254, a1);
        goto LABEL_38;
      }
      v85 = sub_7BE840(0, 0);
      if ( v85 != 27 && v85 != 73 )
        goto LABEL_231;
      v86 = &dword_4D04860;
      v90 = *(_BYTE *)(v12 + 80);
      if ( dword_4D04860 )
      {
        if ( v90 != 19 )
          goto LABEL_231;
      }
      else if ( v90 != 19 || (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 265LL) & 1) != 0 )
      {
LABEL_231:
        if ( (unsigned int)sub_6E5430(0, 0, v86, v87, v88, v89) )
          sub_6854C0(0x1B9u, (FILE *)&v205.m128i_u64[1], v12);
LABEL_37:
        sub_6E6260(a1);
LABEL_38:
        *((_WORD *)a1 + 9) = *((_WORD *)a1 + 9) & 0xFEF7 | ((v206.m128i_i8[1] & 1) << 8) | 8;
LABEL_39:
        v32 = (__int64)a1;
        v33 = 1;
        sub_6E26D0(1, a1);
LABEL_40:
        *((_BYTE *)a1 + 19) = ((a7 | (unsigned __int8)v196) << 7) | *((_BYTE *)a1 + 19) & 0x7F;
        if ( !a4 )
        {
          *(_QWORD *)&dword_4F061D8 = v204;
          sub_7B8B50(v33, v32, m, v35);
        }
        goto LABEL_42;
      }
      v19 = v12;
      v24 = &v205.m128i_u32[2];
      v12 = *(_QWORD *)sub_72EF10(v12, &v205.m128i_u64[1]);
LABEL_92:
      if ( dword_4F077C4 != 2 || a4 || !v196 && (!dword_4D04428 || v175 != 73) )
        goto LABEL_94;
      v105 = *(_QWORD *)(v12 + 88);
      if ( (a2 & 0x2000) != 0 )
      {
        v24 = &dword_4F063F8;
        v19 = 3038;
        sub_6851C0(0xBDEu, &dword_4F063F8);
        a2 &= ~0x2000u;
      }
      sub_7B8B50(v19, v24, v20, v21);
      sub_6CA0E0(0, 0, 0, 0, v105, (char *)&v203, a1, a2);
LABEL_42:
      v36 = v203;
      *(__int64 *)((char *)a1 + 68) = v203;
      *(_QWORD *)dword_4F07508 = v36;
      *(__int64 *)((char *)a1 + 76) = *(_QWORD *)&dword_4F061D8;
      result = (_DWORD *)sub_6E3280(a1, 0);
      if ( a9 )
      {
        result = a9;
        *a9 = 0;
      }
      return result;
    case 0x15:
      if ( (unsigned int)sub_6E5430(v19, v24, v20, v21, v22, v23) )
        sub_6854C0(0x1B9u, (FILE *)dword_4F07508, v12);
      goto LABEL_37;
    case 0x16:
      sub_7B8B50(v19, v24, v20, v21);
      if ( !(unsigned int)sub_7BE5B0(43, 438, 0, 0)
        || (sub_7B8B50(43, 438, v46, v47),
            ++*(_BYTE *)(qword_4F061C8 + 52LL),
            v77 = sub_7C7EE0(v12, 0, &v198),
            --*(_BYTE *)(qword_4F061C8 + 52LL),
            !(unsigned int)sub_7BE5B0(44, 439, 0, 0)) )
      {
        v198 = 1;
        goto LABEL_71;
      }
      if ( v198 )
        goto LABEL_71;
      v78 = *(_QWORD *)(v12 + 88);
      v79 = (_QWORD *)sub_726700(32);
      *v79 = sub_72C390();
      *(_QWORD *)((char *)v79 + 28) = v203;
      v84 = *(_QWORD *)(v78 + 104);
      v79[8] = v77;
      v79[7] = v84;
      if ( (a2 & 0x1000) != 0 )
      {
        sub_6E70E0(v79, a1);
        goto LABEL_38;
      }
      v210[0] = (__int64 *)sub_724DC0(32, 439, v80, v81, v82, v83);
      if ( (dword_4F04C44 != -1
         || (v91 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v91 + 6) & 2) != 0)
         || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 6) & 0x10) != 0
         || (*(_BYTE *)(v91 + 6) & 4) != 0 && (*(_BYTE *)(v91 + 12) & 0x10) == 0)
        && (unsigned int)sub_731EE0(v79) )
      {
        if ( !unk_4F074B0 || !(unsigned int)sub_893F30(v77) )
        {
          sub_70FD90(v79, v210[0]);
          v93 = v210[0];
          goto LABEL_241;
        }
      }
      else
      {
        LODWORD(v209[0]) = 0;
        v92 = sub_693DC0((__int64)v79, v209);
        if ( !LODWORD(v209[0]) )
        {
          sub_72C470(v92, v210[0]);
          v93 = v210[0];
          v210[0][18] = (__int64)v79;
LABEL_241:
          sub_6E6A50(v93, a1);
          *(__int64 *)((char *)a1 + 68) = *(_QWORD *)((char *)v79 + 28);
          sub_724E30(v210);
          goto LABEL_38;
        }
      }
      sub_6E6260(a1);
      *(__int64 *)((char *)a1 + 68) = *(_QWORD *)((char *)v79 + 28);
      sub_724E30(v210);
      goto LABEL_38;
    case 0x17:
      sub_6E6890(728, a1);
      goto LABEL_38;
    default:
      sub_721090(v19);
  }
}
