// Function: sub_32B3F40
// Address: 0x32b3f40
//
__int64 __fastcall sub_32B3F40(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // r14
  int v5; // r15d
  __int16 v6; // ax
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned int v9; // ebx
  __int64 result; // rax
  unsigned int v11; // r8d
  unsigned int v12; // r10d
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // eax
  _QWORD *v18; // r11
  char v19; // al
  unsigned int v20; // r8d
  _BYTE *v21; // rax
  unsigned int v22; // r8d
  unsigned int v23; // r10d
  unsigned int v24; // r11d
  __int64 v25; // r14
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rsi
  __int16 v36; // ax
  __int64 v37; // r15
  __int64 v38; // rax
  unsigned int v39; // edx
  __int64 v40; // rdx
  __int64 v41; // rdi
  __int64 v42; // r13
  __int64 v43; // r10
  __int64 v44; // r15
  __int64 v45; // rax
  int v46; // edx
  __int64 v47; // rax
  _QWORD *v48; // r14
  int v49; // eax
  __int64 v50; // rdx
  unsigned __int64 v51; // rsi
  __int16 v52; // ax
  __int64 v53; // rdx
  unsigned __int8 v54; // al
  int v55; // eax
  __int64 *v56; // rax
  __int64 v57; // r14
  __int64 v58; // rax
  int v59; // edx
  __int64 v60; // rax
  _QWORD *v61; // r15
  unsigned __int16 v62; // ax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdx
  unsigned __int64 v66; // rax
  __int32 v67; // eax
  __int64 v68; // rdx
  int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rdx
  int v72; // eax
  __int64 v73; // rax
  __int16 v77; // ax
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // rdx
  __int64 v82; // rax
  int v83; // edx
  __int64 v84; // r15
  unsigned int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rdi
  __int64 (*v88)(); // r11
  char v89; // al
  __int64 *v90; // rax
  __int64 v91; // rdx
  int v92; // ecx
  __int64 v94; // rsi
  __int32 v97; // eax
  __int64 v98; // rdx
  unsigned __int64 v99; // r15
  unsigned __int64 v100; // rax
  __int64 v101; // rax
  _DWORD *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  int v105; // edx
  bool v107; // al
  __int64 v108; // rsi
  int v109; // eax
  unsigned __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rdx
  unsigned __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // r15
  int v116; // eax
  __int128 v117; // rax
  unsigned __int64 v118; // r15
  unsigned int v119; // edx
  __int16 v120; // dx
  __m128i v121; // rax
  __int32 v122; // r15d
  __m128i v123; // rax
  __int64 v124; // rcx
  unsigned int v125; // edx
  __int64 v126; // rdx
  __int64 v127; // r13
  __int128 v128; // rax
  int v129; // r9d
  unsigned int v130; // eax
  unsigned int v132; // eax
  __int128 v133; // [rsp-30h] [rbp-170h]
  __int128 v134; // [rsp-20h] [rbp-160h]
  unsigned int v135; // [rsp+10h] [rbp-130h]
  int v136; // [rsp+14h] [rbp-12Ch]
  int v137; // [rsp+14h] [rbp-12Ch]
  unsigned int v138; // [rsp+14h] [rbp-12Ch]
  int v139; // [rsp+14h] [rbp-12Ch]
  __int16 v140; // [rsp+18h] [rbp-128h]
  unsigned int v141; // [rsp+18h] [rbp-128h]
  unsigned int v142; // [rsp+18h] [rbp-128h]
  unsigned int v143; // [rsp+18h] [rbp-128h]
  __int64 v144; // [rsp+18h] [rbp-128h]
  int v145; // [rsp+18h] [rbp-128h]
  unsigned int v146; // [rsp+20h] [rbp-120h]
  unsigned int v147; // [rsp+20h] [rbp-120h]
  __int64 v148; // [rsp+20h] [rbp-120h]
  unsigned int v149; // [rsp+20h] [rbp-120h]
  unsigned int v150; // [rsp+20h] [rbp-120h]
  __int64 v151; // [rsp+20h] [rbp-120h]
  int v152; // [rsp+20h] [rbp-120h]
  unsigned int v153; // [rsp+20h] [rbp-120h]
  __int64 v154; // [rsp+28h] [rbp-118h]
  unsigned int v155; // [rsp+30h] [rbp-110h]
  unsigned int v156; // [rsp+30h] [rbp-110h]
  unsigned int v157; // [rsp+30h] [rbp-110h]
  unsigned __int64 v158; // [rsp+30h] [rbp-110h]
  unsigned int v159; // [rsp+30h] [rbp-110h]
  unsigned int v160; // [rsp+30h] [rbp-110h]
  unsigned int v161; // [rsp+30h] [rbp-110h]
  __int64 v162; // [rsp+30h] [rbp-110h]
  unsigned int v163; // [rsp+38h] [rbp-108h]
  unsigned int v164; // [rsp+38h] [rbp-108h]
  unsigned int v165; // [rsp+38h] [rbp-108h]
  unsigned int v166; // [rsp+38h] [rbp-108h]
  __int64 v167; // [rsp+38h] [rbp-108h]
  unsigned int v168; // [rsp+38h] [rbp-108h]
  __int64 v169; // [rsp+38h] [rbp-108h]
  int v170; // [rsp+38h] [rbp-108h]
  unsigned int v171; // [rsp+38h] [rbp-108h]
  unsigned int v172; // [rsp+38h] [rbp-108h]
  __int64 v173; // [rsp+38h] [rbp-108h]
  unsigned int v174; // [rsp+40h] [rbp-100h]
  int v175; // [rsp+40h] [rbp-100h]
  __int64 v176; // [rsp+40h] [rbp-100h]
  __int64 v178; // [rsp+48h] [rbp-F8h]
  __m128i v179; // [rsp+50h] [rbp-F0h] BYREF
  __m128i v180; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v181; // [rsp+70h] [rbp-D0h] BYREF
  int v182; // [rsp+78h] [rbp-C8h]
  __int64 v183; // [rsp+80h] [rbp-C0h]
  __int64 v184; // [rsp+88h] [rbp-B8h]
  __int64 v185; // [rsp+90h] [rbp-B0h]
  __int64 v186; // [rsp+98h] [rbp-A8h]
  __int64 v187; // [rsp+A0h] [rbp-A0h]
  __int64 v188; // [rsp+A8h] [rbp-98h]
  __int128 v189; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v190; // [rsp+C0h] [rbp-80h]
  __int128 v191; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v192; // [rsp+E0h] [rbp-60h]
  __m128i v193; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v194; // [rsp+100h] [rbp-40h]

  v2 = a2;
  v3 = *(_QWORD *)(a2 + 48);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_WORD *)v3;
  v7 = *(_QWORD *)(v3 + 8);
  v8 = *(_QWORD *)v4;
  v9 = *(_DWORD *)(v4 + 8);
  v179.m128i_i16[0] = v6;
  v179.m128i_i64[1] = v7;
  v180 = _mm_loadu_si128(&v179);
  if ( v6 )
  {
    if ( (unsigned __int16)(v6 - 17) <= 0xD3u )
      return 0;
  }
  else if ( sub_30070B0((__int64)&v179) )
  {
    return 0;
  }
  if ( v5 == 222 )
  {
    v15 = *(_QWORD *)(v4 + 40);
    v12 = 2;
    v174 = 0;
    v16 = *(_QWORD *)(v15 + 104);
    v180.m128i_i16[0] = *(_WORD *)(v15 + 96);
    v17 = *(_DWORD *)(v8 + 24);
    v180.m128i_i64[1] = v16;
    if ( v17 != 192 )
      goto LABEL_20;
    goto LABEL_11;
  }
  if ( (unsigned int)(v5 - 191) <= 1 )
  {
    v45 = *(_QWORD *)(v4 + 40);
    if ( *(_DWORD *)(v8 + 24) != 298 )
      return 0;
    v46 = *(_DWORD *)(v45 + 24);
    if ( v46 != 11 && v46 != 35 )
      return 0;
    v47 = *(_QWORD *)(v45 + 96);
    v48 = *(_QWORD **)(v47 + 24);
    if ( *(_DWORD *)(v47 + 32) > 0x40u )
      v48 = (_QWORD *)*v48;
    v49 = *(unsigned __int16 *)(v8 + 96);
    v50 = *(_QWORD *)(v8 + 104);
    LOWORD(v189) = v49;
    *((_QWORD *)&v189 + 1) = v50;
    if ( (_WORD)v49 )
    {
      if ( (unsigned __int16)(v49 - 17) > 0xD3u )
      {
        v193.m128i_i16[0] = v49;
        v193.m128i_i64[1] = v50;
        goto LABEL_44;
      }
      LOWORD(v49) = word_4456580[v49 - 1];
      v86 = 0;
    }
    else
    {
      v169 = v50;
      if ( !sub_30070B0((__int64)&v189) )
      {
        v193.m128i_i64[1] = v169;
        v193.m128i_i16[0] = 0;
        goto LABEL_99;
      }
      LOWORD(v49) = sub_3009970((__int64)&v189, a2, v169, v79, v80);
    }
    v193.m128i_i16[0] = v49;
    v193.m128i_i64[1] = v86;
    if ( (_WORD)v49 )
    {
LABEL_44:
      if ( (_WORD)v49 == 1 || (unsigned __int16)(v49 - 504) <= 7u )
        goto LABEL_188;
      v51 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v49 - 16];
      goto LABEL_47;
    }
LABEL_99:
    v185 = sub_3007260((__int64)&v193);
    v51 = v185;
    v186 = v81;
LABEL_47:
    if ( (unsigned int)v48 >= v51 )
      return 0;
    v11 = (unsigned int)v48;
    a2 = (unsigned int)(v51 - (_DWORD)v48);
    v12 = (v5 == 192) + 2;
    switch ( (_DWORD)a2 )
    {
      case 1:
        v52 = 2;
        v53 = 0;
        break;
      case 2:
        v52 = 3;
        v53 = 0;
        break;
      case 4:
        v52 = 4;
        v53 = 0;
        break;
      case 8:
        v52 = 5;
        v53 = 0;
        break;
      case 0x10:
        v52 = 6;
        v53 = 0;
        break;
      case 0x20:
        v52 = 7;
        v53 = 0;
        break;
      case 0x40:
        v52 = 8;
        v53 = 0;
        break;
      case 0x80:
        v52 = 9;
        v53 = 0;
        break;
      default:
        v52 = sub_3007020(*(_QWORD **)(*a1 + 64LL), a2);
        v11 = (unsigned int)v48;
        v12 = (v5 == 192) + 2;
        break;
    }
    v180.m128i_i16[0] = v52;
    v54 = *(_BYTE *)(v8 + 33);
    v180.m128i_i64[1] = v53;
    v55 = (v54 >> 2) & 3;
    if ( (unsigned int)(v55 - 2) <= 1 && v12 != v55 )
      return 0;
    goto LABEL_9;
  }
  v11 = 0;
  v12 = 0;
  if ( v5 != 186 )
  {
LABEL_9:
    if ( v5 == 192 )
    {
      v174 = 0;
      v8 = v2;
      v9 = 0;
      goto LABEL_11;
    }
    v174 = 0;
    goto LABEL_94;
  }
  v71 = *(_QWORD *)(v4 + 40);
  v72 = *(_DWORD *)(v71 + 24);
  if ( v72 != 35 && v72 != 11 )
    return 0;
  v73 = *(_QWORD *)(v71 + 96);
  if ( *(_DWORD *)(v73 + 32) > 0x40u )
  {
    v84 = v73 + 24;
    v175 = *(_DWORD *)(v73 + 32);
    LODWORD(_R14) = sub_C445E0(v73 + 24);
    v85 = sub_C444A0(v84);
    a2 = v85;
    if ( (_DWORD)_R14 && v175 == (_DWORD)_R14 + v85 )
    {
      v174 = 0;
    }
    else
    {
      v170 = v175;
      LODWORD(_R14) = sub_C44630(v84);
      v174 = sub_C44590(v84);
      a2 = (unsigned int)_R14 + v174 + (unsigned int)a2;
      if ( v170 != (_DWORD)a2 )
        return 0;
    }
  }
  else
  {
    _RDI = *(_QWORD *)(v73 + 24);
    if ( !_RDI )
      return 0;
    if ( (_RDI & (_RDI + 1)) != 0 )
    {
      if ( ((_RDI | (_RDI - 1)) & ((_RDI | (_RDI - 1)) + 1)) != 0 )
        return 0;
      __asm { tzcnt   rax, rdi }
      v174 = _RAX;
      LODWORD(_R14) = sub_39FAC40(_RDI);
    }
    else
    {
      if ( !~_RDI )
      {
        v174 = 0;
        goto LABEL_168;
      }
      v174 = 0;
      __asm { tzcnt   r14, rdi }
    }
  }
  switch ( (_DWORD)_R14 )
  {
    case 1:
      v77 = 2;
      goto LABEL_92;
    case 2:
      v77 = 3;
      goto LABEL_92;
    case 4:
      v77 = 4;
      goto LABEL_92;
    case 8:
      v77 = 5;
      goto LABEL_92;
  }
  v77 = 6;
  if ( (_DWORD)_R14 != 16 )
  {
    if ( (_DWORD)_R14 == 32 )
    {
      v77 = 7;
      goto LABEL_92;
    }
    if ( (_DWORD)_R14 != 64 )
    {
      if ( (_DWORD)_R14 != 128 )
      {
        a2 = (unsigned int)_R14;
        v77 = sub_3007020(*(_QWORD **)(*a1 + 64LL), _R14);
        goto LABEL_93;
      }
      v77 = 9;
      goto LABEL_92;
    }
LABEL_168:
    v77 = 8;
  }
LABEL_92:
  v78 = 0;
LABEL_93:
  v180.m128i_i16[0] = v77;
  v11 = v174;
  v12 = 3;
  v180.m128i_i64[1] = v78;
LABEL_94:
  v17 = *(_DWORD *)(v8 + 24);
  if ( v17 != 192 )
    goto LABEL_95;
LABEL_11:
  v13 = *(_QWORD *)(v8 + 56);
  if ( !v13 )
    return 0;
  v14 = 1;
  do
  {
    if ( *(_DWORD *)(v13 + 8) == v9 )
    {
      if ( !v14 )
        return 0;
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_61;
      if ( *(_DWORD *)(v13 + 8) == v9 )
        return 0;
      v14 = 0;
    }
    v13 = *(_QWORD *)(v13 + 32);
  }
  while ( v13 );
  if ( v14 == 1 )
    return 0;
LABEL_61:
  v56 = *(__int64 **)(v8 + 40);
  v57 = *v56;
  v58 = v56[5];
  if ( *(_DWORD *)(v57 + 24) != 298 )
    return 0;
  v59 = *(_DWORD *)(v58 + 24);
  if ( v59 != 11 && v59 != 35 )
    return 0;
  v60 = *(_QWORD *)(v58 + 96);
  v61 = *(_QWORD **)(v60 + 24);
  if ( *(_DWORD *)(v60 + 32) > 0x40u )
    v61 = (_QWORD *)*v61;
  v62 = *(_WORD *)(v57 + 96);
  v63 = *(_QWORD *)(v57 + 104);
  LOWORD(v191) = v62;
  *((_QWORD *)&v191 + 1) = v63;
  if ( v62 )
  {
    if ( v62 != 1 && (unsigned __int16)(v62 - 504) > 7u )
    {
      v65 = 16LL * (v62 - 1);
      v64 = *(_QWORD *)&byte_444C4A0[v65];
      LOBYTE(v65) = byte_444C4A0[v65 + 8];
      goto LABEL_68;
    }
LABEL_188:
    BUG();
  }
  v168 = v12;
  v64 = sub_3007260((__int64)&v191);
  v12 = v168;
  v187 = v64;
  v188 = v65;
LABEL_68:
  v193.m128i_i64[0] = v64;
  v141 = v12;
  v193.m128i_i8[8] = v65;
  v158 = sub_CA1930(&v193);
  if ( (unsigned int)v61 >= v158 || ((*(_BYTE *)(v57 + 33) >> 2) & 3) == 2 )
    return 0;
  v66 = sub_32844A0((unsigned __int16 *)&v180, a2);
  v11 = (unsigned int)v61;
  v12 = v141;
  if ( v66 > v158 - (unsigned int)v61 )
  {
    if ( v141 == 2 )
      return 0;
    v67 = sub_327FC40(*(_QWORD **)(*a1 + 64LL), (int)v158 - (int)v61);
    v11 = (unsigned int)v61;
    v12 = 3;
    v180.m128i_i32[0] = v67;
    v180.m128i_i64[1] = v68;
  }
  v69 = 1;
  v70 = *(_QWORD *)(v8 + 56);
  do
  {
    if ( *(_DWORD *)(v70 + 8) == v9 )
    {
      if ( !v69 )
        goto LABEL_144;
      v70 = *(_QWORD *)(v70 + 32);
      if ( !v70 )
        goto LABEL_151;
      if ( *(_DWORD *)(v70 + 8) == v9 )
        goto LABEL_144;
      v69 = 0;
    }
    v70 = *(_QWORD *)(v70 + 32);
  }
  while ( v70 );
  if ( v69 == 1 )
    goto LABEL_144;
LABEL_151:
  v103 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 16LL);
  if ( *(_DWORD *)(v103 + 24) != 186 )
    goto LABEL_144;
  v104 = *(_QWORD *)(*(_QWORD *)(v103 + 40) + 40LL);
  v105 = *(_DWORD *)(v104 + 24);
  if ( v105 != 35 && v105 != 11 )
    goto LABEL_144;
  v142 = v12;
  v160 = v11;
  _R14 = *(_QWORD *)(v104 + 96) + 24LL;
  v151 = *(_QWORD *)(v104 + 96);
  v107 = sub_1002450(_R14);
  v108 = v151;
  v11 = v160;
  v12 = v142;
  if ( v107 )
  {
    if ( *(_DWORD *)(v151 + 32) > 0x40u )
    {
      v130 = sub_C445E0(_R14);
      v12 = v142;
      v11 = v160;
      v94 = v130;
    }
    else
    {
      v94 = 64;
      _RAX = ~*(_QWORD *)(v151 + 24);
      __asm { tzcnt   rdx, rax }
      if ( *(_QWORD *)(v151 + 24) != -1 )
        v94 = (unsigned int)_RDX;
    }
    v159 = v12;
    v150 = v11;
    v97 = sub_327FC40(*(_QWORD **)(*a1 + 64LL), v94);
    v193.m128i_i64[1] = v98;
    v193.m128i_i32[0] = v97;
    v99 = sub_32844A0((unsigned __int16 *)&v180, v94);
    v100 = sub_32844A0((unsigned __int16 *)&v193, v94);
    v11 = v150;
    v12 = v159;
    if ( v99 > v100 )
    {
      v101 = *(unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16LL * v9);
      if ( (_WORD)v101 )
      {
        if ( v193.m128i_i16[0]
          && (((int)*(unsigned __int16 *)(a1[1] + 2 * (v193.m128i_u16[0] + 274 * v101 + 71704) + 6) >> (4 * v159)) & 0xF) == 0 )
        {
          v180 = _mm_loadu_si128(&v193);
        }
      }
    }
    goto LABEL_144;
  }
  if ( v142 == 3 )
  {
    if ( *(_DWORD *)(v151 + 32) <= 0x40u )
    {
      _RDI = *(_QWORD *)(v151 + 24);
      if ( _RDI && ((_RDI | (_RDI - 1)) & ((_RDI | (_RDI - 1)) + 1)) == 0 )
      {
        __asm { tzcnt   r14, rdi }
        v132 = sub_39FAC40(_RDI);
        v11 = v160;
        v12 = v142;
        v172 = v132;
        goto LABEL_158;
      }
    }
    else
    {
      v137 = *(_DWORD *)(v151 + 32);
      v172 = sub_C44630(_R14);
      v152 = sub_C444A0(_R14);
      v109 = sub_C44590(_R14);
      v11 = v160;
      LODWORD(_R14) = v109;
      v12 = v142;
      if ( v137 == v109 + v172 + v152 )
      {
LABEL_158:
        v143 = v12;
        v161 = v11;
        v153 = v11 + _R14;
        v110 = sub_32844A0((unsigned __int16 *)&v179, v108);
        v11 = v161;
        v12 = v143;
        if ( v153 < v110 )
        {
          v135 = v143;
          v138 = v161;
          v111 = sub_327FC40(*(_QWORD **)(*a1 + 64LL), v172);
          v144 = v112;
          v162 = v111;
          v113 = sub_32844A0((unsigned __int16 *)&v180, v172);
          v11 = v138;
          v12 = v135;
          if ( (unsigned int)_R14 + v172 <= v113 )
          {
            v114 = *(unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16LL * v9);
            if ( (_WORD)v114 )
            {
              if ( (_WORD)v162
                && (*(_BYTE *)(a1[1] + 2 * ((unsigned __int16)v162 + 274 * v114 + 71704) + 7) & 0xF0) == 0 )
              {
                v180.m128i_i64[0] = v162;
                v174 = _R14;
                v11 = v153;
                v180.m128i_i64[1] = v144;
              }
            }
          }
        }
      }
    }
  }
LABEL_144:
  v102 = *(_DWORD **)(v8 + 40);
  v8 = *(_QWORD *)v102;
  v9 = v102[2];
  v17 = *(_DWORD *)(*(_QWORD *)v102 + 24LL);
LABEL_95:
  if ( v11 )
  {
    LODWORD(v18) = 0;
    goto LABEL_21;
  }
LABEL_20:
  v11 = 0;
  LODWORD(v18) = 0;
  if ( v17 == 190 )
  {
    v82 = *(_QWORD *)(v8 + 56);
    if ( !v82 )
      return 0;
    v83 = 1;
    do
    {
      if ( v9 == *(_DWORD *)(v82 + 8) )
      {
        if ( !v83 )
          return 0;
        v82 = *(_QWORD *)(v82 + 32);
        if ( !v82 )
          goto LABEL_123;
        if ( v9 == *(_DWORD *)(v82 + 8) )
          return 0;
        v83 = 0;
      }
      v82 = *(_QWORD *)(v82 + 32);
    }
    while ( v82 );
    if ( v83 == 1 )
      return 0;
LABEL_123:
    if ( v179.m128i_i16[0] != v180.m128i_i16[0] || !v180.m128i_i16[0] && v179.m128i_i64[1] != v180.m128i_i64[1] )
      return 0;
    v87 = a1[1];
    v88 = *(__int64 (**)())(*(_QWORD *)v87 + 1656LL);
    if ( v88 == sub_2FE3580 )
      return 0;
    v171 = v12;
    v89 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD, __int64))v88)(
            v87,
            v2,
            *(unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16LL * v9),
            *(_QWORD *)(*(_QWORD *)(v8 + 48) + 16LL * v9 + 8),
            v179.m128i_u32[0],
            v179.m128i_i64[1]);
    v12 = v171;
    if ( v89 && ((v90 = *(__int64 **)(v8 + 40), v91 = v90[5], v92 = *(_DWORD *)(v91 + 24), v92 == 11) || v92 == 35) )
    {
      v126 = *(_QWORD *)(v91 + 96);
      v18 = *(_QWORD **)(v126 + 24);
      if ( *(_DWORD *)(v126 + 32) > 0x40u )
        v18 = (_QWORD *)*v18;
      v8 = *v90;
      v11 = 0;
      v17 = *(_DWORD *)(*v90 + 24);
    }
    else
    {
      v17 = *(_DWORD *)(v8 + 24);
      v11 = 0;
      LODWORD(v18) = 0;
    }
  }
LABEL_21:
  v163 = (unsigned int)v18;
  if ( v17 != 298 )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(v8 + 112) + 37LL) & 0xF) != 0 )
    return 0;
  if ( (*(_BYTE *)(v8 + 32) & 8) != 0 )
    return 0;
  v155 = v11;
  v146 = v12;
  v19 = sub_3281800((__int64)a1, v8, v12, &v180, v11);
  v20 = v155;
  if ( !v19 )
    return 0;
  v156 = v163;
  v164 = v20;
  v21 = (_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40LL));
  v22 = v164;
  v23 = v146;
  v24 = v156;
  if ( *v21 )
  {
    v120 = *(_WORD *)(v8 + 96);
    *((_QWORD *)&v191 + 1) = *(_QWORD *)(v8 + 104);
    LOWORD(v191) = v120;
    v121.m128i_i64[0] = sub_3285A00((unsigned __int16 *)&v191);
    v193 = v121;
    v122 = 8 * v121.m128i_i32[0];
    v123.m128i_i64[0] = sub_3285A00((unsigned __int16 *)&v180);
    v24 = v156;
    v193 = v123;
    v23 = v146;
    v22 = v122 - 8 * v123.m128i_i32[0] - v164;
  }
  v25 = v22 >> 3;
  v181 = *(_QWORD *)(v8 + 80);
  if ( v181 )
  {
    v147 = v24;
    v165 = v23;
    sub_325F5D0(&v181);
    v24 = v147;
    v23 = v165;
  }
  v26 = *(_DWORD *)(v8 + 72);
  LOBYTE(v184) = 0;
  v182 = v26;
  v27 = *(_QWORD *)(v8 + 40);
  v28 = *a1;
  v157 = v24;
  v29 = *(_QWORD *)(v27 + 40);
  v30 = *(_QWORD *)(v27 + 48);
  v166 = v23;
  v183 = v25;
  v31 = sub_3409320(v28, v29, v30, v25, v184, (unsigned int)&v181, 1);
  v154 = v32;
  v148 = v31;
  sub_32B3E80((__int64)a1, v31, 1, 0, v33, v34);
  v35 = *(_QWORD *)(v8 + 112);
  if ( v166 )
  {
    v115 = *a1;
    v116 = 256;
    v193 = _mm_loadu_si128((const __m128i *)(v35 + 40));
    v194 = _mm_loadu_si128((const __m128i *)(v35 + 56));
    LOBYTE(v116) = *(_BYTE *)(v35 + 34);
    v139 = *(unsigned __int16 *)(v35 + 32);
    v145 = v116;
    sub_327C6E0((__int64)&v191, (__int64 *)v35, v25);
    v38 = sub_33F1DB0(
            v115,
            v166,
            (unsigned int)&v181,
            v179.m128i_i32[0],
            v179.m128i_i32[2],
            v145,
            **(_QWORD **)(v8 + 40),
            *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL),
            v148,
            v154,
            v191,
            v192,
            v180.m128i_i64[0],
            v180.m128i_i64[1],
            v139,
            (__int64)&v193);
  }
  else
  {
    HIBYTE(v36) = 1;
    v37 = *a1;
    v193 = _mm_loadu_si128((const __m128i *)(v35 + 40));
    v194 = _mm_loadu_si128((const __m128i *)(v35 + 56));
    LOBYTE(v36) = *(_BYTE *)(v35 + 34);
    v136 = *(unsigned __int16 *)(v35 + 32);
    v140 = v36;
    sub_327C6E0((__int64)&v189, (__int64 *)v35, v25);
    v38 = sub_33F1F00(
            v37,
            v179.m128i_i32[0],
            v179.m128i_i32[2],
            (unsigned int)&v181,
            **(_QWORD **)(v8 + 40),
            *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL),
            v148,
            v154,
            v189,
            v190,
            v140,
            v136,
            (__int64)&v193,
            0);
  }
  v149 = v39;
  v167 = v38;
  v40 = *(_QWORD *)(*a1 + 768LL);
  v194.m128i_i64[0] = *a1;
  v193.m128i_i64[1] = v40;
  *(_QWORD *)(v194.m128i_i64[0] + 768) = &v193;
  v41 = *a1;
  v194.m128i_i64[1] = (__int64)a1;
  v193.m128i_i64[0] = (__int64)off_4A360B8;
  sub_34161C0(v41, v8, 1, v38, 1);
  v42 = v149;
  v43 = v167;
  v44 = v149;
  if ( v157 )
  {
    if ( v157 < (unsigned __int64)sub_32844A0((unsigned __int16 *)&v179, v157) )
    {
      v127 = *a1;
      *(_QWORD *)&v128 = sub_3400E40(*a1, v157, v179.m128i_u32[0], v179.m128i_i64[1], &v181);
      *((_QWORD *)&v134 + 1) = v149;
      *(_QWORD *)&v134 = v167;
      v43 = sub_3406EB0(v127, 190, (unsigned int)&v181, v179.m128i_i32[0], v179.m128i_i32[2], v129, v134, v128);
    }
    else
    {
      v43 = sub_3400BD0(*a1, 0, (unsigned int)&v181, v179.m128i_i32[0], v179.m128i_i32[2], 0, 0, v124);
    }
    v42 = v125;
    v44 = v125;
  }
  if ( v174 )
  {
    v173 = v43;
    *(_QWORD *)&v117 = sub_3400BD0(*a1, v174, (unsigned int)&v181, v179.m128i_i32[0], v179.m128i_i32[2], 0, 0);
    v118 = v42 | v44 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v133 + 1) = v118;
    *(_QWORD *)&v133 = v173;
    v176 = sub_3406EB0(*a1, 190, (unsigned int)&v181, v179.m128i_i32[0], v179.m128i_i32[2], 0, v133, v117);
    sub_34161C0(*a1, v2, 0, v176, v119 | v118 & 0xFFFFFFFF00000000LL);
    v43 = v176;
  }
  result = v43;
  *(_QWORD *)(v194.m128i_i64[0] + 768) = v193.m128i_i64[1];
  if ( v181 )
  {
    v178 = v43;
    sub_B91220((__int64)&v181, v181);
    return v178;
  }
  return result;
}
