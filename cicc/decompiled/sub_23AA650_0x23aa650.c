// Function: sub_23AA650
// Address: 0x23aa650
//
unsigned __int64 *__fastcall sub_23AA650(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4)
{
  _QWORD *v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int32 v20; // r15d
  __int64 v21; // rax
  int v22; // r9d
  _BOOL8 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // r9
  unsigned __int64 v29; // r15
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  _BOOL8 v33; // rsi
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  int v43; // eax
  _QWORD *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  _QWORD *v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  _QWORD *v61; // rax
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  __int64 v65; // r9
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  _QWORD *v72; // rax
  _QWORD *v73; // rax
  _QWORD *v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  _QWORD *v77; // rbx
  _QWORD *v78; // r12
  int v80; // r15d
  __m128i v81; // xmm4
  __m128i v82; // xmm3
  __m128i v83; // xmm2
  __m128i v84; // xmm1
  __m128i v85; // xmm0
  _QWORD *v86; // rax
  __int64 v87; // r8
  __int64 v88; // r9
  char *v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int8 v94; // r15
  _QWORD *v95; // rax
  char v96; // al
  __int64 v97; // rax
  __int64 v98; // rax
  __int8 v99; // r15
  __int64 v100; // rax
  bool v101; // r15
  __int64 v102; // rax
  _QWORD *v103; // rax
  _QWORD *v104; // r15
  _QWORD *v105; // rbx
  __int64 v106; // rax
  _QWORD *v107; // rax
  __int64 v108; // rax
  _QWORD *v109; // rax
  _QWORD *v110; // rax
  _QWORD *v111; // rax
  _QWORD *v112; // rax
  __int64 v113; // rdi
  unsigned __int8 v114; // [rsp+66h] [rbp-E9Ah]
  char v115; // [rsp+67h] [rbp-E99h]
  _QWORD *v116; // [rsp+68h] [rbp-E98h]
  unsigned __int64 v118; // [rsp+90h] [rbp-E70h]
  __int64 v120; // [rsp+A8h] [rbp-E58h]
  int v121; // [rsp+B0h] [rbp-E50h]
  __int64 v122; // [rsp+B4h] [rbp-E4Ch]
  int v123; // [rsp+BCh] [rbp-E44h]
  _QWORD v124[4]; // [rsp+C0h] [rbp-E40h] BYREF
  unsigned __int64 v125[6]; // [rsp+E0h] [rbp-E20h] BYREF
  _QWORD *v126; // [rsp+110h] [rbp-DF0h] BYREF
  _QWORD *v127; // [rsp+118h] [rbp-DE8h]
  __int64 v128; // [rsp+120h] [rbp-DE0h]
  __int64 v129; // [rsp+128h] [rbp-DD8h]
  __int64 v130; // [rsp+130h] [rbp-DD0h]
  unsigned __int64 v131[6]; // [rsp+140h] [rbp-DC0h] BYREF
  unsigned __int64 v132[6]; // [rsp+170h] [rbp-D90h] BYREF
  unsigned __int64 v133[6]; // [rsp+1A0h] [rbp-D60h] BYREF
  __m128i v134; // [rsp+1D0h] [rbp-D30h] BYREF
  __m128i v135; // [rsp+1E0h] [rbp-D20h] BYREF
  __m128i v136; // [rsp+1F0h] [rbp-D10h] BYREF
  __m128i v137; // [rsp+200h] [rbp-D00h] BYREF
  __m128i v138; // [rsp+210h] [rbp-CF0h] BYREF
  unsigned int v139; // [rsp+220h] [rbp-CE0h]
  __m128i v140; // [rsp+230h] [rbp-CD0h] BYREF
  __m128i v141; // [rsp+240h] [rbp-CC0h] BYREF
  __m128i v142; // [rsp+250h] [rbp-CB0h] BYREF
  __m128i v143; // [rsp+260h] [rbp-CA0h] BYREF
  __m128i v144; // [rsp+270h] [rbp-C90h] BYREF
  int v145; // [rsp+280h] [rbp-C80h]
  __m128i v146; // [rsp+290h] [rbp-C70h] BYREF
  __m128i v147; // [rsp+2A0h] [rbp-C60h] BYREF
  __m128i v148; // [rsp+2B0h] [rbp-C50h]
  __m128i v149; // [rsp+2C0h] [rbp-C40h]
  __m128i v150; // [rsp+2D0h] [rbp-C30h] BYREF
  unsigned __int64 *v151; // [rsp+2E0h] [rbp-C20h]
  __int64 v152; // [rsp+2E8h] [rbp-C18h]
  unsigned __int64 v153; // [rsp+2F0h] [rbp-C10h] BYREF
  char *v154; // [rsp+2F8h] [rbp-C08h]
  char *v155; // [rsp+300h] [rbp-C00h]
  __int64 v156; // [rsp+308h] [rbp-BF8h]
  char v157[216]; // [rsp+318h] [rbp-BE8h] BYREF
  __int64 v158; // [rsp+3F0h] [rbp-B10h]
  __int64 v159; // [rsp+3F8h] [rbp-B08h]
  __int64 v160; // [rsp+400h] [rbp-B00h]
  int v161; // [rsp+408h] [rbp-AF8h]
  __int64 v162; // [rsp+410h] [rbp-AF0h]
  __int64 v163; // [rsp+418h] [rbp-AE8h]
  char *v164; // [rsp+420h] [rbp-AE0h]
  __int64 v165; // [rsp+428h] [rbp-AD8h]
  char v166; // [rsp+430h] [rbp-AD0h] BYREF
  _QWORD *v167; // [rsp+450h] [rbp-AB0h]
  __int64 v168; // [rsp+458h] [rbp-AA8h]
  _QWORD v169[5]; // [rsp+460h] [rbp-AA0h] BYREF
  char v170; // [rsp+488h] [rbp-A78h] BYREF
  _QWORD v171[2]; // [rsp+4C8h] [rbp-A38h] BYREF
  char v172; // [rsp+4D8h] [rbp-A28h] BYREF
  char *v173; // [rsp+518h] [rbp-9E8h]
  __int64 v174; // [rsp+520h] [rbp-9E0h]
  char v175; // [rsp+528h] [rbp-9D8h] BYREF
  __int64 v176; // [rsp+568h] [rbp-998h]
  __int64 v177; // [rsp+570h] [rbp-990h]
  __int64 v178; // [rsp+578h] [rbp-988h]
  int v179; // [rsp+580h] [rbp-980h]
  char v180; // [rsp+588h] [rbp-978h]
  _BYTE *v181; // [rsp+590h] [rbp-970h]
  __int64 v182; // [rsp+598h] [rbp-968h]
  _BYTE v183[64]; // [rsp+5A0h] [rbp-960h] BYREF
  _QWORD *v184; // [rsp+5E0h] [rbp-920h] BYREF
  __m128i v185; // [rsp+5E8h] [rbp-918h] BYREF
  __m128i v186; // [rsp+5F8h] [rbp-908h]
  __m128i v187; // [rsp+608h] [rbp-8F8h]
  __m128i v188; // [rsp+618h] [rbp-8E8h] BYREF
  __m128i v189; // [rsp+628h] [rbp-8D8h]
  unsigned __int64 v190; // [rsp+638h] [rbp-8C8h]
  int v191; // [rsp+640h] [rbp-8C0h]
  int v192; // [rsp+648h] [rbp-8B8h] BYREF
  __int64 v193; // [rsp+650h] [rbp-8B0h]
  int *v194; // [rsp+658h] [rbp-8A8h]
  int *v195; // [rsp+660h] [rbp-8A0h]
  __int64 v196; // [rsp+668h] [rbp-898h]
  char v197; // [rsp+670h] [rbp-890h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v118 = HIDWORD(a3);
  sub_23A11A0(a2, (__int64)a1, a3);
  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
    *v6 = &unk_4A0D038;
  v184 = v6;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  if ( (_DWORD)v118 == HIDWORD(qword_5033F08) && (_DWORD)a3 == (_DWORD)qword_5033F08 )
  {
    v106 = sub_22077B0(0x20u);
    if ( v106 )
    {
      *(_QWORD *)(v106 + 16) = 0;
      *(_BYTE *)(v106 + 24) = 0;
      *(_QWORD *)(v106 + 8) = a4;
      *(_QWORD *)v106 = &unk_4A0E478;
    }
    v184 = (_QWORD *)v106;
    sub_23A2230(a1, (unsigned __int64 *)&v184);
    if ( v184 )
      (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
LABEL_171:
    LOBYTE(v184) = 0;
    v185 = (__m128i)a4;
    v186.m128i_i32[0] = 0;
    sub_23A23F0(a1, (char *)&v184);
    v184 = 0;
    v185 = 0u;
    v186.m128i_i32[0] = 1;
    sub_23A23F0(a1, (char *)&v184);
    sub_23A3310((__int64)&v184);
    sub_23A2270(a1, (unsigned __int64 *)&v184);
    sub_234A900((__int64)&v184);
    sub_23A1210(a2, (__int64)a1, a3);
    sub_23A2610(a1);
    return a1;
  }
  if ( *(_BYTE *)(a2 + 192) && *(_DWORD *)(a2 + 168) == 3 )
  {
    v134.m128i_i64[0] = 0;
    sub_2241BD0(v146.m128i_i64, a2 + 104);
    sub_2241BD0(v140.m128i_i64, a2 + 40);
    sub_26C1D00((unsigned int)&v184, (unsigned int)&v140, (unsigned int)&v146, 4, (unsigned int)&v134, 0, 0);
    sub_2357AD0(a1, (__m128i *)&v184);
    sub_233AA80((unsigned __int64 *)&v184);
    sub_2240A30((unsigned __int64 *)&v140);
    sub_2240A30((unsigned __int64 *)&v146);
    if ( v134.m128i_i64[0] )
      sub_23569D0((volatile signed __int32 *)(v134.m128i_i64[0] + 8));
    sub_23A2700(a1);
  }
  v7 = sub_22077B0(0x10u);
  if ( v7 )
  {
    *(_DWORD *)(v7 + 8) = 4;
    *(_QWORD *)v7 = &unk_4A0DAB8;
  }
  v184 = (_QWORD *)v7;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  sub_23A0BA0((__int64)&v184, 1);
  sub_23A2670(a1, (__int64)&v184);
  sub_233AAF0((__int64)&v184);
  v8 = (_QWORD *)sub_22077B0(0x10u);
  if ( v8 )
    *v8 = &unk_4A0D478;
  v184 = v8;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  if ( (unsigned int)a3 > 1 )
  {
    v94 = *(_BYTE *)(a2 + 32);
    v95 = (_QWORD *)sub_22077B0(0x10u);
    if ( v95 )
      *v95 = &unk_4A0EFF8;
    v184 = v95;
    v185.m128i_i8[0] = v94;
    sub_23571D0(a1, (__int64 *)&v184);
    sub_233EFE0((__int64 *)&v184);
    v96 = *(_BYTE *)(a2 + 192);
    if ( v96 )
      v96 = *(_DWORD *)(a2 + 168) == 3;
    LOBYTE(v184) = 1;
    BYTE1(v184) = v96;
    sub_23A2390(a1, (__int16 *)&v184);
    v184 = 0;
    v185 = 0u;
    v186 = 0u;
    v97 = sub_22077B0(0x10u);
    if ( v97 )
    {
      *(_BYTE *)(v97 + 8) = 0;
      *(_QWORD *)v97 = &unk_4A0EC78;
    }
    v146.m128i_i64[0] = v97;
    sub_23A32D0((unsigned __int64 *)&v184, (unsigned __int64 *)&v146);
    if ( v146.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v146.m128i_i64[0] + 8LL))(v146.m128i_i64[0]);
    v98 = sub_22077B0(0x10u);
    if ( v98 )
    {
      *(_DWORD *)(v98 + 8) = 2;
      *(_QWORD *)v98 = &unk_4A0EA78;
    }
    v146.m128i_i64[0] = v98;
    sub_23A32D0((unsigned __int64 *)&v184, (unsigned __int64 *)&v146);
    if ( v146.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v146.m128i_i64[0] + 8LL))(v146.m128i_i64[0]);
    sub_291E720(&v140, 0);
    v99 = v140.m128i_i8[0];
    v100 = sub_22077B0(0x10u);
    if ( v100 )
    {
      *(_BYTE *)(v100 + 8) = v99;
      *(_QWORD *)v100 = &unk_4A11C38;
    }
    v146.m128i_i64[0] = v100;
    v146.m128i_i16[4] = 0;
    sub_235A8B0((unsigned __int64 *)&v184, v146.m128i_i64);
    sub_233EFE0(v146.m128i_i64);
    sub_234A9E0(&v146, (unsigned __int64 *)&v184);
    sub_2357280(a1, v146.m128i_i64);
    sub_233F000(v146.m128i_i64);
    if ( (_DWORD)v118 != unk_5033EEC || (v101 = 0, (_DWORD)a3 != unk_5033EE8) )
    {
      v101 = 1;
      if ( (_DWORD)v118 == HIDWORD(qword_5033EE0) )
        v101 = (_DWORD)qword_5033EE0 != (_DWORD)a3;
    }
    v102 = sub_22077B0(0x10u);
    if ( v102 )
    {
      *(_BYTE *)(v102 + 8) = v101;
      *(_QWORD *)v102 = &unk_4A0E8B8;
    }
    v146.m128i_i64[0] = v102;
    sub_23A2230(a1, (unsigned __int64 *)&v146);
    sub_23501E0(v146.m128i_i64);
    v103 = (_QWORD *)sub_22077B0(0x10u);
    if ( v103 )
      *v103 = &unk_4A0CEF8;
    v146.m128i_i64[0] = (__int64)v103;
    sub_23A2230(a1, (unsigned __int64 *)&v146);
    sub_23501E0(v146.m128i_i64);
    v104 = (_QWORD *)v185.m128i_i64[0];
    v105 = v184;
    if ( (_QWORD *)v185.m128i_i64[0] != v184 )
    {
      do
      {
        if ( *v105 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v105 + 8LL))(*v105);
        ++v105;
      }
      while ( v104 != v105 );
      v105 = v184;
    }
    if ( v105 )
      j_j___libc_free_0((unsigned __int64)v105);
  }
  v9 = (_QWORD *)sub_22077B0(0x10u);
  if ( v9 )
    *v9 = &unk_4A0E0B8;
  v184 = v9;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  v10 = (_QWORD *)sub_22077B0(0x10u);
  if ( v10 )
    *v10 = &unk_4A0D3F8;
  v184 = v10;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  v11 = sub_22077B0(0x20u);
  if ( v11 )
  {
    *(_QWORD *)(v11 + 16) = 0;
    *(_BYTE *)(v11 + 24) = 0;
    *(_QWORD *)(v11 + 8) = a4;
    *(_QWORD *)v11 = &unk_4A0E478;
  }
  v184 = (_QWORD *)v11;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  if ( (_DWORD)v118 == unk_5033F04 && (_DWORD)a3 == unk_5033F00 )
    goto LABEL_171;
  v12 = (_QWORD *)sub_22077B0(0x10u);
  if ( v12 )
    *v12 = &unk_4A0CFF8;
  v184 = v12;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  v13 = (_QWORD *)sub_22077B0(0x10u);
  if ( v13 )
    *v13 = &unk_4A0D3B8;
  v184 = v13;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  v14 = (_QWORD *)sub_22077B0(0x10u);
  if ( v14 )
    *v14 = &unk_4A0FFF8;
  v184 = v14;
  v185.m128i_i8[0] = 0;
  sub_23571D0(a1, (__int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v15 = (_QWORD *)sub_22077B0(0x10u);
  if ( v15 )
    *v15 = &unk_4A0CF78;
  v184 = v15;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  v186.m128i_i64[0] = (__int64)&v185;
  v186.m128i_i64[1] = (__int64)&v185;
  v189.m128i_i64[0] = (__int64)&v188;
  v189.m128i_i64[1] = (__int64)&v188;
  v194 = &v192;
  v195 = &v192;
  v185.m128i_i32[0] = 0;
  v185.m128i_i64[1] = 0;
  v187.m128i_i64[0] = 0;
  v188.m128i_i32[0] = 0;
  v188.m128i_i64[1] = 0;
  v190 = 0;
  v192 = 0;
  v193 = 0;
  v196 = 0;
  v197 = 0;
  sub_2358990(a1, (__int64)&v184);
  sub_233A870(&v184);
  LOBYTE(v120) = 0;
  HIDWORD(v120) = 1;
  LOBYTE(v121) = 0;
  memset(v125, 0, 40);
  sub_F10C20((__int64)&v184, v120, v121);
  sub_2353C90(v125, (__int64)&v184, v16, v17, v18, v19);
  sub_233BCC0((__int64)&v184);
  if ( (unsigned int)a3 > 1 )
  {
    v86 = (_QWORD *)sub_22077B0(0x10u);
    if ( v86 )
      *v86 = &unk_4A0EDB8;
    v184 = v86;
    sub_23A1F40(v125, (unsigned __int64 *)&v184);
    sub_233EFE0((__int64 *)&v184);
  }
  sub_23A0D70(a2, (__int64)v125, a3);
  sub_234AAB0((__int64)&v184, (__int64 *)v125, *(_BYTE *)(a2 + 32));
  sub_23571D0(a1, (__int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  sub_25B6750(&v146, 2);
  v20 = v146.m128i_i32[0];
  v21 = sub_22077B0(0x10u);
  if ( v21 )
  {
    *(_DWORD *)(v21 + 8) = v20;
    *(_QWORD *)v21 = &unk_4A0D2B8;
  }
  v184 = (_QWORD *)v21;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  if ( byte_4FDDF48 )
  {
    v80 = dword_4FDE108;
    sub_30D6B60(&v134, (unsigned int)a3, (unsigned int)v118);
    v81 = _mm_loadu_si128(&v134);
    v82 = _mm_loadu_si128(&v135);
    v83 = _mm_loadu_si128(&v136);
    v84 = _mm_loadu_si128(&v137);
    v184 = 0;
    v85 = _mm_loadu_si128(&v138);
    LODWORD(v151) = v139;
    v190 = __PAIR64__(v80, v139);
    v191 = 4;
    v146 = v81;
    v147 = v82;
    v148 = v83;
    v149 = v84;
    v150 = v85;
    v185 = v81;
    v186 = v82;
    v187 = v83;
    v188 = v84;
    v189 = v85;
    sub_2358340(a1, (__int64 *)&v184);
    if ( v184 )
      (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  }
  else
  {
    sub_30D6B60(&v140, (unsigned int)a3, (unsigned int)v118);
    sub_26124A0(
      (unsigned int)&v184,
      1,
      4,
      0,
      0,
      v22,
      *(_OWORD *)&_mm_loadu_si128(&v140),
      *(_OWORD *)&_mm_loadu_si128(&v141),
      *(_OWORD *)&_mm_loadu_si128(&v142),
      *(_OWORD *)&_mm_loadu_si128(&v143),
      *(_OWORD *)&_mm_loadu_si128(&v144),
      v145);
    sub_2357600(a1, (__int64)&v184);
    sub_233A900(&v184);
  }
  if ( LOBYTE(qword_4FF33E0[17]) )
  {
    v23 = 0;
    if ( *(_BYTE *)(a2 + 192) )
      v23 = *(_DWORD *)(a2 + 168) == 3;
    sub_264AF10(&v184, 0, v23);
    v24 = (__int64)&v184;
    sub_2356F30(a1, (__int64 *)&v184);
    v29 = v186.m128i_u64[1];
    if ( v186.m128i_i64[1] )
    {
      if ( *(_QWORD *)v186.m128i_i64[1] != v186.m128i_i64[1] + 16 )
        _libc_free(*(_QWORD *)v186.m128i_i64[1]);
      v24 = 80;
      j_j___libc_free_0(v29);
    }
    if ( v186.m128i_i64[0] )
      sub_23A1CE0(v186.m128i_u64[0], v24, v25, v26, v27, v28);
    if ( v185.m128i_i64[0] )
      sub_23A0670(v185.m128i_u64[0]);
  }
  v30 = (_QWORD *)sub_22077B0(0x10u);
  if ( v30 )
    *v30 = &unk_4A0D3B8;
  v184 = v30;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  v31 = sub_22077B0(0x10u);
  if ( v31 )
  {
    *(_DWORD *)(v31 + 8) = 4;
    *(_QWORD *)v31 = &unk_4A0DAB8;
  }
  v184 = (_QWORD *)v31;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  sub_23A0BA0((__int64)&v184, 1);
  sub_23A2670(a1, (__int64)&v184);
  sub_233AAF0((__int64)&v184);
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v32 = sub_22077B0(0x10u);
  if ( v32 )
  {
    *(_DWORD *)(v32 + 8) = 2;
    *(_QWORD *)v32 = &unk_4A0EA78;
  }
  v184 = (_QWORD *)v32;
  sub_23A32D0((unsigned __int64 *)&v126, (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  v33 = 1;
  if ( (_DWORD)v118 == HIDWORD(qword_5033F08) )
    v33 = (_DWORD)qword_5033F08 != (_DWORD)a3;
  sub_24E6490(&v184, v33);
  sub_235AA40((unsigned __int64 *)&v126, (__m128i *)&v184);
  if ( v185.m128i_i64[1] )
    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v185.m128i_i64[1])(&v184, &v184, 3);
  v34 = (_QWORD *)sub_22077B0(0x10u);
  if ( v34 )
    *v34 = &unk_4A0EBF8;
  v184 = v34;
  sub_23A32D0((unsigned __int64 *)&v126, (unsigned __int64 *)&v184);
  sub_233F000((__int64 *)&v184);
  sub_234A9E0(&v184, (unsigned __int64 *)&v126);
  sub_2357280(a1, (__int64 *)&v184);
  sub_233F000((__int64 *)&v184);
  LOBYTE(v122) = 0;
  HIDWORD(v122) = 1;
  LOBYTE(v123) = 0;
  memset(v131, 0, 40);
  sub_F10C20((__int64)&v184, v122, v123);
  sub_2353C90(v131, (__int64)&v184, v35, v36, v37, v38);
  sub_233BCC0((__int64)&v184);
  sub_23A0D70(a2, (__int64)v131, a3);
  if ( byte_4FDC968 )
  {
    v107 = (_QWORD *)sub_22077B0(0x10u);
    if ( v107 )
      *v107 = &unk_4A0F138;
    v184 = v107;
    sub_23A1F40(v131, (unsigned __int64 *)&v184);
    sub_233EFE0((__int64 *)&v184);
  }
  sub_27DC820(&v184, 0xFFFFFFFFLL);
  sub_2354380(v131, (__int64 *)&v184);
  sub_233B480((__int64)&v184, (__int64)&v184, v39, v40, v41, v42);
  if ( *(_BYTE *)(a2 + 192) )
  {
    v43 = *(_DWORD *)(a2 + 172);
    if ( v43 == 1 )
    {
      v133[0] = *(_QWORD *)(a2 + 184);
      sub_239EC60(v133[0]);
      sub_2241BD0((__int64 *)&v184, a2 + 104);
      sub_2241BD0(v146.m128i_i64, a2 + 72);
      sub_23A2D30(a2, a1, a3, 1, 1u, *(_BYTE *)(a2 + 182), (unsigned __int64 *)&v146, (__int64)&v184, (__int64 *)v133);
    }
    else
    {
      if ( v43 != 2 )
        goto LABEL_63;
      v133[0] = *(_QWORD *)(a2 + 184);
      sub_239EC60(v133[0]);
      sub_2241BD0((__int64 *)&v184, a2 + 104);
      sub_2241BD0(v146.m128i_i64, a2 + 40);
      sub_23A2D30(a2, a1, a3, 0, 1u, *(_BYTE *)(a2 + 182), (unsigned __int64 *)&v146, (__int64)&v184, (__int64 *)v133);
    }
    sub_2240A30((unsigned __int64 *)&v146);
    sub_2240A30((unsigned __int64 *)&v184);
    if ( v133[0] )
      sub_23569D0((volatile signed __int32 *)(v133[0] + 8));
  }
LABEL_63:
  sub_291E720(&v184, 0);
  sub_23A2000(v131, (char *)&v184);
  v44 = (_QWORD *)sub_22077B0(0x10u);
  if ( v44 )
    *v44 = &unk_4A10F78;
  v184 = v44;
  sub_23A1F40(v131, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  sub_234AAB0((__int64)&v184, (__int64 *)v131, *(_BYTE *)(a2 + 32));
  sub_23571D0(a1, (__int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v45 = sub_22077B0(0x10u);
  if ( v45 )
  {
    *(_BYTE *)(v45 + 8) = 0;
    *(_QWORD *)v45 = &unk_4A0EC78;
  }
  v184 = (_QWORD *)v45;
  sub_2357280(a1, (__int64 *)&v184);
  sub_233F000((__int64 *)&v184);
  if ( byte_4FDDAE8 )
  {
    v109 = (_QWORD *)sub_22077B0(0x10u);
    if ( v109 )
      *v109 = &unk_4A0CD38;
    v184 = v109;
    sub_23A2230(a1, (unsigned __int64 *)&v184);
    sub_23501E0((__int64 *)&v184);
    v110 = (_QWORD *)sub_22077B0(0x10u);
    if ( v110 )
      *v110 = &unk_4A127F8;
    v184 = v110;
    v185.m128i_i8[0] = 0;
    sub_23571D0(a1, (__int64 *)&v184);
    sub_233EFE0((__int64 *)&v184);
  }
  v46 = *(_QWORD *)(a2 + 16);
  v146.m128i_i16[4] = 1;
  v146.m128i_i64[0] = v46;
  memset(v132, 0, 40);
  sub_2356430((__int64)&v184, v146.m128i_i64, 1, 0, 0);
  sub_2353940(v132, (__int64 *)&v184);
  sub_233F7F0((__int64)&v185);
  sub_233F7D0((__int64 *)&v184);
  if ( byte_4FDD848 )
  {
    v47 = (_QWORD *)sub_22077B0(0x10u);
    if ( v47 )
      *v47 = &unk_4A101B8;
    v184 = v47;
    sub_23A1F40(v132, (unsigned __int64 *)&v184);
    sub_233EFE0((__int64 *)&v184);
  }
  else
  {
    v184 = 0;
    v185.m128i_i16[2] = 0;
    v146.m128i_i16[6] = 0;
    v151 = &v153;
    v185.m128i_i32[0] = 0;
    v146.m128i_i64[0] = 0;
    v146.m128i_i32[2] = 0;
    v147 = 0u;
    v148 = 0u;
    v149 = 0u;
    v150 = 0u;
    v152 = 0;
    v153 = 0;
    v154 = 0;
    v155 = 0;
    v156 = 0;
    sub_278A360(v157);
    v158 = 0;
    v164 = &v166;
    v159 = 0;
    v160 = 0;
    v161 = 0;
    v162 = 0;
    v163 = 0;
    v168 = 0;
    v169[0] = 0;
    v169[1] = 1;
    v169[3] = 0;
    v169[4] = 1;
    v165 = 0x400000000LL;
    v167 = v169;
    v89 = &v170;
    do
    {
      *(_QWORD *)v89 = -4096;
      v89 += 16;
    }
    while ( v89 != (char *)v171 );
    v171[0] = &v172;
    v173 = &v175;
    v171[1] = 0x400000000LL;
    v174 = 0x800000000LL;
    v176 = 0;
    v177 = 0;
    v178 = 0;
    v179 = 0;
    v180 = 1;
    v181 = v183;
    v182 = 0x400000000LL;
    sub_2353240((__int64)&v184, v146.m128i_i64, (__int64)v183, 0x800000000LL, v87, v88);
    v90 = (_QWORD *)sub_22077B0(0x358u);
    if ( v90 )
    {
      v116 = v90;
      *v90 = &unk_4A11938;
      sub_2353240((__int64)(v90 + 1), (__int64 *)&v184, (__int64)&unk_4A11938, v91, v92, v93);
      v90 = v116;
    }
    v133[0] = (unsigned __int64)v90;
    sub_23A1F40(v132, v133);
    if ( v133[0] )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v133[0] + 8LL))(v133[0]);
    sub_2341D90((__int64)&v184);
    sub_2341D90((__int64)&v146);
  }
  v48 = (_QWORD *)sub_22077B0(0x48u);
  if ( v48 )
  {
    v48[1] = 0;
    v48[2] = 0;
    v48[3] = 0;
    *v48 = &unk_4A10038;
    v48[4] = 0;
    v48[5] = 0;
    v48[6] = 0;
    v48[7] = 0;
    v48[8] = 0;
  }
  v184 = v48;
  sub_23A1F40(v132, (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  v49 = (_QWORD *)sub_22077B0(0x10u);
  if ( v49 )
    *v49 = &unk_4A0F4B8;
  v184 = v49;
  sub_23A1F40(v132, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v50 = (_QWORD *)sub_22077B0(0x10u);
  if ( v50 )
    *v50 = &unk_4A10138;
  v184 = v50;
  sub_23A1F40(v132, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v51 = sub_22077B0(0x10u);
  if ( v51 )
  {
    *(_BYTE *)(v51 + 8) = 0;
    *(_QWORD *)v51 = &unk_4A11AF8;
  }
  v184 = (_QWORD *)v51;
  sub_23A1F40(v132, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  sub_23A0FA0(a2, (__int64)v132, a3);
  v150.m128i_i32[0] = 0;
  v146.m128i_i64[0] = (__int64)&v147;
  v146.m128i_i64[1] = 0x600000000LL;
  v150.m128i_i64[1] = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  if ( !(_BYTE)qword_4FDD5A8 || (unsigned int)a3 <= 1 )
    goto LABEL_84;
  sub_2332320((__int64)&v146, 1, v52, v53, v54, v55);
  v113 = sub_22077B0(0x10u);
  if ( v113 )
    *(_QWORD *)v113 = &unk_4A13D38;
  v184 = (_QWORD *)v113;
  if ( v154 == v155 )
  {
    sub_235B010(&v153, v154, &v184);
    v113 = (__int64)v184;
LABEL_203:
    if ( v113 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v113 + 8LL))(v113);
    goto LABEL_84;
  }
  if ( !v154 )
  {
    v154 = (char *)8;
    goto LABEL_203;
  }
  *(_QWORD *)v154 = v113;
  v154 += 8;
LABEL_84:
  sub_2332320((__int64)&v146, 0, v52, v53, v54, v55);
  v56 = sub_22077B0(0x10u);
  if ( v56 )
  {
    *(_BYTE *)(v56 + 8) = 1;
    *(_QWORD *)v56 = &unk_4A11F78;
  }
  v184 = (_QWORD *)v56;
  sub_23A46F0(&v150.m128i_u64[1], (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  sub_2332320((__int64)&v146, 0, v57, v58, v59, v60);
  v61 = (_QWORD *)sub_22077B0(0x10u);
  if ( v61 )
    *v61 = &unk_4A12038;
  v184 = v61;
  sub_23A46F0(&v150.m128i_u64[1], (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  v114 = *(_BYTE *)(a2 + 13);
  v115 = *(_BYTE *)(a2 + 11) ^ 1;
  sub_2332320((__int64)&v146, 0, *(unsigned __int8 *)(a2 + 11) ^ 1u, v114, v62, v63);
  v64 = sub_22077B0(0x10u);
  if ( v64 )
  {
    *(_DWORD *)(v64 + 8) = a3;
    *(_QWORD *)v64 = &unk_4A12238;
    *(_BYTE *)(v64 + 12) = v115;
    *(_BYTE *)(v64 + 13) = v114;
  }
  v184 = (_QWORD *)v64;
  sub_23A46F0(&v150.m128i_u64[1], (unsigned __int64 *)&v184);
  if ( v184 )
    (*(void (__fastcall **)(_QWORD *))(*v184 + 8LL))(v184);
  sub_23A20C0((__int64)&v184, (__int64)&v146, 0, 1, 0, v65);
  sub_2353940(v132, (__int64 *)&v184);
  sub_233F7F0((__int64)&v185);
  sub_233F7D0((__int64 *)&v184);
  v66 = (_QWORD *)sub_22077B0(0x10u);
  if ( v66 )
    *v66 = &unk_4A0FCF8;
  v184 = v66;
  sub_23A1F40(v132, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  sub_23A95C0(a2, a3, v132, 1);
  sub_23A1010(a2, (__int64)v132, a3);
  v67 = sub_22077B0(0x10u);
  if ( v67 )
  {
    *(_DWORD *)(v67 + 8) = 4;
    *(_QWORD *)v67 = &unk_4A0EBB8;
  }
  v184 = (_QWORD *)v67;
  sub_2357280(a1, (__int64 *)&v184);
  sub_233F000((__int64 *)&v184);
  sub_23A0D70(a2, (__int64)v132, a3);
  sub_27DC820(&v184, 0xFFFFFFFFLL);
  sub_2354380(v132, (__int64 *)&v184);
  sub_233B480((__int64)&v184, (__int64)&v184, v68, v69, v70, v71);
  sub_234AAB0((__int64)&v184, (__int64 *)v132, *(_BYTE *)(a2 + 32));
  sub_23571D0(a1, (__int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  LOBYTE(v184) = 0;
  v185 = (__m128i)a4;
  v186.m128i_i32[0] = 0;
  sub_23A23F0(a1, (char *)&v184);
  v184 = 0;
  v185 = 0u;
  v186.m128i_i32[0] = 1;
  sub_23A23F0(a1, (char *)&v184);
  if ( (_BYTE)qword_4FDD308 )
  {
    v112 = (_QWORD *)sub_22077B0(0x10u);
    if ( v112 )
      *v112 = &unk_4A0D438;
    v184 = v112;
    sub_23A2230(a1, (unsigned __int64 *)&v184);
    sub_23501E0((__int64 *)&v184);
  }
  memset(v133, 0, 40);
  v72 = (_QWORD *)sub_22077B0(0x10u);
  if ( v72 )
    *v72 = &unk_4A0FDB8;
  v184 = v72;
  sub_23A1F40(v133, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v73 = (_QWORD *)sub_22077B0(0x10u);
  if ( v73 )
    *v73 = &unk_4A0F2F8;
  v184 = v73;
  sub_23A1F40(v133, (unsigned __int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v124[0] = 0x100010000000001LL;
  v124[1] = 0x101000101000001LL;
  v124[2] = 0;
  sub_29744D0(&v184, v124);
  sub_23A1F80(v133, (__int64 *)&v184);
  sub_234AAB0((__int64)&v184, (__int64 *)v133, 0);
  sub_23571D0(a1, (__int64 *)&v184);
  sub_233EFE0((__int64 *)&v184);
  v74 = (_QWORD *)sub_22077B0(0x10u);
  if ( v74 )
    *v74 = &unk_4A0D238;
  v184 = v74;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  sub_23A0BA0((__int64)&v184, 1);
  sub_23A2670(a1, (__int64)&v184);
  sub_233AAF0((__int64)&v184);
  if ( *(_BYTE *)(a2 + 26) )
  {
    v111 = (_QWORD *)sub_22077B0(0x10u);
    if ( v111 )
      *v111 = &unk_4A0D8F8;
    v184 = v111;
    sub_23A2230(a1, (unsigned __int64 *)&v184);
    sub_23501E0((__int64 *)&v184);
  }
  v75 = (_QWORD *)sub_22077B0(0x10u);
  if ( v75 )
    *v75 = &unk_4A0DFF8;
  v184 = v75;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  if ( *(_BYTE *)(a2 + 24) )
  {
    v108 = sub_22077B0(0x10u);
    if ( v108 )
    {
      *(_BYTE *)(v108 + 8) = 1;
      *(_QWORD *)v108 = &unk_4A0E738;
    }
    v184 = (_QWORD *)v108;
    sub_23A2230(a1, (unsigned __int64 *)&v184);
    sub_23501E0((__int64 *)&v184);
  }
  v76 = (_QWORD *)sub_22077B0(0x10u);
  if ( v76 )
    *v76 = &unk_4A0CFB8;
  v184 = v76;
  sub_23A2230(a1, (unsigned __int64 *)&v184);
  sub_23501E0((__int64 *)&v184);
  sub_23A1210(a2, (__int64)a1, a3);
  sub_23A2610(a1);
  sub_233F7F0((__int64)v133);
  sub_2337B30((unsigned __int64 *)&v146);
  sub_233F7F0((__int64)v132);
  sub_233F7F0((__int64)v131);
  v77 = v127;
  v78 = v126;
  if ( v127 != v126 )
  {
    do
    {
      if ( *v78 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v78 + 8LL))(*v78);
      ++v78;
    }
    while ( v77 != v78 );
    v78 = v126;
  }
  if ( v78 )
    j_j___libc_free_0((unsigned __int64)v78);
  sub_233F7F0((__int64)v125);
  return a1;
}
