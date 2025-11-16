// Function: sub_2AB5570
// Address: 0x2ab5570
//
__int64 __fastcall sub_2AB5570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rcx
  int v13; // r8d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r11
  __int64 *v17; // rsi
  bool v18; // zf
  __int64 **v19; // rdi
  unsigned int v20; // edx
  __int64 **v21; // rax
  __int64 *v22; // r10
  __int64 v23; // rax
  int v24; // r9d
  __int64 v25; // rax
  int v26; // r9d
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // r11
  __int64 v31; // rax
  __int64 *v32; // rdi
  __int64 v33; // r14
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 *v37; // r11
  char v38; // al
  __int64 v39; // rdi
  char v40; // al
  int v42; // eax
  _BYTE *v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rdi
  __int64 *v46; // r11
  int v47; // eax
  unsigned int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rax
  int v52; // edx
  int v53; // edx
  __int64 v54; // rsi
  unsigned int v55; // r10d
  int v56; // eax
  int v57; // eax
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 *v60; // r10
  __int64 v61; // rcx
  int v62; // edi
  unsigned int v63; // eax
  __int64 **v64; // rax
  __int64 *v65; // rdi
  __int64 v66; // rdx
  int v67; // esi
  __int64 v68; // rax
  int v69; // edx
  signed __int64 v70; // rax
  __int64 **v71; // r14
  __int64 v72; // rcx
  __int64 *v73; // rax
  __int64 *v74; // rdi
  __int64 **v75; // rdx
  __int64 v76; // rcx
  int v77; // edx
  int v78; // edx
  signed __int64 v79; // rax
  bool v80; // al
  int v81; // r9d
  _BYTE *v82; // rsi
  int v83; // eax
  __int64 v84; // rax
  __int64 v85; // rdi
  char v86; // al
  char v87; // al
  __int64 v88; // rax
  __int64 **v89; // rax
  __int64 *v90; // rdi
  __m128i v91; // rax
  __int64 v92; // rdi
  unsigned int v93; // ecx
  __int64 **v94; // rax
  __int64 *v95; // rdi
  int v96; // edx
  int v97; // esi
  __int64 v98; // rax
  int v99; // edx
  __int64 v100; // r14
  int v101; // edx
  __m128i v102; // xmm0
  signed __int64 v103; // r8
  int v104; // edx
  unsigned __int64 v105; // rax
  bool v106; // of
  unsigned __int64 v107; // rax
  signed __int64 v108; // rax
  bool v109; // al
  _BYTE *v110; // rsi
  int v111; // r9d
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // rax
  __int64 *v115; // r11
  __int64 **v116; // rax
  __int64 *v117; // rdi
  __int64 v118; // rax
  __int64 v119; // rax
  int v120; // edx
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rdi
  unsigned int v124; // ecx
  int v125; // edx
  __int64 v126; // rax
  int v127; // edx
  int v128; // edx
  int v129; // esi
  __int64 v130; // rcx
  int v131; // edx
  unsigned __int64 v132; // rax
  unsigned __int64 v133; // rax
  signed __int64 v134; // rax
  int v135; // edi
  signed __int64 v136; // rcx
  __int64 v137; // rax
  bool v138; // cc
  unsigned __int64 v139; // rax
  __int64 *v140; // rdx
  __int64 *v141; // r10
  __int64 v142; // rax
  __int64 v143; // rax
  int v144; // edx
  __int64 v145; // [rsp+8h] [rbp-108h]
  __int64 *v146; // [rsp+10h] [rbp-100h]
  __int64 *v147; // [rsp+18h] [rbp-F8h]
  __int64 *v148; // [rsp+18h] [rbp-F8h]
  __int64 v149; // [rsp+20h] [rbp-F0h]
  int v150; // [rsp+20h] [rbp-F0h]
  int v151; // [rsp+28h] [rbp-E8h]
  __int64 *v152; // [rsp+30h] [rbp-E0h]
  int v153; // [rsp+30h] [rbp-E0h]
  __int64 *v154; // [rsp+38h] [rbp-D8h]
  __int64 **v155; // [rsp+38h] [rbp-D8h]
  int v156; // [rsp+38h] [rbp-D8h]
  __int64 v157; // [rsp+38h] [rbp-D8h]
  __int64 *v158; // [rsp+40h] [rbp-D0h]
  int v159; // [rsp+40h] [rbp-D0h]
  unsigned __int8 v160; // [rsp+40h] [rbp-D0h]
  int v161; // [rsp+40h] [rbp-D0h]
  __int64 **v162; // [rsp+48h] [rbp-C8h]
  unsigned __int8 v163; // [rsp+48h] [rbp-C8h]
  __int64 v164; // [rsp+48h] [rbp-C8h]
  __m128i v165; // [rsp+50h] [rbp-C0h] BYREF
  __int64 **v166; // [rsp+60h] [rbp-B0h]
  __int64 *v167; // [rsp+68h] [rbp-A8h]
  __int64 *v168; // [rsp+70h] [rbp-A0h]
  __int64 v169; // [rsp+78h] [rbp-98h]
  __int64 v170; // [rsp+80h] [rbp-90h]
  __int64 v171; // [rsp+88h] [rbp-88h]
  __int64 **v172; // [rsp+98h] [rbp-78h] BYREF
  __int64 **v173; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v174; // [rsp+A8h] [rbp-68h]
  __m128i v175; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v176; // [rsp+C0h] [rbp-50h] BYREF
  __int64 ***v177; // [rsp+C8h] [rbp-48h]
  _QWORD *v178[8]; // [rsp+D0h] [rbp-40h] BYREF

  v171 = a4;
  if ( *(_DWORD *)(a2 + 280) == *(_DWORD *)(a2 + 276)
    || !BYTE4(v171) && (_DWORD)v171 == 1
    || (unsigned int)*(unsigned __int8 *)(a5 + 8) - 17 > 1 )
  {
    goto LABEL_29;
  }
  if ( (unsigned __int8)(*(_BYTE *)a3 - 68) > 1u )
  {
    v9 = a3;
  }
  else
  {
    if ( !(unsigned __int8)sub_BD36B0(a3) )
      goto LABEL_29;
    v9 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL);
  }
  v10 = *(_QWORD *)(v9 + 16);
  if ( v10 )
  {
    if ( !*(_QWORD *)(v10 + 8) && *(_BYTE *)v9 == 46 )
    {
      v43 = *(_BYTE **)(v10 + 24);
      if ( *v43 == 42 )
        v9 = (__int64)v43;
    }
  }
  v11 = *(unsigned int *)(a2 + 344);
  v12 = *(_QWORD *)(a2 + 328);
  if ( !(_DWORD)v11 )
    goto LABEL_29;
  v13 = v11 - 1;
  v14 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v9 != *v15 )
  {
    v54 = *v15;
    v55 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v56 = 1;
    while ( v54 != -4096 )
    {
      v55 = v13 & (v56 + v55);
      v54 = *(_QWORD *)(v12 + 16LL * v55);
      if ( v9 == v54 )
      {
        v57 = 1;
        while ( v16 != -4096 )
        {
          v111 = v57 + 1;
          v14 = v13 & (v57 + v14);
          v15 = (__int64 *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( v54 == *v15 )
            goto LABEL_11;
          v57 = v111;
        }
        v15 = (__int64 *)(v12 + 16LL * (unsigned int)v11);
        goto LABEL_11;
      }
      ++v56;
    }
    goto LABEL_29;
  }
LABEL_11:
  v17 = (__int64 *)v15[1];
  v18 = *(_BYTE *)v17 == 84;
  v167 = v17;
  if ( v18 )
  {
    v17 = v167;
  }
  else
  {
    v19 = (__int64 **)(v12 + 16 * v11);
    do
    {
      v20 = v13 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v21 = (__int64 **)(v12 + 16LL * v20);
      v22 = *v21;
      if ( *v21 != v17 )
      {
        v42 = 1;
        while ( v22 != (__int64 *)-4096LL )
        {
          v81 = v42 + 1;
          v20 = v13 & (v42 + v20);
          v21 = (__int64 **)(v12 + 16LL * v20);
          v22 = *v21;
          if ( *v21 == v17 )
            goto LABEL_14;
          v42 = v81;
        }
        v21 = v19;
      }
LABEL_14:
      v17 = v21[1];
    }
    while ( *(_BYTE *)v17 != 84 );
  }
  v23 = sub_2AA8FC0(*(_QWORD *)(a2 + 440) + 80LL, (__int64)v17);
  v24 = *(_DWORD *)(v23 + 48);
  v170 = v23;
  if ( (unsigned int)(v24 - 6) <= 3 || (unsigned int)(v24 - 12) <= 3 )
  {
    LODWORD(v166) = v24;
    sub_F6F040(v24);
    v25 = sub_DFDC70(*(_QWORD *)(a2 + 448));
    v26 = (int)v166;
    v169 = v25;
    LODWORD(v168) = v27;
  }
  else
  {
    v46 = *(__int64 **)(a2 + 448);
    LODWORD(v166) = v24;
    v47 = *(_DWORD *)(v170 + 52);
    v168 = v46;
    LODWORD(v176) = v47;
    BYTE4(v176) = 1;
    v48 = sub_1022EF0(v24);
    v49 = sub_DFDC10(v168, v48, a5, v176);
    v26 = (int)v166;
    v169 = v49;
    LODWORD(v168) = v50;
  }
  if ( v26 == 16 )
  {
    v51 = sub_DFD800(*(_QWORD *)(a2 + 448), 0x12u, a5, *(_DWORD *)(a2 + 992), 0, 0, 0, 0, 0, 0);
    v18 = v52 == 1;
    v53 = 1;
    if ( !v18 )
      v53 = (int)v168;
    LODWORD(v168) = v53;
    if ( __OFADD__(v51, v169) )
    {
      v138 = v51 <= 0;
      v139 = 0x8000000000000000LL;
      if ( !v138 )
        v139 = 0x7FFFFFFFFFFFFFFFLL;
      v169 = v139;
    }
    else
    {
      v169 += v51;
    }
  }
  if ( !(unsigned __int8)sub_31A4BE0(*(_QWORD *)(a2 + 496)) && *(_BYTE *)(v170 + 73) )
    goto LABEL_20;
  v29 = sub_986520(v9);
  v30 = *(__int64 **)(v29 + 32);
  if ( v167 != v30 )
  {
    if ( *(_BYTE *)v30 > 0x1Cu )
      goto LABEL_23;
LABEL_41:
    v44 = *(_QWORD *)sub_986520(a3);
    BYTE4(v174) = *(_BYTE *)(a5 + 8) == 18;
    v45 = *(__int64 **)(v44 + 8);
    LODWORD(v174) = *(_DWORD *)(a5 + 32);
    sub_BCE1B0(v45, v174);
    goto LABEL_28;
  }
  v30 = *(__int64 **)v29;
  if ( **(_BYTE **)v29 <= 0x1Cu )
    goto LABEL_41;
LABEL_23:
  v167 = v30;
  v31 = *(_QWORD *)sub_986520(a3);
  BYTE4(v174) = *(_BYTE *)(a5 + 8) == 18;
  v32 = *(__int64 **)(v31 + 8);
  LODWORD(v174) = *(_DWORD *)(a5 + 32);
  v33 = sub_BCE1B0(v32, v174);
  v34 = sub_1022EF0(*(_DWORD *)(v170 + 48));
  v37 = v167;
  if ( v34 != 13 )
  {
LABEL_24:
    v38 = *(_BYTE *)v37;
    goto LABEL_25;
  }
  v35 = (__int64)&v172;
  v176 = (__int64)&v172;
  v177 = &v173;
  v178[0] = &v172;
  v178[1] = &v173;
  v38 = *(_BYTE *)v167;
  if ( *(_BYTE *)v167 != 68 )
    goto LABEL_106;
  v82 = (_BYTE *)*(v167 - 4);
  if ( *v82 == 46 )
  {
    if ( (unsigned __int8)sub_2AA7F60((_QWORD **)&v176, (__int64)v82) )
      goto LABEL_83;
    v38 = *(_BYTE *)v37;
LABEL_106:
    if ( v38 != 69 )
      goto LABEL_25;
    v110 = (_BYTE *)*(v37 - 4);
    if ( *v110 != 46 )
      goto LABEL_25;
    if ( !(unsigned __int8)sub_2AA7F60(v178, (__int64)v110) )
      goto LABEL_24;
LABEL_83:
    v83 = *(unsigned __int8 *)v172;
    v35 = (unsigned int)(v83 - 68);
    if ( (unsigned __int8)(v83 - 68) > 1u )
      goto LABEL_24;
    if ( (_BYTE)v83 != *(_BYTE *)v173 )
      goto LABEL_24;
    v165.m128i_i64[0] = (__int64)v37;
    v166 = v172;
    v162 = v173;
    v167 = (__int64 *)sub_986520((__int64)v172);
    v84 = sub_986520((__int64)v162);
    v37 = (__int64 *)v165.m128i_i64[0];
    v35 = *v167;
    if ( *(_QWORD *)(*v167 + 8) != *(_QWORD *)(*(_QWORD *)v84 + 8LL) )
      goto LABEL_24;
    v85 = *(_QWORD *)(a2 + 416);
    v167 = (__int64 *)v165.m128i_i64[0];
    v86 = sub_D48480(v85, (__int64)v166, v35, v36);
    v37 = (__int64 *)v165.m128i_i64[0];
    if ( v86 )
      goto LABEL_24;
    v87 = sub_D48480(*(_QWORD *)(a2 + 416), (__int64)v173, v35, v36);
    v37 = v167;
    if ( v87 || *(_BYTE *)v172 != *(_BYTE *)v167 && v172 != v173 )
      goto LABEL_24;
    v152 = v167;
    v163 = *(_BYTE *)v172 == 68;
    v88 = sub_986520((__int64)v172);
    v167 = (__int64 *)sub_2AAE030(*(__int64 **)(*(_QWORD *)v88 + 8LL), v33);
    v89 = (__int64 **)sub_2AAE030(v172[1], v33);
    v90 = *(__int64 **)(a2 + 448);
    v166 = v89;
    v91.m128i_i64[0] = sub_DFD060(v90, (unsigned int)*(unsigned __int8 *)v172 - 29, (__int64)v89, (__int64)v167);
    v92 = *(_QWORD *)(a2 + 448);
    v93 = *(_DWORD *)(a2 + 992);
    v155 = v166;
    v165 = v91;
    v94 = (__int64 **)sub_DFD800(v92, 0x11u, (__int64)v166, v93, 0, 0, 0, 0, 0, 0);
    v95 = *(__int64 **)(a2 + 448);
    v159 = v96;
    v97 = *(unsigned __int8 *)v152;
    v166 = v94;
    v98 = sub_DFD060(v95, (unsigned int)(v97 - 29), v33, (__int64)v155);
    v156 = v99;
    v100 = v98;
    sub_DFDCF0(*(__int64 **)(a2 + 448), v163);
    if ( v101 )
      goto LABEL_28;
    v102 = _mm_load_si128(&v165);
    v176 = 2;
    LODWORD(v177) = 0;
    v175 = v102;
    sub_2AA9150((__int64)&v175, (__int64)&v176);
    v104 = 1;
    if ( v159 != 1 )
      v104 = v175.m128i_i32[2];
    v105 = (unsigned __int64)v166 + v175.m128i_i64[0];
    if ( __OFADD__(v175.m128i_i64[0], v166) )
    {
      v105 = 0x7FFFFFFFFFFFFFFFLL;
      if ( (__int64)v166 <= 0 )
        v105 = 0x8000000000000000LL;
    }
    if ( v156 == 1 )
      v104 = 1;
    v106 = __OFADD__(v100, v105);
    v107 = v100 + v105;
    if ( v106 )
    {
      v107 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v100 <= 0 )
        v107 = 0x8000000000000000LL;
    }
    if ( (_DWORD)v168 == 1 )
    {
      v104 = 1;
    }
    else
    {
      v106 = __OFADD__(v169, v107);
      v108 = v169 + v107;
      if ( v106 )
      {
        v108 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v169 <= 0 )
          v108 = 0x8000000000000000LL;
      }
      if ( !v104 )
      {
        v109 = v103 < v108;
LABEL_101:
        if ( !v109 )
          goto LABEL_28;
        *(_DWORD *)(a1 + 8) = 0;
        if ( a3 != v9 )
          v103 = 0;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v103;
        return a1;
      }
    }
    v109 = v104 > 0;
    goto LABEL_101;
  }
LABEL_25:
  if ( (unsigned __int8)(v38 - 68) > 1u
    || (v39 = *(_QWORD *)(a2 + 416), v167 = v37, v40 = sub_D48480(v39, (__int64)v37, v35, v36), v37 = v167, v40) )
  {
    v167 = v37;
    if ( (unsigned int)sub_1022EF0(*(_DWORD *)(v170 + 48)) != 13 )
      goto LABEL_28;
    v176 = (__int64)&v172;
    v177 = &v173;
    if ( *(_BYTE *)v167 != 46 || !(unsigned __int8)sub_2AA7F60((_QWORD **)&v176, (__int64)v167) )
      goto LABEL_28;
    if ( (unsigned __int8)(*(_BYTE *)v172 - 68) <= 1u
      && *(_BYTE *)v172 == *(_BYTE *)v173
      && !(unsigned __int8)sub_D48480(*(_QWORD *)(a2 + 416), (__int64)v172, (__int64)v173, v72)
      && !(unsigned __int8)sub_D48480(*(_QWORD *)(a2 + 416), (__int64)v173, v112, v113) )
    {
      v160 = *(_BYTE *)v172 == 68;
      v167 = *(__int64 **)(*(_QWORD *)sub_986520((__int64)v172) + 8LL);
      v114 = sub_986520((__int64)v173);
      v115 = v167;
      v149 = *(_QWORD *)(*(_QWORD *)v114 + 8LL);
      if ( *((_DWORD *)v167 + 2) >> 8 < *(_DWORD *)(v149 + 8) >> 8 )
        v115 = *(__int64 **)(*(_QWORD *)v114 + 8LL);
      v146 = v115;
      v116 = (__int64 **)sub_2AAE030(v115, v33);
      v117 = *(__int64 **)(a2 + 448);
      v166 = v116;
      v165.m128i_i64[0] = (__int64)v117;
      v118 = sub_2AAE030(v167, v33);
      v119 = sub_DFD060(v117, (unsigned int)*(unsigned __int8 *)v172 - 29, v33, v118);
      v153 = v120;
      v165.m128i_i64[0] = *(_QWORD *)(a2 + 448);
      v147 = (__int64 *)v149;
      v157 = v119;
      v121 = sub_2AAE030((__int64 *)v149, v33);
      v122 = sub_DFD060((__int64 *)v165.m128i_i64[0], (unsigned int)*(unsigned __int8 *)v173 - 29, v33, v121);
      v123 = *(_QWORD *)(a2 + 448);
      v124 = *(_DWORD *)(a2 + 992);
      v151 = v125;
      v165.m128i_i64[0] = v122;
      v126 = sub_DFD800(v123, 0x11u, v33, v124, 0, 0, 0, 0, 0, 0);
      v150 = v127;
      v164 = v126;
      v170 = sub_DFDCF0(*(__int64 **)(a2 + 448), v160);
      v161 = v128;
      if ( v167 != v146 )
        goto LABEL_178;
      if ( v147 == v146 )
      {
        v129 = 0;
        v130 = 0;
        goto LABEL_121;
      }
      if ( v167 == v146 )
        v140 = (__int64 *)v173;
      else
LABEL_178:
        v140 = (__int64 *)v172;
      v141 = *(__int64 **)(a2 + 448);
      v167 = v140;
      v148 = v141;
      v142 = sub_986520((__int64)v140);
      v145 = sub_2AAE030(*(__int64 **)(*(_QWORD *)v142 + 8LL), v33);
      v143 = sub_DFD060(v148, (unsigned int)*(unsigned __int8 *)v167 - 29, (__int64)v166, v145);
      v129 = v144;
      v130 = v143;
LABEL_121:
      if ( !v161 )
      {
        v131 = 1;
        if ( v151 != 1 )
          v131 = v153;
        v132 = v165.m128i_i64[0] + v157;
        if ( __OFADD__(v165.m128i_i64[0], v157) )
        {
          v132 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v165.m128i_i64[0] <= 0 )
            v132 = 0x8000000000000000LL;
        }
        if ( v150 == 1 )
          v131 = 1;
        v106 = __OFADD__(v164, v132);
        v133 = v164 + v132;
        if ( v106 )
        {
          v133 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v164 <= 0 )
            v133 = 0x8000000000000000LL;
        }
        if ( (_DWORD)v168 == 1 )
          v131 = 1;
        v106 = __OFADD__(v169, v133);
        v134 = v169 + v133;
        if ( v106 )
        {
          v134 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v169 <= 0 )
            v134 = 0x8000000000000000LL;
        }
        v135 = v129 == 1;
        if ( !__OFADD__(v130, v170) )
        {
          v136 = v130 + v170;
          goto LABEL_133;
        }
        if ( v130 <= 0 )
        {
          v136 = 0x8000000000000000LL;
LABEL_133:
          if ( v131 == v135 )
          {
            if ( v136 < v134 )
            {
LABEL_135:
              v137 = 0;
              if ( a3 == v9 )
                v137 = v170;
              *(_DWORD *)(a1 + 8) = 0;
              *(_BYTE *)(a1 + 16) = 1;
              *(_QWORD *)a1 = v137;
              return a1;
            }
            goto LABEL_28;
          }
          goto LABEL_157;
        }
        if ( v131 != v135 )
        {
LABEL_157:
          if ( v135 < v131 )
            goto LABEL_135;
        }
      }
LABEL_28:
      if ( a3 == v9 )
      {
LABEL_20:
        v28 = v169;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v28;
        *(_DWORD *)(a1 + 8) = (_DWORD)v168;
        return a1;
      }
LABEL_29:
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
    if ( (unsigned __int8)(*(_BYTE *)a3 - 68) <= 1u )
      goto LABEL_28;
    v73 = (__int64 *)sub_DFD800(*(_QWORD *)(a2 + 448), 0x11u, v33, *(_DWORD *)(a2 + 992), 0, 0, 0, 0, 0, 0);
    v74 = *(__int64 **)(a2 + 448);
    v166 = v75;
    v167 = v73;
    v76 = sub_DFDCF0(v74, 1u);
    if ( v77 )
      goto LABEL_28;
    v78 = (int)v166;
    if ( (_DWORD)v168 == 1 )
    {
      v78 = 1;
    }
    else
    {
      v79 = (signed __int64)v167 + v169;
      if ( __OFADD__(v169, v167) )
      {
        v79 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v169 <= 0 )
          v79 = 0x8000000000000000LL;
      }
      if ( !(_DWORD)v166 )
      {
        v80 = v76 < v79;
LABEL_74:
        if ( !v80 )
          goto LABEL_28;
        *(_DWORD *)(a1 + 8) = 0;
        if ( a3 != v9 )
          v76 = 0;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v76;
        return a1;
      }
    }
    v80 = v78 > 0;
    goto LABEL_74;
  }
  v154 = v167;
  LOBYTE(v166) = *(_BYTE *)v167 == 68;
  v58 = sub_986520((__int64)v167);
  v59 = (__int64 *)sub_2AAE030(*(__int64 **)(*(_QWORD *)v58 + 8LL), v33);
  v60 = *(__int64 **)(a2 + 448);
  v167 = v59;
  v61 = *(_QWORD *)(v170 + 64);
  v158 = v60;
  v62 = *(_DWORD *)(v170 + 48);
  v165.m128i_i32[0] = *(_DWORD *)(v170 + 52);
  v170 = v61;
  v63 = sub_1022EF0(v62);
  v64 = (__int64 **)sub_DFDCC0(v158, v63, (unsigned __int8)v166, v170);
  v65 = *(__int64 **)(a2 + 448);
  v170 = v66;
  v67 = *(unsigned __int8 *)v154;
  v166 = v64;
  v68 = sub_DFD060(v65, (unsigned int)(v67 - 29), v33, (__int64)v167);
  if ( (_DWORD)v170 )
    goto LABEL_28;
  if ( v69 != 1 )
  {
    if ( __OFADD__(v68, v169) )
    {
      v138 = v68 <= 0;
      v70 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v138 )
        v70 = 0x8000000000000000LL;
    }
    else
    {
      v70 = v68 + v169;
    }
    if ( (_DWORD)v168 )
    {
      if ( (int)v168 <= 0 )
        goto LABEL_28;
    }
    else if ( (__int64)v166 >= v70 )
    {
      goto LABEL_28;
    }
  }
  v71 = v166;
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 1;
  if ( a3 != v9 )
    v71 = 0;
  *(_QWORD *)a1 = v71;
  return a1;
}
