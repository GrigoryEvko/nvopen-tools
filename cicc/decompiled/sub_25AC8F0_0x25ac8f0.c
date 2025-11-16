// Function: sub_25AC8F0
// Address: 0x25ac8f0
//
__int64 __fastcall sub_25AC8F0(__int64 *a1, __int64 a2)
{
  __m128i v2; // xmm1
  unsigned __int8 **v3; // rbx
  unsigned __int8 **v4; // r12
  __m128i *v5; // rdi
  __int64 (__fastcall *v6)(__int64); // rax
  __int64 v7; // rax
  unsigned __int8 *v8; // rdi
  __int16 *v9; // rbx
  __int64 *v10; // r12
  __int64 *i; // r15
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int16 *v14; // r15
  __m128i *v15; // rdi
  __int64 (__fastcall *v16)(__int64); // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned int v20; // r12d
  unsigned int v21; // r15d
  __int64 v22; // rax
  unsigned __int8 v23; // r8
  __int64 v24; // r14
  bool v25; // si
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // r15
  __int64 v30; // rbx
  __int64 v31; // r12
  __int64 v32; // r13
  __int64 *v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  char v36; // al
  __int64 v37; // rsi
  _BYTE *v38; // r13
  size_t v39; // r12
  _QWORD *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  unsigned __int8 *v43; // rbx
  unsigned __int8 *v44; // rdi
  unsigned __int8 *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // r14
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 v51; // r13
  __int64 *v52; // rax
  unsigned __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r12
  __int64 v57; // rbx
  __int64 v58; // r8
  __int64 v59; // rsi
  int v60; // edi
  __int64 v61; // rax
  unsigned int v62; // edi
  _QWORD *v63; // r13
  __int64 v64; // rbx
  int v65; // eax
  unsigned int v66; // edx
  unsigned __int8 v67; // cl
  int v68; // r12d
  __int64 v69; // r12
  __int64 v70; // rbx
  __int64 v71; // rdx
  unsigned int v72; // esi
  _QWORD *v73; // rax
  __int64 v74; // r12
  __int64 v75; // r13
  __int64 v76; // rbx
  __int64 v77; // rdx
  unsigned int v78; // esi
  __int64 v79; // rax
  __int64 v80; // r13
  _QWORD *v81; // rax
  __int64 v82; // r12
  __int64 v83; // r13
  unsigned __int8 *v84; // rbx
  __int64 v85; // rdx
  unsigned int v86; // esi
  int v87; // r12d
  unsigned __int64 v88; // r12
  _QWORD *v89; // rbx
  __int64 v90; // rdx
  unsigned int v91; // esi
  __int64 v92; // r12
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // r12
  _BYTE *v96; // rax
  __int64 v97; // r13
  _QWORD *v98; // rax
  __int64 v99; // r9
  __int64 v100; // r14
  __int64 v101; // r13
  __int64 v102; // rbx
  __int64 v103; // rdx
  unsigned int v104; // esi
  _QWORD *v106; // rdi
  __int64 *v107; // rax
  __int64 v108; // [rsp-10h] [rbp-460h]
  __int64 *v110; // [rsp+78h] [rbp-3D8h]
  unsigned __int8 *v111; // [rsp+80h] [rbp-3D0h]
  __int64 v112; // [rsp+88h] [rbp-3C8h]
  __int64 v113; // [rsp+90h] [rbp-3C0h]
  __int64 v114; // [rsp+98h] [rbp-3B8h]
  __int64 v115; // [rsp+B0h] [rbp-3A0h]
  __int64 v116; // [rsp+C0h] [rbp-390h]
  __int128 v117; // [rsp+C8h] [rbp-388h]
  __int64 v118; // [rsp+C8h] [rbp-388h]
  __int64 v119; // [rsp+D8h] [rbp-378h]
  __int128 v120; // [rsp+E0h] [rbp-370h]
  __int64 v121; // [rsp+E8h] [rbp-368h]
  int v122; // [rsp+E8h] [rbp-368h]
  __int64 v123; // [rsp+E8h] [rbp-368h]
  int v124; // [rsp+F8h] [rbp-358h]
  __int64 v125; // [rsp+F8h] [rbp-358h]
  __int64 *v126; // [rsp+F8h] [rbp-358h]
  unsigned int v127; // [rsp+108h] [rbp-348h]
  _QWORD v128[2]; // [rsp+110h] [rbp-340h] BYREF
  __int64 *v129; // [rsp+120h] [rbp-330h] BYREF
  __int64 v130; // [rsp+128h] [rbp-328h]
  _BYTE v131[16]; // [rsp+130h] [rbp-320h] BYREF
  _BYTE v132[32]; // [rsp+140h] [rbp-310h] BYREF
  __int16 v133; // [rsp+160h] [rbp-2F0h]
  __int64 v134; // [rsp+170h] [rbp-2E0h] BYREF
  __int64 v135; // [rsp+178h] [rbp-2D8h]
  __int64 v136; // [rsp+180h] [rbp-2D0h]
  __int64 v137; // [rsp+188h] [rbp-2C8h]
  unsigned __int64 *v138; // [rsp+190h] [rbp-2C0h]
  __int64 v139; // [rsp+198h] [rbp-2B8h]
  unsigned __int64 v140[2]; // [rsp+1A0h] [rbp-2B0h] BYREF
  _QWORD v141[2]; // [rsp+1B0h] [rbp-2A0h] BYREF
  __int64 v142; // [rsp+1C0h] [rbp-290h]
  __int64 v143; // [rsp+1C8h] [rbp-288h]
  __int64 v144; // [rsp+1D0h] [rbp-280h]
  __m128i v145; // [rsp+1E0h] [rbp-270h] BYREF
  _OWORD v146[2]; // [rsp+1F0h] [rbp-260h] BYREF
  __int64 v147; // [rsp+210h] [rbp-240h]
  __int64 v148; // [rsp+218h] [rbp-238h]
  __int64 v149; // [rsp+220h] [rbp-230h]
  __int64 v150; // [rsp+228h] [rbp-228h]
  void **v151; // [rsp+230h] [rbp-220h]
  void **v152; // [rsp+238h] [rbp-218h]
  __int64 v153; // [rsp+240h] [rbp-210h]
  int v154; // [rsp+248h] [rbp-208h]
  __int16 v155; // [rsp+24Ch] [rbp-204h]
  char v156; // [rsp+24Eh] [rbp-202h]
  __int64 v157; // [rsp+250h] [rbp-200h]
  __int64 v158; // [rsp+258h] [rbp-1F8h]
  void *v159; // [rsp+260h] [rbp-1F0h] BYREF
  void *v160; // [rsp+268h] [rbp-1E8h] BYREF
  unsigned __int8 *v161; // [rsp+270h] [rbp-1E0h] BYREF
  __int64 v162; // [rsp+278h] [rbp-1D8h]
  _QWORD v163[2]; // [rsp+280h] [rbp-1D0h] BYREF
  char v164; // [rsp+290h] [rbp-1C0h] BYREF
  __int64 v165; // [rsp+2A0h] [rbp-1B0h]
  __int64 v166; // [rsp+2A8h] [rbp-1A8h]
  __int64 v167; // [rsp+2B0h] [rbp-1A0h]
  __int64 v168; // [rsp+2B8h] [rbp-198h]
  void **v169; // [rsp+2C0h] [rbp-190h]
  void **v170; // [rsp+2C8h] [rbp-188h]
  __int64 v171; // [rsp+2D0h] [rbp-180h]
  int v172; // [rsp+2D8h] [rbp-178h]
  __int16 v173; // [rsp+2DCh] [rbp-174h]
  char v174; // [rsp+2DEh] [rbp-172h]
  __int64 v175; // [rsp+2E0h] [rbp-170h]
  __int64 v176; // [rsp+2E8h] [rbp-168h]
  void *v177; // [rsp+2F0h] [rbp-160h] BYREF
  void *v178; // [rsp+2F8h] [rbp-158h] BYREF
  _QWORD *v179; // [rsp+300h] [rbp-150h] BYREF
  __int64 v180; // [rsp+308h] [rbp-148h]
  _QWORD v181[2]; // [rsp+310h] [rbp-140h] BYREF
  __int16 v182; // [rsp+320h] [rbp-130h] BYREF
  __int64 v183; // [rsp+330h] [rbp-120h]
  __int64 v184; // [rsp+338h] [rbp-118h]
  __int64 v185; // [rsp+340h] [rbp-110h]
  __int64 v186; // [rsp+348h] [rbp-108h]
  void **v187; // [rsp+350h] [rbp-100h]
  void **v188; // [rsp+358h] [rbp-F8h]
  __int64 v189; // [rsp+360h] [rbp-F0h]
  int v190; // [rsp+368h] [rbp-E8h]
  __int16 v191; // [rsp+36Ch] [rbp-E4h]
  char v192; // [rsp+36Eh] [rbp-E2h]
  __int64 v193; // [rsp+370h] [rbp-E0h]
  __int64 v194; // [rsp+378h] [rbp-D8h]
  void *v195; // [rsp+380h] [rbp-D0h] BYREF
  void *v196; // [rsp+388h] [rbp-C8h] BYREF
  __m128i v197; // [rsp+390h] [rbp-C0h] BYREF
  __m128i v198; // [rsp+3A0h] [rbp-B0h] BYREF
  __int128 v199; // [rsp+3B0h] [rbp-A0h]
  __int128 v200; // [rsp+3C0h] [rbp-90h]
  __int64 v201; // [rsp+3D0h] [rbp-80h]
  __int64 v202; // [rsp+3D8h] [rbp-78h]
  void **v203; // [rsp+3E0h] [rbp-70h]
  void **v204; // [rsp+3E8h] [rbp-68h]
  __int64 v205; // [rsp+3F0h] [rbp-60h]
  int v206; // [rsp+3F8h] [rbp-58h]
  __int16 v207; // [rsp+3FCh] [rbp-54h]
  char v208; // [rsp+3FEh] [rbp-52h]
  __int64 v209; // [rsp+400h] [rbp-50h]
  __int64 v210; // [rsp+408h] [rbp-48h]
  void *v211; // [rsp+410h] [rbp-40h] BYREF
  void *v212; // [rsp+418h] [rbp-38h] BYREF

  v138 = v140;
  v129 = (__int64 *)v131;
  v130 = 0x200000000LL;
  v119 = a2;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v139 = 0;
  sub_BA9600(&v197, a2);
  v2 = _mm_load_si128(&v198);
  v120 = v199;
  v145 = _mm_load_si128(&v197);
  v146[0] = v2;
  v117 = v200;
  while ( *(_OWORD *)&v145 != v120 || v146[0] != v117 )
  {
    v3 = &v161;
    v163[1] = 0;
    v4 = &v161;
    v5 = &v145;
    v163[0] = sub_25AC5E0;
    v6 = sub_25AC5C0;
    if ( ((unsigned __int8)sub_25AC5C0 & 1) == 0 )
      goto LABEL_5;
    while ( 1 )
    {
      v6 = *(__int64 (__fastcall **)(__int64))((char *)v6 + v5->m128i_i64[0] - 1);
LABEL_5:
      v7 = v6((__int64)v5);
      if ( v7 )
        break;
      while ( 1 )
      {
        v3 += 2;
        if ( &v164 == (char *)v3 )
LABEL_137:
          BUG();
        v8 = v4[3];
        v6 = (__int64 (__fastcall *)(__int64))v4[2];
        v4 = v3;
        v5 = (__m128i *)((char *)&v145 + (_QWORD)v8);
        if ( ((unsigned __int8)v6 & 1) != 0 )
          break;
        v7 = v6((__int64)v5);
        if ( v7 )
          goto LABEL_9;
      }
    }
LABEL_9:
    LODWORD(v130) = 0;
    sub_B91D10(v7, 19, (__int64)&v129);
    v9 = (__int16 *)&v179;
    v10 = &v129[(unsigned int)v130];
    for ( i = v129; v10 != i; ++i )
    {
      v13 = sub_25AC600(*i);
      if ( v13 )
      {
        if ( *(_DWORD *)(v13 + 32) > 0x40u )
          v12 = **(_QWORD **)(v13 + 24);
        else
          v12 = *(_QWORD *)(v13 + 24);
        v179 = (_QWORD *)v12;
        sub_25AC660((__int64)&v134, &v179);
      }
    }
    v14 = (__int16 *)&v179;
    v181[1] = 0;
    v15 = &v145;
    v181[0] = sub_25AC590;
    v16 = sub_25AC560;
    if ( ((unsigned __int8)sub_25AC560 & 1) != 0 )
LABEL_18:
      v16 = *(__int64 (__fastcall **)(__int64))((char *)v16 + v15->m128i_i64[0] - 1);
    while ( !(unsigned __int8)v16((__int64)v15) )
    {
      v9 += 8;
      if ( &v182 == v9 )
        goto LABEL_137;
      v17 = (_QWORD *)*((_QWORD *)v14 + 3);
      v16 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v14 + 2);
      v14 = v9;
      v15 = (__m128i *)((char *)&v145 + (_QWORD)v17);
      if ( ((unsigned __int8)v16 & 1) != 0 )
        goto LABEL_18;
    }
  }
  v18 = sub_BA8DC0(a2, (__int64)"cfi.functions", 13);
  v19 = v18;
  if ( v18 )
  {
    v124 = sub_B91A00(v18);
    if ( v124 )
    {
      v121 = v19;
      v20 = 0;
      while ( 1 )
      {
        v21 = 2;
        v22 = sub_B91A10(v121, v20);
        v23 = *(_BYTE *)(v22 - 16);
        v24 = v22;
        v25 = (v23 & 2) != 0;
        while ( v25 )
        {
          if ( v21 >= *(_DWORD *)(v24 - 24) )
            goto LABEL_38;
          v26 = *(_QWORD *)(v24 - 32);
LABEL_30:
          v27 = sub_25AC600(*(_QWORD *)(v26 + 8LL * v21));
          if ( v27 )
          {
            if ( *(_DWORD *)(v27 + 32) <= 0x40u )
              v28 = *(_QWORD *)(v27 + 24);
            else
              v28 = **(_QWORD **)(v27 + 24);
            v197.m128i_i64[0] = v28;
            sub_25AC660((__int64)&v134, &v197);
            v23 = *(_BYTE *)(v24 - 16);
            v25 = (v23 & 2) != 0;
          }
          ++v21;
        }
        if ( v21 < ((*(_WORD *)(v24 - 16) >> 6) & 0xFu) )
          break;
LABEL_38:
        if ( v124 == ++v20 )
          goto LABEL_39;
      }
      v26 = v24 + -16 - 8LL * ((v23 >> 2) & 0xF);
      goto LABEL_30;
    }
  }
LABEL_39:
  v29 = *(__int64 **)v119;
  v30 = sub_BCE3C0(*(__int64 **)v119, 0);
  v31 = sub_BCE3C0(v29, 0);
  v32 = sub_BCB2E0(v29);
  v33 = (__int64 *)sub_BCB120(v29);
  *(_QWORD *)&v199 = v30;
  v197.m128i_i64[1] = 0x300000003LL;
  v197.m128i_i64[0] = (__int64)&v198;
  v198.m128i_i64[0] = v32;
  v198.m128i_i64[1] = v31;
  v34 = sub_BCF480(v33, &v198, 3, 0);
  sub_BA8C10(v119, (__int64)"__cfi_check", 0xBu, v34, 0);
  v116 = v35;
  if ( (__m128i *)v197.m128i_i64[0] != &v198 )
    _libc_free(v197.m128i_u64[0]);
  sub_B2CA40(v116, 0);
  v36 = *(_BYTE *)(v116 + 32);
  *(_BYTE *)(v116 + 32) = v36 & 0xF0;
  if ( (v36 & 0x30) != 0 )
    *(_BYTE *)(v116 + 33) |= 0x40u;
  v37 = 12;
  sub_B2F770(v116, 0xCu);
  v140[0] = (unsigned __int64)v141;
  v38 = *(_BYTE **)(v119 + 232);
  v39 = *(_QWORD *)(v119 + 240);
  if ( &v38[v39] && !v38 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v197.m128i_i64[0] = *(_QWORD *)(v119 + 240);
  if ( v39 > 0xF )
  {
    v140[0] = sub_22409D0((__int64)v140, (unsigned __int64 *)&v197, 0);
    v106 = (_QWORD *)v140[0];
    v141[0] = v197.m128i_i64[0];
    goto LABEL_130;
  }
  if ( v39 != 1 )
  {
    if ( !v39 )
    {
      v40 = v141;
      goto LABEL_48;
    }
    v106 = v141;
LABEL_130:
    v37 = (__int64)v38;
    memcpy(v106, v38, v39);
    v39 = v197.m128i_i64[0];
    v40 = (_QWORD *)v140[0];
    goto LABEL_48;
  }
  LOBYTE(v141[0]) = *v38;
  v40 = v141;
LABEL_48:
  v41 = v119;
  v140[1] = v39;
  *((_BYTE *)v40 + v39) = 0;
  v142 = *(_QWORD *)(v119 + 264);
  v143 = *(_QWORD *)(v119 + 272);
  v144 = *(_QWORD *)(v119 + 280);
  v42 = (unsigned int)(v142 - 36);
  if ( (unsigned int)v42 <= 1 || (unsigned int)(v142 - 1) <= 1 )
  {
    v37 = (__int64)"target-features";
    sub_B2CD60(v116, "target-features", 0xFu, "+thumb-mode", 0xBu);
  }
  if ( (*(_BYTE *)(v116 + 2) & 1) != 0 )
    sub_B2C6D0(v116, v37, v42, v41);
  v43 = *(unsigned __int8 **)(v116 + 96);
  v197.m128i_i64[0] = (__int64)"CallSiteTypeId";
  v125 = (__int64)v43;
  LOWORD(v199) = 259;
  sub_BD6B50(v43, (const char **)&v197);
  v44 = v43 + 40;
  v45 = v43 + 80;
  v111 = v44;
  v197.m128i_i64[0] = (__int64)"Addr";
  LOWORD(v199) = 259;
  sub_BD6B50(v44, (const char **)&v197);
  v197.m128i_i64[0] = (__int64)"CFICheckFailData";
  LOWORD(v199) = 259;
  sub_BD6B50(v45, (const char **)&v197);
  v197.m128i_i64[0] = (__int64)"entry";
  LOWORD(v199) = 259;
  v46 = sub_22077B0(0x50u);
  v47 = v46;
  if ( v46 )
    sub_AA4D50(v46, (__int64)v29, (__int64)&v197, v116, 0);
  v197.m128i_i64[0] = (__int64)"exit";
  LOWORD(v199) = 259;
  v48 = sub_22077B0(0x50u);
  v112 = v48;
  if ( v48 )
    sub_AA4D50(v48, (__int64)v29, (__int64)&v197, v116, 0);
  v197.m128i_i64[0] = (__int64)"fail";
  LOWORD(v199) = 259;
  v49 = sub_22077B0(0x50u);
  v113 = v49;
  if ( v49 )
    sub_AA4D50(v49, (__int64)v29, (__int64)&v197, v116, 0);
  v150 = sub_AA48A0(v113);
  v151 = &v159;
  v152 = &v160;
  v145.m128i_i64[0] = (__int64)v146;
  v159 = &unk_49DA100;
  LOWORD(v149) = 0;
  v155 = 512;
  v160 = &unk_49DA0B0;
  v145.m128i_i64[1] = 0x200000000LL;
  v147 = v113;
  v153 = 0;
  v154 = 0;
  v156 = 7;
  v157 = 0;
  v158 = 0;
  v148 = v113 + 48;
  v50 = sub_BCE3C0(v29, 0);
  v51 = sub_BCE3C0(v29, 0);
  v52 = (__int64 *)sub_BCB120(v29);
  v198.m128i_i64[0] = v51;
  v197.m128i_i64[1] = 0x200000002LL;
  v198.m128i_i64[1] = v50;
  v197.m128i_i64[0] = (__int64)&v198;
  v53 = sub_BCF480(v52, &v198, 2, 0);
  v54 = sub_BA8C10(v119, (__int64)"__cfi_check_fail", 0x10u, v53, 0);
  v118 = v55;
  v56 = v54;
  if ( (__m128i *)v197.m128i_i64[0] != &v198 )
    _libc_free(v197.m128i_u64[0]);
  v161 = v45;
  v57 = v158;
  v182 = 257;
  v162 = (__int64)v44;
  LOWORD(v199) = 257;
  v58 = v157 + 56 * v158;
  if ( v157 == v58 )
  {
    v122 = 3;
    v62 = 3;
  }
  else
  {
    v59 = v157;
    v60 = 0;
    do
    {
      v61 = *(_QWORD *)(v59 + 40) - *(_QWORD *)(v59 + 32);
      v59 += 56;
      v60 += v61 >> 3;
    }
    while ( v58 != v59 );
    v62 = v60 + 3;
    v122 = v62 & 0x7FFFFFF;
  }
  v115 = v157;
  LOBYTE(v119) = 16 * (_DWORD)v158 != 0;
  v63 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v158) << 32) | v62);
  if ( v63 )
  {
    sub_B44260((__int64)v63, **(_QWORD **)(v56 + 16), 56, ((_DWORD)v119 << 28) | v122, 0, 0);
    v63[9] = 0;
    sub_B4A290((__int64)v63, v56, v118, (__int64 *)&v161, 2, (__int64)&v197, v115, v57);
  }
  if ( (_BYTE)v155 )
  {
    v107 = (__int64 *)sub_BD5C60((__int64)v63);
    v63[9] = sub_A7A090(v63 + 9, v107, -1, 72);
  }
  if ( *(_BYTE *)v63 > 0x1Cu )
  {
    switch ( *(_BYTE *)v63 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_85;
      case 'T':
      case 'U':
      case 'V':
        v64 = v63[1];
        v65 = *(unsigned __int8 *)(v64 + 8);
        v66 = v65 - 17;
        v67 = *(_BYTE *)(v64 + 8);
        if ( (unsigned int)(v65 - 17) <= 1 )
          v67 = *(_BYTE *)(**(_QWORD **)(v64 + 16) + 8LL);
        if ( v67 <= 3u || v67 == 5 || (v67 & 0xFD) == 4 )
          goto LABEL_85;
        if ( (_BYTE)v65 == 15 )
        {
          if ( (*(_BYTE *)(v64 + 9) & 4) == 0 || !sub_BCB420(v63[1]) )
            break;
          v64 = **(_QWORD **)(v64 + 16);
          v65 = *(unsigned __int8 *)(v64 + 8);
          v66 = v65 - 17;
        }
        else if ( (_BYTE)v65 == 16 )
        {
          do
          {
            v64 = *(_QWORD *)(v64 + 24);
            LOBYTE(v65) = *(_BYTE *)(v64 + 8);
          }
          while ( (_BYTE)v65 == 16 );
          v66 = (unsigned __int8)v65 - 17;
        }
        if ( v66 <= 1 )
          LOBYTE(v65) = *(_BYTE *)(**(_QWORD **)(v64 + 16) + 8LL);
        if ( (unsigned __int8)v65 <= 3u || (_BYTE)v65 == 5 || (v65 & 0xFD) == 4 )
        {
LABEL_85:
          v68 = v154;
          if ( v153 )
            sub_B99FD0((__int64)v63, 3u, v153);
          sub_B45150((__int64)v63, v68);
        }
        break;
      default:
        break;
    }
  }
  (*((void (__fastcall **)(void **, _QWORD *, _QWORD **, __int64, __int64))*v152 + 2))(v152, v63, &v179, v148, v149);
  v69 = v145.m128i_i64[0];
  v70 = v145.m128i_i64[0] + 16LL * v145.m128i_u32[2];
  if ( v145.m128i_i64[0] != v70 )
  {
    do
    {
      v71 = *(_QWORD *)(v69 + 8);
      v72 = *(_DWORD *)v69;
      v69 += 16;
      sub_B99FD0((__int64)v63, v72, v71);
    }
    while ( v70 != v69 );
  }
  LOWORD(v199) = 257;
  v73 = sub_BD2C40(72, 1u);
  v74 = (__int64)v73;
  if ( v73 )
    sub_B4C8F0((__int64)v73, v112, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v152 + 2))(v152, v74, &v197, v148, v149);
  v75 = v145.m128i_i64[0];
  v76 = v145.m128i_i64[0] + 16LL * v145.m128i_u32[2];
  if ( v145.m128i_i64[0] != v76 )
  {
    do
    {
      v77 = *(_QWORD *)(v75 + 8);
      v78 = *(_DWORD *)v75;
      v75 += 16;
      sub_B99FD0(v74, v78, v77);
    }
    while ( v76 != v75 );
  }
  v79 = sub_AA48A0(v112);
  v171 = 0;
  v80 = v79;
  v172 = 0;
  v161 = (unsigned __int8 *)v163;
  v162 = 0x200000000LL;
  LOWORD(v167) = 0;
  v169 = &v177;
  v170 = &v178;
  v173 = 512;
  v168 = v79;
  v174 = 7;
  v177 = &unk_49DA100;
  v175 = 0;
  v165 = v112;
  v178 = &unk_49DA0B0;
  v166 = v112 + 48;
  v176 = 0;
  LOWORD(v199) = 257;
  v81 = sub_BD2C40(72, 0);
  v82 = (__int64)v81;
  if ( v81 )
    sub_B4BB80((__int64)v81, v80, 0, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v170 + 2))(v170, v82, &v197, v166, v167);
  v83 = (__int64)v161;
  v84 = &v161[16 * (unsigned int)v162];
  if ( v161 != v84 )
  {
    do
    {
      v85 = *(_QWORD *)(v83 + 8);
      v86 = *(_DWORD *)v83;
      v83 += 16;
      sub_B99FD0(v82, v86, v85);
    }
    while ( v84 != (unsigned __int8 *)v83 );
  }
  v186 = sub_AA48A0(v47);
  v87 = v139;
  v187 = &v195;
  v188 = &v196;
  v179 = v181;
  v195 = &unk_49DA100;
  v180 = 0x200000000LL;
  LOWORD(v185) = 0;
  v196 = &unk_49DA0B0;
  v189 = 0;
  v190 = 0;
  v191 = 512;
  v192 = 7;
  v193 = 0;
  v194 = 0;
  v183 = v47;
  v184 = v47 + 48;
  LOWORD(v199) = 257;
  v114 = sub_BD2DA0(80);
  if ( v114 )
    sub_B53A60(v114, v125, v113, v87, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v188 + 2))(v188, v114, &v197, v184, v185);
  v88 = (unsigned __int64)v179;
  v89 = &v179[2 * (unsigned int)v180];
  if ( v179 != v89 )
  {
    do
    {
      v90 = *(_QWORD *)(v88 + 8);
      v91 = *(_DWORD *)v88;
      v88 += 16LL;
      sub_B99FD0(v114, v91, v90);
    }
    while ( v89 != (_QWORD *)v88 );
  }
  v110 = (__int64 *)&v138[(unsigned int)v139];
  if ( v110 != (__int64 *)v138 )
  {
    v126 = (__int64 *)v138;
    do
    {
      v92 = *v126;
      v93 = sub_BCB2E0(v29);
      v123 = sub_ACD640(v93, v92, 0);
      v197.m128i_i64[0] = (__int64)"test";
      LOWORD(v199) = 259;
      v94 = sub_22077B0(0x50u);
      v95 = v94;
      if ( v94 )
        sub_AA4D50(v94, (__int64)v29, (__int64)&v197, v116, 0);
      v202 = sub_AA48A0(v95);
      v197.m128i_i64[0] = (__int64)&v198;
      v203 = &v211;
      v133 = 257;
      v204 = &v212;
      v197.m128i_i64[1] = 0x200000000LL;
      v211 = &unk_49DA100;
      LOWORD(v201) = 0;
      v212 = &unk_49DA0B0;
      *((_QWORD *)&v200 + 1) = v95 + 48;
      v207 = 512;
      v205 = 0;
      v206 = 0;
      v208 = 7;
      v209 = 0;
      v210 = 0;
      *(_QWORD *)&v200 = v95;
      v128[0] = v111;
      v96 = sub_B98A20(v123, 512);
      v128[1] = sub_B9F6F0(v29, v96);
      v97 = sub_B33D10((__int64)&v197, 0x166u, 0, 0, (int)v128, 2, v127, (__int64)v132);
      v133 = 257;
      v98 = sub_BD2C40(72, 3u);
      v99 = v108;
      v100 = (__int64)v98;
      if ( v98 )
        sub_B4C9A0((__int64)v98, v112, v113, v97, 3u, v108, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE *, _QWORD, __int64, __int64))*v204 + 2))(
        v204,
        v100,
        v132,
        *((_QWORD *)&v200 + 1),
        v201,
        v99);
      v101 = v197.m128i_i64[0];
      v102 = v197.m128i_i64[0] + 16LL * v197.m128i_u32[2];
      if ( v197.m128i_i64[0] != v102 )
      {
        do
        {
          v103 = *(_QWORD *)(v101 + 8);
          v104 = *(_DWORD *)v101;
          v101 += 16;
          sub_B99FD0(v100, v104, v103);
        }
        while ( v102 != v101 );
      }
      sub_B99FD0(v100, 2u, *a1);
      sub_B53E30(v114, v123, v95);
      nullsub_61();
      v211 = &unk_49DA100;
      nullsub_63();
      if ( (__m128i *)v197.m128i_i64[0] != &v198 )
        _libc_free(v197.m128i_u64[0]);
      ++v126;
    }
    while ( v110 != v126 );
  }
  nullsub_61();
  v195 = &unk_49DA100;
  nullsub_63();
  if ( v179 != v181 )
    _libc_free((unsigned __int64)v179);
  nullsub_61();
  v177 = &unk_49DA100;
  nullsub_63();
  if ( v161 != (unsigned __int8 *)v163 )
    _libc_free((unsigned __int64)v161);
  nullsub_61();
  v159 = &unk_49DA100;
  nullsub_63();
  if ( (_OWORD *)v145.m128i_i64[0] != v146 )
    _libc_free(v145.m128i_u64[0]);
  if ( (_QWORD *)v140[0] != v141 )
    j_j___libc_free_0(v140[0]);
  if ( v129 != (__int64 *)v131 )
    _libc_free((unsigned __int64)v129);
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
  return sub_C7D6A0(v135, 8LL * (unsigned int)v137, 8);
}
