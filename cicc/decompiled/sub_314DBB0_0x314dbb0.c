// Function: sub_314DBB0
// Address: 0x314dbb0
//
unsigned __int64 *__fastcall sub_314DBB0(unsigned __int64 *a1, unsigned __int32 a2, _BYTE *a3, __int32 a4, char a5)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  int *v18; // rax
  int v19; // eax
  int *v20; // rax
  int v21; // eax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  _BYTE *v29; // rax
  unsigned __int64 v30; // r8
  __int64 v31; // rbx
  __int64 v32; // r12
  unsigned __int64 *v33; // rsi
  __int64 v34; // rbx
  _QWORD *v35; // r12
  int *v36; // rax
  int v37; // eax
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  _QWORD *v44; // rax
  __m128i *v45; // rax
  _QWORD *v46; // rax
  unsigned __int64 v47; // rbx
  __int16 v48; // bx
  __int64 v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rax
  _QWORD *v52; // rax
  _QWORD *v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rax
  _QWORD *v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  _QWORD *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  _QWORD *v80; // rax
  __int64 v81; // r9
  unsigned __int8 v82; // al
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  _QWORD *v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  _QWORD *v104; // rax
  _QWORD *v105; // rax
  _QWORD *v106; // rax
  _QWORD *v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  _QWORD *v112; // rax
  _QWORD *v113; // rax
  _QWORD *v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  __int64 v118; // r9
  _QWORD *v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // r8
  __int64 v123; // r9
  _QWORD *v124; // rax
  _QWORD *v125; // rax
  _QWORD *v126; // rax
  __int64 v127; // rax
  char v128; // bl
  __int64 v129; // rax
  _QWORD *v130; // rax
  _QWORD *v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rcx
  __int64 v134; // r8
  __int64 v135; // r9
  _QWORD *v136; // rax
  __int64 v137; // rdx
  __int64 v138; // rcx
  __int64 v139; // r8
  __int64 v140; // r9
  __int64 v142; // rdx
  __int64 v143; // rcx
  __int64 v144; // r8
  __int64 v145; // r9
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 v148; // r8
  __int64 v149; // r9
  _QWORD *v150; // rax
  __int64 v151; // rdx
  __int64 v152; // rcx
  __int64 v153; // r8
  __int64 v154; // r9
  __int64 v155; // r9
  __m128i v156; // xmm5
  __m128i v157; // xmm6
  __m128i v158; // xmm7
  __m128i v159; // xmm0
  __m128i v160; // xmm1
  __int64 v161; // rax
  __m128i v162; // xmm2
  __m128i v163; // xmm3
  __m128i v164; // xmm4
  __m128i v165; // xmm5
  __m128i v166; // xmm6
  int v167; // edx
  __int64 v168; // rdx
  __int64 v169; // rdx
  __int64 v170; // rdx
  __int64 v171; // rdx
  __int64 v172; // rbx
  _QWORD *v173; // r13
  __int64 v174; // rbx
  _QWORD *v175; // r13
  __int64 v176; // rbx
  _QWORD *v177; // r13
  _DWORD *v178; // rax
  int v179; // eax
  _QWORD *v180; // rax
  _QWORD *v181; // rax
  _QWORD *v182; // rax
  _QWORD *v183; // rax
  __int64 v184; // rax
  char v185; // bl
  __int64 v186; // rax
  _QWORD *v187; // rax
  __int64 v188; // rax
  __int64 v189; // rdx
  __int64 v190; // rcx
  __int64 v191; // r8
  __int64 v192; // r9
  char v196; // [rsp+98h] [rbp-E48h]
  __int64 v197; // [rsp+A8h] [rbp-E38h]
  int v198; // [rsp+B0h] [rbp-E30h]
  __int64 v199; // [rsp+B4h] [rbp-E2Ch]
  int v200; // [rsp+BCh] [rbp-E24h]
  __int64 v201; // [rsp+C0h] [rbp-E20h]
  int v202; // [rsp+C8h] [rbp-E18h]
  __int64 v203; // [rsp+CCh] [rbp-E14h]
  int v204; // [rsp+D4h] [rbp-E0Ch]
  __int64 v205; // [rsp+D8h] [rbp-E08h]
  int v206; // [rsp+E0h] [rbp-E00h]
  __int64 v207; // [rsp+E4h] [rbp-DFCh]
  int v208; // [rsp+ECh] [rbp-DF4h]
  __int64 v209; // [rsp+F0h] [rbp-DF0h]
  int v210; // [rsp+F8h] [rbp-DE8h]
  __int64 v211; // [rsp+FCh] [rbp-DE4h]
  int v212; // [rsp+104h] [rbp-DDCh]
  __int64 v213; // [rsp+108h] [rbp-DD8h]
  int v214; // [rsp+110h] [rbp-DD0h]
  __int64 v215; // [rsp+114h] [rbp-DCCh]
  int v216; // [rsp+11Ch] [rbp-DC4h]
  __int64 v217; // [rsp+120h] [rbp-DC0h]
  int v218; // [rsp+128h] [rbp-DB8h]
  __int64 v219; // [rsp+12Ch] [rbp-DB4h]
  int v220; // [rsp+134h] [rbp-DACh]
  __int64 v221; // [rsp+138h] [rbp-DA8h]
  int v222; // [rsp+140h] [rbp-DA0h]
  __int64 v223; // [rsp+144h] [rbp-D9Ch]
  int v224; // [rsp+14Ch] [rbp-D94h]
  __m128i v225; // [rsp+150h] [rbp-D90h] BYREF
  __int64 v226; // [rsp+160h] [rbp-D80h]
  unsigned __int64 v227[6]; // [rsp+170h] [rbp-D70h] BYREF
  unsigned __int64 v228[6]; // [rsp+1A0h] [rbp-D40h] BYREF
  __m128i v229; // [rsp+1D0h] [rbp-D10h] BYREF
  __int64 v230; // [rsp+1E0h] [rbp-D00h]
  __int64 v231; // [rsp+1E8h] [rbp-CF8h]
  __int64 v232; // [rsp+1F0h] [rbp-CF0h]
  __int64 v233; // [rsp+200h] [rbp-CE0h] BYREF
  __int64 v234; // [rsp+208h] [rbp-CD8h]
  __int64 v235; // [rsp+210h] [rbp-CD0h]
  __int64 v236; // [rsp+218h] [rbp-CC8h]
  __int64 v237; // [rsp+220h] [rbp-CC0h]
  unsigned __int64 v238; // [rsp+230h] [rbp-CB0h] BYREF
  __int64 v239; // [rsp+238h] [rbp-CA8h]
  __int64 v240; // [rsp+240h] [rbp-CA0h]
  __int64 v241; // [rsp+248h] [rbp-C98h]
  __int64 v242; // [rsp+250h] [rbp-C90h]
  __m128i v243; // [rsp+260h] [rbp-C80h] BYREF
  __m128i v244; // [rsp+270h] [rbp-C70h] BYREF
  __m128i v245; // [rsp+280h] [rbp-C60h] BYREF
  __m128i v246; // [rsp+290h] [rbp-C50h] BYREF
  __m128i v247; // [rsp+2A0h] [rbp-C40h] BYREF
  int v248; // [rsp+2B0h] [rbp-C30h]
  __m128i v249[48]; // [rsp+2C0h] [rbp-C20h] BYREF
  __m128i v250; // [rsp+5C0h] [rbp-920h] BYREF
  __m128i v251; // [rsp+5D0h] [rbp-910h] BYREF
  __m128i v252; // [rsp+5E0h] [rbp-900h] BYREF
  __m128i v253; // [rsp+5F0h] [rbp-8F0h] BYREF
  __m128i v254; // [rsp+600h] [rbp-8E0h] BYREF
  _QWORD v255[3]; // [rsp+610h] [rbp-8D0h]
  __int64 v256; // [rsp+628h] [rbp-8B8h] BYREF
  __m128i v257; // [rsp+630h] [rbp-8B0h]
  __int64 *v258; // [rsp+640h] [rbp-8A0h]
  __int64 v259; // [rsp+648h] [rbp-898h]
  __m128i v260; // [rsp+650h] [rbp-890h] BYREF
  __int64 v261; // [rsp+660h] [rbp-880h]
  __int64 v262; // [rsp+668h] [rbp-878h]
  __int64 v263; // [rsp+670h] [rbp-870h]
  __int64 v264; // [rsp+678h] [rbp-868h] BYREF
  __m128i v265; // [rsp+680h] [rbp-860h]
  __int64 v266; // [rsp+690h] [rbp-850h]
  __int64 v267; // [rsp+698h] [rbp-848h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  sub_23A0BA0((__int64)&v250, 0);
  sub_23A2670(a1, (__int64)&v250);
  sub_233AAF0((__int64)&v250);
  memset(v227, 0, 40);
  memset(v228, 0, 40);
  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
    *v6 = &unk_4A30FE0;
  v250.m128i_i64[0] = (__int64)v6;
  sub_314D790(v227, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v7 = (_QWORD *)sub_22077B0(0x10u);
  if ( v7 )
    *v7 = &unk_4A0FEB8;
  v250.m128i_i64[0] = (__int64)v7;
  sub_314D790(v227, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v250.m128i_i8[0] = 0;
  sub_314D8C0(v227, v250.m128i_i8);
  v8 = (_QWORD *)sub_22077B0(0x10u);
  if ( v8 )
    *v8 = &unk_4A0FFF8;
  v250.m128i_i64[0] = (__int64)v8;
  sub_314D790(v227, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v249[0].m128i_i64[0] = 0x100000000000001LL;
  v249[0].m128i_i64[1] = 0x1000001000000LL;
  v249[1].m128i_i64[0] = 0;
  sub_29744D0((__int64)&v250, v249);
  sub_23A1F80(v227, v250.m128i_i64);
  sub_234AAB0((__int64)&v250, (__int64 *)v227, 0);
  sub_23571D0(a1, v250.m128i_i64);
  sub_233EFE0(v250.m128i_i64);
  v9 = sub_22077B0(0x10u);
  if ( v9 )
  {
    *(_BYTE *)(v9 + 8) = 0;
    *(_QWORD *)v9 = &unk_4A0EC78;
  }
  v250.m128i_i64[0] = v9;
  sub_2357280(a1, v250.m128i_i64);
  sub_233F000(v250.m128i_i64);
  LOBYTE(v199) = 0;
  HIDWORD(v199) = 1;
  LOBYTE(v200) = 0;
  sub_F10C20((__int64)&v250, v199, v200);
  sub_2353C90(v228, (__int64)&v250, v10, v11, v12, v13);
  sub_233BCC0((__int64)&v250);
  sub_234AAB0((__int64)&v250, (__int64 *)v228, 0);
  sub_23571D0(a1, v250.m128i_i64);
  sub_233EFE0(v250.m128i_i64);
  v14 = (_QWORD *)sub_22077B0(0x10u);
  if ( v14 )
    *v14 = &unk_4A0E5F8;
  v250.m128i_i64[0] = (__int64)v14;
  sub_314D9D0(a1, (unsigned __int64 *)&v250);
  if ( v250.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v250.m128i_i64[0] + 8LL))(v250.m128i_i64[0]);
  if ( !a5 )
  {
    sub_314DAD0(a1);
    if ( !a3[2] )
      goto LABEL_17;
LABEL_106:
    if ( a2 == 1 )
      goto LABEL_24;
    if ( a3[1] )
      goto LABEL_23;
    goto LABEL_178;
  }
  v15 = (_QWORD *)sub_22077B0(0x10u);
  if ( v15 )
    *v15 = &unk_4A0E4F8;
  v250.m128i_i64[0] = (__int64)v15;
  sub_314D9D0(a1, (unsigned __int64 *)&v250);
  sub_23501E0(v250.m128i_i64);
  if ( a3[2] )
    goto LABEL_106;
LABEL_17:
  v16 = (_QWORD *)sub_22077B0(0x10u);
  if ( v16 )
    *v16 = &unk_4A0E5B8;
  v250.m128i_i64[0] = (__int64)v16;
  sub_314D9D0(a1, (unsigned __int64 *)&v250);
  if ( v250.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v250.m128i_i64[0] + 8LL))(v250.m128i_i64[0]);
  if ( a2 == 1 || a3[1] )
  {
LABEL_23:
    if ( a3[2] )
      goto LABEL_24;
    v250.m128i_i8[0] = 1;
    sub_314DA70(a1, v250.m128i_i8);
    goto LABEL_171;
  }
LABEL_178:
  sub_30D6C00((__int64)&v243);
  sub_26124A0(
    (__int64)v249,
    0,
    0,
    0,
    0,
    v155,
    *(_OWORD *)&_mm_loadu_si128(&v243),
    *(_OWORD *)&_mm_loadu_si128(&v244),
    *(_OWORD *)&_mm_loadu_si128(&v245),
    *(_OWORD *)&_mm_loadu_si128(&v246),
    *(_OWORD *)&_mm_loadu_si128(&v247),
    v248);
  v156 = _mm_loadu_si128(v249);
  v258 = 0;
  v157 = _mm_loadu_si128(&v249[1]);
  v158 = _mm_loadu_si128(&v249[2]);
  v259 = 0;
  LODWORD(v255[0]) = v249[5].m128i_i32[0];
  v159 = _mm_loadu_si128(&v249[3]);
  v160 = _mm_loadu_si128(&v249[4]);
  v250 = v156;
  *(__m128i *)((char *)v255 + 4) = *(__m128i *)((char *)&v249[5] + 4);
  v251 = v157;
  v252 = v158;
  v256 = v249[6].m128i_i64[1];
  v249[6].m128i_i64[1] = 0;
  v257 = v249[7];
  v249[7] = 0u;
  v260 = v249[9];
  v253 = v159;
  v254 = v160;
  v261 = v249[10].m128i_i64[0];
  v264 = v249[11].m128i_i64[1];
  v265 = v249[12];
  memset(&v249[9], 0, 24);
  v262 = 0;
  v263 = 0;
  v249[12] = 0u;
  v249[11].m128i_i64[1] = 0;
  v266 = 0;
  v267 = 0;
  v161 = sub_22077B0(0xE8u);
  if ( v161 )
  {
    v162 = _mm_loadu_si128(&v250);
    *(_QWORD *)(v161 + 136) = 0;
    v163 = _mm_loadu_si128(&v251);
    v164 = _mm_loadu_si128(&v252);
    *(_QWORD *)(v161 + 144) = 0;
    v165 = _mm_loadu_si128(&v253);
    v166 = _mm_loadu_si128(&v254);
    *(__m128i *)(v161 + 8) = v162;
    *(_QWORD *)v161 = &unk_4A0D4B8;
    v167 = v255[0];
    *(__m128i *)(v161 + 24) = v163;
    *(_DWORD *)(v161 + 88) = v167;
    v168 = *(_QWORD *)((char *)v255 + 4);
    *(__m128i *)(v161 + 40) = v164;
    *(_QWORD *)(v161 + 92) = v168;
    v169 = *(_QWORD *)((char *)&v255[1] + 4);
    *(__m128i *)(v161 + 56) = v165;
    *(_QWORD *)(v161 + 100) = v169;
    v170 = v256;
    *(__m128i *)(v161 + 72) = v166;
    *(_QWORD *)(v161 + 112) = v170;
    v256 = 0;
    *(_QWORD *)(v161 + 120) = v257.m128i_i64[0];
    v257.m128i_i64[0] = 0;
    *(_QWORD *)(v161 + 128) = v257.m128i_i64[1];
    v257.m128i_i64[1] = 0;
    *(__m128i *)(v161 + 152) = v260;
    v260.m128i_i64[1] = 0;
    *(_QWORD *)(v161 + 168) = v261;
    v261 = 0;
    *(_QWORD *)(v161 + 192) = v264;
    v260.m128i_i64[0] = 0;
    *(_QWORD *)(v161 + 200) = v265.m128i_i64[0];
    v171 = v265.m128i_i64[1];
    *(_QWORD *)(v161 + 176) = 0;
    *(_QWORD *)(v161 + 184) = 0;
    *(_QWORD *)(v161 + 208) = v171;
    v265 = 0u;
    v264 = 0;
    *(_QWORD *)(v161 + 216) = 0;
    *(_QWORD *)(v161 + 224) = 0;
  }
  v238 = v161;
  sub_314D9D0(a1, &v238);
  sub_23501E0((__int64 *)&v238);
  sub_234A900((__int64)&v264);
  sub_234A900((__int64)&v260);
  sub_234A970((__int64)&v256);
  v172 = v249[12].m128i_i64[0];
  v173 = (_QWORD *)v249[11].m128i_i64[1];
  if ( v249[12].m128i_i64[0] != v249[11].m128i_i64[1] )
  {
    do
    {
      if ( *v173 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v173 + 8LL))(*v173);
      ++v173;
    }
    while ( (_QWORD *)v172 != v173 );
    v173 = (_QWORD *)v249[11].m128i_i64[1];
  }
  if ( v173 )
    j_j___libc_free_0((unsigned __int64)v173);
  v174 = v249[9].m128i_i64[1];
  v175 = (_QWORD *)v249[9].m128i_i64[0];
  if ( v249[9].m128i_i64[1] != v249[9].m128i_i64[0] )
  {
    do
    {
      if ( *v175 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v175 + 8LL))(*v175);
      ++v175;
    }
    while ( (_QWORD *)v174 != v175 );
    v175 = (_QWORD *)v249[9].m128i_i64[0];
  }
  if ( v175 )
    j_j___libc_free_0((unsigned __int64)v175);
  v176 = v249[7].m128i_i64[0];
  v177 = (_QWORD *)v249[6].m128i_i64[1];
  if ( v249[7].m128i_i64[0] != v249[6].m128i_i64[1] )
  {
    do
    {
      if ( *v177 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v177 + 8LL))(*v177);
      ++v177;
    }
    while ( (_QWORD *)v176 != v177 );
    v177 = (_QWORD *)v249[6].m128i_i64[1];
  }
  if ( v177 )
    j_j___libc_free_0((unsigned __int64)v177);
LABEL_171:
  if ( !a3[2] )
  {
    sub_23A0BA0((__int64)&v250, 0);
    sub_23A2670(a1, (__int64)&v250);
    sub_233AAF0((__int64)&v250);
  }
LABEL_24:
  v17 = (_QWORD *)sub_22077B0(0x10u);
  if ( v17 )
    *v17 = &unk_4A114F8;
  v250.m128i_i64[0] = (__int64)v17;
  v250.m128i_i8[8] = 0;
  sub_23571D0(a1, v250.m128i_i64);
  sub_233EFE0(v250.m128i_i64);
  v18 = (int *)sub_C94E20((__int64)qword_4F86310);
  if ( v18 )
    v19 = *v18;
  else
    v19 = qword_4F86310[2];
  if ( !v19
    || ((v20 = (int *)sub_C94E20((__int64)qword_4F86310)) == 0 ? (v21 = qword_4F86310[2]) : (v21 = *v20), v21 == 1) )
  {
    v233 = 0;
    v234 = 0;
    v235 = 0;
    v236 = 0;
    v237 = 0;
    v250.m128i_i8[0] = 1;
    sub_314D8C0((unsigned __int64 *)&v233, v250.m128i_i8);
    sub_291E720(&v250, 0);
    sub_23A2000((unsigned __int64 *)&v233, v250.m128i_i8);
    v250.m128i_i8[0] = 1;
    sub_314D8C0((unsigned __int64 *)&v233, v250.m128i_i8);
    if ( a3[6] )
    {
      v150 = (_QWORD *)sub_22077B0(0x18u);
      if ( v150 )
      {
        v150[1] = 0;
        v150[2] = 0;
        *v150 = &unk_4A0EDF8;
      }
      v250.m128i_i64[0] = (__int64)v150;
      sub_314D790((unsigned __int64 *)&v233, (unsigned __int64 *)&v250);
      sub_233EFE0(v250.m128i_i64);
      if ( (_BYTE)qword_5034328 )
      {
        sub_27D05A0(&v250);
        sub_314D920((unsigned __int64 *)&v233, v250.m128i_i32[0]);
      }
      else
      {
        v250.m128i_i16[0] = 1;
        sub_314D860((unsigned __int64 *)&v233, v250.m128i_i16);
      }
      LOBYTE(v201) = 0;
      HIDWORD(v201) = 1;
      LOBYTE(v202) = 0;
      sub_F10C20((__int64)&v250, v201, v202);
      sub_2353C90((unsigned __int64 *)&v233, (__int64)&v250, v151, v152, v153, v154);
      sub_233BCC0((__int64)&v250);
    }
    v249[1].m128i_i64[0] = 0;
    v249[0].m128i_i64[0] = 0x100000000000001LL;
    v249[0].m128i_i64[1] = 0x1000001000000LL;
    sub_29744D0((__int64)&v250, v249);
    sub_23A1F80((unsigned __int64 *)&v233, v250.m128i_i64);
    sub_234AAB0((__int64)&v250, &v233, 0);
    sub_23571D0(a1, v250.m128i_i64);
    sub_233EFE0(v250.m128i_i64);
    v22 = (_QWORD *)sub_22077B0(0x10u);
    if ( v22 )
      *v22 = &unk_4A0D3B8;
    v250.m128i_i64[0] = (__int64)v22;
    sub_314D9D0(a1, (unsigned __int64 *)&v250);
    if ( v250.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v250.m128i_i64[0] + 8LL))(v250.m128i_i64[0]);
    sub_23A0BA0((__int64)&v250, 0);
    sub_23A2670(a1, (__int64)&v250);
    sub_233AAF0((__int64)&v250);
    v23 = sub_22077B0(0x10u);
    if ( v23 )
    {
      *(_BYTE *)(v23 + 8) = 1;
      *(_QWORD *)v23 = &unk_4A0E8B8;
    }
    v250.m128i_i64[0] = v23;
    sub_314D9D0(a1, (unsigned __int64 *)&v250);
    sub_23501E0(v250.m128i_i64);
    v254.m128i_i64[1] = (__int64)&v253.m128i_i64[1];
    v255[0] = &v253.m128i_i64[1];
    v257.m128i_i64[1] = (__int64)&v256;
    v258 = &v256;
    v250.m128i_i32[2] = 0;
    v251.m128i_i64[0] = 0;
    v251.m128i_i64[1] = (__int64)&v250.m128i_i64[1];
    v252 = (__m128i)(unsigned __int64)&v250.m128i_u64[1];
    v253.m128i_i32[2] = 0;
    v254.m128i_i64[0] = 0;
    v255[1] = 0;
    LODWORD(v256) = 0;
    v257.m128i_i64[0] = 0;
    v259 = 0;
    v260.m128i_i8[0] = 0;
    sub_2358990(a1, (__int64)&v250);
    sub_233A870(&v250);
    LOBYTE(v203) = 0;
    HIDWORD(v203) = 1;
    LOBYTE(v204) = 0;
    v238 = 0;
    v239 = 0;
    v240 = 0;
    v241 = 0;
    v242 = 0;
    sub_F10C20((__int64)&v250, v203, v204);
    sub_2353C90(&v238, (__int64)&v250, v24, v25, v26, v27);
    sub_233BCC0((__int64)&v250);
    v249[0].m128i_i64[0] = 0x100000000000001LL;
    v249[0].m128i_i64[1] = 0x1000001000000LL;
    v249[1].m128i_i64[0] = 0;
    sub_29744D0((__int64)&v250, v249);
    sub_23A1F80(&v238, v250.m128i_i64);
    sub_234AAB0((__int64)&v250, (__int64 *)&v238, 0);
    sub_23571D0(a1, v250.m128i_i64);
    sub_233EFE0(v250.m128i_i64);
    v28 = sub_22077B0(0x10u);
    if ( v28 )
    {
      *(_BYTE *)(v28 + 8) = 0;
      *(_QWORD *)v28 = &unk_4A0EC78;
    }
    v250.m128i_i64[0] = v28;
    sub_2357280(a1, v250.m128i_i64);
    sub_233F000(v250.m128i_i64);
    memset(v249, 0, 40);
    if ( a2 > 1 )
    {
      v229.m128i_i32[2] = (int)&loc_1000000;
      v229.m128i_i16[6] = 0;
      v229.m128i_i64[0] = (__int64)&loc_1000000;
      sub_2339E50((__int64)&v250, (__int64)&loc_1000000, v229.m128i_i64[1] & 0xFFFFFFFFFFFFLL);
      sub_314D7D0((unsigned __int64 *)v249, v250.m128i_i64, v142, v143, v144, v145);
      sub_2341D90((__int64)&v250);
      if ( !(_BYTE)qword_5034248 )
      {
        sub_27DC820((__int64)&v250, -1);
        sub_2354380((unsigned __int64 *)v249, v250.m128i_i64);
        sub_233B480((__int64)&v250, (__int64)&v250, v189, v190, v191, v192);
      }
      v230 = 0;
      v229.m128i_i64[0] = 0x100000000000001LL;
      v229.m128i_i64[1] = 0x1000001000000LL;
      sub_29744D0((__int64)&v250, &v229);
      sub_23A1F80((unsigned __int64 *)v249, v250.m128i_i64);
      LOBYTE(v205) = 0;
      HIDWORD(v205) = 1;
      LOBYTE(v206) = 0;
      sub_F10C20((__int64)&v250, v205, v206);
      sub_2353C90((unsigned __int64 *)v249, (__int64)&v250, v146, v147, v148, v149);
      sub_233BCC0((__int64)&v250);
      v29 = a3;
      if ( !a3[7] )
        goto LABEL_43;
    }
    else
    {
      v29 = a3;
      if ( !a3[7] )
      {
LABEL_43:
        sub_234AAB0((__int64)&v250, v249[0].m128i_i64, 0);
        sub_23571D0(a1, v250.m128i_i64);
        sub_233EFE0(v250.m128i_i64);
        v250.m128i_i8[0] = a3[7];
        sub_314DA10(a1, v250.m128i_i8);
        v229.m128i_i8[4] = 0;
        v229.m128i_i32[0] = a4;
        sub_3717E20(&v250, &v229, 0);
        v30 = v250.m128i_u64[1];
        if ( v250.m128i_i64[0] != v250.m128i_i64[1] )
        {
          v31 = v250.m128i_i64[1];
          v32 = v250.m128i_i64[0];
          do
          {
            v33 = (unsigned __int64 *)v32;
            v32 += 8;
            sub_314D9D0(a1, v33);
          }
          while ( v31 != v32 );
          v34 = v250.m128i_i64[1];
          v30 = v250.m128i_i64[0];
          if ( v250.m128i_i64[1] != v250.m128i_i64[0] )
          {
            v35 = (_QWORD *)v250.m128i_i64[0];
            do
            {
              if ( *v35 )
                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v35 + 8LL))(*v35);
              ++v35;
            }
            while ( (_QWORD *)v34 != v35 );
            v30 = v250.m128i_i64[0];
          }
        }
        if ( v30 )
          j_j___libc_free_0(v30);
        if ( a2 > 1 && !a3[1] )
        {
          v250 = 0u;
          v251 = 0u;
          v188 = sub_22077B0(0x28u);
          if ( v188 )
          {
            *(_QWORD *)(v188 + 8) = 0;
            *(_BYTE *)(v188 + 16) = 0;
            *(_DWORD *)(v188 + 20) = 0;
            *(_QWORD *)v188 = &unk_4A0ECB8;
            *(_QWORD *)(v188 + 24) = 0;
            *(_QWORD *)(v188 + 32) = 0;
          }
          v229.m128i_i64[0] = v188;
          sub_2357280(a1, v229.m128i_i64);
          sub_233F000(v229.m128i_i64);
          sub_233F0C0(v250.m128i_i64);
        }
        sub_233F7F0((__int64)v249);
        sub_233F7F0((__int64)&v238);
        sub_233F7F0((__int64)&v233);
        goto LABEL_57;
      }
    }
    if ( v29[4] )
    {
      v229.m128i_i16[4] = 1;
      v229.m128i_i64[0] = __PAIR64__(qword_4FFDDA8[8], qword_4FFDE88[8]);
      sub_2356430((__int64)&v250, v229.m128i_i64, 1, 1, 0);
      sub_2353940((unsigned __int64 *)v249, v250.m128i_i64);
      sub_233F7F0((__int64)&v250.m128i_i64[1]);
      sub_233F7D0(v250.m128i_i64);
    }
    goto LABEL_43;
  }
LABEL_57:
  v36 = (int *)sub_C94E20((__int64)qword_4F86310);
  if ( v36 )
    v37 = *v36;
  else
    v37 = qword_4F86310[2];
  if ( v37 )
  {
    v178 = sub_C94E20((__int64)qword_4F86310);
    v179 = v178 ? *v178 : LODWORD(qword_4F86310[2]);
    if ( v179 != 2 )
      goto LABEL_165;
  }
  if ( a2 > 2 )
  {
    v184 = sub_22077B0(0x10u);
    if ( v184 )
    {
      *(_DWORD *)(v184 + 8) = 2;
      *(_QWORD *)v184 = &unk_4A0EA78;
    }
    v250.m128i_i64[0] = v184;
    sub_2357280(a1, v250.m128i_i64);
    sub_233F000(v250.m128i_i64);
  }
  v229 = 0u;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  if ( (_BYTE)qword_5034088 )
  {
    v183 = (_QWORD *)sub_22077B0(0x10u);
    if ( v183 )
      *v183 = &unk_4A30FE0;
    v250.m128i_i64[0] = (__int64)v183;
    sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
    sub_233EFE0(v250.m128i_i64);
  }
  v38 = (_QWORD *)sub_22077B0(0x10u);
  if ( v38 )
    *v38 = &unk_4A0F1B8;
  v250.m128i_i64[0] = (__int64)v38;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v249[0].m128i_i64[0] = 0x100000000000001LL;
  v249[0].m128i_i64[1] = 0x1000001000000LL;
  v249[1].m128i_i64[0] = 0;
  sub_29744D0((__int64)&v250, v249);
  sub_23A1F80((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_291E720(&v250, 0);
  sub_23A2000((unsigned __int64 *)&v229, v250.m128i_i8);
  v250.m128i_i8[0] = 1;
  sub_314D8C0((unsigned __int64 *)&v229, v250.m128i_i8);
  v39 = (_QWORD *)sub_22077B0(0x10u);
  if ( v39 )
    *v39 = &unk_4A114B8;
  v250.m128i_i64[0] = (__int64)v39;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  LOBYTE(v207) = 0;
  HIDWORD(v207) = 1;
  LOBYTE(v208) = 0;
  sub_F10C20((__int64)&v250, v207, v208);
  sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v40, v41, v42, v43);
  sub_233BCC0((__int64)&v250);
  v44 = (_QWORD *)sub_22077B0(0x10u);
  if ( v44 )
    *v44 = &unk_4A10F78;
  v250.m128i_i64[0] = (__int64)v44;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v249[0].m128i_i64[0] = 0x100000000000001LL;
  v249[0].m128i_i64[1] = 0x1000001000000LL;
  v249[1].m128i_i64[0] = 0;
  sub_29744D0((__int64)&v250, v249);
  sub_23A1F80((unsigned __int64 *)&v229, v250.m128i_i64);
  memset(v249, 0, 0x2F8u);
  sub_2350260(v249[6].m128i_i64, 0);
  v45 = &v249[11];
  do
  {
    v45->m128i_i64[0] = 0;
    v45 += 2;
    v45[-1].m128i_i32[2] = 0;
    v45[-2].m128i_i64[1] = 0;
    v45[-1].m128i_i32[0] = 0;
    v45[-1].m128i_i32[1] = 0;
  }
  while ( &v249[47] != v45 );
  sub_23504B0((__int64)&v250, v249);
  v46 = (_QWORD *)sub_22077B0(0x300u);
  v47 = (unsigned __int64)v46;
  if ( v46 )
  {
    *v46 = &unk_4A10BB8;
    sub_23504B0((__int64)(v46 + 1), &v250);
  }
  v238 = v47;
  sub_314D790((unsigned __int64 *)&v229, &v238);
  sub_233EFE0((__int64 *)&v238);
  sub_233B610((__int64)&v250);
  sub_233B610((__int64)v249);
  sub_28448C0(&v233, 1, 0);
  v48 = v233;
  v49 = sub_22077B0(0x10u);
  if ( v49 )
  {
    *(_WORD *)(v49 + 8) = v48;
    *(_QWORD *)v49 = &unk_4A124B8;
  }
  v250 = (__m128i)(unsigned __int64)v49;
  v238 = 0;
  v251 = 0u;
  v252 = 0u;
  v253.m128i_i32[0] = 0;
  v50 = (_QWORD *)sub_22077B0(0x10u);
  if ( v50 )
    *v50 = &unk_4A0B640;
  v249[0].m128i_i64[0] = (__int64)v50;
  sub_314D790(&v250.m128i_u64[1], (unsigned __int64 *)v249);
  sub_233EFE0(v249[0].m128i_i64);
  v51 = (_QWORD *)sub_22077B0(0x10u);
  if ( v51 )
    *v51 = &unk_4A0B680;
  v249[0].m128i_i64[0] = (__int64)v51;
  sub_314D790(&v250.m128i_u64[1], (unsigned __int64 *)v249);
  sub_233EFE0(v249[0].m128i_i64);
  sub_233F7D0((__int64 *)&v238);
  sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_233F7F0((__int64)&v250.m128i_i64[1]);
  sub_233F7D0(v250.m128i_i64);
  v52 = (_QWORD *)sub_22077B0(0x10u);
  if ( v52 )
    *v52 = &unk_4A12438;
  v250 = (__m128i)(unsigned __int64)v52;
  v238 = 0;
  v251 = 0u;
  v252 = 0u;
  v253.m128i_i32[0] = 0;
  v53 = (_QWORD *)sub_22077B0(0x10u);
  if ( v53 )
    *v53 = &unk_4A0B640;
  v249[0].m128i_i64[0] = (__int64)v53;
  sub_314D790(&v250.m128i_u64[1], (unsigned __int64 *)v249);
  sub_233EFE0(v249[0].m128i_i64);
  v54 = (_QWORD *)sub_22077B0(0x10u);
  if ( v54 )
    *v54 = &unk_4A0B680;
  v249[0].m128i_i64[0] = (__int64)v54;
  sub_314D790(&v250.m128i_u64[1], (unsigned __int64 *)v249);
  sub_233EFE0(v249[0].m128i_i64);
  sub_233F7D0((__int64 *)&v238);
  sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_233F7F0((__int64)&v250.m128i_i64[1]);
  sub_233F7D0(v250.m128i_i64);
  LOBYTE(v209) = 0;
  HIDWORD(v209) = 1;
  LOBYTE(v210) = 0;
  sub_F10C20((__int64)&v250, v209, v210);
  sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v55, v56, v57, v58);
  sub_233BCC0((__int64)&v250);
  v249[0].m128i_i64[0] = 0x100000000000001LL;
  v249[0].m128i_i64[1] = 0x1000001000000LL;
  v249[1].m128i_i64[0] = 0;
  sub_29744D0((__int64)&v250, v249);
  sub_23A1F80((unsigned __int64 *)&v229, v250.m128i_i64);
  if ( a3[4] )
  {
    v249[0].m128i_i16[4] = 1;
    v249[0].m128i_i64[0] = __PAIR64__(qword_4FFDDA8[8], qword_4FFDE88[8]);
    sub_2356430((__int64)&v250, v249[0].m128i_i64, 1, 1, 0);
    sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
    sub_233F7F0((__int64)&v250.m128i_i64[1]);
    sub_233F7D0(v250.m128i_i64);
  }
  if ( a3[6] )
  {
    if ( (_BYTE)qword_5034328 )
    {
      sub_27D05A0(&v250);
      sub_314D920((unsigned __int64 *)&v229, v250.m128i_i32[0]);
    }
    else
    {
      v250.m128i_i16[0] = 1;
      sub_314D860((unsigned __int64 *)&v229, v250.m128i_i16);
    }
    LOBYTE(v211) = 0;
    HIDWORD(v211) = 1;
    LOBYTE(v212) = 0;
    sub_F10C20((__int64)&v250, v211, v212);
    sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v87, v88, v89, v90);
    sub_233BCC0((__int64)&v250);
  }
  v59 = sub_22077B0(0x10u);
  if ( v59 )
  {
    *(_WORD *)(v59 + 8) = 257;
    *(_QWORD *)v59 = &unk_4A124F8;
  }
  v250 = (__m128i)(unsigned __int64)v59;
  v238 = 0;
  v251 = 0u;
  v252 = 0u;
  v253.m128i_i32[0] = 0;
  v60 = (_QWORD *)sub_22077B0(0x10u);
  if ( v60 )
    *v60 = &unk_4A0B640;
  v249[0].m128i_i64[0] = (__int64)v60;
  sub_314D790(&v250.m128i_u64[1], (unsigned __int64 *)v249);
  sub_233EFE0(v249[0].m128i_i64);
  v61 = (_QWORD *)sub_22077B0(0x10u);
  if ( v61 )
    *v61 = &unk_4A0B680;
  v249[0].m128i_i64[0] = (__int64)v61;
  sub_314D790(&v250.m128i_u64[1], (unsigned __int64 *)v249);
  sub_233EFE0(v249[0].m128i_i64);
  sub_233F7D0((__int64 *)&v238);
  sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_233F7F0((__int64)&v250.m128i_i64[1]);
  sub_233F7D0(v250.m128i_i64);
  LOBYTE(v213) = 0;
  HIDWORD(v213) = 1;
  LOBYTE(v214) = 0;
  sub_F10C20((__int64)&v250, v213, v214);
  sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v62, v63, v64, v65);
  sub_233BCC0((__int64)&v250);
  v249[0].m128i_i64[0] = (__int64)v249[1].m128i_i64;
  v249[0].m128i_i64[1] = 0x600000000LL;
  v249[4].m128i_i32[0] = 0;
  v249[4].m128i_i64[1] = 0;
  memset(&v249[5], 0, 40);
  sub_2332320((__int64)v249, 0, v66, v67, v68, v69);
  v70 = sub_22077B0(0x10u);
  if ( v70 )
  {
    *(_BYTE *)(v70 + 8) = 1;
    *(_QWORD *)v70 = &unk_4A11F78;
  }
  v250.m128i_i64[0] = v70;
  sub_314DB70(&v249[4].m128i_u64[1], (unsigned __int64 *)&v250);
  sub_233F7D0(v250.m128i_i64);
  sub_2332320((__int64)v249, 0, v71, v72, v73, v74);
  v75 = (_QWORD *)sub_22077B0(0x10u);
  if ( v75 )
    *v75 = &unk_4A12078;
  v250.m128i_i64[0] = (__int64)v75;
  sub_314DB70(&v249[4].m128i_u64[1], (unsigned __int64 *)&v250);
  sub_233F7D0(v250.m128i_i64);
  sub_2332320((__int64)v249, 0, v76, v77, v78, v79);
  v80 = (_QWORD *)sub_22077B0(0x10u);
  if ( v80 )
    *v80 = &unk_4A12038;
  v250.m128i_i64[0] = (__int64)v80;
  sub_314DB70(&v249[4].m128i_u64[1], (unsigned __int64 *)&v250);
  sub_233F7D0(v250.m128i_i64);
  sub_23A20C0((__int64)&v250, (__int64)v249, 0, 0, 0, v81);
  sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_233F7F0((__int64)&v250.m128i_i64[1]);
  sub_233F7D0(v250.m128i_i64);
  if ( a2 <= 1 )
  {
    LOBYTE(v215) = 0;
    HIDWORD(v215) = 1;
    LOBYTE(v216) = 0;
    sub_F10C20((__int64)&v250, v215, v216);
    sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v91, v92, v93, v94);
    sub_233BCC0((__int64)&v250);
  }
  else
  {
    if ( !(_BYTE)qword_5034168 )
    {
      v250.m128i_i8[0] = 0;
      v251.m128i_i16[4] = 0;
      v82 = a3[10];
      *(__int16 *)((char *)v250.m128i_i16 + 1) = v82;
      *(__int16 *)((char *)&v250.m128i_i16[1] + 1) = v82;
      *(__int16 *)((char *)&v250.m128i_i16[2] + 1) = v82;
      *(__int16 *)((char *)&v250.m128i_i16[3] + 1) = v82;
      v250.m128i_i8[9] = v82;
      v251.m128i_i8[0] = 0;
      v251.m128i_i32[1] = a2;
      sub_2353C00((unsigned __int64 *)&v229, &v250);
    }
    LOBYTE(v215) = 0;
    HIDWORD(v215) = 1;
    LOBYTE(v216) = 0;
    sub_F10C20((__int64)&v250, v215, v216);
    sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v83, v84, v85, v86);
    sub_233BCC0((__int64)&v250);
    if ( !a3[10] && !(_BYTE)qword_5034168 )
    {
      v251.m128i_i32[0] = 0;
      v250 = (__m128i)0x10001000100uLL;
      v251.m128i_i16[4] = 0;
      v251.m128i_i32[1] = a2;
      sub_2353C00((unsigned __int64 *)&v229, &v250);
    }
  }
  sub_29744A0((__int64)&v250);
  sub_23A1F80((unsigned __int64 *)&v229, v250.m128i_i64);
  v95 = (_QWORD *)sub_22077B0(0x48u);
  if ( v95 )
  {
    v95[1] = 0;
    v95[2] = 0;
    v95[3] = 0;
    *v95 = &unk_4A10038;
    v95[4] = 0;
    v95[5] = 0;
    v95[6] = 0;
    v95[7] = 0;
    v95[8] = 0;
  }
  v250.m128i_i64[0] = (__int64)v95;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  sub_291E720(&v250, 0);
  sub_23A2000((unsigned __int64 *)&v229, v250.m128i_i8);
  v250.m128i_i8[0] = 1;
  sub_314D8C0((unsigned __int64 *)&v229, v250.m128i_i8);
  LOBYTE(v217) = 0;
  HIDWORD(v217) = 1;
  LOBYTE(v218) = 0;
  sub_F10C20((__int64)&v250, v217, v218);
  sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v96, v97, v98, v99);
  sub_233BCC0((__int64)&v250);
  v238 = 0x1000000;
  WORD2(v239) = 0;
  LODWORD(v239) = 0;
  sub_2339E50((__int64)&v250, 0x1000000, v239 & 0xFFFFFFFFFFFFLL);
  sub_314D7D0((unsigned __int64 *)&v229, v250.m128i_i64, v100, v101, v102, v103);
  sub_2341D90((__int64)&v250);
  if ( a3[4] )
  {
    v238 = __PAIR64__(qword_4FFDDA8[8], qword_4FFDE88[8]);
    LOWORD(v239) = 1;
    sub_2356430((__int64)&v250, (__int64 *)&v238, 1, 1, 0);
    sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
    sub_233F7F0((__int64)&v250.m128i_i64[1]);
    sub_233F7D0(v250.m128i_i64);
  }
  v104 = (_QWORD *)sub_22077B0(0x10u);
  if ( v104 )
    *v104 = &unk_4A10D38;
  v250.m128i_i64[0] = (__int64)v104;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v105 = (_QWORD *)sub_22077B0(0x10u);
  if ( v105 )
    *v105 = &unk_4A12038;
  v250 = (__m128i)(unsigned __int64)v105;
  v233 = 0;
  v251 = 0u;
  v252 = 0u;
  v253.m128i_i32[0] = 0;
  v106 = (_QWORD *)sub_22077B0(0x10u);
  if ( v106 )
    *v106 = &unk_4A0B640;
  v238 = (unsigned __int64)v106;
  sub_314D790(&v250.m128i_u64[1], &v238);
  sub_233EFE0((__int64 *)&v238);
  v107 = (_QWORD *)sub_22077B0(0x10u);
  if ( v107 )
    *v107 = &unk_4A0B680;
  v238 = (unsigned __int64)v107;
  sub_314D790(&v250.m128i_u64[1], &v238);
  sub_233EFE0((__int64 *)&v238);
  sub_233F7D0(&v233);
  sub_2353940((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_233F7F0((__int64)&v250.m128i_i64[1]);
  sub_233F7D0(v250.m128i_i64);
  LOBYTE(v219) = 0;
  HIDWORD(v219) = 1;
  LOBYTE(v220) = 0;
  sub_F10C20((__int64)&v250, v219, v220);
  sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v108, v109, v110, v111);
  sub_233BCC0((__int64)&v250);
  v112 = (_QWORD *)sub_22077B0(0x10u);
  if ( v112 )
    *v112 = &unk_4A0F1B8;
  v250.m128i_i64[0] = (__int64)v112;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v113 = (_QWORD *)sub_22077B0(0x10u);
  if ( v113 )
    *v113 = &unk_4A0F4B8;
  v250.m128i_i64[0] = (__int64)v113;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v114 = (_QWORD *)sub_22077B0(0x10u);
  if ( v114 )
    *v114 = &unk_4A0ED38;
  v250.m128i_i64[0] = (__int64)v114;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  sub_29744A0((__int64)&v250);
  sub_23A1F80((unsigned __int64 *)&v229, v250.m128i_i64);
  sub_291E720(&v250, 0);
  sub_23A2000((unsigned __int64 *)&v229, v250.m128i_i8);
  v250.m128i_i8[0] = 1;
  sub_314D8C0((unsigned __int64 *)&v229, v250.m128i_i8);
  if ( !*a3 )
  {
    v185 = a3[5];
    v186 = sub_22077B0(0x10u);
    if ( v186 )
    {
      *(_BYTE *)(v186 + 8) = v185;
      *(_QWORD *)v186 = &unk_4A11DB8;
    }
    v250.m128i_i64[0] = v186;
    sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
    if ( v250.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v250.m128i_i64[0] + 8LL))(v250.m128i_i64[0]);
  }
  if ( a3[5] )
  {
    sub_291E720(&v250, 0);
    sub_23A2000((unsigned __int64 *)&v229, v250.m128i_i8);
  }
  LOBYTE(v221) = 0;
  HIDWORD(v221) = 1;
  LOBYTE(v222) = 0;
  sub_F10C20((__int64)&v250, v221, v222);
  sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v115, v116, v117, v118);
  sub_233BCC0((__int64)&v250);
  v119 = (_QWORD *)sub_22077B0(0x10u);
  if ( v119 )
    *v119 = &unk_4A117F8;
  v250.m128i_i64[0] = (__int64)v119;
  sub_314D790((unsigned __int64 *)&v229, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  if ( a3[6] )
  {
    if ( (_BYTE)qword_5034328 )
    {
      sub_27D05A0(&v250);
      sub_314D920((unsigned __int64 *)&v229, v250.m128i_i32[0]);
    }
    else
    {
      v250.m128i_i16[0] = 1;
      sub_314D860((unsigned __int64 *)&v229, v250.m128i_i16);
    }
    LOBYTE(v223) = 0;
    HIDWORD(v223) = 1;
    LOBYTE(v224) = 0;
    sub_F10C20((__int64)&v250, v223, v224);
    sub_2353C90((unsigned __int64 *)&v229, (__int64)&v250, v120, v121, v122, v123);
    sub_233BCC0((__int64)&v250);
  }
  sub_234AAB0((__int64)&v250, v229.m128i_i64, 0);
  sub_23571D0(a1, v250.m128i_i64);
  sub_233EFE0(v250.m128i_i64);
  if ( a3[6] && a3[3] && !(_BYTE)qword_5034328 )
  {
    v187 = (_QWORD *)sub_22077B0(0x10u);
    if ( v187 )
      *v187 = &unk_4A0E578;
    v250.m128i_i64[0] = (__int64)v187;
    sub_314D9D0(a1, (unsigned __int64 *)&v250);
    sub_23501E0(v250.m128i_i64);
  }
  v124 = (_QWORD *)sub_22077B0(0x10u);
  if ( v124 )
    *v124 = &unk_4A0E2B8;
  v250.m128i_i64[0] = (__int64)v124;
  sub_314D9D0(a1, (unsigned __int64 *)&v250);
  sub_23501E0(v250.m128i_i64);
  if ( a2 > 2 )
  {
    sub_23A0BA0((__int64)&v250, 0);
    sub_23A2670(a1, (__int64)&v250);
    sub_233AAF0((__int64)&v250);
  }
  else if ( a2 != 2 )
  {
    goto LABEL_148;
  }
  v180 = (_QWORD *)sub_22077B0(0x10u);
  if ( v180 )
    *v180 = &unk_4A0CF78;
  v250.m128i_i64[0] = (__int64)v180;
  sub_314D9D0(a1, (unsigned __int64 *)&v250);
  if ( v250.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v250.m128i_i64[0] + 8LL))(v250.m128i_i64[0]);
  if ( a2 > 2 )
  {
    v181 = (_QWORD *)sub_22077B0(0x10u);
    if ( v181 )
      *v181 = &unk_4A11478;
    v250.m128i_i64[0] = (__int64)v181;
    v250.m128i_i8[8] = 0;
    sub_23571D0(a1, v250.m128i_i64);
    sub_233EFE0(v250.m128i_i64);
    goto LABEL_216;
  }
LABEL_148:
  if ( a2 > 1 )
  {
LABEL_216:
    v182 = (_QWORD *)sub_22077B0(0x10u);
    if ( v182 )
      *v182 = &unk_4A0CF78;
    v250.m128i_i64[0] = (__int64)v182;
    sub_314D9D0(a1, (unsigned __int64 *)&v250);
    if ( v250.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v250.m128i_i64[0] + 8LL))(v250.m128i_i64[0]);
  }
  v233 = 0;
  v234 = 0;
  v235 = 0;
  v236 = 0;
  v237 = 0;
  v125 = (_QWORD *)sub_22077B0(0x10u);
  if ( v125 )
    *v125 = &unk_4A11778;
  v250.m128i_i64[0] = (__int64)v125;
  sub_314D790((unsigned __int64 *)&v233, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v126 = (_QWORD *)sub_22077B0(0x10u);
  if ( v126 )
    *v126 = &unk_4A11738;
  v250.m128i_i64[0] = (__int64)v126;
  sub_314D790((unsigned __int64 *)&v233, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v127 = sub_22077B0(0x10u);
  if ( v127 )
  {
    *(_BYTE *)(v127 + 8) = 1;
    *(_QWORD *)v127 = &unk_4A11D78;
  }
  v250.m128i_i64[0] = v127;
  sub_314D790((unsigned __int64 *)&v233, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  sub_234AAB0((__int64)&v250, &v233, 0);
  sub_23571D0(a1, v250.m128i_i64);
  sub_233EFE0(v250.m128i_i64);
  v128 = a3[8] ^ 1;
  v196 = a3[9] ^ 1;
  v129 = sub_22077B0(0x10u);
  if ( v129 )
  {
    *(_BYTE *)(v129 + 9) = v128;
    *(_BYTE *)(v129 + 8) = v196;
    *(_QWORD *)v129 = &unk_4A0EA38;
  }
  v250.m128i_i64[0] = v129;
  sub_314D9D0(a1, (unsigned __int64 *)&v250);
  sub_23501E0(v250.m128i_i64);
  v238 = 0;
  v239 = 0;
  v240 = 0;
  v241 = 0;
  v242 = 0;
  v130 = (_QWORD *)sub_22077B0(0x10u);
  if ( v130 )
    *v130 = &unk_4A117B8;
  v250.m128i_i64[0] = (__int64)v130;
  sub_314D790(&v238, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  v131 = (_QWORD *)sub_22077B0(0x10u);
  if ( v131 )
    *v131 = &unk_4A11738;
  v250.m128i_i64[0] = (__int64)v131;
  sub_314D790(&v238, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  LOBYTE(v197) = 0;
  HIDWORD(v197) = 1;
  LOBYTE(v198) = 1;
  sub_F10C20((__int64)&v250, v197, v198);
  sub_2353C90(&v238, (__int64)&v250, v132, v133, v134, v135);
  sub_233BCC0((__int64)&v250);
  if ( (_BYTE)qword_5033FA8 )
  {
    v136 = (_QWORD *)sub_22077B0(0x10u);
    if ( v136 )
      *v136 = &unk_4A11638;
  }
  else
  {
    v136 = (_QWORD *)sub_22077B0(0x10u);
    if ( v136 )
      *v136 = &unk_4A0FC78;
  }
  v250.m128i_i64[0] = (__int64)v136;
  sub_314D790(&v238, (unsigned __int64 *)&v250);
  sub_233EFE0(v250.m128i_i64);
  sub_F10C20((__int64)&v250, v197, v198);
  sub_2353C90(&v238, (__int64)&v250, v137, v138, v139, v140);
  sub_233BCC0((__int64)&v250);
  v225.m128i_i64[0] = 0x100010000000001LL;
  v225.m128i_i64[1] = 0x1000101000000LL;
  v226 = 0;
  sub_29744D0((__int64)&v250, &v225);
  sub_23A1F80(&v238, v250.m128i_i64);
  sub_234AAB0((__int64)&v250, (__int64 *)&v238, 0);
  sub_23571D0(a1, v250.m128i_i64);
  sub_233EFE0(v250.m128i_i64);
  sub_233F7F0((__int64)&v238);
  sub_233F7F0((__int64)&v233);
  sub_2337B30((unsigned __int64 *)v249);
  sub_233F7F0((__int64)&v229);
LABEL_165:
  sub_233F7F0((__int64)v228);
  sub_233F7F0((__int64)v227);
  return a1;
}
