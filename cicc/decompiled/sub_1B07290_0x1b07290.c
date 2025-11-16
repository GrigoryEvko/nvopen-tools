// Function: sub_1B07290
// Address: 0x1b07290
//
__int64 __fastcall sub_1B07290(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned __int8 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 *a18)
{
  char **v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // r12
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // eax
  unsigned int v30; // r12d
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  char *v34; // r13
  char *v35; // r12
  __int64 v36; // r14
  _QWORD *v37; // rdi
  __int128 v38; // rdi
  __int64 v39; // r12
  __int64 v40; // rbx
  __int64 i; // r14
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  __int64 v46; // rax
  void **v47; // r13
  __int64 v48; // rbx
  __int64 *v49; // r14
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rsi
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  __int64 v55; // rax
  char *v56; // rdx
  int v57; // esi
  unsigned int v58; // ecx
  char **v59; // rax
  char *v60; // rdi
  __int64 v61; // rdx
  bool v62; // zf
  char *v63; // rbx
  __int64 v64; // rax
  char v65; // al
  unsigned __int64 *v66; // r12
  int v67; // esi
  int v68; // eax
  unsigned __int64 v69; // rax
  char *v70; // rax
  _QWORD *v71; // rbx
  _QWORD *v72; // r12
  _BYTE *v73; // rsi
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rdi
  unsigned int v77; // esi
  __int64 *v78; // rcx
  __int64 v79; // r9
  _QWORD *v80; // rax
  _QWORD *v81; // rbx
  _QWORD *v82; // r12
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rax
  _BYTE *v86; // r12
  __int64 v87; // r14
  __int64 k; // rbx
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rbx
  __int64 v92; // r12
  __int64 v93; // rdx
  char v94; // al
  unsigned int v95; // edi
  __int64 v96; // rcx
  __int64 v97; // rdx
  __int64 v98; // rsi
  __int64 v99; // rcx
  __int64 v100; // r13
  int v101; // esi
  unsigned int v102; // ecx
  char *v103; // rdx
  __int64 v104; // r8
  __int64 v105; // rdx
  __int64 v106; // rsi
  __int64 *v107; // rax
  __int64 v108; // rsi
  unsigned __int64 v109; // rcx
  __int64 v110; // rcx
  __int64 v111; // rax
  unsigned __int64 v112; // rax
  double v113; // xmm4_8
  double v114; // xmm5_8
  unsigned int v115; // ebx
  unsigned __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rsi
  unsigned __int64 v119; // rdi
  __int64 v120; // rsi
  __int64 v121; // rcx
  unsigned __int64 v122; // rax
  unsigned __int64 v123; // rbx
  int v124; // r12d
  _QWORD *v125; // r13
  __int64 v126; // r14
  _QWORD *v127; // rdi
  __int64 v128; // r13
  __int64 v129; // rbx
  __int64 v130; // rax
  _QWORD *v131; // r12
  int v132; // r12d
  double v133; // xmm4_8
  double v134; // xmm5_8
  _QWORD *v135; // r13
  __int64 v136; // r14
  _QWORD *v137; // rdi
  __int64 v138; // r13
  __int64 v139; // rbx
  int v140; // r8d
  int v141; // r9d
  __int64 v142; // rax
  int v143; // r8d
  int v144; // r9d
  int v145; // r8d
  int v146; // r9d
  int v147; // r8d
  int v148; // r9d
  __int64 *v149; // r12
  __int64 *v150; // rbx
  __int64 *v151; // rcx
  __int64 *v152; // r10
  __int64 v153; // r8
  __int64 *v154; // rdi
  __int64 *v155; // rax
  __int64 *v156; // rsi
  __int64 *v157; // r12
  __int64 *v158; // rbx
  unsigned __int64 v159; // rdi
  __int64 v160; // r10
  __int64 v161; // rsi
  __int64 *v162; // r8
  __int64 *v163; // rax
  __int64 *v164; // rcx
  __int64 *v165; // r12
  __int64 *v166; // rbx
  unsigned __int64 v167; // rdi
  __int64 v168; // r10
  __int64 v169; // rsi
  __int64 *v170; // r8
  __int64 *v171; // rax
  __int64 *v172; // rcx
  __int64 v173; // rdx
  unsigned __int64 v174; // r14
  __int64 *v175; // rbx
  __int64 *v176; // rdx
  __int64 v177; // r12
  __int64 *v178; // rax
  __int64 *v179; // rcx
  unsigned __int64 v180; // rax
  unsigned __int64 v181; // r15
  int v182; // eax
  double v183; // xmm4_8
  double v184; // xmm5_8
  _QWORD *v185; // r14
  __int64 v186; // rax
  _QWORD *v187; // rbx
  __int64 v188; // rdx
  __int64 v189; // rdi
  __int64 v190; // rcx
  unsigned __int64 *v191; // rcx
  unsigned __int64 v192; // rdx
  double v193; // xmm4_8
  double v194; // xmm5_8
  __int64 v195; // rbx
  __int64 v196; // r12
  __int64 v197; // rdx
  char v198; // cl
  unsigned int v199; // eax
  __int64 v200; // rsi
  __int64 v201; // rdx
  __int64 v202; // rdi
  _QWORD *v203; // rbx
  __int64 v204; // r12
  __int64 v205; // rsi
  __int64 v206; // rax
  __int64 v207; // rax
  char v208; // al
  char **v209; // r14
  __int64 v210; // rax
  char *v211; // r14
  char **v212; // rdx
  unsigned int v213; // ecx
  void **v214; // rax
  char **v215; // rdi
  __int64 v216; // rax
  char **v217; // r14
  __int64 v218; // rax
  int v219; // ecx
  char *v220; // rdx
  void **v221; // rdx
  char *v222; // rdx
  char *v223; // rax
  __int64 v224; // rax
  unsigned int v225; // edx
  char *v226; // rcx
  int v227; // r10d
  char **v228; // r9
  int v229; // esi
  int v230; // eax
  char *v231; // rdx
  void **v232; // rdx
  int v233; // r11d
  void **v234; // r10
  int v235; // ecx
  unsigned int v236; // esi
  void *v237; // rdi
  __int64 *v238; // rax
  int v239; // ebx
  char **v240; // r10
  int v241; // ecx
  int v242; // r11d
  char *v243; // rdi
  int v244; // eax
  __int64 v245; // rcx
  int v246; // ecx
  int v247; // r10d
  int v248; // r15d
  void **v249; // r9
  __int64 *v250; // rax
  _QWORD *v251; // r13
  _QWORD *v252; // rbx
  __int64 v253; // rsi
  _QWORD *v254; // rdi
  __int64 v255; // r14
  __int64 v256; // rax
  __int64 v257; // rax
  __int64 v258; // rax
  __int64 v259; // r15
  __int64 v260; // r14
  __int64 v261; // rax
  __int64 v262; // rax
  _BOOL4 v263; // [rsp+1Ch] [rbp-784h]
  _BOOL4 v264; // [rsp+24h] [rbp-77Ch]
  __int64 v265; // [rsp+28h] [rbp-778h]
  __int64 v266; // [rsp+50h] [rbp-750h]
  __int64 v267; // [rsp+58h] [rbp-748h]
  __int64 v270; // [rsp+80h] [rbp-720h]
  __int64 v271; // [rsp+90h] [rbp-710h]
  __int64 v273; // [rsp+A8h] [rbp-6F8h]
  _QWORD *v274; // [rsp+B8h] [rbp-6E8h]
  void **v275; // [rsp+C0h] [rbp-6E0h]
  __int64 v276; // [rsp+D8h] [rbp-6C8h]
  _BYTE *j; // [rsp+D8h] [rbp-6C8h]
  unsigned int v278; // [rsp+E0h] [rbp-6C0h]
  unsigned int v279; // [rsp+E4h] [rbp-6BCh] BYREF
  const __m128i *v280; // [rsp+E8h] [rbp-6B8h] BYREF
  char *v281; // [rsp+F8h] [rbp-6A8h] BYREF
  char *v282; // [rsp+100h] [rbp-6A0h] BYREF
  char *v283; // [rsp+108h] [rbp-698h] BYREF
  _QWORD v284[4]; // [rsp+110h] [rbp-690h] BYREF
  __int64 *v285; // [rsp+130h] [rbp-670h] BYREF
  __int64 *v286; // [rsp+138h] [rbp-668h]
  __int64 v287; // [rsp+140h] [rbp-660h]
  _QWORD *v288[2]; // [rsp+150h] [rbp-650h] BYREF
  __int64 v289; // [rsp+160h] [rbp-640h]
  __int64 *v290; // [rsp+170h] [rbp-630h] BYREF
  __int64 *v291; // [rsp+178h] [rbp-628h]
  __int64 v292; // [rsp+180h] [rbp-620h]
  _QWORD *v293[4]; // [rsp+190h] [rbp-610h] BYREF
  __int64 *v294; // [rsp+1B0h] [rbp-5F0h] BYREF
  __int64 *v295; // [rsp+1B8h] [rbp-5E8h]
  __int64 v296; // [rsp+1C0h] [rbp-5E0h]
  _BYTE *v297; // [rsp+1D0h] [rbp-5D0h] BYREF
  _BYTE *v298; // [rsp+1D8h] [rbp-5C8h]
  _BYTE *v299; // [rsp+1E0h] [rbp-5C0h]
  __int64 v300; // [rsp+1F0h] [rbp-5B0h] BYREF
  __int64 v301; // [rsp+1F8h] [rbp-5A8h]
  __int64 v302; // [rsp+200h] [rbp-5A0h]
  unsigned int v303; // [rsp+208h] [rbp-598h]
  char **v304; // [rsp+210h] [rbp-590h] BYREF
  __int64 v305; // [rsp+218h] [rbp-588h] BYREF
  __int64 v306; // [rsp+220h] [rbp-580h]
  __int64 v307; // [rsp+228h] [rbp-578h]
  void **v308; // [rsp+230h] [rbp-570h]
  __int64 v309; // [rsp+240h] [rbp-560h] BYREF
  __int64 v310; // [rsp+248h] [rbp-558h] BYREF
  __int64 v311; // [rsp+250h] [rbp-550h]
  __int64 v312; // [rsp+258h] [rbp-548h]
  void **v313; // [rsp+260h] [rbp-540h]
  void *v314; // [rsp+270h] [rbp-530h] BYREF
  __int64 v315; // [rsp+278h] [rbp-528h] BYREF
  __int64 v316; // [rsp+280h] [rbp-520h]
  __int64 v317; // [rsp+288h] [rbp-518h]
  __int64 v318; // [rsp+290h] [rbp-510h]
  __int64 v319; // [rsp+298h] [rbp-508h]
  __int64 v320; // [rsp+2A0h] [rbp-500h]
  __int64 v321; // [rsp+2A8h] [rbp-4F8h]
  __int64 v322; // [rsp+2B0h] [rbp-4F0h] BYREF
  _BYTE *v323; // [rsp+2B8h] [rbp-4E8h]
  _BYTE *v324; // [rsp+2C0h] [rbp-4E0h]
  __int64 v325; // [rsp+2C8h] [rbp-4D8h]
  int v326; // [rsp+2D0h] [rbp-4D0h]
  _BYTE v327[40]; // [rsp+2D8h] [rbp-4C8h] BYREF
  const __m128i **v328; // [rsp+300h] [rbp-4A0h] BYREF
  unsigned int *v329; // [rsp+308h] [rbp-498h]
  unsigned int *v330; // [rsp+310h] [rbp-490h]
  __int64 v331; // [rsp+318h] [rbp-488h]
  int v332; // [rsp+320h] [rbp-480h]
  _BYTE v333[40]; // [rsp+328h] [rbp-478h] BYREF
  __m128i v334; // [rsp+350h] [rbp-450h] BYREF
  _BYTE *v335; // [rsp+360h] [rbp-440h]
  __int64 v336; // [rsp+368h] [rbp-438h]
  int v337; // [rsp+370h] [rbp-430h] BYREF
  _BYTE v338[56]; // [rsp+378h] [rbp-428h] BYREF
  char *v339; // [rsp+3B0h] [rbp-3F0h] BYREF
  __int64 v340; // [rsp+3B8h] [rbp-3E8h]
  __int64 v341; // [rsp+3C0h] [rbp-3E0h]
  unsigned int v342; // [rsp+3C8h] [rbp-3D8h]
  char v343[8]; // [rsp+3D0h] [rbp-3D0h] BYREF
  _QWORD *v344; // [rsp+3D8h] [rbp-3C8h]
  unsigned int v345; // [rsp+3E8h] [rbp-3B8h]
  char v346; // [rsp+3F0h] [rbp-3B0h]
  char v347; // [rsp+3F9h] [rbp-3A7h]
  char v348[376]; // [rsp+408h] [rbp-398h] BYREF
  __int64 v349; // [rsp+580h] [rbp-220h]
  char *v350; // [rsp+590h] [rbp-210h] BYREF
  __int64 v351; // [rsp+598h] [rbp-208h] BYREF
  __int64 *v352; // [rsp+5A0h] [rbp-200h] BYREF
  __int64 v353; // [rsp+5A8h] [rbp-1F8h]
  __int64 v354; // [rsp+5B0h] [rbp-1F0h]
  _QWORD v355[2]; // [rsp+5B8h] [rbp-1E8h] BYREF
  unsigned int v356; // [rsp+5C8h] [rbp-1D8h]
  char v357; // [rsp+5D0h] [rbp-1D0h]
  char v358; // [rsp+5D9h] [rbp-1C7h]
  _BYTE v359[376]; // [rsp+5E8h] [rbp-1B8h] BYREF
  __int64 v360; // [rsp+760h] [rbp-40h]

  v19 = *(char ***)(a1 + 32);
  v280 = (const __m128i *)a1;
  v279 = a2;
  v281 = *v19;
  v270 = **(_QWORD **)(a1 + 8);
  if ( !a3 && a2 <= 1 || (a4 == 1 || a4 % a2) && !(unsigned __int8)sub_1B12B90(a1, a2, 0, 1, a5, a6, a15, a16, a17, 1) )
    return 0;
  if ( a15 )
  {
    sub_1465150(a15, (__int64)v280);
    sub_1465150(a15, v270);
  }
  if ( a3 == a2 )
  {
    v259 = *(_QWORD *)v280[2].m128i_i64[0];
    sub_13FD840(&v328, (__int64)v280);
    sub_15C9090((__int64)&v334, &v328);
    sub_15CA330((__int64)&v350, (__int64)"loop-unroll-and-jam", (__int64)"FullyUnrolled", 13, &v334, v259);
    sub_15CAB20((__int64)&v350, "completely unroll and jammed loop with ", 0x27u);
    sub_15C9C50((__int64)&v339, "UnrollCount", 11, a3);
    v260 = sub_17C2270((__int64)&v350, (__int64)&v339);
    sub_15CAB20(v260, " iterations", 0xBu);
    sub_143AA50(a18, v260);
    sub_2240A30(v343);
    sub_2240A30(&v339);
    v350 = (char *)&unk_49ECF68;
    sub_1897B80((__int64)v359);
    if ( v328 )
      sub_161E7C0((__int64)&v328, (__int64)v328);
    goto LABEL_13;
  }
  v20 = *a18;
  v328 = &v280;
  v329 = &v279;
  if ( a4 == 1 )
  {
    v258 = sub_15E0530(v20);
    if ( !sub_1602790(v258) )
    {
      v261 = sub_15E0530(*a18);
      v262 = sub_16033E0(v261);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v262 + 48LL))(v262) )
        goto LABEL_13;
    }
    sub_1B06350((__int64)&v350, (__int64)&v328);
    sub_15CAB20((__int64)&v350, " with run-time trip count", 0x19u);
    sub_18980B0((__int64)&v339, (__int64)&v350);
    v349 = v360;
    v339 = (char *)&unk_49ECF98;
  }
  else
  {
    v21 = sub_15E0530(v20);
    if ( !sub_1602790(v21) )
    {
      v256 = sub_15E0530(*a18);
      v257 = sub_16033E0(v256);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v257 + 48LL))(v257) )
        goto LABEL_13;
    }
    sub_1B06350((__int64)&v350, (__int64)&v328);
    sub_15CAB20((__int64)&v350, " with ", 6u);
    sub_15C9C50((__int64)&v334, "TripMultiple", 12, a4);
    v22 = sub_17C2270((__int64)&v350, (__int64)&v334);
    sub_15CAB20(v22, " trips per branch", 0x11u);
    sub_18980B0((__int64)&v339, v22);
    v349 = *(_QWORD *)(v22 + 464);
    v339 = (char *)&unk_49ECF98;
    sub_2240A30(&v337);
    sub_2240A30(&v334);
  }
  v350 = (char *)&unk_49ECF68;
  sub_1897B80((__int64)v359);
  sub_143AA50(a18, (__int64)&v339);
  v339 = (char *)&unk_49ECF68;
  sub_1897B80((__int64)v348);
LABEL_13:
  v265 = sub_13FC520((__int64)v280);
  v23 = sub_13FCB50((__int64)v280);
  v24 = sub_157EBA0(v23);
  v25 = v24;
  if ( *(_BYTE *)(v24 + 16) != 26 )
    BUG();
  v264 = sub_1377F70((__int64)&v280[3].m128i_i64[1], *(_QWORD *)(v24 - 24));
  v267 = *(_QWORD *)(v25 - 24LL * v264 - 24);
  v26 = sub_13FCB50(v270);
  v27 = sub_157EBA0(v26);
  v28 = sub_15F4DF0(v27, 0);
  v263 = sub_1377F70(v270 + 56, v28);
  v323 = v327;
  v324 = v327;
  v329 = (unsigned int *)v333;
  v330 = (unsigned int *)v333;
  v322 = 0;
  v325 = 4;
  v326 = 0;
  v328 = 0;
  v331 = 4;
  v332 = 0;
  v334.m128i_i64[0] = 0;
  v334.m128i_i64[1] = (__int64)v338;
  v335 = v338;
  v336 = 4;
  v337 = 0;
  sub_1B04FC0((__int64)v280, v270, (__int64)&v328, (__int64)&v322, (__int64)&v334, a16);
  memset(v284, 0, 24);
  v285 = 0;
  v286 = 0;
  v287 = 0;
  v288[0] = 0;
  v288[1] = 0;
  v289 = 0;
  v290 = 0;
  v291 = 0;
  v292 = 0;
  memset(v293, 0, 24);
  v294 = 0;
  v295 = 0;
  v296 = 0;
  sub_15CE600((__int64)v284, &v281);
  v350 = (char *)sub_13FC520(v270);
  sub_1B065E0((__int64)&v285, &v350);
  v350 = **(char ***)(v270 + 32);
  sub_1B065E0((__int64)v288, &v350);
  v350 = (char *)sub_13F9E70(v270);
  sub_1B065E0((__int64)&v290, &v350);
  v350 = (char *)sub_13FA090(v270);
  sub_1B065E0((__int64)v293, &v350);
  v350 = (char *)sub_13F9E70((__int64)v280);
  sub_1B065E0((__int64)&v294, &v350);
  v339 = 0;
  v29 = sub_1454B60(0x56u);
  v342 = v29;
  if ( v29 )
  {
    v340 = sub_22077B0((unsigned __int64)v29 << 6);
    sub_1954940((__int64)&v339);
  }
  else
  {
    v340 = 0;
    v341 = 0;
  }
  v346 = 0;
  v347 = 1;
  v32 = sub_13FC520(v270);
  v33 = sub_157EBA0(v32);
  v350 = 0;
  v351 = 0;
  v352 = 0;
  sub_1B05D90((__int64)v281, v23, (__int64)&v334, (__int64)&v350, (__int64)&v334);
  v34 = v350;
  v35 = (char *)v351;
  v36 = *(_QWORD *)(v33 + 40);
  if ( v350 != (char *)v351 )
  {
    do
    {
      v37 = (_QWORD *)*((_QWORD *)v35 - 1);
      if ( v36 != v37[5] )
        sub_15F22F0(v37, v33);
      v35 -= 8;
    }
    while ( v34 != v35 );
    v35 = v350;
  }
  if ( v35 )
    j_j___libc_free_0(v35, (char *)v352 - v35);
  sub_1AFCDB0((__int64)&v314, (__int64)v280);
  *((_QWORD *)&v38 + 1) = a6;
  *(_QWORD *)&v38 = &v314;
  sub_13FF3D0(v38);
  v266 = v320;
  v271 = v319;
  if ( (unsigned __int8)sub_1626D30(*((_QWORD *)v281 + 7)) )
  {
    v39 = v280[2].m128i_i64[0];
    if ( v280[2].m128i_i64[1] != v39 )
    {
      v276 = v280[2].m128i_i64[1];
      do
      {
        v40 = *(_QWORD *)(*(_QWORD *)v39 + 48LL);
        for ( i = *(_QWORD *)v39 + 40LL; v40 != i; v40 = *(_QWORD *)(v40 + 8) )
        {
          while ( 1 )
          {
            if ( !v40 )
              BUG();
            if ( *(_BYTE *)(v40 - 8) != 78 )
              break;
            v46 = *(_QWORD *)(v40 - 48);
            if ( *(_BYTE *)(v46 + 16)
              || (*(_BYTE *)(v46 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v46 + 36) - 35) > 3 )
            {
              break;
            }
            v40 = *(_QWORD *)(v40 + 8);
            if ( v40 == i )
              goto LABEL_44;
          }
          v42 = sub_15C70A0(v40 + 24);
          if ( v42 )
          {
            v43 = sub_1AFCF60(v42, v279);
            sub_15C7080(&v350, v43);
            if ( (char **)(v40 + 24) == &v350 )
            {
              if ( v350 )
                sub_161E7C0((__int64)&v350, (__int64)v350);
            }
            else
            {
              v44 = *(_QWORD *)(v40 + 24);
              if ( v44 )
                sub_161E7C0(v40 + 24, v44);
              v45 = (unsigned __int8 *)v350;
              *(_QWORD *)(v40 + 24) = v350;
              if ( v45 )
                sub_1623210((__int64)&v350, v45, v40 + 24);
            }
          }
        }
LABEL_44:
        v39 += 8;
      }
      while ( v276 != v39 );
    }
  }
  if ( v279 == 1 )
    goto LABEL_153;
  v278 = 1;
  v47 = (void **)&v339;
  do
  {
    v297 = 0;
    v298 = 0;
    v299 = 0;
    v300 = 0;
    v301 = 0;
    v302 = 0;
    v303 = 0;
    if ( v266 == v271 )
      goto LABEL_119;
    v273 = v266;
    do
    {
      v350 = 0;
      LODWORD(v353) = 128;
      v351 = sub_22077B0(0x2000);
      sub_1954940((__int64)&v350);
      v357 = 0;
      LODWORD(v304) = v278;
      v309 = (__int64)".";
      v358 = 1;
      v310 = (__int64)v304;
      LOWORD(v311) = 2307;
      v48 = sub_1AB5760(*(_QWORD *)(v273 - 8), (__int64)&v350, &v309, 0, 0, 0);
      v282 = (char *)v48;
      v49 = (__int64 *)(*((_QWORD *)v281 + 7) + 72LL);
      sub_15E01D0((__int64)v49, v48);
      v50 = *v49;
      v51 = *(_QWORD *)(v48 + 24);
      *(_QWORD *)(v48 + 32) = v49;
      v50 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v48 + 24) = v50 | v51 & 7;
      *(_QWORD *)(v50 + 8) = v48 + 24;
      *v49 = *v49 & 7 | (v48 + 24);
      if ( sub_183E920((__int64)&v328, *(_QWORD *)(v273 - 8)) )
      {
        sub_1400330((__int64)v280, (__int64)v282, a6);
        v52 = *(_QWORD *)(v273 - 8);
        if ( v52 == *(_QWORD *)v284[0] )
        {
          sub_15CE600((__int64)v284, &v282);
          v52 = *(_QWORD *)(v273 - 8);
          if ( *v285 != v52 )
            goto LABEL_52;
        }
        else if ( *v285 != v52 )
        {
          goto LABEL_52;
        }
        sub_15CE600((__int64)&v285, &v282);
        v52 = *(_QWORD *)(v273 - 8);
LABEL_52:
        if ( v278 != 1 )
          goto LABEL_53;
        goto LABEL_286;
      }
      if ( !sub_183E920((__int64)&v322, *(_QWORD *)(v273 - 8)) )
      {
        sub_183E920((__int64)&v334, *(_QWORD *)(v273 - 8));
        sub_1400330((__int64)v280, (__int64)v282, a6);
        v52 = *(_QWORD *)(v273 - 8);
        if ( v52 == *v293[0] )
        {
          sub_15CE600((__int64)v293, &v282);
          v52 = *(_QWORD *)(v273 - 8);
        }
        if ( *v294 == v52 )
        {
          sub_15CE600((__int64)&v294, &v282);
          v52 = *(_QWORD *)(v273 - 8);
        }
        goto LABEL_52;
      }
      sub_1400330(v270, (__int64)v282, a6);
      v52 = *(_QWORD *)(v273 - 8);
      if ( v52 == *v288[0] )
      {
        sub_15CE600((__int64)v288, &v282);
        v52 = *(_QWORD *)(v273 - 8);
      }
      if ( *v290 != v52 )
        goto LABEL_52;
      sub_15CE600((__int64)&v290, &v282);
      v52 = *(_QWORD *)(v273 - 8);
      if ( v278 != 1 )
      {
LABEL_53:
        v53 = sub_1B06B50((__int64)v47, v52);
        v309 = 6;
        v310 = 0;
        v54 = v53;
        v55 = v53[2];
        v311 = v55;
        if ( v55 != -8 && v55 != 0 && v55 != -16 )
          sub_1649AC0((unsigned __int64 *)&v309, *v54 & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_56;
      }
LABEL_286:
      v311 = v52;
      v309 = 6;
      v310 = 0;
      if ( v52 != -8 && v52 != 0 && v52 != -16 )
        sub_164C220((__int64)&v309);
LABEL_56:
      v56 = v282;
      v57 = v303;
      v283 = v282;
      if ( !v303 )
      {
        ++v300;
LABEL_415:
        v57 = 2 * v303;
LABEL_416:
        sub_176F940((__int64)&v300, v57);
        sub_176A9A0((__int64)&v300, (__int64 *)&v283, &v304);
        v59 = v304;
        v56 = v283;
        v241 = v302 + 1;
        goto LABEL_411;
      }
      v58 = (v303 - 1) & (((unsigned int)v282 >> 9) ^ ((unsigned int)v282 >> 4));
      v59 = (char **)(v301 + 16LL * v58);
      v60 = *v59;
      if ( v282 == *v59 )
        goto LABEL_58;
      v239 = 1;
      v240 = 0;
      while ( v60 != (char *)-8LL )
      {
        if ( v60 == (char *)-16LL && !v240 )
          v240 = v59;
        v58 = (v303 - 1) & (v239 + v58);
        v59 = (char **)(v301 + 16LL * v58);
        v60 = *v59;
        if ( v282 == *v59 )
          goto LABEL_58;
        ++v239;
      }
      if ( v240 )
        v59 = v240;
      ++v300;
      v241 = v302 + 1;
      if ( 4 * ((int)v302 + 1) >= 3 * v303 )
        goto LABEL_415;
      if ( v303 - HIDWORD(v302) - v241 <= v303 >> 3 )
        goto LABEL_416;
LABEL_411:
      LODWORD(v302) = v241;
      if ( *v59 != (char *)-8LL )
        --HIDWORD(v302);
      *v59 = v56;
      v59[1] = 0;
LABEL_58:
      v61 = v311;
      v62 = v311 == -8;
      v59[1] = (char *)v311;
      if ( v61 != 0 && !v62 && v61 != -16 )
        sub_1649B30(&v309);
      v63 = v282;
      v64 = *(_QWORD *)(v273 - 8);
      v305 = 2;
      v306 = 0;
      v307 = v64;
      if ( v64 != 0 && v64 != -8 && v64 != -16 )
        sub_164C220((__int64)&v305);
      v308 = v47;
      v304 = (char **)&unk_49E6B50;
      v65 = sub_12E4800((__int64)v47, (__int64)&v304, &v309);
      v66 = (unsigned __int64 *)v309;
      if ( !v65 )
      {
        v67 = v342;
        ++v339;
        v68 = v341 + 1;
        if ( 4 * ((int)v341 + 1) >= 3 * v342 )
        {
          v67 = 2 * v342;
        }
        else if ( v342 - HIDWORD(v341) - v68 > v342 >> 3 )
        {
LABEL_67:
          LODWORD(v341) = v68;
          v310 = 2;
          v311 = 0;
          v312 = -8;
          v313 = 0;
          if ( v66[3] != -8 )
            --HIDWORD(v341);
          v309 = (__int64)&unk_49EE2B0;
          sub_1455FA0((__int64)&v310);
          sub_1B04A00(v66 + 1, &v305);
          v69 = (unsigned __int64)v308;
          v66[5] = 6;
          v66[6] = 0;
          v66[4] = v69;
          v66[7] = 0;
          goto LABEL_70;
        }
        sub_12E48B0((__int64)v47, v67);
        sub_12E4800((__int64)v47, (__int64)&v304, &v309);
        v66 = (unsigned __int64 *)v309;
        v68 = v341 + 1;
        goto LABEL_67;
      }
LABEL_70:
      v304 = (char **)&unk_49EE2B0;
      if ( v307 != 0 && v307 != -8 && v307 != -16 )
        sub_1649B30(&v305);
      v70 = (char *)v66[7];
      if ( v63 != v70 )
      {
        if ( v70 + 8 != 0 && v70 != 0 && v70 != (char *)-16LL )
          sub_1649B30(v66 + 5);
        v66[7] = (unsigned __int64)v63;
        if ( v63 + 8 != 0 && v63 != 0 && v63 != (char *)-16LL )
          sub_164C220((__int64)(v66 + 5));
      }
      v71 = (_QWORD *)v351;
      v72 = (_QWORD *)(v351 + ((unsigned __int64)(unsigned int)v353 << 6));
      if ( (_DWORD)v352 )
      {
        v305 = 2;
        v306 = 0;
        v307 = -8;
        v304 = (char **)&unk_49E6B50;
        v308 = 0;
        v310 = 2;
        v311 = 0;
        v312 = -16;
        v313 = 0;
        if ( v72 != (_QWORD *)v351 )
        {
          do
          {
            v206 = v71[3];
            if ( v206 != -16 && v206 != -8 )
              break;
            v71 += 8;
          }
          while ( v72 != v71 );
        }
        v309 = (__int64)&unk_49EE2B0;
        sub_1455FA0((__int64)&v310);
        v304 = (char **)&unk_49EE2B0;
        sub_1455FA0((__int64)&v305);
        v274 = (_QWORD *)(v351 + ((unsigned __int64)(unsigned int)v353 << 6));
        if ( v274 != v71 )
        {
          if ( v278 != 1 )
          {
LABEL_304:
            v207 = v71[3];
            v310 = 2;
            v311 = 0;
            if ( v207 )
            {
              v312 = v207;
              if ( v207 != -8 && v207 != -16 )
                sub_164C220((__int64)&v310);
            }
            else
            {
              v312 = 0;
            }
            v313 = v47;
            v309 = (__int64)&unk_49E6B50;
            v208 = sub_12E4800((__int64)v47, (__int64)&v309, &v304);
            v209 = v304;
            if ( v208 )
            {
              v210 = v312;
              goto LABEL_310;
            }
            v229 = v342;
            ++v339;
            v230 = v341 + 1;
            if ( 4 * ((int)v341 + 1) >= 3 * v342 )
            {
              v229 = 2 * v342;
            }
            else if ( v342 - HIDWORD(v341) - v230 > v342 >> 3 )
            {
              goto LABEL_360;
            }
            sub_12E48B0((__int64)v47, v229);
            sub_12E4800((__int64)v47, (__int64)&v309, &v304);
            v209 = v304;
            v230 = v341 + 1;
LABEL_360:
            LODWORD(v341) = v230;
            v231 = v209[3];
            v210 = v312;
            if ( v231 == (char *)-8LL )
            {
              if ( v312 != -8 )
                goto LABEL_365;
            }
            else
            {
              --HIDWORD(v341);
              if ( v231 != (char *)v312 )
              {
                if ( v231 && v231 != (char *)-16LL )
                {
                  sub_1649B30(v209 + 1);
                  v210 = v312;
                }
LABEL_365:
                v209[3] = (char *)v210;
                if ( v210 != 0 && v210 != -8 && v210 != -16 )
                  sub_1649AC0((unsigned __int64 *)v209 + 1, v310 & 0xFFFFFFFFFFFFFFF8LL);
                v210 = v312;
              }
            }
            v232 = v313;
            v209[5] = (char *)6;
            v209[6] = 0;
            v209[4] = (char *)v232;
            v209[7] = 0;
LABEL_310:
            v309 = (__int64)&unk_49EE2B0;
            if ( v210 != -8 && v210 != 0 && v210 != -16 )
              sub_1649B30(&v310);
            v211 = v209[7];
            goto LABEL_314;
          }
          while ( 2 )
          {
            v211 = (char *)v71[3];
LABEL_314:
            v212 = (char **)v71[7];
            v304 = v212;
            if ( v303 )
            {
              v213 = (v303 - 1) & (((unsigned int)v212 >> 9) ^ ((unsigned int)v212 >> 4));
              v214 = (void **)(v301 + 16LL * v213);
              v215 = (char **)*v214;
              if ( v212 == *v214 )
              {
LABEL_316:
                v214[1] = v211;
                v216 = v71[3];
                if ( v216 )
                  goto LABEL_317;
LABEL_384:
                v310 = 2;
                v311 = 0;
                v312 = 0;
                goto LABEL_320;
              }
              v233 = 1;
              v234 = 0;
              while ( v215 != (char **)-8LL )
              {
                if ( v215 != (char **)-16LL || v234 )
                  v214 = v234;
                v213 = (v303 - 1) & (v233 + v213);
                v215 = *(char ***)(v301 + 16LL * v213);
                if ( v212 == v215 )
                {
                  v214 = (void **)(v301 + 16LL * v213);
                  goto LABEL_316;
                }
                ++v233;
                v234 = v214;
                v214 = (void **)(v301 + 16LL * v213);
              }
              if ( v234 )
                v214 = v234;
              ++v300;
              v235 = v302 + 1;
              if ( 4 * ((int)v302 + 1) < 3 * v303 )
              {
                if ( v303 - HIDWORD(v302) - v235 <= v303 >> 3 )
                {
                  sub_176F940((__int64)&v300, v303);
                  sub_176A9A0((__int64)&v300, (__int64 *)&v304, &v309);
                  v214 = (void **)v309;
                  v212 = v304;
                  v235 = v302 + 1;
                }
LABEL_381:
                LODWORD(v302) = v235;
                if ( *v214 != (void *)-8LL )
                  --HIDWORD(v302);
                v214[1] = 0;
                *v214 = v212;
                v214[1] = v211;
                v216 = v71[3];
                if ( !v216 )
                  goto LABEL_384;
LABEL_317:
                v310 = 2;
                v311 = 0;
                v312 = v216;
                if ( v216 != -16 && v216 != -8 )
                  sub_164C220((__int64)&v310);
LABEL_320:
                v313 = v47;
                v309 = (__int64)&unk_49E6B50;
                if ( !v342 )
                {
                  ++v339;
                  goto LABEL_322;
                }
                v218 = v312;
                v225 = (v342 - 1) & (((unsigned int)v312 >> 9) ^ ((unsigned int)v312 >> 4));
                v217 = (char **)(v340 + ((unsigned __int64)v225 << 6));
                v226 = v217[3];
                if ( (char *)v312 != v226 )
                {
                  v227 = 1;
                  v228 = 0;
                  while ( v226 != (char *)-8LL )
                  {
                    if ( v226 != (char *)-16LL || v228 )
                      v217 = v228;
                    v225 = (v342 - 1) & (v227 + v225);
                    v226 = *(char **)(v340 + ((unsigned __int64)v225 << 6) + 24);
                    if ( (char *)v312 == v226 )
                    {
                      v217 = (char **)(v340 + ((unsigned __int64)v225 << 6));
                      goto LABEL_333;
                    }
                    ++v227;
                    v228 = v217;
                    v217 = (char **)(v340 + ((unsigned __int64)v225 << 6));
                  }
                  if ( v228 )
                    v217 = v228;
                  ++v339;
                  v219 = v341 + 1;
                  if ( 4 * ((int)v341 + 1) >= 3 * v342 )
                  {
LABEL_322:
                    sub_12E48B0((__int64)v47, 2 * v342);
                    sub_12E4800((__int64)v47, (__int64)&v309, &v304);
                    v217 = v304;
                    v218 = v312;
                    v219 = v341 + 1;
                  }
                  else if ( v342 - HIDWORD(v341) - v219 <= v342 >> 3 )
                  {
                    sub_12E48B0((__int64)v47, v342);
                    sub_12E4800((__int64)v47, (__int64)&v309, &v304);
                    v217 = v304;
                    v218 = v312;
                    v219 = v341 + 1;
                  }
                  LODWORD(v341) = v219;
                  v220 = v217[3];
                  if ( v220 == (char *)-8LL )
                  {
                    if ( v218 != -8 )
                      goto LABEL_328;
                  }
                  else
                  {
                    --HIDWORD(v341);
                    if ( v220 != (char *)v218 )
                    {
                      if ( v220 != (char *)-16LL && v220 )
                      {
                        sub_1649B30(v217 + 1);
                        v218 = v312;
                      }
LABEL_328:
                      v217[3] = (char *)v218;
                      if ( v218 != -8 && v218 != 0 && v218 != -16 )
                        sub_1649AC0((unsigned __int64 *)v217 + 1, v310 & 0xFFFFFFFFFFFFFFF8LL);
                      v218 = v312;
                    }
                  }
                  v221 = v313;
                  v217[5] = (char *)6;
                  v217[6] = 0;
                  v217[4] = (char *)v221;
                  v217[7] = 0;
                }
LABEL_333:
                v309 = (__int64)&unk_49EE2B0;
                if ( v218 != -8 && v218 != 0 && v218 != -16 )
                  sub_1649B30(&v310);
                v222 = v217[7];
                v223 = (char *)v71[7];
                if ( v222 != v223 )
                {
                  if ( v222 != 0 && v222 + 8 != 0 && v222 != (char *)-16LL )
                  {
                    sub_1649B30(v217 + 5);
                    v223 = (char *)v71[7];
                  }
                  v217[7] = v223;
                  if ( v223 != 0 && v223 + 8 != 0 && v223 != (char *)-16LL )
                    sub_1649AC0((unsigned __int64 *)v217 + 5, v71[5] & 0xFFFFFFFFFFFFFFF8LL);
                }
                for ( v71 += 8; v72 != v71; v71 += 8 )
                {
                  v224 = v71[3];
                  if ( v224 != -16 && v224 != -8 )
                    break;
                }
                if ( v71 == v274 )
                  goto LABEL_81;
                if ( v278 != 1 )
                  goto LABEL_304;
                continue;
              }
            }
            else
            {
              ++v300;
            }
            break;
          }
          sub_176F940((__int64)&v300, 2 * v303);
          if ( !v303 )
          {
            LODWORD(v302) = v302 + 1;
            BUG();
          }
          v212 = v304;
          v236 = (v303 - 1) & (((unsigned int)v304 >> 9) ^ ((unsigned int)v304 >> 4));
          v214 = (void **)(v301 + 16LL * v236);
          v237 = *v214;
          if ( v304 == *v214 )
          {
LABEL_391:
            v235 = v302 + 1;
          }
          else
          {
            v248 = 1;
            v249 = 0;
            while ( v237 != (void *)-8LL )
            {
              if ( !v249 && v237 == (void *)-16LL )
                v249 = v214;
              v236 = (v303 - 1) & (v248 + v236);
              v214 = (void **)(v301 + 16LL * v236);
              v237 = *v214;
              if ( v304 == *v214 )
                goto LABEL_391;
              ++v248;
            }
            v235 = v302 + 1;
            if ( v249 )
              v214 = v249;
          }
          goto LABEL_381;
        }
      }
LABEL_81:
      v73 = v298;
      if ( v298 == v299 )
      {
        sub_1292090((__int64)&v297, v298, &v282);
      }
      else
      {
        if ( v298 )
        {
          *(_QWORD *)v298 = v282;
          v73 = v298;
        }
        v298 = v73 + 8;
      }
      v74 = *(_QWORD *)(v273 - 8);
      if ( v74 == *(_QWORD *)v284[0] )
      {
        sub_1B06EB0(a16, (__int64)v282, v285[v278 - 1]);
        goto LABEL_92;
      }
      if ( v74 == *v288[0] )
      {
        v238 = v290;
        goto LABEL_397;
      }
      if ( v74 == *v293[0] )
      {
        v238 = v294;
LABEL_397:
        sub_1B06EB0(a16, (__int64)v282, v238[v278 - 1]);
        goto LABEL_92;
      }
      v75 = *(unsigned int *)(a16 + 48);
      if ( !(_DWORD)v75 )
        goto LABEL_513;
      v76 = *(_QWORD *)(a16 + 32);
      v77 = (v75 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
      v78 = (__int64 *)(v76 + 16LL * v77);
      v79 = *v78;
      if ( v74 != *v78 )
      {
        v246 = 1;
        while ( v79 != -8 )
        {
          v247 = v246 + 1;
          v77 = (v75 - 1) & (v246 + v77);
          v78 = (__int64 *)(v76 + 16LL * v77);
          v79 = *v78;
          if ( v74 == *v78 )
            goto LABEL_90;
          v246 = v247;
        }
LABEL_513:
        BUG();
      }
LABEL_90:
      if ( v78 == (__int64 *)(v76 + 16 * v75) )
        goto LABEL_513;
      v80 = sub_1B06B50((__int64)v47, **(_QWORD **)(v78[1] + 8));
      sub_1B06EB0(a16, (__int64)v282, v80[2]);
LABEL_92:
      if ( v357 )
      {
        if ( v356 )
        {
          v203 = (_QWORD *)v355[0];
          v204 = v355[0] + 16LL * v356;
          do
          {
            if ( *v203 != -8 && *v203 != -4 )
            {
              v205 = v203[1];
              if ( v205 )
                sub_161E7C0((__int64)(v203 + 1), v205);
            }
            v203 += 2;
          }
          while ( (_QWORD *)v204 != v203 );
        }
        j___libc_free_0(v355[0]);
      }
      if ( (_DWORD)v353 )
      {
        v81 = (_QWORD *)v351;
        v305 = 2;
        v306 = 0;
        v82 = (_QWORD *)(v351 + ((unsigned __int64)(unsigned int)v353 << 6));
        v307 = -8;
        v83 = -8;
        v304 = (char **)&unk_49E6B50;
        v308 = 0;
        v310 = 2;
        v311 = 0;
        v312 = -16;
        v309 = (__int64)&unk_49E6B50;
        v313 = 0;
        while ( 1 )
        {
          v84 = v81[3];
          if ( v84 != v83 )
          {
            v83 = v312;
            if ( v84 != v312 )
            {
              v85 = v81[7];
              if ( v85 != 0 && v85 != -8 && v85 != -16 )
              {
                sub_1649B30(v81 + 5);
                v84 = v81[3];
              }
              v83 = v84;
            }
          }
          *v81 = &unk_49EE2B0;
          if ( v83 != 0 && v83 != -8 && v83 != -16 )
            sub_1649B30(v81 + 1);
          v81 += 8;
          if ( v82 == v81 )
            break;
          v83 = v307;
        }
        v309 = (__int64)&unk_49EE2B0;
        sub_1455FA0((__int64)&v310);
        v304 = (char **)&unk_49EE2B0;
        sub_1455FA0((__int64)&v305);
      }
      j___libc_free_0(v351);
      v273 -= 8;
    }
    while ( v271 != v273 );
    v86 = v297;
    for ( j = v298; j != v86; v86 += 8 )
    {
      v87 = *(_QWORD *)(*(_QWORD *)v86 + 48LL);
      for ( k = *(_QWORD *)v86 + 40LL; k != v87; v87 = *(_QWORD *)(v87 + 8) )
      {
        while ( 1 )
        {
          if ( !v87 )
          {
            sub_1AFD1D0(0, (__int64)v47);
            BUG();
          }
          sub_1AFD1D0(v87 - 24, (__int64)v47);
          if ( *(_BYTE *)(v87 - 8) == 78 )
          {
            v89 = *(_QWORD *)(v87 - 48);
            if ( !*(_BYTE *)(v89 + 16) && (*(_BYTE *)(v89 + 33) & 0x20) != 0 && *(_DWORD *)(v89 + 36) == 4 )
              break;
          }
          v87 = *(_QWORD *)(v87 + 8);
          if ( k == v87 )
            goto LABEL_118;
        }
        sub_14CE830(a17, v87 - 24);
      }
LABEL_118:
      ;
    }
LABEL_119:
    v91 = sub_157F280(*(_QWORD *)(v284[0] + 8LL * v278));
    if ( v90 == v91 )
      goto LABEL_150;
    v275 = v47;
    v92 = v90;
    while ( 2 )
    {
      v93 = 0x17FFFFFFE8LL;
      v94 = *(_BYTE *)(v91 + 23) & 0x40;
      v95 = *(_DWORD *)(v91 + 20) & 0xFFFFFFF;
      if ( v95 )
      {
        v96 = 24LL * *(unsigned int *)(v91 + 56) + 8;
        v97 = 0;
        do
        {
          v98 = v91 - 24LL * v95;
          if ( v94 )
            v98 = *(_QWORD *)(v91 - 8);
          if ( v294[v278] == *(_QWORD *)(v98 + v96) )
          {
            v93 = 24 * v97;
            goto LABEL_128;
          }
          ++v97;
          v96 += 8;
        }
        while ( v95 != (_DWORD)v97 );
        v93 = 0x17FFFFFFE8LL;
      }
LABEL_128:
      if ( v94 )
        v99 = *(_QWORD *)(v91 - 8);
      else
        v99 = v91 - 24LL * v95;
      v100 = *(_QWORD *)(v99 + v93);
      v101 = v303;
      v309 = v100;
      if ( !v303 )
      {
        ++v300;
LABEL_427:
        v101 = 2 * v303;
LABEL_428:
        sub_176F940((__int64)&v300, v101);
        sub_176A9A0((__int64)&v300, &v309, &v350);
        v243 = v350;
        v245 = v309;
        v244 = v302 + 1;
        goto LABEL_423;
      }
      v102 = (v303 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
      v103 = (char *)(v301 + 16LL * v102);
      v104 = *(_QWORD *)v103;
      if ( *(_QWORD *)v103 == v100 )
      {
LABEL_132:
        v105 = *((_QWORD *)v103 + 1);
        if ( !v105 )
          v105 = v100;
        goto LABEL_134;
      }
      v242 = 1;
      v243 = 0;
      while ( v104 != -8 )
      {
        if ( !v243 && v104 == -16 )
          v243 = v103;
        v102 = (v303 - 1) & (v242 + v102);
        v103 = (char *)(v301 + 16LL * v102);
        v104 = *(_QWORD *)v103;
        if ( v100 == *(_QWORD *)v103 )
          goto LABEL_132;
        ++v242;
      }
      if ( !v243 )
        v243 = v103;
      ++v300;
      v244 = v302 + 1;
      if ( 4 * ((int)v302 + 1) >= 3 * v303 )
        goto LABEL_427;
      v245 = v100;
      if ( v303 - HIDWORD(v302) - v244 <= v303 >> 3 )
        goto LABEL_428;
LABEL_423:
      LODWORD(v302) = v244;
      if ( *(_QWORD *)v243 != -8 )
        --HIDWORD(v302);
      *(_QWORD *)v243 = v245;
      v105 = v100;
      *((_QWORD *)v243 + 1) = 0;
      v94 = *(_BYTE *)(v91 + 23) & 0x40;
LABEL_134:
      if ( v94 )
        v106 = *(_QWORD *)(v91 - 8);
      else
        v106 = v91 - 24LL * (*(_DWORD *)(v91 + 20) & 0xFFFFFFF);
      *(_QWORD *)(v106 + 24LL * *(unsigned int *)(v91 + 56) + 8) = v285[v278 - 1];
      if ( (*(_BYTE *)(v91 + 23) & 0x40) != 0 )
        v107 = *(__int64 **)(v91 - 8);
      else
        v107 = (__int64 *)(v91 - 24LL * (*(_DWORD *)(v91 + 20) & 0xFFFFFFF));
      if ( *v107 )
      {
        v108 = v107[1];
        v109 = v107[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v109 = v108;
        if ( v108 )
          *(_QWORD *)(v108 + 16) = *(_QWORD *)(v108 + 16) & 3LL | v109;
      }
      *v107 = v105;
      if ( v105 )
      {
        v110 = *(_QWORD *)(v105 + 8);
        v107[1] = v110;
        if ( v110 )
          *(_QWORD *)(v110 + 16) = (unsigned __int64)(v107 + 1) | *(_QWORD *)(v110 + 16) & 3LL;
        v107[2] = (v105 + 8) | v107[2] & 3;
        *(_QWORD *)(v105 + 8) = v107;
      }
      sub_15F5350(v91, 1u, 1);
      v111 = *(_QWORD *)(v91 + 32);
      if ( !v111 )
        BUG();
      v91 = 0;
      if ( *(_BYTE *)(v111 - 8) == 77 )
        v91 = v111 - 24;
      if ( v92 != v91 )
        continue;
      break;
    }
    v47 = v275;
LABEL_150:
    j___libc_free_0(v301);
    if ( v297 )
      j_j___libc_free_0(v297, v299 - v297);
    ++v278;
  }
  while ( v279 != v278 );
LABEL_153:
  sub_1B06610(v267, *v294, *(v295 - 1), (__int64)&v339);
  v112 = sub_157EBA0(*(v286 - 1));
  sub_1593B40((_QWORD *)(v112 - 24), *v288[0]);
  if ( a3 == a2 )
  {
    while ( 1 )
    {
      v195 = *(_QWORD *)(*(_QWORD *)v284[0] + 48LL);
      if ( !v195 )
        BUG();
      if ( *(_BYTE *)(v195 - 8) != 77 )
        break;
      v196 = v195 - 24;
      v197 = 0x17FFFFFFE8LL;
      v198 = *(_BYTE *)(v195 - 1) & 0x40;
      v199 = *(_DWORD *)(v195 - 4) & 0xFFFFFFF;
      if ( v199 )
      {
        v200 = 24LL * *(unsigned int *)(v195 + 32) + 8;
        v201 = 0;
        do
        {
          v202 = v196 - 24LL * v199;
          if ( v198 )
            v202 = *(_QWORD *)(v195 - 32);
          if ( v265 == *(_QWORD *)(v202 + v200) )
          {
            v197 = 24 * v201;
            goto LABEL_266;
          }
          ++v201;
          v200 += 8;
        }
        while ( v199 != (_DWORD)v201 );
        v197 = 0x17FFFFFFE8LL;
      }
LABEL_266:
      if ( v198 )
        v190 = *(_QWORD *)(v195 - 32);
      else
        v190 = v196 - 24LL * v199;
      sub_164D160(v195 - 24, *(_QWORD *)(v190 + v197), a7, a8, a9, a10, v113, v114, a13, a14);
      sub_157EA20(*(_QWORD *)(v195 + 16) + 40LL, v195 - 24);
      v191 = *(unsigned __int64 **)(v195 + 8);
      v192 = *(_QWORD *)v195 & 0xFFFFFFFFFFFFFFF8LL;
      *v191 = v192 | *v191 & 7;
      *(_QWORD *)(v192 + 8) = v191;
      *(_QWORD *)v195 &= 7uLL;
      *(_QWORD *)(v195 + 8) = 0;
      sub_164BEC0(v195 - 24, v195 - 24, v192, (__int64)v191, a7, a8, a9, a10, v193, v194, a13, a14);
    }
  }
  else
  {
    sub_1B06610(*(_QWORD *)v284[0], *v294, *(v295 - 1), (__int64)&v339);
  }
  v115 = 1;
  if ( v279 != 1 )
  {
    do
    {
      v116 = sub_157EBA0(v285[v115 - 1]);
      v117 = *(_QWORD *)(v284[0] + 8LL * v115);
      if ( *(_QWORD *)(v116 - 24) )
      {
        v118 = *(_QWORD *)(v116 - 16);
        v119 = *(_QWORD *)(v116 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v119 = v118;
        if ( v118 )
          *(_QWORD *)(v118 + 16) = v119 | *(_QWORD *)(v118 + 16) & 3LL;
      }
      *(_QWORD *)(v116 - 24) = v117;
      if ( v117 )
      {
        v120 = *(_QWORD *)(v117 + 8);
        *(_QWORD *)(v116 - 16) = v120;
        if ( v120 )
          *(_QWORD *)(v120 + 16) = (v116 - 16) | *(_QWORD *)(v120 + 16) & 3LL;
        v121 = *(_QWORD *)(v116 - 8);
        v122 = v116 - 24;
        *(_QWORD *)(v122 + 16) = (v117 + 8) | v121 & 3;
        *(_QWORD *)(v117 + 8) = v122;
      }
      ++v115;
    }
    while ( v279 != v115 );
  }
  v123 = sub_157EBA0(*(v291 - 1));
  sub_1593B40((_QWORD *)(v123 + -24 - 24LL * !v263), *v288[0]);
  sub_1593B40((_QWORD *)(v123 + -24 - 24LL * v263), *v293[0]);
  sub_1B04C00(*v288[0], *v285, *(v286 - 1));
  sub_1B04C00(*v288[0], *v290, *(v291 - 1));
  if ( v279 != 1 )
  {
    v124 = 1;
    while ( 1 )
    {
      v125 = (_QWORD *)sub_157EBA0(v290[v124 - 1]);
      v126 = v288[0][v124];
      v127 = sub_1648A60(56, 1u);
      if ( v127 )
        sub_15F8320((__int64)v127, v126, (__int64)v125);
      sub_15F20C0(v125);
      sub_1B04C00(v288[0][v124], v285[v124], *(v286 - 1));
      sub_1B04C00(v288[0][v124], v290[v124], *(v291 - 1));
      v128 = v288[0][v124];
      v129 = sub_157ED20(*v288[0]);
      v130 = *(_QWORD *)(v128 + 48);
      if ( !v130 )
        goto LABEL_518;
      if ( *(_BYTE *)(v130 - 8) == 77 )
        break;
LABEL_172:
      if ( v279 == ++v124 )
        goto LABEL_173;
    }
    while ( 1 )
    {
      sub_15F22F0((_QWORD *)(v130 - 24), v129);
      v130 = *(_QWORD *)(v128 + 48);
      if ( !v130 )
        break;
      if ( *(_BYTE *)(v130 - 8) != 77 )
        goto LABEL_172;
    }
LABEL_518:
    BUG();
  }
LABEL_173:
  v131 = (_QWORD *)sub_157EBA0(*(v295 - 1));
  if ( a3 == a2 )
  {
    v254 = sub_1648A60(56, 1u);
    if ( v254 )
      sub_15F8320((__int64)v254, v267, (__int64)v131);
    sub_15F20C0(v131);
  }
  else
  {
    sub_1593B40(&v131[-3 * !v264 - 3], *(_QWORD *)v284[0]);
  }
  v132 = 1;
  sub_1B04C00(*v293[0], *v290, *(v291 - 1));
  if ( v279 != 1 )
  {
    do
    {
      v135 = (_QWORD *)sub_157EBA0(v294[v132 - 1]);
      v136 = v293[0][v132];
      v137 = sub_1648A60(56, 1u);
      if ( v137 )
        sub_15F8320((__int64)v137, v136, (__int64)v135);
      sub_15F20C0(v135);
      sub_1B04C00(v293[0][v132], v290[v132], *(v291 - 1));
      v138 = v293[0][v132];
      v139 = sub_157ED20(*v293[0]);
      v142 = *(_QWORD *)(v138 + 48);
      if ( !v142 )
        goto LABEL_517;
      if ( *(_BYTE *)(v142 - 8) == 77 )
      {
        while ( 1 )
        {
          sub_15F22F0((_QWORD *)(v142 - 24), v139);
          v142 = *(_QWORD *)(v138 + 48);
          if ( !v142 )
            break;
          if ( *(_BYTE *)(v142 - 8) != 77 )
            goto LABEL_182;
        }
LABEL_517:
        BUG();
      }
LABEL_182:
      ++v132;
    }
    while ( v279 != v132 );
    if ( v132 != 1 )
    {
      v351 = 0x400000000LL;
      v350 = (char *)&v352;
      LOBYTE(v309) = 1;
      sub_1B054F0((__int64)&v350, (unsigned __int8 *)&v309, v285, v288[0], v140, v141);
      LOBYTE(v309) = 1;
      sub_1B054F0((__int64)&v350, (unsigned __int8 *)&v309, v290, v293[0], v143, v144);
      LOBYTE(v309) = 0;
      sub_1B054F0((__int64)&v350, (unsigned __int8 *)&v309, v286 - 1, v288[0], v145, v146);
      LOBYTE(v309) = 0;
      sub_1B054F0((__int64)&v350, (unsigned __int8 *)&v309, v291 - 1, v293[0], v147, v148);
      sub_15DC140(a16, (__int64 *)v350, (unsigned int)v351);
      if ( v350 != (char *)&v352 )
        _libc_free((unsigned __int64)v350);
    }
  }
  v149 = v286;
  v150 = v285;
  v151 = v355;
  v350 = 0;
  v351 = (__int64)v355;
  v152 = v355;
  v352 = v355;
  v353 = 16;
  for ( LODWORD(v354) = 0; v149 != v150; ++v150 )
  {
LABEL_190:
    v153 = *v150;
    if ( v151 != v152 )
      goto LABEL_188;
    v154 = &v151[HIDWORD(v353)];
    if ( v154 != v151 )
    {
      v155 = v151;
      v156 = 0;
      while ( v153 != *v155 )
      {
        if ( *v155 == -2 )
          v156 = v155;
        if ( v154 == ++v155 )
        {
          if ( !v156 )
            goto LABEL_472;
          ++v150;
          *v156 = v153;
          v152 = v352;
          LODWORD(v354) = v354 - 1;
          v151 = (__int64 *)v351;
          ++v350;
          if ( v149 != v150 )
            goto LABEL_190;
          goto LABEL_199;
        }
      }
      continue;
    }
LABEL_472:
    if ( HIDWORD(v353) < (unsigned int)v353 )
    {
      ++HIDWORD(v353);
      *v154 = v153;
      v151 = (__int64 *)v351;
      ++v350;
      v152 = v352;
    }
    else
    {
LABEL_188:
      sub_16CCBA0((__int64)&v350, *v150);
      v152 = v352;
      v151 = (__int64 *)v351;
    }
  }
LABEL_199:
  v157 = v291;
  v158 = v290;
  if ( v290 != v291 )
  {
    v159 = (unsigned __int64)v352;
    v160 = v351;
    do
    {
LABEL_203:
      v161 = *v158;
      if ( v159 != v160 )
        goto LABEL_201;
      v162 = (__int64 *)(v159 + 8LL * HIDWORD(v353));
      if ( v162 != (__int64 *)v159 )
      {
        v163 = (__int64 *)v159;
        v164 = 0;
        while ( v161 != *v163 )
        {
          if ( *v163 == -2 )
            v164 = v163;
          if ( v162 == ++v163 )
          {
            if ( !v164 )
              goto LABEL_481;
            ++v158;
            *v164 = v161;
            v159 = (unsigned __int64)v352;
            LODWORD(v354) = v354 - 1;
            v160 = v351;
            ++v350;
            if ( v157 != v158 )
              goto LABEL_203;
            goto LABEL_212;
          }
        }
        goto LABEL_202;
      }
LABEL_481:
      if ( HIDWORD(v353) < (unsigned int)v353 )
      {
        ++HIDWORD(v353);
        *v162 = v161;
        v160 = v351;
        ++v350;
        v159 = (unsigned __int64)v352;
      }
      else
      {
LABEL_201:
        sub_16CCBA0((__int64)&v350, v161);
        v159 = (unsigned __int64)v352;
        v160 = v351;
      }
LABEL_202:
      ++v158;
    }
    while ( v157 != v158 );
  }
LABEL_212:
  v165 = v295;
  v166 = v294;
  if ( v294 != v295 )
  {
    v167 = (unsigned __int64)v352;
    v168 = v351;
    do
    {
LABEL_216:
      v169 = *v166;
      if ( v167 != v168 )
        goto LABEL_214;
      v170 = (__int64 *)(v167 + 8LL * HIDWORD(v353));
      if ( v170 != (__int64 *)v167 )
      {
        v171 = (__int64 *)v167;
        v172 = 0;
        while ( v169 != *v171 )
        {
          if ( *v171 == -2 )
            v172 = v171;
          if ( v170 == ++v171 )
          {
            if ( !v172 )
              goto LABEL_479;
            ++v166;
            *v172 = v169;
            v167 = (unsigned __int64)v352;
            LODWORD(v354) = v354 - 1;
            v168 = v351;
            ++v350;
            if ( v165 != v166 )
              goto LABEL_216;
            goto LABEL_225;
          }
        }
        goto LABEL_215;
      }
LABEL_479:
      if ( HIDWORD(v353) < (unsigned int)v353 )
      {
        ++HIDWORD(v353);
        *v170 = v169;
        v168 = v351;
        ++v350;
        v167 = (unsigned __int64)v352;
      }
      else
      {
LABEL_214:
        sub_16CCBA0((__int64)&v350, v169);
        v167 = (unsigned __int64)v352;
        v168 = v351;
      }
LABEL_215:
      ++v166;
    }
    while ( v165 != v166 );
  }
LABEL_225:
  v173 = HIDWORD(v353);
  if ( HIDWORD(v353) != (_DWORD)v354 )
  {
    while ( 2 )
    {
      v174 = (unsigned __int64)v352;
      v175 = (__int64 *)v351;
      v176 = &v352[v173];
      if ( v352 != (__int64 *)v351 )
        v176 = &v352[(unsigned int)v353];
      v177 = *v352;
      if ( v352 != v176 )
      {
        v178 = v352;
        while ( 1 )
        {
          v177 = *v178;
          v179 = v178;
          if ( (unsigned __int64)*v178 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v176 == ++v178 )
          {
            v177 = v179[1];
            break;
          }
        }
      }
      v180 = sub_157EBA0(v177);
      v181 = v180;
      if ( *(_BYTE *)(v180 + 16) == 26 && (*(_DWORD *)(v180 + 20) & 0xFFFFFFF) == 1 )
      {
        if ( !sub_1377F70((__int64)&v280[3].m128i_i64[1], *(_QWORD *)(v180 - 24)) )
        {
          v174 = (unsigned __int64)v352;
          v175 = (__int64 *)v351;
          goto LABEL_235;
        }
        v255 = *(_QWORD *)(v181 - 24);
        if ( sub_1AFF430(v255, a6, a7, a8, a9, a10, v133, v134, a13, a14, a15, a16, 0) )
          sub_1B04D00((__int64)&v350, v255);
        else
          sub_1B04D00((__int64)&v350, v177);
        v182 = v354;
      }
      else
      {
LABEL_235:
        if ( (__int64 *)v174 == v175 )
        {
          v250 = &v175[HIDWORD(v353)];
          if ( v175 == v250 )
          {
LABEL_455:
            v175 = v250;
          }
          else
          {
            while ( v177 != *v175 )
            {
              if ( v250 == ++v175 )
                goto LABEL_455;
            }
          }
        }
        else
        {
          v175 = sub_16CC9F0((__int64)&v350, v177);
          if ( v177 == *v175 )
          {
            if ( v352 == (__int64 *)v351 )
              v250 = &v352[HIDWORD(v353)];
            else
              v250 = &v352[(unsigned int)v353];
          }
          else
          {
            if ( v352 != (__int64 *)v351 )
              goto LABEL_238;
            v175 = &v352[HIDWORD(v353)];
            v250 = v175;
          }
        }
        if ( v175 == v250 )
        {
LABEL_238:
          v182 = v354;
        }
        else
        {
          *v175 = -2;
          v182 = v354 + 1;
          LODWORD(v354) = v354 + 1;
        }
      }
      v173 = HIDWORD(v353);
      if ( HIDWORD(v353) == v182 )
        break;
      continue;
    }
  }
  sub_1AFD4E0(v270, 1, a6, a15, a16, a17, a7, a8, a9, a10, v133, v134, a13, a14);
  if ( a3 == a2 )
  {
    v30 = 2;
    sub_1AFD4E0((__int64)v280, 0, a6, a15, a16, a17, a7, a8, a9, a10, v183, v184, a13, a14);
    sub_1401B00(a6, v280);
  }
  else
  {
    v30 = 1;
    if ( v279 <= 1 )
      sub_1AFD4E0((__int64)v280, 0, a6, a15, a16, a17, a7, a8, a9, a10, v183, v184, a13, a14);
    else
      sub_1AFD4E0((__int64)v280, 1, a6, a15, a16, a17, a7, a8, a9, a10, v183, v184, a13, a14);
  }
  if ( v352 != (__int64 *)v351 )
    _libc_free((unsigned __int64)v352);
  if ( v319 )
    j_j___libc_free_0(v319, v321 - v319);
  j___libc_free_0(v316);
  if ( v346 )
  {
    if ( v345 )
    {
      v251 = v344;
      v252 = &v344[2 * v345];
      do
      {
        if ( *v251 != -8 && *v251 != -4 )
        {
          v253 = v251[1];
          if ( v253 )
            sub_161E7C0((__int64)(v251 + 1), v253);
        }
        v251 += 2;
      }
      while ( v252 != v251 );
    }
    j___libc_free_0(v344);
  }
  if ( v342 )
  {
    v185 = (_QWORD *)v340;
    v315 = 2;
    v316 = 0;
    v186 = -8;
    v187 = (_QWORD *)(v340 + ((unsigned __int64)v342 << 6));
    v317 = -8;
    v314 = &unk_49E6B50;
    v350 = (char *)&unk_49E6B50;
    v318 = 0;
    v351 = 2;
    v352 = 0;
    v353 = -16;
    v354 = 0;
    while ( 1 )
    {
      v188 = v185[3];
      if ( v186 != v188 && v188 != v353 )
        sub_1455FA0((__int64)(v185 + 5));
      *v185 = &unk_49EE2B0;
      v189 = (__int64)(v185 + 1);
      v185 += 8;
      sub_1455FA0(v189);
      if ( v187 == v185 )
        break;
      v186 = v317;
    }
    v350 = (char *)&unk_49EE2B0;
    sub_1455FA0((__int64)&v351);
    v314 = &unk_49EE2B0;
    sub_1455FA0((__int64)&v315);
  }
  j___libc_free_0(v340);
  sub_15CE080(&v294);
  sub_15CE080(v293);
  sub_15CE080(&v290);
  if ( v288[0] )
    j_j___libc_free_0(v288[0], v289 - (unsigned __int64)v288[0]);
  if ( v285 )
    j_j___libc_free_0(v285, v287 - (_QWORD)v285);
  sub_15CE080(v284);
  if ( v335 != (_BYTE *)v334.m128i_i64[1] )
    _libc_free((unsigned __int64)v335);
  if ( v330 != v329 )
    _libc_free((unsigned __int64)v330);
  if ( v324 != v323 )
    _libc_free((unsigned __int64)v324);
  return v30;
}
