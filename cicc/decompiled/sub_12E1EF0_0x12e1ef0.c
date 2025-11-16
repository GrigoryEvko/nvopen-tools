// Function: sub_12E1EF0
// Address: 0x12e1ef0
//
__int64 __fastcall sub_12E1EF0(
        __int64 *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9,
        __int64 a10,
        __int64 a11)
{
  _QWORD *v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r11
  __int64 v14; // r10
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __m128i *v20; // r12
  __m128i *v21; // rdi
  __m128i *v22; // r13
  __int64 (__fastcall *v23)(__int64); // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r14
  char v27; // al
  const void *v28; // r13
  size_t v29; // rdx
  size_t v30; // r15
  unsigned __int8 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rcx
  _QWORD *v34; // r9
  __m128i *v35; // r12
  __m128i *v36; // rdi
  __m128i *v37; // r13
  __int64 (__fastcall *v38)(__int64); // rax
  __int64 v39; // rdi
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // r13
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  __int8 v52; // cl
  _BYTE *v53; // rsi
  __int64 v54; // r15
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // r9
  __int64 v59; // r10
  const __m128i *v60; // rax
  const __m128i *v61; // rsi
  unsigned __int64 v62; // rbx
  __int64 v63; // rax
  __m128i *v64; // rdx
  __m128i *v65; // rcx
  __m128i *v66; // rax
  __m128i v67; // xmm2
  __m128i v68; // xmm3
  __m128i v69; // xmm4
  __m128i v70; // xmm5
  __m128i v71; // xmm6
  __int64 v72; // rax
  __m128i v73; // xmm7
  __m128i v74; // xmm2
  __m128i v75; // xmm3
  __m128i v76; // xmm4
  __m128i v77; // xmm5
  __m128i *v78; // rdx
  __int64 v79; // rdx
  __m128i *v80; // rdx
  __m128i *v81; // rdx
  __int8 *v82; // rdx
  volatile signed __int32 *v83; // r14
  signed __int32 v84; // eax
  __int64 v85; // r15
  _QWORD *v86; // r14
  _QWORD *v87; // rbx
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rax
  _QWORD *v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  _DWORD *v94; // rdx
  unsigned int v95; // r14d
  char *v96; // rbx
  char *v97; // r12
  __int64 v98; // r13
  __int64 v99; // rbx
  __int64 v100; // r12
  __int64 v101; // rdi
  __int64 v102; // r8
  __int64 v103; // r12
  __int64 v104; // rbx
  __int64 v105; // rdi
  __int64 v106; // r8
  __int64 v107; // r12
  __int64 v108; // rbx
  __int64 v109; // rdi
  __int64 v110; // r8
  __int64 v111; // r12
  __int64 v112; // rbx
  __int64 v113; // rdi
  __int64 v115; // rax
  __int64 v116; // rdx
  unsigned int v117; // r8d
  __int64 *v118; // r9
  __int64 v119; // rcx
  __int64 v120; // rdi
  const void *v121; // rsi
  size_t v122; // rdx
  size_t v123; // r15
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 *v126; // r10
  __int64 v127; // rax
  void *v128; // rax
  __int64 v129; // rax
  __int64 v130; // rcx
  unsigned int v131; // r9d
  __int64 *v132; // r10
  __int64 v133; // r8
  _BYTE *v134; // rdi
  __int64 v135; // rdi
  char v136; // al
  const void *v137; // r13
  size_t v138; // rdx
  size_t v139; // r12
  __int64 v140; // rax
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 v144; // r9
  unsigned int v145; // r15d
  _QWORD *v146; // r10
  __int64 v147; // rax
  __int64 v148; // rcx
  __int64 v149; // r9
  _QWORD *v150; // r10
  _QWORD *v151; // r8
  void *v152; // rdi
  const void *v153; // rsi
  size_t v154; // rdx
  size_t v155; // r15
  __int64 v156; // rcx
  __int64 v157; // r8
  __int64 v158; // r9
  unsigned int v159; // r13d
  _QWORD *v160; // r10
  __int64 v161; // rax
  __int64 v162; // rcx
  __int64 v163; // r9
  __int64 *v164; // r10
  __int64 v165; // r8
  _BYTE *v166; // rdi
  __int64 v167; // rax
  _BYTE *v168; // rax
  __int64 v169; // rax
  _BYTE *v170; // rax
  __int64 v171; // rax
  void *v172; // rax
  _QWORD *v173; // rbx
  _QWORD *v174; // r15
  signed __int32 v175; // eax
  _BYTE *v176; // rbx
  __int64 v177; // rsi
  int v178; // edx
  int v179; // ecx
  int v180; // r8d
  int v181; // r9d
  __int64 v182; // rdx
  char *v183; // rax
  int v184; // eax
  __int64 v185; // rax
  __m128i si128; // xmm0
  __int64 v187; // r14
  _BYTE *v188; // rdx
  char *v189; // r12
  unsigned int v190; // ebx
  __int64 v191; // rax
  __int64 v192; // rdx
  bool v193; // cf
  unsigned __int64 v194; // rax
  char *v195; // rax
  char *v196; // r13
  signed __int64 v197; // r9
  __int64 *v198; // r9
  char *v199; // r10
  __int64 v200; // rdi
  int v201; // esi
  __int64 v202; // rax
  __int64 v203; // r11
  __int64 v204; // rcx
  char v205; // cl
  size_t v206; // r15
  __int64 v207; // r13
  __int64 v208; // r14
  __int64 v209; // rdi
  void *v210; // r12
  size_t v211; // rdx
  size_t v212; // rbx
  int v213; // eax
  int v214; // eax
  char *v215; // rax
  int v216; // eax
  __int64 v217; // rax
  __int64 v218; // r8
  __int64 v219; // r9
  __int64 *v220; // r10
  unsigned int v221; // ebx
  __int64 v222; // r8
  __int64 v223; // r9
  char *v224; // rcx
  __int64 v225; // rax
  __int64 v226; // r8
  __int64 v227; // r9
  __int64 *v228; // rcx
  __int64 v229; // r12
  void *v230; // rdi
  __int64 v231; // r14
  __int64 v232; // r15
  int v233; // eax
  __int64 v234; // rax
  __int64 v235; // r13
  __int64 v236; // r12
  __int64 v237; // r15
  __int64 v238; // rax
  __int64 v239; // rdx
  int v240; // eax
  __int64 v241; // rax
  int v242; // eax
  char v243; // cl
  char v244; // dl
  __int64 v245; // rax
  __int64 v246; // rcx
  unsigned int v247; // r9d
  __int64 *v248; // r10
  __int64 v249; // r8
  _BYTE *v250; // rdi
  __int64 v251; // rax
  _BYTE *v252; // rax
  __int64 v253; // rax
  void *v254; // rax
  __int64 v255; // r13
  __int64 v256; // r12
  __int64 v257; // r15
  __int64 v258; // rax
  __int64 v259; // rdx
  int v260; // eax
  __int64 v261; // rax
  int v262; // eax
  char v263; // cl
  char v264; // dl
  char *v265; // r8
  __int64 v266; // r12
  __int64 v267; // rbx
  __int64 v268; // rdi
  __int64 v269; // rbx
  __int64 v270; // rax
  unsigned __int64 v271; // r12
  unsigned __int64 v272; // rdx
  __int64 v273; // rcx
  __int64 v274; // rsi
  __int64 v275; // rax
  __int64 *v276; // rdi
  __int64 v277; // r11
  __int32 v278; // edx
  _QWORD *v281; // [rsp+38h] [rbp-27F8h]
  _QWORD *v282; // [rsp+38h] [rbp-27F8h]
  __int64 v283; // [rsp+40h] [rbp-27F0h]
  _BYTE *v284; // [rsp+48h] [rbp-27E8h]
  __int64 v286; // [rsp+50h] [rbp-27E0h]
  __int64 *v287; // [rsp+50h] [rbp-27E0h]
  __int64 *v288; // [rsp+50h] [rbp-27E0h]
  __int64 v291; // [rsp+68h] [rbp-27C8h]
  unsigned int v292; // [rsp+68h] [rbp-27C8h]
  __int64 *v293; // [rsp+68h] [rbp-27C8h]
  unsigned int v294; // [rsp+68h] [rbp-27C8h]
  unsigned int v295; // [rsp+78h] [rbp-27B8h]
  __int64 v296; // [rsp+78h] [rbp-27B8h]
  int v297; // [rsp+80h] [rbp-27B0h]
  __int64 *v298; // [rsp+80h] [rbp-27B0h]
  __int64 v299; // [rsp+80h] [rbp-27B0h]
  _BYTE *v301; // [rsp+88h] [rbp-27A8h]
  __int64 *v302; // [rsp+90h] [rbp-27A0h]
  __int64 *v303; // [rsp+90h] [rbp-27A0h]
  __int64 *v304; // [rsp+90h] [rbp-27A0h]
  _QWORD *v305; // [rsp+90h] [rbp-27A0h]
  int v306; // [rsp+90h] [rbp-27A0h]
  _QWORD *v307; // [rsp+90h] [rbp-27A0h]
  __int64 v308; // [rsp+90h] [rbp-27A0h]
  __int64 v309; // [rsp+90h] [rbp-27A0h]
  __int64 v310; // [rsp+90h] [rbp-27A0h]
  unsigned int v311; // [rsp+98h] [rbp-2798h]
  __int64 v312; // [rsp+98h] [rbp-2798h]
  unsigned int v313; // [rsp+98h] [rbp-2798h]
  __int64 v314; // [rsp+98h] [rbp-2798h]
  __int64 v315; // [rsp+98h] [rbp-2798h]
  _QWORD *v316; // [rsp+98h] [rbp-2798h]
  __int64 *v317; // [rsp+98h] [rbp-2798h]
  __int64 *v318; // [rsp+98h] [rbp-2798h]
  __int64 *v319; // [rsp+98h] [rbp-2798h]
  __int64 *v320; // [rsp+A0h] [rbp-2790h]
  __int64 v321; // [rsp+A0h] [rbp-2790h]
  __int64 *v322; // [rsp+A0h] [rbp-2790h]
  __int64 *v323; // [rsp+A0h] [rbp-2790h]
  signed __int64 v324; // [rsp+A0h] [rbp-2790h]
  int v325; // [rsp+A0h] [rbp-2790h]
  unsigned int v326; // [rsp+A0h] [rbp-2790h]
  unsigned int v327; // [rsp+A0h] [rbp-2790h]
  __int64 *v328; // [rsp+A0h] [rbp-2790h]
  unsigned int v329; // [rsp+A8h] [rbp-2788h]
  __int64 v330; // [rsp+A8h] [rbp-2788h]
  int v331; // [rsp+A8h] [rbp-2788h]
  unsigned int v332; // [rsp+A8h] [rbp-2788h]
  __int64 v333; // [rsp+A8h] [rbp-2788h]
  __int64 v334; // [rsp+A8h] [rbp-2788h]
  void *v335; // [rsp+B0h] [rbp-2780h]
  void *v336; // [rsp+B0h] [rbp-2780h]
  _BYTE *v337; // [rsp+B0h] [rbp-2780h]
  void *v338; // [rsp+B0h] [rbp-2780h]
  __int64 v339; // [rsp+B8h] [rbp-2778h]
  unsigned __int64 v340; // [rsp+B8h] [rbp-2778h]
  __int64 v341; // [rsp+B8h] [rbp-2778h]
  int v342; // [rsp+B8h] [rbp-2778h]
  __int64 *v343; // [rsp+B8h] [rbp-2778h]
  char v344; // [rsp+CBh] [rbp-2765h] BYREF
  int v345; // [rsp+CCh] [rbp-2764h] BYREF
  __int64 v346; // [rsp+D0h] [rbp-2760h] BYREF
  __int64 v347; // [rsp+D8h] [rbp-2758h] BYREF
  char v348[8]; // [rsp+E0h] [rbp-2750h] BYREF
  volatile signed __int32 *v349; // [rsp+E8h] [rbp-2748h]
  _BYTE *v350; // [rsp+F0h] [rbp-2740h] BYREF
  _BYTE *v351; // [rsp+F8h] [rbp-2738h]
  _BYTE *v352; // [rsp+100h] [rbp-2730h]
  __int64 v353; // [rsp+110h] [rbp-2720h] BYREF
  __int64 v354; // [rsp+118h] [rbp-2718h]
  __int64 v355; // [rsp+120h] [rbp-2710h]
  char *v356; // [rsp+130h] [rbp-2700h] BYREF
  char *v357; // [rsp+138h] [rbp-26F8h]
  char *i; // [rsp+140h] [rbp-26F0h]
  __int64 v359; // [rsp+150h] [rbp-26E0h] BYREF
  __int64 v360; // [rsp+158h] [rbp-26D8h]
  __int64 v361; // [rsp+160h] [rbp-26D0h]
  __int64 v362; // [rsp+170h] [rbp-26C0h] BYREF
  __int64 v363; // [rsp+178h] [rbp-26B8h]
  __int64 v364; // [rsp+180h] [rbp-26B0h]
  __int64 v365; // [rsp+190h] [rbp-26A0h] BYREF
  __int64 v366; // [rsp+198h] [rbp-2698h]
  __int64 v367; // [rsp+1A0h] [rbp-2690h]
  __m128i *v368; // [rsp+1B0h] [rbp-2680h] BYREF
  __int64 v369; // [rsp+1B8h] [rbp-2678h]
  __m128i v370; // [rsp+1C0h] [rbp-2670h] BYREF
  _QWORD v371[2]; // [rsp+1D0h] [rbp-2660h] BYREF
  __int64 (__fastcall *v372)(__int64 *, __int64 *, int); // [rsp+1E0h] [rbp-2650h]
  __int64 (__fastcall *v373)(); // [rsp+1E8h] [rbp-2648h]
  void *src; // [rsp+1F0h] [rbp-2640h] BYREF
  __int64 v375; // [rsp+1F8h] [rbp-2638h]
  __int64 v376; // [rsp+200h] [rbp-2630h]
  __int64 v377; // [rsp+208h] [rbp-2628h]
  char *v378; // [rsp+210h] [rbp-2620h]
  __int128 v379; // [rsp+220h] [rbp-2610h] BYREF
  __int128 v380; // [rsp+230h] [rbp-2600h] BYREF
  __int64 v381; // [rsp+240h] [rbp-25F0h]
  char *v382; // [rsp+250h] [rbp-25E0h] BYREF
  _QWORD *v383; // [rsp+258h] [rbp-25D8h]
  __int64 v384; // [rsp+260h] [rbp-25D0h]
  unsigned int v385; // [rsp+268h] [rbp-25C8h]
  _QWORD *v386; // [rsp+278h] [rbp-25B8h]
  unsigned int v387; // [rsp+288h] [rbp-25A8h]
  char v388; // [rsp+290h] [rbp-25A0h]
  char v389; // [rsp+299h] [rbp-2597h]
  _BYTE v390[288]; // [rsp+2A0h] [rbp-2590h] BYREF
  __m128i v391; // [rsp+3C0h] [rbp-2470h] BYREF
  __m128i v392; // [rsp+3D0h] [rbp-2460h] BYREF
  __m128i v393; // [rsp+3E0h] [rbp-2450h] BYREF
  __m128i v394; // [rsp+3F0h] [rbp-2440h] BYREF
  __m128i v395; // [rsp+400h] [rbp-2430h] BYREF
  __m128i *v396; // [rsp+410h] [rbp-2420h]
  __int64 v397; // [rsp+418h] [rbp-2418h]
  __m128i v398; // [rsp+420h] [rbp-2410h] BYREF
  _BYTE v399[4480]; // [rsp+430h] [rbp-2400h] BYREF
  __int64 v400; // [rsp+15B0h] [rbp-1280h] BYREF
  __m128i *v401; // [rsp+15B8h] [rbp-1278h]
  __m128i *v402; // [rsp+15C0h] [rbp-1270h]
  __int8 *v403; // [rsp+15C8h] [rbp-1268h]
  int v404; // [rsp+15D0h] [rbp-1260h]
  __m128i v405; // [rsp+15E0h] [rbp-1250h] BYREF
  __m128i v406; // [rsp+15F0h] [rbp-1240h] BYREF
  __m128i v407; // [rsp+1600h] [rbp-1230h] BYREF
  __m128i v408; // [rsp+1610h] [rbp-1220h] BYREF
  __m128i v409; // [rsp+1620h] [rbp-1210h] BYREF
  __m128i *v410; // [rsp+1630h] [rbp-1200h]
  __int64 v411; // [rsp+1638h] [rbp-11F8h]
  __m128i v412; // [rsp+1640h] [rbp-11F0h] BYREF
  _BYTE v413[4480]; // [rsp+1650h] [rbp-11E0h] BYREF
  __int64 v414; // [rsp+27D0h] [rbp-60h]
  __m128i *v415; // [rsp+27D8h] [rbp-58h]
  __m128i *v416; // [rsp+27E0h] [rbp-50h]
  __int8 *v417; // [rsp+27E8h] [rbp-48h]
  int v418; // [rsp+27F0h] [rbp-40h]

  v11 = (_QWORD *)*a1;
  v361 = 0x1000000000LL;
  v364 = 0x1000000000LL;
  v12 = (__int64)(v11 + 7);
  v13 = (__int64)(v11 + 1);
  v367 = 0x1800000000LL;
  v14 = (__int64)(v11 + 3);
  v15 = v11[4];
  v16 = (__int64)(v11 + 5);
  v17 = v11[6];
  v18 = v11[2];
  v19 = v11[8];
  v359 = 0;
  v360 = 0;
  v362 = 0;
  v363 = 0;
  v365 = 0;
  v366 = 0;
  v344 = 1;
  v283 = v16;
  v291 = v13;
  v335 = (void *)v14;
  v391.m128i_i64[0] = v19;
  v391.m128i_i64[1] = v12;
  v392.m128i_i64[0] = v17;
  v392.m128i_i64[1] = v16;
  v393.m128i_i64[0] = v18;
  v393.m128i_i64[1] = v13;
  v394.m128i_i64[0] = v15;
  v394.m128i_i64[1] = v14;
  v297 = 0;
  if ( v14 == v15 )
    goto LABEL_17;
  do
  {
    do
    {
      v20 = &v405;
      v406.m128i_i64[1] = 0;
      v21 = &v391;
      v406.m128i_i64[0] = (__int64)sub_12D3C60;
      v22 = &v405;
      v407.m128i_i64[1] = 0;
      v407.m128i_i64[0] = (__int64)sub_12D3C80;
      v408.m128i_i64[1] = 0;
      v408.m128i_i64[0] = (__int64)sub_12D3CA0;
      v23 = sub_12D3C40;
      if ( ((unsigned __int8)sub_12D3C40 & 1) == 0 )
        goto LABEL_4;
      while ( 1 )
      {
        v23 = *(__int64 (__fastcall **)(__int64))((char *)v23 + v21->m128i_i64[0] - 1);
LABEL_4:
        v24 = v23((__int64)v21);
        if ( v24 )
          break;
        while ( 1 )
        {
          v25 = v22[1].m128i_i64[1];
          v23 = (__int64 (__fastcall *)(__int64))v22[1].m128i_i64[0];
          v22 = ++v20;
          v21 = (__m128i *)((char *)&v391 + v25);
          if ( ((unsigned __int8)v23 & 1) != 0 )
            break;
          v24 = v23((__int64)v21);
          if ( v24 )
            goto LABEL_7;
        }
      }
LABEL_7:
      v26 = v24;
      v27 = *(_BYTE *)(v24 + 16);
      if ( !v27 )
      {
        if ( (unsigned __int8)sub_15E4F60(v26) )
          goto LABEL_157;
        v331 = sub_12D3D20(v26);
        v153 = (const void *)sub_1649960(v26);
        v155 = v154;
        v159 = sub_16D19C0(&v359, v153, v154);
        v160 = (_QWORD *)(v359 + 8LL * v159);
        if ( *v160 )
        {
          if ( *v160 != -8 )
            goto LABEL_157;
          LODWORD(v361) = v361 - 1;
        }
        v304 = (__int64 *)(v359 + 8LL * v159);
        v161 = malloc(v155 + 17, v153, v155 + 17, v156, v157, v158);
        v164 = v304;
        v165 = v161;
        if ( !v161 )
        {
          if ( v155 == -17 )
          {
            v169 = malloc(1, v153, 0, v162, 0, v163);
            v164 = v304;
            v165 = 0;
            if ( v169 )
            {
              v166 = (_BYTE *)(v169 + 16);
              v165 = v169;
              goto LABEL_187;
            }
          }
          v308 = v165;
          v317 = v164;
          sub_16BD1C0("Allocation failed");
          v164 = v317;
          v165 = v308;
        }
        v166 = (_BYTE *)(v165 + 16);
        if ( v155 + 1 <= 1 )
        {
LABEL_180:
          v166[v155] = 0;
          *(_QWORD *)v165 = v155;
          *(_DWORD *)(v165 + 8) = v331;
          *v164 = v165;
          ++HIDWORD(v360);
          sub_16D1CD0(&v359, v159);
LABEL_157:
          v121 = (const void *)sub_1649960(v26);
          v123 = v122;
          v125 = (unsigned int)sub_16D19C0(a5, v121, v122);
          v126 = (__int64 *)(*(_QWORD *)a5 + 8 * v125);
          if ( *v126 )
          {
            if ( *v126 != -8 )
            {
LABEL_159:
              ++v297;
              goto LABEL_9;
            }
            --*(_DWORD *)(a5 + 16);
          }
          v303 = v126;
          v313 = v125;
          v129 = malloc(v123 + 17, v121, v123 + 17, a5, v124, v125);
          v131 = v313;
          v132 = v303;
          v133 = v129;
          if ( !v129 )
          {
            if ( v123 == -17 )
            {
              v167 = malloc(1, v121, 0, v130, 0, v313);
              v131 = v313;
              v132 = v303;
              v133 = 0;
              if ( v167 )
              {
                v134 = (_BYTE *)(v167 + 16);
                v133 = v167;
                goto LABEL_183;
              }
            }
            v309 = v133;
            v318 = v132;
            v326 = v131;
            sub_16BD1C0("Allocation failed");
            v131 = v326;
            v132 = v318;
            v133 = v309;
          }
          v134 = (_BYTE *)(v133 + 16);
          if ( v123 + 1 <= 1 )
          {
LABEL_166:
            v134[v123] = 0;
            *(_QWORD *)v133 = v123;
            *(_DWORD *)(v133 + 8) = v297;
            *v132 = v133;
            ++*(_DWORD *)(a5 + 12);
            sub_16D1CD0(a5, v131);
            goto LABEL_159;
          }
LABEL_183:
          v314 = v133;
          v322 = v132;
          v332 = v131;
          v168 = memcpy(v134, v121, v123);
          v133 = v314;
          v132 = v322;
          v131 = v332;
          v134 = v168;
          goto LABEL_166;
        }
LABEL_187:
        v315 = v165;
        v323 = v164;
        v170 = memcpy(v166, v153, v155);
        v165 = v315;
        v164 = v323;
        v166 = v170;
        goto LABEL_180;
      }
      if ( v27 != 1 )
        goto LABEL_9;
      v135 = sub_164A820(*(_QWORD *)(v26 - 24));
      v136 = *(_BYTE *)(v135 + 16);
      if ( v136 == 3 )
        goto LABEL_9;
      if ( !v136 )
      {
        v137 = (const void *)sub_1649960(v135);
        v139 = v138;
        v140 = sub_1649960(v26);
        v321 = v141;
        v330 = v140;
        v145 = sub_16D19C0(&v365, v137, v139);
        v146 = (_QWORD *)(v365 + 8LL * v145);
        if ( !*v146 )
          goto LABEL_172;
        if ( *v146 == -8 )
        {
          LODWORD(v367) = v367 - 1;
LABEL_172:
          v281 = (_QWORD *)(v365 + 8LL * v145);
          v147 = malloc(v139 + 25, v137, v139 + 25, v142, v143, v144);
          v150 = v281;
          v151 = (_QWORD *)v147;
          if ( v147 )
          {
LABEL_173:
            v152 = v151 + 3;
            if ( v139 + 1 <= 1 )
              goto LABEL_174;
          }
          else
          {
            if ( v139 != -25 || (v171 = malloc(1, v137, 0, v148, 0, v149), v150 = v281, v151 = 0, !v171) )
            {
              v282 = v151;
              v307 = v150;
              sub_16BD1C0("Allocation failed");
              v150 = v307;
              v151 = v282;
              goto LABEL_173;
            }
            v152 = (void *)(v171 + 24);
            v151 = (_QWORD *)v171;
          }
          v305 = v151;
          v316 = v150;
          v172 = memcpy(v152, v137, v139);
          v151 = v305;
          v150 = v316;
          v152 = v172;
LABEL_174:
          *((_BYTE *)v152 + v139) = 0;
          *v151 = v139;
          v151[1] = v330;
          v151[2] = v321;
          *v150 = v151;
          ++HIDWORD(v366);
          sub_16D1CD0(&v365, v145);
        }
LABEL_9:
        v28 = (const void *)sub_1649960(v26);
        v30 = v29;
        v31 = *(_BYTE *)(v26 + 32) & 0xF;
        v33 = (unsigned int)sub_16D19C0(&v362, v28, v29);
        v34 = (_QWORD *)(v362 + 8 * v33);
        if ( *v34 )
        {
          if ( *v34 != -8 )
            goto LABEL_11;
          LODWORD(v364) = v364 - 1;
        }
        v302 = (__int64 *)(v362 + 8 * v33);
        v311 = v33;
        v115 = malloc(v30 + 17, v30 + 17, v32, v33, v33, v34);
        v117 = v311;
        v118 = v302;
        v119 = v115;
        if ( v115 )
        {
          v120 = v115 + 16;
          goto LABEL_154;
        }
        if ( v30 == -17 )
        {
          v127 = malloc(1, 0, v116, 0, v311, v302);
          v117 = v311;
          v118 = v302;
          v119 = 0;
          if ( v127 )
          {
            v120 = v127 + 16;
            v119 = v127;
            goto LABEL_162;
          }
        }
        v310 = v119;
        v319 = v118;
        v327 = v117;
        sub_16BD1C0("Allocation failed");
        v117 = v327;
        v120 = 16;
        v118 = v319;
        v119 = v310;
LABEL_154:
        if ( v30 + 1 > 1 )
        {
LABEL_162:
          v312 = v119;
          v320 = v118;
          v329 = v117;
          v128 = memcpy((void *)v120, v28, v30);
          v119 = v312;
          v118 = v320;
          v117 = v329;
          v120 = (__int64)v128;
        }
        *(_BYTE *)(v120 + v30) = 0;
        *(_QWORD *)v119 = v30;
        *(_DWORD *)(v119 + 8) = v31;
        *v118 = v119;
        ++HIDWORD(v363);
        sub_16D1CD0(&v362, v117);
      }
LABEL_11:
      v35 = &v405;
      v406.m128i_i64[1] = 0;
      v407.m128i_i64[1] = 0;
      v36 = &v391;
      v406.m128i_i64[0] = (__int64)sub_12D3BB0;
      v37 = &v405;
      v408.m128i_i64[1] = 0;
      v407.m128i_i64[0] = (__int64)sub_12D3BE0;
      v408.m128i_i64[0] = (__int64)sub_12D3C10;
      v38 = sub_12D3B80;
      if ( ((unsigned __int8)sub_12D3B80 & 1) == 0 )
        goto LABEL_13;
      while ( 1 )
      {
        v38 = *(__int64 (__fastcall **)(__int64))((char *)v38 + v36->m128i_i64[0] - 1);
LABEL_13:
        if ( (unsigned __int8)v38((__int64)v36) )
          break;
        while ( 1 )
        {
          v39 = v37[1].m128i_i64[1];
          v38 = (__int64 (__fastcall *)(__int64))v37[1].m128i_i64[0];
          v37 = ++v35;
          v36 = (__m128i *)((char *)&v391 + v39);
          if ( ((unsigned __int8)v38 & 1) != 0 )
            break;
          if ( (unsigned __int8)v38((__int64)v36) )
            goto LABEL_16;
        }
      }
LABEL_16:
      ;
    }
    while ( v335 != (void *)v394.m128i_i64[0] );
LABEL_17:
    ;
  }
  while ( v335 != (void *)v394.m128i_i64[1]
       || v291 != v393.m128i_i64[0]
       || v291 != v393.m128i_i64[1]
       || v283 != v392.m128i_i64[0]
       || v283 != v392.m128i_i64[1]
       || v12 != v391.m128i_i64[0]
       || v12 != v391.m128i_i64[1] );
  v350 = 0;
  v351 = 0;
  v352 = 0;
  v40 = a5;
  src = 0;
  v375 = 0;
  v376 = 0;
  v377 = 0;
  v378 = 0;
  v345 = 0;
  sub_12E0CA0(&v353, a4, (__int64)&v359, (__int64)&v365);
  v41 = (v354 - v353) >> 5;
  v42 = (unsigned int)a3;
  if ( a3 >= (int)v41 )
    v42 = (unsigned int)v41;
  sub_16D4AB0(v390, v42);
  v356 = 0;
  v43 = (v354 - v353) >> 5;
  v357 = 0;
  for ( i = 0; v43; --v43 )
  {
    while ( 1 )
    {
      v44 = sub_22077B0(8);
      v45 = v44;
      if ( v44 )
        sub_1602D10(v44);
      v46 = *(unsigned __int8 *)(a2 + 240);
      v405.m128i_i64[0] = v45;
      sub_16033C0(v45, v46);
      v42 = (__int64)v357;
      if ( v357 != i )
        break;
      sub_12DD860((__int64 *)&v356, v357, (__int64)&v405, v47, v48, v49);
      if ( !--v43 )
        goto LABEL_35;
    }
    if ( v357 )
    {
      *(_QWORD *)v357 = v405.m128i_i64[0];
      v42 = (__int64)v357;
    }
    v42 += 8;
    v357 = (char *)v42;
  }
LABEL_35:
  v346 = 0;
  if ( *(_BYTE *)(a4 + 3288) )
  {
    v42 = 0;
    v184 = sub_16832F0(&v346, 0);
    if ( v184 )
    {
      if ( (unsigned int)(v184 - 5) <= 1 )
      {
        v391.m128i_i64[0] = 90;
        v405.m128i_i64[0] = (__int64)&v406;
        v185 = sub_22409D0(&v405, &v391, 0);
        v42 = 1;
        v405.m128i_i64[0] = v185;
        v406.m128i_i64[0] = v391.m128i_i64[0];
        *(__m128i *)v185 = _mm_load_si128((const __m128i *)&xmmword_4281AD0);
        si128 = _mm_load_si128((const __m128i *)&xmmword_4281AE0);
        qmemcpy((void *)(v185 + 80), "jobserver'", 10);
        *(__m128i *)(v185 + 16) = si128;
        *(__m128i *)(v185 + 32) = _mm_load_si128((const __m128i *)&xmmword_4281AF0);
        *(__m128i *)(v185 + 48) = _mm_load_si128((const __m128i *)&xmmword_4281B00);
        *(__m128i *)(v185 + 64) = _mm_load_si128((const __m128i *)&xmmword_4281B10);
        v405.m128i_i64[1] = v391.m128i_i64[0];
        *(_BYTE *)(v405.m128i_i64[0] + v391.m128i_i64[0]) = 0;
        sub_1C3EFD0(&v405, 1);
        if ( (__m128i *)v405.m128i_i64[0] != &v406 )
        {
          v42 = v406.m128i_i64[0] + 1;
          j_j___libc_free_0(v405.m128i_i64[0], v406.m128i_i64[0] + 1);
        }
        goto LABEL_36;
      }
LABEL_390:
      sub_16BD130("GNU Jobserver support requested, but an error occurred", 1);
    }
  }
LABEL_36:
  v286 = v354;
  if ( v353 != v354 )
  {
    v339 = v353;
    do
    {
      v382 = 0;
      v385 = 128;
      v50 = (_QWORD *)sub_22077B0(0x2000);
      v384 = 0;
      v383 = v50;
      v405.m128i_i64[1] = 2;
      v51 = &v50[8 * (unsigned __int64)v385];
      v405.m128i_i64[0] = (__int64)&unk_49E6B50;
      v406.m128i_i64[0] = 0;
      v406.m128i_i64[1] = -8;
      for ( v407.m128i_i64[0] = 0; v51 != v50; v50 += 8 )
      {
        if ( v50 )
        {
          v52 = v405.m128i_i8[8];
          v50[2] = 0;
          v50[3] = -8;
          *v50 = &unk_49E6B50;
          v50[1] = v52 & 6;
          v50[4] = v407.m128i_i64[0];
        }
      }
      v388 = 0;
      v389 = 1;
      v391.m128i_i64[0] = (__int64)&v345;
      v391.m128i_i64[1] = v339;
      sub_1AB9F40(&v347, *a1, &v382, sub_12D4BD0, &v391);
      v53 = v351;
      v405.m128i_i64[0] = 0;
      if ( v351 == v352 )
      {
        sub_12DDA10((__int64)&v350, v351, &v405);
      }
      else
      {
        if ( v351 )
        {
          *(_QWORD *)v351 = 0;
          v53 = v351;
        }
        v351 = v53 + 8;
      }
      v54 = v347;
      v391 = (__m128i)(unsigned __int64)&v392;
      v55 = v347;
      v381 = 0;
      v392.m128i_i8[0] = 0;
      v405.m128i_i64[0] = (__int64)&unk_49EFBE0;
      v347 = 0;
      v407.m128i_i64[1] = (__int64)&v391;
      v407.m128i_i32[0] = 1;
      v406 = 0u;
      v405.m128i_i64[1] = 0;
      v379 = 0;
      v380 = 0;
      sub_153BF40(v55, &v405, 0, 0, 0, 0);
      if ( v406.m128i_i64[1] != v405.m128i_i64[1] )
        sub_16E7BA0(&v405);
      v56 = *(_QWORD *)(v407.m128i_i64[1] + 8);
      v368 = &v370;
      sub_12D3F10((__int64 *)&v368, *(_BYTE **)v407.m128i_i64[1], *(_QWORD *)v407.m128i_i64[1] + v56);
      sub_16E7BC0(&v405);
      if ( (__m128i *)v391.m128i_i64[0] != &v392 )
        j_j___libc_free_0(v391.m128i_i64[0], v392.m128i_i64[0] + 1);
      v391.m128i_i64[0] = (__int64)&v346;
      v391.m128i_i64[1] = (__int64)&v356;
      v392.m128i_i64[0] = a2;
      v392.m128i_i64[1] = a11;
      v393.m128i_i64[0] = a8;
      v393.m128i_i64[1] = (__int64)&v344;
      v394.m128i_i64[0] = (__int64)&v379;
      v394.m128i_i64[1] = (__int64)a9;
      v395.m128i_i64[0] = a10;
      v395.m128i_i64[1] = (__int64)&v350;
      v396 = &v398;
      if ( v368 == &v370 )
      {
        v398 = _mm_load_si128(&v370);
      }
      else
      {
        v396 = v368;
        v398.m128i_i64[0] = v370.m128i_i64[0];
      }
      v58 = v369;
      v59 = *(_QWORD *)(a4 + 4480);
      v369 = 0;
      v368 = &v370;
      v60 = *(const __m128i **)(a4 + 4496);
      v397 = v58;
      qmemcpy(v399, (const void *)a4, sizeof(v399));
      v61 = *(const __m128i **)(a4 + 4488);
      v370.m128i_i8[0] = 0;
      v400 = v59;
      v401 = 0;
      v402 = 0;
      v403 = 0;
      v62 = (char *)v60 - (char *)v61;
      if ( v60 == v61 )
      {
        v64 = 0;
      }
      else
      {
        if ( v62 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(&v400, v61, v57);
        v63 = sub_22077B0(v62);
        v61 = *(const __m128i **)(a4 + 4488);
        v58 = v397;
        v64 = (__m128i *)v63;
        v59 = v400;
        v60 = *(const __m128i **)(a4 + 4496);
      }
      v401 = v64;
      v402 = v64;
      v403 = &v64->m128i_i8[v62];
      if ( v60 == v61 )
      {
        v66 = v64;
      }
      else
      {
        v65 = v64;
        v66 = (__m128i *)((char *)v64 + (char *)v60 - (char *)v61);
        do
        {
          if ( v65 )
            *v65 = _mm_loadu_si128(v61);
          ++v65;
          ++v61;
        }
        while ( v66 != v65 );
      }
      v67 = _mm_load_si128(&v391);
      v402 = v66;
      v68 = _mm_load_si128(&v392);
      v69 = _mm_load_si128(&v393);
      v410 = &v412;
      v70 = _mm_load_si128(&v394);
      v71 = _mm_load_si128(&v395);
      v404 = v345;
      v405 = v67;
      v406 = v68;
      v407 = v69;
      v408 = v70;
      v409 = v71;
      if ( v396 == &v398 )
      {
        v412 = _mm_load_si128(&v398);
      }
      else
      {
        v410 = v396;
        v412.m128i_i64[0] = v398.m128i_i64[0];
      }
      v411 = v58;
      v397 = 0;
      v396 = &v398;
      v398.m128i_i8[0] = 0;
      qmemcpy(v413, v399, sizeof(v413));
      v414 = v59;
      v415 = v64;
      v416 = v66;
      v417 = &v64->m128i_i8[v62];
      v403 = 0;
      v402 = 0;
      v401 = 0;
      v418 = v345;
      v372 = 0;
      v72 = sub_22077B0(4632);
      if ( v72 )
      {
        v73 = _mm_load_si128(&v405);
        v74 = _mm_load_si128(&v406);
        *(_QWORD *)(v72 + 80) = v72 + 96;
        v75 = _mm_load_si128(&v407);
        v76 = _mm_load_si128(&v408);
        v77 = _mm_load_si128(&v409);
        *(__m128i *)v72 = v73;
        v78 = v410;
        *(__m128i *)(v72 + 16) = v74;
        *(__m128i *)(v72 + 32) = v75;
        *(__m128i *)(v72 + 48) = v76;
        *(__m128i *)(v72 + 64) = v77;
        if ( v78 == &v412 )
        {
          *(__m128i *)(v72 + 96) = _mm_load_si128(&v412);
        }
        else
        {
          *(_QWORD *)(v72 + 80) = v78;
          *(_QWORD *)(v72 + 96) = v412.m128i_i64[0];
        }
        v410 = &v412;
        *(_QWORD *)(v72 + 88) = v411;
        v79 = v414;
        v411 = 0;
        v412.m128i_i8[0] = 0;
        qmemcpy((void *)(v72 + 112), v413, 0x1180u);
        *(_QWORD *)(v72 + 4592) = v79;
        v80 = v415;
        v415 = 0;
        *(_QWORD *)(v72 + 4600) = v80;
        v81 = v416;
        v416 = 0;
        *(_QWORD *)(v72 + 4608) = v81;
        v82 = v417;
        v417 = 0;
        *(_QWORD *)(v72 + 4616) = v82;
        *(_DWORD *)(v72 + 4624) = v418;
      }
      v371[0] = v72;
      v42 = (__int64)v390;
      v373 = sub_12E8D50;
      v372 = sub_12D4D90;
      sub_16D5230(v348, v390, v371);
      if ( v372 )
      {
        v42 = (__int64)v371;
        v372(v371, v371, 3);
      }
      if ( v415 )
      {
        v42 = v417 - (__int8 *)v415;
        j_j___libc_free_0(v415, v417 - (__int8 *)v415);
      }
      if ( v410 != &v412 )
      {
        v42 = v412.m128i_i64[0] + 1;
        j_j___libc_free_0(v410, v412.m128i_i64[0] + 1);
      }
      if ( v401 )
      {
        v42 = v403 - (__int8 *)v401;
        j_j___libc_free_0(v401, v403 - (__int8 *)v401);
      }
      if ( v396 != &v398 )
      {
        v42 = v398.m128i_i64[0] + 1;
        j_j___libc_free_0(v396, v398.m128i_i64[0] + 1);
      }
      v83 = v349;
      if ( v349 )
      {
        if ( &_pthread_key_create )
        {
          v84 = _InterlockedExchangeAdd(v349 + 2, 0xFFFFFFFF);
        }
        else
        {
          v84 = *((_DWORD *)v349 + 2);
          *((_DWORD *)v349 + 2) = v84 - 1;
        }
        if ( v84 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v83 + 16LL))(v83);
          if ( &_pthread_key_create )
          {
            v175 = _InterlockedExchangeAdd(v83 + 3, 0xFFFFFFFF);
          }
          else
          {
            v175 = *((_DWORD *)v83 + 3);
            *((_DWORD *)v83 + 3) = v175 - 1;
          }
          if ( v175 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v83 + 24LL))(v83);
        }
      }
      if ( v368 != &v370 )
      {
        v42 = v370.m128i_i64[0] + 1;
        j_j___libc_free_0(v368, v370.m128i_i64[0] + 1);
      }
      if ( v54 )
      {
        sub_1633490(v54);
        v42 = 736;
        j_j___libc_free_0(v54, 736);
      }
      v85 = v347;
      ++v345;
      if ( v347 )
      {
        sub_1633490(v347);
        v42 = 736;
        j_j___libc_free_0(v85, 736);
      }
      if ( v388 )
      {
        if ( v387 )
        {
          v173 = v386;
          v174 = &v386[2 * v387];
          do
          {
            if ( *v173 != -8 && *v173 != -4 )
            {
              v42 = v173[1];
              if ( v42 )
                sub_161E7C0(v173 + 1);
            }
            v173 += 2;
          }
          while ( v174 != v173 );
        }
        j___libc_free_0(v386);
      }
      if ( v385 )
      {
        v86 = v383;
        v391.m128i_i64[1] = 2;
        v392.m128i_i64[0] = 0;
        v391.m128i_i64[0] = (__int64)&unk_49E6B50;
        v405.m128i_i64[0] = (__int64)&unk_49E6B50;
        v87 = &v383[8 * (unsigned __int64)v385];
        v88 = -8;
        v392.m128i_i64[1] = -8;
        v393.m128i_i64[0] = 0;
        v405.m128i_i64[1] = 2;
        v406.m128i_i64[0] = 0;
        v406.m128i_i64[1] = -16;
        v407.m128i_i64[0] = 0;
        while ( 1 )
        {
          v89 = v86[3];
          if ( v89 != v88 )
          {
            v88 = v406.m128i_i64[1];
            if ( v89 != v406.m128i_i64[1] )
            {
              v90 = v86[7];
              LOBYTE(v42) = v90 != 0;
              if ( v90 != -8 && v90 != 0 && v90 != -16 )
              {
                sub_1649B30(v86 + 5);
                v89 = v86[3];
              }
              v88 = v89;
            }
          }
          *v86 = &unk_49EE2B0;
          if ( v88 != -8 && v88 != 0 && v88 != -16 )
            sub_1649B30(v86 + 1);
          v86 += 8;
          if ( v87 == v86 )
            break;
          v88 = v392.m128i_i64[1];
        }
        v405.m128i_i64[0] = (__int64)&unk_49EE2B0;
        if ( v406.m128i_i64[1] != 0 && v406.m128i_i64[1] != -8 && v406.m128i_i64[1] != -16 )
          sub_1649B30(&v405.m128i_u64[1]);
        v391.m128i_i64[0] = (__int64)&unk_49EE2B0;
        if ( v392.m128i_i64[1] != 0 && v392.m128i_i64[1] != -8 && v392.m128i_i64[1] != -16 )
          sub_1649B30(&v391.m128i_u64[1]);
      }
      j___libc_free_0(v383);
      v339 += 32;
    }
    while ( v286 != v339 );
    v40 = a5;
  }
  sub_16D4EC0(v390);
  if ( v346 && (unsigned int)sub_1682740(&v346) )
    goto LABEL_390;
  if ( *(_QWORD *)a11 )
  {
    v42 = 0;
    if ( (*(unsigned int (__fastcall **)(_QWORD, _QWORD))a11)(*(_QWORD *)(a11 + 8), 0) )
    {
      v95 = 1;
      goto LABEL_113;
    }
  }
  if ( *(int *)(a4 + 4104) >= 0 )
  {
    if ( v344 )
    {
      v91 = a9;
      v269 = a9[1];
      v270 = *a9;
      v271 = v269 + 1;
      if ( (_QWORD *)*a9 == a9 + 2 )
        v272 = 15;
      else
        v272 = a9[2];
      if ( v271 > v272 )
      {
        v42 = a9[1];
        sub_2240BB0(a9, v42, 0, 0, 1);
        v270 = *a9;
      }
      *(_BYTE *)(v270 + v269) = 0;
      a9[1] = v271;
      *(_BYTE *)(*a9 + v269 + 1) = 0;
    }
    else
    {
      *a1 = 0;
    }
    v95 = 0;
    v94 = *(_DWORD **)(a4 + 4480);
    if ( (*v94 & 4) != 0 )
      *v94 ^= 4u;
    goto LABEL_113;
  }
  v176 = v350;
  if ( v350 != v351 )
  {
    v333 = v40;
    do
    {
      v379 = (unsigned __int64)&v380;
      v405 = (__m128i)(unsigned __int64)&unk_49EFC48;
      v407.m128i_i64[1] = (__int64)&v379;
      v407.m128i_i32[0] = 1;
      v406 = 0u;
      sub_16E7A40(&v405, 0, 0, 0);
      sub_153BF40(*(_QWORD *)v176, &v405, 0, 0, 0, 0);
      v177 = a7;
      v392.m128i_i64[1] = 14;
      v391.m128i_i64[0] = v379;
      v391.m128i_i64[1] = DWORD2(v379);
      v392.m128i_i64[0] = (__int64)"<split-module>";
      sub_1509BC0((unsigned int)&v382, a7, v178, v179, v180, v181, v379, DWORD2(v379), (__int64)"<split-module>", 14);
      v182 = (unsigned __int8)v383 & 1;
      LOBYTE(v383) = (2 * v182) | (unsigned __int8)v383 & 0xFD;
      if ( (_BYTE)v182 )
        sub_16BD130("Failed to read bitcode", 1);
      v183 = v382;
      v382 = 0;
      *(_QWORD *)v176 = v183;
      if ( ((unsigned __int8)v383 & 2) != 0 )
        sub_1264230(&v382, a7, v182);
      if ( ((unsigned __int8)v383 & 1) != 0 )
      {
        if ( v382 )
          (*(void (**)(void))(*(_QWORD *)v382 + 8LL))();
      }
      else if ( v382 )
      {
        v336 = v382;
        sub_1633490(v382);
        v177 = 736;
        j_j___libc_free_0(v336, 736);
      }
      v405.m128i_i64[0] = (__int64)&unk_49EFD28;
      sub_16E7960(&v405);
      if ( (__int128 *)v379 != &v380 )
        _libc_free(v379, v177);
      v176 += 8;
    }
    while ( v351 != v176 );
    if ( v176 != v350 )
    {
      v187 = 0;
      while ( 1 )
      {
        v189 = (char *)v376;
        v190 = v377;
        if ( (char *)v376 == v378 )
          break;
        if ( (_DWORD)v377 == 63 )
        {
          LODWORD(v377) = 0;
          v376 += 8;
        }
        else
        {
          LODWORD(v377) = v377 + 1;
        }
        *(_QWORD *)v189 &= ~(1LL << v190);
LABEL_240:
        v188 = v350;
        if ( (v351 - v350) >> 3 <= (unsigned __int64)++v187 )
        {
          v284 = v351;
          v382 = 0;
          v383 = 0;
          v384 = 0x1000000000LL;
          if ( v351 != v350 )
          {
            v301 = v350;
            v338 = 0;
            v206 = 0;
            v207 = v333;
            while ( 1 )
            {
              v208 = *(_QWORD *)(*(_QWORD *)v301 + 32LL);
              v334 = *(_QWORD *)v301 + 24LL;
              if ( v208 != v334 )
                break;
LABEL_285:
              v301 += 8;
              if ( v284 == v301 )
                goto LABEL_286;
            }
            while ( 1 )
            {
              v209 = v208 - 56;
              if ( !v208 )
                v209 = 0;
              v210 = (void *)sub_1649960(v209);
              v212 = v211;
              v341 = *(_QWORD *)v207 + 8LL * *(unsigned int *)(v207 + 8);
              v213 = sub_16D1B30(v207, v210, v211);
              if ( v213 == -1 )
              {
                if ( v341 != *(_QWORD *)v207 + 8LL * *(unsigned int *)(v207 + 8) )
                {
LABEL_265:
                  v338 = v210;
                  v206 = v212;
                  goto LABEL_266;
                }
              }
              else if ( v341 != *(_QWORD *)v207 + 8LL * v213 )
              {
                goto LABEL_265;
              }
              v214 = sub_16D1B30(&v382, v338, v206);
              if ( v214 == -1 || (v215 = &v382[8 * v214], v215 == &v382[8 * (unsigned int)v383]) )
              {
                v306 = 1;
                v325 = 0;
              }
              else
              {
                v325 = *(_DWORD *)(*(_QWORD *)v215 + 8LL);
                v306 = v325 + 1;
              }
              v216 = sub_16D1B30(v207, v338, v206);
              if ( v216 == -1
                || (v217 = *(_QWORD *)v207 + 8LL * v216, v217 == *(_QWORD *)v207 + 8LL * *(unsigned int *)(v207 + 8)) )
              {
                v342 = 0;
              }
              else
              {
                v342 = *(_DWORD *)(*(_QWORD *)v217 + 8LL);
              }
              v219 = (unsigned int)sub_16D19C0(a6, v210, v212);
              v220 = (__int64 *)(*(_QWORD *)a6 + 8 * v219);
              if ( !*v220 )
                goto LABEL_310;
              if ( *v220 == -8 )
                break;
LABEL_279:
              v221 = sub_16D19C0(&v382, v338, v206);
              v224 = &v382[8 * v221];
              if ( !*(_QWORD *)v224 )
                goto LABEL_282;
              if ( *(_QWORD *)v224 == -8 )
              {
                LODWORD(v384) = v384 - 1;
LABEL_282:
                v298 = (__int64 *)&v382[8 * v221];
                v225 = malloc(v206 + 17, v338, v206 + 17, v224, v222, v223);
                v228 = v298;
                v229 = v225;
                if ( !v225 )
                {
                  if ( v206 == -17 )
                  {
                    v253 = malloc(1, v338, 0, v298, v226, v227);
                    v228 = v298;
                    if ( v253 )
                    {
                      v230 = (void *)(v253 + 16);
                      v229 = v253;
LABEL_320:
                      v343 = v228;
                      v254 = memcpy(v230, v338, v206);
                      v228 = v343;
                      v230 = v254;
                      goto LABEL_284;
                    }
                  }
                  v328 = v228;
                  sub_16BD1C0("Allocation failed");
                  v228 = v328;
                }
                v230 = (void *)(v229 + 16);
                if ( v206 + 1 > 1 )
                  goto LABEL_320;
LABEL_284:
                *((_BYTE *)v230 + v206) = 0;
                *(_QWORD *)v229 = v206;
                *(_DWORD *)(v229 + 8) = v306;
                *v228 = v229;
                ++HIDWORD(v383);
                sub_16D1CD0(&v382, v221);
                v208 = *(_QWORD *)(v208 + 8);
                if ( v334 == v208 )
                  goto LABEL_285;
              }
              else
              {
LABEL_266:
                v208 = *(_QWORD *)(v208 + 8);
                if ( v334 == v208 )
                  goto LABEL_285;
              }
            }
            --*(_DWORD *)(a6 + 16);
LABEL_310:
            v287 = v220;
            v292 = v219;
            v245 = malloc(v212 + 17, v210, v212 + 17, a6, v218, v219);
            v247 = v292;
            v248 = v287;
            v249 = v245;
            if ( !v245 )
            {
              if ( v212 == -17 )
              {
                v251 = malloc(1, v210, 0, v246, 0, v292);
                v249 = 0;
                v247 = v292;
                v248 = v287;
                if ( v251 )
                {
                  v250 = (_BYTE *)(v251 + 16);
                  v249 = v251;
                  goto LABEL_317;
                }
              }
              v288 = v248;
              v294 = v247;
              v296 = v249;
              sub_16BD1C0("Allocation failed");
              v249 = v296;
              v247 = v294;
              v248 = v288;
            }
            v250 = (_BYTE *)(v249 + 16);
            if ( v212 + 1 <= 1 )
            {
LABEL_312:
              v250[v212] = 0;
              *(_QWORD *)v249 = v212;
              *(_DWORD *)(v249 + 8) = v342;
              *(_DWORD *)(v249 + 12) = v325;
              *v248 = v249;
              ++*(_DWORD *)(a6 + 12);
              sub_16D1CD0(a6, v247);
              goto LABEL_279;
            }
LABEL_317:
            v293 = v248;
            v295 = v247;
            v299 = v249;
            v252 = memcpy(v250, v210, v212);
            v248 = v293;
            v247 = v295;
            v249 = v299;
            v250 = v252;
            goto LABEL_312;
          }
          goto LABEL_287;
        }
      }
      v191 = (unsigned int)v377 + 8 * (v376 - (_QWORD)src);
      if ( v191 == 0x7FFFFFFFFFFFFFC0LL )
        sub_4262D8((__int64)"vector<bool>::_M_insert_aux");
      v192 = 1;
      if ( v191 )
        v192 = (unsigned int)v377 + 8 * (v376 - (_QWORD)src);
      v193 = __CFADD__(v192, v191);
      v194 = v192 + v191;
      if ( v193 )
      {
        v340 = 0xFFFFFFFFFFFFFF8LL;
        v195 = (char *)sub_22077B0(0xFFFFFFFFFFFFFF8LL);
      }
      else
      {
        if ( v194 > 0x7FFFFFFFFFFFFFC0LL )
          v194 = 0x7FFFFFFFFFFFFFC0LL;
        v340 = 8 * ((v194 + 63) >> 6);
        v195 = (char *)sub_22077B0(v340);
      }
      v196 = v195;
      v197 = v189 - (_BYTE *)src;
      v337 = src;
      if ( v189 != src )
      {
        v324 = v189 - (_BYTE *)src;
        memmove(v195, src, v189 - (_BYTE *)src);
        v197 = v324;
      }
      v198 = (__int64 *)&v196[v197];
      v199 = v189;
      v200 = v190;
      v201 = 0;
      v202 = *v198;
      if ( v190 )
      {
        do
        {
          v204 = (1LL << v201) | v202;
          v202 &= ~(1LL << v201);
          if ( ((1LL << v201) & *(_QWORD *)v199) != 0 )
            v202 = v204;
          v205 = v201 + 1;
          *v198 = v202;
          if ( v201 == 63 )
          {
            v202 = v198[1];
            v199 += 8;
            ++v198;
            v201 = 0;
            v278 = 1;
            v203 = 1;
          }
          else
          {
            v278 = v201 + 2;
            ++v201;
            v203 = 1LL << v205;
          }
          --v200;
        }
        while ( v200 );
        v277 = ~v203;
        v276 = v198;
        if ( v201 == 63 )
        {
          v276 = v198 + 1;
          v278 = 0;
        }
      }
      else
      {
        v276 = v198;
        v277 = -2;
        v278 = 1;
      }
      v273 = (unsigned int)v377;
      *v198 = v277 & v202;
      v274 = v273 + 8 * (v376 - (_QWORD)v189) - v190;
      if ( v274 <= 0 )
      {
LABEL_366:
        v405.m128i_i64[0] = (__int64)v276;
        v405.m128i_i32[2] = v278;
        if ( v337 )
          j_j___libc_free_0(v337, v378 - v337);
        src = v196;
        LODWORD(v375) = 0;
        v378 = &v196[v340];
        v376 = v405.m128i_i64[0];
        LODWORD(v377) = v405.m128i_i32[2];
        goto LABEL_240;
      }
      while ( 2 )
      {
        v275 = *v276 & ~(1LL << v278);
        if ( ((1LL << v190) & *(_QWORD *)v189) != 0 )
          v275 = (1LL << v278) | *v276;
        *v276 = v275;
        if ( v190 == 63 )
        {
          v189 += 8;
          v190 = 0;
          if ( v278 == 63 )
          {
LABEL_365:
            ++v276;
            v278 = 0;
LABEL_360:
            if ( !--v274 )
              goto LABEL_366;
            continue;
          }
        }
        else
        {
          ++v190;
          if ( v278 == 63 )
            goto LABEL_365;
        }
        break;
      }
      ++v278;
      goto LABEL_360;
    }
  }
  v382 = 0;
  v383 = 0;
  v384 = 0x1000000000LL;
LABEL_286:
  v188 = v350;
  v284 = v351;
LABEL_287:
  v391 = (__m128i)(unsigned __int64)&v392;
  v231 = (v284 - v188) >> 3;
  v392.m128i_i8[0] = 0;
  v405 = (__m128i)(unsigned __int64)&v406;
  v406.m128i_i8[0] = 0;
  if ( !(_DWORD)v231 )
  {
    v234 = *a1;
    goto LABEL_295;
  }
  v232 = 0;
  while ( 1 )
  {
    v233 = sub_2241AC0(*(_QWORD *)&v188[v232] + 240LL, off_4CD49B0);
    v188 = v350;
    if ( v233 )
      break;
    v232 += 8;
    if ( 8LL * (unsigned int)v231 == v232 )
    {
      v232 = 0;
      break;
    }
  }
  if ( *(_QWORD *)(*(_QWORD *)&v350[v232] + 248LL) )
  {
    if ( (_DWORD)v231 == 1 )
    {
      v234 = *(_QWORD *)v350;
      *a1 = *(_QWORD *)v350;
      goto LABEL_295;
    }
    v234 = sub_12F5610(&v350, &src, &v405, &v391, a2);
    if ( v234 )
    {
      *a1 = v234;
LABEL_295:
      v235 = *(_QWORD *)(v234 + 32);
      v236 = v234 + 24;
      if ( v235 == v234 + 24 )
        goto LABEL_322;
      while ( 1 )
      {
        v237 = v235 - 56;
        if ( !v235 )
          v237 = 0;
        v238 = sub_1649960(v237);
        v240 = sub_16D1B30(&v362, v238, v239);
        if ( v240 == -1 )
          goto LABEL_299;
        v241 = v362 + 8LL * v240;
        if ( v241 == v362 + 8LL * (unsigned int)v363 )
          goto LABEL_299;
        v242 = *(_DWORD *)(*(_QWORD *)v241 + 8LL);
        v243 = v242 & 0xF;
        if ( (unsigned int)(v242 - 7) <= 1 )
        {
          *(_BYTE *)(v237 + 32) = v243 | *(_BYTE *)(v237 + 32) & 0xC0;
        }
        else
        {
          v244 = v243 | *(_BYTE *)(v237 + 32) & 0xF0;
          *(_BYTE *)(v237 + 32) = v244;
          if ( (v242 & 0xFu) - 7 > 1 && ((v244 & 0x30) == 0 || v243 == 9) )
            goto LABEL_299;
        }
        *(_BYTE *)(v237 + 33) |= 0x40u;
LABEL_299:
        v235 = *(_QWORD *)(v235 + 8);
        if ( v236 == v235 )
        {
          v234 = *a1;
LABEL_322:
          v255 = *(_QWORD *)(v234 + 16);
          v256 = v234 + 8;
          if ( v234 + 8 == v255 )
          {
LABEL_336:
            v95 = 1;
            goto LABEL_337;
          }
          while ( 2 )
          {
            v257 = v255 - 56;
            if ( !v255 )
              v257 = 0;
            v258 = sub_1649960(v257);
            v260 = sub_16D1B30(&v362, v258, v259);
            if ( v260 != -1 )
            {
              v261 = v362 + 8LL * v260;
              if ( v261 != v362 + 8LL * (unsigned int)v363 )
              {
                v262 = *(_DWORD *)(*(_QWORD *)v261 + 8LL);
                v263 = v262 & 0xF;
                if ( (unsigned int)(v262 - 7) <= 1 )
                {
                  *(_BYTE *)(v257 + 32) = v263 | *(_BYTE *)(v257 + 32) & 0xC0;
                  goto LABEL_325;
                }
                v264 = v263 | *(_BYTE *)(v257 + 32) & 0xF0;
                *(_BYTE *)(v257 + 32) = v264;
                if ( (v262 & 0xFu) - 7 <= 1 || (v264 & 0x30) != 0 && v263 != 9 )
LABEL_325:
                  *(_BYTE *)(v257 + 33) |= 0x40u;
              }
            }
            v255 = *(_QWORD *)(v255 + 8);
            if ( v256 == v255 )
              goto LABEL_336;
            continue;
          }
        }
      }
    }
    sub_1C3EFD0(&v405, 1);
  }
  v95 = 0;
  *a1 = 0;
LABEL_337:
  if ( (__m128i *)v405.m128i_i64[0] != &v406 )
    j_j___libc_free_0(v405.m128i_i64[0], v406.m128i_i64[0] + 1);
  if ( (__m128i *)v391.m128i_i64[0] != &v392 )
    j_j___libc_free_0(v391.m128i_i64[0], v392.m128i_i64[0] + 1);
  v42 = HIDWORD(v383);
  v265 = v382;
  if ( HIDWORD(v383) && (_DWORD)v383 )
  {
    v266 = 8LL * (unsigned int)v383;
    v267 = 0;
    do
    {
      v268 = *(_QWORD *)&v265[v267];
      if ( v268 != -8 && v268 )
      {
        _libc_free(v268, v42);
        v265 = v382;
      }
      v267 += 8;
    }
    while ( v266 != v267 );
  }
  _libc_free(v265, v42);
LABEL_113:
  v96 = v357;
  v97 = v356;
  if ( v357 != v356 )
  {
    do
    {
      v98 = *(_QWORD *)v97;
      if ( *(_QWORD *)v97 )
      {
        sub_16025D0(*(_QWORD *)v97, v42, v94, v91, v92, v93);
        v42 = 8;
        j_j___libc_free_0(v98, 8);
      }
      v97 += 8;
    }
    while ( v96 != v97 );
    v97 = v356;
  }
  if ( v97 )
  {
    v42 = i - v97;
    j_j___libc_free_0(v97, i - v97);
  }
  sub_16D56E0(v390);
  v99 = v354;
  v100 = v353;
  if ( v354 != v353 )
  {
    do
    {
      v101 = *(_QWORD *)(v100 + 8);
      v100 += 32;
      j___libc_free_0(v101);
    }
    while ( v99 != v100 );
    v100 = v353;
  }
  if ( v100 )
  {
    v42 = v355 - v100;
    j_j___libc_free_0(v100, v355 - v100);
  }
  if ( src )
  {
    v42 = v378 - (_BYTE *)src;
    j_j___libc_free_0(src, v378 - (_BYTE *)src);
  }
  if ( v350 )
  {
    v42 = v352 - v350;
    j_j___libc_free_0(v350, v352 - v350);
  }
  v102 = v365;
  if ( HIDWORD(v366) && (_DWORD)v366 )
  {
    v103 = 8LL * (unsigned int)v366;
    v104 = 0;
    do
    {
      v105 = *(_QWORD *)(v102 + v104);
      if ( v105 && v105 != -8 )
      {
        _libc_free(v105, v42);
        v102 = v365;
      }
      v104 += 8;
    }
    while ( v103 != v104 );
  }
  _libc_free(v102, v42);
  if ( HIDWORD(v363) )
  {
    v106 = v362;
    if ( (_DWORD)v363 )
    {
      v107 = 8LL * (unsigned int)v363;
      v108 = 0;
      do
      {
        v109 = *(_QWORD *)(v106 + v108);
        if ( v109 != -8 && v109 )
        {
          _libc_free(v109, v42);
          v106 = v362;
        }
        v108 += 8;
      }
      while ( v108 != v107 );
    }
  }
  else
  {
    v106 = v362;
  }
  _libc_free(v106, v42);
  if ( HIDWORD(v360) )
  {
    v110 = v359;
    if ( (_DWORD)v360 )
    {
      v111 = 8LL * (unsigned int)v360;
      v112 = 0;
      do
      {
        v113 = *(_QWORD *)(v110 + v112);
        if ( v113 != -8 && v113 )
        {
          _libc_free(v113, v42);
          v110 = v359;
        }
        v112 += 8;
      }
      while ( v111 != v112 );
    }
  }
  else
  {
    v110 = v359;
  }
  _libc_free(v110, v42);
  return v95;
}
