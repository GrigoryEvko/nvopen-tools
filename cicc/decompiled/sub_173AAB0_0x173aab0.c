// Function: sub_173AAB0
// Address: 0x173aab0
//
__int64 __fastcall sub_173AAB0(
        const __m128i *a1,
        __int64 ***a2,
        double a3,
        double a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  unsigned __int8 *v14; // rdi
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // r13
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v25; // rax
  __int64 v26; // r15
  unsigned __int64 v27; // r13
  char v28; // al
  const __m128i *v29; // r11
  unsigned __int8 **v30; // rcx
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rax
  char v34; // al
  __int16 v35; // ax
  __int64 v36; // rsi
  __int64 v37; // rdx
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 *v40; // rdi
  bool v41; // al
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rcx
  bool v46; // al
  bool v47; // al
  unsigned __int8 *v48; // rax
  __int64 v49; // rcx
  unsigned __int64 v50; // rsi
  __int64 v51; // rcx
  unsigned __int8 *v52; // rax
  __int64 v53; // rcx
  unsigned __int64 v54; // rsi
  __int64 v55; // rcx
  unsigned __int64 v56; // rdx
  char v57; // al
  char v58; // al
  int v59; // ecx
  __int64 v60; // rdi
  char v61; // al
  __int64 v62; // rcx
  __int64 v63; // rax
  char v64; // al
  unsigned __int64 v65; // rdx
  unsigned __int8 *v66; // rax
  char v67; // al
  char v68; // al
  unsigned __int8 **v69; // rdx
  unsigned __int8 *v70; // rax
  char v71; // al
  __int64 *v72; // rax
  char v73; // al
  __int64 v74; // rdx
  __int64 v75; // rcx
  bool v76; // al
  unsigned __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rcx
  bool v81; // al
  char v82; // al
  __int64 v83; // rdx
  char v84; // al
  int v85; // ecx
  __int64 v86; // rdi
  char v87; // al
  __int64 v88; // rcx
  __int64 v89; // rax
  char v90; // al
  bool v91; // al
  __int64 v92; // rax
  __int64 v93; // rcx
  bool v94; // al
  __int64 v95; // rax
  __int64 v96; // r13
  __int64 v97; // rcx
  __int64 v98; // rdx
  __int64 v99; // r9
  __int64 *v100; // r15
  _QWORD *v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // rax
  int v104; // eax
  char v105; // al
  __int64 v106; // rax
  char v107; // al
  int v108; // ecx
  __int64 v109; // rdi
  char v110; // al
  __int64 v111; // rax
  char v112; // al
  _QWORD *v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r13
  const char *v116; // rax
  __int64 **v117; // rdx
  unsigned __int8 *v118; // rax
  bool v119; // al
  __int64 v120; // rax
  __int64 v121; // rcx
  bool v122; // al
  char v123; // al
  __int64 v124; // rdx
  __int64 v125; // rcx
  __int64 *v126; // r11
  __int64 v127; // r13
  const char *v128; // rax
  __int64 **v129; // rdx
  unsigned __int8 *v130; // rax
  char v131; // al
  char v132; // al
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 *v135; // r11
  __int64 v136; // r13
  __int64 v137; // rdi
  __int64 v138; // rax
  bool v139; // al
  char v140; // al
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // rdi
  __int64 v144; // rax
  char v145; // al
  __int64 v146; // rdi
  unsigned __int8 *v147; // r13
  __int64 *v148; // rsi
  unsigned __int8 *v149; // rdx
  bool v150; // zf
  __int64 v151; // rsi
  char v152; // al
  __int64 v153; // rdi
  __int64 v154; // r13
  __int64 *v155; // rax
  _BYTE *v156; // rdi
  __int64 v157; // rax
  double v158; // xmm4_8
  double v159; // xmm5_8
  __int64 v160; // r13
  char v161; // al
  char v162; // al
  char v163; // al
  unsigned int v164; // eax
  __int64 v165; // rsi
  __int64 *v166; // rax
  __int64 v167; // rax
  unsigned int v168; // r8d
  _BYTE *v169; // rdi
  __int64 v170; // rax
  unsigned int v171; // eax
  __int64 *v172; // rax
  __int64 v173; // rax
  unsigned int v174; // r8d
  unsigned int v175; // r15d
  int v176; // r13d
  __int64 v177; // rax
  unsigned int v178; // r9d
  __int64 v179; // rsi
  unsigned int v180; // r15d
  int v181; // r13d
  __int64 v182; // rax
  unsigned int v183; // r9d
  __int64 v184; // rsi
  __int64 v185; // rax
  char v186; // al
  unsigned __int8 **v187; // r13
  char v188; // al
  __int64 v189; // rax
  unsigned __int8 **v190; // r15
  __int64 *v191; // r13
  unsigned __int8 *v192; // rax
  unsigned __int8 *v193; // rax
  _BYTE *v194; // r15
  _BYTE *v195; // rdi
  unsigned __int8 v196; // al
  int v197; // eax
  bool v198; // al
  __int64 *v199; // rax
  char v200; // al
  __int64 v201; // rax
  unsigned int v202; // r15d
  int v203; // eax
  char v204; // al
  __int64 v205; // rdx
  __int64 v206; // rcx
  double v207; // xmm4_8
  double v208; // xmm5_8
  __int64 *v209; // r11
  char v210; // al
  char v211; // al
  __int64 *v212; // r11
  char v213; // al
  __int64 *v214; // r11
  char v215; // al
  __int64 *v216; // r11
  char v217; // al
  char v218; // al
  __int64 *v219; // r11
  __int64 *v220; // r15
  char v221; // al
  char v222; // al
  __int64 v223; // rdx
  __int64 v224; // rcx
  __int64 *v225; // r11
  __int64 *v226; // r15
  char v227; // al
  char v228; // al
  __int64 *v229; // r11
  __int64 v230; // rsi
  __int64 v231; // rdx
  __int64 v232; // rcx
  __int64 **v233; // rbx
  char v234; // al
  __int64 *v235; // r11
  bool v236; // al
  unsigned int v237; // r15d
  __int64 v238; // r15
  int v239; // eax
  __int64 v240; // rdx
  __int64 v241; // rcx
  __int64 v242; // rax
  int v243; // r13d
  __int64 v244; // rdx
  __int64 v245; // rcx
  __int64 v246; // rdx
  __int64 v247; // rcx
  unsigned __int8 *v248; // rax
  unsigned __int8 *v249; // r15
  __int64 v250; // r14
  __int64 v251; // rbx
  __int16 v252; // ax
  unsigned __int8 *v253; // rax
  __int64 v254; // rdx
  unsigned __int8 *v255; // rax
  double v256; // xmm4_8
  double v257; // xmm5_8
  __int64 *v258; // rsi
  char v259; // al
  unsigned __int8 *v260; // rsi
  __int128 v261; // kr00_16
  __int64 v262; // r14
  __int64 *v263; // r13
  unsigned __int8 *v264; // rax
  __int64 v265; // rdx
  unsigned __int8 *v266; // rsi
  __int64 *v267; // rax
  char v268; // al
  __int64 v269; // rdi
  unsigned __int8 *v270; // rax
  char v271; // al
  __int64 v272; // rdi
  __int64 *v273; // rax
  char v274; // al
  __int64 v275; // rdx
  __int64 v276; // rcx
  __int64 v277; // r13
  __int64 v278; // rax
  unsigned __int8 *v279; // r13
  __int64 v280; // r14
  bool v281; // bl
  bool v282; // al
  unsigned __int8 *v283; // rax
  int v284; // eax
  char v285; // al
  char v286; // al
  __int64 v287; // rdi
  __int64 *v288; // rax
  __int64 v289; // rax
  unsigned __int8 *v290; // rax
  char v291; // al
  char v292; // al
  __int64 v293; // rax
  __int64 v294; // rax
  __int64 v295; // r15
  __int64 v296; // rax
  __m128 *v297; // r13
  __int64 v298; // rsi
  __m128 *v299; // r14
  __int64 v300; // rax
  bool v301; // al
  __int64 v302; // rdx
  __int64 v303; // rcx
  __int64 *v304; // r15
  __int64 v305; // rdx
  __int64 v306; // rcx
  unsigned int v307; // r8d
  char v308; // al
  __int64 v309; // rdx
  __int64 v310; // rcx
  __int64 *v311; // r11
  char v312; // al
  char v313; // al
  __int64 *v314; // r13
  __int32 v315; // eax
  __int64 **v316; // rdi
  __int64 v317; // r13
  __int32 v318; // eax
  __int64 **v319; // rdi
  __int64 *v320; // r13
  __int64 *v321; // rdi
  __int32 v322; // eax
  __int64 **v323; // rdi
  __int64 v324; // r13
  __int64 *v325; // rdi
  char v326; // al
  __int64 *v327; // r15
  __int64 v328; // rdx
  __int64 v329; // rcx
  unsigned int v330; // r8d
  char v331; // al
  __int32 v332; // eax
  __int64 **v333; // rdi
  __int64 v334; // rsi
  unsigned __int8 *v335; // rsi
  __int32 v336; // eax
  __int64 **v337; // rdi
  int v338; // eax
  __int64 v339; // rsi
  char v340; // al
  __int64 v341; // rax
  __int64 v342; // rax
  char v343; // al
  char v344; // al
  int v345; // eax
  __int64 v346; // rcx
  __int64 v347; // rsi
  char v348; // al
  __int64 v349; // rax
  char v350; // al
  __int64 v351; // rax
  unsigned int v352; // r15d
  __int64 v353; // rax
  unsigned int v354; // edx
  int v355; // eax
  bool v356; // al
  __int64 *v357; // [rsp+8h] [rbp-168h]
  __int64 *v358; // [rsp+10h] [rbp-160h]
  unsigned int v359; // [rsp+10h] [rbp-160h]
  __int64 *v360; // [rsp+18h] [rbp-158h]
  __int64 v361; // [rsp+18h] [rbp-158h]
  __int64 *v362; // [rsp+20h] [rbp-150h]
  __int64 *v363; // [rsp+20h] [rbp-150h]
  __int64 *v364; // [rsp+20h] [rbp-150h]
  __int64 *v365; // [rsp+20h] [rbp-150h]
  int v366; // [rsp+20h] [rbp-150h]
  bool v367; // [rsp+2Fh] [rbp-141h]
  __int64 v368; // [rsp+40h] [rbp-130h]
  __int64 *v369; // [rsp+40h] [rbp-130h]
  __int64 *v370; // [rsp+40h] [rbp-130h]
  __int64 *v371; // [rsp+40h] [rbp-130h]
  __int64 *v372; // [rsp+40h] [rbp-130h]
  __int64 *v373; // [rsp+40h] [rbp-130h]
  __int64 *v374; // [rsp+48h] [rbp-128h]
  __int64 *v375; // [rsp+48h] [rbp-128h]
  __int64 *v376; // [rsp+48h] [rbp-128h]
  __int64 *v377; // [rsp+48h] [rbp-128h]
  __int64 *v378; // [rsp+48h] [rbp-128h]
  __int64 v379; // [rsp+48h] [rbp-128h]
  __int64 *v380; // [rsp+48h] [rbp-128h]
  __int64 *v381; // [rsp+48h] [rbp-128h]
  __int64 *v382; // [rsp+48h] [rbp-128h]
  __int64 *v383; // [rsp+48h] [rbp-128h]
  __int64 *v384; // [rsp+48h] [rbp-128h]
  __int64 *v385; // [rsp+48h] [rbp-128h]
  __int64 *v386; // [rsp+48h] [rbp-128h]
  __int64 *v387; // [rsp+48h] [rbp-128h]
  __int64 *v388; // [rsp+58h] [rbp-118h]
  unsigned __int8 *v389; // [rsp+58h] [rbp-118h]
  __int64 *v390; // [rsp+58h] [rbp-118h]
  __int64 v391; // [rsp+58h] [rbp-118h]
  __int64 *v392; // [rsp+58h] [rbp-118h]
  __int64 *v393; // [rsp+58h] [rbp-118h]
  __int64 *v394; // [rsp+58h] [rbp-118h]
  __int64 *v395; // [rsp+58h] [rbp-118h]
  __int64 *v396; // [rsp+58h] [rbp-118h]
  __int64 **v397; // [rsp+58h] [rbp-118h]
  __int64 *v398; // [rsp+58h] [rbp-118h]
  __int64 v399; // [rsp+60h] [rbp-110h]
  const __m128i *v400; // [rsp+68h] [rbp-108h]
  unsigned __int64 v401; // [rsp+68h] [rbp-108h]
  const __m128i *v402; // [rsp+68h] [rbp-108h]
  const __m128i *v403; // [rsp+68h] [rbp-108h]
  const __m128i *v404; // [rsp+68h] [rbp-108h]
  __int64 *v405; // [rsp+68h] [rbp-108h]
  unsigned __int64 v406; // [rsp+68h] [rbp-108h]
  const __m128i *v407; // [rsp+68h] [rbp-108h]
  const __m128i *v408; // [rsp+68h] [rbp-108h]
  __int64 v409; // [rsp+70h] [rbp-100h]
  __int64 v410; // [rsp+70h] [rbp-100h]
  __int64 *v412; // [rsp+78h] [rbp-F8h]
  __int64 v413; // [rsp+78h] [rbp-F8h]
  const __m128i *v414; // [rsp+78h] [rbp-F8h]
  const __m128i *v415; // [rsp+78h] [rbp-F8h]
  const __m128i *v416; // [rsp+78h] [rbp-F8h]
  const __m128i *v417; // [rsp+78h] [rbp-F8h]
  const __m128i *v418; // [rsp+78h] [rbp-F8h]
  const __m128i *v419; // [rsp+78h] [rbp-F8h]
  const __m128i *v420; // [rsp+78h] [rbp-F8h]
  const __m128i *v421; // [rsp+78h] [rbp-F8h]
  unsigned __int64 v422; // [rsp+78h] [rbp-F8h]
  const __m128i *v423; // [rsp+78h] [rbp-F8h]
  const __m128i *v424; // [rsp+78h] [rbp-F8h]
  __int64 v425; // [rsp+78h] [rbp-F8h]
  const __m128i *v426; // [rsp+78h] [rbp-F8h]
  unsigned __int64 v427; // [rsp+78h] [rbp-F8h]
  const __m128i *v428; // [rsp+78h] [rbp-F8h]
  __int64 v429; // [rsp+78h] [rbp-F8h]
  const __m128i *v430; // [rsp+78h] [rbp-F8h]
  __int64 v431; // [rsp+78h] [rbp-F8h]
  const __m128i *v432; // [rsp+78h] [rbp-F8h]
  __int64 v433; // [rsp+78h] [rbp-F8h]
  const __m128i *v434; // [rsp+78h] [rbp-F8h]
  int v435; // [rsp+84h] [rbp-ECh] BYREF
  __int64 *v436; // [rsp+88h] [rbp-E8h] BYREF
  __int64 v437; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v438; // [rsp+98h] [rbp-D8h] BYREF
  unsigned __int8 **v439; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v440; // [rsp+A8h] [rbp-C8h] BYREF
  __int128 v441; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 *v442; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int8 *v443; // [rsp+C8h] [rbp-A8h] BYREF
  unsigned __int8 *v444; // [rsp+D0h] [rbp-A0h] BYREF
  int v445; // [rsp+D8h] [rbp-98h]
  __int16 v446; // [rsp+E0h] [rbp-90h]
  unsigned __int8 **v447; // [rsp+F0h] [rbp-80h] BYREF
  __int64 **v448; // [rsp+F8h] [rbp-78h]
  __int16 v449; // [rsp+100h] [rbp-70h]
  __m128 v450; // [rsp+110h] [rbp-60h] BYREF
  __m128i v451; // [rsp+120h] [rbp-50h] BYREF
  _QWORD **v452; // [rsp+130h] [rbp-40h]
  unsigned __int8 **v453; // [rsp+138h] [rbp-38h]

  v10 = (__int64)a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v14 = (unsigned __int8 *)*(a2 - 6);
  v452 = a2;
  v15 = (unsigned __int8 *)*(a2 - 3);
  v450 = v12;
  v451 = v13;
  v16 = sub_13DE4D0(v14, v15, &v450);
  if ( v16 )
  {
    v18 = *(_QWORD *)(v10 + 8);
    if ( !v18 )
      return 0;
    v19 = a1->m128i_i64[0];
    v20 = (__int64)v16;
    do
    {
      v21 = sub_1648700(v18);
      sub_170B990(v19, (__int64)v21);
      v18 = *(_QWORD *)(v18 + 8);
    }
    while ( v18 );
    if ( v10 == v20 )
      v20 = sub_1599EF0(*(__int64 ***)v10);
    sub_164D160(v10, v20, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v22, v23, a9, a10);
    return v10;
  }
  if ( (unsigned __int8)sub_170D400(a1, v10, v17, (__m128i)v12, *(double *)v13.m128i_i64, a5) )
    return v10;
  v25 = (__int64)sub_1707490(
                   (__int64)a1,
                   (unsigned __int8 *)v10,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
  if ( v25 )
    return v25;
  v26 = *(_QWORD *)(v10 - 48);
  v27 = *(_QWORD *)(v10 - 24);
  v409 = a1->m128i_i64[1];
  v450.m128_u64[0] = (unsigned __int64)&v443;
  v450.m128_u64[1] = (unsigned __int64)&v444;
  v451.m128i_i64[0] = (__int64)&v443;
  v451.m128i_i64[1] = (__int64)&v444;
  v28 = sub_1733100(&v450, v10);
  v29 = a1;
  if ( v28 )
    goto LABEL_34;
  v30 = &v444;
  v450.m128_u64[0] = (unsigned __int64)&v443;
  v451.m128i_i64[1] = (__int64)&v443;
  v31 = *(_BYTE *)(v10 + 16);
  v450.m128_u64[1] = (unsigned __int64)&v444;
  v453 = &v444;
  if ( v31 == 52 )
  {
    v56 = *(_QWORD *)(v10 - 48);
    v57 = *(_BYTE *)(v56 + 16);
    if ( v57 == 51 )
    {
      if ( *(_QWORD *)(v56 - 48) )
      {
        v443 = *(unsigned __int8 **)(v56 - 48);
        v402 = a1;
        v422 = v56;
        v76 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 - 24), v56, (__int64)&v444);
        v29 = v402;
        if ( v76 )
          goto LABEL_84;
        v56 = v422;
      }
      v79 = *(_QWORD *)(v56 - 24);
      if ( !v79 )
        goto LABEL_86;
      v80 = v450.m128_u64[0];
      v423 = v29;
      *(_QWORD *)v450.m128_u64[0] = v79;
      v81 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 - 48), v56, v80);
      v29 = v423;
      if ( !v81 )
        goto LABEL_86;
    }
    else
    {
      if ( v57 != 5 || *(_WORD *)(v56 + 18) != 27 )
        goto LABEL_51;
      v104 = *(_DWORD *)(v56 + 20);
      if ( *(_QWORD *)(v56 - 24LL * (v104 & 0xFFFFFFF)) )
      {
        v443 = *(unsigned __int8 **)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
        v406 = v56;
        v105 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 + 24 * (1LL - (v104 & 0xFFFFFFF))));
        v29 = a1;
        if ( v105 )
          goto LABEL_84;
        v56 = v406;
        v104 = *(_DWORD *)(v406 + 20);
      }
      v106 = *(_QWORD *)(v56 + 24 * (1LL - (v104 & 0xFFFFFFF)));
      if ( !v106 )
        goto LABEL_86;
      v430 = v29;
      *(_QWORD *)v450.m128_u64[0] = v106;
      v107 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF)));
      v29 = v430;
      if ( !v107 )
        goto LABEL_86;
    }
LABEL_84:
    v424 = v29;
    if ( sub_17390A0((__int64 **)&v451.m128i_i64[1], *(_QWORD *)(v10 - 24), v77, v78) )
      goto LABEL_34;
    v29 = v424;
    goto LABEL_86;
  }
  if ( v31 != 5 )
    goto LABEL_18;
  if ( *(_WORD *)(v10 + 18) != 28 )
    goto LABEL_17;
  v43 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v73 = *(_BYTE *)(v43 + 16);
  if ( v73 == 51 )
  {
    if ( *(_QWORD *)(v43 - 48) )
    {
      v443 = *(unsigned __int8 **)(v43 - 48);
      v400 = a1;
      v413 = v43;
      v41 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v43 - 24), v43, (__int64)&v444);
      v29 = v400;
      if ( v41 )
        goto LABEL_33;
      v43 = v413;
    }
    v44 = *(_QWORD *)(v43 - 24);
    if ( !v44 )
      goto LABEL_86;
    v45 = v450.m128_u64[0];
    v414 = v29;
    *(_QWORD *)v450.m128_u64[0] = v44;
    v46 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v43 - 48), v43, v45);
    v29 = v414;
    if ( !v46 )
      goto LABEL_86;
    goto LABEL_33;
  }
  if ( v73 == 5 )
  {
    if ( *(_WORD *)(v43 + 18) != 27 )
    {
      v30 = &v444;
      v450.m128_u64[0] = (unsigned __int64)&v443;
      v450.m128_u64[1] = (unsigned __int64)&v444;
      v451.m128i_i64[1] = (__int64)&v443;
      v453 = &v444;
      goto LABEL_17;
    }
    v108 = *(_DWORD *)(v43 + 20);
    v109 = v108 & 0xFFFFFFF;
    if ( *(_QWORD *)(v43 - 24 * v109) )
    {
      v443 = *(unsigned __int8 **)(v43 - 24 * v109);
      v407 = a1;
      v431 = v43;
      v110 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v43 + 24 * (1 - v109)));
      v29 = v407;
      if ( v110 )
        goto LABEL_33;
      v43 = v431;
      v108 = *(_DWORD *)(v431 + 20);
    }
    v111 = *(_QWORD *)(v43 + 24 * (1LL - (v108 & 0xFFFFFFF)));
    if ( !v111
      || (v432 = v29,
          *(_QWORD *)v450.m128_u64[0] = v111,
          v112 = sub_14B2B20(
                   (_QWORD **)&v450.m128_u64[1],
                   *(_QWORD *)(v43 - 24LL * (*(_DWORD *)(v43 + 20) & 0xFFFFFFF))),
          v29 = v432,
          !v112) )
    {
LABEL_86:
      v30 = &v443;
      v82 = *(_BYTE *)(v10 + 16);
      v450.m128_u64[0] = (unsigned __int64)&v443;
      v450.m128_u64[1] = (unsigned __int64)&v444;
      v451.m128i_i64[1] = (__int64)&v443;
      v453 = &v444;
      if ( v82 != 52 )
      {
        if ( v82 != 5 )
          goto LABEL_18;
        goto LABEL_17;
      }
      v56 = *(_QWORD *)(v10 - 48);
LABEL_51:
      v58 = *(_BYTE *)(v56 + 16);
      if ( v58 == 50 )
      {
        if ( !*(_QWORD *)(v56 - 48)
          || (v443 = *(unsigned __int8 **)(v56 - 48),
              v404 = v29,
              v427 = v56,
              v91 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 - 24), v56, (__int64)v30),
              v56 = v427,
              v29 = v404,
              !v91) )
        {
          v92 = *(_QWORD *)(v56 - 24);
          if ( !v92 )
            goto LABEL_18;
          v93 = v450.m128_u64[0];
          v428 = v29;
          *(_QWORD *)v450.m128_u64[0] = v92;
          v94 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 - 48), v56, v93);
          v29 = v428;
          if ( !v94 )
            goto LABEL_18;
        }
      }
      else
      {
        if ( v58 != 5 || *(_WORD *)(v56 + 18) != 26 )
          goto LABEL_18;
        v59 = *(_DWORD *)(v56 + 20);
        v60 = v59 & 0xFFFFFFF;
        if ( *(_QWORD *)(v56 - 24 * v60) )
        {
          v443 = *(unsigned __int8 **)(v56 - 24 * v60);
          v416 = v29;
          v401 = v56;
          v61 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 + 24 * (1 - v60)));
          v29 = v416;
          if ( v61 )
            goto LABEL_59;
          v56 = v401;
          v59 = *(_DWORD *)(v401 + 20);
        }
        v63 = *(_QWORD *)(v56 + 24 * (1LL - (v59 & 0xFFFFFFF)));
        if ( !v63 )
          goto LABEL_18;
        v417 = v29;
        *(_QWORD *)v450.m128_u64[0] = v63;
        v64 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF)));
        v29 = v417;
        if ( !v64 )
          goto LABEL_18;
      }
LABEL_59:
      v418 = v29;
      if ( !sub_173A340((__int64 **)&v451.m128i_i64[1], *(_QWORD *)(v10 - 24), v56, v62) )
      {
LABEL_60:
        v29 = v418;
        goto LABEL_18;
      }
LABEL_34:
      v48 = v443;
      if ( *(_QWORD *)(v10 - 48) )
      {
        v49 = *(_QWORD *)(v10 - 40);
        v50 = *(_QWORD *)(v10 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v50 = v49;
        if ( v49 )
          *(_QWORD *)(v49 + 16) = v50 | *(_QWORD *)(v49 + 16) & 3LL;
      }
      *(_QWORD *)(v10 - 48) = v48;
      if ( v48 )
      {
        v51 = *((_QWORD *)v48 + 1);
        *(_QWORD *)(v10 - 40) = v51;
        if ( v51 )
          *(_QWORD *)(v51 + 16) = (v10 - 40) | *(_QWORD *)(v51 + 16) & 3LL;
        *(_QWORD *)(v10 - 32) = (unsigned __int64)(v48 + 8) | *(_QWORD *)(v10 - 32) & 3LL;
        *((_QWORD *)v48 + 1) = v10 - 48;
      }
      v52 = v444;
      if ( *(_QWORD *)(v10 - 24) )
      {
        v53 = *(_QWORD *)(v10 - 16);
        v54 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v54 = v53;
        if ( v53 )
          *(_QWORD *)(v53 + 16) = v54 | *(_QWORD *)(v53 + 16) & 3LL;
      }
      *(_QWORD *)(v10 - 24) = v52;
      if ( v52 )
      {
        v55 = *((_QWORD *)v52 + 1);
        *(_QWORD *)(v10 - 16) = v55;
        if ( v55 )
          *(_QWORD *)(v55 + 16) = (v10 - 16) | *(_QWORD *)(v55 + 16) & 3LL;
        *(_QWORD *)(v10 - 8) = (unsigned __int64)(v52 + 8) | *(_QWORD *)(v10 - 8) & 3LL;
        *((_QWORD *)v52 + 1) = v10 - 24;
      }
      return v10;
    }
LABEL_33:
    v415 = v29;
    v47 = sub_173A450(
            (__int64 **)&v451.m128i_i64[1],
            *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))),
            *(_DWORD *)(v10 + 20) & 0xFFFFFFF,
            v42);
    v29 = v415;
    if ( v47 )
      goto LABEL_34;
    goto LABEL_86;
  }
LABEL_17:
  if ( *(_WORD *)(v10 + 18) != 28 )
    goto LABEL_18;
  v83 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v84 = *(_BYTE *)(v83 + 16);
  if ( v84 != 50 )
  {
    if ( v84 != 5 || *(_WORD *)(v83 + 18) != 26 )
      goto LABEL_18;
    v85 = *(_DWORD *)(v83 + 20);
    v86 = v85 & 0xFFFFFFF;
    if ( *(_QWORD *)(v83 - 24 * v86) )
    {
      v443 = *(unsigned __int8 **)(v83 - 24 * v86);
      v403 = v29;
      v425 = v83;
      v87 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v83 + 24 * (1 - v86)));
      v29 = v403;
      if ( v87 )
        goto LABEL_97;
      v83 = v425;
      v85 = *(_DWORD *)(v425 + 20);
    }
    v89 = *(_QWORD *)(v83 + 24 * (1LL - (v85 & 0xFFFFFFF)));
    if ( !v89 )
      goto LABEL_18;
    v426 = v29;
    *(_QWORD *)v450.m128_u64[0] = v89;
    v90 = sub_14B2B20((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v83 - 24LL * (*(_DWORD *)(v83 + 20) & 0xFFFFFFF)));
    v29 = v426;
    if ( !v90 )
      goto LABEL_18;
LABEL_97:
    v418 = v29;
    if ( !sub_173A560(
            (__int64 **)&v451.m128i_i64[1],
            *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))),
            *(_DWORD *)(v10 + 20) & 0xFFFFFFF,
            v88) )
      goto LABEL_60;
    goto LABEL_34;
  }
  if ( *(_QWORD *)(v83 - 48) )
  {
    v443 = *(unsigned __int8 **)(v83 - 48);
    v408 = v29;
    v433 = v83;
    v119 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v83 - 24), v83, (__int64)v30);
    v83 = v433;
    v29 = v408;
    if ( v119 )
      goto LABEL_97;
  }
  v120 = *(_QWORD *)(v83 - 24);
  if ( v120 )
  {
    v121 = v450.m128_u64[0];
    v434 = v29;
    *(_QWORD *)v450.m128_u64[0] = v120;
    v122 = sub_171DA10((_QWORD **)&v450.m128_u64[1], *(_QWORD *)(v83 - 48), v83, v121);
    v29 = v434;
    if ( v122 )
      goto LABEL_97;
  }
LABEL_18:
  v32 = *(_QWORD *)(v26 + 8);
  if ( !v32 || *(_QWORD *)(v32 + 8) )
  {
    v33 = *(_QWORD *)(v27 + 8);
    if ( !v33 || *(_QWORD *)(v33 + 8) )
      goto LABEL_26;
  }
  v34 = *(_BYTE *)(v26 + 16);
  switch ( v34 )
  {
    case 51:
      v65 = *(_QWORD *)(v26 - 48);
      if ( !v65 )
        goto LABEL_26;
      v443 = *(unsigned __int8 **)(v26 - 48);
      v66 = *(unsigned __int8 **)(v26 - 24);
      if ( !v66 )
        goto LABEL_26;
      break;
    case 5:
      v35 = *(_WORD *)(v26 + 18);
      if ( v35 != 27 )
        goto LABEL_25;
      v95 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
      v65 = *(_QWORD *)(v26 - 24 * v95);
      if ( !v65 )
        goto LABEL_26;
      v443 = *(unsigned __int8 **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      v66 = *(unsigned __int8 **)(v26 + 24 * (1 - v95));
      if ( !v66 )
        goto LABEL_26;
      break;
    case 50:
      goto LABEL_65;
    default:
      goto LABEL_26;
  }
  v450.m128_u64[0] = v65;
  v419 = v29;
  v444 = v66;
  v450.m128_u64[1] = (unsigned __int64)v66;
  v67 = sub_17336D0((__int64 *)&v450, v27);
  v29 = v419;
  if ( !v67 )
  {
    v68 = *(_BYTE *)(v26 + 16);
    if ( v68 != 50 )
    {
      if ( v68 != 5 )
        goto LABEL_26;
      v35 = *(_WORD *)(v26 + 18);
LABEL_25:
      if ( v35 != 26 )
        goto LABEL_26;
      v103 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
      v69 = *(unsigned __int8 ***)(v26 - 24 * v103);
      if ( !v69 )
        goto LABEL_26;
      v443 = *(unsigned __int8 **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      v70 = *(unsigned __int8 **)(v26 + 24 * (1 - v103));
      if ( !v70 )
        goto LABEL_26;
LABEL_67:
      v444 = v70;
      v448 = (__int64 **)v70;
      v420 = v29;
      v447 = v69;
      v71 = sub_1730BC0((__int64 *)&v447, v27);
      v29 = v420;
      if ( !v71 )
        goto LABEL_26;
      goto LABEL_68;
    }
LABEL_65:
    v69 = *(unsigned __int8 ***)(v26 - 48);
    if ( !v69 )
      goto LABEL_26;
    v443 = *(unsigned __int8 **)(v26 - 48);
    v70 = *(unsigned __int8 **)(v26 - 24);
    if ( !v70 )
      goto LABEL_26;
    goto LABEL_67;
  }
LABEL_68:
  v451.m128i_i16[0] = 257;
  v449 = 257;
  v421 = v29;
  v72 = (__int64 *)sub_172B670(
                     v409,
                     (__int64)v443,
                     (__int64)v444,
                     (__int64 *)&v447,
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
  v25 = sub_15FB630(v72, (__int64)&v450, 0);
  v29 = v421;
  if ( v25 )
    return v25;
LABEL_26:
  v412 = (__int64 *)v29;
  v36 = v10;
  v37 = (__int64)sub_1708300(v29, (unsigned __int8 *)v10, (__m128i)v12, v13, a5);
  v40 = v412;
  if ( !v37 )
  {
    if ( (unsigned __int8)sub_17AD890(v412, v10) )
      return v10;
    v37 = sub_172C850(
            v10,
            v412[1],
            *(double *)v12.m128_u64,
            *(double *)v13.m128i_i64,
            *(double *)a5.m128i_i64,
            v74,
            v75);
    if ( v37 )
    {
      v36 = v10;
      v40 = v412;
      return sub_170E100(v40, v36, v37, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v38, v39, a9, a10);
    }
    v96 = *(_QWORD *)(v10 - 24);
    v405 = v412;
    v97 = v412[330];
    v98 = v412[333];
    v99 = v412[332];
    v100 = *(__int64 **)(v10 - 48);
    v410 = (__int64)v100;
    v429 = v96;
    if ( (unsigned __int8)sub_14BB210((__int64)v100, v96, v98, v97, v10, v99) )
    {
      v451.m128i_i16[0] = 257;
      return sub_15FB440(27, v100, v96, (__int64)&v450, 0);
    }
    v450.m128_u64[0] = (unsigned __int64)&v436;
    v451.m128i_i64[0] = (__int64)&v437;
    if ( sub_17380C0(&v450, v10, v101, v102) )
    {
      v115 = v405[1];
      v116 = sub_1649960(v437);
      v451.m128i_i16[0] = 773;
      v447 = (unsigned __int8 **)v116;
      v448 = v117;
      v450.m128_u64[0] = (unsigned __int64)&v447;
      v450.m128_u64[1] = (unsigned __int64)".not";
      v118 = sub_171CA90(
               v115,
               v437,
               (__int64 *)&v450,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               *(double *)a5.m128i_i64);
      v451.m128i_i16[0] = 257;
      return sub_15FB440(27, v436, (__int64)v118, (__int64)&v450, 0);
    }
    v450.m128_u64[0] = (unsigned __int64)&v436;
    v451.m128i_i64[0] = (__int64)&v437;
    v123 = sub_17386E0(&v450, v10, v113, v114);
    v126 = v405;
    v367 = v123;
    if ( v123 )
    {
      v127 = v405[1];
      v128 = sub_1649960(v437);
      v451.m128i_i16[0] = 773;
      v447 = (unsigned __int8 **)v128;
      v448 = v129;
      v450.m128_u64[0] = (unsigned __int64)&v447;
      v450.m128_u64[1] = (unsigned __int64)".not";
      v130 = sub_171CA90(
               v127,
               v437,
               (__int64 *)&v450,
               *(double *)v12.m128_u64,
               *(double *)v13.m128i_i64,
               *(double *)a5.m128i_i64);
      v451.m128i_i16[0] = 257;
      return sub_15FB440(26, v436, (__int64)v130, (__int64)&v450, 0);
    }
    v368 = v405[1];
    v450.m128_u64[0] = (unsigned __int64)&v442;
    v450.m128_u64[1] = (unsigned __int64)&v442;
    v451.m128i_i64[0] = (__int64)&v443;
    v451.m128i_i64[1] = (__int64)&v444;
    v452 = &v447;
    v131 = *(_BYTE *)(v10 + 16);
    if ( v131 != 52 )
    {
      if ( v131 != 5 || *(_WORD *)(v10 + 18) != 28 )
        goto LABEL_140;
      v338 = *(_DWORD *)(v10 + 20);
      v339 = v338 & 0xFFFFFFF;
      v125 = *(_QWORD *)(v10 - 24 * v339);
      v184 = *(_QWORD *)(v10 + 24 * (1 - v339));
      if ( v125 )
      {
        v442 = (__int64 *)v125;
        v340 = sub_1737B70((__int64 **)&v450.m128_u64[1], v184);
        v126 = v405;
        if ( v340 )
          goto LABEL_219;
        v338 = *(_DWORD *)(v10 + 20);
      }
      v124 = v338 & 0xFFFFFFF;
      v341 = *(_QWORD *)(v10 + 24 * (1 - v124));
      if ( !v341 )
        goto LABEL_140;
      v124 = v450.m128_u64[0];
      *(_QWORD *)v450.m128_u64[0] = v341;
      v184 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      v342 = *(_QWORD *)(v184 + 8);
      if ( !v342 )
        goto LABEL_140;
      if ( *(_QWORD *)(v342 + 8) )
        goto LABEL_140;
      v387 = v126;
      v343 = sub_173A670((__int64 **)&v450.m128_u64[1], v184);
      v126 = v387;
      if ( !v343 )
        goto LABEL_140;
LABEL_219:
      v187 = v447;
      v188 = *((_BYTE *)v447 + 16);
      if ( v188 != 52 )
      {
        if ( v188 != 5 || *((_WORD *)v447 + 9) != 28 )
          goto LABEL_222;
        v345 = *((_DWORD *)v447 + 5);
        v346 = v345 & 0xFFFFFFF;
        v347 = (__int64)v447[-3 * v346];
        v379 = v347;
        if ( v347 )
        {
          v364 = v126;
          v348 = sub_1727B40(v447[3 * (1 - v346)], v347, 24 * (1 - v346), v346);
          v126 = v364;
          if ( v348 )
            goto LABEL_233;
          v345 = *((_DWORD *)v187 + 5);
        }
        v349 = v345 & 0xFFFFFFF;
        v124 = 24 * (1 - v349);
        v125 = *(__int64 *)((char *)v187 + v124);
        v379 = v125;
        if ( !v125 )
          goto LABEL_222;
        v365 = v126;
        v350 = sub_1727B40(v187[-3 * v349], v347, v124, v125);
        v126 = v365;
        if ( !v350 )
          goto LABEL_222;
LABEL_233:
        v362 = v126;
        v451.m128i_i16[0] = 257;
        v199 = (__int64 *)sub_1729500(
                            v368,
                            v444,
                            v379,
                            (__int64 *)&v450,
                            *(double *)v12.m128_u64,
                            *(double *)v13.m128i_i64,
                            *(double *)a5.m128i_i64);
        v451.m128i_i16[0] = 257;
        v25 = sub_15FB440(28, v199, (__int64)v443, (__int64)&v450, 0);
        v126 = v362;
LABEL_226:
        if ( v25 )
          return v25;
LABEL_140:
        v374 = v126;
        v450.m128_u64[0] = (unsigned __int64)&v438;
        v132 = sub_1733C00(&v450, v10, v124, v125);
        v135 = v374;
        if ( !v132 )
          goto LABEL_155;
        v136 = v438;
        if ( (unsigned __int8)(*(_BYTE *)(v438 + 16) - 50) <= 1u )
        {
          v137 = *(_QWORD *)(v438 - 48);
          v138 = *(_QWORD *)(v137 + 8);
          if ( v138 )
            v139 = *(_QWORD *)(v138 + 8) == 0;
          else
            v139 = 0;
          v140 = sub_1727170((_BYTE *)v137, v139, v133, v134);
          v135 = v374;
          if ( v140 )
          {
            v143 = *(_QWORD *)(v438 - 24);
            v144 = *(_QWORD *)(v143 + 8);
            if ( v144 )
              v367 = *(_QWORD *)(v144 + 8) == 0;
            v145 = sub_1727170((_BYTE *)v143, v367, v141, v142);
            v135 = v374;
            if ( v145 )
            {
              v146 = v374[1];
              v450.m128_u64[0] = (unsigned __int64)"notlhs";
              v451.m128i_i16[0] = 259;
              v147 = sub_171CA90(
                       v146,
                       *(_QWORD *)(v438 - 48),
                       (__int64 *)&v450,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              v451.m128i_i16[0] = 259;
              v450.m128_u64[0] = (unsigned __int64)"notrhs";
              v148 = (__int64 *)v147;
              v149 = sub_171CA90(
                       v374[1],
                       *(_QWORD *)(v438 - 24),
                       (__int64 *)&v450,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              v150 = *(_BYTE *)(v438 + 16) == 50;
              v451.m128i_i16[0] = 257;
              if ( !v150 )
                return sub_15FB440(26, v148, (__int64)v149, (__int64)&v450, 0);
              return sub_15FB440(27, v148, (__int64)v149, (__int64)&v450, 0);
            }
          }
          v136 = v438;
        }
        v151 = v136;
        v375 = v135;
        v450.m128_u64[0] = (unsigned __int64)&v436;
        v450.m128_u64[1] = (unsigned __int64)&v437;
        v152 = sub_1733EA0(&v450, v136);
        v135 = v375;
        if ( v152 )
        {
          v153 = v375[1];
          v451.m128i_i16[0] = 257;
          v154 = v437;
          v449 = 257;
          v155 = (__int64 *)sub_171CA90(
                              v153,
                              (__int64)v436,
                              (__int64 *)&v447,
                              *(double *)v12.m128_u64,
                              *(double *)v13.m128i_i64,
                              *(double *)a5.m128i_i64);
          return sub_15FB440(11, v155, v154, (__int64)&v450, 0);
        }
        v160 = v438;
        v450.m128_u64[0] = (unsigned __int64)&v436;
        v451.m128i_i64[0] = (__int64)&v437;
        v161 = *(_BYTE *)(v438 + 16);
        if ( v161 == 49 )
        {
          v151 = *(_QWORD *)(v438 - 48);
          v301 = sub_171DA10(&v450, v151, v133, v134);
          v135 = v375;
          if ( v301 )
          {
            v293 = *(_QWORD *)(v160 - 24);
            if ( v293 )
              goto LABEL_318;
          }
        }
        else
        {
          if ( v161 != 5 || *(_WORD *)(v438 + 18) != 25 )
          {
LABEL_160:
            v162 = *(_BYTE *)(v160 + 16);
            if ( v162 == 49 )
            {
              v169 = *(_BYTE **)(v160 - 48);
              if ( v169[16] > 0x10u || (v170 = *(_QWORD *)(v160 - 24)) == 0 )
              {
LABEL_163:
                v163 = *(_BYTE *)(v160 + 16);
                if ( v163 == 48 )
                {
                  v156 = *(_BYTE **)(v160 - 48);
                  if ( v156[16] > 0x10u )
                    goto LABEL_155;
                  v157 = *(_QWORD *)(v160 - 24);
                  if ( !v157 )
                    goto LABEL_155;
                }
                else
                {
                  if ( v163 != 5 )
                    goto LABEL_155;
                  if ( *(_WORD *)(v160 + 18) != 24 )
                    goto LABEL_155;
                  v133 = *(_DWORD *)(v160 + 20) & 0xFFFFFFF;
                  v156 = *(_BYTE **)(v160 - 24 * v133);
                  if ( !v156 )
                    goto LABEL_155;
                  v157 = *(_QWORD *)(v160 + 24 * (1 - v133));
                  if ( !v157 )
                    goto LABEL_155;
                }
                v150 = v156[16] == 13;
                v437 = v157;
                if ( v150 )
                {
                  v164 = *((_DWORD *)v156 + 8);
                  v133 = *((_QWORD *)v156 + 3);
                  v134 = v164 - 1;
                  v165 = 1LL << ((unsigned __int8)v164 - 1);
                  if ( v164 <= 0x40 )
                    goto LABEL_171;
                }
                else
                {
                  if ( *(_BYTE *)(*(_QWORD *)v156 + 8LL) != 16 )
                    goto LABEL_155;
                  v369 = v135;
                  v167 = sub_15A1020(v156, v151, v133, v134);
                  v135 = v369;
                  if ( !v167 || *(_BYTE *)(v167 + 16) != 13 )
                  {
                    v175 = 0;
                    v176 = *(_DWORD *)(*(_QWORD *)v156 + 32LL);
                    while ( v176 != v175 )
                    {
                      v177 = sub_15A0A60((__int64)v156, v175);
                      if ( !v177 )
                        goto LABEL_199;
                      v134 = *(unsigned __int8 *)(v177 + 16);
                      if ( (_BYTE)v134 != 9 )
                      {
                        if ( (_BYTE)v134 != 13 )
                          goto LABEL_199;
                        v178 = *(_DWORD *)(v177 + 32);
                        v179 = *(_QWORD *)(v177 + 24);
                        v134 = v178 - 1;
                        if ( v178 > 0x40 )
                          v179 = *(_QWORD *)(v179 + 8LL * ((unsigned int)v134 >> 6));
                        if ( (v179 & (1LL << ((unsigned __int8)v178 - 1))) != 0 )
                        {
LABEL_199:
                          v135 = v369;
                          goto LABEL_155;
                        }
                      }
                      ++v175;
                    }
                    goto LABEL_172;
                  }
                  v168 = *(_DWORD *)(v167 + 32);
                  v133 = *(_QWORD *)(v167 + 24);
                  v134 = v168 - 1;
                  v165 = 1LL << ((unsigned __int8)v168 - 1);
                  if ( v168 <= 0x40 )
                    goto LABEL_171;
                }
                v133 = *(_QWORD *)(v133 + 8LL * ((unsigned int)v134 >> 6));
LABEL_171:
                if ( (v133 & v165) == 0 )
                {
LABEL_172:
                  v166 = (__int64 *)sub_15A2B00(
                                      (__int64 *)v156,
                                      *(double *)v12.m128_u64,
                                      *(double *)v13.m128i_i64,
                                      *(double *)a5.m128i_i64);
                  v451.m128i_i16[0] = 257;
                  return sub_15FB440(25, v166, v437, (__int64)&v450, 0);
                }
LABEL_155:
                v376 = v135;
                v450.m128_u64[0] = (unsigned __int64)&v435;
                if ( sub_1733F40(&v450, v10, v133, v134) )
                {
                  *(_WORD *)(v410 + 18) = sub_15FF0F0(v435) | *(_WORD *)(v410 + 18) & 0x8000;
                  return sub_170E100(
                           v376,
                           v10,
                           v410,
                           v12,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64,
                           a6,
                           v158,
                           v159,
                           a9,
                           a10);
                }
                v450.m128_u64[0] = (unsigned __int64)&v441 + 8;
                v204 = sub_13D2630(&v450, (_BYTE *)v429);
                v209 = v376;
                if ( !v204 )
                {
LABEL_248:
                  if ( *(_BYTE *)(v429 + 16) == 13 )
                  {
                    v210 = *(_BYTE *)(v410 + 16);
                    if ( (unsigned __int8)(v210 - 35) <= 0x11u
                      && *(_BYTE *)(*(_QWORD *)(v410 - 24) + 16LL) == 13
                      && v210 == 48 )
                    {
                      v294 = *(_QWORD *)(v410 + 8);
                      if ( v294 )
                      {
                        if ( !*(_QWORD *)(v294 + 8) )
                        {
                          v295 = *(_QWORD *)(v410 - 48);
                          if ( *(_BYTE *)(v295 + 16) == 52 )
                          {
                            v296 = *(_QWORD *)(v295 - 24);
                            if ( *(_BYTE *)(v296 + 16) == 13 )
                            {
                              v398 = v209;
                              v399 = *(_QWORD *)(v410 - 24);
                              sub_13A38D0((__int64)&v447, v296 + 24);
                              sub_16A81B0((__int64)&v447, v399 + 24);
                              sub_1727240((__int64 *)&v447, (__int64 *)(v429 + 24));
                              v451.m128i_i16[0] = 257;
                              v297 = (__m128 *)sub_172C310(
                                                 v398[1],
                                                 *(_QWORD *)(v295 - 48),
                                                 v399,
                                                 (__int64 *)&v450,
                                                 0,
                                                 *(double *)v12.m128_u64,
                                                 *(double *)v13.m128i_i64,
                                                 *(double *)a5.m128i_i64);
                              sub_164B7C0((__int64)v297, v410);
                              if ( v297[1].m128_i8[0] > 0x17u )
                              {
                                v298 = *(_QWORD *)(v10 + 48);
                                v299 = v297 + 3;
                                v450.m128_u64[0] = v298;
                                if ( v298 )
                                {
                                  sub_1623A60((__int64)&v450, v298, 2);
                                  if ( v299 == &v450 )
                                  {
                                    if ( v450.m128_u64[0] )
                                      sub_161E7C0((__int64)&v450, v450.m128_i64[0]);
                                    goto LABEL_328;
                                  }
                                  v334 = v297[3].m128_i64[0];
                                  if ( !v334 )
                                  {
LABEL_348:
                                    v335 = (unsigned __int8 *)v450.m128_u64[0];
                                    v297[3].m128_u64[0] = v450.m128_u64[0];
                                    if ( v335 )
                                      sub_1623210((__int64)&v450, v335, (__int64)&v297[3]);
                                    goto LABEL_328;
                                  }
LABEL_347:
                                  sub_161E7C0((__int64)&v297[3], v334);
                                  goto LABEL_348;
                                }
                                if ( v299 != &v450 )
                                {
                                  v334 = v297[3].m128_i64[0];
                                  if ( v334 )
                                    goto LABEL_347;
                                  v297[3].m128_u64[0] = v450.m128_u64[0];
                                }
                              }
LABEL_328:
                              v300 = sub_15A1070(v297->m128_u64[0], (__int64)&v447);
                              v451.m128i_i16[0] = 257;
                              v10 = sub_15FB440(28, (__int64 *)v297, v300, (__int64)&v450, 0);
                              sub_135E100((__int64 *)&v447);
                              return v10;
                            }
                          }
                        }
                      }
                    }
                  }
                  v381 = v209;
                  v25 = sub_1713A90(
                          v209,
                          (_BYTE *)v10,
                          v12,
                          *(double *)v13.m128i_i64,
                          *(double *)a5.m128i_i64,
                          a6,
                          v207,
                          v208,
                          a9,
                          a10);
                  if ( v25 )
                    return v25;
                  v450.m128_u64[0] = (unsigned __int64)&v444;
                  v450.m128_u64[1] = (unsigned __int64)&v447;
                  v211 = sub_1731410(&v450, v429);
                  v212 = v381;
                  if ( v211
                    || (v450.m128_u64[0] = (unsigned __int64)&v444,
                        v450.m128_u64[1] = (unsigned __int64)&v447,
                        v291 = sub_1731C10(&v450, v429),
                        v212 = v381,
                        v291) )
                  {
                    if ( v444 == (unsigned __int8 *)v410 )
                    {
                      sub_15FB800(v429);
                      v290 = v444;
                      v212 = v381;
                      v444 = (unsigned __int8 *)v447;
                      v447 = (unsigned __int8 **)v290;
                    }
                    if ( v447 == (unsigned __int8 **)v410 )
                    {
                      v385 = v212;
                      sub_15FB800(v10);
                      v289 = v410;
                      v212 = v385;
                      v410 = v429;
                      v429 = v289;
                    }
                  }
                  v382 = v212;
                  v450.m128_u64[0] = (unsigned __int64)&v442;
                  v450.m128_u64[1] = (unsigned __int64)&v443;
                  v213 = sub_1731410(&v450, v410);
                  v214 = v382;
                  if ( v213 )
                  {
                    if ( v442 == (__int64 *)v429 )
                    {
                      v442 = (__int64 *)v443;
                      v443 = (unsigned __int8 *)v429;
                    }
                    else if ( (unsigned __int8 *)v429 != v443 )
                    {
                      goto LABEL_261;
                    }
                    v269 = v382[1];
                    v451.m128i_i16[0] = 257;
                    v449 = 257;
                    v270 = sub_171CA90(
                             v269,
                             v429,
                             (__int64 *)&v447,
                             *(double *)v12.m128_u64,
                             *(double *)v13.m128i_i64,
                             *(double *)a5.m128i_i64);
                    v148 = v442;
                    v149 = v270;
                    return sub_15FB440(26, v148, (__int64)v149, (__int64)&v450, 0);
                  }
                  v450.m128_u64[0] = (unsigned __int64)&v442;
                  v450.m128_u64[1] = (unsigned __int64)&v443;
                  v285 = sub_1731C10(&v450, v410);
                  v214 = v382;
                  if ( v285 )
                  {
                    if ( v442 == (__int64 *)v429 )
                    {
                      v442 = (__int64 *)v443;
                      v443 = (unsigned __int8 *)v429;
                    }
                    else if ( (unsigned __int8 *)v429 != v443 )
                    {
                      goto LABEL_261;
                    }
                    v450.m128_u64[0] = (unsigned __int64)&v444;
                    v286 = sub_13D2630(&v450, (_BYTE *)v429);
                    v214 = v382;
                    if ( !v286 )
                    {
                      v287 = v382[1];
                      v451.m128i_i16[0] = 257;
                      v449 = 257;
                      v288 = (__int64 *)sub_171CA90(
                                          v287,
                                          (__int64)v442,
                                          (__int64 *)&v447,
                                          *(double *)v12.m128_u64,
                                          *(double *)v13.m128i_i64,
                                          *(double *)a5.m128i_i64);
                      return sub_15FB440(26, v288, v429, (__int64)&v450, 0);
                    }
                  }
LABEL_261:
                  v383 = v214;
                  v447 = &v443;
                  v448 = &v442;
                  v215 = sub_17317C0(&v447, v410);
                  v216 = v383;
                  if ( v215 )
                  {
                    v450.m128_u64[1] = (unsigned __int64)&v441 + 8;
                    v450.m128_u64[0] = (unsigned __int64)&v441;
                    v268 = sub_1731D30(&v450, v429);
                    v216 = v383;
                    if ( v268 )
                    {
                      v260 = v443;
                      v261 = v441;
                      if ( v443 == (unsigned __int8 *)v441 )
                        goto LABEL_298;
                      if ( v443 == *((unsigned __int8 **)&v441 + 1) )
                        goto LABEL_292;
                    }
                  }
                  v448 = (__int64 **)&v441 + 1;
                  v371 = v216;
                  v447 = (unsigned __int8 **)&v441;
                  v217 = sub_1731D30(&v447, v410);
                  v216 = v371;
                  if ( !v217 )
                    goto LABEL_263;
                  v450.m128_u64[0] = (unsigned __int64)&v443;
                  v450.m128_u64[1] = (unsigned __int64)&v442;
                  v259 = sub_17317C0(&v450, v429);
                  v216 = v371;
                  if ( !v259 )
                    goto LABEL_263;
                  v260 = v443;
                  v261 = v441;
                  if ( v443 != (unsigned __int8 *)v441 )
                  {
                    if ( v443 == *((unsigned __int8 **)&v441 + 1) )
                    {
LABEL_292:
                      v262 = v216[1];
                      v451.m128i_i16[0] = 257;
                      v449 = 257;
                      v263 = v442;
                      v446 = 257;
                      v264 = sub_171CA90(
                               v262,
                               (__int64)v260,
                               (__int64 *)&v444,
                               *(double *)v12.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)a5.m128i_i64);
                      v265 = v261;
                      v266 = v264;
LABEL_293:
                      v267 = (__int64 *)sub_1729500(
                                          v262,
                                          v266,
                                          v265,
                                          (__int64 *)&v447,
                                          *(double *)v12.m128_u64,
                                          *(double *)v13.m128i_i64,
                                          *(double *)a5.m128i_i64);
                      return sub_15FB440(28, v267, (__int64)v263, (__int64)&v450, 0);
                    }
LABEL_263:
                    v448 = (__int64 **)&v441 + 1;
                    v388 = v216;
                    v447 = (unsigned __int8 **)&v441;
                    v218 = sub_13D5EF0(&v447, v410);
                    v219 = v388;
                    if ( v218 )
                    {
                      v220 = (__int64 *)v441;
                      v372 = v388;
                      v389 = (unsigned __int8 *)*((_QWORD *)&v441 + 1);
                      v450 = (__m128)v441;
                      v221 = sub_1734730(&v450, v429);
                      v149 = v389;
                      v219 = v372;
                      if ( v221 )
                      {
                        v451.m128i_i16[0] = 257;
                        v148 = v220;
                        return sub_15FB440(27, v148, (__int64)v149, (__int64)&v450, 0);
                      }
                    }
                    v448 = (__int64 **)&v441 + 1;
                    v390 = v219;
                    v447 = (unsigned __int8 **)&v441;
                    v222 = sub_17317C0(&v447, v410);
                    v225 = v390;
                    if ( v222 )
                    {
                      v226 = (__int64 *)v441;
                      v373 = v390;
                      v391 = *((_QWORD *)&v441 + 1);
                      v450 = (__m128)v441;
                      v227 = sub_17347F0(&v450, v429);
                      v223 = v391;
                      v225 = v373;
                      if ( v227 )
                      {
                        v451.m128i_i16[0] = 257;
                        return sub_15FB440(27, v226, v391, (__int64)&v450, 0);
                      }
                    }
                    v392 = v225;
                    v450.m128_u64[0] = (unsigned __int64)&v439;
                    v450.m128_u64[1] = (unsigned __int64)&v440;
                    v228 = sub_1735440(&v450, v410, v223, v224);
                    v229 = v392;
                    if ( v228 )
                    {
                      v447 = v439;
                      v271 = sub_13D1F50((__int64 *)&v447, v429);
                      v229 = v392;
                      if ( v271 )
                      {
                        v272 = v392[1];
                        v451.m128i_i16[0] = 257;
                        v449 = 257;
                        v273 = (__int64 *)sub_1729500(
                                            v272,
                                            (unsigned __int8 *)v439,
                                            v440,
                                            (__int64 *)&v447,
                                            *(double *)v12.m128_u64,
                                            *(double *)v13.m128i_i64,
                                            *(double *)a5.m128i_i64);
                        return sub_15FB630(v273, (__int64)&v450, 0);
                      }
                    }
                    v230 = *(_QWORD *)(v10 - 48);
                    if ( *(_BYTE *)(v230 + 16) == 75 )
                    {
                      v254 = *(_QWORD *)(v10 - 24);
                      if ( *(_BYTE *)(v254 + 16) == 75 )
                      {
                        v396 = v229;
                        v255 = sub_172B7E0(
                                 (__int64)v229,
                                 v230,
                                 v254,
                                 *(double *)v12.m128_u64,
                                 *(double *)v13.m128i_i64,
                                 *(double *)a5.m128i_i64);
                        v229 = v396;
                        if ( v255 )
                          return sub_170E100(
                                   v396,
                                   v10,
                                   (__int64)v255,
                                   v12,
                                   *(double *)v13.m128i_i64,
                                   *(double *)a5.m128i_i64,
                                   a6,
                                   v256,
                                   v257,
                                   a9,
                                   a10);
                      }
                    }
                    v393 = v229;
                    v10 = sub_1730740(
                            v229,
                            v10,
                            *(double *)v12.m128_u64,
                            *(double *)v13.m128i_i64,
                            *(double *)a5.m128i_i64);
                    if ( !v10 )
                    {
                      if ( sub_1648CD0(v410, 2) )
                      {
                        v232 = v429;
                        v429 = v410;
                        v410 = v232;
                      }
                      v447 = (unsigned __int8 **)&v439;
                      v233 = *a2;
                      v448 = (__int64 **)&v441;
                      v234 = sub_171FED0(&v447, v429, v231, v232);
                      v235 = v393;
                      if ( !v234 )
                        goto LABEL_277;
                      v236 = sub_1648CD0(v429, 2);
                      v235 = v393;
                      if ( !v236 )
                        goto LABEL_277;
                      v237 = *(_DWORD *)(v441 + 8);
                      if ( v237 > 0x40 )
                      {
                        v384 = v393;
                        v397 = (__int64 **)v441;
                        v284 = sub_16A57B0(v441);
                        v235 = v384;
                        if ( v237 - v284 > 0x40 )
                          goto LABEL_277;
                        v238 = **v397;
                      }
                      else
                      {
                        v238 = *(_QWORD *)v441;
                      }
                      v394 = v235;
                      v239 = sub_16431D0((__int64)v233);
                      v235 = v394;
                      if ( v239 - 1 == v238 )
                      {
                        v450.m128_u64[0] = (unsigned __int64)v439;
                        v450.m128_u64[1] = v429;
                        v274 = sub_17348B0(&v450, v410);
                        v235 = v394;
                        if ( v274 )
                        {
                          v277 = v394[1];
                          v451.m128i_i16[0] = 257;
                          v278 = sub_15A06D0(v233, v410, v275, v276);
                          v279 = sub_17203D0(v277, 40, (__int64)v439, v278, (__int64 *)&v450);
                          v280 = v394[1];
                          v281 = sub_15F2380(v410);
                          v282 = sub_15F2370(v410);
                          v451.m128i_i16[0] = 257;
                          v283 = sub_171CBD0(
                                   v280,
                                   (__int64)v439,
                                   (__int64 *)&v450,
                                   v282,
                                   v281,
                                   *(double *)v12.m128_u64,
                                   *(double *)v13.m128i_i64,
                                   *(double *)a5.m128i_i64);
                          v451.m128i_i16[0] = 257;
                          return sub_14EDD70((__int64)v279, v283, (__int64)v439, (__int64)&v450, 0, 0);
                        }
                      }
LABEL_277:
                      v395 = v235;
                      v444 = (unsigned __int8 *)sub_14B2890(v410, (__int64 *)&v441 + 1, (__int64 *)&v442, 0, 0);
                      v445 = v240;
                      v242 = *(_QWORD *)(v410 + 8);
                      if ( v242 )
                      {
                        if ( !*(_QWORD *)(v242 + 8) )
                        {
                          v243 = (int)v444;
                          if ( (unsigned int)((_DWORD)v444 - 7) > 1
                            && (_DWORD)v444
                            && (unsigned __int8)sub_17279D0((_BYTE *)v429, (__int64)&v441 + 8, v240, v241) )
                          {
                            v450.m128_u64[0] = (unsigned __int64)&v443;
                            if ( sub_171DA10(&v450, (unsigned __int64)v442, v244, v245) )
                            {
                              v258 = (__int64 *)*((_QWORD *)&v441 + 1);
                              *((_QWORD *)&v441 + 1) = v442;
                              v442 = v258;
                            }
                            v450.m128_u64[0] = (unsigned __int64)&v443;
                            if ( sub_171DA10(&v450, *((unsigned __int64 *)&v441 + 1), v246, v247) )
                            {
                              v451.m128i_i16[0] = 257;
                              v248 = sub_171CA90(
                                       v395[1],
                                       (__int64)v442,
                                       (__int64 *)&v450,
                                       *(double *)v12.m128_u64,
                                       *(double *)v13.m128i_i64,
                                       *(double *)a5.m128i_i64);
                              v249 = v443;
                              v250 = (__int64)v248;
                              v451.m128i_i16[0] = 257;
                              v251 = v395[1];
                              v449 = 257;
                              v252 = sub_14AEB90(v243);
                              v253 = sub_17203D0(v251, v252, (__int64)v249, v250, (__int64 *)&v447);
                              return sub_14EDD70((__int64)v253, v249, v250, (__int64)&v450, 0, 0);
                            }
                          }
                        }
                      }
                    }
                    return v10;
                  }
LABEL_298:
                  v262 = v216[1];
                  v451.m128i_i16[0] = 257;
                  v263 = v442;
                  v449 = 257;
                  v446 = 257;
                  v266 = sub_171CA90(
                           v262,
                           (__int64)v260,
                           (__int64 *)&v444,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64);
                  v265 = *((_QWORD *)&v261 + 1);
                  goto LABEL_293;
                }
                v450.m128_u64[0] = (unsigned __int64)&v443;
                v450.m128_u64[1] = (unsigned __int64)&v442;
                if ( (unsigned __int8)sub_17343A0(&v450, v410, v205, v206) )
                {
                  v304 = (__int64 *)*((_QWORD *)&v441 + 1);
                  if ( !sub_1454FB0(*((__int64 *)&v441 + 1)) )
                  {
                    v308 = sub_13CFF40(v304, v410, v305, v306, v307);
                    v311 = v376;
                    if ( !v308 )
                    {
LABEL_335:
                      v386 = v311;
                      v450.m128_u64[0] = (unsigned __int64)&v442;
                      v450.m128_u64[1] = (unsigned __int64)&v443;
                      v312 = sub_1734600(&v450, v410, v309, v310);
                      v209 = v386;
                      if ( v312 )
                      {
                        v313 = sub_17288A0(v386, (__int64)v442, (__int64)v443, 0, v10);
                        v209 = v386;
                        if ( v313 )
                        {
                          v314 = (__int64 *)*((_QWORD *)&v441 + 1);
                          sub_13A38D0((__int64)&v447, (__int64)v443);
                          sub_1727240((__int64 *)&v447, v314);
                          v315 = (int)v448;
                          v316 = *(__int64 ***)v10;
                          LODWORD(v448) = 0;
                          v450.m128_i32[2] = v315;
                          v450.m128_u64[0] = (unsigned __int64)v447;
                          v317 = sub_15A1070((__int64)v316, (__int64)&v450);
                          sub_135E100((__int64 *)&v450);
                          sub_135E100((__int64 *)&v447);
                          sub_170B990(*v386, v410);
                          sub_1593B40((_QWORD *)(v10 - 48), (__int64)v442);
                          sub_1593B40((_QWORD *)(v10 - 24), v317);
                          return v10;
                        }
                      }
                      goto LABEL_248;
                    }
                    sub_13A38D0((__int64)&v447, (__int64)v443);
                    sub_16A7200((__int64)&v447, v304);
                    v318 = (int)v448;
                    v319 = *(__int64 ***)v10;
                    LODWORD(v448) = 0;
                    v450.m128_i32[2] = v318;
                    v450.m128_u64[0] = (unsigned __int64)v447;
                    v320 = (__int64 *)sub_15A1070((__int64)v319, (__int64)&v450);
                    sub_135E100((__int64 *)&v450);
                    v321 = (__int64 *)&v447;
LABEL_339:
                    sub_135E100(v321);
                    v451.m128i_i16[0] = 257;
                    return sub_15FB440(13, v320, (__int64)v442, (__int64)&v450, 0);
                  }
                  sub_13A38D0((__int64)&v444, (__int64)v443);
                  sub_1455DC0((__int64)&v447, (__int64)&v444);
                  sub_16A7800((__int64)&v447, 1u);
                  v322 = (int)v448;
                  v323 = *(__int64 ***)v10;
                  LODWORD(v448) = 0;
                  v450.m128_i32[2] = v322;
                  v450.m128_u64[0] = (unsigned __int64)v447;
                  v324 = sub_15A1070((__int64)v323, (__int64)&v450);
                  sub_135E100((__int64 *)&v450);
                  sub_135E100((__int64 *)&v447);
                  v325 = (__int64 *)&v444;
                }
                else
                {
                  v450.m128_u64[0] = (unsigned __int64)&v442;
                  v450.m128_u64[1] = (unsigned __int64)&v443;
                  v326 = sub_17344D0(&v450, v410, v302, v303);
                  v311 = v376;
                  if ( !v326 )
                    goto LABEL_335;
                  v327 = (__int64 *)*((_QWORD *)&v441 + 1);
                  if ( sub_1454FB0(*((__int64 *)&v441 + 1)) )
                  {
                    sub_13A38D0((__int64)&v444, (__int64)v443);
                    sub_1455DC0((__int64)&v447, (__int64)&v444);
                    sub_16A7800((__int64)&v447, 1u);
                    v336 = (int)v448;
                    v337 = *(__int64 ***)v10;
                    LODWORD(v448) = 0;
                    v450.m128_i32[2] = v336;
                    v450.m128_u64[0] = (unsigned __int64)v447;
                    v320 = (__int64 *)sub_15A1070((__int64)v337, (__int64)&v450);
                    sub_135E100((__int64 *)&v450);
                    sub_135E100((__int64 *)&v447);
                    v321 = (__int64 *)&v444;
                    goto LABEL_339;
                  }
                  v331 = sub_13CFF40(v327, v410, v328, v329, v330);
                  v311 = v376;
                  if ( !v331 )
                    goto LABEL_335;
                  sub_13A38D0((__int64)&v447, (__int64)v443);
                  sub_16A7200((__int64)&v447, v327);
                  v332 = (int)v448;
                  v333 = *(__int64 ***)v10;
                  LODWORD(v448) = 0;
                  v450.m128_i32[2] = v332;
                  v450.m128_u64[0] = (unsigned __int64)v447;
                  v324 = sub_15A1070((__int64)v333, (__int64)&v450);
                  sub_135E100((__int64 *)&v450);
                  v325 = (__int64 *)&v447;
                }
                sub_135E100(v325);
                v451.m128i_i16[0] = 257;
                return sub_15FB440(11, v442, v324, (__int64)&v450, 0);
              }
            }
            else
            {
              if ( v162 != 5 )
                goto LABEL_163;
              if ( *(_WORD *)(v160 + 18) != 25 )
                goto LABEL_163;
              v133 = *(_DWORD *)(v160 + 20) & 0xFFFFFFF;
              v169 = *(_BYTE **)(v160 - 24 * v133);
              if ( !v169 )
                goto LABEL_163;
              v170 = *(_QWORD *)(v160 + 24 * (1 - v133));
              if ( !v170 )
                goto LABEL_163;
            }
            v150 = v169[16] == 13;
            v437 = v170;
            if ( v150 )
            {
              v171 = *((_DWORD *)v169 + 8);
              v133 = *((_QWORD *)v169 + 3);
              v134 = v171 - 1;
              v151 = 1LL << ((unsigned __int8)v171 - 1);
              if ( v171 <= 0x40 )
                goto LABEL_183;
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v169 + 8LL) != 16 )
                goto LABEL_163;
              v370 = v135;
              v173 = sub_15A1020(v169, v151, v133, v134);
              v135 = v370;
              if ( !v173 || *(_BYTE *)(v173 + 16) != 13 )
              {
                v180 = 0;
                v181 = *(_DWORD *)(*(_QWORD *)v169 + 32LL);
                while ( v181 != v180 )
                {
                  v151 = v180;
                  v182 = sub_15A0A60((__int64)v169, v180);
                  if ( !v182 )
                    goto LABEL_210;
                  v134 = *(unsigned __int8 *)(v182 + 16);
                  if ( (_BYTE)v134 != 9 )
                  {
                    if ( (_BYTE)v134 != 13 )
                      goto LABEL_210;
                    v183 = *(_DWORD *)(v182 + 32);
                    v151 = *(_QWORD *)(v182 + 24);
                    v134 = v183 - 1;
                    if ( v183 > 0x40 )
                      v151 = *(_QWORD *)(v151 + 8LL * ((unsigned int)v134 >> 6));
                    if ( (v151 & (1LL << ((unsigned __int8)v183 - 1))) == 0 )
                    {
LABEL_210:
                      v135 = v370;
                      v160 = v438;
                      goto LABEL_163;
                    }
                  }
                  ++v180;
                }
                goto LABEL_184;
              }
              v174 = *(_DWORD *)(v173 + 32);
              v133 = *(_QWORD *)(v173 + 24);
              v134 = v174 - 1;
              v151 = 1LL << ((unsigned __int8)v174 - 1);
              if ( v174 <= 0x40 )
              {
LABEL_183:
                v160 = v438;
                if ( (v133 & v151) == 0 )
                  goto LABEL_163;
LABEL_184:
                v172 = (__int64 *)sub_15A2B00(
                                    (__int64 *)v169,
                                    *(double *)v12.m128_u64,
                                    *(double *)v13.m128i_i64,
                                    *(double *)a5.m128i_i64);
                v451.m128i_i16[0] = 257;
                return sub_15FB440(24, v172, v437, (__int64)&v450, 0);
              }
            }
            v133 = *(_QWORD *)(v133 + 8LL * ((unsigned int)v134 >> 6));
            goto LABEL_183;
          }
          v151 = *(_QWORD *)(v438 - 24LL * (*(_DWORD *)(v438 + 20) & 0xFFFFFFF));
          v292 = sub_14B2B20(&v450, v151);
          v135 = v375;
          if ( v292 )
          {
            v133 = *(_DWORD *)(v160 + 20) & 0xFFFFFFF;
            v293 = *(_QWORD *)(v160 + 24 * (1 - v133));
            if ( v293 )
            {
LABEL_318:
              *(_QWORD *)v451.m128i_i64[0] = v293;
              v451.m128i_i16[0] = 257;
              return sub_15FB440(25, v436, v437, (__int64)&v450, 0);
            }
          }
        }
        v160 = v438;
        goto LABEL_160;
      }
      v194 = *(v447 - 3);
      v379 = (__int64)*(v447 - 6);
      v195 = (_BYTE *)v379;
      if ( !v379 )
        goto LABEL_235;
      v196 = v194[16];
      if ( v196 == 13 )
      {
        v184 = *((unsigned int *)v194 + 8);
        if ( (unsigned int)v184 <= 0x40 )
        {
          v125 = (unsigned int)(64 - v184);
          if ( *((_QWORD *)v194 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v184) )
            goto LABEL_233;
LABEL_235:
          if ( !v194 )
            goto LABEL_222;
          v195 = *(v187 - 6);
LABEL_237:
          v380 = v126;
          v200 = sub_17279D0(v195, v184, v124, v125);
          v126 = v380;
          if ( v200 )
          {
            v379 = (__int64)v194;
            goto LABEL_233;
          }
LABEL_222:
          v189 = *((_QWORD *)v444 + 1);
          if ( !v189 )
            goto LABEL_140;
          if ( *(_QWORD *)(v189 + 8) )
            goto LABEL_140;
          v190 = v447;
          if ( *((_BYTE *)v447 + 16) > 0x10u )
            goto LABEL_140;
          v378 = v126;
          v451.m128i_i16[0] = 257;
          v191 = (__int64 *)sub_1729500(
                              v368,
                              v443,
                              (__int64)v447,
                              (__int64 *)&v450,
                              *(double *)v12.m128_u64,
                              *(double *)v13.m128i_i64,
                              *(double *)a5.m128i_i64);
          v451.m128i_i16[0] = 257;
          v192 = sub_171CA90(
                   v368,
                   (__int64)v190,
                   (__int64 *)&v450,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
          v451.m128i_i16[0] = 257;
          v193 = sub_1729500(
                   v368,
                   (unsigned __int8 *)v442,
                   (__int64)v192,
                   (__int64 *)&v450,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
          v451.m128i_i16[0] = 257;
          v25 = sub_15FB440(27, v191, (__int64)v193, (__int64)&v450, 0);
          v126 = v378;
          goto LABEL_226;
        }
        v360 = v126;
        v197 = sub_16A58F0((__int64)(v194 + 24));
        v184 = (unsigned int)v184;
        v126 = v360;
        v198 = (_DWORD)v184 == v197;
      }
      else
      {
        v125 = *(_QWORD *)v194;
        if ( *(_BYTE *)(*(_QWORD *)v194 + 8LL) != 16 || v196 > 0x10u )
          goto LABEL_237;
        v363 = v126;
        v201 = sub_15A1020(v194, v184, v124, v125);
        v126 = v363;
        if ( !v201 || *(_BYTE *)(v201 + 16) != 13 )
        {
          v351 = *(_QWORD *)v194;
          v361 = (__int64)v194;
          v352 = 0;
          v366 = *(_DWORD *)(v351 + 32);
          while ( v366 != v352 )
          {
            v184 = v352;
            v358 = v126;
            v353 = sub_15A0A60(v361, v352);
            v126 = v358;
            if ( !v353 )
              goto LABEL_234;
            v124 = *(unsigned __int8 *)(v353 + 16);
            if ( (_BYTE)v124 != 9 )
            {
              if ( (_BYTE)v124 != 13 )
                goto LABEL_234;
              v354 = *(_DWORD *)(v353 + 32);
              if ( v354 <= 0x40 )
              {
                v125 = 64 - v354;
                v124 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v354);
                v356 = v124 == *(_QWORD *)(v353 + 24);
              }
              else
              {
                v357 = v358;
                v359 = *(_DWORD *)(v353 + 32);
                v355 = sub_16A58F0(v353 + 24);
                v124 = v359;
                v126 = v357;
                v356 = v359 == v355;
              }
              if ( !v356 )
                goto LABEL_234;
            }
            ++v352;
          }
          goto LABEL_233;
        }
        v202 = *(_DWORD *)(v201 + 32);
        if ( v202 <= 0x40 )
        {
          v125 = 64 - v202;
          v124 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v202);
          v198 = v124 == *(_QWORD *)(v201 + 24);
        }
        else
        {
          v203 = sub_16A58F0(v201 + 24);
          v126 = v363;
          v198 = v202 == v203;
        }
      }
      if ( v198 )
        goto LABEL_233;
LABEL_234:
      v194 = *(v187 - 3);
      goto LABEL_235;
    }
    v184 = *(_QWORD *)(v10 - 24);
    if ( *(_QWORD *)(v10 - 48) )
    {
      v442 = *(__int64 **)(v10 - 48);
      v185 = *(_QWORD *)(v184 + 8);
      if ( !v185 || *(_QWORD *)(v185 + 8) )
      {
LABEL_218:
        v377 = v126;
        *(_QWORD *)v450.m128_u64[0] = v184;
        v184 = *(_QWORD *)(v10 - 48);
        v186 = sub_1737740((__int64 **)&v450.m128_u64[1], v184);
        v126 = v377;
        if ( !v186 )
          goto LABEL_140;
        goto LABEL_219;
      }
      v344 = sub_173A670((__int64 **)&v450.m128_u64[1], v184);
      v126 = v405;
      if ( v344 )
        goto LABEL_219;
      v184 = *(_QWORD *)(v10 - 24);
    }
    if ( !v184 )
      goto LABEL_140;
    goto LABEL_218;
  }
  return sub_170E100(v40, v36, v37, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v38, v39, a9, a10);
}
