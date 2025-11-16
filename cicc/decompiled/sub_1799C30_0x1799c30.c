// Function: sub_1799C30
// Address: 0x1799c30
//
__int64 __fastcall sub_1799C30(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  unsigned __int64 v12; // r12
  __m128 v13; // xmm0
  __m128i v14; // xmm2
  int v15; // r9d
  __int64 v16; // rbx
  __int64 v17; // r12
  _QWORD *v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 i; // rbx
  _QWORD *v23; // rax
  int v24; // eax
  _BYTE *v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rsi
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rcx
  unsigned __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rcx
  double v47; // xmm4_8
  double v48; // xmm5_8
  __int64 v49; // rdx
  unsigned __int8 v50; // al
  unsigned int v51; // ebx
  bool v52; // al
  char v53; // al
  char v54; // dl
  char v55; // al
  unsigned __int8 v56; // al
  __int64 v57; // rbx
  unsigned __int8 v58; // cl
  __int64 v59; // rdx
  __int64 v60; // rdx
  char v61; // al
  __int64 *v62; // rdx
  __int64 v63; // rcx
  __int64 *v64; // rdx
  __int64 v65; // rbx
  __int64 v66; // rdx
  char v67; // dl
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rdi
  __int64 *v71; // rbx
  __int64 v72; // rax
  int v73; // eax
  const char *v74; // rax
  __int64 v75; // rdx
  _QWORD *v76; // rdx
  char v77; // al
  char v78; // cl
  char v79; // cl
  __int64 **v80; // rsi
  __int64 v81; // rbx
  int v82; // eax
  unsigned __int8 v83; // al
  unsigned __int8 v84; // dl
  _QWORD *v85; // rsi
  __int64 v86; // rax
  _QWORD *v87; // rax
  __int64 v88; // rdx
  unsigned __int8 v89; // al
  unsigned int v90; // ebx
  bool v91; // al
  __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // r12
  const char *v96; // rax
  __int64 v97; // rdx
  unsigned __int8 *v98; // rax
  __int64 v99; // r12
  __int64 v100; // rdx
  unsigned __int64 v101; // rcx
  __int64 v102; // rdx
  __int64 v103; // rdx
  unsigned __int64 v104; // rcx
  __int64 v105; // rcx
  __int64 v106; // r14
  __int64 v107; // rdx
  __int64 *v108; // rax
  unsigned int v109; // ebx
  bool v110; // al
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // rax
  __int64 v114; // rax
  int v115; // eax
  int v116; // ebx
  __int64 v117; // rax
  __int64 v118; // rax
  unsigned __int64 v119; // rax
  int v120; // edx
  __int64 **v121; // rbx
  int v122; // eax
  __int16 v123; // si
  __int64 v124; // r12
  int v125; // r14d
  __int64 v126; // r13
  int v127; // eax
  unsigned __int8 *v128; // rax
  unsigned __int8 *v129; // rsi
  __int64 v130; // r12
  const char *v131; // rax
  __int64 v132; // rdx
  _QWORD *v133; // rdx
  double v134; // xmm4_8
  double v135; // xmm5_8
  unsigned __int8 *v136; // rax
  double v137; // xmm4_8
  double v138; // xmm5_8
  int v139; // edx
  int v140; // eax
  __int64 v141; // rcx
  __int64 v142; // rdx
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 *v145; // r8
  __int64 v146; // rax
  __int64 v147; // r8
  __int64 v148; // rbx
  __int64 v149; // r13
  __int64 v150; // r12
  _QWORD *v151; // rax
  double v152; // xmm4_8
  double v153; // xmm5_8
  __int64 v154; // rdx
  __int64 v155; // rdi
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 **v158; // rsi
  int v159; // eax
  int v160; // ebx
  __int64 v161; // rax
  __int64 v162; // rbx
  const char *v163; // rax
  __int64 v164; // rcx
  __int64 v165; // rdx
  __int64 v166; // rdx
  __int64 v167; // rdx
  __int64 v168; // rax
  unsigned int v169; // ebx
  __int64 v170; // rbx
  __int64 v171; // rdi
  unsigned __int64 v172; // rax
  __int64 v173; // rbx
  __int64 v174; // rdi
  __int64 v175; // rsi
  __int64 v176; // rcx
  __int64 v177; // rax
  __int64 v178; // rax
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // r12
  unsigned __int8 *v183; // rax
  unsigned __int8 *v184; // rsi
  __int64 v185; // rdi
  __int64 v186; // rdx
  unsigned __int8 *v187; // rax
  __int64 *v188; // rax
  __int64 v189; // rax
  unsigned int v190; // ebx
  __int64 v191; // rdx
  __int64 v192; // rdi
  __int64 v193; // rdx
  __int64 v194; // rdx
  __int64 v195; // rsi
  _QWORD *v196; // rdi
  _QWORD *v197; // rax
  __int64 v198; // rcx
  unsigned __int8 *v199; // rax
  __int64 v200; // r8
  double v201; // xmm4_8
  double v202; // xmm5_8
  __int64 v203; // rax
  __int64 v204; // rax
  __int64 v205; // rsi
  _QWORD *v206; // rdi
  __int64 v207; // rax
  __int64 v208; // r10
  __int64 v209; // rdx
  int v210; // r8d
  int v211; // r8d
  __int64 *v212; // r9
  __int64 v213; // r8
  int v214; // r9d
  int v215; // r9d
  __int64 *v216; // r11
  __int64 v217; // r11
  int v218; // r9d
  int v219; // r9d
  __int64 *v220; // rcx
  __int64 v221; // r9
  int v222; // ecx
  int v223; // ecx
  __int64 *v224; // rax
  __int64 v225; // rax
  __int64 v226; // rcx
  _QWORD *v227; // rdi
  __int64 v228; // rax
  unsigned __int64 v229; // rcx
  __int64 v230; // rax
  __int64 *v231; // rsi
  __int64 *v232; // rdi
  __int64 v233; // rdx
  __int64 v234; // rsi
  __int64 v235; // rsi
  __int64 v236; // rdx
  unsigned __int8 *v237; // rsi
  double v238; // xmm4_8
  double v239; // xmm5_8
  __int64 v240; // rdx
  __int64 v241; // r12
  const char *v242; // rax
  __int64 v243; // rdx
  unsigned __int8 *v244; // rax
  __int64 v245; // r12
  unsigned int v246; // ebx
  bool v247; // al
  __int64 v248; // rax
  unsigned int v249; // ebx
  __int64 v250; // rax
  unsigned int v251; // ebx
  bool v252; // al
  __int64 v253; // rdx
  __int64 v254; // rcx
  _BYTE *v255; // rax
  _BYTE *v256; // rax
  __int64 v257; // rdx
  __int64 v258; // rcx
  _BYTE *v259; // r9
  __int64 v260; // rax
  _BYTE *v261; // rbx
  unsigned __int64 v262; // rax
  unsigned int v263; // edx
  __int64 *v264; // rax
  int v265; // edx
  char v266; // al
  _BYTE *v267; // r9
  char v268; // al
  __int64 *v269; // r10
  unsigned __int64 v270; // rax
  double v271; // xmm4_8
  double v272; // xmm5_8
  int v273; // edx
  unsigned __int64 v274; // rax
  double v275; // xmm4_8
  double v276; // xmm5_8
  int v277; // edx
  __int64 v278; // rdx
  int v279; // r8d
  __int64 v280; // rax
  unsigned __int64 v281; // rcx
  __int64 v282; // rax
  __int64 v283; // rsi
  __int64 v284; // rsi
  __int64 v285; // rdx
  unsigned __int8 *v286; // rsi
  __int64 v287; // rdx
  __int64 v288; // r8
  double v289; // xmm4_8
  double v290; // xmm5_8
  double v291; // xmm4_8
  double v292; // xmm5_8
  unsigned int v293; // ebx
  bool v294; // al
  __int64 v295; // r12
  const char *v296; // rax
  __int64 v297; // rdx
  __int64 *v298; // rax
  __int64 v299; // rdx
  int v300; // r8d
  __int64 v301; // rax
  unsigned __int64 v302; // rcx
  __int64 v303; // rax
  _QWORD *v304; // rsi
  __int64 v305; // rsi
  __int64 v306; // rdx
  unsigned __int8 *v307; // rsi
  unsigned int v308; // ebx
  __int64 v309; // rax
  int v310; // eax
  bool v311; // al
  __int64 v312; // rax
  unsigned int v313; // ebx
  const char *v314; // rax
  __int64 v315; // rdx
  unsigned int v316; // ebx
  __int64 v317; // rax
  int v318; // eax
  bool v319; // al
  __int64 v320; // rcx
  __int64 v321; // rdx
  __int64 v322; // rcx
  __int64 v323; // rdx
  double v324; // xmm4_8
  double v325; // xmm5_8
  __int64 v326; // r12
  unsigned __int8 *v327; // rax
  __int64 v328; // rdx
  __int64 v329; // rdi
  unsigned __int8 *v330; // rsi
  unsigned __int8 *v331; // rax
  __int64 v332; // r14
  unsigned __int8 *v333; // rax
  __int64 v334; // rdi
  unsigned __int8 *v335; // r12
  unsigned __int8 *v336; // rax
  __int64 v337; // rdi
  __int64 v338; // rsi
  unsigned __int8 *v339; // rax
  __int64 v340; // rsi
  unsigned int v341; // r12d
  __int64 v342; // rax
  unsigned int v343; // ebx
  unsigned int v344; // ebx
  __int64 v345; // rax
  int v346; // eax
  bool v347; // al
  unsigned int v348; // ebx
  __int64 v349; // rax
  int v350; // eax
  bool v351; // al
  __int16 v352; // ax
  unsigned __int8 *v353; // rax
  __int64 v354; // rdi
  __int64 *v355; // rax
  __int16 v356; // ax
  __int64 v357; // r11
  __int64 v358; // rax
  __int64 v359; // rax
  __int64 v360; // r11
  __int64 v361; // r10
  _QWORD *v362; // rax
  _QWORD *v363; // r12
  __int64 v364; // rcx
  unsigned __int64 v365; // rsi
  __int64 v366; // rcx
  __int64 v367; // rcx
  unsigned __int64 v368; // rsi
  __int64 v369; // rcx
  __int64 v370; // rcx
  unsigned __int64 v371; // rsi
  __int64 v372; // rcx
  unsigned __int8 *v373; // rax
  __int64 v374; // [rsp-10h] [rbp-220h]
  int v375; // [rsp+0h] [rbp-210h]
  __int64 *v376; // [rsp+0h] [rbp-210h]
  int v377; // [rsp+0h] [rbp-210h]
  unsigned __int64 *v378; // [rsp+0h] [rbp-210h]
  unsigned __int64 v379; // [rsp+8h] [rbp-208h]
  __int64 v380; // [rsp+8h] [rbp-208h]
  __int64 v381; // [rsp+10h] [rbp-200h]
  int v382; // [rsp+18h] [rbp-1F8h]
  unsigned __int64 v383; // [rsp+18h] [rbp-1F8h]
  __int64 v384; // [rsp+18h] [rbp-1F8h]
  int v385; // [rsp+18h] [rbp-1F8h]
  _BYTE *v386; // [rsp+18h] [rbp-1F8h]
  unsigned __int64 *v387; // [rsp+20h] [rbp-1F0h]
  int v388; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 *v389; // [rsp+20h] [rbp-1F0h]
  __int64 v390; // [rsp+28h] [rbp-1E8h]
  __int64 v391; // [rsp+28h] [rbp-1E8h]
  _QWORD *v392; // [rsp+28h] [rbp-1E8h]
  int v393; // [rsp+28h] [rbp-1E8h]
  __int64 v394; // [rsp+28h] [rbp-1E8h]
  __int64 v395; // [rsp+28h] [rbp-1E8h]
  _BYTE *v396; // [rsp+28h] [rbp-1E8h]
  int v397; // [rsp+30h] [rbp-1E0h]
  __int64 v398; // [rsp+30h] [rbp-1E0h]
  int v399; // [rsp+30h] [rbp-1E0h]
  __int64 v400; // [rsp+30h] [rbp-1E0h]
  _BYTE *v401; // [rsp+30h] [rbp-1E0h]
  __int16 v402; // [rsp+30h] [rbp-1E0h]
  __int64 v403; // [rsp+30h] [rbp-1E0h]
  unsigned int v404; // [rsp+30h] [rbp-1E0h]
  unsigned int v405; // [rsp+30h] [rbp-1E0h]
  int v406; // [rsp+30h] [rbp-1E0h]
  unsigned int v407; // [rsp+30h] [rbp-1E0h]
  unsigned int v408; // [rsp+30h] [rbp-1E0h]
  __int64 v409; // [rsp+30h] [rbp-1E0h]
  __int64 v410; // [rsp+30h] [rbp-1E0h]
  __int64 **v411; // [rsp+38h] [rbp-1D8h]
  __int64 v412; // [rsp+38h] [rbp-1D8h]
  __int64 v413; // [rsp+40h] [rbp-1D0h]
  __int64 v414; // [rsp+40h] [rbp-1D0h]
  __int64 v415; // [rsp+40h] [rbp-1D0h]
  _QWORD *v416; // [rsp+40h] [rbp-1D0h]
  int v417; // [rsp+40h] [rbp-1D0h]
  __int64 v418; // [rsp+40h] [rbp-1D0h]
  int v419; // [rsp+40h] [rbp-1D0h]
  int v420; // [rsp+40h] [rbp-1D0h]
  int v421; // [rsp+40h] [rbp-1D0h]
  int v422; // [rsp+40h] [rbp-1D0h]
  unsigned __int64 v423; // [rsp+40h] [rbp-1D0h]
  int v424; // [rsp+40h] [rbp-1D0h]
  int v425; // [rsp+40h] [rbp-1D0h]
  __int64 v428; // [rsp+68h] [rbp-1A8h]
  __int64 v429; // [rsp+68h] [rbp-1A8h]
  unsigned __int8 *v430; // [rsp+68h] [rbp-1A8h]
  __int64 v431; // [rsp+68h] [rbp-1A8h]
  int v432; // [rsp+7Ch] [rbp-194h] BYREF
  _BYTE *v433; // [rsp+80h] [rbp-190h] BYREF
  _BYTE *v434; // [rsp+88h] [rbp-188h] BYREF
  __int64 v435; // [rsp+90h] [rbp-180h] BYREF
  __int64 *v436; // [rsp+98h] [rbp-178h] BYREF
  __int64 *v437; // [rsp+A0h] [rbp-170h] BYREF
  __int64 *v438; // [rsp+A8h] [rbp-168h] BYREF
  unsigned __int64 v439; // [rsp+B0h] [rbp-160h] BYREF
  int v440; // [rsp+B8h] [rbp-158h]
  __int16 v441; // [rsp+C0h] [rbp-150h]
  __int64 *v442; // [rsp+D0h] [rbp-140h] BYREF
  int v443; // [rsp+D8h] [rbp-138h]
  __int16 v444; // [rsp+E0h] [rbp-130h]
  __int64 *v445; // [rsp+F0h] [rbp-120h] BYREF
  int v446; // [rsp+F8h] [rbp-118h]
  __int16 v447; // [rsp+100h] [rbp-110h]
  unsigned __int64 v448; // [rsp+110h] [rbp-100h] BYREF
  unsigned int v449; // [rsp+118h] [rbp-F8h]
  __int16 v450; // [rsp+120h] [rbp-F0h]
  __int64 *v451; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v452; // [rsp+138h] [rbp-D8h]
  __int16 v453; // [rsp+140h] [rbp-D0h]
  __m128 v454; // [rsp+150h] [rbp-C0h] BYREF
  __m128i v455; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v456; // [rsp+170h] [rbp-A0h]

  v10 = *(_QWORD *)(a2 - 48);
  v11 = *(_QWORD *)(a2 - 72);
  v12 = *(_QWORD *)(a2 - 24);
  v411 = *(__int64 ***)a2;
  if ( *(_BYTE *)(v10 + 16) == 9 || *(_BYTE *)(v12 + 16) == 9 )
  {
    for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
    {
      v23 = sub_1648700(i);
      if ( *((_BYTE *)v23 + 16) == 75 )
      {
        v24 = *((unsigned __int16 *)v23 + 9);
        BYTE1(v24) &= ~0x80u;
        if ( (unsigned int)(v24 - 32) <= 1 )
          return 0;
      }
    }
  }
  v13 = (__m128)_mm_loadu_si128(a1 + 167);
  v14 = _mm_loadu_si128(a1 + 168);
  v428 = a2;
  v456 = a2;
  v454 = v13;
  v455 = v14;
  v413 = sub_13E2B90(v11, v10, v12, (__int64 *)&v454);
  if ( v413 )
  {
    v16 = *(_QWORD *)(a2 + 8);
    if ( v16 )
    {
      v17 = a1->m128i_i64[0];
      do
      {
        v18 = sub_1648700(v16);
        sub_170B990(v17, (__int64)v18);
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v16 );
      if ( a2 == v413 )
        v413 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v413, v13, a4, *(double *)v14.m128i_i64, a6, v19, v20, a9, a10);
      return v428;
    }
    return 0;
  }
  v25 = *(_BYTE **)(a2 - 72);
  v390 = (__int64)v25;
  if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) == 16 && v25[16] <= 0x10u )
  {
    v26 = *(_QWORD *)(*(_QWORD *)v25 + 32LL);
    v454.m128_u64[0] = (unsigned __int64)&v455;
    v454.m128_u64[1] = 0x1000000000LL;
    v397 = v26;
    v414 = (unsigned int)v26;
    if ( (unsigned int)v26 > 0x10 )
    {
      v385 = v26;
      sub_16CD150((__int64)&v454, &v455, (unsigned int)v26, 8, v26, v15);
      LODWORD(v26) = v385;
    }
    v382 = v26;
    v27 = 0;
    v28 = (_QWORD *)sub_16498A0(v390);
    v29 = sub_1643350(v28);
    if ( v382 )
    {
      v383 = v10;
      v30 = v29;
      v379 = v12;
      while ( 1 )
      {
        v36 = sub_15A0A60(v390, v27);
        v39 = v36;
        if ( !v36 )
          break;
        if ( sub_15962C0(v36, (unsigned int)v27, v37, v38) )
        {
          v31 = v27;
        }
        else
        {
          if ( !sub_1593BB0(v39, (unsigned int)v27, v40, v41) )
            break;
          v31 = (unsigned int)(v397 + v27);
        }
        v34 = sub_15A0680(v30, v31, 0);
        v35 = v454.m128_u32[2];
        if ( v454.m128_i32[2] >= (unsigned __int32)v454.m128_i32[3] )
        {
          sub_16CD150((__int64)&v454, &v455, 0, 8, v32, v33);
          v35 = v454.m128_u32[2];
        }
        ++v27;
        *(_QWORD *)(v454.m128_u64[0] + 8 * v35) = v34;
        ++v454.m128_i32[2];
        if ( v27 == v414 )
        {
          v10 = v383;
          v12 = v379;
          goto LABEL_95;
        }
      }
      v10 = v383;
      v12 = v379;
      v42 = v454.m128_u64[0];
      if ( (__m128i *)v454.m128_u64[0] != &v455 )
LABEL_32:
        _libc_free(v42);
    }
    else
    {
LABEL_95:
      v400 = *(_QWORD *)(a2 - 24);
      v392 = *(_QWORD **)(a2 - 48);
      v453 = 257;
      v416 = (_QWORD *)sub_15A01B0((__int64 *)v454.m128_u64[0], v454.m128_u32[2]);
      v87 = sub_1648A60(56, 3u);
      v81 = (__int64)v87;
      if ( v87 )
      {
        sub_15FA660((__int64)v87, v392, v400, v416, (__int64)&v451, 0);
        if ( (__m128i *)v454.m128_u64[0] != &v455 )
          _libc_free(v454.m128_u64[0]);
        return v81;
      }
      v42 = v454.m128_u64[0];
      if ( (__m128i *)v454.m128_u64[0] != &v455 )
        goto LABEL_32;
    }
  }
  v43 = *(_QWORD *)(v11 + 8);
  if ( v43 && !*(_QWORD *)(v43 + 8) && *(_BYTE *)(v11 + 16) == 75 )
  {
    switch ( *(_WORD *)(v11 + 18) & 0x7FFF )
    {
      case 3:
      case 5:
      case 6:
      case 0x21:
      case 0x23:
      case 0x25:
      case 0x27:
      case 0x29:
        *(_WORD *)(v11 + 18) = sub_15FF0F0(*(_WORD *)(v11 + 18) & 0x7FFF) | *(_WORD *)(v11 + 18) & 0x8000;
        if ( *(_QWORD *)(a2 - 48) )
        {
          v100 = *(_QWORD *)(a2 - 40);
          v101 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v101 = v100;
          if ( v100 )
            *(_QWORD *)(v100 + 16) = v101 | *(_QWORD *)(v100 + 16) & 3LL;
        }
        *(_QWORD *)(a2 - 48) = v12;
        if ( v12 )
        {
          v102 = *(_QWORD *)(v12 + 8);
          *(_QWORD *)(a2 - 40) = v102;
          if ( v102 )
            *(_QWORD *)(v102 + 16) = (a2 - 40) | *(_QWORD *)(v102 + 16) & 3LL;
          *(_QWORD *)(a2 - 32) = (v12 + 8) | *(_QWORD *)(a2 - 32) & 3LL;
          *(_QWORD *)(v12 + 8) = a2 - 48;
        }
        if ( *(_QWORD *)(a2 - 24) )
        {
          v103 = *(_QWORD *)(a2 - 16);
          v104 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v104 = v103;
          if ( v103 )
            *(_QWORD *)(v103 + 16) = v104 | *(_QWORD *)(v103 + 16) & 3LL;
        }
        *(_QWORD *)(a2 - 24) = v10;
        v105 = *(_QWORD *)(v10 + 8);
        *(_QWORD *)(a2 - 16) = v105;
        if ( v105 )
          *(_QWORD *)(v105 + 16) = (a2 - 16) | *(_QWORD *)(v105 + 16) & 3LL;
        *(_QWORD *)(a2 - 8) = (v10 + 8) | *(_QWORD *)(a2 - 8) & 3LL;
        *(_QWORD *)(v10 + 8) = a2 - 24;
        sub_15F34F0(a2);
        sub_170B990(a1->m128i_i64[0], v11);
        return v428;
      default:
        break;
    }
  }
  v44 = (__int64)v411;
  if ( *((_BYTE *)v411 + 8) == 16 )
    v44 = *v411[2];
  v45 = 1;
  if ( !sub_1642F90(v44, 1) )
    goto LABEL_46;
  v49 = *(_QWORD *)v10;
  if ( *(_QWORD *)v10 != *(_QWORD *)v11 )
    goto LABEL_46;
  v50 = *(_BYTE *)(v10 + 16);
  if ( v50 == 13 )
  {
    v51 = *(_DWORD *)(v10 + 32);
    if ( v51 <= 0x40 )
      v52 = *(_QWORD *)(v10 + 24) == 1;
    else
      v52 = v51 - 1 == (unsigned int)sub_16A57B0(v10 + 24);
  }
  else
  {
    if ( *(_BYTE *)(v49 + 8) != 16 )
      goto LABEL_131;
    if ( v50 > 0x10u )
      goto LABEL_138;
    v168 = sub_15A1020((_BYTE *)v10, 1, v49, v46);
    if ( !v168 || *(_BYTE *)(v168 + 16) != 13 )
    {
      v308 = 0;
      v421 = *(_DWORD *)(*(_QWORD *)v10 + 32LL);
      while ( v421 != v308 )
      {
        v45 = v308;
        v309 = sub_15A0A60(v10, v308);
        if ( !v309 )
          goto LABEL_130;
        v46 = *(unsigned __int8 *)(v309 + 16);
        if ( (_BYTE)v46 != 9 )
        {
          if ( (_BYTE)v46 != 13 )
            goto LABEL_130;
          v49 = *(unsigned int *)(v309 + 32);
          if ( (unsigned int)v49 <= 0x40 )
          {
            v311 = *(_QWORD *)(v309 + 24) == 1;
          }
          else
          {
            v404 = *(_DWORD *)(v309 + 32);
            v310 = sub_16A57B0(v309 + 24);
            v49 = v404;
            v311 = v404 - 1 == v310;
          }
          if ( !v311 )
            goto LABEL_130;
        }
        ++v308;
      }
      goto LABEL_45;
    }
    v169 = *(_DWORD *)(v168 + 32);
    if ( v169 <= 0x40 )
      v52 = *(_QWORD *)(v168 + 24) == 1;
    else
      v52 = v169 - 1 == (unsigned int)sub_16A57B0(v168 + 24);
  }
  if ( v52 )
  {
LABEL_45:
    v455.m128i_i16[0] = 257;
    return sub_15FB440(27, (__int64 *)v11, v12, (__int64)&v454, 0);
  }
LABEL_130:
  v50 = *(_BYTE *)(v10 + 16);
LABEL_131:
  if ( v50 <= 0x10u )
  {
    if ( sub_1593BB0(v10, v45, v49, v46) )
    {
LABEL_133:
      v106 = a1->m128i_i64[1];
      v451 = (__int64 *)sub_1649960(v11);
      v452 = v107;
      v454.m128_u64[0] = (unsigned __int64)"not.";
      v455.m128i_i16[0] = 1283;
      v454.m128_u64[1] = (unsigned __int64)&v451;
      v108 = (__int64 *)sub_171CA90(v106, v11, (__int64 *)&v454, *(double *)v13.m128_u64, a4, *(double *)v14.m128i_i64);
      v455.m128i_i16[0] = 257;
      return sub_15FB440(26, v108, v12, (__int64)&v454, 0);
    }
    if ( *(_BYTE *)(v10 + 16) == 13 )
    {
      v109 = *(_DWORD *)(v10 + 32);
      if ( v109 <= 0x40 )
        v110 = *(_QWORD *)(v10 + 24) == 0;
      else
        v110 = v109 == (unsigned int)sub_16A57B0(v10 + 24);
      goto LABEL_137;
    }
    if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
    {
      v248 = sub_15A1020((_BYTE *)v10, v45, v49, v46);
      if ( !v248 || *(_BYTE *)(v248 + 16) != 13 )
      {
        v348 = 0;
        v425 = *(_DWORD *)(*(_QWORD *)v10 + 32LL);
        while ( v425 != v348 )
        {
          v45 = v348;
          v349 = sub_15A0A60(v10, v348);
          if ( !v349 )
            goto LABEL_138;
          v49 = *(unsigned __int8 *)(v349 + 16);
          if ( (_BYTE)v49 != 9 )
          {
            if ( (_BYTE)v49 != 13 )
              goto LABEL_138;
            v49 = *(unsigned int *)(v349 + 32);
            if ( (unsigned int)v49 <= 0x40 )
            {
              v351 = *(_QWORD *)(v349 + 24) == 0;
            }
            else
            {
              v408 = *(_DWORD *)(v349 + 32);
              v350 = sub_16A57B0(v349 + 24);
              v49 = v408;
              v351 = v408 == v350;
            }
            if ( !v351 )
              goto LABEL_138;
          }
          ++v348;
        }
        goto LABEL_133;
      }
      v249 = *(_DWORD *)(v248 + 32);
      if ( v249 <= 0x40 )
        v110 = *(_QWORD *)(v248 + 24) == 0;
      else
        v110 = v249 == (unsigned int)sub_16A57B0(v248 + 24);
LABEL_137:
      if ( v110 )
        goto LABEL_133;
    }
  }
LABEL_138:
  if ( sub_1790ED0((_BYTE *)v12, v45, v49, v46) )
    goto LABEL_315;
  if ( (unsigned __int8)sub_1790C20((_BYTE *)v12, v45, v111, v112) )
  {
    v295 = a1->m128i_i64[1];
    v296 = sub_1649960(v11);
    v455.m128i_i16[0] = 1283;
    v451 = (__int64 *)v296;
    v452 = v297;
    v454.m128_u64[0] = (unsigned __int64)"not.";
    v454.m128_u64[1] = (unsigned __int64)&v451;
    v298 = (__int64 *)sub_171CA90(v295, v11, (__int64 *)&v454, *(double *)v13.m128_u64, a4, *(double *)v14.m128i_i64);
    v455.m128i_i16[0] = 257;
    return sub_15FB440(27, v298, v10, (__int64)&v454, 0);
  }
  if ( v11 == v10 )
    goto LABEL_45;
  if ( v11 == v12 )
  {
LABEL_315:
    v455.m128i_i16[0] = 257;
    return sub_15FB440(26, (__int64 *)v11, v10, (__int64)&v454, 0);
  }
  v454.m128_u64[0] = v11;
  if ( sub_13D1F50((__int64 *)&v454, v10) )
  {
    v455.m128i_i16[0] = 257;
    return sub_15FB440(26, (__int64 *)v10, v12, (__int64)&v454, 0);
  }
  v45 = v12;
  v454.m128_u64[0] = v11;
  if ( sub_13D1F50((__int64 *)&v454, v12) )
  {
    v455.m128i_i16[0] = 257;
    return sub_15FB440(27, (__int64 *)v10, v12, (__int64)&v454, 0);
  }
LABEL_46:
  v53 = *((_BYTE *)v411 + 8);
  v54 = v53;
  if ( v53 == 16 )
    v54 = *(_BYTE *)(*v411[2] + 8);
  if ( v54 != 11 )
    goto LABEL_49;
  v88 = *(_QWORD *)v11;
  LOBYTE(v88) = *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16;
  if ( (_BYTE)v88 != (v53 == 16) )
    goto LABEL_49;
  v89 = *(_BYTE *)(v10 + 16);
  if ( v89 == 13 )
  {
    v90 = *(_DWORD *)(v10 + 32);
    if ( v90 <= 0x40 )
      v91 = *(_QWORD *)(v10 + 24) == 1;
    else
      v91 = v90 - 1 == (unsigned int)sub_16A57B0(v10 + 24);
    goto LABEL_103;
  }
  v88 = *(_QWORD *)v10;
  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 || v89 > 0x10u )
    goto LABEL_105;
  v189 = sub_15A1020((_BYTE *)v10, v45, v88, v46);
  if ( v189 && *(_BYTE *)(v189 + 16) == 13 )
  {
    v190 = *(_DWORD *)(v189 + 32);
    if ( v190 <= 0x40 )
      v91 = *(_QWORD *)(v189 + 24) == 1;
    else
      v91 = v190 - 1 == (unsigned int)sub_16A57B0(v189 + 24);
LABEL_103:
    if ( !v91 )
      goto LABEL_105;
    goto LABEL_104;
  }
  v423 = v12;
  v341 = 0;
  v406 = *(_DWORD *)(*(_QWORD *)v10 + 32LL);
  while ( v406 != v341 )
  {
    v45 = v341;
    v342 = sub_15A0A60(v10, v341);
    if ( !v342 )
      goto LABEL_544;
    v46 = *(unsigned __int8 *)(v342 + 16);
    if ( (_BYTE)v46 != 9 )
    {
      if ( (_BYTE)v46 != 13 )
        goto LABEL_544;
      v343 = *(_DWORD *)(v342 + 32);
      if ( v343 <= 0x40 )
      {
        if ( *(_QWORD *)(v342 + 24) != 1 )
        {
LABEL_544:
          v12 = v423;
          goto LABEL_105;
        }
      }
      else if ( (unsigned int)sub_16A57B0(v342 + 24) != v343 - 1 )
      {
        goto LABEL_544;
      }
    }
    ++v341;
  }
  v12 = v423;
LABEL_104:
  if ( sub_1790ED0((_BYTE *)v12, v45, v88, v46) )
  {
    v455.m128i_i16[0] = 257;
    v428 = (__int64)sub_1648A60(56, 1u);
    if ( v428 )
      sub_15FC690(v428, v11, (__int64)v411, (__int64)&v454, 0);
    return v428;
  }
LABEL_105:
  if ( (unsigned __int8)sub_1790D60((_BYTE *)v10, v45, v88, v46) && sub_1790ED0((_BYTE *)v12, v45, v92, v46) )
  {
    v455.m128i_i16[0] = 257;
    v428 = (__int64)sub_1648A60(56, 1u);
    if ( v428 )
      sub_15FC810(v428, v11, (__int64)v411, (__int64)&v454, 0);
    return v428;
  }
  if ( *(_BYTE *)(v10 + 16) > 0x10u )
    goto LABEL_49;
  if ( !sub_1593BB0(v10, v45, v92, v46) )
  {
    if ( *(_BYTE *)(v10 + 16) == 13 )
    {
      v246 = *(_DWORD *)(v10 + 32);
      if ( v246 <= 0x40 )
        v247 = *(_QWORD *)(v10 + 24) == 0;
      else
        v247 = v246 == (unsigned int)sub_16A57B0(v10 + 24);
      if ( !v247 )
        goto LABEL_111;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
        goto LABEL_110;
      v250 = sub_15A1020((_BYTE *)v10, v45, v93, v46);
      if ( v250 && *(_BYTE *)(v250 + 16) == 13 )
      {
        v251 = *(_DWORD *)(v250 + 32);
        if ( v251 <= 0x40 )
          v252 = *(_QWORD *)(v250 + 24) == 0;
        else
          v252 = v251 == (unsigned int)sub_16A57B0(v250 + 24);
        if ( !v252 )
          goto LABEL_110;
      }
      else
      {
        v344 = 0;
        v424 = *(_DWORD *)(*(_QWORD *)v10 + 32LL);
        while ( v424 != v344 )
        {
          v45 = v344;
          v345 = sub_15A0A60(v10, v344);
          if ( !v345 )
            goto LABEL_110;
          v93 = *(unsigned __int8 *)(v345 + 16);
          if ( (_BYTE)v93 != 9 )
          {
            if ( (_BYTE)v93 != 13 )
              goto LABEL_110;
            v93 = *(unsigned int *)(v345 + 32);
            if ( (unsigned int)v93 <= 0x40 )
            {
              v347 = *(_QWORD *)(v345 + 24) == 0;
            }
            else
            {
              v407 = *(_DWORD *)(v345 + 32);
              v346 = sub_16A57B0(v345 + 24);
              v93 = v407;
              v347 = v407 == v346;
            }
            if ( !v347 )
              goto LABEL_110;
          }
          ++v344;
        }
      }
    }
  }
  if ( (unsigned __int8)sub_1790C20((_BYTE *)v12, v45, v93, v46) )
  {
    v241 = a1->m128i_i64[1];
    v242 = sub_1649960(v11);
    v455.m128i_i16[0] = 1283;
    v451 = (__int64 *)v242;
    v452 = v243;
    v454.m128_u64[0] = (unsigned __int64)"not.";
    v454.m128_u64[1] = (unsigned __int64)&v451;
    v244 = sub_171CA90(v241, v11, (__int64 *)&v454, *(double *)v13.m128_u64, a4, *(double *)v14.m128i_i64);
    v455.m128i_i16[0] = 257;
    v245 = (__int64)v244;
    v428 = (__int64)sub_1648A60(56, 1u);
    if ( v428 )
      sub_15FC690(v428, v245, (__int64)v411, (__int64)&v454, 0);
    return v428;
  }
LABEL_110:
  if ( *(_BYTE *)(v10 + 16) > 0x10u )
    goto LABEL_49;
LABEL_111:
  if ( sub_1593BB0(v10, v45, v93, v46) )
    goto LABEL_112;
  if ( *(_BYTE *)(v10 + 16) == 13 )
  {
    v293 = *(_DWORD *)(v10 + 32);
    if ( v293 <= 0x40 )
      v294 = *(_QWORD *)(v10 + 24) == 0;
    else
      v294 = v293 == (unsigned int)sub_16A57B0(v10 + 24);
    goto LABEL_447;
  }
  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
    goto LABEL_49;
  v312 = sub_15A1020((_BYTE *)v10, v45, v94, v46);
  if ( v312 && *(_BYTE *)(v312 + 16) == 13 )
  {
    v313 = *(_DWORD *)(v312 + 32);
    if ( v313 <= 0x40 )
      v294 = *(_QWORD *)(v312 + 24) == 0;
    else
      v294 = v313 == (unsigned int)sub_16A57B0(v312 + 24);
LABEL_447:
    if ( !v294 )
      goto LABEL_49;
    goto LABEL_112;
  }
  v316 = 0;
  v422 = *(_DWORD *)(*(_QWORD *)v10 + 32LL);
  while ( v422 != v316 )
  {
    v45 = v316;
    v317 = sub_15A0A60(v10, v316);
    if ( !v317 )
      goto LABEL_49;
    v94 = *(unsigned __int8 *)(v317 + 16);
    if ( (_BYTE)v94 != 9 )
    {
      if ( (_BYTE)v94 != 13 )
        goto LABEL_49;
      v94 = *(unsigned int *)(v317 + 32);
      if ( (unsigned int)v94 <= 0x40 )
      {
        v319 = *(_QWORD *)(v317 + 24) == 0;
      }
      else
      {
        v405 = *(_DWORD *)(v317 + 32);
        v318 = sub_16A57B0(v317 + 24);
        v94 = v405;
        v319 = v405 == v318;
      }
      if ( !v319 )
        goto LABEL_49;
    }
    ++v316;
  }
LABEL_112:
  if ( (unsigned __int8)sub_1790D60((_BYTE *)v12, v45, v94, v46) )
  {
    v95 = a1->m128i_i64[1];
    v96 = sub_1649960(v11);
    v455.m128i_i16[0] = 1283;
    v451 = (__int64 *)v96;
    v452 = v97;
    v454.m128_u64[0] = (unsigned __int64)"not.";
    v454.m128_u64[1] = (unsigned __int64)&v451;
    v98 = sub_171CA90(v95, v11, (__int64 *)&v454, *(double *)v13.m128_u64, a4, *(double *)v14.m128i_i64);
    v455.m128i_i16[0] = 257;
    v99 = (__int64)v98;
    v428 = (__int64)sub_1648A60(56, 1u);
    if ( v428 )
      sub_15FC810(v428, v99, (__int64)v411, (__int64)&v454, 0);
    return v428;
  }
LABEL_49:
  v55 = *(_BYTE *)(v11 + 16);
  if ( v55 != 76 )
  {
LABEL_50:
    if ( v55 == 75 )
    {
      v45 = a2;
      v86 = sub_1796730(
              a1->m128i_i64,
              (__int64 *)a2,
              v11,
              v46,
              v13,
              a4,
              *(double *)v14.m128i_i64,
              a6,
              v47,
              v48,
              a9,
              a10);
      if ( v86 )
        return v86;
    }
LABEL_51:
    v398 = *(_QWORD *)(a2 - 48);
    v56 = *(_BYTE *)(v398 + 16);
    if ( v56 <= 0x17u )
      goto LABEL_89;
    v57 = *(_QWORD *)(a2 - 24);
    v58 = *(_BYTE *)(v57 + 16);
    v415 = v57;
    if ( v58 <= 0x17u )
      goto LABEL_89;
    v59 = *(_QWORD *)(v398 + 8);
    if ( !v59 )
      goto LABEL_89;
    if ( *(_QWORD *)(v59 + 8) )
      goto LABEL_89;
    v60 = *(_QWORD *)(v57 + 8);
    if ( !v60 || *(_QWORD *)(v60 + 8) )
      goto LABEL_89;
    if ( v56 == 37 )
    {
      if ( v58 != 35 )
        goto LABEL_89;
    }
    else
    {
      if ( v56 != 38 )
      {
        if ( v58 == 37 )
        {
          v384 = *(_QWORD *)(a2 - 48);
          if ( v56 != 35 )
            goto LABEL_89;
        }
        else
        {
          if ( v58 != 38 || v56 != 36 )
            goto LABEL_89;
          v384 = *(_QWORD *)(a2 - 48);
        }
LABEL_63:
        v61 = *(_BYTE *)(v415 + 23) & 0x40;
        if ( v61 )
          v62 = *(__int64 **)(v415 - 8);
        else
          v62 = (__int64 *)(v415 - 24LL * (*(_DWORD *)(v415 + 20) & 0xFFFFFFF));
        v63 = *v62;
        if ( (*(_BYTE *)(v384 + 23) & 0x40) != 0 )
        {
          v64 = *(__int64 **)(v384 - 8);
        }
        else
        {
          v45 = 24LL * (*(_DWORD *)(v384 + 20) & 0xFFFFFFF);
          v64 = (__int64 *)(v384 - v45);
        }
        v65 = *v64;
        v66 = v64[3];
        v381 = v65;
        if ( v63 == v65 )
        {
          v381 = v66;
        }
        else if ( v63 != v66 )
        {
          goto LABEL_89;
        }
        if ( !v381 )
        {
LABEL_89:
          v83 = *(_BYTE *)(v10 + 16);
          if ( v83 > 0x17u )
          {
            v84 = *(_BYTE *)(v12 + 16);
            if ( v84 > 0x17u && v84 == v83 )
            {
              v86 = sub_1792610((__int64)a1, a2, v10, v12);
              if ( v86 )
                return v86;
            }
          }
          v85 = (_QWORD *)a2;
          v86 = sub_1791A10((__int64)a1, (__int64 ***)a2);
          if ( v86 )
            return v86;
          v118 = *((unsigned __int8 *)v411 + 8);
          if ( (_BYTE)v118 == 16 )
            LOBYTE(v118) = *(_BYTE *)(*v411[2] + 8);
          if ( (_BYTE)v118 != 11 && (unsigned __int8)(v118 - 1) > 5u )
            goto LABEL_175;
          v86 = sub_17921E0((__int64)a1, a2, v10, v12);
          if ( v86 )
            return v86;
          v85 = &v433;
          v119 = sub_14B2890(a2, (__int64 *)&v433, (__int64 *)&v434, (unsigned int *)&v432, 0);
          v417 = v119;
          v439 = v119;
          v440 = v120;
          if ( !(_DWORD)v119 )
          {
LABEL_175:
            if ( *(_BYTE *)(*(_QWORD *)(a2 - 72) + 16LL) == 77 )
            {
              v85 = (_QWORD *)a2;
              if ( sub_17903C0(v10, a2) && sub_17903C0(v12, a2) )
              {
                v86 = sub_17127D0(a1->m128i_i64, a2, v200, v13, a4, *(double *)v14.m128i_i64, a6, v201, v202, a9, a10);
                if ( v86 )
                  return v86;
              }
            }
            v139 = *(unsigned __int8 *)(v10 + 16);
            if ( (_BYTE)v139 != 79 )
              goto LABEL_177;
            v197 = *(_QWORD **)(v10 - 72);
            if ( *(_QWORD *)v11 != *v197 )
              goto LABEL_177;
            if ( (_QWORD *)v11 == v197 )
            {
              v340 = *(_QWORD *)(v10 - 48);
              if ( v340 != *(_QWORD *)(a2 - 48) )
              {
                sub_1593B40((_QWORD *)(a2 - 48), v340);
                return v428;
              }
              return 0;
            }
            if ( v12 == *(_QWORD *)(v10 - 24) )
            {
              v141 = *(_QWORD *)(v10 + 8);
              if ( v141 )
              {
                if ( !*(_QWORD *)(v141 + 8) )
                {
                  v455.m128i_i16[0] = 257;
                  v339 = sub_1729500(
                           a1->m128i_i64[1],
                           (unsigned __int8 *)v11,
                           *(_QWORD *)(v10 - 72),
                           (__int64 *)&v454,
                           *(double *)v13.m128_u64,
                           a4,
                           *(double *)v14.m128i_i64);
                  sub_1593B40((_QWORD *)(a2 - 72), (__int64)v339);
                  sub_1593B40((_QWORD *)(a2 - 48), *(_QWORD *)(v10 - 48));
                  return v428;
                }
                v140 = *(unsigned __int8 *)(v12 + 16);
                if ( (_BYTE)v140 != 79 )
                  goto LABEL_181;
                v85 = *(_QWORD **)(v12 - 72);
                if ( *(_QWORD *)v11 != *v85 )
                {
LABEL_179:
                  if ( !*(_QWORD *)(v141 + 8) && (unsigned __int8)v139 > 0x17u )
                  {
                    v141 = (unsigned int)(v139 - 35);
                    if ( (unsigned int)v141 <= 0x11 )
                    {
                      v141 = (unsigned int)(v139 - 24);
                      if ( (unsigned int)v141 > 0x12 )
                      {
                        if ( (unsigned int)(v139 - 44) <= 1 )
                          goto LABEL_181;
                      }
                      else if ( (unsigned int)v141 > 0x10 )
                      {
                        goto LABEL_181;
                      }
                      v193 = *(_QWORD *)(v10 - 48);
                      if ( *(_BYTE *)(v193 + 16) == 79 && v11 == *(_QWORD *)(v193 - 72) )
                      {
                        v195 = *(_QWORD *)(v193 - 48);
                        v196 = (_QWORD *)(v10 - 48);
                        goto LABEL_279;
                      }
                      v194 = *(_QWORD *)(v10 - 24);
                      if ( *(_BYTE *)(v194 + 16) == 79 && v11 == *(_QWORD *)(v194 - 72) )
                      {
                        v195 = *(_QWORD *)(v194 - 48);
                        v196 = (_QWORD *)(v10 - 24);
LABEL_279:
                        sub_1593B40(v196, v195);
                        sub_170B990(a1->m128i_i64[0], v10);
                        return v428;
                      }
                    }
                  }
LABEL_181:
                  v142 = *(_QWORD *)(v12 + 8);
                  if ( v142 )
                  {
                    if ( !*(_QWORD *)(v142 + 8) && (unsigned __int8)v140 > 0x17u )
                    {
                      v142 = (unsigned int)(v140 - 35);
                      if ( (unsigned int)v142 <= 0x11 )
                      {
                        v142 = (unsigned int)(v140 - 24);
                        if ( (unsigned int)v142 > 0x12 )
                        {
                          if ( (unsigned int)(v140 - 44) <= 1 )
                            goto LABEL_184;
                        }
                        else if ( (unsigned int)v142 > 0x10 )
                        {
                          goto LABEL_184;
                        }
                        v203 = *(_QWORD *)(v12 - 48);
                        if ( *(_BYTE *)(v203 + 16) == 79 && v11 == *(_QWORD *)(v203 - 72) )
                        {
                          v205 = *(_QWORD *)(v203 - 24);
                          v206 = (_QWORD *)(v12 - 48);
                          goto LABEL_307;
                        }
                        v204 = *(_QWORD *)(v12 - 24);
                        if ( *(_BYTE *)(v204 + 16) == 79 && v11 == *(_QWORD *)(v204 - 72) )
                        {
                          v205 = *(_QWORD *)(v204 - 24);
                          v206 = (_QWORD *)(v12 - 24);
LABEL_307:
                          sub_1593B40(v206, v205);
                          sub_170B990(a1->m128i_i64[0], v12);
                          return v428;
                        }
                      }
                    }
                  }
LABEL_184:
                  if ( sub_15FB730(v11, (__int64)v85, v142, v141) )
                  {
                    v207 = sub_15FB7C0(v11, (__int64)v85, v143, v144);
                    sub_1593B40((_QWORD *)(a2 - 72), v207);
                    sub_1593B40((_QWORD *)(a2 - 48), v12);
                    sub_1593B40((_QWORD *)(a2 - 24), v10);
                    return v428;
                  }
                  if ( *((_BYTE *)v411 + 8) != 16 )
                  {
LABEL_238:
                    v170 = *(_QWORD *)(a2 + 40);
                    v171 = sub_157F0B0(v170);
                    if ( v171 )
                    {
                      v172 = sub_157EBA0(v171);
                      if ( v172 )
                      {
                        if ( *(_BYTE *)(v172 + 16) == 26 && (*(_DWORD *)(v172 + 20) & 0xFFFFFFF) == 3 )
                        {
                          v322 = *(_QWORD *)(v172 - 24);
                          v323 = *(_QWORD *)(v172 - 48);
                          if ( v323 != v322 && (v170 == v323 || v170 == v322) )
                          {
                            sub_14BCF40(
                              (bool *)&v454,
                              *(_QWORD *)(v172 - 72),
                              *(_QWORD *)(a2 - 72),
                              a1[166].m128i_i64[1],
                              v170 == v322,
                              0);
                            if ( v454.m128_i8[1] )
                            {
                              if ( !v454.m128_i8[0] )
                                v10 = v12;
                              return sub_170E100(
                                       a1->m128i_i64,
                                       a2,
                                       v10,
                                       v13,
                                       a4,
                                       *(double *)v14.m128i_i64,
                                       a6,
                                       v324,
                                       v325,
                                       a9,
                                       a10);
                            }
                          }
                        }
                      }
                    }
                    if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
                      goto LABEL_246;
                    v173 = a1[165].m128i_i64[0];
                    if ( !*(_BYTE *)(v173 + 184) )
                      sub_14CDF70(a1[165].m128i_i64[0]);
                    if ( !*(_DWORD *)(v173 + 16) )
                    {
LABEL_246:
                      v174 = a1->m128i_i64[1];
                      v175 = *(_QWORD *)(a2 - 72);
                      v176 = *(_QWORD *)(a2 - 48);
                      v177 = *(_QWORD *)(a2 - 24);
                      if ( (unsigned __int8)(*(_BYTE *)(v175 + 16) - 75) <= 1u )
                      {
                        v208 = *(_QWORD *)(v175 - 48);
                        if ( v208 )
                        {
                          v209 = *(_QWORD *)(v175 - 24);
                          if ( v209 )
                          {
                            if ( v176 != v208 && v176 != v209 && v177 != v208 && v177 != v209 )
                            {
                              v210 = *(unsigned __int8 *)(v208 + 16);
                              if ( (unsigned __int8)v210 > 0x17u )
                              {
                                v211 = v210 - 24;
                              }
                              else
                              {
                                if ( (_BYTE)v210 != 5 )
                                  goto LABEL_247;
                                v211 = *(unsigned __int16 *)(v208 + 18);
                              }
                              if ( v211 == 47 )
                              {
                                v212 = (*(_BYTE *)(v208 + 23) & 0x40) != 0
                                     ? *(__int64 **)(v208 - 8)
                                     : (__int64 *)(v208 - 24LL * (*(_DWORD *)(v208 + 20) & 0xFFFFFFF));
                                v213 = *v212;
                                if ( *v212 )
                                {
                                  v214 = *(unsigned __int8 *)(v209 + 16);
                                  if ( (unsigned __int8)v214 > 0x17u )
                                  {
                                    v215 = v214 - 24;
                                  }
                                  else
                                  {
                                    if ( (_BYTE)v214 != 5 )
                                      goto LABEL_247;
                                    v215 = *(unsigned __int16 *)(v209 + 18);
                                  }
                                  if ( v215 == 47 )
                                  {
                                    v216 = (*(_BYTE *)(v209 + 23) & 0x40) != 0
                                         ? *(__int64 **)(v209 - 8)
                                         : (__int64 *)(v209 - 24LL * (*(_DWORD *)(v209 + 20) & 0xFFFFFFF));
                                    v217 = *v216;
                                    if ( v217 )
                                    {
                                      v218 = *(unsigned __int8 *)(v176 + 16);
                                      if ( (unsigned __int8)v218 > 0x17u )
                                      {
                                        v219 = v218 - 24;
                                      }
                                      else
                                      {
                                        if ( (_BYTE)v218 != 5 )
                                          goto LABEL_247;
                                        v219 = *(unsigned __int16 *)(v176 + 18);
                                      }
                                      if ( v219 == 47 )
                                      {
                                        v220 = (*(_BYTE *)(v176 + 23) & 0x40) != 0
                                             ? *(__int64 **)(v176 - 8)
                                             : (__int64 *)(v176 - 24LL * (*(_DWORD *)(v176 + 20) & 0xFFFFFFF));
                                        v221 = *v220;
                                        if ( *v220 )
                                        {
                                          v222 = *(unsigned __int8 *)(v177 + 16);
                                          if ( (unsigned __int8)v222 > 0x17u )
                                          {
                                            v223 = v222 - 24;
                                          }
                                          else
                                          {
                                            if ( (_BYTE)v222 != 5 )
                                              goto LABEL_247;
                                            v223 = *(unsigned __int16 *)(v177 + 18);
                                          }
                                          if ( v223 == 47 )
                                          {
                                            v224 = (*(_BYTE *)(v177 + 23) & 0x40) != 0
                                                 ? *(__int64 **)(v177 - 8)
                                                 : (__int64 *)(v177 - 24LL * (*(_DWORD *)(v177 + 20) & 0xFFFFFFF));
                                            v225 = *v224;
                                            if ( v225 )
                                            {
                                              if ( v213 == v221 && v217 == v225 )
                                              {
                                                v320 = *(_QWORD *)(v175 - 24);
                                                v321 = *(_QWORD *)(v175 - 48);
                                                v455.m128i_i16[0] = 257;
                                                v227 = sub_1707C10(v174, v175, v321, v320, (__int64 *)&v454, a2);
LABEL_354:
                                                v455.m128i_i16[0] = 257;
                                                v86 = sub_15FE030((__int64)v227, *(_QWORD *)a2, (__int64)&v454, 0);
                                                if ( !v86 )
                                                  goto LABEL_247;
                                                return v86;
                                              }
                                              if ( v217 == v221 && v213 == v225 )
                                              {
                                                v226 = *(_QWORD *)(v175 - 48);
                                                v455.m128i_i16[0] = 257;
                                                v227 = sub_1707C10(v174, v175, v209, v226, (__int64 *)&v454, a2);
                                                goto LABEL_354;
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
LABEL_247:
                      v86 = sub_17917B0(a2);
                      if ( v86 )
                        return v86;
                      v86 = sub_1791640(a2);
                      if ( v86 )
                        return v86;
                      v178 = sub_1790400(v10, (__int64 *)&v437, (__int64 *)&v438, (__int64 *)&v439);
                      if ( v178 )
                      {
                        v179 = *(_QWORD *)(v178 + 8);
                        if ( v179 )
                        {
                          if ( !*(_QWORD *)(v179 + 8) && sub_1642F90(*v437, 1) && sub_1642F90(*(_QWORD *)v11, 1) )
                          {
                            if ( v438 == (__int64 *)v12 )
                            {
                              v455.m128i_i16[0] = 257;
                              v453 = 257;
                              v326 = a1->m128i_i64[1];
                              v327 = sub_171CA90(
                                       v326,
                                       (__int64)v437,
                                       (__int64 *)&v451,
                                       *(double *)v13.m128_u64,
                                       a4,
                                       *(double *)v14.m128i_i64);
                              v328 = v11;
                              v329 = v326;
                              v330 = v327;
                              goto LABEL_517;
                            }
                            if ( v439 == v12 )
                            {
                              v184 = (unsigned __int8 *)v437;
                              v186 = v11;
                              v455.m128i_i16[0] = 257;
                              v185 = a1->m128i_i64[1];
                              goto LABEL_260;
                            }
                          }
                        }
                      }
                      v180 = sub_1790400(v12, (__int64 *)&v437, (__int64 *)&v438, (__int64 *)&v439);
                      if ( !v180 )
                        return 0;
                      v181 = *(_QWORD *)(v180 + 8);
                      if ( !v181 || *(_QWORD *)(v181 + 8) || !sub_1642F90(*v437, 1) || !sub_1642F90(*(_QWORD *)v11, 1) )
                        return 0;
                      if ( v438 != (__int64 *)v10 )
                      {
                        if ( v439 == v10 )
                        {
                          v455.m128i_i16[0] = 257;
                          v453 = 257;
                          v182 = a1->m128i_i64[1];
                          v183 = sub_171CA90(
                                   v182,
                                   v11,
                                   (__int64 *)&v451,
                                   *(double *)v13.m128_u64,
                                   a4,
                                   *(double *)v14.m128i_i64);
                          v184 = (unsigned __int8 *)v437;
                          v185 = v182;
                          v186 = (__int64)v183;
LABEL_260:
                          v187 = sub_1729500(
                                   v185,
                                   v184,
                                   v186,
                                   (__int64 *)&v454,
                                   *(double *)v13.m128_u64,
                                   a4,
                                   *(double *)v14.m128i_i64);
                          sub_1593B40((_QWORD *)(a2 - 72), (__int64)v187);
                          sub_1593B40((_QWORD *)(a2 - 48), (__int64)v438);
                          sub_1593B40((_QWORD *)(a2 - 24), v439);
                          return v428;
                        }
                        return 0;
                      }
                      v455.m128i_i16[0] = 257;
                      v453 = 257;
                      v332 = a1->m128i_i64[1];
                      v333 = sub_171CA90(
                               v332,
                               v11,
                               (__int64 *)&v451,
                               *(double *)v13.m128_u64,
                               a4,
                               *(double *)v14.m128i_i64);
                      v334 = a1->m128i_i64[1];
                      v450 = 257;
                      v335 = v333;
                      v336 = sub_171CA90(
                               v334,
                               (__int64)v437,
                               (__int64 *)&v448,
                               *(double *)v13.m128_u64,
                               a4,
                               *(double *)v14.m128i_i64);
                      v328 = (__int64)v335;
                      v329 = v332;
                      v330 = v336;
LABEL_517:
                      v331 = sub_1729500(
                               v329,
                               v330,
                               v328,
                               (__int64 *)&v454,
                               *(double *)v13.m128_u64,
                               a4,
                               *(double *)v14.m128i_i64);
                      sub_1593B40((_QWORD *)(a2 - 72), (__int64)v331);
                      sub_1593B40((_QWORD *)(a2 - 48), v439);
                      sub_1593B40((_QWORD *)(a2 - 24), (__int64)v438);
                      return v428;
                    }
                    v454.m128_u64[0] = 0;
                    v287 = a1[166].m128i_i64[1];
                    v374 = a1[166].m128i_i64[0];
                    v288 = a1[165].m128i_i64[0];
                    v454.m128_u64[1] = 1;
                    v455.m128i_i64[0] = 0;
                    v455.m128i_i64[1] = 1;
                    sub_14BB090(v11, (__int64)&v454, v287, 0, v288, a2, v374, 0);
                    if ( sub_1455000((__int64)&v455) )
                    {
                      v428 = sub_170E100(
                               a1->m128i_i64,
                               a2,
                               v10,
                               v13,
                               a4,
                               *(double *)v14.m128i_i64,
                               a6,
                               v289,
                               v290,
                               a9,
                               a10);
                    }
                    else
                    {
                      if ( !sub_1455000((__int64)&v454) )
                      {
                        sub_135E100(v455.m128i_i64);
                        sub_135E100((__int64 *)&v454);
                        goto LABEL_246;
                      }
                      v428 = sub_170E100(
                               a1->m128i_i64,
                               a2,
                               v12,
                               v13,
                               a4,
                               *(double *)v14.m128i_i64,
                               a6,
                               v291,
                               v292,
                               a9,
                               a10);
                    }
                    sub_135E100(v455.m128i_i64);
                    sub_135E100((__int64 *)&v454);
                    return v428;
                  }
                  v145 = v411[4];
                  v449 = (unsigned int)v145;
                  if ( (unsigned int)v145 > 0x40 )
                  {
                    v420 = (int)v145;
                    sub_16A4EF0((__int64)&v448, 0, 0);
                    LODWORD(v452) = v420;
                    sub_16A4EF0((__int64)&v451, -1, 1);
                    v454.m128_i32[2] = v452;
                    if ( (unsigned int)v452 > 0x40 )
                    {
                      sub_16A4FD0((__int64)&v454, (const void **)&v451);
LABEL_189:
                      v146 = sub_17A4D70(a1, a2, &v454, &v448, 0);
                      v147 = v146;
                      if ( v454.m128_i32[2] > 0x40u && v454.m128_u64[0] )
                      {
                        v418 = v146;
                        j_j___libc_free_0_0(v454.m128_u64[0]);
                        v147 = v418;
                      }
                      if ( v147 )
                      {
                        if ( a2 != v147 )
                        {
                          v148 = *(_QWORD *)(a2 + 8);
                          if ( v148 )
                          {
                            v149 = v147;
                            v150 = a1->m128i_i64[0];
                            do
                            {
                              v151 = sub_1648700(v148);
                              sub_170B990(v150, (__int64)v151);
                              v148 = *(_QWORD *)(v148 + 8);
                            }
                            while ( v148 );
                            sub_164D160(a2, v149, v13, a4, *(double *)v14.m128i_i64, a6, v152, v153, a9, a10);
                          }
                          else
                          {
                            v428 = 0;
                          }
                        }
                        if ( (unsigned int)v452 > 0x40 && v451 )
                          j_j___libc_free_0_0(v451);
                        if ( v449 > 0x40 && v448 )
                          j_j___libc_free_0_0(v448);
                        return v428;
                      }
                      sub_135E100((__int64 *)&v451);
                      sub_135E100((__int64 *)&v448);
                      goto LABEL_238;
                    }
                  }
                  else
                  {
                    v448 = 0;
                    LODWORD(v452) = (_DWORD)v145;
                    v454.m128_i32[2] = (int)v145;
                    v451 = (__int64 *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v145);
                  }
                  v454.m128_u64[0] = (unsigned __int64)v451;
                  goto LABEL_189;
                }
              }
              else
              {
                v140 = *(unsigned __int8 *)(v12 + 16);
                if ( (_BYTE)v140 != 79 )
                  goto LABEL_181;
                v85 = *(_QWORD **)(v12 - 72);
                v141 = *(_QWORD *)v11;
                if ( *v85 != *(_QWORD *)v11 )
                  goto LABEL_181;
              }
            }
            else
            {
LABEL_177:
              v140 = *(unsigned __int8 *)(v12 + 16);
              if ( (_BYTE)v140 != 79 )
                goto LABEL_178;
              v85 = *(_QWORD **)(v12 - 72);
              if ( *(_QWORD *)v11 != *v85 )
                goto LABEL_178;
            }
            if ( (_QWORD *)v11 == v85 )
            {
              v338 = *(_QWORD *)(v12 - 24);
              if ( v338 != *(_QWORD *)(a2 - 24) )
              {
                sub_1593B40((_QWORD *)(a2 - 24), v338);
                return v428;
              }
              return 0;
            }
            v140 = 79;
            if ( v10 == *(_QWORD *)(v12 - 48) )
            {
              v198 = *(_QWORD *)(v12 + 8);
              if ( v198 )
              {
                if ( !*(_QWORD *)(v198 + 8) )
                {
                  v455.m128i_i16[0] = 257;
                  v199 = sub_172AC10(
                           a1->m128i_i64[1],
                           v11,
                           *(_QWORD *)(v12 - 72),
                           (__int64 *)&v454,
                           *(double *)v13.m128_u64,
                           a4,
                           *(double *)v14.m128i_i64);
                  sub_1593B40((_QWORD *)(a2 - 72), (__int64)v199);
                  sub_1593B40((_QWORD *)(a2 - 24), *(_QWORD *)(v12 - 24));
                  return v428;
                }
              }
            }
LABEL_178:
            v141 = *(_QWORD *)(v10 + 8);
            if ( !v141 )
              goto LABEL_181;
            goto LABEL_179;
          }
          if ( (_DWORD)v119 != 7 && (_DWORD)v119 != 8 )
          {
            v121 = *(__int64 ***)v433;
            v401 = v433;
            if ( v411 != *(__int64 ***)v433
              || sub_15F4D40((__int64)v411)
              && ((v255 = *(_BYTE **)(v11 - 48), v255 != v401) && v434 != v255
               || (v256 = *(_BYTE **)(v11 - 24), v256 != v401) && v434 != v256) )
            {
              v122 = sub_14AEAE0(v417, v440);
              v123 = v122;
              if ( (unsigned int)(v122 - 32) <= 9 )
              {
                v337 = a1->m128i_i64[1];
                v455.m128i_i16[0] = 257;
                v129 = sub_17203D0(v337, v122, (__int64)v433, (__int64)v434, (__int64 *)&v454);
              }
              else
              {
                v124 = a1->m128i_i64[1];
                v125 = *(_DWORD *)(v124 + 40);
                v126 = *(_QWORD *)(v124 + 32);
                v127 = *(_BYTE *)(*(_QWORD *)(a2 - 72) + 17LL) >> 1;
                if ( v127 == 127 )
                  v127 = -1;
                *(_DWORD *)(v124 + 40) = v127;
                v455.m128i_i16[0] = 257;
                v128 = sub_17290F0(a1->m128i_i64[1], v123, (__int64)v433, (__int64)v434, (__int64 *)&v454, 0);
                *(_DWORD *)(v124 + 40) = v125;
                *(_QWORD *)(v124 + 32) = v126;
                v129 = v128;
              }
              v130 = a1->m128i_i64[1];
              v131 = sub_1649960(a2);
              v452 = v132;
              v451 = (__int64 *)v131;
              v455.m128i_i16[0] = 261;
              v454.m128_u64[0] = (unsigned __int64)&v451;
              v133 = sub_1707C10(v130, (__int64)v129, (__int64)v433, (__int64)v434, (__int64 *)&v454, a2);
              if ( v411 == v121 )
                return sub_170E100(
                         a1->m128i_i64,
                         a2,
                         (__int64)v133,
                         v13,
                         a4,
                         *(double *)v14.m128i_i64,
                         a6,
                         v134,
                         v135,
                         a9,
                         a10);
              v455.m128i_i16[0] = 257;
              v136 = sub_1708970(a1->m128i_i64[1], v432, (__int64)v133, v411, (__int64 *)&v454);
              return sub_170E100(
                       a1->m128i_i64,
                       a2,
                       (__int64)v136,
                       v13,
                       a4,
                       *(double *)v14.m128i_i64,
                       a6,
                       v137,
                       v138,
                       a9,
                       a10);
            }
            v451 = &v435;
            if ( sub_171DA10(&v451, (unsigned __int64)v401, v253, v254) )
            {
              v454.m128_u64[0] = (unsigned __int64)&v436;
              if ( sub_171DA10(&v454, (unsigned __int64)v434, v257, v258)
                && ((unsigned int)sub_1648EF0((__int64)v433) <= 2 || (unsigned int)sub_1648EF0((__int64)v434) <= 2) )
              {
                v352 = sub_14AEB90(v417);
                v455.m128i_i16[0] = 257;
                v353 = sub_17203D0(a1->m128i_i64[1], v352, v435, (__int64)v436, (__int64 *)&v454);
                v354 = a1->m128i_i64[1];
                v455.m128i_i16[0] = 257;
                v355 = sub_1707C10(v354, (__int64)v353, v435, (__int64)v436, (__int64 *)&v454, 0);
                v455.m128i_i16[0] = 257;
                return sub_15FB630(v355, (__int64)&v454, 0);
              }
            }
            v259 = v433;
            v260 = *(_QWORD *)v433;
            if ( *(_BYTE *)(*(_QWORD *)v433 + 8LL) == 16 )
              v260 = **(_QWORD **)(v260 + 16);
            if ( *(_BYTE *)(v260 + 8) != 11 )
              goto LABEL_421;
            v261 = v434;
            v386 = v433;
            v403 = a1->m128i_i64[1];
            v262 = sub_14B2890((__int64)v433, (__int64 *)&v437, (__int64 *)&v438, 0, 0);
            v449 = v263;
            v393 = v262;
            v448 = v262;
            v264 = (__int64 *)sub_14B2890((__int64)v261, (__int64 *)&v442, (__int64 *)&v445, 0, 0);
            LODWORD(v452) = v265;
            v451 = v264;
            if ( v417 == v393 && v417 == (_DWORD)v451 )
            {
              v266 = sub_1648D00((__int64)v386, 3);
              v267 = v386;
              if ( !v266 )
              {
                v268 = sub_1648D00((__int64)v261, 3);
                v267 = v386;
                if ( v268 )
                {
                  v269 = v438;
                  if ( v445 != v437 && v437 != v442 )
                  {
                    if ( v442 != v438 && v445 != v438 )
                      goto LABEL_420;
                    v269 = v437;
                  }
LABEL_584:
                  if ( v261 && v269 )
                  {
                    v394 = (__int64)v269;
                    v356 = sub_14AEAE0(v417, 0);
                    v455.m128i_i16[0] = 257;
                    if ( v261[16] > 0x10u || *(_BYTE *)(v394 + 16) > 0x10u )
                    {
                      v373 = sub_1790840(v403, v356, (__int64)v261, v394, (__int64 *)&v454);
                      v361 = v394;
                      v360 = (__int64)v373;
                    }
                    else
                    {
                      v357 = sub_15A37B0(v356, v261, (_QWORD *)v394, 0);
                      v358 = v403;
                      v409 = v357;
                      v359 = sub_14DBA30(v357, *(_QWORD *)(v358 + 96), 0);
                      v360 = v409;
                      v361 = v394;
                      if ( v359 )
                        v360 = v359;
                    }
                    v395 = v360;
                    v410 = v361;
                    v455.m128i_i16[0] = 257;
                    v362 = sub_1648A60(56, 3u);
                    if ( v362 )
                    {
                      v363 = v362 - 9;
                      v431 = (__int64)v362;
                      sub_15F1EA0((__int64)v362, *(_QWORD *)v261, 55, (__int64)(v362 - 9), 3, 0);
                      if ( *(_QWORD *)(v431 - 72) )
                      {
                        v364 = *(_QWORD *)(v431 - 64);
                        v365 = *(_QWORD *)(v431 - 56) & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v365 = v364;
                        if ( v364 )
                          *(_QWORD *)(v364 + 16) = v365 | *(_QWORD *)(v364 + 16) & 3LL;
                      }
                      *(_QWORD *)(v431 - 72) = v395;
                      if ( v395 )
                      {
                        v366 = *(_QWORD *)(v395 + 8);
                        *(_QWORD *)(v431 - 64) = v366;
                        if ( v366 )
                          *(_QWORD *)(v366 + 16) = (v431 - 64) | *(_QWORD *)(v366 + 16) & 3LL;
                        *(_QWORD *)(v431 - 56) = (v395 + 8) | *(_QWORD *)(v431 - 56) & 3LL;
                        *(_QWORD *)(v395 + 8) = v363;
                      }
                      if ( *(_QWORD *)(v431 - 48) )
                      {
                        v367 = *(_QWORD *)(v431 - 40);
                        v368 = *(_QWORD *)(v431 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v368 = v367;
                        if ( v367 )
                          *(_QWORD *)(v367 + 16) = v368 | *(_QWORD *)(v367 + 16) & 3LL;
                      }
                      *(_QWORD *)(v431 - 48) = v261;
                      v369 = *((_QWORD *)v261 + 1);
                      *(_QWORD *)(v431 - 40) = v369;
                      if ( v369 )
                        *(_QWORD *)(v369 + 16) = (v431 - 40) | *(_QWORD *)(v369 + 16) & 3LL;
                      *(_QWORD *)(v431 - 32) = *(_QWORD *)(v431 - 32) & 3LL | (unsigned __int64)(v261 + 8);
                      *((_QWORD *)v261 + 1) = v431 - 48;
                      if ( *(_QWORD *)(v431 - 24) )
                      {
                        v370 = *(_QWORD *)(v431 - 16);
                        v371 = *(_QWORD *)(v431 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v371 = v370;
                        if ( v370 )
                          *(_QWORD *)(v370 + 16) = v371 | *(_QWORD *)(v370 + 16) & 3LL;
                      }
                      *(_QWORD *)(v431 - 24) = v410;
                      v372 = *(_QWORD *)(v410 + 8);
                      *(_QWORD *)(v431 - 16) = v372;
                      if ( v372 )
                        *(_QWORD *)(v372 + 16) = (v431 - 16) | *(_QWORD *)(v372 + 16) & 3LL;
                      *(_QWORD *)(v431 - 8) = *(_QWORD *)(v431 - 8) & 3LL | (v410 + 8);
                      *(_QWORD *)(v410 + 8) = v431 - 24;
                      sub_164B780(v431, (__int64 *)&v454);
                      return v431;
                    }
                  }
                  goto LABEL_420;
                }
              }
              v396 = v267;
              if ( !(unsigned __int8)sub_1648D00((__int64)v261, 3) )
              {
                v269 = v442;
                if ( v445 == v437 || v445 == v438 )
                {
                  v261 = v396;
                }
                else
                {
                  if ( v437 != v442 && v438 != v442 )
                    goto LABEL_420;
                  v269 = v445;
                  v261 = v396;
                }
                goto LABEL_584;
              }
            }
          }
LABEL_420:
          v259 = v433;
LABEL_421:
          v270 = sub_14B2890((__int64)v259, (__int64 *)&v451, (__int64 *)&v454, 0, 0);
          v442 = (__int64 *)v270;
          v443 = v273;
          if ( (_DWORD)v270 )
          {
            if ( *(_QWORD *)a2 == *(_QWORD *)v433 )
            {
              v86 = sub_17960B0(
                      a1->m128i_i64,
                      (__int64)v433,
                      v270,
                      (__int64)v451,
                      v454.m128_u64[0],
                      a2,
                      v13,
                      a4,
                      *(double *)v14.m128i_i64,
                      a6,
                      v271,
                      v272,
                      a9,
                      a10,
                      v417,
                      v434);
              if ( v86 )
                return v86;
            }
          }
          v85 = &v451;
          v274 = sub_14B2890((__int64)v434, (__int64 *)&v451, (__int64 *)&v454, 0, 0);
          v445 = (__int64 *)v274;
          v446 = v277;
          if ( (_DWORD)v274 )
          {
            v85 = v434;
            if ( *(_QWORD *)a2 == *(_QWORD *)v434 )
            {
              v86 = sub_17960B0(
                      a1->m128i_i64,
                      (__int64)v434,
                      v274,
                      (__int64)v451,
                      v454.m128_u64[0],
                      a2,
                      v13,
                      a4,
                      *(double *)v14.m128i_i64,
                      a6,
                      v275,
                      v276,
                      a9,
                      a10,
                      v417,
                      v433);
              if ( v86 )
                return v86;
            }
          }
          goto LABEL_175;
        }
        v67 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
        if ( v67 == 16 )
          v67 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
        v391 = a1->m128i_i64[1];
        v68 = *(_QWORD *)(a2 - 72);
        v380 = v68;
        if ( (unsigned __int8)(v67 - 1) <= 5u )
        {
          v447 = 257;
          if ( v61 )
            v69 = *(_QWORD *)(v415 - 8);
          else
            v69 = v415 - 24LL * (*(_DWORD *)(v415 + 20) & 0xFFFFFFF);
          v70 = *(_QWORD *)(v69 + 24);
          if ( *(_BYTE *)(v70 + 16) <= 0x10u )
          {
            v71 = (__int64 *)sub_15A2BF0(
                               (__int64 *)v70,
                               257,
                               v69,
                               v68,
                               *(double *)v13.m128_u64,
                               a4,
                               *(double *)v14.m128i_i64);
            v72 = sub_14DBA30((__int64)v71, *(_QWORD *)(v391 + 96), 0);
            if ( v72 )
              v71 = (__int64 *)v72;
LABEL_78:
            if ( *((_BYTE *)v71 + 16) > 0x17u )
            {
              v375 = sub_15F24E0(v384);
              v73 = sub_15F24E0(v415);
              sub_15F2440((__int64)v71, v73 & v375);
            }
            goto LABEL_80;
          }
          v450 = 257;
          v71 = (__int64 *)sub_15FB5B0((__int64 *)v70, (__int64)&v448, 0, v68);
          v299 = *(_QWORD *)(v391 + 32);
          v300 = *(_DWORD *)(v391 + 40);
          if ( v299 )
          {
            v388 = *(_DWORD *)(v391 + 40);
            sub_1625C10((__int64)v71, 3, v299);
            v300 = v388;
          }
          sub_15F2440((__int64)v71, v300);
          v301 = *(_QWORD *)(v391 + 8);
          if ( v301 )
          {
            v389 = *(unsigned __int64 **)(v391 + 16);
            sub_157E9D0(v301 + 40, (__int64)v71);
            v302 = *v389;
            v303 = v71[3] & 7;
            v71[4] = (__int64)v389;
            v302 &= 0xFFFFFFFFFFFFFFF8LL;
            v71[3] = v302 | v303;
            *(_QWORD *)(v302 + 8) = v71 + 3;
            *v389 = *v389 & 7 | (unsigned __int64)(v71 + 3);
          }
          v231 = (__int64 *)&v445;
          v232 = v71;
          sub_164B780((__int64)v71, (__int64 *)&v445);
          v436 = v71;
          if ( *(_QWORD *)(v391 + 80) )
          {
            (*(void (__fastcall **)(__int64, __int64 **))(v391 + 88))(v391 + 64, &v436);
            v304 = *(_QWORD **)v391;
            if ( *(_QWORD *)v391 )
            {
              v454.m128_u64[0] = *(_QWORD *)v391;
              sub_1623A60((__int64)&v454, (__int64)v304, 2);
              v305 = v71[6];
              v306 = (__int64)(v71 + 6);
              if ( v305 )
              {
                sub_161E7C0((__int64)(v71 + 6), v305);
                v306 = (__int64)(v71 + 6);
              }
              v307 = (unsigned __int8 *)v454.m128_u64[0];
              v71[6] = v454.m128_u64[0];
              if ( v307 )
                sub_1623210((__int64)&v454, v307, v306);
            }
            goto LABEL_78;
          }
LABEL_557:
          sub_4263D6(v232, v231, v233);
        }
        if ( *(_BYTE *)(v415 + 16) == 38 )
        {
          v444 = 257;
          if ( v61 )
            v191 = *(_QWORD *)(v415 - 8);
          else
            v191 = v415 - 24LL * (*(_DWORD *)(v415 + 20) & 0xFFFFFFF);
          v192 = *(_QWORD *)(v191 + 24);
          if ( *(_BYTE *)(v192 + 16) <= 0x10u )
          {
            v156 = sub_15A2BF0((__int64 *)v192, v45, v191, 257, *(double *)v13.m128_u64, a4, *(double *)v14.m128i_i64);
LABEL_211:
            v376 = (__int64 *)v156;
            v157 = sub_14DBA30(v156, *(_QWORD *)(v391 + 96), 0);
            v71 = v376;
            if ( v157 )
              v71 = (__int64 *)v157;
            goto LABEL_80;
          }
          v453 = 257;
          v71 = (__int64 *)sub_15FB5B0((__int64 *)v192, (__int64)&v451, 0, 257);
          v278 = *(_QWORD *)(v391 + 32);
          v279 = *(_DWORD *)(v391 + 40);
          if ( v278 )
          {
            v377 = *(_DWORD *)(v391 + 40);
            sub_1625C10((__int64)v71, 3, v278);
            v279 = v377;
          }
          sub_15F2440((__int64)v71, v279);
          v280 = *(_QWORD *)(v391 + 8);
          if ( v280 )
          {
            v378 = *(unsigned __int64 **)(v391 + 16);
            sub_157E9D0(v280 + 40, (__int64)v71);
            v281 = *v378;
            v282 = v71[3] & 7;
            v71[4] = (__int64)v378;
            v281 &= 0xFFFFFFFFFFFFFFF8LL;
            v71[3] = v281 | v282;
            *(_QWORD *)(v281 + 8) = v71 + 3;
            *v378 = *v378 & 7 | (unsigned __int64)(v71 + 3);
          }
          v231 = (__int64 *)&v442;
          v232 = v71;
          sub_164B780((__int64)v71, (__int64 *)&v442);
          v437 = v71;
          if ( !*(_QWORD *)(v391 + 80) )
            goto LABEL_557;
          (*(void (__fastcall **)(__int64, __int64 **))(v391 + 88))(v391 + 64, &v437);
          v283 = *(_QWORD *)v391;
          if ( *(_QWORD *)v391 )
          {
            v454.m128_u64[0] = *(_QWORD *)v391;
            sub_1623A60((__int64)&v454, v283, 2);
            v284 = v71[6];
            v285 = (__int64)(v71 + 6);
            if ( v284 )
            {
              sub_161E7C0((__int64)(v71 + 6), v284);
              v285 = (__int64)(v71 + 6);
            }
            v286 = (unsigned __int8 *)v454.m128_u64[0];
            v71[6] = v454.m128_u64[0];
            if ( v286 )
              sub_1623210((__int64)&v454, v286, v285);
          }
        }
        else
        {
          v441 = 257;
          if ( v61 )
            v154 = *(_QWORD *)(v415 - 8);
          else
            v154 = v415 - 24LL * (*(_DWORD *)(v415 + 20) & 0xFFFFFFF);
          v155 = *(_QWORD *)(v154 + 24);
          if ( *(_BYTE *)(v155 + 16) <= 0x10u )
          {
            v156 = sub_15A2B90((__int64 *)v155, 0, 0, v68, *(double *)v13.m128_u64, a4, *(double *)v14.m128i_i64);
            goto LABEL_211;
          }
          v455.m128i_i16[0] = 257;
          v71 = (__int64 *)sub_15FB530((__int64 *)v155, (__int64)&v454, 0, v68);
          v228 = *(_QWORD *)(v391 + 8);
          if ( v228 )
          {
            v387 = *(unsigned __int64 **)(v391 + 16);
            sub_157E9D0(v228 + 40, (__int64)v71);
            v229 = *v387;
            v230 = v71[3] & 7;
            v71[4] = (__int64)v387;
            v229 &= 0xFFFFFFFFFFFFFFF8LL;
            v71[3] = v229 | v230;
            *(_QWORD *)(v229 + 8) = v71 + 3;
            *v387 = *v387 & 7 | (unsigned __int64)(v71 + 3);
          }
          v231 = (__int64 *)&v439;
          v232 = v71;
          sub_164B780((__int64)v71, (__int64 *)&v439);
          v438 = v71;
          if ( !*(_QWORD *)(v391 + 80) )
            goto LABEL_557;
          (*(void (__fastcall **)(__int64, __int64 **))(v391 + 88))(v391 + 64, &v438);
          v234 = *(_QWORD *)v391;
          if ( *(_QWORD *)v391 )
          {
            v451 = *(__int64 **)v391;
            sub_1623A60((__int64)&v451, v234, 2);
            v235 = v71[6];
            v236 = (__int64)(v71 + 6);
            if ( v235 )
            {
              sub_161E7C0((__int64)(v71 + 6), v235);
              v236 = (__int64)(v71 + 6);
            }
            v237 = (unsigned __int8 *)v451;
            v71[6] = (__int64)v451;
            if ( v237 )
              sub_1623210((__int64)&v451, v237, v236);
          }
        }
LABEL_80:
        if ( v398 == v384 )
        {
          v188 = v71;
          v71 = (__int64 *)v381;
          v381 = (__int64)v188;
        }
        v74 = sub_1649960(a2);
        v452 = v75;
        v451 = (__int64 *)v74;
        v454.m128_u64[0] = (unsigned __int64)&v451;
        v454.m128_u64[1] = (unsigned __int64)".p";
        v455.m128i_i16[0] = 773;
        v76 = sub_1707C10(v391, v380, (__int64)v71, v381, (__int64 *)&v454, a2);
        v77 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
        if ( v77 == 16 )
          v77 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
        v78 = *(_BYTE *)(v415 + 23);
        v455.m128i_i16[0] = 257;
        v79 = v78 & 0x40;
        if ( (unsigned __int8)(v77 - 1) > 5u )
        {
          if ( v79 )
            v158 = *(__int64 ***)(v415 - 8);
          else
            v158 = (__int64 **)(v415 - 24LL * (*(_DWORD *)(v415 + 20) & 0xFFFFFFF));
          v81 = sub_15FB440(11, *v158, (__int64)v76, (__int64)&v454, 0);
        }
        else
        {
          if ( v79 )
            v80 = *(__int64 ***)(v415 - 8);
          else
            v80 = (__int64 **)(v415 - 24LL * (*(_DWORD *)(v415 + 20) & 0xFFFFFFF));
          v81 = sub_15FB440(12, *v80, (__int64)v76, (__int64)&v454, 0);
          v399 = sub_15F24E0(v384);
          v82 = sub_15F24E0(v415);
          sub_15F2440(v81, v82 & v399);
        }
        if ( !v81 )
          goto LABEL_89;
        return v81;
      }
      if ( v58 != 36 )
        goto LABEL_89;
    }
    v384 = *(_QWORD *)(a2 - 24);
    v415 = *(_QWORD *)(a2 - 48);
    goto LABEL_63;
  }
  v113 = *(_QWORD *)(v11 - 48);
  if ( !v113 || v113 != v10 || *(_QWORD *)(v11 - 24) != v12 )
  {
    if ( v12 != v113 )
      goto LABEL_51;
    v114 = *(_QWORD *)(v11 - 24);
    if ( !v114 || v114 != v10 )
      goto LABEL_51;
    v115 = *(unsigned __int16 *)(v11 + 18);
    BYTE1(v115) &= ~0x80u;
    v116 = v115;
    if ( v115 == 1 )
    {
      if ( *(_BYTE *)(v10 + 16) == 14 && !sub_17919D0(v10 + 24) || *(_BYTE *)(v12 + 16) == 14 && !sub_17919D0(v12 + 24) )
        goto LABEL_380;
    }
    else if ( v115 == 14
           && (*(_BYTE *)(v10 + 16) == 14 && !sub_17919D0(v10 + 24)
            || *(_BYTE *)(v12 + 16) == 14 && !sub_17919D0(v12 + 24)) )
    {
      goto LABEL_374;
    }
    v117 = *(_QWORD *)(v11 + 8);
    if ( !v117 || *(_QWORD *)(v117 + 8) )
      goto LABEL_51;
    if ( sub_15FF810(v116) )
    {
      v402 = sub_15FF0F0(*(_WORD *)(v11 + 18) & 0x7FFF);
      v162 = a1->m128i_i64[1];
      v419 = *(_DWORD *)(v162 + 40);
      v412 = *(_QWORD *)(v162 + 32);
      *(_DWORD *)(v162 + 40) = sub_15F24E0(v11);
      v429 = a1->m128i_i64[1];
      v314 = sub_1649960(v11);
      v164 = v10;
      v451 = (__int64 *)v314;
      v452 = v315;
      v166 = v12;
      v455.m128i_i16[0] = 773;
      v454.m128_u64[0] = (unsigned __int64)&v451;
      v454.m128_u64[1] = (unsigned __int64)".inv";
      goto LABEL_227;
    }
LABEL_154:
    v55 = *(_BYTE *)(v11 + 16);
    goto LABEL_50;
  }
  v159 = *(unsigned __int16 *)(v11 + 18);
  BYTE1(v159) &= ~0x80u;
  v160 = v159;
  if ( v159 == 1 )
  {
    if ( (*(_BYTE *)(v10 + 16) != 14 || sub_17919D0(v10 + 24)) && (*(_BYTE *)(v12 + 16) != 14 || sub_17919D0(v12 + 24)) )
    {
LABEL_223:
      v161 = *(_QWORD *)(v11 + 8);
      if ( !v161 || *(_QWORD *)(v161 + 8) )
        goto LABEL_51;
      if ( sub_15FF810(v160) )
      {
        v402 = sub_15FF0F0(*(_WORD *)(v11 + 18) & 0x7FFF);
        v162 = a1->m128i_i64[1];
        v419 = *(_DWORD *)(v162 + 40);
        v412 = *(_QWORD *)(v162 + 32);
        *(_DWORD *)(v162 + 40) = sub_15F24E0(v11);
        v429 = a1->m128i_i64[1];
        v163 = sub_1649960(v11);
        v164 = v12;
        v451 = (__int64 *)v163;
        v455.m128i_i16[0] = 773;
        v454.m128_u64[0] = (unsigned __int64)&v451;
        v454.m128_u64[1] = (unsigned __int64)".inv";
        v452 = v165;
        v166 = v10;
LABEL_227:
        v430 = sub_17290F0(v429, v402, v166, v164, (__int64 *)&v454, 0);
        v451 = (__int64 *)sub_1649960(a2);
        v452 = v167;
        v454.m128_u64[0] = (unsigned __int64)&v451;
        v455.m128i_i16[0] = 773;
        v454.m128_u64[1] = (unsigned __int64)".p";
        v428 = sub_14EDD70((__int64)v430, (_QWORD *)v12, v10, (__int64)&v454, 0, 0);
        *(_DWORD *)(v162 + 40) = v419;
        *(_QWORD *)(v162 + 32) = v412;
        return v428;
      }
      goto LABEL_154;
    }
LABEL_380:
    v240 = v12;
    return sub_170E100(a1->m128i_i64, a2, v240, v13, a4, *(double *)v14.m128i_i64, a6, v238, v239, a9, a10);
  }
  if ( v159 != 14
    || (*(_BYTE *)(v10 + 16) != 14 || sub_17919D0(v10 + 24)) && (*(_BYTE *)(v12 + 16) != 14 || sub_17919D0(v12 + 24)) )
  {
    goto LABEL_223;
  }
LABEL_374:
  v240 = v10;
  return sub_170E100(a1->m128i_i64, a2, v240, v13, a4, *(double *)v14.m128i_i64, a6, v238, v239, a9, a10);
}
