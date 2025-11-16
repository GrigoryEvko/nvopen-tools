// Function: sub_2935C30
// Address: 0x2935c30
//
__int64 __fastcall sub_2935C30(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 result; // rax
  int v9; // edx
  __int64 v10; // rdx
  __int64 *v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rdx
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // rdi
  int v20; // eax
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int64 v24; // r15
  __int64 *v25; // r14
  char v26; // al
  const char *v27; // rbx
  __int64 v28; // r12
  const char *v29; // r13
  bool v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  unsigned __int64 v33; // r8
  const char *v34; // r12
  __int64 *v35; // rdi
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  _BYTE *v40; // r13
  __int64 v41; // r12
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 i; // r15
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 *v52; // r12
  __int64 *v53; // r13
  __int64 v54; // rsi
  __int64 v55; // rsi
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  unsigned __int64 *v59; // rcx
  __int64 v60; // rdx
  char v61; // al
  unsigned __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rdi
  __int64 v65; // r12
  int v66; // eax
  int v67; // esi
  unsigned int v68; // eax
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  unsigned int v72; // ebx
  __int64 *v73; // rdi
  unsigned __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // r12
  __int64 *v77; // rax
  unsigned __int64 v78; // rax
  int v79; // edx
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // r12
  _QWORD *v83; // rax
  __int64 v84; // r9
  __int64 v85; // r14
  unsigned int *v86; // r12
  unsigned int *v87; // rbx
  __int64 v88; // rdx
  char **v89; // r13
  __int64 v90; // rdx
  unsigned __int64 v91; // r15
  __int64 *v92; // r12
  char *v93; // rax
  unsigned int v94; // edx
  int v95; // eax
  __int64 *v96; // rdx
  __int64 v97; // rax
  __int64 *v98; // rax
  unsigned __int64 v99; // rdi
  __int64 v100; // rdx
  __int64 v101; // rcx
  char v102; // al
  unsigned __int64 v103; // r12
  char v104; // al
  unsigned __int64 v105; // rdx
  __int64 v106; // r13
  __int64 v107; // r14
  __int64 *v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rsi
  __int64 (__fastcall *v111)(__int64, unsigned __int64, __int64); // rax
  unsigned int *v112; // rbx
  unsigned int *v113; // r15
  __int64 v114; // rdx
  unsigned int v115; // esi
  __int64 *v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rsi
  __int64 (__fastcall *v119)(__int64, unsigned __int64, __int64); // rax
  unsigned int *v120; // rbx
  unsigned int *v121; // r15
  __int64 v122; // rdx
  unsigned int v123; // esi
  char *v124; // rax
  __int64 v125; // rdx
  unsigned __int16 v126; // cx
  __int64 v127; // rbx
  _QWORD *v128; // rax
  __int64 v129; // r15
  unsigned int *v130; // r13
  unsigned int *v131; // rbx
  __int64 v132; // rdx
  unsigned int v133; // esi
  char *v134; // rax
  __int64 v135; // rdx
  unsigned __int16 v136; // cx
  __int64 v137; // rbx
  _QWORD *v138; // rax
  __int64 v139; // r9
  __int64 v140; // r13
  unsigned int *v141; // r14
  unsigned int *v142; // rbx
  __int64 v143; // rdx
  unsigned int v144; // esi
  unsigned __int64 v145; // rsi
  unsigned __int64 v146; // rax
  __int64 v147; // rdx
  __int64 v148; // rax
  __int64 *v149; // rbx
  __int64 v150; // r8
  __int64 v151; // rax
  __int64 v152; // r13
  unsigned __int64 v153; // rax
  __int64 v154; // rdi
  unsigned __int64 v155; // rax
  unsigned __int64 v156; // rcx
  bool v157; // cf
  unsigned __int64 v158; // rax
  unsigned __int8 *v159; // r15
  const char *v160; // rax
  __int64 v161; // rdx
  __int64 v162; // r14
  __int64 v163; // rax
  __int64 v164; // r15
  unsigned __int64 v165; // rax
  __int64 v166; // r14
  __int64 m; // rax
  char **v168; // rdx
  char v169; // al
  int v170; // eax
  int v171; // eax
  unsigned int v172; // edx
  __int64 v173; // rax
  __int64 v174; // rdx
  __int64 v175; // rdx
  unsigned __int8 *v176; // rbx
  unsigned __int8 *v177; // r12
  char *v178; // r13
  const char *v179; // rax
  __int64 v180; // r13
  __int64 v181; // rdx
  unsigned __int64 v182; // rax
  unsigned __int64 v183; // rsi
  __int64 v184; // rax
  __int64 v185; // rdx
  __int64 v186; // rdx
  bool v187; // zf
  char *v188; // rbx
  const char *v189; // rax
  __int64 v190; // rdx
  __int64 v191; // rax
  const char **v192; // rdi
  const char **v193; // rsi
  __int64 n; // rcx
  __int64 v195; // rax
  __int64 v196; // r9
  __int64 v197; // rax
  __int64 v198; // rax
  __int64 v199; // r15
  __int64 *v200; // rbx
  __int64 v201; // rax
  __int64 v202; // rdi
  __int64 v203; // r8
  __int64 v204; // rax
  __int64 v205; // r9
  __int64 v206; // r13
  unsigned __int64 v207; // rax
  __int64 v208; // rdi
  unsigned __int64 v209; // rax
  unsigned __int64 v210; // rcx
  unsigned __int64 v211; // rax
  unsigned __int8 *v212; // r14
  const char *v213; // rax
  __int64 v214; // rdx
  unsigned __int64 v215; // rax
  __int64 v216; // r14
  __int64 j; // rax
  unsigned __int8 *v218; // rbx
  unsigned __int8 *v219; // r15
  char *v220; // r13
  const char *v221; // rax
  __int64 v222; // r13
  __int64 v223; // rdx
  unsigned __int64 v224; // rax
  unsigned __int64 v225; // rsi
  __int64 v226; // rax
  __int64 v227; // rdx
  __int64 v228; // rdx
  char *v229; // rbx
  const char *v230; // rax
  char v231; // al
  __int64 v232; // rdx
  char **v233; // rdx
  int v234; // eax
  int v235; // edx
  __int64 v236; // rax
  __int64 v237; // rdx
  __int64 v238; // rdx
  __int64 v239; // rax
  __int64 v240; // r14
  __int64 v241; // rax
  __int64 v242; // r8
  __int64 v243; // rbx
  __int64 v244; // r9
  __int64 v245; // rax
  unsigned int *v246; // r15
  unsigned int *v247; // rbx
  __int64 v248; // rdx
  unsigned int v249; // esi
  unsigned int *v250; // r15
  unsigned int *v251; // rbx
  __int64 v252; // rdx
  unsigned int v253; // esi
  __int64 v254; // rdx
  __int64 v255; // rcx
  __int64 v256; // r8
  __int64 v257; // r9
  int v258; // r13d
  const char **v259; // rdi
  const char **v260; // rsi
  __int64 k; // rcx
  __int64 v262; // rax
  __int64 v263; // r8
  __int64 v264; // rax
  __int64 v265; // rax
  __int64 v266; // rax
  __int64 v267; // rdi
  __int64 v268; // r8
  __int64 v269; // rax
  __int64 v270; // r9
  __int64 v271; // r12
  __int64 v272; // rbx
  unsigned __int64 v273; // rdi
  __int64 v274; // [rsp-10h] [rbp-6C0h]
  __int64 v275; // [rsp+8h] [rbp-6A8h]
  __int64 v276; // [rsp+28h] [rbp-688h]
  __int64 v277; // [rsp+30h] [rbp-680h]
  __int64 v278; // [rsp+38h] [rbp-678h]
  __int64 v279; // [rsp+40h] [rbp-670h]
  _BYTE *v280; // [rsp+48h] [rbp-668h]
  __int64 v281; // [rsp+50h] [rbp-660h]
  char v282; // [rsp+5Bh] [rbp-655h]
  __int16 v283; // [rsp+5Ch] [rbp-654h]
  __int16 v284; // [rsp+5Eh] [rbp-652h]
  char v285; // [rsp+60h] [rbp-650h]
  __int64 v286; // [rsp+60h] [rbp-650h]
  __int64 v287; // [rsp+68h] [rbp-648h]
  unsigned __int64 v288; // [rsp+68h] [rbp-648h]
  __int64 v289; // [rsp+70h] [rbp-640h]
  __int64 v290; // [rsp+78h] [rbp-638h]
  __int64 v291; // [rsp+78h] [rbp-638h]
  int v292; // [rsp+80h] [rbp-630h]
  __int64 v293; // [rsp+80h] [rbp-630h]
  unsigned __int8 *v294; // [rsp+88h] [rbp-628h]
  int v295; // [rsp+88h] [rbp-628h]
  __int64 v296; // [rsp+90h] [rbp-620h]
  unsigned __int64 v297; // [rsp+98h] [rbp-618h]
  unsigned __int8 *v298; // [rsp+98h] [rbp-618h]
  __int64 v299; // [rsp+A0h] [rbp-610h]
  __int64 v300; // [rsp+A8h] [rbp-608h]
  __int64 v301; // [rsp+B0h] [rbp-600h]
  _BOOL4 v302; // [rsp+B0h] [rbp-600h]
  unsigned __int8 *v303; // [rsp+B0h] [rbp-600h]
  __int64 v304; // [rsp+B8h] [rbp-5F8h]
  unsigned __int8 *v305; // [rsp+B8h] [rbp-5F8h]
  _BOOL4 v306; // [rsp+B8h] [rbp-5F8h]
  __int64 v307; // [rsp+C0h] [rbp-5F0h]
  char v308; // [rsp+C8h] [rbp-5E8h]
  _BYTE *v309; // [rsp+C8h] [rbp-5E8h]
  unsigned __int64 v310; // [rsp+C8h] [rbp-5E8h]
  __int64 v311; // [rsp+F0h] [rbp-5C0h]
  __int64 v312; // [rsp+F0h] [rbp-5C0h]
  __int64 v313; // [rsp+100h] [rbp-5B0h]
  __int64 v314; // [rsp+100h] [rbp-5B0h]
  unsigned __int8 v315; // [rsp+108h] [rbp-5A8h]
  __int64 v317; // [rsp+110h] [rbp-5A0h]
  char v318; // [rsp+110h] [rbp-5A0h]
  unsigned int v319; // [rsp+110h] [rbp-5A0h]
  unsigned int v320; // [rsp+110h] [rbp-5A0h]
  _BYTE *v321; // [rsp+118h] [rbp-598h]
  __int64 v322; // [rsp+120h] [rbp-590h]
  int v323; // [rsp+128h] [rbp-588h]
  int v324; // [rsp+128h] [rbp-588h]
  int v325; // [rsp+128h] [rbp-588h]
  int v326; // [rsp+128h] [rbp-588h]
  int v327; // [rsp+128h] [rbp-588h]
  unsigned int v328; // [rsp+128h] [rbp-588h]
  bool v329; // [rsp+128h] [rbp-588h]
  char v330; // [rsp+128h] [rbp-588h]
  int v331; // [rsp+128h] [rbp-588h]
  __int64 *v332; // [rsp+128h] [rbp-588h]
  __int64 v333; // [rsp+128h] [rbp-588h]
  unsigned __int64 v334; // [rsp+130h] [rbp-580h] BYREF
  unsigned __int64 v335; // [rsp+138h] [rbp-578h] BYREF
  unsigned __int64 v336; // [rsp+140h] [rbp-570h]
  __int64 v337; // [rsp+148h] [rbp-568h]
  const char *v338; // [rsp+150h] [rbp-560h] BYREF
  __int64 v339; // [rsp+158h] [rbp-558h]
  char *v340; // [rsp+160h] [rbp-550h]
  __int16 v341; // [rsp+170h] [rbp-540h]
  char *v342; // [rsp+180h] [rbp-530h] BYREF
  __int64 v343; // [rsp+188h] [rbp-528h]
  const char *v344; // [rsp+190h] [rbp-520h]
  __int64 v345; // [rsp+198h] [rbp-518h]
  __int16 v346; // [rsp+1A0h] [rbp-510h]
  const char **v347; // [rsp+1B0h] [rbp-500h] BYREF
  __int64 v348; // [rsp+1B8h] [rbp-4F8h]
  const char *v349; // [rsp+1C0h] [rbp-4F0h] BYREF
  __int64 v350; // [rsp+1C8h] [rbp-4E8h]
  __int16 v351; // [rsp+1D0h] [rbp-4E0h]
  __int64 *v352; // [rsp+1E0h] [rbp-4D0h] BYREF
  _BYTE *v353; // [rsp+1E8h] [rbp-4C8h] BYREF
  __int64 v354; // [rsp+1F0h] [rbp-4C0h] BYREF
  _BYTE v355[40]; // [rsp+1F8h] [rbp-4B8h] BYREF
  unsigned int *v356; // [rsp+220h] [rbp-490h] BYREF
  __int64 v357; // [rsp+228h] [rbp-488h]
  _BYTE v358[32]; // [rsp+230h] [rbp-480h] BYREF
  __int64 v359; // [rsp+250h] [rbp-460h]
  __int64 v360; // [rsp+258h] [rbp-458h]
  __int64 v361; // [rsp+260h] [rbp-450h]
  __int64 v362; // [rsp+268h] [rbp-448h]
  void **v363; // [rsp+270h] [rbp-440h]
  __int64 (__fastcall ***v364)(); // [rsp+278h] [rbp-438h]
  __int64 v365; // [rsp+280h] [rbp-430h]
  int v366; // [rsp+288h] [rbp-428h]
  __int16 v367; // [rsp+28Ch] [rbp-424h]
  char v368; // [rsp+28Eh] [rbp-422h]
  __int64 v369; // [rsp+290h] [rbp-420h]
  __int64 v370; // [rsp+298h] [rbp-418h]
  void *v371; // [rsp+2A0h] [rbp-410h] BYREF
  __int64 (__fastcall **v372)(); // [rsp+2A8h] [rbp-408h] BYREF
  _QWORD *v373; // [rsp+2B0h] [rbp-400h]
  __int64 v374; // [rsp+2B8h] [rbp-3F8h]
  _BYTE v375[16]; // [rsp+2C0h] [rbp-3F0h] BYREF
  unsigned __int64 v376; // [rsp+2D0h] [rbp-3E0h] BYREF
  unsigned __int64 v377; // [rsp+2D8h] [rbp-3D8h]
  __int64 v378; // [rsp+2E0h] [rbp-3D0h] BYREF
  unsigned int v379; // [rsp+2E8h] [rbp-3C8h]
  __int64 v380; // [rsp+320h] [rbp-390h] BYREF
  char *v381; // [rsp+328h] [rbp-388h]
  __int64 v382; // [rsp+330h] [rbp-380h]
  int v383; // [rsp+338h] [rbp-378h]
  char v384; // [rsp+33Ch] [rbp-374h]
  char v385; // [rsp+340h] [rbp-370h] BYREF
  __int64 v386; // [rsp+380h] [rbp-330h]
  __int64 v387; // [rsp+388h] [rbp-328h]
  unsigned int **v388; // [rsp+390h] [rbp-320h]
  __int64 *v389; // [rsp+4A0h] [rbp-210h] BYREF
  __int64 v390; // [rsp+4A8h] [rbp-208h]
  __int64 v391; // [rsp+4B0h] [rbp-200h] BYREF
  __int64 *v392; // [rsp+4B8h] [rbp-1F8h]
  unsigned int v393; // [rsp+4C0h] [rbp-1F0h]
  _BYTE v394[40]; // [rsp+4C8h] [rbp-1E8h] BYREF
  __int64 v395; // [rsp+4F0h] [rbp-1C0h]
  char *v396; // [rsp+4F8h] [rbp-1B8h]
  __int64 v397; // [rsp+500h] [rbp-1B0h]
  int v398; // [rsp+508h] [rbp-1A8h]
  char v399; // [rsp+50Ch] [rbp-1A4h]
  char v400; // [rsp+510h] [rbp-1A0h] BYREF
  __int64 v401; // [rsp+550h] [rbp-160h]
  __int64 v402; // [rsp+558h] [rbp-158h]
  unsigned int **v403; // [rsp+560h] [rbp-150h]
  _BYTE *v404; // [rsp+588h] [rbp-128h]
  int v405; // [rsp+590h] [rbp-120h]
  _BYTE v406[64]; // [rsp+598h] [rbp-118h] BYREF
  _BYTE *v407; // [rsp+5D8h] [rbp-D8h]
  _BYTE v408[64]; // [rsp+5E8h] [rbp-C8h] BYREF
  __int64 *v409; // [rsp+628h] [rbp-88h]
  int v410; // [rsp+630h] [rbp-80h]
  _BYTE v411[120]; // [rsp+638h] [rbp-78h] BYREF

  v2 = (__int64)a2;
  if ( !a2[2] )
  {
    sub_B43D60(a2);
    return 1;
  }
  v4 = sub_B43CC0((__int64)a2);
  v5 = a2[9];
  v6 = v4;
  v336 = sub_BDB740(v4, v5);
  v337 = v7;
  if ( (unsigned __int8)sub_B4CE70((__int64)a2) )
    return 0;
  v9 = *(unsigned __int8 *)(v5 + 8);
  if ( (_BYTE)v9 != 12
    && (unsigned __int8)v9 > 3u
    && (_BYTE)v9 != 5
    && (v9 & 0xFB) != 0xA
    && (v9 & 0xFD) != 4
    && ((unsigned __int8)(*(_BYTE *)(v5 + 8) - 15) > 3u && v9 != 20 || !(unsigned __int8)sub_BCEBA0(v5, 0)) )
  {
    return 0;
  }
  if ( (_BYTE)v337 )
    return 0;
  if ( !v336 )
    return 0;
  v389 = (__int64 *)sub_BDB740(v6, v5);
  v390 = v10;
  if ( (unsigned __int64)v389 > qword_50056C8 )
    return 0;
  v362 = sub_BD5C60((__int64)a2);
  v363 = &v371;
  v364 = &v372;
  v367 = 512;
  LOWORD(v361) = 0;
  v356 = (unsigned int *)v358;
  v371 = &unk_49DA100;
  v372 = off_49D3D08;
  v357 = 0x200000000LL;
  v365 = 0;
  v366 = 0;
  v368 = 7;
  v369 = 0;
  v370 = 0;
  v359 = 0;
  v360 = 0;
  v373 = v375;
  v374 = 0;
  v375[0] = 0;
  sub_D5F1F0((__int64)&v356, (__int64)a2);
  if ( (_BYTE)qword_50055E8 )
  {
    sub_2927160((__int64)&v389, v6, (__int64)a2, 1);
    v315 = 0;
    if ( !v390 )
    {
      v380 = 0;
      v376 = (unsigned __int64)&v378;
      v384 = 1;
      v377 = 0x800000000LL;
      v381 = &v385;
      v382 = 8;
      v383 = 0;
      v386 = 0;
      v387 = v6;
      v388 = &v356;
      v315 = sub_2924690((__int64)&v376, (__int64)a2, v15, v16, v17, v18);
      if ( !v384 )
        _libc_free((unsigned __int64)v381);
      if ( (__int64 *)v376 != &v378 )
        _libc_free(v376);
    }
    if ( v409 != (__int64 *)v411 )
      _libc_free((unsigned __int64)v409);
    if ( v407 != v408 )
      _libc_free((unsigned __int64)v407);
    if ( v404 != v406 )
      _libc_free((unsigned __int64)v404);
    v19 = (unsigned __int64)v392;
    if ( v392 != (__int64 *)v394 )
LABEL_19:
      _libc_free(v19);
  }
  else
  {
    v395 = 0;
    v390 = 0x800000000LL;
    v396 = &v400;
    v389 = &v391;
    v397 = 8;
    v398 = 0;
    v399 = 1;
    v401 = 0;
    v402 = v6;
    v403 = &v356;
    v315 = sub_2924690((__int64)&v389, (__int64)a2, v11, v12, v13, v14);
    if ( !v399 )
      _libc_free((unsigned __int64)v396);
    v19 = (unsigned __int64)v389;
    if ( v389 != &v391 )
      goto LABEL_19;
  }
  sub_2927160((__int64)&v389, v6, (__int64)a2, 0);
  if ( v390 )
  {
    v20 = v315;
    BYTE1(v20) = 0;
    goto LABEL_22;
  }
  if ( v391 )
  {
    v21 = (unsigned __int64)v392;
    v322 = *v392;
    v22 = v393;
    v23 = v393;
    if ( !(3LL * v393) )
      goto LABEL_72;
    v24 = v392[1];
    v25 = v392;
    while ( 1 )
    {
      v376 = (unsigned __int64)&v378;
      v377 = 0x600000000LL;
      if ( v25 != (__int64 *)(v21 + 24 * v22) )
        break;
LABEL_69:
      v22 = v23;
      v35 = (__int64 *)v376;
      if ( v25 == (__int64 *)(v21 + 24LL * v23) )
        goto LABEL_116;
      v24 = v25[1];
      v322 = *v25;
      if ( (__int64 *)v376 != &v378 )
      {
        _libc_free(v376);
        v22 = v393;
        v21 = (unsigned __int64)v392;
        v23 = v393;
        if ( v25 == &v392[3 * v393] )
        {
LABEL_72:
          v20 = 1;
          goto LABEL_22;
        }
      }
    }
    v26 = 1;
    v27 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( *v25 >= v24 && v25[1] > v24 )
        {
          v34 = v27;
          if ( v26 )
          {
LABEL_68:
            v23 = v393;
            if ( !(_DWORD)v377 )
              goto LABEL_69;
            v347 = &v349;
            v348 = 0x400000000LL;
            sub_11D2BF0((__int64)&v352, (__int64)&v347);
            v38 = (unsigned int)v377;
            v39 = (unsigned int)v377 + 1LL;
            if ( v39 > HIDWORD(v377) )
            {
              sub_C8D5F0((__int64)&v376, &v378, v39, 8u, v36, v37);
              v38 = (unsigned int)v377;
            }
            *(_QWORD *)(v376 + 8 * v38) = a2;
            LODWORD(v377) = v377 + 1;
            sub_11D3120(&v342, (_BYTE **)v376, (unsigned int)v377, (__int64 *)&v352, 0, 0);
            v344 = v34;
            v342 = (char *)&unk_4A21E50;
            sub_11D7E80((__int64 **)&v342, (__int64)&v376);
            sub_11D2C20((__int64 *)&v352);
            if ( v347 != &v349 )
              _libc_free((unsigned __int64)v347);
            v21 = (unsigned __int64)v392;
          }
          v23 = v393;
          goto LABEL_69;
        }
        if ( v26 )
          break;
LABEL_75:
        if ( v24 < v25[1] )
          v24 = v25[1];
        v25 += 3;
        if ( v25 == (__int64 *)(v21 + 24LL * v393) )
          goto LABEL_115;
        v26 = 0;
      }
      v329 = 0;
      if ( *v25 == v322 )
        v329 = v25[1] == v24;
      v28 = *(_QWORD *)((v25[2] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( *(_BYTE *)v28 == 61 )
      {
        v29 = *(const char **)(v28 + 8);
LABEL_58:
        v30 = sub_B46500(*(unsigned __int8 **)((v25[2] & 0xFFFFFFFFFFFFFFF8LL) + 24));
        v26 = 0;
        if ( !v30 )
        {
          v26 = *(_BYTE *)(v28 + 2) & 1;
          if ( v26 )
          {
            v26 = 0;
          }
          else if ( !v27 || v29 == v27 )
          {
            v26 = v329;
          }
        }
        v32 = (unsigned int)v377;
        v33 = (unsigned int)v377 + 1LL;
        if ( v33 > HIDWORD(v377) )
        {
          v330 = v26;
          sub_C8D5F0((__int64)&v376, &v378, (unsigned int)v377 + 1LL, 8u, v33, v31);
          v32 = (unsigned int)v377;
          v26 = v330;
        }
        v27 = v29;
        *(_QWORD *)(v376 + 8 * v32) = v28;
        LODWORD(v377) = v377 + 1;
        goto LABEL_64;
      }
      if ( *(_BYTE *)v28 == 62 )
      {
        v29 = *(const char **)(*(_QWORD *)(v28 - 64) + 8LL);
        goto LABEL_58;
      }
      if ( !sub_988C10(*(_QWORD *)((v25[2] & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
      {
        v21 = (unsigned __int64)v392;
        goto LABEL_75;
      }
      v26 = v329;
LABEL_64:
      v21 = (unsigned __int64)v392;
      if ( v24 < v25[1] )
        v24 = v25[1];
      v25 += 3;
      if ( v25 == &v392[3 * v393] )
      {
        v34 = v27;
        if ( v26 )
          goto LABEL_68;
LABEL_115:
        v35 = (__int64 *)v376;
LABEL_116:
        if ( v35 != &v378 )
          _libc_free((unsigned __int64)v35);
        goto LABEL_72;
      }
    }
  }
  v40 = v404;
  v321 = &v404[8 * v405];
  if ( v321 != v404 )
  {
    do
    {
      v41 = *(_QWORD *)v40;
      v42 = 32LL * (*(_DWORD *)(*(_QWORD *)v40 + 4LL) & 0x7FFFFFF);
      if ( (*(_BYTE *)(*(_QWORD *)v40 + 7LL) & 0x40) != 0 )
      {
        v43 = *(_QWORD *)(v41 - 8);
        v44 = v43 + v42;
      }
      else
      {
        v43 = v41 - v42;
        v44 = *(_QWORD *)v40;
      }
      for ( i = v43; v44 != i; i += 32 )
      {
        v46 = i;
        sub_29220F0(a1, v46);
      }
      v47 = sub_ACADE0(*(__int64 ***)(v41 + 8));
      sub_BD84D0(v41, v47);
      v378 = v41;
      v376 = 4;
      v377 = 0;
      if ( v41 != -8192 && v41 != -4096 )
        sub_BD73F0((__int64)&v376);
      sub_D6B260(a1 + 216, (char *)&v376, v48, v49, v50, v51);
      if ( v378 != 0 && v378 != -4096 && v378 != -8192 )
        sub_BD60C0(&v376);
      v40 += 8;
    }
    while ( v321 != v40 );
    v2 = (__int64)a2;
    v315 = 1;
  }
  v52 = v409;
  v53 = &v409[v410];
  if ( v53 != v409 )
  {
    do
    {
      v54 = *v52++;
      sub_29220F0(a1, v54);
    }
    while ( v53 != v52 );
    v315 = 1;
  }
  if ( 24LL * v393 )
  {
    v55 = v2;
    v282 = sub_2930B90(a1, v2, (__int64)&v389);
    v58 = *(unsigned int *)(a1 + 864);
    if ( (_DWORD)v58 )
    {
      v59 = &v376;
      do
      {
        v63 = *(_QWORD *)(a1 + 856);
        v64 = *(_QWORD *)(a1 + 832);
        v65 = *(_QWORD *)(v63 + 8 * v58 - 8);
        v66 = *(_DWORD *)(a1 + 848);
        if ( v66 )
        {
          v67 = v66 - 1;
          v68 = (v66 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
          v63 = v64 + 8LL * v68;
          v59 = *(unsigned __int64 **)v63;
          if ( *(_QWORD *)v63 == v65 )
          {
LABEL_128:
            *(_QWORD *)v63 = -8192;
            --*(_DWORD *)(a1 + 840);
            ++*(_DWORD *)(a1 + 844);
          }
          else
          {
            v63 = 1;
            while ( v59 != (unsigned __int64 *)-4096LL )
            {
              v56 = (unsigned int)(v63 + 1);
              v68 = v67 & (v63 + v68);
              v63 = v64 + 8LL * v68;
              v59 = *(unsigned __int64 **)v63;
              if ( v65 == *(_QWORD *)v63 )
                goto LABEL_128;
              v63 = (unsigned int)v56;
            }
          }
        }
        --*(_DWORD *)(a1 + 864);
        v55 = v65;
        sub_2914D90((__int64)&v352, v65, v63, (__int64)v59, v56, v57);
        if ( (_DWORD)v353 )
        {
          v60 = (__int64)v352;
          v317 = v352[(unsigned int)v353 - 1];
          v61 = *(_BYTE *)v317;
          if ( *(_BYTE *)v317 > 0x1Cu )
          {
            if ( v61 == 61 )
            {
              v55 = v65;
              sub_29348F0((__int64 *)&v356, v65, (__int64)&v352);
            }
            else if ( v61 == 62 )
            {
              v311 = *(_QWORD *)(v65 + 40);
              v55 = v352[(unsigned int)v353 - 1];
              sub_B91FC0((__int64 *)&v342, v317);
              _BitScanReverse64(&v74, 1LL << (*(_WORD *)(v317 + 2) >> 1));
              v308 = 63 - (v74 ^ 0x3F);
              v307 = *(_QWORD *)(v317 - 64);
              v304 = *(_QWORD *)(v317 - 32);
              v75 = 32LL * *(unsigned int *)(v65 + 72);
              v332 = (__int64 *)(*(_QWORD *)(v65 - 8) + v75);
              v301 = v75 + 8LL * (*(_DWORD *)(v65 + 4) & 0x7FFFFFF) + *(_QWORD *)(v65 - 8);
              if ( (__int64 *)v301 != v332 )
              {
                v290 = v65;
                v287 = a1;
                do
                {
                  v76 = *v332;
                  v77 = &v378;
                  v376 = 0;
                  v377 = 1;
                  do
                  {
                    *v77 = -4096;
                    v77 += 2;
                  }
                  while ( v77 != &v380 );
                  v78 = *(_QWORD *)(v76 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v78 == v76 + 48 )
                  {
                    v80 = 0;
                  }
                  else
                  {
                    if ( !v78 )
                      BUG();
                    v79 = *(unsigned __int8 *)(v78 - 24);
                    v80 = 0;
                    v81 = v78 - 24;
                    if ( (unsigned int)(v79 - 30) < 0xB )
                      v80 = v81;
                  }
                  sub_D5F1F0((__int64)&v356, v80);
                  v286 = sub_2926390(v304, (__int64 *)&v356, v311, v76, (__int64)&v376);
                  v82 = sub_2926390(v307, (__int64 *)&v356, v311, v76, (__int64)&v376);
                  v351 = 257;
                  v83 = sub_BD2C40(80, unk_3F10A10);
                  v85 = (__int64)v83;
                  if ( v83 )
                    sub_B4D3C0((__int64)v83, v82, v286, 0, v308, v84, 0, 0);
                  v55 = v85;
                  ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, const char ***, __int64, __int64))(*v364)[2])(
                    v364,
                    v85,
                    &v347,
                    v360,
                    v361);
                  v86 = v356;
                  v87 = &v356[4 * (unsigned int)v357];
                  if ( v356 != v87 )
                  {
                    do
                    {
                      v88 = *((_QWORD *)v86 + 1);
                      v55 = *v86;
                      v86 += 4;
                      sub_B99FD0(v85, v55, v88);
                    }
                    while ( v87 != v86 );
                  }
                  if ( v342 || v343 || v344 || v345 )
                  {
                    v55 = (__int64)&v342;
                    sub_B9A100(v85, (__int64 *)&v342);
                  }
                  if ( (v377 & 1) == 0 )
                  {
                    v55 = 16LL * v379;
                    sub_C7D6A0(v378, v55, 8);
                  }
                  ++v332;
                }
                while ( (__int64 *)v301 != v332 );
                v65 = v290;
                a1 = v287;
              }
              sub_B43D60((_QWORD *)v317);
            }
          }
          sub_2914800(v65, v55, v60, (__int64)v59, v56, v57);
          sub_B43D60((_QWORD *)v65);
          v62 = (unsigned __int64)v352;
          if ( v352 == &v354 )
            goto LABEL_125;
        }
        else
        {
          v62 = (unsigned __int64)v352;
          if ( v352 == &v354 )
            goto LABEL_125;
        }
        _libc_free(v62);
LABEL_125:
        v58 = *(unsigned int *)(a1 + 864);
      }
      while ( (_DWORD)v58 );
    }
    sub_2921860(a1 + 936);
    v72 = *(_DWORD *)(a1 + 1088);
    v376 = (unsigned __int64)&v378;
    v377 = 0x800000000LL;
    if ( !v72 )
    {
      v285 = 0;
      v73 = &v378;
LABEL_134:
      v20 = (unsigned __int8)(v315 | v282);
      BYTE1(v20) = v285;
      if ( v73 != &v378 )
      {
        v331 = v20;
        _libc_free((unsigned __int64)v73);
        v20 = v331;
      }
      goto LABEL_22;
    }
    v89 = *(char ***)(a1 + 1080);
    if ( v89 == (char **)(a1 + 1096) )
    {
      v90 = v72;
      if ( v72 > 8 )
      {
        v55 = sub_C8D7D0((__int64)&v376, (__int64)&v378, v72, 0x38u, (unsigned __int64 *)&v352, v71);
        sub_2913A00((__int64 **)&v376, v55, v254, v255, v256, v257);
        v258 = (int)v352;
        if ( (__int64 *)v376 != &v378 )
          _libc_free(v376);
        HIDWORD(v377) = v258;
        v89 = *(char ***)(a1 + 1080);
        v376 = v55;
      }
      v91 = v376;
      v92 = (__int64 *)&v89[7 * *(unsigned int *)(a1 + 1088)];
      while ( v92 != (__int64 *)v89 )
      {
        if ( v91 )
        {
          v93 = *v89;
          *(_DWORD *)(v91 + 16) = 0;
          *(_DWORD *)(v91 + 20) = 2;
          *(_QWORD *)v91 = v93;
          *(_QWORD *)(v91 + 8) = v91 + 24;
          if ( *((_DWORD *)v89 + 4) )
          {
            v55 = (__int64)(v89 + 1);
            sub_29138A0(v91 + 8, v89 + 1, v90, v69, v70, v71);
          }
        }
        v91 += 56LL;
        v89 += 7;
      }
      LODWORD(v377) = v72;
      v271 = *(_QWORD *)(a1 + 1080);
      v272 = v271 + 56LL * *(unsigned int *)(a1 + 1088);
      while ( v271 != v272 )
      {
        v272 -= 56;
        v273 = *(_QWORD *)(v272 + 8);
        if ( v273 != v272 + 24 )
          _libc_free(v273);
      }
      *(_DWORD *)(a1 + 1088) = 0;
    }
    else
    {
      v94 = *(_DWORD *)(a1 + 1092);
      v376 = *(_QWORD *)(a1 + 1080);
      v377 = __PAIR64__(v94, v72);
      *(_QWORD *)(a1 + 1080) = a1 + 1096;
      *(_QWORD *)(a1 + 1088) = 0;
    }
    v285 = 0;
    v275 = a1;
    while ( 1 )
    {
      v95 = v377;
      v73 = (__int64 *)v376;
      if ( !(_DWORD)v377 )
        goto LABEL_134;
      v96 = (__int64 *)(v376 + 56LL * (unsigned int)v377 - 56);
      v352 = (__int64 *)*v96;
      v353 = v355;
      v354 = 0x200000000LL;
      if ( *((_DWORD *)v96 + 4) )
      {
        v55 = (__int64)(v96 + 1);
        sub_29138A0((__int64)&v353, (char **)v96 + 1, (__int64)v96, 0x200000000LL, v70, v71);
        v95 = v377;
        v73 = (__int64 *)v376;
      }
      v97 = (unsigned int)(v95 - 1);
      LODWORD(v377) = v97;
      v98 = &v73[7 * v97];
      v99 = v98[1];
      if ( (__int64 *)v99 != v98 + 3 )
        _libc_free(v99);
      v312 = 0;
      if ( !*(_BYTE *)(v275 + 32) )
        v312 = *(_QWORD *)(v275 + 8);
      v333 = (__int64)v352;
      v100 = 16LL * (unsigned int)v354;
      v101 = (__int64)&v353[v100];
      v280 = &v353[v100];
      if ( v353 != &v353[v100] )
        break;
      v318 = 0;
LABEL_217:
      sub_2914800(v333, v55, v100, v101, v70, v71);
      sub_B43D60((_QWORD *)v333);
      v285 |= v318;
      if ( v353 != v355 )
        _libc_free((unsigned __int64)v353);
    }
    v288 = (unsigned __int64)v353;
    v318 = 0;
    while ( 1 )
    {
      v102 = *(_BYTE *)(v288 + 8);
      if ( v102 == 1 )
        break;
      if ( v102 )
        abort();
      v103 = *(_QWORD *)v288 & 0xFFFFFFFFFFFFFFF8LL;
      v105 = (*(__int64 *)v288 >> 1) & 3;
      if ( ((*(__int64 *)v288 >> 1) & 1) != 0 )
      {
        if ( v105 >> 1 )
        {
          v106 = *(_QWORD *)(v333 - 64);
          v107 = *(_QWORD *)(v333 - 32);
          sub_D5F1F0((__int64)&v356, *(_QWORD *)v288 & 0xFFFFFFFFFFFFFFF8LL);
          v108 = (__int64 *)sub_BD5D20(v103);
          v346 = 773;
          v342 = (char *)v108;
          v343 = v109;
          v344 = ".sroa.speculate.cast.true";
          v110 = *(_QWORD *)(*(_QWORD *)(v103 - 32) + 8LL);
          if ( v110 != *(_QWORD *)(v106 + 8) )
          {
            if ( *(_BYTE *)v106 > 0x15u )
            {
              v351 = 257;
              v106 = sub_B52190(v106, v110, (__int64)&v347, 0, 0);
              ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, char **, __int64, __int64))(*v364)[2])(
                v364,
                v106,
                &v342,
                v360,
                v361);
              v246 = v356;
              v247 = &v356[4 * (unsigned int)v357];
              while ( v247 != v246 )
              {
                v248 = *((_QWORD *)v246 + 1);
                v249 = *v246;
                v246 += 4;
                sub_B99FD0(v106, v249, v248);
              }
            }
            else
            {
              v111 = (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))*((_QWORD *)*v363 + 18);
              if ( v111 == sub_B32D70 )
                v106 = sub_ADB060(v106, v110);
              else
                v106 = v111((__int64)v363, v106, v110);
              if ( *(_BYTE *)v106 > 0x1Cu )
              {
                ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, char **, __int64, __int64))(*v364)[2])(
                  v364,
                  v106,
                  &v342,
                  v360,
                  v361);
                v112 = v356;
                v113 = &v356[4 * (unsigned int)v357];
                if ( v356 != v113 )
                {
                  do
                  {
                    v114 = *((_QWORD *)v112 + 1);
                    v115 = *v112;
                    v112 += 4;
                    sub_B99FD0(v106, v115, v114);
                  }
                  while ( v113 != v112 );
                }
              }
            }
          }
          v116 = (__int64 *)sub_BD5D20(v103);
          v346 = 773;
          v342 = (char *)v116;
          v343 = v117;
          v344 = ".sroa.speculate.cast.false";
          v118 = *(_QWORD *)(*(_QWORD *)(v103 - 32) + 8LL);
          if ( v118 != *(_QWORD *)(v107 + 8) )
          {
            if ( *(_BYTE *)v107 > 0x15u )
            {
              v351 = 257;
              v107 = sub_B52190(v107, v118, (__int64)&v347, 0, 0);
              ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, char **, __int64, __int64))(*v364)[2])(
                v364,
                v107,
                &v342,
                v360,
                v361);
              v250 = v356;
              v251 = &v356[4 * (unsigned int)v357];
              while ( v251 != v250 )
              {
                v252 = *((_QWORD *)v250 + 1);
                v253 = *v250;
                v250 += 4;
                sub_B99FD0(v107, v253, v252);
              }
            }
            else
            {
              v119 = (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))*((_QWORD *)*v363 + 18);
              if ( v119 == sub_B32D70 )
                v107 = sub_ADB060(v107, v118);
              else
                v107 = v119((__int64)v363, v107, v118);
              if ( *(_BYTE *)v107 > 0x1Cu )
              {
                ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, char **, __int64, __int64))(*v364)[2])(
                  v364,
                  v107,
                  &v342,
                  v360,
                  v361);
                v120 = v356;
                v121 = &v356[4 * (unsigned int)v357];
                if ( v356 != v121 )
                {
                  do
                  {
                    v122 = *((_QWORD *)v120 + 1);
                    v123 = *v120;
                    v120 += 4;
                    sub_B99FD0(v107, v123, v122);
                  }
                  while ( v121 != v120 );
                }
              }
            }
          }
          v124 = (char *)sub_BD5D20(v103);
          v346 = 773;
          v342 = v124;
          v343 = v125;
          v344 = ".sroa.speculate.load.true";
          v126 = *(_WORD *)(v103 + 2);
          v127 = *(_QWORD *)(v103 + 8);
          v351 = 257;
          _BitScanReverse64((unsigned __int64 *)&v124, 1LL << (v126 >> 1));
          BYTE1(v124) = HIBYTE(v284);
          LOBYTE(v124) = 63 - ((unsigned __int8)v124 ^ 0x3F);
          v284 = (__int16)v124;
          v128 = sub_BD2C40(80, 1u);
          v129 = (__int64)v128;
          if ( v128 )
            sub_B4D190((__int64)v128, v127, v106, (__int64)&v347, 0, v284, 0, 0);
          ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, char **, __int64, __int64))(*v364)[2])(
            v364,
            v129,
            &v342,
            v360,
            v361);
          v130 = v356;
          v131 = &v356[4 * (unsigned int)v357];
          if ( v356 != v131 )
          {
            do
            {
              v132 = *((_QWORD *)v130 + 1);
              v133 = *v130;
              v130 += 4;
              sub_B99FD0(v129, v133, v132);
            }
            while ( v131 != v130 );
          }
          v134 = (char *)sub_BD5D20(v103);
          v346 = 773;
          v342 = v134;
          v343 = v135;
          v344 = ".sroa.speculate.load.false";
          v136 = *(_WORD *)(v103 + 2);
          v137 = *(_QWORD *)(v103 + 8);
          v351 = 257;
          _BitScanReverse64((unsigned __int64 *)&v134, 1LL << (v136 >> 1));
          BYTE1(v134) = HIBYTE(v283);
          LOBYTE(v134) = 63 - ((unsigned __int8)v134 ^ 0x3F);
          v283 = (__int16)v134;
          v138 = sub_BD2C40(80, 1u);
          v140 = (__int64)v138;
          if ( v138 )
          {
            sub_B4D190((__int64)v138, v137, v107, (__int64)&v347, 0, v283, 0, 0);
            v139 = v274;
          }
          ((void (__fastcall *)(__int64 (__fastcall ***)(), __int64, char **, __int64, __int64, __int64))(*v364)[2])(
            v364,
            v140,
            &v342,
            v360,
            v361,
            v139);
          v141 = v356;
          v142 = &v356[4 * (unsigned int)v357];
          if ( v356 != v142 )
          {
            do
            {
              v143 = *((_QWORD *)v141 + 1);
              v144 = *v141;
              v141 += 4;
              sub_B99FD0(v140, v144, v143);
            }
            while ( v142 != v141 );
          }
          _BitScanReverse64(&v145, 1LL << (*(_WORD *)(v103 + 2) >> 1));
          *(_WORD *)(v129 + 2) = (2 * (63 - (v145 ^ 0x3F))) | *(_WORD *)(v129 + 2) & 0xFF81;
          _BitScanReverse64(&v146, 1LL << (*(_WORD *)(v103 + 2) >> 1));
          *(_WORD *)(v140 + 2) = *(_WORD *)(v140 + 2) & 0xFF81 | (2 * (63 - (v146 ^ 0x3F)));
          sub_B91FC0((__int64 *)&v342, v103);
          if ( v342 || v343 || v344 || v345 )
          {
            sub_B9A100(v129, (__int64 *)&v342);
            sub_B9A100(v140, (__int64 *)&v342);
          }
          v347 = (const char **)sub_BD5D20(v103);
          v349 = ".sroa.speculated";
          v348 = v147;
          v351 = 773;
          v55 = sub_B36550(&v356, *(_QWORD *)(v333 - 96), v129, v140, (__int64)&v347, 0);
          sub_BD84D0(v103, v55);
          goto LABEL_216;
        }
        if ( *(_BYTE *)v103 != 61 )
        {
          if ( *(_BYTE *)v103 != 62 )
            goto LABEL_187;
          v199 = *(_QWORD *)(v103 + 40);
          v200 = (__int64 *)(v103 + 24);
          v334 = 0;
          v335 = 0;
          if ( (*(_BYTE *)(v333 + 7) & 0x20) != 0 )
          {
            v201 = sub_B91C10(v333, 2);
            v202 = *(_QWORD *)(v333 - 96);
            v203 = v201;
            v204 = v300;
            v205 = v312;
          }
          else
          {
            v205 = v312;
            v203 = 0;
            v202 = *(_QWORD *)(v333 - 96);
            v204 = v300;
          }
          LOWORD(v204) = 0;
          v300 = v204;
          v206 = v199 + 48;
          sub_F38250(v202, (__int64 *)(v103 + 24), v204, 0, v203, v205, 0, 0);
          v207 = *(_QWORD *)(v199 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v207 == v199 + 48 )
          {
            v208 = 0;
          }
          else
          {
            if ( !v207 )
              BUG();
            v208 = v207 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v207 - 24) - 30 >= 0xB )
              v208 = 0;
          }
          sub_B4CC70(v208);
          goto LABEL_303;
        }
        v148 = *(_QWORD *)(v103 + 40);
        v149 = (__int64 *)(v103 + 24);
        v334 = 0;
        v335 = 0;
        v313 = v148;
        if ( (*(_BYTE *)(v333 + 7) & 0x20) != 0 )
        {
          v150 = sub_B91C10(v333, 2);
          v151 = v299;
          LOWORD(v151) = 0;
          v299 = v151;
          sub_F38250(*(_QWORD *)(v333 - 96), (__int64 *)(v103 + 24), v151, 0, v150, v312, 0, 0);
        }
        else
        {
          v198 = v299;
          LOWORD(v198) = 0;
          v299 = v198;
          sub_F38250(*(_QWORD *)(v333 - 96), (__int64 *)(v103 + 24), v198, 0, 0, v312, 0, 0);
        }
        v152 = v313 + 48;
        v153 = *(_QWORD *)(v313 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v153 == v313 + 48 )
        {
          v154 = 0;
        }
        else
        {
          if ( !v153 )
            BUG();
          v154 = v153 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v153 - 24) - 30 >= 0xB )
            v154 = 0;
        }
        sub_B4CC70(v154);
      }
      else
      {
        if ( *(_BYTE *)v103 != 61 )
        {
          if ( *(_BYTE *)v103 != 62 )
LABEL_187:
            sub_C64FA0("Only for load and store.", 0, 0);
          v199 = *(_QWORD *)(v103 + 40);
          v200 = (__int64 *)(v103 + 24);
          v334 = 0;
          v335 = 0;
          if ( v105 >> 1 )
          {
            if ( (*(_BYTE *)(v333 + 7) & 0x20) != 0 )
            {
              v266 = sub_B91C10(v333, 2);
              v267 = *(_QWORD *)(v333 - 96);
              v268 = v266;
              v269 = v300;
              v270 = v312;
            }
            else
            {
              v270 = v312;
              v268 = 0;
              v267 = *(_QWORD *)(v333 - 96);
              v269 = v300;
            }
            LOWORD(v269) = 0;
            v300 = v269;
            v206 = v199 + 48;
            sub_F38250(v267, (__int64 *)(v103 + 24), v269, 0, v268, v270, 0, 0);
            goto LABEL_303;
          }
LABEL_368:
          if ( (*(_BYTE *)(v333 + 7) & 0x20) != 0 )
            v244 = sub_B91C10(v333, 2);
          else
            v244 = 0;
          v245 = v278;
          LOWORD(v245) = 0;
          v278 = v245;
          sub_F38330(*(_QWORD *)(v333 - 96), v200, v245, &v334, &v335, v244, v312, 0);
          v206 = v199 + 48;
LABEL_303:
          v209 = *(_QWORD *)(v199 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v206 == v209 )
          {
            v310 = 0;
          }
          else
          {
            if ( !v209 )
              BUG();
            v210 = v209 - 24;
            v157 = (unsigned int)*(unsigned __int8 *)(v209 - 24) - 30 < 0xB;
            v211 = 0;
            if ( v157 )
              v211 = v210;
            v310 = v211;
          }
          v212 = *(unsigned __int8 **)(v103 + 40);
          v303 = v212;
          v213 = sub_BD5D20(v199);
          v351 = 773;
          v347 = (const char **)v213;
          v349 = ".cont";
          v55 = (__int64)&v347;
          v348 = v214;
          sub_BD6B50(v212, (const char **)&v347);
          if ( *(_BYTE *)v103 == 61 )
          {
            v351 = 257;
            v240 = *(_QWORD *)(v103 + 8);
            v241 = sub_BD2DA0(80);
            v289 = v241;
            if ( v241 )
            {
              v242 = (__int64)v200;
              v243 = v241;
              sub_B44260(v241, v240, 55, 0x8000000u, v242, 0);
              *(_DWORD *)(v243 + 72) = 2;
              sub_BD6B50((unsigned __int8 *)v243, (const char **)&v347);
              v55 = *(unsigned int *)(v243 + 72);
              sub_BD2A10(v243, v55, 1);
            }
          }
          v215 = *(_QWORD *)(v199 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v206 != v215 )
          {
            if ( !v215 )
              BUG();
            v293 = v215 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v215 - 24) - 30 <= 0xA )
            {
              v295 = sub_B46E30(v215 - 24);
              if ( v295 )
              {
                v314 = v199;
                v216 = v279;
                v320 = 0;
                for ( j = sub_B46EC0(v293, 0); ; j = sub_B46EC0(v293, v320) )
                {
                  v218 = (unsigned __int8 *)j;
                  v298 = *(unsigned __int8 **)(v310 - 32);
                  v306 = j != (_QWORD)v298;
                  if ( v303 == (unsigned __int8 *)j )
                    break;
                  v219 = (unsigned __int8 *)sub_B47F80((_BYTE *)v103);
                  if ( (unsigned __int8 *)v314 == v218 )
                  {
                    v222 = v314;
                    goto LABEL_356;
                  }
                  v220 = ".else";
                  if ( v218 == v298 )
                    v220 = ".then";
                  v221 = sub_BD5D20(v314);
                  v349 = v220;
                  v222 = (__int64)v218;
                  v351 = 773;
                  v347 = (const char **)v221;
                  v348 = v223;
                  sub_BD6B50(v218, (const char **)&v347);
LABEL_321:
                  v224 = *(_QWORD *)(v222 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v224 == v222 + 48 )
                  {
                    v225 = 0;
                  }
                  else
                  {
                    if ( !v224 )
                      BUG();
                    v225 = v224 - 24;
                    if ( (unsigned int)*(unsigned __int8 *)(v224 - 24) - 30 >= 0xB )
                      v225 = 0;
                  }
                  LOWORD(v216) = 0;
                  v55 = v225 + 24;
                  sub_B44220(v219, v55, v216);
                  v226 = *(_QWORD *)(v333 + 32LL * (v306 + 1) - 96);
                  if ( v226 )
                  {
                    if ( *((_QWORD *)v219 - 4) )
                    {
                      v227 = *((_QWORD *)v219 - 3);
                      **((_QWORD **)v219 - 2) = v227;
                      if ( v227 )
                        *(_QWORD *)(v227 + 16) = *((_QWORD *)v219 - 2);
                    }
                    *((_QWORD *)v219 - 4) = v226;
                    v228 = *(_QWORD *)(v226 + 16);
                    *((_QWORD *)v219 - 3) = v228;
                    if ( v228 )
                    {
                      v55 = (__int64)(v219 - 24);
                      *(_QWORD *)(v228 + 16) = v219 - 24;
                    }
                    *((_QWORD *)v219 - 2) = v226 + 16;
                    *(_QWORD *)(v226 + 16) = v219 - 32;
                  }
                  else if ( *((_QWORD *)v219 - 4) )
                  {
                    v239 = *((_QWORD *)v219 - 3);
                    **((_QWORD **)v219 - 2) = v239;
                    if ( v239 )
                      *(_QWORD *)(v239 + 16) = *((_QWORD *)v219 - 2);
                    *((_QWORD *)v219 - 4) = 0;
                  }
                  if ( *(_BYTE *)v103 == 61 )
                  {
                    v187 = v218 == v298;
                    HIBYTE(v346) = 1;
                    v342 = ".val";
                    v229 = ".else";
                    if ( v187 )
                      v229 = ".then";
                    LOBYTE(v346) = 3;
                    v230 = sub_BD5D20(v103);
                    v341 = 773;
                    v338 = v230;
                    v231 = v346;
                    v339 = v232;
                    v340 = v229;
                    if ( (_BYTE)v346 )
                    {
                      if ( (_BYTE)v346 == 1 )
                      {
                        v259 = (const char **)&v347;
                        v260 = &v338;
                        for ( k = 10; k; --k )
                        {
                          *(_DWORD *)v259 = *(_DWORD *)v260;
                          v260 = (const char **)((char *)v260 + 4);
                          v259 = (const char **)((char *)v259 + 4);
                        }
                      }
                      else
                      {
                        if ( HIBYTE(v346) == 1 )
                        {
                          v233 = (char **)v342;
                          v281 = v343;
                        }
                        else
                        {
                          v233 = &v342;
                          v231 = 2;
                        }
                        v349 = (const char *)v233;
                        v347 = &v338;
                        LOBYTE(v351) = 2;
                        v350 = v281;
                        HIBYTE(v351) = v231;
                      }
                    }
                    else
                    {
                      v351 = 256;
                    }
                    v55 = (__int64)&v347;
                    sub_BD6B50(v219, (const char **)&v347);
                    v234 = *(_DWORD *)(v289 + 4);
                    if ( (v234 & 0x7FFFFFF) == *(_DWORD *)(v289 + 72) )
                    {
                      sub_B48D90(v289);
                      v234 = *(_DWORD *)(v289 + 4);
                    }
                    v235 = (v234 + 1) & 0x7FFFFFF;
                    *(_DWORD *)(v289 + 4) = v235 | v234 & 0xF8000000;
                    v236 = *(_QWORD *)(v289 - 8) + 32LL * (unsigned int)(v235 - 1);
                    if ( *(_QWORD *)v236 )
                    {
                      v237 = *(_QWORD *)(v236 + 8);
                      **(_QWORD **)(v236 + 16) = v237;
                      if ( v237 )
                        *(_QWORD *)(v237 + 16) = *(_QWORD *)(v236 + 16);
                    }
                    *(_QWORD *)v236 = v219;
                    if ( v219 )
                    {
                      v238 = *((_QWORD *)v219 + 2);
                      *(_QWORD *)(v236 + 8) = v238;
                      if ( v238 )
                      {
                        v55 = v236 + 8;
                        *(_QWORD *)(v238 + 16) = v236 + 8;
                      }
                      *(_QWORD *)(v236 + 16) = v219 + 16;
                      *((_QWORD *)v219 + 2) = v236;
                    }
                    *(_QWORD *)(*(_QWORD *)(v289 - 8)
                              + 32LL * *(unsigned int *)(v289 + 72)
                              + 8LL * ((*(_DWORD *)(v289 + 4) & 0x7FFFFFFu) - 1)) = v222;
                  }
                  if ( v295 == ++v320 )
                  {
                    v279 = v216;
                    goto LABEL_358;
                  }
                }
                v222 = v314;
                v219 = (unsigned __int8 *)sub_B47F80((_BYTE *)v103);
LABEL_356:
                sub_B44E20(v219);
                goto LABEL_321;
              }
            }
          }
LABEL_358:
          if ( *(_BYTE *)v103 == 61 )
          {
            sub_BD6B90((unsigned __int8 *)v289, (unsigned __int8 *)v103);
            v55 = v289;
            sub_BD84D0(v103, v289);
          }
          goto LABEL_282;
        }
        v262 = *(_QWORD *)(v103 + 40);
        v149 = (__int64 *)(v103 + 24);
        v334 = 0;
        v335 = 0;
        v313 = v262;
        if ( !(v105 >> 1) )
          goto LABEL_290;
        if ( (*(_BYTE *)(v333 + 7) & 0x20) != 0 )
        {
          v263 = sub_B91C10(v333, 2);
          v264 = v299;
          LOWORD(v264) = 0;
          v299 = v264;
          sub_F38250(*(_QWORD *)(v333 - 96), (__int64 *)(v103 + 24), v264, 0, v263, v312, 0, 0);
        }
        else
        {
          v265 = v299;
          LOWORD(v265) = 0;
          v299 = v265;
          sub_F38250(*(_QWORD *)(v333 - 96), (__int64 *)(v103 + 24), v265, 0, 0, v312, 0, 0);
        }
        v152 = v313 + 48;
      }
LABEL_227:
      v155 = *(_QWORD *)(v313 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v152 == v155 )
      {
        v297 = 0;
      }
      else
      {
        if ( !v155 )
          BUG();
        v156 = v155 - 24;
        v157 = (unsigned int)*(unsigned __int8 *)(v155 - 24) - 30 < 0xB;
        v158 = 0;
        if ( v157 )
          v158 = v156;
        v297 = v158;
      }
      v159 = *(unsigned __int8 **)(v103 + 40);
      v294 = v159;
      v160 = sub_BD5D20(v313);
      v351 = 773;
      v347 = (const char **)v160;
      v349 = ".cont";
      v348 = v161;
      sub_BD6B50(v159, (const char **)&v347);
      v351 = 257;
      v162 = *(_QWORD *)(v103 + 8);
      v163 = sub_BD2DA0(80);
      v164 = v163;
      if ( v163 )
      {
        sub_B44260(v163, v162, 55, 0x8000000u, (__int64)v149, 0);
        *(_DWORD *)(v164 + 72) = 2;
        sub_BD6B50((unsigned __int8 *)v164, (const char **)&v347);
        sub_BD2A10(v164, *(_DWORD *)(v164 + 72), 1);
      }
      v165 = *(_QWORD *)(v313 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v152 != v165 )
      {
        if ( !v165 )
          BUG();
        v291 = v165 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v165 - 24) - 30 <= 0xA )
        {
          v292 = sub_B46E30(v165 - 24);
          if ( v292 )
          {
            v309 = (_BYTE *)v103;
            v166 = v277;
            v319 = 0;
            for ( m = sub_B46EC0(v291, 0); ; m = sub_B46EC0(v291, v319) )
            {
              v176 = (unsigned __int8 *)m;
              v305 = *(unsigned __int8 **)(v297 - 32);
              v302 = m != (_QWORD)v305;
              if ( v294 == (unsigned __int8 *)m )
                break;
              v177 = (unsigned __int8 *)sub_B47F80(v309);
              if ( v176 == (unsigned __int8 *)v313 )
                goto LABEL_275;
              v178 = ".else";
              if ( v176 == v305 )
                v178 = ".then";
              v179 = sub_BD5D20(v313);
              v351 = 773;
              v349 = v178;
              v180 = (__int64)v176;
              v347 = (const char **)v179;
              v348 = v181;
              sub_BD6B50(v176, (const char **)&v347);
LABEL_259:
              v182 = *(_QWORD *)(v180 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v182 == v180 + 48 )
              {
                v183 = 0;
              }
              else
              {
                if ( !v182 )
                  BUG();
                v183 = v182 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v182 - 24) - 30 >= 0xB )
                  v183 = 0;
              }
              LOWORD(v166) = 0;
              sub_B44220(v177, v183 + 24, v166);
              v184 = *(_QWORD *)(v333 + 32LL * (v302 + 1) - 96);
              if ( v184 )
              {
                if ( *((_QWORD *)v177 - 4) )
                {
                  v185 = *((_QWORD *)v177 - 3);
                  **((_QWORD **)v177 - 2) = v185;
                  if ( v185 )
                    *(_QWORD *)(v185 + 16) = *((_QWORD *)v177 - 2);
                }
                *((_QWORD *)v177 - 4) = v184;
                v186 = *(_QWORD *)(v184 + 16);
                *((_QWORD *)v177 - 3) = v186;
                if ( v186 )
                  *(_QWORD *)(v186 + 16) = v177 - 24;
                *((_QWORD *)v177 - 2) = v184 + 16;
                *(_QWORD *)(v184 + 16) = v177 - 32;
              }
              else if ( *((_QWORD *)v177 - 4) )
              {
                v191 = *((_QWORD *)v177 - 3);
                **((_QWORD **)v177 - 2) = v191;
                if ( v191 )
                  *(_QWORD *)(v191 + 16) = *((_QWORD *)v177 - 2);
                *((_QWORD *)v177 - 4) = 0;
              }
              v187 = v176 == v305;
              v188 = ".else";
              v342 = ".val";
              if ( v187 )
                v188 = ".then";
              v346 = 259;
              v189 = sub_BD5D20((__int64)v309);
              v340 = v188;
              v338 = v189;
              v169 = v346;
              v341 = 773;
              v339 = v190;
              if ( (_BYTE)v346 )
              {
                if ( (_BYTE)v346 == 1 )
                {
                  v192 = (const char **)&v347;
                  v193 = &v338;
                  for ( n = 10; n; --n )
                  {
                    *(_DWORD *)v192 = *(_DWORD *)v193;
                    v193 = (const char **)((char *)v193 + 4);
                    v192 = (const char **)((char *)v192 + 4);
                  }
                }
                else
                {
                  if ( HIBYTE(v346) == 1 )
                  {
                    v168 = (char **)v342;
                    v296 = v343;
                  }
                  else
                  {
                    v168 = &v342;
                    v169 = 2;
                  }
                  v349 = (const char *)v168;
                  v347 = &v338;
                  LOBYTE(v351) = 2;
                  v350 = v296;
                  HIBYTE(v351) = v169;
                }
              }
              else
              {
                v351 = 256;
              }
              sub_BD6B50(v177, (const char **)&v347);
              v170 = *(_DWORD *)(v164 + 4) & 0x7FFFFFF;
              if ( v170 == *(_DWORD *)(v164 + 72) )
              {
                sub_B48D90(v164);
                v170 = *(_DWORD *)(v164 + 4) & 0x7FFFFFF;
              }
              v171 = (v170 + 1) & 0x7FFFFFF;
              v172 = v171 | *(_DWORD *)(v164 + 4) & 0xF8000000;
              v173 = *(_QWORD *)(v164 - 8) + 32LL * (unsigned int)(v171 - 1);
              *(_DWORD *)(v164 + 4) = v172;
              if ( *(_QWORD *)v173 )
              {
                v174 = *(_QWORD *)(v173 + 8);
                **(_QWORD **)(v173 + 16) = v174;
                if ( v174 )
                  *(_QWORD *)(v174 + 16) = *(_QWORD *)(v173 + 16);
              }
              *(_QWORD *)v173 = v177;
              if ( v177 )
              {
                v175 = *((_QWORD *)v177 + 2);
                *(_QWORD *)(v173 + 8) = v175;
                if ( v175 )
                  *(_QWORD *)(v175 + 16) = v173 + 8;
                *(_QWORD *)(v173 + 16) = v177 + 16;
                *((_QWORD *)v177 + 2) = v173;
              }
              ++v319;
              *(_QWORD *)(*(_QWORD *)(v164 - 8)
                        + 32LL * *(unsigned int *)(v164 + 72)
                        + 8LL * ((*(_DWORD *)(v164 + 4) & 0x7FFFFFFu) - 1)) = v180;
              if ( v292 == v319 )
              {
                v277 = v166;
                v103 = (unsigned __int64)v309;
                goto LABEL_281;
              }
            }
            v177 = (unsigned __int8 *)sub_B47F80(v309);
LABEL_275:
            sub_B44E20(v177);
            v180 = v313;
            goto LABEL_259;
          }
        }
      }
LABEL_281:
      sub_BD6B90((unsigned __int8 *)v164, (unsigned __int8 *)v103);
      v55 = v164;
      sub_BD84D0(v103, v164);
LABEL_282:
      v318 = 1;
LABEL_216:
      sub_B43D60((_QWORD *)v103);
      v288 += 16LL;
      if ( v280 == (_BYTE *)v288 )
        goto LABEL_217;
    }
    v103 = *(_QWORD *)v288;
    v104 = **(_BYTE **)v288;
    if ( v104 != 61 )
    {
      if ( v104 != 62 )
        goto LABEL_187;
      v199 = *(_QWORD *)(v103 + 40);
      v200 = (__int64 *)(v103 + 24);
      v334 = 0;
      v335 = 0;
      goto LABEL_368;
    }
    v195 = *(_QWORD *)(v103 + 40);
    v149 = (__int64 *)(v103 + 24);
    v334 = 0;
    v335 = 0;
    v313 = v195;
LABEL_290:
    if ( (*(_BYTE *)(v333 + 7) & 0x20) != 0 )
      v196 = sub_B91C10(v333, 2);
    else
      v196 = 0;
    v197 = v276;
    LOWORD(v197) = 0;
    v276 = v197;
    sub_F38330(*(_QWORD *)(v333 - 96), v149, v197, &v334, &v335, v196, v312, 0);
    v152 = v313 + 48;
    goto LABEL_227;
  }
  v20 = v315;
  BYTE1(v20) = 0;
LABEL_22:
  if ( v409 != (__int64 *)v411 )
  {
    v323 = v20;
    _libc_free((unsigned __int64)v409);
    v20 = v323;
  }
  if ( v407 != v408 )
  {
    v324 = v20;
    _libc_free((unsigned __int64)v407);
    v20 = v324;
  }
  if ( v404 != v406 )
  {
    v325 = v20;
    _libc_free((unsigned __int64)v404);
    v20 = v325;
  }
  if ( v392 != (__int64 *)v394 )
  {
    v326 = v20;
    _libc_free((unsigned __int64)v392);
    v20 = v326;
  }
  v372 = off_49D3D08;
  if ( v373 != (_QWORD *)v375 )
  {
    v327 = v20;
    j_j___libc_free_0((unsigned __int64)v373);
    v20 = v327;
  }
  v328 = v20;
  nullsub_61();
  v371 = &unk_49DA100;
  nullsub_63();
  result = v328;
  if ( v356 != (unsigned int *)v358 )
  {
    _libc_free((unsigned __int64)v356);
    return v328;
  }
  return result;
}
