// Function: sub_2703170
// Address: 0x2703170
//
__int64 __fastcall sub_2703170(__int64 a1, __m128i a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r9
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned __int64 v9; // r13
  __int64 v10; // rdx
  size_t v11; // r12
  unsigned __int64 v12; // rsi
  __int64 v13; // r15
  _QWORD *v14; // rcx
  _QWORD *v15; // rax
  __int64 v16; // rbx
  __int64 i; // r12
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 *v21; // r12
  __int64 *v22; // rbx
  __int64 *v23; // r12
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v29; // rbx
  unsigned int v30; // eax
  __int64 v31; // rdi
  unsigned int v32; // eax
  unsigned int v33; // ecx
  _BYTE *v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  _BYTE *v41; // r15
  unsigned __int8 *v42; // rax
  __int64 v43; // r8
  void *v44; // rdx
  char *v45; // rcx
  char *v46; // r14
  char *v47; // r12
  __int64 v48; // r10
  __int64 v49; // rax
  unsigned __int8 *v50; // rbx
  _QWORD *v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rdx
  __int64 *v54; // rax
  __int64 *v55; // r15
  __int64 v56; // r12
  __int64 v57; // r10
  unsigned __int32 v58; // r11d
  __int64 v59; // rdx
  _BYTE **v60; // rax
  _BYTE **v61; // r14
  __int64 v62; // r15
  int v63; // edi
  unsigned int v64; // edx
  __int64 *v65; // rcx
  __int64 *v66; // rax
  _BYTE *v67; // rbx
  _QWORD *v68; // r13
  __int64 v69; // r12
  unsigned __int64 v70; // rcx
  _QWORD *v71; // rbx
  unsigned int v72; // ecx
  int v73; // edx
  _BYTE *v74; // rbx
  __int64 v75; // rdx
  int v76; // edx
  _QWORD **v77; // r14
  _BYTE *v78; // r13
  _QWORD *v79; // rdi
  __int64 v80; // r9
  __int64 v81; // r13
  char *v82; // rax
  __int64 v83; // rdx
  unsigned __int64 *v84; // rbx
  unsigned __int64 *v85; // r15
  __m128i *v86; // rsi
  __int64 v87; // r14
  size_t v88; // r13
  size_t v89; // r12
  size_t v90; // rdx
  signed __int64 v91; // rax
  size_t v92; // r12
  __m128i *v93; // r14
  size_t v94; // rcx
  size_t v95; // rdx
  signed __int64 v96; // rax
  __m128 *v97; // rax
  unsigned __int64 *v98; // rsi
  size_t v99; // rcx
  __int64 v100; // rax
  __int64 *v101; // rdx
  unsigned __int64 *v102; // r14
  _QWORD *v103; // r9
  signed __int64 v104; // rcx
  unsigned int v105; // edi
  __int64 k; // r12
  _BYTE *v107; // rbx
  __int64 *v108; // r14
  unsigned __int64 *v109; // rbx
  unsigned __int64 *v110; // r13
  unsigned __int64 v111; // rdi
  __int64 *v112; // rbx
  __int64 *v113; // r12
  __int64 *v114; // rsi
  __int64 v115; // rbx
  __int64 j; // r13
  __int64 v117; // rdi
  _QWORD **v118; // rbx
  _QWORD **v119; // r12
  _QWORD *v120; // rdi
  _QWORD *v121; // rdx
  __int64 v122; // r8
  __int64 v123; // rdi
  __int64 v124; // rdx
  _QWORD *v125; // rbx
  _QWORD *v126; // rax
  _QWORD *v127; // r12
  __int64 v128; // rsi
  __int64 v129; // rcx
  __int64 v130; // r15
  __int64 *v131; // r12
  _QWORD *v132; // rsi
  unsigned __int64 *v133; // rcx
  unsigned __int64 *v134; // r8
  _QWORD *v135; // rbx
  __int64 v136; // rdi
  char *v137; // r9
  char *v138; // rax
  unsigned __int64 *v139; // rdx
  unsigned __int64 *v140; // rdx
  __int64 v141; // rax
  __int64 v142; // r9
  unsigned __int64 *v143; // rax
  _QWORD **v144; // r12
  __int64 v145; // r9
  unsigned __int64 v146; // r14
  void *v147; // r15
  unsigned __int64 v148; // rax
  __int64 v149; // rdx
  __int64 v150; // r12
  int v151; // eax
  unsigned __int8 *v152; // rax
  _BYTE *v153; // rax
  __int64 *v154; // rax
  unsigned __int64 v155; // r11
  __int64 v156; // r10
  unsigned __int64 v157; // rax
  unsigned __int8 *v158; // rdx
  __int64 v159; // rax
  __int64 v160; // rdx
  __int64 v161; // rax
  __int64 *v162; // r12
  __int64 *v163; // rbx
  __int64 v164; // rdi
  __int64 v165; // r13
  __int64 *v166; // r14
  __int64 *v167; // rbx
  _BYTE *v168; // rsi
  __int64 v169; // r15
  __int64 v170; // r8
  void *v171; // rax
  unsigned __int64 *v172; // r12
  unsigned __int64 v173; // r8
  unsigned __int64 v174; // r12
  unsigned __int64 v175; // rbx
  unsigned __int64 v176; // rdi
  unsigned __int64 v177; // r8
  unsigned __int64 v178; // r12
  unsigned __int64 v179; // rbx
  unsigned __int64 v180; // rdi
  unsigned __int64 v181; // rdi
  unsigned __int64 v182; // rdi
  __int64 v183; // r9
  __int64 v184; // rax
  __int64 v185; // rdx
  size_t v186; // r13
  const void *v187; // r14
  unsigned __int64 v188; // rdx
  _QWORD *v189; // r8
  _QWORD *v190; // rsi
  _QWORD *v191; // rax
  _QWORD **v192; // r14
  _BYTE *v193; // r15
  _QWORD *v194; // rdi
  const void *v195; // rax
  size_t v196; // rdx
  __int64 v197; // rax
  unsigned __int64 *v198; // rsi
  __int64 v199; // rdx
  unsigned __int64 *v200; // r8
  unsigned __int64 *v201; // rax
  unsigned __int64 *v202; // rdi
  size_t v203; // r14
  size_t v204; // rdx
  int v205; // eax
  unsigned int v206; // edi
  __int64 v207; // rcx
  unsigned __int64 v208; // rdi
  unsigned __int64 v209; // rdi
  _QWORD *v210; // r15
  _QWORD *v211; // rcx
  _QWORD *v212; // rdi
  _QWORD *v213; // rcx
  size_t v214; // rbx
  __int64 v215; // r13
  __int64 v216; // r12
  __int64 v217; // rax
  __int64 v218; // rdx
  __int64 v219; // rax
  __int64 v220; // r13
  int v221; // ecx
  __int64 *v222; // rdx
  __int64 v223; // r8
  unsigned __int64 v224; // r14
  __int64 *v225; // rbx
  __int64 v226; // rdi
  __int64 v227; // rax
  unsigned __int64 *v228; // rdx
  unsigned __int64 v229; // r14
  __int64 v230; // rax
  __int64 *v231; // rax
  __int64 v232; // rax
  unsigned int v233; // esi
  __int64 v234; // r15
  _QWORD *v235; // rax
  __int64 v236; // r13
  __int64 v237; // r14
  __int64 v238; // r8
  int v239; // edx
  unsigned __int64 v240; // r12
  __int64 *v241; // rax
  __int64 v242; // rdi
  __int64 *v243; // rbx
  __int64 v244; // rcx
  unsigned __int64 v245; // rax
  __int64 v246; // rdx
  __int64 v247; // rcx
  __int64 *v248; // rbx
  _BYTE *v249; // rsi
  __int64 v250; // rax
  __int64 v251; // rax
  __int32 v252; // edx
  __int64 v253; // r13
  __int64 v254; // r14
  __int64 v255; // r8
  int v256; // edx
  unsigned __int64 v257; // r12
  __int64 *v258; // rax
  __int64 v259; // rdi
  __int64 *v260; // rbx
  __int64 v261; // rcx
  unsigned __int64 v262; // rax
  __int64 v263; // rdx
  __int64 v264; // rcx
  __int64 *v265; // rbx
  _BYTE *v266; // rsi
  __int64 v267; // rax
  __int64 v268; // rax
  __int32 v269; // edx
  __int64 v270; // r12
  __int64 v271; // r8
  int v272; // edi
  __int64 *v273; // rcx
  unsigned int v274; // edx
  __int64 *v275; // rax
  __int64 v276; // rbx
  unsigned __int64 v277; // rdx
  __int64 v278; // rcx
  __int64 *v279; // rbx
  __int64 v280; // rax
  _QWORD *v281; // rcx
  __int64 v282; // r11
  __int64 v283; // r14
  __int64 *v284; // r10
  __int64 *v285; // rsi
  unsigned __int64 v286; // r13
  __int64 v287; // rdi
  char *v288; // r8
  char *v289; // rax
  __int64 *v290; // rdx
  _QWORD *v291; // rax
  _BYTE *v292; // rsi
  __int64 v293; // r12
  __int64 v294; // r8
  int v295; // edi
  __int64 *v296; // rcx
  unsigned int v297; // edx
  __int64 *v298; // rax
  __int64 v299; // rbx
  unsigned __int64 v300; // rdx
  __int64 v301; // rcx
  __int64 *v302; // rbx
  __int64 v303; // rax
  _QWORD *v304; // rcx
  __int64 v305; // r11
  __int64 v306; // r14
  __int64 *v307; // r10
  __int64 *v308; // rsi
  unsigned __int64 v309; // r13
  __int64 v310; // rdi
  char *v311; // r8
  char *v312; // rax
  __int64 *v313; // rdx
  _QWORD *v314; // rax
  _BYTE *v315; // rsi
  __int64 v316; // rax
  _QWORD *v317; // r13
  _QWORD *v318; // rbx
  __int64 v319; // rax
  unsigned __int64 *v320; // rax
  unsigned __int64 v321; // r12
  unsigned __int32 v322; // r12d
  __int64 *v323; // rax
  int v324; // r8d
  unsigned int v325; // eax
  __int64 v326; // r10
  int v327; // esi
  __int64 *v328; // rcx
  __int32 v329; // edx
  __int64 v330; // rdx
  __int64 *v331; // rsi
  int v332; // edi
  unsigned int v333; // ecx
  __int64 v334; // rbx
  unsigned int v335; // ecx
  __int64 v336; // rbx
  int v337; // edi
  unsigned int v338; // edx
  __int64 v339; // rbx
  __int64 v340; // rdx
  unsigned int v341; // ecx
  __int64 *v342; // rdx
  __int64 v343; // rbx
  int v344; // r11d
  unsigned __int32 v345; // r12d
  __int64 *v346; // rax
  int v347; // r8d
  unsigned int v348; // eax
  __int64 v349; // r10
  int v350; // esi
  __int64 *v351; // rcx
  int v352; // esi
  __int64 *v353; // rcx
  int v354; // r11d
  __int64 v355; // rdi
  unsigned int v356; // esi
  __int64 *v357; // rcx
  _BYTE *v358; // rbx
  int v359; // r11d
  __int64 v360; // rax
  __int64 v361; // r8
  unsigned __int64 v362; // r13
  __int64 v363; // rax
  __int32 v364; // ecx
  __int64 v365; // rax
  __int64 v366; // r10
  __int64 *v367; // rdx
  int v368; // esi
  unsigned int v369; // eax
  __int64 *v370; // rdx
  __int64 v371; // r10
  __int64 *v372; // rsi
  int v373; // edi
  __int128 v374; // [rsp-20h] [rbp-3A0h]
  __int128 v375; // [rsp-20h] [rbp-3A0h]
  __int64 v376; // [rsp+8h] [rbp-378h]
  void *s2c; // [rsp+10h] [rbp-370h]
  __m128 *s2; // [rsp+10h] [rbp-370h]
  void *s2d; // [rsp+10h] [rbp-370h]
  void *s2a; // [rsp+10h] [rbp-370h]
  void *s2b; // [rsp+10h] [rbp-370h]
  size_t n; // [rsp+18h] [rbp-368h]
  size_t ne; // [rsp+18h] [rbp-368h]
  size_t na; // [rsp+18h] [rbp-368h]
  size_t nf; // [rsp+18h] [rbp-368h]
  size_t nb; // [rsp+18h] [rbp-368h]
  size_t nc; // [rsp+18h] [rbp-368h]
  size_t ng; // [rsp+18h] [rbp-368h]
  size_t nd; // [rsp+18h] [rbp-368h]
  __int64 v390; // [rsp+20h] [rbp-360h]
  __int64 v391; // [rsp+20h] [rbp-360h]
  _QWORD **v392; // [rsp+20h] [rbp-360h]
  unsigned __int64 *v393; // [rsp+20h] [rbp-360h]
  unsigned __int64 *v394; // [rsp+20h] [rbp-360h]
  __int64 *v395; // [rsp+20h] [rbp-360h]
  __int64 *v396; // [rsp+20h] [rbp-360h]
  __int64 *v397; // [rsp+28h] [rbp-358h]
  _BYTE **v398; // [rsp+30h] [rbp-350h]
  _QWORD *v399; // [rsp+30h] [rbp-350h]
  unsigned __int64 v400; // [rsp+30h] [rbp-350h]
  __int64 v401; // [rsp+30h] [rbp-350h]
  __int64 *v402; // [rsp+30h] [rbp-350h]
  void *v403; // [rsp+38h] [rbp-348h]
  void *v404; // [rsp+38h] [rbp-348h]
  void *v405; // [rsp+38h] [rbp-348h]
  void *v406; // [rsp+38h] [rbp-348h]
  _BYTE **v407; // [rsp+38h] [rbp-348h]
  __int64 v408; // [rsp+40h] [rbp-340h]
  __int64 v409; // [rsp+40h] [rbp-340h]
  __int64 v410; // [rsp+40h] [rbp-340h]
  _QWORD ***v411; // [rsp+40h] [rbp-340h]
  _QWORD *v412; // [rsp+40h] [rbp-340h]
  __int64 *v413; // [rsp+40h] [rbp-340h]
  __int64 v414; // [rsp+40h] [rbp-340h]
  unsigned __int8 *v415; // [rsp+40h] [rbp-340h]
  _BYTE **v416; // [rsp+40h] [rbp-340h]
  __int64 v417; // [rsp+48h] [rbp-338h]
  _BYTE **v418; // [rsp+48h] [rbp-338h]
  __int64 v419; // [rsp+50h] [rbp-330h]
  __int64 v420; // [rsp+50h] [rbp-330h]
  char v421; // [rsp+60h] [rbp-320h]
  __int64 *v422; // [rsp+60h] [rbp-320h]
  __int64 *v423; // [rsp+60h] [rbp-320h]
  unsigned __int32 v424; // [rsp+60h] [rbp-320h]
  __int64 v425; // [rsp+70h] [rbp-310h]
  void *v426; // [rsp+70h] [rbp-310h]
  __int64 v427; // [rsp+70h] [rbp-310h]
  __int64 v428; // [rsp+70h] [rbp-310h]
  __int64 v429; // [rsp+70h] [rbp-310h]
  __int64 v430; // [rsp+70h] [rbp-310h]
  __int64 v431; // [rsp+70h] [rbp-310h]
  __int64 *v432; // [rsp+80h] [rbp-300h]
  __int64 v433; // [rsp+80h] [rbp-300h]
  __int64 *v434; // [rsp+80h] [rbp-300h]
  __int64 v435; // [rsp+80h] [rbp-300h]
  __int64 v436; // [rsp+80h] [rbp-300h]
  __int64 *v438; // [rsp+90h] [rbp-2F0h] BYREF
  __int64 *v439; // [rsp+98h] [rbp-2E8h]
  __int64 v440; // [rsp+A0h] [rbp-2E0h]
  __int64 v441; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 *v442; // [rsp+B8h] [rbp-2C8h]
  __int64 v443; // [rsp+C0h] [rbp-2C0h]
  unsigned int v444; // [rsp+C8h] [rbp-2B8h]
  _BYTE *v445; // [rsp+D0h] [rbp-2B0h] BYREF
  __int64 v446; // [rsp+D8h] [rbp-2A8h] BYREF
  unsigned __int64 *v447; // [rsp+E0h] [rbp-2A0h]
  __int64 *v448; // [rsp+E8h] [rbp-298h]
  __int64 *v449; // [rsp+F0h] [rbp-290h]
  __int64 v450; // [rsp+F8h] [rbp-288h]
  _BYTE *v451; // [rsp+100h] [rbp-280h] BYREF
  __int64 v452; // [rsp+108h] [rbp-278h]
  _QWORD v453[2]; // [rsp+110h] [rbp-270h] BYREF
  __int64 *v454; // [rsp+120h] [rbp-260h]
  __int64 v455; // [rsp+128h] [rbp-258h]
  __int64 v456; // [rsp+130h] [rbp-250h] BYREF
  __m128i v457; // [rsp+140h] [rbp-240h] BYREF
  __m128i v458; // [rsp+150h] [rbp-230h] BYREF
  _QWORD v459[2]; // [rsp+160h] [rbp-220h] BYREF
  _QWORD *v460; // [rsp+170h] [rbp-210h] BYREF
  _QWORD v461[2]; // [rsp+180h] [rbp-200h] BYREF
  __m128i v462; // [rsp+190h] [rbp-1F0h]
  void *s1; // [rsp+1A0h] [rbp-1E0h] BYREF
  __int64 v464; // [rsp+1A8h] [rbp-1D8h]
  __m128i v465; // [rsp+1B0h] [rbp-1D0h] BYREF
  unsigned __int64 *v466; // [rsp+1F0h] [rbp-190h]
  unsigned int v467; // [rsp+1F8h] [rbp-188h]
  char v468; // [rsp+200h] [rbp-180h] BYREF

  v2 = sub_B6AC80(*(_QWORD *)a1, 358);
  v3 = sub_B6AC80(*(_QWORD *)a1, 356);
  v417 = sub_B6AC80(*(_QWORD *)a1, 357);
  v4 = sub_B6AC80(*(_QWORD *)a1, 11);
  v5 = v4;
  if ( !*(_QWORD *)(a1 + 40) )
  {
    if ( v2 && *(_QWORD *)(v2 + 16) && v4 && *(_QWORD *)(v4 + 16) )
    {
      v438 = 0;
      v439 = 0;
      v440 = 0;
      v441 = 0;
      v442 = 0;
      v443 = 0;
      v444 = 0;
      sub_2702830((__int64 *)a1, (__int64)&v438, (__int64)&v441);
      goto LABEL_49;
    }
    if ( (!v3 || !*(_QWORD *)(v3 + 16)) && (!v417 || !*(_QWORD *)(v417 + 16)) )
      return 0;
  }
  v438 = 0;
  v439 = 0;
  v440 = 0;
  v441 = 0;
  v442 = 0;
  v443 = 0;
  v444 = 0;
  sub_2702830((__int64 *)a1, (__int64)&v438, (__int64)&v441);
  if ( !v2 || !v5 )
    goto LABEL_4;
LABEL_49:
  v29 = *(_QWORD *)(v2 + 16);
  if ( !v29 )
    goto LABEL_4;
  v404 = (void *)v3;
  while ( 2 )
  {
    v35 = v29;
    v29 = *(_QWORD *)(v29 + 8);
    v36 = *(_QWORD *)(v35 + 24);
    if ( *(_BYTE *)v36 != 85 )
      goto LABEL_59;
    v37 = *(_QWORD *)(v35 + 24);
    s1 = &v465;
    v451 = v453;
    v464 = 0x100000000LL;
    v452 = 0x100000000LL;
    v38 = sub_B43CB0(v37);
    v39 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 24))(*(_QWORD *)(a1 + 32), v38);
    sub_E02490((__int64)&s1, (unsigned int *)&v451, v36, v39);
    v40 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
    v41 = *(_BYTE **)(*(_QWORD *)(v36 + 32 * (1 - v40)) + 24LL);
    if ( (_DWORD)v452 )
    {
      v42 = sub_BD3990(*(unsigned __int8 **)(v36 - 32 * v40), (__int64)&v451);
      v44 = s1;
      v433 = (__int64)v42;
      v45 = (char *)s1 + 16 * (unsigned int)v464;
      if ( s1 != v45 )
      {
        v390 = v29;
        v6 = (__int64)&v445;
        n = v36;
        v46 = (char *)s1;
        v47 = (char *)s1 + 16 * (unsigned int)v464;
        v48 = a1 + 128;
        do
        {
          v49 = *(_QWORD *)v46;
          v50 = (unsigned __int8 *)*((_QWORD *)v46 + 1);
          v376 = v6;
          v46 += 16;
          s2c = (void *)v48;
          v445 = v41;
          v446 = v49;
          v51 = (_QWORD *)sub_26FA230(v48, (const __m128i *)v6, (__int64)v44, (__int64)v45, v43, v6);
          v52 = sub_26FE470(v51, v50);
          *((_BYTE *)v52 + 24) = 0;
          v458.m128i_i64[1] = (__int64)v50;
          v458.m128i_i64[0] = v433;
          v459[0] = 0;
          sub_26F6620(v52, &v458);
          v48 = (__int64)s2c;
          v6 = v376;
        }
        while ( v47 != v46 );
        v29 = v390;
        v36 = n;
      }
      v30 = v444;
      if ( !v444 )
        goto LABEL_98;
      v31 = (__int64)v442;
    }
    else
    {
      v30 = v444;
      v31 = (__int64)v442;
      if ( !v444 )
        goto LABEL_100;
    }
    v32 = v30 - 1;
    v33 = v32 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
    v34 = *(_BYTE **)(v31 + 56LL * v33);
    if ( v41 != v34 )
    {
      v76 = 1;
      while ( v34 != (_BYTE *)-4096LL )
      {
        v33 = v32 & (v76 + v33);
        v34 = *(_BYTE **)(v31 + 56LL * v33);
        if ( v41 == v34 )
          goto LABEL_53;
        ++v76;
      }
LABEL_98:
      v77 = (_QWORD **)v451;
      v78 = &v451[8 * (unsigned int)v452];
      if ( v451 != v78 )
      {
        do
        {
          v79 = *v77++;
          sub_B43D60(v79);
        }
        while ( v78 != (_BYTE *)v77 );
      }
      goto LABEL_100;
    }
LABEL_53:
    if ( !*(_QWORD *)(a1 + 48) || *v41 )
      goto LABEL_55;
    v435 = *(_QWORD *)(a1 + 48);
    v184 = sub_B91420((__int64)v41);
    v186 = v185;
    v187 = (const void *)v184;
    v188 = sub_B2F650(v184, v185);
    v189 = *(_QWORD **)(v435 + 224);
    v190 = (_QWORD *)(v435 + 216);
    if ( !v189 )
      goto LABEL_289;
    while ( 1 )
    {
      if ( v188 > v189[4] )
      {
        v191 = (_QWORD *)v189[3];
        if ( !v191 )
          goto LABEL_289;
        goto LABEL_285;
      }
      v191 = (_QWORD *)v189[2];
      if ( v188 >= v189[4] )
        break;
      v190 = v189;
      if ( !v191 )
        goto LABEL_289;
LABEL_285:
      v189 = v191;
    }
    v210 = (_QWORD *)v189[3];
    if ( v210 )
    {
      while ( 1 )
      {
        v211 = (_QWORD *)v210[2];
        v212 = (_QWORD *)v210[3];
        if ( v188 >= v210[4] )
        {
          v210 = v190;
          v211 = v212;
        }
        if ( !v211 )
          break;
        v190 = v210;
        v210 = v211;
      }
    }
    else
    {
      v210 = v190;
    }
    if ( v191 )
    {
      while ( 1 )
      {
        v213 = (_QWORD *)v191[3];
        if ( v188 <= v191[4] )
        {
          v213 = (_QWORD *)v191[2];
          v189 = v191;
        }
        if ( !v213 )
          break;
        v191 = v213;
      }
    }
    if ( v189 != v210 )
    {
      v436 = v29;
      v214 = v186;
      v215 = v36;
      v216 = (__int64)v189;
      do
      {
        if ( v214 == *(_QWORD *)(v216 + 48) && (!v214 || !memcmp(*(const void **)(v216 + 40), v187, v214)) )
        {
          v29 = v436;
          goto LABEL_55;
        }
        v216 = sub_220EF30(v216);
      }
      while ( (_QWORD *)v216 != v210 );
      v29 = v436;
      v36 = v215;
    }
LABEL_289:
    v192 = (_QWORD **)v451;
    v193 = &v451[8 * (unsigned int)v452];
    if ( v451 != v193 )
    {
      do
      {
        v194 = *v192++;
        sub_B43D60(v194);
      }
      while ( v193 != (_BYTE *)v192 );
    }
LABEL_100:
    if ( !*(_QWORD *)(v36 + 16) )
      sub_B43D60((_QWORD *)v36);
LABEL_55:
    if ( v451 != (_BYTE *)v453 )
      _libc_free((unsigned __int64)v451);
    if ( s1 != &v465 )
      _libc_free((unsigned __int64)s1);
LABEL_59:
    if ( v29 )
      continue;
    break;
  }
  v3 = (__int64)v404;
LABEL_4:
  if ( v3 )
    sub_2701220(a1, v3);
  if ( v417 )
    sub_2701220(a1, v417);
  if ( *(_QWORD *)(a1 + 48) )
  {
    v432 = *(__int64 **)(a1 + 160);
    v397 = &v432[18 * *(unsigned int *)(a1 + 168)];
    if ( v432 != v397 )
    {
      v7 = a1;
      do
      {
        v8 = *v432;
        if ( !*(_BYTE *)*v432 )
        {
          v9 = v432[1];
          v408 = *(_QWORD *)(v7 + 48);
          v403 = (void *)sub_B91420(*v432);
          v11 = v10;
          v12 = sub_B2F650((__int64)v403, v10);
          v13 = *(_QWORD *)(v408 + 224);
          v14 = (_QWORD *)(v408 + 216);
          if ( v13 )
          {
            while ( 1 )
            {
              while ( v12 > *(_QWORD *)(v13 + 32) )
              {
                v13 = *(_QWORD *)(v13 + 24);
                if ( !v13 )
                  goto LABEL_18;
              }
              v15 = *(_QWORD **)(v13 + 16);
              if ( v12 >= *(_QWORD *)(v13 + 32) )
                break;
              v14 = (_QWORD *)v13;
              v13 = *(_QWORD *)(v13 + 16);
              if ( !v15 )
                goto LABEL_18;
            }
            v121 = *(_QWORD **)(v13 + 24);
            if ( v121 )
            {
              do
              {
                while ( 1 )
                {
                  v122 = v121[2];
                  v123 = v121[3];
                  if ( v12 < v121[4] )
                    break;
                  v121 = (_QWORD *)v121[3];
                  if ( !v123 )
                    goto LABEL_182;
                }
                v14 = v121;
                v121 = (_QWORD *)v121[2];
              }
              while ( v122 );
            }
LABEL_182:
            while ( v15 )
            {
              while ( 1 )
              {
                v124 = v15[3];
                if ( v12 <= v15[4] )
                  break;
                v15 = (_QWORD *)v15[3];
                if ( !v124 )
                  goto LABEL_185;
              }
              v13 = (__int64)v15;
              v15 = (_QWORD *)v15[2];
            }
LABEL_185:
            if ( (_QWORD *)v13 == v14 )
              goto LABEL_18;
            v411 = (_QWORD ***)v7;
            v125 = v14;
            while ( v11 != *(_QWORD *)(v13 + 48) || v11 && memcmp(*(const void **)(v13 + 40), v403, v11) )
            {
              v13 = sub_220EF30(v13);
              if ( (_QWORD *)v13 == v125 )
              {
                v7 = (__int64)v411;
                goto LABEL_18;
              }
            }
            v126 = *(_QWORD **)(v13 + 112);
            v7 = (__int64)v411;
            if ( v126 )
            {
              v127 = (_QWORD *)(v13 + 104);
              do
              {
                while ( 1 )
                {
                  v128 = v126[2];
                  v129 = v126[3];
                  if ( v126[4] >= v9 )
                    break;
                  v126 = (_QWORD *)v126[3];
                  if ( !v129 )
                    goto LABEL_196;
                }
                v127 = v126;
                v126 = (_QWORD *)v126[2];
              }
              while ( v128 );
LABEL_196:
              if ( v127 != (_QWORD *)(v13 + 104) && v127[4] <= v9 )
              {
                v406 = v432 + 2;
                if ( *((_DWORD *)v127 + 10) == 1 )
                {
                  v392 = *v411;
                  v154 = (__int64 *)sub_BCB120(**v411);
                  v155 = v127[7];
                  v156 = v127[6];
                  s1 = &v465;
                  v400 = v155;
                  v414 = v156;
                  v464 = 0;
                  v157 = sub_BCF480(v154, &v465, 0, 0);
                  sub_BA8C10((__int64)v392, v414, v400, v157, 0);
                  if ( s1 != &v465 )
                  {
                    v415 = v158;
                    _libc_free((unsigned __int64)s1);
                    v158 = v415;
                  }
                  LOBYTE(s1) = 0;
                  sub_26FC5B0(v7, (__int64)v406, v158, (unsigned __int8 **)&s1);
                }
                v412 = v127 + 11;
                v130 = v432[15];
                if ( (__int64 *)v130 != v432 + 13 )
                {
                  v399 = v127;
                  v131 = (__int64 *)v7;
                  while ( 1 )
                  {
                    v132 = (_QWORD *)v399[12];
                    if ( !v132 )
                      goto LABEL_220;
                    v133 = *(unsigned __int64 **)(v130 + 40);
                    v134 = *(unsigned __int64 **)(v130 + 32);
                    v135 = v412;
                    v136 = (char *)v133 - (char *)v134;
                    do
                    {
                      v137 = (char *)v132[5];
                      v138 = (char *)v132[4];
                      if ( v137 - v138 > v136 )
                        v137 = &v138[v136];
                      v139 = *(unsigned __int64 **)(v130 + 32);
                      if ( v138 != v137 )
                      {
                        while ( *(_QWORD *)v138 >= *v139 )
                        {
                          if ( *(_QWORD *)v138 > *v139 )
                            goto LABEL_230;
                          v138 += 8;
                          ++v139;
                          if ( v137 == v138 )
                            goto LABEL_229;
                        }
LABEL_210:
                        v132 = (_QWORD *)v132[3];
                        continue;
                      }
LABEL_229:
                      if ( v133 != v139 )
                        goto LABEL_210;
LABEL_230:
                      v135 = v132;
                      v132 = (_QWORD *)v132[2];
                    }
                    while ( v132 );
                    if ( v412 != v135 )
                    {
                      v140 = (unsigned __int64 *)v135[4];
                      v141 = v135[5] - (_QWORD)v140;
                      v142 = (__int64)v134 + v141;
                      if ( v136 > v141 )
                        v133 = (unsigned __int64 *)((char *)v134 + v141);
                      if ( v134 == v133 )
                      {
LABEL_231:
                        if ( (unsigned __int64 *)v135[5] == v140 )
                        {
LABEL_232:
                          v151 = *((_DWORD *)v135 + 14);
                          switch ( v151 )
                          {
                            case 2:
                              v153 = sub_26F9080(*v131, v131[12], v8, v9, v134, v136 >> 3, "unique_member", 0xDu);
                              sub_26FAF90(
                                (__int64)v131,
                                v130 + 56,
                                byte_3F871B3,
                                0,
                                v135[8] != 0,
                                (unsigned __int64)v153);
                              break;
                            case 3:
                              *((_QWORD *)&v374 + 1) = 4;
                              *(_QWORD *)&v374 = "byte";
                              nf = sub_26F9120(v131, v8, v9, v134, v136 >> 3, v131[9], v374, *((_DWORD *)v135 + 18));
                              *((_QWORD *)&v375 + 1) = 3;
                              *(_QWORD *)&v375 = "bit";
                              v152 = (unsigned __int8 *)sub_26F9120(
                                                          v131,
                                                          v8,
                                                          v9,
                                                          *(unsigned __int64 **)(v130 + 32),
                                                          (__int64)(*(_QWORD *)(v130 + 40) - *(_QWORD *)(v130 + 32)) >> 3,
                                                          v131[7],
                                                          v375,
                                                          *((_DWORD *)v135 + 19));
                              sub_26FB610((__int64)v131, v130 + 56, byte_3F871B3, 0, nf, v152);
                              break;
                            case 1:
                              sub_26F9AB0((__int64)v131, v130 + 56, byte_3F871B3, 0, v135[8], v142);
                              break;
                          }
                        }
                      }
                      else
                      {
                        v143 = *(unsigned __int64 **)(v130 + 32);
                        while ( *v143 >= *v140 )
                        {
                          if ( *v143 > *v140 )
                            goto LABEL_232;
                          ++v143;
                          ++v140;
                          if ( v133 == v143 )
                            goto LABEL_231;
                        }
                      }
                    }
LABEL_220:
                    v130 = sub_220EEE0(v130);
                    if ( (__int64 *)v130 == v432 + 13 )
                    {
                      v7 = (__int64)v131;
                      v127 = v399;
                      break;
                    }
                  }
                }
                if ( *((_DWORD *)v127 + 10) == 2 )
                {
                  v144 = *(_QWORD ***)v7;
                  v413 = (__int64 *)sub_BCB120(**(_QWORD ***)v7);
                  sub_26F78E0((__int64 *)&s1, v8, v9, 0, 0, v145, "branch_funnel", 0xDu);
                  v146 = v464;
                  v147 = s1;
                  v458.m128i_i64[0] = (__int64)v459;
                  v458.m128i_i64[1] = 0;
                  v148 = sub_BCF480(v413, v459, 0, 0);
                  sub_BA8C10((__int64)v144, (__int64)v147, v146, v148, 0);
                  v150 = v149;
                  if ( (_QWORD *)v458.m128i_i64[0] != v459 )
                    _libc_free(v458.m128i_u64[0]);
                  if ( s1 != &v465 )
                    j_j___libc_free_0((unsigned __int64)s1);
                  LOBYTE(s1) = 0;
                  sub_2700A50(v7, (__int64)v406, v150, &s1);
                }
              }
            }
          }
        }
LABEL_18:
        v432 += 18;
      }
      while ( v397 != v432 );
    }
    sub_26F6120(a1);
    v16 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
    for ( i = *(_QWORD *)a1 + 8LL; i != v16; v16 = *(_QWORD *)(v16 + 8) )
    {
      v18 = v16 - 56;
      if ( !v16 )
        v18 = 0;
      sub_B98000(v18, 28);
    }
    goto LABEL_23;
  }
  if ( (_DWORD)v443 )
  {
    v53 = *(_QWORD *)(a1 + 40);
    if ( !v53 )
      goto LABEL_76;
    v54 = v442;
    s1 = 0;
    v464 = 0;
    v465.m128i_i64[0] = 0;
    v465.m128i_i32[2] = 0;
    v55 = &v442[7 * v444];
    if ( v442 != v55 )
    {
      while ( 1 )
      {
        v56 = *v54;
        v434 = v54;
        if ( *v54 != -8192 && v56 != -4096 )
          break;
        v54 += 7;
        if ( v55 == v54 )
          goto LABEL_74;
      }
      if ( v55 == v54 )
      {
LABEL_74:
        v57 = 0;
        v58 = 0;
        v405 = (void *)(v53 + 8);
        v409 = *(_QWORD *)(v53 + 24);
        if ( v53 + 8 != v409 )
          goto LABEL_349;
        goto LABEL_75;
      }
      while ( 1 )
      {
        if ( *(_BYTE *)v56 )
          goto LABEL_344;
        v217 = sub_B91420(v56);
        v219 = sub_B2F650(v217, v218);
        v220 = v219;
        if ( !v465.m128i_i32[2] )
          break;
        v6 = v464;
        v221 = 1;
        v222 = 0;
        v223 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v219) >> 31)
              ^ (484763065 * (_DWORD)v219))
             & (v465.m128i_i32[2] - 1);
        v224 = ((0xBF58476D1CE4E5B9LL * v219) >> 31) ^ (0xBF58476D1CE4E5B9LL * v219);
        v225 = (__int64 *)(v464 + 16 * v223);
        v226 = *v225;
        if ( v219 != *v225 )
        {
          while ( v226 != -1 )
          {
            if ( v226 == -2 && !v222 )
              v222 = v225;
            v223 = (v465.m128i_i32[2] - 1) & (unsigned int)(v223 + v221);
            v225 = (__int64 *)(v464 + 16 * v223);
            v226 = *v225;
            if ( v219 == *v225 )
              goto LABEL_339;
            ++v221;
          }
          if ( v222 )
            v225 = v222;
          s1 = (char *)s1 + 1;
          v364 = v465.m128i_i32[0] + 1;
          if ( 4 * (v465.m128i_i32[0] + 1) < (unsigned int)(3 * v465.m128i_i32[2]) )
          {
            if ( v465.m128i_i32[2] - v465.m128i_i32[1] - v364 <= (unsigned __int32)v465.m128i_i32[2] >> 3 )
            {
              sub_262D290((__int64)&s1, v465.m128i_i32[2]);
              if ( !v465.m128i_i32[2] )
              {
LABEL_728:
                ++v465.m128i_i32[0];
                BUG();
              }
              v369 = (v465.m128i_i32[2] - 1) & v224;
              v364 = v465.m128i_i32[0] + 1;
              v370 = (__int64 *)(v464 + 16LL * v369);
              v371 = *v370;
              v225 = v370;
              if ( v220 != *v370 )
              {
                v225 = 0;
                v6 = 1;
                while ( v371 != -1 )
                {
                  if ( !v225 && v371 == -2 )
                    v225 = v370;
                  v369 = (v465.m128i_i32[2] - 1) & (v6 + v369);
                  v370 = (__int64 *)(v464 + 16LL * v369);
                  v371 = *v370;
                  if ( v220 == *v370 )
                  {
                    v225 = (__int64 *)(v464 + 16LL * v369);
                    goto LABEL_638;
                  }
                  v6 = (unsigned int)(v6 + 1);
                }
                if ( !v225 )
                  v225 = v370;
              }
            }
            goto LABEL_638;
          }
LABEL_643:
          sub_262D290((__int64)&s1, 2 * v465.m128i_i32[2]);
          if ( !v465.m128i_i32[2] )
            goto LABEL_728;
          v364 = v465.m128i_i32[0] + 1;
          v365 = (v465.m128i_i32[2] - 1)
               & ((unsigned int)((0xBF58476D1CE4E5B9LL * v220) >> 31)
                ^ (484763065 * (_DWORD)v220));
          v225 = (__int64 *)(v464 + 16 * v365);
          v366 = *v225;
          if ( v220 != *v225 )
          {
            v367 = 0;
            v368 = 1;
            while ( v366 != -1 )
            {
              if ( !v367 && v366 == -2 )
                v367 = v225;
              v6 = (unsigned int)(v368 + 1);
              LODWORD(v365) = (v465.m128i_i32[2] - 1) & (v368 + v365);
              v225 = (__int64 *)(v464 + 16LL * (unsigned int)v365);
              v366 = *v225;
              if ( v220 == *v225 )
                goto LABEL_638;
              ++v368;
            }
            if ( v367 )
              v225 = v367;
          }
LABEL_638:
          v465.m128i_i32[0] = v364;
          if ( *v225 != -1 )
            --v465.m128i_i32[1];
          *v225 = v220;
          v228 = (unsigned __int64 *)(v225 + 1);
          v225[1] = 0;
LABEL_641:
          *v228 = v56 & 0xFFFFFFFFFFFFFFFDLL;
          goto LABEL_344;
        }
LABEL_339:
        v227 = v225[1];
        v228 = (unsigned __int64 *)(v225 + 1);
        v229 = v227 & 0xFFFFFFFFFFFFFFFCLL;
        if ( (v227 & 0xFFFFFFFFFFFFFFFCLL) == 0 )
          goto LABEL_641;
        if ( (v227 & 2) == 0 )
        {
          v360 = sub_22077B0(0x30u);
          if ( v360 )
          {
            *(_QWORD *)v360 = v360 + 16;
            *(_QWORD *)(v360 + 8) = 0x400000000LL;
          }
          v362 = v360 & 0xFFFFFFFFFFFFFFFCLL;
          v225[1] = v360 | 2;
          v363 = *(unsigned int *)((v360 & 0xFFFFFFFFFFFFFFFCLL) + 8);
          if ( v363 + 1 > (unsigned __int64)*(unsigned int *)(v362 + 12) )
          {
            sub_C8D5F0(v362, (const void *)(v362 + 16), v363 + 1, 8u, v361, v6);
            v363 = *(unsigned int *)(v362 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v362 + 8 * v363) = v229;
          ++*(_DWORD *)(v362 + 8);
          v223 = v225[1];
          v229 = v223 & 0xFFFFFFFFFFFFFFFCLL;
        }
        v230 = *(unsigned int *)(v229 + 8);
        if ( v230 + 1 > (unsigned __int64)*(unsigned int *)(v229 + 12) )
        {
          sub_C8D5F0(v229, (const void *)(v229 + 16), v230 + 1, 8u, v223, v6);
          v230 = *(unsigned int *)(v229 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v229 + 8 * v230) = v56;
        ++*(_DWORD *)(v229 + 8);
LABEL_344:
        v231 = v434 + 7;
        if ( v434 + 7 != v55 )
        {
          while ( 1 )
          {
            v56 = *v231;
            v434 = v231;
            if ( *v231 != -8192 && v56 != -4096 )
              break;
            v231 += 7;
            if ( v55 == v231 )
              goto LABEL_348;
          }
          if ( v231 != v55 )
            continue;
        }
LABEL_348:
        v58 = v465.m128i_u32[2];
        v57 = v464;
        v232 = *(_QWORD *)(a1 + 40);
        v405 = (void *)(v232 + 8);
        v409 = *(_QWORD *)(v232 + 24);
        if ( v409 != v232 + 8 )
          goto LABEL_349;
LABEL_485:
        if ( v58 )
        {
          v317 = (_QWORD *)v57;
          v318 = (_QWORD *)(v57 + 16LL * v58);
          do
          {
            if ( *v317 <= 0xFFFFFFFFFFFFFFFDLL )
            {
              v319 = v317[1];
              if ( v319 )
              {
                if ( (v319 & 2) != 0 )
                {
                  v320 = (unsigned __int64 *)(v319 & 0xFFFFFFFFFFFFFFFCLL);
                  v321 = (unsigned __int64)v320;
                  if ( v320 )
                  {
                    if ( (unsigned __int64 *)*v320 != v320 + 2 )
                      _libc_free(*v320);
                    j_j___libc_free_0(v321);
                  }
                }
              }
            }
            v317 += 2;
          }
          while ( v318 != v317 );
          v58 = v465.m128i_u32[2];
          v57 = v464;
        }
LABEL_75:
        sub_C7D6A0(v57, 16LL * v58, 8);
LABEL_76:
        LODWORD(v446) = 0;
        v447 = 0;
        v59 = *(unsigned int *)(a1 + 168);
        v60 = *(_BYTE ***)(a1 + 160);
        v448 = &v446;
        v449 = &v446;
        v450 = 0;
        v59 *= 144;
        v418 = (_BYTE **)((char *)v60 + v59);
        if ( v60 == (_BYTE **)((char *)v60 + v59) )
        {
          sub_26F6120(a1);
LABEL_170:
          v115 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
          for ( j = *(_QWORD *)a1 + 8LL; j != v115; v115 = *(_QWORD *)(v115 + 8) )
          {
            v117 = v115 - 56;
            if ( !v115 )
              v117 = 0;
            sub_B98000(v117, 28);
          }
          v118 = *(_QWORD ***)(a1 + 272);
          v119 = &v118[*(unsigned int *)(a1 + 280)];
          while ( v119 != v118 )
          {
            v120 = *v118++;
            sub_B43D60(v120);
          }
          sub_26F92C0(v447);
          goto LABEL_23;
        }
        v61 = v60;
        v62 = a1;
        v421 = 0;
        while ( 2 )
        {
          v458 = 0u;
          v459[0] = 0;
          if ( !v444 )
          {
            ++v441;
            goto LABEL_91;
          }
          v63 = 1;
          v64 = (v444 - 1) & (((unsigned int)*v61 >> 9) ^ ((unsigned int)*v61 >> 4));
          v65 = 0;
          v66 = &v442[7 * v64];
          v67 = (_BYTE *)*v66;
          if ( (_BYTE *)*v66 != *v61 )
          {
            while ( v67 != (_BYTE *)-4096LL )
            {
              if ( !v65 && v67 == (_BYTE *)-8192LL )
                v65 = v66;
              v354 = v63 + 1;
              v355 = (v444 - 1) & (v64 + v63);
              v64 = v355;
              v66 = &v442[7 * v355];
              v67 = (_BYTE *)*v66;
              if ( *v61 == (_BYTE *)*v66 )
                goto LABEL_79;
              v63 = v354;
            }
            if ( v65 )
              v66 = v65;
            ++v441;
            v73 = v443 + 1;
            if ( 4 * ((int)v443 + 1) >= 3 * v444 )
            {
LABEL_91:
              sub_2702540((__int64)&v441, 2 * v444);
              if ( !v444 )
                goto LABEL_727;
              v72 = (v444 - 1) & (((unsigned int)*v61 >> 9) ^ ((unsigned int)*v61 >> 4));
              v73 = v443 + 1;
              v66 = &v442[7 * v72];
              v74 = (_BYTE *)*v66;
              if ( *v61 != (_BYTE *)*v66 )
              {
                v372 = 0;
                v373 = 1;
                while ( v74 != (_BYTE *)-4096LL )
                {
                  if ( v74 == (_BYTE *)-8192LL && !v372 )
                    v372 = v66;
                  v72 = (v444 - 1) & (v373 + v72);
                  v66 = &v442[7 * v72];
                  v74 = (_BYTE *)*v66;
                  if ( *v61 == (_BYTE *)*v66 )
                    goto LABEL_93;
                  ++v373;
                }
                if ( v372 )
                  v66 = v372;
              }
            }
            else if ( v444 - HIDWORD(v443) - v73 <= v444 >> 3 )
            {
              sub_2702540((__int64)&v441, v444);
              if ( !v444 )
              {
LABEL_727:
                LODWORD(v443) = v443 + 1;
                BUG();
              }
              v356 = (v444 - 1) & (((unsigned int)*v61 >> 9) ^ ((unsigned int)*v61 >> 4));
              v357 = &v442[7 * v356];
              v358 = (_BYTE *)*v357;
              v73 = v443 + 1;
              v66 = v357;
              if ( (_BYTE *)*v357 != *v61 )
              {
                v66 = 0;
                v359 = 1;
                while ( v358 != (_BYTE *)-4096LL )
                {
                  if ( !v66 && v358 == (_BYTE *)-8192LL )
                    v66 = v357;
                  v356 = (v444 - 1) & (v359 + v356);
                  v357 = &v442[7 * v356];
                  v358 = (_BYTE *)*v357;
                  if ( *v61 == (_BYTE *)*v357 )
                  {
                    v66 = &v442[7 * v356];
                    goto LABEL_93;
                  }
                  ++v359;
                }
                if ( !v66 )
                  v66 = v357;
              }
            }
LABEL_93:
            LODWORD(v443) = v73;
            if ( *v66 != -4096 )
              --HIDWORD(v443);
            v75 = (__int64)*v61;
            *((_DWORD *)v66 + 4) = 0;
            v66[3] = 0;
            *v66 = v75;
            v66[4] = (__int64)(v66 + 2);
            v66[5] = (__int64)(v66 + 2);
            v66[6] = 0;
          }
LABEL_79:
          v68 = *(_QWORD **)(v62 + 40);
          v69 = (__int64)(v66 + 1);
          if ( !v68 || **v61 || !v66[6] )
          {
            v70 = (unsigned __int64)v61[1];
            v71 = 0;
LABEL_83:
            if ( sub_26FEE10((__int64 *)v62, (unsigned __int64 *)&v458, v69, v70, (__int64)v68) )
            {
              if ( !(unsigned __int8)sub_26FC690(
                                       v62,
                                       *(_QWORD *)(v62 + 40),
                                       v458.m128i_i64[0],
                                       (v458.m128i_i64[1] - v458.m128i_i64[0]) >> 5,
                                       v61 + 2,
                                       (__int64)v71) )
              {
                v421 |= sub_26FF400(
                          v62,
                          (__int64 *)v458.m128i_i64[0],
                          (v458.m128i_i64[1] - v458.m128i_i64[0]) >> 5,
                          (__int64)(v61 + 2),
                          v71,
                          a2,
                          v80,
                          (__int64)*v61,
                          (unsigned __int64)v61[1]);
                sub_2700B00(
                  (__int64 *)v62,
                  v458.m128i_i64[0],
                  (v458.m128i_i64[1] - v458.m128i_i64[0]) >> 5,
                  (__int64)(v61 + 2),
                  v71,
                  v183,
                  *v61,
                  (unsigned __int64)v61[1]);
              }
              if ( *(_BYTE *)(v62 + 104) || (unsigned __int8)sub_C92250() )
              {
                v425 = v458.m128i_i64[1];
                v81 = v458.m128i_i64[0];
                if ( v458.m128i_i64[0] != v458.m128i_i64[1] )
                {
                  v398 = v61;
                  v391 = v62;
LABEL_115:
                  while ( !*(_BYTE *)(v81 + 25) )
                  {
LABEL_114:
                    v81 += 32;
                    if ( v425 == v81 )
                      goto LABEL_145;
                  }
                  v410 = *(_QWORD *)v81;
                  v82 = (char *)sub_BD5D20(*(_QWORD *)v81);
                  s1 = &v465;
                  sub_26F6410((__int64 *)&s1, v82, (__int64)&v82[v83]);
                  if ( !v447 )
                  {
                    v85 = (unsigned __int64 *)&v446;
                    goto LABEL_135;
                  }
                  v84 = v447;
                  v85 = (unsigned __int64 *)&v446;
                  v86 = (__m128i *)s1;
                  v87 = v81;
                  v88 = v464;
                  while ( 1 )
                  {
                    while ( 1 )
                    {
                      v89 = v84[5];
                      v90 = v88;
                      if ( v89 <= v88 )
                        v90 = v84[5];
                      if ( !v90 )
                        break;
                      LODWORD(v91) = memcmp((const void *)v84[4], v86, v90);
                      if ( !(_DWORD)v91 )
                        break;
LABEL_125:
                      if ( (int)v91 >= 0 )
                        goto LABEL_126;
LABEL_118:
                      v84 = (unsigned __int64 *)v84[3];
                      if ( !v84 )
                        goto LABEL_127;
                    }
                    v91 = v89 - v88;
                    if ( (__int64)(v89 - v88) < 0x80000000LL )
                    {
                      if ( v91 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                        goto LABEL_118;
                      goto LABEL_125;
                    }
LABEL_126:
                    v85 = v84;
                    v84 = (unsigned __int64 *)v84[2];
                    if ( !v84 )
                    {
LABEL_127:
                      v92 = v88;
                      v81 = v87;
                      v93 = v86;
                      if ( v85 == (unsigned __int64 *)&v446 )
                        goto LABEL_135;
                      v94 = v85[5];
                      v95 = v92;
                      if ( v94 <= v92 )
                        v95 = v85[5];
                      if ( v95
                        && (ne = v85[5], LODWORD(v96) = memcmp(v86, (const void *)v85[4], v95), v94 = ne, (_DWORD)v96) )
                      {
LABEL_134:
                        if ( (int)v96 < 0 )
                          goto LABEL_135;
                      }
                      else
                      {
                        v96 = v92 - v94;
                        if ( (__int64)(v92 - v94) < 0x80000000LL )
                        {
                          if ( v96 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                            goto LABEL_134;
LABEL_135:
                          v97 = (__m128 *)sub_22077B0(0x48u);
                          v98 = v85;
                          v85 = (unsigned __int64 *)v97;
                          v97[2].m128_u64[0] = (unsigned __int64)&v97[3];
                          if ( s1 == &v465 )
                          {
                            a2 = _mm_load_si128(&v465);
                            v97[3] = (__m128)a2;
                          }
                          else
                          {
                            v97[2].m128_u64[0] = (unsigned __int64)s1;
                            v97[3].m128_u64[0] = v465.m128i_i64[0];
                          }
                          v99 = v464;
                          v97[4].m128_u64[0] = 0;
                          s2 = v97 + 3;
                          v97[2].m128_u64[1] = v99;
                          na = v99;
                          s1 = &v465;
                          v464 = 0;
                          v465.m128i_i8[0] = 0;
                          v100 = sub_27020D0(&v445, v98, (__int64)&v97[2]);
                          v102 = (unsigned __int64 *)v100;
                          v103 = v101;
                          if ( v101 )
                          {
                            if ( &v446 == v101 || (v104 = na, v100) )
                            {
LABEL_140:
                              LOBYTE(v105) = 1;
                              goto LABEL_141;
                            }
                            v204 = v101[5];
                            v203 = v204;
                            if ( na <= v204 )
                              v204 = na;
                            if ( v204
                              && (s2d = (void *)na,
                                  ng = (size_t)v103,
                                  v205 = memcmp((const void *)v85[4], (const void *)v103[4], v204),
                                  v103 = (_QWORD *)ng,
                                  v104 = (signed __int64)s2d,
                                  (v206 = v205) != 0) )
                            {
LABEL_310:
                              v105 = v206 >> 31;
                            }
                            else
                            {
                              v207 = v104 - v203;
                              LOBYTE(v105) = 0;
                              if ( v207 < 0x80000000LL )
                              {
                                if ( v207 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                                  goto LABEL_140;
                                v206 = v207;
                                goto LABEL_310;
                              }
                            }
LABEL_141:
                            sub_220F040(v105, (__int64)v85, v103, &v446);
                            ++v450;
                          }
                          else
                          {
                            v208 = v85[4];
                            if ( s2 != (__m128 *)v208 )
                              j_j___libc_free_0(v208);
                            v209 = (unsigned __int64)v85;
                            v85 = v102;
                            j_j___libc_free_0(v209);
                          }
                          v93 = (__m128i *)s1;
                        }
                      }
                      v85[8] = v410;
                      if ( v93 == &v465 )
                        goto LABEL_114;
                      v81 += 32;
                      j_j___libc_free_0((unsigned __int64)v93);
                      if ( v425 == v81 )
                      {
LABEL_145:
                        v61 = v398;
                        v62 = v391;
                        break;
                      }
                      goto LABEL_115;
                    }
                  }
                }
              }
            }
            if ( !*(_QWORD *)(v62 + 40) || **v61 )
              goto LABEL_86;
            v159 = sub_B91420((__int64)*v61);
            v161 = sub_B2F650(v159, v160);
            v162 = (__int64 *)v61[7];
            v163 = (__int64 *)v61[6];
            v426 = (void *)v161;
            while ( v162 != v163 )
            {
              v164 = *v163++;
              sub_26FD970(v164, (__int64)v426);
            }
            v165 = (__int64)v61[15];
            v416 = v61 + 13;
            if ( (_BYTE **)v165 == v61 + 13 )
            {
LABEL_86:
              if ( v458.m128i_i64[0] )
                j_j___libc_free_0(v458.m128i_u64[0]);
              v61 += 18;
              if ( v418 == v61 )
              {
                if ( *(_BYTE *)(a1 + 104) )
                {
                  for ( k = (__int64)v448; (__int64 *)k != &v446; k = sub_220EEE0(k) )
                  {
                    v107 = *(_BYTE **)(k + 64);
                    if ( *v107 )
                    {
                      if ( *v107 != 1 )
                        BUG();
                      v107 = (_BYTE *)*((_QWORD *)v107 - 4);
                      if ( *v107 )
                        v107 = 0;
                    }
                    v108 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, _BYTE *))(a1 + 112))(
                                        *(_QWORD *)(a1 + 120),
                                        v107);
                    sub_B17560((__int64)&s1, (__int64)"wholeprogramdevirt", (__int64)"Devirtualized", 13, (__int64)v107);
                    sub_B18290((__int64)&s1, "devirtualized ", 0xEu);
                    sub_B16430((__int64)&v451, "FunctionName", 0xCu, *(_BYTE **)(k + 32), *(_QWORD *)(k + 40));
                    v458.m128i_i64[0] = (__int64)v459;
                    sub_26F69E0(v458.m128i_i64, v451, (__int64)&v451[v452]);
                    v460 = v461;
                    sub_26F69E0((__int64 *)&v460, v454, (__int64)v454 + v455);
                    v462 = _mm_load_si128(&v457);
                    sub_B180C0((__int64)&s1, (unsigned __int64)&v458);
                    if ( v460 != v461 )
                      j_j___libc_free_0((unsigned __int64)v460);
                    if ( (_QWORD *)v458.m128i_i64[0] != v459 )
                      j_j___libc_free_0(v458.m128i_u64[0]);
                    sub_1049740(v108, (__int64)&s1);
                    if ( v454 != &v456 )
                      j_j___libc_free_0((unsigned __int64)v454);
                    if ( v451 != (_BYTE *)v453 )
                      j_j___libc_free_0((unsigned __int64)v451);
                    v109 = v466;
                    s1 = &unk_49D9D40;
                    v110 = &v466[10 * v467];
                    if ( v466 != v110 )
                    {
                      do
                      {
                        v110 -= 10;
                        v111 = v110[4];
                        if ( (unsigned __int64 *)v111 != v110 + 6 )
                          j_j___libc_free_0(v111);
                        if ( (unsigned __int64 *)*v110 != v110 + 2 )
                          j_j___libc_free_0(*v110);
                      }
                      while ( v109 != v110 );
                      v110 = v466;
                    }
                    if ( v110 != (unsigned __int64 *)&v468 )
                      _libc_free((unsigned __int64)v110);
                  }
                }
                sub_26F6120(a1);
                if ( v421 )
                {
                  v112 = v438;
                  v113 = v439;
                  while ( v113 != v112 )
                  {
                    v114 = v112;
                    v112 += 14;
                    sub_26FCE50((__int64 *)a1, v114);
                  }
                }
                goto LABEL_170;
              }
              continue;
            }
            v407 = v61;
            v401 = v62;
            while ( 1 )
            {
              v166 = *(__int64 **)(v165 + 96);
              v167 = *(__int64 **)(v165 + 88);
              if ( v167 != v166 )
                break;
LABEL_280:
              v165 = sub_220EEE0(v165);
              if ( (_BYTE **)v165 == v416 )
              {
                v61 = v407;
                v62 = v401;
                goto LABEL_86;
              }
            }
            while ( 1 )
            {
              v169 = *v167;
              s1 = v426;
              v170 = *(_QWORD *)(v169 + 80);
              if ( v170 )
                goto LABEL_248;
              v171 = (void *)sub_22077B0(0x78u);
              v170 = (__int64)v171;
              if ( v171 )
                memset(v171, 0, 0x78u);
              v172 = *(unsigned __int64 **)(v169 + 80);
              *(_QWORD *)(v169 + 80) = v171;
              if ( !v172 )
              {
LABEL_248:
                v168 = *(_BYTE **)(v170 + 8);
                if ( v168 == *(_BYTE **)(v170 + 16) )
                  goto LABEL_279;
              }
              else
              {
                v173 = v172[12];
                if ( v172[13] != v173 )
                {
                  v393 = v172;
                  v174 = v172[13];
                  nb = (size_t)v167;
                  v175 = v173;
                  do
                  {
                    v176 = *(_QWORD *)(v175 + 16);
                    if ( v176 )
                      j_j___libc_free_0(v176);
                    v175 += 40LL;
                  }
                  while ( v174 != v175 );
                  v172 = v393;
                  v167 = (__int64 *)nb;
                  v173 = v393[12];
                }
                if ( v173 )
                  j_j___libc_free_0(v173);
                v177 = v172[9];
                if ( v172[10] != v177 )
                {
                  v394 = v172;
                  v178 = v172[10];
                  nc = (size_t)v167;
                  v179 = v177;
                  do
                  {
                    v180 = *(_QWORD *)(v179 + 16);
                    if ( v180 )
                      j_j___libc_free_0(v180);
                    v179 += 40LL;
                  }
                  while ( v178 != v179 );
                  v172 = v394;
                  v167 = (__int64 *)nc;
                  v177 = v394[9];
                }
                if ( v177 )
                  j_j___libc_free_0(v177);
                v181 = v172[6];
                if ( v181 )
                  j_j___libc_free_0(v181);
                v182 = v172[3];
                if ( v182 )
                  j_j___libc_free_0(v182);
                if ( *v172 )
                  j_j___libc_free_0(*v172);
                j_j___libc_free_0((unsigned __int64)v172);
                v170 = *(_QWORD *)(v169 + 80);
                v168 = *(_BYTE **)(v170 + 8);
                if ( v168 == *(_BYTE **)(v170 + 16) )
                {
LABEL_279:
                  ++v167;
                  sub_9CA200(v170, v168, &s1);
                  if ( v166 == v167 )
                    goto LABEL_280;
                  continue;
                }
              }
              if ( v168 )
              {
                *(_QWORD *)v168 = s1;
                v168 = *(_BYTE **)(v170 + 8);
              }
              ++v167;
              *(_QWORD *)(v170 + 8) = v168 + 8;
              if ( v166 == v167 )
                goto LABEL_280;
            }
          }
          break;
        }
        v195 = (const void *)sub_B91420((__int64)*v61);
        v197 = sub_9CA790(v68, v195, v196);
        v198 = *(unsigned __int64 **)(v197 + 56);
        v199 = v197;
        v200 = (unsigned __int64 *)(v197 + 48);
        if ( !v198 )
          goto LABEL_301;
        v70 = (unsigned __int64)v61[1];
        v201 = (unsigned __int64 *)(v197 + 48);
        while ( 1 )
        {
          v202 = (unsigned __int64 *)v198[3];
          if ( v198[4] >= v70 )
          {
            v202 = (unsigned __int64 *)v198[2];
            v201 = v198;
          }
          if ( !v202 )
            break;
          v198 = v202;
        }
        if ( v201 == v200 )
        {
LABEL_301:
          s1 = v61 + 1;
          v201 = sub_26FFC30((_QWORD *)(v199 + 40), v200, (unsigned __int64 **)&s1);
          v70 = (unsigned __int64)v61[1];
        }
        else if ( v70 < v201[4] )
        {
          v200 = v201;
          goto LABEL_301;
        }
        v68 = *(_QWORD **)(v62 + 40);
        v71 = v201 + 5;
        goto LABEL_83;
      }
      s1 = (char *)s1 + 1;
      goto LABEL_643;
    }
    v57 = 0;
    v58 = 0;
    v405 = (void *)(v53 + 8);
    v409 = *(_QWORD *)(v53 + 24);
    if ( v409 == v53 + 8 )
      goto LABEL_75;
LABEL_349:
    while ( 1 )
    {
      nd = *(_QWORD *)(v409 + 64);
      if ( *(_QWORD *)(v409 + 56) != nd )
        break;
LABEL_484:
      v424 = v58;
      v431 = v57;
      v316 = sub_220EEE0(v409);
      v57 = v431;
      v409 = v316;
      v58 = v424;
      if ( (void *)v316 == v405 )
        goto LABEL_485;
    }
    v402 = *(__int64 **)(v409 + 56);
    v233 = v58;
LABEL_351:
    v234 = *v402;
    if ( *(_DWORD *)(*v402 + 8) != 1 )
      goto LABEL_482;
    v235 = *(_QWORD **)(v234 + 80);
    if ( !v235 )
      goto LABEL_482;
    v395 = (__int64 *)v235[4];
    if ( (__int64 *)v235[3] == v395 )
      goto LABEL_383;
    v422 = (__int64 *)v235[3];
    while ( 1 )
    {
      v236 = *v422;
      v237 = v422[1];
      if ( !v233 )
      {
        s1 = (char *)s1 + 1;
        goto LABEL_519;
      }
      v238 = v233 - 1;
      v239 = 1;
      v240 = ((0xBF58476D1CE4E5B9LL * v236) >> 31) ^ (0xBF58476D1CE4E5B9LL * v236);
      v241 = 0;
      LODWORD(v242) = v240 & (v233 - 1);
      v243 = (__int64 *)(v57 + 16LL * (unsigned int)v242);
      v244 = *v243;
      if ( v236 != *v243 )
      {
        while ( v244 != -1 )
        {
          if ( !v241 && v244 == -2 )
            v241 = v243;
          v6 = (unsigned int)(v239 + 1);
          v242 = (unsigned int)v238 & ((_DWORD)v242 + v239);
          v243 = (__int64 *)(v57 + 16 * v242);
          v244 = *v243;
          if ( v236 == *v243 )
            goto LABEL_357;
          ++v239;
        }
        if ( v241 )
          v243 = v241;
        s1 = (char *)s1 + 1;
        v252 = v465.m128i_i32[0] + 1;
        if ( 4 * (v465.m128i_i32[0] + 1) < 3 * v233 )
        {
          if ( v233 - (v252 + v465.m128i_i32[1]) <= v233 >> 3 )
          {
            sub_262D290((__int64)&s1, v233);
            if ( !v465.m128i_i32[2] )
              goto LABEL_731;
            v322 = (v465.m128i_i32[2] - 1) & v240;
            v252 = v465.m128i_i32[0] + 1;
            v323 = (__int64 *)(v464 + 16LL * v322);
            v6 = *v323;
            v243 = v323;
            if ( v236 != *v323 )
            {
              v324 = 1;
              v243 = 0;
              while ( v6 != -1 )
              {
                if ( v6 == -2 && !v243 )
                  v243 = v323;
                v322 = (v465.m128i_i32[2] - 1) & (v324 + v322);
                v323 = (__int64 *)(v464 + 16LL * v322);
                v6 = *v323;
                if ( v236 == *v323 )
                {
                  v243 = (__int64 *)(v464 + 16LL * v322);
                  goto LABEL_378;
                }
                ++v324;
              }
              if ( !v243 )
                v243 = v323;
            }
          }
LABEL_378:
          v465.m128i_i32[0] = v252;
          if ( *v243 != -1 )
            --v465.m128i_i32[1];
          *v243 = v236;
          v243[1] = 0;
          goto LABEL_381;
        }
LABEL_519:
        sub_262D290((__int64)&s1, 2 * v233);
        if ( !v465.m128i_i32[2] )
        {
LABEL_731:
          ++v465.m128i_i32[0];
          BUG();
        }
        v252 = v465.m128i_i32[0] + 1;
        v325 = (v465.m128i_i32[2] - 1) & (((0xBF58476D1CE4E5B9LL * v236) >> 31) ^ (484763065 * v236));
        v243 = (__int64 *)(v464 + 16LL * v325);
        v326 = *v243;
        if ( v236 != *v243 )
        {
          v327 = 1;
          v328 = 0;
          while ( v326 != -1 )
          {
            if ( !v328 && v326 == -2 )
              v328 = v243;
            v6 = (unsigned int)(v327 + 1);
            v325 = (v465.m128i_i32[2] - 1) & (v327 + v325);
            v243 = (__int64 *)(v464 + 16LL * v325);
            v326 = *v243;
            if ( v236 == *v243 )
              goto LABEL_378;
            ++v327;
          }
          if ( v328 )
            v243 = v328;
        }
        goto LABEL_378;
      }
LABEL_357:
      v245 = v243[1] & 0xFFFFFFFFFFFFFFFCLL;
      v246 = v243[1] & 2;
      if ( (v243[1] & 2) != 0 )
        break;
      v247 = (__int64)(v243 + 1);
      if ( v245 )
      {
        v427 = (__int64)(v243 + 2);
        goto LABEL_360;
      }
LABEL_381:
      v422 += 2;
      v233 = v465.m128i_u32[2];
      v57 = v464;
      if ( v395 == v422 )
      {
        v235 = *(_QWORD **)(v234 + 80);
        if ( v235 )
        {
LABEL_383:
          v396 = (__int64 *)v235[7];
          if ( (__int64 *)v235[6] == v396 )
            goto LABEL_413;
          v423 = (__int64 *)v235[6];
          while ( 2 )
          {
            v253 = *v423;
            v254 = v423[1];
            if ( !v233 )
            {
              s1 = (char *)s1 + 1;
              goto LABEL_586;
            }
            v255 = v233 - 1;
            v256 = 1;
            v257 = ((0xBF58476D1CE4E5B9LL * v253) >> 31) ^ (0xBF58476D1CE4E5B9LL * v253);
            v258 = 0;
            LODWORD(v259) = v257 & (v233 - 1);
            v260 = (__int64 *)(v57 + 16LL * (unsigned int)v259);
            v261 = *v260;
            if ( *v260 != v253 )
            {
              while ( v261 != -1 )
              {
                if ( v261 == -2 && !v258 )
                  v258 = v260;
                v6 = (unsigned int)(v256 + 1);
                v259 = (unsigned int)v255 & ((_DWORD)v259 + v256);
                v260 = (__int64 *)(v57 + 16 * v259);
                v261 = *v260;
                if ( *v260 == v253 )
                  goto LABEL_387;
                ++v256;
              }
              if ( v258 )
                v260 = v258;
              s1 = (char *)s1 + 1;
              v269 = v465.m128i_i32[0] + 1;
              if ( 4 * (v465.m128i_i32[0] + 1) >= 3 * v233 )
              {
LABEL_586:
                sub_262D290((__int64)&s1, 2 * v233);
                if ( !v465.m128i_i32[2] )
                {
LABEL_733:
                  ++v465.m128i_i32[0];
                  BUG();
                }
                v269 = v465.m128i_i32[0] + 1;
                v348 = (v465.m128i_i32[2] - 1) & (((0xBF58476D1CE4E5B9LL * v253) >> 31) ^ (484763065 * v253));
                v260 = (__int64 *)(v464 + 16LL * v348);
                v349 = *v260;
                if ( *v260 != v253 )
                {
                  v350 = 1;
                  v351 = 0;
                  while ( v349 != -1 )
                  {
                    if ( !v351 && v349 == -2 )
                      v351 = v260;
                    v6 = (unsigned int)(v350 + 1);
                    v348 = (v465.m128i_i32[2] - 1) & (v350 + v348);
                    v260 = (__int64 *)(v464 + 16LL * v348);
                    v349 = *v260;
                    if ( *v260 == v253 )
                      goto LABEL_408;
                    ++v350;
                  }
                  if ( v351 )
                    v260 = v351;
                }
              }
              else if ( v233 - (v269 + v465.m128i_i32[1]) <= v233 >> 3 )
              {
                sub_262D290((__int64)&s1, v233);
                if ( !v465.m128i_i32[2] )
                  goto LABEL_733;
                v345 = (v465.m128i_i32[2] - 1) & v257;
                v269 = v465.m128i_i32[0] + 1;
                v346 = (__int64 *)(v464 + 16LL * v345);
                v6 = *v346;
                v260 = v346;
                if ( *v346 != v253 )
                {
                  v347 = 1;
                  v260 = 0;
                  while ( v6 != -1 )
                  {
                    if ( !v260 && v6 == -2 )
                      v260 = v346;
                    v345 = (v465.m128i_i32[2] - 1) & (v347 + v345);
                    v346 = (__int64 *)(v464 + 16LL * v345);
                    v6 = *v346;
                    if ( *v346 == v253 )
                    {
                      v260 = (__int64 *)(v464 + 16LL * v345);
                      goto LABEL_408;
                    }
                    ++v347;
                  }
                  if ( !v260 )
                    v260 = v346;
                }
              }
LABEL_408:
              v465.m128i_i32[0] = v269;
              if ( *v260 != -1 )
                --v465.m128i_i32[1];
              *v260 = v253;
              v260[1] = 0;
              goto LABEL_411;
            }
LABEL_387:
            v262 = v260[1] & 0xFFFFFFFFFFFFFFFCLL;
            v263 = v260[1] & 2;
            if ( (v260[1] & 2) != 0 )
            {
              v264 = *(_QWORD *)v262;
              v428 = *(_QWORD *)v262 + 8LL * *(unsigned int *)(v262 + 8);
              goto LABEL_390;
            }
            v264 = (__int64)(v260 + 1);
            if ( v262 )
            {
              v428 = (__int64)(v260 + 2);
LABEL_390:
              if ( v428 != v264 )
              {
                v265 = (__int64 *)v264;
                do
                {
                  v267 = *v265;
                  v458.m128i_i64[1] = v254;
                  v458.m128i_i64[0] = v267;
                  v268 = sub_26FA230(a1 + 128, &v458, v263, v264, v255, v6);
                  v451 = (_BYTE *)v234;
                  v266 = *(_BYTE **)(v268 + 40);
                  if ( v266 == *(_BYTE **)(v268 + 48) )
                  {
                    v420 = v268;
                    sub_26FDF70(v268 + 32, v266, &v451);
                    v268 = v420;
                  }
                  else
                  {
                    if ( v266 )
                    {
                      *(_QWORD *)v266 = v234;
                      v266 = *(_BYTE **)(v268 + 40);
                    }
                    *(_QWORD *)(v268 + 40) = v266 + 8;
                  }
                  *(_BYTE *)(v268 + 24) = 0;
                  ++v265;
                }
                while ( (__int64 *)v428 != v265 );
              }
            }
LABEL_411:
            v423 += 2;
            v233 = v465.m128i_u32[2];
            v57 = v464;
            if ( v396 != v423 )
              continue;
            break;
          }
          v235 = *(_QWORD **)(v234 + 80);
          if ( v235 )
          {
LABEL_413:
            s2a = (void *)v235[10];
            if ( (void *)v235[9] == s2a )
              goto LABEL_448;
            v270 = v235[9];
            while ( 2 )
            {
              if ( !v233 )
              {
                s1 = (char *)s1 + 1;
                goto LABEL_556;
              }
              v271 = *(_QWORD *)v270;
              v6 = v233 - 1;
              v272 = 1;
              v273 = 0;
              v274 = v6 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v270) >> 31) ^ (484763065 * *(_DWORD *)v270));
              v275 = (__int64 *)(v57 + 16LL * v274);
              v276 = *v275;
              if ( *v275 != *(_QWORD *)v270 )
              {
                while ( v276 != -1 )
                {
                  if ( !v273 && v276 == -2 )
                    v273 = v275;
                  v274 = v6 & (v272 + v274);
                  v275 = (__int64 *)(v57 + 16LL * v274);
                  v276 = *v275;
                  if ( v271 == *v275 )
                    goto LABEL_417;
                  ++v272;
                }
                if ( v273 )
                  v275 = v273;
                s1 = (char *)s1 + 1;
                v6 = (unsigned int)(v465.m128i_i32[0] + 1);
                if ( 4 * (int)v6 >= 3 * v233 )
                {
LABEL_556:
                  sub_262D290((__int64)&s1, 2 * v233);
                  if ( v465.m128i_i32[2] )
                  {
                    v6 = (unsigned int)(v465.m128i_i32[0] + 1);
                    v338 = (v465.m128i_i32[2] - 1)
                         & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v270) >> 31)
                          ^ (484763065 * *(_DWORD *)v270));
                    v275 = (__int64 *)(v464 + 16LL * v338);
                    v339 = *v275;
                    if ( *(_QWORD *)v270 != *v275 )
                    {
                      v352 = 1;
                      v353 = 0;
                      while ( v339 != -1 )
                      {
                        if ( v339 == -2 && !v353 )
                          v353 = v275;
                        v338 = (v465.m128i_i32[2] - 1) & (v352 + v338);
                        v275 = (__int64 *)(v464 + 16LL * v338);
                        v339 = *v275;
                        if ( *(_QWORD *)v270 == *v275 )
                          goto LABEL_558;
                        ++v352;
                      }
                      if ( v353 )
                        v275 = v353;
                    }
                    goto LABEL_558;
                  }
                }
                else
                {
                  if ( v233 - ((_DWORD)v6 + v465.m128i_i32[1]) > v233 >> 3 )
                    goto LABEL_558;
                  sub_262D290((__int64)&s1, v233);
                  if ( v465.m128i_i32[2] )
                  {
                    v341 = (v465.m128i_i32[2] - 1)
                         & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v270) >> 31)
                          ^ (484763065 * *(_DWORD *)v270));
                    v6 = (unsigned int)(v465.m128i_i32[0] + 1);
                    v342 = (__int64 *)(v464 + 16LL * v341);
                    v343 = *v342;
                    v275 = v342;
                    if ( *v342 != *(_QWORD *)v270 )
                    {
                      v344 = 1;
                      v275 = 0;
                      while ( 1 )
                      {
                        if ( v343 == -1 )
                        {
                          if ( !v275 )
                            v275 = v342;
                          goto LABEL_558;
                        }
                        if ( !v275 && v343 == -2 )
                          v275 = v342;
                        v341 = (v465.m128i_i32[2] - 1) & (v344 + v341);
                        v342 = (__int64 *)(v464 + 16LL * v341);
                        v343 = *v342;
                        if ( *(_QWORD *)v270 == *v342 )
                          break;
                        ++v344;
                      }
                      v275 = (__int64 *)(v464 + 16LL * v341);
                    }
LABEL_558:
                    v465.m128i_i32[0] = v6;
                    if ( *v275 != -1 )
                      --v465.m128i_i32[1];
                    v340 = *(_QWORD *)v270;
                    v275[1] = 0;
                    *v275 = v340;
                    goto LABEL_446;
                  }
                }
                ++v465.m128i_i32[0];
                BUG();
              }
LABEL_417:
              v277 = v275[1] & 0xFFFFFFFFFFFFFFFCLL;
              v278 = v275[1] & 2;
              if ( (v275[1] & 2) != 0 )
              {
                v279 = *(__int64 **)v277;
                v429 = *(_QWORD *)v277 + 8LL * *(unsigned int *)(v277 + 8);
              }
              else
              {
                v279 = v275 + 1;
                if ( !v277 )
                  goto LABEL_446;
                v429 = (__int64)(v275 + 2);
              }
              while ( (__int64 *)v429 != v279 )
              {
                v458.m128i_i64[0] = *v279;
                v458.m128i_i64[1] = *(_QWORD *)(v270 + 8);
                v280 = sub_26FA230(a1 + 128, &v458, v277, v278, v271, v6);
                v281 = *(_QWORD **)(v280 + 96);
                v282 = v280;
                v283 = v280 + 88;
                if ( !v281 )
                {
                  v286 = v280 + 88;
                  goto LABEL_440;
                }
                v284 = *(__int64 **)(v270 + 24);
                v285 = *(__int64 **)(v270 + 16);
                v286 = v280 + 88;
                v287 = (char *)v284 - (char *)v285;
                do
                {
                  v288 = (char *)v281[5];
                  v289 = (char *)v281[4];
                  v6 = v288 - v289;
                  if ( v288 - v289 > v287 )
                    v288 = &v289[v287];
                  v290 = *(__int64 **)(v270 + 16);
                  if ( v289 != v288 )
                  {
                    while ( 1 )
                    {
                      v6 = *v290;
                      if ( *(_QWORD *)v289 < (unsigned __int64)*v290 )
                        break;
                      if ( *(_QWORD *)v289 > (unsigned __int64)*v290 )
                        goto LABEL_496;
                      v289 += 8;
                      ++v290;
                      if ( v288 == v289 )
                        goto LABEL_495;
                    }
LABEL_430:
                    v281 = (_QWORD *)v281[3];
                    continue;
                  }
LABEL_495:
                  if ( v284 != v290 )
                    goto LABEL_430;
LABEL_496:
                  v286 = (unsigned __int64)v281;
                  v281 = (_QWORD *)v281[2];
                }
                while ( v281 );
                if ( v283 == v286 )
                  goto LABEL_440;
                v278 = *(_QWORD *)(v286 + 40);
                v291 = *(_QWORD **)(v286 + 32);
                v277 = v278 - (_QWORD)v291;
                v271 = (__int64)v285 + v278 - (_QWORD)v291;
                if ( v287 > v278 - (__int64)v291 )
                  v284 = (__int64 *)((char *)v285 + v277);
                if ( v285 == v284 )
                {
LABEL_499:
                  if ( (_QWORD *)v278 != v291 )
                    goto LABEL_440;
                }
                else
                {
                  while ( (unsigned __int64)*v285 >= *v291 )
                  {
                    if ( (unsigned __int64)*v285 > *v291 )
                      goto LABEL_441;
                    ++v285;
                    ++v291;
                    if ( v284 == v285 )
                      goto LABEL_499;
                  }
LABEL_440:
                  v451 = (_BYTE *)(v270 + 16);
                  v286 = sub_26F9D50((_QWORD *)(v282 + 80), (_QWORD *)v286, (__int64 *)&v451);
                }
LABEL_441:
                v451 = (_BYTE *)v234;
                v292 = *(_BYTE **)(v286 + 120);
                if ( v292 == *(_BYTE **)(v286 + 128) )
                {
                  sub_26FDF70(v286 + 112, v292, &v451);
                }
                else
                {
                  if ( v292 )
                  {
                    *(_QWORD *)v292 = v234;
                    v292 = *(_BYTE **)(v286 + 120);
                  }
                  *(_QWORD *)(v286 + 120) = v292 + 8;
                }
                ++v279;
                *(_WORD *)(v286 + 80) = 256;
              }
LABEL_446:
              v233 = v465.m128i_u32[2];
              v57 = v464;
              v270 += 40;
              if ( s2a != (void *)v270 )
                continue;
              break;
            }
            v235 = *(_QWORD **)(v234 + 80);
            if ( v235 )
            {
LABEL_448:
              s2b = (void *)v235[13];
              if ( (void *)v235[12] != s2b )
              {
                v293 = v235[12];
                while ( v233 )
                {
                  v294 = *(_QWORD *)v293;
                  v6 = v233 - 1;
                  v295 = 1;
                  v296 = 0;
                  v297 = v6 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v293) >> 31) ^ (484763065 * *(_DWORD *)v293));
                  v298 = (__int64 *)(v57 + 16LL * v297);
                  v299 = *v298;
                  if ( *(_QWORD *)v293 != *v298 )
                  {
                    while ( v299 != -1 )
                    {
                      if ( !v296 && v299 == -2 )
                        v296 = v298;
                      v297 = v6 & (v295 + v297);
                      v298 = (__int64 *)(v57 + 16LL * v297);
                      v299 = *v298;
                      if ( v294 == *v298 )
                        goto LABEL_452;
                      ++v295;
                    }
                    if ( v296 )
                      v298 = v296;
                    s1 = (char *)s1 + 1;
                    v329 = v465.m128i_i32[0] + 1;
                    if ( 4 * (v465.m128i_i32[0] + 1) < 3 * v233 )
                    {
                      if ( v233 - (v329 + v465.m128i_i32[1]) > v233 >> 3 )
                      {
LABEL_536:
                        v465.m128i_i32[0] = v329;
                        if ( *v298 != -1 )
                          --v465.m128i_i32[1];
                        v330 = *(_QWORD *)v293;
                        v298[1] = 0;
                        *v298 = v330;
                        goto LABEL_481;
                      }
                      sub_262D290((__int64)&s1, v233);
                      if ( v465.m128i_i32[2] )
                      {
                        v6 = *(_QWORD *)v293;
                        v331 = 0;
                        v329 = v465.m128i_i32[0] + 1;
                        v332 = 1;
                        v333 = (v465.m128i_i32[2] - 1)
                             & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v293) >> 31)
                              ^ (484763065 * *(_DWORD *)v293));
                        v298 = (__int64 *)(v464 + 16LL * v333);
                        v334 = *v298;
                        if ( *v298 == *(_QWORD *)v293 )
                          goto LABEL_536;
                        while ( v334 != -1 )
                        {
                          if ( v334 == -2 && !v331 )
                            v331 = v298;
                          v333 = (v465.m128i_i32[2] - 1) & (v332 + v333);
                          v298 = (__int64 *)(v464 + 16LL * v333);
                          v334 = *v298;
                          if ( v6 == *v298 )
                            goto LABEL_536;
                          ++v332;
                        }
LABEL_542:
                        if ( v331 )
                          v298 = v331;
                        goto LABEL_536;
                      }
                      goto LABEL_730;
                    }
LABEL_546:
                    sub_262D290((__int64)&s1, 2 * v233);
                    if ( v465.m128i_i32[2] )
                    {
                      v6 = *(_QWORD *)v293;
                      v329 = v465.m128i_i32[0] + 1;
                      v335 = (v465.m128i_i32[2] - 1)
                           & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v293) >> 31)
                            ^ (484763065 * *(_DWORD *)v293));
                      v298 = (__int64 *)(v464 + 16LL * v335);
                      v336 = *v298;
                      if ( *(_QWORD *)v293 == *v298 )
                        goto LABEL_536;
                      v337 = 1;
                      v331 = 0;
                      while ( v336 != -1 )
                      {
                        if ( !v331 && v336 == -2 )
                          v331 = v298;
                        v335 = (v465.m128i_i32[2] - 1) & (v337 + v335);
                        v298 = (__int64 *)(v464 + 16LL * v335);
                        v336 = *v298;
                        if ( v6 == *v298 )
                          goto LABEL_536;
                        ++v337;
                      }
                      goto LABEL_542;
                    }
LABEL_730:
                    ++v465.m128i_i32[0];
                    BUG();
                  }
LABEL_452:
                  v300 = v298[1] & 0xFFFFFFFFFFFFFFFCLL;
                  v301 = v298[1] & 2;
                  if ( (v298[1] & 2) != 0 )
                  {
                    v302 = *(__int64 **)v300;
                    v430 = *(_QWORD *)v300 + 8LL * *(unsigned int *)(v300 + 8);
                  }
                  else
                  {
                    v302 = v298 + 1;
                    if ( !v300 )
                      goto LABEL_481;
                    v430 = (__int64)(v298 + 2);
                  }
                  while ( (__int64 *)v430 != v302 )
                  {
                    v458.m128i_i64[0] = *v302;
                    v458.m128i_i64[1] = *(_QWORD *)(v293 + 8);
                    v303 = sub_26FA230(a1 + 128, &v458, v300, v301, v294, v6);
                    v304 = *(_QWORD **)(v303 + 96);
                    v305 = v303;
                    v306 = v303 + 88;
                    if ( !v304 )
                    {
                      v309 = v303 + 88;
                      goto LABEL_475;
                    }
                    v307 = *(__int64 **)(v293 + 24);
                    v308 = *(__int64 **)(v293 + 16);
                    v309 = v303 + 88;
                    v310 = (char *)v307 - (char *)v308;
                    do
                    {
                      v311 = (char *)v304[5];
                      v312 = (char *)v304[4];
                      v6 = v311 - v312;
                      if ( v311 - v312 > v310 )
                        v311 = &v312[v310];
                      v313 = *(__int64 **)(v293 + 16);
                      if ( v312 != v311 )
                      {
                        while ( 1 )
                        {
                          v6 = *v313;
                          if ( *(_QWORD *)v312 < (unsigned __int64)*v313 )
                            break;
                          if ( *(_QWORD *)v312 > (unsigned __int64)*v313 )
                            goto LABEL_498;
                          v312 += 8;
                          ++v313;
                          if ( v311 == v312 )
                            goto LABEL_497;
                        }
LABEL_465:
                        v304 = (_QWORD *)v304[3];
                        continue;
                      }
LABEL_497:
                      if ( v307 != v313 )
                        goto LABEL_465;
LABEL_498:
                      v309 = (unsigned __int64)v304;
                      v304 = (_QWORD *)v304[2];
                    }
                    while ( v304 );
                    if ( v309 == v306 )
                      goto LABEL_475;
                    v301 = *(_QWORD *)(v309 + 40);
                    v314 = *(_QWORD **)(v309 + 32);
                    v300 = v301 - (_QWORD)v314;
                    v294 = (__int64)v308 + v301 - (_QWORD)v314;
                    if ( v310 > v301 - (__int64)v314 )
                      v307 = (__int64 *)((char *)v308 + v300);
                    if ( v308 == v307 )
                    {
LABEL_501:
                      if ( (_QWORD *)v301 != v314 )
                        goto LABEL_475;
                    }
                    else
                    {
                      while ( (unsigned __int64)*v308 >= *v314 )
                      {
                        if ( (unsigned __int64)*v308 > *v314 )
                          goto LABEL_476;
                        ++v308;
                        ++v314;
                        if ( v307 == v308 )
                          goto LABEL_501;
                      }
LABEL_475:
                      v451 = (_BYTE *)(v293 + 16);
                      v309 = sub_26F9D50((_QWORD *)(v305 + 80), (_QWORD *)v309, (__int64 *)&v451);
                    }
LABEL_476:
                    v451 = (_BYTE *)v234;
                    v315 = *(_BYTE **)(v309 + 96);
                    if ( v315 == *(_BYTE **)(v309 + 104) )
                    {
                      sub_26FDF70(v309 + 88, v315, &v451);
                    }
                    else
                    {
                      if ( v315 )
                      {
                        *(_QWORD *)v315 = v234;
                        v315 = *(_BYTE **)(v309 + 96);
                      }
                      *(_QWORD *)(v309 + 96) = v315 + 8;
                    }
                    *(_BYTE *)(v309 + 80) = 0;
                    ++v302;
                  }
LABEL_481:
                  v233 = v465.m128i_u32[2];
                  v57 = v464;
                  v293 += 40;
                  if ( s2b == (void *)v293 )
                    goto LABEL_482;
                }
                s1 = (char *)s1 + 1;
                goto LABEL_546;
              }
            }
          }
        }
LABEL_482:
        if ( (__int64 *)nd == ++v402 )
        {
          v58 = v233;
          goto LABEL_484;
        }
        goto LABEL_351;
      }
    }
    v247 = *(_QWORD *)v245;
    v427 = *(_QWORD *)v245 + 8LL * *(unsigned int *)(v245 + 8);
LABEL_360:
    if ( v427 != v247 )
    {
      v248 = (__int64 *)v247;
      do
      {
        v250 = *v248;
        v458.m128i_i64[1] = v237;
        v458.m128i_i64[0] = v250;
        v251 = sub_26FA230(a1 + 128, &v458, v246, v247, v238, v6);
        v451 = (_BYTE *)v234;
        v249 = *(_BYTE **)(v251 + 64);
        if ( v249 == *(_BYTE **)(v251 + 72) )
        {
          v419 = v251;
          sub_26FDF70(v251 + 56, v249, &v451);
          v251 = v419;
        }
        else
        {
          if ( v249 )
          {
            *(_QWORD *)v249 = v234;
            v249 = *(_BYTE **)(v251 + 64);
          }
          *(_QWORD *)(v251 + 64) = v249 + 8;
        }
        v246 = 256;
        ++v248;
        *(_WORD *)(v251 + 24) = 256;
      }
      while ( (__int64 *)v427 != v248 );
    }
    goto LABEL_381;
  }
LABEL_23:
  v19 = v444;
  if ( v444 )
  {
    v20 = v442;
    v21 = &v442[7 * v444];
    do
    {
      if ( *v20 != -8192 && *v20 != -4096 )
        sub_26F75B0(v20[3]);
      v20 += 7;
    }
    while ( v21 != v20 );
    v19 = v444;
  }
  sub_C7D6A0((__int64)v442, 56 * v19, 8);
  v22 = v439;
  v23 = v438;
  if ( v439 != v438 )
  {
    do
    {
      v24 = v23[11];
      if ( v24 )
        j_j___libc_free_0(v24);
      v25 = v23[8];
      if ( v25 )
        j_j___libc_free_0(v25);
      v26 = v23[5];
      if ( v26 )
        j_j___libc_free_0(v26);
      v27 = v23[2];
      if ( v27 )
        j_j___libc_free_0(v27);
      v23 += 14;
    }
    while ( v22 != v23 );
    v23 = v438;
  }
  if ( v23 )
    j_j___libc_free_0((unsigned __int64)v23);
  return 1;
}
