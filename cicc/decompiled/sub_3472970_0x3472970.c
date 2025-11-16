// Function: sub_3472970
// Address: 0x3472970
//
__int64 __fastcall sub_3472970(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8,
        __int128 a9,
        unsigned int a10,
        __int64 a11,
        __int64 a12,
        int a13,
        __int128 a14,
        __int128 a15,
        __int128 a16,
        __int128 a17)
{
  __int64 v18; // r12
  int v19; // edi
  unsigned int v20; // eax
  bool v21; // r8
  bool v22; // r11
  bool v23; // al
  bool v24; // r8
  unsigned int v25; // r13d
  unsigned __int16 v27; // bx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rax
  int v33; // r13d
  __int64 v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // edx
  unsigned int v42; // ebx
  unsigned int v43; // esi
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  unsigned int v49; // r15d
  __int64 v50; // rdx
  int v51; // r9d
  __int64 v52; // rdx
  __int64 v53; // r9
  unsigned __int8 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  unsigned __int8 *v64; // r14
  __int64 v65; // rdx
  __int64 v66; // r12
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  unsigned __int8 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r9
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rdx
  __m128i v79; // xmm0
  __int64 v80; // rax
  __m128i v81; // rax
  __int64 v82; // r9
  unsigned __int8 *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r9
  int v86; // r9d
  __int64 v87; // rdx
  unsigned __int8 *v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  unsigned __int8 *v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rdx
  int v99; // r9d
  __int64 v100; // rdx
  __int64 v101; // r9
  __int64 v102; // rdx
  __int64 v103; // r9
  __int128 v104; // rax
  __int64 v105; // r9
  __int64 v106; // r9
  __int64 v107; // rdx
  unsigned __int8 *v108; // rax
  __int64 v109; // rdx
  __int64 v110; // r9
  __int64 v111; // rdx
  __int128 v112; // rax
  __int64 v113; // r13
  __int64 v114; // r15
  __int64 (__fastcall *v115)(__int64, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v116; // rax
  unsigned int v117; // eax
  unsigned int v118; // r15d
  __int64 v119; // rbx
  __int64 v120; // rdx
  __int64 v121; // r13
  int v122; // r9d
  __int128 v123; // rax
  int v124; // r9d
  __int64 v125; // rdx
  int v126; // r9d
  __int64 v127; // rdx
  __int64 v128; // r9
  __int64 v129; // rdx
  __int64 v130; // r9
  __m128i v131; // rax
  unsigned int *v132; // rax
  __int64 v133; // rdx
  __int64 v134; // r9
  unsigned __int8 *v135; // rax
  __int64 v136; // rdx
  int v137; // r9d
  unsigned __int8 *v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rcx
  __int64 v141; // r8
  __int64 v142; // r9
  __int64 v143; // r9
  unsigned __int8 *v144; // rax
  __int64 v145; // rdx
  unsigned int *v146; // rax
  __int64 v147; // rdx
  __int64 v148; // r9
  int v149; // r9d
  __int64 v150; // rdx
  __int64 v151; // rdx
  int v152; // r9d
  __int64 v153; // rdx
  __int64 v154; // r9
  __int64 v155; // rdx
  __int64 v156; // r9
  __int128 v157; // rax
  __int64 v158; // r9
  unsigned __int8 *v159; // rax
  int v160; // r9d
  __int64 v161; // rdx
  unsigned __int8 *v162; // rax
  __int64 v163; // rdx
  __int64 v164; // rcx
  __int64 v165; // r8
  __int64 v166; // r9
  __int64 v167; // r9
  __int64 v168; // rdx
  int v169; // r9d
  unsigned __int8 *v170; // rax
  __int64 v171; // rdx
  __int64 v172; // rcx
  __int64 v173; // r8
  __int64 v174; // r9
  unsigned int v175; // r13d
  __int64 v176; // rbx
  __int64 v177; // r9
  __int64 v178; // rdx
  int v179; // r9d
  __int64 v180; // rdx
  __int64 v181; // r9
  __int64 v182; // rdx
  int v183; // r9d
  __int64 v184; // rdx
  unsigned __int8 *v185; // rax
  __int64 v186; // rdx
  __int64 v187; // rdx
  __int64 v188; // r9
  __int64 v189; // rdx
  __int64 v190; // r9
  unsigned __int8 *v191; // rax
  __int64 v192; // rdx
  __int64 v193; // r9
  __int64 v194; // rdx
  __int64 v195; // rcx
  __int64 v196; // r8
  __int64 v197; // r9
  unsigned __int8 *v198; // rax
  __int64 v199; // rdx
  __int64 v200; // rdx
  int v201; // r9d
  __int64 v202; // rdx
  __int64 v203; // r9
  __int64 v204; // rdx
  __int64 v205; // r9
  __m128i v206; // rax
  unsigned int *v207; // rax
  __int64 v208; // rdx
  __int64 v209; // r9
  unsigned __int8 *v210; // rax
  __int64 v211; // rdx
  int v212; // r9d
  unsigned __int8 *v213; // rax
  __int64 v214; // rdx
  __int64 v215; // rcx
  __int64 v216; // r8
  __int64 v217; // r9
  __int64 v218; // r9
  unsigned __int8 *v219; // rax
  __int64 v220; // rdx
  unsigned int *v221; // rax
  __int64 v222; // rdx
  __int64 v223; // r9
  __int64 v224; // rdx
  __int128 v225; // rax
  __int64 v226; // r9
  __int128 v227; // rax
  __int64 v228; // rdx
  int v229; // r9d
  __int128 v230; // rax
  __int64 v231; // r9
  __int64 v232; // rdx
  __int64 v233; // rdx
  __int128 v234; // [rsp-30h] [rbp-490h]
  __int128 v235; // [rsp-30h] [rbp-490h]
  __int128 v236; // [rsp-20h] [rbp-480h]
  __int128 v237; // [rsp-20h] [rbp-480h]
  __int128 v238; // [rsp-20h] [rbp-480h]
  __int128 v239; // [rsp-20h] [rbp-480h]
  __int128 v240; // [rsp-10h] [rbp-470h]
  __int128 v241; // [rsp-10h] [rbp-470h]
  __int128 v242; // [rsp-10h] [rbp-470h]
  __int128 v243; // [rsp-10h] [rbp-470h]
  __int128 v244; // [rsp-10h] [rbp-470h]
  __int64 v245; // [rsp+8h] [rbp-458h]
  unsigned __int64 v246; // [rsp+8h] [rbp-458h]
  __int128 v247; // [rsp+10h] [rbp-450h]
  int v248; // [rsp+20h] [rbp-440h]
  __int128 v249; // [rsp+20h] [rbp-440h]
  unsigned __int8 *v250; // [rsp+30h] [rbp-430h]
  __int128 v251; // [rsp+30h] [rbp-430h]
  __int64 v252; // [rsp+38h] [rbp-428h]
  unsigned __int64 v253; // [rsp+38h] [rbp-428h]
  int v255; // [rsp+54h] [rbp-40Ch]
  unsigned __int8 *v256; // [rsp+58h] [rbp-408h]
  char v257; // [rsp+58h] [rbp-408h]
  unsigned __int8 *v258; // [rsp+60h] [rbp-400h]
  __int128 v259; // [rsp+60h] [rbp-400h]
  unsigned int v260; // [rsp+60h] [rbp-400h]
  __int64 v261; // [rsp+68h] [rbp-3F8h]
  unsigned __int64 v262; // [rsp+68h] [rbp-3F8h]
  __int64 v263; // [rsp+68h] [rbp-3F8h]
  unsigned __int64 v264; // [rsp+68h] [rbp-3F8h]
  __m128i v265; // [rsp+70h] [rbp-3F0h] BYREF
  __m128i v266; // [rsp+80h] [rbp-3E0h]
  __int64 v267; // [rsp+90h] [rbp-3D0h]
  __int64 v268; // [rsp+98h] [rbp-3C8h]
  unsigned __int8 *v269; // [rsp+A0h] [rbp-3C0h]
  __int64 v270; // [rsp+A8h] [rbp-3B8h]
  unsigned __int8 *v271; // [rsp+B0h] [rbp-3B0h]
  __int64 v272; // [rsp+B8h] [rbp-3A8h]
  unsigned __int8 *v273; // [rsp+C0h] [rbp-3A0h]
  __int64 v274; // [rsp+C8h] [rbp-398h]
  unsigned __int8 *v275; // [rsp+D0h] [rbp-390h]
  __int64 v276; // [rsp+D8h] [rbp-388h]
  unsigned __int8 *v277; // [rsp+E0h] [rbp-380h]
  __int64 v278; // [rsp+E8h] [rbp-378h]
  unsigned __int8 *v279; // [rsp+F0h] [rbp-370h]
  __int64 v280; // [rsp+F8h] [rbp-368h]
  unsigned __int8 *v281; // [rsp+100h] [rbp-360h]
  __int64 v282; // [rsp+108h] [rbp-358h]
  unsigned __int8 *v283; // [rsp+110h] [rbp-350h]
  __int64 v284; // [rsp+118h] [rbp-348h]
  unsigned __int8 *v285; // [rsp+120h] [rbp-340h]
  __int64 v286; // [rsp+128h] [rbp-338h]
  unsigned __int8 *v287; // [rsp+130h] [rbp-330h]
  __int64 v288; // [rsp+138h] [rbp-328h]
  unsigned __int8 *v289; // [rsp+140h] [rbp-320h]
  __int64 v290; // [rsp+148h] [rbp-318h]
  unsigned __int8 *v291; // [rsp+150h] [rbp-310h]
  __int64 v292; // [rsp+158h] [rbp-308h]
  unsigned __int8 *v293; // [rsp+160h] [rbp-300h]
  __int64 v294; // [rsp+168h] [rbp-2F8h]
  unsigned __int8 *v295; // [rsp+170h] [rbp-2F0h]
  __int64 v296; // [rsp+178h] [rbp-2E8h]
  unsigned __int8 *v297; // [rsp+180h] [rbp-2E0h]
  __int64 v298; // [rsp+188h] [rbp-2D8h]
  unsigned __int8 *v299; // [rsp+190h] [rbp-2D0h]
  __int64 v300; // [rsp+198h] [rbp-2C8h]
  unsigned __int8 *v301; // [rsp+1A0h] [rbp-2C0h]
  __int64 v302; // [rsp+1A8h] [rbp-2B8h]
  unsigned __int8 *v303; // [rsp+1B0h] [rbp-2B0h]
  __int64 v304; // [rsp+1B8h] [rbp-2A8h]
  unsigned __int8 *v305; // [rsp+1C0h] [rbp-2A0h]
  __int64 v306; // [rsp+1C8h] [rbp-298h]
  unsigned __int8 *v307; // [rsp+1D0h] [rbp-290h]
  __int64 v308; // [rsp+1D8h] [rbp-288h]
  unsigned __int8 *v309; // [rsp+1E0h] [rbp-280h]
  __int64 v310; // [rsp+1E8h] [rbp-278h]
  unsigned __int8 *v311; // [rsp+1F0h] [rbp-270h]
  __int64 v312; // [rsp+1F8h] [rbp-268h]
  unsigned __int8 *v313; // [rsp+200h] [rbp-260h]
  __int64 v314; // [rsp+208h] [rbp-258h]
  unsigned __int8 *v315; // [rsp+210h] [rbp-250h]
  __int64 v316; // [rsp+218h] [rbp-248h]
  unsigned __int8 *v317; // [rsp+220h] [rbp-240h]
  __int64 v318; // [rsp+228h] [rbp-238h]
  unsigned __int8 *v319; // [rsp+230h] [rbp-230h]
  __int64 v320; // [rsp+238h] [rbp-228h]
  unsigned __int8 *v321; // [rsp+240h] [rbp-220h]
  __int64 v322; // [rsp+248h] [rbp-218h]
  unsigned __int8 *v323; // [rsp+250h] [rbp-210h]
  __int64 v324; // [rsp+258h] [rbp-208h]
  unsigned __int8 *v325; // [rsp+260h] [rbp-200h]
  __int64 v326; // [rsp+268h] [rbp-1F8h]
  unsigned __int8 *v327; // [rsp+270h] [rbp-1F0h]
  __int64 v328; // [rsp+278h] [rbp-1E8h]
  unsigned __int8 *v329; // [rsp+280h] [rbp-1E0h]
  __int64 v330; // [rsp+288h] [rbp-1D8h]
  unsigned __int8 *v331; // [rsp+290h] [rbp-1D0h]
  __int64 v332; // [rsp+298h] [rbp-1C8h]
  unsigned __int8 *v333; // [rsp+2A0h] [rbp-1C0h]
  __int64 v334; // [rsp+2A8h] [rbp-1B8h]
  unsigned __int8 *v335; // [rsp+2B0h] [rbp-1B0h]
  __int64 v336; // [rsp+2B8h] [rbp-1A8h]
  unsigned __int8 *v337; // [rsp+2C0h] [rbp-1A0h]
  __int64 v338; // [rsp+2C8h] [rbp-198h]
  unsigned __int8 *v339; // [rsp+2D0h] [rbp-190h]
  __int64 v340; // [rsp+2D8h] [rbp-188h]
  unsigned __int8 *v341; // [rsp+2E0h] [rbp-180h]
  __int64 v342; // [rsp+2E8h] [rbp-178h]
  __int64 v343; // [rsp+2F0h] [rbp-170h]
  __int64 v344; // [rsp+2F8h] [rbp-168h]
  unsigned __int8 *v345; // [rsp+300h] [rbp-160h]
  __int64 v346; // [rsp+308h] [rbp-158h]
  __int64 v347; // [rsp+310h] [rbp-150h]
  __int64 v348; // [rsp+318h] [rbp-148h]
  unsigned __int8 *v349; // [rsp+320h] [rbp-140h]
  __int64 v350; // [rsp+328h] [rbp-138h]
  unsigned __int8 *v351; // [rsp+330h] [rbp-130h]
  __int64 v352; // [rsp+338h] [rbp-128h]
  unsigned __int8 *v353; // [rsp+340h] [rbp-120h]
  __int64 v354; // [rsp+348h] [rbp-118h]
  unsigned __int8 *v355; // [rsp+350h] [rbp-110h]
  __int64 v356; // [rsp+358h] [rbp-108h]
  unsigned __int8 *v357; // [rsp+360h] [rbp-100h]
  __int64 v358; // [rsp+368h] [rbp-F8h]
  __int64 v359; // [rsp+370h] [rbp-F0h] BYREF
  __int64 v360; // [rsp+378h] [rbp-E8h]
  bool v361; // [rsp+38Ch] [rbp-D4h] BYREF
  bool v362; // [rsp+38Dh] [rbp-D3h] BYREF
  bool v363; // [rsp+38Eh] [rbp-D2h] BYREF
  char v364; // [rsp+38Fh] [rbp-D1h] BYREF
  _QWORD v365[2]; // [rsp+390h] [rbp-D0h] BYREF
  __m128i v366; // [rsp+3A0h] [rbp-C0h] BYREF
  __int128 v367; // [rsp+3B0h] [rbp-B0h] BYREF
  unsigned __int64 v368; // [rsp+3C0h] [rbp-A0h] BYREF
  __int64 v369; // [rsp+3C8h] [rbp-98h]
  __int64 v370; // [rsp+3D0h] [rbp-90h]
  __int64 v371; // [rsp+3D8h] [rbp-88h]
  __int64 v372; // [rsp+3E0h] [rbp-80h]
  __int64 v373; // [rsp+3E8h] [rbp-78h]
  bool *v374; // [rsp+3F0h] [rbp-70h] BYREF
  char *v375; // [rsp+3F8h] [rbp-68h]
  _QWORD *v376; // [rsp+400h] [rbp-60h]
  __int64 v377; // [rsp+408h] [rbp-58h]
  unsigned int **v378; // [rsp+410h] [rbp-50h]
  bool *v379; // [rsp+418h] [rbp-48h]
  bool *v380; // [rsp+420h] [rbp-40h]
  unsigned int *v381; // [rsp+428h] [rbp-38h]

  v266.m128i_i64[0] = a1;
  v255 = a2;
  v265.m128i_i64[0] = a14;
  v359 = a3;
  v18 = a12;
  v256 = (unsigned __int8 *)a15;
  v360 = a4;
  v258 = (unsigned __int8 *)a16;
  v250 = (unsigned __int8 *)a17;
  if ( !a13 )
  {
    v361 = 1;
    v362 = 1;
    v363 = 1;
    goto LABEL_12;
  }
  v19 = 1;
  if ( (_WORD)a10 == 1 )
    goto LABEL_3;
  if ( !(_WORD)a10 )
    return 0;
  v19 = (unsigned __int16)a10;
  if ( *(_QWORD *)(v266.m128i_i64[0] + 8LL * (unsigned __int16)a10 + 112) )
  {
LABEL_3:
    v20 = 1;
    v361 = (*(_BYTE *)(v266.m128i_i64[0] + 500LL * (unsigned int)v19 + 6587) & 0xFB) == 0;
    v21 = v361;
    if ( (_WORD)a10 != 1 )
    {
      if ( !*(_QWORD *)(v266.m128i_i64[0] + 8LL * v19 + 112) )
        goto LABEL_49;
      v20 = v19;
    }
    a2 = v20;
    v362 = (*(_BYTE *)(v266.m128i_i64[0] + 500LL * v20 + 6586) & 0xFB) == 0;
    v22 = v362;
    if ( (_WORD)a10 != 1 )
    {
      if ( !*(_QWORD *)(v266.m128i_i64[0] + 8LL * v19 + 112) )
        goto LABEL_50;
      a2 = (unsigned int)v19;
    }
    v363 = (*(_BYTE *)(v266.m128i_i64[0] + 500 * a2 + 6477) & 0xFB) == 0;
    v23 = v363;
    if ( (_WORD)a10 != 1 )
    {
      if ( !*(_QWORD *)(v266.m128i_i64[0] + 8LL * v19 + 112) )
        goto LABEL_51;
      a2 = (unsigned int)v19;
    }
    if ( (*(_BYTE *)(v266.m128i_i64[0] + 500 * a2 + 6478) & 0xFB) != 0 )
    {
      v24 = v363 || v361 || v362;
      goto LABEL_8;
    }
LABEL_12:
    v364 = 1;
    goto LABEL_13;
  }
  v361 = 0;
  v21 = 0;
LABEL_49:
  v362 = 0;
  v22 = 0;
LABEL_50:
  v363 = 0;
  v23 = 0;
LABEL_51:
  v24 = v22 || v21 || v23;
LABEL_8:
  v364 = 0;
  if ( !v24 )
    return 0;
LABEL_13:
  v27 = v359;
  if ( !(_WORD)v359 )
  {
    if ( sub_30070B0((__int64)&v359) )
    {
      v27 = sub_3009970((__int64)&v359, a2, v28, v29, v30);
LABEL_16:
      LOWORD(v374) = v27;
      v375 = (char *)v31;
      if ( !v27 )
        goto LABEL_17;
LABEL_41:
      if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
        goto LABEL_111;
      v33 = (unsigned __int16)a10;
      v34 = *(_QWORD *)&byte_444C4A0[16 * v27 - 16];
      if ( (_WORD)a10 )
        goto LABEL_18;
LABEL_44:
      if ( !sub_30070B0((__int64)&a10) )
        goto LABEL_19;
      LOWORD(v33) = sub_3009970((__int64)&a10, a2, v46, v47, v48);
      goto LABEL_20;
    }
LABEL_15:
    v31 = v360;
    goto LABEL_16;
  }
  if ( (unsigned __int16)(v359 - 17) > 0xD3u )
    goto LABEL_15;
  v375 = 0;
  v27 = word_4456580[(unsigned __int16)v359 - 1];
  LOWORD(v374) = v27;
  if ( v27 )
    goto LABEL_41;
LABEL_17:
  v32 = sub_3007260((__int64)&v374);
  v33 = (unsigned __int16)a10;
  v370 = v32;
  LODWORD(v34) = v32;
  v371 = v35;
  if ( !(_WORD)a10 )
    goto LABEL_44;
LABEL_18:
  if ( (unsigned __int16)(v33 - 17) > 0xD3u )
  {
LABEL_19:
    v36 = a11;
    goto LABEL_20;
  }
  v36 = 0;
  LOWORD(v33) = word_4456580[v33 - 1];
LABEL_20:
  LOWORD(v368) = v33;
  v369 = v36;
  if ( (_WORD)v33 )
  {
    if ( (_WORD)v33 != 1 && (unsigned __int16)(v33 - 504) > 7u )
    {
      v37 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v33 - 16];
      goto LABEL_22;
    }
LABEL_111:
    BUG();
  }
  v372 = sub_3007260((__int64)&v368);
  LODWORD(v37) = v372;
  v373 = v38;
LABEL_22:
  v39 = sub_33E5110((__int64 *)v18, a10, a11, a10, a11);
  v376 = (_QWORD *)v18;
  v365[0] = v39;
  v374 = &v363;
  v375 = &v364;
  v378 = (unsigned int **)v365;
  v379 = &v361;
  v380 = &v362;
  v365[1] = v40;
  v377 = a5;
  v381 = &a10;
  v366.m128i_i64[0] = 0;
  v366.m128i_i32[2] = 0;
  *(_QWORD *)&v367 = 0;
  DWORD2(v367) = 0;
  if ( !v265.m128i_i64[0] )
  {
    if ( v258 )
      return 0;
    v49 = a10;
    v265.m128i_i64[0] = a11;
    if ( !(unsigned __int8)sub_328A020(v266.m128i_i64[0], 0xD8u, a10, a11, 0) )
      return 0;
    v357 = sub_33FAF80(v18, 216, a5, v49, v265.m128i_i64[0], v265.m128i_i32[0], a7);
    v265.m128i_i64[0] = (__int64)v357;
    *(_QWORD *)&a14 = v357;
    v358 = v50;
    *((_QWORD *)&a14 + 1) = (unsigned int)v50 | *((_QWORD *)&a14 + 1) & 0xFFFFFFFF00000000LL;
    v258 = sub_33FAF80(v18, 216, a5, a10, a11, v51, a7);
    v355 = v258;
    *(_QWORD *)&a16 = v258;
    v356 = v52;
    *((_QWORD *)&a16 + 1) = (unsigned int)v52 | *((_QWORD *)&a16 + 1) & 0xFFFFFFFF00000000LL;
    if ( !v357 )
      return 0;
  }
  v248 = v34;
  LODWORD(v369) = v34;
  if ( (unsigned int)v34 > 0x40 )
  {
    sub_C43690((__int64)&v368, 0, 0);
    v41 = v369;
  }
  else
  {
    v368 = 0;
    v41 = v34;
  }
  v42 = v37;
  v43 = v41 - v37;
  if ( v41 != v41 - (_DWORD)v37 )
  {
    if ( v43 > 0x3F || v41 > 0x40 )
      sub_C43C90(&v368, v43, v41);
    else
      v368 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) << v43;
  }
  if ( (unsigned __int8)sub_33DD210(v18, a8, *((__int64 *)&a8 + 1), (__int64)&v368, 0) )
  {
    v25 = sub_33DD210(v18, a9, *((__int64 *)&a9 + 1), (__int64)&v368, 0);
    if ( (_BYTE)v25 )
    {
      *(_QWORD *)&a14 = v265.m128i_i64[0];
      *(_QWORD *)&a16 = v258;
      if ( *v375 )
      {
        v185 = sub_3411F20(v376, 64, v377, *v378, (__int64)v378[1], v53, a14, a16);
        DWORD2(v367) = 1;
        v354 = v186;
        v353 = v185;
        v366.m128i_i64[0] = (__int64)v185;
        v366.m128i_i32[2] = v186;
        *(_QWORD *)&v367 = v185;
LABEL_65:
        sub_3050D50(a6, v366.m128i_i64[0], v366.m128i_i64[1], v57, v58, v59);
        sub_3050D50(a6, v367, *((__int64 *)&v367 + 1), v61, v62, v63);
        if ( v255 != 58 )
        {
          v64 = sub_3400BD0(v18, 0, a5, a10, a11, 0, a7, 0);
          v66 = v65;
          sub_3050D50(a6, (__int64)v64, v65, v67, v68, v69);
          sub_3050D50(a6, (__int64)v64, v66, v70, v71, v72);
        }
        goto LABEL_36;
      }
      if ( *v380 )
      {
        v54 = sub_3406EB0(v376, 0x3Au, v377, *v381, *((_QWORD *)v381 + 1), v53, a14, a16);
        v352 = v55;
        v351 = v54;
        v366.m128i_i64[0] = (__int64)v54;
        v366.m128i_i32[2] = v55;
        v349 = sub_3406EB0(v376, 0xACu, v377, *v381, *((_QWORD *)v381 + 1), v56, a14, a16);
        v350 = v60;
        *(_QWORD *)&v367 = v349;
        DWORD2(v367) = v60;
        goto LABEL_65;
      }
    }
  }
  if ( (_WORD)v359 )
  {
    if ( (unsigned __int16)(v359 - 17) <= 0xD3u )
      goto LABEL_33;
  }
  else if ( sub_30070B0((__int64)&v359) )
  {
    goto LABEL_33;
  }
  if ( v255 == 58
    && (unsigned int)sub_33DF530(v18, a8, *((__int64 *)&a8 + 1), 0) <= v42
    && (unsigned int)sub_33DF530(v18, a9, *((__int64 *)&a9 + 1), 0) <= v42 )
  {
    *(_QWORD *)&a14 = v265.m128i_i64[0];
    *(_QWORD *)&a16 = v258;
    v25 = sub_343F480(
            (__int64)&v374,
            v265.m128i_i64[0],
            *((__int64 *)&a14 + 1),
            (__int64)v258,
            *((__int64 *)&a16 + 1),
            (__int64)&v366,
            (__int64)&v367,
            1);
    if ( (_BYTE)v25 )
    {
      sub_3050D50(a6, v366.m128i_i64[0], v366.m128i_i64[1], v90, v91, v92);
      sub_3050D50(a6, v367, *((__int64 *)&v367 + 1), v93, v94, v95);
      goto LABEL_36;
    }
  }
LABEL_33:
  *(_QWORD *)&v249 = sub_3400E40(v18, v248 - v42, v359, v360, a5, a7);
  *((_QWORD *)&v249 + 1) = v45;
  if ( !v256 )
  {
    if ( v250 )
      goto LABEL_83;
    v175 = v359;
    v176 = v360;
    if ( !(unsigned __int8)sub_328A020(v266.m128i_i64[0], 0xC0u, v359, v360, 0) )
      goto LABEL_83;
    if ( !(unsigned __int8)sub_328A020(v266.m128i_i64[0], 0xD8u, a10, a11, 0) )
      goto LABEL_83;
    *(_QWORD *)&a15 = sub_3406EB0((_QWORD *)v18, 0xC0u, a5, v175, v176, v177, a8, v249);
    v347 = a15;
    v348 = v178;
    *((_QWORD *)&a15 + 1) = (unsigned int)v178 | *((_QWORD *)&a15 + 1) & 0xFFFFFFFF00000000LL;
    v256 = sub_33FAF80(v18, 216, a5, a10, a11, v179, a7);
    v345 = v256;
    *(_QWORD *)&a15 = v256;
    v346 = v180;
    *((_QWORD *)&a15 + 1) = (unsigned int)v180 | *((_QWORD *)&a15 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&a17 = sub_3406EB0((_QWORD *)v18, 0xC0u, a5, (unsigned int)v359, v360, v181, a9, v249);
    v343 = a17;
    v344 = v182;
    *((_QWORD *)&a17 + 1) = (unsigned int)v182 | *((_QWORD *)&a17 + 1) & 0xFFFFFFFF00000000LL;
    v250 = sub_33FAF80(v18, 216, a5, a10, a11, v183, a7);
    v341 = v250;
    *(_QWORD *)&a17 = v250;
    v342 = v184;
    *((_QWORD *)&a17 + 1) = (unsigned int)v184 | *((_QWORD *)&a17 + 1) & 0xFFFFFFFF00000000LL;
    if ( !v256 )
      goto LABEL_83;
  }
  *(_QWORD *)&a14 = v265.m128i_i64[0];
  *(_QWORD *)&a16 = v258;
  if ( *v375 )
  {
    v88 = sub_3411F20(v376, 64, v377, *v378, (__int64)v378[1], v44, a14, a16);
    DWORD2(v367) = 1;
    v340 = v89;
    v339 = v88;
    v366.m128i_i64[0] = (__int64)v88;
    v366.m128i_i32[2] = v89;
    *(_QWORD *)&v367 = v88;
LABEL_70:
    v79 = _mm_load_si128(&v366);
    v80 = *(unsigned int *)(a6 + 8);
    if ( v80 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
    {
      v265 = v79;
      sub_C8D5F0(a6, (const void *)(a6 + 16), v80 + 1, 0x10u, v76, v77);
      v80 = *(unsigned int *)(a6 + 8);
      v79 = _mm_load_si128(&v265);
    }
    *(__m128i *)(*(_QWORD *)a6 + 16 * v80) = v79;
    ++*(_DWORD *)(a6 + 8);
    if ( v255 == 58 )
    {
      *((_QWORD *)&v243 + 1) = *((_QWORD *)&a17 + 1);
      *(_QWORD *)&a17 = v250;
      v25 = 1;
      *(_QWORD *)&v243 = v250;
      v333 = sub_3406EB0((_QWORD *)v18, 0x3Au, a5, a10, a11, v77, a14, v243);
      *(_QWORD *)&a17 = v333;
      v334 = v187;
      *((_QWORD *)&v238 + 1) = *((_QWORD *)&a15 + 1);
      *((_QWORD *)&a17 + 1) = (unsigned int)v187 | *((_QWORD *)&a17 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&a15 = v256;
      *(_QWORD *)&v238 = v256;
      v331 = sub_3406EB0((_QWORD *)v18, 0x3Au, a5, a10, a11, v188, v238, a16);
      *(_QWORD *)&a15 = v331;
      v332 = v189;
      *((_QWORD *)&a15 + 1) = (unsigned int)v189 | *((_QWORD *)&a15 + 1) & 0xFFFFFFFF00000000LL;
      v191 = sub_3406EB0((_QWORD *)v18, 0x38u, a5, a10, a11, v190, v367, a17);
      v330 = v192;
      *(_QWORD *)&v367 = v191;
      v329 = v191;
      DWORD2(v367) = v192;
      v327 = sub_3406EB0((_QWORD *)v18, 0x38u, a5, a10, a11, v193, v367, a15);
      v328 = v194;
      *(_QWORD *)&v367 = v327;
      DWORD2(v367) = v194;
      sub_3050D50(a6, (__int64)v327, *((__int64 *)&v367 + 1), v195, v196, v197);
      goto LABEL_36;
    }
    v81.m128i_i64[0] = (__int64)sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v77, v79);
    v265 = v81;
    *(_QWORD *)&a17 = v250;
    if ( *v375 )
    {
      v96 = sub_3411F20(v376, 64, v377, *v378, (__int64)v378[1], v82, a14, a17);
      DWORD2(v367) = 1;
      v326 = v97;
      v325 = v96;
      v366.m128i_i64[0] = (__int64)v96;
      v366.m128i_i32[2] = v97;
      *(_QWORD *)&v367 = v96;
    }
    else
    {
      v25 = *v380;
      if ( !(_BYTE)v25 )
        goto LABEL_36;
      v83 = sub_3406EB0(v376, 0x3Au, v377, *v381, *((_QWORD *)v381 + 1), v82, a14, a17);
      v324 = v84;
      v323 = v83;
      v366.m128i_i64[0] = (__int64)v83;
      v366.m128i_i32[2] = v84;
      v321 = sub_3406EB0(v376, 0xACu, v377, *v381, *((_QWORD *)v381 + 1), v85, a14, a17);
      v322 = v87;
      *(_QWORD *)&v367 = v321;
      DWORD2(v367) = v87;
    }
    v252 = *((_QWORD *)&v367 + 1);
    v259 = (__int128)_mm_load_si128(&v366);
    v319 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v86, v79);
    *(_QWORD *)&v259 = v319;
    v320 = v98;
    *((_QWORD *)&v259 + 1) = (unsigned int)v98 | *((_QWORD *)&v259 + 1) & 0xFFFFFFFF00000000LL;
    v317 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v99, v79);
    v318 = v100;
    *(_QWORD *)&v236 = v317;
    v253 = (unsigned int)v100 | v252 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v236 + 1) = v253;
    v315 = sub_3406EB0((_QWORD *)v18, 0xBEu, a5, (unsigned int)v359, v360, v101, v236, v249);
    v316 = v102;
    *((_QWORD *)&v240 + 1) = (unsigned int)v102 | v253 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v240 = v315;
    *(_QWORD *)&v104 = sub_3406EB0((_QWORD *)v18, 0xBBu, a5, (unsigned int)v359, v360, v103, v259, v240);
    v313 = sub_3406EB0((_QWORD *)v18, 0x38u, a5, (unsigned int)v359, v360, v105, *(_OWORD *)&v265, v104);
    v265.m128i_i64[0] = (__int64)v313;
    v314 = v107;
    *(_QWORD *)&a15 = v256;
    v265.m128i_i64[1] = (unsigned int)v107 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( *v375 )
    {
      v198 = sub_3411F20(v376, 64, v377, *v378, (__int64)v378[1], v106, a15, a16);
      DWORD2(v367) = 1;
      v312 = v199;
      v311 = v198;
      v366.m128i_i64[0] = (__int64)v198;
      v366.m128i_i32[2] = v199;
      *(_QWORD *)&v367 = v198;
    }
    else
    {
      v25 = *v380;
      if ( !(_BYTE)v25 )
        goto LABEL_36;
      v108 = sub_3406EB0(v376, 0x3Au, v377, *v381, *((_QWORD *)v381 + 1), v106, a15, a16);
      v310 = v109;
      v309 = v108;
      v366.m128i_i64[0] = (__int64)v108;
      v366.m128i_i32[2] = v109;
      v307 = sub_3406EB0(v376, 0xACu, v377, *v381, *((_QWORD *)v381 + 1), v110, a15, a16);
      v308 = v111;
      *(_QWORD *)&v367 = v307;
      DWORD2(v367) = v111;
    }
    *(_QWORD *)&v112 = sub_3400BD0(v18, 0, a5, a10, a11, 0, v79, 0);
    v113 = *(_QWORD *)(v18 + 64);
    v114 = v266.m128i_i64[0];
    v251 = v112;
    v115 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v266.m128i_i64[0] + 528LL);
    v116 = sub_2E79000(*(__int64 **)(v18 + 40));
    v266.m128i_i64[0] = v114;
    v117 = v115(v114, v116, v113, (unsigned int)v359, v360);
    v118 = v359;
    v119 = v360;
    v121 = v120;
    v260 = v117;
    v257 = v255 == 63;
    if ( (unsigned __int8)sub_328A020(v266.m128i_i64[0], 0x44u, v359, v360, 0)
      && (unsigned __int8)sub_328A020(v266.m128i_i64[0], 0x46u, v118, v119, 0) )
    {
      v263 = *((_QWORD *)&v367 + 1);
      v266 = _mm_load_si128(&v366);
      v295 = sub_33FAF80(v18, 214, a5, v118, v119, v122, v79);
      v266.m128i_i64[0] = (__int64)v295;
      v296 = v200;
      v266.m128i_i64[1] = (unsigned int)v200 | v266.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v293 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v201, v79);
      v294 = v202;
      *(_QWORD *)&v239 = v293;
      v264 = (unsigned int)v202 | v263 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v239 + 1) = v264;
      v291 = sub_3406EB0((_QWORD *)v18, 0xBEu, a5, (unsigned int)v359, v360, v203, v239, v249);
      v292 = v204;
      *((_QWORD *)&v244 + 1) = (unsigned int)v204 | v264 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v244 = v291;
      v206.m128i_i64[0] = (__int64)sub_3406EB0(
                                     (_QWORD *)v18,
                                     0xBBu,
                                     a5,
                                     (unsigned int)v359,
                                     v360,
                                     v205,
                                     *(_OWORD *)&v266,
                                     v244);
      v266 = v206;
      v207 = (unsigned int *)sub_33E5110((__int64 *)v18, (unsigned int)v359, v360, 262, 0);
      v210 = sub_3411F20((_QWORD *)v18, 68, a5, v207, v208, v209, *(_OWORD *)&v265, *(_OWORD *)&v266);
      v290 = v211;
      v265.m128i_i64[0] = (__int64)v210;
      v289 = v210;
      v266.m128i_i64[0] = (__int64)v210;
      v265.m128i_i64[1] = (unsigned int)v211 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v266.m128i_i64[1] = 1;
      v213 = sub_33FAF80(v18, 216, a5, a10, a11, v212, v79);
      sub_3050D50(a6, (__int64)v213, v214, v215, v216, v217);
      v219 = sub_3406EB0((_QWORD *)v18, 0xC0u, a5, (unsigned int)v359, v360, v218, *(_OWORD *)&v265, v249);
      v288 = v220;
      v287 = v219;
      v265.m128i_i64[0] = (__int64)v219;
      v265.m128i_i64[1] = (unsigned int)v220 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( (unsigned __int8)sub_343F480(
                              (__int64)&v374,
                              a15,
                              *((__int64 *)&a15 + 1),
                              a17,
                              *((__int64 *)&a17 + 1),
                              (__int64)&v366,
                              (__int64)&v367,
                              v257) )
      {
        v221 = (unsigned int *)sub_33E5110((__int64 *)v18, a10, a11, 262, 0);
        v285 = sub_3412970((_QWORD *)v18, 70, a5, v221, v222, v223, v367, v251, *(_OWORD *)&v266);
        v286 = v224;
        *(_QWORD *)&v367 = v285;
        DWORD2(v367) = v224;
        goto LABEL_99;
      }
    }
    else
    {
      *(_QWORD *)&v123 = sub_3400BD0(v18, 0, a5, v260, v121, 0, v79, 0);
      v266 = _mm_load_si128(&v366);
      v245 = *((_QWORD *)&v367 + 1);
      v247 = v123;
      v305 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v124, v79);
      v266.m128i_i64[0] = (__int64)v305;
      v306 = v125;
      v266.m128i_i64[1] = (unsigned int)v125 | v266.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v303 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v126, v79);
      v304 = v127;
      *(_QWORD *)&v234 = v303;
      v246 = (unsigned int)v127 | v245 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v234 + 1) = v246;
      v301 = sub_3406EB0((_QWORD *)v18, 0xBEu, a5, (unsigned int)v359, v360, v128, v234, v249);
      v302 = v129;
      *((_QWORD *)&v241 + 1) = (unsigned int)v129 | v246 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v241 = v301;
      v131.m128i_i64[0] = (__int64)sub_3406EB0(
                                     (_QWORD *)v18,
                                     0xBBu,
                                     a5,
                                     (unsigned int)v359,
                                     v360,
                                     v130,
                                     *(_OWORD *)&v266,
                                     v241);
      v266 = v131;
      v132 = (unsigned int *)sub_33E5110((__int64 *)v18, (unsigned int)v359, v360, v260, v121);
      v135 = sub_3412970((_QWORD *)v18, 72, a5, v132, v133, v134, *(_OWORD *)&v265, *(_OWORD *)&v266, v247);
      v300 = v136;
      v265.m128i_i64[0] = (__int64)v135;
      v299 = v135;
      v266.m128i_i64[0] = (__int64)v135;
      v265.m128i_i64[1] = (unsigned int)v136 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v266.m128i_i64[1] = 1;
      v138 = sub_33FAF80(v18, 216, a5, a10, a11, v137, v79);
      sub_3050D50(a6, (__int64)v138, v139, v140, v141, v142);
      v144 = sub_3406EB0((_QWORD *)v18, 0xC0u, a5, (unsigned int)v359, v360, v143, *(_OWORD *)&v265, v249);
      v298 = v145;
      v297 = v144;
      v265.m128i_i64[0] = (__int64)v144;
      v265.m128i_i64[1] = (unsigned int)v145 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( (unsigned __int8)sub_343F480(
                              (__int64)&v374,
                              a15,
                              *((__int64 *)&a15 + 1),
                              a17,
                              *((__int64 *)&a17 + 1),
                              (__int64)&v366,
                              (__int64)&v367,
                              v257) )
      {
        v146 = (unsigned int *)sub_33E5110((__int64 *)v18, a10, a11, v260, v121);
        v283 = sub_3412970((_QWORD *)v18, 72, a5, v146, v147, v148, v367, v251, *(_OWORD *)&v266);
        v284 = v150;
        *(_QWORD *)&v367 = v283;
        DWORD2(v367) = v150;
LABEL_99:
        v261 = *((_QWORD *)&v367 + 1);
        v266 = _mm_load_si128(&v366);
        v281 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v149, v79);
        v266.m128i_i64[0] = (__int64)v281;
        v282 = v151;
        v266.m128i_i64[1] = (unsigned int)v151 | v266.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v279 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v152, v79);
        v280 = v153;
        *(_QWORD *)&v237 = v279;
        v262 = (unsigned int)v153 | v261 & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v237 + 1) = v262;
        v277 = sub_3406EB0((_QWORD *)v18, 0xBEu, a5, (unsigned int)v359, v360, v154, v237, v249);
        v278 = v155;
        *((_QWORD *)&v242 + 1) = (unsigned int)v155 | v262 & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v242 = v277;
        *(_QWORD *)&v157 = sub_3406EB0((_QWORD *)v18, 0xBBu, a5, (unsigned int)v359, v360, v156, *(_OWORD *)&v266, v242);
        v159 = sub_3406EB0((_QWORD *)v18, 0x38u, a5, (unsigned int)v359, v360, v158, *(_OWORD *)&v265, v157);
        v276 = v161;
        v275 = v159;
        v265.m128i_i64[0] = (__int64)v159;
        v265.m128i_i64[1] = (unsigned int)v161 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        if ( v255 == 63 )
        {
          *(_QWORD *)&v225 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v160, v79);
          *(_QWORD *)&v227 = sub_3406EB0(
                               (_QWORD *)v18,
                               0x39u,
                               a5,
                               (unsigned int)v359,
                               v360,
                               v226,
                               *(_OWORD *)&v265,
                               v225);
          v266 = (__m128i)v227;
          v273 = sub_3445FE0(
                   (_QWORD *)v18,
                   a5,
                   a15,
                   *((__int64 *)&a15 + 1),
                   v251,
                   *((__int64 *)&v251 + 1),
                   v227,
                   *(_OWORD *)&v265,
                   0x14u);
          v265.m128i_i64[0] = (__int64)v273;
          v274 = v228;
          v265.m128i_i64[1] = (unsigned int)v228 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v230 = sub_33FAF80(v18, 214, a5, (unsigned int)v359, v360, v229, v79);
          v271 = sub_3406EB0((_QWORD *)v18, 0x39u, a5, (unsigned int)v359, v360, v231, *(_OWORD *)&v265, v230);
          v272 = v232;
          *((_QWORD *)&v235 + 1) = (unsigned int)v232 | v266.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v235 = v271;
          v159 = sub_3445FE0(
                   (_QWORD *)v18,
                   a5,
                   a17,
                   *((__int64 *)&a17 + 1),
                   v251,
                   *((__int64 *)&v251 + 1),
                   v235,
                   *(_OWORD *)&v265,
                   0x14u);
          v270 = v233;
          v269 = v159;
          v265.m128i_i64[1] = (unsigned int)v233 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        }
        v265.m128i_i64[0] = (__int64)v159;
        v25 = 1;
        v162 = sub_33FAF80(v18, 216, a5, a10, a11, v160, v79);
        sub_3050D50(a6, (__int64)v162, v163, v164, v165, v166);
        v265.m128i_i64[0] = (__int64)sub_3406EB0(
                                       (_QWORD *)v18,
                                       0xC0u,
                                       a5,
                                       (unsigned int)v359,
                                       v360,
                                       v167,
                                       *(_OWORD *)&v265,
                                       v249);
        v267 = v265.m128i_i64[0];
        v268 = v168;
        v265.m128i_i64[1] = (unsigned int)v168 | v265.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v170 = sub_33FAF80(v18, 216, a5, a10, a11, v169, v79);
        sub_3050D50(a6, (__int64)v170, v171, v172, v173, v174);
        goto LABEL_36;
      }
    }
LABEL_83:
    v25 = 0;
    goto LABEL_36;
  }
  v25 = *v380;
  if ( (_BYTE)v25 )
  {
    v73 = sub_3406EB0(v376, 0x3Au, v377, *v381, *((_QWORD *)v381 + 1), v44, a14, a16);
    v338 = v74;
    v337 = v73;
    v366.m128i_i64[0] = (__int64)v73;
    v366.m128i_i32[2] = v74;
    v335 = sub_3406EB0(v376, 0xACu, v377, *v381, *((_QWORD *)v381 + 1), v75, a14, a16);
    v336 = v78;
    *(_QWORD *)&v367 = v335;
    DWORD2(v367) = v78;
    goto LABEL_70;
  }
LABEL_36:
  if ( (unsigned int)v369 > 0x40 && v368 )
    j_j___libc_free_0_0(v368);
  return v25;
}
