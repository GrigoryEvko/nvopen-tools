// Function: sub_3921CE0
// Address: 0x3921ce0
//
__int64 __fastcall sub_3921CE0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v3; // r15
  _QWORD *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __m128i v11; // rax
  __int64 v12; // rax
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  __m128i *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  const void *v19; // r9
  __m128i v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm6
  __m128i v23; // xmm7
  __m128i *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rbx
  __int64 *v27; // r13
  char v28; // al
  unsigned __int64 v29; // r14
  int v30; // eax
  __int64 v31; // r12
  __int64 v32; // rcx
  __int64 v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 *v36; // r12
  __int64 v37; // rbx
  _QWORD *v38; // rdx
  void *v39; // rax
  char v40; // cl
  unsigned __int32 v41; // ecx
  int v42; // r14d
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __m128i v46; // xmm2
  __int64 v47; // r13
  int v48; // eax
  _BYTE *v49; // rax
  __int64 *v50; // r9
  __int64 v51; // rcx
  __int64 v52; // r9
  int *v53; // rbx
  int *v54; // r13
  const void *v55; // r12
  size_t v56; // r15
  int v57; // eax
  size_t v58; // r14
  const void *v59; // rdi
  size_t v60; // rcx
  size_t v61; // rbx
  const void *v62; // rsi
  int v63; // eax
  _BYTE *v64; // rsi
  __int64 *v65; // r12
  int v66; // ebx
  __int64 v67; // r14
  __int64 v68; // rbx
  unsigned int v69; // esi
  __int64 v70; // r9
  unsigned int v71; // r8d
  __int64 *v72; // rax
  __int64 v73; // rdi
  unsigned int v74; // eax
  unsigned int v75; // esi
  unsigned __int64 v76; // xmm0_8
  __int64 v77; // r9
  __int64 *v78; // r8
  unsigned int v79; // edi
  __int64 *v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rax
  unsigned __int64 *v83; // rax
  unsigned __int64 v84; // rax
  _BYTE *v85; // rax
  __int64 v86; // r9
  __int64 *v87; // r9
  int *v88; // r14
  int *v89; // r13
  const void *v90; // r12
  size_t v91; // r15
  int v92; // eax
  size_t v93; // rbx
  const void *v94; // rdi
  const void *v95; // r9
  size_t v96; // rbx
  size_t v97; // r14
  const void *v98; // rsi
  int v99; // eax
  _BYTE *v100; // rsi
  __int64 v101; // r10
  __int64 v102; // r12
  unsigned int v103; // esi
  __int64 *v104; // rcx
  unsigned int v105; // edi
  int v106; // r9d
  __int64 v107; // rdx
  __int64 *v108; // r14
  __int64 v109; // r8
  __int64 v110; // r9
  __int64 *v111; // rax
  __int64 v112; // r8
  __int64 v113; // rbx
  __int64 v114; // rax
  int v115; // r13d
  __int64 v116; // rax
  unsigned int v117; // esi
  __int64 v118; // rdx
  unsigned int v119; // ecx
  unsigned int v120; // r8d
  __int64 *v121; // rdi
  __int64 v122; // r11
  int v123; // r14d
  unsigned int v124; // r13d
  unsigned int v125; // edi
  __int64 *v126; // rax
  __int64 v127; // r8
  __int64 v128; // r12
  char v129; // cl
  int v130; // esi
  __int64 v131; // rdx
  __int64 v132; // rax
  __int32 v133; // r13d
  unsigned __int64 v134; // rsi
  __int64 *v135; // r14
  int v136; // r8d
  __int64 v137; // r9
  __int64 v138; // r14
  __int64 v139; // rdx
  __int64 v140; // rsi
  unsigned int v141; // ecx
  __int64 *v142; // rax
  __int64 v143; // rdi
  unsigned int v144; // eax
  char *v145; // rdx
  __m128i v146; // xmm3
  __int64 v147; // rbx
  char v148; // al
  _BYTE *v149; // r13
  unsigned int v150; // esi
  __int64 v151; // r14
  unsigned int v152; // edi
  __int64 v153; // rax
  __int64 v154; // rcx
  void **v155; // rax
  void *v156; // rdx
  __m128i *v157; // rsi
  __int64 v158; // rax
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // rax
  unsigned __int64 *v162; // rax
  int v163; // r9d
  __int64 v164; // rax
  __m128i *v165; // rax
  __m128i v166; // xmm6
  int v167; // r14d
  __int64 v168; // rcx
  __int64 v169; // rax
  __int64 v170; // rax
  __m128i *v171; // rax
  __m128i v172; // xmm2
  unsigned int v173; // edx
  __int64 *v174; // rax
  unsigned __int64 *v175; // rax
  __int64 v176; // rdi
  unsigned __int64 v177; // rax
  int v178; // eax
  unsigned int v179; // esi
  __int32 v180; // r10d
  int v181; // r11d
  __int64 v182; // r8
  unsigned int v183; // ecx
  __int64 *v184; // rax
  __int64 v185; // rdi
  __m128i v186; // rax
  __int64 v187; // rbx
  __int64 v188; // r13
  unsigned int v189; // ecx
  __int64 *v190; // rax
  __int64 v191; // r9
  int v192; // ecx
  unsigned int v193; // r14d
  int v194; // ecx
  __int64 v195; // rdi
  int v196; // r8d
  _QWORD *v197; // r10
  unsigned int v198; // edx
  _QWORD *v199; // rax
  __int64 v200; // rsi
  __int64 v201; // r12
  __int64 v202; // rax
  __int64 v203; // rdx
  __int64 v204; // rsi
  _QWORD *v205; // rax
  __int64 v206; // rcx
  int v207; // r8d
  int v208; // r9d
  __int64 v209; // rax
  __int64 v210; // rdx
  __int64 v211; // rbx
  __int64 v212; // r13
  unsigned int v213; // ecx
  __int64 *v214; // rax
  __int64 v215; // r9
  int v216; // ecx
  unsigned int v217; // r14d
  int v218; // ecx
  __int64 v219; // rdi
  int v220; // r8d
  _QWORD *v221; // r10
  unsigned int v222; // edx
  _QWORD *v223; // rax
  __int64 v224; // rsi
  __int64 v225; // r12
  __int64 v226; // rax
  __int64 v227; // rdx
  __int64 v228; // rsi
  _QWORD *v229; // rax
  __int64 v230; // rcx
  int v231; // r8d
  int v232; // r9d
  __int64 v233; // rax
  __int64 v234; // rdx
  __int64 v235; // rbx
  unsigned __int64 v236; // r9
  unsigned __int16 **v237; // r14
  __int64 v238; // r13
  _QWORD *v239; // rax
  unsigned __int64 v240; // rdx
  __int64 v241; // rsi
  __int64 v242; // rdi
  unsigned __int64 v243; // rbx
  int *v244; // r12
  int *v245; // rbx
  unsigned __int64 v246; // rdi
  unsigned __int64 v247; // rbx
  unsigned int *v248; // r12
  __int64 v249; // rdi
  _BYTE *v250; // rax
  __int64 v251; // rdi
  _BYTE *v252; // rax
  unsigned __int64 v253; // rdi
  unsigned int *v254; // rbx
  unsigned __int64 v255; // rdi
  _QWORD *v256; // rbx
  __int64 v257; // r12
  int v259; // eax
  int v260; // r11d
  __int64 v261; // r9
  unsigned int v262; // edx
  int v263; // edi
  __int64 v264; // r8
  __int64 *v265; // rsi
  int v266; // r8d
  int v267; // r8d
  __int64 v268; // r9
  int v269; // ecx
  __int64 v270; // r11
  __int64 v271; // rsi
  unsigned __int64 v272; // rax
  __int64 v273; // rdx
  __int64 v274; // rsi
  unsigned int v275; // ecx
  __int64 v276; // rdi
  unsigned __int64 v277; // rdx
  unsigned __int64 v278; // rdx
  unsigned __int64 v279; // rax
  int v280; // r13d
  __int64 *v281; // r11
  int v282; // edi
  int v283; // edi
  int v284; // r13d
  __int64 *v285; // rcx
  int v286; // eax
  int v287; // edi
  __int64 v288; // r11
  int v289; // edi
  int v290; // ecx
  int v291; // r10d
  int v292; // r10d
  __int64 v293; // r9
  __int64 v294; // rax
  __int64 v295; // rdx
  int v296; // r11d
  __int64 *v297; // r8
  int v298; // r10d
  int v299; // r10d
  __int64 v300; // rdx
  __int64 v301; // rcx
  int v302; // r11d
  __int64 *v303; // r11
  int v304; // edi
  int v305; // r10d
  int v306; // r10d
  __int64 v307; // r9
  int v308; // r11d
  __int64 v309; // rdx
  __int64 v310; // rax
  int v311; // r10d
  int v312; // r10d
  int v313; // r11d
  __int64 v314; // rdx
  __int64 v315; // rcx
  int v316; // r11d
  int v317; // r11d
  unsigned int v318; // edx
  int v319; // edi
  __int64 v320; // rsi
  __int64 *v321; // r13
  int v322; // edi
  int v323; // r8d
  _QWORD *v324; // rdi
  __int64 v325; // r8
  _QWORD *v326; // rax
  __int64 v327; // rcx
  __int64 v328; // rcx
  __int64 v329; // r12
  _BYTE *v330; // rax
  _BYTE *i; // rdx
  _QWORD *v332; // rax
  _QWORD *v333; // r8
  __int64 v334; // rdi
  _QWORD *v335; // r12
  __int64 v336; // r14
  _DWORD *v337; // rax
  __int64 v338; // rbx
  __int64 v339; // rax
  unsigned __int64 v340; // rbx
  int v341; // eax
  int v342; // r8d
  int v343; // r11d
  int v344; // r11d
  int v345; // edi
  unsigned int v346; // edx
  int v347; // eax
  int v348; // r8d
  int v349; // eax
  int v350; // esi
  __int64 *v351; // r9
  __int64 v352; // r8
  unsigned int v353; // edx
  __int64 v354; // rcx
  int v355; // eax
  int v356; // r13d
  __int64 v357; // r9
  unsigned int v358; // edi
  __int64 v359; // rdx
  __int64 *v360; // rsi
  int v361; // eax
  int v362; // r10d
  int v363; // eax
  int v364; // r13d
  __int64 v365; // r9
  unsigned int v366; // edi
  __int64 *v367; // rcx
  __int64 v368; // rsi
  __int64 *v369; // r11
  int v370; // edi
  int v371; // r8d
  int v372; // r8d
  __int64 v373; // r9
  int v374; // edx
  unsigned int v375; // r13d
  __int64 *v376; // rdi
  __int64 v377; // rsi
  int v378; // edi
  int v379; // eax
  int v380; // r10d
  __int64 v381; // rax
  int v382; // ecx
  __int64 *v383; // r13
  int v384; // r13d
  int v385; // edx
  __int32 v386; // [rsp+0h] [rbp-4D0h]
  __int32 v387; // [rsp+8h] [rbp-4C8h]
  unsigned __int64 v388; // [rsp+10h] [rbp-4C0h]
  __int64 v389; // [rsp+18h] [rbp-4B8h]
  __int64 v390; // [rsp+20h] [rbp-4B0h]
  __int64 v391; // [rsp+28h] [rbp-4A8h]
  __int64 v392; // [rsp+50h] [rbp-480h]
  __int64 v393; // [rsp+50h] [rbp-480h]
  int v394; // [rsp+50h] [rbp-480h]
  int v395; // [rsp+50h] [rbp-480h]
  int v396; // [rsp+50h] [rbp-480h]
  __int64 *v397; // [rsp+58h] [rbp-478h]
  size_t v398; // [rsp+58h] [rbp-478h]
  __int64 *v399; // [rsp+58h] [rbp-478h]
  __int64 v400; // [rsp+58h] [rbp-478h]
  __int64 v401; // [rsp+58h] [rbp-478h]
  size_t v402; // [rsp+58h] [rbp-478h]
  __int64 v403; // [rsp+58h] [rbp-478h]
  unsigned int v404; // [rsp+58h] [rbp-478h]
  unsigned int v405; // [rsp+58h] [rbp-478h]
  __int32 v406; // [rsp+68h] [rbp-468h]
  unsigned int v408; // [rsp+7Ch] [rbp-454h]
  __int64 v409; // [rsp+80h] [rbp-450h]
  int v410; // [rsp+80h] [rbp-450h]
  char v411; // [rsp+80h] [rbp-450h]
  __int64 v412; // [rsp+80h] [rbp-450h]
  _QWORD *v413; // [rsp+80h] [rbp-450h]
  unsigned __int32 v414; // [rsp+88h] [rbp-448h]
  __int32 v415; // [rsp+88h] [rbp-448h]
  int v416; // [rsp+88h] [rbp-448h]
  __int64 *v417; // [rsp+88h] [rbp-448h]
  __int64 v418; // [rsp+88h] [rbp-448h]
  int v419; // [rsp+88h] [rbp-448h]
  int v420; // [rsp+88h] [rbp-448h]
  unsigned int v421; // [rsp+88h] [rbp-448h]
  int v422; // [rsp+88h] [rbp-448h]
  __int64 v423; // [rsp+88h] [rbp-448h]
  unsigned __int64 v424; // [rsp+88h] [rbp-448h]
  __int64 v426; // [rsp+A0h] [rbp-430h]
  __int64 *v427; // [rsp+A0h] [rbp-430h]
  __int64 v428; // [rsp+A0h] [rbp-430h]
  __int64 v429; // [rsp+A0h] [rbp-430h]
  unsigned int v430; // [rsp+A0h] [rbp-430h]
  int v431; // [rsp+A0h] [rbp-430h]
  int v432; // [rsp+A0h] [rbp-430h]
  __int16 v433; // [rsp+A0h] [rbp-430h]
  void *s1[2]; // [rsp+B0h] [rbp-420h] BYREF
  __m128i si128; // [rsp+C0h] [rbp-410h] BYREF
  unsigned int *v436; // [rsp+D0h] [rbp-400h] BYREF
  __int64 v437; // [rsp+D8h] [rbp-3F8h]
  _BYTE v438[16]; // [rsp+E0h] [rbp-3F0h] BYREF
  unsigned __int16 *v439; // [rsp+F0h] [rbp-3E0h] BYREF
  __int64 v440; // [rsp+F8h] [rbp-3D8h]
  _BYTE v441[16]; // [rsp+100h] [rbp-3D0h] BYREF
  __int64 v442; // [rsp+110h] [rbp-3C0h] BYREF
  int v443; // [rsp+118h] [rbp-3B8h] BYREF
  int *v444; // [rsp+120h] [rbp-3B0h]
  int *v445; // [rsp+128h] [rbp-3A8h]
  int *v446; // [rsp+130h] [rbp-3A0h]
  __int64 v447; // [rsp+138h] [rbp-398h]
  __m128i v448; // [rsp+140h] [rbp-390h] BYREF
  __m128i v449; // [rsp+150h] [rbp-380h] BYREF
  __m128i v450; // [rsp+160h] [rbp-370h] BYREF
  __int64 v451; // [rsp+170h] [rbp-360h]
  __m128i v452; // [rsp+180h] [rbp-350h] BYREF
  __m128i v453; // [rsp+190h] [rbp-340h] BYREF
  __m128i v454; // [rsp+1A0h] [rbp-330h] BYREF
  __int64 v455; // [rsp+1B0h] [rbp-320h]
  __m128i v456; // [rsp+1C0h] [rbp-310h] BYREF
  __m128i v457; // [rsp+1D0h] [rbp-300h] BYREF
  __m128i v458; // [rsp+1E0h] [rbp-2F0h] BYREF
  __int64 v459; // [rsp+1F0h] [rbp-2E0h]
  int *v460; // [rsp+200h] [rbp-2D0h] BYREF
  __int64 v461; // [rsp+208h] [rbp-2C8h]
  _BYTE v462[64]; // [rsp+210h] [rbp-2C0h] BYREF
  char *v463; // [rsp+250h] [rbp-280h]
  __int64 v464; // [rsp+258h] [rbp-278h]
  char v465; // [rsp+260h] [rbp-270h] BYREF
  _BYTE *v466; // [rsp+2C0h] [rbp-210h] BYREF
  __int64 v467; // [rsp+2C8h] [rbp-208h]
  _BYTE v468[224]; // [rsp+2D0h] [rbp-200h] BYREF
  char *v469; // [rsp+3B0h] [rbp-120h] BYREF
  __int64 v470; // [rsp+3B8h] [rbp-118h]
  _BYTE v471[272]; // [rsp+3C0h] [rbp-110h] BYREF

  v3 = a1;
  v4 = *(_QWORD **)(a1 + 8);
  v5 = (*(__int64 (__fastcall **)(_QWORD *))(*v4 + 64LL))(v4);
  v6 = *a2;
  v391 = v5;
  v390 = v4[3];
  v7 = v4[1];
  v436 = (unsigned int *)v438;
  v389 = v7;
  v460 = (int *)v462;
  v461 = 0x400000000LL;
  v437 = 0x400000000LL;
  v467 = 0x400000000LL;
  v464 = 0x400000000LL;
  v470 = 0x400000000LL;
  v439 = (unsigned __int16 *)v441;
  v440 = 0x200000000LL;
  v445 = &v443;
  v446 = &v443;
  v456.m128i_i64[0] = (__int64)"__linear_memory";
  v466 = v468;
  v469 = v471;
  v463 = &v465;
  v443 = 0;
  v444 = 0;
  v447 = 0;
  v457.m128i_i16[0] = 259;
  v8 = sub_38BF510(v6, (__int64)&v456);
  v448 = *(__m128i *)(v8 + 40);
  if ( (*(_BYTE *)v8 & 4) != 0 )
  {
    v11 = *(__m128i *)(v8 - 8);
    v11.m128i_i64[0] += 16;
  }
  else
  {
    v11 = 0u;
  }
  v449 = v11;
  v12 = (unsigned int)v467;
  v450.m128i_i8[0] = 2;
  if ( (unsigned int)v467 >= HIDWORD(v467) )
  {
    sub_16CD150((__int64)&v466, v468, 0, 56, v9, v10);
    v12 = (unsigned int)v467;
  }
  v13 = _mm_load_si128(&v449);
  v14 = _mm_load_si128(&v450);
  v15 = (__m128i *)&v466[56 * v12];
  v16 = v451;
  *v15 = _mm_load_si128(&v448);
  v15[3].m128i_i64[0] = v16;
  v15[1] = v13;
  v15[2] = v14;
  LODWORD(v467) = v467 + 1;
  v456.m128i_i64[0] = (__int64)"__indirect_function_table";
  v457.m128i_i16[0] = 259;
  v17 = sub_38BF510(v6, (__int64)&v456);
  v452 = *(__m128i *)(v17 + 40);
  if ( (*(_BYTE *)v17 & 4) != 0 )
  {
    v20 = *(__m128i *)(v17 - 8);
    v20.m128i_i64[0] += 16;
  }
  else
  {
    v20 = 0u;
  }
  v453 = v20;
  v21 = (unsigned int)v467;
  v454.m128i_i8[0] = 1;
  v454.m128i_i8[4] = 112;
  if ( (unsigned int)v467 >= HIDWORD(v467) )
  {
    sub_16CD150((__int64)&v466, v468, 0, 56, v18, (int)v19);
    v21 = (unsigned int)v467;
  }
  v22 = _mm_load_si128(&v453);
  v23 = _mm_load_si128(&v454);
  v24 = (__m128i *)&v466[56 * v21];
  v25 = v455;
  *v24 = _mm_load_si128(&v452);
  v24[3].m128i_i64[0] = v25;
  v24[1] = v22;
  v24[2] = v23;
  LODWORD(v467) = v467 + 1;
  v26 = (__int64 *)a2[8];
  v27 = (__int64 *)a2[7];
  if ( v27 == v26 )
  {
    v35 = a2[4];
    v426 = a2[5];
    if ( v35 != v426 )
      goto LABEL_23;
    v408 = 0;
    goto LABEL_210;
  }
  do
  {
    while ( 1 )
    {
      v31 = *v27;
      v32 = *(unsigned int *)(*v27 + 32);
      if ( !(_DWORD)v32 )
        sub_391EDB0(v3, *v27, v25, v32, v18, (int)v19);
      v28 = *(_BYTE *)(v31 + 8);
      if ( (v28 & 1) != 0 )
        goto LABEL_17;
      v29 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v29 )
        goto LABEL_17;
      if ( (*(_BYTE *)(v31 + 9) & 0xC) == 8 )
      {
        v33 = *(_QWORD *)(v31 + 24);
        *(_BYTE *)(v31 + 8) = v28 | 4;
        v34 = (unsigned __int64)sub_38CE440(v33);
        *(_QWORD *)v31 = v34 | *(_QWORD *)v31 & 7LL;
        if ( v34 )
          break;
      }
      if ( !*(_BYTE *)(v31 + 38) )
      {
        v30 = *(_DWORD *)(v31 + 32);
        if ( v30 )
        {
          if ( v30 == 2 )
          {
            if ( *(_BYTE *)(v31 + 36) )
              sub_16BD130("undefined global symbol cannot be weak", 1u);
            v168 = *(_QWORD *)(v31 + 40);
            v456.m128i_i64[1] = *(_QWORD *)(v31 + 48);
            v169 = 0;
            v456.m128i_i64[0] = v168;
            if ( (*(_BYTE *)v31 & 4) != 0 )
            {
              v175 = *(unsigned __int64 **)(v31 - 8);
              v29 = *v175;
              v169 = (__int64)(v175 + 2);
            }
            v457.m128i_i64[0] = v169;
            v457.m128i_i64[1] = v29;
            v458.m128i_i8[0] = 3;
            v458.m128i_i16[2] = *(_WORD *)(v31 + 128);
            v170 = (unsigned int)v467;
            if ( (unsigned int)v467 >= HIDWORD(v467) )
            {
              sub_16CD150((__int64)&v466, v468, 0, 56, v18, (int)v19);
              v170 = (unsigned int)v467;
            }
            v171 = (__m128i *)&v466[56 * v170];
            *v171 = _mm_load_si128(&v456);
            v172 = _mm_load_si128(&v457);
            LODWORD(v467) = v467 + 1;
            v171[1] = v172;
            v171[2] = _mm_load_si128(&v458);
            v171[3].m128i_i64[0] = v459;
            v173 = *(_DWORD *)(v3 + 972);
            si128.m128i_i64[0] = v31;
            v430 = v173;
            *(_DWORD *)(v3 + 972) = v173 + 1;
            v174 = sub_391EB70(v3 + 160, si128.m128i_i64);
            v25 = v430;
            *((_DWORD *)v174 + 2) = v430;
          }
        }
        else
        {
          v160 = *(_QWORD *)(v31 + 40);
          v456.m128i_i64[1] = *(_QWORD *)(v31 + 48);
          v161 = 0;
          v456.m128i_i64[0] = v160;
          if ( (*(_BYTE *)v31 & 4) != 0 )
          {
            v162 = *(unsigned __int64 **)(v31 - 8);
            v29 = *v162;
            v161 = (__int64)(v162 + 2);
          }
          v457.m128i_i64[0] = v161;
          v457.m128i_i64[1] = v29;
          v458.m128i_i8[0] = 0;
          si128.m128i_i64[0] = v31;
          v458.m128i_i32[1] = *((_DWORD *)sub_391EB70(v3 + 96, si128.m128i_i64) + 2);
          v164 = (unsigned int)v467;
          if ( (unsigned int)v467 >= HIDWORD(v467) )
          {
            sub_16CD150((__int64)&v466, v468, 0, 56, (int)&si128, v163);
            v164 = (unsigned int)v467;
          }
          v165 = (__m128i *)&v466[56 * v164];
          *v165 = _mm_load_si128(&v456);
          v166 = _mm_load_si128(&v457);
          LODWORD(v467) = v467 + 1;
          v165[1] = v166;
          v165[2] = _mm_load_si128(&v458);
          v165[3].m128i_i64[0] = v459;
          v167 = *(_DWORD *)(v3 + 968);
          si128.m128i_i64[0] = v31;
          *(_DWORD *)(v3 + 968) = v167 + 1;
          *((_DWORD *)sub_391EB70(v3 + 160, si128.m128i_i64) + 2) = v167;
        }
      }
LABEL_17:
      if ( v26 == ++v27 )
        goto LABEL_22;
    }
    ++v27;
  }
  while ( v26 != v27 );
LABEL_22:
  v35 = a2[4];
  v426 = a2[5];
  if ( v426 == v35 )
  {
    v408 = 0;
    goto LABEL_60;
  }
LABEL_23:
  v408 = 0;
  v36 = (__int64 *)v35;
  while ( 2 )
  {
    while ( 2 )
    {
      v37 = *v36;
      v38 = *(_QWORD **)(*v36 + 152);
      v39 = *(void **)(*v36 + 160);
      s1[0] = v38;
      s1[1] = v39;
      if ( (unsigned __int64)v39 > 0xA
        && *v38 == 0x72615F74696E692ELL
        && *((_WORD *)v38 + 4) == 24946
        && *((_BYTE *)v38 + 10) == 121 )
      {
        goto LABEL_59;
      }
      v40 = *(_BYTE *)(v37 + 148);
      if ( (unsigned __int8)(v40 - 1) <= 1u )
        goto LABEL_59;
      if ( (unsigned __int8)(v40 - 3) > 7u && (unsigned __int8)(v40 - 13) > 5u )
      {
        si128 = _mm_load_si128((const __m128i *)s1);
        if ( (unsigned __int64)v39 > 0xF && !(*v38 ^ 0x5F6D6F747375632ELL | v38[1] ^ 0x2E6E6F6974636573LL) )
        {
          si128.m128i_i64[0] = (__int64)(v38 + 2);
          si128.m128i_i64[1] = (__int64)v39 - 16;
        }
        v149 = *(_BYTE **)(v37 + 8);
        if ( !v149 )
        {
LABEL_165:
          v456.m128i_i64[0] = v37;
          v157 = *(__m128i **)(v3 + 232);
          if ( v157 == *(__m128i **)(v3 + 240) )
          {
            ++v36;
            sub_3919780((unsigned __int64 *)(v3 + 224), v157, si128.m128i_i64, &v456);
            if ( (__int64 *)v426 == v36 )
              goto LABEL_60;
          }
          else
          {
            if ( v157 )
            {
              v158 = si128.m128i_i64[1];
              v159 = si128.m128i_i64[0];
              v157[1].m128i_i64[0] = v37;
              v157->m128i_i64[1] = v158;
              v157->m128i_i64[0] = v159;
              v157[1].m128i_i64[1] = 0xFFFFFFFF00000000LL;
              v157 = *(__m128i **)(v3 + 232);
            }
            ++v36;
            *(_QWORD *)(v3 + 232) = v157 + 2;
            if ( (__int64 *)v426 == v36 )
              goto LABEL_60;
          }
          continue;
        }
        v150 = *(_DWORD *)(v3 + 184);
        v151 = (__int64)(*(_QWORD *)(v3 + 232) - *(_QWORD *)(v3 + 224)) >> 5;
        if ( v150 )
        {
          v19 = (const void *)(v150 - 1);
          v18 = *(_QWORD *)(v3 + 168);
          v152 = (unsigned int)v19 & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
          v153 = v18 + 16LL * v152;
          v154 = *(_QWORD *)v153;
          if ( v149 == *(_BYTE **)v153 )
            goto LABEL_159;
          v419 = 1;
          v288 = 0;
          while ( v154 != -8 )
          {
            if ( !v288 && v154 == -16 )
              v288 = v153;
            v152 = (unsigned int)v19 & (v419 + v152);
            LODWORD(v18) = v419 + 1;
            v153 = *(_QWORD *)(v3 + 168) + 16LL * v152;
            v154 = *(_QWORD *)v153;
            if ( v149 == *(_BYTE **)v153 )
              goto LABEL_159;
            ++v419;
          }
          v289 = *(_DWORD *)(v3 + 176);
          if ( v288 )
            v153 = v288;
          ++*(_QWORD *)(v3 + 160);
          v290 = v289 + 1;
          if ( 4 * (v289 + 1) < 3 * v150 )
          {
            LODWORD(v18) = v150 >> 3;
            if ( v150 - *(_DWORD *)(v3 + 180) - v290 > v150 >> 3 )
              goto LABEL_338;
            v421 = ((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4);
            sub_391E830(v3 + 160, v150);
            v343 = *(_DWORD *)(v3 + 184);
            if ( !v343 )
            {
LABEL_619:
              ++*(_DWORD *)(v3 + 176);
              BUG();
            }
            v344 = v343 - 1;
            v19 = *(const void **)(v3 + 168);
            v345 = 1;
            v346 = v344 & v421;
            v290 = *(_DWORD *)(v3 + 176) + 1;
            v320 = 0;
            v153 = (__int64)v19 + 16 * (v344 & v421);
            v18 = *(_QWORD *)v153;
            if ( v149 == *(_BYTE **)v153 )
              goto LABEL_338;
            while ( v18 != -8 )
            {
              if ( !v320 && v18 == -16 )
                v320 = v153;
              v346 = v344 & (v345 + v346);
              v153 = (__int64)v19 + 16 * v346;
              v18 = *(_QWORD *)v153;
              if ( v149 == *(_BYTE **)v153 )
                goto LABEL_338;
              ++v345;
            }
            goto LABEL_395;
          }
        }
        else
        {
          ++*(_QWORD *)(v3 + 160);
        }
        sub_391E830(v3 + 160, 2 * v150);
        v316 = *(_DWORD *)(v3 + 184);
        if ( !v316 )
          goto LABEL_619;
        v317 = v316 - 1;
        v19 = *(const void **)(v3 + 168);
        v318 = v317 & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
        v290 = *(_DWORD *)(v3 + 176) + 1;
        v153 = (__int64)v19 + 16 * v318;
        v18 = *(_QWORD *)v153;
        if ( v149 == *(_BYTE **)v153 )
          goto LABEL_338;
        v319 = 1;
        v320 = 0;
        while ( v18 != -8 )
        {
          if ( v18 == -16 && !v320 )
            v320 = v153;
          v318 = v317 & (v319 + v318);
          v153 = (__int64)v19 + 16 * v318;
          v18 = *(_QWORD *)v153;
          if ( v149 == *(_BYTE **)v153 )
            goto LABEL_338;
          ++v319;
        }
LABEL_395:
        if ( v320 )
          v153 = v320;
LABEL_338:
        *(_DWORD *)(v3 + 176) = v290;
        if ( *(_QWORD *)v153 != -8 )
          --*(_DWORD *)(v3 + 180);
        *(_QWORD *)v153 = v149;
        *(_DWORD *)(v153 + 8) = 0;
LABEL_159:
        *(_DWORD *)(v153 + 8) = v151;
        if ( (*v149 & 4) != 0 )
        {
          v155 = (void **)*((_QWORD *)v149 - 1);
          v156 = *v155;
          if ( s1[1] != *v155 || v156 && memcmp(s1[0], v155 + 2, (size_t)v156) )
          {
LABEL_161:
            v456.m128i_i64[0] = (__int64)"section name and begin symbol should match: ";
            v456.m128i_i64[1] = (__int64)s1;
            v457.m128i_i16[0] = 1283;
            sub_16BCFB0((__int64)&v456, 1u);
          }
        }
        else if ( s1[1] )
        {
          goto LABEL_161;
        }
        goto LABEL_165;
      }
      break;
    }
    v414 = *(_DWORD *)(v3 + 704);
    v41 = v414;
    v42 = *(_DWORD *)(v37 + 24)
        * ((*(unsigned int *)(v37 + 24) + (unsigned __int64)v408 - 1)
         / *(unsigned int *)(v37 + 24));
    if ( v414 >= *(_DWORD *)(v3 + 708) )
    {
      sub_391A490(v3 + 696);
      v41 = *(_DWORD *)(v3 + 704);
    }
    v43 = *(_QWORD *)(v3 + 696);
    v44 = v43 + ((unsigned __int64)v41 << 6);
    if ( v44 )
    {
      *(_OWORD *)(v44 + 32) = 0;
      *(_QWORD *)(v44 + 40) = v44 + 56;
      *(_OWORD *)(v44 + 48) = 0;
      *(_OWORD *)v44 = 0;
      *(_QWORD *)(v44 + 48) = 0x400000000LL;
      *(_OWORD *)(v44 + 16) = 0;
      v43 = *(_QWORD *)(v3 + 696);
      v41 = *(_DWORD *)(v3 + 704);
    }
    v45 = v41 + 1;
    v46 = _mm_load_si128((const __m128i *)s1);
    *(_DWORD *)(v3 + 704) = v45;
    v45 <<= 6;
    v47 = v43 + v45 - 64;
    *(_DWORD *)(v47 + 24) = v42;
    *(_QWORD *)v47 = v37;
    *(__m128i *)(v47 + 8) = v46;
    sub_3919C00(v47 + 40, v37, v44, v45, v18, v19);
    v48 = *(_DWORD *)(v47 + 48);
    *(_QWORD *)(v47 + 28) = *(unsigned int *)(v37 + 24);
    v408 = v42 + v48;
    *(_DWORD *)(v37 + 192) = v414;
    v49 = *(_BYTE **)(v37 + 176);
    if ( !v49 )
      goto LABEL_59;
    if ( (*v49 & 4) != 0 )
    {
      v50 = (__int64 *)*((_QWORD *)v49 - 1);
      v51 = *v50;
      v52 = (__int64)(v50 + 2);
    }
    else
    {
      v51 = 0;
      v52 = 0;
    }
    v53 = v444;
    v456.m128i_i64[0] = v52;
    v456.m128i_i64[1] = v51;
    v54 = &v443;
    if ( !v444 )
      goto LABEL_54;
    v397 = v36;
    v55 = (const void *)v52;
    v392 = v3;
    v56 = v51;
    while ( 2 )
    {
      while ( 2 )
      {
        v58 = *((_QWORD *)v53 + 5);
        v59 = (const void *)*((_QWORD *)v53 + 4);
        if ( v58 <= v56 )
        {
          if ( v58 )
          {
            v57 = memcmp(v59, v55, *((_QWORD *)v53 + 5));
            if ( v57 )
              goto LABEL_46;
          }
          if ( v58 == v56 )
            break;
          goto LABEL_41;
        }
        if ( !v56 )
          break;
        v57 = memcmp(v59, v55, v56);
        if ( !v57 )
        {
LABEL_41:
          if ( v58 >= v56 )
            break;
          goto LABEL_42;
        }
LABEL_46:
        if ( v57 < 0 )
        {
LABEL_42:
          v53 = (int *)*((_QWORD *)v53 + 3);
          if ( !v53 )
            goto LABEL_48;
          continue;
        }
        break;
      }
      v54 = v53;
      v53 = (int *)*((_QWORD *)v53 + 2);
      if ( v53 )
        continue;
      break;
    }
LABEL_48:
    v19 = v55;
    v60 = v56;
    v36 = v397;
    v3 = v392;
    if ( v54 == &v443 )
      goto LABEL_54;
    v61 = *((_QWORD *)v54 + 5);
    v62 = (const void *)*((_QWORD *)v54 + 4);
    if ( v60 > v61 )
    {
      if ( !v61 )
        goto LABEL_55;
      v402 = v60;
      v63 = memcmp(v19, v62, *((_QWORD *)v54 + 5));
      v60 = v402;
      if ( v63 )
      {
LABEL_311:
        if ( v63 >= 0 )
          goto LABEL_55;
      }
      else
      {
LABEL_53:
        if ( v60 >= v61 )
          goto LABEL_55;
      }
LABEL_54:
      si128.m128i_i64[0] = (__int64)&v456;
      v54 = (int *)sub_391B5A0(&v442, v54, (const __m128i **)&si128);
      goto LABEL_55;
    }
    if ( v60 )
    {
      v398 = v60;
      v63 = memcmp(v19, v62, v60);
      v60 = v398;
      if ( v63 )
        goto LABEL_311;
    }
    if ( v60 != v61 )
      goto LABEL_53;
LABEL_55:
    si128.m128i_i32[0] = 0;
    si128.m128i_i32[1] = v414;
    v64 = (_BYTE *)*((_QWORD *)v54 + 7);
    if ( v64 == *((_BYTE **)v54 + 8) )
    {
      sub_39195F0((__int64)(v54 + 12), v64, &si128);
    }
    else
    {
      if ( v64 )
      {
        *(_QWORD *)v64 = si128.m128i_i64[0];
        v64 = (_BYTE *)*((_QWORD *)v54 + 7);
      }
      *((_QWORD *)v54 + 7) = v64 + 8;
    }
LABEL_59:
    if ( (__int64 *)v426 != ++v36 )
      continue;
    break;
  }
LABEL_60:
  v427 = (__int64 *)a2[8];
  if ( v427 == (__int64 *)a2[7] )
    goto LABEL_210;
  v65 = (__int64 *)a2[7];
  while ( 2 )
  {
    while ( 2 )
    {
      v67 = *v65;
      if ( (*(_BYTE *)(*v65 + 8) & 1) != 0 && ((*(_BYTE *)v67 & 4) == 0 || !**(_QWORD **)(v67 - 8))
        || (*(_BYTE *)(v67 + 9) & 0xC) == 8 )
      {
        goto LABEL_65;
      }
      if ( *(_BYTE *)(v67 + 38) )
      {
        if ( (*(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_65;
        v66 = *(_DWORD *)(v67 + 32);
        if ( v66 )
        {
LABEL_63:
          if ( v66 != 1 )
          {
            if ( v66 == 2 && (*(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              sub_16BD130("don't yet support defined globals", 1u);
LABEL_65:
            if ( v427 == ++v65 )
              goto LABEL_111;
            continue;
          }
          if ( (*(_BYTE *)(*v65 + 8) & 1) != 0 )
          {
            v176 = *(_QWORD *)(v67 + 136);
            if ( !v176 || (*(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              goto LABEL_65;
          }
          else
          {
            if ( (*(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              goto LABEL_65;
            v176 = *(_QWORD *)(v67 + 136);
            if ( !v176 )
            {
              v186.m128i_i64[0] = sub_3913870((_BYTE *)*v65);
              v457.m128i_i16[0] = 1283;
              si128 = v186;
              v456.m128i_i64[0] = (__int64)"data symbols must have a size set with .size: ";
              v456.m128i_i64[1] = (__int64)&si128;
              sub_16BCFB0((__int64)&v456, 1u);
            }
          }
          v456.m128i_i64[0] = 0;
          if ( !sub_38CF2A0(v176, &v456, a3) )
            sub_16BD130(".size expression must be evaluatable", 1u);
          v177 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v177 )
          {
            if ( (*(_BYTE *)(v67 + 9) & 0xC) != 8
              || (*(_BYTE *)(v67 + 8) |= 4u,
                  v177 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v67 + 24)),
                  *(_QWORD *)v67 = v177 | *(_QWORD *)v67 & 7LL,
                  !v177) )
            {
              BUG();
            }
          }
          v416 = *(_DWORD *)(*(_QWORD *)(v177 + 24) + 192LL);
          v178 = sub_38D0440(a3, v67);
          v179 = *(_DWORD *)(v3 + 216);
          v180 = v456.m128i_i32[0];
          v181 = v178;
          v400 = v3 + 192;
          if ( v179 )
          {
            v182 = *(_QWORD *)(v3 + 200);
            v183 = (v179 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
            v184 = (__int64 *)(v182 + 24LL * v183);
            v185 = *v184;
            if ( v67 == *v184 )
            {
LABEL_198:
              *((_DWORD *)v184 + 3) = v181;
              *((_DWORD *)v184 + 4) = v180;
              *((_DWORD *)v184 + 2) = v416;
              goto LABEL_65;
            }
            v394 = 1;
            v321 = 0;
            while ( v185 != -8 )
            {
              if ( !v321 && v185 == -16 )
                v321 = v184;
              v183 = (v179 - 1) & (v394 + v183);
              v184 = (__int64 *)(v182 + 24LL * v183);
              v185 = *v184;
              if ( v67 == *v184 )
                goto LABEL_198;
              ++v394;
            }
            v322 = *(_DWORD *)(v3 + 208);
            if ( v321 )
              v184 = v321;
            ++*(_QWORD *)(v3 + 192);
            v323 = v322 + 1;
            if ( 4 * (v322 + 1) < 3 * v179 )
            {
              if ( v179 - *(_DWORD *)(v3 + 212) - v323 <= v179 >> 3 )
              {
                v386 = v180;
                v396 = v181;
                sub_39205F0(v400, v179);
                v363 = *(_DWORD *)(v3 + 216);
                if ( !v363 )
                {
LABEL_617:
                  ++*(_DWORD *)(v3 + 208);
                  BUG();
                }
                v364 = v363 - 1;
                v365 = *(_QWORD *)(v3 + 200);
                v181 = v396;
                v180 = v386;
                v366 = (v363 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
                v323 = *(_DWORD *)(v3 + 208) + 1;
                v367 = 0;
                v184 = (__int64 *)(v365 + 24LL * v366);
                v368 = *v184;
                if ( v67 != *v184 )
                {
                  while ( v368 != -8 )
                  {
                    if ( !v367 && v368 == -16 )
                      v367 = v184;
                    v366 = v364 & (v66 + v366);
                    v184 = (__int64 *)(v365 + 24LL * v366);
                    v368 = *v184;
                    if ( v67 == *v184 )
                      goto LABEL_404;
                    ++v66;
                  }
                  if ( v367 )
                    v184 = v367;
                }
              }
              goto LABEL_404;
            }
          }
          else
          {
            ++*(_QWORD *)(v3 + 192);
          }
          v387 = v180;
          v395 = v181;
          sub_39205F0(v400, 2 * v179);
          v355 = *(_DWORD *)(v3 + 216);
          if ( !v355 )
            goto LABEL_617;
          v356 = v355 - 1;
          v357 = *(_QWORD *)(v3 + 200);
          v181 = v395;
          v180 = v387;
          v323 = *(_DWORD *)(v3 + 208) + 1;
          v358 = (v355 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v184 = (__int64 *)(v357 + 24LL * v358);
          v359 = *v184;
          if ( v67 != *v184 )
          {
            v360 = 0;
            while ( v359 != -8 )
            {
              if ( v359 == -16 && !v360 )
                v360 = v184;
              v358 = v356 & (v66 + v358);
              v184 = (__int64 *)(v357 + 24LL * v358);
              v359 = *v184;
              if ( v67 == *v184 )
                goto LABEL_404;
              ++v66;
            }
            if ( v360 )
              v184 = v360;
          }
LABEL_404:
          *(_DWORD *)(v3 + 208) = v323;
          if ( *v184 != -8 )
            --*(_DWORD *)(v3 + 212);
          *v184 = v67;
          v184[1] = 0;
          *((_DWORD *)v184 + 4) = 0;
          goto LABEL_198;
        }
      }
      else
      {
        v66 = *(_DWORD *)(v67 + 32);
        if ( v66 )
          goto LABEL_63;
        if ( (*(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_65;
      }
      break;
    }
    v68 = *(_QWORD *)(v67 + 24);
    if ( v68 )
      sub_16BD130("function sections must contain one function each", 1u);
    if ( !*(_QWORD *)(v67 + 136) )
      sub_16BD130("function symbols must have a size set with .size", 1u);
    v69 = *(_DWORD *)(v3 + 120);
    v415 = *(_DWORD *)(v3 + 968) + v461;
    if ( v69 )
    {
      v70 = *(_QWORD *)(v3 + 104);
      v71 = (v69 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v72 = (__int64 *)(v70 + 16LL * v71);
      v73 = *v72;
      if ( v67 == *v72 )
      {
        v74 = *((_DWORD *)v72 + 2);
        goto LABEL_78;
      }
      v284 = 1;
      v285 = 0;
      while ( v73 != -8 )
      {
        if ( v285 || v73 != -16 )
          v72 = v285;
        v382 = v284 + 1;
        v71 = (v69 - 1) & (v284 + v71);
        v383 = (__int64 *)(v70 + 16LL * v71);
        v73 = *v383;
        if ( v67 == *v383 )
        {
          v74 = *((_DWORD *)v383 + 2);
          goto LABEL_78;
        }
        v284 = v382;
        v285 = v72;
        v72 = (__int64 *)(v70 + 16LL * v71);
      }
      if ( !v285 )
        v285 = v72;
      v286 = *(_DWORD *)(v3 + 112);
      ++*(_QWORD *)(v3 + 96);
      v287 = v286 + 1;
      if ( 4 * (v286 + 1) < 3 * v69 )
      {
        if ( v69 - *(_DWORD *)(v3 + 116) - v287 <= v69 >> 3 )
        {
          v404 = ((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4);
          sub_391E830(v3 + 96, v69);
          v305 = *(_DWORD *)(v3 + 120);
          if ( !v305 )
          {
LABEL_618:
            ++*(_DWORD *)(v3 + 112);
            BUG();
          }
          v306 = v305 - 1;
          v307 = *(_QWORD *)(v3 + 104);
          v297 = 0;
          v308 = 1;
          v287 = *(_DWORD *)(v3 + 112) + 1;
          LODWORD(v309) = v306 & v404;
          v285 = (__int64 *)(v307 + 16LL * (v306 & v404));
          v310 = *v285;
          if ( v67 != *v285 )
          {
            while ( v310 != -8 )
            {
              if ( !v297 && v310 == -16 )
                v297 = v285;
              v309 = v306 & (unsigned int)(v309 + v308);
              v285 = (__int64 *)(v307 + 16 * v309);
              v310 = *v285;
              if ( v67 == *v285 )
                goto LABEL_328;
              ++v308;
            }
            goto LABEL_381;
          }
        }
        goto LABEL_328;
      }
    }
    else
    {
      ++*(_QWORD *)(v3 + 96);
    }
    sub_391E830(v3 + 96, 2 * v69);
    v291 = *(_DWORD *)(v3 + 120);
    if ( !v291 )
      goto LABEL_618;
    v292 = v291 - 1;
    v293 = *(_QWORD *)(v3 + 104);
    LODWORD(v294) = v292 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
    v287 = *(_DWORD *)(v3 + 112) + 1;
    v285 = (__int64 *)(v293 + 16LL * (unsigned int)v294);
    v295 = *v285;
    if ( v67 != *v285 )
    {
      v296 = 1;
      v297 = 0;
      while ( v295 != -8 )
      {
        if ( !v297 && v295 == -16 )
          v297 = v285;
        v294 = v292 & (unsigned int)(v294 + v296);
        v285 = (__int64 *)(v293 + 16 * v294);
        v295 = *v285;
        if ( v67 == *v285 )
          goto LABEL_328;
        ++v296;
      }
LABEL_381:
      if ( v297 )
        v285 = v297;
    }
LABEL_328:
    *(_DWORD *)(v3 + 112) = v287;
    if ( *v285 != -8 )
      --*(_DWORD *)(v3 + 116);
    *v285 = v67;
    v74 = 0;
    *((_DWORD *)v285 + 2) = 0;
LABEL_78:
    v75 = *(_DWORD *)(v3 + 184);
    v76 = _mm_cvtsi32_si128(v74).m128i_u64[0];
    if ( v75 )
    {
      LODWORD(v77) = v75 - 1;
      v78 = *(__int64 **)(v3 + 168);
      v79 = (v75 - 1) & (((unsigned int)v67 >> 4) ^ ((unsigned int)v67 >> 9));
      v80 = &v78[2 * v79];
      v81 = *v80;
      if ( v67 == *v80 )
        goto LABEL_80;
      v280 = 1;
      v281 = 0;
      while ( v81 != -8 )
      {
        if ( v81 != -16 || v281 )
          v80 = v281;
        v79 = v77 & (v280 + v79);
        v81 = v78[2 * v79];
        if ( v67 == v81 )
        {
          v80 = &v78[2 * v79];
          goto LABEL_80;
        }
        ++v280;
        v281 = v80;
        v80 = &v78[2 * v79];
      }
      v282 = *(_DWORD *)(v3 + 176);
      if ( v281 )
        v80 = v281;
      ++*(_QWORD *)(v3 + 160);
      v283 = v282 + 1;
      if ( 4 * v283 < 3 * v75 )
      {
        LODWORD(v78) = v75 >> 3;
        if ( v75 - *(_DWORD *)(v3 + 180) - v283 <= v75 >> 3 )
        {
          v405 = ((unsigned int)v67 >> 4) ^ ((unsigned int)v67 >> 9);
          sub_391E830(v3 + 160, v75);
          v311 = *(_DWORD *)(v3 + 184);
          if ( !v311 )
            goto LABEL_619;
          v312 = v311 - 1;
          v78 = 0;
          v77 = *(_QWORD *)(v3 + 168);
          v313 = 1;
          LODWORD(v314) = v312 & v405;
          v283 = *(_DWORD *)(v3 + 176) + 1;
          v80 = (__int64 *)(v77 + 16LL * (v312 & v405));
          v315 = *v80;
          if ( v67 != *v80 )
          {
            while ( v315 != -8 )
            {
              if ( v315 == -16 && !v78 )
                v78 = v80;
              v314 = v312 & (unsigned int)(v314 + v313);
              v80 = (__int64 *)(v77 + 16 * v314);
              v315 = *v80;
              if ( v67 == *v80 )
                goto LABEL_319;
              ++v313;
            }
            goto LABEL_387;
          }
        }
        goto LABEL_319;
      }
    }
    else
    {
      ++*(_QWORD *)(v3 + 160);
    }
    sub_391E830(v3 + 160, 2 * v75);
    v298 = *(_DWORD *)(v3 + 184);
    if ( !v298 )
      goto LABEL_619;
    v299 = v298 - 1;
    v77 = *(_QWORD *)(v3 + 168);
    LODWORD(v300) = v299 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
    v283 = *(_DWORD *)(v3 + 176) + 1;
    v80 = (__int64 *)(v77 + 16LL * (unsigned int)v300);
    v301 = *v80;
    if ( v67 != *v80 )
    {
      v302 = 1;
      v78 = 0;
      while ( v301 != -8 )
      {
        if ( !v78 && v301 == -16 )
          v78 = v80;
        v300 = v299 & (unsigned int)(v300 + v302);
        v80 = (__int64 *)(v77 + 16 * v300);
        v301 = *v80;
        if ( v67 == *v80 )
          goto LABEL_319;
        ++v302;
      }
LABEL_387:
      if ( v78 )
        v80 = v78;
    }
LABEL_319:
    *(_DWORD *)(v3 + 176) = v283;
    if ( *v80 != -8 )
      --*(_DWORD *)(v3 + 180);
    *v80 = v67;
    *((_DWORD *)v80 + 2) = 0;
LABEL_80:
    *((_DWORD *)v80 + 2) = v415;
    v82 = (unsigned int)v461;
    if ( (unsigned int)v461 >= HIDWORD(v461) )
    {
      sub_16CD150((__int64)&v460, v462, 0, 16, (int)v78, v77);
      v82 = (unsigned int)v461;
    }
    v83 = (unsigned __int64 *)&v460[4 * v82];
    v83[1] = v67;
    *v83 = v76;
    LODWORD(v461) = v461 + 1;
    v84 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v84 )
    {
      if ( (*(_BYTE *)(v67 + 9) & 0xC) != 8
        || (*(_BYTE *)(v67 + 8) |= 4u,
            v84 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v67 + 24)),
            *(_QWORD *)v67 = v84 | *(_QWORD *)v67 & 7LL,
            !v84) )
      {
        BUG();
      }
    }
    v85 = *(_BYTE **)(*(_QWORD *)(v84 + 24) + 176LL);
    if ( !v85 )
      goto LABEL_65;
    v86 = 0;
    if ( (*v85 & 4) != 0 )
    {
      v87 = (__int64 *)*((_QWORD *)v85 - 1);
      v68 = *v87;
      v86 = (__int64)(v87 + 2);
    }
    v88 = v444;
    v456.m128i_i64[0] = v86;
    v456.m128i_i64[1] = v68;
    v89 = &v443;
    if ( !v444 )
      goto LABEL_106;
    v399 = v65;
    v90 = (const void *)v86;
    v393 = v3;
    v91 = v68;
    while ( 2 )
    {
      while ( 2 )
      {
        v93 = *((_QWORD *)v88 + 5);
        v94 = (const void *)*((_QWORD *)v88 + 4);
        if ( v91 >= v93 )
        {
          if ( v93 )
          {
            v92 = memcmp(v94, v90, *((_QWORD *)v88 + 5));
            if ( v92 )
              goto LABEL_98;
          }
          if ( v91 == v93 )
            break;
          goto LABEL_93;
        }
        if ( !v91 )
          break;
        v92 = memcmp(v94, v90, v91);
        if ( !v92 )
        {
LABEL_93:
          if ( v91 <= v93 )
            break;
          goto LABEL_94;
        }
LABEL_98:
        if ( v92 < 0 )
        {
LABEL_94:
          v88 = (int *)*((_QWORD *)v88 + 3);
          if ( !v88 )
            goto LABEL_100;
          continue;
        }
        break;
      }
      v89 = v88;
      v88 = (int *)*((_QWORD *)v88 + 2);
      if ( v88 )
        continue;
      break;
    }
LABEL_100:
    v95 = v90;
    v96 = v91;
    v65 = v399;
    v3 = v393;
    if ( v89 == &v443 )
      goto LABEL_106;
    v97 = *((_QWORD *)v89 + 5);
    v98 = (const void *)*((_QWORD *)v89 + 4);
    if ( v96 > v97 )
    {
      if ( !v97 )
        goto LABEL_107;
      v99 = memcmp(v95, v98, *((_QWORD *)v89 + 5));
      if ( v99 )
      {
LABEL_206:
        if ( v99 >= 0 )
          goto LABEL_107;
      }
      else
      {
LABEL_105:
        if ( v96 >= v97 )
          goto LABEL_107;
      }
LABEL_106:
      si128.m128i_i64[0] = (__int64)&v456;
      v89 = (int *)sub_391B5A0(&v442, v89, (const __m128i **)&si128);
      goto LABEL_107;
    }
    if ( v96 )
    {
      v99 = memcmp(v95, v98, v96);
      if ( v99 )
        goto LABEL_206;
    }
    if ( v96 != v97 )
      goto LABEL_105;
LABEL_107:
    si128.m128i_i32[0] = 1;
    si128.m128i_i32[1] = v415;
    v100 = (_BYTE *)*((_QWORD *)v89 + 7);
    if ( v100 == *((_BYTE **)v89 + 8) )
    {
      sub_39195F0((__int64)(v89 + 12), v100, &si128);
      goto LABEL_65;
    }
    if ( v100 )
    {
      *(_QWORD *)v100 = si128.m128i_i64[0];
      v100 = (_BYTE *)*((_QWORD *)v89 + 7);
    }
    ++v65;
    *((_QWORD *)v89 + 7) = v100 + 8;
    if ( v427 != v65 )
      continue;
    break;
  }
LABEL_111:
  v101 = a2[8];
  if ( v101 == a2[7] )
    goto LABEL_210;
  v428 = v3 + 192;
  v102 = a2[7];
  while ( 2 )
  {
    while ( 2 )
    {
      v113 = *(_QWORD *)v102;
      if ( (*(_BYTE *)(*(_QWORD *)v102 + 9LL) & 0xC) != 8 )
      {
LABEL_118:
        v102 += 8;
        if ( v101 == v102 )
          goto LABEL_126;
        continue;
      }
      break;
    }
    v114 = *(_QWORD *)(v113 + 24);
    v115 = *(_DWORD *)(v113 + 32);
    *(_BYTE *)(v113 + 8) |= 4u;
    v116 = *(_QWORD *)(v114 + 24);
    if ( v115 )
    {
      if ( v115 != 1 )
        sub_16BD130("don't yet support global aliases", 1u);
      v103 = *(_DWORD *)(v3 + 216);
      v104 = *(__int64 **)(v3 + 200);
      if ( v103 )
      {
        v105 = v103 - 1;
        v106 = 1;
        LODWORD(v107) = (v103 - 1) & (((unsigned int)v116 >> 9) ^ ((unsigned int)v116 >> 4));
        v108 = &v104[3 * (unsigned int)v107];
        v109 = *v108;
        if ( v116 != *v108 )
        {
          while ( v109 != -8 )
          {
            v107 = v105 & ((_DWORD)v107 + v106);
            v108 = &v104[3 * v107];
            v109 = *v108;
            if ( v116 == *v108 )
              goto LABEL_116;
            ++v106;
          }
          v108 = &v104[3 * v103];
        }
LABEL_116:
        v110 = (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4)) & v105;
        v111 = &v104[3 * v110];
        v112 = *v111;
        if ( v113 == *v111 )
          goto LABEL_117;
        v420 = 1;
        v303 = 0;
        while ( v112 != -8 )
        {
          if ( v112 != -16 || v303 )
            v111 = v303;
          LODWORD(v110) = v105 & (v420 + v110);
          v112 = v104[3 * (unsigned int)v110];
          if ( v113 == v112 )
          {
            v111 = &v104[3 * (unsigned int)v110];
            goto LABEL_117;
          }
          ++v420;
          v303 = v111;
          v111 = &v104[3 * (unsigned int)v110];
        }
        v304 = *(_DWORD *)(v3 + 208);
        if ( v303 )
          v111 = v303;
        ++*(_QWORD *)(v3 + 192);
        v263 = v304 + 1;
        if ( 4 * v263 < 3 * v103 )
        {
          if ( v103 - *(_DWORD *)(v3 + 212) - v263 <= v103 >> 3 )
          {
            v412 = v101;
            sub_39205F0(v428, v103);
            v349 = *(_DWORD *)(v3 + 216);
            if ( !v349 )
              goto LABEL_617;
            v350 = v349 - 1;
            v351 = 0;
            v352 = *(_QWORD *)(v3 + 200);
            v101 = v412;
            v353 = (v349 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
            v263 = *(_DWORD *)(v3 + 208) + 1;
            v111 = (__int64 *)(v352 + 24LL * v353);
            v354 = *v111;
            if ( v113 != *v111 )
            {
              while ( v354 != -8 )
              {
                if ( v354 == -16 && !v351 )
                  v351 = v111;
                v353 = v350 & (v115 + v353);
                v111 = (__int64 *)(v352 + 24LL * v353);
                v354 = *v111;
                if ( v113 == *v111 )
                  goto LABEL_375;
                ++v115;
              }
              if ( v351 )
                v111 = v351;
            }
          }
          goto LABEL_375;
        }
        v104 = v108;
      }
      else
      {
        ++*(_QWORD *)(v3 + 192);
      }
      v409 = v101;
      v417 = v104;
      sub_39205F0(v428, 2 * v103);
      v259 = *(_DWORD *)(v3 + 216);
      if ( !v259 )
        goto LABEL_617;
      v260 = v259 - 1;
      v261 = *(_QWORD *)(v3 + 200);
      v101 = v409;
      v262 = (v259 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
      v263 = *(_DWORD *)(v3 + 208) + 1;
      v108 = v417;
      v111 = (__int64 *)(v261 + 24LL * v262);
      v264 = *v111;
      if ( v113 != *v111 )
      {
        v265 = 0;
        while ( v264 != -8 )
        {
          if ( v264 == -16 && !v265 )
            v265 = v111;
          v262 = v260 & (v115 + v262);
          v111 = (__int64 *)(v261 + 24LL * v262);
          v264 = *v111;
          if ( v113 == *v111 )
          {
            v108 = v417;
            goto LABEL_375;
          }
          ++v115;
        }
        v108 = v417;
        if ( v265 )
          v111 = v265;
      }
LABEL_375:
      *(_DWORD *)(v3 + 208) = v263;
      if ( *v111 != -8 )
        --*(_DWORD *)(v3 + 212);
      *v111 = v113;
      v111[1] = 0;
      *((_DWORD *)v111 + 4) = 0;
LABEL_117:
      v111[1] = v108[1];
      *((_DWORD *)v111 + 4) = *((_DWORD *)v108 + 4);
      goto LABEL_118;
    }
    v117 = *(_DWORD *)(v3 + 184);
    v118 = *(_QWORD *)(v3 + 168);
    if ( !v117 )
    {
      v123 = *(_DWORD *)(v118 + 8);
      ++*(_QWORD *)(v3 + 160);
      goto LABEL_278;
    }
    v119 = v117 - 1;
    v120 = (v117 - 1) & (((unsigned int)v116 >> 9) ^ ((unsigned int)v116 >> 4));
    v121 = (__int64 *)(v118 + 16LL * v120);
    v122 = *v121;
    if ( v116 == *v121 )
    {
LABEL_123:
      v123 = *((_DWORD *)v121 + 2);
    }
    else
    {
      v378 = 1;
      while ( v122 != -8 )
      {
        v384 = v378 + 1;
        v120 = v119 & (v378 + v120);
        v121 = (__int64 *)(v118 + 16LL * v120);
        v122 = *v121;
        if ( v116 == *v121 )
          goto LABEL_123;
        v378 = v384;
      }
      v123 = *(_DWORD *)(v118 + 16LL * v117 + 8);
    }
    v124 = ((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4);
    v125 = v124 & v119;
    v126 = (__int64 *)(v118 + 16LL * (v124 & v119));
    v127 = *v126;
    if ( v113 != *v126 )
    {
      v422 = 1;
      v369 = 0;
      while ( v127 != -8 )
      {
        if ( !v369 && v127 == -16 )
          v369 = v126;
        v125 = v119 & (v422 + v125);
        v126 = (__int64 *)(v118 + 16LL * v125);
        v127 = *v126;
        if ( v113 == *v126 )
          goto LABEL_125;
        ++v422;
      }
      v370 = *(_DWORD *)(v3 + 176);
      if ( v369 )
        v126 = v369;
      ++*(_QWORD *)(v3 + 160);
      v269 = v370 + 1;
      if ( 4 * (v370 + 1) < 3 * v117 )
      {
        if ( v117 - *(_DWORD *)(v3 + 180) - v269 > v117 >> 3 )
          goto LABEL_280;
        v423 = v101;
        sub_391E830(v3 + 160, v117);
        v371 = *(_DWORD *)(v3 + 184);
        if ( !v371 )
          goto LABEL_619;
        v372 = v371 - 1;
        v373 = *(_QWORD *)(v3 + 168);
        v374 = 1;
        v375 = v372 & v124;
        v101 = v423;
        v269 = *(_DWORD *)(v3 + 176) + 1;
        v376 = 0;
        v126 = (__int64 *)(v373 + 16LL * v375);
        v377 = *v126;
        if ( v113 == *v126 )
          goto LABEL_280;
        while ( v377 != -8 )
        {
          if ( v377 == -16 && !v376 )
            v376 = v126;
          v375 = v372 & (v374 + v375);
          v126 = (__int64 *)(v373 + 16LL * v375);
          v377 = *v126;
          if ( v113 == *v126 )
            goto LABEL_280;
          ++v374;
        }
        goto LABEL_505;
      }
LABEL_278:
      v418 = v101;
      sub_391E830(v3 + 160, 2 * v117);
      v266 = *(_DWORD *)(v3 + 184);
      if ( !v266 )
        goto LABEL_619;
      v267 = v266 - 1;
      v268 = *(_QWORD *)(v3 + 168);
      v101 = v418;
      v269 = *(_DWORD *)(v3 + 176) + 1;
      v270 = v267 & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
      v126 = (__int64 *)(v268 + 16 * v270);
      v271 = *v126;
      if ( v113 == *v126 )
        goto LABEL_280;
      v385 = 1;
      v376 = 0;
      while ( v271 != -8 )
      {
        if ( v271 == -16 && !v376 )
          v376 = v126;
        LODWORD(v270) = v267 & (v385 + v270);
        v126 = (__int64 *)(v268 + 16LL * (unsigned int)v270);
        v271 = *v126;
        if ( v113 == *v126 )
          goto LABEL_280;
        ++v385;
      }
LABEL_505:
      if ( v376 )
        v126 = v376;
LABEL_280:
      *(_DWORD *)(v3 + 176) = v269;
      if ( *v126 != -8 )
        --*(_DWORD *)(v3 + 180);
      *v126 = v113;
      *((_DWORD *)v126 + 2) = 0;
    }
LABEL_125:
    v102 += 8;
    *((_DWORD *)v126 + 2) = v123;
    if ( v101 != v102 )
      continue;
    break;
  }
LABEL_126:
  v128 = a2[7];
  v429 = a2[8];
  if ( v429 != v128 )
  {
    while ( 2 )
    {
      v147 = *(_QWORD *)v128;
      v148 = *(_BYTE *)(*(_QWORD *)v128 + 9LL);
      if ( (v148 & 2) != 0 )
      {
        v129 = *(_BYTE *)(v147 + 8);
LABEL_132:
        v131 = *(_QWORD *)v147;
        v132 = *(_QWORD *)v147;
LABEL_133:
        v133 = *(unsigned __int8 *)(v147 + 36);
        if ( *(_BYTE *)(v147 + 37) )
          v133 |= 4u;
        v134 = v131 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v129 & 0x10) == 0 )
        {
          if ( !v134 )
          {
            if ( (*(_BYTE *)(v147 + 9) & 0xC) != 8 )
            {
LABEL_139:
              v133 |= 0x10u;
              v132 = v131;
              goto LABEL_140;
            }
            *(_BYTE *)(v147 + 8) |= 4u;
            v278 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v147 + 24));
            v132 = v278 | *(_QWORD *)v147 & 7LL;
            *(_QWORD *)v147 = v132;
            v134 = v132 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v278 )
              goto LABEL_136;
          }
          v133 |= 2u;
        }
LABEL_136:
        if ( !v134 )
        {
          if ( (*(_BYTE *)(v147 + 9) & 0xC) != 8 )
          {
            v131 = *(_QWORD *)v147;
            goto LABEL_139;
          }
          *(_BYTE *)(v147 + 8) |= 4u;
          v277 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v147 + 24));
          v132 = v277 | *(_QWORD *)v147 & 7LL;
          *(_QWORD *)v147 = v132;
          if ( !v277 )
          {
            v131 = v132;
            goto LABEL_139;
          }
        }
LABEL_140:
        if ( (v132 & 4) != 0 )
        {
          v135 = *(__int64 **)(v147 - 8);
          v136 = *(_DWORD *)(v147 + 32);
          v137 = *v135;
          v138 = (__int64)(v135 + 2);
          if ( v136 != 1 )
            goto LABEL_142;
LABEL_284:
          if ( (v132 & 0xFFFFFFFFFFFFFFF8LL) != 0
            || (*(_BYTE *)(v147 + 9) & 0xC) == 8
            && (*(_BYTE *)(v147 + 8) |= 4u,
                v401 = v137,
                v410 = v136,
                v272 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v147 + 24)),
                v136 = v410,
                v137 = v401,
                *(_QWORD *)v147 = v272 | *(_QWORD *)v147 & 7LL,
                v272) )
          {
            v273 = *(unsigned int *)(v3 + 216);
            v274 = *(_QWORD *)(v3 + 200);
            if ( (_DWORD)v273 )
            {
              v275 = (v273 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
              v142 = (__int64 *)(v274 + 24LL * v275);
              v276 = *v142;
              if ( v147 == *v142 )
                goto LABEL_289;
              v379 = 1;
              while ( v276 != -8 )
              {
                v380 = v379 + 1;
                v381 = ((_DWORD)v273 - 1) & (v275 + v379);
                v275 = v381;
                v142 = (__int64 *)(v274 + 24 * v381);
                v276 = *v142;
                if ( v147 == *v142 )
                  goto LABEL_289;
                v379 = v380;
              }
            }
            v142 = (__int64 *)(v274 + 24 * v273);
LABEL_289:
            v458.m128i_i64[1] = v142[1];
            LODWORD(v459) = *((_DWORD *)v142 + 4);
            goto LABEL_144;
          }
        }
        else
        {
          v136 = *(_DWORD *)(v147 + 32);
          v137 = 0;
          v138 = 0;
          if ( v136 == 1 )
            goto LABEL_284;
LABEL_142:
          v139 = *(unsigned int *)(v3 + 184);
          v140 = *(_QWORD *)(v3 + 168);
          if ( (_DWORD)v139 )
          {
            v141 = (v139 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
            v142 = (__int64 *)(v140 + 16LL * v141);
            v143 = *v142;
            if ( v147 == *v142 )
              goto LABEL_144;
            v361 = 1;
            while ( v143 != -8 )
            {
              v362 = v361 + 1;
              v141 = (v139 - 1) & (v361 + v141);
              v142 = (__int64 *)(v140 + 16LL * v141);
              v143 = *v142;
              if ( v147 == *v142 )
                goto LABEL_144;
              v361 = v362;
            }
          }
          v142 = (__int64 *)(v140 + 16 * v139);
LABEL_144:
          v406 = *((_DWORD *)v142 + 2);
        }
        v144 = v470;
        *(_DWORD *)(v147 + 16) = v470;
        if ( v144 >= HIDWORD(v470) )
        {
          v403 = v137;
          v411 = v136;
          sub_16CD150((__int64)&v469, v471, 0, 56, v136, v137);
          v144 = v470;
          v137 = v403;
          LOBYTE(v136) = v411;
        }
        v145 = &v469[56 * v144];
        if ( v145 )
        {
          v456.m128i_i64[0] = v138;
          v456.m128i_i64[1] = v137;
          v146 = _mm_load_si128(&v456);
          v458.m128i_i32[2] = v406;
          v457.m128i_i8[0] = v136;
          v457.m128i_i32[1] = v133;
          v457.m128i_i64[1] = 0;
          v458.m128i_i64[0] = 0;
          *(__m128i *)v145 = v146;
          *((__m128i *)v145 + 1) = _mm_load_si128(&v457);
          *((__m128i *)v145 + 2) = _mm_load_si128(&v458);
          *((_QWORD *)v145 + 6) = v459;
          v144 = v470;
        }
        LODWORD(v470) = v144 + 1;
LABEL_150:
        v128 += 8;
        if ( v429 == v128 )
          goto LABEL_210;
        continue;
      }
      break;
    }
    if ( *(_BYTE *)(v147 + 38) )
    {
      if ( (*(_QWORD *)v147 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        if ( (v148 & 0xC) != 8
          || (*(_BYTE *)(v147 + 8) |= 4u,
              v279 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v147 + 24)),
              *(_QWORD *)v147 = v279 | *(_QWORD *)v147 & 7LL,
              !v279) )
        {
LABEL_294:
          *(_DWORD *)(v147 + 16) = -1;
          goto LABEL_150;
        }
      }
    }
    v129 = *(_BYTE *)(v147 + 8);
    if ( (v129 & 1) != 0 )
    {
      v131 = *(_QWORD *)v147;
      v132 = *(_QWORD *)v147;
      if ( (*(_QWORD *)v147 & 4) == 0 || !**(_QWORD **)(v147 - 8) )
        goto LABEL_294;
      v130 = *(_DWORD *)(v147 + 32);
      if ( v130 == 1 )
      {
        if ( *(_QWORD *)(v147 + 136) )
          goto LABEL_133;
        goto LABEL_294;
      }
    }
    else
    {
      v130 = *(_DWORD *)(v147 + 32);
    }
    if ( v130 != 3 )
      goto LABEL_132;
    goto LABEL_294;
  }
LABEL_210:
  v187 = *(_QWORD *)(v3 + 32);
  v188 = *(_QWORD *)(v3 + 40);
  if ( v188 != v187 )
  {
    while ( 1 )
    {
      if ( (unsigned int)(*(_DWORD *)(v187 + 24) - 1) > 1 )
        goto LABEL_215;
      v201 = *(_QWORD *)(v187 + 8);
      if ( (*(_BYTE *)(v201 + 9) & 0xC) == 8 )
      {
        v202 = *(_QWORD *)(v201 + 24);
        *(_BYTE *)(v201 + 8) |= 4u;
        v201 = *(_QWORD *)(v202 + 24);
      }
      v203 = *(unsigned int *)(v3 + 184);
      v204 = *(_QWORD *)(v3 + 168);
      if ( !(_DWORD)v203 )
        goto LABEL_220;
      v189 = (v203 - 1) & (((unsigned int)v201 >> 9) ^ ((unsigned int)v201 >> 4));
      v190 = (__int64 *)(v204 + 16LL * v189);
      v191 = *v190;
      if ( *v190 != v201 )
        break;
LABEL_213:
      v192 = *(_DWORD *)(v3 + 152);
      v193 = *((_DWORD *)v190 + 2);
      v456.m128i_i64[0] = v201;
      if ( v192 )
      {
        v194 = v192 - 1;
        v195 = *(_QWORD *)(v3 + 136);
        v196 = 1;
        v197 = 0;
        v198 = v194 & (((unsigned int)v201 >> 9) ^ ((unsigned int)v201 >> 4));
        v199 = (_QWORD *)(v195 + 16LL * v198);
        v200 = *v199;
        if ( v201 != *v199 )
        {
          while ( v200 != -8 )
          {
            if ( v200 != -16 || v197 )
              v199 = v197;
            v198 = v194 & (v196 + v198);
            v200 = *(_QWORD *)(v195 + 16LL * v198);
            if ( v201 == v200 )
              goto LABEL_215;
            ++v196;
            v197 = v199;
            v199 = (_QWORD *)(v195 + 16LL * v198);
          }
          if ( !v197 )
            v197 = v199;
          goto LABEL_222;
        }
LABEL_215:
        v187 += 40;
        if ( v188 == v187 )
          goto LABEL_225;
      }
      else
      {
        v197 = 0;
LABEL_222:
        v431 = v437;
        v205 = sub_391E9F0(v3 + 128, &v456, v197);
        v208 = v431 + 1;
        *v205 = v456.m128i_i64[0];
        *((_DWORD *)v205 + 2) = v431 + 1;
        v209 = (unsigned int)v437;
        if ( (unsigned int)v437 >= HIDWORD(v437) )
        {
          sub_16CD150((__int64)&v436, v438, 0, 4, v207, v208);
          v209 = (unsigned int)v437;
        }
        v210 = (__int64)v436;
        v187 += 40;
        v436[v209] = v193;
        LODWORD(v437) = v437 + 1;
        sub_391EDB0(v3, v201, v210, v206, v207, v208);
        if ( v188 == v187 )
          goto LABEL_225;
      }
    }
    v341 = 1;
    while ( v191 != -8 )
    {
      v342 = v341 + 1;
      v189 = (v203 - 1) & (v341 + v189);
      v190 = (__int64 *)(v204 + 16LL * v189);
      v191 = *v190;
      if ( v201 == *v190 )
        goto LABEL_213;
      v341 = v342;
    }
LABEL_220:
    v190 = (__int64 *)(v204 + 16 * v203);
    goto LABEL_213;
  }
LABEL_225:
  v211 = *(_QWORD *)(v3 + 64);
  v212 = *(_QWORD *)(v3 + 72);
  if ( v212 != v211 )
  {
    while ( 1 )
    {
      if ( (unsigned int)(*(_DWORD *)(v211 + 24) - 1) > 1 )
        goto LABEL_230;
      v225 = *(_QWORD *)(v211 + 8);
      if ( (*(_BYTE *)(v225 + 9) & 0xC) == 8 )
      {
        v226 = *(_QWORD *)(v225 + 24);
        *(_BYTE *)(v225 + 8) |= 4u;
        v225 = *(_QWORD *)(v226 + 24);
      }
      v227 = *(unsigned int *)(v3 + 184);
      v228 = *(_QWORD *)(v3 + 168);
      if ( !(_DWORD)v227 )
        goto LABEL_235;
      v213 = (v227 - 1) & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
      v214 = (__int64 *)(v228 + 16LL * v213);
      v215 = *v214;
      if ( *v214 != v225 )
        break;
LABEL_228:
      v216 = *(_DWORD *)(v3 + 152);
      v217 = *((_DWORD *)v214 + 2);
      v456.m128i_i64[0] = v225;
      if ( v216 )
      {
        v218 = v216 - 1;
        v219 = *(_QWORD *)(v3 + 136);
        v220 = 1;
        v221 = 0;
        v222 = v218 & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
        v223 = (_QWORD *)(v219 + 16LL * v222);
        v224 = *v223;
        if ( *v223 != v225 )
        {
          while ( v224 != -8 )
          {
            if ( v221 || v224 != -16 )
              v223 = v221;
            v222 = v218 & (v220 + v222);
            v224 = *(_QWORD *)(v219 + 16LL * v222);
            if ( v225 == v224 )
              goto LABEL_230;
            ++v220;
            v221 = v223;
            v223 = (_QWORD *)(v219 + 16LL * v222);
          }
          if ( !v221 )
            v221 = v223;
          goto LABEL_237;
        }
LABEL_230:
        v211 += 40;
        if ( v212 == v211 )
          goto LABEL_240;
      }
      else
      {
        v221 = 0;
LABEL_237:
        v432 = v437;
        v229 = sub_391E9F0(v3 + 128, &v456, v221);
        v232 = v432 + 1;
        *v229 = v456.m128i_i64[0];
        *((_DWORD *)v229 + 2) = v432 + 1;
        v233 = (unsigned int)v437;
        if ( (unsigned int)v437 >= HIDWORD(v437) )
        {
          sub_16CD150((__int64)&v436, v438, 0, 4, v231, v232);
          v233 = (unsigned int)v437;
        }
        v234 = (__int64)v436;
        v211 += 40;
        v436[v233] = v217;
        LODWORD(v437) = v437 + 1;
        sub_391EDB0(v3, v225, v234, v230, v231, v232);
        if ( v212 == v211 )
          goto LABEL_240;
      }
    }
    v347 = 1;
    while ( v215 != -8 )
    {
      v348 = v347 + 1;
      v213 = (v227 - 1) & (v347 + v213);
      v214 = (__int64 *)(v228 + 16LL * v213);
      v215 = *v214;
      if ( v225 == *v214 )
        goto LABEL_228;
      v347 = v348;
    }
LABEL_235:
    v214 = (__int64 *)(v228 + 16 * v227);
    goto LABEL_228;
  }
LABEL_240:
  v235 = a2[4];
  if ( a2[5] != v235 )
  {
    v236 = v388;
    v237 = &v439;
    v238 = a2[5];
    do
    {
      v239 = *(_QWORD **)v235;
      v240 = *(_QWORD *)(*(_QWORD *)v235 + 160LL);
      if ( v240 > 0xA )
      {
        v241 = v239[19];
        if ( *(_QWORD *)v241 == 0x72615F696E69662ELL && *(_WORD *)(v241 + 8) == 24946 && *(_BYTE *)(v241 + 10) == 121 )
          sub_16BD130(".fini_array sections are unsupported", 1u);
        if ( *(_QWORD *)v241 == 0x72615F74696E692ELL && *(_WORD *)(v241 + 8) == 24946 && *(_BYTE *)(v241 + 10) == 121 )
        {
          v324 = v239 + 12;
          if ( v239 + 12 != (_QWORD *)(v239[12] & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v325 = v239[13];
            if ( v324 == (_QWORD *)v325 )
              goto LABEL_560;
            v326 = (_QWORD *)v239[13];
            v327 = 0;
            do
            {
              v326 = (_QWORD *)v326[1];
              ++v327;
            }
            while ( v324 != v326 );
            if ( v327 != 3 )
LABEL_560:
              sub_16BD130("only one .init_array section fragment supported", 1u);
            if ( *(_BYTE *)(v325 + 16) != 1 || (v328 = *(_QWORD *)(v325 + 8), *(_BYTE *)(v328 + 16)) )
              sub_16BD130(".init_array section should be aligned", 1u);
            if ( ((*(_BYTE *)(*(_QWORD *)(v3 + 24) + 8LL) & 1) == 0 ? 4 : 8) != *(_DWORD *)(v328 + 48) )
              sub_16BD130(".init_array section should be aligned for pointers", 1u);
            v329 = *(_QWORD *)(v328 + 8);
            if ( *(_BYTE *)(v329 + 17) || *(_BYTE *)(v329 + 16) != 1 )
              sub_16BD130("only data supported in .init_array section", 1u);
            if ( v240 == 11 )
            {
              v433 = -1;
            }
            else
            {
              if ( *(_BYTE *)(v241 + 11) != 46 )
                sub_16BD130(".init_array section priority should start with '.'", 1u);
              v424 = v236;
              if ( sub_16D2B80(v241 + 12, v240 - 12, 0xAu, (unsigned __int64 *)&v456)
                || (v456.m128i_i64[0] & 0xFFFFFFFFFFFF0000LL) != 0 )
              {
                sub_16BD130("invalid .init_array section priority", 1u);
              }
              v433 = v456.m128i_i16[0];
              v236 = v424;
            }
            v330 = *(_BYTE **)(v329 + 64);
            for ( i = &v330[*(unsigned int *)(v329 + 72)]; i != v330; ++v330 )
            {
              if ( *v330 )
                sub_16BD130("non-symbolic data in .init_array section", 1u);
            }
            v332 = *(_QWORD **)(v329 + 112);
            v333 = &v332[3 * *(unsigned int *)(v329 + 120)];
            if ( v332 != v333 )
            {
              v334 = (__int64)v237;
              v335 = *(_QWORD **)(v329 + 112);
              v336 = v235;
              do
              {
                v337 = (_DWORD *)*v335;
                if ( *(_DWORD *)*v335 != 2 )
                  sub_16BD130("fixups in .init_array should be symbol references", 1u);
                if ( *((_WORD *)v337 + 8) != 111 )
                  sub_16BD130("symbols in .init_array should be for functions", 1u);
                v338 = *(unsigned int *)(*((_QWORD *)v337 + 3) + 16LL);
                if ( (_DWORD)v338 == -1 )
                  sub_16BD130("symbols in .init_array should exist in symbtab", 1u);
                LOWORD(v236) = v433;
                v339 = (unsigned int)v440;
                v340 = (unsigned int)v236 | (unsigned __int64)(v338 << 32);
                v236 = v340;
                if ( (unsigned int)v440 >= HIDWORD(v440) )
                {
                  v413 = v333;
                  sub_16CD150(v334, v441, 0, 8, (int)v333, v340);
                  v339 = (unsigned int)v440;
                  v236 = v340;
                  v333 = v413;
                }
                v335 += 3;
                *(_QWORD *)&v439[4 * v339] = v340;
                LODWORD(v440) = v440 + 1;
              }
              while ( v333 != v335 );
              v235 = v336;
              v237 = (unsigned __int16 **)v334;
            }
          }
        }
      }
      v235 += 8;
    }
    while ( v238 != v235 );
  }
  sub_16E7EE0(*(_QWORD *)(v3 + 8), byte_45301BF, 4u);
  v242 = *(_QWORD *)(v3 + 8);
  v456.m128i_i32[0] = (unsigned int)(*(_DWORD *)(v3 + 16) - 1) < 2 ? 1 : 0x1000000;
  sub_16E7EE0(v242, v456.m128i_i8, 4u);
  sub_391BD70(v3, *(_QWORD *)(v3 + 344), *(unsigned int *)(v3 + 352));
  sub_391BFA0(v3, (__int64)v466, (unsigned int)v467, v408, v437);
  v243 = (unsigned int)v461;
  if ( (_DWORD)v461 )
  {
    v244 = v460;
    sub_391B370(v3, (__int64)&v456, 3);
    sub_391A6C0(v243, *(_QWORD *)(v3 + 8), 0);
    v245 = &v244[4 * v243];
    do
    {
      v246 = *v244;
      v244 += 4;
      sub_391A6C0(v246, *(_QWORD *)(v3 + 8), 0);
    }
    while ( v245 != v244 );
    sub_3919EA0(v3, &v456);
  }
  sub_391BB50(v3);
  v247 = (unsigned int)v437;
  if ( (_DWORD)v437 )
  {
    v248 = v436;
    sub_391B370(v3, (__int64)&v456, 9);
    sub_391A6C0(1u, *(_QWORD *)(v3 + 8), 0);
    sub_391A6C0(0, *(_QWORD *)(v3 + 8), 0);
    v249 = *(_QWORD *)(v3 + 8);
    v250 = *(_BYTE **)(v249 + 24);
    if ( (unsigned __int64)v250 >= *(_QWORD *)(v249 + 16) )
    {
      sub_16E7DE0(v249, 65);
    }
    else
    {
      *(_QWORD *)(v249 + 24) = v250 + 1;
      *v250 = 65;
    }
    sub_391A7B0(1, *(_QWORD *)(v3 + 8), 0);
    v251 = *(_QWORD *)(v3 + 8);
    v252 = *(_BYTE **)(v251 + 24);
    if ( (unsigned __int64)v252 >= *(_QWORD *)(v251 + 16) )
    {
      sub_16E7DE0(v251, 11);
    }
    else
    {
      *(_QWORD *)(v251 + 24) = v252 + 1;
      *v252 = 11;
    }
    v253 = v247;
    v254 = &v248[v247];
    sub_391A6C0(v253, *(_QWORD *)(v3 + 8), 0);
    do
    {
      v255 = *v248++;
      sub_391A6C0(v255, *(_QWORD *)(v3 + 8), 0);
    }
    while ( v254 != v248 );
    sub_3919EA0(v3, &v456);
  }
  sub_3921380(v3, (__int64)a2, a3, (__int64)v460, (unsigned int)v461);
  sub_39215D0(v3);
  sub_3921850(v3, (__int64)a2, a3);
  sub_391CCA0(v3, v469, (unsigned int)v470, v439, (unsigned int)v440, &v442);
  sub_391F970(
    v3,
    *(_DWORD *)(v3 + 56),
    "CODE",
    4,
    *(_QWORD *)(v3 + 32),
    0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(v3 + 40) - *(_QWORD *)(v3 + 32)) >> 3));
  sub_391F970(
    v3,
    *(_DWORD *)(v3 + 88),
    "DATA",
    4,
    *(_QWORD *)(v3 + 64),
    0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(v3 + 72) - *(_QWORD *)(v3 + 64)) >> 3));
  sub_391FDB0(v3);
  v256 = *(_QWORD **)(v3 + 8);
  v257 = (*(__int64 (__fastcall **)(_QWORD *))(*v256 + 64LL))(v256) + v256[3] - v256[1] - (v390 - v389) - v391;
  sub_3919FA0((__int64)&v442, v444);
  if ( v439 != (unsigned __int16 *)v441 )
    _libc_free((unsigned __int64)v439);
  if ( v469 != v471 )
    _libc_free((unsigned __int64)v469);
  if ( v466 != v468 )
    _libc_free((unsigned __int64)v466);
  if ( v436 != (unsigned int *)v438 )
    _libc_free((unsigned __int64)v436);
  if ( v460 != (int *)v462 )
    _libc_free((unsigned __int64)v460);
  return v257;
}
