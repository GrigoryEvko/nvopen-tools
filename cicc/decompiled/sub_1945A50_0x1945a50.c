// Function: sub_1945A50
// Address: 0x1945a50
//
__int64 __fastcall sub_1945A50(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 *v15; // rax
  __int64 v16; // r12
  char *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  int v20; // r8d
  int v21; // r9d
  __int64 i; // rbx
  __int64 v23; // rsi
  __int64 *v24; // rdi
  __int64 v25; // rdx
  __int64 *v26; // rax
  __int64 v27; // rax
  __m128 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 *v31; // r9
  char *v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rax
  int v35; // ecx
  __int64 *v36; // rax
  __int64 v37; // rdi
  int v38; // ecx
  __int64 v39; // r11
  unsigned int v40; // esi
  __int64 *v41; // rax
  __int64 v42; // r10
  int v43; // r15d
  double v44; // xmm4_8
  double v45; // xmm5_8
  int v46; // r10d
  __int64 *v47; // r8
  __int64 v48; // rcx
  __int64 *v49; // rax
  __int64 v50; // rsi
  int v51; // r8d
  int v52; // r9d
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // r14
  __int64 v56; // r15
  __int64 v57; // rdi
  __int64 v58; // rdi
  bool v59; // zf
  int v60; // r13d
  _QWORD *v61; // rbx
  _QWORD *v62; // r12
  unsigned int v63; // eax
  __int64 v64; // rax
  int v65; // eax
  __int64 v66; // rdx
  _QWORD *v67; // rax
  _QWORD *v68; // rdx
  unsigned int v69; // eax
  __int64 v70; // rcx
  _QWORD *v71; // rsi
  __int64 v72; // rax
  _QWORD *v73; // rdi
  __int64 v74; // rax
  __int64 v75; // r12
  bool v76; // al
  __int64 v77; // r14
  unsigned __int64 v78; // r15
  _QWORD *v79; // r13
  unsigned __int64 v80; // r12
  _QWORD *v81; // r15
  unsigned __int8 v82; // al
  unsigned __int64 v83; // rdx
  __int64 v84; // rcx
  unsigned __int64 *v85; // r12
  unsigned __int64 v86; // rsi
  __int64 *v87; // rax
  __int64 *v88; // rdi
  __int64 v89; // rcx
  __int64 v90; // rdx
  __int64 *v91; // r12
  unsigned __int64 v92; // rdx
  __int64 *v93; // rax
  _BOOL4 v94; // r8d
  __int64 v95; // rax
  __int64 *v96; // r12
  unsigned __int64 v97; // rdx
  __int64 *v98; // rax
  _BOOL4 v99; // r13d
  __int64 v100; // rax
  __int64 *v101; // r15
  __int64 v102; // r13
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rbx
  __int64 v106; // r12
  __int64 v107; // r14
  __int64 v108; // rdx
  unsigned __int64 v109; // rax
  char v110; // dl
  __int64 v111; // rsi
  __int64 v112; // rax
  __int64 v113; // r9
  unsigned int v114; // edi
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rcx
  __int64 v118; // rax
  __int64 v119; // rdx
  _QWORD *v120; // r8
  __int64 v121; // rcx
  unsigned __int64 v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rax
  unsigned int v125; // eax
  _BYTE *v126; // rdi
  unsigned int v127; // r12d
  _QWORD *v128; // rbx
  _QWORD *v129; // r13
  __int64 v130; // rax
  __int64 *v132; // rax
  __int64 v133; // rax
  unsigned int v134; // ecx
  _QWORD *v135; // rdi
  unsigned int v136; // eax
  __int64 v137; // rax
  unsigned __int64 v138; // rax
  unsigned __int64 v139; // rax
  int v140; // ebx
  __int64 v141; // r12
  _QWORD *v142; // rax
  _QWORD *k; // rdx
  __int64 v144; // rax
  _QWORD *v145; // rax
  int v146; // eax
  _QWORD *v147; // rsi
  __int64 v148; // rsi
  unsigned __int64 v149; // r14
  __int64 v150; // r13
  __int64 *v151; // r8
  __int64 v152; // r15
  unsigned __int64 v153; // rax
  unsigned __int64 v154; // r12
  __int64 *v155; // rax
  __int64 *v156; // rsi
  __int64 v157; // rcx
  __int64 v158; // rdx
  unsigned __int64 *m; // r10
  unsigned __int64 v160; // rsi
  __int64 *v161; // rax
  __int64 *v162; // r9
  __int64 v163; // rcx
  __int64 v164; // rdx
  unsigned __int64 *v165; // r8
  unsigned __int64 v166; // rsi
  __int64 *v167; // rax
  __int64 *v168; // rdi
  __int64 *v169; // rdx
  unsigned __int64 v170; // rcx
  __int64 *v171; // rax
  _BOOL4 v172; // r9d
  __int64 v173; // rax
  int v174; // r15d
  __int64 v175; // rax
  __int64 v176; // r14
  __int64 v177; // rbx
  __int64 v178; // rax
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // r15
  __int64 v182; // r12
  __int64 v183; // rbx
  __int64 v184; // r13
  __int64 v185; // rbx
  __int64 *v186; // r14
  __int64 v187; // r13
  __int64 v188; // r12
  __int64 v189; // rax
  __int64 v190; // rax
  signed __int64 v191; // rsi
  __int64 v192; // rax
  signed __int64 v193; // rsi
  __int64 v194; // rsi
  _QWORD *v195; // r15
  __int64 v196; // rdx
  __int64 v197; // rcx
  unsigned __int64 v198; // rsi
  __int64 v199; // rcx
  unsigned int v200; // eax
  unsigned __int64 *v201; // rdi
  __int64 v202; // rax
  __int64 v203; // rsi
  double v204; // xmm4_8
  double v205; // xmm5_8
  int v206; // ebx
  unsigned int v207; // eax
  _QWORD *v208; // rdi
  unsigned __int64 v209; // rdx
  _QWORD *v210; // rax
  _QWORD *j; // rdx
  __int64 v212; // rax
  unsigned __int8 *v213; // rdx
  __int64 *v214; // r8
  signed __int64 v215; // rax
  __int64 v216; // r12
  __int64 v217; // rbx
  _QWORD *v218; // r13
  __int64 v219; // rax
  _QWORD *v220; // rax
  __int64 v221; // r14
  _QWORD **v222; // rax
  __int64 *v223; // rax
  __int64 v224; // rsi
  unsigned __int64 *v225; // r12
  __int64 v226; // rax
  unsigned __int64 v227; // rcx
  __int64 v228; // rsi
  unsigned __int8 *v229; // rsi
  __int64 v230; // rax
  __int64 v231; // rbx
  __int64 v232; // rax
  _QWORD *v233; // r15
  __int64 v234; // rbx
  __int64 v235; // rax
  __int64 v236; // rax
  _QWORD *v237; // r15
  __int64 v238; // rax
  __int64 v239; // rax
  __int64 v240; // rax
  __int64 **v241; // rdx
  __int64 v242; // rax
  unsigned int v243; // r12d
  __int64 v244; // rdx
  unsigned int v245; // eax
  __int64 v246; // rdx
  __int64 v247; // rsi
  unsigned int v248; // eax
  int v249; // ebx
  __int64 v250; // r15
  __int64 *v251; // rbx
  __int64 v252; // rax
  __int64 v253; // rcx
  __int64 v254; // r15
  __int64 v255; // rsi
  unsigned __int8 *v256; // rsi
  __int64 **v257; // rdx
  __int64 v258; // rsi
  int v259; // edi
  __int64 *v260; // rbx
  __int64 v261; // rax
  __int64 v262; // rcx
  __int64 v263; // rsi
  __int64 v264; // r13
  __int64 v265; // rax
  __int64 v266; // rdi
  unsigned int v267; // eax
  char v268; // dl
  __int64 v269; // rsi
  __int64 v270; // rcx
  __int64 v271; // r8
  __int64 v272; // rcx
  __int64 v273; // rdx
  unsigned int v274; // ecx
  unsigned __int64 v275; // rax
  __int64 v276; // rdx
  _QWORD *v277; // rdi
  _QWORD *v278; // rdx
  unsigned __int64 v279; // rax
  __int64 v280; // rax
  __int64 v281; // rax
  unsigned __int8 *v282; // rax
  unsigned __int8 *v283; // rdx
  __int64 v284; // rdi
  char v285; // si
  unsigned int v286; // edx
  __int64 v287; // rax
  __int64 v288; // rcx
  __int64 v289; // rdi
  _BYTE *v290; // rsi
  __int64 v291; // rcx
  __int64 v292; // r13
  __int64 v293; // rax
  _QWORD *v294; // rax
  _QWORD *v295; // rax
  int v296; // [rsp-10h] [rbp-5F0h]
  int v297; // [rsp-8h] [rbp-5E8h]
  bool v298; // [rsp+20h] [rbp-5C0h]
  __int64 v299; // [rsp+20h] [rbp-5C0h]
  __int64 v300; // [rsp+28h] [rbp-5B8h]
  __int64 v301; // [rsp+30h] [rbp-5B0h]
  unsigned __int64 v304; // [rsp+48h] [rbp-598h]
  __int64 v305; // [rsp+48h] [rbp-598h]
  bool v306; // [rsp+50h] [rbp-590h]
  _QWORD *v307; // [rsp+50h] [rbp-590h]
  __int64 *v308; // [rsp+50h] [rbp-590h]
  int v309; // [rsp+50h] [rbp-590h]
  unsigned __int64 v310; // [rsp+50h] [rbp-590h]
  unsigned __int64 v311; // [rsp+58h] [rbp-588h]
  __int64 *v312; // [rsp+58h] [rbp-588h]
  __int64 *v313; // [rsp+58h] [rbp-588h]
  _QWORD *v314; // [rsp+58h] [rbp-588h]
  __int64 *v315; // [rsp+58h] [rbp-588h]
  __int64 v316; // [rsp+58h] [rbp-588h]
  __int64 v317; // [rsp+58h] [rbp-588h]
  const void **v318; // [rsp+58h] [rbp-588h]
  __int64 *v319; // [rsp+60h] [rbp-580h]
  _BOOL4 v320; // [rsp+60h] [rbp-580h]
  __int64 v321; // [rsp+60h] [rbp-580h]
  unsigned int v322; // [rsp+60h] [rbp-580h]
  __int64 *v323; // [rsp+60h] [rbp-580h]
  __int64 v324; // [rsp+60h] [rbp-580h]
  __int64 v325; // [rsp+60h] [rbp-580h]
  char v326; // [rsp+68h] [rbp-578h]
  _BOOL4 v327; // [rsp+68h] [rbp-578h]
  __int64 v328; // [rsp+68h] [rbp-578h]
  unsigned __int64 v329; // [rsp+68h] [rbp-578h]
  __int64 *v330; // [rsp+70h] [rbp-570h]
  __int64 v331; // [rsp+70h] [rbp-570h]
  __int64 v332; // [rsp+70h] [rbp-570h]
  __int64 ***v333; // [rsp+70h] [rbp-570h]
  _QWORD *v334; // [rsp+70h] [rbp-570h]
  __int64 v335; // [rsp+70h] [rbp-570h]
  __int64 v336; // [rsp+78h] [rbp-568h]
  __int64 v337; // [rsp+78h] [rbp-568h]
  __int64 v338; // [rsp+78h] [rbp-568h]
  unsigned __int8 *v339; // [rsp+80h] [rbp-560h] BYREF
  _BYTE *v340; // [rsp+88h] [rbp-558h]
  _BYTE *v341; // [rsp+90h] [rbp-550h]
  const char *v342; // [rsp+A0h] [rbp-540h] BYREF
  __int64 v343; // [rsp+A8h] [rbp-538h] BYREF
  __int64 *v344; // [rsp+B0h] [rbp-530h] BYREF
  __int64 *v345; // [rsp+B8h] [rbp-528h]
  __int64 *v346; // [rsp+C0h] [rbp-520h]
  __int64 v347; // [rsp+C8h] [rbp-518h]
  unsigned __int64 v348; // [rsp+F0h] [rbp-4F0h] BYREF
  __int64 v349; // [rsp+F8h] [rbp-4E8h] BYREF
  __int64 *v350; // [rsp+100h] [rbp-4E0h] BYREF
  __int64 *v351; // [rsp+108h] [rbp-4D8h]
  __int64 *v352; // [rsp+110h] [rbp-4D0h]
  unsigned __int64 v353; // [rsp+118h] [rbp-4C8h]
  _QWORD v354[3]; // [rsp+1C0h] [rbp-420h] BYREF
  __int64 v355; // [rsp+1D8h] [rbp-408h]
  _QWORD *v356; // [rsp+1E0h] [rbp-400h]
  __int64 v357; // [rsp+1E8h] [rbp-3F8h]
  unsigned int v358; // [rsp+1F0h] [rbp-3F0h]
  __int64 v359; // [rsp+1F8h] [rbp-3E8h] BYREF
  __int64 v360; // [rsp+200h] [rbp-3E0h]
  __int64 v361; // [rsp+208h] [rbp-3D8h]
  __int64 v362; // [rsp+210h] [rbp-3D0h]
  __int64 v363; // [rsp+218h] [rbp-3C8h] BYREF
  __int64 v364; // [rsp+220h] [rbp-3C0h]
  __int64 v365; // [rsp+228h] [rbp-3B8h]
  __int64 v366; // [rsp+230h] [rbp-3B0h]
  __int64 v367; // [rsp+238h] [rbp-3A8h]
  __int64 v368; // [rsp+240h] [rbp-3A0h]
  __int64 v369; // [rsp+248h] [rbp-398h]
  int v370; // [rsp+250h] [rbp-390h]
  __int64 v371; // [rsp+258h] [rbp-388h]
  _BYTE *v372; // [rsp+260h] [rbp-380h]
  _BYTE *v373; // [rsp+268h] [rbp-378h]
  __int64 v374; // [rsp+270h] [rbp-370h]
  int v375; // [rsp+278h] [rbp-368h]
  _BYTE v376[16]; // [rsp+280h] [rbp-360h] BYREF
  __int64 v377; // [rsp+290h] [rbp-350h]
  __int64 v378; // [rsp+298h] [rbp-348h]
  __int64 v379; // [rsp+2A0h] [rbp-340h]
  _QWORD *v380; // [rsp+2A8h] [rbp-338h]
  __int64 v381; // [rsp+2B0h] [rbp-330h]
  __int64 v382; // [rsp+2B8h] [rbp-328h]
  __int16 v383; // [rsp+2C0h] [rbp-320h]
  __int64 v384[5]; // [rsp+2C8h] [rbp-318h] BYREF
  int v385; // [rsp+2F0h] [rbp-2F0h]
  __int64 v386; // [rsp+2F8h] [rbp-2E8h]
  __int64 v387; // [rsp+300h] [rbp-2E0h]
  __int64 v388; // [rsp+308h] [rbp-2D8h]
  _BYTE *v389; // [rsp+310h] [rbp-2D0h]
  __int64 v390; // [rsp+318h] [rbp-2C8h]
  _BYTE v391[64]; // [rsp+320h] [rbp-2C0h] BYREF
  unsigned __int64 v392; // [rsp+360h] [rbp-280h] BYREF
  __int64 v393; // [rsp+368h] [rbp-278h] BYREF
  unsigned __int64 v394; // [rsp+370h] [rbp-270h] BYREF
  __int64 *v395; // [rsp+378h] [rbp-268h]
  __int64 *v396; // [rsp+380h] [rbp-260h]
  __m128i v397; // [rsp+388h] [rbp-258h] BYREF
  unsigned __int64 v398; // [rsp+398h] [rbp-248h]
  __int64 v399; // [rsp+3A0h] [rbp-240h]
  __int64 v400; // [rsp+3A8h] [rbp-238h]
  __int64 v401; // [rsp+3B0h] [rbp-230h]
  __int64 v402; // [rsp+3B8h] [rbp-228h]
  _BYTE *v403; // [rsp+3C0h] [rbp-220h]
  _BYTE *v404; // [rsp+3C8h] [rbp-218h]
  __int64 v405; // [rsp+3D0h] [rbp-210h]
  int v406; // [rsp+3D8h] [rbp-208h]
  _BYTE v407[128]; // [rsp+3E0h] [rbp-200h] BYREF
  _BYTE *v408; // [rsp+460h] [rbp-180h]
  __int64 v409; // [rsp+468h] [rbp-178h]
  _BYTE v410[256]; // [rsp+470h] [rbp-170h] BYREF
  __int64 v411; // [rsp+570h] [rbp-70h] BYREF
  __int64 v412; // [rsp+578h] [rbp-68h]
  __int64 v413; // [rsp+580h] [rbp-60h]
  int v414; // [rsp+588h] [rbp-58h]
  __int64 v415; // [rsp+590h] [rbp-50h]
  __int64 v416; // [rsp+598h] [rbp-48h]
  __int64 v417; // [rsp+5A0h] [rbp-40h]
  unsigned int v418; // [rsp+5A8h] [rbp-38h]

  sub_1940CE0(a1, a2, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, a7, a8, a9, a10);
  v10 = sub_1481F60(*(_QWORD **)(a1 + 8), a2, a3, a4);
  v11 = *(_QWORD *)(a1 + 24);
  v300 = v10;
  v12 = *(_QWORD *)(a1 + 8);
  v372 = v376;
  v373 = v376;
  v354[2] = "indvars";
  v354[0] = v12;
  v354[1] = v11;
  v355 = 0;
  v356 = 0;
  v357 = 0;
  v358 = 0;
  v359 = 0;
  v360 = 0;
  v361 = 0;
  v362 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v366 = 0;
  v367 = 0;
  v368 = 0;
  v369 = 0;
  v370 = 0;
  v371 = 0;
  v374 = 2;
  v375 = 0;
  v377 = 0;
  v378 = 0;
  v379 = 0;
  v380 = 0;
  v381 = 0;
  v382 = 0;
  v383 = 1;
  v13 = sub_15E0530(*(_QWORD *)(v12 + 24));
  v388 = v11;
  v14 = *(_QWORD *)a1;
  v384[3] = v13;
  v389 = v391;
  v390 = 0x800000000LL;
  v349 = 0x800000000LL;
  v15 = *(__int64 **)(a2 + 32);
  v348 = (unsigned __int64)&v350;
  memset(v384, 0, 24);
  v384[4] = 0;
  v385 = 0;
  v386 = 0;
  v387 = 0;
  LOBYTE(v383) = 0;
  v16 = sub_157EB90(*v15);
  v17 = sub_15E0FD0(79);
  v19 = sub_16321A0(v16, (__int64)v17, v18);
  v306 = 0;
  if ( v19 )
    v306 = *(_QWORD *)(v19 + 8) != 0;
  v342 = (const char *)&v344;
  v343 = 0x800000000LL;
  for ( i = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 48LL); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v133 = (unsigned int)v343;
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    if ( HIDWORD(v343) <= (unsigned int)v343 )
    {
      sub_16CD150((__int64)&v342, &v344, 0, 8, v20, v21);
      v133 = (unsigned int)v343;
    }
    *(_QWORD *)&v342[8 * v133] = i - 24;
    LODWORD(v343) = v343 + 1;
  }
  v336 = a1 + 48;
  if ( !(_DWORD)v343 )
    goto LABEL_46;
  do
  {
    do
    {
      v23 = *(_QWORD *)(a1 + 8);
      v24 = *(__int64 **)&v342[8 * (unsigned int)v133 - 8];
      v25 = *(_QWORD *)(a1 + 16);
      LODWORD(v343) = v133 - 1;
      v26 = *(__int64 **)(a1 + 40);
      v392 = (unsigned __int64)off_49F3848;
      v394 = v23;
      v395 = v26;
      v396 = v24;
      LOBYTE(v398) = 0;
      v393 = v25;
      v397 = (__m128i)(unsigned __int64)v24;
      *(_BYTE *)(a1 + 448) |= sub_1B649E0((_DWORD)v24, v23, v25, v14, v336, (unsigned int)v354, (__int64)&v392);
      if ( v397.m128i_i64[1] )
      {
        v27 = (unsigned int)v349;
        if ( (unsigned int)v349 >= HIDWORD(v349) )
        {
          sub_16CD150((__int64)&v348, &v350, 0, 24, v296, v297);
          v27 = (unsigned int)v349;
        }
        a3 = _mm_loadu_si128(&v397);
        v28 = (__m128 *)(v348 + 24 * v27);
        *v28 = (__m128)a3;
        v28[1].m128_u64[0] = v398;
        LODWORD(v349) = v349 + 1;
      }
      LODWORD(v133) = v343;
    }
    while ( (_DWORD)v343 );
    v29 = (unsigned int)v349;
    if ( !(_DWORD)v349 )
      break;
    do
    {
      v30 = *(_QWORD *)(a1 + 16);
      v31 = *(__int64 **)(a1 + 8);
      v32 = (char *)(v348 + 24 * v29 - 24);
      v33 = *(_QWORD *)v32;
      v392 = *(_QWORD *)v32;
      v34 = *((_QWORD *)v32 + 1);
      v394 = v14;
      v393 = v34;
      v35 = *(_DWORD *)(v14 + 24);
      v36 = 0;
      if ( v35 )
      {
        v37 = *(_QWORD *)(v33 + 40);
        v38 = v35 - 1;
        v39 = *(_QWORD *)(v14 + 8);
        v40 = v38 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
        v41 = (__int64 *)(v39 + 16LL * v40);
        v42 = *v41;
        if ( v37 == *v41 )
        {
LABEL_18:
          v36 = (__int64 *)v41[1];
        }
        else
        {
          v65 = 1;
          while ( v42 != -8 )
          {
            v174 = v65 + 1;
            v40 = v38 & (v65 + v40);
            v41 = (__int64 *)(v39 + 16LL * v40);
            v42 = *v41;
            if ( v37 == *v41 )
              goto LABEL_18;
            v65 = v174;
          }
          v36 = 0;
        }
      }
      v395 = v36;
      v411 = 0;
      v397.m128i_i8[8] = v306;
      v396 = v31;
      v401 = v336;
      v397.m128i_i64[0] = v30;
      v403 = v407;
      v404 = v407;
      v398 = 0;
      v408 = v410;
      v399 = 0;
      v400 = 0;
      v402 = 0;
      v405 = 16;
      v406 = 0;
      v409 = 0x800000000LL;
      v412 = 0;
      v413 = 0;
      v414 = 0;
      v415 = 0;
      v416 = 0;
      v417 = 0;
      v418 = 0;
      v43 = (unsigned __int8)v32[16];
      v411 = 1;
      sub_193E940((__int64)&v411, 0);
      if ( !v414 )
      {
        LODWORD(v413) = v413 + 1;
        BUG();
      }
      v46 = 1;
      v47 = 0;
      LODWORD(v48) = (v414 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v49 = (__int64 *)(v412 + 16LL * (unsigned int)v48);
      v50 = *v49;
      if ( v33 != *v49 )
      {
        while ( v50 != -8 )
        {
          if ( !v47 && v50 == -16 )
            v47 = v49;
          v48 = (v414 - 1) & (unsigned int)(v48 + v46);
          v49 = (__int64 *)(v412 + 16 * v48);
          v50 = *v49;
          if ( v33 == *v49 )
            goto LABEL_21;
          ++v46;
        }
        if ( v47 )
          v49 = v47;
      }
LABEL_21:
      LODWORD(v413) = v413 + 1;
      if ( *v49 != -8 )
        --HIDWORD(v413);
      *v49 = v33;
      *((_DWORD *)v49 + 2) = v43;
      v53 = sub_1943460((__int64 *)&v392, (__int64)v354, (__m128)a3, a4, a5, a6, v44, v45, a9, a10);
      if ( v53 )
      {
        *(_BYTE *)(a1 + 448) = 1;
        v54 = (unsigned int)v343;
        if ( (unsigned int)v343 >= HIDWORD(v343) )
        {
          sub_16CD150((__int64)&v342, &v344, 0, 8, v51, v52);
          v54 = (unsigned int)v343;
        }
        *(_QWORD *)&v342[8 * v54] = v53;
        LODWORD(v343) = v343 + 1;
      }
      if ( v418 )
      {
        v55 = v416;
        v56 = v416 + 48LL * v418;
        do
        {
          while ( *(_QWORD *)v55 == -8 )
          {
            if ( *(_QWORD *)(v55 + 8) != -8 )
              goto LABEL_30;
            v55 += 48;
            if ( v56 == v55 )
              goto LABEL_40;
          }
          if ( *(_QWORD *)v55 != -16 || *(_QWORD *)(v55 + 8) != -16 )
          {
LABEL_30:
            if ( *(_DWORD *)(v55 + 40) > 0x40u )
            {
              v57 = *(_QWORD *)(v55 + 32);
              if ( v57 )
                j_j___libc_free_0_0(v57);
            }
            if ( *(_DWORD *)(v55 + 24) > 0x40u )
            {
              v58 = *(_QWORD *)(v55 + 16);
              if ( v58 )
                j_j___libc_free_0_0(v58);
            }
          }
          v55 += 48;
        }
        while ( v56 != v55 );
      }
LABEL_40:
      j___libc_free_0(v416);
      j___libc_free_0(v412);
      if ( v408 != v410 )
        _libc_free((unsigned __int64)v408);
      if ( v404 != v403 )
        _libc_free((unsigned __int64)v404);
      v59 = (_DWORD)v349 == 1;
      v29 = (unsigned int)(v349 - 1);
      LODWORD(v349) = v349 - 1;
    }
    while ( !v59 );
    LODWORD(v133) = v343;
  }
  while ( (_DWORD)v343 );
LABEL_46:
  if ( v342 != (const char *)&v344 )
    _libc_free((unsigned __int64)v342);
  if ( (__int64 **)v348 != &v350 )
    _libc_free(v348);
  if ( dword_4FAF860 && !sub_14562D0(v300) )
    sub_1941790(a1, a2, (__int64)v354, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v204, v205, a9, a10);
  sub_387CB50(v354, a2, *(_QWORD *)(a1 + 16), v336, 0);
  if ( byte_4FAF6A0
    || !(unsigned __int8)sub_193E1A0(a2, *(_QWORD **)(a1 + 8), (__int64)v354, a3, a4)
    || !sub_193F280(a2, *(_QWORD *)(a1 + 16)) )
  {
    goto LABEL_52;
  }
  v176 = 0;
  v177 = *(_QWORD *)(a1 + 8);
  v328 = *(_QWORD *)(a1 + 16);
  v332 = v177;
  v178 = sub_1456040(v300);
  v304 = sub_1456C90(v177, v178);
  v179 = sub_13F9E70(a2);
  v314 = *(_QWORD **)(sub_157EBA0(v179) - 72);
  v321 = sub_13FCB50(a2);
  v180 = sub_157EB90(**(_QWORD **)(a2 + 32));
  v301 = sub_1632FA0(v180);
  v181 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 48LL);
  v182 = 0;
  while ( 1 )
  {
    if ( !v181 )
      BUG();
    if ( *(_BYTE *)(v181 - 8) != 77 )
      break;
    if ( sub_1456C80(v332, *(_QWORD *)(v181 - 24))
      && (*(_BYTE *)(sub_1456040(v300) + 8) != 15 || *(_BYTE *)(*(_QWORD *)(v181 - 24) + 8LL) == 15) )
    {
      v183 = v181 - 24;
      v184 = sub_146F1B0(v332, v181 - 24);
      if ( *(_WORD *)(v184 + 24) == 7 && a2 == *(_QWORD *)(v184 + 48) && *(_QWORD *)(v184 + 40) == 2 )
      {
        v281 = sub_1456040(**(_QWORD **)(v184 + 32));
        v310 = sub_1456C90(v332, v281);
        if ( v304 <= v310 )
        {
          v282 = *(unsigned __int8 **)(v301 + 24);
          v283 = &v282[*(unsigned int *)(v301 + 32)];
          if ( v282 != v283 )
          {
            while ( v310 != *v282 )
            {
              if ( v283 == ++v282 )
                goto LABEL_317;
            }
            v284 = sub_13A5BC0((_QWORD *)v184, v332);
            if ( !*(_WORD *)(v284 + 24) && sub_1456110(v284) )
            {
              v285 = *(_BYTE *)(v181 - 1) & 0x40;
              v286 = *(_DWORD *)(v181 - 4) & 0xFFFFFFF;
              if ( v286 )
              {
                v287 = 0;
                v288 = 24LL * *(unsigned int *)(v181 + 32) + 8;
                while ( 1 )
                {
                  v289 = v183 - 24LL * v286;
                  if ( v285 )
                    v289 = *(_QWORD *)(v181 - 32);
                  if ( v321 == *(_QWORD *)(v289 + v288) )
                    break;
                  v287 = (unsigned int)(v287 + 1);
                  v288 += 8;
                  if ( v286 == (_DWORD)v287 )
                    goto LABEL_543;
                }
              }
              else
              {
LABEL_543:
                v287 = 0xFFFFFFFFLL;
              }
              if ( v285 )
                v291 = *(_QWORD *)(v181 - 32);
              else
                v291 = v183 - 24LL * v286;
              if ( v183 == sub_193F190(*(_QWORD *)(v291 + 24 * v287), a2, v328) )
              {
                if ( (unsigned __int8)sub_193F750(v181 - 24)
                  || (v293 = sub_193E280(a2)) == 0
                  || (v299 = v293, v183 == sub_193F190(*(_QWORD *)(v293 - 48), a2, v328))
                  || v183 == sub_193F190(*(_QWORD *)(v299 - 24), a2, v328) )
                {
                  v292 = **(_QWORD **)(v184 + 32);
                  if ( !v182 || (unsigned __int8)sub_193E640(v182, v321, v314) )
                  {
                    v176 = v292;
                    v182 = v181 - 24;
                  }
                  else if ( !(unsigned __int8)sub_193E640(v181 - 24, v321, v314) )
                  {
                    v298 = sub_14560B0(v176);
                    if ( v298 == sub_14560B0(v292) )
                    {
                      if ( v310 > sub_1456C90(v332, *(_QWORD *)v182) )
                      {
                        v176 = v292;
                        v182 = v181 - 24;
                      }
                    }
                    else if ( !sub_14560B0(v176) )
                    {
                      v176 = v292;
                      v182 = v181 - 24;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_317:
    v181 = *(_QWORD *)(v181 + 8);
  }
  v333 = (__int64 ***)v182;
  if ( !v182 || *(_WORD *)(v300 + 24) == 7 && !sub_13FC520(*(_QWORD *)(v300 + 48)) )
    goto LABEL_52;
  v185 = sub_13F9E70(a2);
  if ( v185 == sub_13FCB50(a2) )
  {
    v264 = *(_QWORD *)(a1 + 8);
    v265 = sub_1456040(v300);
    v395 = (__int64 *)sub_145CF80(v264, v265, 1, 0);
    v392 = (unsigned __int64)&v394;
    v394 = v300;
    v393 = 0x200000002LL;
    v186 = sub_147DD40(v264, (__int64 *)&v392, 0, 0, a3, a4);
    if ( (unsigned __int64 *)v392 != &v394 )
      _libc_free(v392);
    v266 = sub_13F9E70(a2);
    v267 = *(_DWORD *)(v182 + 20) & 0xFFFFFFF;
    if ( v267 )
    {
      v268 = *(_BYTE *)(v182 + 23) & 0x40;
      v269 = 24LL * *(unsigned int *)(v182 + 56) + 8;
      v270 = 0;
      do
      {
        v271 = v182 - 24LL * v267;
        if ( v268 )
          v271 = *(_QWORD *)(v182 - 8);
        if ( v266 == *(_QWORD *)(v271 + v269) )
        {
          v272 = 24 * v270;
          goto LABEL_499;
        }
        ++v270;
        v269 += 8;
      }
      while ( v267 != (_DWORD)v270 );
      v272 = 0x17FFFFFFE8LL;
    }
    else
    {
      v272 = 0x17FFFFFFE8LL;
      v268 = *(_BYTE *)(v182 + 23) & 0x40;
    }
LABEL_499:
    if ( v268 )
      v273 = *(_QWORD *)(v182 - 8);
    else
      v273 = v182 - 24LL * v267;
    v187 = *(_QWORD *)(v273 + v272);
  }
  else
  {
    v186 = (__int64 *)v300;
    v187 = v182;
  }
  v188 = sub_1940670(v182, (__int64)v186, a2, (__int64)v354, *(_QWORD **)(a1 + 8), a3, a4);
  v189 = sub_13F9E70(a2);
  v329 = sub_157EBA0(v189);
  v309 = 32 - (!sub_1377F70(a2 + 56, *(_QWORD *)(v329 - 24)) - 1);
  v190 = sub_16498A0(v329);
  v392 = 0;
  v395 = (__int64 *)v190;
  v396 = 0;
  v397.m128i_i32[0] = 0;
  v397.m128i_i64[1] = 0;
  v398 = 0;
  v393 = *(_QWORD *)(v329 + 40);
  v394 = v329 + 24;
  v191 = *(_QWORD *)(v329 + 48);
  v348 = v191;
  if ( v191 )
  {
    sub_1623A60((__int64)&v348, v191, 2);
    if ( v392 )
      sub_161E7C0((__int64)&v392, v392);
    v392 = v348;
    if ( v348 )
      sub_1623210((__int64)&v348, (unsigned __int8 *)v348, (__int64)&v392);
  }
  v192 = *(_QWORD *)(v329 - 72);
  if ( *(_BYTE *)(v192 + 16) > 0x17u )
  {
    v193 = *(_QWORD *)(v192 + 48);
    v348 = v193;
    if ( v193 )
    {
      sub_1623A60((__int64)&v348, v193, 2);
      v194 = v392;
      if ( v392 )
        goto LABEL_333;
LABEL_334:
      v392 = v348;
      if ( v348 )
        sub_1623210((__int64)&v348, (unsigned __int8 *)v348, (__int64)&v392);
    }
    else
    {
      v194 = v392;
      if ( v392 )
      {
LABEL_333:
        sub_161E7C0((__int64)&v392, v194);
        goto LABEL_334;
      }
    }
  }
  v322 = sub_1456C90(*(_QWORD *)(a1 + 8), *(_QWORD *)v187);
  if ( v322 <= (unsigned int)sub_1456C90(*(_QWORD *)(a1 + 8), *(_QWORD *)v188) )
    goto LABEL_337;
  v230 = sub_146F1B0(*(_QWORD *)(a1 + 8), (__int64)v333);
  v231 = **(_QWORD **)(v230 + 32);
  v305 = sub_13A5BC0((_QWORD *)v230, *(_QWORD *)(a1 + 8));
  if ( !*(_WORD *)(v231 + 24) && !*((_WORD *)v186 + 12) )
  {
    v335 = *(_QWORD *)(v231 + 32);
    v318 = (const void **)(v335 + 24);
    v242 = v186[4];
    v243 = *(_DWORD *)(v242 + 32);
    LODWORD(v340) = v243;
    if ( v243 > 0x40 )
    {
      sub_16A4FD0((__int64)&v339, (const void **)(v242 + 24));
      if ( (__int64 *)v300 == v186 )
      {
LABEL_425:
        sub_16A5C50((__int64)&v348, (const void **)&v339, v322);
        if ( (unsigned int)v340 > 0x40 && v339 )
          j_j___libc_free_0_0(v339);
        v339 = (unsigned __int8 *)v348;
        LODWORD(v340) = v349;
LABEL_429:
        LODWORD(v343) = 1;
        v342 = 0;
        v244 = *(_QWORD *)(v305 + 32);
        v245 = *(_DWORD *)(v244 + 32);
        v246 = *(_QWORD *)(v244 + 24);
        v247 = 1LL << ((unsigned __int8)v245 - 1);
        if ( v245 > 0x40 )
          v246 = *(_QWORD *)(v246 + 8LL * ((v245 - 1) >> 6));
        v248 = *(_DWORD *)(v335 + 32);
        LODWORD(v349) = v248;
        if ( (v246 & v247) != 0 )
        {
          if ( v248 > 0x40 )
            sub_16A4FD0((__int64)&v348, v318);
          else
            v348 = *(_QWORD *)(v335 + 24);
          sub_16A7590((__int64)&v348, (__int64 *)&v339);
        }
        else
        {
          if ( v248 > 0x40 )
            sub_16A4FD0((__int64)&v348, v318);
          else
            v348 = *(_QWORD *)(v335 + 24);
          sub_16A7200((__int64)&v348, (__int64 *)&v339);
        }
        v249 = v349;
        LODWORD(v349) = 0;
        v250 = v348;
        if ( (unsigned int)v343 > 0x40 && v342 )
          j_j___libc_free_0_0(v342);
        v342 = (const char *)v250;
        LODWORD(v343) = v249;
        if ( (unsigned int)v349 > 0x40 && v348 )
          j_j___libc_free_0_0(v348);
        v188 = sub_15A1070(*(_QWORD *)v187, (__int64)&v342);
        if ( (unsigned int)v343 > 0x40 && v342 )
          j_j___libc_free_0_0(v342);
        if ( (unsigned int)v340 > 0x40 && v339 )
          j_j___libc_free_0_0(v339);
        goto LABEL_337;
      }
      v243 = (unsigned int)v340;
      if ( (unsigned int)v340 > 0x40 )
      {
        if ( v243 - (unsigned int)sub_16A57B0((__int64)&v339) > 0x40 || *(_QWORD *)v339 )
          goto LABEL_425;
        LODWORD(v349) = v243;
        sub_16A4EF0((__int64)&v348, -1, 1);
        goto LABEL_456;
      }
    }
    else
    {
      v339 = *(unsigned __int8 **)(v242 + 24);
      if ( (__int64 *)v300 == v186 )
        goto LABEL_425;
    }
    if ( v339 )
      goto LABEL_425;
    LODWORD(v349) = v243;
    v348 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v243;
LABEL_456:
    sub_16A5C50((__int64)&v342, (const void **)&v348, v322);
    if ( (unsigned int)v340 > 0x40 && v339 )
      j_j___libc_free_0_0(v339);
    v339 = (unsigned __int8 *)v342;
    LODWORD(v340) = v343;
    if ( (unsigned int)v349 > 0x40 && v348 )
      j_j___libc_free_0_0(v348);
    sub_16A7400((__int64)&v339);
    goto LABEL_429;
  }
  v232 = sub_146F1B0(*(_QWORD *)(a1 + 8), v187);
  v233 = *(_QWORD **)(a1 + 8);
  v234 = v232;
  v316 = *(_QWORD *)v187;
  v324 = *(_QWORD *)v188;
  v235 = sub_146F1B0((__int64)v233, v187);
  v236 = sub_14835F0(v233, v235, v324, 0, a3, a4);
  if ( v234 == sub_14747F0((__int64)v233, v236, v316, 0) )
  {
    v342 = "wide.trip.count";
    LOWORD(v344) = 259;
    v257 = *v333;
    if ( *v333 == *(__int64 ***)v188 )
      goto LABEL_337;
    if ( *(_BYTE *)(v188 + 16) <= 0x10u )
    {
      v188 = sub_15A46C0(37, (__int64 ***)v188, v257, 0);
      goto LABEL_337;
    }
    LOWORD(v350) = 257;
    v258 = v188;
    v259 = 37;
    goto LABEL_479;
  }
  v237 = *(_QWORD **)(a1 + 8);
  v317 = *(_QWORD *)v187;
  v325 = *(_QWORD *)v188;
  v238 = sub_146F1B0((__int64)v237, v187);
  v239 = sub_14835F0(v237, v238, v325, 0, a3, a4);
  v240 = sub_147B0D0((__int64)v237, v239, v317, 0);
  BYTE1(v344) = 1;
  if ( v234 == v240 )
  {
    LOBYTE(v344) = 3;
    v342 = "wide.trip.count";
    v257 = *v333;
    if ( *v333 == *(__int64 ***)v188 )
      goto LABEL_337;
    if ( *(_BYTE *)(v188 + 16) <= 0x10u )
    {
      v188 = sub_15A46C0(38, (__int64 ***)v188, v257, 0);
      goto LABEL_337;
    }
    LOWORD(v350) = 257;
    v258 = v188;
    v259 = 38;
LABEL_479:
    v188 = sub_15FDBD0(v259, v258, (__int64)v257, (__int64)&v348, 0);
    if ( v393 )
    {
      v260 = (__int64 *)v394;
      sub_157E9D0(v393 + 40, v188);
      v261 = *(_QWORD *)(v188 + 24);
      v262 = *v260;
      *(_QWORD *)(v188 + 32) = v260;
      v262 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v188 + 24) = v262 | v261 & 7;
      *(_QWORD *)(v262 + 8) = v188 + 24;
      *v260 = *v260 & 7 | (v188 + 24);
    }
    sub_164B780(v188, (__int64 *)&v342);
    if ( !v392 )
      goto LABEL_337;
    v254 = v188 + 48;
    v339 = (unsigned __int8 *)v392;
    sub_1623A60((__int64)&v339, v392, 2);
    v263 = *(_QWORD *)(v188 + 48);
    if ( v263 )
      sub_161E7C0(v188 + 48, v263);
    v256 = v339;
    *(_QWORD *)(v188 + 48) = v339;
    if ( !v256 )
      goto LABEL_337;
LABEL_470:
    sub_1623210((__int64)&v339, v256, v254);
    goto LABEL_337;
  }
  LOBYTE(v344) = 3;
  v342 = "lftr.wideiv";
  v241 = *(__int64 ***)v188;
  if ( *(_QWORD *)v188 != *(_QWORD *)v187 )
  {
    if ( *(_BYTE *)(v187 + 16) <= 0x10u )
    {
      v187 = sub_15A46C0(36, (__int64 ***)v187, v241, 0);
      goto LABEL_337;
    }
    LOWORD(v350) = 257;
    v187 = sub_15FDBD0(36, v187, (__int64)v241, (__int64)&v348, 0);
    if ( v393 )
    {
      v251 = (__int64 *)v394;
      sub_157E9D0(v393 + 40, v187);
      v252 = *(_QWORD *)(v187 + 24);
      v253 = *v251;
      *(_QWORD *)(v187 + 32) = v251;
      v253 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v187 + 24) = v253 | v252 & 7;
      *(_QWORD *)(v253 + 8) = v187 + 24;
      *v251 = *v251 & 7 | (v187 + 24);
    }
    sub_164B780(v187, (__int64 *)&v342);
    if ( v392 )
    {
      v339 = (unsigned __int8 *)v392;
      v254 = v187 + 48;
      sub_1623A60((__int64)&v339, v392, 2);
      v255 = *(_QWORD *)(v187 + 48);
      if ( v255 )
        sub_161E7C0(v187 + 48, v255);
      v256 = v339;
      *(_QWORD *)(v187 + 48) = v339;
      if ( v256 )
        goto LABEL_470;
    }
  }
LABEL_337:
  v342 = "exitcond";
  LOWORD(v344) = 259;
  if ( *(_BYTE *)(v187 + 16) > 0x10u || *(_BYTE *)(v188 + 16) > 0x10u )
  {
    LOWORD(v350) = 257;
    v220 = sub_1648A60(56, 2u);
    v195 = v220;
    if ( v220 )
    {
      v221 = (__int64)v220;
      v222 = *(_QWORD ***)v187;
      if ( *(_BYTE *)(*(_QWORD *)v187 + 8LL) == 16 )
      {
        v334 = v222[4];
        v223 = (__int64 *)sub_1643320(*v222);
        v224 = (__int64)sub_16463B0(v223, (unsigned int)v334);
      }
      else
      {
        v224 = sub_1643320(*v222);
      }
      sub_15FEC10((__int64)v195, v224, 51, v309, v187, v188, (__int64)&v348, 0);
    }
    else
    {
      v221 = 0;
    }
    if ( v393 )
    {
      v225 = (unsigned __int64 *)v394;
      sub_157E9D0(v393 + 40, (__int64)v195);
      v226 = v195[3];
      v227 = *v225;
      v195[4] = v225;
      v227 &= 0xFFFFFFFFFFFFFFF8LL;
      v195[3] = v227 | v226 & 7;
      *(_QWORD *)(v227 + 8) = v195 + 3;
      *v225 = *v225 & 7 | (unsigned __int64)(v195 + 3);
    }
    sub_164B780(v221, (__int64 *)&v342);
    if ( v392 )
    {
      v339 = (unsigned __int8 *)v392;
      sub_1623A60((__int64)&v339, v392, 2);
      v228 = v195[6];
      if ( v228 )
        sub_161E7C0((__int64)(v195 + 6), v228);
      v229 = v339;
      v195[6] = v339;
      if ( v229 )
        sub_1623210((__int64)&v339, v229, (__int64)(v195 + 6));
    }
  }
  else
  {
    v195 = (_QWORD *)sub_15A37B0(v309, (_QWORD *)v187, (_QWORD *)v188, 0);
  }
  v196 = *(_QWORD *)(v329 - 72);
  if ( v196 )
  {
    v197 = *(_QWORD *)(v329 - 64);
    v198 = *(_QWORD *)(v329 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v198 = v197;
    if ( v197 )
      *(_QWORD *)(v197 + 16) = v198 | *(_QWORD *)(v197 + 16) & 3LL;
    *(_QWORD *)(v329 - 72) = v195;
    if ( !v195 )
    {
      v348 = 6;
      v349 = 0;
      v350 = (__int64 *)v196;
      goto LABEL_347;
    }
LABEL_344:
    v199 = v195[1];
    *(_QWORD *)(v329 - 64) = v199;
    if ( v199 )
      *(_QWORD *)(v199 + 16) = (v329 - 64) | *(_QWORD *)(v199 + 16) & 3LL;
    *(_QWORD *)(v329 - 56) = (unsigned __int64)(v195 + 1) | *(_QWORD *)(v329 - 56) & 3LL;
    v195[1] = v329 - 72;
    v348 = 6;
    v349 = 0;
    v350 = (__int64 *)v196;
    if ( v196 )
    {
LABEL_347:
      if ( v196 != -16 && v196 != -8 )
        sub_164C220((__int64)&v348);
    }
  }
  else
  {
    *(_QWORD *)(v329 - 72) = v195;
    if ( v195 )
      goto LABEL_344;
    v348 = 6;
    v349 = 0;
    v350 = 0;
  }
  v200 = *(_DWORD *)(a1 + 56);
  if ( v200 >= *(_DWORD *)(a1 + 60) )
  {
    sub_170B450(v336, 0);
    v200 = *(_DWORD *)(a1 + 56);
  }
  v201 = (unsigned __int64 *)(*(_QWORD *)(a1 + 48) + 24LL * v200);
  if ( v201 )
  {
    *v201 = 6;
    v201[1] = 0;
    v202 = (__int64)v350;
    v59 = v350 == 0;
    v201[2] = (unsigned __int64)v350;
    if ( v202 != -8 && !v59 && v202 != -16 )
      sub_1649AC0(v201, v348 & 0xFFFFFFFFFFFFFFF8LL);
    v200 = *(_DWORD *)(a1 + 56);
  }
  *(_DWORD *)(a1 + 56) = v200 + 1;
  if ( v350 != 0 && v350 + 1 != 0 && v350 != (__int64 *)-16LL )
    sub_1649B30(&v348);
  v203 = v392;
  *(_BYTE *)(a1 + 448) = 1;
  if ( v203 )
    sub_161E7C0((__int64)&v392, v203);
LABEL_52:
  v60 = v357;
  ++v355;
  if ( !v357 )
    goto LABEL_75;
  v61 = v356;
  v62 = &v356[5 * v358];
  v63 = 4 * v357;
  if ( (unsigned int)(4 * v357) < 0x40 )
    v63 = 64;
  if ( v63 >= v358 )
  {
    while ( 1 )
    {
      if ( v61 == v62 )
        goto LABEL_74;
      if ( *v61 != -8 )
        break;
      if ( v61[1] != -8 )
        goto LABEL_58;
LABEL_62:
      v61 += 5;
    }
    if ( *v61 != -16 || v61[1] != -16 )
    {
LABEL_58:
      v64 = v61[4];
      if ( v64 != 0 && v64 != -8 && v64 != -16 )
        sub_1649B30(v61 + 2);
    }
    *v61 = -8;
    v61[1] = -8;
    goto LABEL_62;
  }
  while ( 2 )
  {
    if ( *v61 == -8 )
    {
      if ( v61[1] == -8 )
        goto LABEL_236;
    }
    else if ( *v61 == -16 && v61[1] == -16 )
    {
      goto LABEL_236;
    }
    v144 = v61[4];
    if ( v144 != 0 && v144 != -8 && v144 != -16 )
      sub_1649B30(v61 + 2);
LABEL_236:
    v61 += 5;
    if ( v61 != v62 )
      continue;
    break;
  }
  if ( !v60 )
  {
    if ( v358 )
    {
      j___libc_free_0(v356);
      v356 = 0;
      v357 = 0;
      v358 = 0;
      goto LABEL_75;
    }
LABEL_74:
    v357 = 0;
    goto LABEL_75;
  }
  v206 = 64;
  if ( v60 != 1 )
  {
    _BitScanReverse(&v207, v60 - 1);
    v206 = 1 << (33 - (v207 ^ 0x1F));
    if ( v206 < 64 )
      v206 = 64;
  }
  v208 = v356;
  if ( v358 == v206 )
  {
    v357 = 0;
    v294 = &v356[5 * v358];
    do
    {
      if ( v208 )
      {
        *v208 = -8;
        v208[1] = -8;
      }
      v208 += 5;
    }
    while ( v294 != v208 );
  }
  else
  {
    j___libc_free_0(v356);
    v209 = ((((((((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
              | (4 * v206 / 3u + 1)
              | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 4)
            | (((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
            | (4 * v206 / 3u + 1)
            | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
            | (4 * v206 / 3u + 1)
            | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 4)
          | (((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
          | (4 * v206 / 3u + 1)
          | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 16;
    v358 = (v209
          | (((((((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
              | (4 * v206 / 3u + 1)
              | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 4)
            | (((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
            | (4 * v206 / 3u + 1)
            | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
            | (4 * v206 / 3u + 1)
            | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 4)
          | (((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
          | (4 * v206 / 3u + 1)
          | ((4 * v206 / 3u + 1) >> 1))
         + 1;
    v210 = (_QWORD *)sub_22077B0(
                       40
                     * ((v209
                       | (((((((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
                           | (4 * v206 / 3u + 1)
                           | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 4)
                         | (((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
                         | (4 * v206 / 3u + 1)
                         | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 8)
                       | (((((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
                         | (4 * v206 / 3u + 1)
                         | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v206 / 3u + 1) | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1)) >> 2)
                       | (4 * v206 / 3u + 1)
                       | ((unsigned __int64)(4 * v206 / 3u + 1) >> 1))
                      + 1));
    v357 = 0;
    v356 = v210;
    for ( j = &v210[5 * v358]; j != v210; v210 += 5 )
    {
      if ( v210 )
      {
        *v210 = -8;
        v210[1] = -8;
      }
    }
  }
LABEL_75:
  sub_1940B30((__int64)&v359);
  sub_1940B30((__int64)&v363);
  ++v379;
  if ( (_DWORD)v381 )
  {
    v134 = 4 * v381;
    v66 = (unsigned int)v382;
    if ( (unsigned int)(4 * v381) < 0x40 )
      v134 = 64;
    if ( v134 >= (unsigned int)v382 )
    {
LABEL_78:
      v67 = v380;
      v68 = &v380[v66];
      if ( v380 != v68 )
      {
        do
          *v67++ = -8;
        while ( v68 != v67 );
      }
      v381 = 0;
      goto LABEL_81;
    }
    v135 = v380;
    if ( (_DWORD)v381 == 1 )
    {
      v141 = 1024;
      v140 = 128;
    }
    else
    {
      _BitScanReverse(&v136, v381 - 1);
      v137 = (unsigned int)(1 << (33 - (v136 ^ 0x1F)));
      if ( (int)v137 < 64 )
        v137 = 64;
      if ( (_DWORD)v137 == (_DWORD)v382 )
      {
        v381 = 0;
        v295 = &v380[v137];
        do
        {
          if ( v135 )
            *v135 = -8;
          ++v135;
        }
        while ( v295 != v135 );
        goto LABEL_81;
      }
      v138 = (4 * (int)v137 / 3u + 1) | ((unsigned __int64)(4 * (int)v137 / 3u + 1) >> 1);
      v139 = ((v138 | (v138 >> 2)) >> 4)
           | v138
           | (v138 >> 2)
           | ((((v138 | (v138 >> 2)) >> 4) | v138 | (v138 >> 2)) >> 8);
      v140 = (v139 | (v139 >> 16)) + 1;
      v141 = 8 * ((v139 | (v139 >> 16)) + 1);
    }
    j___libc_free_0(v380);
    LODWORD(v382) = v140;
    v142 = (_QWORD *)sub_22077B0(v141);
    v381 = 0;
    v380 = v142;
    for ( k = &v142[(unsigned int)v382]; k != v142; ++v142 )
    {
      if ( v142 )
        *v142 = -8;
    }
    goto LABEL_81;
  }
  if ( HIDWORD(v381) )
  {
    v66 = (unsigned int)v382;
    if ( (unsigned int)v382 <= 0x40 )
      goto LABEL_78;
    j___libc_free_0(v380);
    v380 = 0;
    v381 = 0;
    LODWORD(v382) = 0;
  }
LABEL_81:
  while ( 1 )
  {
    v69 = *(_DWORD *)(a1 + 56);
    if ( !v69 )
      break;
    while ( 1 )
    {
      v70 = *(_QWORD *)(a1 + 48);
      v392 = 6;
      v393 = 0;
      v71 = (_QWORD *)(v70 + 24LL * v69 - 24);
      v394 = v71[2];
      if ( v394 != 0 && v394 != -8 && v394 != -16 )
      {
        sub_1649AC0(&v392, *v71 & 0xFFFFFFFFFFFFFFF8LL);
        v69 = *(_DWORD *)(a1 + 56);
        v70 = *(_QWORD *)(a1 + 48);
      }
      v72 = v69 - 1;
      *(_DWORD *)(a1 + 56) = v72;
      v73 = (_QWORD *)(v70 + 24 * v72);
      v74 = v73[2];
      if ( v74 != 0 && v74 != -8 && v74 != -16 )
        sub_1649B30(v73);
      v75 = v394;
      if ( !v394 )
        break;
      v76 = v394 != -16 && v394 != -8;
      if ( *(_BYTE *)(v394 + 16) <= 0x17u )
      {
        if ( v76 )
          sub_1649B30(&v392);
        goto LABEL_81;
      }
      if ( v76 )
        sub_1649B30(&v392);
      sub_1AEB370(v75, *(_QWORD *)(a1 + 32));
      v69 = *(_DWORD *)(a1 + 56);
      if ( !v69 )
        goto LABEL_93;
    }
  }
LABEL_93:
  v337 = sub_13FA090(a2);
  if ( v337 )
  {
    v77 = sub_13FC520(a2);
    if ( v77 )
    {
      LODWORD(v343) = 0;
      v344 = 0;
      v345 = &v343;
      v346 = &v343;
      v347 = 0;
      v339 = 0;
      v340 = 0;
      v341 = 0;
      v330 = (__int64 *)sub_157EE30(v337);
      v78 = sub_157EBA0(v77);
      if ( v78 )
        v78 += 24LL;
      v79 = (_QWORD *)v78;
      if ( *(_QWORD *)(v77 + 48) == v78 )
        goto LABEL_248;
      while ( 2 )
      {
        while ( 2 )
        {
          v80 = *v79 & 0xFFFFFFFFFFFFFFF8LL;
          v79 = (_QWORD *)v80;
          if ( !v80 )
            BUG();
          v81 = (_QWORD *)(v80 - 24);
          if ( *(_BYTE *)(v80 - 8) == 77 )
            goto LABEL_248;
          if ( !(unsigned __int8)sub_15F3040(v80 - 24)
            && !sub_15F3330(v80 - 24)
            && !(unsigned __int8)sub_15F2ED0(v80 - 24) )
          {
            v82 = *(_BYTE *)(v80 - 8);
            if ( v82 != 78 )
            {
              v83 = (unsigned int)v82 - 34;
              if ( (unsigned int)v83 <= 0x36 )
              {
                v84 = 0x40018000000001LL;
                if ( _bittest64(&v84, v83) )
                  break;
              }
              if ( v82 == 53 )
                break;
LABEL_109:
              if ( *(_QWORD *)(v80 - 16) )
              {
                v326 = 0;
                v311 = v80;
                v85 = *(unsigned __int64 **)(v80 - 16);
                while ( 1 )
                {
                  v86 = *v85;
                  if ( *(_BYTE *)(*v85 + 16) <= 0x17u )
                    break;
                  v87 = v344;
                  if ( !v344 )
                    break;
                  v88 = &v343;
                  do
                  {
                    while ( 1 )
                    {
                      v89 = v87[2];
                      v90 = v87[3];
                      if ( v87[4] >= v86 )
                        break;
                      v87 = (__int64 *)v87[3];
                      if ( !v90 )
                        goto LABEL_117;
                    }
                    v88 = v87;
                    v87 = (__int64 *)v87[2];
                  }
                  while ( v89 );
LABEL_117:
                  if ( v88 == &v343 || v88[4] > v86 )
                    break;
                  v326 = 1;
LABEL_120:
                  v85 = (unsigned __int64 *)v85[1];
                  if ( !v85 )
                  {
                    v392 = (unsigned __int64)v81;
                    v80 = v311;
                    if ( !v326 )
                      goto LABEL_505;
                    goto LABEL_122;
                  }
                }
                v145 = sub_1648700((__int64)v85);
                if ( *((_BYTE *)v145 + 16) == 77 )
                {
                  v307 = v145;
                  v146 = sub_1648720((__int64)v85);
                  if ( (*((_BYTE *)v307 + 23) & 0x40) != 0 )
                    v147 = (_QWORD *)*(v307 - 1);
                  else
                    v147 = &v307[-3 * (*((_DWORD *)v307 + 5) & 0xFFFFFFF)];
                  v148 = v147[3 * *((unsigned int *)v307 + 14) + 1 + v146];
                }
                else
                {
                  v148 = v145[5];
                }
                if ( v77 == v148 )
                  break;
                if ( !sub_1377F70(a2 + 56, v148) )
                  goto LABEL_120;
                if ( *(_QWORD **)(v77 + 48) != v79 )
                  continue;
LABEL_248:
                if ( v347 )
                {
                  LODWORD(v349) = 0;
                  v351 = &v349;
                  v101 = v345;
                  v352 = &v349;
                  v350 = 0;
                  v353 = 0;
                  LODWORD(v393) = 0;
                  v394 = 0;
                  v395 = &v393;
                  v396 = &v393;
                  v397.m128i_i64[0] = 0;
                  if ( v345 != &v343 )
                  {
                    while ( 1 )
                    {
LABEL_250:
                      v149 = v101[4];
                      if ( (*(_DWORD *)(v149 + 20) & 0xFFFFFFF) == 0 )
                        goto LABEL_273;
                      v150 = 0;
                      v151 = v101;
                      v152 = 24LL * (*(_DWORD *)(v149 + 20) & 0xFFFFFFF);
                      do
                      {
                        while ( 1 )
                        {
                          if ( (*(_BYTE *)(v149 + 23) & 0x40) != 0 )
                            v153 = *(_QWORD *)(v149 - 8);
                          else
                            v153 = v149 - 24LL * (*(_DWORD *)(v149 + 20) & 0xFFFFFFF);
                          v154 = *(_QWORD *)(v153 + v150);
                          if ( *(_BYTE *)(v154 + 16) <= 0x17u )
                            goto LABEL_271;
                          if ( v344 )
                          {
                            v155 = v344;
                            v156 = &v343;
                            do
                            {
                              while ( 1 )
                              {
                                v157 = v155[2];
                                v158 = v155[3];
                                if ( v155[4] >= v154 )
                                  break;
                                v155 = (__int64 *)v155[3];
                                if ( !v158 )
                                  goto LABEL_260;
                              }
                              v156 = v155;
                              v155 = (__int64 *)v155[2];
                            }
                            while ( v157 );
LABEL_260:
                            if ( v156 != &v343 && v156[4] <= v154 )
                              goto LABEL_271;
                          }
                          for ( m = *(unsigned __int64 **)(v154 + 8); m; m = (unsigned __int64 *)m[1] )
                          {
                            v160 = *m;
                            if ( *(_BYTE *)(*m + 16) > 0x17u )
                            {
                              if ( !v344 )
                                goto LABEL_271;
                              v161 = v344;
                              v162 = &v343;
                              do
                              {
                                while ( 1 )
                                {
                                  v163 = v161[2];
                                  v164 = v161[3];
                                  if ( v161[4] >= v160 )
                                    break;
                                  v161 = (__int64 *)v161[3];
                                  if ( !v164 )
                                    goto LABEL_269;
                                }
                                v162 = v161;
                                v161 = (__int64 *)v161[2];
                              }
                              while ( v163 );
LABEL_269:
                              if ( v162 == &v343 || v162[4] > v160 )
                                goto LABEL_271;
                            }
                          }
                          v169 = v350;
                          if ( !v350 )
                          {
                            v169 = &v349;
                            if ( v351 == &v349 )
                            {
                              v169 = &v349;
                              v172 = 1;
                              goto LABEL_296;
                            }
                            goto LABEL_393;
                          }
                          while ( 1 )
                          {
                            v170 = v169[4];
                            v171 = (__int64 *)v169[3];
                            if ( v154 < v170 )
                              v171 = (__int64 *)v169[2];
                            if ( !v171 )
                              break;
                            v169 = v171;
                          }
                          if ( v154 < v170 )
                            break;
                          if ( v154 > v170 )
                            goto LABEL_294;
LABEL_271:
                          v150 += 24;
                          if ( v152 == v150 )
                            goto LABEL_272;
                        }
                        if ( v351 == v169 )
                          goto LABEL_294;
LABEL_393:
                        v315 = v151;
                        v323 = v169;
                        v219 = sub_220EF80(v169);
                        v169 = v323;
                        v151 = v315;
                        if ( *(_QWORD *)(v219 + 32) >= v154 )
                          goto LABEL_271;
LABEL_294:
                        v172 = 1;
                        if ( v169 != &v349 )
                          v172 = v154 < v169[4];
LABEL_296:
                        v308 = v151;
                        v150 += 24;
                        v313 = v169;
                        v320 = v172;
                        v173 = sub_22077B0(40);
                        *(_QWORD *)(v173 + 32) = v154;
                        sub_220F040(v320, v173, v313, &v349);
                        ++v353;
                        v151 = v308;
                      }
                      while ( v152 != v150 );
LABEL_272:
                      v101 = v151;
LABEL_273:
                      v165 = *(unsigned __int64 **)(v149 + 8);
                      if ( !v165 )
                      {
LABEL_146:
                        v101 = (__int64 *)sub_220EF30(v101);
                        if ( v101 == &v343 )
                          goto LABEL_147;
                        continue;
                      }
                      while ( 1 )
                      {
                        v166 = *v165;
                        if ( *(_BYTE *)(*v165 + 16) > 0x17u )
                        {
                          if ( !v344 )
                            break;
                          v167 = v344;
                          v168 = &v343;
                          do
                          {
                            if ( v167[4] < v166 )
                            {
                              v167 = (__int64 *)v167[3];
                            }
                            else
                            {
                              v168 = v167;
                              v167 = (__int64 *)v167[2];
                            }
                          }
                          while ( v167 );
                          if ( v168 == &v343 || v168[4] > v166 )
                            break;
                        }
                        v165 = (unsigned __int64 *)v165[1];
                        if ( !v165 )
                        {
                          v101 = (__int64 *)sub_220EF30(v101);
                          if ( v101 == &v343 )
                            goto LABEL_147;
                          goto LABEL_250;
                        }
                      }
                      v96 = (__int64 *)v394;
                      if ( v394 )
                      {
                        while ( 1 )
                        {
                          v97 = v96[4];
                          v98 = (__int64 *)v96[3];
                          if ( v149 < v97 )
                            v98 = (__int64 *)v96[2];
                          if ( !v98 )
                            break;
                          v96 = v98;
                        }
                        if ( v149 < v97 )
                        {
                          if ( v395 != v96 )
                            goto LABEL_299;
                        }
                        else if ( v149 <= v97 )
                        {
                          goto LABEL_146;
                        }
LABEL_143:
                        v99 = 1;
                        if ( v96 != &v393 )
                          v99 = v149 < v96[4];
                        goto LABEL_145;
                      }
                      v96 = &v393;
                      if ( v395 == &v393 )
                      {
                        v96 = &v393;
                        v99 = 1;
LABEL_145:
                        v100 = sub_22077B0(40);
                        *(_QWORD *)(v100 + 32) = v149;
                        sub_220F040(v99, v100, v96, &v393);
                        ++v397.m128i_i64[0];
                        goto LABEL_146;
                      }
LABEL_299:
                      if ( v149 > *(_QWORD *)(sub_220EF80(v96) + 32) )
                        goto LABEL_143;
                      v101 = (__int64 *)sub_220EF30(v101);
                      if ( v101 == &v343 )
                      {
LABEL_147:
                        if ( v397.m128i_i64[0] < v353 )
                          goto LABEL_148;
                        break;
                      }
                    }
                  }
                  v212 = sub_157EE30(v337);
                  v213 = v339;
                  v214 = (__int64 *)v212;
                  v215 = (v340 - v339) >> 3;
                  if ( (_DWORD)v215 )
                  {
                    v216 = 0;
                    v217 = 8LL * (unsigned int)(v215 - 1);
                    while ( 1 )
                    {
                      v218 = *(_QWORD **)&v213[v216];
                      sub_15F2240(v218, v337, v214);
                      v214 = v218 + 3;
                      if ( v216 == v217 )
                        break;
                      v213 = v339;
                      v216 += 8;
                    }
                  }
LABEL_148:
                  sub_193E770(v394);
                  sub_193E770((__int64)v350);
                }
                if ( v339 )
                  j_j___libc_free_0(v339, v341 - v339);
                sub_193E770((__int64)v344);
                goto LABEL_152;
              }
              v392 = v80 - 24;
LABEL_505:
              if ( (*(_DWORD *)(v80 - 4) & 0xFFFFFFF) == 0 )
                goto LABEL_511;
              v274 = 0;
              v275 = 0;
              v276 = 24LL * (*(_DWORD *)(v80 - 4) & 0xFFFFFFF);
              do
              {
                v277 = &v81[v276 / 0xFFFFFFFFFFFFFFF8LL];
                if ( (*(_BYTE *)(v80 - 1) & 0x40) != 0 )
                  v277 = *(_QWORD **)(v80 - 32);
                v274 -= (*(_BYTE *)(v277[v275 / 8] + 16LL) < 0x18u) - 1;
                v275 += 24LL;
              }
              while ( v276 != v275 );
              if ( v274 <= 1 )
              {
LABEL_511:
                v278 = *(_QWORD **)(v77 + 48);
                if ( (_QWORD *)v80 == v278 )
                {
LABEL_519:
                  sub_15F2240(v81, v337, v330);
                  goto LABEL_248;
                }
                while ( 1 )
                {
                  v279 = *v79 & 0xFFFFFFFFFFFFFFF8LL;
                  v79 = (_QWORD *)v279;
                  if ( !v279 )
                    BUG();
                  if ( *(_BYTE *)(v279 - 8) != 78 )
                    break;
                  v280 = *(_QWORD *)(v279 - 48);
                  if ( *(_BYTE *)(v280 + 16)
                    || (*(_BYTE *)(v280 + 33) & 0x20) == 0
                    || (unsigned int)(*(_DWORD *)(v280 + 36) - 35) > 3 )
                  {
                    break;
                  }
                  if ( v278 == v79 )
                    goto LABEL_519;
                }
                sub_15F2240(v81, v337, v330);
                v330 = (__int64 *)(v392 + 24);
                break;
              }
LABEL_122:
              v91 = v344;
              if ( !v344 )
              {
                v91 = &v343;
                if ( v345 == &v343 )
                {
                  v94 = 1;
LABEL_132:
                  v327 = v94;
                  v95 = sub_22077B0(40);
                  *(_QWORD *)(v95 + 32) = v392;
                  sub_220F040(v327, v95, v91, &v343);
                  ++v347;
                  goto LABEL_537;
                }
                goto LABEL_536;
              }
              while ( 1 )
              {
                v92 = v91[4];
                v93 = (__int64 *)v91[3];
                if ( (unsigned __int64)v81 < v92 )
                  v93 = (__int64 *)v91[2];
                if ( !v93 )
                  break;
                v91 = v93;
              }
              if ( (unsigned __int64)v81 < v92 )
              {
                if ( v91 != v345 )
                {
LABEL_536:
                  if ( *(_QWORD *)(sub_220EF80(v91) + 32) >= (unsigned __int64)v81 )
                  {
LABEL_537:
                    v290 = v340;
                    if ( v340 == v341 )
                    {
                      sub_170B610((__int64)&v339, v340, &v392);
                    }
                    else
                    {
                      if ( v340 )
                      {
                        *(_QWORD *)v340 = v392;
                        v290 = v340;
                      }
                      v340 = v290 + 8;
                    }
                    break;
                  }
                }
              }
              else if ( (unsigned __int64)v81 <= v92 )
              {
                goto LABEL_537;
              }
              v94 = 1;
              if ( v91 != &v343 )
                v94 = (unsigned __int64)v81 < v91[4];
              goto LABEL_132;
            }
            v175 = *(_QWORD *)(v80 - 48);
            if ( *(_BYTE *)(v175 + 16)
              || (*(_BYTE *)(v175 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v175 + 36) - 35) > 3 )
            {
              goto LABEL_109;
            }
          }
          break;
        }
        if ( *(_QWORD **)(v77 + 48) == v79 )
          goto LABEL_248;
        continue;
      }
    }
  }
LABEL_152:
  v392 = (unsigned __int64)&v394;
  v393 = 0x800000000LL;
  sub_13FA0E0(a2, (__int64)&v392);
  v102 = **(_QWORD **)(a2 + 32);
  v312 = (__int64 *)(v392 + 8LL * (unsigned int)v393);
  if ( (__int64 *)v392 == v312 )
    goto LABEL_189;
  v319 = (__int64 *)v392;
  do
  {
    v103 = sub_157F280(*v319);
    v338 = v104;
    v105 = v103;
    if ( v103 != v104 )
    {
      while ( 1 )
      {
        v106 = 0;
        v107 = 8LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF);
        if ( (*(_DWORD *)(v105 + 20) & 0xFFFFFFF) != 0 )
          break;
LABEL_183:
        v124 = *(_QWORD *)(v105 + 32);
        if ( !v124 )
          BUG();
        v105 = 0;
        if ( *(_BYTE *)(v124 - 8) == 77 )
          v105 = v124 - 24;
        if ( v338 == v105 )
          goto LABEL_187;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v105 + 23) & 0x40) != 0 )
            v108 = *(_QWORD *)(v105 - 8);
          else
            v108 = v105 - 24LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF);
          if ( v102 != *(_QWORD *)(v106 + v108 + 24LL * *(unsigned int *)(v105 + 56) + 8) )
            goto LABEL_159;
          v109 = sub_157EBA0(v102);
          v110 = *(_BYTE *)(v109 + 16);
          if ( v110 != 26 )
            break;
          v111 = *(_QWORD *)(v109 - 72);
LABEL_164:
          if ( !sub_13FC1A0(a2, v111) )
            goto LABEL_159;
          v112 = (*(_BYTE *)(v105 + 23) & 0x40) != 0
               ? *(_QWORD *)(v105 - 8)
               : v105 - 24LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF);
          if ( *(_BYTE *)(*(_QWORD *)(v112 + 3 * v106) + 16LL) != 77 )
            goto LABEL_159;
          v331 = *(_QWORD *)(v112 + 3 * v106);
          v113 = sub_13FC520(a2);
          v114 = *(_DWORD *)(v331 + 20) & 0xFFFFFFF;
          if ( !v114 )
            goto LABEL_159;
          v115 = 24LL * *(unsigned int *)(v331 + 56) + 8;
          v116 = 0;
          while ( 1 )
          {
            v117 = v331 - 24LL * v114;
            if ( (*(_BYTE *)(v331 + 23) & 0x40) != 0 )
              v117 = *(_QWORD *)(v331 - 8);
            if ( v113 == *(_QWORD *)(v117 + v115) )
              break;
            ++v116;
            v115 += 8;
            if ( v114 == (_DWORD)v116 )
              goto LABEL_159;
          }
          v118 = *(_QWORD *)(v117 + 24 * v116);
          if ( (*(_BYTE *)(v105 + 23) & 0x40) != 0 )
            v119 = *(_QWORD *)(v105 - 8);
          else
            v119 = v105 - 24LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF);
          v120 = (_QWORD *)(v119 + 3 * v106);
          if ( *v120 )
          {
            v121 = v120[1];
            v122 = v120[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v122 = v121;
            if ( v121 )
              *(_QWORD *)(v121 + 16) = *(_QWORD *)(v121 + 16) & 3LL | v122;
          }
          *v120 = v118;
          if ( !v118 )
            goto LABEL_159;
          v123 = *(_QWORD *)(v118 + 8);
          v120[1] = v123;
          if ( v123 )
            *(_QWORD *)(v123 + 16) = (unsigned __int64)(v120 + 1) | *(_QWORD *)(v123 + 16) & 3LL;
          v106 += 8;
          v120[2] = (v118 + 8) | v120[2] & 3LL;
          *(_QWORD *)(v118 + 8) = v120;
          if ( v106 == v107 )
            goto LABEL_183;
        }
        if ( v110 == 27 )
        {
          if ( (*(_BYTE *)(v109 + 23) & 0x40) != 0 )
            v132 = *(__int64 **)(v109 - 8);
          else
            v132 = (__int64 *)(v109 - 24LL * (*(_DWORD *)(v109 + 20) & 0xFFFFFFF));
          v111 = *v132;
          goto LABEL_164;
        }
LABEL_159:
        v106 += 8;
        if ( v106 == v107 )
          goto LABEL_183;
      }
    }
LABEL_187:
    ++v319;
  }
  while ( v312 != v319 );
  v312 = (__int64 *)v392;
LABEL_189:
  if ( v312 != (__int64 *)&v394 )
    _libc_free((unsigned __int64)v312);
  v125 = sub_1AA7010(**(_QWORD **)(a2 + 32), *(_QWORD *)(a1 + 32));
  LOBYTE(v125) = *(_BYTE *)(a1 + 448) | v125;
  v126 = v389;
  *(_BYTE *)(a1 + 448) = v125;
  v127 = v125;
  if ( v126 != v391 )
    _libc_free((unsigned __int64)v126);
  if ( v384[0] )
    sub_161E7C0((__int64)v384, v384[0]);
  j___libc_free_0(v380);
  if ( v373 != v372 )
    _libc_free((unsigned __int64)v373);
  j___libc_free_0(v368);
  j___libc_free_0(v364);
  j___libc_free_0(v360);
  if ( v358 )
  {
    v128 = v356;
    v129 = &v356[5 * v358];
    do
    {
      while ( *v128 == -8 )
      {
        if ( v128[1] != -8 )
          goto LABEL_200;
        v128 += 5;
        if ( v129 == v128 )
          goto LABEL_207;
      }
      if ( *v128 != -16 || v128[1] != -16 )
      {
LABEL_200:
        v130 = v128[4];
        if ( v130 != 0 && v130 != -8 && v130 != -16 )
          sub_1649B30(v128 + 2);
      }
      v128 += 5;
    }
    while ( v129 != v128 );
  }
LABEL_207:
  j___libc_free_0(v356);
  return v127;
}
