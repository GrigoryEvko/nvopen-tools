// Function: sub_BB0D30
// Address: 0xbb0d30
//
__int64 __fastcall sub_BB0D30(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // edx
  unsigned __int64 v4; // rax
  __int64 v5; // r12
  void *v6; // rdi
  const void *v7; // rsi
  unsigned __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // r8
  const void *v16; // rax
  size_t v17; // rdx
  void *v18; // rdi
  size_t v19; // r13
  __int64 v20; // r13
  int v21; // esi
  __int64 v22; // rcx
  int v23; // ebx
  _QWORD *v24; // r11
  __int64 v25; // rdx
  _QWORD *v26; // rax
  __int64 v27; // rdi
  __int64 *v28; // rdx
  __int64 *v29; // rsi
  __int64 v30; // rdi
  int v31; // r15d
  int v32; // eax
  char v33; // bl
  char v34; // r12
  __int64 v35; // rdx
  __int64 v36; // rcx
  char v37; // al
  char *v38; // rax
  unsigned __int64 v39; // rdx
  int v40; // ecx
  __int64 v41; // rsi
  __int64 v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // rdi
  _QWORD *v45; // rax
  _QWORD *v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // rcx
  unsigned __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rcx
  __int64 v53; // r8
  unsigned __int64 v54; // r15
  unsigned __int64 v55; // rbx
  unsigned __int64 v56; // rdx
  __int64 v57; // rsi
  int v58; // eax
  _BYTE *v59; // rdi
  __int64 v60; // r8
  const char *v61; // rcx
  int v62; // esi
  unsigned __int64 v63; // r10
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // r10
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  __int64 v69; // rsi
  int v70; // eax
  _BYTE *v71; // rdi
  int v72; // esi
  unsigned __int64 v73; // r10
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  unsigned __int64 v76; // r10
  __int64 v77; // rax
  __int64 v78; // rcx
  __m128i *v79; // rax
  __int64 v80; // rax
  __int64 v81; // r15
  __int64 v82; // rax
  __int64 v83; // rbx
  _QWORD *v84; // r12
  __int64 v85; // r8
  __int64 v86; // rax
  _DWORD *v87; // rdx
  int v88; // edi
  unsigned __int64 v89; // r12
  unsigned __int64 v90; // r13
  __int8 *v91; // rax
  __int8 *v92; // rsi
  unsigned __int64 v93; // rcx
  __int64 v94; // rdx
  int v95; // edx
  unsigned __int64 v96; // r12
  unsigned __int64 v97; // r13
  __int8 *v98; // rax
  __int8 *v99; // rsi
  __int64 v100; // rcx
  __int64 v101; // rdx
  __int64 *j; // r13
  __int64 v103; // r8
  unsigned __int64 *v104; // r12
  unsigned __int64 v105; // rcx
  unsigned __int64 *v106; // rbx
  unsigned __int64 *v107; // r14
  int v108; // ebx
  __int64 v109; // r15
  unsigned __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // rax
  __int64 v113; // r15
  unsigned __int64 v114; // rcx
  __int64 v115; // rax
  __int64 v116; // rax
  unsigned __int64 v117; // r15
  _DWORD *v118; // rdx
  unsigned __int64 v119; // rdx
  __int64 v120; // rsi
  int v121; // eax
  _BYTE *v122; // rdi
  const char *v123; // rcx
  int v124; // esi
  unsigned __int64 v125; // r9
  __int64 v126; // rax
  unsigned __int64 v127; // rdx
  unsigned __int64 v128; // r9
  __int64 v129; // r8
  __int64 v130; // rax
  unsigned __int64 v131; // rdx
  __int64 v132; // rsi
  int v133; // eax
  unsigned __int64 v134; // rsi
  _BYTE *v135; // r8
  int v136; // edi
  __int64 v137; // rdx
  unsigned __int64 v138; // rax
  char v139; // r11
  __int64 v140; // r10
  __m128i *v141; // rax
  __m128i *v142; // rcx
  __m128i *v143; // rdx
  __int64 v144; // rcx
  __m128i *v145; // rax
  _QWORD *v146; // rsi
  __int64 v147; // rdx
  unsigned __int64 v148; // rax
  unsigned __int64 v149; // rdi
  unsigned __int64 v150; // rcx
  __m128i *v151; // rax
  __m128i *v152; // rcx
  __m128i *v153; // rdx
  __int64 v154; // rax
  unsigned __int64 v155; // rcx
  __int64 v156; // r8
  _DWORD *v157; // rdx
  __int64 v158; // r13
  unsigned __int64 v159; // rdx
  __int64 v160; // rsi
  int v161; // eax
  _BYTE *v162; // rdi
  const char *v163; // rcx
  int v164; // esi
  unsigned __int64 v165; // r9
  __int64 v166; // rax
  unsigned __int64 v167; // rdx
  unsigned __int64 v168; // r9
  __int64 v169; // rax
  unsigned __int64 v170; // rdx
  __int64 v171; // rsi
  int v172; // eax
  _BYTE *v173; // rdi
  int v174; // esi
  unsigned __int64 v175; // r10
  __int64 v176; // rax
  unsigned __int64 v177; // rdx
  unsigned __int64 v178; // r10
  __int64 v179; // rax
  __m128i *v180; // rax
  __m128i *v181; // rcx
  __m128i *v182; // rdx
  __int64 v183; // rcx
  __m128i *v184; // rax
  _QWORD *v185; // rsi
  __int64 v186; // rdx
  unsigned __int64 v187; // rax
  unsigned __int64 v188; // rdi
  unsigned __int64 v189; // rcx
  __m128i *v190; // rax
  __int64 v191; // rcx
  __m128i *v192; // rdx
  __int64 v193; // r12
  size_t v194; // rax
  _BYTE *v195; // rdi
  size_t v196; // r13
  _BYTE *v197; // rax
  unsigned __int64 v198; // rdx
  __int64 v199; // rsi
  int v200; // eax
  _BYTE *v201; // rdi
  unsigned __int32 v202; // esi
  __int64 v203; // rdx
  unsigned __int64 v204; // rax
  char v205; // r9
  __int64 v206; // r8
  unsigned __int64 v207; // rdx
  __int64 v208; // rsi
  int v209; // eax
  __m128i *v210; // rdi
  int v211; // esi
  unsigned __int64 v212; // r9
  __int64 v213; // rax
  unsigned __int64 v214; // rdx
  unsigned __int64 v215; // r9
  __int64 v216; // rax
  unsigned __int64 v217; // r15
  __int64 v218; // rbx
  unsigned __int64 v219; // rcx
  __int64 v220; // r8
  __int64 v221; // rax
  __int64 v222; // r13
  unsigned __int64 v223; // rcx
  __int64 v224; // r8
  __int64 v225; // rax
  __int64 v226; // rax
  char *v227; // rax
  __int64 v228; // rax
  __int64 v229; // rdx
  unsigned __int64 v230; // rdx
  __int64 v231; // rsi
  int v232; // eax
  _BYTE *v233; // rdi
  unsigned __int32 v234; // esi
  unsigned __int64 v235; // r10
  __int64 v236; // rax
  unsigned __int64 v237; // rdx
  unsigned __int64 v238; // r10
  __int64 v239; // rax
  __int64 v240; // rdx
  unsigned __int64 v241; // r12
  unsigned __int64 v242; // r13
  __int8 *v243; // rax
  __int8 *v244; // rsi
  __int64 v245; // rcx
  __int64 v246; // rdx
  unsigned __int64 v247; // r15
  unsigned __int64 v248; // rcx
  __int64 v249; // r8
  __int64 v250; // rsi
  unsigned __int64 v251; // rdx
  int v252; // eax
  _BYTE *v253; // rdi
  const char *v254; // rcx
  unsigned __int32 v255; // esi
  unsigned __int64 v256; // r9
  __int64 v257; // rax
  unsigned __int64 v258; // rdx
  unsigned __int64 v259; // r9
  __int64 v260; // r8
  __int64 v261; // rax
  __int64 v262; // rsi
  unsigned __int64 v263; // rdx
  int v264; // eax
  char *v265; // r10
  unsigned __int64 v266; // rsi
  int v267; // edi
  unsigned __int64 v268; // r9
  __int64 v269; // rax
  unsigned __int64 v270; // rdx
  unsigned __int64 v271; // r9
  __int64 v272; // rax
  __int64 v273; // rcx
  __m128i *v274; // rax
  __int64 v275; // rax
  unsigned __int64 v276; // rcx
  __int64 v277; // r8
  __int64 v278; // rsi
  unsigned __int64 v279; // rdx
  int v280; // eax
  _BYTE *v281; // rdi
  const char *v282; // rcx
  int v283; // esi
  unsigned __int64 v284; // r9
  __int64 v285; // rax
  unsigned __int64 v286; // rdx
  unsigned __int64 v287; // r9
  __int64 v288; // r8
  __int64 v289; // rax
  __int64 v290; // rsi
  unsigned __int64 v291; // rdx
  int v292; // eax
  _BYTE *v293; // rdi
  int v294; // esi
  __int64 v295; // rdx
  unsigned __int64 v296; // rax
  char v297; // r9
  __int64 v298; // r8
  __int64 v299; // rcx
  __m128i *v300; // rax
  __int64 v301; // rax
  __int64 v302; // rax
  __int64 v303; // rsi
  unsigned __int64 v304; // rdx
  int v305; // eax
  _BYTE *v306; // rdi
  unsigned __int32 v307; // esi
  unsigned __int64 v308; // r9
  __int64 v309; // rax
  unsigned __int64 v310; // rdx
  unsigned __int64 v311; // r9
  __int64 v312; // rax
  __int64 v313; // rax
  unsigned __int64 v314; // rcx
  __int64 v315; // r8
  __int64 v316; // rsi
  unsigned __int64 v317; // rdx
  int v318; // eax
  _BYTE *v319; // rdi
  unsigned __int32 v320; // esi
  __int64 v321; // rdx
  unsigned __int64 v322; // rax
  char v323; // r9
  __int64 v324; // r8
  int v325; // edi
  int v326; // r8d
  int v327; // esi
  _QWORD *v328; // rbx
  int v329; // eax
  __int64 v330; // rax
  unsigned __int64 v331; // r9
  _QWORD *v332; // rax
  _QWORD *v333; // rdx
  __int64 v334; // rsi
  unsigned __int64 v335; // rax
  unsigned __int64 v336; // rdx
  _BYTE *v337; // rsi
  __m128i *v338; // rax
  __m128i v339; // xmm6
  __int64 v340; // rax
  _QWORD *v341; // r12
  _QWORD *v342; // rbx
  __int64 v343; // rdi
  __int64 v344; // rsi
  __int64 result; // rax
  _QWORD *v346; // rax
  __int64 *v347; // rcx
  __int64 v348; // rsi
  __int64 *v349; // rbx
  __int64 *v350; // r13
  __m128i *v351; // rdx
  __m128i *v352; // rsi
  __int64 v353; // rcx
  __int64 v354; // rax
  __int64 *v355; // rdx
  __int64 *v356; // rax
  __int64 v357; // rdx
  __m128i *v358; // r13
  __int64 v359; // rbx
  unsigned __int64 v360; // rax
  __m128i *v361; // rbx
  __m128i *v362; // rdi
  __m128i *v363; // r13
  __m128i *v364; // rbx
  unsigned int i; // r12d
  __int64 v366; // rax
  __m128i v367; // xmm7
  __int64 v370; // [rsp+48h] [rbp-4C8h]
  __int64 v371; // [rsp+50h] [rbp-4C0h]
  __int64 v372; // [rsp+50h] [rbp-4C0h]
  __int64 v373; // [rsp+50h] [rbp-4C0h]
  _DWORD *v374; // [rsp+58h] [rbp-4B8h]
  const __m128i *v375; // [rsp+60h] [rbp-4B0h]
  __int64 *v376; // [rsp+60h] [rbp-4B0h]
  __int64 v377; // [rsp+70h] [rbp-4A0h]
  char v378; // [rsp+70h] [rbp-4A0h]
  char v379; // [rsp+70h] [rbp-4A0h]
  char v380; // [rsp+70h] [rbp-4A0h]
  int v381; // [rsp+70h] [rbp-4A0h]
  int src; // [rsp+78h] [rbp-498h]
  const char *srca; // [rsp+78h] [rbp-498h]
  __int64 v385; // [rsp+88h] [rbp-488h]
  unsigned __int64 v386; // [rsp+88h] [rbp-488h]
  unsigned __int64 v387; // [rsp+90h] [rbp-480h]
  char *v388; // [rsp+98h] [rbp-478h]
  unsigned __int64 v389; // [rsp+98h] [rbp-478h]
  _DWORD *v390; // [rsp+A0h] [rbp-470h]
  unsigned __int64 v391; // [rsp+A0h] [rbp-470h]
  __int64 v392; // [rsp+A0h] [rbp-470h]
  char v393; // [rsp+BFh] [rbp-451h] BYREF
  __int64 v394; // [rsp+C0h] [rbp-450h] BYREF
  unsigned __int64 v395; // [rsp+C8h] [rbp-448h] BYREF
  _QWORD v396[2]; // [rsp+D0h] [rbp-440h] BYREF
  __int64 *v397; // [rsp+E0h] [rbp-430h] BYREF
  __int64 *v398; // [rsp+E8h] [rbp-428h]
  __int64 v399; // [rsp+F0h] [rbp-420h]
  void *v400; // [rsp+100h] [rbp-410h] BYREF
  __m128i *v401; // [rsp+108h] [rbp-408h]
  __m128i *v402; // [rsp+110h] [rbp-400h]
  __int64 v403; // [rsp+120h] [rbp-3F0h] BYREF
  _QWORD *v404; // [rsp+128h] [rbp-3E8h]
  __int64 v405; // [rsp+130h] [rbp-3E0h]
  unsigned int v406; // [rsp+138h] [rbp-3D8h]
  __int64 v407; // [rsp+140h] [rbp-3D0h] BYREF
  __int64 v408; // [rsp+148h] [rbp-3C8h]
  __int64 v409; // [rsp+150h] [rbp-3C0h]
  unsigned int v410; // [rsp+158h] [rbp-3B8h]
  const __m128i *v411; // [rsp+160h] [rbp-3B0h]
  __int64 **v412; // [rsp+168h] [rbp-3A8h]
  unsigned __int64 *v413; // [rsp+170h] [rbp-3A0h]
  _QWORD *v414; // [rsp+178h] [rbp-398h]
  __int64 v415[4]; // [rsp+180h] [rbp-390h] BYREF
  _BYTE *v416; // [rsp+1A0h] [rbp-370h] BYREF
  int v417; // [rsp+1A8h] [rbp-368h]
  _BYTE v418[16]; // [rsp+1B0h] [rbp-360h] BYREF
  __m128i v419[2]; // [rsp+1C0h] [rbp-350h] BYREF
  _BYTE *v420; // [rsp+1E0h] [rbp-330h] BYREF
  int v421; // [rsp+1E8h] [rbp-328h]
  _BYTE v422[16]; // [rsp+1F0h] [rbp-320h] BYREF
  __m128i *v423; // [rsp+200h] [rbp-310h] BYREF
  __int64 v424; // [rsp+208h] [rbp-308h]
  __m128i v425; // [rsp+210h] [rbp-300h] BYREF
  _QWORD v426[2]; // [rsp+220h] [rbp-2F0h] BYREF
  _QWORD v427[2]; // [rsp+230h] [rbp-2E0h] BYREF
  _QWORD *v428; // [rsp+240h] [rbp-2D0h] BYREF
  __int64 v429; // [rsp+248h] [rbp-2C8h]
  _QWORD v430[2]; // [rsp+250h] [rbp-2C0h] BYREF
  __m128i *v431; // [rsp+260h] [rbp-2B0h] BYREF
  __int64 v432; // [rsp+268h] [rbp-2A8h]
  __m128i v433; // [rsp+270h] [rbp-2A0h] BYREF
  _QWORD *v434; // [rsp+280h] [rbp-290h] BYREF
  __int64 v435; // [rsp+288h] [rbp-288h]
  _QWORD v436[2]; // [rsp+290h] [rbp-280h] BYREF
  _QWORD v437[2]; // [rsp+2A0h] [rbp-270h] BYREF
  _QWORD v438[2]; // [rsp+2B0h] [rbp-260h] BYREF
  _QWORD *v439; // [rsp+2C0h] [rbp-250h] BYREF
  __int64 v440; // [rsp+2C8h] [rbp-248h]
  _QWORD v441[2]; // [rsp+2D0h] [rbp-240h] BYREF
  __m128i *v442; // [rsp+2E0h] [rbp-230h] BYREF
  __int64 v443; // [rsp+2E8h] [rbp-228h]
  __m128i v444; // [rsp+2F0h] [rbp-220h] BYREF
  _QWORD *v445; // [rsp+300h] [rbp-210h] BYREF
  __int64 v446; // [rsp+308h] [rbp-208h]
  _QWORD v447[2]; // [rsp+310h] [rbp-200h] BYREF
  __m128i v448; // [rsp+320h] [rbp-1F0h] BYREF
  _QWORD v449[2]; // [rsp+330h] [rbp-1E0h] BYREF
  _QWORD v450[2]; // [rsp+340h] [rbp-1D0h] BYREF
  __int16 v451; // [rsp+350h] [rbp-1C0h] BYREF
  _BYTE *v452; // [rsp+360h] [rbp-1B0h] BYREF
  int v453; // [rsp+368h] [rbp-1A8h]
  _BYTE v454[16]; // [rsp+370h] [rbp-1A0h] BYREF
  __m128i v455[2]; // [rsp+380h] [rbp-190h] BYREF
  _BYTE *v456; // [rsp+3A0h] [rbp-170h] BYREF
  int v457; // [rsp+3A8h] [rbp-168h]
  _BYTE v458[16]; // [rsp+3B0h] [rbp-160h] BYREF
  __m128i v459; // [rsp+3C0h] [rbp-150h] BYREF
  _QWORD v460[2]; // [rsp+3D0h] [rbp-140h] BYREF
  char *v461; // [rsp+3E0h] [rbp-130h] BYREF
  int v462; // [rsp+3E8h] [rbp-128h]
  _BYTE v463[16]; // [rsp+3F0h] [rbp-120h] BYREF
  char v464; // [rsp+400h] [rbp-110h]
  char v465; // [rsp+401h] [rbp-10Fh]
  __m128i v466; // [rsp+410h] [rbp-100h] BYREF
  _QWORD v467[2]; // [rsp+420h] [rbp-F0h] BYREF
  __int16 v468; // [rsp+430h] [rbp-E0h]
  __m128i v469; // [rsp+440h] [rbp-D0h] BYREF
  __m128i v470; // [rsp+450h] [rbp-C0h] BYREF
  __int16 v471; // [rsp+460h] [rbp-B0h]
  __int64 v472; // [rsp+470h] [rbp-A0h] BYREF
  int v473; // [rsp+478h] [rbp-98h] BYREF
  _QWORD *v474; // [rsp+480h] [rbp-90h]
  const __m128i *v475; // [rsp+488h] [rbp-88h]
  int *v476; // [rsp+490h] [rbp-80h]
  __int64 v477; // [rsp+498h] [rbp-78h]
  __m128i v478; // [rsp+4A0h] [rbp-70h] BYREF
  __m128i v479; // [rsp+4B0h] [rbp-60h] BYREF
  __int64 v480; // [rsp+4C0h] [rbp-50h]
  _QWORD v481[9]; // [rsp+4C8h] [rbp-48h] BYREF

  v397 = 0;
  v398 = 0;
  v399 = 0;
  v403 = 0;
  v404 = 0;
  v405 = 0;
  v406 = 0;
  v473 = 0;
  v474 = 0;
  v475 = (const __m128i *)&v473;
  v476 = &v473;
  v477 = 0;
  sub_BAFF00(a1, &v472);
  v3 = *(_DWORD *)(a1 + 56);
  v400 = 0;
  v401 = 0;
  v402 = 0;
  if ( !v3 )
    goto LABEL_2;
  v346 = *(_QWORD **)(a1 + 48);
  if ( *v346 != -8 && *v346 )
  {
    v349 = *(__int64 **)(a1 + 48);
  }
  else
  {
    v347 = v346 + 1;
    do
    {
      do
      {
        v348 = *v347;
        v349 = v347++;
      }
      while ( v348 == -8 );
    }
    while ( !v348 );
  }
  v350 = &v346[v3];
  if ( v350 == v349 )
    goto LABEL_2;
  v351 = 0;
  v352 = 0;
  while ( 1 )
  {
    v353 = *(_QWORD *)*v349;
    v478.m128i_i64[0] = *v349 + 32;
    v478.m128i_i64[1] = v353;
    if ( v352 == v351 )
    {
      sub_A04210((const __m128i **)&v400, v352, &v478);
      v352 = v401;
    }
    else
    {
      if ( v352 )
      {
        *v352 = _mm_load_si128(&v478);
        v352 = v401;
      }
      v401 = ++v352;
    }
    v354 = v349[1];
    v355 = v349 + 1;
    if ( v354 == -8 || !v354 )
      break;
    ++v349;
    if ( v355 == v350 )
      goto LABEL_641;
LABEL_638:
    v351 = v402;
  }
  v356 = v349 + 2;
  do
  {
    do
    {
      v357 = *v356;
      v349 = v356++;
    }
    while ( v357 == -8 );
  }
  while ( !v357 );
  if ( v349 != v350 )
    goto LABEL_638;
LABEL_641:
  v358 = (__m128i *)v400;
  if ( v352 != v400 )
  {
    v359 = (char *)v352 - (_BYTE *)v400;
    _BitScanReverse64(&v360, ((char *)v352 - (_BYTE *)v400) >> 4);
    sub_BB0AE0((__m128i *)v400, v352, 2LL * (int)(63 - (v360 ^ 0x3F)));
    if ( v359 <= 256 )
    {
      sub_A3B670(v358, v352);
    }
    else
    {
      v361 = v358 + 16;
      sub_A3B670(v358, v358 + 16);
      if ( &v358[16] != v352 )
      {
        do
        {
          v362 = v361++;
          sub_A3B600(v362);
        }
        while ( v352 != v361 );
      }
    }
    v363 = (__m128i *)v400;
    v364 = v401;
    v407 = 0;
    v408 = 0;
    v409 = 0;
    v410 = 0;
    if ( v400 != v401 )
    {
      for ( i = 0; ; i = v409 )
      {
        if ( !(unsigned __int8)sub_A19D80((__int64)&v407, v363, &v478) )
        {
          v366 = sub_BB0290((__int64)&v407, v363, v478.m128i_i64[0]);
          v367 = _mm_loadu_si128(v363);
          *(_QWORD *)(v366 + 16) = i;
          *(__m128i *)v366 = v367;
        }
        if ( v364 == ++v363 )
          break;
      }
    }
    goto LABEL_3;
  }
LABEL_2:
  v407 = 0;
  v408 = 0;
  v409 = 0;
  v410 = 0;
LABEL_3:
  v396[1] = &v393;
  v396[0] = a2;
  sub_904010(a2, "digraph Summary {\n");
  v375 = v475;
  if ( v475 == (const __m128i *)&v473 )
    goto LABEL_113;
  while ( 2 )
  {
    v4 = 0;
    if ( (_DWORD)v409 )
    {
      if ( (unsigned __int8)sub_A19D80((__int64)&v407, (const __m128i *)v375[2].m128i_i64, &v478) )
      {
        v338 = (__m128i *)(v478.m128i_i64[0] + 16);
      }
      else
      {
        v338 = (__m128i *)(sub_BB0290((__int64)&v407, (const __m128i *)v375[2].m128i_i64, v478.m128i_i64[0]) + 16);
        v339 = _mm_loadu_si128(v375 + 2);
        v338->m128i_i64[0] = 0;
        v338[-1] = v339;
      }
      v4 = v338->m128i_i64[0];
    }
    v394 = v4;
    v5 = sub_904010(a2, "  // Module: ");
    v6 = *(void **)(v5 + 32);
    v7 = (const void *)v375[2].m128i_i64[0];
    v8 = v375[2].m128i_u64[1];
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 < v8 )
    {
      v5 = sub_CB6200(v5, v7, v375[2].m128i_i64[1]);
    }
    else if ( v8 )
    {
      memcpy(v6, v7, v375[2].m128i_u64[1]);
      *(_QWORD *)(v5 + 32) += v8;
    }
    sub_904010(v5, "\n");
    v9 = sub_904010(a2, "  subgraph cluster_");
    sub_BAE5F0((__int64)&v478, v394, v10, v11);
    v12 = sub_CB6200(v9, v478.m128i_i64[0], v478.m128i_i64[1]);
    sub_904010(v12, " {\n");
    sub_2240A30(&v478);
    sub_904010(a2, "    style = filled;\n");
    sub_904010(a2, "    color = lightgrey;\n");
    v13 = sub_904010(a2, "    label = \"");
    v16 = (const void *)sub_C80C60(v375[2].m128i_i64[0], v375[2].m128i_i64[1], 0, v14, v15);
    v18 = *(void **)(v13 + 32);
    v19 = v17;
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v18 < v17 )
    {
      v13 = sub_CB6200(v13, v16, v17);
    }
    else if ( v17 )
    {
      memcpy(v18, v16, v17);
      *(_QWORD *)(v13 + 32) += v19;
    }
    sub_904010(v13, "\";\n");
    sub_904010(a2, "    node [style=filled,fillcolor=lightblue];\n");
    v20 = v375[4].m128i_i64[1];
    v411 = v375 + 3;
    v412 = &v397;
    v413 = (unsigned __int64 *)&v394;
    v414 = v396;
    v388 = &v375[3].m128i_i8[8];
    if ( &v375[3].m128i_u64[1] != (unsigned __int64 *)v20 )
    {
      while ( 2 )
      {
        v21 = v406;
        if ( v406 )
        {
          v22 = *(_QWORD *)(v20 + 32);
          v23 = 1;
          v24 = 0;
          LODWORD(v25) = (v406 - 1) & (((0xBF58476D1CE4E5B9LL * v22) >> 31) ^ (484763065 * v22));
          v26 = &v404[4 * (unsigned int)v25];
          v27 = *v26;
          if ( v22 == *v26 )
          {
LABEL_14:
            v28 = (__int64 *)v26[2];
            v29 = (__int64 *)v26[3];
            v30 = (__int64)(v26 + 1);
            if ( v28 != v29 )
            {
              if ( v28 )
              {
                *v28 = v394;
                v28 = (__int64 *)v26[2];
              }
              v26[2] = v28 + 1;
LABEL_18:
              v31 = *(_DWORD *)(*(_QWORD *)(v20 + 40) + 12LL);
              LOBYTE(v481[0]) = 0;
              v478 = 0u;
              v479.m128i_i64[0] = 0;
              v479.m128i_i64[1] = (__int64)v481;
              v480 = 0;
              v32 = *(_DWORD *)(*(_QWORD *)(v20 + 40) + 8LL);
              if ( v32 == 1 )
              {
                v33 = v31;
                v34 = v31;
                v469.m128i_i64[0] = (__int64)"function";
                v466.m128i_i64[0] = (__int64)"record";
                v471 = 259;
                v468 = 259;
                v465 = 1;
                v461 = "shape";
                v464 = 3;
                sub_BAF0D0((__int64)&v478, (__int64)&v461, (__int64)&v466, (__int64)&v469);
                goto LABEL_21;
              }
              HIBYTE(v471) = 1;
              v33 = v31;
              v34 = v31;
              if ( !v32 )
              {
                v469.m128i_i64[0] = (__int64)"alias";
                v466.m128i_i64[0] = (__int64)"dotted,filled";
                v461 = "style";
                LOBYTE(v471) = 3;
                v468 = 259;
                v465 = 1;
                v464 = 3;
                sub_BAF0D0((__int64)&v478, (__int64)&v461, (__int64)&v466, (__int64)&v469);
                v466.m128i_i64[0] = (__int64)"box";
                v471 = 257;
                v468 = 259;
                v465 = 1;
                v461 = "shape";
                v464 = 3;
                sub_BAF0D0((__int64)&v478, (__int64)&v461, (__int64)&v466, (__int64)&v469);
                goto LABEL_21;
              }
              v469.m128i_i64[0] = (__int64)"variable";
              v466.m128i_i64[0] = (__int64)"Mrecord";
              LOBYTE(v471) = 3;
              v468 = 259;
              v465 = 1;
              v461 = "shape";
              v464 = 3;
              sub_BAF0D0((__int64)&v478, (__int64)&v461, (__int64)&v466, (__int64)&v469);
              if ( (v31 & 0x80u) != 0 && (v228 = *(_QWORD *)(v20 + 40), *(_DWORD *)(v228 + 8) == 2) )
              {
                v229 = *(unsigned __int8 *)(v228 + 64);
                if ( (v229 & 1) != 0 )
                {
                  v469.m128i_i64[0] = (__int64)"immutable";
                  v471 = 259;
                  sub_BAC200((__int64)&v478, (__int64)&v469, v229, v36);
                  v228 = *(_QWORD *)(v20 + 40);
                  if ( *(_DWORD *)(v228 + 8) != 2 )
                    goto LABEL_21;
                  v35 = *(_BYTE *)(v228 + 64) & 2;
                  if ( (*(_BYTE *)(v228 + 64) & 2) != 0 )
                    goto LABEL_589;
                }
                else
                {
                  v35 = v229 & 2;
                  if ( !(_DWORD)v35 )
                    goto LABEL_342;
LABEL_589:
                  v469.m128i_i64[0] = (__int64)"writeOnly";
                  v471 = 259;
                  sub_BAC200((__int64)&v478, (__int64)&v469, v35, v36);
                  v228 = *(_QWORD *)(v20 + 40);
                  if ( *(_DWORD *)(v228 + 8) != 2 )
                    goto LABEL_21;
                }
LABEL_342:
                if ( (*(_BYTE *)(v228 + 64) & 4) == 0 )
                  goto LABEL_21;
                v469.m128i_i64[0] = (__int64)"constant";
                v471 = 259;
                sub_BAC200((__int64)&v478, (__int64)&v469, v35, v36);
                if ( (v31 & 0x30) != 0 )
                {
LABEL_344:
                  v469.m128i_i64[0] = (__int64)"visibility";
                  v471 = 259;
                  sub_BAC200((__int64)&v478, (__int64)&v469, v35, v36);
                  v37 = BYTE1(v31);
                  if ( (v31 & 0x100) != 0 )
                  {
LABEL_345:
                    v379 = v37;
                    v469.m128i_i64[0] = (__int64)"dsoLocal";
                    v471 = 259;
                    sub_BAC200((__int64)&v478, (__int64)&v469, v35, v36);
                    v37 = v379;
                    if ( (v379 & 2) != 0 )
                    {
LABEL_346:
                      v380 = v37;
                      v469.m128i_i64[0] = (__int64)"canAutoHide";
                      v471 = 259;
                      sub_BAC200((__int64)&v478, (__int64)&v469, v35, v36);
                      v37 = v380;
                    }
LABEL_24:
                    HIBYTE(v471) = 1;
                    if ( (v37 & 4) != 0 )
                      v38 = "declaration";
                    else
                      v38 = "definition";
                    v469.m128i_i64[0] = (__int64)v38;
                    LOBYTE(v471) = 3;
                    sub_BAC200((__int64)&v478, (__int64)&v469, v35, v36);
                    v39 = *(_QWORD *)(v20 + 32);
                    v40 = *(_DWORD *)(a3 + 24);
                    v41 = *(_QWORD *)(a3 + 8);
                    if ( v40 )
                    {
                      v42 = (unsigned int)(v40 - 1);
                      v43 = v42 & (((0xBF58476D1CE4E5B9LL * v39) >> 31) ^ (484763065 * v39));
                      v44 = *(_QWORD *)(v41 + 8LL * v43);
                      if ( v44 == v39 )
                      {
LABEL_28:
                        v469.m128i_i64[0] = (__int64)"preserved";
                        v471 = 259;
                        sub_BAC200((__int64)&v478, (__int64)&v469, v39, v42);
                        v39 = *(_QWORD *)(v20 + 32);
                      }
                      else
                      {
                        v326 = 1;
                        while ( v44 != -1 )
                        {
                          v43 = v42 & (v326 + v43);
                          v44 = *(_QWORD *)(v41 + 8LL * v43);
                          if ( v44 == v39 )
                            goto LABEL_28;
                          ++v326;
                        }
                      }
                    }
                    v45 = *(_QWORD **)(a1 + 16);
                    if ( v45 )
                    {
                      v46 = (_QWORD *)(a1 + 8);
                      do
                      {
                        while ( 1 )
                        {
                          v47 = v45[2];
                          v48 = v45[3];
                          if ( v45[4] >= v39 )
                            break;
                          v45 = (_QWORD *)v45[3];
                          if ( !v48 )
                            goto LABEL_34;
                        }
                        v46 = v45;
                        v45 = (_QWORD *)v45[2];
                      }
                      while ( v47 );
LABEL_34:
                      v49 = 0;
                      if ( v46 != (_QWORD *)(a1 + 8) && v46[4] <= v39 )
                        v49 = (unsigned __int64)(v46 + 4) & 0xFFFFFFFFFFFFFFF8LL;
                    }
                    else
                    {
                      v49 = 0;
                    }
                    v471 = 257;
                    v50 = *(_QWORD *)(v20 + 40);
                    v395 = v49 | *(unsigned __int8 *)(a1 + 343);
                    sub_BAD810(&v459, &v395, v50, v49, 257);
                    v466.m128i_i64[0] = (__int64)&v459;
                    v468 = 260;
                    v465 = 1;
                    v461 = "label";
                    v464 = 3;
                    sub_BAF0D0((__int64)&v478, (__int64)&v461, (__int64)&v466, (__int64)&v469);
                    if ( (_QWORD *)v459.m128i_i64[0] != v460 )
                      j_j___libc_free_0(v459.m128i_i64[0], v460[0] + 1LL);
                    if ( v34 >= 0 )
                    {
                      v471 = 259;
                      HIBYTE(v468) = 1;
                      v469.m128i_i64[0] = (__int64)"dead";
                      v227 = "red";
                    }
                    else
                    {
                      if ( (v33 & 0x40) == 0 )
                        goto LABEL_41;
                      v469.m128i_i64[0] = (__int64)"not eligible to import";
                      v227 = "yellow";
                      v471 = 259;
                      HIBYTE(v468) = 1;
                    }
                    v466.m128i_i64[0] = (__int64)v227;
                    LOBYTE(v468) = 3;
                    v465 = 1;
                    v461 = "fillcolor";
                    v464 = 3;
                    sub_BAF0D0((__int64)&v478, (__int64)&v461, (__int64)&v466, (__int64)&v469);
LABEL_41:
                    v51 = sub_904010(a2, "    ");
                    v54 = v394;
                    v55 = *(_QWORD *)(v20 + 32);
                    v377 = v51;
                    if ( v394 != -1 )
                    {
                      if ( v55 <= 9 )
                      {
                        v420 = v422;
                        sub_2240A50(&v420, 1, 0, v52, v53);
                        v59 = v420;
                        goto LABEL_55;
                      }
                      if ( v55 <= 0x63 )
                      {
                        v420 = v422;
                        sub_2240A50(&v420, 2, 0, v52, v53);
                        v59 = v420;
                        v61 = "000102030405060708091011121314151617181920212223242526272829303132333435363738394041424344"
                              "454647484950515253545556575859606162636465666768697071727374757677787980818283848586878889"
                              "90919293949596979899";
                      }
                      else
                      {
                        if ( v55 <= 0x3E7 )
                        {
                          v57 = 3;
                        }
                        else if ( v55 <= 0x270F )
                        {
                          v57 = 4;
                        }
                        else
                        {
                          v56 = *(_QWORD *)(v20 + 32);
                          LODWORD(v57) = 1;
                          while ( 1 )
                          {
                            v52 = v56;
                            v58 = v57;
                            v57 = (unsigned int)(v57 + 4);
                            v56 /= 0x2710u;
                            if ( v52 <= 0x1869F )
                              break;
                            if ( v52 <= 0xF423F )
                            {
                              v57 = (unsigned int)(v58 + 5);
                              v420 = v422;
                              goto LABEL_52;
                            }
                            if ( v52 <= (unsigned __int64)&loc_98967F )
                            {
                              v57 = (unsigned int)(v58 + 6);
                              break;
                            }
                            if ( v52 <= 0x5F5E0FF )
                            {
                              v57 = (unsigned int)(v58 + 7);
                              break;
                            }
                          }
                        }
                        v420 = v422;
LABEL_52:
                        sub_2240A50(&v420, v57, 0, v52, v53);
                        v59 = v420;
                        v60 = 0x28F5C28F5C28F5C3LL;
                        v61 = "000102030405060708091011121314151617181920212223242526272829303132333435363738394041424344"
                              "454647484950515253545556575859606162636465666768697071727374757677787980818283848586878889"
                              "90919293949596979899";
                        v62 = v421 - 1;
                        do
                        {
                          v63 = v55;
                          v64 = 5
                              * (v55 / 0x64
                               + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v55 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                          v65 = v55;
                          v55 /= 0x64u;
                          v66 = v63 - 4 * v64;
                          v59[v62] = a00010203040506_0[2 * v66 + 1];
                          v67 = (unsigned int)(v62 - 1);
                          v62 -= 2;
                          v59[v67] = a00010203040506_0[2 * v66];
                        }
                        while ( v65 > 0x270F );
                        if ( v65 <= 0x3E7 )
                        {
LABEL_55:
                          *v59 = v55 + 48;
                          if ( v54 > 9 )
                            goto LABEL_56;
LABEL_363:
                          v416 = v418;
                          sub_2240A50(&v416, 1, 0, v61, v60);
                          v71 = v416;
LABEL_68:
                          *v71 = v54 + 48;
                          goto LABEL_69;
                        }
                      }
                      v59[1] = a00010203040506_0[2 * v55 + 1];
                      *v59 = a00010203040506_0[2 * v55];
                      if ( v54 <= 9 )
                        goto LABEL_363;
LABEL_56:
                      if ( v54 <= 0x63 )
                      {
                        v416 = v418;
                        sub_2240A50(&v416, 2, 0, v61, v60);
                        v71 = v416;
                      }
                      else
                      {
                        if ( v54 <= 0x3E7 )
                        {
                          v69 = 3;
                        }
                        else if ( v54 <= 0x270F )
                        {
                          v69 = 4;
                        }
                        else
                        {
                          v68 = v54;
                          LODWORD(v69) = 1;
                          while ( 1 )
                          {
                            v61 = (const char *)v68;
                            v70 = v69;
                            v69 = (unsigned int)(v69 + 4);
                            v68 /= 0x2710u;
                            if ( (unsigned __int64)v61 <= 0x1869F )
                              break;
                            if ( (unsigned __int64)v61 <= 0xF423F )
                            {
                              v416 = v418;
                              v69 = (unsigned int)(v70 + 5);
                              goto LABEL_65;
                            }
                            if ( v61 <= (const char *)&loc_98967F )
                            {
                              v69 = (unsigned int)(v70 + 6);
                              break;
                            }
                            if ( (unsigned __int64)v61 <= 0x5F5E0FF )
                            {
                              v69 = (unsigned int)(v70 + 7);
                              break;
                            }
                          }
                        }
                        v416 = v418;
LABEL_65:
                        sub_2240A50(&v416, v69, 0, v61, v60);
                        v71 = v416;
                        v72 = v417 - 1;
                        do
                        {
                          v73 = v54;
                          v74 = 5
                              * (v54 / 0x64
                               + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v54 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                          v75 = v54;
                          v54 /= 0x64u;
                          v76 = v73 - 4 * v74;
                          v71[v72] = a00010203040506_0[2 * v76 + 1];
                          v77 = (unsigned int)(v72 - 1);
                          v72 -= 2;
                          v71[v77] = a00010203040506_0[2 * v76];
                        }
                        while ( v75 > 0x270F );
                        if ( v75 <= 0x3E7 )
                          goto LABEL_68;
                      }
                      v71[1] = a00010203040506_0[2 * v54 + 1];
                      *v71 = a00010203040506_0[2 * v54];
LABEL_69:
                      sub_BAC150(v415, "M");
                      sub_8FD5D0(v419, (__int64)v415, &v416);
                      if ( v419[0].m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
LABEL_661:
                        sub_4262D8((__int64)"basic_string::append");
                      v79 = (__m128i *)sub_2241490(v419, "_", 1, v78);
                      v469.m128i_i64[0] = (__int64)&v470;
                      if ( (__m128i *)v79->m128i_i64[0] == &v79[1] )
                      {
                        v470 = _mm_loadu_si128(v79 + 1);
                      }
                      else
                      {
                        v469.m128i_i64[0] = v79->m128i_i64[0];
                        v470.m128i_i64[0] = v79[1].m128i_i64[0];
                      }
                      v469.m128i_i64[1] = v79->m128i_i64[1];
                      v79->m128i_i64[0] = (__int64)v79[1].m128i_i64;
                      v79->m128i_i64[1] = 0;
                      v79[1].m128i_i8[0] = 0;
                      sub_8FD5D0(&v466, (__int64)&v469, &v420);
                      sub_2240A30(&v469);
                      sub_2240A30(v419);
                      sub_2240A30(v415);
                      sub_2240A30(&v416);
                      sub_2240A30(&v420);
LABEL_73:
                      v80 = sub_CB6200(v377, v466.m128i_i64[0], v466.m128i_i64[1]);
                      v81 = sub_904010(v80, " ");
                      sub_BAC2C0((__int64)&v469, v478.m128i_i64);
                      v82 = sub_CB6200(v81, v469.m128i_i64[0], v469.m128i_i64[1]);
                      sub_904010(v82, "\n");
                      if ( (__m128i *)v469.m128i_i64[0] != &v470 )
                        j_j___libc_free_0(v469.m128i_i64[0], v470.m128i_i64[0] + 1);
                      if ( (_QWORD *)v466.m128i_i64[0] != v467 )
                        j_j___libc_free_0(v466.m128i_i64[0], v467[0] + 1LL);
                      if ( (_QWORD *)v479.m128i_i64[1] != v481 )
                        j_j___libc_free_0(v479.m128i_i64[1], v481[0] + 1LL);
                      v83 = v478.m128i_i64[1];
                      v84 = (_QWORD *)v478.m128i_i64[0];
                      if ( v478.m128i_i64[1] != v478.m128i_i64[0] )
                      {
                        do
                        {
                          if ( (_QWORD *)*v84 != v84 + 2 )
                            j_j___libc_free_0(*v84, v84[2] + 1LL);
                          v84 += 4;
                        }
                        while ( (_QWORD *)v83 != v84 );
                        v84 = (_QWORD *)v478.m128i_i64[0];
                      }
                      if ( v84 )
                        j_j___libc_free_0(v84, v479.m128i_i64[0] - (_QWORD)v84);
                      v20 = sub_220EEE0(v20);
                      if ( v388 == (char *)v20 )
                        goto LABEL_87;
                      continue;
                    }
                    if ( v55 > 9 )
                    {
                      if ( v55 <= 0x63 )
                      {
                        v466.m128i_i64[0] = (__int64)v467;
                        sub_2240A50(&v466, 2, 0, v52, v53);
                        v233 = (_BYTE *)v466.m128i_i64[0];
                      }
                      else
                      {
                        if ( v55 <= 0x3E7 )
                        {
                          v231 = 3;
                        }
                        else if ( v55 <= 0x270F )
                        {
                          v231 = 4;
                        }
                        else
                        {
                          v230 = *(_QWORD *)(v20 + 32);
                          LODWORD(v231) = 1;
                          while ( 1 )
                          {
                            v52 = v230;
                            v232 = v231;
                            v231 = (unsigned int)(v231 + 4);
                            v230 /= 0x2710u;
                            if ( v52 <= 0x1869F )
                              break;
                            if ( v52 <= 0xF423F )
                            {
                              v231 = (unsigned int)(v232 + 5);
                              v466.m128i_i64[0] = (__int64)v467;
                              goto LABEL_357;
                            }
                            if ( v52 <= (unsigned __int64)&loc_98967F )
                            {
                              v231 = (unsigned int)(v232 + 6);
                              break;
                            }
                            if ( v52 <= 0x5F5E0FF )
                            {
                              v231 = (unsigned int)(v232 + 7);
                              break;
                            }
                          }
                        }
                        v466.m128i_i64[0] = (__int64)v467;
LABEL_357:
                        sub_2240A50(&v466, v231, 0, v52, v53);
                        v233 = (_BYTE *)v466.m128i_i64[0];
                        v234 = v466.m128i_i32[2] - 1;
                        do
                        {
                          v235 = v55;
                          v236 = 5
                               * (v55 / 0x64
                                + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v55 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                          v237 = v55;
                          v55 /= 0x64u;
                          v238 = v235 - 4 * v236;
                          v233[v234] = a00010203040506_0[2 * v238 + 1];
                          v239 = v234 - 1;
                          v234 -= 2;
                          v233[v239] = a00010203040506_0[2 * v238];
                        }
                        while ( v237 > 0x270F );
                        if ( v237 <= 0x3E7 )
                          goto LABEL_360;
                      }
                      v233[1] = a00010203040506_0[2 * v55 + 1];
                      *v233 = a00010203040506_0[2 * v55];
                      goto LABEL_73;
                    }
                    v466.m128i_i64[0] = (__int64)v467;
                    sub_2240A50(&v466, 1, 0, v52, v53);
                    v233 = (_BYTE *)v466.m128i_i64[0];
LABEL_360:
                    *v233 = v55 + 48;
                    goto LABEL_73;
                  }
LABEL_23:
                  if ( (v37 & 2) != 0 )
                    goto LABEL_346;
                  goto LABEL_24;
                }
              }
              else
              {
LABEL_21:
                if ( (v33 & 0x30) != 0 )
                  goto LABEL_344;
              }
              v37 = BYTE1(v31);
              if ( (v31 & 0x100) != 0 )
                goto LABEL_345;
              goto LABEL_23;
            }
LABEL_511:
            sub_9CA200(v30, v29, &v394);
            goto LABEL_18;
          }
          while ( v27 != -1 )
          {
            if ( v27 == -2 && !v24 )
              v24 = v26;
            v25 = (v406 - 1) & ((_DWORD)v25 + v23);
            v26 = &v404[4 * v25];
            v27 = *v26;
            if ( v22 == *v26 )
              goto LABEL_14;
            ++v23;
          }
          if ( v24 )
            v26 = v24;
          ++v403;
          v325 = v405 + 1;
          v478.m128i_i64[0] = (__int64)v26;
          if ( 4 * ((int)v405 + 1) < 3 * v406 )
          {
            if ( v406 - HIDWORD(v405) - v325 > v406 >> 3 )
            {
LABEL_508:
              LODWORD(v405) = v325;
              if ( *v26 != -1 )
                --HIDWORD(v405);
              *v26 = v22;
              v30 = (__int64)(v26 + 1);
              v29 = 0;
              v26[1] = 0;
              v26[2] = 0;
              v26[3] = 0;
              goto LABEL_511;
            }
LABEL_556:
            sub_BB04D0((__int64)&v403, v21);
            sub_BAF210((__int64)&v403, (__int64 *)(v20 + 32), &v478);
            v22 = *(_QWORD *)(v20 + 32);
            v325 = v405 + 1;
            v26 = (_QWORD *)v478.m128i_i64[0];
            goto LABEL_508;
          }
        }
        else
        {
          ++v403;
          v478.m128i_i64[0] = 0;
        }
        break;
      }
      v21 = 2 * v406;
      goto LABEL_556;
    }
LABEL_87:
    sub_904010(a2, "    // Edges:\n");
    v385 = v375[4].m128i_i64[1];
    if ( v388 == (char *)v385 )
      goto LABEL_112;
    while ( 2 )
    {
      v86 = *(_QWORD *)(v385 + 40);
      v87 = *(_DWORD **)(v86 + 40);
      v374 = &v87[2 * *(unsigned int *)(v86 + 48)];
      if ( v374 == v87 )
        goto LABEL_102;
      v390 = *(_DWORD **)(v86 + 40);
      while ( 2 )
      {
        v88 = -1;
        if ( (*(_QWORD *)v390 & 4) == 0 )
          v88 = ((*v390 & 2) != 0) - 3;
        v89 = *(_QWORD *)(*(_QWORD *)v390 & 0xFFFFFFFFFFFFFFF8LL);
        v90 = *(_QWORD *)(v385 + 32);
        v91 = (__int8 *)v375[4].m128i_i64[0];
        if ( !v91 )
          goto LABEL_99;
        v92 = &v375[3].m128i_i8[8];
        do
        {
          while ( 1 )
          {
            v93 = *((_QWORD *)v91 + 2);
            v94 = *((_QWORD *)v91 + 3);
            if ( v89 <= *((_QWORD *)v91 + 4) )
              break;
            v91 = (__int8 *)*((_QWORD *)v91 + 3);
            if ( !v94 )
              goto LABEL_97;
          }
          v92 = v91;
          v91 = (__int8 *)*((_QWORD *)v91 + 2);
        }
        while ( v93 );
LABEL_97:
        if ( v388 == v92 || v89 < *((_QWORD *)v92 + 4) )
        {
LABEL_99:
          v478.m128i_i32[2] = v88;
          v479.m128i_i64[0] = v90;
          v478.m128i_i64[0] = v394;
          v479.m128i_i64[1] = v89;
          sub_BABCF0((__int64)&v397, &v478);
          goto LABEL_100;
        }
        v117 = v394;
        src = v88 + 4;
        v118 = *(_DWORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v118 <= 3u )
        {
          v371 = sub_CB6200(a2, "    ", 4);
        }
        else
        {
          *v118 = 538976288;
          *(_QWORD *)(a2 + 32) += 4LL;
          v371 = a2;
        }
        if ( v117 == -1 )
        {
          if ( v90 > 9 )
          {
            if ( v90 <= 0x63 )
            {
              v423 = &v425;
              sub_2240A50(&v423, 2, 0, v93, v85);
              v210 = v423;
            }
            else
            {
              if ( v90 <= 0x3E7 )
              {
                v208 = 3;
              }
              else if ( v90 <= 0x270F )
              {
                v208 = 4;
              }
              else
              {
                v207 = v90;
                LODWORD(v208) = 1;
                while ( 1 )
                {
                  v93 = v207;
                  v209 = v208;
                  v208 = (unsigned int)(v208 + 4);
                  v207 /= 0x2710u;
                  if ( v93 <= 0x1869F )
                    break;
                  if ( v93 <= 0xF423F )
                  {
                    v423 = &v425;
                    v208 = (unsigned int)(v209 + 5);
                    goto LABEL_278;
                  }
                  if ( v93 <= (unsigned __int64)&loc_98967F )
                  {
                    v208 = (unsigned int)(v209 + 6);
                    break;
                  }
                  if ( v93 <= 0x5F5E0FF )
                  {
                    v208 = (unsigned int)(v209 + 7);
                    break;
                  }
                }
              }
              v423 = &v425;
LABEL_278:
              sub_2240A50(&v423, v208, 0, v93, v85);
              v210 = v423;
              v211 = v424 - 1;
              do
              {
                v212 = v90;
                v213 = 5
                     * (v90 / 0x64
                      + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v90 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                v214 = v90;
                v90 /= 0x64u;
                v215 = v212 - 4 * v213;
                v210->m128i_i8[v211] = a00010203040506_0[2 * v215 + 1];
                v216 = (unsigned int)(v211 - 1);
                v211 -= 2;
                v210->m128i_i8[v216] = a00010203040506_0[2 * v215];
              }
              while ( v214 > 0x270F );
              if ( v214 <= 0x3E7 )
                goto LABEL_281;
            }
            v210->m128i_i8[1] = a00010203040506_0[2 * v90 + 1];
            v210->m128i_i8[0] = a00010203040506_0[2 * v90];
            goto LABEL_282;
          }
          v423 = &v425;
          sub_2240A50(&v423, 1, 0, v93, v85);
          v210 = v423;
LABEL_281:
          v210->m128i_i8[0] = v90 + 48;
LABEL_282:
          v378 = 0;
          goto LABEL_183;
        }
        if ( v90 <= 9 )
        {
          v445 = v447;
          sub_2240A50(&v445, 1, 0, v93, v85);
          v122 = v445;
          goto LABEL_142;
        }
        if ( v90 <= 0x63 )
        {
          v445 = v447;
          sub_2240A50(&v445, 2, 0, v93, v85);
          v122 = v445;
          v123 = "0001020304050607080910111213141516171819202122232425262728293031323334353637383940414243444546474849505"
                 "1525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
        }
        else
        {
          if ( v90 <= 0x3E7 )
          {
            v120 = 3;
          }
          else if ( v90 <= 0x270F )
          {
            v120 = 4;
          }
          else
          {
            v119 = v90;
            LODWORD(v120) = 1;
            while ( 1 )
            {
              v93 = v119;
              v121 = v120;
              v120 = (unsigned int)(v120 + 4);
              v119 /= 0x2710u;
              if ( v93 <= 0x1869F )
                break;
              if ( v93 <= 0xF423F )
              {
                v445 = v447;
                v120 = (unsigned int)(v121 + 5);
                goto LABEL_139;
              }
              if ( v93 <= (unsigned __int64)&loc_98967F )
              {
                v120 = (unsigned int)(v121 + 6);
                break;
              }
              if ( v93 <= 0x5F5E0FF )
              {
                v120 = (unsigned int)(v121 + 7);
                break;
              }
            }
          }
          v445 = v447;
LABEL_139:
          sub_2240A50(&v445, v120, 0, v93, v85);
          v122 = v445;
          v123 = "0001020304050607080910111213141516171819202122232425262728293031323334353637383940414243444546474849505"
                 "1525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
          v124 = v446 - 1;
          do
          {
            v125 = v90;
            v126 = 5
                 * (v90 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v90 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
            v127 = v90;
            v90 /= 0x64u;
            v128 = v125 - 4 * v126;
            v122[v124] = a00010203040506_0[2 * v128 + 1];
            v129 = (unsigned __int8)a00010203040506_0[2 * v128];
            v130 = (unsigned int)(v124 - 1);
            v124 -= 2;
            v122[v130] = v129;
          }
          while ( v127 > 0x270F );
          if ( v127 <= 0x3E7 )
          {
LABEL_142:
            *v122 = v90 + 48;
            if ( v117 > 9 )
              goto LABEL_143;
LABEL_293:
            v439 = v441;
            sub_2240A50(&v439, 1, 0, v123, v129);
            v135 = v439;
            LOBYTE(v134) = v117;
LABEL_155:
            *v135 = v134 + 48;
            goto LABEL_156;
          }
        }
        v122[1] = a00010203040506_0[2 * v90 + 1];
        *v122 = a00010203040506_0[2 * v90];
        if ( v117 <= 9 )
          goto LABEL_293;
LABEL_143:
        if ( v117 <= 0x63 )
        {
          v439 = v441;
          sub_2240A50(&v439, 2, 0, v123, v129);
          v135 = v439;
          v134 = v117;
        }
        else
        {
          if ( v117 <= 0x3E7 )
          {
            v132 = 3;
          }
          else if ( v117 <= 0x270F )
          {
            v132 = 4;
          }
          else
          {
            v131 = v117;
            LODWORD(v132) = 1;
            while ( 1 )
            {
              v123 = (const char *)v131;
              v133 = v132;
              v132 = (unsigned int)(v132 + 4);
              v131 /= 0x2710u;
              if ( (unsigned __int64)v123 <= 0x1869F )
                break;
              if ( (unsigned __int64)v123 <= 0xF423F )
              {
                v439 = v441;
                v132 = (unsigned int)(v133 + 5);
                goto LABEL_152;
              }
              if ( v123 <= (const char *)&loc_98967F )
              {
                v132 = (unsigned int)(v133 + 6);
                break;
              }
              if ( (unsigned __int64)v123 <= 0x5F5E0FF )
              {
                v132 = (unsigned int)(v133 + 7);
                break;
              }
            }
          }
          v439 = v441;
LABEL_152:
          sub_2240A50(&v439, v132, 0, v123, v129);
          v134 = v117;
          v135 = v439;
          v136 = v440 - 1;
          do
          {
            v137 = v134
                 - 20
                 * (v134 / 0x64
                  + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v134 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
            v138 = v134;
            v134 /= 0x64u;
            v139 = a00010203040506_0[2 * v137 + 1];
            LOBYTE(v137) = a00010203040506_0[2 * v137];
            v135[v136] = v139;
            v140 = (unsigned int)(v136 - 1);
            v136 -= 2;
            v135[v140] = v137;
          }
          while ( v138 > 0x270F );
          if ( v138 <= 0x3E7 )
            goto LABEL_155;
        }
        v135[1] = a00010203040506_0[2 * v134 + 1];
        *v135 = a00010203040506_0[2 * v134];
LABEL_156:
        v437[1] = 1;
        v437[0] = v438;
        LOWORD(v438[0]) = 77;
        if ( (unsigned __int64)(v440 + 1) <= 0xF || v439 == v441 || (unsigned __int64)(v440 + 1) > v441[0] )
        {
          v141 = (__m128i *)sub_2241490(v437, v439, v440, v438);
          v442 = &v444;
          v142 = (__m128i *)v141->m128i_i64[0];
          v143 = v141 + 1;
          if ( (__m128i *)v141->m128i_i64[0] != &v141[1] )
            goto LABEL_160;
LABEL_313:
          v444 = _mm_loadu_si128(v141 + 1);
        }
        else
        {
          v141 = (__m128i *)sub_2241130(&v439, 0, 0, v438, 1);
          v442 = &v444;
          v142 = (__m128i *)v141->m128i_i64[0];
          v143 = v141 + 1;
          if ( (__m128i *)v141->m128i_i64[0] == &v141[1] )
            goto LABEL_313;
LABEL_160:
          v442 = v142;
          v444.m128i_i64[0] = v141[1].m128i_i64[0];
        }
        v144 = v141->m128i_i64[1];
        v443 = v144;
        v141->m128i_i64[0] = (__int64)v143;
        v141->m128i_i64[1] = 0;
        v141[1].m128i_i8[0] = 0;
        if ( v443 == 0x3FFFFFFFFFFFFFFFLL )
          goto LABEL_661;
        v145 = (__m128i *)sub_2241490(&v442, "_", 1, v144);
        v478.m128i_i64[0] = (__int64)&v479;
        if ( (__m128i *)v145->m128i_i64[0] == &v145[1] )
        {
          v479 = _mm_loadu_si128(v145 + 1);
        }
        else
        {
          v478.m128i_i64[0] = v145->m128i_i64[0];
          v479.m128i_i64[0] = v145[1].m128i_i64[0];
        }
        v478.m128i_i64[1] = v145->m128i_i64[1];
        v145->m128i_i64[0] = (__int64)v145[1].m128i_i64;
        v146 = v445;
        v145->m128i_i64[1] = 0;
        v147 = v446;
        v145[1].m128i_i8[0] = 0;
        v148 = 15;
        v149 = 15;
        if ( (__m128i *)v478.m128i_i64[0] != &v479 )
          v149 = v479.m128i_i64[0];
        v150 = v478.m128i_i64[1] + v147;
        if ( v478.m128i_i64[1] + v147 <= v149 )
          goto LABEL_170;
        if ( v146 != v447 )
          v148 = v447[0];
        if ( v150 <= v148 )
        {
          v151 = (__m128i *)sub_2241130(&v445, 0, 0, v478.m128i_i64[0], v478.m128i_i64[1]);
          v423 = &v425;
          v152 = (__m128i *)v151->m128i_i64[0];
          v153 = v151 + 1;
          if ( (__m128i *)v151->m128i_i64[0] == &v151[1] )
            goto LABEL_319;
LABEL_171:
          v423 = v152;
          v425.m128i_i64[0] = v151[1].m128i_i64[0];
        }
        else
        {
LABEL_170:
          v151 = (__m128i *)sub_2241490(&v478, v146, v147, v150);
          v423 = &v425;
          v152 = (__m128i *)v151->m128i_i64[0];
          v153 = v151 + 1;
          if ( (__m128i *)v151->m128i_i64[0] != &v151[1] )
            goto LABEL_171;
LABEL_319:
          v425 = _mm_loadu_si128(v151 + 1);
        }
        v424 = v151->m128i_i64[1];
        v151->m128i_i64[0] = (__int64)v153;
        v151->m128i_i64[1] = 0;
        v151[1].m128i_i8[0] = 0;
        if ( (__m128i *)v478.m128i_i64[0] != &v479 )
          j_j___libc_free_0(v478.m128i_i64[0], v479.m128i_i64[0] + 1);
        if ( v442 != &v444 )
          j_j___libc_free_0(v442, v444.m128i_i64[0] + 1);
        if ( (_QWORD *)v437[0] != v438 )
          j_j___libc_free_0(v437[0], v438[0] + 1LL);
        if ( v439 != v441 )
          j_j___libc_free_0(v439, v441[0] + 1LL);
        if ( v445 != v447 )
          j_j___libc_free_0(v445, v447[0] + 1LL);
        v378 = 1;
LABEL_183:
        v154 = sub_CB6200(v371, v423, v424);
        v157 = *(_DWORD **)(v154 + 32);
        v158 = v154;
        if ( *(_QWORD *)(v154 + 24) - (_QWORD)v157 <= 3u )
        {
          v158 = sub_CB6200(v154, " -> ", 4);
        }
        else
        {
          *v157 = 540945696;
          *(_QWORD *)(v154 + 32) += 4LL;
        }
        if ( v117 == -1 )
        {
          if ( v89 > 9 )
          {
            if ( v89 <= 0x63 )
            {
              v469.m128i_i64[0] = (__int64)&v470;
              sub_2240A50(&v469, 2, 0, v155, v156);
              v201 = (_BYTE *)v469.m128i_i64[0];
            }
            else
            {
              if ( v89 <= 0x3E7 )
              {
                v199 = 3;
              }
              else if ( v89 <= 0x270F )
              {
                v199 = 4;
              }
              else
              {
                v198 = v89;
                LODWORD(v199) = 1;
                while ( 1 )
                {
                  v155 = v198;
                  v200 = v199;
                  v199 = (unsigned int)(v199 + 4);
                  v198 /= 0x2710u;
                  if ( v155 <= 0x1869F )
                    break;
                  if ( v155 <= 0xF423F )
                  {
                    v469.m128i_i64[0] = (__int64)&v470;
                    v199 = (unsigned int)(v200 + 5);
                    goto LABEL_254;
                  }
                  if ( v155 <= (unsigned __int64)&loc_98967F )
                  {
                    v199 = (unsigned int)(v200 + 6);
                    break;
                  }
                  if ( v155 <= 0x5F5E0FF )
                  {
                    v199 = (unsigned int)(v200 + 7);
                    break;
                  }
                }
              }
              v469.m128i_i64[0] = (__int64)&v470;
LABEL_254:
              sub_2240A50(&v469, v199, 0, v155, v156);
              v201 = (_BYTE *)v469.m128i_i64[0];
              v202 = v469.m128i_i32[2] - 1;
              do
              {
                v203 = v89
                     - 20
                     * (v89 / 0x64
                      + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v89 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                v204 = v89;
                v89 /= 0x64u;
                v205 = a00010203040506_0[2 * v203 + 1];
                LOBYTE(v203) = a00010203040506_0[2 * v203];
                v201[v202] = v205;
                v206 = v202 - 1;
                v202 -= 2;
                v201[v206] = v203;
              }
              while ( v204 > 0x270F );
              if ( v204 <= 0x3E7 )
              {
LABEL_257:
                *v201 = v89 + 48;
                if ( !v378 )
                  goto LABEL_231;
                goto LABEL_258;
              }
            }
            v201[1] = a00010203040506_0[2 * v89 + 1];
            *v201 = a00010203040506_0[2 * v89];
            goto LABEL_230;
          }
          v469.m128i_i64[0] = (__int64)&v470;
          sub_2240A50(&v469, 1, 0, v155, v156);
          v201 = (_BYTE *)v469.m128i_i64[0];
          goto LABEL_257;
        }
        if ( v89 <= 9 )
        {
          v434 = v436;
          sub_2240A50(&v434, 1, 0, v155, v156);
          v162 = v434;
          goto LABEL_199;
        }
        if ( v89 <= 0x63 )
        {
          v434 = v436;
          sub_2240A50(&v434, 2, 0, v155, v156);
          v162 = v434;
          v163 = "0001020304050607080910111213141516171819202122232425262728293031323334353637383940414243444546474849505"
                 "1525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
        }
        else
        {
          if ( v89 <= 0x3E7 )
          {
            v160 = 3;
          }
          else if ( v89 <= 0x270F )
          {
            v160 = 4;
          }
          else
          {
            v159 = v89;
            LODWORD(v160) = 1;
            while ( 1 )
            {
              v155 = v159;
              v161 = v160;
              v160 = (unsigned int)(v160 + 4);
              v159 /= 0x2710u;
              if ( v155 <= 0x1869F )
                break;
              if ( v155 <= 0xF423F )
              {
                v434 = v436;
                v160 = (unsigned int)(v161 + 5);
                goto LABEL_196;
              }
              if ( v155 <= (unsigned __int64)&loc_98967F )
              {
                v160 = (unsigned int)(v161 + 6);
                break;
              }
              if ( v155 <= 0x5F5E0FF )
              {
                v160 = (unsigned int)(v161 + 7);
                break;
              }
            }
          }
          v434 = v436;
LABEL_196:
          sub_2240A50(&v434, v160, 0, v155, v156);
          v162 = v434;
          v163 = "0001020304050607080910111213141516171819202122232425262728293031323334353637383940414243444546474849505"
                 "1525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
          v164 = v435 - 1;
          do
          {
            v165 = v89;
            v166 = 5
                 * (v89 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v89 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
            v167 = v89;
            v89 /= 0x64u;
            v168 = v165 - 4 * v166;
            v162[v164] = a00010203040506_0[2 * v168 + 1];
            v169 = (unsigned int)(v164 - 1);
            v164 -= 2;
            v162[v169] = a00010203040506_0[2 * v168];
          }
          while ( v167 > 0x270F );
          if ( v167 <= 0x3E7 )
          {
LABEL_199:
            *v162 = v89 + 48;
            if ( v117 > 9 )
              goto LABEL_200;
LABEL_290:
            v428 = v430;
            sub_2240A50(&v428, 1, 0, v163, v430);
            v173 = v428;
LABEL_212:
            *v173 = v117 + 48;
            goto LABEL_213;
          }
        }
        v162[1] = a00010203040506_0[2 * v89 + 1];
        *v162 = a00010203040506_0[2 * v89];
        if ( v117 <= 9 )
          goto LABEL_290;
LABEL_200:
        if ( v117 <= 0x63 )
        {
          v428 = v430;
          sub_2240A50(&v428, 2, 0, v163, v430);
          v173 = v428;
        }
        else
        {
          if ( v117 <= 0x3E7 )
          {
            v171 = 3;
          }
          else if ( v117 <= 0x270F )
          {
            v171 = 4;
          }
          else
          {
            v170 = v117;
            LODWORD(v171) = 1;
            while ( 1 )
            {
              v163 = (const char *)v170;
              v172 = v171;
              v171 = (unsigned int)(v171 + 4);
              v170 /= 0x2710u;
              if ( (unsigned __int64)v163 <= 0x1869F )
                break;
              if ( (unsigned __int64)v163 <= 0xF423F )
              {
                v428 = v430;
                v171 = (unsigned int)(v172 + 5);
                goto LABEL_209;
              }
              if ( v163 <= (const char *)&loc_98967F )
              {
                v171 = (unsigned int)(v172 + 6);
                break;
              }
              if ( (unsigned __int64)v163 <= 0x5F5E0FF )
              {
                v171 = (unsigned int)(v172 + 7);
                break;
              }
            }
          }
          v428 = v430;
LABEL_209:
          sub_2240A50(&v428, v171, 0, v163, v430);
          v173 = v428;
          v174 = v429 - 1;
          do
          {
            v175 = v117;
            v176 = 5
                 * (v117 / 0x64
                  + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v117 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
            v177 = v117;
            v117 /= 0x64u;
            v178 = v175 - 4 * v176;
            v173[v174] = a00010203040506_0[2 * v178 + 1];
            v179 = (unsigned int)(v174 - 1);
            v174 -= 2;
            v173[v179] = a00010203040506_0[2 * v178];
          }
          while ( v177 > 0x270F );
          if ( v177 <= 0x3E7 )
            goto LABEL_212;
        }
        v173[1] = a00010203040506_0[2 * v117 + 1];
        *v173 = a00010203040506_0[2 * v117];
LABEL_213:
        v426[1] = 1;
        LOWORD(v427[0]) = 77;
        v426[0] = v427;
        if ( (unsigned __int64)(v429 + 1) <= 0xF || v428 == v430 || (unsigned __int64)(v429 + 1) > v430[0] )
        {
          v180 = (__m128i *)sub_2241490(v426, v428, v429, v427);
          v431 = &v433;
          v181 = (__m128i *)v180->m128i_i64[0];
          v182 = v180 + 1;
          if ( (__m128i *)v180->m128i_i64[0] != &v180[1] )
            goto LABEL_217;
LABEL_317:
          v433 = _mm_loadu_si128(v180 + 1);
        }
        else
        {
          v180 = (__m128i *)sub_2241130(&v428, 0, 0, v427, 1);
          v431 = &v433;
          v181 = (__m128i *)v180->m128i_i64[0];
          v182 = v180 + 1;
          if ( (__m128i *)v180->m128i_i64[0] == &v180[1] )
            goto LABEL_317;
LABEL_217:
          v431 = v181;
          v433.m128i_i64[0] = v180[1].m128i_i64[0];
        }
        v183 = v180->m128i_i64[1];
        v432 = v183;
        v180->m128i_i64[0] = (__int64)v182;
        v180->m128i_i64[1] = 0;
        v180[1].m128i_i8[0] = 0;
        if ( v432 == 0x3FFFFFFFFFFFFFFFLL )
          goto LABEL_661;
        v184 = (__m128i *)sub_2241490(&v431, "_", 1, v183);
        v478.m128i_i64[0] = (__int64)&v479;
        if ( (__m128i *)v184->m128i_i64[0] == &v184[1] )
        {
          v479 = _mm_loadu_si128(v184 + 1);
        }
        else
        {
          v478.m128i_i64[0] = v184->m128i_i64[0];
          v479.m128i_i64[0] = v184[1].m128i_i64[0];
        }
        v478.m128i_i64[1] = v184->m128i_i64[1];
        v184->m128i_i64[0] = (__int64)v184[1].m128i_i64;
        v185 = v434;
        v184->m128i_i64[1] = 0;
        v186 = v435;
        v184[1].m128i_i8[0] = 0;
        v187 = 15;
        v188 = 15;
        if ( (__m128i *)v478.m128i_i64[0] != &v479 )
          v188 = v479.m128i_i64[0];
        v189 = v478.m128i_i64[1] + v186;
        if ( v478.m128i_i64[1] + v186 <= v188 )
          goto LABEL_227;
        if ( v185 != v436 )
          v187 = v436[0];
        if ( v189 <= v187 )
        {
          v190 = (__m128i *)sub_2241130(&v434, 0, 0, v478.m128i_i64[0], v478.m128i_i64[1]);
          v469.m128i_i64[0] = (__int64)&v470;
          v191 = v190->m128i_i64[0];
          v192 = v190 + 1;
          if ( (__m128i *)v190->m128i_i64[0] == &v190[1] )
            goto LABEL_315;
LABEL_228:
          v469.m128i_i64[0] = v191;
          v470.m128i_i64[0] = v190[1].m128i_i64[0];
        }
        else
        {
LABEL_227:
          v190 = (__m128i *)sub_2241490(&v478, v185, v186, v189);
          v469.m128i_i64[0] = (__int64)&v470;
          v191 = v190->m128i_i64[0];
          v192 = v190 + 1;
          if ( (__m128i *)v190->m128i_i64[0] != &v190[1] )
            goto LABEL_228;
LABEL_315:
          v470 = _mm_loadu_si128(v190 + 1);
        }
        v469.m128i_i64[1] = v190->m128i_i64[1];
        v190->m128i_i64[0] = (__int64)v192;
        v190->m128i_i64[1] = 0;
        v190[1].m128i_i8[0] = 0;
LABEL_230:
        if ( v378 )
        {
LABEL_258:
          if ( (__m128i *)v478.m128i_i64[0] != &v479 )
            j_j___libc_free_0(v478.m128i_i64[0], v479.m128i_i64[0] + 1);
          if ( v431 != &v433 )
            j_j___libc_free_0(v431, v433.m128i_i64[0] + 1);
          if ( (_QWORD *)v426[0] != v427 )
            j_j___libc_free_0(v426[0], v427[0] + 1LL);
          if ( v428 != v430 )
            j_j___libc_free_0(v428, v430[0] + 1LL);
          if ( v434 != v436 )
            j_j___libc_free_0(v434, v436[0] + 1LL);
        }
LABEL_231:
        v193 = sub_CB6200(v158, v469.m128i_i64[0], v469.m128i_i64[1]);
        if ( !off_49799E0[src] )
          goto LABEL_242;
        srca = off_49799E0[src];
        v194 = strlen(srca);
        v195 = *(_BYTE **)(v193 + 32);
        v196 = v194;
        v197 = *(_BYTE **)(v193 + 24);
        if ( v196 > v197 - v195 )
        {
          v193 = sub_CB6200(v193, srca, v196);
LABEL_242:
          v195 = *(_BYTE **)(v193 + 32);
          if ( *(_BYTE **)(v193 + 24) != v195 )
            goto LABEL_236;
          goto LABEL_243;
        }
        if ( v196 )
        {
          memcpy(v195, srca, v196);
          v197 = *(_BYTE **)(v193 + 24);
          v195 = (_BYTE *)(v196 + *(_QWORD *)(v193 + 32));
          *(_QWORD *)(v193 + 32) = v195;
        }
        if ( v197 == v195 )
        {
LABEL_243:
          sub_CB6200(v193, "\n", 1);
          goto LABEL_237;
        }
LABEL_236:
        *v195 = 10;
        ++*(_QWORD *)(v193 + 32);
LABEL_237:
        if ( (__m128i *)v469.m128i_i64[0] != &v470 )
          j_j___libc_free_0(v469.m128i_i64[0], v470.m128i_i64[0] + 1);
        if ( v423 != &v425 )
          j_j___libc_free_0(v423, v425.m128i_i64[0] + 1);
LABEL_100:
        v390 += 2;
        if ( v374 != v390 )
          continue;
        break;
      }
      v86 = *(_QWORD *)(v385 + 40);
      if ( !v86 )
        goto LABEL_111;
LABEL_102:
      v95 = *(_DWORD *)(v86 + 8);
      if ( v95 )
      {
        if ( v95 != 1 )
          goto LABEL_111;
        v240 = *(_QWORD *)(v86 + 64);
        v370 = v240 + 16LL * *(unsigned int *)(v86 + 72);
        if ( v370 == v240 )
          goto LABEL_111;
        v392 = *(_QWORD *)(v86 + 64);
        while ( 1 )
        {
          v241 = *(_QWORD *)(*(_QWORD *)v392 & 0xFFFFFFFFFFFFFFF8LL);
          v242 = *(_QWORD *)(v385 + 32);
          v243 = (__int8 *)v375[4].m128i_i64[0];
          if ( v243 )
          {
            v244 = &v375[3].m128i_i8[8];
            do
            {
              while ( 1 )
              {
                v245 = *((_QWORD *)v243 + 2);
                v246 = *((_QWORD *)v243 + 3);
                if ( v241 <= *((_QWORD *)v243 + 4) )
                  break;
                v243 = (__int8 *)*((_QWORD *)v243 + 3);
                if ( !v246 )
                  goto LABEL_372;
              }
              v244 = v243;
              v243 = (__int8 *)*((_QWORD *)v243 + 2);
            }
            while ( v245 );
LABEL_372:
            if ( v388 != v244 && v241 >= *((_QWORD *)v244 + 4) )
              break;
          }
          v478.m128i_i32[2] = *(_BYTE *)(v392 + 8) & 7;
          v479.m128i_i64[0] = v242;
          v478.m128i_i64[0] = v394;
          v479.m128i_i64[1] = v241;
          sub_BABCF0((__int64)&v397, &v478);
LABEL_375:
          v392 += 16;
          if ( v370 == v392 )
            goto LABEL_111;
        }
        v247 = v394;
        v381 = (*(_BYTE *)(v392 + 8) & 7) + 4;
        v372 = sub_904010(a2, "    ");
        if ( v247 != -1 )
        {
          if ( v242 > 9 )
          {
            if ( v242 <= 0x63 )
            {
              v469.m128i_i64[0] = (__int64)&v470;
              sub_2240A50(&v469, 2, 0, v248, v249);
              v253 = (_BYTE *)v469.m128i_i64[0];
              v254 = "000102030405060708091011121314151617181920212223242526272829303132333435363738394041424344454647484"
                     "95051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
            }
            else
            {
              if ( v242 <= 0x3E7 )
              {
                v250 = 3;
              }
              else if ( v242 <= 0x270F )
              {
                v250 = 4;
              }
              else
              {
                LODWORD(v250) = 1;
                v251 = v242;
                while ( 1 )
                {
                  v248 = v251;
                  v252 = v250;
                  v250 = (unsigned int)(v250 + 4);
                  v251 /= 0x2710u;
                  if ( v248 <= 0x1869F )
                    break;
                  if ( v248 <= 0xF423F )
                  {
                    v469.m128i_i64[0] = (__int64)&v470;
                    v250 = (unsigned int)(v252 + 5);
                    goto LABEL_388;
                  }
                  if ( v248 <= (unsigned __int64)&loc_98967F )
                  {
                    v250 = (unsigned int)(v252 + 6);
                    break;
                  }
                  if ( v248 <= 0x5F5E0FF )
                  {
                    v250 = (unsigned int)(v252 + 7);
                    break;
                  }
                }
              }
              v469.m128i_i64[0] = (__int64)&v470;
LABEL_388:
              sub_2240A50(&v469, v250, 0, v248, v249);
              v253 = (_BYTE *)v469.m128i_i64[0];
              v254 = "000102030405060708091011121314151617181920212223242526272829303132333435363738394041424344454647484"
                     "95051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
              v255 = v469.m128i_i32[2] - 1;
              do
              {
                v256 = v242;
                v257 = 5
                     * (v242 / 0x64
                      + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v242 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                v258 = v242;
                v242 /= 0x64u;
                v259 = v256 - 4 * v257;
                v253[v255] = a00010203040506_0[2 * v259 + 1];
                v260 = (unsigned __int8)a00010203040506_0[2 * v259];
                v261 = v255 - 1;
                v255 -= 2;
                v253[v261] = v260;
              }
              while ( v258 > 0x270F );
              if ( v258 <= 0x3E7 )
              {
LABEL_391:
                *v253 = v242 + 48;
                if ( v247 > 9 )
                  goto LABEL_392;
                goto LABEL_479;
              }
            }
            v253[1] = a00010203040506_0[2 * v242 + 1];
            *v253 = a00010203040506_0[2 * v242];
            if ( v247 > 9 )
            {
LABEL_392:
              if ( v247 <= 0x63 )
              {
                v461 = v463;
                sub_2240A50(&v461, 2, 0, v254, v260);
                v265 = v461;
                v266 = v247;
              }
              else
              {
                if ( v247 <= 0x3E7 )
                {
                  v262 = 3;
                }
                else if ( v247 <= 0x270F )
                {
                  v262 = 4;
                }
                else
                {
                  LODWORD(v262) = 1;
                  v263 = v247;
                  while ( 1 )
                  {
                    v254 = (const char *)v263;
                    v264 = v262;
                    v262 = (unsigned int)(v262 + 4);
                    v263 /= 0x2710u;
                    if ( (unsigned __int64)v254 <= 0x1869F )
                      break;
                    if ( (unsigned __int64)v254 <= 0xF423F )
                    {
                      v461 = v463;
                      v262 = (unsigned int)(v264 + 5);
                      goto LABEL_401;
                    }
                    if ( v254 <= (const char *)&loc_98967F )
                    {
                      v262 = (unsigned int)(v264 + 6);
                      break;
                    }
                    if ( (unsigned __int64)v254 <= 0x5F5E0FF )
                    {
                      v262 = (unsigned int)(v264 + 7);
                      break;
                    }
                  }
                }
                v461 = v463;
LABEL_401:
                sub_2240A50(&v461, v262, 0, v254, v260);
                v265 = v461;
                v266 = v247;
                v267 = v462 - 1;
                do
                {
                  v268 = v266;
                  v269 = 5
                       * (v266 / 0x64
                        + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v266 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                  v270 = v266;
                  v266 /= 0x64u;
                  v271 = v268 - 4 * v269;
                  v265[v267] = a00010203040506_0[2 * v271 + 1];
                  v272 = (unsigned int)(v267 - 1);
                  v267 -= 2;
                  v265[v272] = a00010203040506_0[2 * v271];
                }
                while ( v270 > 0x270F );
                if ( v270 <= 0x3E7 )
                  goto LABEL_404;
              }
              v265[1] = a00010203040506_0[2 * v266 + 1];
              *v265 = a00010203040506_0[2 * v266];
LABEL_405:
              v459.m128i_i64[1] = 1;
              LOWORD(v460[0]) = 77;
              v459.m128i_i64[0] = (__int64)v460;
              sub_8FD5D0(&v466, (__int64)&v459, &v461);
              if ( v466.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
                goto LABEL_661;
              v274 = (__m128i *)sub_2241490(&v466, "_", 1, v273);
              v478.m128i_i64[0] = (__int64)&v479;
              if ( (__m128i *)v274->m128i_i64[0] == &v274[1] )
              {
                v479 = _mm_loadu_si128(v274 + 1);
              }
              else
              {
                v478.m128i_i64[0] = v274->m128i_i64[0];
                v479.m128i_i64[0] = v274[1].m128i_i64[0];
              }
              v478.m128i_i64[1] = v274->m128i_i64[1];
              v274->m128i_i64[0] = (__int64)v274[1].m128i_i64;
              v274->m128i_i64[1] = 0;
              v274[1].m128i_i8[0] = 0;
              sub_8FD5D0(&v448, (__int64)&v478, &v469);
              sub_2240A30(&v478);
              sub_2240A30(&v466);
              sub_2240A30(&v459);
              sub_2240A30(&v461);
              if ( (__m128i *)v469.m128i_i64[0] != &v470 )
                j_j___libc_free_0(v469.m128i_i64[0], v470.m128i_i64[0] + 1);
              v275 = sub_CB6200(v372, v448.m128i_i64[0], v448.m128i_i64[1]);
              v373 = sub_904010(v275, " -> ");
              if ( v241 > 9 )
              {
                if ( v241 <= 0x63 )
                {
                  v456 = v458;
                  sub_2240A50(&v456, 2, 0, v276, v277);
                  v281 = v456;
                  v282 = "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464"
                         "748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
                }
                else
                {
                  if ( v241 <= 0x3E7 )
                  {
                    v278 = 3;
                  }
                  else if ( v241 <= 0x270F )
                  {
                    v278 = 4;
                  }
                  else
                  {
                    LODWORD(v278) = 1;
                    v279 = v241;
                    while ( 1 )
                    {
                      v276 = v279;
                      v280 = v278;
                      v278 = (unsigned int)(v278 + 4);
                      v279 /= 0x2710u;
                      if ( v276 <= 0x1869F )
                        break;
                      if ( v276 <= 0xF423F )
                      {
                        v456 = v458;
                        v278 = (unsigned int)(v280 + 5);
                        goto LABEL_420;
                      }
                      if ( v276 <= (unsigned __int64)&loc_98967F )
                      {
                        v278 = (unsigned int)(v280 + 6);
                        break;
                      }
                      if ( v276 <= 0x5F5E0FF )
                      {
                        v278 = (unsigned int)(v280 + 7);
                        break;
                      }
                    }
                  }
                  v456 = v458;
LABEL_420:
                  sub_2240A50(&v456, v278, 0, v276, v277);
                  v281 = v456;
                  v282 = "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464"
                         "748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
                  v283 = v457 - 1;
                  do
                  {
                    v284 = v241;
                    v285 = 5
                         * (v241 / 0x64
                          + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v241 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                    v286 = v241;
                    v241 /= 0x64u;
                    v287 = v284 - 4 * v285;
                    v281[v283] = a00010203040506_0[2 * v287 + 1];
                    v288 = (unsigned __int8)a00010203040506_0[2 * v287];
                    v289 = (unsigned int)(v283 - 1);
                    v283 -= 2;
                    v281[v289] = v288;
                  }
                  while ( v286 > 0x270F );
                  if ( v286 <= 0x3E7 )
                  {
LABEL_423:
                    v288 = (unsigned int)(v241 + 48);
                    *v281 = v241 + 48;
                    if ( v247 > 9 )
                      goto LABEL_424;
                    goto LABEL_482;
                  }
                }
                v281[1] = a00010203040506_0[2 * v241 + 1];
                *v281 = a00010203040506_0[2 * v241];
                if ( v247 > 9 )
                {
LABEL_424:
                  if ( v247 <= 0x63 )
                  {
                    v452 = v454;
                    sub_2240A50(&v452, 2, 0, v282, v288);
                    v293 = v452;
                  }
                  else
                  {
                    if ( v247 <= 0x3E7 )
                    {
                      v290 = 3;
                    }
                    else if ( v247 <= 0x270F )
                    {
                      v290 = 4;
                    }
                    else
                    {
                      LODWORD(v290) = 1;
                      v291 = v247;
                      while ( 1 )
                      {
                        v282 = (const char *)v291;
                        v292 = v290;
                        v290 = (unsigned int)(v290 + 4);
                        v291 /= 0x2710u;
                        if ( (unsigned __int64)v282 <= 0x1869F )
                          break;
                        if ( (unsigned __int64)v282 <= 0xF423F )
                        {
                          v452 = v454;
                          v290 = (unsigned int)(v292 + 5);
                          goto LABEL_433;
                        }
                        if ( v282 <= (const char *)&loc_98967F )
                        {
                          v290 = (unsigned int)(v292 + 6);
                          break;
                        }
                        if ( (unsigned __int64)v282 <= 0x5F5E0FF )
                        {
                          v290 = (unsigned int)(v292 + 7);
                          break;
                        }
                      }
                    }
                    v452 = v454;
LABEL_433:
                    sub_2240A50(&v452, v290, 0, v282, v288);
                    v293 = v452;
                    v294 = v453 - 1;
                    do
                    {
                      v295 = v247
                           - 20
                           * (v247 / 0x64
                            + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v247 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                      v296 = v247;
                      v247 /= 0x64u;
                      v297 = a00010203040506_0[2 * v295 + 1];
                      LOBYTE(v295) = a00010203040506_0[2 * v295];
                      v293[v294] = v297;
                      v298 = (unsigned int)(v294 - 1);
                      v294 -= 2;
                      v293[v298] = v295;
                    }
                    while ( v296 > 0x270F );
                    if ( v296 <= 0x3E7 )
                      goto LABEL_436;
                  }
                  v293[1] = a00010203040506_0[2 * v247 + 1];
                  *v293 = a00010203040506_0[2 * v247];
                  goto LABEL_437;
                }
LABEL_482:
                v452 = v454;
                sub_2240A50(&v452, 1, 0, v282, v288);
                v293 = v452;
LABEL_436:
                *v293 = v247 + 48;
LABEL_437:
                v450[1] = 1;
                v450[0] = &v451;
                v451 = 77;
                sub_8FD5D0(v455, (__int64)v450, &v452);
                if ( v455[0].m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
                  goto LABEL_661;
                v300 = (__m128i *)sub_2241490(v455, "_", 1, v299);
                v478.m128i_i64[0] = (__int64)&v479;
                if ( (__m128i *)v300->m128i_i64[0] == &v300[1] )
                {
                  v479 = _mm_loadu_si128(v300 + 1);
                }
                else
                {
                  v478.m128i_i64[0] = v300->m128i_i64[0];
                  v479.m128i_i64[0] = v300[1].m128i_i64[0];
                }
                v478.m128i_i64[1] = v300->m128i_i64[1];
                v300->m128i_i64[0] = (__int64)v300[1].m128i_i64;
                v300->m128i_i64[1] = 0;
                v300[1].m128i_i8[0] = 0;
                sub_8FD5D0(&v469, (__int64)&v478, &v456);
                if ( (__m128i *)v478.m128i_i64[0] != &v479 )
                  j_j___libc_free_0(v478.m128i_i64[0], v479.m128i_i64[0] + 1);
                sub_2240A30(v455);
                sub_2240A30(v450);
                sub_2240A30(&v452);
                sub_2240A30(&v456);
                goto LABEL_443;
              }
              v456 = v458;
              sub_2240A50(&v456, 1, 0, v276, v277);
              v281 = v456;
              goto LABEL_423;
            }
LABEL_479:
            v461 = v463;
            sub_2240A50(&v461, 1, 0, v254, v260);
            v265 = v461;
            LOBYTE(v266) = v247;
LABEL_404:
            *v265 = v266 + 48;
            goto LABEL_405;
          }
          v469.m128i_i64[0] = (__int64)&v470;
          sub_2240A50(&v469, 1, 0, v248, v249);
          v253 = (_BYTE *)v469.m128i_i64[0];
          goto LABEL_391;
        }
        if ( v242 > 9 )
        {
          if ( v242 <= 0x63 )
          {
            v448.m128i_i64[0] = (__int64)v449;
            sub_2240A50(&v448, 2, 0, v248, v249);
            v306 = (_BYTE *)v448.m128i_i64[0];
          }
          else
          {
            if ( v242 <= 0x3E7 )
            {
              v303 = 3;
            }
            else if ( v242 <= 0x270F )
            {
              v303 = 4;
            }
            else
            {
              LODWORD(v303) = 1;
              v304 = v242;
              while ( 1 )
              {
                v248 = v304;
                v305 = v303;
                v303 = (unsigned int)(v303 + 4);
                v304 /= 0x2710u;
                if ( v248 <= 0x1869F )
                  break;
                if ( v248 <= 0xF423F )
                {
                  v448.m128i_i64[0] = (__int64)v449;
                  v303 = (unsigned int)(v305 + 5);
                  goto LABEL_457;
                }
                if ( v248 <= (unsigned __int64)&loc_98967F )
                {
                  v303 = (unsigned int)(v305 + 6);
                  break;
                }
                if ( v248 <= 0x5F5E0FF )
                {
                  v303 = (unsigned int)(v305 + 7);
                  break;
                }
              }
            }
            v448.m128i_i64[0] = (__int64)v449;
LABEL_457:
            sub_2240A50(&v448, v303, 0, v248, v249);
            v306 = (_BYTE *)v448.m128i_i64[0];
            v307 = v448.m128i_i32[2] - 1;
            do
            {
              v308 = v242;
              v309 = 5
                   * (v242 / 0x64
                    + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v242 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
              v310 = v242;
              v242 /= 0x64u;
              v311 = v308 - 4 * v309;
              v306[v307] = a00010203040506_0[2 * v311 + 1];
              v312 = v307 - 1;
              v307 -= 2;
              v306[v312] = a00010203040506_0[2 * v311];
            }
            while ( v310 > 0x270F );
            if ( v310 <= 0x3E7 )
              goto LABEL_460;
          }
          v306[1] = a00010203040506_0[2 * v242 + 1];
          *v306 = a00010203040506_0[2 * v242];
LABEL_461:
          v313 = sub_CB6200(v372, v448.m128i_i64[0], v448.m128i_i64[1]);
          v373 = sub_904010(v313, " -> ");
          if ( v241 > 9 )
          {
            if ( v241 <= 0x63 )
            {
              v469.m128i_i64[0] = (__int64)&v470;
              sub_2240A50(&v469, 2, 0, v314, v315);
              v319 = (_BYTE *)v469.m128i_i64[0];
            }
            else
            {
              if ( v241 <= 0x3E7 )
              {
                v316 = 3;
              }
              else if ( v241 <= 0x270F )
              {
                v316 = 4;
              }
              else
              {
                LODWORD(v316) = 1;
                v317 = v241;
                while ( 1 )
                {
                  v314 = v317;
                  v318 = v316;
                  v316 = (unsigned int)(v316 + 4);
                  v317 /= 0x2710u;
                  if ( v314 <= 0x1869F )
                    break;
                  if ( v314 <= 0xF423F )
                  {
                    v469.m128i_i64[0] = (__int64)&v470;
                    v316 = (unsigned int)(v318 + 5);
                    goto LABEL_471;
                  }
                  if ( v314 <= (unsigned __int64)&loc_98967F )
                  {
                    v316 = (unsigned int)(v318 + 6);
                    break;
                  }
                  if ( v314 <= 0x5F5E0FF )
                  {
                    v316 = (unsigned int)(v318 + 7);
                    break;
                  }
                }
              }
              v469.m128i_i64[0] = (__int64)&v470;
LABEL_471:
              sub_2240A50(&v469, v316, 0, v314, v315);
              v319 = (_BYTE *)v469.m128i_i64[0];
              v320 = v469.m128i_i32[2] - 1;
              do
              {
                v321 = v241
                     - 20
                     * (v241 / 0x64
                      + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v241 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
                v322 = v241;
                v241 /= 0x64u;
                v323 = a00010203040506_0[2 * v321 + 1];
                LOBYTE(v321) = a00010203040506_0[2 * v321];
                v319[v320] = v323;
                v324 = v320 - 1;
                v320 -= 2;
                v319[v324] = v321;
              }
              while ( v322 > 0x270F );
              if ( v322 <= 0x3E7 )
                goto LABEL_474;
            }
            v319[1] = a00010203040506_0[2 * v241 + 1];
            *v319 = a00010203040506_0[2 * v241];
            goto LABEL_443;
          }
          v469.m128i_i64[0] = (__int64)&v470;
          sub_2240A50(&v469, 1, 0, v314, v315);
          v319 = (_BYTE *)v469.m128i_i64[0];
LABEL_474:
          *v319 = v241 + 48;
LABEL_443:
          v301 = sub_CB6200(v373, v469.m128i_i64[0], v469.m128i_i64[1]);
          v302 = sub_904010(v301, off_49799E0[v381]);
          sub_904010(v302, "\n");
          if ( (__m128i *)v469.m128i_i64[0] != &v470 )
            j_j___libc_free_0(v469.m128i_i64[0], v470.m128i_i64[0] + 1);
          if ( (_QWORD *)v448.m128i_i64[0] != v449 )
            j_j___libc_free_0(v448.m128i_i64[0], v449[0] + 1LL);
          goto LABEL_375;
        }
        v448.m128i_i64[0] = (__int64)v449;
        sub_2240A50(&v448, 1, 0, v248, v249);
        v306 = (_BYTE *)v448.m128i_i64[0];
LABEL_460:
        *v306 = v242 + 48;
        goto LABEL_461;
      }
      v96 = *(_QWORD *)(*(_QWORD *)(v86 + 56) & 0xFFFFFFFFFFFFFFF8LL);
      v97 = *(_QWORD *)(v385 + 32);
      v98 = (__int8 *)v375[4].m128i_i64[0];
      if ( !v98 )
        goto LABEL_110;
      v99 = &v375[3].m128i_i8[8];
      do
      {
        while ( 1 )
        {
          v100 = *((_QWORD *)v98 + 2);
          v101 = *((_QWORD *)v98 + 3);
          if ( v96 <= *((_QWORD *)v98 + 4) )
            break;
          v98 = (__int8 *)*((_QWORD *)v98 + 3);
          if ( !v101 )
            goto LABEL_108;
        }
        v99 = v98;
        v98 = (__int8 *)*((_QWORD *)v98 + 2);
      }
      while ( v100 );
LABEL_108:
      if ( v388 != v99 && v96 >= *((_QWORD *)v99 + 4) )
      {
        v217 = v394;
        v218 = sub_904010(a2, "    ");
        sub_BAC860(&v478, v217, v97, v219, v220);
        v221 = sub_CB6200(v218, v478.m128i_i64[0], v478.m128i_i64[1]);
        v222 = sub_904010(v221, " -> ");
        sub_BAC860(&v469, v217, v96, v223, v224);
        v225 = sub_CB6200(v222, v469.m128i_i64[0], v469.m128i_i64[1]);
        v226 = sub_904010(v225, " [style=dotted]; // alias");
        sub_904010(v226, "\n");
        if ( (__m128i *)v469.m128i_i64[0] != &v470 )
          j_j___libc_free_0(v469.m128i_i64[0], v470.m128i_i64[0] + 1);
        if ( (__m128i *)v478.m128i_i64[0] != &v479 )
          j_j___libc_free_0(v478.m128i_i64[0], v479.m128i_i64[0] + 1);
      }
      else
      {
LABEL_110:
        v479.m128i_i64[0] = *(_QWORD *)(v385 + 32);
        v478.m128i_i32[2] = -4;
        v478.m128i_i64[0] = v394;
        v479.m128i_i64[1] = v96;
        sub_BABCF0((__int64)&v397, &v478);
      }
LABEL_111:
      v385 = sub_220EEE0(v385);
      if ( v388 != (char *)v385 )
        continue;
      break;
    }
LABEL_112:
    sub_904010(a2, "  }\n");
    v375 = (const __m128i *)sub_220EEE0(v375);
    if ( v375 != (const __m128i *)&v473 )
      continue;
    break;
  }
LABEL_113:
  sub_904010(a2, "  // Cross-module edges:\n");
  v376 = v398;
  if ( v398 != v397 )
  {
    for ( j = v397 + 3; ; j += 4 )
    {
      if ( (unsigned __int8)sub_BAF210((__int64)&v403, j, &v469) )
      {
        v104 = *(unsigned __int64 **)(v469.m128i_i64[0] + 16);
        v105 = *(_QWORD *)(v469.m128i_i64[0] + 8);
        v106 = (unsigned __int64 *)(v469.m128i_i64[0] + 8);
        if ( (unsigned __int64 *)v105 != v104 )
          goto LABEL_117;
        goto LABEL_562;
      }
      v327 = v406;
      v328 = (_QWORD *)v469.m128i_i64[0];
      ++v403;
      v329 = v405 + 1;
      v478.m128i_i64[0] = v469.m128i_i64[0];
      if ( 4 * ((int)v405 + 1) >= 3 * v406 )
      {
        v327 = 2 * v406;
      }
      else if ( v406 - HIDWORD(v405) - v329 > v406 >> 3 )
      {
        goto LABEL_559;
      }
      sub_BB04D0((__int64)&v403, v327);
      sub_BAF210((__int64)&v403, j, &v478);
      v328 = (_QWORD *)v478.m128i_i64[0];
      v329 = v405 + 1;
LABEL_559:
      LODWORD(v405) = v329;
      if ( *v328 != -1 )
        --HIDWORD(v405);
      v330 = *j;
      v106 = v328 + 1;
      v104 = 0;
      *v106 = 0;
      v106[1] = 0;
      *(v106 - 1) = v330;
      v106[2] = 0;
      v105 = *v106;
      if ( *v106 )
        goto LABEL_117;
LABEL_562:
      v331 = *j;
      v332 = *(_QWORD **)(a1 + 16);
      if ( v332 )
      {
        v333 = (_QWORD *)(a1 + 8);
        do
        {
          while ( 1 )
          {
            v334 = v332[2];
            v105 = v332[3];
            if ( v331 <= v332[4] )
              break;
            v332 = (_QWORD *)v332[3];
            if ( !v105 )
              goto LABEL_567;
          }
          v333 = v332;
          v332 = (_QWORD *)v332[2];
        }
        while ( v334 );
LABEL_567:
        v335 = 0;
        if ( (_QWORD *)(a1 + 8) != v333 && v331 >= v333[4] )
          v335 = (unsigned __int64)(v333 + 4) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v335 = 0;
      }
      v336 = *j;
      v478.m128i_i64[0] = *(unsigned __int8 *)(a1 + 343) | v335;
      sub_BAD140(a2, &v478, v336, v105, v103);
      v478.m128i_i64[0] = -1;
      v337 = (_BYTE *)v106[1];
      if ( v337 == (_BYTE *)v106[2] )
      {
        sub_A235E0((__int64)v106, v337, &v478);
        v104 = (unsigned __int64 *)v106[1];
      }
      else
      {
        if ( v337 )
        {
          *(_QWORD *)v337 = -1;
          v337 = (_BYTE *)v106[1];
        }
        v104 = (unsigned __int64 *)(v337 + 8);
        v106[1] = (unsigned __int64)(v337 + 8);
      }
      v105 = *v106;
      if ( (unsigned __int64 *)*v106 != v104 )
      {
LABEL_117:
        v107 = (unsigned __int64 *)v105;
        do
        {
          v389 = *v107;
          v391 = *(j - 3);
          if ( v391 != *v107 )
          {
            v386 = *j;
            v387 = *(j - 1);
            v108 = *((_DWORD *)j - 4) + 4;
            v109 = sub_904010(a2, "  ");
            sub_BAC860(&v478, v391, v387, v110, v111);
            v112 = sub_CB6200(v109, v478.m128i_i64[0], v478.m128i_i64[1]);
            v113 = sub_904010(v112, " -> ");
            sub_BAC860(&v469, v389, v386, v114, v389);
            v115 = sub_CB6200(v113, v469.m128i_i64[0], v469.m128i_i64[1]);
            v116 = sub_904010(v115, off_49799E0[v108]);
            sub_904010(v116, "\n");
            if ( (__m128i *)v469.m128i_i64[0] != &v470 )
              j_j___libc_free_0(v469.m128i_i64[0], v470.m128i_i64[0] + 1);
            if ( (__m128i *)v478.m128i_i64[0] != &v479 )
              j_j___libc_free_0(v478.m128i_i64[0], v479.m128i_i64[0] + 1);
          }
          ++v107;
        }
        while ( v104 != v107 );
      }
      if ( v376 == j + 1 )
        break;
    }
  }
  sub_904010(a2, "}");
  sub_C7D6A0(v408, 24LL * v410, 8);
  if ( v400 )
    j_j___libc_free_0(v400, (char *)v402 - (_BYTE *)v400);
  sub_BAC0D0(v474);
  v340 = v406;
  if ( v406 )
  {
    v341 = v404;
    v342 = &v404[4 * v406];
    do
    {
      while ( 1 )
      {
        if ( *v341 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v343 = v341[1];
          if ( v343 )
            break;
        }
        v341 += 4;
        if ( v342 == v341 )
          goto LABEL_610;
      }
      v344 = v341[3];
      v341 += 4;
      j_j___libc_free_0(v343, v344 - v343);
    }
    while ( v342 != v341 );
LABEL_610:
    v340 = v406;
  }
  result = sub_C7D6A0(v404, 32 * v340, 8);
  if ( v397 )
    return j_j___libc_free_0(v397, v399 - (_QWORD)v397);
  return result;
}
