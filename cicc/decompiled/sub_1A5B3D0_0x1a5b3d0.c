// Function: sub_1A5B3D0
// Address: 0x1a5b3d0
//
__int64 __fastcall sub_1A5B3D0(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
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
        void (__fastcall *a16)(__int64, _QWORD, char *, _QWORD),
        __int64 a17,
        __int64 a18)
{
  __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 *v22; // rax
  __int64 *v23; // r13
  char **v24; // rbx
  __int64 *v26; // rbx
  char **v27; // r13
  _QWORD *v28; // r14
  __int64 v29; // rax
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  char *v32; // rax
  __m128i *v33; // r14
  char **v34; // rbx
  char **v35; // r13
  unsigned __int64 *v36; // rdi
  __int64 (__fastcall *v37)(__int64); // rax
  __int64 *v38; // rax
  char *v39; // rdi
  __int64 v40; // rbx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned int v44; // r8d
  __int64 *v45; // rax
  __int64 v46; // rcx
  void **v47; // rdi
  __int64 *v48; // r12
  __int64 v49; // rcx
  _QWORD *v50; // r8
  int v51; // edx
  __int64 v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // r9
  __int64 *v55; // rbx
  __int64 *v56; // r12
  __int64 v57; // r13
  char **p_p_s; // rbx
  char **v59; // r13
  unsigned __int64 *v60; // rdi
  __int64 (__fastcall *v61)(__int64); // rax
  unsigned __int8 i; // al
  char *v63; // rdi
  __int64 **v64; // rax
  _QWORD *v65; // rax
  __int64 v66; // rbx
  unsigned int v67; // eax
  int v68; // eax
  unsigned __int64 v69; // rdx
  __int64 *v70; // rcx
  __int64 v71; // rax
  __int64 v72; // rbx
  char **v73; // r12
  __int64 v74; // r13
  char v75; // al
  int v76; // r8d
  int v77; // r9d
  unsigned int v78; // eax
  unsigned int v79; // esi
  __int64 v80; // rax
  __int64 v81; // rsi
  unsigned __int64 v82; // rdx
  __int64 *v83; // rbx
  __int64 *v84; // r13
  __int64 v85; // rsi
  __int64 v86; // rbx
  __int64 v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rsi
  __int64 v90; // rbx
  char v91; // al
  const char *v92; // rdx
  char *v93; // rcx
  __int64 v94; // r13
  __int64 v95; // rcx
  __int64 **v96; // rdx
  __int64 *v97; // rsi
  unsigned __int64 v98; // rcx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // rcx
  __int64 v102; // rdx
  __int64 v103; // rsi
  __int64 v104; // rcx
  __int64 v105; // rdx
  __int64 v106; // rdx
  int v107; // ecx
  __int64 v108; // rcx
  _QWORD *v109; // rdx
  __int64 v110; // rsi
  unsigned __int64 v111; // rcx
  __int64 v112; // rcx
  __int64 v113; // rcx
  __int64 v114; // r14
  __int64 *v115; // rbx
  __int64 v116; // r12
  __int64 v117; // r9
  __int64 v118; // rax
  double v119; // xmm4_8
  double v120; // xmm5_8
  __int64 v121; // r15
  __int64 v122; // r14
  __int64 v123; // rcx
  __int64 v124; // r9
  __int64 v125; // rdx
  __int64 v126; // rdx
  int v127; // ecx
  unsigned int v128; // edx
  unsigned int v129; // esi
  unsigned int v130; // r8d
  unsigned int v131; // edi
  int v132; // r11d
  __int64 *v133; // r10
  unsigned int v134; // esi
  __int64 v135; // r12
  __int64 v136; // r13
  _QWORD *v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rbx
  unsigned int v140; // eax
  __int64 v141; // r13
  __int64 v142; // rcx
  __int64 v143; // rax
  __int64 v144; // rax
  int v145; // eax
  __int64 v146; // r13
  __int64 k; // rbx
  __int64 v148; // rax
  __int64 *v149; // r13
  __int64 v150; // r15
  unsigned __int64 v151; // rdi
  unsigned int v152; // r12d
  __int64 v153; // r14
  void *v154; // r15
  __int64 v155; // rax
  __int64 v156; // rdx
  __int64 v157; // rbx
  __int64 v158; // r15
  unsigned int v159; // ecx
  unsigned int v160; // esi
  __int64 v161; // rax
  __int64 v162; // rdx
  __int64 v163; // rax
  void *v164; // rbx
  unsigned __int64 v165; // rdi
  int v166; // r13d
  unsigned __int64 v167; // rax
  unsigned int v168; // r12d
  __int64 v169; // r15
  __int64 v170; // rax
  void *v171; // r13
  void *v172; // r12
  _QWORD *v173; // rax
  _QWORD *v174; // rdi
  __int64 v175; // rax
  __int64 v176; // rdx
  __int64 v177; // r13
  void *v178; // r14
  unsigned __int8 v179; // bl
  int v180; // r12d
  __int64 v181; // r15
  __int64 v182; // rdx
  unsigned int v183; // esi
  __int64 v184; // rax
  unsigned __int64 v185; // rdi
  int v186; // ebx
  unsigned __int64 v187; // r12
  unsigned int v188; // r13d
  int v189; // r8d
  int v190; // r9d
  char v191; // dl
  __int64 v192; // r15
  __int64 *v193; // rax
  __int64 *v194; // rsi
  __int64 *v195; // rcx
  __int64 v196; // rax
  unsigned __int64 v197; // r15
  __int64 *v198; // rax
  unsigned int v199; // eax
  _QWORD *v200; // rax
  __int64 v201; // rcx
  int v202; // r8d
  int v203; // r9d
  __int64 v204; // rbx
  __int64 v205; // r15
  __int64 v206; // r12
  __int64 v207; // rdx
  __int64 v208; // rcx
  int v209; // r8d
  int v210; // r9d
  __int64 v211; // rdx
  unsigned __int64 *v212; // r13
  __int64 v213; // rax
  __int64 v214; // rsi
  __int64 v215; // rdx
  __int64 v216; // rcx
  int v217; // r8d
  int v218; // r9d
  __int64 v219; // r12
  _QWORD *v220; // rdi
  __int64 v221; // rax
  __int64 (__fastcall **v222)(__int64); // rbx
  __int64 (__fastcall **v223)(__int64); // r12
  const char ***v224; // rdi
  __int64 (__fastcall *v225)(__int64); // rax
  __int64 (__fastcall **v226)(__int64); // rax
  __int64 (__fastcall *v227)(__int64); // rdi
  unsigned __int64 *v228; // r12
  __int64 *m; // rbx
  __int64 v230; // rsi
  unsigned int v231; // ecx
  __int64 *v232; // rdx
  __int64 v233; // r10
  __int64 v234; // rsi
  __int64 v235; // r14
  __int64 v236; // rax
  unsigned __int64 v237; // rdi
  int v238; // r8d
  int v239; // r9d
  unsigned __int64 v240; // r10
  unsigned int v241; // ebx
  __int64 v242; // r12
  unsigned int v243; // esi
  __int64 v244; // rax
  __int64 v245; // rax
  __int64 (__fastcall **v246)(__int64); // rbx
  __int64 (__fastcall **v247)(__int64); // r12
  const char ***v248; // rdi
  __int64 (__fastcall *v249)(__int64); // rax
  __int64 (__fastcall *v250)(__int64); // rdi
  __int64 *v251; // rbx
  __int64 v252; // r14
  void **v253; // r12
  __int64 v254; // rdi
  __int64 *v255; // rbx
  __int64 v256; // rdi
  __int64 *v257; // rbx
  __int64 *v258; // r12
  __int64 v259; // rcx
  __int64 *v260; // rax
  __int64 *v261; // rax
  __int64 v262; // rbx
  __int64 *v263; // r14
  __int64 v264; // rdx
  unsigned __int64 v265; // rsi
  __int64 v266; // rdx
  __int64 *v267; // r15
  _QWORD *v268; // rax
  _QWORD *v269; // r13
  __int64 v270; // rdx
  unsigned __int64 v271; // rsi
  __int64 v272; // rdx
  __int64 *v273; // r13
  __int64 v274; // rax
  __int64 (__fastcall *v275)(__int64); // rax
  __int64 v276; // rbx
  __int64 v277; // r15
  __int64 (__fastcall **v278)(__int64); // rdi
  _QWORD *v279; // rax
  __int64 v280; // rdi
  _QWORD *v281; // rbx
  __int64 v282; // rbx
  __int64 v283; // rax
  __int64 v284; // r15
  __int64 (__fastcall **v285)(__int64); // rdi
  __int64 (__fastcall *v286)(__int64); // rax
  __int64 v287; // rdi
  _QWORD *v288; // r12
  _QWORD *n; // rbx
  const char *v290; // r13
  __int64 v291; // rax
  __int64 (__fastcall *v292)(__int64); // rax
  __int64 (__fastcall **v293)(__int64); // rbx
  const char **v294; // rdi
  __int64 (__fastcall **v295)(__int64); // r15
  __int64 *v296; // rax
  __int64 v297; // rdx
  __int64 v298; // rcx
  int v299; // r8d
  int v300; // r9d
  __int64 (__fastcall *v301)(__int64); // rdi
  __int64 (__fastcall *v302)(__int64); // rax
  __int64 (__fastcall **v303)(__int64); // rbx
  const char **v304; // rdi
  __int64 (__fastcall **v305)(__int64); // r15
  __int64 (__fastcall *v306)(__int64); // rdi
  int v307; // edx
  int v308; // r8d
  __int64 v309; // r10
  __int64 v310; // r12
  unsigned int v311; // eax
  __int64 v312; // r13
  __int64 v313; // rsi
  __int64 v314; // rax
  __int64 *v315; // r13
  __int64 *v316; // rbx
  __int64 v317; // rax
  _QWORD *v318; // rax
  __int64 v319; // rcx
  unsigned __int64 v320; // rsi
  __int64 v321; // rcx
  __int64 v322; // rdx
  _QWORD *v323; // rax
  __int64 v324; // rbx
  unsigned int v325; // eax
  __int64 v326; // r12
  __int64 v327; // rcx
  __int64 v328; // r15
  _QWORD *v329; // rdi
  _QWORD *v330; // rdx
  int v331; // r8d
  unsigned int v332; // ecx
  __int64 *v333; // rax
  __int64 v334; // r10
  __int64 v335; // rdx
  __int64 v336; // r8
  unsigned __int64 v337; // rbx
  __int64 v338; // rdx
  __int64 v339; // rcx
  int v340; // r8d
  int v341; // r9d
  __int64 v342; // rax
  __int64 v343; // rax
  int v344; // eax
  __int64 *v345; // rbx
  __int64 *v346; // r13
  _QWORD *v347; // rcx
  int v348; // r8d
  __int64 *v349; // rax
  __int64 v350; // r10
  __int64 v351; // rax
  __int64 v352; // rax
  __int64 v353; // rax
  int v354; // eax
  int v355; // r9d
  int v356; // r9d
  __int64 v357; // rax
  __int64 *v358; // rax
  _QWORD *v359; // [rsp+40h] [rbp-530h]
  __int64 *v360; // [rsp+48h] [rbp-528h]
  _QWORD *v363; // [rsp+70h] [rbp-500h]
  __int64 *v364; // [rsp+80h] [rbp-4F0h]
  char *v366; // [rsp+98h] [rbp-4D8h]
  char *v367; // [rsp+B0h] [rbp-4C0h]
  __int64 v368; // [rsp+B8h] [rbp-4B8h]
  __int64 v369; // [rsp+D0h] [rbp-4A0h]
  __int64 v370; // [rsp+D8h] [rbp-498h]
  __int64 v371; // [rsp+E0h] [rbp-490h]
  __int64 *v372; // [rsp+E8h] [rbp-488h]
  char *v373; // [rsp+F0h] [rbp-480h]
  _BOOL4 v377; // [rsp+110h] [rbp-460h]
  char v378; // [rsp+115h] [rbp-45Bh]
  char v379; // [rsp+116h] [rbp-45Ah]
  unsigned __int8 v380; // [rsp+117h] [rbp-459h]
  unsigned __int64 *v381; // [rsp+118h] [rbp-458h]
  __int64 v382; // [rsp+120h] [rbp-450h]
  __int64 *v383; // [rsp+120h] [rbp-450h]
  __int64 *v384; // [rsp+128h] [rbp-448h]
  unsigned __int64 v385; // [rsp+128h] [rbp-448h]
  unsigned __int64 j; // [rsp+130h] [rbp-440h]
  __int64 *v387; // [rsp+130h] [rbp-440h]
  _QWORD *v388; // [rsp+138h] [rbp-438h]
  __int64 v389; // [rsp+140h] [rbp-430h]
  __int64 *v390; // [rsp+140h] [rbp-430h]
  char **v391; // [rsp+140h] [rbp-430h]
  _QWORD *v392; // [rsp+140h] [rbp-430h]
  __int64 *v393; // [rsp+148h] [rbp-428h]
  unsigned __int64 *v394; // [rsp+148h] [rbp-428h]
  __int64 v395; // [rsp+148h] [rbp-428h]
  __int64 v396; // [rsp+148h] [rbp-428h]
  __int64 *v397; // [rsp+150h] [rbp-420h]
  __int64 v398; // [rsp+150h] [rbp-420h]
  __int64 *v399; // [rsp+150h] [rbp-420h]
  __int64 v400; // [rsp+158h] [rbp-418h]
  int v401; // [rsp+158h] [rbp-418h]
  __int64 *v402; // [rsp+158h] [rbp-418h]
  int v403; // [rsp+158h] [rbp-418h]
  unsigned __int8 v404; // [rsp+158h] [rbp-418h]
  __int64 v405; // [rsp+160h] [rbp-410h]
  __int64 v406; // [rsp+160h] [rbp-410h]
  __int64 (__fastcall *v407)(__int64); // [rsp+160h] [rbp-410h]
  unsigned __int64 v408; // [rsp+160h] [rbp-410h]
  int v409; // [rsp+160h] [rbp-410h]
  __m128i *v410; // [rsp+160h] [rbp-410h]
  __int64 *v411; // [rsp+160h] [rbp-410h]
  __int64 (__fastcall *v412)(__int64); // [rsp+160h] [rbp-410h]
  __int64 v413; // [rsp+160h] [rbp-410h]
  const char **v414; // [rsp+160h] [rbp-410h]
  __int64 v415; // [rsp+168h] [rbp-408h]
  __int64 v416; // [rsp+168h] [rbp-408h]
  __int64 v417; // [rsp+168h] [rbp-408h]
  __int64 v418; // [rsp+168h] [rbp-408h]
  __int64 v419; // [rsp+168h] [rbp-408h]
  __int64 *v420; // [rsp+168h] [rbp-408h]
  _QWORD *v421; // [rsp+168h] [rbp-408h]
  const char **v422; // [rsp+168h] [rbp-408h]
  __int64 v423; // [rsp+168h] [rbp-408h]
  __int64 v424; // [rsp+170h] [rbp-400h] BYREF
  __int64 v425; // [rsp+178h] [rbp-3F8h] BYREF
  __int64 v426; // [rsp+180h] [rbp-3F0h] BYREF
  __int64 v427; // [rsp+188h] [rbp-3E8h] BYREF
  __int64 v428[2]; // [rsp+190h] [rbp-3E0h] BYREF
  const char *v429; // [rsp+1A0h] [rbp-3D0h] BYREF
  const char *v430; // [rsp+1A8h] [rbp-3C8h]
  __int64 v431; // [rsp+1B0h] [rbp-3C0h]
  const char **v432; // [rsp+1B8h] [rbp-3B8h]
  __int64 (__fastcall *v433)(__int64); // [rsp+1C0h] [rbp-3B0h] BYREF
  __int64 *v434; // [rsp+1C8h] [rbp-3A8h]
  __int64 (__fastcall *v435)(__int64 *); // [rsp+1D0h] [rbp-3A0h]
  const char **v436; // [rsp+1D8h] [rbp-398h]
  char **v437; // [rsp+1E0h] [rbp-390h] BYREF
  __int64 v438; // [rsp+1E8h] [rbp-388h]
  _BYTE v439[32]; // [rsp+1F0h] [rbp-380h] BYREF
  __int64 *v440; // [rsp+210h] [rbp-360h] BYREF
  __int64 v441; // [rsp+218h] [rbp-358h]
  char v442; // [rsp+220h] [rbp-350h] BYREF
  const char **v443; // [rsp+240h] [rbp-330h] BYREF
  __int64 v444; // [rsp+248h] [rbp-328h]
  __int64 v445; // [rsp+250h] [rbp-320h] BYREF
  __int64 v446; // [rsp+258h] [rbp-318h]
  __int64 (__fastcall *v447)(__int64); // [rsp+270h] [rbp-300h] BYREF
  __int64 v448; // [rsp+278h] [rbp-2F8h]
  __int64 (__fastcall *v449)(__int64 *); // [rsp+280h] [rbp-2F0h] BYREF
  __int64 v450; // [rsp+288h] [rbp-2E8h]
  __int64 *v451; // [rsp+2A0h] [rbp-2D0h] BYREF
  __int64 v452; // [rsp+2A8h] [rbp-2C8h]
  _BYTE v453[64]; // [rsp+2B0h] [rbp-2C0h] BYREF
  unsigned __int64 v454; // [rsp+2F0h] [rbp-280h] BYREF
  __int64 v455; // [rsp+2F8h] [rbp-278h]
  __int64 *v456; // [rsp+300h] [rbp-270h] BYREF
  __int64 *v457; // [rsp+308h] [rbp-268h]
  __int64 v458; // [rsp+340h] [rbp-230h] BYREF
  __int64 v459; // [rsp+348h] [rbp-228h]
  __int64 v460; // [rsp+350h] [rbp-220h] BYREF
  __int64 *v461; // [rsp+370h] [rbp-200h] BYREF
  __int64 v462; // [rsp+378h] [rbp-1F8h]
  _BYTE v463[32]; // [rsp+380h] [rbp-1F0h] BYREF
  char *p_s; // [rsp+3A0h] [rbp-1D0h] BYREF
  unsigned __int64 v465; // [rsp+3A8h] [rbp-1C8h]
  void *s; // [rsp+3B0h] [rbp-1C0h] BYREF
  _BYTE v467[12]; // [rsp+3B8h] [rbp-1B8h]
  _BYTE v468[104]; // [rsp+3C8h] [rbp-1A8h] BYREF
  __int64 v469; // [rsp+430h] [rbp-140h] BYREF
  __int64 v470; // [rsp+438h] [rbp-138h]
  _QWORD *v471; // [rsp+440h] [rbp-130h] BYREF
  unsigned int v472; // [rsp+448h] [rbp-128h]
  char v473; // [rsp+540h] [rbp-30h] BYREF

  v373 = *(char **)(a2 + 40);
  if ( *(_BYTE *)(a2 + 16) == 26 )
  {
    v18 = *(_QWORD *)(a2 - 72);
    if ( *a3 == v18 )
    {
      v378 = 1;
      v20 = -48;
      v377 = 0;
      v379 = 1;
    }
    else
    {
      v19 = *(_BYTE *)(v18 + 16) == 51;
      v20 = -48;
      v378 = 0;
      if ( !v19 )
        v20 = -24;
      v379 = v19;
      v377 = !v19;
    }
    v369 = 0;
    v370 = a2;
    v21 = *(_QWORD *)(a2 + v20);
  }
  else
  {
    v369 = a2;
    v370 = 0;
    v21 = *(_QWORD *)(sub_13CF970(a2) + 24);
    v378 = 1;
    v377 = 0;
    v379 = 1;
  }
  v424 = v21;
  v22 = (unsigned __int64 *)&v460;
  v458 = 0;
  v459 = 1;
  do
    *v22++ = -8;
  while ( v22 != (unsigned __int64 *)&v461 );
  v461 = (__int64 *)v463;
  v462 = 0x400000000LL;
  if ( v370 )
  {
    v469 = *(_QWORD *)(v370 - 24LL * v377 - 24);
    sub_1A589C0((__int64)&v458, &v469);
  }
  else
  {
    v139 = 0;
    v140 = (*(_DWORD *)(v369 + 20) & 0xFFFFFFFu) >> 1;
    v141 = v140 - 1;
    if ( v140 != 1 )
    {
      do
      {
        v144 = 24;
        if ( (_DWORD)v139 != -2 )
          v144 = 24LL * (unsigned int)(2 * v139 + 3);
        if ( (*(_BYTE *)(v369 + 23) & 0x40) != 0 )
          v142 = *(_QWORD *)(v369 - 8);
        else
          v142 = v369 - 24LL * (*(_DWORD *)(v369 + 20) & 0xFFFFFFF);
        v143 = *(_QWORD *)(v142 + v144);
        if ( v424 != v143 )
        {
          v469 = v143;
          sub_1A589C0((__int64)&v458, &v469);
        }
        ++v139;
      }
      while ( v139 != v141 );
    }
  }
  v437 = (char **)v439;
  v438 = 0x400000000LL;
  sub_13FA0E0(a1, (__int64)&v437);
  v23 = (__int64 *)v437;
  v24 = &v437[(unsigned int)v438];
  if ( v24 == v437 )
  {
    v363 = *(_QWORD **)a1;
  }
  else
  {
    do
    {
      if ( *(_BYTE *)(sub_157ED20(*v23) + 16) == 73 )
      {
        v380 = 0;
        goto LABEL_16;
      }
      ++v23;
    }
    while ( v24 != (char **)v23 );
    v26 = (__int64 *)v437;
    v363 = *(_QWORD **)a1;
    v27 = &v437[(unsigned int)v438];
    if ( v437 != v27 )
    {
      v28 = (_QWORD *)a1;
      while ( 1 )
      {
        v29 = sub_13AE450(a6, *v26);
        v30 = (_QWORD *)v29;
        if ( !v29 )
          break;
        if ( (_QWORD *)v29 == v28 || !v28 )
        {
LABEL_25:
          if ( v27 == (char **)++v26 )
            goto LABEL_33;
        }
        else
        {
          v31 = v28;
          while ( 1 )
          {
            v31 = (_QWORD *)*v31;
            if ( v30 == v31 )
              break;
            if ( !v31 )
              goto LABEL_25;
          }
          ++v26;
          v28 = v30;
          if ( v27 == (char **)v26 )
          {
LABEL_33:
            v368 = (__int64)v28;
            if ( !a18 )
              goto LABEL_36;
            if ( v28 )
            {
LABEL_35:
              sub_1465150(a18, v368);
              goto LABEL_36;
            }
LABEL_198:
            sub_1465DB0(a18, (_QWORD *)a1);
            v368 = 0;
            goto LABEL_36;
          }
        }
      }
      if ( !a18 )
      {
        v368 = 0;
        goto LABEL_36;
      }
      goto LABEL_198;
    }
  }
  v368 = a1;
  if ( a18 )
    goto LABEL_35;
LABEL_36:
  v32 = (char *)&v471;
  v469 = 0;
  v470 = 1;
  do
  {
    *(_QWORD *)v32 = -8;
    v32 += 16;
  }
  while ( v32 != &v473 );
  v33 = (__m128i *)&p_s;
  v454 = (unsigned __int64)v461;
  v456 = &v424;
  v457 = &v425;
  v397 = &v461[(unsigned int)v462];
  v455 = (__int64)v397;
  do
  {
    v34 = &p_s;
    v465 = 0;
    v35 = &p_s;
    p_s = (char *)sub_1A4EBF0;
    v36 = &v454;
    *(_QWORD *)v467 = 0;
    s = sub_1A4EC10;
    v37 = sub_1A4EBF0;
    if ( ((unsigned __int8)sub_1A4EBF0 & 1) == 0 )
      goto LABEL_41;
    while ( 1 )
    {
      v37 = *(__int64 (__fastcall **)(__int64))((char *)v37 + *v36 - 1);
LABEL_41:
      v38 = (__int64 *)v37((__int64)v36);
      if ( v38 )
        break;
      while ( 1 )
      {
        v39 = v35[3];
        v37 = (__int64 (__fastcall *)(__int64))v35[2];
        v34 += 2;
        v35 = v34;
        v36 = (unsigned __int64 *)((char *)&v454 + (_QWORD)v39);
        if ( ((unsigned __int8)v37 & 1) != 0 )
          break;
        v38 = (__int64 *)v37((__int64)v36);
        if ( v38 )
          goto LABEL_44;
      }
    }
LABEL_44:
    v40 = *v38;
    if ( !sub_157F120(*v38) )
    {
      v135 = *(_QWORD *)(v40 + 8);
      if ( v135 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v135) + 16) - 25) > 9u )
        {
          v135 = *(_QWORD *)(v135 + 8);
          if ( !v135 )
            goto LABEL_45;
        }
        v136 = v135;
        v137 = sub_1648700(v135);
LABEL_173:
        v138 = v137[5];
        if ( v373 != (char *)v138 && !sub_15CC8F0(a5, v40, v138) )
          goto LABEL_61;
        while ( 1 )
        {
          v136 = *(_QWORD *)(v136 + 8);
          if ( !v136 )
            break;
          v137 = sub_1648700(v136);
          if ( (unsigned __int8)(*((_BYTE *)v137 + 16) - 25) <= 9u )
            goto LABEL_173;
        }
      }
    }
LABEL_45:
    p_s = (char *)&s;
    v465 = 0x400000000LL;
    v41 = 0;
    v42 = *(unsigned int *)(a5 + 48);
    if ( !(_DWORD)v42 )
      goto LABEL_49;
    v43 = *(_QWORD *)(a5 + 32);
    v44 = (v42 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v45 = (__int64 *)(v43 + 16LL * v44);
    v46 = *v45;
    if ( v40 != *v45 )
    {
      v145 = 1;
      while ( v46 != -8 )
      {
        v356 = v145 + 1;
        v44 = (v42 - 1) & (v145 + v44);
        v45 = (__int64 *)(v43 + 16LL * v44);
        v46 = *v45;
        if ( v40 == *v45 )
          goto LABEL_47;
        v145 = v356;
      }
LABEL_196:
      v41 = 0;
      goto LABEL_49;
    }
LABEL_47:
    if ( v45 == (__int64 *)(v43 + 16 * v42) )
      goto LABEL_196;
    v41 = v45[1];
LABEL_49:
    LODWORD(v465) = 1;
    v47 = &s;
    v415 = v40;
    s = (void *)v41;
    LODWORD(v41) = 1;
    do
    {
      while ( 1 )
      {
        v48 = (__int64 *)v47[(unsigned int)v41 - 1];
        LODWORD(v465) = v41 - 1;
        v49 = *v48;
        v447 = (__int64 (__fastcall *)(__int64))*v48;
        if ( (v470 & 1) != 0 )
        {
          v50 = &v471;
          v51 = 15;
        }
        else
        {
          v128 = v472;
          v50 = v471;
          if ( !v472 )
          {
            v129 = v470;
            ++v469;
            v53 = 0;
            v130 = ((unsigned int)v470 >> 1) + 1;
LABEL_151:
            LODWORD(v54) = 3 * v128;
            goto LABEL_152;
          }
          v51 = v472 - 1;
        }
        LODWORD(v52) = v51 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
        v53 = &v50[2 * (unsigned int)v52];
        v54 = *v53;
        if ( *v53 == v49 )
          goto LABEL_53;
        v132 = 1;
        v133 = 0;
        while ( v54 != -8 )
        {
          if ( !v133 && v54 == -16 )
            v133 = v53;
          v52 = v51 & (unsigned int)(v52 + v132);
          v53 = &v50[2 * v52];
          v54 = *v53;
          if ( v49 == *v53 )
            goto LABEL_53;
          ++v132;
        }
        v129 = v470;
        LODWORD(v54) = 48;
        v128 = 16;
        if ( v133 )
          v53 = v133;
        ++v469;
        v130 = ((unsigned int)v470 >> 1) + 1;
        if ( (v470 & 1) == 0 )
        {
          v128 = v472;
          goto LABEL_151;
        }
LABEL_152:
        if ( 4 * v130 >= (unsigned int)v54 )
        {
          v134 = 2 * v128;
          goto LABEL_166;
        }
        v131 = v128 - HIDWORD(v470) - v130;
        LODWORD(v50) = v128 >> 3;
        if ( v131 <= v128 >> 3 )
        {
          v134 = v128;
LABEL_166:
          sub_1A54EF0((__int64)&v469, v134);
          sub_1A54390((__int64)&v469, (__int64 *)&v447, &v451);
          v53 = v451;
          v49 = (__int64)v447;
          v129 = v470;
        }
        LODWORD(v470) = (2 * (v129 >> 1) + 2) | v129 & 1;
        if ( *v53 != -8 )
          --HIDWORD(v470);
        *v53 = v49;
        v53[1] = 0;
LABEL_53:
        v53[1] = v415;
        v55 = (__int64 *)v48[4];
        if ( (__int64 *)v48[3] != v55 )
          break;
        LODWORD(v41) = v465;
        v47 = (void **)p_s;
        if ( !(_DWORD)v465 )
          goto LABEL_59;
      }
      v41 = (unsigned int)v465;
      v56 = (__int64 *)v48[3];
      do
      {
        v57 = *v56;
        if ( (unsigned int)v41 >= HIDWORD(v465) )
        {
          sub_16CD150((__int64)&p_s, &s, 0, 8, (int)v50, v54);
          v41 = (unsigned int)v465;
        }
        ++v56;
        *(_QWORD *)&p_s[8 * v41] = v57;
        v41 = (unsigned int)(v465 + 1);
        LODWORD(v465) = v465 + 1;
      }
      while ( v55 != v56 );
      v47 = (void **)p_s;
    }
    while ( (_DWORD)v41 );
LABEL_59:
    if ( v47 != &s )
      _libc_free((unsigned __int64)v47);
LABEL_61:
    p_p_s = &p_s;
    v465 = 0;
    v59 = &p_s;
    p_s = (char *)sub_1A4EB90;
    v60 = &v454;
    *(_QWORD *)v467 = 0;
    s = sub_1A4EBC0;
    v61 = sub_1A4EB90;
    if ( ((unsigned __int8)sub_1A4EB90 & 1) != 0 )
LABEL_62:
      v61 = *(__int64 (__fastcall **)(__int64))((char *)v61 + *v60 - 1);
    for ( i = v61((__int64)v60); !i; i = v61((__int64)v60) )
    {
      v63 = v59[3];
      v61 = (__int64 (__fastcall *)(__int64))v59[2];
      p_p_s += 2;
      v59 = p_p_s;
      v60 = (unsigned __int64 *)((char *)&v454 + (_QWORD)v63);
      if ( ((unsigned __int8)v61 & 1) != 0 )
        goto LABEL_62;
    }
  }
  while ( v456 != &v425 || v457 != &v425 || v397 != (__int64 *)v454 || v397 != (__int64 *)v455 );
  v380 = i;
  v367 = (char *)sub_13FC520(a1);
  v371 = sub_1AA91E0(v367, **(_QWORD **)(a1 + 32), a5, a6);
  v451 = (__int64 *)v453;
  v452 = 0x400000000LL;
  v440 = (__int64 *)&v442;
  v441 = 0x400000000LL;
  if ( (unsigned int)v462 > 4 )
    sub_1A53B20((__int64)&v440, (unsigned int)v462);
  v64 = &v456;
  v454 = 0;
  v455 = 1;
  do
  {
    *v64 = (__int64 *)-8LL;
    v64 += 2;
  }
  while ( v64 != (__int64 **)&v458 );
  v360 = &v461[(unsigned int)v462];
  if ( v360 == v461 )
    goto LABEL_301;
  v364 = v461;
  while ( 2 )
  {
    v425 = *v364;
    v65 = (_QWORD *)sub_22077B0(80);
    v66 = (__int64)v65;
    if ( v65 )
    {
      *v65 = 0;
      v67 = sub_1454B60(0x56u);
      *(_DWORD *)(v66 + 24) = v67;
      if ( v67 )
      {
        *(_QWORD *)(v66 + 8) = sub_22077B0((unsigned __int64)v67 << 6);
        sub_1954940(v66);
      }
      else
      {
        *(_QWORD *)(v66 + 8) = 0;
        *(_QWORD *)(v66 + 16) = 0;
      }
      *(_BYTE *)(v66 + 64) = 0;
      *(_BYTE *)(v66 + 73) = 1;
    }
    v68 = v441;
    if ( (unsigned int)v441 >= HIDWORD(v441) )
    {
      sub_1A53B20((__int64)&v440, 0);
      v68 = v441;
    }
    v69 = (unsigned __int64)v440;
    v70 = &v440[v68];
    if ( v70 )
    {
      *v70 = v66;
      v69 = (unsigned __int64)v440;
      v68 = v441;
    }
    v71 = (unsigned int)(v68 + 1);
    LODWORD(v441) = v71;
    v72 = v425;
    v73 = v437;
    v382 = *(_QWORD *)(v69 + 8 * v71 - 8);
    v74 = (unsigned int)v438;
    v75 = sub_1A54450((__int64)&v454, &v425, v33);
    v366 = p_s;
    if ( !v75 )
    {
      ++v454;
      v78 = ((unsigned int)v455 >> 1) + 1;
      if ( (v455 & 1) == 0 )
      {
        v79 = (unsigned int)v457;
        if ( 4 * v78 < 3 * (int)v457 )
          goto LABEL_87;
LABEL_447:
        v79 *= 2;
        goto LABEL_448;
      }
      v79 = 4;
      if ( 4 * v78 >= 0xC )
        goto LABEL_447;
LABEL_87:
      if ( v79 - (v78 + HIDWORD(v455)) <= v79 >> 3 )
      {
LABEL_448:
        sub_1A552B0((__int64)&v454, v79);
        sub_1A54450((__int64)&v454, &v425, v33);
        v366 = p_s;
        v78 = ((unsigned int)v455 >> 1) + 1;
      }
      LODWORD(v455) = v455 & 1 | (2 * v78);
      if ( *(_QWORD *)v366 != -8 )
        --HIDWORD(v455);
      v80 = v425;
      *((_QWORD *)v366 + 1) = 0;
      *(_QWORD *)v366 = v80;
    }
    v81 = v371;
    v427 = v72;
    v447 = (__int64 (__fastcall *)(__int64))&v449;
    v448 = 0x400000000LL;
    v426 = v371;
    v82 = v74 + (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3);
    if ( v82 > 4 )
    {
      sub_16CD150((__int64)&v447, &v449, v82, 8, v76, v77);
      v81 = v426;
    }
    v433 = (__int64 (__fastcall *)(__int64))v382;
    v434 = &v426;
    v435 = (__int64 (__fastcall *)(__int64 *))&v447;
    v428[0] = (__int64)&v469;
    v428[1] = (__int64)&v427;
    v359 = sub_1A56F40((__int64)&v433, v81);
    v83 = *(__int64 **)(a1 + 40);
    if ( *(__int64 **)(a1 + 32) != v83 )
    {
      v405 = v74;
      v84 = *(__int64 **)(a1 + 32);
      do
      {
        while ( 1 )
        {
          v85 = *v84;
          if ( !(unsigned __int8)sub_1A4EF50(v428, *v84) )
            break;
          if ( v83 == ++v84 )
            goto LABEL_98;
        }
        ++v84;
        sub_1A56F40((__int64)&v433, v85);
      }
      while ( v83 != v84 );
LABEL_98:
      v74 = v405;
    }
    v372 = (__int64 *)&v73[v74];
    if ( v73 != (char **)v372 )
    {
      v393 = (__int64 *)v73;
      v384 = (__int64 *)v33;
      do
      {
        v86 = *v393;
        v398 = *v393;
        if ( !(unsigned __int8)sub_1A4EF50(v428, *v393) )
        {
          v87 = *(_QWORD *)(v86 + 48);
          if ( v87 )
            v87 -= 24;
          v88 = sub_1AA8CA0(v86, v87, a5, a6);
          v89 = v86;
          v90 = v88;
          v389 = v88;
          sub_164B7C0(v88, v89);
          p_s = ".split";
          LOWORD(s) = 259;
          v429 = sub_1649960(v90);
          v91 = (char)s;
          v430 = v92;
          if ( (_BYTE)s )
          {
            if ( (_BYTE)s == 1 )
            {
              v443 = &v429;
              LOWORD(v445) = 261;
            }
            else
            {
              v93 = p_s;
              if ( BYTE1(s) != 1 )
              {
                v93 = (char *)v384;
                v91 = 2;
              }
              v444 = (__int64)v93;
              v443 = &v429;
              LOBYTE(v445) = 5;
              BYTE1(v445) = v91;
            }
          }
          else
          {
            LOWORD(v445) = 256;
          }
          sub_164B780(v398, (__int64 *)&v443);
          v388 = sub_1A56F40((__int64)&v433, v398);
          v416 = v388[6];
          v94 = *(_QWORD *)(v398 + 48);
          for ( j = *(_QWORD *)(v398 + 40) & 0xFFFFFFFFFFFFFFF8LL; v94 != j; v416 = *(_QWORD *)(v416 + 8) )
          {
            v115 = (__int64 *)(v94 - 24);
            v116 = v416 - 24;
            if ( !v416 )
              v116 = 0;
            if ( !v94 )
              v115 = 0;
            v117 = sub_157EE30(v389);
            LOWORD(s) = 259;
            if ( v117 )
              v117 -= 24;
            p_s = ".us-phi";
            v400 = v117;
            v406 = *v115;
            v118 = sub_1648B60(64);
            v121 = v118;
            if ( v118 )
            {
              v122 = v118;
              sub_15F1EA0(v118, v406, 53, 0, 0, v400);
              *(_DWORD *)(v121 + 56) = 2;
              sub_164B780(v121, v384);
              sub_1648880(v121, *(_DWORD *)(v121 + 56), 1);
            }
            else
            {
              v122 = 0;
            }
            sub_164D160((__int64)v115, v121, a7, a8, a9, a10, v119, v120, a13, a14);
            v125 = *(_DWORD *)(v121 + 20) & 0xFFFFFFF;
            if ( (_DWORD)v125 == *(_DWORD *)(v121 + 56) )
            {
              sub_15F55D0(v121, v121, v125, v123, v100, v124);
              LODWORD(v125) = *(_DWORD *)(v121 + 20) & 0xFFFFFFF;
            }
            v126 = ((_DWORD)v125 + 1) & 0xFFFFFFF;
            v127 = v126 | *(_DWORD *)(v121 + 20) & 0xF0000000;
            *(_DWORD *)(v121 + 20) = v127;
            if ( (v127 & 0x40000000) != 0 )
              v95 = *(_QWORD *)(v121 - 8);
            else
              v95 = v122 - 24 * v126;
            v96 = (__int64 **)(v95 + 24LL * (unsigned int)(v126 - 1));
            if ( *v96 )
            {
              v97 = v96[1];
              v98 = (unsigned __int64)v96[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v98 = v97;
              if ( v97 )
                v97[2] = v97[2] & 3 | v98;
            }
            *v96 = v115;
            v99 = v115[1];
            v96[1] = (__int64 *)v99;
            if ( v99 )
            {
              v100 = (__int64)(v96 + 1);
              *(_QWORD *)(v99 + 16) = (unsigned __int64)(v96 + 1) | *(_QWORD *)(v99 + 16) & 3LL;
            }
            v96[2] = (__int64 *)((unsigned __int64)(v115 + 1) | (unsigned __int64)v96[2] & 3);
            v115[1] = (__int64)v96;
            v101 = *(_DWORD *)(v121 + 20) & 0xFFFFFFF;
            v102 = (unsigned int)(v101 - 1);
            if ( (*(_BYTE *)(v121 + 23) & 0x40) != 0 )
              v103 = *(_QWORD *)(v121 - 8);
            else
              v103 = v122 - 24 * v101;
            v104 = 3LL * *(unsigned int *)(v121 + 56);
            *(_QWORD *)(v103 + 8 * v102 + 24LL * *(unsigned int *)(v121 + 56) + 8) = v398;
            v105 = *(_DWORD *)(v121 + 20) & 0xFFFFFFF;
            if ( (_DWORD)v105 == *(_DWORD *)(v121 + 56) )
            {
              sub_15F55D0(v121, v103, v105, v104, v100, v124);
              LODWORD(v105) = *(_DWORD *)(v121 + 20) & 0xFFFFFFF;
            }
            v106 = ((_DWORD)v105 + 1) & 0xFFFFFFF;
            v107 = v106 | *(_DWORD *)(v121 + 20) & 0xF0000000;
            *(_DWORD *)(v121 + 20) = v107;
            if ( (v107 & 0x40000000) != 0 )
              v108 = *(_QWORD *)(v121 - 8);
            else
              v108 = v122 - 24 * v106;
            v109 = (_QWORD *)(v108 + 24LL * (unsigned int)(v106 - 1));
            if ( *v109 )
            {
              v110 = v109[1];
              v111 = v109[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v111 = v110;
              if ( v110 )
                *(_QWORD *)(v110 + 16) = *(_QWORD *)(v110 + 16) & 3LL | v111;
            }
            *v109 = v116;
            if ( v116 )
            {
              v112 = *(_QWORD *)(v116 + 8);
              v109[1] = v112;
              if ( v112 )
                *(_QWORD *)(v112 + 16) = (unsigned __int64)(v109 + 1) | *(_QWORD *)(v112 + 16) & 3LL;
              v109[2] = (v116 + 8) | v109[2] & 3LL;
              *(_QWORD *)(v116 + 8) = v109;
            }
            v113 = *(_DWORD *)(v121 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v121 + 23) & 0x40) != 0 )
              v114 = *(_QWORD *)(v121 - 8);
            else
              v114 = v122 - 24 * v113;
            *(_QWORD *)(v114 + 8LL * (unsigned int)(v113 - 1) + 24LL * *(unsigned int *)(v121 + 56) + 8) = v388;
            v94 = *(_QWORD *)(v94 + 8);
          }
        }
        ++v393;
      }
      while ( v372 != v393 );
      v33 = (__m128i *)v384;
    }
    v407 = (__int64 (__fastcall *)(__int64))((char *)v447 + 8 * (unsigned int)v448);
    if ( v447 != v407 )
    {
      v417 = (__int64)v447;
      do
      {
        v146 = *(_QWORD *)(*(_QWORD *)v417 + 48LL);
        for ( k = *(_QWORD *)v417 + 40LL; k != v146; v146 = *(_QWORD *)(v146 + 8) )
        {
          while ( 1 )
          {
            if ( !v146 )
            {
              sub_1B75040(v33, v382, 3, 0, 0);
              sub_1B79630(v33, 0);
              sub_1B75110(v33);
              BUG();
            }
            sub_1B75040(v33, v382, 3, 0, 0);
            sub_1B79630(v33, v146 - 24);
            sub_1B75110(v33);
            if ( *(_BYTE *)(v146 - 8) == 78 )
            {
              v148 = *(_QWORD *)(v146 - 48);
              if ( !*(_BYTE *)(v148 + 16) && (*(_BYTE *)(v148 + 33) & 0x20) != 0 && *(_DWORD *)(v148 + 36) == 4 )
                break;
            }
            v146 = *(_QWORD *)(v146 + 8);
            if ( k == v146 )
              goto LABEL_217;
          }
          sub_14CE830(a15, v146 - 24);
        }
LABEL_217:
        v417 += 8;
      }
      while ( v407 != (__int64 (__fastcall *)(__int64))v417 );
    }
    v399 = *(__int64 **)(a1 + 40);
    if ( *(__int64 **)(a1 + 32) != v399 )
    {
      v394 = (unsigned __int64 *)v33;
      v149 = *(__int64 **)(a1 + 32);
      do
      {
        while ( 1 )
        {
          v150 = *v149;
          if ( (unsigned __int8)sub_1A4EF50(v428, *v149) )
          {
            v151 = sub_157EBA0(v150);
            if ( v151 )
            {
              v401 = sub_15F4D60(v151);
              v408 = sub_157EBA0(v150);
              if ( v401 )
                break;
            }
          }
          if ( v399 == ++v149 )
            goto LABEL_242;
        }
        v390 = v149;
        v152 = 0;
        v153 = v150;
        do
        {
          v443 = (const char **)sub_15F4DF0(v408, v152);
          sub_1A51850(v394, v382, (__int64 *)&v443);
          v154 = s;
          if ( s )
          {
            if ( s != (void *)-16LL && s != (void *)-8LL )
              sub_1649B30(v394);
            v155 = sub_157F280((__int64)v154);
            v157 = v156;
            v158 = v155;
            while ( v157 != v158 )
            {
              v159 = *(_DWORD *)(v158 + 20) & 0xFFFFFFF;
              if ( v159 )
              {
                v160 = 0;
                v161 = 24LL * *(unsigned int *)(v158 + 56) + 8;
                while ( 1 )
                {
                  v162 = v158 - 24LL * v159;
                  if ( (*(_BYTE *)(v158 + 23) & 0x40) != 0 )
                    v162 = *(_QWORD *)(v158 - 8);
                  if ( v153 == *(_QWORD *)(v162 + v161) )
                    break;
                  ++v160;
                  v161 += 8;
                  if ( v159 == v160 )
                    goto LABEL_261;
                }
              }
              else
              {
LABEL_261:
                v160 = -1;
              }
              sub_15F5350(v158, v160, 0);
              v163 = *(_QWORD *)(v158 + 32);
              if ( !v163 )
                BUG();
              v158 = 0;
              if ( *(_BYTE *)(v163 - 8) == 77 )
                v158 = v163 - 24;
            }
          }
          ++v152;
        }
        while ( v401 != v152 );
        ++v149;
      }
      while ( v399 != v390 + 1 );
LABEL_242:
      v33 = (__m128i *)v394;
    }
    v443 = (const char **)v373;
    sub_1A51850((unsigned __int64 *)v33, v382, (__int64 *)&v443);
    v164 = s;
    sub_1455FA0((__int64)v33);
    v165 = sub_157EBA0((__int64)v373);
    if ( v165 )
    {
      v166 = sub_15F4D60(v165);
      v167 = sub_157EBA0((__int64)v373);
      if ( v166 )
      {
        v409 = v166;
        v168 = 0;
        v169 = v167;
        do
        {
          v170 = sub_15F4DF0(v169, v168);
          if ( v170 != v427 )
          {
            v443 = (const char **)v170;
            sub_1A51850((unsigned __int64 *)v33, v382, (__int64 *)&v443);
            v171 = s;
            if ( s )
            {
              sub_1455FA0((__int64)v33);
              sub_157F2D0((__int64)v171, (__int64)v164, 1);
            }
            else
            {
              sub_1455FA0((__int64)v33);
            }
          }
          ++v168;
        }
        while ( v409 != v168 );
      }
    }
    v443 = (const char **)v427;
    sub_1A51850((unsigned __int64 *)v33, v382, (__int64 *)&v443);
    v172 = s;
    sub_1455FA0((__int64)v33);
    v173 = (_QWORD *)sub_157EBA0((__int64)v164);
    sub_15F20C0(v173);
    v174 = sub_1648A60(56, 1u);
    if ( v174 )
      sub_15F8590((__int64)v174, (__int64)v172, (__int64)v164);
    v175 = sub_157F280((__int64)v172);
    v418 = v176;
    v177 = v175;
    if ( v175 == v176 )
      goto LABEL_269;
    v410 = v33;
    v178 = v164;
    while ( 2 )
    {
      v179 = 0;
      v180 = (*(_DWORD *)(v177 + 20) & 0xFFFFFFF) - 1;
      v181 = 8LL * v180;
      if ( (*(_DWORD *)(v177 + 20) & 0xFFFFFFF) != 0 )
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v177 + 23) & 0x40) != 0 )
            v182 = *(_QWORD *)(v177 - 8);
          else
            v182 = v177 - 24LL * (*(_DWORD *)(v177 + 20) & 0xFFFFFFF);
          if ( v178 != *(void **)(v181 + v182 + 24LL * *(unsigned int *)(v177 + 56) + 8) )
            goto LABEL_258;
          if ( v179 )
          {
            v183 = v180--;
            sub_15F5350(v177, v183, 0);
            v181 -= 8;
            if ( v180 == -1 )
              break;
          }
          else
          {
            v179 = v380;
LABEL_258:
            --v180;
            v181 -= 8;
            if ( v180 == -1 )
              break;
          }
        }
      }
      v184 = *(_QWORD *)(v177 + 32);
      if ( !v184 )
        BUG();
      v177 = 0;
      if ( *(_BYTE *)(v184 - 8) == 77 )
        v177 = v184 - 24;
      if ( v418 != v177 )
        continue;
      break;
    }
    v33 = v410;
LABEL_269:
    p_s = 0;
    v465 = (unsigned __int64)v468;
    s = v468;
    *(_QWORD *)v467 = 4;
    *(_DWORD *)&v467[8] = 0;
    v402 = (__int64 *)((char *)v447 + 8 * (unsigned int)v448);
    if ( (char *)v447 == (char *)v402 )
      goto LABEL_298;
    v411 = (__int64 *)v447;
    do
    {
      v419 = *v411;
      v185 = sub_157EBA0(*v411);
      if ( v185 )
      {
        v186 = sub_15F4D60(v185);
        v187 = sub_157EBA0(v419);
        if ( v186 )
        {
          v188 = 0;
          while ( 1 )
          {
            v192 = sub_15F4DF0(v187, v188);
            v193 = (__int64 *)v465;
            if ( s == (void *)v465 )
            {
              v194 = (__int64 *)(v465 + 8LL * *(unsigned int *)&v467[4]);
              if ( (__int64 *)v465 != v194 )
              {
                v195 = 0;
                while ( v192 != *v193 )
                {
                  if ( *v193 == -2 )
                    v195 = v193;
                  if ( v194 == ++v193 )
                  {
                    if ( !v195 )
                      goto LABEL_435;
                    *v195 = v192;
                    --*(_DWORD *)&v467[8];
                    ++p_s;
                    goto LABEL_285;
                  }
                }
                goto LABEL_275;
              }
LABEL_435:
              if ( *(_DWORD *)&v467[4] < *(_DWORD *)v467 )
                break;
            }
            sub_16CCBA0((__int64)v33, v192);
            if ( v191 )
            {
LABEL_285:
              v196 = (unsigned int)v452;
              v197 = v192 & 0xFFFFFFFFFFFFFFFBLL;
              if ( (unsigned int)v452 >= HIDWORD(v452) )
              {
                sub_16CD150((__int64)&v451, v453, 0, 16, v189, v190);
                v196 = (unsigned int)v452;
              }
              ++v188;
              v198 = &v451[2 * v196];
              v198[1] = v197;
              *v198 = v419;
              LODWORD(v452) = v452 + 1;
              if ( v186 == v188 )
                goto LABEL_288;
            }
            else
            {
LABEL_275:
              if ( v186 == ++v188 )
                goto LABEL_288;
            }
          }
          ++*(_DWORD *)&v467[4];
          *v194 = v192;
          ++p_s;
          goto LABEL_285;
        }
      }
LABEL_288:
      ++p_s;
      if ( s == (void *)v465 )
        goto LABEL_293;
      v199 = 4 * (*(_DWORD *)&v467[4] - *(_DWORD *)&v467[8]);
      if ( v199 < 0x20 )
        v199 = 32;
      if ( v199 >= *(_DWORD *)v467 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v467);
LABEL_293:
        *(_QWORD *)&v467[4] = 0;
        goto LABEL_294;
      }
      sub_16CC920((__int64)v33);
LABEL_294:
      ++v411;
    }
    while ( v402 != v411 );
    if ( s != (void *)v465 )
      _libc_free((unsigned __int64)s);
    v402 = (__int64 *)v447;
LABEL_298:
    if ( v402 != (__int64 *)&v449 )
      _libc_free((unsigned __int64)v402);
    ++v364;
    *((_QWORD *)v366 + 1) = v359;
    if ( v360 != v364 )
      continue;
    break;
  }
LABEL_301:
  v200 = (_QWORD *)sub_157EBA0((__int64)v367);
  sub_15F20C0(v200);
  if ( !v378 )
  {
    sub_1A522E0(v33, (__int64 *)&v454);
    v336 = *((_QWORD *)s + 1);
    v337 = v336 & 0xFFFFFFFFFFFFFFFBLL;
    sub_1A4F120((__int64)v367, (__int64)a3, a4, v379, v336, v371, *(double *)a7.m128_u64, a8, a9);
    p_s = v367;
    v465 = v337;
    sub_1A51800((__int64)&v451, v33, v338, v339, v340, v341);
    goto LABEL_314;
  }
  v204 = a2 + 24;
  v205 = (__int64)(v367 + 40);
  if ( !v370 )
  {
    v310 = 0;
    v311 = (*(_DWORD *)(v369 + 20) & 0xFFFFFFFu) >> 1;
    v312 = v311 - 1;
    if ( v311 != 1 )
    {
      do
      {
        v314 = 24;
        if ( (_DWORD)v310 != -2 )
          v314 = 24LL * (unsigned int)(2 * v310 + 3);
        if ( (*(_BYTE *)(v369 + 23) & 0x40) != 0 )
          v313 = *(_QWORD *)(v369 - 8);
        else
          v313 = v369 - 24LL * (*(_DWORD *)(v369 + 20) & 0xFFFFFFF);
        ++v310;
        sub_157F2D0(*(_QWORD *)(v313 + v314), (__int64)v373, 1);
      }
      while ( v312 != v310 );
      v204 = a2 + 24;
    }
    v315 = v461;
    v211 = (__int64)&v461[(unsigned int)v462];
    if ( (__int64 *)v211 != v461 )
    {
      v423 = v204;
      v316 = &v461[(unsigned int)v462];
      do
      {
        v317 = *v315++;
        p_s = v373;
        v465 = v317 | 4;
        sub_1A51800((__int64)&v451, v33, v211, v201, v202, v203);
      }
      while ( v316 != v315 );
      v204 = v423;
    }
    v212 = *(unsigned __int64 **)(a2 + 32);
    if ( v205 != v204 && (unsigned __int64 *)v205 != v212 )
    {
LABEL_305:
      if ( (char *)v205 != v373 + 40 )
        sub_157EA80(v205, (__int64)(v373 + 40), v204, (__int64)v212);
      if ( (unsigned __int64 *)v205 != v212 && v212 != (unsigned __int64 *)v204 )
      {
        v211 = *v212 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v212;
        *v212 = *v212 & 7 | *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        v213 = *((_QWORD *)v367 + 5);
        *(_QWORD *)(v211 + 8) = v205;
        *(_QWORD *)(a2 + 24) = v213 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(a2 + 24) & 7LL;
        *(_QWORD *)((v213 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v204;
        *((_QWORD *)v367 + 5) = v211 | *((_QWORD *)v367 + 5) & 7LL;
      }
      if ( v370 )
        goto LABEL_311;
    }
    if ( (*(_BYTE *)(v369 + 23) & 0x40) != 0 )
    {
      v318 = *(_QWORD **)(v369 - 8);
    }
    else
    {
      v211 = 24LL * (*(_DWORD *)(v369 + 20) & 0xFFFFFFF);
      v318 = (_QWORD *)(v369 - v211);
    }
    if ( v318[3] )
    {
      v319 = v318[4];
      v320 = v318[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v320 = v319;
      if ( v319 )
      {
        v211 = v320 | *(_QWORD *)(v319 + 16) & 3LL;
        *(_QWORD *)(v319 + 16) = v211;
      }
    }
    v318[3] = v371;
    if ( v371 )
    {
      v321 = *(_QWORD *)(v371 + 8);
      v318[4] = v321;
      if ( v321 )
        *(_QWORD *)(v321 + 16) = (unsigned __int64)(v318 + 4) | *(_QWORD *)(v321 + 16) & 3LL;
      v322 = v318[5];
      v323 = v318 + 3;
      v211 = (v371 + 8) | v322 & 3;
      v323[2] = v211;
      *(_QWORD *)(v371 + 8) = v323;
    }
    v324 = 0;
    v325 = (*(_DWORD *)(v369 + 20) & 0xFFFFFFFu) >> 1;
    v326 = v325 - 1;
    if ( v325 == 1 )
    {
LABEL_497:
      v345 = v461;
      v346 = &v461[(unsigned int)v462];
      if ( v346 == v461 )
        goto LABEL_312;
      while ( 1 )
      {
        if ( (v455 & 1) != 0 )
        {
          v347 = &v456;
          v348 = 3;
        }
        else
        {
          v352 = (unsigned int)v457;
          v347 = v456;
          v348 = (_DWORD)v457 - 1;
          if ( !(_DWORD)v457 )
            goto LABEL_504;
        }
        v211 = v348 & (((unsigned int)*v345 >> 9) ^ ((unsigned int)*v345 >> 4));
        v349 = &v347[2 * v211];
        v350 = *v349;
        if ( *v345 != *v349 )
          break;
LABEL_501:
        v351 = v349[1];
        ++v345;
        p_s = v367;
        v465 = v351 & 0xFFFFFFFFFFFFFFFBLL;
        sub_1A51800((__int64)&v451, v33, v211, (__int64)v347, v348, v203);
        if ( v346 == v345 )
          goto LABEL_312;
      }
      v354 = 1;
      while ( v350 != -8 )
      {
        v203 = v354 + 1;
        v357 = v348 & (unsigned int)(v211 + v354);
        v211 = (unsigned int)v357;
        v349 = &v347[2 * v357];
        v350 = *v349;
        if ( *v345 == *v349 )
          goto LABEL_501;
        v354 = v203;
      }
      if ( (v455 & 1) != 0 )
      {
        v353 = 8;
      }
      else
      {
        v352 = (unsigned int)v457;
LABEL_504:
        v353 = 2 * v352;
      }
      v349 = &v347[v353];
      goto LABEL_501;
    }
    while ( 1 )
    {
      v335 = 24;
      if ( (_DWORD)v324 != -2 )
        v335 = 24LL * (unsigned int)(2 * v324 + 3);
      if ( (*(_BYTE *)(v369 + 23) & 0x40) != 0 )
        v327 = *(_QWORD *)(v369 - 8);
      else
        v327 = v369 - 24LL * (*(_DWORD *)(v369 + 20) & 0xFFFFFFF);
      v328 = *(_QWORD *)(v327 + v335);
      v329 = (_QWORD *)(sub_13CF970(v369) + v335);
      if ( v424 == v328 )
      {
        sub_1593B40(v329, v371);
        goto LABEL_483;
      }
      if ( (v455 & 1) != 0 )
      {
        v330 = &v456;
        v331 = 3;
      }
      else
      {
        v342 = (unsigned int)v457;
        v330 = v456;
        v331 = (_DWORD)v457 - 1;
        if ( !(_DWORD)v457 )
          goto LABEL_490;
      }
      v332 = v331 & (((unsigned int)v328 >> 9) ^ ((unsigned int)v328 >> 4));
      v333 = &v330[2 * v332];
      v334 = *v333;
      if ( v328 != *v333 )
        break;
LABEL_482:
      sub_1593B40(v329, v333[1]);
LABEL_483:
      if ( v326 == ++v324 )
        goto LABEL_497;
    }
    v344 = 1;
    while ( v334 != -8 )
    {
      v355 = v344 + 1;
      v332 = v331 & (v344 + v332);
      v333 = &v330[2 * v332];
      v334 = *v333;
      if ( v328 == *v333 )
        goto LABEL_482;
      v344 = v355;
    }
    if ( (v455 & 1) != 0 )
    {
      v343 = 8;
    }
    else
    {
      v342 = (unsigned int)v457;
LABEL_490:
      v343 = 2 * v342;
    }
    v333 = &v330[v343];
    goto LABEL_482;
  }
  v206 = *v461 | 4;
  sub_157F2D0(*v461, (__int64)v373, 1);
  v465 = v206;
  p_s = v373;
  sub_1A51800((__int64)&v451, v33, v207, v208, v209, v210);
  v212 = *(unsigned __int64 **)(a2 + 32);
  if ( v205 != v204 && (unsigned __int64 *)v205 != v212 )
    goto LABEL_305;
LABEL_311:
  sub_1A522E0(v33, (__int64 *)&v454);
  v214 = *((_QWORD *)s + 1);
  sub_1593B40((_QWORD *)(v370 + -24 - 24LL * v377), v214);
  sub_1593B40((_QWORD *)(-24 - 24LL * (1 - v377) + v370), v371);
  v465 = v214 & 0xFFFFFFFFFFFFFFFBLL;
  p_s = v367;
  sub_1A51800((__int64)&v451, v33, v215, v216, v217, v218);
LABEL_312:
  v219 = v424;
  v220 = sub_1648A60(56, 1u);
  if ( v220 )
    sub_15F8590((__int64)v220, v219, (__int64)v373);
LABEL_314:
  sub_15DC140(a5, v451, (unsigned int)v452);
  p_s = (char *)&s;
  v385 = (unsigned __int64)v440;
  v465 = 0x1000000000LL;
  v221 = *(_QWORD *)(a1 + 32);
  v420 = &v440[(unsigned int)v441];
  v391 = &v437[(unsigned int)v438];
  v395 = *(_QWORD *)(a1 + 40);
  v443 = (const char **)v437;
  v444 = (__int64)v391;
  v445 = v221;
  v446 = v395;
  v387 = (__int64 *)v33;
  if ( v395 == v221 )
    goto LABEL_345;
  do
  {
    while ( 2 )
    {
      v222 = &v447;
      v448 = 0;
      v450 = 0;
      v447 = sub_1A4EBF0;
      v223 = &v447;
      v224 = &v443;
      v449 = sub_1A4EC10;
      v225 = sub_1A4EBF0;
      if ( ((unsigned __int8)sub_1A4EBF0 & 1) == 0 )
        goto LABEL_317;
      while ( 1 )
      {
        v225 = *(__int64 (__fastcall **)(__int64))((char *)v225 + (_QWORD)*v224 - 1);
LABEL_317:
        v226 = (__int64 (__fastcall **)(__int64))v225((__int64)v224);
        if ( v226 )
          break;
        while ( 1 )
        {
          v227 = v223[3];
          v225 = v223[2];
          v222 += 2;
          v223 = v222;
          v224 = (const char ***)((char *)&v443 + (_QWORD)v227);
          if ( ((unsigned __int8)v225 & 1) != 0 )
            break;
          v226 = (__int64 (__fastcall **)(__int64))v225((__int64)v224);
          if ( v226 )
            goto LABEL_320;
        }
      }
LABEL_320:
      v228 = (unsigned __int64 *)&v447;
      v412 = *v226;
      for ( m = (__int64 *)v385; v420 != m; LODWORD(v465) = v465 + 1 )
      {
        while ( 1 )
        {
          v234 = *m;
          v433 = v412;
          sub_1A51850(v228, v234, (__int64 *)&v433);
          v235 = (__int64)v449;
          if ( v449 )
            break;
LABEL_325:
          if ( v420 == ++m )
            goto LABEL_339;
        }
        if ( v449 != (__int64 (__fastcall *)(__int64 *))-16LL && v449 != (__int64 (__fastcall *)(__int64 *))-8LL )
          sub_1649B30(v228);
        v236 = *(unsigned int *)(a5 + 48);
        if ( !(_DWORD)v236 )
          goto LABEL_331;
        v230 = *(_QWORD *)(a5 + 32);
        v231 = (v236 - 1) & (((unsigned int)v235 >> 9) ^ ((unsigned int)v235 >> 4));
        v232 = (__int64 *)(v230 + 16LL * v231);
        v233 = *v232;
        if ( v235 != *v232 )
        {
          v307 = 1;
          if ( v233 == -8 )
            goto LABEL_331;
          while ( 1 )
          {
            v308 = v307 + 1;
            v231 = (v236 - 1) & (v231 + v307);
            v232 = (__int64 *)(v230 + 16LL * v231);
            v309 = *v232;
            if ( v235 == *v232 )
              break;
            v307 = v308;
            if ( v309 == -8 )
              goto LABEL_331;
          }
        }
        if ( v232 != (__int64 *)(v230 + 16 * v236) && v232[1] )
          goto LABEL_325;
LABEL_331:
        v237 = sub_157EBA0(v235);
        if ( v237 )
        {
          v403 = sub_15F4D60(v237);
          v240 = sub_157EBA0(v235);
          if ( v403 )
          {
            v383 = m;
            v381 = v228;
            v241 = 0;
            v242 = v240;
            do
            {
              v243 = v241++;
              v244 = sub_15F4DF0(v242, v243);
              sub_157F2D0(v244, v235, 0);
            }
            while ( v241 != v403 );
            m = v383;
            v228 = v381;
          }
        }
        v245 = (unsigned int)v465;
        if ( (unsigned int)v465 >= HIDWORD(v465) )
        {
          sub_16CD150((__int64)v387, &s, 0, 8, v238, v239);
          v245 = (unsigned int)v465;
        }
        ++m;
        *(_QWORD *)&p_s[8 * v245] = v235;
      }
LABEL_339:
      v246 = &v447;
      v448 = 0;
      v450 = 0;
      v447 = sub_1A4EB90;
      v247 = &v447;
      v248 = &v443;
      v449 = sub_1A4EBC0;
      v249 = sub_1A4EB90;
      if ( ((unsigned __int8)sub_1A4EB90 & 1) == 0 )
        goto LABEL_341;
      while ( 1 )
      {
        v249 = *(__int64 (__fastcall **)(__int64))((char *)v249 + (_QWORD)*v248 - 1);
LABEL_341:
        if ( (unsigned __int8)v249((__int64)v248) )
          break;
        while ( 1 )
        {
          v250 = v247[3];
          v249 = v247[2];
          v246 += 2;
          v247 = v246;
          v248 = (const char ***)((char *)&v443 + (_QWORD)v250);
          if ( ((unsigned __int8)v249 & 1) != 0 )
            break;
          if ( (unsigned __int8)v249((__int64)v248) )
            goto LABEL_344;
        }
      }
LABEL_344:
      if ( v395 != v445 )
        continue;
      break;
    }
LABEL_345:
    ;
  }
  while ( v395 != v446 || v391 != (char **)v443 || v391 != (char **)v444 );
  v251 = (__int64 *)p_s;
  v252 = (__int64)v387;
  v253 = (void **)&p_s[8 * (unsigned int)v465];
  if ( p_s != (char *)v253 )
  {
    do
    {
      v254 = *v251++;
      sub_157EE90(v254);
    }
    while ( v253 != (void **)v251 );
    v255 = (__int64 *)p_s;
    v253 = (void **)&p_s[8 * (unsigned int)v465];
    if ( p_s != (char *)v253 )
    {
      do
      {
        v256 = *v255++;
        sub_157F980(v256);
      }
      while ( v253 != (void **)v255 );
      v253 = (void **)p_s;
    }
  }
  if ( v253 != &s )
    _libc_free((unsigned __int64)v253);
  v257 = v440;
  v443 = (const char **)&v445;
  v444 = 0x400000000LL;
  v258 = &v440[(unsigned int)v441];
  if ( v440 != v258 )
  {
    do
    {
      v259 = *v257++;
      sub_1A57210(a1, v437, (unsigned int)v438, v259, a6, (__int64)&v443);
    }
    while ( v258 != v257 );
  }
  sub_1A52980((_QWORD *)a1, (__int64)&v437, a5, a6);
  v447 = (__int64 (__fastcall *)(__int64))&v449;
  v448 = 0x400000000LL;
  v404 = sub_1A593E0(a1, (void ***)v437, (unsigned int)v438, a6, (__int64)&v447);
  if ( v370 )
  {
    sub_1A522E0(v387, (__int64 *)&v454);
    v396 = *((_QWORD *)s + 1);
    v260 = (__int64 *)sub_16498A0(v370);
    if ( v379 )
    {
      v413 = sub_159C4F0(v260);
      v261 = (__int64 *)sub_16498A0(v370);
      v262 = sub_159C540(v261);
    }
    else
    {
      v413 = sub_159C540(v260);
      v358 = (__int64 *)sub_16498A0(v370);
      v262 = sub_159C4F0(v358);
    }
    v392 = &a3[a4];
    if ( a3 != v392 )
    {
      v421 = a3;
      while ( 1 )
      {
        v263 = *(__int64 **)(*v421 + 8LL);
        if ( v263 )
          break;
LABEL_382:
        if ( v392 == ++v421 )
        {
          v252 = (__int64)v387;
          goto LABEL_384;
        }
      }
      while ( 1 )
      {
        v267 = v263;
        v263 = (__int64 *)v263[1];
        v268 = sub_1648700((__int64)v267);
        v269 = v268;
        if ( *((_BYTE *)v268 + 16) <= 0x17u )
          goto LABEL_371;
        if ( sub_15CC8F0(a5, v371, v268[5]) )
        {
          if ( *v267 )
          {
            v264 = v267[1];
            v265 = v267[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v265 = v264;
            if ( v264 )
              *(_QWORD *)(v264 + 16) = v265 | *(_QWORD *)(v264 + 16) & 3LL;
          }
          *v267 = v262;
          if ( v262 )
          {
            v266 = *(_QWORD *)(v262 + 8);
            v267[1] = v266;
            if ( v266 )
              *(_QWORD *)(v266 + 16) = (unsigned __int64)(v267 + 1) | *(_QWORD *)(v266 + 16) & 3LL;
            v267[2] = (v262 + 8) | v267[2] & 3;
            *(_QWORD *)(v262 + 8) = v267;
          }
          goto LABEL_371;
        }
        if ( !sub_15CC8F0(a5, v396, v269[5]) )
          goto LABEL_371;
        if ( *v267 )
        {
          v270 = v267[1];
          v271 = v267[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v271 = v270;
          if ( v270 )
            *(_QWORD *)(v270 + 16) = v271 | *(_QWORD *)(v270 + 16) & 3LL;
        }
        *v267 = v413;
        if ( v413 )
        {
          v272 = *(_QWORD *)(v413 + 8);
          v267[1] = v272;
          if ( v272 )
            *(_QWORD *)(v272 + 16) = (unsigned __int64)(v267 + 1) | *(_QWORD *)(v272 + 16) & 3LL;
          v267[2] = (v413 + 8) | v267[2] & 3;
          *(_QWORD *)(v413 + 8) = v267;
          if ( !v263 )
            goto LABEL_382;
        }
        else
        {
LABEL_371:
          if ( !v263 )
            goto LABEL_382;
        }
      }
    }
  }
LABEL_384:
  v273 = (__int64 *)((char *)v447 + 8 * (unsigned int)v448);
  v274 = (__int64)v443;
  v433 = v447;
  v434 = v273;
  v435 = (__int64 (__fastcall *)(__int64 *))v443;
  v422 = &v443[(unsigned int)v444];
  v436 = v422;
  if ( v443 == v422 )
    goto LABEL_398;
  do
  {
    do
    {
      v275 = sub_1A4ED30;
      v276 = v252;
      v465 = 0;
      v277 = v252;
      v278 = &v433;
      p_s = (char *)sub_1A4ED30;
      s = sub_1A4ED50;
      *(_QWORD *)v467 = 0;
      if ( ((unsigned __int8)sub_1A4ED30 & 1) == 0 )
        goto LABEL_387;
      while ( 1 )
      {
        v275 = *(__int64 (__fastcall **)(__int64))((char *)v275 + (_QWORD)*v278 - 1);
LABEL_387:
        v279 = (_QWORD *)v275((__int64)v278);
        if ( v279 )
          break;
        while ( 1 )
        {
          v280 = *(_QWORD *)(v277 + 24);
          v275 = *(__int64 (__fastcall **)(__int64))(v277 + 16);
          v276 += 16;
          v277 = v276;
          v278 = (__int64 (__fastcall **)(__int64))((char *)&v433 + v280);
          if ( ((unsigned __int8)v275 & 1) != 0 )
            break;
          v279 = (_QWORD *)v275((__int64)v278);
          if ( v279 )
            goto LABEL_390;
        }
      }
LABEL_390:
      v281 = (_QWORD *)*v279;
      sub_1AE4880(*v279, a5, a6, 0);
      sub_1B17100(v281, a5, a6, 1);
      v19 = *v281 == 0;
      v282 = v252;
      v283 = 0;
      if ( !v19 )
        v283 = v368;
      v284 = v252;
      v465 = 0;
      v285 = &v433;
      v368 = v283;
      v286 = sub_1A4ECD0;
      s = sub_1A4ED00;
      p_s = (char *)sub_1A4ECD0;
      *(_QWORD *)v467 = 0;
      if ( ((unsigned __int8)sub_1A4ECD0 & 1) != 0 )
        goto LABEL_393;
      while ( 2 )
      {
        if ( !(unsigned __int8)v286((__int64)v285) )
        {
          while ( 1 )
          {
            v287 = *(_QWORD *)(v284 + 24);
            v286 = *(__int64 (__fastcall **)(__int64))(v284 + 16);
            v282 += 16;
            v284 = v282;
            v285 = (__int64 (__fastcall **)(__int64))((char *)&v433 + v287);
            if ( ((unsigned __int8)v286 & 1) != 0 )
              break;
            if ( (unsigned __int8)v286((__int64)v285) )
              goto LABEL_397;
          }
LABEL_393:
          v286 = *(__int64 (__fastcall **)(__int64))((char *)v286 + (_QWORD)*v285 - 1);
          continue;
        }
        break;
      }
LABEL_397:
      v274 = (__int64)v435;
    }
    while ( (char *)v435 != (char *)v422 );
LABEL_398:
    ;
  }
  while ( (const char **)v274 != v436 || v273 != (__int64 *)v433 || v273 != v434 );
  if ( v404 && (sub_1AE4880(a1, a5, a6, 0), sub_1B17100(a1, a5, a6, 1), (v288 = *(_QWORD **)a1) == 0)
    || (v288 = (_QWORD *)v368, v368 != a1) )
  {
    for ( n = v363; n != v288; n = (_QWORD *)*n )
    {
      sub_1AE4880(n, a5, a6, 0);
      sub_1B17100(n, a5, a6, 1);
    }
  }
  p_s = (char *)&s;
  v465 = 0x400000000LL;
  v429 = (const char *)v447;
  v290 = (char *)v447 + 8 * (unsigned int)v448;
  v291 = (__int64)v443;
  v430 = v290;
  v431 = (__int64)v443;
  v414 = &v443[(unsigned int)v444];
  v432 = v414;
  if ( v443 == v414 )
    goto LABEL_419;
  do
  {
    while ( 2 )
    {
      v292 = sub_1A4ED30;
      v293 = &v433;
      v294 = &v429;
      v434 = 0;
      v433 = sub_1A4ED30;
      v435 = sub_1A4ED50;
      v295 = &v433;
      v436 = 0;
      if ( ((unsigned __int8)sub_1A4ED30 & 1) == 0 )
        goto LABEL_408;
      while ( 1 )
      {
        v292 = *(__int64 (__fastcall **)(__int64))((char *)v292 + (_QWORD)*v294 - 1);
LABEL_408:
        v296 = (__int64 *)v292((__int64)v294);
        if ( v296 )
          break;
        while ( 1 )
        {
          v301 = v295[3];
          v292 = v295[2];
          v293 += 2;
          v295 = v293;
          v294 = (const char **)((char *)&v429 + (_QWORD)v301);
          if ( ((unsigned __int8)v292 & 1) != 0 )
            break;
          v296 = (__int64 *)v292((__int64)v294);
          if ( v296 )
            goto LABEL_411;
        }
      }
LABEL_411:
      v428[0] = *v296;
      if ( v363 == *(_QWORD **)v428[0] )
        sub_1A51A10(v252, v428, v297, v298, v299, v300);
      v302 = sub_1A4ECD0;
      v434 = 0;
      v303 = &v433;
      v304 = &v429;
      v436 = 0;
      v433 = sub_1A4ECD0;
      v435 = sub_1A4ED00;
      v305 = &v433;
      if ( ((unsigned __int8)sub_1A4ECD0 & 1) != 0 )
        goto LABEL_414;
      while ( 2 )
      {
        if ( !(unsigned __int8)v302((__int64)v304) )
        {
          while ( 1 )
          {
            v306 = v305[3];
            v302 = v305[2];
            v303 += 2;
            v305 = v303;
            v304 = (const char **)((char *)&v429 + (_QWORD)v306);
            if ( ((unsigned __int8)v302 & 1) != 0 )
              break;
            if ( (unsigned __int8)v302((__int64)v304) )
              goto LABEL_418;
          }
LABEL_414:
          v302 = *(__int64 (__fastcall **)(__int64))((char *)v302 + (_QWORD)*v304 - 1);
          continue;
        }
        break;
      }
LABEL_418:
      v291 = v431;
      if ( (const char **)v431 != v414 )
        continue;
      break;
    }
LABEL_419:
    ;
  }
  while ( (const char **)v291 != v432 || v290 != v429 || v290 != v430 );
  a16(a17, v404, p_s, (unsigned int)v465);
  if ( p_s != (char *)&s )
    _libc_free((unsigned __int64)p_s);
  if ( (char *)v447 != (char *)&v449 )
    _libc_free((unsigned __int64)v447);
  if ( v443 != (const char **)&v445 )
    _libc_free((unsigned __int64)v443);
  if ( (v455 & 1) == 0 )
    j___libc_free_0(v456);
  sub_1A52020((unsigned __int64 *)&v440);
  if ( v451 != (__int64 *)v453 )
    _libc_free((unsigned __int64)v451);
  if ( (v470 & 1) == 0 )
    j___libc_free_0(v471);
LABEL_16:
  if ( v437 != (char **)v439 )
    _libc_free((unsigned __int64)v437);
  if ( v461 != (__int64 *)v463 )
    _libc_free((unsigned __int64)v461);
  if ( (v459 & 1) == 0 )
    j___libc_free_0(v460);
  return v380;
}
