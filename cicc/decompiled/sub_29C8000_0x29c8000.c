// Function: sub_29C8000
// Address: 0x29c8000
//
__int64 __fastcall sub_29C8000(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        void *a5,
        size_t a6,
        unsigned __int8 *a7,
        size_t a8,
        unsigned __int8 *a9,
        unsigned __int64 a10)
{
  __m128i *v10; // r12
  unsigned int v11; // esi
  __int64 v12; // rcx
  unsigned int v13; // edx
  __m128i **v14; // rax
  __m128i *v15; // r8
  __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 *v23; // r13
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  char v28; // al
  char v29; // dl
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // r12
  __m128i *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  _DWORD *v42; // rax
  __int64 v43; // rax
  _BYTE *v44; // rax
  unsigned __int8 v45; // dl
  unsigned __int8 v46; // dl
  __int64 *v47; // rax
  __int64 v48; // rax
  size_t v49; // rdx
  size_t v50; // r13
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r9
  __int64 v54; // rax
  const void *v55; // rsi
  __int64 v56; // r8
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  const void *v66; // rsi
  __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rax
  const void *v74; // rsi
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rax
  const void *v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  unsigned __int16 *v86; // rbx
  __int64 *v87; // r15
  char v88; // si
  __int64 v89; // r8
  int v90; // edx
  unsigned int *v91; // rax
  __int64 v92; // r9
  __int64 *v93; // rax
  __int64 v94; // r12
  __int64 v95; // rax
  __m128i *v96; // rdi
  unsigned __int64 v97; // rax
  __m128i si128; // xmm0
  const char *v99; // rax
  size_t v100; // rdx
  _WORD *v101; // rdi
  unsigned __int8 *v102; // rsi
  unsigned __int64 v103; // rax
  _BYTE *v104; // rdi
  unsigned __int64 v105; // rax
  char v106; // al
  unsigned __int64 v107; // r15
  char v108; // si
  __int64 v109; // rdi
  int v110; // ecx
  char *v111; // rax
  __int64 v112; // r10
  unsigned __int8 **v113; // rax
  __int64 v114; // r12
  __int64 v115; // rax
  __m128i *v116; // rdi
  unsigned __int64 v117; // rax
  __m128i v118; // xmm0
  __int64 v119; // rax
  unsigned __int8 v120; // dl
  __int64 v121; // rax
  __int64 v122; // rdi
  void *v123; // rax
  size_t v124; // rdx
  _WORD *v125; // rdi
  __int64 v126; // rax
  __int64 v127; // rax
  unsigned __int8 v128; // dl
  unsigned __int8 **v129; // rax
  unsigned __int8 *v130; // rax
  unsigned __int8 v131; // dl
  __int64 v132; // rax
  __int64 v133; // rdi
  void *v134; // rax
  size_t v135; // rdx
  _DWORD *v136; // rdi
  _WORD *v137; // rdi
  unsigned __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rcx
  __int64 v141; // r8
  __int64 v142; // r9
  __int64 v143; // rsi
  char v144; // dl
  unsigned int v145; // ecx
  unsigned __int64 *v146; // r15
  __int64 v147; // rdx
  __int64 v148; // rcx
  __int64 v149; // rcx
  __int64 v150; // r8
  __int64 v151; // r9
  __int64 v152; // rdx
  _DWORD *v153; // rax
  __int64 v154; // rax
  void *v155; // rdi
  __int64 v156; // r12
  size_t v157; // r13
  char *v158; // r15
  __int64 v159; // rax
  void *v160; // rdi
  __int64 v161; // r12
  __int16 *v162; // rax
  __int16 *v163; // rax
  __int16 *v164; // rax
  __int64 v165; // rax
  __int64 v166; // rdx
  __int64 v167; // rdx
  __int64 v168; // rcx
  __int64 v169; // r8
  __int64 v170; // r9
  __int64 *v171; // rax
  __int64 v172; // rax
  __int64 v173; // rdx
  __int64 v174; // rcx
  __int64 v175; // r8
  __int64 v176; // r9
  __int64 v177; // rax
  void *v178; // rax
  void *v179; // rsi
  __int64 v180; // rdx
  __int64 v181; // rdx
  __int64 v182; // rcx
  __int64 v183; // r8
  __int64 v184; // r9
  __int64 v185; // rax
  void *v186; // rax
  void *v187; // rsi
  __int64 v188; // rdx
  __int64 v189; // rdx
  __int64 v190; // rcx
  __int64 v191; // r8
  __int64 v192; // r9
  __int64 v193; // rax
  void *v194; // rax
  void *v195; // rsi
  __int64 v196; // rdx
  __int64 v197; // rdx
  __int64 v198; // rcx
  __int64 v199; // r8
  __int64 v200; // r9
  __int64 v201; // rax
  void *v202; // rax
  void *v203; // rsi
  __int64 v204; // rdx
  __int16 *v205; // r12
  unsigned __int64 v206; // r13
  unsigned __int16 *v207; // rdi
  __int64 *v208; // rbx
  __int64 *v209; // r12
  __int64 v210; // rax
  __int64 *v211; // rbx
  __int64 *v212; // r12
  __int64 v213; // rax
  __int64 v215; // rax
  __int64 v216; // rax
  __int64 v217; // rax
  unsigned __int8 v218; // dl
  char *v219; // rcx
  __int64 v220; // rdx
  __int64 v221; // r8
  __int64 v222; // rax
  unsigned __int8 v223; // dl
  unsigned __int8 **v224; // rax
  unsigned __int8 *v225; // rax
  unsigned __int8 v226; // dl
  char *v227; // rcx
  __int64 v228; // rdx
  __int64 v229; // r8
  __int16 *v230; // rax
  unsigned __int16 *v231; // r12
  __m128i *v232; // r8
  __int64 v233; // rax
  __int64 v234; // rax
  int v235; // eax
  int v236; // r11d
  unsigned __int8 v237; // al
  __int64 v238; // rdx
  __int64 v239; // rdx
  unsigned __int8 v240; // al
  __m128i **v241; // r13
  __int64 v242; // rax
  __m128i **j; // r15
  int v244; // eax
  void *v245; // rdi
  __int64 v246; // rax
  void *v247; // rdi
  __int64 v248; // r12
  __int64 v249; // r12
  unsigned __int8 *v250; // rax
  size_t v251; // rdx
  void *v252; // rdi
  unsigned __int64 v253; // rax
  __int64 v254; // rax
  _WORD *v255; // rdx
  _DWORD *v256; // rdx
  __int64 v257; // rax
  __int64 v258; // rax
  __int64 v259; // rax
  __int64 v260; // rax
  char *v261; // rax
  __int64 v262; // rdx
  __int16 *v263; // rax
  __m128i **v264; // rbx
  __m128i *v265; // r12
  int v266; // eax
  int v267; // r9d
  __int64 v268; // rax
  __int64 v269; // rax
  __int64 v270; // r12
  __m128i *v271; // rax
  __m128i v272; // xmm0
  __int64 v273; // rax
  _WORD *v274; // rdx
  __int64 v275; // r12
  void *v276; // rdi
  _BYTE *v277; // rax
  __int64 v278; // rax
  __int64 v279; // rax
  _WORD *v280; // rdx
  __int64 *v281; // rax
  __int64 v282; // rax
  __int64 v283; // rax
  char *v284; // rax
  unsigned __int64 v285; // rdx
  __int16 *v286; // rax
  __m128i **v287; // rbx
  __m128i *v288; // r12
  int v289; // r11d
  __int64 v290; // rax
  __int64 v291; // [rsp+8h] [rbp-4C8h]
  __int64 v292; // [rsp+8h] [rbp-4C8h]
  unsigned __int64 v293; // [rsp+10h] [rbp-4C0h]
  unsigned __int64 v294; // [rsp+10h] [rbp-4C0h]
  __m128i *v295; // [rsp+10h] [rbp-4C0h]
  size_t v296; // [rsp+10h] [rbp-4C0h]
  size_t v297; // [rsp+10h] [rbp-4C0h]
  __int64 v298; // [rsp+10h] [rbp-4C0h]
  char v300; // [rsp+38h] [rbp-498h]
  char v301; // [rsp+38h] [rbp-498h]
  __m128i *v305; // [rsp+70h] [rbp-460h]
  __int64 i; // [rsp+78h] [rbp-458h]
  __m128i *v308; // [rsp+98h] [rbp-438h]
  char v309; // [rsp+98h] [rbp-438h]
  size_t v310; // [rsp+98h] [rbp-438h]
  unsigned __int64 v311; // [rsp+98h] [rbp-438h]
  unsigned __int16 *v312; // [rsp+98h] [rbp-438h]
  size_t v313; // [rsp+98h] [rbp-438h]
  size_t v314; // [rsp+98h] [rbp-438h]
  unsigned __int16 *v315; // [rsp+98h] [rbp-438h]
  __int64 *v316; // [rsp+A0h] [rbp-430h]
  __int64 *v317; // [rsp+A0h] [rbp-430h]
  unsigned __int8 *v318; // [rsp+A0h] [rbp-430h]
  __int64 *v319; // [rsp+B0h] [rbp-420h]
  unsigned __int8 v320; // [rsp+B0h] [rbp-420h]
  void *v321; // [rsp+B8h] [rbp-418h]
  char *v322; // [rsp+B8h] [rbp-418h]
  unsigned int v323; // [rsp+C0h] [rbp-410h] BYREF
  __int64 (__fastcall **v324)(); // [rsp+C8h] [rbp-408h]
  __int64 v325; // [rsp+D0h] [rbp-400h] BYREF
  char v326; // [rsp+D8h] [rbp-3F8h]
  __int16 *v327; // [rsp+E0h] [rbp-3F0h] BYREF
  __int16 *v328; // [rsp+E8h] [rbp-3E8h]
  __int16 *v329; // [rsp+F0h] [rbp-3E0h]
  __int64 *v330; // [rsp+100h] [rbp-3D0h] BYREF
  __int64 v331; // [rsp+108h] [rbp-3C8h]
  __int64 v332; // [rsp+110h] [rbp-3C0h] BYREF
  unsigned int v333; // [rsp+118h] [rbp-3B8h]
  __int64 v334; // [rsp+130h] [rbp-3A0h]
  char *v335; // [rsp+138h] [rbp-398h]
  __int64 v336; // [rsp+140h] [rbp-390h]
  __int64 v337; // [rsp+148h] [rbp-388h]
  __int64 *v338; // [rsp+150h] [rbp-380h] BYREF
  __int64 v339; // [rsp+158h] [rbp-378h]
  __int64 v340; // [rsp+160h] [rbp-370h] BYREF
  void *v341; // [rsp+168h] [rbp-368h]
  __int64 v342; // [rsp+170h] [rbp-360h]
  __int64 v343; // [rsp+178h] [rbp-358h]
  __int64 *v344; // [rsp+180h] [rbp-350h] BYREF
  __int64 v345; // [rsp+188h] [rbp-348h]
  __int64 v346; // [rsp+190h] [rbp-340h] BYREF
  void *v347; // [rsp+198h] [rbp-338h]
  __int64 v348; // [rsp+1A0h] [rbp-330h]
  __int64 v349; // [rsp+1A8h] [rbp-328h]
  unsigned __int64 v350[2]; // [rsp+1B0h] [rbp-320h] BYREF
  __int64 v351; // [rsp+1C0h] [rbp-310h] BYREF
  void *v352; // [rsp+1C8h] [rbp-308h]
  __int64 v353; // [rsp+1D0h] [rbp-300h]
  __int64 v354; // [rsp+1D8h] [rbp-2F8h]
  unsigned __int64 v355[2]; // [rsp+1E0h] [rbp-2F0h] BYREF
  __int64 v356; // [rsp+1F0h] [rbp-2E0h] BYREF
  void *v357; // [rsp+1F8h] [rbp-2D8h]
  __int64 v358; // [rsp+200h] [rbp-2D0h]
  __int64 v359; // [rsp+208h] [rbp-2C8h]
  __int64 *v360; // [rsp+210h] [rbp-2C0h] BYREF
  __int64 v361; // [rsp+218h] [rbp-2B8h]
  __int64 v362; // [rsp+220h] [rbp-2B0h] BYREF
  void *v363; // [rsp+228h] [rbp-2A8h]
  __int64 v364; // [rsp+230h] [rbp-2A0h]
  __int64 v365; // [rsp+238h] [rbp-298h]
  __int64 *v366; // [rsp+240h] [rbp-290h] BYREF
  __int64 v367; // [rsp+248h] [rbp-288h]
  __int64 v368; // [rsp+250h] [rbp-280h] BYREF
  char *v369; // [rsp+258h] [rbp-278h]
  __int64 v370; // [rsp+260h] [rbp-270h]
  __int64 v371; // [rsp+268h] [rbp-268h]
  unsigned __int8 **v372; // [rsp+270h] [rbp-260h] BYREF
  __int64 v373; // [rsp+278h] [rbp-258h]
  unsigned __int8 *v374; // [rsp+280h] [rbp-250h] BYREF
  __int64 v375; // [rsp+288h] [rbp-248h]
  __int64 v376; // [rsp+290h] [rbp-240h]
  __int64 v377; // [rsp+298h] [rbp-238h]
  unsigned int v378; // [rsp+2A0h] [rbp-230h]
  __int64 v379; // [rsp+2E0h] [rbp-1F0h] BYREF
  void *src; // [rsp+2E8h] [rbp-1E8h]
  __int64 v381; // [rsp+2F0h] [rbp-1E0h]
  unsigned int v382; // [rsp+2F8h] [rbp-1D8h]
  __int64 *v383; // [rsp+300h] [rbp-1D0h] BYREF
  __int64 v384; // [rsp+308h] [rbp-1C8h]
  __int64 v385; // [rsp+310h] [rbp-1C0h] BYREF
  void *v386; // [rsp+318h] [rbp-1B8h]
  __int64 v387; // [rsp+320h] [rbp-1B0h]
  unsigned int v388; // [rsp+328h] [rbp-1A8h]
  __int64 *v389; // [rsp+330h] [rbp-1A0h] BYREF
  __int64 v390; // [rsp+338h] [rbp-198h]
  __int64 v391; // [rsp+340h] [rbp-190h] BYREF
  void *v392; // [rsp+348h] [rbp-188h]
  __int64 v393; // [rsp+350h] [rbp-180h]
  unsigned int v394; // [rsp+358h] [rbp-178h]
  __int64 *v395; // [rsp+360h] [rbp-170h] BYREF
  __int64 v396; // [rsp+368h] [rbp-168h]
  __int64 v397; // [rsp+370h] [rbp-160h] BYREF
  void *v398; // [rsp+378h] [rbp-158h]
  __int64 v399; // [rsp+380h] [rbp-150h]
  unsigned int v400; // [rsp+388h] [rbp-148h]
  __m128i **v401; // [rsp+390h] [rbp-140h] BYREF
  __int64 v402; // [rsp+398h] [rbp-138h]
  __m128i *v403; // [rsp+3A0h] [rbp-130h] BYREF
  size_t v404; // [rsp+3A8h] [rbp-128h] BYREF
  __int16 *v405; // [rsp+3B0h] [rbp-120h] BYREF
  __int16 *v406; // [rsp+3B8h] [rbp-118h] BYREF
  const char *v407; // [rsp+3C0h] [rbp-110h]
  __int64 v408; // [rsp+3C8h] [rbp-108h]
  __m128i *v409[3]; // [rsp+3E0h] [rbp-F0h] BYREF
  unsigned __int16 v410; // [rsp+3F8h] [rbp-D8h] BYREF
  char *v411; // [rsp+400h] [rbp-D0h]
  __int64 v412; // [rsp+408h] [rbp-C8h]
  __m128i *v413[3]; // [rsp+420h] [rbp-B0h] BYREF
  __int64 v414; // [rsp+438h] [rbp-98h] BYREF
  char *v415; // [rsp+440h] [rbp-90h]
  __int64 v416; // [rsp+448h] [rbp-88h]
  __m128i *v417[3]; // [rsp+460h] [rbp-70h] BYREF
  unsigned __int16 v418; // [rsp+478h] [rbp-58h] BYREF
  char *v419; // [rsp+480h] [rbp-50h]
  __int64 v420; // [rsp+488h] [rbp-48h]
  char v421; // [rsp+4A0h] [rbp-30h] BYREF

  if ( !sub_BA8DC0(a1, (__int64)"llvm.dbg.cu", 11) )
  {
    v281 = sub_29C0AE0();
    v282 = sub_A51340((__int64)v281, a5, a6);
    sub_904010(v282, ": Skipping module without debug info\n");
    return 0;
  }
  v379 = 0;
  v383 = &v385;
  v389 = &v391;
  v395 = &v397;
  src = 0;
  v381 = 0;
  v382 = 0;
  v384 = 0;
  v385 = 0;
  v386 = 0;
  v387 = 0;
  v388 = 0;
  v390 = 0;
  v391 = 0;
  v392 = 0;
  v393 = 0;
  v394 = 0;
  v396 = 0;
  v397 = 0;
  v398 = 0;
  v399 = 0;
  v400 = 0;
  v401 = &v403;
  v402 = 0;
  for ( i = a2; a3 != i; i = *(_QWORD *)(i + 8) )
  {
    v10 = (__m128i *)(i - 56);
    if ( !i )
      v10 = 0;
    if ( !sub_B2FC80((__int64)v10) && !sub_B2FC80((__int64)v10) && !(unsigned __int8)sub_B2FC00(v10) )
    {
      v11 = *(_DWORD *)(a4 + 24);
      v12 = *(_QWORD *)(a4 + 8);
      if ( v11 )
      {
        v13 = (v11 - 1) & (((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9));
        v14 = (__m128i **)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v10 == *v14 )
        {
LABEL_12:
          if ( v14 != (__m128i **)(v12 + 16LL * v11) )
          {
            v16 = sub_B92180((__int64)v10);
            v403 = v10;
            v404 = v16;
            sub_29C6AD0((__int64)&v379, (__int64 *)&v403, (__int64 *)&v404, v17, v18, v19);
            if ( v16 )
            {
              v237 = *(_BYTE *)(v16 - 16);
              v238 = (v237 & 2) != 0 ? *(_QWORD *)(v16 - 32) : v16 - 16 - 8LL * ((v237 >> 2) & 0xF);
              v239 = *(_QWORD *)(v238 + 56);
              if ( v239 )
              {
                v240 = *(_BYTE *)(v239 - 16);
                if ( (v240 & 2) != 0 )
                {
                  v241 = *(__m128i ***)(v239 - 32);
                  v242 = *(unsigned int *)(v239 - 24);
                }
                else
                {
                  v241 = (__m128i **)(v239 - 16 - 8LL * ((v240 >> 2) & 0xF));
                  v242 = (*(_WORD *)(v239 - 16) >> 6) & 0xF;
                }
                for ( j = &v241[v242]; j != v241; ++v241 )
                {
                  if ( !*v241 )
                    BUG();
                  if ( (*v241)->m128i_i8[0] == 26 )
                  {
                    v403 = *v241;
                    *(_DWORD *)sub_29C5270((__int64)&v397, (__int64 *)&v403, v239, v20, v21, v22) = 0;
                  }
                }
              }
            }
            v305 = (__m128i *)((char *)v10 + 72);
            v308 = (__m128i *)v10[5].m128i_i64[0];
            if ( &v10[4].m128i_u64[1] != (unsigned __int64 *)v308 )
            {
              while ( 1 )
              {
                if ( !v308 )
                  BUG();
                v23 = (unsigned __int64 *)v308[2].m128i_i64[0];
                v319 = &v308[1].m128i_i64[1];
                if ( &v308[1].m128i_u64[1] != v23 )
                  break;
LABEL_50:
                v308 = (__m128i *)v308->m128i_i64[1];
                if ( v305 == v308 )
                  goto LABEL_4;
              }
              while ( 1 )
              {
                if ( !v23 )
                  BUG();
                v28 = *((_BYTE *)v23 - 24);
                v29 = v28;
                if ( v28 == 84 )
                  goto LABEL_20;
                if ( (int)qword_5008C88 <= 0 )
                  goto LABEL_18;
                v30 = v23[5];
                if ( v30 )
                {
                  v31 = sub_B14240(v30);
                  v33 = v32;
                  v34 = v31;
                  if ( v31 != v32 )
                  {
                    while ( *(_BYTE *)(v34 + 32) )
                    {
                      v34 = *(_QWORD *)(v34 + 8);
                      if ( v34 == v32 )
                        goto LABEL_39;
                    }
LABEL_29:
                    if ( v34 != v33 )
                    {
                      if ( v16 )
                      {
                        v35 = *(__m128i **)(v34 + 24);
                        v403 = v35;
                        if ( v35 )
                          sub_B96E90((__int64)&v403, (__int64)v35, 1);
                        v36 = sub_B10D40((__int64)&v403);
                        if ( v403 )
                        {
                          v321 = (void *)v36;
                          sub_B91220((__int64)&v403, (__int64)v403);
                          v36 = (__int64)v321;
                        }
                        if ( !v36 && !sub_B12EE0(v34) )
                        {
                          v403 = (__m128i *)sub_B12000(v34 + 72);
                          v42 = (_DWORD *)sub_29C5270((__int64)&v397, (__int64 *)&v403, v38, v39, v40, v41);
                          ++*v42;
                        }
                      }
                      while ( 1 )
                      {
                        v34 = *(_QWORD *)(v34 + 8);
                        if ( v34 == v33 )
                          break;
                        if ( !*(_BYTE *)(v34 + 32) )
                          goto LABEL_29;
                      }
                    }
                  }
LABEL_39:
                  v28 = *((_BYTE *)v23 - 24);
                }
                if ( v28 != 85 || (v37 = *(v23 - 7)) == 0 )
                {
LABEL_19:
                  v24 = sub_B10CD0((__int64)(v23 + 3));
                  v403 = (__m128i *)(v23 - 3);
                  LOBYTE(v404) = v24 != 0;
                  sub_29C6DE0((__int64)&v385, (__int64 *)&v403, &v404, v25, v26, v27);
LABEL_20:
                  v23 = (unsigned __int64 *)v23[1];
                  if ( v319 == (__int64 *)v23 )
                    goto LABEL_50;
                  continue;
                }
                if ( !*(_BYTE *)v37 && *(_QWORD *)(v37 + 24) == v23[7] && (*(_BYTE *)(v37 + 33) & 0x20) != 0 )
                {
                  v145 = *(_DWORD *)(v37 + 36);
                  if ( v145 > 0x45 )
                  {
                    if ( v145 != 71 )
                      goto LABEL_44;
LABEL_173:
                    if ( !v16 )
                      goto LABEL_176;
                    v146 = v23 - 3;
                    if ( sub_B10D40((__int64)(v23 + 3)) )
                      goto LABEL_175;
                    v147 = *((_DWORD *)v23 - 5) & 0x7FFFFFF;
                    v148 = *(_QWORD *)(v23[-4 * v147 - 3] + 24);
                    v374 = (unsigned __int8 *)v148;
                    if ( *(_BYTE *)v148 == 4 )
                    {
                      if ( !*(_DWORD *)(v148 + 144)
                        && !(unsigned __int8)sub_AF4500(*(_QWORD *)(v146[4 * (2 - v147)] + 24)) )
                      {
                        goto LABEL_175;
                      }
LABEL_182:
                      sub_B58DC0(&v403, &v374);
                      if ( sub_29C12C0((__int64 *)&v403) )
                        goto LABEL_175;
                      v152 = *((_DWORD *)v23 - 5) & 0x7FFFFFF;
                      v403 = *(__m128i **)(v146[4 * (1 - v152)] + 24);
                      v153 = (_DWORD *)sub_29C5270((__int64)&v397, (__int64 *)&v403, v152, v149, v150, v151);
                      ++*v153;
                      v29 = *((_BYTE *)v23 - 24);
LABEL_18:
                      if ( v29 != 85 )
                        goto LABEL_19;
                    }
                    else
                    {
                      if ( (unsigned __int8)(*(_BYTE *)v148 - 5) > 0x1Fu )
                        goto LABEL_182;
LABEL_175:
                      v28 = *((_BYTE *)v23 - 24);
LABEL_176:
                      if ( v28 != 85 )
                        goto LABEL_19;
                    }
                    v37 = *(v23 - 7);
                    goto LABEL_44;
                  }
                  if ( v145 > 0x43 )
                    goto LABEL_173;
                }
LABEL_44:
                if ( !v37
                  || *(_BYTE *)v37
                  || *(_QWORD *)(v37 + 24) != v23[7]
                  || (*(_BYTE *)(v37 + 33) & 0x20) == 0
                  || (unsigned int)(*(_DWORD *)(v37 + 36) - 68) > 3 )
                {
                  goto LABEL_19;
                }
                v23 = (unsigned __int64 *)v23[1];
                if ( v319 == (__int64 *)v23 )
                  goto LABEL_50;
              }
            }
          }
        }
        else
        {
          v266 = 1;
          while ( v15 != (__m128i *)-4096LL )
          {
            v267 = v266 + 1;
            v13 = (v11 - 1) & (v266 + v13);
            v14 = (__m128i **)(v12 + 16LL * v13);
            v15 = *v14;
            if ( v10 == *v14 )
              goto LABEL_12;
            v266 = v267;
          }
        }
      }
    }
LABEL_4:
    ;
  }
  v43 = sub_BA8DC0(a1, (__int64)"llvm.dbg.cu", 11);
  v44 = (_BYTE *)sub_B91A10(v43, 0);
  if ( *v44 == 16 )
  {
LABEL_57:
    v46 = *(v44 - 16);
    if ( (v46 & 2) != 0 )
      v47 = (__int64 *)*((_QWORD *)v44 - 4);
    else
      v47 = (__int64 *)&v44[-8 * ((v46 >> 2) & 0xF) - 16];
    v48 = *v47;
    v322 = (char *)v48;
    if ( v48 )
    {
      v322 = (char *)sub_B91420(v48);
      v50 = v49;
    }
    else
    {
      v50 = 0;
    }
    goto LABEL_61;
  }
  v45 = *(v44 - 16);
  if ( (v45 & 2) != 0 )
  {
    v44 = (_BYTE *)**((_QWORD **)v44 - 4);
    if ( v44 )
      goto LABEL_57;
  }
  else
  {
    v44 = *(_BYTE **)&v44[-8 * ((v45 >> 2) & 0xF) - 16];
    if ( v44 )
      goto LABEL_57;
  }
  v50 = 0;
  v322 = (char *)byte_3F871B3;
LABEL_61:
  v334 = 0;
  v335 = 0;
  v336 = 0;
  v337 = 0;
  sub_C7D6A0(0, 0, 8);
  v54 = *(unsigned int *)(a4 + 24);
  LODWORD(v337) = v54;
  if ( (_DWORD)v54 )
  {
    v335 = (char *)sub_C7D670(16 * v54, 8);
    v55 = *(const void **)(a4 + 8);
    v336 = *(_QWORD *)(a4 + 16);
    memcpy(v335, v55, 16LL * (unsigned int)v337);
  }
  else
  {
    v335 = 0;
    v336 = 0;
  }
  v339 = 0;
  v338 = &v340;
  v56 = *(unsigned int *)(a4 + 40);
  if ( (_DWORD)v56 )
    sub_29C1190((__int64)&v338, a4 + 32, v51, v52, v56, v53);
  v340 = 0;
  v341 = 0;
  v342 = 0;
  v343 = 0;
  sub_C7D6A0(0, 0, 8);
  LODWORD(v343) = v382;
  if ( v382 )
  {
    v341 = (void *)sub_C7D670(16LL * v382, 8);
    v342 = v381;
    memcpy(v341, src, 16LL * (unsigned int)v343);
  }
  else
  {
    v341 = 0;
    v342 = 0;
  }
  v345 = 0;
  v344 = &v346;
  if ( (_DWORD)v384 )
    sub_29C1190((__int64)&v344, (__int64)&v383, v57, v58, v59, v60);
  v346 = 0;
  v347 = 0;
  v348 = 0;
  v349 = 0;
  sub_C7D6A0(0, 0, 8);
  v65 = *(unsigned int *)(a4 + 72);
  LODWORD(v349) = v65;
  if ( (_DWORD)v65 )
  {
    v347 = (void *)sub_C7D670(16 * v65, 8);
    v66 = *(const void **)(a4 + 56);
    v348 = *(_QWORD *)(a4 + 64);
    memcpy(v347, v66, 16LL * (unsigned int)v349);
  }
  else
  {
    v347 = 0;
    v348 = 0;
  }
  v350[1] = 0;
  v350[0] = (unsigned __int64)&v351;
  if ( *(_DWORD *)(a4 + 88) )
    sub_29C1060((__int64)v350, a4 + 80, v61, v62, v63, v64);
  v351 = 0;
  v352 = 0;
  v353 = 0;
  v354 = 0;
  sub_C7D6A0(0, 0, 8);
  LODWORD(v354) = v388;
  if ( v388 )
  {
    v352 = (void *)sub_C7D670(16LL * v388, 8);
    v353 = v387;
    memcpy(v352, v386, 16LL * (unsigned int)v354);
  }
  else
  {
    v352 = 0;
    v353 = 0;
  }
  v355[1] = 0;
  v355[0] = (unsigned __int64)&v356;
  if ( (_DWORD)v390 )
    sub_29C1060((__int64)v355, (__int64)&v389, v67, (unsigned int)v390, v68, v69);
  v356 = 0;
  v357 = 0;
  v358 = 0;
  v359 = 0;
  sub_C7D6A0(0, 0, 8);
  v73 = *(unsigned int *)(a4 + 120);
  LODWORD(v359) = v73;
  if ( (_DWORD)v73 )
  {
    v357 = (void *)sub_C7D670(16 * v73, 8);
    v74 = *(const void **)(a4 + 104);
    v358 = *(_QWORD *)(a4 + 112);
    memcpy(v357, v74, 16LL * (unsigned int)v359);
  }
  else
  {
    v357 = 0;
    v358 = 0;
  }
  v361 = 0;
  v360 = &v362;
  v75 = *(unsigned int *)(a4 + 136);
  if ( (_DWORD)v75 )
    sub_29C2230((__int64)&v360, a4 + 128, v75, v70, v71, v72);
  v362 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  sub_C7D6A0(0, 0, 8);
  v80 = *(unsigned int *)(a4 + 168);
  LODWORD(v365) = v80;
  if ( (_DWORD)v80 )
  {
    v363 = (void *)sub_C7D670(16 * v80, 8);
    v81 = *(const void **)(a4 + 152);
    v364 = *(_QWORD *)(a4 + 160);
    memcpy(v363, v81, 16LL * (unsigned int)v365);
  }
  else
  {
    v363 = 0;
    v364 = 0;
  }
  v367 = 0;
  v366 = &v368;
  if ( *(_DWORD *)(a4 + 184) )
    sub_29C0F30((__int64)&v366, a4 + 176, v76, v77, v78, v79);
  v368 = 0;
  v369 = 0;
  v370 = 0;
  v371 = 0;
  sub_C7D6A0(0, 0, 8);
  LODWORD(v371) = v400;
  if ( v400 )
  {
    v369 = (char *)sub_C7D670(16LL * v400, 8);
    v370 = v399;
    memcpy(v369, v398, 16LL * (unsigned int)v371);
  }
  else
  {
    v369 = 0;
    v370 = 0;
  }
  v86 = (unsigned __int16 *)&v374;
  v373 = 0;
  v372 = &v374;
  if ( (_DWORD)v402 )
    sub_29C0F30((__int64)&v372, (__int64)&v401, v82, v83, v84, v85);
  v327 = 0;
  v328 = 0;
  v300 = a10 != 0;
  v329 = 0;
  v316 = &v344[2 * (unsigned int)v345];
  if ( v344 != v316 )
  {
    v87 = v344;
    v88 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( v87[1] )
          goto LABEL_91;
        v89 = *v87;
        if ( (_DWORD)v337 )
          break;
LABEL_316:
        if ( a10 )
        {
          sub_124B680(&v403, "metadata", 8u);
          LOWORD(v406) = 5;
          v407 = "DISubprogram";
          v408 = 12;
          if ( !(unsigned __int8)sub_C6A630("DISubprogram", 12, 0) )
          {
            sub_C6B0E0((__int64 *)&v330, (__int64)"DISubprogram", 0xCu);
            sub_125BB70((__int64)v86, (__int64)&v330);
            sub_C6BC50((unsigned __int16 *)&v406);
            sub_C6A4F0((__int64)&v406, v86);
            sub_C6BC50(v86);
            if ( v330 != &v332 )
              j_j___libc_free_0((unsigned __int64)v330);
          }
          sub_124B680(v409, "name", 4u);
          v284 = (char *)sub_BD5D20(*v87);
          sub_1255B60((__int64)&v410, v284, v285);
          sub_124B680(v413, "action", 6u);
          LOWORD(v414) = 5;
          v415 = "not-generate";
          v416 = 12;
          if ( !(unsigned __int8)sub_C6A630("not-generate", 12, 0) )
          {
            sub_C6B0E0((__int64 *)&v330, (__int64)"not-generate", 0xCu);
            sub_125BB70((__int64)v86, (__int64)&v330);
            sub_C6BC50((unsigned __int16 *)&v414);
            sub_C6A4F0((__int64)&v414, v86);
            sub_C6BC50(v86);
            if ( v330 != &v332 )
              j_j___libc_free_0((unsigned __int64)v330);
          }
          sub_125BF70((__int64)&v330, (__int64)&v403, 3);
          v330 = (__int64 *)((char *)v330 + 1);
          LOWORD(v374) = 7;
          v376 = v331;
          v375 = 1;
          v377 = v332;
          v331 = 0;
          v378 = v333;
          v286 = v328;
          v332 = 0;
          v333 = 0;
          if ( v328 == v329 )
          {
            sub_29C1F00((unsigned __int64 *)&v327, v328, (__int64)v86);
          }
          else
          {
            if ( v328 )
            {
              sub_C6A4F0((__int64)v328, v86);
              v286 = v328;
            }
            v328 = v286 + 20;
          }
          sub_C6BC50(v86);
          sub_C6B900((__int64)&v330);
          sub_C7D6A0(v331, (unsigned __int64)v333 << 6, 8);
          v315 = v86;
          v287 = v417;
          do
          {
            v287 -= 8;
            sub_C6BC50((unsigned __int16 *)v287 + 12);
            v288 = *v287;
            if ( *v287 )
            {
              if ( (__m128i *)v288->m128i_i64[0] != &v288[1] )
                j_j___libc_free_0(v288->m128i_i64[0]);
              j_j___libc_free_0((unsigned __int64)v288);
            }
          }
          while ( v287 != &v403 );
          v86 = v315;
          v88 = 0;
          goto LABEL_91;
        }
        if ( (_BYTE)qword_5008FC8 )
          v245 = sub_CB7330();
        else
          v245 = sub_CB72A0();
        v246 = sub_904010((__int64)v245, "ERROR: ");
        v247 = *(void **)(v246 + 32);
        v248 = v246;
        if ( *(_QWORD *)(v246 + 24) - (_QWORD)v247 < a8 )
        {
          v248 = sub_CB6200(v246, a7, a8);
        }
        else if ( a8 )
        {
          memcpy(v247, a7, a8);
          *(_QWORD *)(v248 + 32) += a8;
        }
        v249 = sub_904010(v248, " did not generate DISubprogram for ");
        v250 = (unsigned __int8 *)sub_BD5D20(*v87);
        v252 = *(void **)(v249 + 32);
        if ( *(_QWORD *)(v249 + 24) - (_QWORD)v252 < v251 )
        {
          v249 = sub_CB6200(v249, v250, v251);
        }
        else if ( v251 )
        {
          v313 = v251;
          memcpy(v252, v250, v251);
          *(_QWORD *)(v249 + 32) += v313;
        }
        v94 = sub_904010(v249, " from ");
        v253 = *(_QWORD *)(v94 + 24);
        v104 = *(_BYTE **)(v94 + 32);
        if ( v253 - (unsigned __int64)v104 >= v50 )
        {
          if ( v50 )
          {
            memcpy(v104, v322, v50);
            v253 = *(_QWORD *)(v94 + 24);
            v104 = (_BYTE *)(v50 + *(_QWORD *)(v94 + 32));
            *(_QWORD *)(v94 + 32) = v104;
          }
          if ( v253 > (unsigned __int64)v104 )
            goto LABEL_117;
LABEL_329:
          sub_CB5D20(v94, 10);
          v88 = 0;
          goto LABEL_91;
        }
        v283 = sub_CB6200(v94, (unsigned __int8 *)v322, v50);
        v104 = *(_BYTE **)(v283 + 32);
        v94 = v283;
        if ( *(_QWORD *)(v283 + 24) <= (unsigned __int64)v104 )
          goto LABEL_329;
LABEL_117:
        v88 = 0;
        v87 += 2;
        *(_QWORD *)(v94 + 32) = v104 + 1;
        *v104 = 10;
        if ( v316 == v87 )
        {
LABEL_118:
          v309 = v88;
          v106 = sub_29C3AB0(
                   (__int64)&v346,
                   (__int64)&v351,
                   (__int64)&v356,
                   a7,
                   a8,
                   v300,
                   (unsigned __int8 *)v322,
                   v50,
                   (unsigned __int64 *)&v327);
          goto LABEL_119;
        }
      }
      v90 = (v337 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
      v91 = (unsigned int *)&v335[16 * v90];
      v92 = *(_QWORD *)v91;
      if ( v89 != *(_QWORD *)v91 )
      {
        v244 = 1;
        while ( v92 != -4096 )
        {
          v289 = v244 + 1;
          v90 = (v337 - 1) & (v244 + v90);
          v91 = (unsigned int *)&v335[16 * v90];
          v92 = *(_QWORD *)v91;
          if ( v89 == *(_QWORD *)v91 )
            goto LABEL_95;
          v244 = v289;
        }
        goto LABEL_316;
      }
LABEL_95:
      if ( v91 == (unsigned int *)&v335[16 * (unsigned int)v337] )
        goto LABEL_316;
      v93 = &v338[2 * v91[2]];
      if ( v93 == &v338[2 * (unsigned int)v339] )
        goto LABEL_316;
      if ( v93[1] )
      {
        if ( a10 )
        {
          sub_124B680(&v403, "metadata", 8u);
          LOWORD(v406) = 5;
          v407 = "DISubprogram";
          v408 = 12;
          if ( !(unsigned __int8)sub_C6A630("DISubprogram", 12, 0) )
          {
            sub_C6B0E0((__int64 *)&v330, (__int64)"DISubprogram", 0xCu);
            sub_125BB70((__int64)v86, (__int64)&v330);
            sub_C6BC50((unsigned __int16 *)&v406);
            sub_C6A4F0((__int64)&v406, v86);
            sub_C6BC50(v86);
            if ( v330 != &v332 )
              j_j___libc_free_0((unsigned __int64)v330);
          }
          sub_124B680(v409, "name", 4u);
          v261 = (char *)sub_BD5D20(*v87);
          v410 = 5;
          v412 = v262;
          v298 = (__int64)v261;
          v311 = v262;
          v411 = v261;
          if ( !(unsigned __int8)sub_C6A630(v261, v262, 0) )
          {
            sub_C6B0E0((__int64 *)&v330, v298, v311);
            sub_125BB70((__int64)v86, (__int64)&v330);
            sub_C6BC50(&v410);
            sub_C6A4F0((__int64)&v410, v86);
            sub_C6BC50(v86);
            if ( v330 != &v332 )
              j_j___libc_free_0((unsigned __int64)v330);
          }
          sub_124B680(v413, "action", 6u);
          LOWORD(v414) = 5;
          v415 = "drop";
          v416 = 4;
          if ( !(unsigned __int8)sub_C6A630("drop", 4, 0) )
          {
            sub_C6B0E0((__int64 *)&v330, (__int64)"drop", 4u);
            sub_125BB70((__int64)v86, (__int64)&v330);
            sub_C6BC50((unsigned __int16 *)&v414);
            sub_C6A4F0((__int64)&v414, v86);
            sub_C6BC50(v86);
            if ( v330 != &v332 )
              j_j___libc_free_0((unsigned __int64)v330);
          }
          sub_125BF70((__int64)&v330, (__int64)&v403, 3);
          v330 = (__int64 *)((char *)v330 + 1);
          LOWORD(v374) = 7;
          v376 = v331;
          v375 = 1;
          v377 = v332;
          v331 = 0;
          v378 = v333;
          v263 = v328;
          v332 = 0;
          v333 = 0;
          if ( v328 == v329 )
          {
            sub_29C1F00((unsigned __int64 *)&v327, v328, (__int64)v86);
          }
          else
          {
            if ( v328 )
            {
              sub_C6A4F0((__int64)v328, v86);
              v263 = v328;
            }
            v328 = v263 + 20;
          }
          sub_C6BC50(v86);
          sub_C6B900((__int64)&v330);
          sub_C7D6A0(v331, (unsigned __int64)v333 << 6, 8);
          v312 = v86;
          v264 = v417;
          do
          {
            v264 -= 8;
            sub_C6BC50((unsigned __int16 *)v264 + 12);
            v265 = *v264;
            if ( *v264 )
            {
              if ( (__m128i *)v265->m128i_i64[0] != &v265[1] )
                j_j___libc_free_0(v265->m128i_i64[0]);
              j_j___libc_free_0((unsigned __int64)v265);
            }
          }
          while ( v264 != &v403 );
          v86 = v312;
          v88 = 0;
          goto LABEL_91;
        }
        if ( (_BYTE)qword_5008FC8 )
          v94 = (__int64)sub_CB7330();
        else
          v94 = (__int64)sub_CB72A0();
        v95 = *(_QWORD *)(v94 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v94 + 24) - v95) <= 6 )
        {
          v260 = sub_CB6200(v94, (unsigned __int8 *)"ERROR: ", 7u);
          v96 = *(__m128i **)(v260 + 32);
          v94 = v260;
        }
        else
        {
          *(_DWORD *)v95 = 1330795077;
          *(_WORD *)(v95 + 4) = 14930;
          *(_BYTE *)(v95 + 6) = 32;
          v96 = (__m128i *)(*(_QWORD *)(v94 + 32) + 7LL);
          *(_QWORD *)(v94 + 32) = v96;
        }
        v97 = *(_QWORD *)(v94 + 24) - (_QWORD)v96;
        if ( v97 < a8 )
        {
          v258 = sub_CB6200(v94, a7, a8);
          v96 = *(__m128i **)(v258 + 32);
          v94 = v258;
          if ( *(_QWORD *)(v258 + 24) - (_QWORD)v96 > 0x18u )
            goto LABEL_107;
        }
        else
        {
          if ( a8 )
          {
            memcpy(v96, a7, a8);
            v278 = *(_QWORD *)(v94 + 24);
            v96 = (__m128i *)(*(_QWORD *)(v94 + 32) + a8);
            *(_QWORD *)(v94 + 32) = v96;
            v97 = v278 - (_QWORD)v96;
          }
          if ( v97 > 0x18 )
          {
LABEL_107:
            si128 = _mm_load_si128((const __m128i *)&xmmword_439AB40);
            v96[1].m128i_i8[8] = 32;
            v96[1].m128i_i64[0] = 0x666F206D6172676FLL;
            *v96 = si128;
            *(_QWORD *)(v94 + 32) += 25LL;
            goto LABEL_108;
          }
        }
        v94 = sub_CB6200(v94, " dropped DISubprogram of ", 0x19u);
LABEL_108:
        v99 = sub_BD5D20(*v87);
        v101 = *(_WORD **)(v94 + 32);
        v102 = (unsigned __int8 *)v99;
        v103 = *(_QWORD *)(v94 + 24) - (_QWORD)v101;
        if ( v103 < v100 )
        {
          v269 = sub_CB6200(v94, v102, v100);
          v101 = *(_WORD **)(v269 + 32);
          v94 = v269;
          v103 = *(_QWORD *)(v269 + 24) - (_QWORD)v101;
        }
        else if ( v100 )
        {
          v314 = v100;
          memcpy(v101, v102, v100);
          v279 = *(_QWORD *)(v94 + 24);
          v280 = (_WORD *)(*(_QWORD *)(v94 + 32) + v314);
          *(_QWORD *)(v94 + 32) = v280;
          v101 = v280;
          v103 = v279 - (_QWORD)v280;
        }
        if ( v103 <= 5 )
        {
          v268 = sub_CB6200(v94, (unsigned __int8 *)" from ", 6u);
          v104 = *(_BYTE **)(v268 + 32);
          v94 = v268;
        }
        else
        {
          *(_DWORD *)v101 = 1869768224;
          v101[2] = 8301;
          v104 = (_BYTE *)(*(_QWORD *)(v94 + 32) + 6LL);
          *(_QWORD *)(v94 + 32) = v104;
        }
        v105 = *(_QWORD *)(v94 + 24);
        if ( v105 - (unsigned __int64)v104 < v50 )
        {
          v259 = sub_CB6200(v94, (unsigned __int8 *)v322, v50);
          v104 = *(_BYTE **)(v259 + 32);
          v94 = v259;
          v105 = *(_QWORD *)(v259 + 24);
        }
        else if ( v50 )
        {
          memcpy(v104, v322, v50);
          v105 = *(_QWORD *)(v94 + 24);
          v104 = (_BYTE *)(v50 + *(_QWORD *)(v94 + 32));
          *(_QWORD *)(v94 + 32) = v104;
        }
        if ( (unsigned __int64)v104 < v105 )
          goto LABEL_117;
        goto LABEL_329;
      }
LABEL_91:
      v87 += 2;
      if ( v316 == v87 )
        goto LABEL_118;
    }
  }
  v309 = 1;
  v106 = sub_29C3AB0(
           (__int64)&v346,
           (__int64)&v351,
           (__int64)&v356,
           a7,
           a8,
           v300,
           (unsigned __int8 *)v322,
           v50,
           (unsigned __int64 *)&v327);
LABEL_119:
  v301 = v106;
  v317 = &v366[2 * (unsigned int)v367];
  if ( v366 != v317 )
  {
    v107 = (unsigned __int64)v366;
    v108 = 1;
    while ( 1 )
    {
      v109 = *(_QWORD *)v107;
      if ( (_DWORD)v371 )
      {
        v110 = (v371 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
        v111 = &v369[16 * v110];
        v112 = *(_QWORD *)v111;
        if ( v109 != *(_QWORD *)v111 )
        {
          v235 = 1;
          while ( v112 != -4096 )
          {
            v236 = v235 + 1;
            v110 = (v371 - 1) & (v235 + v110);
            v111 = &v369[16 * v110];
            v112 = *(_QWORD *)v111;
            if ( v109 == *(_QWORD *)v111 )
              goto LABEL_123;
            v235 = v236;
          }
          goto LABEL_162;
        }
LABEL_123:
        if ( v111 != &v369[16 * (unsigned int)v371] )
        {
          v113 = &v372[2 * *((unsigned int *)v111 + 2)];
          if ( v113 != &v372[2 * (unsigned int)v373] && *((_DWORD *)v113 + 2) < *(_DWORD *)(v107 + 8) )
            break;
        }
      }
LABEL_162:
      v107 += 16LL;
      if ( v317 == (__int64 *)v107 )
        goto LABEL_163;
    }
    if ( !a10 )
    {
      if ( (_BYTE)qword_5008FC8 )
        v114 = (__int64)sub_CB7330();
      else
        v114 = (__int64)sub_CB72A0();
      v115 = *(_QWORD *)(v114 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v114 + 24) - v115) <= 8 )
      {
        v234 = sub_CB6200(v114, "WARNING: ", 9u);
        v116 = *(__m128i **)(v234 + 32);
        v114 = v234;
      }
      else
      {
        *(_BYTE *)(v115 + 8) = 32;
        *(_QWORD *)v115 = 0x3A474E494E524157LL;
        v116 = (__m128i *)(*(_QWORD *)(v114 + 32) + 9LL);
        *(_QWORD *)(v114 + 32) = v116;
      }
      v117 = *(_QWORD *)(v114 + 24) - (_QWORD)v116;
      if ( v117 < a8 )
      {
        v233 = sub_CB6200(v114, a7, a8);
        v116 = *(__m128i **)(v233 + 32);
        v114 = v233;
        v117 = *(_QWORD *)(v233 + 24) - (_QWORD)v116;
      }
      else if ( a8 )
      {
        memcpy(v116, a7, a8);
        v254 = *(_QWORD *)(v114 + 24);
        v116 = (__m128i *)(*(_QWORD *)(v114 + 32) + a8);
        *(_QWORD *)(v114 + 32) = v116;
        v117 = v254 - (_QWORD)v116;
      }
      if ( v117 <= 0x24 )
      {
        v114 = sub_CB6200(v114, " drops dbg.value()/dbg.declare() for ", 0x25u);
      }
      else
      {
        v118 = _mm_load_si128((const __m128i *)&xmmword_439AB50);
        v116[2].m128i_i32[0] = 1919903264;
        v116[2].m128i_i8[4] = 32;
        *v116 = v118;
        v116[1] = _mm_load_si128((const __m128i *)&xmmword_439AB60);
        *(_QWORD *)(v114 + 32) += 37LL;
      }
      v119 = *(_QWORD *)v107;
      v120 = *(_BYTE *)(*(_QWORD *)v107 - 16LL);
      if ( (v120 & 2) != 0 )
        v121 = *(_QWORD *)(v119 - 32);
      else
        v121 = v119 - 16 - 8LL * ((v120 >> 2) & 0xF);
      v122 = *(_QWORD *)(v121 + 8);
      if ( v122 )
      {
        v123 = (void *)sub_B91420(v122);
        v125 = *(_WORD **)(v114 + 32);
        if ( v124 <= *(_QWORD *)(v114 + 24) - (_QWORD)v125 )
        {
          if ( v124 )
          {
            v296 = v124;
            memcpy(v125, v123, v124);
            v255 = (_WORD *)(*(_QWORD *)(v114 + 32) + v296);
            *(_QWORD *)(v114 + 32) = v255;
            v125 = v255;
          }
LABEL_142:
          if ( *(_QWORD *)(v114 + 24) - (_QWORD)v125 <= 5u )
          {
            v114 = sub_CB6200(v114, (unsigned __int8 *)" from ", 6u);
            v126 = *(_QWORD *)(v114 + 32);
          }
          else
          {
            *(_DWORD *)v125 = 1869768224;
            v125[2] = 8301;
            v126 = *(_QWORD *)(v114 + 32) + 6LL;
            *(_QWORD *)(v114 + 32) = v126;
          }
          if ( (unsigned __int64)(*(_QWORD *)(v114 + 24) - v126) <= 8 )
          {
            v114 = sub_CB6200(v114, (unsigned __int8 *)"function ", 9u);
          }
          else
          {
            *(_BYTE *)(v126 + 8) = 32;
            *(_QWORD *)v126 = 0x6E6F6974636E7566LL;
            *(_QWORD *)(v114 + 32) += 9LL;
          }
          v127 = *(_QWORD *)v107;
          v128 = *(_BYTE *)(*(_QWORD *)v107 - 16LL);
          if ( (v128 & 2) != 0 )
            v129 = *(unsigned __int8 ***)(v127 - 32);
          else
            v129 = (unsigned __int8 **)(v127 - 16 - 8LL * ((v128 >> 2) & 0xF));
          v130 = sub_AF34D0(*v129);
          v131 = *(v130 - 16);
          if ( (v131 & 2) != 0 )
            v132 = *((_QWORD *)v130 - 4);
          else
            v132 = (__int64)&v130[-8 * ((v131 >> 2) & 0xF) - 16];
          v133 = *(_QWORD *)(v132 + 16);
          if ( v133 )
          {
            v134 = (void *)sub_B91420(v133);
            v136 = *(_DWORD **)(v114 + 32);
            if ( v135 <= *(_QWORD *)(v114 + 24) - (_QWORD)v136 )
            {
              if ( v135 )
              {
                v297 = v135;
                memcpy(v136, v134, v135);
                v256 = (_DWORD *)(*(_QWORD *)(v114 + 32) + v297);
                *(_QWORD *)(v114 + 32) = v256;
                v136 = v256;
              }
              goto LABEL_154;
            }
            v114 = sub_CB6200(v114, (unsigned __int8 *)v134, v135);
          }
          v136 = *(_DWORD **)(v114 + 32);
LABEL_154:
          if ( *(_QWORD *)(v114 + 24) - (_QWORD)v136 <= 6u )
          {
            v216 = sub_CB6200(v114, " (file ", 7u);
            v137 = *(_WORD **)(v216 + 32);
            v114 = v216;
          }
          else
          {
            *v136 = 1768302624;
            *((_WORD *)v136 + 2) = 25964;
            *((_BYTE *)v136 + 6) = 32;
            v137 = (_WORD *)(*(_QWORD *)(v114 + 32) + 7LL);
            *(_QWORD *)(v114 + 32) = v137;
          }
          v138 = *(_QWORD *)(v114 + 24) - (_QWORD)v137;
          if ( v138 < v50 )
          {
            v215 = sub_CB6200(v114, (unsigned __int8 *)v322, v50);
            v137 = *(_WORD **)(v215 + 32);
            v114 = v215;
            v138 = *(_QWORD *)(v215 + 24) - (_QWORD)v137;
          }
          else if ( v50 )
          {
            memcpy(v137, v322, v50);
            v257 = *(_QWORD *)(v114 + 24);
            v137 = (_WORD *)(v50 + *(_QWORD *)(v114 + 32));
            *(_QWORD *)(v114 + 32) = v137;
            v138 = v257 - (_QWORD)v137;
          }
          if ( v138 <= 1 )
          {
            sub_CB6200(v114, (unsigned __int8 *)")\n", 2u);
          }
          else
          {
            *v137 = 2601;
            *(_QWORD *)(v114 + 32) += 2LL;
          }
LABEL_161:
          v108 = 0;
          goto LABEL_162;
        }
        v114 = sub_CB6200(v114, (unsigned __int8 *)v123, v124);
      }
      v125 = *(_WORD **)(v114 + 32);
      goto LABEL_142;
    }
    sub_124B680(&v403, "metadata", 8u);
    LOWORD(v406) = 5;
    v407 = "dbg-var-intrinsic";
    v408 = 17;
    if ( !(unsigned __int8)sub_C6A630("dbg-var-intrinsic", 17, 0) )
    {
      sub_C6B0E0((__int64 *)&v330, (__int64)"dbg-var-intrinsic", 0x11u);
      sub_125BB70((__int64)v86, (__int64)&v330);
      sub_C6BC50((unsigned __int16 *)&v406);
      sub_C6A4F0((__int64)&v406, v86);
      sub_C6BC50(v86);
      if ( v330 != &v332 )
        j_j___libc_free_0((unsigned __int64)v330);
    }
    sub_124B680(v409, "name", 4u);
    v217 = *(_QWORD *)v107;
    v218 = *(_BYTE *)(*(_QWORD *)v107 - 16LL);
    if ( (v218 & 2) != 0 )
    {
      v219 = *(char **)(*(_QWORD *)(v217 - 32) + 8LL);
      if ( v219 )
      {
LABEL_273:
        v219 = (char *)sub_B91420((__int64)v219);
        v221 = v220;
LABEL_274:
        v411 = v219;
        v412 = v221;
        v291 = (__int64)v219;
        v293 = v221;
        v410 = 5;
        if ( !(unsigned __int8)sub_C6A630(v219, v221, 0) )
        {
          sub_C6B0E0((__int64 *)&v330, v291, v293);
          sub_125BB70((__int64)v86, (__int64)&v330);
          sub_C6BC50(&v410);
          sub_C6A4F0((__int64)&v410, v86);
          sub_C6BC50(v86);
          if ( v330 != &v332 )
            j_j___libc_free_0((unsigned __int64)v330);
        }
        sub_124B680(v413, "fn-name", 7u);
        v222 = *(_QWORD *)v107;
        v223 = *(_BYTE *)(*(_QWORD *)v107 - 16LL);
        if ( (v223 & 2) != 0 )
          v224 = *(unsigned __int8 ***)(v222 - 32);
        else
          v224 = (unsigned __int8 **)(v222 - 16 - 8LL * ((v223 >> 2) & 0xF));
        v225 = sub_AF34D0(*v224);
        v226 = *(v225 - 16);
        if ( (v226 & 2) != 0 )
        {
          v227 = *(char **)(*((_QWORD *)v225 - 4) + 16LL);
          if ( v227 )
          {
LABEL_279:
            v227 = (char *)sub_B91420((__int64)v227);
            v229 = v228;
            goto LABEL_280;
          }
        }
        else
        {
          v227 = *(char **)&v225[-8 * ((v226 >> 2) & 0xF)];
          if ( v227 )
            goto LABEL_279;
        }
        v229 = 0;
LABEL_280:
        v415 = v227;
        v416 = v229;
        v292 = (__int64)v227;
        v294 = v229;
        LOWORD(v414) = 5;
        if ( !(unsigned __int8)sub_C6A630(v227, v229, 0) )
        {
          sub_C6B0E0((__int64 *)&v330, v292, v294);
          sub_125BB70((__int64)v86, (__int64)&v330);
          sub_C6BC50((unsigned __int16 *)&v414);
          sub_C6A4F0((__int64)&v414, v86);
          sub_C6BC50(v86);
          if ( v330 != &v332 )
            j_j___libc_free_0((unsigned __int64)v330);
        }
        sub_124B680(v417, "action", 6u);
        v418 = 5;
        v419 = "drop";
        v420 = 4;
        if ( !(unsigned __int8)sub_C6A630("drop", 4, 0) )
        {
          sub_C6B0E0((__int64 *)&v330, (__int64)"drop", 4u);
          sub_125BB70((__int64)v86, (__int64)&v330);
          sub_C6BC50(&v418);
          sub_C6A4F0((__int64)&v418, v86);
          sub_C6BC50(v86);
          if ( v330 != &v332 )
            j_j___libc_free_0((unsigned __int64)v330);
        }
        sub_125BF70((__int64)&v330, (__int64)&v403, 4);
        v330 = (__int64 *)((char *)v330 + 1);
        LOWORD(v374) = 7;
        v376 = v331;
        v375 = 1;
        v377 = v332;
        v331 = 0;
        v378 = v333;
        v230 = v328;
        v332 = 0;
        v333 = 0;
        if ( v328 == v329 )
        {
          sub_29C1F00((unsigned __int64 *)&v327, v328, (__int64)v86);
        }
        else
        {
          if ( v328 )
          {
            sub_C6A4F0((__int64)v328, v86);
            v230 = v328;
          }
          v328 = v230 + 20;
        }
        sub_C6BC50(v86);
        v231 = (unsigned __int16 *)&v421;
        sub_C6B900((__int64)&v330);
        sub_C7D6A0(v331, (unsigned __int64)v333 << 6, 8);
        do
        {
          v231 -= 32;
          sub_C6BC50(v231 + 12);
          v232 = *(__m128i **)v231;
          if ( *(_QWORD *)v231 )
          {
            if ( (__m128i *)v232->m128i_i64[0] != &v232[1] )
            {
              v295 = *(__m128i **)v231;
              j_j___libc_free_0(v232->m128i_i64[0]);
              v232 = v295;
            }
            j_j___libc_free_0((unsigned __int64)v232);
          }
        }
        while ( v231 != (unsigned __int16 *)&v403 );
        goto LABEL_161;
      }
    }
    else
    {
      v219 = *(char **)(v217 - 16 - 8LL * ((v218 >> 2) & 0xF) + 8);
      if ( v219 )
        goto LABEL_273;
    }
    v221 = 0;
    goto LABEL_274;
  }
  v108 = 1;
LABEL_163:
  v320 = v108 & v309 & v301;
  if ( a8 )
  {
    v318 = a7;
    v310 = a8;
  }
  else
  {
    v318 = (unsigned __int8 *)a5;
    v310 = a6;
  }
  if ( a10 && v328 != v327 )
  {
    v323 = 0;
    v324 = sub_2241E40();
    sub_CB7060((__int64)v86, a9, a10, (__int64)&v323, 7u);
    if ( v323 )
    {
      v270 = (__int64)sub_CB72A0();
      v271 = *(__m128i **)(v270 + 32);
      if ( *(_QWORD *)(v270 + 24) - (_QWORD)v271 <= 0x14u )
      {
        v270 = sub_CB6200(v270, "Could not open file: ", 0x15u);
      }
      else
      {
        v272 = _mm_load_si128((const __m128i *)&xmmword_439AB70);
        v271[1].m128i_i32[0] = 979725417;
        v271[1].m128i_i8[4] = 32;
        *v271 = v272;
        *(_QWORD *)(v270 + 32) += 21LL;
      }
      (*((void (__fastcall **)(__m128i **, __int64 (__fastcall **)(), _QWORD))*v324 + 4))(&v403, v324, v323);
      v273 = sub_CB6200(v270, (unsigned __int8 *)v403, v404);
      v274 = *(_WORD **)(v273 + 32);
      v275 = v273;
      if ( *(_QWORD *)(v273 + 24) - (_QWORD)v274 <= 1u )
      {
        v290 = sub_CB6200(v273, (unsigned __int8 *)", ", 2u);
        v276 = *(void **)(v290 + 32);
        v275 = v290;
      }
      else
      {
        *v274 = 8236;
        v276 = (void *)(*(_QWORD *)(v273 + 32) + 2LL);
        *(_QWORD *)(v273 + 32) = v276;
      }
      v143 = (__int64)a9;
      if ( *(_QWORD *)(v275 + 24) - (_QWORD)v276 < a10 )
      {
        v275 = sub_CB6200(v275, a9, a10);
        v277 = *(_BYTE **)(v275 + 32);
      }
      else
      {
        memcpy(v276, a9, a10);
        v277 = (_BYTE *)(a10 + *(_QWORD *)(v275 + 32));
        *(_QWORD *)(v275 + 32) = v277;
      }
      if ( (unsigned __int64)v277 >= *(_QWORD *)(v275 + 24) )
      {
        v143 = 10;
        sub_CB5D20(v275, 10);
      }
      else
      {
        *(_QWORD *)(v275 + 32) = v277 + 1;
        *v277 = 10;
      }
      if ( v403 != (__m128i *)&v405 )
      {
        v143 = (__int64)v405 + 1;
        j_j___libc_free_0((unsigned __int64)v403);
      }
    }
    else
    {
      v143 = (__int64)v86;
      sub_CB7190((__int64)&v325, (__int64)v86, v139, v140, v141, v142);
      v144 = v326 & 1;
      v326 = (2 * (v326 & 1)) | v326 & 0xFD;
      if ( v144 )
        goto LABEL_169;
      v154 = sub_904010((__int64)v86, "{\"file\":\"");
      v155 = *(void **)(v154 + 32);
      v156 = v154;
      if ( *(_QWORD *)(v154 + 24) - (_QWORD)v155 < v50 )
      {
        v156 = sub_CB6200(v154, (unsigned __int8 *)v322, v50);
      }
      else if ( v50 )
      {
        memcpy(v155, v322, v50);
        *(_QWORD *)(v156 + 32) += v50;
      }
      v157 = 7;
      sub_904010(v156, "\", ");
      v158 = "no-name";
      if ( a8 )
      {
        v158 = (char *)a7;
        v157 = a8;
      }
      v159 = sub_904010((__int64)v86, "\"pass\":\"");
      v160 = *(void **)(v159 + 32);
      v161 = v159;
      if ( *(_QWORD *)(v159 + 24) - (_QWORD)v160 >= v157 )
      {
        memcpy(v160, v158, v157);
        *(_QWORD *)(v161 + 32) += v157;
      }
      else
      {
        v161 = sub_CB6200(v159, (unsigned __int8 *)v158, v157);
      }
      sub_904010(v161, "\", ");
      LOWORD(v403) = 8;
      v162 = v327;
      v327 = 0;
      v404 = (size_t)v162;
      v163 = v328;
      v328 = 0;
      v405 = v163;
      v164 = v329;
      v329 = 0;
      v406 = v164;
      sub_C6D380((__int64)&v330, (unsigned __int16 *)&v403, 1u);
      sub_C6BC50((unsigned __int16 *)&v403);
      v165 = sub_904010((__int64)v86, "\"bugs\": ");
      v403 = (__m128i *)&v405;
      v415 = (char *)v165;
      v413[2] = 0;
      v414 = 0;
      v416 = 0;
      v404 = 0x1000000001LL;
      LODWORD(v405) = 0;
      BYTE4(v405) = 0;
      sub_C6C710((__int64)&v403, (unsigned __int16 *)&v330, v166);
      if ( v403 != (__m128i *)&v405 )
        _libc_free((unsigned __int64)v403);
      v143 = (__int64)"}\n";
      sub_904010((__int64)v86, "}\n");
      sub_C6BC50((unsigned __int16 *)&v330);
      if ( (v326 & 2) != 0 )
LABEL_169:
        sub_29C20D0(&v325, v143);
      if ( (v326 & 1) != 0 )
      {
        if ( v325 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v325 + 8LL))(v325);
      }
      else if ( (_DWORD)v325 != -1 )
      {
        sub_C837B0((unsigned int)v325, (__int64)"}\n", v167, v168, v169, v170);
      }
      sub_CB7080((__int64)v86, (__int64)"}\n");
    }
    sub_CB5B00((int *)v86, v143);
  }
  v171 = sub_29C0AE0();
  v172 = sub_A51340((__int64)v171, v318, v310);
  if ( v320 )
    sub_904010(v172, ": PASS\n");
  else
    sub_904010(v172, ": FAIL\n");
  sub_C7D6A0(*(_QWORD *)(a4 + 8), 16LL * *(unsigned int *)(a4 + 24), 8);
  v177 = v382;
  *(_DWORD *)(a4 + 24) = v382;
  if ( (_DWORD)v177 )
  {
    v178 = (void *)sub_C7D670(16 * v177, 8);
    v179 = src;
    *(_QWORD *)(a4 + 8) = v178;
    v180 = *(unsigned int *)(a4 + 24);
    *(_QWORD *)(a4 + 16) = v381;
    memcpy(v178, v179, 16 * v180);
  }
  else
  {
    *(_QWORD *)(a4 + 8) = 0;
    *(_QWORD *)(a4 + 16) = 0;
  }
  sub_29C1190(a4 + 32, (__int64)&v383, v173, v174, v175, v176);
  sub_C7D6A0(*(_QWORD *)(a4 + 56), 16LL * *(unsigned int *)(a4 + 72), 8);
  v185 = v388;
  *(_DWORD *)(a4 + 72) = v388;
  if ( (_DWORD)v185 )
  {
    v186 = (void *)sub_C7D670(16 * v185, 8);
    v187 = v386;
    *(_QWORD *)(a4 + 56) = v186;
    v188 = *(unsigned int *)(a4 + 72);
    *(_QWORD *)(a4 + 64) = v387;
    memcpy(v186, v187, 16 * v188);
  }
  else
  {
    *(_QWORD *)(a4 + 56) = 0;
    *(_QWORD *)(a4 + 64) = 0;
  }
  sub_29C1060(a4 + 80, (__int64)&v389, v181, v182, v183, v184);
  sub_C7D6A0(*(_QWORD *)(a4 + 104), 16LL * *(unsigned int *)(a4 + 120), 8);
  v193 = v394;
  *(_DWORD *)(a4 + 120) = v394;
  if ( (_DWORD)v193 )
  {
    v194 = (void *)sub_C7D670(16 * v193, 8);
    v195 = v392;
    *(_QWORD *)(a4 + 104) = v194;
    v196 = *(unsigned int *)(a4 + 120);
    *(_QWORD *)(a4 + 112) = v393;
    memcpy(v194, v195, 16 * v196);
  }
  else
  {
    *(_QWORD *)(a4 + 104) = 0;
    *(_QWORD *)(a4 + 112) = 0;
  }
  sub_29C2230(a4 + 128, (__int64)&v395, v189, v190, v191, v192);
  sub_C7D6A0(*(_QWORD *)(a4 + 152), 16LL * *(unsigned int *)(a4 + 168), 8);
  v201 = v400;
  *(_DWORD *)(a4 + 168) = v400;
  if ( (_DWORD)v201 )
  {
    v202 = (void *)sub_C7D670(16 * v201, 8);
    v203 = v398;
    *(_QWORD *)(a4 + 152) = v202;
    v204 = *(unsigned int *)(a4 + 168);
    *(_QWORD *)(a4 + 160) = v399;
    memcpy(v202, v203, 16 * v204);
  }
  else
  {
    *(_QWORD *)(a4 + 152) = 0;
    *(_QWORD *)(a4 + 160) = 0;
  }
  sub_29C0F30(a4 + 176, (__int64)&v401, v197, v198, v199, v200);
  v205 = v328;
  v206 = (unsigned __int64)v327;
  if ( v328 != v327 )
  {
    do
    {
      v207 = (unsigned __int16 *)v206;
      v206 += 40LL;
      sub_C6BC50(v207);
    }
    while ( v205 != (__int16 *)v206 );
    v206 = (unsigned __int64)v327;
  }
  if ( v206 )
    j_j___libc_free_0(v206);
  if ( v372 != (unsigned __int8 **)v86 )
    _libc_free((unsigned __int64)v372);
  sub_C7D6A0((__int64)v369, 16LL * (unsigned int)v371, 8);
  if ( v366 != &v368 )
    _libc_free((unsigned __int64)v366);
  sub_C7D6A0((__int64)v363, 16LL * (unsigned int)v365, 8);
  v208 = v360;
  v209 = &v360[4 * (unsigned int)v361];
  if ( v360 != v209 )
  {
    do
    {
      v210 = *(v209 - 1);
      v209 -= 4;
      if ( v210 != 0 && v210 != -4096 && v210 != -8192 )
        sub_BD60C0(v209 + 1);
    }
    while ( v208 != v209 );
    v209 = v360;
  }
  if ( v209 != &v362 )
    _libc_free((unsigned __int64)v209);
  sub_C7D6A0((__int64)v357, 16LL * (unsigned int)v359, 8);
  if ( (__int64 *)v355[0] != &v356 )
    _libc_free(v355[0]);
  sub_C7D6A0((__int64)v352, 16LL * (unsigned int)v354, 8);
  if ( (__int64 *)v350[0] != &v351 )
    _libc_free(v350[0]);
  sub_C7D6A0((__int64)v347, 16LL * (unsigned int)v349, 8);
  if ( v344 != &v346 )
    _libc_free((unsigned __int64)v344);
  sub_C7D6A0((__int64)v341, 16LL * (unsigned int)v343, 8);
  if ( v338 != &v340 )
    _libc_free((unsigned __int64)v338);
  sub_C7D6A0((__int64)v335, 16LL * (unsigned int)v337, 8);
  if ( v401 != &v403 )
    _libc_free((unsigned __int64)v401);
  sub_C7D6A0((__int64)v398, 16LL * v400, 8);
  v211 = v395;
  v212 = &v395[4 * (unsigned int)v396];
  if ( v395 != v212 )
  {
    do
    {
      v213 = *(v212 - 1);
      v212 -= 4;
      if ( v213 != 0 && v213 != -4096 && v213 != -8192 )
        sub_BD60C0(v212 + 1);
    }
    while ( v211 != v212 );
    v212 = v395;
  }
  if ( v212 != &v397 )
    _libc_free((unsigned __int64)v212);
  sub_C7D6A0((__int64)v392, 16LL * v394, 8);
  if ( v389 != &v391 )
    _libc_free((unsigned __int64)v389);
  sub_C7D6A0((__int64)v386, 16LL * v388, 8);
  if ( v383 != &v385 )
    _libc_free((unsigned __int64)v383);
  sub_C7D6A0((__int64)src, 16LL * v382, 8);
  return v320;
}
