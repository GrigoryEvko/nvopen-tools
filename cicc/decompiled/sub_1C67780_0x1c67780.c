// Function: sub_1C67780
// Address: 0x1c67780
//
_BOOL8 __fastcall sub_1C67780(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 **a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __m128 a8,
        __m128 a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v13; // r14
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rbx
  _QWORD *v20; // r12
  char v21; // r8
  __int64 v22; // rax
  char v23; // al
  __int64 *v24; // rdi
  int v25; // esi
  int v26; // eax
  __int64 v27; // rax
  _BYTE *v28; // rsi
  _QWORD *v29; // rdi
  __int64 v30; // r13
  __int64 v31; // r14
  __int64 *v32; // r12
  _QWORD *v33; // r12
  char v34; // r8
  __int64 v35; // rax
  char v36; // al
  int v37; // esi
  int v38; // eax
  int v39; // eax
  __int64 *v40; // rbx
  __int64 v41; // r13
  char v42; // al
  __int64 *v43; // rsi
  int v44; // eax
  __int64 v45; // rax
  __int64 *v46; // rbx
  __int64 v47; // r13
  char v48; // al
  __int64 *v49; // rsi
  int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // r13
  unsigned int v56; // r12d
  __int64 v57; // rsi
  __int64 *v58; // rax
  char v59; // al
  __int64 v60; // r15
  int v61; // esi
  int v62; // eax
  int v63; // eax
  char v64; // al
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 *v67; // rsi
  _QWORD *v68; // r15
  _QWORD *v69; // rbx
  _QWORD *v70; // r12
  __int64 v71; // rdi
  _QWORD *v72; // rbx
  _QWORD *v73; // r12
  __int64 v74; // rdi
  unsigned __int64 *v76; // r15
  __int64 v77; // rbx
  char v78; // r8
  __int64 v79; // rax
  char v80; // al
  __int64 v81; // rbx
  int v82; // esi
  int v83; // eax
  int v84; // eax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // r12
  unsigned __int64 *v88; // rax
  _QWORD *v89; // r15
  __int64 v90; // r14
  __int64 v91; // rbx
  __int64 i; // r12
  unsigned __int64 v93; // rax
  __int64 v94; // r15
  __int64 v95; // rdx
  int v96; // esi
  __int64 v97; // rcx
  __int64 *v98; // rax
  __int64 v99; // r8
  __int64 *v100; // r13
  __int64 v101; // rdx
  int v102; // r12d
  __int64 v103; // rcx
  unsigned int v104; // ecx
  int *v105; // rax
  int v106; // r9d
  _BYTE *v107; // rsi
  _QWORD *v108; // rdx
  int v109; // esi
  unsigned __int64 v110; // rdi
  int v111; // r8d
  _DWORD *v112; // rdi
  int v113; // edx
  _QWORD *v114; // rdi
  int v115; // r14d
  int v116; // esi
  int v117; // eax
  int v118; // eax
  unsigned int v119; // esi
  int *v120; // rdx
  int v121; // edi
  char v122; // al
  __int64 v123; // r15
  __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // r14
  __int64 v127; // r13
  __int64 *v128; // rdx
  _QWORD **v129; // rax
  _QWORD **v130; // r10
  __int64 *v131; // r12
  int v132; // ebx
  __int64 v133; // rdx
  _QWORD *v134; // rsi
  unsigned int v135; // edx
  __int64 *v136; // rdi
  __int64 v137; // r9
  int v138; // edi
  unsigned int v139; // eax
  __int64 *v140; // rcx
  __int64 v141; // rsi
  int v142; // ecx
  char v143; // al
  char **v144; // rdx
  __int64 **v145; // rbx
  __int64 v146; // rax
  __int64 v147; // r13
  __int64 v148; // r11
  unsigned int v149; // eax
  __int64 *v150; // rcx
  __int64 v151; // rdi
  int v152; // esi
  __int64 *v153; // r12
  __int64 *v154; // r10
  int v155; // edx
  char v156; // al
  _QWORD *v157; // rdx
  int v158; // esi
  int v159; // eax
  __int64 v160; // rdx
  unsigned __int64 v161; // rax
  __int64 v162; // rdx
  __int64 *v163; // r12
  char *v164; // rbx
  char v165; // r8
  double v166; // xmm4_8
  double v167; // xmm5_8
  char *v168; // rax
  unsigned int v169; // esi
  __int64 v170; // rcx
  __int64 v171; // rdx
  char *v172; // rax
  __int64 v173; // r8
  __int64 *v174; // r13
  __int64 v175; // rdi
  __int64 v176; // rdx
  int v177; // r12d
  __int64 ******v178; // rsi
  unsigned int v179; // edx
  __int64 *******v180; // rcx
  __int64 ******v181; // r9
  int v182; // ecx
  int v183; // r11d
  char v184; // al
  int v185; // esi
  int v186; // eax
  char *v187; // rdi
  int v188; // edx
  __int64 v189; // r11
  unsigned __int64 v190; // rax
  int v191; // ebx
  __int64 *v192; // r13
  __int64 v193; // rdx
  __int64 v194; // rcx
  _QWORD *v195; // rax
  __int64 *v196; // r15
  __int64 v197; // r13
  __m128i *v198; // r12
  __m128i *v199; // rdi
  signed __int64 v200; // rbx
  __int64 v201; // r9
  signed __int64 v202; // rdx
  unsigned __int64 v203; // rax
  unsigned __int64 v204; // r15
  __int64 v205; // rbx
  __int64 v206; // r13
  __int64 v207; // r14
  __int64 *v208; // r12
  __int64 v209; // r8
  __int64 *v210; // r13
  __m128i *v211; // rbx
  const __m128i *v212; // rdi
  __int64 v213; // r14
  __int64 *v214; // rax
  __int64 v215; // rbx
  unsigned int ****v216; // rax
  unsigned int ***v217; // r15
  __int64 *v218; // r12
  int v219; // edi
  unsigned int *v220; // rsi
  unsigned int v221; // edx
  unsigned int **v222; // r8
  unsigned int *v223; // r9
  __int64 ***v224; // r9
  __int64 **v225; // rdx
  unsigned int v226; // eax
  __int64 ***v227; // r8
  __int64 **v228; // r10
  _QWORD *v229; // r13
  _QWORD *v230; // rbx
  _QWORD *v231; // r14
  _QWORD *v232; // r12
  __int64 v233; // rbx
  __int64 v234; // r12
  __int64 v235; // rdi
  __int64 v236; // rsi
  __int64 v237; // rbx
  __int64 v238; // r12
  __int64 v239; // rdi
  __int64 v240; // rsi
  __int64 *v241; // r12
  int v242; // eax
  __int64 v243; // rbx
  __int64 v244; // rax
  int v245; // esi
  unsigned int v246; // edx
  int *v247; // rbx
  int v248; // ecx
  __int64 v249; // rdx
  __int64 v250; // rcx
  __int64 v251; // r15
  __int64 v252; // r12
  __m128i *v253; // rsi
  const __m128i *v254; // rdx
  int v255; // esi
  int v256; // eax
  int j; // edx
  int v258; // r9d
  int v259; // ebx
  int v260; // r12d
  __int64 *v261; // rdi
  int v262; // ecx
  __int64 *v263; // rsi
  int v264; // esi
  int v265; // esi
  _QWORD *v266; // rax
  __int64 v267; // rdx
  _QWORD *v268; // r13
  _QWORD *v269; // rbx
  _QWORD *v270; // r8
  __int64 v271; // rdi
  __int64 v272; // rax
  __int64 v273; // r12
  __int64 v274; // r14
  __int64 v275; // rdi
  int v276; // r8d
  __int64 v277; // rsi
  unsigned __int32 v278; // ecx
  __int64 *v279; // rdx
  __int64 v280; // r9
  __int64 *v281; // rax
  __int64 v282; // rax
  __int64 v283; // rax
  __int64 *v284; // rbx
  __int64 v285; // r12
  char v286; // al
  _QWORD *v287; // rdx
  unsigned __int64 v288; // rdi
  __int64 v289; // r15
  __int64 v290; // r13
  __int64 v291; // rax
  unsigned int v292; // esi
  int v293; // eax
  int v294; // eax
  int v295; // r8d
  int v296; // r10d
  __int64 v297; // r13
  char v298; // al
  _QWORD *v299; // rdx
  int v300; // esi
  int v301; // eax
  int v302; // edi
  int v303; // r11d
  int v304; // edx
  int v305; // r10d
  int v306; // r15d
  int *v307; // rdi
  int v308; // eax
  __int64 *v309; // r11
  __int64 *v310; // r11
  unsigned int ***v311; // [rsp+0h] [rbp-460h]
  bool v312; // [rsp+Fh] [rbp-451h]
  __int64 *v313; // [rsp+10h] [rbp-450h]
  __int64 v314; // [rsp+18h] [rbp-448h]
  unsigned int v315; // [rsp+20h] [rbp-440h]
  __int64 v316; // [rsp+20h] [rbp-440h]
  __int64 v317; // [rsp+28h] [rbp-438h]
  __int64 v318; // [rsp+28h] [rbp-438h]
  __int64 v319; // [rsp+30h] [rbp-430h]
  int v320; // [rsp+30h] [rbp-430h]
  __int64 v321; // [rsp+30h] [rbp-430h]
  _QWORD *v322; // [rsp+30h] [rbp-430h]
  __int64 v323; // [rsp+38h] [rbp-428h]
  char *v324; // [rsp+38h] [rbp-428h]
  __int64 v325; // [rsp+38h] [rbp-428h]
  __int64 *v327; // [rsp+40h] [rbp-420h]
  __int64 *v328; // [rsp+40h] [rbp-420h]
  _QWORD *v329; // [rsp+40h] [rbp-420h]
  _QWORD *v330; // [rsp+40h] [rbp-420h]
  int v331; // [rsp+48h] [rbp-418h]
  __int64 v332; // [rsp+48h] [rbp-418h]
  unsigned __int64 v333; // [rsp+48h] [rbp-418h]
  __int64 v335; // [rsp+50h] [rbp-410h]
  __int64 *v336; // [rsp+50h] [rbp-410h]
  __int64 v337; // [rsp+50h] [rbp-410h]
  __int64 v339; // [rsp+58h] [rbp-408h]
  __int64 v340; // [rsp+58h] [rbp-408h]
  __int64 v341; // [rsp+58h] [rbp-408h]
  int v342; // [rsp+58h] [rbp-408h]
  __int64 *v343; // [rsp+58h] [rbp-408h]
  __int64 v344; // [rsp+58h] [rbp-408h]
  __int64 v345; // [rsp+58h] [rbp-408h]
  __int64 *src; // [rsp+60h] [rbp-400h]
  _QWORD *srca; // [rsp+60h] [rbp-400h]
  __int64 *srcb; // [rsp+60h] [rbp-400h]
  __int64 *srcc; // [rsp+60h] [rbp-400h]
  unsigned int ***v351; // [rsp+68h] [rbp-3F8h]
  _QWORD *v352; // [rsp+68h] [rbp-3F8h]
  int v353; // [rsp+74h] [rbp-3ECh] BYREF
  __int64 v354; // [rsp+78h] [rbp-3E8h] BYREF
  __int64 v355; // [rsp+80h] [rbp-3E0h] BYREF
  __int64 v356; // [rsp+88h] [rbp-3D8h] BYREF
  __int64 v357; // [rsp+90h] [rbp-3D0h] BYREF
  __int64 v358; // [rsp+98h] [rbp-3C8h]
  __int64 v359; // [rsp+A0h] [rbp-3C0h]
  __int64 *v360; // [rsp+B0h] [rbp-3B0h] BYREF
  __int64 *v361; // [rsp+B8h] [rbp-3A8h]
  __int64 *v362; // [rsp+C0h] [rbp-3A0h]
  __int64 *v363; // [rsp+D0h] [rbp-390h] BYREF
  __int64 *v364; // [rsp+D8h] [rbp-388h]
  __int64 *v365; // [rsp+E0h] [rbp-380h]
  unsigned __int64 v366; // [rsp+F0h] [rbp-370h] BYREF
  __int64 v367; // [rsp+F8h] [rbp-368h]
  __int64 v368; // [rsp+100h] [rbp-360h]
  _QWORD v369[2]; // [rsp+110h] [rbp-350h] BYREF
  __int64 v370; // [rsp+120h] [rbp-340h]
  char *v371; // [rsp+130h] [rbp-330h] BYREF
  __m128i *v372; // [rsp+138h] [rbp-328h]
  const __m128i *v373; // [rsp+140h] [rbp-320h]
  __int64 v374; // [rsp+150h] [rbp-310h] BYREF
  _QWORD *v375; // [rsp+158h] [rbp-308h]
  __int64 v376; // [rsp+160h] [rbp-300h]
  unsigned int v377; // [rsp+168h] [rbp-2F8h]
  __int64 v378; // [rsp+170h] [rbp-2F0h] BYREF
  _QWORD *v379; // [rsp+178h] [rbp-2E8h]
  __int64 v380; // [rsp+180h] [rbp-2E0h]
  unsigned int v381; // [rsp+188h] [rbp-2D8h]
  __int64 v382; // [rsp+190h] [rbp-2D0h] BYREF
  _QWORD *v383; // [rsp+198h] [rbp-2C8h]
  __int64 v384; // [rsp+1A0h] [rbp-2C0h]
  unsigned int v385; // [rsp+1A8h] [rbp-2B8h]
  __int64 v386; // [rsp+1B0h] [rbp-2B0h] BYREF
  __int64 v387; // [rsp+1B8h] [rbp-2A8h]
  __int64 v388; // [rsp+1C0h] [rbp-2A0h]
  unsigned int v389; // [rsp+1C8h] [rbp-298h]
  __int64 v390; // [rsp+1D0h] [rbp-290h] BYREF
  __int64 v391; // [rsp+1D8h] [rbp-288h]
  __int64 v392; // [rsp+1E0h] [rbp-280h]
  unsigned int v393; // [rsp+1E8h] [rbp-278h]
  __int64 v394; // [rsp+1F0h] [rbp-270h] BYREF
  __int64 v395; // [rsp+1F8h] [rbp-268h]
  __int64 v396; // [rsp+200h] [rbp-260h]
  unsigned int v397; // [rsp+208h] [rbp-258h]
  __int64 v398; // [rsp+210h] [rbp-250h] BYREF
  __int64 v399; // [rsp+218h] [rbp-248h]
  __int64 v400; // [rsp+220h] [rbp-240h]
  __int64 v401; // [rsp+228h] [rbp-238h]
  _QWORD *v402; // [rsp+230h] [rbp-230h] BYREF
  __int64 v403; // [rsp+238h] [rbp-228h]
  __int64 v404; // [rsp+240h] [rbp-220h]
  __int64 v405; // [rsp+248h] [rbp-218h]
  _QWORD *v406; // [rsp+250h] [rbp-210h] BYREF
  _QWORD *v407; // [rsp+258h] [rbp-208h]
  __int64 v408; // [rsp+260h] [rbp-200h]
  unsigned int v409; // [rsp+268h] [rbp-1F8h]
  __m128i v410; // [rsp+270h] [rbp-1F0h] BYREF
  __m128i v411; // [rsp+280h] [rbp-1E0h] BYREF
  _QWORD v412[6]; // [rsp+290h] [rbp-1D0h] BYREF
  int v413; // [rsp+2C0h] [rbp-1A0h]
  __int64 v414; // [rsp+2C8h] [rbp-198h]
  __int64 v415; // [rsp+2D0h] [rbp-190h]
  __int64 v416; // [rsp+2D8h] [rbp-188h]
  __int64 v417; // [rsp+2E0h] [rbp-180h]
  __int64 v418; // [rsp+2E8h] [rbp-178h]
  __int64 v419; // [rsp+2F0h] [rbp-170h]
  __int64 v420; // [rsp+2F8h] [rbp-168h]
  __int64 v421; // [rsp+300h] [rbp-160h]
  __int64 v422; // [rsp+308h] [rbp-158h]
  __int64 v423; // [rsp+310h] [rbp-150h]
  __int64 v424; // [rsp+318h] [rbp-148h]
  int v425; // [rsp+320h] [rbp-140h]
  __int64 v426; // [rsp+328h] [rbp-138h]
  _BYTE *v427; // [rsp+330h] [rbp-130h]
  _BYTE *v428; // [rsp+338h] [rbp-128h]
  __int64 v429; // [rsp+340h] [rbp-120h]
  int v430; // [rsp+348h] [rbp-118h]
  _BYTE v431[16]; // [rsp+350h] [rbp-110h] BYREF
  __int64 v432; // [rsp+360h] [rbp-100h]
  __int64 v433; // [rsp+368h] [rbp-F8h]
  __int64 v434; // [rsp+370h] [rbp-F0h]
  __int64 v435; // [rsp+378h] [rbp-E8h]
  __int64 v436; // [rsp+380h] [rbp-E0h]
  __int64 v437; // [rsp+388h] [rbp-D8h]
  __int16 v438; // [rsp+390h] [rbp-D0h]
  __int64 v439; // [rsp+398h] [rbp-C8h]
  __int64 v440; // [rsp+3A0h] [rbp-C0h]
  __int64 v441; // [rsp+3A8h] [rbp-B8h]
  __int64 v442; // [rsp+3B0h] [rbp-B0h]
  __int64 v443; // [rsp+3B8h] [rbp-A8h]
  int v444; // [rsp+3C0h] [rbp-A0h]
  __int64 v445; // [rsp+3C8h] [rbp-98h]
  __int64 v446; // [rsp+3D0h] [rbp-90h]
  __int64 v447; // [rsp+3D8h] [rbp-88h]
  char *v448; // [rsp+3E0h] [rbp-80h]
  __int64 v449; // [rsp+3E8h] [rbp-78h]
  char v450; // [rsp+3F0h] [rbp-70h] BYREF

  v13 = a1;
  v15 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 24LL) + 40LL));
  memset(&v412[3], 0, 24);
  v16 = v15;
  v17 = *(_QWORD *)(a1 + 184);
  v412[2] = "BaseAddressStrengthReduce";
  v412[1] = v16;
  v412[0] = v17;
  v427 = v431;
  v428 = v431;
  v413 = 0;
  v414 = 0;
  v415 = 0;
  v416 = 0;
  v417 = 0;
  v418 = 0;
  v419 = 0;
  v420 = 0;
  v421 = 0;
  v422 = 0;
  v423 = 0;
  v424 = 0;
  v425 = 0;
  v426 = 0;
  v429 = 2;
  v430 = 0;
  v432 = 0;
  v433 = 0;
  v434 = 0;
  v435 = 0;
  v436 = 0;
  v437 = 0;
  v438 = 1;
  v18 = sub_15E0530(*(_QWORD *)(v17 + 24));
  v447 = v16;
  v442 = v18;
  v448 = &v450;
  v449 = 0x800000000LL;
  v439 = 0;
  v441 = 0;
  v443 = 0;
  v444 = 0;
  v445 = 0;
  v446 = 0;
  v440 = 0;
  v374 = 0;
  v375 = 0;
  v376 = 0;
  v377 = 0;
  v357 = 0;
  v358 = 0;
  v359 = 0;
  v378 = 0;
  v379 = 0;
  v380 = 0;
  v381 = 0;
  v360 = 0;
  v361 = 0;
  v362 = 0;
  v382 = 0;
  v383 = 0;
  v384 = 0;
  v385 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v19 = *a4;
  src = a4[1];
  if ( *a4 != src )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v394 = *v19;
        v30 = *(_QWORD *)v394;
        v398 = *(_QWORD *)v394;
        if ( byte_4FBC760 && sub_1C51E40(v30) )
          goto LABEL_14;
        v31 = *(_QWORD *)(a1 + 184);
        v32 = sub_1C57390(a1 + 32, &v398);
        *((_DWORD *)v32 + 2) = sub_1CCB2B0(v31, v30);
        if ( *(_BYTE *)(sub_1456040(v398) + 8) == 15 )
          break;
        v20 = &v383[4 * v385];
        v406 = (_QWORD *)sub_1456040(v398);
        v21 = sub_1C50900((__int64)&v382, (__int64 *)&v406, &v410);
        v22 = v410.m128i_i64[0];
        if ( !v21 )
          v22 = (__int64)&v383[4 * v385];
        if ( v20 == (_QWORD *)v22 )
        {
          v66 = sub_1456040(v398);
          v67 = v364;
          v410.m128i_i64[0] = v66;
          if ( v364 == v365 )
          {
            sub_1C56460((__int64)&v363, v364, &v410);
          }
          else
          {
            if ( v364 )
            {
              *v364 = v66;
              v67 = v364;
            }
            v364 = v67 + 1;
          }
        }
        v406 = (_QWORD *)sub_1456040(v398);
        v23 = sub_1C50900((__int64)&v382, (__int64 *)&v406, &v410);
        v24 = (__int64 *)v410.m128i_i64[0];
        if ( !v23 )
        {
          v25 = v385;
          ++v382;
          v26 = v384 + 1;
          if ( 4 * ((int)v384 + 1) >= 3 * v385 )
          {
            v25 = 2 * v385;
          }
          else if ( v385 - HIDWORD(v384) - v26 > v385 >> 3 )
          {
LABEL_9:
            LODWORD(v384) = v26;
            if ( *v24 != -8 )
              --HIDWORD(v384);
            v27 = (__int64)v406;
            goto LABEL_12;
          }
          sub_1C54050((__int64)&v382, v25);
          sub_1C50900((__int64)&v382, (__int64 *)&v406, &v410);
          v24 = (__int64 *)v410.m128i_i64[0];
          v26 = v384 + 1;
          goto LABEL_9;
        }
LABEL_27:
        v28 = (_BYTE *)v24[2];
        if ( v28 == (_BYTE *)v24[3] )
        {
          v29 = v24 + 1;
          goto LABEL_13;
        }
        if ( v28 )
        {
          *(_QWORD *)v28 = v394;
          v28 = (_BYTE *)v24[2];
        }
        ++v19;
        v24[2] = (__int64)(v28 + 8);
        if ( src == v19 )
        {
LABEL_31:
          v13 = a1;
          goto LABEL_32;
        }
      }
      v402 = (_QWORD *)sub_1CCDC20(*(_QWORD *)(***(_QWORD ***)(v394 + 8) + 24LL));
      v406 = v402;
      v33 = &v379[4 * v381];
      v34 = sub_1C50590((__int64)&v378, (__int64 *)&v406, &v410);
      v35 = v410.m128i_i64[0];
      if ( !v34 )
        v35 = (__int64)&v379[4 * v381];
      if ( v33 == (_QWORD *)v35 )
      {
        v263 = v361;
        if ( v361 == v362 )
        {
          sub_1C55D40((__int64)&v360, v361, &v402);
        }
        else
        {
          if ( v361 )
          {
            *v361 = (__int64)v402;
            v263 = v361;
          }
          v361 = v263 + 1;
        }
      }
      v36 = sub_1C50590((__int64)&v378, (__int64 *)&v402, &v410);
      v24 = (__int64 *)v410.m128i_i64[0];
      if ( v36 )
        goto LABEL_27;
      v37 = v381;
      ++v378;
      v38 = v380 + 1;
      if ( 4 * ((int)v380 + 1) >= 3 * v381 )
      {
        v37 = 2 * v381;
LABEL_410:
        sub_1C54240((__int64)&v378, v37);
        sub_1C50590((__int64)&v378, (__int64 *)&v402, &v410);
        v24 = (__int64 *)v410.m128i_i64[0];
        v38 = v380 + 1;
        goto LABEL_24;
      }
      if ( v381 - HIDWORD(v380) - v38 <= v381 >> 3 )
        goto LABEL_410;
LABEL_24:
      LODWORD(v380) = v38;
      if ( *v24 != -8 )
        --HIDWORD(v380);
      v27 = (__int64)v402;
LABEL_12:
      *v24 = v27;
      v28 = 0;
      v29 = v24 + 1;
      *v29 = 0;
      v29[1] = 0;
      v29[2] = 0;
LABEL_13:
      sub_1C50F10((__int64)v29, v28, &v394);
LABEL_14:
      if ( src == ++v19 )
        goto LABEL_31;
    }
  }
LABEL_32:
  v39 = sub_1BFBA30(*(unsigned int **)(v13 + 208), a2, 0);
  if ( v39 <= 0 )
  {
    v331 = -1;
  }
  else
  {
    if ( v39 < a3 )
    {
      v312 = 0;
      v68 = v412;
      goto LABEL_80;
    }
    v331 = 0;
    if ( dword_4FBCF40 >= a3 )
      v331 = v39 - a3;
  }
  v327 = v361;
  if ( v360 != v361 )
  {
    v40 = v360;
    while ( 1 )
    {
      v41 = *(_QWORD *)(v13 + 200);
      v42 = sub_1C50590((__int64)&v378, v40, &v410);
      v43 = (__int64 *)v410.m128i_i64[0];
      if ( !v42 )
        break;
LABEL_44:
      ++v40;
      sub_1C65B90(v13, v43 + 1, v41, a3, (__int64)&v374, (__int64)&v357);
      if ( v327 == v40 )
        goto LABEL_45;
    }
    ++v378;
    v44 = v380 + 1;
    if ( 4 * ((int)v380 + 1) >= 3 * v381 )
    {
      v265 = 2 * v381;
    }
    else
    {
      if ( v381 - HIDWORD(v380) - v44 > v381 >> 3 )
      {
LABEL_41:
        LODWORD(v380) = v44;
        if ( *v43 != -8 )
          --HIDWORD(v380);
        v45 = *v40;
        v43[1] = 0;
        v43[2] = 0;
        *v43 = v45;
        v43[3] = 0;
        goto LABEL_44;
      }
      v265 = v381;
    }
    sub_1C54240((__int64)&v378, v265);
    sub_1C50590((__int64)&v378, v40, &v410);
    v43 = (__int64 *)v410.m128i_i64[0];
    v44 = v380 + 1;
    goto LABEL_41;
  }
LABEL_45:
  v328 = v364;
  if ( v363 != v364 )
  {
    v46 = v363;
    while ( 1 )
    {
      v47 = *(_QWORD *)(v13 + 200);
      v48 = sub_1C50900((__int64)&v382, v46, &v410);
      v49 = (__int64 *)v410.m128i_i64[0];
      if ( !v48 )
        break;
LABEL_53:
      ++v46;
      sub_1C65B90(v13, v49 + 1, v47, a3, (__int64)&v374, (__int64)&v357);
      if ( v328 == v46 )
        goto LABEL_54;
    }
    ++v382;
    v50 = v384 + 1;
    if ( 4 * ((int)v384 + 1) >= 3 * v385 )
    {
      v264 = 2 * v385;
    }
    else
    {
      if ( v385 - HIDWORD(v384) - v50 > v385 >> 3 )
      {
LABEL_50:
        LODWORD(v384) = v50;
        if ( *v49 != -8 )
          --HIDWORD(v384);
        v51 = *v46;
        v49[1] = 0;
        v49[2] = 0;
        *v49 = v51;
        v49[3] = 0;
        goto LABEL_53;
      }
      v264 = v385;
    }
    sub_1C54050((__int64)&v382, v264);
    sub_1C50900((__int64)&v382, v46, &v410);
    v49 = (__int64 *)v410.m128i_i64[0];
    v50 = v384 + 1;
    goto LABEL_50;
  }
LABEL_54:
  v52 = v357;
  v386 = 0;
  v387 = 0;
  v388 = 0;
  v389 = 0;
  v53 = (v358 - v357) >> 3;
  v366 = 0;
  LODWORD(v54) = v53;
  v367 = 0;
  v368 = 0;
  v390 = 0;
  v391 = 0;
  v392 = 0;
  v393 = 0;
  if ( !(_DWORD)v53 )
  {
    v394 = 0;
    v395 = 0;
    v396 = 0;
    v397 = 0;
    v315 = 0;
    v312 = 0;
    goto LABEL_152;
  }
  v55 = 0;
  v339 = 8LL * (unsigned int)(v53 - 1);
  v56 = 0;
  while ( 1 )
  {
    if ( sub_1C53600((__int64)&v374, (__int64 *)(v52 + v55))[1] )
    {
      v58 = sub_1C53600((__int64)&v374, (__int64 *)(v55 + v357));
      LODWORD(v402) = (__int64)(*(_QWORD *)(v58[1] + 8) - *(_QWORD *)v58[1]) >> 3;
    }
    else
    {
      LODWORD(v402) = 0;
    }
    v59 = sub_1C504F0((__int64)&v390, (int *)&v402, &v410);
    v60 = v410.m128i_i64[0];
    if ( v59 )
      goto LABEL_72;
    v61 = v393;
    ++v390;
    v62 = v392 + 1;
    if ( 4 * ((int)v392 + 1) >= 3 * v393 )
    {
      v61 = 2 * v393;
LABEL_332:
      sub_1C53A10((__int64)&v390, v61);
      sub_1C504F0((__int64)&v390, (int *)&v402, &v410);
      v60 = v410.m128i_i64[0];
      v62 = v392 + 1;
      goto LABEL_69;
    }
    if ( v393 - HIDWORD(v392) - v62 <= v393 >> 3 )
      goto LABEL_332;
LABEL_69:
    LODWORD(v392) = v62;
    if ( *(_DWORD *)v60 != -1 )
      --HIDWORD(v392);
    v63 = (int)v402;
    *(_QWORD *)(v60 + 8) = 0;
    *(_QWORD *)(v60 + 16) = 0;
    *(_DWORD *)v60 = v63;
    *(_QWORD *)(v60 + 24) = 0;
LABEL_72:
    v406 = *(_QWORD **)(v357 + v55);
    v64 = sub_1C506F0((__int64)&v374, (__int64 *)&v406, &v410);
    v65 = v410.m128i_i64[0];
    if ( v64 )
    {
      v410.m128i_i64[0] = (__int64)&v374;
      v411.m128i_i64[0] = v65;
      v410.m128i_i64[1] = v374;
      v411.m128i_i64[1] = (__int64)&v375[2 * v377];
      v57 = *(_QWORD *)(v60 + 16);
      if ( v57 == *(_QWORD *)(v60 + 24) )
        goto LABEL_74;
    }
    else
    {
      v410.m128i_i64[0] = (__int64)&v374;
      v411.m128i_i64[0] = (__int64)&v375[2 * v377];
      v410.m128i_i64[1] = v374;
      v411.m128i_i64[1] = v411.m128i_i64[0];
      v57 = *(_QWORD *)(v60 + 16);
      if ( v57 == *(_QWORD *)(v60 + 24) )
      {
LABEL_74:
        sub_1C50A60((const __m128i **)(v60 + 8), (const __m128i *)v57, &v410);
        goto LABEL_60;
      }
    }
    if ( v57 )
    {
      a8 = (__m128)_mm_loadu_si128(&v410);
      *(__m128 *)v57 = a8;
      a9 = (__m128)_mm_loadu_si128(&v411);
      *(__m128 *)(v57 + 16) = a9;
      v57 = *(_QWORD *)(v60 + 16);
    }
    *(_QWORD *)(v60 + 16) = v57 + 32;
LABEL_60:
    if ( v56 < (unsigned int)v402 )
      v56 = (unsigned int)v402;
    if ( v339 == v55 )
      break;
    v52 = v357;
    v55 += 8;
  }
  v315 = v56;
  LODWORD(v406) = 2;
  if ( v56 > 1 )
  {
    v76 = &v366;
    while ( 1 )
    {
      v77 = v391 + 32LL * v393;
      v78 = sub_1C504F0((__int64)&v390, (int *)&v406, &v410);
      v79 = v410.m128i_i64[0];
      if ( !v78 )
        v79 = v391 + 32LL * v393;
      if ( v77 == v79 )
        goto LABEL_104;
      v80 = sub_1C504F0((__int64)&v390, (int *)&v406, &v410);
      v81 = v410.m128i_i64[0];
      if ( v80 )
      {
        v85 = *(_QWORD *)(v410.m128i_i64[0] + 8);
        v86 = (*(_QWORD *)(v410.m128i_i64[0] + 16) - v85) >> 5;
        if ( (_DWORD)v86 )
        {
          v87 = (unsigned int)(v86 - 1);
          v88 = v76;
          v89 = (_QWORD *)v13;
          v90 = v410.m128i_i64[0];
          v91 = (__int64)v88;
          v340 = 32 * v87;
          for ( i = 0; ; i += 32 )
          {
            sub_1C62760(
              v89,
              **(char ***)(v85 + i + 16),
              *(_QWORD *)(*(_QWORD *)(v85 + i + 16) + 8LL),
              (__int64)&v386,
              v91,
              v89[25]);
            if ( v340 == i )
              break;
            v85 = *(_QWORD *)(v90 + 8);
          }
          v13 = (__int64)v89;
          v76 = (unsigned __int64 *)v91;
        }
        goto LABEL_104;
      }
      v82 = v393;
      ++v390;
      v83 = v392 + 1;
      if ( 4 * ((int)v392 + 1) >= 3 * v393 )
        break;
      if ( v393 - HIDWORD(v392) - v83 <= v393 >> 3 )
        goto LABEL_408;
LABEL_111:
      LODWORD(v392) = v83;
      if ( *(_DWORD *)v81 != -1 )
        --HIDWORD(v392);
      v84 = (int)v406;
      *(_QWORD *)(v81 + 8) = 0;
      *(_QWORD *)(v81 + 16) = 0;
      *(_DWORD *)v81 = v84;
      *(_QWORD *)(v81 + 24) = 0;
LABEL_104:
      LODWORD(v406) = (_DWORD)v406 + 1;
      if ( (unsigned int)v406 > v315 )
        goto LABEL_119;
    }
    v82 = 2 * v393;
LABEL_408:
    sub_1C53A10((__int64)&v390, v82);
    sub_1C504F0((__int64)&v390, (int *)&v406, &v410);
    v81 = v410.m128i_i64[0];
    v83 = v392 + 1;
    goto LABEL_111;
  }
LABEL_119:
  v93 = v366;
  v394 = 0;
  v395 = 0;
  v396 = 0;
  v312 = (_DWORD)v388 != 0;
  v397 = 0;
  v54 = (__int64)(v367 - v366) >> 3;
  if ( (_DWORD)v54 )
  {
    v335 = v13;
    v94 = 0;
    v341 = 8LL * (unsigned int)(v54 - 1);
    LODWORD(v54) = 0;
    while ( 1 )
    {
      v95 = *(_QWORD *)(v93 + v94);
      v96 = v389;
      v402 = (_QWORD *)v95;
      if ( !v389 )
        break;
      LODWORD(v97) = (v389 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
      v98 = (__int64 *)(v387 + 16LL * (unsigned int)v97);
      v99 = *v98;
      if ( v95 == *v98 )
      {
        v100 = (__int64 *)v98[1];
        goto LABEL_124;
      }
      v260 = 1;
      v261 = 0;
      while ( v99 != -8 )
      {
        if ( v99 != -16 || v261 )
          v98 = v261;
        v97 = (v389 - 1) & ((_DWORD)v97 + v260);
        v310 = (__int64 *)(v387 + 16 * v97);
        v99 = *v310;
        if ( v95 == *v310 )
        {
          v100 = (__int64 *)v310[1];
          goto LABEL_124;
        }
        ++v260;
        v261 = v98;
        v98 = (__int64 *)(v387 + 16 * v97);
      }
      if ( !v261 )
        v261 = v98;
      ++v386;
      v262 = v388 + 1;
      if ( 4 * ((int)v388 + 1) >= 3 * v389 )
        goto LABEL_361;
      if ( v389 - HIDWORD(v388) - v262 <= v389 >> 3 )
        goto LABEL_362;
LABEL_357:
      LODWORD(v388) = v262;
      if ( *v261 != -8 )
        --HIDWORD(v388);
      *v261 = v95;
      v100 = 0;
      v261[1] = 0;
LABEL_124:
      v101 = *v100;
      if ( v100[1] != *v100 )
      {
        v102 = 0;
        v103 = 0;
        while ( 1 )
        {
          v108 = *(_QWORD **)(v101 + 8 * v103);
          v109 = v397;
          v406 = v108;
          v110 = (__int64)(v108[1] - *v108) >> 3;
          LODWORD(v398) = v110;
          v111 = v110;
          if ( (unsigned int)v54 < v110 )
            LODWORD(v54) = v110;
          if ( !v397 )
            break;
          v104 = (v397 - 1) & (37 * v110);
          v105 = (int *)(v395 + 32LL * v104);
          v106 = *v105;
          if ( *v105 != (_DWORD)v110 )
          {
            v115 = 1;
            v112 = 0;
            while ( v106 != -1 )
            {
              if ( !v112 && v106 == -2 )
                v112 = v105;
              v104 = (v397 - 1) & (v115 + v104);
              v105 = (int *)(v395 + 32LL * v104);
              v106 = *v105;
              if ( *v105 == v111 )
                goto LABEL_127;
              ++v115;
            }
            if ( !v112 )
              v112 = v105;
            ++v394;
            v113 = v396 + 1;
            if ( 4 * ((int)v396 + 1) < 3 * v397 )
            {
              if ( v397 - HIDWORD(v396) - v113 <= v397 >> 3 )
              {
LABEL_137:
                sub_1C53E70((__int64)&v394, v109);
                sub_1C50450((__int64)&v394, (int *)&v398, &v410);
                v112 = (_DWORD *)v410.m128i_i64[0];
                v111 = v398;
                v113 = v396 + 1;
              }
              LODWORD(v396) = v113;
              if ( *v112 != -1 )
                --HIDWORD(v396);
              *v112 = v111;
              v107 = 0;
              v114 = v112 + 2;
              *v114 = 0;
              v114[1] = 0;
              v114[2] = 0;
              goto LABEL_141;
            }
LABEL_136:
            v109 = 2 * v397;
            goto LABEL_137;
          }
LABEL_127:
          v107 = (_BYTE *)*((_QWORD *)v105 + 2);
          if ( v107 != *((_BYTE **)v105 + 3) )
          {
            if ( v107 )
            {
              *(_QWORD *)v107 = v108;
              v107 = (_BYTE *)*((_QWORD *)v105 + 2);
            }
            *((_QWORD *)v105 + 2) = v107 + 8;
            goto LABEL_131;
          }
          v114 = v105 + 2;
LABEL_141:
          sub_1C50BF0((__int64)v114, v107, &v406);
LABEL_131:
          v101 = *v100;
          v103 = (unsigned int)++v102;
          if ( v102 == (v100[1] - *v100) >> 3 )
            goto LABEL_142;
        }
        ++v394;
        goto LABEL_136;
      }
LABEL_142:
      if ( v341 == v94 )
      {
        v13 = v335;
        goto LABEL_152;
      }
      v93 = v366;
      v94 += 8;
    }
    ++v386;
LABEL_361:
    v96 = 2 * v389;
LABEL_362:
    sub_1C532A0((__int64)&v386, v96);
    sub_1C507A0((__int64)&v386, (__int64 *)&v402, &v410);
    v261 = (__int64 *)v410.m128i_i64[0];
    v95 = (__int64)v402;
    v262 = v388 + 1;
    goto LABEL_357;
  }
LABEL_152:
  v398 = 0;
  v399 = 0;
  v400 = 0;
  v401 = 0;
  v402 = 0;
  v403 = 0;
  v404 = 0;
  v405 = 0;
  LODWORD(v369[0]) = v54;
  if ( (unsigned int)v54 <= 1 )
    goto LABEL_211;
  v323 = v13;
  while ( 2 )
  {
    if ( !v397 )
      goto LABEL_159;
    v119 = (v397 - 1) & (37 * v54);
    v120 = (int *)(v395 + 32LL * v119);
    v121 = *v120;
    if ( (_DWORD)v54 != *v120 )
    {
      for ( j = 1; ; j = v258 )
      {
        if ( v121 == -1 )
          goto LABEL_159;
        v258 = j + 1;
        v119 = (v397 - 1) & (j + v119);
        v120 = (int *)(v395 + 32LL * v119);
        v121 = *v120;
        if ( *v120 == (_DWORD)v54 )
          break;
      }
    }
    if ( v120 == (int *)(v395 + 32LL * v397) )
      goto LABEL_159;
    v122 = sub_1C50450((__int64)&v394, (int *)v369, &v410);
    v123 = v410.m128i_i64[0];
    if ( !v122 )
    {
      v116 = v397;
      ++v394;
      v117 = v396 + 1;
      if ( 4 * ((int)v396 + 1) >= 3 * v397 )
      {
        v116 = 2 * v397;
      }
      else if ( v397 - HIDWORD(v396) - v117 > v397 >> 3 )
      {
LABEL_156:
        LODWORD(v396) = v117;
        if ( *(_DWORD *)v123 != -1 )
          --HIDWORD(v396);
        v118 = v369[0];
        *(_QWORD *)(v123 + 8) = 0;
        *(_QWORD *)(v123 + 16) = 0;
        *(_DWORD *)v123 = v118;
        *(_QWORD *)(v123 + 24) = 0;
        goto LABEL_159;
      }
      sub_1C53E70((__int64)&v394, v116);
      sub_1C50450((__int64)&v394, (int *)v369, &v410);
      v123 = v410.m128i_i64[0];
      v117 = v396 + 1;
      goto LABEL_156;
    }
    v124 = *(_QWORD *)(v410.m128i_i64[0] + 8);
    v125 = (*(_QWORD *)(v410.m128i_i64[0] + 16) - v124) >> 3;
    if ( !(_DWORD)v125 )
      goto LABEL_159;
    v126 = 0;
    v127 = 8LL * (unsigned int)(v125 - 1);
    while ( 2 )
    {
      v128 = *(__int64 **)(v124 + v126);
      v406 = 0;
      v371 = (char *)v128;
      v129 = (_QWORD **)*v128;
      v130 = (_QWORD **)v128[1];
      v131 = (__int64 *)(v399 + 8LL * (unsigned int)v401);
      if ( v130 == (_QWORD **)*v128 )
      {
        v133 = 0;
      }
      else
      {
        v132 = v401 - 1;
        do
        {
          v134 = *v129;
          if ( (_DWORD)v401 )
          {
            v135 = v132 & (((unsigned int)*v134 >> 9) ^ ((unsigned int)*v134 >> 4));
            v136 = (__int64 *)(v399 + 8LL * v135);
            v137 = *v136;
            if ( *v134 == *v136 )
            {
LABEL_168:
              if ( v136 != v131 )
                goto LABEL_180;
            }
            else
            {
              v138 = 1;
              while ( v137 != -8 )
              {
                v135 = v132 & (v138 + v135);
                v342 = v138 + 1;
                v136 = (__int64 *)(v399 + 8LL * v135);
                v137 = *v136;
                if ( *v134 == *v136 )
                  goto LABEL_168;
                v138 = v342;
              }
            }
          }
          v133 = v134[1];
          ++v129;
          v406 = (_QWORD *)v133;
        }
        while ( v130 != v129 );
      }
      if ( (_DWORD)v401 )
      {
        v139 = (v401 - 1) & (((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4));
        v140 = (__int64 *)(v399 + 8LL * v139);
        v141 = *v140;
        if ( v133 == *v140 )
        {
LABEL_179:
          if ( v140 != v131 )
            goto LABEL_180;
        }
        else
        {
          v142 = 1;
          while ( v141 != -8 )
          {
            v302 = v142 + 1;
            v139 = (v401 - 1) & (v142 + v139);
            v140 = (__int64 *)(v399 + 8LL * v139);
            v141 = *v140;
            if ( v133 == *v140 )
              goto LABEL_179;
            v142 = v302;
          }
        }
      }
      v143 = sub_1C50240((__int64)&v402, (__int64 *)&v371, &v410);
      v144 = (char **)v410.m128i_i64[0];
      if ( !v143 )
      {
        v255 = v405;
        v402 = (_QWORD *)((char *)v402 + 1);
        v256 = v404 + 1;
        if ( 4 * ((int)v404 + 1) >= (unsigned int)(3 * v405) )
        {
          v255 = 2 * v405;
        }
        else if ( (int)v405 - HIDWORD(v404) - v256 > (unsigned int)v405 >> 3 )
        {
LABEL_335:
          LODWORD(v404) = v256;
          if ( *v144 != (char *)-8LL )
            --HIDWORD(v404);
          *v144 = v371;
          goto LABEL_185;
        }
        sub_1C52C60((__int64)&v402, v255);
        sub_1C50240((__int64)&v402, (__int64 *)&v371, &v410);
        v144 = (char **)v410.m128i_i64[0];
        v256 = v404 + 1;
        goto LABEL_335;
      }
LABEL_185:
      v145 = *(__int64 ***)v371;
      if ( *((_QWORD *)v371 + 1) == *(_QWORD *)v371 )
        goto LABEL_197;
      v146 = v127;
      v147 = *((_QWORD *)v371 + 1);
      v148 = v146;
      while ( 2 )
      {
        while ( 2 )
        {
          v152 = v401;
          v153 = *v145;
          if ( !(_DWORD)v401 )
          {
            ++v398;
            goto LABEL_191;
          }
          v149 = (v401 - 1) & (((unsigned int)*v153 >> 9) ^ ((unsigned int)*v153 >> 4));
          v150 = (__int64 *)(v399 + 8LL * v149);
          v151 = *v150;
          if ( *v153 == *v150 )
          {
LABEL_188:
            if ( (__int64 **)v147 == ++v145 )
              goto LABEL_196;
            continue;
          }
          break;
        }
        v320 = 1;
        v154 = 0;
        while ( v151 != -8 )
        {
          if ( v154 || v151 != -16 )
            v150 = v154;
          v149 = (v401 - 1) & (v320 + v149);
          v151 = *(_QWORD *)(v399 + 8LL * v149);
          if ( *v153 == v151 )
            goto LABEL_188;
          ++v320;
          v154 = v150;
          v150 = (__int64 *)(v399 + 8LL * v149);
        }
        if ( !v154 )
          v154 = v150;
        ++v398;
        v155 = v400 + 1;
        if ( 4 * ((int)v400 + 1) >= (unsigned int)(3 * v401) )
        {
LABEL_191:
          v319 = v148;
          v152 = 2 * v401;
LABEL_192:
          sub_1C52AC0((__int64)&v398, v152);
          sub_1C503A0((__int64)&v398, v153, &v410);
          v154 = (__int64 *)v410.m128i_i64[0];
          v148 = v319;
          v155 = v400 + 1;
          goto LABEL_193;
        }
        if ( (int)v401 - HIDWORD(v400) - v155 <= (unsigned int)v401 >> 3 )
        {
          v319 = v148;
          goto LABEL_192;
        }
LABEL_193:
        LODWORD(v400) = v155;
        if ( *v154 != -8 )
          --HIDWORD(v400);
        ++v145;
        *v154 = *v153;
        if ( (__int64 **)v147 != v145 )
          continue;
        break;
      }
LABEL_196:
      v127 = v148;
LABEL_197:
      v156 = sub_1C503A0((__int64)&v398, (__int64 *)&v406, &v410);
      v157 = (_QWORD *)v410.m128i_i64[0];
      if ( !v156 )
      {
        v158 = v401;
        ++v398;
        v159 = v400 + 1;
        if ( 4 * ((int)v400 + 1) >= (unsigned int)(3 * v401) )
        {
          v158 = 2 * v401;
        }
        else if ( (int)v401 - HIDWORD(v400) - v159 > (unsigned int)v401 >> 3 )
        {
          goto LABEL_200;
        }
        sub_1C52AC0((__int64)&v398, v158);
        sub_1C503A0((__int64)&v398, (__int64 *)&v406, &v410);
        v157 = (_QWORD *)v410.m128i_i64[0];
        v159 = v400 + 1;
LABEL_200:
        LODWORD(v400) = v159;
        if ( *v157 != -8 )
          --HIDWORD(v400);
        *v157 = v406;
      }
LABEL_180:
      if ( v127 != v126 )
      {
        v124 = *(_QWORD *)(v123 + 8);
        v126 += 8;
        continue;
      }
      break;
    }
LABEL_159:
    LODWORD(v54) = LODWORD(v369[0]) - 1;
    LODWORD(v369[0]) = v54;
    if ( (unsigned int)v54 > 1 )
      continue;
    break;
  }
  v13 = v323;
LABEL_211:
  v406 = 0;
  v407 = 0;
  v408 = 0;
  v409 = 0;
  if ( dword_4FBD100 )
    sub_1C66A70((_QWORD *)v13, &v366, (__int64)&v386, (__int64)&v402, (__int64)&v406);
  v68 = v412;
  sub_1C51BF0(*(_QWORD **)(v13 + 280));
  v160 = v367;
  *(_QWORD *)(v13 + 280) = 0;
  *(_QWORD *)(v13 + 288) = v13 + 272;
  *(_QWORD *)(v13 + 296) = v13 + 272;
  v161 = v366;
  *(_QWORD *)(v13 + 304) = 0;
  v410 = 0u;
  v162 = (__int64)(v160 - v161) >> 3;
  v411.m128i_i64[0] = 0;
  v411.m128i_i32[2] = 0;
  v353 = 0;
  if ( !(_DWORD)v162 )
    goto LABEL_283;
  v321 = 0;
  v68 = v412;
  v163 = v369;
  v314 = 8LL * (unsigned int)(v162 - 1);
  while ( 2 )
  {
    v164 = (char *)&v407[4 * v409];
    v356 = *(_QWORD *)(v161 + v321);
    v369[0] = v356;
    v165 = sub_1C50850((__int64)&v406, v163, &v371);
    v168 = v371;
    if ( !v165 )
      v168 = (char *)&v407[4 * v409];
    if ( v164 == v168 )
    {
LABEL_218:
      v169 = v389;
      if ( v389 )
        goto LABEL_219;
LABEL_243:
      ++v386;
LABEL_244:
      v169 *= 2;
LABEL_245:
      sub_1C532A0((__int64)&v386, v169);
      sub_1C507A0((__int64)&v386, &v356, &v371);
      v187 = v371;
      v170 = v356;
      v188 = v388 + 1;
      goto LABEL_348;
    }
    v184 = sub_1C50850((__int64)&v406, &v356, &v371);
    v324 = v371;
    if ( v184 )
    {
      v189 = *((_QWORD *)v371 + 1);
      v190 = 0xAAAAAAAAAAAAAAABLL * ((*((_QWORD *)v371 + 2) - v189) >> 3);
      if ( (_DWORD)v190 )
      {
        v337 = 0;
        v317 = 24LL * (unsigned int)v190;
        while ( 1 )
        {
          v191 = 0;
          v369[0] = 0;
          v369[1] = 0;
          v192 = (__int64 *)(v189 + v337);
          v370 = 0;
          v371 = 0;
          v372 = 0;
          v373 = 0;
          sub_1C62450((_QWORD *)v13, (_QWORD *)(v189 + v337), (__int64)v163);
          v193 = *v192;
          v194 = 0;
          if ( v192[1] != *v192 )
          {
            v195 = v68;
            v196 = v192;
            v197 = (__int64)v195;
            do
            {
              sub_1C650B0(
                v13,
                *(__int64 *******)(v193 + 8 * v194),
                v356,
                (_QWORD **)v163,
                (__int64 **)&v371,
                v197,
                a6,
                a7,
                (__int64)&v410,
                a5);
              v193 = *v196;
              v194 = (unsigned int)++v191;
            }
            while ( v191 != (v196[1] - *v196) >> 3 );
            v68 = (_QWORD *)v197;
          }
          if ( v371 )
            j_j___libc_free_0(v371, (char *)v373 - v371);
          if ( v369[0] )
            j_j___libc_free_0(v369[0], v370 - v369[0]);
          v337 += 24;
          if ( v337 == v317 )
            break;
          v189 = *((_QWORD *)v324 + 1);
        }
      }
      goto LABEL_218;
    }
    v185 = v409;
    v406 = (_QWORD *)((char *)v406 + 1);
    v186 = v408 + 1;
    if ( 4 * ((int)v408 + 1) >= 3 * v409 )
    {
      v185 = 2 * v409;
    }
    else if ( v409 - HIDWORD(v408) - v186 > v409 >> 3 )
    {
      goto LABEL_240;
    }
    sub_1C53C20((__int64)&v406, v185);
    sub_1C50850((__int64)&v406, &v356, &v371);
    v324 = v371;
    v186 = v408 + 1;
LABEL_240:
    LODWORD(v408) = v186;
    if ( *(_QWORD *)v324 != -8 )
      --HIDWORD(v408);
    *(_QWORD *)v324 = v356;
    *((_QWORD *)v324 + 1) = 0;
    *((_QWORD *)v324 + 2) = 0;
    *((_QWORD *)v324 + 3) = 0;
    v169 = v389;
    if ( !v389 )
      goto LABEL_243;
LABEL_219:
    v170 = v356;
    LODWORD(v171) = (v169 - 1) & (((unsigned int)v356 >> 9) ^ ((unsigned int)v356 >> 4));
    v172 = (char *)(v387 + 16LL * (unsigned int)v171);
    v173 = *(_QWORD *)v172;
    if ( v356 == *(_QWORD *)v172 )
    {
      v174 = (__int64 *)*((_QWORD *)v172 + 1);
    }
    else
    {
      v259 = 1;
      v187 = 0;
      while ( v173 != -8 )
      {
        if ( v187 || v173 != -16 )
          v172 = v187;
        v171 = (v169 - 1) & ((_DWORD)v171 + v259);
        v309 = (__int64 *)(v387 + 16 * v171);
        v173 = *v309;
        if ( v356 == *v309 )
        {
          v174 = (__int64 *)v309[1];
          goto LABEL_221;
        }
        ++v259;
        v187 = v172;
        v172 = (char *)(v387 + 16 * v171);
      }
      if ( !v187 )
        v187 = v172;
      ++v386;
      v188 = v388 + 1;
      if ( 4 * ((int)v388 + 1) >= 3 * v169 )
        goto LABEL_244;
      if ( v169 - HIDWORD(v388) - v188 <= v169 >> 3 )
        goto LABEL_245;
LABEL_348:
      LODWORD(v388) = v188;
      if ( *(_QWORD *)v187 != -8 )
        --HIDWORD(v388);
      *(_QWORD *)v187 = v170;
      v174 = 0;
      *((_QWORD *)v187 + 1) = 0;
    }
LABEL_221:
    v175 = *v174;
    if ( v174[1] != *v174 )
    {
      v336 = v163;
      v176 = 0;
      v177 = 0;
      do
      {
        if ( (_DWORD)v405 )
        {
          v178 = *(__int64 *******)(v175 + 8 * v176);
          v179 = (v405 - 1) & (((unsigned int)v178 >> 9) ^ ((unsigned int)v178 >> 4));
          v180 = (__int64 *******)(v403 + 8LL * v179);
          v181 = *v180;
          if ( v178 == *v180 )
          {
LABEL_225:
            if ( v180 != (__int64 *******)(v403 + 8LL * (unsigned int)v405) )
            {
              sub_1C637F0(
                v13,
                v178,
                v356,
                &v353,
                v331,
                (__int64)v68,
                a6,
                a7,
                *(double *)a8.m128_u64,
                *(double *)a9.m128_u64,
                v166,
                v167,
                a12,
                a13,
                (__int64)&v410,
                a5);
              v175 = *v174;
            }
          }
          else
          {
            v182 = 1;
            while ( v181 != (__int64 ******)-8LL )
            {
              v183 = v182 + 1;
              v179 = (v405 - 1) & (v182 + v179);
              v180 = (__int64 *******)(v403 + 8LL * v179);
              v181 = *v180;
              if ( v178 == *v180 )
                goto LABEL_225;
              v182 = v183;
            }
          }
        }
        v176 = (unsigned int)++v177;
      }
      while ( v177 != (v174[1] - v175) >> 3 );
      v163 = v336;
    }
    if ( v175 )
      j_j___libc_free_0(v175, v174[2] - v175);
    j_j___libc_free_0(v174, 24);
    if ( v314 != v321 )
    {
      v161 = v366;
      v321 += 8;
      continue;
    }
    break;
  }
  v210 = v163;
  if ( v353 > 0 && dword_4FBD1E0 > 2 )
  {
    v371 = 0;
    v241 = &v390;
    v372 = 0;
    v373 = 0;
    LODWORD(v356) = v315;
    if ( v315 )
    {
      v330 = v68;
      while ( 1 )
      {
        v243 = v391 + 32LL * v393;
        v209 = (unsigned int)sub_1C504F0((__int64)v241, (int *)&v356, v210);
        v244 = v369[0];
        if ( !(_BYTE)v209 )
          v244 = v391 + 32LL * v393;
        if ( v243 == v244 )
          goto LABEL_315;
        v245 = v393;
        if ( !v393 )
          break;
        v242 = v356;
        v209 = (unsigned int)v356;
        v246 = (v393 - 1) & (37 * v356);
        v247 = (int *)(v391 + 32LL * v246);
        v248 = *v247;
        if ( *v247 != (_DWORD)v356 )
        {
          v306 = 1;
          v307 = 0;
          while ( v248 != -1 )
          {
            if ( !v307 && v248 == -2 )
              v307 = v247;
            v246 = (v393 - 1) & (v306 + v246);
            v247 = (int *)(v391 + 32LL * v246);
            v248 = *v247;
            if ( (_DWORD)v356 == *v247 )
              goto LABEL_322;
            ++v306;
          }
          if ( v307 )
            v247 = v307;
          ++v390;
          v308 = v392 + 1;
          if ( 4 * ((int)v392 + 1) < 3 * v393 )
          {
            if ( v393 - HIDWORD(v392) - v308 > v393 >> 3 )
            {
LABEL_480:
              LODWORD(v392) = v308;
              if ( *v247 != -1 )
                --HIDWORD(v392);
              *v247 = v209;
              *((_QWORD *)v247 + 1) = 0;
              *((_QWORD *)v247 + 2) = 0;
              *((_QWORD *)v247 + 3) = 0;
LABEL_315:
              v242 = v356;
              goto LABEL_316;
            }
LABEL_485:
            sub_1C53A10((__int64)v241, v245);
            sub_1C504F0((__int64)v241, (int *)&v356, v210);
            v247 = (int *)v369[0];
            v209 = (unsigned int)v356;
            v308 = v392 + 1;
            goto LABEL_480;
          }
LABEL_484:
          v245 = 2 * v393;
          goto LABEL_485;
        }
LABEL_322:
        v249 = *((_QWORD *)v247 + 1);
        v250 = (*((_QWORD *)v247 + 2) - v249) >> 5;
        if ( !(_DWORD)v250 )
          goto LABEL_316;
        srcc = v241;
        v251 = 0;
        v252 = 32LL * (unsigned int)(v250 - 1);
        while ( 1 )
        {
          v253 = v372;
          v254 = (const __m128i *)(v251 + v249);
          if ( v372 != v373 )
            break;
          sub_1C50A60((const __m128i **)&v371, v372, v254);
          if ( v251 == v252 )
            goto LABEL_330;
LABEL_327:
          v249 = *((_QWORD *)v247 + 1);
          v251 += 32;
        }
        if ( v372 )
        {
          *v372 = _mm_loadu_si128(v254);
          v253[1] = _mm_loadu_si128(v254 + 1);
          v253 = v372;
        }
        v372 = v253 + 2;
        if ( v251 != v252 )
          goto LABEL_327;
LABEL_330:
        v241 = srcc;
        v242 = v356;
LABEL_316:
        LODWORD(v356) = v242 - 1;
        if ( v242 == 1 )
        {
          v198 = v372;
          v199 = (__m128i *)v371;
          v68 = v330;
          v200 = (char *)v372 - v371;
          v201 = v13 + 32;
          v202 = ((char *)v372 - v371) >> 5;
          v203 = v202;
          if ( (_DWORD)v202 )
          {
            srca = (_QWORD *)v13;
            v343 = v210;
            v332 = 32LL * (unsigned int)(v202 - 1);
            v204 = 0;
            v205 = v13 + 32;
            while ( 1 )
            {
              v207 = srca[23];
              v369[0] = *(_QWORD *)v199[v204 / 0x10 + 1].m128i_i64[0];
              v206 = v369[0];
              v208 = sub_1C57390(v205, v343);
              *((_DWORD *)v208 + 2) = sub_1CCB2B0(v207, v206);
              if ( v204 == v332 )
                break;
              v199 = (__m128i *)v371;
              v204 += 32LL;
            }
            v198 = v372;
            v199 = (__m128i *)v371;
            v201 = v205;
            v13 = (__int64)srca;
            v210 = v343;
            v68 = v330;
            v200 = (char *)v372 - v371;
            v203 = ((char *)v372 - v371) >> 5;
          }
          if ( v199 != v198 )
          {
            _BitScanReverse64(&v203, v203);
            v344 = v201;
            sub_1C585C0(v199, v198, 2LL * (int)(63 - (v203 ^ 0x3F)), v201, v209, v201);
            if ( v200 <= 512 )
            {
              sub_1C579A0(v199, v198, v344);
            }
            else
            {
              v211 = v199 + 32;
              sub_1C579A0(v199, v199 + 32, v344);
              if ( &v199[32] != v198 )
              {
                do
                {
                  v212 = v211;
                  v211 += 2;
                  sub_1C574B0(v212, v344);
                }
                while ( v198 != v211 );
              }
            }
            v198 = (__m128i *)v371;
            v203 = ((char *)v372 - v371) >> 5;
          }
          if ( !(_DWORD)v203 )
            goto LABEL_281;
          srcb = v210;
          v322 = v68;
          v316 = 32LL * (unsigned int)v203;
          v333 = 0;
          v329 = (_QWORD *)v13;
          v213 = a5;
          while ( 1 )
          {
            v214 = (__int64 *)v198[v333 / 0x10 + 1].m128i_i64[0];
            v215 = *v214;
            v216 = (unsigned int ****)v214[1];
            v318 = v215;
            v351 = v216[1];
            if ( v351 != *v216 )
              break;
LABEL_279:
            v333 += 32LL;
            if ( v316 == v333 )
            {
              v68 = v322;
LABEL_281:
              if ( v198 )
                j_j___libc_free_0(v198, (char *)v373 - (char *)v198);
              goto LABEL_283;
            }
          }
          v217 = *v216;
          while ( 2 )
          {
            if ( !(_DWORD)v401 )
              goto LABEL_277;
            v218 = (__int64 *)*v217;
            v219 = v401 - 1;
            v220 = **v217;
            v221 = (v401 - 1) & (((unsigned int)v220 >> 9) ^ ((unsigned int)v220 >> 4));
            v222 = (unsigned int **)(v399 + 8LL * v221);
            v223 = *v222;
            if ( v220 != *v222 )
            {
              v295 = 1;
              while ( v223 != (unsigned int *)-8LL )
              {
                v296 = v295 + 1;
                v221 = v219 & (v295 + v221);
                v222 = (unsigned int **)(v399 + 8LL * v221);
                v223 = *v222;
                if ( v220 == *v222 )
                  goto LABEL_274;
                v295 = v296;
              }
              goto LABEL_277;
            }
LABEL_274:
            v224 = (__int64 ***)(v399 + 8LL * (unsigned int)v401);
            if ( v224 == (__int64 ***)v222 )
              goto LABEL_277;
            v225 = (__int64 **)v218[1];
            v226 = v219 & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
            v227 = (__int64 ***)(v399 + 8LL * v226);
            v228 = *v227;
            if ( v225 == *v227 )
            {
LABEL_276:
              if ( v224 != v227 )
                goto LABEL_277;
            }
            else
            {
              v276 = 1;
              while ( v228 != (__int64 **)-8LL )
              {
                v303 = v276 + 1;
                v226 = v219 & (v276 + v226);
                v227 = (__int64 ***)(v399 + 8LL * v226);
                v228 = *v227;
                if ( v225 == *v227 )
                  goto LABEL_276;
                v276 = v303;
              }
            }
            sub_1C620D0(v329, v220, v225, &v354, (unsigned __int64 *)&v355, v329[25], 0);
            if ( *(_BYTE *)(v354 + 16) == 18 )
            {
              v277 = *v218;
              if ( v354 == *(_QWORD *)(*(_QWORD *)(**(_QWORD **)*v218 + 16LL) + 40LL) )
                goto LABEL_415;
              goto LABEL_277;
            }
            v277 = *v218;
LABEL_415:
            if ( !v411.m128i_i32[2] )
            {
LABEL_277:
              if ( v351 == ++v217 )
              {
                v198 = (__m128i *)v371;
                goto LABEL_279;
              }
              continue;
            }
            break;
          }
          v278 = (v411.m128i_i32[2] - 1) & (((unsigned int)v277 >> 9) ^ ((unsigned int)v277 >> 4));
          v279 = (__int64 *)(v410.m128i_i64[1] + 16LL * v278);
          v280 = *v279;
          if ( v277 != *v279 )
          {
            v304 = 1;
            while ( v280 != -8 )
            {
              v305 = v304 + 1;
              v278 = (v411.m128i_i32[2] - 1) & (v304 + v278);
              v279 = (__int64 *)(v410.m128i_i64[1] + 16LL * v278);
              v280 = *v279;
              if ( v277 == *v279 )
                goto LABEL_417;
              v304 = v305;
            }
            goto LABEL_277;
          }
LABEL_417:
          if ( v279 == (__int64 *)(v410.m128i_i64[1] + 16LL * v411.m128i_u32[2]) )
            goto LABEL_277;
          v281 = sub_1C538E0((__int64)&v410, v218);
          v282 = sub_145DC80(v329[23], v281[1]);
          v325 = sub_13A5B00(v329[23], v282, v318, 0, 0);
          v283 = v218[1];
          v284 = *(__int64 **)v283;
          v345 = *(_QWORD *)v283 + 8LL * *(unsigned int *)(v283 + 8);
          if ( *(_QWORD *)v283 != v345 )
          {
            v313 = v218;
            v311 = v217;
            v285 = 0;
            while ( 2 )
            {
              v288 = v355;
              v289 = *v284;
              if ( *(_BYTE *)(v355 + 16) == 18 )
              {
                if ( v355 == *(_QWORD *)(*(_QWORD *)(v289 + 16) + 40LL) )
                  v288 = *(_QWORD *)(v289 + 16);
                else
                  v288 = sub_157EBA0(v355);
              }
              if ( !v285 )
                v285 = sub_38767A0(v322, v325, 0, v288);
              v290 = v285;
              if ( !sub_14560B0(*(_QWORD *)v289) )
              {
                v291 = sub_13A5B00(v329[23], v325, *(_QWORD *)v289, 0, 0);
                v290 = sub_38767A0(v322, v291, 0, *(_QWORD *)(v289 + 16));
              }
              v356 = *(_QWORD *)(v289 + 16);
              v286 = sub_1463A20(v213, &v356, srcb);
              v287 = (_QWORD *)v369[0];
              if ( v286 )
                goto LABEL_421;
              v292 = *(_DWORD *)(v213 + 24);
              v293 = *(_DWORD *)(v213 + 16);
              ++*(_QWORD *)v213;
              v294 = v293 + 1;
              if ( 4 * v294 >= 3 * v292 )
              {
                v292 *= 2;
              }
              else if ( v292 - *(_DWORD *)(v213 + 20) - v294 > v292 >> 3 )
              {
                goto LABEL_431;
              }
              sub_1467110(v213, v292);
              sub_1463A20(v213, &v356, srcb);
              v287 = (_QWORD *)v369[0];
              v294 = *(_DWORD *)(v213 + 16) + 1;
LABEL_431:
              *(_DWORD *)(v213 + 16) = v294;
              if ( *v287 != -8 )
                --*(_DWORD *)(v213 + 20);
              *v287 = v356;
LABEL_421:
              ++v284;
              sub_1C51F30(*(__int64 ****)(v289 + 24), v290, *(_QWORD *)(v289 + 16));
              if ( (__int64 *)v345 == v284 )
              {
                v297 = v285;
                v217 = v311;
                v218 = v313;
                goto LABEL_440;
              }
              continue;
            }
          }
          v297 = 0;
LABEL_440:
          sub_1C538E0((__int64)&v410, v218 + 1)[1] = v297;
          v298 = sub_1C503A0((__int64)&v398, v218 + 1, srcb);
          v299 = (_QWORD *)v369[0];
          if ( v298 )
            goto LABEL_277;
          v300 = v401;
          ++v398;
          v301 = v400 + 1;
          if ( 4 * ((int)v400 + 1) >= (unsigned int)(3 * v401) )
          {
            v300 = 2 * v401;
          }
          else if ( (int)v401 - HIDWORD(v400) - v301 > (unsigned int)v401 >> 3 )
          {
LABEL_443:
            LODWORD(v400) = v301;
            if ( *v299 != -8 )
              --HIDWORD(v400);
            *v299 = v218[1];
            goto LABEL_277;
          }
          sub_1C52AC0((__int64)&v398, v300);
          sub_1C503A0((__int64)&v398, v218 + 1, srcb);
          v299 = (_QWORD *)v369[0];
          v301 = v400 + 1;
          goto LABEL_443;
        }
      }
      ++v390;
      goto LABEL_484;
    }
  }
LABEL_283:
  if ( (_DWORD)v376 )
  {
    v266 = v375;
    v267 = 2LL * v377;
    v268 = &v375[v267];
    if ( v375 != &v375[v267] )
    {
      while ( 1 )
      {
        v269 = v266;
        if ( *v266 != -16 && *v266 != -8 )
          break;
        v266 += 2;
        if ( v268 == v266 )
          goto LABEL_284;
      }
      if ( v268 != v266 )
      {
        while ( 1 )
        {
          v270 = (_QWORD *)v269[1];
          v271 = *v270;
          v272 = (__int64)(v270[1] - *v270) >> 3;
          if ( !(_DWORD)v272 )
            goto LABEL_397;
          v273 = 0;
          v274 = 8LL * (unsigned int)(v272 - 1);
          while ( 1 )
          {
            v275 = *(_QWORD *)(v271 + v273);
            if ( v275 )
            {
              j_j___libc_free_0(v275, 16);
              v270 = (_QWORD *)v269[1];
            }
            if ( v273 == v274 )
              break;
            v271 = *v270;
            v273 += 8;
          }
          if ( v270 )
            break;
LABEL_400:
          v269 += 2;
          if ( v269 != v268 )
          {
            while ( *v269 == -8 || *v269 == -16 )
            {
              v269 += 2;
              if ( v268 == v269 )
                goto LABEL_284;
            }
            if ( v269 != v268 )
              continue;
          }
          goto LABEL_284;
        }
        v271 = *v270;
LABEL_397:
        if ( v271 )
        {
          v352 = v270;
          j_j___libc_free_0(v271, v270[2] - v271);
          v270 = v352;
        }
        j_j___libc_free_0(v270, 24);
        goto LABEL_400;
      }
    }
  }
LABEL_284:
  j___libc_free_0(v410.m128i_i64[1]);
  if ( v409 )
  {
    v229 = v407;
    v230 = &v407[4 * v409];
    do
    {
      if ( *v229 != -16 && *v229 != -8 )
      {
        v231 = (_QWORD *)v229[2];
        v232 = (_QWORD *)v229[1];
        if ( v231 != v232 )
        {
          do
          {
            if ( *v232 )
              j_j___libc_free_0(*v232, v232[2] - *v232);
            v232 += 3;
          }
          while ( v231 != v232 );
          v232 = (_QWORD *)v229[1];
        }
        if ( v232 )
          j_j___libc_free_0(v232, v229[3] - (_QWORD)v232);
      }
      v229 += 4;
    }
    while ( v230 != v229 );
  }
  j___libc_free_0(v407);
  j___libc_free_0(v403);
  j___libc_free_0(v399);
  if ( v397 )
  {
    v233 = v395;
    v234 = v395 + 32LL * v397;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v233 <= 0xFFFFFFFD )
        {
          v235 = *(_QWORD *)(v233 + 8);
          if ( v235 )
            break;
        }
        v233 += 32;
        if ( v234 == v233 )
          goto LABEL_302;
      }
      v236 = *(_QWORD *)(v233 + 24);
      v233 += 32;
      j_j___libc_free_0(v235, v236 - v235);
    }
    while ( v234 != v233 );
  }
LABEL_302:
  j___libc_free_0(v395);
  if ( v393 )
  {
    v237 = v391;
    v238 = v391 + 32LL * v393;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v237 <= 0xFFFFFFFD )
        {
          v239 = *(_QWORD *)(v237 + 8);
          if ( v239 )
            break;
        }
        v237 += 32;
        if ( v238 == v237 )
          goto LABEL_308;
      }
      v240 = *(_QWORD *)(v237 + 24);
      v237 += 32;
      j_j___libc_free_0(v239, v240 - v239);
    }
    while ( v238 != v237 );
  }
LABEL_308:
  j___libc_free_0(v391);
  if ( v366 )
    j_j___libc_free_0(v366, v368 - v366);
  j___libc_free_0(v387);
LABEL_80:
  if ( v363 )
    j_j___libc_free_0(v363, (char *)v365 - (char *)v363);
  if ( v385 )
  {
    v69 = v383;
    v70 = &v383[4 * v385];
    do
    {
      if ( *v69 != -16 && *v69 != -8 )
      {
        v71 = v69[1];
        if ( v71 )
          j_j___libc_free_0(v71, v69[3] - v71);
      }
      v69 += 4;
    }
    while ( v70 != v69 );
  }
  j___libc_free_0(v383);
  if ( v360 )
    j_j___libc_free_0(v360, (char *)v362 - (char *)v360);
  if ( v381 )
  {
    v72 = v379;
    v73 = &v379[4 * v381];
    do
    {
      if ( *v72 != -16 && *v72 != -8 )
      {
        v74 = v72[1];
        if ( v74 )
          j_j___libc_free_0(v74, v72[3] - v74);
      }
      v72 += 4;
    }
    while ( v73 != v72 );
  }
  j___libc_free_0(v379);
  if ( v357 )
    j_j___libc_free_0(v357, v359 - v357);
  j___libc_free_0(v375);
  sub_194A930((__int64)v68);
  return v312;
}
