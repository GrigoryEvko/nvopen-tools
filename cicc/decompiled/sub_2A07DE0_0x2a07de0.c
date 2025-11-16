// Function: sub_2A07DE0
// Address: 0x2a07de0
//
__int64 __fastcall sub_2A07DE0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 a7,
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 **v18; // r14
  __int64 **v19; // rbx
  const char *v20; // r13
  const char **v21; // rax
  const char **v22; // rdx
  const char **v23; // rbx
  const char **v24; // r11
  __int64 v25; // r14
  __int64 v26; // rax
  unsigned int v27; // edi
  __int64 v28; // rdx
  __int64 v29; // rsi
  unsigned int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // rcx
  const char **v33; // r15
  int v34; // r11d
  __int64 *v35; // rax
  unsigned int v36; // ecx
  __int64 *v37; // rdx
  __int64 v38; // rdi
  const char *v39; // r12
  int v40; // edx
  __int64 v41; // rcx
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // rax
  int v47; // edx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  int v50; // edx
  unsigned __int64 v51; // rax
  char *v52; // rdx
  const char *v53; // rax
  char *v54; // rdx
  const char *v55; // rax
  char *v56; // rdx
  unsigned __int64 v57; // rax
  int v58; // edx
  unsigned __int64 v59; // rax
  bool v60; // cf
  __int64 v61; // rdx
  __int64 v62; // rcx
  const char **v63; // r8
  const char **v64; // rdi
  unsigned __int64 v65; // rax
  const char **v66; // rbx
  int v67; // eax
  __int64 v68; // rax
  __int64 v69; // r14
  unsigned int v70; // r15d
  const char *v71; // r13
  __int64 v72; // rsi
  _QWORD *v73; // rax
  int v74; // eax
  __int64 v75; // r9
  __int64 v76; // rcx
  __int64 v77; // rdx
  unsigned int v78; // r14d
  __int64 v79; // rsi
  __int64 v80; // r8
  _QWORD *v81; // rax
  _QWORD *v82; // rdx
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  double v85; // xmm0_8
  __int64 v86; // r8
  __int64 v87; // r9
  unsigned int v88; // esi
  const char **v89; // rax
  __int64 v90; // r9
  unsigned int v91; // edx
  __int64 v92; // rbx
  __int64 v93; // rax
  unsigned __int64 v94; // rdx
  __int64 *v95; // rcx
  __int64 v96; // r13
  int v97; // esi
  __int64 v98; // rdi
  __int64 v99; // rsi
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rcx
  __int64 v103; // rax
  unsigned __int64 v104; // rdx
  char v105; // cl
  __int64 v106; // rbx
  __int64 v107; // rax
  __int64 v108; // r14
  __int64 v109; // rbx
  void **v110; // r12
  __int64 v111; // rcx
  __int64 *v112; // rdx
  unsigned __int64 v113; // r13
  unsigned __int64 v114; // rdi
  __int64 v115; // rax
  __int64 v116; // r9
  __int64 v117; // rdx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // r13
  __int64 v121; // rax
  unsigned __int64 v122; // rdx
  __int64 v123; // rsi
  int v124; // ecx
  __int64 v125; // rdi
  int v126; // ecx
  unsigned int v127; // edx
  __int64 *v128; // rax
  __int64 v129; // r10
  _QWORD *v130; // r15
  __int64 v131; // rax
  __int64 v132; // rax
  unsigned int v133; // edx
  __int64 v134; // rcx
  unsigned int v135; // eax
  __int64 v136; // rax
  __int64 v137; // rdx
  unsigned int v138; // eax
  __int64 v139; // r15
  int v140; // edx
  __int64 v141; // rdx
  __int64 v142; // r13
  unsigned int v143; // ecx
  __int64 v144; // rsi
  unsigned int v145; // edx
  unsigned __int64 v146; // rsi
  __int64 v147; // rdx
  __int64 v148; // r10
  _QWORD *v149; // rdx
  __int64 v150; // rax
  int v151; // edx
  unsigned int v152; // edx
  unsigned __int64 v153; // rcx
  __int64 v154; // rdx
  unsigned __int64 *v155; // rdx
  __int64 *v156; // rax
  __int64 v157; // r9
  __int64 *v158; // rbx
  __int64 v159; // r13
  __int64 *v160; // r15
  __int64 v161; // rdi
  unsigned __int64 v162; // rax
  int v163; // edx
  unsigned __int8 *v164; // r12
  unsigned __int8 *v165; // rax
  _QWORD *v166; // rax
  __int64 v167; // r12
  unsigned __int64 v168; // rax
  int v169; // edx
  unsigned __int8 *v170; // r14
  unsigned __int8 *v171; // rax
  int v172; // eax
  unsigned int v173; // r13d
  int v174; // r12d
  __int64 v175; // rbx
  unsigned int v176; // edx
  __int64 v177; // rax
  __int64 v178; // rcx
  unsigned int v179; // eax
  __int64 v180; // r12
  __int64 v181; // r13
  _QWORD *v182; // rdi
  __int64 v183; // rsi
  _QWORD *v184; // rax
  int v185; // r8d
  __int64 v186; // r9
  size_t v187; // rdx
  __int64 v188; // r8
  __int64 v189; // rax
  unsigned __int64 v190; // rcx
  __int64 v191; // rdx
  __int64 j; // rbx
  _QWORD *v193; // rax
  __int64 v194; // r9
  __int64 v195; // r15
  __int64 v196; // rsi
  __int64 v197; // rax
  __int64 v198; // rax
  __int64 v199; // r10
  __int64 v200; // rsi
  _QWORD *v201; // rax
  _QWORD *v202; // rdx
  unsigned __int64 *v203; // rax
  _QWORD *v204; // rdx
  unsigned __int64 v205; // rcx
  unsigned __int64 *v206; // rdi
  unsigned __int64 v207; // rax
  __int64 *v208; // rax
  _QWORD *v209; // rax
  __int64 v210; // r10
  _QWORD *v211; // rdi
  __int64 v212; // rax
  __int64 v213; // rsi
  __int64 v214; // rax
  __int64 v215; // rax
  _QWORD *v216; // rax
  __int64 v217; // rdx
  __int64 v218; // rax
  __int64 v219; // r12
  __int64 v220; // rax
  __int64 v221; // rdx
  __int64 v222; // rbx
  __int64 v223; // rsi
  __int64 v224; // rax
  __int64 v225; // rax
  __int64 v226; // r13
  __int64 v227; // rsi
  _QWORD *v228; // rax
  _QWORD *v229; // rdx
  __int64 v230; // rax
  int v231; // r11d
  char *v232; // rdi
  unsigned int v233; // edx
  char *v234; // rcx
  __int64 v235; // rsi
  int v236; // edi
  __int64 v237; // rdx
  unsigned __int64 *v238; // rdi
  void **v239; // rdx
  char *v240; // rdx
  __int64 v241; // rdx
  int v242; // eax
  int v243; // eax
  unsigned int v244; // ecx
  __int64 v245; // rax
  __int64 v246; // rcx
  __int64 v247; // rcx
  __int64 v248; // rax
  _QWORD *v249; // rax
  unsigned __int64 v250; // rax
  int v251; // edx
  unsigned __int64 v252; // rax
  __int64 *v253; // rsi
  __int64 v254; // rax
  const char *v255; // rax
  __int64 v256; // rdx
  __int64 v257; // rax
  char *v258; // r14
  __int64 v259; // rax
  char *v260; // rbx
  __int64 v261; // rdx
  __int64 v262; // rax
  unsigned int v263; // edi
  char *v264; // rdx
  __int64 v265; // r8
  int v266; // r11d
  int v267; // r11d
  unsigned int v268; // edx
  __int64 v269; // rsi
  int v270; // eax
  int v271; // r8d
  __int64 v272; // rsi
  unsigned int v273; // edx
  __int64 v274; // rsi
  __int64 v275; // r8
  _QWORD *v276; // rcx
  __int64 *v277; // rax
  __int64 *v278; // r12
  __int64 v279; // rsi
  __int64 *v280; // rbx
  _QWORD *v281; // rax
  __int64 v282; // rax
  __int64 v283; // r8
  __int64 v284; // rsi
  unsigned int v285; // edx
  unsigned int *v286; // rdi
  __int64 *v287; // rbx
  char *v288; // r12
  __int64 v289; // rax
  unsigned __int64 *v290; // rax
  unsigned __int64 v291; // rdx
  unsigned __int64 *v292; // r14
  __int64 v293; // rax
  __int64 v294; // rax
  __int64 v295; // rax
  _QWORD *v296; // r12
  _QWORD *v297; // rbx
  __int64 v298; // rsi
  __int64 v299; // rdx
  __int64 v300; // r10
  __int64 v301; // rbx
  __int64 v302; // r12
  unsigned __int64 v303; // r8
  unsigned __int64 v304; // rdi
  __int64 *v305; // rax
  __int64 v306; // rdx
  __int64 *v307; // r14
  __int64 v308; // r13
  __int64 *v309; // rbx
  __int64 v310; // rax
  __int64 v311; // rdx
  unsigned int v312; // eax
  unsigned int v313; // ecx
  __int64 v314; // r15
  __int64 v315; // rdx
  unsigned int v316; // eax
  __int64 v317; // r13
  __int64 v318; // rcx
  _QWORD *v319; // rdi
  __int64 v320; // rsi
  _QWORD *v321; // rax
  __int64 v322; // r8
  int v323; // r9d
  __int64 v324; // rcx
  size_t v325; // rdx
  __int64 v326; // r9
  __int64 v327; // rax
  unsigned __int64 v328; // rcx
  __int64 v329; // rdx
  __int64 v330; // r12
  __int64 v331; // rax
  int v332; // edi
  __int64 v333; // rdx
  unsigned int v334; // r10d
  __int64 v335; // rsi
  _QWORD *v336; // rax
  _QWORD *v337; // rdx
  unsigned int v338; // edx
  __int64 v339; // rax
  __int64 *v340; // rax
  __int64 v341; // rcx
  __int64 v342; // r8
  __int64 v343; // r9
  __int64 v344; // rax
  __int64 *v345; // rbx
  __int64 *v346; // r12
  unsigned __int64 v347; // rdi
  unsigned __int64 v348; // rdi
  void **v350; // rax
  __int64 v351; // r12
  __int64 v352; // rbx
  void **v353; // r13
  unsigned __int64 v354; // r8
  unsigned __int64 v355; // rdi
  __int64 *v356; // rax
  __int64 *v357; // r12
  __int64 v358; // rdi
  __int64 *v359; // rbx
  unsigned int *v360; // rsi
  __int64 v361; // rdx
  int v362; // edi
  __int64 *v363; // rsi
  __int64 v364; // rax
  unsigned __int64 v365; // rdx
  int v366; // edi
  __int64 v367; // rdx
  __int64 v368; // rsi
  int v369; // edi
  int v370; // edi
  int v371; // r10d
  __int64 v372; // rdi
  unsigned __int64 v373; // [rsp+8h] [rbp-3F8h]
  unsigned __int64 v374; // [rsp+8h] [rbp-3F8h]
  void **v375; // [rsp+10h] [rbp-3F0h]
  __int64 v376; // [rsp+10h] [rbp-3F0h]
  __int64 *v377; // [rsp+30h] [rbp-3D0h]
  __int64 v378; // [rsp+38h] [rbp-3C8h]
  __int64 v380; // [rsp+48h] [rbp-3B8h]
  __int64 v381; // [rsp+48h] [rbp-3B8h]
  __int64 v382; // [rsp+48h] [rbp-3B8h]
  __int64 v383; // [rsp+50h] [rbp-3B0h]
  __int64 v384; // [rsp+58h] [rbp-3A8h]
  __int64 v385; // [rsp+60h] [rbp-3A0h]
  unsigned __int8 *v386; // [rsp+68h] [rbp-398h]
  __int64 v387; // [rsp+78h] [rbp-388h]
  unsigned int v388; // [rsp+78h] [rbp-388h]
  __int64 v389; // [rsp+78h] [rbp-388h]
  __int64 v390; // [rsp+80h] [rbp-380h]
  __int64 *v391; // [rsp+90h] [rbp-370h]
  size_t v392; // [rsp+98h] [rbp-368h]
  __int64 v394; // [rsp+A8h] [rbp-358h]
  __int64 v395; // [rsp+B0h] [rbp-350h]
  __int64 v396; // [rsp+B8h] [rbp-348h]
  int v397; // [rsp+C0h] [rbp-340h]
  __int64 v398; // [rsp+C0h] [rbp-340h]
  __int64 v399; // [rsp+C8h] [rbp-338h]
  __int64 *v400; // [rsp+C8h] [rbp-338h]
  int *v403; // [rsp+E0h] [rbp-320h]
  __int64 v404; // [rsp+E8h] [rbp-318h]
  int v405; // [rsp+E8h] [rbp-318h]
  __int64 v406; // [rsp+F0h] [rbp-310h]
  int v407; // [rsp+F0h] [rbp-310h]
  __int64 v408; // [rsp+F0h] [rbp-310h]
  __int64 v409; // [rsp+F0h] [rbp-310h]
  unsigned int v410; // [rsp+F0h] [rbp-310h]
  __int64 v411; // [rsp+F0h] [rbp-310h]
  char *v412; // [rsp+F0h] [rbp-310h]
  __int64 v413; // [rsp+F0h] [rbp-310h]
  int v414; // [rsp+F8h] [rbp-308h]
  __int64 v415; // [rsp+F8h] [rbp-308h]
  __int64 *v416; // [rsp+F8h] [rbp-308h]
  __int64 *v417; // [rsp+100h] [rbp-300h]
  unsigned __int64 v418; // [rsp+100h] [rbp-300h]
  char *v419; // [rsp+100h] [rbp-300h]
  char *v420; // [rsp+100h] [rbp-300h]
  char *v421; // [rsp+100h] [rbp-300h]
  __int64 v422; // [rsp+100h] [rbp-300h]
  int v424; // [rsp+110h] [rbp-2F0h]
  const char *v425; // [rsp+110h] [rbp-2F0h]
  unsigned int v426; // [rsp+110h] [rbp-2F0h]
  int *v427; // [rsp+110h] [rbp-2F0h]
  __int64 v428; // [rsp+110h] [rbp-2F0h]
  __int64 v429; // [rsp+110h] [rbp-2F0h]
  __int64 v430; // [rsp+110h] [rbp-2F0h]
  __int64 v431; // [rsp+118h] [rbp-2E8h]
  __int64 *v432; // [rsp+118h] [rbp-2E8h]
  __int64 *v433; // [rsp+118h] [rbp-2E8h]
  __int64 v434; // [rsp+118h] [rbp-2E8h]
  __int64 v435; // [rsp+118h] [rbp-2E8h]
  __int64 v436; // [rsp+118h] [rbp-2E8h]
  __int64 v437; // [rsp+118h] [rbp-2E8h]
  __int64 *v438; // [rsp+120h] [rbp-2E0h]
  int v439; // [rsp+120h] [rbp-2E0h]
  __int64 v440; // [rsp+120h] [rbp-2E0h]
  _QWORD *v441; // [rsp+120h] [rbp-2E0h]
  __int64 v442; // [rsp+120h] [rbp-2E0h]
  __int64 v443; // [rsp+120h] [rbp-2E0h]
  __int64 v444; // [rsp+120h] [rbp-2E0h]
  unsigned __int8 *v445; // [rsp+120h] [rbp-2E0h]
  __int64 v446; // [rsp+120h] [rbp-2E0h]
  __int64 v447; // [rsp+128h] [rbp-2D8h]
  __int64 v448; // [rsp+128h] [rbp-2D8h]
  __int64 v449; // [rsp+130h] [rbp-2D0h] BYREF
  __int64 *v450; // [rsp+138h] [rbp-2C8h]
  __int64 v451; // [rsp+140h] [rbp-2C0h]
  unsigned int v452; // [rsp+148h] [rbp-2B8h]
  __int64 v453; // [rsp+150h] [rbp-2B0h] BYREF
  __int64 *v454; // [rsp+158h] [rbp-2A8h]
  __int64 v455; // [rsp+160h] [rbp-2A0h]
  unsigned int v456; // [rsp+168h] [rbp-298h]
  __int64 v457[4]; // [rsp+170h] [rbp-290h] BYREF
  unsigned int v458; // [rsp+190h] [rbp-270h]
  unsigned __int64 v459; // [rsp+198h] [rbp-268h]
  __int64 v460; // [rsp+1A0h] [rbp-260h]
  __int64 *v461; // [rsp+1B0h] [rbp-250h] BYREF
  __int64 v462; // [rsp+1B8h] [rbp-248h] BYREF
  __int64 v463; // [rsp+1C0h] [rbp-240h] BYREF
  __int64 v464; // [rsp+1C8h] [rbp-238h]
  __int64 v465; // [rsp+1D0h] [rbp-230h]
  const char *v466; // [rsp+1F0h] [rbp-210h] BYREF
  __int64 v467; // [rsp+1F8h] [rbp-208h] BYREF
  const char *v468; // [rsp+200h] [rbp-200h] BYREF
  __int64 v469; // [rsp+208h] [rbp-1F8h]
  void **i; // [rsp+210h] [rbp-1F0h]
  char *v471; // [rsp+230h] [rbp-1D0h] BYREF
  __int64 v472; // [rsp+238h] [rbp-1C8h]
  _BYTE v473[48]; // [rsp+240h] [rbp-1C0h] BYREF
  __int64 *v474; // [rsp+270h] [rbp-190h] BYREF
  __int64 v475; // [rsp+278h] [rbp-188h]
  _BYTE v476[64]; // [rsp+280h] [rbp-180h] BYREF
  _BYTE *v477; // [rsp+2C0h] [rbp-140h] BYREF
  __int64 v478; // [rsp+2C8h] [rbp-138h]
  _BYTE v479[48]; // [rsp+2D0h] [rbp-130h] BYREF
  _BYTE *v480; // [rsp+300h] [rbp-100h] BYREF
  __int64 v481; // [rsp+308h] [rbp-F8h]
  _BYTE v482[48]; // [rsp+310h] [rbp-F0h] BYREF
  const char **v483; // [rsp+340h] [rbp-C0h] BYREF
  __int64 v484; // [rsp+348h] [rbp-B8h] BYREF
  const char *v485; // [rsp+350h] [rbp-B0h] BYREF
  unsigned int v486; // [rsp+358h] [rbp-A8h] BYREF
  __int16 v487; // [rsp+360h] [rbp-A0h]
  _QWORD *v488; // [rsp+368h] [rbp-98h]
  unsigned int v489; // [rsp+378h] [rbp-88h]
  char v490; // [rsp+380h] [rbp-80h]
  _BYTE *v491; // [rsp+388h] [rbp-78h] BYREF
  __int64 v492; // [rsp+390h] [rbp-70h]
  _BYTE v493[104]; // [rsp+398h] [rbp-68h] BYREF

  v8 = a1;
  sub_D33BC0((__int64)v457, a1);
  sub_D4E470(v457, a3);
  v394 = **(_QWORD **)(a1 + 32);
  v431 = sub_D4B130(a1);
  v395 = sub_D47930(a1);
  v474 = (__int64 *)v476;
  v475 = 0x400000000LL;
  sub_D47670(a1, (__int64)&v474);
  v12 = *(__int64 **)(a1 + 40);
  v13 = *(__int64 **)(a1 + 32);
  v449 = 0;
  v450 = 0;
  v451 = 0;
  v452 = 0;
  v417 = v12;
  v438 = v13;
  if ( v12 != v13 )
  {
    while ( 1 )
    {
      v14 = *v438;
      if ( *v438 )
      {
        v15 = (unsigned int)(*(_DWORD *)(v14 + 44) + 1);
        v16 = *(_DWORD *)(v14 + 44) + 1;
      }
      else
      {
        v15 = 0;
        v16 = 0;
      }
      if ( *(_DWORD *)(a5 + 32) <= v16 )
      {
        v483 = &v485;
        v484 = 0x1000000000LL;
        BUG();
      }
      v17 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v15);
      v483 = &v485;
      v484 = 0x1000000000LL;
      v18 = *(__int64 ***)(v17 + 24);
      v19 = &v18[*(unsigned int *)(v17 + 32)];
      if ( v19 != v18 )
      {
        while ( 1 )
        {
          v20 = (const char *)**v18;
          if ( *(_BYTE *)(v8 + 84) )
          {
            v21 = *(const char ***)(v8 + 64);
            v22 = &v21[*(unsigned int *)(v8 + 76)];
            if ( v21 == v22 )
              goto LABEL_110;
            while ( v20 != *v21 )
            {
              if ( v22 == ++v21 )
                goto LABEL_110;
            }
LABEL_11:
            if ( v19 == ++v18 )
              goto LABEL_12;
          }
          else
          {
            if ( sub_C8CA60(v8 + 56, **v18) )
              goto LABEL_11;
LABEL_110:
            v93 = (unsigned int)v484;
            v94 = (unsigned int)v484 + 1LL;
            if ( v94 > HIDWORD(v484) )
            {
              sub_C8D5F0((__int64)&v483, &v485, v94, 8u, v10, v11);
              v93 = (unsigned int)v484;
            }
            ++v18;
            v483[v93] = v20;
            LODWORD(v484) = v484 + 1;
            if ( v19 == v18 )
            {
LABEL_12:
              v23 = v483;
              v24 = &v483[(unsigned int)v484];
              goto LABEL_13;
            }
          }
        }
      }
      v23 = &v485;
      v24 = &v485;
LABEL_13:
      v25 = *(_QWORD *)(*(_QWORD *)(v14 + 72) + 80LL);
      if ( v25 )
        v25 -= 24;
      if ( v14 != v25 && v395 != v25 )
      {
        v26 = 0;
        v27 = *(_DWORD *)(a5 + 32);
        v28 = (unsigned int)(*(_DWORD *)(v14 + 44) + 1);
        if ( (unsigned int)v28 < v27 )
          v26 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v28);
        if ( v395 )
        {
          v424 = *(_DWORD *)(v395 + 44);
          v29 = (unsigned int)(v424 + 1);
          v30 = v424 + 1;
        }
        else
        {
          v29 = 0;
          v30 = 0;
        }
        v31 = 0;
        if ( v27 > v30 )
          v31 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v29);
        while ( v26 != v31 )
        {
          if ( *(_DWORD *)(v26 + 16) < *(_DWORD *)(v31 + 16) )
          {
            v32 = v26;
            v26 = v31;
            v31 = v32;
          }
          v26 = *(_QWORD *)(v26 + 8);
        }
        v25 = *(_QWORD *)v31;
      }
      if ( v23 != v24 )
        break;
LABEL_40:
      if ( v24 != &v485 )
        _libc_free((unsigned __int64)v24);
      if ( v417 == ++v438 )
        goto LABEL_43;
    }
    v406 = v8;
    v33 = v24;
    while ( 1 )
    {
      v39 = *v23;
      if ( !v452 )
        break;
      v11 = v452 - 1;
      v34 = 1;
      v10 = (__int64)v450;
      v35 = 0;
      v36 = v11 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v37 = &v450[2 * v36];
      v38 = *v37;
      if ( v39 == (const char *)*v37 )
      {
LABEL_31:
        ++v23;
        v37[1] = v25;
        if ( v33 == v23 )
          goto LABEL_39;
      }
      else
      {
        while ( v38 != -4096 )
        {
          if ( v38 == -8192 && !v35 )
            v35 = v37;
          v36 = v11 & (v34 + v36);
          v37 = &v450[2 * v36];
          v38 = *v37;
          if ( v39 == (const char *)*v37 )
            goto LABEL_31;
          ++v34;
        }
        if ( !v35 )
          v35 = v37;
        ++v449;
        v40 = v451 + 1;
        if ( 4 * ((int)v451 + 1) < 3 * v452 )
        {
          if ( v452 - HIDWORD(v451) - v40 <= v452 >> 3 )
          {
            sub_22E02D0((__int64)&v449, v452);
            if ( !v452 )
            {
LABEL_680:
              LODWORD(v451) = v451 + 1;
              BUG();
            }
            v11 = (__int64)v450;
            v95 = 0;
            LODWORD(v96) = (v452 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v40 = v451 + 1;
            v97 = 1;
            v35 = &v450[2 * (unsigned int)v96];
            v10 = *v35;
            if ( v39 != (const char *)*v35 )
            {
              while ( v10 != -4096 )
              {
                if ( v10 == -8192 && !v95 )
                  v95 = v35;
                v96 = (v452 - 1) & ((_DWORD)v96 + v97);
                v35 = &v450[2 * v96];
                v10 = *v35;
                if ( v39 == (const char *)*v35 )
                  goto LABEL_36;
                ++v97;
              }
              if ( v95 )
                v35 = v95;
            }
          }
          goto LABEL_36;
        }
LABEL_34:
        sub_22E02D0((__int64)&v449, 2 * v452);
        if ( !v452 )
          goto LABEL_680;
        v10 = v452 - 1;
        v40 = v451 + 1;
        LODWORD(v41) = v10 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v35 = &v450[2 * (unsigned int)v41];
        v11 = *v35;
        if ( v39 != (const char *)*v35 )
        {
          v362 = 1;
          v363 = 0;
          while ( v11 != -4096 )
          {
            if ( !v363 && v11 == -8192 )
              v363 = v35;
            v41 = (unsigned int)v10 & ((_DWORD)v41 + v362);
            v35 = &v450[2 * v41];
            v11 = *v35;
            if ( v39 == (const char *)*v35 )
              goto LABEL_36;
            ++v362;
          }
          if ( v363 )
            v35 = v363;
        }
LABEL_36:
        LODWORD(v451) = v40;
        if ( *v35 != -4096 )
          --HIDWORD(v451);
        *v35 = (__int64)v39;
        ++v23;
        v42 = v35 + 1;
        *v42 = 0;
        *v42 = v25;
        if ( v33 == v23 )
        {
LABEL_39:
          v8 = v406;
          v24 = v483;
          goto LABEL_40;
        }
      }
    }
    ++v449;
    goto LABEL_34;
  }
LABEL_43:
  v43 = *(_QWORD *)(v394 + 72);
  v487 = 257;
  v385 = v43;
  v44 = sub_F41C30(v431, v394, a5, a3, 0, (void **)&v483);
  v487 = 257;
  v399 = v44;
  v45 = v44 + 48;
  v46 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v46 == v45 )
  {
    v48 = 0;
  }
  else
  {
    if ( !v46 )
      BUG();
    v47 = *(unsigned __int8 *)(v46 - 24);
    v48 = v46 - 24;
    if ( (unsigned int)(v47 - 30) >= 0xB )
      v48 = 0;
  }
  v396 = sub_F36960(v399, (__int64 *)(v48 + 24), 0, a5, a3, 0, (void **)&v483, 0);
  v487 = 257;
  v49 = *(_QWORD *)(v396 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v49 == v396 + 48 )
  {
    v51 = 0;
  }
  else
  {
    if ( !v49 )
      BUG();
    v50 = *(unsigned __int8 *)(v49 - 24);
    v51 = v49 - 24;
    if ( (unsigned int)(v50 - 30) >= 0xB )
      v51 = 0;
  }
  v386 = (unsigned __int8 *)sub_F36960(v396, (__int64 *)(v51 + 24), 0, a5, a3, 0, (void **)&v483, 0);
  v483 = (const char **)sub_BD5D20(v394);
  v487 = 773;
  v484 = (__int64)v52;
  v485 = ".peel.begin";
  sub_BD6B50((unsigned __int8 *)v399, (const char **)&v483);
  v53 = sub_BD5D20(v394);
  v487 = 773;
  v483 = (const char **)v53;
  v484 = (__int64)v54;
  v485 = ".peel.next";
  sub_BD6B50((unsigned __int8 *)v396, (const char **)&v483);
  v55 = sub_BD5D20(v431);
  v487 = 773;
  v483 = (const char **)v55;
  v484 = (__int64)v56;
  v485 = ".peel.newph";
  sub_BD6B50(v386, (const char **)&v483);
  v57 = *(_QWORD *)(v395 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v57 == v395 + 48 )
  {
    v390 = 0;
  }
  else
  {
    if ( !v57 )
      BUG();
    v58 = *(unsigned __int8 *)(v57 - 24);
    v59 = v57 - 24;
    v60 = (unsigned int)(v58 - 30) < 0xB;
    v61 = 0;
    if ( v60 )
      v61 = v59;
    v390 = v61;
  }
  v453 = 0;
  v461 = &v463;
  v462 = 0x600000000LL;
  v454 = 0;
  v455 = 0;
  v456 = 0;
  sub_D46D90(v8, (__int64)&v461);
  v432 = &v461[(unsigned int)v462];
  if ( v461 != v432 )
  {
    v447 = (__int64)v461;
    while ( 1 )
    {
      v65 = *(_QWORD *)(*(_QWORD *)v447 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v65 == *(_QWORD *)v447 + 48LL )
        goto LABEL_131;
      if ( !v65 )
        BUG();
      v66 = (const char **)(v65 - 24);
      if ( (unsigned int)*(unsigned __int8 *)(v65 - 24) - 30 > 0xA )
      {
LABEL_131:
        v467 = 0xC00000000LL;
        v466 = (const char *)&v468;
        sub_BC8C10(0, (__int64)&v466);
        v64 = (const char **)v466;
        goto LABEL_59;
      }
      v467 = 0xC00000000LL;
      v466 = (const char *)&v468;
      if ( !(unsigned __int8)sub_BC8C10((__int64)v66, (__int64)&v466) )
        goto LABEL_58;
      v67 = sub_B46E30((__int64)v66);
      v64 = (const char **)v466;
      v407 = v67;
      v425 = &v466[4 * (unsigned int)v467];
      if ( v67 )
        break;
LABEL_59:
      if ( v64 != &v468 )
        _libc_free((unsigned __int64)v64);
      v447 += 8;
      if ( v432 == (__int64 *)v447 )
      {
        v432 = v461;
        goto LABEL_134;
      }
    }
    v68 = v8 + 56;
    v69 = v8;
    v414 = 0;
    v70 = 0;
    v71 = v466;
    v439 = 0;
    v404 = v68;
    while ( 1 )
    {
      if ( v425 == v71 )
      {
LABEL_76:
        v8 = v69;
        if ( !v439 )
          goto LABEL_58;
        v472 = 0xC00000000LL;
        v471 = v473;
        v426 = v467;
        v74 = sub_B46E30((__int64)v66);
        v76 = v426;
        v397 = v74;
        v77 = v426;
        v403 = (int *)&v466[4 * v426];
        if ( !v74 )
          goto LABEL_90;
        v427 = (int *)v466;
        v78 = 0;
        while ( 2 )
        {
          if ( v403 == v427 )
          {
LABEL_89:
            v77 = (unsigned int)v467;
LABEL_90:
            v478 = 0xC00000000LL;
            v477 = v479;
            if ( (_DWORD)v77 )
              sub_2A045D0((__int64)&v477, (char **)&v466, v77, v76, (__int64)&v477, v75);
            v481 = 0xC00000000LL;
            v480 = v482;
            if ( (_DWORD)v472 )
              sub_2A045D0((__int64)&v480, &v471, v77, v76, (__int64)&v477, v75);
            v87 = (unsigned int)v478;
            v483 = v66;
            v484 = (__int64)&v486;
            v485 = (const char *)0xC00000000LL;
            if ( (_DWORD)v478 )
              sub_2A044F0((__int64)&v484, (__int64)&v477, v77, v76, (__int64)&v477, (unsigned int)v478);
            v492 = 0xC00000000LL;
            v491 = v493;
            if ( !(_DWORD)v481 )
            {
              v88 = v456;
              if ( v456 )
                goto LABEL_98;
LABEL_639:
              ++v453;
LABEL_640:
              sub_2A07790((__int64)&v453, 2 * v88);
              if ( v456 )
              {
                v89 = v483;
                v90 = (__int64)v454;
                v62 = (v456 - 1) & (((unsigned int)v483 >> 9) ^ ((unsigned int)v483 >> 4));
                v92 = (__int64)&v454[17 * v62];
                v367 = (unsigned int)(v455 + 1);
                v63 = *(const char ***)v92;
                if ( v483 != *(const char ***)v92 )
                {
                  v370 = 1;
                  v368 = 0;
                  while ( v63 != (const char **)-4096LL )
                  {
                    if ( v63 == (const char **)-8192LL && !v368 )
                      v368 = v92;
                    v62 = (v456 - 1) & (v370 + (_DWORD)v62);
                    v92 = (__int64)&v454[17 * (unsigned int)v62];
                    v63 = *(const char ***)v92;
                    if ( v483 == *(const char ***)v92 )
                      goto LABEL_622;
                    ++v370;
                  }
LABEL_631:
                  if ( v368 )
                    v92 = v368;
                }
                goto LABEL_622;
              }
LABEL_676:
              LODWORD(v455) = v455 + 1;
              BUG();
            }
            sub_2A044F0((__int64)&v491, (__int64)&v480, v77, v76, (unsigned int)v481, v87);
            v88 = v456;
            if ( !v456 )
              goto LABEL_639;
LABEL_98:
            v89 = v483;
            v90 = v88 - 1;
            v91 = v90 & (((unsigned int)v483 >> 9) ^ ((unsigned int)v483 >> 4));
            v62 = 17LL * v91;
            v92 = (__int64)&v454[17 * v91];
            v63 = *(const char ***)v92;
            if ( v483 != *(const char ***)v92 )
            {
              v366 = 1;
              v62 = 0;
              while ( v63 != (const char **)-4096LL )
              {
                if ( v63 == (const char **)-8192LL && !v62 )
                  v62 = v92;
                v371 = v366 + 1;
                v372 = (unsigned int)v90 & (v91 + v366);
                v91 = v372;
                v92 = (__int64)&v454[17 * v372];
                v63 = *(const char ***)v92;
                if ( v483 == *(const char ***)v92 )
                  goto LABEL_99;
                v366 = v371;
              }
              if ( v62 )
                v92 = v62;
              ++v453;
              v367 = (unsigned int)(v455 + 1);
              if ( 4 * (int)v367 >= 3 * v88 )
                goto LABEL_640;
              v62 = v88 - HIDWORD(v455) - (unsigned int)v367;
              if ( (unsigned int)v62 > v88 >> 3 )
                goto LABEL_622;
              sub_2A07790((__int64)&v453, v88);
              if ( !v456 )
                goto LABEL_676;
              v89 = v483;
              v90 = (__int64)v454;
              v62 = (v456 - 1) & (((unsigned int)v483 >> 9) ^ ((unsigned int)v483 >> 4));
              v368 = 0;
              v92 = (__int64)&v454[17 * v62];
              v367 = (unsigned int)(v455 + 1);
              v369 = 1;
              v63 = *(const char ***)v92;
              if ( v483 != *(const char ***)v92 )
              {
                while ( v63 != (const char **)-4096LL )
                {
                  if ( !v368 && v63 == (const char **)-8192LL )
                    v368 = v92;
                  v62 = (v456 - 1) & (v369 + (_DWORD)v62);
                  v92 = (__int64)&v454[17 * (unsigned int)v62];
                  v63 = *(const char ***)v92;
                  if ( v483 == *(const char ***)v92 )
                    goto LABEL_622;
                  ++v369;
                }
                goto LABEL_631;
              }
LABEL_622:
              LODWORD(v455) = v367;
              if ( *(_QWORD *)v92 != -4096 )
                --HIDWORD(v455);
              *(_QWORD *)v92 = v89;
              *(_QWORD *)(v92 + 8) = v92 + 24;
              *(_QWORD *)(v92 + 16) = 0xC00000000LL;
              if ( (_DWORD)v485 )
                sub_2A045D0(v92 + 8, (char **)&v484, v367, v62, (__int64)v63, v90);
              *(_QWORD *)(v92 + 80) = 0xC00000000LL;
              *(_QWORD *)(v92 + 72) = v92 + 88;
              if ( (_DWORD)v492 )
                sub_2A044F0(v92 + 72, (__int64)&v491, v367, v62, (__int64)v63, v90);
            }
LABEL_99:
            if ( v491 != v493 )
              _libc_free((unsigned __int64)v491);
            if ( (unsigned int *)v484 != &v486 )
              _libc_free(v484);
            if ( v480 != v482 )
              _libc_free((unsigned __int64)v480);
            if ( v477 != v479 )
              _libc_free((unsigned __int64)v477);
            if ( v471 != v473 )
              _libc_free((unsigned __int64)v471);
LABEL_58:
            v64 = (const char **)v466;
            goto LABEL_59;
          }
          v79 = sub_B46EC0((__int64)v66, v78);
          if ( *(_BYTE *)(v8 + 84) )
          {
            v81 = *(_QWORD **)(v8 + 64);
            v82 = &v81[*(unsigned int *)(v8 + 76)];
            if ( v81 == v82 )
            {
LABEL_605:
              v364 = (unsigned int)v472;
              v76 = HIDWORD(v472);
              v365 = (unsigned int)v472 + 1LL;
              if ( v365 > HIDWORD(v472) )
              {
                sub_C8D5F0((__int64)&v471, v473, v365, 4u, v80, v75);
                v364 = (unsigned int)v472;
              }
              *(_DWORD *)&v471[4 * v364] = 0;
              LODWORD(v472) = v472 + 1;
              goto LABEL_88;
            }
            while ( *v81 != v79 )
            {
              if ( v82 == ++v81 )
                goto LABEL_605;
            }
          }
          else if ( !sub_C8CA60(v8 + 56, v79) )
          {
            goto LABEL_605;
          }
          v76 = HIDWORD(v472);
          v83 = (unsigned int)v472;
          v84 = (unsigned int)v472 + 1LL;
          v85 = (double)*v427 / (double)v439 * (double)v414;
          v86 = (unsigned int)(int)v85;
          if ( v84 > HIDWORD(v472) )
          {
            sub_C8D5F0((__int64)&v471, v473, v84, 4u, v86, v75);
            v83 = (unsigned int)v472;
            LODWORD(v86) = (int)v85;
          }
          *(_DWORD *)&v471[4 * v83] = v86;
          LODWORD(v472) = v472 + 1;
LABEL_88:
          ++v427;
          if ( ++v78 == v397 )
            goto LABEL_89;
          continue;
        }
      }
      v72 = sub_B46EC0((__int64)v66, v70);
      if ( *(_BYTE *)(v69 + 84) )
      {
        v73 = *(_QWORD **)(v69 + 64);
        v62 = (__int64)&v73[*(unsigned int *)(v69 + 76)];
        if ( v73 == (_QWORD *)v62 )
          goto LABEL_479;
        while ( *v73 != v72 )
        {
          if ( (_QWORD *)v62 == ++v73 )
            goto LABEL_479;
        }
      }
      else if ( !sub_C8CA60(v404, v72) )
      {
LABEL_479:
        v414 += *(_DWORD *)v71;
        goto LABEL_75;
      }
      v439 += *(_DWORD *)v71;
LABEL_75:
      ++v70;
      v71 += 4;
      if ( v70 == v407 )
        goto LABEL_76;
    }
  }
LABEL_134:
  if ( v432 != &v463 )
    _libc_free((unsigned __int64)v432);
  v98 = *(_QWORD *)(v8 + 32);
  v99 = *(_QWORD *)(v8 + 40) - v98;
  v471 = v473;
  v472 = 0x600000000LL;
  sub_F46230(v98, v99 >> 3, (__int64)&v471, v62, (__int64)v63);
  v102 = a2;
  if ( a2 )
  {
    v405 = 0;
    while ( 2 )
    {
      v483 = 0;
      v486 = 128;
      v477 = v479;
      v478 = 0x800000000LL;
      v103 = sub_C7D670(0x2000, 8);
      v485 = 0;
      v484 = v103;
      v467 = 2;
      v104 = v103 + ((unsigned __int64)v486 << 6);
      v466 = (const char *)&unk_49DD7B0;
      v468 = 0;
      v469 = -4096;
      for ( i = 0; v104 != v103; v103 += 64 )
      {
        if ( v103 )
        {
          v105 = v467;
          *(_QWORD *)(v103 + 16) = 0;
          *(_QWORD *)(v103 + 24) = -4096;
          *(_QWORD *)v103 = &unk_49DD7B0;
          *(_QWORD *)(v103 + 8) = v105 & 6;
          *(_QWORD *)(v103 + 32) = i;
        }
      }
      v490 = 0;
      v384 = (unsigned int)v472;
      v377 = (__int64 *)v471;
      v106 = **(_QWORD **)(v8 + 32);
      v440 = v106;
      v398 = sub_D47930(v8);
      v107 = sub_D4B130(v8);
      v108 = v460;
      v383 = v107;
      v433 = *(__int64 **)v8;
      v428 = *(_QWORD *)(v106 + 72);
      v418 = v459;
      if ( v460 != v459 )
      {
        v415 = v8;
        v109 = a5;
        v110 = (void **)&v483;
        while ( 1 )
        {
          v466 = ".peel";
          LOWORD(i) = 259;
          v120 = sub_F4B360(*(_QWORD *)(v108 - 8), (__int64)v110, (__int64 *)&v466, v428, 0);
          v121 = (unsigned int)v478;
          v122 = (unsigned int)v478 + 1LL;
          if ( v122 > HIDWORD(v478) )
          {
            sub_C8D5F0((__int64)&v477, v479, v122, 8u, v118, v119);
            v121 = (unsigned int)v478;
          }
          *(_QWORD *)&v477[8 * v121] = v120;
          LODWORD(v478) = v478 + 1;
          v123 = *(_QWORD *)(v108 - 8);
          if ( v433 )
          {
            v124 = *(_DWORD *)(a3 + 24);
            v125 = *(_QWORD *)(a3 + 8);
            if ( v124 )
            {
              v126 = v124 - 1;
              v127 = v126 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
              v128 = (__int64 *)(v125 + 16LL * v127);
              v129 = *v128;
              if ( v123 == *v128 )
              {
LABEL_159:
                if ( v415 == v128[1] )
                {
                  sub_D4F330(v433, v120, a3);
                  v123 = *(_QWORD *)(v108 - 8);
                }
              }
              else
              {
                v270 = 1;
                while ( v129 != -4096 )
                {
                  v271 = v270 + 1;
                  v127 = v126 & (v270 + v127);
                  v128 = (__int64 *)(v125 + 16LL * v127);
                  v129 = *v128;
                  if ( v123 == *v128 )
                    goto LABEL_159;
                  v270 = v271;
                }
              }
            }
          }
          v130 = sub_2A07A50((__int64)v110, v123);
          v131 = v130[2];
          if ( v120 != v131 )
          {
            if ( v131 != 0 && v131 != -4096 && v131 != -8192 )
              sub_BD60C0(v130);
            v130[2] = v120;
            if ( v120 != -4096 && v120 != 0 && v120 != -8192 )
              sub_BD73F0((__int64)v130);
          }
          v132 = *(_QWORD *)(v108 - 8);
          v133 = *(_DWORD *)(v109 + 32);
          if ( v440 != v132 )
          {
            if ( v132 )
            {
              v134 = (unsigned int)(*(_DWORD *)(v132 + 44) + 1);
              v135 = *(_DWORD *)(v132 + 44) + 1;
            }
            else
            {
              v134 = 0;
              v135 = 0;
            }
            if ( v135 >= v133 )
              BUG();
            v136 = sub_2A07A50((__int64)v110, **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v109 + 24) + 8 * v134) + 8LL))[2];
            if ( v136 )
            {
              v137 = (unsigned int)(*(_DWORD *)(v136 + 44) + 1);
              v138 = *(_DWORD *)(v136 + 44) + 1;
            }
            else
            {
              v137 = 0;
              v138 = 0;
            }
            if ( v138 >= *(_DWORD *)(v109 + 32) )
            {
              *(_BYTE *)(v109 + 112) = 0;
              v115 = sub_22077B0(0x50u);
              if ( v115 )
              {
                *(_QWORD *)v115 = v120;
                v139 = 0;
                v140 = 0;
                *(_QWORD *)(v115 + 8) = 0;
                goto LABEL_178;
              }
              v139 = 0;
            }
            else
            {
              v139 = *(_QWORD *)(*(_QWORD *)(v109 + 24) + 8 * v137);
              *(_BYTE *)(v109 + 112) = 0;
              v115 = sub_22077B0(0x50u);
              if ( v115 )
              {
                *(_QWORD *)v115 = v120;
                *(_QWORD *)(v115 + 8) = v139;
                if ( v139 )
                  v140 = *(_DWORD *)(v139 + 16) + 1;
                else
                  v140 = 0;
LABEL_178:
                *(_DWORD *)(v115 + 16) = v140;
                *(_QWORD *)(v115 + 24) = v115 + 40;
                *(_QWORD *)(v115 + 32) = 0x400000000LL;
                *(_QWORD *)(v115 + 72) = -1;
              }
            }
            if ( v120 )
            {
              v141 = (unsigned int)(*(_DWORD *)(v120 + 44) + 1);
              v142 = 8 * v141;
            }
            else
            {
              v142 = 0;
              LODWORD(v141) = 0;
            }
            v143 = *(_DWORD *)(v109 + 32);
            if ( v143 > (unsigned int)v141 )
              goto LABEL_144;
            v144 = *(_QWORD *)(v109 + 104);
            v145 = v141 + 1;
            if ( *(_DWORD *)(v144 + 88) >= v145 )
              v145 = *(_DWORD *)(v144 + 88);
            v410 = v145;
            v146 = v145;
            v147 = v143;
            if ( v146 == v143 )
            {
LABEL_144:
              v111 = *(_QWORD *)(v109 + 24);
            }
            else
            {
              v148 = 8 * v146;
              if ( v146 < v143 )
              {
                v111 = *(_QWORD *)(v109 + 24);
                v299 = v111 + 8 * v147;
                v300 = v111 + v148;
                if ( v299 != v300 )
                {
                  v389 = v115;
                  v380 = v109;
                  v301 = v300;
                  v375 = v110;
                  v302 = v299;
                  do
                  {
                    v303 = *(_QWORD *)(v302 - 8);
                    v302 -= 8;
                    if ( v303 )
                    {
                      v304 = *(_QWORD *)(v303 + 24);
                      if ( v304 != v303 + 40 )
                      {
                        v373 = v303;
                        _libc_free(v304);
                        v303 = v373;
                      }
                      j_j___libc_free_0(v303);
                    }
                  }
                  while ( v301 != v302 );
                  v109 = v380;
                  v115 = v389;
                  v110 = v375;
                  v111 = *(_QWORD *)(v380 + 24);
                }
              }
              else
              {
                if ( v146 > *(unsigned int *)(v109 + 36) )
                {
                  v387 = v115;
                  sub_B1B4E0(v109 + 24, v146);
                  v147 = *(unsigned int *)(v109 + 32);
                  v148 = 8 * v146;
                  v115 = v387;
                }
                v111 = *(_QWORD *)(v109 + 24);
                v149 = (_QWORD *)(v111 + 8 * v147);
                if ( v149 != (_QWORD *)(v111 + v148) )
                {
                  do
                  {
                    if ( v149 )
                      *v149 = 0;
                    ++v149;
                  }
                  while ( (_QWORD *)(v111 + v148) != v149 );
                  v111 = *(_QWORD *)(v109 + 24);
                }
              }
              *(_DWORD *)(v109 + 32) = v410;
            }
            v112 = (__int64 *)(v111 + v142);
            v113 = *(_QWORD *)(v111 + v142);
            *v112 = v115;
            if ( !v113 )
              goto LABEL_149;
            goto LABEL_146;
          }
          v150 = (unsigned int)(*(_DWORD *)(v399 + 44) + 1);
          if ( (unsigned int)v150 >= v133 )
            break;
          v139 = *(_QWORD *)(*(_QWORD *)(v109 + 24) + 8 * v150);
          *(_BYTE *)(v109 + 112) = 0;
          v115 = sub_22077B0(0x50u);
          if ( v115 )
          {
            *(_QWORD *)v115 = v120;
            *(_QWORD *)(v115 + 8) = v139;
            if ( v139 )
              v151 = *(_DWORD *)(v139 + 16) + 1;
            else
              v151 = 0;
LABEL_203:
            *(_DWORD *)(v115 + 16) = v151;
            *(_QWORD *)(v115 + 24) = v115 + 40;
            *(_QWORD *)(v115 + 32) = 0x400000000LL;
            *(_QWORD *)(v115 + 72) = -1;
          }
LABEL_204:
          if ( v120 )
          {
            v152 = *(_DWORD *)(v120 + 44) + 1;
            v411 = 8LL * v152;
          }
          else
          {
            v411 = 0;
            v152 = 0;
          }
          v153 = *(unsigned int *)(v109 + 32);
          if ( (unsigned int)v153 > v152 )
            goto LABEL_207;
          v272 = *(_QWORD *)(v109 + 104);
          v273 = v152 + 1;
          if ( *(_DWORD *)(v272 + 88) >= v273 )
            v273 = *(_DWORD *)(v272 + 88);
          v274 = v273;
          v388 = v273;
          if ( v273 == v153 )
          {
LABEL_207:
            v154 = *(_QWORD *)(v109 + 24);
          }
          else
          {
            v275 = 8LL * v273;
            if ( v273 < v153 )
            {
              v154 = *(_QWORD *)(v109 + 24);
              v382 = v154 + v275;
              if ( v154 + 8 * v153 != v154 + v275 )
              {
                v376 = v115;
                v350 = v110;
                v351 = v109;
                v352 = v154 + 8 * v153;
                v353 = v350;
                do
                {
                  v354 = *(_QWORD *)(v352 - 8);
                  v352 -= 8;
                  if ( v354 )
                  {
                    v355 = *(_QWORD *)(v354 + 24);
                    if ( v355 != v354 + 40 )
                    {
                      v374 = v354;
                      _libc_free(v355);
                      v354 = v374;
                    }
                    j_j___libc_free_0(v354);
                  }
                }
                while ( v382 != v352 );
                v109 = v351;
                v115 = v376;
                v110 = v353;
                v154 = *(_QWORD *)(v109 + 24);
              }
            }
            else
            {
              if ( v273 > (unsigned __int64)*(unsigned int *)(v109 + 36) )
              {
                v381 = v115;
                sub_B1B4E0(v109 + 24, v273);
                v153 = *(unsigned int *)(v109 + 32);
                v275 = 8 * v274;
                v115 = v381;
              }
              v154 = *(_QWORD *)(v109 + 24);
              v276 = (_QWORD *)(v154 + 8 * v153);
              if ( v276 != (_QWORD *)(v154 + v275) )
              {
                do
                {
                  if ( v276 )
                    *v276 = 0;
                  ++v276;
                }
                while ( (_QWORD *)(v154 + v275) != v276 );
                v154 = *(_QWORD *)(v109 + 24);
              }
            }
            *(_DWORD *)(v109 + 32) = v388;
          }
          v155 = (unsigned __int64 *)(v411 + v154);
          v113 = *v155;
          *v155 = v115;
          if ( !v113 )
            goto LABEL_149;
LABEL_146:
          v114 = *(_QWORD *)(v113 + 24);
          if ( v114 != v113 + 40 )
          {
            v408 = v115;
            _libc_free(v114);
            v115 = v408;
          }
          v409 = v115;
          j_j___libc_free_0(v113);
          v115 = v409;
LABEL_149:
          if ( v139 )
          {
            v117 = *(unsigned int *)(v139 + 32);
            if ( v117 + 1 > (unsigned __int64)*(unsigned int *)(v139 + 36) )
            {
              v413 = v115;
              sub_C8D5F0(v139 + 24, (const void *)(v139 + 40), v117 + 1, 8u, v117 + 1, v116);
              v117 = *(unsigned int *)(v139 + 32);
              v115 = v413;
            }
            *(_QWORD *)(*(_QWORD *)(v139 + 24) + 8 * v117) = v115;
            ++*(_DWORD *)(v139 + 32);
          }
          v108 -= 8;
          if ( v418 == v108 )
          {
            v8 = v415;
            goto LABEL_211;
          }
        }
        *(_BYTE *)(v109 + 112) = 0;
        v115 = sub_22077B0(0x50u);
        if ( !v115 )
        {
          v139 = 0;
          goto LABEL_204;
        }
        *(_QWORD *)v115 = v120;
        v139 = 0;
        v151 = 0;
        *(_QWORD *)(v115 + 8) = 0;
        goto LABEL_203;
      }
LABEL_211:
      v466 = "Peel";
      LODWORD(v468) = v405;
      LOWORD(i) = 2307;
      sub_CA0F50((__int64 *)&v461, (void **)&v466);
      v391 = v461;
      v392 = v462;
      v156 = (__int64 *)sub_AA48A0(v440);
      sub_F4CD20(v377, v384, (__int64)v477, (unsigned int)v478, v156, v157, v391, v392);
      if ( v461 != &v463 )
        j_j___libc_free_0((unsigned __int64)v461);
      v158 = *(__int64 **)(v8 + 16);
      if ( *(__int64 **)(v8 + 8) != v158 )
      {
        v159 = (__int64)v433;
        v434 = v8;
        v160 = *(__int64 **)(v8 + 8);
        do
        {
          v161 = *v160++;
          sub_F75080(v161, v159, (__int64)&v483, a3, 0);
        }
        while ( v158 != v160 );
        v8 = v434;
      }
      v162 = *(_QWORD *)(v399 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v162 == v399 + 48 )
      {
        v164 = 0;
      }
      else
      {
        if ( !v162 )
          BUG();
        v163 = *(unsigned __int8 *)(v162 - 24);
        v164 = 0;
        v165 = (unsigned __int8 *)(v162 - 24);
        if ( (unsigned int)(v163 - 30) < 0xB )
          v164 = v165;
      }
      v166 = sub_2A07A50((__int64)&v483, v440);
      sub_B46F90(v164, 0, v166[2]);
      v167 = sub_2A07A50((__int64)&v483, v398)[2];
      v168 = *(_QWORD *)(v167 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v168 == v167 + 48 )
      {
        v170 = 0;
      }
      else
      {
        if ( !v168 )
          BUG();
        v169 = *(unsigned __int8 *)(v168 - 24);
        v170 = 0;
        v171 = (unsigned __int8 *)(v168 - 24);
        if ( (unsigned int)(v169 - 30) < 0xB )
          v170 = v171;
      }
      v172 = sub_B46E30((__int64)v170);
      if ( v172 )
      {
        v435 = v167;
        v173 = 0;
        v174 = v172;
        do
        {
          if ( v440 == sub_B46EC0((__int64)v170, v173) )
          {
            v167 = v435;
            sub_B46F90(v170, v173, v396);
            goto LABEL_230;
          }
          ++v173;
        }
        while ( v174 != v173 );
        v167 = v435;
      }
LABEL_230:
      v175 = 0;
      v176 = *(_DWORD *)(a5 + 32);
      v177 = (unsigned int)(*(_DWORD *)(v167 + 44) + 1);
      if ( (unsigned int)v177 < v176 )
        v175 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v177);
      if ( v396 )
      {
        v178 = (unsigned int)(*(_DWORD *)(v396 + 44) + 1);
        v179 = *(_DWORD *)(v396 + 44) + 1;
      }
      else
      {
        v178 = 0;
        v179 = 0;
      }
      if ( v176 <= v179 )
      {
        *(_BYTE *)(a5 + 112) = 0;
        BUG();
      }
      v180 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v178);
      *(_BYTE *)(a5 + 112) = 0;
      v181 = *(_QWORD *)(v180 + 8);
      if ( v175 != v181 )
      {
        v466 = (const char *)v180;
        v182 = *(_QWORD **)(v181 + 24);
        v183 = (__int64)&v182[*(unsigned int *)(v181 + 32)];
        v184 = sub_2A043B0(v182, v183, (__int64 *)&v466);
        v186 = (__int64)(v184 + 1);
        if ( v184 + 1 != (_QWORD *)v183 )
        {
          v187 = v183 - v186;
          v183 = (__int64)(v184 + 1);
          memmove(v184, v184 + 1, v187);
          v185 = *(_DWORD *)(v181 + 32);
        }
        v188 = (unsigned int)(v185 - 1);
        *(_DWORD *)(v181 + 32) = v188;
        *(_QWORD *)(v180 + 8) = v175;
        v189 = *(unsigned int *)(v175 + 32);
        v190 = *(unsigned int *)(v175 + 36);
        if ( v189 + 1 > v190 )
        {
          v183 = v175 + 40;
          sub_C8D5F0(v175 + 24, (const void *)(v175 + 40), v189 + 1, 8u, v188, v186);
          v189 = *(unsigned int *)(v175 + 32);
        }
        v191 = *(_QWORD *)(v175 + 24);
        *(_QWORD *)(v191 + 8 * v189) = v180;
        ++*(_DWORD *)(v175 + 32);
        if ( *(_DWORD *)(v180 + 16) != *(_DWORD *)(*(_QWORD *)(v180 + 8) + 16LL) + 1 )
          sub_2A04730(v180, v183, v191, v190, v188, v186);
      }
      v436 = v8;
      for ( j = *(_QWORD *)(v440 + 56); ; j = *(_QWORD *)(j + 8) )
      {
        if ( !j )
          BUG();
        if ( *(_BYTE *)(j - 24) != 84 )
          break;
        v193 = sub_2A07A50((__int64)&v483, j - 24);
        v194 = j - 24;
        v195 = v193[2];
        if ( v405 )
        {
          v196 = *(_QWORD *)(v195 - 8);
          v197 = 0x1FFFFFFFE0LL;
          if ( (*(_DWORD *)(v195 + 4) & 0x7FFFFFF) != 0 )
          {
            v198 = 0;
            do
            {
              if ( v398 == *(_QWORD *)(v196 + 32LL * *(unsigned int *)(v195 + 72) + 8 * v198) )
              {
                v197 = 32 * v198;
                goto LABEL_251;
              }
              ++v198;
            }
            while ( (*(_DWORD *)(v195 + 4) & 0x7FFFFFF) != (_DWORD)v198 );
            v197 = 0x1FFFFFFFE0LL;
          }
LABEL_251:
          v199 = *(_QWORD *)(v196 + v197);
          if ( *(_BYTE *)v199 > 0x1Cu )
          {
            v200 = *(_QWORD *)(v199 + 40);
            if ( *(_BYTE *)(v436 + 84) )
            {
              v201 = *(_QWORD **)(v436 + 64);
              v202 = &v201[*(unsigned int *)(v436 + 76)];
              if ( v201 != v202 )
              {
                while ( v200 != *v201 )
                {
                  if ( v202 == ++v201 )
                    goto LABEL_266;
                }
LABEL_257:
                v429 = v194;
                v441 = sub_2A07A50(a8, v199);
                v203 = sub_2A07A50((__int64)&v483, v429);
                v204 = v441;
                v205 = v203[2];
                v206 = v203;
                v207 = v441[2];
                if ( v205 != v207 )
                {
                  if ( v205 != 0 && v205 != -4096 && v205 != -8192 )
                  {
                    sub_BD60C0(v206);
                    v204 = v441;
                    v207 = v441[2];
                  }
                  v206[2] = v207;
                  if ( v207 != -4096 && v207 != 0 && v207 != -8192 )
                    sub_BD6050(v206, *v204 & 0xFFFFFFFFFFFFFFF8LL);
                }
                goto LABEL_264;
              }
            }
            else
            {
              v430 = v199;
              v208 = sub_C8CA60(v436 + 56, v200);
              v194 = j - 24;
              v199 = v430;
              if ( v208 )
                goto LABEL_257;
            }
          }
LABEL_266:
          v442 = v199;
          v209 = sub_2A07A50((__int64)&v483, v194);
          v210 = v442;
          v211 = v209;
          v212 = v209[2];
          if ( v442 != v212 )
          {
            if ( v212 != -4096 && v212 != 0 && v212 != -8192 )
            {
              sub_BD60C0(v211);
              v210 = v442;
            }
            v211[2] = v210;
            if ( v210 != -8192 && v210 != -4096 )
              goto LABEL_272;
          }
        }
        else
        {
          if ( (*(_DWORD *)(v195 + 4) & 0x7FFFFFF) != 0 )
          {
            v213 = *(_QWORD *)(v195 - 8);
            v214 = 0;
            do
            {
              if ( v383 == *(_QWORD *)(v213 + 32LL * *(unsigned int *)(v195 + 72) + 8 * v214) )
              {
                v215 = 32 * v214;
                goto LABEL_278;
              }
              ++v214;
            }
            while ( (*(_DWORD *)(v195 + 4) & 0x7FFFFFF) != (_DWORD)v214 );
            v215 = 0x1FFFFFFFE0LL;
          }
          else
          {
            v215 = 0x1FFFFFFFE0LL;
            v213 = *(_QWORD *)(v195 - 8);
          }
LABEL_278:
          v443 = *(_QWORD *)(v213 + v215);
          v216 = sub_2A07A50((__int64)&v483, j - 24);
          v217 = v443;
          v211 = v216;
          v218 = v216[2];
          if ( v218 != v443 )
          {
            if ( v218 != -4096 && v218 != 0 && v218 != -8192 )
            {
              sub_BD60C0(v211);
              v217 = v443;
            }
            v211[2] = v217;
            if ( v217 != -4096 && v217 != 0 && v217 != -8192 )
LABEL_272:
              sub_BD73F0((__int64)v211);
          }
        }
LABEL_264:
        sub_B43D60((_QWORD *)v195);
      }
      v8 = v436;
      v416 = v474;
      v400 = &v474[2 * (unsigned int)v475];
      if ( v474 != v400 )
      {
        do
        {
          v219 = *v416;
          v220 = sub_AA5930(v416[1]);
          v444 = v221;
          v222 = v220;
          while ( v444 != v222 )
          {
            v223 = *(_QWORD *)(v222 - 8);
            v224 = 0x1FFFFFFFE0LL;
            if ( (*(_DWORD *)(v222 + 4) & 0x7FFFFFF) != 0 )
            {
              v225 = 0;
              do
              {
                if ( v219 == *(_QWORD *)(v223 + 32LL * *(unsigned int *)(v222 + 72) + 8 * v225) )
                {
                  v224 = 32 * v225;
                  goto LABEL_294;
                }
                ++v225;
              }
              while ( (*(_DWORD *)(v222 + 4) & 0x7FFFFFF) != (_DWORD)v225 );
              v224 = 0x1FFFFFFFE0LL;
            }
LABEL_294:
            v226 = *(_QWORD *)(v223 + v224);
            if ( *(_BYTE *)v226 > 0x1Cu )
            {
              v227 = *(_QWORD *)(v226 + 40);
              if ( *(_BYTE *)(v436 + 84) )
              {
                v228 = *(_QWORD **)(v436 + 64);
                v229 = &v228[*(unsigned int *)(v436 + 76)];
                if ( v228 != v229 )
                {
                  while ( v227 != *v228 )
                  {
                    if ( v229 == ++v228 )
                      goto LABEL_301;
                  }
LABEL_300:
                  v226 = sub_2A07A50((__int64)&v483, v226)[2];
                }
              }
              else if ( sub_C8CA60(v436 + 56, v227) )
              {
                goto LABEL_300;
              }
            }
LABEL_301:
            v469 = v219;
            v467 = 2;
            v468 = 0;
            if ( v219 != -8192 && v219 != 0 && v219 != -4096 )
              sub_BD73F0((__int64)&v467);
            v466 = (const char *)&unk_49DD7B0;
            i = (void **)&v483;
            if ( !v486 )
            {
              v483 = (const char **)((char *)v483 + 1);
              goto LABEL_305;
            }
            v230 = v469;
            v263 = (v486 - 1) & (((unsigned int)v469 >> 9) ^ ((unsigned int)v469 >> 4));
            v264 = (char *)(v484 + ((unsigned __int64)v263 << 6));
            v265 = *((_QWORD *)v264 + 3);
            if ( v265 != v469 )
            {
              v266 = 1;
              v234 = 0;
              while ( v265 != -4096 )
              {
                if ( v265 == -8192 && !v234 )
                  v234 = v264;
                v263 = (v486 - 1) & (v266 + v263);
                v264 = (char *)(v484 + ((unsigned __int64)v263 << 6));
                v265 = *((_QWORD *)v264 + 3);
                if ( v469 == v265 )
                  goto LABEL_368;
                ++v266;
              }
              if ( !v234 )
                v234 = v264;
              v483 = (const char **)((char *)v483 + 1);
              v236 = (_DWORD)v485 + 1;
              if ( 4 * ((int)v485 + 1) >= 3 * v486 )
              {
LABEL_305:
                sub_CF32C0((__int64)&v483, 2 * v486);
                if ( !v486 )
                  goto LABEL_387;
                v230 = v469;
                v231 = 1;
                v232 = 0;
                v233 = (v486 - 1) & (((unsigned int)v469 >> 9) ^ ((unsigned int)v469 >> 4));
                v234 = (char *)(v484 + ((unsigned __int64)v233 << 6));
                v235 = *((_QWORD *)v234 + 3);
                if ( v469 != v235 )
                {
                  while ( v235 != -4096 )
                  {
                    if ( v235 == -8192 && !v232 )
                      v232 = v234;
                    v233 = (v486 - 1) & (v231 + v233);
                    v234 = (char *)(v484 + ((unsigned __int64)v233 << 6));
                    v235 = *((_QWORD *)v234 + 3);
                    if ( v469 == v235 )
                      goto LABEL_307;
                    ++v231;
                  }
                  goto LABEL_384;
                }
LABEL_307:
                v236 = (_DWORD)v485 + 1;
              }
              else if ( v486 - HIDWORD(v485) - v236 <= v486 >> 3 )
              {
                sub_CF32C0((__int64)&v483, v486);
                if ( v486 )
                {
                  v230 = v469;
                  v267 = 1;
                  v232 = 0;
                  v268 = (v486 - 1) & (((unsigned int)v469 >> 9) ^ ((unsigned int)v469 >> 4));
                  v234 = (char *)(v484 + ((unsigned __int64)v268 << 6));
                  v269 = *((_QWORD *)v234 + 3);
                  if ( v469 != v269 )
                  {
                    while ( v269 != -4096 )
                    {
                      if ( !v232 && v269 == -8192 )
                        v232 = v234;
                      v268 = (v486 - 1) & (v267 + v268);
                      v234 = (char *)(v484 + ((unsigned __int64)v268 << 6));
                      v269 = *((_QWORD *)v234 + 3);
                      if ( v469 == v269 )
                        goto LABEL_307;
                      ++v267;
                    }
LABEL_384:
                    if ( v232 )
                      v234 = v232;
                  }
                  goto LABEL_307;
                }
LABEL_387:
                v230 = v469;
                v234 = 0;
                goto LABEL_307;
              }
              LODWORD(v485) = v236;
              v237 = *((_QWORD *)v234 + 3);
              v238 = (unsigned __int64 *)(v234 + 8);
              if ( v237 == -4096 )
              {
                if ( v230 != -4096 )
                  goto LABEL_313;
              }
              else
              {
                --HIDWORD(v485);
                if ( v230 != v237 )
                {
                  if ( v237 != -8192 && v237 )
                  {
                    v412 = v234;
                    v419 = v234 + 8;
                    sub_BD60C0(v238);
                    v230 = v469;
                    v234 = v412;
                    v238 = (unsigned __int64 *)v419;
                  }
LABEL_313:
                  *((_QWORD *)v234 + 3) = v230;
                  if ( v230 == 0 || v230 == -4096 || v230 == -8192 )
                  {
                    v230 = v469;
                  }
                  else
                  {
                    v420 = v234;
                    sub_BD6050(v238, v467 & 0xFFFFFFFFFFFFFFF8LL);
                    v230 = v469;
                    v234 = v420;
                  }
                }
              }
              v239 = i;
              *((_QWORD *)v234 + 5) = 6;
              *((_QWORD *)v234 + 6) = 0;
              *((_QWORD *)v234 + 7) = 0;
              *((_QWORD *)v234 + 4) = v239;
              v240 = v234 + 40;
              goto LABEL_317;
            }
LABEL_368:
            v240 = v264 + 40;
LABEL_317:
            v466 = (const char *)&unk_49DB368;
            if ( v230 != -4096 && v230 != 0 && v230 != -8192 )
            {
              v421 = v240;
              sub_BD60C0(&v467);
              v240 = v421;
            }
            v241 = *((_QWORD *)v240 + 2);
            v242 = *(_DWORD *)(v222 + 4) & 0x7FFFFFF;
            if ( v242 == *(_DWORD *)(v222 + 72) )
            {
              v422 = v241;
              sub_B48D90(v222);
              v241 = v422;
              v242 = *(_DWORD *)(v222 + 4) & 0x7FFFFFF;
            }
            v243 = (v242 + 1) & 0x7FFFFFF;
            v244 = v243 | *(_DWORD *)(v222 + 4) & 0xF8000000;
            v245 = *(_QWORD *)(v222 - 8) + 32LL * (unsigned int)(v243 - 1);
            *(_DWORD *)(v222 + 4) = v244;
            if ( *(_QWORD *)v245 )
            {
              v246 = *(_QWORD *)(v245 + 8);
              **(_QWORD **)(v245 + 16) = v246;
              if ( v246 )
                *(_QWORD *)(v246 + 16) = *(_QWORD *)(v245 + 16);
            }
            *(_QWORD *)v245 = v226;
            if ( v226 )
            {
              v247 = *(_QWORD *)(v226 + 16);
              *(_QWORD *)(v245 + 8) = v247;
              if ( v247 )
                *(_QWORD *)(v247 + 16) = v245 + 8;
              *(_QWORD *)(v245 + 16) = v226 + 16;
              *(_QWORD *)(v226 + 16) = v245;
            }
            *(_QWORD *)(*(_QWORD *)(v222 - 8)
                      + 32LL * *(unsigned int *)(v222 + 72)
                      + 8LL * ((*(_DWORD *)(v222 + 4) & 0x7FFFFFFu) - 1)) = v241;
            sub_DACA20(a4, v436, v222);
            v248 = *(_QWORD *)(v222 + 32);
            if ( !v248 )
              BUG();
            v222 = 0;
            if ( *(_BYTE *)(v248 - 24) == 84 )
              v222 = v248 - 24;
          }
          v416 += 2;
        }
        while ( v400 != v416 );
        v8 = v436;
      }
      if ( (_DWORD)v485 )
      {
        v287 = (__int64 *)v484;
        v288 = (char *)(v484 + ((unsigned __int64)v486 << 6));
        if ( (char *)v484 != v288 )
        {
          while ( 1 )
          {
            v289 = v287[3];
            if ( v289 != -4096 && v289 != -8192 )
              break;
            v287 += 8;
            if ( v288 == (char *)v287 )
              goto LABEL_336;
          }
          while ( v287 != (__int64 *)v288 )
          {
            v290 = sub_2A07A50(a8, v287[3]);
            v291 = v290[2];
            v292 = v290;
            v293 = v287[7];
            if ( v291 != v293 )
            {
              if ( v291 != 0 && v291 != -4096 && v291 != -8192 )
              {
                sub_BD60C0(v292);
                v293 = v287[7];
              }
              v292[2] = v293;
              if ( v293 != -4096 && v293 != 0 && v293 != -8192 )
                sub_BD6050(v292, v287[5] & 0xFFFFFFFFFFFFFFF8LL);
            }
            v287 += 8;
            if ( v287 == (__int64 *)v288 )
              break;
            while ( 1 )
            {
              v294 = v287[3];
              if ( v294 != -8192 && v294 != -4096 )
                break;
              v287 += 8;
              if ( v288 == (char *)v287 )
                goto LABEL_336;
            }
          }
        }
      }
LABEL_336:
      sub_F45F60((__int64)v477, (unsigned int)v478, (__int64)&v483);
      if ( !v405 )
      {
        if ( (_DWORD)v451 )
        {
          v305 = v450;
          v306 = 2LL * v452;
          v307 = &v450[v306];
          if ( v450 != &v450[v306] )
          {
            while ( 1 )
            {
              v308 = *v305;
              v309 = v305;
              if ( *v305 != -4096 && v308 != -8192 )
                break;
              v305 += 2;
              if ( v307 == v305 )
                goto LABEL_337;
            }
            if ( v305 != v307 )
            {
              v446 = v8;
              while ( 1 )
              {
                v310 = sub_2A07A50(a8, v309[1])[2];
                if ( v310 )
                {
                  v311 = (unsigned int)(*(_DWORD *)(v310 + 44) + 1);
                  v312 = *(_DWORD *)(v310 + 44) + 1;
                }
                else
                {
                  v312 = 0;
                  v311 = 0;
                }
                v313 = *(_DWORD *)(a5 + 32);
                v314 = 0;
                if ( v312 < v313 )
                  v314 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v311);
                if ( v308 )
                {
                  v315 = (unsigned int)(*(_DWORD *)(v308 + 44) + 1);
                  v316 = *(_DWORD *)(v308 + 44) + 1;
                }
                else
                {
                  v316 = 0;
                  v315 = 0;
                }
                if ( v313 <= v316 )
                {
                  *(_BYTE *)(a5 + 112) = 0;
                  BUG();
                }
                v317 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v315);
                *(_BYTE *)(a5 + 112) = 0;
                v318 = *(_QWORD *)(v317 + 8);
                if ( v314 != v318 )
                {
                  v466 = (const char *)v317;
                  v319 = *(_QWORD **)(v318 + 24);
                  v437 = v318;
                  v320 = (__int64)&v319[*(unsigned int *)(v318 + 32)];
                  v321 = sub_2A043B0(v319, v320, (__int64 *)&v466);
                  v324 = v437;
                  if ( v321 + 1 != (_QWORD *)v320 )
                  {
                    v325 = v320 - (_QWORD)(v321 + 1);
                    v320 = (__int64)(v321 + 1);
                    memmove(v321, v321 + 1, v325);
                    v324 = v437;
                    v323 = *(_DWORD *)(v437 + 32);
                  }
                  v326 = (unsigned int)(v323 - 1);
                  *(_DWORD *)(v324 + 32) = v326;
                  *(_QWORD *)(v317 + 8) = v314;
                  v327 = *(unsigned int *)(v314 + 32);
                  v328 = *(unsigned int *)(v314 + 36);
                  if ( v327 + 1 > v328 )
                  {
                    v320 = v314 + 40;
                    sub_C8D5F0(v314 + 24, (const void *)(v314 + 40), v327 + 1, 8u, v322, v326);
                    v327 = *(unsigned int *)(v314 + 32);
                  }
                  v329 = *(_QWORD *)(v314 + 24);
                  *(_QWORD *)(v329 + 8 * v327) = v317;
                  ++*(_DWORD *)(v314 + 32);
                  if ( *(_DWORD *)(v317 + 16) != *(_DWORD *)(*(_QWORD *)(v317 + 8) + 16LL) + 1 )
                    sub_2A04730(v317, v320, v329, v328, v322, v326);
                }
                v309 += 2;
                if ( v309 == v307 )
                  break;
                while ( *v309 == -4096 || *v309 == -8192 )
                {
                  v309 += 2;
                  if ( v307 == v309 )
                    goto LABEL_506;
                }
                if ( v309 == v307 )
                  break;
                v308 = *v309;
              }
LABEL_506:
              v8 = v446;
            }
          }
        }
      }
LABEL_337:
      if ( (_DWORD)v455 )
      {
        v277 = v454;
        v278 = &v454[17 * v456];
        if ( v454 != v278 )
        {
          while ( 1 )
          {
            v279 = *v277;
            v280 = v277;
            if ( *v277 != -8192 && v279 != -4096 )
              break;
            v277 += 17;
            if ( v278 == v277 )
              goto LABEL_338;
          }
          if ( v277 != v278 )
          {
            do
            {
              v281 = sub_2A07A50((__int64)&v483, v279);
              sub_BC8EC0(v281[2], (unsigned int *)v280[1], *((unsigned int *)v280 + 4), 0);
              v282 = 0;
              v283 = v280[9];
              v284 = 4LL * *((unsigned int *)v280 + 20);
              if ( v284 )
              {
                do
                {
                  v285 = *(_DWORD *)(v283 + v282);
                  if ( v285 )
                  {
                    v286 = (unsigned int *)(v282 + v280[1]);
                    if ( v285 < *v286 && v285 < *v286 - v285 )
                      v285 = *v286 - v285;
                    *v286 = v285;
                  }
                  v282 += 4;
                }
                while ( v284 != v282 );
              }
              v280 += 17;
              if ( v280 == v278 )
                break;
              while ( 1 )
              {
                v279 = *v280;
                if ( *v280 != -8192 && v279 != -4096 )
                  break;
                v280 += 17;
                if ( v278 == v280 )
                  goto LABEL_338;
              }
            }
            while ( v280 != v278 );
          }
        }
      }
LABEL_338:
      v249 = sub_2A07A50((__int64)&v483, v390);
      sub_B99FD0(v249[2], 0x12u, 0);
      LOWORD(i) = 257;
      v250 = *(_QWORD *)(v396 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v250 == v396 + 48 )
      {
        v252 = 0;
      }
      else
      {
        if ( !v250 )
          BUG();
        v251 = *(unsigned __int8 *)(v250 - 24);
        v252 = v250 - 24;
        if ( (unsigned int)(v251 - 30) >= 0xB )
          v252 = 0;
      }
      v253 = (__int64 *)(v252 + 24);
      v254 = v378;
      LOWORD(v254) = 0;
      v378 = v254;
      v445 = (unsigned __int8 *)sub_F36960(v396, v253, v254, a5, a3, 0, (void **)&v466, 0);
      v255 = sub_BD5D20(v394);
      LOWORD(i) = 773;
      v466 = v255;
      v467 = v256;
      v468 = ".peel.next";
      sub_BD6B50(v445, &v466);
      sub_B2C300(
        v385,
        (__int64 *)(v396 + 24),
        v385,
        (unsigned __int64 *)(*(_QWORD *)v477 + 24LL),
        (unsigned __int64 *)(v385 + 72));
      if ( v490 )
      {
        v295 = v489;
        v490 = 0;
        if ( v489 )
        {
          v296 = v488;
          v297 = &v488[2 * v489];
          do
          {
            if ( *v296 != -8192 && *v296 != -4096 )
            {
              v298 = v296[1];
              if ( v298 )
                sub_B91220((__int64)(v296 + 1), v298);
            }
            v296 += 2;
          }
          while ( v297 != v296 );
          v295 = v489;
        }
        sub_C7D6A0((__int64)v488, 16 * v295, 8);
      }
      v257 = v486;
      if ( v486 )
      {
        v258 = (char *)v484;
        v462 = 2;
        v463 = 0;
        v259 = -4096;
        v260 = (char *)(v484 + ((unsigned __int64)v486 << 6));
        v464 = -4096;
        v461 = (__int64 *)&unk_49DD7B0;
        v465 = 0;
        v467 = 2;
        v468 = 0;
        v469 = -8192;
        v466 = (const char *)&unk_49DD7B0;
        i = 0;
        while ( 1 )
        {
          v261 = *((_QWORD *)v258 + 3);
          if ( v259 != v261 )
          {
            v259 = v469;
            if ( v261 != v469 )
            {
              v262 = *((_QWORD *)v258 + 7);
              if ( v262 != 0 && v262 != -4096 && v262 != -8192 )
              {
                sub_BD60C0((_QWORD *)v258 + 5);
                v261 = *((_QWORD *)v258 + 3);
              }
              v259 = v261;
            }
          }
          *(_QWORD *)v258 = &unk_49DB368;
          if ( v259 != -4096 && v259 != 0 && v259 != -8192 )
            sub_BD60C0((_QWORD *)v258 + 1);
          v258 += 64;
          if ( v260 == v258 )
            break;
          v259 = v464;
        }
        v466 = (const char *)&unk_49DB368;
        if ( v469 != -4096 && v469 != 0 && v469 != -8192 )
          sub_BD60C0(&v467);
        v461 = (__int64 *)&unk_49DB368;
        if ( v464 != 0 && v464 != -4096 && v464 != -8192 )
          sub_BD60C0(&v462);
        v257 = v486;
      }
      sub_C7D6A0(v484, v257 << 6, 8);
      if ( v477 != v479 )
        _libc_free((unsigned __int64)v477);
      ++v405;
      v399 = v396;
      if ( a2 != v405 )
      {
        v396 = (__int64)v445;
        continue;
      }
      break;
    }
  }
  v330 = *(_QWORD *)(v394 + 56);
  while ( 1 )
  {
    if ( !v330 )
      BUG();
    if ( *(_BYTE *)(v330 - 24) != 84 )
      break;
    v101 = *(_QWORD *)(v330 - 32);
    v331 = v101;
    v332 = *(_DWORD *)(v330 - 20) & 0x7FFFFFF;
    if ( v332 )
    {
      v102 = *(unsigned int *)(v330 + 48);
      v333 = 0;
      v334 = *(_DWORD *)(v330 + 48);
      do
      {
        if ( v395 == *(_QWORD *)(v101 + 32 * v102 + 8 * v333) )
        {
          v100 = *(_QWORD *)(v101 + 32 * v333);
          if ( *(_BYTE *)v100 > 0x1Cu )
          {
LABEL_516:
            v335 = *(_QWORD *)(v100 + 40);
            if ( !*(_BYTE *)(v8 + 84) )
              goto LABEL_540;
            goto LABEL_517;
          }
LABEL_526:
          v338 = 0;
          v101 = v100 + 16;
          while ( 1 )
          {
            v102 = 32 * v102 + 8LL * v338;
            if ( v386 != *(unsigned __int8 **)(v331 + v102) )
              goto LABEL_527;
            v339 = 32LL * v338 + v331;
            if ( *(_QWORD *)v339 )
            {
              v102 = *(_QWORD *)(v339 + 8);
              **(_QWORD **)(v339 + 16) = v102;
              if ( v102 )
                *(_QWORD *)(v102 + 16) = *(_QWORD *)(v339 + 16);
            }
            *(_QWORD *)v339 = v100;
            if ( v100 )
            {
              v102 = *(_QWORD *)(v100 + 16);
              *(_QWORD *)(v339 + 8) = v102;
              if ( v102 )
                *(_QWORD *)(v102 + 16) = v339 + 8;
              ++v338;
              *(_QWORD *)(v339 + 16) = v101;
              *(_QWORD *)(v100 + 16) = v339;
              if ( v332 == v338 )
                goto LABEL_537;
            }
            else
            {
LABEL_527:
              if ( v332 == ++v338 )
                goto LABEL_537;
            }
            v331 = *(_QWORD *)(v330 - 32);
            v102 = *(unsigned int *)(v330 + 48);
          }
        }
        ++v333;
      }
      while ( v332 != (_DWORD)v333 );
      v100 = *(_QWORD *)(v101 + 0x1FFFFFFFE0LL);
      if ( *(_BYTE *)v100 <= 0x1Cu )
      {
LABEL_525:
        v331 = v101;
        v102 = v334;
        goto LABEL_526;
      }
      v335 = *(_QWORD *)(v100 + 40);
      if ( !*(_BYTE *)(v8 + 84) )
      {
LABEL_540:
        v448 = v100;
        v340 = sub_C8CA60(v8 + 56, v335);
        v100 = v448;
        if ( v340 )
          goto LABEL_521;
        goto LABEL_522;
      }
LABEL_517:
      v336 = *(_QWORD **)(v8 + 64);
      v337 = &v336[*(unsigned int *)(v8 + 76)];
      if ( v336 != v337 )
      {
        while ( v335 != *v336 )
        {
          if ( v337 == ++v336 )
            goto LABEL_523;
        }
LABEL_521:
        v100 = sub_2A07A50(a8, v100)[2];
LABEL_522:
        v332 = *(_DWORD *)(v330 - 20) & 0x7FFFFFF;
      }
LABEL_523:
      if ( v332 )
      {
        v101 = *(_QWORD *)(v330 - 32);
        v334 = *(_DWORD *)(v330 + 48);
        goto LABEL_525;
      }
LABEL_537:
      v330 = *(_QWORD *)(v330 + 8);
    }
    else
    {
      v100 = *(_QWORD *)(v101 + 0x1FFFFFFFE0LL);
      if ( *(_BYTE *)v100 > 0x1Cu )
        goto LABEL_516;
      v330 = *(_QWORD *)(v330 + 8);
    }
  }
  if ( (_DWORD)v455 )
  {
    v102 = v456;
    v356 = v454;
    v357 = &v454[17 * v456];
    if ( v454 != v357 )
    {
      while ( 1 )
      {
        v358 = *v356;
        v359 = v356;
        if ( *v356 != -8192 && v358 != -4096 )
          break;
        v356 += 17;
        if ( v357 == v356 )
          goto LABEL_543;
      }
      if ( v356 != v357 )
      {
        while ( 1 )
        {
          v360 = (unsigned int *)v359[1];
          v361 = *((unsigned int *)v359 + 4);
          v359 += 17;
          sub_BC8EC0(v358, v360, v361, 0);
          if ( v359 == v357 )
            break;
          while ( *v359 == -4096 || *v359 == -8192 )
          {
            v359 += 17;
            if ( v357 == v359 )
              goto LABEL_543;
          }
          if ( v359 == v357 )
            break;
          v358 = *v359;
        }
      }
    }
  }
LABEL_543:
  v483 = (const char **)sub_D4A2B0(v8, "llvm.loop.peeled.count", 0x16u, v102, v100, v101);
  if ( BYTE4(v483) )
    a2 += (unsigned int)v483;
  sub_F6DC70(v8, "llvm.loop.peeled.count", a2, v341, v342, v343);
  if ( *(_QWORD *)v8 )
    v8 = *(_QWORD *)v8;
  sub_DAC8B0(a4, (_QWORD *)v8);
  sub_D9D700(a4, 0);
  sub_F6AC10((char *)v8, a5, a3, a4, a6, 0, a7);
  if ( v471 != v473 )
    _libc_free((unsigned __int64)v471);
  v344 = v456;
  if ( v456 )
  {
    v345 = v454;
    v346 = &v454[17 * v456];
    do
    {
      if ( *v345 != -4096 && *v345 != -8192 )
      {
        v347 = v345[9];
        if ( (__int64 *)v347 != v345 + 11 )
          _libc_free(v347);
        v348 = v345[1];
        if ( (__int64 *)v348 != v345 + 3 )
          _libc_free(v348);
      }
      v345 += 17;
    }
    while ( v346 != v345 );
    v344 = v456;
  }
  sub_C7D6A0((__int64)v454, 136 * v344, 8);
  sub_C7D6A0((__int64)v450, 16LL * v452, 8);
  if ( v474 != (__int64 *)v476 )
    _libc_free((unsigned __int64)v474);
  if ( v459 )
    j_j___libc_free_0(v459);
  sub_C7D6A0(v457[2], 16LL * v458, 8);
  return 1;
}
