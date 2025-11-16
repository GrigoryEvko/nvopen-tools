// Function: sub_14FCE40
// Address: 0x14fce40
//
__int64 *__fastcall sub_14FCE40(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r15
  __int64 v4; // r14
  const char *v5; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r15
  unsigned __int64 v12; // rax
  int v13; // r12d
  __int64 v14; // r14
  __m128i *v15; // rbx
  __m128i *v16; // r12
  __int64 v17; // rdi
  _QWORD *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rdi
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // r13d
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // rsi
  __int64 v36; // rdx
  int v37; // eax
  __int64 v38; // r13
  __int64 v39; // r8
  int v40; // ebx
  __int64 v41; // r14
  __int64 *v42; // rsi
  int v43; // eax
  char v44; // dl
  int v45; // r8d
  unsigned int v46; // r14d
  unsigned int v47; // ecx
  int v48; // ecx
  char v49; // al
  _BYTE *v50; // rbx
  int v51; // r13d
  _QWORD *v52; // rax
  __int64 v53; // r12
  __int64 v54; // rax
  int v55; // esi
  __int64 v56; // rsi
  __int64 v57; // rdx
  unsigned __int64 v58; // rcx
  __int64 v59; // r8
  const char *v60; // r13
  __int64 v61; // rax
  unsigned __int8 *v62; // rsi
  __int64 v63; // rdx
  int v64; // r13d
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // ebx
  __int64 v68; // r12
  __int64 v69; // rax
  _BYTE *v70; // rbx
  char *v71; // rax
  _BYTE *v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r12
  __int64 v77; // rbx
  __int64 v78; // r14
  __int64 j; // rax
  __int64 v80; // r8
  __int64 v81; // rax
  unsigned __int64 v82; // rcx
  unsigned int v83; // r13d
  __int64 v84; // rdx
  __int64 v85; // r14
  const char *v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rsi
  unsigned int v89; // eax
  __int64 v90; // r9
  unsigned int v91; // r13d
  __int64 v92; // rbx
  _BYTE *v93; // r12
  unsigned int v94; // eax
  __int64 *v95; // r14
  const char *v96; // r13
  __int64 v97; // rax
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  __int64 v100; // rax
  __int64 v101; // rsi
  unsigned __int64 v102; // rcx
  __int64 v103; // rdx
  unsigned int v104; // esi
  __int64 v105; // rdi
  unsigned int v106; // edx
  __int64 *v107; // r13
  __int64 v108; // rcx
  __int64 v109; // rcx
  __int64 v110; // rax
  __int64 v111; // r14
  __int64 v112; // rax
  __int64 v113; // rbx
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // r12
  unsigned __int64 v117; // rcx
  __int64 v118; // r14
  __int64 v119; // rax
  int v120; // r14d
  unsigned __int64 v121; // rcx
  __int64 v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rsi
  __int64 v125; // rax
  __int64 v126; // rdx
  unsigned __int64 v127; // rdi
  unsigned __int64 v128; // rax
  __int64 v129; // r12
  unsigned __int64 v130; // rcx
  __int64 v131; // rdi
  __int64 v132; // r13
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // r14
  __int64 v136; // rax
  __int64 v137; // rax
  int v138; // edx
  __int64 v139; // rdx
  int v140; // r12d
  __int64 v141; // rax
  unsigned __int8 v142; // r13
  __int64 v143; // rax
  const char *v144; // r12
  __int64 v145; // rax
  __int64 v146; // rbx
  __int64 v147; // r13
  __int64 v148; // r12
  __int64 v149; // rax
  __int64 v150; // r12
  __int64 v151; // rdx
  unsigned __int64 v152; // rcx
  __int64 v153; // r13
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // r12
  unsigned int v158; // r14d
  __int64 v159; // rdx
  unsigned __int64 v160; // rcx
  __int64 v161; // r13
  __int64 v162; // rax
  __int64 v163; // r13
  __int64 v164; // rbx
  __int64 v165; // r12
  __int64 v166; // rax
  _BYTE *v167; // r13
  __int64 v168; // r14
  __int64 v169; // rdx
  unsigned __int64 v170; // rcx
  __int64 v171; // rdx
  __int64 v172; // rax
  int v173; // ecx
  __int64 v174; // r14
  unsigned __int64 v175; // r14
  __int64 v176; // rax
  __int64 v177; // rbx
  __int64 v178; // r13
  __int64 v179; // rsi
  __int64 v180; // r13
  unsigned __int64 v181; // rdx
  __int64 v182; // rbx
  __int64 v183; // r12
  __int64 v184; // rax
  unsigned __int64 v185; // r12
  __int64 v186; // r14
  int v187; // r15d
  __int64 v188; // rax
  unsigned __int64 v189; // rax
  __int64 v190; // rsi
  __int64 v191; // rdx
  unsigned __int64 v192; // rsi
  __int64 v193; // rcx
  int v194; // eax
  __int64 v195; // rax
  int v196; // esi
  const char *v197; // rsi
  __int64 *v198; // rax
  __int64 v199; // rdi
  unsigned __int64 v200; // rsi
  __int64 v201; // rsi
  __int64 v202; // rdx
  __int64 v203; // rax
  unsigned int v204; // edx
  __int64 v205; // rcx
  unsigned __int64 v206; // rsi
  __int64 v207; // rax
  __int64 v208; // rcx
  unsigned __int64 v209; // rsi
  unsigned __int64 v210; // rax
  unsigned __int64 v211; // rsi
  __int64 v212; // r12
  __int64 v213; // r13
  unsigned int v214; // edx
  int v215; // eax
  __int64 v216; // r14
  __int64 v217; // rbx
  const char *v218; // r14
  __int64 *v219; // r13
  __int64 *v220; // r12
  __int64 v221; // rax
  unsigned __int64 v222; // rax
  __int64 v223; // r13
  unsigned __int64 v224; // rbx
  __int64 v225; // r14
  __int64 v226; // rax
  __int64 v227; // r8
  _BYTE *v228; // rbx
  __int64 *v229; // r13
  unsigned __int64 v230; // rbx
  __int64 v231; // rbx
  __int64 v232; // rax
  __int64 v233; // r12
  int v234; // ebx
  __int64 v235; // rax
  unsigned int v236; // r14d
  unsigned __int64 v237; // rcx
  __int64 v238; // rdx
  __int64 v239; // rdi
  __int64 v240; // rbx
  __int64 v241; // rbx
  __int64 v242; // r14
  __int64 v243; // rsi
  __int64 v244; // rdx
  unsigned __int64 v245; // rcx
  __int64 v246; // rax
  __int64 v247; // r12
  __int64 v248; // r13
  __int64 v249; // rdx
  int v250; // eax
  __int64 v251; // rbx
  __int64 v252; // r13
  __int64 v253; // r15
  unsigned __int64 v254; // rcx
  __int64 v255; // rdx
  const char *v256; // rax
  __int64 v257; // r8
  __int64 v258; // rax
  int v259; // r14d
  __int64 v260; // r13
  char v261; // dl
  unsigned __int64 v262; // rbx
  int v263; // r12d
  __int64 v264; // rax
  __int64 v265; // r13
  char v266; // dl
  unsigned __int64 v267; // rbx
  __int64 v268; // r13
  __int64 v269; // rdx
  unsigned __int64 v270; // rax
  __int64 v271; // rdi
  __int64 v272; // rsi
  __int64 v273; // rdx
  unsigned int v274; // r12d
  unsigned __int64 v275; // rax
  int v276; // ebx
  __int64 v277; // r13
  __int64 v278; // rax
  __int64 v279; // rbx
  __int64 v280; // r12
  __int64 v281; // rax
  unsigned int v282; // r14d
  __int64 v283; // r8
  int v284; // r14d
  __int64 v285; // rax
  unsigned __int64 v286; // rdx
  __int64 v287; // rax
  int v288; // ebx
  int v289; // r14d
  __int64 v290; // rax
  __int64 v291; // rax
  unsigned __int64 v292; // rdx
  _BOOL4 v293; // r13d
  __int64 v294; // rax
  _BOOL8 v295; // r12
  __int64 v296; // rax
  int v297; // r14d
  __int64 v298; // rax
  __int64 v299; // r10
  __int64 v300; // rcx
  unsigned int v301; // esi
  __int16 v302; // ax
  __int64 v303; // rdx
  __int64 v304; // r14
  _BOOL8 v305; // r12
  __int64 v306; // rax
  __int64 v307; // rax
  int v308; // edx
  __int64 v309; // rax
  __int64 v310; // r14
  __int16 v311; // ax
  __int64 v312; // rax
  const char *v313; // r12
  __int64 v314; // r13
  __int64 v315; // rax
  int v316; // ebx
  const char *v317; // r13
  __int64 v318; // r14
  __m128i *v319; // rax
  int v320; // esi
  __int64 v321; // rdx
  __int64 v322; // rax
  __int64 v323; // r12
  __int64 v324; // rax
  __int64 v325; // r14
  __int64 v326; // rdx
  __int64 v327; // r14
  __int64 v328; // r14
  __int64 v329; // rax
  _BYTE *v330; // r13
  const char *v331; // r12
  const char *v332; // rbx
  __int64 v333; // rsi
  const char *v334; // rdi
  __int64 v335; // r14
  int v336; // r13d
  int v337; // r14d
  int v338; // ebx
  __int64 v339; // rsi
  __int64 v340; // rax
  _BYTE *v341; // r12
  int v342; // r13d
  _QWORD *v343; // rax
  __int64 v344; // r12
  __int64 v345; // rax
  int v346; // esi
  __int64 v347; // rax
  __int64 v348; // rax
  int v349; // r13d
  __int64 v350; // rax
  unsigned __int8 v351; // r12
  __int64 v352; // rax
  __int64 v353; // r14
  __int16 v354; // dx
  __int64 v355; // rcx
  __int64 v356; // r12
  __int64 v357; // rdi
  __int64 v358; // rsi
  __int64 *v359; // rbx
  __int64 v360; // rax
  __int64 v361; // rdi
  __int64 v362; // r14
  __int64 v363; // r14
  __int64 v364; // r14
  const char *v365; // rsi
  __int64 v366; // r14
  __int64 v367; // r14
  bool v368; // zf
  __int64 *v369; // r12
  __int64 v370; // rax
  __int64 v371; // rax
  __int64 v372; // r14
  const char *v373; // r13
  __int64 v374; // rbx
  __int64 v375; // rax
  __int64 v376; // rax
  __int64 v377; // rdi
  const char *v378; // rcx
  const char *v379; // rax
  __int64 v380; // rdx
  __int64 v381; // r14
  __int64 v382; // r14
  __int64 v383; // rax
  __int64 v384; // r14
  __int64 v385; // r13
  __int64 v386; // rax
  __int64 v387; // r14
  __int64 v388; // r14
  __m128i *v389; // rdx
  int v390; // eax
  __int64 v391; // rsi
  int v392; // r12d
  __int64 v393; // rax
  __m128i *v394; // rcx
  __int64 v395; // rbx
  __m128i *v396; // r10
  __int64 v397; // r11
  unsigned int v398; // edx
  __int64 v399; // rax
  _BYTE *v400; // r12
  int v401; // edx
  unsigned int v402; // eax
  __int64 v403; // r14
  const char *v404; // rax
  __int64 v405; // r14
  __int64 v406; // r14
  __int64 v407; // rax
  __int64 v408; // rax
  __int64 v409; // rcx
  __int64 v410; // r12
  unsigned __int64 v411; // rsi
  __int64 v412; // r14
  __int64 v413; // rbx
  __int64 v414; // rax
  __int64 v415; // rdx
  unsigned int v416; // r13d
  __int64 v417; // rax
  unsigned int v418; // r12d
  unsigned __int64 v419; // r8
  __int64 v420; // r12
  __int64 v421; // rax
  __int64 v422; // rax
  __int64 v423; // rdx
  unsigned int v424; // r12d
  unsigned __int64 v425; // rdx
  __int64 v426; // rcx
  __int64 v427; // r12
  __int64 v428; // rax
  __int64 v429; // rdx
  __int64 v430; // r13
  const char *v431; // rdi
  const char *v432; // r12
  const char *v433; // r15
  __int64 v434; // r14
  const char *v435; // rax
  __int64 v436; // r14
  __int64 v437; // rax
  __int64 v438; // rax
  __int64 v439; // rax
  __int64 *v440; // r12
  const char *v441; // r14
  __int64 *v442; // r13
  __int64 v443; // rax
  __int64 v444; // rbx
  __int64 v445; // r14
  const char *v446; // rax
  __int64 v447; // rdx
  unsigned __int64 v448; // rcx
  __int64 v449; // r14
  __int64 v450; // rax
  __int64 v451; // r13
  __int64 v452; // rsi
  __int64 v453; // r14
  __int64 *v454; // r14
  __int64 v455; // rax
  __int64 v456; // r12
  _BYTE *v457; // rax
  __int64 v458; // rax
  __int64 v459; // r14
  __int64 v460; // rax
  __int64 v461; // r14
  __int64 v462; // r14
  __int64 v463; // r14
  __int64 v464; // rax
  __int64 v465; // r14
  __int64 v466; // r14
  __int64 v467; // r14
  __int64 v468; // r14
  __m128i *v469; // [rsp+8h] [rbp-408h]
  __m128i *v470; // [rsp+8h] [rbp-408h]
  const char *v471; // [rsp+8h] [rbp-408h]
  __int64 v472; // [rsp+30h] [rbp-3E0h]
  __m128i *v473; // [rsp+30h] [rbp-3E0h]
  __int64 v474; // [rsp+30h] [rbp-3E0h]
  int v475; // [rsp+30h] [rbp-3E0h]
  __m128i *v476; // [rsp+30h] [rbp-3E0h]
  __int64 v477; // [rsp+38h] [rbp-3D8h]
  unsigned int v478; // [rsp+40h] [rbp-3D0h]
  __int64 v479; // [rsp+40h] [rbp-3D0h]
  __int16 v480; // [rsp+48h] [rbp-3C8h]
  unsigned int v481; // [rsp+48h] [rbp-3C8h]
  int v482; // [rsp+48h] [rbp-3C8h]
  int v483; // [rsp+48h] [rbp-3C8h]
  __int64 v484; // [rsp+50h] [rbp-3C0h]
  __int64 v485; // [rsp+50h] [rbp-3C0h]
  unsigned __int64 v486; // [rsp+50h] [rbp-3C0h]
  __int64 v487; // [rsp+50h] [rbp-3C0h]
  int v488; // [rsp+50h] [rbp-3C0h]
  unsigned int v489; // [rsp+50h] [rbp-3C0h]
  int v490; // [rsp+58h] [rbp-3B8h]
  __int64 v491; // [rsp+58h] [rbp-3B8h]
  __int64 v492; // [rsp+58h] [rbp-3B8h]
  __int64 v493; // [rsp+58h] [rbp-3B8h]
  bool v494; // [rsp+58h] [rbp-3B8h]
  __int64 v495; // [rsp+58h] [rbp-3B8h]
  __int64 v496; // [rsp+58h] [rbp-3B8h]
  int v497; // [rsp+60h] [rbp-3B0h]
  unsigned __int64 v498; // [rsp+60h] [rbp-3B0h]
  __int64 v499; // [rsp+60h] [rbp-3B0h]
  __int64 v500; // [rsp+60h] [rbp-3B0h]
  __int64 v501; // [rsp+60h] [rbp-3B0h]
  unsigned int v502; // [rsp+60h] [rbp-3B0h]
  unsigned int v503; // [rsp+60h] [rbp-3B0h]
  int v504; // [rsp+68h] [rbp-3A8h]
  int v505; // [rsp+68h] [rbp-3A8h]
  int v506; // [rsp+68h] [rbp-3A8h]
  int v507; // [rsp+68h] [rbp-3A8h]
  __int64 v508; // [rsp+68h] [rbp-3A8h]
  int v509; // [rsp+70h] [rbp-3A0h]
  __int64 v510; // [rsp+70h] [rbp-3A0h]
  __int64 v511; // [rsp+70h] [rbp-3A0h]
  unsigned __int8 v512; // [rsp+70h] [rbp-3A0h]
  int v513; // [rsp+70h] [rbp-3A0h]
  __int64 v514; // [rsp+70h] [rbp-3A0h]
  __int64 v515; // [rsp+70h] [rbp-3A0h]
  unsigned int v516; // [rsp+70h] [rbp-3A0h]
  unsigned __int8 v517; // [rsp+70h] [rbp-3A0h]
  __int64 v518; // [rsp+70h] [rbp-3A0h]
  _BOOL4 v519; // [rsp+70h] [rbp-3A0h]
  unsigned int v520; // [rsp+70h] [rbp-3A0h]
  __int64 v521; // [rsp+70h] [rbp-3A0h]
  unsigned int v522; // [rsp+70h] [rbp-3A0h]
  int i; // [rsp+70h] [rbp-3A0h]
  __int64 v524; // [rsp+70h] [rbp-3A0h]
  unsigned int v525; // [rsp+7Ch] [rbp-394h]
  __int64 v526; // [rsp+80h] [rbp-390h]
  __int64 v527; // [rsp+88h] [rbp-388h]
  __int64 v528; // [rsp+98h] [rbp-378h]
  __int64 v529; // [rsp+98h] [rbp-378h]
  __int64 v530; // [rsp+98h] [rbp-378h]
  unsigned int v531; // [rsp+A0h] [rbp-370h]
  unsigned int v532; // [rsp+A4h] [rbp-36Ch]
  __int64 v533; // [rsp+B0h] [rbp-360h]
  __int64 v535; // [rsp+B8h] [rbp-358h]
  __int64 v536; // [rsp+C0h] [rbp-350h]
  unsigned int v538; // [rsp+D0h] [rbp-340h] BYREF
  unsigned int v539; // [rsp+D4h] [rbp-33Ch] BYREF
  const char *v540; // [rsp+D8h] [rbp-338h] BYREF
  _BYTE *v541; // [rsp+E0h] [rbp-330h] BYREF
  __int64 *v542; // [rsp+E8h] [rbp-328h] BYREF
  __int64 *v543; // [rsp+F0h] [rbp-320h] BYREF
  unsigned int v544; // [rsp+F8h] [rbp-318h]
  __m128i *v545; // [rsp+100h] [rbp-310h] BYREF
  __m128i *v546; // [rsp+108h] [rbp-308h]
  __int64 v547; // [rsp+110h] [rbp-300h]
  const char *v548; // [rsp+120h] [rbp-2F0h] BYREF
  _BYTE *v549; // [rsp+128h] [rbp-2E8h]
  _BYTE *v550; // [rsp+130h] [rbp-2E0h]
  const char *v551; // [rsp+140h] [rbp-2D0h] BYREF
  __int64 v552; // [rsp+148h] [rbp-2C8h]
  _WORD v553[64]; // [rsp+150h] [rbp-2C0h] BYREF
  char *v554; // [rsp+1D0h] [rbp-240h] BYREF
  __int64 v555; // [rsp+1D8h] [rbp-238h]
  char v556; // [rsp+1E0h] [rbp-230h] BYREF
  char v557; // [rsp+1E1h] [rbp-22Fh]

  v3 = a1;
  v4 = a2;
  v536 = a2 + 32;
  if ( (unsigned __int8)sub_15127D0(a2 + 32, 12, 0) )
  {
    v557 = 1;
    v5 = "Invalid record";
LABEL_3:
    v554 = (char *)v5;
    v556 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v554);
    return v3;
  }
  v535 = a2 + 608;
  if ( (unsigned __int8)sub_15160B0(a2 + 608) )
  {
    v557 = 1;
    v5 = "Invalid function metadata: incoming forward references";
    goto LABEL_3;
  }
  v7 = *(_QWORD *)(a2 + 560);
  *(_DWORD *)(a2 + 656) = 0;
  v527 = v7;
  v526 = *(_QWORD *)(a2 + 552);
  v525 = sub_1516170(v535);
  if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
  {
    sub_15E08E0(a3);
    v8 = *(_QWORD *)(a3 + 88);
    v533 = v8 + 40LL * *(_QWORD *)(a3 + 96);
    if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
    {
      sub_15E08E0(a3);
      v8 = *(_QWORD *)(a3 + 88);
    }
  }
  else
  {
    v8 = *(_QWORD *)(a3 + 88);
    v533 = v8 + 40LL * *(_QWORD *)(a3 + 96);
  }
  if ( v533 != v8 )
  {
    v9 = v8;
    do
    {
      v554 = (char *)v9;
      v9 += 40;
      sub_14EFCA0(a2 + 552, (__int64 *)&v554);
    }
    while ( v533 != v9 );
    v4 = a2;
  }
  v10 = *(_QWORD *)(v4 + 560) - *(_QWORD *)(v4 + 552);
  v540 = 0;
  v545 = 0;
  v546 = 0;
  v547 = 0;
  v531 = 0;
  v532 = -1431655765 * (v10 >> 3);
  v554 = &v556;
  v555 = 0x4000000000LL;
  v528 = 0;
  v11 = v4;
  while ( 2 )
  {
    while ( 2 )
    {
      v12 = sub_14ECC00(v536, 0);
      switch ( (_DWORD)v12 )
      {
        case 1:
          v18 = (_QWORD *)v11;
          v3 = a1;
          if ( v545 == v546 )
          {
            v19 = v18[70];
            v20 = *(_QWORD *)(v19 - 8);
            v21 = 0xAAAAAAAAAAAAAAABLL * ((v527 - v526) >> 3);
            if ( *(_BYTE *)(v20 + 16) == 17 && !*(_QWORD *)(v20 + 24) )
            {
              v273 = v18[69];
              v274 = -1431655765 * ((v527 - v526) >> 3);
              v275 = 0xAAAAAAAAAAAAAAABLL * ((v19 - v273) >> 3);
              if ( (_DWORD)v21 != (_DWORD)v275 )
              {
                v276 = v275;
                while ( 1 )
                {
                  v277 = *(_QWORD *)(v273 + 24LL * v274 + 16);
                  if ( v277 && *(_BYTE *)(v277 + 16) == 17 && !*(_QWORD *)(v277 + 24) )
                  {
                    v278 = sub_1599EF0(*(_QWORD *)v277);
                    sub_164D160(v277, v278);
                    sub_164BE60(v277);
                    j_j___libc_free_0(v277, 40);
                  }
                  if ( ++v274 == v276 )
                    break;
                  v273 = v18[69];
                }
              }
              v551 = "Never resolved value found in function";
              v553[0] = 259;
              sub_14EE4B0(a1, (__int64)(v18 + 1), (__int64)&v551);
            }
            else if ( (unsigned __int8)sub_15160B0(v535) )
            {
              v551 = "Invalid function metadata: outgoing forward refs";
              v553[0] = 259;
              sub_14EE4B0(a1, (__int64)(v18 + 1), (__int64)&v551);
            }
            else
            {
              v268 = v18[70];
              v269 = v18[69];
              v270 = 0xAAAAAAAAAAAAAAABLL * ((v268 - v269) >> 3);
              if ( (unsigned int)v21 > v270 )
              {
                sub_14EF7A0((__int64)(v18 + 69), (unsigned int)v21 - v270);
              }
              else if ( (unsigned int)v21 < v270 )
              {
                v279 = v269 + 24LL * (unsigned int)v21;
                if ( v268 != v279 )
                {
                  v280 = v279;
                  do
                  {
                    v281 = *(_QWORD *)(v280 + 16);
                    if ( v281 != 0 && v281 != -8 && v281 != -16 )
                      sub_1649B30(v280);
                    v280 += 24;
                  }
                  while ( v268 != v280 );
                  v18[70] = v279;
                }
              }
              sub_15167A0(v535, v525);
              v271 = v18[171];
              v272 = v18[173];
              v18[171] = 0;
              v18[172] = 0;
              v18[173] = 0;
              if ( v271 )
                j_j___libc_free_0(v271, v272 - v271);
              *a1 = 1;
              v551 = 0;
              sub_14ECA90((__int64 *)&v551);
            }
          }
          else
          {
            v551 = "Operand bundles found with no consumer";
            v553[0] = 259;
            sub_14EE4B0(a1, (__int64)(v18 + 1), (__int64)&v551);
          }
          goto LABEL_21;
        case 2:
          switch ( HIDWORD(v12) )
          {
            case 0xB:
              sub_14F9A40((__int64 *)&v551, (_QWORD *)v11);
              v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
              if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                goto LABEL_517;
              v551 = 0;
              sub_14ECA90((__int64 *)&v551);
              v532 = -1431655765 * ((__int64)(*(_QWORD *)(v11 + 560) - *(_QWORD *)(v11 + 552)) >> 3);
              continue;
            case 0xE:
              sub_14F8D00((__int64 *)&v551, v11, 0);
              v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
              if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                goto LABEL_517;
              goto LABEL_440;
            case 0xF:
              sub_1522BE0(&v551, v535, 0);
              v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
              if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                goto LABEL_517;
              goto LABEL_440;
            case 0x10:
              sub_1521F30(&v551, v535, a3, v11 + 648);
              v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
              if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                goto LABEL_517;
              goto LABEL_440;
            case 0x12:
              sub_14FC3B0((__int64 *)&v551, v11);
              v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
              if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                goto LABEL_517;
LABEL_440:
              v551 = 0;
              sub_14ECA90((__int64 *)&v551);
              continue;
            default:
              if ( (unsigned __int8)sub_14ED8F0(v536) )
                goto LABEL_663;
              continue;
          }
        case 0:
          v14 = v11;
          v3 = a1;
          v551 = "Malformed block";
          v553[0] = 259;
          sub_14EE4B0(a1, v14 + 8, (__int64)&v551);
          goto LABEL_21;
      }
    }
    LODWORD(v555) = 0;
    v541 = 0;
    v13 = sub_1510D70(v536, HIDWORD(v12), &v554, 0);
    switch ( v13 )
    {
      case 1:
        if ( !(_DWORD)v555 )
          goto LABEL_663;
        v99 = *(_QWORD *)v554;
        if ( !*(_QWORD *)v554 )
          goto LABEL_663;
        v100 = *(_QWORD *)(v11 + 1376);
        v101 = *(_QWORD *)(v11 + 1368);
        v102 = (v100 - v101) >> 3;
        if ( v99 > v102 )
        {
          sub_14F2040(v11 + 1368, v99 - v102);
          v100 = *(_QWORD *)(v11 + 1376);
        }
        else if ( v99 < v102 )
        {
          v103 = v101 + 8 * v99;
          if ( v100 != v103 )
          {
            *(_QWORD *)(v11 + 1376) = v103;
            v100 = v103;
          }
        }
        v104 = *(_DWORD *)(v11 + 1568);
        if ( !v104 )
          goto LABEL_782;
        v105 = *(_QWORD *)(v11 + 1552);
        v106 = (v104 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v107 = (__int64 *)(v105 + 32LL * v106);
        v108 = *v107;
        if ( a3 == *v107 )
          goto LABEL_183;
        while ( 2 )
        {
          if ( v108 == -8 )
            goto LABEL_782;
          v106 = (v104 - 1) & (v13 + v106);
          v107 = (__int64 *)(v105 + 32LL * v106);
          v108 = *v107;
          if ( a3 != *v107 )
          {
            ++v13;
            continue;
          }
          break;
        }
LABEL_183:
        v109 = v100 - *(_QWORD *)(v11 + 1368);
        if ( v107 == (__int64 *)(v105 + 32LL * v104) )
        {
LABEL_782:
          v450 = (v100 - *(_QWORD *)(v11 + 1368)) >> 3;
          if ( (_DWORD)v450 )
          {
            v451 = 0;
            v530 = 8LL * (unsigned int)v450;
            do
            {
              v452 = *(_QWORD *)(v11 + 432);
              v453 = *(_QWORD *)(v11 + 1368);
              v553[0] = 257;
              v454 = (__int64 *)(v451 + v453);
              v455 = sub_22077B0(64);
              v456 = v455;
              if ( v455 )
                sub_157FB60(v455, v452, &v551, a3, 0);
              *v454 = v456;
              v451 += 8;
            }
            while ( v451 != v530 );
          }
        }
        else
        {
          v110 = v107[2] - v107[1];
          if ( v110 > (unsigned __int64)v109 )
          {
            v111 = v11;
            v3 = a1;
            v551 = "Invalid ID";
            v553[0] = 259;
            sub_14EE4B0(a1, v111 + 8, (__int64)&v551);
            goto LABEL_21;
          }
          v355 = v109 >> 3;
          v520 = v110 >> 3;
          v529 = (unsigned int)v355;
          if ( (_DWORD)v355 )
          {
            v356 = 0;
            do
            {
              if ( v520 > (unsigned int)v356 && (v357 = *(_QWORD *)(v107[1] + 8 * v356)) != 0 )
              {
                sub_157FA80(v357, a3, 0);
                *(_QWORD *)(*(_QWORD *)(v11 + 1368) + 8 * v356) = *(_QWORD *)(v107[1] + 8 * v356);
              }
              else
              {
                v358 = *(_QWORD *)(v11 + 432);
                v359 = (__int64 *)(*(_QWORD *)(v11 + 1368) + 8 * v356);
                v553[0] = 257;
                v360 = sub_22077B0(64);
                if ( v360 )
                {
                  v508 = v360;
                  sub_157FB60(v360, v358, &v551, a3, 0);
                  v360 = v508;
                }
                *v359 = v360;
              }
              ++v356;
            }
            while ( v529 != v356 );
          }
          v361 = v107[1];
          if ( v361 )
            j_j___libc_free_0(v361, v107[3] - v361);
          *v107 = -16;
          --*(_DWORD *)(v11 + 1560);
          ++*(_DWORD *)(v11 + 1564);
        }
        v528 = **(_QWORD **)(v11 + 1368);
        continue;
      case 2:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543) )
          goto LABEL_663;
        if ( (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v542, v532, *v543, (__int64 *)&v548) )
          goto LABEL_663;
        v87 = (unsigned int)v542;
        if ( (int)v542 + 1 > (unsigned int)v555 )
          goto LABEL_663;
        v88 = *v543;
        LODWORD(v542) = (_DWORD)v542 + 1;
        v89 = sub_14EA280(*(_QWORD *)&v554[8 * v87], v88);
        v91 = v89;
        if ( v89 == -1 )
          goto LABEL_663;
        v553[0] = 257;
        v56 = (__int64)&v541;
        v541 = (_BYTE *)sub_15FB440(v89, v90, v548, &v551, 0);
        sub_14EF3D0(v11 + 648, &v541);
        v92 = (unsigned int)v542;
        if ( (unsigned int)v542 >= (unsigned int)v555 )
          goto LABEL_538;
        v93 = v541;
        if ( v91 <= 0x17 && (v58 = v91, (((unsigned __int64)&loc_80A800 >> v91) & 1) != 0) )
        {
          v437 = (__int64)v554;
          if ( (v554[8 * (unsigned int)v542] & 2) != 0 )
          {
            v56 = 1;
            sub_15F2330(v541, 1);
            v437 = (__int64)v554;
            v93 = v541;
          }
          v57 = (unsigned int)v542;
          if ( (*(_BYTE *)(v437 + 8LL * (unsigned int)v542) & 1) != 0 )
          {
            v56 = 1;
            sub_15F2310(v93, 1);
            v93 = v541;
          }
        }
        else if ( v91 - 17 <= 1 || v91 - 24 <= 1 )
        {
          if ( (v554[8 * (unsigned int)v542] & 1) != 0 )
          {
            v56 = 1;
            sub_15F2350(v541, 1);
            v93 = v541;
          }
        }
        else if ( (unsigned __int8)sub_14EE080(v541) )
        {
          v94 = sub_14E9B90(*(_QWORD *)&v554[8 * v92]);
          v56 = v94;
          if ( v94 )
          {
            sub_15F2440(v93, v94);
            goto LABEL_538;
          }
        }
        goto LABEL_165;
      case 3:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543) )
          goto LABEL_663;
        if ( (_DWORD)v542 + 2 != (_DWORD)v555 )
          goto LABEL_663;
        v147 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)&v554[8 * (unsigned int)v542]);
        v148 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 1)];
        if ( (unsigned int)v148 > 0xC )
          goto LABEL_663;
        v282 = v148 + 36;
        if ( !v147 )
          goto LABEL_663;
        v548 = 0;
        v541 = (_BYTE *)sub_1568FE0(v282, v543, v147, &v548);
        if ( v541 )
        {
          v283 = v11 + 648;
          if ( v548 )
          {
            sub_14EF3D0(v11 + 648, &v548);
            sub_14F2510(v528 + 40, (__int64 *)(v528 + 40), (__int64)v548);
            v283 = v11 + 648;
          }
        }
        else
        {
          if ( !(unsigned __int8)sub_15FC090(v282, v543, v147) )
          {
            v468 = v11;
            v3 = a1;
            v551 = "Invalid cast";
            v553[0] = 259;
            sub_14EE4B0(a1, v468 + 8, (__int64)&v551);
            goto LABEL_21;
          }
          v553[0] = 257;
          v457 = (_BYTE *)sub_15FDBD0(v282, v543, v147, &v551, 0);
          v283 = v11 + 648;
          v541 = v457;
        }
        v56 = (__int64)&v541;
        sub_14EF3D0(v283, &v541);
        v60 = v541;
        goto LABEL_102;
      case 4:
      case 30:
      case 43:
        v539 = 0;
        if ( v13 == 43 )
        {
          v539 = 1;
          v368 = *(_QWORD *)v554 == 0;
          v539 = 2;
          v494 = !v368;
          v385 = sub_14EFEB0((_QWORD *)v11, *((_QWORD *)v554 + 1));
          if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542) )
            goto LABEL_663;
          v369 = v542;
          v386 = *v542;
          if ( v385 )
          {
            if ( *(_BYTE *)(v386 + 8) == 16 )
              v386 = **(_QWORD **)(v386 + 16);
            v521 = *(_QWORD *)(v386 + 24);
            if ( v385 != v521 )
            {
              v387 = v11;
              v3 = a1;
              v551 = "Explicit gep type does not match pointee type of pointer operand";
              v553[0] = 259;
              sub_14EE4B0(a1, v387 + 8, (__int64)&v551);
              goto LABEL_21;
            }
LABEL_630:
            v551 = (const char *)v553;
            v552 = 0x1000000000LL;
            if ( v539 == (_DWORD)v555 )
              goto LABEL_637;
            while ( !sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v543) )
            {
              v371 = (unsigned int)v552;
              if ( (unsigned int)v552 >= HIDWORD(v552) )
              {
                sub_16CD150(&v551, v553, 0, 8);
                v371 = (unsigned int)v552;
              }
              *(_QWORD *)&v551[8 * v371] = v543;
              LODWORD(v552) = v552 + 1;
              if ( v539 == (_DWORD)v555 )
              {
                v369 = v542;
LABEL_637:
                v372 = (unsigned int)v552;
                v373 = v551;
                LOWORD(v550) = 257;
                v502 = v552 + 1;
                if ( !v521 )
                {
                  v458 = *v369;
                  if ( *(_BYTE *)(*v369 + 8) == 16 )
                    v458 = **(_QWORD **)(v458 + 16);
                  v521 = *(_QWORD *)(v458 + 24);
                }
                v374 = sub_1648A60(72, v502);
                if ( v374 )
                {
                  v375 = *v369;
                  if ( *(_BYTE *)(*v369 + 8) == 16 )
                    v375 = **(_QWORD **)(v375 + 16);
                  v481 = *(_DWORD *)(v375 + 8) >> 8;
                  v376 = sub_15F9F50(v521, v373, v372);
                  v377 = sub_1646BA0(v376, v481);
                  if ( *(_BYTE *)(*v369 + 8) == 16 )
                  {
                    v377 = sub_16463B0(v377, *(_QWORD *)(*v369 + 32));
                  }
                  else
                  {
                    v378 = &v373[8 * v372];
                    if ( v373 != v378 )
                    {
                      v379 = v373;
                      while ( 1 )
                      {
                        v380 = **(_QWORD **)v379;
                        if ( *(_BYTE *)(v380 + 8) == 16 )
                          break;
                        v379 += 8;
                        if ( v378 == v379 )
                          goto LABEL_647;
                      }
                      v377 = sub_16463B0(v377, *(_QWORD *)(v380 + 32));
                    }
                  }
LABEL_647:
                  sub_15F1EA0(v374, v377, 32, v374 - 24LL * v502, v502, 0);
                  *(_QWORD *)(v374 + 56) = v521;
                  *(_QWORD *)(v374 + 64) = sub_15F9F50(v521, v373, v372);
                  sub_15F9CE0(v374, v369, v373, v372, &v548);
                }
                v56 = (__int64)&v541;
                v541 = (_BYTE *)v374;
                sub_14EF3D0(v11 + 648, &v541);
                if ( v494 )
                {
                  v56 = 1;
                  sub_15FA2E0(v541, 1);
                }
LABEL_650:
                v334 = v551;
                if ( v551 == (const char *)v553 )
                {
LABEL_101:
                  v60 = v541;
                  goto LABEL_102;
                }
LABEL_562:
                _libc_free((unsigned __int64)v334);
                goto LABEL_101;
              }
            }
            v381 = v11;
            v3 = a1;
            v548 = "Invalid record";
            LOWORD(v550) = 259;
            sub_14EE4B0(a1, v381 + 8, (__int64)&v548);
            v86 = v551;
            if ( v551 != (const char *)v553 )
              goto LABEL_773;
            goto LABEL_21;
          }
        }
        else
        {
          if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542) )
            goto LABEL_663;
          v368 = v13 == 30;
          v369 = v542;
          v494 = v368;
        }
        v370 = *v369;
        if ( *(_BYTE *)(*v369 + 8) == 16 )
          v370 = **(_QWORD **)(v370 + 16);
        v521 = *(_QWORD *)(v370 + 24);
        goto LABEL_630;
      case 5:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542) )
          goto LABEL_663;
        if ( (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v539, v532, *v542, (__int64 *)&v543) )
          goto LABEL_663;
        v98 = sub_1643320(*(_QWORD *)(v11 + 432));
        if ( (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v539, v532, v98, (__int64 *)&v548) )
          goto LABEL_663;
        goto LABEL_174;
      case 6:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543)
          || sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v548) )
        {
          goto LABEL_663;
        }
        v95 = v543;
        if ( *(_BYTE *)(*v543 + 8) != 16 )
          goto LABEL_388;
        v96 = v548;
        v553[0] = 257;
        v97 = sub_1648A60(56, 2);
        v70 = (_BYTE *)v97;
        if ( v97 )
          sub_15FA320(v97, v95, v96, &v551, 0);
        goto LABEL_131;
      case 7:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542) )
          goto LABEL_663;
        if ( *(_BYTE *)(*v542 + 8) != 16 )
          goto LABEL_388;
        if ( (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v539, v532, *(_QWORD *)(*v542 + 24), (__int64 *)&v543)
          || sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v548) )
        {
          goto LABEL_663;
        }
        v218 = v548;
        v553[0] = 257;
        v219 = v543;
        v220 = v542;
        v221 = sub_1648A60(56, 3);
        v70 = (_BYTE *)v221;
        if ( v221 )
          sub_15FA480(v221, v220, v219, v218, &v551, 0);
        goto LABEL_131;
      case 8:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542)
          || (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v539, v532, *v542, (__int64 *)&v543)
          || sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v548) )
        {
          goto LABEL_663;
        }
        if ( *(_BYTE *)(*v542 + 8) != 16 || *(_BYTE *)(*v543 + 8) != 16 )
          goto LABEL_388;
        v553[0] = 257;
        v155 = sub_1648A60(56, 3);
        v70 = (_BYTE *)v155;
        if ( v155 )
          sub_15FA660(v155, v542, v543, v548, &v551, 0);
        goto LABEL_131;
      case 9:
      case 28:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543)
          || (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v542, v532, *v543, (__int64 *)&v548) )
        {
          goto LABEL_663;
        }
        v42 = v543;
        v43 = (int)v542;
        v511 = *(_QWORD *)&v554[8 * (unsigned int)v542];
        v44 = *(_BYTE *)(*v543 + 8);
        if ( v44 == 16 )
          v44 = *(_BYTE *)(**(_QWORD **)(*v543 + 16) + 8LL);
        v45 = v555;
        v46 = 0;
        if ( (unsigned __int8)(v44 - 1) <= 5u )
        {
          v47 = (_DWORD)v542 + 1;
          v46 = 0;
          if ( (int)v542 + 1 < (unsigned int)v555 )
          {
            LODWORD(v542) = (_DWORD)v542 + 1;
            v46 = sub_14E9B90(*(_QWORD *)&v554[8 * v47]);
            v43 = v48;
          }
        }
        if ( v43 + 1 != v45 )
          goto LABEL_663;
        v49 = *(_BYTE *)(*v42 + 8);
        if ( v49 == 16 )
          v49 = *(_BYTE *)(**(_QWORD **)(*v42 + 16) + 8LL);
        v553[0] = 257;
        if ( (unsigned __int8)(v49 - 1) > 5u )
        {
          v50 = (_BYTE *)sub_1648A60(56, 2);
          if ( v50 )
          {
            v342 = (int)v543;
            v507 = (int)v548;
            v343 = (_QWORD *)*v543;
            if ( *(_BYTE *)(*v543 + 8) == 16 )
            {
              v344 = v343[4];
              v345 = sub_1643320(*v343);
              v346 = sub_16463B0(v345, (unsigned int)v344);
            }
            else
            {
              v346 = sub_1643320(*v343);
            }
            sub_15FEC10((_DWORD)v50, v346, 51, v511, v342, v507, (__int64)&v551, 0);
          }
        }
        else
        {
          v50 = (_BYTE *)sub_1648A60(56, 2);
          if ( v50 )
          {
            v51 = (int)v543;
            v505 = (int)v548;
            v52 = (_QWORD *)*v543;
            if ( *(_BYTE *)(*v543 + 8) == 16 )
            {
              v53 = v52[4];
              v54 = sub_1643320(*v52);
              v55 = sub_16463B0(v54, (unsigned int)v53);
            }
            else
            {
              v55 = sub_1643320(*v52);
            }
            sub_15FEC10((_DWORD)v50, v55, 52, v511, v51, v505, (__int64)&v551, 0);
          }
        }
        v541 = v50;
        if ( v46 )
          sub_15F2440(v541, v46);
        goto LABEL_100;
      case 10:
        if ( !(_DWORD)v555 )
        {
          v68 = *(_QWORD *)(v11 + 432);
          v69 = sub_1648A60(56, 0);
          v70 = (_BYTE *)v69;
          if ( v69 )
            sub_15F6F90(v69, v68, 0, 0);
LABEL_131:
          v541 = v70;
          goto LABEL_100;
        }
        LODWORD(v543) = 0;
        v548 = 0;
        if ( !sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v543, v532, (__int64 *)&v548)
          && (_DWORD)v543 == (_DWORD)v555 )
        {
          v313 = v548;
          v314 = *(_QWORD *)(v11 + 432);
          v315 = sub_1648A60(56, v548 != 0);
          v70 = (_BYTE *)v315;
          if ( v315 )
            sub_15F6F90(v315, v314, v313, 0);
          goto LABEL_131;
        }
        goto LABEL_663;
      case 11:
        if ( (v555 & 0xFFFFFFFD) != 1 )
          goto LABEL_663;
        v126 = *(_QWORD *)(v11 + 1368);
        v127 = *(unsigned int *)v554;
        v128 = (*(_QWORD *)(v11 + 1376) - v126) >> 3;
        if ( v128 <= v127 )
          goto LABEL_663;
        v129 = *(_QWORD *)(v126 + 8 * v127);
        if ( !v129 )
          goto LABEL_663;
        if ( (_DWORD)v555 == 1 )
        {
          v438 = sub_1648A60(56, 1);
          v70 = (_BYTE *)v438;
          if ( v438 )
            sub_15F8320(v438, v129, 0);
        }
        else
        {
          v130 = *((unsigned int *)v554 + 2);
          v131 = *(_QWORD *)(v11 + 432);
          if ( v128 <= v130 )
          {
            v304 = v11;
            v3 = a1;
            v464 = sub_1643320(v131);
            sub_14EA380(v304, (__int64)&v554, 2u, v532, v464);
            goto LABEL_820;
          }
          v132 = *(_QWORD *)(v126 + 8 * v130);
          v133 = sub_1643320(v131);
          v134 = sub_14EA380(v11, (__int64)&v554, 2u, v532, v133);
          v135 = v134;
          if ( !v132 || !v134 )
          {
            v304 = v11;
            v3 = a1;
            goto LABEL_820;
          }
          v136 = sub_1648A60(56, 3);
          v70 = (_BYTE *)v136;
          if ( v136 )
            sub_15F83E0(v136, v129, v132, v135, 0);
        }
        goto LABEL_131;
      case 12:
        if ( *(_QWORD *)v554 >> 16 == 1205 )
        {
          v407 = sub_14EFEB0((_QWORD *)v11, *((_QWORD *)v554 + 1));
          v503 = *(_DWORD *)(v407 + 8) >> 8;
          v408 = sub_14EA380(v11, (__int64)&v554, 2u, v532, v407);
          v409 = *(_QWORD *)(v11 + 1368);
          v410 = v408;
          v411 = *((unsigned int *)v554 + 6);
          if ( (*(_QWORD *)(v11 + 1376) - v409) >> 3 <= v411 )
            goto LABEL_663;
          v412 = *(_QWORD *)(v409 + 8 * v411);
          if ( !v408 || !v412 )
            goto LABEL_663;
          v413 = *((_QWORD *)v554 + 4);
          v414 = sub_1648B60(64);
          v60 = (const char *)v414;
          if ( v414 )
            sub_15FFAB0(v414, v410, v412, (unsigned int)v413, 0);
          v551 = v60;
          v56 = (__int64)&v551;
          sub_14EF3D0(v11 + 648, &v551);
          if ( (_DWORD)v413 )
          {
            v471 = v60;
            v475 = 0;
            v522 = 5;
            do
            {
              v415 = (__int64)v554;
              v551 = (const char *)v553;
              v552 = 0x100000000LL;
              v416 = v522 + 1;
              v483 = *(_QWORD *)&v554[8 * v522];
              if ( v483 )
              {
                for ( i = 0; i != v483; ++i )
                {
                  v418 = v416 + 1;
                  v422 = *(_QWORD *)(v415 + 8LL * v416);
                  v544 = 1;
                  v543 = 0;
                  v495 = v422;
                  if ( v503 > 0x40 )
                  {
                    v417 = v418;
                    v418 = v416 + 2;
                    v489 = *(_QWORD *)(v415 + 8 * v417);
                    v419 = v489;
                  }
                  else
                  {
                    v489 = 1;
                    v419 = 1;
                  }
                  sub_14EA060((__int64)&v548, v415 + 8LL * v418, v419, v503);
                  if ( v544 > 0x40 && v543 )
                    j_j___libc_free_0_0(v543);
                  v543 = (__int64 *)v548;
                  v416 = v418 + v489;
                  v544 = (unsigned int)v549;
                  if ( v495 )
                  {
                    v56 = (__int64)&v543;
                    v420 = sub_159C0E0(*(_QWORD *)(v11 + 432), &v543);
                    v421 = (unsigned int)v552;
                    if ( (unsigned int)v552 >= HIDWORD(v552) )
                    {
                      v56 = (__int64)v553;
                      sub_16CD150(&v551, v553, 0, 8);
                      v421 = (unsigned int)v552;
                    }
                    *(_QWORD *)&v551[8 * v421] = v420;
                    LODWORD(v552) = v552 + 1;
                  }
                  else
                  {
                    if ( v503 <= 0x40 )
                    {
                      v425 = 1;
                      v424 = 1;
                    }
                    else
                    {
                      v423 = v416++;
                      v424 = *(_QWORD *)&v554[8 * v423];
                      v425 = v424;
                    }
                    v426 = v416;
                    v416 += v424;
                    sub_14EA060((__int64)&v548, (__int64)&v554[8 * v426], v425, v503);
                    while ( 1 )
                    {
                      v56 = (__int64)&v548;
                      if ( (int)sub_16A9900(&v543, &v548) > 0 )
                        break;
                      v427 = sub_159C0E0(*(_QWORD *)(v11 + 432), &v543);
                      v428 = (unsigned int)v552;
                      if ( (unsigned int)v552 >= HIDWORD(v552) )
                      {
                        sub_16CD150(&v551, v553, 0, 8);
                        v428 = (unsigned int)v552;
                      }
                      *(_QWORD *)&v551[8 * v428] = v427;
                      LODWORD(v552) = v552 + 1;
                      sub_16A7400(&v543);
                    }
                    if ( (unsigned int)v549 > 0x40 && v548 )
                      j_j___libc_free_0_0(v548);
                  }
                  if ( v544 > 0x40 && v543 )
                    j_j___libc_free_0_0(v543);
                  v415 = (__int64)v554;
                }
              }
              v522 = v416 + 1;
              v58 = *(unsigned int *)(v415 + 8LL * v416);
              v429 = *(_QWORD *)(v11 + 1368);
              if ( (*(_QWORD *)(v11 + 1376) - v429) >> 3 <= v58 )
                v430 = 0;
              else
                v430 = *(_QWORD *)(v429 + 8 * v58);
              v431 = v551;
              v57 = (unsigned int)v552;
              v432 = v551;
              if ( v551 != &v551[8 * (unsigned int)v552] )
              {
                v496 = v11;
                v433 = &v551[8 * (unsigned int)v552];
                do
                {
                  v56 = *(_QWORD *)v432;
                  v432 += 8;
                  sub_15FFFB0(v471, v56, v430);
                }
                while ( v433 != v432 );
                v11 = v496;
                v431 = v551;
              }
              if ( v431 != (const char *)v553 )
                _libc_free((unsigned __int64)v431);
              ++v475;
            }
            while ( (_DWORD)v413 != v475 );
            v60 = v471;
          }
        }
        else
        {
          if ( (unsigned int)v555 <= 2 )
            goto LABEL_663;
          if ( (v555 & 1) == 0 )
            goto LABEL_663;
          v113 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
          v114 = sub_14EA380(v11, (__int64)&v554, 1u, v532, v113);
          v115 = *(_QWORD *)(v11 + 1368);
          v116 = v114;
          v117 = *((unsigned int *)v554 + 4);
          if ( (*(_QWORD *)(v11 + 1376) - v115) >> 3 <= v117 )
            goto LABEL_663;
          v118 = *(_QWORD *)(v115 + 8 * v117);
          if ( v114 == 0 || v113 == 0 || !v118 )
            goto LABEL_663;
          v498 = ((unsigned __int64)(unsigned int)v555 - 3) >> 1;
          v119 = sub_1648B60(64);
          v60 = (const char *)v119;
          if ( v119 )
            sub_15FFAB0(v119, v116, v118, (unsigned int)v498, 0);
          v551 = v60;
          v120 = 0;
          v56 = (__int64)&v551;
          sub_14EF3D0(v11 + 648, &v551);
          if ( (_DWORD)v498 )
          {
            while ( 1 )
            {
              v124 = *(_QWORD *)&v554[8 * (2 * v120 + 3)];
              if ( *(_BYTE *)(v113 + 8) == 8 )
              {
                v125 = sub_1521F50(v535, v124);
                v56 = sub_1628DA0(*(_QWORD *)v113, v125);
              }
              else
              {
                v56 = sub_1522F40(v11 + 552, v124);
              }
              if ( v56 && *(_BYTE *)(v56 + 16) != 13 )
                v56 = 0;
              v121 = *(unsigned int *)&v554[8 * (2 * v120 + 4)];
              v122 = *(_QWORD *)(v11 + 1368);
              if ( (*(_QWORD *)(v11 + 1376) - v122) >> 3 <= v121 )
                break;
              v123 = *(_QWORD *)(v122 + 8 * v121);
              if ( !v56 || !v123 )
                break;
              ++v120;
              sub_15FFFB0(v60, v56, v123);
              if ( (_DWORD)v498 == v120 )
                goto LABEL_382;
            }
            v364 = v11;
            v3 = a1;
            if ( v60 )
            {
              sub_15F2000(v60);
              sub_1648B90(v60);
            }
            v551 = "Invalid record";
            v553[0] = 259;
            sub_14EE4B0(a1, v364 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
LABEL_382:
        v541 = v60;
        goto LABEL_102;
      case 13:
        if ( (unsigned int)v555 <= 3 )
          goto LABEL_663;
        v539 = 1;
        v205 = *(_QWORD *)(v11 + 1296);
        v514 = 0;
        v206 = (unsigned int)*(_QWORD *)v554 - 1;
        if ( v206 < (*(_QWORD *)(v11 + 1304) - v205) >> 3 )
          v514 = *(_QWORD *)(v205 + 8 * v206);
        v207 = *((_QWORD *)v554 + 1);
        v208 = *(_QWORD *)(v11 + 1368);
        v539 = 3;
        v209 = *((unsigned int *)v554 + 4);
        v480 = v207;
        v210 = (*(_QWORD *)(v11 + 1376) - v208) >> 3;
        if ( v210 <= v209 )
          LODWORD(v499) = 0;
        else
          v499 = *(_QWORD *)(v208 + 8 * v209);
        v539 = 4;
        v211 = *((unsigned int *)v554 + 6);
        if ( v210 <= v211 )
          LODWORD(v491) = 0;
        else
          v491 = *(_QWORD *)(v208 + 8 * v211);
        if ( (v480 & 0x2000) != 0 )
        {
          v539 = 5;
          v212 = sub_14EFEB0((_QWORD *)v11, *((_QWORD *)v554 + 4));
          if ( *(_BYTE *)(v212 + 8) != 12 )
          {
            v449 = v11;
            v3 = a1;
            v551 = "Explicit invoke type is not a function type";
            v553[0] = 259;
            sub_14EE4B0(a1, v449 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        else
        {
          v212 = 0;
        }
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542) )
          goto LABEL_663;
        if ( *(_BYTE *)(*v542 + 8) != 15 )
        {
          v367 = v11;
          v3 = a1;
          v551 = "Callee is not a pointer";
          v553[0] = 259;
          sub_14EE4B0(a1, v367 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v213 = *(_QWORD *)(*v542 + 24);
        if ( v212 )
        {
          if ( v212 != v213 )
          {
            v467 = v11;
            v3 = a1;
            v551 = "Explicit invoke type does not match pointee type of callee operand";
            v553[0] = 259;
            sub_14EE4B0(a1, v467 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        else if ( *(_BYTE *)(v213 + 8) != 12 )
        {
LABEL_795:
          v459 = v11;
          v3 = a1;
          v551 = "Callee is not of pointer to function type";
          v553[0] = 259;
          sub_14EE4B0(a1, v459 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v214 = v539;
        if ( v539 + *(_DWORD *)(v213 + 12) - 1 > (unsigned int)v555 )
        {
LABEL_837:
          v462 = v11;
          v3 = a1;
          v551 = "Insufficient operands to call";
          v553[0] = 259;
          sub_14EE4B0(a1, v462 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v551 = (const char *)v553;
        v552 = 0x1000000000LL;
        v215 = *(_DWORD *)(v213 + 12);
        if ( v215 != 1 )
        {
          v216 = v11;
          v484 = 8LL * (unsigned int)(v215 - 2) + 16;
          v217 = 8;
          while ( 1 )
          {
            v548 = (const char *)sub_14EA380(
                                   v11,
                                   (__int64)&v554,
                                   v214,
                                   v532,
                                   *(_QWORD *)(*(_QWORD *)(v213 + 16) + v217));
            sub_12A9700((__int64)&v551, &v548);
            if ( !*(_QWORD *)&v551[8 * (unsigned int)v552 - 8] )
              break;
            v217 += 8;
            v214 = ++v539;
            if ( v484 == v217 )
              goto LABEL_529;
          }
          v3 = a1;
          v548 = "Invalid record";
          LOWORD(v550) = 259;
          sub_14EE4B0(a1, v216 + 8, (__int64)&v548);
LABEL_342:
          v86 = v551;
          if ( v551 != (const char *)v553 )
            goto LABEL_773;
          goto LABEL_21;
        }
LABEL_529:
        if ( *(_DWORD *)(v213 + 8) >> 8 )
        {
          while ( v539 != (_DWORD)v555 )
          {
            if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v543) )
              goto LABEL_703;
            sub_12A9700((__int64)&v551, &v543);
          }
        }
        else if ( (_DWORD)v555 != v539 )
        {
LABEL_703:
          v406 = v11;
          v3 = a1;
          v548 = "Invalid record";
          LOWORD(v550) = 259;
          sub_14EE4B0(a1, v406 + 8, (__int64)&v548);
          goto LABEL_342;
        }
        LOWORD(v550) = 257;
        v316 = (int)v542;
        v317 = v551;
        v487 = (unsigned int)v552;
        v318 = *(_QWORD *)(*v542 + 24);
        if ( v545 == v546 )
        {
          v320 = 0;
        }
        else
        {
          v319 = v545;
          v320 = 0;
          do
          {
            v321 = v319[2].m128i_i64[1] - v319[2].m128i_i64[0];
            v319 = (__m128i *)((char *)v319 + 56);
            v320 += v321 >> 3;
          }
          while ( v546 != v319 );
        }
        v469 = v545;
        v472 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v546 - (char *)v545) >> 3);
        v478 = v320 + v552 + 3;
        v322 = sub_1648AB0(72, v478, 1840700272 * (unsigned int)(((char *)v546 - (char *)v545) >> 3));
        v323 = v322;
        if ( v322 )
        {
          sub_15F1EA0(v322, **(_QWORD **)(v318 + 16), 5, v322 - 24LL * v478, v478, 0);
          *(_QWORD *)(v323 + 56) = 0;
          sub_15F6500(v323, v318, v316, v499, v491, (unsigned int)&v548, (__int64)v317, v487, (__int64)v469, v472);
        }
        v541 = (_BYTE *)v323;
        sub_14EF420(&v545);
        v56 = (__int64)&v541;
        sub_14EF3D0(v11 + 648, &v541);
        v93 = v541;
        v57 = *((unsigned __int16 *)v541 + 9);
        LOWORD(v57) = v57 & 0x8003;
        *((_WORD *)v541 + 9) = v57 | (4 * (v480 & 0x3FF));
        *((_QWORD *)v93 + 7) = v514;
        if ( v551 != (const char *)v553 )
        {
          _libc_free((unsigned __int64)v551);
LABEL_538:
          v93 = v541;
        }
LABEL_165:
        v60 = v93;
LABEL_102:
        if ( !v528 )
        {
          v335 = v11;
          v3 = a1;
          sub_164BEC0(v541, v56, v57, v58, v59);
          v551 = "Invalid instruction with no BB";
          v553[0] = 259;
          sub_14EE4B0(a1, v335 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        if ( v546 != v545 )
        {
          v362 = v11;
          v3 = a1;
          sub_164BEC0(v60, v56, v57, v58, v59);
          v551 = "Operand bundles found with no consumer";
          v553[0] = 259;
          sub_14EE4B0(a1, v362 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        sub_157E9D0(v528 + 40, v60);
        v61 = *(_QWORD *)(v528 + 40);
        *((_QWORD *)v60 + 4) = v528 + 40;
        v62 = v541;
        *((_QWORD *)v60 + 3) = v61 & 0xFFFFFFFFFFFFFFF8LL | *((_QWORD *)v60 + 3) & 7LL;
        *(_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v60 + 24;
        *(_QWORD *)(v528 + 40) = *(_QWORD *)(v528 + 40) & 7LL | (unsigned __int64)(v60 + 24);
        if ( (unsigned int)v62[16] - 25 <= 9 )
        {
          v63 = *(_QWORD *)(v11 + 1368);
          v528 = 0;
          if ( ++v531 < (unsigned __int64)((*(_QWORD *)(v11 + 1376) - v63) >> 3) )
            v528 = *(_QWORD *)(v63 + 8LL * v531);
        }
        if ( *(_BYTE *)(*(_QWORD *)v62 + 8LL) )
          sub_15234A0(v11 + 552, v62, v532++);
        continue;
      case 15:
        v112 = sub_1648A60(56, 0);
        v70 = (_BYTE *)v112;
        if ( v112 )
          sub_15F82A0(v112, *(_QWORD *)(v11 + 432), 0);
        goto LABEL_131;
      case 16:
        if ( !(_DWORD)v555 )
          goto LABEL_663;
        if ( (v555 & 1) == 0 )
          goto LABEL_663;
        v182 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
        if ( !v182 )
          goto LABEL_663;
        v183 = (unsigned int)v555;
        v553[0] = 257;
        v184 = sub_1648B60(64);
        v185 = (unsigned __int64)(v183 - 1) >> 1;
        v60 = (const char *)v184;
        if ( v184 )
        {
          sub_15F1EA0(v184, v182, 53, 0, 0, 0);
          *((_DWORD *)v60 + 14) = v185;
          sub_164B780(v60, &v551);
          sub_1648880(v60, *((unsigned int *)v60 + 14), 1);
        }
        v56 = (__int64)&v551;
        v551 = v60;
        sub_14EF3D0(v11 + 648, &v551);
        v513 = v555 - 1;
        if ( (_DWORD)v555 == 1 )
          goto LABEL_382;
        v186 = v11;
        v187 = 0;
        while ( 1 )
        {
          v204 = v187 + 1;
          if ( *(_BYTE *)(v186 + 1656) )
          {
            v188 = (__int64)v554;
            if ( (_DWORD)v555 == v204 )
            {
              v191 = 0;
            }
            else
            {
              v189 = *(_QWORD *)&v554[8 * v204];
              if ( (v189 & 1) != 0 )
              {
                v190 = v532 + (unsigned int)(v189 >> 1);
                if ( v189 == 1 )
                  v190 = v532;
              }
              else
              {
                v190 = v532 - (unsigned int)(v189 >> 1);
              }
              if ( *(_BYTE *)(v182 + 8) == 8 )
              {
                v312 = sub_1521F50(v535, v190);
                v191 = sub_1628DA0(*(_QWORD *)v182, v312);
              }
              else
              {
                v191 = sub_1522F40(v186 + 552, v190);
              }
              v188 = (__int64)v554;
            }
          }
          else
          {
            v191 = sub_14EA380(v186, (__int64)&v554, v204, v532, v182);
            v188 = (__int64)v554;
          }
          v192 = *(unsigned int *)(v188 + 8LL * (unsigned int)(v187 + 2));
          v187 += 2;
          v193 = *(_QWORD *)(v186 + 1368);
          if ( (*(_QWORD *)(v186 + 1376) - v193) >> 3 <= v192 )
            break;
          v58 = *(_QWORD *)(v193 + 8 * v192);
          if ( !v191 || !v58 )
            break;
          v194 = *((_DWORD *)v60 + 5) & 0xFFFFFFF;
          if ( v194 == *((_DWORD *)v60 + 14) )
          {
            v486 = v58;
            v493 = v191;
            sub_15F55D0(v60);
            v58 = v486;
            v191 = v493;
            v194 = *((_DWORD *)v60 + 5) & 0xFFFFFFF;
          }
          v195 = (v194 + 1) & 0xFFFFFFF;
          v196 = v195 | *((_DWORD *)v60 + 5) & 0xF0000000;
          *((_DWORD *)v60 + 5) = v196;
          if ( (v196 & 0x40000000) != 0 )
            v197 = (const char *)*((_QWORD *)v60 - 1);
          else
            v197 = &v60[-24 * v195];
          v198 = (__int64 *)&v197[24 * (unsigned int)(v195 - 1)];
          if ( *v198 )
          {
            v199 = v198[1];
            v200 = v198[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v200 = v199;
            if ( v199 )
              *(_QWORD *)(v199 + 16) = *(_QWORD *)(v199 + 16) & 3LL | v200;
          }
          *v198 = v191;
          v201 = *(_QWORD *)(v191 + 8);
          v198[1] = v201;
          if ( v201 )
            *(_QWORD *)(v201 + 16) = (unsigned __int64)(v198 + 1) | *(_QWORD *)(v201 + 16) & 3LL;
          v59 = v198[2] & 3 | (v191 + 8);
          v198[2] = v59;
          *(_QWORD *)(v191 + 8) = v198;
          v202 = *((_DWORD *)v60 + 5) & 0xFFFFFFF;
          v203 = (unsigned int)(v202 - 1);
          if ( (v60[23] & 0x40) != 0 )
            v56 = *((_QWORD *)v60 - 1);
          else
            v56 = (__int64)&v60[-24 * v202];
          v57 = 3LL * *((unsigned int *)v60 + 14);
          *(_QWORD *)(v56 + 8 * v203 + 24LL * *((unsigned int *)v60 + 14) + 8) = v58;
          if ( v187 == v513 )
          {
            v11 = v186;
            goto LABEL_382;
          }
        }
        v3 = a1;
        v551 = "Invalid record";
        v553[0] = 259;
        sub_14EE4B0(a1, v186 + 8, (__int64)&v551);
        goto LABEL_21;
      case 19:
        if ( (_DWORD)v555 != 4 )
          goto LABEL_663;
        v175 = *((_QWORD *)v554 + 3);
        v512 = v175;
        v176 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
        v177 = v176;
        if ( (v175 & 0x40) != 0 )
          goto LABEL_287;
        if ( !v176 || *(_BYTE *)(v176 + 8) != 15 )
        {
          v366 = v11;
          v3 = a1;
          v551 = "Old-style alloca with a non-pointer type";
          v553[0] = 259;
          sub_14EE4B0(a1, v366 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v177 = *(_QWORD *)(v176 + 24);
LABEL_287:
        v178 = sub_14EFEB0((_QWORD *)v11, *((_QWORD *)v554 + 1));
        v179 = *((_QWORD *)v554 + 2);
        if ( v178 && *(_BYTE *)(v178 + 8) == 8 )
        {
          v460 = sub_1521F50(v535, v179);
          v180 = sub_1628DA0(*(_QWORD *)v178, v460);
        }
        else
        {
          v180 = sub_1522F40(v11 + 552, v179);
        }
        v181 = v175;
        LOBYTE(v181) = v175 & 0x1F;
        sub_14EEAA0((__int64 *)&v551, v11, v181, (int *)&v548);
        v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
        if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_517;
        v551 = 0;
        sub_14ECA90((__int64 *)&v551);
        if ( !v177 || !v180 )
          goto LABEL_663;
        v308 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(v11 + 440)) + 4);
        v553[0] = 257;
        v506 = v308;
        v309 = sub_1648A60(64, 1);
        v310 = v309;
        if ( v309 )
          sub_15F8A50(v309, v177, v506, v180, (_DWORD)v548, (unsigned int)&v551, 0);
        v311 = *(_WORD *)(v310 + 18);
        v541 = (_BYTE *)v310;
        *(_WORD *)(v310 + 18) = (v512 >> 7 << 6) | v512 & 0x20 | v311 & 0xFF9F;
        goto LABEL_100;
      case 20:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v548) )
          goto LABEL_663;
        v172 = (unsigned int)v542;
        v173 = (_DWORD)v542 + 3;
        if ( (_DWORD)v542 + 2 == (_DWORD)v555 )
        {
          if ( v173 != (_DWORD)v542 + 2 )
          {
            sub_14EEB00((__int64 *)&v551, v11, 0, *(_QWORD *)v548);
            v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
            if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              goto LABEL_517;
            v551 = 0;
            sub_14ECA90((__int64 *)&v551);
LABEL_612:
            v174 = *(_QWORD *)(*(_QWORD *)v548 + 24LL);
LABEL_280:
            sub_14EEAA0((__int64 *)&v551, v11, *(_QWORD *)&v554[8 * (unsigned int)v542], (int *)&v543);
            v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
            if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) == 0 )
            {
              v551 = 0;
              sub_14ECA90((__int64 *)&v551);
              v553[0] = 257;
              v293 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 1)] != 0;
              v294 = sub_1648A60(64, 1);
              v70 = (_BYTE *)v294;
              if ( v294 )
                sub_15F90A0(v294, v174, (_DWORD)v548, (unsigned int)&v551, v293, (_DWORD)v543, 0);
              goto LABEL_131;
            }
LABEL_517:
            v3 = a1;
            v551 = (const char *)(v22 | 1);
            *a1 = 0;
            sub_14ECA50(a1, &v551);
            sub_14ECA90((__int64 *)&v551);
            goto LABEL_21;
          }
LABEL_278:
          LODWORD(v542) = (_DWORD)v542 + 1;
          v174 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)&v554[8 * v172]);
          sub_14EEB00((__int64 *)&v551, v11, v174, *(_QWORD *)v548);
          v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_517;
          v551 = 0;
          sub_14ECA90((__int64 *)&v551);
          if ( v174 )
            goto LABEL_280;
          goto LABEL_612;
        }
        if ( v173 == (_DWORD)v555 )
          goto LABEL_278;
        goto LABEL_663;
      case 23:
        if ( (unsigned int)v555 <= 2 )
          goto LABEL_663;
        v163 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
        v164 = sub_14EA380(v11, (__int64)&v554, 1u, v532, v163);
        v165 = sub_14EFEB0((_QWORD *)v11, *((_QWORD *)v554 + 2));
        if ( v164 == 0 || v163 == 0 || !v165 )
          goto LABEL_663;
        v553[0] = 257;
        v166 = sub_1648A60(56, 1);
        v167 = (_BYTE *)v166;
        if ( v166 )
        {
          v168 = v166 - 24;
          sub_15F1EA0(v166, v165, 58, v166 - 24, 1, 0);
          if ( *((_QWORD *)v167 - 3) )
          {
            v169 = *((_QWORD *)v167 - 2);
            v170 = *((_QWORD *)v167 - 1) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v170 = v169;
            if ( v169 )
              *(_QWORD *)(v169 + 16) = v170 | *(_QWORD *)(v169 + 16) & 3LL;
          }
          *((_QWORD *)v167 - 3) = v164;
          v171 = *(_QWORD *)(v164 + 8);
          *((_QWORD *)v167 - 2) = v171;
          if ( v171 )
            *(_QWORD *)(v171 + 16) = (unsigned __int64)(v167 - 16) | *(_QWORD *)(v171 + 16) & 3LL;
          *((_QWORD *)v167 - 1) = *((_QWORD *)v167 - 1) & 3LL | (v164 + 8);
          *(_QWORD *)(v164 + 8) = v168;
          sub_164B780(v167, &v551);
        }
        goto LABEL_274;
      case 24:
      case 44:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v548) )
          goto LABEL_663;
        if ( v13 == 44 )
        {
          if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v543) )
            goto LABEL_663;
        }
        else if ( (unsigned __int8)sub_14EA9B0(
                                     v11,
                                     (__int64)&v554,
                                     &v539,
                                     v532,
                                     *(_QWORD *)(*(_QWORD *)v548 + 24LL),
                                     (__int64 *)&v543) )
        {
          goto LABEL_663;
        }
        if ( v539 + 2 == (_DWORD)v555 )
        {
          sub_14EEB00((__int64 *)&v551, v11, *v543, *(_QWORD *)v548);
          v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) == 0 )
          {
            v551 = 0;
            sub_14ECA90((__int64 *)&v551);
            sub_14EEAA0((__int64 *)&v551, v11, *(_QWORD *)&v554[8 * v539], (int *)&v542);
            v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
            if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) == 0 )
            {
              v551 = 0;
              sub_14ECA90((__int64 *)&v551);
              v295 = *(_QWORD *)&v554[8 * v539 + 8] != 0;
              v296 = sub_1648A60(64, 2);
              v70 = (_BYTE *)v296;
              if ( v296 )
                sub_15F9630(v296, v543, v548, v295, (unsigned int)v542, 0);
              goto LABEL_131;
            }
          }
          goto LABEL_517;
        }
        goto LABEL_663;
      case 26:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543) )
          goto LABEL_663;
        v258 = (unsigned int)v542;
        if ( (_DWORD)v542 == (_DWORD)v555 )
        {
          v466 = v11;
          v3 = a1;
          v551 = "EXTRACTVAL: Invalid instruction with 0 indices";
          v553[0] = 259;
          sub_14EE4B0(a1, v466 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v551 = (const char *)v553;
        v552 = 0x400000000LL;
        v259 = v555;
        v260 = *v543;
        while ( 2 )
        {
          v261 = *(_BYTE *)(v260 + 8);
          v262 = *(_QWORD *)&v554[8 * v258];
          if ( (unsigned __int8)(v261 - 13) > 1u )
          {
            v445 = v11;
            BYTE1(v550) = 1;
            v3 = a1;
            v446 = "EXTRACTVAL: Invalid type";
            goto LABEL_772;
          }
          if ( v262 != (unsigned int)v262 )
          {
            BYTE1(v550) = 1;
            v445 = v11;
            v3 = a1;
            v446 = "Invalid value";
            goto LABEL_772;
          }
          if ( v261 != 13 )
          {
            if ( v262 >= *(_QWORD *)(v260 + 32) )
            {
              v445 = v11;
              BYTE1(v550) = 1;
              v3 = a1;
              v446 = "EXTRACTVAL: Invalid array index";
              goto LABEL_772;
            }
            if ( HIDWORD(v552) <= (unsigned int)v552 )
              sub_16CD150(&v551, v553, 0, 4);
            *(_DWORD *)&v551[4 * (unsigned int)v552] = v262;
            LODWORD(v552) = v552 + 1;
            v260 = **(_QWORD **)(v260 + 16);
LABEL_416:
            v258 = (unsigned int)((_DWORD)v542 + 1);
            LODWORD(v542) = v258;
            if ( (_DWORD)v258 == v259 )
            {
              LOWORD(v550) = 257;
              v541 = (_BYTE *)sub_14EDF60(v543, (__int64)v551, (unsigned int)v552, (__int64)&v548, 0);
              goto LABEL_561;
            }
            continue;
          }
          break;
        }
        if ( v262 < *(unsigned int *)(v260 + 12) )
        {
          if ( HIDWORD(v552) <= (unsigned int)v552 )
            sub_16CD150(&v551, v553, 0, 4);
          *(_DWORD *)&v551[4 * (unsigned int)v552] = v262;
          LODWORD(v552) = v552 + 1;
          v260 = *(_QWORD *)(*(_QWORD *)(v260 + 16) + 8 * v262);
          goto LABEL_416;
        }
        v445 = v11;
        BYTE1(v550) = 1;
        v3 = a1;
        v446 = "EXTRACTVAL: Invalid struct index";
LABEL_772:
        v548 = v446;
        LOBYTE(v550) = 3;
        sub_14EE4B0(v3, v445 + 8, (__int64)&v548);
        v86 = v551;
        if ( v551 != (const char *)v553 )
LABEL_773:
          _libc_free((unsigned __int64)v86);
LABEL_21:
        if ( v554 != &v556 )
          _libc_free((unsigned __int64)v554);
        v15 = v546;
        v16 = v545;
        if ( v546 != v545 )
        {
          do
          {
            v17 = v16[2].m128i_i64[0];
            if ( v17 )
              j_j___libc_free_0(v17, v16[3].m128i_i64[0] - v17);
            if ( (__m128i *)v16->m128i_i64[0] != &v16[1] )
              j_j___libc_free_0(v16->m128i_i64[0], v16[1].m128i_i64[0] + 1);
            v16 = (__m128i *)((char *)v16 + 56);
          }
          while ( v15 != v16 );
          v16 = v545;
        }
        if ( v16 )
          j_j___libc_free_0(v16, v547 - (_QWORD)v16);
        if ( v540 )
          sub_161E7C0(&v540);
        return v3;
      case 27:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542)
          || sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v543) )
        {
          goto LABEL_663;
        }
        v263 = v555;
        v264 = v539;
        if ( v539 == (_DWORD)v555 )
        {
          v465 = v11;
          v3 = a1;
          v551 = "INSERTVAL: Invalid instruction with 0 indices";
          v553[0] = 259;
          sub_14EE4B0(a1, v465 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v551 = (const char *)v553;
        v552 = 0x400000000LL;
        v265 = *v542;
        do
        {
          v266 = *(_BYTE *)(v265 + 8);
          v267 = *(_QWORD *)&v554[8 * v264];
          if ( (unsigned __int8)(v266 - 13) > 1u )
          {
            v434 = v11;
            BYTE1(v550) = 1;
            v3 = a1;
            v435 = "INSERTVAL: Invalid type";
            goto LABEL_750;
          }
          if ( v267 != (unsigned int)v267 )
          {
            BYTE1(v550) = 1;
            v434 = v11;
            v3 = a1;
            v435 = "Invalid value";
            goto LABEL_750;
          }
          if ( v266 == 13 )
          {
            if ( v267 >= *(unsigned int *)(v265 + 12) )
            {
              v434 = v11;
              BYTE1(v550) = 1;
              v3 = a1;
              v435 = "INSERTVAL: Invalid struct index";
              goto LABEL_750;
            }
            if ( (unsigned int)v552 >= HIDWORD(v552) )
              sub_16CD150(&v551, v553, 0, 4);
            *(_DWORD *)&v551[4 * (unsigned int)v552] = v267;
            LODWORD(v552) = v552 + 1;
            v265 = *(_QWORD *)(*(_QWORD *)(v265 + 16) + 8 * v267);
          }
          else
          {
            if ( v267 >= *(_QWORD *)(v265 + 32) )
            {
              v434 = v11;
              BYTE1(v550) = 1;
              v3 = a1;
              v435 = "INSERTVAL: Invalid array index";
              goto LABEL_750;
            }
            if ( HIDWORD(v552) <= (unsigned int)v552 )
              sub_16CD150(&v551, v553, 0, 4);
            *(_DWORD *)&v551[4 * (unsigned int)v552] = v267;
            LODWORD(v552) = v552 + 1;
            v265 = **(_QWORD **)(v265 + 16);
          }
          v264 = v539 + 1;
          v539 = v264;
        }
        while ( (_DWORD)v264 != v263 );
        v440 = v543;
        if ( v265 != *v543 )
        {
          v434 = v11;
          BYTE1(v550) = 1;
          v3 = a1;
          v435 = "Inserted value type doesn't match aggregate type";
LABEL_750:
          v548 = v435;
          LOBYTE(v550) = 3;
          sub_14EE4B0(v3, v434 + 8, (__int64)&v548);
          v86 = v551;
          if ( v551 != (const char *)v553 )
            goto LABEL_773;
          goto LABEL_21;
        }
        LOWORD(v550) = 257;
        v441 = v551;
        v442 = v542;
        v524 = (unsigned int)v552;
        v443 = sub_1648A60(88, 2);
        v444 = v443;
        if ( v443 )
        {
          sub_15F1EA0(v443, *v442, 63, v443 - 48, 2, 0);
          *(_QWORD *)(v444 + 64) = 0x400000000LL;
          *(_QWORD *)(v444 + 56) = v444 + 72;
          sub_15FAD90(v444, v442, v440, v441, v524, &v548);
        }
        v541 = (_BYTE *)v444;
        goto LABEL_561;
      case 29:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542)
          || (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v539, v532, *v542, (__int64 *)&v543)
          || sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v548) )
        {
          goto LABEL_663;
        }
        v239 = *(_QWORD *)(v11 + 432);
        v240 = *(_QWORD *)v548;
        if ( *(_BYTE *)(*(_QWORD *)v548 + 8LL) == 16 )
        {
          v241 = *(_QWORD *)(v240 + 24);
          if ( v241 != sub_1643320(v239) )
          {
LABEL_388:
            v242 = v11;
            v3 = a1;
            v551 = "Invalid type for value";
            v553[0] = 259;
            sub_14EE4B0(a1, v242 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        else if ( v240 != sub_1643320(v239) )
        {
          goto LABEL_388;
        }
LABEL_174:
        v553[0] = 257;
        v541 = (_BYTE *)sub_14EDD70((__int64)v548, v542, (__int64)v543, (__int64)&v551, 0, 0);
        goto LABEL_100;
      case 31:
        if ( (unsigned int)v555 <= 1 )
          goto LABEL_663;
        v231 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
        v232 = sub_14EA380(v11, (__int64)&v554, 1u, v532, v231);
        v233 = v232;
        if ( !v231 || !v232 )
          goto LABEL_663;
        v234 = v555;
        v235 = sub_1648B60(64);
        v60 = (const char *)v235;
        if ( v235 )
          sub_1600240(v235, v233, (unsigned int)(v234 - 2), 0);
        v551 = v60;
        v56 = (__int64)&v551;
        sub_14EF3D0(v11 + 648, &v551);
        if ( v234 == 2 )
          goto LABEL_382;
        v236 = 2;
        while ( 1 )
        {
          v237 = *(unsigned int *)&v554[8 * v236];
          v238 = *(_QWORD *)(v11 + 1368);
          if ( (*(_QWORD *)(v11 + 1376) - v238) >> 3 <= v237 )
            break;
          v56 = *(_QWORD *)(v238 + 8 * v237);
          if ( !v56 )
            break;
          ++v236;
          sub_1600410(v60);
          if ( v234 == v236 )
            goto LABEL_382;
        }
        v304 = v11;
        v3 = a1;
        if ( v60 )
        {
          sub_15F2000(v60);
          sub_1648B90(v60);
        }
        goto LABEL_820;
      case 33:
        if ( (!v528 || (v230 = *(_QWORD *)(v528 + 40) & 0xFFFFFFFFFFFFFFF8LL, v230 == v528 + 40))
          && (!v531
           || (v324 = *(_QWORD *)(*(_QWORD *)(v11 + 1368) + 8LL * (v531 - 1))) == 0
           || (v230 = *(_QWORD *)(v324 + 40) & 0xFFFFFFFFFFFFFFF8LL, v230 == v324 + 40))
          || !v230 )
        {
          v325 = v11;
          v3 = a1;
          v551 = "Invalid record";
          v541 = 0;
          v553[0] = 259;
          sub_14EE4B0(a1, v325 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v228 = (_BYTE *)(v230 - 24);
        v541 = v228;
        v229 = (__int64 *)(v228 + 48);
        v551 = v540;
        if ( v540 )
        {
          sub_1623A60(&v551, v540, 2);
          if ( v229 != (__int64 *)&v551 )
            goto LABEL_619;
          if ( v551 )
            sub_161E7C0(v228 + 48);
          continue;
        }
        if ( v229 == (__int64 *)&v551 )
          continue;
        goto LABEL_615;
      case 34:
        if ( (unsigned int)v555 <= 2 )
          goto LABEL_663;
        v243 = (__int64)v554;
        v539 = 1;
        v244 = *(_QWORD *)(v11 + 1296);
        v492 = 0;
        v245 = (unsigned int)*(_QWORD *)v554 - 1;
        if ( v245 < (*(_QWORD *)(v11 + 1304) - v244) >> 3 )
          v492 = *(_QWORD *)(v244 + 8 * v245);
        v539 = 2;
        v500 = *((_QWORD *)v554 + 1);
        v516 = v500 & 0x20000;
        if ( (v500 & 0x20000) != 0 )
        {
          v539 = 3;
          v516 = sub_14E9B90(*((_DWORD *)v554 + 4));
          if ( !v516 )
          {
            v461 = v11;
            v3 = a1;
            v551 = "Fast math flags indicator set for call with no FMF";
            v553[0] = 259;
            sub_14EE4B0(a1, v461 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        if ( (v500 & 0x8000) != 0 )
        {
          v246 = v539++;
          v247 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)(v243 + 8 * v246));
          if ( *(_BYTE *)(v247 + 8) != 12 )
          {
            v405 = v11;
            v3 = a1;
            v551 = "Explicit call type is not a function type";
            v553[0] = 259;
            sub_14EE4B0(a1, v405 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        else
        {
          v247 = 0;
        }
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v542) )
          goto LABEL_663;
        if ( *(_BYTE *)(*v542 + 8) != 15 )
        {
          v363 = v11;
          v3 = a1;
          v551 = "Callee is not a pointer type";
          v553[0] = 259;
          sub_14EE4B0(a1, v363 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v248 = *(_QWORD *)(*v542 + 24);
        if ( v247 )
        {
          if ( v247 != v248 )
          {
            v463 = v11;
            v3 = a1;
            v551 = "Explicit call type does not match pointee type of callee operand";
            v553[0] = 259;
            sub_14EE4B0(a1, v463 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        else if ( *(_BYTE *)(v248 + 8) != 12 )
        {
          goto LABEL_795;
        }
        v249 = v539;
        if ( v539 + *(_DWORD *)(v248 + 12) - 1 > (unsigned int)v555 )
          goto LABEL_837;
        v551 = (const char *)v553;
        v552 = 0x1000000000LL;
        v250 = *(_DWORD *)(v248 + 12);
        if ( v250 == 1 )
          goto LABEL_678;
        v251 = v248;
        v252 = v11;
        v253 = 8;
        v485 = 8LL * (unsigned int)(v250 - 2) + 16;
        do
        {
          v257 = *(_QWORD *)(*(_QWORD *)(v251 + 16) + v253);
          if ( *(_BYTE *)(v257 + 8) == 7 )
          {
            v254 = *(unsigned int *)&v554[8 * v249];
            v255 = *(_QWORD *)(v252 + 1368);
            if ( (*(_QWORD *)(v252 + 1376) - v255) >> 3 <= v254 )
              v256 = 0;
            else
              v256 = *(const char **)(v255 + 8 * v254);
          }
          else
          {
            v256 = (const char *)sub_14EA380(v252, (__int64)&v554, v249, v532, v257);
          }
          v548 = v256;
          sub_12A9700((__int64)&v551, &v548);
          if ( !*(_QWORD *)&v551[8 * (unsigned int)v552 - 8] )
          {
            v3 = a1;
            v548 = "Invalid record";
            LOWORD(v550) = 259;
            sub_14EE4B0(a1, v252 + 8, (__int64)&v548);
            goto LABEL_675;
          }
          v253 += 8;
          v249 = ++v539;
        }
        while ( v485 != v253 );
        v11 = v252;
        v248 = v251;
LABEL_678:
        if ( !(*(_DWORD *)(v248 + 8) >> 8) )
        {
          if ( (_DWORD)v555 == v539 )
            goto LABEL_680;
LABEL_697:
          v403 = v11;
          BYTE1(v550) = 1;
          v3 = a1;
          v404 = "Invalid record";
LABEL_824:
          v548 = v404;
          LOBYTE(v550) = 3;
          sub_14EE4B0(v3, v403 + 8, (__int64)&v548);
LABEL_675:
          v86 = v551;
          if ( v551 == (const char *)v553 )
            goto LABEL_21;
          goto LABEL_773;
        }
        while ( v539 != (_DWORD)v555 )
        {
          if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v543) )
            goto LABEL_697;
          sub_12A9700((__int64)&v551, &v543);
        }
LABEL_680:
        LOWORD(v550) = 257;
        v482 = (int)v551;
        v388 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v546 - (char *)v545) >> 3);
        v479 = (unsigned int)v552;
        v488 = (int)v542;
        if ( v545 == v546 )
        {
          v392 = v552 + 1;
          v476 = v545;
          v395 = sub_1648AB0(
                   72,
                   (unsigned int)(v552 + 1),
                   1840700272 * (unsigned int)(((char *)v546 - (char *)v545) >> 3));
          if ( !v395 )
            goto LABEL_687;
          v396 = v476;
          v397 = v388;
          v398 = 0;
        }
        else
        {
          v389 = v545;
          v390 = 0;
          do
          {
            v391 = v389[2].m128i_i64[1] - v389[2].m128i_i64[0];
            v389 = (__m128i *)((char *)v389 + 56);
            v390 += v391 >> 3;
          }
          while ( v546 != v389 );
          v392 = v552 + 1;
          v470 = v545;
          v473 = v546;
          v393 = sub_1648AB0(
                   72,
                   (unsigned int)(v552 + 1 + v390),
                   1840700272 * (unsigned int)(((char *)v546 - (char *)v545) >> 3));
          v394 = v470;
          v395 = v393;
          if ( !v393 )
            goto LABEL_687;
          v396 = v470;
          v397 = v388;
          v398 = 0;
          do
          {
            v399 = v394[2].m128i_i64[1] - v394[2].m128i_i64[0];
            v394 = (__m128i *)((char *)v394 + 56);
            v398 += v399 >> 3;
          }
          while ( v473 != v394 );
        }
        v474 = (__int64)v396;
        v477 = v397;
        sub_15F1EA0(v395, **(_QWORD **)(v248 + 16), 54, v395 - 24 * (v479 + v398) - 24, v398 + v392, 0);
        *(_QWORD *)(v395 + 56) = 0;
        sub_15F5B40(v395, v248, v488, v482, v479, (unsigned int)&v548, v474, v477);
LABEL_687:
        v541 = (_BYTE *)v395;
        sub_14EF420(&v545);
        v56 = (__int64)&v541;
        sub_14EF3D0(v11 + 648, &v541);
        v400 = v541;
        v401 = *((unsigned __int16 *)v541 + 9);
        LOWORD(v401) = v401 & 0x8000;
        v402 = v401 | *((_WORD *)v541 + 9) & 3 | (4 * (((unsigned int)v500 >> 1) & 0x3FF));
        v57 = 2;
        if ( (v500 & 0x4000) == 0 )
          v57 = v500 & 1;
        v58 = 3;
        if ( (v500 & 0x10000) != 0 )
          v57 = 3;
        *((_WORD *)v541 + 9) = v57 | v402 & 0x8FFC;
        *((_QWORD *)v400 + 7) = v492;
        if ( !v516 )
          goto LABEL_650;
        if ( (unsigned __int8)sub_14EE080(v400) )
        {
          v56 = v516;
          sub_15F2440(v400, v516);
          goto LABEL_650;
        }
        BYTE1(v550) = 1;
        v403 = v11;
        v3 = a1;
        v404 = "Fast-math-flags specified for call without floating-point scalar or vector return type";
        goto LABEL_824;
      case 35:
        if ( (!v528 || (v222 = *(_QWORD *)(v528 + 40) & 0xFFFFFFFFFFFFFFF8LL, v222 == v528 + 40))
          && (!v531
           || (v326 = *(_QWORD *)(*(_QWORD *)(v11 + 1368) + 8LL * (v531 - 1))) == 0
           || (v222 = *(_QWORD *)(v326 + 40) & 0xFFFFFFFFFFFFFFF8LL, v222 == v326 + 40))
          || !v222 )
        {
          v541 = 0;
          v304 = v11;
          v3 = a1;
LABEL_820:
          v551 = "Invalid record";
          v553[0] = 259;
          sub_14EE4B0(v3, v304 + 8, (__int64)&v551);
          goto LABEL_21;
        }
        v541 = (_BYTE *)(v222 - 24);
        if ( (unsigned int)v555 <= 3 )
        {
          v304 = v11;
          v3 = a1;
          goto LABEL_820;
        }
        v223 = 0;
        v224 = *(_QWORD *)v554;
        v225 = *((_QWORD *)v554 + 3);
        v515 = *((_QWORD *)v554 + 1);
        v226 = *((_QWORD *)v554 + 2);
        if ( (_DWORD)v226 )
        {
          v223 = sub_1518150(v535, (unsigned int)(v226 - 1));
          if ( !v223 )
            goto LABEL_663;
        }
        if ( (_DWORD)v225 )
        {
          v227 = sub_1518150(v535, (unsigned int)(v225 - 1));
          if ( !v227 )
            goto LABEL_663;
        }
        else
        {
          v227 = 0;
        }
        sub_15C7110(&v551, (unsigned int)v224, (unsigned int)v515, v223, v227);
        if ( v540 )
          sub_161E7C0(&v540);
        v540 = v551;
        if ( v551 )
          sub_1623210(&v551, v551, &v540);
        v228 = v541;
        v229 = (__int64 *)(v541 + 48);
        v551 = v540;
        if ( !v540 )
        {
          if ( v229 == (__int64 *)&v551 )
            continue;
LABEL_615:
          if ( !*((_QWORD *)v228 + 6) )
            continue;
LABEL_616:
          sub_161E7C0(v229);
LABEL_617:
          v365 = v551;
          *((_QWORD *)v228 + 6) = v551;
          if ( v365 )
            sub_1623210(&v551, v365, v229);
          continue;
        }
        sub_1623A60(&v551, v540, 2);
        if ( v229 != (__int64 *)&v551 )
        {
LABEL_619:
          if ( *((_QWORD *)v228 + 6) )
            goto LABEL_616;
          goto LABEL_617;
        }
        if ( v551 )
          sub_161E7C0(&v551);
        continue;
      case 36:
        if ( (_DWORD)v555 != 2 )
          goto LABEL_663;
        if ( (unsigned int)*(_QWORD *)v554 > 5 )
        {
          v140 = 7;
        }
        else
        {
          v140 = dword_42926C0[(unsigned int)*(_QWORD *)v554];
          if ( v140 <= 2 )
            goto LABEL_663;
        }
        v141 = *((_QWORD *)v554 + 1);
        v142 = v141;
        if ( (unsigned int)v141 > 1 )
        {
          v142 = 1;
          if ( (unsigned int)v141 < *(_DWORD *)(v11 + 1768) )
            v142 = *(_BYTE *)(*(_QWORD *)(v11 + 1760) + (unsigned int)v141);
        }
        v143 = sub_1648A60(64, 0);
        v70 = (_BYTE *)v143;
        if ( v143 )
          sub_15F9C80(v143, *(_QWORD *)(v11 + 432), (unsigned int)v140, v142, 0);
        goto LABEL_131;
      case 37:
      case 46:
        v538 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v538, v532, (__int64 *)&v542) )
          goto LABEL_663;
        if ( v13 == 46 )
        {
          if ( sub_14EFF60(v11, (__int64)&v554, &v538, v532, (__int64 *)&v543) )
            goto LABEL_663;
        }
        else if ( (unsigned __int8)sub_14EA9B0(
                                     v11,
                                     (__int64)&v554,
                                     &v538,
                                     v532,
                                     *(_QWORD *)(*v542 + 24),
                                     (__int64 *)&v543) )
        {
          goto LABEL_663;
        }
        if ( !(unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v538, v532, *v543, (__int64 *)&v548)
          && v538 + 3 <= (unsigned int)v555
          && v538 + 5 >= (unsigned int)v555 )
        {
          v64 = 7;
          v65 = *(_QWORD *)&v554[8 * v538 + 8];
          if ( (unsigned int)v65 > 5 || (v64 = dword_42926C0[(unsigned int)v65], v64 > 1) )
          {
            v66 = *(_QWORD *)&v554[8 * v538 + 16];
            v67 = (unsigned __int8)v66;
            if ( (unsigned int)v66 > 1 )
            {
              v67 = 1;
              if ( (unsigned int)v66 < *(_DWORD *)(v11 + 1768) )
                v67 = *(unsigned __int8 *)(*(_QWORD *)(v11 + 1760) + (unsigned int)v66);
            }
            sub_14EEB00((__int64 *)&v551, v11, *v543, *v542);
            v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
            if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) == 0 )
            {
              v551 = 0;
              sub_14ECA90((__int64 *)&v551);
              if ( (unsigned int)v555 > 6 )
              {
                v297 = 7;
                v347 = *(_QWORD *)&v554[8 * v538 + 24];
                if ( (unsigned int)v347 <= 5 )
                  v297 = dword_42926C0[(unsigned int)v347];
              }
              else
              {
                switch ( v64 )
                {
                  case 2:
                  case 5:
                    v297 = 2;
                    break;
                  case 3:
                  case 7:
                    v297 = v64;
                    break;
                  case 4:
                  case 6:
                    v297 = 4;
                    break;
                }
              }
              v298 = sub_1648A60(64, 3);
              v299 = v298;
              if ( v298 )
              {
                v518 = v298;
                sub_15F99E0(v298, (_DWORD)v542, (_DWORD)v543, (_DWORD)v548, v64, v297, v67, 0);
                v299 = v518;
              }
              v300 = (__int64)v554;
              v541 = (_BYTE *)v299;
              v301 = v538;
              v302 = *(_WORD *)(v299 + 18) & 0x8000 | *(_WORD *)(v299 + 18) & 0x7FFE | (*(_QWORD *)&v554[8 * v538] != 0);
              *(_WORD *)(v299 + 18) = v302;
              if ( (unsigned int)v555 <= 7 )
              {
                sub_14F2510(v528 + 40, (__int64 *)(v528 + 40), v299);
                v553[0] = 257;
                v539 = 0;
                v541 = (_BYTE *)sub_14EDF60(v541, (__int64)&v539, 1, (__int64)&v551, 0);
              }
              else
              {
                v303 = v301 + 4;
                LOBYTE(v303) = *(_QWORD *)(v300 + 8 * v303) != 0;
                *(_WORD *)(v299 + 18) = v302 & 0x7EFF | ((_WORD)v303 << 8) | v302 & 0x8000;
              }
              goto LABEL_100;
            }
            goto LABEL_517;
          }
        }
        goto LABEL_663;
      case 38:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543) )
          goto LABEL_663;
        if ( *(_BYTE *)(*v543 + 8) != 15 )
          goto LABEL_663;
        if ( (unsigned __int8)sub_14EA9B0(v11, (__int64)&v554, &v542, v532, *(_QWORD *)(*v543 + 24), (__int64 *)&v548) )
          goto LABEL_663;
        if ( (_DWORD)v542 + 4 != (_DWORD)v555 )
          goto LABEL_663;
        v146 = *(_QWORD *)&v554[8 * (unsigned int)v542];
        if ( (unsigned int)v146 > 0xA )
          goto LABEL_663;
        v348 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 2)];
        if ( (unsigned int)v348 > 5 )
        {
          v349 = 7;
        }
        else
        {
          v349 = dword_42926C0[(unsigned int)v348];
          if ( v349 <= 1 )
            goto LABEL_663;
        }
        v350 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 3)];
        v351 = v350;
        if ( (unsigned int)v350 > 1 )
        {
          if ( (unsigned int)v350 >= *(_DWORD *)(v11 + 1768) )
            v351 = 1;
          else
            v351 = *(_BYTE *)(*(_QWORD *)(v11 + 1760) + (unsigned int)v350);
        }
        v352 = sub_1648A60(64, 2);
        v353 = v352;
        if ( v352 )
          sub_15F9C10(v352, v146, (_DWORD)v543, (_DWORD)v548, v349, v351, 0);
        v354 = *(_WORD *)(v353 + 18);
        v541 = (_BYTE *)v353;
        *(_WORD *)(v353 + 18) = v354 & 0x8000 | v354 & 0x7FFE | (*(_QWORD *)&v554[8 * ((_DWORD)v542 + 1)] != 0);
        goto LABEL_100;
      case 39:
        LODWORD(v543) = 0;
        v548 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v543, v532, (__int64 *)&v548) )
          goto LABEL_663;
        v144 = v548;
        v145 = sub_1648A60(56, 1);
        v70 = (_BYTE *)v145;
        if ( v145 )
          sub_15F7290(v145, v144, 0);
        goto LABEL_131;
      case 40:
      case 47:
        LODWORD(v543) = 0;
        if ( v13 == 47 )
        {
          if ( (unsigned int)v555 <= 2 )
            goto LABEL_663;
          LODWORD(v543) = 1;
          v31 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
          if ( !v31 )
            goto LABEL_663;
        }
        else
        {
          if ( (unsigned int)v555 <= 3 )
            goto LABEL_663;
          LODWORD(v543) = 1;
          v31 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)v554);
          if ( !v31 )
            goto LABEL_663;
          if ( v13 == 40 )
          {
            v548 = 0;
            if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v543, v532, (__int64 *)&v548) )
              goto LABEL_663;
            if ( (*(_BYTE *)(a3 + 18) & 8) != 0 )
            {
              v383 = sub_15E38F0(a3);
              if ( v548 != (const char *)v383 )
              {
                v384 = v11;
                v3 = a1;
                v551 = "Personality function mismatch";
                v553[0] = 259;
                sub_14EE4B0(a1, v384 + 8, (__int64)&v551);
                goto LABEL_21;
              }
            }
            else
            {
              sub_15E3D80(a3, v548, v32, v33, v34);
            }
          }
        }
        v35 = (unsigned int)v543;
        v37 = (_DWORD)v543 + 2;
        LODWORD(v543) = (_DWORD)v543 + 1;
        v36 = (unsigned int)v543;
        v38 = *(_QWORD *)&v554[8 * v35];
        LODWORD(v543) = v37;
        v39 = *(_QWORD *)&v554[8 * v36];
        v553[0] = 257;
        v504 = v39;
        v490 = v39;
        v510 = sub_15F5910(v31, (unsigned int)v39, &v551, 0);
        v40 = 0;
        *(_WORD *)(v510 + 18) = (v38 != 0) | *(_WORD *)(v510 + 18) & 0xFFFE;
        if ( v490 )
        {
          do
          {
            LODWORD(v543) = (_DWORD)v543 + 1;
            if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v543, v532, (__int64 *)&v548) )
            {
              v41 = v11;
              v3 = a1;
              sub_15F2000(v510);
              sub_1648B90(v510);
              v551 = "Invalid record";
              v553[0] = 259;
              sub_14EE4B0(a1, v41 + 8, (__int64)&v551);
              goto LABEL_21;
            }
            ++v40;
            sub_15F5A60(v510, v548);
          }
          while ( v504 != v40 );
        }
        v541 = (_BYTE *)v510;
LABEL_100:
        v56 = (__int64)&v541;
        sub_14EF3D0(v11 + 648, &v541);
        goto LABEL_101;
      case 41:
        LODWORD(v542) = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v548) )
          goto LABEL_663;
        v137 = (unsigned int)v542;
        v138 = (_DWORD)v542 + 5;
        if ( (_DWORD)v542 + 4 == (_DWORD)v555 )
        {
          if ( (_DWORD)v542 + 4 != v138 )
          {
            v139 = 0;
LABEL_221:
            sub_14EEB00((__int64 *)&v551, v11, v139, *(_QWORD *)v548);
            v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
            if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              goto LABEL_517;
            v284 = 7;
            v551 = 0;
            sub_14ECA90((__int64 *)&v551);
            v285 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 2)];
            if ( (unsigned int)v285 > 5
              || (v284 = dword_42926C0[(unsigned int)v285], (unsigned int)(v284 - 5) > 1) && v284 )
            {
              v286 = *(_QWORD *)&v554[8 * (unsigned int)v542];
              if ( v286 )
              {
                v287 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 3)];
                v288 = (unsigned __int8)v287;
                if ( (unsigned int)v287 > 1 )
                {
                  v288 = 1;
                  if ( (unsigned int)v287 < *(_DWORD *)(v11 + 1768) )
                    v288 = *(unsigned __int8 *)(*(_QWORD *)(v11 + 1760) + (unsigned int)v287);
                }
                sub_14EEAA0((__int64 *)&v551, v11, v286, (int *)&v543);
                v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
                if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                  goto LABEL_517;
                v551 = 0;
                sub_14ECA90((__int64 *)&v551);
                v553[0] = 257;
                v519 = *(_QWORD *)&v554[8 * ((_DWORD)v542 + 1)] != 0;
                v307 = sub_1648A60(64, 1);
                v167 = (_BYTE *)v307;
                if ( v307 )
                  sub_15F8F80(
                    v307,
                    *(_QWORD *)(*(_QWORD *)v548 + 24LL),
                    (_DWORD)v548,
                    (unsigned int)&v551,
                    v519,
                    (_DWORD)v543,
                    v284,
                    v288,
                    0);
LABEL_274:
                v541 = v167;
                goto LABEL_100;
              }
            }
LABEL_663:
            v382 = v11;
            v3 = a1;
            v551 = "Invalid record";
            v553[0] = 259;
            sub_14EE4B0(a1, v382 + 8, (__int64)&v551);
            goto LABEL_21;
          }
        }
        else if ( (_DWORD)v555 != v138 )
        {
          goto LABEL_663;
        }
        LODWORD(v542) = (_DWORD)v542 + 1;
        v139 = sub_14EFEB0((_QWORD *)v11, *(_QWORD *)&v554[8 * v137]);
        goto LABEL_221;
      case 42:
      case 45:
        v539 = 0;
        if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v548) || *(_BYTE *)(*(_QWORD *)v548 + 8LL) != 15 )
          goto LABEL_663;
        if ( v13 == 45 )
        {
          if ( sub_14EFF60(v11, (__int64)&v554, &v539, v532, (__int64 *)&v543) )
            goto LABEL_663;
        }
        else if ( (unsigned __int8)sub_14EA9B0(
                                     v11,
                                     (__int64)&v554,
                                     &v539,
                                     v532,
                                     *(_QWORD *)(*(_QWORD *)v548 + 24LL),
                                     (__int64 *)&v543) )
        {
          goto LABEL_663;
        }
        if ( v539 + 4 != (_DWORD)v555 )
          goto LABEL_663;
        sub_14EEB00((__int64 *)&v551, v11, *v543, *(_QWORD *)v548);
        v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
        if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_517;
        v289 = 7;
        v551 = 0;
        sub_14ECA90((__int64 *)&v551);
        v290 = *(_QWORD *)&v554[8 * v539 + 16];
        if ( (unsigned int)v290 <= 5 )
        {
          v289 = dword_42926C0[(unsigned int)v290];
          if ( (v289 & 0xFFFFFFFB) == 0 || v289 == 6 )
            goto LABEL_663;
        }
        v291 = *(_QWORD *)&v554[8 * v539 + 24];
        if ( (unsigned int)v291 <= 1 )
          goto LABEL_489;
        v517 = 1;
        if ( (unsigned int)v291 < *(_DWORD *)(v11 + 1768) )
        {
          LOBYTE(v291) = *(_BYTE *)(*(_QWORD *)(v11 + 1760) + (unsigned int)v291);
LABEL_489:
          v517 = v291;
        }
        v292 = *(_QWORD *)&v554[8 * v539];
        if ( !v292 )
          goto LABEL_663;
        sub_14EEAA0((__int64 *)&v551, v11, v292, (int *)&v542);
        v22 = (unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL;
        if ( ((unsigned __int64)v551 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_517;
        v551 = 0;
        sub_14ECA90((__int64 *)&v551);
        v305 = *(_QWORD *)&v554[8 * v539 + 8] != 0;
        v306 = sub_1648A60(64, 2);
        v167 = (_BYTE *)v306;
        if ( v306 )
          sub_15F9480(v306, (_DWORD)v543, (_DWORD)v548, v305, (_DWORD)v542, v289, v517, 0);
        goto LABEL_274;
      case 48:
        if ( (unsigned __int64)(unsigned int)v555 - 1 > 1 )
          goto LABEL_663;
        v156 = sub_16432D0(*(_QWORD *)(v11 + 432));
        v157 = sub_14EA380(v11, (__int64)&v554, 0, v532, v156);
        if ( !v157 )
          goto LABEL_663;
        v158 = v555;
        if ( (_DWORD)v555 == 2 )
        {
          v159 = *(_QWORD *)(v11 + 1368);
          v160 = *((unsigned int *)v554 + 2);
          if ( (*(_QWORD *)(v11 + 1376) - v159) >> 3 <= v160 )
            goto LABEL_663;
          v161 = *(_QWORD *)(v159 + 8 * v160);
          if ( !v161 )
            goto LABEL_663;
        }
        else
        {
          v161 = 0;
          v158 = 1;
        }
        v162 = sub_1648A60(56, v158);
        v70 = (_BYTE *)v162;
        if ( v162 )
          sub_15F76D0(v162, v157, v161, v158, 0);
        goto LABEL_131;
      case 49:
        if ( (_DWORD)v555 != 2 )
          goto LABEL_663;
        v149 = sub_16432D0(*(_QWORD *)(v11 + 432));
        v150 = sub_14EA380(v11, (__int64)&v554, 0, v532, v149);
        if ( !v150 )
          goto LABEL_663;
        v151 = *(_QWORD *)(v11 + 1368);
        v152 = *((unsigned int *)v554 + 2);
        if ( (*(_QWORD *)(v11 + 1376) - v151) >> 3 <= v152 )
          goto LABEL_663;
        v153 = *(_QWORD *)(v151 + 8 * v152);
        if ( !v153 )
          goto LABEL_663;
        v154 = sub_1648A60(56, 2);
        v70 = (_BYTE *)v154;
        if ( v154 )
          sub_15F7960(v154, v150, v153, 0);
        goto LABEL_131;
      case 50:
      case 51:
        if ( (unsigned int)v555 <= 1 )
          goto LABEL_663;
        v24 = *(_QWORD *)(v11 + 432);
        LODWORD(v542) = 0;
        v25 = sub_16432D0(v24);
        v26 = (unsigned int)v542;
        LODWORD(v542) = (_DWORD)v542 + 1;
        v497 = sub_14EA380(v11, (__int64)&v554, v26, v532, v25);
        v27 = (unsigned int)v542;
        LODWORD(v542) = (_DWORD)v542 + 1;
        v28 = *(_QWORD *)&v554[8 * v27];
        v551 = (const char *)v553;
        v552 = 0x200000000LL;
        v509 = v28;
        if ( !(_DWORD)v28 )
          goto LABEL_565;
        v29 = 0;
        do
        {
          if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543) )
            goto LABEL_550;
          v30 = (unsigned int)v552;
          if ( (unsigned int)v552 >= HIDWORD(v552) )
          {
            sub_16CD150(&v551, v553, 0, 8);
            v30 = (unsigned int)v552;
          }
          ++v29;
          *(_QWORD *)&v551[8 * v30] = v543;
          LODWORD(v552) = v552 + 1;
        }
        while ( v509 != v29 );
LABEL_565:
        if ( (_DWORD)v542 != (_DWORD)v555 )
        {
LABEL_550:
          v327 = v11;
          v3 = a1;
          v548 = "Invalid record";
          LOWORD(v550) = 259;
          sub_14EE4B0(a1, v327 + 8, (__int64)&v548);
          v86 = v551;
          if ( v551 != (const char *)v553 )
            goto LABEL_773;
          goto LABEL_21;
        }
        v336 = v552;
        v337 = (int)v551;
        LOWORD(v550) = 257;
        v338 = v552 + 1;
        v339 = (unsigned int)(v552 + 1);
        if ( v13 == 51 )
        {
          v439 = sub_1648A60(56, v339);
          v341 = (_BYTE *)v439;
          if ( v439 )
            sub_15F8230(v439, 49, v497, v337, v336, v338, (__int64)&v548, 0);
        }
        else
        {
          v340 = sub_1648A60(56, v339);
          v341 = (_BYTE *)v340;
          if ( v340 )
            sub_15F8230(v340, 50, v497, v337, v336, v338, (__int64)&v548, 0);
        }
        v541 = v341;
        goto LABEL_561;
      case 52:
        if ( (unsigned int)v555 <= 1 )
          goto LABEL_663;
        v73 = sub_16432D0(*(_QWORD *)(v11 + 432));
        v74 = sub_14EA380(v11, (__int64)&v554, 0, v532, v73);
        v75 = (__int64)v554;
        v76 = v74;
        v77 = *((_QWORD *)v554 + 1);
        v551 = (const char *)v553;
        v552 = 0x200000000LL;
        if ( (_DWORD)v77 )
        {
          v78 = (unsigned int)(v77 + 2);
          for ( j = 2; ; j = v83 )
          {
            v82 = *(unsigned int *)(v75 + 8 * j);
            v83 = j + 1;
            v84 = *(_QWORD *)(v11 + 1368);
            if ( (*(_QWORD *)(v11 + 1376) - v84) >> 3 <= v82 )
              break;
            v80 = *(_QWORD *)(v84 + 8 * v82);
            if ( !v80 )
              break;
            v81 = (unsigned int)v552;
            if ( (unsigned int)v552 >= HIDWORD(v552) )
            {
              v501 = *(_QWORD *)(v84 + 8 * v82);
              sub_16CD150(&v551, v553, 0, 8);
              v81 = (unsigned int)v552;
              v80 = v501;
            }
            *(_QWORD *)&v551[8 * v81] = v80;
            LODWORD(v552) = v552 + 1;
            if ( v83 == (_DWORD)v78 )
              goto LABEL_553;
            v75 = (__int64)v554;
          }
LABEL_153:
          v85 = v11;
          v3 = a1;
          v548 = "Invalid record";
          LOWORD(v550) = 259;
          sub_14EE4B0(a1, v85 + 8, (__int64)&v548);
          v86 = v551;
          if ( v551 != (const char *)v553 )
            goto LABEL_773;
          goto LABEL_21;
        }
        v78 = 2;
LABEL_553:
        if ( (_DWORD)v78 + 1 == (_DWORD)v555 )
        {
          v447 = *(_QWORD *)(v11 + 1368);
          v448 = *(unsigned int *)&v554[8 * v78];
          if ( (*(_QWORD *)(v11 + 1376) - v447) >> 3 <= v448 )
            goto LABEL_153;
          v328 = *(_QWORD *)(v447 + 8 * v448);
          if ( !v328 )
            goto LABEL_153;
        }
        else
        {
          if ( (_DWORD)v555 != (_DWORD)v78 )
            goto LABEL_153;
          v328 = 0;
        }
        LOWORD(v550) = 257;
        v329 = sub_1648B60(64);
        v330 = (_BYTE *)v329;
        if ( v329 )
          sub_15F7B50(v329, v76, v328, (unsigned int)v77, &v548, 0);
        v331 = v551;
        v332 = &v551[8 * (unsigned int)v552];
        if ( v332 != v551 )
        {
          do
          {
            v333 = *(_QWORD *)v331;
            v331 += 8;
            sub_15F7DB0(v330, v333);
          }
          while ( v332 != v331 );
        }
        v541 = v330;
LABEL_561:
        v56 = (__int64)&v541;
        sub_14EF3D0(v11 + 648, &v541);
        v334 = v551;
        if ( v551 == (const char *)v553 )
          goto LABEL_101;
        goto LABEL_562;
      case 55:
        if ( !(_DWORD)v555 )
          goto LABEL_663;
        v71 = v554;
        if ( *(_QWORD *)v554 >= (unsigned __int64)((__int64)(*(_QWORD *)(v11 + 1744) - *(_QWORD *)(v11 + 1736)) >> 5) )
          goto LABEL_663;
        v548 = 0;
        v549 = 0;
        v550 = 0;
        LODWORD(v542) = 1;
        if ( (_DWORD)v555 == 1 )
          goto LABEL_142;
        do
        {
          if ( sub_14EFF60(v11, (__int64)&v554, (unsigned int *)&v542, v532, (__int64 *)&v543) )
          {
            v436 = v11;
            v3 = a1;
            v551 = "Invalid record";
            v553[0] = 259;
            sub_14EE4B0(a1, v436 + 8, (__int64)&v551);
            if ( v548 )
              j_j___libc_free_0(v548, v550 - v548);
            goto LABEL_21;
          }
          v72 = v549;
          if ( v549 == v550 )
          {
            sub_1287830((__int64)&v548, v549, &v543);
          }
          else
          {
            if ( v549 )
            {
              *(_QWORD *)v549 = v543;
              v72 = v549;
            }
            v549 = v72 + 8;
          }
        }
        while ( (_DWORD)v542 != (_DWORD)v555 );
        v71 = v554;
LABEL_142:
        sub_14F2920(&v545, (_QWORD *)(*(_QWORD *)(v11 + 1736) + 32LL * *(_QWORD *)v71), (__int64 *)&v548);
        if ( v548 )
          j_j___libc_free_0(v548, v550 - v548);
        continue;
      default:
        v23 = v11;
        v3 = a1;
        v551 = "Invalid value";
        v553[0] = 259;
        sub_14EE4B0(a1, v23 + 8, (__int64)&v551);
        goto LABEL_21;
    }
  }
}
