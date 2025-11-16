// Function: sub_955A70
// Address: 0x955a70
//
__int64 __fastcall sub_955A70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  unsigned int v8; // ebx
  _QWORD *v9; // rbx
  void *v10; // rbx
  size_t v11; // rdx
  size_t v12; // rsi
  size_t v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 result; // rax
  unsigned __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r13
  __m128i *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // r14
  unsigned int **v27; // r14
  __int64 v28; // rax
  __int64 v29; // rcx
  int v30; // r13d
  __int64 v31; // rax
  int v32; // r12d
  int v33; // ebx
  unsigned __int64 v34; // rsi
  __int64 v35; // rax
  _QWORD *v36; // rdi
  __m128i *v37; // rdi
  __int64 v38; // r14
  __int64 v39; // r13
  __m128i *v40; // r14
  __m128i *v41; // r13
  __int64 v42; // rdi
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 *v51; // rdi
  int v52; // ebx
  __int64 v53; // rax
  unsigned __int64 v54; // rsi
  __int64 v55; // rsi
  _QWORD *v56; // rdi
  __int64 v57; // r13
  __int64 v58; // r14
  __m128i *v59; // r13
  __m128i *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r14
  __m128i *v63; // rbx
  __int64 v64; // rax
  __int64 v65; // rbx
  __int64 v66; // rax
  __int64 v67; // r11
  unsigned __int64 v68; // rdx
  _BYTE *v69; // rbx
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  _BYTE *v72; // rbx
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  _BYTE *v75; // rbx
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  __int64 *v78; // rdi
  __int64 v79; // rax
  unsigned __int64 v80; // rsi
  __int64 v81; // rsi
  _QWORD *v82; // rdi
  __m128i *v83; // r13
  __int64 v84; // rcx
  __int64 v85; // rax
  __int64 v86; // rax
  unsigned __int64 *v87; // r13
  __m128i *v88; // rax
  __int64 *v89; // rdi
  __int64 v90; // rax
  unsigned __int64 v91; // rsi
  __int64 v92; // rax
  __int64 *v93; // rdi
  __int64 v94; // rax
  unsigned __int64 v95; // rsi
  __int64 v96; // rax
  __int64 v97; // r13
  __int64 v98; // r8
  __int64 v99; // r14
  __m128i *v100; // r13
  __int64 v101; // rbx
  __m128i *v102; // rax
  int v103; // eax
  int v104; // edx
  int v105; // ebx
  unsigned __int64 v106; // rax
  unsigned int v107; // eax
  unsigned int **v108; // r12
  __int64 v109; // rax
  __m128i *v110; // rax
  __int64 *v111; // rdi
  __int64 v112; // rax
  unsigned __int64 v113; // rsi
  __int64 v114; // rax
  _QWORD *v115; // rdi
  __int64 v116; // rsi
  __m128i *v117; // rax
  __int64 v118; // rax
  unsigned int **v119; // r12
  __int64 v120; // rax
  __m128i *v121; // rax
  __int64 *v122; // rdi
  __int64 v123; // rax
  unsigned __int64 v124; // rsi
  __int64 v125; // r13
  __int64 v126; // r14
  __m128i *v127; // r14
  __int64 v128; // rdi
  __int64 v129; // r13
  __int64 v130; // rax
  __int64 v131; // rdx
  __int64 v132; // r9
  _BYTE *v133; // rbx
  __int64 v134; // rax
  unsigned __int64 v135; // rdx
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // r13
  __int64 v139; // rax
  unsigned __int64 v140; // rdx
  __int64 *v141; // rdi
  int v142; // ebx
  int v143; // r14d
  __int64 v144; // rdx
  __int64 v145; // rax
  __int64 v146; // rdi
  __int64 v147; // rax
  __int64 v148; // rax
  unsigned int **v149; // r14
  unsigned __int64 *v150; // r13
  __int64 v151; // rbx
  __int64 v152; // rcx
  unsigned int v153; // ebx
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rdx
  unsigned __int64 v157; // rax
  char v158; // r8
  __int64 v159; // rax
  __int64 v160; // r13
  unsigned int *v161; // rbx
  unsigned int *v162; // r12
  __int64 v163; // rdx
  __int64 v164; // rsi
  __int64 v165; // rax
  __int64 v166; // rsi
  __int64 v167; // r13
  __int64 v168; // rax
  __int64 v169; // rax
  __int64 v170; // r13
  _QWORD *v171; // rax
  __int64 v172; // rax
  __int64 v173; // rax
  __int64 v174; // rax
  __m128i *v175; // rbx
  __m128i *v176; // rax
  __int64 *v177; // rdi
  __m128i *v178; // r13
  __int64 v179; // rax
  unsigned int **v180; // rdi
  unsigned __int64 v181; // rsi
  __int64 v182; // rax
  __int64 v183; // rbx
  char v184; // al
  __int64 v185; // r13
  _BYTE *v186; // rcx
  __int64 *v187; // rax
  __int64 nn; // r13
  int v189; // eax
  _QWORD *v190; // rcx
  char i1; // al
  const char *v192; // r14
  _BYTE *v193; // r8
  __m128i *v194; // r13
  _QWORD *v195; // rax
  __int64 v196; // rax
  size_t v197; // rdx
  __int64 v198; // rcx
  __int64 v199; // rax
  __int64 v200; // rbx
  __m128i *v201; // rax
  __int64 v202; // rdx
  __int64 v203; // r14
  __int64 v204; // rax
  unsigned __int64 v205; // rdx
  unsigned int v206; // eax
  __int64 v207; // rbx
  __int64 v208; // r14
  __int64 v209; // rax
  __int64 v210; // rax
  unsigned __int64 v211; // rsi
  int v212; // edx
  __int64 v213; // rax
  __int64 v214; // rax
  _QWORD *v215; // rdi
  __int64 v216; // r14
  __m128i *v217; // rax
  __int64 v218; // rax
  __m128i *v219; // rax
  __int64 *v220; // rdi
  __int64 v221; // rax
  unsigned __int64 v222; // rsi
  __int64 v223; // rax
  __int64 v224; // rbx
  __m128i *v225; // rax
  __int64 *v226; // rdi
  __m128i *v227; // rax
  __int64 v228; // rax
  unsigned __int64 v229; // rsi
  __int64 v230; // rax
  __int64 v231; // r14
  __int64 v232; // rbx
  char mm; // al
  unsigned __int64 v234; // r13
  __int64 v235; // rbx
  __m128i *v236; // rax
  __int64 v237; // rbx
  __int64 v238; // rcx
  unsigned __int64 v239; // rax
  __int64 v240; // rax
  int v241; // r8d
  __int64 v242; // rax
  int v243; // r9d
  __int64 v244; // r13
  unsigned int *v245; // r14
  unsigned int *v246; // rbx
  __int64 v247; // rdx
  __int64 v248; // rsi
  __int64 v249; // rax
  __int64 v250; // rax
  __int64 v251; // r14
  __m128i *v252; // r13
  __m128i *v253; // rax
  __int64 v254; // rax
  __m128i *v255; // rax
  __int64 *v256; // rdi
  __int64 v257; // rax
  unsigned __int64 v258; // rsi
  __int64 v259; // rax
  __m128i *v260; // rax
  __int64 v261; // rsi
  __int64 v262; // rax
  __int64 v263; // rbx
  __int64 v264; // rdx
  __int64 v265; // r13
  __int64 v266; // r14
  __int64 v267; // rax
  char v268; // dl
  __int64 v269; // rdi
  __int64 v270; // r14
  __int64 v271; // rdi
  __int64 v272; // r14
  __int64 v273; // rax
  __int64 v274; // rax
  __int64 v275; // rax
  __int64 v276; // rax
  _QWORD *v277; // rdi
  __int64 v278; // rsi
  __int64 v279; // r13
  __m128i *v280; // rbx
  __int64 v281; // rsi
  __int64 v282; // r13
  unsigned int v283; // eax
  unsigned __int64 v284; // r14
  __int64 v285; // rdi
  __int64 v286; // rax
  __m128i *v287; // rax
  __int64 *v288; // rdi
  __int64 kk; // r14
  __m128i *v290; // rbx
  __m128i *v291; // r14
  __int64 v292; // rax
  __m128i *v293; // r8
  unsigned int **v294; // r12
  __int64 v295; // rax
  __int64 v296; // rax
  __int64 v297; // rax
  unsigned int v298; // eax
  __int64 v299; // rdi
  __int64 v300; // r13
  unsigned int v301; // r14d
  __int64 v302; // rax
  __int64 v303; // r13
  __int64 v304; // rax
  __int64 v305; // rax
  __int64 *v306; // rdi
  __int64 v307; // rsi
  __int64 v308; // rax
  unsigned __int64 v309; // rsi
  unsigned int **v310; // r14
  __int64 v311; // r15
  int v312; // ebx
  __int64 v313; // rax
  char v314; // al
  __int16 v315; // cx
  __int64 v316; // rax
  int v317; // r9d
  __int64 v318; // r13
  unsigned int *v319; // r15
  unsigned int *v320; // rbx
  __int64 v321; // rdx
  __int64 v322; // rsi
  __int64 v323; // rdi
  __int64 v324; // r13
  __int64 v325; // rax
  int v326; // ebx
  unsigned int *v327; // r13
  unsigned int *v328; // rbx
  __int64 v329; // rdx
  __int64 v330; // rsi
  unsigned int v331; // eax
  __int64 v332; // r13
  __int64 v333; // rax
  __int64 v334; // rax
  __int64 v335; // rdi
  int v336; // eax
  __int64 v337; // rdi
  int v338; // r13d
  __int64 v339; // rax
  char v340; // al
  int v341; // r15d
  __int64 v342; // rax
  __int64 v343; // r9
  __int64 v344; // rbx
  unsigned int *v345; // r15
  unsigned int *v346; // r13
  __int64 v347; // rdx
  __int64 v348; // rsi
  __int64 v349; // rax
  __int64 v350; // rax
  __int64 v351; // rdi
  _BYTE *v352; // r13
  __int64 (__fastcall *v353)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v354; // r15
  __int64 v355; // rax
  unsigned __int64 v356; // rdx
  __int64 v357; // r13
  __int64 v358; // rax
  __int64 v359; // rbx
  __int64 v360; // rax
  unsigned __int64 v361; // rdx
  __int64 v362; // rax
  __int64 *v363; // rdi
  int v364; // r13d
  int v365; // ebx
  __int64 v366; // rax
  unsigned __int64 v367; // rsi
  __int64 v368; // r13
  __int64 v369; // rax
  int v370; // r14d
  __int64 v371; // rbx
  int v372; // r9d
  __int64 v373; // rsi
  __int64 v374; // r13
  unsigned int *v375; // r12
  unsigned int *v376; // r13
  __int64 v377; // rdx
  _QWORD *v378; // rdi
  _QWORD *v379; // rdx
  int v380; // ecx
  int v381; // eax
  __int64 v382; // rax
  int v383; // esi
  unsigned int *v384; // r13
  unsigned int *v385; // rbx
  __int64 v386; // rdx
  __int64 v387; // rsi
  __int64 v388; // rax
  _QWORD *v389; // rdi
  __int64 v390; // rax
  __int64 v391; // r14
  unsigned int v392; // r15d
  __m128i *i; // rax
  unsigned int v394; // ebx
  __int64 v395; // rdx
  __int64 v396; // rdi
  __int64 v397; // r15
  __int64 v398; // r12
  __int64 v399; // rax
  _BYTE *v400; // rax
  unsigned __int64 v401; // r13
  char v402; // al
  unsigned __int64 v403; // rax
  char v404; // dl
  unsigned __int64 v405; // rax
  __int64 j; // rbx
  __int64 v407; // rdi
  __int64 v408; // rax
  __int64 v409; // rcx
  char v410; // dl
  unsigned __int64 v411; // rax
  _QWORD *v412; // rax
  _QWORD *v413; // rax
  __int64 v414; // rcx
  __int64 v415; // rbx
  _QWORD *v416; // r12
  _BYTE *v417; // rsi
  __int64 v418; // rax
  __int64 ii; // r14
  __int64 v420; // r13
  __int64 jj; // rbx
  __int64 v422; // rax
  _BYTE *v423; // r11
  int v424; // eax
  __int64 v425; // rax
  __int64 v426; // rdi
  __int64 (__fastcall *v427)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v428; // rax
  __int64 v429; // r9
  __m128i *v430; // rdi
  bool v431; // al
  __int64 v432; // rax
  __int64 v433; // rax
  __int64 v434; // rax
  int v435; // eax
  __int64 k; // rax
  __int64 *v437; // rbx
  __int64 *v438; // r13
  __int64 v439; // rdx
  __m128i *v440; // rsi
  unsigned __int64 m; // rax
  __int64 n; // rbx
  __int64 v443; // rax
  __int64 v444; // rcx
  __m128i *v445; // rsi
  __int64 v446; // rsi
  __m128i *v447; // r13
  __int64 v448; // r14
  __int64 v449; // rdx
  __int64 v450; // rsi
  __int64 v451; // rdi
  __int64 (__fastcall *v452)(__int64, __int64, __int64); // rax
  __int64 v453; // rax
  __int64 v454; // rbx
  unsigned int *v455; // rbx
  unsigned int *v456; // r12
  __int64 v457; // rdx
  __int64 v458; // rsi
  __int64 v459; // rax
  __int64 v460; // rbx
  __int64 v461; // rax
  unsigned __int64 v462; // rdx
  __m128i *v463; // r14
  __int64 v464; // rdx
  __int64 v465; // rsi
  __int64 v466; // rdi
  __int64 (__fastcall *v467)(__int64, __int64, __int64); // rax
  __int64 v468; // rax
  __int64 v469; // rbx
  unsigned int *v470; // rbx
  unsigned int *v471; // r12
  __int64 v472; // rdx
  __int64 v473; // rsi
  __int64 v474; // rbx
  __int64 v475; // rax
  unsigned __int64 v476; // rdx
  char v477; // bl
  __m128i *v478; // r14
  __int64 v479; // rax
  unsigned __int64 v480; // rdx
  __int64 v481; // r8
  char v482; // bl
  unsigned __int64 v483; // r14
  char v484; // bl
  unsigned __int64 v485; // rbx
  char v486; // al
  void *v487; // r8
  __int64 v488; // rax
  __int64 v489; // rax
  __int64 v490; // rax
  unsigned __int64 v491; // rdx
  char v492; // al
  unsigned int *v493; // rax
  unsigned int *v494; // rcx
  unsigned int *v495; // r12
  unsigned int *v496; // rbx
  __int64 v497; // rdx
  __int64 v498; // rsi
  unsigned int *v499; // rax
  unsigned int *v500; // rcx
  unsigned int *v501; // r12
  unsigned int *v502; // rbx
  __int64 v503; // rdx
  __int64 v504; // rsi
  __int64 v505; // [rsp-10h] [rbp-2E0h]
  __int64 v506; // [rsp-10h] [rbp-2E0h]
  unsigned __int64 v507; // [rsp-10h] [rbp-2E0h]
  __int64 v508; // [rsp-8h] [rbp-2D8h]
  __int64 v509; // [rsp+8h] [rbp-2C8h]
  __int64 v510; // [rsp+10h] [rbp-2C0h]
  __int64 v511; // [rsp+10h] [rbp-2C0h]
  __int64 v512; // [rsp+18h] [rbp-2B8h]
  __int64 v513; // [rsp+18h] [rbp-2B8h]
  __int64 v514; // [rsp+20h] [rbp-2B0h]
  __int64 v515; // [rsp+20h] [rbp-2B0h]
  __int64 v516; // [rsp+28h] [rbp-2A8h]
  unsigned int **v517; // [rsp+28h] [rbp-2A8h]
  bool v518; // [rsp+28h] [rbp-2A8h]
  __int64 v519; // [rsp+30h] [rbp-2A0h]
  unsigned int v520; // [rsp+30h] [rbp-2A0h]
  __int64 v521; // [rsp+30h] [rbp-2A0h]
  __int64 v522; // [rsp+30h] [rbp-2A0h]
  unsigned __int64 v523; // [rsp+30h] [rbp-2A0h]
  __int64 v524; // [rsp+38h] [rbp-298h]
  __int64 v525; // [rsp+38h] [rbp-298h]
  unsigned int v526; // [rsp+38h] [rbp-298h]
  __int64 v527; // [rsp+38h] [rbp-298h]
  __int64 v528; // [rsp+40h] [rbp-290h]
  __m128i *v529; // [rsp+40h] [rbp-290h]
  __int64 v530; // [rsp+40h] [rbp-290h]
  __int64 v531; // [rsp+40h] [rbp-290h]
  __int64 v532; // [rsp+40h] [rbp-290h]
  unsigned __int16 v533; // [rsp+48h] [rbp-288h]
  unsigned int v534; // [rsp+48h] [rbp-288h]
  __int64 v535; // [rsp+48h] [rbp-288h]
  __int64 v536; // [rsp+48h] [rbp-288h]
  _BYTE *v537; // [rsp+48h] [rbp-288h]
  __int64 v538; // [rsp+48h] [rbp-288h]
  __m128i *v539; // [rsp+50h] [rbp-280h]
  _QWORD *v540; // [rsp+50h] [rbp-280h]
  __m128i *v541; // [rsp+50h] [rbp-280h]
  __int64 v542; // [rsp+50h] [rbp-280h]
  __int64 v543; // [rsp+50h] [rbp-280h]
  __m128i *v544; // [rsp+58h] [rbp-278h]
  __m128i *v545; // [rsp+58h] [rbp-278h]
  unsigned __int8 v546; // [rsp+58h] [rbp-278h]
  int v547; // [rsp+58h] [rbp-278h]
  __m128i *v548; // [rsp+58h] [rbp-278h]
  __int64 v549; // [rsp+58h] [rbp-278h]
  __int64 v550; // [rsp+58h] [rbp-278h]
  __int64 v551; // [rsp+58h] [rbp-278h]
  __int64 v552; // [rsp+58h] [rbp-278h]
  __m128i *v553; // [rsp+60h] [rbp-270h]
  char v554; // [rsp+60h] [rbp-270h]
  __m128i *v555; // [rsp+60h] [rbp-270h]
  unsigned int v556; // [rsp+60h] [rbp-270h]
  unsigned int **v557; // [rsp+60h] [rbp-270h]
  unsigned int v558; // [rsp+60h] [rbp-270h]
  _BYTE *v559; // [rsp+60h] [rbp-270h]
  __int64 v560; // [rsp+60h] [rbp-270h]
  __int64 v561; // [rsp+60h] [rbp-270h]
  __int64 v562; // [rsp+60h] [rbp-270h]
  _BYTE *v563; // [rsp+60h] [rbp-270h]
  __int64 v564; // [rsp+60h] [rbp-270h]
  __int16 v565; // [rsp+68h] [rbp-268h]
  __int64 v566; // [rsp+68h] [rbp-268h]
  __int64 v567; // [rsp+68h] [rbp-268h]
  __int64 v568; // [rsp+68h] [rbp-268h]
  int v569; // [rsp+68h] [rbp-268h]
  __int64 v570; // [rsp+68h] [rbp-268h]
  __m128i *v571; // [rsp+68h] [rbp-268h]
  __int64 v572; // [rsp+68h] [rbp-268h]
  __int64 v573; // [rsp+68h] [rbp-268h]
  __int64 v574; // [rsp+68h] [rbp-268h]
  int srcf; // [rsp+70h] [rbp-260h]
  int src; // [rsp+70h] [rbp-260h]
  void *srcg; // [rsp+70h] [rbp-260h]
  _DWORD *srca; // [rsp+70h] [rbp-260h]
  int srch; // [rsp+70h] [rbp-260h]
  void *srci; // [rsp+70h] [rbp-260h]
  _QWORD *srcb; // [rsp+70h] [rbp-260h]
  unsigned __int64 srcj; // [rsp+70h] [rbp-260h]
  unsigned int srcc; // [rsp+70h] [rbp-260h]
  _BYTE *srck; // [rsp+70h] [rbp-260h]
  _BYTE *srcd; // [rsp+70h] [rbp-260h]
  __int64 srce; // [rsp+70h] [rbp-260h]
  void *srcl; // [rsp+70h] [rbp-260h]
  __int64 *v588; // [rsp+78h] [rbp-258h] BYREF
  __int64 v589; // [rsp+88h] [rbp-248h] BYREF
  __int64 v590[2]; // [rsp+90h] [rbp-240h] BYREF
  unsigned __int64 v591[4]; // [rsp+A0h] [rbp-230h] BYREF
  __int16 v592; // [rsp+C0h] [rbp-210h]
  __m128i *v593; // [rsp+D0h] [rbp-200h] BYREF
  __int64 v594; // [rsp+D8h] [rbp-1F8h]
  __m128i v595; // [rsp+E0h] [rbp-1F0h] BYREF
  __int16 v596; // [rsp+F0h] [rbp-1E0h]
  __int64 v597; // [rsp+100h] [rbp-1D0h] BYREF
  __int64 v598; // [rsp+108h] [rbp-1C8h]
  __int64 v599; // [rsp+110h] [rbp-1C0h]
  int v600; // [rsp+118h] [rbp-1B8h]
  __int64 v601; // [rsp+120h] [rbp-1B0h]
  __int64 v602; // [rsp+128h] [rbp-1A8h]
  __int64 v603; // [rsp+130h] [rbp-1A0h]
  __m128i *v604; // [rsp+140h] [rbp-190h] BYREF
  __int64 v605; // [rsp+148h] [rbp-188h]
  __m128i v606; // [rsp+150h] [rbp-180h] BYREF
  __int16 v607; // [rsp+160h] [rbp-170h]
  __int64 v608; // [rsp+190h] [rbp-140h] BYREF
  __int64 v609; // [rsp+198h] [rbp-138h]
  _QWORD v610[2]; // [rsp+1A0h] [rbp-130h] BYREF
  __int16 v611; // [rsp+1B0h] [rbp-120h]

  v4 = a2;
  v5 = *(_QWORD *)(a3 + 72);
  v588 = (__int64 *)a3;
  v6 = *(_QWORD *)(v5 + 16);
  if ( (*(_BYTE *)(*(_QWORD *)(v5 + 56) + 199LL) & 2) == 0 )
    sub_91B8A0("unexpected: builtin expression that is not an intrinsic!", (_DWORD *)(a3 + 36), 1);
  v7 = sub_91B6C0(*(_QWORD *)(v5 + 56));
  if ( !v7 )
    sub_91B8A0("unexpected: intrinsic cannot be unnamed!", (_DWORD *)v588 + 9, 1);
  v8 = sub_913450(*(_QWORD *)(a2 + 32), (const char *)v7);
  if ( !v8 )
  {
LABEL_4:
    v9 = **(_QWORD ***)(a2 + 32);
    sub_2241BD0(&v597, v9 + 29);
    v601 = v9[33];
    v602 = v9[34];
    v603 = v9[35];
    v10 = (void *)sub_CC5ED0((unsigned int)v601);
    v12 = v11;
    if ( !v11
      || ((v13 = strlen((const char *)v7), v13 <= 4) || *(_DWORD *)v7 != 1836477548 || *(_BYTE *)(v7 + 4) != 46
        ? (v534 = sub_B6ACB0(v10, v12))
        : (v534 = sub_B60C50(v7, v13, v14, v13)),
          !v534) )
    {
      sub_91B8A0("unexpected: unable to lookup intrinsic!", (_DWORD *)v588 + 9, 1);
    }
    v391 = 0;
    v608 = (__int64)v610;
    v609 = 0x1000000000LL;
    if ( !(unsigned __int8)sub_B60C20(v534) )
      v391 = *(_QWORD *)(sub_B6E160(**(_QWORD **)(v4 + 32), v534, 0, 0) + 24);
    v527 = v4 + 48;
    if ( v6 )
    {
      v531 = a1;
      v392 = 0;
      v550 = v4;
      for ( i = sub_92F410(v4, v6); ; i = sub_92F410(v550, v6) )
      {
        v398 = (__int64)i;
        if ( (unsigned int)sub_8D29A0(*(_QWORD *)v6) )
        {
          v607 = 257;
          v399 = sub_BCB2B0(*(_QWORD *)(v550 + 120));
          v400 = (_BYTE *)sub_ACD640(v399, 0, 0);
          v398 = sub_92B530((unsigned int **)v527, 0x21u, v398, v400, (__int64)&v604);
        }
        v394 = v392 + 1;
        if ( v391 )
        {
          if ( *(_DWORD *)(v391 + 12) - 1 <= v392 )
            sub_91B8A0(
              "unexpected: mismatch between number of call arguments and expected number of function parameters!",
              (_DWORD *)v588 + 9,
              1);
          v395 = *(_QWORD *)(v391 + 16);
          v396 = *(_QWORD *)(v398 + 8);
          v397 = *(_QWORD *)(v395 + 8LL * v394);
          if ( v396 != v397 )
          {
            if ( !(unsigned __int8)sub_BCAF30(v396, *(_QWORD *)(v395 + 8LL * v394)) )
              sub_91B8A0("unexpected: cannot losslessly bitcast argument to parameter type!", (_DWORD *)v588 + 9, 1);
            v607 = 257;
            v398 = sub_949E90((unsigned int **)v527, 0x31u, v398, v397, (__int64)&v604, 0, (unsigned int)v593, 0);
          }
        }
        sub_94F890((__int64)&v608, v398);
        v6 = *(_QWORD *)(v6 + 16);
        if ( !v6 )
          break;
        v392 = v394;
      }
      a1 = v531;
      v4 = v550;
    }
    v401 = *v588;
    if ( sub_91B770(*v588) )
    {
      v402 = *(_BYTE *)(v401 + 140);
      if ( v402 == 12 )
      {
        v403 = v401;
        do
        {
          v403 = *(_QWORD *)(v403 + 160);
          v404 = *(_BYTE *)(v403 + 140);
        }
        while ( v404 == 12 );
        if ( v404 != 10 )
LABEL_418:
          sub_91B8A0(
            "builtin functions that map to LLVM intrinsics support only struct aggregate type!",
            (_DWORD *)v588 + 9,
            1);
        v405 = v401;
        do
          v405 = *(_QWORD *)(v405 + 160);
        while ( *(_BYTE *)(v405 + 140) == 12 );
      }
      else
      {
        if ( v402 != 10 )
          goto LABEL_418;
        v405 = v401;
      }
      for ( j = *(_QWORD *)(v405 + 160); j; j = *(_QWORD *)(j + 112) )
      {
        v407 = *(_QWORD *)(j + 120);
        if ( *(_QWORD *)(j + 112) )
        {
          if ( !(unsigned int)sub_8D2DD0(v407) && !(unsigned int)sub_8D2E30(*(_QWORD *)(j + 120)) )
            sub_91B8A0(
              "nvvm builtin function that return struct types should use simple types for return struct fields!",
              (_DWORD *)v588 + 9,
              1);
        }
        else
        {
          if ( !(unsigned int)sub_8D3440(v407) )
            goto LABEL_473;
          for ( k = *(_QWORD *)(j + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
            ;
          if ( *(_QWORD *)(k + 128) )
LABEL_473:
            sub_91B8A0(
              "nvvm builtin function that return struct types need a dummy field to prevent padding!",
              (_DWORD *)v588 + 9,
              1);
        }
      }
    }
    v408 = sub_BCB120(*(_QWORD *)(v4 + 40));
    v410 = *(_BYTE *)(v401 + 140);
    v532 = v408;
    if ( v410 == 12 )
    {
      v411 = v401;
      do
      {
        v411 = *(_QWORD *)(v411 + 160);
        v410 = *(_BYTE *)(v411 + 140);
      }
      while ( v410 == 12 );
    }
    if ( v410 != 1 )
      v532 = sub_91A390(*(_QWORD *)(v4 + 32) + 8LL, v401, 0, v409);
    if ( v391 )
    {
      v412 = *(_QWORD **)(v391 + 16);
      srcd = (_BYTE *)*v412;
      if ( (_DWORD)v609 != *(_DWORD *)(v391 + 12) - 1 )
        sub_91B8A0("LLVM intrinsic cannot be called with the given signature", (_DWORD *)v588 + 9, 1);
      v413 = v412 + 1;
      v414 = v608 + 8LL * (unsigned int)v609;
      if ( v414 != v608 )
      {
        v415 = v608;
        v521 = v4;
        v416 = v413;
        do
        {
          if ( *v416 != *(_QWORD *)(*(_QWORD *)v415 + 8LL) )
            sub_91B8A0("LLVM intrinsic cannot be called with the given signature", (_DWORD *)v588 + 9, 1);
          v415 += 8;
          ++v416;
        }
        while ( v414 != v415 );
        v4 = v521;
      }
    }
    else
    {
      srcd = (_BYTE *)v532;
      if ( (unsigned int)sub_8D29A0(v401) )
        srcd = (_BYTE *)sub_BCB2A0(*(_QWORD *)(v4 + 120));
      if ( srcd[8] == 15 )
      {
        v604 = &v606;
        v605 = 0x400000000LL;
        for ( m = v401; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
          ;
        for ( n = *(_QWORD *)(m + 160); n; n = *(_QWORD *)(n + 112) )
        {
          if ( !*(_QWORD *)(n + 112) )
            break;
          v443 = (unsigned int)sub_8D29A0(*(_QWORD *)(n + 120))
               ? sub_BCB2A0(*(_QWORD *)(v4 + 120))
               : sub_91A390(*(_QWORD *)(v4 + 32) + 8LL, *(_QWORD *)(n + 120), 0, v444);
          sub_94F8E0((__int64)&v604, v443);
        }
        v445 = v604;
        srcd = (_BYTE *)sub_BD0B90(*(_QWORD *)(v4 + 40), v604, (unsigned int)v605, 0);
        if ( v604 != &v606 )
          _libc_free(v604, v445);
      }
      v604 = &v606;
      v605 = 0x600000000LL;
      v437 = (__int64 *)(v608 + 8LL * (unsigned int)v609);
      if ( v437 != (__int64 *)v608 )
      {
        v523 = v401;
        v438 = (__int64 *)v608;
        do
        {
          v439 = *v438++;
          sub_94F8E0((__int64)&v604, *(_QWORD *)(v439 + 8));
        }
        while ( v437 != v438 );
        v401 = v523;
      }
      v440 = v604;
      v391 = sub_BCF480(srcd, v604, (unsigned int)v605, 0);
      if ( v604 != &v606 )
        _libc_free(v604, v440);
    }
    v604 = &v606;
    v605 = 0x600000000LL;
    if ( !(unsigned __int8)sub_B6E220(v534, v391, &v604) )
      sub_91B8A0("LLVM intrinsic cannot be called with the given signature", (_DWORD *)v588 + 9, 1);
    HIDWORD(v591[0]) = 0;
    v596 = 257;
    v417 = srcd;
    v418 = sub_B35180(v527, (_DWORD)srcd, v534, v608, v609, v591[0], (__int64)&v593);
    v590[0] = v4;
    v522 = v418;
    v590[1] = (__int64)&v588;
    if ( srcd[8] == 15 )
    {
      for ( ii = sub_ACA8A0(v532); *(_BYTE *)(v401 + 140) == 12; v401 = *(_QWORD *)(v401 + 160) )
        ;
      v420 = *(_QWORD *)(v401 + 160);
      if ( *((_DWORD *)srcd + 3) )
      {
        for ( jj = 0; *((_DWORD *)srcd + 3) > (unsigned int)jj; ++jj )
        {
          v596 = 257;
          LODWORD(v591[0]) = jj;
          v536 = sub_94D3D0((unsigned int **)v527, v522, (__int64)v591, 1, (__int64)&v593);
          v560 = *(_QWORD *)(*(_QWORD *)(v532 + 16) + 8 * jj);
          v424 = sub_8D29A0(*(_QWORD *)(v420 + 120));
          v425 = sub_94A4F0(v590, v536, v424 != 0, v560);
          v426 = *(_QWORD *)(v4 + 128);
          LODWORD(v589) = jj;
          v592 = 257;
          v423 = (_BYTE *)v425;
          v427 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v426 + 88LL);
          if ( v427 == sub_9482E0 )
          {
            if ( *(_BYTE *)ii > 0x15u || *v423 > 0x15u )
            {
LABEL_455:
              v537 = v423;
              v596 = 257;
              v428 = sub_BD2C40(104, unk_3F148BC);
              if ( v428 )
              {
                v429 = v509;
                v561 = v428;
                LOWORD(v429) = 0;
                sub_B44260(v428, *(_QWORD *)(ii + 8), 65, 2, 0, v429);
                *(_QWORD *)(v561 + 80) = 0x400000000LL;
                *(_QWORD *)(v561 + 72) = v561 + 88;
                sub_B4FD20(v561, ii, v537, &v589, 1, &v593);
                v428 = v561;
              }
              v562 = v428;
              (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136)
                                                                                          + 16LL))(
                *(_QWORD *)(v4 + 136),
                v428,
                v591,
                *(_QWORD *)(v527 + 56),
                *(_QWORD *)(v527 + 64));
              v417 = (_BYTE *)v562;
              sub_94AAF0((unsigned int **)v527, v562);
              ii = v562;
              goto LABEL_452;
            }
            v417 = v423;
            v559 = v423;
            v422 = sub_AAAE30(ii, v423, &v589, 1);
            v423 = v559;
          }
          else
          {
            v563 = v423;
            v417 = (_BYTE *)ii;
            v422 = v427(v426, (_BYTE *)ii, v423, (__int64)&v589, 1);
            v423 = v563;
          }
          if ( !v422 )
            goto LABEL_455;
          ii = v422;
LABEL_452:
          v420 = *(_QWORD *)(v420 + 112);
        }
      }
    }
    else
    {
      ii = v418;
      if ( (_BYTE *)v532 != srcd )
      {
        v435 = sub_8D29A0(v401);
        v417 = (_BYTE *)v522;
        ii = sub_94A4F0(v590, v522, v435 != 0, v532);
      }
    }
    *(_BYTE *)(a1 + 12) &= ~1u;
    v430 = v604;
    *(_QWORD *)a1 = ii;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    if ( v430 != &v606 )
      _libc_free(v430, v417);
    if ( (_QWORD *)v608 != v610 )
      _libc_free(v608, v417);
    sub_2240A30(&v597);
    return a1;
  }
  v15 = *v588;
  if ( sub_91B770(*v588) )
    sub_91B8A0("builtin functions cannot return aggregates!", (_DWORD *)v588 + 9, 1);
  switch ( v8 )
  {
    case 2u:
      v611 = 257;
      HIDWORD(v604) = 0;
      v273 = sub_B33D10((int)a2 + 48, 8259, 0, 0, 0, 0, (unsigned int)v604, (__int64)&v608);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v273;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 6u:
      v274 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
      v275 = sub_ACD640(v274, 0, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v275;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 7u:
      v611 = 257;
      v169 = sub_BD2C40(72, unk_3F148B8);
      v170 = v169;
      if ( v169 )
        sub_B4C8A0(v169, *(_QWORD *)(a2 + 120), 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v170,
        &v608,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      sub_94AAF0((unsigned int **)(a2 + 48), v170);
      v171 = (_QWORD *)sub_945CA0(a2, (__int64)"tmp", 0, 0);
      sub_92FEA0(a2, v171, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 8u:
      v172 = sub_94BE50(a2, 8925);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v172;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 9u:
      v173 = sub_94BE50(a2, 9296);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v173;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xAu:
      v174 = sub_94BE50(a2, 9301);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v174;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xBu:
    case 0xCu:
    case 0xDu:
      v93 = *(__int64 **)(a2 + 32);
      v611 = 257;
      v94 = sub_90A810(v93, dword_3F14778[v8 - 11], 0, 0);
      v95 = 0;
      if ( v94 )
        v95 = *(_QWORD *)(v94 + 24);
      v96 = sub_921880((unsigned int **)(v4 + 48), v95, v94, 0, 0, (__int64)&v608, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v96;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xEu:
      v285 = *(_QWORD *)(a2 + 120);
      v611 = 257;
      v286 = sub_BCB2D0(v285);
      v287 = (__m128i *)sub_ACD640(v286, 4, 0);
      v288 = *(__int64 **)(a2 + 32);
      v604 = v287;
      v221 = sub_90A810(v288, 9052, 0, 0);
      v222 = 0;
      if ( v221 )
        v222 = *(_QWORD *)(v221 + 24);
      goto LABEL_233;
    case 0x12u:
      v261 = *(_QWORD *)(v588[9] + 16);
      v611 = 257;
      HIDWORD(v604) = 0;
      v597 = (__int64)sub_92F410(v4, v261);
      v262 = sub_B33D10((int)v4 + 48, 8257, 0, 0, (unsigned int)&v597, 1, (unsigned int)v604, (__int64)&v608);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v262;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xB5u:
    case 0xB6u:
    case 0xB7u:
    case 0xB8u:
      v83 = sub_92F410(a2, *(_QWORD *)(v588[9] + 16));
      v85 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v588, 0, v84);
      v611 = 257;
      v86 = sub_949E90((unsigned int **)(a2 + 48), 0x31u, (__int64)v83, v85, (__int64)&v608, 0, (unsigned int)v604, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v86;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xB9u:
    case 0xBDu:
    case 0xBEu:
    case 0xBFu:
    case 0xC0u:
      sub_94C360(a1, a2, v8, (unsigned __int64 *)v588);
      return a1;
    case 0xBAu:
    case 0xBBu:
    case 0xBCu:
      v87 = *(unsigned __int64 **)(v588[9] + 16);
      v597 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v87, 0, v16);
      v611 = 257;
      v88 = sub_92F410(a2, (__int64)v87);
      v89 = *(__int64 **)(a2 + 32);
      v604 = v88;
      v90 = sub_90A810(v89, 15, (__int64)&v597, 1u);
      v91 = 0;
      if ( v90 )
        v91 = *(_QWORD *)(v90 + 24);
      v92 = sub_921880((unsigned int **)(v4 + 48), v91, v90, (int)&v604, 1, (__int64)&v608, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v92;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xC1u:
    case 0xC2u:
    case 0xC3u:
    case 0xC4u:
      sub_94A030(a1, a2, v8, (__int64)v588, 0);
      return a1;
    case 0xC5u:
    case 0xC6u:
    case 0xC7u:
    case 0xC8u:
      sub_94A030(a1, a2, v8, (__int64)v588, 1);
      return a1;
    case 0xC9u:
      v216 = sub_BCB2A0(*(_QWORD *)(a2 + 40));
      v217 = sub_92F410(a2, v6);
      v218 = sub_92C9E0(a2, (__int64)v217, 0, v216, 0, 0, (_DWORD *)(v6 + 36));
      sub_B33B40(a2 + 48, v218, 0, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xCAu:
      sub_94C360(a1, a2, 202, (unsigned __int64 *)v588);
      return a1;
    case 0xCBu:
      if ( (unsigned int)sub_91B6E0(v15) <= 0x3F )
      {
        v296 = sub_BCB2D0(*(_QWORD *)(a2 + 120));
        v297 = sub_ACD640(v296, 0, 0);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v297;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      else
      {
        v611 = 257;
        v219 = sub_92F410(a2, v6);
        v220 = *(__int64 **)(a2 + 32);
        v604 = v219;
        v221 = sub_90A810(v220, 8825, 0, 0);
        v222 = 0;
        if ( v221 )
          v222 = *(_QWORD *)(v221 + 24);
LABEL_233:
        v223 = sub_921880((unsigned int **)(v4 + 48), v222, v221, (int)&v604, 1, (__int64)&v608, 0);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v223;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      return a1;
    case 0xCCu:
      v224 = *(_QWORD *)(v588[9] + 16);
      v597 = (__int64)sub_92F410(a2, v224);
      v225 = sub_92F410(a2, *(_QWORD *)(v224 + 16));
      v226 = *(__int64 **)(a2 + 32);
      v598 = (__int64)v225;
      v227 = *(__m128i **)(v597 + 8);
      v611 = 257;
      v604 = v227;
      v605 = (__int64)v227;
      v228 = sub_90A810(v226, 8170, (__int64)&v604, 2u);
      v229 = 0;
      if ( v228 )
        v229 = *(_QWORD *)(v228 + 24);
      v230 = sub_921880((unsigned int **)(v4 + 48), v229, v228, (int)&v597, 2, (__int64)&v608, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v230;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xCDu:
      v166 = *(_QWORD *)(v588[9] + 16);
      v167 = *(_QWORD *)(v166 + 16);
      v611 = 257;
      HIDWORD(v597) = 0;
      v604 = sub_92F410(v4, v166);
      v605 = (__int64)sub_92F410(v4, v167);
      v168 = sub_B33D10((int)v4 + 48, 8258, 0, 0, (unsigned int)&v604, 2, (unsigned int)v597, (__int64)&v608);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v168;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0xCFu:
    case 0xD2u:
      sub_94A6D0(a1, a2, 1, (unsigned __int64 *)v588);
      return a1;
    case 0xD0u:
      sub_94D160(a1, a2, 0x2013u, 1, 1, v588[9]);
      return a1;
    case 0xD1u:
      sub_94D160(a1, a2, 0x2013u, 1, 2, v588[9]);
      return a1;
    case 0xD3u:
      sub_94D160(a1, a2, 0x2014u, 1, 1, v588[9]);
      return a1;
    case 0xD4u:
      sub_94D160(a1, a2, 0x2014u, 1, 2, v588[9]);
      return a1;
    case 0xD5u:
    case 0xD8u:
LABEL_707:
      BUG();
    case 0xD6u:
      sub_94D160(a1, a2, 0x2010u, 11, 1, v588[9]);
      return a1;
    case 0xD7u:
      sub_94D160(a1, a2, 0x2010u, 11, 2, v588[9]);
      return a1;
    case 0xD9u:
      sub_94D160(a1, a2, 0x2011u, 11, 1, v588[9]);
      return a1;
    case 0xDAu:
      sub_94D160(a1, a2, 0x2011u, 11, 2, v588[9]);
      return a1;
    case 0xDBu:
    case 0xDEu:
      sub_94A6D0(a1, a2, 0, (unsigned __int64 *)v588);
      return a1;
    case 0xDCu:
      sub_94D160(a1, a2, 0x2013u, 0, 1, v588[9]);
      return a1;
    case 0xDDu:
      sub_94D160(a1, a2, 0x2013u, 0, 2, v588[9]);
      return a1;
    case 0xDFu:
      sub_94D160(a1, a2, 0x2014u, 0, 1, v588[9]);
      return a1;
    case 0xE0u:
      sub_94D160(a1, a2, 0x2014u, 0, 2, v588[9]);
      return a1;
    case 0xE1u:
      v608 = (__int64)v610;
      v609 = 0x400000000LL;
      sub_9493D0(a2, 0, 0, v588[9], &v604, (__int64)&v608);
      v276 = sub_94AEF0((unsigned int **)(a2 + 48), 0, *(_QWORD *)v608, *(_QWORD *)(v608 + 8), 0, 2, 1u);
      sub_949050((unsigned int **)(a2 + 48), v276, (int)v604, 0, 0);
      v277 = (_QWORD *)v608;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v277 != v610 )
        _libc_free(v277, v506);
      return a1;
    case 0xE2u:
    case 0xE3u:
      v608 = (__int64)v610;
      v119 = (unsigned int **)(a2 + 48);
      v609 = 0x400000000LL;
      sub_9493D0(a2, 0, (v8 != 226) + 1, v588[9], &v593, (__int64)&v608);
      v120 = sub_BCB2F0(*(_QWORD *)(a2 + 40));
      v121 = (__m128i *)sub_BCE760(v120, 0);
      v122 = *(__int64 **)(a2 + 32);
      v597 = (__int64)v121;
      v123 = sub_90A810(v122, 8210, (__int64)&v597, 1u);
      v124 = 0;
      v607 = 257;
      if ( v123 )
        v124 = *(_QWORD *)(v123 + 24);
      v116 = sub_921880(v119, v124, v123, v608, v609, (__int64)&v604, 0);
      sub_949050(v119, v116, (int)v593, 0, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      v115 = (_QWORD *)v608;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v115 != v610 )
        goto LABEL_314;
      return a1;
    case 0xE4u:
    case 0xE7u:
      sub_94A6D0(a1, a2, 8, (unsigned __int64 *)v588);
      return a1;
    case 0xE5u:
      sub_94D160(a1, a2, 0x2013u, 8, 1, v588[9]);
      return a1;
    case 0xE6u:
      sub_94D160(a1, a2, 0x2013u, 8, 2, v588[9]);
      return a1;
    case 0xE8u:
      sub_94D160(a1, a2, 0x2014u, 8, 1, v588[9]);
      return a1;
    case 0xE9u:
      sub_94D160(a1, a2, 0x2014u, 8, 2, v588[9]);
      return a1;
    case 0xEAu:
    case 0xEDu:
      sub_94A6D0(a1, a2, 10, (unsigned __int64 *)v588);
      return a1;
    case 0xEBu:
      sub_94D160(a1, a2, 0x2013u, 10, 1, v588[9]);
      return a1;
    case 0xECu:
      sub_94D160(a1, a2, 0x2013u, 10, 2, v588[9]);
      return a1;
    case 0xEEu:
      sub_94D160(a1, a2, 0x2014u, 10, 1, v588[9]);
      return a1;
    case 0xEFu:
      sub_94D160(a1, a2, 0x2014u, 10, 2, v588[9]);
      return a1;
    case 0xF0u:
    case 0xF3u:
      sub_94A6D0(a1, a2, 7, (unsigned __int64 *)v588);
      return a1;
    case 0xF1u:
      sub_94D160(a1, a2, 0x2013u, 7, 1, v588[9]);
      return a1;
    case 0xF2u:
      sub_94D160(a1, a2, 0x2013u, 7, 2, v588[9]);
      return a1;
    case 0xF4u:
      sub_94D160(a1, a2, 0x2014u, 7, 1, v588[9]);
      return a1;
    case 0xF5u:
      sub_94D160(a1, a2, 0x2014u, 7, 2, v588[9]);
      return a1;
    case 0xF6u:
    case 0xF9u:
      sub_94A6D0(a1, a2, 9, (unsigned __int64 *)v588);
      return a1;
    case 0xF7u:
      sub_94D160(a1, a2, 0x2013u, 9, 1, v588[9]);
      return a1;
    case 0xF8u:
      sub_94D160(a1, a2, 0x2013u, 9, 2, v588[9]);
      return a1;
    case 0xFAu:
      sub_94D160(a1, a2, 0x2014u, 9, 1, v588[9]);
      return a1;
    case 0xFBu:
      sub_94D160(a1, a2, 0x2014u, 9, 2, v588[9]);
      return a1;
    case 0xFCu:
      sub_94D300(a1, a2, 0x2008u, v588[9]);
      return a1;
    case 0xFDu:
      sub_94D160(a1, a2, 0x2013u, 12, 1, v588[9]);
      return a1;
    case 0xFEu:
      sub_94D160(a1, a2, 0x2013u, 12, 2, v588[9]);
      return a1;
    case 0xFFu:
      sub_94D300(a1, a2, 0x2007u, v588[9]);
      return a1;
    case 0x100u:
      sub_94D160(a1, a2, 0x2013u, 13, 1, v588[9]);
      return a1;
    case 0x101u:
      sub_94D160(a1, a2, 0x2013u, 13, 2, v588[9]);
      return a1;
    case 0x102u:
    case 0x105u:
      sub_94A6D0(a1, a2, 3, (unsigned __int64 *)v588);
      return a1;
    case 0x103u:
      sub_94D160(a1, a2, 0x2013u, 3, 1, v588[9]);
      return a1;
    case 0x104u:
      sub_94D160(a1, a2, 0x2013u, 3, 2, v588[9]);
      return a1;
    case 0x106u:
      sub_94D160(a1, a2, 0x2014u, 3, 1, v588[9]);
      return a1;
    case 0x107u:
      sub_94D160(a1, a2, 0x2014u, 3, 2, v588[9]);
      return a1;
    case 0x108u:
    case 0x10Bu:
      sub_94A6D0(a1, a2, 5, (unsigned __int64 *)v588);
      return a1;
    case 0x109u:
      sub_94D160(a1, a2, 0x2013u, 5, 1, v588[9]);
      return a1;
    case 0x10Au:
      sub_94D160(a1, a2, 0x2013u, 5, 2, v588[9]);
      return a1;
    case 0x10Cu:
      sub_94D160(a1, a2, 0x2014u, 5, 1, v588[9]);
      return a1;
    case 0x10Du:
      sub_94D160(a1, a2, 0x2014u, 5, 2, v588[9]);
      return a1;
    case 0x10Eu:
    case 0x111u:
      sub_94A6D0(a1, a2, 6, (unsigned __int64 *)v588);
      return a1;
    case 0x10Fu:
      sub_94D160(a1, a2, 0x2013u, 6, 1, v588[9]);
      return a1;
    case 0x110u:
      sub_94D160(a1, a2, 0x2013u, 6, 2, v588[9]);
      return a1;
    case 0x112u:
      sub_94D160(a1, a2, 0x2014u, 6, 1, v588[9]);
      return a1;
    case 0x113u:
      sub_94D160(a1, a2, 0x2014u, 6, 2, v588[9]);
      return a1;
    case 0x12Eu:
    case 0x12Fu:
    case 0x130u:
    case 0x131u:
    case 0x132u:
    case 0x133u:
    case 0x134u:
    case 0x135u:
    case 0x152u:
    case 0x153u:
    case 0x154u:
    case 0x155u:
    case 0x156u:
    case 0x157u:
    case 0x158u:
    case 0x159u:
    case 0x18Bu:
    case 0x18Cu:
    case 0x18Du:
    case 0x18Eu:
    case 0x18Fu:
    case 0x190u:
    case 0x191u:
    case 0x192u:
      sub_954F10(a1, a2, v8, v588[9]);
      return a1;
    case 0x15Fu:
      sub_94D570(a1, a2, 0, (unsigned __int64 *)v588, 0, 0);
      return a1;
    case 0x160u:
      sub_94D570(a1, a2, 1u, (unsigned __int64 *)v588, 0, 0);
      return a1;
    case 0x161u:
      sub_94D570(a1, a2, 2u, (unsigned __int64 *)v588, 0, 0);
      return a1;
    case 0x162u:
      sub_94D570(a1, a2, 3u, (unsigned __int64 *)v588, 1, 0);
      return a1;
    case 0x163u:
      sub_94D570(a1, a2, 0, (unsigned __int64 *)v588, 0, 1);
      return a1;
    case 0x164u:
      sub_94D570(a1, a2, 1u, (unsigned __int64 *)v588, 0, 1);
      return a1;
    case 0x165u:
      sub_94D570(a1, a2, 2u, (unsigned __int64 *)v588, 0, 1);
      return a1;
    case 0x166u:
      sub_94D570(a1, a2, 3u, (unsigned __int64 *)v588, 1, 1);
      return a1;
    case 0x167u:
      v175 = sub_92F410(a2, v6);
      v176 = sub_92F410(a2, *(_QWORD *)(v6 + 16));
      v177 = *(__int64 **)(a2 + 32);
      v178 = v176;
      v597 = v175->m128i_i64[1];
      v179 = sub_90A810(v177, 91, (__int64)&v597, 1u);
      v180 = (unsigned int **)(a2 + 48);
      v181 = 0;
      v604 = v175;
      v605 = (__int64)v178;
      v611 = 257;
      if ( v179 )
        v181 = *(_QWORD *)(v179 + 24);
      v182 = sub_921880(v180, v181, v179, (int)&v604, 2, (__int64)&v608, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v182;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x168u:
      goto LABEL_260;
    case 0x169u:
    case 0x16Au:
      sub_94ED50(a1, a2, 0x2332u, (unsigned __int64 *)v588, 0);
      return a1;
    case 0x16Bu:
    case 0x16Cu:
      sub_94ED50(a1, a2, 0x2330u, (unsigned __int64 *)v588, 1);
      return a1;
    case 0x16Du:
      if ( (unsigned int)sub_91B6E0(v15) <= 0x3F )
      {
LABEL_260:
        v260 = sub_92F410(a2, v6);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v260;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      else
      {
        v604 = sub_92F410(a2, v6);
        v255 = sub_92F410(a2, *(_QWORD *)(v6 + 16));
        v256 = *(__int64 **)(a2 + 32);
        v605 = (__int64)v255;
        v611 = 257;
        v257 = sub_90A810(v256, 9005, 0, 0);
        v258 = 0;
        if ( v257 )
          v258 = *(_QWORD *)(v257 + 24);
        v259 = sub_921880((unsigned int **)(v4 + 48), v258, v257, (int)&v604, 2, (__int64)&v608, 0);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v259;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      return a1;
    case 0x16Eu:
    case 0x1A1u:
    case 0x1A2u:
    case 0x1A3u:
    case 0x1A4u:
    case 0x1A5u:
    case 0x1A6u:
    case 0x1A7u:
    case 0x1A8u:
    case 0x1A9u:
    case 0x1AAu:
    case 0x1ABu:
    case 0x1ACu:
    case 0x1ADu:
    case 0x1AEu:
    case 0x1AFu:
    case 0x1B0u:
    case 0x1B1u:
    case 0x1B2u:
    case 0x1B3u:
    case 0x1B4u:
    case 0x1B5u:
    case 0x1B6u:
    case 0x1B7u:
    case 0x1B8u:
    case 0x1B9u:
    case 0x1BAu:
    case 0x1BBu:
    case 0x1BCu:
    case 0x1BDu:
    case 0x1BEu:
    case 0x1BFu:
    case 0x1C0u:
    case 0x1C1u:
    case 0x1C2u:
    case 0x1C3u:
    case 0x1C4u:
    case 0x1C5u:
    case 0x1C6u:
    case 0x1C7u:
    case 0x1C8u:
    case 0x1C9u:
    case 0x1CAu:
    case 0x1CEu:
    case 0x1CFu:
    case 0x1D0u:
    case 0x1D1u:
    case 0x1D5u:
    case 0x1D6u:
    case 0x1D7u:
    case 0x1D8u:
    case 0x1D9u:
      sub_9502D0(a1, a2, v8, v6, v588);
      return a1;
    case 0x16Fu:
      sub_94C5F0(a1, a2, v6, 4u);
      return a1;
    case 0x170u:
      sub_94C5F0(a1, a2, v6, 8u);
      return a1;
    case 0x171u:
      sub_94C5F0(a1, a2, v6, 0x10u);
      return a1;
    case 0x172u:
    case 0x175u:
      v149 = (unsigned int **)(a2 + 48);
      v150 = *(unsigned __int64 **)(v588[9] + 16);
      v151 = v150[2];
      srci = *(void **)(v151 + 16);
      sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v588, 0, *(_QWORD *)(a2 + 32));
      sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v150, 0, v152);
      v571 = sub_92F410(a2, (__int64)srci);
      v153 = (unsigned int)sub_92F410(a2, v151);
      v556 = (unsigned int)sub_92F410(a2, (__int64)v150);
      v154 = sub_AA4E30(*(_QWORD *)(a2 + 96));
      v155 = sub_9208B0(v154, v571->m128i_i64[1]);
      v609 = v156;
      v608 = (unsigned __int64)(v155 + 7) >> 3;
      v157 = sub_CA1930(&v608);
      v158 = -1;
      if ( v157 )
      {
        _BitScanReverse64(&v157, v157);
        v158 = 63 - (v157 ^ 0x3F);
      }
      v546 = v158;
      v611 = 257;
      v159 = sub_BD2C40(80, unk_3F148C4);
      v160 = v159;
      if ( v159 )
        sub_B4D5A0(v159, v556, v153, (_DWORD)v571, v546, 2, 2, 1, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v160,
        &v608,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      v161 = *(unsigned int **)(a2 + 48);
      v162 = &v161[4 * *(unsigned int *)(a2 + 56)];
      while ( v162 != v161 )
      {
        v163 = *((_QWORD *)v161 + 1);
        v164 = *v161;
        v161 += 4;
        sub_B99FD0(v160, v164, v163);
      }
      v611 = 257;
      LODWORD(v604) = 0;
      v165 = sub_94D3D0(v149, v160, (__int64)&v604, 1, (__int64)&v608);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v165;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x173u:
      sub_94D160(a1, a2, 0x1FF7u, 14, 1, v588[9]);
      return a1;
    case 0x174u:
      sub_94D160(a1, a2, 0x1FF7u, 14, 2, v588[9]);
      return a1;
    case 0x176u:
      sub_94D160(a1, a2, 0x1FF8u, 14, 1, v588[9]);
      return a1;
    case 0x177u:
      sub_94D160(a1, a2, 0x1FF8u, 14, 2, v588[9]);
      return a1;
    case 0x178u:
      sub_94D160(a1, a2, 0x1FF6u, 14, 0, v588[9]);
      return a1;
    case 0x179u:
      v609 = 0x400000000LL;
      v294 = (unsigned int **)(a2 + 48);
      v608 = (__int64)v610;
      sub_9493D0(a2, 14, 0, v588[9], (__m128i **)&v597, (__int64)&v608);
      v295 = sub_94AD80(
               (unsigned int **)(a2 + 48),
               *(_QWORD *)v608,
               *(_QWORD *)(v608 + 8),
               *(_QWORD *)(v608 + 16),
               0,
               2,
               2,
               1u);
      v607 = 257;
      LODWORD(v593) = 0;
      v116 = sub_94D3D0((unsigned int **)(a2 + 48), v295, (__int64)&v593, 1, (__int64)&v604);
      sub_949050(v294, v116, v597, 0, 0);
      v115 = (_QWORD *)v608;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v115 != v610 )
        goto LABEL_314;
      return a1;
    case 0x17Au:
    case 0x17Bu:
      v608 = (__int64)v610;
      v108 = (unsigned int **)(a2 + 48);
      v609 = 0x400000000LL;
      sub_9493D0(a2, 14, (v8 != 378) + 1, v588[9], &v593, (__int64)&v608);
      v109 = sub_BCB2F0(*(_QWORD *)(a2 + 40));
      v110 = (__m128i *)sub_BCE760(v109, 0);
      v111 = *(__int64 **)(a2 + 32);
      v597 = (__int64)v110;
      v112 = sub_90A810(v111, 8181, (__int64)&v597, 1u);
      v113 = 0;
      v607 = 257;
      if ( v112 )
        v113 = *(_QWORD *)(v112 + 24);
      v114 = sub_921880(v108, v113, v112, v608, v609, (__int64)&v604, 0);
      sub_949050(v108, v114, (int)v593, 0, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      v115 = (_QWORD *)v608;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      v116 = v505;
      if ( v115 != v610 )
LABEL_314:
        _libc_free(v115, v116);
      return a1;
    case 0x194u:
      for ( kk = v6; (*(_BYTE *)(kk + 27) & 2) != 0; kk = *(_QWORD *)(kk + 72) )
      {
        if ( !(unsigned int)sub_8D2E30(*(_QWORD *)kk) )
          break;
        if ( *(_BYTE *)(kk + 56) != 5 )
          break;
        if ( !(unsigned int)sub_8D2E30(**(_QWORD **)(kk + 72)) )
          break;
      }
      v290 = sub_92F410(a2, kk);
      v291 = sub_92F410(a2, *(_QWORD *)(v6 + 16));
      v292 = *(_QWORD *)(v6 + 16);
      v293 = *(__m128i **)(v292 + 16);
      if ( v293 )
        v293 = sub_92F410(a2, *(_QWORD *)(v292 + 16));
      sub_B37E40(a2 + 48, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 352LL), v290, v291, v293);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v290;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x195u:
      sub_94C070(a1, a2, v6, 0x2452u, 0x2453u, 0x2454u);
      return a1;
    case 0x196u:
      sub_94C070(a1, a2, v6, 0x244Du, 0x244Eu, 0x244Fu);
      return a1;
    case 0x197u:
      sub_94C070(a1, a2, v6, 0x2487u, 0x2488u, 0x2489u);
      return a1;
    case 0x198u:
      sub_94C070(a1, a2, v6, 0x2457u, 0x2458u, 0x2459u);
      return a1;
    case 0x199u:
    case 0x291u:
    case 0x292u:
    case 0x293u:
    case 0x294u:
    case 0x295u:
    case 0x296u:
    case 0x298u:
      sub_948130(a1, a2, v8, (__int64)v588);
      return a1;
    case 0x19Au:
    case 0x297u:
    case 0x299u:
      srca = (_DWORD *)v588 + 9;
      v97 = *(_QWORD *)(v588[9] + 16);
      v98 = *(_QWORD *)(v97 + 16);
      v99 = *(_QWORD *)(v98 + 16);
      v608 = 1;
      if ( v8 == 410 )
      {
        v574 = v98;
        v431 = sub_91CB00(*(_QWORD *)(v99 + 16), &v608);
        v98 = v574;
        if ( !v431 )
          sub_91B8A0("align value for memset was not constant", srca, 1);
      }
      v568 = v98;
      v100 = sub_92F410(a2, v97);
      v101 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
      v102 = sub_92F410(a2, v568);
      v103 = sub_92C9E0(a2, (__int64)v102, 0, v101, 0, 0, srca);
      v104 = 0;
      v105 = v103;
      if ( v608 )
      {
        _BitScanReverse64(&v106, v608);
        LOBYTE(v104) = 63 - (v106 ^ 0x3F);
        BYTE1(v104) = 1;
      }
      v569 = v104;
      v107 = (unsigned int)sub_92F410(a2, v99);
      sub_B34240(a2 + 48, (_DWORD)v100, v105, v107, v569, 0, 0, 0, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v100;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x19Bu:
      sub_9483E0((__int64)v588, (unsigned __int64 *)v590, v591);
      v281 = *(_QWORD *)(*(_QWORD *)(v588[9] + 16) + 16LL);
      v282 = *(_QWORD *)(*(_QWORD *)(v281 + 16) + 16LL);
      v529 = sub_92F410(v4, v281);
      v283 = (unsigned int)sub_92F410(v4, v282);
      v284 = v591[0];
      v520 = v283;
      if ( !v591[0] )
      {
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = 0;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
        return a1;
      }
      v323 = *(_QWORD *)(v4 + 120);
      v517 = (unsigned int **)(v4 + 48);
      v607 = 257;
      v324 = sub_BCB2D0(v323);
      v325 = sub_AA4E30(*(_QWORD *)(v4 + 96));
      v326 = (unsigned __int8)sub_AE5020(v325, v324);
      v611 = 257;
      v515 = sub_BD2C40(80, unk_3F10A14);
      if ( v515 )
        sub_B4D190(v515, v324, v520, (unsigned int)&v608, 0, v326, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __m128i **, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
        *(_QWORD *)(v4 + 136),
        v515,
        &v604,
        *(_QWORD *)(v4 + 104),
        *(_QWORD *)(v4 + 112));
      v327 = *(unsigned int **)(v4 + 48);
      v328 = &v327[4 * *(unsigned int *)(v4 + 56)];
      while ( v328 != v327 )
      {
        v329 = *((_QWORD *)v327 + 1);
        v330 = *v327;
        v327 += 4;
        sub_B99FD0(v515, v330, v329);
      }
      v331 = sub_39FAC40(v284);
      v332 = LODWORD(v590[0]);
      v526 = v331;
      v333 = sub_BCB2D0(*(_QWORD *)(v4 + 120));
      v334 = sub_ACD640(v333, v332, 0);
      v335 = *(_QWORD *)(v4 + 120);
      v610[0] = v334;
      v608 = (__int64)v610;
      v609 = 0xA00000001LL;
      v513 = a1;
      v549 = sub_BCB2B0(v335);
      srcc = 0;
      while ( 1 )
      {
        v607 = 257;
        v336 = sub_94B060(v517, v549, (__int64)v529, srcc, (__int64)&v604);
        v337 = *(_QWORD *)(v4 + 96);
        LOWORD(v601) = 257;
        v338 = v336;
        v339 = sub_AA4E30(v337);
        v340 = sub_AE5020(v339, v549);
        v341 = v533;
        v607 = 257;
        LOBYTE(v341) = v340;
        v533 = v341;
        v342 = sub_BD2C40(80, unk_3F10A14);
        v344 = v342;
        if ( v342 )
        {
          sub_B4D190(v342, v549, v338, (unsigned int)&v604, 0, v341, 0, 0);
          v343 = v508;
        }
        (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v4 + 136) + 16LL))(
          *(_QWORD *)(v4 + 136),
          v344,
          &v597,
          *(_QWORD *)(v4 + 104),
          *(_QWORD *)(v4 + 112),
          v343);
        v345 = *(unsigned int **)(v4 + 48);
        v346 = &v345[4 * *(unsigned int *)(v4 + 56)];
        while ( v346 != v345 )
        {
          v347 = *((_QWORD *)v345 + 1);
          v348 = *v345;
          v345 += 4;
          sub_B99FD0(v344, v348, v347);
        }
        LOWORD(v601) = 257;
        v349 = sub_BCB2B0(*(_QWORD *)(v4 + 120));
        v350 = sub_ACD640(v349, 0, 0);
        v351 = *(_QWORD *)(v4 + 128);
        v352 = (_BYTE *)v350;
        v353 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v351 + 56LL);
        if ( v353 == sub_928890 )
        {
          if ( *(_BYTE *)v344 > 0x15u || *v352 > 0x15u )
          {
LABEL_365:
            v607 = 257;
            v354 = sub_BD2C40(72, unk_3F10FD0);
            if ( v354 )
            {
              v379 = *(_QWORD **)(v344 + 8);
              v380 = *((unsigned __int8 *)v379 + 8);
              if ( (unsigned int)(v380 - 17) > 1 )
              {
                v383 = sub_BCB2A0(*v379);
              }
              else
              {
                v381 = *((_DWORD *)v379 + 8);
                BYTE4(v593) = (_BYTE)v380 == 18;
                LODWORD(v593) = v381;
                v382 = sub_BCB2A0(*v379);
                v383 = sub_BCE1B0(v382, v593);
              }
              sub_B523C0(v354, v383, 53, 33, v344, (_DWORD)v352, (__int64)&v604, 0, 0, 0);
            }
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
              *(_QWORD *)(v4 + 136),
              v354,
              &v597,
              *(_QWORD *)(v4 + 104),
              *(_QWORD *)(v4 + 112));
            v384 = *(unsigned int **)(v4 + 48);
            v385 = &v384[4 * *(unsigned int *)(v4 + 56)];
            while ( v385 != v384 )
            {
              v386 = *((_QWORD *)v384 + 1);
              v387 = *v384;
              v384 += 4;
              sub_B99FD0(v354, v387, v386);
            }
            goto LABEL_350;
          }
          v354 = sub_AAB310(33, v344, v352);
        }
        else
        {
          v354 = v353(v351, 33u, (_BYTE *)v344, v352);
        }
        if ( !v354 )
          goto LABEL_365;
LABEL_350:
        v355 = (unsigned int)v609;
        v356 = (unsigned int)v609 + 1LL;
        if ( v356 > HIDWORD(v609) )
        {
          sub_C8D5F0(&v608, v610, v356, 8);
          v355 = (unsigned int)v609;
        }
        ++srcc;
        *(_QWORD *)(v608 + 8 * v355) = v354;
        LODWORD(v609) = v609 + 1;
        if ( v526 <= srcc )
        {
          v357 = LODWORD(v591[0]);
          a1 = v513;
          v358 = sub_BCB2D0(*(_QWORD *)(v4 + 120));
          v359 = sub_ACD640(v358, v357, 0);
          v360 = (unsigned int)v609;
          v361 = (unsigned int)v609 + 1LL;
          if ( v361 > HIDWORD(v609) )
          {
            sub_C8D5F0(&v608, v610, v361, 8);
            v360 = (unsigned int)v609;
          }
          *(_QWORD *)(v608 + 8 * v360) = v359;
          LODWORD(v609) = v609 + 1;
          v362 = (unsigned int)v609;
          if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
          {
            sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
            v362 = (unsigned int)v609;
          }
          *(_QWORD *)(v608 + 8 * v362) = v515;
          v363 = *(__int64 **)(v4 + 32);
          v607 = 257;
          v600 = 9233;
          v364 = v608;
          v365 = v609 + 1;
          v597 = 0x240C0000240BLL;
          v598 = 0x240E0000240DLL;
          v599 = 0x24100000240FLL;
          LODWORD(v609) = v609 + 1;
          v366 = sub_90A810(v363, *((unsigned int *)&v597 + v526 - 1), 0, 0);
          v367 = 0;
          if ( v366 )
            v367 = *(_QWORD *)(v366 + 24);
          v368 = sub_921880(v517, v367, v366, v364, v365, (__int64)&v604, 0);
          v369 = sub_AA4E30(*(_QWORD *)(v4 + 96));
          v370 = (unsigned __int8)sub_AE5020(v369, *(_QWORD *)(v368 + 8));
          v607 = 257;
          v371 = sub_BD2C40(80, unk_3F10A10);
          if ( v371 )
            sub_B4D3C0(v371, v368, v520, 0, v370, v372, 0, 0);
          v373 = v371;
          (*(void (__fastcall **)(_QWORD, __int64, __m128i **, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
            *(_QWORD *)(v4 + 136),
            v371,
            &v604,
            *(_QWORD *)(v4 + 104),
            *(_QWORD *)(v4 + 112));
          v374 = 4LL * *(unsigned int *)(v4 + 56);
          v375 = *(unsigned int **)(v4 + 48);
          v376 = &v375[v374];
          while ( v376 != v375 )
          {
            v377 = *((_QWORD *)v375 + 1);
            v373 = *v375;
            v375 += 4;
            sub_B99FD0(v371, v373, v377);
          }
          *(_BYTE *)(v513 + 12) &= ~1u;
          v378 = (_QWORD *)v608;
          *(_QWORD *)v513 = 0;
          *(_DWORD *)(v513 + 8) = 0;
          *(_DWORD *)(v513 + 16) = 0;
          if ( v378 != v610 )
            _libc_free(v378, v373);
          return a1;
        }
      }
    case 0x19Cu:
      sub_9483E0((__int64)v588, v591, (unsigned __int64 *)&v593);
      v278 = *(_QWORD *)(*(_QWORD *)(v588[9] + 16) + 16LL);
      v279 = *(_QWORD *)(*(_QWORD *)(v278 + 16) + 16LL);
      v548 = sub_92F410(v4, v278);
      v280 = sub_92F410(v4, v279);
      if ( v593 )
      {
        v298 = sub_39FAC40(v593);
        v299 = *(_QWORD *)(v4 + 120);
        v300 = LODWORD(v591[0]);
        v606.m128i_i32[2] = 9257;
        v558 = v298;
        v301 = v298;
        v604 = (__m128i *)0x242400002423LL;
        v605 = 0x242600002425LL;
        v606.m128i_i64[0] = 0x242800002427LL;
        v302 = sub_BCB2D0(v299);
        v303 = sub_ACD640(v302, v300, 0);
        srcj = (unsigned int)v593;
        v304 = sub_BCB2D0(*(_QWORD *)(v4 + 120));
        v305 = sub_ACD640(v304, srcj, 0);
        v306 = *(__int64 **)(v4 + 32);
        v599 = v305;
        v307 = *((unsigned int *)&v604 + v301 - 1);
        v611 = 257;
        v597 = (__int64)v280;
        v598 = v303;
        v308 = sub_90A810(v306, v307, 0, 0);
        v309 = 0;
        if ( v308 )
          v309 = *(_QWORD *)(v308 + 24);
        v310 = (unsigned int **)(v4 + 48);
        v535 = sub_921880((unsigned int **)(v4 + 48), v309, v308, (int)&v597, 3, (__int64)&v608, 0);
        v525 = a1;
        v530 = sub_BCB2B0(*(_QWORD *)(v4 + 120));
        LODWORD(v597) = 0;
        do
        {
          v311 = v535;
          if ( v558 != 1 )
          {
            v611 = 257;
            v311 = sub_94D3D0(v310, v535, (__int64)&v597, 1, (__int64)&v608);
          }
          v611 = 257;
          v312 = sub_94B060(v310, v530, (__int64)v548, v597, (__int64)&v608);
          v313 = sub_AA4E30(*(_QWORD *)(v4 + 96));
          v314 = sub_AE5020(v313, *(_QWORD *)(v311 + 8));
          HIBYTE(v315) = HIBYTE(v565);
          v611 = 257;
          LOBYTE(v315) = v314;
          v565 = v315;
          v316 = sub_BD2C40(80, unk_3F10A10);
          v318 = v316;
          if ( v316 )
            sub_B4D3C0(v316, v311, v312, 0, (unsigned __int8)v565, v317, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
            *(_QWORD *)(v4 + 136),
            v318,
            &v608,
            *(_QWORD *)(v4 + 104),
            *(_QWORD *)(v4 + 112));
          v319 = *(unsigned int **)(v4 + 48);
          v320 = &v319[4 * *(unsigned int *)(v4 + 56)];
          while ( v320 != v319 )
          {
            v321 = *((_QWORD *)v319 + 1);
            v322 = *v319;
            v319 += 4;
            sub_B99FD0(v318, v322, v321);
          }
          LODWORD(v597) = v597 + 1;
        }
        while ( v558 > (unsigned int)v597 );
        a1 = v525;
        *(_BYTE *)(v525 + 12) &= ~1u;
        *(_QWORD *)v525 = 0;
        *(_DWORD *)(v525 + 8) = 0;
        *(_DWORD *)(v525 + 16) = 0;
      }
      else
      {
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = 0;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
      }
      return a1;
    case 0x19Du:
      sub_94F250(a1, a2, 0x24F5u, v588[9]);
      return a1;
    case 0x19Eu:
      sub_94F250(a1, a2, 0x24EDu, v588[9]);
      return a1;
    case 0x19Fu:
      sub_94F250(a1, a2, 0x24E9u, v588[9]);
      return a1;
    case 0x1A0u:
      sub_94F250(a1, a2, 0x24F1u, v588[9]);
      return a1;
    case 0x1CBu:
      sub_94F430(a1, a2, 0x1CBu, 0x2017u, 0, v588[9]);
      return a1;
    case 0x1CCu:
      sub_94F430(a1, a2, 0x1CCu, 0x2017u, 1, v588[9]);
      return a1;
    case 0x1CDu:
      sub_94F430(a1, a2, 0x1CDu, 0x2017u, 2, v588[9]);
      return a1;
    case 0x1D2u:
      sub_94F430(a1, a2, 0x1D2u, 0x2018u, 0, v588[9]);
      return a1;
    case 0x1D3u:
      sub_94F430(a1, a2, 0x1D3u, 0x2018u, 1, v588[9]);
      return a1;
    case 0x1D4u:
      sub_94F430(a1, a2, 0x1D4u, 0x2018u, 2, v588[9]);
      return a1;
    case 0x27Fu:
    case 0x280u:
      v117 = sub_92F410(a2, *(_QWORD *)(v588[9] + 16));
      v118 = sub_927810(a2, (__int64)v117, v8 == 639);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v118;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x281u:
      v231 = *(_QWORD *)(v588[9] + 16);
      v232 = **(_QWORD **)(v231 + 16);
      for ( mm = *(_BYTE *)(v232 + 140); mm == 12; mm = *(_BYTE *)(v232 + 140) )
        v232 = *(_QWORD *)(v232 + 160);
      if ( mm != 6 )
        sub_91B8A0("expected va_arg builtin second argument tobe of pointer type", (_DWORD *)v588 + 9, 1);
      v234 = *(_QWORD *)(v232 + 160);
      v235 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, v234, 0, v16);
      v236 = sub_92F410(a2, v231);
      v237 = sub_927EF0(a2, (__int64)v236, v235);
      v611 = 259;
      v608 = (__int64)"varg_temp";
      v572 = sub_921D70(a2, v234, (__int64)&v608, v238);
      v557 = (unsigned int **)(a2 + 48);
      if ( *(char *)(v234 + 142) >= 0 && *(_BYTE *)(v234 + 140) == 12 )
        v239 = (unsigned int)sub_8D4AB0(v234);
      else
        v239 = *(unsigned int *)(v234 + 136);
      if ( v239 )
      {
        _BitScanReverse64(&v239, v239);
        v241 = (unsigned __int8)(63 - (v239 ^ 0x3F));
      }
      else
      {
        v240 = sub_AA4E30(*(_QWORD *)(a2 + 96));
        v241 = (unsigned __int8)sub_AE5020(v240, *(_QWORD *)(v237 + 8));
      }
      v547 = v241;
      v611 = 257;
      v242 = sub_BD2C40(80, unk_3F10A10);
      v244 = v242;
      if ( v242 )
        sub_B4D3C0(v242, v237, v572, 0, v547, v243, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v244,
        &v608,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      v245 = *(unsigned int **)(a2 + 48);
      v246 = &v245[4 * *(unsigned int *)(a2 + 56)];
      while ( v246 != v245 )
      {
        v247 = *((_QWORD *)v245 + 1);
        v248 = *v245;
        v245 += 4;
        sub_B99FD0(v244, v248, v247);
      }
      v249 = *(_QWORD *)(v4 + 32);
      v611 = 257;
      v250 = sub_949E90(v557, 0x31u, v572, *(_QWORD *)(v249 + 696), (__int64)&v608, 0, (unsigned int)v604, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v250;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x282u:
      v251 = *(_QWORD *)(v588[9] + 16);
      v252 = sub_92F410(a2, *(_QWORD *)(v251 + 16));
      v253 = sub_92F410(a2, v251);
      v254 = sub_927A80(a2, (__int64)v253, (__int64)v252);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v254;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    case 0x283u:
      sub_94CF30(a1, a2, v588[9], 0x2062u);
      return a1;
    case 0x284u:
      sub_94CF30(a1, a2, v588[9], 0x21C4u);
      return a1;
    case 0x285u:
      sub_94CF30(a1, a2, v588[9], 0x24BFu);
      return a1;
    case 0x286u:
      sub_94CF30(a1, a2, v588[9], 0x296Fu);
      return a1;
    case 0x287u:
      v183 = *(_QWORD *)(v588[9] + 16);
      v184 = *(_BYTE *)(v183 + 24);
      v185 = v183;
      if ( v184 != 1 )
        goto LABEL_319;
      do
      {
        if ( (*(_BYTE *)(v185 + 56) & 0xEF) != 5 )
          goto LABEL_185;
        v185 = *(_QWORD *)(v185 + 72);
        v184 = *(_BYTE *)(v185 + 24);
      }
      while ( v184 == 1 );
LABEL_319:
      if ( v184 != 2 )
LABEL_185:
        sub_91B8A0("unexpected operand in tex/surf handler", (_DWORD *)v588 + 9, 1);
      v186 = *(_BYTE **)(v185 + 56);
      if ( v186[173] != 2 )
        sub_91B8A0("expected first operand to be constant string", (_DWORD *)v588 + 9, 1);
      v187 = *(__int64 **)(v183 + 16);
      for ( nn = *v187; *(_BYTE *)(nn + 140) == 12; nn = *(_QWORD *)(nn + 160) )
        ;
      srcb = v186;
      v189 = sub_8D2E30(nn);
      v190 = srcb;
      if ( v189 )
      {
        v390 = sub_8D46C0(nn);
        v190 = srcb;
        nn = v390;
        for ( i1 = *(_BYTE *)(v390 + 140); i1 == 12; i1 = *(_BYTE *)(nn + 140) )
          nn = *(_QWORD *)(nn + 160);
      }
      else
      {
        i1 = *(_BYTE *)(nn + 140);
      }
      v192 = "void";
      if ( i1 != 1 )
      {
        if ( i1 == 2 )
        {
          switch ( *(_BYTE *)(nn + 160) )
          {
            case 0:
              v192 = "char_as_schar";
              if ( !dword_4F06B98 )
                v192 = "char_as_uchar";
              break;
            case 1:
              v192 = "schar";
              break;
            case 2:
              v192 = "uchar";
              break;
            case 3:
              v192 = "short";
              break;
            case 4:
              v192 = "ushort";
              break;
            case 5:
              v192 = "int";
              break;
            case 6:
              v192 = "uint";
              break;
            case 7:
              v192 = "long";
              break;
            case 8:
              v192 = "ulong";
              break;
            case 9:
              v192 = "longlong";
              break;
            case 0xA:
              v192 = "ulonglong";
              break;
            default:
              v192 = byte_3F871B3;
              break;
          }
        }
        else
        {
          v192 = "float";
          if ( i1 != 3 )
          {
            v192 = *(const char **)(nn + 8);
            if ( !v192 )
              v192 = byte_3F871B3;
          }
        }
      }
      v193 = (_BYTE *)v190[23];
      v194 = (__m128i *)(v190[22] - 1LL);
      v608 = (__int64)v610;
      if ( &v193[(_QWORD)v194] && !v193 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v604 = v194;
      if ( (unsigned __int64)v194 > 0xF )
      {
        srck = v193;
        v388 = sub_22409D0(&v608, &v604, 0);
        v193 = srck;
        v608 = v388;
        v389 = (_QWORD *)v388;
        v610[0] = v604;
LABEL_375:
        memcpy(v389, v193, (size_t)v194);
        v194 = v604;
        v195 = (_QWORD *)v608;
        goto LABEL_202;
      }
      if ( v194 == (__m128i *)1 )
      {
        LOBYTE(v610[0]) = *v193;
        v195 = v610;
        goto LABEL_202;
      }
      if ( v194 )
      {
        v389 = v610;
        goto LABEL_375;
      }
      v195 = v610;
LABEL_202:
      v609 = (__int64)v194;
      v194->m128i_i8[(_QWORD)v195] = 0;
      if ( v609 == 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_682;
      v196 = sub_2241490(&v608, "_", 1, v190);
      v604 = &v606;
      if ( *(_QWORD *)v196 == v196 + 16 )
      {
        v606 = _mm_loadu_si128((const __m128i *)(v196 + 16));
      }
      else
      {
        v604 = *(__m128i **)v196;
        v606.m128i_i64[0] = *(_QWORD *)(v196 + 16);
      }
      v605 = *(_QWORD *)(v196 + 8);
      *(_QWORD *)v196 = v196 + 16;
      *(_QWORD *)(v196 + 8) = 0;
      *(_BYTE *)(v196 + 16) = 0;
      v197 = strlen(v192);
      if ( v197 > 0x3FFFFFFFFFFFFFFFLL - v605 )
LABEL_682:
        sub_4262D8((__int64)"basic_string::append");
      v199 = sub_2241490(&v604, v192, v197, v198);
      v593 = &v595;
      if ( *(_QWORD *)v199 == v199 + 16 )
      {
        v595 = _mm_loadu_si128((const __m128i *)(v199 + 16));
      }
      else
      {
        v593 = *(__m128i **)v199;
        v595.m128i_i64[0] = *(_QWORD *)(v199 + 16);
      }
      v594 = *(_QWORD *)(v199 + 8);
      *(_QWORD *)v199 = v199 + 16;
      *(_QWORD *)(v199 + 8) = 0;
      *(_BYTE *)(v199 + 16) = 0;
      if ( v604 != &v606 )
        j_j___libc_free_0(v604, v606.m128i_i64[0] + 1);
      if ( (_QWORD *)v608 != v610 )
        j_j___libc_free_0(v608, v610[0] + 1LL);
      v604 = &v606;
      v605 = 0x800000000LL;
      v608 = (__int64)v610;
      v609 = 0x800000000LL;
      if ( *(_QWORD *)(v183 + 16) )
      {
        v200 = *(_QWORD *)(v183 + 16);
        do
        {
          v201 = sub_92F410(a2, v200);
          v202 = (unsigned int)v605;
          if ( (unsigned __int64)(unsigned int)v605 + 1 > HIDWORD(v605) )
          {
            v541 = v201;
            sub_C8D5F0(&v604, &v606, (unsigned int)v605 + 1LL, 8);
            v202 = (unsigned int)v605;
            v201 = v541;
          }
          v604->m128i_i64[v202] = (__int64)v201;
          LODWORD(v605) = v605 + 1;
          v203 = v201->m128i_i64[1];
          v204 = (unsigned int)v609;
          v205 = (unsigned int)v609 + 1LL;
          if ( v205 > HIDWORD(v609) )
          {
            sub_C8D5F0(&v608, v610, v205, 8);
            v204 = (unsigned int)v609;
          }
          *(_QWORD *)(v608 + 8 * v204) = v203;
          v206 = v609 + 1;
          LODWORD(v609) = v609 + 1;
          v200 = *(_QWORD *)(v200 + 16);
        }
        while ( v200 );
      }
      else
      {
        v206 = 0;
      }
      v207 = v608;
      v208 = v206;
      v209 = sub_BCB120(*(_QWORD *)(a2 + 40));
      v210 = sub_BCF480(v209, v207, v208, 0);
      v211 = sub_BA8CA0(**(_QWORD **)(a2 + 32), v593, v594, v210);
      LOWORD(v601) = 257;
      sub_921880((unsigned int **)(v4 + 48), v211, v212, (int)v604, v605, (__int64)&v597, 0);
      v213 = sub_BCB2D0(*(_QWORD *)(v4 + 40));
      v214 = sub_AD6530(v213);
      *(_BYTE *)(a1 + 12) &= ~1u;
      v215 = (_QWORD *)v608;
      *(_QWORD *)a1 = v214;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v215 != v610 )
        _libc_free(v215, v211);
      if ( v604 != &v606 )
        _libc_free(v604, v211);
      if ( v593 != &v595 )
        j_j___libc_free_0(v593, v595.m128i_i64[0] + 1);
      return a1;
    case 0x288u:
    case 0x289u:
    case 0x28Au:
    case 0x28Bu:
    case 0x28Cu:
    case 0x28Du:
    case 0x28Eu:
    case 0x28Fu:
    case 0x290u:
    case 0x29Au:
    case 0x29Bu:
    case 0x29Cu:
    case 0x29Du:
    case 0x29Eu:
    case 0x29Fu:
    case 0x2A0u:
    case 0x2A1u:
    case 0x2A2u:
    case 0x2A3u:
    case 0x2A4u:
    case 0x2A5u:
      v18 = (unsigned __int64 *)v588;
      v604 = &v606;
      v608 = (__int64)v610;
      v605 = 0x800000000LL;
      v609 = 0x800000000LL;
      v19 = v588[9];
      v20 = *(_QWORD *)(v19 + 16);
      if ( v20 )
      {
        do
        {
          v21 = sub_92F410(a2, v20);
          v22 = (unsigned int)v605;
          v23 = (unsigned int)v605 + 1LL;
          if ( v23 > HIDWORD(v605) )
          {
            sub_C8D5F0(&v604, &v606, v23, 8);
            v22 = (unsigned int)v605;
          }
          v604->m128i_i64[v22] = (__int64)v21;
          v24 = (unsigned int)v609;
          LODWORD(v605) = v605 + 1;
          v25 = (unsigned int)v609 + 1LL;
          v26 = v21->m128i_i64[1];
          if ( v25 > HIDWORD(v609) )
          {
            sub_C8D5F0(&v608, v610, v25, 8);
            v24 = (unsigned int)v609;
          }
          *(_QWORD *)(v608 + 8 * v24) = v26;
          LODWORD(v609) = v609 + 1;
          v20 = *(_QWORD *)(v20 + 16);
        }
        while ( v20 );
        v19 = v18[9];
      }
      v27 = (unsigned int **)(a2 + 48);
      v28 = sub_917010(*(_QWORD *)(a2 + 32), *(_QWORD *)(v19 + 56), 0);
      v30 = v28;
      if ( !*(_BYTE *)v28 && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v28 + 24) + 16LL) + 8LL) == 7 )
      {
        LOWORD(v601) = 257;
        v34 = *(_QWORD *)(v28 + 24);
        sub_921880((unsigned int **)(v4 + 48), v34, v28, (int)v604, v605, (__int64)&v597, 0);
        v434 = sub_BCB2D0(*(_QWORD *)(v4 + 40));
        v35 = sub_AD6530(v434);
      }
      else
      {
        v31 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v18, 0, v29);
        v32 = (int)v604;
        v33 = v605;
        LOWORD(v601) = 257;
        v34 = sub_BCF480(v31, v608, (unsigned int)v609, 0);
        v35 = sub_921880(v27, v34, v30, v32, v33, (__int64)&v597, 0);
      }
      *(_BYTE *)(a1 + 12) &= ~1u;
      v36 = (_QWORD *)v608;
      *(_QWORD *)a1 = v35;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v36 != v610 )
        _libc_free(v36, v34);
      v37 = v604;
      if ( v604 != &v606 )
        goto LABEL_31;
      return a1;
    case 0x2A6u:
    case 0x2A7u:
    case 0x2B0u:
    case 0x2B1u:
    case 0x2BAu:
    case 0x2BBu:
      sub_94DCB0(a1, a2, v8, (__int64)v588, 1, 1);
      return a1;
    case 0x2A8u:
    case 0x2B2u:
    case 0x2BCu:
      sub_94DCB0(a1, a2, v8, (__int64)v588, 0, 1);
      return a1;
    case 0x2A9u:
    case 0x2B3u:
    case 0x2BDu:
    case 0x2C8u:
    case 0x2CDu:
    case 0x2D2u:
    case 0x2E0u:
    case 0x2E6u:
      sub_94DCB0(a1, a2, v8, (__int64)v588, 0, 0);
      return a1;
    case 0x2AAu:
    case 0x2ABu:
    case 0x2B4u:
    case 0x2B5u:
    case 0x2BEu:
    case 0x2BFu:
    case 0x2D3u:
    case 0x2D4u:
    case 0x2D5u:
    case 0x2E1u:
    case 0x2E7u:
      sub_94CAB0(a1, a2, v8, (__int64)v588);
      return a1;
    case 0x2ACu:
    case 0x2ADu:
    case 0x2AEu:
    case 0x2AFu:
    case 0x2B6u:
    case 0x2B7u:
    case 0x2B8u:
    case 0x2B9u:
    case 0x2C0u:
    case 0x2C1u:
    case 0x2C2u:
    case 0x2C3u:
    case 0x2D6u:
    case 0x2D7u:
    case 0x2D8u:
    case 0x2D9u:
    case 0x2DAu:
    case 0x2DBu:
    case 0x2E2u:
    case 0x2E3u:
    case 0x2E8u:
      sub_94E0D0(a1, a2, v8, (__int64)v588);
      return a1;
    case 0x2C4u:
    case 0x2C5u:
    case 0x2C6u:
    case 0x2C7u:
    case 0x2C9u:
    case 0x2CAu:
    case 0x2CBu:
    case 0x2CCu:
    case 0x2CEu:
    case 0x2CFu:
    case 0x2D0u:
    case 0x2D1u:
    case 0x2DCu:
    case 0x2DDu:
    case 0x2DEu:
    case 0x2DFu:
    case 0x2E4u:
    case 0x2E5u:
      sub_94DCB0(a1, a2, v8, (__int64)v588, 1, 0);
      return a1;
    case 0x2E9u:
    case 0x2EAu:
    case 0x2EBu:
    case 0x2ECu:
    case 0x2EDu:
    case 0x2EEu:
      v57 = *(_QWORD *)(v588[9] + 16);
      v567 = *(_QWORD *)(v57 + 16);
      v58 = *(_QWORD *)(*(_QWORD *)(v567 + 16) + 16LL);
      srcg = *(void **)(v567 + 16);
      v554 = sub_9480A0(
               *(_QWORD *)(v58 + 16),
               3u,
               "unexpected 'rowcol' operand",
               "'rowcol' operand can be 0, 1, 2, or 3 only",
               (_DWORD *)v588 + 9);
      v59 = sub_92F410(a2, v57);
      v539 = sub_92F410(a2, v567);
      v544 = sub_92F410(a2, (__int64)srcg);
      v60 = sub_92F410(a2, v58);
      v61 = *(_QWORD *)(a2 + 32);
      v62 = (__int64)v60;
      v608 = (__int64)v610;
      v609 = 0x2000000000LL;
      sub_953BA0(v61, v8, v554, (unsigned __int64 *)&v593, &v589, v590, v591);
      v63 = v593;
      v64 = sub_BCB2E0(*(_QWORD *)(a2 + 40));
      v65 = sub_ACD640(v64, v63, 0);
      v66 = (unsigned int)v609;
      v67 = (__int64)v539;
      v68 = (unsigned int)v609 + 1LL;
      if ( v68 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v68, 8);
        v66 = (unsigned int)v609;
        v67 = (__int64)v539;
      }
      *(_QWORD *)(v608 + 8 * v66) = v65;
      LODWORD(v609) = v609 + 1;
      v69 = sub_94B510(a2, v589, v67);
      v70 = (unsigned int)v609;
      v71 = (unsigned int)v609 + 1LL;
      if ( v71 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v71, 8);
        v70 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v70) = v69;
      LODWORD(v609) = v609 + 1;
      v72 = sub_94B510(a2, v590[0], (__int64)v544);
      v73 = (unsigned int)v609;
      v74 = (unsigned int)v609 + 1LL;
      if ( v74 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v74, 8);
        v73 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v73) = v72;
      LODWORD(v609) = v609 + 1;
      v75 = sub_94B510(a2, v591[0], v62);
      v76 = (unsigned int)v609;
      v77 = (unsigned int)v609 + 1LL;
      if ( v77 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v77, 8);
        v76 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v76) = v75;
      v78 = *(__int64 **)(a2 + 32);
      LODWORD(v609) = v609 + 1;
      v597 = v591[0];
      v598 = v589;
      v599 = v590[0];
      v79 = sub_90A810(v78, 9062, (__int64)&v597, 3u);
      v80 = 0;
      v607 = 257;
      if ( v79 )
        v80 = *(_QWORD *)(v79 + 24);
      v81 = sub_921880((unsigned int **)(v4 + 48), v80, v79, v608, v609, (__int64)&v604, 0);
      sub_94B940(v4, v81, (__int64)v59);
      *(_BYTE *)(a1 + 12) &= ~1u;
      v82 = (_QWORD *)v608;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v82 != v610 )
        _libc_free(v82, v81);
      return a1;
    case 0x2EFu:
    case 0x2F0u:
      v125 = *(_QWORD *)(*(_QWORD *)(v588[9] + 16) + 16LL);
      v570 = *(_QWORD *)(v588[9] + 16);
      v126 = *(_QWORD *)(v125 + 16);
      srch = sub_9480A0(
               *(_QWORD *)(v126 + 16),
               1u,
               "unexpected 'rowcol' operand",
               "'rowcol' operand can be 0 or 1 only",
               (_DWORD *)v588 + 9);
      v555 = sub_92F410(a2, v570);
      v545 = sub_92F410(a2, v125);
      v127 = sub_92F410(a2, v126);
      sub_954350(*(_QWORD *)(a2 + 32), v8, srch, (__int64 *)&v593, v591);
      v128 = *(_QWORD *)(a2 + 40);
      v608 = (__int64)v610;
      v609 = 0x1000000000LL;
      v129 = sub_BCB2E0(v128);
      v130 = sub_AD64C0(v129, v593, 0);
      v131 = (unsigned int)v609;
      v132 = (__int64)v545;
      if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
      {
        v542 = v130;
        sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
        v131 = (unsigned int)v609;
        v130 = v542;
        v132 = (__int64)v545;
      }
      *(_QWORD *)(v608 + 8 * v131) = v130;
      LODWORD(v609) = v609 + 1;
      v133 = sub_94B510(a2, v591[0], v132);
      v134 = (unsigned int)v609;
      v135 = (unsigned int)v609 + 1LL;
      if ( v135 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v135, 8);
        v134 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v134) = v133;
      LODWORD(v609) = v609 + 1;
      v136 = (unsigned int)v609;
      if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
        v136 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v136) = v555;
      LODWORD(v609) = v609 + 1;
      v137 = (unsigned int)v609;
      if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
        v137 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v137) = v127;
      LODWORD(v609) = v609 + 1;
      v138 = sub_AD64C0(v129, 0, 0);
      v139 = (unsigned int)v609;
      v140 = (unsigned int)v609 + 1LL;
      if ( v140 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v140, 8);
        v139 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v139) = v138;
      v141 = *(__int64 **)(a2 + 32);
      v142 = v609 + 1;
      v143 = v608;
      LODWORD(v609) = v609 + 1;
      v597 = v591[0];
      v144 = v555->m128i_i64[1];
      v607 = 257;
      v598 = v144;
      v145 = sub_90A810(v141, 9145, (__int64)&v597, 2u);
      v55 = 0;
      if ( v145 )
        v55 = *(_QWORD *)(v145 + 24);
      sub_921880((unsigned int **)(v4 + 48), v55, v145, v143, v142, (__int64)&v604, 0);
      *(_BYTE *)(a1 + 12) &= ~1u;
      v56 = (_QWORD *)v608;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v56 != v610 )
        goto LABEL_44;
      return a1;
    case 0x2F1u:
    case 0x2F2u:
    case 0x2F3u:
    case 0x2F4u:
    case 0x2F5u:
    case 0x2F6u:
    case 0x2F7u:
    case 0x2F8u:
    case 0x2F9u:
    case 0x2FAu:
    case 0x2FBu:
    case 0x2FCu:
      v38 = *(_QWORD *)(v588[9] + 16);
      v39 = *(_QWORD *)(v38 + 16);
      v566 = *(_QWORD *)(v39 + 16);
      srcf = sub_9480A0(
               *(_QWORD *)(v566 + 16),
               1u,
               "unexpected 'rowcol' operand",
               "'rowcol' operand can be 0 or 1 only",
               (_DWORD *)v588 + 9);
      v40 = sub_92F410(a2, v38);
      v41 = sub_92F410(a2, v39);
      v553 = sub_92F410(a2, v566);
      sub_9547E0(*(_QWORD *)(a2 + 32), v8, srcf, (__int64 *)&v593, v591);
      v42 = *(_QWORD *)(a2 + 40);
      v608 = (__int64)v610;
      v609 = 0x1000000000LL;
      v43 = sub_BCB2E0(v42);
      v44 = sub_AD64C0(v43, v593, 0);
      v45 = (unsigned int)v609;
      if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
      {
        v551 = v44;
        sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
        v45 = (unsigned int)v609;
        v44 = v551;
      }
      *(_QWORD *)(v608 + 8 * v45) = v44;
      LODWORD(v609) = v609 + 1;
      v46 = (unsigned int)v609;
      if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
        v46 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v46) = v41;
      LODWORD(v609) = v609 + 1;
      v47 = (unsigned int)v609;
      if ( (unsigned __int64)(unsigned int)v609 + 1 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, (unsigned int)v609 + 1LL, 8);
        v47 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v47) = v553;
      LODWORD(v609) = v609 + 1;
      v48 = sub_AD64C0(v43, 0, 0);
      v49 = (unsigned int)v609;
      v50 = (unsigned int)v609 + 1LL;
      if ( v50 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v50, 8);
        v49 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v49) = v48;
      v51 = *(__int64 **)(a2 + 32);
      v52 = v609 + 1;
      src = v608;
      LODWORD(v609) = v609 + 1;
      v597 = v591[0];
      v598 = v41->m128i_i64[1];
      v607 = 257;
      v53 = sub_90A810(v51, 9067, (__int64)&v597, 2u);
      v54 = 0;
      if ( v53 )
        v54 = *(_QWORD *)(v53 + 24);
      v55 = sub_921880((unsigned int **)(v4 + 48), v54, v53, src, v52, (__int64)&v604, 0);
      sub_94B940(v4, v55, (__int64)v40);
      *(_BYTE *)(a1 + 12) &= ~1u;
      v56 = (_QWORD *)v608;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      if ( v56 != v610 )
LABEL_44:
        _libc_free(v56, v55);
      return a1;
    case 0x2FDu:
      v263 = *(_QWORD *)(v6 + 16);
      v512 = *(_QWORD *)(*(_QWORD *)(v263 + 16) + 16LL);
      v528 = *(_QWORD *)(v512 + 16);
      v524 = *(_QWORD *)(v528 + 16);
      v519 = *(_QWORD *)(v524 + 16);
      v264 = *(_QWORD *)(*(_QWORD *)(v519 + 16) + 16LL);
      v516 = *(_QWORD *)(v519 + 16);
      v265 = *(_QWORD *)(v264 + 16);
      v573 = v264;
      v540 = *(_QWORD **)(v265 + 16);
      v514 = v540[2];
      v266 = sub_620FD0(*(_QWORD *)(v264 + 56), &v608);
      if ( (_DWORD)v608 )
        sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v573 + 36), 1);
      v267 = *v540;
      v268 = *(_BYTE *)(*v540 + 140LL);
      if ( v266 )
      {
        while ( v268 == 12 )
        {
          v267 = *(_QWORD *)(v267 + 160);
          v268 = *(_BYTE *)(v267 + 140);
        }
        v269 = *(_QWORD *)(v263 + 56);
        if ( v268 == 2 && *(_BYTE *)(v267 + 160) == 10 )
        {
          v432 = sub_620FD0(v269, &v608);
          if ( !(_DWORD)v608 )
          {
            switch ( v432 )
            {
              case 8LL:
                v446 = 10774;
                goto LABEL_509;
              case 16LL:
                v446 = 10690;
                goto LABEL_509;
              case 24LL:
                v446 = 10734;
                goto LABEL_509;
              case 32LL:
                v446 = 10742;
                goto LABEL_509;
              case 40LL:
                v446 = 10746;
                goto LABEL_509;
              case 48LL:
                v446 = 10750;
                goto LABEL_509;
              case 56LL:
                v446 = 10754;
                goto LABEL_509;
              case 64LL:
                v446 = 10758;
                goto LABEL_509;
              case 72LL:
                v446 = 10762;
                goto LABEL_509;
              case 80LL:
                v446 = 10766;
                goto LABEL_509;
              case 88LL:
                v446 = 10770;
                goto LABEL_509;
              case 96LL:
                v446 = 10778;
                goto LABEL_509;
              case 104LL:
                v446 = 10654;
                goto LABEL_509;
              case 112LL:
                v446 = 10658;
                goto LABEL_509;
              case 120LL:
                v446 = 10662;
                goto LABEL_509;
              case 128LL:
                v446 = 10666;
                goto LABEL_509;
              case 136LL:
                v446 = 10670;
                goto LABEL_509;
              case 144LL:
                v446 = 10674;
                goto LABEL_509;
              case 152LL:
                v446 = 10678;
                goto LABEL_509;
              case 160LL:
                v446 = 10682;
                goto LABEL_509;
              case 168LL:
                v446 = 10686;
                goto LABEL_509;
              case 176LL:
                v446 = 10694;
                goto LABEL_509;
              case 184LL:
                v446 = 10698;
                goto LABEL_509;
              case 192LL:
                v446 = 10702;
                goto LABEL_509;
              case 200LL:
                v446 = 10706;
                goto LABEL_509;
              case 208LL:
                v446 = 10710;
                goto LABEL_509;
              case 216LL:
                v446 = 10714;
                goto LABEL_509;
              case 224LL:
                v446 = 10718;
                goto LABEL_509;
              case 232LL:
                v446 = 10722;
                goto LABEL_509;
              case 240LL:
                v446 = 10726;
                goto LABEL_509;
              case 248LL:
                v446 = 10730;
                goto LABEL_509;
              case 256LL:
                v446 = 10738;
                goto LABEL_509;
              default:
                goto LABEL_707;
            }
          }
          sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v263 + 36), 1);
        }
        v270 = sub_620FD0(v269, &v608);
        if ( !(_DWORD)v608 )
        {
          switch ( v270 )
          {
            case 8LL:
              v446 = 10775;
              goto LABEL_509;
            case 16LL:
              v446 = 10691;
              goto LABEL_509;
            case 24LL:
              v446 = 10735;
              goto LABEL_509;
            case 32LL:
              v446 = 10743;
              goto LABEL_509;
            case 40LL:
              v446 = 10747;
              goto LABEL_509;
            case 48LL:
              v446 = 10751;
              goto LABEL_509;
            case 56LL:
              v446 = 10755;
              goto LABEL_509;
            case 64LL:
              v446 = 10759;
              goto LABEL_509;
            case 72LL:
              v446 = 10763;
              goto LABEL_509;
            case 80LL:
              v446 = 10767;
              goto LABEL_509;
            case 88LL:
              v446 = 10771;
              goto LABEL_509;
            case 96LL:
              v446 = 10779;
              goto LABEL_509;
            case 104LL:
              v446 = 10655;
              goto LABEL_509;
            case 112LL:
              v446 = 10659;
              goto LABEL_509;
            case 120LL:
              v446 = 10663;
              goto LABEL_509;
            case 128LL:
              v446 = 10667;
              goto LABEL_509;
            case 136LL:
              v446 = 10671;
              goto LABEL_509;
            case 144LL:
              v446 = 10675;
              goto LABEL_509;
            case 152LL:
              v446 = 10679;
              goto LABEL_509;
            case 160LL:
              v446 = 10683;
              goto LABEL_509;
            case 168LL:
              v446 = 10687;
              goto LABEL_509;
            case 176LL:
              v446 = 10695;
              goto LABEL_509;
            case 184LL:
              v446 = 10699;
              goto LABEL_509;
            case 192LL:
              v446 = 10703;
              goto LABEL_509;
            case 200LL:
              v446 = 10707;
              goto LABEL_509;
            case 208LL:
              v446 = 10711;
              goto LABEL_509;
            case 216LL:
              v446 = 10715;
              goto LABEL_509;
            case 224LL:
              v446 = 10719;
              goto LABEL_509;
            case 232LL:
              v446 = 10723;
              goto LABEL_509;
            case 240LL:
              v446 = 10727;
              goto LABEL_509;
            case 248LL:
              v446 = 10731;
              goto LABEL_509;
            case 256LL:
              v446 = 10739;
              goto LABEL_509;
            default:
              goto LABEL_707;
          }
        }
        sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v263 + 36), 1);
      }
      while ( v268 == 12 )
      {
        v267 = *(_QWORD *)(v267 + 160);
        v268 = *(_BYTE *)(v267 + 140);
      }
      v271 = *(_QWORD *)(v263 + 56);
      if ( v268 == 2 && *(_BYTE *)(v267 + 160) == 10 )
      {
        v433 = sub_620FD0(v271, &v608);
        if ( !(_DWORD)v608 )
        {
          switch ( v433 )
          {
            case 8LL:
              v446 = 10776;
              goto LABEL_509;
            case 16LL:
              v446 = 10692;
              goto LABEL_509;
            case 24LL:
              v446 = 10736;
              goto LABEL_509;
            case 32LL:
              v446 = 10744;
              goto LABEL_509;
            case 40LL:
              v446 = 10748;
              goto LABEL_509;
            case 48LL:
              v446 = 10752;
              goto LABEL_509;
            case 56LL:
              v446 = 10756;
              goto LABEL_509;
            case 64LL:
              v446 = 10760;
              goto LABEL_509;
            case 72LL:
              v446 = 10764;
              goto LABEL_509;
            case 80LL:
              v446 = 10768;
              goto LABEL_509;
            case 88LL:
              v446 = 10772;
              goto LABEL_509;
            case 96LL:
              v446 = 10780;
              goto LABEL_509;
            case 104LL:
              v446 = 10656;
              goto LABEL_509;
            case 112LL:
              v446 = 10660;
              goto LABEL_509;
            case 120LL:
              v446 = 10664;
              goto LABEL_509;
            case 128LL:
              v446 = 10668;
              goto LABEL_509;
            case 136LL:
              v446 = 10672;
              goto LABEL_509;
            case 144LL:
              v446 = 10676;
              goto LABEL_509;
            case 152LL:
              v446 = 10680;
              goto LABEL_509;
            case 160LL:
              v446 = 10684;
              goto LABEL_509;
            case 168LL:
              v446 = 10688;
              goto LABEL_509;
            case 176LL:
              v446 = 10696;
              goto LABEL_509;
            case 184LL:
              v446 = 10700;
              goto LABEL_509;
            case 192LL:
              v446 = 10704;
              goto LABEL_509;
            case 200LL:
              v446 = 10708;
              goto LABEL_509;
            case 208LL:
              v446 = 10712;
              goto LABEL_509;
            case 216LL:
              v446 = 10716;
              goto LABEL_509;
            case 224LL:
              v446 = 10720;
              goto LABEL_509;
            case 232LL:
              v446 = 10724;
              goto LABEL_509;
            case 240LL:
              v446 = 10728;
              goto LABEL_509;
            case 248LL:
              v446 = 10732;
              goto LABEL_509;
            case 256LL:
              v446 = 10740;
              goto LABEL_509;
            default:
              goto LABEL_707;
          }
        }
        sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v263 + 36), 1);
      }
      v272 = sub_620FD0(v271, &v608);
      if ( (_DWORD)v608 )
        sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v263 + 36), 1);
      return result;
    case 0x301u:
    case 0x302u:
      v146 = *(_QWORD *)(a2 + 40);
      v611 = 257;
      v147 = sub_BCB2B0(v146);
      v148 = sub_921B80(a2, v147, (__int64)&v608, 0x104u, v6);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v148;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    default:
      goto LABEL_4;
  }
  switch ( v272 )
  {
    case 8LL:
      v446 = 10777;
      goto LABEL_509;
    case 16LL:
      v446 = 10693;
      goto LABEL_509;
    case 24LL:
      v446 = 10737;
      goto LABEL_509;
    case 32LL:
      v446 = 10745;
      goto LABEL_509;
    case 40LL:
      v446 = 10749;
      goto LABEL_509;
    case 48LL:
      v446 = 10753;
      goto LABEL_509;
    case 56LL:
      v446 = 10757;
      goto LABEL_509;
    case 64LL:
      v446 = 10761;
      goto LABEL_509;
    case 72LL:
      v446 = 10765;
      goto LABEL_509;
    case 80LL:
      v446 = 10769;
      goto LABEL_509;
    case 88LL:
      v446 = 10773;
      goto LABEL_509;
    case 96LL:
      v446 = 10781;
      goto LABEL_509;
    case 104LL:
      v446 = 10657;
      goto LABEL_509;
    case 112LL:
      v446 = 10661;
      goto LABEL_509;
    case 120LL:
      v446 = 10665;
      goto LABEL_509;
    case 128LL:
      v446 = 10669;
      goto LABEL_509;
    case 136LL:
      v446 = 10673;
      goto LABEL_509;
    case 144LL:
      v446 = 10677;
      goto LABEL_509;
    case 152LL:
      v446 = 10681;
      goto LABEL_509;
    case 160LL:
      v446 = 10685;
      goto LABEL_509;
    case 168LL:
      v446 = 10689;
      goto LABEL_509;
    case 176LL:
      v446 = 10697;
      goto LABEL_509;
    case 184LL:
      v446 = 10701;
      goto LABEL_509;
    case 192LL:
      v446 = 10705;
      goto LABEL_509;
    case 200LL:
      v446 = 10709;
      goto LABEL_509;
    case 208LL:
      v446 = 10713;
      goto LABEL_509;
    case 216LL:
      v446 = 10717;
      goto LABEL_509;
    case 224LL:
      v446 = 10721;
      goto LABEL_509;
    case 232LL:
      v446 = 10725;
      goto LABEL_509;
    case 240LL:
      v446 = 10729;
      goto LABEL_509;
    case 248LL:
      v446 = 10733;
      goto LABEL_509;
    case 256LL:
      v446 = 10741;
LABEL_509:
      v552 = sub_90A810(*(__int64 **)(v4 + 32), v446, 0, 0);
      v608 = (__int64)v610;
      v609 = 0x900000001LL;
      v610[0] = 0;
      v447 = sub_92F410(v4, v265);
      if ( (*(_BYTE *)(v552 + 2) & 1) != 0 )
        sub_B2C6D0(v552);
      v448 = *(_QWORD *)(*(_QWORD *)(v552 + 96) + 48LL);
      v538 = v4 + 48;
      v597 = sub_BD5D20(v447);
      LOWORD(v601) = 773;
      v598 = v449;
      v599 = (__int64)".asvecptr";
      v450 = sub_BCE760(v448, 0);
      if ( v450 != v447->m128i_i64[1] )
      {
        if ( v447->m128i_i8[0] > 0x15u )
        {
          v607 = 257;
          v447 = (__m128i *)sub_B52210(v447, v450, &v604, 0, 0);
          (*(void (__fastcall **)(_QWORD, __m128i *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
            *(_QWORD *)(v4 + 136),
            v447,
            &v597,
            *(_QWORD *)(v4 + 104),
            *(_QWORD *)(v4 + 112));
          v493 = *(unsigned int **)(v4 + 48);
          v494 = &v493[4 * *(unsigned int *)(v4 + 56)];
          if ( v493 != v494 )
          {
            v511 = v4;
            v495 = *(unsigned int **)(v4 + 48);
            v496 = v494;
            do
            {
              v497 = *((_QWORD *)v495 + 1);
              v498 = *v495;
              v495 += 4;
              sub_B99FD0(v447, v498, v497);
            }
            while ( v496 != v495 );
            v4 = v511;
          }
        }
        else
        {
          v451 = *(_QWORD *)(v4 + 128);
          v452 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v451 + 136LL);
          if ( v452 == sub_928970 )
            v447 = (__m128i *)sub_ADAFB0(v447, v450);
          else
            v447 = (__m128i *)v452(v451, (__int64)v447, v450);
          if ( v447->m128i_i8[0] > 0x1Cu )
          {
            (*(void (__fastcall **)(_QWORD, __m128i *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
              *(_QWORD *)(v4 + 136),
              v447,
              &v597,
              *(_QWORD *)(v4 + 104),
              *(_QWORD *)(v4 + 112));
            v453 = *(_QWORD *)(v4 + 48);
            v454 = 16LL * *(unsigned int *)(v4 + 56);
            if ( v453 != v453 + v454 )
            {
              v564 = v4;
              v455 = (unsigned int *)(v453 + v454);
              v456 = *(unsigned int **)(v4 + 48);
              do
              {
                v457 = *((_QWORD *)v456 + 1);
                v458 = *v456;
                v456 += 4;
                sub_B99FD0(v447, v458, v457);
              }
              while ( v455 != v456 );
              v4 = v564;
            }
          }
        }
      }
      v459 = sub_620FD0(*(_QWORD *)(v516 + 56), &v604);
      if ( (_DWORD)v604 )
        sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v516 + 36), 1);
      v518 = v459 != 0;
      if ( v459 )
        v460 = sub_926480(v4, (unsigned __int64)v447, v448, 4, 0);
      else
        v460 = sub_ACA8A0(v448);
      v461 = (unsigned int)v609;
      v462 = (unsigned int)v609 + 1LL;
      if ( v462 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v462, 8);
        v461 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v461) = v460;
      LODWORD(v609) = v609 + 1;
      v463 = sub_92F410(v4, (__int64)v540);
      if ( (*(_BYTE *)(v552 + 2) & 1) != 0 )
        sub_B2C6D0(v552);
      v543 = *(_QWORD *)(*(_QWORD *)(v552 + 96) + 88LL);
      if ( (unsigned int)*(unsigned __int8 *)(v543 + 8) - 17 > 1 )
      {
        v490 = (unsigned int)v609;
        v491 = (unsigned int)v609 + 1LL;
        if ( v491 > HIDWORD(v609) )
        {
          sub_C8D5F0(&v608, v610, v491, 8);
          v490 = (unsigned int)v609;
        }
        v477 = 1;
        *(_QWORD *)(v608 + 8 * v490) = v463;
        LODWORD(v609) = v609 + 1;
LABEL_540:
        v478 = sub_92F410(v4, v514);
        if ( (*(_BYTE *)(v552 + 2) & 1) != 0 )
          sub_B2C6D0(v552);
        v479 = (unsigned int)v609;
        v480 = (unsigned int)v609 + 1LL;
        if ( v480 > HIDWORD(v609) )
        {
          sub_C8D5F0(&v608, v610, v480, 8);
          v479 = (unsigned int)v609;
        }
        v481 = 0;
        *(_QWORD *)(v608 + 8 * v479) = v478;
        LODWORD(v609) = v609 + 1;
        if ( v477 )
        {
          v492 = sub_620FD0(*(_QWORD *)(v512 + 56), &v604);
          if ( (_DWORD)v604 )
            sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v512 + 36), 1);
          v481 = 8LL * (v492 & 1);
        }
        srce = v481;
        v482 = sub_620FD0(*(_QWORD *)(v528 + 56), &v604);
        if ( (_DWORD)v604 )
          sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v528 + 36), 1);
        v483 = srce & 0xFFFFFFFFFFFFFFFDLL | (2LL * (v482 & 1));
        v484 = sub_620FD0(*(_QWORD *)(v524 + 56), &v604);
        if ( (_DWORD)v604 )
          sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v524 + 36), 1);
        v485 = v483 & 0xFFFFFFFFFFFFFFEFLL | (16LL * (v484 & 1));
        v486 = sub_620FD0(*(_QWORD *)(v519 + 56), &v604);
        v487 = (void *)v485;
        if ( (_DWORD)v604 )
          sub_91B8A0("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(v519 + 36), 1);
        LOBYTE(v487) = v518 | (4 * (v486 & 1)) | v485 & 0xFA;
        if ( (*(_BYTE *)(v552 + 2) & 1) != 0 )
        {
          srcl = v487;
          sub_B2C6D0(v552);
          v487 = srcl;
        }
        v488 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(v552 + 96) + 8LL), v487, 0);
        *(_QWORD *)v608 = v488;
        v604 = (__m128i *)"mmafrag";
        v607 = 259;
        v489 = sub_921880((unsigned int **)v538, *(_QWORD *)(v552 + 24), v552, v608, v609, (__int64)&v604, 0);
        sub_923130(v4, v489, (unsigned __int64)v447, 4, 0);
        *(_BYTE *)(a1 + 12) &= ~1u;
        v37 = (__m128i *)v608;
        *(_QWORD *)a1 = 0;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
        v34 = v507;
        if ( v37 != (__m128i *)v610 )
LABEL_31:
          _libc_free(v37, v34);
        return a1;
      }
      v597 = sub_BD5D20(v463);
      LOWORD(v601) = 773;
      v598 = v464;
      v599 = (__int64)".asvecptr";
      v465 = sub_BCE760(v543, 0);
      if ( v465 == v463->m128i_i64[1] )
        goto LABEL_537;
      if ( v463->m128i_i8[0] > 0x15u )
      {
        v607 = 257;
        v463 = (__m128i *)sub_B52210(v463, v465, &v604, 0, 0);
        (*(void (__fastcall **)(_QWORD, __m128i *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
          *(_QWORD *)(v4 + 136),
          v463,
          &v597,
          *(_QWORD *)(v538 + 56),
          *(_QWORD *)(v538 + 64));
        v499 = *(unsigned int **)(v4 + 48);
        v500 = &v499[4 * *(unsigned int *)(v4 + 56)];
        if ( v499 == v500 )
          goto LABEL_537;
        v510 = v4;
        v501 = *(unsigned int **)(v4 + 48);
        v502 = v500;
        do
        {
          v503 = *((_QWORD *)v501 + 1);
          v504 = *v501;
          v501 += 4;
          sub_B99FD0(v463, v504, v503);
        }
        while ( v502 != v501 );
      }
      else
      {
        v466 = *(_QWORD *)(v4 + 128);
        v467 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v466 + 136LL);
        if ( v467 == sub_928970 )
          v463 = (__m128i *)sub_ADAFB0(v463, v465);
        else
          v463 = (__m128i *)v467(v466, (__int64)v463, v465);
        if ( v463->m128i_i8[0] <= 0x1Cu )
          goto LABEL_537;
        (*(void (__fastcall **)(_QWORD, __m128i *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v4 + 136) + 16LL))(
          *(_QWORD *)(v4 + 136),
          v463,
          &v597,
          *(_QWORD *)(v538 + 56),
          *(_QWORD *)(v538 + 64));
        v468 = *(_QWORD *)(v4 + 48);
        v469 = 16LL * *(unsigned int *)(v4 + 56);
        if ( v468 == v468 + v469 )
          goto LABEL_537;
        v510 = v4;
        v470 = (unsigned int *)(v468 + v469);
        v471 = *(unsigned int **)(v4 + 48);
        do
        {
          v472 = *((_QWORD *)v471 + 1);
          v473 = *v471;
          v471 += 4;
          sub_B99FD0(v463, v473, v472);
        }
        while ( v470 != v471 );
      }
      v4 = v510;
LABEL_537:
      v474 = sub_926480(v4, (unsigned __int64)v463, v543, 4, 0);
      v475 = (unsigned int)v609;
      v476 = (unsigned int)v609 + 1LL;
      if ( v476 > HIDWORD(v609) )
      {
        sub_C8D5F0(&v608, v610, v476, 8);
        v475 = (unsigned int)v609;
      }
      *(_QWORD *)(v608 + 8 * v475) = v474;
      v477 = 0;
      LODWORD(v609) = v609 + 1;
      goto LABEL_540;
    default:
      goto LABEL_707;
  }
}
