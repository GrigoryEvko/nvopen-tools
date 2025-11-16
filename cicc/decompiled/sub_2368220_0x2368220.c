// Function: sub_2368220
// Address: 0x2368220
//
__int64 *__fastcall sub_2368220(__int64 *a1, __m128i *a2, unsigned __int64 *a3, __int64 a4)
{
  const __m128i *v5; // r15
  const __m128i *v6; // r14
  const void *v7; // rbx
  size_t v8; // r12
  __int64 v9; // rdi
  __m128i *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r12
  const __m128i *v13; // rax
  __int64 v14; // r15
  __int64 v15; // r12
  __m128i v16; // xmm4
  __int64 v17; // rbx
  __int64 kk; // r12
  unsigned __int64 *v19; // rsi
  __m128i *v20; // rdi
  char *m128i_i8; // rsi
  bool v22; // al
  unsigned __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rbx
  signed __int64 v28; // r12
  unsigned __int64 v29; // r12
  __int64 v30; // r15
  __int64 v31; // r12
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r12
  unsigned __int64 v35; // r14
  __int64 v36; // r15
  __int64 v37; // r14
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r12
  unsigned __int64 v44; // r14
  __int64 v45; // r15
  __int64 v46; // r14
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 v49; // r15
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // r12
  unsigned __int64 v53; // r14
  __int64 v54; // r15
  __int64 v55; // r14
  __int64 v56; // r12
  __int64 v57; // rax
  __int64 v58; // r15
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // r12
  unsigned __int64 v62; // r14
  __int64 v63; // r15
  const __m128i *v64; // r15
  signed __int64 v65; // rax
  __int64 v66; // r14
  __int64 v67; // rbx
  unsigned __int64 v68; // rax
  __int64 v69; // r14
  const __m128i *v70; // rbx
  unsigned __int64 v71; // r12
  __m128i *v72; // r13
  bool v73; // bl
  __m128i *v74; // r14
  __int64 v75; // r14
  __int64 v76; // rbx
  unsigned __int64 v77; // rax
  __int64 v78; // r14
  const __m128i *v79; // rbx
  unsigned __int64 v80; // r12
  __m128i *v81; // r13
  bool v82; // bl
  __m128i *v83; // r14
  __int64 v84; // r14
  __int64 v85; // rbx
  unsigned __int64 v86; // rax
  __int64 v87; // r14
  const __m128i *v88; // rbx
  unsigned __int64 v89; // r12
  __m128i *v90; // r13
  bool v91; // bl
  __m128i *v92; // r14
  __int64 v93; // r14
  __int64 v94; // rbx
  unsigned __int64 v95; // rax
  __int64 v96; // r14
  const __m128i *v97; // rbx
  unsigned __int64 v98; // r12
  __m128i *v99; // r13
  bool v100; // bl
  __m128i *v101; // r14
  char v102; // bl
  char v103; // r14
  _QWORD *v104; // rax
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // r15
  void *v110; // r15
  size_t v111; // r12
  _QWORD *v113; // rax
  unsigned int v114; // eax
  unsigned int v115; // ebx
  __int64 v116; // rdx
  __int64 v117; // r14
  __int64 v118; // rbx
  _QWORD *v119; // r12
  __int64 v120; // rax
  _QWORD *v121; // rax
  __int64 v122; // rdx
  _QWORD *v123; // rax
  _QWORD *v124; // rax
  _QWORD *v125; // rax
  char v126; // dl
  __m128i *v127; // rsi
  __int32 v128; // ebx
  __int64 v129; // rax
  char v130; // dl
  _QWORD *v131; // rax
  _QWORD *v132; // rax
  __int8 v133; // bl
  __int64 v134; // rax
  char v135; // dl
  _QWORD *v136; // rax
  _QWORD *v137; // rax
  __int64 v138; // rbx
  _QWORD *v139; // rax
  _QWORD *v140; // rax
  _QWORD *v141; // rax
  _QWORD *v142; // rax
  __int64 v143; // rbx
  _QWORD *v144; // rax
  _QWORD *v145; // rax
  __int8 v146; // bl
  __int64 v147; // rax
  char v148; // dl
  _QWORD *v149; // rax
  _QWORD *v150; // rax
  _QWORD *v151; // rax
  _QWORD *v152; // rax
  _QWORD *v153; // rax
  _QWORD *v154; // rax
  _QWORD *v155; // rax
  _QWORD *v156; // rax
  _QWORD *v157; // rax
  _QWORD *v158; // rax
  _QWORD *v159; // rax
  _QWORD *v160; // rax
  _QWORD *v161; // rax
  _QWORD *v162; // rax
  _QWORD *v163; // rax
  _QWORD *v164; // rax
  __int64 v165; // rbx
  __int64 *m; // r12
  char v167; // dl
  __m128i *v168; // rsi
  _QWORD *v169; // rax
  _QWORD *v170; // rax
  _QWORD *v171; // rax
  _QWORD *v172; // rax
  _QWORD *v173; // rax
  _QWORD *v174; // rax
  _QWORD *v175; // rax
  _QWORD *v176; // rax
  _QWORD *v177; // rax
  _QWORD *v178; // rax
  _QWORD *v179; // rax
  _QWORD *v180; // rax
  _QWORD *v181; // rax
  _QWORD *v182; // rax
  _QWORD *v183; // rax
  _QWORD *v184; // rax
  _QWORD *v185; // rax
  _QWORD *v186; // rax
  _QWORD *v187; // rax
  _QWORD *v188; // rax
  _QWORD *v189; // rax
  _QWORD *v190; // rax
  _QWORD *v191; // rax
  _QWORD *v192; // rax
  _QWORD *v193; // rax
  _QWORD *v194; // rax
  _QWORD *v195; // rax
  _QWORD *v196; // rax
  _QWORD *v197; // rax
  _QWORD *v198; // rax
  _QWORD *v199; // rax
  _QWORD *v200; // rax
  __int64 v201; // rdx
  __int64 v202; // rcx
  __int64 v203; // r8
  __int64 v204; // r9
  _QWORD *v205; // rax
  __int64 v206; // rdx
  __int64 v207; // rcx
  __int64 v208; // r8
  __int64 v209; // r9
  __int64 v210; // rbx
  char v211; // dl
  __m128i *v212; // rsi
  _QWORD *v213; // rax
  _QWORD *v214; // rax
  _QWORD *v215; // rax
  _QWORD *v216; // rax
  _QWORD *v217; // rax
  _QWORD *v218; // rax
  _QWORD *v219; // rax
  _QWORD *v220; // rax
  __m128i *v221; // rdi
  __int64 i; // rcx
  _QWORD *v223; // rax
  _QWORD *v224; // rax
  _QWORD *v225; // rax
  _QWORD *v226; // rax
  _QWORD *v227; // rax
  _QWORD *v228; // rax
  _QWORD *v229; // rax
  _QWORD *v230; // rax
  _QWORD *v231; // rax
  _QWORD *v232; // rax
  _QWORD *v233; // rax
  _QWORD *v234; // rax
  _QWORD *v235; // rax
  _QWORD *v236; // rax
  _QWORD *v237; // rax
  _QWORD *v238; // rax
  _QWORD *v239; // rax
  _QWORD *v240; // rax
  _QWORD *v241; // rax
  _QWORD *v242; // rax
  _QWORD *v243; // rax
  _QWORD *v244; // rax
  _QWORD *v245; // rax
  _QWORD *v246; // rax
  _QWORD *v247; // rax
  _QWORD *v248; // rax
  _QWORD *v249; // rax
  _QWORD *v250; // rax
  _QWORD *v251; // rax
  _QWORD *v252; // rax
  __int64 v253; // rdx
  __int64 v254; // rcx
  __int64 v255; // r8
  __int64 v256; // r9
  _QWORD *v257; // rax
  __int64 v258; // rdx
  __int64 v259; // rcx
  __int64 v260; // r8
  __int64 v261; // r9
  _QWORD *v262; // rbx
  __int64 v263; // rbx
  _QWORD *v264; // rax
  __int64 v265; // rbx
  _QWORD *v266; // rax
  _QWORD *v267; // rax
  _QWORD *v268; // rax
  _QWORD *v269; // rax
  _QWORD *v270; // rax
  _QWORD *v271; // rax
  __int64 v272; // rbx
  _QWORD *v273; // rax
  _QWORD *v274; // rax
  _QWORD *v275; // rax
  _QWORD *v276; // rax
  _QWORD *v277; // rax
  _QWORD *v278; // rax
  _QWORD *v279; // rax
  _QWORD *v280; // rax
  _QWORD *v281; // rax
  _QWORD *v282; // rax
  __int64 v283; // rbx
  _QWORD *v284; // rax
  __int64 v285; // rbx
  _QWORD *v286; // rax
  __int64 v287; // rbx
  _QWORD *v288; // rax
  _QWORD *v289; // rax
  _QWORD *v290; // rax
  __int64 v291; // rdx
  __int64 v292; // rdx
  __int64 v293; // rdx
  __int64 v294; // rbx
  _QWORD *v295; // rax
  __int64 v296; // rdx
  __int64 v297; // rcx
  __int64 v298; // r8
  __int64 v299; // r9
  char v300; // dl
  __m128i *v301; // rsi
  _QWORD *v302; // rax
  __int64 v303; // rbx
  _QWORD *v304; // rax
  void *v305; // rbx
  _QWORD *v306; // rax
  void *v307; // rbx
  _QWORD *v308; // rax
  void *v309; // rbx
  _QWORD *v310; // rax
  void *v311; // rbx
  _QWORD *v312; // rax
  void *v313; // rbx
  _QWORD *v314; // rax
  void *v315; // rbx
  _QWORD *v316; // rax
  void *v317; // rax
  __int64 v318; // rbx
  _QWORD *v319; // rax
  void *v320; // rax
  _QWORD *v321; // rax
  _QWORD *v322; // rax
  _QWORD *v323; // rax
  _QWORD *v324; // rax
  _QWORD *v325; // rax
  _QWORD *v326; // rax
  _QWORD *v327; // rax
  _QWORD *v328; // rax
  _QWORD *v329; // rax
  _QWORD *v330; // rax
  __m128i *v331; // rdi
  __int64 j; // rcx
  _QWORD *v333; // rax
  __int64 v334; // rax
  _QWORD *v335; // rax
  _QWORD *v336; // rax
  _QWORD *v337; // rax
  _QWORD *v338; // rax
  _QWORD *v339; // rax
  _QWORD *v340; // rax
  _QWORD *v341; // rax
  _QWORD *v342; // rax
  _QWORD *v343; // rax
  _QWORD *v344; // rax
  _QWORD *v345; // rax
  _QWORD *v346; // rax
  _QWORD *v347; // rax
  _QWORD *v348; // rax
  _QWORD *v349; // rax
  _QWORD *v350; // rax
  _QWORD *v351; // rax
  _QWORD *v352; // rax
  _QWORD *v353; // rax
  _QWORD *v354; // rax
  _QWORD *v355; // rax
  _QWORD *v356; // rax
  __int64 v357; // rbx
  _QWORD *v358; // rax
  _QWORD *v359; // rax
  _QWORD *v360; // rax
  __int64 v361; // rdx
  __int64 v362; // rcx
  __int64 v363; // r8
  __int64 v364; // r9
  _QWORD *v365; // rax
  _QWORD *v366; // rax
  __int64 v367; // rbx
  _QWORD *v368; // rax
  __int64 v369; // rbx
  _QWORD *v370; // rax
  _QWORD *v371; // rax
  _QWORD *v372; // rax
  _QWORD *v373; // rax
  _QWORD *v374; // rax
  _QWORD *v375; // rax
  __int32 v376; // ebx
  __int64 v377; // rax
  __int64 v378; // rbx
  _QWORD *v379; // rax
  _QWORD *v380; // rax
  _QWORD *v381; // rax
  _QWORD *v382; // rax
  _QWORD *v383; // rax
  _QWORD *v384; // rax
  __int64 v385; // rdx
  __int64 v386; // rcx
  __int64 v387; // r8
  __int64 v388; // r9
  __int64 v389; // rbx
  _QWORD *v390; // rax
  _QWORD *v391; // rax
  _QWORD *v392; // rax
  _QWORD *v393; // rax
  _QWORD *v394; // rbx
  __int64 v395; // rbx
  _QWORD *v396; // rax
  _QWORD *v397; // rax
  _QWORD *v398; // rax
  _QWORD *v399; // rax
  _QWORD *v400; // rbx
  void *v401; // rax
  __int64 v402; // rbx
  _QWORD *v403; // rax
  void *v404; // rbx
  _QWORD *v405; // rax
  void *v406; // rbx
  _QWORD *v407; // rax
  void *v408; // rax
  __int64 v409; // rbx
  _QWORD *v410; // rax
  void *v411; // rax
  __int64 v412; // rbx
  _QWORD *v413; // rax
  void *v414; // rbx
  _QWORD *v415; // rax
  void *v416; // rbx
  _QWORD *v417; // rax
  void *v418; // rbx
  _QWORD *v419; // rax
  void *v420; // rbx
  _QWORD *v421; // rax
  void *v422; // rbx
  _QWORD *v423; // rax
  void *v424; // rbx
  _QWORD *v425; // rax
  void *v426; // rbx
  _QWORD *v427; // rax
  void *v428; // rax
  __int64 v429; // rbx
  _QWORD *v430; // rax
  void *v431; // rax
  __int64 v432; // rbx
  _QWORD *v433; // rax
  void *v434; // rbx
  _QWORD *v435; // rax
  void *v436; // rax
  __int64 v437; // rbx
  _QWORD *v438; // rax
  void *v439; // rbx
  _QWORD *v440; // rax
  void *v441; // rax
  __int64 v442; // rbx
  _QWORD *v443; // rax
  void *v444; // rbx
  _QWORD *v445; // rax
  void *v446; // rbx
  _QWORD *v447; // rax
  void *v448; // rbx
  _QWORD *v449; // rax
  _QWORD *v450; // rax
  _QWORD *v451; // rax
  _QWORD *v452; // rax
  __int64 v453; // rax
  __int64 v454; // rax
  _QWORD *v455; // rax
  _QWORD *v456; // rax
  _QWORD *v457; // rax
  _QWORD *v458; // rax
  _QWORD *v459; // rax
  _QWORD *v460; // rax
  _QWORD *v461; // rax
  __int64 v462; // rbx
  _QWORD *v463; // rax
  _QWORD *v464; // rax
  __m128i *v465; // rdi
  __int64 k; // rcx
  __int64 v467; // rdx
  __int64 v468; // rcx
  __int64 v469; // r8
  __int64 v470; // r9
  __int64 v471; // rbx
  _QWORD *v472; // rax
  _QWORD *v473; // rax
  _QWORD *v474; // rax
  _QWORD *v475; // rax
  _QWORD *v476; // rax
  _QWORD *v477; // rax
  _QWORD *v478; // rax
  _QWORD *v479; // rax
  _QWORD *v480; // rax
  _QWORD *v481; // rax
  _QWORD *v482; // rax
  _QWORD *v483; // rax
  _QWORD *v484; // rax
  _QWORD *v485; // rax
  _QWORD *v486; // rax
  __m128i *v487; // rsi
  __m128i *v488; // rdi
  __int64 n; // rcx
  __m128i *v490; // rdi
  __m128i *v491; // rsi
  __int64 ii; // rcx
  __m128i *v493; // rdi
  __m128i *v494; // rsi
  __int64 jj; // rcx
  char v496; // dl
  __m128i *v497; // rsi
  __int64 v498; // rdx
  __m128i *v499; // rsi
  __m128i v500; // kr00_16
  __m128i v501; // kr10_16
  __m128i v502; // kr20_16
  __int64 v503; // rax
  __int64 v504; // rdx
  unsigned __int64 v505; // r13
  __int64 v506; // r14
  __int64 v507; // rbx
  _QWORD *v508; // rax
  __int64 v509; // rdi
  char v510; // dl
  char v511; // dl
  __int8 v512; // bl
  __int64 v513; // rax
  __int8 v514; // bl
  __int64 v515; // rax
  char v516; // dl
  char v517; // dl
  __int8 v518; // bl
  __int64 v519; // rax
  __int8 v520; // bl
  char v521; // dl
  char v522; // dl
  __m128i *v523; // rsi
  char v524; // bl
  void *v525; // r12
  __int64 v526; // rax
  __int32 v527; // r12d
  void *v528; // rbx
  __int64 v529; // rax
  char v530; // dl
  __m128i *v531; // rsi
  char v532; // dl
  char v533; // dl
  __m128i *v534; // rsi
  char v535; // dl
  __m128i v536; // xmm3
  __int64 v537; // rbx
  __m128i v538; // kr40_16
  __int64 v539; // rax
  __int8 v540; // r12
  __int64 v541; // rbx
  __int64 v542; // rax
  char v543; // dl
  __m128i *v544; // rsi
  char v545; // dl
  __int8 v546; // bl
  __int64 v547; // rax
  __int8 v548; // bl
  __int64 v549; // rax
  char v550; // dl
  char v551; // dl
  __m128i *v552; // rsi
  __int8 v553; // bl
  __int64 v554; // rax
  __int64 v555; // rax
  char v556; // dl
  __m128i *v557; // rsi
  char v558; // dl
  __int16 v559; // bx
  __int64 v560; // rax
  __int8 v561; // bl
  __int64 v562; // rax
  char v563; // dl
  void *v564; // r15
  size_t v565; // r12
  __int64 v566; // r8
  __int64 v567; // r9
  __int64 v568; // r8
  __int64 v569; // r9
  __int64 v570; // r9
  __int64 v571; // r8
  __int64 v572; // r9
  __int64 v573; // r9
  __int8 v574; // bl
  __int64 v575; // rax
  char v576; // dl
  __m128i *v577; // rsi
  void *v578; // rax
  void *v579; // rax
  void *v580; // rax
  void *v581; // rax
  __int64 v582; // r9
  char v583; // dl
  void *v584; // rax
  char v585; // dl
  char v586; // dl
  __int64 v587; // r14
  __int64 v588; // rbx
  __m128i v589; // xmm3
  __int64 v590; // rax
  char v591; // al
  __int8 v592; // bl
  __int64 v593; // rax
  unsigned int v594; // eax
  unsigned int v595; // ebx
  __int64 v596; // rdx
  __int64 v597; // r14
  char v598; // dl
  char v599; // [rsp+Ch] [rbp-2D24h]
  __int16 v600; // [rsp+Ch] [rbp-2D24h]
  char v601; // [rsp+10h] [rbp-2D20h]
  __int64 v602; // [rsp+10h] [rbp-2D20h]
  const __m128i *v603; // [rsp+18h] [rbp-2D18h]
  const __m128i *v604; // [rsp+20h] [rbp-2D10h]
  __m128i v605; // [rsp+20h] [rbp-2D10h]
  __int64 v606; // [rsp+28h] [rbp-2D08h]
  _QWORD *v607; // [rsp+28h] [rbp-2D08h]
  _QWORD *v608; // [rsp+28h] [rbp-2D08h]
  _QWORD *v609; // [rsp+28h] [rbp-2D08h]
  _QWORD *v610; // [rsp+28h] [rbp-2D08h]
  __int64 v611; // [rsp+38h] [rbp-2CF8h]
  __int64 v612; // [rsp+38h] [rbp-2CF8h]
  __int64 v613; // [rsp+38h] [rbp-2CF8h]
  __int64 v614; // [rsp+38h] [rbp-2CF8h]
  __int64 v615; // [rsp+38h] [rbp-2CF8h]
  __m128i v616; // [rsp+38h] [rbp-2CF8h]
  __m128i *v617; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v618; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v619; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v620; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v621; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v622; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v623; // [rsp+40h] [rbp-2CF0h]
  unsigned __int64 v624; // [rsp+40h] [rbp-2CF0h]
  __m128i v627; // [rsp+60h] [rbp-2CD0h] BYREF
  unsigned __int64 v628; // [rsp+70h] [rbp-2CC0h]
  __int64 v629; // [rsp+78h] [rbp-2CB8h]
  unsigned __int64 v630; // [rsp+80h] [rbp-2CB0h]
  __m128i v631; // [rsp+90h] [rbp-2CA0h] BYREF
  unsigned __int64 v632; // [rsp+A0h] [rbp-2C90h]
  __int64 v633; // [rsp+A8h] [rbp-2C88h]
  __int64 mm; // [rsp+B0h] [rbp-2C80h]
  __m128i v635; // [rsp+C0h] [rbp-2C70h] BYREF
  __int64 v636; // [rsp+D0h] [rbp-2C60h]
  __int64 v637; // [rsp+D8h] [rbp-2C58h]
  __int64 nn; // [rsp+E0h] [rbp-2C50h] BYREF
  __m128i v639[352]; // [rsp+100h] [rbp-2C30h] BYREF
  __m128i v640[355]; // [rsp+1700h] [rbp-1630h] BYREF

  v5 = *(const __m128i **)(a4 + 24);
  v6 = *(const __m128i **)(a4 + 16);
  v617 = a2;
  if ( v6 == v5 )
  {
    v110 = *(void **)a4;
    v111 = *(_QWORD *)(a4 + 8);
    if ( sub_9691B0(*(const void **)a4, v111, "require<aa>", 11) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A127B8;
    }
    else if ( sub_9691B0(v110, v111, "invalidate<aa>", 14) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A127F8;
    }
    else if ( sub_9691B0(v110, v111, "require<access-info>", 20) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A12838;
    }
    else if ( sub_9691B0(v110, v111, "invalidate<access-info>", 23) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A12878;
    }
    else if ( sub_9691B0(v110, v111, "require<assumptions>", 20) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A128B8;
    }
    else if ( sub_9691B0(v110, v111, "invalidate<assumptions>", 23) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A128F8;
    }
    else if ( sub_9691B0(v110, v111, "require<bb-sections-profile-reader>", 35) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A12938;
    }
    else if ( sub_9691B0(v110, v111, "invalidate<bb-sections-profile-reader>", 38) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A12978;
    }
    else if ( sub_9691B0(v110, v111, "require<block-freq>", 19) )
    {
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A129B8;
    }
    else
    {
      if ( !sub_9691B0(v110, v111, "invalidate<block-freq>", 22) )
      {
        if ( sub_9691B0(v110, v111, "require<branch-prob>", 20) )
        {
          v123 = (_QWORD *)sub_22077B0(0x10u);
          if ( v123 )
            *v123 = &unk_4A12A38;
          v640[0].m128i_i64[0] = (__int64)v123;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<branch-prob>", 23) )
        {
          v125 = (_QWORD *)sub_22077B0(0x10u);
          if ( v125 )
            *v125 = &unk_4A12A78;
          v640[0].m128i_i64[0] = (__int64)v125;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<cycles>", 15) )
        {
          v124 = (_QWORD *)sub_22077B0(0x10u);
          if ( v124 )
            *v124 = &unk_4A12AB8;
          v640[0].m128i_i64[0] = (__int64)v124;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<cycles>", 18) )
        {
          v158 = (_QWORD *)sub_22077B0(0x10u);
          if ( v158 )
            *v158 = &unk_4A12AF8;
          v640[0].m128i_i64[0] = (__int64)v158;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<da>", 11) )
        {
          v157 = (_QWORD *)sub_22077B0(0x10u);
          if ( v157 )
            *v157 = &unk_4A12B38;
          v640[0].m128i_i64[0] = (__int64)v157;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<da>", 14) )
        {
          v156 = (_QWORD *)sub_22077B0(0x10u);
          if ( v156 )
            *v156 = &unk_4A12B78;
          v640[0].m128i_i64[0] = (__int64)v156;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<debug-ata>", 18) )
        {
          v155 = (_QWORD *)sub_22077B0(0x10u);
          if ( v155 )
            *v155 = &unk_4A12BB8;
          v640[0].m128i_i64[0] = (__int64)v155;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<debug-ata>", 21) )
        {
          v154 = (_QWORD *)sub_22077B0(0x10u);
          if ( v154 )
            *v154 = &unk_4A12BF8;
          v640[0].m128i_i64[0] = (__int64)v154;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<demanded-bits>", 22) )
        {
          v153 = (_QWORD *)sub_22077B0(0x10u);
          if ( v153 )
            *v153 = &unk_4A12C38;
          v640[0].m128i_i64[0] = (__int64)v153;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<demanded-bits>", 25) )
        {
          v152 = (_QWORD *)sub_22077B0(0x10u);
          if ( v152 )
            *v152 = &unk_4A12C78;
          v640[0].m128i_i64[0] = (__int64)v152;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<domfrontier>", 20) )
        {
          v151 = (_QWORD *)sub_22077B0(0x10u);
          if ( v151 )
            *v151 = &unk_4A12CB8;
          v640[0].m128i_i64[0] = (__int64)v151;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<domfrontier>", 23) )
        {
          v162 = (_QWORD *)sub_22077B0(0x10u);
          if ( v162 )
            *v162 = &unk_4A12CF8;
          v640[0].m128i_i64[0] = (__int64)v162;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<domtree>", 16) )
        {
          v161 = (_QWORD *)sub_22077B0(0x10u);
          if ( v161 )
            *v161 = &unk_4A12D38;
          v640[0].m128i_i64[0] = (__int64)v161;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<domtree>", 19) )
        {
          v160 = (_QWORD *)sub_22077B0(0x10u);
          if ( v160 )
            *v160 = &unk_4A12D78;
          v640[0].m128i_i64[0] = (__int64)v160;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<func-properties>", 24) )
        {
          v159 = (_QWORD *)sub_22077B0(0x10u);
          if ( v159 )
            *v159 = &unk_4A12DB8;
          v640[0].m128i_i64[0] = (__int64)v159;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<func-properties>", 27) )
        {
          v164 = (_QWORD *)sub_22077B0(0x10u);
          if ( v164 )
            *v164 = &unk_4A12DF8;
          v640[0].m128i_i64[0] = (__int64)v164;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<machine-function-info>", 30) )
        {
          v163 = (_QWORD *)sub_22077B0(0x10u);
          if ( v163 )
            *v163 = &unk_4A12E38;
          v640[0].m128i_i64[0] = (__int64)v163;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<machine-function-info>", 33) )
        {
          v150 = (_QWORD *)sub_22077B0(0x10u);
          if ( v150 )
            *v150 = &unk_4A12E78;
          v640[0].m128i_i64[0] = (__int64)v150;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<gc-function>", 20) )
        {
          v149 = (_QWORD *)sub_22077B0(0x10u);
          if ( v149 )
            *v149 = &unk_4A12EB8;
          v640[0].m128i_i64[0] = (__int64)v149;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<gc-function>", 23) )
        {
          v186 = (_QWORD *)sub_22077B0(0x10u);
          if ( v186 )
            *v186 = &unk_4A12EF8;
          v640[0].m128i_i64[0] = (__int64)v186;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<inliner-size-estimator>", 31) )
        {
          v185 = (_QWORD *)sub_22077B0(0x10u);
          if ( v185 )
            *v185 = &unk_4A12F38;
          v640[0].m128i_i64[0] = (__int64)v185;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<inliner-size-estimator>", 34) )
        {
          v184 = (_QWORD *)sub_22077B0(0x10u);
          if ( v184 )
            *v184 = &unk_4A12F78;
          v640[0].m128i_i64[0] = (__int64)v184;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<last-run-tracking>", 26) )
        {
          v183 = (_QWORD *)sub_22077B0(0x10u);
          if ( v183 )
            *v183 = &unk_4A12FB8;
          v640[0].m128i_i64[0] = (__int64)v183;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<last-run-tracking>", 29) )
        {
          v182 = (_QWORD *)sub_22077B0(0x10u);
          if ( v182 )
            *v182 = &unk_4A12FF8;
          v640[0].m128i_i64[0] = (__int64)v182;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<lazy-value-info>", 24) )
        {
          v181 = (_QWORD *)sub_22077B0(0x10u);
          if ( v181 )
            *v181 = &unk_4A13038;
          v640[0].m128i_i64[0] = (__int64)v181;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<lazy-value-info>", 27) )
        {
          v180 = (_QWORD *)sub_22077B0(0x10u);
          if ( v180 )
            *v180 = &unk_4A13078;
          v640[0].m128i_i64[0] = (__int64)v180;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<loops>", 14) )
        {
          v179 = (_QWORD *)sub_22077B0(0x10u);
          if ( v179 )
            *v179 = &unk_4A130B8;
          v640[0].m128i_i64[0] = (__int64)v179;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<loops>", 17) )
        {
          v178 = (_QWORD *)sub_22077B0(0x10u);
          if ( v178 )
            *v178 = &unk_4A130F8;
          v640[0].m128i_i64[0] = (__int64)v178;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<memdep>", 15) )
        {
          v177 = (_QWORD *)sub_22077B0(0x10u);
          if ( v177 )
            *v177 = &unk_4A13138;
          v640[0].m128i_i64[0] = (__int64)v177;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<memdep>", 18) )
        {
          v176 = (_QWORD *)sub_22077B0(0x10u);
          if ( v176 )
            *v176 = &unk_4A13178;
          v640[0].m128i_i64[0] = (__int64)v176;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<memoryssa>", 18) )
        {
          v175 = (_QWORD *)sub_22077B0(0x10u);
          if ( v175 )
            *v175 = &unk_4A131B8;
          v640[0].m128i_i64[0] = (__int64)v175;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<memoryssa>", 21) )
        {
          v174 = (_QWORD *)sub_22077B0(0x10u);
          if ( v174 )
            *v174 = &unk_4A131F8;
          v640[0].m128i_i64[0] = (__int64)v174;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<no-op-function>", 23) )
        {
          v173 = (_QWORD *)sub_22077B0(0x10u);
          if ( v173 )
            *v173 = &unk_4A13238;
          v640[0].m128i_i64[0] = (__int64)v173;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<no-op-function>", 26) )
        {
          v172 = (_QWORD *)sub_22077B0(0x10u);
          if ( v172 )
            *v172 = &unk_4A13278;
          v640[0].m128i_i64[0] = (__int64)v172;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<opt-remark-emit>", 24) )
        {
          v171 = (_QWORD *)sub_22077B0(0x10u);
          if ( v171 )
            *v171 = &unk_4A132B8;
          v640[0].m128i_i64[0] = (__int64)v171;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<opt-remark-emit>", 27) )
        {
          v194 = (_QWORD *)sub_22077B0(0x10u);
          if ( v194 )
            *v194 = &unk_4A132F8;
          v640[0].m128i_i64[0] = (__int64)v194;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<pass-instrumentation>", 29) )
        {
          v193 = (_QWORD *)sub_22077B0(0x10u);
          if ( v193 )
            *v193 = &unk_4A13338;
          v640[0].m128i_i64[0] = (__int64)v193;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<pass-instrumentation>", 32) )
        {
          v192 = (_QWORD *)sub_22077B0(0x10u);
          if ( v192 )
            *v192 = &unk_4A13378;
          v640[0].m128i_i64[0] = (__int64)v192;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<phi-values>", 19) )
        {
          v191 = (_QWORD *)sub_22077B0(0x10u);
          if ( v191 )
            *v191 = &unk_4A133B8;
          v640[0].m128i_i64[0] = (__int64)v191;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<phi-values>", 22) )
        {
          v190 = (_QWORD *)sub_22077B0(0x10u);
          if ( v190 )
            *v190 = &unk_4A133F8;
          v640[0].m128i_i64[0] = (__int64)v190;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<postdomtree>", 20) )
        {
          v189 = (_QWORD *)sub_22077B0(0x10u);
          if ( v189 )
            *v189 = &unk_4A13438;
          v640[0].m128i_i64[0] = (__int64)v189;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<postdomtree>", 23) )
        {
          v188 = (_QWORD *)sub_22077B0(0x10u);
          if ( v188 )
            *v188 = &unk_4A13478;
          v640[0].m128i_i64[0] = (__int64)v188;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<regions>", 16) )
        {
          v187 = (_QWORD *)sub_22077B0(0x10u);
          if ( v187 )
            *v187 = &unk_4A134B8;
          v640[0].m128i_i64[0] = (__int64)v187;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<regions>", 19) )
        {
          v198 = (_QWORD *)sub_22077B0(0x10u);
          if ( v198 )
            *v198 = &unk_4A134F8;
          v640[0].m128i_i64[0] = (__int64)v198;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<scalar-evolution>", 25) )
        {
          v197 = (_QWORD *)sub_22077B0(0x10u);
          if ( v197 )
            *v197 = &unk_4A13538;
          v640[0].m128i_i64[0] = (__int64)v197;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<scalar-evolution>", 28) )
        {
          v196 = (_QWORD *)sub_22077B0(0x10u);
          if ( v196 )
            *v196 = &unk_4A13578;
          v640[0].m128i_i64[0] = (__int64)v196;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<should-not-run-function-passes>", 39) )
        {
          v195 = (_QWORD *)sub_22077B0(0x10u);
          if ( v195 )
            *v195 = &unk_4A135B8;
          v640[0].m128i_i64[0] = (__int64)v195;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<should-not-run-function-passes>", 42) )
        {
          v200 = (_QWORD *)sub_22077B0(0x10u);
          if ( v200 )
            *v200 = &unk_4A135F8;
          v640[0].m128i_i64[0] = (__int64)v200;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<should-run-extra-vector-passes>", 39) )
        {
          v199 = (_QWORD *)sub_22077B0(0x10u);
          if ( v199 )
            *v199 = &unk_4A13638;
          v640[0].m128i_i64[0] = (__int64)v199;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<should-run-extra-vector-passes>", 42) )
        {
          v170 = (_QWORD *)sub_22077B0(0x10u);
          if ( v170 )
            *v170 = &unk_4A13678;
          v640[0].m128i_i64[0] = (__int64)v170;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<ssp-layout>", 19) )
        {
          v169 = (_QWORD *)sub_22077B0(0x10u);
          if ( v169 )
            *v169 = &unk_4A136B8;
          v640[0].m128i_i64[0] = (__int64)v169;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<ssp-layout>", 22) )
        {
          v247 = (_QWORD *)sub_22077B0(0x10u);
          if ( v247 )
            *v247 = &unk_4A136F8;
          v640[0].m128i_i64[0] = (__int64)v247;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<stack-safety-local>", 27) )
        {
          v246 = (_QWORD *)sub_22077B0(0x10u);
          if ( v246 )
            *v246 = &unk_4A13738;
          v640[0].m128i_i64[0] = (__int64)v246;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<stack-safety-local>", 30) )
        {
          v245 = (_QWORD *)sub_22077B0(0x10u);
          if ( v245 )
            *v245 = &unk_4A13778;
          v640[0].m128i_i64[0] = (__int64)v245;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<target-ir>", 18) )
        {
          v244 = (_QWORD *)sub_22077B0(0x10u);
          if ( v244 )
            *v244 = &unk_4A137B8;
          v640[0].m128i_i64[0] = (__int64)v244;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<target-ir>", 21) )
        {
          v243 = (_QWORD *)sub_22077B0(0x10u);
          if ( v243 )
            *v243 = &unk_4A137F8;
          v640[0].m128i_i64[0] = (__int64)v243;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<target-lib-info>", 24) )
        {
          v242 = (_QWORD *)sub_22077B0(0x10u);
          if ( v242 )
            *v242 = &unk_4A13838;
          v640[0].m128i_i64[0] = (__int64)v242;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<target-lib-info>", 27) )
        {
          v241 = (_QWORD *)sub_22077B0(0x10u);
          if ( v241 )
            *v241 = &unk_4A13878;
          v640[0].m128i_i64[0] = (__int64)v241;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<uniformity>", 19) )
        {
          v240 = (_QWORD *)sub_22077B0(0x10u);
          if ( v240 )
            *v240 = &unk_4A138B8;
          v640[0].m128i_i64[0] = (__int64)v240;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<uniformity>", 22) )
        {
          v239 = (_QWORD *)sub_22077B0(0x10u);
          if ( v239 )
            *v239 = &unk_4A138F8;
          v640[0].m128i_i64[0] = (__int64)v239;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<verify>", 15) )
        {
          v238 = (_QWORD *)sub_22077B0(0x10u);
          if ( v238 )
            *v238 = &unk_4A13938;
          v640[0].m128i_i64[0] = (__int64)v238;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<verify>", 18) )
        {
          v237 = (_QWORD *)sub_22077B0(0x10u);
          if ( v237 )
            *v237 = &unk_4A13978;
          v640[0].m128i_i64[0] = (__int64)v237;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<rpa>", 12) )
        {
          v236 = (_QWORD *)sub_22077B0(0x10u);
          if ( v236 )
            *v236 = &unk_4A139B8;
          v640[0].m128i_i64[0] = (__int64)v236;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<rpa>", 15) )
        {
          v235 = (_QWORD *)sub_22077B0(0x10u);
          if ( v235 )
            *v235 = &unk_4A139F8;
          v640[0].m128i_i64[0] = (__int64)v235;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<merge-sets>", 19) )
        {
          v234 = (_QWORD *)sub_22077B0(0x10u);
          if ( v234 )
            *v234 = &unk_4A13A38;
          v640[0].m128i_i64[0] = (__int64)v234;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<merge-sets>", 22) )
        {
          v233 = (_QWORD *)sub_22077B0(0x10u);
          if ( v233 )
            *v233 = &unk_4A13A78;
          v640[0].m128i_i64[0] = (__int64)v233;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<basic-aa>", 17) )
        {
          v232 = (_QWORD *)sub_22077B0(0x10u);
          if ( v232 )
            *v232 = &unk_4A13AB8;
          v640[0].m128i_i64[0] = (__int64)v232;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<basic-aa>", 20) )
        {
          v231 = (_QWORD *)sub_22077B0(0x10u);
          if ( v231 )
            *v231 = &unk_4A13AF8;
          v640[0].m128i_i64[0] = (__int64)v231;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<objc-arc-aa>", 20) )
        {
          v230 = (_QWORD *)sub_22077B0(0x10u);
          if ( v230 )
            *v230 = &unk_4A13B38;
          v640[0].m128i_i64[0] = (__int64)v230;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<objc-arc-aa>", 23) )
        {
          v229 = (_QWORD *)sub_22077B0(0x10u);
          if ( v229 )
            *v229 = &unk_4A13B78;
          v640[0].m128i_i64[0] = (__int64)v229;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<scev-aa>", 16) )
        {
          v228 = (_QWORD *)sub_22077B0(0x10u);
          if ( v228 )
            *v228 = &unk_4A13BB8;
          v640[0].m128i_i64[0] = (__int64)v228;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<scev-aa>", 19) )
        {
          v227 = (_QWORD *)sub_22077B0(0x10u);
          if ( v227 )
            *v227 = &unk_4A13BF8;
          v640[0].m128i_i64[0] = (__int64)v227;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<scoped-noalias-aa>", 26) )
        {
          v226 = (_QWORD *)sub_22077B0(0x10u);
          if ( v226 )
            *v226 = &unk_4A13C38;
          v640[0].m128i_i64[0] = (__int64)v226;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<scoped-noalias-aa>", 29) )
        {
          v225 = (_QWORD *)sub_22077B0(0x10u);
          if ( v225 )
            *v225 = &unk_4A13C78;
          v640[0].m128i_i64[0] = (__int64)v225;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "require<tbaa>", 13) )
        {
          v224 = (_QWORD *)sub_22077B0(0x10u);
          if ( v224 )
            *v224 = &unk_4A13CB8;
          v640[0].m128i_i64[0] = (__int64)v224;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else if ( sub_9691B0(v110, v111, "invalidate<tbaa>", 16) )
        {
          v223 = (_QWORD *)sub_22077B0(0x10u);
          if ( v223 )
            *v223 = &unk_4A13CF8;
          v640[0].m128i_i64[0] = (__int64)v223;
          sub_2353900(a3, (unsigned __int64 *)v640);
          if ( v640[0].m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
        }
        else
        {
          if ( sub_9691B0(v110, v111, "aa-eval", 7) )
          {
            v221 = v640;
            for ( i = 18; i; --i )
            {
              v221->m128i_i32[0] = 0;
              v221 = (__m128i *)((char *)v221 + 4);
            }
            sub_2354830(a3, v640[0].m128i_i64);
            sub_309FA40(v640);
            v640[0].m128i_i64[0] = 0;
            *a1 = 1;
            sub_9C66B0(v640[0].m128i_i64);
            return a1;
          }
          if ( sub_9691B0(v110, v111, "adce", 4) )
          {
            v220 = (_QWORD *)sub_22077B0(0x10u);
            if ( v220 )
              *v220 = &unk_4A0ED38;
            v640[0].m128i_i64[0] = (__int64)v220;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "add-discriminators", 18) )
          {
            v219 = (_QWORD *)sub_22077B0(0x10u);
            if ( v219 )
              *v219 = &unk_4A0ED78;
            v640[0].m128i_i64[0] = (__int64)v219;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "aggressive-instcombine", 22) )
          {
            v218 = (_QWORD *)sub_22077B0(0x10u);
            if ( v218 )
              *v218 = &unk_4A0EDB8;
            v640[0].m128i_i64[0] = (__int64)v218;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "alignment-from-assumptions", 26) )
          {
            v217 = (_QWORD *)sub_22077B0(0x18u);
            if ( v217 )
            {
              v217[1] = 0;
              v217[2] = 0;
              *v217 = &unk_4A0EDF8;
            }
            v640[0].m128i_i64[0] = (__int64)v217;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "annotation-remarks", 18) )
          {
            v216 = (_QWORD *)sub_22077B0(0x10u);
            if ( v216 )
              *v216 = &unk_4A0EE38;
            v640[0].m128i_i64[0] = (__int64)v216;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "assume-builder", 14) )
          {
            v215 = (_QWORD *)sub_22077B0(0x10u);
            if ( v215 )
              *v215 = &unk_4A0EE78;
            v640[0].m128i_i64[0] = (__int64)v215;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "assume-simplify", 15) )
          {
            v274 = (_QWORD *)sub_22077B0(0x10u);
            if ( v274 )
              *v274 = &unk_4A0EEB8;
            v640[0].m128i_i64[0] = (__int64)v274;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "atomic-expand", 13) )
          {
            v272 = a2->m128i_i64[0];
            v273 = (_QWORD *)sub_22077B0(0x10u);
            if ( v273 )
            {
              v273[1] = v272;
              *v273 = &unk_4A0EEF8;
            }
            v640[0].m128i_i64[0] = (__int64)v273;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "bdce", 4) )
          {
            v271 = (_QWORD *)sub_22077B0(0x10u);
            if ( v271 )
              *v271 = &unk_4A0EF38;
            v640[0].m128i_i64[0] = (__int64)v271;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "break-crit-edges", 16) )
          {
            v270 = (_QWORD *)sub_22077B0(0x10u);
            if ( v270 )
              *v270 = &unk_4A0EF78;
            v640[0].m128i_i64[0] = (__int64)v270;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "callbr-prepare", 14) )
          {
            v269 = (_QWORD *)sub_22077B0(0x10u);
            if ( v269 )
              *v269 = &unk_4A0EFB8;
            v640[0].m128i_i64[0] = (__int64)v269;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else if ( sub_9691B0(v110, v111, "callsite-splitting", 18) )
          {
            v268 = (_QWORD *)sub_22077B0(0x10u);
            if ( v268 )
              *v268 = &unk_4A0EFF8;
            v640[0].m128i_i64[0] = (__int64)v268;
            sub_2353900(a3, (unsigned __int64 *)v640);
            if ( v640[0].m128i_i64[0] )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
          }
          else
          {
            if ( sub_9691B0(v110, v111, "chr", 3) )
            {
              sub_23FD580(v639);
              v267 = (_QWORD *)sub_22077B0(0x10u);
              if ( v267 )
                *v267 = &unk_4A0F038;
              v640[0].m128i_i64[0] = (__int64)v267;
              sub_2353900(a3, (unsigned __int64 *)v640);
              if ( v640[0].m128i_i64[0] )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
              goto LABEL_2039;
            }
            if ( sub_9691B0(v110, v111, "codegenprepare", 14) )
            {
              v265 = a2->m128i_i64[0];
              v266 = (_QWORD *)sub_22077B0(0x10u);
              if ( v266 )
              {
                v266[1] = v265;
                *v266 = &unk_4A0F078;
              }
              v640[0].m128i_i64[0] = (__int64)v266;
              sub_2353900(a3, (unsigned __int64 *)v640);
              if ( v640[0].m128i_i64[0] )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
            }
            else if ( sub_9691B0(v110, v111, "complex-deinterleaving", 22) )
            {
              v263 = a2->m128i_i64[0];
              v264 = (_QWORD *)sub_22077B0(0x10u);
              if ( v264 )
              {
                v264[1] = v263;
                *v264 = &unk_4A0F0B8;
              }
              v640[0].m128i_i64[0] = (__int64)v264;
              sub_2353900(a3, (unsigned __int64 *)v640);
              if ( v640[0].m128i_i64[0] )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
            }
            else
            {
              if ( sub_9691B0(v110, v111, "consthoist", 10) )
              {
                memset(v639, 0, 0x15F8u);
                sub_233ADF0((__int64)v639);
                sub_2367B50((__int64)v640, (__int64)v639, v253, v254, v255, v256);
                v257 = (_QWORD *)sub_22077B0(0x1600u);
                v262 = v257;
                if ( v257 )
                {
                  *v257 = &unk_4A0F0F8;
                  sub_2367B50((__int64)(v257 + 1), (__int64)v640, v258, v259, v260, v261);
                }
                v635.m128i_i64[0] = (__int64)v262;
                sub_2353900(a3, (unsigned __int64 *)&v635);
                if ( v635.m128i_i64[0] )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v635.m128i_i64[0] + 8LL))(v635.m128i_i64[0]);
                sub_233AEF0((__int64)v640);
                sub_233AEF0((__int64)v639);
                v640[0].m128i_i64[0] = 0;
                *a1 = 1;
                sub_9C66B0(v640[0].m128i_i64);
                return a1;
              }
              if ( sub_9691B0(v110, v111, "constraint-elimination", 22) )
              {
                v252 = (_QWORD *)sub_22077B0(0x10u);
                if ( v252 )
                  *v252 = &unk_4A0F138;
                v640[0].m128i_i64[0] = (__int64)v252;
                sub_2353900(a3, (unsigned __int64 *)v640);
                if ( v640[0].m128i_i64[0] )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
              }
              else if ( sub_9691B0(v110, v111, "coro-elide", 10) )
              {
                v251 = (_QWORD *)sub_22077B0(0x10u);
                if ( v251 )
                  *v251 = &unk_4A0F178;
                v640[0].m128i_i64[0] = (__int64)v251;
                sub_2353900(a3, (unsigned __int64 *)v640);
                if ( v640[0].m128i_i64[0] )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
              }
              else if ( sub_9691B0(v110, v111, "correlated-propagation", 22) )
              {
                v250 = (_QWORD *)sub_22077B0(0x10u);
                if ( v250 )
                  *v250 = &unk_4A0F1B8;
                v640[0].m128i_i64[0] = (__int64)v250;
                sub_2353900(a3, (unsigned __int64 *)v640);
                if ( v640[0].m128i_i64[0] )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
              }
              else
              {
                if ( sub_9691B0(v110, v111, "count-visits", 12) )
                {
                  v640[0] = 0u;
                  v640[1].m128i_i64[0] = 0x1000000000LL;
                  sub_2354710(a3, (__int64)v640);
                  sub_B72400(v640[0].m128i_i64, (__int64)v640);
                  v640[0].m128i_i64[0] = 0;
                  *a1 = 1;
                  sub_9C66B0(v640[0].m128i_i64);
                  return a1;
                }
                if ( sub_9691B0(v110, v111, "dce", 3) )
                {
                  v249 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v249 )
                    *v249 = &unk_4A0F238;
                  v640[0].m128i_i64[0] = (__int64)v249;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "declare-to-assign", 17) )
                {
                  v248 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v248 )
                    *v248 = &unk_4A0F278;
                  v640[0].m128i_i64[0] = (__int64)v248;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dfa-jump-threading", 18) )
                {
                  v282 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v282 )
                    *v282 = &unk_4A0F2B8;
                  v640[0].m128i_i64[0] = (__int64)v282;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "div-rem-pairs", 13) )
                {
                  v281 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v281 )
                    *v281 = &unk_4A0F2F8;
                  v640[0].m128i_i64[0] = (__int64)v281;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dot-cfg", 7) )
                {
                  v280 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v280 )
                    *v280 = &unk_4A0F338;
                  v640[0].m128i_i64[0] = (__int64)v280;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dot-cfg-only", 12) )
                {
                  v279 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v279 )
                    *v279 = &unk_4A0F378;
                  v640[0].m128i_i64[0] = (__int64)v279;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dot-dom", 7) )
                {
                  v278 = (_QWORD *)sub_22077B0(0x20u);
                  if ( v278 )
                  {
                    v278[3] = 3;
                    v278[2] = "dom";
                    *v278 = &unk_4A0F3B8;
                    v278[1] = &unk_4A0B588;
                  }
                  v640[0].m128i_i64[0] = (__int64)v278;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dot-dom-only", 12) )
                {
                  v277 = (_QWORD *)sub_22077B0(0x20u);
                  if ( v277 )
                  {
                    v277[3] = 7;
                    v277[2] = "domonly";
                    *v277 = &unk_4A0F3F8;
                    v277[1] = &unk_4A0B5A0;
                  }
                  v640[0].m128i_i64[0] = (__int64)v277;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dot-post-dom", 12) )
                {
                  v276 = (_QWORD *)sub_22077B0(0x20u);
                  if ( v276 )
                  {
                    v276[3] = 7;
                    v276[2] = "postdom";
                    *v276 = &unk_4A0F438;
                    v276[1] = &unk_4A0B5B8;
                  }
                  v640[0].m128i_i64[0] = (__int64)v276;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dot-post-dom-only", 17) )
                {
                  v275 = (_QWORD *)sub_22077B0(0x20u);
                  if ( v275 )
                  {
                    v275[3] = 11;
                    v275[2] = "postdomonly";
                    *v275 = &unk_4A0F478;
                    v275[1] = &unk_4A0B5D0;
                  }
                  v640[0].m128i_i64[0] = (__int64)v275;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dse", 3) )
                {
                  v289 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v289 )
                    *v289 = &unk_4A0F4B8;
                  v640[0].m128i_i64[0] = (__int64)v289;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "dwarf-eh-prepare", 16) )
                {
                  v287 = a2->m128i_i64[0];
                  v288 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v288 )
                  {
                    v288[1] = v287;
                    *v288 = &unk_4A0F4F8;
                  }
                  v640[0].m128i_i64[0] = (__int64)v288;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "expand-large-div-rem", 20) )
                {
                  v285 = a2->m128i_i64[0];
                  v286 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v286 )
                  {
                    v286[1] = v285;
                    *v286 = &unk_4A0F538;
                  }
                  v640[0].m128i_i64[0] = (__int64)v286;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "expand-large-fp-convert", 23) )
                {
                  v283 = a2->m128i_i64[0];
                  v284 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v284 )
                  {
                    v284[1] = v283;
                    *v284 = &unk_4A0F578;
                  }
                  v640[0].m128i_i64[0] = (__int64)v284;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else if ( sub_9691B0(v110, v111, "expand-memcmp", 13) )
                {
                  v294 = a2->m128i_i64[0];
                  v295 = (_QWORD *)sub_22077B0(0x10u);
                  if ( v295 )
                  {
                    v295[1] = v294;
                    *v295 = &unk_4A0F5B8;
                  }
                  v640[0].m128i_i64[0] = (__int64)v295;
                  sub_2353900(a3, (unsigned __int64 *)v640);
                  if ( v640[0].m128i_i64[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                }
                else
                {
                  if ( sub_9691B0(v110, v111, "extra-vector-passes", 19) )
                  {
                    memset(v639, 0, 40);
                    memset(v640, 0, 40);
                    v290 = (_QWORD *)sub_22077B0(0x30u);
                    if ( v290 )
                    {
                      v290[4] = 0;
                      v290[5] = 0;
                      *v290 = &unk_4A0F5F8;
                      v291 = v640[0].m128i_i64[0];
                      v640[0].m128i_i64[0] = 0;
                      v290[1] = v291;
                      v292 = v640[0].m128i_i64[1];
                      v640[0].m128i_i64[1] = 0;
                      v290[2] = v292;
                      v293 = v640[1].m128i_i64[0];
                      v640[1].m128i_i64[0] = 0;
                      v290[3] = v293;
                    }
                    v635.m128i_i64[0] = (__int64)v290;
                    sub_2353900(a3, (unsigned __int64 *)&v635);
                    if ( v635.m128i_i64[0] )
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v635.m128i_i64[0] + 8LL))(v635.m128i_i64[0]);
                    sub_233F7F0((__int64)v640);
                    sub_233F7F0((__int64)v639);
                    v640[0].m128i_i64[0] = 0;
                    *a1 = 1;
                    sub_9C66B0(v640[0].m128i_i64);
                    return a1;
                  }
                  if ( sub_9691B0(v110, v111, "fix-irreducible", 15) )
                  {
                    v214 = (_QWORD *)sub_22077B0(0x10u);
                    if ( v214 )
                      *v214 = &unk_4A0F638;
                    v640[0].m128i_i64[0] = (__int64)v214;
                    sub_2353900(a3, (unsigned __int64 *)v640);
                    if ( v640[0].m128i_i64[0] )
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                  }
                  else if ( sub_9691B0(v110, v111, "flatten-cfg", 11) )
                  {
                    v213 = (_QWORD *)sub_22077B0(0x10u);
                    if ( v213 )
                      *v213 = &unk_4A0F678;
                    v640[0].m128i_i64[0] = (__int64)v213;
                    sub_2353900(a3, (unsigned __int64 *)v640);
                    if ( v640[0].m128i_i64[0] )
                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                  }
                  else
                  {
                    if ( sub_9691B0(v110, v111, "float2int", 9) )
                    {
                      memset(v640, 0, 0x108u);
                      sub_233B280((__int64)v640);
                      sub_2366420(a3, (__int64)v640, v385, v386, v387, v388);
                      sub_233B360((__int64)v640);
                      v640[0].m128i_i64[0] = 0;
                      *a1 = 1;
                      sub_9C66B0(v640[0].m128i_i64);
                      return a1;
                    }
                    if ( sub_9691B0(v110, v111, "gc-lowering", 11) )
                    {
                      v384 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v384 )
                        *v384 = &unk_4A0F6F8;
                      v640[0].m128i_i64[0] = (__int64)v384;
                      sub_2353900(a3, (unsigned __int64 *)v640);
                      if ( v640[0].m128i_i64[0] )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                    }
                    else if ( sub_9691B0(v110, v111, "guard-widening", 14) )
                    {
                      v383 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v383 )
                        *v383 = &unk_4A0F738;
                      v640[0].m128i_i64[0] = (__int64)v383;
                      sub_2353900(a3, (unsigned __int64 *)v640);
                      if ( v640[0].m128i_i64[0] )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                    }
                    else if ( sub_9691B0(v110, v111, "gvn-hoist", 9) )
                    {
                      v382 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v382 )
                        *v382 = &unk_4A0F778;
                      v640[0].m128i_i64[0] = (__int64)v382;
                      sub_2353900(a3, (unsigned __int64 *)v640);
                      if ( v640[0].m128i_i64[0] )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                    }
                    else if ( sub_9691B0(v110, v111, "gvn-sink", 8) )
                    {
                      v381 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v381 )
                        *v381 = &unk_4A0F7B8;
                      v640[0].m128i_i64[0] = (__int64)v381;
                      sub_2353900(a3, (unsigned __int64 *)v640);
                      if ( v640[0].m128i_i64[0] )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                    }
                    else if ( sub_9691B0(v110, v111, "helloworld", 10) )
                    {
                      v380 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v380 )
                        *v380 = &unk_4A0F7F8;
                      v640[0].m128i_i64[0] = (__int64)v380;
                      sub_2353900(a3, (unsigned __int64 *)v640);
                      if ( v640[0].m128i_i64[0] )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                    }
                    else if ( sub_9691B0(v110, v111, "indirectbr-expand", 17) )
                    {
                      v378 = a2->m128i_i64[0];
                      v379 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v379 )
                      {
                        v379[1] = v378;
                        *v379 = &unk_4A0F838;
                      }
                      v640[0].m128i_i64[0] = (__int64)v379;
                      sub_2353900(a3, (unsigned __int64 *)v640);
                      if ( v640[0].m128i_i64[0] )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                    }
                    else
                    {
                      if ( sub_9691B0(v110, v111, "infer-address-spaces", 20) )
                      {
                        sub_27D05A0(v639);
                        v376 = v639[0].m128i_i32[0];
                        v377 = sub_22077B0(0x10u);
                        if ( v377 )
                        {
                          *(_DWORD *)(v377 + 8) = v376;
                          *(_QWORD *)v377 = &unk_4A0F878;
                        }
                        v640[0].m128i_i64[0] = v377;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        goto LABEL_2039;
                      }
                      if ( sub_9691B0(v110, v111, "infer-alignment", 15) )
                      {
                        v375 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v375 )
                          *v375 = &unk_4A0F8B8;
                        v640[0].m128i_i64[0] = (__int64)v375;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "inject-tli-mappings", 19) )
                      {
                        v374 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v374 )
                          *v374 = &unk_4A0F8F8;
                        v640[0].m128i_i64[0] = (__int64)v374;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "instcount", 9) )
                      {
                        v373 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v373 )
                          *v373 = &unk_4A0F938;
                        v640[0].m128i_i64[0] = (__int64)v373;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "instnamer", 9) )
                      {
                        v372 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v372 )
                          *v372 = &unk_4A0F978;
                        v640[0].m128i_i64[0] = (__int64)v372;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "instsimplify", 12) )
                      {
                        v371 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v371 )
                          *v371 = &unk_4A0F9B8;
                        v640[0].m128i_i64[0] = (__int64)v371;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "interleaved-access", 18) )
                      {
                        v369 = a2->m128i_i64[0];
                        v370 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v370 )
                        {
                          v370[1] = v369;
                          *v370 = &unk_4A0F9F8;
                        }
                        v640[0].m128i_i64[0] = (__int64)v370;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "interleaved-load-combine", 24) )
                      {
                        v367 = a2->m128i_i64[0];
                        v368 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v368 )
                        {
                          v368[1] = v367;
                          *v368 = &unk_4A0FA38;
                        }
                        v640[0].m128i_i64[0] = (__int64)v368;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "invalidate<all>", 15) )
                      {
                        v366 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v366 )
                          *v366 = &unk_4A0FA78;
                        v640[0].m128i_i64[0] = (__int64)v366;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else if ( sub_9691B0(v110, v111, "irce", 4) )
                      {
                        v365 = (_QWORD *)sub_22077B0(0x10u);
                        if ( v365 )
                          *v365 = &unk_4A0FAB8;
                        v640[0].m128i_i64[0] = (__int64)v365;
                        sub_2353900(a3, (unsigned __int64 *)v640);
                        if ( v640[0].m128i_i64[0] )
                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                      }
                      else
                      {
                        if ( sub_9691B0(v110, v111, "jump-threading", 14) )
                        {
                          sub_27DC820(v640, 0xFFFFFFFFLL);
                          sub_2354380(a3, v640[0].m128i_i64);
                          sub_233B480((__int64)v640, (__int64)v640, v361, v362, v363, v364);
                          v640[0].m128i_i64[0] = 0;
                          *a1 = 1;
                          sub_9C66B0(v640[0].m128i_i64);
                          return a1;
                        }
                        if ( sub_9691B0(v110, v111, "jump-table-to-switch", 20) )
                        {
                          v360 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v360 )
                            *v360 = &unk_4A0FB38;
                          v640[0].m128i_i64[0] = (__int64)v360;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "kcfi", 4) )
                        {
                          v359 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v359 )
                            *v359 = &unk_4A0FB78;
                          v640[0].m128i_i64[0] = (__int64)v359;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "kernel-info", 11) )
                        {
                          v357 = a2->m128i_i64[0];
                          v358 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v358 )
                          {
                            v358[1] = v357;
                            *v358 = &unk_4A0FBB8;
                          }
                          v640[0].m128i_i64[0] = (__int64)v358;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lcssa", 5) )
                        {
                          v356 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v356 )
                            *v356 = &unk_4A0B680;
                          v640[0].m128i_i64[0] = (__int64)v356;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "libcalls-shrinkwrap", 19) )
                        {
                          v355 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v355 )
                            *v355 = &unk_4A0FBF8;
                          v640[0].m128i_i64[0] = (__int64)v355;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lint", 4) )
                        {
                          v354 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v354 )
                            *v354 = &unk_4A0FC38;
                          v640[0].m128i_i64[0] = (__int64)v354;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "load-store-vectorizer", 21) )
                        {
                          v353 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v353 )
                            *v353 = &unk_4A0FC78;
                          v640[0].m128i_i64[0] = (__int64)v353;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-data-prefetch", 18) )
                        {
                          v352 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v352 )
                            *v352 = &unk_4A0FCB8;
                          v640[0].m128i_i64[0] = (__int64)v352;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-distribute", 15) )
                        {
                          v351 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v351 )
                            *v351 = &unk_4A0FCF8;
                          v640[0].m128i_i64[0] = (__int64)v351;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-fusion", 11) )
                        {
                          v350 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v350 )
                            *v350 = &unk_4A0FD38;
                          v640[0].m128i_i64[0] = (__int64)v350;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-load-elim", 14) )
                        {
                          v349 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v349 )
                            *v349 = &unk_4A0FD78;
                          v640[0].m128i_i64[0] = (__int64)v349;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-simplify", 13) )
                        {
                          v348 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v348 )
                            *v348 = &unk_4A0B640;
                          v640[0].m128i_i64[0] = (__int64)v348;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-sink", 9) )
                        {
                          v347 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v347 )
                            *v347 = &unk_4A0FDB8;
                          v640[0].m128i_i64[0] = (__int64)v347;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "loop-versioning", 15) )
                        {
                          v346 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v346 )
                            *v346 = &unk_4A0FDF8;
                          v640[0].m128i_i64[0] = (__int64)v346;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lower-atomic", 12) )
                        {
                          v345 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v345 )
                            *v345 = &unk_4A0FE38;
                          v640[0].m128i_i64[0] = (__int64)v345;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lower-constant-intrinsics", 25) )
                        {
                          v344 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v344 )
                            *v344 = &unk_4A0FE78;
                          v640[0].m128i_i64[0] = (__int64)v344;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lower-expect", 12) )
                        {
                          v343 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v343 )
                            *v343 = &unk_4A0FEB8;
                          v640[0].m128i_i64[0] = (__int64)v343;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lower-guard-intrinsic", 21) )
                        {
                          v342 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v342 )
                            *v342 = &unk_4A0FEF8;
                          v640[0].m128i_i64[0] = (__int64)v342;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lower-invoke", 12) )
                        {
                          v341 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v341 )
                            *v341 = &unk_4A0FF38;
                          v640[0].m128i_i64[0] = (__int64)v341;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "lower-widenable-condition", 25) )
                        {
                          v340 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v340 )
                            *v340 = &unk_4A0FF78;
                          v640[0].m128i_i64[0] = (__int64)v340;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "make-guards-explicit", 20) )
                        {
                          v339 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v339 )
                            *v339 = &unk_4A0FFB8;
                          v640[0].m128i_i64[0] = (__int64)v339;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "mem2reg", 7) )
                        {
                          v338 = (_QWORD *)sub_22077B0(0x10u);
                          if ( v338 )
                            *v338 = &unk_4A0FFF8;
                          v640[0].m128i_i64[0] = (__int64)v338;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else if ( sub_9691B0(v110, v111, "memcpyopt", 9) )
                        {
                          v337 = (_QWORD *)sub_22077B0(0x48u);
                          if ( v337 )
                          {
                            v337[1] = 0;
                            v337[2] = 0;
                            v337[3] = 0;
                            *v337 = &unk_4A10038;
                            v337[4] = 0;
                            v337[5] = 0;
                            v337[6] = 0;
                            v337[7] = 0;
                            v337[8] = 0;
                          }
                          v640[0].m128i_i64[0] = (__int64)v337;
                          sub_2353900(a3, (unsigned __int64 *)v640);
                          if ( v640[0].m128i_i64[0] )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                        }
                        else
                        {
                          if ( sub_9691B0(v110, v111, "memprof", 7) )
                          {
                            nullsub_1510(v639);
                            v336 = (_QWORD *)sub_22077B0(0x10u);
                            if ( v336 )
                              *v336 = &unk_4A10078;
                            v640[0].m128i_i64[0] = (__int64)v336;
                            sub_2353900(a3, (unsigned __int64 *)v640);
                            if ( v640[0].m128i_i64[0] )
                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            goto LABEL_2039;
                          }
                          if ( sub_9691B0(v110, v111, "mergeicmps", 10) )
                          {
                            v335 = (_QWORD *)sub_22077B0(0x10u);
                            if ( v335 )
                              *v335 = &unk_4A100B8;
                            v640[0].m128i_i64[0] = (__int64)v335;
                            sub_2353900(a3, (unsigned __int64 *)v640);
                            if ( v640[0].m128i_i64[0] )
                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                          }
                          else if ( sub_9691B0(v110, v111, "mergereturn", 11) )
                          {
                            v334 = sub_22077B0(0x10u);
                            if ( v334 )
                            {
                              *(_BYTE *)(v334 + 8) = 0;
                              *(_QWORD *)v334 = &unk_4A100F8;
                            }
                            v640[0].m128i_i64[0] = v334;
                            sub_2353900(a3, (unsigned __int64 *)v640);
                            if ( v640[0].m128i_i64[0] )
                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                          }
                          else if ( sub_9691B0(v110, v111, "move-auto-init", 14) )
                          {
                            v333 = (_QWORD *)sub_22077B0(0x10u);
                            if ( v333 )
                              *v333 = &unk_4A10138;
                            v640[0].m128i_i64[0] = (__int64)v333;
                            sub_2353900(a3, (unsigned __int64 *)v640);
                            if ( v640[0].m128i_i64[0] )
                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                          }
                          else
                          {
                            if ( sub_9691B0(v110, v111, "nary-reassociate", 16) )
                            {
                              v331 = v640;
                              for ( j = 20; j; --j )
                              {
                                v331->m128i_i32[0] = 0;
                                v331 = (__m128i *)((char *)v331 + 4);
                              }
                              sub_2354190(a3, v640[0].m128i_i64);
                              sub_234B9B0((__int64)v640[3].m128i_i64);
                              v640[0].m128i_i64[0] = 0;
                              *a1 = 1;
                              sub_9C66B0(v640[0].m128i_i64);
                              return a1;
                            }
                            if ( sub_9691B0(v110, v111, "newgvn", 6) )
                            {
                              v330 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v330 )
                                *v330 = &unk_4A101B8;
                              v640[0].m128i_i64[0] = (__int64)v330;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "no-op-function", 14) )
                            {
                              v329 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v329 )
                                *v329 = &unk_4A101F8;
                              v640[0].m128i_i64[0] = (__int64)v329;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "normalize", 9) )
                            {
                              v328 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v328 )
                                *v328 = &unk_4A10238;
                              v640[0].m128i_i64[0] = (__int64)v328;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "objc-arc", 8) )
                            {
                              v327 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v327 )
                                *v327 = &unk_4A10278;
                              v640[0].m128i_i64[0] = (__int64)v327;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "objc-arc-contract", 17) )
                            {
                              v326 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v326 )
                                *v326 = &unk_4A102B8;
                              v640[0].m128i_i64[0] = (__int64)v326;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "objc-arc-expand", 15) )
                            {
                              v325 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v325 )
                                *v325 = &unk_4A102F8;
                              v640[0].m128i_i64[0] = (__int64)v325;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "pa-eval", 7) )
                            {
                              v324 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v324 )
                                *v324 = &unk_4A10338;
                              v640[0].m128i_i64[0] = (__int64)v324;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "partially-inline-libcalls", 25) )
                            {
                              v323 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v323 )
                                *v323 = &unk_4A10378;
                              v640[0].m128i_i64[0] = (__int64)v323;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "pgo-memop-opt", 13) )
                            {
                              v322 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v322 )
                                *v322 = &unk_4A103B8;
                              v640[0].m128i_i64[0] = (__int64)v322;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else if ( sub_9691B0(v110, v111, "place-safepoints", 16) )
                            {
                              v321 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v321 )
                                *v321 = &unk_4A103F8;
                              v640[0].m128i_i64[0] = (__int64)v321;
                              sub_2353900(a3, (unsigned __int64 *)v640);
                              if ( v640[0].m128i_i64[0] )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                            }
                            else
                            {
                              if ( sub_9691B0(v110, v111, "print", 5) )
                              {
                                sub_230B630(v639[0].m128i_i64, byte_3F871B3);
                                v320 = sub_CB72A0();
                                sub_22F3020(v640, (__int64)v320, (unsigned __int8 **)v639);
                                sub_2354080(a3, (__int64)v640);
                                sub_2240A30(&v640[0].m128i_u64[1]);
                                sub_2240A30((unsigned __int64 *)v639);
                                v640[0].m128i_i64[0] = 0;
                                *a1 = 1;
                                sub_9C66B0(v640[0].m128i_i64);
                                return a1;
                              }
                              if ( sub_9691B0(v110, v111, "print-alias-sets", 16) )
                              {
                                v317 = sub_CB72A0();
                                sub_FD70F0(v639, (__int64)v317);
                                v318 = v639[0].m128i_i64[0];
                                v319 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v319 )
                                {
                                  v319[1] = v318;
                                  *v319 = &unk_4A10478;
                                }
                                v640[0].m128i_i64[0] = (__int64)v319;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                goto LABEL_2039;
                              }
                              if ( sub_9691B0(v110, v111, "print-cfg-sccs", 14) )
                              {
                                v315 = sub_CB72A0();
                                v316 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v316 )
                                {
                                  v316[1] = v315;
                                  *v316 = &unk_4A104B8;
                                }
                                v640[0].m128i_i64[0] = (__int64)v316;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print-memderefs", 15) )
                              {
                                v313 = sub_CB72A0();
                                v314 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v314 )
                                {
                                  v314[1] = v313;
                                  *v314 = &unk_4A104F8;
                                }
                                v640[0].m128i_i64[0] = (__int64)v314;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print-mustexecute", 17) )
                              {
                                v311 = sub_CB72A0();
                                v312 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v312 )
                                {
                                  v312[1] = v311;
                                  *v312 = &unk_4A10538;
                                }
                                v640[0].m128i_i64[0] = (__int64)v312;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print-predicateinfo", 19) )
                              {
                                v309 = sub_CB72A0();
                                v310 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v310 )
                                {
                                  v310[1] = v309;
                                  *v310 = &unk_4A10578;
                                }
                                v640[0].m128i_i64[0] = (__int64)v310;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print<access-info>", 18) )
                              {
                                v307 = sub_CB72A0();
                                v308 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v308 )
                                {
                                  v308[1] = v307;
                                  *v308 = &unk_4A105B8;
                                }
                                v640[0].m128i_i64[0] = (__int64)v308;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print<assumptions>", 18) )
                              {
                                v305 = sub_CB72A0();
                                v306 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v306 )
                                {
                                  v306[1] = v305;
                                  *v306 = &unk_4A105F8;
                                }
                                v640[0].m128i_i64[0] = (__int64)v306;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print<block-freq>", 17) )
                              {
                                v448 = sub_CB72A0();
                                v449 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v449 )
                                {
                                  v449[1] = v448;
                                  *v449 = &unk_4A10638;
                                }
                                v640[0].m128i_i64[0] = (__int64)v449;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print<branch-prob>", 18) )
                              {
                                v446 = sub_CB72A0();
                                v447 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v447 )
                                {
                                  v447[1] = v446;
                                  *v447 = &unk_4A10678;
                                }
                                v640[0].m128i_i64[0] = (__int64)v447;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else if ( sub_9691B0(v110, v111, "print<cost-model>", 17) )
                              {
                                v444 = sub_CB72A0();
                                v445 = (_QWORD *)sub_22077B0(0x10u);
                                if ( v445 )
                                {
                                  v445[1] = v444;
                                  *v445 = &unk_4A106B8;
                                }
                                v640[0].m128i_i64[0] = (__int64)v445;
                                sub_2353900(a3, (unsigned __int64 *)v640);
                                if ( v640[0].m128i_i64[0] )
                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                              }
                              else
                              {
                                if ( sub_9691B0(v110, v111, "print<cycles>", 13) )
                                {
                                  v441 = sub_CB72A0();
                                  sub_11FC3A0(v639, (__int64)v441);
                                  v442 = v639[0].m128i_i64[0];
                                  v443 = (_QWORD *)sub_22077B0(0x10u);
                                  if ( v443 )
                                  {
                                    v443[1] = v442;
                                    *v443 = &unk_4A106F8;
                                  }
                                  v640[0].m128i_i64[0] = (__int64)v443;
                                  sub_2353900(a3, (unsigned __int64 *)v640);
                                  if ( v640[0].m128i_i64[0] )
                                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                  goto LABEL_2039;
                                }
                                if ( sub_9691B0(v110, v111, "print<da>", 9) )
                                {
                                  v640[0].m128i_i8[8] = 0;
                                  v640[0].m128i_i64[0] = (__int64)sub_CB72A0();
                                  sub_2353B90(a3, v640[0].m128i_i64);
                                  v640[0].m128i_i64[0] = 0;
                                  *a1 = 1;
                                  sub_9C66B0(v640[0].m128i_i64);
                                  return a1;
                                }
                                if ( sub_9691B0(v110, v111, "print<debug-ata>", 16) )
                                {
                                  v439 = sub_CB72A0();
                                  v440 = (_QWORD *)sub_22077B0(0x10u);
                                  if ( v440 )
                                  {
                                    v440[1] = v439;
                                    *v440 = &unk_4A10778;
                                  }
                                  v640[0].m128i_i64[0] = (__int64)v440;
                                  sub_2353900(a3, (unsigned __int64 *)v640);
                                  if ( v640[0].m128i_i64[0] )
                                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                }
                                else
                                {
                                  if ( sub_9691B0(v110, v111, "print<delinearization>", 22) )
                                  {
                                    v436 = sub_CB72A0();
                                    sub_30B8A00(v639, v436);
                                    v437 = v639[0].m128i_i64[0];
                                    v438 = (_QWORD *)sub_22077B0(0x10u);
                                    if ( v438 )
                                    {
                                      v438[1] = v437;
                                      *v438 = &unk_4A107B8;
                                    }
                                    v640[0].m128i_i64[0] = (__int64)v438;
                                    sub_2353900(a3, (unsigned __int64 *)v640);
                                    if ( v640[0].m128i_i64[0] )
                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    goto LABEL_2039;
                                  }
                                  if ( sub_9691B0(v110, v111, "print<demanded-bits>", 20) )
                                  {
                                    v434 = sub_CB72A0();
                                    v435 = (_QWORD *)sub_22077B0(0x10u);
                                    if ( v435 )
                                    {
                                      v435[1] = v434;
                                      *v435 = &unk_4A107F8;
                                    }
                                    v640[0].m128i_i64[0] = (__int64)v435;
                                    sub_2353900(a3, (unsigned __int64 *)v640);
                                    if ( v640[0].m128i_i64[0] )
                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                  }
                                  else
                                  {
                                    if ( sub_9691B0(v110, v111, "print<domfrontier>", 18) )
                                    {
                                      v431 = sub_CB72A0();
                                      sub_22A4600(v639, (__int64)v431);
                                      v432 = v639[0].m128i_i64[0];
                                      v433 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v433 )
                                      {
                                        v433[1] = v432;
                                        *v433 = &unk_4A10838;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v433;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                      goto LABEL_2039;
                                    }
                                    if ( sub_9691B0(v110, v111, "print<domtree>", 14) )
                                    {
                                      v428 = sub_CB72A0();
                                      sub_B1A240(v639, (__int64)v428);
                                      v429 = v639[0].m128i_i64[0];
                                      v430 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v430 )
                                      {
                                        v430[1] = v429;
                                        *v430 = &unk_4A10878;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v430;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                      goto LABEL_2039;
                                    }
                                    if ( sub_9691B0(v110, v111, "print<func-properties>", 22) )
                                    {
                                      v426 = sub_CB72A0();
                                      v427 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v427 )
                                      {
                                        v427[1] = v426;
                                        *v427 = &unk_4A108B8;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v427;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else if ( sub_9691B0(v110, v111, "print<inline-cost>", 18) )
                                    {
                                      v424 = sub_CB72A0();
                                      v425 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v425 )
                                      {
                                        v425[1] = v424;
                                        *v425 = &unk_4A108F8;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v425;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else if ( sub_9691B0(v110, v111, "print<inliner-size-estimator>", 29) )
                                    {
                                      v422 = sub_CB72A0();
                                      v423 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v423 )
                                      {
                                        v423[1] = v422;
                                        *v423 = &unk_4A10938;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v423;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else if ( sub_9691B0(v110, v111, "print<lazy-value-info>", 22) )
                                    {
                                      v420 = sub_CB72A0();
                                      v421 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v421 )
                                      {
                                        v421[1] = v420;
                                        *v421 = &unk_4A10978;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v421;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else if ( sub_9691B0(v110, v111, "print<loops>", 12) )
                                    {
                                      v418 = sub_CB72A0();
                                      v419 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v419 )
                                      {
                                        v419[1] = v418;
                                        *v419 = &unk_4A109B8;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v419;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else if ( sub_9691B0(v110, v111, "print<memoryssa-walker>", 23) )
                                    {
                                      v416 = sub_CB72A0();
                                      v417 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v417 )
                                      {
                                        v417[1] = v416;
                                        *v417 = &unk_4A109F8;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v417;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else if ( sub_9691B0(v110, v111, "print<phi-values>", 17) )
                                    {
                                      v414 = sub_CB72A0();
                                      v415 = (_QWORD *)sub_22077B0(0x10u);
                                      if ( v415 )
                                      {
                                        v415[1] = v414;
                                        *v415 = &unk_4A10A38;
                                      }
                                      v640[0].m128i_i64[0] = (__int64)v415;
                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                      if ( v640[0].m128i_i64[0] )
                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                    }
                                    else
                                    {
                                      if ( sub_9691B0(v110, v111, "print<postdomtree>", 18) )
                                      {
                                        v411 = sub_CB72A0();
                                        sub_104C7E0(v639, (__int64)v411);
                                        v412 = v639[0].m128i_i64[0];
                                        v413 = (_QWORD *)sub_22077B0(0x10u);
                                        if ( v413 )
                                        {
                                          v413[1] = v412;
                                          *v413 = &unk_4A10A78;
                                        }
                                        v640[0].m128i_i64[0] = (__int64)v413;
                                        sub_2353900(a3, (unsigned __int64 *)v640);
                                        if ( v640[0].m128i_i64[0] )
                                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                        goto LABEL_2039;
                                      }
                                      if ( sub_9691B0(v110, v111, "print<regions>", 14) )
                                      {
                                        v408 = sub_CB72A0();
                                        sub_22DC510(v639, (__int64)v408);
                                        v409 = v639[0].m128i_i64[0];
                                        v410 = (_QWORD *)sub_22077B0(0x10u);
                                        if ( v410 )
                                        {
                                          v410[1] = v409;
                                          *v410 = &unk_4A10AB8;
                                        }
                                        v640[0].m128i_i64[0] = (__int64)v410;
                                        sub_2353900(a3, (unsigned __int64 *)v640);
                                        if ( v640[0].m128i_i64[0] )
                                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                        goto LABEL_2039;
                                      }
                                      if ( sub_9691B0(v110, v111, "print<scalar-evolution>", 23) )
                                      {
                                        v406 = sub_CB72A0();
                                        v407 = (_QWORD *)sub_22077B0(0x10u);
                                        if ( v407 )
                                        {
                                          v407[1] = v406;
                                          *v407 = &unk_4A10AF8;
                                        }
                                        v640[0].m128i_i64[0] = (__int64)v407;
                                        sub_2353900(a3, (unsigned __int64 *)v640);
                                        if ( v640[0].m128i_i64[0] )
                                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                      }
                                      else if ( sub_9691B0(v110, v111, "print<stack-safety-local>", 25) )
                                      {
                                        v404 = sub_CB72A0();
                                        v405 = (_QWORD *)sub_22077B0(0x10u);
                                        if ( v405 )
                                        {
                                          v405[1] = v404;
                                          *v405 = &unk_4A10B38;
                                        }
                                        v640[0].m128i_i64[0] = (__int64)v405;
                                        sub_2353900(a3, (unsigned __int64 *)v640);
                                        if ( v640[0].m128i_i64[0] )
                                          (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                      }
                                      else
                                      {
                                        if ( sub_9691B0(v110, v111, "print<uniformity>", 17) )
                                        {
                                          v401 = sub_CB72A0();
                                          sub_10564D0(v639, (__int64)v401);
                                          v402 = v639[0].m128i_i64[0];
                                          v403 = (_QWORD *)sub_22077B0(0x10u);
                                          if ( v403 )
                                          {
                                            v403[1] = v402;
                                            *v403 = &unk_4A10B78;
                                          }
                                          v640[0].m128i_i64[0] = (__int64)v403;
                                          sub_2353900(a3, (unsigned __int64 *)v640);
                                          if ( v640[0].m128i_i64[0] )
                                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                          goto LABEL_2039;
                                        }
                                        if ( sub_9691B0(v110, v111, "reassociate", 11) )
                                        {
                                          memset(v639, 0, 0x2F8u);
                                          sub_23503A0(v639[0].m128i_i64);
                                          sub_23504B0((__int64)v640, v639);
                                          v399 = (_QWORD *)sub_22077B0(0x300u);
                                          v400 = v399;
                                          if ( v399 )
                                          {
                                            *v399 = &unk_4A10BB8;
                                            sub_23504B0((__int64)(v399 + 1), v640);
                                          }
                                          v635.m128i_i64[0] = (__int64)v400;
                                          sub_2353900(a3, (unsigned __int64 *)&v635);
                                          if ( v635.m128i_i64[0] )
                                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v635.m128i_i64[0] + 8LL))(v635.m128i_i64[0]);
                                          sub_233B610((__int64)v640);
                                          sub_233B610((__int64)v639);
                                          v640[0].m128i_i64[0] = 0;
                                          *a1 = 1;
                                          sub_9C66B0(v640[0].m128i_i64);
                                          return a1;
                                        }
                                        if ( sub_9691B0(v110, v111, "redundant-dbg-inst-elim", 23) )
                                        {
                                          v398 = (_QWORD *)sub_22077B0(0x10u);
                                          if ( v398 )
                                            *v398 = &unk_4A10BF8;
                                          v640[0].m128i_i64[0] = (__int64)v398;
                                          sub_2353900(a3, (unsigned __int64 *)v640);
                                          if ( v640[0].m128i_i64[0] )
                                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                        }
                                        else if ( sub_9691B0(v110, v111, "reg2mem", 7) )
                                        {
                                          v397 = (_QWORD *)sub_22077B0(0x10u);
                                          if ( v397 )
                                            *v397 = &unk_4A10C38;
                                          v640[0].m128i_i64[0] = (__int64)v397;
                                          sub_2353900(a3, (unsigned __int64 *)v640);
                                          if ( v640[0].m128i_i64[0] )
                                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                        }
                                        else if ( sub_9691B0(v110, v111, "safe-stack", 10) )
                                        {
                                          v395 = a2->m128i_i64[0];
                                          v396 = (_QWORD *)sub_22077B0(0x10u);
                                          if ( v396 )
                                          {
                                            v396[1] = v395;
                                            *v396 = &unk_4A10C78;
                                          }
                                          v640[0].m128i_i64[0] = (__int64)v396;
                                          sub_2353900(a3, (unsigned __int64 *)v640);
                                          if ( v640[0].m128i_i64[0] )
                                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                        }
                                        else
                                        {
                                          if ( sub_9691B0(v110, v111, "sandbox-vectorizer", 18) )
                                          {
                                            sub_2BDEB70(v639);
                                            sub_2BDEC80(v640, v639);
                                            v393 = (_QWORD *)sub_22077B0(0x90u);
                                            v394 = v393;
                                            if ( v393 )
                                            {
                                              *v393 = &unk_4A10CB8;
                                              sub_2BDEC80(v393 + 1, v640);
                                            }
                                            v635.m128i_i64[0] = (__int64)v394;
                                            sub_2353900(a3, (unsigned __int64 *)&v635);
                                            if ( v635.m128i_i64[0] )
                                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v635.m128i_i64[0] + 8LL))(v635.m128i_i64[0]);
                                            sub_2BDE430(v640);
                                            sub_2BDE430(v639);
                                            v640[0].m128i_i64[0] = 0;
                                            *a1 = 1;
                                            sub_9C66B0(v640[0].m128i_i64);
                                            return a1;
                                          }
                                          if ( sub_9691B0(v110, v111, "scalarize-masked-mem-intrin", 27) )
                                          {
                                            v392 = (_QWORD *)sub_22077B0(0x10u);
                                            if ( v392 )
                                              *v392 = &unk_4A10CF8;
                                            v640[0].m128i_i64[0] = (__int64)v392;
                                            sub_2353900(a3, (unsigned __int64 *)v640);
                                            if ( v640[0].m128i_i64[0] )
                                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                          }
                                          else if ( sub_9691B0(v110, v111, "sccp", 4) )
                                          {
                                            v391 = (_QWORD *)sub_22077B0(0x10u);
                                            if ( v391 )
                                              *v391 = &unk_4A10D38;
                                            v640[0].m128i_i64[0] = (__int64)v391;
                                            sub_2353900(a3, (unsigned __int64 *)v640);
                                            if ( v640[0].m128i_i64[0] )
                                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                          }
                                          else if ( sub_9691B0(v110, v111, "select-optimize", 15) )
                                          {
                                            v389 = a2->m128i_i64[0];
                                            v390 = (_QWORD *)sub_22077B0(0x10u);
                                            if ( v390 )
                                            {
                                              v390[1] = v389;
                                              *v390 = &unk_4A10D78;
                                            }
                                            v640[0].m128i_i64[0] = (__int64)v390;
                                            sub_2353900(a3, (unsigned __int64 *)v640);
                                            if ( v640[0].m128i_i64[0] )
                                              (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                          }
                                          else
                                          {
                                            if ( sub_9691B0(v110, v111, "separate-const-offset-from-gep", 30) )
                                            {
                                              v640[0].m128i_i8[0] = 0;
                                              sub_2353A60(a3, v640[0].m128i_i8);
                                              v640[0].m128i_i64[0] = 0;
                                              *a1 = 1;
                                              sub_9C66B0(v640[0].m128i_i64);
                                              return a1;
                                            }
                                            if ( sub_9691B0(v110, v111, "sink", 4) )
                                            {
                                              sub_2977AB0(v640, 0);
                                              sub_2354020(a3, v640[0].m128i_i8);
                                              v640[0].m128i_i64[0] = 0;
                                              *a1 = 1;
                                              sub_9C66B0(v640[0].m128i_i64);
                                              return a1;
                                            }
                                            if ( sub_9691B0(v110, v111, "sjlj-eh-prepare", 15) )
                                            {
                                              v471 = a2->m128i_i64[0];
                                              v472 = (_QWORD *)sub_22077B0(0x10u);
                                              if ( v472 )
                                              {
                                                v472[1] = v471;
                                                *v472 = &unk_4A10E38;
                                              }
                                              v640[0].m128i_i64[0] = (__int64)v472;
                                              sub_2353900(a3, (unsigned __int64 *)v640);
                                              if ( v640[0].m128i_i64[0] )
                                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                            }
                                            else
                                            {
                                              if ( sub_9691B0(v110, v111, "slp-vectorizer", 14) )
                                              {
                                                v465 = v640;
                                                for ( k = 42; k; --k )
                                                {
                                                  v465->m128i_i32[0] = 0;
                                                  v465 = (__m128i *)((char *)v465 + 4);
                                                }
                                                sub_233BB10((__int64)v640);
                                                sub_2359A40(a3, v640[0].m128i_i64, v467, v468, v469, v470);
                                                sub_233BBD0((__int64)v640);
                                                v640[0].m128i_i64[0] = 0;
                                                *a1 = 1;
                                                sub_9C66B0(v640[0].m128i_i64);
                                                return a1;
                                              }
                                              if ( sub_9691B0(v110, v111, "slsr", 4) )
                                              {
                                                v464 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v464 )
                                                  *v464 = &unk_4A10EB8;
                                                v640[0].m128i_i64[0] = (__int64)v464;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "stack-protector", 15) )
                                              {
                                                v462 = a2->m128i_i64[0];
                                                v463 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v463 )
                                                {
                                                  v463[1] = v462;
                                                  *v463 = &unk_4A10EF8;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v463;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "strip-gc-relocates", 18) )
                                              {
                                                v461 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v461 )
                                                  *v461 = &unk_4A10F38;
                                                v640[0].m128i_i64[0] = (__int64)v461;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "tailcallelim", 12) )
                                              {
                                                v460 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v460 )
                                                  *v460 = &unk_4A10F78;
                                                v640[0].m128i_i64[0] = (__int64)v460;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "transform-warning", 17) )
                                              {
                                                v459 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v459 )
                                                  *v459 = &unk_4A10FB8;
                                                v640[0].m128i_i64[0] = (__int64)v459;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "trigger-crash-function", 22) )
                                              {
                                                v458 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v458 )
                                                  *v458 = off_49D2A68;
                                                v640[0].m128i_i64[0] = (__int64)v458;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "trigger-verifier-error", 22) )
                                              {
                                                v457 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v457 )
                                                  *v457 = off_49D2AA8;
                                                v640[0].m128i_i64[0] = (__int64)v457;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "tsan", 4) )
                                              {
                                                v456 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v456 )
                                                  *v456 = &unk_4A10FF8;
                                                v640[0].m128i_i64[0] = (__int64)v456;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "unify-loop-exits", 16) )
                                              {
                                                v455 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v455 )
                                                  *v455 = &unk_4A11038;
                                                v640[0].m128i_i64[0] = (__int64)v455;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "vector-combine", 14) )
                                              {
                                                v454 = sub_22077B0(0x10u);
                                                if ( v454 )
                                                {
                                                  *(_BYTE *)(v454 + 8) = 0;
                                                  *(_QWORD *)v454 = &unk_4A11078;
                                                }
                                                v640[0].m128i_i64[0] = v454;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify", 6) )
                                              {
                                                v453 = sub_22077B0(0x10u);
                                                if ( v453 )
                                                {
                                                  *(_BYTE *)(v453 + 8) = 1;
                                                  *(_QWORD *)v453 = &unk_4A110B8;
                                                }
                                                v640[0].m128i_i64[0] = v453;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<cycles>", 14) )
                                              {
                                                v452 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v452 )
                                                  *v452 = &unk_4A110F8;
                                                v640[0].m128i_i64[0] = (__int64)v452;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<domtree>", 15) )
                                              {
                                                v451 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v451 )
                                                  *v451 = &unk_4A11138;
                                                v640[0].m128i_i64[0] = (__int64)v451;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<loops>", 13) )
                                              {
                                                v450 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v450 )
                                                  *v450 = &unk_4A11178;
                                                v640[0].m128i_i64[0] = (__int64)v450;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<memoryssa>", 17) )
                                              {
                                                v480 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v480 )
                                                  *v480 = &unk_4A111B8;
                                                v640[0].m128i_i64[0] = (__int64)v480;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<regions>", 15) )
                                              {
                                                v479 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v479 )
                                                  *v479 = &unk_4A111F8;
                                                v640[0].m128i_i64[0] = (__int64)v479;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<safepoint-ir>", 20) )
                                              {
                                                v478 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v478 )
                                                  *v478 = &unk_4A11238;
                                                v640[0].m128i_i64[0] = (__int64)v478;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "verify<scalar-evolution>", 24) )
                                              {
                                                v477 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v477 )
                                                  *v477 = &unk_4A11278;
                                                v640[0].m128i_i64[0] = (__int64)v477;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "view-cfg", 8) )
                                              {
                                                v476 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v476 )
                                                  *v476 = &unk_4A112B8;
                                                v640[0].m128i_i64[0] = (__int64)v476;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "view-cfg-only", 13) )
                                              {
                                                v475 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v475 )
                                                  *v475 = &unk_4A112F8;
                                                v640[0].m128i_i64[0] = (__int64)v475;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "view-dom", 8) )
                                              {
                                                v474 = (_QWORD *)sub_22077B0(0x20u);
                                                if ( v474 )
                                                {
                                                  v474[3] = 3;
                                                  v474[2] = "dom";
                                                  *v474 = &unk_4A11338;
                                                  v474[1] = &unk_4A0B528;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v474;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "view-dom-only", 13) )
                                              {
                                                v473 = (_QWORD *)sub_22077B0(0x20u);
                                                if ( v473 )
                                                {
                                                  v473[3] = 7;
                                                  v473[2] = "domonly";
                                                  *v473 = &unk_4A11378;
                                                  v473[1] = &unk_4A0B540;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v473;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "view-post-dom", 13) )
                                              {
                                                v484 = (_QWORD *)sub_22077B0(0x20u);
                                                if ( v484 )
                                                {
                                                  v484[3] = 7;
                                                  v484[2] = "postdom";
                                                  *v484 = &unk_4A113B8;
                                                  v484[1] = &unk_4A0B558;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v484;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "view-post-dom-only", 18) )
                                              {
                                                v483 = (_QWORD *)sub_22077B0(0x20u);
                                                if ( v483 )
                                                {
                                                  v483[3] = 11;
                                                  v483[2] = "postdomonly";
                                                  *v483 = &unk_4A113F8;
                                                  v483[1] = &unk_4A0B570;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v483;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "wasm-eh-prepare", 15) )
                                              {
                                                v482 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v482 )
                                                  *v482 = &unk_4A11438;
                                                v640[0].m128i_i64[0] = (__int64)v482;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "basic-dbe", 9) )
                                              {
                                                v481 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v481 )
                                                  *v481 = &unk_4A11478;
                                                v640[0].m128i_i64[0] = (__int64)v481;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "branch-dist", 11) )
                                              {
                                                v486 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v486 )
                                                  *v486 = &unk_4A114B8;
                                                v640[0].m128i_i64[0] = (__int64)v486;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "byval-mem2reg", 13) )
                                              {
                                                v485 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v485 )
                                                  *v485 = &unk_4A114F8;
                                                v640[0].m128i_i64[0] = (__int64)v485;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "bypass-slow-division", 20) )
                                              {
                                                v303 = a2->m128i_i64[0];
                                                v304 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v304 )
                                                {
                                                  v304[1] = v303;
                                                  *v304 = &unk_4A11538;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v304;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "normalize-gep", 13) )
                                              {
                                                v302 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v302 )
                                                  *v302 = &unk_4A11578;
                                                v640[0].m128i_i64[0] = (__int64)v302;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "nvvm-reflect-pp", 15) )
                                              {
                                                v142 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v142 )
                                                  *v142 = &unk_4A115B8;
                                                v640[0].m128i_i64[0] = (__int64)v142;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "nvvm-peephole-optimizer", 23) )
                                              {
                                                v141 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v141 )
                                                  *v141 = &unk_4A115F8;
                                                v640[0].m128i_i64[0] = (__int64)v141;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "old-load-store-vectorizer", 25) )
                                              {
                                                v140 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v140 )
                                                  *v140 = &unk_4A11638;
                                                v640[0].m128i_i64[0] = (__int64)v140;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "print<merge-sets>", 17) )
                                              {
                                                v138 = sub_C5F790((__int64)v110, v111);
                                                v139 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v139 )
                                                {
                                                  v139[1] = v138;
                                                  *v139 = &unk_4A11678;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v139;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "remat", 5) )
                                              {
                                                v145 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v145 )
                                                  *v145 = &unk_4A116B8;
                                                v640[0].m128i_i64[0] = (__int64)v145;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "print<rpa>", 10) )
                                              {
                                                v143 = sub_C5F790((__int64)v110, v111);
                                                v144 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v144 )
                                                {
                                                  v144[1] = v143;
                                                  *v144 = &unk_4A116F8;
                                                }
                                                v640[0].m128i_i64[0] = (__int64)v144;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "propagate-alignment", 19) )
                                              {
                                                v137 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v137 )
                                                  *v137 = &unk_4A11738;
                                                v640[0].m128i_i64[0] = (__int64)v137;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "reuse-local-memory", 18) )
                                              {
                                                v136 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v136 )
                                                  *v136 = &unk_4A11778;
                                                v640[0].m128i_i64[0] = (__int64)v136;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else if ( sub_9691B0(v110, v111, "set-local-array-alignment", 25) )
                                              {
                                                v132 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v132 )
                                                  *v132 = &unk_4A117B8;
                                                v640[0].m128i_i64[0] = (__int64)v132;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                              else
                                              {
                                                if ( !sub_9691B0(v110, v111, "sinking2", 8) )
                                                {
                                                  if ( sub_9691B0(v110, v111, "d2ir-scalarizer", 15) )
                                                  {
                                                    memset(&v639[1], 0, 24);
                                                    v639[0] = (__m128i)0x100000000uLL;
                                                    v635.m128i_i64[0] = 0x2E00000015LL;
                                                    v635.m128i_i64[1] = 0x290000002ALL;
                                                    v636 = 0x140000002DLL;
                                                    v637 = 0x1300000017LL;
                                                    LODWORD(nn) = 22;
                                                    sub_23665C0(
                                                      (__int64)&v639[0].m128i_i64[1],
                                                      &v635,
                                                      (_DWORD *)&nn + 1);
                                                    sub_233A070((__int64)v640, (__int64)v639);
                                                    sub_2353AC0(a3, v640[0].m128i_i32);
                                                    sub_2342640((__int64)&v640[0].m128i_i64[1]);
                                                    sub_2342640((__int64)&v639[0].m128i_i64[1]);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v110, v111, "sink<rp-aware>", 14) )
                                                  {
                                                    sub_2977AB0(v640, 1);
                                                    sub_2354020(a3, v640[0].m128i_i8);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0((char *)v110, v111, "cfguard", 7u) )
                                                  {
                                                    sub_234BDD0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2335AC0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "cfguard",
                                                      7u);
                                                    v126 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v126 )
                                                    {
                                                      v127 = v640;
                                                      sub_234BE70(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v128 = v640[0].m128i_i32[0];
                                                      v129 = sub_22077B0(0x10u);
                                                      if ( v129 )
                                                      {
                                                        *(_DWORD *)(v129 + 8) = v128;
                                                        *(_QWORD *)v129 = &unk_4A11878;
                                                      }
                                                      v639[0].m128i_i64[0] = v129;
                                                      v127 = v639;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2351D50(v640, (__int64)v127);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "early-cse",
                                                                          9u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A630,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "early-cse",
                                                      9u);
                                                    v130 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v130 )
                                                      goto LABEL_2028;
                                                    v133 = v640[0].m128i_i8[0];
                                                    v134 = sub_22077B0(0x10u);
                                                    if ( v134 )
                                                    {
                                                      *(_BYTE *)(v134 + 8) = v133;
                                                      *(_QWORD *)v134 = &unk_4A118B8;
                                                    }
                                                    v639[0].m128i_i64[0] = v134;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
LABEL_2046:
                                                    v639[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v639[0].m128i_i64);
                                                    goto LABEL_2029;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "ee-instrument",
                                                                          0xDu) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A5F0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "ee-instrument",
                                                      0xDu);
                                                    v135 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v135 )
                                                      goto LABEL_2028;
                                                    v146 = v640[0].m128i_i8[0];
                                                    v147 = sub_22077B0(0x10u);
                                                    if ( v147 )
                                                    {
                                                      *(_BYTE *)(v147 + 8) = v146;
                                                      *(_QWORD *)v147 = &unk_4A118F8;
                                                    }
                                                    v639[0].m128i_i64[0] = v147;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "function-simplification",
                                                                          0x17u) )
                                                  {
                                                    sub_234BEE0(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2335980,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "function-simplification",
                                                      0x17u);
                                                    v148 = v639[0].m128i_i8[8] & 1;
                                                    v639[0].m128i_i8[8] = (2 * (v639[0].m128i_i8[8] & 1))
                                                                        | v639[0].m128i_i8[8] & 0xFD;
                                                    if ( v148 )
                                                    {
                                                      a2 = v639;
                                                      sub_234BF80(a1, v639[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      sub_23A54A0(v640, a2, v639[0].m128i_i64[0], 0);
                                                      v165 = v640[0].m128i_i64[1];
                                                      for ( m = (__int64 *)v640[0].m128i_i64[0]; (__int64 *)v165 != m; ++m )
                                                      {
                                                        a2 = (__m128i *)a3[1];
                                                        if ( a2 == (__m128i *)a3[2] )
                                                        {
                                                          sub_2353750(a3, a2->m128i_i8, m);
                                                        }
                                                        else
                                                        {
                                                          if ( a2 )
                                                          {
                                                            a2->m128i_i64[0] = *m;
                                                            *m = 0;
                                                          }
                                                          a3[1] += 8LL;
                                                        }
                                                      }
                                                      sub_233F7F0((__int64)v640);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                    }
                                                    sub_2352750(v639, (__int64)a2);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "gvn",
                                                                          3u) )
                                                  {
                                                    sub_234BFF0(
                                                      (__int64)&v635,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2335540,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "gvn",
                                                      3u);
                                                    v167 = v636 & 1;
                                                    LOBYTE(v636) = (2 * (v636 & 1)) | v636 & 0xFD;
                                                    if ( v167 )
                                                    {
                                                      v168 = &v635;
                                                      sub_234C090(a1, v635.m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      qmemcpy(v640, &v635, 0xEu);
                                                      sub_2339E50(
                                                        (__int64)v639,
                                                        v640[0].m128i_i64[0],
                                                        v640[0].m128i_i64[1] & 0xFFFFFFFFFFFFLL);
                                                      sub_2353240(
                                                        (__int64)v640,
                                                        v639[0].m128i_i64,
                                                        v201,
                                                        v202,
                                                        v203,
                                                        v204);
                                                      v205 = (_QWORD *)sub_22077B0(0x358u);
                                                      v210 = (__int64)v205;
                                                      if ( v205 )
                                                      {
                                                        *v205 = &unk_4A11938;
                                                        sub_2353240(
                                                          (__int64)(v205 + 1),
                                                          v640[0].m128i_i64,
                                                          v206,
                                                          v207,
                                                          v208,
                                                          v209);
                                                      }
                                                      v168 = &v631;
                                                      v631.m128i_i64[0] = v210;
                                                      sub_2353900(a3, (unsigned __int64 *)&v631);
                                                      if ( v631.m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v631.m128i_i64[0]
                                                                                        + 8LL))(v631.m128i_i64[0]);
                                                      sub_2341D90((__int64)v640);
                                                      sub_2341D90((__int64)v639);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                    }
                                                    sub_2352330(&v635, (__int64)v168);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "instcombine",
                                                                          0xBu) )
                                                  {
                                                    sub_234C100(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_23350D0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "instcombine",
                                                      0xBu);
                                                    v211 = v639[1].m128i_i8[0] & 1;
                                                    v639[1].m128i_i8[0] = (2 * (v639[1].m128i_i8[0] & 1))
                                                                        | v639[1].m128i_i8[0] & 0xFD;
                                                    if ( v211 )
                                                    {
                                                      v212 = v639;
                                                      sub_234C1A0(a1, v639[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v635.m128i_i32[2] = v639[0].m128i_i32[2];
                                                      v635.m128i_i64[0] = v639[0].m128i_i64[0];
                                                      sub_F10C20(
                                                        (__int64)v640,
                                                        v639[0].m128i_i64[0],
                                                        v639[0].m128i_i32[2]);
                                                      v212 = v640;
                                                      sub_2353C90(a3, (__int64)v640, v296, v297, v298, v299);
                                                      sub_233BCC0((__int64)v640);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                    }
                                                    sub_2352120(v639, (__int64)v212);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "loop-unroll",
                                                                          0xBu) )
                                                  {
                                                    sub_234C5C0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2334C40,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "loop-unroll",
                                                      0xBu);
                                                    v300 = v640[2].m128i_i8[0] & 1;
                                                    v640[2].m128i_i8[0] = (2 * (v640[2].m128i_i8[0] & 1))
                                                                        | v640[2].m128i_i8[0] & 0xFD;
                                                    if ( v300 )
                                                    {
                                                      v301 = v640;
                                                      sub_234C660(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v487 = v640;
                                                      v488 = &v635;
                                                      for ( n = 7; n; --n )
                                                      {
                                                        v488->m128i_i32[0] = v487->m128i_i32[0];
                                                        v487 = (__m128i *)((char *)v487 + 4);
                                                        v488 = (__m128i *)((char *)v488 + 4);
                                                      }
                                                      v490 = v639;
                                                      v491 = &v635;
                                                      for ( ii = 7; ii; --ii )
                                                      {
                                                        v490->m128i_i32[0] = v491->m128i_i32[0];
                                                        v491 = (__m128i *)((char *)v491 + 4);
                                                        v490 = (__m128i *)((char *)v490 + 4);
                                                      }
                                                      v493 = &v631;
                                                      v494 = v639;
                                                      for ( jj = 7; jj; --jj )
                                                      {
                                                        v493->m128i_i32[0] = v494->m128i_i32[0];
                                                        v494 = (__m128i *)((char *)v494 + 4);
                                                        v493 = (__m128i *)((char *)v493 + 4);
                                                      }
                                                      v301 = &v631;
                                                      sub_2353C00(a3, &v631);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2351CB0(v640, (__int64)v301);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "loop-vectorize",
                                                                          0xEu) )
                                                  {
                                                    sub_234C6D0(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2334970,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "loop-vectorize",
                                                      0xEu);
                                                    v496 = v639[0].m128i_i8[8] & 1;
                                                    v639[0].m128i_i8[8] = (2 * (v639[0].m128i_i8[8] & 1))
                                                                        | v639[0].m128i_i8[8] & 0xFD;
                                                    if ( v496 )
                                                    {
                                                      v497 = v639;
                                                      sub_234C770(a1, v639[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      sub_2AB7C70(v640, v639[0].m128i_u16[0]);
                                                      v600 = v640[0].m128i_i16[0];
                                                      v602 = v640[0].m128i_i64[1];
                                                      v500 = v640[1];
                                                      v605 = v640[4];
                                                      v501 = v640[2];
                                                      v616 = v640[5];
                                                      v502 = v640[3];
                                                      v503 = sub_22077B0(0x68u);
                                                      if ( v503 )
                                                      {
                                                        *(_QWORD *)(v503 + 32) = v500.m128i_i64[1];
                                                        *(_QWORD *)(v503 + 40) = v501.m128i_i64[0];
                                                        *(_WORD *)(v503 + 8) = v600;
                                                        *(_QWORD *)v503 = &unk_4A119F8;
                                                        *(_QWORD *)(v503 + 16) = v602;
                                                        *(_QWORD *)(v503 + 24) = v500.m128i_i64[0];
                                                        *(_QWORD *)(v503 + 48) = v501.m128i_i64[1];
                                                        *(__m128i *)(v503 + 56) = v502;
                                                        *(__m128i *)(v503 + 72) = v605;
                                                        *(__m128i *)(v503 + 88) = v616;
                                                      }
                                                      v497 = &v635;
                                                      v635.m128i_i64[0] = v503;
                                                      sub_2353900(a3, (unsigned __int64 *)&v635);
                                                      if ( v635.m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v635.m128i_i64[0]
                                                                                        + 8LL))(v635.m128i_i64[0]);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                    }
                                                    sub_23521C0(v639, (__int64)v497);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "lower-allow-check",
                                                                          0x11u) )
                                                  {
                                                    sub_234C7E0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_234FBC0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "lower-allow-check",
                                                      0x11u);
                                                    v498 = v640[1].m128i_i8[8] & 1;
                                                    v640[1].m128i_i8[8] = (2 * v498) | v640[1].m128i_i8[8] & 0xFD;
                                                    if ( (_BYTE)v498 )
                                                    {
                                                      v499 = v640;
                                                      sub_234C880(a1, v640[0].m128i_i64);
LABEL_1817:
                                                      sub_2351F40(v640, (__int64)v499);
                                                      return a1;
                                                    }
                                                    sub_234C8F0(&v635, (const void **)v640, v498);
                                                    sub_2308710(v639, (__int64)&v635, v504);
                                                    v506 = v639[0].m128i_i64[1];
                                                    v505 = v639[0].m128i_i64[0];
                                                    v507 = v639[1].m128i_i64[0];
                                                    memset(v639, 0, 24);
                                                    v508 = (_QWORD *)sub_22077B0(0x20u);
                                                    if ( v508 )
                                                    {
                                                      v508[1] = v505;
                                                      v499 = &v631;
                                                      v508[2] = v506;
                                                      v508[3] = v507;
                                                      *v508 = &unk_4A11A38;
                                                      v631.m128i_i64[0] = (__int64)v508;
                                                      sub_2353900(a3, (unsigned __int64 *)&v631);
                                                      v509 = v631.m128i_i64[0];
                                                      if ( !v631.m128i_i64[0] )
                                                      {
LABEL_1829:
                                                        sub_23425C0((unsigned __int64 *)v639);
                                                        sub_23425C0((unsigned __int64 *)&v635);
                                                        v639[0].m128i_i64[0] = 0;
                                                        *a1 = 1;
                                                        sub_9C66B0(v639[0].m128i_i64);
                                                        goto LABEL_1817;
                                                      }
                                                      v505 = 0;
                                                      v507 = 0;
                                                    }
                                                    else
                                                    {
                                                      v631.m128i_i64[0] = 0;
                                                      sub_2353900(a3, (unsigned __int64 *)&v631);
                                                      v509 = v631.m128i_i64[0];
                                                      if ( !v631.m128i_i64[0] )
                                                      {
LABEL_1827:
                                                        v499 = (__m128i *)(v507 - v505);
                                                        if ( v505 )
                                                          j_j___libc_free_0(v505);
                                                        goto LABEL_1829;
                                                      }
                                                    }
                                                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v509 + 8LL))(v509);
                                                    goto LABEL_1827;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "lower-matrix-intrinsics",
                                                                          0x17u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A5B0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "lower-matrix-intrinsics",
                                                      0x17u);
                                                    v510 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v510 )
                                                      goto LABEL_2028;
                                                    v512 = v640[0].m128i_i8[0];
                                                    v513 = sub_22077B0(0x10u);
                                                    if ( v513 )
                                                    {
                                                      *(_BYTE *)(v513 + 8) = v512;
                                                      *(_QWORD *)v513 = &unk_4A11A78;
                                                    }
                                                    v639[0].m128i_i64[0] = v513;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "lower-switch",
                                                                          0xCu) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A570,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "lower-switch",
                                                      0xCu);
                                                    v511 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v511 )
                                                      goto LABEL_2028;
                                                    v514 = v640[0].m128i_i8[0];
                                                    v515 = sub_22077B0(0x10u);
                                                    if ( v515 )
                                                    {
                                                      *(_BYTE *)(v515 + 8) = v514;
                                                      *(_QWORD *)v515 = &unk_4A11AB8;
                                                    }
                                                    v639[0].m128i_i64[0] = v515;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "mldst-motion",
                                                                          0xCu) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2334730,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "mldst-motion",
                                                      0xCu);
                                                    v516 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v516 )
                                                      goto LABEL_2028;
                                                    v518 = v640[0].m128i_i8[0];
                                                    v519 = sub_22077B0(0x10u);
                                                    if ( v519 )
                                                    {
                                                      *(_BYTE *)(v519 + 8) = v518;
                                                      *(_QWORD *)v519 = &unk_4A11AF8;
                                                    }
                                                    v639[0].m128i_i64[0] = v519;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "print<da>",
                                                                          9u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A530,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "print<da>",
                                                      9u);
                                                    v517 = v639[0].m128i_i8[8] & 1;
                                                    v639[0].m128i_i8[8] = (2 * (v639[0].m128i_i8[8] & 1))
                                                                        | v639[0].m128i_i8[8] & 0xFD;
                                                    if ( !v517 )
                                                    {
                                                      v520 = v639[0].m128i_i8[0];
                                                      v640[0].m128i_i64[0] = (__int64)sub_CB72A0();
                                                      v640[0].m128i_i8[8] = v520;
                                                      sub_2353B90(a3, v640[0].m128i_i64);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                      goto LABEL_1854;
                                                    }
LABEL_1853:
                                                    sub_234B1B0(a1, v639[0].m128i_i64);
LABEL_1854:
                                                    sub_2351C10(v639);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "print<memoryssa>",
                                                                          0x10u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A4F0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "print<memoryssa>",
                                                      0x10u);
                                                    v521 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v521 )
                                                      goto LABEL_2028;
                                                    v524 = v640[0].m128i_i8[0] ^ 1;
                                                    v525 = sub_CB72A0();
                                                    v526 = sub_22077B0(0x18u);
                                                    if ( v526 )
                                                    {
                                                      *(_QWORD *)(v526 + 8) = v525;
                                                      *(_BYTE *)(v526 + 16) = v524;
                                                      *(_QWORD *)v526 = &unk_4A11B38;
                                                    }
                                                    v639[0].m128i_i64[0] = v526;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "print<stack-lifetime>",
                                                                          0x15u) )
                                                  {
                                                    sub_234C990(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2334520,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "print<stack-lifetime>",
                                                      0x15u);
                                                    v522 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v522 )
                                                    {
                                                      v523 = v640;
                                                      sub_234CA30(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v527 = v640[0].m128i_i32[0];
                                                      v528 = sub_CB72A0();
                                                      v529 = sub_22077B0(0x18u);
                                                      if ( v529 )
                                                      {
                                                        *(_DWORD *)(v529 + 8) = v527;
                                                        *(_QWORD *)(v529 + 16) = v528;
                                                        *(_QWORD *)v529 = &unk_4A11B78;
                                                      }
                                                      v639[0].m128i_i64[0] = v529;
                                                      v523 = v639;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_23525A0(v640, (__int64)v523);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "scalarizer",
                                                                          0xAu) )
                                                  {
                                                    sub_234CAA0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_23341B0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "scalarizer",
                                                      0xAu);
                                                    v530 = v640[2].m128i_i8[8] & 1;
                                                    v640[2].m128i_i8[8] = (2 * (v640[2].m128i_i8[8] & 1))
                                                                        | v640[2].m128i_i8[8] & 0xFD;
                                                    if ( v530 )
                                                    {
                                                      v531 = v640;
                                                      sub_234CB40(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      sub_233A070((__int64)&v635, (__int64)v640);
                                                      sub_233A070((__int64)v639, (__int64)&v635);
                                                      v531 = v639;
                                                      sub_2353AC0(a3, v639[0].m128i_i32);
                                                      sub_2342640((__int64)&v639[0].m128i_i64[1]);
                                                      sub_2342640((__int64)&v635.m128i_i64[1]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2352440(v640[0].m128i_i64, (__int64)v531);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "separate-const-offset-from-gep",
                                                                          0x1Eu) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A4B0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "separate-const-offset-from-gep",
                                                      0x1Eu);
                                                    v532 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( !v532 )
                                                    {
                                                      v639[0].m128i_i8[0] = v640[0].m128i_i8[0];
                                                      sub_2353A60(a3, v639[0].m128i_i8);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
LABEL_2029:
                                                      sub_2351C10(v640);
                                                      return a1;
                                                    }
                                                    goto LABEL_2028;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "simplifycfg",
                                                                          0xBu) )
                                                  {
                                                    sub_234CBB0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2333A00,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "simplifycfg",
                                                      0xBu);
                                                    v533 = v640[1].m128i_i8[8] & 1;
                                                    v640[1].m128i_i8[8] = (2 * (v640[1].m128i_i8[8] & 1))
                                                                        | v640[1].m128i_i8[8] & 0xFD;
                                                    if ( v533 )
                                                    {
                                                      v534 = v640;
                                                      sub_234CC50(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v536 = _mm_loadu_si128(v640);
                                                      v639[1].m128i_i64[0] = v640[1].m128i_i64[0];
                                                      v639[0] = v536;
                                                      sub_29744D0(&v635, v639);
                                                      v537 = v636;
                                                      v538 = v635;
                                                      v539 = sub_22077B0(0x20u);
                                                      if ( v539 )
                                                      {
                                                        *(__m128i *)(v539 + 8) = v538;
                                                        *(_QWORD *)(v539 + 24) = v537;
                                                        *(_QWORD *)v539 = &unk_4A11BB8;
                                                      }
                                                      v534 = v639;
                                                      v639[0].m128i_i64[0] = v539;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2352080(v640, (__int64)v534);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "speculative-execution",
                                                                          0x15u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A470,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "speculative-execution",
                                                      0x15u);
                                                    v535 = v639[0].m128i_i8[8] & 1;
                                                    v639[0].m128i_i8[8] = (2 * (v639[0].m128i_i8[8] & 1))
                                                                        | v639[0].m128i_i8[8] & 0xFD;
                                                    if ( !v535 )
                                                    {
                                                      sub_297B2F0(v640, v639[0].m128i_u8[0]);
                                                      v540 = v640[0].m128i_i8[0];
                                                      v541 = v640[0].m128i_i64[1];
                                                      v542 = sub_22077B0(0x18u);
                                                      if ( v542 )
                                                      {
                                                        *(_BYTE *)(v542 + 8) = v540;
                                                        *(_QWORD *)(v542 + 16) = v541;
                                                        *(_QWORD *)v542 = &unk_4A11BF8;
                                                      }
                                                      v640[0].m128i_i64[0] = v542;
                                                      sub_2353900(a3, (unsigned __int64 *)v640);
                                                      if ( v640[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0]
                                                                                        + 8LL))(v640[0].m128i_i64[0]);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                      goto LABEL_1854;
                                                    }
                                                    goto LABEL_1853;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "sroa",
                                                                          4u) )
                                                  {
                                                    sub_234CCC0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_23338A0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "sroa",
                                                      4u);
                                                    v543 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v543 )
                                                    {
                                                      v544 = v640;
                                                      sub_234CD60(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      sub_291E720(v639, v640[0].m128i_u8[0]);
                                                      v546 = v639[0].m128i_i8[0];
                                                      v547 = sub_22077B0(0x10u);
                                                      if ( v547 )
                                                      {
                                                        *(_BYTE *)(v547 + 8) = v546;
                                                        *(_QWORD *)v547 = &unk_4A11C38;
                                                      }
                                                      v544 = v639;
                                                      v639[0].m128i_i64[0] = v547;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2352500(v640, (__int64)v544);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "structurizecfg",
                                                                          0xEu) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A430,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "structurizecfg",
                                                      0xEu);
                                                    v545 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v545 )
                                                      goto LABEL_2028;
                                                    sub_298B490(v639, v640[0].m128i_u8[0]);
                                                    v548 = v639[0].m128i_i8[0];
                                                    v549 = sub_22077B0(0x10u);
                                                    if ( v549 )
                                                    {
                                                      *(_BYTE *)(v549 + 8) = v548;
                                                      *(_QWORD *)v549 = &unk_4A11C78;
                                                    }
                                                    v639[0].m128i_i64[0] = v549;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "win-eh-prepare",
                                                                          0xEu) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A3F0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "win-eh-prepare",
                                                      0xEu);
                                                    v550 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v550 )
                                                      goto LABEL_2028;
                                                    v553 = v640[0].m128i_i8[0];
                                                    v554 = sub_22077B0(0x10u);
                                                    if ( v554 )
                                                    {
                                                      *(_BYTE *)(v554 + 8) = v553;
                                                      *(_QWORD *)v554 = &unk_4A11CB8;
                                                    }
                                                    v639[0].m128i_i64[0] = v554;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "bounds-checking",
                                                                          0xFu) )
                                                  {
                                                    sub_234CDD0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2333450,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "bounds-checking",
                                                      0xFu);
                                                    v551 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v551 )
                                                    {
                                                      v552 = v640;
                                                      sub_234CE70(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v627.m128i_i32[0] = v640[0].m128i_i32[0];
                                                      v627.m128i_i16[2] = v640[0].m128i_i16[2];
                                                      v635.m128i_i32[0] = v640[0].m128i_i32[0];
                                                      v635.m128i_i16[2] = v640[0].m128i_i16[2];
                                                      v639[0].m128i_i32[0] = v640[0].m128i_i32[0];
                                                      v639[0].m128i_i16[2] = v640[0].m128i_i16[2];
                                                      v631.m128i_i32[0] = v640[0].m128i_i32[0];
                                                      v631.m128i_i16[2] = v640[0].m128i_i16[2];
                                                      v555 = sub_22077B0(0x10u);
                                                      if ( v555 )
                                                      {
                                                        *(_QWORD *)v555 = &unk_4A11CF8;
                                                        *(_DWORD *)(v555 + 8) = v635.m128i_i32[0];
                                                        *(_WORD *)(v555 + 12) = v635.m128i_i16[2];
                                                      }
                                                      v639[0].m128i_i64[0] = v555;
                                                      v552 = v639;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2352A70(v640, (__int64)v552);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "memory-space-opt",
                                                                          0x10u) )
                                                  {
                                                    sub_234CEE0(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_23331A0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "memory-space-opt",
                                                      0x10u);
                                                    v556 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v556 )
                                                    {
                                                      v557 = v640;
                                                      sub_234CF80(a1, v640[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v559 = v640[0].m128i_i16[0];
                                                      v560 = sub_22077B0(0x10u);
                                                      if ( v560 )
                                                      {
                                                        *(_WORD *)(v560 + 8) = v559;
                                                        *(_QWORD *)v560 = &unk_4A11D38;
                                                      }
                                                      v639[0].m128i_i64[0] = v560;
                                                      v557 = v639;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      v639[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v639[0].m128i_i64);
                                                    }
                                                    sub_2352640(v640, (__int64)v557);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "lower-aggr-copies",
                                                                          0x11u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A3B0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "lower-aggr-copies",
                                                      0x11u);
                                                    v558 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v558 )
                                                      goto LABEL_2028;
                                                    v561 = v640[0].m128i_i8[0];
                                                    v562 = sub_22077B0(0x10u);
                                                    if ( v562 )
                                                    {
                                                      *(_BYTE *)(v562 + 8) = v561;
                                                      *(_QWORD *)v562 = &unk_4A11D78;
                                                    }
                                                    v639[0].m128i_i64[0] = v562;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "lower-struct-args",
                                                                          0x11u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A370,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "lower-struct-args",
                                                      0x11u);
                                                    v563 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( v563 )
                                                      goto LABEL_2028;
                                                    v574 = v640[0].m128i_i8[0];
                                                    v575 = sub_22077B0(0x10u);
                                                    if ( v575 )
                                                    {
                                                      *(_BYTE *)(v575 + 8) = v574;
                                                      *(_QWORD *)v575 = &unk_4A11DB8;
                                                    }
                                                    v639[0].m128i_i64[0] = v575;
                                                    sub_2353900(a3, (unsigned __int64 *)v639);
                                                    if ( v639[0].m128i_i64[0] )
                                                      (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                      + 8LL))(v639[0].m128i_i64[0]);
                                                    goto LABEL_2046;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "process-restrict",
                                                                          0x10u) )
                                                  {
                                                    sub_234B110(
                                                      (__int64)v640,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_233A330,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "process-restrict",
                                                      0x10u);
                                                    v586 = v640[0].m128i_i8[8] & 1;
                                                    v640[0].m128i_i8[8] = (2 * (v640[0].m128i_i8[8] & 1))
                                                                        | v640[0].m128i_i8[8] & 0xFD;
                                                    if ( !v586 )
                                                    {
                                                      v592 = v640[0].m128i_i8[0];
                                                      v593 = sub_22077B0(0x10u);
                                                      if ( v593 )
                                                      {
                                                        *(_BYTE *)(v593 + 8) = v592;
                                                        *(_QWORD *)v593 = &unk_4A11DF8;
                                                      }
                                                      v639[0].m128i_i64[0] = v593;
                                                      sub_2353900(a3, (unsigned __int64 *)v639);
                                                      if ( v639[0].m128i_i64[0] )
                                                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v639[0].m128i_i64[0]
                                                                                        + 8LL))(v639[0].m128i_i64[0]);
                                                      goto LABEL_2046;
                                                    }
LABEL_2028:
                                                    sub_234B1B0(a1, v640[0].m128i_i64);
                                                    goto LABEL_2029;
                                                  }
                                                  v564 = *(void **)a4;
                                                  v565 = *(_QWORD *)(a4 + 8);
                                                  if ( sub_9691B0(*(const void **)a4, v565, "loop-flatten", 12) )
                                                  {
                                                    sub_235CF40((__int64)v640, 0, 0, 0, v566, v567);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-interchange", 16) )
                                                  {
                                                    sub_235D290((__int64)v640, 0, 0, 0, v568, v569);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-unroll-and-jam", 19) )
                                                  {
                                                    v639[0].m128i_i32[0] = 2;
                                                    sub_235D930((__int64)v640, v639[0].m128i_i32, 0, 0, 0, v570);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "no-op-loopnest", 14) )
                                                  {
                                                    sub_235D5E0((__int64)v640, 0, 0, 0, v571, v572);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "canon-freeze", 12) )
                                                  {
                                                    sub_23555C0((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "dot-ddg", 7) )
                                                  {
                                                    sub_2354E50((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "guard-widening", 14) )
                                                  {
                                                    sub_2354B20((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "extra-simple-loop-unswitch-passes", 33) )
                                                  {
                                                    v640[0].m128i_i64[0] = (__int64)v640[1].m128i_i64;
                                                    memset(&v640[4], 0, 56);
                                                    v640[0].m128i_i64[1] = 0x600000000LL;
                                                    sub_2356730((__int64)v639, (__int64)v640, 0, 0, 0, v573);
                                                    sub_2353940(a3, v639[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v639[0].m128i_i64[1]);
                                                    sub_233F7D0(v639[0].m128i_i64);
                                                    sub_2337B30((unsigned __int64 *)v640);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "indvars", 7) )
                                                  {
                                                    v639[0].m128i_i8[0] = 1;
                                                    sub_2355A00((__int64)v640, v639[0].m128i_i8, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "invalidate<all>", 15) )
                                                  {
                                                    sub_2354C30((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-bound-split", 16) )
                                                  {
                                                    sub_23553A0((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-deletion", 13) )
                                                  {
                                                    sub_2355180((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-idiom", 10) )
                                                  {
                                                    sub_23558F0((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-idiom-vectorize", 20) )
                                                  {
                                                    v639[0].m128i_i64[0] = 0x1000000000LL;
                                                    sub_2355FA0((__int64)v640, v639[0].m128i_i64, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-instsimplify", 17) )
                                                  {
                                                    sub_23557E0((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-predication", 16) )
                                                  {
                                                    sub_2354D40((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-reduce", 11) )
                                                  {
                                                    sub_23554B0((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-term-fold", 14) )
                                                  {
                                                    sub_2354F60((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-simplifycfg", 16) )
                                                  {
                                                    sub_23556D0((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-unroll-full", 16) )
                                                  {
                                                    v639[0].m128i_i32[0] = 2;
                                                    v639[0].m128i_i16[2] = 0;
                                                    sub_2356300((__int64)v640, v639[0].m128i_i32, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-versioning-licm", 20) )
                                                  {
                                                    sub_2355070((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "no-op-loop", 10) )
                                                  {
                                                    sub_2355290((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "print", 5) )
                                                  {
                                                    sub_230B630(v635.m128i_i64, byte_3F871B3);
                                                    v584 = sub_CB72A0();
                                                    sub_283D800(v639, v584, &v635);
                                                    sub_2356560((__int64)v640, (__int64)v639, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    sub_2240A30(&v639[0].m128i_u64[1]);
                                                    sub_2240A30((unsigned __int64 *)&v635);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "print<ddg>", 10) )
                                                  {
                                                    v581 = sub_CB72A0();
                                                    sub_2355B20((__int64)v640, (__int64)v581, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "print<iv-users>", 15) )
                                                  {
                                                    v580 = sub_CB72A0();
                                                    sub_2355D60((__int64)v640, (__int64)v580, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "print<loop-cache-cost>", 22) )
                                                  {
                                                    v579 = sub_CB72A0();
                                                    sub_2355C40((__int64)v640, (__int64)v579, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "print<loopnest>", 15) )
                                                  {
                                                    v578 = sub_CB72A0();
                                                    sub_2355E80((__int64)v640, (__int64)v578, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( sub_9691B0(v564, v565, "loop-index-split", 16) )
                                                  {
                                                    sub_2354A10((__int64)v640, 0, 0, 0);
                                                    sub_2353940(a3, v640[0].m128i_i64);
                                                    sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                    sub_233F7D0(v640[0].m128i_i64);
                                                    v640[0].m128i_i64[0] = 0;
                                                    *a1 = 1;
                                                    sub_9C66B0(v640[0].m128i_i64);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0((char *)v564, v565, "licm", 4u) )
                                                  {
                                                    sub_234D100(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2332EB0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "licm",
                                                      4u);
                                                    v576 = v639[1].m128i_i8[0] & 1;
                                                    v639[1].m128i_i8[0] = (2 * (v639[1].m128i_i8[0] & 1))
                                                                        | v639[1].m128i_i8[0] & 0xFD;
                                                    if ( !v576 )
                                                    {
                                                      v635.m128i_i64[0] = v639[0].m128i_i64[0];
                                                      v635.m128i_i16[4] = v639[0].m128i_i16[4];
                                                      sub_2356430((__int64)v640, v635.m128i_i64, 0, 0, 0);
                                                      v577 = v640;
                                                      sub_2353940(a3, v640[0].m128i_i64);
                                                      sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                      sub_233F7D0(v640[0].m128i_i64);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                      goto LABEL_1996;
                                                    }
LABEL_1995:
                                                    v577 = v639;
                                                    sub_234D1A0(a1, v639[0].m128i_i64);
LABEL_1996:
                                                    sub_2352290(v639, (__int64)v577);
                                                    return a1;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "lnicm",
                                                                          5u) )
                                                  {
                                                    sub_234D100(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2332EB0,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "lnicm",
                                                      5u);
                                                    v583 = v639[1].m128i_i8[0] & 1;
                                                    v639[1].m128i_i8[0] = (2 * (v639[1].m128i_i8[0] & 1))
                                                                        | v639[1].m128i_i8[0] & 0xFD;
                                                    if ( !v583 )
                                                    {
                                                      v635.m128i_i64[0] = v639[0].m128i_i64[0];
                                                      v635.m128i_i16[4] = v639[0].m128i_i16[4];
                                                      sub_235B1C0((__int64)v640, v635.m128i_i64, 0, 0, 0, v582);
                                                      v577 = v640;
                                                      sub_2353940(a3, v640[0].m128i_i64);
                                                      sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                      sub_233F7D0(v640[0].m128i_i64);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                      goto LABEL_1996;
                                                    }
                                                    goto LABEL_1995;
                                                  }
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          *(_QWORD *)(a4 + 8),
                                                                          "loop-rotate",
                                                                          0xBu) )
                                                  {
                                                    sub_234D210(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2332C00,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "loop-rotate",
                                                      0xBu);
                                                    v585 = v639[0].m128i_i8[8] & 1;
                                                    v639[0].m128i_i8[8] = (2 * (v639[0].m128i_i8[8] & 1))
                                                                        | v639[0].m128i_i8[8] & 0xFD;
                                                    if ( v585 )
                                                    {
                                                      sub_9C9930(a1, v639[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      sub_28448C0(v640, v639[0].m128i_u8[0], v639[0].m128i_u8[1]);
                                                      v635.m128i_i16[0] = v640[0].m128i_i16[0];
                                                      sub_23560C0((__int64)v640, v635.m128i_i16, 0, 0, 0);
                                                      sub_2353940(a3, v640[0].m128i_i64);
                                                      sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                      sub_233F7D0(v640[0].m128i_i64);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                    }
LABEL_2017:
                                                    sub_23521F0(v639);
                                                    return a1;
                                                  }
                                                  v9 = *(_QWORD *)a4;
                                                  v10 = *(__m128i **)(a4 + 8);
                                                  if ( (unsigned __int8)sub_2337DE0(
                                                                          *(char **)a4,
                                                                          (__int64)v10,
                                                                          "simple-loop-unswitch",
                                                                          0x14u) )
                                                  {
                                                    sub_234D210(
                                                      (__int64)v639,
                                                      (void (__fastcall *)(__int64, const void *, __int64))sub_2332990,
                                                      *(const void **)a4,
                                                      *(_QWORD *)(a4 + 8),
                                                      "simple-loop-unswitch",
                                                      0x14u);
                                                    v598 = v639[0].m128i_i8[8] & 1;
                                                    v639[0].m128i_i8[8] = (2 * (v639[0].m128i_i8[8] & 1))
                                                                        | v639[0].m128i_i8[8] & 0xFD;
                                                    if ( v598 )
                                                    {
                                                      sub_9C9930(a1, v639[0].m128i_i64);
                                                    }
                                                    else
                                                    {
                                                      v635.m128i_i16[0] = v639[0].m128i_i16[0];
                                                      sub_23561E0((__int64)v640, v635.m128i_i16, 0, 0, 0);
                                                      sub_2353940(a3, v640[0].m128i_i64);
                                                      sub_233F7F0((__int64)&v640[0].m128i_i64[1]);
                                                      sub_233F7D0(v640[0].m128i_i64);
                                                      v640[0].m128i_i64[0] = 0;
                                                      *a1 = 1;
                                                      sub_9C66B0(v640[0].m128i_i64);
                                                    }
                                                    goto LABEL_2017;
                                                  }
                                                  v587 = v617[108].m128i_i64[0];
                                                  v588 = v587 + 32LL * v617[108].m128i_u32[2];
                                                  do
                                                  {
                                                    if ( v588 == v587 )
                                                    {
                                                      v594 = sub_C63BB0();
                                                      v640[0].m128i_i64[1] = 27;
                                                      v595 = v594;
                                                      v597 = v596;
                                                      v640[0].m128i_i64[0] = (__int64)"unknown function pass '{0}'";
                                                      v640[1].m128i_i64[0] = (__int64)&v640[3].m128i_i64[1];
                                                      v640[1].m128i_i64[1] = 1;
                                                      v640[2].m128i_i8[0] = 1;
                                                      v640[2].m128i_i64[1] = (__int64)&unk_4A0B5E8;
                                                      v640[3].m128i_i64[1] = (__int64)&v640[2].m128i_i64[1];
                                                      v640[3].m128i_i64[0] = a4;
                                                      sub_23328D0((__int64)v639, (__int64)v640);
                                                      sub_23058C0(a1, (__int64)v639, v595, v597);
                                                      sub_2240A30((unsigned __int64 *)v639);
                                                      return a1;
                                                    }
                                                    v11 = *(_QWORD *)(a4 + 16);
                                                    v589 = _mm_loadu_si128((const __m128i *)a4);
                                                    v590 = *(_QWORD *)(a4 + 24) - v11;
                                                    v640[0].m128i_i64[0] = v11;
                                                    v639[0] = v589;
                                                    v640[0].m128i_i64[1] = 0xCCCCCCCCCCCCCCCDLL * (v590 >> 3);
                                                    if ( !*(_QWORD *)(v587 + 16) )
LABEL_523:
                                                      sub_4263D6(v9, v10, v11);
                                                    v9 = v587;
                                                    v10 = v639;
                                                    v591 = (*(__int64 (__fastcall **)(__int64, __m128i *, unsigned __int64 *, __m128i *))(v587 + 24))(
                                                             v587,
                                                             v639,
                                                             a3,
                                                             v640);
                                                    v587 += 32;
                                                  }
                                                  while ( !v591 );
LABEL_2039:
                                                  v640[0].m128i_i64[0] = 0;
                                                  *a1 = 1;
                                                  sub_9C66B0(v640[0].m128i_i64);
                                                  return a1;
                                                }
                                                v131 = (_QWORD *)sub_22077B0(0x10u);
                                                if ( v131 )
                                                  *v131 = &unk_4A117F8;
                                                v640[0].m128i_i64[0] = (__int64)v131;
                                                sub_2353900(a3, (unsigned __int64 *)v640);
                                                if ( v640[0].m128i_i64[0] )
                                                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v640[0].m128i_i64[0] + 8LL))(v640[0].m128i_i64[0]);
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        v640[0].m128i_i64[0] = 0;
        *a1 = 1;
        sub_9C66B0(v640[0].m128i_i64);
        return a1;
      }
      v113 = (_QWORD *)sub_22077B0(0x10u);
      if ( v113 )
        *v113 = &unk_4A129F8;
    }
    v640[0].m128i_i64[0] = (__int64)v113;
    sub_2353900(a3, (unsigned __int64 *)v640);
    sub_233EFE0(v640[0].m128i_i64);
    v640[0].m128i_i64[0] = 0;
    *a1 = 1;
    sub_9C66B0(v640[0].m128i_i64);
    return a1;
  }
  v7 = *(const void **)a4;
  v8 = *(_QWORD *)(a4 + 8);
  if ( sub_9691B0(*(const void **)a4, v8, "function", 8) )
  {
    memset(v640, 0, 40);
    sub_2377250(v639, a2, v640, v6, 0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)v6) >> 3));
    if ( (v639[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v639[0].m128i_i64[0] = v639[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
      *a1 = 0;
      sub_23055C0(a1, v639);
      sub_9C66B0(v639[0].m128i_i64);
    }
    else
    {
      v639[0].m128i_i64[0] = 0;
      sub_9C66B0(v639[0].m128i_i64);
      v17 = v640[0].m128i_i64[1];
      for ( kk = v640[0].m128i_i64[0]; v17 != kk; kk += 8 )
      {
        v19 = (unsigned __int64 *)kk;
        sub_2353900(a3, v19);
      }
      v639[0].m128i_i64[0] = 0;
      *a1 = 1;
      sub_9C66B0(v639[0].m128i_i64);
    }
    sub_233F7F0((__int64)v640);
    return a1;
  }
  if ( !sub_9691B0(v7, v8, "loop", 4) && !sub_9691B0(v7, v8, "loop-mssa", 9) )
  {
    v9 = (__int64)v7;
    v10 = (__m128i *)v8;
    if ( sub_9691B0(v7, v8, "machine-function", 16) )
    {
      memset(v640, 0, 40);
      sub_2361A00(
        (unsigned __int64 *)v639,
        v617,
        (unsigned __int64 *)v640,
        v6,
        0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)v6) >> 3));
      if ( (v639[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v639[0].m128i_i64[0] = v639[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        *a1 = 0;
        sub_23055C0(a1, v639);
        sub_9C66B0(v639[0].m128i_i64);
      }
      else
      {
        v639[0].m128i_i64[0] = 0;
        sub_9C66B0(v639[0].m128i_i64);
        sub_234DE90(&v631, (unsigned __int64 *)v640);
        v120 = v631.m128i_i64[0];
        v631.m128i_i64[0] = 0;
        v635.m128i_i64[0] = v120;
        v121 = (_QWORD *)sub_22077B0(0x10u);
        if ( v121 )
        {
          *v121 = &unk_4A12778;
          v122 = v635.m128i_i64[0];
          v635.m128i_i64[0] = 0;
          v121[1] = v122;
        }
        v639[0].m128i_i64[0] = (__int64)v121;
        sub_2353900(a3, (unsigned __int64 *)v639);
        sub_233EFE0(v639[0].m128i_i64);
        sub_233F0A0(v635.m128i_i64);
        sub_233F0A0(v631.m128i_i64);
        v639[0].m128i_i64[0] = 0;
        *a1 = 1;
        sub_9C66B0(v639[0].m128i_i64);
      }
      v118 = v640[0].m128i_i64[1];
      v119 = (_QWORD *)v640[0].m128i_i64[0];
      if ( v640[0].m128i_i64[1] != v640[0].m128i_i64[0] )
      {
        do
        {
          if ( *v119 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v119 + 8LL))(*v119);
          ++v119;
        }
        while ( (_QWORD *)v118 != v119 );
        v119 = (_QWORD *)v640[0].m128i_i64[0];
      }
      if ( v119 )
        j_j___libc_free_0((unsigned __int64)v119);
    }
    else
    {
      v12 = v617[108].m128i_i64[0];
      if ( v12 == v12 + 32LL * v617[108].m128i_u32[2] )
      {
LABEL_431:
        v114 = sub_C63BB0();
        v640[0].m128i_i64[1] = 46;
        v115 = v114;
        v117 = v116;
        v640[0].m128i_i64[0] = (__int64)"invalid use of '{0}' pass as function pipeline";
        v640[1].m128i_i64[0] = (__int64)&v640[3].m128i_i64[1];
        v640[1].m128i_i64[1] = 1;
        v640[2].m128i_i8[0] = 1;
        v640[2].m128i_i64[1] = (__int64)&unk_4A0B5E8;
        v640[3].m128i_i64[1] = (__int64)&v640[2].m128i_i64[1];
        v640[3].m128i_i64[0] = a4;
        sub_23328D0((__int64)v639, (__int64)v640);
        sub_23058C0(a1, (__int64)v639, v115, v117);
        sub_2240A30((unsigned __int64 *)v639);
      }
      else
      {
        v13 = v5;
        v14 = v617[108].m128i_i64[0];
        v15 = v12 + 32LL * v617[108].m128i_u32[2];
        while ( 1 )
        {
          v16 = _mm_loadu_si128((const __m128i *)a4);
          v635.m128i_i64[0] = (__int64)v6;
          v631 = v16;
          v635.m128i_i64[1] = 0xCCCCCCCCCCCCCCCDLL * (((char *)v13 - (char *)v6) >> 3);
          if ( !*(_QWORD *)(v14 + 16) )
            goto LABEL_523;
          v10 = &v631;
          v9 = v14;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, unsigned __int64 *, __m128i *))(v14 + 24))(
                 v14,
                 &v631,
                 a3,
                 &v635) )
          {
            break;
          }
          v14 += 32;
          if ( v15 == v14 )
            goto LABEL_431;
          v13 = *(const __m128i **)(a4 + 24);
          v6 = *(const __m128i **)(a4 + 16);
        }
        v640[0].m128i_i64[0] = 0;
        *a1 = 1;
        sub_9C66B0(v640[0].m128i_i64);
      }
    }
    return a1;
  }
  v639[0].m128i_i64[0] = (__int64)v639[1].m128i_i64;
  v639[0].m128i_i64[1] = 0x600000000LL;
  v639[4].m128i_i32[0] = 0;
  v639[4].m128i_i64[1] = 0;
  memset(&v639[5], 0, 40);
  sub_235CCD0(
    (unsigned __int64 *)v640,
    (__int64)a2,
    (unsigned __int64 *)v639,
    v6,
    0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)v6) >> 3));
  if ( (v640[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v640[0].m128i_i64[0] = v640[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    *a1 = 0;
    sub_23055C0(a1, v640);
    sub_9C66B0(v640[0].m128i_i64);
    goto LABEL_417;
  }
  v640[0].m128i_i64[0] = 0;
  sub_9C66B0(v640[0].m128i_i64);
  v20 = *(__m128i **)a4;
  m128i_i8 = *(char **)(a4 + 8);
  v22 = sub_9691B0(*(const void **)a4, (size_t)m128i_i8, "loop-mssa", 9);
  v27 = *(_QWORD *)(a4 + 16);
  v599 = v22;
  v606 = *(_QWORD *)(a4 + 24);
  v28 = 0xCCCCCCCCCCCCCCCDLL * ((v606 - v27) >> 3);
  if ( v28 >> 2 <= 0 )
  {
LABEL_75:
    if ( v28 != 2 )
    {
      if ( v28 != 3 )
      {
        if ( v28 != 1 )
        {
LABEL_78:
          v601 = 0;
          goto LABEL_79;
        }
LABEL_490:
        v20 = (__m128i *)v27;
        if ( !(unsigned __int8)sub_2366CE0((const __m128i *)v27, (__int64)m128i_i8, v23) )
          goto LABEL_78;
LABEL_491:
        v601 = v606 != v27;
        goto LABEL_79;
      }
      v20 = (__m128i *)v27;
      if ( (unsigned __int8)sub_2366CE0((const __m128i *)v27, (__int64)m128i_i8, v23) )
        goto LABEL_491;
      v27 += 40;
    }
    v20 = (__m128i *)v27;
    if ( (unsigned __int8)sub_2366CE0((const __m128i *)v27, (__int64)m128i_i8, v23) )
      goto LABEL_491;
    v27 += 40;
    goto LABEL_490;
  }
  v611 = v27 + 160 * (v28 >> 2);
  while ( 1 )
  {
    v627 = _mm_loadu_si128((const __m128i *)v27);
    v29 = *(_QWORD *)(v27 + 24) - *(_QWORD *)(v27 + 16);
    v628 = 0;
    v629 = 0;
    v630 = 0;
    if ( v29 )
    {
      if ( v29 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_504;
      v30 = sub_22077B0(v29);
    }
    else
    {
      v29 = 0;
      v30 = 0;
    }
    v628 = v30;
    v629 = v30;
    v630 = v30 + v29;
    v31 = *(_QWORD *)(v27 + 24);
    if ( v31 != *(_QWORD *)(v27 + 16) )
    {
      v32 = *(_QWORD *)(v27 + 16);
      do
      {
        if ( v30 )
        {
          *(__m128i *)v30 = _mm_loadu_si128((const __m128i *)v32);
          sub_23667F0((__m128i **)(v30 + 16), (const __m128i **)(v32 + 16), v23);
        }
        v32 += 40;
        v30 += 40;
      }
      while ( v31 != v32 );
    }
    v20 = &v627;
    m128i_i8 = "simple-loop-unswitch";
    v629 = v30;
    v33 = sub_C931B0(v627.m128i_i64, "simple-loop-unswitch", 0x14u, 0);
    v34 = v629;
    v35 = v628;
    v36 = v33;
    if ( v629 != v628 )
    {
      do
      {
        v20 = (__m128i *)(v35 + 16);
        v35 += 40LL;
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v34 != v35 );
      v35 = v628;
    }
    if ( v35 )
    {
      v20 = (__m128i *)v35;
      m128i_i8 = (char *)(v630 - v35);
      j_j___libc_free_0(v35);
    }
    if ( v36 != -1 )
    {
      v601 = v606 != v27;
      goto LABEL_79;
    }
    v631 = _mm_loadu_si128((const __m128i *)(v27 + 40));
    v37 = *(_QWORD *)(v27 + 64);
    v38 = *(_QWORD *)(v27 + 56);
    v632 = 0;
    v633 = 0;
    mm = 0;
    v23 = v37 - v38;
    if ( v37 == v38 )
    {
      v40 = 0;
    }
    else
    {
      if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_504;
      v618 = v37 - v38;
      v39 = sub_22077B0(v37 - v38);
      v37 = *(_QWORD *)(v27 + 64);
      v38 = *(_QWORD *)(v27 + 56);
      v23 = v618;
      v40 = v39;
    }
    v41 = v40 + v23;
    v632 = v40;
    v633 = v40;
    for ( mm = v41; v37 != v38; v40 += 40 )
    {
      if ( v40 )
      {
        *(__m128i *)v40 = _mm_loadu_si128((const __m128i *)v38);
        sub_23667F0((__m128i **)(v40 + 16), (const __m128i **)(v38 + 16), v41);
      }
      v38 += 40;
    }
    v20 = &v631;
    m128i_i8 = "simple-loop-unswitch";
    v633 = v40;
    v42 = sub_C931B0(v631.m128i_i64, "simple-loop-unswitch", 0x14u, 0);
    v43 = v633;
    v44 = v632;
    v45 = v42;
    if ( v633 != v632 )
    {
      do
      {
        v20 = (__m128i *)(v44 + 16);
        v44 += 40LL;
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v43 != v44 );
      v44 = v632;
    }
    if ( v44 )
    {
      v20 = (__m128i *)v44;
      m128i_i8 = (char *)(mm - v44);
      j_j___libc_free_0(v44);
    }
    if ( v45 != -1 )
    {
      v601 = v606 != v27 + 40;
      goto LABEL_79;
    }
    v635 = _mm_loadu_si128((const __m128i *)(v27 + 80));
    v46 = *(_QWORD *)(v27 + 104);
    v47 = *(_QWORD *)(v27 + 96);
    v636 = 0;
    v637 = 0;
    nn = 0;
    v23 = v46 - v47;
    if ( v46 == v47 )
    {
      v49 = 0;
    }
    else
    {
      if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_504;
      v619 = v46 - v47;
      v48 = sub_22077B0(v46 - v47);
      v46 = *(_QWORD *)(v27 + 104);
      v47 = *(_QWORD *)(v27 + 96);
      v23 = v619;
      v49 = v48;
    }
    v50 = v49 + v23;
    v636 = v49;
    v637 = v49;
    for ( nn = v50; v46 != v47; v49 += 40 )
    {
      if ( v49 )
      {
        *(__m128i *)v49 = _mm_loadu_si128((const __m128i *)v47);
        sub_23667F0((__m128i **)(v49 + 16), (const __m128i **)(v47 + 16), v50);
      }
      v47 += 40;
    }
    v20 = &v635;
    m128i_i8 = "simple-loop-unswitch";
    v637 = v49;
    v51 = sub_C931B0(v635.m128i_i64, "simple-loop-unswitch", 0x14u, 0);
    v52 = v637;
    v53 = v636;
    v54 = v51;
    if ( v637 != v636 )
    {
      do
      {
        v20 = (__m128i *)(v53 + 16);
        v53 += 40LL;
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v52 != v53 );
      v53 = v636;
    }
    if ( v53 )
    {
      v20 = (__m128i *)v53;
      m128i_i8 = (char *)(nn - v53);
      j_j___libc_free_0(v53);
    }
    if ( v54 != -1 )
    {
      v601 = v606 != v27 + 80;
      goto LABEL_79;
    }
    v640[0] = _mm_loadu_si128((const __m128i *)(v27 + 120));
    v55 = *(_QWORD *)(v27 + 144);
    v56 = *(_QWORD *)(v27 + 136);
    memset(&v640[1], 0, 24);
    v23 = v55 - v56;
    if ( v55 == v56 )
    {
      v58 = 0;
    }
    else
    {
      if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_504:
        sub_4261EA(v20, m128i_i8, v23);
      v620 = v55 - v56;
      v57 = sub_22077B0(v55 - v56);
      v55 = *(_QWORD *)(v27 + 144);
      v56 = *(_QWORD *)(v27 + 136);
      v23 = v620;
      v58 = v57;
    }
    v59 = v58 + v23;
    v640[1].m128i_i64[0] = v58;
    v640[1].m128i_i64[1] = v58;
    for ( v640[2].m128i_i64[0] = v59; v55 != v56; v58 += 40 )
    {
      if ( v58 )
      {
        *(__m128i *)v58 = _mm_loadu_si128((const __m128i *)v56);
        sub_23667F0((__m128i **)(v58 + 16), (const __m128i **)(v56 + 16), v59);
      }
      v56 += 40;
    }
    m128i_i8 = "simple-loop-unswitch";
    v20 = v640;
    v640[1].m128i_i64[1] = v58;
    v60 = sub_C931B0(v640[0].m128i_i64, "simple-loop-unswitch", 0x14u, 0);
    v61 = v640[1].m128i_i64[1];
    v62 = v640[1].m128i_u64[0];
    v63 = v60;
    if ( v640[1].m128i_i64[1] != v640[1].m128i_i64[0] )
    {
      do
      {
        v20 = (__m128i *)(v62 + 16);
        v62 += 40LL;
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v61 != v62 );
      v62 = v640[1].m128i_u64[0];
    }
    if ( v62 )
    {
      v20 = (__m128i *)v62;
      m128i_i8 = (char *)(v640[2].m128i_i64[0] - v62);
      j_j___libc_free_0(v62);
    }
    if ( v63 != -1 )
      break;
    v27 += 160;
    if ( v27 == v611 )
    {
      v28 = 0xCCCCCCCCCCCCCCCDLL * ((v606 - v27) >> 3);
      goto LABEL_75;
    }
  }
  v601 = v606 != v27 + 120;
LABEL_79:
  v64 = *(const __m128i **)(a4 + 16);
  v603 = *(const __m128i **)(a4 + 24);
  v65 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v603 - (char *)v64) >> 3);
  v23 = v65 >> 2;
  if ( v65 >> 2 <= 0 )
  {
LABEL_143:
    if ( v65 != 2 )
    {
      if ( v65 != 3 )
      {
        if ( v65 != 1 )
        {
LABEL_146:
          v102 = 0;
          goto LABEL_147;
        }
LABEL_451:
        if ( !(unsigned __int8)sub_2366BC0(v64, (__int64)m128i_i8, v23) )
          goto LABEL_146;
LABEL_452:
        v102 = v603 != v64;
        goto LABEL_147;
      }
      if ( (unsigned __int8)sub_2366BC0(v64, (__int64)m128i_i8, v23) )
        goto LABEL_452;
      v64 = (const __m128i *)((char *)v64 + 40);
    }
    if ( (unsigned __int8)sub_2366BC0(v64, (__int64)m128i_i8, v23) )
      goto LABEL_452;
    v64 = (const __m128i *)((char *)v64 + 40);
    goto LABEL_451;
  }
  v604 = &v64[10 * v23];
  while ( 1 )
  {
    v66 = v64[1].m128i_i64[1];
    v67 = v64[1].m128i_i64[0];
    v607 = (_QWORD *)v64->m128i_i64[0];
    v612 = v64->m128i_i64[1];
    v621 = v66 - v67;
    if ( v66 == v67 )
    {
      v71 = 0;
      if ( v612 != 16 )
        goto LABEL_96;
      v72 = 0;
LABEL_89:
      v23 = *v607 ^ 0x6572702D706F6F6CLL;
      v24 = v23 | v607[1] ^ 0x6E6F697461636964LL;
      v73 = v24 == 0;
      goto LABEL_90;
    }
    if ( v621 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_504;
    v20 = (__m128i *)(v66 - v67);
    v68 = sub_22077B0(v621);
    v69 = v64[1].m128i_i64[1];
    v70 = (const __m128i *)v64[1].m128i_i64[0];
    v71 = v68;
    if ( (const __m128i *)v69 == v70 )
    {
      v72 = (__m128i *)v68;
      if ( v612 != 16 )
      {
        if ( v68 )
        {
          m128i_i8 = (char *)v621;
          v20 = (__m128i *)v68;
          j_j___libc_free_0(v68);
        }
        goto LABEL_96;
      }
      goto LABEL_89;
    }
    v72 = (__m128i *)v68;
    do
    {
      if ( v72 )
      {
        m128i_i8 = v70[1].m128i_i8;
        v20 = v72 + 1;
        *v72 = _mm_loadu_si128(v70);
        sub_23667F0((__m128i **)&v72[1], (const __m128i **)&v70[1], v23);
      }
      v70 = (const __m128i *)((char *)v70 + 40);
      v72 = (__m128i *)((char *)v72 + 40);
    }
    while ( (const __m128i *)v69 != v70 );
    v73 = 0;
    if ( v612 == 16 )
      goto LABEL_89;
LABEL_90:
    if ( (__m128i *)v71 != v72 )
    {
      v74 = (__m128i *)v71;
      do
      {
        v20 = v74 + 1;
        v74 = (__m128i *)((char *)v74 + 40);
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v74 != v72 );
    }
    if ( v71 )
    {
      m128i_i8 = (char *)v621;
      v20 = (__m128i *)v71;
      j_j___libc_free_0(v71);
    }
    if ( v73 )
      goto LABEL_452;
LABEL_96:
    v75 = v64[4].m128i_i64[0];
    v76 = v64[3].m128i_i64[1];
    v608 = (_QWORD *)v64[2].m128i_i64[1];
    v613 = v64[3].m128i_i64[0];
    v622 = v75 - v76;
    if ( v75 == v76 )
    {
      v80 = 0;
      if ( v613 != 16 )
        goto LABEL_111;
      v81 = 0;
LABEL_104:
      v23 = *v608 ^ 0x6572702D706F6F6CLL;
      v24 = v23 | v608[1] ^ 0x6E6F697461636964LL;
      v82 = v24 == 0;
      goto LABEL_105;
    }
    if ( v622 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_504;
    v20 = (__m128i *)(v75 - v76);
    v77 = sub_22077B0(v622);
    v78 = v64[4].m128i_i64[0];
    v79 = (const __m128i *)v64[3].m128i_i64[1];
    v80 = v77;
    if ( (const __m128i *)v78 == v79 )
    {
      v81 = (__m128i *)v77;
      if ( v613 != 16 )
      {
        if ( v77 )
        {
          m128i_i8 = (char *)v622;
          v20 = (__m128i *)v77;
          j_j___libc_free_0(v77);
        }
        goto LABEL_111;
      }
      goto LABEL_104;
    }
    v81 = (__m128i *)v77;
    do
    {
      if ( v81 )
      {
        m128i_i8 = v79[1].m128i_i8;
        v20 = v81 + 1;
        *v81 = _mm_loadu_si128(v79);
        sub_23667F0((__m128i **)&v81[1], (const __m128i **)&v79[1], v23);
      }
      v79 = (const __m128i *)((char *)v79 + 40);
      v81 = (__m128i *)((char *)v81 + 40);
    }
    while ( (const __m128i *)v78 != v79 );
    v82 = 0;
    if ( v613 == 16 )
      goto LABEL_104;
LABEL_105:
    if ( (__m128i *)v80 != v81 )
    {
      v83 = (__m128i *)v80;
      do
      {
        v20 = v83 + 1;
        v83 = (__m128i *)((char *)v83 + 40);
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v83 != v81 );
    }
    if ( v80 )
    {
      m128i_i8 = (char *)v622;
      v20 = (__m128i *)v80;
      j_j___libc_free_0(v80);
    }
    if ( v82 )
    {
      v102 = v603 != (const __m128i *)&v64[2].m128i_u64[1];
      goto LABEL_147;
    }
LABEL_111:
    v84 = v64[6].m128i_i64[1];
    v85 = v64[6].m128i_i64[0];
    v609 = (_QWORD *)v64[5].m128i_i64[0];
    v614 = v64[5].m128i_i64[1];
    v623 = v84 - v85;
    if ( v84 == v85 )
    {
      v89 = 0;
      if ( v614 != 16 )
        goto LABEL_126;
      v90 = 0;
LABEL_119:
      v23 = *v609 ^ 0x6572702D706F6F6CLL;
      v24 = v23 | v609[1] ^ 0x6E6F697461636964LL;
      v91 = v24 == 0;
      goto LABEL_120;
    }
    if ( v623 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_504;
    v20 = (__m128i *)(v84 - v85);
    v86 = sub_22077B0(v623);
    v87 = v64[6].m128i_i64[1];
    v88 = (const __m128i *)v64[6].m128i_i64[0];
    v89 = v86;
    if ( (const __m128i *)v87 == v88 )
    {
      v90 = (__m128i *)v86;
      if ( v614 != 16 )
      {
        if ( v86 )
        {
          m128i_i8 = (char *)v623;
          v20 = (__m128i *)v86;
          j_j___libc_free_0(v86);
        }
        goto LABEL_126;
      }
      goto LABEL_119;
    }
    v90 = (__m128i *)v86;
    do
    {
      if ( v90 )
      {
        m128i_i8 = v88[1].m128i_i8;
        v20 = v90 + 1;
        *v90 = _mm_loadu_si128(v88);
        sub_23667F0((__m128i **)&v90[1], (const __m128i **)&v88[1], v23);
      }
      v88 = (const __m128i *)((char *)v88 + 40);
      v90 = (__m128i *)((char *)v90 + 40);
    }
    while ( (const __m128i *)v87 != v88 );
    v91 = 0;
    if ( v614 == 16 )
      goto LABEL_119;
LABEL_120:
    if ( (__m128i *)v89 != v90 )
    {
      v92 = (__m128i *)v89;
      do
      {
        v20 = v92 + 1;
        v92 = (__m128i *)((char *)v92 + 40);
        sub_234A6B0((unsigned __int64 *)v20);
      }
      while ( v92 != v90 );
    }
    if ( v89 )
    {
      m128i_i8 = (char *)v623;
      v20 = (__m128i *)v89;
      j_j___libc_free_0(v89);
    }
    if ( v91 )
    {
      v102 = v603 != &v64[5];
      goto LABEL_147;
    }
LABEL_126:
    v93 = v64[9].m128i_i64[0];
    v94 = v64[8].m128i_i64[1];
    v610 = (_QWORD *)v64[7].m128i_i64[1];
    v615 = v64[8].m128i_i64[0];
    v624 = v93 - v94;
    if ( v93 != v94 )
      break;
    v98 = 0;
    if ( v615 == 16 )
    {
      v99 = 0;
      goto LABEL_134;
    }
LABEL_141:
    v64 += 10;
    if ( v604 == v64 )
    {
      v23 = 0xCCCCCCCCCCCCCCCDLL;
      v65 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v603 - (char *)v64) >> 3);
      goto LABEL_143;
    }
  }
  if ( v624 > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_504;
  v20 = (__m128i *)(v93 - v94);
  v95 = sub_22077B0(v624);
  v96 = v64[9].m128i_i64[0];
  v97 = (const __m128i *)v64[8].m128i_i64[1];
  v98 = v95;
  if ( (const __m128i *)v96 == v97 )
  {
    v99 = (__m128i *)v95;
    if ( v615 != 16 )
    {
      if ( v95 )
      {
        m128i_i8 = (char *)v624;
        v20 = (__m128i *)v95;
        j_j___libc_free_0(v95);
      }
      goto LABEL_141;
    }
    goto LABEL_134;
  }
  v99 = (__m128i *)v95;
  do
  {
    if ( v99 )
    {
      m128i_i8 = v97[1].m128i_i8;
      v20 = v99 + 1;
      *v99 = _mm_loadu_si128(v97);
      sub_23667F0((__m128i **)&v99[1], (const __m128i **)&v97[1], v23);
    }
    v97 = (const __m128i *)((char *)v97 + 40);
    v99 = (__m128i *)((char *)v99 + 40);
  }
  while ( (const __m128i *)v96 != v97 );
  v100 = 0;
  if ( v615 == 16 )
  {
LABEL_134:
    v23 = *v610 ^ 0x6572702D706F6F6CLL;
    v24 = v23 | v610[1] ^ 0x6E6F697461636964LL;
    v100 = v24 == 0;
  }
  if ( (__m128i *)v98 != v99 )
  {
    v101 = (__m128i *)v98;
    do
    {
      v20 = v101 + 1;
      v101 = (__m128i *)((char *)v101 + 40);
      sub_234A6B0((unsigned __int64 *)v20);
    }
    while ( v101 != v99 );
  }
  if ( v98 )
  {
    m128i_i8 = (char *)v624;
    v20 = (__m128i *)v98;
    j_j___libc_free_0(v98);
  }
  if ( !v100 )
    goto LABEL_141;
  v102 = v603 != (const __m128i *)&v64[7].m128i_u64[1];
LABEL_147:
  v103 = v639[5].m128i_i64[0] == v639[4].m128i_i64[1];
  sub_2337A80((__int64)v640, (__int64)v639, v23, v24, v25, v26);
  v104 = (_QWORD *)sub_22077B0(0x80u);
  v109 = (__int64)v104;
  if ( v104 )
  {
    *v104 = &unk_4A0B4E8;
    sub_2337A80((__int64)(v104 + 1), (__int64)v640, v105, v106, v107, v108);
  }
  v631.m128i_i64[0] = v109;
  sub_2354930((__int64)&v635, &v631, v599, v601, v102, v103);
  sub_233F7D0(v631.m128i_i64);
  sub_2337B30((unsigned __int64 *)v640);
  sub_2353940(a3, v635.m128i_i64);
  sub_233F7F0((__int64)&v635.m128i_i64[1]);
  sub_233F7D0(v635.m128i_i64);
  v640[0].m128i_i64[0] = 0;
  *a1 = 1;
  sub_9C66B0(v640[0].m128i_i64);
LABEL_417:
  sub_2337B30((unsigned __int64 *)v639);
  return a1;
}
