// Function: sub_1743DA0
// Address: 0x1743da0
//
__int64 __fastcall sub_1743DA0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // r12
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // rbx
  _QWORD *v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rsi
  __int64 v27; // rdi
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rdx
  unsigned int v31; // r8d
  __int64 v32; // rcx
  __int64 v33; // rdi
  __int64 v34; // rax
  int v35; // eax
  bool v36; // al
  __int64 v37; // rax
  __int64 *v38; // rax
  unsigned int v39; // r11d
  int v40; // eax
  __int64 v41; // rcx
  __int64 *v42; // rdi
  __int64 v43; // r14
  __int64 v44; // rsi
  __int64 v45; // r11
  int v46; // esi
  __int64 *v47; // rbx
  __int64 *v48; // rax
  __int64 **v49; // rdi
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 *v52; // r12
  _QWORD *v53; // rax
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rcx
  int v57; // ecx
  int v58; // ecx
  __int64 **v59; // rax
  __int64 *v60; // rax
  __int64 *v61; // rdi
  _QWORD *v62; // rax
  __int64 **v63; // r15
  __int64 v64; // r12
  _QWORD *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rax
  char v69; // dl
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // rax
  double v73; // xmm4_8
  double v74; // xmm5_8
  __int64 v75; // rax
  __int64 v76; // rcx
  _QWORD *v77; // rbx
  __int64 v78; // rcx
  __int64 v79; // r8
  unsigned __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // r14
  __int64 v83; // rsi
  __int64 v84; // r14
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // rdx
  _QWORD *v88; // rax
  int v89; // r13d
  unsigned __int64 v90; // r14
  char v91; // si
  unsigned int v92; // r12d
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rbx
  _BYTE *v97; // r14
  unsigned __int8 v98; // al
  unsigned int v99; // r13d
  bool v100; // al
  char v101; // al
  __int64 v102; // rdx
  __int64 v103; // rsi
  __int64 v104; // r13
  __int64 v105; // r14
  __int64 v106; // rax
  __int64 v107; // rax
  double v108; // xmm4_8
  double v109; // xmm5_8
  __int64 v110; // rdx
  __int64 v111; // rdi
  __int64 v112; // r13
  __int64 v113; // r14
  __int64 v114; // rdx
  __int64 v115; // rbx
  _QWORD *v116; // rax
  _QWORD *v117; // rax
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // r14
  _QWORD *v121; // r13
  bool v122; // al
  unsigned int v123; // r11d
  __int64 v124; // rax
  unsigned int v125; // eax
  char v126; // al
  __int64 v127; // rdx
  __int64 v128; // rdx
  bool v129; // al
  unsigned int v130; // r11d
  __int64 v131; // rax
  unsigned int v132; // eax
  char v133; // al
  __int64 v134; // rdx
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 v137; // r13
  __int64 v138; // rsi
  __int64 v139; // rax
  _QWORD *v140; // rbx
  __int64 **v141; // r14
  unsigned int v142; // ecx
  unsigned __int64 v143; // rax
  unsigned __int64 v144; // r8
  __int64 v145; // rsi
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 v148; // rdx
  _QWORD *v149; // rax
  _QWORD *v150; // rbx
  __int64 *v151; // r12
  __int64 v152; // rax
  __int64 v153; // rax
  __int64 v154; // rax
  bool v155; // al
  bool v156; // al
  bool v157; // al
  unsigned int v158; // r11d
  _QWORD *v159; // rt0
  __int64 v160; // rax
  int v161; // eax
  __int64 v162; // rdx
  __int64 v163; // rbx
  bool v164; // al
  bool v165; // al
  bool v166; // al
  unsigned int v167; // r11d
  __int64 v168; // rax
  int v169; // eax
  __int64 v170; // rdx
  __int64 v171; // r8
  unsigned int v172; // eax
  __int64 v173; // rdi
  _QWORD *v174; // rdx
  __int64 v175; // rcx
  __int64 v176; // rcx
  __int64 v177; // rdx
  __int64 v178; // rax
  __int64 v179; // rdi
  __int64 v180; // rdx
  unsigned __int8 *v181; // rax
  __int64 v182; // rdi
  __int64 **v183; // rcx
  unsigned __int8 *v184; // rax
  __int64 v185; // rdi
  unsigned __int8 *v186; // rax
  __int64 v187; // rdi
  __int64 v188; // rdx
  _QWORD *v189; // rax
  int v190; // eax
  __int64 **v191; // rdx
  __int64 v192; // rax
  _QWORD *v193; // rax
  __int64 *v194; // rax
  __int64 **v195; // rdx
  __int64 v196; // rdx
  __int64 v197; // rax
  int v198; // eax
  __int64 *v199; // rdi
  __int64 v200; // rax
  bool v201; // zf
  __int64 v202; // rax
  _QWORD *v203; // rax
  __int64 v204; // rsi
  unsigned __int64 v205; // rdi
  char v206; // al
  __int64 v207; // r14
  _QWORD *v208; // rdi
  __int64 v209; // rax
  double v210; // xmm4_8
  double v211; // xmm5_8
  __int64 v212; // r13
  __int64 v213; // rax
  unsigned __int64 v214; // rax
  __int64 v215; // r14
  unsigned __int64 v216; // rbx
  __int64 v217; // r13
  char v218; // al
  __int64 v219; // rax
  __int64 v220; // rdx
  __int64 v221; // rdi
  __int64 v222; // r14
  unsigned int v223; // edx
  __int64 *v224; // rax
  __int64 v225; // rdx
  unsigned int v226; // esi
  int v227; // r8d
  __int64 *v228; // rax
  int v229; // edx
  __int64 v230; // rbx
  int v231; // eax
  int v232; // ebx
  __int64 v233; // rdi
  __int64 v234; // rbx
  __int64 v235; // rdx
  __int64 v236; // rax
  __int64 v237; // rdi
  __int64 v238; // r14
  __int64 v239; // rdx
  __int64 v240; // rcx
  __int64 i; // r12
  __int64 v242; // r15
  __int64 v243; // rax
  __int64 v244; // rdi
  __int64 *v245; // rax
  int v246; // r8d
  int v247; // r9d
  _QWORD *v248; // rax
  __int64 v249; // rdx
  __int64 v250; // rcx
  __int64 v251; // rax
  __int64 v252; // rsi
  __int64 *v253; // rdi
  __int64 v254; // r12
  __int64 v255; // r13
  _QWORD *v256; // rax
  unsigned __int64 v257; // rdi
  _BYTE *v258; // rdi
  unsigned __int8 v259; // al
  __int64 v260; // rdi
  bool v261; // al
  __int64 v262; // rdi
  __int64 v263; // rdx
  __int64 *v264; // rcx
  __int64 *v265; // rdi
  __int64 v266; // r8
  __int64 v267; // r14
  __int64 v268; // rdx
  void *v269; // rax
  __int64 v270; // rax
  void *v271; // rax
  __int64 v272; // r9
  __int64 v273; // rdx
  __int64 v274; // r14
  __int64 v275; // rbx
  char v276; // al
  __int64 **v277; // rax
  __int64 v278; // rdx
  __int64 v279; // r14
  __int64 v280; // rbx
  __int64 v281; // rdx
  __int64 v282; // rcx
  __int64 v283; // rdx
  __int64 v284; // rcx
  unsigned __int8 v285; // al
  char v286; // al
  unsigned __int8 v287; // dl
  __int64 v288; // rax
  __int64 v289; // rsi
  __int64 v290; // rdx
  __int64 v291; // r14
  __int64 v292; // rcx
  char v293; // al
  _QWORD *v294; // rax
  __int16 v295; // bx
  int v296; // edx
  __int64 v297; // rsi
  __int64 v298; // r13
  __int64 v299; // rax
  bool v300; // al
  int v301; // eax
  __int64 v302; // rsi
  __int64 v303; // r14
  int v304; // ebx
  int v305; // eax
  __int64 v306; // rax
  __int64 v307; // rdi
  _QWORD *v308; // r12
  __int64 v309; // r15
  _QWORD *v310; // rax
  unsigned __int64 v311; // rbx
  __int64 v312; // rax
  __int64 v313; // rdx
  __int64 v314; // rcx
  char v315; // al
  __int64 v316; // r14
  __int64 v317; // rdx
  __int64 v318; // rcx
  __int64 *v319; // rax
  __int64 v320; // rdi
  __int64 v321; // rdx
  __int64 *v322; // rax
  __int64 v323; // rdi
  __int64 v324; // rdx
  __int64 v325; // rsi
  __int64 v326; // rdi
  __int64 *v327; // rdx
  __int64 v328; // rbx
  __int64 v329; // rcx
  __int64 v330; // rax
  _QWORD *v331; // r14
  _QWORD *v332; // rax
  __int64 v333; // rax
  __int64 v334; // rsi
  __int64 v335; // rax
  __int64 v336; // rdx
  __int64 *v337; // rsi
  __int64 v338; // rsi
  unsigned __int8 *v339; // rsi
  __int64 v340; // rax
  __int64 v341; // rdi
  __int64 v342; // rdx
  __int64 v343; // rax
  __int64 **v344; // rdi
  __int64 *v345; // r14
  __int64 v346; // r8
  char v347; // al
  __int64 v348; // rdx
  __int64 v349; // rcx
  __int64 v350; // r8
  char v351; // al
  char v352; // al
  __int64 v353; // r8
  __int64 *v354; // rbx
  bool v355; // al
  int v356; // edx
  __int64 v357; // rbx
  __int64 v358; // rbx
  __int64 v359; // r14
  __int64 **v360; // rdi
  __int64 v361; // r12
  __int64 *v362; // rax
  __int64 v363; // rdx
  __int64 v364; // rax
  _BYTE *v365; // r14
  unsigned int v366; // ebx
  __int64 v367; // rax
  __int64 v368; // rdx
  __int64 v369; // rcx
  __int64 v370; // rax
  __int64 v371; // rdi
  char v372; // al
  __int64 v373; // r11
  __int64 v374; // rbx
  unsigned int v375; // eax
  int v376; // r13d
  __int64 *v377; // rbx
  __int64 v378; // rsi
  __int64 v379; // rax
  unsigned __int64 v380; // rax
  int v381; // eax
  unsigned __int64 v382; // rax
  int v383; // eax
  bool v385; // al
  __int64 *v386; // rdx
  unsigned __int8 v387; // al
  bool v388; // al
  __int64 v389; // rax
  __int64 v390; // rsi
  __int64 v391; // rax
  __int64 *v392; // rax
  __int64 v393; // rax
  __int64 v394; // rcx
  __int64 *v395; // rax
  __int64 v396; // rsi
  unsigned __int64 v397; // rdi
  __int64 v398; // rsi
  int v399; // eax
  __int64 v400; // r14
  __int64 v401; // rcx
  __int64 v402; // r14
  char *v403; // rsi
  char *v404; // rbx
  __int64 v405; // rdi
  __int64 v406; // rax
  __int64 v407; // rdx
  __int64 v408; // rax
  __int64 v409; // r14
  unsigned int v410; // ebx
  __int64 v411; // rax
  __int64 v412; // rdx
  __int64 v413; // rax
  __int64 v414; // r13
  __int64 v415; // rdx
  __int64 v416; // rcx
  __int64 v417; // rax
  __int64 v418; // r12
  __int64 v419; // rdx
  _QWORD *v420; // rbx
  __int64 v421; // r13
  _QWORD *v422; // rax
  __int64 v423; // rax
  __int64 v424; // rax
  __int64 v425; // r14
  __int64 v426; // rbx
  __int64 *v427; // r9
  __int64 v428; // rax
  __int64 v429; // r14
  __int64 v430; // rbx
  __int64 v431; // rax
  __int64 **v432; // rax
  __int64 *v433; // r9
  __int64 **v434; // r10
  __int64 v435; // rdx
  unsigned __int8 *v436; // rax
  int v437; // esi
  __int64 *v438; // rdi
  __int64 v439; // rax
  __int64 v440; // rax
  __int64 v441; // rax
  __int64 **v442; // rcx
  int v443; // esi
  __int64 **v444; // rax
  __int64 v445; // rdi
  unsigned __int8 *v446; // rax
  __int64 v447; // rdi
  __int64 v448; // rdx
  __int64 v449; // rdi
  __int64 v450; // rbx
  __int64 v451; // r13
  __int64 v452; // r12
  _QWORD *v453; // rax
  double v454; // xmm4_8
  double v455; // xmm5_8
  _QWORD *m; // rbx
  unsigned __int8 *v457; // rax
  __int64 **v461; // rbx
  __int64 v462; // rax
  __int64 **v463; // rdx
  __int64 v464; // rax
  __int64 *v465; // rsi
  __int8 *v466; // rax
  __int64 **v467; // rcx
  __int64 v468; // rax
  __int64 *v469; // rsi
  __int64 **v470; // rdx
  __int64 v471; // rax
  __int64 **v472; // rcx
  __int64 v473; // rax
  __int64 *v474; // rsi
  int v475; // eax
  __int64 **v476; // rdx
  __int64 v477; // rax
  __int64 **v478; // rcx
  __int64 v479; // rax
  __int64 v480; // rax
  __int64 v481; // rsi
  __int64 v482; // r14
  __int64 v483; // rax
  __int64 v484; // rax
  __int64 **v485; // r14
  __int64 **k; // rbx
  __int64 v487; // r13
  __int64 v488; // r12
  __int64 j; // rbx
  __int64 v490; // rax
  __m128i *v491; // rax
  __int64 v492; // rax
  __int64 v493; // rcx
  __int64 *v494; // rdi
  _QWORD *v495; // rbx
  __int64 v496; // rcx
  __int64 v497; // rax
  __int64 *v498; // rdi
  __int64 *v499; // rdi
  __int64 v500; // rax
  unsigned int v501; // r13d
  _QWORD *v502; // rax
  unsigned int v503; // r13d
  __int64 v504; // rax
  char v505; // dl
  bool v506; // al
  __int16 *v507; // r9
  _QWORD *v508; // rax
  _QWORD *v509; // rax
  __int64 v510; // rax
  double v511; // xmm4_8
  double v512; // xmm5_8
  __int64 v513; // rax
  __int64 *v514; // rax
  __int64 v515; // r14
  __int64 *v516; // rax
  __int64 *v517; // rax
  __int64 v518; // rbx
  __int64 *v519; // rax
  __int64 *v520; // rax
  __int64 v521; // rdi
  __int64 v522; // r13
  double v523; // xmm4_8
  double v524; // xmm5_8
  int v525; // ebx
  __int64 v526; // rsi
  __int64 v527; // rax
  bool v528; // al
  __int64 v529; // rdx
  __int64 v530; // rcx
  __int64 v531; // r8
  int v532; // r14d
  __int64 *v533; // rax
  __int64 v534; // r14
  __int64 v535; // rax
  __int64 v536; // rdi
  int v537; // eax
  unsigned __int8 v538; // al
  __int64 v539; // rdi
  int v540; // eax
  bool v541; // al
  int v542; // eax
  unsigned __int8 v543; // al
  __int64 v544; // rdi
  int v545; // eax
  bool v546; // al
  int v547; // eax
  int v548; // eax
  __int64 *v549; // rax
  bool v550; // al
  __int64 v551; // rax
  __int64 v552; // rax
  __int64 *v553; // rbx
  __int64 v554; // rdi
  __int64 v555; // rax
  __int64 v556; // rdi
  __int64 v557; // rax
  _QWORD *v558; // rax
  __int64 v559; // rax
  _QWORD *v560; // rax
  __int64 v561; // rax
  __int64 v562; // rdi
  __int64 *v563; // rax
  _QWORD *v564; // rax
  __int64 v565; // rax
  _QWORD *v566; // rax
  __int64 v567; // rax
  __int64 v568; // rdi
  __int64 v569; // rax
  __int64 v570; // r12
  __int64 v571; // rax
  __int64 v572; // rax
  __int64 v573; // r10
  __int64 *v574; // rax
  __int64 **v575; // rdx
  __int64 *v576; // rax
  __int64 **v577; // rdx
  __int64 *v578; // rax
  __int64 v579; // rax
  __int64 v580; // rax
  unsigned int v581; // r14d
  int v582; // eax
  bool v583; // al
  int v584; // eax
  __int64 **v585; // rcx
  unsigned int v586; // ebx
  __int64 v587; // rax
  char v588; // dl
  __int64 v589; // rax
  unsigned int v590; // ecx
  __int64 v591; // rax
  unsigned int v592; // ecx
  int v593; // eax
  bool v594; // al
  __int64 v595; // rax
  unsigned int v596; // edx
  int v597; // eax
  bool v598; // al
  __int64 v599; // rax
  __int64 v600; // rax
  __int64 **v601; // rax
  __int64 *v602; // r9
  __int64 **v603; // r10
  __int64 v604; // rdx
  unsigned __int8 *v605; // rax
  __int64 v606; // rbx
  int v607; // eax
  __int64 v608; // rdi
  _QWORD *v609; // rax
  int v610; // edx
  __int64 v611; // rdi
  _QWORD *v612; // r13
  double v613; // xmm4_8
  double v614; // xmm5_8
  __int64 v615; // rax
  __int64 v616; // rax
  __int64 v617; // rdx
  __int64 v618; // r13
  __int64 v619; // r15
  __int64 v620; // r12
  __int64 v621; // rbx
  _QWORD *v622; // rax
  double v623; // xmm4_8
  double v624; // xmm5_8
  __int64 v625; // rbx
  unsigned int v626; // r13d
  __int64 v627; // rax
  char v628; // si
  bool v629; // al
  __int64 v630; // rax
  __int64 v631; // rdi
  __int64 v632; // rsi
  __int64 v633; // rax
  __int64 v634; // rdx
  bool v635; // al
  __int64 v636; // rdx
  __int64 v637; // r8
  __int64 *v638; // rax
  __int64 v639; // rax
  __int64 v640; // rsi
  _QWORD *v641; // rtt
  __int64 v642; // rdx
  _QWORD *v643; // rt0
  __int64 v644; // rdi
  __int64 v645; // rsi
  _QWORD *v646; // rax
  __int64 v647; // rdx
  unsigned int v648; // ebx
  unsigned __int8 *v649; // rax
  __int64 *v650; // rbx
  __int64 *v651; // r13
  __int64 v652; // rsi
  int v653; // eax
  __int64 *v654; // r13
  _QWORD *v655; // rax
  __int64 v656; // rax
  __int64 v657; // rax
  int v658; // edx
  __int64 v659; // rax
  __int64 v660; // rsi
  char v661; // al
  __int64 v662; // rax
  unsigned __int64 v663; // rdi
  __int64 **v664; // rax
  __int64 *v665; // rsi
  __int64 v666; // rax
  __int64 v667; // r9
  __int64 v668; // rax
  unsigned __int64 v669; // rsi
  __int64 **v670; // rax
  __int64 v671; // rax
  char v672; // al
  void *v673; // rax
  __int64 v674; // r9
  __int64 v675; // rdx
  __int64 v676; // rcx
  __int64 v677; // rax
  __int64 v678; // rax
  __int64 v679; // rax
  int v680; // esi
  __int64 *v681; // rdi
  __int64 *v682; // rax
  __int64 v683; // rcx
  __int64 v684; // r9
  __int64 v685; // rax
  __int64 v686; // rax
  __int64 v687; // r14
  _BYTE *v688; // rdi
  _BYTE *v689; // rdi
  __int64 v690; // rbx
  __int64 v691; // rax
  __int64 v692; // rdx
  __int64 v693; // r13
  __int64 v694; // rdi
  _QWORD *v695; // rt1
  __int64 v696; // [rsp+0h] [rbp-150h]
  __int64 v697; // [rsp+28h] [rbp-128h]
  __int64 v698; // [rsp+30h] [rbp-120h]
  int v699; // [rsp+30h] [rbp-120h]
  unsigned int v700; // [rsp+30h] [rbp-120h]
  int v701; // [rsp+30h] [rbp-120h]
  __int64 v702; // [rsp+38h] [rbp-118h]
  __int64 v703; // [rsp+38h] [rbp-118h]
  __int64 v704; // [rsp+38h] [rbp-118h]
  __int64 v705; // [rsp+38h] [rbp-118h]
  __int64 v706; // [rsp+38h] [rbp-118h]
  int v707; // [rsp+38h] [rbp-118h]
  unsigned int v708; // [rsp+40h] [rbp-110h]
  __int64 v709; // [rsp+40h] [rbp-110h]
  unsigned __int64 v710; // [rsp+40h] [rbp-110h]
  int v711; // [rsp+40h] [rbp-110h]
  __int64 v712; // [rsp+40h] [rbp-110h]
  __int64 v713; // [rsp+40h] [rbp-110h]
  __int64 v714; // [rsp+40h] [rbp-110h]
  int v715; // [rsp+40h] [rbp-110h]
  int v716; // [rsp+40h] [rbp-110h]
  unsigned int v717; // [rsp+40h] [rbp-110h]
  _QWORD *v718; // [rsp+40h] [rbp-110h]
  unsigned int v719; // [rsp+40h] [rbp-110h]
  int v720; // [rsp+40h] [rbp-110h]
  unsigned int v721; // [rsp+40h] [rbp-110h]
  __int64 *v722; // [rsp+40h] [rbp-110h]
  __int64 *v723; // [rsp+40h] [rbp-110h]
  int v724; // [rsp+48h] [rbp-108h]
  _QWORD *v725; // [rsp+48h] [rbp-108h]
  __int64 *v726; // [rsp+48h] [rbp-108h]
  unsigned __int8 v727; // [rsp+48h] [rbp-108h]
  __int64 v728; // [rsp+48h] [rbp-108h]
  __int64 v729; // [rsp+48h] [rbp-108h]
  __int64 *v730; // [rsp+48h] [rbp-108h]
  __int64 v731; // [rsp+48h] [rbp-108h]
  __int64 v732; // [rsp+48h] [rbp-108h]
  __int32 v733; // [rsp+48h] [rbp-108h]
  int v734; // [rsp+48h] [rbp-108h]
  __int64 *v735; // [rsp+48h] [rbp-108h]
  unsigned int v736; // [rsp+48h] [rbp-108h]
  int v737; // [rsp+48h] [rbp-108h]
  __int64 v738; // [rsp+48h] [rbp-108h]
  __int64 v739; // [rsp+48h] [rbp-108h]
  __int64 v740; // [rsp+48h] [rbp-108h]
  __int64 v741; // [rsp+48h] [rbp-108h]
  int v742; // [rsp+48h] [rbp-108h]
  int v743; // [rsp+48h] [rbp-108h]
  int v744; // [rsp+48h] [rbp-108h]
  __int64 v745; // [rsp+48h] [rbp-108h]
  unsigned int v746; // [rsp+50h] [rbp-100h]
  char v747; // [rsp+50h] [rbp-100h]
  __int64 *v748; // [rsp+50h] [rbp-100h]
  __int64 v749; // [rsp+50h] [rbp-100h]
  unsigned int v750; // [rsp+50h] [rbp-100h]
  unsigned int v751; // [rsp+50h] [rbp-100h]
  __int64 v752; // [rsp+50h] [rbp-100h]
  unsigned int v753; // [rsp+50h] [rbp-100h]
  unsigned int v754; // [rsp+50h] [rbp-100h]
  unsigned int v755; // [rsp+50h] [rbp-100h]
  unsigned int v756; // [rsp+50h] [rbp-100h]
  unsigned int v757; // [rsp+50h] [rbp-100h]
  unsigned int v758; // [rsp+50h] [rbp-100h]
  unsigned int v759; // [rsp+50h] [rbp-100h]
  void *v760; // [rsp+50h] [rbp-100h]
  unsigned int v761; // [rsp+50h] [rbp-100h]
  __int64 v762; // [rsp+50h] [rbp-100h]
  __int64 v763; // [rsp+50h] [rbp-100h]
  unsigned int v764; // [rsp+50h] [rbp-100h]
  __int64 v765; // [rsp+50h] [rbp-100h]
  int v766; // [rsp+50h] [rbp-100h]
  unsigned int v767; // [rsp+50h] [rbp-100h]
  unsigned int v768; // [rsp+50h] [rbp-100h]
  int v769; // [rsp+50h] [rbp-100h]
  __int64 *v770; // [rsp+50h] [rbp-100h]
  __int64 **v771; // [rsp+50h] [rbp-100h]
  __int64 **v772; // [rsp+50h] [rbp-100h]
  __int64 v773; // [rsp+50h] [rbp-100h]
  __int64 v774; // [rsp+50h] [rbp-100h]
  int v775; // [rsp+50h] [rbp-100h]
  __int64 v776; // [rsp+50h] [rbp-100h]
  int v777; // [rsp+50h] [rbp-100h]
  __int64 *v778; // [rsp+50h] [rbp-100h]
  __int64 **v779; // [rsp+50h] [rbp-100h]
  int v780; // [rsp+50h] [rbp-100h]
  _BYTE *v781; // [rsp+50h] [rbp-100h]
  int v782; // [rsp+50h] [rbp-100h]
  __int64 ***v783; // [rsp+50h] [rbp-100h]
  int v784; // [rsp+50h] [rbp-100h]
  __int64 v785; // [rsp+58h] [rbp-F8h]
  __int64 *v786; // [rsp+58h] [rbp-F8h]
  __int64 v787; // [rsp+58h] [rbp-F8h]
  __int64 v788; // [rsp+58h] [rbp-F8h]
  __int64 v789; // [rsp+58h] [rbp-F8h]
  __int64 v790; // [rsp+58h] [rbp-F8h]
  _QWORD *v791; // [rsp+58h] [rbp-F8h]
  __int64 *v792; // [rsp+58h] [rbp-F8h]
  __int64 **v793; // [rsp+58h] [rbp-F8h]
  __int16 *v794; // [rsp+58h] [rbp-F8h]
  __int64 v795; // [rsp+58h] [rbp-F8h]
  __int64 v796; // [rsp+58h] [rbp-F8h]
  __int64 v797; // [rsp+58h] [rbp-F8h]
  __int64 v798; // [rsp+58h] [rbp-F8h]
  __int64 v799; // [rsp+58h] [rbp-F8h]
  __int64 v800; // [rsp+58h] [rbp-F8h]
  __int64 v801; // [rsp+58h] [rbp-F8h]
  unsigned int v802; // [rsp+58h] [rbp-F8h]
  __int64 v803; // [rsp+58h] [rbp-F8h]
  __int64 *v804; // [rsp+68h] [rbp-E8h] BYREF
  __int64 *v805; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v806; // [rsp+78h] [rbp-D8h] BYREF
  __int64 v807; // [rsp+80h] [rbp-D0h] BYREF
  __int64 *v808; // [rsp+88h] [rbp-C8h] BYREF
  __int64 *v809; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v810; // [rsp+98h] [rbp-B8h]
  __int64 v811; // [rsp+A0h] [rbp-B0h]
  char *v812; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 **v813; // [rsp+B8h] [rbp-98h] BYREF
  __int64 **v814; // [rsp+C0h] [rbp-90h]
  __m128i v815; // [rsp+D0h] [rbp-80h] BYREF
  __m128i v816; // [rsp+E0h] [rbp-70h] BYREF
  __int64 **v817; // [rsp+F0h] [rbp-60h]
  __int64 **v818; // [rsp+F8h] [rbp-58h]

  v10 = a2;
  v11 = a2;
  v12 = a1;
  v13 = _mm_loadu_si128((const __m128i *)(a1 + 2672));
  v14 = _mm_loadu_si128((const __m128i *)(a1 + 2688));
  v817 = (__int64 **)a2;
  v785 = a2 | 4;
  v815 = v13;
  v816 = v14;
  v15 = sub_13D3F90(a2 | 4, v815.m128i_i64);
  if ( v15 )
  {
    v16 = *(_QWORD *)(a2 + 8);
    if ( v16 )
    {
      v17 = *(_QWORD *)a1;
      v18 = v15;
      do
      {
        v19 = sub_1648700(v16);
        sub_170B990(v17, (__int64)v19);
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v16 );
      if ( a2 == v18 )
        v18 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v18, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v20, v21, a9, a10);
      return v11;
    }
    return 0;
  }
  v25 = sub_140B650(a2, *(_QWORD **)(a1 + 2648));
  if ( v25 )
    return sub_170BD90(a1, a2);
  v26 = 30;
  v27 = sub_15F2060(v10) + 112;
  if ( (unsigned __int8)sub_1560180(v27, 30) )
  {
    v26 = 0xFFFFFFFFLL;
    if ( !(unsigned __int8)sub_1560260((_QWORD *)(v10 + 56), -1, 30) )
    {
      v37 = *(_QWORD *)(v10 - 24);
      if ( *(_BYTE *)(v37 + 16)
        || (v26 = 0xFFFFFFFFLL, v815.m128i_i64[0] = *(_QWORD *)(v37 + 112), !(unsigned __int8)sub_1560260(&v815, -1, 30)) )
      {
        v815.m128i_i64[0] = *(_QWORD *)(v10 + 56);
        v38 = (__int64 *)sub_16498A0(v10);
        *(_QWORD *)(v10 + 56) = sub_1563AB0(v815.m128i_i64, v38, -1, 30);
        return v11;
      }
    }
  }
  v30 = *(_QWORD *)(v10 - 24);
  if ( *(_BYTE *)(v30 + 16) || (*(_BYTE *)(v30 + 33) & 0x20) == 0 )
    return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
  v31 = *(_DWORD *)(v30 + 36);
  v32 = v31 - 133;
  if ( (unsigned int)v32 > 5 )
    goto LABEL_40;
  v26 = 2LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
  v33 = *(_QWORD *)(v10 + 24 * v26);
  if ( *(_BYTE *)(v33 + 16) <= 0x10u )
  {
    if ( sub_1593BB0(v33, v26, v30, v32) )
      return sub_170BC50(v12, v10);
    v30 = *(_QWORD *)(v10 - 24);
    if ( *(_BYTE *)(v30 + 16) )
      goto LABEL_1104;
    v31 = *(_DWORD *)(v30 + 36);
    v32 = v31 - 133;
  }
  if ( (unsigned int)v32 <= 4 && ((1LL << v32) & 0x15) != 0 )
  {
    v34 = *(_QWORD *)(v10 + 24 * (3LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
    v32 = *(unsigned int *)(v34 + 32);
    if ( (unsigned int)v32 <= 0x40 )
    {
      v36 = *(_QWORD *)(v34 + 24) == 0;
    }
    else
    {
      v702 = v30;
      v708 = *(_DWORD *)(v34 + 32);
      v746 = v31;
      v35 = sub_16A57B0(v34 + 24);
      v32 = v708;
      v31 = v746;
      v30 = v702;
      v36 = v708 == v35;
    }
    if ( !v36 )
      return 0;
  }
  if ( v31 - 135 <= 1 )
  {
    v197 = sub_1649C60(*(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))));
    if ( *(_BYTE *)(v197 + 16) == 3 )
    {
      v198 = *(_BYTE *)(v197 + 80) & 1;
      v747 = v198;
      if ( v198 )
      {
        v199 = (__int64 *)sub_15F2050(v10);
        v200 = *(_QWORD *)(v10 - 24);
        if ( *(_BYTE *)(v200 + 16) )
          goto LABEL_1104;
        v201 = *(_DWORD *)(v200 + 36) == 136;
        v202 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v815.m128i_i64[0] = **(_QWORD **)(v10 - 24 * v202);
        v815.m128i_i64[1] = **(_QWORD **)(v10 + 24 * (1 - v202));
        v816.m128i_i64[0] = **(_QWORD **)(v10 + 24 * (2 - v202));
        v203 = (_QWORD *)sub_15E26F0(v199, (unsigned int)v201 + 133, v815.m128i_i64, 3);
        v201 = *(_QWORD *)(v10 - 24) == 0;
        v32 = v10 - 24;
        *(_QWORD *)(v10 + 64) = *(_QWORD *)(*v203 + 24LL);
        if ( !v201 )
        {
          v204 = *(_QWORD *)(v10 - 16);
          v205 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v205 = v204;
          if ( v204 )
            *(_QWORD *)(v204 + 16) = v205 | *(_QWORD *)(v204 + 16) & 3LL;
        }
        *(_QWORD *)(v10 - 24) = v203;
        v26 = v203[1];
        *(_QWORD *)(v10 - 16) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = (v10 - 16) | *(_QWORD *)(v26 + 16) & 3LL;
        *(_QWORD *)(v10 - 8) = (unsigned __int64)(v203 + 1) | *(_QWORD *)(v10 - 8) & 3LL;
        v203[1] = v32;
        v30 = *(_QWORD *)(v10 - 24);
        goto LABEL_29;
      }
    }
    v30 = *(_QWORD *)(v10 - 24);
  }
  v747 = 0;
LABEL_29:
  if ( *(_BYTE *)(v30 + 16) )
    goto LABEL_1104;
  if ( (unsigned int)(*(_DWORD *)(v30 + 36) - 133) <= 3 )
  {
    v709 = sub_1649C60(*(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))));
    if ( v709 == sub_1649C60(*(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))) )
      return sub_170BC50(v12, v10);
    if ( v747 )
      return v10;
    v30 = *(_QWORD *)(v10 - 24);
    if ( *(_BYTE *)(v30 + 16) )
      BUG();
  }
  else if ( v747 )
  {
    return v10;
  }
  v31 = *(_DWORD *)(v30 + 36);
LABEL_40:
  if ( v31 != 3846 )
  {
    if ( v31 != 3857 )
      goto LABEL_42;
    v812 = (char *)sub_1649960(v10);
    v816.m128i_i16[0] = 261;
    v190 = *(_DWORD *)(v10 + 20);
    v813 = v191;
    v192 = v190 & 0xFFFFFFF;
    v815.m128i_i64[0] = (__int64)&v812;
    v26 = *(_QWORD *)(v10 - 24 * v192);
    v153 = sub_15FB440(19, (__int64 *)v26, *(_QWORD *)(v10 + 24 * (1 - v192)), (__int64)&v815, 0);
    goto LABEL_271;
  }
  v32 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v193 = *(_QWORD **)(v32 + 24);
  if ( *(_DWORD *)(v32 + 32) > 0x40u )
    v193 = (_QWORD *)*v193;
  if ( ((unsigned __int8)v193 & 7) == 1 && *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 3 )
  {
    v194 = (__int64 *)sub_1649960(v10);
    v813 = v195;
    LODWORD(v195) = *(_DWORD *)(v10 + 20);
    v812 = (char *)v194;
    v196 = (unsigned int)v195 & 0xFFFFFFF;
    v816.m128i_i16[0] = 261;
    v815.m128i_i64[0] = (__int64)&v812;
    v26 = *(_QWORD *)(v10 + 24 * (1 - v196));
    v153 = sub_15FB440(19, (__int64 *)v26, *(_QWORD *)(v10 + 24 * (2 - v196)), (__int64)&v815, 0);
LABEL_271:
    if ( v153 )
      return v153;
    v30 = *(_QWORD *)(v10 - 24);
  }
  if ( *(_BYTE *)(v30 + 16) )
    goto LABEL_1104;
LABEL_42:
  v39 = *(_DWORD *)(v30 + 36);
  if ( v39 > 0x1D8 )
  {
    if ( v39 <= 0xF72 )
    {
      if ( v39 > 0xF22 )
      {
        switch ( v39 )
        {
          case 0xF23u:
            v258 = *(_BYTE **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
            v259 = v258[16];
            if ( v259 == 14 )
            {
              v260 = (__int64)(v258 + 24);
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v258 + 8LL) != 16 )
                return sub_1742F80(
                         v12,
                         v785,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v28,
                         v29,
                         a9,
                         a10);
              if ( v259 > 0x10u )
                return sub_1742F80(
                         v12,
                         v785,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v28,
                         v29,
                         a9,
                         a10);
              v600 = sub_15A1020(v258, v26, *(_QWORD *)v258, v32);
              if ( !v600 || *(_BYTE *)(v600 + 16) != 14 )
                return sub_1742F80(
                         v12,
                         v785,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v28,
                         v29,
                         a9,
                         a10);
              v260 = v600 + 24;
            }
            v261 = sub_173D800(v260);
            v262 = *(_QWORD *)v10;
            if ( v261 )
              v67 = sub_15A0600(v262);
            else
              v67 = sub_15A0640(v262);
            return sub_170E100(
                     (__int64 *)v12,
                     v10,
                     v67,
                     a3,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64,
                     a6,
                     v28,
                     v29,
                     a9,
                     a10);
          case 0xF46u:
          case 0xF47u:
          case 0xF48u:
          case 0xF49u:
          case 0xF4Au:
            goto LABEL_119;
          case 0xF57u:
            v249 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
            v250 = *(_QWORD *)(v10 + 24 * (1 - v249));
            v251 = 24 * (2 - v249);
            v809 = (__int64 *)v250;
            v252 = *(_QWORD *)(v10 + v251);
            if ( *(_BYTE *)(v250 + 16) <= 0x10u && *(_BYTE *)(v252 + 16) > 0x10u )
            {
              v809 = *(__int64 **)(v10 + v251);
              v252 = v250;
            }
            v815.m128i_i64[0] = 0x3FF0000000000000LL;
            if ( !(unsigned __int8)sub_13D6AF0((double *)v815.m128i_i64, v252) )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v253 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v10 + 40) + 56LL) + 40LL);
            v815.m128i_i64[0] = *(_QWORD *)v10;
            v254 = sub_15E26F0(v253, 3640, v815.m128i_i64, 1);
            v815.m128i_i64[0] = (__int64)&v816;
            v815.m128i_i64[1] = 0x300000000LL;
            v812 = *(char **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
            sub_12A9700((__int64)&v815, &v812);
            sub_12A9700((__int64)&v815, &v809);
            v812 = *(char **)(v10 + 24 * (3LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
            sub_12A9700((__int64)&v815, &v812);
            LOWORD(v814) = 257;
            v726 = (__int64 *)v815.m128i_i64[0];
            v788 = v815.m128i_u32[2];
            v711 = v815.m128i_i32[2] + 1;
            v255 = *(_QWORD *)(*(_QWORD *)v254 + 24LL);
            v256 = sub_1648AB0(72, v815.m128i_i32[2] + 1, 0);
            v11 = (__int64)v256;
            if ( v256 )
            {
              sub_15F1EA0((__int64)v256, **(_QWORD **)(v255 + 16), 54, (__int64)&v256[-3 * v788 - 3], v711, 0);
              *(_QWORD *)(v11 + 56) = 0;
              sub_15F5B40(v11, v255, v254, v726, v788, (__int64)&v812, 0, 0);
            }
            sub_15F2500(v11, v10);
            v257 = v815.m128i_i64[0];
            if ( (__m128i *)v815.m128i_i64[0] != &v816 )
LABEL_365:
              _libc_free(v257);
            return v11;
          case 0xF68u:
          case 0xF6Au:
          case 0xF6Cu:
            goto LABEL_165;
          case 0xF6Eu:
          case 0xF70u:
          case 0xF72u:
            goto LABEL_148;
          default:
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
      }
      if ( v39 != 990 )
      {
        if ( v39 > 0x3DE )
        {
          if ( v39 == 1004 )
          {
            v67 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v67 + 16) > 0x10u )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            return sub_170E100(
                     (__int64 *)v12,
                     v10,
                     v67,
                     a3,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64,
                     a6,
                     v28,
                     v29,
                     a9,
                     a10);
          }
          if ( v39 <= 0x3EC )
          {
            if ( v39 == 995 )
            {
              v110 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
              v111 = *(_QWORD *)(v10 + 24 * (5 - v110));
              if ( *(_BYTE *)(v111 + 16) != 13 )
                v111 = 0;
              v112 = *(_QWORD *)(v10 + 24 * (3 - v110));
              v113 = *(_QWORD *)(v10 + 24 * (4 - v110));
              if ( *(_BYTE *)(v112 + 16) == 13 && *(_BYTE *)(v113 + 16) == 13 )
              {
                if ( v111 )
                {
                  v114 = -24 * v110;
                  v115 = *(_QWORD *)(v10 + v114);
                  if ( !sub_1595F50(v111, v26, v114, v32) )
                  {
                    v116 = *(_QWORD **)(v112 + 24);
                    if ( *(_DWORD *)(v112 + 32) > 0x40u )
                      v116 = (_QWORD *)*v116;
                    if ( v116 == (_QWORD *)15 )
                    {
                      v117 = *(_QWORD **)(v113 + 24);
                      if ( *(_DWORD *)(v113 + 32) > 0x40u )
                        v117 = (_QWORD *)*v117;
                      if ( v117 == (_QWORD *)15 && *(_BYTE *)(v115 + 16) != 9 )
                      {
                        v118 = sub_1599EF0(*(__int64 ***)v115);
                        sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), v118);
                        return v10;
                      }
                    }
                  }
                }
              }
            }
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
          if ( v39 > 0x40F )
          {
            if ( v39 - 3637 <= 2 )
            {
LABEL_119:
              v105 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
              if ( *(_BYTE *)(v105 + 16) == 37 )
              {
                v106 = *(_QWORD *)(v105 - 48);
                if ( *(_BYTE *)(v106 + 16) == 13 && sub_13D01C0(v106 + 24) && *(_QWORD *)(v105 - 24) )
                {
                  v809 = *(__int64 **)(v105 - 24);
                  v812 = 0;
                  v813 = 0;
                  v814 = 0;
                  sub_1287830((__int64)&v812, 0, &v809);
                  v816.m128i_i16[0] = 259;
                  v815.m128i_i64[0] = (__int64)"abs";
                  if ( !*(_BYTE *)(*(_QWORD *)(v10 - 24) + 16LL) )
                    v25 = *(_QWORD *)(v10 - 24);
                  v107 = sub_17287F0(v25, (__int64 *)v812, ((char *)v813 - v812) >> 3, (__int64)&v815, v10);
                  v11 = sub_170E100(
                          (__int64 *)v12,
                          v10,
                          v107,
                          a3,
                          *(double *)v13.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a6,
                          v108,
                          v109,
                          a9,
                          a10);
                  if ( v812 )
                    j_j___libc_free_0(v812, (char *)v814 - v812);
                  return v11;
                }
              }
            }
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
          if ( v39 <= 0x40D )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
LABEL_108:
          v95 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
          v96 = *(_QWORD *)(v10 - 24 * v95);
          v97 = *(_BYTE **)(v10 + 24 * (1 - v95));
          v98 = v97[16];
          if ( v98 == 13 )
          {
            v99 = *((_DWORD *)v97 + 8);
            if ( v99 <= 0x40 )
              v100 = *((_QWORD *)v97 + 3) == 0;
            else
              v100 = v99 == (unsigned int)sub_16A57B0((__int64)(v97 + 24));
          }
          else
          {
            if ( *(_BYTE *)(*(_QWORD *)v97 + 8LL) != 16 || v98 > 0x10u )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v500 = sub_15A1020(v97, v26, *(_QWORD *)v97, v32);
            if ( !v500 || *(_BYTE *)(v500 + 16) != 13 )
            {
              v503 = 0;
              v775 = *(_DWORD *)(*(_QWORD *)v97 + 32LL);
              while ( v775 != v503 )
              {
                v504 = sub_15A0A60((__int64)v97, v503);
                if ( !v504 )
                  return sub_1742F80(
                           v12,
                           v785,
                           a3,
                           *(double *)v13.m128i_i64,
                           *(double *)v14.m128i_i64,
                           a6,
                           v28,
                           v29,
                           a9,
                           a10);
                v505 = *(_BYTE *)(v504 + 16);
                if ( v505 != 9 )
                {
                  if ( v505 != 13 )
                    return sub_1742F80(
                             v12,
                             v785,
                             a3,
                             *(double *)v13.m128i_i64,
                             *(double *)v14.m128i_i64,
                             a6,
                             v28,
                             v29,
                             a9,
                             a10);
                  if ( *(_DWORD *)(v504 + 32) <= 0x40u )
                  {
                    v506 = *(_QWORD *)(v504 + 24) == 0;
                  }
                  else
                  {
                    v737 = *(_DWORD *)(v504 + 32);
                    v506 = v737 == (unsigned int)sub_16A57B0(v504 + 24);
                  }
                  if ( !v506 )
                    return sub_1742F80(
                             v12,
                             v785,
                             a3,
                             *(double *)v13.m128i_i64,
                             *(double *)v14.m128i_i64,
                             a6,
                             v28,
                             v29,
                             a9,
                             a10);
                }
                ++v503;
              }
              goto LABEL_112;
            }
            v501 = *(_DWORD *)(v500 + 32);
            if ( v501 <= 0x40 )
              v100 = *(_QWORD *)(v500 + 24) == 0;
            else
              v100 = v501 == (unsigned int)sub_16A57B0(v500 + 24);
          }
          if ( !v100 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
LABEL_112:
          v101 = *(_BYTE *)(v96 + 16);
          if ( v101 == 52 )
          {
            v103 = *(_QWORD *)(v96 - 48);
            if ( !v103 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v104 = *(_QWORD *)(v96 - 24);
            if ( !v104 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
          }
          else
          {
            if ( v101 != 5 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            if ( *(_WORD *)(v96 + 18) != 28 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v102 = *(_DWORD *)(v96 + 20) & 0xFFFFFFF;
            v103 = *(_QWORD *)(v96 - 24 * v102);
            if ( !v103 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v104 = *(_QWORD *)(v96 + 24 * (1 - v102));
            if ( !v104 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
          }
          v11 = v10;
          sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), v103);
          sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v104);
          return v11;
        }
        if ( v39 == 955 )
        {
          v67 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
          v206 = *(_BYTE *)(v67 + 16);
          if ( v206 == 9 )
            return sub_170E100(
                     (__int64 *)v12,
                     v10,
                     v67,
                     a3,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64,
                     a6,
                     v28,
                     v29,
                     a9,
                     a10);
          if ( v206 != 14 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v207 = v67 + 24;
          v760 = *(void **)(v67 + 32);
          if ( v760 == sub_16982C0() )
            sub_169C630(&v815.m128i_i64[1], (__int64)v760, 1);
          else
            sub_1699170((__int64)&v815.m128i_i64[1], (__int64)v760, 1);
          if ( (unsigned int)sub_173D6E0((__int64)&v815, v207, 0, a3.m128_f32[0]) )
          {
            sub_127D120(&v815.m128i_i64[1]);
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
          v208 = (_QWORD *)sub_16498A0(v10);
LABEL_301:
          v209 = sub_159CCF0(v208, (__int64)&v815);
          v11 = sub_170E100(
                  (__int64 *)v12,
                  v10,
                  v209,
                  a3,
                  *(double *)v13.m128i_i64,
                  *(double *)v14.m128i_i64,
                  a6,
                  v210,
                  v211,
                  a9,
                  a10);
          sub_127D120(&v815.m128i_i64[1]);
          return v11;
        }
        if ( v39 <= 0x3BB )
        {
          if ( v39 != 941 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v188 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
          if ( *(_BYTE *)(v188 + 16) != 13 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v189 = *(_QWORD **)(v188 + 24);
          if ( *(_DWORD *)(v188 + 32) > 0x40u )
            v189 = (_QWORD *)*v189;
          if ( !v189 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          return sub_170BC50(v12, v10);
        }
        if ( v39 == 959 )
        {
          v67 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
          if ( *(_BYTE *)(v67 + 16) != 9 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          return sub_170E100(
                   (__int64 *)v12,
                   v10,
                   v67,
                   a3,
                   *(double *)v13.m128i_i64,
                   *(double *)v14.m128i_i64,
                   a6,
                   v28,
                   v29,
                   a9,
                   a10);
        }
        if ( v39 != 980 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      }
      v171 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v120 = *(_QWORD *)(v10 - 24 * v171);
      if ( *(_BYTE *)(v120 + 16) != 9 )
      {
        v172 = *(_DWORD *)(*(_QWORD *)v10 + 8LL) >> 8;
        v173 = *(_QWORD *)(v10 + 24 * (2 - v171));
        if ( *(_BYTE *)(v173 + 16) != 13 )
          goto LABEL_254;
        v174 = *(_QWORD **)(v173 + 24);
        if ( *(_DWORD *)(v173 + 32) > 0x40u )
          v174 = (_QWORD *)*v174;
        v724 = (int)v174;
        v175 = (unsigned int)v174 & (v172 - 1);
        if ( ((unsigned int)v174 & (v172 - 1)) == 0 )
        {
          v67 = sub_15A06D0(*(__int64 ***)v10, v26, (__int64)v174, v175);
          return sub_170E100(
                   (__int64 *)v12,
                   v10,
                   v67,
                   a3,
                   *(double *)v13.m128i_i64,
                   *(double *)v14.m128i_i64,
                   a6,
                   v28,
                   v29,
                   a9,
                   a10);
        }
        v25 = *(_QWORD *)(v10 + 24 * (2 - v171));
        if ( (unsigned int)v174 >= v172 )
        {
          v630 = sub_159C470(*(_QWORD *)v173, (unsigned int)v175, 0);
          v631 = 2;
          v632 = v630;
          v633 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        }
        else
        {
LABEL_254:
          v176 = *(_QWORD *)(v10 + 24 * (1 - v171));
          if ( *(_BYTE *)(v176 + 16) != 13 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v177 = *(_QWORD *)(v176 + 24);
          if ( *(_DWORD *)(v176 + 32) > 0x40u )
            v177 = *(_QWORD *)v177;
          if ( (unsigned int)v177 < v172 )
          {
            if ( !v25 )
            {
              if ( (_DWORD)v177 )
                return sub_1742F80(
                         v12,
                         v785,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v28,
                         v29,
                         a9,
                         a10);
              v759 = v39;
              v178 = sub_159C470(*(_QWORD *)v176, v172, 0);
              v179 = *(_QWORD *)(v12 + 8);
              v180 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
              v816.m128i_i16[0] = 257;
              v181 = sub_171D0D0(
                       v179,
                       v178,
                       *(_QWORD *)(v10 + 24 * (2 - v180)),
                       v815.m128i_i64,
                       0,
                       0,
                       *(double *)a3.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64);
              v182 = *(_QWORD *)(v12 + 8);
              v183 = *(__int64 ***)v10;
              v816.m128i_i16[0] = 257;
              v184 = sub_1708970(v182, 37, (__int64)v181, v183, v815.m128i_i64);
              v185 = *(_QWORD *)(v12 + 8);
              v816.m128i_i16[0] = 257;
              v787 = (__int64)v184;
              v186 = sub_173DC60(
                       v185,
                       v120,
                       (__int64)v184,
                       v815.m128i_i64,
                       0,
                       0,
                       *(double *)a3.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64);
              v816.m128i_i16[0] = 257;
              v187 = *(_QWORD *)(v12 + 8);
              if ( v759 == 980 )
                v121 = sub_173DE00(
                         v187,
                         (__int64)v186,
                         v787,
                         v815.m128i_i64,
                         0,
                         *(double *)a3.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64);
              else
                v121 = sub_172C310(
                         v187,
                         (__int64)v186,
                         v787,
                         v815.m128i_i64,
                         0,
                         *(double *)a3.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64);
              goto LABEL_262;
            }
            v644 = *(_QWORD *)(v12 + 8);
            if ( (int)v177 + v724 < v172 )
            {
              v802 = v39;
              v648 = v172 - v724;
              v816.m128i_i16[0] = 257;
              v649 = sub_173E3E0(
                       v644,
                       v120,
                       v172 - v724 - (unsigned int)v177,
                       v815.m128i_i64,
                       0,
                       0,
                       *(double *)a3.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64);
              v816.m128i_i16[0] = 257;
              v177 = v648;
              v644 = *(_QWORD *)(v12 + 8);
              v645 = (__int64)v649;
              if ( v802 != 980 )
                goto LABEL_967;
            }
            else
            {
              v816.m128i_i16[0] = 257;
              v177 = (unsigned int)v177;
              v645 = v120;
              if ( v39 != 980 )
              {
LABEL_967:
                v646 = sub_173E800(
                         v644,
                         v645,
                         v177,
                         v815.m128i_i64,
                         0,
                         *(double *)a3.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64);
LABEL_968:
                v121 = v646;
LABEL_262:
                sub_164B7C0((__int64)v121, v10);
LABEL_263:
                v67 = (__int64)v121;
                return sub_170E100(
                         (__int64 *)v12,
                         v10,
                         v67,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v28,
                         v29,
                         a9,
                         a10);
              }
            }
            v121 = sub_173E590(
                     v644,
                     v645,
                     v177,
                     v815.m128i_i64,
                     0,
                     *(double *)a3.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64);
            goto LABEL_262;
          }
          v631 = 1;
          v632 = sub_159C470(*(_QWORD *)v176, (unsigned int)v177 & (v172 - 1), 0);
          v633 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        }
        sub_1593B40((_QWORD *)(v10 + 24 * (v631 - v633)), v632);
        return v10;
      }
LABEL_448:
      v67 = v120;
      return sub_170E100(
               (__int64 *)v12,
               v10,
               v67,
               a3,
               *(double *)v13.m128i_i64,
               *(double *)v14.m128i_i64,
               a6,
               v28,
               v29,
               a9,
               a10);
    }
    if ( v39 == 4209 )
    {
      v220 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v221 = *(_QWORD *)(v10 - 24 * v220);
      if ( *(_BYTE *)(v221 + 16) != 13 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v222 = *(_QWORD *)(v10 + 24 * (1 - v220));
      if ( *(_BYTE *)(v222 + 16) != 13 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v223 = *(_DWORD *)(v221 + 32);
      v224 = *(__int64 **)(v221 + 24);
      if ( v223 > 0x40 )
        v225 = *v224;
      else
        v225 = (__int64)((_QWORD)v224 << (64 - (unsigned __int8)v223)) >> (64 - (unsigned __int8)v223);
      v226 = *(_DWORD *)(v222 + 32);
      v227 = v225;
      v228 = *(__int64 **)(v222 + 24);
      v229 = v225 & 0xFFFFFF;
      if ( v226 > 0x40 )
        v230 = *v228;
      else
        v230 = (__int64)((_QWORD)v228 << (64 - (unsigned __int8)v226)) >> (64 - (unsigned __int8)v226);
      v231 = v230;
      v232 = v230 & 0xFFFFFF;
      if ( (v227 & 0x800000) != 0 )
        v229 |= 0xFF000000;
      if ( (v231 & 0x800000) != 0 )
        v232 |= 0xFF000000;
      v151 = (__int64 *)sub_159C580(*(_QWORD *)v221, v229);
      v152 = sub_159C580(*(_QWORD *)v222, v232);
LABEL_213:
      v816.m128i_i16[0] = 257;
      return sub_15FB440(15, v151, v152, (__int64)&v815, 0);
    }
    if ( v39 > 0x1071 )
    {
      if ( v39 == 4228 )
      {
        if ( *(_BYTE *)(v12 + 2729) )
        {
          v233 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
          if ( !(unsigned __int8)sub_173D4E0(v233, v26, v30, v32, v31) )
            v233 = 0;
          v234 = *(_QWORD *)sub_13CF970(v233);
          v201 = !sub_173D4D0(v234);
          v236 = *(_QWORD *)(v235 + 48);
          if ( v201 )
            v234 = 0;
          if ( *(_BYTE *)(v236 + 16) != 13 )
LABEL_1106:
            BUG();
          if ( *(_DWORD *)(v236 + 32) <= 0x40u )
            v725 = *(_QWORD **)(v236 + 24);
          else
            v725 = **(_QWORD ***)(v236 + 24);
          v237 = *(_QWORD *)(v10 + 40);
          v238 = 1;
          v815.m128i_i64[0] = (__int64)&v816;
          v815.m128i_i64[1] = 0x800000000LL;
          v703 = v12;
          v710 = sub_157EBA0(v237);
          v698 = v10;
          for ( i = *(_QWORD *)(v10 + 32); ; i = *(_QWORD *)(i + 8) )
          {
            v242 = 0;
            if ( i )
              v242 = i - 24;
            if ( v710 == v242 )
              break;
            if ( *(_BYTE *)(v242 + 16) == 78 )
            {
              v243 = *(_QWORD *)(v242 - 24);
              if ( !*(_BYTE *)(v243 + 16) && (*(_BYTE *)(v243 + 33) & 0x20) != 0 && *(_DWORD *)(v243 + 36) == 4228 )
              {
                v244 = *(_QWORD *)(v242 - 24LL * (*(_DWORD *)(v242 + 20) & 0xFFFFFFF));
                if ( !(unsigned __int8)sub_173D4E0(v244, v26, v239, v240, (int)i - 24) )
                  v244 = 0;
                v245 = (__int64 *)sub_13CF970(v244);
                if ( !sub_173D4D0(*v245) )
                  v240 = 0;
                if ( v234 == v240 )
                {
                  v239 = *(_QWORD *)(v239 + 48);
                  if ( *(_BYTE *)(v239 + 16) != 13 )
                    goto LABEL_1106;
                  v248 = *(_QWORD **)(v239 + 24);
                  if ( *(_DWORD *)(v239 + 32) > 0x40u )
                    v248 = (_QWORD *)*v248;
                  if ( v725 == v248 )
                  {
                    ++v238;
                    if ( v815.m128i_i32[2] >= (unsigned __int32)v815.m128i_i32[3] )
                    {
                      v26 = (__int64)&v816;
                      sub_16CD150((__int64)&v815, &v816, 0, 8, v246, v247);
                    }
                    v239 = v815.m128i_u32[2];
                    *(_QWORD *)(v815.m128i_i64[0] + 8LL * v815.m128i_u32[2]) = v242;
                    ++v815.m128i_i32[2];
                  }
                }
              }
            }
          }
          v12 = v703;
          if ( v238 == 1 )
          {
            if ( (__m128i *)v815.m128i_i64[0] != &v816 )
              _libc_free(v815.m128i_u64[0]);
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
          v650 = (__int64 *)v815.m128i_i64[0];
          v651 = (__int64 *)(v815.m128i_i64[0] + 8LL * v815.m128i_u32[2]);
          if ( v651 != (__int64 *)v815.m128i_i64[0] )
          {
            do
            {
              v652 = *v650++;
              sub_170BC50(v703, v652);
            }
            while ( v651 != v650 );
          }
          v653 = *(_DWORD *)(v698 + 20);
          v654 = *(__int64 **)(v703 + 8);
          LOWORD(v814) = 257;
          v809 = *(__int64 **)(v698 - 24LL * (v653 & 0xFFFFFFF));
          v655 = (_QWORD *)sub_16498A0(v698);
          v656 = sub_1643360(v655);
          v657 = sub_159C470(v656, v238, 0);
          v658 = *(_DWORD *)(v698 + 20);
          v810 = v657;
          v811 = *(_QWORD *)(v698 + 24 * (2LL - (v658 & 0xFFFFFFF)));
          v659 = *(_QWORD *)(v698 - 24);
          if ( *(_BYTE *)(v659 + 16) )
            goto LABEL_1104;
          sub_15E8450(v654, *(_DWORD *)(v659 + 36), &v809, 3, 0, (int)&v812);
          v25 = sub_170BC50(v703, v698);
          if ( (__m128i *)v815.m128i_i64[0] != &v816 )
            _libc_free(v815.m128i_u64[0]);
        }
        return v25;
      }
      if ( v39 == 4389 )
      {
        v135 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v136 = *(_QWORD *)(v10 + 24 * (1 - v135));
        if ( *(_BYTE *)(v136 + 16) > 0x10u )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v137 = *(_QWORD *)(v10 + 24 * (2 - v135));
        if ( *(_BYTE *)(v137 + 16) > 0x10u )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v138 = *(_QWORD *)(v10 + 24 * (3 - v135));
        if ( *(_BYTE *)(v138 + 16) > 0x10u )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v139 = *(_QWORD *)(v10 - 24 * v135);
        v140 = *(_QWORD **)(v139 + 24);
        if ( *(_DWORD *)(v139 + 32) > 0x40u )
          v140 = (_QWORD *)*v140;
        v141 = *(__int64 ***)v10;
        v752 = v136;
        v142 = sub_1643030(*(_QWORD *)v10);
        v143 = *(_QWORD *)(v138 + 24);
        if ( *(_DWORD *)(v138 + 32) > 0x40u )
          v143 = *(_QWORD *)v143;
        if ( ((unsigned __int8)v140 & 2) != 0 )
        {
          v143 = (v142 - 1) & (unsigned int)v143;
        }
        else if ( v142 <= v143 )
        {
          v143 = v142;
        }
        v144 = *(_QWORD *)(v752 + 24);
        if ( *(_DWORD *)(v752 + 32) > 0x40u )
          v144 = **(_QWORD **)(v752 + 24);
        v145 = *(_QWORD *)(v137 + 24);
        if ( *(_DWORD *)(v137 + 32) > 0x40u )
          v145 = *(_QWORD *)v145;
        if ( ((unsigned __int8)v140 & 1) != 0 )
        {
          if ( v143 )
          {
            if ( v143 - 1 <= 0x3E )
              v145 = (v144 >> v143) | (v145 << ((unsigned __int8)v142 - (unsigned __int8)v143));
            goto LABEL_204;
          }
        }
        else
        {
          if ( !v143 )
          {
LABEL_204:
            v67 = sub_15A0680((__int64)v141, v145, 0);
            return sub_170E100(
                     (__int64 *)v12,
                     v10,
                     v67,
                     a3,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64,
                     a6,
                     v28,
                     v29,
                     a9,
                     a10);
          }
          if ( v143 - 1 <= 0x3E )
          {
            v145 = (v144 >> ((unsigned __int8)v142 - (unsigned __int8)v143)) | (v145 << v143);
            goto LABEL_204;
          }
        }
        v145 = v144;
        goto LABEL_204;
      }
      if ( v39 != 4210 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v146 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v147 = *(_QWORD *)(v10 - 24 * v146);
      if ( *(_BYTE *)(v147 + 16) != 13 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v148 = *(_QWORD *)(v10 + 24 * (1 - v146));
      if ( *(_BYTE *)(v148 + 16) != 13 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v149 = *(_QWORD **)(v147 + 24);
      if ( *(_DWORD *)(v147 + 32) > 0x40u )
        v149 = (_QWORD *)*v149;
      v150 = *(_QWORD **)(v148 + 24);
      if ( *(_DWORD *)(v148 + 32) > 0x40u )
        v150 = (_QWORD *)*v150;
      v151 = (__int64 *)sub_159C470(*(_QWORD *)v147, (unsigned int)v149 & 0xFFFFFF, 0);
      v152 = sub_159C470(*v151, (unsigned int)v150 & 0xFFFFFF, 0);
      goto LABEL_213;
    }
    if ( v39 > 0x1049 )
    {
      if ( v39 - 4170 > 2 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v163 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v120 = *(_QWORD *)(v10 + 24 * (1 - v163));
      if ( *(_BYTE *)(v120 + 16) == 13 )
      {
        v756 = *(_DWORD *)(v30 + 36);
        v164 = sub_13D01C0(v120 + 24);
        v39 = v756;
        if ( v164 )
          goto LABEL_448;
      }
      v121 = *(_QWORD **)(v10 - 24 * v163);
      if ( v121 )
      {
        if ( *((_BYTE *)v121 + 16) == 13 )
        {
          v757 = v39;
          v165 = sub_13D01C0((__int64)(v121 + 3));
          v39 = v757;
          if ( v165 )
            goto LABEL_263;
        }
        if ( (_QWORD *)v120 == v121 )
          goto LABEL_448;
        v758 = v39;
        v166 = sub_173DBE0((__int64)v121);
        v167 = v758;
        if ( !v166 )
        {
          if ( !sub_173DBE0(v120) )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v167 = v758;
          v695 = (_QWORD *)v120;
          v120 = (__int64)v121;
          v121 = v695;
        }
        v168 = *(v121 - 3);
        if ( *(_BYTE *)(v168 + 16) )
          goto LABEL_1104;
        v169 = *(_DWORD *)(v168 + 36);
        if ( v167 != v169 )
        {
          if ( (unsigned int)(v169 - 4117) > 2 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v170 = *((_DWORD *)v121 + 5) & 0xFFFFFFF;
          if ( v120 != v121[-3 * v170] && v120 != v121[3 * (1 - v170)] )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
LABEL_246:
          sub_170E100(
            (__int64 *)v12,
            v10,
            v120,
            a3,
            *(double *)v13.m128i_i64,
            *(double *)v14.m128i_i64,
            a6,
            v28,
            v29,
            a9,
            a10);
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
LABEL_926:
        v634 = *((_DWORD *)v121 + 5) & 0xFFFFFFF;
        if ( v120 != v121[-3 * v634] && v120 != v121[3 * (1 - v634)] )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        goto LABEL_263;
      }
LABEL_935:
      BUG();
    }
    if ( v39 <= 0x1046 )
    {
      if ( v39 > 0x1014 )
      {
        if ( v39 - 4117 > 2 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v154 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v120 = *(_QWORD *)(v10 - 24 * v154);
        v121 = *(_QWORD **)(v10 + 24 * (1 - v154));
        if ( *((_BYTE *)v121 + 16) == 13 )
        {
          v753 = *(_DWORD *)(v30 + 36);
          v155 = sub_13D01C0((__int64)(v121 + 3));
          v39 = v753;
          if ( v155 )
            goto LABEL_448;
        }
        if ( v120 )
        {
          if ( *(_BYTE *)(v120 + 16) == 13 )
          {
            v754 = v39;
            v156 = sub_13D01C0(v120 + 24);
            v39 = v754;
            if ( v156 )
              goto LABEL_263;
          }
          if ( v121 == (_QWORD *)v120 )
            goto LABEL_263;
          v755 = v39;
          v157 = sub_173DBE0(v120);
          v158 = v755;
          if ( v157 )
          {
            v159 = (_QWORD *)v120;
            v120 = (__int64)v121;
            v121 = v159;
          }
          else
          {
            v635 = sub_173DBE0((__int64)v121);
            v158 = v755;
            if ( !v635 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
          }
          v160 = *(v121 - 3);
          if ( *(_BYTE *)(v160 + 16) )
            goto LABEL_1104;
          v161 = *(_DWORD *)(v160 + 36);
          if ( v161 != v158 )
          {
            if ( (unsigned int)(v161 - 4170) > 2 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v162 = *((_DWORD *)v121 + 5) & 0xFFFFFFF;
            if ( v120 != v121[-3 * v162] && v120 != v121[3 * (1 - v162)] )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            goto LABEL_246;
          }
          goto LABEL_926;
        }
        goto LABEL_935;
      }
      if ( v39 <= 0x1011 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
LABEL_165:
      v128 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v121 = *(_QWORD **)(v10 - 24 * v128);
      if ( !v121 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v120 = *(_QWORD *)(v10 + 24 * (1 - v128));
      if ( !v120 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      if ( v121 == (_QWORD *)v120 )
        goto LABEL_263;
      v751 = v39;
      v129 = sub_173DBE0(*(_QWORD *)(v10 - 24 * v128));
      v130 = v751;
      if ( !v129 )
      {
        if ( !sub_173DBE0(v120) )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v130 = v751;
        v641 = (_QWORD *)v120;
        v120 = (__int64)v121;
        v121 = v641;
      }
      v131 = *(v121 - 3);
      if ( *(_BYTE *)(v131 + 16) )
        goto LABEL_1104;
      v132 = *(_DWORD *)(v131 + 36);
      if ( v130 == v132 )
      {
        v647 = *((_DWORD *)v121 + 5) & 0xFFFFFFF;
        if ( v120 == v121[-3 * v647] || v120 == v121[3 * (1 - v647)] )
          goto LABEL_263;
        if ( v130 > 0xF72 || v130 <= 0xF6D || ((1LL << ((unsigned __int8)v130 - 110)) & 0x15) == 0 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      }
      else
      {
        if ( v132 > 0xF72 )
        {
          if ( v132 - 4167 > 2 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          goto LABEL_179;
        }
        if ( v132 <= 0xF6D || ((1LL << ((unsigned __int8)v132 - 110)) & 0x15) == 0 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      }
      if ( !sub_173DB60(v120) || (v133 = *(_BYTE *)(v120 + 17) >> 1, v133 != 127) && (v133 & 2) == 0 )
      {
        if ( *(_BYTE *)(v120 + 16) != 14 || sub_173D800(v120 + 24) )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      }
LABEL_179:
      v134 = *((_DWORD *)v121 + 5) & 0xFFFFFFF;
      if ( v120 != v121[-3 * v134] && v120 != v121[3 * (1 - v134)] )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      goto LABEL_246;
    }
LABEL_148:
    v119 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
    v120 = *(_QWORD *)(v10 - 24 * v119);
    if ( !v120 )
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    v121 = *(_QWORD **)(v10 + 24 * (1 - v119));
    if ( !v121 )
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    if ( v121 == (_QWORD *)v120 )
      goto LABEL_263;
    v750 = v39;
    v122 = sub_173DBE0(*(_QWORD *)(v10 - 24 * v119));
    v123 = v750;
    if ( !v122 )
    {
      if ( !sub_173DBE0((__int64)v121) )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v123 = v750;
      v643 = v121;
      v121 = (_QWORD *)v120;
      v120 = (__int64)v643;
    }
    v124 = *(_QWORD *)(v120 - 24);
    if ( *(_BYTE *)(v124 + 16) )
      goto LABEL_1104;
    v125 = *(_DWORD *)(v124 + 36);
    if ( v123 == v125 )
    {
      v642 = *(_DWORD *)(v120 + 20) & 0xFFFFFFF;
      if ( v121 == *(_QWORD **)(v120 - 24 * v642) || v121 == *(_QWORD **)(v120 + 24 * (1 - v642)) )
        goto LABEL_448;
      if ( v123 > 0xF6C || v123 <= 0xF67 || ((1LL << ((unsigned __int8)v123 - 104)) & 0x15) == 0 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    }
    else
    {
      if ( v125 > 0xF6C )
      {
        if ( v125 - 4114 > 2 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
LABEL_162:
        v127 = *(_DWORD *)(v120 + 20) & 0xFFFFFFF;
        if ( v121 == *(_QWORD **)(v120 - 24 * v127) || v121 == *(_QWORD **)(v120 + 24 * (1 - v127)) )
          sub_170E100(
            (__int64 *)v12,
            v10,
            (__int64)v121,
            a3,
            *(double *)v13.m128i_i64,
            *(double *)v14.m128i_i64,
            a6,
            v28,
            v29,
            a9,
            a10);
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      }
      if ( v125 <= 0xF67 || ((1LL << ((unsigned __int8)v125 - 104)) & 0x15) == 0 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    }
    if ( !sub_173DB60((__int64)v121) || (v126 = *((_BYTE *)v121 + 17) >> 1, v126 != 127) && (v126 & 2) == 0 )
    {
      if ( *((_BYTE *)v121 + 16) != 14 || sub_173D800((__int64)(v121 + 3)) )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    }
    goto LABEL_162;
  }
  if ( v39 > 0x1B1 )
  {
    switch ( v39 )
    {
      case 0x1B2u:
        v290 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v291 = *(_QWORD *)(v10 - 24 * v290);
        v292 = *(_QWORD *)(v10 + 24 * (1 - v290));
        v293 = *(_BYTE *)(v292 + 16);
        if ( v293 == 13 )
        {
          v294 = *(_QWORD **)(v292 + 24);
          if ( *(_DWORD *)(v292 + 32) > 0x40u )
            v294 = (_QWORD *)*v294;
          v295 = (__int16)v294;
          v296 = (unsigned __int16)v294 & 0x3FF;
          if ( v296 == 1023 )
          {
            v67 = sub_15A0680(*(_QWORD *)v10, 1, 0);
            return sub_170E100(
                     (__int64 *)v12,
                     v10,
                     v67,
                     a3,
                     *(double *)v13.m128i_i64,
                     *(double *)v14.m128i_i64,
                     a6,
                     v28,
                     v29,
                     a9,
                     a10);
          }
          if ( ((unsigned __int16)v294 & 0x3FF) != 0 )
          {
            if ( (_DWORD)v294 != 3 )
            {
              v297 = *(unsigned __int8 *)(v291 + 16);
              if ( (_BYTE)v297 == 14 )
              {
                v298 = v291 + 24;
                if ( ((unsigned __int8)v294 & 1) != 0 )
                {
                  if ( !sub_173D800(v291 + 24) )
                  {
LABEL_416:
                    if ( (v295 & 4) == 0
                      || (*(void **)(v291 + 32) == sub_16982C0()
                        ? (v299 = *(_QWORD *)(v291 + 40) + 8LL)
                        : (v299 = v291 + 32),
                          (*(_BYTE *)(v299 + 18) & 7) != 0 || !sub_173D840(v291 + 24)) )
                    {
                      if ( ((v295 & 8) == 0
                         || !sub_173D8B0(
                               v291 + 24,
                               v297,
                               *(double *)a3.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)v14.m128i_i64)
                         || !sub_173D840(v291 + 24))
                        && ((v295 & 0x10) == 0
                         || !sub_173D870(
                               v291 + 24,
                               v297,
                               *(double *)a3.m128_u64,
                               *(double *)v13.m128i_i64,
                               *(double *)v14.m128i_i64)
                         || !sub_173D840(v291 + 24)) )
                      {
                        if ( (v295 & 0x20) == 0
                          || (*(void **)(v291 + 32) == sub_16982C0()
                            ? (v685 = *(_QWORD *)(v291 + 40) + 8LL)
                            : (v685 = v291 + 32),
                              (*(_BYTE *)(v685 + 18) & 7) != 3 || !sub_173D840(v291 + 24)) )
                        {
                          if ( (v295 & 0x40) == 0
                            || (*(void **)(v291 + 32) == sub_16982C0()
                              ? (v686 = *(_QWORD *)(v291 + 40) + 8LL)
                              : (v686 = v291 + 32),
                                (*(_BYTE *)(v686 + 18) & 7) != 3 || sub_173D840(v291 + 24)) )
                          {
                            if ( ((v295 & 0x80u) == 0
                               || !sub_173D870(
                                     v291 + 24,
                                     v297,
                                     *(double *)a3.m128_u64,
                                     *(double *)v13.m128i_i64,
                                     *(double *)v14.m128i_i64)
                               || sub_173D840(v291 + 24))
                              && ((v295 & 0x100) == 0
                               || !sub_173D8B0(
                                     v291 + 24,
                                     v297,
                                     *(double *)a3.m128_u64,
                                     *(double *)v13.m128i_i64,
                                     *(double *)v14.m128i_i64)
                               || sub_173D840(v291 + 24)) )
                            {
                              if ( (v295 & 0x200) != 0 )
                              {
                                if ( *(void **)(v291 + 32) == sub_16982C0() )
                                  v687 = *(_QWORD *)(v291 + 40) + 8LL;
                                else
                                  v687 = v291 + 32;
                                v300 = 0;
                                if ( (*(_BYTE *)(v687 + 18) & 7) == 0 )
                                  v300 = !sub_173D840(v298);
                              }
                              else
                              {
                                v300 = 0;
                              }
                              goto LABEL_422;
                            }
                          }
                        }
                      }
                    }
LABEL_421:
                    v300 = 1;
LABEL_422:
                    v67 = sub_15A0680(*(_QWORD *)v10, v300, 0);
                    return sub_170E100(
                             (__int64 *)v12,
                             v10,
                             v67,
                             a3,
                             *(double *)v13.m128i_i64,
                             *(double *)v14.m128i_i64,
                             a6,
                             v28,
                             v29,
                             a9,
                             a10);
                  }
                  v688 = (_BYTE *)(v291 + 32);
                  if ( *(void **)(v291 + 32) == sub_16982C0() )
                    v688 = (_BYTE *)(*(_QWORD *)(v291 + 40) + 8LL);
                  if ( (unsigned __int8)sub_169B470(v688) )
                    goto LABEL_421;
                }
                if ( (v295 & 2) != 0 && sub_173D800(v291 + 24) )
                {
                  v689 = (_BYTE *)(v291 + 32);
                  if ( *(void **)(v291 + 32) == sub_16982C0() )
                    v689 = (_BYTE *)(*(_QWORD *)(v291 + 40) + 8LL);
                  if ( !(unsigned __int8)sub_169B470(v689) )
                    goto LABEL_421;
                }
                goto LABEL_416;
              }
              if ( (_BYTE)v297 != 9 )
              {
                if ( v296 == (_DWORD)v294 )
                  return sub_1742F80(
                           v12,
                           v785,
                           a3,
                           *(double *)v13.m128i_i64,
                           *(double *)v14.m128i_i64,
                           a6,
                           v28,
                           v29,
                           a9,
                           a10);
                v816.m128i_i16[0] = 257;
                v690 = *(_QWORD *)(v12 + 8);
                v812 = (char *)v291;
                v691 = sub_15A0680(*(_QWORD *)v292, (unsigned __int16)v294 & 0x3FF, 0);
                v692 = *(_QWORD *)(v10 - 24);
                v813 = (__int64 **)v691;
                if ( *(_BYTE *)(v692 + 16) )
                  goto LABEL_1105;
                v693 = sub_172C570(v690, *(_QWORD *)(v692 + 24), v692, (__int64 *)&v812, 2, v815.m128i_i64, 0);
                sub_164B7C0(v693, v10);
                v67 = v693;
                return sub_170E100(
                         (__int64 *)v12,
                         v10,
                         v67,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v28,
                         v29,
                         a9,
                         a10);
              }
LABEL_68:
              v67 = sub_1599EF0(*(__int64 ***)v10);
              return sub_170E100(
                       (__int64 *)v12,
                       v10,
                       v67,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            }
            v694 = *(_QWORD *)(v12 + 8);
            v816.m128i_i16[0] = 257;
            v646 = sub_17290F0(v694, 8, v291, v291, v815.m128i_i64, 0);
            goto LABEL_968;
          }
        }
        else
        {
          if ( *(_BYTE *)(v291 + 16) == 9 )
            goto LABEL_68;
          if ( v293 != 9 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
        v67 = sub_15A0680(*(_QWORD *)v10, 0, 0);
        return sub_170E100(
                 (__int64 *)v12,
                 v10,
                 v67,
                 a3,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64,
                 a6,
                 v28,
                 v29,
                 a9,
                 a10);
      case 0x1B3u:
LABEL_527:
        v399 = *(_DWORD *)(v10 + 20);
        v813 = &v809;
        v400 = *(_QWORD *)(v10 - 24LL * (v399 & 0xFFFFFFF));
        if ( !(unsigned __int8)sub_171FB50((__int64)&v812, v400, v30, v32) )
        {
          v816.m128i_i64[0] = (__int64)&v809;
          v815.m128i_i32[0] = 96;
          v815.m128i_i32[2] = 0;
          if ( !(unsigned __int8)sub_173F0F0((__int64)&v815, v400) )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
        v11 = v10;
        sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), (__int64)v809);
        return v11;
      case 0x1B8u:
      case 0x1B9u:
      case 0x1BBu:
      case 0x1BCu:
        v66 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        if ( *(_BYTE *)(*(_QWORD *)(v10 - 24 * v66) + 16LL) == 9
          && *(_BYTE *)(*(_QWORD *)(v10 + 24 * (1 - v66)) + 16LL) == 9 )
        {
          goto LABEL_68;
        }
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      case 0x1BDu:
        v273 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v274 = *(_QWORD *)(v10 - 24 * v273);
        v275 = *(_QWORD *)(v10 + 24 * (1 - v273));
        v276 = *(_BYTE *)(v274 + 16);
        if ( v276 == 14 )
        {
          if ( *(_BYTE *)(v275 + 16) != 14 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v277 = *(__int64 ***)v10;
          if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
            v277 = (__int64 **)*v277[2];
          switch ( *((_BYTE *)v277 + 8) )
          {
            case 1:
              v507 = (__int16 *)sub_1698260();
              break;
            case 2:
              v507 = (__int16 *)sub_1698270();
              break;
            case 3:
              v507 = (__int16 *)sub_1698280();
              break;
            case 4:
              v507 = (__int16 *)sub_16982A0();
              break;
            case 5:
              v507 = (__int16 *)sub_1698290();
              break;
            case 6:
              v507 = (__int16 *)sub_16982C0();
              break;
          }
          v794 = v507;
          sub_169C7A0(&v813, (__int64 *)(v274 + 32));
          sub_169C7A0(&v815.m128i_i64[1], (__int64 *)(v275 + 32));
          sub_16A3360((__int64)&v812, v794, 3u, (bool *)&v808);
          sub_16A3360((__int64)&v815, v794, 3u, (bool *)&v808);
          v508 = (_QWORD *)sub_16498A0(v10);
          v809 = (__int64 *)sub_159CCF0(v508, (__int64)&v812);
          v509 = (_QWORD *)sub_16498A0(v10);
          v810 = sub_159CCF0(v509, (__int64)&v815);
          v510 = sub_15A01B0((__int64 *)&v809, 2);
          v11 = sub_170E100(
                  (__int64 *)v12,
                  v10,
                  v510,
                  a3,
                  *(double *)v13.m128i_i64,
                  *(double *)v14.m128i_i64,
                  a6,
                  v511,
                  v512,
                  a9,
                  a10);
          sub_127D120(&v815.m128i_i64[1]);
          sub_127D120(&v813);
          return v11;
        }
        if ( v276 != 9 || *(_BYTE *)(v275 + 16) != 9 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        goto LABEL_68;
      case 0x1CCu:
      case 0x1CDu:
        v87 = *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v87 + 16) != 13 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v88 = *(_QWORD **)(v87 + 24);
        if ( *(_DWORD *)(v87 + 32) > 0x40u )
          v88 = (_QWORD *)*v88;
        v89 = (int)v88;
        if ( (_DWORD)v88 == 15 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v749 = v12;
        v90 = 0;
        v91 = 0;
        v92 = v39;
        while ( 2 )
        {
          if ( v92 == 461 )
          {
            if ( v90 <= 1 )
            {
              if ( (v89 & (3 << (2 * v90))) == 0 )
              {
LABEL_93:
                v93 = *(_QWORD *)(v10 + 24 * (v90 + 2 - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
                if ( *(_BYTE *)(v93 + 16) != 9 )
                {
                  v94 = sub_1599EF0(*(__int64 ***)v93);
                  sub_1593B40((_QWORD *)(v10 + 24 * (v90 + 2 - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v94);
                  v91 = 1;
                }
              }
LABEL_95:
              ++v90;
              continue;
            }
          }
          else if ( v90 != 4 )
          {
            if ( (v89 & (1 << v90)) == 0 )
              goto LABEL_93;
            goto LABEL_95;
          }
          break;
        }
        v12 = v749;
        if ( v91 )
          return v10;
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      case 0x1CEu:
      case 0x1D8u:
        v75 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v76 = *(_QWORD *)(v10 + 24 * (2 - v75));
        v748 = (__int64 *)v76;
        if ( *(_BYTE *)(v76 + 16) != 13 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v77 = *(_QWORD **)(v76 + 24);
        if ( *(_DWORD *)(v76 + 32) > 0x40u )
          v77 = (_QWORD *)*v77;
        if ( v39 == 472 )
        {
          if ( (unsigned __int64)(v77 - 4) > 9 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
        else if ( (unsigned __int64)v77 > 0xF )
        {
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
        v78 = 1;
        v79 = *(_QWORD *)(v10 - 24 * v75);
        v80 = 1 - v75;
        v81 = 24 * (1 - v75);
        v82 = *(_QWORD *)(v10 + v81);
        if ( *(_BYTE *)(v79 + 16) <= 0x10u )
        {
          if ( *(_BYTE *)(v82 + 16) <= 0x10u )
          {
            v83 = v79;
            v84 = sub_15A37B0((unsigned __int16)v77, (_QWORD *)v79, *(_QWORD **)(v10 + v81), 0);
            if ( sub_1593BB0(v84, v83, v85, v86) )
            {
              v67 = sub_15A4460(v84, *(__int64 ***)v10, 0);
              return sub_170E100(
                       (__int64 *)v12,
                       v10,
                       v67,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            }
            v815.m128i_i64[0] = *(_QWORD *)v10;
            v514 = (__int64 *)sub_15F2050(v10);
            v515 = sub_15E26F0(v514, 184, v815.m128i_i64, 1);
            v516 = (__int64 *)sub_16498A0(v10);
            v809 = (__int64 *)sub_161FF10(v516, "exec", 4u);
            v517 = (__int64 *)sub_16498A0(v10);
            v518 = sub_1627350(v517, (__int64 *)&v809, (__int64 *)1, 0, 1);
            v519 = (__int64 *)sub_16498A0(v10);
            v520 = (__int64 *)sub_1628DA0(v519, v518);
            v816.m128i_i16[0] = 257;
            v521 = *(_QWORD *)(v12 + 8);
            v812 = (char *)v520;
            v522 = sub_172C570(v521, *(_QWORD *)(*(_QWORD *)v515 + 24LL), v515, (__int64 *)&v812, 1, v815.m128i_i64, 0);
            sub_173DC10(v522, -1, 8);
            goto LABEL_753;
          }
          v795 = v79;
          v525 = sub_15FF5D0((unsigned int)v77);
          sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), v82);
          sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v795);
          v526 = v525;
LABEL_755:
          v527 = sub_159C470(*v748, v526, 0);
          sub_1593B40((_QWORD *)(v10 + 24 * (2LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v527);
          return v10;
        }
        if ( (unsigned __int64)(v77 - 4) > 1 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        if ( v77 != (_QWORD *)32 )
          goto LABEL_758;
        v538 = *(_BYTE *)(v82 + 16);
        if ( v538 == 13 )
        {
          if ( *(_DWORD *)(v82 + 32) > 0x40u )
          {
            v716 = *(_DWORD *)(v82 + 32);
            v539 = v82 + 24;
            v739 = v79;
LABEL_773:
            v540 = sub_16A57B0(v539);
            v79 = v739;
            v541 = v716 - 1 == v540;
            goto LABEL_774;
          }
          v541 = *(_QWORD *)(v82 + 24) == 1;
        }
        else
        {
          v78 = *(_QWORD *)v82;
          if ( *(_BYTE *)(*(_QWORD *)v82 + 8LL) != 16 || v538 > 0x10u )
            goto LABEL_758;
          v739 = v79;
          v552 = sub_15A1020((_BYTE *)v82, v26, v80, v78);
          v79 = v739;
          if ( !v552 || *(_BYTE *)(v552 + 16) != 13 )
          {
            v590 = 0;
            v743 = *(_DWORD *)(*(_QWORD *)v82 + 32LL);
            while ( v743 != v590 )
            {
              v26 = v590;
              v705 = v79;
              v719 = v590;
              v591 = sub_15A0A60(v82, v590);
              v592 = v719;
              v79 = v705;
              if ( !v591 )
                goto LABEL_777;
              v80 = *(unsigned __int8 *)(v591 + 16);
              if ( (_BYTE)v80 != 9 )
              {
                if ( (_BYTE)v80 != 13 )
                  goto LABEL_777;
                v80 = *(unsigned int *)(v591 + 32);
                if ( (unsigned int)v80 <= 0x40 )
                {
                  v594 = *(_QWORD *)(v591 + 24) == 1;
                }
                else
                {
                  v700 = v719;
                  v720 = *(_DWORD *)(v591 + 32);
                  v593 = sub_16A57B0(v591 + 24);
                  v79 = v705;
                  v592 = v700;
                  v80 = (unsigned int)(v720 - 1);
                  v594 = (_DWORD)v80 == v593;
                }
                if ( !v594 )
                  goto LABEL_777;
              }
              v590 = v592 + 1;
            }
            goto LABEL_775;
          }
          if ( *(_DWORD *)(v552 + 32) > 0x40u )
          {
            v716 = *(_DWORD *)(v552 + 32);
            v539 = v552 + 24;
            goto LABEL_773;
          }
          v541 = *(_QWORD *)(v552 + 24) == 1;
        }
LABEL_774:
        if ( !v541 )
          goto LABEL_777;
LABEL_775:
        v542 = *(unsigned __int8 *)(v79 + 16);
        if ( (unsigned __int8)v542 > 0x17u )
        {
          v584 = v542 - 24;
        }
        else
        {
          if ( (_BYTE)v542 != 5 )
            goto LABEL_777;
          v584 = *(unsigned __int16 *)(v79 + 18);
        }
        if ( v584 == 37 )
        {
          v585 = (*(_BYTE *)(v79 + 23) & 0x40) != 0
               ? *(__int64 ***)(v79 - 8)
               : (__int64 **)(v79 - 24LL * (*(_DWORD *)(v79 + 20) & 0xFFFFFFF));
          v549 = *v585;
          if ( *v585 )
          {
LABEL_789:
            v26 = 1;
            v741 = v79;
            v550 = sub_1642F90(*v549, 1);
            v79 = v741;
            if ( v550 )
            {
              v551 = sub_15A06D0(*(__int64 ***)v82, 1, v80, v78);
              sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v551);
              v526 = 33;
              goto LABEL_755;
            }
            goto LABEL_758;
          }
        }
LABEL_777:
        v543 = *(_BYTE *)(v82 + 16);
        if ( v543 == 13 )
        {
          v26 = *(unsigned int *)(v82 + 32);
          if ( (unsigned int)v26 > 0x40 )
          {
            v717 = *(_DWORD *)(v82 + 32);
            v544 = v82 + 24;
            v740 = v79;
LABEL_780:
            v545 = sub_16A58F0(v544);
            v26 = v717;
            v79 = v740;
            v546 = v717 == v545;
            goto LABEL_781;
          }
          v78 = (unsigned int)(64 - v26);
          v546 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) == *(_QWORD *)(v82 + 24);
        }
        else
        {
          v78 = *(_QWORD *)v82;
          if ( *(_BYTE *)(*(_QWORD *)v82 + 8LL) != 16 || v543 > 0x10u )
            goto LABEL_758;
          v740 = v79;
          v589 = sub_15A1020((_BYTE *)v82, v26, v80, v78);
          v79 = v740;
          if ( !v589 || *(_BYTE *)(v589 + 16) != 13 )
          {
            v26 = 0;
            v744 = *(_DWORD *)(*(_QWORD *)v82 + 32LL);
            while ( v744 != (_DWORD)v26 )
            {
              v706 = v79;
              v595 = sub_15A0A60(v82, v26);
              v79 = v706;
              if ( !v595 )
                goto LABEL_758;
              v80 = *(unsigned __int8 *)(v595 + 16);
              v26 = (unsigned int)v26;
              if ( (_BYTE)v80 != 9 )
              {
                if ( (_BYTE)v80 != 13 )
                  goto LABEL_758;
                v596 = *(_DWORD *)(v595 + 32);
                if ( v596 <= 0x40 )
                {
                  v78 = 64 - v596;
                  v80 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v596);
                  v598 = v80 == *(_QWORD *)(v595 + 24);
                }
                else
                {
                  v721 = *(_DWORD *)(v595 + 32);
                  v597 = sub_16A58F0(v595 + 24);
                  v80 = v721;
                  v79 = v706;
                  v26 = (unsigned int)v26;
                  v598 = v721 == v597;
                }
                if ( !v598 )
                  goto LABEL_758;
              }
              v26 = (unsigned int)(v26 + 1);
            }
LABEL_782:
            v547 = *(unsigned __int8 *)(v79 + 16);
            if ( (unsigned __int8)v547 > 0x17u )
            {
              v548 = v547 - 24;
            }
            else
            {
              if ( (_BYTE)v547 != 5 )
                goto LABEL_758;
              v548 = *(unsigned __int16 *)(v79 + 18);
            }
            if ( v548 == 38 )
            {
              v78 = (*(_BYTE *)(v79 + 23) & 0x40) != 0
                  ? *(_QWORD *)(v79 - 8)
                  : v79 - 24LL * (*(_DWORD *)(v79 + 20) & 0xFFFFFFF);
              v549 = *(__int64 **)v78;
              if ( *(_QWORD *)v78 )
                goto LABEL_789;
            }
LABEL_758:
            if ( *(_BYTE *)(v82 + 16) <= 0x10u )
            {
              v738 = v79;
              v528 = sub_1593BB0(v82, v26, v80, v78);
              v531 = v738;
              if ( !v528 )
              {
                if ( *(_BYTE *)(v82 + 16) == 13 )
                {
                  if ( *(_DWORD *)(v82 + 32) <= 0x40u )
                  {
                    if ( *(_QWORD *)(v82 + 24) )
                      return sub_1742F80(
                               v12,
                               v785,
                               a3,
                               *(double *)v13.m128i_i64,
                               *(double *)v14.m128i_i64,
                               a6,
                               v28,
                               v29,
                               a9,
                               a10);
                  }
                  else
                  {
                    v715 = *(_DWORD *)(v82 + 32);
                    v537 = sub_16A57B0(v82 + 24);
                    v531 = v738;
                    if ( v715 != v537 )
                      return sub_1742F80(
                               v12,
                               v785,
                               a3,
                               *(double *)v13.m128i_i64,
                               *(double *)v14.m128i_i64,
                               a6,
                               v28,
                               v29,
                               a9,
                               a10);
                  }
                }
                else
                {
                  if ( *(_BYTE *)(*(_QWORD *)v82 + 8LL) != 16 )
                    return sub_1742F80(
                             v12,
                             v785,
                             a3,
                             *(double *)v13.m128i_i64,
                             *(double *)v14.m128i_i64,
                             a6,
                             v28,
                             v29,
                             a9,
                             a10);
                  v580 = sub_15A1020((_BYTE *)v82, v26, v529, v530);
                  v531 = v738;
                  if ( v580 && *(_BYTE *)(v580 + 16) == 13 )
                  {
                    v581 = *(_DWORD *)(v580 + 32);
                    if ( v581 <= 0x40 )
                    {
                      v583 = *(_QWORD *)(v580 + 24) == 0;
                    }
                    else
                    {
                      v582 = sub_16A57B0(v580 + 24);
                      v531 = v738;
                      v583 = v581 == v582;
                    }
                    if ( !v583 )
                      return sub_1742F80(
                               v12,
                               v785,
                               a3,
                               *(double *)v13.m128i_i64,
                               *(double *)v14.m128i_i64,
                               a6,
                               v28,
                               v29,
                               a9,
                               a10);
                  }
                  else
                  {
                    v704 = v738;
                    v718 = v77;
                    v586 = 0;
                    v742 = *(_DWORD *)(*(_QWORD *)v82 + 32LL);
                    while ( v742 != v586 )
                    {
                      v587 = sub_15A0A60(v82, v586);
                      if ( !v587 )
                        return sub_1742F80(
                                 v12,
                                 v785,
                                 a3,
                                 *(double *)v13.m128i_i64,
                                 *(double *)v14.m128i_i64,
                                 a6,
                                 v28,
                                 v29,
                                 a9,
                                 a10);
                      v588 = *(_BYTE *)(v587 + 16);
                      if ( v588 != 9 )
                      {
                        if ( v588 != 13 )
                          return sub_1742F80(
                                   v12,
                                   v785,
                                   a3,
                                   *(double *)v13.m128i_i64,
                                   *(double *)v14.m128i_i64,
                                   a6,
                                   v28,
                                   v29,
                                   a9,
                                   a10);
                        if ( *(_DWORD *)(v587 + 32) <= 0x40u )
                        {
                          if ( *(_QWORD *)(v587 + 24) )
                            return sub_1742F80(
                                     v12,
                                     v785,
                                     a3,
                                     *(double *)v13.m128i_i64,
                                     *(double *)v14.m128i_i64,
                                     a6,
                                     v28,
                                     v29,
                                     a9,
                                     a10);
                        }
                        else
                        {
                          v699 = *(_DWORD *)(v587 + 32);
                          if ( v699 != (unsigned int)sub_16A57B0(v587 + 24) )
                            return sub_1742F80(
                                     v12,
                                     v785,
                                     a3,
                                     *(double *)v13.m128i_i64,
                                     *(double *)v14.m128i_i64,
                                     a6,
                                     v28,
                                     v29,
                                     a9,
                                     a10);
                        }
                      }
                      ++v586;
                    }
                    v77 = v718;
                    v531 = v704;
                  }
                }
              }
              v815.m128i_i64[0] = (__int64)&v807;
              v815.m128i_i64[1] = (__int64)&v808;
              v816.m128i_i64[0] = (__int64)&v809;
              v816.m128i_i64[1] = (__int64)&v807;
              v817 = &v808;
              v818 = &v809;
              if ( (unsigned __int8)sub_173F190((__int64)&v815, v531) )
              {
                if ( v77 == (_QWORD *)32 )
                  LODWORD(v807) = sub_15FF0F0(v807);
                v532 = (unsigned int)v807 < 0x10 ? 462 : 472;
                v815.m128i_i64[0] = *v808;
                v533 = (__int64 *)sub_15F2050(v10);
                v534 = sub_15E26F0(v533, v532, v815.m128i_i64, 1);
                v812 = (char *)v808;
                v813 = (__int64 **)v809;
                v535 = sub_159C470(*v748, (unsigned int)v807, 0);
                v536 = *(_QWORD *)(v12 + 8);
                v814 = (__int64 **)v535;
                v816.m128i_i16[0] = 257;
                v522 = sub_172C570(
                         v536,
                         *(_QWORD *)(*(_QWORD *)v534 + 24LL),
                         v534,
                         (__int64 *)&v812,
                         3,
                         v815.m128i_i64,
                         0);
LABEL_753:
                sub_164B7C0(v522, v10);
                return sub_170E100(
                         (__int64 *)v12,
                         v10,
                         v522,
                         a3,
                         *(double *)v13.m128i_i64,
                         *(double *)v14.m128i_i64,
                         a6,
                         v523,
                         v524,
                         a9,
                         a10);
              }
            }
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
          v26 = *(unsigned int *)(v589 + 32);
          if ( (unsigned int)v26 > 0x40 )
          {
            v717 = *(_DWORD *)(v589 + 32);
            v544 = v589 + 24;
            goto LABEL_780;
          }
          v78 = (unsigned int)(64 - v26);
          v546 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) == *(_QWORD *)(v589 + 24);
        }
LABEL_781:
        if ( !v546 )
          goto LABEL_758;
        goto LABEL_782;
      case 0x1D2u:
        v278 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        v279 = *(_QWORD *)(v10 - 24 * v278);
        v280 = *(_QWORD *)(v10 + 24 * (1 - v278));
        v728 = v280;
        v763 = *(_QWORD *)(v10 + 24 * (2 - v278));
        if ( (unsigned __int8)sub_173D590((_BYTE *)v279, v26, v278, v32) || *(_BYTE *)(v279 + 16) == 9 )
        {
          v499 = *(__int64 **)(v12 + 8);
          v816.m128i_i16[0] = 257;
          v495 = sub_15E83D0(v499, 139, (__int64 *)v280, v763, (int)&v815);
        }
        else if ( (unsigned __int8)sub_173D590((_BYTE *)v280, v26, v281, v282) || *(_BYTE *)(v280 + 16) == 9 )
        {
          v498 = *(__int64 **)(v12 + 8);
          v816.m128i_i16[0] = 257;
          v495 = sub_15E83D0(v498, 139, (__int64 *)v279, v763, (int)&v815);
        }
        else
        {
          if ( !(unsigned __int8)sub_173D590((_BYTE *)v763, v26, v283, v284) && *(_BYTE *)(v763 + 16) != 9 )
            goto LABEL_400;
          v494 = *(__int64 **)(v12 + 8);
          v816.m128i_i16[0] = 257;
          v495 = sub_15E83D0(v494, 132, (__int64 *)v279, v280, (int)&v815);
        }
        if ( v495 )
        {
          sub_15F2500((__int64)v495, v10);
          sub_164B7C0((__int64)v495, v10);
          v67 = (__int64)v495;
          return sub_170E100(
                   (__int64 *)v12,
                   v10,
                   v67,
                   a3,
                   *(double *)v13.m128i_i64,
                   *(double *)v14.m128i_i64,
                   a6,
                   v28,
                   v29,
                   a9,
                   a10);
        }
LABEL_400:
        v285 = *(_BYTE *)(v728 + 16);
        if ( *(_BYTE *)(v279 + 16) > 0x10u )
        {
          if ( v285 > 0x10u )
            goto LABEL_612;
          if ( *(_BYTE *)(v763 + 16) > 0x10u )
          {
LABEL_406:
            v289 = v279;
            v11 = v10;
            sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), v289);
            sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v763);
            sub_1593B40((_QWORD *)(v10 + 24 * (2LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v728);
            return v11;
          }
          v286 = 0;
          if ( *(_BYTE *)(v279 + 16) > 0x10u )
          {
LABEL_612:
            if ( *(_BYTE *)(v279 + 16) != 14 || *(_BYTE *)(v728 + 16) != 14 || *(_BYTE *)(v763 + 16) != 14 )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            v461 = *(__int64 ***)(v279 + 32);
            v714 = v728 + 24;
            v697 = v279 + 24;
            v793 = (__int64 **)sub_16982C0();
            if ( v461 == v793 )
              v462 = *(_QWORD *)(v279 + 40) + 8LL;
            else
              v462 = v279 + 32;
            v463 = *(__int64 ***)(v728 + 32);
            if ( (*(_BYTE *)(v462 + 18) & 7) == 1 )
            {
              v465 = (__int64 *)(v728 + 32);
              if ( v793 != v463 )
                goto LABEL_622;
            }
            else
            {
              if ( v793 == v463 )
                v464 = *(_QWORD *)(v728 + 40) + 8LL;
              else
                v464 = v728 + 32;
              if ( (*(_BYTE *)(v464 + 18) & 7) == 1 )
              {
                v465 = (__int64 *)(v279 + 32);
                if ( v461 != v793 )
                {
LABEL_622:
                  sub_16986C0(&v815.m128i_i64[1], v465);
                  goto LABEL_623;
                }
              }
              else
              {
                v201 = (unsigned int)sub_14A9E40(v697, v714) == 0;
                v492 = v279 + 24;
                if ( v201 )
                  v492 = v728 + 24;
                v465 = (__int64 *)(v492 + 8);
                if ( v793 != *(__int64 ***)(v492 + 8) )
                  goto LABEL_622;
              }
            }
            sub_169C6E0(&v815.m128i_i64[1], (__int64)v465);
LABEL_623:
            v466 = &v815.m128i_i8[8];
            if ( v793 == (__int64 **)v815.m128i_i64[1] )
              v466 = (__int8 *)(v816.m128i_i64[0] + 8);
            v467 = *(__int64 ***)(v763 + 32);
            if ( (v466[18] & 7) == 1 )
            {
              v469 = (__int64 *)(v763 + 32);
              if ( v793 != v467 )
                goto LABEL_630;
            }
            else
            {
              if ( v793 == v467 )
                v468 = *(_QWORD *)(v763 + 40) + 8LL;
              else
                v468 = v763 + 32;
              if ( (*(_BYTE *)(v468 + 18) & 7) == 1 )
              {
                v469 = &v815.m128i_i64[1];
                if ( v793 != (__int64 **)v815.m128i_i64[1] )
                {
LABEL_630:
                  sub_16986C0(&v813, v469);
                  goto LABEL_631;
                }
              }
              else
              {
                v201 = (unsigned int)sub_14A9E40((__int64)&v815, v763 + 24) == 0;
                v491 = (__m128i *)(v763 + 24);
                if ( !v201 )
                  v491 = &v815;
                v469 = &v491->m128i_i64[1];
                if ( v793 != (__int64 **)v491->m128i_i64[1] )
                  goto LABEL_630;
              }
            }
            sub_169C6E0(&v813, (__int64)v469);
LABEL_631:
            if ( (__int64 **)v815.m128i_i64[1] == v793 )
            {
              if ( v816.m128i_i64[0] )
              {
                v487 = v12;
                v488 = v816.m128i_i64[0];
                for ( j = v816.m128i_i64[0] + 32LL * *(_QWORD *)(v816.m128i_i64[0] - 8);
                      v488 != j;
                      sub_127D120((_QWORD *)(j + 8)) )
                {
                  j -= 32;
                }
                v490 = v488;
                v12 = v487;
                j_j_j___libc_free_0_0(v490 - 8);
              }
            }
            else
            {
              sub_1698460((__int64)&v815.m128i_i64[1]);
            }
            if ( (unsigned int)sub_14A9E40((__int64)&v812, v697) == 1 )
            {
              v470 = *(__int64 ***)(v728 + 32);
              v471 = v728 + 32;
              if ( v793 == v470 )
                v471 = *(_QWORD *)(v728 + 40) + 8LL;
              v472 = *(__int64 ***)(v763 + 32);
              if ( (*(_BYTE *)(v471 + 18) & 7) != 1 )
              {
                if ( v793 == v472 )
                  v473 = *(_QWORD *)(v763 + 40) + 8LL;
                else
                  v473 = v763 + 32;
                if ( (*(_BYTE *)(v473 + 18) & 7) == 1 )
                {
                  v474 = (__int64 *)(v728 + 32);
                  if ( v793 != v470 )
                  {
LABEL_641:
                    sub_16986C0(&v815.m128i_i64[1], v474);
LABEL_642:
                    if ( v793 == v813 )
                    {
                      v485 = v814;
                      if ( v814 )
                      {
                        for ( k = &v814[4 * (_QWORD)*(v814 - 1)]; v485 != k; sub_127D120(k + 1) )
                          k -= 4;
                        j_j_j___libc_free_0_0(v485 - 1);
                      }
                    }
                    else
                    {
                      sub_1698460((__int64)&v813);
                    }
                    v208 = *(_QWORD **)(*(_QWORD *)(v12 + 8) + 24LL);
                    goto LABEL_301;
                  }
LABEL_655:
                  sub_169C6E0(&v815.m128i_i64[1], (__int64)v474);
                  goto LABEL_642;
                }
                v481 = v763 + 24;
                v482 = v728 + 24;
                goto LABEL_662;
              }
LABEL_679:
              v474 = (__int64 *)(v763 + 32);
              if ( v793 != v472 )
                goto LABEL_641;
              goto LABEL_655;
            }
            v475 = sub_14A9E40((__int64)&v812, v714);
            v476 = *(__int64 ***)(v279 + 32);
            if ( v475 == 1 )
            {
              v483 = v279 + 32;
              if ( v793 == v476 )
                v483 = *(_QWORD *)(v279 + 40) + 8LL;
              v472 = *(__int64 ***)(v763 + 32);
              if ( (*(_BYTE *)(v483 + 18) & 7) == 1 )
                goto LABEL_679;
              if ( v793 == v472 )
                v484 = *(_QWORD *)(v763 + 40) + 8LL;
              else
                v484 = v763 + 32;
              if ( (*(_BYTE *)(v484 + 18) & 7) != 1 )
              {
                v481 = v763 + 24;
                v482 = v279 + 24;
LABEL_662:
                v201 = (unsigned int)sub_14A9E40(v482, v481) == 0;
                v480 = v482;
                if ( v201 )
                  v480 = v763 + 24;
                goto LABEL_654;
              }
            }
            else
            {
              v477 = v279 + 32;
              if ( v793 == v476 )
                v477 = *(_QWORD *)(v279 + 40) + 8LL;
              v478 = *(__int64 ***)(v728 + 32);
              if ( (*(_BYTE *)(v477 + 18) & 7) == 1 )
              {
                v474 = (__int64 *)(v728 + 32);
                if ( v793 != v478 )
                  goto LABEL_641;
                goto LABEL_655;
              }
              if ( v793 == v478 )
                v479 = *(_QWORD *)(v728 + 40) + 8LL;
              else
                v479 = v728 + 32;
              if ( (*(_BYTE *)(v479 + 18) & 7) != 1 )
              {
                v201 = (unsigned int)sub_14A9E40(v697, v714) == 0;
                v480 = v279 + 24;
                if ( v201 )
                  v480 = v728 + 24;
LABEL_654:
                v474 = (__int64 *)(v480 + 8);
                if ( v793 != *(__int64 ***)(v480 + 8) )
                  goto LABEL_641;
                goto LABEL_655;
              }
            }
            v474 = (__int64 *)(v279 + 32);
            if ( v793 != v476 )
              goto LABEL_641;
            goto LABEL_655;
          }
        }
        else
        {
          if ( v285 > 0x10u )
          {
            v513 = v279;
            v279 = v728;
            v728 = v513;
            v286 = 1;
          }
          else
          {
            v286 = 0;
          }
          v287 = *(_BYTE *)(v279 + 16);
          if ( *(_BYTE *)(v763 + 16) > 0x10u )
          {
            if ( v287 <= 0x10u )
            {
              v288 = v279;
              v279 = v763;
              v763 = v288;
            }
            goto LABEL_406;
          }
          if ( v287 > 0x10u )
            goto LABEL_704;
        }
        if ( *(_BYTE *)(v728 + 16) > 0x10u )
        {
          v496 = v763;
          v497 = v279;
          v279 = v728;
          v763 = v497;
          v728 = v496;
          goto LABEL_406;
        }
LABEL_704:
        if ( v286 )
        {
          v493 = v728;
          v728 = v763;
          v763 = v493;
          goto LABEL_406;
        }
        goto LABEL_612;
      case 0x1D5u:
      case 0x1D6u:
        v68 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
        v69 = *(_BYTE *)(v68 + 16);
        if ( v69 == 14 )
        {
          sub_173D940((__int64)&v815, v68 + 24, &v812, 0);
          v70 = *(_QWORD *)(v10 - 24);
          if ( !*(_BYTE *)(v70 + 16) )
          {
            if ( *(_DWORD *)(v70 + 36) == 470 )
            {
              v502 = (_QWORD *)sub_16498A0(v10);
              v72 = sub_159CCF0(v502, (__int64)&v815);
            }
            else
            {
              v71 = (int)v812;
              if ( (unsigned int)((_DWORD)v812 + 0x7FFFFFFF) > 0xFFFFFFFD )
              {
                LODWORD(v812) = 0;
                v71 = 0;
              }
              v72 = sub_15A0680(*(_QWORD *)v10, v71, 0);
            }
            v11 = sub_170E100(
                    (__int64 *)v12,
                    v10,
                    v72,
                    a3,
                    *(double *)v13.m128i_i64,
                    *(double *)v14.m128i_i64,
                    a6,
                    v73,
                    v74,
                    a9,
                    a10);
            sub_127D120(&v815.m128i_i64[1]);
            return v11;
          }
          goto LABEL_1104;
        }
        if ( v69 == 9 )
          goto LABEL_68;
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      default:
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    }
  }
  switch ( v39 )
  {
    case 4u:
      v311 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      v312 = sub_15F3430(v10);
      v815.m128i_i32[0] = 4;
      v815.m128i_i32[2] = 0;
      v816.m128i_i64[0] = v311;
      if ( sub_173F2F0((__int64)&v815, v312) && sub_1602380(v10) )
        return sub_170BC50(v12, v10);
      v315 = *(_BYTE *)(v311 + 16);
      v316 = *(_QWORD *)(v10 - 24);
      if ( v315 == 50 )
      {
        if ( *(_QWORD *)(v311 - 48) )
        {
          v806 = *(_QWORD *)(v311 - 48);
          v572 = *(_QWORD *)(v311 - 24);
          if ( v572 )
            goto LABEL_805;
        }
      }
      else if ( v315 == 5 && *(_WORD *)(v311 + 18) == 26 )
      {
        v313 = *(_DWORD *)(v311 + 20) & 0xFFFFFFF;
        if ( *(_QWORD *)(v311 - 24 * v313) )
        {
          v806 = *(_QWORD *)(v311 - 24 * v313);
          v572 = *(_QWORD *)(v311 + 24 * (1 - v313));
          if ( v572 )
          {
LABEL_805:
            v573 = *(_QWORD *)(v12 + 8);
            v807 = v572;
            v799 = v573;
            v574 = (__int64 *)sub_1649960(v10);
            v813 = v575;
            v812 = (char *)v574;
            v816.m128i_i16[0] = 261;
            v815.m128i_i64[0] = (__int64)&v812;
            sub_173EF80(v799, v316, &v806, 1, v815.m128i_i64, 0);
            v800 = *(_QWORD *)(v12 + 8);
            v576 = (__int64 *)sub_1649960(v10);
            v815.m128i_i64[0] = (__int64)&v812;
            v813 = v577;
            v327 = &v807;
            v812 = (char *)v576;
            v325 = v316;
            v816.m128i_i16[0] = 261;
            v326 = v800;
            goto LABEL_433;
          }
        }
      }
      v815.m128i_i64[0] = (__int64)&v806;
      v815.m128i_i64[1] = (__int64)&v807;
      if ( sub_173F330(v815.m128i_i64, v311, v313, v314) )
      {
        v789 = *(_QWORD *)(v12 + 8);
        v319 = (__int64 *)sub_1649960(v10);
        v320 = *(_QWORD *)(v12 + 8);
        v810 = v321;
        v815.m128i_i64[0] = (__int64)&v809;
        v809 = v319;
        v816.m128i_i16[0] = 261;
        LOWORD(v814) = 257;
        v808 = (__int64 *)sub_171CA90(
                            v320,
                            v806,
                            (__int64 *)&v812,
                            *(double *)a3.m128_u64,
                            *(double *)v13.m128i_i64,
                            *(double *)v14.m128i_i64);
        sub_173EF80(v789, v316, (__int64 *)&v808, 1, v815.m128i_i64, 0);
        v790 = *(_QWORD *)(v12 + 8);
        v322 = (__int64 *)sub_1649960(v10);
        v323 = *(_QWORD *)(v12 + 8);
        v816.m128i_i16[0] = 261;
        v810 = v324;
        v809 = v322;
        v815.m128i_i64[0] = (__int64)&v809;
        LOWORD(v814) = 257;
        v808 = (__int64 *)sub_171CA90(
                            v323,
                            v807,
                            (__int64 *)&v812,
                            *(double *)a3.m128_u64,
                            *(double *)v13.m128i_i64,
                            *(double *)v14.m128i_i64);
        v325 = v316;
        v326 = v790;
        v327 = (__int64 *)&v808;
LABEL_433:
        sub_173EF80(v326, v325, v327, 1, v815.m128i_i64, 0);
        return sub_170BC50(v12, v10);
      }
      v815.m128i_i64[0] = (__int64)&v809;
      v815.m128i_i64[1] = (__int64)&v812;
      if ( (unsigned __int8)sub_173FB90((__int64)&v815, v311, v317, v318)
        && (_DWORD)v809 == 33
        && v812[16] == 54
        && *(_BYTE *)(*(_QWORD *)v812 + 8LL) == 15
        && (unsigned __int8)sub_14AFF20(v10, (__int64)v812, *(_QWORD *)(v12 + 2656)) )
      {
        v638 = (__int64 *)sub_16498A0(v10);
        v639 = sub_1627350(v638, 0, 0, 0, 1);
        sub_1625C10((__int64)v812, 11, v639);
        return sub_170BC50(v12, v10);
      }
      v636 = *(_QWORD *)(v12 + 2664);
      v696 = *(_QWORD *)(v12 + 2656);
      v637 = *(_QWORD *)(v12 + 2640);
      v815.m128i_i64[0] = 0;
      v815.m128i_i64[1] = 1;
      v816.m128i_i64[0] = 0;
      v816.m128i_i64[1] = 1;
      sub_14BB090(v311, (__int64)&v815, v636, 0, v637, v10, v696, 0);
      if ( sub_1454FB0((__int64)&v816) && sub_1602380(v10) )
      {
        v11 = sub_170BC50(v12, v10);
        sub_135E100(v816.m128i_i64);
        sub_135E100(v815.m128i_i64);
        return v11;
      }
      sub_14CDA00(*(_QWORD *)(v12 + 2640), v10);
      sub_135E100(v816.m128i_i64);
      sub_135E100(v815.m128i_i64);
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    case 6u:
      v301 = *(_DWORD *)(v10 + 20);
      v805 = 0;
      v815.m128i_i32[0] = 6;
      v815.m128i_i32[2] = 0;
      v302 = *(_QWORD *)(v10 - 24LL * (v301 & 0xFFFFFFF));
      v816.m128i_i64[0] = (__int64)&v805;
      v804 = (__int64 *)v302;
      if ( (unsigned __int8)sub_173F040((__int64)&v815, v302) )
      {
        v303 = *v805;
        v304 = sub_1643030(*v805);
        v305 = sub_1643030(*v804);
        v306 = sub_15A0680(v303, (unsigned int)(v304 - v305), 0);
        v307 = *(_QWORD *)(v12 + 8);
        v816.m128i_i16[0] = 257;
        v308 = sub_172C310(
                 v307,
                 (__int64)v805,
                 v306,
                 v815.m128i_i64,
                 0,
                 *(double *)a3.m128_u64,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64);
        v309 = *v804;
        v816.m128i_i16[0] = 257;
        v310 = sub_1648A60(56, 1u);
        v11 = (__int64)v310;
        if ( v310 )
          sub_15FC510((__int64)v310, (__int64)v308, v309, (__int64)&v815, 0);
        return v11;
      }
      if ( !sub_1642F90(*v804, 64) )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v553 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v10 + 40) + 56LL) + 40LL);
      v815.m128i_i64[0] = (__int64)&v816;
      v815.m128i_i64[1] = 0x300000000LL;
      v796 = sub_15E26F0(v553, 5292, 0, 0);
      sub_12A9700((__int64)&v815, &v804);
      v554 = *(_QWORD *)(v12 + 8);
      v812 = "bswap_permute_get_lo";
      LOWORD(v814) = 259;
      v806 = sub_172C570(
               v554,
               *(_QWORD *)(v796 + 24),
               v796,
               (__int64 *)v815.m128i_i64[0],
               v815.m128i_u32[2],
               (__int64 *)&v812,
               0);
      v555 = sub_15E26F0(v553, 5291, 0, 0);
      v815.m128i_i32[2] = 0;
      v797 = v555;
      sub_12A9700((__int64)&v815, &v804);
      v556 = *(_QWORD *)(v12 + 8);
      v812 = "bswap_permute_get_hi";
      LOWORD(v814) = 259;
      v807 = sub_172C570(
               v556,
               *(_QWORD *)(v797 + 24),
               v797,
               (__int64 *)v815.m128i_i64[0],
               v815.m128i_u32[2],
               (__int64 *)&v812,
               0);
      v557 = sub_15E26F0(v553, 4227, 0, 0);
      v815.m128i_i32[2] = 0;
      v798 = v557;
      sub_12A9700((__int64)&v815, &v806);
      v558 = (_QWORD *)sub_16498A0(v10);
      v559 = sub_1643350(v558);
      v812 = (char *)sub_159C470(v559, 0, 1u);
      sub_12A9700((__int64)&v815, &v812);
      v560 = (_QWORD *)sub_16498A0(v10);
      v561 = sub_1643350(v560);
      v812 = (char *)sub_159C470(v561, 291, 1u);
      sub_12A9700((__int64)&v815, &v812);
      v562 = *(_QWORD *)(v12 + 8);
      v812 = "bswap_permute_lo";
      LOWORD(v814) = 259;
      v563 = (__int64 *)sub_172C570(
                          v562,
                          *(_QWORD *)(v798 + 24),
                          v798,
                          (__int64 *)v815.m128i_i64[0],
                          v815.m128i_u32[2],
                          (__int64 *)&v812,
                          0);
      v815.m128i_i32[2] = 0;
      v808 = v563;
      sub_12A9700((__int64)&v815, &v807);
      v564 = (_QWORD *)sub_16498A0(v10);
      v565 = sub_1643350(v564);
      v812 = (char *)sub_159C470(v565, 0, 1u);
      sub_12A9700((__int64)&v815, &v812);
      v566 = (_QWORD *)sub_16498A0(v10);
      v567 = sub_1643350(v566);
      v812 = (char *)sub_159C470(v567, 291, 1u);
      sub_12A9700((__int64)&v815, &v812);
      v568 = *(_QWORD *)(v12 + 8);
      v812 = "bswap_permute_hi";
      LOWORD(v814) = 259;
      v809 = (__int64 *)sub_172C570(
                          v568,
                          *(_QWORD *)(v798 + 24),
                          v798,
                          (__int64 *)v815.m128i_i64[0],
                          v815.m128i_u32[2],
                          (__int64 *)&v812,
                          0);
      v569 = sub_15E26F0(v553, 4225, 0, 0);
      v815.m128i_i32[2] = 0;
      v570 = v569;
      sub_12A9700((__int64)&v815, &v809);
      sub_12A9700((__int64)&v815, &v808);
      v812 = "bswap_permute_pack";
      LOWORD(v814) = 259;
      v571 = sub_17287F0(v570, (__int64 *)v815.m128i_i64[0], v815.m128i_u32[2], (__int64)&v812, 0);
      v257 = v815.m128i_i64[0];
      v11 = v571;
      if ( (__m128i *)v815.m128i_i64[0] == &v816 )
        return v11;
      goto LABEL_365;
    case 8u:
    case 0x61u:
    case 0x8Cu:
    case 0xBBu:
    case 0xBCu:
    case 0xCEu:
      v54 = *(_DWORD *)(v10 + 20);
      goto LABEL_53;
    case 0x1Eu:
      goto LABEL_527;
    case 0x1Fu:
    case 0x21u:
      v377 = *(__int64 **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      v378 = (__int64)v377;
      sub_14C2530(
        (__int64)&v815,
        v377,
        *(_QWORD *)(v12 + 2664),
        0,
        *(_QWORD *)(v12 + 2640),
        v10,
        *(_QWORD *)(v12 + 2656),
        0);
      v379 = *(_QWORD *)(v10 - 24);
      if ( *(_BYTE *)(v379 + 16) )
        goto LABEL_1104;
      v768 = v816.m128i_u32[2];
      if ( *(_DWORD *)(v379 + 36) == 33 )
      {
        if ( v816.m128i_i32[2] > 0x40u )
        {
          v734 = sub_16A58A0((__int64)&v816);
        }
        else
        {
          _RAX = v816.m128i_i64[0];
          if ( v816.m128i_i64[0] )
            __asm { tzcnt   rax, rax }
          else
            LODWORD(_RAX) = 64;
          if ( v816.m128i_i32[2] < (unsigned int)_RAX )
            LODWORD(_RAX) = v816.m128i_i32[2];
          v734 = _RAX;
        }
        if ( v815.m128i_i32[2] > 0x40u )
        {
          LODWORD(_R13) = sub_16A58F0((__int64)&v815);
        }
        else
        {
          _R13 = ~v815.m128i_i64[0];
          if ( v815.m128i_i64[0] == -1 )
            LODWORD(_R13) = 64;
          else
            __asm { tzcnt   r13, r13 }
        }
      }
      else
      {
        if ( v816.m128i_i32[2] > 0x40u )
        {
          v734 = sub_16A57B0((__int64)&v816);
        }
        else
        {
          if ( v816.m128i_i64[0] )
          {
            _BitScanReverse64(&v380, v816.m128i_u64[0]);
            v381 = v380 ^ 0x3F;
          }
          else
          {
            v381 = 64;
          }
          v734 = v816.m128i_i32[2] + v381 - 64;
        }
        if ( v815.m128i_i32[2] > 0x40u )
        {
          v383 = sub_16A5810((__int64)&v815);
        }
        else if ( v815.m128i_i64[0] << (64 - v815.m128i_i8[8]) == -1 )
        {
          v383 = 64;
        }
        else
        {
          _BitScanReverse64(&v382, ~(v815.m128i_i64[0] << (64 - v815.m128i_i8[8])));
          v383 = v382 ^ 0x3F;
        }
        LODWORD(_R13) = v383;
      }
      if ( (_DWORD)_R13 == v734 )
      {
        v616 = sub_15A0680(*v377, (unsigned int)_R13, 0);
        v617 = *(_QWORD *)(v10 + 8);
        v618 = v616;
        if ( v617 )
        {
          v783 = (__int64 ***)v10;
          v619 = v12;
          v620 = *(_QWORD *)v12;
          v621 = v617;
          do
          {
            v622 = sub_1648700(v621);
            sub_170B990(v620, (__int64)v622);
            v621 = *(_QWORD *)(v621 + 8);
          }
          while ( v621 );
          v12 = v619;
          if ( v783 == (__int64 ***)v618 )
            v618 = sub_1599EF0(*v783);
          sub_164D160(
            (__int64)v783,
            v618,
            a3,
            *(double *)v13.m128i_i64,
            *(double *)v14.m128i_i64,
            a6,
            v623,
            v624,
            a9,
            a10);
          goto LABEL_488;
        }
LABEL_487:
        v11 = 0;
        goto LABEL_488;
      }
      if ( v768 <= 0x40 )
        v385 = v816.m128i_i64[0] == 0;
      else
        v385 = (unsigned int)sub_16A57B0((__int64)&v816) == v768;
      if ( v385 )
      {
        v378 = *(_QWORD *)(v12 + 2664);
        if ( !(unsigned __int8)sub_14BFF20(
                                 (__int64)v377,
                                 v378,
                                 0,
                                 *(_QWORD *)(v12 + 2640),
                                 v10,
                                 *(_QWORD *)(v12 + 2656)) )
          goto LABEL_514;
      }
      v386 = *(__int64 **)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
      v387 = *((_BYTE *)v386 + 16);
      if ( v387 == 13 )
      {
        if ( *((_DWORD *)v386 + 8) <= 0x40u )
        {
          v388 = v386[3] == 1;
        }
        else
        {
          v769 = *((_DWORD *)v386 + 8);
          v388 = v769 - 1 == (unsigned int)sub_16A57B0((__int64)(v386 + 3));
        }
      }
      else
      {
        if ( *(_BYTE *)(*v386 + 8) != 16 || v387 > 0x10u )
          goto LABEL_520;
        v781 = *(_BYTE **)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
        v615 = sub_15A1020(v781, v378, (__int64)v386, *v386);
        if ( !v615 || *(_BYTE *)(v615 + 16) != 13 )
        {
          v722 = v377;
          v625 = (__int64)v781;
          v707 = _R13;
          v626 = 0;
          v784 = *(_DWORD *)(*(_QWORD *)v781 + 32LL);
          while ( v784 != v626 )
          {
            v627 = sub_15A0A60(v625, v626);
            if ( !v627 )
              goto LABEL_520;
            v628 = *(_BYTE *)(v627 + 16);
            if ( v628 != 9 )
            {
              if ( v628 != 13 )
                goto LABEL_520;
              if ( *(_DWORD *)(v627 + 32) <= 0x40u )
              {
                v629 = *(_QWORD *)(v627 + 24) == 1;
              }
              else
              {
                v701 = *(_DWORD *)(v627 + 32);
                v629 = v701 - 1 == (unsigned int)sub_16A57B0(v627 + 24);
              }
              if ( !v629 )
                goto LABEL_520;
            }
            ++v626;
          }
          v377 = v722;
          LODWORD(_R13) = v707;
LABEL_514:
          v374 = *v377;
          if ( *(_BYTE *)(v374 + 8) == 11
            && *(_DWORD *)(v374 + 8) >> 8 != 1
            && (!*(_QWORD *)(v10 + 48) && *(__int16 *)(v10 + 18) >= 0 || !sub_1625790(v10, 4)) )
          {
            v389 = sub_159C470(v374, (unsigned int)_R13, 0);
            v812 = (char *)sub_1624210(v389);
            v390 = (unsigned int)(v734 + 1);
LABEL_519:
            v391 = sub_159C470(v374, v390, 0);
            v813 = (__int64 **)sub_1624210(v391);
            v392 = (__int64 *)sub_16498A0(v10);
            v393 = sub_1627350(v392, (__int64 *)&v812, (__int64 *)2, 0, 1);
            sub_1625C10(v10, 4, v393);
LABEL_488:
            if ( v816.m128i_i32[2] > 0x40u && v816.m128i_i64[0] )
              j_j___libc_free_0_0(v816.m128i_i64[0]);
            if ( v815.m128i_i32[2] > 0x40u && v815.m128i_i64[0] )
              j_j___libc_free_0_0(v815.m128i_i64[0]);
            if ( v11 )
              return v11;
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
          goto LABEL_487;
        }
        if ( *(_DWORD *)(v615 + 32) <= 0x40u )
        {
          v388 = *(_QWORD *)(v615 + 24) == 1;
        }
        else
        {
          v782 = *(_DWORD *)(v615 + 32);
          v388 = v782 - 1 == (unsigned int)sub_16A57B0(v615 + 24);
        }
      }
      if ( v388 )
        goto LABEL_514;
LABEL_520:
      v394 = sub_159C4F0(*(__int64 **)(*(_QWORD *)(v12 + 8) + 24LL));
      v395 = (__int64 *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
      if ( *v395 )
      {
        v396 = v395[1];
        v397 = v395[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v397 = v396;
        if ( v396 )
          *(_QWORD *)(v396 + 16) = v397 | *(_QWORD *)(v396 + 16) & 3LL;
      }
      *v395 = v394;
      if ( v394 )
      {
        v398 = *(_QWORD *)(v394 + 8);
        v395[1] = v398;
        if ( v398 )
          *(_QWORD *)(v398 + 16) = (unsigned __int64)(v395 + 1) | *(_QWORD *)(v398 + 16) & 3LL;
        v395[2] = (v394 + 8) | v395[2] & 3;
        *(_QWORD *)(v394 + 8) = v395;
      }
      goto LABEL_488;
    case 0x20u:
      v373 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      v374 = *(_QWORD *)v373;
      if ( *(_BYTE *)(*(_QWORD *)v373 + 8LL) != 11 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v375 = *(_DWORD *)(v374 + 8) >> 8;
      v815.m128i_i32[2] = v375;
      if ( v375 > 0x40 )
      {
        v774 = v373;
        v736 = v375;
        sub_16A4EF0((__int64)&v815, 0, 0);
        v816.m128i_i32[2] = v736;
        sub_16A4EF0((__int64)&v816, 0, 0);
        v373 = v774;
      }
      else
      {
        v815.m128i_i64[0] = 0;
        v816.m128i_i32[2] = v375;
        v816.m128i_i64[0] = 0;
      }
      sub_14BB090(
        v373,
        (__int64)&v815,
        *(_QWORD *)(v12 + 2664),
        0,
        *(_QWORD *)(v12 + 2640),
        v10,
        *(_QWORD *)(v12 + 2656),
        0);
      if ( v816.m128i_i32[2] > 0x40u )
        v767 = sub_16A5940((__int64)&v816);
      else
        v767 = sub_39FAC40(v816.m128i_i64[0]);
      v733 = v815.m128i_i32[2];
      if ( v815.m128i_i32[2] > 0x40u )
        v376 = sub_16A5940((__int64)&v815);
      else
        v376 = sub_39FAC40(v815.m128i_i64[0]);
      if ( *(_DWORD *)(v374 + 8) >> 8 == 1
        || (*(_QWORD *)(v10 + 48) || *(__int16 *)(v10 + 18) < 0) && sub_1625790(v10, 4) )
      {
        goto LABEL_487;
      }
      v579 = sub_159C470(v374, v767, 0);
      v812 = (char *)sub_1624210(v579);
      v390 = (unsigned int)(v733 + 1 - v376);
      goto LABEL_519;
    case 0x4Cu:
      v370 = sub_164F980(v10);
      v371 = v370;
      if ( !*(_QWORD *)(v10 + 8) )
        return sub_170BC50(v12, v10);
      v372 = *(_BYTE *)(v370 + 16);
      if ( v372 == 9 )
        goto LABEL_68;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 15 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      if ( v372 == 15 )
      {
        v67 = sub_1599A20(*(__int64 ***)v10);
        return sub_170E100(
                 (__int64 *)v12,
                 v10,
                 v67,
                 a3,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64,
                 a6,
                 v28,
                 v29,
                 a9,
                 a10);
      }
      if ( (unsigned __int8)sub_14BFF20(
                              v371,
                              *(_QWORD *)(v12 + 2664),
                              0,
                              *(_QWORD *)(v12 + 2640),
                              v10,
                              *(_QWORD *)(v12 + 2656)) )
        sub_173DC10(v10, 0, 32);
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    case 0x4Fu:
      v409 = sub_173FD00(*(_QWORD *)(v10 + 40) + 40LL, v10);
      if ( dword_4FA2600 )
      {
        v410 = 0;
        do
        {
          if ( !(unsigned __int8)sub_14AF470(v409, 0, 0, 0) )
            break;
          v411 = *(_QWORD *)(v409 + 32);
          if ( v411 == *(_QWORD *)(v409 + 40) + 40LL || !v411 )
            v409 = 0;
          else
            v409 = v411 - 24;
          ++v410;
        }
        while ( v410 < dword_4FA2600 );
      }
      v812 = 0;
      v815.m128i_i32[0] = 79;
      v815.m128i_i32[2] = 0;
      v816.m128i_i64[0] = (__int64)&v812;
      if ( !(unsigned __int8)sub_173FD20((__int64)&v815, v409) )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v792 = *(__int64 **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      if ( v812 == (char *)v792 )
        return sub_170BC50(v12, v409);
      for ( m = (_QWORD *)sub_173FD00(*(_QWORD *)(v10 + 40) + 40LL, v10); m != (_QWORD *)v409; m = (_QWORD *)v773 )
      {
        v773 = sub_173FD00(m[5] + 40LL, (__int64)m);
        sub_15F22F0(m, v10);
      }
      v816.m128i_i16[0] = 257;
      v457 = sub_1729500(
               *(_QWORD *)(v12 + 8),
               (unsigned __int8 *)v792,
               (__int64)v812,
               v815.m128i_i64,
               *(double *)a3.m128_u64,
               *(double *)v13.m128i_i64,
               *(double *)v14.m128i_i64);
      sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), (__int64)v457);
      return sub_170BC50(v12, (__int64)m);
    case 0x60u:
      v54 = *(_DWORD *)(v10 + 20);
      v401 = *(_QWORD *)(v10 - 24LL * (v54 & 0xFFFFFFF));
      if ( *(_BYTE *)(v401 + 16) == 79 )
      {
        v402 = *(_QWORD *)(v401 - 72);
        if ( v402 )
        {
          v403 = *(char **)(v401 - 48);
          if ( (unsigned __int8)v403[16] <= 0x10u )
          {
            v404 = *(char **)(v401 - 24);
            if ( (unsigned __int8)v404[16] <= 0x10u )
            {
              v816.m128i_i16[0] = 257;
              v405 = *(_QWORD *)(v12 + 8);
              v812 = v403;
              if ( !*(_BYTE *)(v30 + 16) )
              {
                v406 = sub_172C570(v405, *(_QWORD *)(v30 + 24), v30, (__int64 *)&v812, 1, v815.m128i_i64, 0);
                v407 = *(_QWORD *)(v10 - 24);
                v816.m128i_i16[0] = 257;
                v812 = v404;
                v791 = (_QWORD *)v406;
                if ( !*(_BYTE *)(v407 + 16) )
                {
                  v408 = sub_172C570(
                           *(_QWORD *)(v12 + 8),
                           *(_QWORD *)(v407 + 24),
                           v407,
                           (__int64 *)&v812,
                           1,
                           v815.m128i_i64,
                           0);
                  v816.m128i_i16[0] = 257;
                  return sub_14EDD70(v402, v791, v408, (__int64)&v815, 0, 0);
                }
              }
LABEL_1105:
              BUG();
            }
          }
        }
      }
LABEL_53:
      v55 = *(_QWORD *)(v10 - 24LL * (v54 & 0xFFFFFFF));
      v56 = *(_QWORD *)(v55 + 8);
      if ( !v56 || *(_QWORD *)(v56 + 8) )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v57 = *(unsigned __int8 *)(v55 + 16);
      if ( (unsigned __int8)v57 > 0x17u )
      {
        v58 = v57 - 24;
      }
      else
      {
        if ( (_BYTE)v57 != 5 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v58 = *(unsigned __int16 *)(v55 + 18);
      }
      if ( v58 == 44 )
      {
        v59 = (*(_BYTE *)(v55 + 23) & 0x40) != 0
            ? *(__int64 ***)(v55 - 8)
            : (__int64 **)(v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF));
        v60 = *v59;
        if ( v60 )
        {
          v816.m128i_i16[0] = 257;
          v61 = *(__int64 **)(v12 + 8);
          v812 = (char *)v60;
          if ( !*(_BYTE *)(v30 + 16) )
          {
            v62 = sub_15E8450(v61, v39, (__int64 **)&v812, 1, v10, (int)&v815);
            v63 = *(__int64 ***)v10;
            v816.m128i_i16[0] = 257;
            v64 = (__int64)v62;
            v65 = sub_1648A60(56, 1u);
            v11 = (__int64)v65;
            if ( v65 )
              sub_15FCB10((__int64)v65, v64, (__int64)v63, (__int64)&v815, 0);
            return v11;
          }
LABEL_1104:
          BUG();
        }
      }
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    case 0x63u:
      goto LABEL_450;
    case 0x64u:
      if ( sub_15F2480(v10) )
      {
        v606 = *(_QWORD *)(v12 + 8);
        v780 = *(_DWORD *)(v606 + 40);
        v801 = *(_QWORD *)(v606 + 32);
        *(_DWORD *)(v606 + 40) = sub_15F24E0(v10);
        v607 = *(_DWORD *)(v10 + 20);
        v608 = *(_QWORD *)(v12 + 8);
        v816.m128i_i16[0] = 257;
        v609 = sub_173E060(
                 v608,
                 *(_QWORD *)(v10 - 24LL * (v607 & 0xFFFFFFF)),
                 *(_QWORD *)(v10 + 24 * (1LL - (v607 & 0xFFFFFFF))),
                 v815.m128i_i64,
                 0,
                 *(double *)a3.m128_u64,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64);
        v610 = *(_DWORD *)(v10 + 20);
        v611 = *(_QWORD *)(v12 + 8);
        v816.m128i_i16[0] = 257;
        v612 = sub_173E220(
                 v611,
                 (__int64)v609,
                 *(_QWORD *)(v10 + 24 * (2LL - (v610 & 0xFFFFFFF))),
                 v815.m128i_i64,
                 0,
                 *(double *)a3.m128_u64,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64);
        sub_164B7C0((__int64)v612, v10);
        v11 = sub_170E100(
                (__int64 *)v12,
                v10,
                (__int64)v612,
                a3,
                *(double *)v13.m128i_i64,
                *(double *)v14.m128i_i64,
                a6,
                v613,
                v614,
                a9,
                a10);
        *(_DWORD *)(v606 + 40) = v780;
        *(_QWORD *)(v606 + 32) = v801;
        return v11;
      }
LABEL_450:
      v340 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v341 = -24 * v340;
      v342 = 1 - v340;
      v343 = 24 * (1 - v340);
      v344 = (__int64 **)(v10 + v341);
      v345 = *v344;
      v346 = *(_QWORD *)(v10 + v343);
      if ( *((_BYTE *)*v344 + 16) <= 0x10u && *(_BYTE *)(v346 + 16) > 0x10u )
      {
        v776 = *(_QWORD *)(v10 + v343);
        sub_1593B40(v344, v776);
        sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), (__int64)v345);
        v578 = v345;
        v345 = (__int64 *)v776;
        v346 = (__int64)v578;
      }
      v731 = v346;
      v813 = &v808;
      v347 = sub_171FB50((__int64)&v812, (__int64)v345, v342, v32);
      v350 = v731;
      if ( v347
        && (v815.m128i_i64[1] = (__int64)&v809, v351 = sub_171FB50((__int64)&v815, v731, v348, v349), v350 = v731, v351) )
      {
        sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), (__int64)v808);
        sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), (__int64)v809);
      }
      else
      {
        v765 = v350;
        LODWORD(v812) = 96;
        LODWORD(v813) = 0;
        v814 = &v808;
        v352 = sub_173F0F0((__int64)&v812, (__int64)v345);
        v353 = v765;
        if ( v352
          && (v354 = v808,
              v815.m128i_i32[0] = 96,
              v815.m128i_i32[2] = 0,
              v816.m128i_i64[0] = (__int64)v808,
              v355 = sub_173F140((__int64)&v815, v765),
              v353 = v765,
              v355) )
        {
          sub_1593B40((_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)), (__int64)v354);
          sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), (__int64)v808);
        }
        else
        {
          v815.m128i_i64[0] = 0x3FF0000000000000LL;
          if ( !(unsigned __int8)sub_13D6AF0((double *)v815.m128i_i64, v353) )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          v356 = *(_DWORD *)(v10 + 20);
          v816.m128i_i16[0] = 257;
          v357 = sub_15FB440(12, v345, *(_QWORD *)(v10 + 24 * (2LL - (v356 & 0xFFFFFFF))), (__int64)&v815, 0);
          sub_15F2500(v357, v10);
          return v357;
        }
      }
      return v10;
    case 0x73u:
    case 0xCBu:
      v425 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      v426 = sub_1649C60(v425);
      v427 = (__int64 *)sub_164AA50(v425);
      if ( (__int64 *)v426 == v427 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v428 = *(_QWORD *)(v10 - 24);
      if ( *(_BYTE *)(v428 + 16) )
        goto LABEL_1104;
      v429 = *v427;
      v201 = *(_DWORD *)(v428 + 36) == 115;
      v430 = *(_QWORD *)(v12 + 8);
      v431 = *v427;
      if ( v201 )
      {
        if ( *(_BYTE *)(v429 + 8) == 16 )
          v431 = **(_QWORD **)(v429 + 16);
        v778 = v427;
        v601 = (__int64 **)sub_16471D0(*(_QWORD **)(v430 + 24), *(_DWORD *)(v431 + 8) >> 8);
        v602 = v778;
        v603 = v601;
        if ( (__int64 **)v429 != v601 )
        {
          v604 = (__int64)v778;
          v816.m128i_i16[0] = 257;
          v779 = v601;
          v605 = sub_1708970(v430, 47, v604, v601, v815.m128i_i64);
          v603 = v779;
          v602 = (__int64 *)v605;
        }
        v437 = 115;
        v735 = v602;
        v772 = v603;
        v438 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v430 + 8) + 56LL) + 40LL);
        v815.m128i_i64[0] = (__int64)v603;
      }
      else
      {
        if ( *(_BYTE *)(v429 + 8) == 16 )
          v431 = **(_QWORD **)(v429 + 16);
        v770 = v427;
        v432 = (__int64 **)sub_16471D0(*(_QWORD **)(v430 + 24), *(_DWORD *)(v431 + 8) >> 8);
        v433 = v770;
        v434 = v432;
        if ( (__int64 **)v429 != v432 )
        {
          v435 = (__int64)v770;
          v816.m128i_i16[0] = 257;
          v771 = v432;
          v436 = sub_1708970(v430, 47, v435, v432, v815.m128i_i64);
          v434 = v771;
          v433 = (__int64 *)v436;
        }
        v437 = 203;
        v735 = v433;
        v772 = v434;
        v438 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v430 + 8) + 56LL) + 40LL);
        v815.m128i_i64[0] = (__int64)v434;
      }
      v439 = sub_15E26F0(v438, v437, v815.m128i_i64, 1);
      v816.m128i_i16[0] = 257;
      v812 = (char *)v735;
      v440 = sub_172C570(v430, *(_QWORD *)(v439 + 24), v439, (__int64 *)&v812, 1, v815.m128i_i64, 0);
      v67 = v440;
      if ( (__int64 **)v429 != v772 )
      {
        v816.m128i_i16[0] = 257;
        v67 = (__int64)sub_1708970(v430, 47, v440, (__int64 **)v429, v815.m128i_i64);
      }
      v441 = *(_QWORD *)v67;
      if ( *(_BYTE *)(*(_QWORD *)v67 + 8LL) == 16 )
        v441 = **(_QWORD **)(v441 + 16);
      v442 = *(__int64 ***)v10;
      v443 = *(_DWORD *)(v441 + 8) >> 8;
      v444 = *(__int64 ***)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
        v444 = (__int64 **)*v442[2];
      if ( v443 != *((_DWORD *)v444 + 2) >> 8 )
      {
        v445 = *(_QWORD *)(v12 + 8);
        v816.m128i_i16[0] = 257;
        v446 = sub_1708970(v445, 48, v67, v442, v815.m128i_i64);
        v442 = *(__int64 ***)v10;
        v67 = (__int64)v446;
      }
      if ( *(__int64 ***)v67 == v442 )
        return sub_170E100(
                 (__int64 *)v12,
                 v10,
                 v67,
                 a3,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64,
                 a6,
                 v28,
                 v29,
                 a9,
                 a10);
      v447 = *(_QWORD *)(v12 + 8);
      v816.m128i_i16[0] = 257;
      v67 = (__int64)sub_1708970(v447, 47, v67, v442, v815.m128i_i64);
      if ( v67 )
        return sub_170E100(
                 (__int64 *)v12,
                 v10,
                 v67,
                 a3,
                 *(double *)v13.m128i_i64,
                 *(double *)v14.m128i_i64,
                 a6,
                 v28,
                 v29,
                 a9,
                 a10);
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    case 0x75u:
      v423 = sub_15F2060(v10);
      if ( !(unsigned __int8)sub_1560180(v423 + 112, 42) )
      {
        v424 = sub_15F2060(v10);
        if ( !(unsigned __int8)sub_1560180(v424 + 112, 43) )
        {
          if ( (unsigned __int8)sub_173FF40(v10, 117, 116, v12) )
            return 0;
        }
      }
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
    case 0x80u:
      v448 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v449 = *(_QWORD *)(v10 + 24 * (2 - v448));
      if ( *(_BYTE *)(v449 + 16) > 0x10u )
        return 0;
      if ( !sub_1593BB0(v449, v26, v448, v32) )
        return 0;
      v450 = *(_QWORD *)(v10 + 8);
      v451 = *(_QWORD *)(v10 + 24 * (3LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
      if ( !v450 )
        return 0;
      v452 = *(_QWORD *)v12;
      do
      {
        v453 = sub_1648700(v450);
        sub_170B990(v452, (__int64)v453);
        v450 = *(_QWORD *)(v450 + 8);
      }
      while ( v450 );
      if ( v10 == v451 )
        v451 = sub_1599EF0(*(__int64 ***)v10);
      sub_164D160(v10, v451, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v454, v455, a9, a10);
      return v11;
    case 0x81u:
      v363 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v364 = 24 * (2 - v363);
      v365 = *(_BYTE **)(v10 + v364);
      if ( v365[16] > 0x10u )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v328 = *(_QWORD *)(v12 + 8);
      if ( sub_1596070(*(_QWORD *)(v10 + v364), v26, v363, v32) || v365[16] == 9 )
        goto LABEL_435;
      v732 = v328;
      v366 = 0;
      v766 = *(_DWORD *)(*(_QWORD *)v365 + 32LL);
      while ( 2 )
      {
        if ( v766 == v366 )
        {
          v328 = v732;
LABEL_435:
          v329 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
          v330 = *(_QWORD *)(v10 + 24 * (1 - v329));
          v331 = *(_QWORD **)(v330 + 24);
          if ( *(_DWORD *)(v330 + 32) > 0x40u )
            v331 = (_QWORD *)*v331;
          v764 = (unsigned int)v331;
          v729 = *(_QWORD *)(v10 - 24 * v329);
          v815.m128i_i64[0] = (__int64)"unmaskedload";
          v816.m128i_i16[0] = 259;
          v332 = sub_1648A60(64, 1u);
          v120 = (__int64)v332;
          if ( v332 )
            sub_15F9210((__int64)v332, *(_QWORD *)(*(_QWORD *)v729 + 24LL), v729, 0, 0, 0);
          v333 = *(_QWORD *)(v328 + 8);
          if ( v333 )
          {
            v730 = *(__int64 **)(v328 + 16);
            sub_157E9D0(v333 + 40, v120);
            v334 = *v730;
            v335 = *(_QWORD *)(v120 + 24) & 7LL;
            *(_QWORD *)(v120 + 32) = v730;
            v334 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v120 + 24) = v334 | v335;
            *(_QWORD *)(v334 + 8) = v120 + 24;
            *v730 = *v730 & 7 | (v120 + 24);
          }
          sub_164B780(v120, v815.m128i_i64);
          v812 = (char *)v120;
          if ( !*(_QWORD *)(v328 + 80) )
            sub_4263D6(v120, &v815, v336);
          (*(void (__fastcall **)(__int64, char **))(v328 + 88))(v328 + 64, &v812);
          v337 = *(__int64 **)v328;
          if ( *(_QWORD *)v328 )
          {
            v812 = *(char **)v328;
            sub_1623A60((__int64)&v812, (__int64)v337, 2);
            v338 = *(_QWORD *)(v120 + 48);
            if ( v338 )
              sub_161E7C0(v120 + 48, v338);
            v339 = (unsigned __int8 *)v812;
            *(_QWORD *)(v120 + 48) = v812;
            if ( v339 )
              sub_1623210((__int64)&v812, v339, v120 + 48);
            sub_15F8F50(v120, v764);
            goto LABEL_448;
          }
          sub_15F8F50(v120, v764);
          if ( v120 )
            goto LABEL_448;
        }
        else
        {
          v367 = sub_15A0A60((__int64)v365, v366);
          if ( v367 )
          {
            v713 = v367;
            if ( sub_1596070(v367, v366, v368, v369) || *(_BYTE *)(v713 + 16) == 9 )
            {
              ++v366;
              continue;
            }
          }
        }
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      }
    case 0x82u:
      v23 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v24 = *(_QWORD *)(v10 + 24 * (3 - v23));
      if ( *(_BYTE *)(v24 + 16) > 0x10u || !sub_1593BB0(v24, v26, v23, v32) )
        return 0;
      return sub_170BC50(v12, v10);
    case 0x83u:
      v412 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v413 = 24 * (3 - v412);
      v414 = *(_QWORD *)(v10 + v413);
      if ( *(_BYTE *)(v414 + 16) > 0x10u )
        return 0;
      if ( sub_1593BB0(*(_QWORD *)(v10 + v413), v26, v412, v32) )
        return sub_170BC50(v12, v10);
      if ( !sub_1596070(v414, v26, v415, v416) )
        return 0;
      v417 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v418 = *(_QWORD *)(v10 + 24 * (1 - v417));
      v419 = *(_QWORD *)(v10 + 24 * (2 - v417));
      v420 = *(_QWORD **)(v419 + 24);
      if ( *(_DWORD *)(v419 + 32) > 0x40u )
        v420 = (_QWORD *)*v420;
      v421 = *(_QWORD *)(v10 - 24 * v417);
      v422 = sub_1648A60(64, 2u);
      v11 = (__int64)v422;
      if ( v422 )
        sub_15F9630((__int64)v422, v421, v418, 0, (unsigned int)v420, 0);
      return v11;
    case 0x84u:
    case 0x8Bu:
      v263 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v264 = (__int64 *)(1 - v263);
      v265 = (__int64 *)(v10 - 24 * v263);
      v266 = *v265;
      v267 = *(_QWORD *)(v10 + 24 * (1 - v263));
      v268 = *(unsigned __int8 *)(*v265 + 16);
      if ( (_BYTE)v268 == 14 )
      {
        if ( *(_BYTE *)(v267 + 16) != 14 )
        {
          v660 = v267;
          v803 = *v265;
          v11 = v10;
          sub_1593B40(v265, v660);
          sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v803);
          return v11;
        }
        if ( v267 == v266 )
          goto LABEL_386;
      }
      else
      {
        if ( v267 == v266 )
          goto LABEL_386;
        v661 = *(_BYTE *)(v267 + 16);
        if ( v661 != 14 )
        {
          if ( (_BYTE)v268 == 9 )
            goto LABEL_385;
          if ( v661 == 9 )
            goto LABEL_386;
          v809 = 0;
          v812 = 0;
          if ( v39 != 139 )
          {
            if ( v661 == 78 )
            {
              v662 = *(_QWORD *)(v267 - 24);
              if ( !*(_BYTE *)(v662 + 16) && *(_DWORD *)(v662 + 36) == 132 )
              {
                v663 = v267 & 0xFFFFFFFFFFFFFFF8LL;
                v264 = (__int64 *)(*(_DWORD *)((v267 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
                v664 = (__int64 **)((v267 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (_QWORD)v264);
                v665 = *v664;
                if ( *v664 && (v809 = *v664, (v666 = *(_QWORD *)(v663 + 24 * (1LL - (_QWORD)v264))) != 0) )
                {
                  v812 = *(char **)(v663 + 24 * (1LL - (_QWORD)v264));
                  if ( (__int64 *)v266 == v665 || v266 == v666 )
                    goto LABEL_385;
                }
                else
                {
                  v25 = 0;
                }
              }
            }
LABEL_380:
            if ( (_BYTE)v268 == 78 )
            {
              v668 = *(_QWORD *)(v266 - 24);
              if ( !*(_BYTE *)(v668 + 16) && *(_DWORD *)(v668 + 36) == 132 )
              {
                v669 = v266 & 0xFFFFFFFFFFFFFFF8LL;
                v268 = *(_DWORD *)((v266 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
                v670 = (__int64 **)((v266 & 0xFFFFFFFFFFFFFFF8LL) - 24 * v268);
                v264 = *v670;
                if ( *v670 )
                {
                  v809 = *v670;
                  v671 = *(_QWORD *)(v669 + 24 * (1 - v268));
                  if ( v671 )
                  {
                    v812 = *(char **)(v669 + 24 * (1 - v268));
                    if ( (__int64 *)v267 == v264 || v267 == v671 )
                      goto LABEL_386;
                  }
                }
              }
            }
            if ( v25 )
            {
              v762 = v266;
              v271 = sub_16982C0();
              v266 = v762;
              if ( *(void **)(v25 + 32) == v271 )
              {
                v667 = *(_QWORD *)(v25 + 40);
                if ( (*(_BYTE *)(v667 + 26) & 7) == 0 )
                {
                  v272 = v667 + 8;
LABEL_384:
                  if ( (*(_BYTE *)(v272 + 18) & 8) == 0 )
                  {
LABEL_385:
                    v266 = v267;
LABEL_386:
                    v67 = v266;
                    return sub_170E100(
                             (__int64 *)v12,
                             v10,
                             v67,
                             a3,
                             *(double *)v13.m128i_i64,
                             *(double *)v14.m128i_i64,
                             a6,
                             v28,
                             v29,
                             a9,
                             a10);
                  }
                }
              }
              else
              {
                v272 = v25 + 32;
                if ( (*(_BYTE *)(v25 + 50) & 7) == 0 )
                  goto LABEL_384;
              }
            }
LABEL_1030:
            v745 = v266;
            v813 = &v808;
            if ( (unsigned __int8)sub_171FB50((__int64)&v812, v266, v268, (__int64)v264) )
            {
              v815.m128i_i64[1] = (__int64)&v809;
              if ( (unsigned __int8)sub_171FB50((__int64)&v815, v267, v675, v676) )
              {
                if ( (v677 = *(_QWORD *)(v745 + 8)) != 0 && !*(_QWORD *)(v677 + 8)
                  || (v678 = *(_QWORD *)(v267 + 8)) != 0 && !*(_QWORD *)(v678 + 8) )
                {
                  v679 = *(_QWORD *)(v10 - 24);
                  if ( !*(_BYTE *)(v679 + 16) )
                  {
                    v680 = 139;
                    v681 = *(__int64 **)(v12 + 8);
                    if ( *(_DWORD *)(v679 + 36) != 132 )
                      v680 = 132;
                    v816.m128i_i16[0] = 257;
                    v812 = (char *)v808;
                    v813 = (__int64 **)v809;
                    v682 = sub_15E8450(v681, v680, (__int64 **)&v812, 2, v10, (int)&v815);
                    v816.m128i_i16[0] = 257;
                    v11 = sub_15FB5B0(v682, (__int64)&v815, 0, v683);
                    sub_15F2530((unsigned __int8 *)v11, v10, 1);
                    return v11;
                  }
                  goto LABEL_1104;
                }
              }
            }
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          }
LABEL_1019:
          v723 = (__int64 *)v266;
          v816.m128i_i64[0] = (__int64)&v809;
          v817 = (__int64 **)&v812;
          v815.m128i_i32[0] = 139;
          v815.m128i_i32[2] = 0;
          v816.m128i_i32[2] = 1;
          if ( (unsigned __int8)sub_173EFC0((__int64)&v815, v267) && (v723 == v809 || v723 == (__int64 *)v812) )
            goto LABEL_385;
          v817 = (__int64 **)&v812;
          v815.m128i_i32[0] = 139;
          v815.m128i_i32[2] = 0;
          v816.m128i_i64[0] = (__int64)&v809;
          v816.m128i_i32[2] = 1;
          v672 = sub_173EFC0((__int64)&v815, (__int64)v723);
          v266 = (__int64)v723;
          if ( v672 && ((__int64 *)v267 == v809 || (char *)v267 == v812) )
            goto LABEL_386;
          if ( !v25 )
            goto LABEL_1030;
          v673 = sub_16982C0();
          v266 = (__int64)v723;
          if ( *(void **)(v25 + 32) == v673 )
          {
            v684 = *(_QWORD *)(v25 + 40);
            if ( (*(_BYTE *)(v684 + 26) & 7) != 0 )
              goto LABEL_1030;
            v674 = v684 + 8;
          }
          else
          {
            if ( (*(_BYTE *)(v25 + 50) & 7) != 0 )
              goto LABEL_1030;
            v674 = v25 + 32;
          }
          if ( (*(_BYTE *)(v674 + 18) & 8) != 0 )
            goto LABEL_385;
          goto LABEL_1030;
        }
      }
      v712 = *v265;
      v727 = *(_BYTE *)(*v265 + 16);
      v761 = v39;
      v269 = sub_16982C0();
      v268 = v727;
      v266 = v712;
      if ( *(void **)(v267 + 32) == v269 )
        v270 = *(_QWORD *)(v267 + 40) + 8LL;
      else
        v270 = v267 + 32;
      if ( (*(_BYTE *)(v270 + 18) & 7) == 1 )
        goto LABEL_386;
      if ( v727 == 9 )
        goto LABEL_385;
      v809 = 0;
      v25 = v267;
      v812 = 0;
      if ( v761 != 139 )
        goto LABEL_380;
      goto LABEL_1019;
    case 0x90u:
      v67 = sub_140EAC0((__int64 *)v10, *(_QWORD *)(v12 + 2664), *(_QWORD *)(v12 + 2648), 0);
      if ( !v67 )
        return 0;
      return sub_170E100(
               (__int64 *)v12,
               v10,
               v67,
               a3,
               *(double *)v13.m128i_i64,
               *(double *)v14.m128i_i64,
               a6,
               v28,
               v29,
               a9,
               a10);
    case 0x93u:
      v358 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      v359 = *(_QWORD *)(v10 + 24 * (1 - v358));
      if ( *(_BYTE *)(v359 + 16) != 13 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      if ( sub_1454FB0(v359 + 24) )
      {
        v360 = *(__int64 ***)v10;
        v816.m128i_i16[0] = 257;
        v361 = *(_QWORD *)(v10 - 24 * v358);
        v362 = (__int64 *)sub_15A10B0((__int64)v360, 1.0);
        return sub_15FB440(19, v362, v361, (__int64)&v815, 0);
      }
      if ( *(_DWORD *)(v359 + 32) > 0x40u )
      {
        v777 = *(_DWORD *)(v359 + 32);
        if ( v777 - (unsigned int)sub_16A57B0(v359 + 24) > 0x40 )
          return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        v599 = **(_QWORD **)(v359 + 24);
      }
      else
      {
        v599 = *(_QWORD *)(v359 + 24);
      }
      if ( v599 != 2 )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v816.m128i_i16[0] = 257;
      return sub_15FB440(16, *(__int64 **)(v10 - 24 * v358), *(_QWORD *)(v10 - 24 * v358), (__int64)&v815, 0);
    case 0xBDu:
    case 0xC3u:
    case 0xD1u:
    case 0xD2u:
      v40 = *(_DWORD *)(v10 + 20);
      v41 = v40 & 0xFFFFFFF;
      v42 = (__int64 *)(v10 - 24 * v41);
      v43 = *v42;
      if ( *(_BYTE *)(*v42 + 16) > 0x10u )
        goto LABEL_47;
      v44 = *(_QWORD *)(v10 + 24 * (1 - v41));
      if ( *(_BYTE *)(v44 + 16) <= 0x10u )
        goto LABEL_47;
      sub_1593B40(v42, v44);
      v640 = v43;
      v11 = v10;
      sub_1593B40((_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))), v640);
      return v11;
    case 0xC6u:
    case 0xD3u:
      v40 = *(_DWORD *)(v10 + 20);
LABEL_47:
      v45 = v39 - 189;
      v46 = 6;
      if ( (unsigned int)v45 <= 0x16 )
        v46 = dword_42B1FE0[v45];
      v808 = 0;
      v809 = 0;
      if ( !(unsigned __int8)sub_175CDB0(
                               v12,
                               v46,
                               *(_QWORD *)(v10 - 24LL * (v40 & 0xFFFFFFF)),
                               *(_QWORD *)(v10 + 24 * (1LL - (v40 & 0xFFFFFFF))),
                               v10,
                               (unsigned int)&v808,
                               (__int64)&v809) )
        return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
      v47 = v809;
      v786 = v808;
      v48 = (__int64 *)sub_1599EF0((__int64 **)*v808);
      v49 = *(__int64 ***)v10;
      v812 = (char *)v48;
      v813 = (__int64 **)v47;
      v51 = sub_159F090(v49, (__int64 *)&v812, 2, v50);
      v816.m128i_i16[0] = 257;
      LODWORD(v807) = 0;
      v52 = (__int64 *)v51;
      v53 = sub_1648A60(88, 2u);
      v11 = (__int64)v53;
      if ( v53 )
      {
        sub_15F1EA0((__int64)v53, *v52, 63, (__int64)(v53 - 6), 2, 0);
        *(_QWORD *)(v11 + 64) = 0x400000000LL;
        *(_QWORD *)(v11 + 56) = v11 + 72;
        sub_15FAD90(v11, (__int64)v52, (__int64)v786, &v807, 1, (__int64)&v815);
      }
      return v11;
    case 0xC9u:
      v212 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      if ( !v212 || !sub_173DBE0(*(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))) )
        goto LABEL_306;
      v213 = *(_QWORD *)(v212 - 24);
      if ( *(_BYTE *)(v213 + 16) )
        goto LABEL_1104;
      if ( *(_DWORD *)(v213 + 36) == 202 && v10 == sub_15F3430(v212) )
        return sub_170BC50(v12, v10);
LABEL_306:
      v214 = sub_157EBA0(*(_QWORD *)(v10 + 40));
      v215 = *(_QWORD *)(v10 + 32);
      v216 = v214;
      while ( 2 )
      {
        if ( v215 )
          v217 = v215 - 24;
        else
          v217 = 0;
        v218 = *(_BYTE *)(v217 + 16);
        if ( v216 == v217 )
        {
          if ( v218 != 25 && v218 != 30 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
        }
        else
        {
          if ( v218 == 53 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          if ( v218 != 78 )
          {
LABEL_317:
            v215 = *(_QWORD *)(v215 + 8);
            continue;
          }
          v219 = *(_QWORD *)(v217 - 24);
          if ( *(_BYTE *)(v219 + 16) || (*(_BYTE *)(v219 + 33) & 0x20) == 0 )
            return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
          if ( *(_DWORD *)(v219 + 36) != 201 )
          {
            if ( (unsigned __int8)sub_15F3040(v217) || sub_15F3330(v217) )
              return sub_1742F80(
                       v12,
                       v785,
                       a3,
                       *(double *)v13.m128i_i64,
                       *(double *)v14.m128i_i64,
                       a6,
                       v28,
                       v29,
                       a9,
                       a10);
            goto LABEL_317;
          }
        }
        break;
      }
      break;
    case 0xE4u:
    case 0xE5u:
      goto LABEL_108;
    default:
      return sub_1742F80(v12, v785, a3, *(double *)v13.m128i_i64, *(double *)v14.m128i_i64, a6, v28, v29, a9, a10);
  }
  return sub_170BC50(v12, v10);
}
