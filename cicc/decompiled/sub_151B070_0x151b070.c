// Function: sub_151B070
// Address: 0x151b070
//
__int64 *__fastcall sub_151B070(
        __m128i *a1,
        __int64 *a2,
        __int64 **a3,
        int a4,
        __int64 a5,
        unsigned int *a6,
        __int64 a7,
        unsigned __int64 a8)
{
  __int64 *v8; // r15
  unsigned int *v9; // rbx
  __int64 **v10; // r14
  __int64 v11; // r13
  __int64 *v12; // rax
  unsigned int v13; // r12d
  int v14; // r8d
  __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // ecx
  int v18; // eax
  __int64 v19; // rdx
  int v20; // r10d
  int v21; // eax
  __int64 v22; // rax
  int v23; // esi
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v28; // rdx
  __int64 *v29; // r14
  __int64 *v30; // r12
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned int v34; // r12d
  __int64 v35; // rdi
  __int64 v36; // rsi
  unsigned int v37; // edx
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rbx
  __int64 *v40; // r12
  int v41; // eax
  _BYTE *v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // r9d
  unsigned int v46; // r8d
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // rcx
  unsigned int v49; // r11d
  unsigned __int64 v50; // rax
  _QWORD *v51; // rbx
  unsigned int v52; // edi
  unsigned __int64 v53; // rsi
  unsigned __int64 v54; // rsi
  const char *v55; // rax
  _QWORD *v56; // rdi
  __int64 *v57; // rcx
  unsigned int v58; // r15d
  int v59; // r12d
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r10
  __int64 v63; // rax
  bool v64; // zf
  __int64 v65; // rax
  char v66; // al
  __int64 v67; // rax
  unsigned __int64 v68; // rax
  const char *v69; // rax
  __int64 *v70; // r12
  __int64 v71; // rdx
  __int64 *v72; // rcx
  bool v73; // r14
  _BYTE *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // r12
  int v81; // eax
  int v82; // r8d
  int v83; // ecx
  __int64 v84; // rax
  int v85; // eax
  unsigned int v86; // r14d
  __int64 v87; // rsi
  __int64 *v88; // rax
  int v89; // r8d
  __int64 v90; // rdx
  __int64 v91; // rdx
  int v92; // ecx
  int v93; // eax
  __int64 *v94; // rax
  int v95; // r8d
  __int64 v96; // rdx
  __int64 v97; // rdx
  int v98; // ecx
  int v99; // eax
  __int64 *v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rdx
  int v103; // r8d
  __int64 v104; // rdx
  int v105; // ecx
  int v106; // eax
  __int64 v107; // rdx
  int v108; // r10d
  int v109; // eax
  __int64 v110; // rax
  int v111; // esi
  int v112; // eax
  __int64 v113; // rax
  __int64 v114; // rcx
  __int64 *v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rax
  __int64 v118; // rax
  _BYTE *v119; // rsi
  int v120; // eax
  int v121; // edx
  int v122; // r10d
  __int64 *v123; // rax
  __int64 v124; // rcx
  int v125; // eax
  __int64 v126; // rax
  _BYTE *v127; // rsi
  __int64 v128; // rax
  __int64 v129; // rax
  int v130; // r9d
  __int64 *v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rdx
  int v134; // r8d
  int v135; // eax
  __int64 v136; // rdx
  __int64 v137; // rcx
  int v138; // r10d
  int v139; // eax
  __int64 v140; // rax
  int v141; // esi
  int v142; // eax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 *v145; // rdx
  __int64 v146; // rcx
  unsigned __int64 v147; // r9
  __int64 v148; // rdi
  __int64 v149; // rax
  __int64 *v150; // rdx
  char v151; // cl
  __int64 v152; // rsi
  int v153; // edi
  __int64 v154; // rax
  __int64 v155; // r12
  __int64 v156; // rsi
  __int64 v157; // rdi
  __int64 v158; // rax
  __int64 v159; // rax
  _BYTE *v160; // rsi
  int v161; // eax
  int v162; // ecx
  int v163; // r9d
  unsigned __int64 *v164; // rax
  __int64 v165; // rdx
  __int64 v166; // r8
  int v167; // eax
  int v168; // r12d
  __int64 v169; // rdx
  int v170; // eax
  int v171; // esi
  __int64 v172; // rax
  int v173; // eax
  __int64 v174; // rax
  __int64 v175; // rsi
  __int64 *v176; // rax
  char v177; // cl
  unsigned __int64 v178; // rdx
  __int64 *v179; // rax
  __int64 v180; // rdx
  __int64 v181; // r12
  char v182; // al
  __int64 v183; // rdx
  __int64 v184; // rcx
  __int64 *v185; // rax
  __int64 v186; // rdx
  __int64 v187; // rdx
  __int64 v188; // rcx
  int v189; // esi
  int v190; // edx
  __int64 v191; // rax
  int v192; // eax
  __int64 v193; // rax
  __int64 v194; // r14
  __int64 v195; // rsi
  __int64 v196; // rdx
  __int64 v197; // rax
  __int64 *v198; // rax
  __int64 v199; // rdx
  __int64 v200; // rdx
  __int64 v201; // rdi
  __int64 v202; // rax
  _BYTE *v203; // rsi
  __int64 v204; // rcx
  __int64 *v205; // rax
  __int64 v206; // rax
  __int64 v207; // r12
  int v208; // edx
  __int64 v209; // r10
  __int64 v210; // rax
  __int64 v211; // rax
  __int64 v212; // r14
  char v213; // di
  __int64 v214; // r9
  int v215; // r11d
  int v216; // r11d
  int v217; // esi
  __int64 *v218; // rcx
  unsigned int i; // edx
  __int64 *v220; // rax
  __int64 v221; // r8
  __int64 v222; // rax
  __int64 *v223; // rdx
  __int8 v224; // r12
  __int64 v225; // rax
  _BYTE *v226; // rsi
  __int64 v227; // rax
  __int64 v228; // rax
  _BYTE *v229; // rsi
  __int64 v230; // rax
  __int64 v231; // rax
  _BYTE *v232; // rsi
  __int64 v233; // rax
  int v234; // eax
  int v235; // ecx
  int v236; // r9d
  __int64 *v237; // rax
  __int64 v238; // rdx
  __int64 v239; // r8
  int v240; // eax
  __int64 v241; // rsi
  int v242; // edx
  int v243; // eax
  __int64 v244; // rsi
  __int64 v245; // rdi
  __int64 v246; // rax
  __int64 v247; // rax
  unsigned int v248; // edi
  __int64 *v249; // rax
  char v250; // cl
  __int64 v251; // rsi
  __int64 v252; // rsi
  __int64 v253; // rax
  __int64 v254; // rsi
  int v255; // r10d
  int v256; // eax
  __int64 v257; // rax
  int v258; // esi
  int v259; // eax
  __int64 v260; // rdi
  __int64 v261; // rax
  __int64 *v262; // rax
  int v263; // edx
  __int64 v264; // rcx
  __int64 v265; // r11
  __int64 v266; // r8
  __int64 v267; // r10
  int v268; // eax
  __int64 v269; // rax
  __int64 *v270; // rax
  __int64 v271; // r10
  __int64 v272; // rcx
  bool v273; // r8
  __int64 v274; // rax
  unsigned __int64 v275; // rdx
  __int64 v276; // rdi
  unsigned __int64 v277; // rsi
  __int64 *v278; // rdx
  unsigned __int64 v279; // rax
  unsigned int v280; // r8d
  __int64 *v281; // rax
  __int64 v282; // rdi
  int v283; // edx
  __int64 v284; // rax
  int v285; // eax
  __int64 **v286; // rbx
  unsigned int v287; // r14d
  unsigned int v288; // r12d
  __int64 v289; // r9
  __int64 v290; // rsi
  __int64 v291; // rax
  __int32 v292; // eax
  unsigned int v293; // r14d
  __int64 v294; // rax
  int v295; // eax
  __int64 v296; // rax
  __int64 *v297; // rax
  __int64 v298; // rdx
  __int64 v299; // rax
  _BYTE *v300; // rsi
  int v301; // eax
  int v302; // edx
  int v303; // r10d
  __int64 *v304; // rax
  __int64 v305; // rcx
  int v306; // eax
  __int64 v307; // rax
  __int64 *v308; // rax
  int v309; // r11d
  __int64 v310; // rcx
  __int64 v311; // rdx
  int v312; // r10d
  int v313; // eax
  __int64 v314; // rax
  int v315; // esi
  int v316; // eax
  __int64 v317; // rax
  __int64 v318; // rax
  __int64 v319; // rdx
  __int64 v320; // rsi
  __int64 v321; // rax
  __int64 v322; // rax
  int v323; // ecx
  _QWORD *v324; // rax
  __int64 v325; // rdx
  __int64 v326; // rdx
  char v327; // al
  __int64 *v328; // r12
  __int64 v329; // rcx
  __int64 *v330; // rax
  __int64 v331; // rdi
  __int64 v332; // rdx
  __int64 v333; // rdx
  int v334; // r12d
  __int64 v335; // rdi
  __int64 v336; // rax
  _BYTE *v337; // rsi
  int v338; // r9d
  __int64 *v339; // rax
  unsigned __int64 v340; // rdx
  _BYTE *v341; // rsi
  __int64 v342; // rdi
  __int64 v343; // rax
  __int64 v344; // rax
  __int64 v345; // rax
  int v346; // edx
  int v347; // r9d
  __int64 v348; // rcx
  __int64 v349; // rsi
  __int64 v350; // r14
  __int64 v351; // rax
  __int64 v352; // rdi
  __int64 v353; // rax
  __int64 *v354; // rax
  int v355; // r8d
  __int64 v356; // rdi
  __int64 v357; // rdx
  __int64 v358; // rdx
  int v359; // r10d
  int v360; // eax
  __int64 v361; // rcx
  int v362; // edx
  int v363; // eax
  __int64 v364; // rax
  __int64 *v365; // rax
  __int64 v366; // rdx
  __int64 v367; // rdx
  __int64 v368; // r10
  __int64 v369; // rdx
  int v370; // r9d
  int v371; // eax
  __int64 v372; // rdx
  int v373; // r8d
  int v374; // eax
  __int64 v375; // rax
  __int64 *v376; // rax
  int v377; // r10d
  __int64 v378; // rdx
  __int64 v379; // r8
  __int64 v380; // rcx
  int v381; // eax
  __int64 v382; // rax
  int v383; // esi
  int v384; // eax
  unsigned __int64 v385; // r8
  __int64 *v386; // rax
  bool v387; // si
  unsigned __int64 v388; // rdx
  bool v389; // r12
  __int64 v390; // rcx
  __int64 v391; // rax
  __int64 v392; // rdi
  int v393; // ecx
  __int64 v394; // rsi
  __int64 v395; // rax
  __int64 v396; // rsi
  __int64 v397; // rax
  __int64 v398; // rcx
  __int64 v399; // rax
  bool v400; // cc
  __int64 v401; // rdx
  _BYTE *v402; // rsi
  __int64 v403; // rdi
  __int64 v404; // rax
  __int64 *v405; // rax
  __int64 v406; // rdx
  __int64 v407; // rdx
  __int64 v408; // rdx
  int v409; // ecx
  __int64 v410; // rsi
  int v411; // edx
  int v412; // eax
  __int64 v413; // rax
  _BYTE *v414; // rsi
  __int64 v415; // rax
  int v416; // eax
  __int64 v417; // rax
  __int64 v418; // r14
  __int64 v419; // rdi
  unsigned __int64 v420; // rdx
  __int64 *v421; // rax
  __int64 v422; // rdx
  __int64 v423; // r12
  __int64 v424; // rdx
  __int64 v425; // rdx
  __int64 v426; // rdx
  __int64 v427; // rdx
  __int64 v428; // rdx
  int v429; // r9d
  __int64 v430; // rdx
  _BOOL4 v431; // r8d
  int v432; // ecx
  int v433; // eax
  __int64 v434; // rsi
  int v435; // edx
  int v436; // eax
  __int64 v437; // r12
  __int64 v438; // rax
  __int64 v439; // rax
  __m128i *v440; // rsi
  __int64 v441; // rax
  __int64 *v442; // rax
  __int64 v443; // r10
  __int64 v444; // rdx
  __int64 v445; // rsi
  __int64 v446; // rcx
  __int64 v447; // rax
  _BOOL4 v448; // r14d
  __int64 v449; // rax
  __int64 *v450; // rax
  __int64 v451; // r10
  __int64 v452; // rdx
  __int64 v453; // rcx
  __int64 v454; // rax
  __int64 v455; // rax
  __int64 v456; // rsi
  __int64 v457; // rax
  _BYTE *v458; // rsi
  __int64 v459; // rax
  __int64 v460; // rax
  __int64 v461; // rsi
  __int64 v462; // rdx
  __int64 v463; // rax
  __int64 v464; // rax
  __int64 v465; // rax
  __int64 *v466; // rdx
  unsigned int v467; // r10d
  unsigned __int64 v468; // r14
  __int64 v469; // rdi
  __int64 v470; // rdx
  char v471; // al
  _BYTE *v472; // rax
  __int64 v473; // r12
  __int64 v474; // rdx
  int v475; // ecx
  int v476; // eax
  __int64 v477; // rdx
  int v478; // r10d
  int v479; // eax
  __int64 v480; // rax
  int v481; // esi
  int v482; // eax
  __int64 v483; // rax
  __int64 v484; // rsi
  __int64 v485; // rax
  __int64 v486; // rdx
  __int64 v487; // rcx
  unsigned __int64 v488; // rax
  __int64 v489; // rax
  __int64 v490; // rax
  _BYTE *v491; // rsi
  int v492; // eax
  int v493; // edx
  int v494; // r10d
  __int64 *v495; // rax
  __int64 v496; // rcx
  int v497; // eax
  __int64 *v498; // rdx
  char v499; // r8
  __int64 v500; // rax
  int v501; // r10d
  int v502; // eax
  __int64 v503; // rax
  int v504; // esi
  int v505; // eax
  __int64 v506; // rdi
  __int64 v507; // rbx
  __int64 v508; // r12
  __int64 v509; // rax
  unsigned __int64 v510; // rbx
  _BYTE *v511; // rax
  unsigned __int64 v512; // r12
  __int64 v513; // rax
  __int64 v514; // rsi
  unsigned __int64 v515; // r9
  __int64 v516; // r10
  __int64 v517; // rdi
  __int64 v518; // rax
  __int64 v519; // rsi
  char v520; // cl
  __int64 v521; // rax
  unsigned __int64 v522; // rdx
  __int64 v523; // rdi
  unsigned __int64 v524; // rsi
  int v525; // eax
  __int64 v526; // rdx
  int v527; // ecx
  int v528; // eax
  __int64 v529; // rdx
  int v530; // ecx
  int v531; // eax
  __int64 v532; // rdx
  int v533; // r8d
  __int64 v534; // rdx
  int v535; // ecx
  int v536; // eax
  __int64 v537; // rdx
  int v538; // r10d
  int v539; // eax
  __int64 v540; // rax
  int v541; // esi
  int v542; // eax
  __int64 v543; // rax
  int v544; // r9d
  __int64 *v545; // rax
  __int64 v546; // rdx
  __int64 v547; // rdx
  int v548; // r8d
  int v549; // eax
  __int64 v550; // rdx
  __int64 v551; // rcx
  int v552; // r10d
  int v553; // eax
  __int64 v554; // rax
  int v555; // esi
  int v556; // eax
  __int64 v557; // rdx
  __int64 v558; // r10
  __int64 v559; // rdx
  int v560; // r9d
  int v561; // eax
  __int64 v562; // rdx
  int v563; // r8d
  int v564; // eax
  __int64 v565; // rdx
  int v566; // r10d
  int v567; // eax
  __int64 v568; // rcx
  int v569; // edx
  int v570; // eax
  __int64 v571; // rdx
  int v572; // r10d
  int v573; // eax
  __int64 v574; // rax
  int v575; // esi
  int v576; // eax
  __int64 v577; // rax
  _BYTE *v578; // rsi
  int v579; // eax
  int v580; // edx
  int v581; // r10d
  __int64 *v582; // rax
  __int64 v583; // rcx
  int v584; // eax
  int v585; // eax
  __int64 v586; // rax
  int v587; // esi
  int v588; // eax
  __int64 v589; // rax
  __int64 v590; // rsi
  __int64 v591; // rdx
  __int64 v592; // rax
  __int64 v593; // rax
  __int64 v594; // rax
  __int64 v595; // rax
  __int64 v596; // rsi
  __int64 v597; // rax
  __int64 v598; // rax
  __int64 v599; // rax
  __int64 v600; // rax
  __int64 v601; // rax
  __int64 v602; // rax
  _BYTE *v603; // rsi
  __int64 v604; // rax
  __int64 v605; // rax
  _BYTE *v606; // rsi
  __int64 v607; // rax
  int v608; // eax
  int v609; // ecx
  int v610; // r9d
  __int64 *v611; // rax
  __int64 v612; // rdx
  __int64 v613; // r8
  int v614; // eax
  __int64 v615; // rsi
  int v616; // edx
  int v617; // eax
  __int64 v618; // rsi
  __int64 v619; // rdi
  __int64 v620; // rax
  __int64 v621; // rsi
  __int64 v622; // rax
  __int64 v623; // rsi
  __int64 v624; // rax
  __int64 v625; // rcx
  __int64 v626; // rax
  __int64 v627; // rdx
  _BYTE *v628; // rsi
  __int64 v629; // rdi
  __int64 v630; // rax
  __int64 *v631; // rax
  __int64 v632; // rdx
  __int64 v633; // rdx
  __int64 v634; // rdx
  int v635; // ecx
  __int64 v636; // rsi
  int v637; // edx
  int v638; // eax
  __int64 v639; // rax
  _BYTE *v640; // rsi
  __int64 v641; // rax
  int v642; // eax
  int v643; // eax
  int v644; // ecx
  int v645; // r9d
  __int64 *v646; // rax
  __int64 v647; // rdx
  __int64 v648; // r8
  int v649; // eax
  int v650; // r12d
  __int64 v651; // rdx
  int v652; // eax
  int v653; // esi
  __int64 v654; // rax
  int v655; // eax
  __int64 v656; // rdx
  __int64 v657; // rdx
  __int64 v658; // rcx
  __int64 *v659; // rax
  __int64 v660; // rdx
  __int64 v661; // rdx
  __int64 v662; // rcx
  int v663; // esi
  int v664; // edx
  __int64 v665; // rax
  int v666; // eax
  __int64 v667; // rax
  __int64 v668; // rdx
  __int64 v669; // rax
  __int64 v670; // rdx
  __int64 *v671; // rax
  __int64 v672; // rdx
  __int64 v673; // rax
  _BYTE *v674; // rsi
  __int64 v675; // rax
  __int64 v676; // rax
  __int64 v677; // rdi
  __int64 v678; // rax
  __int64 v679; // rax
  __int64 v680; // rcx
  __int64 *v681; // rax
  __int64 v682; // rdx
  __int64 v683; // rdx
  __int64 v684; // rcx
  int v685; // esi
  int v686; // edx
  __int64 v687; // rax
  int v688; // eax
  __int64 v689; // rsi
  __int64 v690; // rax
  __int64 v691; // rsi
  __int64 v692; // rax
  __int64 v693; // rcx
  __int64 *v694; // rax
  __int64 v695; // rdx
  __int64 v696; // rdx
  __int64 v697; // rcx
  int v698; // esi
  int v699; // edx
  __int64 v700; // rax
  int v701; // eax
  __int64 v702; // rdx
  _QWORD *v703; // rax
  __int64 v704; // rdi
  char v705; // al
  __int64 v706; // rax
  char v707; // di
  __int64 v708; // r9
  int v709; // eax
  int v710; // esi
  int v711; // r11d
  __int64 *v712; // rcx
  unsigned int m; // edx
  __int64 v714; // r8
  unsigned int v715; // edx
  unsigned int v716; // esi
  int v717; // r11d
  __int64 v718; // rax
  __int64 v719; // rcx
  __int64 v720; // r8
  unsigned int v721; // eax
  __int64 v722; // rdx
  __int64 *v723; // r14
  unsigned int v724; // esi
  unsigned int v725; // edx
  int v726; // r11d
  __int64 v727; // rdi
  int v728; // eax
  int v729; // esi
  int v730; // r9d
  __int64 *v731; // rcx
  unsigned int j; // edx
  __int64 v733; // r8
  __int64 v734; // rdi
  int v735; // eax
  int v736; // esi
  int v737; // r9d
  unsigned int k; // edx
  __int64 v739; // r8
  unsigned int v740; // edx
  unsigned int v741; // edx
  __int64 v742; // rdi
  int v743; // eax
  int v744; // esi
  int v745; // r9d
  __int64 *v746; // rcx
  unsigned int n; // edx
  __int64 v748; // r8
  __int64 v749; // rdi
  int v750; // eax
  int v751; // esi
  int v752; // r9d
  unsigned int ii; // edx
  __int64 v754; // r8
  unsigned int v755; // edx
  unsigned int v756; // edx
  unsigned int v757; // edx
  unsigned int v758; // edx
  int v759; // [rsp+0h] [rbp-180h]
  int v760; // [rsp+0h] [rbp-180h]
  int v761; // [rsp+0h] [rbp-180h]
  int v762; // [rsp+0h] [rbp-180h]
  int v763; // [rsp+8h] [rbp-178h]
  int v764; // [rsp+8h] [rbp-178h]
  int v765; // [rsp+8h] [rbp-178h]
  int v766; // [rsp+8h] [rbp-178h]
  int v767; // [rsp+8h] [rbp-178h]
  int v768; // [rsp+8h] [rbp-178h]
  __int64 v769; // [rsp+10h] [rbp-170h]
  __int64 v770; // [rsp+10h] [rbp-170h]
  __int64 v771; // [rsp+18h] [rbp-168h]
  __int64 v772; // [rsp+18h] [rbp-168h]
  __int64 v773; // [rsp+28h] [rbp-158h]
  __int64 v774; // [rsp+28h] [rbp-158h]
  __int64 v775; // [rsp+28h] [rbp-158h]
  __int64 v776; // [rsp+28h] [rbp-158h]
  __int64 v777; // [rsp+30h] [rbp-150h]
  __int64 v778; // [rsp+30h] [rbp-150h]
  int v779; // [rsp+30h] [rbp-150h]
  __int64 v780; // [rsp+30h] [rbp-150h]
  __int64 v781; // [rsp+38h] [rbp-148h]
  __int64 v782; // [rsp+38h] [rbp-148h]
  int v783; // [rsp+38h] [rbp-148h]
  int v784; // [rsp+38h] [rbp-148h]
  __int64 v785; // [rsp+38h] [rbp-148h]
  __int64 v786; // [rsp+40h] [rbp-140h]
  __int64 v787; // [rsp+40h] [rbp-140h]
  _BOOL4 v788; // [rsp+40h] [rbp-140h]
  _BOOL4 v789; // [rsp+40h] [rbp-140h]
  __int64 v790; // [rsp+40h] [rbp-140h]
  __int64 v791; // [rsp+48h] [rbp-138h]
  int v792; // [rsp+48h] [rbp-138h]
  _BOOL4 v793; // [rsp+48h] [rbp-138h]
  _BOOL4 v794; // [rsp+48h] [rbp-138h]
  int v795; // [rsp+48h] [rbp-138h]
  _BOOL4 v796; // [rsp+48h] [rbp-138h]
  int v797; // [rsp+50h] [rbp-130h]
  int v798; // [rsp+50h] [rbp-130h]
  __int64 v799; // [rsp+50h] [rbp-130h]
  int v800; // [rsp+50h] [rbp-130h]
  _BOOL4 v801; // [rsp+50h] [rbp-130h]
  _BOOL4 v802; // [rsp+50h] [rbp-130h]
  __int64 v803; // [rsp+50h] [rbp-130h]
  int v804; // [rsp+50h] [rbp-130h]
  _BOOL4 v805; // [rsp+50h] [rbp-130h]
  int v806; // [rsp+50h] [rbp-130h]
  __int64 v807; // [rsp+58h] [rbp-128h]
  __int64 v808; // [rsp+58h] [rbp-128h]
  __int64 v809; // [rsp+58h] [rbp-128h]
  int v810; // [rsp+58h] [rbp-128h]
  int v811; // [rsp+58h] [rbp-128h]
  _BOOL4 v812; // [rsp+58h] [rbp-128h]
  __int64 v813; // [rsp+58h] [rbp-128h]
  int v814; // [rsp+58h] [rbp-128h]
  int v815; // [rsp+58h] [rbp-128h]
  _BOOL4 v816; // [rsp+58h] [rbp-128h]
  int v817; // [rsp+58h] [rbp-128h]
  int v818; // [rsp+58h] [rbp-128h]
  __int64 v819; // [rsp+58h] [rbp-128h]
  __int32 v820; // [rsp+60h] [rbp-120h]
  __int64 v821; // [rsp+60h] [rbp-120h]
  int v822; // [rsp+60h] [rbp-120h]
  int v823; // [rsp+60h] [rbp-120h]
  __int64 v824; // [rsp+60h] [rbp-120h]
  __int64 v825; // [rsp+60h] [rbp-120h]
  __int64 v826; // [rsp+60h] [rbp-120h]
  __int64 v827; // [rsp+60h] [rbp-120h]
  __int64 v828; // [rsp+60h] [rbp-120h]
  __int64 v829; // [rsp+60h] [rbp-120h]
  int v830; // [rsp+60h] [rbp-120h]
  __int64 v831; // [rsp+60h] [rbp-120h]
  __int64 v832; // [rsp+60h] [rbp-120h]
  int v833; // [rsp+60h] [rbp-120h]
  __int64 v834; // [rsp+68h] [rbp-118h]
  __int64 v835; // [rsp+68h] [rbp-118h]
  __int64 v836; // [rsp+68h] [rbp-118h]
  unsigned __int64 v837; // [rsp+68h] [rbp-118h]
  __int64 v838; // [rsp+68h] [rbp-118h]
  int v839; // [rsp+68h] [rbp-118h]
  int v840; // [rsp+68h] [rbp-118h]
  int v841; // [rsp+68h] [rbp-118h]
  unsigned __int64 v842; // [rsp+68h] [rbp-118h]
  int v843; // [rsp+68h] [rbp-118h]
  __int64 v844; // [rsp+68h] [rbp-118h]
  __int64 v845; // [rsp+68h] [rbp-118h]
  int v846; // [rsp+68h] [rbp-118h]
  unsigned __int64 v847; // [rsp+68h] [rbp-118h]
  int v848; // [rsp+68h] [rbp-118h]
  int v849; // [rsp+68h] [rbp-118h]
  int v850; // [rsp+68h] [rbp-118h]
  __int64 v851; // [rsp+68h] [rbp-118h]
  __int64 v852; // [rsp+70h] [rbp-110h]
  int v853; // [rsp+70h] [rbp-110h]
  __int64 v854; // [rsp+70h] [rbp-110h]
  int v855; // [rsp+70h] [rbp-110h]
  _BOOL4 v856; // [rsp+70h] [rbp-110h]
  __int64 v857; // [rsp+70h] [rbp-110h]
  __int64 v858; // [rsp+70h] [rbp-110h]
  __int64 v859; // [rsp+70h] [rbp-110h]
  int v860; // [rsp+70h] [rbp-110h]
  __int64 v861; // [rsp+70h] [rbp-110h]
  _BOOL4 v862; // [rsp+70h] [rbp-110h]
  int v863; // [rsp+70h] [rbp-110h]
  __int64 v864; // [rsp+70h] [rbp-110h]
  __int64 v865; // [rsp+70h] [rbp-110h]
  int v866; // [rsp+70h] [rbp-110h]
  __int64 v867; // [rsp+70h] [rbp-110h]
  __int64 v868; // [rsp+70h] [rbp-110h]
  _BOOL4 v869; // [rsp+70h] [rbp-110h]
  __int64 v870; // [rsp+78h] [rbp-108h]
  __int64 v871; // [rsp+78h] [rbp-108h]
  int v872; // [rsp+78h] [rbp-108h]
  __int64 v873; // [rsp+78h] [rbp-108h]
  int v874; // [rsp+78h] [rbp-108h]
  int v875; // [rsp+78h] [rbp-108h]
  _BOOL4 v876; // [rsp+78h] [rbp-108h]
  unsigned __int64 v877; // [rsp+78h] [rbp-108h]
  __int64 v878; // [rsp+78h] [rbp-108h]
  int v879; // [rsp+78h] [rbp-108h]
  int v880; // [rsp+78h] [rbp-108h]
  int v881; // [rsp+78h] [rbp-108h]
  __int64 v882; // [rsp+78h] [rbp-108h]
  __int64 v883; // [rsp+78h] [rbp-108h]
  __int64 v884; // [rsp+78h] [rbp-108h]
  __int64 v885; // [rsp+78h] [rbp-108h]
  __int64 v886; // [rsp+78h] [rbp-108h]
  int v887; // [rsp+78h] [rbp-108h]
  int v888; // [rsp+78h] [rbp-108h]
  __int64 v889; // [rsp+78h] [rbp-108h]
  int v890; // [rsp+78h] [rbp-108h]
  __int64 v891; // [rsp+78h] [rbp-108h]
  int v892; // [rsp+78h] [rbp-108h]
  int v893; // [rsp+78h] [rbp-108h]
  _BOOL4 v894; // [rsp+78h] [rbp-108h]
  _BOOL4 v895; // [rsp+78h] [rbp-108h]
  int v896; // [rsp+78h] [rbp-108h]
  int v897; // [rsp+78h] [rbp-108h]
  _BOOL4 v898; // [rsp+78h] [rbp-108h]
  __int32 v899; // [rsp+80h] [rbp-100h]
  int v900; // [rsp+80h] [rbp-100h]
  int v901; // [rsp+80h] [rbp-100h]
  int v902; // [rsp+80h] [rbp-100h]
  int v903; // [rsp+80h] [rbp-100h]
  int v904; // [rsp+80h] [rbp-100h]
  int v905; // [rsp+80h] [rbp-100h]
  __int64 v906; // [rsp+80h] [rbp-100h]
  int v907; // [rsp+80h] [rbp-100h]
  __int64 v908; // [rsp+80h] [rbp-100h]
  unsigned int *v909; // [rsp+80h] [rbp-100h]
  __int64 v910; // [rsp+80h] [rbp-100h]
  int v911; // [rsp+80h] [rbp-100h]
  int v912; // [rsp+80h] [rbp-100h]
  unsigned __int64 v913; // [rsp+80h] [rbp-100h]
  unsigned __int64 v914; // [rsp+80h] [rbp-100h]
  __int64 v915; // [rsp+80h] [rbp-100h]
  __int64 v916; // [rsp+80h] [rbp-100h]
  int v917; // [rsp+80h] [rbp-100h]
  int v918; // [rsp+80h] [rbp-100h]
  int v919; // [rsp+80h] [rbp-100h]
  int v920; // [rsp+80h] [rbp-100h]
  int v921; // [rsp+80h] [rbp-100h]
  int v922; // [rsp+80h] [rbp-100h]
  int v923; // [rsp+80h] [rbp-100h]
  __int64 v924; // [rsp+80h] [rbp-100h]
  unsigned __int64 v925; // [rsp+80h] [rbp-100h]
  unsigned __int64 v926; // [rsp+80h] [rbp-100h]
  __int64 v927; // [rsp+80h] [rbp-100h]
  _BOOL4 v928; // [rsp+80h] [rbp-100h]
  _BOOL4 v929; // [rsp+80h] [rbp-100h]
  unsigned __int64 v930; // [rsp+80h] [rbp-100h]
  unsigned __int64 v931; // [rsp+80h] [rbp-100h]
  int v932; // [rsp+88h] [rbp-F8h]
  int v933; // [rsp+88h] [rbp-F8h]
  __int64 *v934; // [rsp+88h] [rbp-F8h]
  int v935; // [rsp+88h] [rbp-F8h]
  int v936; // [rsp+88h] [rbp-F8h]
  int v937; // [rsp+88h] [rbp-F8h]
  int v938; // [rsp+88h] [rbp-F8h]
  int v939; // [rsp+88h] [rbp-F8h]
  int v940; // [rsp+88h] [rbp-F8h]
  __int64 v941; // [rsp+88h] [rbp-F8h]
  int v942; // [rsp+88h] [rbp-F8h]
  int v943; // [rsp+88h] [rbp-F8h]
  int v944; // [rsp+88h] [rbp-F8h]
  int v945; // [rsp+88h] [rbp-F8h]
  __int64 v946; // [rsp+88h] [rbp-F8h]
  __int64 v947; // [rsp+88h] [rbp-F8h]
  int v948; // [rsp+88h] [rbp-F8h]
  int v949; // [rsp+88h] [rbp-F8h]
  int v950; // [rsp+88h] [rbp-F8h]
  int v951; // [rsp+88h] [rbp-F8h]
  int v952; // [rsp+88h] [rbp-F8h]
  __int64 v953; // [rsp+88h] [rbp-F8h]
  __int64 v954; // [rsp+88h] [rbp-F8h]
  int v955; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v956; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v957; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v958; // [rsp+88h] [rbp-F8h]
  int v959; // [rsp+88h] [rbp-F8h]
  __int64 v960; // [rsp+88h] [rbp-F8h]
  __int64 v961; // [rsp+88h] [rbp-F8h]
  int v962; // [rsp+88h] [rbp-F8h]
  int v963; // [rsp+88h] [rbp-F8h]
  int v964; // [rsp+88h] [rbp-F8h]
  int v965; // [rsp+88h] [rbp-F8h]
  int v966; // [rsp+88h] [rbp-F8h]
  int v967; // [rsp+88h] [rbp-F8h]
  int v968; // [rsp+88h] [rbp-F8h]
  int v969; // [rsp+88h] [rbp-F8h]
  int v970; // [rsp+88h] [rbp-F8h]
  int v971; // [rsp+88h] [rbp-F8h]
  int v972; // [rsp+88h] [rbp-F8h]
  __int64 v973; // [rsp+88h] [rbp-F8h]
  int v974; // [rsp+88h] [rbp-F8h]
  int v975; // [rsp+88h] [rbp-F8h]
  __int64 v976; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v977; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v978; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v979; // [rsp+88h] [rbp-F8h]
  int v980; // [rsp+88h] [rbp-F8h]
  int v981; // [rsp+88h] [rbp-F8h]
  int v982; // [rsp+88h] [rbp-F8h]
  int v983; // [rsp+88h] [rbp-F8h]
  __int64 v984; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v985; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v986; // [rsp+88h] [rbp-F8h]
  int v987; // [rsp+88h] [rbp-F8h]
  __int64 v988; // [rsp+88h] [rbp-F8h]
  __int64 v989; // [rsp+88h] [rbp-F8h]
  int v990; // [rsp+90h] [rbp-F0h]
  int v991; // [rsp+90h] [rbp-F0h]
  int v992; // [rsp+90h] [rbp-F0h]
  int v993; // [rsp+90h] [rbp-F0h]
  int v994; // [rsp+90h] [rbp-F0h]
  int v995; // [rsp+90h] [rbp-F0h]
  int v996; // [rsp+90h] [rbp-F0h]
  __int64 v997; // [rsp+90h] [rbp-F0h]
  int v998; // [rsp+90h] [rbp-F0h]
  __int64 v999; // [rsp+90h] [rbp-F0h]
  int v1000; // [rsp+90h] [rbp-F0h]
  __int64 v1001; // [rsp+90h] [rbp-F0h]
  int v1002; // [rsp+90h] [rbp-F0h]
  int v1003; // [rsp+90h] [rbp-F0h]
  char v1004; // [rsp+90h] [rbp-F0h]
  char v1005; // [rsp+90h] [rbp-F0h]
  char v1006; // [rsp+90h] [rbp-F0h]
  int v1007; // [rsp+90h] [rbp-F0h]
  __int64 v1008; // [rsp+90h] [rbp-F0h]
  int v1009; // [rsp+90h] [rbp-F0h]
  int v1010; // [rsp+90h] [rbp-F0h]
  int v1011; // [rsp+90h] [rbp-F0h]
  int v1012; // [rsp+90h] [rbp-F0h]
  __int64 v1013; // [rsp+90h] [rbp-F0h]
  int v1014; // [rsp+90h] [rbp-F0h]
  int v1015; // [rsp+90h] [rbp-F0h]
  int v1016; // [rsp+90h] [rbp-F0h]
  unsigned __int64 v1017; // [rsp+90h] [rbp-F0h]
  __int64 v1018; // [rsp+90h] [rbp-F0h]
  __int64 v1019; // [rsp+90h] [rbp-F0h]
  __int64 v1020; // [rsp+90h] [rbp-F0h]
  unsigned int v1021; // [rsp+90h] [rbp-F0h]
  int v1022; // [rsp+90h] [rbp-F0h]
  int v1023; // [rsp+90h] [rbp-F0h]
  int v1024; // [rsp+90h] [rbp-F0h]
  __int64 v1025; // [rsp+90h] [rbp-F0h]
  int v1026; // [rsp+90h] [rbp-F0h]
  char v1027; // [rsp+90h] [rbp-F0h]
  char v1028; // [rsp+90h] [rbp-F0h]
  __int64 *v1029; // [rsp+90h] [rbp-F0h]
  int v1030; // [rsp+90h] [rbp-F0h]
  int v1031; // [rsp+90h] [rbp-F0h]
  int v1032; // [rsp+90h] [rbp-F0h]
  int v1033; // [rsp+90h] [rbp-F0h]
  __int64 v1034; // [rsp+90h] [rbp-F0h]
  __int64 v1035; // [rsp+90h] [rbp-F0h]
  int v1036; // [rsp+90h] [rbp-F0h]
  int v1037; // [rsp+90h] [rbp-F0h]
  int v1038; // [rsp+90h] [rbp-F0h]
  int v1039; // [rsp+90h] [rbp-F0h]
  int v1040; // [rsp+90h] [rbp-F0h]
  int v1041; // [rsp+90h] [rbp-F0h]
  __int64 v1042; // [rsp+90h] [rbp-F0h]
  __int64 v1043; // [rsp+90h] [rbp-F0h]
  int v1044; // [rsp+90h] [rbp-F0h]
  int v1045; // [rsp+90h] [rbp-F0h]
  unsigned int v1046; // [rsp+90h] [rbp-F0h]
  unsigned int v1047; // [rsp+90h] [rbp-F0h]
  __int64 v1048; // [rsp+98h] [rbp-E8h]
  __int64 v1049; // [rsp+98h] [rbp-E8h]
  __int64 v1050; // [rsp+98h] [rbp-E8h]
  int v1051; // [rsp+98h] [rbp-E8h]
  int v1052; // [rsp+98h] [rbp-E8h]
  int v1053; // [rsp+98h] [rbp-E8h]
  __int64 v1054; // [rsp+98h] [rbp-E8h]
  __int64 v1055; // [rsp+98h] [rbp-E8h]
  unsigned int v1056; // [rsp+98h] [rbp-E8h]
  int v1057; // [rsp+98h] [rbp-E8h]
  __int64 v1058; // [rsp+98h] [rbp-E8h]
  int v1059; // [rsp+98h] [rbp-E8h]
  char v1060; // [rsp+98h] [rbp-E8h]
  char v1061; // [rsp+98h] [rbp-E8h]
  char v1062; // [rsp+98h] [rbp-E8h]
  int v1063; // [rsp+98h] [rbp-E8h]
  bool v1064; // [rsp+98h] [rbp-E8h]
  int v1065; // [rsp+98h] [rbp-E8h]
  __int64 v1066; // [rsp+98h] [rbp-E8h]
  __int64 v1067; // [rsp+98h] [rbp-E8h]
  int v1068; // [rsp+98h] [rbp-E8h]
  __int64 v1069; // [rsp+98h] [rbp-E8h]
  __int64 v1070; // [rsp+98h] [rbp-E8h]
  int v1071; // [rsp+98h] [rbp-E8h]
  int v1072; // [rsp+98h] [rbp-E8h]
  __int64 v1073; // [rsp+98h] [rbp-E8h]
  __int64 v1074; // [rsp+98h] [rbp-E8h]
  __int64 v1075; // [rsp+98h] [rbp-E8h]
  __int64 v1076; // [rsp+98h] [rbp-E8h]
  __int64 v1077; // [rsp+98h] [rbp-E8h]
  __int64 v1078; // [rsp+98h] [rbp-E8h]
  __int64 v1079; // [rsp+98h] [rbp-E8h]
  char v1080; // [rsp+98h] [rbp-E8h]
  char v1081; // [rsp+98h] [rbp-E8h]
  __int64 v1082; // [rsp+98h] [rbp-E8h]
  bool v1083; // [rsp+98h] [rbp-E8h]
  int v1084; // [rsp+98h] [rbp-E8h]
  int v1085; // [rsp+98h] [rbp-E8h]
  int v1086; // [rsp+98h] [rbp-E8h]
  __int64 v1087; // [rsp+98h] [rbp-E8h]
  int v1088; // [rsp+98h] [rbp-E8h]
  int v1089; // [rsp+98h] [rbp-E8h]
  __int64 v1090; // [rsp+98h] [rbp-E8h]
  __int64 v1091; // [rsp+98h] [rbp-E8h]
  __int64 v1092; // [rsp+98h] [rbp-E8h]
  __int64 v1093; // [rsp+98h] [rbp-E8h]
  char v1094; // [rsp+98h] [rbp-E8h]
  __int64 v1095; // [rsp+98h] [rbp-E8h]
  __int64 v1096; // [rsp+98h] [rbp-E8h]
  char v1097; // [rsp+ABh] [rbp-D5h] BYREF
  int v1098; // [rsp+ACh] [rbp-D4h] BYREF
  _QWORD *v1099; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v1100; // [rsp+B8h] [rbp-C8h] BYREF
  unsigned __int64 v1101; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v1102; // [rsp+C8h] [rbp-B8h]
  _BYTE v1103[16]; // [rsp+D0h] [rbp-B0h] BYREF
  _QWORD v1104[4]; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v1105; // [rsp+100h] [rbp-80h] BYREF
  _QWORD v1106[14]; // [rsp+110h] [rbp-70h] BYREF

  v8 = (__int64 *)a1;
  v9 = a6;
  v1104[2] = a6;
  v1097 = 0;
  v1104[0] = a2;
  v1104[1] = &v1097;
  v1104[3] = a5;
  v1099 = v1104;
  v10 = a3;
  v11 = (__int64)a2;
  switch ( a4 )
  {
    case 1:
      v70 = *a3;
      v71 = *((unsigned int *)a3 + 2);
      v1105.m128i_i64[0] = (__int64)v1106;
      v72 = &v70[v71];
      v73 = &v70[v71] != 0 && v70 == 0;
      if ( v73 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v1101 = (v71 * 8) >> 3;
      v74 = v1106;
      if ( (unsigned __int64)v71 > 15 )
      {
        v1029 = &v70[v71];
        v74 = (_BYTE *)sub_22409D0(&v1105, &v1101, 0);
        v72 = v1029;
        v1105.m128i_i64[0] = (__int64)v74;
        v1106[0] = v1101;
      }
      if ( v70 != v72 )
      {
        do
        {
          v75 = *v70++;
          *v74++ = v75;
        }
        while ( v72 != v70 );
        v74 = (_BYTE *)v1105.m128i_i64[0];
      }
      v1105.m128i_i64[1] = v1101;
      v74[v1101] = 0;
      v76 = v1105.m128i_i64[1];
      v77 = v1105.m128i_i64[0];
      if ( v1105.m128i_i64[1] > 0xFuLL )
        v73 = (*(_QWORD *)v1105.m128i_i64[0] ^ 0x6365762E6D766C6CLL
             | *(_QWORD *)(v1105.m128i_i64[0] + 8) ^ 0x2E72657A69726F74LL) == 0;
      *(_BYTE *)(v11 + 1009) |= v73;
      v78 = sub_161FF10(*(_QWORD *)(v11 + 240), v77, v76);
      sub_15194E0(v11, v78, *v9);
      ++*v9;
      if ( (_QWORD *)v1105.m128i_i64[0] != v1106 )
        j_j___libc_free_0(v1105.m128i_i64[0], v1106[0] + 1LL);
      goto LABEL_14;
    case 2:
      if ( *((_DWORD *)a3 + 2) != 2 )
        goto LABEL_64;
      v64 = a2[34] == 0;
      LODWORD(v1101) = **a3;
      if ( v64 )
        goto LABEL_767;
      if ( (*(_BYTE *)(((__int64 (__fastcall *)(__int64 *, unsigned __int64 *))a2[35])(a2 + 32, &v1101) + 8) & 0xF7) == 0 )
        goto LABEL_64;
      v13 = *v9;
      goto LABEL_464;
    case 3:
      goto LABEL_17;
    case 4:
      v39 = *((unsigned int *)a3 + 2);
      v40 = *a3;
      v1101 = (unsigned __int64)v1103;
      v1102 = 0x800000000LL;
      v41 = v39;
      if ( v39 > 8 )
      {
        sub_16CD150(&v1101, v1103, v39, 1);
        v42 = (_BYTE *)(v1101 + (unsigned int)v1102);
      }
      else
      {
        if ( !(8 * v39) )
          goto LABEL_36;
        v42 = v1103;
      }
      v43 = 0;
      do
      {
        v42[v43] = v40[v43];
        ++v43;
      }
      while ( v39 != v43 );
      v41 = v39 + v1102;
LABEL_36:
      *((_DWORD *)v10 + 2) = 0;
      v44 = a2[29];
      LODWORD(v1102) = v41;
      v45 = *(_DWORD *)(v44 + 36);
      v46 = *(_DWORD *)(v44 + 32);
      if ( v45 <= v46 )
      {
        v488 = *(_QWORD *)(v44 + 24);
        *(_DWORD *)(v44 + 32) = v46 - v45;
        v54 = v488 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v45));
        *(_QWORD *)(v44 + 24) = v488 >> v45;
        goto LABEL_44;
      }
      v1049 = 0;
      if ( v46 )
        v1049 = *(_QWORD *)(v44 + 24);
      v47 = *(_QWORD *)(v44 + 16);
      v48 = *(_QWORD *)(v44 + 8);
      v49 = v45 - v46;
      if ( v47 >= v48 )
        goto LABEL_771;
      v50 = v47 + 8;
      v51 = (_QWORD *)(v47 + *(_QWORD *)v44);
      if ( v48 < v47 + 8 )
      {
        *(_QWORD *)(v44 + 24) = 0;
        v966 = v48 - v47;
        if ( (_DWORD)v48 == (_DWORD)v47 )
        {
          *(_DWORD *)(v44 + 32) = 0;
LABEL_771:
          sub_16BD130("Unexpected end of file", 1);
        }
        v516 = (unsigned int)(v48 - v47);
        v517 = 0;
        v518 = 0;
        do
        {
          v519 = *((unsigned __int8 *)v51 + v518);
          v520 = 8 * v518++;
          v517 |= v519 << v520;
          *(_QWORD *)(v44 + 24) = v517;
        }
        while ( v518 != v516 );
        v50 = v47 + v518;
        v52 = 8 * v966;
      }
      else
      {
        v52 = 64;
        *(_QWORD *)(v44 + 24) = *v51;
      }
      *(_QWORD *)(v44 + 16) = v50;
      *(_DWORD *)(v44 + 32) = v52;
      if ( v52 < v49 )
        goto LABEL_771;
      v53 = *(_QWORD *)(v44 + 24);
      *(_DWORD *)(v44 + 32) = v46 - v45 + v52;
      *(_QWORD *)(v44 + 24) = v53 >> v49;
      v54 = v1049 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v46 - (unsigned __int8)v45 + 64)) & v53) << v46);
LABEL_44:
      if ( (unsigned int)sub_1510D70(*(_QWORD *)(v11 + 232), v54, (__int64)v10, 0) == 10 )
      {
        v507 = *((unsigned int *)v10 + 2);
        v508 = sub_1632440(*(_QWORD *)(v11 + 248), v1101, (unsigned int)v1102);
        if ( !(_DWORD)v507 )
        {
LABEL_492:
          v38 = v1101;
          if ( (_BYTE *)v1101 != v1103 )
            goto LABEL_29;
          goto LABEL_14;
        }
        v509 = 8 * v507;
        v510 = 0;
        v1082 = v509;
        while ( 1 )
        {
          v511 = (_BYTE *)sub_1517EB0(v11, (*v10)[v510 / 8]);
          if ( !v511 || (unsigned __int8)(*v511 - 4) > 0x1Eu )
            break;
          v510 += 8LL;
          sub_1623CA0(v508, v511);
          if ( v510 == v1082 )
            goto LABEL_492;
        }
        BYTE1(v1106[0]) = 1;
        v55 = "Invalid record";
      }
      else
      {
        BYTE1(v1106[0]) = 1;
        v55 = "METADATA_NAME not followed by METADATA_NAMED_NODE";
      }
      v1105.m128i_i64[0] = (__int64)v55;
      LOBYTE(v1106[0]) = 3;
      sub_1514BE0(v8, (__int64)&v1105);
      v56 = (_QWORD *)v1101;
      if ( (_BYTE *)v1101 != v1103 )
LABEL_47:
        _libc_free((unsigned __int64)v56);
      return v8;
    case 5:
      v1097 = 1;
LABEL_17:
      v28 = *((unsigned int *)a3 + 2);
      v1105.m128i_i64[0] = (__int64)v1106;
      v1105.m128i_i64[1] = 0x800000000LL;
      if ( (unsigned int)v28 > 8 )
      {
        sub_16CD150(&v1105, v1106, v28, 8);
        v28 = *((unsigned int *)v10 + 2);
      }
      v29 = *v10;
      v30 = &v29[v28];
      if ( v29 == v30 )
      {
        v33 = v1105.m128i_u32[2];
      }
      else
      {
        do
        {
          v31 = 0;
          if ( (unsigned int)*v29 )
            v31 = sub_15217C0(v1099, (unsigned int)*v29 - 1);
          v32 = v1105.m128i_u32[2];
          if ( v1105.m128i_i32[2] >= (unsigned __int32)v1105.m128i_i32[3] )
          {
            v961 = v31;
            sub_16CD150(&v1105, v1106, 0, 8);
            v32 = v1105.m128i_u32[2];
            v31 = v961;
          }
          ++v29;
          *(_QWORD *)(v1105.m128i_i64[0] + 8 * v32) = v31;
          v33 = (unsigned int)++v1105.m128i_i32[2];
        }
        while ( v30 != v29 );
      }
      v34 = *v9;
      v35 = a2[30];
      if ( v1097 )
        v36 = sub_1627350(v35, v1105.m128i_i64[0], v33, 1, 1);
      else
        v36 = sub_1627350(v35, v1105.m128i_i64[0], v33, 0, 1);
      goto LABEL_27;
    case 6:
      sub_151ABA0(v1105.m128i_i64, (__int64)a2, a3);
      v68 = v1105.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v1105.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_62;
      goto LABEL_14;
    case 7:
      if ( *((_DWORD *)a3 + 2) != 5 )
        goto LABEL_64;
      v79 = (*a3)[3];
      v80 = (*a3)[2];
      v1050 = (*a3)[1];
      v1097 = **a3 != 0;
      v81 = sub_15217C0(v1104, v79);
      v82 = 0;
      v83 = v81;
      v84 = (*v10)[4];
      if ( (_DWORD)v84 )
      {
        v993 = v83;
        v85 = sub_15217C0(v1099, (unsigned int)(v84 - 1));
        v83 = v993;
        v82 = v85;
      }
      v86 = *v9;
      v87 = sub_15B9E00(*(_QWORD *)(v11 + 240), v1050, v80, v83, v82, v1097 != 0, 1);
      goto LABEL_80;
    case 8:
      a1 = (__m128i *)*((unsigned int *)a3 + 2);
      if ( ((_DWORD)a3[1] & 1) != 0 )
        goto LABEL_64;
      v1105.m128i_i64[0] = (__int64)v1106;
      v1105.m128i_i64[1] = 0x800000000LL;
      v57 = &v1100;
      if ( !(_DWORD)a1 )
        goto LABEL_459;
      v934 = v8;
      v58 = 0;
      v59 = (int)a1;
      while ( 1 )
      {
        a3 = (__int64 **)*v10;
        v64 = *(_QWORD *)(v11 + 272) == 0;
        LODWORD(v1100) = (*v10)[v58];
        if ( v64 )
          goto LABEL_767;
        a1 = (__m128i *)(v11 + 256);
        a2 = &v1100;
        v65 = (*(__int64 (__fastcall **)(__int64, __int64 *, __int64 **, __int64 *))(v11 + 280))(
                v11 + 256,
                &v1100,
                a3,
                v57);
        if ( !v65 )
        {
          v8 = v934;
          v1103[1] = 1;
          v1101 = (unsigned __int64)"Invalid record";
          v1103[0] = 3;
          sub_1514BE0(v934, (__int64)&v1101);
          v56 = (_QWORD *)v1105.m128i_i64[0];
          if ( (_QWORD *)v1105.m128i_i64[0] != v1106 )
            goto LABEL_47;
          return v8;
        }
        v66 = *(_BYTE *)(v65 + 8);
        if ( v66 == 8 )
        {
          a1 = (__m128i *)v1104;
          a2 = (__int64 *)(*v10)[v58 + 1];
          v62 = sub_15217C0(v1104, a2);
          v63 = v1105.m128i_u32[2];
          if ( v1105.m128i_i32[2] >= (unsigned __int32)v1105.m128i_i32[3] )
            goto LABEL_445;
        }
        else
        {
          if ( !v66 )
          {
            v67 = v1105.m128i_u32[2];
            if ( v1105.m128i_i32[2] >= (unsigned __int32)v1105.m128i_i32[3] )
            {
              a2 = v1106;
              a1 = &v1105;
              sub_16CD150(&v1105, v1106, 0, 8);
              v67 = v1105.m128i_u32[2];
            }
            *(_QWORD *)(v1105.m128i_i64[0] + 8 * v67) = 0;
            ++v1105.m128i_i32[2];
            goto LABEL_53;
          }
          a2 = (__int64 *)(*v10)[v58 + 1];
          a1 = (__m128i *)sub_1522F40(*(_QWORD *)(v11 + 224), a2);
          v62 = sub_1624210(a1, a2, v60, v61);
          v63 = v1105.m128i_u32[2];
          if ( v1105.m128i_i32[2] >= (unsigned __int32)v1105.m128i_i32[3] )
          {
LABEL_445:
            a2 = v1106;
            a1 = &v1105;
            v884 = v62;
            sub_16CD150(&v1105, v1106, 0, 8);
            v63 = v1105.m128i_u32[2];
            v62 = v884;
          }
        }
        *(_QWORD *)(v1105.m128i_i64[0] + 8 * v63) = v62;
        ++v1105.m128i_i32[2];
LABEL_53:
        v58 += 2;
        if ( v58 == v59 )
        {
          v8 = v934;
LABEL_459:
          v34 = *v9;
          v483 = sub_1627350(*(_QWORD *)(v11 + 240), v1105.m128i_i64[0], v1105.m128i_u32[2], 0, 1);
LABEL_460:
          v36 = v483;
LABEL_27:
          v37 = v34;
LABEL_28:
          sub_15194E0(v11, v36, v37);
          ++*v9;
          v38 = v1105.m128i_i64[0];
          if ( (_QWORD *)v1105.m128i_i64[0] == v1106 )
            goto LABEL_14;
LABEL_29:
          _libc_free(v38);
LABEL_14:
          *v8 = 1;
          return v8;
        }
      }
    case 9:
      v295 = *((_DWORD *)a3 + 2);
      if ( (v295 & 1) != 0 )
        goto LABEL_64;
      if ( v295 != 2 )
      {
        v13 = *a6;
LABEL_269:
        v296 = sub_1627350(a2[30], 0, 0, 0, 1);
        goto LABEL_270;
      }
      v64 = a2[34] == 0;
      v1098 = **a3;
      if ( v64 )
LABEL_767:
        sub_4263D6(a1, a2, a3);
      v599 = ((__int64 (__fastcall *)(__int64 *, int *))a2[35])(a2 + 32, &v1098);
      v13 = *v9;
      if ( (*(_BYTE *)(v599 + 8) & 0xF7) == 0 )
        goto LABEL_269;
LABEL_464:
      v484 = (*v10)[1];
      v485 = sub_1522F40(*(_QWORD *)(v11 + 224), v484);
      v296 = sub_1624210(v485, v484, v486, v487);
LABEL_270:
      v26 = v296;
      goto LABEL_13;
    case 12:
      v280 = *((_DWORD *)a3 + 2);
      if ( v280 <= 3 )
        goto LABEL_64;
      v281 = *a3;
      v282 = (*a3)[1];
      v283 = *((_DWORD *)*a3 + 4);
      v1008 = (*v10)[1];
      v1097 = **v10 != 0;
      if ( v283 || (unsigned int)v282 > 0xFFFF )
        goto LABEL_64;
      v950 = 0;
      v284 = v281[3];
      if ( (_DWORD)v284 )
      {
        v285 = sub_15217C0(v1104, (unsigned int)(v284 - 1));
        v280 = *((_DWORD *)v10 + 2);
        v950 = v285;
      }
      v1105.m128i_i64[0] = (__int64)v1106;
      v1105.m128i_i64[1] = 0x800000000LL;
      if ( v280 == 4 )
      {
        v292 = 0;
      }
      else
      {
        v909 = v9;
        v286 = v10;
        v287 = 4;
        v288 = v280;
        do
        {
          v289 = 0;
          v290 = (*v286)[v287];
          if ( (_DWORD)v290 )
            v289 = sub_15217C0(v1099, (unsigned int)(v290 - 1));
          v291 = v1105.m128i_u32[2];
          if ( v1105.m128i_i32[2] >= (unsigned __int32)v1105.m128i_i32[3] )
          {
            v885 = v289;
            sub_16CD150(&v1105, v1106, 0, 8);
            v291 = v1105.m128i_u32[2];
            v289 = v885;
          }
          ++v287;
          *(_QWORD *)(v1105.m128i_i64[0] + 8 * v291) = v289;
          v292 = ++v1105.m128i_i32[2];
        }
        while ( v287 != v288 );
        v9 = v909;
      }
      v293 = *v9;
      v294 = sub_15BA790(*(_QWORD *)(v11 + 240), v1008, v950, v1105.m128i_i32[0], v292, v1097 != 0, 1);
      v37 = v293;
      v36 = v294;
      goto LABEL_28;
    case 13:
      v278 = *a3;
      v279 = (unsigned __int64)**v10 >> 1;
      if ( v279 )
      {
        if ( v279 != 1 )
        {
          BYTE1(v1106[0]) = 1;
          v69 = "Invalid record: Unsupported version of DISubrange";
          goto LABEL_65;
        }
        v512 = ~((unsigned __int64)v278[*((unsigned int *)v10 + 2) - 1] >> 1);
        if ( (v278[*((unsigned int *)v10 + 2) - 1] & 1) == 0 )
          v512 = (unsigned __int64)v278[*((unsigned int *)v10 + 2) - 1] >> 1;
        v513 = v278[1];
        v514 = 0;
        if ( (_DWORD)v513 )
          v514 = sub_15217C0(v1104, (unsigned int)(v513 - 1));
        v195 = sub_15BB200(*(_QWORD *)(v11 + 240), v514, v512, 0, 1);
      }
      else
      {
        v515 = ~((unsigned __int64)v278[*((unsigned int *)v10 + 2) - 1] >> 1);
        if ( (v278[*((unsigned int *)v10 + 2) - 1] & 1) == 0 )
          v515 = (unsigned __int64)v278[*((unsigned int *)v10 + 2) - 1] >> 1;
        v195 = sub_15BB740(a2[30], v278[1], v515, 0, 1);
      }
      goto LABEL_176;
    case 14:
      if ( *((_DWORD *)a3 + 2) != 3 )
        goto LABEL_64;
      v270 = *a3;
      v13 = *a6;
      v271 = 0;
      v1097 = *(_BYTE *)*a3 & 1;
      v272 = v270[2];
      v273 = (*v270 & 2) != 0;
      if ( v1097 )
      {
        if ( (_DWORD)v272 )
        {
          v1064 = (*v270 & 2) != 0;
          v274 = sub_15217C0(v1104, (unsigned int)(v272 - 1));
          v273 = v1064;
          v271 = v274;
          v270 = *v10;
        }
        v275 = v270[1];
        v276 = a2[30];
        v277 = ~(v275 >> 1);
        if ( (v275 & 1) == 0 )
          v277 = v275 >> 1;
        v26 = sub_15BC290(v276, v277, v273, v271, 1, 1);
      }
      else
      {
        if ( (_DWORD)v272 )
        {
          v1083 = (*v270 & 2) != 0;
          v521 = sub_15217C0(v1104, (unsigned int)(v272 - 1));
          v273 = v1083;
          v271 = v521;
          v270 = *v10;
        }
        v522 = v270[1];
        v523 = a2[30];
        v524 = ~(v522 >> 1);
        if ( (v522 & 1) == 0 )
          v524 = v522 >> 1;
        v26 = sub_15BC290(v523, v524, v273, v271, 0, 1);
      }
      goto LABEL_13;
    case 15:
      if ( *((_DWORD *)a3 + 2) != 6 )
        goto LABEL_64;
      v262 = *a3;
      v263 = 0;
      v13 = *a6;
      v64 = **v10 == 0;
      v264 = (*v10)[2];
      v265 = (*v10)[5];
      v266 = (*v10)[4];
      v1097 = **v10 != 0;
      v267 = v262[3];
      if ( v64 )
      {
        if ( (_DWORD)v264 )
        {
          v967 = v266;
          v1030 = v265;
          v1084 = v262[3];
          v525 = sub_15217C0(v1104, (unsigned int)(v264 - 1));
          LODWORD(v266) = v967;
          LODWORD(v265) = v1030;
          v263 = v525;
          LODWORD(v267) = v1084;
          v262 = *v10;
        }
        v269 = sub_15BC830(a2[30], v262[1], v263, v267, v266, v265, 0, 1);
      }
      else
      {
        if ( (_DWORD)v264 )
        {
          v949 = v266;
          v1007 = v265;
          v1063 = v262[3];
          v268 = sub_15217C0(v1104, (unsigned int)(v264 - 1));
          LODWORD(v266) = v949;
          LODWORD(v265) = v1007;
          v263 = v268;
          LODWORD(v267) = v1063;
          v262 = *v10;
        }
        v269 = sub_15BC830(a2[30], v262[1], v263, v267, v266, v265, 1, 1);
      }
      v26 = v269;
      goto LABEL_13;
    case 16:
      v247 = *((unsigned int *)a3 + 2);
      v248 = v247;
      if ( (_DWORD)v247 == 3 )
      {
        v498 = *a3;
        v250 = 0;
        v13 = *a6;
        v249 = *v10;
        v1097 = **v10 != 0;
        if ( v1097 )
          goto LABEL_485;
      }
      else
      {
        if ( (unsigned __int64)(v247 - 5) > 1 )
          goto LABEL_64;
        v249 = *a3;
        LOBYTE(a3) = **a3 != 0;
        v250 = 0;
        v64 = (*v10)[3] == 0;
        v1097 = **v10 != 0;
        if ( !v64 )
        {
          v251 = v249[4];
          v250 = 0;
          if ( v251 )
          {
            v679 = sub_1521990(&v1099, v251, a3, 0);
            LOBYTE(a3) = v1097;
            v250 = 1;
            v870 = v679;
            v249 = *v10;
            v899 = *((_DWORD *)*v10 + 6);
            v248 = *((_DWORD *)v10 + 2);
          }
        }
        v13 = *v9;
        if ( (_BYTE)a3 )
        {
          if ( v248 > 5 )
          {
            v834 = 0;
            v252 = v249[5];
            if ( (_DWORD)v252 )
            {
              v1004 = v250;
              v1060 = (char)a3;
              v253 = sub_15217C0(v1099, (unsigned int)(v252 - 1));
              LOBYTE(a3) = v1060;
              v250 = v1004;
              v834 = v253;
              v249 = *v10;
            }
            goto LABEL_226;
          }
LABEL_485:
          LOBYTE(a3) = 0;
LABEL_226:
          v254 = v249[2];
          v255 = 0;
          if ( (_DWORD)v254 )
          {
            v1005 = v250;
            v1061 = (char)a3;
            v256 = sub_15217C0(v1099, (unsigned int)(v254 - 1));
            v250 = v1005;
            LOBYTE(a3) = v1061;
            v255 = v256;
            v249 = *v10;
          }
          v257 = v249[1];
          v258 = 0;
          if ( (_DWORD)v257 )
          {
            v948 = v255;
            v1006 = v250;
            v1062 = (char)a3;
            v259 = sub_15217C0(v1099, (unsigned int)(v257 - 1));
            v255 = v948;
            v250 = v1006;
            LOBYTE(a3) = v1062;
            v258 = v259;
          }
          LOBYTE(v1102) = (_BYTE)a3;
          v260 = *(_QWORD *)(v11 + 240);
          if ( (_BYTE)a3 )
            v1101 = v834;
          LOBYTE(v1106[0]) = v250;
          if ( v250 )
          {
            v1105.m128i_i32[0] = v899;
            v1105.m128i_i64[1] = v870;
          }
          v261 = sub_15BF650(v260, v258, v255, (unsigned int)&v1105, (unsigned int)&v1101, 1, 1);
          goto LABEL_235;
        }
        v498 = v249;
        if ( v248 > 5 )
        {
          v852 = 0;
          v600 = v249[5];
          if ( (_DWORD)v600 )
          {
            v1094 = v250;
            v601 = sub_15217C0(v1099, (unsigned int)(v600 - 1));
            v498 = *v10;
            v250 = v1094;
            v852 = v601;
          }
          v499 = 1;
LABEL_476:
          v500 = v498[2];
          v501 = 0;
          if ( (_DWORD)v500 )
          {
            v1027 = v499;
            v1080 = v250;
            v502 = sub_15217C0(v1099, (unsigned int)(v500 - 1));
            v498 = *v10;
            v499 = v1027;
            v250 = v1080;
            v501 = v502;
          }
          v503 = v498[1];
          v504 = 0;
          if ( (_DWORD)v503 )
          {
            v965 = v501;
            v1028 = v499;
            v1081 = v250;
            v505 = sub_15217C0(v1099, (unsigned int)(v503 - 1));
            v501 = v965;
            v499 = v1028;
            v250 = v1081;
            v504 = v505;
          }
          LOBYTE(v1102) = v499;
          v506 = *(_QWORD *)(v11 + 240);
          if ( v499 )
            v1101 = v852;
          LOBYTE(v1106[0]) = v250;
          if ( v250 )
          {
            v1105.m128i_i32[0] = v899;
            v1105.m128i_i64[1] = v870;
          }
          v261 = sub_15BF650(v506, v504, v501, (unsigned int)&v1105, (unsigned int)&v1101, 0, 1);
LABEL_235:
          v26 = v261;
          goto LABEL_13;
        }
      }
      v499 = 0;
      goto LABEL_476;
    case 17:
      v222 = *((unsigned int *)a3 + 2);
      if ( (unsigned __int64)(v222 - 12) > 1 )
        goto LABEL_64;
      v223 = *a3;
      v224 = 0;
      if ( v222 == 13 )
      {
        v669 = v223[12];
        v224 = 0;
        if ( v669 )
        {
          v224 = 1;
          v820 = v669 - 1;
        }
      }
      v225 = v223[10];
      v226 = 0;
      v64 = *v223 == 0;
      v1097 = *v223 != 0;
      v1003 = v225;
      v1056 = *a6;
      v227 = v223[11];
      if ( v64 )
      {
        if ( (_DWORD)v227 )
          v226 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v227 - 1));
        v602 = sub_1519FE0(v11, v226);
        v603 = 0;
        v889 = v602;
        v976 = (*v10)[9];
        v845 = (*v10)[8];
        v604 = (*v10)[6];
        v924 = (*v10)[7];
        if ( (_DWORD)v604 )
          v603 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v604 - 1));
        v605 = sub_1519FE0(v11, v603);
        v606 = 0;
        v864 = v605;
        v607 = (*v10)[5];
        if ( (_DWORD)v607 )
          v606 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v607 - 1));
        v608 = sub_1519FE0(v11, v606);
        v609 = 0;
        v610 = v608;
        v611 = *v10;
        v612 = (*v10)[3];
        v613 = (*v10)[4];
        if ( (_DWORD)v612 )
        {
          v803 = (*v10)[4];
          v814 = v610;
          v614 = sub_15217C0(v1099, (unsigned int)(v612 - 1));
          LODWORD(v613) = v803;
          v610 = v814;
          v609 = v614;
          v611 = *v10;
        }
        v615 = v611[2];
        v616 = 0;
        if ( (_DWORD)v615 )
        {
          v795 = v613;
          v804 = v610;
          v815 = v609;
          v617 = sub_15217C0(v1099, (unsigned int)(v615 - 1));
          LODWORD(v613) = v795;
          v610 = v804;
          v616 = v617;
          v609 = v815;
          v611 = *v10;
        }
        v618 = v611[1];
        v619 = *(_QWORD *)(v11 + 240);
        v1105.m128i_i8[4] = v224;
        if ( v224 )
          v1105.m128i_i32[0] = v820;
        v246 = sub_15BD310(
                 v619,
                 v618,
                 v616,
                 v609,
                 v613,
                 v610,
                 v864,
                 v924,
                 v845,
                 v976,
                 (__int64)&v1105,
                 v1003,
                 v889,
                 0,
                 1);
      }
      else
      {
        if ( (_DWORD)v227 )
          v226 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v227 - 1));
        v228 = sub_1519FE0(v11, v226);
        v229 = 0;
        v878 = v228;
        v947 = (*v10)[9];
        v838 = (*v10)[8];
        v230 = (*v10)[6];
        v908 = (*v10)[7];
        if ( (_DWORD)v230 )
          v229 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v230 - 1));
        v231 = sub_1519FE0(v11, v229);
        v232 = 0;
        v858 = v231;
        v233 = (*v10)[5];
        if ( (_DWORD)v233 )
          v232 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v233 - 1));
        v234 = sub_1519FE0(v11, v232);
        v235 = 0;
        v236 = v234;
        v237 = *v10;
        v238 = (*v10)[3];
        v239 = (*v10)[4];
        if ( (_DWORD)v238 )
        {
          v799 = (*v10)[4];
          v810 = v236;
          v240 = sub_15217C0(v1099, (unsigned int)(v238 - 1));
          LODWORD(v239) = v799;
          v236 = v810;
          v235 = v240;
          v237 = *v10;
        }
        v241 = v237[2];
        v242 = 0;
        if ( (_DWORD)v241 )
        {
          v792 = v239;
          v800 = v236;
          v811 = v235;
          v243 = sub_15217C0(v1099, (unsigned int)(v241 - 1));
          LODWORD(v239) = v792;
          v236 = v800;
          v242 = v243;
          v235 = v811;
          v237 = *v10;
        }
        v244 = v237[1];
        v245 = *(_QWORD *)(v11 + 240);
        v1105.m128i_i8[4] = v224;
        if ( v224 )
          v1105.m128i_i32[0] = v820;
        v246 = sub_15BD310(
                 v245,
                 v244,
                 v242,
                 v235,
                 v239,
                 v236,
                 v858,
                 v908,
                 v838,
                 v947,
                 (__int64)&v1105,
                 v1003,
                 v878,
                 1,
                 1);
      }
      v175 = v246;
      goto LABEL_152;
    case 18:
      if ( (unsigned __int64)*((unsigned int *)a3 + 2) - 16 > 1 )
        goto LABEL_64;
      v1002 = 0;
      v198 = *a3;
      v1097 = **a3;
      v199 = v198[2];
      v1097 &= 1u;
      v837 = *v198;
      v946 = v198[1];
      if ( (_DWORD)v199 )
      {
        v1002 = sub_15217C0(v1104, (unsigned int)(v199 - 1));
        v198 = *v10;
      }
      v1059 = 0;
      v200 = v198[3];
      if ( (_DWORD)v200 )
      {
        v1059 = sub_15217C0(v1099, (unsigned int)(v200 - 1));
        v198 = *v10;
      }
      v201 = v198[4];
      v202 = v198[5];
      v203 = 0;
      v907 = v201;
      if ( (_DWORD)v202 )
        v203 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v202 - 1));
      v857 = sub_1519FE0(v11, v203);
      v205 = *v10;
      v877 = (*v10)[8];
      if ( v877 > 0xFFFFFFFF )
        goto LABEL_457;
      v781 = v205[7];
      v798 = v205[10];
      v777 = v205[12];
      v808 = v205[10];
      v206 = sub_1521990(&v1099, v205[15], v808, v204);
      v207 = v206;
      if ( *(_BYTE *)(v11 + 1012)
        && byte_4F9DD80 != 1
        && v206
        && ((v208 = v808, (((_DWORD)v946 - 2) & 0xFFFFFFFD) == 0) || (v946 & 0xFFFFFFFB) == 0x13) )
      {
        v786 = 0;
        v209 = 0;
        v824 = 0;
        v809 = 0;
        v791 = 0;
        v798 = v208 | 4;
        v210 = 0;
      }
      else
      {
        v824 = 0;
        v791 = sub_1521950(v11, &v1099, (*v10)[6]);
        v671 = *v10;
        v672 = (*v10)[11];
        v809 = (*v10)[9];
        if ( (_DWORD)v672 )
        {
          v824 = sub_15217C0(v1099, (unsigned int)(v672 - 1));
          v671 = *v10;
        }
        v673 = v671[13];
        v674 = 0;
        if ( (_DWORD)v673 )
          v674 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v673 - 1));
        v675 = sub_1519FE0(v11, v674);
        v209 = 0;
        v786 = v675;
        v676 = (*v10)[14];
        if ( (_DWORD)v676 )
          v209 = sub_15217C0(v1099, (unsigned int)(v676 - 1));
        v210 = 0;
        if ( *((_DWORD *)v10 + 2) > 0x10u )
        {
          v702 = (*v10)[16];
          v210 = 0;
          if ( (_DWORD)v702 )
          {
            v776 = v209;
            v210 = sub_15217C0(v1099, (unsigned int)(v702 - 1));
            v209 = v776;
          }
        }
        if ( !v207 )
          goto LABEL_694;
      }
      v773 = v209;
      v211 = sub_15BE040(
               *(_QWORD *)(v11 + 240),
               v207,
               v946,
               v1002,
               v1059,
               v201,
               v857,
               v791,
               v781,
               v877,
               v809,
               v798,
               v824,
               v777,
               v786,
               v209,
               v210);
      v209 = v773;
      v212 = v211;
      if ( v211 )
        goto LABEL_192;
LABEL_694:
      v677 = *(_QWORD *)(v11 + 240);
      if ( v1097 )
        v678 = sub_15BDB40(
                 v677,
                 v946,
                 v1002,
                 v1059,
                 v907,
                 v857,
                 v791,
                 v781,
                 v877,
                 v809,
                 v798,
                 v824,
                 v777,
                 v786,
                 v209,
                 v207,
                 0,
                 1,
                 1);
      else
        v678 = sub_15BDB40(
                 v677,
                 v946,
                 v1002,
                 v1059,
                 v907,
                 v857,
                 v791,
                 v781,
                 v877,
                 v809,
                 v798,
                 v824,
                 v777,
                 v786,
                 v209,
                 v207,
                 0,
                 0,
                 1);
      v212 = v678;
LABEL_192:
      if ( v837 > 1 || !v207 )
        goto LABEL_199;
      if ( (*(_BYTE *)(v212 + 28) & 4) != 0 )
      {
        v213 = *(_BYTE *)(v11 + 160) & 1;
        if ( v213 )
        {
          v214 = v11 + 168;
          v215 = 1;
        }
        else
        {
          v724 = *(_DWORD *)(v11 + 176);
          v214 = *(_QWORD *)(v11 + 168);
          v215 = v724;
          if ( !v724 )
          {
            v725 = *(_DWORD *)(v11 + 160);
            ++*(_QWORD *)(v11 + 152);
            v220 = 0;
            v726 = (v725 >> 1) + 1;
            goto LABEL_763;
          }
        }
        v216 = v215 - 1;
        v217 = 1;
        v218 = 0;
        for ( i = v216 & (((unsigned int)v207 >> 9) ^ ((unsigned int)v207 >> 4)); ; i = v216 & v757 )
        {
          v220 = (__int64 *)(v214 + 16LL * i);
          v221 = *v220;
          if ( v207 == *v220 )
            goto LABEL_199;
          if ( v221 == -8 )
            break;
          if ( v221 != -16 || v218 )
            v220 = v218;
          v757 = v217 + i;
          v218 = v220;
          ++v217;
        }
        v725 = *(_DWORD *)(v11 + 160);
        v724 = 1;
        if ( v218 )
          v220 = v218;
        ++*(_QWORD *)(v11 + 152);
        v726 = (v725 >> 1) + 1;
        if ( !v213 )
          v724 = *(_DWORD *)(v11 + 176);
LABEL_763:
        if ( 4 * v726 < 3 * v724 )
        {
          if ( v724 - *(_DWORD *)(v11 + 164) - v726 > v724 >> 3 )
            goto LABEL_765;
          sub_1519820(v11 + 152, v724);
          if ( (*(_BYTE *)(v11 + 160) & 1) != 0 )
          {
            v727 = v11 + 168;
            v728 = 1;
            goto LABEL_774;
          }
          v728 = *(_DWORD *)(v11 + 176);
          v727 = *(_QWORD *)(v11 + 168);
          if ( v728 )
          {
LABEL_774:
            v729 = v728 - 1;
            v730 = 1;
            v731 = 0;
            for ( j = (v728 - 1) & (((unsigned int)v207 >> 9) ^ ((unsigned int)v207 >> 4)); ; j = v729 & v740 )
            {
              v220 = (__int64 *)(v727 + 16LL * j);
              v733 = *v220;
              if ( v207 == *v220 )
                break;
              if ( v733 == -8 )
                goto LABEL_782;
              if ( v731 || v733 != -16 )
                v220 = v731;
              v740 = v730 + j;
              v731 = v220;
              ++v730;
            }
LABEL_776:
            v725 = *(_DWORD *)(v11 + 160);
LABEL_765:
            *(_DWORD *)(v11 + 160) = (2 * (v725 >> 1) + 2) | v725 & 1;
            if ( *v220 != -8 )
              --*(_DWORD *)(v11 + 164);
LABEL_751:
            *v220 = v207;
            v220[1] = v212;
LABEL_199:
            sub_15194E0(v11, v212, *v9);
            ++*v9;
            goto LABEL_14;
          }
LABEL_837:
          *(_DWORD *)(v11 + 160) = (2 * (*(_DWORD *)(v11 + 160) >> 1) + 2) | *(_DWORD *)(v11 + 160) & 1;
          BUG();
        }
        sub_1519820(v11 + 152, 2 * v724);
        if ( (*(_BYTE *)(v11 + 160) & 1) != 0 )
        {
          v734 = v11 + 168;
          v735 = 1;
        }
        else
        {
          v735 = *(_DWORD *)(v11 + 176);
          v734 = *(_QWORD *)(v11 + 168);
          if ( !v735 )
            goto LABEL_837;
        }
        v736 = v735 - 1;
        v737 = 1;
        v731 = 0;
        for ( k = (v735 - 1) & (((unsigned int)v207 >> 9) ^ ((unsigned int)v207 >> 4)); ; k = v736 & v741 )
        {
          v220 = (__int64 *)(v734 + 16LL * k);
          v739 = *v220;
          if ( v207 == *v220 )
            goto LABEL_776;
          if ( v739 == -8 )
            break;
          if ( v731 || v739 != -16 )
            v220 = v731;
          v741 = v737 + k;
          v731 = v220;
          ++v737;
        }
LABEL_782:
        if ( v731 )
          v220 = v731;
        goto LABEL_776;
      }
      v707 = *(_BYTE *)(v11 + 128) & 1;
      if ( v707 )
      {
        v708 = v11 + 136;
        v709 = 1;
      }
      else
      {
        v716 = *(_DWORD *)(v11 + 144);
        v708 = *(_QWORD *)(v11 + 136);
        v709 = v716;
        if ( !v716 )
        {
          v715 = *(_DWORD *)(v11 + 128);
          ++*(_QWORD *)(v11 + 120);
          v220 = 0;
          v717 = (v715 >> 1) + 1;
          goto LABEL_747;
        }
      }
      v710 = v709 - 1;
      v711 = 1;
      v712 = 0;
      for ( m = (v709 - 1) & (((unsigned int)v207 >> 9) ^ ((unsigned int)v207 >> 4)); ; m = v710 & v755 )
      {
        v220 = (__int64 *)(v708 + 16LL * m);
        v714 = *v220;
        if ( v207 == *v220 )
          goto LABEL_199;
        if ( v714 == -8 )
          break;
        if ( v714 != -16 || v712 )
          v220 = v712;
        v755 = v711 + m;
        v712 = v220;
        ++v711;
      }
      v715 = *(_DWORD *)(v11 + 128);
      v716 = 1;
      if ( v712 )
        v220 = v712;
      ++*(_QWORD *)(v11 + 120);
      v717 = (v715 >> 1) + 1;
      if ( !v707 )
        v716 = *(_DWORD *)(v11 + 144);
LABEL_747:
      if ( 4 * v717 < 3 * v716 )
      {
        if ( v716 - *(_DWORD *)(v11 + 132) - v717 > v716 >> 3 )
          goto LABEL_749;
        sub_1519820(v11 + 120, v716);
        if ( (*(_BYTE *)(v11 + 128) & 1) != 0 )
        {
          v742 = v11 + 136;
          v743 = 1;
          goto LABEL_804;
        }
        v743 = *(_DWORD *)(v11 + 144);
        v742 = *(_QWORD *)(v11 + 136);
        if ( v743 )
        {
LABEL_804:
          v744 = v743 - 1;
          v745 = 1;
          v746 = 0;
          for ( n = (v743 - 1) & (((unsigned int)v207 >> 9) ^ ((unsigned int)v207 >> 4)); ; n = v744 & v756 )
          {
            v220 = (__int64 *)(v742 + 16LL * n);
            v748 = *v220;
            if ( v207 == *v220 )
              break;
            if ( v748 == -8 )
              goto LABEL_812;
            if ( v748 != -16 || v746 )
              v220 = v746;
            v756 = v745 + n;
            v746 = v220;
            ++v745;
          }
LABEL_806:
          v715 = *(_DWORD *)(v11 + 128);
LABEL_749:
          *(_DWORD *)(v11 + 128) = (2 * (v715 >> 1) + 2) | v715 & 1;
          if ( *v220 != -8 )
            --*(_DWORD *)(v11 + 132);
          goto LABEL_751;
        }
LABEL_838:
        *(_DWORD *)(v11 + 128) = (2 * (*(_DWORD *)(v11 + 128) >> 1) + 2) | *(_DWORD *)(v11 + 128) & 1;
        BUG();
      }
      sub_1519820(v11 + 120, 2 * v716);
      if ( (*(_BYTE *)(v11 + 128) & 1) != 0 )
      {
        v749 = v11 + 136;
        v750 = 1;
      }
      else
      {
        v750 = *(_DWORD *)(v11 + 144);
        v749 = *(_QWORD *)(v11 + 136);
        if ( !v750 )
          goto LABEL_838;
      }
      v751 = v750 - 1;
      v752 = 1;
      v746 = 0;
      for ( ii = (v750 - 1) & (((unsigned int)v207 >> 9) ^ ((unsigned int)v207 >> 4)); ; ii = v751 & v758 )
      {
        v220 = (__int64 *)(v749 + 16LL * ii);
        v754 = *v220;
        if ( v207 == *v220 )
          goto LABEL_806;
        if ( v754 == -8 )
          break;
        if ( v754 != -16 || v746 )
          v220 = v746;
        v758 = v752 + ii;
        v746 = v220;
        ++v752;
      }
LABEL_812:
      if ( v746 )
        v220 = v746;
      goto LABEL_806;
    case 19:
      v465 = *((unsigned int *)a3 + 2);
      if ( (unsigned __int64)(v465 - 3) > 1 )
        goto LABEL_64;
      v466 = *a3;
      v467 = 0;
      v468 = **v10;
      if ( v465 == 4 )
        v467 = *((unsigned __int8 *)v466 + 24);
      v469 = v466[1];
      v470 = v466[2];
      v471 = v468 & 1;
      v1097 = v468 & 1;
      if ( (_DWORD)v470 )
      {
        v1021 = v467;
        v472 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v470 - 1));
        v467 = v1021;
        v473 = (__int64)v472;
        if ( v468 > 1 )
        {
LABEL_442:
          v471 = v1097;
          goto LABEL_443;
        }
        if ( v472 )
        {
          if ( *v472 == 4 )
          {
            v705 = v472[1];
            if ( v705 != 1 )
            {
              if ( v705 == 2 )
              {
                v718 = sub_1627350(a2[27], 0, 0, 2, 1);
                v467 = v1021;
                v720 = v718;
                v721 = *((_DWORD *)a2 + 48);
                if ( v721 >= *((_DWORD *)a2 + 49) )
                {
                  v989 = v720;
                  sub_1516860((__int64)(a2 + 23), 0);
                  v721 = *((_DWORD *)a2 + 48);
                  v720 = v989;
                  v467 = v1021;
                }
                v722 = a2[23];
                v723 = (__int64 *)(v722 + 16LL * v721);
                if ( v723 )
                {
                  *v723 = v473;
                  v988 = v720;
                  v1046 = v467;
                  sub_1623A60(v723, v473, 2);
                  v467 = v1046;
                  v723[1] = v988;
                  v722 = a2[23];
                  ++*((_DWORD *)a2 + 48);
                }
                else
                {
                  *((_DWORD *)a2 + 48) = v721 + 1;
                  if ( v720 )
                  {
                    v1047 = v467;
                    sub_16307F0(v720, 0, v722, v719, v720);
                    v722 = a2[23];
                    v467 = v1047;
                  }
                }
                v473 = *(_QWORD *)(v722 + 16LL * *((unsigned int *)a2 + 48) - 8);
              }
              else
              {
                v706 = sub_151A3A0((__int64)a2, v473);
                v467 = v1021;
                v473 = v706;
              }
            }
          }
          goto LABEL_442;
        }
      }
      else if ( v468 > 1 )
      {
        v473 = 0;
LABEL_443:
        v86 = *v9;
        v87 = sub_15BEF40(a2[30], (unsigned int)v469, v467, v473, v471 != 0, 1);
LABEL_80:
        sub_15194E0(v11, v87, v86);
        ++*v9;
        goto LABEL_14;
      }
      v473 = 0;
      goto LABEL_442;
    case 20:
      v420 = *((unsigned int *)a3 + 2);
      if ( v420 - 14 > 5 )
        goto LABEL_64;
      v1097 = 1;
      v421 = *v10;
      switch ( v420 )
      {
        case 0x13uLL:
          v794 = v421[18] != 0;
          break;
        case 0x12uLL:
          v794 = 0;
          break;
        case 0x11uLL:
          v862 = 0;
          v794 = 0;
LABEL_681:
          v802 = v421[16] != 0;
          goto LABEL_682;
        case 0xEuLL:
          v813 = 0;
          v802 = 1;
          v862 = 0;
          v794 = 0;
          v883 = 0;
          goto LABEL_393;
        default:
          v802 = 1;
          v862 = 0;
          v794 = 0;
LABEL_682:
          v883 = 0;
          v813 = v421[14];
          if ( v420 > 0xF )
          {
            v670 = v421[15];
            if ( (_DWORD)v670 )
            {
              v883 = sub_15217C0(v1104, (unsigned int)(v670 - 1));
              v421 = *v10;
            }
          }
LABEL_393:
          v422 = v421[13];
          v423 = 0;
          if ( (_DWORD)v422 )
          {
            v423 = sub_15217C0(v1099, (unsigned int)(v422 - 1));
            v421 = *v10;
          }
          v916 = 0;
          v424 = v421[12];
          if ( (_DWORD)v424 )
          {
            v916 = sub_15217C0(v1099, (unsigned int)(v424 - 1));
            v421 = *v10;
          }
          v960 = 0;
          v425 = v421[10];
          if ( (_DWORD)v425 )
          {
            v960 = sub_15217C0(v1099, (unsigned int)(v425 - 1));
            v421 = *v10;
          }
          v1019 = 0;
          v426 = v421[9];
          if ( (_DWORD)v426 )
          {
            v1019 = sub_15217C0(v1099, (unsigned int)(v426 - 1));
            v421 = *v10;
          }
          v427 = v421[7];
          v1074 = 0;
          v844 = v421[8];
          if ( (_DWORD)v427 )
          {
            v1074 = sub_15217C0(v1099, (unsigned int)(v427 - 1));
            v421 = *v10;
          }
          v428 = v421[5];
          v429 = 0;
          v827 = v421[6];
          if ( (_DWORD)v428 )
          {
            v429 = sub_15217C0(v1099, (unsigned int)(v428 - 1));
            v421 = *v10;
          }
          v430 = v421[3];
          v431 = v421[4] != 0;
          v432 = 0;
          if ( (_DWORD)v430 )
          {
            v783 = v429;
            v788 = v421[4] != 0;
            v433 = sub_15217C0(v1099, (unsigned int)(v430 - 1));
            v429 = v783;
            v431 = v788;
            v432 = v433;
            v421 = *v10;
          }
          v434 = v421[2];
          v435 = 0;
          if ( (_DWORD)v434 )
          {
            v779 = v429;
            v784 = v432;
            v789 = v431;
            v436 = sub_15217C0(v1099, (unsigned int)(v434 - 1));
            v429 = v779;
            v432 = v784;
            v435 = v436;
            v431 = v789;
            v421 = *v10;
          }
          v437 = sub_15B0DC0(
                   *(_QWORD *)(v11 + 240),
                   v421[1],
                   v435,
                   v432,
                   v431,
                   v429,
                   v827,
                   v1074,
                   v844,
                   v1019,
                   v960,
                   v916,
                   v423,
                   v883,
                   v813,
                   v802,
                   v862,
                   v794,
                   1);
          sub_15194E0(v11, v437, *v9);
          ++*v9;
          v438 = (*v10)[11];
          if ( (_DWORD)v438 )
          {
            v439 = sub_1517EB0(v11, (int)v438 - 1);
            if ( v439 )
            {
              v1105.m128i_i64[0] = v437;
              v440 = *(__m128i **)(v11 + 688);
              v1105.m128i_i64[1] = v439;
              if ( v440 == *(__m128i **)(v11 + 696) )
              {
                sub_1517670((const __m128i **)(v11 + 680), v440, &v1105);
              }
              else
              {
                if ( v440 )
                {
                  *v440 = _mm_loadu_si128(&v1105);
                  v440 = *(__m128i **)(v11 + 688);
                }
                *(_QWORD *)(v11 + 688) = v440 + 1;
              }
            }
          }
          goto LABEL_14;
      }
      v862 = v421[17] != 0;
      goto LABEL_681;
    case 21:
      v385 = *((unsigned int *)a3 + 2);
      if ( v385 - 18 > 3 )
        goto LABEL_64;
      v386 = *a3;
      v387 = 1;
      if ( (*(_BYTE *)*a3 & 1) == 0 )
        v387 = v386[8] != 0;
      v1097 = v387;
      v388 = *v386;
      v389 = v385 == 18 && (unsigned __int64)*v386 > 1;
      if ( v389 )
        goto LABEL_64;
      v390 = v386[15];
      if ( (_DWORD)v390 )
      {
        v1017 = *v386;
        v391 = sub_15217C0(v1104, (unsigned int)(v390 - 1));
        v385 = *((unsigned int *)v10 + 2);
        v388 = v1017;
        v392 = v391;
        v1073 = v391;
        v387 = v1097;
        v386 = *v10;
        LOBYTE(v393) = v385 > 0x12;
        v389 = v392 != 0 && v385 > 0x12 && v1017 <= 1;
      }
      else
      {
        v1073 = 0;
        v393 = v385 > 0x12;
      }
      v393 = (unsigned __int8)v393;
      if ( v387 )
      {
        v1018 = 0;
        if ( v385 > 0x14 )
        {
          v689 = v386[20];
          if ( (_DWORD)v689 )
          {
            v896 = (unsigned __int8)v393;
            v930 = v385;
            v985 = v388;
            v690 = sub_15217C0(v1099, (unsigned int)(v689 - 1));
            v393 = v896;
            v385 = v930;
            v1018 = v690;
            v388 = v985;
            v386 = *v10;
          }
        }
        v861 = 0;
        v394 = v386[v393 + 17];
        if ( (_DWORD)v394 )
        {
          v881 = v393;
          v913 = v385;
          v956 = v388;
          v395 = sub_15217C0(v1099, (unsigned int)(v394 - 1));
          v393 = v881;
          v385 = v913;
          v861 = v395;
          v388 = v956;
          v386 = *v10;
        }
        v882 = 0;
        v396 = v386[v393 + 16];
        if ( (_DWORD)v396 )
        {
          v841 = v393;
          v914 = v385;
          v957 = v388;
          v397 = sub_15217C0(v1099, (unsigned int)(v396 - 1));
          v393 = v841;
          v385 = v914;
          v882 = v397;
          v388 = v957;
          v386 = *v10;
        }
        v915 = 0;
        v398 = v386[v393 + 15];
        if ( (_DWORD)v398 )
        {
          v842 = v385;
          v958 = v388;
          v399 = sub_15217C0(v1099, (unsigned int)(v398 - 1));
          v385 = v842;
          v388 = v958;
          v915 = v399;
          v386 = *v10;
        }
        v959 = 0;
        v400 = v388 <= 1;
        v401 = 0;
        if ( !v400 )
          v401 = v1073;
        v787 = v401;
        v812 = v386[14] != 0;
        v778 = v386[13];
        if ( v385 > 0x13 )
          v959 = *((_DWORD *)v386 + 38);
        v402 = 0;
        v774 = v386[12];
        v403 = v386[11];
        v404 = v386[10];
        if ( (_DWORD)v404 )
          v402 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v404 - 1));
        v826 = 0;
        v782 = sub_1519FE0(v11, v402);
        v405 = *v10;
        v406 = (*v10)[6];
        v771 = (*v10)[9];
        v801 = (*v10)[8] != 0;
        v793 = (*v10)[7] != 0;
        if ( (_DWORD)v406 )
        {
          v826 = sub_15217C0(v1099, (unsigned int)(v406 - 1));
          v405 = *v10;
        }
        v407 = v405[4];
        v843 = 0;
        v769 = v405[5];
        if ( (_DWORD)v407 )
        {
          v843 = sub_15217C0(v1099, (unsigned int)(v407 - 1));
          v405 = *v10;
        }
        v408 = v405[3];
        v409 = 0;
        if ( (_DWORD)v408 )
        {
          v409 = sub_15217C0(v1099, (unsigned int)(v408 - 1));
          v405 = *v10;
        }
        v410 = v405[2];
        v411 = 0;
        if ( (_DWORD)v410 )
        {
          v763 = v409;
          v412 = sub_15217C0(v1099, (unsigned int)(v410 - 1));
          v409 = v763;
          v411 = v412;
          v405 = *v10;
        }
        v413 = v405[1];
        v414 = 0;
        if ( (_DWORD)v413 )
        {
          v759 = v409;
          v764 = v411;
          v415 = sub_15217C0(v1099, (unsigned int)(v413 - 1));
          v409 = v759;
          v411 = v764;
          v414 = (_BYTE *)v415;
        }
        v760 = v409;
        v765 = v411;
        v416 = sub_1519FE0(v11, v414);
        v417 = sub_15BFC70(
                 *(_QWORD *)(v11 + 240),
                 v416,
                 v765,
                 v760,
                 v843,
                 v769,
                 v826,
                 v793,
                 v801,
                 v771,
                 v782,
                 v403,
                 v774,
                 v959,
                 v778,
                 v812,
                 v787,
                 v915,
                 v882,
                 v861,
                 v1018,
                 1,
                 1);
      }
      else
      {
        v1043 = 0;
        if ( v385 > 0x14 )
        {
          v691 = v386[20];
          if ( (_DWORD)v691 )
          {
            v897 = (unsigned __int8)v393;
            v931 = v385;
            v986 = v388;
            v692 = sub_15217C0(v1099, (unsigned int)(v691 - 1));
            v393 = v897;
            v385 = v931;
            v1043 = v692;
            v388 = v986;
            v386 = *v10;
          }
        }
        v865 = 0;
        v621 = v386[v393 + 17];
        if ( (_DWORD)v621 )
        {
          v890 = v393;
          v925 = v385;
          v977 = v388;
          v622 = sub_15217C0(v1099, (unsigned int)(v621 - 1));
          v393 = v890;
          v385 = v925;
          v865 = v622;
          v388 = v977;
          v386 = *v10;
        }
        v891 = 0;
        v623 = v386[v393 + 16];
        if ( (_DWORD)v623 )
        {
          v846 = v393;
          v926 = v385;
          v978 = v388;
          v624 = sub_15217C0(v1099, (unsigned int)(v623 - 1));
          v393 = v846;
          v385 = v926;
          v891 = v624;
          v388 = v978;
          v386 = *v10;
        }
        v927 = 0;
        v625 = v386[v393 + 15];
        if ( (_DWORD)v625 )
        {
          v847 = v385;
          v979 = v388;
          v626 = sub_15217C0(v1099, (unsigned int)(v625 - 1));
          v385 = v847;
          v388 = v979;
          v927 = v626;
          v386 = *v10;
        }
        v980 = 0;
        v400 = v388 <= 1;
        v627 = 0;
        if ( !v400 )
          v627 = v1073;
        v790 = v627;
        v816 = v386[14] != 0;
        v780 = v386[13];
        if ( v385 > 0x13 )
          v980 = *((_DWORD *)v386 + 38);
        v628 = 0;
        v775 = v386[12];
        v629 = v386[11];
        v630 = v386[10];
        if ( (_DWORD)v630 )
          v628 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v630 - 1));
        v828 = 0;
        v785 = sub_1519FE0(v11, v628);
        v631 = *v10;
        v632 = (*v10)[6];
        v772 = (*v10)[9];
        v805 = (*v10)[8] != 0;
        v796 = (*v10)[7] != 0;
        if ( (_DWORD)v632 )
        {
          v828 = sub_15217C0(v1099, (unsigned int)(v632 - 1));
          v631 = *v10;
        }
        v633 = v631[4];
        v848 = 0;
        v770 = v631[5];
        if ( (_DWORD)v633 )
        {
          v848 = sub_15217C0(v1099, (unsigned int)(v633 - 1));
          v631 = *v10;
        }
        v634 = v631[3];
        v635 = 0;
        if ( (_DWORD)v634 )
        {
          v635 = sub_15217C0(v1099, (unsigned int)(v634 - 1));
          v631 = *v10;
        }
        v636 = v631[2];
        v637 = 0;
        if ( (_DWORD)v636 )
        {
          v766 = v635;
          v638 = sub_15217C0(v1099, (unsigned int)(v636 - 1));
          v635 = v766;
          v637 = v638;
          v631 = *v10;
        }
        v639 = v631[1];
        v640 = 0;
        if ( (_DWORD)v639 )
        {
          v761 = v635;
          v767 = v637;
          v641 = sub_15217C0(v1099, (unsigned int)(v639 - 1));
          v635 = v761;
          v637 = v767;
          v640 = (_BYTE *)v641;
        }
        v762 = v635;
        v768 = v637;
        v642 = sub_1519FE0(v11, v640);
        v417 = sub_15BFC70(
                 *(_QWORD *)(v11 + 240),
                 v642,
                 v768,
                 v762,
                 v848,
                 v770,
                 v828,
                 v796,
                 v805,
                 v772,
                 v785,
                 v629,
                 v775,
                 v980,
                 v780,
                 v816,
                 v790,
                 v927,
                 v891,
                 v865,
                 v1043,
                 0,
                 1);
      }
      v418 = v417;
      sub_15194E0(v11, v417, *v9);
      ++*v9;
      if ( v389 && *(_BYTE *)v1073 == 1 )
      {
        v419 = *(_QWORD *)(v1073 + 136);
        if ( !*(_BYTE *)(v419 + 16) )
        {
          v1105.m128i_i64[0] = *(_QWORD *)(v1073 + 136);
          if ( (*(_BYTE *)(v419 + 34) & 0x40) != 0 )
          {
            sub_151A8D0(v11 + 704, v1105.m128i_i64)[1] = v418;
          }
          else if ( v419 + 72 != (*(_QWORD *)(v419 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
          {
            sub_1627150(v419, v418);
          }
        }
      }
      goto LABEL_14;
    case 22:
      if ( *((_DWORD *)a3 + 2) != 5 )
        goto LABEL_64;
      v376 = *a3;
      v377 = 0;
      v13 = *a6;
      v64 = **a3 == 0;
      v378 = (*a3)[2];
      v379 = (*v10)[4];
      v380 = (*v10)[3];
      v1097 = **v10 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v378 )
        {
          v1040 = v380;
          v1088 = v379;
          v585 = sub_15217C0(v1104, (unsigned int)(v378 - 1));
          LODWORD(v380) = v1040;
          LODWORD(v379) = v1088;
          v377 = v585;
          v376 = *v10;
        }
        v586 = v376[1];
        v587 = 0;
        if ( (_DWORD)v586 )
        {
          v975 = v380;
          v1041 = v379;
          v1089 = v377;
          v588 = sub_15217C0(v1099, (unsigned int)(v586 - 1));
          LODWORD(v380) = v975;
          LODWORD(v379) = v1041;
          v377 = v1089;
          v587 = v588;
        }
        v26 = sub_15C06A0(*(_QWORD *)(v11 + 240), v587, v377, v380, v379, 0, 1);
      }
      else
      {
        if ( (_DWORD)v378 )
        {
          v1015 = v380;
          v1071 = v379;
          v381 = sub_15217C0(v1104, (unsigned int)(v378 - 1));
          LODWORD(v380) = v1015;
          LODWORD(v379) = v1071;
          v377 = v381;
          v376 = *v10;
        }
        v382 = v376[1];
        v383 = 0;
        if ( (_DWORD)v382 )
        {
          v955 = v380;
          v1016 = v379;
          v1072 = v377;
          v384 = sub_15217C0(v1099, (unsigned int)(v382 - 1));
          LODWORD(v380) = v955;
          LODWORD(v379) = v1016;
          v377 = v1072;
          v383 = v384;
        }
        v26 = sub_15C06A0(*(_QWORD *)(v11 + 240), v383, v377, v380, v379, 1, 1);
      }
      goto LABEL_13;
    case 23:
      if ( *((_DWORD *)a3 + 2) != 4 )
        goto LABEL_64;
      v450 = *a3;
      v451 = 0;
      v13 = *a6;
      v64 = **a3 == 0;
      v452 = (*a3)[2];
      v453 = (*v10)[3];
      v1097 = **v10 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v452 )
        {
          v1091 = v453;
          v594 = sub_15217C0(v1104, (unsigned int)(v452 - 1));
          v453 = v1091;
          v451 = v594;
          v450 = *v10;
        }
        v595 = v450[1];
        v596 = 0;
        if ( (_DWORD)v595 )
        {
          v1042 = v453;
          v1092 = v451;
          v597 = sub_15217C0(v1099, (unsigned int)(v595 - 1));
          v453 = v1042;
          v451 = v1092;
          v596 = v597;
        }
        v26 = sub_15C0C90(*(_QWORD *)(v11 + 240), v596, v451, v453, 0, 1);
      }
      else
      {
        if ( (_DWORD)v452 )
        {
          v1076 = v453;
          v454 = sub_15217C0(v1104, (unsigned int)(v452 - 1));
          v453 = v1076;
          v451 = v454;
          v450 = *v10;
        }
        v455 = v450[1];
        v456 = 0;
        if ( (_DWORD)v455 )
        {
          v1020 = v453;
          v1077 = v451;
          v457 = sub_15217C0(v1099, (unsigned int)(v455 - 1));
          v453 = v1020;
          v451 = v1077;
          v456 = v457;
        }
        v26 = sub_15C0C90(*(_QWORD *)(v11 + 240), v456, v451, v453, 1, 1);
      }
      goto LABEL_13;
    case 24:
      v441 = *((unsigned int *)a3 + 2);
      if ( v441 == 3 )
      {
        v442 = *a3;
        v443 = 0;
        v444 = (*a3)[2];
        if ( (_DWORD)v444 )
        {
LABEL_418:
          v443 = sub_15217C0(v1104, (unsigned int)(v444 - 1));
          v442 = *v10;
        }
      }
      else
      {
        if ( v441 != 5 )
          goto LABEL_64;
        v442 = *a3;
        v443 = 0;
        v444 = (*a3)[3];
        if ( (_DWORD)v444 )
          goto LABEL_418;
      }
      v13 = *v9;
      v445 = 0;
      v1097 = *(_BYTE *)v442 & 1;
      v446 = *v442;
      v447 = v442[1];
      v448 = (v446 & 2) != 0;
      if ( v1097 )
      {
        if ( (_DWORD)v447 )
        {
          v1075 = v443;
          v449 = sub_15217C0(v1099, (unsigned int)(v447 - 1));
          v443 = v1075;
          v445 = v449;
        }
        v26 = sub_15C1230(*(_QWORD *)(v11 + 240), v445, v443, v448, 1, 1);
      }
      else
      {
        if ( (_DWORD)v447 )
        {
          v1093 = v443;
          v598 = sub_15217C0(v1099, (unsigned int)(v447 - 1));
          v443 = v1093;
          v445 = v598;
        }
        v26 = sub_15C1230(*(_QWORD *)(v11 + 240), v445, v443, v448, 0, 1);
      }
LABEL_13:
      sub_15194E0(v11, v26, v13);
      ++*v9;
      goto LABEL_14;
    case 25:
      if ( *((_DWORD *)a3 + 2) != 3 )
        goto LABEL_64;
      v458 = 0;
      v13 = *a6;
      v64 = **a3 == 0;
      v459 = (*a3)[2];
      v1097 = **a3 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v459 )
          v458 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v459 - 1));
        v589 = sub_1519FE0(v11, v458);
        v590 = 0;
        v591 = v589;
        v592 = (*v10)[1];
        if ( (_DWORD)v592 )
        {
          v1090 = v591;
          v593 = sub_15217C0(v1099, (unsigned int)(v592 - 1));
          v591 = v1090;
          v590 = v593;
        }
        v26 = sub_15C24D0(*(_QWORD *)(v11 + 240), v590, v591, 0, 1);
      }
      else
      {
        if ( (_DWORD)v459 )
          v458 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v459 - 1));
        v460 = sub_1519FE0(v11, v458);
        v461 = 0;
        v462 = v460;
        v463 = (*v10)[1];
        if ( (_DWORD)v463 )
        {
          v1078 = v462;
          v464 = sub_15217C0(v1099, (unsigned int)(v463 - 1));
          v462 = v1078;
          v461 = v464;
        }
        v26 = sub_15C24D0(*(_QWORD *)(v11 + 240), v461, v462, 1, 1);
      }
      goto LABEL_13;
    case 26:
      if ( *((_DWORD *)a3 + 2) != 5 )
        goto LABEL_64;
      v297 = *a3;
      v13 = *a6;
      v1065 = 0;
      v64 = **a3 == 0;
      v298 = (*a3)[4];
      v1097 = *v297 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v298 )
        {
          v1065 = sub_15217C0(v1104, (unsigned int)(v298 - 1));
          v297 = *v10;
        }
        v577 = v297[3];
        v578 = 0;
        if ( (_DWORD)v577 )
          v578 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v577 - 1));
        v579 = sub_1519FE0(v11, v578);
        v580 = 0;
        v581 = v579;
        v582 = *v10;
        v583 = (*v10)[2];
        if ( (_DWORD)v583 )
        {
          v1039 = v581;
          v584 = sub_15217C0(v1099, (unsigned int)(v583 - 1));
          v581 = v1039;
          v580 = v584;
          v582 = *v10;
        }
        v307 = sub_15C2A60(*(_QWORD *)(v11 + 240), v582[1], v580, v581, v1065, 0, 1);
      }
      else
      {
        if ( (_DWORD)v298 )
        {
          v1065 = sub_15217C0(v1104, (unsigned int)(v298 - 1));
          v297 = *v10;
        }
        v299 = v297[3];
        v300 = 0;
        if ( (_DWORD)v299 )
          v300 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v299 - 1));
        v301 = sub_1519FE0(v11, v300);
        v302 = 0;
        v303 = v301;
        v304 = *v10;
        v305 = (*v10)[2];
        if ( (_DWORD)v305 )
        {
          v1009 = v303;
          v306 = sub_15217C0(v1099, (unsigned int)(v305 - 1));
          v303 = v1009;
          v302 = v306;
          v304 = *v10;
        }
        v307 = sub_15C2A60(*(_QWORD *)(v11 + 240), v304[1], v302, v303, v1065, 1, 1);
      }
      v26 = v307;
      goto LABEL_13;
    case 27:
      if ( (unsigned __int64)*((unsigned int *)a3 + 2) - 11 > 1 )
        goto LABEL_64;
      v176 = *a3;
      v177 = *(_BYTE *)*a3 & 1;
      v1097 = v177;
      v178 = (unsigned __int64)*v176 >> 1;
      v1057 = v178;
      if ( (_DWORD)v178 == 1 )
      {
        v13 = *a6;
        v984 = 0;
        v657 = v176[10];
        v1096 = v176[11];
        if ( v177 )
        {
          if ( (_DWORD)v657 )
          {
            v984 = sub_15217C0(v1104, (unsigned int)(v657 - 1));
            v176 = *v10;
          }
          v928 = v176[8] != 0;
          v894 = v176[7] != 0;
          v1044 = 0;
          v867 = sub_1521950(a2, &v1099, v176[6]);
          v659 = *v10;
          v660 = (*v10)[4];
          v831 = (*v10)[5];
          if ( (_DWORD)v660 )
          {
            v1044 = sub_15217C0(v1099, (unsigned int)(v660 - 1));
            v659 = *v10;
          }
          v849 = sub_1521990(&v1099, v659[3], v660, v658);
          v663 = 0;
          v664 = sub_1521990(&v1099, (*v10)[2], v661, v662);
          v665 = (*v10)[1];
          if ( (_DWORD)v665 )
          {
            v817 = v664;
            v666 = sub_15217C0(v1099, (unsigned int)(v665 - 1));
            v664 = v817;
            v663 = v666;
          }
          v667 = sub_15C2FB0(*(_QWORD *)(v11 + 240), v663, v664, v849, v1044, v831, v867, v894, v928, v984, v1096, 1, 1);
        }
        else
        {
          if ( (_DWORD)v657 )
          {
            v984 = sub_15217C0(v1104, (unsigned int)(v657 - 1));
            v176 = *v10;
          }
          v929 = v176[8] != 0;
          v895 = v176[7] != 0;
          v1045 = 0;
          v868 = sub_1521950(a2, &v1099, v176[6]);
          v681 = *v10;
          v682 = (*v10)[4];
          v832 = (*v10)[5];
          if ( (_DWORD)v682 )
          {
            v1045 = sub_15217C0(v1099, (unsigned int)(v682 - 1));
            v681 = *v10;
          }
          v850 = sub_1521990(&v1099, v681[3], v682, v680);
          v685 = 0;
          v686 = sub_1521990(&v1099, (*v10)[2], v683, v684);
          v687 = (*v10)[1];
          if ( (_DWORD)v687 )
          {
            v818 = v686;
            v688 = sub_15217C0(v1099, (unsigned int)(v687 - 1));
            v686 = v818;
            v685 = v688;
          }
          v667 = sub_15C2FB0(*(_QWORD *)(v11 + 240), v685, v686, v850, v1045, v832, v868, v895, v929, v984, v1096, 0, 1);
        }
        v26 = v667;
        goto LABEL_13;
      }
      if ( (_DWORD)v178 )
      {
LABEL_64:
        BYTE1(v1106[0]) = 1;
        v69 = "Invalid record";
        goto LABEL_65;
      }
      *((_BYTE *)a2 + 1010) = 1;
      v179 = *v10;
      v180 = (*v10)[9];
      if ( (_DWORD)v180 )
      {
        v1001 = sub_15217C0(v1104, (unsigned int)(v180 - 1));
        if ( *((_DWORD *)v10 + 2) > 0xBu )
        {
          v1057 = (*v10)[11];
          if ( (unsigned __int64)(*v10)[11] > 0xFFFFFFFF )
            goto LABEL_457;
        }
        if ( v1001 && *(_BYTE *)v1001 == 1 )
        {
          v181 = *(_QWORD *)(v1001 + 136);
          v182 = *(_BYTE *)(v181 + 16);
          if ( v182 == 3 )
          {
LABEL_161:
            v1001 = 0;
            v179 = *v10;
            goto LABEL_162;
          }
          if ( v182 != 13 )
          {
            v181 = 0;
            goto LABEL_161;
          }
          v1105.m128i_i64[0] = 16;
          v703 = *(_QWORD **)(v181 + 24);
          if ( *(_DWORD *)(v181 + 32) > 0x40u )
            v703 = (_QWORD *)*v703;
          v704 = a2[30];
          v181 = 0;
          v1105.m128i_i64[1] = (__int64)v703;
          v1106[0] = 159;
          v1001 = sub_15C4420(v704, &v1105, 3, 0, 1);
          v179 = *v10;
        }
        else
        {
          v179 = *v10;
          v181 = 0;
        }
      }
      else if ( *((_DWORD *)v10 + 2) <= 0xBu )
      {
        v1001 = 0;
        v181 = 0;
      }
      else
      {
        if ( (unsigned __int64)v179[11] > 0xFFFFFFFF )
          goto LABEL_457;
        v1057 = v179[11];
        v181 = 0;
        v1001 = 0;
      }
LABEL_162:
      v183 = v179[10];
      v906 = 0;
      if ( v1097 )
      {
        if ( (_DWORD)v183 )
        {
          v906 = sub_15217C0(v1099, (unsigned int)(v183 - 1));
          v179 = *v10;
        }
        v876 = v179[8] != 0;
        v856 = v179[7] != 0;
        v945 = 0;
        v836 = sub_1521950(a2, &v1099, v179[6]);
        v185 = *v10;
        v186 = (*v10)[4];
        v807 = (*v10)[5];
        if ( (_DWORD)v186 )
        {
          v945 = sub_15217C0(v1099, (unsigned int)(v186 - 1));
          v185 = *v10;
        }
        v823 = sub_1521990(&v1099, v185[3], v186, v184);
        v189 = 0;
        v190 = sub_1521990(&v1099, (*v10)[2], v187, v188);
        v191 = (*v10)[1];
        if ( (_DWORD)v191 )
        {
          v797 = v190;
          v192 = sub_15217C0(v1099, (unsigned int)(v191 - 1));
          v190 = v797;
          v189 = v192;
        }
        v193 = sub_15C2FB0(*(_QWORD *)(v11 + 240), v189, v190, v823, v945, v807, v836, v856, v876, v906, v1057, 1, 1);
      }
      else
      {
        if ( (_DWORD)v183 )
        {
          v906 = sub_15217C0(v1099, (unsigned int)(v183 - 1));
          v179 = *v10;
        }
        v898 = v179[8] != 0;
        v869 = v179[7] != 0;
        v987 = 0;
        v851 = sub_1521950(a2, &v1099, v179[6]);
        v694 = *v10;
        v695 = (*v10)[4];
        v819 = (*v10)[5];
        if ( (_DWORD)v695 )
        {
          v987 = sub_15217C0(v1099, (unsigned int)(v695 - 1));
          v694 = *v10;
        }
        v833 = sub_1521990(&v1099, v694[3], v695, v693);
        v698 = 0;
        v699 = sub_1521990(&v1099, (*v10)[2], v696, v697);
        v700 = (*v10)[1];
        if ( (_DWORD)v700 )
        {
          v806 = v699;
          v701 = sub_15217C0(v1099, (unsigned int)(v700 - 1));
          v699 = v806;
          v698 = v701;
        }
        v193 = sub_15C2FB0(*(_QWORD *)(v11 + 240), v698, v699, v833, v987, v819, v851, v869, v898, v906, v1057, 0, 1);
      }
      v194 = v193;
      v195 = v193;
      if ( v181 | v1001 )
      {
        v196 = v1001;
        if ( !v1001 )
          v196 = sub_15C4420(*(_QWORD *)(v11 + 240), 0, 0, 0, 1);
        v197 = sub_15C5570(*(_QWORD *)(v11 + 240), v194, v196, 1, 1);
        v195 = v197;
        if ( v181 )
        {
          v1058 = v197;
          sub_1626A90(v181, v197);
          v195 = v1058;
          if ( !v1001 )
            v195 = v194;
        }
      }
LABEL_176:
      sub_15194E0(v11, v195, *v9);
      ++*v9;
      goto LABEL_14;
    case 28:
      v149 = *((unsigned int *)a3 + 2);
      if ( (unsigned __int64)(v149 - 8) > 2 )
        goto LABEL_64;
      v150 = *a3;
      v151 = *(_BYTE *)*v10 & 1;
      v1097 = v151;
      if ( (*(_BYTE *)v150 & 2) != 0 )
      {
        if ( (unsigned __int64)v150[8] > 0xFFFFFFFF )
        {
LABEL_457:
          BYTE1(v1106[0]) = 1;
          v69 = "Alignment value is too large";
LABEL_65:
          v1105.m128i_i64[0] = (__int64)v69;
          LOBYTE(v1106[0]) = 3;
          sub_1514BE0(v8, (__int64)&v1105);
          return v8;
        }
        v1000 = v150[7];
        v156 = 40;
        v155 = 24;
        v905 = v150[8];
        v158 = 48;
        v941 = 32;
        v873 = 16;
        v854 = 8;
      }
      else
      {
        v905 = 0;
        v152 = (v149 != 8) + 7LL;
        v153 = v150[v152];
        v154 = 8 * v152;
        v155 = 8 * v152 - 32;
        v156 = 8 * v152 - 16;
        v1000 = v153;
        v854 = v154 - 48;
        v873 = v154 - 40;
        v157 = v154 - 24;
        v158 = v154 - 8;
        v941 = v157;
      }
      v1056 = *a6;
      v835 = *(__int64 *)((char *)v150 + v158);
      v159 = *(__int64 *)((char *)v150 + v156);
      v160 = 0;
      if ( v151 )
      {
        if ( (_DWORD)v159 )
          v160 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v159 - 1));
        v161 = sub_1519FE0(v11, v160);
        v162 = 0;
        v163 = v161;
        v164 = (unsigned __int64 *)*v10;
        v165 = *(__int64 *)((char *)*v10 + v155);
        v166 = *(__int64 *)((char *)*v10 + v941);
        if ( (_DWORD)v165 )
        {
          v821 = *(__int64 *)((char *)*v10 + v941);
          v942 = v163;
          v167 = sub_15217C0(v1099, (unsigned int)(v165 - 1));
          LODWORD(v166) = v821;
          v163 = v942;
          v162 = v167;
          v164 = (unsigned __int64 *)*v10;
        }
        v168 = 0;
        v169 = *(unsigned __int64 *)((char *)v164 + v873);
        if ( (_DWORD)v169 )
        {
          v822 = v166;
          v874 = v163;
          v943 = v162;
          v170 = sub_15217C0(v1099, (unsigned int)(v169 - 1));
          LODWORD(v166) = v822;
          v163 = v874;
          v168 = v170;
          v162 = v943;
          v164 = (unsigned __int64 *)*v10;
        }
        v171 = 0;
        v172 = *(unsigned __int64 *)((char *)v164 + v854);
        if ( (_DWORD)v172 )
        {
          v855 = v166;
          v875 = v163;
          v944 = v162;
          v173 = sub_15217C0(v1099, (unsigned int)(v172 - 1));
          LODWORD(v166) = v855;
          v163 = v875;
          v162 = v944;
          v171 = v173;
        }
        v174 = sub_15C37C0(*(_QWORD *)(v11 + 240), v171, v168, v162, v166, v163, v835, v1000, v905, 1, 1);
      }
      else
      {
        if ( (_DWORD)v159 )
          v160 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v159 - 1));
        v643 = sub_1519FE0(v11, v160);
        v644 = 0;
        v645 = v643;
        v646 = *v10;
        v647 = *(__int64 *)((char *)*v10 + v155);
        v648 = *(__int64 *)((char *)*v10 + v941);
        if ( (_DWORD)v647 )
        {
          v829 = *(__int64 *)((char *)*v10 + v941);
          v981 = v645;
          v649 = sub_15217C0(v1099, (unsigned int)(v647 - 1));
          LODWORD(v648) = v829;
          v645 = v981;
          v644 = v649;
          v646 = *v10;
        }
        v650 = 0;
        v651 = *(__int64 *)((char *)v646 + v873);
        if ( (_DWORD)v651 )
        {
          v830 = v648;
          v892 = v645;
          v982 = v644;
          v652 = sub_15217C0(v1099, (unsigned int)(v651 - 1));
          LODWORD(v648) = v830;
          v645 = v892;
          v650 = v652;
          v644 = v982;
          v646 = *v10;
        }
        v653 = 0;
        v654 = *(__int64 *)((char *)v646 + v854);
        if ( (_DWORD)v654 )
        {
          v866 = v648;
          v893 = v645;
          v983 = v644;
          v655 = sub_15217C0(v1099, (unsigned int)(v654 - 1));
          LODWORD(v648) = v866;
          v645 = v893;
          v644 = v983;
          v653 = v655;
        }
        v174 = sub_15C37C0(*(_QWORD *)(v11 + 240), v653, v650, v644, v648, v645, v835, v1000, v905, 0, 1);
      }
      v175 = v174;
LABEL_152:
      sub_15194E0(v11, v175, v1056);
      ++*v9;
      goto LABEL_14;
    case 29:
      v144 = *((unsigned int *)a3 + 2);
      if ( !*((_DWORD *)a3 + 2) )
        goto LABEL_64;
      v145 = *a3;
      v146 = *v145++;
      v1097 = v146 & 1;
      v147 = *(v145 - 1);
      v1101 = (unsigned __int64)v145;
      v1102 = v144 - 1;
      v1105.m128i_i64[1] = 0x600000000LL;
      v1105.m128i_i64[0] = (__int64)v1106;
      sub_1515740(&v1100, (__int64)a2, v147 >> 1, (__int64)&v1101, (__int64)&v1105);
      if ( (v1100 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
        v34 = *v9;
        v148 = a2[30];
        if ( v1097 )
        {
          v36 = sub_15C4420(v148, v1101, v1102, 1, 1);
          goto LABEL_27;
        }
        v483 = sub_15C4420(v148, v1101, v1102, 0, 1);
        goto LABEL_460;
      }
      v56 = (_QWORD *)v1105.m128i_i64[0];
      *v8 = v1100 & 0xFFFFFFFFFFFFFFFELL | 1;
      if ( v56 != v1106 )
        goto LABEL_47;
      return v8;
    case 30:
      if ( *((_DWORD *)a3 + 2) != 8 )
        goto LABEL_64;
      v127 = 0;
      v13 = *a6;
      v64 = **a3 == 0;
      v128 = (*a3)[7];
      v1097 = **a3 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v128 )
          v127 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v128 - 1));
        v543 = sub_1519FE0(v11, v127);
        v544 = 0;
        v1087 = v543;
        v545 = *v10;
        v546 = (*v10)[5];
        v1034 = (*v10)[6];
        if ( (_DWORD)v546 )
        {
          v544 = sub_15217C0(v1099, (unsigned int)(v546 - 1));
          v545 = *v10;
        }
        v547 = v545[4];
        v548 = 0;
        if ( (_DWORD)v547 )
        {
          v970 = v544;
          v549 = sub_15217C0(v1099, (unsigned int)(v547 - 1));
          v544 = v970;
          v548 = v549;
          v545 = *v10;
        }
        v550 = v545[2];
        v551 = v545[3];
        v552 = 0;
        if ( (_DWORD)v550 )
        {
          v886 = v545[3];
          v920 = v544;
          v971 = v548;
          v553 = sub_15217C0(v1099, (unsigned int)(v550 - 1));
          LODWORD(v551) = v886;
          v544 = v920;
          v552 = v553;
          v548 = v971;
          v545 = *v10;
        }
        v554 = v545[1];
        v555 = 0;
        if ( (_DWORD)v554 )
        {
          v863 = v551;
          v887 = v544;
          v921 = v548;
          v972 = v552;
          v556 = sub_15217C0(v1099, (unsigned int)(v554 - 1));
          LODWORD(v551) = v863;
          v544 = v887;
          v548 = v921;
          v552 = v972;
          v555 = v556;
        }
        v143 = sub_15C5B60(*(_QWORD *)(v11 + 240), v555, v552, v551, v548, v544, v1034, v1087, 0, 1);
      }
      else
      {
        if ( (_DWORD)v128 )
          v127 = (_BYTE *)sub_15217C0(v1104, (unsigned int)(v128 - 1));
        v129 = sub_1519FE0(v11, v127);
        v130 = 0;
        v1055 = v129;
        v131 = *v10;
        v132 = (*v10)[5];
        v999 = (*v10)[6];
        if ( (_DWORD)v132 )
        {
          v130 = sub_15217C0(v1099, (unsigned int)(v132 - 1));
          v131 = *v10;
        }
        v133 = v131[4];
        v134 = 0;
        if ( (_DWORD)v133 )
        {
          v938 = v130;
          v135 = sub_15217C0(v1099, (unsigned int)(v133 - 1));
          v130 = v938;
          v134 = v135;
          v131 = *v10;
        }
        v136 = v131[2];
        v137 = v131[3];
        v138 = 0;
        if ( (_DWORD)v136 )
        {
          v871 = v131[3];
          v903 = v130;
          v939 = v134;
          v139 = sub_15217C0(v1099, (unsigned int)(v136 - 1));
          LODWORD(v137) = v871;
          v130 = v903;
          v138 = v139;
          v134 = v939;
          v131 = *v10;
        }
        v140 = v131[1];
        v141 = 0;
        if ( (_DWORD)v140 )
        {
          v853 = v137;
          v872 = v130;
          v904 = v134;
          v940 = v138;
          v142 = sub_15217C0(v1099, (unsigned int)(v140 - 1));
          LODWORD(v137) = v853;
          v130 = v872;
          v134 = v904;
          v138 = v940;
          v141 = v142;
        }
        v143 = sub_15C5B60(*(_QWORD *)(v11 + 240), v141, v138, v137, v134, v130, v999, v1055, 1, 1);
      }
      v26 = v143;
      goto LABEL_13;
    case 31:
      v114 = *((unsigned int *)a3 + 2);
      if ( (unsigned __int64)(v114 - 6) > 1 )
        goto LABEL_64;
      v115 = *a3;
      v13 = *a6;
      v64 = **a3 == 0;
      v116 = (*a3)[5];
      v1097 = **v10 != 0;
      if ( v64 )
      {
        v1079 = 0;
        if ( (_DWORD)v116 )
        {
          v1025 = v114;
          v489 = sub_15217C0(v1104, (unsigned int)(v116 - 1));
          v114 = v1025;
          v1079 = v489;
          v115 = *v10;
        }
        v1026 = 0;
        v964 = 0;
        if ( v114 == 7 )
        {
          v668 = v115[6];
          v1026 = *((_DWORD *)v115 + 8);
          if ( (_DWORD)v668 )
          {
            v964 = sub_15217C0(v1099, (unsigned int)(v668 - 1));
            v115 = *v10;
          }
        }
        v490 = v115[3];
        v491 = 0;
        if ( (_DWORD)v490 )
          v491 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v490 - 1));
        v492 = sub_1519FE0(v11, v491);
        v493 = 0;
        v494 = v492;
        v495 = *v10;
        v496 = (*v10)[2];
        if ( (_DWORD)v496 )
        {
          v918 = v494;
          v497 = sub_15217C0(v1099, (unsigned int)(v496 - 1));
          v494 = v918;
          v493 = v497;
          v495 = *v10;
        }
        v126 = sub_15C6270(*(_QWORD *)(v11 + 240), v495[1], v493, v494, v964, v1026, v1079, 0, 1);
      }
      else
      {
        v1054 = 0;
        if ( (_DWORD)v116 )
        {
          v997 = v114;
          v117 = sub_15217C0(v1104, (unsigned int)(v116 - 1));
          v114 = v997;
          v1054 = v117;
          v115 = *v10;
        }
        v998 = 0;
        v937 = 0;
        if ( v114 == 7 )
        {
          v656 = v115[6];
          v998 = *((_DWORD *)v115 + 8);
          if ( (_DWORD)v656 )
          {
            v937 = sub_15217C0(v1099, (unsigned int)(v656 - 1));
            v115 = *v10;
          }
        }
        v118 = v115[3];
        v119 = 0;
        if ( (_DWORD)v118 )
          v119 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v118 - 1));
        v120 = sub_1519FE0(v11, v119);
        v121 = 0;
        v122 = v120;
        v123 = *v10;
        v124 = (*v10)[2];
        if ( (_DWORD)v124 )
        {
          v902 = v122;
          v125 = sub_15217C0(v1099, (unsigned int)(v124 - 1));
          v122 = v902;
          v121 = v125;
          v123 = *v10;
        }
        v126 = sub_15C6270(*(_QWORD *)(v11 + 240), v123[1], v121, v122, v937, v998, v1054, 1, 1);
      }
      v26 = v126;
      goto LABEL_13;
    case 32:
      if ( *((_DWORD *)a3 + 2) != 6 )
        goto LABEL_64;
      v100 = *a3;
      v13 = *a6;
      v1053 = 0;
      v64 = **a3 == 0;
      v101 = (*a3)[5];
      v1097 = *v100 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v101 )
        {
          v1053 = sub_15217C0(v1104, (unsigned int)(v101 - 1));
          v100 = *v10;
        }
        v532 = v100[4];
        v533 = 0;
        if ( (_DWORD)v532 )
        {
          v533 = sub_15217C0(v1099, (unsigned int)(v532 - 1));
          v100 = *v10;
        }
        v534 = v100[3];
        v535 = 0;
        if ( (_DWORD)v534 )
        {
          v1031 = v533;
          v536 = sub_15217C0(v1099, (unsigned int)(v534 - 1));
          v533 = v1031;
          v535 = v536;
          v100 = *v10;
        }
        v537 = v100[2];
        v538 = 0;
        if ( (_DWORD)v537 )
        {
          v968 = v533;
          v1032 = v535;
          v539 = sub_15217C0(v1099, (unsigned int)(v537 - 1));
          v533 = v968;
          v535 = v1032;
          v538 = v539;
          v100 = *v10;
        }
        v540 = v100[1];
        v541 = 0;
        if ( (_DWORD)v540 )
        {
          v919 = v533;
          v969 = v535;
          v1033 = v538;
          v542 = sub_15217C0(v1099, (unsigned int)(v540 - 1));
          v533 = v919;
          v535 = v969;
          v538 = v1033;
          v541 = v542;
        }
        v113 = sub_15C1EB0(*(_QWORD *)(v11 + 240), v541, v538, v535, v533, v1053, 0, 1);
      }
      else
      {
        if ( (_DWORD)v101 )
        {
          v1053 = sub_15217C0(v1104, (unsigned int)(v101 - 1));
          v100 = *v10;
        }
        v102 = v100[4];
        v103 = 0;
        if ( (_DWORD)v102 )
        {
          v103 = sub_15217C0(v1099, (unsigned int)(v102 - 1));
          v100 = *v10;
        }
        v104 = v100[3];
        v105 = 0;
        if ( (_DWORD)v104 )
        {
          v994 = v103;
          v106 = sub_15217C0(v1099, (unsigned int)(v104 - 1));
          v103 = v994;
          v105 = v106;
          v100 = *v10;
        }
        v107 = v100[2];
        v108 = 0;
        if ( (_DWORD)v107 )
        {
          v935 = v103;
          v995 = v105;
          v109 = sub_15217C0(v1099, (unsigned int)(v107 - 1));
          v103 = v935;
          v105 = v995;
          v108 = v109;
          v100 = *v10;
        }
        v110 = v100[1];
        v111 = 0;
        if ( (_DWORD)v110 )
        {
          v901 = v103;
          v936 = v105;
          v996 = v108;
          v112 = sub_15217C0(v1099, (unsigned int)(v110 - 1));
          v103 = v901;
          v105 = v936;
          v108 = v996;
          v111 = v112;
        }
        v113 = sub_15C1EB0(*(_QWORD *)(v11 + 240), v111, v108, v105, v103, v1053, 1, 1);
      }
      v26 = v113;
      goto LABEL_13;
    case 33:
      if ( *((_DWORD *)a3 + 2) != 5 )
        goto LABEL_64;
      v94 = *a3;
      v95 = 0;
      v13 = *a6;
      v64 = **a3 == 0;
      v96 = (*a3)[4];
      v1097 = **v10 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v96 )
        {
          v95 = sub_15217C0(v1104, (unsigned int)(v96 - 1));
          v94 = *v10;
        }
        v526 = v94[3];
        v527 = 0;
        if ( (_DWORD)v526 )
        {
          v1085 = v95;
          v528 = sub_15217C0(v1099, (unsigned int)(v526 - 1));
          v95 = v1085;
          v527 = v528;
          v94 = *v10;
        }
        v26 = sub_15C68B0(a2[30], v94[1], v94[2], v527, v95, 0, 1);
      }
      else
      {
        if ( (_DWORD)v96 )
        {
          v95 = sub_15217C0(v1104, (unsigned int)(v96 - 1));
          v94 = *v10;
        }
        v97 = v94[3];
        v98 = 0;
        if ( (_DWORD)v97 )
        {
          v1052 = v95;
          v99 = sub_15217C0(v1099, (unsigned int)(v97 - 1));
          v95 = v1052;
          v98 = v99;
          v94 = *v10;
        }
        v26 = sub_15C68B0(a2[30], v94[1], v94[2], v98, v95, 1, 1);
      }
      goto LABEL_13;
    case 34:
      if ( *((_DWORD *)a3 + 2) != 5 )
        goto LABEL_64;
      v88 = *a3;
      v89 = 0;
      v13 = *a6;
      v64 = **a3 == 0;
      v90 = (*a3)[4];
      v1097 = **v10 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v90 )
        {
          v89 = sub_15217C0(v1104, (unsigned int)(v90 - 1));
          v88 = *v10;
        }
        v529 = v88[3];
        v530 = 0;
        if ( (_DWORD)v529 )
        {
          v1086 = v89;
          v531 = sub_15217C0(v1099, (unsigned int)(v529 - 1));
          v89 = v1086;
          v530 = v531;
          v88 = *v10;
        }
        v26 = sub_15C6E80(a2[30], v88[1], v88[2], v530, v89, 0, 1);
      }
      else
      {
        if ( (_DWORD)v90 )
        {
          v89 = sub_15217C0(v1104, (unsigned int)(v90 - 1));
          v88 = *v10;
        }
        v91 = v88[3];
        v92 = 0;
        if ( (_DWORD)v91 )
        {
          v1051 = v89;
          v93 = sub_15217C0(v1099, (unsigned int)(v91 - 1));
          v89 = v1051;
          v92 = v93;
          v88 = *v10;
        }
        v26 = sub_15C6E80(a2[30], v88[1], v88[2], v92, v89, 1, 1);
      }
      goto LABEL_13;
    case 35:
      v328 = *a3;
      v329 = *((unsigned int *)a3 + 2);
      v1105.m128i_i64[0] = (__int64)a2;
      v1105.m128i_i64[1] = (__int64)a6;
      sub_1515D60(
        (__int64 *)&v1101,
        (__int64)a2,
        v328,
        v329,
        a7,
        a8,
        (void (__fastcall *)(__int64, __int64, _QWORD))sub_1519750,
        (__int64)&v1105);
      v68 = v1101 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v1101 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_62;
      goto LABEL_14;
    case 36:
      v323 = *((_DWORD *)a3 + 2);
      if ( (v323 & 1) == 0 )
        goto LABEL_64;
      v324 = (_QWORD *)a2[28];
      v325 = **a3;
      if ( -1431655765 * (unsigned int)((__int64)(v324[1] - *v324) >> 3) <= (unsigned int)v325 )
        goto LABEL_64;
      v326 = *(_QWORD *)(*v324 + 24LL * (unsigned int)v325 + 16);
      v327 = *(_BYTE *)(v326 + 16);
      if ( v327 != 3 && v327 )
        goto LABEL_14;
      sub_1518010(v1105.m128i_i64, (__int64)a2, v326, (__int64)(*v10 + 1), v323 - 1);
      v68 = v1105.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v1105.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
        goto LABEL_14;
LABEL_62:
      a1->m128i_i64[0] = v68 | 1;
      return v8;
    case 37:
      if ( *((_DWORD *)a3 + 2) != 3 )
        goto LABEL_64;
      v318 = (*a3)[2];
      v1097 = **a3 != 0;
      if ( !(_DWORD)v318 || (v319 = sub_15217C0(v1104, (unsigned int)(v318 - 1))) == 0 )
        v319 = sub_15C4420(a2[30], 0, 0, 0, 1);
      v320 = 0;
      v13 = *v9;
      v321 = (*v10)[1];
      if ( v1097 )
      {
        if ( (_DWORD)v321 )
        {
          v1067 = v319;
          v322 = sub_15217C0(v1099, (unsigned int)(v321 - 1));
          v319 = v1067;
          v320 = v322;
        }
        v26 = sub_15C5570(*(_QWORD *)(v11 + 240), v320, v319, 1, 1);
      }
      else
      {
        if ( (_DWORD)v321 )
        {
          v1095 = v319;
          v620 = sub_15217C0(v1099, (unsigned int)(v321 - 1));
          v319 = v1095;
          v320 = v620;
        }
        v26 = sub_15C5570(*(_QWORD *)(v11 + 240), v320, v319, 0, 1);
      }
      goto LABEL_13;
    case 40:
      if ( *((_DWORD *)a3 + 2) != 5 )
        goto LABEL_64;
      v308 = *a3;
      v13 = *a6;
      v309 = 0;
      v310 = (*a3)[3];
      v1066 = (*a3)[4];
      v1097 = *(_BYTE *)*a3 & 1;
      if ( v1097 )
      {
        if ( (_DWORD)v310 )
        {
          v309 = sub_15217C0(v1104, (unsigned int)(v310 - 1));
          v308 = *v10;
        }
        v311 = v308[2];
        v312 = 0;
        if ( (_DWORD)v311 )
        {
          v1010 = v309;
          v313 = sub_15217C0(v1099, (unsigned int)(v311 - 1));
          v309 = v1010;
          v312 = v313;
          v308 = *v10;
        }
        v314 = v308[1];
        v315 = 0;
        if ( (_DWORD)v314 )
        {
          v951 = v309;
          v1011 = v312;
          v316 = sub_15217C0(v1099, (unsigned int)(v314 - 1));
          v309 = v951;
          v312 = v1011;
          v315 = v316;
        }
        v317 = sub_15C3EA0(*(_QWORD *)(v11 + 240), v315, v312, v309, v1066, 1, 1);
      }
      else
      {
        if ( (_DWORD)v310 )
        {
          v309 = sub_15217C0(v1104, (unsigned int)(v310 - 1));
          v308 = *v10;
        }
        v571 = v308[2];
        v572 = 0;
        if ( (_DWORD)v571 )
        {
          v1037 = v309;
          v573 = sub_15217C0(v1099, (unsigned int)(v571 - 1));
          v309 = v1037;
          v572 = v573;
          v308 = *v10;
        }
        v574 = v308[1];
        v575 = 0;
        if ( (_DWORD)v574 )
        {
          v974 = v309;
          v1038 = v572;
          v576 = sub_15217C0(v1099, (unsigned int)(v574 - 1));
          v309 = v974;
          v572 = v1038;
          v575 = v576;
        }
        v317 = sub_15C3EA0(*(_QWORD *)(v11 + 240), v575, v572, v309, v1066, 0, 1);
      }
      v26 = v317;
      goto LABEL_13;
    case 41:
      if ( *((_DWORD *)a3 + 2) != 8 )
        goto LABEL_64;
      v354 = *a3;
      v355 = 0;
      v13 = *a6;
      v356 = (*a3)[7];
      v64 = **a3 == 0;
      v357 = (*a3)[4];
      v1097 = **v10 != 0;
      v953 = v354[6];
      v1069 = v354[5];
      if ( v64 )
      {
        if ( (_DWORD)v357 )
        {
          v355 = sub_15217C0(v1104, (unsigned int)(v357 - 1));
          v354 = *v10;
        }
        v565 = v354[3];
        v566 = 0;
        if ( (_DWORD)v565 )
        {
          v922 = v355;
          v567 = sub_15217C0(v1099, (unsigned int)(v565 - 1));
          v355 = v922;
          v566 = v567;
          v354 = *v10;
        }
        v568 = v354[2];
        v569 = 0;
        if ( (_DWORD)v568 )
        {
          v888 = v355;
          v923 = v566;
          v570 = sub_15217C0(v1099, (unsigned int)(v568 - 1));
          v355 = v888;
          v566 = v923;
          v569 = v570;
          v354 = *v10;
        }
        v364 = sub_15BCE80(a2[30], v354[1], v569, v566, v355, v1069, v953, v356, 0, 1);
      }
      else
      {
        if ( (_DWORD)v357 )
        {
          v355 = sub_15217C0(v1104, (unsigned int)(v357 - 1));
          v354 = *v10;
        }
        v358 = v354[3];
        v359 = 0;
        if ( (_DWORD)v358 )
        {
          v911 = v355;
          v360 = sub_15217C0(v1099, (unsigned int)(v358 - 1));
          v355 = v911;
          v359 = v360;
          v354 = *v10;
        }
        v361 = v354[2];
        v362 = 0;
        if ( (_DWORD)v361 )
        {
          v880 = v355;
          v912 = v359;
          v363 = sub_15217C0(v1099, (unsigned int)(v361 - 1));
          v355 = v880;
          v359 = v912;
          v362 = v363;
          v354 = *v10;
        }
        v364 = sub_15BCE80(a2[30], v354[1], v362, v359, v355, v1069, v953, v356, 1, 1);
      }
      v26 = v364;
      goto LABEL_13;
    case 42:
      if ( *((_DWORD *)a3 + 2) != 12 )
        goto LABEL_64;
      v1068 = 0;
      v330 = *a3;
      v331 = (*a3)[1];
      v1097 = **a3;
      v332 = v330[2];
      v1097 &= 1u;
      v1012 = v331;
      if ( (_DWORD)v332 )
      {
        v1068 = sub_15217C0(v1104, (unsigned int)(v332 - 1));
        v330 = *v10;
      }
      v333 = v330[3];
      v334 = 0;
      if ( (_DWORD)v333 )
      {
        v334 = sub_15217C0(v1099, (unsigned int)(v333 - 1));
        v330 = *v10;
      }
      v335 = v330[4];
      v336 = v330[5];
      v337 = 0;
      v952 = v335;
      if ( (_DWORD)v336 )
        v337 = (_BYTE *)sub_15217C0(v1099, (unsigned int)(v336 - 1));
      v338 = sub_1519FE0(v11, v337);
      v339 = *v10;
      v340 = (*v10)[8];
      if ( v340 > 0xFFFFFFFF )
        goto LABEL_457;
      v341 = 0;
      v910 = v339[7];
      v342 = v339[10];
      v343 = v339[6];
      v879 = v342;
      if ( (_DWORD)v343 )
      {
        v839 = v338;
        v859 = (*v10)[8];
        v344 = sub_15217C0(v1099, (unsigned int)(v343 - 1));
        v338 = v839;
        LODWORD(v340) = v859;
        v341 = (_BYTE *)v344;
      }
      v840 = v338;
      v860 = v340;
      v345 = sub_1519FE0(v11, v341);
      v346 = v860;
      v347 = v840;
      v348 = v345;
      v349 = (*v10)[11];
      v350 = (*v10)[9];
      v351 = 0;
      if ( (_DWORD)v349 )
      {
        v825 = v348;
        v351 = sub_15217C0(v1099, (unsigned int)(v349 - 1));
        v348 = v825;
        v347 = v840;
        v346 = v860;
      }
      v352 = *(_QWORD *)(v11 + 240);
      if ( v1097 )
        v353 = sub_15BE7F0(v352, v1012, v1068, v334, v952, v347, v348, v910, v346, v350, v879, v351, 1, 1);
      else
        v353 = sub_15BE7F0(v352, v1012, v1068, v334, v952, v347, v348, v910, v346, v350, v879, v351, 0, 1);
      v195 = v353;
      goto LABEL_176;
    case 43:
      if ( *((_DWORD *)a3 + 2) != 8 )
        goto LABEL_64;
      v365 = *a3;
      v13 = *a6;
      v1070 = 0;
      v64 = **a3 == 0;
      v366 = (*a3)[7];
      v1097 = *v365 != 0;
      if ( v64 )
      {
        if ( (_DWORD)v366 )
        {
          v1070 = sub_15217C0(v1104, (unsigned int)(v366 - 1));
          v365 = *v10;
        }
        v557 = v365[6];
        v558 = 0;
        if ( (_DWORD)v557 )
        {
          v558 = sub_15217C0(v1099, (unsigned int)(v557 - 1));
          v365 = *v10;
        }
        v559 = v365[5];
        v560 = 0;
        if ( (_DWORD)v559 )
        {
          v1035 = v558;
          v561 = sub_15217C0(v1099, (unsigned int)(v559 - 1));
          v558 = v1035;
          v560 = v561;
          v365 = *v10;
        }
        v562 = v365[4];
        v563 = 0;
        if ( (_DWORD)v562 )
        {
          v973 = v558;
          v1036 = v560;
          v564 = sub_15217C0(v1099, (unsigned int)(v562 - 1));
          v558 = v973;
          v560 = v1036;
          v563 = v564;
          v365 = *v10;
        }
        v375 = sub_15BBC00(a2[30], v365[1], v365[2], v365[3] != 0, v563, v560, v558, v1070, 0, 1);
      }
      else
      {
        if ( (_DWORD)v366 )
        {
          v1070 = sub_15217C0(v1104, (unsigned int)(v366 - 1));
          v365 = *v10;
        }
        v367 = v365[6];
        v368 = 0;
        if ( (_DWORD)v367 )
        {
          v368 = sub_15217C0(v1099, (unsigned int)(v367 - 1));
          v365 = *v10;
        }
        v369 = v365[5];
        v370 = 0;
        if ( (_DWORD)v369 )
        {
          v1013 = v368;
          v371 = sub_15217C0(v1099, (unsigned int)(v369 - 1));
          v368 = v1013;
          v370 = v371;
          v365 = *v10;
        }
        v372 = v365[4];
        v373 = 0;
        if ( (_DWORD)v372 )
        {
          v954 = v368;
          v1014 = v370;
          v374 = sub_15217C0(v1099, (unsigned int)(v372 - 1));
          v368 = v954;
          v370 = v1014;
          v373 = v374;
          v365 = *v10;
        }
        v375 = sub_15BBC00(a2[30], v365[1], v365[2], v365[3] != 0, v373, v370, v368, v1070, 1, 1);
      }
      v26 = v375;
      goto LABEL_13;
    case 44:
      v12 = *a3;
      v13 = *a6;
      v14 = 0;
      v15 = (*a3)[4];
      v1048 = (*a3)[5];
      v1097 = *(_BYTE *)*a3 & 1;
      if ( v1097 )
      {
        if ( (_DWORD)v15 )
        {
          v14 = sub_15217C0(v1104, (unsigned int)(v15 - 1));
          v12 = *v10;
        }
        v16 = v12[3];
        v17 = 0;
        if ( (_DWORD)v16 )
        {
          v990 = v14;
          v18 = sub_15217C0(v1099, (unsigned int)(v16 - 1));
          v14 = v990;
          v17 = v18;
          v12 = *v10;
        }
        v19 = v12[2];
        v20 = 0;
        if ( (_DWORD)v19 )
        {
          v932 = v14;
          v991 = v17;
          v21 = sub_15217C0(v1099, (unsigned int)(v19 - 1));
          v14 = v932;
          v17 = v991;
          v20 = v21;
          v12 = *v10;
        }
        v22 = v12[1];
        v23 = 0;
        if ( (_DWORD)v22 )
        {
          v900 = v14;
          v933 = v17;
          v992 = v20;
          v24 = sub_15217C0(v1099, (unsigned int)(v22 - 1));
          v14 = v900;
          v17 = v933;
          v20 = v992;
          v23 = v24;
        }
        v25 = sub_15C1830(*(_QWORD *)(v11 + 240), v23, v20, v17, v14, v1048, 1, 1);
      }
      else
      {
        if ( (_DWORD)v15 )
        {
          v14 = sub_15217C0(v1104, (unsigned int)(v15 - 1));
          v12 = *v10;
        }
        v474 = v12[3];
        v475 = 0;
        if ( (_DWORD)v474 )
        {
          v1022 = v14;
          v476 = sub_15217C0(v1099, (unsigned int)(v474 - 1));
          v14 = v1022;
          v475 = v476;
          v12 = *v10;
        }
        v477 = v12[2];
        v478 = 0;
        if ( (_DWORD)v477 )
        {
          v962 = v14;
          v1023 = v475;
          v479 = sub_15217C0(v1099, (unsigned int)(v477 - 1));
          v14 = v962;
          v475 = v1023;
          v478 = v479;
          v12 = *v10;
        }
        v480 = v12[1];
        v481 = 0;
        if ( (_DWORD)v480 )
        {
          v917 = v14;
          v963 = v475;
          v1024 = v478;
          v482 = sub_15217C0(v1099, (unsigned int)(v480 - 1));
          v14 = v917;
          v475 = v963;
          v478 = v1024;
          v481 = v482;
        }
        v25 = sub_15C1830(*(_QWORD *)(v11 + 240), v481, v478, v475, v14, v1048, 0, 1);
      }
      v26 = v25;
      goto LABEL_13;
    default:
      goto LABEL_14;
  }
}
