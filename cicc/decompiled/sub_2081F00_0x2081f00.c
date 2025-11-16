// Function: sub_2081F00
// Address: 0x2081f00
//
char *__fastcall sub_2081F00(
        __int64 *a1,
        __int64 *a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v12; // rax
  int v13; // edx
  __m128i *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdx
  char *v17; // r12
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // r10
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 (*v28)(); // rax
  __int32 v29; // edx
  __m128i v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rdi
  __int64 v33; // rax
  unsigned __int8 v34; // al
  __int64 v35; // rax
  __int32 v36; // edx
  int v37; // edx
  unsigned __int64 v38; // r12
  __int64 v39; // rdx
  unsigned __int64 v40; // rdx
  unsigned __int8 *v41; // rax
  int v42; // r8d
  int v43; // r9d
  int v44; // r8d
  int v45; // r9d
  __int64 v46; // rbx
  __int64 v47; // rdx
  __int64 v48; // rcx
  int v49; // r8d
  int v50; // r9d
  unsigned __int32 v51; // eax
  __int64 *v52; // rax
  int v53; // edx
  __int64 v54; // rdx
  __int64 v55; // rcx
  int v56; // r8d
  int v57; // r9d
  __int128 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  int v61; // r8d
  int v62; // r9d
  __int64 v63; // rdx
  __int64 v64; // rcx
  int v65; // r8d
  int v66; // r9d
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rax
  int v71; // edx
  _QWORD *v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rax
  int v75; // edx
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // r15
  __int64 v79; // rdx
  __int64 v80; // rcx
  int v81; // r8d
  int v82; // r9d
  __int64 v83; // rdx
  __int64 v84; // rcx
  int v85; // r8d
  int v86; // r9d
  __int128 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  int v90; // r8d
  int v91; // r9d
  __int64 v92; // rdx
  __int64 v93; // rcx
  int v94; // r8d
  int v95; // r9d
  __int64 v96; // rcx
  __int64 *v97; // rax
  unsigned int v98; // edx
  __int16 v99; // cx
  __int64 v100; // rsi
  _QWORD *v101; // r11
  __int64 v102; // rdx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rax
  int v106; // edx
  __m128i *v107; // rcx
  __m128 *v108; // rax
  unsigned int v109; // ebx
  __int64 v110; // rbx
  char *v111; // r13
  unsigned __int64 v112; // rax
  char *v113; // rbx
  __int64 v114; // rcx
  __int64 v115; // rdi
  const __m128i *v116; // rax
  __int32 v117; // esi
  __m128i v118; // xmm3
  __m128i *v119; // rdx
  __int64 *v120; // rbx
  __int64 v121; // r12
  unsigned int v122; // r14d
  __int64 v123; // rdx
  __int64 v124; // rcx
  int v125; // r8d
  int v126; // r9d
  const __m128i *v127; // rsi
  __int64 v128; // rdx
  __int64 v129; // rcx
  int v130; // r8d
  int v131; // r9d
  _QWORD *v132; // r13
  __int64 v133; // r9
  __int64 v134; // rbx
  __int64 *v135; // rax
  _OWORD *v136; // rdi
  __m128i *v137; // rdi
  int v138; // edx
  int v139; // eax
  int v140; // r10d
  __int64 *v141; // rbx
  __int64 *v142; // r12
  __int64 v143; // rdx
  __int64 v144; // r13
  __int64 *v145; // rax
  unsigned int v146; // edx
  const void ***v147; // rdx
  __int64 v148; // rax
  int v149; // edx
  __int64 v150; // rbx
  __int64 *v151; // rax
  __int64 *v152; // rax
  int v153; // edx
  int v154; // r12d
  __int64 *v155; // rbx
  __int64 *v156; // rax
  __int64 *v157; // rbx
  __int64 *v158; // r12
  __int64 v159; // rdx
  __int64 v160; // r13
  __int64 *v161; // rsi
  __int64 v162; // rdx
  __int64 v163; // rcx
  __int64 v164; // r8
  int v165; // r9d
  __int64 *v166; // rax
  __int16 *v167; // rdx
  __int64 *v168; // rax
  int v169; // edx
  __int64 v170; // r13
  __int64 v171; // rax
  int v172; // ecx
  __int64 (__fastcall *v173)(__int64, unsigned __int8); // rax
  __int64 v174; // rdx
  int v175; // eax
  __int64 *v176; // r13
  int v177; // ebx
  int v178; // edx
  __int64 v179; // r8
  __m128i v180; // rax
  __int64 v181; // r9
  __int64 v182; // r13
  __int64 v183; // rdx
  __int64 v184; // rbx
  __int64 *v185; // rax
  __int64 v186; // r8
  __int64 v187; // rdx
  __int64 *v188; // r13
  __int128 v189; // rax
  __int64 v190; // rax
  int v191; // edx
  __int64 v192; // rdx
  __int64 v193; // r13
  _QWORD *v194; // rbx
  __int64 v195; // rax
  char v196; // dl
  __int64 v197; // r15
  unsigned __int8 v198; // al
  __int64 *v199; // rax
  __int64 v200; // rcx
  __int64 *v201; // r12
  __int64 *v202; // rdx
  int v203; // eax
  _QWORD *v204; // rax
  __int64 *v205; // r9
  int v206; // r8d
  __int64 v207; // rsi
  __int64 *v208; // rax
  __int64 v209; // rdx
  __int64 *v210; // rax
  __int64 v211; // rdx
  __int64 v212; // rdx
  __int64 v213; // rax
  unsigned __int8 v214; // dl
  __int64 v215; // rsi
  __int64 v216; // rbx
  __int64 v217; // rdx
  char v218; // di
  unsigned int v219; // eax
  const void **v220; // r8
  __m128i *v221; // rbx
  __int128 v222; // rax
  __m128i v223; // rax
  __int64 *v224; // rdi
  __int64 *v225; // rax
  unsigned __int64 v226; // rdx
  __int64 v227; // rdx
  __m128i *v228; // rax
  __int64 *v229; // rdi
  __int64 *v230; // rax
  unsigned __int64 v231; // rdx
  __int128 v232; // rax
  __int64 *v233; // rax
  __int64 v234; // rdx
  __int64 *v235; // rbx
  __int64 v236; // rax
  const void **v237; // rcx
  __int64 v238; // r10
  __int64 v239; // rdx
  __int64 v240; // r11
  unsigned __int64 v241; // rdx
  __int64 *v242; // rsi
  __int16 *v243; // rdx
  __int16 *v244; // rax
  __m128i *v245; // rdx
  __int16 *v246; // r10
  unsigned __int64 v247; // r9
  __int64 v248; // rax
  __int8 v249; // si
  __int64 v250; // rax
  __int64 v251; // rcx
  __int64 *v252; // r11
  unsigned __int32 v253; // r8d
  const void **v254; // r12
  __int64 v255; // rdx
  unsigned int v256; // esi
  int v257; // edx
  __int64 *v258; // rax
  int v259; // edx
  unsigned __int64 v260; // r9
  __int64 *v261; // rbx
  __int64 v262; // r15
  unsigned int v263; // r13d
  unsigned int v264; // r12d
  __int64 v265; // rax
  __int64 v266; // rcx
  __int64 v267; // rdx
  __int64 v268; // rsi
  unsigned int v269; // edi
  __int64 *v270; // rcx
  __int64 v271; // r8
  int v272; // r10d
  __int64 *v273; // rax
  _QWORD *v274; // r9
  __int32 v275; // edx
  __int64 v276; // rax
  _QWORD *v277; // rax
  __int64 *v278; // rdi
  int v279; // edx
  __int64 *v280; // rax
  unsigned int v281; // edx
  __int64 *v282; // rax
  __int64 *v283; // rbx
  __int64 v284; // rdx
  __int64 v285; // r13
  __int64 *v286; // r12
  __int64 v287; // rax
  unsigned __int8 v288; // al
  const void ***v289; // rax
  int v290; // edx
  __int64 v291; // r9
  __int64 *v292; // rax
  int v293; // edx
  __int64 *v294; // r12
  int v295; // ebx
  __int64 *v296; // rax
  __int64 v297; // rsi
  __int64 *v298; // rbx
  __int64 v299; // rdx
  __int64 *v300; // r12
  unsigned __int64 v301; // rdx
  unsigned __int64 v302; // r13
  __int64 *v303; // rax
  unsigned int v304; // edx
  const void ***v305; // rax
  __int64 *v306; // rax
  int v307; // edx
  __int64 *v308; // rbx
  __int64 *v309; // rax
  __int64 *v310; // rbx
  __int64 v311; // rax
  __int64 v312; // rdx
  __int64 v313; // r13
  __int64 v314; // r12
  __int64 *v315; // rax
  unsigned __int64 v316; // rdx
  __int128 v317; // rax
  __int64 v318; // rax
  int v319; // edx
  __int64 v320; // rbx
  __int64 *v321; // rax
  __int128 v322; // rax
  __int64 v323; // r12
  __int64 v324; // rdx
  __int64 v325; // r13
  __int64 v326; // rax
  __int64 v327; // rax
  const void **v328; // rdx
  __int64 v329; // rax
  int v330; // edx
  __int64 v331; // rbx
  __int64 *v332; // rax
  __int64 v333; // rbx
  __int64 v334; // rax
  __int64 v335; // r12
  __int64 v336; // rax
  __m128i *v337; // rsi
  _QWORD *v338; // r13
  __int64 *v339; // rax
  __int64 v340; // r9
  __int64 v341; // rdx
  _QWORD *v342; // rax
  int v343; // edx
  unsigned __int64 v344; // r12
  unsigned __int64 v345; // rsi
  __int64 *v346; // rax
  __int64 v347; // rbx
  __int64 *v348; // r8
  __int64 v349; // rax
  unsigned int v350; // edx
  __m128i *v351; // rsi
  _QWORD *v352; // rax
  int v353; // edx
  int v354; // r12d
  _QWORD *v355; // rbx
  __int64 *v356; // rax
  _QWORD *v357; // r12
  __int64 v358; // rax
  unsigned __int8 v359; // al
  __int64 v360; // rcx
  __int64 v361; // r8
  __int64 v362; // r9
  _QWORD *v363; // rax
  int v364; // edx
  _QWORD *v365; // rbx
  __int64 *v366; // rax
  __int64 v367; // r12
  __int32 v368; // edx
  __int32 v369; // edx
  int v370; // edx
  __int32 v371; // edx
  int v372; // edx
  __int64 *v373; // rax
  _QWORD *v374; // rdi
  __int32 v375; // edx
  __int64 v376; // rcx
  __int64 v377; // r8
  __int64 v378; // r9
  __int64 v379; // rsi
  _QWORD *v380; // rdi
  __int64 v381; // rdx
  __int64 v382; // rcx
  __int64 v383; // r8
  __int64 v384; // r9
  _QWORD *v385; // rax
  __int64 *v386; // rdi
  __int32 v387; // edx
  __int64 *v388; // rax
  int v389; // edx
  __int64 *v390; // r12
  __int64 v391; // rdx
  __int64 *v392; // rcx
  const void ***v393; // rax
  __int64 v394; // rbx
  int v395; // edx
  int v396; // r12d
  __int64 *v397; // rax
  __int64 *v398; // r12
  __int64 v399; // rdx
  __int64 *v400; // rcx
  const void ***v401; // rax
  __int64 v402; // rbx
  int v403; // edx
  int v404; // r12d
  __int64 *v405; // rax
  __int64 *v406; // r12
  __int64 v407; // rdx
  __int64 *v408; // rcx
  const void ***v409; // rax
  __int64 v410; // rbx
  int v411; // edx
  int v412; // r12d
  __int64 *v413; // rax
  __int64 v414; // r13
  __int64 v415; // rax
  unsigned __int8 v416; // bl
  __int64 v417; // r9
  __int64 v418; // rdx
  unsigned int v419; // r12d
  unsigned __int64 v420; // r12
  __int64 v421; // r13
  const char *v422; // rax
  __int64 v423; // rdx
  __int64 v424; // r13
  const char *v425; // rsi
  __int64 v426; // rax
  __int128 v427; // rax
  __int64 v428; // rax
  __int64 v429; // rdx
  __int64 v430; // r13
  __int64 v431; // r12
  __int64 *v432; // rax
  unsigned __int64 v433; // rdx
  __int64 *v434; // rax
  int v435; // edx
  __int64 *v436; // rbx
  __int64 *v437; // rax
  unsigned int v438; // r12d
  __int64 *v439; // rax
  unsigned int v440; // edx
  unsigned __int8 *v441; // rax
  unsigned int v442; // r13d
  __int64 *v443; // rbx
  __int64 v444; // rdx
  __int64 *v445; // rax
  unsigned __int64 v446; // rdx
  __int64 *v447; // rax
  int v448; // edx
  int v449; // r12d
  __int64 *v450; // rbx
  __int64 *v451; // rax
  __int64 *v452; // r12
  __int64 v453; // rdx
  __int64 v454; // r13
  __int64 v455; // rax
  __int64 v456; // rax
  unsigned int v457; // eax
  __int64 v458; // rdx
  __int64 v459; // rcx
  __int64 v460; // r9
  __int64 *v461; // rbx
  const void ***v462; // rax
  int v463; // edx
  __int64 v464; // r9
  __int64 *v465; // rax
  __int64 v466; // rsi
  int v467; // edx
  __int64 *v468; // rax
  __int64 *v469; // rbx
  __int64 *v470; // r12
  __int64 v471; // rdx
  __int64 v472; // r13
  __int64 v473; // rax
  unsigned __int8 v474; // al
  __int64 v475; // rax
  int v476; // edx
  __int64 v477; // rbx
  __int64 *v478; // rax
  __int64 v479; // r12
  __int64 v480; // rax
  unsigned int v481; // eax
  __int64 v482; // r10
  __int64 (*v483)(); // rcx
  __int64 v484; // rdx
  __int64 *v485; // r12
  unsigned __int64 v486; // rdx
  unsigned __int64 v487; // r13
  __int64 *v488; // rax
  unsigned int v489; // edx
  __int64 *v490; // rax
  unsigned __int64 v491; // rdx
  unsigned __int64 v492; // r13
  __int64 v493; // r12
  __int64 v494; // rdx
  __int64 *v495; // rax
  unsigned int v496; // edx
  __int64 *v497; // rbx
  int v498; // edx
  __int64 *v499; // rax
  __int64 *v500; // rbx
  __int64 *v501; // rax
  __int64 v502; // rdx
  __int64 v503; // rdx
  __int64 *v504; // r12
  __int16 *v505; // rdx
  __int16 *v506; // r13
  __int64 *v507; // rax
  unsigned int v508; // edx
  const void ***v509; // rax
  __int64 *v510; // rax
  int v511; // edx
  __int64 *v512; // rbx
  __int64 *v513; // rax
  __m128i v514; // rax
  __int64 v515; // rcx
  int v516; // r8d
  int v517; // r9d
  __int64 v518; // rax
  char v519; // al
  __int64 *v520; // rbx
  __int64 *v521; // r12
  __int64 v522; // rdx
  __int64 v523; // r13
  __int64 v524; // rax
  unsigned __int8 v525; // al
  __int64 v526; // rax
  int v527; // edx
  __int64 v528; // rbx
  __int64 *v529; // rax
  __int64 *v530; // rax
  __int64 *v531; // rbx
  __int64 v532; // rdx
  __int64 v533; // r13
  __int64 *v534; // r12
  const void ***v535; // rax
  int v536; // edx
  __int64 v537; // r9
  char *(*v538)(); // rax
  __int64 *v539; // rax
  __int64 v540; // rdx
  __int64 v541; // rdi
  const void ***v542; // rdx
  __int64 v543; // rax
  int v544; // edx
  int v545; // r12d
  __int64 v546; // rbx
  __int64 *v547; // rax
  __int64 *v548; // rax
  __int64 *v549; // r10
  __int64 v550; // rdx
  __int64 v551; // r13
  __int64 *v552; // r12
  unsigned __int8 *v553; // rdx
  __int64 v554; // rax
  const void **v555; // rbx
  __int64 v556; // rcx
  int v557; // eax
  __int64 v558; // rsi
  __int64 v559; // rax
  int v560; // edx
  int v561; // r12d
  __int64 v562; // rbx
  __int64 *v563; // rax
  __int64 v564; // rbx
  __int64 v565; // rax
  __int64 v566; // rdi
  __int64 v567; // rdx
  _QWORD *v568; // rax
  __int64 *v569; // rax
  int v570; // edx
  int v571; // r12d
  __int64 *v572; // rbx
  __int64 *v573; // rax
  __int64 *v574; // r12
  unsigned __int64 v575; // rdx
  __int64 *v576; // rsi
  const void ***v577; // rax
  __int64 v578; // rbx
  int v579; // edx
  int v580; // r12d
  __int64 *v581; // rax
  __int64 v582; // rax
  unsigned int v583; // eax
  __int64 v584; // rax
  __int64 v585; // r12
  int v586; // edx
  int v587; // ebx
  __int64 *v588; // rax
  __int64 *v589; // rbx
  unsigned __int64 v590; // rdx
  __int64 *v591; // r12
  unsigned __int64 v592; // r13
  const void ***v593; // rax
  __int64 v594; // rbx
  int v595; // edx
  int v596; // r12d
  __int64 *v597; // rax
  __int64 *v598; // r12
  __int128 v599; // rax
  __int64 v600; // rax
  __int64 v601; // rdi
  int v602; // edx
  __int32 v603; // edx
  __int64 *v604; // rax
  __int64 *v605; // rbx
  __int32 v606; // edx
  const void ***v607; // rax
  int v608; // edx
  __int64 v609; // r9
  __int64 *v610; // rax
  __int64 *v611; // rax
  __int64 *v612; // r12
  __int64 v613; // rdx
  __int64 *v614; // rax
  unsigned __int64 v615; // rdx
  __int64 *v616; // rax
  int v617; // edx
  __int64 v618; // rdi
  __int64 *v619; // rax
  __int32 v620; // edx
  __int32 v621; // r13d
  __int64 v622; // r12
  __int64 v623; // rdx
  __int128 v624; // rax
  __int64 v625; // rax
  __int64 v626; // rdx
  __int64 v627; // rbx
  char v628; // r8
  _QWORD *v629; // r10
  __int64 *v630; // rdx
  __int64 *v631; // rax
  __int64 *v632; // rax
  int v633; // edx
  __int64 v634; // rax
  int v635; // edx
  __int64 *v636; // rax
  __int64 v637; // rdx
  __int64 v638; // r13
  __int64 v639; // r12
  __int128 v640; // rax
  __int128 v641; // rax
  int v642; // eax
  int v643; // edx
  __int64 v644; // rbx
  __int64 v645; // rdx
  __int64 v646; // rbx
  __int64 v647; // rax
  int v648; // eax
  bool v649; // al
  char v650; // r8
  __int64 *v651; // r10
  __int64 *v652; // rax
  __int64 *v653; // rax
  __int64 *v654; // rax
  unsigned __int64 v655; // rdx
  __int64 v656; // r13
  __int64 (*v657)(void); // rax
  int v658; // ebx
  __int64 v659; // rax
  __int64 v660; // rdx
  __int64 v661; // r12
  int v662; // r12d
  __int64 v663; // rax
  __int64 v664; // rdx
  int v665; // edx
  bool v666; // zf
  __int64 v667; // rbx
  __int64 v668; // rax
  __int64 v669; // r13
  const char *v670; // rax
  __int64 v671; // rdx
  const char *v672; // rsi
  __int64 v673; // rax
  __int64 *v674; // r12
  __int64 v675; // r13
  __int64 v676; // rdx
  __int64 v677; // rax
  unsigned int v678; // edx
  __int64 *v679; // rax
  unsigned __int64 v680; // rdx
  __int64 v681; // rdi
  __int64 v682; // rax
  unsigned int v683; // r12d
  __int64 v684; // rdx
  __int64 (*v685)(); // rax
  __int64 v686; // rbx
  unsigned int v687; // edx
  __int64 v688; // r9
  int v689; // r11d
  _QWORD *v690; // rax
  _QWORD *v691; // r10
  __int64 v692; // rdx
  __int64 v693; // rbx
  int v694; // edx
  int v695; // r13d
  int v696; // r12d
  __int64 *v697; // rax
  int v698; // edx
  __int64 v699; // rax
  unsigned int v700; // r12d
  __int64 v701; // r13
  __int64 v702; // rdx
  __int64 v703; // rax
  __int64 (*v704)(); // rdx
  __int64 *v705; // r13
  int v706; // eax
  _QWORD *v707; // r10
  int v708; // ecx
  __int64 v709; // rax
  __int64 *v710; // rax
  __int64 v711; // rdx
  __int64 v712; // r12
  unsigned int v713; // edx
  int v714; // r8d
  __int64 (*v715)(); // rax
  __int64 *v716; // rax
  __int64 *v717; // r12
  __int64 v718; // rdx
  unsigned int v719; // r13d
  unsigned __int64 v720; // rdx
  __int64 *v721; // rcx
  __int64 v722; // r10
  __int64 v723; // rbx
  unsigned __int64 v724; // r11
  __int64 v725; // rax
  unsigned __int8 v726; // dl
  __int64 v727; // rbx
  int v728; // edx
  int v729; // r12d
  __int64 *v730; // rax
  __int64 *v731; // rbx
  __int64 *v732; // r12
  __int64 v733; // rdx
  __int64 v734; // r13
  __int64 *v735; // rax
  unsigned int v736; // edx
  const void ***v737; // rdx
  __int64 v738; // rax
  int v739; // edx
  __int64 v740; // rbx
  __int64 *v741; // rax
  __int64 v742; // rax
  __int64 v743; // rdx
  __int64 v744; // rcx
  int v745; // r8d
  int v746; // r9d
  __int64 *v747; // r12
  __int64 v748; // rdx
  __int64 v749; // r13
  __int64 v750; // r8
  __int64 v751; // r9
  __int64 v752; // rdx
  __int64 v753; // rax
  __int64 v754; // rax
  __int64 *v755; // rbx
  __int64 v756; // rdx
  const void ***v757; // rax
  int v758; // edx
  __int64 v759; // r9
  __int64 *v760; // r12
  int v761; // edx
  __int64 *v762; // rax
  __int64 v763; // rsi
  int v764; // eax
  __int64 v765; // rax
  __int64 v766; // rdx
  bool v767; // cc
  _QWORD *v768; // rdx
  bool v769; // cf
  __int64 v770; // rdx
  unsigned int v771; // ebx
  __int32 v772; // edx
  int v773; // edx
  __int32 v774; // edx
  int v775; // edx
  __int32 v776; // edx
  int v777; // edx
  __int64 *v778; // rax
  __int32 v779; // edx
  _QWORD *v780; // rax
  __int64 *v781; // rax
  _QWORD *v782; // rdi
  __int64 v783; // rax
  __int64 v784; // rdx
  __int64 v785; // r13
  __int64 v786; // r12
  __int64 v787; // rcx
  __int64 v788; // r8
  __int64 v789; // r9
  __int64 v790; // rax
  int v791; // edx
  __m128i v792; // rax
  __int64 v793; // rcx
  int v794; // r8d
  int v795; // r9d
  __int64 v796; // rdi
  int v797; // edx
  __int64 v798; // r12
  __int64 v799; // rdx
  __int64 v800; // rbx
  unsigned __int64 v801; // rdx
  unsigned __int64 v802; // r9
  __int64 v803; // r8
  int v804; // ecx
  __int64 v805; // rbx
  __int64 *v806; // rax
  unsigned int v807; // edx
  __int64 *v808; // rax
  __int64 *v809; // rbx
  __int64 *v810; // r12
  __int64 v811; // rdx
  __int64 v812; // r13
  __int64 v813; // rax
  unsigned __int8 v814; // al
  __int64 v815; // rax
  int v816; // edx
  __int64 v817; // rbx
  __int64 *v818; // rax
  __int64 *v819; // rax
  __int64 *v820; // r10
  __int64 v821; // rdx
  __int64 v822; // r13
  __int64 *v823; // r12
  unsigned __int8 *v824; // rdx
  __int64 v825; // rax
  const void **v826; // rbx
  __int64 v827; // rcx
  int v828; // eax
  __int64 v829; // rsi
  __int64 v830; // rax
  __int64 v831; // rbx
  int v832; // edx
  int v833; // r12d
  __int64 *v834; // rax
  __int64 v835; // rdx
  __int64 v836; // r13
  __int64 v837; // r12
  __int64 v838; // rax
  __int64 v839; // rcx
  char v840; // al
  __int64 v841; // rax
  _QWORD *v842; // rax
  int v843; // r8d
  __int64 *v844; // r9
  __int64 *v845; // rax
  __int32 v846; // edx
  __int64 v847; // rdx
  __int128 v848; // rax
  int v849; // ebx
  __int64 v850; // rax
  __int64 v851; // r13
  __int64 v852; // rdx
  __int64 v853; // r12
  char v854; // r8
  _QWORD *v855; // r10
  __int64 *v856; // rax
  __int64 *v857; // rax
  int v858; // edx
  __int64 v859; // rax
  int v860; // edx
  unsigned int v861; // ebx
  __int64 *v862; // rax
  __int64 v863; // rdx
  __int64 v864; // r13
  __int64 *v865; // r12
  __int128 v866; // rax
  __int128 v867; // rax
  unsigned int v868; // eax
  __int64 v869; // rdx
  __int64 v870; // rax
  int v871; // eax
  bool v872; // al
  char v873; // r8
  __int64 v874; // r10
  __int64 *v875; // rax
  __int64 *v876; // rax
  __int64 v877; // r8
  unsigned __int64 v878; // rdx
  __int64 v879; // rax
  int v880; // edx
  _QWORD *v881; // r12
  __int64 v882; // rax
  unsigned __int8 v883; // al
  __int64 v884; // r9
  _QWORD *v885; // rax
  int v886; // edx
  _QWORD *v887; // rbx
  __int64 *v888; // rax
  __int64 v889; // rax
  __int64 *v890; // rax
  __int64 v891; // rdx
  __int64 *v892; // rax
  __int64 v893; // rdx
  __m128i v894; // rax
  __int64 v895; // rdi
  __int64 v896; // rax
  int v897; // edx
  int v898; // ebx
  __m128i v899; // rax
  __int64 v900; // rcx
  int v901; // r8d
  int v902; // r9d
  __int64 v903; // rdx
  __int64 v904; // rcx
  int v905; // r8d
  int v906; // r9d
  __int64 v907; // rdx
  __int64 v908; // rcx
  int v909; // r8d
  int v910; // r9d
  __int64 v911; // rdx
  __int64 v912; // rcx
  int v913; // r8d
  int v914; // r9d
  __int64 v915; // r9
  __int64 v916; // rbx
  __m128i *v917; // rsi
  __int64 *v918; // rax
  __int64 v919; // rax
  __int64 *v920; // rax
  __int64 v921; // rdx
  __m128i v922; // rax
  __int64 v923; // rdi
  __int64 v924; // rax
  int v925; // edx
  int v926; // ebx
  __m128i v927; // rax
  __int64 v928; // rcx
  int v929; // r8d
  int v930; // r9d
  __int64 v931; // rdx
  __int64 v932; // rcx
  int v933; // r8d
  int v934; // r9d
  __int64 v935; // rdx
  __int64 v936; // rcx
  int v937; // r8d
  int v938; // r9d
  __int64 v939; // r9
  __int64 v940; // rbx
  __int64 *v941; // r12
  __int64 v942; // rdx
  __int16 *v943; // r13
  __int64 v944; // rcx
  __int64 v945; // r8
  __int64 v946; // r9
  __int128 v947; // rax
  __int64 *v948; // r15
  __int64 *v949; // rax
  __int64 v950; // rdx
  __int64 *v951; // rax
  int v952; // edx
  __int64 v953; // r12
  __int64 *v954; // rax
  unsigned int v955; // r13d
  unsigned int v956; // edx
  unsigned __int8 *v957; // rax
  const void **v958; // r8
  unsigned int v959; // ebx
  int v960; // eax
  __int64 v961; // r12
  int v962; // edx
  int v963; // ebx
  __int64 *v964; // rax
  unsigned int v965; // r12d
  __int64 *v966; // rax
  unsigned int v967; // edx
  unsigned __int8 *v968; // rax
  unsigned int v969; // r13d
  __int64 *v970; // rbx
  __int64 v971; // rdx
  __int64 *v972; // rax
  unsigned __int64 v973; // rdx
  __int64 *v974; // rax
  int v975; // edx
  int v976; // r12d
  __int64 *v977; // rbx
  __int64 *v978; // rax
  __int128 v979; // rax
  __int64 *v980; // rax
  unsigned __int64 v981; // rdx
  __int64 *v982; // rax
  __int32 v983; // edx
  __int32 v984; // r13d
  __int64 v985; // r12
  __int64 v986; // rdx
  __int128 v987; // rax
  __int64 v988; // rax
  __int64 v989; // rdx
  __int64 v990; // rbx
  char v991; // r8
  _QWORD *v992; // r10
  __int64 *v993; // rdx
  __int64 *v994; // rax
  __int64 *v995; // rax
  int v996; // edx
  __int64 *v997; // rax
  __int64 v998; // rdx
  __int64 v999; // r13
  __int64 v1000; // r12
  __int128 v1001; // rax
  __int128 v1002; // rax
  int v1003; // eax
  int v1004; // edx
  __int64 v1005; // rbx
  __int64 v1006; // rdx
  __int64 v1007; // rbx
  __int64 v1008; // rax
  int v1009; // eax
  bool v1010; // al
  char v1011; // r8
  __int64 *v1012; // r10
  __int64 *v1013; // rax
  __int64 *v1014; // rax
  __int64 *v1015; // rax
  __int64 v1016; // rdx
  __int64 *v1017; // rbx
  __int64 *v1018; // r12
  __int64 v1019; // rdx
  __int64 v1020; // r13
  __int64 *v1021; // rax
  unsigned int v1022; // edx
  const void ***v1023; // rdx
  __int64 v1024; // rax
  int v1025; // edx
  __int64 v1026; // rbx
  __int64 *v1027; // rax
  __int64 *v1028; // rbx
  __int64 *v1029; // r12
  __int64 v1030; // rdx
  __int64 v1031; // r13
  __int64 v1032; // rax
  unsigned __int8 v1033; // al
  __int64 v1034; // rax
  int v1035; // edx
  __int64 v1036; // rbx
  __int64 *v1037; // rax
  unsigned int v1038; // r13d
  __int64 v1039; // rdi
  __int64 v1040; // rax
  __int64 *v1041; // rdi
  __int32 v1042; // edx
  unsigned int v1043; // edx
  __int64 v1044; // rax
  unsigned int v1045; // eax
  const void **v1046; // rdx
  const void **v1047; // rbx
  unsigned int v1048; // edx
  __int128 v1049; // rax
  unsigned __int64 v1050; // rax
  __int16 *v1051; // rdx
  __int64 *v1052; // r12
  int v1053; // edx
  __int64 *v1054; // rax
  unsigned int v1055; // ebx
  __int64 *v1056; // rax
  __int64 v1057; // rdx
  __int64 v1058; // r13
  __int64 *v1059; // r12
  unsigned __int8 *v1060; // rax
  __int64 v1061; // rdx
  const void ***v1062; // rax
  int v1063; // edx
  __int64 v1064; // r9
  __int64 *v1065; // rax
  int v1066; // edx
  __int64 *v1067; // rbx
  __int64 *v1068; // rax
  __int32 v1069; // edx
  int v1070; // eax
  __int64 *v1071; // rbx
  const void ***v1072; // rax
  int v1073; // edx
  __int64 v1074; // r9
  __int64 *v1075; // rax
  __int64 v1076; // rdi
  __int64 *v1077; // rbx
  int v1078; // edx
  int v1079; // r14d
  __int64 *v1080; // rax
  _QWORD *v1081; // r13
  __int64 v1082; // rdx
  __int64 v1083; // rcx
  int v1084; // r8d
  int v1085; // r9d
  unsigned int v1086; // r13d
  bool v1087; // sf
  __int64 v1088; // rax
  __int64 v1089; // rdx
  __int64 v1090; // rbx
  int v1091; // ebx
  __int64 v1092; // rax
  __int64 v1093; // rdx
  int v1094; // eax
  unsigned int v1095; // r12d
  __int64 v1096; // rax
  unsigned __int16 *v1097; // rax
  char v1098; // dl
  __int64 v1099; // rcx
  unsigned int v1100; // esi
  __int64 *v1101; // r10
  __int64 v1102; // r10
  unsigned int v1103; // edx
  int v1104; // eax
  int v1105; // r8d
  unsigned int v1106; // eax
  unsigned __int64 *v1107; // rax
  int v1108; // edx
  char v1109; // al
  __int64 v1110; // rax
  __int64 *v1111; // rax
  __int64 v1112; // r9
  __int32 v1113; // edx
  __int64 v1114; // rax
  unsigned __int8 v1115; // al
  __int64 v1116; // rax
  __int64 v1117; // rsi
  __int64 v1118; // rdi
  __int32 v1119; // edx
  void (***v1120)(); // rdi
  void (*v1121)(); // r8
  _QWORD *v1122; // rax
  int v1123; // r8d
  __int64 *v1124; // r9
  bool v1125; // al
  bool v1126; // al
  unsigned int v1127; // eax
  unsigned int v1128; // ecx
  __int64 v1129; // rax
  const void **v1130; // rdx
  unsigned __int8 v1131; // al
  __int64 v1132; // rax
  __int64 *v1133; // rbx
  int v1134; // edx
  int v1135; // r12d
  __int64 *v1136; // rax
  __int64 v1137; // r13
  char v1138; // al
  unsigned int v1139; // edx
  __int64 *v1140; // r8
  int v1141; // eax
  unsigned int v1142; // ecx
  unsigned __int64 v1143; // r11
  unsigned int v1144; // esi
  __int64 *v1145; // rbx
  unsigned int v1146; // r12d
  unsigned int v1147; // r14d
  __int64 v1148; // rdi
  __int64 v1149; // rax
  __int64 *v1150; // rax
  unsigned int v1151; // edx
  __int64 v1152; // rax
  unsigned int v1153; // edx
  int v1154; // r15d
  __int32 v1155; // r12d
  __int64 v1156; // rdx
  __int64 v1157; // rax
  __int64 v1158; // r13
  __int64 v1159; // r12
  __int64 *v1160; // rax
  int v1161; // r8d
  __int64 *v1162; // r9
  unsigned __int16 v1163; // cx
  __int64 v1164; // rax
  __int64 v1165; // rax
  __int64 v1166; // rax
  __m128i v1167; // xmm7
  char v1168; // al
  __int64 *v1169; // rbx
  int v1170; // edx
  __int64 *v1171; // rax
  __int64 v1172; // rax
  unsigned int v1173; // edx
  __int64 *v1174; // rax
  __int64 v1175; // rax
  _QWORD *v1176; // rax
  unsigned __int64 v1177; // rdx
  __int64 *v1178; // r12
  int v1179; // edx
  unsigned int v1180; // edx
  __int64 v1181; // rax
  char v1182; // al
  int v1183; // edx
  unsigned int v1184; // edx
  __int64 v1185; // rax
  __int32 v1186; // edx
  __int64 v1187; // rax
  __m128i v1188; // rax
  __int64 v1189; // rcx
  int v1190; // r8d
  int v1191; // r9d
  __int64 *v1192; // rax
  __int64 v1193; // rax
  char v1194; // al
  __int64 *v1195; // rbx
  __int64 *v1196; // rax
  __int64 v1197; // rdx
  __int64 *v1198; // r12
  __int64 v1199; // rdx
  __int64 v1200; // r13
  __int16 *v1201; // rdx
  __int64 *v1202; // rax
  unsigned int v1203; // edx
  const void ***v1204; // rax
  __int64 *v1205; // rbx
  int v1206; // edx
  __int64 *v1207; // rax
  unsigned __int16 v1208; // cx
  __int64 v1209; // rcx
  __int64 *v1210; // rax
  __int64 *v1211; // rbx
  __int64 v1212; // r13
  int v1213; // r12d
  unsigned __int8 *v1214; // rsi
  __int64 v1215; // rdx
  int v1216; // ecx
  int v1217; // r10d
  __int64 v1218; // rax
  unsigned int v1219; // edi
  __int64 v1220; // rsi
  int v1221; // edx
  unsigned int i; // eax
  __int64 v1223; // rcx
  char v1224; // al
  unsigned __int64 v1225; // rdx
  __int64 v1226; // rax
  unsigned __int64 v1227; // rdx
  __int64 v1228; // rdx
  __int128 v1229; // rax
  __int64 *v1230; // rax
  unsigned __int64 v1231; // rdx
  __int128 v1232; // rax
  __int64 *v1233; // rax
  unsigned __int64 v1234; // rdx
  __int64 *v1235; // rax
  unsigned __int64 v1236; // rdx
  __int128 v1237; // rax
  __int64 *v1238; // rax
  unsigned __int64 v1239; // rdx
  __int64 *v1240; // rax
  unsigned __int64 v1241; // rdx
  __int128 v1242; // rax
  __int64 *v1243; // rax
  unsigned __int64 v1244; // rdx
  __int64 *v1245; // rax
  unsigned __int64 v1246; // rdx
  __int128 v1247; // rax
  __int64 *v1248; // rax
  __int64 *v1249; // r10
  __int64 v1250; // rdx
  int v1251; // edx
  __int64 v1252; // rdx
  __int64 *v1253; // rax
  __int64 v1254; // rax
  unsigned int v1255; // edi
  __int64 v1256; // r11
  int v1257; // r8d
  unsigned int j; // eax
  __int64 v1259; // rsi
  __int64 v1260; // rax
  int v1261; // r15d
  __int64 v1262; // rax
  _DWORD *v1263; // rax
  _DWORD *v1264; // rcx
  int v1265; // edx
  __int32 v1266; // ebx
  __int64 v1267; // r13
  __int64 v1268; // r14
  __int64 v1269; // rt1
  unsigned int v1270; // r12d
  int *v1271; // r15
  unsigned int v1272; // ebx
  unsigned int v1273; // ecx
  __int64 v1274; // rax
  int v1275; // r8d
  __int64 *v1276; // r9
  unsigned __int64 v1277; // rdx
  __int64 v1278; // rax
  unsigned __int64 v1279; // rdx
  __int128 v1280; // rax
  __int64 *v1281; // rax
  unsigned __int64 v1282; // rdx
  __int64 v1283; // rdx
  __int128 v1284; // rax
  __int64 *v1285; // rax
  unsigned __int64 v1286; // rdx
  __int128 v1287; // rax
  __int64 *v1288; // rax
  unsigned __int64 v1289; // rdx
  __int64 *v1290; // rax
  unsigned __int64 v1291; // rdx
  __int128 v1292; // rax
  __int64 *v1293; // rax
  unsigned __int64 v1294; // rdx
  __int64 *v1295; // rax
  unsigned __int64 v1296; // rdx
  __int128 v1297; // rax
  __int64 *v1298; // rax
  __int64 *v1299; // r10
  __int64 v1300; // rdx
  int v1301; // edx
  unsigned __int64 v1302; // rdx
  __int64 v1303; // rax
  unsigned __int64 v1304; // rdx
  __int128 v1305; // rax
  __int64 *v1306; // rax
  unsigned __int64 v1307; // rdx
  __int64 v1308; // rdx
  __int128 v1309; // rax
  __int64 *v1310; // rax
  unsigned __int64 v1311; // rdx
  __int128 v1312; // rax
  __int64 *v1313; // rax
  unsigned __int64 v1314; // rdx
  __int64 *v1315; // rax
  unsigned __int64 v1316; // rdx
  __int128 v1317; // rax
  __int64 *v1318; // rax
  unsigned __int64 v1319; // rdx
  __int64 *v1320; // rax
  unsigned __int64 v1321; // rdx
  __int128 v1322; // rax
  __int64 *v1323; // rax
  unsigned __int64 v1324; // rdx
  __int64 *v1325; // rax
  unsigned __int64 v1326; // rdx
  __int128 v1327; // rax
  __int64 *v1328; // rax
  __int64 *v1329; // r10
  __int64 v1330; // rdx
  int v1331; // edx
  int v1332; // edx
  __int128 v1333; // rax
  __int64 *v1334; // rax
  unsigned __int64 v1335; // rdx
  int v1336; // edx
  int v1337; // esi
  void *v1338; // rax
  __int64 v1339; // rcx
  __int128 v1340; // rax
  __int64 *v1341; // rax
  unsigned __int64 v1342; // rdx
  __int128 v1343; // rax
  __int64 *v1344; // rax
  unsigned __int64 v1345; // rdx
  __int64 *v1346; // rax
  unsigned __int64 v1347; // rdx
  __int128 v1348; // rax
  __int64 *v1349; // rax
  unsigned __int64 v1350; // rdx
  __int64 *v1351; // rax
  unsigned __int64 v1352; // rdx
  __int128 v1353; // rax
  __int64 *v1354; // rax
  unsigned __int64 v1355; // rdx
  __int64 *v1356; // rax
  unsigned __int64 v1357; // rdx
  __int128 v1358; // rax
  __int64 *v1359; // rax
  unsigned __int64 v1360; // rdx
  __int64 *v1361; // rax
  unsigned __int64 v1362; // rdx
  __int128 v1363; // rax
  __int64 *v1364; // rax
  unsigned __int64 v1365; // rdx
  __int64 *v1366; // rax
  unsigned __int64 v1367; // rdx
  __int128 v1368; // rax
  __int64 *v1369; // rax
  unsigned __int64 v1370; // rdx
  __int128 v1371; // rax
  __int64 *v1372; // rax
  unsigned __int64 v1373; // rdx
  __int64 *v1374; // rax
  unsigned __int64 v1375; // rdx
  __int128 v1376; // rax
  __int128 v1377; // rax
  __int64 *v1378; // rax
  unsigned __int64 v1379; // rdx
  __int128 v1380; // rax
  __int64 *v1381; // rax
  unsigned __int64 v1382; // rdx
  __int64 *v1383; // rax
  unsigned __int64 v1384; // rdx
  __int128 v1385; // rax
  __int64 *v1386; // rax
  unsigned __int64 v1387; // rdx
  __int64 *v1388; // rax
  unsigned __int64 v1389; // rdx
  __int128 v1390; // rax
  __int64 *v1391; // rax
  unsigned __int64 v1392; // rdx
  __int64 *v1393; // rax
  unsigned __int64 v1394; // rdx
  __int128 v1395; // rax
  __int64 *v1396; // rax
  unsigned __int64 v1397; // rdx
  __int64 *v1398; // rax
  unsigned __int64 v1399; // rdx
  __int128 v1400; // rax
  __int64 *v1401; // rax
  unsigned __int64 v1402; // rdx
  __int128 v1403; // rax
  __int64 *v1404; // rax
  unsigned __int64 v1405; // rdx
  __int64 *v1406; // rax
  unsigned __int64 v1407; // rdx
  __int128 v1408; // rax
  __int128 v1409; // rax
  __int64 *v1410; // rax
  unsigned __int64 v1411; // rdx
  __int128 v1412; // rax
  __int64 *v1413; // rax
  unsigned __int64 v1414; // rdx
  __int64 *v1415; // rax
  unsigned __int64 v1416; // rdx
  __int128 v1417; // rax
  __int64 *v1418; // rax
  unsigned __int64 v1419; // rdx
  __int64 *v1420; // rax
  unsigned __int64 v1421; // rdx
  __int128 v1422; // rax
  __int64 *v1423; // rax
  unsigned __int64 v1424; // rdx
  __int64 *v1425; // rax
  unsigned __int64 v1426; // rdx
  __int128 v1427; // rax
  __int64 *v1428; // rax
  unsigned __int64 v1429; // rdx
  __int64 *v1430; // rax
  unsigned __int64 v1431; // rdx
  __int128 v1432; // rax
  __int64 *v1433; // rax
  unsigned __int64 v1434; // rdx
  __int64 *v1435; // rax
  unsigned __int64 v1436; // rdx
  __int128 v1437; // rax
  __int64 *v1438; // rax
  unsigned __int64 v1439; // rdx
  __int128 v1440; // rax
  __int64 *v1441; // rax
  unsigned __int64 v1442; // rdx
  __int64 *v1443; // rax
  unsigned __int64 v1444; // rdx
  __int128 v1445; // rax
  __int128 v1446; // rax
  __int64 *v1447; // rax
  unsigned __int64 v1448; // rdx
  int v1449; // edx
  __int64 v1450; // r13
  __int64 v1451; // rbx
  __int32 v1452; // edx
  __m128i v1453; // rax
  __m128i *v1454; // rdi
  __m128i v1455; // xmm7
  __int64 v1456; // rcx
  const void **v1457; // r8
  unsigned __int64 v1458; // r9
  __int64 v1459; // r10
  __int64 *v1460; // rax
  int v1461; // edx
  __int64 v1462; // r11
  __int64 v1463; // rbx
  __int64 v1464; // r12
  __int64 v1465; // rax
  __int64 v1466; // r15
  __int64 v1467; // r14
  __int64 v1468; // r12
  __int64 v1469; // r13
  __int64 k; // rbx
  __int64 v1471; // rax
  __int64 v1472; // r11
  __int64 v1473; // rax
  int v1474; // r8d
  __int64 *v1475; // r9
  unsigned int v1476; // eax
  unsigned int v1477; // eax
  __int64 v1478; // [rsp-40h] [rbp-17C0h]
  unsigned int v1479; // [rsp-30h] [rbp-17B0h]
  unsigned __int64 v1480; // [rsp-30h] [rbp-17B0h]
  __int128 v1481; // [rsp-20h] [rbp-17A0h]
  __int128 v1482; // [rsp-20h] [rbp-17A0h]
  __int128 v1483; // [rsp-20h] [rbp-17A0h]
  __int128 v1484; // [rsp-20h] [rbp-17A0h]
  __int128 v1485; // [rsp-20h] [rbp-17A0h]
  __int128 v1486; // [rsp-20h] [rbp-17A0h]
  __int128 v1487; // [rsp-20h] [rbp-17A0h]
  __int128 v1488; // [rsp-20h] [rbp-17A0h]
  __int128 v1489; // [rsp-10h] [rbp-1790h]
  __int64 v1490; // [rsp-10h] [rbp-1790h]
  __int128 v1491; // [rsp-10h] [rbp-1790h]
  __int128 v1492; // [rsp-10h] [rbp-1790h]
  __int128 v1493; // [rsp-10h] [rbp-1790h]
  __int128 v1494; // [rsp-10h] [rbp-1790h]
  __int128 v1495; // [rsp-10h] [rbp-1790h]
  __int128 v1496; // [rsp-10h] [rbp-1790h]
  __int128 v1497; // [rsp-10h] [rbp-1790h]
  __int128 v1498; // [rsp-10h] [rbp-1790h]
  __int128 v1499; // [rsp-10h] [rbp-1790h]
  __int128 v1500; // [rsp-10h] [rbp-1790h]
  __int128 v1501; // [rsp-10h] [rbp-1790h]
  __int128 v1502; // [rsp-10h] [rbp-1790h]
  __int128 v1503; // [rsp-10h] [rbp-1790h]
  __int128 v1504; // [rsp-10h] [rbp-1790h]
  __int128 v1505; // [rsp-10h] [rbp-1790h]
  __int128 v1506; // [rsp-10h] [rbp-1790h]
  __int128 v1507; // [rsp-10h] [rbp-1790h]
  __int128 v1508; // [rsp-10h] [rbp-1790h]
  __int128 v1509; // [rsp-10h] [rbp-1790h]
  __int128 v1510; // [rsp-10h] [rbp-1790h]
  __int128 v1511; // [rsp-10h] [rbp-1790h]
  __int128 v1512; // [rsp-10h] [rbp-1790h]
  __int128 v1513; // [rsp-10h] [rbp-1790h]
  __int128 v1514; // [rsp-10h] [rbp-1790h]
  __int128 v1515; // [rsp-10h] [rbp-1790h]
  __int128 v1516; // [rsp-10h] [rbp-1790h]
  __int128 v1517; // [rsp-10h] [rbp-1790h]
  __int64 *v1518; // [rsp-10h] [rbp-1790h]
  __int128 v1519; // [rsp-10h] [rbp-1790h]
  __int128 v1520; // [rsp-10h] [rbp-1790h]
  __int128 v1521; // [rsp-10h] [rbp-1790h]
  __int128 v1522; // [rsp-10h] [rbp-1790h]
  __int128 v1523; // [rsp-10h] [rbp-1790h]
  __int128 v1524; // [rsp-10h] [rbp-1790h]
  __int128 v1525; // [rsp-10h] [rbp-1790h]
  __int128 v1526; // [rsp-10h] [rbp-1790h]
  int v1527; // [rsp-8h] [rbp-1788h]
  __int64 v1528; // [rsp-8h] [rbp-1788h]
  __int16 v1529; // [rsp+Ch] [rbp-1774h]
  char v1530; // [rsp+Fh] [rbp-1771h]
  __int64 v1531; // [rsp+30h] [rbp-1750h]
  __int64 *v1532; // [rsp+48h] [rbp-1738h]
  __int64 *v1533; // [rsp+48h] [rbp-1738h]
  unsigned int v1534; // [rsp+70h] [rbp-1710h]
  __int64 *v1535; // [rsp+70h] [rbp-1710h]
  unsigned __int64 v1536; // [rsp+78h] [rbp-1708h]
  unsigned int v1537; // [rsp+80h] [rbp-1700h]
  unsigned int v1538; // [rsp+88h] [rbp-16F8h]
  __int64 v1539; // [rsp+88h] [rbp-16F8h]
  __int64 v1540; // [rsp+90h] [rbp-16F0h]
  __int64 v1541; // [rsp+98h] [rbp-16E8h]
  unsigned int v1542; // [rsp+A0h] [rbp-16E0h]
  int v1543; // [rsp+A0h] [rbp-16E0h]
  __int64 v1544; // [rsp+A0h] [rbp-16E0h]
  __int64 v1545; // [rsp+A0h] [rbp-16E0h]
  __int64 v1546; // [rsp+A8h] [rbp-16D8h]
  unsigned int v1547; // [rsp+A8h] [rbp-16D8h]
  __int64 *v1548; // [rsp+A8h] [rbp-16D8h]
  __int64 v1549; // [rsp+A8h] [rbp-16D8h]
  __int64 *v1550; // [rsp+B0h] [rbp-16D0h]
  unsigned __int16 *v1551; // [rsp+B0h] [rbp-16D0h]
  char v1552; // [rsp+B0h] [rbp-16D0h]
  __int64 *v1553; // [rsp+B0h] [rbp-16D0h]
  char v1554; // [rsp+B0h] [rbp-16D0h]
  __int64 v1555; // [rsp+B0h] [rbp-16D0h]
  int v1556; // [rsp+C0h] [rbp-16C0h]
  _QWORD *v1557; // [rsp+C0h] [rbp-16C0h]
  __int64 v1558; // [rsp+C0h] [rbp-16C0h]
  __int64 *v1559; // [rsp+C0h] [rbp-16C0h]
  __m128i *v1560; // [rsp+C0h] [rbp-16C0h]
  _QWORD *v1561; // [rsp+C0h] [rbp-16C0h]
  __int32 v1562; // [rsp+C0h] [rbp-16C0h]
  __int64 v1563; // [rsp+C0h] [rbp-16C0h]
  unsigned int v1564; // [rsp+D0h] [rbp-16B0h]
  __int64 v1565; // [rsp+D0h] [rbp-16B0h]
  unsigned int v1566; // [rsp+D0h] [rbp-16B0h]
  _QWORD *v1567; // [rsp+D0h] [rbp-16B0h]
  __int32 v1568; // [rsp+D0h] [rbp-16B0h]
  __int128 v1569; // [rsp+D0h] [rbp-16B0h]
  char v1570; // [rsp+D0h] [rbp-16B0h]
  __int64 v1571; // [rsp+D0h] [rbp-16B0h]
  char v1572; // [rsp+D0h] [rbp-16B0h]
  char v1573; // [rsp+D0h] [rbp-16B0h]
  __int128 v1574; // [rsp+D0h] [rbp-16B0h]
  char v1575; // [rsp+D0h] [rbp-16B0h]
  unsigned __int64 v1576; // [rsp+D0h] [rbp-16B0h]
  __int64 v1577; // [rsp+D0h] [rbp-16B0h]
  __int64 *v1578; // [rsp+D0h] [rbp-16B0h]
  int v1579; // [rsp+D0h] [rbp-16B0h]
  __int64 *v1580; // [rsp+D0h] [rbp-16B0h]
  __int64 *v1581; // [rsp+D0h] [rbp-16B0h]
  __int16 *v1582; // [rsp+D8h] [rbp-16A8h]
  unsigned __int64 v1583; // [rsp+D8h] [rbp-16A8h]
  unsigned __int64 v1584; // [rsp+D8h] [rbp-16A8h]
  unsigned __int64 v1585; // [rsp+D8h] [rbp-16A8h]
  const void **v1586; // [rsp+E0h] [rbp-16A0h]
  __int64 v1587; // [rsp+E0h] [rbp-16A0h]
  __int64 v1588; // [rsp+E0h] [rbp-16A0h]
  __int64 v1589; // [rsp+E0h] [rbp-16A0h]
  int v1590; // [rsp+E0h] [rbp-16A0h]
  __int128 v1591; // [rsp+E0h] [rbp-16A0h]
  int v1592; // [rsp+E0h] [rbp-16A0h]
  __int64 *v1593; // [rsp+E0h] [rbp-16A0h]
  __int64 v1594; // [rsp+E0h] [rbp-16A0h]
  __int64 v1595; // [rsp+E0h] [rbp-16A0h]
  _QWORD *v1596; // [rsp+E0h] [rbp-16A0h]
  int v1597; // [rsp+E0h] [rbp-16A0h]
  __int64 v1598; // [rsp+E0h] [rbp-16A0h]
  __int128 v1599; // [rsp+E0h] [rbp-16A0h]
  int v1600; // [rsp+E0h] [rbp-16A0h]
  __int64 *v1601; // [rsp+E0h] [rbp-16A0h]
  char *v1602; // [rsp+E0h] [rbp-16A0h]
  __int64 v1603; // [rsp+E0h] [rbp-16A0h]
  unsigned __int16 v1604; // [rsp+E0h] [rbp-16A0h]
  unsigned __int16 v1605; // [rsp+E0h] [rbp-16A0h]
  unsigned __int16 v1606; // [rsp+E0h] [rbp-16A0h]
  _QWORD *v1607; // [rsp+E0h] [rbp-16A0h]
  __int64 v1608; // [rsp+E0h] [rbp-16A0h]
  __int64 v1609; // [rsp+E0h] [rbp-16A0h]
  __int64 v1610; // [rsp+E0h] [rbp-16A0h]
  bool v1611; // [rsp+E0h] [rbp-16A0h]
  unsigned __int64 v1612; // [rsp+E8h] [rbp-1698h]
  unsigned __int64 v1613; // [rsp+E8h] [rbp-1698h]
  __int64 v1614; // [rsp+E8h] [rbp-1698h]
  unsigned __int64 v1615; // [rsp+E8h] [rbp-1698h]
  bool v1616; // [rsp+F0h] [rbp-1690h]
  __int64 *v1617; // [rsp+F0h] [rbp-1690h]
  __int64 v1618; // [rsp+F0h] [rbp-1690h]
  const void ***v1619; // [rsp+F0h] [rbp-1690h]
  __int64 v1620; // [rsp+F0h] [rbp-1690h]
  __int64 v1621; // [rsp+F0h] [rbp-1690h]
  __int64 v1622; // [rsp+F0h] [rbp-1690h]
  int v1623; // [rsp+F0h] [rbp-1690h]
  __int128 v1624; // [rsp+F0h] [rbp-1690h]
  __int64 v1625; // [rsp+F0h] [rbp-1690h]
  _QWORD *v1626; // [rsp+F0h] [rbp-1690h]
  unsigned int v1627; // [rsp+F0h] [rbp-1690h]
  __int64 v1628; // [rsp+F0h] [rbp-1690h]
  __int128 v1629; // [rsp+F0h] [rbp-1690h]
  __int128 v1630; // [rsp+F0h] [rbp-1690h]
  int v1631; // [rsp+F0h] [rbp-1690h]
  __int128 v1632; // [rsp+F0h] [rbp-1690h]
  unsigned int v1633; // [rsp+F0h] [rbp-1690h]
  __int64 v1634; // [rsp+F0h] [rbp-1690h]
  __int64 v1635; // [rsp+F0h] [rbp-1690h]
  __int64 *v1636; // [rsp+F0h] [rbp-1690h]
  __int64 v1637; // [rsp+F0h] [rbp-1690h]
  unsigned int v1638; // [rsp+F0h] [rbp-1690h]
  unsigned int v1639; // [rsp+F0h] [rbp-1690h]
  __int64 v1640; // [rsp+F0h] [rbp-1690h]
  __int64 v1641; // [rsp+F0h] [rbp-1690h]
  __int64 v1642; // [rsp+F0h] [rbp-1690h]
  __int64 v1643; // [rsp+F0h] [rbp-1690h]
  unsigned __int64 v1644; // [rsp+F0h] [rbp-1690h]
  __int64 v1645; // [rsp+F0h] [rbp-1690h]
  __int64 v1646; // [rsp+F0h] [rbp-1690h]
  unsigned __int64 v1647; // [rsp+F0h] [rbp-1690h]
  __int64 v1648; // [rsp+F0h] [rbp-1690h]
  __int64 v1649; // [rsp+F0h] [rbp-1690h]
  __int64 v1650; // [rsp+F0h] [rbp-1690h]
  __int16 *v1651; // [rsp+F0h] [rbp-1690h]
  __int64 v1652; // [rsp+F0h] [rbp-1690h]
  __int64 v1653; // [rsp+F0h] [rbp-1690h]
  __int64 v1654; // [rsp+F0h] [rbp-1690h]
  __int64 v1655; // [rsp+F0h] [rbp-1690h]
  __int64 v1656; // [rsp+F0h] [rbp-1690h]
  __int64 v1657; // [rsp+F0h] [rbp-1690h]
  __int64 v1658; // [rsp+F0h] [rbp-1690h]
  __int64 v1659; // [rsp+F0h] [rbp-1690h]
  __int64 v1660; // [rsp+F0h] [rbp-1690h]
  __int64 v1661; // [rsp+F0h] [rbp-1690h]
  __int64 v1662; // [rsp+F0h] [rbp-1690h]
  __int64 v1663; // [rsp+F0h] [rbp-1690h]
  __int64 v1664; // [rsp+F0h] [rbp-1690h]
  __int64 v1665; // [rsp+F0h] [rbp-1690h]
  __int64 v1666; // [rsp+F0h] [rbp-1690h]
  __int64 v1667; // [rsp+F0h] [rbp-1690h]
  __int64 v1668; // [rsp+F0h] [rbp-1690h]
  __int64 v1669; // [rsp+F0h] [rbp-1690h]
  __int64 v1670; // [rsp+F0h] [rbp-1690h]
  __int64 v1671; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1672; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1673; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1674; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1675; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1676; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1677; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1678; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1679; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1680; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1681; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1682; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1683; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1684; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1685; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1686; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1687; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1688; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1689; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1690; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1691; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1692; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1693; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1694; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1695; // [rsp+F8h] [rbp-1688h]
  unsigned __int64 v1696; // [rsp+F8h] [rbp-1688h]
  __int64 v1697; // [rsp+100h] [rbp-1680h]
  unsigned __int8 v1698; // [rsp+100h] [rbp-1680h]
  int v1699; // [rsp+100h] [rbp-1680h]
  __int64 *v1700; // [rsp+100h] [rbp-1680h]
  __int64 *v1701; // [rsp+100h] [rbp-1680h]
  __int128 v1702; // [rsp+100h] [rbp-1680h]
  _QWORD *v1703; // [rsp+100h] [rbp-1680h]
  __int64 *v1704; // [rsp+100h] [rbp-1680h]
  __int128 v1705; // [rsp+100h] [rbp-1680h]
  unsigned __int8 v1706; // [rsp+100h] [rbp-1680h]
  __int64 *v1707; // [rsp+100h] [rbp-1680h]
  __int64 *v1708; // [rsp+100h] [rbp-1680h]
  __int64 v1709; // [rsp+100h] [rbp-1680h]
  int v1710; // [rsp+100h] [rbp-1680h]
  __int64 v1711; // [rsp+100h] [rbp-1680h]
  __int128 v1712; // [rsp+100h] [rbp-1680h]
  __int64 v1713; // [rsp+100h] [rbp-1680h]
  unsigned int v1714; // [rsp+100h] [rbp-1680h]
  int v1715; // [rsp+100h] [rbp-1680h]
  __int64 *v1716; // [rsp+100h] [rbp-1680h]
  __int128 v1717; // [rsp+100h] [rbp-1680h]
  int v1718; // [rsp+100h] [rbp-1680h]
  __int128 v1719; // [rsp+100h] [rbp-1680h]
  __int128 v1720; // [rsp+100h] [rbp-1680h]
  __int64 *v1721; // [rsp+100h] [rbp-1680h]
  __int128 v1722; // [rsp+100h] [rbp-1680h]
  int v1723; // [rsp+100h] [rbp-1680h]
  __int128 v1724; // [rsp+100h] [rbp-1680h]
  __int128 v1725; // [rsp+100h] [rbp-1680h]
  unsigned int v1726; // [rsp+100h] [rbp-1680h]
  __int64 v1727; // [rsp+100h] [rbp-1680h]
  __int64 v1728; // [rsp+100h] [rbp-1680h]
  unsigned int v1729; // [rsp+100h] [rbp-1680h]
  __int64 *v1730; // [rsp+100h] [rbp-1680h]
  __int64 v1731; // [rsp+100h] [rbp-1680h]
  __int64 v1732; // [rsp+100h] [rbp-1680h]
  __int64 v1733; // [rsp+100h] [rbp-1680h]
  __int64 v1734; // [rsp+100h] [rbp-1680h]
  __int64 v1735; // [rsp+100h] [rbp-1680h]
  __int64 v1736; // [rsp+100h] [rbp-1680h]
  void *v1737; // [rsp+100h] [rbp-1680h]
  __int64 v1738; // [rsp+108h] [rbp-1678h]
  unsigned __int64 v1739; // [rsp+108h] [rbp-1678h]
  __int64 v1740; // [rsp+108h] [rbp-1678h]
  unsigned __int64 v1741; // [rsp+108h] [rbp-1678h]
  __int64 v1742; // [rsp+108h] [rbp-1678h]
  __int64 v1743; // [rsp+108h] [rbp-1678h]
  unsigned __int64 v1744; // [rsp+108h] [rbp-1678h]
  unsigned __int64 v1745; // [rsp+108h] [rbp-1678h]
  unsigned __int64 v1746; // [rsp+108h] [rbp-1678h]
  unsigned __int32 src; // [rsp+110h] [rbp-1670h]
  unsigned int srca; // [rsp+110h] [rbp-1670h]
  char srcbb; // [rsp+110h] [rbp-1670h]
  char *srcb; // [rsp+110h] [rbp-1670h]
  __int64 srcbc; // [rsp+110h] [rbp-1670h]
  __int128 srcbd; // [rsp+110h] [rbp-1670h]
  unsigned int srcbe; // [rsp+110h] [rbp-1670h]
  __int128 srcbf; // [rsp+110h] [rbp-1670h]
  __int64 srcbg; // [rsp+110h] [rbp-1670h]
  char srcc; // [rsp+110h] [rbp-1670h]
  __int128 srcd; // [rsp+110h] [rbp-1670h]
  __int64 srcbh; // [rsp+110h] [rbp-1670h]
  __int64 srce; // [rsp+110h] [rbp-1670h]
  __int128 srcbi; // [rsp+110h] [rbp-1670h]
  __int64 *srcbj; // [rsp+110h] [rbp-1670h]
  __int64 srcbk; // [rsp+110h] [rbp-1670h]
  const void **srcf; // [rsp+110h] [rbp-1670h]
  __int64 srcbl; // [rsp+110h] [rbp-1670h]
  __int64 *srcbm; // [rsp+110h] [rbp-1670h]
  __int128 srcbn; // [rsp+110h] [rbp-1670h]
  __int128 srcbo; // [rsp+110h] [rbp-1670h]
  __int64 *srcbp; // [rsp+110h] [rbp-1670h]
  __int128 srcbq; // [rsp+110h] [rbp-1670h]
  int srcg; // [rsp+110h] [rbp-1670h]
  int srch; // [rsp+110h] [rbp-1670h]
  __int64 srcbr; // [rsp+110h] [rbp-1670h]
  char srci; // [rsp+110h] [rbp-1670h]
  __int32 srcj; // [rsp+110h] [rbp-1670h]
  __int128 srcbs; // [rsp+110h] [rbp-1670h]
  __int64 srck; // [rsp+110h] [rbp-1670h]
  __int64 srcbt; // [rsp+110h] [rbp-1670h]
  _QWORD *srcbu; // [rsp+110h] [rbp-1670h]
  int srcbv; // [rsp+110h] [rbp-1670h]
  __int128 srcl; // [rsp+110h] [rbp-1670h]
  __int64 srcbw; // [rsp+110h] [rbp-1670h]
  __int64 srcbx; // [rsp+110h] [rbp-1670h]
  _QWORD *srcm; // [rsp+110h] [rbp-1670h]
  __int64 *srcn; // [rsp+110h] [rbp-1670h]
  __int64 *srcby; // [rsp+110h] [rbp-1670h]
  __int64 srco; // [rsp+110h] [rbp-1670h]
  __int64 srcbz; // [rsp+110h] [rbp-1670h]
  unsigned __int8 srcp; // [rsp+110h] [rbp-1670h]
  __int64 srcca; // [rsp+110h] [rbp-1670h]
  __int64 srccb; // [rsp+110h] [rbp-1670h]
  __int128 srccc; // [rsp+110h] [rbp-1670h]
  const void **srccd; // [rsp+110h] [rbp-1670h]
  const void **srcq; // [rsp+110h] [rbp-1670h]
  __int128 srcce; // [rsp+110h] [rbp-1670h]
  __int64 srcr; // [rsp+110h] [rbp-1670h]
  int srcs; // [rsp+110h] [rbp-1670h]
  __int64 srccf; // [rsp+110h] [rbp-1670h]
  unsigned __int8 srct; // [rsp+110h] [rbp-1670h]
  __int64 srccg; // [rsp+110h] [rbp-1670h]
  __int64 *srcch; // [rsp+110h] [rbp-1670h]
  __int128 srcci; // [rsp+110h] [rbp-1670h]
  __int128 srccj; // [rsp+110h] [rbp-1670h]
  __int64 srcck; // [rsp+110h] [rbp-1670h]
  int srccl; // [rsp+110h] [rbp-1670h]
  char srccm; // [rsp+110h] [rbp-1670h]
  __int64 *srccn; // [rsp+110h] [rbp-1670h]
  __int64 *srcco; // [rsp+110h] [rbp-1670h]
  unsigned __int64 srccp; // [rsp+110h] [rbp-1670h]
  __int128 srcu; // [rsp+110h] [rbp-1670h]
  __int64 srcv; // [rsp+110h] [rbp-1670h]
  __int64 srccq; // [rsp+110h] [rbp-1670h]
  unsigned int srcw; // [rsp+110h] [rbp-1670h]
  __int64 srccr; // [rsp+110h] [rbp-1670h]
  __int128 srcx; // [rsp+110h] [rbp-1670h]
  __int64 srcy; // [rsp+110h] [rbp-1670h]
  __int64 srccs; // [rsp+110h] [rbp-1670h]
  __int128 srcz; // [rsp+110h] [rbp-1670h]
  __int64 srcba; // [rsp+110h] [rbp-1670h]
  __int64 srcct; // [rsp+110h] [rbp-1670h]
  __int64 srccu; // [rsp+110h] [rbp-1670h]
  __int64 srccv; // [rsp+110h] [rbp-1670h]
  unsigned __int64 src_8; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8a; // [rsp+118h] [rbp-1668h]
  __int64 src_8h; // [rsp+118h] [rbp-1668h]
  __int64 src_8b; // [rsp+118h] [rbp-1668h]
  __int64 src_8c; // [rsp+118h] [rbp-1668h]
  __int32 src_8d; // [rsp+118h] [rbp-1668h]
  __int64 src_8i; // [rsp+118h] [rbp-1668h]
  __int16 *src_8j; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8e; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8k; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8f; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8l; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8g; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8m; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8n; // [rsp+118h] [rbp-1668h]
  unsigned __int64 src_8o; // [rsp+118h] [rbp-1668h]
  int v1838; // [rsp+198h] [rbp-15E8h]
  int v1839; // [rsp+1A8h] [rbp-15D8h]
  __int64 *v1840; // [rsp+200h] [rbp-1580h]
  __int32 v1841; // [rsp+208h] [rbp-1578h]
  __int64 *v1842; // [rsp+210h] [rbp-1570h]
  __int32 v1843; // [rsp+218h] [rbp-1568h]
  __int64 *v1844; // [rsp+220h] [rbp-1560h]
  __int32 v1845; // [rsp+228h] [rbp-1558h]
  int v1846; // [rsp+238h] [rbp-1548h]
  int v1847; // [rsp+268h] [rbp-1518h]
  int v1848; // [rsp+278h] [rbp-1508h]
  _QWORD *v1849; // [rsp+2B0h] [rbp-14D0h]
  __int64 *v1850; // [rsp+2D0h] [rbp-14B0h]
  __int32 v1851; // [rsp+2D8h] [rbp-14A8h]
  __int64 *v1852; // [rsp+2E0h] [rbp-14A0h]
  __int32 v1853; // [rsp+2E8h] [rbp-1498h]
  int v1854; // [rsp+318h] [rbp-1468h]
  int v1855; // [rsp+388h] [rbp-13F8h]
  int v1856; // [rsp+398h] [rbp-13E8h]
  int v1857; // [rsp+3A8h] [rbp-13D8h]
  int v1858; // [rsp+3C8h] [rbp-13B8h]
  int v1859; // [rsp+3D8h] [rbp-13A8h]
  int v1860; // [rsp+3E8h] [rbp-1398h]
  int v1861; // [rsp+418h] [rbp-1368h]
  int v1862; // [rsp+428h] [rbp-1358h]
  int v1863; // [rsp+438h] [rbp-1348h]
  int v1864; // [rsp+448h] [rbp-1338h]
  int v1865; // [rsp+458h] [rbp-1328h]
  int v1866; // [rsp+468h] [rbp-1318h]
  int v1867; // [rsp+478h] [rbp-1308h]
  int v1868; // [rsp+488h] [rbp-12F8h]
  int v1869; // [rsp+498h] [rbp-12E8h]
  int v1870; // [rsp+4D8h] [rbp-12A8h]
  int v1871; // [rsp+518h] [rbp-1268h]
  __int64 v1872; // [rsp+560h] [rbp-1220h]
  __int64 *v1873; // [rsp+570h] [rbp-1210h]
  int v1874; // [rsp+5E8h] [rbp-1198h]
  int v1875; // [rsp+5F8h] [rbp-1188h]
  int v1876; // [rsp+608h] [rbp-1178h]
  __int64 *v1877; // [rsp+618h] [rbp-1168h] BYREF
  __int64 v1878; // [rsp+620h] [rbp-1160h] BYREF
  int v1879; // [rsp+628h] [rbp-1158h]
  __int64 v1880[2]; // [rsp+630h] [rbp-1150h] BYREF
  __m128i v1881; // [rsp+640h] [rbp-1140h] BYREF
  __m128i v1882; // [rsp+650h] [rbp-1130h] BYREF
  __m128i v1883; // [rsp+660h] [rbp-1120h] BYREF
  __m128i v1884; // [rsp+670h] [rbp-1110h] BYREF
  __m128i v1885; // [rsp+680h] [rbp-1100h] BYREF
  __int64 v1886; // [rsp+690h] [rbp-10F0h]
  __m128i v1887; // [rsp+6A0h] [rbp-10E0h] BYREF
  __int64 v1888; // [rsp+6B0h] [rbp-10D0h]
  __m128i v1889; // [rsp+6C0h] [rbp-10C0h] BYREF
  _BYTE v1890[32]; // [rsp+6D0h] [rbp-10B0h] BYREF
  __m128i v1891; // [rsp+6F0h] [rbp-1090h] BYREF
  _BYTE v1892[64]; // [rsp+700h] [rbp-1080h] BYREF
  __m128i v1893; // [rsp+740h] [rbp-1040h] BYREF
  __int64 v1894[8]; // [rsp+750h] [rbp-1030h] BYREF
  __m128i v1895; // [rsp+790h] [rbp-FF0h] BYREF
  _OWORD v1896[12]; // [rsp+7A0h] [rbp-FE0h] BYREF
  __m128i v1897[6]; // [rsp+860h] [rbp-F20h] BYREF
  int v1898; // [rsp+8C0h] [rbp-EC0h]
  __int64 v1899; // [rsp+8C8h] [rbp-EB8h]
  char *v1900; // [rsp+8D0h] [rbp-EB0h]
  __int64 v1901; // [rsp+8D8h] [rbp-EA8h]
  char v1902; // [rsp+8E0h] [rbp-EA0h] BYREF
  _DWORD *v1903; // [rsp+8E8h] [rbp-E98h]
  int v1904; // [rsp+8F0h] [rbp-E90h]
  _BYTE *v1905; // [rsp+EE0h] [rbp-8A0h]
  __int64 v1906; // [rsp+EE8h] [rbp-898h]
  _BYTE v1907[512]; // [rsp+EF0h] [rbp-890h] BYREF
  _BYTE *v1908; // [rsp+10F0h] [rbp-690h]
  __int64 v1909; // [rsp+10F8h] [rbp-688h]
  _BYTE v1910[1536]; // [rsp+1100h] [rbp-680h] BYREF
  _BYTE *v1911; // [rsp+1700h] [rbp-80h]
  __int64 v1912; // [rsp+1708h] [rbp-78h]
  _BYTE v1913[112]; // [rsp+1710h] [rbp-70h] BYREF

  v9 = (__int64)a2;
  v10 = (__int64)a1;
  v12 = a1[69];
  v13 = *((_DWORD *)a1 + 134);
  v1878 = 0;
  v14 = *(__m128i **)(v12 + 16);
  v15 = *a1;
  v1879 = v13;
  if ( !v15 )
    goto LABEL_8;
  a2 = *(__int64 **)(v15 + 48);
  if ( &v1878 == (__int64 *)(v15 + 48) )
    goto LABEL_6;
  v1878 = *(_QWORD *)(v15 + 48);
  if ( !a2 || (a1 = &v1878, sub_1623A60((__int64)&v1878, (__int64)a2, 2), !*(_QWORD *)v10) )
  {
LABEL_8:
    v1877 = 0;
    goto LABEL_9;
  }
  a2 = *(__int64 **)(*(_QWORD *)v10 + 48LL);
LABEL_6:
  v1877 = a2;
  if ( a2 )
  {
    a1 = (__int64 *)&v1877;
    sub_1623A60((__int64)&v1877, (__int64)a2, 2);
  }
LABEL_9:
  if ( a3 > 0xDA )
  {
    if ( a3 == 4492 )
    {
      v1616 = 0;
      v1556 = 2;
      v23 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v24 = **(_QWORD **)(v9 + 24 * (1 - v23));
      v25 = 2;
    }
    else
    {
      if ( a3 > 0x118C )
      {
        if ( a3 == 6295 )
        {
LABEL_79:
          v17 = 0;
          goto LABEL_17;
        }
        if ( a3 - 7217 <= 0xC && ((1LL << ((unsigned __int8)a3 - 49)) & 0x1C67) != 0 )
        {
          v21 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
          v22 = *((unsigned __int16 *)v21 + 12);
          if ( v22 == 32 || v22 == 10 )
          {
            sub_2077400(v10, v9, a3, (__int64)&v1878, *(double *)a7.m128i_i64, a8, a9);
          }
          else
          {
            switch ( v20 )
            {
              case 0LL:
                v1038 = 7214;
                break;
              case 1LL:
                v1038 = 7215;
                break;
              case 2LL:
                v1038 = 7216;
                break;
              case 3LL:
              case 4LL:
              case 7LL:
              case 8LL:
              case 9LL:
              case 12LL:
                v1038 = 7226;
                break;
              case 5LL:
                v1038 = 7220;
                break;
              case 6LL:
                v1038 = 7221;
                break;
              case 10LL:
                v1038 = 7224;
                break;
              case 11LL:
                v1038 = 7225;
                break;
              default:
                goto LABEL_16;
            }
            v1039 = *(_QWORD *)(v10 + 552);
            *(__m128i *)((char *)v1897 + 8) = 0;
            src_8i = v19;
            v1897[0].m128i_i64[0] = (__int64)v21;
            v1897[1].m128i_i64[1] = 0;
            v1897[0].m128i_i32[2] = v19;
            v1040 = sub_1D38BB0(v1039, 0, (__int64)&v1878, 5, 0, 0, (__m128i)0LL, *(double *)a8.m128i_i64, a9, 0);
            v1041 = *(__int64 **)(v10 + 552);
            v1897[1].m128i_i64[0] = v1040;
            v1897[1].m128i_i32[2] = v1042;
            v1873 = sub_204D450(v1041, 0x2Au, 0, (__int64)&v1878, (__int64)v1897, 2, 0.0, *(double *)a8.m128i_i64, a9);
            v1739 = v1043 | src_8i & 0xFFFFFFFF00000000LL;
            srccg = *(_QWORD *)v9;
            v1044 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
            LOBYTE(v1045) = sub_204D4D0((__int64)v14, v1044, srccg);
            v1047 = v1046;
            *((_QWORD *)&v1486 + 1) = v1739;
            *(_QWORD *)&v1486 = v1873;
            v1633 = v1045;
            v1872 = sub_1D309E0(
                      *(__int64 **)(v10 + 552),
                      158,
                      (__int64)&v1878,
                      v1045,
                      v1046,
                      0,
                      0.0,
                      *(double *)a8.m128i_i64,
                      *(double *)a9.m128i_i64,
                      v1486);
            v1613 = v1048 | v1739 & 0xFFFFFFFF00000000LL;
            srcch = *(__int64 **)(v10 + 552);
            *(_QWORD *)&v1049 = sub_20685E0(
                                  v10,
                                  *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)),
                                  (__m128i)0LL,
                                  a8,
                                  a9);
            v1725 = v1049;
            v1050 = sub_1D38BB0(
                      *(_QWORD *)(v10 + 552),
                      v1038,
                      (__int64)&v1878,
                      5,
                      0,
                      0,
                      (__m128i)0LL,
                      *(double *)a8.m128i_i64,
                      a9,
                      0);
            v1052 = sub_1D3A900(
                      srcch,
                      0x2Bu,
                      (__int64)&v1878,
                      v1633,
                      v1047,
                      0,
                      (__m128)0LL,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1050,
                      v1051,
                      v1725,
                      v1872,
                      v1613);
            LODWORD(v1047) = v1053;
            v1895.m128i_i64[0] = v9;
            v1054 = sub_205F5C0(v10 + 8, v1895.m128i_i64);
            v1054[1] = (__int64)v1052;
            *((_DWORD *)v1054 + 4) = (_DWORD)v1047;
          }
          goto LABEL_79;
        }
        goto LABEL_16;
      }
      if ( a3 == 3660 )
      {
LABEL_91:
        v152 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        v1897[0].m128i_i64[0] = v9;
        v154 = v153;
        v155 = v152;
        v156 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
        v1854 = v154;
        v17 = 0;
        v156[1] = (__int64)v155;
        *((_DWORD *)v156 + 4) = v1854;
        goto LABEL_17;
      }
      if ( a3 - 4043 > 0xE || ((1LL << ((unsigned __int8)a3 + 53)) & 0x4011) == 0 )
      {
LABEL_16:
        sub_2077400(v10, v9, a3, (__int64)&v1878, *(double *)a7.m128i_i64, a8, a9);
        v17 = 0;
        goto LABEL_17;
      }
      v23 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v1616 = a3 == 4043 || a3 == 4057;
      v24 = **(_QWORD **)(v9 + 24 * (1 - v23));
      if ( v1616 )
      {
        v1556 = 1;
        v24 = *(_QWORD *)v9;
        v25 = 1;
      }
      else
      {
        v1556 = 2;
        v25 = 2;
      }
    }
    v1550 = *(__int64 **)(v9 + 24 * (v25 - v23));
    v26 = *v1550;
    v1889.m128i_i64[0] = (__int64)v1890;
    v1565 = v26;
    v1891.m128i_i64[0] = (__int64)v1892;
    v1891.m128i_i64[1] = 0x400000000LL;
    v1889.m128i_i64[1] = 0x400000000LL;
    v27 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
    sub_20C7CE0(v14, v27, v24, &v1891, &v1889, 0);
    src = v1891.m128i_u32[2];
    if ( !v1891.m128i_i32[2] )
    {
LABEL_438:
      if ( (_BYTE *)v1889.m128i_i64[0] != v1890 )
        _libc_free(v1889.m128i_u64[0]);
      v137 = (__m128i *)v1891.m128i_i64[0];
      if ( (_BYTE *)v1891.m128i_i64[0] == v1892 )
        goto LABEL_79;
      goto LABEL_78;
    }
    v28 = *(__int64 (**)())(v14->m128i_i64[0] + 1296);
    if ( v28 == sub_2043C70 )
    {
      if ( !v1616 )
      {
        v1542 = 2;
        LOBYTE(v14) = 0;
        v1529 = 0;
LABEL_35:
        v1885 = 0u;
        v1886 = 0;
        sub_14A8180(v9, v1885.m128i_i64, 0);
        goto LABEL_36;
      }
      v1885 = 0u;
      v1886 = 0;
      sub_14A8180(v9, v1885.m128i_i64, 0);
      v1163 = 1;
    }
    else
    {
      v1208 = ((__int64 (__fastcall *)(__m128i *, __int64))v28)(v14, v9);
      LOBYTE(v14) = (v1208 & 4) != 0;
      v1529 = v1208 & 4;
      if ( !v1616 )
      {
        v1542 = v1208 | 2;
        goto LABEL_35;
      }
      v1885 = 0u;
      v1606 = v1208 | 1;
      v1886 = 0;
      sub_14A8180(v9, v1885.m128i_i64, 0);
      v1163 = v1606;
      if ( v1529 )
      {
        v1542 = v1606;
        goto LABEL_36;
      }
    }
    v1164 = v1565;
    if ( *(_BYTE *)(v1565 + 8) == 16 )
      v1164 = **(_QWORD **)(v1565 + 16);
    if ( *(_DWORD *)(v1164 + 8) >> 8 == 101 )
      goto LABEL_559;
    v14 = *(__m128i **)(v10 + 568);
    if ( !v14 )
    {
      v1542 = v1163;
      v1529 = 0;
      goto LABEL_36;
    }
    v1604 = v1163;
    v1165 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
    v1166 = sub_127FA20(v1165, v24);
    v1167 = _mm_loadu_si128(&v1885);
    v1897[0].m128i_i64[0] = (__int64)v1550;
    v1897[0].m128i_i64[1] = (unsigned __int64)(v1166 + 7) >> 3;
    v1897[1] = v1167;
    v1897[2].m128i_i64[0] = v1886;
    v1168 = sub_134CBB0((__int64)v14, (__int64)v1897, 0);
    v1163 = v1604;
    LOBYTE(v14) = v1168;
    if ( v1168 )
    {
LABEL_559:
      v1605 = v1163;
      sub_204D410((__int64)v1880, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
      v1881.m128i_i64[0] = 0;
      v1881.m128i_i32[2] = 0;
      v1542 = v1605 | 0x20;
      v1530 = !v1616 || src > 0x40;
      if ( v1530 )
      {
        v1529 = 0;
        goto LABEL_38;
      }
      v1530 = 1;
      v1186 = 0;
      v1529 = 0;
      v1187 = *(_QWORD *)(v10 + 552) + 88LL;
LABEL_561:
      v1881.m128i_i64[0] = v1187;
      v1881.m128i_i32[2] = v1186;
      goto LABEL_39;
    }
    v1529 = 0;
    v1542 = v1604;
LABEL_36:
    sub_204D410((__int64)v1880, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
    v1881.m128i_i64[0] = 0;
    v1881.m128i_i32[2] = 0;
    v1530 = (unsigned __int8)v14 | (src > 0x40 || !v1616);
    if ( v1530 )
    {
      v1530 = 0;
LABEL_38:
      v1881.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1881.m128i_i32[2] = v29;
LABEL_39:
      v30.m128i_i64[0] = (__int64)sub_20685E0(
                                    v10,
                                    *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)),
                                    a7,
                                    a8,
                                    a9);
      v31 = *(_QWORD *)(v10 + 552);
      v32 = *(_QWORD *)(v31 + 32);
      v1882 = v30;
      v33 = sub_1E0A0C0(v32);
      v34 = sub_2046180(v33, 0);
      v35 = sub_1D38BB0(v31, a3, (__int64)v1880, v34, 0, 1, a7, *(double *)a8.m128i_i64, a9, 0);
      v1883.m128i_i32[2] = v36;
      v37 = *(_DWORD *)(v9 + 20);
      v1883.m128i_i64[0] = v35;
      v38 = src;
      v1884.m128i_i64[0] = (__int64)sub_20685E0(
                                      v10,
                                      *(__int64 **)(v9 + 24 * (!v1616 + 2LL - (v37 & 0xFFFFFFF))),
                                      a7,
                                      a8,
                                      a9);
      v1884.m128i_i64[1] = v39;
      v1535 = sub_20685E0(v10, v1550, a7, a8, a9);
      v1536 = v40;
      v41 = (unsigned __int8 *)(v1535[5] + 16LL * (unsigned int)v40);
      v1586 = (const void **)*((_QWORD *)v41 + 1);
      v1566 = *v41;
      v1547 = sub_15603A0((_QWORD *)(v9 + 56), v1556);
      v1893.m128i_i64[0] = (__int64)v1894;
      v1893.m128i_i64[1] = 0x400000000LL;
      v1897[0].m128i_i64[0] = 0;
      v1897[0].m128i_i32[2] = 0;
      sub_202F910((__int64)&v1893, src, v1897, (__int64)v1894, v42, v43);
      if ( src >= 0x40 )
        v38 = 64;
      v1895.m128i_i64[1] = 0x400000000LL;
      v1897[0].m128i_i64[0] = 0;
      v46 = 0;
      v1897[0].m128i_i32[2] = 0;
      v1895.m128i_i64[0] = (__int64)v1896;
      sub_202F910((__int64)&v1895, v38, v1897, (__int64)v1896, v44, v45);
      v51 = src;
      v1531 = v9;
      srca = 0;
      v1539 = v51 - 1;
      while ( 1 )
      {
        v78 = 16 * v46;
        v1897[0].m128i_i64[0] = (__int64)v1897[1].m128i_i64;
        v1897[0].m128i_i64[1] = 0x800000000LL;
        sub_1D23890((__int64)v1897, &v1881, v47, v48, v49, v50);
        sub_1D23890((__int64)v1897, &v1883, v79, v80, v81, v82);
        sub_1D23890((__int64)v1897, &v1882, v83, v84, v85, v86);
        if ( v1616 )
        {
          v1533 = *(__int64 **)(v10 + 552);
          *(_QWORD *)&v87 = sub_1D38BB0(
                              (__int64)v1533,
                              *(_QWORD *)(v1889.m128i_i64[0] + 8 * v46),
                              (__int64)v1880,
                              v1566,
                              v1586,
                              0,
                              a7,
                              *(double *)a8.m128i_i64,
                              a9,
                              0);
          v1887.m128i_i64[0] = (__int64)sub_1D332F0(
                                          v1533,
                                          52,
                                          (__int64)v1880,
                                          v1566,
                                          v1586,
                                          3u,
                                          *(double *)a7.m128i_i64,
                                          *(double *)a8.m128i_i64,
                                          a9,
                                          (__int64)v1535,
                                          v1536,
                                          v87);
          v1887.m128i_i64[1] = v88;
          sub_1D23890((__int64)v1897, &v1887, v88, v89, v90, v91);
          sub_1D23890((__int64)v1897, &v1884, v92, v93, v94, v95);
          v70 = sub_1D252B0(
                  *(_QWORD *)(v10 + 552),
                  *(unsigned int *)(v1891.m128i_i64[0] + 16 * v46),
                  *(_QWORD *)(v1891.m128i_i64[0] + 16 * v46 + 8),
                  1,
                  0);
        }
        else
        {
          v52 = sub_20685E0(v10, *(__int64 **)(v1531 + 24 * (1LL - (*(_DWORD *)(v1531 + 20) & 0xFFFFFFF))), a7, a8, a9);
          v54 = (unsigned int)(v46 + v53);
          v1887.m128i_i64[0] = (__int64)v52;
          v1887.m128i_i32[2] = v54;
          sub_1D23890((__int64)v1897, &v1887, v54, v55, v56, v57);
          v1532 = *(__int64 **)(v10 + 552);
          *(_QWORD *)&v58 = sub_1D38BB0(
                              (__int64)v1532,
                              *(_QWORD *)(v1889.m128i_i64[0] + 8 * v46),
                              (__int64)v1880,
                              v1566,
                              v1586,
                              0,
                              a7,
                              *(double *)a8.m128i_i64,
                              a9,
                              0);
          v1887.m128i_i64[0] = (__int64)sub_1D332F0(
                                          v1532,
                                          52,
                                          (__int64)v1880,
                                          v1566,
                                          v1586,
                                          3u,
                                          *(double *)a7.m128i_i64,
                                          *(double *)a8.m128i_i64,
                                          a9,
                                          (__int64)v1535,
                                          v1536,
                                          v58);
          v1887.m128i_i64[1] = v59;
          sub_1D23890((__int64)v1897, &v1887, v59, v60, v61, v62);
          sub_1D23890((__int64)v1897, &v1884, v63, v64, v65, v66);
          v70 = sub_1D29190(*(_QWORD *)(v10 + 552), 1u, 0, v67, v68, v69);
        }
        v96 = v70;
        v72 = *(_QWORD **)(v10 + 552);
        v73 = *(_QWORD *)(v1889.m128i_i64[0] + 8 * v46);
        v1887.m128i_i64[0] = (__int64)v1550;
        LOBYTE(v1888) = 0;
        v1887.m128i_i64[1] = v73;
        v74 = *v1550;
        if ( *(_BYTE *)(*v1550 + 8) == 16 )
          v74 = **(_QWORD **)(v74 + 16);
        HIDWORD(v1888) = *(_DWORD *)(v74 + 8) >> 8;
        v48 = sub_1D251C0(
                v72,
                44,
                (__int64)v1880,
                v96,
                v71,
                v1547,
                (__int64 *)v1897[0].m128i_i64[0],
                v1897[0].m128i_u32[2],
                *(_QWORD *)(v78 + v1891.m128i_i64[0]),
                *(_QWORD *)(v78 + v1891.m128i_i64[0] + 8),
                *(_OWORD *)&v1887,
                v1888,
                v1542,
                0,
                (__int64)&v1885);
        if ( v1616 )
        {
          v76 = v1893.m128i_i64[0];
          *(_QWORD *)(v1893.m128i_i64[0] + 16 * v46) = v48;
          *(_DWORD *)(v76 + v78 + 8) = v75;
        }
        v47 = (unsigned int)(*(_DWORD *)(v48 + 60) - 1);
        v77 = v1895.m128i_i64[0] + 16LL * srca;
        *(_QWORD *)v77 = v48;
        *(_DWORD *)(v77 + 8) = v47;
        if ( (__m128i *)v1897[0].m128i_i64[0] != &v1897[1] )
          _libc_free(v1897[0].m128i_u64[0]);
        ++srca;
        if ( v46 == v1539 )
          break;
        if ( srca == 64 )
        {
          *((_QWORD *)&v1513 + 1) = 64;
          *(_QWORD *)&v1513 = v1895.m128i_i64[0];
          srca = 0;
          v1881.m128i_i64[0] = (__int64)sub_1D359D0(
                                          *(__int64 **)(v10 + 552),
                                          2,
                                          (__int64)v1880,
                                          1,
                                          0,
                                          0,
                                          *(double *)a7.m128i_i64,
                                          *(double *)a8.m128i_i64,
                                          a9,
                                          v1513);
          v1881.m128i_i32[2] = v1069;
          v47 = v1528;
        }
        ++v46;
      }
      v1071 = *(__int64 **)(v10 + 552);
      if ( v1616 )
      {
        if ( !v1530 )
        {
          *((_QWORD *)&v1519 + 1) = srca;
          *(_QWORD *)&v1519 = v1895.m128i_i64[0];
          v1188.m128i_i64[0] = (__int64)sub_1D359D0(
                                          v1071,
                                          2,
                                          (__int64)v1880,
                                          1,
                                          0,
                                          0,
                                          *(double *)a7.m128i_i64,
                                          *(double *)a8.m128i_i64,
                                          a9,
                                          v1519);
          v1897[0] = v1188;
          if ( v1529 )
            sub_2045100(*(_QWORD *)(v10 + 552), v1897[0].m128i_i64[0], v1897[0].m128i_i32[2]);
          else
            sub_1D23890(v10 + 104, v1897, v1188.m128i_i64[1], v1189, v1190, v1191);
          v1071 = *(__int64 **)(v10 + 552);
        }
        *(_QWORD *)&srccj = v1893.m128i_i64[0];
        *((_QWORD *)&srccj + 1) = v1893.m128i_u32[2];
        v1072 = (const void ***)sub_1D25C30((__int64)v1071, (unsigned __int8 *)v1891.m128i_i64[0], v1891.m128i_u32[2]);
        v1075 = sub_1D36D80(
                  v1071,
                  51,
                  (__int64)v1880,
                  v1072,
                  v1073,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  v1074,
                  srccj);
        v1076 = v10 + 8;
        v1897[0].m128i_i64[0] = v1531;
        v1077 = v1075;
        v1079 = v1078;
        v1080 = sub_205F5C0(v1076, v1897[0].m128i_i64);
        v1080[1] = (__int64)v1077;
        *((_DWORD *)v1080 + 4) = v1079;
      }
      else
      {
        *((_QWORD *)&v1516 + 1) = srca;
        *(_QWORD *)&v1516 = v1895.m128i_i64[0];
        v1169 = sub_1D359D0(
                  v1071,
                  2,
                  (__int64)v1880,
                  1,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  v1516);
        srccl = v1170;
        sub_2045100(*(_QWORD *)(v10 + 552), (__int64)v1169, v1170);
        v1897[0].m128i_i64[0] = v1531;
        v1171 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
        v1171[1] = (__int64)v1169;
        *((_DWORD *)v1171 + 4) = srccl;
      }
      if ( (_OWORD *)v1895.m128i_i64[0] != v1896 )
        _libc_free(v1895.m128i_u64[0]);
      if ( (__int64 *)v1893.m128i_i64[0] != v1894 )
        _libc_free(v1893.m128i_u64[0]);
      sub_17CD270(v1880);
      goto LABEL_438;
    }
    v1215 = *(_QWORD *)(v10 + 552);
    v1187 = *(_QWORD *)(v1215 + 176);
    v1186 = *(_DWORD *)(v1215 + 184);
    goto LABEL_561;
  }
  if ( !a3 )
    goto LABEL_16;
  v16 = a3;
  switch ( a3 )
  {
    case 1u:
      v881 = *(_QWORD **)(v10 + 552);
      v882 = sub_1E0A0C0(v881[4]);
      v883 = sub_2046180(v882, 0);
      v885 = sub_1D2B300(v881, 0x16u, (__int64)&v1878, v883, 0, v884);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v881) = v886;
      v887 = v885;
      v888 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1875 = (int)v881;
      v17 = 0;
      v888[1] = (__int64)v887;
      *((_DWORD *)v888 + 4) = v1875;
      goto LABEL_17;
    case 2u:
      v1028 = *(__int64 **)(v10 + 552);
      v1029 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1031 = v1030;
      v1032 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v1033 = sub_2046180(v1032, 0);
      *((_QWORD *)&v1512 + 1) = v1031;
      *(_QWORD *)&v1512 = v1029;
      v1034 = sub_1D309E0(
                v1028,
                214,
                (__int64)&v1878,
                v1033,
                0,
                0,
                *(double *)a7.m128i_i64,
                *(double *)a8.m128i_i64,
                *(double *)a9.m128i_i64,
                v1512);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v1029) = v1035;
      v1036 = v1034;
      v1037 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1037[1] = v1036;
      *((_DWORD *)v1037 + 4) = (_DWORD)v1029;
      v17 = 0;
      goto LABEL_17;
    case 3u:
    case 0x73u:
    case 0x95u:
    case 0xCBu:
      goto LABEL_91;
    case 4u:
    case 0x28u:
    case 0x71u:
    case 0xBFu:
    case 0xD7u:
      goto LABEL_79;
    case 5u:
      v1017 = *(__int64 **)(v10 + 552);
      v1018 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1020 = v1019;
      v1021 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1023 = (const void ***)(v1021[5] + 16LL * v1022);
      *((_QWORD *)&v1511 + 1) = v1020;
      *(_QWORD *)&v1511 = v1018;
      v1024 = sub_1D309E0(
                v1017,
                131,
                (__int64)&v1878,
                *(unsigned __int8 *)v1023,
                v1023[1],
                0,
                *(double *)a7.m128i_i64,
                *(double *)a8.m128i_i64,
                *(double *)a9.m128i_i64,
                v1511);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v1018) = v1025;
      v1026 = v1024;
      v1027 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1858 = (int)v1018;
      v17 = 0;
      v1027[1] = v1026;
      *((_DWORD *)v1027 + 4) = v1858;
      goto LABEL_17;
    case 6u:
      v731 = *(__int64 **)(v10 + 552);
      v732 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v734 = v733;
      v735 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v737 = (const void ***)(v735[5] + 16LL * v736);
      *((_QWORD *)&v1507 + 1) = v734;
      *(_QWORD *)&v1507 = v732;
      v738 = sub_1D309E0(
               v731,
               127,
               (__int64)&v1878,
               *(unsigned __int8 *)v737,
               v737[1],
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1507);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v732) = v739;
      v740 = v738;
      v741 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v741[1] = v740;
      *((_DWORD *)v741 + 4) = (_DWORD)v732;
      v17 = 0;
      goto LABEL_17;
    case 7u:
    case 8u:
    case 0x1Eu:
    case 0x60u:
    case 0x61u:
    case 0x8Cu:
    case 0xBBu:
    case 0xBCu:
    case 0xC2u:
    case 0xC4u:
    case 0xCEu:
      if ( a3 == 140 )
      {
        v140 = 177;
      }
      else
      {
        if ( a3 > 0x8C )
        {
          switch ( a3 )
          {
            case 0xBBu:
              v140 = 176;
              goto LABEL_89;
            case 0xBCu:
              v140 = 178;
              goto LABEL_89;
            case 0xC2u:
              v140 = 165;
              goto LABEL_89;
            case 0xC4u:
              v140 = 164;
              goto LABEL_89;
            case 0xCEu:
              v140 = 175;
              goto LABEL_89;
            default:
              goto LABEL_716;
          }
        }
        if ( a3 == 30 )
        {
          v140 = 166;
        }
        else if ( a3 <= 0x1E )
        {
          v1070 = 103;
          if ( a3 != 7 )
            v1070 = 174;
          v140 = v1070;
        }
        else
        {
          v139 = 163;
          if ( a3 != 96 )
            v139 = 179;
          v140 = v139;
        }
      }
LABEL_89:
      srcbe = v140;
      v141 = *(__int64 **)(v10 + 552);
      v142 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v144 = v143;
      v145 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v147 = (const void ***)(v145[5] + 16LL * v146);
      *((_QWORD *)&v1489 + 1) = v144;
      *(_QWORD *)&v1489 = v142;
      v148 = sub_1D309E0(
               v141,
               srcbe,
               (__int64)&v1878,
               *(unsigned __int8 *)v147,
               v147[1],
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1489);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v142) = v149;
      v150 = v148;
      v151 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1865 = (int)v142;
      v17 = 0;
      v151[1] = v150;
      *((_DWORD *)v151 + 4) = v1865;
      goto LABEL_17;
    case 9u:
      v17 = "__clear_cache";
      v538 = *(char *(**)())(v14->m128i_i64[0] + 1248);
      if ( v538 != sub_2043C50 )
        v17 = (char *)((__int64 (__fastcall *)(__m128i *))v538)(v14);
      goto LABEL_17;
    case 0xAu:
      v333 = *(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL);
      v334 = *(_QWORD *)(v333 + 32);
      v1897[1].m128i_i16[0] = 259;
      v1897[0].m128i_i64[0] = (__int64)"annotation";
      v335 = sub_38BF8E0(v334 + 168, v1897, 1, 1);
      v336 = *(_QWORD *)(*(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)) + 24LL);
      v1897[0].m128i_i64[0] = v335;
      v1897[0].m128i_i64[1] = v336;
      v337 = *(__m128i **)(v333 + 504);
      if ( v337 == *(__m128i **)(v333 + 512) )
      {
        sub_205B270((const __m128i **)(v333 + 496), v337, v1897);
      }
      else
      {
        if ( v337 )
        {
          *v337 = _mm_loadu_si128(v1897);
          v337 = *(__m128i **)(v333 + 504);
        }
        *(_QWORD *)(v333 + 504) = v337 + 1;
      }
      v338 = *(_QWORD **)(v10 + 552);
      v339 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v340 = v335;
      v17 = 0;
      v342 = sub_1D2A890(v338, 0xC3u, (__int64)&v1878, (__int64)v339, v341, v340);
      sub_2045100(*(_QWORD *)(v10 + 552), (__int64)v342, v343);
      goto LABEL_17;
    case 0xBu:
      srcbj = *(__int64 **)(v10 + 552);
      *(_QWORD *)&v322 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1704 = srcbj;
      v323 = sub_1D309E0(
               srcbj,
               158,
               (__int64)&v1878,
               8,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v322);
      v325 = v324;
      srcbk = *(_QWORD *)v9;
      v326 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      LOBYTE(v327) = sub_204D4D0((__int64)v14, v326, srcbk);
      *((_QWORD *)&v1483 + 1) = v325;
      *(_QWORD *)&v1483 = v323;
      v329 = sub_1D309E0(
               v1704,
               157,
               (__int64)&v1878,
               v327,
               v328,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1483);
      LODWORD(v323) = v330;
      v331 = v329;
      v1897[0].m128i_i64[0] = v9;
      v332 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1859 = v323;
      v17 = 0;
      v332[1] = v331;
      *((_DWORD *)v332 + 4) = v1859;
      goto LABEL_17;
    case 0xCu:
      v310 = *(__int64 **)(v10 + 552);
      v311 = sub_1D38BB0((__int64)v310, 0, (__int64)&v1878, 5, 0, 1, a7, *(double *)a8.m128i_i64, a9, 0);
      v313 = v312;
      v314 = v311;
      v315 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      *((_QWORD *)&v1482 + 1) = v313;
      *(_QWORD *)&v1482 = v314;
      *(_QWORD *)&v317 = sub_1D332F0(
                           v310,
                           154,
                           (__int64)&v1878,
                           8,
                           0,
                           0,
                           *(double *)a7.m128i_i64,
                           *(double *)a8.m128i_i64,
                           a9,
                           (__int64)v315,
                           v316,
                           v1482);
      v318 = sub_1D309E0(
               v310,
               158,
               (__int64)&v1878,
               4,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v317);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v314) = v319;
      v320 = v318;
      v321 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1860 = v314;
      v17 = 0;
      v321[1] = v320;
      *((_DWORD *)v321 + 4) = v1860;
      goto LABEL_17;
    case 0xDu:
      v298 = *(__int64 **)(v10 + 552);
      *(_QWORD *)&srcbi = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&srcbi + 1) = v299;
      v300 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v302 = v301;
      v303 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v305 = (const void ***)(v303[5] + 16LL * v304);
      v306 = sub_1D332F0(
               v298,
               101,
               (__int64)&v1878,
               *(unsigned __int8 *)v305,
               v305[1],
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v300,
               v302,
               srcbi);
      LODWORD(v300) = v307;
      v308 = v306;
      v1897[0].m128i_i64[0] = v9;
      v309 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1862 = (int)v300;
      v17 = 0;
      v309[1] = (__int64)v308;
      *((_DWORD *)v309 + 4) = v1862;
      goto LABEL_17;
    case 0x1Fu:
      v548 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v549 = *(__int64 **)(v10 + 552);
      v551 = v550;
      v552 = v548;
      v553 = (unsigned __int8 *)(v548[5] + 16LL * (unsigned int)v550);
      v554 = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v555 = (const void **)*((_QWORD *)v553 + 1);
      v556 = *v553;
      if ( *(_DWORD *)(v554 + 32) <= 0x40u )
      {
        v558 = *(_QWORD *)(v554 + 24) == 0 ? 129 : 133;
      }
      else
      {
        v1622 = *v553;
        v1710 = *(_DWORD *)(v554 + 32);
        srcbp = *(__int64 **)(v10 + 552);
        v557 = sub_16A57B0(v554 + 24);
        v556 = v1622;
        v549 = srcbp;
        v558 = 4 * (unsigned int)(v1710 != v557) + 129;
      }
      *((_QWORD *)&v1503 + 1) = v551;
      *(_QWORD *)&v1503 = v552;
      v559 = sub_1D309E0(
               v549,
               v558,
               (__int64)&v1878,
               v556,
               v555,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1503);
      v1897[0].m128i_i64[0] = v9;
      v561 = v560;
      v562 = v559;
      v563 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1856 = v561;
      v17 = 0;
      v563[1] = v562;
      *((_DWORD *)v563 + 4) = v1856;
      goto LABEL_17;
    case 0x20u:
      v539 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v541 = v540;
      v542 = (const void ***)(v539[5] + 16LL * (unsigned int)v540);
      *((_QWORD *)&v1502 + 1) = v541;
      *(_QWORD *)&v1502 = v539;
      v543 = sub_1D309E0(
               *(__int64 **)(v10 + 552),
               130,
               (__int64)&v1878,
               *(unsigned __int8 *)v542,
               v542[1],
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1502);
      v1897[0].m128i_i64[0] = v9;
      v545 = v544;
      v546 = v543;
      v547 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1855 = v545;
      v17 = 0;
      v547[1] = v546;
      *((_DWORD *)v547 + 4) = v1855;
      goto LABEL_17;
    case 0x21u:
      v819 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v820 = *(__int64 **)(v10 + 552);
      v822 = v821;
      v823 = v819;
      v824 = (unsigned __int8 *)(v819[5] + 16LL * (unsigned int)v821);
      v825 = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v826 = (const void **)*((_QWORD *)v824 + 1);
      v827 = *v824;
      if ( *(_DWORD *)(v825 + 32) <= 0x40u )
      {
        v829 = *(_QWORD *)(v825 + 24) == 0 ? 128 : 132;
      }
      else
      {
        v1628 = *v824;
        v1718 = *(_DWORD *)(v825 + 32);
        srcby = *(__int64 **)(v10 + 552);
        v828 = sub_16A57B0(v825 + 24);
        v827 = v1628;
        v820 = srcby;
        v829 = 4 * (unsigned int)(v1718 != v828) + 128;
      }
      *((_QWORD *)&v1510 + 1) = v822;
      *(_QWORD *)&v1510 = v823;
      v830 = sub_1D309E0(
               v820,
               v829,
               (__int64)&v1878,
               v827,
               v826,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1510);
      v1897[0].m128i_i64[0] = v9;
      v831 = v830;
      v833 = v832;
      v834 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1857 = v833;
      v17 = 0;
      v834[1] = v831;
      *((_DWORD *)v834 + 4) = v1857;
      goto LABEL_17;
    case 0x23u:
    case 0x24u:
      v192 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v193 = *(_QWORD *)(*(_QWORD *)(v9 + 24 * (1 - v192)) + 24LL);
      v194 = *(_QWORD **)(*(_QWORD *)(v9 + 24 * (2 - v192)) + 24LL);
      sub_2052000(v10, v193, (__int64)v194);
      v195 = sub_1601A30(v9, 1);
      v1895.m128i_i64[0] = v195;
      if ( !v195 )
        goto LABEL_79;
      v196 = *(_BYTE *)(v195 + 16);
      if ( v196 == 9 || !*(_QWORD *)(v195 + 8) && v196 != 17 )
        goto LABEL_79;
      srcc = v196 == 17 || *(_WORD *)(v193 + 32) != 0;
      v1897[0].m128i_i32[2] = sub_127FA20(*(_QWORD *)(v10 + 560), *(_QWORD *)v195);
      if ( v1897[0].m128i_i32[2] > 0x40u )
        sub_16A4EF0((__int64)v1897, 0, 0);
      else
        v1897[0].m128i_i64[0] = 0;
      v197 = sub_164A410(v1895.m128i_i64[0], *(_QWORD *)(v10 + 560), (__int64)v1897);
      v198 = *(_BYTE *)(v197 + 16);
      if ( v198 <= 0x17u )
      {
        if ( v198 != 17 )
          goto LABEL_109;
        v1154 = sub_1FDEA40(*(_QWORD *)(v10 + 712), v197);
      }
      else
      {
        if ( v198 != 53 )
          goto LABEL_109;
        if ( !(unsigned __int8)sub_15F8F00(v197) )
          goto LABEL_109;
        v1218 = *(_QWORD *)(v10 + 712);
        v1219 = *(_DWORD *)(v1218 + 360);
        if ( !v1219 )
          goto LABEL_109;
        v1220 = *(_QWORD *)(v1218 + 344);
        v1221 = 1;
        for ( i = (v1219 - 1) & (((unsigned int)v197 >> 4) ^ ((unsigned int)v197 >> 9)); ; i = (v1219 - 1) & v1477 )
        {
          v1223 = v1220 + 16LL * i;
          if ( v197 == *(_QWORD *)v1223 )
            break;
          if ( *(_QWORD *)v1223 == -8 )
            goto LABEL_109;
          v1477 = v1221 + i;
          ++v1221;
        }
        if ( v1223 == v1220 + 16LL * v1219 )
          goto LABEL_109;
        v1154 = *(_DWORD *)(v1223 + 8);
      }
      if ( v1154 != 0x7FFFFFFF )
      {
        if ( a3 != 35 )
        {
LABEL_117:
          sub_135E100(v1897[0].m128i_i64);
          goto LABEL_79;
        }
        v1155 = v1897[0].m128i_i32[2];
        if ( v1897[0].m128i_i32[2] <= 0x40u )
        {
          v1156 = v1897[0].m128i_i64[0];
          if ( !v1897[0].m128i_i64[0] )
            goto LABEL_532;
        }
        else
        {
          if ( v1155 == (unsigned int)sub_16A57B0((__int64)v1897) )
          {
LABEL_532:
            v1157 = sub_1D240D0(
                      *(_QWORD *)(v10 + 552),
                      v193,
                      (__int64)v194,
                      v1154,
                      1,
                      (__int64 *)&v1877,
                      *(_DWORD *)(v10 + 536));
            v1158 = *(_QWORD *)(v10 + 552);
            v1159 = v1157;
            v1160 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
            sub_1D30360(v1158, v1159, (__int64)v1160, srcc, v1161, v1162);
            goto LABEL_117;
          }
          v1156 = *(_QWORD *)v1897[0].m128i_i64[0];
        }
        v194 = (_QWORD *)sub_15C48E0(v194, 0, v1156, 0, 0);
        goto LABEL_532;
      }
LABEL_109:
      v199 = sub_205F5C0(v10 + 8, v1895.m128i_i64);
      v200 = v199[1];
      v201 = v199;
      v202 = v199 + 1;
      if ( !v200 )
      {
        if ( *(_BYTE *)(v1895.m128i_i64[0] + 16) != 17 )
          goto LABEL_568;
        v1730 = v199 + 1;
        v1192 = sub_205F5C0(v10 + 40, v1895.m128i_i64);
        v202 = v1730;
        v200 = v1192[1];
        v201[1] = v200;
        *((_DWORD *)v201 + 4) = *((_DWORD *)v1192 + 4);
        if ( !v200 )
          goto LABEL_568;
      }
      if ( *(_BYTE *)(v1895.m128i_i64[0] + 16) == 71 )
        v1895.m128i_i64[0] = *(_QWORD *)(v1895.m128i_i64[0] - 24);
      v203 = *(unsigned __int16 *)(v200 + 24);
      if ( (v203 == 14 || v203 == 36) && srcc )
      {
        v1185 = sub_1D240D0(
                  *(_QWORD *)(v10 + 552),
                  v193,
                  (__int64)v194,
                  *(_DWORD *)(v200 + 84),
                  1,
                  (__int64 *)&v1877,
                  *(_DWORD *)(v10 + 536));
        v205 = v1518;
        v207 = v1185;
      }
      else
      {
        if ( *(_BYTE *)(v1895.m128i_i64[0] + 16) == 17 )
        {
LABEL_568:
          srccn = v202;
          v1193 = sub_15C70A0((__int64)&v1877);
          sub_2054040(v10, v1895.m128i_i64[0], v193, v194, v1193, 1u, srccn);
          goto LABEL_117;
        }
        v204 = sub_1D24380(
                 *(_QWORD *)(v10 + 552),
                 v193,
                 (__int64)v194,
                 v200,
                 *((_DWORD *)v201 + 4),
                 1,
                 (__int64 *)&v1877,
                 *(_DWORD *)(v10 + 536));
        v206 = v1527;
        v207 = (__int64)v204;
      }
      sub_1D30360(*(_QWORD *)(v10 + 552), v207, v201[1], srcc, v206, v205);
      goto LABEL_117;
    case 0x25u:
      v17 = 0;
      v742 = sub_1D23D40(
               *(_QWORD *)(v10 + 552),
               *(_QWORD *)(*(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)) + 24LL),
               (__int64 *)&v1877,
               *(_DWORD *)(v10 + 536));
      sub_1D18540(*(_QWORD *)(v10 + 552), v742, v743, v744, v745, v746);
      goto LABEL_17;
    case 0x26u:
      v835 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v836 = *(_QWORD *)(*(_QWORD *)(v9 + 24 * (1 - v835)) + 24LL);
      v837 = *(_QWORD *)(*(_QWORD *)(v9 + 24 * (2 - v835)) + 24LL);
      sub_2052000(v10, v836, v837);
      v838 = sub_1601A30(v9, 0);
      v1889.m128i_i64[0] = v838;
      v839 = v838;
      if ( !v838 )
        goto LABEL_79;
      v840 = *(_BYTE *)(v838 + 16);
      if ( (unsigned __int8)(v840 - 13) <= 1u || v840 == 9 )
      {
        v1122 = sub_1D242B0(*(_QWORD *)(v10 + 552), v836, v837, v839, (__int64 *)&v1877, *(_DWORD *)(v10 + 536));
        sub_1D30360(*(_QWORD *)(v10 + 552), (__int64)v1122, 0, 0, v1123, v1124);
        goto LABEL_79;
      }
      v1891 = _mm_loadu_si128((const __m128i *)(sub_205F5C0(v10 + 8, v1889.m128i_i64) + 1));
      if ( v1891.m128i_i64[0] )
        goto LABEL_330;
      if ( *(_BYTE *)(v1889.m128i_i64[0] + 16) == 77 )
      {
        if ( (unsigned __int8)sub_1FDDF40(*(_QWORD *)(v10 + 712), v1889.m128i_i64[0]) )
        {
          v1891.m128i_i64[0] = (__int64)sub_20685E0(v10, (__int64 *)v1889.m128i_i64[0], a7, a8, a9);
          v1891.m128i_i32[2] = v1452;
          v1453.m128i_i64[0] = sub_2043E60(v1891.m128i_i64[0]);
          v1454 = *(__m128i **)(v10 + 552);
          v1455 = _mm_loadu_si128(v1454 + 11);
          v1897[1] = v1453;
          *((_QWORD *)&v1526 + 1) = 2;
          *(_QWORD *)&v1526 = v1897;
          v1897[0] = v1455;
          v1460 = sub_1D359D0(
                    v1454->m128i_i64,
                    2,
                    v1459,
                    v1456,
                    v1457,
                    v1458,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    v1526);
          sub_2045100((__int64)v1454, (__int64)v1460, v1461);
        }
        if ( v1891.m128i_i64[0] )
          goto LABEL_330;
        v1252 = v1889.m128i_i64[0];
        if ( *(_BYTE *)(v1889.m128i_i64[0] + 16) != 17 )
          goto LABEL_614;
      }
      else
      {
        v1209 = v1889.m128i_i64[0];
        if ( *(_BYTE *)(v1889.m128i_i64[0] + 16) != 17 )
          goto LABEL_577;
      }
      v1253 = sub_205F5C0(v10 + 40, v1889.m128i_i64);
      v1891.m128i_i64[0] = v1253[1];
      v1891.m128i_i32[2] = *((_DWORD *)v1253 + 4);
      if ( !v1891.m128i_i64[0] )
      {
        v1252 = v1889.m128i_i64[0];
LABEL_614:
        v1209 = v1252;
        if ( *(_BYTE *)(v1252 + 16) == 77 )
        {
          v1254 = *(_QWORD *)(v10 + 712);
          v1255 = *(_DWORD *)(v1254 + 232);
          if ( v1255 )
          {
            v1256 = *(_QWORD *)(v1254 + 216);
            v1257 = 1;
            for ( j = (v1255 - 1) & (((unsigned int)v1252 >> 9) ^ ((unsigned int)v1252 >> 4)); ; j = (v1255 - 1) & v1476 )
            {
              v1259 = v1256 + 16LL * j;
              if ( v1252 == *(_QWORD *)v1259 )
                break;
              if ( *(_QWORD *)v1259 == -8 )
                goto LABEL_577;
              v1476 = v1257 + j;
              ++v1257;
            }
            if ( v1259 != v1256 + 16LL * v1255 )
            {
              v1260 = *(_QWORD *)(v10 + 552);
              v1261 = *(_DWORD *)(v1259 + 8);
              v1895.m128i_i8[4] = 0;
              v1733 = *(_QWORD *)v1252;
              srccq = sub_1E0A0C0(*(_QWORD *)(v1260 + 32));
              v1262 = sub_16498A0(v1889.m128i_i64[0]);
              sub_204E3C0((__int64)v1897, v1262, (__int64)v14, srccq, v1261, v1733, (unsigned int *)&v1895);
              v1263 = v1903;
              v1264 = &v1903[v1904];
              if ( v1903 == v1264 )
                goto LABEL_706;
              v1265 = 0;
              do
                v1265 += *v1263++;
              while ( v1264 != v1263 );
              if ( v1265 <= 1 )
              {
LABEL_706:
                v1473 = sub_1D241C0(
                          *(_QWORD *)(v10 + 552),
                          v836,
                          v837,
                          v1261,
                          0,
                          (__int64 *)&v1877,
                          *(_DWORD *)(v10 + 536));
                sub_1D30360(*(_QWORD *)(v10 + 552), v1473, 0, 0, v1474, v1475);
              }
              else
              {
                v1266 = 0;
                sub_15B1130((__int64)&v1895, v836);
                if ( v1895.m128i_i8[8] )
                  v1266 = v1895.m128i_i32[0];
                sub_15B1350((__int64)&v1895, *(unsigned __int64 **)(v837 + 24), *(unsigned __int64 **)(v837 + 32));
                if ( LOBYTE(v1896[0]) )
                  v1266 = v1895.m128i_i32[0];
                v1269 = v836;
                v1267 = v10;
                v1268 = v1269;
                sub_20505F0((__int64)&v1895, (__int64)v1897);
                v1607 = (_QWORD *)v837;
                v1270 = v1266;
                v1271 = (int *)v1895.m128i_i64[0];
                v1734 = v1895.m128i_i64[0] + 8LL * v1895.m128i_u32[2];
                v1272 = 0;
                while ( (int *)v1734 != v1271 )
                {
                  v1273 = v1271[1];
                  if ( v1272 >= v1270 )
                    break;
                  srcw = v1272 + v1273;
                  if ( v1272 + v1273 > v1270 )
                    v1273 = v1270 - v1272;
                  v1579 = *v1271;
                  sub_15C4EF0((__int64)&v1893, v1607, v1272, v1273);
                  if ( v1893.m128i_i8[8] )
                  {
                    v1274 = sub_1D241C0(
                              *(_QWORD *)(v1267 + 552),
                              v1268,
                              v1893.m128i_i64[0],
                              v1579,
                              0,
                              (__int64 *)&v1877,
                              *(_DWORD *)(v1267 + 536));
                    sub_1D30360(*(_QWORD *)(v1267 + 552), v1274, 0, 0, v1275, v1276);
                  }
                  else
                  {
                    srcw = v1272;
                  }
                  v1272 = srcw;
                  v1271 += 2;
                }
                if ( (_OWORD *)v1895.m128i_i64[0] != v1896 )
                  _libc_free(v1895.m128i_u64[0]);
              }
              sub_2052A10((unsigned __int64 *)v1897);
              goto LABEL_79;
            }
          }
        }
LABEL_577:
        if ( !*(_QWORD *)(v1209 + 8) )
          goto LABEL_79;
        v1210 = sub_2061930(v10 + 72, v1889.m128i_i64);
        v1895.m128i_i64[0] = v9;
        v1211 = v1210;
        v1212 = v1210[2];
        if ( v1212 == v1210[3] )
        {
          sub_205A820(v1210 + 1, v1210[2], &v1895, (__int64 *)&v1877, (int *)(v10 + 536));
          goto LABEL_79;
        }
        v1897[0].m128i_i64[0] = (__int64)v1877;
        if ( v1877 )
        {
          sub_1623A60((__int64)v1897, (__int64)v1877, 2);
          if ( !v1212 )
          {
LABEL_584:
            if ( v1897[0].m128i_i64[0] )
              sub_161E7C0((__int64)v1897, v1897[0].m128i_i64[0]);
            goto LABEL_586;
          }
        }
        else if ( !v1212 )
        {
LABEL_586:
          v1211[2] += 24;
          goto LABEL_79;
        }
        v1213 = *(_DWORD *)(v10 + 536);
        *(_QWORD *)v1212 = v9;
        v1214 = (unsigned __int8 *)v1897[0].m128i_i64[0];
        *(_QWORD *)(v1212 + 8) = v1897[0].m128i_i64[0];
        if ( v1214 )
        {
          sub_1623210((__int64)v1897, v1214, v1212 + 8);
          v1897[0].m128i_i64[0] = 0;
        }
        *(_DWORD *)(v1212 + 16) = v1213;
        goto LABEL_584;
      }
LABEL_330:
      v841 = sub_15C70A0((__int64)&v1877);
      if ( !(unsigned __int8)sub_2054040(v10, v1889.m128i_i64[0], v836, (_QWORD *)v837, v841, 0, v1891.m128i_i64) )
      {
        v842 = sub_2054060(
                 v10,
                 v1891.m128i_i64[0],
                 v1891.m128i_i32[2],
                 v836,
                 v837,
                 (__int64 *)&v1877,
                 *(_DWORD *)(v10 + 536));
        sub_1D30360(*(_QWORD *)(v10 + 552), (__int64)v842, v1891.m128i_i64[0], 0, v843, v844);
      }
      goto LABEL_79;
    case 0x27u:
    case 0xCDu:
      v1895.m128i_i64[0] = *(_QWORD *)(v9 + 56);
      v1897[0].m128i_i64[0] = sub_1560340(&v1895, -1, "trap-func-name", 0xEu);
      v186 = sub_155D8B0(v1897[0].m128i_i64);
      if ( v187 )
      {
        v1110 = *(_QWORD *)(v10 + 552);
        v1897[1].m128i_i64[1] = 0xFFFFFFFF00000020LL;
        v1897[5].m128i_i64[0] = v1110;
        v1901 = 0x2000000000LL;
        v1906 = 0x2000000000LL;
        v1909 = 0x2000000000LL;
        v1911 = v1913;
        memset(v1897, 0, 24);
        memset(&v1897[2], 0, 48);
        v1898 = 0;
        v1899 = 0;
        v1900 = &v1902;
        v1905 = v1907;
        v1908 = v1910;
        v1912 = 0x400000000LL;
        v1897[5].m128i_i64[1] = v1878;
        if ( v1878 )
        {
          v1634 = v186;
          sub_1623A60((__int64)&v1897[5].m128i_i64[1], v1878, 2);
          v186 = v1634;
        }
        v1602 = (char *)v186;
        v1898 = v1879;
        v1111 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
        v1112 = *(_QWORD *)(v10 + 552);
        v1897[0].m128i_i64[0] = (__int64)v1111;
        v1635 = v1112;
        v1897[0].m128i_i32[2] = v1113;
        v1114 = sub_1E0A0C0(*(_QWORD *)(v1112 + 32));
        v1115 = sub_2046180(v1114, 0);
        v1116 = sub_1D27640(v1635, v1602, v1115, 0);
        v1117 = v1897[4].m128i_i64[1];
        *(__int64 *)((char *)&v1897[1].m128i_i64[1] + 4) = 0;
        v1897[4] = 0u;
        v1118 = v1897[3].m128i_i64[1];
        v1897[1].m128i_i64[0] = *(_QWORD *)v9;
        v1897[2].m128i_i64[1] = v1116;
        v1897[3].m128i_i32[0] = v1119;
        v1897[3].m128i_i64[1] = 0;
        if ( v1118 )
          j_j___libc_free_0(v1118, v1117 - v1118);
        v1120 = *(void (****)())(v1897[5].m128i_i64[0] + 16);
        v1121 = **v1120;
        if ( v1121 != nullsub_684 )
          ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v1121)(
            v1120,
            *(_QWORD *)(v1897[5].m128i_i64[0] + 32),
            0,
            &v1897[3].m128i_u64[1]);
        sub_2056920((__int64)&v1895, v14, v1897, a7, a8, a9);
        sub_2045100(*(_QWORD *)(v10 + 552), *(__int64 *)&v1896[0], SDWORD2(v1896[0]));
        if ( v1911 != v1913 )
          _libc_free((unsigned __int64)v1911);
        if ( v1908 != v1910 )
          _libc_free((unsigned __int64)v1908);
        if ( v1905 != v1907 )
          _libc_free((unsigned __int64)v1905);
        if ( v1900 != &v1902 )
          _libc_free((unsigned __int64)v1900);
        if ( v1897[5].m128i_i64[1] )
          sub_161E7C0((__int64)&v1897[5].m128i_i64[1], v1897[5].m128i_i64[1]);
        if ( v1897[3].m128i_i64[1] )
          j_j___libc_free_0(v1897[3].m128i_i64[1], v1897[4].m128i_i64[1] - v1897[3].m128i_i64[1]);
      }
      else
      {
        v188 = *(__int64 **)(v10 + 552);
        *(_QWORD *)&v189 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
        v190 = sub_1D309E0(
                 v188,
                 (unsigned int)(a3 != 205) + 215,
                 (__int64)&v1878,
                 1,
                 0,
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 v189);
        sub_2045100((__int64)v188, v190, v191);
      }
      goto LABEL_79;
    case 0x29u:
      v809 = *(__int64 **)(v10 + 552);
      v810 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v812 = v811;
      v813 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v814 = sub_2046180(v813, 0);
      *((_QWORD *)&v1509 + 1) = v812;
      *(_QWORD *)&v1509 = v810;
      v815 = sub_1D309E0(
               v809,
               27,
               (__int64)&v1878,
               v814,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1509);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v810) = v816;
      v817 = v815;
      v818 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v818[1] = v817;
      *((_DWORD *)v818 + 4) = (_DWORD)v810;
      v17 = 0;
      goto LABEL_17;
    case 0x2Au:
    case 0x2Bu:
      v170 = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v171 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v172 = sub_2046180(v171, 0);
      v173 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(v14->m128i_i64[0] + 288);
      if ( v173 == sub_1D45FB0 )
      {
        v174 = v14[7].m128i_i64[(unsigned __int8)v172 + 1];
      }
      else
      {
        srccm = v172;
        v1181 = v173((__int64)v14, v172);
        LOBYTE(v172) = srccm;
        v174 = v1181;
      }
      v1698 = v172;
      v175 = sub_1FE0180(*(_QWORD *)(v10 + 712), v170, v174);
      v176 = *(__int64 **)(v10 + 552);
      v177 = v175;
      sub_204D410((__int64)&v1895, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
      v1588 = v1698;
      srcbg = *(_QWORD *)(v10 + 552) + 88LL;
      v1619 = (const void ***)sub_1D252B0((__int64)v176, v1698, 0, 1, 0);
      v1699 = v178;
      v1897[0].m128i_i64[0] = srcbg;
      v1897[0].m128i_i32[2] = 0;
      v180.m128i_i64[0] = (__int64)sub_1D2A660(v176, v177, v1588, 0, v179, v1588);
      v1897[1] = v180;
      *((_QWORD *)&v1491 + 1) = 2;
      *(_QWORD *)&v1491 = v1897;
      v1700 = sub_1D36D80(
                v176,
                47,
                (__int64)&v1895,
                v1619,
                v1699,
                *(double *)a7.m128i_i64,
                *(double *)a8.m128i_i64,
                a9,
                v181,
                v1491);
      v182 = (__int64)v1700;
      v184 = v183;
      sub_17CD270(v1895.m128i_i64);
      if ( a3 == 42 )
      {
        v1178 = *(__int64 **)(v10 + 552);
        sub_204D410((__int64)v1897, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
        v182 = sub_1D323C0(
                 v1178,
                 (__int64)v1700,
                 v184,
                 (__int64)v1897,
                 5,
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64);
        LODWORD(v184) = v1179;
        sub_17CD270(v1897[0].m128i_i64);
      }
      v1897[0].m128i_i64[0] = v9;
      v17 = 0;
      v185 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v185[1] = v182;
      *((_DWORD *)v185 + 4) = v184;
      goto LABEL_17;
    case 0x2Cu:
    case 0x2Du:
      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL) + 520LL) = 1;
      v157 = *(__int64 **)(v10 + 552);
      v158 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
      v160 = v159;
      v161 = *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      *(_QWORD *)&srcbf = sub_20685E0(v10, v161, a7, a8, a9);
      *((_QWORD *)&srcbf + 1) = v162;
      v166 = sub_2051DF0(
               (__int64 *)v10,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v161,
               v162,
               v163,
               v164,
               v165);
      v1490 = (__int64)v158;
      v17 = 0;
      v168 = sub_1D3A900(
               v157,
               0x1Cu,
               (__int64)&v1878,
               1u,
               0,
               0,
               (__m128)a7,
               *(double *)a8.m128i_i64,
               a9,
               (unsigned __int64)v166,
               v167,
               srcbf,
               v1490,
               v160);
      sub_2045100((__int64)v157, (__int64)v168, v169);
      goto LABEL_17;
    case 0x2Eu:
      v567 = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v567 + 16) != 13 )
        BUG();
      v568 = *(_QWORD **)(v567 + 24);
      if ( *(_DWORD *)(v567 + 32) > 0x40u )
        v568 = (_QWORD *)*v568;
      *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL) + 32LL) + 1728LL) = (_DWORD)v568;
      v17 = 0;
      goto LABEL_17;
    case 0x2Fu:
      v17 = 0;
      v564 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL) + 56LL);
      v565 = sub_1649C60(*(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v566 = *(_QWORD *)(v10 + 712);
      v1897[0].m128i_i64[0] = v565;
      *(_DWORD *)(v564 + 72) = *((_DWORD *)sub_2061B80(v566 + 336, v1897[0].m128i_i64) + 2);
      goto LABEL_17;
    case 0x30u:
      v612 = *(__int64 **)(v10 + 552);
      *(_QWORD *)&srcbq = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      *((_QWORD *)&srcbq + 1) = v613;
      v614 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v616 = sub_1D332F0(
               v612,
               30,
               (__int64)&v1878,
               1,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v614,
               v615,
               srcbq);
      goto LABEL_239;
    case 0x32u:
      memset(v1897, 0, 32);
      v1897[0].m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, 0.0, *(double *)a8.m128i_i64, a9);
      v1897[0].m128i_i32[2] = v603;
      v604 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), (__m128i)0LL, a8, a9);
      v605 = *(__int64 **)(v10 + 552);
      v1897[1].m128i_i64[0] = (__int64)v604;
      v1897[1].m128i_i32[2] = v606;
      v607 = (const void ***)sub_1D252B0((__int64)v605, 5, 0, 1, 0);
      *((_QWORD *)&v1506 + 1) = 2;
      *(_QWORD *)&v1506 = v1897;
      v610 = sub_1D36D80(v605, 29, (__int64)&v1878, v607, v608, 0.0, *(double *)a8.m128i_i64, a9, v609, v1506);
      v1895.m128i_i64[0] = v9;
      v294 = v610;
      v611 = sub_205F5C0(v10 + 8, v1895.m128i_i64);
      v611[1] = (__int64)v294;
      *((_DWORD *)v611 + 4) = 0;
      goto LABEL_151;
    case 0x33u:
      v598 = *(__int64 **)(v10 + 552);
      *(_QWORD *)&v599 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v600 = sub_1D309E0(
               v598,
               31,
               (__int64)&v1878,
               1,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v599);
      v601 = (__int64)v598;
      v17 = 0;
      sub_2045100(v601, v600, v602);
      goto LABEL_17;
    case 0x34u:
      v582 = sub_20C8260(*(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v583 = sub_1E0C1F0(*(_QWORD **)(*(_QWORD *)(v10 + 552) + 32LL), v582);
      v584 = sub_1D38BB0(*(_QWORD *)(v10 + 552), v583, (__int64)&v1878, 5, 0, 0, a7, *(double *)a8.m128i_i64, a9, 0);
      v1897[0].m128i_i64[0] = v9;
      v585 = v584;
      v587 = v586;
      v588 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v588[1] = v585;
      v17 = 0;
      *((_DWORD *)v588 + 4) = v587;
      goto LABEL_17;
    case 0x35u:
      v17 = 0;
      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL) + 521LL) = 1;
      goto LABEL_17;
    case 0x36u:
      v589 = *(__int64 **)(v10 + 552);
      v591 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v592 = v590;
      v593 = (const void ***)(v591[5] + 16LL * (unsigned int)v590);
      if ( *(_BYTE *)v593 == 9 && (unsigned int)(dword_4FCEF88 - 1) <= 0x11 )
      {
        *(_QWORD *)&v1333 = sub_2048150(
                              (__int64)v589,
                              0x3FB8AA3Bu,
                              (__int64)&v1878,
                              *(double *)a7.m128i_i64,
                              *(double *)a8.m128i_i64,
                              a9);
        v1334 = sub_1D332F0(
                  v589,
                  78,
                  (__int64)&v1878,
                  9,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  (__int64)v591,
                  v592,
                  v1333);
        v594 = sub_2048260((__int64)v1334, v1335, (__int64)&v1878, v589, a7, *(double *)a8.m128i_i64, a9);
        v596 = v1336;
      }
      else
      {
        *((_QWORD *)&v1505 + 1) = v590;
        *(_QWORD *)&v1505 = v591;
        v594 = sub_1D309E0(
                 v589,
                 172,
                 (__int64)&v1878,
                 *(unsigned __int8 *)v593,
                 v593[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 v1505);
        v596 = v595;
      }
      v1897[0].m128i_i64[0] = v9;
      v597 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1868 = v596;
      v17 = 0;
      v597[1] = v594;
      *((_DWORD *)v597 + 4) = v1868;
      goto LABEL_17;
    case 0x37u:
      v574 = *(__int64 **)(v10 + 552);
      v576 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v577 = (const void ***)(v576[5] + 16LL * (unsigned int)v575);
      if ( *(_BYTE *)v577 == 9 && (unsigned int)(dword_4FCEF88 - 1) <= 0x11 )
      {
        v578 = sub_2048260((__int64)v576, v575, (__int64)&v1878, v574, a7, *(double *)a8.m128i_i64, a9);
        v580 = v1332;
      }
      else
      {
        *((_QWORD *)&v1504 + 1) = v575;
        *(_QWORD *)&v1504 = v576;
        v578 = sub_1D309E0(
                 v574,
                 173,
                 (__int64)&v1878,
                 *(unsigned __int8 *)v577,
                 v577[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 v1504);
        v580 = v579;
      }
      v1897[0].m128i_i64[0] = v9;
      v581 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1867 = v580;
      v17 = 0;
      v581[1] = v578;
      *((_DWORD *)v581 + 4) = v1867;
      goto LABEL_17;
    case 0x38u:
      v569 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1897[0].m128i_i64[0] = v9;
      v571 = v570;
      v572 = v569;
      v573 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1847 = v571;
      v17 = 0;
      v573[1] = (__int64)v572;
      *((_DWORD *)v573 + 4) = v1847;
      goto LABEL_17;
    case 0x39u:
    case 0x3Au:
    case 0x3Bu:
    case 0x3Cu:
    case 0x3Du:
    case 0x3Eu:
    case 0x3Fu:
    case 0x40u:
    case 0x41u:
    case 0x42u:
    case 0x43u:
    case 0x44u:
    case 0x45u:
    case 0x46u:
    case 0x47u:
    case 0x48u:
    case 0x49u:
    case 0x4Au:
      v17 = 0;
      sub_2078580(v10, v9, a7, a8, a9);
      goto LABEL_17;
    case 0x4Bu:
      v17 = 0;
      sub_20A0700(v10, v9);
      goto LABEL_17;
    case 0x4Cu:
      v17 = 0;
      sub_209B000(v10, v9);
      goto LABEL_17;
    case 0x4Du:
      v17 = 0;
      sub_2099CA0(v10, v9);
      goto LABEL_17;
    case 0x4Eu:
      v344 = 0;
      if ( sub_1642D30(v9) )
      {
        v1131 = *(_BYTE *)(v9 + 16);
        if ( v1131 > 0x17u )
        {
          if ( v1131 == 78 )
          {
            v344 = v9 | 4;
          }
          else if ( v1131 == 29 )
          {
            v344 = v9 & 0xFFFFFFFFFFFFFFFBLL;
          }
        }
      }
      v345 = v344;
      v17 = 0;
      sub_209EC00(v10, v345, 0);
      goto LABEL_17;
    case 0x50u:
    case 0x51u:
      v17 = 0;
      sub_207F710(v10, v9 | 4, 0, a7, a8, a9);
      goto LABEL_17;
    case 0x52u:
      v17 = 0;
      sub_207F0F0(v10, v9, a7, a8, a9);
      goto LABEL_17;
    case 0x53u:
    case 0x54u:
    case 0x55u:
    case 0x56u:
    case 0x57u:
    case 0x58u:
    case 0x59u:
    case 0x5Au:
    case 0x5Bu:
    case 0x5Cu:
    case 0x5Du:
    case 0x5Eu:
    case 0x5Fu:
      v138 = a3;
      v17 = 0;
      sub_2080770(v10, v9, v138, a7, a8, a9);
      goto LABEL_17;
    case 0x62u:
      v352 = sub_1D2B300(*(_QWORD **)(v10 + 552), 0x9Bu, (__int64)&v1878, 5u, 0, a6);
      v1897[0].m128i_i64[0] = v9;
      v354 = v353;
      v355 = v352;
      v356 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1848 = v354;
      v17 = 0;
      v356[1] = (__int64)v355;
      *((_DWORD *)v356 + 4) = v1848;
      goto LABEL_17;
    case 0x63u:
      v500 = *(__int64 **)(v10 + 552);
      v501 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
      v1738 = v502;
      v1709 = (__int64)v501;
      *(_QWORD *)&srcbo = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&srcbo + 1) = v503;
      v504 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v506 = v505;
      v507 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v509 = (const void ***)(v507[5] + 16LL * v508);
      v510 = sub_1D3A900(
               v500,
               0x63u,
               (__int64)&v1878,
               *(unsigned __int8 *)v509,
               v509[1],
               0,
               (__m128)a7,
               *(double *)a8.m128i_i64,
               a9,
               (unsigned __int64)v504,
               v506,
               srcbo,
               v1709,
               v1738);
      LODWORD(v504) = v511;
      v512 = v510;
      v1897[0].m128i_i64[0] = v9;
      v513 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1861 = (int)v504;
      v17 = 0;
      v513[1] = (__int64)v512;
      *((_DWORD *)v513 + 4) = v1861;
      goto LABEL_17;
    case 0x64u:
      v479 = *(_QWORD *)v9;
      v480 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      LOBYTE(v481) = sub_204D4D0((__int64)v14, v480, v479);
      v482 = v10 + 8;
      if ( *(_DWORD *)(*(_QWORD *)(v10 + 544) + 816LL) == 2
        || (v483 = *(__int64 (**)())(v14->m128i_i64[0] + 920), v483 == sub_1F3CBF0)
        || (v1194 = ((__int64 (__fastcall *)(__m128i *, _QWORD))v483)(v14, v481), v482 = v10 + 8, !v1194) )
      {
        v1621 = v482;
        v1707 = *(__int64 **)(v10 + 552);
        *(_QWORD *)&srcbn = sub_20685E0(
                              v10,
                              *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                              a7,
                              a8,
                              a9);
        *((_QWORD *)&srcbn + 1) = v484;
        v485 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        v487 = v486;
        v488 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        v490 = sub_1D332F0(
                 v1707,
                 78,
                 (__int64)&v1878,
                 *(unsigned __int8 *)(v488[5] + 16LL * v489),
                 *(const void ***)(v488[5] + 16LL * v489 + 8),
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 a9,
                 (__int64)v485,
                 v487,
                 srcbn);
        v492 = v491;
        v493 = (__int64)v490;
        v1708 = *(__int64 **)(v10 + 552);
        *(_QWORD *)&srcbn = sub_20685E0(
                              v10,
                              *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                              a7,
                              a8,
                              a9);
        *((_QWORD *)&srcbn + 1) = v494;
        v495 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        v497 = sub_1D332F0(
                 v1708,
                 76,
                 (__int64)&v1878,
                 *(unsigned __int8 *)(v495[5] + 16LL * v496),
                 *(const void ***)(v495[5] + 16LL * v496 + 8),
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 a9,
                 v493,
                 v492,
                 srcbn);
        LODWORD(v493) = v498;
        v1897[0].m128i_i64[0] = v9;
        v499 = sub_205F5C0(v1621, v1897[0].m128i_i64);
        v499[1] = (__int64)v497;
        *((_DWORD *)v499 + 4) = v493;
      }
      else
      {
        v1195 = *(__int64 **)(v10 + 552);
        v1196 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
        v1743 = v1197;
        v1731 = (__int64)v1196;
        v1198 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
        v1200 = v1199;
        srcco = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        src_8j = v1201;
        v1202 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        v1204 = (const void ***)(v1202[5] + 16LL * v1203);
        *((_QWORD *)&v1488 + 1) = v1200;
        *(_QWORD *)&v1488 = v1198;
        v1205 = sub_1D3A900(
                  v1195,
                  0x63u,
                  (__int64)&v1878,
                  *(unsigned __int8 *)v1204,
                  v1204[1],
                  0,
                  (__m128)a7,
                  *(double *)a8.m128i_i64,
                  a9,
                  (unsigned __int64)srcco,
                  src_8j,
                  v1488,
                  v1731,
                  v1743);
        LODWORD(v1198) = v1206;
        v1897[0].m128i_i64[0] = v9;
        v1207 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
        v1207[1] = (__int64)v1205;
        *((_DWORD *)v1207 + 4) = (_DWORD)v1198;
      }
      goto LABEL_79;
    case 0x65u:
      v469 = *(__int64 **)(v10 + 552);
      v470 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v472 = v471;
      v473 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v474 = sub_2046180(v473, 0);
      *((_QWORD *)&v1499 + 1) = v472;
      *(_QWORD *)&v1499 = v470;
      v475 = sub_1D309E0(
               v469,
               20,
               (__int64)&v1878,
               v474,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1499);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v470) = v476;
      v477 = v475;
      v478 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1874 = (int)v470;
      v17 = 0;
      v478[1] = v477;
      *((_DWORD *)v478 + 4) = v1874;
      goto LABEL_17;
    case 0x66u:
    case 0x67u:
      v208 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1887.m128i_i64[1] = v209;
      LODWORD(v209) = *(_DWORD *)(v9 + 20);
      v1887.m128i_i64[0] = (__int64)v208;
      v210 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1 - (v209 & 0xFFFFFFF))), a7, a8, a9);
      v1889.m128i_i64[1] = v211;
      LODWORD(v211) = *(_DWORD *)(v9 + 20);
      v1889.m128i_i64[0] = (__int64)v210;
      *(_QWORD *)&srcd = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (2 - (v211 & 0xFFFFFFF))), a7, a8, a9);
      *((_QWORD *)&srcd + 1) = v212;
      v1568 = v1887.m128i_i32[2];
      v213 = *(_QWORD *)(v1887.m128i_i64[0] + 40) + 16LL * v1887.m128i_u32[2];
      v214 = *(_BYTE *)v213;
      v1620 = v1887.m128i_i64[0];
      v1701 = *(__int64 **)(v10 + 552);
      v215 = (unsigned int)(a3 != 102) + 125;
      v1891.m128i_i64[1] = *(_QWORD *)(v213 + 8);
      v1891.m128i_i8[0] = v214;
      v1589 = v1889.m128i_i64[0];
      if ( v1887.m128i_i64[0] == v1889.m128i_i64[0] && v1887.m128i_i32[2] == v1889.m128i_i32[2] )
      {
        v1132 = 1;
        if ( v214 == 1 )
        {
LABEL_511:
          if ( !v14[151].m128i_i8[259 * v1132 + 6 + (unsigned int)v215] )
          {
            v1133 = sub_1D332F0(
                      v1701,
                      v215,
                      (__int64)&v1878,
                      v1891.m128i_u32[0],
                      (const void **)v1891.m128i_i64[1],
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1887.m128i_i64[0],
                      v1887.m128i_u64[1],
                      srcd);
            v1135 = v1134;
            v1897[0].m128i_i64[0] = v9;
            v1136 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
            v1136[1] = (__int64)v1133;
            *((_DWORD *)v1136 + 4) = v1135;
            goto LABEL_79;
          }
LABEL_120:
          if ( (unsigned __int8)(v214 - 14) > 0x5Fu )
          {
LABEL_121:
            v216 = v1891.m128i_i64[1];
            v217 = v1891.m128i_i64[1];
            v218 = v1891.m128i_i8[0];
            goto LABEL_122;
          }
LABEL_472:
          v1109 = sub_1F7E0F0((__int64)&v1891);
          v216 = v1891.m128i_i64[1];
          v218 = v1109;
LABEL_122:
          v1897[0].m128i_i8[0] = v218;
          v1897[0].m128i_i64[1] = v217;
          if ( v218 )
            v219 = sub_2045180(v218);
          else
            v219 = sub_1F58D40((__int64)v1897);
          v220 = (const void **)v216;
          v221 = &v1895;
          *(_QWORD *)&v222 = sub_1D38BB0(
                               (__int64)v1701,
                               v219,
                               (__int64)&v1878,
                               v1891.m128i_u32[0],
                               v220,
                               0,
                               a7,
                               *(double *)a8.m128i_i64,
                               a9,
                               0);
          v1702 = v222;
          v223.m128i_i64[0] = (__int64)sub_1D332F0(
                                         *(__int64 **)(v10 + 552),
                                         58,
                                         (__int64)&v1878,
                                         v1891.m128i_u32[0],
                                         (const void **)v1891.m128i_i64[1],
                                         0,
                                         *(double *)a7.m128i_i64,
                                         *(double *)a8.m128i_i64,
                                         a9,
                                         srcd,
                                         *((unsigned __int64 *)&srcd + 1),
                                         v222);
          v224 = *(__int64 **)(v10 + 552);
          v1893 = v223;
          v225 = sub_1D332F0(
                   v224,
                   53,
                   (__int64)&v1878,
                   v1891.m128i_u32[0],
                   (const void **)v1891.m128i_i64[1],
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   v1702,
                   *((unsigned __int64 *)&v1702 + 1),
                   srcd);
          v1895.m128i_i64[0] = (__int64)sub_1D332F0(
                                          *(__int64 **)(v10 + 552),
                                          58,
                                          (__int64)&v1878,
                                          v1891.m128i_u32[0],
                                          (const void **)v1891.m128i_i64[1],
                                          0,
                                          *(double *)a7.m128i_i64,
                                          *(double *)a8.m128i_i64,
                                          a9,
                                          (__int64)v225,
                                          v226,
                                          v1702);
          v228 = &v1895;
          v229 = *(__int64 **)(v10 + 552);
          if ( a3 == 102 )
            v228 = &v1893;
          v1895.m128i_i64[1] = v227;
          v230 = sub_1D332F0(
                   v229,
                   122,
                   (__int64)&v1878,
                   v1891.m128i_u32[0],
                   (const void **)v1891.m128i_i64[1],
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   v1887.m128i_i64[0],
                   v1887.m128i_u64[1],
                   (__int128)*v228);
          src_8a = v231;
          if ( a3 != 102 )
            v221 = &v1893;
          srcbh = (__int64)v230;
          *(_QWORD *)&v232 = sub_1D332F0(
                               *(__int64 **)(v10 + 552),
                               124,
                               (__int64)&v1878,
                               v1891.m128i_u32[0],
                               (const void **)v1891.m128i_i64[1],
                               0,
                               *(double *)a7.m128i_i64,
                               *(double *)a8.m128i_i64,
                               a9,
                               v1889.m128i_i64[0],
                               v1889.m128i_u64[1],
                               (__int128)*v221);
          v233 = sub_1D332F0(
                   *(__int64 **)(v10 + 552),
                   119,
                   (__int64)&v1878,
                   v1891.m128i_u32[0],
                   (const void **)v1891.m128i_i64[1],
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   srcbh,
                   src_8a,
                   v232);
          srce = v234;
          v235 = v233;
          if ( v1620 == v1589 && v1568 == v1889.m128i_i32[2] )
            goto LABEL_136;
          v236 = sub_1D38BB0(
                   *(_QWORD *)(v10 + 552),
                   0,
                   (__int64)&v1878,
                   v1891.m128i_u32[0],
                   (const void **)v1891.m128i_i64[1],
                   0,
                   a7,
                   *(double *)a8.m128i_i64,
                   a9,
                   0);
          v237 = 0;
          v238 = v236;
          v240 = v239;
          v241 = 2;
          if ( v1891.m128i_i8[0] )
          {
            if ( (unsigned __int8)(v1891.m128i_i8[0] - 14) > 0x5Fu )
            {
LABEL_131:
              *((_QWORD *)&v1481 + 1) = v240;
              *(_QWORD *)&v1481 = v238;
              v242 = sub_1F81070(
                       *(__int64 **)(v10 + 552),
                       (__int64)&v1878,
                       v241,
                       v237,
                       v1893.m128i_u64[0],
                       (__int16 *)v1893.m128i_i64[1],
                       (__m128)a7,
                       *(double *)a8.m128i_i64,
                       a9,
                       v1481,
                       0x11u);
              v244 = v243;
              v245 = &v1887;
              v246 = v244;
              if ( a3 != 102 )
                v245 = &v1889;
              v247 = (unsigned __int64)v242;
              v248 = v242[5] + 16LL * (unsigned int)v244;
              v249 = *(_BYTE *)v248;
              v250 = *(_QWORD *)(v248 + 8);
              v251 = v245->m128i_i64[1];
              v1897[0].m128i_i8[0] = v249;
              v252 = *(__int64 **)(v10 + 552);
              v1897[0].m128i_i64[1] = v250;
              v254 = (const void **)v1891.m128i_i64[1];
              v253 = v1891.m128i_i32[0];
              v255 = v245->m128i_i64[0];
              if ( v249 )
              {
                v256 = ((unsigned __int8)(v249 - 14) < 0x60u) + 134;
              }
              else
              {
                v1562 = v1891.m128i_i32[0];
                v1576 = v247;
                v1582 = v246;
                v1603 = v255;
                v1614 = v251;
                v1636 = v252;
                v1125 = sub_1F58D20((__int64)v1897);
                v253 = v1562;
                v247 = v1576;
                v246 = v1582;
                v255 = v1603;
                v251 = v1614;
                v252 = v1636;
                v256 = 134 - (!v1125 - 1);
              }
              v235 = sub_1D3A900(
                       v252,
                       v256,
                       (__int64)&v1878,
                       v253,
                       v254,
                       0,
                       (__m128)a7,
                       *(double *)a8.m128i_i64,
                       a9,
                       v247,
                       v246,
                       __PAIR128__(v251, v255),
                       (__int64)v235,
                       srce);
              LODWORD(srce) = v257;
LABEL_136:
              v1897[0].m128i_i64[0] = v9;
              v258 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
              v258[1] = (__int64)v235;
              *((_DWORD *)v258 + 4) = srce;
              goto LABEL_79;
            }
            v1128 = word_4307B00[(unsigned __int8)(v1891.m128i_i8[0] - 14)];
          }
          else
          {
            v1637 = v236;
            v1671 = v240;
            v1126 = sub_1F58D20((__int64)&v1891);
            v238 = v1637;
            v240 = v1671;
            v241 = 2;
            v237 = 0;
            if ( !v1126 )
              goto LABEL_131;
            v1127 = sub_1F58D30((__int64)&v1891);
            LODWORD(v241) = 2;
            v238 = v1637;
            v240 = v1671;
            v1128 = v1127;
          }
          v1728 = v238;
          v1740 = v240;
          v1129 = sub_1F7DEB0(*(_QWORD **)(v10 + 768), v241, 0, v1128, 0);
          v238 = v1728;
          v240 = v1740;
          v237 = v1130;
          v241 = v1129;
          goto LABEL_131;
        }
        if ( v214 )
        {
          v1132 = v214;
          if ( !v14[7].m128i_i64[v214 + 1] )
            goto LABEL_120;
          goto LABEL_511;
        }
      }
      else if ( v214 )
      {
        goto LABEL_120;
      }
      if ( !sub_1F58D20((__int64)&v1891) )
        goto LABEL_121;
      goto LABEL_472;
    case 0x69u:
      v346 = (__int64 *)sub_1649C60(*(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v347 = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v348 = sub_20685E0(v10, v346, a7, a8, a9);
      v349 = *(_QWORD *)(v10 + 720);
      v350 = *((_DWORD *)v348 + 21);
      v1897[0].m128i_i64[1] = v347;
      v1897[0].m128i_i64[0] = v350 | 0xFFFFFFFF00000000LL;
      v351 = *(__m128i **)(v349 + 32);
      if ( v351 == *(__m128i **)(v349 + 40) )
      {
        sub_205C110((const __m128i **)(v349 + 24), v351, v1897);
      }
      else
      {
        if ( v351 )
        {
          *v351 = _mm_loadu_si128(v1897);
          v351 = *(__m128i **)(v349 + 32);
        }
        *(_QWORD *)(v349 + 32) = v351 + 1;
      }
      goto LABEL_79;
    case 0x6Bu:
      v452 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v454 = v453;
      v455 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v1706 = sub_2046180(v455, 0);
      srcbl = *(_QWORD *)v9;
      v456 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      LOBYTE(v457) = sub_204D4D0((__int64)v14, v456, srcbl);
      if ( v1706 != (_BYTE)v457 || v458 && !v1706 )
        sub_16BD130("Wrong result type for @llvm.get.dynamic.area.offset intrinsic!", 1u);
      v461 = *(__int64 **)(v10 + 552);
      v462 = (const void ***)sub_1D29190((__int64)v461, v457, v458, v459, v1706, v460);
      *((_QWORD *)&v1498 + 1) = v454;
      *(_QWORD *)&v1498 = v452;
      v465 = sub_1D37410(
               v461,
               243,
               (__int64)&v1878,
               v462,
               v463,
               v464,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               v1498);
      v466 = (__int64)v452;
      v17 = 0;
      LODWORD(v461) = v467;
      srcbm = v465;
      sub_2045100(*(_QWORD *)(v10 + 552), v466, v454);
      v1897[0].m128i_i64[0] = v9;
      v468 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      *((_DWORD *)v468 + 4) = (_DWORD)v461;
      v468[1] = (__int64)srcbm;
      goto LABEL_17;
    case 0x6Cu:
      v1897[0].m128i_i64[0] = (__int64)v1897[1].m128i_i64;
      v1897[0].m128i_i64[1] = 0x1000000000LL;
      sub_1D23890((__int64)v1897, (const __m128i *)(*(_QWORD *)(v10 + 552) + 176LL), a3, a4, a5, a6);
      v514.m128i_i64[0] = (__int64)sub_20685E0(
                                     v10,
                                     *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)),
                                     a7,
                                     a8,
                                     a9);
      v1895 = v514;
      sub_1D23890((__int64)v1897, &v1895, v514.m128i_i64[1], v515, v516, v517);
      v518 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v1551 = sub_14AC610(
                *(unsigned __int16 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                v1889.m128i_i64,
                v518);
      v519 = *((_BYTE *)v1551 + 16);
      if ( v519 && v519 != 3 )
        goto LABEL_210;
      v1081 = *(_QWORD **)(v10 + 552);
      sub_204D410((__int64)&v1893, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
      v1895.m128i_i64[0] = (__int64)sub_1D29600(v1081, (__int64)v1551, (__int64)&v1893, 6u, 0, 0, 1, 0);
      v1895.m128i_i32[2] = v1082;
      sub_1D23890((__int64)v1897, &v1895, v1082, v1083, v1084, v1085);
      sub_17CD270(v1893.m128i_i64);
      LODWORD(v1081) = *(_DWORD *)(v9 + 20);
      v1895.m128i_i64[0] = (__int64)v1896;
      v1086 = (unsigned int)v1081 & 0xFFFFFFF;
      v1087 = *(char *)(v9 + 23) < 0;
      v1895.m128i_i64[1] = 0x800000000LL;
      if ( !v1087 )
        goto LABEL_461;
      v1088 = sub_1648A40(v9);
      v1090 = v1088 + v1089;
      if ( *(char *)(v9 + 23) >= 0 )
      {
        if ( (unsigned int)(v1090 >> 4) )
          goto LABEL_715;
      }
      else if ( (unsigned int)((v1090 - sub_1648A40(v9)) >> 4) )
      {
        if ( *(char *)(v9 + 23) < 0 )
        {
          v1091 = *(_DWORD *)(sub_1648A40(v9) + 8);
          if ( *(char *)(v9 + 23) >= 0 )
            BUG();
          v1092 = sub_1648A40(v9);
          v1094 = *(_DWORD *)(v1092 + v1093 - 4) - v1091;
LABEL_451:
          v109 = 1;
          v1543 = v1086 - 1 - v1094;
          if ( v1543 != 1 )
          {
            v1095 = v1534;
            do
            {
              v1096 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
              v1097 = sub_14AC610(
                        *(unsigned __int16 **)(v9 + 24 * (v109 - (unsigned __int64)(*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                        v1889.m128i_i64,
                        v1096);
              v1098 = *((_BYTE *)v1097 + 16);
              if ( v1098 != 3 && v1098 || v1551 != v1097 )
                sub_16BD130("all llvm.icall.branch.funnel operands must refer to the same GlobalValue", 1u);
              v97 = sub_20685E0(
                      v10,
                      *(__int64 **)(v9 + 24 * (v109 + 1 - (unsigned __int64)(*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                      a7,
                      a8,
                      a9);
              v99 = *((_WORD *)v97 + 12);
              if ( (unsigned __int16)(v99 - 12) > 1u && (unsigned __int16)(v99 - 34) > 1u )
LABEL_210:
                sub_16BD130("llvm.icall.branch.funnel operand must be a GlobalValue", 1u);
              v100 = *(_QWORD *)v10;
              v101 = *(_QWORD **)(v10 + 552);
              v1617 = v97;
              v1893.m128i_i64[0] = v1889.m128i_i64[0];
              v102 = v97[5] + 16LL * v98;
              v1567 = v101;
              v1587 = v97[12];
              srcbb = *(_BYTE *)v102;
              v1697 = *(_QWORD *)(v102 + 8);
              sub_204D410((__int64)&v1891, v100, *(_DWORD *)(v10 + 536));
              LOBYTE(v1095) = srcbb;
              v1893.m128i_i64[1] = (__int64)sub_1D29600(v1567, v1617[11], (__int64)&v1891, v1095, v1697, v1587, 1, 0);
              v105 = v1895.m128i_u32[2];
              LODWORD(v1894[0]) = v106;
              if ( v1895.m128i_i32[2] >= (unsigned __int32)v1895.m128i_i32[3] )
              {
                sub_16CD150((__int64)&v1895, v1896, 0, 24, v103, v104);
                v105 = v1895.m128i_u32[2];
              }
              v107 = &v1893;
              a8 = _mm_loadu_si128(&v1893);
              v108 = (__m128 *)(v1895.m128i_i64[0] + 24 * v105);
              *v108 = (__m128)a8;
              v108[1].m128_u64[0] = v1894[0];
              ++v1895.m128i_i32[2];
              if ( v1891.m128i_i64[0] )
                sub_161E7C0((__int64)&v1891, v1891.m128i_i64[0]);
              v109 += 2;
            }
            while ( v109 != v1543 );
          }
          v110 = 24LL * v1895.m128i_u32[2];
          v111 = (char *)(v1895.m128i_i64[0] + v110);
          if ( v1895.m128i_i64[0] == v1895.m128i_i64[0] + v110 )
            goto LABEL_74;
          srcb = (char *)v1895.m128i_i64[0];
          _BitScanReverse64(&v112, 0xAAAAAAAAAAAAAAABLL * (v110 >> 3));
          sub_2045490(
            v1895.m128i_i64[0],
            (__m128i *)(v1895.m128i_i64[0] + v110),
            2LL * (int)(63 - (v112 ^ 0x3F)),
            (__int64)v107,
            v103,
            v104);
          if ( (unsigned __int64)v110 <= 0x180 )
          {
            sub_2045A50(srcb, v111);
          }
          else
          {
            v113 = srcb + 384;
            sub_2045A50(srcb, srcb + 384);
            if ( v111 == srcb + 384 )
            {
              v120 = (__int64 *)v1895.m128i_i64[0];
              v1618 = v1895.m128i_i64[0] + 24LL * v1895.m128i_u32[2];
LABEL_70:
              if ( v120 != (__int64 *)v1618 )
              {
                v121 = v10;
                v122 = v1537;
                do
                {
                  LOBYTE(v122) = 5;
                  srcbc = *(_QWORD *)(v121 + 552);
                  sub_204D410((__int64)&v1891, *(_QWORD *)v121, *(_DWORD *)(v121 + 536));
                  v1893.m128i_i64[0] = sub_1D38BB0(
                                         srcbc,
                                         *v120,
                                         (__int64)&v1891,
                                         v122,
                                         0,
                                         1,
                                         a7,
                                         *(double *)a8.m128i_i64,
                                         a9,
                                         0);
                  v1893.m128i_i32[2] = v123;
                  sub_1D23890((__int64)v1897, &v1893, v123, v124, v125, v126);
                  sub_17CD270(v1891.m128i_i64);
                  v127 = (const __m128i *)(v120 + 1);
                  v120 += 3;
                  sub_1D23890((__int64)v1897, v127, v128, v129, v130, v131);
                }
                while ( (__int64 *)v1618 != v120 );
                v10 = v121;
              }
LABEL_74:
              *(_QWORD *)&srcbd = v1897[0].m128i_i64[0];
              v132 = *(_QWORD **)(v10 + 552);
              *((_QWORD *)&srcbd + 1) = v1897[0].m128i_u32[2];
              sub_204D410((__int64)&v1893, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
              v134 = sub_1D2CDB0(v132, 33, (__int64)&v1893, 1, 0, v133, srcbd);
              sub_17CD270(v1893.m128i_i64);
              sub_2045100(*(_QWORD *)(v10 + 552), v134, 0);
              v1893.m128i_i64[0] = v9;
              v135 = sub_205F5C0(v10 + 8, v1893.m128i_i64);
              v135[1] = v134;
              *((_DWORD *)v135 + 4) = 0;
              v136 = (_OWORD *)v1895.m128i_i64[0];
              *(_BYTE *)(v10 + 760) = 1;
              if ( v136 != v1896 )
                _libc_free((unsigned __int64)v136);
LABEL_76:
              v137 = (__m128i *)v1897[0].m128i_i64[0];
LABEL_77:
              if ( v137 == &v1897[1] )
                goto LABEL_79;
LABEL_78:
              _libc_free((unsigned __int64)v137);
              goto LABEL_79;
            }
            do
            {
              v114 = *(_QWORD *)v113;
              v115 = *((_QWORD *)v113 + 1);
              v116 = (const __m128i *)(v113 - 24);
              v117 = *((_DWORD *)v113 + 4);
              if ( *((_QWORD *)v113 - 3) <= *(_QWORD *)v113 )
              {
                v119 = (__m128i *)v113;
              }
              else
              {
                do
                {
                  v118 = _mm_loadu_si128(v116);
                  v116[2].m128i_i64[1] = v116[1].m128i_i64[0];
                  v119 = (__m128i *)v116;
                  v116 = (const __m128i *)((char *)v116 - 24);
                  v116[3] = v118;
                }
                while ( v114 < v116->m128i_i64[0] );
              }
              v113 += 24;
              v119->m128i_i64[0] = v114;
              v119->m128i_i64[1] = v115;
              v119[1].m128i_i32[0] = v117;
            }
            while ( v111 != v113 );
          }
          v120 = (__int64 *)v1895.m128i_i64[0];
          v1618 = v1895.m128i_i64[0] + 24LL * v1895.m128i_u32[2];
          goto LABEL_70;
        }
LABEL_715:
        BUG();
      }
LABEL_461:
      v1094 = 0;
      goto LABEL_451;
    case 0x6Du:
      v367 = sub_1649C60(*(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))));
      memset(v1897, 0, sizeof(v1897));
      v1897[0].m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1897[0].m128i_i32[2] = v368;
      v1852 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1853 = v369;
      v370 = *(_DWORD *)(v9 + 20);
      v1897[1].m128i_i64[0] = (__int64)v1852;
      v1897[1].m128i_i32[2] = v1853;
      v1850 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (v370 & 0xFFFFFFF))), a7, a8, a9);
      v1851 = v371;
      v372 = *(_DWORD *)(v9 + 20);
      v1897[2].m128i_i64[0] = (__int64)v1850;
      v1897[2].m128i_i32[2] = v1851;
      v373 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (2LL - (v372 & 0xFFFFFFF))), a7, a8, a9);
      v374 = *(_QWORD **)(v10 + 552);
      v1897[3].m128i_i64[0] = (__int64)v373;
      v1897[3].m128i_i32[2] = v375;
      v379 = v367;
      v17 = 0;
      v1849 = sub_1D2AD90(
                v374,
                *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)),
                4LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF),
                v376,
                v377,
                v378);
      v380 = *(_QWORD **)(v10 + 552);
      v1897[4].m128i_i64[0] = (__int64)v1849;
      v1897[4].m128i_i32[2] = v381;
      v385 = sub_1D2AD90(v380, v379, v381, v382, v383, v384);
      v386 = *(__int64 **)(v10 + 552);
      v1897[5].m128i_i64[0] = (__int64)v385;
      *((_QWORD *)&v1494 + 1) = 6;
      v1897[5].m128i_i32[2] = v387;
      *(_QWORD *)&v1494 = v1897;
      v388 = sub_1D359D0(
               v386,
               213,
               (__int64)&v1878,
               1,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               v1494);
      sub_2045100(*(_QWORD *)(v10 + 552), (__int64)v388, v389);
      goto LABEL_17;
    case 0x72u:
      v357 = *(_QWORD **)(v10 + 552);
      v358 = sub_1E0A0C0(v357[4]);
      v359 = sub_2046180(v358, 0);
      v363 = sub_1D2B530(v357, v359, 0, v360, v361, v362);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v357) = v364;
      v365 = v363;
      v366 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1839 = (int)v357;
      v17 = 0;
      v366[1] = (__int64)v365;
      *((_DWORD *)v366 + 4) = v1839;
      goto LABEL_17;
    case 0x74u:
    case 0x75u:
      if ( !(unsigned int)sub_1700720(*(_QWORD *)(v10 + 544)) )
        goto LABEL_79;
      v259 = *(_DWORD *)(v9 + 20);
      v260 = *(_QWORD *)(v10 + 560);
      v1897[0].m128i_i64[0] = (__int64)v1897[1].m128i_i64;
      v1897[0].m128i_i64[1] = 0x400000000LL;
      sub_14AD470(*(_QWORD *)(v9 + 24 * (1LL - (v259 & 0xFFFFFFF))), (__int64)v1897, v260, 0, 6u);
      v261 = (__int64 *)v1897[0].m128i_i64[0];
      v262 = v1897[0].m128i_i64[0] + 8LL * v1897[0].m128i_u32[2];
      v137 = (__m128i *)v1897[0].m128i_i64[0];
      if ( v1897[0].m128i_i64[0] == v262 )
        goto LABEL_77;
      v263 = v1564;
      v264 = (a3 != 117) + 239;
      while ( 2 )
      {
        v265 = *v261;
        if ( !*v261 || *(_BYTE *)(v265 + 16) != 53 )
          goto LABEL_140;
        v266 = *(_QWORD *)(v10 + 712);
        v267 = *(unsigned int *)(v266 + 360);
        if ( (_DWORD)v267 )
        {
          v268 = *(_QWORD *)(v266 + 344);
          v269 = (v267 - 1) & (((unsigned int)v265 >> 9) ^ ((unsigned int)v265 >> 4));
          v270 = (__int64 *)(v268 + 16LL * v269);
          v271 = *v270;
          if ( v265 == *v270 )
          {
LABEL_145:
            if ( v270 != (__int64 *)(v268 + 16 * v267) )
            {
              v272 = *((_DWORD *)v270 + 2);
              v1895 = 0;
              v1896[0] = 0;
              v1590 = v272;
              v273 = sub_2051C20((__int64 *)v10, 0.0, *(double *)a8.m128i_i64, a9);
              v274 = *(_QWORD **)(v10 + 552);
              v1895.m128i_i64[0] = (__int64)v273;
              v1703 = v274;
              v1895.m128i_i32[2] = v275;
              v276 = sub_1E0A0C0(v274[4]);
              LOBYTE(v263) = sub_2046180(v276, *(_DWORD *)(v276 + 4));
              v277 = sub_1D299D0(v1703, v1590, v263, 0, 1);
              v278 = *(__int64 **)(v10 + 552);
              *(_QWORD *)&v1896[0] = v277;
              DWORD2(v1896[0]) = v279;
              *((_QWORD *)&v1492 + 1) = 2;
              *(_QWORD *)&v1492 = &v1895;
              v280 = sub_1D359D0(v278, v264, (__int64)&v1878, 1, 0, 0, 0.0, *(double *)a8.m128i_i64, a9, v1492);
              src_8 = v281 | src_8 & 0xFFFFFFFF00000000LL;
              sub_2045100(*(_QWORD *)(v10 + 552), (__int64)v280, src_8);
LABEL_140:
              if ( (__int64 *)v262 == ++v261 )
                goto LABEL_76;
              continue;
            }
          }
          else
          {
            v1216 = 1;
            while ( v271 != -8 )
            {
              v1217 = v1216 + 1;
              v269 = (v267 - 1) & (v1216 + v269);
              v270 = (__int64 *)(v268 + 16LL * v269);
              v271 = *v270;
              if ( v265 == *v270 )
                goto LABEL_145;
              v1216 = v1217;
            }
          }
        }
        goto LABEL_76;
      }
    case 0x78u:
      v1558 = 0;
      v656 = *(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL);
      v657 = *(__int64 (**)(void))(**(_QWORD **)(v656 + 16) + 40LL);
      if ( v657 != sub_1D00B00 )
        v1558 = v657();
      v658 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      if ( *(char *)(v9 + 23) >= 0 )
        goto LABEL_444;
      v659 = sub_1648A40(v9);
      v661 = v659 + v660;
      if ( *(char *)(v9 + 23) >= 0 )
      {
        if ( (unsigned int)(v661 >> 4) )
          goto LABEL_718;
      }
      else if ( (unsigned int)((v661 - sub_1648A40(v9)) >> 4) )
      {
        if ( *(char *)(v9 + 23) < 0 )
        {
          v662 = *(_DWORD *)(sub_1648A40(v9) + 8);
          if ( *(char *)(v9 + 23) >= 0 )
            BUG();
          v663 = sub_1648A40(v9);
          v665 = *(_DWORD *)(v663 + v664 - 4) - v662;
          goto LABEL_280;
        }
LABEL_718:
        BUG();
      }
LABEL_444:
      v665 = 0;
LABEL_280:
      v666 = v658 - 1 == v665;
      v1571 = (unsigned int)(v658 - 1 - v665);
      v667 = 0;
      if ( !v666 )
      {
        v1548 = (__int64 *)v656;
        do
        {
          v668 = sub_1649C60(*(_QWORD *)(v9 + 24 * (v667 - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))));
          if ( *(_BYTE *)(v668 + 16) != 15 )
          {
            v1895.m128i_i64[0] = v668;
            srcj = *((_DWORD *)sub_2061B80(*(_QWORD *)(v10 + 712) + 336LL, v1895.m128i_i64) + 2);
            v669 = v1548[4] + 168;
            v670 = sub_1E0A440(v1548);
            v672 = v670;
            if ( v671 && *v670 == 1 )
            {
              --v671;
              v672 = v670 + 1;
            }
            v1594 = sub_38BF760(v669, v672, v671, (unsigned int)v667);
            v673 = *(_QWORD *)(v10 + 712);
            v674 = *(__int64 **)(v673 + 792);
            v1625 = *(_QWORD *)(v673 + 784);
            v1713 = *(_QWORD *)(v1625 + 56);
            v675 = (__int64)sub_1E0B640(v1713, *(_QWORD *)(v1558 + 8) + 1536LL, (__int64 *)&v1877, 0);
            sub_1DD5BA0((__int64 *)(v1625 + 16), v675);
            v676 = *v674;
            v677 = *(_QWORD *)v675;
            *(_QWORD *)(v675 + 8) = v674;
            v676 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v675 = v676 | v677 & 7;
            *(_QWORD *)(v676 + 8) = v675;
            *v674 = v675 | *v674 & 7;
            v1897[0].m128i_i8[0] = 15;
            v1897[1].m128i_i64[0] = 0;
            v1897[0].m128i_i32[0] &= 0xFFF000FF;
            v1897[1].m128i_i64[1] = v1594;
            v1897[0].m128i_i32[2] = 0;
            v1897[2].m128i_i32[0] = 0;
            sub_1E1A9C0(v675, v1713, v1897);
            v1897[0].m128i_i64[0] = 5;
            v1897[1].m128i_i64[0] = 0;
            v1897[1].m128i_i32[2] = srcj;
            sub_1E1A9C0(v675, v1713, v1897);
          }
          ++v667;
        }
        while ( v667 != v1571 );
      }
      goto LABEL_79;
    case 0x79u:
      v414 = *(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL);
      v415 = sub_1E0A0C0(v414);
      v416 = sub_2046180(v415, 0);
      v417 = sub_1649C60(*(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v418 = *(_QWORD *)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v419 = *(_DWORD *)(v418 + 32);
      if ( v419 > 0x40 )
      {
        srcck = v417;
        v1727 = *(_QWORD *)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
        v1104 = sub_16A57B0(v418 + 24);
        v417 = srcck;
        v1105 = v1104;
        v1106 = v419;
        LODWORD(v420) = 0x7FFFFFFF;
        if ( v1106 - v1105 <= 0x40 )
        {
          LODWORD(v420) = 0x7FFFFFFF;
          v1107 = *(unsigned __int64 **)(v1727 + 24);
          if ( *v1107 <= 0x7FFFFFFF )
            v420 = *v1107;
        }
      }
      else
      {
        v420 = *(_QWORD *)(v418 + 24);
        if ( v420 > 0x7FFFFFFF )
          LODWORD(v420) = 0x7FFFFFFF;
      }
      v421 = *(_QWORD *)(v414 + 32);
      v422 = sub_1649960(v417);
      v424 = v421 + 168;
      v425 = v422;
      if ( v423 && *v422 == 1 )
      {
        --v423;
        v425 = v422 + 1;
      }
      v426 = sub_38BF760(v424, v425, v423, (unsigned int)v420);
      *(_QWORD *)&v427 = sub_1D45A20(*(_QWORD *)(v10 + 552), v426, v416, 0);
      v428 = sub_1D309E0(
               *(__int64 **)(v10 + 552),
               23,
               (__int64)&v1878,
               v416,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v427);
      v430 = v429;
      v431 = v428;
      v432 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
      *((_QWORD *)&v1484 + 1) = v430;
      *(_QWORD *)&v1484 = v431;
      v434 = sub_1D332F0(
               *(__int64 **)(v10 + 552),
               52,
               (__int64)&v1878,
               v416,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v432,
               v433,
               v1484);
      LODWORD(v431) = v435;
      v436 = v434;
      v1897[0].m128i_i64[0] = v9;
      v437 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1838 = v431;
      v17 = 0;
      v437[1] = (__int64)v436;
      *((_DWORD *)v437 + 4) = v1838;
      goto LABEL_17;
    case 0x7Au:
      v406 = *(__int64 **)(v10 + 552);
      v408 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v409 = (const void ***)(v408[5] + 16LL * (unsigned int)v407);
      if ( *(_BYTE *)v409 == 9 && (unsigned int)(dword_4FCEF88 - 1) <= 0x11 )
      {
        *((_QWORD *)&v1524 + 1) = v407;
        *(_QWORD *)&v1524 = v408;
        v1609 = sub_1D309E0(
                  v406,
                  158,
                  (__int64)&v1878,
                  5,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  *(double *)a9.m128i_i64,
                  v1524);
        v1647 = v1302;
        v1303 = sub_20468E0(v406, v1609, v1302, a7, *(double *)a8.m128i_i64, a9, (__int64)v14, (__int64)&v1878);
        src_8l = v1304;
        srccs = v1303;
        *(_QWORD *)&v1305 = sub_2048150(
                              (__int64)v406,
                              0x3F317218u,
                              (__int64)&v1878,
                              *(double *)a7.m128i_i64,
                              *(double *)a8.m128i_i64,
                              a9);
        v1306 = sub_1D332F0(
                  v406,
                  78,
                  (__int64)&v1878,
                  9,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  srccs,
                  src_8l,
                  v1305);
        v1746 = v1307;
        v1736 = (__int64)v1306;
        *(_QWORD *)&srcz = sub_2043F30(v406, v1609, v1647, (__int64)&v1878, a7, *(double *)a8.m128i_i64, a9);
        *((_QWORD *)&srcz + 1) = v1308;
        if ( (unsigned int)dword_4FCEF88 <= 6 )
        {
          *(_QWORD *)&v1437 = sub_2048150(
                                (__int64)v406,
                                0xBE74C456,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1438 = sub_1D332F0(
                    v406,
                    78,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcz,
                    *((unsigned __int64 *)&srcz + 1),
                    v1437);
          v1696 = v1439;
          v1669 = (__int64)v1438;
          *(_QWORD *)&v1440 = sub_2048150(
                                (__int64)v406,
                                0x3FB3A2B1u,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1441 = sub_1D332F0(
                    v406,
                    76,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    v1669,
                    v1696,
                    v1440);
          v1443 = sub_1D332F0(
                    v406,
                    78,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    (__int64)v1441,
                    v1442,
                    srcz);
          src_8o = v1444;
          srccv = (__int64)v1443;
          *(_QWORD *)&v1445 = sub_2048150(
                                (__int64)v406,
                                0x3F949A29u,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1329 = sub_1D332F0(
                    v406,
                    77,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srccv,
                    src_8o,
                    v1445);
          v1330 = (unsigned int)v1330;
        }
        else
        {
          if ( (unsigned int)dword_4FCEF88 > 0xC )
          {
            *(_QWORD *)&v1409 = sub_2048150(
                                  (__int64)v406,
                                  0xBC91E5AC,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1410 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      srcz,
                      *((unsigned __int64 *)&srcz + 1),
                      v1409);
            v1691 = v1411;
            v1664 = (__int64)v1410;
            *(_QWORD *)&v1412 = sub_2048150(
                                  (__int64)v406,
                                  0x3E4350AAu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1413 = sub_1D332F0(
                      v406,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1664,
                      v1691,
                      v1412);
            v1415 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1413,
                      v1414,
                      srcz);
            v1692 = v1416;
            v1665 = (__int64)v1415;
            *(_QWORD *)&v1417 = sub_2048150(
                                  (__int64)v406,
                                  0x3F60D3E3u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1418 = sub_1D332F0(
                      v406,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1665,
                      v1692,
                      v1417);
            v1420 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1418,
                      v1419,
                      srcz);
            v1693 = v1421;
            v1666 = (__int64)v1420;
            *(_QWORD *)&v1422 = sub_2048150(
                                  (__int64)v406,
                                  0x4011CDF0u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1423 = sub_1D332F0(
                      v406,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1666,
                      v1693,
                      v1422);
            v1425 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1423,
                      v1424,
                      srcz);
            v1694 = v1426;
            v1667 = (__int64)v1425;
            *(_QWORD *)&v1427 = sub_2048150(
                                  (__int64)v406,
                                  0x406CFD1Cu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1428 = sub_1D332F0(
                      v406,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1667,
                      v1694,
                      v1427);
            v1430 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1428,
                      v1429,
                      srcz);
            v1695 = v1431;
            v1668 = (__int64)v1430;
            *(_QWORD *)&v1432 = sub_2048150(
                                  (__int64)v406,
                                  0x408797CBu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1433 = sub_1D332F0(
                      v406,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1668,
                      v1695,
                      v1432);
            v1435 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1433,
                      v1434,
                      srcz);
            src_8g = v1436;
            srcba = (__int64)v1435;
            *(_QWORD *)&v1327 = sub_2048150(
                                  (__int64)v406,
                                  0x4006DCABu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
          }
          else
          {
            *(_QWORD *)&v1309 = sub_2048150(
                                  (__int64)v406,
                                  0xBD67B6D6,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1310 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      srcz,
                      *((unsigned __int64 *)&srcz + 1),
                      v1309);
            v1677 = v1311;
            v1648 = (__int64)v1310;
            *(_QWORD *)&v1312 = sub_2048150(
                                  (__int64)v406,
                                  0x3EE4F4B8u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1313 = sub_1D332F0(
                      v406,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1648,
                      v1677,
                      v1312);
            v1315 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1313,
                      v1314,
                      srcz);
            v1678 = v1316;
            v1649 = (__int64)v1315;
            *(_QWORD *)&v1317 = sub_2048150(
                                  (__int64)v406,
                                  0x3FBC278Bu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1318 = sub_1D332F0(
                      v406,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1649,
                      v1678,
                      v1317);
            v1320 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1318,
                      v1319,
                      srcz);
            v1679 = v1321;
            v1650 = (__int64)v1320;
            *(_QWORD *)&v1322 = sub_2048150(
                                  (__int64)v406,
                                  0x40348E95u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1323 = sub_1D332F0(
                      v406,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1650,
                      v1679,
                      v1322);
            v1325 = sub_1D332F0(
                      v406,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1323,
                      v1324,
                      srcz);
            src_8g = v1326;
            srcba = (__int64)v1325;
            *(_QWORD *)&v1327 = sub_2048150(
                                  (__int64)v406,
                                  0x3FDEF31Au,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
          }
          v1328 = sub_1D332F0(
                    v406,
                    77,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcba,
                    src_8g,
                    v1327);
          v1329 = v1328;
          v1330 = (unsigned int)v1330;
        }
        *((_QWORD *)&v1525 + 1) = v1330;
        *(_QWORD *)&v1525 = v1329;
        v410 = (__int64)sub_1D332F0(
                          v406,
                          76,
                          (__int64)&v1878,
                          9,
                          0,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)a8.m128i_i64,
                          a9,
                          v1736,
                          v1746,
                          v1525);
        v412 = v1331;
      }
      else
      {
        *((_QWORD *)&v1497 + 1) = v407;
        *(_QWORD *)&v1497 = v408;
        v410 = sub_1D309E0(
                 v406,
                 169,
                 (__int64)&v1878,
                 *(unsigned __int8 *)v409,
                 v409[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 v1497);
        v412 = v411;
      }
      v1897[0].m128i_i64[0] = v9;
      v413 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1871 = v412;
      v17 = 0;
      v413[1] = v410;
      *((_DWORD *)v413 + 4) = v1871;
      goto LABEL_17;
    case 0x7Bu:
      v398 = *(__int64 **)(v10 + 552);
      v400 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v401 = (const void ***)(v400[5] + 16LL * (unsigned int)v399);
      if ( *(_BYTE *)v401 == 9 && (unsigned int)(dword_4FCEF88 - 1) <= 0x11 )
      {
        *((_QWORD *)&v1522 + 1) = v399;
        *(_QWORD *)&v1522 = v400;
        v1608 = sub_1D309E0(
                  v398,
                  158,
                  (__int64)&v1878,
                  5,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  *(double *)a9.m128i_i64,
                  v1522);
        v1644 = v1277;
        v1278 = sub_20468E0(v398, v1608, v1277, a7, *(double *)a8.m128i_i64, a9, (__int64)v14, (__int64)&v1878);
        src_8k = v1279;
        srccr = v1278;
        *(_QWORD *)&v1280 = sub_2048150(
                              (__int64)v398,
                              0x3E9A209Au,
                              (__int64)&v1878,
                              *(double *)a7.m128i_i64,
                              *(double *)a8.m128i_i64,
                              a9);
        v1281 = sub_1D332F0(
                  v398,
                  78,
                  (__int64)&v1878,
                  9,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  a9,
                  srccr,
                  src_8k,
                  v1280);
        v1745 = v1282;
        v1735 = (__int64)v1281;
        *(_QWORD *)&srcx = sub_2043F30(v398, v1608, v1644, (__int64)&v1878, a7, *(double *)a8.m128i_i64, a9);
        *((_QWORD *)&srcx + 1) = v1283;
        if ( (unsigned int)dword_4FCEF88 <= 6 )
        {
          *(_QWORD *)&v1400 = sub_2048150(
                                (__int64)v398,
                                0xBDD49A13,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1401 = sub_1D332F0(
                    v398,
                    78,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcx,
                    *((unsigned __int64 *)&srcx + 1),
                    v1400);
          v1690 = v1402;
          v1663 = (__int64)v1401;
          *(_QWORD *)&v1403 = sub_2048150(
                                (__int64)v398,
                                0x3F1C0789u,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1404 = sub_1D332F0(
                    v398,
                    76,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    v1663,
                    v1690,
                    v1403);
          v1406 = sub_1D332F0(
                    v398,
                    78,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    (__int64)v1404,
                    v1405,
                    srcx);
          src_8n = v1407;
          srccu = (__int64)v1406;
          *(_QWORD *)&v1408 = sub_2048150(
                                (__int64)v398,
                                0x3F011300u,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1299 = sub_1D332F0(
                    v398,
                    77,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srccu,
                    src_8n,
                    v1408);
          v1300 = (unsigned int)v1300;
        }
        else
        {
          if ( (unsigned int)dword_4FCEF88 > 0xC )
          {
            *(_QWORD *)&v1377 = sub_2048150(
                                  (__int64)v398,
                                  0x3C5D51CEu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1378 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      srcx,
                      *((unsigned __int64 *)&srcx + 1),
                      v1377);
            v1686 = v1379;
            v1659 = (__int64)v1378;
            *(_QWORD *)&v1380 = sub_2048150(
                                  (__int64)v398,
                                  0x3E00685Au,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1381 = sub_1D332F0(
                      v398,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1659,
                      v1686,
                      v1380);
            v1383 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1381,
                      v1382,
                      srcx);
            v1687 = v1384;
            v1660 = (__int64)v1383;
            *(_QWORD *)&v1385 = sub_2048150(
                                  (__int64)v398,
                                  0x3EFB6798u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1386 = sub_1D332F0(
                      v398,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1660,
                      v1687,
                      v1385);
            v1388 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1386,
                      v1387,
                      srcx);
            v1688 = v1389;
            v1661 = (__int64)v1388;
            *(_QWORD *)&v1390 = sub_2048150(
                                  (__int64)v398,
                                  0x3F88D192u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1391 = sub_1D332F0(
                      v398,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1661,
                      v1688,
                      v1390);
            v1393 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1391,
                      v1392,
                      srcx);
            v1689 = v1394;
            v1662 = (__int64)v1393;
            *(_QWORD *)&v1395 = sub_2048150(
                                  (__int64)v398,
                                  0x3FC4316Cu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1396 = sub_1D332F0(
                      v398,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1662,
                      v1689,
                      v1395);
            v1398 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1396,
                      v1397,
                      srcx);
            src_8f = v1399;
            srcy = (__int64)v1398;
            *(_QWORD *)&v1297 = sub_2048150(
                                  (__int64)v398,
                                  0x3F57CE70u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
          }
          else
          {
            *(_QWORD *)&v1284 = sub_2048150(
                                  (__int64)v398,
                                  0x3D431F31u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1285 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      srcx,
                      *((unsigned __int64 *)&srcx + 1),
                      v1284);
            v1675 = v1286;
            v1645 = (__int64)v1285;
            *(_QWORD *)&v1287 = sub_2048150(
                                  (__int64)v398,
                                  0x3EA21FB2u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1288 = sub_1D332F0(
                      v398,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1645,
                      v1675,
                      v1287);
            v1290 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1288,
                      v1289,
                      srcx);
            v1676 = v1291;
            v1646 = (__int64)v1290;
            *(_QWORD *)&v1292 = sub_2048150(
                                  (__int64)v398,
                                  0x3F6AE232u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1293 = sub_1D332F0(
                      v398,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1646,
                      v1676,
                      v1292);
            v1295 = sub_1D332F0(
                      v398,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1293,
                      v1294,
                      srcx);
            src_8f = v1296;
            srcy = (__int64)v1295;
            *(_QWORD *)&v1297 = sub_2048150(
                                  (__int64)v398,
                                  0x3F25F7C3u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
          }
          v1298 = sub_1D332F0(
                    v398,
                    77,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcy,
                    src_8f,
                    v1297);
          v1299 = v1298;
          v1300 = (unsigned int)v1300;
        }
        *((_QWORD *)&v1523 + 1) = v1300;
        *(_QWORD *)&v1523 = v1299;
        v402 = (__int64)sub_1D332F0(
                          v398,
                          76,
                          (__int64)&v1878,
                          9,
                          0,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)a8.m128i_i64,
                          a9,
                          v1735,
                          v1745,
                          v1523);
        v404 = v1301;
      }
      else
      {
        *((_QWORD *)&v1496 + 1) = v399;
        *(_QWORD *)&v1496 = v400;
        v402 = sub_1D309E0(
                 v398,
                 171,
                 (__int64)&v1878,
                 *(unsigned __int8 *)v401,
                 v401[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 v1496);
        v404 = v403;
      }
      v1897[0].m128i_i64[0] = v9;
      v405 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1869 = v404;
      v17 = 0;
      v405[1] = v402;
      *((_DWORD *)v405 + 4) = v1869;
      goto LABEL_17;
    case 0x7Cu:
      v390 = *(__int64 **)(v10 + 552);
      v392 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v393 = (const void ***)(v392[5] + 16LL * (unsigned int)v391);
      if ( *(_BYTE *)v393 == 9 && (unsigned int)(dword_4FCEF88 - 1) <= 0x11 )
      {
        *((_QWORD *)&v1520 + 1) = v391;
        *(_QWORD *)&v1520 = v392;
        v1640 = sub_1D309E0(
                  v390,
                  158,
                  (__int64)&v1878,
                  5,
                  0,
                  0,
                  *(double *)a7.m128i_i64,
                  *(double *)a8.m128i_i64,
                  *(double *)a9.m128i_i64,
                  v1520);
        srccp = v1225;
        v1226 = sub_20468E0(v390, v1640, v1225, a7, *(double *)a8.m128i_i64, a9, (__int64)v14, (__int64)&v1878);
        v1744 = v1227;
        v1732 = v1226;
        *(_QWORD *)&srcu = sub_2043F30(v390, v1640, srccp, (__int64)&v1878, a7, *(double *)a8.m128i_i64, a9);
        *((_QWORD *)&srcu + 1) = v1228;
        if ( (unsigned int)dword_4FCEF88 <= 6 )
        {
          *(_QWORD *)&v1368 = sub_2048150(
                                (__int64)v390,
                                0xBEB08FE0,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1369 = sub_1D332F0(
                    v390,
                    78,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcu,
                    *((unsigned __int64 *)&srcu + 1),
                    v1368);
          v1685 = v1370;
          v1658 = (__int64)v1369;
          *(_QWORD *)&v1371 = sub_2048150(
                                (__int64)v390,
                                0x40019463u,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1372 = sub_1D332F0(
                    v390,
                    76,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    v1658,
                    v1685,
                    v1371);
          v1374 = sub_1D332F0(
                    v390,
                    78,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    (__int64)v1372,
                    v1373,
                    srcu);
          src_8m = v1375;
          srcct = (__int64)v1374;
          *(_QWORD *)&v1376 = sub_2048150(
                                (__int64)v390,
                                0x3FD6633Du,
                                (__int64)&v1878,
                                *(double *)a7.m128i_i64,
                                *(double *)a8.m128i_i64,
                                a9);
          v1249 = sub_1D332F0(
                    v390,
                    77,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcct,
                    src_8m,
                    v1376);
          v1250 = (unsigned int)v1250;
        }
        else
        {
          if ( (unsigned int)dword_4FCEF88 > 0xC )
          {
            *(_QWORD *)&v1340 = sub_2048150(
                                  (__int64)v390,
                                  0xBCD2769E,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1341 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      srcu,
                      *((unsigned __int64 *)&srcu + 1),
                      v1340);
            v1680 = v1342;
            v1653 = (__int64)v1341;
            *(_QWORD *)&v1343 = sub_2048150(
                                  (__int64)v390,
                                  0x3E8CE0B9u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1344 = sub_1D332F0(
                      v390,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1653,
                      v1680,
                      v1343);
            v1346 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1344,
                      v1345,
                      srcu);
            v1681 = v1347;
            v1654 = (__int64)v1346;
            *(_QWORD *)&v1348 = sub_2048150(
                                  (__int64)v390,
                                  0x3FA22AE7u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1349 = sub_1D332F0(
                      v390,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1654,
                      v1681,
                      v1348);
            v1351 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1349,
                      v1350,
                      srcu);
            v1682 = v1352;
            v1655 = (__int64)v1351;
            *(_QWORD *)&v1353 = sub_2048150(
                                  (__int64)v390,
                                  0x40525723u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1354 = sub_1D332F0(
                      v390,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1655,
                      v1682,
                      v1353);
            v1356 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1354,
                      v1355,
                      srcu);
            v1683 = v1357;
            v1656 = (__int64)v1356;
            *(_QWORD *)&v1358 = sub_2048150(
                                  (__int64)v390,
                                  0x40AAF200u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1359 = sub_1D332F0(
                      v390,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1656,
                      v1683,
                      v1358);
            v1361 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1359,
                      v1360,
                      srcu);
            v1684 = v1362;
            v1657 = (__int64)v1361;
            *(_QWORD *)&v1363 = sub_2048150(
                                  (__int64)v390,
                                  0x40C39DADu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1364 = sub_1D332F0(
                      v390,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1657,
                      v1684,
                      v1363);
            v1366 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1364,
                      v1365,
                      srcu);
            src_8e = v1367;
            srcv = (__int64)v1366;
            *(_QWORD *)&v1247 = sub_2048150(
                                  (__int64)v390,
                                  0x4042902Cu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
          }
          else
          {
            *(_QWORD *)&v1229 = sub_2048150(
                                  (__int64)v390,
                                  0xBDA7262E,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1230 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      srcu,
                      *((unsigned __int64 *)&srcu + 1),
                      v1229);
            v1672 = v1231;
            v1641 = (__int64)v1230;
            *(_QWORD *)&v1232 = sub_2048150(
                                  (__int64)v390,
                                  0x3F25280Bu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1233 = sub_1D332F0(
                      v390,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1641,
                      v1672,
                      v1232);
            v1235 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1233,
                      v1234,
                      srcu);
            v1673 = v1236;
            v1642 = (__int64)v1235;
            *(_QWORD *)&v1237 = sub_2048150(
                                  (__int64)v390,
                                  0x4007B923u,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1238 = sub_1D332F0(
                      v390,
                      77,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1642,
                      v1673,
                      v1237);
            v1240 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1238,
                      v1239,
                      srcu);
            v1674 = v1241;
            v1643 = (__int64)v1240;
            *(_QWORD *)&v1242 = sub_2048150(
                                  (__int64)v390,
                                  0x40823E2Fu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
            v1243 = sub_1D332F0(
                      v390,
                      76,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      v1643,
                      v1674,
                      v1242);
            v1245 = sub_1D332F0(
                      v390,
                      78,
                      (__int64)&v1878,
                      9,
                      0,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1243,
                      v1244,
                      srcu);
            src_8e = v1246;
            srcv = (__int64)v1245;
            *(_QWORD *)&v1247 = sub_2048150(
                                  (__int64)v390,
                                  0x4020D29Cu,
                                  (__int64)&v1878,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  a9);
          }
          v1248 = sub_1D332F0(
                    v390,
                    77,
                    (__int64)&v1878,
                    9,
                    0,
                    0,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    srcv,
                    src_8e,
                    v1247);
          v1249 = v1248;
          v1250 = (unsigned int)v1250;
        }
        *((_QWORD *)&v1521 + 1) = v1250;
        *(_QWORD *)&v1521 = v1249;
        v394 = (__int64)sub_1D332F0(
                          v390,
                          76,
                          (__int64)&v1878,
                          9,
                          0,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)a8.m128i_i64,
                          a9,
                          v1732,
                          v1744,
                          v1521);
        v396 = v1251;
      }
      else
      {
        *((_QWORD *)&v1495 + 1) = v391;
        *(_QWORD *)&v1495 = v392;
        v394 = sub_1D309E0(
                 v390,
                 170,
                 (__int64)&v1878,
                 *(unsigned __int8 *)v393,
                 v393[1],
                 0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 v1495);
        v396 = v395;
      }
      v1897[0].m128i_i64[0] = v9;
      v397 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1870 = v396;
      v17 = 0;
      v397[1] = v394;
      *((_DWORD *)v397 + 4) = v1870;
      goto LABEL_17;
    case 0x7Du:
      v17 = &aCLongjmp[(v14[3].m128i_i8[11] ^ 1) + 3];
      goto LABEL_17;
    case 0x7Eu:
      v17 = 0;
      sub_20748F0(v10, v9, 1, a7, a8, a9);
      goto LABEL_17;
    case 0x7Fu:
      v17 = 0;
      sub_2074D20(v10, v9, 1, a7, a8, a9);
      goto LABEL_17;
    case 0x80u:
      v17 = 0;
      sub_2081050(v10, v9, a7, a8, a9);
      goto LABEL_17;
    case 0x81u:
      v17 = 0;
      sub_2074D20(v10, v9, 0, a7, a8, a9);
      goto LABEL_17;
    case 0x82u:
      v17 = 0;
      sub_2081870(v10, v9, a7, a8, a9);
      goto LABEL_17;
    case 0x83u:
      v17 = 0;
      sub_20748F0(v10, v9, 0, a7, a8, a9);
      goto LABEL_17;
    case 0x84u:
      v438 = 181;
      v439 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v441 = (unsigned __int8 *)(v439[5] + 16LL * v440);
      v442 = *v441;
      srcf = (const void **)*((_QWORD *)v441 + 1);
      if ( sub_15F24B0(v9) )
        v438 = !sub_204D480((__int64)v14, 0xB7u, v442) ? 181 : 183;
      v443 = *(__int64 **)(v10 + 552);
      *(_QWORD *)&v1705 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&v1705 + 1) = v444;
      v445 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v447 = sub_1D332F0(
               v443,
               v438,
               (__int64)&v1878,
               v442,
               srcf,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v445,
               v446,
               v1705);
      v449 = v448;
      v450 = v447;
      v1897[0].m128i_i64[0] = v9;
      v451 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1863 = v449;
      v17 = 0;
      v451[1] = (__int64)v450;
      *((_DWORD *)v451 + 4) = v1863;
      goto LABEL_17;
    case 0x85u:
      v636 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v638 = v637;
      v639 = (__int64)v636;
      *(_QWORD *)&v640 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1712 = v640;
      *(_QWORD *)&v641 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1624 = v641;
      srch = sub_15603A0((_QWORD *)(v9 + 56), 0);
      v642 = sub_15603A0((_QWORD *)(v9 + 56), 1);
      v643 = srch;
      if ( !v642 )
        v642 = 1;
      if ( !srch )
        v643 = 1;
      v644 = v643 | (unsigned int)v642;
      v645 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v646 = -v644 & v644;
      v647 = *(_QWORD *)(v9 + 24 * (3 - v645));
      if ( *(_DWORD *)(v647 + 32) <= 0x40u )
      {
        v649 = *(_QWORD *)(v647 + 24) == 0;
      }
      else
      {
        v1592 = *(_DWORD *)(v647 + 32);
        srcbr = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
        v648 = sub_16A57B0(v647 + 24);
        v645 = srcbr;
        v649 = v1592 == v648;
      }
      v650 = 0;
      srci = !v649;
      if ( (*(_WORD *)(v9 + 18) & 3u) - 1 <= 1 )
      {
        v650 = sub_20C8B80(v9 | 4, **(_QWORD **)(v10 + 552));
        v645 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v651 = *(__int64 **)(v10 + 552);
      v652 = *(__int64 **)(v9 + 24 * (1 - v645));
      if ( v652 )
      {
        v1897[0] = (__m128i)(unsigned __int64)v652;
        v1897[1].m128i_i8[0] = 0;
        v652 = (__int64 *)*v652;
        if ( *((_BYTE *)v652 + 8) == 16 )
          v652 = *(__int64 **)v652[2];
        LODWORD(v652) = *((_DWORD *)v652 + 2) >> 8;
      }
      else
      {
        memset(v1897, 0, 24);
      }
      v1897[1].m128i_i32[1] = (int)v652;
      v653 = *(__int64 **)(v9 - 24 * v645);
      if ( v653 )
      {
        v1895 = (__m128i)(unsigned __int64)v653;
        LOBYTE(v1896[0]) = 0;
        v653 = (__int64 *)*v653;
        if ( *((_BYTE *)v653 + 8) == 16 )
          v653 = *(__int64 **)v653[2];
        LODWORD(v653) = *((_DWORD *)v653 + 2) >> 8;
      }
      else
      {
        v1895 = 0u;
        *(_QWORD *)&v1896[0] = 0;
      }
      v1570 = v650;
      v1593 = v651;
      DWORD1(v1896[0]) = (_DWORD)v653;
      v654 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v634 = (__int64)sub_1D3E790(
                        v1593,
                        (unsigned __int64)v654,
                        v655,
                        (__int64)&v1878,
                        v639,
                        v638,
                        (__m128)a7,
                        a8,
                        a9,
                        v1712,
                        v1624,
                        v646,
                        srci,
                        0,
                        v1570,
                        v1895.m128i_i64[0],
                        v1895.m128i_i64[1],
                        *(__int64 *)&v1896[0],
                        v1897[0].m128i_i64[0],
                        v1897[0].m128i_i64[1],
                        v1897[1].m128i_i64[0]);
      goto LABEL_254;
    case 0x86u:
      v619 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v621 = v620;
      v622 = (__int64)v619;
      *(_QWORD *)&v1591 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&v1591 + 1) = v623;
      *(_QWORD *)&v624 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1569 = v624;
      v1623 = sub_15603A0((_QWORD *)(v9 + 56), 0);
      srcg = sub_15603A0((_QWORD *)(v9 + 56), 1);
      v625 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v1711 = **(_QWORD **)(v9 + 24 * (2 - v625));
      v626 = *(_QWORD *)(v9 + 24 * (3 - v625));
      if ( *(_DWORD *)(v626 + 32) <= 0x40u )
        v627 = *(_QWORD *)(v626 + 24);
      else
        v627 = **(_QWORD **)(v626 + 24);
      v628 = 0;
      if ( (*(_WORD *)(v9 + 18) & 3u) - 1 <= 1 )
      {
        v628 = sub_20C8B80(v9 | 4, **(_QWORD **)(v10 + 552));
        v625 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v629 = *(_QWORD **)(v10 + 552);
      v630 = *(__int64 **)(v9 + 24 * (1 - v625));
      if ( v630 )
      {
        v1897[0] = (__m128i)(unsigned __int64)v630;
        v1897[1].m128i_i8[0] = 0;
        v630 = (__int64 *)*v630;
        if ( *((_BYTE *)v630 + 8) == 16 )
          v630 = *(__int64 **)v630[2];
        LODWORD(v630) = *((_DWORD *)v630 + 2) >> 8;
      }
      else
      {
        memset(v1897, 0, 24);
      }
      v1897[1].m128i_i32[1] = (int)v630;
      v631 = *(__int64 **)(v9 - 24 * v625);
      if ( v631 )
      {
        v1895 = (__m128i)(unsigned __int64)v631;
        LOBYTE(v1896[0]) = 0;
        v631 = (__int64 *)*v631;
        if ( *((_BYTE *)v631 + 8) == 16 )
          v631 = *(__int64 **)v631[2];
        LODWORD(v631) = *((_DWORD *)v631 + 2) >> 8;
      }
      else
      {
        v1895 = 0u;
        *(_QWORD *)&v1896[0] = 0;
      }
      v1552 = v628;
      v1557 = v629;
      DWORD1(v1896[0]) = (_DWORD)v631;
      v632 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v634 = sub_1D278A0(
               v1557,
               (__int64)v632,
               v633,
               (__int64)&v1878,
               v622,
               v621,
               v1623,
               v1591,
               srcg,
               v1569,
               v1711,
               v627,
               v1552);
      goto LABEL_254;
    case 0x87u:
      v997 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v999 = v998;
      v1000 = (__int64)v997;
      *(_QWORD *)&v1001 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      v1724 = v1001;
      *(_QWORD *)&v1002 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      v1632 = v1002;
      srcs = sub_15603A0((_QWORD *)(v9 + 56), 0);
      v1003 = sub_15603A0((_QWORD *)(v9 + 56), 1);
      v1004 = srcs;
      if ( !v1003 )
        v1003 = 1;
      if ( !srcs )
        v1004 = 1;
      v1005 = v1004 | (unsigned int)v1003;
      v1006 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v1007 = -v1005 & v1005;
      v1008 = *(_QWORD *)(v9 + 24 * (3 - v1006));
      if ( *(_DWORD *)(v1008 + 32) <= 0x40u )
      {
        v1010 = *(_QWORD *)(v1008 + 24) == 0;
      }
      else
      {
        v1600 = *(_DWORD *)(v1008 + 32);
        srccf = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
        v1009 = sub_16A57B0(v1008 + 24);
        v1006 = srccf;
        v1010 = v1600 == v1009;
      }
      v1011 = 0;
      srct = !v1010;
      if ( (*(_WORD *)(v9 + 18) & 3u) - 1 <= 1 )
      {
        v1011 = sub_20C8B80(v9 | 4, **(_QWORD **)(v10 + 552));
        v1006 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v1012 = *(__int64 **)(v10 + 552);
      v1013 = *(__int64 **)(v9 + 24 * (1 - v1006));
      if ( v1013 )
      {
        v1897[0] = (__m128i)(unsigned __int64)v1013;
        v1897[1].m128i_i8[0] = 0;
        v1013 = (__int64 *)*v1013;
        if ( *((_BYTE *)v1013 + 8) == 16 )
          v1013 = *(__int64 **)v1013[2];
        LODWORD(v1013) = *((_DWORD *)v1013 + 2) >> 8;
      }
      else
      {
        memset(v1897, 0, 24);
      }
      v1897[1].m128i_i32[1] = (int)v1013;
      v1014 = *(__int64 **)(v9 - 24 * v1006);
      if ( v1014 )
      {
        v1895 = (__m128i)(unsigned __int64)v1014;
        LOBYTE(v1896[0]) = 0;
        v1014 = (__int64 *)*v1014;
        if ( *((_BYTE *)v1014 + 8) == 16 )
          v1014 = *(__int64 **)v1014[2];
        LODWORD(v1014) = *((_DWORD *)v1014 + 2) >> 8;
      }
      else
      {
        v1895 = 0u;
        *(_QWORD *)&v1896[0] = 0;
      }
      v1575 = v1011;
      v1601 = v1012;
      DWORD1(v1896[0]) = (_DWORD)v1014;
      v1015 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v634 = sub_1D3F1E0(
               v1601,
               (__int64)v1015,
               v1016,
               (__int64)&v1878,
               v1000,
               v999,
               a7,
               a8,
               v1724,
               v1632,
               v1007,
               srct,
               v1575,
               *(_OWORD *)&v1895,
               *(__int64 *)&v1896[0],
               *(_OWORD *)v1897,
               v1897[1].m128i_i64[0]);
      goto LABEL_254;
    case 0x88u:
      v982 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v984 = v983;
      v985 = (__int64)v982;
      *(_QWORD *)&v1599 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&v1599 + 1) = v986;
      *(_QWORD *)&v987 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1574 = v987;
      v1631 = sub_15603A0((_QWORD *)(v9 + 56), 0);
      v1723 = sub_15603A0((_QWORD *)(v9 + 56), 1);
      v988 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      srcr = **(_QWORD **)(v9 + 24 * (2 - v988));
      v989 = *(_QWORD *)(v9 + 24 * (3 - v988));
      if ( *(_DWORD *)(v989 + 32) <= 0x40u )
        v990 = *(_QWORD *)(v989 + 24);
      else
        v990 = **(_QWORD **)(v989 + 24);
      v991 = 0;
      if ( (*(_WORD *)(v9 + 18) & 3u) - 1 <= 1 )
      {
        v991 = sub_20C8B80(v9 | 4, **(_QWORD **)(v10 + 552));
        v988 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v992 = *(_QWORD **)(v10 + 552);
      v993 = *(__int64 **)(v9 + 24 * (1 - v988));
      if ( v993 )
      {
        v1897[0] = (__m128i)(unsigned __int64)v993;
        v1897[1].m128i_i8[0] = 0;
        v993 = (__int64 *)*v993;
        if ( *((_BYTE *)v993 + 8) == 16 )
          v993 = *(__int64 **)v993[2];
        LODWORD(v993) = *((_DWORD *)v993 + 2) >> 8;
      }
      else
      {
        memset(v1897, 0, 24);
      }
      v1897[1].m128i_i32[1] = (int)v993;
      v994 = *(__int64 **)(v9 - 24 * v988);
      if ( v994 )
      {
        v1895 = (__m128i)(unsigned __int64)v994;
        LOBYTE(v1896[0]) = 0;
        v994 = (__int64 *)*v994;
        if ( *((_BYTE *)v994 + 8) == 16 )
          v994 = *(__int64 **)v994[2];
        LODWORD(v994) = *((_DWORD *)v994 + 2) >> 8;
      }
      else
      {
        v1895 = 0u;
        *(_QWORD *)&v1896[0] = 0;
      }
      v1554 = v991;
      v1561 = v992;
      DWORD1(v1896[0]) = (_DWORD)v994;
      v995 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v634 = sub_1D27F90(
               v1561,
               (__int64)v995,
               v996,
               (__int64)&v1878,
               v985,
               v984,
               v1631,
               v1599,
               v1723,
               v1574,
               srcr,
               v990,
               v1554);
LABEL_254:
      v17 = 0;
      sub_2054630(v10, v634, v635);
      goto LABEL_17;
    case 0x89u:
      v861 = 1;
      v862 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v864 = v863;
      v865 = v862;
      *(_QWORD *)&v866 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1720 = v866;
      *(_QWORD *)&v867 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1630 = v867;
      v868 = sub_15603A0((_QWORD *)(v9 + 56), 0);
      if ( v868 )
        v861 = v868;
      v869 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v870 = *(_QWORD *)(v9 + 24 * (3 - v869));
      if ( *(_DWORD *)(v870 + 32) <= 0x40u )
      {
        v872 = *(_QWORD *)(v870 + 24) == 0;
      }
      else
      {
        v1597 = *(_DWORD *)(v870 + 32);
        srcbz = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
        v871 = sub_16A57B0(v870 + 24);
        v869 = srcbz;
        v872 = v1597 == v871;
      }
      v873 = 0;
      srcp = !v872;
      if ( (*(_WORD *)(v9 + 18) & 3u) - 1 <= 1 )
      {
        v873 = sub_20C8B80(v9 | 4, **(_QWORD **)(v10 + 552));
        v869 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v874 = *(_QWORD *)(v10 + 552);
      v875 = *(__int64 **)(v9 - 24 * v869);
      if ( v875 )
      {
        v1897[0] = (__m128i)(unsigned __int64)v875;
        v1897[1].m128i_i8[0] = 0;
        v875 = (__int64 *)*v875;
        if ( *((_BYTE *)v875 + 8) == 16 )
          v875 = *(__int64 **)v875[2];
        LODWORD(v875) = *((_DWORD *)v875 + 2) >> 8;
      }
      else
      {
        memset(v1897, 0, 24);
      }
      v1573 = v873;
      v1598 = v874;
      v1897[1].m128i_i32[1] = (int)v875;
      v876 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v877 = (__int64)v865;
      v17 = 0;
      v879 = sub_1D43030(
               v1598,
               (__int64)v876,
               v878,
               (__int64)&v1878,
               v877,
               v864,
               a7,
               a8,
               a9,
               v1720,
               v1630,
               v861,
               srcp,
               v1573,
               *(_OWORD *)v1897,
               v1897[1].m128i_i64[0]);
      sub_2054630(v10, v879, v880);
      goto LABEL_17;
    case 0x8Au:
      v845 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      src_8d = v846;
      srco = (__int64)v845;
      *(_QWORD *)&v1719 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&v1719 + 1) = v847;
      *(_QWORD *)&v848 = sub_20685E0(
                           v10,
                           *(__int64 **)(v9 + 24 * (2LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                           a7,
                           a8,
                           a9);
      v1629 = v848;
      v849 = sub_15603A0((_QWORD *)(v9 + 56), 0);
      v850 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v851 = **(_QWORD **)(v9 + 24 * (2 - v850));
      v852 = *(_QWORD *)(v9 + 24 * (3 - v850));
      if ( *(_DWORD *)(v852 + 32) <= 0x40u )
        v853 = *(_QWORD *)(v852 + 24);
      else
        v853 = **(_QWORD **)(v852 + 24);
      v854 = 0;
      if ( (*(_WORD *)(v9 + 18) & 3u) - 1 <= 1 )
      {
        v854 = sub_20C8B80(v9 | 4, **(_QWORD **)(v10 + 552));
        v850 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v855 = *(_QWORD **)(v10 + 552);
      v856 = *(__int64 **)(v9 - 24 * v850);
      if ( v856 )
      {
        v1897[0] = (__m128i)(unsigned __int64)v856;
        v1897[1].m128i_i8[0] = 0;
        v856 = (__int64 *)*v856;
        if ( *((_BYTE *)v856 + 8) == 16 )
          v856 = *(__int64 **)v856[2];
        LODWORD(v856) = *((_DWORD *)v856 + 2) >> 8;
      }
      else
      {
        memset(v1897, 0, 24);
      }
      v1572 = v854;
      v1596 = v855;
      v1897[1].m128i_i32[1] = (int)v856;
      v857 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1479 = v853;
      v17 = 0;
      v859 = sub_1D28680(
               v1596,
               (__int64)v857,
               v858,
               (__int64)&v1878,
               srco,
               src_8d,
               v849,
               v1719,
               v1629,
               v851,
               v1479,
               v1572);
      sub_2054630(v10, v859, v860);
      goto LABEL_17;
    case 0x8Bu:
      v965 = 180;
      v966 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v968 = (unsigned __int8 *)(v966[5] + 16LL * v967);
      v969 = *v968;
      srcq = (const void **)*((_QWORD *)v968 + 1);
      if ( sub_15F24B0(v9) )
        v965 = !sub_204D480((__int64)v14, 0xB6u, v969) ? 180 : 182;
      v970 = *(__int64 **)(v10 + 552);
      *(_QWORD *)&v1722 = sub_20685E0(
                            v10,
                            *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                            a7,
                            a8,
                            a9);
      *((_QWORD *)&v1722 + 1) = v971;
      v972 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v974 = sub_1D332F0(
               v970,
               v965,
               (__int64)&v1878,
               v969,
               srcq,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v972,
               v973,
               v1722);
      v976 = v975;
      v977 = v974;
      v1897[0].m128i_i64[0] = v9;
      v978 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1864 = v976;
      v17 = 0;
      v978[1] = (__int64)v977;
      *((_DWORD *)v978 + 4) = v1864;
      goto LABEL_17;
    case 0x90u:
      a2 = *(__int64 **)(v9 - 24);
      a1 = (__int64 *)v10;
      v953 = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v953 + 16) != 13 )
      {
LABEL_716:
        sub_20685E0((__int64)a1, a2, a7, a8, a9);
        BUG();
      }
      v954 = sub_20685E0(v10, a2, a7, a8, a9);
      v955 = *(_DWORD *)(v953 + 32);
      v957 = (unsigned __int8 *)(v954[5] + 16LL * v956);
      v958 = (const void **)*((_QWORD *)v957 + 1);
      v959 = *v957;
      if ( v955 <= 0x40 )
      {
        if ( !*(_QWORD *)(v953 + 24) )
          goto LABEL_366;
      }
      else
      {
        srccd = (const void **)*((_QWORD *)v957 + 1);
        v960 = sub_16A57B0(v953 + 24);
        v958 = srccd;
        if ( v955 == v960 )
        {
LABEL_366:
          v961 = sub_1D38BB0(
                   *(_QWORD *)(v10 + 552),
                   -1,
                   (__int64)&v1878,
                   v959,
                   v958,
                   0,
                   a7,
                   *(double *)a8.m128i_i64,
                   a9,
                   0);
          v963 = v962;
          goto LABEL_367;
        }
      }
      v961 = sub_1D38BB0(*(_QWORD *)(v10 + 552), 0, (__int64)&v1878, v959, v958, 0, a7, *(double *)a8.m128i_i64, a9, 0);
      v963 = v1108;
LABEL_367:
      v1897[0].m128i_i64[0] = v9;
      v964 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v964[1] = v961;
      v17 = 0;
      *((_DWORD *)v964 + 4) = v963;
      goto LABEL_17;
    case 0x91u:
      *(_QWORD *)&v979 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v612 = *(__int64 **)(v10 + 552);
      srcce = v979;
      v980 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v616 = sub_1D332F0(
               v612,
               210,
               (__int64)&v1878,
               1,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v980,
               v981,
               srcce);
      goto LABEL_239;
    case 0x92u:
      v717 = *(__int64 **)(v10 + 552);
      v1716 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
      v719 = v718;
      *((_QWORD *)&srcl + 1) = v718;
      v721 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v722 = (__int64)v721;
      v723 = 16LL * (unsigned int)v720;
      v724 = v720;
      *(_QWORD *)&srcl = v1716;
      v725 = v723 + v721[5];
      v726 = *(_BYTE *)v725;
      if ( *(_BYTE *)v725 != 9 )
        goto LABEL_304;
      if ( *(_BYTE *)(v1716[5] + 16LL * v719) != 9 )
        goto LABEL_304;
      if ( (unsigned int)(dword_4FCEF88 - 1) > 0x11 )
        goto LABEL_304;
      v1337 = *((unsigned __int16 *)v721 + 12);
      if ( v1337 != 11 && v1337 != 33 )
        goto LABEL_304;
      v1580 = v721;
      v1585 = v724;
      v1610 = (__int64)v721;
      a7.m128i_i64[0] = 1092616192;
      v1651 = (__int16 *)sub_1698270();
      sub_169D3B0((__int64)&v1895, (__m128i)0x41200000u);
      sub_169E320(&v1897[0].m128i_i64[1], v1895.m128i_i64, v1651);
      sub_1698460((__int64)&v1895);
      v1652 = v1610;
      v1611 = sub_1D16110(v1610, (__int64)v1897);
      v1338 = sub_16982C0();
      v1339 = v1652;
      v1737 = v1338;
      v722 = (__int64)v1580;
      v724 = v1585;
      if ( (void *)v1897[0].m128i_i64[1] == v1338 )
      {
        v1670 = v1897[1].m128i_i64[0];
        if ( v1897[1].m128i_i64[0] )
        {
          v1581 = v717;
          v1450 = v1339;
          v1555 = v722;
          v1563 = v723;
          v1451 = v1897[1].m128i_i64[0] + 32LL * *(_QWORD *)(v1897[1].m128i_i64[0] - 8);
          while ( v1670 != v1451 )
          {
            v1451 -= 32;
            if ( v1737 == *(void **)(v1451 + 8) )
            {
              v1462 = *(_QWORD *)(v1451 + 16);
              if ( v1462 )
              {
                v1549 = v1451;
                v1463 = *(_QWORD *)(v1451 + 16);
                v1464 = v1462 + 32LL * *(_QWORD *)(v1462 - 8);
                while ( v1463 != v1464 )
                {
                  v1464 -= 32;
                  if ( v1737 == *(void **)(v1464 + 8) )
                  {
                    v1465 = *(_QWORD *)(v1464 + 16);
                    if ( v1465 )
                    {
                      v1545 = v9;
                      v1466 = v10;
                      v1467 = v1464;
                      v1540 = v1450;
                      v1468 = *(_QWORD *)(v1464 + 16);
                      v1469 = v1463;
                      for ( k = v1465 + 32LL * *(_QWORD *)(v1465 - 8); v1468 != k; sub_127D120((_QWORD *)(k + 8)) )
                        k -= 32;
                      v1471 = v1468;
                      v1463 = v1469;
                      v1464 = v1467;
                      v1450 = v1540;
                      v10 = v1466;
                      v9 = v1545;
                      j_j_j___libc_free_0_0(v1471 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v1464 + 8);
                  }
                }
                v1472 = v1463;
                v1451 = v1549;
                j_j_j___libc_free_0_0(v1472 - 8);
              }
            }
            else
            {
              sub_1698460(v1451 + 8);
            }
          }
          v717 = v1581;
          v723 = v1563;
          j_j_j___libc_free_0_0(v1670 - 8);
          v722 = v1555;
          v724 = v1585;
          v1339 = v1450;
        }
      }
      else
      {
        sub_1698460((__int64)&v1897[0].m128i_i64[1]);
        v722 = (__int64)v1580;
        v724 = v1585;
        v1339 = v1652;
      }
      if ( v1611 )
      {
        *(_QWORD *)&v1446 = sub_2048150(
                              (__int64)v717,
                              0x40549A78u,
                              (__int64)&v1878,
                              COERCE_DOUBLE(1092616192),
                              *(double *)a8.m128i_i64,
                              a9);
        v1447 = sub_1D332F0(
                  v717,
                  78,
                  (__int64)&v1878,
                  9,
                  0,
                  0,
                  COERCE_DOUBLE(1092616192),
                  *(double *)a8.m128i_i64,
                  a9,
                  srcl,
                  *((unsigned __int64 *)&srcl + 1),
                  v1446);
        v727 = sub_2048260(
                 (__int64)v1447,
                 v1448,
                 (__int64)&v1878,
                 v717,
                 (__m128i)0x41200000u,
                 *(double *)a8.m128i_i64,
                 a9);
        v729 = v1449;
      }
      else
      {
        v725 = v723 + *(_QWORD *)(v1339 + 40);
        v726 = *(_BYTE *)v725;
LABEL_304:
        v727 = (__int64)sub_1D332F0(
                          v717,
                          168,
                          (__int64)&v1878,
                          v726,
                          *(const void ***)(v725 + 8),
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)a8.m128i_i64,
                          a9,
                          v722,
                          v724,
                          srcl);
        v729 = v728;
      }
      v1897[0].m128i_i64[0] = v9;
      v730 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1866 = v729;
      v17 = 0;
      v730[1] = v727;
      *((_DWORD *)v730 + 4) = v1866;
      goto LABEL_17;
    case 0x93u:
      v798 = *(_QWORD *)(v10 + 552);
      v1553 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))), a7, a8, a9);
      v800 = v799;
      v1559 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v802 = v801;
      v803 = (__int64)v1559;
      v804 = *((unsigned __int16 *)v1553 + 12);
      src_8c = v800;
      if ( v804 != 32 )
      {
        v805 = (unsigned int)v801;
        if ( v804 != 10 )
          goto LABEL_319;
      }
      v1099 = v1553[11];
      v1100 = *(_DWORD *)(v1099 + 32);
      v1101 = *(__int64 **)(v1099 + 24);
      if ( v1100 > 0x40 )
        v1102 = *v1101;
      else
        v1102 = (__int64)((_QWORD)v1101 << (64 - (unsigned __int8)v1100)) >> (64 - (unsigned __int8)v1100);
      v805 = (unsigned int)v801;
      v1544 = 16LL * (unsigned int)v801;
      v1726 = abs32(v1102);
      if ( v1726 )
      {
        v1583 = v801;
        v1638 = v801;
        v1137 = **(_QWORD **)(v798 + 32) + 112LL;
        v1138 = sub_1560180(v1137, 34);
        v1139 = v1638;
        v1140 = v1559;
        v802 = v1583;
        if ( !v1138 )
        {
          v1224 = sub_1560180(v1137, 17);
          v1139 = v1638;
          v1140 = v1559;
          v802 = v1583;
          if ( !v1224 )
            goto LABEL_517;
        }
        v1577 = (__int64)v1140;
        v1584 = v802;
        v1639 = v1139;
        v1141 = sub_39FAC40(v1726);
        _BitScanReverse(&v1142, v1726);
        v1139 = v1639;
        v803 = v1577;
        v802 = v1584;
        if ( v1141 + 31 - (v1142 ^ 0x1F) > 6 )
        {
LABEL_319:
          *((_QWORD *)&v1508 + 1) = src_8c;
          *(_QWORD *)&v1508 = v1553;
          v806 = sub_1D332F0(
                   (__int64 *)v798,
                   167,
                   (__int64)&v1878,
                   *(unsigned __int8 *)(v1559[5] + 16 * v805),
                   *(const void ***)(v1559[5] + 16 * v805 + 8),
                   0,
                   *(double *)a7.m128i_i64,
                   *(double *)a8.m128i_i64,
                   a9,
                   v803,
                   v802,
                   v1508);
        }
        else
        {
LABEL_517:
          v1578 = (__int64 *)v798;
          v1143 = v802;
          v1144 = v1139;
          v1145 = v1559;
          v1541 = v10;
          v1146 = v1538;
          v1147 = v1726;
          v1627 = 0;
          srcn = 0;
          do
          {
            if ( (v1147 & 1) != 0 )
            {
              if ( srcn )
              {
                v1148 = v1546;
                v1149 = srcn[5] + 16LL * v1627;
                LOBYTE(v1148) = *(_BYTE *)v1149;
                v1546 = v1148;
                v1615 = v1627 | v1612 & 0xFFFFFFFF00000000LL;
                *((_QWORD *)&v1514 + 1) = v1144 | v1143 & 0xFFFFFFFF00000000LL;
                *(_QWORD *)&v1514 = v1145;
                v1741 = *((_QWORD *)&v1514 + 1);
                v1150 = sub_1D332F0(
                          v1578,
                          78,
                          (__int64)&v1878,
                          (unsigned int)v1148,
                          *(const void ***)(v1149 + 8),
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)a8.m128i_i64,
                          a9,
                          (__int64)srcn,
                          v1615,
                          v1514);
                v1143 = v1741;
                srcn = v1150;
                v1627 = v1151;
                v1612 = v1151 | v1615 & 0xFFFFFFFF00000000LL;
              }
              else
              {
                v1627 = v1144;
                srcn = v1145;
              }
            }
            v1152 = v1145[5] + 16LL * v1144;
            LOBYTE(v1146) = *(_BYTE *)v1152;
            *((_QWORD *)&v1515 + 1) = v1144 | v1143 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v1515 = v1145;
            v1742 = *((_QWORD *)&v1515 + 1);
            v1145 = sub_1D332F0(
                      v1578,
                      78,
                      (__int64)&v1878,
                      v1146,
                      *(const void ***)(v1152 + 8),
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9,
                      (__int64)v1145,
                      *((unsigned __int64 *)&v1515 + 1),
                      v1515);
            v1144 = v1153;
            v1147 >>= 1;
            v1143 = v1153 | v1742 & 0xFFFFFFFF00000000LL;
          }
          while ( v1147 );
          v10 = v1541;
          v1172 = v1553[11];
          v1173 = *(_DWORD *)(v1172 + 32);
          v1174 = *(__int64 **)(v1172 + 24);
          if ( v1173 > 0x40 )
            v1175 = *v1174;
          else
            v1175 = (__int64)((_QWORD)v1174 << (64 - (unsigned __int8)v1173)) >> (64 - (unsigned __int8)v1173);
          if ( v1175 >= 0 )
            goto LABEL_321;
          v1176 = sub_1D364E0(
                    (__int64)v1578,
                    (__int64)&v1878,
                    *(unsigned __int8 *)(v1544 + v1559[5]),
                    *(const void ***)(v1544 + v1559[5] + 8),
                    0,
                    1.0,
                    *(double *)a8.m128i_i64,
                    a9);
          *((_QWORD *)&v1517 + 1) = v1627 | v1612 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v1517 = srcn;
          v806 = sub_1D332F0(
                   v1578,
                   79,
                   (__int64)&v1878,
                   *(unsigned __int8 *)(v1559[5] + v1544),
                   *(const void ***)(v1559[5] + v1544 + 8),
                   0,
                   1.0,
                   *(double *)a8.m128i_i64,
                   a9,
                   (__int64)v1176,
                   v1177,
                   v1517);
        }
        srcn = v806;
        v1627 = v807;
        goto LABEL_321;
      }
      srcn = sub_1D364E0(
               v798,
               (__int64)&v1878,
               *(unsigned __int8 *)(v1559[5] + 16LL * (unsigned int)v801),
               *(const void ***)(v1559[5] + 16LL * (unsigned int)v801 + 8),
               0,
               1.0,
               *(double *)a8.m128i_i64,
               a9);
      v1627 = v1103;
LABEL_321:
      v17 = 0;
      v1897[0].m128i_i64[0] = v9;
      v808 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v808[1] = (__int64)srcn;
      *((_DWORD *)v808 + 4) = v1627;
LABEL_17:
      if ( v1877 )
        sub_161E7C0((__int64)&v1877, (__int64)v1877);
      if ( v1878 )
        sub_161E7C0((__int64)&v1878, v1878);
      return v17;
    case 0x94u:
      v764 = *(_DWORD *)(v9 + 20);
      memset(v1897, 0, 80);
      v765 = v764 & 0xFFFFFFF;
      v766 = *(_QWORD *)(v9 + 24 * (1 - v765));
      v767 = *(_DWORD *)(v766 + 32) <= 0x40u;
      v768 = *(_QWORD **)(v766 + 24);
      if ( !v767 )
        v768 = (_QWORD *)*v768;
      v769 = (_DWORD)v768 == 0;
      v770 = *(_QWORD *)(v10 + 552);
      v771 = 1 - (v769 - 1);
      v1897[0].m128i_i64[0] = *(_QWORD *)(v770 + 176);
      v1897[0].m128i_i32[2] = *(_DWORD *)(v770 + 184);
      v1844 = sub_20685E0(v10, *(__int64 **)(v9 - 24 * v765), (__m128i)0LL, a8, a9);
      v1845 = v772;
      v773 = *(_DWORD *)(v9 + 20);
      v1897[1].m128i_i64[0] = (__int64)v1844;
      v1897[1].m128i_i32[2] = v1845;
      v1842 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1LL - (v773 & 0xFFFFFFF))), (__m128i)0LL, a8, a9);
      v1843 = v774;
      v775 = *(_DWORD *)(v9 + 20);
      v1897[2].m128i_i64[0] = (__int64)v1842;
      v1897[2].m128i_i32[2] = v1843;
      v1840 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (2LL - (v775 & 0xFFFFFFF))), (__m128i)0LL, a8, a9);
      v1841 = v776;
      v777 = *(_DWORD *)(v9 + 20);
      v1897[3].m128i_i64[0] = (__int64)v1840;
      v1897[3].m128i_i32[2] = v1841;
      v778 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (3LL - (v777 & 0xFFFFFFF))), (__m128i)0LL, a8, a9);
      v1895 = 0u;
      v1897[4].m128i_i64[0] = (__int64)v778;
      v1897[4].m128i_i32[2] = v779;
      v780 = *(_QWORD **)(v10 + 552);
      *(_QWORD *)&v1896[0] = 0;
      srcm = v780;
      v781 = *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      if ( v781 )
      {
        v1893 = (__m128i)*(unsigned __int64 *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
        LOBYTE(v1894[0]) = 0;
        v781 = (__int64 *)*v781;
        if ( *((_BYTE *)v781 + 8) == 16 )
          v781 = *(__int64 **)v781[2];
        LODWORD(v781) = *((_DWORD *)v781 + 2) >> 8;
      }
      else
      {
        v1893 = 0u;
        v1894[0] = 0;
      }
      v782 = *(_QWORD **)(v10 + 768);
      HIDWORD(v1894[0]) = (_DWORD)v781;
      v783 = sub_1F7DE30(v782, 8u);
      v785 = v784;
      v786 = v783;
      v790 = sub_1D29190(*(_QWORD *)(v10 + 552), 1u, 0, v787, v788, v789);
      v1478 = v786;
      v17 = 0;
      v792.m128i_i64[0] = sub_1D251C0(
                            srcm,
                            217,
                            (__int64)&v1878,
                            v790,
                            v791,
                            0,
                            v1897[0].m128i_i64,
                            5,
                            v1478,
                            v785,
                            *(_OWORD *)&v1893,
                            v1894[0],
                            v771,
                            0,
                            (__int64)&v1895);
      v1891 = v792;
      sub_1D23890(v10 + 104, &v1891, v792.m128i_i64[1], v793, v794, v795);
      v1891.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, 0.0, *(double *)a8.m128i_i64, a9);
      v796 = *(_QWORD *)(v10 + 552);
      v1891.m128i_i32[2] = v797;
      sub_2045100(v796, v1891.m128i_i64[0], v797);
      goto LABEL_17;
    case 0xB8u:
      srcbw = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v747 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v749 = v748;
      *(_QWORD *)&v1717 = sub_1D2AF60(*(_QWORD **)(v10 + 552), *(_QWORD *)(srcbw + 24), v748, srcbw, v750, v751);
      *((_QWORD *)&v1717 + 1) = v752;
      srcbx = *(_QWORD *)v9;
      v753 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      LOBYTE(v754) = sub_204D4D0((__int64)v14, v753, srcbx);
      v755 = *(__int64 **)(v10 + 552);
      v757 = (const void ***)sub_1D252B0((__int64)v755, v754, v756, 1, 0);
      *((_QWORD *)&v1485 + 1) = v749;
      *(_QWORD *)&v1485 = v747;
      v760 = sub_1D37440(
               v755,
               24,
               (__int64)&v1878,
               v757,
               v758,
               v759,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               v1485,
               v1717);
      LODWORD(v755) = v761;
      v1897[0].m128i_i64[0] = v9;
      v762 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v763 = (__int64)v760;
      v762[1] = (__int64)v760;
      v17 = 0;
      *((_DWORD *)v762 + 4) = (_DWORD)v755;
      sub_2045100(*(_QWORD *)(v10 + 552), v763, 1);
      goto LABEL_17;
    case 0xB9u:
      v530 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v531 = *(__int64 **)(v10 + 552);
      v533 = v532;
      v534 = v530;
      v535 = (const void ***)sub_1D252B0((__int64)v531, 6, 0, 1, 0);
      *((_QWORD *)&v1501 + 1) = v533;
      *(_QWORD *)&v1501 = v534;
      v292 = sub_1D37410(
               v531,
               211,
               (__int64)&v1878,
               v535,
               v536,
               v537,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               v1501);
      goto LABEL_150;
    case 0xBAu:
      v520 = *(__int64 **)(v10 + 552);
      v521 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v523 = v522;
      v524 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v525 = sub_2046180(v524, 0);
      *((_QWORD *)&v1500 + 1) = v523;
      *(_QWORD *)&v1500 = v521;
      v526 = sub_1D309E0(
               v520,
               21,
               (__int64)&v1878,
               v525,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               v1500);
      v1897[0].m128i_i64[0] = v9;
      LODWORD(v521) = v527;
      v528 = v526;
      v529 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v1876 = (int)v521;
      v17 = 0;
      v529[1] = v528;
      *((_DWORD *)v529 + 4) = v1876;
      goto LABEL_17;
    case 0xBDu:
    case 0xC3u:
    case 0xC6u:
    case 0xD1u:
    case 0xD2u:
    case 0xD3u:
      switch ( a3 )
      {
        case 0xBDu:
          v1055 = 70;
          goto LABEL_415;
        case 0xC3u:
          v1055 = 74;
          goto LABEL_415;
        case 0xC6u:
          v1055 = 72;
          goto LABEL_415;
        case 0xD1u:
          v1055 = 71;
          goto LABEL_415;
        case 0xD2u:
          v1055 = 75;
          goto LABEL_415;
        case 0xD3u:
          v1055 = 73;
LABEL_415:
          v1056 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
          v1058 = v1057;
          v1059 = v1056;
          *(_QWORD *)&srcci = sub_20685E0(
                                v10,
                                *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))),
                                a7,
                                a8,
                                a9);
          v1060 = (unsigned __int8 *)(v1059[5] + 16LL * (unsigned int)v1058);
          *((_QWORD *)&srcci + 1) = v1061;
          v1062 = (const void ***)sub_1D252B0(*(_QWORD *)(v10 + 552), *v1060, *((_QWORD *)v1060 + 1), 2, 0);
          *((_QWORD *)&v1487 + 1) = v1058;
          *(_QWORD *)&v1487 = v1059;
          v1065 = sub_1D37440(
                    *(__int64 **)(v10 + 552),
                    v1055,
                    (__int64)&v1878,
                    v1062,
                    v1063,
                    v1064,
                    *(double *)a7.m128i_i64,
                    *(double *)a8.m128i_i64,
                    a9,
                    v1487,
                    srcci);
          LODWORD(v1059) = v1066;
          v1067 = v1065;
          v1897[0].m128i_i64[0] = v9;
          v1068 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
          v1846 = (int)v1059;
          v17 = 0;
          v1068[1] = (__int64)v1067;
          *((_DWORD *)v1068 + 4) = v1846;
          goto LABEL_17;
        default:
          goto LABEL_716;
      }
    case 0xBEu:
      v17 = &aBuiltinSetjmp[(v14[3].m128i_i8[10] ^ 1) + 9];
      goto LABEL_17;
    case 0xC7u:
      v699 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL));
      v700 = (unsigned __int8)sub_2046180(v699, 0);
      v701 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(v10 + 552) + 32LL) + 40LL);
      v1893.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1893.m128i_i64[1] = v702;
      v703 = v14->m128i_i64[0];
      v704 = *(__int64 (**)())(v14->m128i_i64[0] + 1472);
      if ( v704 == sub_2043CB0 )
        goto LABEL_296;
      if ( ((unsigned __int8 (__fastcall *)(__m128i *))v704)(v14) )
      {
        v712 = sub_2046710(*(_QWORD **)(v10 + 552), (__int64)&v1878, (__int128 *)v1893.m128i_i8);
        v714 = v1180;
        src_8b = v1180;
      }
      else
      {
        v703 = v14->m128i_i64[0];
LABEL_296:
        v705 = (__int64 *)(*(__int64 (__fastcall **)(__m128i *, __int64))(v703 + 536))(v14, v701);
        v706 = sub_15AAE50(*(_QWORD *)(v10 + 560), *v705);
        v1895 = (__m128i)(unsigned __int64)v705;
        v707 = *(_QWORD **)(v10 + 552);
        memset(v1897, 0, 24);
        v708 = v706;
        LOBYTE(v1896[0]) = 0;
        v709 = *v705;
        if ( *(_BYTE *)(*v705 + 8) == 16 )
          v709 = **(_QWORD **)(v709 + 16);
        v1715 = v708;
        srcbu = v707;
        DWORD1(v1896[0]) = *(_DWORD *)(v709 + 8) >> 8;
        v710 = sub_20685E0(v10, v705, a7, a8, a9);
        v712 = sub_1D2B730(
                 srcbu,
                 v700,
                 0,
                 (__int64)&v1878,
                 v1893.m128i_i64[0],
                 v1893.m128i_i64[1],
                 (__int64)v710,
                 v711,
                 *(_OWORD *)&v1895,
                 *(__int64 *)&v1896[0],
                 v1715,
                 4u,
                 (__int64)v1897,
                 0);
        v714 = v713;
        src_8b = v713;
      }
      v715 = *(__int64 (**)())(v14->m128i_i64[0] + 544);
      if ( v715 != sub_1F2AB40 )
      {
        v1729 = v714;
        v1182 = ((__int64 (__fastcall *)(__m128i *))v715)(v14);
        v714 = v1729;
        if ( v1182 )
        {
          v712 = (*(__int64 (__fastcall **)(__m128i *, _QWORD, __int64, unsigned __int64, __int64 *))(v14->m128i_i64[0] + 1480))(
                   v14,
                   *(_QWORD *)(v10 + 552),
                   v712,
                   v1729 | src_8b & 0xFFFFFFFF00000000LL,
                   &v1878);
          v714 = v1183;
        }
      }
      srcbv = v714;
      sub_2045100(*(_QWORD *)(v10 + 552), v1893.m128i_i64[0], v1893.m128i_i32[2]);
      v1897[0].m128i_i64[0] = v9;
      v716 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v716[1] = v712;
      v17 = 0;
      *((_DWORD *)v716 + 4) = srcbv;
      goto LABEL_17;
    case 0xC8u:
      v681 = *(_QWORD *)(*(_QWORD *)(v10 + 552) + 32LL);
      srck = *(_QWORD *)(v681 + 56);
      v682 = sub_1E0A0C0(v681);
      v683 = (unsigned __int8)sub_2046180(v682, 0);
      v1893.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1893.m128i_i64[1] = v684;
      v685 = *(__int64 (**)())(v14->m128i_i64[0] + 1472);
      if ( v685 == sub_2043CB0 || !((unsigned __int8 (__fastcall *)(__m128i *))v685)(v14) )
      {
        v686 = (__int64)sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
        v688 = v687;
      }
      else
      {
        v686 = sub_2046710(*(_QWORD **)(v10 + 552), (__int64)&v1878, (__int128 *)v1893.m128i_i8);
        v688 = v1184;
      }
      v1595 = v688;
      v1891.m128i_i64[0] = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v689 = *((_DWORD *)sub_2061B80(*(_QWORD *)(v10 + 712) + 336LL, v1891.m128i_i64) + 2);
      *(_DWORD *)(srck + 68) = v689;
      v1714 = v689;
      v690 = sub_1D299D0(*(_QWORD **)(v10 + 552), v689, v683, 0, 0);
      v691 = *(_QWORD **)(v10 + 552);
      src_8h = v692;
      v1895 = 0u;
      *(_QWORD *)&v1896[0] = 0;
      srcbt = (__int64)v690;
      v1626 = v691;
      sub_1E341E0((__int64)v1897, v691[4], v1714, 0);
      v693 = sub_1D2BF40(
               v1626,
               v1893.m128i_i64[0],
               v1893.m128i_i64[1],
               (__int64)&v1878,
               v686,
               v1595,
               srcbt,
               src_8h,
               *(_OWORD *)v1897,
               v1897[1].m128i_i64[0],
               0,
               4u,
               (__int64)&v1895);
      v695 = v694;
      v1895.m128i_i64[0] = v9;
      v696 = v694;
      v697 = sub_205F5C0(v10 + 8, v1895.m128i_i64);
      v698 = v696;
      v17 = 0;
      v697[1] = v693;
      *((_DWORD *)v697 + 4) = v695;
      sub_2045100(*(_QWORD *)(v10 + 552), v693, v698);
      goto LABEL_17;
    case 0xC9u:
      *(_QWORD *)&srcbs = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v612 = *(__int64 **)(v10 + 552);
      *((_QWORD *)&srcbs + 1) = v678;
      v679 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v616 = sub_1D332F0(
               v612,
               200,
               (__int64)&v1878,
               1,
               0,
               0,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               (__int64)v679,
               v680,
               srcbs);
LABEL_239:
      v618 = (__int64)v612;
      v17 = 0;
      sub_2045100(v618, (__int64)v616, v617);
      goto LABEL_17;
    case 0xCAu:
      v282 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v283 = *(__int64 **)(v10 + 552);
      v285 = v284;
      v286 = v282;
      v287 = sub_1E0A0C0(v283[4]);
      v288 = sub_2046180(v287, 0);
      v289 = (const void ***)sub_1D252B0((__int64)v283, v288, 0, 1, 0);
      *((_QWORD *)&v1493 + 1) = v285;
      *(_QWORD *)&v1493 = v286;
      v292 = sub_1D37410(
               v283,
               199,
               (__int64)&v1878,
               v289,
               v290,
               v291,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               a9,
               v1493);
LABEL_150:
      v1897[0].m128i_i64[0] = v9;
      v294 = v292;
      v295 = v293;
      v296 = sub_205F5C0(v10 + 8, v1897[0].m128i_i64);
      v296[1] = (__int64)v294;
      *((_DWORD *)v296 + 4) = v295;
LABEL_151:
      v297 = (__int64)v294;
      v17 = 0;
      sub_2045100(*(_QWORD *)(v10 + 552), v297, 1);
      goto LABEL_17;
    case 0xD4u:
      v17 = 0;
      sub_207E6C0(v10, v9, a7, a8, a9, v16, a4, a5, a6);
      goto LABEL_17;
    case 0xD5u:
      v17 = 0;
      sub_207E520(v10, v9, a7, a8, a9, v16, a4, a5, a6);
      goto LABEL_17;
    case 0xD6u:
      v17 = 0;
      sub_207DD20(v10, v9, a7, a8, a9, v16, a4, a5, a6);
      goto LABEL_17;
    case 0xD8u:
      v940 = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v1721 = *(__int64 **)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      v941 = sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v943 = (__int16 *)v942;
      *(_QWORD *)&v947 = sub_1D2AF60(*(_QWORD **)(v10 + 552), *(_QWORD *)(v940 + 24), v942, v944, v945, v946);
      v948 = *(__int64 **)(v10 + 552);
      srccc = v947;
      v949 = sub_20685E0(v10, v1721, a7, a8, a9);
      v1480 = (unsigned __int64)v941;
      v17 = 0;
      v951 = sub_1D3A900(
               v948,
               0x19u,
               (__int64)&v1878,
               1u,
               0,
               0,
               (__m128)a7,
               *(double *)a8.m128i_i64,
               a9,
               v1480,
               v943,
               srccc,
               (__int64)v949,
               v950);
      sub_2045100((__int64)v948, (__int64)v951, v952);
      goto LABEL_17;
    case 0xD9u:
      v919 = **(_QWORD **)(v10 + 552);
      if ( *(_DWORD *)(v919 + 504) != 32 || *(_DWORD *)(v919 + 516) != 9 )
        goto LABEL_79;
      v1560 = &v1889;
      sub_204D410((__int64)&v1889, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
      v1897[0].m128i_i64[0] = (__int64)v1897[1].m128i_i64;
      v1897[0].m128i_i64[1] = 0x800000000LL;
      v920 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1891.m128i_i64[1] = v921;
      LODWORD(v921) = *(_DWORD *)(v9 + 20);
      v1891.m128i_i64[0] = (__int64)v920;
      v922.m128i_i64[0] = (__int64)sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1 - (v921 & 0xFFFFFFF))), a7, a8, a9);
      v923 = *(_QWORD *)(v10 + 552);
      v1893 = v922;
      v924 = sub_1D252B0(v923, 1, 0, 111, 0);
      v926 = v925;
      srccb = v924;
      v927.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1895 = v927;
      sub_1D23890((__int64)v1897, &v1891, v927.m128i_i64[1], v928, v929, v930);
      sub_1D23890((__int64)v1897, &v1893, v931, v932, v933, v934);
      sub_1D23890((__int64)v1897, &v1895, v935, v936, v937, v938);
      v916 = sub_1D23DE0(
               *(_QWORD **)(v10 + 552),
               31,
               (__int64)&v1889,
               srccb,
               v926,
               v939,
               (__int64 *)v1897[0].m128i_i64[0],
               v1897[0].m128i_u32[2]);
      sub_2045100(*(_QWORD *)(v10 + 552), v916, 0);
      v1887.m128i_i64[0] = v9;
      v917 = &v1887;
      goto LABEL_356;
    case 0xDAu:
      v889 = **(_QWORD **)(v10 + 552);
      if ( *(_DWORD *)(v889 + 504) != 32 || *(_DWORD *)(v889 + 516) != 9 )
        goto LABEL_79;
      v1560 = &v1887;
      sub_204D410((__int64)&v1887, *(_QWORD *)v10, *(_DWORD *)(v10 + 536));
      v1897[0].m128i_i64[0] = (__int64)v1897[1].m128i_i64;
      v1897[0].m128i_i64[1] = 0x800000000LL;
      v890 = sub_20685E0(v10, *(__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)), a7, a8, a9);
      v1889.m128i_i64[1] = v891;
      LODWORD(v891) = *(_DWORD *)(v9 + 20);
      v1889.m128i_i64[0] = (__int64)v890;
      v892 = sub_20685E0(v10, *(__int64 **)(v9 + 24 * (1 - (v891 & 0xFFFFFFF))), a7, a8, a9);
      v1891.m128i_i64[1] = v893;
      LODWORD(v893) = *(_DWORD *)(v9 + 20);
      v1891.m128i_i64[0] = (__int64)v892;
      v894.m128i_i64[0] = (__int64)sub_20685E0(v10, *(__int64 **)(v9 + 24 * (2 - (v893 & 0xFFFFFFF))), a7, a8, a9);
      v895 = *(_QWORD *)(v10 + 552);
      v1893 = v894;
      v896 = sub_1D252B0(v895, 1, 0, 111, 0);
      v898 = v897;
      srcca = v896;
      v899.m128i_i64[0] = (__int64)sub_2051C20((__int64 *)v10, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v1895 = v899;
      sub_1D23890((__int64)v1897, &v1889, v899.m128i_i64[1], v900, v901, v902);
      sub_1D23890((__int64)v1897, &v1891, v903, v904, v905, v906);
      sub_1D23890((__int64)v1897, &v1893, v907, v908, v909, v910);
      sub_1D23890((__int64)v1897, &v1895, v911, v912, v913, v914);
      v916 = sub_1D23DE0(
               *(_QWORD **)(v10 + 552),
               32,
               (__int64)&v1887,
               srcca,
               v898,
               v915,
               (__int64 *)v1897[0].m128i_i64[0],
               v1897[0].m128i_u32[2]);
      sub_2045100(*(_QWORD *)(v10 + 552), v916, 0);
      v1885.m128i_i64[0] = v9;
      v917 = &v1885;
LABEL_356:
      v918 = sub_205F5C0(v10 + 8, v917->m128i_i64);
      v918[1] = v916;
      *((_DWORD *)v918 + 4) = 0;
      if ( (__m128i *)v1897[0].m128i_i64[0] != &v1897[1] )
        _libc_free(v1897[0].m128i_u64[0]);
      sub_17CD270(v1560->m128i_i64);
      goto LABEL_79;
    default:
      goto LABEL_16;
  }
}
