// Function: sub_12D6300
// Address: 0x12d6300
//
__int64 __fastcall sub_12D6300(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  int v3; // r13d
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // r8d
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r14
  int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // r8d
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r14
  int v19; // r12d
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // r8d
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // r14
  int v26; // r12d
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // r8d
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // r14
  int v33; // r12d
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // r8d
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // r14
  int v40; // r12d
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // r8d
  __int64 v44; // rdx
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r12
  __int64 v49; // r14
  bool v50; // zf
  __int64 v51; // r14
  int v52; // r12d
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // r8d
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rsi
  const char *v59; // rdi
  __int64 v60; // r13
  int v61; // r15d
  int v62; // r14d
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r13
  __int64 v66; // r15
  __int64 v67; // rax
  int v68; // eax
  int v69; // r14d
  __int64 v70; // rax
  int v71; // r8d
  __int64 v72; // rdx
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r13
  __int64 v77; // r15
  __int64 v78; // r15
  __int64 v79; // rax
  int v80; // eax
  int v81; // r14d
  __int64 v82; // rax
  int v83; // r8d
  __int64 v84; // rdx
  __int64 v85; // r14
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // r13
  __int64 v89; // r15
  __int64 v90; // r15
  __int64 v91; // rax
  int v92; // eax
  int v93; // r14d
  __int64 v94; // rax
  int v95; // r8d
  __int64 v96; // rsi
  __int64 v97; // rdx
  __int64 v98; // r15
  __int64 v99; // r13
  __int64 v100; // rax
  int v101; // ecx
  __int64 v102; // rdx
  __int64 v103; // rax
  int v104; // eax
  int v105; // r14d
  __int64 v106; // rax
  int v107; // r8d
  __int64 v108; // rsi
  __int64 v109; // rdx
  __int64 v110; // r15
  __int64 v111; // r13
  __int64 v112; // rax
  int v113; // ecx
  __int64 v114; // rdx
  __int64 v115; // rax
  int v116; // eax
  int v117; // r14d
  __int64 v118; // rax
  int v119; // r8d
  __int64 v120; // rsi
  __int64 v121; // rdx
  __int64 v122; // r15
  __int64 v123; // r13
  __int64 v124; // rax
  int v125; // ecx
  __int64 v126; // rdx
  __int64 v127; // rax
  int v128; // eax
  int v129; // r14d
  __int64 v130; // rax
  int v131; // r8d
  __int64 v132; // rsi
  __int64 v133; // rdx
  __int64 v134; // r15
  __int64 v135; // r13
  __int64 v136; // rax
  int v137; // ecx
  __int64 v138; // rdx
  __int64 v139; // rax
  int v140; // eax
  int v141; // r14d
  __int64 v142; // rax
  int v143; // r8d
  __int64 v144; // rsi
  __int64 v145; // rdx
  __int64 v146; // r15
  __int64 v147; // r13
  __int64 v148; // rax
  int v149; // ecx
  __int64 v150; // rdx
  __int64 v151; // rax
  int v152; // eax
  int v153; // r14d
  __int64 v154; // rax
  int v155; // r8d
  __int64 v156; // rsi
  __int64 v157; // rdx
  __int64 v158; // r15
  __int64 v159; // r13
  __int64 v160; // rax
  int v161; // ecx
  __int64 v162; // rdx
  __int64 v163; // rax
  int v164; // eax
  int v165; // r14d
  __int64 v166; // rax
  int v167; // r8d
  __int64 v168; // rsi
  __int64 v169; // rdx
  __int64 v170; // r15
  __int64 v171; // r13
  __int64 v172; // rax
  int v173; // ecx
  __int64 v174; // rdx
  __int64 v175; // rax
  int v176; // eax
  int v177; // r14d
  __int64 v178; // rax
  int v179; // r8d
  __int64 v180; // rsi
  __int64 v181; // rdx
  __int64 v182; // r15
  __int64 v183; // r13
  __int64 v184; // rax
  int v185; // ecx
  __int64 v186; // rdx
  __int64 v187; // rax
  int v188; // eax
  int v189; // r14d
  __int64 v190; // rax
  int v191; // r8d
  __int64 v192; // rsi
  __int64 v193; // rdx
  __int64 v194; // r15
  __int64 v195; // r13
  __int64 v196; // rax
  int v197; // ecx
  __int64 v198; // rdx
  __int64 v199; // rax
  int v200; // eax
  int v201; // r14d
  __int64 v202; // rax
  int v203; // r8d
  __int64 v204; // rsi
  __int64 v205; // rdx
  __int64 v206; // r15
  __int64 v207; // r13
  __int64 v208; // rax
  int v209; // ecx
  __int64 v210; // rdx
  __int64 v211; // rax
  int v212; // eax
  int v213; // r14d
  __int64 v214; // rax
  int v215; // r8d
  __int64 v216; // rsi
  __int64 v217; // rdx
  __int64 v218; // r15
  __int64 v219; // r13
  __int64 v220; // rax
  int v221; // ecx
  __int64 v222; // rdx
  __int64 v223; // rax
  int v224; // eax
  int v225; // r14d
  __int64 v226; // rax
  int v227; // r8d
  __int64 v228; // rsi
  __int64 v229; // rdx
  __int64 v230; // r15
  __int64 v231; // r13
  __int64 v232; // rax
  int v233; // ecx
  __int64 v234; // rdx
  __int64 v235; // rax
  int v236; // eax
  int v237; // r14d
  __int64 v238; // rax
  int v239; // r8d
  __int64 v240; // rsi
  __int64 v241; // rdx
  __int64 v242; // r15
  __int64 v243; // r13
  __int64 v244; // rax
  int v245; // ecx
  __int64 v246; // rdx
  __int64 v247; // rax
  int v248; // eax
  int v249; // r14d
  __int64 v250; // rax
  int v251; // r8d
  __int64 v252; // rsi
  __int64 v253; // rdx
  __int64 v254; // r15
  __int64 v255; // r13
  __int64 v256; // rax
  int v257; // ecx
  __int64 v258; // rdx
  __int64 v259; // rax
  int v260; // eax
  int v261; // r14d
  __int64 v262; // rax
  int v263; // r8d
  __int64 v264; // rsi
  __int64 v265; // rdx
  __int64 v266; // r15
  __int64 v267; // r13
  __int64 v268; // rax
  int v269; // ecx
  __int64 v270; // rdx
  __int64 v271; // rax
  int v272; // eax
  int v273; // r14d
  __int64 v274; // rax
  int v275; // r8d
  __int64 v276; // rsi
  __int64 v277; // rdx
  __int64 v278; // r15
  __int64 v279; // r13
  __int64 v280; // rax
  int v281; // ecx
  __int64 v282; // rdx
  __int64 v283; // rax
  int v284; // eax
  int v285; // r14d
  __int64 v286; // rax
  int v287; // r8d
  __int64 v288; // rsi
  __int64 v289; // rdx
  __int64 v290; // r15
  __int64 v291; // r13
  __int64 v292; // rax
  int v293; // ecx
  __int64 v294; // rdx
  __int64 v295; // rax
  int v296; // eax
  int v297; // r14d
  __int64 v298; // rax
  int v299; // r8d
  __int64 v300; // rdx
  __int64 v301; // r14
  __int64 v302; // rax
  __int64 v303; // rdx
  __int64 v304; // r13
  __int64 v305; // r15
  __int64 v306; // r15
  __int64 v307; // rax
  int v308; // eax
  int v309; // r14d
  __int64 v310; // rax
  int v311; // r8d
  __int64 v312; // rsi
  __int64 v313; // rdx
  __int64 v314; // r15
  __int64 v315; // r13
  __int64 v316; // rax
  int v317; // ecx
  __int64 v318; // rdx
  __int64 v319; // rax
  int v320; // eax
  int v321; // r14d
  __int64 v322; // rax
  int v323; // r8d
  __int64 v324; // rdx
  __int64 v325; // r14
  __int64 v326; // rax
  __int64 v327; // rdx
  __int64 v328; // r13
  __int64 v329; // r15
  __int64 v330; // r15
  __int64 v331; // rax
  int v332; // eax
  int v333; // r14d
  __int64 v334; // rax
  int v335; // r8d
  __int64 v336; // rdx
  __int64 v337; // r14
  __int64 v338; // rax
  __int64 v339; // rdx
  __int64 v340; // r13
  __int64 v341; // r15
  __int64 v342; // r15
  __int64 v343; // rax
  int v344; // eax
  int v345; // r14d
  __int64 v346; // rax
  int v347; // r8d
  __int64 v348; // rsi
  __int64 v349; // rdx
  __int64 v350; // r15
  __int64 v351; // r13
  __int64 v352; // rax
  int v353; // ecx
  __int64 v354; // rdx
  __int64 v355; // rax
  int v356; // eax
  int v357; // r14d
  __int64 v358; // rax
  int v359; // r8d
  __int64 v360; // rdx
  __int64 v361; // r14
  __int64 v362; // rax
  __int64 v363; // rdx
  __int64 v364; // r13
  __int64 v365; // r15
  __int64 v366; // r15
  __int64 v367; // rax
  int v368; // eax
  int v369; // r14d
  __int64 v370; // rax
  int v371; // r8d
  __int64 v372; // rdx
  __int64 v373; // r14
  __int64 v374; // rax
  __int64 v375; // rdx
  __int64 v376; // r13
  __int64 v377; // r15
  __int64 v378; // r15
  __int64 v379; // rax
  int v380; // eax
  int v381; // r14d
  __int64 v382; // rax
  int v383; // r8d
  __int64 v384; // rsi
  __int64 v385; // rdx
  __int64 v386; // r15
  __int64 v387; // r13
  __int64 v388; // rax
  int v389; // ecx
  __int64 v390; // rdx
  __int64 v391; // rax
  int v392; // eax
  int v393; // r14d
  __int64 v394; // rax
  int v395; // r8d
  __int64 v396; // rsi
  __int64 v397; // rdx
  __int64 v398; // r15
  __int64 v399; // r13
  __int64 v400; // rax
  int v401; // ecx
  __int64 v402; // rdx
  __int64 v403; // rax
  int v404; // eax
  int v405; // r14d
  __int64 v406; // rax
  int v407; // r8d
  __int64 v408; // rsi
  __int64 v409; // rdx
  __int64 v410; // r15
  __int64 v411; // r13
  __int64 v412; // rax
  int v413; // ecx
  __int64 v414; // rdx
  __int64 v415; // rax
  int v416; // eax
  int v417; // r14d
  __int64 v418; // rax
  int v419; // r8d
  __int64 v420; // rsi
  __int64 v421; // rdx
  __int64 v422; // r15
  __int64 v423; // r13
  __int64 v424; // rax
  int v425; // ecx
  __int64 v426; // rdx
  __int64 v427; // rax
  int v428; // eax
  int v429; // r14d
  __int64 v430; // rax
  int v431; // r8d
  __int64 v432; // rsi
  __int64 v433; // rdx
  __int64 v434; // r15
  __int64 v435; // r13
  __int64 v436; // rax
  int v437; // ecx
  __int64 v438; // rdx
  __int64 v439; // rax
  int v440; // eax
  int v441; // r14d
  __int64 v442; // rax
  int v443; // r8d
  __int64 v444; // rsi
  __int64 v445; // rdx
  __int64 v446; // r15
  __int64 v447; // r13
  __int64 v448; // rax
  int v449; // ecx
  __int64 v450; // rdx
  __int64 v451; // rax
  int v452; // eax
  int v453; // r14d
  __int64 v454; // rax
  int v455; // r8d
  __int64 v456; // rsi
  __int64 v457; // rdx
  __int64 v458; // r15
  __int64 v459; // r13
  __int64 v460; // rax
  int v461; // ecx
  __int64 v462; // rdx
  __int64 v463; // rax
  int v464; // eax
  int v465; // r14d
  __int64 v466; // rax
  int v467; // r8d
  __int64 v468; // rsi
  __int64 v469; // rdx
  __int64 v470; // r15
  __int64 v471; // r13
  __int64 v472; // rax
  int v473; // ecx
  __int64 v474; // rdx
  __int64 v475; // rax
  int v476; // eax
  int v477; // r14d
  __int64 v478; // rax
  int v479; // r8d
  __int64 v480; // rsi
  __int64 v481; // rdx
  __int64 v482; // r15
  __int64 v483; // r13
  __int64 v484; // rax
  int v485; // ecx
  __int64 v486; // rdx
  __int64 v487; // rax
  int v488; // eax
  int v489; // r14d
  __int64 v490; // rax
  int v491; // r8d
  __int64 v492; // rsi
  __int64 v493; // rdx
  __int64 v494; // r15
  __int64 v495; // r13
  __int64 v496; // rax
  int v497; // ecx
  __int64 v498; // rdx
  __int64 v499; // rax
  int v500; // eax
  int v501; // r14d
  __int64 v502; // rax
  int v503; // r8d
  __int64 v504; // rsi
  __int64 v505; // rdx
  __int64 v506; // r15
  __int64 v507; // r13
  __int64 v508; // rax
  int v509; // ecx
  __int64 v510; // rdx
  __int64 v511; // rax
  int v512; // eax
  int v513; // r14d
  __int64 v514; // rax
  int v515; // r8d
  __int64 v516; // rsi
  __int64 v517; // rdx
  __int64 v518; // r15
  __int64 v519; // r13
  __int64 v520; // rax
  int v521; // ecx
  __int64 v522; // rdx
  __int64 v523; // rax
  int v524; // eax
  int v525; // r14d
  __int64 v526; // rax
  int v527; // r8d
  __int64 v528; // rsi
  __int64 v529; // rdx
  __int64 v530; // r15
  __int64 v531; // r13
  __int64 v532; // rax
  int v533; // ecx
  __int64 v534; // rdx
  __int64 v535; // rax
  int v536; // eax
  int v537; // r14d
  __int64 v538; // rax
  int v539; // r8d
  __int64 v540; // rsi
  __int64 v541; // rdx
  __int64 v542; // r15
  __int64 v543; // r13
  __int64 v544; // rax
  int v545; // ecx
  __int64 v546; // rdx
  __int64 v547; // rax
  int v548; // eax
  int v549; // r14d
  __int64 v550; // rax
  int v551; // r8d
  __int64 v552; // rsi
  __int64 v553; // rdx
  __int64 v554; // r15
  __int64 v555; // r13
  __int64 v556; // rax
  int v557; // ecx
  __int64 v558; // rdx
  __int64 v559; // rax
  int v560; // eax
  int v561; // r14d
  __int64 v562; // rax
  int v563; // r8d
  __int64 v564; // rsi
  __int64 v565; // rdx
  __int64 v566; // r15
  __int64 v567; // r13
  __int64 v568; // rax
  int v569; // ecx
  __int64 v570; // rdx
  __int64 v571; // rax
  int v572; // eax
  int v573; // r14d
  __int64 v574; // rax
  int v575; // r8d
  __int64 v576; // rdx
  __int64 v577; // r14
  __int64 v578; // rax
  __int64 v579; // rdx
  __int64 v580; // r13
  __int64 v581; // r15
  __int64 v582; // r15
  __int64 v583; // rax
  int v584; // eax
  int v585; // r14d
  __int64 v586; // rax
  int v587; // r8d
  __int64 v588; // rsi
  __int64 v589; // rdx
  __int64 v590; // r15
  __int64 v591; // r13
  __int64 v592; // rax
  int v593; // ecx
  __int64 v594; // rdx
  __int64 v595; // rax
  int v596; // eax
  int v597; // r14d
  __int64 v598; // rax
  int v599; // r8d
  __int64 v600; // rsi
  __int64 v601; // rdx
  __int64 v602; // r15
  __int64 v603; // r13
  __int64 v604; // rax
  int v605; // ecx
  __int64 v606; // rdx
  __int64 v607; // rax
  int v608; // eax
  int v609; // r14d
  __int64 v610; // rax
  int v611; // r8d
  __int64 v612; // rsi
  __int64 v613; // rdx
  __int64 v614; // r15
  __int64 v615; // r13
  __int64 v616; // rax
  int v617; // ecx
  __int64 v618; // rdx
  __int64 v619; // rax
  int v620; // eax
  int v621; // r14d
  __int64 v622; // rax
  int v623; // r8d
  __int64 v624; // rdx
  __int64 v625; // r14
  __int64 v626; // rax
  __int64 v627; // rdx
  __int64 v628; // r13
  __int64 v629; // r15
  __int64 v630; // r15
  __int64 v631; // rax
  int v632; // eax
  int v633; // r14d
  __int64 v634; // rax
  int v635; // r8d
  __int64 v636; // rsi
  __int64 v637; // rdx
  __int64 v638; // r15
  __int64 v639; // r13
  __int64 v640; // rax
  int v641; // ecx
  __int64 v642; // rdx
  __int64 v643; // rax
  int v644; // eax
  int v645; // r14d
  __int64 v646; // rax
  int v647; // r8d
  __int64 v648; // rsi
  __int64 v649; // rdx
  __int64 v650; // r15
  __int64 v651; // r13
  __int64 v652; // rax
  int v653; // ecx
  __int64 v654; // rdx
  __int64 v655; // rax
  int v656; // eax
  int v657; // r14d
  __int64 v658; // rax
  int v659; // r8d
  __int64 v660; // rsi
  __int64 v661; // rdx
  __int64 v662; // r15
  __int64 v663; // r13
  __int64 v664; // rax
  int v665; // ecx
  __int64 v666; // rdx
  __int64 v667; // rax
  int v668; // eax
  int v669; // r14d
  __int64 v670; // rax
  int v671; // r8d
  __int64 v672; // rsi
  __int64 v673; // rdx
  __int64 v674; // r15
  __int64 v675; // r13
  __int64 v676; // rax
  int v677; // ecx
  __int64 v678; // rdx
  __int64 v679; // rax
  int v680; // eax
  int v681; // r14d
  __int64 v682; // rax
  int v683; // r8d
  __int64 v684; // rsi
  __int64 v685; // rdx
  __int64 v686; // r15
  __int64 v687; // r13
  __int64 v688; // rax
  int v689; // ecx
  __int64 v690; // rdx
  __int64 v691; // rax
  int v692; // eax
  int v693; // r14d
  __int64 v694; // rax
  int v695; // r8d
  __int64 v696; // rsi
  __int64 v697; // rdx
  __int64 v698; // r15
  __int64 v699; // r13
  __int64 v700; // rax
  int v701; // ecx
  __int64 v702; // rdx
  __int64 v703; // rax
  int v704; // eax
  int v705; // r14d
  __int64 v706; // rax
  int v707; // r8d
  __int64 v708; // rsi
  __int64 v709; // rdx
  __int64 v710; // r15
  __int64 v711; // r13
  __int64 v712; // rax
  int v713; // ecx
  __int64 v714; // rdx
  __int64 v715; // rax
  int v716; // eax
  int v717; // r14d
  __int64 v718; // rax
  int v719; // r8d
  __int64 v720; // rdx
  __int64 v721; // r14
  __int64 v722; // rax
  __int64 v723; // rdx
  __int64 v724; // r13
  __int64 v725; // r15
  __int64 v726; // r15
  __int64 v727; // rax
  int v728; // eax
  int v729; // r14d
  __int64 v730; // rax
  int v731; // r8d
  __int64 v732; // rsi
  __int64 v733; // rdx
  __int64 v734; // r15
  __int64 v735; // r13
  __int64 v736; // rax
  int v737; // ecx
  __int64 v738; // rdx
  __int64 v739; // rax
  int v740; // eax
  int v741; // r14d
  __int64 v742; // rax
  int v743; // r8d
  __int64 v744; // rsi
  __int64 v745; // rdx
  __int64 v746; // r15
  __int64 v747; // r13
  __int64 v748; // rax
  int v749; // ecx
  __int64 v750; // rdx
  __int64 v751; // rax
  int v752; // eax
  int v753; // r14d
  __int64 v754; // rax
  int v755; // r8d
  __int64 v756; // rsi
  __int64 v757; // rdx
  __int64 v758; // r15
  __int64 v759; // r13
  __int64 v760; // rax
  int v761; // ecx
  __int64 v762; // rdx
  __int64 v763; // rax
  int v764; // eax
  int v765; // r14d
  __int64 v766; // rax
  int v767; // r8d
  __int64 v768; // rdx
  __int64 v769; // r14
  __int64 v770; // rax
  __int64 v771; // rdx
  __int64 v772; // r13
  __int64 v773; // r15
  __int64 v774; // r15
  __int64 v775; // rax
  int v776; // eax
  int v777; // r14d
  __int64 v778; // rax
  int v779; // r8d
  __int64 v780; // rsi
  __int64 v781; // rdx
  __int64 v782; // r15
  __int64 v783; // r13
  __int64 v784; // rax
  int v785; // ecx
  __int64 v786; // rdx
  __int64 v787; // rax
  int v788; // eax
  int v789; // r14d
  __int64 v790; // rax
  int v791; // r8d
  __int64 v792; // rsi
  __int64 v793; // rdx
  __int64 v794; // r15
  __int64 v795; // r13
  __int64 v796; // rax
  int v797; // ecx
  __int64 v798; // rdx
  __int64 v799; // rax
  int v800; // eax
  int v801; // r14d
  __int64 v802; // rax
  int v803; // r8d
  __int64 v804; // rsi
  __int64 v805; // rdx
  __int64 v806; // r15
  __int64 v807; // r13
  __int64 v808; // rax
  int v809; // ecx
  __int64 v810; // rdx
  __int64 v811; // rax
  int v812; // eax
  int v813; // r14d
  __int64 v814; // rax
  int v815; // r8d
  __int64 v816; // rsi
  __int64 v817; // rdx
  __int64 v818; // r15
  __int64 v819; // r13
  __int64 v820; // rax
  int v821; // ecx
  __int64 v822; // rdx
  __int64 v823; // rax
  int v824; // eax
  int v825; // r14d
  __int64 v826; // rax
  int v827; // r8d
  __int64 v828; // rsi
  __int64 v829; // rdx
  __int64 v830; // r15
  __int64 v831; // r13
  __int64 v832; // rax
  int v833; // ecx
  __int64 v834; // rdx
  __int64 v835; // rax
  int v836; // eax
  int v837; // r14d
  __int64 v838; // rax
  int v839; // r8d
  __int64 v840; // rsi
  __int64 v841; // rdx
  __int64 v842; // r15
  __int64 v843; // r13
  __int64 v844; // rax
  int v845; // ecx
  __int64 v846; // rdx
  __int64 v847; // rax
  int v848; // eax
  int v849; // r14d
  __int64 v850; // rax
  int v851; // r8d
  __int64 v852; // rsi
  __int64 v853; // rdx
  __int64 v854; // r15
  __int64 v855; // r13
  __int64 v856; // rax
  int v857; // ecx
  __int64 v858; // rdx
  __int64 v859; // rax
  int v860; // eax
  int v861; // r14d
  __int64 v862; // rax
  int v863; // r8d
  __int64 v864; // rsi
  __int64 v865; // rdx
  __int64 v866; // r15
  __int64 v867; // r13
  __int64 v868; // rax
  int v869; // ecx
  __int64 v870; // rdx
  __int64 v871; // rax
  int v872; // eax
  int v873; // r14d
  __int64 v874; // rax
  int v875; // r8d
  __int64 v876; // rsi
  __int64 v877; // rdx
  __int64 v878; // r15
  __int64 v879; // r13
  __int64 v880; // rax
  int v881; // ecx
  __int64 v882; // rdx
  __int64 v883; // rax
  int v884; // eax
  int v885; // r14d
  __int64 v886; // rax
  int v887; // r8d
  __int64 v888; // rsi
  __int64 v889; // rdx
  __int64 v890; // r15
  __int64 v891; // r13
  __int64 v892; // rax
  int v893; // ecx
  __int64 v894; // rdx
  __int64 v895; // rax
  int v896; // eax
  int v897; // r14d
  __int64 v898; // rax
  int v899; // r8d
  __int64 v900; // rsi
  __int64 v901; // rdx
  __int64 v902; // r15
  __int64 v903; // r13
  __int64 v904; // rax
  int v905; // ecx
  __int64 v906; // rdx
  __int64 v907; // rax
  int v908; // eax
  int v909; // r14d
  __int64 v910; // rax
  int v911; // r8d
  __int64 v912; // rdx
  __int64 v913; // r14
  __int64 v914; // rax
  __int64 v915; // rdx
  __int64 v916; // r13
  __int64 v917; // r15
  __int64 v918; // r15
  __int64 v919; // rax
  int v920; // eax
  int v921; // r14d
  __int64 v922; // rax
  int v923; // r8d
  __int64 v924; // rsi
  __int64 v925; // rdx
  __int64 v926; // r15
  __int64 v927; // r13
  __int64 v928; // rax
  int v929; // ecx
  __int64 v930; // rdx
  __int64 v931; // rax
  int v932; // eax
  int v933; // r14d
  __int64 v934; // rax
  int v935; // r8d
  __int64 v936; // rsi
  __int64 v937; // rdx
  __int64 v938; // r15
  __int64 v939; // r13
  __int64 v940; // rax
  int v941; // ecx
  __int64 v942; // rdx
  __int64 v943; // rax
  int v944; // eax
  int v945; // r14d
  __int64 v946; // rax
  int v947; // r8d
  __int64 v948; // rsi
  __int64 v949; // rdx
  __int64 v950; // r15
  __int64 v951; // r13
  __int64 v952; // rax
  int v953; // ecx
  __int64 v954; // rdx
  __int64 v955; // rax
  int v956; // eax
  int v957; // r14d
  __int64 v958; // rax
  int v959; // r8d
  __int64 v960; // rdx
  __int64 v961; // r14
  __int64 v962; // rax
  __int64 v963; // rdx
  __int64 v964; // r13
  __int64 v965; // r15
  __int64 v966; // r15
  __int64 v967; // rax
  int v968; // eax
  int v969; // r14d
  __int64 v970; // rax
  int v971; // r8d
  __int64 v972; // rsi
  __int64 v973; // rdx
  __int64 v974; // r15
  __int64 v975; // rax
  int v976; // eax
  int v977; // r14d
  __int64 v978; // rax
  int v979; // r8d
  __int64 v980; // rsi
  __int64 v981; // rdx
  __int64 v982; // r15
  __int64 v983; // rax
  int v984; // eax
  int v985; // r14d
  __int64 v986; // rax
  int v987; // r8d
  __int64 v988; // rsi
  __int64 v989; // rdx
  __int64 v990; // r15
  __int64 v991; // r13
  __int64 v992; // rax
  int v993; // ecx
  __int64 v994; // rdx
  __int64 v995; // rax
  int v996; // eax
  int v997; // r14d
  __int64 v998; // rax
  int v999; // r8d
  __int64 v1000; // rsi
  __int64 v1001; // rdx
  __int64 v1002; // r15
  __int64 v1003; // r13
  __int64 v1004; // rax
  int v1005; // ecx
  __int64 v1006; // rdx
  __int64 v1007; // rax
  int v1008; // eax
  int v1009; // r14d
  __int64 v1010; // rax
  int v1011; // r8d
  __int64 v1012; // rsi
  __int64 v1013; // rdx
  __int64 v1014; // r15
  __int64 v1015; // r13
  __int64 v1016; // rax
  int v1017; // ecx
  __int64 v1018; // rdx
  __int64 v1019; // rax
  int v1020; // eax
  int v1021; // r14d
  __int64 v1022; // rax
  int v1023; // r8d
  __int64 v1024; // rdx
  __int64 v1025; // r14
  __int64 v1026; // rax
  __int64 v1027; // rdx
  __int64 v1028; // r13
  __int64 v1029; // r15
  __int64 v1030; // r15
  __int64 v1031; // rax
  int v1032; // eax
  int v1033; // r14d
  __int64 v1034; // rax
  int v1035; // r8d
  __int64 v1036; // rsi
  __int64 v1037; // rdx
  __int64 v1038; // r15
  __int64 v1039; // r13
  __int64 v1040; // rax
  int v1041; // ecx
  __int64 v1042; // rdx
  __int64 v1043; // rax
  int v1044; // eax
  int v1045; // r14d
  __int64 v1046; // rax
  int v1047; // r8d
  __int64 v1048; // rsi
  __int64 v1049; // rdx
  __int64 v1050; // r15
  __int64 v1051; // r13
  __int64 v1052; // rax
  int v1053; // ecx
  __int64 v1054; // rdx
  __int64 v1055; // rax
  int v1056; // eax
  int v1057; // r14d
  __int64 v1058; // rax
  int v1059; // r8d
  __int64 v1060; // rsi
  __int64 v1061; // rdx
  __int64 v1062; // r15
  __int64 v1063; // r13
  __int64 v1064; // rax
  int v1065; // ecx
  __int64 v1066; // rdx
  __int64 v1067; // rax
  int v1068; // eax
  int v1069; // r14d
  __int64 v1070; // rax
  int v1071; // r8d
  __int64 v1072; // rdx
  __int64 v1073; // r14
  __int64 v1074; // rax
  __int64 v1075; // rdx
  __int64 v1076; // r13
  __int64 v1077; // r15
  __int64 v1078; // r15
  __int64 v1079; // rax
  int v1080; // eax
  int v1081; // r14d
  __int64 v1082; // rax
  int v1083; // r8d
  __int64 v1084; // rsi
  __int64 v1085; // rdx
  __int64 v1086; // r15
  __int64 v1087; // r13
  __int64 v1088; // rax
  int v1089; // ecx
  __int64 v1090; // rdx
  __int64 v1091; // rax
  int v1092; // eax
  int v1093; // r14d
  __int64 v1094; // rax
  int v1095; // r8d
  __int64 v1096; // rdx
  __int64 v1097; // rax
  __int64 v1098; // r13
  size_t v1099; // rcx
  const char *v1100; // r15
  int v1101; // r14d
  __int64 v1102; // rax
  __int64 v1103; // rdx
  __int64 v1104; // r13
  __int64 v1105; // r15
  int v1106; // r14d
  __int64 v1107; // rax
  __int64 v1108; // rax
  int v1109; // r8d
  __int64 v1110; // rsi
  __int64 v1111; // rdx
  __int64 v1112; // r15
  int v1113; // r14d
  __int64 v1114; // r13
  __int64 v1115; // rax
  int v1116; // ecx
  __int64 v1117; // rdx
  __int64 v1118; // rax
  __int64 v1119; // rax
  int v1120; // r8d
  __int64 v1121; // rsi
  __int64 v1122; // rdx
  __int64 v1123; // r15
  int v1124; // r14d
  __int64 v1125; // r13
  __int64 v1126; // rax
  int v1127; // ecx
  __int64 v1128; // rdx
  __int64 v1129; // rax
  __int64 v1130; // rax
  int v1131; // r8d
  __int64 v1132; // rsi
  __int64 v1133; // rdx
  __int64 v1134; // r15
  int v1135; // r14d
  __int64 v1136; // r13
  __int64 v1137; // rax
  int v1138; // ecx
  __int64 v1139; // rdx
  __int64 v1140; // rax
  __int64 v1141; // rax
  int v1142; // r8d
  __int64 v1143; // rsi
  __int64 v1144; // rdx
  __int64 v1145; // r15
  int v1146; // r14d
  __int64 v1147; // r13
  __int64 v1148; // rax
  int v1149; // ecx
  __int64 v1150; // rdx
  __int64 v1151; // rax
  __int64 v1152; // rax
  int v1153; // r8d
  __int64 v1154; // rsi
  __int64 v1155; // rdx
  __int64 v1156; // r15
  int v1157; // r14d
  __int64 v1158; // rax
  __int64 v1159; // rax
  int v1160; // r8d
  __int64 v1161; // rsi
  __int64 v1162; // rdx
  __int64 v1163; // r15
  int v1164; // r14d
  __int64 v1165; // r13
  __int64 v1166; // rax
  int v1167; // ecx
  __int64 v1168; // rdx
  __int64 v1169; // r13
  __int64 v1170; // rax
  int v1171; // ecx
  __int64 v1172; // rdx
  __int64 v1173; // rax
  __int64 v1174; // rax
  int v1175; // r8d
  __int64 v1176; // rsi
  __int64 v1177; // rdx
  __int64 v1178; // r15
  int v1179; // r14d
  __int64 v1180; // r13
  __int64 v1181; // rax
  int v1182; // ecx
  __int64 v1183; // rdx
  __int64 v1184; // rax
  __int64 v1185; // rax
  int v1186; // r8d
  __int64 v1187; // rdx
  __int64 v1188; // rax
  __int64 v1189; // rsi
  const char *v1190; // rdi
  __int64 v1191; // r13
  int v1192; // r14d
  int v1193; // r15d
  __int64 v1194; // rax
  __int64 v1195; // rdx
  __int64 v1196; // r13
  __int64 v1197; // r15
  int v1198; // r14d
  __int64 v1199; // rax
  __int64 v1200; // rax
  int v1201; // r8d
  __int64 v1202; // rsi
  __int64 v1203; // rdx
  __int64 v1204; // r15
  int v1205; // r14d
  __int64 v1206; // r13
  __int64 v1207; // rax
  int v1208; // ecx
  __int64 v1209; // rdx
  __int64 v1210; // rax
  __int64 v1211; // rax
  int v1212; // r8d
  __int64 v1213; // rsi
  __int64 v1214; // rdx
  __int64 v1215; // r15
  int v1216; // r14d
  __int64 v1217; // r13
  __int64 v1218; // rax
  int v1219; // ecx
  __int64 v1220; // rdx
  __int64 v1221; // rax
  __int64 v1222; // rax
  int v1223; // r8d
  __int64 v1224; // rdx
  __int64 v1225; // rax
  __int64 v1226; // rsi
  const char *v1227; // rdi
  __int64 v1228; // r13
  int v1229; // r14d
  int v1230; // r15d
  __int64 v1231; // rax
  __int64 v1232; // rdx
  __int64 v1233; // r13
  __int64 v1234; // r15
  int v1235; // r14d
  __int64 v1236; // rax
  __int64 v1237; // rax
  int v1238; // r8d
  __int64 v1239; // rdx
  __int64 v1240; // rax
  __int64 v1241; // rsi
  const char *v1242; // rdi
  __int64 v1243; // r13
  int v1244; // r14d
  int v1245; // r15d
  __int64 v1246; // rax
  __int64 v1247; // rdx
  __int64 v1248; // r13
  __int64 v1249; // r15
  int v1250; // r14d
  __int64 v1251; // rax
  __int64 v1252; // rax
  int v1253; // r8d
  __int64 v1254; // rdx
  __int64 v1255; // rax
  __int64 v1256; // rsi
  const char *v1257; // rdi
  __int64 v1258; // r13
  int v1259; // r14d
  int v1260; // r15d
  __int64 v1261; // rax
  __int64 v1262; // rdx
  __int64 v1263; // r13
  __int64 v1264; // r15
  int v1265; // r14d
  __int64 v1266; // rax
  __int64 v1267; // rax
  int v1268; // r8d
  __int64 v1269; // rsi
  __int64 v1270; // rdx
  __int64 v1271; // r15
  int v1272; // r14d
  __int64 v1273; // r13
  __int64 v1274; // rax
  int v1275; // ecx
  __int64 v1276; // rdx
  __int64 v1277; // rax
  __int64 v1278; // rax
  int v1279; // r8d
  __int64 v1280; // rdx
  __int64 v1281; // r14
  __int64 v1282; // rax
  __int64 v1283; // rdx
  __int64 v1284; // r13
  __int64 v1285; // r15
  __int64 v1286; // r15
  int v1287; // r14d
  __int64 v1288; // rax
  __int64 v1289; // rax
  int v1290; // r8d
  __int64 v1291; // rsi
  __int64 v1292; // rdx
  __int64 v1293; // r15
  int v1294; // r14d
  __int64 v1295; // r13
  __int64 v1296; // rax
  int v1297; // ecx
  __int64 v1298; // rdx
  __int64 v1299; // rax
  __int64 v1300; // rax
  int v1301; // r8d
  __int64 v1302; // rdx
  __int64 v1303; // rax
  __int64 v1304; // rsi
  const char *v1305; // rdi
  __int64 v1306; // r13
  int v1307; // r14d
  int v1308; // r15d
  __int64 v1309; // rax
  __int64 v1310; // rdx
  __int64 v1311; // r13
  __int64 v1312; // r15
  int v1313; // r14d
  __int64 v1314; // rax
  __int64 v1315; // rax
  int v1316; // r8d
  __int64 v1317; // rsi
  __int64 v1318; // rdx
  __int64 v1319; // r15
  int v1320; // r14d
  __int64 v1321; // r13
  __int64 v1322; // rax
  int v1323; // ecx
  __int64 v1324; // rdx
  __int64 v1325; // rax
  __int64 v1326; // rax
  int v1327; // r8d
  __int64 v1328; // rsi
  __int64 v1329; // rdx
  __int64 v1330; // r15
  int v1331; // r14d
  __int64 v1332; // r13
  __int64 v1333; // rax
  int v1334; // ecx
  __int64 v1335; // rdx
  __int64 v1336; // rax
  __int64 v1337; // rax
  int v1338; // r8d
  __int64 v1339; // rdx
  __int64 v1340; // r13
  __int64 v1341; // rax
  int v1342; // ecx
  __int64 v1343; // rdx
  const char **v1345; // rax
  size_t v1346; // rax
  const char **v1347; // rax
  size_t v1348; // rax
  const char **v1349; // rax
  size_t v1350; // rax
  const char **v1351; // rax
  size_t v1352; // rax
  const char **v1353; // rax
  size_t v1354; // rax
  const char **v1355; // rax
  const char **v1356; // rax
  size_t v1357; // rax
  size_t v1358; // [rsp+8h] [rbp-C88h]
  const char *v1359; // [rsp+8h] [rbp-C88h]
  const char *v1360; // [rsp+8h] [rbp-C88h]
  const char *v1361; // [rsp+8h] [rbp-C88h]
  const char *v1362; // [rsp+8h] [rbp-C88h]
  const char *v1363; // [rsp+8h] [rbp-C88h]
  const char *v1364; // [rsp+8h] [rbp-C88h]
  __int64 v1365; // [rsp+8h] [rbp-C88h]
  __int64 v1366; // [rsp+8h] [rbp-C88h]
  __int64 v1367; // [rsp+8h] [rbp-C88h]
  __int64 v1368; // [rsp+8h] [rbp-C88h]
  __int64 v1369; // [rsp+8h] [rbp-C88h]
  __int64 v1370; // [rsp+8h] [rbp-C88h]
  __int64 v1371; // [rsp+8h] [rbp-C88h]
  __int64 v1372[2]; // [rsp+10h] [rbp-C80h] BYREF
  __int64 v1373[2]; // [rsp+20h] [rbp-C70h] BYREF
  __int64 v1374[2]; // [rsp+30h] [rbp-C60h] BYREF
  __int64 v1375[2]; // [rsp+40h] [rbp-C50h] BYREF
  __int64 v1376[2]; // [rsp+50h] [rbp-C40h] BYREF
  __int64 v1377[2]; // [rsp+60h] [rbp-C30h] BYREF
  __int64 v1378[2]; // [rsp+70h] [rbp-C20h] BYREF
  __int64 v1379[2]; // [rsp+80h] [rbp-C10h] BYREF
  __int64 v1380[2]; // [rsp+90h] [rbp-C00h] BYREF
  __int64 v1381[2]; // [rsp+A0h] [rbp-BF0h] BYREF
  __int64 v1382[2]; // [rsp+B0h] [rbp-BE0h] BYREF
  __int64 v1383[2]; // [rsp+C0h] [rbp-BD0h] BYREF
  __int64 v1384[2]; // [rsp+D0h] [rbp-BC0h] BYREF
  __int64 v1385[2]; // [rsp+E0h] [rbp-BB0h] BYREF
  __int64 v1386[2]; // [rsp+F0h] [rbp-BA0h] BYREF
  __int64 v1387[2]; // [rsp+100h] [rbp-B90h] BYREF
  __int64 v1388[2]; // [rsp+110h] [rbp-B80h] BYREF
  __int64 v1389[2]; // [rsp+120h] [rbp-B70h] BYREF
  __int64 v1390[2]; // [rsp+130h] [rbp-B60h] BYREF
  __int64 v1391[2]; // [rsp+140h] [rbp-B50h] BYREF
  __int64 v1392[2]; // [rsp+150h] [rbp-B40h] BYREF
  __int64 v1393[2]; // [rsp+160h] [rbp-B30h] BYREF
  __int64 v1394[2]; // [rsp+170h] [rbp-B20h] BYREF
  __int64 v1395[2]; // [rsp+180h] [rbp-B10h] BYREF
  __int64 v1396[2]; // [rsp+190h] [rbp-B00h] BYREF
  __int64 v1397[2]; // [rsp+1A0h] [rbp-AF0h] BYREF
  __int64 v1398[2]; // [rsp+1B0h] [rbp-AE0h] BYREF
  __int64 v1399[2]; // [rsp+1C0h] [rbp-AD0h] BYREF
  __int64 v1400[2]; // [rsp+1D0h] [rbp-AC0h] BYREF
  __int64 v1401[2]; // [rsp+1E0h] [rbp-AB0h] BYREF
  __int64 v1402[2]; // [rsp+1F0h] [rbp-AA0h] BYREF
  __int64 v1403[2]; // [rsp+200h] [rbp-A90h] BYREF
  __int64 v1404[2]; // [rsp+210h] [rbp-A80h] BYREF
  __int64 v1405[2]; // [rsp+220h] [rbp-A70h] BYREF
  __int64 v1406[2]; // [rsp+230h] [rbp-A60h] BYREF
  __int64 v1407[2]; // [rsp+240h] [rbp-A50h] BYREF
  __int64 v1408[2]; // [rsp+250h] [rbp-A40h] BYREF
  __int64 v1409[2]; // [rsp+260h] [rbp-A30h] BYREF
  __int64 v1410[2]; // [rsp+270h] [rbp-A20h] BYREF
  __int64 v1411[2]; // [rsp+280h] [rbp-A10h] BYREF
  __int64 v1412[2]; // [rsp+290h] [rbp-A00h] BYREF
  __int64 v1413[2]; // [rsp+2A0h] [rbp-9F0h] BYREF
  __int64 v1414[2]; // [rsp+2B0h] [rbp-9E0h] BYREF
  __int64 v1415[2]; // [rsp+2C0h] [rbp-9D0h] BYREF
  __int64 v1416[2]; // [rsp+2D0h] [rbp-9C0h] BYREF
  __int64 v1417[2]; // [rsp+2E0h] [rbp-9B0h] BYREF
  __int64 v1418[2]; // [rsp+2F0h] [rbp-9A0h] BYREF
  __int64 v1419[2]; // [rsp+300h] [rbp-990h] BYREF
  __int64 v1420[2]; // [rsp+310h] [rbp-980h] BYREF
  __int64 v1421[2]; // [rsp+320h] [rbp-970h] BYREF
  __int64 v1422[2]; // [rsp+330h] [rbp-960h] BYREF
  __int64 v1423[2]; // [rsp+340h] [rbp-950h] BYREF
  __int64 v1424[2]; // [rsp+350h] [rbp-940h] BYREF
  __int64 v1425[2]; // [rsp+360h] [rbp-930h] BYREF
  __int64 v1426[2]; // [rsp+370h] [rbp-920h] BYREF
  __int64 v1427[2]; // [rsp+380h] [rbp-910h] BYREF
  __int64 v1428[2]; // [rsp+390h] [rbp-900h] BYREF
  __int64 v1429[2]; // [rsp+3A0h] [rbp-8F0h] BYREF
  __int64 v1430[2]; // [rsp+3B0h] [rbp-8E0h] BYREF
  __int64 v1431[2]; // [rsp+3C0h] [rbp-8D0h] BYREF
  __int64 v1432[2]; // [rsp+3D0h] [rbp-8C0h] BYREF
  __int64 v1433[2]; // [rsp+3E0h] [rbp-8B0h] BYREF
  __int64 v1434[2]; // [rsp+3F0h] [rbp-8A0h] BYREF
  __int64 v1435[2]; // [rsp+400h] [rbp-890h] BYREF
  __int64 v1436[2]; // [rsp+410h] [rbp-880h] BYREF
  __int64 v1437[2]; // [rsp+420h] [rbp-870h] BYREF
  __int64 v1438[2]; // [rsp+430h] [rbp-860h] BYREF
  __int64 v1439[2]; // [rsp+440h] [rbp-850h] BYREF
  __int64 v1440[2]; // [rsp+450h] [rbp-840h] BYREF
  __int64 v1441[2]; // [rsp+460h] [rbp-830h] BYREF
  __int64 v1442[2]; // [rsp+470h] [rbp-820h] BYREF
  __int64 v1443[2]; // [rsp+480h] [rbp-810h] BYREF
  __int64 v1444[2]; // [rsp+490h] [rbp-800h] BYREF
  __int64 v1445[2]; // [rsp+4A0h] [rbp-7F0h] BYREF
  __int64 v1446[2]; // [rsp+4B0h] [rbp-7E0h] BYREF
  __int64 v1447[2]; // [rsp+4C0h] [rbp-7D0h] BYREF
  __int64 v1448[2]; // [rsp+4D0h] [rbp-7C0h] BYREF
  __int64 v1449[2]; // [rsp+4E0h] [rbp-7B0h] BYREF
  __int64 v1450[2]; // [rsp+4F0h] [rbp-7A0h] BYREF
  __int64 v1451[2]; // [rsp+500h] [rbp-790h] BYREF
  __int64 v1452[2]; // [rsp+510h] [rbp-780h] BYREF
  __int64 v1453[2]; // [rsp+520h] [rbp-770h] BYREF
  __int64 v1454[2]; // [rsp+530h] [rbp-760h] BYREF
  __int64 v1455[2]; // [rsp+540h] [rbp-750h] BYREF
  __int64 v1456[2]; // [rsp+550h] [rbp-740h] BYREF
  __int64 v1457[2]; // [rsp+560h] [rbp-730h] BYREF
  __int64 v1458[2]; // [rsp+570h] [rbp-720h] BYREF
  __int64 v1459[2]; // [rsp+580h] [rbp-710h] BYREF
  __int64 v1460[2]; // [rsp+590h] [rbp-700h] BYREF
  __int64 v1461[2]; // [rsp+5A0h] [rbp-6F0h] BYREF
  __int64 v1462[2]; // [rsp+5B0h] [rbp-6E0h] BYREF
  __int64 v1463[2]; // [rsp+5C0h] [rbp-6D0h] BYREF
  __int64 v1464[2]; // [rsp+5D0h] [rbp-6C0h] BYREF
  __int64 v1465[2]; // [rsp+5E0h] [rbp-6B0h] BYREF
  __int64 v1466[2]; // [rsp+5F0h] [rbp-6A0h] BYREF
  __int64 v1467[2]; // [rsp+600h] [rbp-690h] BYREF
  __int64 v1468[2]; // [rsp+610h] [rbp-680h] BYREF
  __int64 v1469[2]; // [rsp+620h] [rbp-670h] BYREF
  __int64 v1470[2]; // [rsp+630h] [rbp-660h] BYREF
  __int64 v1471[2]; // [rsp+640h] [rbp-650h] BYREF
  __int64 v1472[2]; // [rsp+650h] [rbp-640h] BYREF
  __int64 v1473[2]; // [rsp+660h] [rbp-630h] BYREF
  __int64 v1474[2]; // [rsp+670h] [rbp-620h] BYREF
  __int64 v1475[2]; // [rsp+680h] [rbp-610h] BYREF
  __int64 v1476[2]; // [rsp+690h] [rbp-600h] BYREF
  __int64 v1477[2]; // [rsp+6A0h] [rbp-5F0h] BYREF
  __int64 v1478[2]; // [rsp+6B0h] [rbp-5E0h] BYREF
  __int64 v1479[2]; // [rsp+6C0h] [rbp-5D0h] BYREF
  __int64 v1480[2]; // [rsp+6D0h] [rbp-5C0h] BYREF
  __int64 v1481[2]; // [rsp+6E0h] [rbp-5B0h] BYREF
  __int64 v1482[2]; // [rsp+6F0h] [rbp-5A0h] BYREF
  __int64 v1483[2]; // [rsp+700h] [rbp-590h] BYREF
  __int64 v1484[2]; // [rsp+710h] [rbp-580h] BYREF
  __int64 v1485[2]; // [rsp+720h] [rbp-570h] BYREF
  __int64 v1486[2]; // [rsp+730h] [rbp-560h] BYREF
  __int64 v1487[2]; // [rsp+740h] [rbp-550h] BYREF
  __int64 v1488[2]; // [rsp+750h] [rbp-540h] BYREF
  __int64 v1489[2]; // [rsp+760h] [rbp-530h] BYREF
  __int64 v1490[2]; // [rsp+770h] [rbp-520h] BYREF
  __int64 v1491[2]; // [rsp+780h] [rbp-510h] BYREF
  __int64 v1492[2]; // [rsp+790h] [rbp-500h] BYREF
  __int64 v1493[2]; // [rsp+7A0h] [rbp-4F0h] BYREF
  __int64 v1494[2]; // [rsp+7B0h] [rbp-4E0h] BYREF
  __int64 v1495[2]; // [rsp+7C0h] [rbp-4D0h] BYREF
  __int64 v1496[2]; // [rsp+7D0h] [rbp-4C0h] BYREF
  __int64 v1497[2]; // [rsp+7E0h] [rbp-4B0h] BYREF
  __int64 v1498[2]; // [rsp+7F0h] [rbp-4A0h] BYREF
  __int64 v1499[2]; // [rsp+800h] [rbp-490h] BYREF
  __int64 v1500[2]; // [rsp+810h] [rbp-480h] BYREF
  __int64 v1501[2]; // [rsp+820h] [rbp-470h] BYREF
  __int64 v1502[2]; // [rsp+830h] [rbp-460h] BYREF
  __int64 v1503[2]; // [rsp+840h] [rbp-450h] BYREF
  __int64 v1504[2]; // [rsp+850h] [rbp-440h] BYREF
  __int64 v1505[2]; // [rsp+860h] [rbp-430h] BYREF
  __int64 v1506[2]; // [rsp+870h] [rbp-420h] BYREF
  __int64 v1507[2]; // [rsp+880h] [rbp-410h] BYREF
  __int64 v1508[2]; // [rsp+890h] [rbp-400h] BYREF
  __int64 v1509[2]; // [rsp+8A0h] [rbp-3F0h] BYREF
  __int64 v1510[2]; // [rsp+8B0h] [rbp-3E0h] BYREF
  __int64 v1511[2]; // [rsp+8C0h] [rbp-3D0h] BYREF
  __int64 v1512[2]; // [rsp+8D0h] [rbp-3C0h] BYREF
  __int64 v1513[2]; // [rsp+8E0h] [rbp-3B0h] BYREF
  __int64 v1514[2]; // [rsp+8F0h] [rbp-3A0h] BYREF
  __int64 v1515[2]; // [rsp+900h] [rbp-390h] BYREF
  __int64 v1516[2]; // [rsp+910h] [rbp-380h] BYREF
  __int64 v1517[2]; // [rsp+920h] [rbp-370h] BYREF
  __int64 v1518[2]; // [rsp+930h] [rbp-360h] BYREF
  __int64 v1519[2]; // [rsp+940h] [rbp-350h] BYREF
  __int64 v1520[2]; // [rsp+950h] [rbp-340h] BYREF
  __int64 v1521[2]; // [rsp+960h] [rbp-330h] BYREF
  __int64 v1522[2]; // [rsp+970h] [rbp-320h] BYREF
  __int64 v1523[2]; // [rsp+980h] [rbp-310h] BYREF
  __int64 v1524[2]; // [rsp+990h] [rbp-300h] BYREF
  __int64 v1525[2]; // [rsp+9A0h] [rbp-2F0h] BYREF
  __int64 v1526[2]; // [rsp+9B0h] [rbp-2E0h] BYREF
  __int64 v1527[2]; // [rsp+9C0h] [rbp-2D0h] BYREF
  __int64 v1528[2]; // [rsp+9D0h] [rbp-2C0h] BYREF
  __int64 v1529[2]; // [rsp+9E0h] [rbp-2B0h] BYREF
  __int64 v1530[2]; // [rsp+9F0h] [rbp-2A0h] BYREF
  __int64 v1531[2]; // [rsp+A00h] [rbp-290h] BYREF
  __int64 v1532[2]; // [rsp+A10h] [rbp-280h] BYREF
  __int64 v1533[2]; // [rsp+A20h] [rbp-270h] BYREF
  __int64 v1534[2]; // [rsp+A30h] [rbp-260h] BYREF
  __int64 v1535[2]; // [rsp+A40h] [rbp-250h] BYREF
  __int64 v1536[2]; // [rsp+A50h] [rbp-240h] BYREF
  __int64 v1537[2]; // [rsp+A60h] [rbp-230h] BYREF
  __int64 v1538[2]; // [rsp+A70h] [rbp-220h] BYREF
  __int64 v1539[2]; // [rsp+A80h] [rbp-210h] BYREF
  __int64 v1540[2]; // [rsp+A90h] [rbp-200h] BYREF
  __int64 v1541[2]; // [rsp+AA0h] [rbp-1F0h] BYREF
  __int64 v1542[2]; // [rsp+AB0h] [rbp-1E0h] BYREF
  __int64 v1543[2]; // [rsp+AC0h] [rbp-1D0h] BYREF
  __int64 v1544[2]; // [rsp+AD0h] [rbp-1C0h] BYREF
  __int64 v1545[2]; // [rsp+AE0h] [rbp-1B0h] BYREF
  __int64 v1546[2]; // [rsp+AF0h] [rbp-1A0h] BYREF
  __int64 v1547[2]; // [rsp+B00h] [rbp-190h] BYREF
  __int64 v1548[2]; // [rsp+B10h] [rbp-180h] BYREF
  __int64 v1549[2]; // [rsp+B20h] [rbp-170h] BYREF
  __int64 v1550[2]; // [rsp+B30h] [rbp-160h] BYREF
  __int64 v1551[2]; // [rsp+B40h] [rbp-150h] BYREF
  __int64 v1552[2]; // [rsp+B50h] [rbp-140h] BYREF
  __int64 v1553[2]; // [rsp+B60h] [rbp-130h] BYREF
  __int64 v1554[2]; // [rsp+B70h] [rbp-120h] BYREF
  __int64 v1555[2]; // [rsp+B80h] [rbp-110h] BYREF
  __int64 v1556[2]; // [rsp+B90h] [rbp-100h] BYREF
  __int64 v1557[2]; // [rsp+BA0h] [rbp-F0h] BYREF
  __int64 v1558[2]; // [rsp+BB0h] [rbp-E0h] BYREF
  __int64 v1559[2]; // [rsp+BC0h] [rbp-D0h] BYREF
  __int64 v1560[2]; // [rsp+BD0h] [rbp-C0h] BYREF
  __int64 v1561[2]; // [rsp+BE0h] [rbp-B0h] BYREF
  __int64 v1562[2]; // [rsp+BF0h] [rbp-A0h] BYREF
  __int64 v1563[2]; // [rsp+C00h] [rbp-90h] BYREF
  __int64 v1564[2]; // [rsp+C10h] [rbp-80h] BYREF
  __int64 v1565[2]; // [rsp+C20h] [rbp-70h] BYREF
  __int64 v1566[2]; // [rsp+C30h] [rbp-60h] BYREF
  __int64 v1567[2]; // [rsp+C40h] [rbp-50h] BYREF
  __int64 v1568[8]; // [rsp+C50h] [rbp-40h] BYREF

  v2 = 0;
  v3 = 0;
  v5 = *(_DWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)a1 = v5;
  v6 = sub_12D6170(a2 + 120, 1u);
  if ( v6 )
  {
    if ( *(_DWORD *)(v6 + 56) )
      v2 = **(_QWORD **)(v6 + 48);
    v3 = *(_DWORD *)(v6 + 40);
  }
  v7 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 1);
  v8 = *(_DWORD *)a1;
  v9 = v2;
  v1372[1] = v10;
  v11 = 0;
  v1372[0] = v7;
  v12 = 0;
  sub_12D6090(a1 + 16, v9, v3, v1372, v8);
  v13 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 2u);
  if ( v13 )
  {
    if ( *(_DWORD *)(v13 + 56) )
      v11 = **(_QWORD **)(v13 + 48);
    v12 = *(_DWORD *)(v13 + 40);
  }
  v14 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 2);
  v15 = *(_DWORD *)a1;
  v16 = v11;
  v1373[1] = v17;
  LODWORD(v17) = v12;
  v18 = 0;
  v1373[0] = v14;
  v19 = 0;
  sub_12D6090(a1 + 40, v16, v17, v1373, v15);
  v20 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 3u);
  if ( v20 )
  {
    if ( *(_DWORD *)(v20 + 56) )
      v18 = **(_QWORD **)(v20 + 48);
    v19 = *(_DWORD *)(v20 + 40);
  }
  v21 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 3);
  v22 = *(_DWORD *)a1;
  v23 = v18;
  v1374[1] = v24;
  LODWORD(v24) = v19;
  v25 = 0;
  v1374[0] = v21;
  v26 = 0;
  sub_12D6090(a1 + 64, v23, v24, v1374, v22);
  v27 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 4u);
  if ( v27 )
  {
    if ( *(_DWORD *)(v27 + 56) )
      v25 = **(_QWORD **)(v27 + 48);
    v26 = *(_DWORD *)(v27 + 40);
  }
  v28 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 4);
  v29 = *(_DWORD *)a1;
  v30 = v25;
  v1375[1] = v31;
  LODWORD(v31) = v26;
  v32 = 0;
  v1375[0] = v28;
  v33 = 0;
  sub_12D6090(a1 + 88, v30, v31, v1375, v29);
  v34 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 5u);
  if ( v34 )
  {
    if ( *(_DWORD *)(v34 + 56) )
      v32 = **(_QWORD **)(v34 + 48);
    v33 = *(_DWORD *)(v34 + 40);
  }
  v35 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 5);
  v36 = *(_DWORD *)a1;
  v37 = v32;
  v1376[1] = v38;
  LODWORD(v38) = v33;
  v39 = 0;
  v1376[0] = v35;
  v40 = 0;
  sub_12D6090(a1 + 112, v37, v38, v1376, v36);
  v41 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 6u);
  if ( v41 )
  {
    if ( *(_DWORD *)(v41 + 56) )
      v39 = **(_QWORD **)(v41 + 48);
    v40 = *(_DWORD *)(v41 + 40);
  }
  v42 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 6);
  v43 = *(_DWORD *)a1;
  v1377[1] = v44;
  v1377[0] = v42;
  sub_12D6090(a1 + 136, v39, v40, v1377, v43);
  v45 = sub_12D6240(*(_QWORD *)(a1 + 8), 7u, "0");
  v46 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 7);
  *(_BYTE *)(a1 + 160) = v45;
  v48 = v46;
  v49 = v47;
  LODWORD(v46) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v48 + 36) == 0;
  *(_DWORD *)(a1 + 164) = HIDWORD(v45);
  *(_DWORD *)(a1 + 168) = v46;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 172) = *(_DWORD *)(v48 + 32);
  }
  else if ( sub_1691920(v47, *(unsigned __int16 *)(v48 + 40)) )
  {
    *(_DWORD *)(a1 + 172) = *(_DWORD *)(sub_1691920(v49, *(unsigned __int16 *)(v48 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 172) = 0;
  }
  v51 = 0;
  v52 = 0;
  v53 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 8u);
  if ( v53 )
  {
    if ( *(_DWORD *)(v53 + 56) )
      v51 = **(_QWORD **)(v53 + 48);
    v52 = *(_DWORD *)(v53 + 40);
  }
  v54 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 8);
  v55 = *(_DWORD *)a1;
  v1378[1] = v56;
  v1378[0] = v54;
  sub_12D6090(a1 + 176, v51, v52, v1378, v55);
  v57 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 9u);
  v58 = 1;
  v59 = "1";
  v60 = v57;
  if ( v57 )
  {
    if ( *(_DWORD *)(v57 + 56) )
    {
      v1356 = *(const char ***)(v57 + 48);
      v58 = 0;
      v59 = *v1356;
      if ( *v1356 )
      {
        v1364 = *v1356;
        v1357 = strlen(v59);
        v59 = v1364;
        v58 = v1357;
      }
    }
  }
  if ( (unsigned __int8)sub_16D2BB0(v59, v58, 0, v1568) || (v61 = v1568[0], v1568[0] != SLODWORD(v1568[0])) )
    v61 = 0;
  v62 = 0;
  if ( v60 )
    v62 = *(_DWORD *)(v60 + 40);
  v63 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 9);
  *(_DWORD *)(a1 + 200) = v61;
  v65 = v63;
  LODWORD(v63) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 204) = v62;
  v50 = *(_BYTE *)(v65 + 36) == 0;
  *(_DWORD *)(a1 + 208) = v63;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 212) = *(_DWORD *)(v65 + 32);
  }
  else
  {
    v1365 = v64;
    if ( sub_1691920(v64, *(unsigned __int16 *)(v65 + 40)) )
      *(_DWORD *)(a1 + 212) = *(_DWORD *)(sub_1691920(v1365, *(unsigned __int16 *)(v65 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 212) = 0;
  }
  v66 = 0;
  v67 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xAu);
  if ( v67 )
  {
    if ( *(_DWORD *)(v67 + 56) )
      v66 = **(_QWORD **)(v67 + 48);
    v68 = *(_DWORD *)(v67 + 40);
  }
  else
  {
    v68 = 0;
  }
  v69 = v68;
  v70 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 10);
  v71 = *(_DWORD *)a1;
  v1379[1] = v72;
  v1379[0] = v70;
  sub_12D6090(a1 + 216, v66, v69, v1379, v71);
  v73 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xBu, "0");
  v74 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 11);
  *(_BYTE *)(a1 + 240) = v73;
  v76 = v74;
  v77 = v75;
  LODWORD(v74) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v76 + 36) == 0;
  *(_DWORD *)(a1 + 244) = HIDWORD(v73);
  *(_DWORD *)(a1 + 248) = v74;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 252) = *(_DWORD *)(v76 + 32);
  }
  else if ( sub_1691920(v75, *(unsigned __int16 *)(v76 + 40)) )
  {
    *(_DWORD *)(a1 + 252) = *(_DWORD *)(sub_1691920(v77, *(unsigned __int16 *)(v76 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 252) = 0;
  }
  v78 = 0;
  v79 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCu);
  if ( v79 )
  {
    if ( *(_DWORD *)(v79 + 56) )
      v78 = **(_QWORD **)(v79 + 48);
    v80 = *(_DWORD *)(v79 + 40);
  }
  else
  {
    v80 = 0;
  }
  v81 = v80;
  v82 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 12);
  v83 = *(_DWORD *)a1;
  v1380[1] = v84;
  v1380[0] = v82;
  sub_12D6090(a1 + 256, v78, v81, v1380, v83);
  v85 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xDu, "0");
  v86 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 13);
  *(_BYTE *)(a1 + 280) = v85;
  v88 = v86;
  v89 = v87;
  LODWORD(v86) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v88 + 36) == 0;
  *(_DWORD *)(a1 + 284) = HIDWORD(v85);
  *(_DWORD *)(a1 + 288) = v86;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 292) = *(_DWORD *)(v88 + 32);
  }
  else if ( sub_1691920(v87, *(unsigned __int16 *)(v88 + 40)) )
  {
    *(_DWORD *)(a1 + 292) = *(_DWORD *)(sub_1691920(v89, *(unsigned __int16 *)(v88 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 292) = 0;
  }
  v90 = 0;
  v91 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xEu);
  if ( v91 )
  {
    if ( *(_DWORD *)(v91 + 56) )
      v90 = **(_QWORD **)(v91 + 48);
    v92 = *(_DWORD *)(v91 + 40);
  }
  else
  {
    v92 = 0;
  }
  v93 = v92;
  v94 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 14);
  v95 = *(_DWORD *)a1;
  v96 = v90;
  v1381[1] = v97;
  v98 = 0;
  v1381[0] = v94;
  sub_12D6090(a1 + 296, v96, v93, v1381, v95);
  v99 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xFu, "0");
  v100 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 15);
  v101 = *(_DWORD *)a1;
  v1382[1] = v102;
  v1382[0] = v100;
  sub_12D6100(a1 + 320, v99, v1382, v101);
  v103 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x10u);
  if ( v103 )
  {
    if ( *(_DWORD *)(v103 + 56) )
      v98 = **(_QWORD **)(v103 + 48);
    v104 = *(_DWORD *)(v103 + 40);
  }
  else
  {
    v104 = 0;
  }
  v105 = v104;
  v106 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 16);
  v107 = *(_DWORD *)a1;
  v108 = v98;
  v1383[1] = v109;
  v110 = 0;
  v1383[0] = v106;
  sub_12D6090(a1 + 336, v108, v105, v1383, v107);
  v111 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x11u, "0");
  v112 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 17);
  v113 = *(_DWORD *)a1;
  v1384[1] = v114;
  v1384[0] = v112;
  sub_12D6100(a1 + 360, v111, v1384, v113);
  v115 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x12u);
  if ( v115 )
  {
    if ( *(_DWORD *)(v115 + 56) )
      v110 = **(_QWORD **)(v115 + 48);
    v116 = *(_DWORD *)(v115 + 40);
  }
  else
  {
    v116 = 0;
  }
  v117 = v116;
  v118 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 18);
  v119 = *(_DWORD *)a1;
  v120 = v110;
  v1385[1] = v121;
  v122 = 0;
  v1385[0] = v118;
  sub_12D6090(a1 + 376, v120, v117, v1385, v119);
  v123 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x13u, "1");
  v124 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 19);
  v125 = *(_DWORD *)a1;
  v1386[1] = v126;
  v1386[0] = v124;
  sub_12D6100(a1 + 400, v123, v1386, v125);
  v127 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x14u);
  if ( v127 )
  {
    if ( *(_DWORD *)(v127 + 56) )
      v122 = **(_QWORD **)(v127 + 48);
    v128 = *(_DWORD *)(v127 + 40);
  }
  else
  {
    v128 = 0;
  }
  v129 = v128;
  v130 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 20);
  v131 = *(_DWORD *)a1;
  v132 = v122;
  v1387[1] = v133;
  v134 = 0;
  v1387[0] = v130;
  sub_12D6090(a1 + 416, v132, v129, v1387, v131);
  v135 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x15u, "0");
  v136 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 21);
  v137 = *(_DWORD *)a1;
  v1388[1] = v138;
  v1388[0] = v136;
  sub_12D6100(a1 + 440, v135, v1388, v137);
  v139 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x16u);
  if ( v139 )
  {
    if ( *(_DWORD *)(v139 + 56) )
      v134 = **(_QWORD **)(v139 + 48);
    v140 = *(_DWORD *)(v139 + 40);
  }
  else
  {
    v140 = 0;
  }
  v141 = v140;
  v142 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 22);
  v143 = *(_DWORD *)a1;
  v144 = v134;
  v1389[1] = v145;
  v146 = 0;
  v1389[0] = v142;
  sub_12D6090(a1 + 456, v144, v141, v1389, v143);
  v147 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x17u, "0");
  v148 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 23);
  v149 = *(_DWORD *)a1;
  v1390[1] = v150;
  v1390[0] = v148;
  sub_12D6100(a1 + 480, v147, v1390, v149);
  v151 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x18u);
  if ( v151 )
  {
    if ( *(_DWORD *)(v151 + 56) )
      v146 = **(_QWORD **)(v151 + 48);
    v152 = *(_DWORD *)(v151 + 40);
  }
  else
  {
    v152 = 0;
  }
  v153 = v152;
  v154 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 24);
  v155 = *(_DWORD *)a1;
  v156 = v146;
  v1391[1] = v157;
  v158 = 0;
  v1391[0] = v154;
  sub_12D6090(a1 + 496, v156, v153, v1391, v155);
  v159 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x19u, "1");
  v160 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 25);
  v161 = *(_DWORD *)a1;
  v1392[1] = v162;
  v1392[0] = v160;
  sub_12D6100(a1 + 520, v159, v1392, v161);
  v163 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x1Au);
  if ( v163 )
  {
    if ( *(_DWORD *)(v163 + 56) )
      v158 = **(_QWORD **)(v163 + 48);
    v164 = *(_DWORD *)(v163 + 40);
  }
  else
  {
    v164 = 0;
  }
  v165 = v164;
  v166 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 26);
  v167 = *(_DWORD *)a1;
  v168 = v158;
  v1393[1] = v169;
  v170 = 0;
  v1393[0] = v166;
  sub_12D6090(a1 + 536, v168, v165, v1393, v167);
  v171 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x1Bu, "0");
  v172 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 27);
  v173 = *(_DWORD *)a1;
  v1394[1] = v174;
  v1394[0] = v172;
  sub_12D6100(a1 + 560, v171, v1394, v173);
  v175 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x1Cu);
  if ( v175 )
  {
    if ( *(_DWORD *)(v175 + 56) )
      v170 = **(_QWORD **)(v175 + 48);
    v176 = *(_DWORD *)(v175 + 40);
  }
  else
  {
    v176 = 0;
  }
  v177 = v176;
  v178 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 28);
  v179 = *(_DWORD *)a1;
  v180 = v170;
  v1395[1] = v181;
  v182 = 0;
  v1395[0] = v178;
  sub_12D6090(a1 + 576, v180, v177, v1395, v179);
  v183 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x1Du, "0");
  v184 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 29);
  v185 = *(_DWORD *)a1;
  v1396[1] = v186;
  v1396[0] = v184;
  sub_12D6100(a1 + 600, v183, v1396, v185);
  v187 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x1Eu);
  if ( v187 )
  {
    if ( *(_DWORD *)(v187 + 56) )
      v182 = **(_QWORD **)(v187 + 48);
    v188 = *(_DWORD *)(v187 + 40);
  }
  else
  {
    v188 = 0;
  }
  v189 = v188;
  v190 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 30);
  v191 = *(_DWORD *)a1;
  v192 = v182;
  v1397[1] = v193;
  v194 = 0;
  v1397[0] = v190;
  sub_12D6090(a1 + 616, v192, v189, v1397, v191);
  v195 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x1Fu, "0");
  v196 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 31);
  v197 = *(_DWORD *)a1;
  v1398[1] = v198;
  v1398[0] = v196;
  sub_12D6100(a1 + 640, v195, v1398, v197);
  v199 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x20u);
  if ( v199 )
  {
    if ( *(_DWORD *)(v199 + 56) )
      v194 = **(_QWORD **)(v199 + 48);
    v200 = *(_DWORD *)(v199 + 40);
  }
  else
  {
    v200 = 0;
  }
  v201 = v200;
  v202 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 32);
  v203 = *(_DWORD *)a1;
  v204 = v194;
  v1399[1] = v205;
  v206 = 0;
  v1399[0] = v202;
  sub_12D6090(a1 + 656, v204, v201, v1399, v203);
  v207 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x21u, "0");
  v208 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 33);
  v209 = *(_DWORD *)a1;
  v1400[1] = v210;
  v1400[0] = v208;
  sub_12D6100(a1 + 680, v207, v1400, v209);
  v211 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x22u);
  if ( v211 )
  {
    if ( *(_DWORD *)(v211 + 56) )
      v206 = **(_QWORD **)(v211 + 48);
    v212 = *(_DWORD *)(v211 + 40);
  }
  else
  {
    v212 = 0;
  }
  v213 = v212;
  v214 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 34);
  v215 = *(_DWORD *)a1;
  v216 = v206;
  v1401[1] = v217;
  v218 = 0;
  v1401[0] = v214;
  sub_12D6090(a1 + 696, v216, v213, v1401, v215);
  v219 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x23u, "0");
  v220 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 35);
  v221 = *(_DWORD *)a1;
  v1402[1] = v222;
  v1402[0] = v220;
  sub_12D6100(a1 + 720, v219, v1402, v221);
  v223 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x24u);
  if ( v223 )
  {
    if ( *(_DWORD *)(v223 + 56) )
      v218 = **(_QWORD **)(v223 + 48);
    v224 = *(_DWORD *)(v223 + 40);
  }
  else
  {
    v224 = 0;
  }
  v225 = v224;
  v226 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 36);
  v227 = *(_DWORD *)a1;
  v228 = v218;
  v1403[1] = v229;
  v230 = 0;
  v1403[0] = v226;
  sub_12D6090(a1 + 736, v228, v225, v1403, v227);
  v231 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x25u, "0");
  v232 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 37);
  v233 = *(_DWORD *)a1;
  v1404[1] = v234;
  v1404[0] = v232;
  sub_12D6100(a1 + 760, v231, v1404, v233);
  v235 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x26u);
  if ( v235 )
  {
    if ( *(_DWORD *)(v235 + 56) )
      v230 = **(_QWORD **)(v235 + 48);
    v236 = *(_DWORD *)(v235 + 40);
  }
  else
  {
    v236 = 0;
  }
  v237 = v236;
  v238 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 38);
  v239 = *(_DWORD *)a1;
  v240 = v230;
  v1405[1] = v241;
  v242 = 0;
  v1405[0] = v238;
  sub_12D6090(a1 + 776, v240, v237, v1405, v239);
  v243 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x27u, "0");
  v244 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 39);
  v245 = *(_DWORD *)a1;
  v1406[1] = v246;
  v1406[0] = v244;
  sub_12D6100(a1 + 800, v243, v1406, v245);
  v247 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x28u);
  if ( v247 )
  {
    if ( *(_DWORD *)(v247 + 56) )
      v242 = **(_QWORD **)(v247 + 48);
    v248 = *(_DWORD *)(v247 + 40);
  }
  else
  {
    v248 = 0;
  }
  v249 = v248;
  v250 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 40);
  v251 = *(_DWORD *)a1;
  v252 = v242;
  v1407[1] = v253;
  v254 = 0;
  v1407[0] = v250;
  sub_12D6090(a1 + 816, v252, v249, v1407, v251);
  v255 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x29u, "0");
  v256 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 41);
  v257 = *(_DWORD *)a1;
  v1408[1] = v258;
  v1408[0] = v256;
  sub_12D6100(a1 + 840, v255, v1408, v257);
  v259 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x2Au);
  if ( v259 )
  {
    if ( *(_DWORD *)(v259 + 56) )
      v254 = **(_QWORD **)(v259 + 48);
    v260 = *(_DWORD *)(v259 + 40);
  }
  else
  {
    v260 = 0;
  }
  v261 = v260;
  v262 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 42);
  v263 = *(_DWORD *)a1;
  v264 = v254;
  v1409[1] = v265;
  v266 = 0;
  v1409[0] = v262;
  sub_12D6090(a1 + 856, v264, v261, v1409, v263);
  v267 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x2Bu, "0");
  v268 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 43);
  v269 = *(_DWORD *)a1;
  v1410[1] = v270;
  v1410[0] = v268;
  sub_12D6100(a1 + 880, v267, v1410, v269);
  v271 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x2Cu);
  if ( v271 )
  {
    if ( *(_DWORD *)(v271 + 56) )
      v266 = **(_QWORD **)(v271 + 48);
    v272 = *(_DWORD *)(v271 + 40);
  }
  else
  {
    v272 = 0;
  }
  v273 = v272;
  v274 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 44);
  v275 = *(_DWORD *)a1;
  v276 = v266;
  v1411[1] = v277;
  v278 = 0;
  v1411[0] = v274;
  sub_12D6090(a1 + 896, v276, v273, v1411, v275);
  v279 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x2Du, "0");
  v280 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 45);
  v281 = *(_DWORD *)a1;
  v1412[1] = v282;
  v1412[0] = v280;
  sub_12D6100(a1 + 920, v279, v1412, v281);
  v283 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x2Eu);
  if ( v283 )
  {
    if ( *(_DWORD *)(v283 + 56) )
      v278 = **(_QWORD **)(v283 + 48);
    v284 = *(_DWORD *)(v283 + 40);
  }
  else
  {
    v284 = 0;
  }
  v285 = v284;
  v286 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 46);
  v287 = *(_DWORD *)a1;
  v288 = v278;
  v1413[1] = v289;
  v290 = 0;
  v1413[0] = v286;
  sub_12D6090(a1 + 936, v288, v285, v1413, v287);
  v291 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x2Fu, "0");
  v292 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 47);
  v293 = *(_DWORD *)a1;
  v1414[1] = v294;
  v1414[0] = v292;
  sub_12D6100(a1 + 960, v291, v1414, v293);
  v295 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x30u);
  if ( v295 )
  {
    if ( *(_DWORD *)(v295 + 56) )
      v290 = **(_QWORD **)(v295 + 48);
    v296 = *(_DWORD *)(v295 + 40);
  }
  else
  {
    v296 = 0;
  }
  v297 = v296;
  v298 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 48);
  v299 = *(_DWORD *)a1;
  v1415[1] = v300;
  v1415[0] = v298;
  sub_12D6090(a1 + 976, v290, v297, v1415, v299);
  v301 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x31u, "0");
  v302 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 49);
  *(_BYTE *)(a1 + 1000) = v301;
  v304 = v302;
  v305 = v303;
  LODWORD(v302) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v304 + 36) == 0;
  *(_DWORD *)(a1 + 1004) = HIDWORD(v301);
  *(_DWORD *)(a1 + 1008) = v302;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1012) = *(_DWORD *)(v304 + 32);
  }
  else if ( sub_1691920(v303, *(unsigned __int16 *)(v304 + 40)) )
  {
    *(_DWORD *)(a1 + 1012) = *(_DWORD *)(sub_1691920(v305, *(unsigned __int16 *)(v304 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 1012) = 0;
  }
  v306 = 0;
  v307 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x32u);
  if ( v307 )
  {
    if ( *(_DWORD *)(v307 + 56) )
      v306 = **(_QWORD **)(v307 + 48);
    v308 = *(_DWORD *)(v307 + 40);
  }
  else
  {
    v308 = 0;
  }
  v309 = v308;
  v310 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 50);
  v311 = *(_DWORD *)a1;
  v312 = v306;
  v1416[1] = v313;
  v314 = 0;
  v1416[0] = v310;
  sub_12D6090(a1 + 1016, v312, v309, v1416, v311);
  v315 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x33u, "0");
  v316 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 51);
  v317 = *(_DWORD *)a1;
  v1417[1] = v318;
  v1417[0] = v316;
  sub_12D6100(a1 + 1040, v315, v1417, v317);
  v319 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x34u);
  if ( v319 )
  {
    if ( *(_DWORD *)(v319 + 56) )
      v314 = **(_QWORD **)(v319 + 48);
    v320 = *(_DWORD *)(v319 + 40);
  }
  else
  {
    v320 = 0;
  }
  v321 = v320;
  v322 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 52);
  v323 = *(_DWORD *)a1;
  v1418[1] = v324;
  v1418[0] = v322;
  sub_12D6090(a1 + 1056, v314, v321, v1418, v323);
  v325 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x35u, "0");
  v326 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 53);
  *(_BYTE *)(a1 + 1080) = v325;
  v328 = v326;
  v329 = v327;
  LODWORD(v326) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v328 + 36) == 0;
  *(_DWORD *)(a1 + 1084) = HIDWORD(v325);
  *(_DWORD *)(a1 + 1088) = v326;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1092) = *(_DWORD *)(v328 + 32);
  }
  else if ( sub_1691920(v327, *(unsigned __int16 *)(v328 + 40)) )
  {
    *(_DWORD *)(a1 + 1092) = *(_DWORD *)(sub_1691920(v329, *(unsigned __int16 *)(v328 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 1092) = 0;
  }
  v330 = 0;
  v331 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x36u);
  if ( v331 )
  {
    if ( *(_DWORD *)(v331 + 56) )
      v330 = **(_QWORD **)(v331 + 48);
    v332 = *(_DWORD *)(v331 + 40);
  }
  else
  {
    v332 = 0;
  }
  v333 = v332;
  v334 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 54);
  v335 = *(_DWORD *)a1;
  v1419[1] = v336;
  v1419[0] = v334;
  sub_12D6090(a1 + 1096, v330, v333, v1419, v335);
  v337 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x37u, "0");
  v338 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 55);
  *(_BYTE *)(a1 + 1120) = v337;
  v340 = v338;
  v341 = v339;
  LODWORD(v338) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v340 + 36) == 0;
  *(_DWORD *)(a1 + 1124) = HIDWORD(v337);
  *(_DWORD *)(a1 + 1128) = v338;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1132) = *(_DWORD *)(v340 + 32);
  }
  else if ( sub_1691920(v339, *(unsigned __int16 *)(v340 + 40)) )
  {
    *(_DWORD *)(a1 + 1132) = *(_DWORD *)(sub_1691920(v341, *(unsigned __int16 *)(v340 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 1132) = 0;
  }
  v342 = 0;
  v343 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x38u);
  if ( v343 )
  {
    if ( *(_DWORD *)(v343 + 56) )
      v342 = **(_QWORD **)(v343 + 48);
    v344 = *(_DWORD *)(v343 + 40);
  }
  else
  {
    v344 = 0;
  }
  v345 = v344;
  v346 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 56);
  v347 = *(_DWORD *)a1;
  v348 = v342;
  v1420[1] = v349;
  v350 = 0;
  v1420[0] = v346;
  sub_12D6090(a1 + 1136, v348, v345, v1420, v347);
  v351 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x39u, "0");
  v352 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 57);
  v353 = *(_DWORD *)a1;
  v1421[1] = v354;
  v1421[0] = v352;
  sub_12D6100(a1 + 1160, v351, v1421, v353);
  v355 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x3Au);
  if ( v355 )
  {
    if ( *(_DWORD *)(v355 + 56) )
      v350 = **(_QWORD **)(v355 + 48);
    v356 = *(_DWORD *)(v355 + 40);
  }
  else
  {
    v356 = 0;
  }
  v357 = v356;
  v358 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 58);
  v359 = *(_DWORD *)a1;
  v1422[1] = v360;
  v1422[0] = v358;
  sub_12D6090(a1 + 1176, v350, v357, v1422, v359);
  v361 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x3Bu, "0");
  v362 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 59);
  *(_BYTE *)(a1 + 1200) = v361;
  v364 = v362;
  v365 = v363;
  LODWORD(v362) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v364 + 36) == 0;
  *(_DWORD *)(a1 + 1204) = HIDWORD(v361);
  *(_DWORD *)(a1 + 1208) = v362;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1212) = *(_DWORD *)(v364 + 32);
  }
  else if ( sub_1691920(v363, *(unsigned __int16 *)(v364 + 40)) )
  {
    *(_DWORD *)(a1 + 1212) = *(_DWORD *)(sub_1691920(v365, *(unsigned __int16 *)(v364 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 1212) = 0;
  }
  v366 = 0;
  v367 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x3Cu);
  if ( v367 )
  {
    if ( *(_DWORD *)(v367 + 56) )
      v366 = **(_QWORD **)(v367 + 48);
    v368 = *(_DWORD *)(v367 + 40);
  }
  else
  {
    v368 = 0;
  }
  v369 = v368;
  v370 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 60);
  v371 = *(_DWORD *)a1;
  v1423[1] = v372;
  v1423[0] = v370;
  sub_12D6090(a1 + 1216, v366, v369, v1423, v371);
  v373 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x3Du, "0");
  v374 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 61);
  *(_BYTE *)(a1 + 1240) = v373;
  v376 = v374;
  v377 = v375;
  LODWORD(v374) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v376 + 36) == 0;
  *(_DWORD *)(a1 + 1244) = HIDWORD(v373);
  *(_DWORD *)(a1 + 1248) = v374;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1252) = *(_DWORD *)(v376 + 32);
  }
  else if ( sub_1691920(v375, *(unsigned __int16 *)(v376 + 40)) )
  {
    *(_DWORD *)(a1 + 1252) = *(_DWORD *)(sub_1691920(v377, *(unsigned __int16 *)(v376 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 1252) = 0;
  }
  v378 = 0;
  v379 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x3Eu);
  if ( v379 )
  {
    if ( *(_DWORD *)(v379 + 56) )
      v378 = **(_QWORD **)(v379 + 48);
    v380 = *(_DWORD *)(v379 + 40);
  }
  else
  {
    v380 = 0;
  }
  v381 = v380;
  v382 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 62);
  v383 = *(_DWORD *)a1;
  v384 = v378;
  v1424[1] = v385;
  v386 = 0;
  v1424[0] = v382;
  sub_12D6090(a1 + 1256, v384, v381, v1424, v383);
  v387 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x3Fu, "0");
  v388 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 63);
  v389 = *(_DWORD *)a1;
  v1425[1] = v390;
  v1425[0] = v388;
  sub_12D6100(a1 + 1280, v387, v1425, v389);
  v391 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x40u);
  if ( v391 )
  {
    if ( *(_DWORD *)(v391 + 56) )
      v386 = **(_QWORD **)(v391 + 48);
    v392 = *(_DWORD *)(v391 + 40);
  }
  else
  {
    v392 = 0;
  }
  v393 = v392;
  v394 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 64);
  v395 = *(_DWORD *)a1;
  v396 = v386;
  v1426[1] = v397;
  v398 = 0;
  v1426[0] = v394;
  sub_12D6090(a1 + 1296, v396, v393, v1426, v395);
  v399 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x41u, "0");
  v400 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 65);
  v401 = *(_DWORD *)a1;
  v1427[1] = v402;
  v1427[0] = v400;
  sub_12D6100(a1 + 1320, v399, v1427, v401);
  v403 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x42u);
  if ( v403 )
  {
    if ( *(_DWORD *)(v403 + 56) )
      v398 = **(_QWORD **)(v403 + 48);
    v404 = *(_DWORD *)(v403 + 40);
  }
  else
  {
    v404 = 0;
  }
  v405 = v404;
  v406 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 66);
  v407 = *(_DWORD *)a1;
  v408 = v398;
  v1428[1] = v409;
  v410 = 0;
  v1428[0] = v406;
  sub_12D6090(a1 + 1336, v408, v405, v1428, v407);
  v411 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x43u, "0");
  v412 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 67);
  v413 = *(_DWORD *)a1;
  v1429[1] = v414;
  v1429[0] = v412;
  sub_12D6100(a1 + 1360, v411, v1429, v413);
  v415 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x44u);
  if ( v415 )
  {
    if ( *(_DWORD *)(v415 + 56) )
      v410 = **(_QWORD **)(v415 + 48);
    v416 = *(_DWORD *)(v415 + 40);
  }
  else
  {
    v416 = 0;
  }
  v417 = v416;
  v418 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 68);
  v419 = *(_DWORD *)a1;
  v420 = v410;
  v1430[1] = v421;
  v422 = 0;
  v1430[0] = v418;
  sub_12D6090(a1 + 1376, v420, v417, v1430, v419);
  v423 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x45u, "0");
  v424 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 69);
  v425 = *(_DWORD *)a1;
  v1431[1] = v426;
  v1431[0] = v424;
  sub_12D6100(a1 + 1400, v423, v1431, v425);
  v427 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x46u);
  if ( v427 )
  {
    if ( *(_DWORD *)(v427 + 56) )
      v422 = **(_QWORD **)(v427 + 48);
    v428 = *(_DWORD *)(v427 + 40);
  }
  else
  {
    v428 = 0;
  }
  v429 = v428;
  v430 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 70);
  v431 = *(_DWORD *)a1;
  v432 = v422;
  v1432[1] = v433;
  v434 = 0;
  v1432[0] = v430;
  sub_12D6090(a1 + 1416, v432, v429, v1432, v431);
  v435 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x47u, "0");
  v436 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 71);
  v437 = *(_DWORD *)a1;
  v1433[1] = v438;
  v1433[0] = v436;
  sub_12D6100(a1 + 1440, v435, v1433, v437);
  v439 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x48u);
  if ( v439 )
  {
    if ( *(_DWORD *)(v439 + 56) )
      v434 = **(_QWORD **)(v439 + 48);
    v440 = *(_DWORD *)(v439 + 40);
  }
  else
  {
    v440 = 0;
  }
  v441 = v440;
  v442 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 72);
  v443 = *(_DWORD *)a1;
  v444 = v434;
  v1434[1] = v445;
  v446 = 0;
  v1434[0] = v442;
  sub_12D6090(a1 + 1456, v444, v441, v1434, v443);
  v447 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x49u, "0");
  v448 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 73);
  v449 = *(_DWORD *)a1;
  v1435[1] = v450;
  v1435[0] = v448;
  sub_12D6100(a1 + 1480, v447, v1435, v449);
  v451 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x4Au);
  if ( v451 )
  {
    if ( *(_DWORD *)(v451 + 56) )
      v446 = **(_QWORD **)(v451 + 48);
    v452 = *(_DWORD *)(v451 + 40);
  }
  else
  {
    v452 = 0;
  }
  v453 = v452;
  v454 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 74);
  v455 = *(_DWORD *)a1;
  v456 = v446;
  v1436[1] = v457;
  v458 = 0;
  v1436[0] = v454;
  sub_12D6090(a1 + 1496, v456, v453, v1436, v455);
  v459 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x4Bu, "0");
  v460 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 75);
  v461 = *(_DWORD *)a1;
  v1437[1] = v462;
  v1437[0] = v460;
  sub_12D6100(a1 + 1520, v459, v1437, v461);
  v463 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x4Cu);
  if ( v463 )
  {
    if ( *(_DWORD *)(v463 + 56) )
      v458 = **(_QWORD **)(v463 + 48);
    v464 = *(_DWORD *)(v463 + 40);
  }
  else
  {
    v464 = 0;
  }
  v465 = v464;
  v466 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 76);
  v467 = *(_DWORD *)a1;
  v468 = v458;
  v1438[1] = v469;
  v470 = 0;
  v1438[0] = v466;
  sub_12D6090(a1 + 1536, v468, v465, v1438, v467);
  v471 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x4Du, "0");
  v472 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 77);
  v473 = *(_DWORD *)a1;
  v1439[1] = v474;
  v1439[0] = v472;
  sub_12D6100(a1 + 1560, v471, v1439, v473);
  v475 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x4Eu);
  if ( v475 )
  {
    if ( *(_DWORD *)(v475 + 56) )
      v470 = **(_QWORD **)(v475 + 48);
    v476 = *(_DWORD *)(v475 + 40);
  }
  else
  {
    v476 = 0;
  }
  v477 = v476;
  v478 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 78);
  v479 = *(_DWORD *)a1;
  v480 = v470;
  v1440[1] = v481;
  v482 = 0;
  v1440[0] = v478;
  sub_12D6090(a1 + 1576, v480, v477, v1440, v479);
  v483 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x4Fu, "0");
  v484 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 79);
  v485 = *(_DWORD *)a1;
  v1441[1] = v486;
  v1441[0] = v484;
  sub_12D6100(a1 + 1600, v483, v1441, v485);
  v487 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x50u);
  if ( v487 )
  {
    if ( *(_DWORD *)(v487 + 56) )
      v482 = **(_QWORD **)(v487 + 48);
    v488 = *(_DWORD *)(v487 + 40);
  }
  else
  {
    v488 = 0;
  }
  v489 = v488;
  v490 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 80);
  v491 = *(_DWORD *)a1;
  v492 = v482;
  v1442[1] = v493;
  v494 = 0;
  v1442[0] = v490;
  sub_12D6090(a1 + 1616, v492, v489, v1442, v491);
  v495 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x51u, "0");
  v496 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 81);
  v497 = *(_DWORD *)a1;
  v1443[1] = v498;
  v1443[0] = v496;
  sub_12D6100(a1 + 1640, v495, v1443, v497);
  v499 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x52u);
  if ( v499 )
  {
    if ( *(_DWORD *)(v499 + 56) )
      v494 = **(_QWORD **)(v499 + 48);
    v500 = *(_DWORD *)(v499 + 40);
  }
  else
  {
    v500 = 0;
  }
  v501 = v500;
  v502 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 82);
  v503 = *(_DWORD *)a1;
  v504 = v494;
  v1444[1] = v505;
  v506 = 0;
  v1444[0] = v502;
  sub_12D6090(a1 + 1656, v504, v501, v1444, v503);
  v507 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x53u, "0");
  v508 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 83);
  v509 = *(_DWORD *)a1;
  v1445[1] = v510;
  v1445[0] = v508;
  sub_12D6100(a1 + 1680, v507, v1445, v509);
  v511 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x54u);
  if ( v511 )
  {
    if ( *(_DWORD *)(v511 + 56) )
      v506 = **(_QWORD **)(v511 + 48);
    v512 = *(_DWORD *)(v511 + 40);
  }
  else
  {
    v512 = 0;
  }
  v513 = v512;
  v514 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 84);
  v515 = *(_DWORD *)a1;
  v516 = v506;
  v1446[1] = v517;
  v518 = 0;
  v1446[0] = v514;
  sub_12D6090(a1 + 1696, v516, v513, v1446, v515);
  v519 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x55u, "0");
  v520 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 85);
  v521 = *(_DWORD *)a1;
  v1447[1] = v522;
  v1447[0] = v520;
  sub_12D6100(a1 + 1720, v519, v1447, v521);
  v523 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x56u);
  if ( v523 )
  {
    if ( *(_DWORD *)(v523 + 56) )
      v518 = **(_QWORD **)(v523 + 48);
    v524 = *(_DWORD *)(v523 + 40);
  }
  else
  {
    v524 = 0;
  }
  v525 = v524;
  v526 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 86);
  v527 = *(_DWORD *)a1;
  v528 = v518;
  v1448[1] = v529;
  v530 = 0;
  v1448[0] = v526;
  sub_12D6090(a1 + 1736, v528, v525, v1448, v527);
  v531 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x57u, "0");
  v532 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 87);
  v533 = *(_DWORD *)a1;
  v1449[1] = v534;
  v1449[0] = v532;
  sub_12D6100(a1 + 1760, v531, v1449, v533);
  v535 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x58u);
  if ( v535 )
  {
    if ( *(_DWORD *)(v535 + 56) )
      v530 = **(_QWORD **)(v535 + 48);
    v536 = *(_DWORD *)(v535 + 40);
  }
  else
  {
    v536 = 0;
  }
  v537 = v536;
  v538 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 88);
  v539 = *(_DWORD *)a1;
  v540 = v530;
  v1450[1] = v541;
  v542 = 0;
  v1450[0] = v538;
  sub_12D6090(a1 + 1776, v540, v537, v1450, v539);
  v543 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x59u, "0");
  v544 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 89);
  v545 = *(_DWORD *)a1;
  v1451[1] = v546;
  v1451[0] = v544;
  sub_12D6100(a1 + 1800, v543, v1451, v545);
  v547 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x5Au);
  if ( v547 )
  {
    if ( *(_DWORD *)(v547 + 56) )
      v542 = **(_QWORD **)(v547 + 48);
    v548 = *(_DWORD *)(v547 + 40);
  }
  else
  {
    v548 = 0;
  }
  v549 = v548;
  v550 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 90);
  v551 = *(_DWORD *)a1;
  v552 = v542;
  v1452[1] = v553;
  v554 = 0;
  v1452[0] = v550;
  sub_12D6090(a1 + 1816, v552, v549, v1452, v551);
  v555 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x5Bu, "0");
  v556 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 91);
  v557 = *(_DWORD *)a1;
  v1453[1] = v558;
  v1453[0] = v556;
  sub_12D6100(a1 + 1840, v555, v1453, v557);
  v559 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x5Cu);
  if ( v559 )
  {
    if ( *(_DWORD *)(v559 + 56) )
      v554 = **(_QWORD **)(v559 + 48);
    v560 = *(_DWORD *)(v559 + 40);
  }
  else
  {
    v560 = 0;
  }
  v561 = v560;
  v562 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 92);
  v563 = *(_DWORD *)a1;
  v564 = v554;
  v1454[1] = v565;
  v566 = 0;
  v1454[0] = v562;
  sub_12D6090(a1 + 1856, v564, v561, v1454, v563);
  v567 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x5Du, "1");
  v568 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 93);
  v569 = *(_DWORD *)a1;
  v1455[1] = v570;
  v1455[0] = v568;
  sub_12D6100(a1 + 1880, v567, v1455, v569);
  v571 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x5Eu);
  if ( v571 )
  {
    if ( *(_DWORD *)(v571 + 56) )
      v566 = **(_QWORD **)(v571 + 48);
    v572 = *(_DWORD *)(v571 + 40);
  }
  else
  {
    v572 = 0;
  }
  v573 = v572;
  v574 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 94);
  v575 = *(_DWORD *)a1;
  v1456[1] = v576;
  v1456[0] = v574;
  sub_12D6090(a1 + 1896, v566, v573, v1456, v575);
  v577 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x5Fu, "1");
  v578 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 95);
  *(_BYTE *)(a1 + 1920) = v577;
  v580 = v578;
  v581 = v579;
  LODWORD(v578) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v580 + 36) == 0;
  *(_DWORD *)(a1 + 1924) = HIDWORD(v577);
  *(_DWORD *)(a1 + 1928) = v578;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1932) = *(_DWORD *)(v580 + 32);
  }
  else if ( sub_1691920(v579, *(unsigned __int16 *)(v580 + 40)) )
  {
    *(_DWORD *)(a1 + 1932) = *(_DWORD *)(sub_1691920(v581, *(unsigned __int16 *)(v580 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 1932) = 0;
  }
  v582 = 0;
  v583 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x60u);
  if ( v583 )
  {
    if ( *(_DWORD *)(v583 + 56) )
      v582 = **(_QWORD **)(v583 + 48);
    v584 = *(_DWORD *)(v583 + 40);
  }
  else
  {
    v584 = 0;
  }
  v585 = v584;
  v586 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 96);
  v587 = *(_DWORD *)a1;
  v588 = v582;
  v1457[1] = v589;
  v590 = 0;
  v1457[0] = v586;
  sub_12D6090(a1 + 1936, v588, v585, v1457, v587);
  v591 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x61u, "0");
  v592 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 97);
  v593 = *(_DWORD *)a1;
  v1458[1] = v594;
  v1458[0] = v592;
  sub_12D6100(a1 + 1960, v591, v1458, v593);
  v595 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x62u);
  if ( v595 )
  {
    if ( *(_DWORD *)(v595 + 56) )
      v590 = **(_QWORD **)(v595 + 48);
    v596 = *(_DWORD *)(v595 + 40);
  }
  else
  {
    v596 = 0;
  }
  v597 = v596;
  v598 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 98);
  v599 = *(_DWORD *)a1;
  v600 = v590;
  v1459[1] = v601;
  v602 = 0;
  v1459[0] = v598;
  sub_12D6090(a1 + 1976, v600, v597, v1459, v599);
  v603 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x63u, "0");
  v604 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 99);
  v605 = *(_DWORD *)a1;
  v1460[1] = v606;
  v1460[0] = v604;
  sub_12D6100(a1 + 2000, v603, v1460, v605);
  v607 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x64u);
  if ( v607 )
  {
    if ( *(_DWORD *)(v607 + 56) )
      v602 = **(_QWORD **)(v607 + 48);
    v608 = *(_DWORD *)(v607 + 40);
  }
  else
  {
    v608 = 0;
  }
  v609 = v608;
  v610 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 100);
  v611 = *(_DWORD *)a1;
  v612 = v602;
  v1461[1] = v613;
  v614 = 0;
  v1461[0] = v610;
  sub_12D6090(a1 + 2016, v612, v609, v1461, v611);
  v615 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x65u, "0");
  v616 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 101);
  v617 = *(_DWORD *)a1;
  v1462[1] = v618;
  v1462[0] = v616;
  sub_12D6100(a1 + 2040, v615, v1462, v617);
  v619 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x66u);
  if ( v619 )
  {
    if ( *(_DWORD *)(v619 + 56) )
      v614 = **(_QWORD **)(v619 + 48);
    v620 = *(_DWORD *)(v619 + 40);
  }
  else
  {
    v620 = 0;
  }
  v621 = v620;
  v622 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 102);
  v623 = *(_DWORD *)a1;
  v1463[1] = v624;
  v1463[0] = v622;
  sub_12D6090(a1 + 2056, v614, v621, v1463, v623);
  v625 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x67u, "0");
  v626 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 103);
  *(_BYTE *)(a1 + 2080) = v625;
  v628 = v626;
  v629 = v627;
  LODWORD(v626) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v628 + 36) == 0;
  *(_DWORD *)(a1 + 2084) = HIDWORD(v625);
  *(_DWORD *)(a1 + 2088) = v626;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 2092) = *(_DWORD *)(v628 + 32);
  }
  else if ( sub_1691920(v627, *(unsigned __int16 *)(v628 + 40)) )
  {
    *(_DWORD *)(a1 + 2092) = *(_DWORD *)(sub_1691920(v629, *(unsigned __int16 *)(v628 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 2092) = 0;
  }
  v630 = 0;
  v631 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x68u);
  if ( v631 )
  {
    if ( *(_DWORD *)(v631 + 56) )
      v630 = **(_QWORD **)(v631 + 48);
    v632 = *(_DWORD *)(v631 + 40);
  }
  else
  {
    v632 = 0;
  }
  v633 = v632;
  v634 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 104);
  v635 = *(_DWORD *)a1;
  v636 = v630;
  v1464[1] = v637;
  v638 = 0;
  v1464[0] = v634;
  sub_12D6090(a1 + 2096, v636, v633, v1464, v635);
  v639 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x69u, "0");
  v640 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 105);
  v641 = *(_DWORD *)a1;
  v1465[1] = v642;
  v1465[0] = v640;
  sub_12D6100(a1 + 2120, v639, v1465, v641);
  v643 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x6Au);
  if ( v643 )
  {
    if ( *(_DWORD *)(v643 + 56) )
      v638 = **(_QWORD **)(v643 + 48);
    v644 = *(_DWORD *)(v643 + 40);
  }
  else
  {
    v644 = 0;
  }
  v645 = v644;
  v646 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 106);
  v647 = *(_DWORD *)a1;
  v648 = v638;
  v1466[1] = v649;
  v650 = 0;
  v1466[0] = v646;
  sub_12D6090(a1 + 2136, v648, v645, v1466, v647);
  v651 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x6Bu, "0");
  v652 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 107);
  v653 = *(_DWORD *)a1;
  v1467[1] = v654;
  v1467[0] = v652;
  sub_12D6100(a1 + 2160, v651, v1467, v653);
  v655 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x6Cu);
  if ( v655 )
  {
    if ( *(_DWORD *)(v655 + 56) )
      v650 = **(_QWORD **)(v655 + 48);
    v656 = *(_DWORD *)(v655 + 40);
  }
  else
  {
    v656 = 0;
  }
  v657 = v656;
  v658 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 108);
  v659 = *(_DWORD *)a1;
  v660 = v650;
  v1468[1] = v661;
  v662 = 0;
  v1468[0] = v658;
  sub_12D6090(a1 + 2176, v660, v657, v1468, v659);
  v663 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x6Du, "0");
  v664 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 109);
  v665 = *(_DWORD *)a1;
  v1469[1] = v666;
  v1469[0] = v664;
  sub_12D6100(a1 + 2200, v663, v1469, v665);
  v667 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x6Eu);
  if ( v667 )
  {
    if ( *(_DWORD *)(v667 + 56) )
      v662 = **(_QWORD **)(v667 + 48);
    v668 = *(_DWORD *)(v667 + 40);
  }
  else
  {
    v668 = 0;
  }
  v669 = v668;
  v670 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 110);
  v671 = *(_DWORD *)a1;
  v672 = v662;
  v1470[1] = v673;
  v674 = 0;
  v1470[0] = v670;
  sub_12D6090(a1 + 2216, v672, v669, v1470, v671);
  v675 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x6Fu, "0");
  v676 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 111);
  v677 = *(_DWORD *)a1;
  v1471[1] = v678;
  v1471[0] = v676;
  sub_12D6100(a1 + 2240, v675, v1471, v677);
  v679 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x70u);
  if ( v679 )
  {
    if ( *(_DWORD *)(v679 + 56) )
      v674 = **(_QWORD **)(v679 + 48);
    v680 = *(_DWORD *)(v679 + 40);
  }
  else
  {
    v680 = 0;
  }
  v681 = v680;
  v682 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 112);
  v683 = *(_DWORD *)a1;
  v684 = v674;
  v1472[1] = v685;
  v686 = 0;
  v1472[0] = v682;
  sub_12D6090(a1 + 2256, v684, v681, v1472, v683);
  v687 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x71u, "0");
  v688 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 113);
  v689 = *(_DWORD *)a1;
  v1473[1] = v690;
  v1473[0] = v688;
  sub_12D6100(a1 + 2280, v687, v1473, v689);
  v691 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x72u);
  if ( v691 )
  {
    if ( *(_DWORD *)(v691 + 56) )
      v686 = **(_QWORD **)(v691 + 48);
    v692 = *(_DWORD *)(v691 + 40);
  }
  else
  {
    v692 = 0;
  }
  v693 = v692;
  v694 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 114);
  v695 = *(_DWORD *)a1;
  v696 = v686;
  v1474[1] = v697;
  v698 = 0;
  v1474[0] = v694;
  sub_12D6090(a1 + 2296, v696, v693, v1474, v695);
  v699 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x73u, "0");
  v700 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 115);
  v701 = *(_DWORD *)a1;
  v1475[1] = v702;
  v1475[0] = v700;
  sub_12D6100(a1 + 2320, v699, v1475, v701);
  v703 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x74u);
  if ( v703 )
  {
    if ( *(_DWORD *)(v703 + 56) )
      v698 = **(_QWORD **)(v703 + 48);
    v704 = *(_DWORD *)(v703 + 40);
  }
  else
  {
    v704 = 0;
  }
  v705 = v704;
  v706 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 116);
  v707 = *(_DWORD *)a1;
  v708 = v698;
  v1476[1] = v709;
  v710 = 0;
  v1476[0] = v706;
  sub_12D6090(a1 + 2336, v708, v705, v1476, v707);
  v711 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x75u, "1");
  v712 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 117);
  v713 = *(_DWORD *)a1;
  v1477[1] = v714;
  v1477[0] = v712;
  sub_12D6100(a1 + 2360, v711, v1477, v713);
  v715 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x76u);
  if ( v715 )
  {
    if ( *(_DWORD *)(v715 + 56) )
      v710 = **(_QWORD **)(v715 + 48);
    v716 = *(_DWORD *)(v715 + 40);
  }
  else
  {
    v716 = 0;
  }
  v717 = v716;
  v718 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 118);
  v719 = *(_DWORD *)a1;
  v1478[1] = v720;
  v1478[0] = v718;
  sub_12D6090(a1 + 2376, v710, v717, v1478, v719);
  v721 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x77u, "0");
  v722 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 119);
  *(_BYTE *)(a1 + 2400) = v721;
  v724 = v722;
  v725 = v723;
  LODWORD(v722) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v724 + 36) == 0;
  *(_DWORD *)(a1 + 2404) = HIDWORD(v721);
  *(_DWORD *)(a1 + 2408) = v722;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 2412) = *(_DWORD *)(v724 + 32);
  }
  else if ( sub_1691920(v723, *(unsigned __int16 *)(v724 + 40)) )
  {
    *(_DWORD *)(a1 + 2412) = *(_DWORD *)(sub_1691920(v725, *(unsigned __int16 *)(v724 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 2412) = 0;
  }
  v726 = 0;
  v727 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x78u);
  if ( v727 )
  {
    if ( *(_DWORD *)(v727 + 56) )
      v726 = **(_QWORD **)(v727 + 48);
    v728 = *(_DWORD *)(v727 + 40);
  }
  else
  {
    v728 = 0;
  }
  v729 = v728;
  v730 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 120);
  v731 = *(_DWORD *)a1;
  v732 = v726;
  v1479[1] = v733;
  v734 = 0;
  v1479[0] = v730;
  sub_12D6090(a1 + 2416, v732, v729, v1479, v731);
  v735 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x79u, "0");
  v736 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 121);
  v737 = *(_DWORD *)a1;
  v1480[1] = v738;
  v1480[0] = v736;
  sub_12D6100(a1 + 2440, v735, v1480, v737);
  v739 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x7Au);
  if ( v739 )
  {
    if ( *(_DWORD *)(v739 + 56) )
      v734 = **(_QWORD **)(v739 + 48);
    v740 = *(_DWORD *)(v739 + 40);
  }
  else
  {
    v740 = 0;
  }
  v741 = v740;
  v742 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 122);
  v743 = *(_DWORD *)a1;
  v744 = v734;
  v1481[1] = v745;
  v746 = 0;
  v1481[0] = v742;
  sub_12D6090(a1 + 2456, v744, v741, v1481, v743);
  v747 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x7Bu, "0");
  v748 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 123);
  v749 = *(_DWORD *)a1;
  v1482[1] = v750;
  v1482[0] = v748;
  sub_12D6100(a1 + 2480, v747, v1482, v749);
  v751 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x7Cu);
  if ( v751 )
  {
    if ( *(_DWORD *)(v751 + 56) )
      v746 = **(_QWORD **)(v751 + 48);
    v752 = *(_DWORD *)(v751 + 40);
  }
  else
  {
    v752 = 0;
  }
  v753 = v752;
  v754 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 124);
  v755 = *(_DWORD *)a1;
  v756 = v746;
  v1483[1] = v757;
  v758 = 0;
  v1483[0] = v754;
  sub_12D6090(a1 + 2496, v756, v753, v1483, v755);
  v759 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x7Du, "0");
  v760 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 125);
  v761 = *(_DWORD *)a1;
  v1484[1] = v762;
  v1484[0] = v760;
  sub_12D6100(a1 + 2520, v759, v1484, v761);
  v763 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x7Eu);
  if ( v763 )
  {
    if ( *(_DWORD *)(v763 + 56) )
      v758 = **(_QWORD **)(v763 + 48);
    v764 = *(_DWORD *)(v763 + 40);
  }
  else
  {
    v764 = 0;
  }
  v765 = v764;
  v766 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 126);
  v767 = *(_DWORD *)a1;
  v1485[1] = v768;
  v1485[0] = v766;
  sub_12D6090(a1 + 2536, v758, v765, v1485, v767);
  v769 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x7Fu, "0");
  v770 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 127);
  *(_BYTE *)(a1 + 2560) = v769;
  v772 = v770;
  v773 = v771;
  LODWORD(v770) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v772 + 36) == 0;
  *(_DWORD *)(a1 + 2564) = HIDWORD(v769);
  *(_DWORD *)(a1 + 2568) = v770;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 2572) = *(_DWORD *)(v772 + 32);
  }
  else if ( sub_1691920(v771, *(unsigned __int16 *)(v772 + 40)) )
  {
    *(_DWORD *)(a1 + 2572) = *(_DWORD *)(sub_1691920(v773, *(unsigned __int16 *)(v772 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 2572) = 0;
  }
  v774 = 0;
  v775 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x80u);
  if ( v775 )
  {
    if ( *(_DWORD *)(v775 + 56) )
      v774 = **(_QWORD **)(v775 + 48);
    v776 = *(_DWORD *)(v775 + 40);
  }
  else
  {
    v776 = 0;
  }
  v777 = v776;
  v778 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 128);
  v779 = *(_DWORD *)a1;
  v780 = v774;
  v1486[1] = v781;
  v782 = 0;
  v1486[0] = v778;
  sub_12D6090(a1 + 2576, v780, v777, v1486, v779);
  v783 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x81u, "0");
  v784 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 129);
  v785 = *(_DWORD *)a1;
  v1487[1] = v786;
  v1487[0] = v784;
  sub_12D6100(a1 + 2600, v783, v1487, v785);
  v787 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x82u);
  if ( v787 )
  {
    if ( *(_DWORD *)(v787 + 56) )
      v782 = **(_QWORD **)(v787 + 48);
    v788 = *(_DWORD *)(v787 + 40);
  }
  else
  {
    v788 = 0;
  }
  v789 = v788;
  v790 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 130);
  v791 = *(_DWORD *)a1;
  v792 = v782;
  v1488[1] = v793;
  v794 = 0;
  v1488[0] = v790;
  sub_12D6090(a1 + 2616, v792, v789, v1488, v791);
  v795 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x83u, "0");
  v796 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 131);
  v797 = *(_DWORD *)a1;
  v1489[1] = v798;
  v1489[0] = v796;
  sub_12D6100(a1 + 2640, v795, v1489, v797);
  v799 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x84u);
  if ( v799 )
  {
    if ( *(_DWORD *)(v799 + 56) )
      v794 = **(_QWORD **)(v799 + 48);
    v800 = *(_DWORD *)(v799 + 40);
  }
  else
  {
    v800 = 0;
  }
  v801 = v800;
  v802 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 132);
  v803 = *(_DWORD *)a1;
  v804 = v794;
  v1490[1] = v805;
  v806 = 0;
  v1490[0] = v802;
  sub_12D6090(a1 + 2656, v804, v801, v1490, v803);
  v807 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x85u, "0");
  v808 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 133);
  v809 = *(_DWORD *)a1;
  v1491[1] = v810;
  v1491[0] = v808;
  sub_12D6100(a1 + 2680, v807, v1491, v809);
  v811 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x86u);
  if ( v811 )
  {
    if ( *(_DWORD *)(v811 + 56) )
      v806 = **(_QWORD **)(v811 + 48);
    v812 = *(_DWORD *)(v811 + 40);
  }
  else
  {
    v812 = 0;
  }
  v813 = v812;
  v814 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 134);
  v815 = *(_DWORD *)a1;
  v816 = v806;
  v1492[1] = v817;
  v818 = 0;
  v1492[0] = v814;
  sub_12D6090(a1 + 2696, v816, v813, v1492, v815);
  v819 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x87u, "0");
  v820 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 135);
  v821 = *(_DWORD *)a1;
  v1493[1] = v822;
  v1493[0] = v820;
  sub_12D6100(a1 + 2720, v819, v1493, v821);
  v823 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x88u);
  if ( v823 )
  {
    if ( *(_DWORD *)(v823 + 56) )
      v818 = **(_QWORD **)(v823 + 48);
    v824 = *(_DWORD *)(v823 + 40);
  }
  else
  {
    v824 = 0;
  }
  v825 = v824;
  v826 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 136);
  v827 = *(_DWORD *)a1;
  v828 = v818;
  v1494[1] = v829;
  v830 = 0;
  v1494[0] = v826;
  sub_12D6090(a1 + 2736, v828, v825, v1494, v827);
  v831 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x89u, "0");
  v832 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 137);
  v833 = *(_DWORD *)a1;
  v1495[1] = v834;
  v1495[0] = v832;
  sub_12D6100(a1 + 2760, v831, v1495, v833);
  v835 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x8Au);
  if ( v835 )
  {
    if ( *(_DWORD *)(v835 + 56) )
      v830 = **(_QWORD **)(v835 + 48);
    v836 = *(_DWORD *)(v835 + 40);
  }
  else
  {
    v836 = 0;
  }
  v837 = v836;
  v838 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 138);
  v839 = *(_DWORD *)a1;
  v840 = v830;
  v1496[1] = v841;
  v842 = 0;
  v1496[0] = v838;
  sub_12D6090(a1 + 2776, v840, v837, v1496, v839);
  v843 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x8Bu, "0");
  v844 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 139);
  v845 = *(_DWORD *)a1;
  v1497[1] = v846;
  v1497[0] = v844;
  sub_12D6100(a1 + 2800, v843, v1497, v845);
  v847 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x8Cu);
  if ( v847 )
  {
    if ( *(_DWORD *)(v847 + 56) )
      v842 = **(_QWORD **)(v847 + 48);
    v848 = *(_DWORD *)(v847 + 40);
  }
  else
  {
    v848 = 0;
  }
  v849 = v848;
  v850 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 140);
  v851 = *(_DWORD *)a1;
  v852 = v842;
  v1498[1] = v853;
  v854 = 0;
  v1498[0] = v850;
  sub_12D6090(a1 + 2816, v852, v849, v1498, v851);
  v855 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x8Du, "1");
  v856 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 141);
  v857 = *(_DWORD *)a1;
  v1499[1] = v858;
  v1499[0] = v856;
  sub_12D6100(a1 + 2840, v855, v1499, v857);
  v859 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x8Eu);
  if ( v859 )
  {
    if ( *(_DWORD *)(v859 + 56) )
      v854 = **(_QWORD **)(v859 + 48);
    v860 = *(_DWORD *)(v859 + 40);
  }
  else
  {
    v860 = 0;
  }
  v861 = v860;
  v862 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 142);
  v863 = *(_DWORD *)a1;
  v864 = v854;
  v1500[1] = v865;
  v866 = 0;
  v1500[0] = v862;
  sub_12D6090(a1 + 2856, v864, v861, v1500, v863);
  v867 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x8Fu, "1");
  v868 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 143);
  v869 = *(_DWORD *)a1;
  v1501[1] = v870;
  v1501[0] = v868;
  sub_12D6100(a1 + 2880, v867, v1501, v869);
  v871 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x90u);
  if ( v871 )
  {
    if ( *(_DWORD *)(v871 + 56) )
      v866 = **(_QWORD **)(v871 + 48);
    v872 = *(_DWORD *)(v871 + 40);
  }
  else
  {
    v872 = 0;
  }
  v873 = v872;
  v874 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 144);
  v875 = *(_DWORD *)a1;
  v876 = v866;
  v1502[1] = v877;
  v878 = 0;
  v1502[0] = v874;
  sub_12D6090(a1 + 2896, v876, v873, v1502, v875);
  v879 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x91u, "0");
  v880 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 145);
  v881 = *(_DWORD *)a1;
  v1503[1] = v882;
  v1503[0] = v880;
  sub_12D6100(a1 + 2920, v879, v1503, v881);
  v883 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x92u);
  if ( v883 )
  {
    if ( *(_DWORD *)(v883 + 56) )
      v878 = **(_QWORD **)(v883 + 48);
    v884 = *(_DWORD *)(v883 + 40);
  }
  else
  {
    v884 = 0;
  }
  v885 = v884;
  v886 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 146);
  v887 = *(_DWORD *)a1;
  v888 = v878;
  v1504[1] = v889;
  v890 = 0;
  v1504[0] = v886;
  sub_12D6090(a1 + 2936, v888, v885, v1504, v887);
  v891 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x93u, "0");
  v892 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 147);
  v893 = *(_DWORD *)a1;
  v1505[1] = v894;
  v1505[0] = v892;
  sub_12D6100(a1 + 2960, v891, v1505, v893);
  v895 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x94u);
  if ( v895 )
  {
    if ( *(_DWORD *)(v895 + 56) )
      v890 = **(_QWORD **)(v895 + 48);
    v896 = *(_DWORD *)(v895 + 40);
  }
  else
  {
    v896 = 0;
  }
  v897 = v896;
  v898 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 148);
  v899 = *(_DWORD *)a1;
  v900 = v890;
  v1506[1] = v901;
  v902 = 0;
  v1506[0] = v898;
  sub_12D6090(a1 + 2976, v900, v897, v1506, v899);
  v903 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x95u, "0");
  v904 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 149);
  v905 = *(_DWORD *)a1;
  v1507[1] = v906;
  v1507[0] = v904;
  sub_12D6100(a1 + 3000, v903, v1507, v905);
  v907 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x96u);
  if ( v907 )
  {
    if ( *(_DWORD *)(v907 + 56) )
      v902 = **(_QWORD **)(v907 + 48);
    v908 = *(_DWORD *)(v907 + 40);
  }
  else
  {
    v908 = 0;
  }
  v909 = v908;
  v910 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 150);
  v911 = *(_DWORD *)a1;
  v1508[1] = v912;
  v1508[0] = v910;
  sub_12D6090(a1 + 3016, v902, v909, v1508, v911);
  v913 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x97u, "1");
  v914 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 151);
  *(_BYTE *)(a1 + 3040) = v913;
  v916 = v914;
  v917 = v915;
  LODWORD(v914) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v916 + 36) == 0;
  *(_DWORD *)(a1 + 3044) = HIDWORD(v913);
  *(_DWORD *)(a1 + 3048) = v914;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 3052) = *(_DWORD *)(v916 + 32);
  }
  else if ( sub_1691920(v915, *(unsigned __int16 *)(v916 + 40)) )
  {
    *(_DWORD *)(a1 + 3052) = *(_DWORD *)(sub_1691920(v917, *(unsigned __int16 *)(v916 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 3052) = 0;
  }
  v918 = 0;
  v919 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x98u);
  if ( v919 )
  {
    if ( *(_DWORD *)(v919 + 56) )
      v918 = **(_QWORD **)(v919 + 48);
    v920 = *(_DWORD *)(v919 + 40);
  }
  else
  {
    v920 = 0;
  }
  v921 = v920;
  v922 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 152);
  v923 = *(_DWORD *)a1;
  v924 = v918;
  v1509[1] = v925;
  v926 = 0;
  v1509[0] = v922;
  sub_12D6090(a1 + 3056, v924, v921, v1509, v923);
  v927 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x99u, "0");
  v928 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 153);
  v929 = *(_DWORD *)a1;
  v1510[1] = v930;
  v1510[0] = v928;
  sub_12D6100(a1 + 3080, v927, v1510, v929);
  v931 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x9Au);
  if ( v931 )
  {
    if ( *(_DWORD *)(v931 + 56) )
      v926 = **(_QWORD **)(v931 + 48);
    v932 = *(_DWORD *)(v931 + 40);
  }
  else
  {
    v932 = 0;
  }
  v933 = v932;
  v934 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 154);
  v935 = *(_DWORD *)a1;
  v936 = v926;
  v1511[1] = v937;
  v938 = 0;
  v1511[0] = v934;
  sub_12D6090(a1 + 3096, v936, v933, v1511, v935);
  v939 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x9Bu, "1");
  v940 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 155);
  v941 = *(_DWORD *)a1;
  v1512[1] = v942;
  v1512[0] = v940;
  sub_12D6100(a1 + 3120, v939, v1512, v941);
  v943 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x9Cu);
  if ( v943 )
  {
    if ( *(_DWORD *)(v943 + 56) )
      v938 = **(_QWORD **)(v943 + 48);
    v944 = *(_DWORD *)(v943 + 40);
  }
  else
  {
    v944 = 0;
  }
  v945 = v944;
  v946 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 156);
  v947 = *(_DWORD *)a1;
  v948 = v938;
  v1513[1] = v949;
  v950 = 0;
  v1513[0] = v946;
  sub_12D6090(a1 + 3136, v948, v945, v1513, v947);
  v951 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x9Du, "1");
  v952 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 157);
  v953 = *(_DWORD *)a1;
  v1514[1] = v954;
  v1514[0] = v952;
  sub_12D6100(a1 + 3160, v951, v1514, v953);
  v955 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0x9Eu);
  if ( v955 )
  {
    if ( *(_DWORD *)(v955 + 56) )
      v950 = **(_QWORD **)(v955 + 48);
    v956 = *(_DWORD *)(v955 + 40);
  }
  else
  {
    v956 = 0;
  }
  v957 = v956;
  v958 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 158);
  v959 = *(_DWORD *)a1;
  v1515[1] = v960;
  v1515[0] = v958;
  sub_12D6090(a1 + 3176, v950, v957, v1515, v959);
  v961 = sub_12D6240(*(_QWORD *)(a1 + 8), 0x9Fu, "1");
  v962 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 159);
  *(_BYTE *)(a1 + 3200) = v961;
  v964 = v962;
  v965 = v963;
  LODWORD(v962) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v964 + 36) == 0;
  *(_DWORD *)(a1 + 3204) = HIDWORD(v961);
  *(_DWORD *)(a1 + 3208) = v962;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 3212) = *(_DWORD *)(v964 + 32);
  }
  else if ( sub_1691920(v963, *(unsigned __int16 *)(v964 + 40)) )
  {
    *(_DWORD *)(a1 + 3212) = *(_DWORD *)(sub_1691920(v965, *(unsigned __int16 *)(v964 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 3212) = 0;
  }
  v966 = 0;
  v967 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xA0u);
  if ( v967 )
  {
    if ( *(_DWORD *)(v967 + 56) )
      v966 = **(_QWORD **)(v967 + 48);
    v968 = *(_DWORD *)(v967 + 40);
  }
  else
  {
    v968 = 0;
  }
  v969 = v968;
  v970 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 160);
  v971 = *(_DWORD *)a1;
  v972 = v966;
  v1516[1] = v973;
  v974 = 0;
  v1516[0] = v970;
  sub_12D6090(a1 + 3216, v972, v969, v1516, v971);
  v975 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xA1u);
  if ( v975 )
  {
    if ( *(_DWORD *)(v975 + 56) )
      v974 = **(_QWORD **)(v975 + 48);
    v976 = *(_DWORD *)(v975 + 40);
  }
  else
  {
    v976 = 0;
  }
  v977 = v976;
  v978 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 161);
  v979 = *(_DWORD *)a1;
  v980 = v974;
  v1517[1] = v981;
  v982 = 0;
  v1517[0] = v978;
  sub_12D6090(a1 + 3240, v980, v977, v1517, v979);
  v983 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xA2u);
  if ( v983 )
  {
    if ( *(_DWORD *)(v983 + 56) )
      v982 = **(_QWORD **)(v983 + 48);
    v984 = *(_DWORD *)(v983 + 40);
  }
  else
  {
    v984 = 0;
  }
  v985 = v984;
  v986 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 162);
  v987 = *(_DWORD *)a1;
  v988 = v982;
  v1518[1] = v989;
  v990 = 0;
  v1518[0] = v986;
  sub_12D6090(a1 + 3264, v988, v985, v1518, v987);
  v991 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xA3u, "0");
  v992 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 163);
  v993 = *(_DWORD *)a1;
  v1519[1] = v994;
  v1519[0] = v992;
  sub_12D6100(a1 + 3288, v991, v1519, v993);
  v995 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xA4u);
  if ( v995 )
  {
    if ( *(_DWORD *)(v995 + 56) )
      v990 = **(_QWORD **)(v995 + 48);
    v996 = *(_DWORD *)(v995 + 40);
  }
  else
  {
    v996 = 0;
  }
  v997 = v996;
  v998 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 164);
  v999 = *(_DWORD *)a1;
  v1000 = v990;
  v1520[1] = v1001;
  v1002 = 0;
  v1520[0] = v998;
  sub_12D6090(a1 + 3304, v1000, v997, v1520, v999);
  v1003 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xA5u, "1");
  v1004 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 165);
  v1005 = *(_DWORD *)a1;
  v1521[1] = v1006;
  v1521[0] = v1004;
  sub_12D6100(a1 + 3328, v1003, v1521, v1005);
  v1007 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xA6u);
  if ( v1007 )
  {
    if ( *(_DWORD *)(v1007 + 56) )
      v1002 = **(_QWORD **)(v1007 + 48);
    v1008 = *(_DWORD *)(v1007 + 40);
  }
  else
  {
    v1008 = 0;
  }
  v1009 = v1008;
  v1010 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 166);
  v1011 = *(_DWORD *)a1;
  v1012 = v1002;
  v1522[1] = v1013;
  v1014 = 0;
  v1522[0] = v1010;
  sub_12D6090(a1 + 3344, v1012, v1009, v1522, v1011);
  v1015 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xA7u, "0");
  v1016 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 167);
  v1017 = *(_DWORD *)a1;
  v1523[1] = v1018;
  v1523[0] = v1016;
  sub_12D6100(a1 + 3368, v1015, v1523, v1017);
  v1019 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xA8u);
  if ( v1019 )
  {
    if ( *(_DWORD *)(v1019 + 56) )
      v1014 = **(_QWORD **)(v1019 + 48);
    v1020 = *(_DWORD *)(v1019 + 40);
  }
  else
  {
    v1020 = 0;
  }
  v1021 = v1020;
  v1022 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 168);
  v1023 = *(_DWORD *)a1;
  v1524[1] = v1024;
  v1524[0] = v1022;
  sub_12D6090(a1 + 3384, v1014, v1021, v1524, v1023);
  v1025 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xA9u, "0");
  v1026 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 169);
  *(_BYTE *)(a1 + 3408) = v1025;
  v1028 = v1026;
  v1029 = v1027;
  LODWORD(v1026) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v1028 + 36) == 0;
  *(_DWORD *)(a1 + 3412) = HIDWORD(v1025);
  *(_DWORD *)(a1 + 3416) = v1026;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 3420) = *(_DWORD *)(v1028 + 32);
  }
  else if ( sub_1691920(v1027, *(unsigned __int16 *)(v1028 + 40)) )
  {
    *(_DWORD *)(a1 + 3420) = *(_DWORD *)(sub_1691920(v1029, *(unsigned __int16 *)(v1028 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 3420) = 0;
  }
  v1030 = 0;
  v1031 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xAAu);
  if ( v1031 )
  {
    if ( *(_DWORD *)(v1031 + 56) )
      v1030 = **(_QWORD **)(v1031 + 48);
    v1032 = *(_DWORD *)(v1031 + 40);
  }
  else
  {
    v1032 = 0;
  }
  v1033 = v1032;
  v1034 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 170);
  v1035 = *(_DWORD *)a1;
  v1036 = v1030;
  v1525[1] = v1037;
  v1038 = 0;
  v1525[0] = v1034;
  sub_12D6090(a1 + 3424, v1036, v1033, v1525, v1035);
  v1039 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xABu, "0");
  v1040 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 171);
  v1041 = *(_DWORD *)a1;
  v1526[1] = v1042;
  v1526[0] = v1040;
  sub_12D6100(a1 + 3448, v1039, v1526, v1041);
  v1043 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xACu);
  if ( v1043 )
  {
    if ( *(_DWORD *)(v1043 + 56) )
      v1038 = **(_QWORD **)(v1043 + 48);
    v1044 = *(_DWORD *)(v1043 + 40);
  }
  else
  {
    v1044 = 0;
  }
  v1045 = v1044;
  v1046 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 172);
  v1047 = *(_DWORD *)a1;
  v1048 = v1038;
  v1527[1] = v1049;
  v1050 = 0;
  v1527[0] = v1046;
  sub_12D6090(a1 + 3464, v1048, v1045, v1527, v1047);
  v1051 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xADu, "0");
  v1052 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 173);
  v1053 = *(_DWORD *)a1;
  v1528[1] = v1054;
  v1528[0] = v1052;
  sub_12D6100(a1 + 3488, v1051, v1528, v1053);
  v1055 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xAEu);
  if ( v1055 )
  {
    if ( *(_DWORD *)(v1055 + 56) )
      v1050 = **(_QWORD **)(v1055 + 48);
    v1056 = *(_DWORD *)(v1055 + 40);
  }
  else
  {
    v1056 = 0;
  }
  v1057 = v1056;
  v1058 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 174);
  v1059 = *(_DWORD *)a1;
  v1060 = v1050;
  v1529[1] = v1061;
  v1062 = 0;
  v1529[0] = v1058;
  sub_12D6090(a1 + 3504, v1060, v1057, v1529, v1059);
  v1063 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xAFu, "0");
  v1064 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 175);
  v1065 = *(_DWORD *)a1;
  v1530[1] = v1066;
  v1530[0] = v1064;
  sub_12D6100(a1 + 3528, v1063, v1530, v1065);
  v1067 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xB0u);
  if ( v1067 )
  {
    if ( *(_DWORD *)(v1067 + 56) )
      v1062 = **(_QWORD **)(v1067 + 48);
    v1068 = *(_DWORD *)(v1067 + 40);
  }
  else
  {
    v1068 = 0;
  }
  v1069 = v1068;
  v1070 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 176);
  v1071 = *(_DWORD *)a1;
  v1531[1] = v1072;
  v1531[0] = v1070;
  sub_12D6090(a1 + 3544, v1062, v1069, v1531, v1071);
  v1073 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xB1u, "0");
  v1074 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 177);
  *(_BYTE *)(a1 + 3568) = v1073;
  v1076 = v1074;
  v1077 = v1075;
  LODWORD(v1074) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v1076 + 36) == 0;
  *(_DWORD *)(a1 + 3572) = HIDWORD(v1073);
  *(_DWORD *)(a1 + 3576) = v1074;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 3580) = *(_DWORD *)(v1076 + 32);
  }
  else if ( sub_1691920(v1075, *(unsigned __int16 *)(v1076 + 40)) )
  {
    *(_DWORD *)(a1 + 3580) = *(_DWORD *)(sub_1691920(v1077, *(unsigned __int16 *)(v1076 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 3580) = 0;
  }
  v1078 = 0;
  v1079 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xB2u);
  if ( v1079 )
  {
    if ( *(_DWORD *)(v1079 + 56) )
      v1078 = **(_QWORD **)(v1079 + 48);
    v1080 = *(_DWORD *)(v1079 + 40);
  }
  else
  {
    v1080 = 0;
  }
  v1081 = v1080;
  v1082 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 178);
  v1083 = *(_DWORD *)a1;
  v1084 = v1078;
  v1532[1] = v1085;
  v1086 = 0;
  v1532[0] = v1082;
  sub_12D6090(a1 + 3584, v1084, v1081, v1532, v1083);
  v1087 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xB3u, "0");
  v1088 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 179);
  v1089 = *(_DWORD *)a1;
  v1533[1] = v1090;
  v1533[0] = v1088;
  sub_12D6100(a1 + 3608, v1087, v1533, v1089);
  v1091 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xB4u);
  if ( v1091 )
  {
    if ( *(_DWORD *)(v1091 + 56) )
      v1086 = **(_QWORD **)(v1091 + 48);
    v1092 = *(_DWORD *)(v1091 + 40);
  }
  else
  {
    v1092 = 0;
  }
  v1093 = v1092;
  v1094 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 180);
  v1095 = *(_DWORD *)a1;
  v1534[1] = v1096;
  v1534[0] = v1094;
  sub_12D6090(a1 + 3624, v1086, v1093, v1534, v1095);
  v1097 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xB5u);
  v1098 = v1097;
  if ( v1097 )
  {
    v1099 = 0;
    v1100 = byte_3F871B3;
    if ( *(_DWORD *)(v1097 + 56) )
    {
      v1355 = *(const char ***)(v1097 + 48);
      v1100 = *v1355;
      if ( *v1355 )
        v1099 = strlen(*v1355);
    }
    v1101 = *(_DWORD *)(v1098 + 40);
  }
  else
  {
    v1100 = byte_3F871B3;
    v1099 = 0;
    v1101 = 0;
  }
  v1358 = v1099;
  v1102 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 181);
  *(_QWORD *)(a1 + 3648) = v1100;
  v1104 = v1102;
  LODWORD(v1102) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 3664) = v1101;
  *(_QWORD *)(a1 + 3656) = v1358;
  v50 = *(_BYTE *)(v1104 + 36) == 0;
  *(_DWORD *)(a1 + 3668) = v1102;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 3672) = *(_DWORD *)(v1104 + 32);
  }
  else
  {
    v1368 = v1103;
    if ( sub_1691920(v1103, *(unsigned __int16 *)(v1104 + 40)) )
      *(_DWORD *)(a1 + 3672) = *(_DWORD *)(sub_1691920(v1368, *(unsigned __int16 *)(v1104 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 3672) = 0;
  }
  v1105 = 0;
  v1106 = 0;
  v1107 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xB6u);
  if ( v1107 )
  {
    if ( *(_DWORD *)(v1107 + 56) )
      v1105 = **(_QWORD **)(v1107 + 48);
    v1106 = *(_DWORD *)(v1107 + 40);
  }
  v1108 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 182);
  v1109 = *(_DWORD *)a1;
  v1110 = v1105;
  v1535[1] = v1111;
  LODWORD(v1111) = v1106;
  v1112 = 0;
  v1535[0] = v1108;
  v1113 = 0;
  sub_12D6090(a1 + 3680, v1110, v1111, v1535, v1109);
  v1114 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xB7u, "0");
  v1115 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 183);
  v1116 = *(_DWORD *)a1;
  v1536[1] = v1117;
  v1536[0] = v1115;
  sub_12D6100(a1 + 3704, v1114, v1536, v1116);
  v1118 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xB8u);
  if ( v1118 )
  {
    if ( *(_DWORD *)(v1118 + 56) )
      v1112 = **(_QWORD **)(v1118 + 48);
    v1113 = *(_DWORD *)(v1118 + 40);
  }
  v1119 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 184);
  v1120 = *(_DWORD *)a1;
  v1121 = v1112;
  v1537[1] = v1122;
  LODWORD(v1122) = v1113;
  v1123 = 0;
  v1537[0] = v1119;
  v1124 = 0;
  sub_12D6090(a1 + 3720, v1121, v1122, v1537, v1120);
  v1125 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xB9u, "0");
  v1126 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 185);
  v1127 = *(_DWORD *)a1;
  v1538[1] = v1128;
  v1538[0] = v1126;
  sub_12D6100(a1 + 3744, v1125, v1538, v1127);
  v1129 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xBAu);
  if ( v1129 )
  {
    if ( *(_DWORD *)(v1129 + 56) )
      v1123 = **(_QWORD **)(v1129 + 48);
    v1124 = *(_DWORD *)(v1129 + 40);
  }
  v1130 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 186);
  v1131 = *(_DWORD *)a1;
  v1132 = v1123;
  v1539[1] = v1133;
  LODWORD(v1133) = v1124;
  v1134 = 0;
  v1539[0] = v1130;
  v1135 = 0;
  sub_12D6090(a1 + 3760, v1132, v1133, v1539, v1131);
  v1136 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xBBu, "0");
  v1137 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 187);
  v1138 = *(_DWORD *)a1;
  v1540[1] = v1139;
  v1540[0] = v1137;
  sub_12D6100(a1 + 3784, v1136, v1540, v1138);
  v1140 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xBCu);
  if ( v1140 )
  {
    if ( *(_DWORD *)(v1140 + 56) )
      v1134 = **(_QWORD **)(v1140 + 48);
    v1135 = *(_DWORD *)(v1140 + 40);
  }
  v1141 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 188);
  v1142 = *(_DWORD *)a1;
  v1143 = v1134;
  v1541[1] = v1144;
  LODWORD(v1144) = v1135;
  v1145 = 0;
  v1541[0] = v1141;
  v1146 = 0;
  sub_12D6090(a1 + 3800, v1143, v1144, v1541, v1142);
  v1147 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xBDu, "0");
  v1148 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 189);
  v1149 = *(_DWORD *)a1;
  v1542[1] = v1150;
  v1542[0] = v1148;
  sub_12D6100(a1 + 3824, v1147, v1542, v1149);
  v1151 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xBEu);
  if ( v1151 )
  {
    if ( *(_DWORD *)(v1151 + 56) )
      v1145 = **(_QWORD **)(v1151 + 48);
    v1146 = *(_DWORD *)(v1151 + 40);
  }
  v1152 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 190);
  v1153 = *(_DWORD *)a1;
  v1154 = v1145;
  v1543[1] = v1155;
  LODWORD(v1155) = v1146;
  v1156 = 0;
  v1543[0] = v1152;
  v1157 = 0;
  sub_12D6090(a1 + 3840, v1154, v1155, v1543, v1153);
  v1158 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xBFu);
  if ( v1158 )
  {
    if ( *(_DWORD *)(v1158 + 56) )
      v1156 = **(_QWORD **)(v1158 + 48);
    v1157 = *(_DWORD *)(v1158 + 40);
  }
  v1159 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 191);
  v1160 = *(_DWORD *)a1;
  v1161 = v1156;
  v1544[1] = v1162;
  LODWORD(v1162) = v1157;
  v1163 = 0;
  v1544[0] = v1159;
  v1164 = 0;
  sub_12D6090(a1 + 3864, v1161, v1162, v1544, v1160);
  v1165 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xC0u, "0");
  v1166 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 192);
  v1167 = *(_DWORD *)a1;
  v1545[1] = v1168;
  v1545[0] = v1166;
  sub_12D6100(a1 + 3888, v1165, v1545, v1167);
  v1169 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xC1u, "0");
  v1170 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 193);
  v1171 = *(_DWORD *)a1;
  v1546[1] = v1172;
  v1546[0] = v1170;
  sub_12D6100(a1 + 3904, v1169, v1546, v1171);
  v1173 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xC2u);
  if ( v1173 )
  {
    if ( *(_DWORD *)(v1173 + 56) )
      v1163 = **(_QWORD **)(v1173 + 48);
    v1164 = *(_DWORD *)(v1173 + 40);
  }
  v1174 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 194);
  v1175 = *(_DWORD *)a1;
  v1176 = v1163;
  v1547[1] = v1177;
  LODWORD(v1177) = v1164;
  v1178 = 0;
  v1547[0] = v1174;
  v1179 = 0;
  sub_12D6090(a1 + 3920, v1176, v1177, v1547, v1175);
  v1180 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xC3u, "0");
  v1181 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 195);
  v1182 = *(_DWORD *)a1;
  v1548[1] = v1183;
  v1548[0] = v1181;
  sub_12D6100(a1 + 3944, v1180, v1548, v1182);
  v1184 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xC4u);
  if ( v1184 )
  {
    if ( *(_DWORD *)(v1184 + 56) )
      v1178 = **(_QWORD **)(v1184 + 48);
    v1179 = *(_DWORD *)(v1184 + 40);
  }
  v1185 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 196);
  v1186 = *(_DWORD *)a1;
  v1549[1] = v1187;
  v1549[0] = v1185;
  sub_12D6090(a1 + 3960, v1178, v1179, v1549, v1186);
  v1188 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xC5u);
  v1189 = 2;
  v1190 = "20";
  v1191 = v1188;
  if ( v1188 )
  {
    if ( *(_DWORD *)(v1188 + 56) )
    {
      v1353 = *(const char ***)(v1188 + 48);
      v1189 = 0;
      v1190 = *v1353;
      if ( *v1353 )
      {
        v1363 = *v1353;
        v1354 = strlen(v1190);
        v1190 = v1363;
        v1189 = v1354;
      }
    }
  }
  if ( (unsigned __int8)sub_16D2BB0(v1190, v1189, 0, v1568) || (v1192 = v1568[0], v1568[0] != SLODWORD(v1568[0])) )
    v1192 = 0;
  v1193 = 0;
  if ( v1191 )
    v1193 = *(_DWORD *)(v1191 + 40);
  v1194 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 197);
  *(_DWORD *)(a1 + 3984) = v1192;
  v1196 = v1194;
  LODWORD(v1194) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 3988) = v1193;
  v50 = *(_BYTE *)(v1196 + 36) == 0;
  *(_DWORD *)(a1 + 3992) = v1194;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 3996) = *(_DWORD *)(v1196 + 32);
  }
  else
  {
    v1367 = v1195;
    if ( sub_1691920(v1195, *(unsigned __int16 *)(v1196 + 40)) )
      *(_DWORD *)(a1 + 3996) = *(_DWORD *)(sub_1691920(v1367, *(unsigned __int16 *)(v1196 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 3996) = 0;
  }
  v1197 = 0;
  v1198 = 0;
  v1199 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xC6u);
  if ( v1199 )
  {
    if ( *(_DWORD *)(v1199 + 56) )
      v1197 = **(_QWORD **)(v1199 + 48);
    v1198 = *(_DWORD *)(v1199 + 40);
  }
  v1200 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 198);
  v1201 = *(_DWORD *)a1;
  v1202 = v1197;
  v1550[1] = v1203;
  LODWORD(v1203) = v1198;
  v1204 = 0;
  v1550[0] = v1200;
  v1205 = 0;
  sub_12D6090(a1 + 4000, v1202, v1203, v1550, v1201);
  v1206 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xC7u, "0");
  v1207 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 199);
  v1208 = *(_DWORD *)a1;
  v1551[1] = v1209;
  v1551[0] = v1207;
  sub_12D6100(a1 + 4024, v1206, v1551, v1208);
  v1210 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xC8u);
  if ( v1210 )
  {
    if ( *(_DWORD *)(v1210 + 56) )
      v1204 = **(_QWORD **)(v1210 + 48);
    v1205 = *(_DWORD *)(v1210 + 40);
  }
  v1211 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 200);
  v1212 = *(_DWORD *)a1;
  v1213 = v1204;
  v1552[1] = v1214;
  LODWORD(v1214) = v1205;
  v1215 = 0;
  v1552[0] = v1211;
  v1216 = 0;
  sub_12D6090(a1 + 4040, v1213, v1214, v1552, v1212);
  v1217 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xC9u, "0");
  v1218 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 201);
  v1219 = *(_DWORD *)a1;
  v1553[1] = v1220;
  v1553[0] = v1218;
  sub_12D6100(a1 + 4064, v1217, v1553, v1219);
  v1221 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCAu);
  if ( v1221 )
  {
    if ( *(_DWORD *)(v1221 + 56) )
      v1215 = **(_QWORD **)(v1221 + 48);
    v1216 = *(_DWORD *)(v1221 + 40);
  }
  v1222 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 202);
  v1223 = *(_DWORD *)a1;
  v1554[1] = v1224;
  v1554[0] = v1222;
  sub_12D6090(a1 + 4080, v1215, v1216, v1554, v1223);
  v1225 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCBu);
  v1226 = 2;
  v1227 = "-1";
  v1228 = v1225;
  if ( v1225 )
  {
    if ( *(_DWORD *)(v1225 + 56) )
    {
      v1351 = *(const char ***)(v1225 + 48);
      v1226 = 0;
      v1227 = *v1351;
      if ( *v1351 )
      {
        v1362 = *v1351;
        v1352 = strlen(v1227);
        v1227 = v1362;
        v1226 = v1352;
      }
    }
  }
  if ( (unsigned __int8)sub_16D2BB0(v1227, v1226, 0, v1568) || (v1229 = v1568[0], v1568[0] != SLODWORD(v1568[0])) )
    v1229 = 0;
  v1230 = 0;
  if ( v1228 )
    v1230 = *(_DWORD *)(v1228 + 40);
  v1231 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 203);
  *(_DWORD *)(a1 + 4104) = v1229;
  v1233 = v1231;
  LODWORD(v1231) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 4108) = v1230;
  v50 = *(_BYTE *)(v1233 + 36) == 0;
  *(_DWORD *)(a1 + 4112) = v1231;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 4116) = *(_DWORD *)(v1233 + 32);
  }
  else
  {
    v1366 = v1232;
    if ( sub_1691920(v1232, *(unsigned __int16 *)(v1233 + 40)) )
      *(_DWORD *)(a1 + 4116) = *(_DWORD *)(sub_1691920(v1366, *(unsigned __int16 *)(v1233 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 4116) = 0;
  }
  v1234 = 0;
  v1235 = 0;
  v1236 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCCu);
  if ( v1236 )
  {
    if ( *(_DWORD *)(v1236 + 56) )
      v1234 = **(_QWORD **)(v1236 + 48);
    v1235 = *(_DWORD *)(v1236 + 40);
  }
  v1237 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 204);
  v1238 = *(_DWORD *)a1;
  v1555[1] = v1239;
  v1555[0] = v1237;
  sub_12D6090(a1 + 4120, v1234, v1235, v1555, v1238);
  v1240 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCDu);
  v1241 = 2;
  v1242 = "-1";
  v1243 = v1240;
  if ( v1240 )
  {
    if ( *(_DWORD *)(v1240 + 56) )
    {
      v1349 = *(const char ***)(v1240 + 48);
      v1241 = 0;
      v1242 = *v1349;
      if ( *v1349 )
      {
        v1361 = *v1349;
        v1350 = strlen(v1242);
        v1242 = v1361;
        v1241 = v1350;
      }
    }
  }
  if ( (unsigned __int8)sub_16D2BB0(v1242, v1241, 0, v1568) || (v1244 = v1568[0], v1568[0] != SLODWORD(v1568[0])) )
    v1244 = 0;
  v1245 = 0;
  if ( v1243 )
    v1245 = *(_DWORD *)(v1243 + 40);
  v1246 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 205);
  *(_DWORD *)(a1 + 4144) = v1244;
  v1248 = v1246;
  LODWORD(v1246) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 4148) = v1245;
  v50 = *(_BYTE *)(v1248 + 36) == 0;
  *(_DWORD *)(a1 + 4152) = v1246;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 4156) = *(_DWORD *)(v1248 + 32);
  }
  else
  {
    v1371 = v1247;
    if ( sub_1691920(v1247, *(unsigned __int16 *)(v1248 + 40)) )
      *(_DWORD *)(a1 + 4156) = *(_DWORD *)(sub_1691920(v1371, *(unsigned __int16 *)(v1248 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 4156) = 0;
  }
  v1249 = 0;
  v1250 = 0;
  v1251 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCEu);
  if ( v1251 )
  {
    if ( *(_DWORD *)(v1251 + 56) )
      v1249 = **(_QWORD **)(v1251 + 48);
    v1250 = *(_DWORD *)(v1251 + 40);
  }
  v1252 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 206);
  v1253 = *(_DWORD *)a1;
  v1556[1] = v1254;
  v1556[0] = v1252;
  sub_12D6090(a1 + 4160, v1249, v1250, v1556, v1253);
  v1255 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xCFu);
  v1256 = 2;
  v1257 = "-1";
  v1258 = v1255;
  if ( v1255 )
  {
    if ( *(_DWORD *)(v1255 + 56) )
    {
      v1347 = *(const char ***)(v1255 + 48);
      v1256 = 0;
      v1257 = *v1347;
      if ( *v1347 )
      {
        v1360 = *v1347;
        v1348 = strlen(v1257);
        v1257 = v1360;
        v1256 = v1348;
      }
    }
  }
  if ( (unsigned __int8)sub_16D2BB0(v1257, v1256, 0, v1568) || (v1259 = v1568[0], v1568[0] != SLODWORD(v1568[0])) )
    v1259 = 0;
  v1260 = 0;
  if ( v1258 )
    v1260 = *(_DWORD *)(v1258 + 40);
  v1261 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 207);
  *(_DWORD *)(a1 + 4184) = v1259;
  v1263 = v1261;
  LODWORD(v1261) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 4188) = v1260;
  v50 = *(_BYTE *)(v1263 + 36) == 0;
  *(_DWORD *)(a1 + 4192) = v1261;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 4196) = *(_DWORD *)(v1263 + 32);
  }
  else
  {
    v1370 = v1262;
    if ( sub_1691920(v1262, *(unsigned __int16 *)(v1263 + 40)) )
      *(_DWORD *)(a1 + 4196) = *(_DWORD *)(sub_1691920(v1370, *(unsigned __int16 *)(v1263 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 4196) = 0;
  }
  v1264 = 0;
  v1265 = 0;
  v1266 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xD0u);
  if ( v1266 )
  {
    if ( *(_DWORD *)(v1266 + 56) )
      v1264 = **(_QWORD **)(v1266 + 48);
    v1265 = *(_DWORD *)(v1266 + 40);
  }
  v1267 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 208);
  v1268 = *(_DWORD *)a1;
  v1269 = v1264;
  v1557[1] = v1270;
  LODWORD(v1270) = v1265;
  v1271 = 0;
  v1557[0] = v1267;
  v1272 = 0;
  sub_12D6090(a1 + 4200, v1269, v1270, v1557, v1268);
  v1273 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xD1u, "0");
  v1274 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 209);
  v1275 = *(_DWORD *)a1;
  v1558[1] = v1276;
  v1558[0] = v1274;
  sub_12D6100(a1 + 4224, v1273, v1558, v1275);
  v1277 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xD2u);
  if ( v1277 )
  {
    if ( *(_DWORD *)(v1277 + 56) )
      v1271 = **(_QWORD **)(v1277 + 48);
    v1272 = *(_DWORD *)(v1277 + 40);
  }
  v1278 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 210);
  v1279 = *(_DWORD *)a1;
  v1559[1] = v1280;
  v1559[0] = v1278;
  sub_12D6090(a1 + 4240, v1271, v1272, v1559, v1279);
  v1281 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xD3u, "1");
  v1282 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 211);
  *(_BYTE *)(a1 + 4264) = v1281;
  v1284 = v1282;
  v1285 = v1283;
  LODWORD(v1282) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v1284 + 36) == 0;
  *(_DWORD *)(a1 + 4268) = HIDWORD(v1281);
  *(_DWORD *)(a1 + 4272) = v1282;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 4276) = *(_DWORD *)(v1284 + 32);
  }
  else if ( sub_1691920(v1283, *(unsigned __int16 *)(v1284 + 40)) )
  {
    *(_DWORD *)(a1 + 4276) = *(_DWORD *)(sub_1691920(v1285, *(unsigned __int16 *)(v1284 + 40)) + 32);
  }
  else
  {
    *(_DWORD *)(a1 + 4276) = 0;
  }
  v1286 = 0;
  v1287 = 0;
  v1288 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xD4u);
  if ( v1288 )
  {
    if ( *(_DWORD *)(v1288 + 56) )
      v1286 = **(_QWORD **)(v1288 + 48);
    v1287 = *(_DWORD *)(v1288 + 40);
  }
  v1289 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 212);
  v1290 = *(_DWORD *)a1;
  v1291 = v1286;
  v1560[1] = v1292;
  LODWORD(v1292) = v1287;
  v1293 = 0;
  v1560[0] = v1289;
  v1294 = 0;
  sub_12D6090(a1 + 4280, v1291, v1292, v1560, v1290);
  v1295 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xD5u, "0");
  v1296 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 213);
  v1297 = *(_DWORD *)a1;
  v1561[1] = v1298;
  v1561[0] = v1296;
  sub_12D6100(a1 + 4304, v1295, v1561, v1297);
  v1299 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xD6u);
  if ( v1299 )
  {
    if ( *(_DWORD *)(v1299 + 56) )
      v1293 = **(_QWORD **)(v1299 + 48);
    v1294 = *(_DWORD *)(v1299 + 40);
  }
  v1300 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 214);
  v1301 = *(_DWORD *)a1;
  v1562[1] = v1302;
  v1562[0] = v1300;
  sub_12D6090(a1 + 4320, v1293, v1294, v1562, v1301);
  v1303 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xD7u);
  v1304 = 1;
  v1305 = "0";
  v1306 = v1303;
  if ( v1303 )
  {
    if ( *(_DWORD *)(v1303 + 56) )
    {
      v1345 = *(const char ***)(v1303 + 48);
      v1304 = 0;
      v1305 = *v1345;
      if ( *v1345 )
      {
        v1359 = *v1345;
        v1346 = strlen(v1305);
        v1305 = v1359;
        v1304 = v1346;
      }
    }
  }
  if ( (unsigned __int8)sub_16D2BB0(v1305, v1304, 0, v1568) || (v1307 = v1568[0], v1568[0] != SLODWORD(v1568[0])) )
    v1307 = 0;
  v1308 = 0;
  if ( v1306 )
    v1308 = *(_DWORD *)(v1306 + 40);
  v1309 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 215);
  *(_DWORD *)(a1 + 4344) = v1307;
  v1311 = v1309;
  LODWORD(v1309) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 4348) = v1308;
  v50 = *(_BYTE *)(v1311 + 36) == 0;
  *(_DWORD *)(a1 + 4352) = v1309;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 4356) = *(_DWORD *)(v1311 + 32);
  }
  else
  {
    v1369 = v1310;
    if ( sub_1691920(v1310, *(unsigned __int16 *)(v1311 + 40)) )
      *(_DWORD *)(a1 + 4356) = *(_DWORD *)(sub_1691920(v1369, *(unsigned __int16 *)(v1311 + 40)) + 32);
    else
      *(_DWORD *)(a1 + 4356) = 0;
  }
  v1312 = 0;
  v1313 = 0;
  v1314 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xD8u);
  if ( v1314 )
  {
    if ( *(_DWORD *)(v1314 + 56) )
      v1312 = **(_QWORD **)(v1314 + 48);
    v1313 = *(_DWORD *)(v1314 + 40);
  }
  v1315 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 216);
  v1316 = *(_DWORD *)a1;
  v1317 = v1312;
  v1563[1] = v1318;
  LODWORD(v1318) = v1313;
  v1319 = 0;
  v1563[0] = v1315;
  v1320 = 0;
  sub_12D6090(a1 + 4360, v1317, v1318, v1563, v1316);
  v1321 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xD9u, "0");
  v1322 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 217);
  v1323 = *(_DWORD *)a1;
  v1564[1] = v1324;
  v1564[0] = v1322;
  sub_12D6100(a1 + 4384, v1321, v1564, v1323);
  v1325 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xDAu);
  if ( v1325 )
  {
    if ( *(_DWORD *)(v1325 + 56) )
      v1319 = **(_QWORD **)(v1325 + 48);
    v1320 = *(_DWORD *)(v1325 + 40);
  }
  v1326 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 218);
  v1327 = *(_DWORD *)a1;
  v1328 = v1319;
  v1565[1] = v1329;
  LODWORD(v1329) = v1320;
  v1330 = 0;
  v1565[0] = v1326;
  v1331 = 0;
  sub_12D6090(a1 + 4400, v1328, v1329, v1565, v1327);
  v1332 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xDBu, "1");
  v1333 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 219);
  v1334 = *(_DWORD *)a1;
  v1566[1] = v1335;
  v1566[0] = v1333;
  sub_12D6100(a1 + 4424, v1332, v1566, v1334);
  v1336 = sub_12D6170(*(_QWORD *)(a1 + 8) + 120LL, 0xDCu);
  if ( v1336 )
  {
    if ( *(_DWORD *)(v1336 + 56) )
      v1330 = **(_QWORD **)(v1336 + 48);
    v1331 = *(_DWORD *)(v1336 + 40);
  }
  v1337 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 220);
  v1338 = *(_DWORD *)a1;
  v1567[1] = v1339;
  v1567[0] = v1337;
  sub_12D6090(a1 + 4440, v1330, v1331, v1567, v1338);
  v1340 = sub_12D6240(*(_QWORD *)(a1 + 8), 0xDDu, "0");
  v1341 = sub_1691920(*(_QWORD *)(a1 + 8) + 8LL, 221);
  v1342 = *(_DWORD *)a1;
  v1568[1] = v1343;
  v1568[0] = v1341;
  return sub_12D6100(a1 + 4464, v1340, v1568, v1342);
}
