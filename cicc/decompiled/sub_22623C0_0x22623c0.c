// Function: sub_22623C0
// Address: 0x22623c0
//
__int64 __fastcall sub_22623C0(__int64 a1, __int64 a2)
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
  __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // r14
  __int64 v59; // r12
  __int64 v60; // rax
  int v61; // ecx
  __int64 v62; // rsi
  __int64 v63; // rdx
  int v64; // r12d
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // r8d
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // r12
  __int64 v73; // r14
  __int64 v74; // r14
  int v75; // r12d
  __int64 v76; // rax
  __int64 v77; // rax
  int v78; // r8d
  __int64 v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // r14
  __int64 v82; // r12
  __int64 v83; // rax
  int v84; // ecx
  __int64 v85; // rsi
  __int64 v86; // rdx
  int v87; // r12d
  __int64 v88; // rax
  __int64 v89; // rax
  int v90; // r8d
  __int64 v91; // rdx
  __int64 v92; // r13
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // r12
  __int64 v96; // r14
  __int64 v97; // r14
  int v98; // r12d
  __int64 v99; // rax
  __int64 v100; // rax
  int v101; // r8d
  __int64 v102; // rsi
  __int64 v103; // rdx
  __int64 v104; // r14
  __int64 v105; // r12
  __int64 v106; // rax
  int v107; // ecx
  __int64 v108; // rsi
  __int64 v109; // rdx
  int v110; // r12d
  __int64 v111; // rax
  __int64 v112; // rax
  int v113; // r8d
  __int64 v114; // rdx
  __int64 v115; // r13
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // r12
  __int64 v119; // r14
  __int64 v120; // r14
  int v121; // r12d
  __int64 v122; // rax
  __int64 v123; // rax
  int v124; // r8d
  __int64 v125; // rdx
  __int64 v126; // r13
  __int64 v127; // rax
  __int64 v128; // rdx
  __int64 v129; // r12
  __int64 v130; // r14
  __int64 v131; // r14
  int v132; // r12d
  __int64 v133; // rax
  __int64 v134; // rax
  int v135; // r8d
  __int64 v136; // rsi
  __int64 v137; // rdx
  __int64 v138; // r14
  __int64 v139; // r12
  __int64 v140; // rax
  int v141; // ecx
  __int64 v142; // rsi
  __int64 v143; // rdx
  int v144; // r12d
  __int64 v145; // rax
  __int64 v146; // rax
  int v147; // r8d
  __int64 v148; // rsi
  __int64 v149; // rdx
  __int64 v150; // r14
  __int64 v151; // r12
  __int64 v152; // rax
  int v153; // ecx
  __int64 v154; // rsi
  __int64 v155; // rdx
  int v156; // r12d
  __int64 v157; // rax
  __int64 v158; // rax
  int v159; // r8d
  __int64 v160; // rdx
  __int64 v161; // r13
  __int64 v162; // rax
  __int64 v163; // rdx
  __int64 v164; // r12
  __int64 v165; // r14
  __int64 v166; // r14
  int v167; // r12d
  __int64 v168; // rax
  __int64 v169; // rax
  int v170; // r8d
  __int64 v171; // rdx
  __int64 v172; // r13
  __int64 v173; // rax
  __int64 v174; // rdx
  __int64 v175; // r12
  __int64 v176; // r14
  __int64 v177; // r14
  int v178; // r12d
  __int64 v179; // rax
  __int64 v180; // rax
  int v181; // r8d
  __int64 v182; // rsi
  __int64 v183; // rdx
  __int64 v184; // r14
  int v185; // r12d
  __int64 v186; // rax
  __int64 v187; // rax
  int v188; // r8d
  __int64 v189; // rsi
  __int64 v190; // rdx
  __int64 v191; // r14
  int v192; // r12d
  __int64 v193; // rax
  __int64 v194; // rax
  int v195; // r8d
  __int64 v196; // rdx
  __int64 v197; // r13
  __int64 v198; // rax
  __int64 v199; // rdx
  __int64 v200; // r12
  __int64 v201; // r14
  __int64 v202; // r14
  int v203; // r12d
  __int64 v204; // rax
  __int64 v205; // rax
  int v206; // r8d
  __int64 v207; // rdx
  __int64 v208; // r13
  __int64 v209; // rax
  __int64 v210; // rdx
  __int64 v211; // r12
  __int64 v212; // r14
  __int64 v213; // r14
  int v214; // r12d
  __int64 v215; // rax
  __int64 v216; // rax
  int v217; // r8d
  __int64 v218; // rdx
  __int64 v219; // r13
  __int64 v220; // rax
  __int64 v221; // rdx
  __int64 v222; // r12
  __int64 v223; // r14
  __int64 v224; // r14
  int v225; // r12d
  __int64 v226; // rax
  __int64 v227; // rax
  int v228; // r8d
  __int64 v229; // rsi
  __int64 v230; // rdx
  __int64 v231; // r14
  __int64 v232; // r12
  __int64 v233; // rax
  int v234; // ecx
  __int64 v235; // rsi
  __int64 v236; // rdx
  int v237; // r12d
  __int64 v238; // rax
  __int64 v239; // rax
  int v240; // r8d
  __int64 v241; // rdx
  __int64 v242; // r13
  __int64 v243; // rax
  __int64 v244; // rdx
  __int64 v245; // r12
  __int64 v246; // r14
  __int64 v247; // r14
  int v248; // r12d
  __int64 v249; // rax
  __int64 v250; // rax
  int v251; // r8d
  __int64 v252; // rdx
  __int64 v253; // r13
  __int64 v254; // rax
  __int64 v255; // rdx
  __int64 v256; // r12
  __int64 v257; // r14
  __int64 v258; // r14
  int v259; // r12d
  __int64 v260; // rax
  __int64 v261; // rax
  int v262; // r8d
  __int64 v263; // rsi
  __int64 v264; // rdx
  __int64 v265; // r14
  __int64 v266; // r12
  __int64 v267; // rax
  int v268; // ecx
  __int64 v269; // rsi
  __int64 v270; // rdx
  int v271; // r12d
  __int64 v272; // rax
  __int64 v273; // rax
  int v274; // r8d
  __int64 v275; // rdx
  __int64 v276; // r13
  __int64 v277; // rax
  __int64 v278; // rdx
  __int64 v279; // r12
  __int64 v280; // r14
  __int64 v281; // r14
  int v282; // r12d
  __int64 v283; // rax
  __int64 v284; // rax
  int v285; // r8d
  __int64 v286; // rsi
  __int64 v287; // rdx
  __int64 v288; // r14
  __int64 v289; // r12
  __int64 v290; // rax
  int v291; // ecx
  __int64 v292; // rsi
  __int64 v293; // rdx
  int v294; // r12d
  __int64 v295; // rax
  __int64 v296; // rax
  int v297; // r8d
  __int64 v298; // rdx
  __int64 v299; // rax
  __int64 v300; // r12
  size_t v301; // r15
  const char *v302; // r14
  int v303; // r13d
  __int64 v304; // rax
  __int64 v305; // rdx
  __int64 v306; // r12
  __int64 v307; // r14
  int v308; // r12d
  __int64 v309; // rax
  __int64 v310; // rax
  int v311; // r8d
  __int64 v312; // rdx
  __int64 v313; // r13
  __int64 v314; // rax
  __int64 v315; // rdx
  __int64 v316; // r12
  __int64 v317; // r14
  __int64 v318; // r14
  int v319; // r12d
  __int64 v320; // rax
  __int64 v321; // rax
  int v322; // r8d
  __int64 v323; // rsi
  __int64 v324; // rdx
  __int64 v325; // r14
  __int64 v326; // r12
  __int64 v327; // rax
  int v328; // ecx
  __int64 v329; // rsi
  __int64 v330; // rdx
  int v331; // r12d
  __int64 v332; // rax
  __int64 v333; // rax
  int v334; // r8d
  __int64 v335; // rsi
  __int64 v336; // rdx
  __int64 v337; // r14
  __int64 v338; // r12
  __int64 v339; // rax
  int v340; // ecx
  __int64 v341; // rsi
  __int64 v342; // rdx
  int v343; // r12d
  __int64 v344; // rax
  __int64 v345; // rax
  int v346; // r8d
  __int64 v347; // rdx
  __int64 v348; // r13
  __int64 v349; // rax
  __int64 v350; // rdx
  __int64 v351; // r12
  __int64 v352; // r14
  __int64 v353; // r14
  int v354; // r12d
  __int64 v355; // rax
  __int64 v356; // rax
  int v357; // r8d
  __int64 v358; // rsi
  __int64 v359; // rdx
  __int64 v360; // r14
  __int64 v361; // r12
  __int64 v362; // rax
  int v363; // ecx
  __int64 v364; // rsi
  __int64 v365; // rdx
  int v366; // r12d
  __int64 v367; // rax
  __int64 v368; // rax
  int v369; // r8d
  __int64 v370; // rsi
  __int64 v371; // rdx
  __int64 v372; // r14
  __int64 v373; // r12
  __int64 v374; // rax
  int v375; // ecx
  __int64 v376; // rsi
  __int64 v377; // rdx
  int v378; // r12d
  __int64 v379; // rax
  __int64 v380; // rax
  int v381; // r8d
  __int64 v382; // rdx
  __int64 v383; // rax
  __int64 v384; // r12
  size_t v385; // r15
  const char *v386; // r14
  int v387; // r13d
  __int64 v388; // rax
  __int64 v389; // rdx
  __int64 v390; // r12
  __int64 v391; // r14
  int v392; // r12d
  __int64 v393; // rax
  __int64 v394; // rax
  int v395; // r8d
  __int64 v396; // rdx
  __int64 v397; // r13
  __int64 v398; // rax
  __int64 v399; // rdx
  __int64 v400; // r12
  __int64 v401; // r14
  __int64 v402; // r14
  int v403; // r12d
  __int64 v404; // rax
  __int64 v405; // rax
  int v406; // r8d
  __int64 v407; // rdx
  __int64 v408; // rax
  __int64 v409; // rsi
  const char *v410; // rdi
  __int64 v411; // r12
  int v412; // r14d
  int v413; // r15d
  __int64 v414; // rax
  __int64 v415; // rdx
  __int64 v416; // r12
  __int64 v417; // r15
  int v418; // r14d
  __int64 v419; // rax
  __int64 v420; // rax
  int v421; // r8d
  __int64 v422; // rsi
  __int64 v423; // rdx
  __int64 v424; // r15
  int v425; // r14d
  __int64 v426; // r12
  __int64 v427; // rax
  int v428; // ecx
  __int64 v429; // rdx
  __int64 v430; // rax
  __int64 v431; // rax
  int v432; // r8d
  __int64 v433; // rsi
  __int64 v434; // rdx
  __int64 v435; // r15
  int v436; // r14d
  __int64 v437; // r12
  __int64 v438; // rax
  int v439; // ecx
  __int64 v440; // rdx
  __int64 v441; // rax
  __int64 v442; // rax
  int v443; // r8d
  __int64 v444; // rdx
  __int64 v445; // rax
  __int64 v446; // rsi
  const char *v447; // rdi
  __int64 v448; // r12
  int v449; // r14d
  int v450; // r15d
  __int64 v451; // rax
  __int64 v452; // rdx
  __int64 v453; // r12
  __int64 v454; // r15
  int v455; // r14d
  __int64 v456; // rax
  __int64 v457; // rax
  int v458; // r8d
  __int64 v459; // rdx
  __int64 v460; // rax
  __int64 v461; // rsi
  const char *v462; // rdi
  __int64 v463; // r12
  int v464; // r14d
  int v465; // r15d
  __int64 v466; // rax
  __int64 v467; // rdx
  __int64 v468; // r12
  __int64 v469; // r15
  int v470; // r14d
  __int64 v471; // rax
  __int64 v472; // rax
  int v473; // r8d
  __int64 v474; // rsi
  __int64 v475; // rdx
  __int64 v476; // r15
  int v477; // r14d
  __int64 v478; // r12
  __int64 v479; // rax
  int v480; // ecx
  __int64 v481; // rdx
  __int64 v482; // rax
  __int64 v483; // rax
  int v484; // r8d
  __int64 v485; // rdx
  __int64 v486; // r14
  __int64 v487; // rax
  __int64 v488; // rdx
  __int64 v489; // r12
  __int64 v490; // r15
  __int64 v491; // r15
  int v492; // r14d
  __int64 v493; // rax
  __int64 v494; // rax
  int v495; // r8d
  __int64 v496; // rdx
  __int64 v497; // r14
  __int64 v498; // rax
  __int64 v499; // rdx
  __int64 v500; // r12
  __int64 v501; // r15
  __int64 v502; // r15
  int v503; // r14d
  __int64 v504; // rax
  __int64 v505; // rax
  int v506; // r8d
  __int64 v507; // rdx
  __int64 v508; // rax
  __int64 v509; // rsi
  const char *v510; // rdi
  __int64 v511; // r12
  int v512; // r15d
  int v513; // r14d
  __int64 v514; // rax
  __int64 v515; // rdx
  __int64 v516; // r12
  __int64 v517; // r13
  __int64 result; // rax
  const char **v519; // rax
  size_t v520; // rax
  const char **v521; // rax
  size_t v522; // rax
  const char **v523; // rax
  size_t v524; // rax
  const char **v525; // rax
  size_t v526; // rax
  const char **v527; // rax
  const char **v528; // rax
  const char *v529; // [rsp+8h] [rbp-418h]
  const char *v530; // [rsp+8h] [rbp-418h]
  const char *v531; // [rsp+8h] [rbp-418h]
  const char *v532; // [rsp+8h] [rbp-418h]
  __int64 v533; // [rsp+8h] [rbp-418h]
  __int64 v534; // [rsp+8h] [rbp-418h]
  __int64 v535; // [rsp+8h] [rbp-418h]
  __int64 v536; // [rsp+8h] [rbp-418h]
  __int64 v537; // [rsp+8h] [rbp-418h]
  __int64 v538; // [rsp+18h] [rbp-408h] BYREF
  __int64 v539[2]; // [rsp+20h] [rbp-400h] BYREF
  __int64 v540[2]; // [rsp+30h] [rbp-3F0h] BYREF
  __int64 v541[2]; // [rsp+40h] [rbp-3E0h] BYREF
  __int64 v542[2]; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 v543[2]; // [rsp+60h] [rbp-3C0h] BYREF
  __int64 v544[2]; // [rsp+70h] [rbp-3B0h] BYREF
  __int64 v545[2]; // [rsp+80h] [rbp-3A0h] BYREF
  __int64 v546[2]; // [rsp+90h] [rbp-390h] BYREF
  __int64 v547[2]; // [rsp+A0h] [rbp-380h] BYREF
  __int64 v548[2]; // [rsp+B0h] [rbp-370h] BYREF
  __int64 v549[2]; // [rsp+C0h] [rbp-360h] BYREF
  __int64 v550[2]; // [rsp+D0h] [rbp-350h] BYREF
  __int64 v551[2]; // [rsp+E0h] [rbp-340h] BYREF
  __int64 v552[2]; // [rsp+F0h] [rbp-330h] BYREF
  __int64 v553[2]; // [rsp+100h] [rbp-320h] BYREF
  __int64 v554[2]; // [rsp+110h] [rbp-310h] BYREF
  __int64 v555[2]; // [rsp+120h] [rbp-300h] BYREF
  __int64 v556[2]; // [rsp+130h] [rbp-2F0h] BYREF
  __int64 v557[2]; // [rsp+140h] [rbp-2E0h] BYREF
  __int64 v558[2]; // [rsp+150h] [rbp-2D0h] BYREF
  __int64 v559[2]; // [rsp+160h] [rbp-2C0h] BYREF
  __int64 v560[2]; // [rsp+170h] [rbp-2B0h] BYREF
  __int64 v561[2]; // [rsp+180h] [rbp-2A0h] BYREF
  __int64 v562[2]; // [rsp+190h] [rbp-290h] BYREF
  __int64 v563[2]; // [rsp+1A0h] [rbp-280h] BYREF
  __int64 v564[2]; // [rsp+1B0h] [rbp-270h] BYREF
  __int64 v565[2]; // [rsp+1C0h] [rbp-260h] BYREF
  __int64 v566[2]; // [rsp+1D0h] [rbp-250h] BYREF
  __int64 v567[2]; // [rsp+1E0h] [rbp-240h] BYREF
  __int64 v568[2]; // [rsp+1F0h] [rbp-230h] BYREF
  __int64 v569[2]; // [rsp+200h] [rbp-220h] BYREF
  __int64 v570[2]; // [rsp+210h] [rbp-210h] BYREF
  __int64 v571[2]; // [rsp+220h] [rbp-200h] BYREF
  __int64 v572[2]; // [rsp+230h] [rbp-1F0h] BYREF
  __int64 v573[2]; // [rsp+240h] [rbp-1E0h] BYREF
  __int64 v574[2]; // [rsp+250h] [rbp-1D0h] BYREF
  __int64 v575[2]; // [rsp+260h] [rbp-1C0h] BYREF
  __int64 v576[2]; // [rsp+270h] [rbp-1B0h] BYREF
  __int64 v577[2]; // [rsp+280h] [rbp-1A0h] BYREF
  __int64 v578[2]; // [rsp+290h] [rbp-190h] BYREF
  __int64 v579[2]; // [rsp+2A0h] [rbp-180h] BYREF
  __int64 v580[2]; // [rsp+2B0h] [rbp-170h] BYREF
  __int64 v581[2]; // [rsp+2C0h] [rbp-160h] BYREF
  __int64 v582[2]; // [rsp+2D0h] [rbp-150h] BYREF
  __int64 v583[2]; // [rsp+2E0h] [rbp-140h] BYREF
  __int64 v584[2]; // [rsp+2F0h] [rbp-130h] BYREF
  __int64 v585[2]; // [rsp+300h] [rbp-120h] BYREF
  __int64 v586[2]; // [rsp+310h] [rbp-110h] BYREF
  __int64 v587[2]; // [rsp+320h] [rbp-100h] BYREF
  __int64 v588[2]; // [rsp+330h] [rbp-F0h] BYREF
  __int64 v589[2]; // [rsp+340h] [rbp-E0h] BYREF
  __int64 v590[2]; // [rsp+350h] [rbp-D0h] BYREF
  __int64 v591[2]; // [rsp+360h] [rbp-C0h] BYREF
  __int64 v592[2]; // [rsp+370h] [rbp-B0h] BYREF
  __int64 v593[2]; // [rsp+380h] [rbp-A0h] BYREF
  __int64 v594[2]; // [rsp+390h] [rbp-90h] BYREF
  __int64 v595[2]; // [rsp+3A0h] [rbp-80h] BYREF
  __int64 v596[2]; // [rsp+3B0h] [rbp-70h] BYREF
  __int64 v597[2]; // [rsp+3C0h] [rbp-60h] BYREF
  __int64 v598[2]; // [rsp+3D0h] [rbp-50h] BYREF
  __int64 v599[8]; // [rsp+3E0h] [rbp-40h] BYREF

  v2 = 0;
  v3 = 0;
  v5 = *(_DWORD *)(a2 + 176);
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)a1 = v5;
  v6 = sub_2262230(a2 + 184, 1u);
  if ( v6 )
  {
    if ( *(_DWORD *)(v6 + 56) )
      v2 = **(_QWORD **)(v6 + 48);
    v3 = *(_DWORD *)(v6 + 40);
  }
  v7 = sub_22F59B0(*(_QWORD *)(a1 + 8), 1);
  v8 = *(_DWORD *)a1;
  v9 = v2;
  v539[1] = v10;
  v11 = 0;
  v539[0] = v7;
  v12 = 0;
  sub_2261F60(a1 + 16, v9, v3, v539, v8);
  v13 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 2u);
  if ( v13 )
  {
    if ( *(_DWORD *)(v13 + 56) )
      v11 = **(_QWORD **)(v13 + 48);
    v12 = *(_DWORD *)(v13 + 40);
  }
  v14 = sub_22F59B0(*(_QWORD *)(a1 + 8), 2);
  v15 = *(_DWORD *)a1;
  v16 = v11;
  v540[1] = v17;
  LODWORD(v17) = v12;
  v18 = 0;
  v540[0] = v14;
  v19 = 0;
  sub_2261F60(a1 + 40, v16, v17, v540, v15);
  v20 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 3u);
  if ( v20 )
  {
    if ( *(_DWORD *)(v20 + 56) )
      v18 = **(_QWORD **)(v20 + 48);
    v19 = *(_DWORD *)(v20 + 40);
  }
  v21 = sub_22F59B0(*(_QWORD *)(a1 + 8), 3);
  v22 = *(_DWORD *)a1;
  v23 = v18;
  v541[1] = v24;
  LODWORD(v24) = v19;
  v25 = 0;
  v541[0] = v21;
  v26 = 0;
  sub_2261F60(a1 + 64, v23, v24, v541, v22);
  v27 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 4u);
  if ( v27 )
  {
    if ( *(_DWORD *)(v27 + 56) )
      v25 = **(_QWORD **)(v27 + 48);
    v26 = *(_DWORD *)(v27 + 40);
  }
  v28 = sub_22F59B0(*(_QWORD *)(a1 + 8), 4);
  v29 = *(_DWORD *)a1;
  v30 = v25;
  v542[1] = v31;
  LODWORD(v31) = v26;
  v32 = 0;
  v542[0] = v28;
  v33 = 0;
  sub_2261F60(a1 + 88, v30, v31, v542, v29);
  v34 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 5u);
  if ( v34 )
  {
    if ( *(_DWORD *)(v34 + 56) )
      v32 = **(_QWORD **)(v34 + 48);
    v33 = *(_DWORD *)(v34 + 40);
  }
  v35 = sub_22F59B0(*(_QWORD *)(a1 + 8), 5);
  v36 = *(_DWORD *)a1;
  v37 = v32;
  v543[1] = v38;
  LODWORD(v38) = v33;
  v39 = 0;
  v543[0] = v35;
  v40 = 0;
  sub_2261F60(a1 + 112, v37, v38, v543, v36);
  v41 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 6u);
  if ( v41 )
  {
    if ( *(_DWORD *)(v41 + 56) )
      v39 = **(_QWORD **)(v41 + 48);
    v40 = *(_DWORD *)(v41 + 40);
  }
  v42 = sub_22F59B0(*(_QWORD *)(a1 + 8), 6);
  v43 = *(_DWORD *)a1;
  v544[1] = v44;
  v544[0] = v42;
  sub_2261F60(a1 + 136, v39, v40, v544, v43);
  v45 = sub_2262300(*(_QWORD *)(a1 + 8), 7u, "0");
  v46 = sub_22F59B0(*(_QWORD *)(a1 + 8), 7);
  *(_BYTE *)(a1 + 160) = v45;
  v48 = v46;
  v49 = v47;
  LODWORD(v46) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v48 + 44) == 0;
  *(_DWORD *)(a1 + 164) = HIDWORD(v45);
  *(_DWORD *)(a1 + 168) = v46;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 172) = *(_DWORD *)(v48 + 40);
  }
  else if ( sub_22F59B0(v47, *(unsigned __int16 *)(v48 + 56)) )
  {
    *(_DWORD *)(a1 + 172) = *(_DWORD *)(sub_22F59B0(v49, *(unsigned __int16 *)(v48 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 172) = 0;
  }
  v51 = 0;
  v52 = 0;
  v53 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 8u);
  if ( v53 )
  {
    if ( *(_DWORD *)(v53 + 56) )
      v51 = **(_QWORD **)(v53 + 48);
    v52 = *(_DWORD *)(v53 + 40);
  }
  v54 = sub_22F59B0(*(_QWORD *)(a1 + 8), 8);
  v55 = *(_DWORD *)a1;
  v56 = v51;
  v545[1] = v57;
  v545[0] = v54;
  v58 = 0;
  sub_2261F60(a1 + 176, v56, v52, v545, v55);
  v59 = sub_2262300(*(_QWORD *)(a1 + 8), 9u, "0");
  v60 = sub_22F59B0(*(_QWORD *)(a1 + 8), 9);
  v61 = *(_DWORD *)a1;
  v62 = v59;
  v546[1] = v63;
  v64 = 0;
  v546[0] = v60;
  sub_2261FD0(a1 + 200, v62, v546, v61);
  v65 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0xAu);
  if ( v65 )
  {
    if ( *(_DWORD *)(v65 + 56) )
      v58 = **(_QWORD **)(v65 + 48);
    v64 = *(_DWORD *)(v65 + 40);
  }
  v66 = sub_22F59B0(*(_QWORD *)(a1 + 8), 10);
  v67 = *(_DWORD *)a1;
  v547[1] = v68;
  v547[0] = v66;
  sub_2261F60(a1 + 216, v58, v64, v547, v67);
  v69 = sub_2262300(*(_QWORD *)(a1 + 8), 0xBu, "0");
  v70 = sub_22F59B0(*(_QWORD *)(a1 + 8), 11);
  *(_BYTE *)(a1 + 240) = v69;
  v72 = v70;
  v73 = v71;
  LODWORD(v70) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v72 + 44) == 0;
  *(_DWORD *)(a1 + 244) = HIDWORD(v69);
  *(_DWORD *)(a1 + 248) = v70;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 252) = *(_DWORD *)(v72 + 40);
  }
  else if ( sub_22F59B0(v71, *(unsigned __int16 *)(v72 + 56)) )
  {
    *(_DWORD *)(a1 + 252) = *(_DWORD *)(sub_22F59B0(v73, *(unsigned __int16 *)(v72 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 252) = 0;
  }
  v74 = 0;
  v75 = 0;
  v76 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0xCu);
  if ( v76 )
  {
    if ( *(_DWORD *)(v76 + 56) )
      v74 = **(_QWORD **)(v76 + 48);
    v75 = *(_DWORD *)(v76 + 40);
  }
  v77 = sub_22F59B0(*(_QWORD *)(a1 + 8), 12);
  v78 = *(_DWORD *)a1;
  v79 = v74;
  v548[1] = v80;
  v548[0] = v77;
  v81 = 0;
  sub_2261F60(a1 + 256, v79, v75, v548, v78);
  v82 = sub_2262300(*(_QWORD *)(a1 + 8), 0xDu, "0");
  v83 = sub_22F59B0(*(_QWORD *)(a1 + 8), 13);
  v84 = *(_DWORD *)a1;
  v85 = v82;
  v549[1] = v86;
  v87 = 0;
  v549[0] = v83;
  sub_2261FD0(a1 + 280, v85, v549, v84);
  v88 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0xEu);
  if ( v88 )
  {
    if ( *(_DWORD *)(v88 + 56) )
      v81 = **(_QWORD **)(v88 + 48);
    v87 = *(_DWORD *)(v88 + 40);
  }
  v89 = sub_22F59B0(*(_QWORD *)(a1 + 8), 14);
  v90 = *(_DWORD *)a1;
  v550[1] = v91;
  v550[0] = v89;
  sub_2261F60(a1 + 296, v81, v87, v550, v90);
  v92 = sub_2262300(*(_QWORD *)(a1 + 8), 0xFu, "1");
  v93 = sub_22F59B0(*(_QWORD *)(a1 + 8), 15);
  *(_BYTE *)(a1 + 320) = v92;
  v95 = v93;
  v96 = v94;
  LODWORD(v93) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v95 + 44) == 0;
  *(_DWORD *)(a1 + 324) = HIDWORD(v92);
  *(_DWORD *)(a1 + 328) = v93;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 332) = *(_DWORD *)(v95 + 40);
  }
  else if ( sub_22F59B0(v94, *(unsigned __int16 *)(v95 + 56)) )
  {
    *(_DWORD *)(a1 + 332) = *(_DWORD *)(sub_22F59B0(v96, *(unsigned __int16 *)(v95 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 332) = 0;
  }
  v97 = 0;
  v98 = 0;
  v99 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x10u);
  if ( v99 )
  {
    if ( *(_DWORD *)(v99 + 56) )
      v97 = **(_QWORD **)(v99 + 48);
    v98 = *(_DWORD *)(v99 + 40);
  }
  v100 = sub_22F59B0(*(_QWORD *)(a1 + 8), 16);
  v101 = *(_DWORD *)a1;
  v102 = v97;
  v551[1] = v103;
  v551[0] = v100;
  v104 = 0;
  sub_2261F60(a1 + 336, v102, v98, v551, v101);
  v105 = sub_2262300(*(_QWORD *)(a1 + 8), 0x11u, "1");
  v106 = sub_22F59B0(*(_QWORD *)(a1 + 8), 17);
  v107 = *(_DWORD *)a1;
  v108 = v105;
  v552[1] = v109;
  v110 = 0;
  v552[0] = v106;
  sub_2261FD0(a1 + 360, v108, v552, v107);
  v111 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x12u);
  if ( v111 )
  {
    if ( *(_DWORD *)(v111 + 56) )
      v104 = **(_QWORD **)(v111 + 48);
    v110 = *(_DWORD *)(v111 + 40);
  }
  v112 = sub_22F59B0(*(_QWORD *)(a1 + 8), 18);
  v113 = *(_DWORD *)a1;
  v553[1] = v114;
  v553[0] = v112;
  sub_2261F60(a1 + 376, v104, v110, v553, v113);
  v115 = sub_2262300(*(_QWORD *)(a1 + 8), 0x13u, "0");
  v116 = sub_22F59B0(*(_QWORD *)(a1 + 8), 19);
  *(_BYTE *)(a1 + 400) = v115;
  v118 = v116;
  v119 = v117;
  LODWORD(v116) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v118 + 44) == 0;
  *(_DWORD *)(a1 + 404) = HIDWORD(v115);
  *(_DWORD *)(a1 + 408) = v116;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 412) = *(_DWORD *)(v118 + 40);
  }
  else if ( sub_22F59B0(v117, *(unsigned __int16 *)(v118 + 56)) )
  {
    *(_DWORD *)(a1 + 412) = *(_DWORD *)(sub_22F59B0(v119, *(unsigned __int16 *)(v118 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 412) = 0;
  }
  v120 = 0;
  v121 = 0;
  v122 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x14u);
  if ( v122 )
  {
    if ( *(_DWORD *)(v122 + 56) )
      v120 = **(_QWORD **)(v122 + 48);
    v121 = *(_DWORD *)(v122 + 40);
  }
  v123 = sub_22F59B0(*(_QWORD *)(a1 + 8), 20);
  v124 = *(_DWORD *)a1;
  v554[1] = v125;
  v554[0] = v123;
  sub_2261F60(a1 + 416, v120, v121, v554, v124);
  v126 = sub_2262300(*(_QWORD *)(a1 + 8), 0x15u, "0");
  v127 = sub_22F59B0(*(_QWORD *)(a1 + 8), 21);
  *(_BYTE *)(a1 + 440) = v126;
  v129 = v127;
  v130 = v128;
  LODWORD(v127) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v129 + 44) == 0;
  *(_DWORD *)(a1 + 444) = HIDWORD(v126);
  *(_DWORD *)(a1 + 448) = v127;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 452) = *(_DWORD *)(v129 + 40);
  }
  else if ( sub_22F59B0(v128, *(unsigned __int16 *)(v129 + 56)) )
  {
    *(_DWORD *)(a1 + 452) = *(_DWORD *)(sub_22F59B0(v130, *(unsigned __int16 *)(v129 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 452) = 0;
  }
  v131 = 0;
  v132 = 0;
  v133 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x16u);
  if ( v133 )
  {
    if ( *(_DWORD *)(v133 + 56) )
      v131 = **(_QWORD **)(v133 + 48);
    v132 = *(_DWORD *)(v133 + 40);
  }
  v134 = sub_22F59B0(*(_QWORD *)(a1 + 8), 22);
  v135 = *(_DWORD *)a1;
  v136 = v131;
  v555[1] = v137;
  v555[0] = v134;
  v138 = 0;
  sub_2261F60(a1 + 456, v136, v132, v555, v135);
  v139 = sub_2262300(*(_QWORD *)(a1 + 8), 0x17u, "1");
  v140 = sub_22F59B0(*(_QWORD *)(a1 + 8), 23);
  v141 = *(_DWORD *)a1;
  v142 = v139;
  v556[1] = v143;
  v144 = 0;
  v556[0] = v140;
  sub_2261FD0(a1 + 480, v142, v556, v141);
  v145 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x18u);
  if ( v145 )
  {
    if ( *(_DWORD *)(v145 + 56) )
      v138 = **(_QWORD **)(v145 + 48);
    v144 = *(_DWORD *)(v145 + 40);
  }
  v146 = sub_22F59B0(*(_QWORD *)(a1 + 8), 24);
  v147 = *(_DWORD *)a1;
  v148 = v138;
  v557[1] = v149;
  v557[0] = v146;
  v150 = 0;
  sub_2261F60(a1 + 496, v148, v144, v557, v147);
  v151 = sub_2262300(*(_QWORD *)(a1 + 8), 0x19u, "1");
  v152 = sub_22F59B0(*(_QWORD *)(a1 + 8), 25);
  v153 = *(_DWORD *)a1;
  v154 = v151;
  v558[1] = v155;
  v156 = 0;
  v558[0] = v152;
  sub_2261FD0(a1 + 520, v154, v558, v153);
  v157 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x1Au);
  if ( v157 )
  {
    if ( *(_DWORD *)(v157 + 56) )
      v150 = **(_QWORD **)(v157 + 48);
    v156 = *(_DWORD *)(v157 + 40);
  }
  v158 = sub_22F59B0(*(_QWORD *)(a1 + 8), 26);
  v159 = *(_DWORD *)a1;
  v559[1] = v160;
  v559[0] = v158;
  sub_2261F60(a1 + 536, v150, v156, v559, v159);
  v161 = sub_2262300(*(_QWORD *)(a1 + 8), 0x1Bu, "0");
  v162 = sub_22F59B0(*(_QWORD *)(a1 + 8), 27);
  *(_BYTE *)(a1 + 560) = v161;
  v164 = v162;
  v165 = v163;
  LODWORD(v162) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v164 + 44) == 0;
  *(_DWORD *)(a1 + 564) = HIDWORD(v161);
  *(_DWORD *)(a1 + 568) = v162;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 572) = *(_DWORD *)(v164 + 40);
  }
  else if ( sub_22F59B0(v163, *(unsigned __int16 *)(v164 + 56)) )
  {
    *(_DWORD *)(a1 + 572) = *(_DWORD *)(sub_22F59B0(v165, *(unsigned __int16 *)(v164 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 572) = 0;
  }
  v166 = 0;
  v167 = 0;
  v168 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x1Cu);
  if ( v168 )
  {
    if ( *(_DWORD *)(v168 + 56) )
      v166 = **(_QWORD **)(v168 + 48);
    v167 = *(_DWORD *)(v168 + 40);
  }
  v169 = sub_22F59B0(*(_QWORD *)(a1 + 8), 28);
  v170 = *(_DWORD *)a1;
  v560[1] = v171;
  v560[0] = v169;
  sub_2261F60(a1 + 576, v166, v167, v560, v170);
  v172 = sub_2262300(*(_QWORD *)(a1 + 8), 0x1Du, "1");
  v173 = sub_22F59B0(*(_QWORD *)(a1 + 8), 29);
  *(_BYTE *)(a1 + 600) = v172;
  v175 = v173;
  v176 = v174;
  LODWORD(v173) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v175 + 44) == 0;
  *(_DWORD *)(a1 + 604) = HIDWORD(v172);
  *(_DWORD *)(a1 + 608) = v173;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 612) = *(_DWORD *)(v175 + 40);
  }
  else if ( sub_22F59B0(v174, *(unsigned __int16 *)(v175 + 56)) )
  {
    *(_DWORD *)(a1 + 612) = *(_DWORD *)(sub_22F59B0(v176, *(unsigned __int16 *)(v175 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 612) = 0;
  }
  v177 = 0;
  v178 = 0;
  v179 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x1Eu);
  if ( v179 )
  {
    if ( *(_DWORD *)(v179 + 56) )
      v177 = **(_QWORD **)(v179 + 48);
    v178 = *(_DWORD *)(v179 + 40);
  }
  v180 = sub_22F59B0(*(_QWORD *)(a1 + 8), 30);
  v181 = *(_DWORD *)a1;
  v182 = v177;
  v561[1] = v183;
  LODWORD(v183) = v178;
  v184 = 0;
  v561[0] = v180;
  v185 = 0;
  sub_2261F60(a1 + 616, v182, v183, v561, v181);
  v186 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x1Fu);
  if ( v186 )
  {
    if ( *(_DWORD *)(v186 + 56) )
      v184 = **(_QWORD **)(v186 + 48);
    v185 = *(_DWORD *)(v186 + 40);
  }
  v187 = sub_22F59B0(*(_QWORD *)(a1 + 8), 31);
  v188 = *(_DWORD *)a1;
  v189 = v184;
  v562[1] = v190;
  LODWORD(v190) = v185;
  v191 = 0;
  v562[0] = v187;
  v192 = 0;
  sub_2261F60(a1 + 640, v189, v190, v562, v188);
  v193 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x20u);
  if ( v193 )
  {
    if ( *(_DWORD *)(v193 + 56) )
      v191 = **(_QWORD **)(v193 + 48);
    v192 = *(_DWORD *)(v193 + 40);
  }
  v194 = sub_22F59B0(*(_QWORD *)(a1 + 8), 32);
  v195 = *(_DWORD *)a1;
  v563[1] = v196;
  v563[0] = v194;
  sub_2261F60(a1 + 664, v191, v192, v563, v195);
  v197 = sub_2262300(*(_QWORD *)(a1 + 8), 0x21u, "0");
  v198 = sub_22F59B0(*(_QWORD *)(a1 + 8), 33);
  *(_BYTE *)(a1 + 688) = v197;
  v200 = v198;
  v201 = v199;
  LODWORD(v198) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v200 + 44) == 0;
  *(_DWORD *)(a1 + 692) = HIDWORD(v197);
  *(_DWORD *)(a1 + 696) = v198;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 700) = *(_DWORD *)(v200 + 40);
  }
  else if ( sub_22F59B0(v199, *(unsigned __int16 *)(v200 + 56)) )
  {
    *(_DWORD *)(a1 + 700) = *(_DWORD *)(sub_22F59B0(v201, *(unsigned __int16 *)(v200 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 700) = 0;
  }
  v202 = 0;
  v203 = 0;
  v204 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x22u);
  if ( v204 )
  {
    if ( *(_DWORD *)(v204 + 56) )
      v202 = **(_QWORD **)(v204 + 48);
    v203 = *(_DWORD *)(v204 + 40);
  }
  v205 = sub_22F59B0(*(_QWORD *)(a1 + 8), 34);
  v206 = *(_DWORD *)a1;
  v564[1] = v207;
  v564[0] = v205;
  sub_2261F60(a1 + 704, v202, v203, v564, v206);
  v208 = sub_2262300(*(_QWORD *)(a1 + 8), 0x23u, "0");
  v209 = sub_22F59B0(*(_QWORD *)(a1 + 8), 35);
  *(_BYTE *)(a1 + 728) = v208;
  v211 = v209;
  v212 = v210;
  LODWORD(v209) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v211 + 44) == 0;
  *(_DWORD *)(a1 + 732) = HIDWORD(v208);
  *(_DWORD *)(a1 + 736) = v209;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 740) = *(_DWORD *)(v211 + 40);
  }
  else if ( sub_22F59B0(v210, *(unsigned __int16 *)(v211 + 56)) )
  {
    *(_DWORD *)(a1 + 740) = *(_DWORD *)(sub_22F59B0(v212, *(unsigned __int16 *)(v211 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 740) = 0;
  }
  v213 = 0;
  v214 = 0;
  v215 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x24u);
  if ( v215 )
  {
    if ( *(_DWORD *)(v215 + 56) )
      v213 = **(_QWORD **)(v215 + 48);
    v214 = *(_DWORD *)(v215 + 40);
  }
  v216 = sub_22F59B0(*(_QWORD *)(a1 + 8), 36);
  v217 = *(_DWORD *)a1;
  v565[1] = v218;
  v565[0] = v216;
  sub_2261F60(a1 + 744, v213, v214, v565, v217);
  v219 = sub_2262300(*(_QWORD *)(a1 + 8), 0x25u, "0");
  v220 = sub_22F59B0(*(_QWORD *)(a1 + 8), 37);
  *(_BYTE *)(a1 + 768) = v219;
  v222 = v220;
  v223 = v221;
  LODWORD(v220) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v222 + 44) == 0;
  *(_DWORD *)(a1 + 772) = HIDWORD(v219);
  *(_DWORD *)(a1 + 776) = v220;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 780) = *(_DWORD *)(v222 + 40);
  }
  else if ( sub_22F59B0(v221, *(unsigned __int16 *)(v222 + 56)) )
  {
    *(_DWORD *)(a1 + 780) = *(_DWORD *)(sub_22F59B0(v223, *(unsigned __int16 *)(v222 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 780) = 0;
  }
  v224 = 0;
  v225 = 0;
  v226 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x26u);
  if ( v226 )
  {
    if ( *(_DWORD *)(v226 + 56) )
      v224 = **(_QWORD **)(v226 + 48);
    v225 = *(_DWORD *)(v226 + 40);
  }
  v227 = sub_22F59B0(*(_QWORD *)(a1 + 8), 38);
  v228 = *(_DWORD *)a1;
  v229 = v224;
  v566[1] = v230;
  v566[0] = v227;
  v231 = 0;
  sub_2261F60(a1 + 784, v229, v225, v566, v228);
  v232 = sub_2262300(*(_QWORD *)(a1 + 8), 0x27u, "0");
  v233 = sub_22F59B0(*(_QWORD *)(a1 + 8), 39);
  v234 = *(_DWORD *)a1;
  v235 = v232;
  v567[1] = v236;
  v237 = 0;
  v567[0] = v233;
  sub_2261FD0(a1 + 808, v235, v567, v234);
  v238 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x28u);
  if ( v238 )
  {
    if ( *(_DWORD *)(v238 + 56) )
      v231 = **(_QWORD **)(v238 + 48);
    v237 = *(_DWORD *)(v238 + 40);
  }
  v239 = sub_22F59B0(*(_QWORD *)(a1 + 8), 40);
  v240 = *(_DWORD *)a1;
  v568[1] = v241;
  v568[0] = v239;
  sub_2261F60(a1 + 824, v231, v237, v568, v240);
  v242 = sub_2262300(*(_QWORD *)(a1 + 8), 0x29u, "0");
  v243 = sub_22F59B0(*(_QWORD *)(a1 + 8), 41);
  *(_BYTE *)(a1 + 848) = v242;
  v245 = v243;
  v246 = v244;
  LODWORD(v243) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v245 + 44) == 0;
  *(_DWORD *)(a1 + 852) = HIDWORD(v242);
  *(_DWORD *)(a1 + 856) = v243;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 860) = *(_DWORD *)(v245 + 40);
  }
  else if ( sub_22F59B0(v244, *(unsigned __int16 *)(v245 + 56)) )
  {
    *(_DWORD *)(a1 + 860) = *(_DWORD *)(sub_22F59B0(v246, *(unsigned __int16 *)(v245 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 860) = 0;
  }
  v247 = 0;
  v248 = 0;
  v249 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x2Au);
  if ( v249 )
  {
    if ( *(_DWORD *)(v249 + 56) )
      v247 = **(_QWORD **)(v249 + 48);
    v248 = *(_DWORD *)(v249 + 40);
  }
  v250 = sub_22F59B0(*(_QWORD *)(a1 + 8), 42);
  v251 = *(_DWORD *)a1;
  v569[1] = v252;
  v569[0] = v250;
  sub_2261F60(a1 + 864, v247, v248, v569, v251);
  v253 = sub_2262300(*(_QWORD *)(a1 + 8), 0x2Bu, "0");
  v254 = sub_22F59B0(*(_QWORD *)(a1 + 8), 43);
  *(_BYTE *)(a1 + 888) = v253;
  v256 = v254;
  v257 = v255;
  LODWORD(v254) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v256 + 44) == 0;
  *(_DWORD *)(a1 + 892) = HIDWORD(v253);
  *(_DWORD *)(a1 + 896) = v254;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 900) = *(_DWORD *)(v256 + 40);
  }
  else if ( sub_22F59B0(v255, *(unsigned __int16 *)(v256 + 56)) )
  {
    *(_DWORD *)(a1 + 900) = *(_DWORD *)(sub_22F59B0(v257, *(unsigned __int16 *)(v256 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 900) = 0;
  }
  v258 = 0;
  v259 = 0;
  v260 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x2Cu);
  if ( v260 )
  {
    if ( *(_DWORD *)(v260 + 56) )
      v258 = **(_QWORD **)(v260 + 48);
    v259 = *(_DWORD *)(v260 + 40);
  }
  v261 = sub_22F59B0(*(_QWORD *)(a1 + 8), 44);
  v262 = *(_DWORD *)a1;
  v263 = v258;
  v570[1] = v264;
  v570[0] = v261;
  v265 = 0;
  sub_2261F60(a1 + 904, v263, v259, v570, v262);
  v266 = sub_2262300(*(_QWORD *)(a1 + 8), 0x2Du, "0");
  v267 = sub_22F59B0(*(_QWORD *)(a1 + 8), 45);
  v268 = *(_DWORD *)a1;
  v269 = v266;
  v571[1] = v270;
  v271 = 0;
  v571[0] = v267;
  sub_2261FD0(a1 + 928, v269, v571, v268);
  v272 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x2Eu);
  if ( v272 )
  {
    if ( *(_DWORD *)(v272 + 56) )
      v265 = **(_QWORD **)(v272 + 48);
    v271 = *(_DWORD *)(v272 + 40);
  }
  v273 = sub_22F59B0(*(_QWORD *)(a1 + 8), 46);
  v274 = *(_DWORD *)a1;
  v572[1] = v275;
  v572[0] = v273;
  sub_2261F60(a1 + 944, v265, v271, v572, v274);
  v276 = sub_2262300(*(_QWORD *)(a1 + 8), 0x2Fu, "0");
  v277 = sub_22F59B0(*(_QWORD *)(a1 + 8), 47);
  *(_BYTE *)(a1 + 968) = v276;
  v279 = v277;
  v280 = v278;
  LODWORD(v277) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v279 + 44) == 0;
  *(_DWORD *)(a1 + 972) = HIDWORD(v276);
  *(_DWORD *)(a1 + 976) = v277;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 980) = *(_DWORD *)(v279 + 40);
  }
  else if ( sub_22F59B0(v278, *(unsigned __int16 *)(v279 + 56)) )
  {
    *(_DWORD *)(a1 + 980) = *(_DWORD *)(sub_22F59B0(v280, *(unsigned __int16 *)(v279 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 980) = 0;
  }
  v281 = 0;
  v282 = 0;
  v283 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x30u);
  if ( v283 )
  {
    if ( *(_DWORD *)(v283 + 56) )
      v281 = **(_QWORD **)(v283 + 48);
    v282 = *(_DWORD *)(v283 + 40);
  }
  v284 = sub_22F59B0(*(_QWORD *)(a1 + 8), 48);
  v285 = *(_DWORD *)a1;
  v286 = v281;
  v573[1] = v287;
  v573[0] = v284;
  v288 = 0;
  sub_2261F60(a1 + 984, v286, v282, v573, v285);
  v289 = sub_2262300(*(_QWORD *)(a1 + 8), 0x31u, "0");
  v290 = sub_22F59B0(*(_QWORD *)(a1 + 8), 49);
  v291 = *(_DWORD *)a1;
  v292 = v289;
  v574[1] = v293;
  v294 = 0;
  v574[0] = v290;
  sub_2261FD0(a1 + 1008, v292, v574, v291);
  v295 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x32u);
  if ( v295 )
  {
    if ( *(_DWORD *)(v295 + 56) )
      v288 = **(_QWORD **)(v295 + 48);
    v294 = *(_DWORD *)(v295 + 40);
  }
  v296 = sub_22F59B0(*(_QWORD *)(a1 + 8), 50);
  v297 = *(_DWORD *)a1;
  v575[1] = v298;
  v575[0] = v296;
  sub_2261F60(a1 + 1024, v288, v294, v575, v297);
  v299 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x33u);
  v300 = v299;
  if ( v299 )
  {
    v301 = 0;
    v302 = byte_3F871B3;
    if ( *(_DWORD *)(v299 + 56) )
    {
      v528 = *(const char ***)(v299 + 48);
      v302 = *v528;
      if ( *v528 )
        v301 = strlen(*v528);
    }
    v303 = *(_DWORD *)(v300 + 40);
  }
  else
  {
    v302 = byte_3F871B3;
    v301 = 0;
    v303 = 0;
  }
  v304 = sub_22F59B0(*(_QWORD *)(a1 + 8), 51);
  *(_QWORD *)(a1 + 1048) = v302;
  v306 = v304;
  LODWORD(v304) = *(_DWORD *)a1;
  *(_QWORD *)(a1 + 1056) = v301;
  v50 = *(_BYTE *)(v306 + 44) == 0;
  *(_DWORD *)(a1 + 1064) = v303;
  *(_DWORD *)(a1 + 1068) = v304;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1072) = *(_DWORD *)(v306 + 40);
  }
  else
  {
    v533 = v305;
    if ( sub_22F59B0(v305, *(unsigned __int16 *)(v306 + 56)) )
      *(_DWORD *)(a1 + 1072) = *(_DWORD *)(sub_22F59B0(v533, *(unsigned __int16 *)(v306 + 56)) + 40);
    else
      *(_DWORD *)(a1 + 1072) = 0;
  }
  v307 = 0;
  v308 = 0;
  v309 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x34u);
  if ( v309 )
  {
    if ( *(_DWORD *)(v309 + 56) )
      v307 = **(_QWORD **)(v309 + 48);
    v308 = *(_DWORD *)(v309 + 40);
  }
  v310 = sub_22F59B0(*(_QWORD *)(a1 + 8), 52);
  v311 = *(_DWORD *)a1;
  v576[1] = v312;
  v576[0] = v310;
  sub_2261F60(a1 + 1080, v307, v308, v576, v311);
  v313 = sub_2262300(*(_QWORD *)(a1 + 8), 0x35u, "0");
  v314 = sub_22F59B0(*(_QWORD *)(a1 + 8), 53);
  *(_BYTE *)(a1 + 1104) = v313;
  v316 = v314;
  v317 = v315;
  LODWORD(v314) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v316 + 44) == 0;
  *(_DWORD *)(a1 + 1108) = HIDWORD(v313);
  *(_DWORD *)(a1 + 1112) = v314;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1116) = *(_DWORD *)(v316 + 40);
  }
  else if ( sub_22F59B0(v315, *(unsigned __int16 *)(v316 + 56)) )
  {
    *(_DWORD *)(a1 + 1116) = *(_DWORD *)(sub_22F59B0(v317, *(unsigned __int16 *)(v316 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 1116) = 0;
  }
  v318 = 0;
  v319 = 0;
  v320 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x36u);
  if ( v320 )
  {
    if ( *(_DWORD *)(v320 + 56) )
      v318 = **(_QWORD **)(v320 + 48);
    v319 = *(_DWORD *)(v320 + 40);
  }
  v321 = sub_22F59B0(*(_QWORD *)(a1 + 8), 54);
  v322 = *(_DWORD *)a1;
  v323 = v318;
  v577[1] = v324;
  v577[0] = v321;
  v325 = 0;
  sub_2261F60(a1 + 1120, v323, v319, v577, v322);
  v326 = sub_2262300(*(_QWORD *)(a1 + 8), 0x37u, "0");
  v327 = sub_22F59B0(*(_QWORD *)(a1 + 8), 55);
  v328 = *(_DWORD *)a1;
  v329 = v326;
  v578[1] = v330;
  v331 = 0;
  v578[0] = v327;
  sub_2261FD0(a1 + 1144, v329, v578, v328);
  v332 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x38u);
  if ( v332 )
  {
    if ( *(_DWORD *)(v332 + 56) )
      v325 = **(_QWORD **)(v332 + 48);
    v331 = *(_DWORD *)(v332 + 40);
  }
  v333 = sub_22F59B0(*(_QWORD *)(a1 + 8), 56);
  v334 = *(_DWORD *)a1;
  v335 = v325;
  v579[1] = v336;
  v579[0] = v333;
  v337 = 0;
  sub_2261F60(a1 + 1160, v335, v331, v579, v334);
  v338 = sub_2262300(*(_QWORD *)(a1 + 8), 0x39u, "0");
  v339 = sub_22F59B0(*(_QWORD *)(a1 + 8), 57);
  v340 = *(_DWORD *)a1;
  v341 = v338;
  v580[1] = v342;
  v343 = 0;
  v580[0] = v339;
  sub_2261FD0(a1 + 1184, v341, v580, v340);
  v344 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x3Au);
  if ( v344 )
  {
    if ( *(_DWORD *)(v344 + 56) )
      v337 = **(_QWORD **)(v344 + 48);
    v343 = *(_DWORD *)(v344 + 40);
  }
  v345 = sub_22F59B0(*(_QWORD *)(a1 + 8), 58);
  v346 = *(_DWORD *)a1;
  v581[1] = v347;
  v581[0] = v345;
  sub_2261F60(a1 + 1200, v337, v343, v581, v346);
  v348 = sub_2262300(*(_QWORD *)(a1 + 8), 0x3Bu, "0");
  v349 = sub_22F59B0(*(_QWORD *)(a1 + 8), 59);
  *(_BYTE *)(a1 + 1224) = v348;
  v351 = v349;
  v352 = v350;
  LODWORD(v349) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v351 + 44) == 0;
  *(_DWORD *)(a1 + 1228) = HIDWORD(v348);
  *(_DWORD *)(a1 + 1232) = v349;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1236) = *(_DWORD *)(v351 + 40);
  }
  else if ( sub_22F59B0(v350, *(unsigned __int16 *)(v351 + 56)) )
  {
    *(_DWORD *)(a1 + 1236) = *(_DWORD *)(sub_22F59B0(v352, *(unsigned __int16 *)(v351 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 1236) = 0;
  }
  v353 = 0;
  v354 = 0;
  v355 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x3Cu);
  if ( v355 )
  {
    if ( *(_DWORD *)(v355 + 56) )
      v353 = **(_QWORD **)(v355 + 48);
    v354 = *(_DWORD *)(v355 + 40);
  }
  v356 = sub_22F59B0(*(_QWORD *)(a1 + 8), 60);
  v357 = *(_DWORD *)a1;
  v358 = v353;
  v582[1] = v359;
  v582[0] = v356;
  v360 = 0;
  sub_2261F60(a1 + 1240, v358, v354, v582, v357);
  v361 = sub_2262300(*(_QWORD *)(a1 + 8), 0x3Du, "0");
  v362 = sub_22F59B0(*(_QWORD *)(a1 + 8), 61);
  v363 = *(_DWORD *)a1;
  v364 = v361;
  v583[1] = v365;
  v366 = 0;
  v583[0] = v362;
  sub_2261FD0(a1 + 1264, v364, v583, v363);
  v367 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x3Eu);
  if ( v367 )
  {
    if ( *(_DWORD *)(v367 + 56) )
      v360 = **(_QWORD **)(v367 + 48);
    v366 = *(_DWORD *)(v367 + 40);
  }
  v368 = sub_22F59B0(*(_QWORD *)(a1 + 8), 62);
  v369 = *(_DWORD *)a1;
  v370 = v360;
  v584[1] = v371;
  v584[0] = v368;
  v372 = 0;
  sub_2261F60(a1 + 1280, v370, v366, v584, v369);
  v373 = sub_2262300(*(_QWORD *)(a1 + 8), 0x3Fu, "0");
  v374 = sub_22F59B0(*(_QWORD *)(a1 + 8), 63);
  v375 = *(_DWORD *)a1;
  v376 = v373;
  v585[1] = v377;
  v378 = 0;
  v585[0] = v374;
  sub_2261FD0(a1 + 1304, v376, v585, v375);
  v379 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x40u);
  if ( v379 )
  {
    if ( *(_DWORD *)(v379 + 56) )
      v372 = **(_QWORD **)(v379 + 48);
    v378 = *(_DWORD *)(v379 + 40);
  }
  v380 = sub_22F59B0(*(_QWORD *)(a1 + 8), 64);
  v381 = *(_DWORD *)a1;
  v586[1] = v382;
  v586[0] = v380;
  sub_2261F60(a1 + 1320, v372, v378, v586, v381);
  v383 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x41u);
  v384 = v383;
  if ( v383 )
  {
    v385 = 0;
    v386 = byte_3F871B3;
    if ( *(_DWORD *)(v383 + 56) )
    {
      v527 = *(const char ***)(v383 + 48);
      v386 = *v527;
      if ( *v527 )
        v385 = strlen(*v527);
    }
    v387 = *(_DWORD *)(v384 + 40);
  }
  else
  {
    v386 = byte_3F871B3;
    v385 = 0;
    v387 = 0;
  }
  v388 = sub_22F59B0(*(_QWORD *)(a1 + 8), 65);
  *(_QWORD *)(a1 + 1344) = v386;
  v390 = v388;
  LODWORD(v388) = *(_DWORD *)a1;
  *(_QWORD *)(a1 + 1352) = v385;
  v50 = *(_BYTE *)(v390 + 44) == 0;
  *(_DWORD *)(a1 + 1360) = v387;
  *(_DWORD *)(a1 + 1364) = v388;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1368) = *(_DWORD *)(v390 + 40);
  }
  else
  {
    v537 = v389;
    if ( sub_22F59B0(v389, *(unsigned __int16 *)(v390 + 56)) )
      *(_DWORD *)(a1 + 1368) = *(_DWORD *)(sub_22F59B0(v537, *(unsigned __int16 *)(v390 + 56)) + 40);
    else
      *(_DWORD *)(a1 + 1368) = 0;
  }
  v391 = 0;
  v392 = 0;
  v393 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x42u);
  if ( v393 )
  {
    if ( *(_DWORD *)(v393 + 56) )
      v391 = **(_QWORD **)(v393 + 48);
    v392 = *(_DWORD *)(v393 + 40);
  }
  v394 = sub_22F59B0(*(_QWORD *)(a1 + 8), 66);
  v395 = *(_DWORD *)a1;
  v587[1] = v396;
  v587[0] = v394;
  sub_2261F60(a1 + 1376, v391, v392, v587, v395);
  v397 = sub_2262300(*(_QWORD *)(a1 + 8), 0x43u, "0");
  v398 = sub_22F59B0(*(_QWORD *)(a1 + 8), 67);
  *(_BYTE *)(a1 + 1400) = v397;
  v400 = v398;
  v401 = v399;
  LODWORD(v398) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v400 + 44) == 0;
  *(_DWORD *)(a1 + 1404) = HIDWORD(v397);
  *(_DWORD *)(a1 + 1408) = v398;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1412) = *(_DWORD *)(v400 + 40);
  }
  else if ( sub_22F59B0(v399, *(unsigned __int16 *)(v400 + 56)) )
  {
    *(_DWORD *)(a1 + 1412) = *(_DWORD *)(sub_22F59B0(v401, *(unsigned __int16 *)(v400 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 1412) = 0;
  }
  v402 = 0;
  v403 = 0;
  v404 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x44u);
  if ( v404 )
  {
    if ( *(_DWORD *)(v404 + 56) )
      v402 = **(_QWORD **)(v404 + 48);
    v403 = *(_DWORD *)(v404 + 40);
  }
  v405 = sub_22F59B0(*(_QWORD *)(a1 + 8), 68);
  v406 = *(_DWORD *)a1;
  v588[1] = v407;
  v588[0] = v405;
  sub_2261F60(a1 + 1416, v402, v403, v588, v406);
  v408 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x45u);
  v409 = 2;
  v410 = "20";
  v411 = v408;
  if ( v408 )
  {
    if ( *(_DWORD *)(v408 + 56) )
    {
      v525 = *(const char ***)(v408 + 48);
      v409 = 0;
      v410 = *v525;
      if ( *v525 )
      {
        v532 = *v525;
        v526 = strlen(v410);
        v410 = v532;
        v409 = v526;
      }
    }
  }
  if ( sub_C93CC0((__int64)v410, v409, 0, v599) || (v412 = v599[0], v599[0] != SLODWORD(v599[0])) )
    v412 = 0;
  v413 = 0;
  if ( v411 )
    v413 = *(_DWORD *)(v411 + 40);
  v414 = sub_22F59B0(*(_QWORD *)(a1 + 8), 69);
  *(_DWORD *)(a1 + 1440) = v412;
  v416 = v414;
  LODWORD(v414) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 1444) = v413;
  v50 = *(_BYTE *)(v416 + 44) == 0;
  *(_DWORD *)(a1 + 1448) = v414;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1452) = *(_DWORD *)(v416 + 40);
  }
  else
  {
    v536 = v415;
    if ( sub_22F59B0(v415, *(unsigned __int16 *)(v416 + 56)) )
      *(_DWORD *)(a1 + 1452) = *(_DWORD *)(sub_22F59B0(v536, *(unsigned __int16 *)(v416 + 56)) + 40);
    else
      *(_DWORD *)(a1 + 1452) = 0;
  }
  v417 = 0;
  v418 = 0;
  v419 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x46u);
  if ( v419 )
  {
    if ( *(_DWORD *)(v419 + 56) )
      v417 = **(_QWORD **)(v419 + 48);
    v418 = *(_DWORD *)(v419 + 40);
  }
  v420 = sub_22F59B0(*(_QWORD *)(a1 + 8), 70);
  v421 = *(_DWORD *)a1;
  v422 = v417;
  v589[1] = v423;
  LODWORD(v423) = v418;
  v424 = 0;
  v589[0] = v420;
  v425 = 0;
  sub_2261F60(a1 + 1456, v422, v423, v589, v421);
  v426 = sub_2262300(*(_QWORD *)(a1 + 8), 0x47u, "0");
  v427 = sub_22F59B0(*(_QWORD *)(a1 + 8), 71);
  v428 = *(_DWORD *)a1;
  v590[1] = v429;
  v590[0] = v427;
  sub_2261FD0(a1 + 1480, v426, v590, v428);
  v430 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x48u);
  if ( v430 )
  {
    if ( *(_DWORD *)(v430 + 56) )
      v424 = **(_QWORD **)(v430 + 48);
    v425 = *(_DWORD *)(v430 + 40);
  }
  v431 = sub_22F59B0(*(_QWORD *)(a1 + 8), 72);
  v432 = *(_DWORD *)a1;
  v433 = v424;
  v591[1] = v434;
  LODWORD(v434) = v425;
  v435 = 0;
  v591[0] = v431;
  v436 = 0;
  sub_2261F60(a1 + 1496, v433, v434, v591, v432);
  v437 = sub_2262300(*(_QWORD *)(a1 + 8), 0x49u, "0");
  v438 = sub_22F59B0(*(_QWORD *)(a1 + 8), 73);
  v439 = *(_DWORD *)a1;
  v592[1] = v440;
  v592[0] = v438;
  sub_2261FD0(a1 + 1520, v437, v592, v439);
  v441 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x4Au);
  if ( v441 )
  {
    if ( *(_DWORD *)(v441 + 56) )
      v435 = **(_QWORD **)(v441 + 48);
    v436 = *(_DWORD *)(v441 + 40);
  }
  v442 = sub_22F59B0(*(_QWORD *)(a1 + 8), 74);
  v443 = *(_DWORD *)a1;
  v593[1] = v444;
  v593[0] = v442;
  sub_2261F60(a1 + 1536, v435, v436, v593, v443);
  v445 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x4Bu);
  v446 = 2;
  v447 = "-1";
  v448 = v445;
  if ( v445 )
  {
    if ( *(_DWORD *)(v445 + 56) )
    {
      v523 = *(const char ***)(v445 + 48);
      v446 = 0;
      v447 = *v523;
      if ( *v523 )
      {
        v531 = *v523;
        v524 = strlen(v447);
        v447 = v531;
        v446 = v524;
      }
    }
  }
  if ( sub_C93CC0((__int64)v447, v446, 0, v599) || (v449 = v599[0], v599[0] != SLODWORD(v599[0])) )
    v449 = 0;
  v450 = 0;
  if ( v448 )
    v450 = *(_DWORD *)(v448 + 40);
  v451 = sub_22F59B0(*(_QWORD *)(a1 + 8), 75);
  *(_DWORD *)(a1 + 1560) = v449;
  v453 = v451;
  LODWORD(v451) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 1564) = v450;
  v50 = *(_BYTE *)(v453 + 44) == 0;
  *(_DWORD *)(a1 + 1568) = v451;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1572) = *(_DWORD *)(v453 + 40);
  }
  else
  {
    v535 = v452;
    if ( sub_22F59B0(v452, *(unsigned __int16 *)(v453 + 56)) )
      *(_DWORD *)(a1 + 1572) = *(_DWORD *)(sub_22F59B0(v535, *(unsigned __int16 *)(v453 + 56)) + 40);
    else
      *(_DWORD *)(a1 + 1572) = 0;
  }
  v454 = 0;
  v455 = 0;
  v456 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x4Cu);
  if ( v456 )
  {
    if ( *(_DWORD *)(v456 + 56) )
      v454 = **(_QWORD **)(v456 + 48);
    v455 = *(_DWORD *)(v456 + 40);
  }
  v457 = sub_22F59B0(*(_QWORD *)(a1 + 8), 76);
  v458 = *(_DWORD *)a1;
  v594[1] = v459;
  v594[0] = v457;
  sub_2261F60(a1 + 1576, v454, v455, v594, v458);
  v460 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x4Du);
  v461 = 2;
  v462 = "-1";
  v463 = v460;
  if ( v460 )
  {
    if ( *(_DWORD *)(v460 + 56) )
    {
      v521 = *(const char ***)(v460 + 48);
      v461 = 0;
      v462 = *v521;
      if ( *v521 )
      {
        v530 = *v521;
        v522 = strlen(v462);
        v462 = v530;
        v461 = v522;
      }
    }
  }
  if ( sub_C93CC0((__int64)v462, v461, 0, v599) || (v464 = v599[0], v599[0] != SLODWORD(v599[0])) )
    v464 = 0;
  v465 = 0;
  if ( v463 )
    v465 = *(_DWORD *)(v463 + 40);
  v466 = sub_22F59B0(*(_QWORD *)(a1 + 8), 77);
  *(_DWORD *)(a1 + 1600) = v464;
  v468 = v466;
  LODWORD(v466) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 1604) = v465;
  v50 = *(_BYTE *)(v468 + 44) == 0;
  *(_DWORD *)(a1 + 1608) = v466;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1612) = *(_DWORD *)(v468 + 40);
  }
  else
  {
    v534 = v467;
    if ( sub_22F59B0(v467, *(unsigned __int16 *)(v468 + 56)) )
      *(_DWORD *)(a1 + 1612) = *(_DWORD *)(sub_22F59B0(v534, *(unsigned __int16 *)(v468 + 56)) + 40);
    else
      *(_DWORD *)(a1 + 1612) = 0;
  }
  v469 = 0;
  v470 = 0;
  v471 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x4Eu);
  if ( v471 )
  {
    if ( *(_DWORD *)(v471 + 56) )
      v469 = **(_QWORD **)(v471 + 48);
    v470 = *(_DWORD *)(v471 + 40);
  }
  v472 = sub_22F59B0(*(_QWORD *)(a1 + 8), 78);
  v473 = *(_DWORD *)a1;
  v474 = v469;
  v595[1] = v475;
  LODWORD(v475) = v470;
  v476 = 0;
  v595[0] = v472;
  v477 = 0;
  sub_2261F60(a1 + 1616, v474, v475, v595, v473);
  v478 = sub_2262300(*(_QWORD *)(a1 + 8), 0x4Fu, "1");
  v479 = sub_22F59B0(*(_QWORD *)(a1 + 8), 79);
  v480 = *(_DWORD *)a1;
  v596[1] = v481;
  v596[0] = v479;
  sub_2261FD0(a1 + 1640, v478, v596, v480);
  v482 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x50u);
  if ( v482 )
  {
    if ( *(_DWORD *)(v482 + 56) )
      v476 = **(_QWORD **)(v482 + 48);
    v477 = *(_DWORD *)(v482 + 40);
  }
  v483 = sub_22F59B0(*(_QWORD *)(a1 + 8), 80);
  v484 = *(_DWORD *)a1;
  v597[1] = v485;
  v597[0] = v483;
  sub_2261F60(a1 + 1656, v476, v477, v597, v484);
  v486 = sub_2262300(*(_QWORD *)(a1 + 8), 0x51u, "0");
  v487 = sub_22F59B0(*(_QWORD *)(a1 + 8), 81);
  *(_BYTE *)(a1 + 1680) = v486;
  v489 = v487;
  v490 = v488;
  LODWORD(v487) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v489 + 44) == 0;
  *(_DWORD *)(a1 + 1684) = HIDWORD(v486);
  *(_DWORD *)(a1 + 1688) = v487;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1692) = *(_DWORD *)(v489 + 40);
  }
  else if ( sub_22F59B0(v488, *(unsigned __int16 *)(v489 + 56)) )
  {
    *(_DWORD *)(a1 + 1692) = *(_DWORD *)(sub_22F59B0(v490, *(unsigned __int16 *)(v489 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 1692) = 0;
  }
  v491 = 0;
  v492 = 0;
  v493 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x52u);
  if ( v493 )
  {
    if ( *(_DWORD *)(v493 + 56) )
      v491 = **(_QWORD **)(v493 + 48);
    v492 = *(_DWORD *)(v493 + 40);
  }
  v494 = sub_22F59B0(*(_QWORD *)(a1 + 8), 82);
  v495 = *(_DWORD *)a1;
  v598[1] = v496;
  v598[0] = v494;
  sub_2261F60(a1 + 1696, v491, v492, v598, v495);
  v497 = sub_2262300(*(_QWORD *)(a1 + 8), 0x53u, "0");
  v498 = sub_22F59B0(*(_QWORD *)(a1 + 8), 83);
  *(_BYTE *)(a1 + 1720) = v497;
  v500 = v498;
  v501 = v499;
  LODWORD(v498) = *(_DWORD *)a1;
  v50 = *(_BYTE *)(v500 + 44) == 0;
  *(_DWORD *)(a1 + 1724) = HIDWORD(v497);
  *(_DWORD *)(a1 + 1728) = v498;
  if ( v50 )
  {
    *(_DWORD *)(a1 + 1732) = *(_DWORD *)(v500 + 40);
  }
  else if ( sub_22F59B0(v499, *(unsigned __int16 *)(v500 + 56)) )
  {
    *(_DWORD *)(a1 + 1732) = *(_DWORD *)(sub_22F59B0(v501, *(unsigned __int16 *)(v500 + 56)) + 40);
  }
  else
  {
    *(_DWORD *)(a1 + 1732) = 0;
  }
  v502 = 0;
  v503 = 0;
  v504 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x54u);
  if ( v504 )
  {
    if ( *(_DWORD *)(v504 + 56) )
      v502 = **(_QWORD **)(v504 + 48);
    v503 = *(_DWORD *)(v504 + 40);
  }
  v505 = sub_22F59B0(*(_QWORD *)(a1 + 8), 84);
  v506 = *(_DWORD *)a1;
  v599[1] = v507;
  v599[0] = v505;
  sub_2261F60(a1 + 1736, v502, v503, v599, v506);
  v508 = sub_2262230(*(_QWORD *)(a1 + 8) + 184LL, 0x55u);
  v509 = 1;
  v510 = "0";
  v511 = v508;
  if ( v508 )
  {
    if ( *(_DWORD *)(v508 + 56) )
    {
      v519 = *(const char ***)(v508 + 48);
      v509 = 0;
      v510 = *v519;
      if ( *v519 )
      {
        v529 = *v519;
        v520 = strlen(v510);
        v510 = v529;
        v509 = v520;
      }
    }
  }
  if ( sub_C93CC0((__int64)v510, v509, 0, &v538) || (v512 = v538, v538 != (int)v538) )
    v512 = 0;
  v513 = 0;
  if ( v511 )
    v513 = *(_DWORD *)(v511 + 40);
  v514 = sub_22F59B0(*(_QWORD *)(a1 + 8), 85);
  *(_DWORD *)(a1 + 1760) = v512;
  v516 = v514;
  LODWORD(v514) = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 1764) = v513;
  v517 = v515;
  v50 = *(_BYTE *)(v516 + 44) == 0;
  *(_DWORD *)(a1 + 1768) = v514;
  if ( v50 )
  {
    result = *(unsigned int *)(v516 + 40);
    *(_DWORD *)(a1 + 1772) = result;
  }
  else
  {
    result = sub_22F59B0(v515, *(unsigned __int16 *)(v516 + 56));
    if ( result )
    {
      result = *(unsigned int *)(sub_22F59B0(v517, *(unsigned __int16 *)(v516 + 56)) + 40);
      *(_DWORD *)(a1 + 1772) = result;
    }
    else
    {
      *(_DWORD *)(a1 + 1772) = 0;
    }
  }
  return result;
}
