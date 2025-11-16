// Function: sub_376DE90
// Address: 0x376de90
//
unsigned __int64 __fastcall sub_376DE90(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r8
  __int64 *v8; // r14
  int v9; // edi
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  unsigned int i; // edx
  char *v13; // rax
  int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // r13
  _QWORD *v19; // r13
  _QWORD *v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 *v24; // rdx
  _BYTE *v25; // rdx
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  const __m128i *v29; // rcx
  unsigned __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rbx
  const __m128i *v34; // r13
  const __m128i *v35; // rbx
  __int16 v37; // ax
  bool v38; // al
  __int16 v39; // ax
  bool v40; // al
  __int16 v41; // ax
  bool v42; // al
  __int16 v43; // ax
  unsigned __int64 v44; // rcx
  __int64 v45; // r13
  __int64 v46; // rax
  signed __int64 v47; // rax
  __int64 v48; // r12
  bool v49; // al
  __int64 v50; // rax
  __int16 v51; // dx
  __int64 v52; // rax
  bool v53; // al
  __int64 v54; // rax
  __int16 v55; // dx
  __int64 v56; // rax
  bool v57; // al
  __int64 v58; // rax
  __int16 v59; // dx
  __int64 v60; // rax
  bool v61; // al
  __int64 v62; // rax
  __int16 v63; // dx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  int v67; // ebx
  signed int v68; // esi
  unsigned __int16 v69; // ax
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rsi
  unsigned __int64 v73; // r12
  __m128i *v74; // rax
  __m128i *v75; // rdx
  __m128i *j; // rdx
  unsigned int v77; // ebx
  __m128i *v78; // r9
  int v79; // r9d
  bool v80; // r8
  __int64 v81; // r14
  unsigned int *v82; // rdx
  __int64 v83; // rax
  unsigned __int16 v84; // di
  __int64 v85; // rax
  bool v86; // al
  __int64 v87; // rcx
  __int16 v88; // ax
  __int64 v89; // rdx
  __int16 v90; // ax
  __int64 v91; // r13
  int v92; // r9d
  unsigned __int32 v93; // r14d
  unsigned int v94; // esi
  __int64 v95; // r9
  __int64 v96; // rsi
  __m128i *v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rt2
  __int64 v100; // rdx
  unsigned int v101; // eax
  __int64 v102; // rsi
  __int64 *v103; // r12
  __int32 v104; // r15d
  __int64 v105; // rcx
  __int64 v106; // rax
  int v107; // ecx
  unsigned __int64 v108; // rbx
  __int32 v109; // edx
  __int32 v110; // r13d
  int v111; // edx
  int v112; // esi
  int v113; // r11d
  __m128i *v114; // rdi
  unsigned int ii; // r10d
  __int64 v116; // rax
  unsigned int v117; // r10d
  __int16 v118; // ax
  bool v119; // al
  __int16 v120; // ax
  bool v121; // al
  __int16 v122; // ax
  bool v123; // al
  __int64 v124; // rdx
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // rsi
  __int64 v128; // rdi
  __int64 v129; // rax
  __int64 (*v130)(); // rax
  __int64 v131; // rdx
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rdx
  int v135; // eax
  __int64 v136; // rdx
  unsigned __int64 v137; // r13
  unsigned int v138; // r12d
  __int64 v139; // r15
  int v140; // r14d
  unsigned __int64 v141; // rbx
  __int64 *v142; // rdx
  __int64 v143; // rdx
  __int64 v144; // rax
  __int64 v145; // rax
  bool v146; // al
  unsigned int v147; // eax
  __int64 v148; // rax
  bool v149; // al
  unsigned int v150; // eax
  __int16 v151; // r13
  __int64 v152; // rax
  bool v153; // al
  unsigned int v154; // eax
  __int16 v155; // r13
  __int64 v156; // rax
  bool v157; // al
  unsigned int v158; // eax
  __int16 v159; // r13
  __int64 v160; // rax
  unsigned int v161; // eax
  __int16 v162; // r13
  __int64 v163; // rax
  unsigned int v164; // eax
  __int16 v165; // r13
  __int64 v166; // rax
  bool v167; // al
  unsigned int v168; // eax
  __int64 v169; // rax
  bool v170; // al
  unsigned int v171; // eax
  __int64 v172; // rax
  unsigned int v173; // eax
  __int64 v174; // rax
  bool v175; // al
  unsigned int v176; // eax
  __int16 v177; // r13
  __int64 v178; // rax
  bool v179; // al
  unsigned int v180; // eax
  __int64 v181; // rax
  bool v182; // al
  unsigned int v183; // eax
  __int64 v184; // rax
  bool v185; // al
  unsigned int v186; // eax
  __int64 v187; // rax
  bool v188; // al
  unsigned int v189; // eax
  __int16 v190; // r13
  __int64 v191; // rax
  bool v192; // al
  unsigned int v193; // eax
  __int16 v194; // r13
  __int64 v195; // rax
  bool v196; // al
  unsigned int v197; // eax
  __int16 v198; // r13
  __int64 v199; // rax
  bool v200; // al
  unsigned int v201; // eax
  __int16 v202; // r13
  __int64 v203; // rax
  unsigned int v204; // eax
  __int16 v205; // r13
  __int64 v206; // rax
  bool v207; // al
  unsigned int v208; // eax
  __int16 v209; // r13
  __int64 v210; // rax
  bool v211; // al
  unsigned int v212; // eax
  __int16 v213; // r13
  __int64 v214; // rax
  bool v215; // al
  unsigned int v216; // eax
  __int16 v217; // r13
  __int64 v218; // rax
  unsigned int v219; // eax
  __int16 v220; // r13
  __int64 v221; // rax
  bool v222; // al
  unsigned int v223; // eax
  __int16 v224; // r13
  __int64 v225; // rax
  unsigned int v226; // eax
  __int16 v227; // r13
  __int64 v228; // rax
  bool v229; // al
  unsigned int v230; // eax
  __int16 v231; // r13
  __int64 v232; // rax
  bool v233; // al
  unsigned int v234; // eax
  __int64 v235; // rax
  bool v236; // al
  unsigned int v237; // eax
  __int64 v238; // rax
  unsigned int v239; // eax
  __int64 v240; // rax
  bool v241; // al
  unsigned int v242; // eax
  __int64 v243; // rax
  unsigned int v244; // eax
  __int64 v245; // rax
  unsigned int v246; // eax
  __int64 v247; // rax
  unsigned int v248; // eax
  __int64 v249; // rax
  unsigned int v250; // eax
  __int64 v251; // rax
  unsigned int v252; // eax
  __int64 v253; // rax
  unsigned int v254; // eax
  __int64 v255; // rax
  bool v256; // al
  unsigned int v257; // eax
  __int64 v258; // rax
  bool v259; // al
  unsigned int v260; // eax
  __int64 v261; // rax
  unsigned int v262; // eax
  unsigned __int16 v263; // r13
  __int64 v264; // rax
  char v265; // bl
  bool v266; // al
  __int64 v267; // rax
  __int64 v268; // r13
  __int64 v269; // rax
  __int64 v270; // rdx
  __int64 v271; // rax
  __int64 v272; // rcx
  __int64 v273; // rdx
  __int64 v274; // rax
  __int64 v275; // rcx
  __int64 v276; // rsi
  __int16 v277; // r13
  __int64 v278; // rax
  bool v279; // al
  unsigned int v280; // eax
  __int64 v281; // rax
  unsigned int v282; // eax
  __int64 v283; // rax
  unsigned int v284; // eax
  __int64 v285; // rax
  bool v286; // al
  unsigned int v287; // eax
  __int64 v288; // rax
  bool v289; // al
  unsigned int v290; // eax
  __int64 v291; // rax
  unsigned int v292; // eax
  __int64 v293; // rax
  bool v294; // al
  unsigned int v295; // eax
  __int64 v296; // rax
  bool v297; // al
  unsigned int v298; // eax
  __int64 v299; // rax
  unsigned int v300; // eax
  __int64 v301; // rax
  bool v302; // al
  unsigned int v303; // eax
  __int64 v304; // rax
  unsigned int v305; // eax
  __int64 v306; // rax
  bool v307; // al
  unsigned int v308; // eax
  __int64 v309; // rcx
  __int64 v310; // r13
  unsigned __int16 v311; // dx
  __int64 v312; // rcx
  __int64 v313; // rax
  __int16 v314; // bx
  __int64 v315; // rax
  bool v316; // al
  unsigned int v317; // eax
  __int16 v318; // r13
  __int64 v319; // rax
  bool v320; // al
  unsigned int v321; // eax
  __int16 v322; // r13
  __int64 v323; // rax
  unsigned int v324; // eax
  __int16 v325; // r13
  __int64 v326; // rax
  bool v327; // al
  unsigned int v328; // eax
  __int64 v329; // rax
  unsigned int v330; // eax
  __int64 v331; // rax
  bool v332; // al
  unsigned int v333; // eax
  __int64 v334; // rax
  unsigned int v335; // eax
  __int64 v336; // rax
  bool v337; // al
  unsigned int v338; // eax
  __int64 v339; // rax
  unsigned int v340; // eax
  __int64 v341; // rax
  bool v342; // al
  unsigned int v343; // eax
  __int64 v344; // rax
  unsigned int v345; // eax
  __int64 v346; // rax
  bool v347; // al
  unsigned int v348; // eax
  __int64 v349; // rax
  unsigned int v350; // eax
  __int64 v351; // rax
  bool v352; // al
  unsigned int v353; // eax
  __int64 v354; // rax
  unsigned int v355; // eax
  __int64 v356; // rax
  bool v357; // al
  unsigned int v358; // eax
  __int64 v359; // rax
  unsigned int v360; // eax
  __int64 v361; // rax
  bool v362; // al
  unsigned int v363; // eax
  __int64 v364; // rax
  unsigned int v365; // eax
  __int64 v366; // rax
  bool v367; // al
  unsigned int v368; // eax
  __int64 v369; // rax
  unsigned int v370; // eax
  __int64 v371; // rax
  bool v372; // al
  unsigned int v373; // eax
  __int64 v374; // rax
  unsigned int v375; // eax
  __int16 v376; // r13
  __int64 v377; // rax
  bool v378; // al
  unsigned int v379; // eax
  __int64 v380; // rax
  bool v381; // al
  unsigned int v382; // eax
  __int64 v383; // rax
  bool v384; // al
  unsigned int v385; // eax
  __int64 v386; // rax
  bool v387; // al
  unsigned int v388; // eax
  __int64 v389; // rax
  bool v390; // al
  unsigned int v391; // eax
  __int64 v392; // rax
  bool v393; // al
  unsigned int v394; // eax
  __int64 v395; // rax
  bool v396; // al
  unsigned int v397; // eax
  __int64 v398; // rax
  bool v399; // al
  unsigned int v400; // eax
  __int64 v401; // rax
  bool v402; // al
  unsigned int v403; // eax
  __int64 v404; // rax
  bool v405; // al
  unsigned int v406; // eax
  __int64 v407; // rax
  bool v408; // al
  unsigned int v409; // eax
  __int64 v410; // rax
  bool v411; // al
  unsigned int v412; // eax
  __int64 v413; // rax
  bool v414; // al
  unsigned int v415; // eax
  __int64 v416; // rax
  bool v417; // al
  unsigned int v418; // eax
  __int16 v419; // r13
  __int64 v420; // rax
  unsigned int v421; // eax
  __int16 v422; // r13
  __int64 v423; // rax
  bool v424; // al
  unsigned int v425; // eax
  __int64 v426; // rax
  unsigned int v427; // eax
  __int64 v428; // rax
  bool v429; // al
  unsigned int v430; // eax
  __int64 v431; // rax
  unsigned int v432; // eax
  __int64 v433; // rax
  bool v434; // al
  unsigned int v435; // eax
  __int64 v436; // rax
  unsigned int v437; // eax
  __int64 v438; // rax
  bool v439; // al
  unsigned int v440; // eax
  __int64 v441; // rax
  bool v442; // al
  unsigned int v443; // eax
  __int64 v444; // rax
  unsigned int v445; // eax
  __int64 v446; // rax
  bool v447; // al
  unsigned int v448; // eax
  __int64 v449; // rax
  unsigned int v450; // esi
  char v451; // di
  int v452; // edi
  _QWORD *v453; // r10
  int v454; // esi
  int v455; // r11d
  unsigned int v456; // eax
  __m128i *v457; // rdx
  unsigned int v458; // eax
  unsigned int v459; // esi
  unsigned int v460; // eax
  int v461; // r8d
  unsigned int v462; // edx
  __int32 v463; // r9d
  unsigned int v464; // eax
  int v465; // edx
  __int64 v466; // rsi
  __int64 v467; // rcx
  __int64 v468; // rax
  __int64 v469; // rdx
  unsigned __int16 *v470; // rax
  __int16 v471; // r12
  unsigned __int16 v472; // bx
  __int64 v473; // r9
  unsigned int v474; // r13d
  __int64 v475; // rsi
  __m128i *v476; // r12
  unsigned __int64 v477; // rdx
  __m128i *v478; // rax
  int v479; // r8d
  __m128i *v480; // rcx
  __m128i *n; // rdx
  unsigned __int8 *v482; // rax
  __int32 v483; // edx
  __int32 v484; // edi
  __int64 v485; // rdx
  __m128i *v486; // rax
  int v487; // r9d
  unsigned __int8 *v488; // rax
  __m128i *v489; // rcx
  __int32 v490; // edx
  __int64 v491; // rax
  __int64 v492; // rsi
  __int64 v493; // rax
  __int64 v494; // rax
  __int64 v495; // rax
  __int64 v496; // rdx
  unsigned __int64 v497; // rdx
  __int64 *v498; // rax
  __int64 v499; // rsi
  unsigned int v500; // eax
  unsigned int *v501; // rdx
  __int16 v502; // ax
  __int64 v503; // r8
  __int64 v504; // r9
  __int64 v505; // rsi
  unsigned __int64 v506; // r13
  __m128i *v507; // rax
  __m128i *v508; // rdx
  __m128i *m; // rdx
  int v510; // eax
  unsigned int v511; // r13d
  unsigned __int32 v512; // ebx
  unsigned __int64 v513; // r12
  bool v514; // al
  unsigned __int64 v515; // r9
  __m128i *v516; // rax
  unsigned __int64 v517; // r9
  unsigned int *v518; // r15
  __int64 v519; // rax
  __int16 v520; // si
  __int64 v521; // rax
  unsigned __int16 *v522; // rax
  unsigned __int16 v523; // bx
  unsigned __int16 v524; // r13
  int v525; // eax
  __int64 v526; // r9
  __int64 v527; // rcx
  __int64 v528; // rdx
  __int64 v529; // rsi
  _QWORD *v530; // rdi
  __int64 v531; // rax
  __m128i v532; // xmm6
  unsigned __int8 *v533; // r8
  unsigned int v534; // edx
  int v535; // eax
  int v536; // r9d
  _QWORD *v537; // rdi
  unsigned __int16 v538; // ax
  __int128 v539; // rax
  unsigned int v540; // edx
  int v541; // r9d
  unsigned __int8 *v542; // r10
  __int64 v543; // rax
  unsigned int v544; // edx
  unsigned __int64 v545; // r13
  unsigned __int64 v546; // rdx
  unsigned __int64 v547; // r11
  unsigned __int8 **v548; // rax
  __int64 v549; // rax
  unsigned __int16 v550; // ax
  __int64 v551; // r8
  __int64 v552; // r9
  __int64 v553; // rsi
  int v554; // eax
  unsigned __int64 v555; // rbx
  __m128i *v556; // rax
  __m128i *v557; // rdx
  __m128i *k; // rdx
  unsigned __int64 v559; // r14
  unsigned int v560; // r15d
  __m128i *v561; // r9
  __int64 v562; // rcx
  __int64 v563; // rbx
  unsigned int *v564; // r12
  __int64 v565; // rax
  __int16 v566; // di
  __int64 v567; // rax
  bool v568; // al
  bool v569; // al
  __int64 v570; // r9
  __int64 v571; // rcx
  __int64 v572; // rax
  _QWORD *v573; // rdi
  const __m128i *v574; // roff
  __m128i v575; // xmm6
  unsigned __int8 *v576; // rax
  __int64 v577; // r8
  __m128i *v578; // r9
  unsigned __int8 *v579; // r12
  __int64 v580; // rax
  unsigned __int64 v581; // r9
  unsigned __int64 v582; // rdx
  unsigned __int8 **v583; // rax
  unsigned __int8 *v584; // rax
  unsigned __int32 v585; // edx
  __int64 v586; // rdx
  __m128i *v587; // rax
  _QWORD *v588; // rsi
  int v589; // eax
  int v590; // r10d
  int v591; // r8d
  int v592; // eax
  int v593; // r10d
  unsigned int jj; // esi
  unsigned int v595; // esi
  _QWORD *v596; // rdi
  int v597; // edx
  int v598; // r10d
  unsigned int kk; // eax
  unsigned int v600; // eax
  _QWORD *v601; // rdi
  int v602; // edx
  int v603; // r10d
  unsigned int mm; // eax
  unsigned int v605; // eax
  int v606; // r8d
  int v607; // eax
  int v608; // eax
  __int64 v609; // rax
  __int16 v610; // cx
  __int64 v611; // rax
  int v613; // edx
  int v614; // edx
  unsigned int v615; // eax
  unsigned __int8 *v616; // rax
  __m128i *v617; // r9
  __int32 v618; // edx
  __int64 v619; // rcx
  __int64 v620; // rdx
  __int64 v621; // rax
  __int64 v622; // rdx
  __int64 v623; // rbx
  __int64 v624; // r9
  unsigned __int8 *v625; // rax
  __int32 v626; // edx
  __int32 v627; // edi
  __int64 v628; // rdx
  __m128i *v629; // rax
  unsigned __int8 *v630; // rax
  _QWORD *v631; // rbx
  unsigned __int8 *v632; // rax
  __int64 v633; // rdx
  __int64 v634; // r9
  unsigned __int8 *v635; // rax
  unsigned __int8 *v636; // rbx
  __int64 v637; // rax
  unsigned __int64 v638; // rdx
  unsigned __int8 **v639; // rax
  __int64 v640; // rax
  unsigned __int8 **v641; // rax
  int v642; // r9d
  unsigned __int8 *v643; // r12
  __int64 v644; // rdx
  __int64 v645; // r13
  __int64 v646; // rbx
  __int64 v647; // rax
  _QWORD *v648; // rdi
  __int64 v649; // r9
  unsigned int v650; // edx
  __int64 v651; // r12
  unsigned __int64 v652; // r13
  __int64 v653; // rax
  unsigned __int64 v654; // rdx
  __int64 *v655; // rax
  __int64 v656; // rax
  _QWORD *v657; // rdi
  unsigned __int16 v658; // si
  __int64 v659; // rax
  __int64 v660; // rdx
  unsigned __int64 v661; // rdx
  __int64 *v662; // rax
  __m128i *v663; // rdi
  __int16 v664; // ax
  __int16 v665; // ax
  unsigned int v666; // edx
  __int64 v667; // rax
  __int16 v668; // cx
  __int64 v669; // rax
  bool v670; // al
  __int64 v671; // rax
  __int16 v672; // cx
  __int64 v673; // rax
  bool v674; // al
  unsigned __int8 *v675; // rax
  unsigned int v676; // edx
  unsigned __int8 *v677; // rax
  unsigned __int8 *v678; // rbx
  __int64 v679; // rax
  unsigned __int8 *v680; // rdx
  unsigned __int8 *v681; // r13
  unsigned __int64 v682; // rdx
  unsigned __int8 **v683; // rax
  __int64 v684; // rax
  unsigned __int8 **v685; // rax
  unsigned __int8 **v686; // rax
  _QWORD *v687; // rbx
  __int128 v688; // rax
  __int64 v689; // r9
  __int64 v690; // r8
  unsigned __int16 v691; // si
  __int64 v692; // rcx
  __int64 v693; // rdi
  __int64 v694; // rdi
  __int32 v695; // edx
  __int32 v696; // edx
  __int32 v697; // esi
  unsigned int v698; // eax
  unsigned __int8 *v699; // rsi
  __m128i *v700; // rax
  __int32 v701; // edx
  __int32 v702; // esi
  __int128 v703; // [rsp-30h] [rbp-3C0h]
  __int128 v704; // [rsp-20h] [rbp-3B0h]
  __int128 v705; // [rsp-20h] [rbp-3B0h]
  __int128 v706; // [rsp-10h] [rbp-3A0h]
  __int128 v707; // [rsp-10h] [rbp-3A0h]
  __int128 v708; // [rsp-10h] [rbp-3A0h]
  __int128 v709; // [rsp-10h] [rbp-3A0h]
  __int64 v710; // [rsp-10h] [rbp-3A0h]
  __int128 v711; // [rsp-10h] [rbp-3A0h]
  __int128 v712; // [rsp-10h] [rbp-3A0h]
  __int64 v713; // [rsp-10h] [rbp-3A0h]
  __int64 v714; // [rsp-10h] [rbp-3A0h]
  __int64 v715; // [rsp-8h] [rbp-398h]
  __int64 v716; // [rsp-8h] [rbp-398h]
  __int16 v717; // [rsp+Ah] [rbp-386h]
  unsigned __int16 v718; // [rsp+10h] [rbp-380h]
  __int16 v719; // [rsp+12h] [rbp-37Eh]
  __int64 *v720; // [rsp+20h] [rbp-370h]
  __int16 v721; // [rsp+22h] [rbp-36Eh]
  __int64 *v722; // [rsp+28h] [rbp-368h]
  unsigned __int16 v723; // [rsp+28h] [rbp-368h]
  __int16 v724; // [rsp+2Ah] [rbp-366h]
  unsigned __int16 v725; // [rsp+30h] [rbp-360h]
  bool v726; // [rsp+30h] [rbp-360h]
  char v727; // [rsp+30h] [rbp-360h]
  unsigned __int64 v728; // [rsp+38h] [rbp-358h]
  unsigned __int16 v729; // [rsp+38h] [rbp-358h]
  __int16 v730; // [rsp+38h] [rbp-358h]
  unsigned __int8 *v731; // [rsp+38h] [rbp-358h]
  bool v732; // [rsp+40h] [rbp-350h]
  unsigned int v733; // [rsp+40h] [rbp-350h]
  unsigned int v734; // [rsp+40h] [rbp-350h]
  __int64 v735; // [rsp+40h] [rbp-350h]
  unsigned __int64 v736; // [rsp+40h] [rbp-350h]
  unsigned __int64 v737; // [rsp+48h] [rbp-348h]
  const __m128i *v738; // [rsp+48h] [rbp-348h]
  unsigned __int8 *v739; // [rsp+48h] [rbp-348h]
  __int64 v740; // [rsp+50h] [rbp-340h]
  __int64 *v741; // [rsp+50h] [rbp-340h]
  __int64 *v742; // [rsp+50h] [rbp-340h]
  const __m128i *v743; // [rsp+70h] [rbp-320h]
  __int32 v744; // [rsp+70h] [rbp-320h]
  unsigned int v745; // [rsp+70h] [rbp-320h]
  unsigned int v746; // [rsp+70h] [rbp-320h]
  unsigned int *v747; // [rsp+70h] [rbp-320h]
  int v748; // [rsp+70h] [rbp-320h]
  unsigned __int8 *v749; // [rsp+70h] [rbp-320h]
  const __m128i *v750; // [rsp+80h] [rbp-310h]
  __int64 v751; // [rsp+80h] [rbp-310h]
  __int64 v752; // [rsp+80h] [rbp-310h]
  int v753; // [rsp+80h] [rbp-310h]
  unsigned __int64 v754; // [rsp+80h] [rbp-310h]
  unsigned __int16 v755; // [rsp+80h] [rbp-310h]
  bool v756; // [rsp+80h] [rbp-310h]
  unsigned __int64 v757; // [rsp+80h] [rbp-310h]
  int v758; // [rsp+80h] [rbp-310h]
  int v759; // [rsp+80h] [rbp-310h]
  unsigned __int64 v760; // [rsp+80h] [rbp-310h]
  __m128i *v761; // [rsp+80h] [rbp-310h]
  __m128i *v762; // [rsp+80h] [rbp-310h]
  __int64 v763; // [rsp+88h] [rbp-308h]
  __int64 v764; // [rsp+88h] [rbp-308h]
  __int64 v765; // [rsp+88h] [rbp-308h]
  __int64 v768; // [rsp+150h] [rbp-240h] BYREF
  int v769; // [rsp+158h] [rbp-238h]
  __m128i v770; // [rsp+160h] [rbp-230h] BYREF
  __m128i v771; // [rsp+170h] [rbp-220h]
  _BYTE *v772; // [rsp+180h] [rbp-210h] BYREF
  __int64 v773; // [rsp+188h] [rbp-208h]
  _BYTE v774[32]; // [rsp+190h] [rbp-200h] BYREF
  __m128i v775; // [rsp+1B0h] [rbp-1E0h] BYREF
  __m128i v776; // [rsp+1C0h] [rbp-1D0h]
  unsigned __int8 *v777; // [rsp+1D0h] [rbp-1C0h]
  __int64 v778; // [rsp+1D8h] [rbp-1B8h]
  __m128i *v779; // [rsp+1E0h] [rbp-1B0h] BYREF
  __int64 v780; // [rsp+1E8h] [rbp-1A8h]
  __m128i v781; // [rsp+1F0h] [rbp-1A0h] BYREF
  _BYTE *v782; // [rsp+240h] [rbp-150h] BYREF
  __int64 v783; // [rsp+248h] [rbp-148h]
  _BYTE v784[128]; // [rsp+250h] [rbp-140h] BYREF
  __m128i v785; // [rsp+2D0h] [rbp-C0h] BYREF
  _BYTE v786[176]; // [rsp+2E0h] [rbp-B0h] BYREF

  v7 = a3;
  v8 = (__int64 *)a1;
  v9 = *(_BYTE *)(a1 + 32) & 1;
  if ( v9 )
  {
    v10 = v8 + 5;
    v11 = 63;
  }
  else
  {
    v15 = *((unsigned int *)v8 + 12);
    v10 = (_QWORD *)v8[5];
    if ( !(_DWORD)v15 )
    {
LABEL_43:
      v11 = 32 * v15;
LABEL_44:
      v13 = (char *)v10 + v11;
      goto LABEL_10;
    }
    v11 = (unsigned int)(v15 - 1);
  }
  a7 = 1;
  for ( i = v11 & (a3 + ((a2 >> 9) ^ (a2 >> 4))); ; i = v11 & v14 )
  {
    v13 = (char *)&v10[4 * i];
    if ( *(_QWORD *)v13 == a2 && *((_DWORD *)v13 + 2) == (_DWORD)v7 )
      break;
    if ( !*(_QWORD *)v13 && *((_DWORD *)v13 + 2) == -1 )
    {
      if ( !(_BYTE)v9 )
      {
        v15 = *((unsigned int *)v8 + 12);
        goto LABEL_43;
      }
      v11 = 2048;
      goto LABEL_44;
    }
    v14 = a7 + i;
    a7 = (unsigned int)(a7 + 1);
  }
LABEL_10:
  v16 = 256;
  if ( !(_BYTE)v9 )
    v16 = 4LL * *((unsigned int *)v8 + 12);
  if ( v13 != (char *)&v10[v16] )
    return *((_QWORD *)v13 + 2);
  v782 = v784;
  v783 = 0x800000000LL;
  v19 = *(_QWORD **)(a2 + 40);
  v20 = &v19[5 * *(unsigned int *)(a2 + 64)];
  if ( v19 == v20 )
  {
    v25 = v784;
    v11 = 0;
  }
  else
  {
    do
    {
      v21 = sub_376DE90(v8, *v19, v19[1], v11, v7, a7);
      a7 = v22;
      v23 = (unsigned int)v783;
      v7 = v21;
      if ( (unsigned __int64)(unsigned int)v783 + 1 > HIDWORD(v783) )
      {
        v751 = v21;
        v763 = a7;
        sub_C8D5F0((__int64)&v782, v784, (unsigned int)v783 + 1LL, 0x10u, v21, a7);
        v23 = (unsigned int)v783;
        v7 = v751;
        a7 = v763;
      }
      v24 = (__int64 *)&v782[16 * v23];
      v19 += 5;
      *v24 = v7;
      v24[1] = a7;
      v11 = (unsigned int)(v783 + 1);
      LODWORD(v783) = v783 + 1;
    }
    while ( v20 != v19 );
    v25 = v782;
  }
  v26 = sub_33EC210((_QWORD *)*v8, (__int64 *)a2, (__int64)v25, v11);
  v29 = (const __m128i *)v26[6];
  v30 = (unsigned __int64)v26;
  v31 = 16LL * *((unsigned int *)v26 + 17);
  v743 = v29;
  v750 = &v29[(unsigned __int64)v31 / 0x10];
  v32 = v31 >> 4;
  v33 = v31 >> 6;
  if ( v33 )
  {
    v34 = v29;
    v35 = &v29[4 * v33];
    while ( 1 )
    {
      v43 = v34->m128i_i16[0];
      a4 = _mm_loadu_si128(v34);
      v785 = a4;
      if ( v43 ? (unsigned __int16)(v43 - 17) <= 0xD3u : sub_30070B0((__int64)&v785) )
        goto LABEL_40;
      v37 = v34[1].m128i_i16[0];
      v785 = _mm_loadu_si128(v34 + 1);
      if ( v37 )
        v38 = (unsigned __int16)(v37 - 17) <= 0xD3u;
      else
        v38 = sub_30070B0((__int64)&v785);
      if ( v38 )
      {
        ++v34;
        goto LABEL_40;
      }
      v39 = v34[2].m128i_i16[0];
      v785 = _mm_loadu_si128(v34 + 2);
      if ( v39 )
        v40 = (unsigned __int16)(v39 - 17) <= 0xD3u;
      else
        v40 = sub_30070B0((__int64)&v785);
      if ( v40 )
      {
        v34 += 2;
        goto LABEL_40;
      }
      v41 = v34[3].m128i_i16[0];
      v785 = _mm_loadu_si128(v34 + 3);
      if ( v41 )
        v42 = (unsigned __int16)(v41 - 17) <= 0xD3u;
      else
        v42 = sub_30070B0((__int64)&v785);
      if ( v42 )
      {
        v34 += 3;
        goto LABEL_40;
      }
      v34 += 4;
      if ( v35 == v34 )
      {
        v32 = v750 - v34;
        goto LABEL_46;
      }
    }
  }
  v34 = v29;
LABEL_46:
  if ( v32 == 2 )
    goto LABEL_140;
  if ( v32 == 3 )
  {
    v118 = v34->m128i_i16[0];
    v785 = _mm_loadu_si128(v34);
    if ( v118 )
      v119 = (unsigned __int16)(v118 - 17) <= 0xD3u;
    else
      v119 = sub_30070B0((__int64)&v785);
    if ( v119 )
      goto LABEL_40;
    ++v34;
LABEL_140:
    v120 = v34->m128i_i16[0];
    v785 = _mm_loadu_si128(v34);
    if ( v120 )
      v121 = (unsigned __int16)(v120 - 17) <= 0xD3u;
    else
      v121 = sub_30070B0((__int64)&v785);
    if ( v121 )
      goto LABEL_40;
    ++v34;
    goto LABEL_144;
  }
  if ( v32 != 1 )
    goto LABEL_49;
LABEL_144:
  v122 = v34->m128i_i16[0];
  v785 = _mm_loadu_si128(v34);
  if ( v122 )
    v123 = (unsigned __int16)(v122 - 17) <= 0xD3u;
  else
    v123 = sub_30070B0((__int64)&v785);
  if ( !v123 )
    goto LABEL_49;
LABEL_40:
  if ( v750 == v34 )
  {
LABEL_49:
    v45 = *(_QWORD *)(v30 + 40);
    v46 = 40LL * *(unsigned int *)(v30 + 64);
    v752 = v45 + v46;
    v47 = 0xCCCCCCCCCCCCCCCDLL * (v46 >> 3);
    if ( !(v47 >> 2) )
      goto LABEL_874;
    v48 = v45 + 160 * (v47 >> 2);
    do
    {
      v62 = *(_QWORD *)(*(_QWORD *)v45 + 48LL) + 16LL * *(unsigned int *)(v45 + 8);
      v63 = *(_WORD *)v62;
      v64 = *(_QWORD *)(v62 + 8);
      v785.m128i_i16[0] = v63;
      v785.m128i_i64[1] = v64;
      if ( v63 )
        v49 = (unsigned __int16)(v63 - 17) <= 0xD3u;
      else
        v49 = sub_30070B0((__int64)&v785);
      if ( v49 )
        goto LABEL_69;
      v50 = *(_QWORD *)(*(_QWORD *)(v45 + 40) + 48LL) + 16LL * *(unsigned int *)(v45 + 48);
      v51 = *(_WORD *)v50;
      v52 = *(_QWORD *)(v50 + 8);
      v785.m128i_i16[0] = v51;
      v785.m128i_i64[1] = v52;
      if ( v51 )
        v53 = (unsigned __int16)(v51 - 17) <= 0xD3u;
      else
        v53 = sub_30070B0((__int64)&v785);
      if ( v53 )
      {
        v45 += 40;
        goto LABEL_69;
      }
      v54 = *(_QWORD *)(*(_QWORD *)(v45 + 80) + 48LL) + 16LL * *(unsigned int *)(v45 + 88);
      v55 = *(_WORD *)v54;
      v56 = *(_QWORD *)(v54 + 8);
      v785.m128i_i16[0] = v55;
      v785.m128i_i64[1] = v56;
      if ( v55 )
        v57 = (unsigned __int16)(v55 - 17) <= 0xD3u;
      else
        v57 = sub_30070B0((__int64)&v785);
      if ( v57 )
      {
        v45 += 80;
        goto LABEL_69;
      }
      v58 = *(_QWORD *)(*(_QWORD *)(v45 + 120) + 48LL) + 16LL * *(unsigned int *)(v45 + 128);
      v59 = *(_WORD *)v58;
      v60 = *(_QWORD *)(v58 + 8);
      v785.m128i_i16[0] = v59;
      v785.m128i_i64[1] = v60;
      if ( v59 )
        v61 = (unsigned __int16)(v59 - 17) <= 0xD3u;
      else
        v61 = sub_30070B0((__int64)&v785);
      if ( v61 )
      {
        v45 += 120;
        goto LABEL_69;
      }
      v45 += 160;
    }
    while ( v48 != v45 );
    v47 = 0xCCCCCCCCCCCCCCCDLL * ((v752 - v45) >> 3);
LABEL_874:
    if ( v47 != 2 )
    {
      if ( v47 != 3 )
      {
        if ( v47 == 1 )
          goto LABEL_877;
        goto LABEL_70;
      }
      v667 = *(_QWORD *)(*(_QWORD *)v45 + 48LL) + 16LL * *(unsigned int *)(v45 + 8);
      v668 = *(_WORD *)v667;
      v669 = *(_QWORD *)(v667 + 8);
      v785.m128i_i16[0] = v668;
      v785.m128i_i64[1] = v669;
      if ( v668 )
        v670 = (unsigned __int16)(v668 - 17) <= 0xD3u;
      else
        v670 = sub_30070B0((__int64)&v785);
      if ( v670 )
        goto LABEL_69;
      v45 += 40;
    }
    v671 = *(_QWORD *)(*(_QWORD *)v45 + 48LL) + 16LL * *(unsigned int *)(v45 + 8);
    v672 = *(_WORD *)v671;
    v673 = *(_QWORD *)(v671 + 8);
    v785.m128i_i16[0] = v672;
    v785.m128i_i64[1] = v673;
    if ( v672 )
      v674 = (unsigned __int16)(v672 - 17) <= 0xD3u;
    else
      v674 = sub_30070B0((__int64)&v785);
    if ( !v674 )
    {
      v45 += 40;
LABEL_877:
      v609 = *(_QWORD *)(*(_QWORD *)v45 + 48LL) + 16LL * *(unsigned int *)(v45 + 8);
      v610 = *(_WORD *)v609;
      v611 = *(_QWORD *)(v609 + 8);
      v785.m128i_i16[0] = v610;
      v785.m128i_i64[1] = v611;
      if ( !(v610 ? (unsigned __int16)(v610 - 17) <= 0xD3u : sub_30070B0((__int64)&v785)) )
      {
LABEL_70:
        v17 = sub_376D860((__int64)v8, a2, a3, v30);
        goto LABEL_71;
      }
    }
LABEL_69:
    if ( v752 == v45 )
      goto LABEL_70;
  }
  v44 = *(unsigned int *)(a2 + 24);
  switch ( (int)v44 )
  {
    case 55:
      v270 = *(unsigned int *)(v30 + 24);
      if ( (unsigned int)v270 > 0x1F3 )
        goto LABEL_833;
      v271 = v743->m128i_u16[0];
      if ( (_WORD)v271 )
      {
        LOBYTE(v67) = *(_BYTE *)(v270 + v8[1] + 500 * v271 + 6414);
        if ( (_BYTE)v67 )
          goto LABEL_79;
      }
      goto LABEL_125;
    case 56:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 76:
    case 77:
    case 78:
    case 79:
    case 80:
    case 81:
    case 82:
    case 83:
    case 84:
    case 85:
    case 86:
    case 87:
    case 96:
    case 97:
    case 98:
    case 99:
    case 100:
    case 149:
    case 150:
    case 152:
    case 154:
    case 171:
    case 172:
    case 173:
    case 174:
    case 175:
    case 176:
    case 177:
    case 178:
    case 179:
    case 180:
    case 181:
    case 182:
    case 183:
    case 184:
    case 185:
    case 186:
    case 187:
    case 188:
    case 189:
    case 190:
    case 191:
    case 192:
    case 193:
    case 194:
    case 195:
    case 196:
    case 197:
    case 198:
    case 199:
    case 200:
    case 201:
    case 203:
    case 204:
    case 205:
    case 206:
    case 207:
    case 213:
    case 214:
    case 215:
    case 216:
    case 222:
    case 223:
    case 224:
    case 225:
    case 226:
    case 227:
    case 228:
    case 229:
    case 230:
    case 233:
    case 244:
    case 245:
    case 246:
    case 248:
    case 249:
    case 250:
    case 251:
    case 252:
    case 253:
    case 254:
    case 255:
    case 256:
    case 257:
    case 258:
    case 259:
    case 260:
    case 261:
    case 262:
    case 263:
    case 264:
    case 265:
    case 266:
    case 267:
    case 268:
    case 269:
    case 270:
    case 271:
    case 272:
    case 273:
    case 274:
    case 279:
    case 280:
    case 281:
    case 282:
    case 283:
    case 284:
    case 285:
    case 286:
    case 287:
    case 288:
    case 289:
    case 364:
    case 391:
    case 392:
      v65 = *(unsigned int *)(v30 + 24);
      if ( (unsigned int)v65 > 0x1F3 )
        goto LABEL_833;
      v66 = v743->m128i_u16[0];
      if ( !(_WORD)v66 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v65 + v8[1] + 500 * v66 + 6414);
      goto LABEL_79;
    case 88:
    case 89:
    case 90:
    case 91:
    case 92:
    case 93:
    case 94:
    case 95:
      v126 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 80LL) + 96LL);
      v27 = *(_QWORD *)(v126 + 24);
      if ( *(_DWORD *)(v126 + 32) > 0x40u )
        v27 = *(_QWORD *)v27;
      v127 = *(unsigned int *)(v30 + 24);
      v128 = v8[1];
      v129 = v743->m128i_u16[0];
      v44 = v743->m128i_u64[1];
      if ( (unsigned int)v127 > 0x1F3 )
      {
        LOBYTE(v67) = 4;
      }
      else
      {
        if ( !(_WORD)v129 )
          goto LABEL_158;
        v28 = (unsigned int)v127;
        LOBYTE(v67) = *(_BYTE *)((unsigned int)v127 + v128 + 500 * v129 + 6414);
        if ( !(_BYTE)v67 )
        {
          if ( (unsigned int)(v127 - 88) > 7 )
            goto LABEL_1032;
          v130 = *(__int64 (**)())(*(_QWORD *)v128 + 656LL);
          if ( v130 == sub_2FE31B0
            || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, unsigned __int64, __int64))v130)(
                  v128,
                  v127,
                  v743->m128i_u16[0],
                  v44,
                  v27) )
          {
LABEL_158:
            LOBYTE(v67) = 2;
          }
        }
      }
      goto LABEL_79;
    case 101:
    case 102:
    case 103:
    case 104:
    case 105:
    case 106:
    case 107:
    case 108:
    case 109:
    case 110:
    case 111:
    case 112:
    case 113:
    case 114:
    case 115:
    case 116:
    case 117:
    case 118:
    case 119:
    case 120:
    case 121:
    case 122:
    case 123:
    case 124:
    case 125:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 132:
    case 133:
    case 134:
    case 135:
    case 136:
    case 137:
    case 138:
    case 139:
    case 140:
    case 141:
    case 142:
    case 143:
    case 144:
    case 145:
    case 146:
    case 147:
    case 148:
      v100 = v8[1];
      if ( (unsigned int)(v44 - 143) <= 1 )
      {
        v101 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                                   + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
        goto LABEL_120;
      }
      v101 = v743->m128i_u16[0];
      if ( (unsigned int)(v44 - 147) > 1 )
      {
LABEL_120:
        v44 = *(unsigned int *)(v30 + 24);
        if ( (unsigned int)v44 <= 0x1F3 )
        {
          if ( !(_WORD)v101 )
          {
LABEL_124:
            if ( *(_BYTE *)(v100 + 537006) )
            {
LABEL_125:
              v785.m128i_i64[0] = (__int64)v786;
              v785.m128i_i64[1] = 0x800000000LL;
LABEL_126:
              sub_376AE20(v8, v30, (__int64)&v785, v44, v27, v28, a4);
              goto LABEL_127;
            }
            v466 = *(unsigned int *)(v30 + 24);
            v467 = (unsigned int)(v466 - 101);
            if ( (unsigned int)v467 <= 0x2F )
            {
              v44 = (unsigned __int16)aAbcd_1[v467];
              if ( (_WORD)v101 )
              {
                if ( !*(_BYTE *)(v44 + v100 + 500LL * v101 + 6414) )
                {
                  v468 = (unsigned __int16)word_4456580[v101 - 1];
                  if ( (_WORD)v468 )
                  {
                    v469 = 500 * v468 + v100;
                    if ( *(_BYTE *)(v466 + v469 + 6414) == 2 && !*(_BYTE *)(v44 + v469 + 6414) )
                    {
LABEL_732:
                      v785.m128i_i64[0] = (__int64)v786;
                      v785.m128i_i64[1] = 0x800000000LL;
                      goto LABEL_733;
                    }
                  }
                }
              }
              goto LABEL_125;
            }
            goto LABEL_1032;
          }
          v102 = (unsigned int)v44;
          v44 = v100 + 500LL * (unsigned __int16)v101;
          LOBYTE(v67) = *(_BYTE *)(v102 + v44 + 6414);
          goto LABEL_123;
        }
LABEL_833:
        v785.m128i_i64[0] = (__int64)v786;
        v785.m128i_i64[1] = 0x800000000LL;
        goto LABEL_166;
      }
      v690 = *(_QWORD *)(v30 + 40);
      v691 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v690 + 40) + 48LL) + 16LL * *(unsigned int *)(v690 + 48));
      v692 = *(int *)(*(_QWORD *)(v690 + 120) + 96LL);
      v27 = 9 * v692;
      v693 = (v691 >> 3) + 36 * v692 - v692;
      v44 = 4 * (v691 & 7u);
      v67 = (*(_DWORD *)(v100 + 4 * v693 + 521536) >> (4 * (v691 & 7))) & 0xF;
      if ( !v67 )
      {
        v44 = *(unsigned int *)(v30 + 24);
        if ( (unsigned int)v44 > 0x1F3 )
          goto LABEL_833;
        if ( !v691 )
          goto LABEL_124;
        v694 = (unsigned int)v44;
        v44 = v100 + 500LL * v691;
        LOBYTE(v67) = *(_BYTE *)(v694 + v44 + 6414);
      }
LABEL_123:
      if ( (_BYTE)v67 == 2 )
        goto LABEL_124;
LABEL_79:
      v785.m128i_i64[0] = (__int64)v786;
      v785.m128i_i64[1] = 0x800000000LL;
      if ( (_BYTE)v67 == 1 )
      {
        v68 = *(_DWORD *)(v30 + 24);
        if ( v68 <= 233 )
        {
          if ( v68 > 207 )
          {
            switch ( v68 )
            {
              case 208:
                goto LABEL_739;
              case 220:
              case 221:
                v500 = v68 - 143;
                goto LABEL_765;
              case 226:
              case 227:
                goto LABEL_786;
              case 230:
              case 233:
                goto LABEL_1032;
              default:
                goto LABEL_86;
            }
          }
          if ( v68 <= 142 )
          {
            if ( v68 <= 140 )
            {
              if ( v68 > 104 )
              {
                if ( (unsigned int)(v68 - 106) > 1 )
                  goto LABEL_86;
              }
              else if ( v68 <= 100 )
              {
LABEL_86:
                v725 = **(_WORD **)(v30 + 48);
                v69 = sub_332CC60(v8[1], v68, v725);
                v72 = *(_QWORD *)(v30 + 80);
                v729 = v69;
                v770.m128i_i64[0] = v72;
                if ( v72 )
                  sub_B96E90((__int64)&v770, v72, 1);
                v770.m128i_i32[2] = *(_DWORD *)(v30 + 72);
                v73 = *(unsigned int *)(v30 + 64);
                v74 = &v781;
                v75 = &v781;
                v779 = &v781;
                v780 = 0x400000000LL;
                if ( v73 )
                {
                  if ( v73 > 4 )
                  {
                    v72 = (__int64)&v781;
                    sub_C8D5F0((__int64)&v779, &v781, v73, 0x10u, v70, v71);
                    v75 = v779;
                    v74 = &v779[(unsigned int)v780];
                  }
                  for ( j = &v75[v73]; j != v74; ++v74 )
                  {
                    if ( v74 )
                    {
                      v74->m128i_i64[0] = 0;
                      v74->m128i_i32[2] = 0;
                    }
                  }
                  LODWORD(v780) = v73;
                }
                v77 = 0;
                if ( *(_DWORD *)(v30 + 64) )
                {
                  v722 = v8;
                  while ( 1 )
                  {
                    v80 = sub_33CB110(*(_DWORD *)(v30 + 24))
                       && (v775.m128i_i64[0] = sub_33CB160(*(_DWORD *)(v30 + 24)), v775.m128i_i8[4])
                       && v775.m128i_i32[0] == v77;
                    v81 = 40LL * v77;
                    v82 = (unsigned int *)(v81 + *(_QWORD *)(v30 + 40));
                    v83 = *(_QWORD *)(*(_QWORD *)v82 + 48LL) + 16LL * v82[2];
                    v84 = *(_WORD *)v83;
                    v85 = *(_QWORD *)(v83 + 8);
                    v775.m128i_i16[0] = v84;
                    v775.m128i_i64[1] = v85;
                    if ( v84 )
                    {
                      if ( (unsigned __int16)(v84 - 17) <= 0xD3u && !v80 )
                      {
                        v88 = word_4456580[v84 - 1];
                        v89 = 0;
LABEL_108:
                        LOWORD(v772) = v88;
                        v773 = v89;
                        if ( v88 )
                        {
                          if ( (unsigned __int16)(v88 - 10) <= 6u
                            || (unsigned __int16)(v88 - 126) <= 0x31u
                            || (unsigned __int16)(v88 - 208) <= 0x14u )
                          {
                            goto LABEL_112;
                          }
                        }
                        else if ( (unsigned __int8)sub_3007030((__int64)&v772) )
                        {
LABEL_112:
                          if ( (unsigned __int16)(v729 - 17) <= 0xD3u )
                          {
                            v90 = word_4456580[v729 - 1];
                            if ( (unsigned __int16)(v90 - 10) <= 6u
                              || (unsigned __int16)(v90 - 126) <= 0x31u
                              || (unsigned __int16)(v90 - 208) <= 0x14u )
                            {
                              v91 = v77;
                              if ( sub_33CB110(*(_DWORD *)(v30 + 24)) )
                              {
                                v775.m128i_i64[0] = sub_33CB1F0(*(_DWORD *)(v30 + 24));
                                v93 = v775.m128i_i32[0];
                                v775.m128i_i64[0] = sub_33CB160(*(_DWORD *)(v30 + 24));
                                HIWORD(v94) = v717;
                                LOWORD(v94) = v729;
                                v96 = sub_340F900(
                                        (_QWORD *)*v722,
                                        0x1C9u,
                                        (__int64)&v770,
                                        v94,
                                        0,
                                        v95,
                                        *(_OWORD *)(*(_QWORD *)(v30 + 40) + 40LL * v77),
                                        *(_OWORD *)(*(_QWORD *)(v30 + 40) + 40LL * v775.m128i_u32[0]),
                                        *(_OWORD *)(*(_QWORD *)(v30 + 40) + 40LL * v93));
                                v97 = v779;
                                v99 = v96;
                                v72 = v98;
                                v779[v77].m128i_i64[0] = v99;
                                v97[v91].m128i_i32[2] = v98;
                              }
                              else
                              {
                                HIWORD(v698) = v719;
                                LOWORD(v698) = v729;
                                v714 = *(_QWORD *)(v81 + *(_QWORD *)(v30 + 40));
                                v699 = sub_33FAF80(*v722, 233, (__int64)&v770, v698, 0, v92, a4);
                                v700 = v779;
                                v779[v77].m128i_i64[0] = (__int64)v699;
                                v700[v91].m128i_i32[2] = v701;
                                v72 = v714;
                              }
                              goto LABEL_101;
                            }
                          }
                        }
                        v72 = 234;
                        HIWORD(v615) = v721;
                        LOWORD(v615) = v729;
                        v616 = sub_33FAF80(*v722, 234, (__int64)&v770, v615, 0, v79, a4);
                        v617 = &v779[v77];
                        v617->m128i_i64[0] = (__int64)v616;
                        v617->m128i_i32[2] = v618;
                        goto LABEL_101;
                      }
                    }
                    else
                    {
                      v732 = v80;
                      v740 = (__int64)v82;
                      v86 = sub_30070B0((__int64)&v775);
                      v82 = (unsigned int *)v740;
                      if ( v86 && !v732 )
                      {
                        v88 = sub_3009970((__int64)&v775, v72, v740, v87, 0);
                        goto LABEL_108;
                      }
                    }
                    v78 = &v779[v77];
                    v78->m128i_i64[0] = *(_QWORD *)v82;
                    v78->m128i_i32[2] = v82[2];
LABEL_101:
                    if ( *(_DWORD *)(v30 + 64) == ++v77 )
                    {
                      v8 = v722;
                      break;
                    }
                  }
                }
                v643 = sub_33FBA10(
                         (_QWORD *)*v8,
                         *(unsigned int *)(v30 + 24),
                         (__int64)&v770,
                         v729,
                         0,
                         *(_DWORD *)(v30 + 28),
                         (__int64)v779,
                         (unsigned int)v780);
                v645 = v644;
                if ( ((unsigned __int16)(v725 - 10) <= 6u
                   || (unsigned __int16)(v725 - 126) <= 0x31u
                   || (unsigned __int16)(v725 - 208) <= 0x14u)
                  && ((unsigned __int16)(v729 - 10) <= 6u
                   || (unsigned __int16)(v729 - 126) <= 0x31u
                   || (unsigned __int16)(v729 - 208) <= 0x14u)
                  || (unsigned __int16)(v725 - 17) <= 0xD3u
                  && ((v664 = word_4456580[v725 - 1], (unsigned __int16)(v664 - 10) <= 6u)
                   || (unsigned __int16)(v664 - 126) <= 0x31u
                   || (unsigned __int16)(v664 - 208) <= 0x14u)
                  && (unsigned __int16)(v729 - 17) <= 0xD3u
                  && ((v665 = word_4456580[v729 - 1], (unsigned __int16)(v665 - 10) <= 6u)
                   || (unsigned __int16)(v665 - 126) <= 0x31u
                   || (unsigned __int16)(v665 - 208) <= 0x14u) )
                {
                  if ( sub_33CB110(*(_DWORD *)(v30 + 24)) )
                  {
                    v775.m128i_i64[0] = sub_33CB1F0(*(_DWORD *)(v30 + 24));
                    v646 = v775.m128i_u32[0];
                    v647 = sub_33CB160(*(_DWORD *)(v30 + 24));
                    v648 = (_QWORD *)*v8;
                    v775.m128i_i64[0] = v647;
                    *((_QWORD *)&v703 + 1) = v645;
                    *(_QWORD *)&v703 = v643;
                    v651 = sub_340F900(
                             v648,
                             0x1C8u,
                             (__int64)&v770,
                             v725,
                             0,
                             v649,
                             v703,
                             *(_OWORD *)(*(_QWORD *)(v30 + 40) + 40LL * (unsigned int)v647),
                             *(_OWORD *)(*(_QWORD *)(v30 + 40) + 40 * v646));
                  }
                  else
                  {
                    v687 = (_QWORD *)*v8;
                    *(_QWORD *)&v688 = sub_3400D50(*v8, 0, (__int64)&v770, 1u, a4);
                    *((_QWORD *)&v705 + 1) = v645;
                    *(_QWORD *)&v705 = v643;
                    v651 = (__int64)sub_3406EB0(v687, 0xE6u, (__int64)&v770, v725, 0, v689, v705, v688);
                  }
                  v652 = v650 | v645 & 0xFFFFFFFF00000000LL;
                }
                else
                {
                  v651 = (__int64)sub_33FAF80(*v8, 234, (__int64)&v770, v725, 0, v642, a4);
                  v652 = v666 | v645 & 0xFFFFFFFF00000000LL;
                }
                v653 = v785.m128i_u32[2];
                v654 = v785.m128i_u32[2] + 1LL;
                if ( v654 > v785.m128i_u32[3] )
                {
                  sub_C8D5F0((__int64)&v785, v786, v654, 0x10u, v27, v28);
                  v653 = v785.m128i_u32[2];
                }
                v655 = (__int64 *)(v785.m128i_i64[0] + 16 * v653);
                *v655 = v651;
                v655[1] = v652;
                ++v785.m128i_i32[2];
                if ( v779 != &v781 )
                  _libc_free((unsigned __int64)v779);
                if ( v770.m128i_i64[0] )
                  sub_B91220((__int64)&v770, v770.m128i_i64[0]);
                goto LABEL_127;
              }
              v718 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                              + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
              v550 = sub_332CC60(v8[1], v68, v718);
              v553 = *(_QWORD *)(v30 + 80);
              v723 = v550;
              v768 = v553;
              if ( v553 )
                sub_B96E90((__int64)&v768, v553, 1);
              v554 = *(_DWORD *)(v30 + 72);
              v555 = *(unsigned int *)(v30 + 64);
              v780 = 0x500000000LL;
              v769 = v554;
              v556 = &v781;
              v557 = &v781;
              v779 = &v781;
              if ( v555 )
              {
                if ( v555 > 5 )
                {
                  sub_C8D5F0((__int64)&v779, &v781, v555, 0x10u, v551, v552);
                  v557 = v779;
                  v556 = &v779[(unsigned int)v780];
                }
                for ( k = &v557[v555]; k != v556; ++v556 )
                {
                  if ( v556 )
                  {
                    v556->m128i_i64[0] = 0;
                    v556->m128i_i32[2] = 0;
                  }
                }
                LODWORD(v780) = v555;
              }
              v772 = v774;
              v773 = 0x200000000LL;
              if ( *(_DWORD *)(v30 + 64) != 1 )
              {
                v720 = v8;
                v559 = v30;
                v560 = 1;
                while ( 1 )
                {
                  v562 = 5LL * v560;
                  v563 = 40LL * v560;
                  v564 = (unsigned int *)(v563 + *(_QWORD *)(v559 + 40));
                  v565 = *(_QWORD *)(*(_QWORD *)v564 + 48LL) + 16LL * v564[2];
                  v566 = *(_WORD *)v565;
                  v567 = *(_QWORD *)(v565 + 8);
                  v775.m128i_i16[0] = v566;
                  v775.m128i_i64[1] = v567;
                  if ( v566 )
                  {
                    if ( (unsigned __int16)(v566 - 17) > 0xD3u )
                      goto LABEL_821;
                  }
                  else
                  {
                    v568 = sub_30070B0((__int64)&v775);
                    v562 = 5LL * v560;
                    if ( !v568 )
                      goto LABEL_821;
                  }
                  v735 = v562;
                  v569 = sub_33CB110(*(_DWORD *)(v559 + 24));
                  v571 = v735;
                  if ( !v569
                    || (v572 = sub_33CB160(*(_DWORD *)(v559 + 24)), v571 = v735, v770.m128i_i64[0] = v572, !BYTE4(v572))
                    || v770.m128i_i32[0] != v560 )
                  {
                    v573 = (_QWORD *)*v720;
                    v574 = *(const __m128i **)(v559 + 40);
                    v770 = _mm_loadu_si128(v574);
                    v575 = _mm_loadu_si128((const __m128i *)((char *)v574 + 8 * v571));
                    *((_QWORD *)&v707 + 1) = 2;
                    *(_QWORD *)&v707 = &v770;
                    v775.m128i_i64[1] = 0;
                    v775.m128i_i16[0] = v723;
                    v776.m128i_i16[0] = 1;
                    v776.m128i_i64[1] = 0;
                    v771 = v575;
                    v576 = sub_3411BE0(v573, 0x92u, (__int64)&v768, (unsigned __int16 *)&v775, 2, v570, v707);
                    v578 = &v779[v560];
                    v579 = v576;
                    v578->m128i_i64[0] = (__int64)v576;
                    v578->m128i_i32[2] = 0;
                    v580 = (unsigned int)v773;
                    v581 = v728 & 0xFFFFFFFF00000000LL | 1;
                    v582 = (unsigned int)v773 + 1LL;
                    v728 = v581;
                    if ( v582 > HIDWORD(v773) )
                    {
                      v736 = v581;
                      sub_C8D5F0((__int64)&v772, v774, v582, 0x10u, v577, v581);
                      v580 = (unsigned int)v773;
                      v581 = v736;
                    }
                    v583 = (unsigned __int8 **)&v772[16 * v580];
                    *v583 = v579;
                    v583[1] = (unsigned __int8 *)v581;
                    LODWORD(v773) = v773 + 1;
                    goto LABEL_822;
                  }
                  v564 = (unsigned int *)(*(_QWORD *)(v559 + 40) + v563);
LABEL_821:
                  v561 = &v779[v560];
                  v561->m128i_i64[0] = *(_QWORD *)v564;
                  v561->m128i_i32[2] = v564[2];
LABEL_822:
                  if ( *(_DWORD *)(v559 + 64) == ++v560 )
                  {
                    v30 = v559;
                    v8 = v720;
                    break;
                  }
                }
              }
              v621 = sub_33E5110(
                       (__int64 *)*v8,
                       v723,
                       0,
                       *(unsigned __int16 *)(*(_QWORD *)(v30 + 48) + 16LL),
                       *(_QWORD *)(*(_QWORD *)(v30 + 48) + 24LL));
              v623 = v622;
              v747 = (unsigned int *)v621;
              *((_QWORD *)&v708 + 1) = (unsigned int)v773;
              *(_QWORD *)&v708 = v772;
              v625 = sub_33FC220((_QWORD *)*v8, 2, (__int64)&v768, 1, 0, v624, v708);
              v627 = v626;
              v628 = (__int64)v625;
              v629 = v779;
              v779->m128i_i64[0] = v628;
              v629->m128i_i32[2] = v627;
              v630 = sub_3410740(
                       (_QWORD *)*v8,
                       *(unsigned int *)(v30 + 24),
                       (__int64)&v768,
                       v747,
                       v623,
                       *(_DWORD *)(v30 + 28),
                       a4,
                       v779,
                       (unsigned int)v780);
              v631 = (_QWORD *)*v8;
              v775.m128i_i64[0] = (__int64)v630;
              v775.m128i_i32[2] = 1;
              v776.m128i_i64[0] = (__int64)v630;
              v776.m128i_i32[2] = 0;
              v632 = sub_3400D50((__int64)v631, 0, (__int64)&v768, 1u, a4);
              v778 = v633;
              *((_QWORD *)&v709 + 1) = 3;
              *(_QWORD *)&v709 = &v775;
              v770.m128i_i16[0] = v718;
              v771.m128i_i16[0] = 1;
              v777 = v632;
              v770.m128i_i64[1] = 0;
              v771.m128i_i64[1] = 0;
              v635 = sub_3411BE0(v631, 0x91u, (__int64)&v768, (unsigned __int16 *)&v770, 2, v634, v709);
              v28 = v710;
              v636 = v635;
              v637 = v785.m128i_u32[2];
              v638 = v785.m128i_u32[2] + 1LL;
              if ( v638 > v785.m128i_u32[3] )
              {
                sub_C8D5F0((__int64)&v785, v786, v638, 0x10u, v27, v710);
                v637 = v785.m128i_u32[2];
              }
              v639 = (unsigned __int8 **)(v785.m128i_i64[0] + 16 * v637);
              *v639 = v636;
              v639[1] = 0;
              ++v785.m128i_i32[2];
              v640 = v785.m128i_u32[2];
              if ( (unsigned __int64)v785.m128i_u32[2] + 1 > v785.m128i_u32[3] )
              {
                sub_C8D5F0((__int64)&v785, v786, v785.m128i_u32[2] + 1LL, 0x10u, v27, v28);
                v640 = v785.m128i_u32[2];
              }
              v641 = (unsigned __int8 **)(v785.m128i_i64[0] + 16 * v640);
              *v641 = v636;
              v641[1] = (unsigned __int8 *)1;
              ++v785.m128i_i32[2];
              if ( v772 != v774 )
                _libc_free((unsigned __int64)v772);
              if ( v779 != &v781 )
                _libc_free((unsigned __int64)v779);
              if ( v768 )
                sub_B91220((__int64)&v768, v768);
LABEL_127:
              v744 = v785.m128i_i32[2];
              goto LABEL_128;
            }
LABEL_786:
            v522 = *(unsigned __int16 **)(v30 + 48);
            v523 = *v522;
            v524 = sub_332CC60(v8[1], v68, *v522);
            v525 = *(_DWORD *)(v30 + 24);
            if ( v525 > 239 )
            {
              v756 = (unsigned int)(v525 - 242) <= 1;
            }
            else
            {
              if ( v525 > 237 )
              {
                v756 = 1;
LABEL_789:
                v526 = (unsigned int)v525;
                if ( v525 == 227 )
                {
                  v527 = v8[1];
                  v528 = 1;
                  if ( v524 == 1 || v524 && (v528 = v524, *(_QWORD *)(v527 + 8LL * v524 + 112)) )
                  {
                    v526 = 226;
                    if ( (*(_BYTE *)(v527 + 500 * v528 + 6640) & 0xFB) != 0 )
                      v526 = 227;
                  }
LABEL_793:
                  v529 = *(_QWORD *)(v30 + 80);
                  v476 = (__m128i *)&v772;
                  v772 = (_BYTE *)v529;
                  if ( v529 )
                  {
                    v745 = v526;
                    sub_B96E90((__int64)&v772, v529, 1);
                    v526 = v745;
                  }
                  v530 = (_QWORD *)*v8;
                  LODWORD(v773) = *(_DWORD *)(v30 + 72);
                  v531 = *(_QWORD *)(v30 + 40);
                  if ( v756 )
                  {
                    v775 = _mm_loadu_si128((const __m128i *)v531);
                    v532 = _mm_loadu_si128((const __m128i *)(v531 + 40));
                    *((_QWORD *)&v706 + 1) = 2;
                    *(_QWORD *)&v706 = &v775;
                    LOWORD(v779) = v524;
                    v780 = 0;
                    v781.m128i_i16[0] = 1;
                    v781.m128i_i64[1] = 0;
                    v776 = v532;
                    v727 = 1;
                    v533 = sub_3411BE0(v530, v526, (__int64)&v772, (unsigned __int16 *)&v779, 2, v526, v706);
                    v746 = v534;
                    v731 = v533;
                  }
                  else
                  {
                    v675 = sub_33FAF80((__int64)v530, (unsigned int)v526, (__int64)&v772, v524, 0, v526, a4);
                    v727 = 0;
                    v746 = v676;
                    v533 = v675;
                    v731 = 0;
                  }
                  v535 = *(_DWORD *)(v30 + 24);
                  if ( v535 == 142 || (v536 = 3, v535 == 227) )
                    v536 = 4;
                  v537 = (_QWORD *)*v8;
                  v538 = v523;
                  if ( (unsigned __int16)(v523 - 17) <= 0xD3u )
                    v538 = word_4456580[v523 - 1];
                  v734 = v536;
                  v739 = v533;
                  *(_QWORD *)&v539 = sub_33F7D60(v537, v538, 0);
                  *((_QWORD *)&v704 + 1) = v746;
                  *(_QWORD *)&v704 = v739;
                  sub_3406EB0(v537, v734, (__int64)&v772, v524, 0, v734, v704, v539);
                  v715 = v540;
                  v542 = sub_33FAF80(*v8, 216, (__int64)&v772, v523, 0, v541, a4);
                  v543 = v785.m128i_u32[2];
                  v545 = v544 | v715 & 0xFFFFFFFF00000000LL;
                  v546 = v785.m128i_u32[2] + 1LL;
                  v547 = v545;
                  if ( v546 > v785.m128i_u32[3] )
                  {
                    v749 = v542;
                    sub_C8D5F0((__int64)&v785, v786, v546, 0x10u, v27, v28);
                    v543 = v785.m128i_u32[2];
                    v542 = v749;
                    v547 = v545;
                  }
                  v548 = (unsigned __int8 **)(v785.m128i_i64[0] + 16 * v543);
                  *v548 = v542;
                  v548[1] = (unsigned __int8 *)v547;
                  v549 = (unsigned int)++v785.m128i_i32[2];
                  if ( v756 )
                  {
                    if ( v549 + 1 > (unsigned __int64)v785.m128i_u32[3] )
                    {
                      sub_C8D5F0((__int64)&v785, v786, v549 + 1, 0x10u, v27, v28);
                      v549 = v785.m128i_u32[2];
                    }
                    v686 = (unsigned __int8 **)(v785.m128i_i64[0] + 16 * v549);
                    v686[1] = (unsigned __int8 *)(v727 & 1);
                    *v686 = v731;
                    ++v785.m128i_i32[2];
                  }
                  goto LABEL_804;
                }
LABEL_894:
                if ( v525 == 142 )
                {
                  v619 = v8[1];
                  v620 = 1;
                  if ( v524 == 1 || v524 && (v620 = v524, *(_QWORD *)(v619 + 8LL * v524 + 112)) )
                  {
                    v526 = 141;
                    if ( (*(_BYTE *)(v619 + 500 * v620 + 6555) & 0xFB) != 0 )
                      v526 = 142;
                  }
                }
                goto LABEL_793;
              }
              v756 = 0;
              if ( (unsigned int)(v525 - 101) > 0x2F )
                goto LABEL_789;
              v756 = 1;
            }
            v526 = (unsigned int)v525;
            goto LABEL_894;
          }
          v500 = v68 - 143;
          if ( (unsigned int)(v68 - 143) > 1 )
            goto LABEL_86;
LABEL_765:
          v476 = (__m128i *)&v772;
          v726 = v500 < 6;
          v501 = (unsigned int *)(*(_QWORD *)(v30 + 40) + (v500 < 6 ? 0x28 : 0));
          v502 = sub_332CC60(v8[1], v68, *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v501 + 48LL) + 16LL * v501[2]));
          v505 = *(_QWORD *)(v30 + 80);
          v730 = v502;
          v772 = (_BYTE *)v505;
          if ( v505 )
            sub_B96E90((__int64)&v772, v505, 1);
          LODWORD(v773) = *(_DWORD *)(v30 + 72);
          v506 = *(unsigned int *)(v30 + 64);
          v507 = &v781;
          v508 = &v781;
          v779 = &v781;
          v780 = 0x400000000LL;
          if ( v506 )
          {
            if ( v506 > 4 )
            {
              sub_C8D5F0((__int64)&v779, &v781, v506, 0x10u, v503, v504);
              v508 = v779;
              v507 = &v779[(unsigned int)v780];
            }
            for ( m = &v508[v506]; m != v507; ++v507 )
            {
              if ( v507 )
              {
                v507->m128i_i64[0] = 0;
                v507->m128i_i32[2] = 0;
              }
            }
            LODWORD(v780) = v506;
          }
          v510 = *(_DWORD *)(v30 + 24);
          if ( v510 == 144 || (v733 = 213, v510 == 221) )
            v733 = 214;
          v511 = 0;
          if ( *(_DWORD *)(v30 + 64) )
          {
            HIWORD(v512) = v724;
            v513 = v30;
            do
            {
              v517 = v511;
              v518 = (unsigned int *)(*(_QWORD *)(v513 + 40) + 40LL * v511);
              v519 = *(_QWORD *)(*(_QWORD *)v518 + 48LL) + 16LL * v518[2];
              v520 = *(_WORD *)v519;
              v521 = *(_QWORD *)(v519 + 8);
              v775.m128i_i16[0] = v520;
              v775.m128i_i64[1] = v521;
              if ( v520 )
              {
                v514 = (unsigned __int16)(v520 - 17) <= 0xD3u;
              }
              else
              {
                v514 = sub_30070B0((__int64)&v775);
                v517 = v511;
              }
              v515 = v517;
              if ( v514 )
              {
                LOWORD(v512) = v730;
                v716 = *((_QWORD *)v518 + 1);
                v757 = v515 * 16;
                v584 = sub_33FAF80(*v8, v733, (__int64)&v772, v512, 0, v515 * 16, a4);
                v512 = v585;
                v586 = (__int64)v584;
                v587 = v779;
                *(__int64 *)((char *)v779->m128i_i64 + v757) = v586;
                *(__int32 *)((char *)&v587->m128i_i32[2] + v757) = v512;
                HIWORD(v512) = WORD1(v716);
              }
              else
              {
                v516 = v779;
                v779[v515].m128i_i64[0] = *(_QWORD *)v518;
                v516[v515].m128i_i32[2] = v518[2];
              }
              ++v511;
            }
            while ( *(_DWORD *)(v513 + 64) != v511 );
            v30 = v513;
            v476 = (__m128i *)&v772;
          }
          v656 = *(_QWORD *)(v30 + 48);
          v657 = (_QWORD *)*v8;
          v658 = *(_WORD *)v656;
          if ( v726 )
          {
            v775.m128i_i64[1] = *(_QWORD *)(v656 + 8);
            v776.m128i_i16[0] = 1;
            v775.m128i_i16[0] = v658;
            v776.m128i_i64[1] = 0;
            *((_QWORD *)&v712 + 1) = (unsigned int)v780;
            *(_QWORD *)&v712 = v779;
            v677 = sub_3411BE0(
                     v657,
                     *(_DWORD *)(v30 + 24),
                     (__int64)&v772,
                     (unsigned __int16 *)&v775,
                     2,
                     (__int64)v779,
                     v712);
            v28 = v713;
            v678 = v677;
            v679 = v785.m128i_u32[2];
            v681 = v680;
            v682 = v785.m128i_u32[2] + 1LL;
            if ( v682 > v785.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v785, v786, v682, 0x10u, v27, v713);
              v679 = v785.m128i_u32[2];
            }
            v683 = (unsigned __int8 **)(v785.m128i_i64[0] + 16 * v679);
            *v683 = v678;
            v683[1] = v681;
            ++v785.m128i_i32[2];
            v684 = v785.m128i_u32[2];
            if ( (unsigned __int64)v785.m128i_u32[2] + 1 > v785.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v785, v786, v785.m128i_u32[2] + 1LL, 0x10u, v27, v28);
              v684 = v785.m128i_u32[2];
            }
            v685 = (unsigned __int8 **)(v785.m128i_i64[0] + 16 * v684);
            *v685 = v678;
            v685[1] = (unsigned __int8 *)1;
            v663 = v779;
            ++v785.m128i_i32[2];
            if ( v779 != &v781 )
              goto LABEL_934;
          }
          else
          {
            *((_QWORD *)&v711 + 1) = (unsigned int)v780;
            *(_QWORD *)&v711 = v779;
            v27 = (__int64)sub_33FC220(
                             v657,
                             *(unsigned int *)(v30 + 24),
                             (__int64)&v772,
                             v658,
                             *(_QWORD *)(v656 + 8),
                             (__int64)v779,
                             v711);
            v659 = v785.m128i_u32[2];
            v28 = v660;
            v661 = v785.m128i_u32[2] + 1LL;
            if ( v661 > v785.m128i_u32[3] )
            {
              v762 = (__m128i *)v27;
              v765 = v28;
              sub_C8D5F0((__int64)&v785, v786, v661, 0x10u, v27, v28);
              v659 = v785.m128i_u32[2];
              v27 = (__int64)v762;
              v28 = v765;
            }
            v662 = (__int64 *)(v785.m128i_i64[0] + 16 * v659);
            *v662 = v27;
            v662[1] = v28;
            v663 = v779;
            ++v785.m128i_i32[2];
            if ( v779 != &v781 )
LABEL_934:
              _libc_free((unsigned __int64)v663);
          }
LABEL_804:
          v499 = (__int64)v772;
          if ( !v772 )
            goto LABEL_127;
          goto LABEL_759;
        }
        if ( v68 == 463 )
        {
LABEL_739:
          v470 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                                    + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
          v471 = *v470;
          v472 = sub_332CC60(v8[1], v68, *v470);
          if ( (unsigned __int16)(v471 - 10) <= 6u || (unsigned __int16)(v471 - 126) <= 0x31u )
            v474 = 233;
          else
            v474 = (unsigned __int16)(v471 - 208) < 0x15u ? 233 : 215;
          v475 = *(_QWORD *)(v30 + 80);
          v476 = &v775;
          v775.m128i_i64[0] = v475;
          if ( v475 )
            sub_B96E90((__int64)&v775, v475, 1);
          v775.m128i_i32[2] = *(_DWORD *)(v30 + 72);
          v477 = *(unsigned int *)(v30 + 64);
          v478 = &v781;
          v779 = &v781;
          v479 = v477;
          v780 = 0x500000000LL;
          if ( v477 )
          {
            v480 = &v781;
            if ( v477 > 5 )
            {
              v748 = v477;
              v760 = v477;
              sub_C8D5F0((__int64)&v779, &v781, v477, 0x10u, v477, v473);
              v480 = v779;
              v479 = v748;
              v477 = v760;
              v478 = &v779[(unsigned int)v780];
            }
            for ( n = &v480[v477]; n != v478; ++v478 )
            {
              if ( v478 )
              {
                v478->m128i_i64[0] = 0;
                v478->m128i_i32[2] = 0;
              }
            }
            LODWORD(v780) = v479;
          }
          v482 = sub_33FAF80(*v8, v474, (__int64)&v775, v472, 0, v473, a4);
          v484 = v483;
          v485 = (__int64)v482;
          v486 = v779;
          v779->m128i_i64[0] = v485;
          v486->m128i_i32[2] = v484;
          v488 = sub_33FAF80(*v8, v474, (__int64)&v775, v472, 0, v487, a4);
          v489 = v779;
          v779[1].m128i_i64[0] = (__int64)v488;
          v489[1].m128i_i32[2] = v490;
          v491 = *(_QWORD *)(v30 + 40);
          v489[2].m128i_i64[0] = *(_QWORD *)(v491 + 80);
          v489[2].m128i_i32[2] = *(_DWORD *)(v491 + 88);
          v492 = *(unsigned int *)(v30 + 24);
          if ( (_DWORD)v492 == 463 )
          {
            v493 = *(_QWORD *)(v30 + 40);
            v489[3].m128i_i64[0] = *(_QWORD *)(v493 + 120);
            v489[3].m128i_i32[2] = *(_DWORD *)(v493 + 128);
            v494 = *(_QWORD *)(v30 + 40);
            v489[4].m128i_i64[0] = *(_QWORD *)(v494 + 160);
            v489[4].m128i_i32[2] = *(_DWORD *)(v494 + 168);
            v492 = *(unsigned int *)(v30 + 24);
          }
          v27 = (__int64)sub_33FBA10(
                           (_QWORD *)*v8,
                           v492,
                           (__int64)&v775,
                           **(unsigned __int16 **)(v30 + 48),
                           0,
                           *(_DWORD *)(v30 + 28),
                           (__int64)v489,
                           (unsigned int)v780);
          v495 = v785.m128i_u32[2];
          v28 = v496;
          v497 = v785.m128i_u32[2] + 1LL;
          if ( v497 > v785.m128i_u32[3] )
          {
            v761 = (__m128i *)v27;
            v764 = v28;
            sub_C8D5F0((__int64)&v785, v786, v497, 0x10u, v27, v28);
            v495 = v785.m128i_u32[2];
            v27 = (__int64)v761;
            v28 = v764;
          }
          v498 = (__int64 *)(v785.m128i_i64[0] + 16 * v495);
          *v498 = v27;
          v498[1] = v28;
          ++v785.m128i_i32[2];
          if ( v779 != &v781 )
            _libc_free((unsigned __int64)v779);
          v499 = v775.m128i_i64[0];
          if ( !v775.m128i_i64[0] )
            goto LABEL_127;
LABEL_759:
          sub_B91220((__int64)v476, v499);
          goto LABEL_127;
        }
        if ( v68 > 463 )
          goto LABEL_86;
        if ( v68 > 434 )
        {
          if ( v68 != 438 )
            goto LABEL_86;
        }
        else if ( v68 <= 432 )
        {
          goto LABEL_86;
        }
        goto LABEL_1032;
      }
LABEL_163:
      if ( (unsigned __int8)v67 <= 1u )
        goto LABEL_733;
      if ( (_BYTE)v67 == 2 )
        goto LABEL_126;
      if ( (_BYTE)v67 != 4 )
LABEL_1032:
        BUG();
LABEL_166:
      v133 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, _QWORD, __int64))(*(_QWORD *)v8[1] + 2416LL))(
               v8[1],
               v30,
               0,
               *v8);
      v28 = v133;
      if ( !v133 )
        goto LABEL_126;
      if ( v30 == v133 && !(_DWORD)v134 )
        goto LABEL_127;
      v135 = *(_DWORD *)(v30 + 68);
      if ( v135 == 1 )
      {
        sub_3050D50((__int64)&v785, v28, v134, v44, v27, v28);
        v744 = v785.m128i_i32[2];
      }
      else
      {
        v136 = v785.m128i_u32[2];
        v744 = v785.m128i_i32[2];
        if ( v135 )
        {
          v137 = v737;
          v754 = v30;
          v138 = 0;
          v139 = v28;
          v742 = v8;
          v140 = v135;
          v27 = 0xFFFFFFFF00000000LL;
          do
          {
            v141 = v137 & 0xFFFFFFFF00000000LL | v138;
            v137 = v141;
            if ( v136 + 1 > (unsigned __int64)v785.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v785, v786, v136 + 1, 0x10u, 0xFFFFFFFF00000000LL, v28);
              v136 = v785.m128i_u32[2];
              v27 = 0xFFFFFFFF00000000LL;
            }
            v142 = (__int64 *)(v785.m128i_i64[0] + 16 * v136);
            ++v138;
            *v142 = v139;
            v142[1] = v141;
            v136 = (unsigned int)++v785.m128i_i32[2];
          }
          while ( v140 != v138 );
          v744 = v136;
          v30 = v754;
          v8 = v742;
        }
      }
LABEL_128:
      if ( !v744 )
      {
LABEL_733:
        v17 = sub_376D860((__int64)v8, a2, a3, v30);
        goto LABEL_703;
      }
      v103 = (__int64 *)v785.m128i_i64[0];
      *((_BYTE *)v8 + 16) = 1;
      v104 = 0;
      v738 = (const __m128i *)(v8 + 3);
      v741 = v103;
      v105 = (unsigned int)(a2 >> 9) ^ (unsigned int)(a2 >> 4);
      do
      {
        v753 = v105;
        v106 = sub_376DE90(v8, *v103, v103[1], v105, v27, v28);
        v107 = v753;
        v108 = v106;
        *v103 = v106;
        v110 = v109;
        *((_DWORD *)v103 + 2) = v109;
        LOBYTE(v109) = *((_BYTE *)v8 + 32);
        v781.m128i_i64[0] = v106;
        v111 = v109 & 1;
        v781.m128i_i32[2] = v110;
        if ( v111 )
        {
          v28 = (__int64)(v8 + 5);
          v112 = 63;
        }
        else
        {
          v450 = *((_DWORD *)v8 + 12);
          v28 = v8[5];
          if ( !v450 )
          {
            v460 = *((_DWORD *)v8 + 8);
            ++v8[3];
            v114 = 0;
            v461 = (v460 >> 1) + 1;
LABEL_706:
            v28 = 3 * v450;
            goto LABEL_707;
          }
          v112 = v450 - 1;
        }
        v113 = 1;
        v114 = 0;
        for ( ii = v753 & v112; ; ii = v112 & v117 )
        {
          v116 = v28 + 32LL * ii;
          v27 = *(_QWORD *)v116;
          if ( *(_QWORD *)v116 != a2 )
            break;
          if ( *(_DWORD *)(v116 + 8) == v104 )
            goto LABEL_691;
LABEL_135:
          v117 = v113 + ii;
          ++v113;
        }
        if ( v27 )
          goto LABEL_135;
        v606 = *(_DWORD *)(v116 + 8);
        if ( v606 != -1 )
        {
          if ( !v114 && v606 == -2 )
            v114 = (__m128i *)(v28 + 32LL * ii);
          goto LABEL_135;
        }
        if ( !v114 )
          v114 = (__m128i *)(v28 + 32LL * ii);
        v460 = *((_DWORD *)v8 + 8);
        ++v8[3];
        v461 = (v460 >> 1) + 1;
        if ( !(_BYTE)v111 )
        {
          v450 = *((_DWORD *)v8 + 12);
          goto LABEL_706;
        }
        v28 = 192;
        v450 = 64;
LABEL_707:
        if ( (unsigned int)v28 > 4 * v461 )
        {
          v462 = v450 - *((_DWORD *)v8 + 9) - v461;
          v27 = v450 >> 3;
          if ( v462 > (unsigned int)v27 )
            goto LABEL_709;
          sub_376D3B0(v738, v450);
          v107 = v753;
          if ( (v8[4] & 1) != 0 )
          {
            v27 = (__int64)(v8 + 5);
            v592 = 63;
LABEL_845:
            v593 = 1;
            v28 = 0;
            for ( jj = v753 & v592; ; jj = v592 & v595 )
            {
              v114 = (__m128i *)(v27 + 32LL * jj);
              if ( v114->m128i_i64[0] == a2 )
              {
                if ( v114->m128i_i32[2] == v104 )
                  goto LABEL_950;
              }
              else if ( !v114->m128i_i64[0] )
              {
                v696 = v114->m128i_i32[2];
                if ( v696 == -1 )
                  goto LABEL_1016;
                if ( v696 == -2 && !v28 )
                  v28 = v27 + 32LL * jj;
              }
              v595 = v593 + jj;
              ++v593;
            }
          }
          v608 = *((_DWORD *)v8 + 12);
          v27 = v8[5];
          if ( v608 )
          {
            v592 = v608 - 1;
            goto LABEL_845;
          }
LABEL_1033:
          *((_DWORD *)v8 + 8) = (2 * (*((_DWORD *)v8 + 8) >> 1) + 2) | v8[4] & 1;
          BUG();
        }
        sub_376D3B0(v738, 2 * v450);
        v107 = v753;
        if ( (v8[4] & 1) != 0 )
        {
          v588 = v8 + 5;
          v589 = 63;
        }
        else
        {
          v607 = *((_DWORD *)v8 + 12);
          v588 = (_QWORD *)v8[5];
          if ( !v607 )
            goto LABEL_1033;
          v589 = v607 - 1;
        }
        v590 = 1;
        v28 = 0;
        v27 = v753 & (unsigned int)v589;
        while ( 2 )
        {
          v114 = (__m128i *)&v588[4 * (unsigned int)v27];
          if ( v114->m128i_i64[0] == a2 )
          {
            if ( v114->m128i_i32[2] == v104 )
              goto LABEL_950;
            goto LABEL_842;
          }
          if ( v114->m128i_i64[0] )
          {
LABEL_842:
            v591 = v590 + v27;
            ++v590;
            v27 = v589 & (unsigned int)v591;
            continue;
          }
          break;
        }
        v695 = v114->m128i_i32[2];
        if ( v695 != -1 )
        {
          if ( v695 == -2 && !v28 )
            v28 = (__int64)&v588[4 * (unsigned int)v27];
          goto LABEL_842;
        }
LABEL_1016:
        if ( v28 )
          v114 = (__m128i *)v28;
LABEL_950:
        v460 = *((_DWORD *)v8 + 8);
LABEL_709:
        *((_DWORD *)v8 + 8) = (2 * (v460 >> 1) + 2) | v460 & 1;
        if ( v114->m128i_i64[0] || v114->m128i_i32[2] != -1 )
          --*((_DWORD *)v8 + 9);
        v114->m128i_i32[2] = v104;
        v114->m128i_i64[0] = a2;
        v114[1] = _mm_load_si128(&v781);
LABEL_691:
        if ( a2 == v108 && v110 == v104 )
          goto LABEL_701;
        v451 = *((_BYTE *)v8 + 32);
        v781.m128i_i64[0] = v108;
        v781.m128i_i32[2] = v110;
        v452 = v451 & 1;
        if ( v452 )
        {
          v453 = v8 + 5;
          v454 = 63;
          goto LABEL_694;
        }
        v459 = *((_DWORD *)v8 + 12);
        v453 = (_QWORD *)v8[5];
        if ( !v459 )
        {
          v464 = *((_DWORD *)v8 + 8);
          ++v8[3];
          v27 = 0;
          v465 = (v464 >> 1) + 1;
LABEL_720:
          v28 = 3 * v459;
          goto LABEL_721;
        }
        v454 = v459 - 1;
LABEL_694:
        v455 = 1;
        v27 = 0;
        v456 = v454 & (v110 + ((v108 >> 9) ^ (v108 >> 4)));
        while ( 2 )
        {
          v457 = (__m128i *)&v453[4 * v456];
          v28 = v457->m128i_i64[0];
          if ( v457->m128i_i64[0] == v108 )
          {
            if ( v457->m128i_i32[2] == v110 )
              goto LABEL_701;
            if ( v28 )
              goto LABEL_697;
          }
          else if ( v28 )
          {
LABEL_697:
            v458 = v455 + v456;
            ++v455;
            v456 = v454 & v458;
            continue;
          }
          break;
        }
        v463 = v457->m128i_i32[2];
        if ( v463 != -1 )
        {
          if ( !v27 && v463 == -2 )
            v27 = (__int64)&v453[4 * v456];
          goto LABEL_697;
        }
        v464 = *((_DWORD *)v8 + 8);
        if ( !v27 )
          v27 = (__int64)v457;
        ++v8[3];
        v465 = (v464 >> 1) + 1;
        if ( !(_BYTE)v452 )
        {
          v459 = *((_DWORD *)v8 + 12);
          goto LABEL_720;
        }
        v28 = 192;
        v459 = 64;
LABEL_721:
        if ( (unsigned int)v28 <= 4 * v465 )
        {
          v758 = v107;
          sub_376D3B0(v738, 2 * v459);
          v107 = v758;
          if ( (v8[4] & 1) != 0 )
          {
            v596 = v8 + 5;
            v597 = 63;
LABEL_851:
            v598 = 1;
            v28 = 0;
            for ( kk = v597 & (v110 + ((v108 >> 9) ^ (v108 >> 4))); ; kk = v597 & v600 )
            {
              v27 = (__int64)&v596[4 * kk];
              if ( *(_QWORD *)v27 == v108 && *(_DWORD *)(v27 + 8) == v110 )
                break;
              if ( !*(_QWORD *)v27 )
              {
                v697 = *(_DWORD *)(v27 + 8);
                if ( v697 == -1 )
                {
LABEL_1024:
                  if ( v28 )
                    v27 = v28;
                  break;
                }
                if ( v697 == -2 && !v28 )
                  v28 = (__int64)&v596[4 * kk];
              }
              v600 = v598 + kk;
              ++v598;
            }
LABEL_968:
            v464 = *((_DWORD *)v8 + 8);
            goto LABEL_723;
          }
          v613 = *((_DWORD *)v8 + 12);
          v596 = (_QWORD *)v8[5];
          if ( v613 )
          {
            v597 = v613 - 1;
            goto LABEL_851;
          }
LABEL_1034:
          *((_DWORD *)v8 + 8) = (2 * (*((_DWORD *)v8 + 8) >> 1) + 2) | v8[4] & 1;
          BUG();
        }
        if ( v459 - *((_DWORD *)v8 + 9) - v465 <= v459 >> 3 )
        {
          v759 = v107;
          sub_376D3B0(v738, v459);
          v107 = v759;
          if ( (v8[4] & 1) != 0 )
          {
            v601 = v8 + 5;
            v602 = 63;
          }
          else
          {
            v614 = *((_DWORD *)v8 + 12);
            v601 = (_QWORD *)v8[5];
            if ( !v614 )
              goto LABEL_1034;
            v602 = v614 - 1;
          }
          v603 = 1;
          v28 = 0;
          for ( mm = v602 & (v110 + ((v108 >> 9) ^ (v108 >> 4))); ; mm = v602 & v605 )
          {
            v27 = (__int64)&v601[4 * mm];
            if ( *(_QWORD *)v27 == v108 && *(_DWORD *)(v27 + 8) == v110 )
              break;
            if ( !*(_QWORD *)v27 )
            {
              v702 = *(_DWORD *)(v27 + 8);
              if ( v702 == -1 )
                goto LABEL_1024;
              if ( v702 == -2 && !v28 )
                v28 = (__int64)&v601[4 * mm];
            }
            v605 = v603 + mm;
            ++v603;
          }
          goto LABEL_968;
        }
LABEL_723:
        *((_DWORD *)v8 + 8) = (2 * (v464 >> 1) + 2) | v464 & 1;
        if ( *(_QWORD *)v27 || *(_DWORD *)(v27 + 8) != -1 )
          --*((_DWORD *)v8 + 9);
        *(_QWORD *)v27 = v108;
        *(_DWORD *)(v27 + 8) = v110;
        *(__m128i *)(v27 + 16) = _mm_load_si128(&v781);
LABEL_701:
        ++v104;
        v103 += 2;
        v105 = (unsigned int)(v107 + 1);
      }
      while ( v744 != v104 );
      v17 = v741[2 * a3];
LABEL_703:
      if ( (_BYTE *)v785.m128i_i64[0] != v786 )
        _libc_free(v785.m128i_u64[0]);
LABEL_71:
      if ( v782 != v784 )
        _libc_free((unsigned __int64)v782);
      return v17;
    case 208:
      v272 = *(_QWORD *)(v30 + 40);
      v273 = v8[1];
      v274 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v272 + 48LL) + 16LL * *(unsigned int *)(v272 + 8));
      v275 = *(int *)(*(_QWORD *)(v272 + 80) + 96LL);
      v276 = ((unsigned __int16)v274 >> 3) + 36 * v275 - v275;
      v44 = 4 * (unsigned int)(v274 & 7);
      v67 = (*(_DWORD *)(v273 + 4 * v276 + 521536) >> (4 * (v274 & 7))) & 0xF;
      if ( v67 )
        goto LABEL_79;
      v44 = *(unsigned int *)(v30 + 24);
      if ( (unsigned int)v44 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v274 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v44 + v273 + 500 * v274 + 6414);
      goto LABEL_79;
    case 220:
    case 221:
    case 275:
    case 276:
    case 277:
    case 278:
    case 376:
    case 377:
    case 382:
    case 383:
    case 384:
    case 385:
    case 386:
    case 387:
    case 388:
    case 389:
    case 390:
    case 498:
      v124 = *(unsigned int *)(v30 + 24);
      if ( (unsigned int)v124 > 0x1F3 )
        goto LABEL_833;
      v125 = *(unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      if ( !(_WORD)v125 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v124 + v8[1] + 500 * v125 + 6414);
      goto LABEL_79;
    case 298:
      v263 = *(_WORD *)(v30 + 96);
      v264 = *(_QWORD *)(v30 + 104);
      v265 = *(_BYTE *)(v30 + 33) >> 2;
      v785.m128i_i16[0] = v263;
      v67 = v265 & 3;
      v785.m128i_i64[1] = v264;
      v266 = sub_32801E0((__int64)&v785);
      if ( v67 && v266 )
      {
        v449 = v743->m128i_u16[0];
        if ( v263 && (_WORD)v449 )
        {
          v44 = (unsigned int)(4 * v67);
          LOBYTE(v67) = ((int)*(unsigned __int16 *)(v8[1] + 2 * (v263 + 274 * v449 + 71704) + 6) >> (4 * v67)) & 0xF;
        }
        else
        {
          LOBYTE(v67) = 2;
        }
      }
      else
      {
        LOBYTE(v67) = 0;
      }
      goto LABEL_79;
    case 299:
      v267 = *(_QWORD *)(v30 + 104);
      v268 = *(unsigned __int16 *)(v30 + 96);
      LOBYTE(v67) = 0;
      v785.m128i_i16[0] = *(_WORD *)(v30 + 96);
      v785.m128i_i64[1] = v267;
      if ( sub_32801E0((__int64)&v785) )
      {
        LOBYTE(v67) = *(_BYTE *)(v30 + 33) & 4;
        if ( (_BYTE)v67 )
        {
          v269 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                                     + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
          if ( (_WORD)v268 && (_WORD)v269 )
            LOBYTE(v67) = *(_BYTE *)(v268 + v8[1] + 274 * v269 + 443718);
          else
            LOBYTE(v67) = 2;
        }
      }
      goto LABEL_79;
    case 374:
    case 375:
      v143 = *(unsigned int *)(v30 + 24);
      if ( (unsigned int)v143 > 0x1F3 )
        goto LABEL_833;
      v144 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      if ( !(_WORD)v144 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v143 + v8[1] + 500 * v144 + 6414);
      goto LABEL_79;
    case 378:
    case 379:
    case 380:
    case 381:
      v131 = *(unsigned int *)(v30 + 24);
      if ( (unsigned int)v131 > 0x1F3 )
        goto LABEL_833;
      v132 = *(unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      if ( !(_WORD)v132 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v131 + v8[1] + 500 * v132 + 6414);
      if ( (_BYTE)v67 == 1 )
        goto LABEL_732;
      v785.m128i_i64[0] = (__int64)v786;
      v785.m128i_i64[1] = 0x800000000LL;
      goto LABEL_163;
    case 395:
      LOWORD(v67) = v743->m128i_i16[0];
      v245 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v245;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v246 = *(_DWORD *)(v30 + 24);
      if ( v246 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v246 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 396:
      LOWORD(v67) = v743->m128i_i16[0];
      v247 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v247;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v248 = *(_DWORD *)(v30 + 24);
      if ( v248 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v248 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 397:
      LOWORD(v67) = v743->m128i_i16[0];
      v249 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v249;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v250 = *(_DWORD *)(v30 + 24);
      if ( v250 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v250 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 398:
      LOWORD(v67) = v743->m128i_i16[0];
      v251 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v251;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v252 = *(_DWORD *)(v30 + 24);
      if ( v252 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v252 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 399:
      LOWORD(v67) = v743->m128i_i16[0];
      v253 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v253;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v254 = *(_DWORD *)(v30 + 24);
      if ( v254 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v254 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 400:
      v255 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v255;
      v256 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v256 )
        goto LABEL_732;
      v257 = *(_DWORD *)(v30 + 24);
      if ( v257 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v257 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 401:
      v258 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v258;
      v259 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v259 )
        goto LABEL_732;
      v260 = *(_DWORD *)(v30 + 24);
      if ( v260 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v260 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 402:
      LOWORD(v67) = v743->m128i_i16[0];
      v261 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v261;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v262 = *(_DWORD *)(v30 + 24);
      if ( v262 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v262 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 403:
      v380 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v380;
      v381 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v381 )
        goto LABEL_732;
      v382 = *(_DWORD *)(v30 + 24);
      if ( v382 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v382 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 404:
      v383 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v383;
      v384 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v384 )
        goto LABEL_732;
      v385 = *(_DWORD *)(v30 + 24);
      if ( v385 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v385 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 405:
      v386 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v386;
      v387 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v387 )
        goto LABEL_732;
      v388 = *(_DWORD *)(v30 + 24);
      if ( v388 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v388 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 406:
      v389 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v389;
      v390 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v390 )
        goto LABEL_732;
      v391 = *(_DWORD *)(v30 + 24);
      if ( v391 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v391 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 407:
      v392 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v392;
      v393 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v393 )
        goto LABEL_732;
      v394 = *(_DWORD *)(v30 + 24);
      if ( v394 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v394 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 408:
      v395 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v395;
      v396 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v396 )
        goto LABEL_732;
      v397 = *(_DWORD *)(v30 + 24);
      if ( v397 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v397 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 409:
      v398 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v398;
      v399 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v399 )
        goto LABEL_732;
      v400 = *(_DWORD *)(v30 + 24);
      if ( v400 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v400 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 410:
      v401 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v401;
      v402 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v402 )
        goto LABEL_732;
      v403 = *(_DWORD *)(v30 + 24);
      if ( v403 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v403 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 411:
      v404 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v404;
      v405 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v405 )
        goto LABEL_732;
      v406 = *(_DWORD *)(v30 + 24);
      if ( v406 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v406 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 412:
      v407 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v407;
      v408 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v408 )
        goto LABEL_732;
      v409 = *(_DWORD *)(v30 + 24);
      if ( v409 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v409 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 413:
      v410 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v410;
      v411 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v411 )
        goto LABEL_732;
      v412 = *(_DWORD *)(v30 + 24);
      if ( v412 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v412 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 414:
      v413 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v413;
      v414 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v414 )
        goto LABEL_732;
      v415 = *(_DWORD *)(v30 + 24);
      if ( v415 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v415 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 415:
      v235 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v235;
      v236 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v236 )
        goto LABEL_732;
      v237 = *(_DWORD *)(v30 + 24);
      if ( v237 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v237 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 416:
      LOWORD(v67) = v743->m128i_i16[0];
      v238 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v238;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v239 = *(_DWORD *)(v30 + 24);
      if ( v239 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v239 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 417:
      v240 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v240;
      v241 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v241 )
        goto LABEL_732;
      v242 = *(_DWORD *)(v30 + 24);
      if ( v242 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v242 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 418:
      LOWORD(v67) = v743->m128i_i16[0];
      v243 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v243;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v244 = *(_DWORD *)(v30 + 24);
      if ( v244 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v244 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 419:
      v416 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v416;
      v417 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v417 )
        goto LABEL_732;
      v418 = *(_DWORD *)(v30 + 24);
      if ( v418 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v418 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 420:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      v419 = v743->m128i_i16[0];
      v420 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v420;
      if ( !sub_32801E0((__int64)&v785) && v419 != 1 )
        goto LABEL_732;
      v421 = *(_DWORD *)(v30 + 24);
      if ( v421 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v421 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 421:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      v423 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v422 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v423;
      v424 = sub_32801E0((__int64)&v785);
      if ( v422 != 1 && !v424 )
        goto LABEL_732;
      v425 = *(_DWORD *)(v30 + 24);
      if ( v425 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v425 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 422:
      LOWORD(v67) = v743->m128i_i16[0];
      v426 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v426;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v427 = *(_DWORD *)(v30 + 24);
      if ( v427 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v427 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 423:
      v428 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v428;
      v429 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v429 )
        goto LABEL_732;
      v430 = *(_DWORD *)(v30 + 24);
      if ( v430 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v430 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 424:
      LOWORD(v67) = v743->m128i_i16[0];
      v431 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v431;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v432 = *(_DWORD *)(v30 + 24);
      if ( v432 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v432 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 425:
      v433 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v433;
      v434 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v434 )
        goto LABEL_732;
      v435 = *(_DWORD *)(v30 + 24);
      if ( v435 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v435 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 426:
      LOWORD(v67) = v743->m128i_i16[0];
      v436 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v436;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v437 = *(_DWORD *)(v30 + 24);
      if ( v437 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v437 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 427:
      v438 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v438;
      v439 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v439 )
        goto LABEL_732;
      v440 = *(_DWORD *)(v30 + 24);
      if ( v440 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v440 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 428:
      v441 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v441;
      v442 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v442 )
        goto LABEL_732;
      v443 = *(_DWORD *)(v30 + 24);
      if ( v443 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v443 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 429:
      LOWORD(v67) = v743->m128i_i16[0];
      v444 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v444;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v445 = *(_DWORD *)(v30 + 24);
      if ( v445 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v445 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 430:
      v446 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v446;
      v447 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v447 )
        goto LABEL_732;
      v448 = *(_DWORD *)(v30 + 24);
      if ( v448 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v448 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 431:
      LOWORD(v67) = v743->m128i_i16[0];
      v329 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v329;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v330 = *(_DWORD *)(v30 + 24);
      if ( v330 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v330 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 432:
      v331 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v331;
      v332 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v332 )
        goto LABEL_732;
      v333 = *(_DWORD *)(v30 + 24);
      if ( v333 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v333 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 433:
      LOWORD(v67) = v743->m128i_i16[0];
      v334 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v334;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v335 = *(_DWORD *)(v30 + 24);
      if ( v335 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v335 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 434:
      v336 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v336;
      v337 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v337 )
        goto LABEL_732;
      v338 = *(_DWORD *)(v30 + 24);
      if ( v338 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v338 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 435:
      LOWORD(v67) = v743->m128i_i16[0];
      v339 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v339;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v340 = *(_DWORD *)(v30 + 24);
      if ( v340 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v340 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 436:
      v341 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v341;
      v342 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v342 )
        goto LABEL_732;
      v343 = *(_DWORD *)(v30 + 24);
      if ( v343 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v343 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 437:
      LOWORD(v67) = v743->m128i_i16[0];
      v344 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v344;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v345 = *(_DWORD *)(v30 + 24);
      if ( v345 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v345 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 438:
      v346 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v346;
      v347 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v347 )
        goto LABEL_732;
      v348 = *(_DWORD *)(v30 + 24);
      if ( v348 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v348 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 439:
      LOWORD(v67) = v743->m128i_i16[0];
      v349 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v349;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v350 = *(_DWORD *)(v30 + 24);
      if ( v350 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v350 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 440:
      v351 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v351;
      v352 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v352 )
        goto LABEL_732;
      v353 = *(_DWORD *)(v30 + 24);
      if ( v353 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v353 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 441:
      LOWORD(v67) = v743->m128i_i16[0];
      v354 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v354;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v355 = *(_DWORD *)(v30 + 24);
      if ( v355 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v355 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 442:
      v356 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v356;
      v357 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v357 )
        goto LABEL_732;
      v358 = *(_DWORD *)(v30 + 24);
      if ( v358 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v358 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 443:
      LOWORD(v67) = v743->m128i_i16[0];
      v359 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v359;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v360 = *(_DWORD *)(v30 + 24);
      if ( v360 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v360 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 444:
      v361 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v361;
      v362 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v362 )
        goto LABEL_732;
      v363 = *(_DWORD *)(v30 + 24);
      if ( v363 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v363 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 445:
      LOWORD(v67) = v743->m128i_i16[0];
      v364 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v364;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v365 = *(_DWORD *)(v30 + 24);
      if ( v365 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v365 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 446:
      v366 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v366;
      v367 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v367 )
        goto LABEL_732;
      v368 = *(_DWORD *)(v30 + 24);
      if ( v368 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v368 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 447:
      LOWORD(v67) = v743->m128i_i16[0];
      v369 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v369;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v370 = *(_DWORD *)(v30 + 24);
      if ( v370 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v370 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 448:
      v371 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v371;
      v372 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v372 )
        goto LABEL_732;
      v373 = *(_DWORD *)(v30 + 24);
      if ( v373 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v373 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 449:
      LOWORD(v67) = v743->m128i_i16[0];
      v374 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v374;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v375 = *(_DWORD *)(v30 + 24);
      if ( v375 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v375 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 450:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      v377 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v376 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v377;
      v378 = sub_32801E0((__int64)&v785);
      if ( v376 != 1 && !v378 )
        goto LABEL_732;
      v379 = *(_DWORD *)(v30 + 24);
      if ( v379 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v379 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 451:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      v278 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v277 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v278;
      v279 = sub_32801E0((__int64)&v785);
      if ( v277 != 1 && !v279 )
        goto LABEL_732;
      v280 = *(_DWORD *)(v30 + 24);
      if ( v280 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v280 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 452:
      LOWORD(v67) = v743->m128i_i16[0];
      v281 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v281;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v282 = *(_DWORD *)(v30 + 24);
      if ( v282 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v282 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 453:
      LOWORD(v67) = v743->m128i_i16[0];
      v283 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v283;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v284 = *(_DWORD *)(v30 + 24);
      if ( v284 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v284 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 454:
      v285 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v285;
      v286 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v286 )
        goto LABEL_732;
      v287 = *(_DWORD *)(v30 + 24);
      if ( v287 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v287 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 455:
      v288 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v288;
      v289 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v289 )
        goto LABEL_732;
      v290 = *(_DWORD *)(v30 + 24);
      if ( v290 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v290 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 456:
      LOWORD(v67) = v743->m128i_i16[0];
      v291 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v291;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v292 = *(_DWORD *)(v30 + 24);
      if ( v292 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v292 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 457:
      v293 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v293;
      v294 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v294 )
        goto LABEL_732;
      v295 = *(_DWORD *)(v30 + 24);
      if ( v295 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v295 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 458:
      v296 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v296;
      v297 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v297 )
        goto LABEL_732;
      v298 = *(_DWORD *)(v30 + 24);
      if ( v298 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v298 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 459:
      LOWORD(v67) = v743->m128i_i16[0];
      v299 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v299;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v300 = *(_DWORD *)(v30 + 24);
      if ( v300 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v300 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 460:
      v301 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v301;
      v302 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v302 )
        goto LABEL_732;
      v303 = *(_DWORD *)(v30 + 24);
      if ( v303 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v303 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 461:
      LOWORD(v67) = v743->m128i_i16[0];
      v304 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v304;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v305 = *(_DWORD *)(v30 + 24);
      if ( v305 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v305 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 462:
      v306 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v306;
      v307 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v307 )
        goto LABEL_732;
      v308 = *(_DWORD *)(v30 + 24);
      if ( v308 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v308 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 463:
      v309 = *(_QWORD *)(v30 + 40);
      v310 = v8[1];
      v311 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)v309 + 48LL) + 16LL * *(unsigned int *)(v309 + 8));
      v312 = *(int *)(*(_QWORD *)(v309 + 80) + 96LL);
      v755 = v311;
      v313 = (v311 >> 3) + 36 * v312 - v312;
      v44 = 4 * (v311 & 7u);
      v67 = (*(_DWORD *)(v310 + 4 * v313 + 521536) >> (4 * (v311 & 7))) & 0xF;
      if ( v67 )
        goto LABEL_79;
      v315 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v314 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v315;
      v316 = sub_32801E0((__int64)&v785);
      if ( v314 != 1 && !v316 )
        goto LABEL_732;
      v317 = *(_DWORD *)(v30 + 24);
      if ( v317 > 0x1F3 )
        goto LABEL_833;
      if ( !v755 )
        goto LABEL_125;
      v44 = v317;
      LOBYTE(v67) = *(_BYTE *)(v317 + v310 + 500LL * v755 + 6414);
      goto LABEL_79;
    case 464:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(**(_QWORD **)(v30 + 40) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 8LL));
      v319 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v318 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v319;
      v320 = sub_32801E0((__int64)&v785);
      if ( v318 != 1 && !v320 )
        goto LABEL_732;
      v321 = *(_DWORD *)(v30 + 24);
      if ( v321 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v321 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 465:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v322 = v743->m128i_i16[0];
      v323 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v323;
      if ( !sub_32801E0((__int64)&v785) && v322 != 1 )
        goto LABEL_732;
      v324 = *(_DWORD *)(v30 + 24);
      if ( v324 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v324 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 466:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v326 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v325 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v326;
      v327 = sub_32801E0((__int64)&v785);
      if ( v325 != 1 && !v327 )
        goto LABEL_732;
      v328 = *(_DWORD *)(v30 + 24);
      if ( v328 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v328 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 467:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v178 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v177 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v178;
      v179 = sub_32801E0((__int64)&v785);
      if ( v177 != 1 && !v179 )
        goto LABEL_732;
      v180 = *(_DWORD *)(v30 + 24);
      if ( v180 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v180 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 468:
      v181 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v181;
      v182 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v182 )
        goto LABEL_732;
      v183 = *(_DWORD *)(v30 + 24);
      if ( v183 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v183 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 469:
      v184 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v184;
      v185 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v185 )
        goto LABEL_732;
      v186 = *(_DWORD *)(v30 + 24);
      if ( v186 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v186 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 470:
      v187 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v187;
      v188 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v188 )
        goto LABEL_732;
      v189 = *(_DWORD *)(v30 + 24);
      if ( v189 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v189 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 471:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v191 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v190 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v191;
      v192 = sub_32801E0((__int64)&v785);
      if ( v190 != 1 && !v192 )
        goto LABEL_732;
      v193 = *(_DWORD *)(v30 + 24);
      if ( v193 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v193 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 472:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v195 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v194 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v195;
      v196 = sub_32801E0((__int64)&v785);
      if ( v194 != 1 && !v196 )
        goto LABEL_732;
      v197 = *(_DWORD *)(v30 + 24);
      if ( v197 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v197 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 473:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v199 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v198 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v199;
      v200 = sub_32801E0((__int64)&v785);
      if ( v198 != 1 && !v200 )
        goto LABEL_732;
      v201 = *(_DWORD *)(v30 + 24);
      if ( v201 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v201 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 474:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v202 = v743->m128i_i16[0];
      v203 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v203;
      if ( !sub_32801E0((__int64)&v785) && v202 != 1 )
        goto LABEL_732;
      v204 = *(_DWORD *)(v30 + 24);
      if ( v204 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v204 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 475:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v206 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v205 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v206;
      v207 = sub_32801E0((__int64)&v785);
      if ( v205 != 1 && !v207 )
        goto LABEL_732;
      v208 = *(_DWORD *)(v30 + 24);
      if ( v208 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v208 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 476:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v210 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v209 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v210;
      v211 = sub_32801E0((__int64)&v785);
      if ( v209 != 1 && !v211 )
        goto LABEL_732;
      v212 = *(_DWORD *)(v30 + 24);
      if ( v212 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v212 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 477:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v214 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v213 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v214;
      v215 = sub_32801E0((__int64)&v785);
      if ( v213 != 1 && !v215 )
        goto LABEL_732;
      v216 = *(_DWORD *)(v30 + 24);
      if ( v216 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v216 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 478:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v217 = v743->m128i_i16[0];
      v218 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v218;
      if ( !sub_32801E0((__int64)&v785) && v217 != 1 )
        goto LABEL_732;
      v219 = *(_DWORD *)(v30 + 24);
      if ( v219 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v219 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 479:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v221 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v220 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v221;
      v222 = sub_32801E0((__int64)&v785);
      if ( v220 != 1 && !v222 )
        goto LABEL_732;
      v223 = *(_DWORD *)(v30 + 24);
      if ( v223 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v223 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 480:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v224 = v743->m128i_i16[0];
      v225 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v225;
      if ( !sub_32801E0((__int64)&v785) && v224 != 1 )
        goto LABEL_732;
      v226 = *(_DWORD *)(v30 + 24);
      if ( v226 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v226 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 481:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v228 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v227 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v228;
      v229 = sub_32801E0((__int64)&v785);
      if ( v227 != 1 && !v229 )
        goto LABEL_732;
      v230 = *(_DWORD *)(v30 + 24);
      if ( v230 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v230 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 482:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v232 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v231 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v232;
      v233 = sub_32801E0((__int64)&v785);
      if ( v231 != 1 && !v233 )
        goto LABEL_732;
      v234 = *(_DWORD *)(v30 + 24);
      if ( v234 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v234 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 483:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v152 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v151 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v152;
      v153 = sub_32801E0((__int64)&v785);
      if ( v151 != 1 && !v153 )
        goto LABEL_732;
      v154 = *(_DWORD *)(v30 + 24);
      if ( v154 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v154 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 484:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v156 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v155 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v156;
      v157 = sub_32801E0((__int64)&v785);
      if ( v155 != 1 && !v157 )
        goto LABEL_732;
      v158 = *(_DWORD *)(v30 + 24);
      if ( v158 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v158 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 485:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v159 = v743->m128i_i16[0];
      v160 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v160;
      if ( !sub_32801E0((__int64)&v785) && v159 != 1 )
        goto LABEL_732;
      v161 = *(_DWORD *)(v30 + 24);
      if ( v161 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v161 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 486:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v162 = v743->m128i_i16[0];
      v163 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v163;
      if ( !sub_32801E0((__int64)&v785) && v162 != 1 )
        goto LABEL_732;
      v164 = *(_DWORD *)(v30 + 24);
      if ( v164 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v164 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 487:
      LOWORD(v67) = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(v30 + 40) + 48LL));
      v166 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v165 = v785.m128i_i16[0];
      v785.m128i_i64[1] = v166;
      v167 = sub_32801E0((__int64)&v785);
      if ( v165 != 1 && !v167 )
        goto LABEL_732;
      v168 = *(_DWORD *)(v30 + 24);
      if ( v168 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v168 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 488:
      v169 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v169;
      v170 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v170 )
        goto LABEL_732;
      v171 = *(_DWORD *)(v30 + 24);
      if ( v171 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v171 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 489:
      LOWORD(v67) = v743->m128i_i16[0];
      v172 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      v785.m128i_i64[1] = v172;
      if ( !sub_32801E0((__int64)&v785) && (_WORD)v67 != 1 )
        goto LABEL_732;
      v173 = *(_DWORD *)(v30 + 24);
      if ( v173 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v173 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 490:
      v174 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v174;
      v175 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v175 )
        goto LABEL_732;
      v176 = *(_DWORD *)(v30 + 24);
      if ( v176 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v176 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 491:
      v145 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v145;
      v146 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v146 )
        goto LABEL_732;
      v147 = *(_DWORD *)(v30 + 24);
      if ( v147 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v147 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    case 492:
      v148 = v743->m128i_i64[1];
      v785.m128i_i16[0] = v743->m128i_i16[0];
      LOWORD(v67) = v785.m128i_i16[0];
      v785.m128i_i64[1] = v148;
      v149 = sub_32801E0((__int64)&v785);
      if ( (_WORD)v67 != 1 && !v149 )
        goto LABEL_732;
      v150 = *(_DWORD *)(v30 + 24);
      if ( v150 > 0x1F3 )
        goto LABEL_833;
      if ( !(_WORD)v67 )
        goto LABEL_125;
      LOBYTE(v67) = *(_BYTE *)(v150 + v8[1] + 500LL * (unsigned __int16)v67 + 6414);
      goto LABEL_79;
    default:
      goto LABEL_70;
  }
}
