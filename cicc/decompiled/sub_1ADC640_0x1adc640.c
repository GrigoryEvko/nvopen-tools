// Function: sub_1ADC640
// Address: 0x1adc640
//
_BOOL8 __fastcall sub_1ADC640(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __m128 a6,
        __m128i a7,
        __m128i si128,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  char v16; // dl
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 *v19; // rax
  unsigned __int64 v21; // rdi
  bool v22; // al
  int v23; // eax
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rbx
  _QWORD *v28; // rdi
  __int64 v29; // rbx
  __int16 v30; // ax
  __int64 v31; // rbx
  __int64 v32; // rax
  size_t v33; // rdx
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rbx
  __int64 v37; // rdx
  int v38; // r13d
  __int64 v39; // rax
  char v40; // al
  unsigned __int64 *v41; // r15
  __int64 *v42; // rax
  unsigned __int64 v43; // rax
  _QWORD *v44; // rdi
  __int64 v45; // r12
  unsigned __int64 v46; // r15
  _QWORD *v47; // rdi
  char v48; // r15
  unsigned int v49; // eax
  int v50; // r8d
  int v51; // r9d
  _QWORD *v52; // r15
  __int64 v53; // rax
  unsigned __int8 *v54; // rax
  double v55; // xmm4_8
  double v56; // xmm5_8
  __int64 *v57; // r12
  __int64 *v58; // rcx
  __int64 v59; // rax
  __int64 v60; // r8
  __int64 v61; // rbx
  __m128i v62; // rax
  __int64 v63; // r14
  __int64 *v64; // r12
  unsigned __int64 v65; // rbx
  int v66; // edx
  int v67; // r13d
  __int64 v68; // rsi
  unsigned __int8 *v69; // r13
  unsigned __int8 *v70; // rbx
  __int64 v71; // r12
  _QWORD *v72; // rsi
  _QWORD *v73; // rdi
  __int64 *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 *v77; // rbx
  __int64 v78; // rdi
  unsigned __int64 v79; // rbx
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // r12
  __int64 v83; // r12
  unsigned __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r12
  __int64 v88; // rax
  __int64 v89; // r12
  __int64 v90; // rax
  unsigned __int64 v91; // r13
  char *v92; // r8
  __int64 v93; // r13
  char *v94; // rax
  void *v95; // r9
  void *v96; // rdi
  __int64 v97; // rcx
  unsigned __int64 v98; // rsi
  __m128 *v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 *v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // rax
  __int64 v106; // rsi
  unsigned __int32 v107; // eax
  __int64 v108; // rdi
  __int64 v109; // rsi
  double v110; // xmm4_8
  double v111; // xmm5_8
  __int64 v112; // rax
  __int64 *v113; // rbx
  __int64 v114; // rdi
  __int64 v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rcx
  int v118; // r8d
  int v119; // r9d
  double v120; // xmm4_8
  double v121; // xmm5_8
  __int64 v122; // rdi
  __int64 *v123; // rsi
  __int64 v124; // rdx
  __int64 *v125; // r13
  _QWORD *v126; // r13
  _QWORD *v127; // rbx
  _QWORD *j; // r12
  __int64 v129; // rax
  __int64 v130; // rax
  unsigned __int64 v131; // rsi
  char *v132; // rax
  __int64 v133; // rsi
  _QWORD *v134; // r13
  _QWORD *v135; // rbx
  __int64 v136; // rdx
  __int64 v137; // rax
  __int64 v138; // rdi
  unsigned __int64 *v139; // r8
  int v140; // r9d
  __int64 v141; // rax
  unsigned __int64 *v142; // rbx
  __int64 *v143; // r15
  unsigned __int64 *v144; // r14
  __int64 v145; // rax
  __int64 v146; // rax
  unsigned __int64 v147; // rcx
  unsigned __int64 v148; // rax
  __int64 *v149; // rsi
  __int64 **v150; // r12
  __int64 **v151; // rbx
  __int64 *v152; // rdi
  __int64 v153; // rdx
  int v154; // r8d
  int v155; // r9d
  unsigned int k; // r12d
  __int64 v157; // rax
  __int64 v158; // r15
  int v159; // r15d
  __int64 v160; // rax
  int v161; // eax
  __int64 v162; // rbx
  __int64 v163; // rax
  __int64 v164; // rbx
  __int64 v165; // rax
  unsigned __int64 v166; // r14
  int v167; // ebx
  int v168; // eax
  int v169; // r15d
  __int64 v170; // rax
  int v171; // eax
  __int64 *v172; // rbx
  __int64 *v173; // r13
  __int64 *v174; // r12
  __int64 *v175; // r14
  __int64 v176; // rax
  __int64 v177; // rax
  __int16 v178; // ax
  __int64 *v179; // rax
  __int64 v180; // rax
  __int64 *v181; // rax
  __int64 *v182; // rcx
  int v183; // r8d
  __int64 v184; // rax
  __int64 v185; // r14
  __int64 *v186; // rax
  int v187; // r9d
  __int64 v188; // rax
  __int64 v189; // rdx
  __int64 v190; // r14
  int v191; // r14d
  __int64 v192; // rax
  __int64 v193; // rdx
  __int64 v194; // rax
  __int64 *v195; // r8
  __int64 v196; // rdx
  __int64 *v197; // r14
  __int64 v198; // rcx
  unsigned __int64 v199; // rdx
  __m128i *v200; // rax
  int v201; // esi
  int v202; // r14d
  __int64 v203; // rax
  void *v204; // r9
  __int64 v205; // r8
  __int64 *v206; // rdx
  __int64 v207; // rdx
  __int64 v208; // rsi
  unsigned __int8 *v209; // rsi
  char v210; // cl
  __int16 v211; // dx
  double v212; // xmm4_8
  double v213; // xmm5_8
  unsigned int v214; // r15d
  __int64 v215; // r8
  int v216; // r9d
  __int64 v217; // rax
  unsigned __int64 v218; // rdx
  char *v219; // r11
  __int64 v220; // r9
  void *v221; // rdi
  __int64 v222; // rsi
  __int64 v223; // r15
  __int64 v224; // rdx
  __int64 v225; // rax
  unsigned __int64 v226; // r14
  unsigned __int64 v227; // rdx
  __int64 v228; // rax
  __int64 v229; // r12
  __int64 v230; // rbx
  _QWORD *v231; // rax
  __int64 v232; // r14
  __int64 *v233; // rbx
  __int64 v234; // r14
  __int64 v235; // rax
  __int64 *v236; // rsi
  double v237; // xmm4_8
  double v238; // xmm5_8
  __int64 *v239; // rcx
  __int64 v240; // r15
  unsigned __int8 v241; // al
  unsigned __int64 v242; // rdx
  unsigned __int64 v243; // rax
  __int64 *v244; // rcx
  __int64 *v245; // rax
  __int64 v246; // rax
  unsigned __int64 v247; // r13
  unsigned __int32 v248; // eax
  __m128 *v249; // rbx
  __int64 *v250; // rax
  __int64 v251; // rdx
  __int64 v252; // r12
  double v253; // xmm4_8
  double v254; // xmm5_8
  __int64 v255; // rbx
  _QWORD *v256; // r12
  __int64 v257; // rdi
  unsigned __int64 v258; // rax
  __int64 v259; // rdi
  int v260; // edx
  unsigned __int64 v261; // rax
  __int64 v262; // rcx
  __int64 v263; // rax
  int v264; // r8d
  int v265; // r9d
  __int64 v266; // rdx
  _WORD *v267; // r14
  __int64 *v268; // rcx
  _QWORD *v269; // r12
  unsigned __int64 v270; // rax
  __int64 v271; // r13
  __int64 v272; // rax
  size_t v273; // rbx
  int v274; // r8d
  __int64 v275; // rax
  __int64 v276; // rdx
  __int64 v277; // r12
  int v278; // r12d
  __int64 v279; // rax
  __int64 v280; // rdx
  __int64 v281; // rcx
  int v282; // eax
  __int64 *v283; // r9
  __int64 v284; // rdx
  __m128i *v285; // rax
  __int64 *v286; // r15
  __int64 v287; // rdx
  int v288; // ecx
  unsigned __int64 v289; // r12
  __int64 v290; // rax
  _QWORD *v291; // rax
  __int64 v292; // rdx
  __int64 v293; // r12
  char v294; // al
  int v295; // r15d
  __int64 *v296; // r15
  __int64 v297; // rax
  __int64 v298; // rcx
  __int64 v299; // rsi
  unsigned __int8 *v300; // rsi
  bool v301; // zf
  __int64 v302; // r12
  _QWORD *v303; // rax
  _QWORD *v304; // rbx
  unsigned __int64 *v305; // r12
  __int64 v306; // rdx
  unsigned __int64 v307; // rdi
  __int64 v308; // r13
  __int64 *v309; // r15
  __int64 v310; // rdi
  __int64 *v311; // rbx
  char **v312; // rcx
  __int64 *v313; // rbx
  __int64 *v314; // rax
  unsigned __int64 v315; // rbx
  _QWORD *v316; // rdi
  __int64 v317; // rax
  unsigned int v318; // eax
  __int64 v319; // r12
  __int64 v320; // rax
  unsigned __int32 v321; // eax
  unsigned __int32 v322; // eax
  __int64 v323; // rax
  __int64 **v324; // rax
  __int64 *v325; // r9
  __int64 v326; // r15
  __int64 v327; // rbx
  __int64 v328; // r14
  __int64 v329; // rdx
  __m128i *v330; // rdi
  size_t v331; // rdx
  _QWORD *v332; // rax
  __int64 v333; // rbx
  __int64 *v334; // r12
  __int64 v335; // rdi
  _QWORD *v336; // rdi
  __int64 v337; // rax
  __int64 v338; // r15
  __int64 v339; // rax
  __int64 v340; // r13
  _QWORD *v341; // rax
  _QWORD *v342; // r14
  unsigned __int64 v343; // rsi
  __int64 v344; // rax
  __int64 v345; // rsi
  __int64 v346; // rdx
  unsigned __int8 *v347; // rsi
  __int64 v348; // r15
  size_t v349; // r12
  __int64 v350; // r14
  unsigned __int8 v351; // al
  __int64 v352; // r12
  __m128i *v353; // rax
  size_t v354; // rdx
  __int64 *v355; // r12
  __int64 v356; // rax
  unsigned __int64 v357; // r15
  _QWORD *v358; // rax
  double v359; // xmm4_8
  double v360; // xmm5_8
  unsigned __int64 v361; // rcx
  __int64 v362; // rax
  __int64 v363; // rax
  __int64 v364; // r13
  __int64 v365; // r12
  __int64 v366; // r14
  unsigned __int8 *v367; // rax
  int v368; // ebx
  size_t v369; // rdx
  __int64 v370; // rax
  double v371; // xmm4_8
  double v372; // xmm5_8
  double v373; // xmm4_8
  double v374; // xmm5_8
  __int64 v375; // rcx
  __int64 v376; // r9
  __int64 v377; // rbx
  unsigned __int64 v378; // r12
  __int64 v379; // rsi
  __int64 v380; // rdx
  unsigned int v381; // eax
  unsigned __int64 v382; // rbx
  __int64 v383; // rdx
  __int64 v384; // rsi
  unsigned __int8 *v385; // rsi
  __int64 v386; // r12
  __m128i *v387; // rax
  __m128i *v388; // r15
  __int64 v389; // rax
  __int64 *v390; // rdi
  __int64 *v391; // r12
  __int64 v392; // rax
  __int64 v393; // rcx
  __int64 v394; // rax
  __int64 v395; // r12
  __int64 v396; // rax
  __int64 v397; // rax
  __int64 v398; // rax
  int v399; // eax
  int v400; // ebx
  __int64 *v401; // r13
  __int64 v402; // r15
  __int64 *kk; // r14
  __int64 v404; // rdi
  unsigned __int64 v405; // rax
  int v406; // r8d
  int v407; // r9d
  __int64 v408; // rdx
  __int64 *v409; // r12
  __int64 *v410; // rax
  __int64 *v411; // rbx
  __int64 *ii; // r13
  _QWORD *v413; // rax
  _QWORD *v414; // rbx
  unsigned __int64 v415; // rsi
  __int64 v416; // rax
  __int64 v417; // rsi
  unsigned __int8 *v418; // rsi
  double v419; // xmm4_8
  double v420; // xmm5_8
  _QWORD *v421; // r12
  __int64 *v422; // rbx
  __int64 v423; // rdi
  unsigned __int64 v424; // r8
  __int64 *v425; // r14
  unsigned __int64 v426; // rcx
  __int64 v427; // rax
  unsigned __int64 *v428; // rcx
  unsigned __int64 v429; // rdx
  double v430; // xmm4_8
  double v431; // xmm5_8
  unsigned __int64 *v432; // rcx
  unsigned __int64 v433; // rdx
  __int64 v434; // rdx
  __int64 v435; // r12
  __int64 v436; // r8
  __int64 v437; // rsi
  double v438; // xmm4_8
  double v439; // xmm5_8
  _QWORD *v440; // rbx
  __int64 v441; // r12
  __int64 v442; // rdi
  size_t v443; // rdx
  __int64 v444; // rsi
  __int64 v445; // r12
  _QWORD *v446; // r14
  unsigned __int64 *v447; // r13
  __int64 *v448; // rbx
  unsigned __int64 v449; // rdx
  unsigned __int64 v450; // rcx
  __int64 v451; // rax
  __int64 *v452; // r12
  __int64 v453; // r13
  __int64 v454; // r12
  __int64 v455; // rbx
  __int64 v456; // rax
  __int64 *v457; // r14
  __int64 *v458; // rbx
  __int64 v459; // r13
  __int64 v460; // rax
  unsigned __int8 *v461; // rsi
  _QWORD *v462; // rbx
  _QWORD *v463; // r12
  __int64 v464; // r12
  __int64 v465; // rax
  __int64 *v466; // rcx
  _QWORD *v467; // rbx
  unsigned __int64 *v468; // rcx
  unsigned __int64 v469; // rdx
  double v470; // xmm4_8
  double v471; // xmm5_8
  __m128i *v472; // rbx
  __int64 v473; // rsi
  __int64 *v474; // rbx
  __int64 *v475; // r14
  __int64 m128i_i64; // rdx
  __m128i *v477; // r15
  __int64 v478; // rsi
  unsigned __int8 *v479; // rsi
  __int64 v480; // rbx
  __int64 v481; // rax
  __int64 v482; // rax
  double v483; // xmm4_8
  double v484; // xmm5_8
  __int64 v485; // rdi
  char v486; // al
  __int64 v487; // rbx
  _QWORD *v488; // rax
  __int64 v489; // rdi
  __int64 v490; // rax
  __int64 v491; // rax
  unsigned __int64 v492; // kr10_8
  __int64 v493; // r14
  _QWORD *v494; // rax
  __int64 v495; // rax
  __int64 i; // r12
  __int64 v497; // rdi
  __int64 v498; // r13
  __int64 v499; // r12
  __int64 v500; // rdi
  unsigned __int64 v501; // rax
  __int64 v502; // rax
  double v503; // xmm4_8
  double v504; // xmm5_8
  __int64 v505; // rax
  double v506; // xmm4_8
  double v507; // xmm5_8
  __int64 v508; // r12
  __int64 v509; // [rsp-10h] [rbp-500h]
  unsigned __int64 *v510; // [rsp+18h] [rbp-4D8h]
  __int64 v511; // [rsp+20h] [rbp-4D0h]
  _QWORD *v512; // [rsp+38h] [rbp-4B8h]
  unsigned __int64 v513; // [rsp+48h] [rbp-4A8h]
  __int64 *v514; // [rsp+48h] [rbp-4A8h]
  __int64 v515; // [rsp+50h] [rbp-4A0h]
  unsigned __int64 v516; // [rsp+58h] [rbp-498h]
  __int64 v517; // [rsp+58h] [rbp-498h]
  __int64 *v518; // [rsp+60h] [rbp-490h]
  char *v519; // [rsp+60h] [rbp-490h]
  __int64 *v520; // [rsp+60h] [rbp-490h]
  __int64 v521; // [rsp+68h] [rbp-488h]
  size_t na; // [rsp+70h] [rbp-480h]
  size_t nd; // [rsp+70h] [rbp-480h]
  void *nc; // [rsp+70h] [rbp-480h]
  int nb; // [rsp+70h] [rbp-480h]
  size_t n; // [rsp+70h] [rbp-480h]
  __int64 v527; // [rsp+78h] [rbp-478h]
  __int64 *m; // [rsp+80h] [rbp-470h]
  _QWORD *v530; // [rsp+88h] [rbp-468h]
  char v531; // [rsp+88h] [rbp-468h]
  void *v532; // [rsp+90h] [rbp-460h]
  char *v533; // [rsp+90h] [rbp-460h]
  char v534; // [rsp+90h] [rbp-460h]
  unsigned __int64 v535; // [rsp+98h] [rbp-458h]
  char v536; // [rsp+A6h] [rbp-44Ah]
  bool v537; // [rsp+A7h] [rbp-449h]
  __int64 v538; // [rsp+A8h] [rbp-448h]
  __int64 v541; // [rsp+B8h] [rbp-438h]
  int v542; // [rsp+B8h] [rbp-438h]
  int v543; // [rsp+B8h] [rbp-438h]
  __int64 v544; // [rsp+B8h] [rbp-438h]
  __int64 v545; // [rsp+B8h] [rbp-438h]
  __int64 *v546; // [rsp+B8h] [rbp-438h]
  __int64 v547; // [rsp+D0h] [rbp-420h]
  int v548; // [rsp+D0h] [rbp-420h]
  __int64 *jj; // [rsp+D8h] [rbp-418h]
  __int64 *v551; // [rsp+D8h] [rbp-418h]
  unsigned __int64 v552; // [rsp+D8h] [rbp-418h]
  char v553; // [rsp+E0h] [rbp-410h]
  __int64 v554; // [rsp+E0h] [rbp-410h]
  __int16 v555; // [rsp+E0h] [rbp-410h]
  __int64 v556; // [rsp+E0h] [rbp-410h]
  __m128i *v557; // [rsp+E0h] [rbp-410h]
  __int64 v558; // [rsp+E0h] [rbp-410h]
  unsigned __int64 *v559; // [rsp+E0h] [rbp-410h]
  __int64 v560; // [rsp+E8h] [rbp-408h]
  _QWORD *v561; // [rsp+E8h] [rbp-408h]
  const void *v562; // [rsp+E8h] [rbp-408h]
  __int64 *v563; // [rsp+E8h] [rbp-408h]
  __int64 v564; // [rsp+E8h] [rbp-408h]
  __int64 *v565; // [rsp+E8h] [rbp-408h]
  __int64 v566; // [rsp+F0h] [rbp-400h]
  __int64 v567; // [rsp+F0h] [rbp-400h]
  int v568; // [rsp+F0h] [rbp-400h]
  __int64 *v569; // [rsp+F0h] [rbp-400h]
  unsigned int v570; // [rsp+F0h] [rbp-400h]
  unsigned __int64 *v571; // [rsp+F0h] [rbp-400h]
  __int64 v572; // [rsp+F0h] [rbp-400h]
  __int64 *v573; // [rsp+F0h] [rbp-400h]
  __int64 *v574; // [rsp+F0h] [rbp-400h]
  _QWORD *v575; // [rsp+F0h] [rbp-400h]
  __int64 *v576; // [rsp+F8h] [rbp-3F8h]
  __int64 *v577; // [rsp+F8h] [rbp-3F8h]
  __int64 *v578; // [rsp+F8h] [rbp-3F8h]
  __int64 *v579; // [rsp+F8h] [rbp-3F8h]
  __int64 *v580; // [rsp+F8h] [rbp-3F8h]
  __int64 *v581; // [rsp+100h] [rbp-3F0h]
  __int64 **v582; // [rsp+100h] [rbp-3F0h]
  _QWORD *v583; // [rsp+100h] [rbp-3F0h]
  void *v584; // [rsp+100h] [rbp-3F0h]
  void *v585; // [rsp+100h] [rbp-3F0h]
  __int64 v586; // [rsp+108h] [rbp-3E8h] BYREF
  unsigned __int8 *v587[2]; // [rsp+110h] [rbp-3E0h] BYREF
  __int16 v588; // [rsp+120h] [rbp-3D0h]
  __m128i *v589; // [rsp+130h] [rbp-3C0h] BYREF
  unsigned __int64 v590; // [rsp+138h] [rbp-3B8h]
  __int64 v591; // [rsp+140h] [rbp-3B0h]
  __int16 v592; // [rsp+150h] [rbp-3A0h] BYREF
  _QWORD *v593; // [rsp+158h] [rbp-398h]
  _QWORD *v594; // [rsp+160h] [rbp-390h]
  __int64 v595; // [rsp+168h] [rbp-388h]
  void *src; // [rsp+170h] [rbp-380h] BYREF
  __int64 v597; // [rsp+178h] [rbp-378h]
  _QWORD v598[4]; // [rsp+180h] [rbp-370h] BYREF
  char *v599; // [rsp+1A0h] [rbp-350h] BYREF
  __int64 v600; // [rsp+1A8h] [rbp-348h]
  _BYTE v601[32]; // [rsp+1B0h] [rbp-340h] BYREF
  __m128i *v602; // [rsp+1D0h] [rbp-320h] BYREF
  __int64 v603; // [rsp+1D8h] [rbp-318h]
  __m128i v604[2]; // [rsp+1E0h] [rbp-310h] BYREF
  __int64 *v605; // [rsp+200h] [rbp-2F0h] BYREF
  __int64 v606; // [rsp+208h] [rbp-2E8h]
  _BYTE v607[64]; // [rsp+210h] [rbp-2E0h] BYREF
  __int64 v608; // [rsp+250h] [rbp-2A0h] BYREF
  __int64 v609; // [rsp+258h] [rbp-298h]
  __int64 v610; // [rsp+260h] [rbp-290h] BYREF
  unsigned int v611; // [rsp+268h] [rbp-288h]
  _QWORD *v612; // [rsp+278h] [rbp-278h]
  unsigned int v613; // [rsp+288h] [rbp-268h]
  char v614; // [rsp+290h] [rbp-260h]
  char v615; // [rsp+299h] [rbp-257h]
  __m128i *v616; // [rsp+2A0h] [rbp-250h] BYREF
  size_t v617; // [rsp+2A8h] [rbp-248h] BYREF
  __m128i v618; // [rsp+2B0h] [rbp-240h] BYREF
  __int64 v619; // [rsp+2C0h] [rbp-230h]
  int v620; // [rsp+2C8h] [rbp-228h]
  __int64 v621; // [rsp+2D0h] [rbp-220h]
  __int64 v622; // [rsp+2D8h] [rbp-218h]
  __m128i v623; // [rsp+2F0h] [rbp-200h] BYREF
  __int64 v624; // [rsp+300h] [rbp-1F0h] BYREF
  __int64 v625; // [rsp+308h] [rbp-1E8h]
  __int64 *v626; // [rsp+310h] [rbp-1E0h]
  int v627; // [rsp+318h] [rbp-1D8h]
  __int64 v628; // [rsp+320h] [rbp-1D0h]
  __int64 v629; // [rsp+328h] [rbp-1C8h]

  v535 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = *(unsigned int *)(a2 + 96);
  v14 = *(_QWORD *)(a2 + 88);
  v586 = a1;
  v15 = v14 + 24 * v13;
  *(_DWORD *)(a2 + 48) = 0;
  if ( v14 == v15 )
  {
    v17 = a1 & 0xFFFFFFFFFFFFFFF8LL;
    v16 = a1;
  }
  else
  {
    do
    {
      v15 -= 24;
      sub_1455FA0(v15);
    }
    while ( v14 != v15 );
    v16 = v586;
    v17 = v586 & 0xFFFFFFFFFFFFFFF8LL;
  }
  *(_DWORD *)(a2 + 96) = 0;
  *(_DWORD *)(a2 + 304) = 0;
  v18 = v17 - 24;
  v19 = (__int64 *)(v17 - 72);
  if ( (v16 & 4) != 0 )
    v19 = (__int64 *)v18;
  v538 = *v19;
  if ( *(_BYTE *)(*v19 + 16) )
    return 0;
  v537 = sub_15E4F60(v538);
  if ( v537 )
    return 0;
  v21 = v586 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v586 & 4) != 0 )
    v22 = (unsigned int)sub_165C280(v21) != 0;
  else
    v22 = (unsigned int)sub_165C2D0(v21) != 0;
  if ( v22 )
  {
    v23 = sub_1AD4AC0(&v586);
    if ( v23 )
    {
      v24 = 0;
      v25 = 16 * ((unsigned int)(v23 - 1) + 1LL);
      do
      {
        v26 = 0;
        if ( *(char *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0 )
          v26 = sub_1648A40(v586 & 0xFFFFFFFFFFFFFFF8LL);
        if ( *(_DWORD *)(*(_QWORD *)(v26 + v24) + 8LL) > 1u )
          return 0;
        v24 += 16;
      }
      while ( v25 != v24 );
    }
  }
  v27 = v586 & 0xFFFFFFFFFFFFFFF8LL;
  v28 = (_QWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v586 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v28, -1, 30) )
    {
LABEL_21:
      v553 = 1;
      goto LABEL_22;
    }
    v389 = *(_QWORD *)(v27 - 24);
    if ( !*(_BYTE *)(v389 + 16) )
      goto LABEL_568;
LABEL_555:
    v553 = 0;
    goto LABEL_22;
  }
  if ( (unsigned __int8)sub_1560260(v28, -1, 30) )
    goto LABEL_21;
  v389 = *(_QWORD *)(v27 - 72);
  if ( *(_BYTE *)(v389 + 16) )
    goto LABEL_555;
LABEL_568:
  v623.m128i_i64[0] = *(_QWORD *)(v389 + 112);
  v553 = sub_1560260(&v623, -1, 30);
LABEL_22:
  v29 = *(_QWORD *)(*(_QWORD *)(v535 + 40) + 56LL);
  v512 = *(_QWORD **)(v535 + 40);
  v30 = *(_WORD *)(v538 + 18);
  v521 = v29;
  if ( (v30 & 0x4000) != 0 )
  {
    if ( (*(_BYTE *)(v29 + 19) & 0x40) != 0 )
    {
      v31 = sub_15E0FA0(v29);
      v32 = sub_15E0FA0(v538);
      v33 = *(_QWORD *)(v32 + 8);
      if ( v33 != *(_QWORD *)(v31 + 8) || v33 && memcmp(*(const void **)v32, *(const void **)v31, v33) )
        return 0;
      v30 = *(_WORD *)(v538 + 18);
    }
    else
    {
      v396 = sub_15E0FA0(v538);
      v623.m128i_i64[0] = (__int64)&v624;
      sub_1AD33C0(v623.m128i_i64, *(_BYTE **)v396, *(_QWORD *)v396 + *(_QWORD *)(v396 + 8));
      sub_15E4280(v29, &v623);
      sub_2240A30(&v623);
      v30 = *(_WORD *)(v538 + 18);
    }
  }
  if ( (v30 & 8) != 0 )
  {
    v394 = sub_15E38F0(v538);
    v395 = sub_1649C60(v394);
    if ( (*(_BYTE *)(v521 + 18) & 8) == 0 )
    {
      if ( !v395 )
        goto LABEL_30;
      goto LABEL_565;
    }
    v397 = sub_15E38F0(v521);
    v398 = sub_1649C60(v397);
    if ( !v395 )
    {
      v395 = v398;
LABEL_758:
      if ( !v395 )
        goto LABEL_30;
LABEL_573:
      v399 = sub_14DD7D0(v395);
      v400 = v399;
      if ( v399 > 10 )
      {
        if ( v399 != 12 )
          goto LABEL_30;
      }
      else if ( v399 <= 6 )
      {
        goto LABEL_30;
      }
      sub_1AD4B10((__int64)&v623, &v586, 1);
      if ( (_BYTE)v625 && (v515 = *(_QWORD *)v623.m128i_i64[0]) != 0 )
      {
        if ( v400 == 9 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v623.m128i_i64[0] + 16LL) == 73 )
          {
            for ( i = *(_QWORD *)(v538 + 80); i != v538 + 72; i = *(_QWORD *)(i + 8) )
            {
              v497 = i - 24;
              if ( !i )
                v497 = 0;
              if ( *(_BYTE *)(sub_157ED20(v497) + 16) == 34 )
                return v537;
            }
          }
        }
        else if ( (unsigned int)(v400 - 7) <= 1 )
        {
          v498 = *(_QWORD *)(v538 + 80);
          if ( v498 != v538 + 72 )
          {
            v499 = 0x40018000000001LL;
            do
            {
              v500 = v498 - 24;
              if ( !v498 )
                v500 = 0;
              v501 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v500) + 16) - 34;
              if ( (unsigned int)v501 <= 0x36 )
              {
                if ( _bittest64(&v499, v501) )
                  return v537;
              }
              v498 = *(_QWORD *)(v498 + 8);
            }
            while ( v498 != v538 + 72 );
          }
        }
        v536 = (v586 >> 2) & 1;
        if ( ((v586 >> 2) & 1) != 0 )
        {
          v623 = 0u;
          v624 = 0;
          LODWORD(v625) = 0;
          v491 = sub_1AD6830(v515, (__int64)&v623);
          v536 = 0;
          if ( v491 )
            v536 = *(_BYTE *)(v491 + 16) != 16;
          j___libc_free_0(v623.m128i_i64[1]);
        }
      }
      else
      {
        v536 = 0;
        v515 = 0;
      }
      goto LABEL_31;
    }
    if ( !v398 )
    {
LABEL_565:
      sub_15E3D80(v521, v395);
      goto LABEL_30;
    }
    if ( v398 == v395 )
      goto LABEL_573;
    return 0;
  }
  if ( (*(_BYTE *)(v521 + 18) & 8) != 0 )
  {
    v490 = sub_15E38F0(v521);
    v395 = sub_1649C60(v490);
    goto LABEL_758;
  }
LABEL_30:
  v536 = 0;
  v515 = 0;
LABEL_31:
  sub_1ADACC0(v521, v538);
  v34 = *(_QWORD *)(v521 + 72);
  v527 = v521 + 72;
  v592 = 0;
  v513 = v34 & 0xFFFFFFFFFFFFFFF8LL;
  v605 = (__int64 *)v607;
  v606 = 0x800000000LL;
  v593 = 0;
  v594 = 0;
  v595 = 0;
  v608 = 0;
  v35 = sub_1454B60(0x56u);
  v611 = v35;
  if ( v35 )
  {
    v609 = sub_22077B0((unsigned __int64)v35 << 6);
    sub_1954940((__int64)&v608);
  }
  else
  {
    v609 = 0;
    v610 = 0;
  }
  v614 = 0;
  v616 = &v618;
  v617 = 0x400000000LL;
  v615 = 1;
  v516 = sub_1632FA0(*(_QWORD *)(v521 + 40));
  v576 = (__int64 *)((v586 & 0xFFFFFFFFFFFFFFF8LL) - 24LL
                                                   * (*(_DWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( (*(_BYTE *)(v538 + 18) & 1) != 0 )
  {
    sub_15E08E0(v538, v538);
    v36 = *(_QWORD *)(v538 + 88);
    if ( (*(_BYTE *)(v538 + 18) & 1) != 0 )
      sub_15E08E0(v538, v538);
    v37 = *(_QWORD *)(v538 + 88);
  }
  else
  {
    v36 = *(_QWORD *)(v538 + 88);
    v37 = v36;
  }
  v560 = v37 + 40LL * *(_QWORD *)(v538 + 96);
  if ( v36 != v560 )
  {
    v38 = 0;
    while ( 1 )
    {
      v45 = *v576;
      v46 = v586 & 0xFFFFFFFFFFFFFFF8LL;
      v47 = (_QWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v586 & 4) != 0 )
        break;
      if ( (unsigned __int8)sub_1560290(v47, v38, 6) )
      {
LABEL_58:
        v48 = sub_15E03C0(v36);
        v49 = sub_15603A0((_QWORD *)(v538 + 112), v38);
        v52 = sub_1AD38C0(v45, v535, v538, a2, v49, v48);
        v45 = *v576;
        if ( v52 != (_QWORD *)*v576 )
        {
          v53 = (unsigned int)v617;
          if ( (unsigned int)v617 >= HIDWORD(v617) )
          {
            sub_16CD150((__int64)&v616, &v618, 0, 16, v50, v51);
            v53 = (unsigned int)v617;
          }
          v54 = (unsigned __int8 *)&v616[v53];
          *((_QWORD *)v54 + 1) = v45;
          v45 = (__int64)v52;
          *(_QWORD *)v54 = v52;
          LODWORD(v617) = v617 + 1;
        }
        goto LABEL_40;
      }
      v39 = *(_QWORD *)(v46 - 72);
      if ( !*(_BYTE *)(v39 + 16) )
        goto LABEL_39;
LABEL_40:
      v625 = v36;
      v623.m128i_i64[1] = 2;
      v624 = 0;
      if ( v36 != -8 && v36 != 0 && v36 != -16 )
        sub_164C220((__int64)&v623.m128i_i64[1]);
      v626 = &v608;
      v623.m128i_i64[0] = (__int64)&unk_49E6B50;
      v40 = sub_12E4800((__int64)&v608, (__int64)&v623, &v602);
      v41 = (unsigned __int64 *)v602;
      if ( !v40 )
      {
        v41 = (unsigned __int64 *)sub_1ADC5B0((__int64)&v608, (__int64)&v623, (__int64)v602);
        sub_1AD3210(v41 + 1, &v623.m128i_i64[1]);
        v42 = v626;
        v41[5] = 6;
        v41[6] = 0;
        v41[4] = (unsigned __int64)v42;
        v41[7] = 0;
      }
      v623.m128i_i64[0] = (__int64)&unk_49EE2B0;
      sub_1455FA0((__int64)&v623.m128i_i64[1]);
      v43 = v41[7];
      v44 = v41 + 5;
      if ( v45 != v43 )
      {
        if ( v43 != 0 && v43 != -8 && v43 != -16 )
        {
          sub_1649B30(v44);
          v44 = v41 + 5;
        }
        v41[7] = v45;
        if ( v45 != -8 && v45 != 0 && v45 != -16 )
          sub_164C220((__int64)v44);
      }
      v576 += 3;
      v36 += 40;
      ++v38;
      if ( v36 == v560 )
        goto LABEL_62;
    }
    if ( (unsigned __int8)sub_1560290(v47, v38, 6) )
      goto LABEL_58;
    v39 = *(_QWORD *)(v46 - 24);
    if ( *(_BYTE *)(v39 + 16) )
      goto LABEL_40;
LABEL_39:
    v623.m128i_i64[0] = *(_QWORD *)(v39 + 112);
    if ( (unsigned __int8)sub_1560290(&v623, v38, 6) )
      goto LABEL_58;
    goto LABEL_40;
  }
LABEL_62:
  sub_1AD41A0(v586, a2);
  sub_1AB8DC0(
    v521,
    v538,
    (__int64)&v608,
    0,
    (__int64)&v605,
    (__int64)".i",
    a6,
    *(double *)a7.m128i_i64,
    *(double *)si128.m128i_i64,
    a9,
    v55,
    v56,
    a12,
    a13,
    (__int64)&v592);
  v57 = *(__int64 **)(a2 + 24);
  v514 = *(__int64 **)(v513 + 8);
  if ( v57 )
  {
    v58 = *(__int64 **)(a2 + 32);
    if ( v58 )
    {
      v59 = *(_QWORD *)(v538 + 80);
      v60 = v59 - 24;
      if ( !v59 )
        v60 = 0;
      sub_1AD3E20((__int64)v512, (__int64)&v608, v57, v58, v60);
      v57 = *(__int64 **)(a2 + 24);
    }
  }
  v61 = *(_QWORD *)(a2 + 16);
  v62.m128i_i64[0] = sub_15E44B0(v538);
  v623 = v62;
  sub_1AD3A90(v538, (__int64)&v608, (__int64)&v623, v535, v61, v57);
  v63 = *(_QWORD *)(a2 + 16);
  v64 = *(__int64 **)(a2 + 24);
  v65 = sub_15E44B0(v538);
  v67 = v66;
  if ( v66 )
  {
    if ( v63 )
    {
      sub_1441B50((__int64)&v623, v63, v535, v64);
      if ( v623.m128i_i8[8] )
      {
        v68 = v65 - v623.m128i_i64[0];
        if ( v65 < v623.m128i_i64[0] )
          v68 = 0;
        sub_15E4450(v538, v68, v67, 0);
      }
    }
  }
  v69 = (unsigned __int8 *)v616;
  v70 = (unsigned __int8 *)&v616[(unsigned int)v617];
  if ( v616 != (__m128i *)v70 )
  {
    v71 = (__int64)(v514 - 3);
    if ( !v514 )
      v71 = 0;
    do
    {
      v72 = (_QWORD *)*((_QWORD *)v69 + 1);
      v73 = *(_QWORD **)v69;
      v69 += 16;
      sub_1AD4770(v73, v72, *(_QWORD *)(v521 + 40), v71);
    }
    while ( v70 != v69 );
  }
  sub_1AD4B10((__int64)&v599, &v586, 0);
  if ( v601[8] )
  {
    v74 = &v624;
    v623.m128i_i64[1] = 0x200000000LL;
    v623.m128i_i64[0] = (__int64)&v624;
    v561 = v593;
    v530 = v594;
    if ( v593 != v594 )
    {
      v75 = 0;
      while ( 1 )
      {
        v76 = v561[2];
        v547 = v76;
        if ( v76 )
        {
          if ( *(_BYTE *)(v76 + 16) > 0x17u )
            break;
        }
LABEL_127:
        v561 += 3;
        if ( v561 == v530 )
        {
          v113 = &v74[7 * v75];
          if ( v113 != v74 )
          {
            do
            {
              v114 = *(v113 - 3);
              v113 -= 7;
              if ( v114 )
                j_j___libc_free_0(v114, v113[6] - v114);
              if ( (__int64 *)*v113 != v113 + 2 )
                j_j___libc_free_0(*v113, v113[2] + 1);
            }
            while ( v113 != v74 );
            v74 = (__int64 *)v623.m128i_i64[0];
          }
          if ( v74 != &v624 )
            _libc_free((unsigned __int64)v74);
          goto LABEL_137;
        }
      }
      v77 = &v74[7 * v75];
      while ( v77 != v74 )
      {
        while ( 1 )
        {
          v78 = *(v77 - 3);
          v77 -= 7;
          if ( v78 )
            j_j___libc_free_0(v78, v77[6] - v78);
          if ( (__int64 *)*v77 == v77 + 2 )
            break;
          j_j___libc_free_0(*v77, v77[2] + 1);
          if ( v77 == v74 )
            goto LABEL_88;
        }
      }
LABEL_88:
      v623.m128i_i32[2] = 0;
      v79 = sub_1AD4D60(v547) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(char *)(v79 + 23) >= 0 )
        goto LABEL_92;
      v80 = sub_1648A40(v79);
      v82 = v80 + v81;
      if ( *(char *)(v79 + 23) >= 0 )
      {
        v508 = v82 >> 4;
        v84 = (unsigned int)v508;
        if ( v623.m128i_i32[3] >= (unsigned int)v508 )
        {
LABEL_119:
          if ( *(_BYTE *)(v547 + 16) == 78 )
            v109 = sub_15F60C0(v547, (__int64 *)v623.m128i_i64[0], v623.m128i_u32[2], v547);
          else
            v109 = sub_15F6AA0(v547, (__int64 *)v623.m128i_i64[0], v623.m128i_u32[2], v547);
          sub_164D160(v547, v109, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v110, v111, a12, a13);
          v112 = v561[2];
          if ( v112 )
          {
            if ( v112 != -16 && v112 != -8 )
              sub_1649B30(v561);
            v561[2] = 0;
          }
          sub_15F20C0((_QWORD *)v547);
          v74 = (__int64 *)v623.m128i_i64[0];
          v75 = v623.m128i_u32[2];
          goto LABEL_127;
        }
      }
      else
      {
        v83 = (v82 - sub_1648A40(v79)) >> 4;
        v84 = (unsigned int)v83;
        if ( v623.m128i_i32[3] >= (unsigned int)v83 )
        {
LABEL_92:
          if ( *(char *)(v79 + 23) < 0 )
          {
            v85 = sub_1648A40(v79);
            v87 = v85 + v86;
            if ( *(char *)(v79 + 23) >= 0 )
              v88 = v87 >> 4;
            else
              LODWORD(v88) = (v87 - sub_1648A40(v79)) >> 4;
            if ( (_DWORD)v88 )
            {
              v89 = 0;
              v566 = 16LL * (unsigned int)v88;
              while ( 1 )
              {
LABEL_111:
                v101 = 0;
                if ( *(char *)(v79 + 23) < 0 )
                  v101 = sub_1648A40(v79);
                v102 = (__int64 *)(v89 + v101);
                v103 = *v102;
                v104 = *((unsigned int *)v102 + 2);
                v105 = *((unsigned int *)v102 + 3);
                v106 = 3LL * (*(_DWORD *)(v79 + 20) & 0xFFFFFFF);
                v591 = v103;
                v589 = (__m128i *)(v79 + 24 * v104 - 8 * v106);
                v590 = 0xAAAAAAAAAAAAAAABLL * ((24 * v105 - 24 * v104) >> 3);
                if ( !*(_DWORD *)(v103 + 8) )
                  break;
                v107 = v623.m128i_u32[2];
                if ( v623.m128i_i32[2] >= (unsigned __int32)v623.m128i_i32[3] )
                {
                  sub_1740340((__int64)&v623, 0);
                  v107 = v623.m128i_u32[2];
                }
                v108 = v623.m128i_i64[0] + 56LL * v107;
                if ( v108 )
                {
                  sub_1AD5550(v108, (__int64)&v589);
                  v107 = v623.m128i_u32[2];
                }
                v89 += 16;
                v623.m128i_i32[2] = v107 + 1;
                if ( v566 == v89 )
                  goto LABEL_119;
              }
              v90 = v600;
              src = 0;
              v597 = 0;
              v91 = v600 + v590;
              v598[0] = 0;
              if ( v600 + v590 > 0xFFFFFFFFFFFFFFFLL )
                sub_4262D8((__int64)"vector::reserve");
              v92 = 0;
              if ( !v91 )
                goto LABEL_102;
              v93 = 8 * v91;
              v94 = (char *)sub_22077B0(v93);
              v95 = src;
              v92 = v94;
              if ( v597 - (__int64)src > 0 )
              {
                v532 = src;
                v132 = (char *)memmove(v94, src, v597 - (_QWORD)src);
                v95 = v532;
                v92 = v132;
                v133 = v598[0] - (_QWORD)v532;
              }
              else
              {
                if ( !src )
                  goto LABEL_101;
                v133 = v598[0] - (_QWORD)src;
              }
              v533 = v92;
              j_j___libc_free_0(v95, v133);
              v92 = v533;
LABEL_101:
              src = v92;
              v90 = v600;
              v597 = (__int64)v92;
              v598[0] = &v92[v93];
LABEL_102:
              sub_1AD2EB0((__int64)&src, v92, v599, &v599[24 * v90]);
              sub_1AD2EB0((__int64)&src, (char *)v597, v589->m128i_i8, &v589->m128i_i8[24 * v590]);
              if ( v623.m128i_i32[2] >= (unsigned __int32)v623.m128i_i32[3] )
                sub_1740340((__int64)&v623, 0);
              v602 = v604;
              sub_1AD2E00((__int64 *)&v602, "deopt", (__int64)"");
              v96 = src;
              src = 0;
              v97 = v597;
              v98 = v598[0];
              v597 = 0;
              v598[0] = 0;
              v99 = (__m128 *)(v623.m128i_i64[0] + 56LL * v623.m128i_u32[2]);
              if ( v99 )
              {
                v99->m128_u64[0] = (unsigned __int64)&v99[1];
                if ( v602 == v604 )
                {
                  si128 = _mm_load_si128(v604);
                  v99[1] = (__m128)si128;
                }
                else
                {
                  v99->m128_u64[0] = (unsigned __int64)v602;
                  v99[1].m128_u64[0] = v604[0].m128i_i64[0];
                }
                v100 = v603;
                v99[2].m128_u64[0] = (unsigned __int64)v96;
                v99[2].m128_u64[1] = v97;
                v99->m128_u64[1] = v100;
                v99[3].m128_u64[0] = v98;
              }
              else
              {
                v131 = v98 - (_QWORD)v96;
                if ( v96 )
                  j_j___libc_free_0(v96, v131);
                if ( v602 != v604 )
                  j_j___libc_free_0(v602, v604[0].m128i_i64[0] + 1);
              }
              ++v623.m128i_i32[2];
              if ( src )
                j_j___libc_free_0(src, v598[0] - (_QWORD)src);
              v89 += 16;
              if ( v566 == v89 )
                goto LABEL_119;
              goto LABEL_111;
            }
          }
          goto LABEL_119;
        }
      }
      sub_1740340((__int64)&v623, v84);
      goto LABEL_92;
    }
  }
LABEL_137:
  if ( *(_QWORD *)a2 )
    sub_1AD4D90(v586, (__int64)&v608, (_QWORD *)a2);
  v115 = sub_1626D20(v538);
  sub_1AD34F0(v521, (__int64)v514, v535, v115 != 0);
  sub_1ADAFF0(
    v586,
    (unsigned __int64)&v608,
    a6,
    *(double *)a7.m128i_i64,
    *(double *)si128.m128i_i64,
    a9,
    v120,
    v121,
    a12,
    a13,
    v116,
    v117,
    v118,
    v119);
  sub_1AD9010(v586, (__int64)&v608, v516, a3);
  v122 = v586;
  v123 = &v608;
  sub_1AD3CB0(v586, (__int64)&v608);
  if ( *(_QWORD *)(a2 + 8) )
  {
    v125 = v514 - 3;
    if ( !v514 )
      v125 = 0;
    v126 = v125 + 3;
    if ( v126 != (_QWORD *)v527 )
    {
      while ( 1 )
      {
        v127 = (_QWORD *)v126[3];
        for ( j = v126 + 2; v127 != j; v127 = (_QWORD *)v127[1] )
        {
          while ( 1 )
          {
            if ( !v127 )
LABEL_808:
              BUG();
            if ( *((_BYTE *)v127 - 8) == 78 )
            {
              v129 = *(v127 - 6);
              if ( !*(_BYTE *)(v129 + 16) && (*(_BYTE *)(v129 + 33) & 0x20) != 0 && *(_DWORD *)(v129 + 36) == 4 )
                break;
            }
            v127 = (_QWORD *)v127[1];
            if ( v127 == j )
              goto LABEL_153;
          }
          v130 = *(_QWORD *)(a2 + 8);
          if ( !*(_QWORD *)(v130 + 16) )
            sub_4263D6(v122, v123, v124);
          v123 = v127 - 3;
          v122 = (*(__int64 (__fastcall **)(_QWORD, __int64))(v130 + 24))(*(_QWORD *)(a2 + 8), v521);
          sub_14CE830(v122, (__int64)(v127 - 3));
        }
LABEL_153:
        v126 = (_QWORD *)v126[1];
        if ( (_QWORD *)v527 == v126 )
          break;
        if ( !v126 )
          goto LABEL_807;
      }
    }
  }
  if ( v616 != &v618 )
    _libc_free((unsigned __int64)v616);
  if ( v614 )
  {
    if ( v613 )
    {
      v462 = v612;
      v463 = &v612[2 * v613];
      do
      {
        if ( *v462 != -4 && *v462 != -8 )
          sub_17CD270(v462 + 1);
        v462 += 2;
      }
      while ( v463 != v462 );
    }
    j___libc_free_0(v612);
  }
  if ( v611 )
  {
    v134 = (_QWORD *)v609;
    v617 = 2;
    v618.m128i_i64[0] = 0;
    v135 = (_QWORD *)(v609 + ((unsigned __int64)v611 << 6));
    v618.m128i_i64[1] = -8;
    v616 = (__m128i *)&unk_49E6B50;
    v623.m128i_i64[0] = (__int64)&unk_49E6B50;
    v136 = -8;
    v619 = 0;
    v623.m128i_i64[1] = 2;
    v624 = 0;
    v625 = -16;
    v626 = 0;
    while ( 1 )
    {
      v137 = v134[3];
      if ( v137 != v136 && v137 != v625 )
        sub_1455FA0((__int64)(v134 + 5));
      *v134 = &unk_49EE2B0;
      v138 = (__int64)(v134 + 1);
      v134 += 8;
      sub_1455FA0(v138);
      if ( v135 == v134 )
        break;
      v136 = v618.m128i_i64[1];
    }
    v623.m128i_i64[0] = (__int64)&unk_49EE2B0;
    sub_1455FA0((__int64)&v623.m128i_i64[1]);
    v616 = (__m128i *)&unk_49EE2B0;
    sub_1455FA0((__int64)&v617);
  }
  j___libc_free_0(v609);
  v141 = *(_QWORD *)(v521 + 80);
  if ( !v141 || (v142 = *(unsigned __int64 **)(v141 + 24), !v514) )
LABEL_807:
    BUG();
  v143 = (__int64 *)v514[3];
  v511 = (__int64)(v514 - 3);
  v510 = (unsigned __int64 *)(v514 + 2);
  if ( v514 + 2 != v143 )
  {
    v567 = a2 + 40;
    v562 = (const void *)(a2 + 56);
    do
    {
      v144 = (unsigned __int64 *)v143[1];
      if ( *((_BYTE *)v143 - 8) == 53 )
      {
        if ( !*(v143 - 2) )
        {
          v390 = v143 - 3;
          v143 = (__int64 *)v143[1];
          sub_15F20C0(v390);
          continue;
        }
        if ( *(_BYTE *)(*(v143 - 6) + 16) <= 0x10u && (*((_BYTE *)v143 - 6) & 0x20) == 0 )
        {
          v145 = *(unsigned int *)(a2 + 48);
          if ( (unsigned int)v145 >= *(_DWORD *)(a2 + 52) )
          {
            sub_16CD150(v567, v562, 0, 8, (int)v139, v140);
            v145 = *(unsigned int *)(a2 + 48);
          }
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8 * v145) = v143 - 3;
          ++*(_DWORD *)(a2 + 48);
          while ( 1 )
          {
            if ( !v144 )
              goto LABEL_808;
            v139 = v144 - 3;
            if ( *((_BYTE *)v144 - 8) != 53
              || *(_BYTE *)(*(v144 - 6) + 16) > 0x10u
              || (*((_BYTE *)v144 - 6) & 0x20) != 0 )
            {
              break;
            }
            v180 = *(unsigned int *)(a2 + 48);
            if ( (unsigned int)v180 >= *(_DWORD *)(a2 + 52) )
            {
              sub_16CD150(v567, v562, 0, 8, (int)v139, v140);
              v180 = *(unsigned int *)(a2 + 48);
              v139 = v144 - 3;
            }
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8 * v180) = v139;
            ++*(_DWORD *)(a2 + 48);
            v144 = (unsigned __int64 *)v144[1];
          }
          v146 = *(_QWORD *)(v521 + 80);
          if ( v146 )
            v146 -= 24;
          if ( v143 != (__int64 *)v144 && v142 != v144 )
          {
            if ( (unsigned __int64 *)(v146 + 40) != v510 )
              sub_157EA80(v146 + 40, (__int64)v510, (__int64)v143, (__int64)v144);
            if ( v144 != v142 && v143 != (__int64 *)v144 )
            {
              v147 = *v144 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v143 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v144;
              *v144 = *v144 & 7 | *v143 & 0xFFFFFFFFFFFFFFF8LL;
              v148 = *v142;
              *(_QWORD *)(v147 + 8) = v142;
              v148 &= 0xFFFFFFFFFFFFFFF8LL;
              *v143 = v148 | *v143 & 7;
              *(_QWORD *)(v148 + 8) = v143;
              *v142 = v147 | *v142 & 7;
            }
          }
        }
      }
      v143 = (__int64 *)v144;
    }
    while ( v510 != v144 );
  }
  v149 = *(__int64 **)(v521 + 40);
  sub_15A5590((__int64)&v623, v149, 1, 0);
  v150 = *(__int64 ***)(a2 + 40);
  v151 = &v150[*(unsigned int *)(a2 + 48)];
  while ( v151 != v150 )
  {
    v152 = *v150++;
    v149 = v152;
    sub_1AEA710(v152, v152, &v623, 0, 0, 0);
  }
  sub_129E320((__int64)&v623, (__int64)v149);
  src = v598;
  v597 = 0x400000000LL;
  v600 = 0x400000000LL;
  v599 = v601;
  for ( k = *(_DWORD *)(*(_QWORD *)(v538 + 24) + 12LL) - 1; ; ++k )
  {
    v166 = v586 & 0xFFFFFFFFFFFFFFF8LL;
    v167 = *(_DWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
    if ( (v586 & 4) != 0 )
    {
      if ( *(char *)(v166 + 23) < 0 )
      {
        v157 = sub_1648A40(v586 & 0xFFFFFFFFFFFFFFF8LL);
        v158 = v157 + v153;
        if ( *(char *)(v166 + 23) >= 0 )
        {
          if ( (unsigned int)(v158 >> 4) )
LABEL_810:
            BUG();
        }
        else if ( (unsigned int)((v158 - sub_1648A40(v166)) >> 4) )
        {
          if ( *(char *)(v166 + 23) >= 0 )
            goto LABEL_810;
          v159 = *(_DWORD *)(sub_1648A40(v166) + 8);
          if ( *(char *)(v166 + 23) >= 0 )
            goto LABEL_809;
          v160 = sub_1648A40(v166);
          v161 = *(_DWORD *)(v160 + v153 - 4) - v159;
          goto LABEL_208;
        }
      }
      v161 = 0;
LABEL_208:
      if ( k >= v167 - 1 - v161 )
        break;
      goto LABEL_209;
    }
    v168 = sub_165C2D0(v586 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v168 )
    {
      if ( *(char *)(v166 + 23) >= 0 )
        goto LABEL_810;
      v169 = *(_DWORD *)(sub_1648A40(v166) + 8);
      if ( *(char *)(v166 + 23) >= 0 )
LABEL_809:
        BUG();
      v170 = sub_1648A40(v166);
      v168 = *(_DWORD *)(v170 + v153 - 4) - v169;
    }
    if ( k >= v167 - 3 - v168 )
      break;
LABEL_209:
    v162 = *(_QWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL)
                     + 24 * (k - (unsigned __int64)(*(_DWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
    v163 = (unsigned int)v597;
    if ( (unsigned int)v597 >= HIDWORD(v597) )
    {
      sub_16CD150((__int64)&src, v598, 0, 8, v154, v155);
      v163 = (unsigned int)v597;
    }
    *((_QWORD *)src + v163) = v162;
    LODWORD(v597) = v597 + 1;
    v623.m128i_i64[0] = *(_QWORD *)((v586 & 0xFFFFFFFFFFFFFFF8LL) + 56);
    v164 = sub_1560230(&v623, k);
    v165 = (unsigned int)v600;
    if ( (unsigned int)v600 >= HIDWORD(v600) )
    {
      sub_16CD150((__int64)&v599, v601, 0, 8, v154, v155);
      v165 = (unsigned int)v600;
    }
    v153 = (__int64)v599;
    *(_QWORD *)&v599[8 * v165] = v164;
    LODWORD(v600) = v600 + 1;
  }
  v531 = v592;
  if ( (_BYTE)v592 )
  {
    v548 = 0;
    if ( *(_BYTE *)(v535 + 16) == 78 )
    {
      v171 = *(_WORD *)(v535 + 18) & 3;
      v153 = *(_WORD *)(v535 + 18) & 3;
      if ( (*(_WORD *)(v535 + 18) & 3) == 3 )
        v171 = 0;
      v548 = v171;
    }
    v534 = 0;
    v531 = 0;
    v563 = v514;
    if ( v514 != (__int64 *)v527 )
    {
      while ( 1 )
      {
        v172 = (__int64 *)v563[3];
        v577 = v563 + 2;
        if ( v172 != v563 + 2 )
          break;
LABEL_242:
        v563 = (__int64 *)v563[1];
        if ( (__int64 *)v527 == v563 )
          goto LABEL_296;
        if ( !v563 )
          goto LABEL_807;
      }
      while ( 1 )
      {
        v173 = v172;
        v172 = (__int64 *)v172[1];
        if ( *((_BYTE *)v173 - 8) != 78 )
        {
LABEL_228:
          if ( v172 == v577 )
            goto LABEL_242;
          continue;
        }
        v174 = v173 - 3;
        v175 = v173 - 3;
        if ( (_DWORD)v597 )
        {
          if ( a5 )
          {
            v176 = *(v173 - 6);
            if ( !*(_BYTE *)(v176 + 16) && a5 == v176 )
              break;
          }
          if ( (*((_WORD *)v173 - 3) & 3) == 2 )
            break;
        }
LABEL_235:
        v177 = *(v175 - 3);
        if ( !*(_BYTE *)(v177 + 16) )
          v531 |= *(_DWORD *)(v177 + 36) == 75;
        v153 = *((_WORD *)v175 + 9) & 3;
        if ( (*((_WORD *)v175 + 9) & 3) != 3 && (*((_WORD *)v175 + 9) & 3) >= v548 )
          v153 = (unsigned int)v548;
        v178 = v153 | *((_WORD *)v175 + 9) & 0xFFFC;
        *((_WORD *)v175 + 9) = v178;
        v534 |= (v178 & 3) == 2;
        if ( !v553 )
          goto LABEL_228;
        v623.m128i_i64[0] = v175[7];
        v179 = (__int64 *)sub_16498A0((__int64)v175);
        v623.m128i_i64[0] = sub_1563AB0(v623.m128i_i64, v179, -1, 30);
        v175[7] = v623.m128i_i64[0];
        if ( v172 == v577 )
          goto LABEL_242;
      }
      v181 = (__int64 *)v173[4];
      v623.m128i_i64[0] = (__int64)&v624;
      v602 = (__m128i *)v181;
      v623.m128i_i64[1] = 0x800000000LL;
      if ( v181 || (_DWORD)v600 )
      {
        if ( *(_DWORD *)(v173[5] + 12) == 1 )
        {
          v218 = 8;
          v184 = 0;
        }
        else
        {
          v214 = 0;
          do
          {
            v215 = sub_1560230(&v602, v214);
            v217 = v623.m128i_u32[2];
            if ( v623.m128i_i32[2] >= (unsigned __int32)v623.m128i_i32[3] )
            {
              v544 = v215;
              sub_16CD150((__int64)&v623, &v624, 0, 8, v215, v216);
              v217 = v623.m128i_u32[2];
              v215 = v544;
            }
            ++v214;
            *(_QWORD *)(v623.m128i_i64[0] + 8 * v217) = v215;
            v184 = (unsigned int)++v623.m128i_i32[2];
          }
          while ( v214 < *(_DWORD *)(v173[5] + 12) - 1 );
          v218 = v623.m128i_u32[3] - v184;
        }
        v183 = v600;
        v219 = v599;
        v220 = 8LL * (unsigned int)v600;
        if ( (unsigned int)v600 <= v218 )
        {
          v182 = (__int64 *)v623.m128i_i64[0];
          v221 = (void *)(v623.m128i_i64[0] + 8 * v184);
        }
        else
        {
          v519 = v599;
          nd = 8LL * (unsigned int)v600;
          v542 = v600;
          sub_16CD150((__int64)&v623, &v624, (unsigned int)v600 + v184, 8, v600, v220);
          v182 = (__int64 *)v623.m128i_i64[0];
          LODWORD(v184) = v623.m128i_i32[2];
          v219 = v519;
          v220 = nd;
          v183 = v542;
          v221 = (void *)(v623.m128i_i64[0] + 8LL * v623.m128i_u32[2]);
        }
        if ( v220 )
        {
          v543 = v183;
          memcpy(v221, v219, v220);
          v182 = (__int64 *)v623.m128i_i64[0];
          LODWORD(v184) = v623.m128i_i32[2];
          v183 = v543;
        }
      }
      else
      {
        v182 = &v624;
        v183 = 0;
        LODWORD(v184) = 0;
      }
      v518 = v182;
      v623.m128i_i32[2] = v183 + v184;
      na = (unsigned int)(v183 + v184);
      v541 = sub_1560240(&v602);
      v185 = sub_1560250(&v602);
      v186 = (__int64 *)sub_16498A0((__int64)(v173 - 3));
      v602 = (__m128i *)sub_155FDB0(v186, v185, v541, v518, na);
      if ( *((char *)v173 - 1) < 0 )
      {
        v188 = sub_1648A40((__int64)(v173 - 3));
        v190 = v188 + v189;
        if ( *((char *)v173 - 1) >= 0 )
        {
          if ( (unsigned int)(v190 >> 4) )
            goto LABEL_810;
        }
        else if ( (unsigned int)((v190 - sub_1648A40((__int64)(v173 - 3))) >> 4) )
        {
          if ( *((char *)v173 - 1) >= 0 )
            goto LABEL_810;
          v191 = *(_DWORD *)(sub_1648A40((__int64)(v173 - 3)) + 8);
          if ( *((char *)v173 - 1) >= 0 )
            goto LABEL_809;
          v192 = sub_1648A40((__int64)(v173 - 3));
          v194 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v192 + v193 - 4) - v191);
          goto LABEL_260;
        }
      }
      v194 = -24;
LABEL_260:
      v195 = (__int64 *)((char *)v174 + v194);
      v196 = *((_DWORD *)v173 - 1) & 0xFFFFFFF;
      v616 = &v618;
      v617 = 0x600000000LL;
      v197 = &v174[-3 * v196];
      v198 = v194 + 24 * v196;
      v199 = 0xAAAAAAAAAAAAAAABLL * (v198 >> 3);
      v200 = &v618;
      v201 = 0;
      if ( (unsigned __int64)v198 > 0x90 )
      {
        v520 = v195;
        nb = -1431655765 * (v198 >> 3);
        sub_16CD150((__int64)&v616, &v618, v199, 8, (int)v195, v187);
        v201 = v617;
        v195 = v520;
        LODWORD(v199) = nb;
        v200 = (__m128i *)((char *)v616 + 8 * (unsigned int)v617);
      }
      if ( v195 != v197 )
      {
        do
        {
          if ( v200 )
            v200->m128i_i64[0] = *v197;
          v197 += 3;
          v200 = (__m128i *)((char *)v200 + 8);
        }
        while ( v195 != v197 );
        v201 = v617;
      }
      v202 = v597;
      LODWORD(v617) = v199 + v201;
      v203 = (unsigned int)(v199 + v201);
      v204 = src;
      v205 = 8LL * (unsigned int)v597;
      if ( (unsigned int)v597 > HIDWORD(v617) - (unsigned __int64)(unsigned int)v203 )
      {
        nc = src;
        v545 = 8LL * (unsigned int)v597;
        sub_16CD150((__int64)&v616, &v618, v203 + (unsigned int)v597, 8, v205, (int)src);
        v203 = (unsigned int)v617;
        v204 = nc;
        v205 = v545;
      }
      v206 = (__int64 *)v616;
      if ( v205 )
      {
        memcpy((char *)v616 + 8 * v203, v204, v205);
        v206 = (__int64 *)v616;
        LODWORD(v203) = v617;
      }
      LOWORD(v610) = 257;
      LODWORD(v617) = v202 + v203;
      v175 = sub_1AD45C0(
               *(_QWORD *)(*(_QWORD *)*(v173 - 6) + 24LL),
               *(v173 - 6),
               v206,
               (unsigned int)(v202 + v203),
               0,
               0,
               (__int64)&v608,
               (__int64)(v173 - 3));
      v608 = v173[3];
      if ( v608 )
        sub_1AD3290(&v608);
      v207 = (__int64)(v175 + 6);
      if ( v175 + 6 != &v608 )
      {
        v208 = v175[6];
        if ( v208 )
        {
          sub_161E7C0((__int64)(v175 + 6), v208);
          v207 = (__int64)(v175 + 6);
        }
        v209 = (unsigned __int8 *)v608;
        v175[6] = v608;
        if ( v209 )
        {
          sub_1623210((__int64)&v608, v209, v207);
          v608 = 0;
        }
      }
      sub_17CD270(&v608);
      v210 = *((_WORD *)v175 + 9);
      v211 = *((_WORD *)v175 + 9) & 0x8000;
      v175[7] = (__int64)v602;
      *((_WORD *)v175 + 9) = v211 | v210 & 3 | (4 * ((*((_WORD *)v173 - 3) >> 2) & 0xDFFF));
      sub_164D160(
        (__int64)(v173 - 3),
        (__int64)v175,
        a6,
        *(double *)a7.m128i_i64,
        *(double *)si128.m128i_i64,
        a9,
        v212,
        v213,
        a12,
        a13);
      sub_15F20C0(v173 - 3);
      if ( v616 != &v618 )
        _libc_free((unsigned __int64)v616);
      if ( (__int64 *)v623.m128i_i64[0] != &v624 )
        _libc_free(v623.m128i_u64[0]);
      goto LABEL_235;
    }
  }
  else
  {
    v534 = 0;
  }
LABEL_296:
  if ( a4 )
  {
    v154 = *(_DWORD *)(a2 + 48);
    if ( v154 )
    {
      v222 = v514[3];
      if ( v222 )
        v222 -= 24;
      v223 = 0;
      sub_17CE510((__int64)&v616, v222, 0, 0, 0);
      v224 = *(unsigned int *)(a2 + 48);
      v564 = 8 * v224;
      if ( !(_DWORD)v224 )
      {
LABEL_323:
        sub_17CD270((__int64 *)&v616);
        goto LABEL_324;
      }
      while ( 2 )
      {
        v228 = *(_QWORD *)(a2 + 40);
        v229 = *(_QWORD *)(v228 + v223);
        if ( (*(_BYTE *)(v229 + 18) & 0x40) != 0 || (unsigned __int8)sub_1AD3310(*(__int64 **)(v228 + v223)) )
          goto LABEL_305;
        v230 = *(_QWORD *)(v229 - 24);
        if ( *(_BYTE *)(v230 + 16) != 13 )
          goto LABEL_309;
        v225 = sub_1632FA0(*(_QWORD *)(v521 + 40));
        v226 = sub_12BE0A0(v225, *(_QWORD *)(v229 + 56));
        if ( *(_DWORD *)(v230 + 32) <= 0x40u )
        {
          v227 = *(_QWORD *)(v230 + 24);
          goto LABEL_304;
        }
        v568 = *(_DWORD *)(v230 + 32);
        if ( v568 - (unsigned int)sub_16A57B0(v230 + 24) > 0x40 )
          goto LABEL_309;
        v227 = **(_QWORD **)(v230 + 24);
LABEL_304:
        if ( !v227 )
        {
LABEL_305:
          v223 += 8;
          if ( v564 == v223 )
            goto LABEL_323;
          continue;
        }
        break;
      }
      if ( v227 == -1 || (v492 = v226, v493 = v227 * v226, !is_mul_ok(v227, v492)) )
      {
LABEL_309:
        v554 = 0;
        v231 = sub_15E7DE0((__int64 *)&v616, (_QWORD *)v229, 0);
      }
      else
      {
        v494 = (_QWORD *)sub_16498A0(v229);
        v495 = sub_1643360(v494);
        v554 = sub_159C470(v495, v493, 0);
        v231 = sub_15E7DE0((__int64 *)&v616, (_QWORD *)v229, v554);
      }
      v232 = (__int64)v231;
      if ( byte_4FB65C0 )
      {
        v485 = *(_QWORD *)(v229 + 56);
        v486 = *(_BYTE *)(v485 + 8);
        if ( v486 == 11 )
        {
          v487 = sub_15A0680(v485, 0, 0);
        }
        else if ( (unsigned __int8)(v486 - 1) > 5u )
        {
          if ( v486 == 13 )
            v487 = sub_1598F00((__int64 **)v485);
          else
            v487 = sub_1599EF0((__int64 **)v485);
        }
        else
        {
          a6 = 0;
          v487 = sub_15A10B0(v485, 0.0);
        }
        v488 = sub_1648A60(64, 2u);
        v489 = (__int64)v488;
        if ( v488 )
        {
          v575 = v488;
          sub_15F9650((__int64)v488, v487, v229, 0, 0);
          v489 = (__int64)v575;
        }
        sub_15F2180(v489, v232);
      }
      v233 = v605;
      v569 = &v605[(unsigned int)v606];
      if ( v605 != v569 )
      {
        do
        {
          v234 = *v233;
          if ( (!v534 || !sub_157EBE0(*(_QWORD *)(v234 + 40))) && (!v531 || !sub_157ECB0(*(_QWORD *)(v234 + 40))) )
          {
            v235 = sub_16498A0(v234);
            v623.m128i_i64[0] = 0;
            v625 = v235;
            v626 = 0;
            v627 = 0;
            v628 = 0;
            v629 = 0;
            v623.m128i_i64[1] = *(_QWORD *)(v234 + 40);
            v624 = v234 + 24;
            v236 = *(__int64 **)(v234 + 48);
            v608 = (__int64)v236;
            if ( v236 )
            {
              sub_1623A60((__int64)&v608, (__int64)v236, 2);
              if ( v623.m128i_i64[0] )
                sub_161E7C0((__int64)&v623, v623.m128i_i64[0]);
              v623.m128i_i64[0] = v608;
              if ( v608 )
                sub_1623210((__int64)&v608, (unsigned __int8 *)v608, (__int64)&v623);
            }
            sub_15E7E90(v623.m128i_i64, (_QWORD *)v229, v554);
            sub_17CD270(v623.m128i_i64);
          }
          ++v233;
        }
        while ( v569 != v233 );
      }
      goto LABEL_305;
    }
  }
LABEL_324:
  if ( HIBYTE(v592) )
  {
    v452 = *(__int64 **)(v521 + 40);
    v453 = sub_15E26F0(v452, 202, 0, 0);
    v454 = sub_15E26F0(v452, 201, 0, 0);
    v455 = v514[3];
    v456 = sub_157E9C0(v511);
    v623.m128i_i64[1] = (__int64)(v514 - 3);
    v623.m128i_i64[0] = 0;
    v625 = v456;
    v626 = 0;
    v627 = 0;
    v628 = 0;
    v629 = 0;
    v624 = v455;
    if ( v510 != (unsigned __int64 *)v455 )
    {
      if ( !v455 )
        goto LABEL_807;
      v616 = *(__m128i **)(v455 + 24);
      if ( v616 )
        sub_1AD3290((__int64 *)&v616);
      sub_1AD34B0(v623.m128i_i64, (unsigned __int8 **)&v616);
      sub_17CD270((__int64 *)&v616);
    }
    v616 = (__m128i *)"savedstack";
    v618.m128i_i16[0] = 259;
    v457 = (__int64 *)sub_1285290(v623.m128i_i64, *(_QWORD *)(v453 + 24), v453, 0, 0, (__int64)&v616, 0);
    sub_17CD270(v623.m128i_i64);
    v458 = v605;
    if ( v605 != &v605[(unsigned int)v606] )
    {
      v573 = &v605[(unsigned int)v606];
      do
      {
        v459 = *v458;
        if ( (!v534 || !sub_157EBE0(*(_QWORD *)(v459 + 40))) && (!v531 || !sub_157ECB0(*(_QWORD *)(v459 + 40))) )
        {
          v460 = sub_16498A0(v459);
          v623.m128i_i64[0] = 0;
          v625 = v460;
          v626 = 0;
          v627 = 0;
          v628 = 0;
          v629 = 0;
          v623.m128i_i64[1] = *(_QWORD *)(v459 + 40);
          v624 = v459 + 24;
          v461 = *(unsigned __int8 **)(v459 + 48);
          v616 = (__m128i *)v461;
          if ( v461 )
          {
            sub_1623A60((__int64)&v616, (__int64)v461, 2);
            if ( v623.m128i_i64[0] )
              sub_161E7C0((__int64)&v623, v623.m128i_i64[0]);
            v623.m128i_i64[0] = (__int64)v616;
            if ( v616 )
              sub_1623210((__int64)&v616, (unsigned __int8 *)v616, (__int64)&v623);
          }
          v618.m128i_i16[0] = 257;
          v608 = (__int64)v457;
          sub_1285290(v623.m128i_i64, *(_QWORD *)(v454 + 24), v454, (int)&v608, 1, (__int64)&v616, 0);
          sub_17CD270(v623.m128i_i64);
          v153 = v509;
        }
        ++v458;
      }
      while ( v573 != v458 );
    }
  }
  if ( *(_BYTE *)(v535 + 16) == 29 )
  {
    if ( *(_BYTE *)(sub_157ED20(*(_QWORD *)(v535 - 24)) + 16) == 88 )
      sub_1AD70C0(v535, v511, &v592, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v237, v238, a12, a13);
    else
      sub_1AD7F30(v535, v511, &v592, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v237, v238, a12, a13);
  }
  if ( v515 )
  {
    for ( m = v514; (__int64 *)v527 != m; m = (__int64 *)m[1] )
    {
      v623.m128i_i64[0] = (__int64)&v624;
      v623.m128i_i64[1] = 0x100000000LL;
      if ( !m )
        goto LABEL_807;
      n = (size_t)(m - 3);
      v578 = (__int64 *)m[3];
      v546 = m + 2;
      if ( m + 2 == v578 )
        goto LABEL_362;
      do
      {
        while ( 1 )
        {
          v239 = (__int64 *)v578[1];
          v608 = 0;
          v240 = (__int64)(v578 - 3);
          v241 = *((_BYTE *)v578 - 8);
          v578 = v239;
          if ( v241 > 0x17u )
          {
            if ( v241 == 78 )
            {
              v242 = v240 | 4;
              v243 = v240 & 0xFFFFFFFFFFFFFFF8LL;
              goto LABEL_338;
            }
            if ( v241 == 29 )
              break;
          }
LABEL_334:
          if ( v546 == v578 )
            goto LABEL_362;
        }
        v242 = v240 & 0xFFFFFFFFFFFFFFFBLL;
        v243 = v240 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_338:
        v608 = v242;
        if ( !v243 )
          goto LABEL_334;
        v244 = (__int64 *)(v243 - 24);
        v245 = (__int64 *)(v243 - 72);
        if ( (v242 & 4) != 0 )
          v245 = v244;
        v246 = sub_1649C60(*v245);
        if ( *(_BYTE *)(v246 + 16) || (*(_BYTE *)(v246 + 33) & 0x20) == 0 )
          goto LABEL_343;
        v315 = v608 & 0xFFFFFFFFFFFFFFF8LL;
        v316 = (_QWORD *)((v608 & 0xFFFFFFFFFFFFFFF8LL) + 56);
        if ( (v608 & 4) != 0 )
        {
          if ( (unsigned __int8)sub_1560260(v316, -1, 30) )
            goto LABEL_334;
          v317 = *(_QWORD *)(v315 - 24);
          if ( !*(_BYTE *)(v317 + 16) )
          {
LABEL_444:
            v616 = *(__m128i **)(v317 + 112);
            if ( (unsigned __int8)sub_1560260(&v616, -1, 30) )
              goto LABEL_334;
          }
        }
        else
        {
          if ( (unsigned __int8)sub_1560260(v316, -1, 30) )
            goto LABEL_334;
          v317 = *(_QWORD *)(v315 - 72);
          if ( !*(_BYTE *)(v317 + 16) )
            goto LABEL_444;
        }
LABEL_343:
        sub_1AD4B10((__int64)&v616, &v608, 1);
        if ( v618.m128i_i8[8] )
          goto LABEL_334;
        v247 = v608 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v608 & 4) != 0 )
        {
          sub_1752100(v608 & 0xFFFFFFFFFFFFFFF8LL, (__int64)&v623);
          v248 = v623.m128i_u32[2];
          goto LABEL_346;
        }
        v318 = sub_165C2D0(v608 & 0xFFFFFFFFFFFFFFF8LL);
        if ( !v318 )
        {
          v248 = v623.m128i_u32[2];
LABEL_346:
          if ( v248 >= v623.m128i_i32[3] )
            goto LABEL_470;
          goto LABEL_347;
        }
        v517 = v240;
        v319 = 0;
        v556 = 16LL * v318;
        do
        {
          v323 = 0;
          if ( *(char *)(v247 + 23) < 0 )
            v323 = sub_1648A40(v247);
          v324 = (__int64 **)(v319 + v323);
          v325 = *v324;
          v326 = *((unsigned int *)v324 + 2);
          v327 = *(_DWORD *)(v247 + 20) & 0xFFFFFFF;
          v570 = *((_DWORD *)v324 + 3);
          v321 = v623.m128i_u32[2];
          if ( v623.m128i_i32[2] >= (unsigned __int32)v623.m128i_i32[3] )
          {
            v551 = v325;
            sub_1740340((__int64)&v623, 0);
            v321 = v623.m128i_u32[2];
            v325 = v551;
          }
          v328 = v623.m128i_i64[0] + 56LL * v321;
          if ( v328 )
          {
            *(_QWORD *)(v328 + 8) = 0;
            *(_QWORD *)v328 = v328 + 16;
            *(_BYTE *)(v328 + 16) = 0;
            *(_QWORD *)(v328 + 32) = 0;
            *(_QWORD *)(v328 + 40) = 0;
            *(_QWORD *)(v328 + 48) = 0;
            v329 = *v325;
            v616 = &v618;
            sub_1AD2E00((__int64 *)&v616, (_BYTE *)v325 + 16, (__int64)v325 + v329 + 16);
            v330 = *(__m128i **)v328;
            if ( v616 == &v618 )
            {
              v331 = v617;
              if ( v617 )
              {
                if ( v617 == 1 )
                  v330->m128i_i8[0] = v618.m128i_i8[0];
                else
                  memcpy(v330, &v618, v617);
                v331 = v617;
                v330 = *(__m128i **)v328;
              }
              *(_QWORD *)(v328 + 8) = v331;
              v330->m128i_i8[v331] = 0;
              v330 = v616;
              goto LABEL_451;
            }
            if ( (__m128i *)(v328 + 16) == v330 )
            {
              *(_QWORD *)v328 = v616;
              *(_QWORD *)(v328 + 8) = v617;
              *(_QWORD *)(v328 + 16) = v618.m128i_i64[0];
            }
            else
            {
              *(_QWORD *)v328 = v616;
              v320 = *(_QWORD *)(v328 + 16);
              *(_QWORD *)(v328 + 8) = v617;
              *(_QWORD *)(v328 + 16) = v618.m128i_i64[0];
              if ( v330 )
              {
                v616 = v330;
                v618.m128i_i64[0] = v320;
LABEL_451:
                v617 = 0;
                v330->m128i_i8[0] = 0;
                if ( v616 != &v618 )
                  j_j___libc_free_0(v616, v618.m128i_i64[0] + 1);
                sub_1AD2EB0(
                  v328 + 32,
                  *(char **)(v328 + 40),
                  (char *)(v247 + 24 * v326 - 24 * v327),
                  (char *)(v247 + 24 * v326 - 24 * v327 + 24LL * v570 - 24 * v326));
                v321 = v623.m128i_u32[2];
                goto LABEL_454;
              }
            }
            v330 = &v618;
            v616 = &v618;
            goto LABEL_451;
          }
LABEL_454:
          v322 = v321 + 1;
          v319 += 16;
          v623.m128i_i32[2] = v322;
        }
        while ( v556 != v319 );
        v240 = v517;
        if ( v322 >= v623.m128i_i32[3] )
LABEL_470:
          sub_1740340((__int64)&v623, 0);
LABEL_347:
        v616 = &v618;
        sub_1AD2E00((__int64 *)&v616, "funclet", (__int64)"");
        v249 = (__m128 *)(v623.m128i_i64[0] + 56LL * v623.m128i_u32[2]);
        if ( v249 )
        {
          v249->m128_u64[0] = (unsigned __int64)&v249[1];
          if ( v616 == &v618 )
          {
            a7 = _mm_load_si128(&v618);
            v249[1] = (__m128)a7;
          }
          else
          {
            v249->m128_u64[0] = (unsigned __int64)v616;
            v249[1].m128_u64[0] = v618.m128i_i64[0];
          }
          v249->m128_u64[1] = v617;
          v617 = 0;
          v616 = &v618;
          v618.m128i_i8[0] = 0;
          v249[2].m128_u64[0] = 0;
          v249[2].m128_u64[1] = 0;
          v249[3].m128_u64[0] = 0;
          v250 = (__int64 *)sub_22077B0(8);
          v249[2].m128_u64[0] = (unsigned __int64)v250;
          v249[3].m128_u64[0] = (unsigned __int64)(v250 + 1);
          *v250 = v515;
          v249[2].m128_u64[1] = (unsigned __int64)(v250 + 1);
        }
        if ( v616 != &v618 )
          j_j___libc_free_0(v616, v618.m128i_i64[0] + 1);
        v251 = (unsigned int)++v623.m128i_i32[2];
        if ( (v608 & 4) != 0 )
          v252 = sub_15F60C0(v240, (__int64 *)v623.m128i_i64[0], v251, v240);
        else
          v252 = sub_15F6AA0(v240, (__int64 *)v623.m128i_i64[0], v251, v240);
        sub_164B7C0(v252, v240);
        sub_164D160(v240, v252, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v253, v254, a12, a13);
        sub_15F20C0((_QWORD *)v240);
        v255 = v623.m128i_i64[0];
        v256 = (_QWORD *)(v623.m128i_i64[0] + 56LL * v623.m128i_u32[2]);
        while ( (_QWORD *)v255 != v256 )
        {
          while ( 1 )
          {
            v257 = *(v256 - 3);
            v256 -= 7;
            if ( v257 )
              j_j___libc_free_0(v257, v256[6] - v257);
            if ( (_QWORD *)*v256 == v256 + 2 )
              break;
            j_j___libc_free_0(*v256, v256[2] + 1LL);
            if ( (_QWORD *)v255 == v256 )
              goto LABEL_361;
          }
        }
LABEL_361:
        v623.m128i_i32[2] = 0;
      }
      while ( v546 != v578 );
LABEL_362:
      v258 = sub_157EBA0(n);
      if ( *(_BYTE *)(v258 + 16) == 32 && (*(_BYTE *)(v258 + 18) & 1) == 0 && v536 )
        sub_1AEE6A0(v258, 0, 0, 0);
      v259 = sub_157ED20(n);
      v260 = *(unsigned __int8 *)(v259 + 16);
      v261 = (unsigned int)(v260 - 34);
      if ( (unsigned int)v261 <= 0x36 && (v262 = 0x40018000000001LL, _bittest64(&v262, v261)) )
      {
        if ( (_BYTE)v260 == 34 )
        {
          v332 = (_QWORD *)sub_13CF970(v259);
          if ( *(_BYTE *)(*v332 + 16LL) == 16 )
            sub_1593B40(v332, v515);
        }
        else if ( *(_BYTE *)(*(_QWORD *)(v259 - 24) + 16LL) == 16 )
        {
          sub_1593B40((_QWORD *)(v259 - 24), v515);
        }
        v153 = v623.m128i_u32[2];
        v333 = v623.m128i_i64[0];
        v334 = (__int64 *)(v623.m128i_i64[0] + 56LL * v623.m128i_u32[2]);
        if ( (__int64 *)v623.m128i_i64[0] != v334 )
        {
          do
          {
            v335 = *(v334 - 3);
            v334 -= 7;
            if ( v335 )
              j_j___libc_free_0(v335, v334[6] - v335);
            if ( (__int64 *)*v334 != v334 + 2 )
              j_j___libc_free_0(*v334, v334[2] + 1);
          }
          while ( (__int64 *)v333 != v334 );
          v334 = (__int64 *)v623.m128i_i64[0];
        }
        if ( v334 != &v624 )
          _libc_free((unsigned __int64)v334);
      }
      else
      {
        sub_1AD4CC0((__int64)&v623);
      }
    }
  }
  if ( v531 )
  {
    if ( *(_QWORD *)v535 == **(_QWORD **)(*(_QWORD *)(v521 + 24) + 16LL) )
    {
      v409 = &v605[(unsigned int)v606];
      v410 = sub_1AD2CA0(v605, (__int64)v409);
      v411 = v410;
      if ( v409 != v410 )
      {
        for ( ii = v410 + 1; v409 != ii; ++ii )
        {
          if ( !sub_157ECB0(*(_QWORD *)(*ii + 40)) )
            *v411++ = *ii;
        }
      }
      LODWORD(v606) = v411 - v605;
    }
    else
    {
      v623.m128i_i64[0] = (__int64)&v624;
      v623.m128i_i64[1] = 0x800000000LL;
      v616 = **(__m128i ***)(*(_QWORD *)(v521 + 24) + 16LL);
      v263 = sub_15E26F0(*(__int64 **)(v521 + 40), 75, (__int64 *)&v616, 1);
      v266 = (unsigned int)v606;
      v267 = (_WORD *)v263;
      v268 = &v605[(unsigned int)v606];
      v581 = v605;
      for ( jj = v268; jj != v581; ++v581 )
      {
        v269 = (_QWORD *)*v581;
        v270 = sub_157ECB0(*(_QWORD *)(*v581 + 40));
        v271 = v270;
        if ( !v270 )
        {
          v481 = v623.m128i_u32[2];
          if ( v623.m128i_i32[2] >= (unsigned __int32)v623.m128i_i32[3] )
          {
            sub_16CD150((__int64)&v623, &v624, 0, 8, v264, v265);
            v481 = v623.m128i_u32[2];
          }
          v266 = v623.m128i_i64[0];
          *(_QWORD *)(v623.m128i_i64[0] + 8 * v481) = v269;
          ++v623.m128i_i32[2];
          continue;
        }
        v272 = *(_QWORD *)(v270 - 24);
        if ( *(_BYTE *)(v272 + 16) )
          BUG();
        v555 = (*(_WORD *)(v272 + 18) >> 4) & 0x3FF;
        v267[9] = (16 * v555) | v267[9] & 0xC00F;
        v273 = v269[5];
        sub_15F20C0(v269);
        if ( *(char *)(v271 + 23) < 0 )
        {
          v275 = sub_1648A40(v271);
          v277 = v275 + v276;
          if ( *(char *)(v271 + 23) >= 0 )
          {
            if ( (unsigned int)(v277 >> 4) )
              goto LABEL_810;
          }
          else if ( (unsigned int)((v277 - sub_1648A40(v271)) >> 4) )
          {
            if ( *(char *)(v271 + 23) >= 0 )
              goto LABEL_810;
            v278 = *(_DWORD *)(sub_1648A40(v271) + 8);
            if ( *(char *)(v271 + 23) >= 0 )
              goto LABEL_809;
            v279 = sub_1648A40(v271);
            v281 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v279 + v280 - 4) - v278);
            goto LABEL_381;
          }
        }
        v281 = -24;
LABEL_381:
        v282 = *(_DWORD *)(v271 + 20);
        v283 = (__int64 *)(v271 + v281);
        v603 = 0x400000000LL;
        v284 = 3LL * (v282 & 0xFFFFFFF);
        v285 = v604;
        v284 *= 8;
        v286 = (__int64 *)(v271 - v284);
        v287 = v281 + v284;
        v602 = v604;
        v288 = 0;
        v289 = 0xAAAAAAAAAAAAAAABLL * (v287 >> 3);
        if ( (unsigned __int64)v287 > 0x60 )
        {
          v565 = v283;
          sub_16CD150((__int64)&v602, v604, 0xAAAAAAAAAAAAAAABLL * (v287 >> 3), 8, v274, (int)v283);
          v288 = v603;
          v283 = v565;
          v285 = (__m128i *)((char *)v602 + 8 * (unsigned int)v603);
        }
        if ( v286 != v283 )
        {
          do
          {
            if ( v285 )
              v285->m128i_i64[0] = *v286;
            v286 += 3;
            v285 = (__m128i *)((char *)v285 + 8);
          }
          while ( v283 != v286 );
          v288 = v603;
        }
        v608 = (__int64)&v610;
        LODWORD(v603) = v289 + v288;
        v609 = 0x100000000LL;
        sub_1752100(v271, (__int64)&v608);
        sub_15F20C0((_QWORD *)v271);
        v290 = sub_157E9C0(v273);
        v618.m128i_i64[0] = v273 + 40;
        v588 = 257;
        LOWORD(v591) = 257;
        v618.m128i_i64[1] = v290;
        v616 = 0;
        v619 = 0;
        v620 = 0;
        v621 = 0;
        v622 = 0;
        v617 = v273;
        v291 = sub_1AD45C0(
                 *(_QWORD *)(*(_QWORD *)v267 + 24LL),
                 (__int64)v267,
                 v602->m128i_i64,
                 (unsigned int)v603,
                 (__int64 *)v608,
                 (unsigned int)v609,
                 (__int64)&v589,
                 0);
        v292 = *v291;
        v293 = (__int64)v291;
        v294 = *(_BYTE *)(*v291 + 8LL);
        if ( v294 == 16 )
          v294 = *(_BYTE *)(**(_QWORD **)(v292 + 16) + 8LL);
        if ( (unsigned __int8)(v294 - 1) <= 5u || *(_BYTE *)(v293 + 16) == 76 )
        {
          v295 = v620;
          if ( v619 )
            sub_1625C10(v293, 3, v619);
          sub_15F2440(v293, v295);
        }
        if ( v617 )
        {
          v296 = (__int64 *)v618.m128i_i64[0];
          sub_157E9D0(v617 + 40, v293);
          v297 = *(_QWORD *)(v293 + 24);
          v298 = *v296;
          *(_QWORD *)(v293 + 32) = v296;
          v298 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v293 + 24) = v298 | v297 & 7;
          *(_QWORD *)(v298 + 8) = v293 + 24;
          *v296 = *v296 & 7 | (v293 + 24);
        }
        sub_164B780(v293, (__int64 *)v587);
        if ( v616 )
        {
          v589 = v616;
          sub_1623A60((__int64)&v589, (__int64)v616, 2);
          v299 = *(_QWORD *)(v293 + 48);
          if ( v299 )
            sub_161E7C0(v293 + 48, v299);
          v300 = (unsigned __int8 *)v589;
          *(_QWORD *)(v293 + 48) = v589;
          if ( v300 )
            sub_1623210((__int64)&v589, v300, v293 + 48);
        }
        *(_WORD *)(v293 + 18) = (4 * v555) | *(_WORD *)(v293 + 18) & 0x8003;
        v301 = *(_BYTE *)(*(_QWORD *)v293 + 8LL) == 0;
        LOWORD(v591) = 257;
        if ( v301 )
        {
          v302 = v618.m128i_i64[1];
          v303 = sub_1648A60(56, 0);
          v304 = v303;
          if ( v303 )
            sub_15F6F90((__int64)v303, v302, 0, 0);
          if ( v617 )
          {
            v305 = (unsigned __int64 *)v618.m128i_i64[0];
            sub_157E9D0(v617 + 40, (__int64)v304);
            v306 = v304[3];
            v307 = *v305;
            v304[4] = v305;
            v307 &= 0xFFFFFFFFFFFFFFF8LL;
            v304[3] = v307 | v306 & 7;
            *(_QWORD *)(v307 + 8) = v304 + 3;
            *v305 = *v305 & 7 | (unsigned __int64)(v304 + 3);
          }
          sub_164B780((__int64)v304, (__int64 *)&v589);
          if ( v616 )
          {
            v587[0] = (unsigned __int8 *)v616;
            sub_1AD3290((__int64 *)v587);
            sub_1AD34B0(v304 + 6, v587);
            sub_17CD270((__int64 *)v587);
          }
        }
        else
        {
          v558 = v618.m128i_i64[1];
          v413 = sub_1648A60(56, 1u);
          v414 = v413;
          if ( v413 )
            sub_15F6F90((__int64)v413, v558, v293, 0);
          if ( v617 )
          {
            v559 = (unsigned __int64 *)v618.m128i_i64[0];
            sub_157E9D0(v617 + 40, (__int64)v414);
            v415 = *v559;
            v416 = v414[3] & 7LL;
            v414[4] = v559;
            v415 &= 0xFFFFFFFFFFFFFFF8LL;
            v414[3] = v415 | v416;
            *(_QWORD *)(v415 + 8) = v414 + 3;
            *v559 = *v559 & 7 | (unsigned __int64)(v414 + 3);
          }
          sub_164B780((__int64)v414, (__int64 *)&v589);
          if ( v616 )
          {
            v587[0] = (unsigned __int8 *)v616;
            sub_1623A60((__int64)v587, (__int64)v616, 2);
            v417 = v414[6];
            if ( v417 )
              sub_161E7C0((__int64)(v414 + 6), v417);
            v418 = v587[0];
            v414[6] = v587[0];
            if ( v418 )
              sub_1623210((__int64)v587, v418, (__int64)(v414 + 6));
          }
        }
        sub_17CD270((__int64 *)&v616);
        v266 = (unsigned int)v609;
        v308 = v608;
        v309 = (__int64 *)(v608 + 56LL * (unsigned int)v609);
        if ( (__int64 *)v608 != v309 )
        {
          do
          {
            v310 = *(v309 - 3);
            v309 -= 7;
            if ( v310 )
              j_j___libc_free_0(v310, v309[6] - v310);
            if ( (__int64 *)*v309 != v309 + 2 )
              j_j___libc_free_0(*v309, v309[2] + 1);
          }
          while ( (__int64 *)v308 != v309 );
          v309 = (__int64 *)v608;
        }
        if ( v309 != &v610 )
          _libc_free((unsigned __int64)v309);
        if ( v602 != v604 )
          _libc_free((unsigned __int64)v602);
      }
      sub_1AD5740((__int64)&v605, (__int64)&v623, v266, (__int64)v268, v264, v265);
      if ( (__int64 *)v623.m128i_i64[0] != &v624 )
        _libc_free(v623.m128i_u64[0]);
    }
  }
  if ( v534 )
  {
    v582 = **(__int64 ****)(*(_QWORD *)(v521 + 24) + 16LL);
    if ( *(_QWORD *)(v535 + 8) )
      v537 = *(_QWORD *)v535 != **(_QWORD **)(*(_QWORD *)(v521 + 24) + 16LL);
    v311 = v605;
    v312 = (char **)&v589;
    v623.m128i_i64[0] = (__int64)&v624;
    v623.m128i_i64[1] = 0x800000000LL;
    v579 = &v605[(unsigned int)v606];
    if ( v605 != v579 )
    {
      do
      {
        v348 = *v311;
        v340 = sub_157EBE0(*(_QWORD *)(*v311 + 40));
        if ( v340 )
        {
          if ( v537 )
          {
            v349 = *(_QWORD *)(v348 + 40);
            if ( (*(_DWORD *)(v348 + 20) & 0xFFFFFFF) != 0
              && (v350 = *(_QWORD *)(v348 - 24LL * (*(_DWORD *)(v348 + 20) & 0xFFFFFFF))) != 0
              && (v351 = *(_BYTE *)(v350 + 16), v351 > 0x17u) )
            {
              v336 = (_QWORD *)v348;
              if ( v351 == 71 )
              {
                sub_15F20C0((_QWORD *)v348);
                v336 = (_QWORD *)v350;
              }
              sub_15F20C0(v336);
            }
            else
            {
              sub_15F20C0((_QWORD *)v348);
            }
            v337 = sub_157E9C0(v349);
            v617 = v349;
            v618.m128i_i64[1] = v337;
            v338 = v337;
            v616 = 0;
            v619 = 0;
            v620 = 0;
            v621 = 0;
            v622 = 0;
            v618.m128i_i64[0] = v349 + 40;
            v604[0].m128i_i16[0] = 257;
            if ( v582 != *(__int64 ***)v340 )
            {
              if ( *(_BYTE *)(v340 + 16) > 0x10u )
              {
                LOWORD(v610) = 257;
                v340 = sub_15FDBD0(47, v340, (__int64)v582, (__int64)&v608, 0);
                if ( v617 )
                {
                  v391 = (__int64 *)v618.m128i_i64[0];
                  sub_157E9D0(v617 + 40, v340);
                  v392 = *(_QWORD *)(v340 + 24);
                  v393 = *v391;
                  *(_QWORD *)(v340 + 32) = v391;
                  v393 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v340 + 24) = v393 | v392 & 7;
                  *(_QWORD *)(v393 + 8) = v340 + 24;
                  *v391 = *v391 & 7 | (v340 + 24);
                }
                sub_164B780(v340, (__int64 *)&v602);
                sub_12A86E0((__int64 *)&v616, v340);
                v338 = v618.m128i_i64[1];
              }
              else
              {
                v339 = sub_15A46C0(47, (__int64 ***)v340, v582, 0);
                v338 = v618.m128i_i64[1];
                v340 = v339;
              }
            }
            LOWORD(v610) = 257;
            v341 = sub_1648A60(56, v340 != 0);
            v342 = v341;
            if ( v341 )
              sub_15F6F90((__int64)v341, v338, v340, 0);
            if ( v617 )
            {
              v571 = (unsigned __int64 *)v618.m128i_i64[0];
              sub_157E9D0(v617 + 40, (__int64)v342);
              v343 = *v571;
              v344 = v342[3] & 7LL;
              v342[4] = v571;
              v343 &= 0xFFFFFFFFFFFFFFF8LL;
              v342[3] = v343 | v344;
              *(_QWORD *)(v343 + 8) = v342 + 3;
              *v571 = *v571 & 7 | (unsigned __int64)(v342 + 3);
            }
            sub_164B780((__int64)v342, &v608);
            if ( v616 )
            {
              v589 = v616;
              sub_1623A60((__int64)&v589, (__int64)v616, 2);
              v345 = v342[6];
              v346 = (__int64)(v342 + 6);
              if ( v345 )
              {
                sub_161E7C0((__int64)(v342 + 6), v345);
                v346 = (__int64)(v342 + 6);
              }
              v347 = (unsigned __int8 *)v589;
              v342[6] = v589;
              if ( v347 )
                sub_1623210((__int64)&v589, v347, v346);
            }
            sub_17CD270((__int64 *)&v616);
          }
        }
        else
        {
          v451 = v623.m128i_u32[2];
          if ( v623.m128i_i32[2] >= (unsigned __int32)v623.m128i_i32[3] )
          {
            sub_16CD150((__int64)&v623, &v624, 0, 8, v154, v155);
            v451 = v623.m128i_u32[2];
          }
          v153 = v623.m128i_i64[0];
          *(_QWORD *)(v623.m128i_i64[0] + 8 * v451) = v348;
          ++v623.m128i_i32[2];
        }
        ++v311;
      }
      while ( v579 != v311 );
    }
    sub_1AD5740((__int64)&v605, (__int64)&v623, v153, (__int64)v312, v154, v155);
    if ( (__int64 *)v623.m128i_i64[0] != &v624 )
      _libc_free(v623.m128i_u64[0]);
  }
  if ( (_BYTE)v592 )
  {
    if ( !*(_QWORD *)a2 )
    {
      v401 = v514;
      if ( v514 != (__int64 *)v527 )
      {
        while ( 1 )
        {
          v402 = v401[3];
          for ( kk = v401 + 2; kk != (__int64 *)v402; v402 = *(_QWORD *)(v402 + 8) )
          {
            while ( 1 )
            {
              v404 = v402 - 24;
              if ( !v402 )
                v404 = 0;
              v405 = sub_1AD4D60(v404);
              if ( (v405 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                break;
              v402 = *(_QWORD *)(v402 + 8);
              if ( kk == (__int64 *)v402 )
                goto LABEL_588;
            }
            v408 = *(unsigned int *)(a2 + 304);
            if ( (unsigned int)v408 >= *(_DWORD *)(a2 + 308) )
            {
              v584 = (void *)v405;
              sub_16CD150(a2 + 296, (const void *)(a2 + 312), 0, 8, v406, v407);
              v408 = *(unsigned int *)(a2 + 304);
              v405 = (unsigned __int64)v584;
            }
            *(_QWORD *)(*(_QWORD *)(a2 + 296) + 8 * v408) = v405;
            ++*(_DWORD *)(a2 + 304);
          }
LABEL_588:
          v401 = (__int64 *)v401[1];
          if ( (__int64 *)v527 == v401 )
            break;
          if ( !v401 )
            goto LABEL_807;
        }
      }
    }
  }
  v608 = *(_QWORD *)(v535 + 48);
  if ( !v608 || (sub_1AD3290(&v608), !v608) )
  {
LABEL_513:
    if ( (_DWORD)v606 != 1 )
      goto LABEL_514;
    if ( v514 == (__int64 *)v527 )
      goto LABEL_514;
    v464 = (__int64)v514;
    v465 = 0;
    do
    {
      v464 = *(_QWORD *)(v464 + 8);
      ++v465;
    }
    while ( v527 != v464 );
    if ( v465 != 1 )
      goto LABEL_514;
    v466 = (__int64 *)v514[3];
    if ( v510 != (unsigned __int64 *)v466 )
      sub_1AD56A0((__int64)(v512 + 5), (unsigned __int64 *)(v535 + 24), (__int64)v510, v466, v510);
    v467 = (_QWORD *)(*(_QWORD *)(v521 + 72) & 0xFFFFFFFFFFFFFFF8LL);
    sub_15E0220(v464, (__int64)(v467 - 3));
    v468 = (unsigned __int64 *)v467[1];
    v469 = *v467 & 0xFFFFFFFFFFFFFFF8LL;
    *v468 = v469 | *v468 & 7;
    *(_QWORD *)(v469 + 8) = v468;
    *v467 &= 7uLL;
    v467[1] = 0;
    sub_157EF40((__int64)(v467 - 3));
    j_j___libc_free_0(v467 - 3, 64);
    if ( *(_BYTE *)(v535 + 16) == 29 )
    {
      v472 = (__m128i *)sub_1AD4720(*(_QWORD *)(v535 - 48), v535);
      v623.m128i_i64[0] = *(_QWORD *)(*v605 + 48);
      if ( v623.m128i_i64[0] )
        sub_1AD3290(v623.m128i_i64);
      if ( &v472[3] != &v623 )
        sub_1AD34B0(v472[3].m128i_i64, (unsigned __int8 **)&v623);
      sub_17CD270(v623.m128i_i64);
    }
    if ( !*(_QWORD *)(v535 + 8) )
      goto LABEL_717;
    if ( (*(_DWORD *)(*v605 + 20) & 0xFFFFFFF) != 0 )
    {
      v473 = *(_QWORD *)(*v605 - 24LL * (*(_DWORD *)(*v605 + 20) & 0xFFFFFFF));
      if ( v473 && v535 == v473 )
      {
        v505 = sub_1599EF0(*(__int64 ***)v535);
        sub_164D160(v535, v505, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v506, v507, a12, a13);
        goto LABEL_717;
      }
    }
    else
    {
      v473 = 0;
    }
    sub_164D160(v535, v473, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v470, v471, a12, a13);
LABEL_717:
    sub_15F20C0((_QWORD *)v535);
    sub_15F20C0((_QWORD *)*v605);
    goto LABEL_634;
  }
  v313 = v514;
  if ( v514 != (__int64 *)v527 )
  {
    while ( 1 )
    {
      v314 = (__int64 *)v313[3];
      if ( v313 + 2 != v314 )
      {
        while ( v314 )
        {
          if ( v314[3] )
            goto LABEL_513;
          v314 = (__int64 *)v314[1];
          if ( v313 + 2 == v314 )
            goto LABEL_437;
        }
        goto LABEL_807;
      }
LABEL_437:
      v313 = (__int64 *)v313[1];
      if ( (__int64 *)v527 == v313 )
        break;
      if ( !v313 )
        goto LABEL_807;
    }
    v574 = v313;
    v580 = v514;
    while ( 1 )
    {
      v474 = (__int64 *)v580[3];
      v475 = v580 + 2;
      if ( v580 + 2 != v474 )
        break;
LABEL_733:
      v580 = (__int64 *)v580[1];
      if ( v574 == v580 )
        goto LABEL_513;
      if ( !v580 )
        goto LABEL_807;
    }
    while ( 2 )
    {
      while ( 2 )
      {
        v477 = (__m128i *)(v474 - 3);
        if ( !v474 )
          v477 = 0;
        v623.m128i_i64[0] = v608;
        if ( !v608 )
        {
          m128i_i64 = (__int64)v477[3].m128i_i64;
          if ( &v477[3] == &v623 )
            goto LABEL_724;
          v478 = v477[3].m128i_i64[0];
          if ( !v478 )
            goto LABEL_724;
          goto LABEL_730;
        }
        sub_1623A60((__int64)&v623, v608, 2);
        m128i_i64 = (__int64)v477[3].m128i_i64;
        if ( &v477[3] == &v623 )
        {
          if ( v623.m128i_i64[0] )
            sub_161E7C0((__int64)&v623, v623.m128i_i64[0]);
LABEL_724:
          v474 = (__int64 *)v474[1];
          if ( v475 == v474 )
            goto LABEL_733;
          continue;
        }
        break;
      }
      v478 = v477[3].m128i_i64[0];
      if ( v478 )
      {
LABEL_730:
        v585 = (void *)m128i_i64;
        sub_161E7C0(m128i_i64, v478);
        m128i_i64 = (__int64)v585;
      }
      v479 = (unsigned __int8 *)v623.m128i_i64[0];
      v477[3].m128i_i64[0] = v623.m128i_i64[0];
      if ( !v479 )
        goto LABEL_724;
      sub_1623210((__int64)&v623, v479, m128i_i64);
      v474 = (__int64 *)v474[1];
      if ( v475 == v474 )
        goto LABEL_733;
      continue;
    }
  }
LABEL_514:
  if ( *(_BYTE *)(v535 + 16) == 29 )
  {
    v352 = *(_QWORD *)(v535 - 48);
    v353 = (__m128i *)sub_1648A60(56, 1u);
    v557 = v353;
    if ( v353 )
      sub_15F8320((__int64)v353, v352, v535);
    v616 = (__m128i *)sub_1649960(v538);
    v623.m128i_i64[0] = (__int64)&v616;
    v623.m128i_i64[1] = (__int64)".exit";
    v617 = v354;
    LOWORD(v624) = 773;
    v583 = (_QWORD *)sub_157FBF0(v512, &v557[1].m128i_i64[1], (__int64)&v623);
  }
  else
  {
    v616 = (__m128i *)sub_1649960(v538);
    v623.m128i_i64[0] = (__int64)&v616;
    v623.m128i_i64[1] = (__int64)".exit";
    v617 = v443;
    LOWORD(v624) = 773;
    v557 = 0;
    v583 = (_QWORD *)sub_157FBF0(v512, (__int64 *)(v535 + 24), (__int64)&v623);
  }
  v355 = *(__int64 **)(a2 + 24);
  if ( v355 )
  {
    v356 = sub_1368AA0(*(__int64 **)(a2 + 24), (__int64)v512);
    sub_136C010(v355, (__int64)v583, v356);
  }
  v357 = sub_157EBA0((__int64)v512);
  v358 = (_QWORD *)sub_13CF970(v357);
  sub_1593B40(v358, v511);
  if ( v514 != (__int64 *)v527 && v583 + 3 != (_QWORD *)v527 && (__int64 *)v527 != v514 )
  {
    v361 = *(_QWORD *)(v521 + 72) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*v514 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v527;
    *(_QWORD *)(v521 + 72) = *(_QWORD *)(v521 + 72) & 7LL | *v514 & 0xFFFFFFFFFFFFFFF8LL;
    v362 = v583[3];
    *(_QWORD *)(v361 + 8) = v583 + 3;
    v362 &= 0xFFFFFFFFFFFFFFF8LL;
    *v514 = v362 | *v514 & 7;
    *(_QWORD *)(v362 + 8) = v514;
    v583[3] = v361 | v583[3] & 7LL;
  }
  v363 = (unsigned int)v606;
  v364 = *(_QWORD *)(v535 + 8);
  if ( (unsigned int)v606 <= 1 )
  {
    if ( !(_DWORD)v606 )
    {
      if ( v364 )
      {
        v364 = 0;
        v482 = sub_1599EF0(*(__int64 ***)v535);
        sub_164D160(v535, v482, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v483, v484, a12, a13);
      }
      goto LABEL_622;
    }
    if ( v364 )
    {
      if ( (*(_DWORD *)(*v605 + 20) & 0xFFFFFFF) == 0 )
      {
        v444 = 0;
LABEL_653:
        sub_164D160(v535, v444, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v359, v360, a12, a13);
        goto LABEL_654;
      }
      v444 = *(_QWORD *)(*v605 - 24LL * (*(_DWORD *)(*v605 + 20) & 0xFFFFFFF));
      if ( !v444 || v535 != v444 )
        goto LABEL_653;
      v502 = sub_1599EF0(*(__int64 ***)v535);
      sub_164D160(v535, v502, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v503, v504, a12, a13);
    }
LABEL_654:
    v445 = *(_QWORD *)(*v605 + 40);
    v446 = (_QWORD *)(v445 + 40);
    sub_164D160(v445, (__int64)v583, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v359, v360, a12, a13);
    if ( v445 + 40 != (*(_QWORD *)(v445 + 40) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v447 = (unsigned __int64 *)v583[6];
      v448 = *(__int64 **)(v445 + 48);
      if ( v446 != v447 )
      {
        if ( v583 + 5 != v446 )
          sub_157EA80((__int64)(v583 + 5), v445 + 40, *(_QWORD *)(v445 + 48), v445 + 40);
        if ( v446 != v447 && v446 != v448 )
        {
          v449 = *(_QWORD *)(v445 + 40) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v448 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v446;
          *(_QWORD *)(v445 + 40) = *(_QWORD *)(v445 + 40) & 7LL | *v448 & 0xFFFFFFFFFFFFFFF8LL;
          v450 = *v447;
          *(_QWORD *)(v449 + 8) = v447;
          v450 &= 0xFFFFFFFFFFFFFFF8LL;
          *v448 = v450 | *v448 & 7;
          *(_QWORD *)(v450 + 8) = v448;
          *v447 = v449 | *v447 & 7;
        }
      }
    }
    if ( v557 )
    {
      v623.m128i_i64[0] = *(_QWORD *)(*v605 + 48);
      if ( v623.m128i_i64[0] )
        sub_1AD3290(v623.m128i_i64);
      if ( &v557[3] != &v623 )
        sub_1AD34B0(v557[3].m128i_i64, (unsigned __int8 **)&v623);
      sub_17CD270(v623.m128i_i64);
    }
    v364 = 0;
    sub_15F20C0((_QWORD *)*v605);
    sub_157F980(v445);
    goto LABEL_622;
  }
  if ( !v364 )
  {
    v616 = 0;
LABEL_536:
    v382 = 0;
    v572 = 8 * v363;
    v552 = v357;
    while ( 1 )
    {
      v386 = v605[v382 / 8];
      v387 = (__m128i *)sub_1648A60(56, 1u);
      v388 = v387;
      if ( v387 )
        sub_15F8320((__int64)v387, (__int64)v583, v386);
      if ( &v616 == (__m128i **)(v386 + 48) )
        goto LABEL_538;
      sub_17CD270((__int64 *)&v616);
      v616 = *(__m128i **)(v386 + 48);
      if ( v616 )
        break;
      v623.m128i_i64[0] = 0;
LABEL_540:
      v383 = (__int64)v388[3].m128i_i64;
      if ( &v388[3] != &v623 )
      {
        v384 = v388[3].m128i_i64[0];
        if ( v384 )
        {
          sub_161E7C0((__int64)v388[3].m128i_i64, v384);
          v383 = (__int64)v388[3].m128i_i64;
        }
        v385 = (unsigned __int8 *)v623.m128i_i64[0];
        v388[3].m128i_i64[0] = v623.m128i_i64[0];
        if ( v385 )
        {
          sub_1623210((__int64)&v623, v385, v383);
          v623.m128i_i64[0] = 0;
        }
      }
      v382 += 8LL;
      sub_17CD270(v623.m128i_i64);
      sub_15F20C0((_QWORD *)v386);
      if ( v382 == v572 )
      {
        v357 = v552;
        goto LABEL_615;
      }
    }
    sub_1AD3290((__int64 *)&v616);
LABEL_538:
    v623.m128i_i64[0] = (__int64)v616;
    if ( v616 )
      sub_1AD3290(v623.m128i_i64);
    goto LABEL_540;
  }
  v365 = v583[6];
  if ( v365 )
    v365 -= 24;
  v366 = **(_QWORD **)(*(_QWORD *)(v538 + 24) + 16LL);
  v367 = (unsigned __int8 *)sub_1649960(v535);
  v368 = v606;
  v616 = (__m128i *)v367;
  v617 = v369;
  LOWORD(v624) = 261;
  v623.m128i_i64[0] = (__int64)&v616;
  v370 = sub_1648B60(64);
  v364 = v370;
  if ( !v370 )
  {
    sub_164D160(v535, 0, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v371, v372, a12, a13);
    v363 = (unsigned int)v606;
LABEL_535:
    v616 = 0;
    if ( !(_DWORD)v363 )
      goto LABEL_615;
    goto LABEL_536;
  }
  sub_15F1EA0(v370, v366, 53, 0, 0, v365);
  *(_DWORD *)(v364 + 56) = v368;
  sub_164B780(v364, v623.m128i_i64);
  sub_1648880(v364, *(_DWORD *)(v364 + 56), 1);
  sub_164D160(v535, v364, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v373, v374, a12, a13);
  if ( (_DWORD)v606 )
  {
    v377 = 8LL * (unsigned int)v606;
    v378 = 0;
    do
    {
      v379 = 0;
      v380 = v605[v378 / 8];
      v381 = *(_DWORD *)(v380 + 20) & 0xFFFFFFF;
      if ( v381 )
      {
        v375 = 4LL * v381;
        v379 = *(_QWORD *)(v380 - 24LL * v381);
      }
      v378 += 8LL;
      sub_1704F80(v364, v379, *(_QWORD *)(v380 + 40), v375, *(_QWORD *)(v380 + 40), v376);
    }
    while ( v377 != v378 );
    v363 = (unsigned int)v606;
    goto LABEL_535;
  }
  v616 = 0;
LABEL_615:
  if ( v557 )
  {
    v623.m128i_i64[0] = (__int64)v616;
    if ( v616 )
      sub_1AD3290(v623.m128i_i64);
    if ( &v557[3] != &v623 )
      sub_1AD34B0(v557[3].m128i_i64, (unsigned __int8 **)&v623);
    sub_17CD270(v623.m128i_i64);
  }
  sub_17CD270((__int64 *)&v616);
LABEL_622:
  sub_15F20C0((_QWORD *)v535);
  if ( v534 )
  {
    v480 = v583[1];
    if ( v480 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v480) + 16) - 25) > 9u )
      {
        v480 = *(_QWORD *)(v480 + 8);
        if ( !v480 )
          goto LABEL_741;
      }
    }
    else
    {
LABEL_741:
      sub_157F980((__int64)v583);
    }
  }
  v421 = *(_QWORD **)(v357 - 24);
  sub_164D160(
    (__int64)v421,
    (__int64)v512,
    a6,
    *(double *)a7.m128i_i64,
    *(double *)si128.m128i_i64,
    a9,
    v419,
    v420,
    a12,
    a13);
  v422 = v421 + 5;
  v423 = (__int64)(v512 + 5);
  if ( v421 + 5 != (_QWORD *)(v421[5] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v424 = v357 + 24;
    v425 = (__int64 *)v421[6];
    if ( v422 != (__int64 *)(v357 + 24) )
    {
      if ( (__int64 *)v423 != v422 )
      {
        sub_157EA80(v423, (__int64)(v421 + 5), v421[6], (__int64)(v421 + 5));
        v424 = v357 + 24;
      }
      if ( v422 != v425 )
      {
        v426 = v421[5] & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v425 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v422;
        v421[5] = v421[5] & 7LL | *v425 & 0xFFFFFFFFFFFFFFF8LL;
        v427 = *(_QWORD *)(v357 + 24);
        *(_QWORD *)(v426 + 8) = v424;
        v427 &= 0xFFFFFFFFFFFFFFF8LL;
        *v425 = v427 | *v425 & 7;
        *(_QWORD *)(v427 + 8) = v425;
        *(_QWORD *)(v357 + 24) = v426 | *(_QWORD *)(v357 + 24) & 7LL;
      }
    }
  }
  sub_157EA20(v423, v357);
  v428 = *(unsigned __int64 **)(v357 + 32);
  v429 = *(_QWORD *)(v357 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  *v428 = v429 | *v428 & 7;
  *(_QWORD *)(v429 + 8) = v428;
  *(_QWORD *)(v357 + 24) &= 7uLL;
  *(_QWORD *)(v357 + 32) = 0;
  sub_164BEC0(
    v357,
    v357,
    v429,
    (__int64)v428,
    a6,
    *(double *)a7.m128i_i64,
    *(double *)si128.m128i_i64,
    a9,
    v430,
    v431,
    a12,
    a13);
  sub_15E0220(v527, (__int64)v421);
  v432 = (unsigned __int64 *)v421[4];
  v433 = v421[3] & 0xFFFFFFFFFFFFFFF8LL;
  *v432 = v433 | *v432 & 7;
  *(_QWORD *)(v433 + 8) = v432;
  v421[3] &= 7uLL;
  v421[4] = 0;
  sub_157EF40((__int64)v421);
  j_j___libc_free_0(v421, 64);
  if ( v364 )
  {
    v435 = *(_QWORD *)(a2 + 8);
    if ( v435 )
      v435 = sub_1AD4AA0(*(_QWORD *)(a2 + 8), v521, v434);
    v623 = (__m128i)(unsigned __int64)sub_1632FA0(*(_QWORD *)(v521 + 40));
    v624 = 0;
    v625 = v435;
    v626 = 0;
    v437 = sub_13E3350(v364, &v623, 0, 1, v436);
    if ( v437 )
    {
      sub_164D160(v364, v437, a6, *(double *)a7.m128i_i64, *(double *)si128.m128i_i64, a9, v438, v439, a12, a13);
      sub_15F20C0((_QWORD *)v364);
    }
  }
LABEL_634:
  sub_17CD270(&v608);
  if ( v599 != v601 )
    _libc_free((unsigned __int64)v599);
  if ( src != v598 )
    _libc_free((unsigned __int64)src);
  v440 = v594;
  v441 = (__int64)v593;
  if ( v594 != v593 )
  {
    do
    {
      v442 = v441;
      v441 += 24;
      sub_1455FA0(v442);
    }
    while ( v440 != (_QWORD *)v441 );
    v441 = (__int64)v593;
  }
  if ( v441 )
    j_j___libc_free_0(v441, v595 - v441);
  if ( v605 != (__int64 *)v607 )
    _libc_free((unsigned __int64)v605);
  return 1;
}
