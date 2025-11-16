// Function: sub_194D450
// Address: 0x194d450
//
__int64 __fastcall sub_194D450(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i si128,
        __m128i a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned int v12; // r14d
  __int64 *v14; // r15
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 *v19; // r12
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r14
  unsigned int v22; // eax
  __int64 *v23; // rdx
  __int64 v24; // r12
  const __m128i *v25; // rdi
  void *v26; // rax
  __m128 *v27; // rdx
  __int64 v28; // r14
  void *v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  __m128 *v32; // rdx
  const __m128i *v33; // rbx
  const __m128i *v34; // r12
  __m128 *v35; // rdx
  _QWORD *v36; // rdx
  __int64 v37; // rdi
  unsigned int v38; // eax
  bool v39; // cc
  size_t v40; // rdx
  char *v41; // rsi
  _QWORD *v42; // rax
  unsigned __int64 v43; // r8
  char *v44; // rax
  char *v45; // rsi
  unsigned int v46; // eax
  unsigned int v47; // eax
  unsigned int v48; // ecx
  __int64 v49; // r9
  _BYTE *v50; // rdx
  __int64 v51; // rdx
  _QWORD *v52; // rdx
  __int64 v53; // rdx
  void *v54; // rdx
  _QWORD *v55; // rax
  void *v56; // rdx
  __int64 v57; // r13
  unsigned int v58; // eax
  __int64 v59; // rdi
  _BYTE *v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rdi
  int v63; // r14d
  unsigned __int64 v64; // rax
  int v65; // esi
  __int64 v66; // r14
  __int64 v67; // r12
  unsigned int v68; // r13d
  __int64 v69; // rax
  __int64 *v70; // r13
  __int64 v71; // r12
  unsigned __int64 v72; // rax
  unsigned __int32 v73; // ecx
  __int64 v74; // r12
  __int64 v75; // rax
  unsigned int v76; // eax
  __int64 *v77; // rbx
  __int64 v78; // rax
  int v79; // r10d
  __int64 v80; // rax
  __int64 v81; // rax
  int v82; // r10d
  int v83; // eax
  unsigned int v84; // eax
  __int64 v85; // rdx
  __int64 v86; // rsi
  __int64 v87; // rbx
  __int64 v88; // r12
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // r12
  __int64 v92; // rax
  bool v93; // zf
  int v94; // r10d
  unsigned int v95; // ebx
  int v96; // eax
  bool v97; // al
  __int64 v98; // rax
  __int64 v99; // rax
  unsigned __int64 v100; // rax
  __int64 v101; // r12
  __int64 v102; // rax
  __int64 v103; // r12
  unsigned __int64 v104; // rcx
  void (__fastcall *v105)(__m128i *, __int64, __m128 *, void **); // rax
  __int64 v106; // rsi
  __int64 v107; // rax
  const __m128i *v108; // r15
  __int64 v109; // r13
  _QWORD *v110; // r14
  __int64 v111; // r12
  __int64 v112; // rax
  __int64 v113; // rdx
  unsigned __int32 v114; // eax
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // r12
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // r11
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // rax
  void *v131; // r11
  __int64 v132; // r12
  __int64 v133; // rsi
  int v134; // r8d
  int v135; // r9d
  __int64 v136; // rax
  __m128 *v137; // rax
  __int64 *v138; // rax
  __int64 v139; // rsi
  __int64 v140; // r12
  unsigned __int64 v141; // r15
  __int64 v142; // rax
  char v143; // r15
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // r12
  __int64 v147; // rax
  unsigned __int64 v148; // rax
  _QWORD *v149; // rax
  _QWORD *i; // rdx
  char v151; // cl
  _QWORD *v152; // rax
  _QWORD *v153; // rdx
  char v154; // cl
  __int64 v155; // rax
  __int64 v156; // r8
  __int64 v157; // rbx
  __int64 v158; // r12
  __int64 v159; // rdi
  __int64 v160; // rsi
  __int64 v161; // rax
  __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // rsi
  __int64 *p_j; // r15
  __int64 v166; // rbx
  __int64 *v167; // r12
  __int64 v168; // rsi
  __int64 v169; // rdi
  _QWORD *v170; // r15
  _QWORD *v171; // r12
  double v172; // xmm0_8
  double v173; // xmm4_8
  double v174; // xmm5_8
  double v175; // xmm0_8
  double v176; // xmm4_8
  double v177; // xmm5_8
  __int64 *v178; // r12
  _QWORD *v179; // r12
  __int64 v180; // rax
  _QWORD *v181; // r15
  __int64 v182; // rdx
  __int64 v183; // rax
  _QWORD *v184; // r12
  __int64 v185; // rax
  _QWORD *v186; // r15
  __int64 v187; // rdx
  __int64 v188; // rax
  const char *v189; // rsi
  _QWORD *v190; // rbx
  _QWORD *v191; // r12
  __int64 v192; // rax
  __int64 v193; // rdi
  __int64 v194; // rdx
  unsigned __int64 v195; // r12
  _BYTE *v196; // rbx
  __int64 v197; // rax
  __int64 *v198; // rdx
  __int64 v199; // rsi
  unsigned __int64 v200; // rcx
  __int64 v201; // rcx
  __m128i v202; // xmm4
  __int64 *v203; // rax
  __int64 v204; // rax
  unsigned __int64 v205; // rax
  __int64 v206; // rdi
  __int64 v207; // rsi
  __int64 v208; // rax
  __int64 v209; // rax
  __int64 v210; // rax
  __int64 *v211; // rax
  _QWORD *v212; // rbx
  _QWORD *v213; // r12
  __int64 v214; // rsi
  _QWORD *v215; // rbx
  _QWORD *v216; // r12
  __int64 v217; // rsi
  __int64 v218; // rax
  __int64 v219; // rax
  __int64 v220; // rax
  __int64 v221; // rdx
  __int64 v222; // rdi
  __m128i v223; // xmm0
  __int64 v224; // r12
  const char *v225; // rax
  size_t v226; // rdx
  _WORD *v227; // rdi
  char *v228; // rsi
  size_t v229; // r14
  unsigned __int64 v230; // rax
  __int64 v231; // rsi
  __int64 v232; // rax
  void *v233; // rdx
  __int64 v234; // rdi
  __int64 v235; // rax
  __int64 *v236; // r12
  __int64 v237; // rax
  __int64 v238; // rax
  __int64 v239; // rax
  __int64 v240; // r12
  __int64 v241; // rax
  __int64 v242; // rbx
  __int64 v243; // rax
  __int64 v244; // rax
  __int64 v245; // r12
  __int64 v246; // rax
  char v247; // al
  unsigned __int64 v248; // rax
  __int64 v249; // rax
  __int64 v250; // rax
  __int64 v251; // rax
  int v252; // eax
  bool v253; // al
  bool v254; // bl
  char v255; // al
  char v256; // al
  char v257; // r11
  bool v258; // al
  bool v259; // dl
  unsigned __int64 v260; // rax
  __int64 v261; // rax
  __int64 v262; // rax
  __int64 v263; // rax
  __int64 v264; // rax
  __int64 v265; // rax
  __int64 v266; // rax
  __int64 v267; // rax
  __int64 v268; // rax
  __int64 v269; // rax
  __int64 v270; // [rsp+8h] [rbp-958h]
  __int64 v271; // [rsp+10h] [rbp-950h]
  __int64 v272; // [rsp+18h] [rbp-948h]
  __int64 v273; // [rsp+20h] [rbp-940h]
  bool v274; // [rsp+2Bh] [rbp-935h]
  _BOOL4 v275; // [rsp+2Ch] [rbp-934h]
  __int64 v276; // [rsp+30h] [rbp-930h]
  __int64 v277; // [rsp+30h] [rbp-930h]
  __int64 v278; // [rsp+30h] [rbp-930h]
  void *v279; // [rsp+30h] [rbp-930h]
  __int64 v280; // [rsp+38h] [rbp-928h]
  __int64 v281; // [rsp+40h] [rbp-920h]
  __int64 v282; // [rsp+48h] [rbp-918h]
  __int64 v283; // [rsp+48h] [rbp-918h]
  unsigned __int32 v284; // [rsp+48h] [rbp-918h]
  __int64 v285; // [rsp+50h] [rbp-910h]
  __int64 v286; // [rsp+50h] [rbp-910h]
  __int64 v287; // [rsp+50h] [rbp-910h]
  int v288; // [rsp+50h] [rbp-910h]
  __int64 v289; // [rsp+58h] [rbp-908h]
  __int64 v290; // [rsp+60h] [rbp-900h]
  int v291; // [rsp+60h] [rbp-900h]
  __int64 v292; // [rsp+60h] [rbp-900h]
  __int64 v293; // [rsp+60h] [rbp-900h]
  int v294; // [rsp+68h] [rbp-8F8h]
  int v295; // [rsp+68h] [rbp-8F8h]
  char v296; // [rsp+68h] [rbp-8F8h]
  __int64 v297; // [rsp+68h] [rbp-8F8h]
  __int64 v298; // [rsp+68h] [rbp-8F8h]
  __int64 v299; // [rsp+68h] [rbp-8F8h]
  char v300; // [rsp+68h] [rbp-8F8h]
  unsigned __int64 v301; // [rsp+70h] [rbp-8F0h]
  __int64 v302; // [rsp+70h] [rbp-8F0h]
  __int64 v303; // [rsp+78h] [rbp-8E8h]
  __int64 v304; // [rsp+78h] [rbp-8E8h]
  unsigned int v305; // [rsp+80h] [rbp-8E0h]
  __int64 v306; // [rsp+80h] [rbp-8E0h]
  unsigned __int8 v307; // [rsp+80h] [rbp-8E0h]
  __int64 v308; // [rsp+80h] [rbp-8E0h]
  __int64 v309; // [rsp+88h] [rbp-8D8h]
  __int64 *v310; // [rsp+88h] [rbp-8D8h]
  unsigned __int64 v311; // [rsp+88h] [rbp-8D8h]
  __int64 v312; // [rsp+90h] [rbp-8D0h]
  char v313; // [rsp+90h] [rbp-8D0h]
  char v314; // [rsp+90h] [rbp-8D0h]
  char v315; // [rsp+90h] [rbp-8D0h]
  __int64 *v316; // [rsp+98h] [rbp-8C8h]
  __int64 v317; // [rsp+A0h] [rbp-8C0h]
  unsigned __int8 v318; // [rsp+A0h] [rbp-8C0h]
  bool v319; // [rsp+A0h] [rbp-8C0h]
  __int64 v320; // [rsp+B0h] [rbp-8B0h]
  unsigned __int32 v321; // [rsp+B0h] [rbp-8B0h]
  __int64 v323; // [rsp+B8h] [rbp-8A8h]
  unsigned __int8 v325; // [rsp+C0h] [rbp-8A0h]
  __int64 *v326; // [rsp+C0h] [rbp-8A0h]
  __int64 v327; // [rsp+C8h] [rbp-898h]
  __int64 v328; // [rsp+C8h] [rbp-898h]
  bool v329; // [rsp+C8h] [rbp-898h]
  void (__fastcall *v330)(__m128i *, __int64, __m128 *, void **); // [rsp+C8h] [rbp-898h]
  __int64 v331; // [rsp+C8h] [rbp-898h]
  __int64 v332; // [rsp+D0h] [rbp-890h]
  __int64 v333; // [rsp+D0h] [rbp-890h]
  __int64 v334; // [rsp+D0h] [rbp-890h]
  __int64 *v335; // [rsp+D0h] [rbp-890h]
  __int64 v336; // [rsp+D0h] [rbp-890h]
  __int64 v337; // [rsp+D8h] [rbp-888h]
  __int64 v338; // [rsp+D8h] [rbp-888h]
  __int64 v340; // [rsp+D8h] [rbp-888h]
  unsigned __int64 v341; // [rsp+D8h] [rbp-888h]
  const __m128i *v342; // [rsp+D8h] [rbp-888h]
  unsigned __int64 v343; // [rsp+D8h] [rbp-888h]
  unsigned __int8 v344; // [rsp+D8h] [rbp-888h]
  __m128 v345; // [rsp+E0h] [rbp-880h] BYREF
  char v346; // [rsp+F0h] [rbp-870h]
  __int64 v347; // [rsp+100h] [rbp-860h] BYREF
  unsigned __int8 v348; // [rsp+108h] [rbp-858h]
  __int64 v349; // [rsp+110h] [rbp-850h]
  unsigned __int8 v350; // [rsp+118h] [rbp-848h]
  unsigned __int8 v351; // [rsp+120h] [rbp-840h]
  const char *v352; // [rsp+130h] [rbp-830h] BYREF
  __int64 v353; // [rsp+138h] [rbp-828h] BYREF
  __int64 v354; // [rsp+140h] [rbp-820h]
  __int64 v355; // [rsp+148h] [rbp-818h]
  __int64 v356; // [rsp+150h] [rbp-810h]
  __int64 v357; // [rsp+158h] [rbp-808h]
  const char *v358; // [rsp+160h] [rbp-800h] BYREF
  __int64 v359; // [rsp+168h] [rbp-7F8h] BYREF
  __int64 v360; // [rsp+170h] [rbp-7F0h]
  __int64 v361; // [rsp+178h] [rbp-7E8h]
  __int64 v362; // [rsp+180h] [rbp-7E0h]
  __int64 v363; // [rsp+188h] [rbp-7D8h]
  const char *v364; // [rsp+190h] [rbp-7D0h] BYREF
  __int64 v365; // [rsp+198h] [rbp-7C8h] BYREF
  __int64 v366; // [rsp+1A0h] [rbp-7C0h] BYREF
  __int64 v367; // [rsp+1A8h] [rbp-7B8h] BYREF
  __int64 j; // [rsp+1B0h] [rbp-7B0h] BYREF
  __int64 v369; // [rsp+1B8h] [rbp-7A8h] BYREF
  __int64 v370[5]; // [rsp+1C0h] [rbp-7A0h] BYREF
  int v371; // [rsp+1E8h] [rbp-778h]
  __int64 v372; // [rsp+1F0h] [rbp-770h]
  __int64 v373; // [rsp+1F8h] [rbp-768h]
  _BYTE *v374; // [rsp+210h] [rbp-750h] BYREF
  __int64 v375; // [rsp+218h] [rbp-748h]
  _BYTE v376[160]; // [rsp+220h] [rbp-740h] BYREF
  __int64 v377; // [rsp+2C0h] [rbp-6A0h] BYREF
  __int64 v378; // [rsp+2C8h] [rbp-698h]
  __int64 v379; // [rsp+2D0h] [rbp-690h]
  __int64 v380; // [rsp+2D8h] [rbp-688h]
  __int64 v381; // [rsp+2E0h] [rbp-680h]
  __int64 v382; // [rsp+2E8h] [rbp-678h]
  __int64 v383; // [rsp+2F0h] [rbp-670h]
  __int64 *v384; // [rsp+2F8h] [rbp-668h]
  __int64 v385; // [rsp+300h] [rbp-660h]
  __int64 v386; // [rsp+308h] [rbp-658h]
  __int64 v387; // [rsp+310h] [rbp-650h]
  unsigned __int64 v388; // [rsp+318h] [rbp-648h]
  unsigned __int64 v389; // [rsp+320h] [rbp-640h]
  char *v390; // [rsp+328h] [rbp-638h] BYREF
  __int64 v391; // [rsp+330h] [rbp-630h]
  __int64 v392; // [rsp+338h] [rbp-628h]
  unsigned __int64 v393; // [rsp+340h] [rbp-620h]
  __int64 v394; // [rsp+348h] [rbp-618h]
  _BOOL4 v395; // [rsp+350h] [rbp-610h]
  __int64 *v396; // [rsp+358h] [rbp-608h]
  __int64 v397; // [rsp+360h] [rbp-600h]
  __int64 v398; // [rsp+368h] [rbp-5F8h]
  __int64 v399; // [rsp+370h] [rbp-5F0h]
  bool v400; // [rsp+378h] [rbp-5E8h]
  char v401; // [rsp+379h] [rbp-5E7h]
  __int64 v402; // [rsp+380h] [rbp-5E0h] BYREF
  __int64 v403; // [rsp+388h] [rbp-5D8h]
  __int64 v404; // [rsp+390h] [rbp-5D0h]
  __int64 v405; // [rsp+398h] [rbp-5C8h] BYREF
  _QWORD *v406; // [rsp+3A0h] [rbp-5C0h]
  __int64 v407; // [rsp+3A8h] [rbp-5B8h]
  unsigned int v408; // [rsp+3B0h] [rbp-5B0h]
  _QWORD *v409; // [rsp+3C0h] [rbp-5A0h]
  unsigned int v410; // [rsp+3D0h] [rbp-590h]
  char v411; // [rsp+3D8h] [rbp-588h]
  char v412; // [rsp+3E1h] [rbp-57Fh]
  const char *v413; // [rsp+3E8h] [rbp-578h] BYREF
  __int64 v414; // [rsp+3F0h] [rbp-570h]
  __int64 v415; // [rsp+3F8h] [rbp-568h]
  __int64 v416; // [rsp+400h] [rbp-560h]
  __int64 v417; // [rsp+408h] [rbp-558h]
  int v418; // [rsp+410h] [rbp-550h]
  __int64 v419; // [rsp+418h] [rbp-548h]
  __int64 v420; // [rsp+420h] [rbp-540h]
  __int64 v421; // [rsp+428h] [rbp-538h]
  __int64 v422; // [rsp+430h] [rbp-530h]
  __int16 v423; // [rsp+438h] [rbp-528h]
  void *v424; // [rsp+440h] [rbp-520h] BYREF
  __int64 v425; // [rsp+448h] [rbp-518h] BYREF
  __int64 v426; // [rsp+450h] [rbp-510h]
  __int64 v427; // [rsp+458h] [rbp-508h] BYREF
  _QWORD *v428; // [rsp+460h] [rbp-500h]
  __int64 v429; // [rsp+468h] [rbp-4F8h]
  unsigned int v430; // [rsp+470h] [rbp-4F0h]
  _QWORD *v431; // [rsp+480h] [rbp-4E0h]
  unsigned int v432; // [rsp+490h] [rbp-4D0h]
  char v433; // [rsp+498h] [rbp-4C8h]
  char v434; // [rsp+4A1h] [rbp-4BFh]
  _QWORD v435[5]; // [rsp+4A8h] [rbp-4B8h] BYREF
  int v436; // [rsp+4D0h] [rbp-490h]
  __int64 v437; // [rsp+4D8h] [rbp-488h]
  __int64 v438; // [rsp+4E0h] [rbp-480h]
  __int64 v439; // [rsp+4E8h] [rbp-478h]
  __int64 v440; // [rsp+4F0h] [rbp-470h]
  __int16 v441; // [rsp+4F8h] [rbp-468h]
  __m128i v442; // [rsp+500h] [rbp-460h] BYREF
  const char *v443; // [rsp+510h] [rbp-450h] BYREF
  __int64 v444; // [rsp+518h] [rbp-448h]
  _QWORD *v445; // [rsp+520h] [rbp-440h]
  __int64 v446; // [rsp+528h] [rbp-438h] BYREF
  unsigned int v447; // [rsp+530h] [rbp-430h]
  __int64 v448; // [rsp+538h] [rbp-428h]
  __int64 v449; // [rsp+540h] [rbp-420h]
  __int64 v450; // [rsp+548h] [rbp-418h]
  __int64 v451; // [rsp+550h] [rbp-410h]
  __int64 v452; // [rsp+558h] [rbp-408h]
  __int64 v453; // [rsp+560h] [rbp-400h]
  __int64 v454; // [rsp+568h] [rbp-3F8h]
  __int64 v455; // [rsp+570h] [rbp-3F0h]
  __int64 v456; // [rsp+578h] [rbp-3E8h]
  __int64 v457; // [rsp+580h] [rbp-3E0h]
  __int64 v458; // [rsp+588h] [rbp-3D8h]
  int v459; // [rsp+590h] [rbp-3D0h]
  __int64 v460; // [rsp+598h] [rbp-3C8h]
  _BYTE *v461; // [rsp+5A0h] [rbp-3C0h]
  _BYTE *v462; // [rsp+5A8h] [rbp-3B8h]
  __int64 v463; // [rsp+5B0h] [rbp-3B0h]
  int v464; // [rsp+5B8h] [rbp-3A8h]
  _BYTE v465[16]; // [rsp+5C0h] [rbp-3A0h] BYREF
  __int64 v466; // [rsp+5D0h] [rbp-390h]
  __int64 v467; // [rsp+5D8h] [rbp-388h]
  __int64 v468; // [rsp+5E0h] [rbp-380h]
  __int64 v469; // [rsp+5E8h] [rbp-378h]
  __int64 v470; // [rsp+5F0h] [rbp-370h]
  __int64 v471; // [rsp+5F8h] [rbp-368h]
  __int16 v472; // [rsp+600h] [rbp-360h]
  __int64 v473[5]; // [rsp+608h] [rbp-358h] BYREF
  int v474; // [rsp+630h] [rbp-330h]
  __int64 v475; // [rsp+638h] [rbp-328h]
  __int64 v476; // [rsp+640h] [rbp-320h]
  __int64 v477; // [rsp+648h] [rbp-318h]
  _BYTE *v478; // [rsp+650h] [rbp-310h]
  __int64 v479; // [rsp+658h] [rbp-308h]
  _BYTE v480[64]; // [rsp+660h] [rbp-300h] BYREF
  const __m128i *v481; // [rsp+6A0h] [rbp-2C0h] BYREF
  __int64 v482; // [rsp+6A8h] [rbp-2B8h]
  _BYTE v483[688]; // [rsp+6B0h] [rbp-2B0h] BYREF

  if ( (unsigned int)dword_4FB0000 <= (unsigned __int64)((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3) )
    return 0;
  v14 = a1;
  v15 = a2;
  v16 = sub_13FC520(a2);
  v317 = v16;
  if ( !v16 )
    return 0;
  v17 = sub_157E9C0(v16);
  v18 = *(__int64 **)(a2 + 32);
  v19 = *(__int64 **)(a2 + 40);
  v316 = (__int64 *)v17;
  v481 = (const __m128i *)v483;
  v482 = 0x1000000000LL;
  if ( v18 == v19 )
    return 0;
  do
  {
    v20 = sub_157EBA0(*v18);
    v21 = v20;
    if ( *(_BYTE *)(v20 + 16) == 26 && (*(_DWORD *)(v20 + 20) & 0xFFFFFFF) != 1 )
    {
      v327 = *a1;
      v332 = a1[1];
      v337 = *(_QWORD *)(v20 + 40);
      if ( v337 != sub_13FCB50(a2) )
      {
        sub_16AF710(&v424, 0xFu, 0x10u);
        if ( byte_4FAFC80 == 1
          || !v332
          || (v22 = sub_1377370(v332, *(_QWORD *)(v21 + 40), 0), (unsigned int)v424 <= v22) )
        {
          v442.m128i_i64[0] = 0;
          v442.m128i_i64[1] = (__int64)&v446;
          v443 = (const char *)&v446;
          v444 = 8;
          LODWORD(v445) = 0;
          v23 = (*(_BYTE *)(v21 + 23) & 0x40) != 0
              ? *(__int64 **)(v21 - 8)
              : (__int64 *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
          sub_194A200(a2, v327, v23, (__int64)&v481, (__int64)&v442);
          if ( v443 != (const char *)v442.m128i_i64[1] )
            _libc_free((unsigned __int64)v443);
        }
      }
    }
    ++v18;
  }
  while ( v19 != v18 );
  if ( (_DWORD)v482 )
  {
    if ( byte_4FAFE40 )
    {
      v26 = sub_16E8CB0();
      v27 = (__m128 *)*((_QWORD *)v26 + 3);
      v28 = (__int64)v26;
      if ( *((_QWORD *)v26 + 2) - (_QWORD)v27 <= 0x15u )
      {
        sub_16E7EE0((__int64)v26, "irce: looking at loop ", 0x16u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42BEB50);
        v27[1].m128_i32[0] = 1869573152;
        v27[1].m128_i16[2] = 8304;
        *v27 = (__m128)si128;
        *((_QWORD *)v26 + 3) += 22LL;
      }
      sub_13FA7B0((_QWORD **)a2, v28, 0, 0);
      v29 = *(void **)(v28 + 24);
      if ( *(_QWORD *)(v28 + 16) - (_QWORD)v29 <= 0xEu )
      {
        v30 = sub_16E7EE0(v28, "irce: loop has ", 0xFu);
      }
      else
      {
        v30 = v28;
        qmemcpy(v29, "irce: loop has ", 15);
        *(_QWORD *)(v28 + 24) += 15LL;
      }
      v31 = sub_16E7A90(v30, (unsigned int)v482);
      v32 = *(__m128 **)(v31 + 24);
      if ( *(_QWORD *)(v31 + 16) - (_QWORD)v32 <= 0x19u )
      {
        sub_16E7EE0(v31, " inductive range checks: \n", 0x1Au);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42BEB60);
        qmemcpy(&v32[1], " checks: \n", 10);
        *v32 = (__m128)si128;
        *(_QWORD *)(v31 + 24) += 26LL;
      }
      v33 = v481;
      v34 = (const __m128i *)((char *)v481 + 40 * (unsigned int)v482);
      if ( v481 != v34 )
      {
        while ( 1 )
        {
          v35 = *(__m128 **)(v28 + 24);
          if ( *(_QWORD *)(v28 + 16) - (_QWORD)v35 <= 0x14u )
          {
            sub_16E7EE0(v28, "InductiveRangeCheck:\n", 0x15u);
            v36 = *(_QWORD **)(v28 + 24);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_42BEB70);
            v35[1].m128_i32[0] = 980116325;
            v35[1].m128_i8[4] = 10;
            *v35 = (__m128)si128;
            v36 = (_QWORD *)(*(_QWORD *)(v28 + 24) + 21LL);
            *(_QWORD *)(v28 + 24) = v36;
          }
          if ( *(_QWORD *)(v28 + 16) - (_QWORD)v36 <= 7u )
          {
            v37 = sub_16E7EE0(v28, "  Kind: ", 8u);
            v38 = v33[2].m128i_u32[0];
            v39 = v38 <= 3;
            if ( v38 == 3 )
            {
LABEL_144:
              v40 = 16;
              v41 = "RANGE_CHECK_BOTH";
              goto LABEL_39;
            }
          }
          else
          {
            v37 = v28;
            *v36 = 0x203A646E694B2020LL;
            *(_QWORD *)(v28 + 24) += 8LL;
            v38 = v33[2].m128i_u32[0];
            v39 = v38 <= 3;
            if ( v38 == 3 )
              goto LABEL_144;
          }
          v40 = 19;
          v41 = "RANGE_CHECK_UNKNOWN";
          if ( v39 )
          {
            v41 = "RANGE_CHECK_LOWER";
            v40 = 17;
            if ( v38 != 1 )
              v41 = "RANGE_CHECK_UPPER";
          }
LABEL_39:
          v42 = *(_QWORD **)(v37 + 24);
          if ( *(_QWORD *)(v37 + 16) - (_QWORD)v42 < v40 )
          {
            v37 = sub_16E7EE0(v37, v41, v40);
            v50 = *(_BYTE **)(v37 + 24);
            if ( *(_BYTE **)(v37 + 16) == v50 )
              goto LABEL_136;
          }
          else
          {
            v43 = (unsigned __int64)(v42 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *v42 = *(_QWORD *)v41;
            *(_QWORD *)((char *)v42 + v40 - 8) = *(_QWORD *)&v41[v40 - 8];
            v44 = (char *)v42 - v43;
            v45 = (char *)(v41 - v44);
            v46 = (v40 + (_DWORD)v44) & 0xFFFFFFF8;
            if ( v46 >= 8 )
            {
              v47 = v46 & 0xFFFFFFF8;
              v48 = 0;
              do
              {
                v49 = v48;
                v48 += 8;
                *(_QWORD *)(v43 + v49) = *(_QWORD *)&v45[v49];
              }
              while ( v48 < v47 );
            }
            v50 = (_BYTE *)(*(_QWORD *)(v37 + 24) + v40);
            *(_QWORD *)(v37 + 24) = v50;
            if ( *(_BYTE **)(v37 + 16) == v50 )
            {
LABEL_136:
              sub_16E7EE0(v37, "\n", 1u);
              goto LABEL_45;
            }
          }
          *v50 = 10;
          ++*(_QWORD *)(v37 + 24);
LABEL_45:
          v51 = *(_QWORD *)(v28 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(v28 + 16) - v51) <= 8 )
          {
            sub_16E7EE0(v28, "  Begin: ", 9u);
          }
          else
          {
            *(_BYTE *)(v51 + 8) = 32;
            *(_QWORD *)v51 = 0x3A6E696765422020LL;
            *(_QWORD *)(v28 + 24) += 9LL;
          }
          sub_1456620(v33->m128i_i64[0], v28);
          v52 = *(_QWORD **)(v28 + 24);
          if ( *(_QWORD *)(v28 + 16) - (_QWORD)v52 <= 7u )
          {
            sub_16E7EE0(v28, "  Step: ", 8u);
          }
          else
          {
            *v52 = 0x203A706574532020LL;
            *(_QWORD *)(v28 + 24) += 8LL;
          }
          sub_1456620(v33->m128i_i64[1], v28);
          v53 = *(_QWORD *)(v28 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(v28 + 16) - v53) <= 6 )
          {
            sub_16E7EE0(v28, "  End: ", 7u);
          }
          else
          {
            *(_DWORD *)v53 = 1850023968;
            *(_WORD *)(v53 + 4) = 14948;
            *(_BYTE *)(v53 + 6) = 32;
            *(_QWORD *)(v28 + 24) += 7LL;
          }
          sub_1456620(v33[1].m128i_i64[0], v28);
          v54 = *(void **)(v28 + 24);
          if ( *(_QWORD *)(v28 + 16) - (_QWORD)v54 <= 0xCu )
          {
            sub_16E7EE0(v28, "\n  CheckUse: ", 0xDu);
          }
          else
          {
            qmemcpy(v54, "\n  CheckUse: ", 13);
            *(_QWORD *)(v28 + 24) += 13LL;
          }
          v55 = sub_1648700(v33[1].m128i_i64[1]);
          sub_155C2B0((__int64)v55, v28, 0);
          v56 = *(void **)(v28 + 24);
          if ( *(_QWORD *)(v28 + 16) - (_QWORD)v56 <= 9u )
          {
            v57 = sub_16E7EE0(v28, " Operand: ", 0xAu);
          }
          else
          {
            v57 = v28;
            qmemcpy(v56, " Operand: ", 10);
            *(_QWORD *)(v28 + 24) += 10LL;
          }
          v58 = sub_1648720(v33[1].m128i_i64[1]);
          v59 = sub_16E7A90(v57, v58);
          v60 = *(_BYTE **)(v59 + 24);
          if ( *(_BYTE **)(v59 + 16) == v60 )
          {
            sub_16E7EE0(v59, "\n", 1u);
          }
          else
          {
            *v60 = 10;
            ++*(_QWORD *)(v59 + 24);
          }
          v33 = (const __m128i *)((char *)v33 + 40);
          if ( v34 == v33 )
          {
            v15 = a2;
            break;
          }
        }
      }
    }
    v24 = v14[1];
    v338 = *v14;
    if ( !(unsigned __int8)sub_13FCBF0(v15) )
      goto LABEL_20;
    v289 = sub_13FCB50(v15);
    v61 = sub_157EBA0(v289);
    if ( *(_QWORD *)(v61 + 48) || *(__int16 *)(v61 + 18) < 0 )
    {
      if ( sub_1625940(v61, "irce.loop.clone", 0xFu) )
        goto LABEL_20;
    }
    v62 = sub_157EBA0(v289);
    if ( !v62 )
      goto LABEL_20;
    v63 = sub_15F4D60(v62);
    v64 = sub_157EBA0(v289);
    v65 = v63;
    if ( !v63 )
      goto LABEL_20;
    v333 = v24;
    v66 = v15 + 56;
    v67 = v64;
    v328 = v15;
    v68 = 0;
    while ( 1 )
    {
      v69 = sub_15F4DF0(v67, v68);
      if ( !sub_1377F70(v66, v69) )
        break;
      if ( v65 == ++v68 )
        goto LABEL_20;
    }
    v70 = (__int64 *)v328;
    v71 = v333;
    v281 = **(_QWORD **)(v328 + 32);
    v334 = sub_13FC520(v328);
    if ( !v334 )
      goto LABEL_20;
    v72 = sub_157EBA0(v289);
    v301 = v72;
    if ( *(_BYTE *)(v72 + 16) != 26 || (*(_DWORD *)(v72 + 20) & 0xFFFFFFF) == 1 )
      goto LABEL_20;
    v329 = v281 == *(_QWORD *)(v72 - 24);
    v73 = 0;
    v290 = *(_QWORD *)(v72 - 24);
    v275 = v329;
    if ( v71 )
      v73 = sub_1377370(v71, *(_QWORD *)(v72 + 40), v329);
    if ( !byte_4FAFC80 )
    {
      v321 = v73;
      sub_16AF710(&v442, 1u, dword_4FAFD60);
      if ( v442.m128i_i32[0] < v321 )
        goto LABEL_20;
    }
    v74 = *(_QWORD *)(v301 - 72);
    if ( *(_BYTE *)(v74 + 16) != 75 )
      goto LABEL_20;
    if ( *(_BYTE *)(**(_QWORD **)(v74 - 48) + 8LL) != 11 )
      goto LABEL_20;
    v75 = sub_1474160(v338, (__int64)v70, v289);
    LOBYTE(v76) = sub_14562D0(v75);
    v12 = v76;
    if ( (_BYTE)v76 )
      goto LABEL_20;
    v77 = *(__int64 **)(v74 - 48);
    v303 = (__int64)v77;
    v305 = *(_WORD *)(v74 + 18) & 0x7FFF;
    v312 = sub_146F1B0(v338, (__int64)v77);
    v320 = *v77;
    v280 = *(_QWORD *)(v74 - 24);
    v78 = sub_146F1B0(v338, v280);
    v79 = v305;
    v309 = v78;
    if ( *(_WORD *)(v312 + 24) != 7 )
    {
      if ( *(_WORD *)(v78 + 24) != 7 )
        goto LABEL_20;
      v79 = sub_15FF5D0(v305);
      v80 = v312;
      v312 = v309;
      v309 = v80;
      v303 = v280;
      v280 = (__int64)v77;
    }
    if ( *(_QWORD *)(v312 + 40) != 2 )
      goto LABEL_20;
    v294 = v79;
    v81 = sub_13A5BC0((_QWORD *)v312, v338);
    if ( *(_WORD *)(v81 + 24) )
      goto LABEL_20;
    v82 = v294;
    v306 = *(_QWORD *)(v81 + 32);
    v83 = *(unsigned __int16 *)(v74 + 18);
    BYTE1(v83) &= ~0x80u;
    if ( (unsigned int)(v83 - 32) <= 1 && (*(_BYTE *)(v312 + 26) & 4) == 0 )
    {
      v239 = sub_1456040(**(_QWORD **)(v312 + 32));
      v240 = sub_1644900(*(_QWORD **)v239, 2 * (*(_DWORD *)(v239 + 8) >> 8));
      v241 = sub_147B0D0(v338, v312, v240, 0);
      v82 = v294;
      v242 = v241;
      if ( *(_WORD *)(v241 + 24) != 7 )
        goto LABEL_359;
      v288 = v294;
      v299 = sub_147B0D0(v338, **(_QWORD **)(v312 + 32), v240, 0);
      v243 = sub_13A5BC0((_QWORD *)v312, v338);
      v244 = sub_147B0D0(v338, v243, v240, 0);
      v82 = v288;
      v245 = v244;
      if ( v299 != **(_QWORD **)(v242 + 32) || (v246 = sub_13A5BC0((_QWORD *)v242, v338), v82 = v288, v245 != v246) )
      {
LABEL_359:
        if ( (*(_BYTE *)(v312 + 26) & 4) == 0 )
          goto LABEL_20;
      }
    }
    v84 = *(_DWORD *)(v306 + 32);
    v85 = *(_QWORD *)(v306 + 24);
    v86 = 1LL << ((unsigned __int8)v84 - 1);
    if ( v84 > 0x40 )
      v85 = *(_QWORD *)(v85 + 8LL * ((v84 - 1) >> 6));
    v276 = v306 + 24;
    v87 = v85 & v86;
    v295 = v82;
    v274 = (v85 & v86) == 0;
    sub_15FF7F0(v82);
    v88 = **(_QWORD **)(v312 + 32);
    v89 = sub_13A5BC0((_QWORD *)v312, v338);
    v90 = sub_1480620(v338, v89, 0);
    v91 = sub_13A5B00(v338, v88, v90, 0, 0);
    v285 = sub_146F1B0(v338, v306);
    v92 = sub_159C470(v320, 1, 0);
    v93 = v87 == 0;
    v94 = v295;
    v282 = v92;
    v95 = *(_DWORD *)(v306 + 32);
    if ( v93 )
    {
      if ( v95 <= 0x40 )
      {
        v97 = *(_QWORD *)(v306 + 24) == 1;
      }
      else
      {
        v96 = sub_16A57B0(v276);
        v94 = v295;
        v97 = v95 - 1 == v96;
      }
      if ( !v97 )
        goto LABEL_378;
      if ( v94 == 33 && v329 )
      {
        if ( !(unsigned __int8)sub_1948D70(v91, (__int64)v70, v338)
          || (v247 = sub_1948D70(v309, (__int64)v70, v338), v94 = 36, !v247) )
        {
          v94 = 40;
        }
        goto LABEL_91;
      }
      if ( v94 == 32 && v281 != v290 )
      {
        if ( (*(_BYTE *)(v312 + 26) & 2) != 0 && (v296 = sub_1949540(v309, (__int64)v70, v338, 0)) != 0 )
        {
          v249 = sub_1456040(v309);
          v250 = sub_145CF80(v338, v249, 1, 0);
          v251 = sub_14806B0(v338, v309, v250, 0, 0);
          v94 = 34;
          v309 = v251;
        }
        else
        {
          v296 = sub_1949540(v309, (__int64)v70, v338, 1);
          if ( v296 )
          {
            v264 = sub_1456040(v309);
            v265 = sub_145CF80(v338, v264, 1, 0);
            v266 = sub_14806B0(v338, v309, v265, 0, 0);
            v94 = 38;
            v309 = v266;
          }
          else
          {
            v94 = 32;
          }
        }
      }
      else
      {
LABEL_378:
        v296 = v329 && ((v94 - 36) & 0xFFFFFFFB) == 0;
        if ( v296 )
        {
LABEL_91:
          v296 = 0;
LABEL_92:
          v291 = v94;
          v313 = sub_15FF7F0(v94);
          if ( (v313 || byte_4FAFBA0)
            && (unsigned __int8)sub_1949930(v91, v309, v285, v291, v329, (__int64)v70, si128, a6, v338) )
          {
            if ( !v329 && !v296 )
            {
              v248 = sub_157EBA0(v334);
              sub_17CE510((__int64)&v442, v248, 0, 0, 0);
              LOWORD(v426) = 257;
              v280 = sub_12899C0(v442.m128i_i64, v280, v282, (__int64)&v424, 0, 0);
              sub_17CD270(v442.m128i_i64);
            }
            goto LABEL_98;
          }
          goto LABEL_20;
        }
      }
      if ( v281 == v290 || (v94 & 0xFFFFFFFB) != 0x22 )
        goto LABEL_20;
      goto LABEL_92;
    }
    if ( v95 <= 0x40 )
    {
      v253 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v95) == *(_QWORD *)(v306 + 24);
    }
    else
    {
      v252 = sub_16A58F0(v276);
      v94 = v295;
      v253 = v95 == v252;
    }
    if ( !v253 )
      goto LABEL_403;
    v254 = v329 && v94 == 33;
    if ( v254 )
    {
      v313 = sub_1949C30(v91, v309, v285, 38, v329, (__int64)v70, si128, a6, v338);
      if ( v313 )
        goto LABEL_98;
      goto LABEL_20;
    }
    if ( v94 == 32 && v281 != v290 )
    {
      if ( (*(_BYTE *)(v312 + 26) & 2) != 0 && (v255 = sub_1949670(v309, (__int64)v70, v338, 0)) != 0 )
      {
        v315 = v255;
        v267 = sub_1456040(v309);
        v268 = sub_145CF80(v338, v267, 1, 0);
        v269 = sub_13A5B00(v338, v309, v268, 0, 0);
        v257 = v315;
        v94 = 36;
        v309 = v269;
      }
      else
      {
        v256 = sub_1949670(v309, (__int64)v70, v338, 1);
        v257 = v256;
        if ( v256 )
        {
          v314 = v256;
          v261 = sub_1456040(v309);
          v262 = sub_145CF80(v338, v261, 1, 0);
          v263 = sub_13A5B00(v338, v309, v262, 0, 0);
          v257 = v314;
          v94 = 40;
          v309 = v263;
        }
        else
        {
          v94 = 32;
        }
      }
      v258 = v94 == 40;
      v259 = ((v94 - 36) & 0xFFFFFFFB) == 0;
    }
    else
    {
LABEL_403:
      v258 = v94 == 40;
      v254 = v94 == 38;
      v257 = v329 && (v94 & 0xFFFFFFFB) == 34;
      if ( v257 )
      {
        v257 = 0;
LABEL_394:
        v313 = v254 || v258;
        if ( v254 || v258 || byte_4FAFBA0 )
        {
          v300 = v257;
          if ( (unsigned __int8)sub_1949C30(v91, v309, v285, v94, v329, (__int64)v70, si128, a6, v338) )
          {
            if ( !v329 && !v300 )
            {
              v260 = sub_157EBA0(v334);
              sub_17CE510((__int64)&v442, v260, 0, 0, 0);
              LOWORD(v426) = 257;
              v280 = sub_156E1C0(v442.m128i_i64, v280, v282, (__int64)&v424, 0, 0);
              sub_17CD270(v442.m128i_i64);
            }
LABEL_98:
            v297 = *(_QWORD *)(v301 - 24LL * v329 - 24);
            v98 = sub_157EB90(v334);
            v99 = sub_1632FA0(v98);
            sub_194A780((__int64)&v442, v338, v99, (__int64)"irce");
            v100 = sub_157EBA0(v334);
            v292 = sub_38767A0(&v442, v91, v320, v100);
            sub_194A930((__int64)&v442);
            v442.m128i_i64[0] = (__int64)"indvar.start";
            LOWORD(v443) = 259;
            sub_164B780(v292, v442.m128i_i64);
            v101 = *v14;
            v340 = sub_146F1B0(*v14, v306);
            v102 = sub_146F1B0(*v14, v303);
            v346 = 0;
            v103 = sub_14806B0(v101, v102, v340, 0, 0);
            v104 = sub_157EBA0(v317);
            v374 = v376;
            v375 = 0x400000000LL;
            v105 = (void (__fastcall *)(__m128i *, __int64, __m128 *, void **))sub_1948EE0;
            if ( v313 )
              v105 = (void (__fastcall *)(__m128i *, __int64, __m128 *, void **))sub_19497C0;
            v341 = v104;
            v330 = v105;
            v372 = 0;
            v373 = 0;
            v370[3] = sub_16498A0(v104);
            v106 = *(_QWORD *)(v341 + 48);
            v371 = 0;
            v107 = *(_QWORD *)(v341 + 40);
            v370[0] = 0;
            v370[1] = v107;
            v370[4] = 0;
            v370[2] = v341 + 24;
            v442.m128i_i64[0] = v106;
            if ( v106 )
            {
              sub_1623A60((__int64)&v442, v106, 2);
              v370[0] = v442.m128i_i64[0];
              if ( v442.m128i_i64[0] )
                sub_1623210((__int64)&v442, (unsigned __int8 *)v442.m128i_i64[0], (__int64)v370);
            }
            v342 = (const __m128i *)((char *)v481 + 40 * (unsigned int)v482);
            if ( v481 != v342 )
            {
              v335 = v14;
              v108 = v481;
              v310 = v70;
              v109 = v103;
              v318 = v12;
              while ( 1 )
              {
                if ( *(_QWORD *)(v109 + 40) == 2 )
                {
                  v110 = (_QWORD *)*v335;
                  v111 = **(_QWORD **)(v109 + 32);
                  v112 = sub_13A5BC0((_QWORD *)v109, *v335);
                  if ( !*(_WORD *)(v112 + 24) )
                  {
                    v113 = v108->m128i_i64[1];
                    if ( !*(_WORD *)(v113 + 24) && v112 == v113 )
                      break;
                  }
                }
LABEL_105:
                v108 = (const __m128i *)((char *)v108 + 40);
                if ( v342 == v108 )
                {
                  v12 = v318;
                  v14 = v335;
                  v70 = v310;
                  goto LABEL_147;
                }
              }
              v286 = v108->m128i_i64[0];
              v114 = *(_DWORD *)(sub_1456040(**(_QWORD **)(v109 + 32)) + 8) >> 8;
              v442.m128i_i32[2] = v114;
              if ( v114 > 0x40 )
              {
                v284 = v114;
                sub_16A4EF0((__int64)&v442, -1, 1);
                v115 = ~(1LL << ((unsigned __int8)v284 - 1));
                if ( v442.m128i_i32[2] > 0x40u )
                {
                  *(_QWORD *)(v442.m128i_i64[0] + 8LL * ((v284 - 1) >> 6)) &= v115;
LABEL_113:
                  v271 = sub_145CF40((__int64)v110, (__int64)&v442);
                  if ( v442.m128i_i32[2] > 0x40u && v442.m128i_i64[0] )
                    j_j___libc_free_0_0(v442.m128i_i64[0]);
                  v273 = sub_14806B0((__int64)v110, v286, v111, 0, 0);
                  v116 = sub_1456040(v273);
                  v117 = sub_145CF80((__int64)v110, v116, 0, 0);
                  v118 = v108[1].m128i_i64[0];
                  v277 = v117;
                  v272 = *(_QWORD *)(v109 + 48);
                  v119 = sub_1456040(v118);
                  v283 = sub_145CF80((__int64)v110, v119, 1, 0);
                  if ( !(unsigned __int8)sub_1948D70(v118, v272, (__int64)v110) )
                  {
                    v120 = sub_1456040(v118);
                    v270 = sub_145CF80((__int64)v110, v120, 0, 0);
                    if ( sub_146D950((__int64)v110, v118, v272)
                      && (unsigned __int8)sub_148B410((__int64)v110, v272, 0x28u, v118, v270) )
                    {
                      v283 = v277;
                    }
                    else
                    {
                      v287 = sub_1480620((__int64)v110, v283, 0);
                      v121 = sub_1480950(v110, v118, v277, si128, a6);
                      v443 = (const char *)sub_147A9C0(v110, v121, v287, si128, a6);
                      v442.m128i_i64[0] = (__int64)&v443;
                      v444 = v283;
                      v442.m128i_i64[1] = 0x200000002LL;
                      v283 = (__int64)sub_147DD40((__int64)v110, v442.m128i_i64, 0, 0, si128, a6);
                      if ( (const char **)v442.m128i_i64[0] != &v443 )
                        _libc_free(v442.m128i_u64[0]);
                    }
                  }
                  if ( v313 )
                  {
                    v122 = sub_14806B0((__int64)v110, v277, v271, 0, 0);
                    v123 = sub_147A9C0(v110, v273, v122, si128, a6);
                    v124 = sub_14806B0((__int64)v110, v277, v123, 4, 0);
                  }
                  else
                  {
                    v219 = sub_1480950(v110, v277, v273, si128, a6);
                    v124 = sub_14806B0((__int64)v110, v277, v219, 2, 0);
                  }
                  v443 = (const char *)v124;
                  v442.m128i_i64[0] = (__int64)&v443;
                  v444 = v283;
                  v442.m128i_i64[1] = 0x200000002LL;
                  v125 = sub_147EE30(v110, (__int64 **)&v442, 0, 0, si128, a6);
                  v126 = v125;
                  if ( (const char **)v442.m128i_i64[0] != &v443 )
                  {
                    v278 = v125;
                    _libc_free(v442.m128i_u64[0]);
                    v126 = v278;
                  }
                  v279 = (void *)v126;
                  if ( v313 )
                  {
                    v127 = sub_14806B0((__int64)v110, v118, v271, 0, 0);
                    v128 = sub_147A9C0(v110, v273, v127, si128, a6);
                    v129 = sub_14806B0((__int64)v110, v118, v128, 4, 0);
                  }
                  else
                  {
                    v218 = sub_1480950(v110, v118, v273, si128, a6);
                    v129 = sub_14806B0((__int64)v110, v118, v218, 2, 0);
                  }
                  v443 = (const char *)v129;
                  v442.m128i_i64[0] = (__int64)&v443;
                  v444 = v283;
                  v442.m128i_i64[1] = 0x200000002LL;
                  v130 = sub_147EE30(v110, (__int64 **)&v442, 0, 0, si128, a6);
                  v131 = v279;
                  v132 = v130;
                  if ( (const char **)v442.m128i_i64[0] != &v443 )
                  {
                    _libc_free(v442.m128i_u64[0]);
                    v131 = v279;
                  }
                  LOBYTE(v426) = 1;
                  v424 = v131;
                  v133 = *v335;
                  v425 = v132;
                  v330(&v442, v133, &v345, &v424);
                  if ( (_BYTE)v443 )
                  {
                    v136 = (unsigned int)v375;
                    if ( (unsigned int)v375 >= HIDWORD(v375) )
                    {
                      sub_16CD150((__int64)&v374, v376, 0, 40, v134, v135);
                      v136 = (unsigned int)v375;
                    }
                    a6 = _mm_loadu_si128(v108);
                    v137 = (__m128 *)&v374[40 * v136];
                    *v137 = (__m128)a6;
                    a7 = _mm_loadu_si128(v108 + 1);
                    LODWORD(v375) = v375 + 1;
                    v137[1] = (__m128)a7;
                    v93 = v346 == 0;
                    v137[2].m128_u64[0] = v108[2].m128i_u64[0];
                    if ( v93 )
                    {
                      v202 = _mm_load_si128(&v442);
                      v346 = 1;
                      v345 = (__m128)v202;
                    }
                    else
                    {
                      a8 = _mm_load_si128(&v442);
                      v345 = (__m128)a8;
                    }
                  }
                  goto LABEL_105;
                }
              }
              else
              {
                v442.m128i_i64[0] = 0xFFFFFFFFFFFFFFFFLL >> -(char)v114;
                v115 = ~(1LL << ((unsigned __int8)v114 - 1));
              }
              v442.m128i_i64[0] &= v115;
              goto LABEL_113;
            }
LABEL_147:
            if ( v346 )
            {
              v138 = (__int64 *)v70[4];
              v139 = v14[3];
              v140 = *v14;
              v331 = v14[2];
              v141 = v345.m128_u64[1];
              v343 = v345.m128_u64[0];
              v377 = *(_QWORD *)(*v138 + 56);
              v378 = sub_157E9C0(*v138);
              v380 = v331;
              v382 = a3;
              v388 = v343;
              v383 = a4;
              v390 = "main";
              v392 = v289;
              v391 = v281;
              v381 = v139;
              v393 = v301;
              v389 = v141;
              v394 = v297;
              v379 = v140;
              v395 = v275;
              v384 = v70;
              v396 = (__int64 *)v303;
              v385 = 0;
              v386 = 0;
              v387 = 0;
              v397 = v292;
              v398 = v306;
              v399 = v280;
              v400 = v274;
              v401 = v313;
              v385 = sub_1474160(v140, (__int64)v70, v289);
              v142 = sub_13FC520((__int64)v384);
              v143 = v401;
              v323 = v142;
              v386 = v142;
              v387 = v142;
              sub_1949EA0((__int64)&v347, (__int64)&v377, v401, si128, a6);
              v344 = v351;
              if ( v351 )
              {
                v307 = v348;
                v144 = 0;
                if ( v348 )
                  v144 = v347;
                v302 = v144;
                v325 = v350;
                v145 = 0;
                if ( v350 )
                  v145 = v349;
                v304 = v145;
                v319 = v400;
                v336 = *v396;
                v146 = sub_1632FA0(*(_QWORD *)(v377 + 40));
                v443 = "irce";
                v461 = v465;
                v462 = v465;
                v472 = 1;
                v442.m128i_i64[0] = v379;
                v442.m128i_i64[1] = v146;
                v444 = 0;
                v445 = 0;
                v446 = 0;
                v447 = 0;
                v448 = 0;
                v449 = 0;
                v450 = 0;
                v451 = 0;
                v452 = 0;
                v453 = 0;
                v454 = 0;
                v455 = 0;
                v456 = 0;
                v457 = 0;
                v458 = 0;
                v459 = 0;
                v460 = 0;
                v463 = 2;
                v464 = 0;
                v466 = 0;
                v467 = 0;
                v468 = 0;
                v469 = 0;
                v470 = 0;
                v471 = 0;
                v147 = sub_15E0530(*(_QWORD *)(v379 + 24));
                memset(v473, 0, 24);
                v473[3] = v147;
                v478 = v480;
                v473[4] = 0;
                v474 = 0;
                v475 = 0;
                v476 = 0;
                v477 = v146;
                v479 = 0x800000000LL;
                v148 = sub_157EBA0(v386);
                v402 = 0;
                v311 = v148;
                v403 = 0;
                v404 = 0;
                v405 = 0;
                v408 = 128;
                v149 = (_QWORD *)sub_22077B0(0x2000);
                v407 = 0;
                v406 = v149;
                v425 = 2;
                v428 = 0;
                for ( i = &v149[8 * (unsigned __int64)v408]; i != v149; v149 += 8 )
                {
                  if ( v149 )
                  {
                    v151 = v425;
                    v149[2] = 0;
                    v149[3] = -8;
                    *v149 = &unk_49E6B50;
                    v149[1] = v151 & 6;
                    v149[4] = v428;
                  }
                }
                v411 = 0;
                v412 = 1;
                v413 = byte_3F871B3;
                v414 = 0;
                v415 = 0;
                v416 = 0;
                v417 = 0;
                v418 = -1;
                v419 = 0;
                v420 = 0;
                v421 = 0;
                v422 = 0;
                v423 = 256;
                v424 = 0;
                v425 = 0;
                v426 = 0;
                v427 = 0;
                v430 = 128;
                v152 = (_QWORD *)sub_22077B0(0x2000);
                v429 = 0;
                v428 = v152;
                v365 = 2;
                v153 = &v152[8 * (unsigned __int64)v430];
                v364 = (const char *)&unk_49E6B50;
                v366 = 0;
                v367 = -8;
                for ( j = 0; v153 != v152; v152 += 8 )
                {
                  if ( v152 )
                  {
                    v154 = v365;
                    v152[2] = 0;
                    v152[3] = -8;
                    *v152 = &unk_49E6B50;
                    v152[1] = v154 & 6;
                    v152[4] = j;
                  }
                }
                v433 = 0;
                v434 = 1;
                v435[0] = byte_3F871B3;
                memset(&v435[1], 0, 32);
                v436 = -1;
                v437 = 0;
                v438 = 0;
                v439 = 0;
                v440 = 0;
                v441 = 256;
                if ( v319 )
                {
                  v155 = sub_145CF80(v379, v336, -1, 1u);
                  v156 = v302;
                  v298 = v155;
                  if ( !v307 )
                  {
                    if ( v325 )
                    {
                      if ( (unsigned __int8)sub_3870CB0(v304, v311, v379) )
                      {
                        v157 = sub_38767A0(&v442, v304, v336, v311);
                        LOWORD(v366) = 259;
                        v364 = "exit.mainloop.at";
                        sub_164B780(v157, (__int64 *)&v364);
LABEL_166:
                        sub_194C320(&v377, (__int64)&v424, "postloop");
                        v352 = 0;
                        v353 = 0;
                        v354 = 0;
                        v355 = 0;
                        v356 = 0;
                        v357 = 0;
                        v358 = 0;
                        v359 = 0;
                        v360 = 0;
                        v361 = 0;
                        v362 = 0;
                        v363 = 0;
                        goto LABEL_167;
                      }
LABEL_275:
                      v344 = 0;
LABEL_192:
                      if ( v433 )
                      {
                        if ( v432 )
                        {
                          v215 = v431;
                          v216 = &v431[2 * v432];
                          do
                          {
                            if ( *v215 != -8 && *v215 != -4 )
                            {
                              v217 = v215[1];
                              if ( v217 )
                                sub_161E7C0((__int64)(v215 + 1), v217);
                            }
                            v215 += 2;
                          }
                          while ( v216 != v215 );
                        }
                        j___libc_free_0(v431);
                      }
                      if ( v430 )
                      {
                        v179 = v428;
                        v353 = 2;
                        v354 = 0;
                        v355 = -8;
                        v352 = (const char *)&unk_49E6B50;
                        v180 = -8;
                        v356 = 0;
                        v181 = &v428[8 * (unsigned __int64)v430];
                        v359 = 2;
                        v360 = 0;
                        v361 = -16;
                        v358 = (const char *)&unk_49E6B50;
                        v362 = 0;
                        while ( 1 )
                        {
                          v182 = v179[3];
                          if ( v180 != v182 )
                          {
                            v180 = v361;
                            if ( v182 != v361 )
                            {
                              v183 = v179[7];
                              if ( v183 != 0 && v183 != -8 && v183 != -16 )
                              {
                                sub_1649B30(v179 + 5);
                                v182 = v179[3];
                              }
                              v180 = v182;
                            }
                          }
                          *v179 = &unk_49EE2B0;
                          if ( v180 != 0 && v180 != -8 && v180 != -16 )
                            sub_1649B30(v179 + 1);
                          v179 += 8;
                          if ( v181 == v179 )
                            break;
                          v180 = v355;
                        }
                        v358 = (const char *)&unk_49EE2B0;
                        if ( v361 != -8 && v361 != 0 && v361 != -16 )
                          sub_1649B30(&v359);
                        v352 = (const char *)&unk_49EE2B0;
                        if ( v355 != -8 && v355 != 0 && v355 != -16 )
                          sub_1649B30(&v353);
                      }
                      j___libc_free_0(v428);
                      if ( v424 )
                        j_j___libc_free_0(v424, v426 - (_QWORD)v424);
                      if ( v411 )
                      {
                        if ( v410 )
                        {
                          v212 = v409;
                          v213 = &v409[2 * v410];
                          do
                          {
                            if ( *v212 != -4 && *v212 != -8 )
                            {
                              v214 = v212[1];
                              if ( v214 )
                                sub_161E7C0((__int64)(v212 + 1), v214);
                            }
                            v212 += 2;
                          }
                          while ( v213 != v212 );
                        }
                        j___libc_free_0(v409);
                      }
                      if ( v408 )
                      {
                        v184 = v406;
                        v359 = 2;
                        v360 = 0;
                        v361 = -8;
                        v358 = (const char *)&unk_49E6B50;
                        v185 = -8;
                        v362 = 0;
                        v186 = &v406[8 * (unsigned __int64)v408];
                        v425 = 2;
                        v426 = 0;
                        v427 = -16;
                        v424 = &unk_49E6B50;
                        v428 = 0;
                        while ( 1 )
                        {
                          v187 = v184[3];
                          if ( v185 != v187 )
                          {
                            v185 = v427;
                            if ( v187 != v427 )
                            {
                              v188 = v184[7];
                              if ( v188 != 0 && v188 != -8 && v188 != -16 )
                              {
                                sub_1649B30(v184 + 5);
                                v187 = v184[3];
                              }
                              v185 = v187;
                            }
                          }
                          *v184 = &unk_49EE2B0;
                          if ( v185 != 0 && v185 != -8 && v185 != -16 )
                            sub_1649B30(v184 + 1);
                          v184 += 8;
                          if ( v186 == v184 )
                            break;
                          v185 = v361;
                        }
                        v424 = &unk_49EE2B0;
                        if ( v427 != 0 && v427 != -8 && v427 != -16 )
                          sub_1649B30(&v425);
                        v358 = (const char *)&unk_49EE2B0;
                        if ( v361 != -8 && v361 != 0 && v361 != -16 )
                          sub_1649B30(&v359);
                      }
                      j___libc_free_0(v406);
                      if ( v402 )
                        j_j___libc_free_0(v402, v404 - v402);
                      if ( v478 != v480 )
                        _libc_free((unsigned __int64)v478);
                      v189 = (const char *)v473[0];
                      if ( v473[0] )
                        sub_161E7C0((__int64)v473, v473[0]);
                      j___libc_free_0(v469);
                      if ( v462 != v461 )
                        _libc_free((unsigned __int64)v462);
                      j___libc_free_0(v457);
                      j___libc_free_0(v453);
                      j___libc_free_0(v449);
                      if ( v447 )
                      {
                        v190 = v445;
                        v191 = &v445[5 * v447];
                        do
                        {
                          if ( *v190 == -8 )
                          {
                            if ( v190[1] != -8 )
                              goto LABEL_245;
                          }
                          else if ( *v190 != -16 || v190[1] != -16 )
                          {
LABEL_245:
                            v192 = v190[4];
                            if ( v192 != 0 && v192 != -8 && v192 != -16 )
                              sub_1649B30(v190 + 2);
                          }
                          v190 += 5;
                        }
                        while ( v191 != v190 );
                      }
                      v193 = (__int64)v445;
                      j___libc_free_0(v445);
                      if ( v344 )
                      {
                        if ( byte_4FAFF20 )
                        {
                          v220 = sub_16BA580(v193, (__int64)v189, v194);
                          v221 = *(_QWORD *)(v220 + 24);
                          v222 = v220;
                          if ( (unsigned __int64)(*(_QWORD *)(v220 + 16) - v221) <= 0x11 )
                          {
                            v189 = "irce: in function ";
                            sub_16E7EE0(v220, "irce: in function ", 0x12u);
                          }
                          else
                          {
                            v223 = _mm_load_si128((const __m128i *)&xmmword_42BEB80);
                            *(_WORD *)(v221 + 16) = 8302;
                            *(__m128i *)v221 = v223;
                            *(_QWORD *)(v220 + 24) += 18LL;
                          }
                          v224 = sub_16BA580(v222, (__int64)v189, v221);
                          v225 = sub_1649960(*(_QWORD *)(*(_QWORD *)v70[4] + 56LL));
                          v227 = *(_WORD **)(v224 + 24);
                          v228 = (char *)v225;
                          v229 = v226;
                          v230 = *(_QWORD *)(v224 + 16) - (_QWORD)v227;
                          if ( v226 > v230 )
                          {
                            v237 = sub_16E7EE0(v224, v228, v226);
                            v227 = *(_WORD **)(v237 + 24);
                            v224 = v237;
                            v230 = *(_QWORD *)(v237 + 16) - (_QWORD)v227;
                          }
                          else if ( v226 )
                          {
                            memcpy(v227, v228, v226);
                            v238 = *(_QWORD *)(v224 + 16);
                            v227 = (_WORD *)(v229 + *(_QWORD *)(v224 + 24));
                            *(_QWORD *)(v224 + 24) = v227;
                            v230 = v238 - (_QWORD)v227;
                          }
                          if ( v230 <= 1 )
                          {
                            v231 = (__int64)": ";
                            v227 = (_WORD *)v224;
                            sub_16E7EE0(v224, ": ", 2u);
                          }
                          else
                          {
                            v231 = 8250;
                            *v227 = 8250;
                            *(_QWORD *)(v224 + 24) += 2LL;
                          }
                          v232 = sub_16BA580((__int64)v227, v231, v226);
                          v233 = *(void **)(v232 + 24);
                          v234 = v232;
                          if ( *(_QWORD *)(v232 + 16) - (_QWORD)v233 <= 0xBu )
                          {
                            v231 = (__int64)"constrained ";
                            sub_16E7EE0(v232, "constrained ", 0xCu);
                          }
                          else
                          {
                            qmemcpy(v233, "constrained ", 12);
                            *(_QWORD *)(v232 + 24) += 12LL;
                          }
                          v235 = sub_16BA580(v234, v231, (__int64)v233);
                          sub_13FA7B0((_QWORD **)v70, v235, 0, 0);
                        }
                        v195 = (unsigned __int64)v374;
                        v196 = &v374[40 * (unsigned int)v375];
                        if ( v374 != v196 )
                        {
                          do
                          {
                            v197 = sub_159C4F0(v316);
                            v198 = *(__int64 **)(v195 + 24);
                            if ( *v198 )
                            {
                              v199 = v198[1];
                              v200 = v198[2] & 0xFFFFFFFFFFFFFFFCLL;
                              *(_QWORD *)v200 = v199;
                              if ( v199 )
                                *(_QWORD *)(v199 + 16) = *(_QWORD *)(v199 + 16) & 3LL | v200;
                            }
                            *v198 = v197;
                            if ( v197 )
                            {
                              v201 = *(_QWORD *)(v197 + 8);
                              v198[1] = v201;
                              if ( v201 )
                                *(_QWORD *)(v201 + 16) = (unsigned __int64)(v198 + 1) | *(_QWORD *)(v201 + 16) & 3LL;
                              v198[2] = (v197 + 8) | v198[2] & 3;
                              *(_QWORD *)(v197 + 8) = v198;
                            }
                            v195 += 40LL;
                          }
                          while ( v196 != (_BYTE *)v195 );
                        }
                        v12 = v344;
                      }
                      goto LABEL_264;
                    }
                    goto LABEL_337;
                  }
                }
                else
                {
                  v298 = sub_145CF80(v379, v336, -1, 1u);
                  if ( !v325 )
                  {
                    v293 = 0;
                    if ( !v307 )
                    {
LABEL_337:
                      v352 = 0;
                      v353 = 0;
                      v354 = 0;
                      v355 = 0;
                      v356 = 0;
                      v357 = 0;
                      v358 = 0;
                      v359 = 0;
                      v360 = 0;
                      v361 = 0;
                      v362 = 0;
                      v363 = 0;
                      goto LABEL_289;
                    }
                    goto LABEL_329;
                  }
                  if ( !sub_1949540(v304, (__int64)v384, v379, v143) )
                    goto LABEL_275;
                  v366 = v304;
                  v364 = (const char *)&v366;
                  v367 = v298;
                  v365 = 0x200000002LL;
                  v203 = sub_147DD40(v379, (__int64 *)&v364, 0, 0, si128, a6);
                  v156 = (__int64)v203;
                  if ( v364 != (const char *)&v366 )
                  {
                    v326 = v203;
                    _libc_free((unsigned __int64)v364);
                    v156 = (__int64)v326;
                  }
                  v325 = v307;
                }
                v308 = v156;
                if ( !(unsigned __int8)sub_3870CB0(v156, v311, v379) )
                  goto LABEL_275;
                v204 = sub_38767A0(&v442, v308, v336, v311);
                v364 = "exit.preloop.at";
                v293 = v204;
                LOWORD(v366) = 259;
                sub_164B780(v204, (__int64 *)&v364);
                if ( !v325 )
                {
                  v157 = 0;
                  sub_194C320(&v377, (__int64)&v402, "preloop");
                  v352 = 0;
                  v353 = 0;
                  v354 = 0;
                  v355 = 0;
                  v356 = 0;
                  v357 = 0;
                  goto LABEL_285;
                }
                if ( v319 )
                {
                  if ( !(unsigned __int8)sub_3870CB0(v304, v311, v379) )
                    goto LABEL_275;
                  v157 = sub_38767A0(&v442, v304, v336, v311);
                  LOWORD(v366) = 259;
                  v364 = "exit.mainloop.at";
                  sub_164B780(v157, (__int64 *)&v364);
                  goto LABEL_334;
                }
LABEL_329:
                if ( !sub_1949540(v302, (__int64)v384, v379, v143) )
                  goto LABEL_275;
                v366 = v302;
                v367 = v298;
                v364 = (const char *)&v366;
                v365 = 0x200000002LL;
                v236 = sub_147DD40(v379, (__int64 *)&v364, 0, 0, si128, a6);
                if ( v364 != (const char *)&v366 )
                  _libc_free((unsigned __int64)v364);
                if ( !(unsigned __int8)sub_3870CB0(v236, v311, v379) )
                  goto LABEL_275;
                v157 = sub_38767A0(&v442, v236, v336, v311);
                LOWORD(v366) = 259;
                v364 = "exit.mainloop.at";
                sub_164B780(v157, (__int64 *)&v364);
                if ( !v325 )
                  goto LABEL_166;
LABEL_334:
                sub_194C320(&v377, (__int64)&v402, "preloop");
                sub_194C320(&v377, (__int64)&v424, "postloop");
                v352 = 0;
                v353 = 0;
                v354 = 0;
                v355 = 0;
                v356 = 0;
                v357 = 0;
                v325 = v344;
LABEL_285:
                v205 = sub_157EBA0(v323);
                sub_1648780(v205, v391, v414);
                v387 = sub_1949170(v377, v378, (__int64)&v390, v323, "mainloop");
                sub_194AE30(
                  (__int64 *)&v364,
                  &v377,
                  (__int64)&v413,
                  v323,
                  v293,
                  v387,
                  *(double *)si128.m128i_i64,
                  *(double *)a6.m128i_i64,
                  *(double *)a7.m128i_i64);
                v206 = v354;
                v207 = v356;
                v352 = v364;
                v353 = v365;
                v208 = v366;
                v366 = 0;
                v354 = v208;
                v209 = v367;
                v367 = 0;
                v355 = v209;
                v210 = j;
                j = 0;
                v356 = v210;
                if ( v206 )
                {
                  j_j___libc_free_0(v206, v207 - v206);
                  v357 = v369;
                  if ( v366 )
                    j_j___libc_free_0(v366, j - v366);
                }
                else
                {
                  v357 = v369;
                }
                sub_1949270((__int64)&v390, v387, (__int64)&v352);
                v358 = 0;
                v359 = 0;
                v360 = 0;
                v361 = 0;
                v362 = 0;
                v363 = 0;
                if ( v325 )
                {
LABEL_167:
                  v158 = sub_1949170(v377, v378, (__int64)v435, v323, "postloop");
                  sub_194AE30(
                    (__int64 *)&v364,
                    &v377,
                    (__int64)&v390,
                    v387,
                    v157,
                    v158,
                    *(double *)si128.m128i_i64,
                    *(double *)a6.m128i_i64,
                    *(double *)a7.m128i_i64);
                  v159 = v360;
                  v160 = v362;
                  v358 = v364;
                  v359 = v365;
                  v161 = v366;
                  v366 = 0;
                  v360 = v161;
                  v162 = v367;
                  v367 = 0;
                  v361 = v162;
                  v163 = j;
                  j = 0;
                  v362 = v163;
                  if ( v159 )
                  {
                    j_j___libc_free_0(v159, v160 - v159);
                    v363 = v369;
                    if ( v366 )
                      j_j___libc_free_0(v366, j - v366);
                  }
                  else
                  {
                    v363 = v369;
                  }
                  sub_1949270((__int64)v435, v158, (__int64)&v358);
                  v164 = v387;
                  if ( v323 == v387 )
                    v164 = 0;
                  v364 = (const char *)v158;
                  v367 = (__int64)v358;
                  v365 = (__int64)v352;
                  v366 = v353;
                  j = v359;
                  v369 = v164;
                  if ( v158 )
                  {
                    if ( v352 )
                    {
                      if ( v353 )
                      {
                        if ( v358 )
                        {
                          if ( v359 )
                          {
                            p_j = &v369;
                            if ( v164 )
                              p_j = v370;
                            v166 = *v384;
                            if ( *v384 )
                            {
LABEL_180:
                              v167 = (__int64 *)&v364;
                              do
                              {
                                v168 = *v167++;
                                sub_1400330(v166, v168, v381);
                              }
                              while ( p_j != v167 );
                            }
LABEL_182:
                            v169 = v380;
                            *(_QWORD *)(v380 + 64) = v377;
                            sub_15D3930(v169);
                            if ( v403 == v402 )
                            {
                              if ( v424 != (void *)v425 )
                              {
                                v171 = sub_194CE70((__int64)&v377, (__int64)v384, *v384, (__int64)&v427, 0);
LABEL_186:
                                if ( v171 )
                                {
                                  sub_1AE5120(v171, v380, v381, v379);
                                  v175 = sub_1AFB400(v171, v380, v381, v379, 0, 1);
                                  sub_1948FD0(
                                    (__int64)v171,
                                    v175,
                                    *(double *)a6.m128i_i64,
                                    *(double *)a7.m128i_i64,
                                    *(double *)a8.m128i_i64,
                                    v176,
                                    v177,
                                    a11,
                                    a12);
                                }
                              }
                            }
                            else
                            {
                              v170 = sub_194CE70((__int64)&v377, (__int64)v384, *v384, (__int64)&v405, 0);
                              if ( (void *)v425 != v424 )
                              {
                                v171 = sub_194CE70((__int64)&v377, (__int64)v384, *v384, (__int64)&v427, 0);
                                if ( !v170 )
                                  goto LABEL_186;
                                goto LABEL_185;
                              }
                              v171 = 0;
                              if ( v170 )
                              {
LABEL_185:
                                sub_1AE5120(v170, v380, v381, v379);
                                v172 = sub_1AFB400(v170, v380, v381, v379, 0, 1);
                                sub_1948FD0(
                                  (__int64)v170,
                                  v172,
                                  *(double *)a6.m128i_i64,
                                  *(double *)a7.m128i_i64,
                                  *(double *)a8.m128i_i64,
                                  v173,
                                  v174,
                                  a11,
                                  a12);
                                goto LABEL_186;
                              }
                            }
                            v178 = v384;
                            sub_1AE5120(v384, v380, v381, v379);
                            sub_1AFB400(v178, v380, v381, v379, 0, 1);
                            if ( v360 )
                              j_j___libc_free_0(v360, v362 - v360);
                            if ( v354 )
                              j_j___libc_free_0(v354, v356 - v354);
                            goto LABEL_192;
                          }
                          p_j = &j;
                        }
                        else
                        {
                          p_j = &v367;
                        }
                      }
                      else
                      {
                        p_j = &v366;
                      }
                    }
                    else
                    {
                      p_j = &v365;
                    }
LABEL_291:
                    v211 = p_j + 1;
                    do
                    {
                      if ( *v211 )
                        *p_j++ = *v211;
                      ++v211;
                    }
                    while ( v211 != v370 );
                    v166 = *v384;
                    if ( *v384 && p_j != (__int64 *)&v364 )
                      goto LABEL_180;
                    goto LABEL_182;
                  }
                  goto LABEL_349;
                }
LABEL_289:
                if ( v323 == v387 )
                {
                  v364 = 0;
                  v367 = 0;
                  p_j = (__int64 *)&v364;
                  v365 = (__int64)v352;
                  j = 0;
                  v366 = v353;
                  v369 = 0;
                  goto LABEL_291;
                }
                v364 = 0;
                v367 = 0;
                v365 = (__int64)v352;
                j = 0;
                v366 = v353;
                v369 = v387;
LABEL_349:
                p_j = (__int64 *)&v364;
                goto LABEL_291;
              }
              v12 = 0;
            }
LABEL_264:
            if ( v370[0] )
              sub_161E7C0((__int64)v370, v370[0]);
            if ( v374 != v376 )
              _libc_free((unsigned __int64)v374);
            goto LABEL_21;
          }
        }
LABEL_20:
        v12 = 0;
LABEL_21:
        v25 = v481;
        goto LABEL_22;
      }
      v259 = ((v94 - 36) & 0xFFFFFFFB) == 0;
    }
    if ( v281 == v290 || !v259 )
      goto LABEL_20;
    goto LABEL_394;
  }
  v25 = v481;
  v12 = 0;
LABEL_22:
  if ( v25 != (const __m128i *)v483 )
    _libc_free((unsigned __int64)v25);
  return v12;
}
