// Function: sub_2DC4260
// Address: 0x2dc4260
//
__int64 __fastcall sub_2DC4260(
        __int64 a1,
        __int64 p_src,
        __int64 *a3,
        __int64 *a4,
        __int64 a5,
        __int64 *a6,
        unsigned __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  unsigned __int64 *v12; // rax
  _QWORD *v13; // r15
  __int64 j; // rbx
  __int64 v15; // rdi
  unsigned int v17; // ebx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int8 v20; // r12
  char v21; // bl
  _QWORD *v22; // r12
  _QWORD *v23; // r13
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  _QWORD *v33; // r12
  _QWORD *v34; // rbx
  unsigned __int64 v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  _QWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // r9
  unsigned int *v45; // r13
  __int64 v46; // rax
  unsigned int v47; // esi
  __int64 v48; // rcx
  unsigned __int64 v49; // r8
  unsigned int *v50; // r14
  unsigned __int64 v51; // r12
  unsigned __int64 v52; // rax
  _QWORD *v53; // r14
  __int64 v54; // r9
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // r10
  _BYTE *v58; // rdi
  __int64 v59; // rdx
  char *v60; // rdi
  __int64 v61; // r8
  char *v62; // rdx
  unsigned int v63; // ecx
  __int64 v64; // rsi
  char *v65; // r9
  __int64 v66; // r10
  unsigned __int64 v67; // rdx
  unsigned __int64 v68; // r13
  char *v69; // rdx
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rax
  _QWORD *v73; // rax
  __int64 v74; // rax
  __int64 v75; // r12
  __int64 v76; // rax
  unsigned __int64 v77; // rbx
  unsigned __int8 v78; // dl
  unsigned int v79; // ebx
  _BYTE *v80; // rsi
  unsigned int v81; // eax
  __int64 v82; // r15
  __int64 v83; // r14
  __int64 v84; // rax
  void *v85; // r12
  __int64 v86; // r10
  _QWORD *v87; // r12
  _QWORD *v88; // rbx
  unsigned __int64 v89; // rsi
  _QWORD *v90; // rax
  _QWORD *v91; // rdi
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // rax
  _QWORD *v95; // rdi
  __int64 v96; // rcx
  __int64 v97; // rdx
  bool v98; // zf
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // r9
  unsigned __int64 v103; // rbx
  _QWORD *v104; // r14
  void (__fastcall *v105)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v106; // rax
  unsigned __int64 v107; // r15
  unsigned __int64 v108; // r13
  unsigned __int64 v109; // rbx
  unsigned __int64 *v110; // rdx
  unsigned __int64 v111; // rax
  unsigned __int64 v112; // rax
  int v113; // edx
  unsigned __int8 *v114; // rdi
  unsigned __int8 *v115; // rax
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 v118; // rax
  __int64 v119; // r9
  __int64 v120; // r12
  unsigned int *v121; // rax
  int v122; // ecx
  unsigned int *v123; // rdx
  unsigned __int8 v124; // r9
  __int64 v125; // rbx
  unsigned int v126; // r13d
  unsigned __int64 i; // rax
  __int64 v128; // rax
  __int64 v129; // r14
  __int64 v130; // rbx
  __int64 v131; // r15
  _QWORD *v132; // rax
  __int64 v133; // r9
  __int64 v134; // r12
  __int64 v135; // r8
  __int64 v136; // r9
  unsigned int *v137; // r15
  unsigned int *v138; // rbx
  __int64 v139; // rdx
  unsigned __int64 v140; // r8
  unsigned __int64 *v141; // rdi
  _QWORD *v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rbx
  __int64 v146; // r12
  __int64 v147; // r14
  int v148; // eax
  int v149; // eax
  unsigned int v150; // edx
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 v153; // rdx
  unsigned __int64 v154; // r15
  unsigned int *v155; // rbx
  int v156; // r12d
  _QWORD *v157; // rax
  __int64 **v158; // r13
  __int64 **v159; // r12
  __int64 v160; // r13
  unsigned __int64 v161; // rax
  _QWORD *v162; // rax
  __int64 v163; // rax
  int v164; // r14d
  __int64 v165; // rdx
  unsigned __int64 v166; // rax
  _QWORD *v167; // rax
  __int64 **v168; // rcx
  unsigned __int64 v169; // rax
  __int64 v170; // rdx
  __int64 v171; // r12
  __int64 v172; // rcx
  __int64 v173; // rbx
  __int64 v174; // r14
  _QWORD *v175; // rax
  __int64 v176; // r9
  __int64 v177; // r12
  __int64 v178; // rcx
  __int64 v179; // r8
  __int64 v180; // r9
  unsigned int *v181; // r14
  unsigned int *v182; // rbx
  __int64 v183; // rdx
  __int64 v184; // rbx
  __int64 v185; // r13
  int v186; // eax
  int v187; // eax
  unsigned int v188; // edx
  __int64 v189; // rax
  __int64 v190; // rdx
  __int64 v191; // rdx
  __int64 v192; // rbx
  __int64 v193; // r13
  int v194; // eax
  int v195; // eax
  unsigned int v196; // edx
  __int64 v197; // rax
  __int64 v198; // rdx
  __int64 v199; // rdx
  __int64 v200; // rbx
  char *v201; // r14
  _QWORD *v202; // rax
  __int64 **v203; // r12
  _QWORD *v204; // rax
  __int64 **v205; // rax
  _BYTE *v206; // rax
  _BYTE *v207; // rdx
  __int64 v208; // rax
  __int64 v209; // rbx
  __int64 v210; // r12
  int v211; // eax
  int v212; // eax
  unsigned int v213; // edx
  __int64 v214; // rax
  __int64 v215; // rdx
  __int64 v216; // rdx
  _BYTE *v217; // rax
  __int64 v218; // rax
  __int64 v219; // r15
  _QWORD *v220; // rax
  __int64 v221; // r9
  __int64 v222; // r12
  __int64 v223; // r8
  __int64 v224; // r9
  unsigned int *v225; // r15
  unsigned int *v226; // rbx
  __int64 v227; // rdx
  __int64 v228; // rax
  _QWORD *v229; // rax
  __int64 v230; // rax
  __int64 v231; // rax
  __int64 v232; // rbx
  __int64 v233; // r12
  __int64 v234; // r13
  int v235; // eax
  int v236; // eax
  unsigned int v237; // edx
  __int64 v238; // rax
  __int64 v239; // rdx
  __int64 v240; // rdx
  __int64 v241; // r15
  _QWORD *v242; // rax
  __int64 v243; // r12
  __int64 v244; // rcx
  __int64 v245; // r8
  __int64 v246; // r9
  unsigned int *v247; // r15
  unsigned int *v248; // rbx
  __int64 v249; // rdx
  unsigned __int64 v250; // rdx
  __int64 v251; // rcx
  unsigned __int64 v252; // rdi
  unsigned __int64 v253; // r15
  unsigned __int64 v254; // r14
  unsigned __int64 v255; // r13
  unsigned __int64 v256; // r12
  unsigned __int64 v257; // rax
  __int64 v258; // rdx
  unsigned __int64 v259; // r8
  unsigned __int64 v260; // rbx
  unsigned __int64 *v261; // rdx
  unsigned __int64 v262; // r13
  unsigned __int64 v263; // r8
  __int64 v264; // rsi
  unsigned __int64 v265; // rdi
  unsigned __int64 v266; // r10
  __int64 v267; // r8
  __int64 v268; // r9
  __int64 v269; // rcx
  __int64 *v270; // rax
  int v271; // r13d
  __int64 v272; // rdx
  __int64 v273; // rcx
  __int64 v274; // rdx
  __int64 v275; // rax
  unsigned __int64 v276; // r12
  _QWORD *v277; // rax
  __int64 **v278; // rbx
  __int64 (__fastcall *v279)(__int64, unsigned int, _BYTE *, __int64); // rax
  _QWORD *v280; // rax
  _QWORD *v281; // r10
  unsigned int *v282; // rbx
  unsigned int *v283; // r12
  __int64 v284; // rdx
  unsigned int v285; // r12d
  _QWORD *v286; // rax
  __int64 v287; // r12
  __int64 v288; // rax
  unsigned __int64 v289; // rdx
  unsigned int v290; // r12d
  __int64 **v291; // r13
  _QWORD *v292; // rax
  __int64 v293; // rax
  unsigned __int8 v294; // r9
  __int64 **v295; // r12
  int v296; // edx
  unsigned __int64 v297; // rax
  _QWORD *v298; // rax
  __int64 **v299; // rcx
  _BYTE *v300; // rdx
  char v301; // al
  unsigned __int8 v302; // r9
  __int64 v303; // r13
  _QWORD *v304; // r12
  char v305; // al
  unsigned __int64 v306; // rax
  _BOOL4 v307; // edx
  bool v308; // bl
  unsigned int v309; // ebx
  __int64 v310; // r13
  _QWORD *v311; // rax
  unsigned __int64 v312; // rax
  _QWORD *v313; // rax
  __int64 v314; // rax
  __int64 **v315; // rax
  _BYTE *v316; // rdx
  unsigned __int64 v317; // r8
  unsigned __int64 v318; // rbx
  unsigned __int8 *v319; // r8
  __int64 v320; // rdx
  int v321; // ebx
  unsigned int v322; // r13d
  __int64 v323; // rax
  unsigned int v324; // eax
  __int64 v325; // rax
  _BYTE *v326; // r14
  __int64 (__fastcall *v327)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v328; // rbx
  int v329; // eax
  _QWORD *v330; // rax
  unsigned int *v331; // r13
  unsigned int *v332; // r14
  __int64 v333; // rdx
  unsigned int v334; // esi
  unsigned __int8 *v335; // rax
  __int64 v336; // rdx
  unsigned int v337; // ebx
  bool v338; // al
  __int64 v339; // r13
  _BYTE *v340; // rax
  bool v341; // r9
  unsigned int v342; // ebx
  int v343; // eax
  unsigned int v344; // r13d
  bool v345; // bl
  __int64 v346; // rax
  unsigned int v347; // ebx
  unsigned int v348; // eax
  __int64 v349; // rdx
  unsigned int v350; // ebx
  __int64 v351; // rbx
  _BYTE *v352; // rax
  unsigned int v353; // ebx
  bool v354; // al
  int v355; // eax
  unsigned int v356; // ecx
  bool v357; // bl
  __int64 v358; // rax
  unsigned int v359; // ecx
  unsigned int v360; // ebx
  int v361; // eax
  __int64 v362; // [rsp-8h] [rbp-778h]
  unsigned __int64 v363; // [rsp+8h] [rbp-768h]
  __int64 v364; // [rsp+10h] [rbp-760h]
  unsigned __int64 v365; // [rsp+18h] [rbp-758h]
  __int64 v366; // [rsp+20h] [rbp-750h]
  __int64 v367; // [rsp+28h] [rbp-748h]
  unsigned __int64 v368; // [rsp+30h] [rbp-740h]
  _BYTE *v369; // [rsp+38h] [rbp-738h]
  _QWORD *v370; // [rsp+40h] [rbp-730h]
  _QWORD *v371; // [rsp+48h] [rbp-728h]
  unsigned __int64 v376; // [rsp+98h] [rbp-6D8h]
  char v377; // [rsp+A7h] [rbp-6C9h]
  unsigned int v378; // [rsp+B0h] [rbp-6C0h]
  unsigned __int64 v379; // [rsp+B8h] [rbp-6B8h]
  unsigned __int64 v380; // [rsp+B8h] [rbp-6B8h]
  _QWORD *v381; // [rsp+C0h] [rbp-6B0h]
  __int64 v382; // [rsp+C0h] [rbp-6B0h]
  unsigned __int64 v383; // [rsp+C8h] [rbp-6A8h]
  unsigned int v384; // [rsp+C8h] [rbp-6A8h]
  __int64 v385; // [rsp+D0h] [rbp-6A0h]
  unsigned __int8 v386; // [rsp+D8h] [rbp-698h]
  _QWORD *v387; // [rsp+D8h] [rbp-698h]
  _QWORD *v388; // [rsp+D8h] [rbp-698h]
  bool v389; // [rsp+D8h] [rbp-698h]
  unsigned int v390; // [rsp+D8h] [rbp-698h]
  __int64 v392; // [rsp+F0h] [rbp-680h]
  unsigned __int64 v393; // [rsp+F8h] [rbp-678h]
  signed __int64 v394; // [rsp+F8h] [rbp-678h]
  _QWORD *v395; // [rsp+F8h] [rbp-678h]
  _QWORD *v396; // [rsp+F8h] [rbp-678h]
  unsigned __int8 v397; // [rsp+F8h] [rbp-678h]
  bool v398; // [rsp+F8h] [rbp-678h]
  unsigned __int8 *v399; // [rsp+F8h] [rbp-678h]
  unsigned __int8 *v400; // [rsp+F8h] [rbp-678h]
  __int64 v401; // [rsp+100h] [rbp-670h]
  unsigned __int64 v402; // [rsp+108h] [rbp-668h]
  _QWORD *v403; // [rsp+108h] [rbp-668h]
  unsigned __int64 v404; // [rsp+108h] [rbp-668h]
  __int64 v405; // [rsp+108h] [rbp-668h]
  unsigned __int8 v406; // [rsp+108h] [rbp-668h]
  unsigned __int8 v407; // [rsp+108h] [rbp-668h]
  unsigned __int8 *v408; // [rsp+108h] [rbp-668h]
  unsigned __int8 *v409; // [rsp+108h] [rbp-668h]
  int v410; // [rsp+108h] [rbp-668h]
  __int64 v411; // [rsp+108h] [rbp-668h]
  int v412; // [rsp+108h] [rbp-668h]
  unsigned int v413; // [rsp+110h] [rbp-660h]
  __int64 v414; // [rsp+110h] [rbp-660h]
  _QWORD *v415; // [rsp+110h] [rbp-660h]
  _BYTE *v416; // [rsp+110h] [rbp-660h]
  __int64 v417; // [rsp+110h] [rbp-660h]
  unsigned int *v418; // [rsp+120h] [rbp-650h]
  __int64 v419; // [rsp+120h] [rbp-650h]
  _QWORD *v420; // [rsp+120h] [rbp-650h]
  unsigned __int8 v421; // [rsp+120h] [rbp-650h]
  void *v422; // [rsp+120h] [rbp-650h]
  unsigned __int64 v423; // [rsp+120h] [rbp-650h]
  size_t v424; // [rsp+120h] [rbp-650h]
  unsigned __int64 v425; // [rsp+128h] [rbp-648h]
  __int64 v426; // [rsp+128h] [rbp-648h]
  __int64 v427; // [rsp+128h] [rbp-648h]
  __int64 v428; // [rsp+128h] [rbp-648h]
  char v429; // [rsp+130h] [rbp-640h]
  __int64 v430; // [rsp+130h] [rbp-640h]
  __int64 v431; // [rsp+130h] [rbp-640h]
  __int64 v432; // [rsp+130h] [rbp-640h]
  unsigned int v433; // [rsp+130h] [rbp-640h]
  _QWORD *v434; // [rsp+130h] [rbp-640h]
  __int64 v435; // [rsp+130h] [rbp-640h]
  unsigned __int8 v436; // [rsp+130h] [rbp-640h]
  int v437; // [rsp+130h] [rbp-640h]
  unsigned __int8 v438; // [rsp+130h] [rbp-640h]
  _QWORD *v439; // [rsp+138h] [rbp-638h]
  unsigned __int8 v440; // [rsp+140h] [rbp-630h]
  __int64 **v441; // [rsp+140h] [rbp-630h]
  _QWORD *v442; // [rsp+148h] [rbp-628h]
  unsigned int v443; // [rsp+154h] [rbp-61Ch] BYREF
  __int64 v444; // [rsp+158h] [rbp-618h] BYREF
  unsigned int v445; // [rsp+160h] [rbp-610h] BYREF
  unsigned int *v446; // [rsp+168h] [rbp-608h]
  unsigned int v447; // [rsp+170h] [rbp-600h]
  char v448; // [rsp+178h] [rbp-5F8h] BYREF
  unsigned int v449; // [rsp+198h] [rbp-5D8h]
  char v450; // [rsp+19Ch] [rbp-5D4h]
  char *v451; // [rsp+1A0h] [rbp-5D0h]
  unsigned int v452; // [rsp+1A8h] [rbp-5C8h]
  char v453; // [rsp+1B0h] [rbp-5C0h] BYREF
  void *src; // [rsp+1C0h] [rbp-5B0h] BYREF
  __int64 v455; // [rsp+1C8h] [rbp-5A8h]
  _BYTE v456[16]; // [rsp+1D0h] [rbp-5A0h] BYREF
  __int16 v457; // [rsp+1E0h] [rbp-590h]
  __int64 v458; // [rsp+250h] [rbp-520h] BYREF
  unsigned __int64 v459; // [rsp+258h] [rbp-518h]
  char *v460; // [rsp+260h] [rbp-510h] BYREF
  unsigned __int64 v461; // [rsp+268h] [rbp-508h]
  __int16 v462; // [rsp+270h] [rbp-500h]
  __int64 v463; // [rsp+2E0h] [rbp-490h] BYREF
  unsigned __int64 v464; // [rsp+2E8h] [rbp-488h]
  __int64 v465; // [rsp+2F0h] [rbp-480h]
  __int64 v466; // [rsp+2F8h] [rbp-478h]
  __int64 v467; // [rsp+300h] [rbp-470h] BYREF
  unsigned int v468; // [rsp+308h] [rbp-468h]
  __int64 v469; // [rsp+310h] [rbp-460h] BYREF
  unsigned __int64 v470; // [rsp+318h] [rbp-458h]
  unsigned __int64 v471; // [rsp+320h] [rbp-450h] BYREF
  _BYTE *v472; // [rsp+328h] [rbp-448h]
  _BYTE *v473; // [rsp+330h] [rbp-440h] BYREF
  __int64 v474; // [rsp+338h] [rbp-438h]
  __int64 v475; // [rsp+340h] [rbp-430h]
  unsigned __int8 v476; // [rsp+348h] [rbp-428h]
  _BYTE *v477; // [rsp+350h] [rbp-420h]
  __int64 v478; // [rsp+358h] [rbp-418h]
  unsigned int *v479; // [rsp+360h] [rbp-410h] BYREF
  __int64 v480; // [rsp+368h] [rbp-408h]
  _BYTE v481[32]; // [rsp+370h] [rbp-400h] BYREF
  unsigned __int64 v482; // [rsp+390h] [rbp-3E0h]
  char *v483; // [rsp+398h] [rbp-3D8h]
  __int64 v484; // [rsp+3A0h] [rbp-3D0h]
  _QWORD *v485; // [rsp+3A8h] [rbp-3C8h]
  void **v486; // [rsp+3B0h] [rbp-3C0h]
  void **v487; // [rsp+3B8h] [rbp-3B8h]
  __int64 v488; // [rsp+3C0h] [rbp-3B0h]
  int v489; // [rsp+3C8h] [rbp-3A8h]
  __int16 v490; // [rsp+3CCh] [rbp-3A4h]
  char v491; // [rsp+3CEh] [rbp-3A2h]
  __int64 v492; // [rsp+3D0h] [rbp-3A0h]
  __int64 v493; // [rsp+3D8h] [rbp-398h]
  void *v494; // [rsp+3E0h] [rbp-390h] BYREF
  void *v495; // [rsp+3E8h] [rbp-388h] BYREF
  void *dest; // [rsp+3F0h] [rbp-380h] BYREF
  __int64 v497; // [rsp+3F8h] [rbp-378h]
  _BYTE v498[128]; // [rsp+400h] [rbp-370h] BYREF
  unsigned __int64 v499[94]; // [rsp+480h] [rbp-2F0h] BYREF

  memset(v499, 0, 0x2C0u);
  v385 = p_src;
  if ( a7 )
  {
    v499[66] = 0;
    v499[0] = (unsigned __int64)&v499[2];
    v499[1] = 0x1000000000LL;
    v499[67] = 0;
    v499[68] = a7;
    v499[69] = 0;
    LOBYTE(v499[70]) = 1;
    v499[71] = 0;
    v499[72] = (unsigned __int64)&v499[75];
    v499[73] = 8;
    LODWORD(v499[74]) = 0;
    BYTE4(v499[74]) = 1;
    LOWORD(v499[83]) = 0;
    memset(&v499[84], 0, 24);
    LOBYTE(v499[87]) = 1;
  }
  v7 = sub_B2BEC0(p_src);
  v11 = *(_QWORD *)(p_src + 80);
  v377 = 0;
  v369 = (_BYTE *)v7;
  v439 = (_QWORD *)v11;
  v401 = p_src + 72;
  if ( v11 == p_src + 72 )
    goto LABEL_173;
  do
  {
    v12 = 0;
    if ( LOBYTE(v499[87]) )
      v12 = v499;
    v392 = (__int64)v12;
    if ( !v439 )
      BUG();
    v442 = v439 + 3;
    v13 = (_QWORD *)v439[4];
    if ( v13 == v439 + 3 )
    {
LABEL_20:
      v439 = (_QWORD *)v439[1];
      goto LABEL_21;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v13 )
LABEL_508:
          BUG();
        if ( *((_BYTE *)v13 - 24) == 85 )
          break;
LABEL_13:
        v13 = (_QWORD *)v13[1];
        if ( v442 == v13 )
          goto LABEL_20;
      }
      if ( !(unsigned __int8)sub_A73ED0(v13 + 6, 23) && !(unsigned __int8)sub_B49560((__int64)(v13 - 3), 23) )
        break;
      if ( (unsigned __int8)sub_A73ED0(v13 + 6, 4) )
        break;
      p_src = 4;
      if ( (unsigned __int8)sub_B49560((__int64)(v13 - 3), 4) )
        break;
      v13 = (_QWORD *)v13[1];
      if ( v442 == v13 )
        goto LABEL_20;
    }
    p_src = *(v13 - 7);
    if ( !p_src )
      goto LABEL_13;
    if ( *(_BYTE *)p_src )
      goto LABEL_13;
    if ( *(_QWORD *)(p_src + 24) != v13[7] )
      goto LABEL_13;
    if ( !sub_981210(*a3, p_src, &v443) )
      goto LABEL_13;
    v17 = v443;
    LOBYTE(v8) = v443 == 186 || v443 == 357;
    v440 = v8;
    if ( !(_BYTE)v8 )
      goto LABEL_13;
    v18 = sub_B43CB0((__int64)(v13 - 3));
    p_src = 18;
    v429 = sub_B2D610(v18, 18);
    if ( v429 )
      goto LABEL_13;
    v8 = *((_DWORD *)v13 - 5) & 0x7FFFFFF;
    v19 = v13[4 * (2 - v8) - 3];
    if ( *(_BYTE *)v19 != 17 )
      goto LABEL_13;
    v8 = *(_QWORD *)(v19 + 24);
    v402 = *(_DWORD *)(v19 + 32) <= 0x40u ? *(_QWORD *)(v19 + 24) : *(_QWORD *)v8;
    if ( !v402 )
      goto LABEL_13;
    v20 = 1;
    v386 = v440;
    if ( v17 != 186 )
    {
      v386 = sub_988330((__int64)(v13 - 3));
      v20 = v386;
    }
    p_src = (__int64)a4;
    v21 = sub_11F3070(v13[2], a5, a6);
    sub_DFABC0((__int64)&v445, a4, v21, v20);
    if ( !v445 )
      goto LABEL_146;
    v22 = sub_C52410();
    v23 = v22 + 1;
    v24 = sub_C959E0();
    v25 = (_QWORD *)v22[2];
    if ( v25 )
    {
      v26 = v22 + 1;
      do
      {
        while ( 1 )
        {
          v27 = v25[2];
          v28 = v25[3];
          if ( v24 <= v25[4] )
            break;
          v25 = (_QWORD *)v25[3];
          if ( !v28 )
            goto LABEL_46;
        }
        v26 = v25;
        v25 = (_QWORD *)v25[2];
      }
      while ( v27 );
LABEL_46:
      if ( v23 != v26 && v24 >= v26[4] )
        v23 = v26;
    }
    if ( v23 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_57;
    v29 = v23[7];
    if ( !v29 )
      goto LABEL_57;
    v30 = v23 + 6;
    do
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v29 + 16);
        v32 = *(_QWORD *)(v29 + 24);
        if ( *(_DWORD *)(v29 + 32) >= dword_501D5A8 )
          break;
        v29 = *(_QWORD *)(v29 + 24);
        if ( !v32 )
          goto LABEL_55;
      }
      v30 = (_QWORD *)v29;
      v29 = *(_QWORD *)(v29 + 16);
    }
    while ( v31 );
LABEL_55:
    if ( v23 + 6 == v30 || dword_501D5A8 < *((_DWORD *)v30 + 8) || !*((_DWORD *)v30 + 9) )
    {
LABEL_57:
      if ( v21 )
        goto LABEL_155;
    }
    else
    {
      v449 = qword_501D628;
      if ( v21 )
      {
LABEL_155:
        v87 = sub_C52410();
        v88 = v87 + 1;
        v89 = sub_C959E0();
        v90 = (_QWORD *)v87[2];
        if ( v90 )
        {
          v91 = v87 + 1;
          do
          {
            while ( 1 )
            {
              v92 = v90[2];
              v93 = v90[3];
              if ( v89 <= v90[4] )
                break;
              v90 = (_QWORD *)v90[3];
              if ( !v93 )
                goto LABEL_160;
            }
            v91 = v90;
            v90 = (_QWORD *)v90[2];
          }
          while ( v92 );
LABEL_160:
          if ( v88 != v91 && v89 >= v91[4] )
            v88 = v91;
        }
        if ( v88 != (_QWORD *)((char *)sub_C52410() + 8) )
        {
          v94 = v88[7];
          if ( v94 )
          {
            v95 = v88 + 6;
            do
            {
              while ( 1 )
              {
                v96 = *(_QWORD *)(v94 + 16);
                v97 = *(_QWORD *)(v94 + 24);
                if ( *(_DWORD *)(v94 + 32) >= dword_501D3E8 )
                  break;
                v94 = *(_QWORD *)(v94 + 24);
                if ( !v97 )
                  goto LABEL_169;
              }
              v95 = (_QWORD *)v94;
              v94 = *(_QWORD *)(v94 + 16);
            }
            while ( v96 );
LABEL_169:
            if ( v88 + 6 != v95 && dword_501D3E8 >= *((_DWORD *)v95 + 8) && *((_DWORD *)v95 + 9) )
              v445 = qword_501D468;
          }
        }
        goto LABEL_74;
      }
    }
    v33 = sub_C52410();
    v34 = v33 + 1;
    v35 = sub_C959E0();
    v36 = (_QWORD *)v33[2];
    if ( v36 )
    {
      v37 = v33 + 1;
      do
      {
        while ( 1 )
        {
          v38 = v36[2];
          v39 = v36[3];
          if ( v35 <= v36[4] )
            break;
          v36 = (_QWORD *)v36[3];
          if ( !v39 )
            goto LABEL_63;
        }
        v37 = v36;
        v36 = (_QWORD *)v36[2];
      }
      while ( v38 );
LABEL_63:
      if ( v37 != v34 && v35 >= v37[4] )
        v34 = v37;
    }
    if ( v34 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v40 = v34[7];
      if ( v40 )
      {
        v41 = v34 + 6;
        do
        {
          while ( 1 )
          {
            v42 = *(_QWORD *)(v40 + 16);
            v43 = *(_QWORD *)(v40 + 24);
            if ( *(_DWORD *)(v40 + 32) >= dword_501D4C8 )
              break;
            v40 = *(_QWORD *)(v40 + 24);
            if ( !v43 )
              goto LABEL_72;
          }
          v41 = (_QWORD *)v40;
          v40 = *(_QWORD *)(v40 + 16);
        }
        while ( v42 );
LABEL_72:
        if ( v34 + 6 != v41 && dword_501D4C8 >= *((_DWORD *)v41 + 8) && *((_DWORD *)v41 + 9) )
          v445 = qword_501D548;
      }
    }
LABEL_74:
    v463 = (__int64)(v13 - 3);
    v464 = 0;
    v467 = v402;
    v465 = 0;
    v470 = v449;
    v466 = 0;
    v476 = v386;
    v468 = 0;
    v477 = v369;
    v469 = 0;
    v471 = 0;
    v472 = 0;
    v473 = 0;
    v474 = 0;
    v475 = 0;
    v478 = v392;
    v485 = (_QWORD *)sub_BD5C60((__int64)(v13 - 3));
    v486 = &v494;
    v487 = &v495;
    v480 = 0x200000000LL;
    v494 = &unk_49DA100;
    v479 = (unsigned int *)v481;
    v490 = 512;
    v495 = &unk_49DA0B0;
    v488 = 0;
    v489 = 0;
    v491 = 7;
    v492 = 0;
    v493 = 0;
    v482 = 0;
    v483 = 0;
    LOWORD(v484) = 0;
    sub_D5F1F0((__int64)&v479, (__int64)(v13 - 3));
    v45 = v446;
    dest = v498;
    v497 = 0x800000000LL;
    v46 = v447;
    if ( v447 )
    {
      while ( 1 )
      {
        v289 = *v45;
        if ( v402 >= v289 )
          break;
        if ( !--v46 )
        {
          v289 = v45[1];
          ++v45;
          break;
        }
        ++v45;
      }
    }
    else
    {
      v289 = *v446;
    }
    v47 = 0;
    v48 = 0x800000000LL;
    v468 = v289;
    v49 = v402;
    v413 = v445;
    v458 = (__int64)&v460;
    v459 = 0x800000000LL;
    v418 = &v45[v46];
    v378 = 0;
    v393 = 0;
    v371 = v13;
    v370 = v13 - 3;
    v50 = v45;
    do
    {
      if ( v50 == v418 )
        break;
      while ( 1 )
      {
        v51 = *v50;
        v48 = v413;
        v52 = v49 / v51;
        v44 = v51;
        v425 = v49 % v51;
        v289 = v47;
        if ( v47 + v49 / v51 > v413 )
        {
          v13 = v371;
          v53 = v370;
          v455 = 0x800000000LL;
          src = v456;
          goto LABEL_86;
        }
        if ( v51 <= v49 )
          break;
        if ( ++v50 == v418 )
          goto LABEL_84;
      }
      v107 = 0;
      v108 = v393;
      v379 = v49;
      v109 = v383;
      while ( 1 )
      {
        v109 = v51 | v109 & 0xFFFFFFFF00000000LL;
        if ( v289 + 1 > HIDWORD(v459) )
        {
          v376 = v52;
          v384 = v44;
          sub_C8D5F0((__int64)&v458, &v460, v289 + 1, 0x10u, v289 + 1, v44);
          v289 = (unsigned int)v459;
          v52 = v376;
          v44 = v384;
        }
        v110 = (unsigned __int64 *)(v458 + 16 * v289);
        ++v107;
        v110[1] = v108;
        v108 += v51;
        *v110 = v109;
        v47 = v459 + 1;
        LODWORD(v459) = v459 + 1;
        if ( v52 <= v107 )
          break;
        v289 = v47;
      }
      v111 = v52 - 1;
      v48 = 0;
      v383 = v109;
      v289 = v51 + v393;
      v49 = v425;
      if ( v51 > v379 )
        v111 = 0;
      v378 -= ((unsigned int)v44 < 2) - 1;
      ++v50;
      v393 = v289 + v51 * v111;
    }
    while ( v425 );
LABEL_84:
    v13 = v371;
    v53 = v370;
    src = v456;
    v455 = 0x800000000LL;
    if ( v47 )
      sub_2DC1F40((__int64)&src, (char **)&v458, v289, v48, v49, v44);
LABEL_86:
    if ( (char **)v458 != &v460 )
      _libc_free(v458);
    p_src = (__int64)&src;
    sub_2DC1F40((__int64)&dest, (char **)&src, v289, v48, v49, v44);
    if ( src != v456 )
      _libc_free((unsigned __int64)src);
    v469 = v378;
    v55 = (unsigned int)v497;
    if ( !v450 || (unsigned int)(v497 - 1) <= 1 )
      goto LABEL_100;
    if ( v402 <= 1 || v468 <= 1 || (v56 = v402 / v468, (v57 = v402 % v468) == 0) || (v250 = v56 + 1, v56 + 1 > v445) )
    {
      src = v456;
LABEL_96:
      v58 = src;
      goto LABEL_97;
    }
    v458 = (__int64)&v460;
    v459 = 0x800000000LL;
    if ( v402 < v468 )
    {
      v251 = 0;
      v268 = v57 - v468;
      v365 = v468 | v365 & 0xFFFFFFFF00000000LL;
      v267 = v365;
    }
    else
    {
      LODWORD(v251) = 0;
      v420 = v13;
      v252 = 8;
      v253 = v402 / v468;
      v396 = v53;
      v254 = v468;
      v255 = 0;
      v256 = 0;
      v257 = v363;
      while ( 1 )
      {
        v258 = (unsigned int)v251;
        v259 = (unsigned int)v251 + 1LL;
        v260 = v254 | v257 & 0xFFFFFFFF00000000LL;
        v257 = v260;
        if ( v259 > v252 )
        {
          v380 = v57;
          sub_C8D5F0((__int64)&v458, &v460, (unsigned int)v251 + 1LL, 0x10u, v259, v54);
          v258 = (unsigned int)v459;
          v257 = v260;
          v57 = v380;
        }
        v261 = (unsigned __int64 *)(v458 + 16 * v258);
        ++v255;
        v261[1] = v256;
        v256 += v254;
        *v261 = v260;
        v251 = (unsigned int)(v459 + 1);
        LODWORD(v459) = v459 + 1;
        if ( v253 <= v255 )
          break;
        v252 = HIDWORD(v459);
      }
      v262 = v254;
      v263 = v253;
      v264 = 1;
      v363 = v257;
      v265 = v254 | v365 & 0xFFFFFFFF00000000LL;
      v250 = v251 + 1;
      v13 = v420;
      if ( v402 >= v254 )
        v264 = v263;
      v266 = v57 - v254;
      v365 = v254 | v365 & 0xFFFFFFFF00000000LL;
      v53 = v396;
      v267 = v265;
      p_src = v262 * v264;
      v268 = p_src + v266;
      if ( v250 > HIDWORD(v459) )
      {
        p_src = (__int64)&v460;
        v417 = v268;
        sub_C8D5F0((__int64)&v458, &v460, v250, 0x10u, v265, v268);
        v251 = (unsigned int)v459;
        v268 = v417;
        v267 = v265;
      }
    }
    v269 = 16 * v251;
    v270 = (__int64 *)(v269 + v458);
    *v270 = v267;
    v270[1] = v268;
    v98 = (_DWORD)v459 == -1;
    LODWORD(v459) = v459 + 1;
    src = v456;
    v455 = 0x800000000LL;
    if ( !v98 )
    {
      p_src = (__int64)&v458;
      sub_2DC1F40((__int64)&src, (char **)&v458, v250, v269, v267, v268);
    }
    if ( (char **)v458 != &v460 )
      _libc_free(v458);
    v271 = v455;
    if ( !(_DWORD)v455 || (_DWORD)v497 && (unsigned int)v455 >= (unsigned int)v497 )
      goto LABEL_96;
    v272 = (unsigned int)v455;
    if ( (unsigned int)v497 >= (unsigned __int64)(unsigned int)v455 )
    {
      p_src = (__int64)src;
      memmove(dest, src, 16LL * (unsigned int)v455);
      LODWORD(v497) = v271;
      v58 = src;
    }
    else
    {
      if ( (unsigned int)v455 > HIDWORD(v497) )
      {
        LODWORD(v497) = 0;
        sub_C8D5F0((__int64)&dest, v498, (unsigned int)v455, 0x10u, v267, v268);
        v272 = (unsigned int)v455;
        v273 = 0;
      }
      else
      {
        v273 = 16LL * (unsigned int)v497;
        if ( (_DWORD)v497 )
        {
          v424 = 16LL * (unsigned int)v497;
          memmove(dest, src, v424);
          v272 = (unsigned int)v455;
          v273 = v424;
        }
      }
      v58 = src;
      v274 = 16 * v272;
      p_src = (__int64)src + v273;
      if ( (char *)src + v273 != (char *)src + v274 )
      {
        memcpy((char *)dest + v273, (const void *)p_src, v274 - v273);
        v58 = src;
      }
      LODWORD(v497) = v271;
    }
    v469 = 1;
LABEL_97:
    if ( v58 != v456 )
      _libc_free((unsigned __int64)v58);
    v55 = (unsigned int)v497;
LABEL_100:
    if ( !v386 )
    {
      if ( v452 )
      {
        v59 = (unsigned int)v55;
        if ( (unsigned int)v55 > 1 )
        {
          while ( 1 )
          {
            v60 = (char *)dest;
            p_src = (__int64)dest + 16 * v59 - 16;
            v61 = *(_QWORD *)(p_src - 16 + 8);
            if ( *(_QWORD *)(p_src + 8) != v61 + *(unsigned int *)(p_src - 16) )
            {
LABEL_205:
              v55 = (unsigned int)v497;
              goto LABEL_117;
            }
            v62 = v451;
            v63 = *(_DWORD *)p_src + *(_DWORD *)(p_src - 16);
            v64 = 4LL * v452;
            v65 = &v451[v64];
            v66 = v64 >> 2;
            p_src = v64 >> 4;
            if ( p_src )
            {
              p_src = (__int64)&v451[16 * p_src];
              while ( v63 != *(_DWORD *)v62 )
              {
                if ( v63 == *((_DWORD *)v62 + 1) )
                {
                  v62 += 4;
                  goto LABEL_111;
                }
                if ( v63 == *((_DWORD *)v62 + 2) )
                {
                  v62 += 8;
                  goto LABEL_111;
                }
                if ( v63 == *((_DWORD *)v62 + 3) )
                {
                  v62 += 12;
                  goto LABEL_111;
                }
                v62 += 16;
                if ( (char *)p_src == v62 )
                {
                  v66 = (v65 - v62) >> 2;
                  goto LABEL_198;
                }
              }
              goto LABEL_111;
            }
LABEL_198:
            if ( v66 == 2 )
              goto LABEL_202;
            if ( v66 == 3 )
              break;
            if ( v66 != 1 )
              goto LABEL_205;
LABEL_204:
            if ( v63 != *(_DWORD *)v62 )
              goto LABEL_205;
LABEL_111:
            if ( v65 == v62 )
              goto LABEL_205;
            v67 = (unsigned int)(v55 - 2);
            v68 = v63;
            LODWORD(v497) = v55 - 2;
            p_src = v67;
            if ( v67 >= HIDWORD(v497) )
            {
              v140 = v368 & 0xFFFFFFFF00000000LL | (unsigned int)v61;
              v368 = v140;
              if ( HIDWORD(v497) < (unsigned __int64)(unsigned int)(v55 - 1) )
              {
                p_src = (__int64)v498;
                v423 = v140;
                sub_C8D5F0((__int64)&dest, v498, (unsigned int)(v55 - 1), 0x10u, v140, HIDWORD(v497));
                v60 = (char *)dest;
                v67 = (unsigned int)v497;
                v140 = v423;
              }
              v141 = (unsigned __int64 *)&v60[16 * v67];
              *v141 = v140;
              v141[1] = v68;
              v55 = (unsigned int)(v497 + 1);
              LODWORD(v497) = v497 + 1;
            }
            else
            {
              v69 = (char *)dest + 16 * v67;
              if ( v69 )
              {
                *(_DWORD *)v69 = v61;
                *((_QWORD *)v69 + 1) = v63;
                p_src = (unsigned int)v497;
              }
              v55 = (unsigned int)(p_src + 1);
              LODWORD(v497) = p_src + 1;
            }
            v59 = (unsigned int)v55;
            if ( (unsigned int)v55 <= 1 )
              goto LABEL_117;
          }
          if ( v63 == *(_DWORD *)v62 )
            goto LABEL_111;
          v62 += 4;
LABEL_202:
          if ( v63 == *(_DWORD *)v62 )
            goto LABEL_111;
          v62 += 4;
          goto LABEL_204;
        }
      }
    }
LABEL_117:
    if ( !(_DWORD)v55 )
      goto LABEL_140;
    if ( v476 )
      LODWORD(v55) = v55 / v470 - ((v55 % v470 == 0) - 1);
    if ( (_DWORD)v55 != 1 )
    {
      v70 = *(_QWORD *)(v463 + 40);
      v71 = v366;
      v458 = (__int64)"endblock";
      LOWORD(v71) = 0;
      v419 = v70;
      v462 = 259;
      v366 = v71;
      v474 = sub_F36990(v70, (__int64 *)(v463 + 24), v71, v478, 0, 0, (void **)&v458, 0);
      v72 = v367;
      LOWORD(v72) = 1;
      v367 = v72;
      sub_A88F30((__int64)&v479, v474, *(_QWORD *)(v474 + 56), 1);
      v458 = (__int64)"phi.res";
      v462 = 259;
      v73 = (_QWORD *)sub_BD5C60(v463);
      v74 = sub_BCB2D0(v73);
      v475 = sub_D5C860((__int64 *)&v479, v74, 2, (__int64)&v458);
      v414 = v474;
      v430 = *(_QWORD *)(v474 + 72);
      v458 = (__int64)"res_block";
      v462 = 259;
      v75 = sub_BD5C60(v463);
      v76 = sub_22077B0(0x50u);
      v77 = v76;
      if ( v76 )
        sub_AA4D50(v76, v75, (__int64)&v458, v430, v414);
      v78 = v476;
      v464 = v77;
      if ( !v476 )
      {
        v285 = 8 * v468;
        v286 = (_QWORD *)sub_BD5C60(v463);
        v287 = sub_BCCE00(v286, v285);
        LOWORD(v484) = 0;
        v482 = v464;
        v483 = (char *)(v464 + 48);
        v458 = (__int64)"phi.src1";
        v462 = 259;
        v465 = sub_D5C860((__int64 *)&v479, v287, v469, (__int64)&v458);
        v458 = (__int64)"phi.src2";
        v462 = 259;
        v288 = sub_D5C860((__int64 *)&v479, v287, v469, (__int64)&v458);
        v78 = v476;
        v466 = v288;
      }
      v415 = v13;
      v79 = 0;
      v403 = v53;
      while ( 1 )
      {
        v81 = v497;
        if ( v78 )
          v81 = (unsigned int)v497 / v470 - (((unsigned int)v497 % v470 == 0) - 1);
        if ( v79 >= v81 )
          break;
        v82 = *(_QWORD *)(v474 + 72);
        v431 = v474;
        v458 = (__int64)"loadbb";
        v462 = 259;
        v83 = sub_BD5C60(v463);
        v84 = sub_22077B0(0x50u);
        v85 = (void *)v84;
        if ( v84 )
          sub_AA4D50(v84, v83, (__int64)&v458, v82, v431);
        src = v85;
        v80 = v472;
        if ( v472 == v473 )
        {
          sub_9319A0((__int64)&v471, v472, &src);
        }
        else
        {
          if ( v472 )
          {
            *(_QWORD *)v472 = v85;
            v80 = v472;
          }
          v472 = v80 + 8;
        }
        v78 = v476;
        ++v79;
      }
      v13 = v415;
      v53 = v403;
      v112 = *(_QWORD *)(v70 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v112 == v70 + 48 )
      {
        v114 = 0;
      }
      else
      {
        if ( !v112 )
          goto LABEL_508;
        v113 = *(unsigned __int8 *)(v112 - 24);
        v114 = 0;
        v115 = (unsigned __int8 *)(v112 - 24);
        if ( (unsigned int)(v113 - 30) < 0xB )
          v114 = v115;
      }
      sub_B46F90(v114, 0, *(_QWORD *)v471);
      if ( v478 )
      {
        v118 = *(_QWORD *)v471;
        v458 = v419;
        v460 = (char *)v419;
        v459 = v118 & 0xFFFFFFFFFFFFFFFBLL;
        v461 = v474 | 4;
        sub_FFB3D0(v478, (unsigned __int64 *)&v458, 2, v419, v116, v117);
      }
    }
    p_src = *(_QWORD *)(v463 + 48);
    v458 = p_src;
    if ( !p_src || (sub_B96E90((__int64)&v458, p_src, 1), (v120 = v458) == 0) )
    {
      p_src = 0;
      sub_93FB40((__int64)&v479, 0);
      v120 = v458;
      goto LABEL_330;
    }
    v121 = v479;
    v122 = v480;
    v123 = &v479[4 * (unsigned int)v480];
    if ( v479 != v123 )
    {
      while ( *v121 )
      {
        v121 += 4;
        if ( v123 == v121 )
          goto LABEL_332;
      }
      *((_QWORD *)v121 + 1) = v458;
      goto LABEL_224;
    }
LABEL_332:
    if ( (unsigned int)v480 >= (unsigned __int64)HIDWORD(v480) )
    {
      v317 = (unsigned int)v480 + 1LL;
      v318 = v364 & 0xFFFFFFFF00000000LL;
      v364 &= 0xFFFFFFFF00000000LL;
      if ( HIDWORD(v480) < v317 )
      {
        p_src = (__int64)v481;
        sub_C8D5F0((__int64)&v479, v481, v317, 0x10u, v317, v119);
        v123 = &v479[4 * (unsigned int)v480];
      }
      *(_QWORD *)v123 = v318;
      *((_QWORD *)v123 + 1) = v120;
      v120 = v458;
      LODWORD(v480) = v480 + 1;
    }
    else
    {
      if ( v123 )
      {
        *v123 = 0;
        *((_QWORD *)v123 + 1) = v120;
        v122 = v480;
        v120 = v458;
      }
      LODWORD(v480) = v122 + 1;
    }
LABEL_330:
    if ( v120 )
    {
LABEL_224:
      p_src = v120;
      sub_B91220((__int64)&v458, v120);
    }
    v124 = v476;
    LODWORD(v125) = v497;
    if ( v476 )
    {
      if ( (unsigned int)((unsigned int)v497 / v470) - (((unsigned int)v497 % v470 == 0) - 1) != 1 )
      {
        v387 = v13;
        v126 = 0;
        LODWORD(src) = 0;
        v381 = v53;
        for ( i = (unsigned int)v497; ; i = (unsigned int)v497 )
        {
          if ( v124 )
            LODWORD(i) = i / v470 - ((i % v470 == 0) - 1);
          if ( v126 >= (unsigned int)i )
            break;
          v128 = sub_2DC37C0((__int64)&v463, v126, &src);
          v129 = v474;
          v130 = v128;
          v432 = v126++;
          if ( v432 != ((__int64)&v472[-v471] >> 3) - 1 )
            v129 = *(_QWORD *)(v471 + 8LL * v126);
          v131 = v464;
          v394 = v482;
          v132 = sub_BD2C40(72, 3u);
          v134 = (__int64)v132;
          if ( v132 )
            sub_B4C9A0((__int64)v132, v131, v129, v130, 3u, v133, 0, 0);
          p_src = v134;
          v462 = 257;
          (*((void (__fastcall **)(void **, __int64, __int64 *, char *, __int64))*v487 + 2))(
            v487,
            v134,
            &v458,
            v483,
            v484);
          v137 = v479;
          v138 = &v479[4 * (unsigned int)v480];
          if ( v479 != v138 )
          {
            do
            {
              v139 = *((_QWORD *)v137 + 1);
              p_src = *v137;
              v137 += 4;
              sub_B99FD0(v134, p_src, v139);
            }
            while ( v138 != v137 );
          }
          if ( v478 )
          {
            p_src = (__int64)&v458;
            v461 = v129 & 0xFFFFFFFFFFFFFFFBLL;
            v458 = v394;
            v459 = v464 & 0xFFFFFFFFFFFFFFFBLL;
            v460 = (char *)v394;
            sub_FFB3D0(v478, (unsigned __int64 *)&v458, 2, v394, v135, v136);
          }
          if ( v432 == ((__int64)&v472[-v471] >> 3) - 1 )
          {
            v142 = (_QWORD *)sub_BD5C60(v463);
            v143 = sub_BCB2D0(v142);
            p_src = 0;
            v144 = sub_ACD640(v143, 0, 0);
            v145 = v475;
            v146 = v144;
            v147 = *(_QWORD *)(v471 + 8 * v432);
            v148 = *(_DWORD *)(v475 + 4) & 0x7FFFFFF;
            if ( v148 == *(_DWORD *)(v475 + 72) )
            {
              sub_B48D90(v475);
              v148 = *(_DWORD *)(v145 + 4) & 0x7FFFFFF;
            }
            v149 = (v148 + 1) & 0x7FFFFFF;
            v150 = v149 | *(_DWORD *)(v145 + 4) & 0xF8000000;
            v151 = *(_QWORD *)(v145 - 8) + 32LL * (unsigned int)(v149 - 1);
            *(_DWORD *)(v145 + 4) = v150;
            if ( *(_QWORD *)v151 )
            {
              v152 = *(_QWORD *)(v151 + 8);
              **(_QWORD **)(v151 + 16) = v152;
              if ( v152 )
                *(_QWORD *)(v152 + 16) = *(_QWORD *)(v151 + 16);
            }
            *(_QWORD *)v151 = v146;
            if ( v146 )
            {
              v153 = *(_QWORD *)(v146 + 16);
              *(_QWORD *)(v151 + 8) = v153;
              if ( v153 )
              {
                p_src = v151 + 8;
                *(_QWORD *)(v153 + 16) = v151 + 8;
              }
              *(_QWORD *)(v151 + 16) = v146 + 16;
              *(_QWORD *)(v146 + 16) = v151;
            }
            *(_QWORD *)(*(_QWORD *)(v145 - 8)
                      + 32LL * *(unsigned int *)(v145 + 72)
                      + 8LL * ((*(_DWORD *)(v145 + 4) & 0x7FFFFFFu) - 1)) = v147;
          }
          v124 = v476;
        }
        v13 = v387;
        v53 = v381;
        sub_2DC30A0((__int64)&v463);
        v86 = v475;
        goto LABEL_138;
      }
      LODWORD(v444) = 0;
      v275 = sub_2DC37C0((__int64)&v463, 0, &v444);
      p_src = 257;
      v457 = 257;
      v276 = v275;
      v277 = (_QWORD *)sub_BD5C60(v463);
      v278 = (__int64 **)sub_BCB2D0(v277);
      if ( v278 == *(__int64 ***)(v276 + 8) )
      {
        v86 = v276;
        goto LABEL_138;
      }
      v279 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v486 + 15);
      if ( v279 != sub_920130 )
      {
        v86 = v279((__int64)v486, 39u, (_BYTE *)v276, (__int64)v278);
        goto LABEL_376;
      }
      if ( *(_BYTE *)v276 > 0x15u )
        goto LABEL_377;
      v86 = (unsigned __int8)sub_AC4810(0x27u) ? sub_ADAB70(39, v276, v278, 0) : sub_AA93C0(0x27u, v276, (__int64)v278);
LABEL_376:
      if ( !v86 )
      {
LABEL_377:
        v462 = 257;
        v280 = sub_BD2C40(72, 1u);
        v281 = v280;
        if ( v280 )
        {
          v434 = v280;
          sub_B515B0((__int64)v280, v276, (__int64)v278, (__int64)&v458, 0, 0);
          v281 = v434;
        }
        v435 = (__int64)v281;
        p_src = (__int64)v281;
        (*((void (__fastcall **)(void **, _QWORD *, void **, char *, __int64))*v487 + 2))(v487, v281, &src, v483, v484);
        v282 = v479;
        v86 = v435;
        v283 = &v479[4 * (unsigned int)v480];
        if ( v479 != v283 )
        {
          do
          {
            v284 = *((_QWORD *)v282 + 1);
            p_src = *v282;
            v282 += 4;
            sub_B99FD0(v435, p_src, v284);
          }
          while ( v283 != v282 );
          v86 = v435;
        }
        goto LABEL_138;
      }
      goto LABEL_139;
    }
    v433 = 0;
    if ( (_DWORD)v497 == 1 )
    {
      v290 = 8 * v467;
      if ( *v477 || v467 == 1 )
      {
        v436 = v476;
        v291 = 0;
        v292 = (_QWORD *)sub_BD5C60(v463);
        v293 = sub_BCCE00(v292, v290);
        v294 = v436;
        v295 = (__int64 **)v293;
      }
      else
      {
        v438 = v476;
        LODWORD(v310) = 0;
        v311 = (_QWORD *)sub_BD5C60(v463);
        v295 = (__int64 **)sub_BCCE00(v311, v290);
        if ( 8 * v467 > 0 )
        {
          _BitScanReverse64(&v312, 8 * v467 - 1);
          v310 = 1LL << (64 - ((unsigned __int8)v312 ^ 0x3Fu));
        }
        v313 = (_QWORD *)sub_BD5C60(v463);
        v314 = sub_BCCE00(v313, v310);
        v294 = v438;
        v291 = (__int64 **)v314;
      }
      v296 = v468;
      if ( v467 > 0 )
      {
        if ( v467 != 1 )
        {
          _BitScanReverse64(&v297, v467 - 1);
          v125 = 1LL << (64 - ((unsigned __int8)v297 ^ 0x3Fu));
        }
        if ( v468 < (unsigned int)v125 )
          v296 = v125;
      }
      v421 = v294;
      v437 = v296;
      v298 = (_QWORD *)sub_BD5C60(v463);
      v299 = (__int64 **)sub_BCCE00(v298, 8 * v437);
      if ( (unsigned __int64)(v467 - 1) <= 1 )
      {
        v315 = (__int64 **)sub_BCB2D0(v485);
        p_src = sub_2DC22F0((__int64)&v463, v295, v291, v315, 0);
        v462 = 257;
        v86 = sub_929DE0(&v479, (_BYTE *)p_src, v316, (__int64)&v458, 0, 0);
        goto LABEL_138;
      }
      v406 = v421;
      v422 = (void *)sub_2DC22F0((__int64)&v463, v295, v291, v299, 0);
      v416 = v300;
      v301 = sub_BD36B0(v463);
      v302 = v406;
      v429 = v301;
      if ( !v301 )
        goto LABEL_398;
      v303 = v463;
      v304 = *(_QWORD **)(*(_QWORD *)(v463 + 16) + 24LL);
      v305 = *(_BYTE *)v304;
      if ( *(_BYTE *)v304 != 55 )
        goto LABEL_392;
      v319 = (unsigned __int8 *)*(v304 - 4);
      if ( !v319 )
        BUG();
      v320 = *v319;
      v321 = *(_DWORD *)(*(_QWORD *)(v463 + 8) + 8LL) >> 8;
      if ( (_BYTE)v320 != 17 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v319 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v320 <= 0x15u )
        {
          v335 = sub_AD7630((__int64)v319, 0, v320);
          v302 = v406;
          v319 = v335;
          if ( v335 && *v335 == 17 )
            goto LABEL_415;
          goto LABEL_418;
        }
        goto LABEL_396;
      }
LABEL_415:
      v322 = *((_DWORD *)v319 + 8);
      if ( v322 <= 0x40 )
      {
        v323 = *((_QWORD *)v319 + 3);
        goto LABEL_417;
      }
      v397 = v302;
      v408 = v319;
      v329 = sub_C444A0((__int64)(v319 + 24));
      v302 = v397;
      if ( v322 - v329 > 0x40 )
        goto LABEL_418;
      v323 = **((_QWORD **)v408 + 3);
LABEL_417:
      if ( v321 - 1 != v323 )
      {
LABEL_418:
        v303 = v463;
        v305 = *(_BYTE *)v304;
LABEL_392:
        v407 = v302;
        v458 = 38;
        if ( v305 == 82 )
        {
          v306 = sub_B53900((__int64)v304);
          src = (void *)sub_B53630(v306, v458);
          v308 = v307;
          LODWORD(v455) = v307;
          if ( !v307 || *(v304 - 8) != v303 )
            goto LABEL_395;
          v336 = *(v304 - 4);
          if ( *(_BYTE *)v336 == 17 )
          {
            v337 = *(_DWORD *)(v336 + 32);
            if ( v337 )
            {
              if ( v337 <= 0x40 )
                v338 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v337) == *(_QWORD *)(v336 + 24);
              else
                v338 = v337 == (unsigned int)sub_C445E0(v336 + 24);
LABEL_448:
              if ( !v338 )
                goto LABEL_395;
            }
            goto LABEL_449;
          }
          v339 = *(_QWORD *)(v336 + 8);
          v398 = v407;
          if ( (unsigned int)*(unsigned __int8 *)(v339 + 8) - 17 > 1 || *(_BYTE *)v336 > 0x15u )
            goto LABEL_395;
          v409 = (unsigned __int8 *)*(v304 - 4);
          v340 = sub_AD7630(v336, 0, v336);
          v341 = v398;
          if ( v340 && *v340 == 17 )
          {
            v342 = *((_DWORD *)v340 + 8);
            if ( v342 )
            {
              if ( v342 <= 0x40 )
                v338 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v342) == *((_QWORD *)v340 + 3);
              else
                v338 = v342 == (unsigned int)sub_C445E0((__int64)(v340 + 24));
              goto LABEL_448;
            }
LABEL_449:
            v309 = 39;
LABEL_397:
            if ( !sub_B532B0(v309) )
            {
LABEL_398:
              HIDWORD(v444) = 0;
              v462 = 257;
              src = v422;
              v455 = (__int64)v416;
              p_src = sub_BCB2D0(v485);
              v86 = sub_B35180((__int64)&v479, p_src, 0x16Au, (__int64)&src, 2u, v444, (__int64)&v458);
              goto LABEL_138;
            }
            v462 = 257;
            v348 = sub_B52EF0(v309);
            v328 = sub_92B530(&v479, v348, (__int64)v422, v416, (__int64)&v458);
LABEL_427:
            p_src = v328;
            sub_BD84D0((__int64)v304, v328);
            sub_B43D60(v304);
            sub_B43D60((_QWORD *)v463);
            goto LABEL_140;
          }
          if ( *(_BYTE *)(v339 + 8) == 17 )
          {
            v343 = *(_DWORD *)(v339 + 32);
            v399 = v409;
            v344 = 0;
            v389 = v308;
            v345 = v341;
            v410 = v343;
            while ( v410 != v344 )
            {
              v346 = sub_AD69F0(v399, v344);
              if ( !v346 )
                goto LABEL_395;
              if ( *(_BYTE *)v346 != 13 )
              {
                if ( *(_BYTE *)v346 != 17 )
                  goto LABEL_395;
                v347 = *(_DWORD *)(v346 + 32);
                if ( v347 )
                {
                  if ( v347 <= 0x40 )
                    v345 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v347) == *(_QWORD *)(v346 + 24);
                  else
                    v345 = v347 == (unsigned int)sub_C445E0(v346 + 24);
                  if ( !v345 )
                    goto LABEL_395;
                }
                else
                {
                  v345 = v389;
                }
              }
              ++v344;
            }
            if ( v345 )
              goto LABEL_449;
          }
LABEL_395:
          v303 = v463;
        }
LABEL_396:
        v459 = v303;
        v309 = 41;
        v458 = 40;
        v460 = 0;
        if ( (unsigned __int8)sub_2DC40B0(&v458, v304) )
          goto LABEL_397;
        v309 = 42;
        if ( *(_BYTE *)v304 != 82 || v463 != *(v304 - 8) || *(_BYTE *)*(v304 - 4) > 0x15u )
          goto LABEL_397;
        v411 = *(v304 - 4);
        if ( sub_AC30F0(v411) )
        {
LABEL_478:
          v309 = sub_B53900((__int64)v304);
          goto LABEL_397;
        }
        if ( *(_BYTE *)v411 == 17 )
        {
          v350 = *(_DWORD *)(v411 + 32);
          if ( v350 <= 0x40 )
          {
            if ( !*(_QWORD *)(v411 + 24) )
              goto LABEL_478;
          }
          else if ( v350 == (unsigned int)sub_C444A0(v411 + 24) )
          {
            goto LABEL_478;
          }
        }
        else
        {
          v351 = *(_QWORD *)(v411 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v351 + 8) - 17 <= 1 )
          {
            v352 = sub_AD7630(v411, 0, v349);
            if ( v352 && *v352 == 17 )
            {
              v353 = *((_DWORD *)v352 + 8);
              if ( v353 <= 0x40 )
                v354 = *((_QWORD *)v352 + 3) == 0;
              else
                v354 = v353 == (unsigned int)sub_C444A0((__int64)(v352 + 24));
              if ( v354 )
                goto LABEL_478;
            }
            else if ( *(_BYTE *)(v351 + 8) == 17 )
            {
              v355 = *(_DWORD *)(v351 + 32);
              v400 = (unsigned __int8 *)v411;
              v356 = 0;
              v357 = 0;
              v412 = v355;
              while ( v412 != v356 )
              {
                v390 = v356;
                v358 = sub_AD69F0(v400, v356);
                if ( !v358 )
                  goto LABEL_484;
                v359 = v390;
                if ( *(_BYTE *)v358 != 13 )
                {
                  if ( *(_BYTE *)v358 != 17 )
                    goto LABEL_484;
                  v360 = *(_DWORD *)(v358 + 32);
                  if ( v360 <= 0x40 )
                  {
                    v357 = *(_QWORD *)(v358 + 24) == 0;
                  }
                  else
                  {
                    v361 = sub_C444A0(v358 + 24);
                    v359 = v390;
                    v357 = v360 == v361;
                  }
                  if ( !v357 )
                    goto LABEL_484;
                }
                v356 = v359 + 1;
              }
              if ( v357 )
                goto LABEL_478;
            }
          }
        }
LABEL_484:
        v309 = 42;
        goto LABEL_397;
      }
      if ( !sub_B532B0(40) )
        goto LABEL_398;
      v462 = 257;
      v324 = sub_B52EF0(0x28u);
      v325 = sub_92B530(&v479, v324, (__int64)v422, v416, (__int64)&v458);
      v457 = 257;
      v326 = (_BYTE *)v325;
      v441 = (__int64 **)v304[1];
      if ( v441 == *(__int64 ***)(v325 + 8) )
      {
        v328 = v325;
        goto LABEL_427;
      }
      v327 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v486 + 15);
      if ( v327 == sub_920130 )
      {
        if ( *v326 <= 0x15u )
        {
          if ( (unsigned __int8)sub_AC4810(0x27u) )
            v328 = sub_ADAB70(39, (unsigned __int64)v326, v441, 0);
          else
            v328 = sub_AA93C0(0x27u, (unsigned __int64)v326, (__int64)v441);
          goto LABEL_426;
        }
      }
      else
      {
        v328 = v327((__int64)v486, 39u, v326, (__int64)v441);
LABEL_426:
        if ( v328 )
          goto LABEL_427;
      }
      v462 = 257;
      v330 = sub_BD2C40(72, 1u);
      v328 = (__int64)v330;
      if ( v330 )
        sub_B515B0((__int64)v330, (__int64)v326, (__int64)v441, (__int64)&v458, 0, 0);
      (*((void (__fastcall **)(void **, __int64, void **, char *, __int64))*v487 + 2))(v487, v328, &src, v483, v484);
      v331 = v479;
      v332 = &v479[4 * (unsigned int)v480];
      if ( v479 != v332 )
      {
        do
        {
          v333 = *((_QWORD *)v331 + 1);
          v334 = *v331;
          v331 += 4;
          sub_B99FD0(v328, v334, v333);
        }
        while ( v332 != v331 );
      }
      goto LABEL_427;
    }
    v395 = v13;
    v388 = v53;
    while ( (unsigned int)v125 > v433 )
    {
      v154 = v433;
      v155 = (unsigned int *)((char *)dest + 16 * v433);
      v156 = *v155;
      if ( *v155 == 1 )
      {
        v200 = *((_QWORD *)v155 + 1);
        v201 = *(char **)(v471 + 8LL * v433);
        LOWORD(v484) = 0;
        v482 = (unsigned __int64)v201;
        v483 = v201 + 48;
        v202 = (_QWORD *)sub_BD5C60(v463);
        v203 = (__int64 **)sub_BCB2D0(v202);
        v204 = (_QWORD *)sub_BD5C60(v463);
        v205 = (__int64 **)sub_BCB2B0(v204);
        v206 = (_BYTE *)sub_2DC22F0((__int64)&v463, v205, 0, v203, v200);
        v462 = 257;
        v208 = sub_929DE0(&v479, v206, v207, (__int64)&v458, 0, 0);
        v209 = v475;
        v210 = v208;
        v211 = *(_DWORD *)(v475 + 4) & 0x7FFFFFF;
        if ( v211 == *(_DWORD *)(v475 + 72) )
        {
          sub_B48D90(v475);
          v211 = *(_DWORD *)(v209 + 4) & 0x7FFFFFF;
        }
        v212 = (v211 + 1) & 0x7FFFFFF;
        v213 = v212 | *(_DWORD *)(v209 + 4) & 0xF8000000;
        v214 = *(_QWORD *)(v209 - 8) + 32LL * (unsigned int)(v212 - 1);
        *(_DWORD *)(v209 + 4) = v213;
        if ( *(_QWORD *)v214 )
        {
          v215 = *(_QWORD *)(v214 + 8);
          **(_QWORD **)(v214 + 16) = v215;
          if ( v215 )
            *(_QWORD *)(v215 + 16) = *(_QWORD *)(v214 + 16);
        }
        *(_QWORD *)v214 = v210;
        if ( v210 )
        {
          v216 = *(_QWORD *)(v210 + 16);
          *(_QWORD *)(v214 + 8) = v216;
          if ( v216 )
            *(_QWORD *)(v216 + 16) = v214 + 8;
          *(_QWORD *)(v214 + 16) = v210 + 16;
          *(_QWORD *)(v210 + 16) = v214;
        }
        ++v433;
        *(_QWORD *)(*(_QWORD *)(v209 - 8)
                  + 32LL * *(unsigned int *)(v209 + 72)
                  + 8LL * ((*(_DWORD *)(v209 + 4) & 0x7FFFFFFu) - 1)) = v201;
        if ( v154 >= ((__int64)&v472[-v471] >> 3) - 1 )
        {
          v241 = v474;
          v242 = sub_BD2C40(72, 1u);
          v243 = (__int64)v242;
          if ( v242 )
            sub_B4C8F0((__int64)v242, v241, 1u, 0, 0);
          p_src = v243;
          v462 = 257;
          (*((void (__fastcall **)(void **, __int64, __int64 *, char *, __int64))*v487 + 2))(
            v487,
            v243,
            &v458,
            v483,
            v484);
          v247 = v479;
          v248 = &v479[4 * (unsigned int)v480];
          if ( v479 != v248 )
          {
            do
            {
              v249 = *((_QWORD *)v247 + 1);
              p_src = *v247;
              v247 += 4;
              sub_B99FD0(v243, p_src, v249);
            }
            while ( v248 != v247 );
          }
          if ( v478 )
          {
            p_src = (__int64)&v458;
            v458 = (__int64)v201;
            v459 = v474 & 0xFFFFFFFFFFFFFFFBLL;
            sub_FFB3D0(v478, (unsigned __int64 *)&v458, 1, v244, v245, v246);
          }
        }
        else
        {
          v462 = 257;
          v217 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v210 + 8), 0, 0);
          v218 = sub_92B530(&v479, 0x21u, v210, v217, (__int64)&v458);
          v219 = v474;
          v382 = v218;
          v405 = *(_QWORD *)(v471 + 8LL * v433);
          v220 = sub_BD2C40(72, 3u);
          v222 = (__int64)v220;
          if ( v220 )
            sub_B4C9A0((__int64)v220, v219, v405, v382, 3u, v221, 0, 0);
          p_src = v222;
          v462 = 257;
          (*((void (__fastcall **)(void **, __int64, __int64 *, char *, __int64))*v487 + 2))(
            v487,
            v222,
            &v458,
            v483,
            v484);
          v225 = v479;
          v226 = &v479[4 * (unsigned int)v480];
          if ( v479 != v226 )
          {
            do
            {
              v227 = *((_QWORD *)v225 + 1);
              p_src = *v225;
              v225 += 4;
              sub_B99FD0(v222, p_src, v227);
            }
            while ( v226 != v225 );
          }
          if ( v478 )
          {
            p_src = (__int64)&v458;
            v458 = (__int64)v201;
            v459 = v474 & 0xFFFFFFFFFFFFFFFBLL;
            v228 = *(_QWORD *)(v471 + 8LL * v433);
            v460 = v201;
            v461 = v228 & 0xFFFFFFFFFFFFFFFBLL;
            sub_FFB3D0(v478, (unsigned __int64 *)&v458, 2, v433, v223, v224);
          }
        }
      }
      else
      {
        v157 = (_QWORD *)sub_BD5C60(v463);
        v158 = 0;
        v159 = (__int64 **)sub_BCCE00(v157, 8 * v156);
        if ( !*v477 )
        {
          LODWORD(v160) = 8 * *v155;
          if ( (_DWORD)v160 )
          {
            _BitScanReverse64(&v161, (unsigned int)v160 - 1LL);
            v160 = 1LL << (64 - ((unsigned __int8)v161 ^ 0x3Fu));
          }
          v162 = (_QWORD *)sub_BD5C60(v463);
          v158 = (__int64 **)sub_BCCE00(v162, v160);
        }
        v163 = *v155;
        v164 = v468;
        if ( (_DWORD)v163 )
        {
          LODWORD(v165) = 1;
          v166 = v163 - 1;
          if ( v166 )
          {
            _BitScanReverse64(&v166, v166);
            v165 = 1LL << (64 - ((unsigned __int8)v166 ^ 0x3Fu));
          }
          if ( v468 < (unsigned int)v165 )
            v164 = v165;
        }
        v167 = (_QWORD *)sub_BD5C60(v463);
        v168 = (__int64 **)sub_BCCE00(v167, 8 * v164);
        v169 = *(_QWORD *)(v471 + 8LL * v433);
        LOWORD(v484) = 0;
        v482 = v169;
        v483 = (char *)(v169 + 48);
        v171 = sub_2DC22F0((__int64)&v463, v159, v158, v168, v155[2]);
        v172 = v170;
        if ( !v476 )
        {
          v184 = v465;
          v185 = *(_QWORD *)(v471 + 8LL * v433);
          v186 = *(_DWORD *)(v465 + 4) & 0x7FFFFFF;
          if ( v186 == *(_DWORD *)(v465 + 72) )
          {
            v428 = v170;
            sub_B48D90(v465);
            v172 = v428;
            v186 = *(_DWORD *)(v184 + 4) & 0x7FFFFFF;
          }
          v187 = (v186 + 1) & 0x7FFFFFF;
          v188 = v187 | *(_DWORD *)(v184 + 4) & 0xF8000000;
          v189 = *(_QWORD *)(v184 - 8) + 32LL * (unsigned int)(v187 - 1);
          *(_DWORD *)(v184 + 4) = v188;
          if ( *(_QWORD *)v189 )
          {
            v190 = *(_QWORD *)(v189 + 8);
            **(_QWORD **)(v189 + 16) = v190;
            if ( v190 )
              *(_QWORD *)(v190 + 16) = *(_QWORD *)(v189 + 16);
          }
          *(_QWORD *)v189 = v171;
          if ( v171 )
          {
            v191 = *(_QWORD *)(v171 + 16);
            *(_QWORD *)(v189 + 8) = v191;
            if ( v191 )
              *(_QWORD *)(v191 + 16) = v189 + 8;
            *(_QWORD *)(v189 + 16) = v171 + 16;
            *(_QWORD *)(v171 + 16) = v189;
          }
          *(_QWORD *)(*(_QWORD *)(v184 - 8)
                    + 32LL * *(unsigned int *)(v184 + 72)
                    + 8LL * ((*(_DWORD *)(v184 + 4) & 0x7FFFFFFu) - 1)) = v185;
          v192 = v466;
          v193 = *(_QWORD *)(v471 + 8LL * v433);
          v194 = *(_DWORD *)(v466 + 4) & 0x7FFFFFF;
          if ( v194 == *(_DWORD *)(v466 + 72) )
          {
            v427 = v172;
            sub_B48D90(v466);
            v172 = v427;
            v194 = *(_DWORD *)(v192 + 4) & 0x7FFFFFF;
          }
          v195 = (v194 + 1) & 0x7FFFFFF;
          v196 = v195 | *(_DWORD *)(v192 + 4) & 0xF8000000;
          v197 = *(_QWORD *)(v192 - 8) + 32LL * (unsigned int)(v195 - 1);
          *(_DWORD *)(v192 + 4) = v196;
          if ( *(_QWORD *)v197 )
          {
            v198 = *(_QWORD *)(v197 + 8);
            **(_QWORD **)(v197 + 16) = v198;
            if ( v198 )
              *(_QWORD *)(v198 + 16) = *(_QWORD *)(v197 + 16);
          }
          *(_QWORD *)v197 = v172;
          if ( v172 )
          {
            v199 = *(_QWORD *)(v172 + 16);
            *(_QWORD *)(v197 + 8) = v199;
            if ( v199 )
              *(_QWORD *)(v199 + 16) = v197 + 8;
            *(_QWORD *)(v197 + 16) = v172 + 16;
            *(_QWORD *)(v172 + 16) = v197;
          }
          *(_QWORD *)(*(_QWORD *)(v192 - 8)
                    + 32LL * *(unsigned int *)(v192 + 72)
                    + 8LL * ((*(_DWORD *)(v192 + 4) & 0x7FFFFFFu) - 1)) = v193;
        }
        v462 = 257;
        ++v433;
        v173 = sub_92B530(&v479, 0x20u, v171, (_BYTE *)v172, (__int64)&v458);
        if ( v154 == ((__int64)&v472[-v471] >> 3) - 1 )
          v426 = v474;
        else
          v426 = *(_QWORD *)(v471 + 8LL * v433);
        v174 = v464;
        v404 = v482;
        v175 = sub_BD2C40(72, 3u);
        v177 = (__int64)v175;
        if ( v175 )
        {
          sub_B4C9A0((__int64)v175, v426, v174, v173, 3u, 0, 0, 0);
          v176 = v362;
        }
        p_src = v177;
        v462 = 257;
        (*((void (__fastcall **)(void **, __int64, __int64 *, char *, __int64, __int64))*v487 + 2))(
          v487,
          v177,
          &v458,
          v483,
          v484,
          v176);
        v181 = v479;
        v182 = &v479[4 * (unsigned int)v480];
        if ( v479 != v182 )
        {
          do
          {
            v183 = *((_QWORD *)v181 + 1);
            p_src = *v181;
            v181 += 4;
            sub_B99FD0(v177, p_src, v183);
          }
          while ( v182 != v181 );
        }
        if ( v478 )
        {
          p_src = (__int64)&v458;
          v458 = v404;
          v460 = (char *)v404;
          v459 = v426 & 0xFFFFFFFFFFFFFFFBLL;
          v461 = v464 & 0xFFFFFFFFFFFFFFFBLL;
          sub_FFB3D0(v478, (unsigned __int64 *)&v458, 2, v178, v179, v180);
        }
        if ( v154 == ((__int64)&v472[-v471] >> 3) - 1 )
        {
          v229 = (_QWORD *)sub_BD5C60(v463);
          v230 = sub_BCB2D0(v229);
          p_src = 0;
          v231 = sub_ACD640(v230, 0, 0);
          v232 = v475;
          v233 = v231;
          v234 = *(_QWORD *)(v471 + 8 * v154);
          v235 = *(_DWORD *)(v475 + 4) & 0x7FFFFFF;
          if ( v235 == *(_DWORD *)(v475 + 72) )
          {
            sub_B48D90(v475);
            v235 = *(_DWORD *)(v232 + 4) & 0x7FFFFFF;
          }
          v236 = (v235 + 1) & 0x7FFFFFF;
          v237 = v236 | *(_DWORD *)(v232 + 4) & 0xF8000000;
          v238 = *(_QWORD *)(v232 - 8) + 32LL * (unsigned int)(v236 - 1);
          *(_DWORD *)(v232 + 4) = v237;
          if ( *(_QWORD *)v238 )
          {
            v239 = *(_QWORD *)(v238 + 8);
            **(_QWORD **)(v238 + 16) = v239;
            if ( v239 )
              *(_QWORD *)(v239 + 16) = *(_QWORD *)(v238 + 16);
          }
          *(_QWORD *)v238 = v233;
          if ( v233 )
          {
            v240 = *(_QWORD *)(v233 + 16);
            *(_QWORD *)(v238 + 8) = v240;
            if ( v240 )
            {
              p_src = v238 + 8;
              *(_QWORD *)(v240 + 16) = v238 + 8;
            }
            *(_QWORD *)(v238 + 16) = v233 + 16;
            *(_QWORD *)(v233 + 16) = v238;
          }
          *(_QWORD *)(*(_QWORD *)(v232 - 8)
                    + 32LL * *(unsigned int *)(v232 + 72)
                    + 8LL * ((*(_DWORD *)(v232 + 4) & 0x7FFFFFFu) - 1)) = v234;
        }
      }
      LODWORD(v125) = v497;
      if ( v476 )
        LODWORD(v125) = (unsigned int)v497 / v470 - (((unsigned int)v497 % v470 == 0) - 1);
    }
    v13 = v395;
    v53 = v388;
    sub_2DC30A0((__int64)&v463);
    v86 = v475;
LABEL_138:
    v429 = v440;
    if ( v86 )
    {
LABEL_139:
      p_src = v86;
      sub_BD84D0((__int64)v53, v86);
      sub_B43D60(v53);
      v429 = v440;
    }
LABEL_140:
    if ( dest != v498 )
      _libc_free((unsigned __int64)dest);
    nullsub_61();
    v494 = &unk_49DA100;
    nullsub_63();
    if ( v479 != (unsigned int *)v481 )
      _libc_free((unsigned __int64)v479);
    if ( v471 )
    {
      p_src = (__int64)&v473[-v471];
      j_j___libc_free_0(v471);
    }
LABEL_146:
    if ( v451 != &v453 )
      _libc_free((unsigned __int64)v451);
    if ( v446 != (unsigned int *)&v448 )
      _libc_free((unsigned __int64)v446);
    if ( !v429 )
      goto LABEL_13;
    v439 = *(_QWORD **)(v385 + 80);
    v377 = v429;
LABEL_21:
    v11 = v401;
  }
  while ( v439 != (_QWORD *)v401 );
  if ( v377 )
  {
    for ( j = *(_QWORD *)(v385 + 80); j != v401; j = *(_QWORD *)(j + 8) )
    {
      v15 = j - 24;
      if ( !j )
        v15 = 0;
      sub_F61E50(v15, 0);
    }
    v464 = (unsigned __int64)&v467;
    v467 = (__int64)&unk_4F81450;
    v465 = 0x100000002LL;
    LODWORD(v466) = 0;
    BYTE4(v466) = 1;
    v469 = 0;
    v470 = (unsigned __int64)&v473;
    v471 = 2;
    LODWORD(v472) = 0;
    BYTE4(v472) = 1;
    v463 = 1;
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v467, (__int64)&v463);
    p_src = a1 + 80;
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v473, (__int64)&v469);
    if ( BYTE4(v472) )
    {
      if ( BYTE4(v466) )
        goto LABEL_29;
    }
    else
    {
      _libc_free(v470);
      if ( BYTE4(v466) )
        goto LABEL_29;
    }
    _libc_free(v464);
LABEL_29:
    if ( !LOBYTE(v499[87]) )
      return a1;
    goto LABEL_174;
  }
LABEL_173:
  v98 = LOBYTE(v499[87]) == 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  if ( v98 )
    return a1;
LABEL_174:
  LOBYTE(v499[87]) = 0;
  sub_FFCE90((__int64)v499, p_src, v8, v11, v9, v10);
  sub_FFD870((__int64)v499, p_src, v99, v100, v101, v102);
  sub_FFBC40((__int64)v499, p_src);
  v103 = v499[85];
  v104 = (_QWORD *)v499[84];
  if ( v499[85] != v499[84] )
  {
    do
    {
      v105 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v104[7];
      *v104 = &unk_49E5048;
      if ( v105 )
        v105(v104 + 5, v104 + 5, 3);
      *v104 = &unk_49DB368;
      v106 = v104[3];
      if ( v106 != -4096 && v106 != 0 && v106 != -8192 )
        sub_BD60C0(v104 + 1);
      v104 += 9;
    }
    while ( (_QWORD *)v103 != v104 );
    v104 = (_QWORD *)v499[84];
  }
  if ( v104 )
    j_j___libc_free_0((unsigned __int64)v104);
  if ( !BYTE4(v499[74]) )
    _libc_free(v499[72]);
  if ( (unsigned __int64 *)v499[0] != &v499[2] )
    _libc_free(v499[0]);
  return a1;
}
