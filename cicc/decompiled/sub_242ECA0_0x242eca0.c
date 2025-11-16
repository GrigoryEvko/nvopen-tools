// Function: sub_242ECA0
// Address: 0x242eca0
//
__int64 __fastcall sub_242ECA0(
        __int64 a1,
        __int64 a2,
        char a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        int a7,
        __int64 *a8)
{
  int v8; // esi
  int v9; // eax
  int v10; // eax
  __int64 result; // rax
  __int64 v12; // rax
  char *v13; // rdx
  __int64 v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // r15
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // rax
  int v20; // eax
  char *v21; // rsi
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // r15d
  __int64 v25; // rax
  __int64 v26; // r14
  unsigned int **v27; // rsi
  unsigned __int32 v28; // eax
  __int64 v29; // rdi
  unsigned int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // rdi
  unsigned int **v33; // r14
  unsigned int **v34; // rbx
  unsigned int *v35; // rdi
  __int64 v36; // rdi
  unsigned int *v37; // r12
  __int64 v38; // rbx
  __int64 *v39; // r13
  __int64 **v40; // rax
  __int64 **v41; // r13
  _BYTE *v42; // rbx
  __int64 v43; // rax
  char v44; // dl
  __int64 v45; // rbx
  __int64 *v46; // r12
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rcx
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // r15
  unsigned __int64 v57; // rax
  int v58; // edx
  unsigned __int64 v59; // rax
  bool v60; // cf
  __int64 v61; // rdx
  __int64 v62; // r14
  unsigned __int64 v63; // r12
  unsigned int v64; // r13d
  int v65; // esi
  __int64 v66; // r10
  int v67; // esi
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // rdi
  __int64 *v71; // r9
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rsi
  char v75; // al
  unsigned __int64 v76; // rcx
  __int64 v77; // rax
  unsigned __int64 v78; // rsi
  unsigned __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rbx
  __int64 v82; // r9
  __int64 v83; // rdx
  _QWORD *v84; // rax
  _QWORD *v85; // rdx
  __int64 v86; // r8
  unsigned int v87; // ecx
  int v88; // r9d
  int v89; // r11d
  unsigned __int64 *v90; // rbx
  unsigned __int64 *v91; // r12
  unsigned __int64 *v92; // rbx
  unsigned __int64 *v93; // r12
  __int64 **v94; // rbx
  __int64 **v95; // r12
  __int64 *v96; // rax
  __int64 v97; // rdi
  __int64 *v98; // rbx
  __int64 *v99; // r13
  __int64 *v100; // rax
  __int64 v101; // rsi
  __int64 *v102; // rax
  __int64 v103; // r14
  __int64 v104; // r15
  __int64 v105; // rcx
  int v106; // ebx
  __int64 v107; // rax
  int v108; // r12d
  __int64 v109; // rax
  __int64 v110; // rcx
  unsigned __int64 v111; // rdi
  unsigned __int64 *v112; // rdx
  int v113; // eax
  unsigned __int64 *v114; // rcx
  __int64 v115; // rax
  unsigned __int64 v116; // rdi
  __int64 v117; // rax
  __int64 v118; // rbx
  _QWORD *v119; // r13
  unsigned __int64 v120; // rdi
  __int64 v121; // r12
  unsigned __int64 v122; // rdi
  unsigned __int64 v123; // rdi
  unsigned __int64 v124; // rdi
  unsigned __int64 v125; // rdi
  unsigned __int64 v126; // r12
  int v127; // r9d
  __int64 v128; // rax
  unsigned __int64 v129; // rdi
  __int64 v130; // rbx
  __int64 v131; // r13
  _QWORD *v132; // r14
  unsigned __int64 v133; // rdi
  __int64 v134; // r15
  unsigned __int64 v135; // rdi
  unsigned __int64 v136; // rdi
  unsigned __int64 *v137; // r9
  __int64 v138; // r11
  __int64 v139; // rax
  __int64 *v140; // r12
  __int64 v141; // rcx
  unsigned __int64 *v142; // rbx
  unsigned __int64 *v143; // r13
  unsigned __int64 *v144; // r12
  unsigned __int64 v145; // rax
  unsigned __int64 v146; // rdi
  __int64 *v147; // r9
  __int64 v148; // rdx
  __int64 *v149; // rbx
  __int64 v150; // r13
  __int64 v151; // rsi
  unsigned __int64 v152; // rdi
  unsigned __int64 *v153; // r12
  __int64 *v154; // r12
  __int64 v155; // rax
  unsigned __int64 *v156; // r12
  __int64 v157; // r11
  unsigned __int64 *v158; // rbx
  __int64 *v159; // rcx
  _QWORD *v160; // rdx
  __int64 v161; // rax
  __int64 v162; // rdi
  __int64 v163; // rsi
  __int64 v164; // r9
  unsigned int v165; // r8d
  __int64 *v166; // rax
  __int64 v167; // r10
  __int64 v168; // r8
  __int64 v169; // r8
  __int64 v170; // r8
  __int64 v171; // rsi
  __int64 v172; // rdi
  __int64 v173; // r10
  unsigned int v174; // r9d
  __int64 *v175; // rsi
  __int64 v176; // rbx
  __int64 v177; // r8
  __int64 v178; // r8
  unsigned __int64 *v179; // r12
  __int64 v180; // r11
  unsigned __int64 *v181; // rbx
  unsigned __int64 *v182; // r12
  __int64 v183; // r11
  __int64 v184; // r12
  __int64 *v185; // r13
  __int64 *v186; // r15
  _QWORD *v187; // rax
  __int64 v188; // rbx
  __int64 v189; // rsi
  __int64 v190; // rcx
  __int64 v191; // r8
  unsigned int v192; // edi
  __int64 *v193; // rdx
  __int64 v194; // r9
  __int64 v195; // rsi
  __int64 v196; // rsi
  __int64 v197; // rsi
  __int64 v198; // r8
  __int64 v199; // rcx
  __int64 v200; // r8
  unsigned int v201; // edi
  __int64 *v202; // rdx
  __int64 v203; // r9
  __int64 v204; // rsi
  __int64 v205; // rsi
  unsigned __int64 v206; // rsi
  bool v207; // zf
  unsigned __int64 v208; // rax
  int v209; // edx
  __int64 v210; // rax
  unsigned __int64 v211; // rax
  __int64 v212; // rcx
  __int64 v213; // rbx
  __int64 j; // r12
  unsigned int v215; // r15d
  char *v216; // rsi
  _QWORD *v217; // rbx
  _QWORD *v218; // r12
  _BYTE *v219; // rdi
  __int64 v220; // rax
  __int64 v221; // rax
  _QWORD *v222; // rbx
  _QWORD *v223; // r12
  unsigned __int64 v224; // rdi
  unsigned __int64 *v225; // rbx
  unsigned __int64 *v226; // r12
  __int64 v227; // rcx
  __int64 *v228; // rsi
  __int64 v229; // rcx
  __int64 v230; // rcx
  __int64 v231; // rdx
  __int64 v232; // rdx
  int v233; // esi
  int v234; // edx
  int v235; // edx
  int v236; // eax
  unsigned int v237; // eax
  unsigned int *v238; // r9
  __int64 v239; // r8
  __int64 v240; // rax
  unsigned __int64 v241; // r9
  __int64 *v242; // rax
  __int64 v243; // rdx
  unsigned __int8 *v244; // rax
  unsigned __int8 v245; // cl
  unsigned int v246; // ecx
  __int64 v247; // rdi
  unsigned int v248; // edx
  __int64 *v249; // rax
  __int64 v250; // r9
  __int64 v251; // rsi
  __int64 v252; // rax
  __int64 v253; // rax
  __int64 v254; // r8
  __int64 v255; // r9
  __int64 v256; // rbx
  __int64 v257; // rax
  int i; // eax
  int v259; // ecx
  __int64 v260; // rbx
  __int64 *v261; // rax
  _QWORD *v262; // r12
  __int64 v263; // r13
  __int64 v264; // r8
  __int64 v265; // r9
  __int64 v266; // rax
  int v267; // edx
  __int64 *v268; // rax
  __int64 v269; // r14
  __int16 v270; // dx
  __int64 v271; // r13
  char v272; // bl
  char v273; // r12
  __int64 v274; // r14
  _QWORD *v275; // rdi
  __int16 v276; // ax
  __int64 v277; // rdi
  __int64 v278; // rsi
  __int64 v279; // r9
  __int64 v280; // r12
  unsigned int *v281; // rax
  int v282; // esi
  unsigned int *v283; // rdx
  __int64 v284; // r13
  __int64 v285; // rax
  __int64 v286; // rax
  __int64 (__fastcall *v287)(__int64, __int64, unsigned __int8 *, _BYTE **, __int64, int); // rax
  _BYTE **v288; // rax
  __int64 *v289; // r10
  _BYTE **v290; // rcx
  __int64 v291; // r12
  __int64 v292; // rax
  unsigned __int8 v293; // bl
  __int64 v294; // r13
  __int64 v295; // rax
  __int64 v296; // rax
  __int64 (__fastcall **v297)(); // rdx
  unsigned __int64 v298; // rax
  _QWORD *v299; // rax
  __int64 v300; // r14
  __int64 v301; // rsi
  unsigned int *v302; // r12
  unsigned int *v303; // rbx
  __int64 v304; // rdx
  __int64 v305; // rax
  __int64 v306; // rax
  __int64 v307; // rbx
  _QWORD *v308; // r13
  unsigned __int64 v309; // rdi
  __int64 v310; // r12
  unsigned __int64 v311; // rdi
  int v312; // r14d
  int v313; // r11d
  int v314; // r11d
  __int64 v315; // r13
  __int64 v316; // rax
  char v317; // al
  char v318; // bl
  _QWORD *v319; // rax
  __int64 v320; // r9
  _BYTE *v321; // r14
  __int64 v322; // rsi
  unsigned int *v323; // r13
  unsigned int *v324; // rbx
  __int64 v325; // rdx
  __int64 v326; // rax
  _BYTE *v327; // rax
  __int64 v328; // rbx
  __int64 v329; // rax
  char v330; // al
  __int16 v331; // cx
  _QWORD *v332; // rax
  __int64 v333; // r9
  unsigned int *v334; // r12
  unsigned int *v335; // rbx
  __int64 v336; // rdx
  __int64 v337; // rax
  unsigned int *v338; // r13
  unsigned int *v339; // rbx
  __int64 v340; // rdx
  unsigned int v341; // esi
  __int64 v342; // rsi
  int v343; // edx
  int v344; // edx
  char v345; // dl
  int v346; // esi
  _QWORD *v347; // rcx
  _QWORD *v348; // rax
  __int64 v349; // rdx
  _QWORD *v350; // rdx
  _QWORD *v351; // rcx
  unsigned __int64 v352; // r12
  unsigned __int64 v353; // rdi
  __int64 v354; // rax
  __int64 v355; // r14
  __int64 v356; // rbx
  _QWORD *v357; // r13
  unsigned __int64 v358; // rdi
  __int64 v359; // r15
  unsigned __int64 v360; // rdi
  unsigned __int64 v361; // rdi
  unsigned __int64 v362; // rdi
  __int64 v363; // rax
  __int64 v364; // r14
  __int64 v365; // rbx
  _QWORD *v366; // r13
  unsigned __int64 v367; // rdi
  __int64 v368; // r15
  unsigned __int64 v369; // rdi
  unsigned __int64 v370; // rdi
  unsigned __int64 v371; // r9
  unsigned __int64 v372; // r14
  int v373; // ebx
  __int64 v374; // rax
  unsigned __int64 v375; // rdi
  __int64 v376; // r15
  __int64 v377; // rbx
  _QWORD *v378; // r12
  unsigned __int64 v379; // rdi
  __int64 v380; // r13
  unsigned __int64 v381; // rdi
  unsigned __int64 v382; // rdi
  int v383; // ebx
  unsigned __int64 v384; // rbx
  __int64 *v385; // rax
  int v386; // eax
  int v387; // r8d
  __int64 v388; // rdx
  __int64 v389; // rdx
  __int64 v390; // [rsp-10h] [rbp-430h]
  __int64 v391; // [rsp+0h] [rbp-420h]
  _QWORD *v392; // [rsp+8h] [rbp-418h]
  unsigned __int64 v393; // [rsp+10h] [rbp-410h]
  __int64 v394; // [rsp+18h] [rbp-408h]
  unsigned int v395; // [rsp+24h] [rbp-3FCh]
  __int64 v396; // [rsp+28h] [rbp-3F8h]
  unsigned __int64 v397; // [rsp+30h] [rbp-3F0h]
  __int64 v398; // [rsp+38h] [rbp-3E8h]
  unsigned __int64 v400; // [rsp+50h] [rbp-3D0h]
  _QWORD *v401; // [rsp+58h] [rbp-3C8h]
  unsigned __int64 v402; // [rsp+58h] [rbp-3C8h]
  int v403; // [rsp+60h] [rbp-3C0h]
  __int64 v404; // [rsp+68h] [rbp-3B8h]
  _QWORD *v405; // [rsp+70h] [rbp-3B0h]
  _QWORD *v406; // [rsp+70h] [rbp-3B0h]
  __int16 v407; // [rsp+7Ah] [rbp-3A6h]
  char v409; // [rsp+7Eh] [rbp-3A2h]
  char v410; // [rsp+7Fh] [rbp-3A1h]
  __int64 v411; // [rsp+88h] [rbp-398h]
  int v412; // [rsp+88h] [rbp-398h]
  unsigned __int64 v413; // [rsp+88h] [rbp-398h]
  _BYTE *v414; // [rsp+90h] [rbp-390h]
  int v415; // [rsp+98h] [rbp-388h]
  int v416; // [rsp+9Ch] [rbp-384h]
  __int64 v417; // [rsp+A0h] [rbp-380h]
  _QWORD *v418; // [rsp+A0h] [rbp-380h]
  __int64 v419; // [rsp+A8h] [rbp-378h]
  __int64 v420; // [rsp+A8h] [rbp-378h]
  unsigned __int64 v421; // [rsp+B0h] [rbp-370h]
  char v422; // [rsp+B8h] [rbp-368h]
  signed __int64 v423; // [rsp+B8h] [rbp-368h]
  unsigned __int64 v425; // [rsp+C8h] [rbp-358h]
  __int64 v426; // [rsp+C8h] [rbp-358h]
  __int64 v427; // [rsp+C8h] [rbp-358h]
  __int64 v428; // [rsp+C8h] [rbp-358h]
  unsigned __int64 v429; // [rsp+D0h] [rbp-350h]
  _QWORD *v430; // [rsp+D0h] [rbp-350h]
  __int64 *v431; // [rsp+D0h] [rbp-350h]
  _QWORD *v432; // [rsp+D0h] [rbp-350h]
  __int64 v433; // [rsp+D0h] [rbp-350h]
  unsigned __int64 v434; // [rsp+D0h] [rbp-350h]
  __int64 v435; // [rsp+D8h] [rbp-348h]
  __int64 v436; // [rsp+D8h] [rbp-348h]
  __int64 v437; // [rsp+D8h] [rbp-348h]
  _QWORD *v438; // [rsp+D8h] [rbp-348h]
  __int64 *v439; // [rsp+D8h] [rbp-348h]
  __int64 v440; // [rsp+D8h] [rbp-348h]
  __int64 v441; // [rsp+D8h] [rbp-348h]
  __int64 *v442; // [rsp+D8h] [rbp-348h]
  __int64 v443; // [rsp+D8h] [rbp-348h]
  __int64 v444; // [rsp+E0h] [rbp-340h]
  __int64 v445; // [rsp+E0h] [rbp-340h]
  char v446; // [rsp+E0h] [rbp-340h]
  __int64 v447; // [rsp+E0h] [rbp-340h]
  __int64 v448; // [rsp+E0h] [rbp-340h]
  __int64 v449; // [rsp+E0h] [rbp-340h]
  unsigned __int64 *v450; // [rsp+E0h] [rbp-340h]
  unsigned __int64 *v451; // [rsp+E0h] [rbp-340h]
  __int64 v452; // [rsp+E0h] [rbp-340h]
  __int64 v453; // [rsp+E0h] [rbp-340h]
  __int64 v454; // [rsp+E0h] [rbp-340h]
  __int64 v455; // [rsp+E0h] [rbp-340h]
  unsigned int *v456; // [rsp+E0h] [rbp-340h]
  __int64 v457; // [rsp+E0h] [rbp-340h]
  int v458; // [rsp+E0h] [rbp-340h]
  _QWORD *v459; // [rsp+E0h] [rbp-340h]
  unsigned __int64 v460; // [rsp+E8h] [rbp-338h]
  _QWORD *v461; // [rsp+E8h] [rbp-338h]
  __int64 v462; // [rsp+E8h] [rbp-338h]
  unsigned __int64 v463; // [rsp+E8h] [rbp-338h]
  __int64 *v464; // [rsp+E8h] [rbp-338h]
  _QWORD *v465; // [rsp+E8h] [rbp-338h]
  __int64 v466; // [rsp+E8h] [rbp-338h]
  __int64 v467; // [rsp+E8h] [rbp-338h]
  _QWORD *v468; // [rsp+E8h] [rbp-338h]
  __int64 *v469; // [rsp+E8h] [rbp-338h]
  __int64 v470; // [rsp+E8h] [rbp-338h]
  unsigned __int8 *v471; // [rsp+E8h] [rbp-338h]
  __int64 v472; // [rsp+F0h] [rbp-330h]
  __int64 v473; // [rsp+F0h] [rbp-330h]
  __int64 v474; // [rsp+F8h] [rbp-328h]
  int v476; // [rsp+108h] [rbp-318h]
  unsigned int v477; // [rsp+10Ch] [rbp-314h]
  int v478; // [rsp+124h] [rbp-2FCh] BYREF
  __int64 v479; // [rsp+128h] [rbp-2F8h]
  _BYTE *v480; // [rsp+130h] [rbp-2F0h] BYREF
  __int64 v481; // [rsp+138h] [rbp-2E8h]
  unsigned __int64 v482; // [rsp+140h] [rbp-2E0h] BYREF
  char *v483; // [rsp+148h] [rbp-2D8h]
  char *v484; // [rsp+150h] [rbp-2D0h]
  unsigned __int8 v485[32]; // [rsp+160h] [rbp-2C0h] BYREF
  __int16 v486; // [rsp+180h] [rbp-2A0h]
  unsigned __int64 v487; // [rsp+190h] [rbp-290h] BYREF
  __int64 (__fastcall **v488)(); // [rsp+198h] [rbp-288h]
  unsigned __int64 v489; // [rsp+1A0h] [rbp-280h]
  unsigned int v490; // [rsp+1A8h] [rbp-278h]
  __int16 v491; // [rsp+1B0h] [rbp-270h]
  _QWORD *v492; // [rsp+1C0h] [rbp-260h] BYREF
  __int64 *v493; // [rsp+1C8h] [rbp-258h]
  _QWORD v494[3]; // [rsp+1D0h] [rbp-250h] BYREF
  _QWORD *v495; // [rsp+1E8h] [rbp-238h]
  __int64 v496; // [rsp+1F0h] [rbp-230h]
  unsigned int v497; // [rsp+1F8h] [rbp-228h]
  char v498; // [rsp+200h] [rbp-220h]
  __int64 v499; // [rsp+208h] [rbp-218h]
  __int64 *v500; // [rsp+210h] [rbp-210h]
  __int64 v501; // [rsp+218h] [rbp-208h]
  __int16 v502; // [rsp+220h] [rbp-200h]
  __int64 *v503; // [rsp+230h] [rbp-1F0h] BYREF
  __int64 v504; // [rsp+238h] [rbp-1E8h]
  _BYTE v505[128]; // [rsp+240h] [rbp-1E0h] BYREF
  char *v506; // [rsp+2C0h] [rbp-160h] BYREF
  __int64 v507; // [rsp+2C8h] [rbp-158h]
  unsigned __int64 *v508[2]; // [rsp+2D0h] [rbp-150h] BYREF
  __int16 v509; // [rsp+2E0h] [rbp-140h]
  __int64 v510; // [rsp+2F0h] [rbp-130h]
  __int64 v511; // [rsp+2F8h] [rbp-128h]
  __int64 v512; // [rsp+300h] [rbp-120h]
  _QWORD *v513; // [rsp+308h] [rbp-118h]
  void **v514; // [rsp+310h] [rbp-110h]
  void **v515; // [rsp+318h] [rbp-108h]
  __int64 v516; // [rsp+320h] [rbp-100h]
  int v517; // [rsp+328h] [rbp-F8h]
  __int16 v518; // [rsp+32Ch] [rbp-F4h]
  char v519; // [rsp+32Eh] [rbp-F2h]
  __int64 v520; // [rsp+330h] [rbp-F0h]
  __int64 v521; // [rsp+338h] [rbp-E8h]
  void *v522; // [rsp+340h] [rbp-E0h] BYREF
  void *v523; // [rsp+348h] [rbp-D8h] BYREF
  char *v524; // [rsp+350h] [rbp-D0h] BYREF
  size_t v525; // [rsp+358h] [rbp-C8h]
  __int64 v526; // [rsp+360h] [rbp-C0h] BYREF
  _BYTE v527[184]; // [rsp+368h] [rbp-B8h] BYREF

  v8 = *(unsigned __int8 *)(a1 + 4);
  v9 = *(unsigned __int8 *)(a1 + 2);
  if ( (unsigned __int8)v9 <= 0x40u )
  {
    v10 = v8 + 2 * (5 * v9 - 240) - 48;
    if ( v10 > 110 )
      goto LABEL_3;
  }
  else
  {
    v10 = v8 + 100 * (v9 - 65) + 2 * (5 * *(unsigned __int8 *)(a1 + 3) - 240) - 48;
    if ( v10 > 110 )
    {
LABEL_3:
      *(_DWORD *)(a1 + 88) = v10;
      goto LABEL_4;
    }
  }
  *(_DWORD *)(a1 + 88) = 111;
  *(_DWORD *)(a1 + 2) = 707866946;
LABEL_4:
  v410 = *(_BYTE *)(a1 + 1);
  result = sub_B91A00(a2);
  v477 = 0;
  v416 = result;
  if ( (_DWORD)result )
  {
    do
    {
      v12 = sub_B91A10(a2, v477);
      v13 = *(char **)(v12 + 24);
      v414 = (_BYTE *)v12;
      if ( v13 )
        goto LABEL_6;
      v482 = 0;
      v14 = *(_QWORD *)(a1 + 128);
      v503 = (__int64 *)v505;
      v504 = 0x800000000LL;
      v15 = *(_BYTE *)(v14 + 312);
      v483 = 0;
      v484 = 0;
      *(_DWORD *)(a1 + 72) = v15 ^ 1;
      v474 = v14 + 24;
      if ( v14 + 24 == *(_QWORD *)(v14 + 32) )
      {
        v21 = 0;
        goto LABEL_22;
      }
      v403 = 0;
      v16 = *(_QWORD *)(v14 + 32);
      do
      {
        v17 = 0;
        if ( v16 )
          v17 = v16 - 56;
        v18 = (_QWORD *)v17;
        v472 = sub_B92180(v17);
        if ( v472 )
        {
          if ( (unsigned __int8)sub_2425610((__int64)v18, (unsigned int *)&v478) )
          {
            v422 = sub_2428BD0(a1, (__int64)v18);
            if ( v422 )
            {
              if ( (*((_BYTE *)v18 + 2) & 8) != 0 )
              {
                v19 = sub_B2E500((__int64)v18);
                v20 = sub_B2A630(v19);
                if ( v20 > 10 )
                {
                  if ( v20 == 12 )
                    goto LABEL_20;
                }
                else if ( v20 > 6 )
                {
                  goto LABEL_20;
                }
              }
              if ( !(unsigned __int8)sub_B2D610((__int64)v18, 33) )
              {
                v409 = sub_B2D610((__int64)v18, 66);
                if ( !v409 )
                {
                  v415 = *(_DWORD *)(v472 + 16);
                  sub_2427090(&v524, (_BYTE *)v472);
                  v45 = sub_BC1CD0(*a8, &unk_4F8E5A8, (__int64)v18) + 8;
                  v46 = (__int64 *)(sub_BC1CD0(*a5, &unk_4F8D9A8, (__int64)v18) + 8);
                  sub_F429C0((__int64)v18, 0, v45, (__int64)v46, v47, v48);
                  v499 = v45;
                  v492 = v18;
                  v493 = 0;
                  memset(v494, 0, sizeof(v494));
                  v495 = 0;
                  v496 = 0;
                  v497 = 0;
                  v498 = 0;
                  v500 = v46;
                  v501 = 0;
                  v502 = 0;
                  v49 = v18[10];
                  v50 = v49 - 24;
                  if ( !v49 )
                    v50 = 0;
                  v51 = v50;
                  v411 = v50;
                  v52 = sub_FDC4B0((__int64)v46);
                  v53 = 0;
                  if ( !(_BYTE)v502 )
                    v53 = v52;
                  v397 = v53;
                  v396 = sub_242A560((__int64)&v492, 0, v51, v53);
                  v54 = *(_QWORD *)(v51 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v54 == v51 + 48 )
                    goto LABEL_448;
                  if ( !v54 )
LABEL_116:
                    BUG();
                  if ( (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 > 0xA || !(unsigned int)sub_B46E30(v54 - 24) )
                  {
LABEL_448:
                    sub_242A560((__int64)&v492, v411, 0, v397);
                    goto LABEL_131;
                  }
                  v421 = 0;
                  v460 = 0;
                  v401 = v492 + 9;
                  v405 = (_QWORD *)v492[10];
                  if ( v405 == v492 + 9 )
                  {
                    v419 = 0;
                    v435 = 0;
LABEL_358:
                    if ( 2 * v460 < 3 * v421 )
                    {
                      *(_QWORD *)(v435 + 16) = v421;
                      *(_QWORD *)(v419 + 16) = v460 + 1;
                    }
                    goto LABEL_131;
                  }
                  v400 = 0;
                  v419 = 0;
                  v398 = 0;
                  v435 = 0;
                  v392 = v18;
                  v391 = v16;
                  while ( 2 )
                  {
                    if ( !v405 )
                      BUG();
                    v56 = (__int64)(v405 - 3);
                    v57 = v405[3] & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (_QWORD *)v57 == v405 + 3 )
                    {
                      v62 = 0;
                    }
                    else
                    {
                      if ( !v57 )
                        goto LABEL_608;
                      v58 = *(unsigned __int8 *)(v57 - 24);
                      v59 = v57 - 24;
                      v60 = (unsigned int)(v58 - 30) < 0xB;
                      v61 = 0;
                      if ( v60 )
                        v61 = v59;
                      v62 = v61;
                    }
                    v429 = 2;
                    if ( v500 )
                      v429 = sub_FDD860(v500, (__int64)(v405 - 3));
                    v476 = sub_B46E30(v62);
                    if ( v476 )
                    {
                      v63 = 2;
                      v64 = 0;
                      while ( 1 )
                      {
                        v81 = sub_B46EC0(v62, v64);
                        v446 = sub_D0E970(v62, v64, 0);
                        v82 = v429;
                        if ( v446 )
                        {
                          v82 = -1;
                          if ( v429 <= 0x4189374BC6A7EELL )
                            v82 = 1000 * v429;
                        }
                        v425 = v82;
                        if ( v499 )
                        {
                          LODWORD(v506) = sub_FF0430(v499, v56, v81);
                          v63 = sub_F02E20((unsigned int *)&v506, v425);
                          if ( !HIBYTE(v502) )
                            goto LABEL_92;
                        }
                        else if ( !HIBYTE(v502) )
                        {
                          goto LABEL_94;
                        }
                        v65 = *(_DWORD *)(v501 + 24);
                        v66 = *(_QWORD *)(v501 + 8);
                        if ( !v65 )
                          goto LABEL_92;
                        v67 = v65 - 1;
                        v68 = v67 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
                        v69 = (__int64 *)(v66 + 16LL * v68);
                        v70 = *v69;
                        v71 = v69;
                        if ( v81 != *v69 )
                        {
                          v86 = *v69;
                          v87 = v67 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
                          v88 = 1;
                          while ( v86 != -4096 )
                          {
                            v89 = v88 + 1;
                            v87 = v67 & (v88 + v87);
                            v71 = (__int64 *)(v66 + 16LL * v87);
                            v86 = *v71;
                            if ( v81 == *v71 )
                              goto LABEL_90;
                            v88 = v89;
                          }
                          goto LABEL_92;
                        }
LABEL_90:
                        v72 = v71[1];
                        if ( !v72 || v81 != **(_QWORD **)(v72 + 32) )
                          goto LABEL_92;
                        if ( v81 != v70 )
                        {
                          for ( i = 1; ; i = v259 )
                          {
                            if ( v70 == -4096 )
                              BUG();
                            v259 = i + 1;
                            v68 = v67 & (i + v68);
                            v69 = (__int64 *)(v66 + 16LL * v68);
                            v70 = *v69;
                            if ( v81 == *v69 )
                              break;
                          }
                        }
                        v83 = v69[1];
                        if ( *(_BYTE *)(v83 + 84) )
                        {
                          v84 = *(_QWORD **)(v83 + 64);
                          v85 = &v84[*(unsigned int *)(v83 + 76)];
                          if ( v84 != v85 )
                          {
                            while ( v56 != *v84 )
                            {
                              if ( v85 == ++v84 )
                                goto LABEL_122;
                            }
LABEL_92:
                            if ( !v63 )
                              v63 = 1;
                            goto LABEL_94;
                          }
LABEL_122:
                          v63 = 1;
                        }
                        else
                        {
                          if ( sub_C8CA60(v83 + 56, v56) )
                            goto LABEL_92;
                          v63 = 1;
                        }
LABEL_94:
                        v73 = sub_242A560((__int64)&v492, v56, v81, v63);
                        v74 = *(_QWORD *)(v73 + 8);
                        *(_BYTE *)(v73 + 42) = v446;
                        if ( v74 )
                        {
                          v444 = v73;
                          v75 = sub_F35EF0(*(_QWORD *)v73, v74);
                          v73 = v444;
                          if ( v75 )
                            *(_BYTE *)(v444 + 41) = 1;
                        }
                        v76 = v460;
                        if ( v63 > v460 )
                        {
                          v77 = v435;
                          if ( v411 == v56 )
                          {
                            v76 = v63;
                            v77 = v73;
                          }
                          v460 = v76;
                          v435 = v77;
                        }
                        v78 = *(_QWORD *)(v81 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                        if ( v78 != v81 + 48 )
                        {
                          if ( !v78 )
                            goto LABEL_116;
                          v445 = v73;
                          if ( (unsigned int)*(unsigned __int8 *)(v78 - 24) - 30 <= 0xA
                            && !(unsigned int)sub_B46E30(v78 - 24) )
                          {
                            v79 = v421;
                            v80 = v445;
                            if ( v421 < v63 )
                              v79 = v63;
                            else
                              v80 = v419;
                            v419 = v80;
                            v421 = v79;
                          }
                        }
                        if ( v476 == ++v64 )
                          goto LABEL_76;
                      }
                    }
                    v498 = 1;
                    v55 = sub_242A560((__int64)&v492, v56, 0, v429);
                    if ( v429 > v400 )
                    {
                      v400 = v429;
                      v398 = v55;
                    }
LABEL_76:
                    v405 = (_QWORD *)v405[1];
                    if ( v401 != v405 )
                      continue;
                    break;
                  }
                  v18 = v392;
                  v16 = v391;
                  if ( v397 >= v400 && 2 * v397 < 3 * v400 )
                  {
                    *(_QWORD *)(v396 + 16) = v400;
                    *(_QWORD *)(v398 + 16) = v397 + 1;
                  }
                  if ( v460 >= v421 )
                    goto LABEL_358;
LABEL_131:
                  v90 = (unsigned __int64 *)v494[0];
                  v91 = (unsigned __int64 *)v493;
                  sub_2426FD0((__int64 *)&v506, v493, (__int64)(v494[0] - (_QWORD)v493) >> 3);
                  if ( v508[0] )
                    sub_2429450(v91, v90, v508[0], v507);
                  else
                    sub_2427D20(v91, v90);
                  v92 = v508[0];
                  v93 = &v508[0][v507];
                  if ( v508[0] != v93 )
                  {
                    do
                    {
                      if ( *v92 )
                        j_j___libc_free_0(*v92);
                      ++v92;
                    }
                    while ( v93 != v92 );
                    v93 = v508[0];
                  }
                  j_j___libc_free_0((unsigned __int64)v93);
                  v94 = (__int64 **)v493;
                  v95 = (__int64 **)v494[0];
                  if ( v493 != (__int64 *)v494[0] )
                  {
                    do
                    {
                      v96 = *v94;
                      if ( !*((_BYTE *)*v94 + 41) )
                      {
                        if ( *((_BYTE *)v96 + 42) )
                        {
                          v97 = v96[1];
                          if ( v97 )
                          {
                            if ( sub_AA5E90(v97) && (unsigned __int8)sub_24260D0((__int64)&v492, **v94, (*v94)[1]) )
                              *((_BYTE *)*v94 + 40) = 1;
                          }
                        }
                      }
                      ++v94;
                    }
                    while ( v95 != v94 );
                    v98 = (__int64 *)v494[0];
                    if ( v493 != (__int64 *)v494[0] )
                    {
                      v461 = v18;
                      v99 = v493;
                      do
                      {
                        v100 = (__int64 *)*v99;
                        if ( !*(_BYTE *)(*v99 + 41) )
                        {
                          v101 = *v100;
                          if ( v498 || v101 )
                          {
                            if ( (unsigned __int8)sub_24260D0((__int64)&v492, v101, v100[1]) )
                              *(_BYTE *)(*v99 + 40) = 1;
                          }
                        }
                        ++v99;
                      }
                      while ( v98 != v99 );
                      v102 = v493;
                      v18 = v461;
                      v462 = (__int64)(v494[0] - (_QWORD)v493) >> 3;
                      if ( v493 != (__int64 *)v494[0] )
                      {
                        v103 = 0;
                        v436 = v16;
                        while ( 1 )
                        {
                          v104 = v102[v103];
                          if ( a3 )
                            *(_BYTE *)(v104 + 40) = 0;
                          ++v103;
                          *(_QWORD *)(v104 + 24) = sub_242AC80((__int64)&v492, v104, a1 + 368);
                          if ( v462 == v103 )
                            break;
                          v102 = v493;
                        }
                        v16 = v436;
                      }
                    }
                  }
                  v105 = v18[10];
                  v106 = v478;
                  v107 = v105 - 24;
                  if ( !v105 )
                    v107 = 0;
                  v420 = v107;
                  v108 = *(_DWORD *)(a1 + 88);
                  v109 = sub_22077B0(0x140u);
                  v463 = v109;
                  if ( v109 )
                    sub_242E400(v109, (char *)a1, (__int64)v18, v472, v106, v403, v108);
                  v110 = *(unsigned int *)(a1 + 184);
                  v111 = *(_QWORD *)(a1 + 176);
                  v487 = v463;
                  v112 = &v487;
                  v437 = v111;
                  v113 = v110;
                  if ( v110 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 188) )
                  {
                    v428 = -1;
                    if ( v111 <= (unsigned __int64)&v487 && (unsigned __int64)&v487 < v111 + 8 * v110 )
                    {
                      v428 = (__int64)((__int64)&v487 - v111) >> 3;
                      v409 = v422;
                    }
                    v437 = sub_C8D7D0(a1 + 176, a1 + 192, v110 + 1, 8u, (unsigned __int64 *)&v506, v110 + 1);
                    v347 = (_QWORD *)v437;
                    v348 = *(_QWORD **)(a1 + 176);
                    v349 = *(unsigned int *)(a1 + 184);
                    v459 = &v348[v349];
                    if ( v348 != &v348[v349] )
                    {
                      v350 = (_QWORD *)(v437 + v349 * 8);
                      do
                      {
                        if ( v347 )
                        {
                          *v347 = *v348;
                          *v348 = 0;
                        }
                        ++v347;
                        ++v348;
                      }
                      while ( v347 != v350 );
                      v351 = *(_QWORD **)(a1 + 176);
                      v418 = v351;
                      v459 = &v351[*(unsigned int *)(a1 + 184)];
                      if ( v351 != v459 )
                      {
                        v406 = v18;
                        v404 = v16;
                        do
                        {
                          v352 = *--v459;
                          if ( *v459 )
                          {
                            v353 = *(_QWORD *)(v352 + 296);
                            if ( *(_DWORD *)(v352 + 308) )
                            {
                              v354 = *(unsigned int *)(v352 + 304);
                              if ( (_DWORD)v354 )
                              {
                                v355 = 8 * v354;
                                v356 = 0;
                                do
                                {
                                  v357 = *(_QWORD **)(v353 + v356);
                                  if ( v357 != (_QWORD *)-8LL && v357 )
                                  {
                                    v358 = v357[6];
                                    v359 = *v357 + 193LL;
                                    if ( (_QWORD *)v358 != v357 + 8 )
                                      _libc_free(v358);
                                    v360 = v357[2];
                                    if ( (_QWORD *)v360 != v357 + 4 )
                                      j_j___libc_free_0(v360);
                                    sub_C7D6A0((__int64)v357, v359, 8);
                                    v353 = *(_QWORD *)(v352 + 296);
                                  }
                                  v356 += 8;
                                }
                                while ( v355 != v356 );
                              }
                            }
                            _libc_free(v353);
                            v361 = *(_QWORD *)(v352 + 216);
                            if ( v361 != v352 + 232 )
                              _libc_free(v361);
                            v362 = *(_QWORD *)(v352 + 176);
                            v413 = v352 + 80;
                            if ( *(_DWORD *)(v352 + 188) )
                            {
                              v363 = *(unsigned int *)(v352 + 184);
                              if ( (_DWORD)v363 )
                              {
                                v364 = 8 * v363;
                                v365 = 0;
                                do
                                {
                                  v366 = *(_QWORD **)(v362 + v365);
                                  if ( v366 != (_QWORD *)-8LL && v366 )
                                  {
                                    v367 = v366[6];
                                    v368 = *v366 + 193LL;
                                    if ( (_QWORD *)v367 != v366 + 8 )
                                      _libc_free(v367);
                                    v369 = v366[2];
                                    if ( (_QWORD *)v369 != v366 + 4 )
                                      j_j___libc_free_0(v369);
                                    sub_C7D6A0((__int64)v366, v368, 8);
                                    v362 = *(_QWORD *)(v352 + 176);
                                  }
                                  v365 += 8;
                                }
                                while ( v364 != v365 );
                              }
                            }
                            _libc_free(v362);
                            v370 = *(_QWORD *)(v352 + 96);
                            if ( v370 != v352 + 112 )
                              _libc_free(v370);
                            v371 = (unsigned __int64)*(unsigned int *)(v352 + 72) << 7;
                            v433 = *(_QWORD *)(v352 + 64);
                            v372 = v433 + v371;
                            if ( v433 != v433 + v371 )
                            {
                              v402 = v352;
                              do
                              {
                                v373 = *(_DWORD *)(v372 - 12);
                                v372 -= 128LL;
                                if ( v373 && (v374 = *(unsigned int *)(v372 + 112), (_DWORD)v374) )
                                {
                                  v375 = *(_QWORD *)(v372 + 104);
                                  v376 = 8 * v374;
                                  v377 = 0;
                                  do
                                  {
                                    v378 = *(_QWORD **)(v375 + v377);
                                    if ( v378 && v378 != (_QWORD *)-8LL )
                                    {
                                      v379 = v378[6];
                                      v380 = *v378 + 193LL;
                                      if ( (_QWORD *)v379 != v378 + 8 )
                                        _libc_free(v379);
                                      v381 = v378[2];
                                      if ( (_QWORD *)v381 != v378 + 4 )
                                        j_j___libc_free_0(v381);
                                      sub_C7D6A0((__int64)v378, v380, 8);
                                      v375 = *(_QWORD *)(v372 + 104);
                                    }
                                    v377 += 8;
                                  }
                                  while ( v376 != v377 );
                                }
                                else
                                {
                                  v375 = *(_QWORD *)(v372 + 104);
                                }
                                _libc_free(v375);
                                v382 = *(_QWORD *)(v372 + 24);
                                if ( v382 != v372 + 40 )
                                  _libc_free(v382);
                              }
                              while ( v433 != v372 );
                              v352 = v402;
                              v372 = *(_QWORD *)(v402 + 64);
                            }
                            if ( v413 != v372 )
                              _libc_free(v372);
                            sub_C7D6A0(*(_QWORD *)(v352 + 40), 16LL * *(unsigned int *)(v352 + 56), 8);
                            j_j___libc_free_0(v352);
                          }
                        }
                        while ( v418 != v459 );
                        v18 = v406;
                        v16 = v404;
                        v459 = *(_QWORD **)(a1 + 176);
                      }
                    }
                    v383 = (int)v506;
                    if ( (_QWORD *)(a1 + 192) != v459 )
                      _libc_free((unsigned __int64)v459);
                    v110 = *(unsigned int *)(a1 + 184);
                    *(_QWORD *)(a1 + 176) = v437;
                    *(_DWORD *)(a1 + 188) = v383;
                    v112 = (unsigned __int64 *)(v437 + 8 * v428);
                    if ( !v409 )
                      v112 = &v487;
                    v113 = v110;
                  }
                  v114 = (unsigned __int64 *)(v437 + 8 * v110);
                  if ( v114 )
                  {
                    *v114 = *v112;
                    *v112 = 0;
                    v463 = v487;
                    v113 = *(_DWORD *)(a1 + 184);
                  }
                  v115 = (unsigned int)(v113 + 1);
                  *(_DWORD *)(a1 + 184) = v115;
                  if ( v463 )
                  {
                    v116 = *(_QWORD *)(v463 + 296);
                    if ( *(_DWORD *)(v463 + 308) )
                    {
                      v117 = *(unsigned int *)(v463 + 304);
                      if ( (_DWORD)v117 )
                      {
                        v438 = v18;
                        v118 = 0;
                        v447 = 8 * v117;
                        do
                        {
                          v119 = *(_QWORD **)(v116 + v118);
                          if ( v119 != (_QWORD *)-8LL && v119 )
                          {
                            v120 = v119[6];
                            v121 = *v119 + 193LL;
                            if ( (_QWORD *)v120 != v119 + 8 )
                              _libc_free(v120);
                            v122 = v119[2];
                            if ( (_QWORD *)v122 != v119 + 4 )
                              j_j___libc_free_0(v122);
                            sub_C7D6A0((__int64)v119, v121, 8);
                            v116 = *(_QWORD *)(v463 + 296);
                          }
                          v118 += 8;
                        }
                        while ( v447 != v118 );
                        v18 = v438;
                      }
                    }
                    _libc_free(v116);
                    v123 = *(_QWORD *)(v463 + 216);
                    if ( v123 != v463 + 232 )
                      _libc_free(v123);
                    v124 = *(_QWORD *)(v463 + 176);
                    if ( *(_DWORD *)(v463 + 188) )
                    {
                      v306 = *(unsigned int *)(v463 + 184);
                      if ( (_DWORD)v306 )
                      {
                        v432 = v18;
                        v307 = 0;
                        v457 = 8 * v306;
                        do
                        {
                          v308 = *(_QWORD **)(v124 + v307);
                          if ( v308 != (_QWORD *)-8LL && v308 )
                          {
                            v309 = v308[6];
                            v310 = *v308 + 193LL;
                            if ( (_QWORD *)v309 != v308 + 8 )
                              _libc_free(v309);
                            v311 = v308[2];
                            if ( (_QWORD *)v311 != v308 + 4 )
                              j_j___libc_free_0(v311);
                            sub_C7D6A0((__int64)v308, v310, 8);
                            v124 = *(_QWORD *)(v463 + 176);
                          }
                          v307 += 8;
                        }
                        while ( v457 != v307 );
                        v18 = v432;
                      }
                    }
                    _libc_free(v124);
                    v125 = *(_QWORD *)(v463 + 96);
                    if ( v125 != v463 + 112 )
                      _libc_free(v125);
                    v448 = *(_QWORD *)(v463 + 64);
                    v126 = v448 + ((unsigned __int64)*(unsigned int *)(v463 + 72) << 7);
                    if ( v448 != v126 )
                    {
                      v430 = v18;
                      v426 = v16;
                      do
                      {
                        v127 = *(_DWORD *)(v126 - 12);
                        v126 -= 128LL;
                        if ( v127 && (v128 = *(unsigned int *)(v126 + 112), (_DWORD)v128) )
                        {
                          v129 = *(_QWORD *)(v126 + 104);
                          v130 = 8 * v128;
                          v131 = 0;
                          do
                          {
                            v132 = *(_QWORD **)(v129 + v131);
                            if ( v132 && v132 != (_QWORD *)-8LL )
                            {
                              v133 = v132[6];
                              v134 = *v132 + 193LL;
                              if ( (_QWORD *)v133 != v132 + 8 )
                                _libc_free(v133);
                              v135 = v132[2];
                              if ( (_QWORD *)v135 != v132 + 4 )
                                j_j___libc_free_0(v135);
                              sub_C7D6A0((__int64)v132, v134, 8);
                              v129 = *(_QWORD *)(v126 + 104);
                            }
                            v131 += 8;
                          }
                          while ( v130 != v131 );
                        }
                        else
                        {
                          v129 = *(_QWORD *)(v126 + 104);
                        }
                        _libc_free(v129);
                        v136 = *(_QWORD *)(v126 + 24);
                        if ( v136 != v126 + 40 )
                          _libc_free(v136);
                      }
                      while ( v448 != v126 );
                      v18 = v430;
                      v16 = v426;
                      v126 = *(_QWORD *)(v463 + 64);
                    }
                    if ( v463 + 80 != v126 )
                      _libc_free(v126);
                    sub_C7D6A0(*(_QWORD *)(v463 + 40), 16LL * *(unsigned int *)(v463 + 56), 8);
                    j_j___libc_free_0(v463);
                    v115 = *(unsigned int *)(a1 + 184);
                  }
                  v137 = (unsigned __int64 *)v494[0];
                  v464 = v493;
                  v138 = *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * v115 - 8);
                  v423 = v494[0] - (_QWORD)v493;
                  v139 = (__int64)(v494[0] - (_QWORD)v493) >> 5;
                  v427 = (__int64)(v494[0] - (_QWORD)v493) >> 3;
                  if ( v139 <= 0 )
                  {
                    v231 = (__int64)(v494[0] - (_QWORD)v493) >> 3;
                    v140 = v493;
LABEL_349:
                    if ( v231 != 2 )
                    {
                      if ( v231 != 3 )
                      {
                        if ( v231 != 1 )
                          goto LABEL_232;
                        goto LABEL_352;
                      }
                      v388 = *v140;
                      if ( *(_BYTE *)(*v140 + 41) || !*(_BYTE *)(v388 + 40) && !*(_QWORD *)(v388 + 24) )
                        goto LABEL_207;
                      ++v140;
                    }
                    v389 = *v140;
                    if ( *(_BYTE *)(*v140 + 41) || !*(_BYTE *)(v389 + 40) && !*(_QWORD *)(v389 + 24) )
                      goto LABEL_207;
                    ++v140;
LABEL_352:
                    v232 = *v140;
                    if ( *(_BYTE *)(*v140 + 41) || !*(_BYTE *)(v232 + 40) && !*(_QWORD *)(v232 + 24) )
                      goto LABEL_207;
                    goto LABEL_232;
                  }
                  v140 = v493;
                  while ( 1 )
                  {
                    v141 = *v140;
                    if ( *(_BYTE *)(*v140 + 41) || !*(_BYTE *)(v141 + 40) && !*(_QWORD *)(v141 + 24) )
                      break;
                    v227 = v140[1];
                    v228 = v140 + 1;
                    if ( *(_BYTE *)(v227 + 41)
                      || !*(_BYTE *)(v227 + 40) && !*(_QWORD *)(v227 + 24)
                      || (v229 = v140[2], v228 = v140 + 2, *(_BYTE *)(v229 + 41))
                      || !*(_BYTE *)(v229 + 40) && !*(_QWORD *)(v229 + 24)
                      || (v230 = v140[3], v228 = v140 + 3, *(_BYTE *)(v230 + 41))
                      || !*(_BYTE *)(v230 + 40) && !*(_QWORD *)(v230 + 24) )
                    {
                      v140 = v228;
                      break;
                    }
                    v140 += 4;
                    if ( &v493[4 * v139] == v140 )
                    {
                      v231 = (__int64)(v494[0] - (_QWORD)v140) >> 3;
                      goto LABEL_349;
                    }
                  }
LABEL_207:
                  if ( v140 != (__int64 *)v494[0] )
                  {
                    v142 = (unsigned __int64 *)(v140 + 1);
                    if ( (__int64 *)v494[0] == v140 + 1 )
                    {
                      v148 = 0;
                      goto LABEL_225;
                    }
                    v449 = v138;
                    v465 = v18;
                    v143 = (unsigned __int64 *)v140;
                    v144 = (unsigned __int64 *)v494[0];
                    do
                    {
                      v145 = *v142;
                      if ( !*(_BYTE *)(*v142 + 41) && (*(_BYTE *)(v145 + 40) || *(_QWORD *)(v145 + 24)) )
                      {
                        *v142 = 0;
                        v146 = *v143;
                        *v143 = v145;
                        if ( v146 )
                          j_j___libc_free_0(v146);
                        ++v143;
                      }
                      ++v142;
                    }
                    while ( v144 != v142 );
                    v147 = (__int64 *)v144;
                    v140 = (__int64 *)v143;
                    v138 = v449;
                    v18 = v465;
                    v142 = (unsigned __int64 *)v494[0];
                    if ( v147 == v140 )
                    {
                      v137 = (unsigned __int64 *)v494[0];
                      v464 = v493;
                      v423 = v494[0] - (_QWORD)v493;
                      v139 = (__int64)(v494[0] - (_QWORD)v493) >> 5;
                      v427 = (__int64)(v494[0] - (_QWORD)v493) >> 3;
                    }
                    else
                    {
                      v148 = v494[0] - (_QWORD)v147;
                      if ( v147 != (__int64 *)v494[0] && v148 > 0 )
                      {
                        v149 = v147;
                        v439 = v147;
                        v431 = v140;
                        v150 = v148 >> 3;
                        do
                        {
                          v151 = *v149;
                          *v149 = 0;
                          v152 = *v140;
                          *v140 = v151;
                          if ( v152 )
                            j_j___libc_free_0(v152);
                          ++v149;
                          ++v140;
                          --v150;
                        }
                        while ( v150 );
                        v142 = (unsigned __int64 *)v494[0];
                        v18 = v465;
                        v138 = v449;
                        v140 = v431;
                        v148 = v494[0] - (_QWORD)v439;
                      }
LABEL_225:
                      v137 = (unsigned __int64 *)((char *)v140 + v148);
                      v153 = v137;
                      if ( v142 != v137 )
                      {
                        v466 = v138;
                        v450 = v137;
                        do
                        {
                          if ( *v153 )
                            j_j___libc_free_0(*v153);
                          ++v153;
                        }
                        while ( v142 != v153 );
                        v137 = v450;
                        v138 = v466;
                        v494[0] = v450;
                      }
                      v464 = v493;
                      v423 = (char *)v137 - (char *)v493;
                      v139 = ((char *)v137 - (char *)v493) >> 5;
                      v427 = ((char *)v137 - (char *)v493) >> 3;
                    }
                  }
LABEL_232:
                  if ( v139 <= 0 )
                  {
                    v305 = v427;
                    v154 = v464;
LABEL_450:
                    if ( v305 != 2 )
                    {
                      if ( v305 != 3 )
                      {
                        if ( v305 != 1 )
                          goto LABEL_246;
                        goto LABEL_453;
                      }
                      if ( !*(_QWORD *)(*v154 + 24) )
                        goto LABEL_239;
                      ++v154;
                    }
                    if ( !*(_QWORD *)(*v154 + 24) )
                      goto LABEL_239;
                    ++v154;
LABEL_453:
                    if ( !*(_QWORD *)(*v154 + 24) )
                      goto LABEL_239;
                    goto LABEL_246;
                  }
                  v154 = v464;
                  while ( *(_QWORD *)(*v154 + 24) )
                  {
                    if ( !*(_QWORD *)(v154[1] + 24) )
                    {
                      ++v154;
                      break;
                    }
                    if ( !*(_QWORD *)(v154[2] + 24) )
                    {
                      v154 += 2;
                      break;
                    }
                    if ( !*(_QWORD *)(v154[3] + 24) )
                    {
                      v154 += 3;
                      break;
                    }
                    v154 += 4;
                    if ( !--v139 )
                    {
                      v305 = ((char *)v137 - (char *)v154) >> 3;
                      goto LABEL_450;
                    }
                  }
LABEL_239:
                  if ( v154 == (__int64 *)v137 )
                  {
                    v423 = (char *)v154 - (char *)v464;
                    v427 = v154 - v464;
                  }
                  else
                  {
                    v440 = v138;
                    v451 = v137;
                    sub_2426FD0((__int64 *)&v506, v154, ((char *)v137 - (char *)v154) >> 3);
                    v155 = sub_2427A00((unsigned __int64 *)v154, v451, (__int64)v506, v508[0], v507);
                    v156 = v508[0];
                    v157 = v440;
                    v452 = v155;
                    v158 = &v508[0][v507];
                    if ( v508[0] != v158 )
                    {
                      do
                      {
                        if ( *v156 )
                          j_j___libc_free_0(*v156);
                        ++v156;
                      }
                      while ( v158 != v156 );
                      v157 = v440;
                      v158 = v508[0];
                    }
                    v441 = v157;
                    j_j___libc_free_0((unsigned __int64)v158);
                    v423 = v452 - (_QWORD)v464;
                    v138 = v441;
                    v427 = (v452 - (__int64)v464) >> 3;
                    v464 = v493;
                  }
LABEL_246:
                  v159 = v464;
                  if ( v423 )
                  {
                    while ( 2 )
                    {
                      v160 = (_QWORD *)*v159;
                      v161 = v138 + 80;
                      v162 = *(_QWORD *)*v159;
                      if ( !v162 )
                        goto LABEL_253;
                      v163 = *(unsigned int *)(v138 + 56);
                      v164 = *(_QWORD *)(v138 + 40);
                      if ( (_DWORD)v163 )
                      {
                        v165 = (v163 - 1) & (((unsigned int)v162 >> 9) ^ ((unsigned int)v162 >> 4));
                        v166 = (__int64 *)(v164 + 16LL * v165);
                        v167 = *v166;
                        if ( v162 == *v166 )
                        {
LABEL_250:
                          v168 = *(_QWORD *)(v138 + 64);
                          if ( v166 != (__int64 *)(v164 + 16 * v163) )
                          {
                            v169 = ((unsigned __int64)*((unsigned int *)v166 + 2) << 7) + v168;
                            goto LABEL_252;
                          }
LABEL_379:
                          v169 = ((unsigned __int64)*(unsigned int *)(v138 + 72) << 7) + v168;
LABEL_252:
                          v161 = v169 + 8;
LABEL_253:
                          v170 = v160[1];
                          v171 = v138 + 200;
                          if ( !v170 )
                            goto LABEL_259;
                          v172 = *(unsigned int *)(v138 + 56);
                          v173 = *(_QWORD *)(v138 + 40);
                          if ( (_DWORD)v172 )
                          {
                            v174 = (v172 - 1) & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
                            v175 = (__int64 *)(v173 + 16LL * v174);
                            v176 = *v175;
                            if ( v170 == *v175 )
                            {
LABEL_256:
                              v177 = *(_QWORD *)(v138 + 64);
                              if ( v175 != (__int64 *)(v173 + 16 * v172) )
                              {
                                v178 = ((unsigned __int64)*((unsigned int *)v175 + 2) << 7) + v177;
                                goto LABEL_258;
                              }
LABEL_366:
                              v178 = ((unsigned __int64)*(unsigned int *)(v138 + 72) << 7) + v177;
LABEL_258:
                              v171 = v178 + 8;
LABEL_259:
                              ++v159;
                              *((_DWORD *)v160 + 8) = *(_DWORD *)(v161 + 8);
                              *((_DWORD *)v160 + 9) = *(_DWORD *)(v171 + 8);
                              if ( &v464[v427] == v159 )
                                goto LABEL_260;
                              continue;
                            }
                            v233 = 1;
                            while ( v176 != -4096 )
                            {
                              v174 = (v172 - 1) & (v233 + v174);
                              v458 = v233 + 1;
                              v175 = (__int64 *)(v173 + 16LL * v174);
                              v176 = *v175;
                              if ( v170 == *v175 )
                                goto LABEL_256;
                              v233 = v458;
                            }
                          }
                          v177 = *(_QWORD *)(v138 + 64);
                          goto LABEL_366;
                        }
                        v236 = 1;
                        while ( v167 != -4096 )
                        {
                          v312 = v236 + 1;
                          v165 = (v163 - 1) & (v236 + v165);
                          v166 = (__int64 *)(v164 + 16LL * v165);
                          v167 = *v166;
                          if ( v162 == *v166 )
                            goto LABEL_250;
                          v236 = v312;
                        }
                      }
                      break;
                    }
                    v168 = *(_QWORD *)(v138 + 64);
                    goto LABEL_379;
                  }
LABEL_260:
                  v453 = v138;
                  v179 = (unsigned __int64 *)((char *)v464 + v423);
                  sub_2426FD0((__int64 *)&v506, v464, v427);
                  if ( v508[0] )
                    sub_24299B0((unsigned __int64 *)v464, v179, v508[0], v507);
                  else
                    sub_2427F00((unsigned __int64 *)v464, v179);
                  v180 = v453;
                  v181 = v508[0];
                  v182 = &v508[0][v507];
                  if ( v508[0] != v182 )
                  {
                    do
                    {
                      if ( *v181 )
                        j_j___libc_free_0(*v181);
                      ++v181;
                    }
                    while ( v182 != v181 );
                    v180 = v453;
                    v182 = v508[0];
                  }
                  v467 = v180;
                  j_j___libc_free_0((unsigned __int64)v182);
                  v183 = v467;
                  if ( (__int64 *)v494[0] != v493 )
                  {
                    v468 = v18;
                    v184 = v183;
                    v185 = v493;
                    v454 = v16;
                    v186 = (__int64 *)v494[0];
                    while ( 2 )
                    {
                      v187 = (_QWORD *)*v185;
                      v188 = v184 + 80;
                      v189 = *(_QWORD *)*v185;
                      if ( !v189 )
                        goto LABEL_275;
                      v190 = *(unsigned int *)(v184 + 56);
                      v191 = *(_QWORD *)(v184 + 40);
                      if ( (_DWORD)v190 )
                      {
                        v192 = (v190 - 1) & (((unsigned int)v189 >> 9) ^ ((unsigned int)v189 >> 4));
                        v193 = (__int64 *)(v191 + 16LL * v192);
                        v194 = *v193;
                        if ( v189 == *v193 )
                        {
LABEL_272:
                          v195 = *(_QWORD *)(v184 + 64);
                          if ( v193 != (__int64 *)(v191 + 16 * v190) )
                          {
                            v196 = ((unsigned __int64)*((unsigned int *)v193 + 2) << 7) + v195;
                            goto LABEL_274;
                          }
LABEL_370:
                          v196 = ((unsigned __int64)*(unsigned int *)(v184 + 72) << 7) + v195;
LABEL_274:
                          v188 = v196 + 8;
LABEL_275:
                          v197 = v187[1];
                          v198 = v184 + 200;
                          if ( v197 )
                          {
                            v199 = *(unsigned int *)(v184 + 56);
                            v200 = *(_QWORD *)(v184 + 40);
                            if ( (_DWORD)v199 )
                            {
                              v201 = (v199 - 1) & (((unsigned int)v197 >> 9) ^ ((unsigned int)v197 >> 4));
                              v202 = (__int64 *)(v200 + 16LL * v201);
                              v203 = *v202;
                              if ( v197 == *v202 )
                              {
LABEL_278:
                                v204 = *(_QWORD *)(v184 + 64);
                                if ( v202 != (__int64 *)(v200 + 16 * v199) )
                                {
                                  v205 = ((unsigned __int64)*((unsigned int *)v202 + 2) << 7) + v204;
LABEL_280:
                                  v198 = v205 + 8;
                                  goto LABEL_281;
                                }
LABEL_375:
                                v205 = ((unsigned __int64)*(unsigned int *)(v184 + 72) << 7) + v204;
                                goto LABEL_280;
                              }
                              v235 = 1;
                              while ( v203 != -4096 )
                              {
                                v313 = v235 + 1;
                                v201 = (v199 - 1) & (v235 + v201);
                                v202 = (__int64 *)(v200 + 16LL * v201);
                                v203 = *v202;
                                if ( v197 == *v202 )
                                  goto LABEL_278;
                                v235 = v313;
                              }
                            }
                            v204 = *(_QWORD *)(v184 + 64);
                            goto LABEL_375;
                          }
LABEL_281:
                          v206 = *(unsigned int *)(v188 + 28);
                          v207 = v187[3] == 0;
                          v208 = *(unsigned int *)(v188 + 24);
                          v209 = *(_DWORD *)(v188 + 24);
                          if ( v208 >= v206 )
                          {
                            v241 = v207 | v393 & 0xFFFFFFFF00000000LL;
                            v393 = v241;
                            if ( v206 < v208 + 1 )
                            {
                              v434 = v241;
                              v443 = v198;
                              sub_C8D5F0(v188 + 16, (const void *)(v188 + 32), v208 + 1, 0x10u, v198, v241);
                              v208 = *(unsigned int *)(v188 + 24);
                              v241 = v434;
                              v198 = v443;
                            }
                            v242 = (__int64 *)(*(_QWORD *)(v188 + 16) + 16 * v208);
                            *v242 = v198;
                            v242[1] = v241;
                            ++*(_DWORD *)(v188 + 24);
                          }
                          else
                          {
                            v210 = *(_QWORD *)(v188 + 16) + 16 * v208;
                            if ( v210 )
                            {
                              *(_QWORD *)v210 = v198;
                              *(_DWORD *)(v210 + 8) = v207;
                              v209 = *(_DWORD *)(v188 + 24);
                            }
                            *(_DWORD *)(v188 + 24) = v209 + 1;
                          }
                          if ( v186 == ++v185 )
                          {
                            v18 = v468;
                            v16 = v454;
                            v183 = v184;
                            goto LABEL_287;
                          }
                          continue;
                        }
                        v234 = 1;
                        while ( v194 != -4096 )
                        {
                          v314 = v234 + 1;
                          v192 = (v190 - 1) & (v234 + v192);
                          v193 = (__int64 *)(v191 + 16LL * v192);
                          v194 = *v193;
                          if ( v189 == *v193 )
                            goto LABEL_272;
                          v234 = v314;
                        }
                      }
                      break;
                    }
                    v195 = *(_QWORD *)(v184 + 64);
                    goto LABEL_370;
                  }
LABEL_287:
                  if ( (*(_BYTE *)(v472 + 32) & 0x40) != 0 )
                  {
LABEL_288:
                    v211 = (unsigned __int64)*(unsigned int *)(v183 + 72) << 7;
                    v469 = *(__int64 **)(v183 + 64);
                    v442 = (__int64 *)((char *)v469 + v211);
                    if ( v469 != (__int64 *)((char *)v469 + v211) )
                    {
                      v417 = v16;
                      while ( 1 )
                      {
                        v455 = *v469;
                        v212 = v469[3];
                        v213 = v212 + 16LL * *((unsigned int *)v469 + 8);
                        for ( j = v212; v213 != j; j += 16 )
                        {
                          v215 = *(_DWORD *)(*(_QWORD *)j + 8LL);
                          do
                          {
                            while ( 1 )
                            {
                              LOBYTE(v506) = v215;
                              v216 = v483;
                              if ( v483 != v484 )
                                break;
                              sub_C8FB10((__int64)&v482, v483, (char *)&v506);
                              v215 >>= 8;
                              if ( !v215 )
                                goto LABEL_297;
                            }
                            if ( v483 )
                            {
                              *v483 = v215;
                              v216 = v483;
                            }
                            v215 >>= 8;
                            v483 = v216 + 1;
                          }
                          while ( v215 );
LABEL_297:
                          ;
                        }
                        v217 = *(_QWORD **)(v455 + 56);
                        v218 = (_QWORD *)(v455 + 48);
                        if ( v217 != (_QWORD *)(v455 + 48) )
                          break;
LABEL_315:
                        v469 += 16;
                        v415 = 0;
                        if ( v442 == v469 )
                        {
                          v16 = v417;
                          goto LABEL_317;
                        }
                      }
                      while ( v217 )
                      {
                        if ( *((_BYTE *)v217 - 24) == 85
                          && (v220 = *(v217 - 7)) != 0
                          && !*(_BYTE *)v220
                          && *(_QWORD *)(v220 + 24) == v217[7]
                          && (*(_BYTE *)(v220 + 33) & 0x20) != 0
                          && (unsigned int)(*(_DWORD *)(v220 + 36) - 68) <= 3 )
                        {
                          v217 = (_QWORD *)v217[1];
                          if ( v218 == v217 )
                            goto LABEL_315;
                        }
                        else
                        {
                          if ( v217[3] )
                          {
                            if ( (unsigned int)sub_B10CE0((__int64)(v217 + 3)) )
                            {
                              if ( !(unsigned __int8)sub_B10EB0((__int64)(v217 + 3))
                                && (unsigned int)sub_B10CE0((__int64)(v217 + 3)) != v415 )
                              {
                                v415 = sub_B10CE0((__int64)(v217 + 3));
                                v219 = (_BYTE *)sub_B10D00((__int64)(v217 + 3));
                                if ( *v219 != 20 && v472 == sub_AE7A60(v219) )
                                {
                                  v456 = (unsigned int *)sub_24257B0(v469 + 1, v524, v525);
                                  v237 = sub_B10CE0((__int64)(v217 + 3));
                                  v238 = v456;
                                  v239 = v237;
                                  v240 = v456[12];
                                  if ( v240 + 1 > (unsigned __int64)v456[13] )
                                  {
                                    v412 = v239;
                                    sub_C8D5F0((__int64)(v456 + 10), v456 + 14, v240 + 1, 4u, v239, (__int64)v456);
                                    v238 = v456;
                                    LODWORD(v239) = v412;
                                    v240 = v456[12];
                                  }
                                  *(_DWORD *)(*((_QWORD *)v238 + 5) + 4 * v240) = v239;
                                  ++v238[12];
                                }
                              }
                            }
                          }
                          v217 = (_QWORD *)v217[1];
                          if ( v218 == v217 )
                            goto LABEL_315;
                        }
                      }
LABEL_608:
                      BUG();
                    }
LABEL_317:
                    if ( !v410 )
                      goto LABEL_318;
                    v260 = sub_B92180((__int64)v18);
                    v261 = (__int64 *)sub_BCB2E0(*(_QWORD **)(a1 + 168));
                    v262 = sub_BCD420(v261, v427);
                    v263 = sub_AD6530((__int64)v262, v427);
                    v506 = "__llvm_gcov_ctr";
                    v509 = 259;
                    BYTE4(v487) = 0;
                    v471 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
                    if ( v471 )
                      sub_B30000((__int64)v471, *(_QWORD *)(a1 + 128), v262, 0, 7, v263, (__int64)&v506, 0, 0, v487, 0);
                    if ( *(_DWORD *)(*(_QWORD *)(a1 + 128) + 284LL) == 8 )
                      sub_B31A00((__int64)v471, (__int64)"__llvm_gcov_ctr_section", 23);
                    v266 = (unsigned int)v504;
                    v267 = v504;
                    if ( (unsigned int)v504 >= (unsigned __int64)HIDWORD(v504) )
                    {
                      if ( HIDWORD(v504) < (unsigned __int64)(unsigned int)v504 + 1 )
                      {
                        sub_C8D5F0((__int64)&v503, v505, (unsigned int)v504 + 1LL, 0x10u, v264, v265);
                        v266 = (unsigned int)v504;
                      }
                      v385 = &v503[2 * v266];
                      v385[1] = v260;
                      *v385 = (__int64)v471;
                      LODWORD(v504) = v504 + 1;
                    }
                    else
                    {
                      v268 = &v503[2 * (unsigned int)v504];
                      if ( v268 )
                      {
                        v268[1] = v260;
                        *v268 = (__int64)v471;
                        v267 = v504;
                      }
                      LODWORD(v504) = v267 + 1;
                    }
                    if ( !v423 )
                    {
LABEL_318:
                      v221 = v497;
                      if ( v497 )
                      {
                        v222 = v495;
                        v223 = &v495[2 * v497];
                        do
                        {
                          if ( *v222 != -8192 && *v222 != -4096 )
                          {
                            v224 = v222[1];
                            if ( v224 )
                              j_j___libc_free_0(v224);
                          }
                          v222 += 2;
                        }
                        while ( v223 != v222 );
                        v221 = v497;
                      }
                      sub_C7D6A0((__int64)v495, 16 * v221, 8);
                      v225 = (unsigned __int64 *)v494[0];
                      v226 = (unsigned __int64 *)v493;
                      if ( (__int64 *)v494[0] != v493 )
                      {
                        do
                        {
                          if ( *v226 )
                            j_j___libc_free_0(*v226);
                          ++v226;
                        }
                        while ( v225 != v226 );
                        v226 = (unsigned __int64 *)v493;
                      }
                      if ( v226 )
                        j_j___libc_free_0((unsigned __int64)v226);
                      if ( v524 != v527 )
                        _libc_free((unsigned __int64)v524);
                      ++v403;
                      goto LABEL_20;
                    }
                    v473 = 0;
                    while ( 2 )
                    {
                      v269 = v493[v473];
                      v271 = sub_AA5190(*(_QWORD *)(v269 + 24));
                      if ( v271 )
                      {
                        v272 = v270;
                        v273 = HIBYTE(v270);
                      }
                      else
                      {
                        v273 = 0;
                        v272 = 0;
                      }
                      v274 = *(_QWORD *)(v269 + 24);
                      v275 = (_QWORD *)sub_AA48A0(v274);
                      v516 = 0;
                      v513 = v275;
                      v506 = (char *)v508;
                      v507 = 0x200000000LL;
                      v517 = 0;
                      v514 = &v522;
                      v518 = 512;
                      v515 = &v523;
                      v519 = 7;
                      v522 = &unk_49DA100;
                      v520 = 0;
                      v510 = v274;
                      v523 = &unk_49DA0B0;
                      LOBYTE(v276) = v272;
                      HIBYTE(v276) = v273;
                      v511 = v271;
                      v521 = 0;
                      LOWORD(v512) = v276;
                      if ( v271 != v274 + 48 )
                      {
                        v277 = v271 - 24;
                        if ( !v271 )
                          v277 = 0;
                        v278 = *(_QWORD *)sub_B46C60(v277);
                        v487 = v278;
                        if ( !v278 || (sub_B96E90((__int64)&v487, v278, 1), (v280 = v487) == 0) )
                        {
                          sub_93FB40((__int64)&v506, 0);
                          v280 = v487;
                          goto LABEL_492;
                        }
                        v281 = (unsigned int *)v506;
                        v282 = v507;
                        v283 = (unsigned int *)&v506[16 * (unsigned int)v507];
                        if ( v506 != (char *)v283 )
                        {
                          while ( *v281 )
                          {
                            v281 += 4;
                            if ( v283 == v281 )
                              goto LABEL_488;
                          }
                          *((_QWORD *)v281 + 1) = v487;
                          goto LABEL_427;
                        }
LABEL_488:
                        if ( (unsigned int)v507 >= (unsigned __int64)HIDWORD(v507) )
                        {
                          v384 = v394 & 0xFFFFFFFF00000000LL;
                          v394 &= 0xFFFFFFFF00000000LL;
                          if ( HIDWORD(v507) < (unsigned __int64)(unsigned int)v507 + 1 )
                          {
                            sub_C8D5F0(
                              (__int64)&v506,
                              v508,
                              (unsigned int)v507 + 1LL,
                              0x10u,
                              (unsigned int)v507 + 1LL,
                              v279);
                            v283 = (unsigned int *)&v506[16 * (unsigned int)v507];
                          }
                          *(_QWORD *)v283 = v384;
                          *((_QWORD *)v283 + 1) = v280;
                          v280 = v487;
                          LODWORD(v507) = v507 + 1;
                        }
                        else
                        {
                          if ( v283 )
                          {
                            *v283 = 0;
                            *((_QWORD *)v283 + 1) = v280;
                            v282 = v507;
                            v280 = v487;
                          }
                          LODWORD(v507) = v282 + 1;
                        }
LABEL_492:
                        if ( v280 )
LABEL_427:
                          sub_B91220((__int64)&v487, v280);
                        v275 = v513;
                      }
                      v486 = 257;
                      v284 = *((_QWORD *)v471 + 3);
                      v285 = sub_BCB2E0(v275);
                      v480 = (_BYTE *)sub_ACD640(v285, 0, 0);
                      v286 = sub_BCB2E0(v513);
                      v481 = sub_ACD640(v286, v473, 0);
                      v287 = (__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))*((_QWORD *)*v514 + 8);
                      if ( v287 != sub_920540 )
                      {
                        v291 = v287((__int64)v514, v284, v471, &v480, 2, 3);
                        goto LABEL_434;
                      }
                      if ( !sub_BCEA30(v284) && *v471 <= 0x15u )
                      {
                        v288 = sub_24254A0(&v480, (__int64)&v482);
                        if ( v290 == v288 )
                        {
                          LOBYTE(v491) = 0;
                          v291 = sub_AD9FD0(v284, v471, v289, 2, 3u, (__int64)&v487, 0);
                          if ( (_BYTE)v491 )
                          {
                            LOBYTE(v491) = 0;
                            if ( v490 > 0x40 && v489 )
                              j_j___libc_free_0_0(v489);
                            if ( (unsigned int)v488 > 0x40 && v487 )
                              j_j___libc_free_0_0(v487);
                          }
LABEL_434:
                          if ( v291 )
                          {
LABEL_435:
                            if ( *(_BYTE *)(a1 + 7) )
                            {
                              v292 = sub_BCB2E0(v513);
                              v293 = -1;
                              v294 = sub_ACD640(v292, 1, 0);
                              v295 = sub_AA4E30(v510);
                              v296 = sub_9208B0(v295, *(_QWORD *)(v294 + 8));
                              v488 = v297;
                              v487 = (unsigned __int64)(v296 + 7) >> 3;
                              v298 = sub_CA1930(&v487);
                              if ( v298 )
                              {
                                _BitScanReverse64(&v298, v298);
                                v293 = 63 - (v298 ^ 0x3F);
                              }
                              v491 = 257;
                              v299 = sub_BD2C40(80, unk_3F148C0);
                              v300 = (__int64)v299;
                              if ( v299 )
                                sub_B4D750((__int64)v299, 1, v291, v294, v293, 2, 1, 0, 0);
                              v301 = v300;
                              (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64))*v515 + 2))(
                                v515,
                                v300,
                                &v487,
                                v511,
                                v512);
                              v302 = (unsigned int *)v506;
                              v303 = (unsigned int *)&v506[16 * (unsigned int)v507];
                              if ( v506 != (char *)v303 )
                              {
                                do
                                {
                                  v304 = *((_QWORD *)v302 + 1);
                                  v301 = *v302;
                                  v302 += 4;
                                  sub_B99FD0(v300, v301, v304);
                                }
                                while ( v303 != v302 );
                              }
                            }
                            else
                            {
                              v315 = sub_BCB2E0(v513);
                              v486 = 259;
                              *(_QWORD *)v485 = "gcov_ctr";
                              v316 = sub_AA4E30(v510);
                              v317 = sub_AE5020(v316, v315);
                              v491 = 257;
                              v318 = v317;
                              v319 = sub_BD2C40(80, unk_3F10A14);
                              v321 = v319;
                              if ( v319 )
                              {
                                sub_B4D190((__int64)v319, v315, v291, (__int64)&v487, 0, v318, 0, 0);
                                v320 = v390;
                              }
                              v322 = (__int64)v321;
                              (*((void (__fastcall **)(void **, _BYTE *, unsigned __int8 *, __int64, __int64, __int64))*v515
                               + 2))(
                                v515,
                                v321,
                                v485,
                                v511,
                                v512,
                                v320);
                              v323 = (unsigned int *)v506;
                              v324 = (unsigned int *)&v506[16 * (unsigned int)v507];
                              if ( v506 != (char *)v324 )
                              {
                                do
                                {
                                  v325 = *((_QWORD *)v323 + 1);
                                  v322 = *v323;
                                  v323 += 4;
                                  sub_B99FD0((__int64)v321, v322, v325);
                                }
                                while ( v324 != v323 );
                              }
                              sub_B9D8E0((__int64)v321, v322);
                              v491 = 257;
                              v326 = sub_BCB2E0(v513);
                              v327 = (_BYTE *)sub_ACD640(v326, 1, 0);
                              v328 = sub_929C50((unsigned int **)&v506, v321, v327, (__int64)&v487, 0, 0);
                              v329 = sub_AA4E30(v510);
                              v330 = sub_AE5020(v329, *(_QWORD *)(v328 + 8));
                              HIBYTE(v331) = HIBYTE(v407);
                              v491 = 257;
                              LOBYTE(v331) = v330;
                              v407 = v331;
                              v332 = sub_BD2C40(80, unk_3F10A10);
                              v300 = (__int64)v332;
                              if ( v332 )
                                sub_B4D3C0((__int64)v332, v328, v291, 0, v407, v333, 0, 0);
                              v301 = v300;
                              (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64))*v515 + 2))(
                                v515,
                                v300,
                                &v487,
                                v511,
                                v512);
                              v334 = (unsigned int *)v506;
                              v335 = (unsigned int *)&v506[16 * (unsigned int)v507];
                              if ( v506 != (char *)v335 )
                              {
                                do
                                {
                                  v336 = *((_QWORD *)v334 + 1);
                                  v301 = *v334;
                                  v334 += 4;
                                  sub_B99FD0(v300, v301, v336);
                                }
                                while ( v335 != v334 );
                              }
                            }
                            sub_B9D8E0(v300, v301);
                            nullsub_61();
                            v522 = &unk_49DA100;
                            nullsub_63();
                            if ( v506 != (char *)v508 )
                              _libc_free((unsigned __int64)v506);
                            if ( ++v473 == v427 )
                              goto LABEL_318;
                            continue;
                          }
                        }
                      }
                      break;
                    }
                    v491 = 257;
                    v291 = (__int64)sub_BD2C40(88, 3u);
                    if ( v291 )
                    {
                      v337 = *((_QWORD *)v471 + 1);
                      v395 = v395 & 0xE0000000 | 3;
                      if ( (unsigned int)*(unsigned __int8 *)(v337 + 8) - 17 <= 1 )
                        goto LABEL_498;
                      v342 = *((_QWORD *)v480 + 1);
                      v343 = *(unsigned __int8 *)(v342 + 8);
                      if ( v343 == 17 )
                        goto LABEL_583;
                      if ( v343 == 18 )
                        goto LABEL_507;
                      v342 = *(_QWORD *)(v481 + 8);
                      v344 = *(unsigned __int8 *)(v342 + 8);
                      if ( v344 == 17 )
                      {
LABEL_583:
                        v345 = 0;
LABEL_508:
                        v346 = *(_DWORD *)(v342 + 32);
                        BYTE4(v479) = v345;
                        LODWORD(v479) = v346;
                        v337 = sub_BCE1B0((__int64 *)v337, v479);
                      }
                      else if ( v344 == 18 )
                      {
LABEL_507:
                        v345 = 1;
                        goto LABEL_508;
                      }
LABEL_498:
                      sub_B44260(v291, v337, 34, v395, 0, 0);
                      *(_QWORD *)(v291 + 72) = v284;
                      *(_QWORD *)(v291 + 80) = sub_B4DC50(v284, (__int64)&v480, 2);
                      sub_B4D9A0(v291, (__int64)v471, (__int64 *)&v480, 2, (__int64)&v487);
                    }
                    sub_B4DDE0(v291, 3);
                    (*((void (__fastcall **)(void **, __int64, unsigned __int8 *, __int64, __int64))*v515 + 2))(
                      v515,
                      v291,
                      v485,
                      v511,
                      v512);
                    v338 = (unsigned int *)v506;
                    v339 = (unsigned int *)&v506[16 * (unsigned int)v507];
                    if ( v506 != (char *)v339 )
                    {
                      do
                      {
                        v340 = *((_QWORD *)v338 + 1);
                        v341 = *v338;
                        v338 += 4;
                        sub_B99FD0(v291, v341, v340);
                      }
                      while ( v339 != v338 );
                    }
                    goto LABEL_435;
                  }
                  v246 = *(_DWORD *)(v183 + 56);
                  v247 = *(_QWORD *)(v183 + 40);
                  if ( v246 )
                  {
                    v248 = (v246 - 1) & (((unsigned int)v420 >> 9) ^ ((unsigned int)v420 >> 4));
                    v249 = (__int64 *)(v247 + 16LL * v248);
                    v250 = *v249;
                    if ( v420 == *v249 )
                    {
LABEL_394:
                      v251 = *(_QWORD *)(v183 + 64);
                      if ( v249 != (__int64 *)(v247 + 16LL * v246) )
                      {
                        v252 = v251 + ((unsigned __int64)*((unsigned int *)v249 + 2) << 7);
LABEL_396:
                        v470 = v183;
                        v253 = sub_24257B0((__int64 *)(v252 + 8), v524, v525);
                        v183 = v470;
                        v256 = v253;
                        v257 = *(unsigned int *)(v253 + 48);
                        if ( v257 + 1 > (unsigned __int64)*(unsigned int *)(v256 + 52) )
                        {
                          sub_C8D5F0(v256 + 40, (const void *)(v256 + 56), v257 + 1, 4u, v254, v255);
                          v257 = *(unsigned int *)(v256 + 48);
                          v183 = v470;
                        }
                        *(_DWORD *)(*(_QWORD *)(v256 + 40) + 4 * v257) = v415;
                        ++*(_DWORD *)(v256 + 48);
                        goto LABEL_288;
                      }
LABEL_510:
                      v252 = v251 + ((unsigned __int64)*(unsigned int *)(v183 + 72) << 7);
                      goto LABEL_396;
                    }
                    v386 = 1;
                    while ( v250 != -4096 )
                    {
                      v387 = v386 + 1;
                      v248 = (v246 - 1) & (v386 + v248);
                      v249 = (__int64 *)(v247 + 16LL * v248);
                      v250 = *v249;
                      if ( v420 == *v249 )
                        goto LABEL_394;
                      v386 = v387;
                    }
                  }
                  v251 = *(_QWORD *)(v183 + 64);
                  goto LABEL_510;
                }
              }
            }
          }
        }
LABEL_20:
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v474 != v16 );
      v21 = (char *)v482;
      v13 = &v483[-v482];
LABEL_22:
      LODWORD(v480) = -1;
      sub_1098F90((unsigned int *)&v480, v21, (__int64)v13);
      v24 = (unsigned int)v480;
      v25 = *(unsigned int *)(a1 + 104);
      if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 108) )
      {
        sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v25 + 1, 4u, v22, v23);
        v25 = *(unsigned int *)(a1 + 104);
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4 * v25) = v24;
      ++*(_DWORD *)(a1 + 104);
      if ( *(_BYTE *)a1 )
      {
        LODWORD(v487) = 0;
        v488 = sub_2241E40();
        sub_2427400((__int64 *)&v506, *(_QWORD *)(a1 + 128), v414, 0);
        sub_CB7060((__int64)&v524, v506, v507, (__int64)&v487, 0);
        if ( v506 != (char *)v508 )
          j_j___libc_free_0((unsigned __int64)v506);
        if ( (_DWORD)v487 )
        {
          v26 = *(_QWORD *)(a1 + 168);
          (*((void (__fastcall **)(_QWORD **))*v488 + 4))(&v492);
          v27 = (unsigned int **)&v506;
          v506 = "failed to open coverage notes file for writing: ";
          v508[0] = (unsigned __int64 *)&v492;
          v509 = 1027;
          sub_B6ECE0(v26, (__int64)&v506);
          if ( v492 != v494 )
          {
            v27 = (unsigned int **)(v494[0] + 1LL);
            j_j___libc_free_0((unsigned __int64)v492);
          }
          sub_CB5B00((int *)&v524, (__int64)v27);
          if ( v503 != (__int64 *)v505 )
            _libc_free((unsigned __int64)v503);
          if ( v482 )
            j_j___libc_free_0(v482);
          goto LABEL_6;
        }
        *(_QWORD *)(a1 + 80) = &v524;
        if ( *(_DWORD *)(a1 + 72) )
        {
          sub_CB6200((__int64)&v524, "oncg", 4u);
          v243 = a1 + 6;
          v244 = v485;
          do
          {
            v245 = *(_BYTE *)(v243 - 1);
            ++v244;
            --v243;
            *(v244 - 1) = v245;
          }
          while ( v244 != &v485[4] );
          sub_CB6200((__int64)&v524, v485, 4u);
        }
        else
        {
          sub_CB6200((__int64)&v524, "gcno", 4u);
          sub_CB6200((__int64)&v524, (unsigned __int8 *)(a1 + 2), 4u);
        }
        v28 = v24;
        v29 = *(_QWORD *)(a1 + 80);
        if ( *(_DWORD *)(a1 + 72) != 1 )
          v28 = _byteswap_ulong(v24);
        LODWORD(v506) = v28;
        sub_CB6200(v29, (unsigned __int8 *)&v506, 4u);
        v30 = 1;
        v31 = *(_QWORD *)(a1 + 80);
        if ( *(_DWORD *)(a1 + 72) != 1 )
          v30 = (unsigned int)&loc_1000000;
        LODWORD(v506) = v30;
        sub_CB6200(v31, (unsigned __int8 *)&v506, 4u);
        sub_CB6200(*(_QWORD *)(a1 + 80), (unsigned __int8 *)".", 1u);
        sub_CB6C70(*(_QWORD *)(a1 + 80), 3u);
        v32 = *(_QWORD *)(a1 + 80);
        LODWORD(v506) = 0;
        sub_CB6200(v32, (unsigned __int8 *)&v506, 4u);
        v33 = *(unsigned int ***)(a1 + 176);
        v34 = &v33[*(unsigned int *)(a1 + 184)];
        while ( v34 != v33 )
        {
          v35 = *v33++;
          sub_2429A80(v35, v24);
        }
        LODWORD(v506) = 0;
        sub_CB6200(*(_QWORD *)(a1 + 80), (unsigned __int8 *)&v506, 4u);
        v36 = *(_QWORD *)(a1 + 80);
        LODWORD(v506) = 0;
        sub_CB6200(v36, (unsigned __int8 *)&v506, 4u);
        sub_CB7080((__int64)&v524, (__int64)&v506);
        sub_CB5B00((int *)&v524, (__int64)&v506);
      }
      if ( v410 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 128) + 284LL) == 8 )
        {
          v37 = (unsigned int *)sub_242AEB0(a1, (__int64)v503, v504);
          v38 = sub_2427F80(a1, v503, (unsigned int)v504);
          v39 = **(__int64 ***)(a1 + 128);
          v492 = (_QWORD *)sub_BCE3C0(v39, 0);
          v493 = (__int64 *)sub_BCE3C0(v39, 0);
          v40 = (__int64 **)sub_BD0B90(v39, &v492, 2, 0);
          v507 = v38;
          v41 = v40;
          v506 = (char *)v37;
          v524 = "__llvm_covinit_functions";
          v527[9] = 1;
          v527[8] = 3;
          BYTE4(v487) = 0;
          v42 = sub_BD2C40(88, unk_3F0FAE8);
          if ( v42 )
            sub_B30000((__int64)v42, *(_QWORD *)(a1 + 128), v41, 0, 8, 0, (__int64)&v524, 0, 0, v487, 0);
          v43 = sub_AD24A0(v41, (__int64 *)&v506, 2);
          sub_B30160((__int64)v42, v43);
          v44 = v42[32];
          v42[32] = v44 & 0xCF;
          if ( (v44 & 0xFu) - 7 <= 1 )
            v42[33] |= 0x40u;
          sub_ED12E0((__int64)&v524, 13, *(_DWORD *)(*(_QWORD *)(a1 + 128) + 284LL), 1u);
          sub_B31A00((__int64)v42, (__int64)v524, v525);
          if ( v524 != (char *)&v526 )
            j_j___libc_free_0((unsigned __int64)v524);
          sub_B2F770((__int64)v42, 3u);
          v42[80] |= 1u;
        }
        else
        {
          sub_242DBD0(a1, (__int64)&v503);
        }
      }
      if ( v503 != (__int64 *)v505 )
        _libc_free((unsigned __int64)v503);
      if ( v482 )
        j_j___libc_free_0(v482);
      v410 = 0;
LABEL_6:
      result = ++v477;
    }
    while ( v416 != v477 );
  }
  return result;
}
