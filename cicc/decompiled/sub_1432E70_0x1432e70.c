// Function: sub_1432E70
// Address: 0x1432e70
//
__int64 __fastcall sub_1432E70(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  _QWORD *v5; // rax
  _BYTE *v6; // r12
  __int64 v7; // r13
  _QWORD *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // r13
  unsigned __int8 v13; // al
  __int64 v14; // rax
  __int64 *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rdx
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // r14
  __int64 ii; // rdx
  __int64 v44; // rsi
  __int64 v45; // rax
  _QWORD *v46; // rcx
  _QWORD *v47; // rax
  int v48; // r11d
  int v49; // r8d
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rbx
  __int64 v53; // r14
  __int64 v54; // rdi
  __int64 *v55; // rbx
  __int64 *v56; // r14
  __int64 v57; // rdi
  __int64 v58; // rsi
  size_t v59; // rdx
  __int64 v60; // rbx
  __int64 v61; // r12
  __int64 v62; // rdi
  size_t v63; // rdx
  __int64 *v64; // rbx
  __int64 *v65; // r12
  __int64 v66; // rdi
  _QWORD *jj; // r13
  __int64 v68; // r14
  _QWORD *kk; // r13
  __int64 v70; // rsi
  _QWORD *v71; // rax
  _BYTE *v72; // r14
  _QWORD *v73; // r13
  __int64 v74; // rax
  __int64 v75; // rdx
  bool v76; // bl
  _QWORD *v77; // rax
  __int64 v78; // r12
  __int64 *v79; // rax
  __int64 v80; // r15
  _QWORD *v81; // r14
  char v83; // dl
  unsigned __int64 v84; // rax
  __int64 v85; // r12
  char v86; // cl
  __int64 v87; // rdi
  __int64 v88; // r14
  char v89; // al
  unsigned __int8 v90; // al
  __int64 *v91; // r12
  __int64 *v92; // rbx
  __int64 *v93; // rbx
  _QWORD *v94; // rax
  char *v95; // rbx
  char *v96; // r12
  _BYTE *v97; // rbx
  unsigned int v98; // edx
  unsigned __int64 v99; // r12
  __int64 v100; // rax
  __int16 v101; // si
  __int16 v102; // dx
  unsigned __int64 v103; // r13
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rdx
  int v106; // eax
  __int64 v107; // r12
  __int64 v108; // rdx
  char v109; // r8
  char v110; // al
  __int64 v111; // rax
  __int64 v112; // rcx
  __int64 v113; // rax
  __int64 v114; // r12
  __int64 v115; // r14
  __int64 *v116; // rbx
  __int64 *v117; // rbx
  _QWORD *v118; // rax
  char *v119; // rbx
  char *v120; // r15
  __int64 v121; // r14
  __int64 v122; // rdx
  bool v123; // zf
  _QWORD *v124; // rax
  char *v125; // rbx
  char *v126; // r12
  _BYTE *v127; // rbx
  __int64 v128; // r12
  char v129; // al
  unsigned int v130; // edx
  unsigned int v131; // eax
  unsigned int v132; // ecx
  _QWORD *v133; // rdi
  unsigned int v134; // eax
  __int64 v135; // rax
  unsigned __int64 v136; // rax
  unsigned __int64 v137; // rax
  __int64 v138; // r14
  int v139; // eax
  __int64 v140; // r14
  _QWORD *v141; // rax
  _QWORD *n; // rdx
  unsigned int v143; // ecx
  char *v144; // rdi
  unsigned int v145; // edx
  int v146; // edx
  unsigned __int64 v147; // rax
  int v148; // ebx
  __int64 v149; // r12
  _QWORD *v150; // rax
  _QWORD *m; // rdx
  unsigned int v152; // r8d
  _QWORD *v153; // r10
  unsigned int v154; // r8d
  int v155; // r8d
  unsigned __int64 v156; // r8
  __int64 v157; // r8
  int v158; // eax
  __int64 v159; // r8
  _QWORD *v160; // rax
  _QWORD *v161; // rcx
  unsigned int v162; // ecx
  _QWORD *v163; // rdi
  unsigned int v164; // eax
  __int64 v165; // rax
  unsigned __int64 v166; // rax
  unsigned __int64 v167; // rax
  int v168; // ebx
  __int64 v169; // r12
  _QWORD *v170; // rax
  _QWORD *k; // rdx
  unsigned int v172; // ecx
  _QWORD *v173; // rdi
  unsigned int v174; // eax
  __int64 v175; // rax
  unsigned __int64 v176; // rax
  unsigned __int64 v177; // rax
  int v178; // ebx
  __int64 v179; // r12
  _QWORD *v180; // rax
  _QWORD *j; // rdx
  __int64 *v182; // rbx
  __int64 *v183; // r14
  __int64 *v184; // rbx
  __int64 *v185; // r14
  __int64 *v186; // rbx
  _QWORD *v187; // rax
  char *v188; // r12
  char *v189; // r13
  _BYTE *v190; // rdi
  unsigned int v191; // edx
  int v192; // edx
  char v193; // al
  _QWORD *v194; // rax
  _QWORD *v195; // r15
  unsigned __int64 *v196; // r14
  size_t v197; // r13
  unsigned int v198; // ecx
  size_t *v199; // rax
  size_t v200; // rdi
  _QWORD *v201; // rax
  _QWORD *v202; // rsi
  _QWORD *v203; // rcx
  _QWORD *v204; // rax
  _QWORD *v205; // r14
  _QWORD *v206; // r13
  size_t v207; // r9
  __int64 v208; // rax
  _QWORD *v209; // rax
  _QWORD *v210; // r13
  _QWORD *v211; // rbx
  __int64 v212; // r14
  __int64 v213; // rdi
  __int64 *v214; // rax
  __int64 *v215; // r13
  __int64 *v216; // r12
  __int64 v217; // r15
  __int64 *v218; // rbx
  __int64 *v219; // r14
  __int64 v220; // rdi
  __int64 v221; // rax
  __int64 v222; // rax
  void *v223; // rdi
  unsigned int v224; // eax
  __int64 v225; // rdx
  unsigned __int64 v226; // rdi
  __int64 v227; // rax
  __int64 v228; // rdi
  __int64 v229; // rdi
  unsigned __int64 *v230; // r12
  unsigned __int64 *v231; // rbx
  unsigned __int64 v232; // rdi
  unsigned __int64 *v233; // r12
  _QWORD *v234; // r12
  __int64 v235; // r14
  __int64 v236; // rax
  unsigned __int64 *v237; // rax
  unsigned __int64 *v238; // rbx
  unsigned __int64 *v239; // r12
  unsigned __int64 v240; // rdi
  unsigned __int64 *v241; // rbx
  unsigned __int64 v242; // rdi
  int v243; // r11d
  size_t *v244; // r10
  int v245; // ecx
  unsigned int v246; // edx
  size_t v247; // r11
  int v248; // edi
  size_t *v249; // rsi
  int v250; // edi
  unsigned int v251; // edx
  size_t v252; // r9
  char *v253; // rax
  _QWORD *v254; // rax
  _QWORD *v255; // rax
  _QWORD *v256; // rax
  _QWORD *v257; // rax
  __int64 v258; // [rsp+8h] [rbp-738h]
  int v259; // [rsp+14h] [rbp-72Ch]
  unsigned int v260; // [rsp+20h] [rbp-720h]
  __int64 v261; // [rsp+20h] [rbp-720h]
  __int64 v262; // [rsp+20h] [rbp-720h]
  __int64 v263; // [rsp+28h] [rbp-718h]
  __int64 v264; // [rsp+28h] [rbp-718h]
  __int64 v265; // [rsp+28h] [rbp-718h]
  __int64 v266; // [rsp+28h] [rbp-718h]
  __int64 v268; // [rsp+48h] [rbp-6F8h]
  __int64 v269; // [rsp+58h] [rbp-6E8h]
  __int64 v270; // [rsp+60h] [rbp-6E0h]
  __int64 v271; // [rsp+68h] [rbp-6D8h]
  __int64 v272; // [rsp+70h] [rbp-6D0h]
  __int64 v273; // [rsp+78h] [rbp-6C8h]
  __int64 v274; // [rsp+78h] [rbp-6C8h]
  __int64 *v275; // [rsp+80h] [rbp-6C0h]
  int v277; // [rsp+90h] [rbp-6B0h]
  int v278; // [rsp+94h] [rbp-6ACh]
  __int64 v279; // [rsp+98h] [rbp-6A8h]
  __int64 v280; // [rsp+A0h] [rbp-6A0h]
  __int64 v281; // [rsp+A0h] [rbp-6A0h]
  _QWORD *v282; // [rsp+A8h] [rbp-698h]
  __int64 v283; // [rsp+B0h] [rbp-690h]
  __int64 *v284; // [rsp+B8h] [rbp-688h]
  __int64 *v285; // [rsp+C0h] [rbp-680h]
  char v286; // [rsp+C8h] [rbp-678h]
  char *v287; // [rsp+C8h] [rbp-678h]
  __int64 *v288; // [rsp+D0h] [rbp-670h]
  __int64 v289; // [rsp+D0h] [rbp-670h]
  __int64 v290; // [rsp+D8h] [rbp-668h]
  bool v291; // [rsp+D8h] [rbp-668h]
  char v292; // [rsp+E0h] [rbp-660h]
  unsigned int v293; // [rsp+E0h] [rbp-660h]
  unsigned __int64 v294; // [rsp+E0h] [rbp-660h]
  __int64 i; // [rsp+E0h] [rbp-660h]
  __int64 v296; // [rsp+E8h] [rbp-658h]
  char v297; // [rsp+E8h] [rbp-658h]
  char v298; // [rsp+F8h] [rbp-648h]
  unsigned __int64 v299; // [rsp+F8h] [rbp-648h]
  __int64 *v300; // [rsp+F8h] [rbp-648h]
  _QWORD *v301; // [rsp+F8h] [rbp-648h]
  unsigned __int64 v302; // [rsp+F8h] [rbp-648h]
  char v303; // [rsp+102h] [rbp-63Eh]
  char v304; // [rsp+103h] [rbp-63Dh]
  int v305; // [rsp+104h] [rbp-63Ch]
  char v306; // [rsp+108h] [rbp-638h]
  char v307; // [rsp+110h] [rbp-630h]
  _QWORD *v308; // [rsp+110h] [rbp-630h]
  __int64 v309; // [rsp+110h] [rbp-630h]
  __int64 v311; // [rsp+120h] [rbp-620h]
  unsigned __int8 v312; // [rsp+120h] [rbp-620h]
  _QWORD *v313; // [rsp+120h] [rbp-620h]
  unsigned __int64 v314; // [rsp+120h] [rbp-620h]
  _QWORD *v315; // [rsp+120h] [rbp-620h]
  __int64 v316; // [rsp+128h] [rbp-618h]
  __int64 v317; // [rsp+128h] [rbp-618h]
  unsigned __int64 *v319; // [rsp+128h] [rbp-618h]
  size_t v320; // [rsp+128h] [rbp-618h]
  char v321; // [rsp+147h] [rbp-5F9h] BYREF
  __int64 v322; // [rsp+148h] [rbp-5F8h] BYREF
  __int64 v323[2]; // [rsp+150h] [rbp-5F0h] BYREF
  __int64 v324; // [rsp+160h] [rbp-5E0h]
  _QWORD v325[2]; // [rsp+170h] [rbp-5D0h] BYREF
  __int64 v326; // [rsp+180h] [rbp-5C0h]
  _QWORD v327[2]; // [rsp+190h] [rbp-5B0h] BYREF
  __int64 v328; // [rsp+1A0h] [rbp-5A0h]
  __int64 v329; // [rsp+1B0h] [rbp-590h] BYREF
  __int64 v330; // [rsp+1B8h] [rbp-588h]
  __int64 v331; // [rsp+1C0h] [rbp-580h]
  __int64 v332[2]; // [rsp+1D0h] [rbp-570h] BYREF
  __int64 v333; // [rsp+1E0h] [rbp-560h]
  __int64 v334; // [rsp+1F0h] [rbp-550h] BYREF
  __int64 v335; // [rsp+1F8h] [rbp-548h]
  __int64 v336; // [rsp+200h] [rbp-540h]
  __int64 v337; // [rsp+208h] [rbp-538h]
  __int64 *v338; // [rsp+210h] [rbp-530h] BYREF
  __int64 *v339; // [rsp+218h] [rbp-528h]
  __int64 v340; // [rsp+220h] [rbp-520h] BYREF
  int v341; // [rsp+228h] [rbp-518h]
  __int64 v342; // [rsp+230h] [rbp-510h] BYREF
  _QWORD *v343; // [rsp+238h] [rbp-508h]
  __int64 v344; // [rsp+240h] [rbp-500h]
  unsigned int v345; // [rsp+248h] [rbp-4F8h]
  __int64 v346; // [rsp+250h] [rbp-4F0h]
  __int64 v347; // [rsp+258h] [rbp-4E8h]
  __int64 v348; // [rsp+260h] [rbp-4E0h]
  __int64 v349; // [rsp+270h] [rbp-4D0h] BYREF
  _QWORD *v350; // [rsp+278h] [rbp-4C8h]
  __int64 v351; // [rsp+280h] [rbp-4C0h]
  __int64 v352; // [rsp+288h] [rbp-4B8h]
  __int64 v353; // [rsp+290h] [rbp-4B0h]
  __int64 v354; // [rsp+298h] [rbp-4A8h]
  __int64 v355; // [rsp+2A0h] [rbp-4A0h]
  __int64 v356; // [rsp+2B0h] [rbp-490h] BYREF
  void *s; // [rsp+2B8h] [rbp-488h]
  __int64 v358; // [rsp+2C0h] [rbp-480h]
  __int64 v359; // [rsp+2C8h] [rbp-478h]
  __int64 v360; // [rsp+2D0h] [rbp-470h]
  __int64 v361; // [rsp+2D8h] [rbp-468h]
  __int64 v362; // [rsp+2E0h] [rbp-460h]
  __int64 v363; // [rsp+2F0h] [rbp-450h] BYREF
  _QWORD *v364; // [rsp+2F8h] [rbp-448h]
  __int64 v365; // [rsp+300h] [rbp-440h]
  __int64 v366; // [rsp+308h] [rbp-438h]
  __int64 v367; // [rsp+310h] [rbp-430h]
  __int64 v368; // [rsp+318h] [rbp-428h]
  __int64 v369; // [rsp+320h] [rbp-420h]
  __int64 v370; // [rsp+330h] [rbp-410h] BYREF
  _QWORD *v371; // [rsp+338h] [rbp-408h]
  __int64 v372; // [rsp+340h] [rbp-400h]
  __int64 v373; // [rsp+348h] [rbp-3F8h]
  __int64 v374; // [rsp+350h] [rbp-3F0h]
  __int64 v375; // [rsp+358h] [rbp-3E8h]
  __int64 v376; // [rsp+360h] [rbp-3E0h]
  size_t v377; // [rsp+370h] [rbp-3D0h] BYREF
  __int64 v378; // [rsp+378h] [rbp-3C8h] BYREF
  __int64 v379; // [rsp+380h] [rbp-3C0h]
  __int64 v380; // [rsp+388h] [rbp-3B8h]
  __int64 *v381; // [rsp+390h] [rbp-3B0h]
  __int64 *v382; // [rsp+398h] [rbp-3A8h]
  __int64 v383; // [rsp+3A0h] [rbp-3A0h]
  size_t v384; // [rsp+3B0h] [rbp-390h] BYREF
  __int64 v385; // [rsp+3B8h] [rbp-388h] BYREF
  __int64 v386; // [rsp+3C0h] [rbp-380h]
  __int64 v387; // [rsp+3C8h] [rbp-378h]
  __int64 v388; // [rsp+3D0h] [rbp-370h]
  __int64 v389; // [rsp+3D8h] [rbp-368h]
  char *v390; // [rsp+3E0h] [rbp-360h]
  __int64 v391; // [rsp+3F0h] [rbp-350h] BYREF
  _BYTE *v392; // [rsp+3F8h] [rbp-348h]
  _BYTE *v393; // [rsp+400h] [rbp-340h]
  __int64 v394; // [rsp+408h] [rbp-338h]
  int v395; // [rsp+410h] [rbp-330h]
  _BYTE v396[72]; // [rsp+418h] [rbp-328h] BYREF
  __int64 v397; // [rsp+460h] [rbp-2E0h] BYREF
  _BYTE *v398; // [rsp+468h] [rbp-2D8h]
  _BYTE *v399; // [rsp+470h] [rbp-2D0h]
  __int64 v400; // [rsp+478h] [rbp-2C8h]
  int v401; // [rsp+480h] [rbp-2C0h]
  _BYTE v402[72]; // [rsp+488h] [rbp-2B8h] BYREF
  unsigned __int64 *v403; // [rsp+4D0h] [rbp-270h] BYREF
  _QWORD *v404; // [rsp+4D8h] [rbp-268h]
  unsigned __int64 v405[2]; // [rsp+4E0h] [rbp-260h] BYREF
  __int64 *v406; // [rsp+4F0h] [rbp-250h]
  _QWORD v407[2]; // [rsp+4F8h] [rbp-248h] BYREF
  unsigned __int64 v408; // [rsp+508h] [rbp-238h]
  unsigned __int64 v409; // [rsp+510h] [rbp-230h]
  unsigned __int64 *v410; // [rsp+518h] [rbp-228h]
  unsigned int v411; // [rsp+520h] [rbp-220h]
  char v412; // [rsp+528h] [rbp-218h] BYREF
  unsigned __int64 *v413; // [rsp+548h] [rbp-1F8h]
  unsigned int v414; // [rsp+550h] [rbp-1F0h]
  __int64 v415; // [rsp+558h] [rbp-1E8h] BYREF
  __m128i v416; // [rsp+570h] [rbp-1D0h] BYREF
  char *v417; // [rsp+580h] [rbp-1C0h] BYREF
  char *v418; // [rsp+588h] [rbp-1B8h]
  _QWORD *v419; // [rsp+590h] [rbp-1B0h]
  __int64 v420; // [rsp+598h] [rbp-1A8h]
  __int64 v421; // [rsp+5A0h] [rbp-1A0h]
  __int64 v422; // [rsp+5A8h] [rbp-198h]
  __int64 v423; // [rsp+5B0h] [rbp-190h]
  __int64 v424; // [rsp+5B8h] [rbp-188h]
  _BYTE *v425; // [rsp+5C0h] [rbp-180h]
  _BYTE *v426; // [rsp+5C8h] [rbp-178h]
  __int64 v427; // [rsp+5D0h] [rbp-170h]
  int v428; // [rsp+5D8h] [rbp-168h]
  _BYTE v429[128]; // [rsp+5E0h] [rbp-160h] BYREF
  __int64 v430; // [rsp+660h] [rbp-E0h]
  _BYTE *v431; // [rsp+668h] [rbp-D8h]
  _BYTE *v432; // [rsp+670h] [rbp-D0h]
  __int64 v433; // [rsp+678h] [rbp-C8h]
  int v434; // [rsp+680h] [rbp-C0h]
  _BYTE v435[184]; // [rsp+688h] [rbp-B8h] BYREF

  v4 = a1;
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  v268 = a1 + 8;
  *(_QWORD *)(a1 + 64) = 0x2800000000LL;
  *(_QWORD *)(a1 + 104) = a1 + 88;
  *(_QWORD *)(a1 + 112) = a1 + 88;
  *(_QWORD *)(a1 + 152) = a1 + 136;
  *(_QWORD *)(a1 + 160) = a1 + 136;
  *(_WORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_BYTE *)(a1 + 178) = 1;
  *(_DWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = a1 + 192;
  *(_QWORD *)(a1 + 216) = a1 + 192;
  *(_QWORD *)(a1 + 256) = a1 + 240;
  *(_QWORD *)(a1 + 264) = a1 + 240;
  *(_QWORD *)(a1 + 296) = a1 + 312;
  *(_QWORD *)(a1 + 304) = 0x400000000LL;
  *(_QWORD *)(a1 + 344) = a1 + 360;
  *(_QWORD *)(a1 + 384) = a1 + 280;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 1;
  v392 = v396;
  v393 = v396;
  v391 = 0;
  v394 = 8;
  v395 = 0;
  v397 = 0;
  v398 = v402;
  v399 = v402;
  v400 = 8;
  v401 = 0;
  sub_1633E30(a2, &v397, 0);
  sub_1633E30(a2, &v397, 1);
  v5 = v399;
  v334 = 0;
  v335 = 0;
  v336 = 0;
  v337 = 0;
  if ( v399 == v398 )
    v6 = &v399[8 * HIDWORD(v400)];
  else
    v6 = &v399[8 * (unsigned int)v400];
  if ( v399 != v6 )
  {
    while ( 1 )
    {
      v7 = *v5;
      v8 = v5;
      if ( *v5 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v6 == (_BYTE *)++v5 )
        goto LABEL_6;
    }
    if ( v6 != (_BYTE *)v5 )
    {
      while ( 1 )
      {
        if ( (*(_BYTE *)(v7 + 32) & 0xFu) - 7 > 1 )
          goto LABEL_375;
        v194 = v392;
        if ( v393 != v392 )
          break;
        v202 = &v392[8 * HIDWORD(v394)];
        if ( v392 == (_BYTE *)v202 )
        {
LABEL_470:
          if ( HIDWORD(v394) >= (unsigned int)v394 )
            break;
          ++HIDWORD(v394);
          *v202 = v7;
          ++v391;
        }
        else
        {
          v203 = 0;
          while ( *v194 != v7 )
          {
            if ( *v194 == -2 )
              v203 = v194;
            if ( v202 == ++v194 )
            {
              if ( !v203 )
                goto LABEL_470;
              *v203 = v7;
              --v395;
              ++v391;
              break;
            }
          }
        }
LABEL_371:
        sub_15E4EB0(&v403);
        v195 = v404;
        v196 = v403;
        sub_16C1840(&v416);
        sub_16C1A90(&v416, v196, v195);
        sub_16C1AA0(&v416, &v384);
        v197 = v384;
        if ( v403 != v405 )
          j_j___libc_free_0(v403, v405[0] + 1);
        if ( (_DWORD)v337 )
        {
          v198 = (v337 - 1) & (37 * v197);
          v199 = (size_t *)(v335 + 8LL * v198);
          v200 = *v199;
          if ( v197 == *v199 )
            goto LABEL_375;
          v243 = 1;
          v244 = 0;
          while ( v200 != -1 )
          {
            if ( v200 == -2 && !v244 )
              v244 = v199;
            v198 = (v337 - 1) & (v243 + v198);
            v199 = (size_t *)(v335 + 8LL * v198);
            v200 = *v199;
            if ( v197 == *v199 )
              goto LABEL_375;
            ++v243;
          }
          if ( v244 )
            v199 = v244;
          ++v334;
          v245 = v336 + 1;
          if ( 4 * ((int)v336 + 1) < (unsigned int)(3 * v337) )
          {
            if ( (int)v337 - HIDWORD(v336) - v245 > (unsigned int)v337 >> 3 )
              goto LABEL_478;
            sub_142F750((__int64)&v334, v337);
            if ( !(_DWORD)v337 )
            {
LABEL_545:
              LODWORD(v336) = v336 + 1;
              BUG();
            }
            v250 = 1;
            v249 = 0;
            v251 = (v337 - 1) & (37 * v197);
            v245 = v336 + 1;
            v199 = (size_t *)(v335 + 8LL * v251);
            v252 = *v199;
            if ( v197 == *v199 )
              goto LABEL_478;
            while ( v252 != -1 )
            {
              if ( v252 == -2 && !v249 )
                v249 = v199;
              v251 = (v337 - 1) & (v250 + v251);
              v199 = (size_t *)(v335 + 8LL * v251);
              v252 = *v199;
              if ( v197 == *v199 )
                goto LABEL_478;
              ++v250;
            }
            goto LABEL_501;
          }
        }
        else
        {
          ++v334;
        }
        sub_142F750((__int64)&v334, 2 * v337);
        if ( !(_DWORD)v337 )
          goto LABEL_545;
        v246 = (v337 - 1) & (37 * v197);
        v245 = v336 + 1;
        v199 = (size_t *)(v335 + 8LL * v246);
        v247 = *v199;
        if ( v197 == *v199 )
          goto LABEL_478;
        v248 = 1;
        v249 = 0;
        while ( v247 != -1 )
        {
          if ( v247 == -2 && !v249 )
            v249 = v199;
          v246 = (v337 - 1) & (v248 + v246);
          v199 = (size_t *)(v335 + 8LL * v246);
          v247 = *v199;
          if ( v197 == *v199 )
            goto LABEL_478;
          ++v248;
        }
LABEL_501:
        if ( v249 )
          v199 = v249;
LABEL_478:
        LODWORD(v336) = v245;
        if ( *v199 != -1 )
          --HIDWORD(v336);
        *v199 = v197;
LABEL_375:
        v201 = v8 + 1;
        if ( v8 + 1 != (_QWORD *)v6 )
        {
          while ( 1 )
          {
            v7 = *v201;
            v8 = v201;
            if ( *v201 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v6 == (_BYTE *)++v201 )
              goto LABEL_378;
          }
          if ( v6 != (_BYTE *)v201 )
            continue;
        }
LABEL_378:
        v4 = a1;
        goto LABEL_6;
      }
      sub_16CCBA0(&v391, v7);
      goto LABEL_371;
    }
  }
LABEL_6:
  v321 = 0;
  if ( a2[12] )
  {
    v418 = (char *)v4;
    v416.m128i_i64[0] = (__int64)&v321;
    v416.m128i_i64[1] = (__int64)a2;
    v417 = (char *)&v334;
    sub_168D470(a2, sub_14306A0, &v416);
  }
  v282 = (_QWORD *)a2[4];
  if ( a2 + 3 != v282 )
  {
    v316 = v4;
    while ( 1 )
    {
      v9 = 0;
      if ( v282 )
        v9 = (__int64)(v282 - 7);
      v283 = v9;
      v10 = v9;
      if ( (unsigned __int8)sub_15E4F60(v9) )
        goto LABEL_125;
      if ( *(_QWORD *)(a3 + 16) )
      {
        v275 = 0;
        v288 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(a3 + 24))(a3, v10);
        goto LABEL_15;
      }
      sub_15E44B0(v283);
      if ( !v192 )
      {
        v275 = 0;
        v288 = 0;
        goto LABEL_15;
      }
      v418 = 0;
      v416.m128i_i64[1] = 0x100000000LL;
      v416.m128i_i64[0] = (__int64)&v417;
      v423 = v283;
      v419 = 0;
      v420 = 0;
      LODWORD(v421) = 0;
      LOBYTE(v424) = 0;
      HIDWORD(v424) = 0;
      sub_15D3930(&v416);
      sub_14019E0((__int64)&v403, (__int64)&v416);
      if ( (_DWORD)v421 )
      {
        v210 = v419;
        v211 = &v419[2 * (unsigned int)v421];
        do
        {
          if ( *v210 != -8 && *v210 != -16 )
          {
            v212 = v210[1];
            if ( v212 )
            {
              v213 = *(_QWORD *)(v212 + 24);
              if ( v213 )
                j_j___libc_free_0(v213, *(_QWORD *)(v212 + 40) - v213);
              j_j___libc_free_0(v212, 56);
            }
          }
          v210 += 2;
        }
        while ( v211 != v210 );
      }
      j___libc_free_0(v419);
      if ( (char **)v416.m128i_i64[0] != &v417 )
        _libc_free(v416.m128i_u64[0]);
      v416 = 0u;
      v425 = v429;
      v426 = v429;
      v431 = v435;
      v432 = v435;
      v417 = 0;
      v418 = 0;
      v419 = 0;
      v420 = 0;
      v421 = 0;
      v422 = 0;
      v424 = 0;
      v427 = 16;
      v428 = 0;
      v430 = 0;
      v433 = 16;
      v434 = 0;
      sub_137CAE0((__int64)&v416, (__int64 *)v283, (__int64)&v403, 0);
      v214 = (__int64 *)sub_22077B0(8);
      v288 = v214;
      if ( v214 )
        sub_13702A0(v214, (const void *)v283, (__int64)&v416, (__int64)&v403);
      if ( v432 != v431 )
        _libc_free((unsigned __int64)v432);
      if ( v426 != v425 )
        _libc_free((unsigned __int64)v426);
      j___libc_free_0(v420);
      if ( (_DWORD)v418 )
      {
        v234 = (_QWORD *)v416.m128i_i64[1];
        v378 = 2;
        v379 = 0;
        v235 = v416.m128i_i64[1] + 40LL * (unsigned int)v418;
        v380 = -8;
        v377 = (size_t)&unk_49E8A80;
        v381 = 0;
        v385 = 2;
        v386 = 0;
        v387 = -16;
        v384 = (size_t)&unk_49E8A80;
        v388 = 0;
        do
        {
          v236 = v234[3];
          *v234 = &unk_49EE2B0;
          if ( v236 != 0 && v236 != -8 && v236 != -16 )
            sub_1649B30(v234 + 1);
          v234 += 5;
        }
        while ( (_QWORD *)v235 != v234 );
        v384 = (size_t)&unk_49EE2B0;
        if ( v387 != -8 && v387 != 0 && v387 != -16 )
          sub_1649B30(&v385);
        v377 = (size_t)&unk_49EE2B0;
        if ( v380 != 0 && v380 != -8 && v380 != -16 )
          sub_1649B30(&v378);
      }
      j___libc_free_0(v416.m128i_i64[1]);
      sub_142D890((__int64)&v403);
      v215 = (__int64 *)v407[0];
      v216 = v406;
      if ( v406 != (__int64 *)v407[0] )
      {
        do
        {
          v217 = *v216;
          v218 = *(__int64 **)(*v216 + 8);
          v219 = *(__int64 **)(*v216 + 16);
          if ( v218 == v219 )
          {
            *(_BYTE *)(v217 + 160) = 1;
          }
          else
          {
            do
            {
              v220 = *v218++;
              sub_13FACC0(v220);
            }
            while ( v219 != v218 );
            *(_BYTE *)(v217 + 160) = 1;
            v221 = *(_QWORD *)(v217 + 8);
            if ( v221 != *(_QWORD *)(v217 + 16) )
              *(_QWORD *)(v217 + 16) = v221;
          }
          v222 = *(_QWORD *)(v217 + 32);
          if ( v222 != *(_QWORD *)(v217 + 40) )
            *(_QWORD *)(v217 + 40) = v222;
          ++*(_QWORD *)(v217 + 56);
          v223 = *(void **)(v217 + 72);
          if ( v223 == *(void **)(v217 + 64) )
          {
            *(_QWORD *)v217 = 0;
          }
          else
          {
            v224 = 4 * (*(_DWORD *)(v217 + 84) - *(_DWORD *)(v217 + 88));
            v225 = *(unsigned int *)(v217 + 80);
            if ( v224 < 0x20 )
              v224 = 32;
            if ( (unsigned int)v225 > v224 )
              sub_16CC920(v217 + 56);
            else
              memset(v223, -1, 8 * v225);
            v226 = *(_QWORD *)(v217 + 72);
            v227 = *(_QWORD *)(v217 + 64);
            *(_QWORD *)v217 = 0;
            if ( v226 != v227 )
              _libc_free(v226);
          }
          v228 = *(_QWORD *)(v217 + 32);
          if ( v228 )
            j_j___libc_free_0(v228, *(_QWORD *)(v217 + 48) - v228);
          v229 = *(_QWORD *)(v217 + 8);
          if ( v229 )
            j_j___libc_free_0(v229, *(_QWORD *)(v217 + 24) - v229);
          ++v216;
        }
        while ( v215 != v216 );
        if ( v406 != (__int64 *)v407[0] )
          v407[0] = v406;
      }
      v230 = v413;
      v231 = &v413[2 * v414];
      if ( v413 != v231 )
      {
        do
        {
          v232 = *v230;
          v230 += 2;
          _libc_free(v232);
        }
        while ( v230 != v231 );
      }
      v414 = 0;
      if ( v411 )
      {
        v237 = v410;
        v415 = 0;
        v238 = &v410[v411];
        v239 = v410 + 1;
        v408 = *v410;
        v409 = v408 + 4096;
        if ( v238 != v410 + 1 )
        {
          do
          {
            v240 = *v239++;
            _libc_free(v240);
          }
          while ( v238 != v239 );
          v237 = v410;
        }
        v411 = 1;
        _libc_free(*v237);
        v241 = v413;
        v233 = &v413[2 * v414];
        if ( v413 == v233 )
          goto LABEL_443;
        do
        {
          v242 = *v241;
          v241 += 2;
          _libc_free(v242);
        }
        while ( v233 != v241 );
      }
      v233 = v413;
LABEL_443:
      if ( v233 != (unsigned __int64 *)&v415 )
        _libc_free((unsigned __int64)v233);
      if ( v410 != (unsigned __int64 *)&v412 )
        _libc_free((unsigned __int64)v410);
      if ( v406 )
        j_j___libc_free_0(v406, v407[1] - (_QWORD)v406);
      j___libc_free_0(v404);
      v275 = v288;
LABEL_15:
      v286 = 1;
      if ( HIDWORD(v394) == v395 )
        v286 = v321;
      v342 = 0;
      v343 = 0;
      v344 = 0;
      v345 = 0;
      v346 = 0;
      v347 = 0;
      v348 = 0;
      v349 = 0;
      v350 = 0;
      v351 = 0;
      v352 = 0;
      v353 = 0;
      v354 = 0;
      v355 = 0;
      v356 = 0;
      s = 0;
      v358 = 0;
      v359 = 0;
      v360 = 0;
      v361 = 0;
      v362 = 0;
      v363 = 0;
      v364 = 0;
      v365 = 0;
      v366 = 0;
      v367 = 0;
      v368 = 0;
      v369 = 0;
      v370 = 0;
      v371 = 0;
      v372 = 0;
      v373 = 0;
      v374 = 0;
      v375 = 0;
      v376 = 0;
      v377 = 0;
      v378 = 0;
      v379 = 0;
      v380 = 0;
      v381 = 0;
      v382 = 0;
      v383 = 0;
      v384 = 0;
      v385 = 0;
      v386 = 0;
      v387 = 0;
      v388 = 0;
      v389 = 0;
      v390 = 0;
      sub_14DE650(&v322);
      v404 = v407;
      v405[0] = (unsigned __int64)v407;
      v403 = 0;
      v405[1] = 8;
      LODWORD(v406) = 0;
      sub_1430880(v316, v283, (__int64)&v349, (__int64)&v403);
      v304 = 0;
      v305 = 0;
      v296 = *(_QWORD *)(v283 + 80);
      if ( v296 != v283 + 72 )
      {
        while ( 1 )
        {
          if ( !v296 )
            BUG();
          v11 = *(_QWORD *)(v296 + 24);
          v311 = v296 + 16;
          if ( v11 != v296 + 16 )
            break;
LABEL_31:
          v296 = *(_QWORD *)(v296 + 8);
          if ( v283 + 72 == v296 )
            goto LABEL_32;
        }
        while ( 1 )
        {
LABEL_25:
          if ( !v11 )
            BUG();
          if ( *(_BYTE *)(v11 - 8) != 78 )
            break;
          v14 = *(_QWORD *)(v11 - 48);
          if ( *(_BYTE *)(v14 + 16)
            || (*(_BYTE *)(v14 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v14 + 36) - 35) > 3 )
          {
            break;
          }
          v11 = *(_QWORD *)(v11 + 8);
          if ( v311 == v11 )
            goto LABEL_31;
        }
        v12 = v11 - 24;
        ++v305;
        sub_1430880(v316, v11 - 24, (__int64)&v349, (__int64)&v403);
        v13 = *(_BYTE *)(v11 - 8);
        if ( v13 <= 0x17u )
          goto LABEL_24;
        if ( v13 != 78 )
        {
          if ( v13 == 29 )
          {
            v83 = v12 & 0xFB;
            v84 = v12 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              v85 = 0;
              goto LABEL_167;
            }
          }
LABEL_24:
          v11 = *(_QWORD *)(v11 + 8);
          if ( v311 == v11 )
            goto LABEL_31;
          goto LABEL_25;
        }
        v83 = v12 | 4;
        v84 = v12 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_24;
        if ( v286 )
        {
          v85 = v11 - 24;
          v86 = v304;
          if ( *(_BYTE *)(*(_QWORD *)(v11 - 48) + 16LL) == 20 )
            v86 = v286;
          v304 = v86;
        }
        else
        {
          v85 = v11 - 24;
        }
LABEL_167:
        if ( (v83 & 4) != 0 )
          v87 = *(_QWORD *)(v84 - 24);
        else
          v87 = *(_QWORD *)(v84 - 72);
        if ( !*(_BYTE *)(v87 + 16) || (v88 = sub_1649F00(v87), v89 = *(_BYTE *)(v88 + 16), v87 = v88, !v89) )
        {
          v88 = v87;
LABEL_172:
          if ( v85 && (*(_BYTE *)(v87 + 33) & 0x20) != 0 )
          {
            sub_14325C0(v85, (__int64)&v356, (__int64)&v363, (__int64)&v370, (__int64)&v377, (__int64)&v384);
          }
          else
          {
            sub_1441B50(&v329, a4, v11 - 24, v288);
            v90 = 0;
            if ( (_BYTE)v330 )
            {
              if ( a4 )
              {
                v107 = v329;
                v109 = sub_1441CD0(a4, v329);
                v90 = 3;
                if ( !v109 )
                  v90 = ((unsigned __int8)sub_1441D60(a4, v107, v108) == 0) + 1;
              }
            }
            if ( dword_4F99BC0 )
            {
              v291 = 0;
              v293 = 1;
            }
            else
            {
              v293 = v90;
              v291 = v90 == 0 && v288 != 0;
            }
            sub_15E4EB0(&v338);
            v91 = v339;
            v92 = v338;
            sub_16C1840(&v416);
            sub_16C1A90(&v416, v92, v91);
            sub_16C1AA0(&v416, v332);
            v93 = (__int64 *)v332[0];
            if ( v338 != &v340 )
              j_j___libc_free_0(v338, v340 + 1);
            v338 = v93;
            if ( *(_BYTE *)(v316 + 178) )
            {
              v416.m128i_i64[0] = 0;
            }
            else
            {
              v416.m128i_i64[1] = 0;
              v416.m128i_i64[0] = (__int64)byte_3F871B3;
            }
            v417 = 0;
            v418 = 0;
            v419 = 0;
            v94 = sub_142DA40((_QWORD *)v316, (unsigned __int64 *)&v338, &v416);
            v95 = v418;
            v96 = v417;
            v308 = v94;
            v299 = (unsigned __int64)(v94 + 4);
            if ( v418 != v417 )
            {
              do
              {
                if ( *(_QWORD *)v96 )
                  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v96 + 8LL))(*(_QWORD *)v96);
                v96 += 8;
              }
              while ( v95 != v96 );
              v96 = v417;
            }
            if ( v96 )
              j_j___libc_free_0(v96, (char *)v419 - v96);
            v308[5] = v88;
            v416.m128i_i64[0] = (4LL * *(unsigned __int8 *)(v316 + 178)) | v299 & 0xFFFFFFFFFFFFFFFBLL;
            v97 = (_BYTE *)sub_14318B0((__int64)&v342, (unsigned __int64 *)&v416);
            v98 = *v97 & 7;
            if ( v98 < v293 )
              LOBYTE(v98) = v293;
            *v97 = v98 & 7 | *v97 & 0xF8;
            if ( v291 )
            {
              v99 = sub_1368AA0(v288, v296 - 24);
              v100 = sub_1368DC0((__int64)v288);
              if ( v100 )
              {
                v416.m128i_i64[0] = v99;
                v101 = 8;
                v416.m128i_i16[4] = 8;
                if ( v99 )
                {
                  v416.m128i_i64[0] = sub_16CB530(v99, v100);
                  v416.m128i_i16[4] = v102;
                  sub_1371BB0((__int64)&v416, 8);
                  v99 = v416.m128i_i64[0];
                  v101 = v416.m128i_i16[4];
                }
                v103 = *(_DWORD *)v97 >> 3;
                if ( (int)sub_1371720(v99, v101, 1u, 0) < 0 )
                  goto LABEL_213;
                if ( (int)sub_1371720(v416.m128i_u64[0], v416.m128i_i16[4], 0xFFFFFFFFFFFFFFFFLL, 0) >= 0 )
                {
                  v105 = -1;
                  v104 = -1;
                }
                else
                {
                  v104 = v416.m128i_i64[0];
                  if ( v416.m128i_i16[4] > 0 )
                  {
                    v104 = v416.m128i_i64[0] << v416.m128i_i8[8];
                    v105 = v416.m128i_i64[0] << v416.m128i_i8[8];
                    if ( v103 >= v416.m128i_i64[0] << v416.m128i_i8[8] )
                      v105 = v103;
                  }
                  else
                  {
                    if ( v416.m128i_i16[4] )
                      v104 = (unsigned __int64)v416.m128i_i64[0] >> -v416.m128i_i8[8];
                    v105 = v104;
                    if ( v103 >= v104 )
                      v105 = v103;
                  }
                }
                v103 += v104;
                v106 = 0x1FFFFFFF;
                if ( v105 <= v103 )
                {
LABEL_213:
                  v106 = 0x1FFFFFFF;
                  if ( v103 <= 0x1FFFFFFF )
                    v106 = v103;
                }
                *(_DWORD *)v97 = (8 * v106) | *(_DWORD *)v97 & 7;
              }
            }
          }
          goto LABEL_24;
        }
        if ( v89 == 1 )
        {
          v87 = sub_164A820(*(_QWORD *)(v88 - 24));
          v110 = *(_BYTE *)(v87 + 16);
          if ( !v110 )
            goto LABEL_172;
          if ( v110 != 3 )
            BUG();
        }
        if ( (!v85 || *(_BYTE *)(*(_QWORD *)(v85 - 24) + 16LL) != 20) && *(_BYTE *)(v88 + 16) > 0x10u )
        {
          if ( *(_QWORD *)(v11 + 24) || *(__int16 *)(v11 - 6) < 0 )
          {
            v111 = sub_1625790(v11 - 24, 23);
            v309 = v111;
            v112 = v111;
            if ( v111 )
            {
              v113 = 8LL * *(unsigned int *)(v111 + 8);
              if ( v112 - v113 != v112 )
              {
                v281 = v11;
                v274 = v11 - 24;
                v114 = v112 - v113;
                do
                {
                  if ( *(_QWORD *)v114 )
                  {
                    v115 = *(_QWORD *)(*(_QWORD *)v114 + 136LL);
                    if ( v115 )
                    {
                      sub_15E4EB0(&v338);
                      v116 = v339;
                      v300 = v338;
                      sub_16C1840(&v416);
                      sub_16C1A90(&v416, v300, v116);
                      sub_16C1AA0(&v416, v332);
                      v117 = (__int64 *)v332[0];
                      if ( v338 != &v340 )
                        j_j___libc_free_0(v338, v340 + 1);
                      v338 = v117;
                      if ( *(_BYTE *)(v316 + 178) )
                      {
                        v416.m128i_i64[0] = 0;
                      }
                      else
                      {
                        v416.m128i_i64[1] = 0;
                        v416.m128i_i64[0] = (__int64)byte_3F871B3;
                      }
                      v417 = 0;
                      v418 = 0;
                      v419 = 0;
                      v118 = sub_142DA40((_QWORD *)v316, (unsigned __int64 *)&v338, &v416);
                      v119 = v418;
                      v120 = v417;
                      v301 = v118;
                      v294 = (unsigned __int64)(v118 + 4);
                      if ( v418 != v417 )
                      {
                        do
                        {
                          if ( *(_QWORD *)v120 )
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v120 + 8LL))(*(_QWORD *)v120);
                          v120 += 8;
                        }
                        while ( v119 != v120 );
                        v120 = v417;
                      }
                      if ( v120 )
                        j_j___libc_free_0(v120, (char *)v419 - v120);
                      v301[5] = v115;
                      v416.m128i_i64[0] = v294 & 0xFFFFFFFFFFFFFFFBLL | (4LL * *(unsigned __int8 *)(v316 + 178));
                      sub_14318B0((__int64)&v342, (unsigned __int64 *)&v416);
                    }
                  }
                  v114 += 8;
                }
                while ( v309 != v114 );
                v11 = v281;
                v12 = v274;
              }
            }
          }
          v121 = sub_14DE7C0(&v322, v12, v327, v332, &v329);
          v122 *= 16;
          for ( i = v121 + v122; i != v121; *v127 = *v127 & 0xF8 | v131 )
          {
            v123 = *(_BYTE *)(v316 + 178) == 0;
            v338 = *(__int64 **)v121;
            if ( v123 )
            {
              v416.m128i_i64[1] = 0;
              v416.m128i_i64[0] = (__int64)byte_3F871B3;
            }
            else
            {
              v416.m128i_i64[0] = 0;
            }
            v417 = 0;
            v418 = 0;
            v419 = 0;
            v124 = sub_142DA40((_QWORD *)v316, (unsigned __int64 *)&v338, &v416);
            v125 = v418;
            v126 = v417;
            v302 = (unsigned __int64)(v124 + 4);
            if ( v418 != v417 )
            {
              do
              {
                if ( *(_QWORD *)v126 )
                  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v126 + 8LL))(*(_QWORD *)v126);
                v126 += 8;
              }
              while ( v125 != v126 );
              v126 = v417;
            }
            if ( v126 )
              j_j___libc_free_0(v126, (char *)v419 - v126);
            v416.m128i_i64[0] = (4LL * *(unsigned __int8 *)(v316 + 178)) | v302 & 0xFFFFFFFFFFFFFFFBLL;
            v127 = (_BYTE *)sub_14318B0((__int64)&v342, (unsigned __int64 *)&v416);
            if ( a4 )
            {
              v128 = *(_QWORD *)(v121 + 8);
              v129 = sub_1441CD0(a4, v128);
              v130 = 3;
              if ( !v129 )
                v130 = ((unsigned __int8)sub_1441D60(a4, v128, 3) == 0) + 1;
              v131 = *v127 & 7;
              if ( v131 < v130 )
                LOBYTE(v131) = v130;
            }
            else
            {
              LOBYTE(v131) = *v127 & 7;
            }
            v121 += 16;
          }
        }
        goto LABEL_24;
      }
LABEL_32:
      sub_15E4750(&v338, v283);
      v15 = v339;
      if ( (_DWORD)v340 )
      {
        v185 = v339;
        v186 = &v339[v341];
        if ( v339 != v186 )
        {
          while ( (unsigned __int64)*v185 > 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v186 == ++v185 )
              goto LABEL_33;
          }
          if ( v186 != v185 )
          {
            do
            {
              v332[0] = *v185;
              if ( *(_BYTE *)(v316 + 178) )
              {
                v416.m128i_i64[0] = 0;
              }
              else
              {
                v416.m128i_i64[1] = 0;
                v416.m128i_i64[0] = (__int64)byte_3F871B3;
              }
              v417 = 0;
              v418 = 0;
              v419 = 0;
              v187 = sub_142DA40((_QWORD *)v316, (unsigned __int64 *)v332, &v416);
              v188 = v418;
              v189 = v417;
              v314 = (unsigned __int64)(v187 + 4);
              if ( v418 != v417 )
              {
                do
                {
                  if ( *(_QWORD *)v189 )
                    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v189 + 8LL))(*(_QWORD *)v189);
                  v189 += 8;
                }
                while ( v188 != v189 );
                v189 = v417;
              }
              if ( v189 )
                j_j___libc_free_0(v189, (char *)v419 - v189);
              v416.m128i_i64[0] = (4LL * *(unsigned __int8 *)(v316 + 178)) | v314 & 0xFFFFFFFFFFFFFFFBLL;
              v190 = (_BYTE *)sub_14318B0((__int64)&v342, (unsigned __int64 *)&v416);
              v191 = *v190 & 7;
              if ( v191 < 3 * (unsigned int)(dword_4F99BC0 != 2) + 1 )
                LOBYTE(v191) = 3 * (dword_4F99BC0 != 2) + 1;
              ++v185;
              *v190 = v191 & 7 | *v190 & 0xF8;
              if ( v185 == v186 )
                break;
              while ( (unsigned __int64)*v185 > 0xFFFFFFFFFFFFFFFDLL )
              {
                if ( v186 == ++v185 )
                  goto LABEL_357;
              }
            }
            while ( v186 != v185 );
LABEL_357:
            v15 = v339;
          }
        }
      }
LABEL_33:
      j___libc_free_0(v15);
      sub_15E64D0(v283);
      if ( v16 && (v312 = *(_BYTE *)(v283 + 32) & 0xF, (unsigned int)v312 - 7 <= 1) )
      {
        v304 = 1;
        v303 = 1;
      }
      else if ( v304 )
      {
        v312 = *(_BYTE *)(v283 + 32) & 0xF;
        v193 = v304;
        v304 = 0;
        v303 = v193;
      }
      else
      {
        if ( *(_DWORD *)(*(_QWORD *)(v283 + 24) + 8LL) >> 8 )
        {
          v303 = 1;
        }
        else
        {
          v416.m128i_i64[0] = *(_QWORD *)(v283 + 112);
          v303 = sub_1560180(&v416, 26);
        }
        v312 = *(_BYTE *)(v283 + 32) & 0xF;
      }
      v307 = (*(_BYTE *)(v283 + 33) & 0x40) != 0;
      v306 = sub_1560180(v283 + 112, 36);
      v298 = sub_1560180(v283 + 112, 37);
      v297 = sub_1560180(v283 + 112, 27);
      v292 = sub_1560260(v283 + 112, 0, 20);
      sub_142E680(&v384, 0);
      v17 = v388;
      v388 = 0;
      v290 = v17;
      v18 = v389;
      v389 = 0;
      v289 = v18;
      v19 = v390;
      v390 = 0;
      v287 = v19;
      sub_142E680(&v377, 0);
      v20 = v381;
      ++v370;
      v381 = 0;
      v285 = v20;
      v21 = v382;
      v382 = 0;
      v284 = v21;
      v22 = v383;
      v383 = 0;
      v280 = v22;
      if ( !(_DWORD)v372 )
      {
        if ( !HIDWORD(v372) )
          goto LABEL_42;
        v23 = (unsigned int)v373;
        if ( (unsigned int)v373 > 0x40 )
        {
          j___libc_free_0(v371);
          v371 = 0;
          v372 = 0;
          LODWORD(v373) = 0;
          goto LABEL_42;
        }
LABEL_39:
        v24 = v371;
        v25 = &v371[2 * v23];
        if ( v371 != v25 )
        {
          do
          {
            *v24 = 0;
            v24 += 2;
            *(v24 - 1) = -1;
          }
          while ( v25 != v24 );
        }
        v372 = 0;
        goto LABEL_42;
      }
      v172 = 4 * v372;
      v23 = (unsigned int)v373;
      if ( (unsigned int)(4 * v372) < 0x40 )
        v172 = 64;
      if ( v172 >= (unsigned int)v373 )
        goto LABEL_39;
      v173 = v371;
      if ( (_DWORD)v372 == 1 )
      {
        v179 = 2048;
        v178 = 128;
LABEL_329:
        j___libc_free_0(v371);
        LODWORD(v373) = v178;
        v180 = (_QWORD *)sub_22077B0(v179);
        v372 = 0;
        v371 = v180;
        for ( j = &v180[2 * (unsigned int)v373]; j != v180; v180 += 2 )
        {
          if ( v180 )
          {
            *v180 = 0;
            v180[1] = -1;
          }
        }
        goto LABEL_42;
      }
      _BitScanReverse(&v174, v372 - 1);
      v175 = (unsigned int)(1 << (33 - (v174 ^ 0x1F)));
      if ( (int)v175 < 64 )
        v175 = 64;
      if ( (_DWORD)v175 != (_DWORD)v373 )
      {
        v176 = (4 * (int)v175 / 3u + 1) | ((unsigned __int64)(4 * (int)v175 / 3u + 1) >> 1);
        v177 = ((v176 | (v176 >> 2)) >> 4)
             | v176
             | (v176 >> 2)
             | ((((v176 | (v176 >> 2)) >> 4) | v176 | (v176 >> 2)) >> 8);
        v178 = (v177 | (v177 >> 16)) + 1;
        v179 = 16 * ((v177 | (v177 >> 16)) + 1);
        goto LABEL_329;
      }
      v372 = 0;
      v257 = &v371[2 * v175];
      do
      {
        if ( v173 )
        {
          *v173 = 0;
          v173[1] = -1;
        }
        v173 += 2;
      }
      while ( v257 != v173 );
LABEL_42:
      v26 = v374;
      ++v363;
      v374 = 0;
      v279 = v26;
      v27 = v375;
      v375 = 0;
      v273 = v27;
      v28 = v376;
      v376 = 0;
      v272 = v28;
      if ( !(_DWORD)v365 )
      {
        if ( !HIDWORD(v365) )
          goto LABEL_48;
        v29 = (unsigned int)v366;
        if ( (unsigned int)v366 > 0x40 )
        {
          j___libc_free_0(v364);
          v364 = 0;
          v365 = 0;
          LODWORD(v366) = 0;
          goto LABEL_48;
        }
LABEL_45:
        v30 = v364;
        v31 = &v364[2 * v29];
        if ( v364 != v31 )
        {
          do
          {
            *v30 = 0;
            v30 += 2;
            *(v30 - 1) = -1;
          }
          while ( v31 != v30 );
        }
        v365 = 0;
        goto LABEL_48;
      }
      v162 = 4 * v365;
      v29 = (unsigned int)v366;
      if ( (unsigned int)(4 * v365) < 0x40 )
        v162 = 64;
      if ( v162 >= (unsigned int)v366 )
        goto LABEL_45;
      v163 = v364;
      if ( (_DWORD)v365 == 1 )
      {
        v169 = 2048;
        v168 = 128;
LABEL_316:
        j___libc_free_0(v364);
        LODWORD(v366) = v168;
        v170 = (_QWORD *)sub_22077B0(v169);
        v365 = 0;
        v364 = v170;
        for ( k = &v170[2 * (unsigned int)v366]; k != v170; v170 += 2 )
        {
          if ( v170 )
          {
            *v170 = 0;
            v170[1] = -1;
          }
        }
        goto LABEL_48;
      }
      _BitScanReverse(&v164, v365 - 1);
      v165 = (unsigned int)(1 << (33 - (v164 ^ 0x1F)));
      if ( (int)v165 < 64 )
        v165 = 64;
      if ( (_DWORD)v165 != (_DWORD)v366 )
      {
        v166 = (4 * (int)v165 / 3u + 1) | ((unsigned __int64)(4 * (int)v165 / 3u + 1) >> 1);
        v167 = ((v166 | (v166 >> 2)) >> 4)
             | v166
             | (v166 >> 2)
             | ((((v166 | (v166 >> 2)) >> 4) | v166 | (v166 >> 2)) >> 8);
        v168 = (v167 | (v167 >> 16)) + 1;
        v169 = 16 * ((v167 | (v167 >> 16)) + 1);
        goto LABEL_316;
      }
      v365 = 0;
      v255 = &v364[2 * v165];
      do
      {
        if ( v163 )
        {
          *v163 = 0;
          v163[1] = -1;
        }
        v163 += 2;
      }
      while ( v255 != v163 );
LABEL_48:
      v32 = v367;
      v367 = 0;
      ++v356;
      v271 = v32;
      v33 = v368;
      v368 = 0;
      v270 = v33;
      v34 = v369;
      v369 = 0;
      v269 = v34;
      if ( !(_DWORD)v358 )
      {
        if ( !HIDWORD(v358) )
          goto LABEL_54;
        v35 = (unsigned int)v359;
        if ( (unsigned int)v359 > 0x40 )
        {
          j___libc_free_0(s);
          s = 0;
          v358 = 0;
          LODWORD(v359) = 0;
          goto LABEL_54;
        }
LABEL_51:
        if ( 8 * v35 )
          memset(s, 255, 8 * v35);
        v358 = 0;
        goto LABEL_54;
      }
      v143 = 4 * v358;
      v35 = (unsigned int)v359;
      if ( (unsigned int)(4 * v358) < 0x40 )
        v143 = 64;
      if ( v143 >= (unsigned int)v359 )
        goto LABEL_51;
      v144 = (char *)s;
      if ( (_DWORD)v358 == 1 )
      {
        v149 = 1024;
        v148 = 128;
LABEL_290:
        j___libc_free_0(s);
        LODWORD(v359) = v148;
        v150 = (_QWORD *)sub_22077B0(v149);
        v358 = 0;
        s = v150;
        for ( m = &v150[(unsigned int)v359]; m != v150; ++v150 )
        {
          if ( v150 )
            *v150 = -1;
        }
        goto LABEL_54;
      }
      _BitScanReverse(&v145, v358 - 1);
      v146 = 1 << (33 - (v145 ^ 0x1F));
      if ( v146 < 64 )
        v146 = 64;
      if ( (_DWORD)v359 != v146 )
      {
        v147 = (((4 * v146 / 3u + 1)
               | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)
               | (((4 * v146 / 3u + 1) | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)) >> 2)) >> 4)
             | (4 * v146 / 3u + 1)
             | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)
             | (((4 * v146 / 3u + 1) | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)) >> 2)
             | (((((4 * v146 / 3u + 1)
                 | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)
                 | (((4 * v146 / 3u + 1) | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)) >> 2)) >> 4)
               | (4 * v146 / 3u + 1)
               | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)
               | (((4 * v146 / 3u + 1) | ((unsigned __int64)(4 * v146 / 3u + 1) >> 1)) >> 2)) >> 8);
        v148 = (v147 | (v147 >> 16)) + 1;
        v149 = 8 * ((v147 | (v147 >> 16)) + 1);
        goto LABEL_290;
      }
      v358 = 0;
      v253 = (char *)s + 8 * (unsigned int)v359;
      do
      {
        if ( v144 )
          *(_QWORD *)v144 = -1;
        v144 += 8;
      }
      while ( v253 != v144 );
LABEL_54:
      v36 = v360;
      v360 = 0;
      v37 = v361;
      v38 = v362;
      v361 = 0;
      ++v342;
      v362 = 0;
      if ( !(_DWORD)v344 )
      {
        if ( !HIDWORD(v344) )
          goto LABEL_60;
        v39 = v345;
        if ( v345 > 0x40 )
        {
          v266 = v36;
          j___libc_free_0(v343);
          v36 = v266;
          v343 = 0;
          v344 = 0;
          v345 = 0;
          goto LABEL_60;
        }
LABEL_57:
        v40 = v343;
        v41 = &v343[2 * v39];
        if ( v343 != v41 )
        {
          do
          {
            *v40 = -8;
            v40 += 2;
          }
          while ( v41 != v40 );
        }
        v344 = 0;
        goto LABEL_60;
      }
      v132 = 4 * v344;
      v39 = v345;
      if ( (unsigned int)(4 * v344) < 0x40 )
        v132 = 64;
      if ( v132 >= v345 )
        goto LABEL_57;
      v133 = v343;
      if ( (_DWORD)v344 == 1 )
      {
        v140 = 2048;
        v139 = 128;
LABEL_277:
        v263 = v36;
        v260 = v139;
        j___libc_free_0(v343);
        v345 = v260;
        v141 = (_QWORD *)sub_22077B0(v140);
        v344 = 0;
        v343 = v141;
        v36 = v263;
        for ( n = &v141[2 * v345]; n != v141; v141 += 2 )
        {
          if ( v141 )
            *v141 = -8;
        }
        goto LABEL_60;
      }
      _BitScanReverse(&v134, v344 - 1);
      v135 = (unsigned int)(1 << (33 - (v134 ^ 0x1F)));
      if ( (int)v135 < 64 )
        v135 = 64;
      if ( (_DWORD)v135 != v345 )
      {
        v136 = (4 * (int)v135 / 3u + 1) | ((unsigned __int64)(4 * (int)v135 / 3u + 1) >> 1);
        v137 = ((v136 | (v136 >> 2)) >> 4)
             | v136
             | (v136 >> 2)
             | ((((v136 | (v136 >> 2)) >> 4) | v136 | (v136 >> 2)) >> 8);
        v138 = (v137 | (v137 >> 16)) + 1;
        v139 = (v137 | (v137 >> 16)) + 1;
        v140 = 16 * v138;
        goto LABEL_277;
      }
      v344 = 0;
      v254 = &v343[2 * v135];
      do
      {
        if ( v133 )
          *v133 = -8;
        v133 += 2;
      }
      while ( v254 != v133 );
LABEL_60:
      v42 = v346;
      v346 = 0;
      ii = v347;
      v44 = v348;
      v347 = 0;
      ++v349;
      v348 = 0;
      if ( (_DWORD)v351 )
      {
        v152 = 4 * v351;
        v45 = (unsigned int)v352;
        if ( (unsigned int)(4 * v351) < 0x40 )
          v152 = 64;
        if ( v152 >= (unsigned int)v352 )
        {
LABEL_63:
          v46 = v350;
          v47 = &v350[v45];
          if ( v350 != v47 )
          {
            do
              *v46++ = -8;
            while ( v47 != v46 );
          }
          v351 = 0;
          goto LABEL_66;
        }
        v153 = v350;
        if ( (_DWORD)v351 == 1 )
        {
          v159 = 1024;
          v158 = 128;
        }
        else
        {
          _BitScanReverse(&v154, v351 - 1);
          v155 = 1 << (33 - (v154 ^ 0x1F));
          if ( v155 < 64 )
            v155 = 64;
          if ( (_DWORD)v352 == v155 )
          {
            v351 = 0;
            v256 = &v350[(unsigned int)v352];
            do
            {
              if ( v153 )
                *v153 = -8;
              ++v153;
            }
            while ( v256 != v153 );
            goto LABEL_66;
          }
          v156 = (4 * v155 / 3u + 1)
               | ((unsigned __int64)(4 * v155 / 3u + 1) >> 1)
               | (((4 * v155 / 3u + 1) | ((unsigned __int64)(4 * v155 / 3u + 1) >> 1)) >> 2);
          v157 = ((v156 >> 4)
                | v156
                | (((v156 >> 4) | v156) >> 8)
                | (((v156 >> 4) | v156 | (((v156 >> 4) | v156) >> 8)) >> 16))
               + 1;
          v158 = v157;
          v159 = 8 * v157;
        }
        v261 = ii;
        v264 = v36;
        v258 = v159;
        v259 = v158;
        j___libc_free_0(v350);
        LODWORD(v352) = v259;
        v160 = (_QWORD *)sub_22077B0(v258);
        v351 = 0;
        v350 = v160;
        v36 = v264;
        v161 = &v160[(unsigned int)v352];
        for ( ii = v261; v161 != v160; ++v160 )
        {
          if ( v160 )
            *v160 = -8;
        }
      }
      else if ( HIDWORD(v351) )
      {
        v45 = (unsigned int)v352;
        if ( (unsigned int)v352 <= 0x40 )
          goto LABEL_63;
        v262 = ii;
        v265 = v36;
        j___libc_free_0(v350);
        v36 = v265;
        v350 = 0;
        ii = v262;
        v351 = 0;
        LODWORD(v352) = 0;
      }
LABEL_66:
      v325[1] = ii;
      v48 = v277;
      v326 = v44;
      v323[0] = v353;
      v323[1] = v354;
      v324 = v355;
      LOBYTE(v48) = v277 & 0x80 | ((v307 << 6) | v312 | (16 * v303)) & 0x7F;
      v327[0] = v36;
      v277 = v48;
      v328 = v38;
      v355 = 0;
      v354 = 0;
      v325[0] = v42;
      v353 = 0;
      v327[1] = v37;
      v49 = v278;
      LOBYTE(v49) = v278 & 0xF0 | ((8 * v292) | (2 * v298) | v306 | (4 * v297)) & 0xF;
      v278 = v49;
      v329 = v271;
      v330 = v270;
      v331 = v269;
      v332[0] = v279;
      v332[1] = v273;
      v333 = v272;
      v338 = v285;
      v339 = v284;
      v340 = v280;
      v416.m128i_i64[0] = v290;
      v416.m128i_i64[1] = v289;
      v417 = v287;
      v50 = sub_22077B0(104);
      v51 = v50;
      if ( v50 )
        sub_142CF20(v50, v277, v305, v278, v323, v325, v327, &v329, v332, (__int64 *)&v338, &v416);
      v52 = v416.m128i_i64[1];
      v53 = v416.m128i_i64[0];
      if ( v416.m128i_i64[1] != v416.m128i_i64[0] )
      {
        do
        {
          v54 = *(_QWORD *)(v53 + 16);
          if ( v54 )
            j_j___libc_free_0(v54, *(_QWORD *)(v53 + 32) - v54);
          v53 += 40;
        }
        while ( v52 != v53 );
        v53 = v416.m128i_i64[0];
      }
      if ( v53 )
        j_j___libc_free_0(v53, &v417[-v53]);
      v55 = v339;
      v56 = v338;
      if ( v339 != v338 )
      {
        do
        {
          v57 = v56[2];
          if ( v57 )
            j_j___libc_free_0(v57, v56[4] - v57);
          v56 += 5;
        }
        while ( v55 != v56 );
        v56 = v338;
      }
      if ( v56 )
        j_j___libc_free_0(v56, v340 - (_QWORD)v56);
      if ( v332[0] )
        j_j___libc_free_0(v332[0], v333 - v332[0]);
      if ( v329 )
        j_j___libc_free_0(v329, v331 - v329);
      if ( v327[0] )
        j_j___libc_free_0(v327[0], v328 - v327[0]);
      if ( v325[0] )
        j_j___libc_free_0(v325[0], v326 - v325[0]);
      if ( v323[0] )
        j_j___libc_free_0(v323[0], v324 - v323[0]);
      if ( v304 )
      {
        sub_15E4EB0(&v338);
        v182 = v338;
        v183 = v339;
        sub_16C1840(&v416);
        sub_16C1A90(&v416, v182, v183);
        sub_16C1AA0(&v416, v332);
        v184 = (__int64 *)v332[0];
        if ( v338 != &v340 )
          j_j___libc_free_0(v338, v340 + 1);
        v338 = v184;
        sub_142F900((__int64)&v416, (__int64)&v334, (__int64 *)&v338);
      }
      v58 = v283;
      v416.m128i_i64[0] = v51;
      sub_142ED30(v316, v283, &v416);
      if ( v416.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v416.m128i_i64[0] + 8LL))(v416.m128i_i64[0]);
      if ( (_QWORD *)v405[0] != v404 )
        _libc_free(v405[0]);
      if ( v322 )
        j_j___libc_free_0_0(v322);
      v60 = v389;
      v61 = v388;
      if ( v389 != v388 )
      {
        do
        {
          v62 = *(_QWORD *)(v61 + 16);
          if ( v62 )
          {
            v58 = *(_QWORD *)(v61 + 32) - v62;
            j_j___libc_free_0(v62, v58);
          }
          v61 += 40;
        }
        while ( v60 != v61 );
        v61 = v388;
      }
      if ( v61 )
      {
        v58 = (__int64)&v390[-v61];
        j_j___libc_free_0(v61, &v390[-v61]);
      }
      sub_142DC00(&v384, v58, v59);
      j___libc_free_0(v385);
      v64 = v382;
      v65 = v381;
      if ( v382 != v381 )
      {
        do
        {
          v66 = v65[2];
          if ( v66 )
          {
            v58 = v65[4] - v66;
            j_j___libc_free_0(v66, v58);
          }
          v65 += 5;
        }
        while ( v64 != v65 );
        v65 = v381;
      }
      if ( v65 )
      {
        v58 = v383 - (_QWORD)v65;
        j_j___libc_free_0(v65, v383 - (_QWORD)v65);
      }
      sub_142DC00(&v377, v58, v63);
      j___libc_free_0(v378);
      if ( v374 )
        j_j___libc_free_0(v374, v376 - v374);
      j___libc_free_0(v371);
      if ( v367 )
        j_j___libc_free_0(v367, v369 - v367);
      j___libc_free_0(v364);
      if ( v360 )
        j_j___libc_free_0(v360, v362 - v360);
      j___libc_free_0(s);
      if ( v353 )
        j_j___libc_free_0(v353, v355 - v353);
      j___libc_free_0(v350);
      if ( v346 )
        j_j___libc_free_0(v346, v348 - v346);
      j___libc_free_0(v343);
      if ( v275 )
      {
        sub_1368A00(v275);
        j_j___libc_free_0(v275, 8);
      }
LABEL_125:
      v282 = (_QWORD *)v282[1];
      if ( a2 + 3 == v282 )
      {
        v4 = v316;
        break;
      }
    }
  }
  for ( jj = (_QWORD *)a2[2]; a2 + 1 != jj; jj = (_QWORD *)jj[1] )
  {
    v68 = (__int64)(jj - 7);
    if ( !jj )
      v68 = 0;
    if ( !(unsigned __int8)sub_15E4F60(v68) )
      sub_1430F70(v4, v68, (__int64)&v334);
  }
  for ( kk = (_QWORD *)a2[6]; a2 + 5 != kk; kk = (_QWORD *)kk[1] )
  {
    v70 = (__int64)(kk - 6);
    if ( !kk )
      v70 = 0;
    sub_142FB50(v4, v70, (__int64)&v334);
  }
  v71 = v393;
  if ( v393 == v392 )
    v72 = &v393[8 * HIDWORD(v394)];
  else
    v72 = &v393[8 * (unsigned int)v394];
  if ( v393 != v72 )
  {
    while ( 1 )
    {
      v73 = v71;
      if ( *v71 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v72 == (_BYTE *)++v71 )
        goto LABEL_142;
    }
    if ( v72 != (_BYTE *)v71 )
    {
      v204 = v72;
      v205 = v73;
      v206 = v204;
      do
      {
        sub_15E4EB0(&v403);
        v315 = v404;
        v319 = v403;
        sub_16C1840(&v416);
        sub_16C1A90(&v416, v319, v315);
        sub_16C1AA0(&v416, &v384);
        v207 = v384;
        if ( v403 != v405 )
        {
          v320 = v384;
          j_j___libc_free_0(v403, v405[0] + 1);
          v207 = v320;
        }
        v208 = sub_16342D0(v4, v207, 1);
        *(_BYTE *)(v208 + 12) |= 0x10u;
        v209 = v205 + 1;
        if ( v205 + 1 == v206 )
          break;
        for ( ++v205; *v209 >= 0xFFFFFFFFFFFFFFFELL; v205 = v209 )
        {
          if ( v206 == ++v209 )
            goto LABEL_142;
        }
      }
      while ( v206 != v205 );
    }
  }
LABEL_142:
  sub_142BA70(v4, (__int64)"llvm.used", 9);
  sub_142BA70(v4, (__int64)"llvm.compiler.used", 18);
  sub_142BA70(v4, (__int64)"llvm.global_ctors", 17);
  sub_142BA70(v4, (__int64)"llvm.global_dtors", 17);
  sub_142BA70(v4, (__int64)"llvm.global.annotations", 23);
  v74 = sub_16328F0(a2, "ThinLTO", 7);
  if ( v74 )
  {
    v75 = *(_QWORD *)(v74 + 136);
    v76 = 1;
    if ( v75 )
    {
      v77 = *(_QWORD **)(v75 + 24);
      if ( *(_DWORD *)(v75 + 32) > 0x40u )
        v77 = (_QWORD *)*v77;
      v76 = v77 != 0;
    }
  }
  else
  {
    v76 = 1;
  }
  v78 = *(_QWORD *)(v4 + 24);
  if ( v78 != v268 )
  {
    v317 = v4;
    do
    {
      v79 = *(__int64 **)(v78 + 56);
      if ( *(__int64 **)(v78 + 64) != v79 )
      {
        v80 = *v79;
        if ( !v76
          || (v81 = *(_QWORD **)(v80 + 48), v81 != sub_142CBD0(*(_QWORD **)(v80 + 40), (__int64)v81, (__int64)&v334))
          || *(_DWORD *)(v80 + 8) == 1
          && (v313 = *(_QWORD **)(v80 + 80), v313 != sub_142C880(*(_QWORD **)(v80 + 72), (__int64)v313, (__int64)&v334)) )
        {
          *(_BYTE *)(v80 + 12) |= 0x10u;
        }
      }
      v78 = sub_220EEE0(v78);
    }
    while ( v78 != v268 );
    v4 = v317;
  }
  j___libc_free_0(v335);
  if ( v399 != v398 )
    _libc_free((unsigned __int64)v399);
  if ( v393 != v392 )
    _libc_free((unsigned __int64)v393);
  return v4;
}
