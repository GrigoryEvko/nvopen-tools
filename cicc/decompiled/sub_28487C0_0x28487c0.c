// Function: sub_28487C0
// Address: 0x28487c0
//
__int64 __fastcall sub_28487C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  char *v11; // r15
  char *v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // r14
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rbx
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rax
  unsigned int v21; // eax
  unsigned __int64 v22; // rdi
  __int64 v23; // rdi
  _QWORD *v24; // rax
  _QWORD *i; // rdx
  __int64 v26; // rdx
  char *v27; // rbx
  char *v28; // r15
  __int64 v29; // rcx
  __int64 v30; // r14
  char *v31; // r13
  int v32; // eax
  char *v33; // r12
  __int64 v34; // rax
  char *v35; // r12
  char *v36; // r15
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rdi
  __int64 *v41; // rax
  char *v42; // r13
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  unsigned __int64 v48; // r12
  char *v49; // r15
  void (__fastcall *v50)(char *, char *, __int64); // rax
  __int64 v51; // rax
  _QWORD *v53; // rbx
  _QWORD *v54; // r12
  __int64 v55; // rax
  _QWORD *v56; // rbx
  _QWORD *v57; // r12
  __int64 v58; // rax
  unsigned __int64 v59; // rbx
  char *v60; // rax
  void *v61; // r13
  char *v62; // r12
  unsigned __int64 *v63; // rsi
  __int64 v64; // rdx
  __int64 *v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r12
  unsigned __int64 v71; // rax
  int v72; // edx
  unsigned __int64 v73; // rax
  unsigned __int64 v74; // rax
  int v75; // edx
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // r13
  __int64 v80; // rax
  char *v81; // r15
  char *v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  unsigned __int64 v85; // rax
  int v86; // edx
  _QWORD *v87; // rdi
  _QWORD *v88; // rax
  __int64 v89; // r12
  __int64 v90; // rbx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 v93; // rdx
  __int64 v94; // r13
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // r9
  __int64 v99; // rax
  __int64 v100; // rbx
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 *v103; // rdx
  __int64 *v104; // rbx
  __int64 *v105; // r13
  __int64 v106; // r14
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rdx
  unsigned __int64 v111; // r12
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // rax
  unsigned __int64 v115; // rdx
  unsigned __int64 *v116; // rax
  char v117; // r14
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // r8
  __int64 v121; // r9
  _QWORD *v122; // r12
  _QWORD *v123; // r15
  void (__fastcall *v124)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v125; // rax
  __int64 v126; // rsi
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // r8
  __int64 v130; // r9
  __int64 v131; // r8
  __int64 v132; // r9
  __int64 **v133; // rdx
  __int64 v134; // rcx
  void **v135; // rax
  int v136; // eax
  void **v137; // rax
  int v138; // eax
  __int64 *v139; // rax
  __int64 **v140; // rsi
  __int64 v141; // r14
  int v142; // ecx
  __int64 v143; // r9
  unsigned int v144; // esi
  unsigned int v145; // edx
  __int64 *v146; // rax
  __int64 v147; // r8
  char *v148; // rdx
  char *v149; // r15
  __int64 v150; // rax
  int v151; // ecx
  unsigned int jj; // eax
  _QWORD *v153; // rdx
  __int64 *v154; // rbx
  _BYTE *v155; // rax
  __int64 **v156; // rdi
  __int64 **v157; // rdx
  __int64 **v158; // rax
  __int64 v159; // rcx
  __int64 v160; // rax
  __int64 *v161; // r12
  __int64 v162; // r15
  __int64 *n; // r14
  _BYTE *v164; // rax
  __int64 **v165; // rdi
  __int64 **v166; // rdx
  __int64 **v167; // rax
  __int64 v168; // rcx
  __int64 v169; // rcx
  _QWORD *v170; // rdi
  __int64 v171; // r8
  __int64 v172; // r9
  _QWORD *v173; // rbx
  __int64 v174; // r11
  char *v175; // rax
  char *v176; // rsi
  _BYTE *v177; // rsi
  __int64 *v178; // rax
  __int64 v179; // r12
  __int64 v180; // r8
  __int64 v181; // r9
  unsigned __int64 **v182; // rcx
  __int64 v183; // r12
  __int64 v184; // rax
  __int64 v185; // r13
  unsigned __int64 v186; // rax
  __int64 v187; // r15
  unsigned int v188; // ebx
  __int64 *v189; // rdx
  __int64 v190; // rcx
  __int64 v191; // r8
  __int64 v192; // r9
  __int64 v193; // r14
  _QWORD *v194; // rax
  __int64 v195; // rdi
  _QWORD *v196; // rax
  _QWORD *v197; // rdx
  unsigned __int8 v198; // dl
  unsigned __int8 v199; // bl
  _QWORD *v200; // rax
  _QWORD *v201; // rdx
  int v202; // r15d
  unsigned __int64 v203; // rax
  int v204; // edx
  __int64 v205; // rax
  bool v206; // cf
  __int64 v207; // rdx
  _QWORD *v208; // rax
  __int64 v209; // rbx
  __int64 v210; // r13
  char *v211; // r13
  char *v212; // r12
  __int64 v213; // rdx
  unsigned int v214; // esi
  __int64 v215; // r9
  _QWORD *v216; // rax
  _QWORD *v217; // r13
  __int64 v218; // rdx
  _QWORD *v219; // rbx
  __int64 v220; // rsi
  __int64 v221; // rdi
  int v222; // esi
  __int64 v223; // rcx
  __int64 v224; // r9
  int v225; // esi
  unsigned int v226; // edx
  __int64 *v227; // rax
  __int64 v228; // r10
  __int64 *v229; // r8
  __int64 v230; // r8
  __int64 *v231; // rax
  __int64 v232; // r15
  void **v233; // rbx
  void **kk; // r14
  _BYTE *v235; // rax
  void **v236; // r8
  void **v237; // rdx
  void **v238; // rax
  __int64 v239; // rdi
  __int64 v240; // rbx
  _QWORD *v241; // rdi
  _QWORD *v242; // r12
  char *v243; // rax
  char *v244; // rsi
  unsigned __int64 v245; // rax
  _BYTE *v246; // rsi
  __int64 *v247; // rax
  unsigned int v248; // ecx
  __int64 v249; // r8
  __int64 v250; // rax
  __int64 v251; // r12
  unsigned __int64 *v252; // rax
  _QWORD *v253; // rax
  unsigned __int64 v254; // r14
  char *v255; // rbx
  __int64 v256; // rax
  __int64 v257; // rcx
  unsigned int v258; // edx
  void **v259; // r12
  char *v260; // rsi
  __int64 *mm; // r15
  _BYTE *v262; // rax
  void **v263; // rdi
  void **v264; // rdx
  void **v265; // rax
  __int64 v266; // rcx
  __int64 v267; // rcx
  __int64 v268; // r8
  __int64 v269; // r9
  unsigned __int64 **v270; // r13
  __int64 *v271; // rbx
  __int64 *v272; // rax
  __int64 *v273; // rax
  __int64 v274; // r12
  unsigned int v275; // r11d
  int v276; // r8d
  int v277; // ebx
  int v278; // r8d
  __int64 *v279; // rax
  char *v280; // rbx
  char *v281; // r15
  __int64 v282; // r13
  int v283; // esi
  unsigned int v284; // ecx
  __int64 *v285; // rdx
  __int64 v286; // r11
  _QWORD *v287; // r12
  __int64 v288; // rsi
  _QWORD *v289; // rax
  _QWORD *v290; // rdx
  char *v291; // rdx
  unsigned int v292; // esi
  unsigned int v293; // r8d
  __int64 *v294; // rax
  __int64 v295; // rdi
  int k; // edx
  _QWORD *v297; // rdx
  _QWORD *v298; // rax
  unsigned int v299; // ecx
  unsigned int v300; // edx
  int v301; // eax
  int v302; // r8d
  unsigned __int64 v303; // rax
  _BYTE *v304; // rsi
  int j; // eax
  int v306; // edi
  int v307; // edi
  unsigned int v308; // eax
  int v309; // esi
  int v310; // esi
  int v311; // edi
  __int64 *v312; // rcx
  unsigned int ii; // edx
  __int64 v314; // r9
  int v315; // edx
  int v316; // ecx
  __int64 *v317; // rdx
  int v318; // ecx
  int v319; // esi
  int v320; // esi
  __int64 v321; // rdi
  int v322; // r8d
  unsigned int m; // ebx
  __int64 *v324; // rcx
  __int64 v325; // rdx
  unsigned int v326; // edx
  unsigned int v327; // ebx
  __int64 v328; // [rsp-8h] [rbp-C58h]
  __int64 *v329; // [rsp+28h] [rbp-C28h]
  __int64 v330; // [rsp+38h] [rbp-C18h]
  __int64 *v331; // [rsp+38h] [rbp-C18h]
  unsigned int v332; // [rsp+40h] [rbp-C10h]
  __int64 v333; // [rsp+40h] [rbp-C10h]
  __int64 v334; // [rsp+68h] [rbp-BE8h]
  __int64 *v335; // [rsp+68h] [rbp-BE8h]
  _QWORD *v336; // [rsp+68h] [rbp-BE8h]
  __int64 *v337; // [rsp+68h] [rbp-BE8h]
  __int64 v338; // [rsp+68h] [rbp-BE8h]
  __int64 v339; // [rsp+78h] [rbp-BD8h]
  __int64 v340; // [rsp+80h] [rbp-BD0h]
  __int64 *v341; // [rsp+80h] [rbp-BD0h]
  __int64 *v342; // [rsp+80h] [rbp-BD0h]
  __int64 *v343; // [rsp+80h] [rbp-BD0h]
  __int64 v344; // [rsp+88h] [rbp-BC8h]
  __int64 v345; // [rsp+90h] [rbp-BC0h]
  unsigned int v346; // [rsp+98h] [rbp-BB8h]
  __int64 *v347; // [rsp+98h] [rbp-BB8h]
  __int64 v348; // [rsp+A0h] [rbp-BB0h]
  __int64 *v349; // [rsp+A8h] [rbp-BA8h]
  char v350; // [rsp+B7h] [rbp-B99h]
  char v352; // [rsp+C0h] [rbp-B90h]
  __int64 *v353; // [rsp+C0h] [rbp-B90h]
  __int64 v354; // [rsp+C0h] [rbp-B90h]
  int v355; // [rsp+C0h] [rbp-B90h]
  void **v356; // [rsp+C0h] [rbp-B90h]
  char v359[32]; // [rsp+E0h] [rbp-B70h] BYREF
  __int16 v360; // [rsp+100h] [rbp-B50h]
  __int64 *v361; // [rsp+110h] [rbp-B40h] BYREF
  unsigned __int64 v362; // [rsp+118h] [rbp-B38h]
  __int64 v363; // [rsp+120h] [rbp-B30h] BYREF
  int v364; // [rsp+128h] [rbp-B28h]
  char v365; // [rsp+12Ch] [rbp-B24h]
  __int16 v366; // [rsp+130h] [rbp-B20h] BYREF
  char *v367; // [rsp+140h] [rbp-B10h] BYREF
  __int64 v368; // [rsp+148h] [rbp-B08h]
  char v369[8]; // [rsp+150h] [rbp-B00h] BYREF
  unsigned int v370; // [rsp+158h] [rbp-AF8h]
  __int64 *v371; // [rsp+160h] [rbp-AF0h]
  __int64 v372; // [rsp+170h] [rbp-AE0h] BYREF
  __int64 v373; // [rsp+178h] [rbp-AD8h]
  __int64 v374; // [rsp+180h] [rbp-AD0h]
  _QWORD *v375; // [rsp+188h] [rbp-AC8h]
  void **v376; // [rsp+190h] [rbp-AC0h]
  _QWORD *v377; // [rsp+198h] [rbp-AB8h]
  __int64 v378; // [rsp+1A0h] [rbp-AB0h]
  int v379; // [rsp+1A8h] [rbp-AA8h]
  __int16 v380; // [rsp+1ACh] [rbp-AA4h]
  char v381; // [rsp+1AEh] [rbp-AA2h]
  __int64 v382; // [rsp+1B0h] [rbp-AA0h]
  __int64 v383; // [rsp+1B8h] [rbp-A98h]
  void *v384; // [rsp+1C0h] [rbp-A90h] BYREF
  _QWORD v385[33]; // [rsp+1C8h] [rbp-A88h] BYREF
  _QWORD v386[96]; // [rsp+2D0h] [rbp-980h] BYREF
  __int64 *v387; // [rsp+5D0h] [rbp-680h] BYREF
  unsigned __int64 v388; // [rsp+5D8h] [rbp-678h]
  __int64 v389; // [rsp+5E0h] [rbp-670h] BYREF
  __int64 v390; // [rsp+5E8h] [rbp-668h] BYREF
  __int64 *v391[2]; // [rsp+5F0h] [rbp-660h] BYREF
  __int64 v392; // [rsp+600h] [rbp-650h] BYREF
  unsigned __int64 v393; // [rsp+608h] [rbp-648h]
  __int64 v394; // [rsp+610h] [rbp-640h]
  unsigned int v395; // [rsp+618h] [rbp-638h]
  char v396; // [rsp+61Ch] [rbp-634h]
  void *src; // [rsp+620h] [rbp-630h] BYREF
  char *v398; // [rsp+628h] [rbp-628h]
  char *v399; // [rsp+630h] [rbp-620h]
  unsigned __int64 v400[2]; // [rsp+638h] [rbp-618h] BYREF
  _BYTE v401[288]; // [rsp+648h] [rbp-608h] BYREF
  __int64 v402; // [rsp+768h] [rbp-4E8h] BYREF
  _BYTE *v403; // [rsp+770h] [rbp-4E0h]
  __int64 v404; // [rsp+778h] [rbp-4D8h]
  int v405; // [rsp+780h] [rbp-4D0h]
  char v406; // [rsp+784h] [rbp-4CCh]
  _BYTE v407[64]; // [rsp+788h] [rbp-4C8h] BYREF
  char *v408; // [rsp+7C8h] [rbp-488h] BYREF
  __int64 v409; // [rsp+7D0h] [rbp-480h]
  char v410; // [rsp+7D8h] [rbp-478h] BYREF
  __int64 v411; // [rsp+7E0h] [rbp-470h]
  __int64 v412; // [rsp+7E8h] [rbp-468h]
  __int64 v413; // [rsp+7F0h] [rbp-460h]
  __int64 v414; // [rsp+7F8h] [rbp-458h]
  char v415; // [rsp+800h] [rbp-450h]
  __int64 v416; // [rsp+808h] [rbp-448h]
  _BYTE *v417; // [rsp+810h] [rbp-440h]
  __int64 v418; // [rsp+818h] [rbp-438h]
  int v419; // [rsp+820h] [rbp-430h]
  char v420; // [rsp+824h] [rbp-42Ch]
  _BYTE v421[32]; // [rsp+828h] [rbp-428h] BYREF
  __int64 v422; // [rsp+848h] [rbp-408h]
  __int64 v423; // [rsp+850h] [rbp-400h]
  __int64 v424; // [rsp+858h] [rbp-3F8h]
  __int64 v425; // [rsp+860h] [rbp-3F0h]
  __int16 v426; // [rsp+868h] [rbp-3E8h]
  char *v427; // [rsp+870h] [rbp-3E0h]
  char *v428; // [rsp+878h] [rbp-3D8h]
  __int64 v429; // [rsp+880h] [rbp-3D0h]
  int v430; // [rsp+888h] [rbp-3C8h]
  char v431; // [rsp+88Ch] [rbp-3C4h]
  char v432; // [rsp+890h] [rbp-3C0h] BYREF
  int v433; // [rsp+8A0h] [rbp-3B0h] BYREF
  __int64 v434; // [rsp+8A8h] [rbp-3A8h]
  int *v435; // [rsp+8B0h] [rbp-3A0h]
  int *v436; // [rsp+8B8h] [rbp-398h]
  __int64 v437; // [rsp+8C0h] [rbp-390h]
  __int16 v438; // [rsp+8D0h] [rbp-380h]
  _QWORD *v439; // [rsp+8D8h] [rbp-378h]
  _QWORD *v440; // [rsp+8E0h] [rbp-370h]
  __int64 v441; // [rsp+8E8h] [rbp-368h]
  unsigned __int64 *v442; // [rsp+8F0h] [rbp-360h] BYREF
  __int64 v443; // [rsp+8F8h] [rbp-358h]
  _BYTE v444[256]; // [rsp+900h] [rbp-350h] BYREF
  __int16 v445; // [rsp+A00h] [rbp-250h]
  __int64 v446; // [rsp+A08h] [rbp-248h]
  char *v447; // [rsp+A10h] [rbp-240h]
  __int64 v448; // [rsp+A18h] [rbp-238h]
  int v449; // [rsp+A20h] [rbp-230h]
  char v450; // [rsp+A24h] [rbp-22Ch]
  char v451; // [rsp+A28h] [rbp-228h] BYREF
  __int64 *v452; // [rsp+A68h] [rbp-1E8h]
  __int64 v453; // [rsp+A70h] [rbp-1E0h]
  _BYTE v454[64]; // [rsp+A78h] [rbp-1D8h] BYREF
  __int64 v455; // [rsp+AB8h] [rbp-198h]
  char *v456; // [rsp+AC0h] [rbp-190h]
  __int64 v457; // [rsp+AC8h] [rbp-188h]
  int v458; // [rsp+AD0h] [rbp-180h]
  char v459; // [rsp+AD4h] [rbp-17Ch]
  char v460; // [rsp+AD8h] [rbp-178h] BYREF
  __int64 *v461; // [rsp+B18h] [rbp-138h]
  __int64 v462; // [rsp+B20h] [rbp-130h]
  _BYTE v463[64]; // [rsp+B28h] [rbp-128h] BYREF
  __int64 v464; // [rsp+B68h] [rbp-E8h]
  char *v465; // [rsp+B70h] [rbp-E0h]
  __int64 v466; // [rsp+B78h] [rbp-D8h]
  int v467; // [rsp+B80h] [rbp-D0h]
  char v468; // [rsp+B84h] [rbp-CCh]
  char v469; // [rsp+B88h] [rbp-C8h] BYREF
  __int64 *v470; // [rsp+BC8h] [rbp-88h]
  __int64 v471; // [rsp+BD0h] [rbp-80h]
  _BYTE v472[120]; // [rsp+BD8h] [rbp-78h] BYREF

  v6 = a5[9];
  memset(v386, 0, sizeof(v386));
  v339 = a6;
  v349 = (__int64 *)v6;
  if ( v6 )
  {
    v387 = (__int64 *)v6;
    v386[0] = v6;
    v386[1] = &v386[3];
    v389 = 0x1000000000LL;
    v386[2] = 0x1000000000LL;
    v388 = (unsigned __int64)&v390;
    v402 = 0;
    v403 = v407;
    v404 = 8;
    v405 = 0;
    v406 = 1;
    v408 = &v410;
    v409 = 0x800000000LL;
    v433 = 0;
    v434 = 0;
    v435 = &v433;
    v436 = &v433;
    v437 = 0;
    sub_C8CF70((__int64)&v386[51], &v386[55], 8, (__int64)v407, (__int64)&v402);
    v6 = (unsigned int)v409;
    v386[64] = 0x800000000LL;
    v386[63] = &v386[65];
    if ( (_DWORD)v409 )
    {
      v6 = (__int64)&v408;
      sub_2845F60((__int64)&v386[63], (__int64)&v408, v7, v8, v9, v10);
    }
    if ( v434 )
    {
      v386[91] = v434;
      LODWORD(v386[90]) = v433;
      v386[92] = v435;
      v386[93] = v436;
      *(_QWORD *)(v434 + 8) = &v386[90];
      v434 = 0;
      v386[94] = v437;
      v435 = &v433;
      v436 = &v433;
      v437 = 0;
    }
    else
    {
      LODWORD(v386[90]) = 0;
      v386[91] = 0;
      v386[92] = &v386[90];
      v386[93] = &v386[90];
      v386[94] = 0;
    }
    LOBYTE(v386[95]) = 1;
    sub_2845A40(0);
    v11 = v408;
    v12 = &v408[24 * (unsigned int)v409];
    if ( v408 != v12 )
    {
      do
      {
        v13 = *((_QWORD *)v12 - 1);
        v12 -= 24;
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD60C0(v12);
      }
      while ( v11 != v12 );
      v12 = v408;
    }
    if ( v12 != &v410 )
      _libc_free((unsigned __int64)v12);
    if ( !v406 )
      _libc_free((unsigned __int64)v403);
    v14 = v388;
    v15 = (__int64 *)(v388 + 24LL * (unsigned int)v389);
    if ( (__int64 *)v388 != v15 )
    {
      do
      {
        v16 = *(v15 - 1);
        v15 -= 3;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          sub_BD60C0(v15);
      }
      while ( (__int64 *)v14 != v15 );
      v15 = (__int64 *)v388;
    }
    if ( v15 != &v390 )
      _libc_free((unsigned __int64)v15);
    v17 = 0;
    if ( LOBYTE(v386[95]) )
      v17 = v386;
    v349 = v17;
  }
  v344 = a5[4];
  v345 = a5[3];
  v348 = a5[2];
  v350 = byte_50006C8;
  if ( !byte_50006C8 || !sub_D47930((__int64)a3) )
  {
    v350 = 0;
    goto LABEL_36;
  }
  v392 = 0;
  v18 = a3[5] - a3[4];
  v387 = a3;
  v391[1] = a3;
  v19 = (unsigned int)(v18 >> 3);
  v388 = v345;
  v389 = v348;
  v390 = v344;
  v391[0] = v349;
  v20 = ((((((((v19 | (v19 >> 1)) >> 2) | v19 | (v19 >> 1)) >> 4) | ((v19 | (v19 >> 1)) >> 2) | v19 | (v19 >> 1)) >> 8)
        | ((((v19 | (v19 >> 1)) >> 2) | v19 | (v19 >> 1)) >> 4)
        | ((v19 | (v19 >> 1)) >> 2)
        | v19
        | (v19 >> 1)) >> 16)
      | ((((((v19 | (v19 >> 1)) >> 2) | v19 | (v19 >> 1)) >> 4) | ((v19 | (v19 >> 1)) >> 2) | v19 | (v19 >> 1)) >> 8)
      | ((((v19 | (v19 >> 1)) >> 2) | v19 | (v19 >> 1)) >> 4)
      | ((v19 | (v19 >> 1)) >> 2)
      | v19
      | (v19 >> 1);
  if ( (_DWORD)v20 == -1 )
  {
    v393 = 0;
    v394 = 0;
    v395 = 0;
  }
  else
  {
    v21 = 4 * (v20 + 1);
    v22 = (((((((v21 / 3 + 1) | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 2)
            | (v21 / 3 + 1)
            | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 4)
          | (((v21 / 3 + 1) | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 2)
          | (v21 / 3 + 1)
          | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 8)
        | (((((v21 / 3 + 1) | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 2)
          | (v21 / 3 + 1)
          | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 4)
        | (((v21 / 3 + 1) | ((unsigned __int64)(v21 / 3 + 1) >> 1)) >> 2)
        | (v21 / 3 + 1)
        | ((unsigned __int64)(v21 / 3 + 1) >> 1);
    v23 = ((v22 >> 16) | v22) + 1;
    v395 = v23;
    v24 = (_QWORD *)sub_C7D670(16 * v23, 8);
    v394 = 0;
    v393 = (unsigned __int64)v24;
    for ( i = &v24[2 * v395]; i != v24; v24 += 2 )
    {
      if ( v24 )
        *v24 = -4096;
    }
    v19 = (unsigned int)((a3[5] - a3[4]) >> 3);
  }
  src = 0;
  v398 = 0;
  v399 = 0;
  if ( v19 )
  {
    v59 = 8 * v19;
    v60 = (char *)sub_22077B0(v59);
    v61 = src;
    v62 = v60;
    if ( v398 - (_BYTE *)src > 0 )
    {
      memmove(v60, src, v398 - (_BYTE *)src);
    }
    else if ( !src )
    {
LABEL_107:
      src = v62;
      v398 = v62;
      v399 = &v62[v59];
      goto LABEL_108;
    }
    j_j___libc_free_0((unsigned __int64)v61);
    goto LABEL_107;
  }
LABEL_108:
  v63 = (unsigned __int64 *)v348;
  v422 = 0;
  v400[0] = (unsigned __int64)v401;
  v400[1] = 0x1000000000LL;
  v443 = 0x1000000000LL;
  v428 = &v432;
  v442 = (unsigned __int64 *)v444;
  v447 = &v451;
  v438 = 0;
  v445 = 0;
  v452 = (__int64 *)v454;
  v424 = v348;
  v453 = 0x800000000LL;
  v423 = 0;
  v425 = 0;
  LOBYTE(v426) = 0;
  v427 = 0;
  v429 = 8;
  v430 = 0;
  v431 = 1;
  v439 = 0;
  v440 = 0;
  v441 = 0;
  v446 = 0;
  v448 = 8;
  v449 = 0;
  v450 = 1;
  v455 = 0;
  v456 = &v460;
  v461 = (__int64 *)v463;
  v457 = 8;
  v458 = 0;
  v459 = 1;
  v462 = 0x800000000LL;
  v464 = 0;
  v465 = &v469;
  v466 = 8;
  v467 = 0;
  v468 = 1;
  v470 = (__int64 *)v472;
  v471 = 0x800000000LL;
  sub_2846420((__int64 *)&v387);
  if ( (_BYTE)v445 )
    goto LABEL_157;
  if ( !(_DWORD)v471 )
    goto LABEL_157;
  if ( HIBYTE(v445) )
    goto LABEL_157;
  v65 = v387;
  v64 = (unsigned int)v453 + (unsigned __int64)(unsigned int)(HIDWORD(v466) - v467);
  if ( v64 != (unsigned int)((v387[5] - v387[4]) >> 3) )
    goto LABEL_157;
  if ( (_DWORD)v462 )
  {
    v63 = (unsigned __int64 *)v389;
    if ( sub_D48C30((__int64)v387, v389, 0) )
    {
      v65 = v387;
      goto LABEL_115;
    }
LABEL_157:
    v350 = 0;
    v117 = 0;
    goto LABEL_158;
  }
LABEL_115:
  sub_DAC8B0(v390, v65);
  if ( (_DWORD)v453 )
    sub_D9D700(v390, 0);
  if ( (_DWORD)v462 )
  {
    LOWORD(v371) = 257;
    v334 = sub_D4B130((__int64)v387);
    v70 = v334 + 48;
    v71 = *(_QWORD *)(v334 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v334 + 48 == v71 )
    {
      v73 = 0;
    }
    else
    {
      if ( !v71 )
        goto LABEL_543;
      v72 = *(unsigned __int8 *)(v71 - 24);
      v73 = v71 - 24;
      if ( (unsigned int)(v72 - 30) >= 0xB )
        v73 = 0;
    }
    v340 = sub_F36960(v334, (__int64 *)(v73 + 24), 0, v389, v388, v391[0], (void **)&v367, 0);
    v74 = *(_QWORD *)(v334 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v74 == v70 )
    {
      v76 = 0;
    }
    else
    {
      if ( !v74 )
        goto LABEL_543;
      v75 = *(unsigned __int8 *)(v74 - 24);
      v76 = 0;
      v77 = v74 - 24;
      if ( (unsigned int)(v75 - 30) < 0xB )
        v76 = v77;
    }
    v375 = (_QWORD *)sub_BD5C60(v76);
    v376 = &v384;
    v377 = v385;
    v384 = &unk_49DA100;
    v367 = v369;
    v368 = 0x200000000LL;
    v380 = 512;
    LOWORD(v374) = 0;
    v385[0] = &unk_49DA0B0;
    v378 = 0;
    v379 = 0;
    v381 = 7;
    v382 = 0;
    v383 = 0;
    v372 = 0;
    v373 = 0;
    sub_D5F1F0((__int64)&v367, v76);
    v78 = sub_BCB2D0(v375);
    v366 = 257;
    v79 = sub_ACD640(v78, 0, 0);
    v80 = sub_BD2DA0(80);
    v330 = v80;
    if ( v80 )
      sub_B53A60(v80, v79, v340, 10, 0, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64 **, __int64, __int64))(*v377 + 16LL))(
      v377,
      v330,
      &v361,
      v373,
      v374);
    v81 = v367;
    v82 = &v367[16 * (unsigned int)v368];
    if ( v367 != v82 )
    {
      do
      {
        v83 = *((_QWORD *)v81 + 1);
        v84 = *(_DWORD *)v81;
        v81 += 16;
        sub_B99FD0(v330, v84, v83);
      }
      while ( v82 != v81 );
    }
    v85 = *(_QWORD *)(v334 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v85 == v70 )
    {
      v87 = 0;
    }
    else
    {
      if ( !v85 )
        goto LABEL_543;
      v86 = *(unsigned __int8 *)(v85 - 24);
      v87 = 0;
      v88 = (_QWORD *)(v85 - 24);
      if ( (unsigned int)(v86 - 30) < 0xB )
        v87 = v88;
    }
    sub_B43D60(v87);
    v332 = 1;
    v353 = v461;
    v329 = &v461[(unsigned int)v462];
    if ( v461 != v329 )
    {
      while ( 1 )
      {
        v89 = *v353;
        v361 = &v363;
        v362 = 0x400000000LL;
        v90 = sub_AA5930(v89);
        v94 = v93;
        v95 = (unsigned int)v362;
        if ( v90 != v93 )
        {
          while ( 1 )
          {
            if ( v95 + 1 > (unsigned __int64)HIDWORD(v362) )
            {
              sub_C8D5F0((__int64)&v361, &v363, v95 + 1, 8u, v91, v92);
              v95 = (unsigned int)v362;
            }
            v361[v95] = v90;
            v95 = (unsigned int)(v362 + 1);
            LODWORD(v362) = v362 + 1;
            if ( !v90 )
              BUG();
            v96 = *(_QWORD *)(v90 + 32);
            if ( !v96 )
              break;
            v90 = 0;
            if ( *(_BYTE *)(v96 - 24) == 84 )
              v90 = v96 - 24;
            if ( v94 == v90 )
              goto LABEL_143;
          }
LABEL_543:
          BUG();
        }
LABEL_143:
        v97 = sub_AA4FF0(v89);
        if ( !v97 )
LABEL_232:
          BUG();
        v99 = (unsigned int)v362;
        if ( *(_BYTE *)(v97 - 24) != 95 )
          goto LABEL_236;
        v100 = v97 - 24;
        v101 = (unsigned int)v362;
        if ( HIDWORD(v362) <= (unsigned __int64)(unsigned int)v362 )
          break;
        v102 = (__int64)v361;
        v103 = &v361[(unsigned int)v362];
        if ( v103 )
        {
          *v103 = v100;
          LODWORD(v99) = v362;
          v102 = (__int64)v361;
        }
        v99 = (unsigned int)(v99 + 1);
        LODWORD(v362) = v99;
LABEL_149:
        v104 = (__int64 *)(v102 + 8 * v99);
        v105 = (__int64 *)v102;
        while ( v104 != v105 )
        {
          v106 = *v105++;
          sub_DAC8D0(v390, (_BYTE *)v106);
          v107 = sub_ACADE0(*(__int64 ***)(v106 + 8));
          sub_BD84D0(v106, v107);
          sub_B43D60((_QWORD *)v106);
        }
        v108 = sub_BCB2D0(v375);
        v109 = sub_ACD640(v108, v332, 0);
        v110 = v89;
        v111 = v89 & 0xFFFFFFFFFFFFFFFBLL;
        sub_B53E30(v330, v109, v110);
        v114 = (unsigned int)v443;
        v115 = (unsigned int)v443 + 1LL;
        if ( v115 > HIDWORD(v443) )
        {
          sub_C8D5F0((__int64)&v442, v444, v115, 0x10u, v112, v113);
          v114 = (unsigned int)v443;
        }
        v116 = &v442[2 * v114];
        v116[1] = v111;
        *v116 = v334;
        LODWORD(v443) = v443 + 1;
        if ( v361 != &v363 )
          _libc_free((unsigned __int64)v361);
        if ( v329 == ++v353 )
          goto LABEL_237;
        ++v332;
      }
      if ( HIDWORD(v362) < (unsigned __int64)(unsigned int)v362 + 1 )
      {
        sub_C8D5F0((__int64)&v361, &v363, (unsigned int)v362 + 1LL, 8u, (unsigned int)v362 + 1LL, v98);
        v101 = (unsigned int)v362;
      }
      v361[v101] = v100;
      v99 = (unsigned int)(v362 + 1);
      LODWORD(v362) = v362 + 1;
LABEL_236:
      v102 = (__int64)v361;
      goto LABEL_149;
    }
LABEL_237:
    v141 = v388;
    v142 = *(_DWORD *)(v388 + 24);
    v143 = *(_QWORD *)(v388 + 8);
    if ( v142 )
    {
      v144 = v142 - 1;
      v145 = (v142 - 1) & (((unsigned int)v334 >> 9) ^ ((unsigned int)v334 >> 4));
      v146 = (__int64 *)(v143 + 16LL * v145);
      v147 = *v146;
      if ( v334 != *v146 )
      {
        for ( j = 1; ; j = v306 )
        {
          if ( v147 == -4096 )
            goto LABEL_279;
          v306 = j + 1;
          v145 = v144 & (j + v145);
          v146 = (__int64 *)(v143 + 16LL * v145);
          v147 = *v146;
          if ( v334 == *v146 )
            break;
        }
      }
      v335 = (__int64 *)v146[1];
      if ( v335 )
      {
        v148 = v456;
        if ( v459 )
          v149 = &v456[8 * HIDWORD(v457)];
        else
          v149 = &v456[8 * (unsigned int)v457];
        if ( v456 != v149 )
        {
          while ( 1 )
          {
            v150 = *(_QWORD *)v148;
            if ( *(_QWORD *)v148 < 0xFFFFFFFFFFFFFFFELL )
              break;
            v148 += 8;
            if ( v149 == v148 )
              goto LABEL_245;
          }
          if ( v149 != v148 )
          {
            v354 = 0;
            v280 = v149;
            v281 = v148;
            v282 = (__int64)v387;
LABEL_430:
            if ( !v142 )
            {
              if ( !v282 )
                goto LABEL_541;
              goto LABEL_441;
            }
            v283 = v142 - 1;
            v284 = (v142 - 1) & (((unsigned int)v150 >> 9) ^ ((unsigned int)v150 >> 4));
            v285 = (__int64 *)(v143 + 16LL * v284);
            v286 = *v285;
            if ( v150 != *v285 )
            {
              for ( k = 1; ; k = v307 )
              {
                if ( v286 == -4096 )
                {
                  if ( v282 )
                    goto LABEL_441;
                  goto LABEL_457;
                }
                v307 = k + 1;
                v284 = v283 & (k + v284);
                v285 = (__int64 *)(v143 + 16LL * v284);
                v286 = *v285;
                if ( v150 == *v285 )
                  break;
              }
            }
            v287 = (_QWORD *)v285[1];
            if ( !v287 )
            {
              if ( !v282 )
                goto LABEL_541;
              goto LABEL_441;
            }
            while ( 1 )
            {
              v288 = **(_QWORD **)(v282 + 32);
              if ( *((_BYTE *)v287 + 84) )
              {
                v289 = (_QWORD *)v287[8];
                v290 = &v289[*((unsigned int *)v287 + 19)];
                if ( v289 != v290 )
                {
                  while ( v288 != *v289 )
                  {
                    if ( v290 == ++v289 )
                      goto LABEL_450;
                  }
LABEL_438:
                  if ( (_QWORD *)v282 != v287 )
                  {
LABEL_439:
                    if ( !v354 )
                    {
                      v354 = (__int64)v287;
                      goto LABEL_441;
                    }
                    v297 = (_QWORD *)*v287;
                    v298 = *(_QWORD **)v354;
                    if ( *v287 )
                    {
                      v299 = 1;
                      do
                      {
                        v297 = (_QWORD *)*v297;
                        ++v299;
                      }
                      while ( v297 );
                      if ( v298 )
                        goto LABEL_463;
                      v300 = 1;
                    }
                    else
                    {
                      if ( !v298 )
                        goto LABEL_441;
                      v299 = 1;
LABEL_463:
                      v300 = 1;
                      do
                      {
                        v298 = (_QWORD *)*v298;
                        ++v300;
                      }
                      while ( v298 );
                    }
                    if ( v299 <= v300 )
                      v287 = (_QWORD *)v354;
                    v354 = (__int64)v287;
                    goto LABEL_441;
                  }
LABEL_457:
                  v287 = *(_QWORD **)v282;
                  if ( *(_QWORD *)v282 )
                    goto LABEL_439;
LABEL_441:
                  v291 = v281 + 8;
                  if ( v281 + 8 != v280 )
                  {
                    while ( 1 )
                    {
                      v150 = *(_QWORD *)v291;
                      v281 = v291;
                      if ( *(_QWORD *)v291 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      v291 += 8;
                      if ( v280 == v291 )
                        goto LABEL_444;
                    }
                    if ( v280 != v291 )
                    {
                      v143 = *(_QWORD *)(v141 + 8);
                      v142 = *(_DWORD *)(v141 + 24);
                      goto LABEL_430;
                    }
                  }
LABEL_444:
                  if ( v335 == (__int64 *)v354 )
                    goto LABEL_279;
                  v141 = v388;
                  v143 = *(_QWORD *)(v388 + 8);
                  v292 = *(_DWORD *)(v388 + 24);
                  if ( v354 )
                  {
                    if ( v292 )
                    {
                      v293 = (v292 - 1) & (((unsigned int)v340 >> 4) ^ ((unsigned int)v340 >> 9));
                      v294 = (__int64 *)(v143 + 16LL * v293);
                      v295 = *v294;
                      if ( v340 == *v294 )
                        goto LABEL_448;
                      v316 = 1;
                      v317 = 0;
                      while ( v295 != -4096 )
                      {
                        if ( !v317 && v295 == -8192 )
                          v317 = v294;
                        v293 = (v292 - 1) & (v316 + v293);
                        v294 = (__int64 *)(v143 + 16LL * v293);
                        v295 = *v294;
                        if ( v340 == *v294 )
                          goto LABEL_448;
                        ++v316;
                      }
                      v318 = *(_DWORD *)(v388 + 16);
                      if ( v317 )
                        v294 = v317;
                      ++*(_QWORD *)v388;
                      v315 = v318 + 1;
                      if ( 4 * (v318 + 1) < 3 * v292 )
                      {
                        if ( v292 - *(_DWORD *)(v141 + 20) - v315 <= v292 >> 3 )
                        {
                          sub_D4F150(v141, v292);
                          v319 = *(_DWORD *)(v141 + 24);
                          if ( v319 )
                          {
                            v320 = v319 - 1;
                            v322 = 1;
                            v294 = 0;
                            for ( m = v320 & (((unsigned int)v340 >> 4) ^ ((unsigned int)v340 >> 9)); ; m = v320 & v327 )
                            {
                              v321 = *(_QWORD *)(v141 + 8);
                              v324 = (__int64 *)(v321 + 16LL * m);
                              v325 = *v324;
                              if ( v340 == *v324 )
                              {
                                v315 = *(_DWORD *)(v141 + 16) + 1;
                                v294 = (__int64 *)(v321 + 16LL * m);
                                goto LABEL_505;
                              }
                              if ( v325 == -4096 )
                                break;
                              if ( v294 || v325 != -8192 )
                                v324 = v294;
                              v327 = v322 + m;
                              v294 = v324;
                              ++v322;
                            }
                            if ( !v294 )
                              v294 = (__int64 *)(v321 + 16LL * m);
                            v315 = *(_DWORD *)(v141 + 16) + 1;
                            goto LABEL_505;
                          }
LABEL_542:
                          ++*(_DWORD *)(v141 + 16);
                          BUG();
                        }
LABEL_505:
                        *(_DWORD *)(v141 + 16) = v315;
                        if ( *v294 != -4096 )
                          --*(_DWORD *)(v141 + 20);
                        v294[1] = 0;
                        *v294 = v340;
LABEL_448:
                        v294[1] = v354;
LABEL_248:
                        v154 = v335;
                        do
                        {
                          v361 = (__int64 *)v340;
                          v155 = sub_2845980((_QWORD *)v154[4], v154[5], (__int64 *)&v361);
                          sub_F681A0((__int64)(v154 + 4), v155);
                          if ( *((_BYTE *)v154 + 84) )
                          {
                            v156 = (__int64 **)v154[8];
                            v157 = &v156[*((unsigned int *)v154 + 19)];
                            v158 = v156;
                            if ( v156 != v157 )
                            {
                              while ( v361 != *v158 )
                              {
                                if ( v157 == ++v158 )
                                  goto LABEL_255;
                              }
                              v159 = (unsigned int)(*((_DWORD *)v154 + 19) - 1);
                              *((_DWORD *)v154 + 19) = v159;
                              *v158 = v156[v159];
                              ++v154[7];
                            }
                          }
                          else
                          {
                            v279 = sub_C8CA60((__int64)(v154 + 7), (__int64)v361);
                            if ( v279 )
                            {
                              *v279 = -2;
                              ++*((_DWORD *)v154 + 20);
                              ++v154[7];
                            }
                          }
LABEL_255:
                          v154 = (__int64 *)*v154;
                        }
                        while ( v154 != (__int64 *)v354 );
                        v160 = (__int64)v387;
                        v161 = (__int64 *)v387[4];
                        if ( v161 != (__int64 *)v387[5] )
                        {
                          v341 = (__int64 *)v387[5];
                          do
                          {
                            v162 = *v161;
                            for ( n = v335; n != (__int64 *)v354; n = (__int64 *)*n )
                            {
                              v361 = (__int64 *)v162;
                              v164 = sub_2845980((_QWORD *)n[4], n[5], (__int64 *)&v361);
                              sub_F681A0((__int64)(n + 4), v164);
                              if ( *((_BYTE *)n + 84) )
                              {
                                v165 = (__int64 **)n[8];
                                v166 = &v165[*((unsigned int *)n + 19)];
                                v167 = v165;
                                if ( v165 != v166 )
                                {
                                  while ( v361 != *v167 )
                                  {
                                    if ( v166 == ++v167 )
                                      goto LABEL_265;
                                  }
                                  v168 = (unsigned int)(*((_DWORD *)n + 19) - 1);
                                  *((_DWORD *)n + 19) = v168;
                                  *v167 = v165[v168];
                                  ++n[7];
                                }
                              }
                              else
                              {
                                v272 = sub_C8CA60((__int64)(n + 7), (__int64)v361);
                                if ( v272 )
                                {
                                  *v272 = -2;
                                  ++*((_DWORD *)n + 20);
                                  ++n[7];
                                }
                              }
LABEL_265:
                              ;
                            }
                            ++v161;
                          }
                          while ( v341 != v161 );
                          v160 = (__int64)v387;
                        }
                        v361 = (__int64 *)v160;
                        v170 = sub_28458C0((_QWORD *)v335[1], v335[2], (__int64 *)&v361);
                        v173 = (_QWORD *)*v170;
                        v175 = *(char **)(v174 + 16);
                        v176 = (char *)(v170 + 1);
                        if ( v175 != (char *)(v170 + 1) )
                        {
                          memmove(v170, v176, v175 - v176);
                          v176 = (char *)v335[2];
                        }
                        v335[2] = (__int64)(v176 - 8);
                        *v173 = 0;
                        if ( v354 )
                        {
                          v361 = v387;
                          *v387 = v354;
                          v177 = *(_BYTE **)(v354 + 16);
                          if ( v177 == *(_BYTE **)(v354 + 24) )
                          {
                            sub_D4C7F0(v354 + 8, v177, &v361);
                            v178 = v335;
                          }
                          else
                          {
                            if ( v177 )
                              *(_QWORD *)v177 = v361;
                            *(_QWORD *)(v354 + 16) += 8LL;
                            v178 = v335;
                          }
                        }
                        else
                        {
                          v303 = v388;
                          v361 = v387;
                          v304 = *(_BYTE **)(v388 + 40);
                          if ( v304 == *(_BYTE **)(v388 + 48) )
                          {
                            sub_D4C7F0(v388 + 32, v304, &v361);
                            v178 = v335;
                          }
                          else
                          {
                            if ( v304 )
                              *(_QWORD *)v304 = v387;
                            *(_QWORD *)(v303 + 40) += 8LL;
                            v178 = v335;
                          }
                        }
                        do
                        {
                          v179 = (__int64)v178;
                          v178 = (__int64 *)*v178;
                        }
                        while ( v178 != (__int64 *)v354 );
                        if ( v391[0] )
                          sub_D75690(v391[0], v442, (unsigned int)v443, v389, 1);
                        else
                          sub_FFB3D0((__int64)v400, v442, (unsigned int)v443, v169, v171, v172);
                        LODWORD(v443) = 0;
                        sub_11D2180(v179, v389, v388, v390, v180, v181);
                        sub_D9D700(v390, 0);
                        goto LABEL_279;
                      }
                    }
                    else
                    {
                      ++*(_QWORD *)v388;
                    }
                    sub_D4F150(v141, 2 * v292);
                    v309 = *(_DWORD *)(v141 + 24);
                    if ( v309 )
                    {
                      v310 = v309 - 1;
                      v311 = 1;
                      v312 = 0;
                      for ( ii = v310 & (((unsigned int)v340 >> 9) ^ ((unsigned int)v340 >> 4)); ; ii = v310 & v326 )
                      {
                        v294 = (__int64 *)(*(_QWORD *)(v141 + 8) + 16LL * ii);
                        v314 = *v294;
                        if ( v340 == *v294 )
                        {
                          v315 = *(_DWORD *)(v141 + 16) + 1;
                          goto LABEL_505;
                        }
                        if ( v314 == -4096 )
                          break;
                        if ( v312 || v314 != -8192 )
                          v294 = v312;
                        v326 = v311 + ii;
                        v312 = v294;
                        ++v311;
                      }
                      if ( v312 )
                        v294 = v312;
                      v315 = *(_DWORD *)(v141 + 16) + 1;
                      goto LABEL_505;
                    }
                    goto LABEL_542;
                  }
                  if ( !v292 )
                    goto LABEL_248;
                  v144 = v292 - 1;
                  break;
                }
              }
              else if ( sub_C8CA60((__int64)(v287 + 7), v288) )
              {
                goto LABEL_438;
              }
LABEL_450:
              v287 = (_QWORD *)*v287;
              if ( !v287 )
                goto LABEL_441;
            }
          }
        }
LABEL_245:
        v151 = 1;
        for ( jj = v144 & (((unsigned int)v340 >> 9) ^ ((unsigned int)v340 >> 4)); ; jj = v144 & v308 )
        {
          v153 = (_QWORD *)(v143 + 16LL * jj);
          if ( v340 == *v153 )
          {
            *v153 = -8192;
            --*(_DWORD *)(v141 + 16);
            ++*(_DWORD *)(v141 + 20);
            v354 = 0;
            goto LABEL_248;
          }
          if ( *v153 == -4096 )
            break;
          v308 = v151 + jj;
          ++v151;
        }
        v354 = 0;
        goto LABEL_248;
      }
    }
LABEL_279:
    if ( v391[0] )
    {
      sub_D75690(v391[0], v442, (unsigned int)v443, v389, 1);
      LODWORD(v443) = 0;
      if ( byte_4F8F8E8[0] )
        nullsub_390();
    }
    nullsub_61();
    v384 = &unk_49DA100;
    nullsub_63();
    if ( v367 != v369 )
      _libc_free((unsigned __int64)v367);
  }
  v182 = &v442;
  v342 = v470;
  v331 = &v470[(unsigned int)v471];
  if ( v470 != v331 )
  {
    while ( 1 )
    {
      v183 = *v342;
      v184 = sub_2845AB0(*v342);
      v365 = 1;
      v185 = v184;
      v361 = 0;
      v363 = 2;
      v362 = (unsigned __int64)&v366;
      v364 = 0;
      v186 = *(_QWORD *)(v183 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v183 + 48 != v186 )
      {
        if ( !v186 )
          goto LABEL_232;
        v187 = v186 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v186 - 24) - 30 <= 0xA )
        {
          v355 = sub_B46E30(v187);
          if ( v355 )
            break;
        }
      }
      v346 = 0;
LABEL_305:
      v199 = *((_BYTE *)v387 + 84);
      if ( v199 )
      {
        v200 = (_QWORD *)v387[8];
        v201 = &v200[*((unsigned int *)v387 + 19)];
        if ( v200 != v201 )
        {
          while ( v185 != *v200 )
          {
            if ( v201 == ++v200 )
              goto LABEL_311;
          }
          v199 = 0;
        }
      }
      else
      {
        v199 = sub_C8CA60((__int64)(v387 + 7), v185) == 0;
      }
LABEL_311:
      v202 = 1;
      if ( v346 > 1 )
      {
        do
        {
          ++v202;
          sub_AA5980(v185, v183, v199);
        }
        while ( v202 != v346 );
        if ( v391[0] )
          sub_D6D880(v391[0], v183, v185);
      }
      v375 = (_QWORD *)sub_AA48A0(v183);
      v367 = v369;
      v384 = &unk_49DA100;
      v368 = 0x200000000LL;
      v376 = &v384;
      v377 = v385;
      v378 = 0;
      v379 = 0;
      v380 = 512;
      v381 = 7;
      v382 = 0;
      v383 = 0;
      v372 = 0;
      v373 = 0;
      LOWORD(v374) = 0;
      v385[0] = &unk_49DA0B0;
      v203 = *(_QWORD *)(v183 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v183 + 48 == v203 )
      {
        v336 = 0;
        sub_D5F1F0((__int64)&v367, 0);
      }
      else
      {
        if ( !v203 )
          goto LABEL_543;
        v204 = *(unsigned __int8 *)(v203 - 24);
        v205 = v203 - 24;
        v206 = (unsigned int)(v204 - 30) < 0xB;
        v207 = 0;
        if ( v206 )
          v207 = v205;
        v336 = (_QWORD *)v207;
        sub_D5F1F0((__int64)&v367, v207);
      }
      v360 = 257;
      v208 = sub_BD2C40(72, 1u);
      v209 = (__int64)v208;
      if ( v208 )
        sub_B4C8F0((__int64)v208, v185, 1u, 0, 0);
      (*(void (__fastcall **)(_QWORD *, __int64, char *, __int64, __int64))(*v377 + 16LL))(v377, v209, v359, v373, v374);
      v210 = 16LL * (unsigned int)v368;
      if ( v367 != &v367[v210] )
      {
        v333 = v183;
        v211 = &v367[v210];
        v212 = v367;
        do
        {
          v213 = *((_QWORD *)v212 + 1);
          v214 = *(_DWORD *)v212;
          v212 += 16;
          sub_B99FD0(v209, v214, v213);
        }
        while ( v211 != v212 );
        v183 = v333;
      }
      sub_B43D60(v336);
      v216 = (_QWORD *)v362;
      if ( v365 )
        v217 = (_QWORD *)(v362 + 8LL * HIDWORD(v363));
      else
        v217 = (_QWORD *)(v362 + 8LL * (unsigned int)v363);
      if ( (_QWORD *)v362 != v217 )
      {
        while ( 1 )
        {
          v218 = *v216;
          v219 = v216;
          if ( *v216 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v217 == ++v216 )
            goto LABEL_331;
        }
        if ( v216 != v217 )
        {
          v248 = v443;
          v249 = v183;
          do
          {
            v250 = v248;
            v251 = v218 | 4;
            if ( (unsigned __int64)v248 + 1 > HIDWORD(v443) )
            {
              v338 = v249;
              sub_C8D5F0((__int64)&v442, v444, v248 + 1LL, 0x10u, v249, v215);
              v250 = (unsigned int)v443;
              v249 = v338;
            }
            v252 = &v442[2 * v250];
            *v252 = v249;
            v252[1] = v251;
            v248 = v443 + 1;
            v253 = v219 + 1;
            LODWORD(v443) = v443 + 1;
            if ( v219 + 1 == v217 )
              break;
            v218 = *v253;
            for ( ++v219; *v253 >= 0xFFFFFFFFFFFFFFFELL; v219 = v253 )
            {
              if ( v217 == ++v253 )
                goto LABEL_331;
              v218 = *v253;
            }
          }
          while ( v219 != v217 );
        }
      }
LABEL_331:
      nullsub_61();
      v384 = &unk_49DA100;
      nullsub_63();
      if ( v367 != v369 )
        _libc_free((unsigned __int64)v367);
      if ( !v365 )
        _libc_free(v362);
      if ( v331 == ++v342 )
        goto LABEL_336;
    }
    v346 = 0;
    v188 = 0;
    while ( 1 )
    {
      v193 = sub_B46EC0(v187, v188);
      if ( v185 != v193 )
        break;
      ++v346;
LABEL_304:
      if ( ++v188 == v355 )
        goto LABEL_305;
    }
    if ( v365 )
    {
      v194 = (_QWORD *)v362;
      v190 = HIDWORD(v363);
      v189 = (__int64 *)(v362 + 8LL * HIDWORD(v363));
      if ( (__int64 *)v362 != v189 )
      {
        while ( v193 != *v194 )
        {
          if ( v189 == ++v194 )
            goto LABEL_371;
        }
LABEL_296:
        v195 = (__int64)v387;
        if ( *((_BYTE *)v387 + 84) )
          goto LABEL_297;
        goto LABEL_370;
      }
LABEL_371:
      if ( HIDWORD(v363) < (unsigned int)v363 )
      {
        ++HIDWORD(v363);
        *v189 = v193;
        v361 = (__int64 *)((char *)v361 + 1);
        goto LABEL_296;
      }
    }
    sub_C8CC70((__int64)&v361, v193, (__int64)v189, v190, v191, v192);
    v195 = (__int64)v387;
    if ( *((_BYTE *)v387 + 84) )
    {
LABEL_297:
      v196 = *(_QWORD **)(v195 + 64);
      v197 = &v196[*(unsigned int *)(v195 + 76)];
      if ( v196 == v197 )
      {
LABEL_373:
        v198 = 1;
      }
      else
      {
        while ( v193 != *v196 )
        {
          if ( v197 == ++v196 )
            goto LABEL_373;
        }
        v198 = 0;
      }
      goto LABEL_302;
    }
LABEL_370:
    v198 = sub_C8CA60(v195 + 56, v193) == 0;
LABEL_302:
    sub_AA5980(v193, v183, v198);
    if ( v391[0] )
      sub_D6D7F0(v391[0], v183, v193);
    goto LABEL_304;
  }
LABEL_336:
  v220 = (unsigned int)v453;
  if ( (_DWORD)v453 )
  {
    if ( v391[0] )
    {
      sub_2848130((__int64)&v367, v452, &v452[(unsigned int)v453], (__int64)v182, (__int64)v452, v69);
      sub_D6F970(v391[0], (__int64)&v367);
      if ( v371 != &v372 )
        _libc_free((unsigned __int64)v371);
      sub_C7D6A0(v368, 8LL * v370, 8);
      v220 = (unsigned int)v453;
    }
    v337 = &v452[v220];
    if ( v452 != v337 )
    {
      v347 = v452;
      do
      {
        v221 = v388;
        v222 = *(_DWORD *)(v388 + 24);
        v223 = *v347;
        v224 = *(_QWORD *)(v388 + 8);
        if ( v222 )
        {
          v225 = v222 - 1;
          v226 = v225 & (((unsigned int)v223 >> 9) ^ ((unsigned int)v223 >> 4));
          v227 = (__int64 *)(v224 + 16LL * v226);
          v228 = *v227;
          v229 = v227;
          if ( v223 == *v227 )
          {
LABEL_346:
            v230 = v229[1];
            if ( v230 && v223 == **(_QWORD **)(v230 + 32) )
            {
              if ( v223 != v228 )
              {
                v301 = 1;
                while ( v228 != -4096 )
                {
                  v302 = v301 + 1;
                  v226 = v225 & (v301 + v226);
                  v227 = (__int64 *)(v224 + 16LL * v226);
                  v228 = *v227;
                  if ( v223 == *v227 )
                    goto LABEL_349;
                  v301 = v302;
                }
LABEL_541:
                BUG();
              }
LABEL_349:
              v231 = (__int64 *)v227[1];
              v232 = *v231;
              v343 = v231;
              if ( *v231 )
              {
                do
                {
                  v233 = (void **)v343[5];
                  for ( kk = (void **)v343[4]; v233 != kk; ++kk )
                  {
                    v367 = (char *)*kk;
                    v235 = sub_2845980(*(_QWORD **)(v232 + 32), *(_QWORD *)(v232 + 40), (__int64 *)&v367);
                    sub_F681A0(v232 + 32, v235);
                    if ( *(_BYTE *)(v232 + 84) )
                    {
                      v236 = *(void ***)(v232 + 64);
                      v237 = &v236[*(unsigned int *)(v232 + 76)];
                      v238 = v236;
                      if ( v236 != v237 )
                      {
                        while ( v367 != *v238 )
                        {
                          if ( v237 == ++v238 )
                            goto LABEL_358;
                        }
                        v239 = (unsigned int)(*(_DWORD *)(v232 + 76) - 1);
                        *(_DWORD *)(v232 + 76) = v239;
                        *v238 = v236[v239];
                        ++*(_QWORD *)(v232 + 56);
                      }
                    }
                    else
                    {
                      v247 = sub_C8CA60(v232 + 56, (__int64)v367);
                      if ( v247 )
                      {
                        *v247 = -2;
                        ++*(_DWORD *)(v232 + 80);
                        ++*(_QWORD *)(v232 + 56);
                      }
                    }
LABEL_358:
                    ;
                  }
                  v232 = *(_QWORD *)v232;
                }
                while ( v232 );
                v240 = *v343;
                v367 = (char *)v343;
                v241 = sub_28458C0(*(_QWORD **)(v240 + 8), *(_QWORD *)(v240 + 16), (__int64 *)&v367);
                v242 = (_QWORD *)*v241;
                v243 = *(char **)(v240 + 16);
                v244 = (char *)(v241 + 1);
                if ( v243 != (char *)(v241 + 1) )
                {
                  memmove(v241, v244, v243 - v244);
                  v244 = *(char **)(v240 + 16);
                }
                *(_QWORD *)(v240 + 16) = v244 - 8;
                *v242 = 0;
                v245 = v388;
                v367 = (char *)v343;
                v246 = *(_BYTE **)(v388 + 40);
                if ( v246 == *(_BYTE **)(v388 + 48) )
                {
                  sub_D4C7F0(v388 + 32, v246, &v367);
                  v221 = v388;
                }
                else
                {
                  v221 = v388;
                  if ( v246 )
                  {
                    *(_QWORD *)v246 = v343;
                    v246 = *(_BYTE **)(v245 + 40);
                    v221 = v388;
                  }
                  *(_QWORD *)(v245 + 40) = v246 + 8;
                }
              }
              sub_D4F720(v221, v343);
            }
          }
          else
          {
            v274 = *v227;
            v275 = v225 & (((unsigned int)v223 >> 9) ^ ((unsigned int)v223 >> 4));
            v276 = 1;
            while ( v274 != -4096 )
            {
              v277 = v276 + 1;
              v275 = v225 & (v275 + v276);
              v229 = (__int64 *)(v224 + 16LL * v275);
              v274 = *v229;
              if ( v223 == *v229 )
                goto LABEL_346;
              v276 = v277;
            }
          }
        }
        ++v347;
      }
      while ( v337 != v347 );
      v220 = (unsigned int)v453;
      v337 = &v452[(unsigned int)v453];
      if ( v452 != v337 )
      {
        v356 = (void **)v452;
        do
        {
          v254 = v388;
          v255 = (char *)*v356;
          v256 = *(unsigned int *)(v388 + 24);
          v257 = *(_QWORD *)(v388 + 8);
          if ( (_DWORD)v256 )
          {
            v258 = (v256 - 1) & (((unsigned int)v255 >> 9) ^ ((unsigned int)v255 >> 4));
            v259 = (void **)(v257 + 16LL * v258);
            v260 = (char *)*v259;
            if ( v255 == *v259 )
            {
LABEL_397:
              if ( v259 != (void **)(v257 + 16 * v256) )
              {
                for ( mm = (__int64 *)v259[1]; mm; mm = (__int64 *)*mm )
                {
                  v367 = v255;
                  v262 = sub_2845980((_QWORD *)mm[4], mm[5], (__int64 *)&v367);
                  sub_F681A0((__int64)(mm + 4), v262);
                  if ( *((_BYTE *)mm + 84) )
                  {
                    v263 = (void **)mm[8];
                    v264 = &v263[*((unsigned int *)mm + 19)];
                    v265 = v263;
                    if ( v263 != v264 )
                    {
                      while ( v367 != *v265 )
                      {
                        if ( v264 == ++v265 )
                          goto LABEL_405;
                      }
                      v266 = (unsigned int)(*((_DWORD *)mm + 19) - 1);
                      *((_DWORD *)mm + 19) = v266;
                      *v265 = v263[v266];
                      ++mm[7];
                    }
                  }
                  else
                  {
                    v273 = sub_C8CA60((__int64)(mm + 7), (__int64)v367);
                    if ( v273 )
                    {
                      *v273 = -2;
                      ++*((_DWORD *)mm + 20);
                      ++mm[7];
                    }
                  }
LABEL_405:
                  ;
                }
                *v259 = (void *)-8192LL;
                --*(_DWORD *)(v254 + 16);
                ++*(_DWORD *)(v254 + 20);
              }
            }
            else
            {
              v278 = 1;
              while ( v260 != (char *)-4096LL )
              {
                v258 = (v256 - 1) & (v278 + v258);
                v259 = (void **)(v257 + 16LL * v258);
                v260 = (char *)*v259;
                if ( v255 == *v259 )
                  goto LABEL_397;
                ++v278;
              }
            }
          }
          ++v356;
        }
        while ( v337 != (__int64 *)v356 );
        v220 = (unsigned int)v453;
        v337 = v452;
      }
    }
    sub_F34190(v337, v220, (__int64)&v442, 1u);
    v63 = v442;
    sub_FFB3D0((__int64)v400, v442, (unsigned int)v443, v267, v268, v269);
    v270 = (unsigned __int64 **)v452;
    LODWORD(v443) = 0;
    v271 = &v452[(unsigned int)v453];
    if ( v452 != v271 )
    {
      do
      {
        v63 = *v270++;
        sub_FFBF00((__int64)v400, v63);
      }
      while ( v271 != (__int64 *)v270 );
    }
  }
  else
  {
    v63 = v442;
    sub_FFB3D0((__int64)v400, v442, (unsigned int)v443, (__int64)v182, v68, v69);
    LODWORD(v443) = 0;
  }
  if ( v391[0] )
  {
    v64 = (__int64)byte_4F8F8E8;
    if ( byte_4F8F8E8[0] )
    {
      v63 = 0;
      nullsub_390();
    }
  }
  v117 = HIBYTE(v445);
LABEL_158:
  if ( v470 != (__int64 *)v472 )
    _libc_free((unsigned __int64)v470);
  if ( !v468 )
    _libc_free((unsigned __int64)v465);
  if ( v461 != (__int64 *)v463 )
    _libc_free((unsigned __int64)v461);
  if ( !v459 )
    _libc_free((unsigned __int64)v456);
  if ( v452 != (__int64 *)v454 )
    _libc_free((unsigned __int64)v452);
  if ( !v450 )
    _libc_free((unsigned __int64)v447);
  if ( v442 != (unsigned __int64 *)v444 )
    _libc_free((unsigned __int64)v442);
  sub_FFCE90((__int64)v400, (__int64)v63, v64, (__int64)v65, v66, v67);
  sub_FFD870((__int64)v400, (__int64)v63, v118, v119, v120, v121);
  sub_FFBC40((__int64)v400, (__int64)v63);
  v122 = v440;
  v123 = v439;
  if ( v440 != v439 )
  {
    do
    {
      v124 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v123[7];
      *v123 = &unk_49E5048;
      if ( v124 )
        v124(v123 + 5, v123 + 5, 3);
      *v123 = &unk_49DB368;
      v125 = v123[3];
      if ( v125 != 0 && v125 != -4096 && v125 != -8192 )
        sub_BD60C0(v123 + 1);
      v123 += 9;
    }
    while ( v122 != v123 );
    v123 = v439;
  }
  if ( v123 )
    j_j___libc_free_0((unsigned __int64)v123);
  if ( !v431 )
    _libc_free((unsigned __int64)v428);
  if ( (_BYTE *)v400[0] != v401 )
    _libc_free(v400[0]);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  v6 = 16LL * v395;
  sub_C7D6A0(v393, v6, 8);
  if ( v117 )
  {
    v126 = (__int64)a3;
    sub_22D0060(*(_QWORD *)(v339 + 8), (__int64)a3, (__int64)"loop-simplifycfg", 16);
    if ( a3 == *(__int64 **)(v339 + 16) )
      *(_BYTE *)(v339 + 24) = 1;
    goto LABEL_191;
  }
LABEL_36:
  v26 = (__int64)v421;
  v411 = 0;
  v413 = v348;
  v426 = 0;
  v387 = &v389;
  v27 = (char *)a3[5];
  v28 = (char *)a3[4];
  v388 = 0x1000000000LL;
  v29 = (__int64)v369;
  v368 = 0x1000000000LL;
  v415 = 0;
  v412 = 0;
  v417 = v421;
  v414 = 0;
  v30 = (v27 - v28) >> 3;
  v416 = 0;
  v418 = 8;
  v419 = 0;
  v420 = 1;
  v427 = 0;
  v428 = 0;
  v429 = 0;
  v367 = v369;
  if ( (unsigned __int64)(v27 - v28) > 0x80 )
  {
    v6 = (v27 - v28) >> 3;
    sub_F39130((__int64)&v367, v6, (__int64)v421, (__int64)v369, (__int64)a5, a6);
    v32 = v368;
    v31 = v367;
    v29 = 3LL * (unsigned int)v368;
    v33 = &v367[24 * (unsigned int)v368];
  }
  else
  {
    v31 = v369;
    v32 = 0;
    v33 = v369;
  }
  if ( v28 != v27 )
  {
    do
    {
      if ( v33 )
      {
        v34 = *(_QWORD *)v28;
        *(_QWORD *)v33 = 6;
        *((_QWORD *)v33 + 1) = 0;
        *((_QWORD *)v33 + 2) = v34;
        LOBYTE(v29) = v34 != 0;
        LOBYTE(v26) = v34 != -4096;
        if ( ((unsigned __int8)v26 & (v34 != 0)) != 0 && v34 != -8192 )
          sub_BD73F0((__int64)v33);
      }
      v28 += 8;
      v33 += 24;
    }
    while ( v27 != v28 );
    v31 = v367;
    v32 = v368;
  }
  LODWORD(v368) = v30 + v32;
  v35 = &v31[24 * (unsigned int)(v30 + v32)];
  if ( v31 != v35 )
  {
    v36 = v31;
    v352 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v37 = *((_QWORD *)v36 + 2);
        if ( v37 )
        {
          v38 = sub_AA54C0(*((_QWORD *)v36 + 2));
          v39 = v38;
          if ( v38 )
          {
            if ( sub_AA56F0(v38) )
            {
              v6 = *(unsigned int *)(v345 + 24);
              v40 = *(_QWORD *)(v345 + 8);
              if ( (_DWORD)v6 )
              {
                v6 = (unsigned int)(v6 - 1);
                v29 = (unsigned int)v6 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                v41 = (__int64 *)(v40 + 16 * v29);
                a6 = *v41;
                if ( v39 != *v41 )
                {
                  v138 = 1;
                  while ( a6 != -4096 )
                  {
                    v26 = (unsigned int)(v138 + 1);
                    v29 = (unsigned int)v6 & (v138 + (_DWORD)v29);
                    v41 = (__int64 *)(v40 + 16LL * (unsigned int)v29);
                    a6 = *v41;
                    if ( v39 == *v41 )
                      goto LABEL_53;
                    v138 = v26;
                  }
                  goto LABEL_47;
                }
LABEL_53:
                v29 = (__int64)a3;
                if ( a3 == (__int64 *)v41[1] )
                {
                  v6 = (__int64)&v387;
                  sub_F39690(v37, (__int64)&v387, v345, v349, 0, 0, 0);
                  v26 = v328;
                  v352 = 1;
                  if ( v349 )
                    break;
                }
              }
            }
          }
        }
LABEL_47:
        v36 += 24;
        if ( v35 == v36 )
          goto LABEL_57;
      }
      v352 = byte_4F8F8E8[0];
      if ( byte_4F8F8E8[0] )
      {
        v6 = 0;
        nullsub_390();
        goto LABEL_47;
      }
      v36 += 24;
      v352 = 1;
      if ( v35 == v36 )
      {
LABEL_57:
        if ( v352 )
        {
          v6 = 0;
          sub_D9D700(v344, 0);
        }
        v42 = v367;
        v350 |= v352;
        v35 = &v367[24 * (unsigned int)v368];
        if ( v367 != v35 )
        {
          do
          {
            v43 = *((_QWORD *)v35 - 1);
            v35 -= 24;
            LOBYTE(v29) = v43 != -4096;
            LOBYTE(v26) = v43 != 0;
            if ( ((v43 != 0) & (unsigned __int8)v29) != 0 && v43 != -8192 )
              sub_BD60C0(v35);
          }
          while ( v42 != v35 );
          v35 = v367;
        }
        break;
      }
    }
  }
  if ( v35 != v369 )
    _libc_free((unsigned __int64)v35);
  sub_FFCE90((__int64)&v387, v6, v26, v29, (__int64)a5, a6);
  sub_FFD870((__int64)&v387, v6, v44, v45, v46, v47);
  sub_FFBC40((__int64)&v387, v6);
  v48 = (unsigned __int64)v428;
  v49 = v427;
  if ( v428 != v427 )
  {
    do
    {
      v50 = (void (__fastcall *)(char *, char *, __int64))*((_QWORD *)v49 + 7);
      *(_QWORD *)v49 = &unk_49E5048;
      if ( v50 )
        v50(v49 + 40, v49 + 40, 3);
      *(_QWORD *)v49 = &unk_49DB368;
      v51 = *((_QWORD *)v49 + 3);
      if ( v51 != 0 && v51 != -4096 && v51 != -8192 )
        sub_BD60C0((_QWORD *)v49 + 1);
      v49 += 72;
    }
    while ( (char *)v48 != v49 );
    v49 = v427;
  }
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( !v420 )
    _libc_free((unsigned __int64)v417);
  if ( v387 != &v389 )
    _libc_free((unsigned __int64)v387);
  if ( !v350 )
  {
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
    goto LABEL_83;
  }
  v126 = (__int64)a3;
  sub_DAC8B0(v344, a3);
LABEL_191:
  sub_22D0390((__int64)&v387, v126, v127, v128, v129, v130);
  if ( a5[9] )
  {
    if ( v396 )
    {
      v133 = (__int64 **)(v393 + 8LL * HIDWORD(v394));
      v134 = HIDWORD(v394);
      if ( (__int64 **)v393 == v133 )
      {
LABEL_216:
        v136 = v395;
      }
      else
      {
        v135 = (void **)v393;
        while ( *v135 != &unk_4F8F810 )
        {
          if ( v133 == (__int64 **)++v135 )
            goto LABEL_216;
        }
        --HIDWORD(v394);
        v133 = *(__int64 ***)(v393 + 8LL * HIDWORD(v394));
        *v135 = v133;
        v134 = HIDWORD(v394);
        ++v392;
        v136 = v395;
      }
    }
    else
    {
      v139 = sub_C8CA60((__int64)&v392, (__int64)&unk_4F8F810);
      if ( v139 )
      {
        *v139 = -2;
        ++v392;
        v134 = HIDWORD(v394);
        v136 = ++v395;
      }
      else
      {
        v134 = HIDWORD(v394);
        v136 = v395;
      }
    }
    if ( v136 == (_DWORD)v134 )
    {
      if ( BYTE4(v390) )
      {
        v137 = (void **)v388;
        v140 = (__int64 **)(v388 + 8LL * HIDWORD(v389));
        v134 = HIDWORD(v389);
        v133 = (__int64 **)v388;
        if ( (__int64 **)v388 != v140 )
        {
          while ( *v133 != &qword_4F82400 )
          {
            if ( v140 == ++v133 )
            {
LABEL_203:
              while ( *v137 != &unk_4F8F810 )
              {
                if ( v133 == (__int64 **)++v137 )
                  goto LABEL_219;
              }
              goto LABEL_204;
            }
          }
          goto LABEL_204;
        }
        goto LABEL_219;
      }
      if ( sub_C8CA60((__int64)&v387, (__int64)&qword_4F82400) )
        goto LABEL_204;
    }
    if ( !BYTE4(v390) )
      goto LABEL_221;
    v137 = (void **)v388;
    v134 = HIDWORD(v389);
    v133 = (__int64 **)(v388 + 8LL * HIDWORD(v389));
    if ( v133 != (__int64 **)v388 )
      goto LABEL_203;
LABEL_219:
    if ( (unsigned int)v389 > (unsigned int)v134 )
    {
      HIDWORD(v389) = v134 + 1;
      *v133 = (__int64 *)&unk_4F8F810;
      v387 = (__int64 *)((char *)v387 + 1);
      goto LABEL_204;
    }
LABEL_221:
    sub_C8CC70((__int64)&v387, (__int64)&unk_4F8F810, (__int64)v133, v134, v131, v132);
  }
LABEL_204:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v391, (__int64)&v387);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&src, (__int64)&v392);
  if ( !v396 )
    _libc_free(v393);
  if ( !BYTE4(v390) )
    _libc_free(v388);
LABEL_83:
  if ( LOBYTE(v386[95]) )
  {
    LOBYTE(v386[95]) = 0;
    sub_2845A40((_QWORD *)v386[91]);
    v53 = (_QWORD *)v386[63];
    v54 = (_QWORD *)(v386[63] + 24LL * LODWORD(v386[64]));
    if ( (_QWORD *)v386[63] != v54 )
    {
      do
      {
        v55 = *(v54 - 1);
        v54 -= 3;
        if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
          sub_BD60C0(v54);
      }
      while ( v53 != v54 );
      v54 = (_QWORD *)v386[63];
    }
    if ( v54 != &v386[65] )
      _libc_free((unsigned __int64)v54);
    if ( !BYTE4(v386[54]) )
      _libc_free(v386[52]);
    v56 = (_QWORD *)v386[1];
    v57 = (_QWORD *)(v386[1] + 24LL * LODWORD(v386[2]));
    if ( (_QWORD *)v386[1] != v57 )
    {
      do
      {
        v58 = *(v57 - 1);
        v57 -= 3;
        if ( v58 != 0 && v58 != -4096 && v58 != -8192 )
          sub_BD60C0(v57);
      }
      while ( v56 != v57 );
      v57 = (_QWORD *)v386[1];
    }
    if ( v57 != &v386[3] )
      _libc_free((unsigned __int64)v57);
  }
  return a1;
}
