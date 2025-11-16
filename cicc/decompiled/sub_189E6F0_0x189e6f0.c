// Function: sub_189E6F0
// Address: 0x189e6f0
//
__int64 __fastcall sub_189E6F0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 *v11; // r12
  __int64 result; // rax
  int v13; // edx
  __int64 v14; // rbx
  unsigned __int64 v15; // rax
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // r13
  unsigned __int64 v19; // rax
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // r12d
  unsigned __int64 v24; // rax
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rax
  int v28; // eax
  bool v29; // cf
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  __int64 *v32; // r12
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // r12
  int v36; // eax
  void **v37; // rbx
  __int64 v38; // rax
  void **v39; // r13
  unsigned int v40; // edx
  unsigned __int64 *v41; // rcx
  void *v42; // rdi
  void *v43; // rax
  int v44; // esi
  int v45; // edx
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rax
  int v49; // ecx
  __int64 v50; // rax
  double v51; // xmm4_8
  double v52; // xmm5_8
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 *v55; // rax
  __int64 v56; // rdx
  int v57; // r9d
  __int64 *v58; // rbx
  __int64 *v59; // r13
  __int64 v60; // rbx
  _QWORD **v61; // rcx
  unsigned int v62; // esi
  int v63; // r13d
  int v64; // edx
  __int64 v65; // rax
  __int64 v66; // r14
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // r9
  unsigned __int64 v69; // r8
  __int64 v70; // r10
  __int64 *v71; // rax
  int v72; // r8d
  __int64 *v73; // r14
  __int64 *v74; // r12
  __int64 v75; // rdi
  __int64 v76; // r15
  unsigned __int64 v77; // rdi
  int v78; // r13d
  unsigned __int64 v79; // r14
  unsigned int v80; // r12d
  unsigned int v81; // edx
  __int64 v82; // rsi
  __int64 v83; // rax
  int v84; // r8d
  int v85; // r9d
  __int64 v86; // rax
  unsigned __int64 v87; // rdi
  __int64 v88; // r14
  unsigned __int64 v89; // rax
  __int64 v90; // r12
  unsigned __int64 v91; // rax
  __int64 v92; // r14
  _QWORD *v93; // rdx
  unsigned __int64 *v94; // r13
  unsigned __int64 v95; // rdi
  _QWORD *v96; // rax
  _QWORD *v97; // rbx
  _QWORD *v98; // r13
  __int64 j; // rax
  unsigned __int64 v100; // rdx
  _QWORD *v101; // rax
  _QWORD *v102; // rax
  double v103; // xmm4_8
  double v104; // xmm5_8
  __int64 v105; // rbx
  __int64 *v106; // r12
  __int64 v107; // r14
  int v108; // r8d
  __int64 v109; // r9
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 *v112; // rbx
  __int64 *v113; // r12
  _QWORD *v114; // rax
  int v115; // r8d
  unsigned __int64 *v116; // r14
  __int64 v117; // r9
  __int64 v118; // rax
  char *v119; // rbx
  __int64 v120; // rax
  char *v121; // r12
  __int64 v122; // rdx
  __int64 v123; // rax
  __int64 v124; // rcx
  __int64 v125; // rax
  __int64 v126; // rax
  unsigned __int64 *v127; // rbx
  __int64 v128; // rbx
  __int64 v129; // rcx
  __int64 v130; // rax
  __int64 v131; // r14
  __int64 v132; // rax
  double v133; // xmm4_8
  double v134; // xmm5_8
  __int64 v135; // r15
  __int64 v136; // r12
  __int64 v137; // r13
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // r8
  __int64 v141; // r9
  double v142; // xmm4_8
  double v143; // xmm5_8
  int v144; // eax
  __int64 v145; // rax
  int v146; // edx
  __int64 v147; // rdx
  __int64 *v148; // rax
  __int64 v149; // rcx
  unsigned __int64 v150; // rsi
  __int64 v151; // rcx
  __int64 v152; // rax
  __int64 v153; // rcx
  __int64 v154; // rcx
  __int64 *v155; // r14
  __int64 v156; // r13
  __int64 v157; // rbx
  __int64 v158; // rax
  __int64 v159; // r8
  __int64 v160; // r9
  unsigned int v161; // edi
  __int64 v162; // rdx
  __int64 v163; // rax
  __int64 v164; // rdx
  __int64 v165; // rsi
  int v166; // eax
  __int64 v167; // rax
  int v168; // edx
  __int64 v169; // rdx
  __int64 *v170; // rax
  __int64 v171; // rdi
  unsigned __int64 v172; // rdx
  __int64 v173; // rdx
  __int64 v174; // rdx
  __int64 v175; // rsi
  unsigned int v176; // edi
  unsigned int v177; // esi
  __int64 v178; // rax
  __int64 v179; // rdx
  __int64 v180; // rax
  __int64 v181; // rdx
  __int64 *v182; // rax
  __int64 v183; // rsi
  __int64 *v184; // rcx
  __int64 v185; // rdx
  __int64 *v186; // rdx
  int v187; // r8d
  int v188; // r9d
  unsigned __int64 *v189; // r12
  unsigned __int64 *v190; // rbx
  _QWORD *v191; // rdi
  unsigned __int64 *v192; // r14
  __int64 *v193; // r12
  __int64 *v194; // rbx
  __int64 v195; // rdi
  unsigned __int64 v196; // rax
  __int64 *v197; // rax
  __int64 *v198; // rbx
  __int64 *v199; // r12
  int v200; // eax
  _BYTE *v201; // r9
  __int64 *v202; // r12
  __int64 *k; // rbx
  __int64 v204; // r14
  _QWORD *v205; // rsi
  __int64 v206; // r8
  _BYTE *v207; // r10
  int v208; // eax
  unsigned __int64 v209; // rax
  int v210; // r8d
  int v211; // r9d
  unsigned __int64 v212; // rdx
  __int64 v213; // rax
  __int64 v214; // r12
  __int64 *v215; // rax
  char *v216; // r12
  char *v217; // r14
  __int64 v218; // rax
  __int64 *v219; // r14
  __int64 *v220; // r13
  __int64 v221; // r15
  __int64 *v222; // rbx
  __int64 *v223; // r12
  __int64 v224; // rdi
  __int64 v225; // rax
  __int64 v226; // rax
  void *v227; // rdi
  unsigned int v228; // eax
  __int64 v229; // rdx
  unsigned __int64 v230; // rdi
  __int64 v231; // rax
  __int64 v232; // rdi
  __int64 v233; // rdi
  unsigned __int64 *v234; // r13
  unsigned __int64 *v235; // r12
  unsigned __int64 v236; // rdi
  unsigned __int64 *v237; // rdx
  unsigned __int64 *v238; // r12
  unsigned __int64 *v239; // r13
  unsigned __int64 v240; // rdi
  unsigned __int64 *v241; // r13
  unsigned __int64 *v242; // r12
  unsigned __int64 v243; // rdi
  _QWORD *v244; // r12
  _QWORD *v245; // r13
  __int64 v246; // r14
  __int64 v247; // rdi
  double v248; // xmm4_8
  double v249; // xmm5_8
  _BOOL8 v250; // rax
  unsigned __int64 v251; // rdi
  _BOOL8 v252; // rax
  __int64 v253; // rt0
  __int64 v254; // rax
  int v255; // r8d
  int v256; // r9d
  __int64 v257; // rax
  __int64 v258; // rax
  unsigned int v259; // esi
  __int64 v260; // rcx
  char *v261; // rdi
  unsigned int v262; // r14d
  __int64 v263; // rdx
  char *v264; // rbx
  char *v265; // r12
  __int64 v266; // rsi
  __int64 *v267; // r12
  __int64 v268; // rax
  __int64 v269; // rax
  __int64 v270; // r8
  __int64 v271; // rax
  __int64 v272; // rbx
  __int64 v273; // rax
  __int64 v274; // rax
  signed __int64 v275; // rdx
  __int64 *v276; // r13
  __int64 v277; // rax
  __int64 v278; // r8
  __int64 v279; // rax
  __int16 v280; // ax
  int v281; // r11d
  unsigned __int64 *v282; // r10
  char *v283; // r13
  char *v284; // rbx
  __int64 v285; // rax
  __int64 v286; // rbx
  __int64 *v287; // r13
  __int64 *i; // r14
  __int64 v289; // rdi
  __int64 v290; // rax
  __int64 v291; // rax
  void *v292; // rdi
  unsigned int v293; // eax
  __int64 v294; // rdx
  __int64 v295; // rdi
  __int64 v296; // rdi
  unsigned __int64 *v297; // r13
  unsigned __int64 *v298; // rbx
  unsigned __int64 v299; // rdi
  unsigned __int64 *v300; // rbx
  unsigned __int64 *v301; // r13
  unsigned __int64 v302; // rdi
  unsigned __int64 *v303; // r13
  unsigned __int64 *v304; // r14
  unsigned __int64 *v305; // rbx
  unsigned __int64 v306; // rdi
  _QWORD *v307; // r14
  _QWORD *v308; // rbx
  __int64 v309; // r13
  __int64 v310; // rdi
  int v311; // eax
  char *v312; // r10
  __int64 v313; // rax
  __int64 v314; // rax
  int v315; // r11d
  __int64 v316; // rax
  __int64 v317; // [rsp+10h] [rbp-9D0h]
  unsigned __int64 v318; // [rsp+18h] [rbp-9C8h]
  __int64 v319; // [rsp+18h] [rbp-9C8h]
  __int64 v322; // [rsp+38h] [rbp-9A8h]
  __int64 v323; // [rsp+48h] [rbp-998h]
  unsigned int v324; // [rsp+58h] [rbp-988h]
  __int64 v325; // [rsp+68h] [rbp-978h]
  __int64 v326; // [rsp+68h] [rbp-978h]
  __int64 v327; // [rsp+70h] [rbp-970h]
  __int64 v328; // [rsp+78h] [rbp-968h]
  _QWORD *v329; // [rsp+80h] [rbp-960h]
  __int64 v330; // [rsp+80h] [rbp-960h]
  __int64 v331; // [rsp+80h] [rbp-960h]
  unsigned __int64 v332; // [rsp+88h] [rbp-958h]
  __int64 v333; // [rsp+88h] [rbp-958h]
  __int64 v334; // [rsp+88h] [rbp-958h]
  __int64 *v335; // [rsp+88h] [rbp-958h]
  __int64 v336; // [rsp+88h] [rbp-958h]
  _QWORD *v337; // [rsp+90h] [rbp-950h]
  __int64 v338; // [rsp+90h] [rbp-950h]
  int v339; // [rsp+90h] [rbp-950h]
  unsigned __int64 v340; // [rsp+98h] [rbp-948h]
  __int64 v341; // [rsp+98h] [rbp-948h]
  int v342; // [rsp+98h] [rbp-948h]
  __int64 *v343; // [rsp+98h] [rbp-948h]
  __int64 v344; // [rsp+98h] [rbp-948h]
  __int64 v345; // [rsp+98h] [rbp-948h]
  __int64 v346; // [rsp+98h] [rbp-948h]
  __int64 *v347; // [rsp+98h] [rbp-948h]
  __int64 v348; // [rsp+98h] [rbp-948h]
  _BOOL8 v349; // [rsp+98h] [rbp-948h]
  _BOOL8 v350; // [rsp+98h] [rbp-948h]
  __int64 v351; // [rsp+98h] [rbp-948h]
  __int64 *v352; // [rsp+98h] [rbp-948h]
  __int64 v353; // [rsp+98h] [rbp-948h]
  __int64 v354; // [rsp+A8h] [rbp-938h] BYREF
  __int64 *v355[2]; // [rsp+B0h] [rbp-930h] BYREF
  __int64 *v356; // [rsp+C0h] [rbp-920h]
  __int64 v357; // [rsp+D0h] [rbp-910h] BYREF
  __int64 v358; // [rsp+D8h] [rbp-908h]
  __int64 v359; // [rsp+E0h] [rbp-900h]
  __int64 v360; // [rsp+E8h] [rbp-8F8h]
  __int64 v361; // [rsp+F0h] [rbp-8F0h]
  __int64 v362; // [rsp+F8h] [rbp-8E8h]
  __int64 v363; // [rsp+100h] [rbp-8E0h]
  __int64 v364; // [rsp+110h] [rbp-8D0h] BYREF
  __int64 v365; // [rsp+118h] [rbp-8C8h]
  __int64 v366; // [rsp+120h] [rbp-8C0h]
  __int64 v367; // [rsp+128h] [rbp-8B8h]
  __int64 v368; // [rsp+130h] [rbp-8B0h]
  __int64 v369; // [rsp+138h] [rbp-8A8h]
  __int64 v370; // [rsp+140h] [rbp-8A0h]
  __int64 v371; // [rsp+150h] [rbp-890h] BYREF
  __int64 v372; // [rsp+158h] [rbp-888h]
  __int64 v373; // [rsp+160h] [rbp-880h]
  __int64 v374; // [rsp+168h] [rbp-878h]
  __int64 v375; // [rsp+170h] [rbp-870h]
  __int64 v376; // [rsp+178h] [rbp-868h]
  __int64 v377; // [rsp+180h] [rbp-860h]
  unsigned __int64 v378[2]; // [rsp+190h] [rbp-850h] BYREF
  char v379; // [rsp+1A0h] [rbp-840h] BYREF
  __int64 v380; // [rsp+1A8h] [rbp-838h]
  _QWORD *v381; // [rsp+1B0h] [rbp-830h]
  __int64 v382; // [rsp+1B8h] [rbp-828h]
  unsigned int v383; // [rsp+1C0h] [rbp-820h]
  __int64 *v384; // [rsp+1D0h] [rbp-810h]
  char v385; // [rsp+1D8h] [rbp-808h]
  int v386; // [rsp+1DCh] [rbp-804h]
  _QWORD *v387; // [rsp+1E0h] [rbp-800h] BYREF
  _BYTE *v388; // [rsp+1E8h] [rbp-7F8h]
  _BYTE *v389; // [rsp+1F0h] [rbp-7F0h] BYREF
  __int64 *v390; // [rsp+200h] [rbp-7E0h]
  __int64 v391; // [rsp+210h] [rbp-7D0h] BYREF
  __int64 *v392; // [rsp+240h] [rbp-7A0h] BYREF
  __int64 v393; // [rsp+248h] [rbp-798h]
  char v394; // [rsp+250h] [rbp-790h] BYREF
  __int64 v395; // [rsp+258h] [rbp-788h]
  _QWORD *v396; // [rsp+260h] [rbp-780h]
  __int64 v397; // [rsp+268h] [rbp-778h]
  unsigned int v398; // [rsp+270h] [rbp-770h]
  __int64 *v399; // [rsp+280h] [rbp-760h]
  char v400; // [rsp+288h] [rbp-758h]
  int v401; // [rsp+28Ch] [rbp-754h]
  __int64 v402; // [rsp+290h] [rbp-750h]
  __int64 v403; // [rsp+298h] [rbp-748h]
  __int64 v404; // [rsp+2A0h] [rbp-740h]
  _BYTE v405[32]; // [rsp+2B0h] [rbp-730h] BYREF
  __int64 *v406; // [rsp+2D0h] [rbp-710h]
  __int64 v407; // [rsp+2E0h] [rbp-700h] BYREF
  __int64 v408; // [rsp+2F8h] [rbp-6E8h]
  __int64 v409; // [rsp+308h] [rbp-6D8h]
  __int64 v410; // [rsp+320h] [rbp-6C0h] BYREF
  __int64 *v411; // [rsp+328h] [rbp-6B8h]
  _BYTE *v412; // [rsp+330h] [rbp-6B0h] BYREF
  __int64 v413; // [rsp+338h] [rbp-6A8h]
  _BYTE v414[68]; // [rsp+340h] [rbp-6A0h] BYREF
  int v415; // [rsp+384h] [rbp-65Ch]
  unsigned __int64 *v416; // [rsp+388h] [rbp-658h]
  __int64 v417; // [rsp+390h] [rbp-650h]
  __int64 *v418; // [rsp+398h] [rbp-648h]
  __int64 *v419; // [rsp+3A0h] [rbp-640h]
  char *v420; // [rsp+3B0h] [rbp-630h] BYREF
  __int64 v421; // [rsp+3B8h] [rbp-628h]
  __int64 *v422; // [rsp+3D0h] [rbp-610h]
  __int64 *v423; // [rsp+3D8h] [rbp-608h]
  __int64 v424; // [rsp+3E0h] [rbp-600h]
  unsigned __int64 v425; // [rsp+3E8h] [rbp-5F8h]
  unsigned __int64 v426; // [rsp+3F0h] [rbp-5F0h]
  unsigned __int64 *v427; // [rsp+3F8h] [rbp-5E8h]
  unsigned int v428; // [rsp+400h] [rbp-5E0h]
  _BYTE v429[32]; // [rsp+408h] [rbp-5D8h] BYREF
  unsigned __int64 *v430; // [rsp+428h] [rbp-5B8h]
  unsigned int v431; // [rsp+430h] [rbp-5B0h]
  _QWORD v432[3]; // [rsp+438h] [rbp-5A8h] BYREF
  char *v433; // [rsp+450h] [rbp-590h] BYREF
  void *src; // [rsp+458h] [rbp-588h] BYREF
  unsigned __int64 v435; // [rsp+460h] [rbp-580h]
  __int64 v436; // [rsp+468h] [rbp-578h]
  __int64 v437; // [rsp+470h] [rbp-570h]
  __int64 v438; // [rsp+478h] [rbp-568h]
  __int64 v439; // [rsp+480h] [rbp-560h]
  __int64 v440; // [rsp+488h] [rbp-558h]
  __int64 v441; // [rsp+498h] [rbp-548h]
  _BYTE *v442; // [rsp+4A0h] [rbp-540h]
  _BYTE *v443; // [rsp+4A8h] [rbp-538h]
  __int64 v444; // [rsp+4B0h] [rbp-530h]
  int v445; // [rsp+4B8h] [rbp-528h]
  _BYTE v446[128]; // [rsp+4C0h] [rbp-520h] BYREF
  __int64 v447; // [rsp+540h] [rbp-4A0h]
  _BYTE *v448; // [rsp+548h] [rbp-498h]
  _BYTE *v449; // [rsp+550h] [rbp-490h]
  __int64 v450; // [rsp+558h] [rbp-488h]
  int v451; // [rsp+560h] [rbp-480h]
  _BYTE v452[136]; // [rsp+568h] [rbp-478h] BYREF
  void *v453; // [rsp+5F0h] [rbp-3F0h] BYREF
  void *v454; // [rsp+5F8h] [rbp-3E8h] BYREF
  unsigned __int64 v455; // [rsp+600h] [rbp-3E0h]
  __m128 v456; // [rsp+608h] [rbp-3D8h]
  __int64 v457; // [rsp+618h] [rbp-3C8h]
  __int64 v458; // [rsp+620h] [rbp-3C0h]
  __m128 v459; // [rsp+628h] [rbp-3B8h]
  __int64 v460; // [rsp+638h] [rbp-3A8h]
  char v461; // [rsp+640h] [rbp-3A0h]
  _BYTE *v462; // [rsp+648h] [rbp-398h] BYREF
  __int64 v463; // [rsp+650h] [rbp-390h]
  _BYTE v464[352]; // [rsp+658h] [rbp-388h] BYREF
  char v465; // [rsp+7B8h] [rbp-228h]
  int v466; // [rsp+7BCh] [rbp-224h]
  __int64 v467; // [rsp+7C0h] [rbp-220h]
  unsigned __int64 *v468; // [rsp+7D0h] [rbp-210h] BYREF
  __int64 v469; // [rsp+7D8h] [rbp-208h] BYREF
  unsigned __int64 v470; // [rsp+7E0h] [rbp-200h] BYREF
  __int64 v471; // [rsp+7E8h] [rbp-1F8h]
  __int64 v472; // [rsp+7F0h] [rbp-1F0h] BYREF
  char *v473; // [rsp+7F8h] [rbp-1E8h]
  unsigned __int64 v474; // [rsp+800h] [rbp-1E0h]
  __int64 v475; // [rsp+808h] [rbp-1D8h]
  char v476; // [rsp+810h] [rbp-1D0h]
  __int64 v477; // [rsp+818h] [rbp-1C8h]
  _QWORD v478[55]; // [rsp+828h] [rbp-1B8h] BYREF

  v10 = a1;
  v11 = a2;
  if ( (unsigned __int8)sub_15E3650((__int64)a2, 0)
    || (unsigned __int8)sub_1560180((__int64)(a2 + 14), 3)
    || (unsigned __int8)sub_1560180((__int64)(a2 + 14), 26)
    || sub_1441DA0(*(_QWORD **)(a1 + 48), (__int64)a2)
    || !a2[1] )
  {
    return 0;
  }
  sub_143A950(v355, a2);
  if ( (unsigned __int8)sub_1441AE0(*(_QWORD **)(a1 + 48)) )
  {
    sub_15E44B0((__int64)a2);
    if ( v13 )
    {
      if ( !byte_4FAD100 )
      {
        sub_1898540(&v354, a1, a2, (__int64 *)v355);
        if ( v354 )
        {
          sub_189DCA0(
            (__int64)&v410,
            (__int64)a2,
            v354,
            (__int64)v355,
            a3,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            v51,
            v52,
            a9,
            a10);
          if ( !*(_DWORD *)(v417 + 8) )
          {
LABEL_409:
            sub_1897560(
              (__int64)&v410,
              a3,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v53,
              v54,
              a9,
              a10);
            if ( v354 )
              sub_1897290(v354);
            goto LABEL_11;
          }
          v380 = 0;
          v378[0] = (unsigned __int64)&v379;
          v378[1] = 0x100000000LL;
          v381 = 0;
          v384 = v411;
          v382 = 0;
          v383 = 0;
          v385 = 0;
          v386 = 0;
          sub_15D3930((__int64)v378);
          sub_14019E0((__int64)&v420, (__int64)v378);
          v442 = v446;
          v443 = v446;
          v448 = v452;
          v449 = v452;
          v433 = 0;
          src = 0;
          v435 = 0;
          v436 = 0;
          v437 = 0;
          v438 = 0;
          v439 = 0;
          v440 = 0;
          v441 = 0;
          v444 = 16;
          v445 = 0;
          v447 = 0;
          v450 = 16;
          v451 = 0;
          sub_137CAE0((__int64)&v433, v411, (__int64)&v420, 0);
          v55 = (__int64 *)sub_22077B0(8);
          v58 = v55;
          if ( v55 )
            sub_13702A0(v55, v411, (__int64)&v433, (__int64)&v420);
          v59 = v418;
          v418 = v58;
          if ( v59 )
          {
            sub_1368A00(v59);
            j_j___libc_free_0(v59, 8);
          }
          v358 = 0;
          v359 = 0;
          v360 = 0;
          v361 = 0;
          v362 = 0;
          v363 = 0;
          v365 = 0;
          v366 = 0;
          v367 = 0;
          v368 = 0;
          v369 = 0;
          v370 = 0;
          v372 = 0;
          v373 = 0;
          v374 = 0;
          v375 = 0;
          v376 = 0;
          v377 = 0;
          v357 = 0;
          v364 = 0;
          v371 = 0;
          v60 = *(_QWORD *)v417;
          v338 = *(_QWORD *)v417 + 104LL * *(unsigned int *)(v417 + 8);
          if ( *(_QWORD *)v417 != v338 )
          {
            v61 = &v387;
            do
            {
              v392 = (__int64 *)&v394;
              v393 = 0x800000000LL;
              v72 = *(_DWORD *)(v60 + 8);
              if ( v72 )
              {
                sub_18971B0((__int64)&v392, v60, v56, (__int64)v61, v72, v57);
                v62 = (unsigned int)v392;
                v64 = v393;
                v402 = *(_QWORD *)(v60 + 80);
                v73 = &v392[(unsigned int)v393];
                v403 = *(_QWORD *)(v60 + 88);
                v404 = *(_QWORD *)(v60 + 96);
                if ( v392 == v73 )
                {
                  v63 = 0;
                }
                else
                {
                  v74 = v392;
                  v63 = 0;
                  do
                  {
                    v75 = *v74++;
                    v63 += sub_1897310(v75);
                  }
                  while ( v73 != v74 );
                  v62 = (unsigned int)v392;
                  v64 = v393;
                }
              }
              else
              {
                v62 = (unsigned int)&v394;
                v63 = 0;
                v64 = v393;
                v402 = *(_QWORD *)(v60 + 80);
                v403 = *(_QWORD *)(v60 + 88);
                v404 = *(_QWORD *)(v60 + 96);
              }
              sub_1AC09B0((unsigned int)v405, v62, v64, (unsigned int)v378, 0, (_DWORD)v418, (__int64)&v433, 0, 0);
              sub_1ABF1D0(v405, &v357, &v364, &v371);
              if ( v369 == v368 || byte_4FAD020 )
              {
                v65 = sub_1AC1F00(v405);
                v66 = v65;
                if ( v65 )
                {
                  v67 = (unsigned __int64)sub_1648700(*(_QWORD *)(v65 + 8));
                  v68 = v67 & 0xFFFFFFFFFFFFFFFBLL;
                  if ( *(_BYTE *)(v67 + 16) == 78 )
                    v68 |= 4u;
                  v69 = v68 & 0xFFFFFFFFFFFFFFF8LL;
                  v70 = *(_QWORD *)((v68 & 0xFFFFFFFFFFFFFFF8LL) + 40);
                  if ( (unsigned int)v413 >= HIDWORD(v413) )
                  {
                    v317 = *(_QWORD *)((v68 & 0xFFFFFFFFFFFFFFF8LL) + 40);
                    v318 = v68 & 0xFFFFFFFFFFFFFFF8LL;
                    sub_16CD150((__int64)&v412, v414, 0, 16, v69, v68);
                    v70 = v317;
                    v69 = v318;
                  }
                  v71 = (__int64 *)&v412[16 * (unsigned int)v413];
                  *v71 = v66;
                  v71[1] = v70;
                  LODWORD(v413) = v413 + 1;
                  v415 += v63;
                  if ( byte_4FACF40 )
                  {
                    v280 = *(_WORD *)(v66 + 18) & 0xC00F;
                    LOBYTE(v280) = v280 | 0x90;
                    *(_WORD *)(v66 + 18) = v280;
                    *(_WORD *)(v69 + 18) = *(_WORD *)(v69 + 18) & 0x8003 | 0x24;
                  }
                }
                else
                {
                  v276 = v419;
                  v277 = sub_15E0530(*v419);
                  if ( sub_1602790(v277)
                    || (v313 = sub_15E0530(*v276),
                        v314 = sub_16033E0(v313),
                        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v314 + 48LL))(v314)) )
                  {
                    v278 = *(_QWORD *)(*v392 + 48);
                    if ( v278 )
                      v278 -= 24;
                    sub_15CA5C0((__int64)&v468, (__int64)"partial-inlining", (__int64)"ExtractFailed", 13, v278);
                    sub_15CAB20((__int64)&v468, "Failed to extract region at block ", 0x22u);
                    sub_15C9340((__int64)&v387, "Block", 5u, *v392);
                    v279 = sub_17C21B0((__int64)&v468, (__int64)&v387);
                    LODWORD(v454) = *(_DWORD *)(v279 + 8);
                    BYTE4(v454) = *(_BYTE *)(v279 + 12);
                    v455 = *(_QWORD *)(v279 + 16);
                    a3 = (__m128)_mm_loadu_si128((const __m128i *)(v279 + 24));
                    v456 = a3;
                    v457 = *(_QWORD *)(v279 + 40);
                    v453 = &unk_49ECF68;
                    v458 = *(_QWORD *)(v279 + 48);
                    a4 = _mm_loadu_si128((const __m128i *)(v279 + 56));
                    v459 = (__m128)a4;
                    v461 = *(_BYTE *)(v279 + 80);
                    if ( v461 )
                      v460 = *(_QWORD *)(v279 + 72);
                    v462 = v464;
                    v463 = 0x400000000LL;
                    if ( *(_DWORD *)(v279 + 96) )
                    {
                      v319 = v279;
                      sub_1897E20((__int64)&v462, v279 + 88);
                      v279 = v319;
                    }
                    v465 = *(_BYTE *)(v279 + 456);
                    v466 = *(_DWORD *)(v279 + 460);
                    v467 = *(_QWORD *)(v279 + 464);
                    v453 = &unk_49ECFC8;
                    if ( v390 != &v391 )
                      j_j___libc_free_0(v390, v391 + 1);
                    if ( v387 != &v389 )
                      j_j___libc_free_0(v387, v389 + 1);
                    v468 = (unsigned __int64 *)&unk_49ECF68;
                    sub_1897B80((__int64)v478);
                    sub_143AA50(v276, (__int64)&v453);
                    v453 = &unk_49ECF68;
                    sub_1897B80((__int64)&v462);
                  }
                }
              }
              if ( v408 )
                j_j___libc_free_0(v408, v409 - v408);
              j___libc_free_0(v407);
              if ( v392 != (__int64 *)&v394 )
                _libc_free((unsigned __int64)v392);
              v60 += 104;
            }
            while ( v338 != v60 );
            v10 = a1;
            v11 = a2;
          }
          v339 = v413;
          if ( v375 )
            j_j___libc_free_0(v375, v377 - v375);
          j___libc_free_0(v372);
          if ( v368 )
            j_j___libc_free_0(v368, v370 - v368);
          j___libc_free_0(v365);
          if ( v361 )
            j_j___libc_free_0(v361, v363 - v361);
          j___libc_free_0(v358);
          if ( v449 != v448 )
            _libc_free((unsigned __int64)v449);
          if ( v443 != v442 )
            _libc_free((unsigned __int64)v443);
          j___libc_free_0(v438);
          if ( (_DWORD)v436 )
          {
            v454 = (void *)2;
            v283 = (char *)src;
            v455 = 0;
            v456 = (__m128)0xFFFFFFFFFFFFFFF8LL;
            v453 = &unk_49E8A80;
            v284 = (char *)src + 40 * (unsigned int)v436;
            v469 = 2;
            v470 = 0;
            v471 = -16;
            v468 = (unsigned __int64 *)&unk_49E8A80;
            v472 = 0;
            while ( v284 != v283 )
            {
              v285 = *((_QWORD *)v283 + 3);
              *(_QWORD *)v283 = &unk_49EE2B0;
              if ( v285 != 0 && v285 != -8 && v285 != -16 )
                sub_1649B30((_QWORD *)v283 + 1);
              v283 += 40;
            }
            v468 = (unsigned __int64 *)&unk_49EE2B0;
            if ( v471 != 0 && v471 != -8 && v471 != -16 )
              sub_1649B30(&v469);
            v453 = &unk_49EE2B0;
            if ( v456.m128_u64[0] != 0 && v456.m128_u64[0] != -8 && v456.m128_u64[0] != -16 )
              sub_1649B30(&v454);
          }
          j___libc_free_0(src);
          sub_142D890((__int64)&v420);
          v335 = v423;
          v352 = v422;
          if ( v422 != v423 )
          {
            do
            {
              v286 = *v352;
              v287 = *(__int64 **)(*v352 + 16);
              for ( i = *(__int64 **)(*v352 + 8); v287 != i; ++i )
              {
                v289 = *i;
                sub_13FACC0(v289);
              }
              *(_BYTE *)(v286 + 160) = 1;
              v290 = *(_QWORD *)(v286 + 8);
              if ( v290 != *(_QWORD *)(v286 + 16) )
                *(_QWORD *)(v286 + 16) = v290;
              v291 = *(_QWORD *)(v286 + 32);
              if ( v291 != *(_QWORD *)(v286 + 40) )
                *(_QWORD *)(v286 + 40) = v291;
              ++*(_QWORD *)(v286 + 56);
              v292 = *(void **)(v286 + 72);
              if ( v292 != *(void **)(v286 + 64) )
              {
                v293 = 4 * (*(_DWORD *)(v286 + 84) - *(_DWORD *)(v286 + 88));
                v294 = *(unsigned int *)(v286 + 80);
                if ( v293 < 0x20 )
                  v293 = 32;
                if ( (unsigned int)v294 > v293 )
                  sub_16CC920(v286 + 56);
                else
                  memset(v292, -1, 8 * v294);
                v292 = *(void **)(v286 + 72);
              }
              *(_QWORD *)v286 = 0;
              if ( v292 != *(void **)(v286 + 64) )
                _libc_free((unsigned __int64)v292);
              v295 = *(_QWORD *)(v286 + 32);
              if ( v295 )
                j_j___libc_free_0(v295, *(_QWORD *)(v286 + 48) - v295);
              v296 = *(_QWORD *)(v286 + 8);
              if ( v296 )
                j_j___libc_free_0(v296, *(_QWORD *)(v286 + 24) - v296);
              ++v352;
            }
            while ( v335 != v352 );
            if ( v422 != v423 )
              v423 = v422;
          }
          v297 = v430;
          v298 = &v430[2 * v431];
          if ( v430 != v298 )
          {
            do
            {
              v299 = *v297;
              v297 += 2;
              _libc_free(v299);
            }
            while ( v298 != v297 );
          }
          v431 = 0;
          if ( v428 )
          {
            v432[0] = 0;
            v300 = &v427[v428];
            v301 = v427 + 1;
            v425 = *v427;
            v426 = v425 + 4096;
            while ( v300 != v301 )
            {
              v302 = *v301++;
              _libc_free(v302);
            }
            v428 = 1;
            _libc_free(*v427);
            v303 = v430;
            v304 = v430;
            v305 = &v430[2 * v431];
            if ( v430 == v305 )
            {
LABEL_485:
              if ( v303 != v432 )
                _libc_free((unsigned __int64)v303);
              if ( v427 != (unsigned __int64 *)v429 )
                _libc_free((unsigned __int64)v427);
              if ( v422 )
                j_j___libc_free_0(v422, v424 - (_QWORD)v422);
              j___libc_free_0(v421);
              if ( v383 )
              {
                v307 = v381;
                v308 = &v381[2 * v383];
                do
                {
                  if ( *v307 != -16 && *v307 != -8 )
                  {
                    v309 = v307[1];
                    if ( v309 )
                    {
                      v310 = *(_QWORD *)(v309 + 24);
                      if ( v310 )
                        j_j___libc_free_0(v310, *(_QWORD *)(v309 + 40) - v310);
                      j_j___libc_free_0(v309, 56);
                    }
                  }
                  v307 += 2;
                }
                while ( v308 != v307 );
              }
              j___libc_free_0(v381);
              if ( (char *)v378[0] != &v379 )
                _libc_free(v378[0]);
              if ( v339 && (_DWORD)v413 && (unsigned __int8)sub_189A480(v10, (__int64)&v410) )
              {
                sub_1897560(
                  (__int64)&v410,
                  a3,
                  *(double *)a4.m128i_i64,
                  *(double *)a5.m128i_i64,
                  *(double *)a6.m128i_i64,
                  v53,
                  v54,
                  a9,
                  a10);
                result = 1;
                if ( v354 )
                {
                  sub_1897290(v354);
                  result = 1;
                }
                goto LABEL_37;
              }
              goto LABEL_409;
            }
            do
            {
              v306 = *v304;
              v304 += 2;
              _libc_free(v306);
            }
            while ( v305 != v304 );
          }
          v303 = v430;
          goto LABEL_485;
        }
      }
    }
  }
LABEL_11:
  v14 = v11[10];
  if ( v14 )
    v14 -= 24;
  v15 = sub_157EBA0(v14);
  if ( *(_BYTE *)(v15 + 16) != 26 || (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) == 1 )
    goto LABEL_36;
  v337 = (_QWORD *)sub_22077B0(112);
  if ( v337 )
  {
    memset(v337, 0, 0x70u);
    v337[8] = v337 + 10;
    *v337 = v337 + 2;
    v337[1] = 0x400000000LL;
    v337[9] = 0x400000000LL;
    v16 = 1;
  }
  else
  {
    v16 = MEMORY[8] + 1;
  }
  if ( v16 >= dword_4FACAE0 )
  {
LABEL_31:
    v31 = v337[8];
    if ( (_QWORD *)v31 != v337 + 10 )
      _libc_free(v31);
    if ( (_QWORD *)*v337 != v337 + 2 )
      _libc_free(*v337);
    j_j___libc_free_0(v337, 112);
LABEL_36:
    result = 0;
    goto LABEL_37;
  }
  v327 = v10;
  v325 = (__int64)v11;
  while ( 1 )
  {
    v30 = sub_157EBA0(v14);
    if ( !v30 || (unsigned int)sub_15F4D60(v30) != 2 )
      goto LABEL_31;
    v17 = sub_157EBA0(v14);
    v18 = sub_15F4DF0(v17, 0);
    v19 = sub_157EBA0(v14);
    v20 = sub_15F4DF0(v19, 1u);
    if ( *(_BYTE *)(sub_157EBA0(v18) + 16) == 25 )
    {
      v33 = v18;
      v34 = v20;
    }
    else
    {
      if ( *(_BYTE *)(sub_157EBA0(v20) + 16) != 25 )
        goto LABEL_22;
      v33 = v20;
      v34 = v18;
    }
    if ( v33 )
      break;
LABEL_22:
    v453 = (void *)v18;
    v21 = sub_157EBA0(v20);
    v22 = v21;
    if ( v21 )
    {
      v340 = v21;
      v23 = sub_15F4D60(v21);
      v24 = sub_157EBA0(v20);
      v22 = v340;
    }
    else
    {
      v24 = 0;
      v23 = 0;
    }
    LODWORD(v471) = v23;
    v468 = (unsigned __int64 *)v24;
    LODWORD(v469) = 0;
    v470 = v22;
    if ( !sub_1897CF0((__int64)&v468, &v453) )
    {
      v453 = (void *)v20;
      v46 = sub_157EBA0(v18);
      v47 = v46;
      if ( v46 )
      {
        v332 = v46;
        v342 = sub_15F4D60(v46);
        v48 = sub_157EBA0(v18);
        v49 = v342;
        v47 = v332;
      }
      else
      {
        v48 = 0;
        v49 = 0;
      }
      v468 = (unsigned __int64 *)v48;
      LODWORD(v469) = 0;
      v470 = v47;
      LODWORD(v471) = v49;
      if ( !sub_1897CF0((__int64)&v468, &v453) )
        goto LABEL_31;
      v50 = v20;
      v20 = v18;
      v18 = v50;
    }
    if ( !v18 )
      goto LABEL_31;
    v27 = *((unsigned int *)v337 + 2);
    if ( (unsigned int)v27 >= *((_DWORD *)v337 + 3) )
    {
      sub_16CD150((__int64)v337, v337 + 2, 0, 8, v25, v26);
      v27 = *((unsigned int *)v337 + 2);
    }
    *(_QWORD *)(*v337 + 8 * v27) = v14;
    v28 = *((_DWORD *)v337 + 2);
    v29 = v28 + 2 < (unsigned int)dword_4FACAE0;
    *((_DWORD *)v337 + 2) = v28 + 1;
    if ( !v29 )
      goto LABEL_31;
    v14 = v20;
  }
  v35 = v325;
  if ( *((_DWORD *)v337 + 2) >= *((_DWORD *)v337 + 3) )
  {
    v336 = v33;
    v353 = v34;
    sub_16CD150((__int64)v337, v337 + 2, 0, 8, v34, v33);
    v33 = v336;
    v34 = v353;
  }
  *(_QWORD *)(*v337 + 8LL * *((unsigned int *)v337 + 2)) = v14;
  v36 = *((_DWORD *)v337 + 2);
  v37 = (void **)*v337;
  v337[6] = v33;
  v38 = (unsigned int)(v36 + 1);
  v39 = &v37[v38];
  *((_DWORD *)v337 + 2) = v38;
  v337[7] = v34;
  v433 = 0;
  src = 0;
  v435 = 0;
  v436 = 0;
  if ( v37 != v39 )
  {
    while ( 1 )
    {
      v43 = *v37;
      v44 = v436;
      v453 = *v37;
      if ( !(_DWORD)v436 )
        break;
      v40 = (v436 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v41 = (unsigned __int64 *)((char *)src + 8 * v40);
      v42 = (void *)*v41;
      if ( v43 != (void *)*v41 )
      {
        v281 = 1;
        v282 = 0;
        while ( v42 != (void *)-8LL )
        {
          if ( v42 == (void *)-16LL && !v282 )
            v282 = v41;
          v40 = (v436 - 1) & (v281 + v40);
          v41 = (unsigned __int64 *)((char *)src + 8 * v40);
          v42 = (void *)*v41;
          if ( v43 == (void *)*v41 )
            goto LABEL_47;
          ++v281;
        }
        if ( v282 )
          v41 = v282;
        ++v433;
        v45 = v435 + 1;
        if ( 4 * ((int)v435 + 1) < (unsigned int)(3 * v436) )
        {
          if ( (int)v436 - HIDWORD(v435) - v45 <= (unsigned int)v436 >> 3 )
          {
LABEL_51:
            sub_13B3D40((__int64)&v433, v44);
            sub_1898220((__int64)&v433, (__int64 *)&v453, &v468);
            v41 = v468;
            v43 = v453;
            v45 = v435 + 1;
          }
          LODWORD(v435) = v45;
          if ( *v41 != -8 )
            --HIDWORD(v435);
          *v41 = (unsigned __int64)v43;
          goto LABEL_47;
        }
LABEL_50:
        v44 = 2 * v436;
        goto LABEL_51;
      }
LABEL_47:
      if ( v39 == ++v37 )
        goto LABEL_89;
    }
    ++v433;
    goto LABEL_50;
  }
LABEL_89:
  v453 = 0;
  v454 = 0;
  v455 = 0;
  v456.m128_u64[0] = 0;
  j___libc_free_0(0);
  v456.m128_i32[0] = v436;
  if ( (_DWORD)v436 )
  {
    v454 = (void *)sub_22077B0(8LL * (unsigned int)v436);
    v455 = v435;
    memcpy(v454, src, 8LL * v456.m128_u32[0]);
  }
  else
  {
    v454 = 0;
    v455 = 0;
  }
  v468 = 0;
  v469 = 0;
  v470 = 0;
  v471 = 0;
  j___libc_free_0(0);
  LODWORD(v471) = v436;
  if ( (_DWORD)v436 )
  {
    v469 = sub_22077B0(8LL * (unsigned int)v436);
    v470 = v435;
    memcpy((void *)v469, src, 8LL * (unsigned int)v471);
  }
  else
  {
    v469 = 0;
    v470 = 0;
  }
  v472 = 0;
  v473 = 0;
  v474 = 0;
  v475 = 0;
  j___libc_free_0(0);
  LODWORD(v475) = v456.m128_i32[0];
  if ( v456.m128_i32[0] )
  {
    v473 = (char *)sub_22077B0(8LL * v456.m128_u32[0]);
    v474 = v455;
    memcpy(v473, v454, 8LL * (unsigned int)v475);
  }
  else
  {
    v473 = 0;
    v474 = 0;
  }
  v343 = (__int64 *)*v337;
  v333 = *v337 + 8LL * *((unsigned int *)v337 + 2);
  if ( *v337 != v333 )
  {
    while ( 1 )
    {
      v76 = *v343;
      v77 = sub_157EBA0(*v343);
      if ( v77 )
      {
        v78 = sub_15F4D60(v77);
        v79 = sub_157EBA0(v76);
        if ( v78 )
          break;
      }
LABEL_108:
      if ( (unsigned __int8)sub_1897480((__int64)&v472, v76) )
      {
LABEL_104:
        j___libc_free_0(v473);
        j___libc_free_0(v469);
        j___libc_free_0(v454);
        j___libc_free_0(src);
        goto LABEL_31;
      }
      if ( (__int64 *)v333 == ++v343 )
      {
        v10 = v327;
        v35 = v325;
        goto LABEL_111;
      }
    }
    v80 = 0;
    while ( 2 )
    {
      v83 = sub_15F4DF0(v79, v80);
      if ( (_DWORD)v471 )
      {
        v81 = (v471 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
        v82 = *(_QWORD *)(v469 + 8LL * v81);
        if ( v83 != v82 )
        {
          v84 = 1;
          while ( v82 != -8 )
          {
            v85 = v84 + 1;
            v81 = (v471 - 1) & (v84 + v81);
            v82 = *(_QWORD *)(v469 + 8LL * v81);
            if ( v83 == v82 )
              goto LABEL_100;
            ++v84;
          }
          goto LABEL_102;
        }
      }
      else
      {
LABEL_102:
        if ( v83 == v337[6] )
        {
          v86 = *((unsigned int *)v337 + 18);
          if ( (unsigned int)v86 >= *((_DWORD *)v337 + 19) )
          {
            sub_16CD150((__int64)(v337 + 8), v337 + 10, 0, 8, v84, v85);
            v86 = *((unsigned int *)v337 + 18);
          }
          *(_QWORD *)(v337[8] + 8 * v86) = v76;
          ++*((_DWORD *)v337 + 18);
        }
        else if ( v83 != v337[7] )
        {
          goto LABEL_104;
        }
      }
LABEL_100:
      if ( v78 == ++v80 )
        goto LABEL_108;
      continue;
    }
  }
LABEL_111:
  if ( *((_DWORD *)v337 + 2) + 1 >= (unsigned int)dword_4FACAE0 )
    goto LABEL_119;
  v344 = v35;
  while ( 2 )
  {
    v410 = v337[7];
    v87 = sub_157EBA0(v410);
    if ( v87 )
    {
      if ( (unsigned int)sub_15F4D60(v87) == 2 )
      {
        v88 = v410;
        if ( !(unsigned __int8)sub_1897480((__int64)&v453, v410) )
        {
          v89 = sub_157EBA0(v88);
          v90 = sub_15F4DF0(v89, 0);
          v91 = sub_157EBA0(v410);
          v92 = sub_15F4DF0(v91, 1u);
          if ( *(_BYTE *)(sub_157EBA0(v90) + 16) != 25 )
          {
            if ( *(_BYTE *)(sub_157EBA0(v92) + 16) != 25 )
              break;
            v253 = v90;
            v90 = v92;
            v92 = v253;
          }
          if ( v90 )
          {
            if ( v337[6] == v90 )
            {
              v254 = sub_157F0B0(v92);
              if ( v410 == v254 )
              {
                v257 = *((unsigned int *)v337 + 2);
                if ( (unsigned int)v257 >= *((_DWORD *)v337 + 3) )
                {
                  sub_16CD150((__int64)v337, v337 + 2, 0, 8, v255, v256);
                  v257 = *((unsigned int *)v337 + 2);
                }
                *(_QWORD *)(*v337 + 8 * v257) = v410;
                v258 = *((unsigned int *)v337 + 18);
                ++*((_DWORD *)v337 + 2);
                v337[7] = v92;
                if ( (unsigned int)v258 >= *((_DWORD *)v337 + 19) )
                {
                  sub_16CD150((__int64)(v337 + 8), v337 + 10, 0, 8, v255, v256);
                  v258 = *((unsigned int *)v337 + 18);
                }
                *(_QWORD *)(v337[8] + 8 * v258) = v410;
                v259 = v436;
                ++*((_DWORD *)v337 + 18);
                if ( v259 )
                {
                  v260 = v410;
                  v261 = (char *)src + 8 * ((v259 - 1) & (((unsigned int)v410 >> 9) ^ ((unsigned int)v410 >> 4)));
                  v262 = (v259 - 1) & (((unsigned int)v410 >> 9) ^ ((unsigned int)v410 >> 4));
                  v263 = *(_QWORD *)v261;
                  if ( *(_QWORD *)v261 == v410 )
                    goto LABEL_367;
                  v311 = 1;
                  v312 = 0;
                  while ( v263 != -8 )
                  {
                    if ( !v312 && v263 == -16 )
                      v312 = v261;
                    v315 = v311 + 1;
                    v316 = (v259 - 1) & (v262 + v311);
                    v261 = (char *)src + 8 * v316;
                    v262 = v316;
                    v263 = *(_QWORD *)v261;
                    if ( v410 == *(_QWORD *)v261 )
                      goto LABEL_367;
                    v311 = v315;
                  }
                  if ( v312 )
                    v261 = v312;
                  ++v433;
                  if ( 4 * ((int)v435 + 1) < 3 * v259 )
                  {
                    if ( v259 - HIDWORD(v435) - ((_DWORD)v435 + 1) > v259 >> 3 )
                    {
LABEL_514:
                      LODWORD(v435) = v435 + 1;
                      if ( *(_QWORD *)v261 != -8 )
                        --HIDWORD(v435);
                      *(_QWORD *)v261 = v260;
LABEL_367:
                      if ( *((_DWORD *)v337 + 2) + 1 >= (unsigned int)dword_4FACAE0 )
                        break;
                      continue;
                    }
LABEL_519:
                    sub_13B3D40((__int64)&v433, v259);
                    sub_1898220((__int64)&v433, &v410, &v420);
                    v261 = v420;
                    v260 = v410;
                    goto LABEL_514;
                  }
                }
                else
                {
                  ++v433;
                }
                v259 *= 2;
                goto LABEL_519;
              }
            }
          }
        }
      }
    }
    break;
  }
  v35 = v344;
LABEL_119:
  j___libc_free_0(v473);
  j___libc_free_0(v469);
  j___libc_free_0(v454);
  j___libc_free_0(src);
  v410 = v35;
  v412 = v414;
  v411 = 0;
  v413 = 0x400000000LL;
  v414[64] = 0;
  v415 = 0;
  v416 = 0;
  v417 = 0;
  v418 = 0;
  v419 = (__int64 *)v355;
  v93 = (_QWORD *)sub_22077B0(112);
  if ( v93 )
  {
    memset(v93, 0, 0x70u);
    v93[1] = 0x400000000LL;
    *v93 = v93 + 2;
    v93[8] = v93 + 10;
    v93[9] = 0x400000000LL;
  }
  v94 = v416;
  v416 = v93;
  if ( v94 )
  {
    v95 = v94[8];
    if ( (unsigned __int64 *)v95 != v94 + 10 )
      _libc_free(v95);
    if ( (unsigned __int64 *)*v94 != v94 + 2 )
      _libc_free(*v94);
    j_j___libc_free_0(v94, 112);
  }
  v468 = 0;
  LODWORD(v471) = 128;
  v96 = (_QWORD *)sub_22077B0(0x2000);
  v470 = 0;
  v97 = v96;
  v469 = (__int64)v96;
  v454 = (void *)2;
  v455 = 0;
  v456 = (__m128)0xFFFFFFFFFFFFFFF8LL;
  v453 = &unk_49E6B50;
  v98 = &v96[8 * (unsigned __int64)(unsigned int)v471];
  for ( j = -8; v98 != v97; v97 += 8 )
  {
    if ( v97 )
    {
      v100 = (unsigned __int64)v454;
      v97[2] = 0;
      v97[3] = j;
      v97[1] = v100 & 6;
      if ( j != 0 && j != -8 && j != -16 )
      {
        sub_1649AC0(v97 + 1, v100 & 0xFFFFFFFFFFFFFFF8LL);
        j = v456.m128_u64[0];
      }
      *v97 = &unk_49E6B50;
      v97[4] = v456.m128_u64[1];
    }
  }
  v453 = &unk_49EE2B0;
  if ( j != 0 && j != -8 && j != -16 )
    sub_1649B30(&v454);
  v476 = 0;
  BYTE1(v477) = 1;
  v411 = (__int64 *)sub_1AB6FF0(v35, &v468, 0);
  v101 = sub_189DAB0((__int64)&v468, v337[6]);
  v416[6] = v101[2];
  v102 = sub_189DAB0((__int64)&v468, v337[7]);
  v416[7] = v102[2];
  if ( *v337 != *v337 + 8LL * *((unsigned int *)v337 + 2) )
  {
    v345 = v35;
    v105 = *v337 + 8LL * *((unsigned int *)v337 + 2);
    v106 = (__int64 *)*v337;
    do
    {
      v107 = (__int64)v416;
      v109 = sub_189DAB0((__int64)&v468, *v106)[2];
      v110 = *(unsigned int *)(v107 + 8);
      if ( (unsigned int)v110 >= *(_DWORD *)(v107 + 12) )
      {
        v331 = v109;
        sub_16CD150(v107, (const void *)(v107 + 16), 0, 8, v108, v109);
        v110 = *(unsigned int *)(v107 + 8);
        v109 = v331;
      }
      ++v106;
      *(_QWORD *)(*(_QWORD *)v107 + 8 * v110) = v109;
      ++*(_DWORD *)(v107 + 8);
    }
    while ( (__int64 *)v105 != v106 );
    v35 = v345;
  }
  v111 = v337[8];
  if ( v111 != v111 + 8LL * *((unsigned int *)v337 + 18) )
  {
    v346 = v35;
    v112 = (__int64 *)(v111 + 8LL * *((unsigned int *)v337 + 18));
    v113 = (__int64 *)v337[8];
    do
    {
      v114 = sub_189DAB0((__int64)&v468, *v113);
      v116 = v416;
      v117 = v114[2];
      v118 = *((unsigned int *)v416 + 18);
      if ( (unsigned int)v118 >= *((_DWORD *)v416 + 19) )
      {
        v330 = v117;
        sub_16CD150((__int64)(v416 + 8), v416 + 10, 0, 8, v115, v117);
        v118 = *((unsigned int *)v116 + 18);
        v117 = v330;
      }
      ++v113;
      *(_QWORD *)(v116[8] + 8 * v118) = v117;
      ++*((_DWORD *)v116 + 18);
    }
    while ( v112 != v113 );
    v35 = v346;
  }
  sub_164D160(
    v35,
    (__int64)v411,
    a3,
    *(double *)a4.m128i_i64,
    *(double *)a5.m128i_i64,
    *(double *)a6.m128i_i64,
    v103,
    v104,
    a9,
    a10);
  if ( v476 )
  {
    if ( (_DWORD)v475 )
    {
      v264 = v473;
      v265 = &v473[16 * (unsigned int)v475];
      do
      {
        if ( *(_QWORD *)v264 != -4 && *(_QWORD *)v264 != -8 )
        {
          v266 = *((_QWORD *)v264 + 1);
          if ( v266 )
            sub_161E7C0((__int64)(v264 + 8), v266);
        }
        v264 += 16;
      }
      while ( v265 != v264 );
    }
    j___libc_free_0(v473);
  }
  if ( (_DWORD)v471 )
  {
    v119 = (char *)v469;
    src = (void *)2;
    v435 = 0;
    v120 = -8;
    v121 = (char *)(v469 + ((unsigned __int64)(unsigned int)v471 << 6));
    v436 = -8;
    v433 = (char *)&unk_49E6B50;
    v437 = 0;
    v454 = (void *)2;
    v455 = 0;
    v456 = (__m128)0xFFFFFFFFFFFFFFF0LL;
    v453 = &unk_49E6B50;
    while ( 1 )
    {
      v122 = *((_QWORD *)v119 + 3);
      if ( v122 != v120 )
      {
        v120 = v456.m128_u64[0];
        if ( v122 != v456.m128_u64[0] )
        {
          v123 = *((_QWORD *)v119 + 7);
          if ( v123 != 0 && v123 != -8 && v123 != -16 )
          {
            sub_1649B30((_QWORD *)v119 + 5);
            v122 = *((_QWORD *)v119 + 3);
          }
          v120 = v122;
        }
      }
      *(_QWORD *)v119 = &unk_49EE2B0;
      if ( v120 != -8 && v120 != 0 && v120 != -16 )
        sub_1649B30((_QWORD *)v119 + 1);
      v119 += 64;
      if ( v121 == v119 )
        break;
      v120 = v436;
    }
    v453 = &unk_49EE2B0;
    if ( v456.m128_u64[0] != 0 && v456.m128_u64[0] != -8 && v456.m128_u64[0] != -16 )
      sub_1649B30(&v454);
    v433 = (char *)&unk_49EE2B0;
    if ( v436 != 0 && v436 != -8 && v436 != -16 )
      sub_1649B30(&src);
  }
  j___libc_free_0(v469);
  if ( !v416 )
    goto LABEL_248;
  v124 = v416[6];
  v125 = *(_QWORD *)(v124 + 48);
  v329 = (_QWORD *)v124;
  v328 = v124 + 40;
  if ( v125 == v124 + 40 )
    goto LABEL_248;
  if ( !v125 )
    BUG();
  if ( *(_BYTE *)(v125 - 8) != 77 )
    goto LABEL_248;
  v324 = *((_DWORD *)v416 + 18) + 1;
  if ( (*(_DWORD *)(v125 - 4) & 0xFFFFFFFu) <= v324 )
    goto LABEL_248;
  LOWORD(v470) = 257;
  v126 = sub_157ED20(v124);
  v127 = v416;
  v127[6] = sub_157FBF0(v329, (__int64 *)(v126 + 24), (__int64)&v468);
  v128 = v329[6];
  v129 = *(_QWORD *)(v416[6] + 48);
  v130 = v129 - 24;
  if ( !v129 )
    v130 = 0;
  v323 = v130;
  v468 = &v470;
  v469 = 0x400000000LL;
  if ( v328 == v128 )
  {
    v193 = (__int64 *)v416[8];
    v192 = v416;
    v194 = &v193[*((unsigned int *)v416 + 18)];
    if ( v193 != v194 )
      goto LABEL_245;
    goto LABEL_248;
  }
  v322 = v10;
  while ( 2 )
  {
    if ( !v128 )
      BUG();
    if ( *(_BYTE *)(v128 - 8) == 77 )
    {
      LOWORD(v455) = 257;
      v131 = *(_QWORD *)(v128 - 24);
      v132 = sub_1648B60(64);
      v135 = v132;
      if ( v132 )
      {
        v136 = v132;
        sub_15F1EA0(v132, v131, 53, 0, 0, v323);
        *(_DWORD *)(v135 + 56) = v324;
        sub_164B780(v135, (__int64 *)&v453);
        sub_1648880(v135, *(_DWORD *)(v135 + 56), 1);
      }
      else
      {
        v136 = 0;
      }
      v137 = v128 - 24;
      sub_164D160(
        v128 - 24,
        v135,
        a3,
        *(double *)a4.m128i_i64,
        *(double *)a5.m128i_i64,
        *(double *)a6.m128i_i64,
        v133,
        v134,
        a9,
        a10);
      v323 = sub_157ED20(v416[6]);
      v144 = *(_DWORD *)(v135 + 20) & 0xFFFFFFF;
      if ( v144 == *(_DWORD *)(v135 + 56) )
      {
        sub_15F55D0(v135, v135, v138, v139, v140, v141);
        v144 = *(_DWORD *)(v135 + 20) & 0xFFFFFFF;
      }
      v145 = (v144 + 1) & 0xFFFFFFF;
      v146 = v145 | *(_DWORD *)(v135 + 20) & 0xF0000000;
      *(_DWORD *)(v135 + 20) = v146;
      if ( (v146 & 0x40000000) != 0 )
        v147 = *(_QWORD *)(v135 - 8);
      else
        v147 = v136 - 24 * v145;
      v148 = (__int64 *)(v147 + 24LL * (unsigned int)(v145 - 1));
      if ( *v148 )
      {
        v149 = v148[1];
        v150 = v148[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v150 = v149;
        if ( v149 )
          *(_QWORD *)(v149 + 16) = v150 | *(_QWORD *)(v149 + 16) & 3LL;
      }
      *v148 = v137;
      v151 = *(_QWORD *)(v128 - 16);
      v148[1] = v151;
      if ( v151 )
        *(_QWORD *)(v151 + 16) = (unsigned __int64)(v148 + 1) | *(_QWORD *)(v151 + 16) & 3LL;
      v148[2] = v148[2] & 3 | (v128 - 16);
      *(_QWORD *)(v128 - 16) = v148;
      v152 = *(_DWORD *)(v135 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v135 + 23) & 0x40) != 0 )
        v153 = *(_QWORD *)(v135 - 8);
      else
        v153 = v136 - 24 * v152;
      *(_QWORD *)(v153 + 8LL * (unsigned int)(v152 - 1) + 24LL * *(unsigned int *)(v135 + 56) + 8) = v329;
      v154 = v416[8];
      v155 = (__int64 *)v154;
      v347 = (__int64 *)(v154 + 8LL * *((unsigned int *)v416 + 18));
      if ( (__int64 *)v154 != v347 )
      {
        v156 = v128;
        v157 = v128 - 24;
        do
        {
          v158 = 0x17FFFFFFE8LL;
          v159 = *v155;
          v160 = *(_BYTE *)(v156 - 1) & 0x40;
          v161 = *(_DWORD *)(v156 - 4) & 0xFFFFFFF;
          if ( v161 )
          {
            v162 = 24LL * *(unsigned int *)(v156 + 32) + 8;
            v163 = 0;
            do
            {
              v154 = v157 - 24LL * v161;
              if ( (_BYTE)v160 )
                v154 = *(_QWORD *)(v156 - 32);
              if ( v159 == *(_QWORD *)(v154 + v162) )
              {
                v158 = 24 * v163;
                goto LABEL_202;
              }
              ++v163;
              v162 += 8;
            }
            while ( v161 != (_DWORD)v163 );
            v158 = 0x17FFFFFFE8LL;
          }
LABEL_202:
          if ( (_BYTE)v160 )
            v164 = *(_QWORD *)(v156 - 32);
          else
            v164 = v157 - 24LL * v161;
          v165 = *(_QWORD *)(v164 + v158);
          v166 = *(_DWORD *)(v135 + 20) & 0xFFFFFFF;
          if ( v166 == *(_DWORD *)(v135 + 56) )
          {
            v326 = *v155;
            sub_15F55D0(v135, v165, v164, v154, v159, v160);
            v159 = v326;
            v166 = *(_DWORD *)(v135 + 20) & 0xFFFFFFF;
          }
          v167 = (v166 + 1) & 0xFFFFFFF;
          v168 = v167 | *(_DWORD *)(v135 + 20) & 0xF0000000;
          *(_DWORD *)(v135 + 20) = v168;
          if ( (v168 & 0x40000000) != 0 )
            v169 = *(_QWORD *)(v135 - 8);
          else
            v169 = v136 - 24 * v167;
          v170 = (__int64 *)(v169 + 24LL * (unsigned int)(v167 - 1));
          if ( *v170 )
          {
            v171 = v170[1];
            v172 = v170[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v172 = v171;
            if ( v171 )
              *(_QWORD *)(v171 + 16) = *(_QWORD *)(v171 + 16) & 3LL | v172;
          }
          *v170 = v165;
          if ( v165 )
          {
            v173 = *(_QWORD *)(v165 + 8);
            v170[1] = v173;
            if ( v173 )
              *(_QWORD *)(v173 + 16) = (unsigned __int64)(v170 + 1) | *(_QWORD *)(v173 + 16) & 3LL;
            v170[2] = (v165 + 8) | v170[2] & 3;
            *(_QWORD *)(v165 + 8) = v170;
          }
          v174 = *(_DWORD *)(v135 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v135 + 23) & 0x40) != 0 )
            v175 = *(_QWORD *)(v135 - 8);
          else
            v175 = v136 - 24 * v174;
          *(_QWORD *)(v175 + 8LL * (unsigned int)(v174 - 1) + 24LL * *(unsigned int *)(v135 + 56) + 8) = v159;
          v176 = *(_DWORD *)(v156 - 4) & 0xFFFFFFF;
          if ( v176 )
          {
            v177 = 0;
            v178 = 24LL * *(unsigned int *)(v156 + 32) + 8;
            while ( 1 )
            {
              v179 = v157 - 24LL * v176;
              if ( (*(_BYTE *)(v156 - 1) & 0x40) != 0 )
                v179 = *(_QWORD *)(v156 - 32);
              if ( v159 == *(_QWORD *)(v179 + v178) )
                break;
              ++v177;
              v178 += 8;
              if ( v176 == v177 )
                goto LABEL_265;
            }
          }
          else
          {
LABEL_265:
            v177 = -1;
          }
          ++v155;
          sub_15F5350(v157, v177, 1);
        }
        while ( v347 != v155 );
        v180 = v157;
        v128 = v156;
        v137 = v180;
      }
      v181 = 24LL * (*(_DWORD *)(v128 - 4) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v128 - 1) & 0x40) != 0 )
      {
        v182 = *(__int64 **)(v128 - 32);
        v183 = *v182;
      }
      else
      {
        v182 = (__int64 *)(v137 - v181);
        v183 = *(_QWORD *)(v137 - 24LL * (*(_DWORD *)(v128 - 4) & 0xFFFFFFF));
      }
      v184 = &v182[(unsigned __int64)v181 / 8];
      v185 = (__int64)(0xAAAAAAAAAAAAAAABLL * (v181 >> 3)) >> 2;
      if ( v185 )
      {
        v186 = &v182[12 * v185];
        while ( v183 == *v182 )
        {
          if ( v183 != v182[3] )
          {
            v182 += 3;
            break;
          }
          if ( v183 != v182[6] )
          {
            v182 += 6;
            break;
          }
          if ( v183 != v182[9] )
          {
            v182 += 9;
            break;
          }
          v182 += 12;
          if ( v186 == v182 )
            goto LABEL_404;
        }
LABEL_234:
        if ( v184 == v182 )
        {
LABEL_235:
          if ( v183 )
          {
            sub_164D160(
              v137,
              v183,
              a3,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v142,
              v143,
              a9,
              a10);
            if ( (unsigned int)v469 >= HIDWORD(v469) )
              sub_16CD150((__int64)&v468, &v470, 0, 8, v187, v188);
            v468[(unsigned int)v469] = v137;
            LODWORD(v469) = v469 + 1;
          }
        }
        v128 = *(_QWORD *)(v128 + 8);
        if ( v328 == v128 )
          break;
        continue;
      }
LABEL_404:
      v275 = (char *)v184 - (char *)v182;
      if ( (char *)v184 - (char *)v182 != 48 )
      {
        if ( v275 != 72 )
        {
          if ( v275 != 24 )
            goto LABEL_235;
          goto LABEL_407;
        }
        if ( v183 != *v182 )
          goto LABEL_234;
        v182 += 3;
      }
      if ( v183 != *v182 )
        goto LABEL_234;
      v182 += 3;
LABEL_407:
      if ( v183 != *v182 )
        goto LABEL_234;
      goto LABEL_235;
    }
    break;
  }
  v189 = v468;
  v10 = v322;
  v190 = &v468[(unsigned int)v469];
  if ( v468 != v190 )
  {
    do
    {
      v191 = (_QWORD *)*v189++;
      sub_15F20C0(v191);
    }
    while ( v190 != v189 );
  }
  v192 = v416;
  v193 = (__int64 *)v416[8];
  v194 = &v193[*((unsigned int *)v416 + 18)];
  if ( v193 != v194 )
  {
LABEL_245:
    while ( 1 )
    {
      v195 = *v193++;
      v196 = sub_157EBA0(v195);
      sub_1648780(v196, (__int64)v329, v192[6]);
      if ( v194 == v193 )
        break;
      v192 = v416;
    }
  }
  if ( v468 != &v470 )
    _libc_free((unsigned __int64)v468);
LABEL_248:
  v395 = 0;
  v392 = (__int64 *)&v394;
  v393 = 0x100000000LL;
  v399 = v411;
  v396 = 0;
  v397 = 0;
  v398 = 0;
  v400 = 0;
  v401 = 0;
  sub_15D3930((__int64)&v392);
  sub_14019E0((__int64)&v420, (__int64)&v392);
  v442 = v446;
  v443 = v446;
  v448 = v452;
  v449 = v452;
  v433 = 0;
  src = 0;
  v435 = 0;
  v436 = 0;
  v437 = 0;
  v438 = 0;
  v439 = 0;
  v440 = 0;
  v441 = 0;
  v444 = 16;
  v445 = 0;
  v447 = 0;
  v450 = 16;
  v451 = 0;
  sub_137CAE0((__int64)&v433, v411, (__int64)&v420, 0);
  v197 = (__int64 *)sub_22077B0(8);
  v198 = v197;
  if ( v197 )
    sub_13702A0(v197, v411, (__int64)&v433, (__int64)&v420);
  v199 = v418;
  v418 = v198;
  if ( v199 )
  {
    sub_1368A00(v199);
    j_j___libc_free_0(v199, 8);
  }
  v387 = 0;
  v388 = 0;
  v389 = 0;
  sub_1292090((__int64)&v387, 0, v416 + 7);
  v200 = sub_1897310(v416[7]);
  v415 += v200;
  v201 = v388;
  v202 = (__int64 *)v411[10];
  for ( k = v411 + 9; k != v202; v202 = (__int64 *)v202[1] )
  {
    v204 = (__int64)(v202 - 3);
    if ( !v202 )
      v204 = 0;
    v468 = (unsigned __int64 *)v204;
    if ( v204 != v416[6] )
    {
      v205 = (_QWORD *)(*v416 + 8LL * *((unsigned int *)v416 + 2));
      if ( v205 == sub_18970F0((_QWORD *)*v416, (__int64)v205, (__int64 *)&v468) && v204 != *(_QWORD *)(v206 + 56) )
      {
        if ( v389 == v201 )
        {
          sub_15D0700((__int64)&v387, v201, &v468);
        }
        else
        {
          if ( v201 )
          {
            *(_QWORD *)v201 = v204;
            v207 = v388;
          }
          v388 = v207 + 8;
        }
        v208 = sub_1897310(v204);
        v201 = v388;
        v415 += v208;
      }
    }
  }
  sub_1AC09B0(
    (unsigned int)&v468,
    (_DWORD)v387,
    (v201 - (_BYTE *)v387) >> 3,
    (unsigned int)&v392,
    0,
    (_DWORD)v418,
    (__int64)&v433,
    1,
    0);
  v348 = sub_1AC1F00(&v468);
  if ( v477 )
    j_j___libc_free_0(v477, v478[0] - v477);
  j___libc_free_0(v474);
  if ( v348 )
  {
    v209 = (unsigned __int64)sub_1648700(*(_QWORD *)(v348 + 8));
    v212 = v209 & 0xFFFFFFFFFFFFFFFBLL;
    if ( *(_BYTE *)(v209 + 16) == 78 )
      v212 |= 4u;
    v213 = (unsigned int)v413;
    v214 = *(_QWORD *)((v212 & 0xFFFFFFFFFFFFFFF8LL) + 40);
    if ( (unsigned int)v413 >= HIDWORD(v413) )
    {
      sub_16CD150((__int64)&v412, v414, 0, 16, v210, v211);
      v213 = (unsigned int)v413;
    }
    v215 = (__int64 *)&v412[16 * v213];
    v215[1] = v214;
    *v215 = v348;
    LODWORD(v413) = v413 + 1;
  }
  else
  {
    v267 = v419;
    v268 = sub_15E0530(*v419);
    if ( sub_1602790(v268)
      || (v273 = sub_15E0530(*v267),
          v274 = sub_16033E0(v273),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v274 + 48LL))(v274)) )
    {
      v269 = *(_QWORD *)(*v387 + 48LL);
      v270 = v269 - 24;
      if ( !v269 )
        v270 = 0;
      sub_15CA5C0((__int64)&v468, (__int64)"partial-inlining", (__int64)"ExtractFailed", 13, v270);
      sub_15CAB20((__int64)&v468, "Failed to extract region at block ", 0x22u);
      sub_15C9340((__int64)v405, "Block", 5u, *v387);
      v271 = sub_17C21B0((__int64)&v468, (__int64)v405);
      v272 = v271;
      LODWORD(v454) = *(_DWORD *)(v271 + 8);
      BYTE4(v454) = *(_BYTE *)(v271 + 12);
      v455 = *(_QWORD *)(v271 + 16);
      a5 = _mm_loadu_si128((const __m128i *)(v271 + 24));
      v456 = (__m128)a5;
      v457 = *(_QWORD *)(v271 + 40);
      v453 = &unk_49ECF68;
      v458 = *(_QWORD *)(v271 + 48);
      a6 = _mm_loadu_si128((const __m128i *)(v271 + 56));
      v459 = (__m128)a6;
      v461 = *(_BYTE *)(v271 + 80);
      if ( v461 )
        v460 = *(_QWORD *)(v271 + 72);
      v462 = v464;
      v463 = 0x400000000LL;
      if ( *(_DWORD *)(v271 + 96) )
        sub_1897E20((__int64)&v462, v271 + 88);
      v465 = *(_BYTE *)(v272 + 456);
      v466 = *(_DWORD *)(v272 + 460);
      v467 = *(_QWORD *)(v272 + 464);
      v453 = &unk_49ECFC8;
      if ( v406 != &v407 )
        j_j___libc_free_0(v406, v407 + 1);
      sub_2240A30(v405);
      v468 = (unsigned __int64 *)&unk_49ECF68;
      sub_1897B80((__int64)v478);
      sub_143AA50(v267, (__int64)&v453);
      v453 = &unk_49ECF68;
      sub_1897B80((__int64)&v462);
    }
  }
  if ( v387 )
    j_j___libc_free_0(v387, v389 - (_BYTE *)v387);
  if ( v449 != v448 )
    _libc_free((unsigned __int64)v449);
  if ( v443 != v442 )
    _libc_free((unsigned __int64)v443);
  j___libc_free_0(v438);
  if ( (_DWORD)v436 )
  {
    v216 = (char *)src;
    v454 = (void *)2;
    v455 = 0;
    v217 = (char *)src + 40 * (unsigned int)v436;
    v456 = (__m128)0xFFFFFFFFFFFFFFF8LL;
    v453 = &unk_49E8A80;
    v469 = 2;
    v470 = 0;
    v471 = -16;
    v468 = (unsigned __int64 *)&unk_49E8A80;
    v472 = 0;
    do
    {
      v218 = *((_QWORD *)v216 + 3);
      *(_QWORD *)v216 = &unk_49EE2B0;
      if ( v218 != -8 && v218 != 0 && v218 != -16 )
        sub_1649B30((_QWORD *)v216 + 1);
      v216 += 40;
    }
    while ( v217 != v216 );
    v468 = (unsigned __int64 *)&unk_49EE2B0;
    if ( v471 != 0 && v471 != -8 && v471 != -16 )
      sub_1649B30(&v469);
    v453 = &unk_49EE2B0;
    if ( v456.m128_u64[0] != 0 && v456.m128_u64[0] != -8 && v456.m128_u64[0] != -16 )
      sub_1649B30(&v454);
  }
  j___libc_free_0(src);
  sub_142D890((__int64)&v420);
  v219 = v423;
  v220 = v422;
  if ( v422 != v423 )
  {
    v334 = v10;
    do
    {
      v221 = *v220;
      v222 = *(__int64 **)(*v220 + 8);
      v223 = *(__int64 **)(*v220 + 16);
      if ( v222 == v223 )
      {
        *(_BYTE *)(v221 + 160) = 1;
      }
      else
      {
        do
        {
          v224 = *v222++;
          sub_13FACC0(v224);
        }
        while ( v223 != v222 );
        *(_BYTE *)(v221 + 160) = 1;
        v225 = *(_QWORD *)(v221 + 8);
        if ( v225 != *(_QWORD *)(v221 + 16) )
          *(_QWORD *)(v221 + 16) = v225;
      }
      v226 = *(_QWORD *)(v221 + 32);
      if ( v226 != *(_QWORD *)(v221 + 40) )
        *(_QWORD *)(v221 + 40) = v226;
      ++*(_QWORD *)(v221 + 56);
      v227 = *(void **)(v221 + 72);
      if ( v227 == *(void **)(v221 + 64) )
      {
        *(_QWORD *)v221 = 0;
      }
      else
      {
        v228 = 4 * (*(_DWORD *)(v221 + 84) - *(_DWORD *)(v221 + 88));
        v229 = *(unsigned int *)(v221 + 80);
        if ( v228 < 0x20 )
          v228 = 32;
        if ( v228 < (unsigned int)v229 )
          sub_16CC920(v221 + 56);
        else
          memset(v227, -1, 8 * v229);
        v230 = *(_QWORD *)(v221 + 72);
        v231 = *(_QWORD *)(v221 + 64);
        *(_QWORD *)v221 = 0;
        if ( v230 != v231 )
          _libc_free(v230);
      }
      v232 = *(_QWORD *)(v221 + 32);
      if ( v232 )
        j_j___libc_free_0(v232, *(_QWORD *)(v221 + 48) - v232);
      v233 = *(_QWORD *)(v221 + 8);
      if ( v233 )
        j_j___libc_free_0(v233, *(_QWORD *)(v221 + 24) - v233);
      ++v220;
    }
    while ( v219 != v220 );
    v10 = v334;
    if ( v422 != v423 )
      v423 = v422;
  }
  v234 = v430;
  v235 = &v430[2 * v431];
  if ( v430 != v235 )
  {
    do
    {
      v236 = *v234;
      v234 += 2;
      _libc_free(v236);
    }
    while ( v234 != v235 );
  }
  v431 = 0;
  if ( v428 )
  {
    v237 = v427;
    v432[0] = 0;
    v238 = &v427[v428];
    v239 = v427 + 1;
    v425 = *v427;
    v426 = v425 + 4096;
    if ( v238 != v427 + 1 )
    {
      do
      {
        v240 = *v239++;
        _libc_free(v240);
      }
      while ( v238 != v239 );
      v237 = v427;
    }
    v428 = 1;
    _libc_free(*v237);
    v241 = v430;
    v242 = &v430[2 * v431];
    if ( v430 != v242 )
    {
      do
      {
        v243 = *v241;
        v241 += 2;
        _libc_free(v243);
      }
      while ( v242 != v241 );
      goto LABEL_326;
    }
  }
  else
  {
LABEL_326:
    v241 = v430;
  }
  if ( v241 != v432 )
    _libc_free((unsigned __int64)v241);
  if ( v427 != (unsigned __int64 *)v429 )
    _libc_free((unsigned __int64)v427);
  if ( v422 )
    j_j___libc_free_0(v422, v424 - (_QWORD)v422);
  j___libc_free_0(v421);
  if ( v398 )
  {
    v244 = v396;
    v245 = &v396[2 * v398];
    do
    {
      if ( *v244 != -16 && *v244 != -8 )
      {
        v246 = v244[1];
        if ( v246 )
        {
          v247 = *(_QWORD *)(v246 + 24);
          if ( v247 )
            j_j___libc_free_0(v247, *(_QWORD *)(v246 + 40) - v247);
          j_j___libc_free_0(v246, 56);
        }
      }
      v244 += 2;
    }
    while ( v245 != v244 );
  }
  j___libc_free_0(v396);
  if ( v392 != (__int64 *)&v394 )
    _libc_free((unsigned __int64)v392);
  v250 = v348 && (_DWORD)v413 && (unsigned __int8)sub_189A480(v10, (__int64)&v410);
  v349 = v250;
  sub_1897560(
    (__int64)&v410,
    a3,
    *(double *)a4.m128i_i64,
    *(double *)a5.m128i_i64,
    *(double *)a6.m128i_i64,
    v248,
    v249,
    a9,
    a10);
  v251 = v337[8];
  v252 = v349;
  if ( (_QWORD *)v251 != v337 + 10 )
  {
    _libc_free(v251);
    v252 = v349;
  }
  if ( (_QWORD *)*v337 != v337 + 2 )
  {
    v350 = v252;
    _libc_free(*v337);
    v252 = v350;
  }
  v351 = v252;
  j_j___libc_free_0(v337, 112);
  result = v351;
LABEL_37:
  v32 = v356;
  if ( v356 )
  {
    v341 = result;
    sub_1368A00(v356);
    j_j___libc_free_0(v32, 8);
    return v341;
  }
  return result;
}
