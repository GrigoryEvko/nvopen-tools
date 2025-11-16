// Function: sub_2770AC0
// Address: 0x2770ac0
//
__int64 __fastcall sub_2770AC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  int v14; // ecx
  __int64 v15; // r8
  int v16; // ecx
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r10
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  __int64 *v23; // rcx
  __int64 v24; // rsi
  __int64 *v25; // rdi
  _BYTE *v26; // rax
  bool v27; // zf
  __int64 v28; // rcx
  unsigned int v29; // esi
  unsigned int v30; // eax
  __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // r8
  int v34; // r11d
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // r10
  __int64 *v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rdi
  unsigned int v42; // edx
  __int64 v43; // rax
  __int64 *v44; // rbx
  __int64 *v45; // r14
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // r15
  char v50; // cl
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 *v58; // rcx
  __int64 *v59; // rdx
  __int64 *v60; // rax
  unsigned __int64 v61; // r12
  __int64 *v62; // rbx
  unsigned __int64 v63; // r14
  unsigned __int64 v64; // rdi
  unsigned __int64 *v65; // rdi
  __int64 *v66; // rbx
  unsigned __int64 v67; // r12
  unsigned __int64 v68; // rdi
  unsigned __int64 *v69; // rdi
  unsigned __int64 v70; // rbx
  unsigned __int64 v71; // r12
  unsigned __int64 *v72; // rdi
  __int64 *v73; // rbx
  unsigned __int64 v74; // r12
  unsigned __int64 v75; // rdi
  unsigned __int64 *v76; // rdi
  __int64 v77; // rdi
  __int64 *v78; // rbx
  unsigned __int64 v79; // r12
  unsigned __int64 v80; // rdi
  unsigned __int64 *v81; // rdi
  const void **v82; // r13
  unsigned __int64 v83; // r12
  unsigned __int64 v84; // rdi
  __int64 v85; // r14
  unsigned __int64 v86; // rbx
  unsigned __int64 v87; // rdi
  unsigned __int64 *v88; // rdi
  __int64 v89; // rcx
  _QWORD *v90; // r13
  _QWORD *v91; // rax
  unsigned __int64 **i; // rbx
  _QWORD *v93; // r12
  __int64 v94; // rcx
  __int64 v95; // rcx
  unsigned __int64 *v96; // rsi
  unsigned __int64 v97; // rdx
  __int64 v98; // rdi
  unsigned __int64 v99; // rdx
  __int64 *v100; // r8
  unsigned int v101; // eax
  __int64 *v102; // r8
  __int64 v103; // rax
  __int64 v104; // rsi
  unsigned __int64 *v105; // rdi
  __int64 v106; // rsi
  unsigned __int64 v107; // rcx
  __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // r10
  unsigned __int64 v111; // r9
  __int64 v112; // r8
  __int64 v113; // r11
  __int64 v114; // r10
  __int64 v115; // r9
  __int64 v116; // r8
  __int64 v117; // rax
  __int64 v118; // r13
  __int64 v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // r8
  __m128i v123; // xmm1
  __m128i v124; // xmm2
  __m128i v125; // xmm3
  unsigned __int64 *v126; // r14
  unsigned __int64 *v127; // rbx
  __int64 v128; // rax
  unsigned __int64 *v129; // r12
  unsigned __int64 v130; // rdi
  __int64 *v131; // rbx
  unsigned __int64 v132; // rdi
  unsigned __int64 *v133; // rdi
  int v134; // r12d
  char v135; // si
  unsigned int v136; // r14d
  __int64 v137; // rsi
  __int64 *v138; // rsi
  int v139; // edi
  int v140; // eax
  __int64 v141; // rcx
  __int64 v142; // r8
  __int64 v143; // r9
  __int64 v144; // rdx
  __int64 v145; // rsi
  unsigned __int64 v146; // rbx
  __int64 v147; // r12
  char *v148; // rbx
  char *j; // r13
  unsigned int v150; // eax
  __int64 v151; // rax
  __int64 v152; // rcx
  __int64 v153; // rsi
  __int64 v154; // r8
  __int64 v155; // r9
  char v156; // bl
  __int64 v157; // rbx
  unsigned __int64 v158; // r12
  unsigned __int64 v159; // rdi
  __int64 v160; // rbx
  unsigned __int64 v161; // r12
  unsigned __int64 v162; // rdi
  char v163; // al
  __int64 v164; // r13
  __int64 v165; // r15
  unsigned __int64 v166; // r14
  __int64 v167; // rax
  _QWORD *v168; // r14
  unsigned __int64 v169; // rax
  __int64 v170; // r14
  int v171; // r13d
  unsigned int v172; // r15d
  __int64 *v173; // rdx
  __int64 v174; // rcx
  __int64 v175; // rsi
  __int64 v176; // r9
  __int64 *v177; // rax
  __int64 v178; // rbx
  __int64 v179; // rdx
  __int64 v180; // rax
  const void **v181; // r12
  __int64 v182; // rax
  unsigned __int64 v183; // rsi
  __int64 v184; // r13
  __int64 v185; // rbx
  __int64 v186; // r12
  int v187; // r14d
  __int64 v188; // rdi
  __int64 v189; // rax
  __int64 v190; // r13
  unsigned __int64 v191; // rax
  __int64 v192; // rbx
  int v193; // r15d
  unsigned int v194; // r12d
  __int64 *v195; // rdx
  __int64 v196; // rcx
  __int64 v197; // r8
  __int64 v198; // r9
  __int64 v199; // r14
  _QWORD *v200; // rax
  __int64 v201; // r14
  unsigned __int16 v202; // bx
  __int64 v203; // rcx
  _QWORD *v204; // rdi
  __int64 v205; // r8
  __int64 v206; // r9
  const void **v207; // rax
  char v208; // dl
  const __m128i *v209; // rsi
  __int64 *k; // r15
  __int64 v211; // rax
  __int64 v212; // r9
  unsigned __int64 v213; // rbx
  __int64 v214; // r13
  __int64 v215; // rax
  __int64 *v216; // r14
  size_t v217; // r12
  __int64 v218; // rbx
  __int64 v219; // rax
  __int64 v220; // r14
  __int64 v221; // rsi
  unsigned __int64 v222; // rdx
  __int64 *v223; // r13
  __int64 *v224; // rbx
  unsigned __int64 m; // rax
  __int64 v226; // rdi
  unsigned int v227; // ecx
  __int64 v228; // rax
  __int64 *v229; // rbx
  __int64 *v230; // r12
  __int64 v231; // rsi
  __int64 v232; // rdi
  unsigned __int64 v233; // rbx
  unsigned __int64 v234; // r12
  unsigned __int64 v235; // rdi
  __int64 *v236; // rax
  __int64 v237; // r15
  const void **v238; // rbx
  const void **v239; // r12
  unsigned __int64 v240; // rdi
  __int64 v241; // rax
  _QWORD *v242; // r13
  _QWORD *v243; // rbx
  __int64 v244; // r14
  unsigned __int64 v245; // r12
  unsigned __int64 v246; // rdi
  __int64 v247; // rsi
  __int64 v248; // rdx
  __int64 v249; // rcx
  __int64 v250; // r8
  __int64 v251; // r9
  __int64 v252; // rdx
  __int64 v253; // rcx
  __int64 v254; // r8
  __int64 v255; // r9
  _QWORD *v256; // rbx
  _QWORD *v257; // r14
  void (__fastcall *v258)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v259; // rax
  __int64 v260; // rbx
  const char *v261; // rax
  __int64 v262; // rdx
  __int64 v263; // rbx
  __int64 v264; // rcx
  unsigned int v265; // eax
  __int64 v266; // rdx
  __int64 v267; // rax
  __int64 v268; // rbx
  __int64 v269; // rcx
  __int64 v270; // r14
  __int64 *v271; // rcx
  __int64 *v272; // r14
  __int64 *v273; // r12
  __int64 v274; // rcx
  __int64 *v275; // r8
  __int64 v276; // rdi
  unsigned __int64 v277; // r12
  __int64 *v278; // r13
  __int64 v279; // rax
  unsigned int v280; // esi
  __int64 v281; // r13
  __int64 *v282; // rsi
  unsigned __int64 v283; // rdi
  __int64 *v284; // r13
  __int64 v285; // r14
  __int64 v286; // rax
  __int64 *v287; // rax
  __int64 *v288; // r12
  __int64 v289; // r13
  __int64 v290; // rax
  __int64 v291; // rdx
  __int64 v292; // r8
  __int64 v293; // r9
  __m128i v294; // xmm4
  __m128i v295; // xmm5
  __m128i v296; // xmm6
  unsigned __int64 *v297; // r15
  unsigned __int64 v298; // rdi
  __int64 v299; // rax
  __int64 v300; // rax
  __int64 v301; // rax
  unsigned __int64 v302; // r14
  __int64 *v303; // rbx
  __int64 *v304; // rax
  unsigned __int64 v305; // r13
  __int64 *v306; // rax
  unsigned __int64 v307; // rdi
  unsigned __int64 *v308; // rdi
  __int64 v309; // rax
  __int64 v310; // rax
  __int64 v311; // rdx
  const void **v312; // rbx
  __int64 v313; // r14
  unsigned __int64 v314; // r9
  int v315; // eax
  __int64 v316; // r13
  unsigned __int64 v317; // r12
  __int64 *v318; // rbx
  unsigned __int64 v319; // r12
  unsigned __int64 v320; // rdi
  unsigned __int64 *v321; // rdi
  __int64 v322; // r15
  char *v323; // r12
  char *v324; // r14
  unsigned int v325; // eax
  unsigned __int64 v326; // r12
  const void **v327; // rax
  unsigned __int64 v328; // r15
  __int64 v329; // rdx
  const void *v330; // rcx
  __int64 v331; // r13
  __int64 v332; // rbx
  unsigned __int64 v333; // rdi
  unsigned __int64 *v334; // rdi
  int v335; // r13d
  const void **v336; // [rsp+18h] [rbp-B68h]
  char v337; // [rsp+20h] [rbp-B60h]
  __int64 v339; // [rsp+30h] [rbp-B50h]
  const void **v340; // [rsp+38h] [rbp-B48h]
  unsigned __int8 v341; // [rsp+40h] [rbp-B40h]
  __int64 v342; // [rsp+48h] [rbp-B38h]
  const void **v343; // [rsp+48h] [rbp-B38h]
  __int64 v344; // [rsp+48h] [rbp-B38h]
  __int64 *v345; // [rsp+48h] [rbp-B38h]
  unsigned __int64 **v346; // [rsp+50h] [rbp-B30h]
  __int64 v348; // [rsp+60h] [rbp-B20h]
  __int64 *v349; // [rsp+70h] [rbp-B10h]
  int v350; // [rsp+70h] [rbp-B10h]
  const void *v351; // [rsp+70h] [rbp-B10h]
  const void **v352; // [rsp+70h] [rbp-B10h]
  __int64 *v353; // [rsp+70h] [rbp-B10h]
  __int64 *v354; // [rsp+70h] [rbp-B10h]
  const void **v355; // [rsp+78h] [rbp-B08h]
  char v356; // [rsp+80h] [rbp-B00h]
  unsigned __int64 v357; // [rsp+80h] [rbp-B00h]
  unsigned __int64 v358; // [rsp+80h] [rbp-B00h]
  __int64 *v359; // [rsp+80h] [rbp-B00h]
  __int64 v360; // [rsp+88h] [rbp-AF8h]
  __int64 v361; // [rsp+88h] [rbp-AF8h]
  __int64 v362; // [rsp+88h] [rbp-AF8h]
  __int64 v363; // [rsp+88h] [rbp-AF8h]
  __int64 *v364; // [rsp+88h] [rbp-AF8h]
  __int64 *v365; // [rsp+88h] [rbp-AF8h]
  const void **v366; // [rsp+88h] [rbp-AF8h]
  unsigned __int64 v367; // [rsp+90h] [rbp-AF0h] BYREF
  __int64 *v368; // [rsp+98h] [rbp-AE8h]
  __int64 *v369; // [rsp+A0h] [rbp-AE0h]
  unsigned __int64 v370; // [rsp+B0h] [rbp-AD0h] BYREF
  unsigned __int64 v371; // [rsp+B8h] [rbp-AC8h]
  __int64 v372; // [rsp+C0h] [rbp-AC0h]
  unsigned int v373; // [rsp+C8h] [rbp-AB8h]
  __int64 *v374; // [rsp+D0h] [rbp-AB0h] BYREF
  __int64 *v375; // [rsp+D8h] [rbp-AA8h]
  __int64 *v376; // [rsp+E0h] [rbp-AA0h]
  unsigned int v377; // [rsp+E8h] [rbp-A98h]
  const void **v378; // [rsp+F0h] [rbp-A90h]
  __int64 v379; // [rsp+F8h] [rbp-A88h]
  const void *v380; // [rsp+100h] [rbp-A80h] BYREF
  __int64 v381; // [rsp+108h] [rbp-A78h]
  _QWORD *v382; // [rsp+110h] [rbp-A70h]
  __int64 *v383; // [rsp+118h] [rbp-A68h]
  unsigned __int64 v384; // [rsp+120h] [rbp-A60h]
  __int64 *v385; // [rsp+128h] [rbp-A58h]
  __int64 *v386; // [rsp+130h] [rbp-A50h]
  __int64 v387; // [rsp+138h] [rbp-A48h]
  _QWORD *v388; // [rsp+140h] [rbp-A40h]
  __int64 v389; // [rsp+150h] [rbp-A30h] BYREF
  char *v390; // [rsp+158h] [rbp-A28h]
  const void *v391; // [rsp+160h] [rbp-A20h]
  const void *v392; // [rsp+168h] [rbp-A18h]
  unsigned __int64 v393; // [rsp+170h] [rbp-A10h]
  __int64 v394; // [rsp+178h] [rbp-A08h]
  unsigned __int64 v395; // [rsp+180h] [rbp-A00h]
  const void *v396; // [rsp+188h] [rbp-9F8h]
  const void *v397; // [rsp+190h] [rbp-9F0h]
  __m128i v398; // [rsp+1A0h] [rbp-9E0h] BYREF
  __int64 v399; // [rsp+1B0h] [rbp-9D0h]
  _BYTE *v400; // [rsp+1B8h] [rbp-9C8h] BYREF
  __int64 v401; // [rsp+1C0h] [rbp-9C0h]
  _BYTE v402[72]; // [rsp+1C8h] [rbp-9B8h] BYREF
  __m128i *v403; // [rsp+210h] [rbp-970h] BYREF
  __m128i *v404; // [rsp+218h] [rbp-968h]
  _QWORD v405[16]; // [rsp+220h] [rbp-960h] BYREF
  const void **v406; // [rsp+2A0h] [rbp-8E0h] BYREF
  __int64 v407; // [rsp+2A8h] [rbp-8D8h]
  _BYTE v408[144]; // [rsp+2B0h] [rbp-8D0h] BYREF
  __int64 v409; // [rsp+340h] [rbp-840h] BYREF
  __int64 *v410; // [rsp+348h] [rbp-838h]
  __int64 v411; // [rsp+350h] [rbp-830h]
  __int64 v412; // [rsp+358h] [rbp-828h]
  char v413; // [rsp+360h] [rbp-820h] BYREF
  __int64 v414; // [rsp+3E0h] [rbp-7A0h] BYREF
  char *v415; // [rsp+3E8h] [rbp-798h]
  __int64 v416; // [rsp+3F0h] [rbp-790h]
  unsigned __int64 *v417; // [rsp+3F8h] [rbp-788h]
  char v418; // [rsp+400h] [rbp-780h] BYREF
  __int64 *v419; // [rsp+500h] [rbp-680h] BYREF
  __m128i v420; // [rsp+508h] [rbp-678h]
  __int64 v421; // [rsp+518h] [rbp-668h]
  __int64 v422; // [rsp+520h] [rbp-660h]
  char v423[8]; // [rsp+528h] [rbp-658h] BYREF
  unsigned __int64 v424; // [rsp+530h] [rbp-650h]
  char v425; // [rsp+544h] [rbp-63Ch]
  char v426[256]; // [rsp+548h] [rbp-638h] BYREF
  unsigned __int64 v427; // [rsp+648h] [rbp-538h]
  __int64 v428; // [rsp+650h] [rbp-530h]
  __int64 v429; // [rsp+658h] [rbp-528h]
  __int64 *v430; // [rsp+660h] [rbp-520h] BYREF
  unsigned __int64 v431; // [rsp+668h] [rbp-518h]
  __int64 v432; // [rsp+670h] [rbp-510h] BYREF
  __m128i v433; // [rsp+678h] [rbp-508h] BYREF
  __int64 v434; // [rsp+688h] [rbp-4F8h]
  __m128i v435; // [rsp+690h] [rbp-4F0h]
  __m128i v436; // [rsp+6A0h] [rbp-4E0h]
  unsigned __int64 *v437; // [rsp+6B0h] [rbp-4D0h] BYREF
  __int64 v438; // [rsp+6B8h] [rbp-4C8h]
  _BYTE v439[320]; // [rsp+6C0h] [rbp-4C0h] BYREF
  char v440; // [rsp+800h] [rbp-380h]
  int v441; // [rsp+804h] [rbp-37Ch]
  __int64 v442; // [rsp+808h] [rbp-378h]
  __int64 v443; // [rsp+810h] [rbp-370h]
  __int64 v444; // [rsp+818h] [rbp-368h]
  __int64 v445; // [rsp+820h] [rbp-360h]
  unsigned int v446; // [rsp+828h] [rbp-358h]
  __int64 v447; // [rsp+830h] [rbp-350h]
  __int64 v448; // [rsp+838h] [rbp-348h]
  __int64 *v449; // [rsp+840h] [rbp-340h]
  __int64 v450; // [rsp+848h] [rbp-338h]
  _BYTE v451[32]; // [rsp+850h] [rbp-330h] BYREF
  __int64 *v452; // [rsp+870h] [rbp-310h]
  __int64 v453; // [rsp+878h] [rbp-308h]
  _QWORD v454[2]; // [rsp+880h] [rbp-300h] BYREF
  __int64 *v455; // [rsp+890h] [rbp-2F0h] BYREF
  unsigned __int64 v456; // [rsp+898h] [rbp-2E8h]
  __int64 v457; // [rsp+8A0h] [rbp-2E0h] BYREF
  __m128i v458; // [rsp+8A8h] [rbp-2D8h] BYREF
  __int64 v459; // [rsp+8B8h] [rbp-2C8h]
  __m128i v460; // [rsp+8C0h] [rbp-2C0h] BYREF
  __m128i v461; // [rsp+8D0h] [rbp-2B0h] BYREF
  unsigned __int64 *v462; // [rsp+8E0h] [rbp-2A0h] BYREF
  unsigned int v463; // [rsp+8E8h] [rbp-298h]
  unsigned __int64 *v464; // [rsp+8F0h] [rbp-290h] BYREF
  char v465; // [rsp+8F8h] [rbp-288h]
  char v466; // [rsp+A30h] [rbp-150h]
  int v467; // [rsp+A34h] [rbp-14Ch]
  __int64 v468; // [rsp+A38h] [rbp-148h]
  __int64 v469; // [rsp+AA0h] [rbp-E0h]
  __int64 v470; // [rsp+AA8h] [rbp-D8h]
  __int64 v471; // [rsp+AB0h] [rbp-D0h]
  __int64 v472; // [rsp+AB8h] [rbp-C8h]
  char v473; // [rsp+AC0h] [rbp-C0h]
  __int64 v474; // [rsp+AC8h] [rbp-B8h]
  char *v475; // [rsp+AD0h] [rbp-B0h]
  __int64 v476; // [rsp+AD8h] [rbp-A8h]
  int v477; // [rsp+AE0h] [rbp-A0h]
  char v478; // [rsp+AE4h] [rbp-9Ch]
  char v479; // [rsp+AE8h] [rbp-98h] BYREF
  __int16 v480; // [rsp+B28h] [rbp-58h]
  _QWORD *v481; // [rsp+B30h] [rbp-50h]
  _QWORD *v482; // [rsp+B38h] [rbp-48h]
  __int64 v483; // [rsp+B40h] [rbp-40h]

  if ( (unsigned __int8)sub_B2D610(a2, 47) )
    return 0;
  v337 = sub_B2D610(a2, 18);
  if ( v337 )
    return 0;
  if ( byte_4FFB0A8 )
    sub_11FAD00(a2);
  v341 = 0;
  v406 = (const void **)v408;
  v407 = 0x200000000LL;
  *(_BYTE *)a1 = 0;
  v342 = a2 + 72;
  v348 = *(_QWORD *)(a2 + 80);
  if ( v348 != a2 + 72 )
  {
    do
    {
      if ( !v348 )
        BUG();
      v3 = *(_QWORD *)(v348 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v3 == v348 + 24 || !v3 || (v4 = v3 - 24, (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA) )
LABEL_570:
        BUG();
      if ( *(_BYTE *)(v3 - 24) == 32 )
      {
        v399 = 0;
        v398.m128i_i64[0] = (__int64)&off_49D3CE8;
        v5 = *(__int64 **)(a1 + 40);
        v398.m128i_i64[1] = *(_QWORD *)(a1 + 24);
        v400 = v402;
        v401 = 0x400000000LL;
        if ( (unsigned __int8)sub_2766590((__int64)&v398, v3 - 24) )
        {
          v399 = v4;
LABEL_15:
          v455 = &v457;
          v456 = 0x400000000LL;
          if ( (_DWORD)v401 )
            sub_27656D0((__int64)&v455, (__int64)&v400, v6, v7, (unsigned int)v401, v8);
          sub_27678E0(a1, *(_BYTE **)(a1 + 16), (unsigned __int64)&v455, v7);
          if ( v455 != &v457 )
            _libc_free((unsigned __int64)v455);
          v455 = &v457;
          v456 = 0x400000000LL;
          if ( (_DWORD)v401 )
          {
            sub_27656D0((__int64)&v455, (__int64)&v400, v9, v10, v11, v12);
            v134 = v456;
            if ( v455 != &v457 )
              _libc_free((unsigned __int64)v455);
            v135 = v341;
            if ( v134 )
              v135 = 1;
            v341 = v135;
          }
          v13 = *(_QWORD *)(a1 + 24);
          v14 = *(_DWORD *)(v13 + 24);
          v15 = *(_QWORD *)(v13 + 8);
          if ( v14 )
          {
            v16 = v14 - 1;
            v17 = v16 & (((unsigned int)(v348 - 24) >> 9) ^ ((unsigned int)(v348 - 24) >> 4));
            v18 = (__int64 *)(v15 + 16LL * v17);
            v19 = *v18;
            if ( v348 - 24 == *v18 )
            {
LABEL_22:
              v20 = (_QWORD *)v18[1];
LABEL_23:
              v21 = v20;
              do
              {
                v22 = v21;
                v21 = (_QWORD *)*v21;
              }
              while ( v21 );
              v339 = 0;
              LODWORD(v380) = 0;
              v23 = *(__int64 **)(a1 + 40);
              v381 = v399;
              v24 = *(_QWORD *)(v399 + 40);
              v383 = v23;
              v382 = (_QWORD *)v24;
              v387 = v13;
              v25 = &v432;
              v384 = 0;
              v385 = 0;
              v386 = 0;
              v388 = v22;
              v389 = 0;
              v390 = 0;
              v391 = 0;
              LODWORD(v392) = 0;
              v26 = **(_BYTE ***)(v399 - 8);
              v27 = *v26 == 84;
              v430 = &v432;
              if ( !v27 )
                v26 = 0;
              v28 = 0;
              v29 = 0;
              v455 = 0;
              v457 = 16;
              v432 = (__int64)v26;
              v458.m128i_i32[0] = 0;
              v458.m128i_i8[4] = 1;
              v431 = 0x800000001LL;
              v456 = (unsigned __int64)&v458.m128i_u64[1];
              v30 = 1;
              while ( 2 )
              {
                v31 = v25[v30 - 1];
                LODWORD(v431) = v30 - 1;
                v32 = *(_QWORD *)(v31 + 40);
                if ( v29 )
                {
                  v33 = v29 - 1;
                  v34 = 1;
                  LODWORD(v35) = v33 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                  v36 = v28 + 16LL * (unsigned int)v35;
                  v37 = 0;
                  v38 = *(_QWORD *)v36;
                  if ( v32 == *(_QWORD *)v36 )
                  {
LABEL_30:
                    v39 = (__int64 *)(v36 + 8);
                    goto LABEL_31;
                  }
                  while ( v38 != -4096 )
                  {
                    if ( v38 == -8192 && !v37 )
                      v37 = (__int64 *)v36;
                    v12 = (unsigned int)(v34 + 1);
                    v35 = (unsigned int)v33 & ((_DWORD)v35 + v34);
                    v36 = v28 + 16 * v35;
                    v38 = *(_QWORD *)v36;
                    if ( v32 == *(_QWORD *)v36 )
                      goto LABEL_30;
                    ++v34;
                  }
                  if ( !v37 )
                    v37 = (__int64 *)v36;
                  ++v389;
                  v36 = (unsigned int)((_DWORD)v391 + 1);
                  if ( 4 * (int)v36 < 3 * v29 )
                  {
                    v28 = v29 >> 3;
                    if ( v29 - ((_DWORD)v36 + HIDWORD(v391)) <= (unsigned int)v28 )
                    {
                      sub_2769990((__int64)&v389, v29);
                      if ( !(_DWORD)v392 )
                        goto LABEL_571;
                      v33 = (__int64)v390;
                      v12 = 0;
                      v136 = ((_DWORD)v392 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                      v36 = (unsigned int)((_DWORD)v391 + 1);
                      v28 = 1;
                      v37 = (__int64 *)&v390[16 * v136];
                      v137 = *v37;
                      if ( v32 != *v37 )
                      {
                        while ( v137 != -4096 )
                        {
                          if ( !v12 && v137 == -8192 )
                            v12 = (__int64)v37;
                          v136 = ((_DWORD)v392 - 1) & (v28 + v136);
                          v37 = (__int64 *)&v390[16 * v136];
                          v137 = *v37;
                          if ( v32 == *v37 )
                            goto LABEL_192;
                          v28 = (unsigned int)(v28 + 1);
                        }
                        if ( v12 )
                          v37 = (__int64 *)v12;
                      }
                    }
LABEL_192:
                    LODWORD(v391) = v36;
                    if ( *v37 != -4096 )
                      --HIDWORD(v391);
                    *v37 = v32;
                    v39 = v37 + 1;
                    *v39 = 0;
LABEL_31:
                    *v39 = v31;
                    if ( !v458.m128i_i8[4] )
                      goto LABEL_61;
                    v40 = (_QWORD *)v456;
                    v28 = HIDWORD(v457);
                    v36 = v456 + 8LL * HIDWORD(v457);
                    if ( v456 == v36 )
                    {
LABEL_123:
                      if ( HIDWORD(v457) < (unsigned int)v457 )
                      {
                        ++HIDWORD(v457);
                        *(_QWORD *)v36 = v31;
                        v455 = (__int64 *)((char *)v455 + 1);
                        goto LABEL_36;
                      }
LABEL_61:
                      sub_C8CC70((__int64)&v455, v31, v36, v28, v33, v12);
                      goto LABEL_36;
                    }
                    while ( v31 != *v40 )
                    {
                      if ( (_QWORD *)v36 == ++v40 )
                        goto LABEL_123;
                    }
LABEL_36:
                    v41 = *(_QWORD *)(v31 - 8);
                    v42 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
                    v43 = 32LL * *(unsigned int *)(v31 + 72);
                    v44 = (__int64 *)(v41 + v43);
                    v45 = (__int64 *)(v41 + v43 + 8LL * v42);
                    if ( v45 != (__int64 *)(v41 + v43) )
                    {
                      while ( 1 )
                      {
                        v46 = *v44;
                        v47 = 0x1FFFFFFFE0LL;
                        if ( v42 )
                        {
                          v48 = 0;
                          do
                          {
                            if ( v46 == *(_QWORD *)(v41 + 32LL * *(unsigned int *)(v31 + 72) + 8 * v48) )
                            {
                              v47 = 32 * v48;
                              goto LABEL_42;
                            }
                            ++v48;
                          }
                          while ( v42 != (_DWORD)v48 );
                          v49 = *(_BYTE **)(v41 + 0x1FFFFFFFE0LL);
                          if ( *v49 != 84 )
                            goto LABEL_43;
                        }
                        else
                        {
LABEL_42:
                          v49 = *(_BYTE **)(v41 + v47);
                          if ( *v49 != 84 )
                            goto LABEL_43;
                        }
                        v50 = *((_BYTE *)v388 + 84);
                        if ( !v50 )
                          break;
                        v51 = (_QWORD *)v388[8];
                        v52 = &v51[*((unsigned int *)v388 + 19)];
                        if ( v51 != v52 )
                        {
                          while ( v46 != *v51 )
                          {
                            if ( v52 == ++v51 )
                              goto LABEL_62;
                          }
LABEL_51:
                          if ( !v458.m128i_i8[4] )
                            goto LABEL_63;
                          goto LABEL_52;
                        }
LABEL_62:
                        v50 = 0;
                        if ( !v458.m128i_i8[4] )
                        {
LABEL_63:
                          v356 = v50;
                          v57 = sub_C8CA60((__int64)&v455, (__int64)v49);
                          v50 = v356;
                          if ( v57 )
                            goto LABEL_43;
                          goto LABEL_55;
                        }
LABEL_52:
                        v53 = (_QWORD *)v456;
                        v54 = (_QWORD *)(v456 + 8LL * HIDWORD(v457));
                        if ( (_QWORD *)v456 != v54 )
                        {
                          while ( v49 != (_BYTE *)*v53 )
                          {
                            if ( v54 == ++v53 )
                              goto LABEL_55;
                          }
LABEL_43:
                          if ( v45 == ++v44 )
                            goto LABEL_59;
                          goto LABEL_44;
                        }
LABEL_55:
                        if ( !v50 )
                          goto LABEL_43;
                        v55 = (unsigned int)v431;
                        v56 = (unsigned int)v431 + 1LL;
                        if ( v56 > HIDWORD(v431) )
                        {
                          sub_C8D5F0((__int64)&v430, &v432, v56, 8u, v33, v12);
                          v55 = (unsigned int)v431;
                        }
                        ++v44;
                        v430[v55] = (__int64)v49;
                        LODWORD(v431) = v431 + 1;
                        if ( v45 == v44 )
                          goto LABEL_59;
LABEL_44:
                        v41 = *(_QWORD *)(v31 - 8);
                        v42 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
                      }
                      v50 = sub_C8CA60((__int64)(v388 + 7), v46) != 0;
                      goto LABEL_51;
                    }
LABEL_59:
                    v30 = v431;
                    if ( (_DWORD)v431 )
                    {
                      v25 = v430;
                      v28 = (__int64)v390;
                      v29 = (unsigned int)v392;
                      continue;
                    }
                    if ( !v458.m128i_i8[4] )
                      _libc_free(v456);
                    if ( v430 != &v432 )
                      _libc_free((unsigned __int64)v430);
                    if ( !(_DWORD)v391 )
                    {
                      v288 = v383;
                      v289 = *v383;
                      v290 = sub_B2BE50(*v383);
                      if ( sub_B6EA50(v290)
                        || (v309 = sub_B2BE50(v289),
                            v310 = sub_B6F970(v309),
                            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v310 + 48LL))(v310)) )
                      {
                        sub_B176B0(
                          (__int64)&v455,
                          (__int64)"dfa-jump-threading",
                          (__int64)"SwitchNotPredictable",
                          20,
                          v381);
                        sub_B18290((__int64)&v455, "Switch instruction is not predictable.", 0x26u);
                        v294 = _mm_loadu_si128(&v458);
                        v295 = _mm_loadu_si128(&v460);
                        LODWORD(v431) = v456;
                        v296 = _mm_loadu_si128(&v461);
                        v433 = v294;
                        BYTE4(v431) = BYTE4(v456);
                        v435 = v295;
                        v432 = v457;
                        v430 = (__int64 *)&unk_49D9D40;
                        v436 = v296;
                        v434 = v459;
                        v437 = (unsigned __int64 *)v439;
                        v438 = 0x400000000LL;
                        if ( v463 )
                          sub_2768F90((__int64)&v437, (__int64)&v462, v291, v463, v292, v293);
                        v455 = (__int64 *)&unk_49D9D40;
                        v440 = v466;
                        v441 = v467;
                        v442 = v468;
                        v430 = (__int64 *)&unk_49D9DB0;
                        sub_23FD590((__int64)&v462);
                        sub_1049740(v288, (__int64)&v430);
                        v430 = (__int64 *)&unk_49D9D40;
                        sub_23FD590((__int64)&v437);
                      }
                      goto LABEL_96;
                    }
                    v89 = **(_QWORD **)(v381 - 8);
                    v90 = *(_QWORD **)(v89 + 40);
                    v431 = (unsigned __int64)&v433.m128i_u64[1];
                    v430 = 0;
                    v432 = 8;
                    v433.m128i_i32[0] = 0;
                    v433.m128i_i8[4] = 1;
                    sub_276F9C0((__int64 *)&v367, &v380, &v389, v89, (__int64)&v430, v12);
                    if ( v90 == v382 )
                    {
                      v301 = v367;
                      v302 = v384;
                      v367 = 0;
                      v303 = v385;
                      v384 = v301;
                      v304 = v368;
                      v305 = v302;
                      v368 = 0;
                      v385 = v304;
                      v306 = v369;
                      v369 = 0;
                      v386 = v306;
                      while ( v303 != (__int64 *)v305 )
                      {
                        if ( *(_DWORD *)(v305 + 88) > 0x40u )
                        {
                          v307 = *(_QWORD *)(v305 + 80);
                          if ( v307 )
                            j_j___libc_free_0_0(v307);
                        }
                        v308 = (unsigned __int64 *)v305;
                        v305 += 112LL;
                        sub_2767770(v308);
                      }
                      if ( v302 )
                        j_j___libc_free_0(v302);
                    }
                    else
                    {
                      sub_276A3A0(&v370, (__int64)&v380, v90, (__int64)v382, (__int64)&v430, 1);
                      v357 = v371;
                      v91 = (_QWORD *)v370;
                      if ( v371 != v370 )
                      {
                        v374 = 0;
                        v375 = 0;
                        v376 = 0;
                        v346 = (unsigned __int64 **)v368;
                        if ( (__int64 *)v367 == v368 )
                        {
                          v60 = 0;
                          v59 = 0;
                          v58 = 0;
                          goto LABEL_66;
                        }
                        for ( i = (unsigned __int64 **)(v367 + 80); ; i += 14 )
                        {
                          v93 = v91;
                          if ( (_QWORD *)v357 != v91 )
                            break;
LABEL_156:
                          if ( v346 == i + 4 )
                          {
                            v58 = v374;
                            v59 = v375;
                            v60 = v376;
LABEL_66:
                            v61 = v384;
                            v62 = v385;
                            v384 = (unsigned __int64)v58;
                            v385 = v59;
                            v386 = v60;
                            v63 = v61;
                            v374 = 0;
                            v375 = 0;
                            v376 = 0;
                            while ( v62 != (__int64 *)v63 )
                            {
                              if ( *(_DWORD *)(v63 + 88) > 0x40u )
                              {
                                v64 = *(_QWORD *)(v63 + 80);
                                if ( v64 )
                                  j_j___libc_free_0_0(v64);
                              }
                              v65 = (unsigned __int64 *)v63;
                              v63 += 112LL;
                              sub_2767770(v65);
                            }
                            if ( v61 )
                              j_j___libc_free_0(v61);
                            v66 = v375;
                            v67 = (unsigned __int64)v374;
                            if ( v375 != v374 )
                            {
                              do
                              {
                                if ( *(_DWORD *)(v67 + 88) > 0x40u )
                                {
                                  v68 = *(_QWORD *)(v67 + 80);
                                  if ( v68 )
                                    j_j___libc_free_0_0(v68);
                                }
                                v69 = (unsigned __int64 *)v67;
                                v67 += 112LL;
                                sub_2767770(v69);
                              }
                              while ( v66 != (__int64 *)v67 );
                              v67 = (unsigned __int64)v374;
                            }
                            if ( v67 )
                              j_j___libc_free_0(v67);
                            v70 = v371;
                            v71 = v370;
                            if ( v371 != v370 )
                            {
                              do
                              {
                                v72 = (unsigned __int64 *)v71;
                                v71 += 80LL;
                                sub_2767770(v72);
                              }
                              while ( v70 != v71 );
                              v71 = v370;
                            }
                            if ( v71 )
                              j_j___libc_free_0(v71);
                            v73 = v368;
                            v74 = v367;
                            if ( v368 == (__int64 *)v367 )
                              goto LABEL_92;
                            do
                            {
                              if ( *(_DWORD *)(v74 + 88) > 0x40u )
                              {
                                v75 = *(_QWORD *)(v74 + 80);
                                if ( v75 )
                                  j_j___libc_free_0_0(v75);
                              }
                              v76 = (unsigned __int64 *)v74;
                              v74 += 112LL;
                              sub_2767770(v76);
                            }
                            while ( v73 != (__int64 *)v74 );
                            goto LABEL_91;
                          }
                          v91 = (_QWORD *)v370;
                          v357 = v371;
                        }
                        while ( 2 )
                        {
                          v103 = *(i - 4) - *(i - 3) + ((*(i - 1) - *(i - 5) - 1) << 6);
                          v104 = (char *)*(i - 6) - (char *)*(i - 8);
                          v455 = 0;
                          v456 = 0;
                          v457 = 0;
                          v458 = 0u;
                          v459 = 0;
                          v460 = 0u;
                          v461 = 0u;
                          sub_2768EA0((__int64 *)&v455, v103 + (v104 >> 3));
                          v105 = *(i - 1);
                          v106 = (__int64)*(i - 8);
                          v107 = (unsigned __int64)*(i - 7);
                          v108 = (__int64)*(i - 6);
                          v109 = (__int64)*(i - 5);
                          v110 = (__int64)*(i - 4);
                          v111 = (unsigned __int64)*(i - 3);
                          v112 = (__int64)*(i - 2);
                          v419 = (__int64 *)v457;
                          v417 = v105;
                          v410 = (__int64 *)v107;
                          v420 = v458;
                          v409 = v106;
                          v411 = v108;
                          v412 = v109;
                          v421 = v459;
                          v414 = v110;
                          v415 = (char *)v111;
                          v416 = v112;
                          sub_2769DA0(&v403, (__int64)&v409, &v414, (__int64)&v419);
                          v463 = *((_DWORD *)i + 2);
                          if ( v463 > 0x40 )
                            sub_C43780((__int64)&v462, (const void **)i);
                          else
                            v462 = *i;
                          v464 = i[2];
                          v465 = *((_BYTE *)i + 24);
                          v97 = v93[3];
                          v113 = v93[6];
                          v114 = v93[7];
                          v115 = v93[8];
                          v116 = v93[9];
                          v98 = v93[4];
                          v96 = (unsigned __int64 *)v93[5];
                          v117 = ((__int64)(v93[2] - v97) >> 3) + 1;
                          if ( v117 >= 0 )
                          {
                            v94 = v93[2] + 8LL;
                            if ( v117 > 63 )
                            {
                              v95 = v117 >> 6;
                              goto LABEL_138;
                            }
                          }
                          else
                          {
                            v95 = ~((unsigned __int64)~v117 >> 6);
LABEL_138:
                            v96 += v95;
                            v97 = *v96;
                            v98 = *v96 + 512;
                            v94 = *v96 + 8 * (v117 - (v95 << 6));
                          }
                          v415 = (char *)v97;
                          v421 = v116;
                          v414 = v94;
                          v416 = v98;
                          v417 = v96;
                          v419 = (__int64 *)v113;
                          v420.m128i_i64[0] = v114;
                          v420.m128i_i64[1] = v115;
                          v409 = v460.m128i_i64[0];
                          v99 = *(_QWORD *)v461.m128i_i64[1];
                          v412 = v461.m128i_i64[1];
                          v410 = (__int64 *)v99;
                          v411 = v99 + 512;
                          sub_276F640((unsigned __int64 *)&v455, &v409, &v414, (__int64 *)&v419);
                          v100 = v375;
                          if ( v375 == v376 )
                          {
                            sub_276BEE0((unsigned __int64 *)&v374, (__int64)v375, &v455);
                          }
                          else
                          {
                            if ( v375 )
                            {
                              v349 = v375;
                              sub_276A0C0(v375, &v455);
                              v101 = v463;
                              v102 = v349;
                              *((_DWORD *)v349 + 22) = v463;
                              if ( v101 > 0x40 )
                              {
                                sub_C43780((__int64)(v349 + 10), (const void **)&v462);
                                v102 = v349;
                              }
                              else
                              {
                                v349[10] = (__int64)v462;
                              }
                              v102[12] = (__int64)v464;
                              *((_BYTE *)v102 + 104) = v465;
                              v100 = v375;
                            }
                            v375 = v100 + 14;
                          }
                          if ( v463 > 0x40 && v462 )
                            j_j___libc_free_0_0((unsigned __int64)v462);
                          v93 += 10;
                          sub_2767770((unsigned __int64 *)&v455);
                          if ( (_QWORD *)v357 == v93 )
                            goto LABEL_156;
                          continue;
                        }
                      }
                      if ( v371 )
                        j_j___libc_free_0(v371);
                    }
                    v131 = v368;
                    v74 = v367;
                    if ( v368 == (__int64 *)v367 )
                      goto LABEL_92;
                    do
                    {
                      if ( *(_DWORD *)(v74 + 88) > 0x40u )
                      {
                        v132 = *(_QWORD *)(v74 + 80);
                        if ( v132 )
                          j_j___libc_free_0_0(v132);
                      }
                      v133 = (unsigned __int64 *)v74;
                      v74 += 112LL;
                      sub_2767770(v133);
                    }
                    while ( v131 != (__int64 *)v74 );
LABEL_91:
                    v74 = v367;
LABEL_92:
                    if ( v74 )
                      j_j___libc_free_0(v74);
                    if ( !v433.m128i_i8[4] )
                      _libc_free(v431);
LABEL_96:
                    v77 = (__int64)v390;
                    sub_C7D6A0((__int64)v390, 16LL * (unsigned int)v392, 8);
                    v78 = v385;
                    v79 = v384;
                    if ( !(-1227133513 * (unsigned int)((__int64)((__int64)v385 - v384) >> 4)) )
                    {
                      if ( v385 != (__int64 *)v384 )
                      {
                        do
                        {
                          if ( *(_DWORD *)(v79 + 88) > 0x40u )
                          {
                            v80 = *(_QWORD *)(v79 + 80);
                            if ( v80 )
                              j_j___libc_free_0_0(v80);
                          }
                          v81 = (unsigned __int64 *)v79;
                          v79 += 112LL;
                          sub_2767770(v81);
                        }
                        while ( v78 != (__int64 *)v79 );
                        v79 = v384;
                      }
                      if ( v79 )
                        j_j___libc_free_0(v79);
                      goto LABEL_105;
                    }
                    v311 = (unsigned int)v407;
                    v145 = HIDWORD(v407);
                    v312 = &v380;
                    v313 = (__int64)v406;
                    v314 = (unsigned int)v407 + 1LL;
                    v315 = v407;
                    if ( v314 > HIDWORD(v407) )
                    {
                      v326 = -1;
                      if ( v406 <= &v380 && &v380 < &v406[9 * (unsigned int)v407] )
                      {
                        v337 = 1;
                        v326 = 0x8E38E38E38E38E39LL * (&v380 - v406);
                      }
                      v145 = (__int64)v408;
                      v77 = (__int64)&v406;
                      v313 = sub_C8D7D0(
                               (__int64)&v406,
                               (__int64)v408,
                               (unsigned int)v407 + 1LL,
                               0x48u,
                               (unsigned __int64 *)&v455,
                               v314);
                      v327 = v406;
                      v328 = (unsigned __int64)&v406[9 * (unsigned int)v407];
                      if ( v406 != (const void **)v328 )
                      {
                        v329 = v313;
                        do
                        {
                          if ( v329 )
                          {
                            *(_DWORD *)v329 = *(_DWORD *)v327;
                            *(_QWORD *)(v329 + 8) = v327[1];
                            *(_QWORD *)(v329 + 16) = v327[2];
                            *(_QWORD *)(v329 + 24) = v327[3];
                            *(_QWORD *)(v329 + 32) = v327[4];
                            *(_QWORD *)(v329 + 40) = v327[5];
                            *(_QWORD *)(v329 + 48) = v327[6];
                            v330 = v327[7];
                            v327[6] = 0;
                            v327[5] = 0;
                            v327[4] = 0;
                            *(_QWORD *)(v329 + 56) = v330;
                            *(_QWORD *)(v329 + 64) = v327[8];
                          }
                          v327 += 9;
                          v329 += 72;
                        }
                        while ( (const void **)v328 != v327 );
                        v145 = (__int64)v406;
                        v366 = v406;
                        v328 = (unsigned __int64)&v406[9 * (unsigned int)v407];
                        if ( (const void **)v328 != v406 )
                        {
                          do
                          {
                            v77 = *(_QWORD *)(v328 - 40);
                            v331 = *(_QWORD *)(v328 - 32);
                            v328 -= 72LL;
                            v332 = v77;
                            if ( v331 != v77 )
                            {
                              do
                              {
                                if ( *(_DWORD *)(v332 + 88) > 0x40u )
                                {
                                  v333 = *(_QWORD *)(v332 + 80);
                                  if ( v333 )
                                    j_j___libc_free_0_0(v333);
                                }
                                v334 = (unsigned __int64 *)v332;
                                v332 += 112;
                                sub_2767770(v334);
                              }
                              while ( v331 != v332 );
                              v77 = *(_QWORD *)(v328 + 32);
                            }
                            if ( v77 )
                            {
                              v145 = *(_QWORD *)(v328 + 48) - v77;
                              j_j___libc_free_0(v77);
                            }
                          }
                          while ( v366 != (const void **)v328 );
                          v312 = &v380;
                          v328 = (unsigned __int64)v406;
                        }
                      }
                      v335 = (int)v455;
                      if ( (_BYTE *)v328 != v408 )
                      {
                        v77 = v328;
                        _libc_free(v328);
                      }
                      v311 = (unsigned int)v407;
                      v406 = (const void **)v313;
                      HIDWORD(v407) = v335;
                      v315 = v407;
                      if ( v337 )
                      {
                        v145 = 9 * v326;
                        v312 = (const void **)(v313 + 72 * v326);
                      }
                    }
                    v144 = 9 * v311;
                    v316 = v313 + 8 * v144;
                    if ( v316 )
                    {
                      *(_DWORD *)v316 = *(_DWORD *)v312;
                      *(_QWORD *)(v316 + 8) = v312[1];
                      *(_QWORD *)(v316 + 16) = v312[2];
                      *(_QWORD *)(v316 + 24) = v312[3];
                      v317 = (_BYTE *)v312[5] - (_BYTE *)v312[4];
                      *(_QWORD *)(v316 + 32) = 0;
                      *(_QWORD *)(v316 + 40) = 0;
                      *(_QWORD *)(v316 + 48) = 0;
                      if ( v317 )
                      {
                        if ( v317 > 0x7FFFFFFFFFFFFFC0LL )
LABEL_529:
                          sub_4261EA(v77, v145, v144);
                        v339 = sub_22077B0(v317);
                      }
                      else
                      {
                        v317 = 0;
                      }
                      v322 = v339;
                      *(_QWORD *)(v316 + 32) = v339;
                      *(_QWORD *)(v316 + 40) = v339;
                      *(_QWORD *)(v316 + 48) = v339 + v317;
                      v323 = (char *)v312[5];
                      v324 = (char *)v312[4];
                      if ( v323 != v324 )
                      {
                        do
                        {
                          if ( v322 )
                          {
                            sub_276A0C0((__int64 *)v322, v324);
                            v325 = *((_DWORD *)v324 + 22);
                            *(_DWORD *)(v322 + 88) = v325;
                            if ( v325 <= 0x40 )
                              *(_QWORD *)(v322 + 80) = *((_QWORD *)v324 + 10);
                            else
                              sub_C43780(v322 + 80, (const void **)v324 + 10);
                            *(_QWORD *)(v322 + 96) = *((_QWORD *)v324 + 12);
                            *(_BYTE *)(v322 + 104) = v324[104];
                          }
                          v324 += 112;
                          v322 += 112;
                        }
                        while ( v323 != v324 );
                        v339 = v322;
                      }
                      *(_QWORD *)(v316 + 40) = v339;
                      *(_QWORD *)(v316 + 56) = v312[7];
                      *(_QWORD *)(v316 + 64) = v312[8];
                      v315 = v407;
                    }
                    v318 = v385;
                    v319 = v384;
                    LODWORD(v407) = v315 + 1;
                    if ( v385 != (__int64 *)v384 )
                    {
                      do
                      {
                        if ( *(_DWORD *)(v319 + 88) > 0x40u )
                        {
                          v320 = *(_QWORD *)(v319 + 80);
                          if ( v320 )
                            j_j___libc_free_0_0(v320);
                        }
                        v321 = (unsigned __int64 *)v319;
                        v319 += 112LL;
                        sub_2767770(v321);
                      }
                      while ( v318 != (__int64 *)v319 );
                      v319 = v384;
                    }
                    if ( v319 )
                      j_j___libc_free_0(v319);
                    if ( v400 != v402 )
                      _libc_free((unsigned __int64)v400);
                    goto LABEL_108;
                  }
                }
                else
                {
                  ++v389;
                }
                break;
              }
              sub_2769990((__int64)&v389, 2 * v29);
              if ( !(_DWORD)v392 )
              {
LABEL_571:
                LODWORD(v391) = (_DWORD)v391 + 1;
                BUG();
              }
              v33 = (unsigned int)((_DWORD)v392 - 1);
              v28 = (unsigned int)v33 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
              v36 = (unsigned int)((_DWORD)v391 + 1);
              v37 = (__int64 *)&v390[16 * v28];
              v12 = *v37;
              if ( v32 != *v37 )
              {
                v138 = 0;
                v139 = 1;
                while ( v12 != -4096 )
                {
                  if ( !v138 && v12 == -8192 )
                    v138 = v37;
                  v28 = (unsigned int)v33 & (v139 + (_DWORD)v28);
                  v37 = (__int64 *)&v390[16 * (unsigned int)v28];
                  v12 = *v37;
                  if ( v32 == *v37 )
                    goto LABEL_192;
                  ++v139;
                }
                if ( v138 )
                  v37 = v138;
              }
              goto LABEL_192;
            }
            v140 = 1;
            while ( v19 != -4096 )
            {
              v12 = (unsigned int)(v140 + 1);
              v17 = v16 & (v140 + v17);
              v18 = (__int64 *)(v15 + 16LL * v17);
              v19 = *v18;
              if ( v348 - 24 == *v18 )
                goto LABEL_22;
              v140 = v12;
            }
          }
          v20 = 0;
          goto LABEL_23;
        }
        v118 = *v5;
        v119 = sub_B2BE50(*v5);
        if ( sub_B6EA50(v119)
          || (v299 = sub_B2BE50(v118),
              v300 = sub_B6F970(v299),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v300 + 48LL))(v300)) )
        {
          sub_B176B0((__int64)&v455, (__int64)"dfa-jump-threading", (__int64)"SwitchNotPredictable", 20, v4);
          sub_B18290((__int64)&v455, "Switch instruction is not predictable.", 0x26u);
          v123 = _mm_loadu_si128(&v458);
          v124 = _mm_loadu_si128(&v460);
          v437 = (unsigned __int64 *)v439;
          LODWORD(v431) = v456;
          v125 = _mm_loadu_si128(&v461);
          v430 = (__int64 *)&unk_49D9D40;
          BYTE4(v431) = BYTE4(v456);
          v433 = v123;
          v432 = v457;
          v435 = v124;
          v434 = v459;
          v438 = 0x400000000LL;
          v436 = v125;
          if ( v463 )
          {
            sub_2768F90((__int64)&v437, (__int64)&v462, v120, v121, v122, v463);
            v455 = (__int64 *)&unk_49D9D40;
            v297 = v462;
            v440 = v466;
            v441 = v467;
            v442 = v468;
            v430 = (__int64 *)&unk_49D9DB0;
            v126 = &v462[10 * v463];
            if ( v462 != v126 )
            {
              do
              {
                v126 -= 10;
                v298 = v126[4];
                if ( (unsigned __int64 *)v298 != v126 + 6 )
                  j_j___libc_free_0(v298);
                if ( (unsigned __int64 *)*v126 != v126 + 2 )
                  j_j___libc_free_0(*v126);
              }
              while ( v297 != v126 );
              v126 = v462;
            }
          }
          else
          {
            v126 = v462;
            v440 = v466;
            v441 = v467;
            v442 = v468;
            v430 = (__int64 *)&unk_49D9DB0;
          }
          if ( v126 != (unsigned __int64 *)&v464 )
            _libc_free((unsigned __int64)v126);
          sub_1049740(v5, (__int64)&v430);
          v127 = v437;
          v430 = (__int64 *)&unk_49D9D40;
          v128 = 10LL * (unsigned int)v438;
          v129 = &v437[v128];
          if ( v437 != &v437[v128] )
          {
            do
            {
              v129 -= 10;
              v130 = v129[4];
              if ( (unsigned __int64 *)v130 != v129 + 6 )
                j_j___libc_free_0(v130);
              if ( (unsigned __int64 *)*v129 != v129 + 2 )
                j_j___libc_free_0(*v129);
            }
            while ( v127 != v129 );
            v129 = v437;
          }
          if ( v129 != (unsigned __int64 *)v439 )
            _libc_free((unsigned __int64)v129);
        }
        if ( v399 )
          goto LABEL_15;
LABEL_105:
        if ( v400 != v402 )
          _libc_free((unsigned __int64)v400);
      }
      v348 = *(_QWORD *)(v348 + 8);
    }
    while ( v342 != v348 );
  }
LABEL_108:
  sub_D50AF0(*(_QWORD *)(a1 + 24));
  v414 = 0;
  v415 = &v418;
  v416 = 32;
  LODWORD(v417) = 0;
  BYTE4(v417) = 1;
  if ( !(_DWORD)v407 )
    goto LABEL_109;
  v77 = a2;
  sub_30AB9D0(a2, *(_QWORD *)(a1 + 8), &v414);
  v144 = 9LL * (unsigned int)v407;
  v336 = &v406[9 * (unsigned int)v407];
  if ( v336 == v406 )
  {
    v163 = BYTE4(v417);
    goto LABEL_253;
  }
  v355 = v406;
  do
  {
    v145 = (__int64)v355;
    LODWORD(v389) = *(_DWORD *)v355;
    v390 = (char *)v355[1];
    v391 = v355[2];
    v392 = v355[3];
    v146 = (_BYTE *)v355[5] - (_BYTE *)v355[4];
    v393 = 0;
    v394 = 0;
    v395 = 0;
    if ( v146 )
    {
      if ( v146 > 0x7FFFFFFFFFFFFFC0LL )
        goto LABEL_529;
      v147 = sub_22077B0(v146);
    }
    else
    {
      v146 = 0;
      v147 = 0;
    }
    v393 = v147;
    v394 = v147;
    v395 = v147 + v146;
    v148 = (char *)v355[5];
    for ( j = (char *)v355[4]; v148 != j; v147 += 112 )
    {
      if ( v147 )
      {
        sub_276A0C0((__int64 *)v147, j);
        v150 = *((_DWORD *)j + 22);
        *(_DWORD *)(v147 + 88) = v150;
        if ( v150 <= 0x40 )
          *(_QWORD *)(v147 + 80) = *((_QWORD *)j + 10);
        else
          sub_C43780(v147 + 80, (const void **)j + 10);
        *(_QWORD *)(v147 + 96) = *((_QWORD *)j + 12);
        *(_BYTE *)(v147 + 104) = j[104];
      }
      j += 112;
    }
    v394 = v147;
    v396 = v355[7];
    v397 = v355[8];
    sub_C8CD80((__int64)&v455, (__int64)&v458.m128i_i64[1], (__int64)&v414, v141, v142, v143);
    v419 = &v389;
    v151 = *(_QWORD *)(a1 + 40);
    v152 = *(_QWORD *)(a1 + 8);
    v153 = *(_QWORD *)(a1 + 16);
    v421 = *(_QWORD *)(a1 + 32);
    v420.m128i_i64[0] = v153;
    v420.m128i_i64[1] = v152;
    v422 = v151;
    sub_C8CD80((__int64)v423, (__int64)v426, (__int64)&v455, v152, v154, v155);
    v427 = 0;
    v428 = 0;
    v429 = 0;
    if ( !v458.m128i_i8[4] )
      _libc_free(v456);
    v77 = (__int64)&v419;
    v156 = sub_276AF50((__int64 *)&v419);
    if ( !v156 )
      goto LABEL_233;
    v469 = 0;
    v455 = &v457;
    v456 = 0x1000000000LL;
    v470 = 0;
    v471 = v420.m128i_i64[0];
    v475 = &v479;
    v480 = 0;
    v472 = 0;
    v473 = 0;
    v474 = 0;
    v476 = 8;
    v477 = 0;
    v478 = 1;
    v481 = 0;
    v482 = 0;
    v483 = 0;
    v164 = v419[4];
    v165 = v419[5];
    v166 = v419[2];
    while ( v165 != v164 )
    {
      while ( 1 )
      {
        v430 = (__int64 *)v166;
        v167 = *(_QWORD *)(v164 + 16);
        if ( v167 == *(_QWORD *)(v164 + 24) )
          break;
        *(_QWORD *)(v167 - 8) = v166;
        *(_QWORD *)(v164 + 16) -= 8LL;
        v164 += 112;
        if ( v165 == v164 )
          goto LABEL_261;
      }
      v77 = v164;
      v164 += 112;
      sub_2769700((unsigned __int64 *)v77, &v430);
    }
LABEL_261:
    v168 = (_QWORD *)(v166 + 48);
    v370 = 0;
    v378 = &v380;
    v371 = 0;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    v375 = 0;
    v376 = 0;
    v377 = 0;
    v379 = 0;
    v409 = 0;
    v410 = (__int64 *)&v413;
    v411 = 16;
    LODWORD(v412) = 0;
    BYTE4(v412) = 1;
    v169 = *v168 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v169 != v168 )
    {
      if ( !v169 )
        goto LABEL_330;
      v170 = v169 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v169 - 24) - 30 <= 0xA )
      {
        v77 = v169 - 24;
        v171 = sub_B46E30(v170);
        if ( v171 )
        {
          v172 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v175 = sub_B46EC0(v170, v172);
              if ( v156 )
                break;
LABEL_325:
              v77 = (__int64)&v409;
              ++v172;
              sub_C8CC70((__int64)&v409, v175, (__int64)v173, v174, v142, v176);
              v156 = BYTE4(v412);
              if ( v171 == v172 )
                goto LABEL_272;
            }
            v177 = v410;
            v77 = HIDWORD(v411);
            v173 = &v410[HIDWORD(v411)];
            if ( v410 == v173 )
            {
LABEL_327:
              if ( HIDWORD(v411) >= (unsigned int)v411 )
                goto LABEL_325;
              v77 = (unsigned int)(HIDWORD(v411) + 1);
              ++v172;
              ++HIDWORD(v411);
              *v173 = v175;
              v156 = BYTE4(v412);
              ++v409;
              if ( v171 == v172 )
                break;
            }
            else
            {
              while ( v175 != *v177 )
              {
                if ( v173 == ++v177 )
                  goto LABEL_327;
              }
              if ( v171 == ++v172 )
                break;
            }
          }
        }
      }
    }
LABEL_272:
    if ( v419[4] != v419[5] )
    {
      v360 = v419[5];
      v178 = v419[4];
      do
      {
        v179 = v178;
        v77 = (__int64)&v419;
        v178 += 112;
        sub_276C1A0((__int64 *)&v419, (__int64)&v374, v179, (__int64)&v370, (__int64)&v409, (__int64)&v455);
      }
      while ( v360 != v178 );
      v180 = v419[4];
      v343 = (const void **)v419[5];
      if ( (const void **)v180 != v343 )
      {
        v181 = (const void **)(v180 + 80);
        LODWORD(v381) = *(_DWORD *)(v180 + 88);
        if ( (unsigned int)v381 <= 0x40 )
        {
LABEL_277:
          v380 = *v181;
          v182 = (__int64)*(v181 - 4);
          if ( (const void *)v182 == *(v181 - 3) )
            goto LABEL_316;
          goto LABEL_278;
        }
        while ( 1 )
        {
          sub_C43780((__int64)&v380, v181);
          v182 = (__int64)*(v181 - 4);
          if ( (const void *)v182 == *(v181 - 3) )
LABEL_316:
            v182 = *((_QWORD *)*(v181 - 1) - 1) + 512LL;
LABEL_278:
          v77 = *(_QWORD *)(v182 - 8);
          v361 = sub_2766DC0(v77, (__int64)&v380, (__int64)&v370);
          v183 = *(_QWORD *)(v361 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          v358 = v183;
          if ( v361 + 48 == v183 || !v183 || (unsigned int)*(unsigned __int8 *)(v183 - 24) - 30 > 0xA )
            goto LABEL_570;
          if ( *(_BYTE *)(v183 - 24) == 32 )
          {
            v350 = *(_DWORD *)(v183 - 20);
            v184 = ((v350 & 0x7FFFFFFu) >> 1) - 1;
            v185 = *(_QWORD *)(v183 - 32);
            if ( (v350 & 0x7FFFFFFu) >> 1 != 1 )
            {
              v340 = v181;
              v186 = 0;
              v351 = v380;
              while ( 1 )
              {
                v187 = v186++;
                v188 = *(_QWORD *)(v185 + 32LL * (unsigned int)(2 * v186));
                if ( *(_DWORD *)(v188 + 32) > 0x40u )
                {
                  if ( sub_C43C50(v188 + 24, &v380) )
                    goto LABEL_288;
                }
                else if ( *(const void **)(v188 + 24) == v351 )
                {
LABEL_288:
                  v181 = v340;
                  v189 = 32;
                  if ( v187 != -2 )
                    v189 = 32LL * (unsigned int)(2 * v187 + 3);
                  v190 = *(_QWORD *)(v185 + v189);
                  if ( v190 )
                    goto LABEL_291;
                  break;
                }
                if ( v184 == v186 )
                {
                  v181 = v340;
                  break;
                }
              }
            }
            v190 = *(_QWORD *)(v185 + 32);
LABEL_291:
            v403 = 0;
            v404 = 0;
            v431 = (unsigned __int64)&v433.m128i_u64[1];
            v405[0] = 0;
            v430 = 0;
            v432 = 4;
            v433.m128i_i32[0] = 0;
            v433.m128i_i8[4] = 1;
            v191 = *(_QWORD *)(v361 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v361 + 48 != v191 )
            {
              if ( !v191 )
LABEL_330:
                BUG();
              v192 = v191 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v191 - 24) - 30 <= 0xA )
              {
                v193 = sub_B46E30(v192);
                if ( v193 )
                {
                  v352 = v181;
                  v194 = 0;
                  while ( 1 )
                  {
                    v199 = sub_B46EC0(v192, v194);
                    if ( v199 == v190 )
                      goto LABEL_302;
                    if ( v433.m128i_i8[4] )
                    {
                      v200 = (_QWORD *)v431;
                      v195 = (__int64 *)(v431 + 8LL * HIDWORD(v432));
                      if ( (__int64 *)v431 != v195 )
                      {
                        while ( v199 != *v200 )
                        {
                          if ( v195 == ++v200 )
                            goto LABEL_323;
                        }
                        goto LABEL_302;
                      }
LABEL_323:
                      if ( HIDWORD(v432) < (unsigned int)v432 )
                      {
                        ++HIDWORD(v432);
                        *v195 = v199;
                        v430 = (__int64 *)((char *)v430 + 1);
                        goto LABEL_318;
                      }
                    }
                    sub_C8CC70((__int64)&v430, v199, (__int64)v195, v196, v197, v198);
                    if ( !v208 )
                    {
LABEL_302:
                      if ( v193 == ++v194 )
                        goto LABEL_303;
                    }
                    else
                    {
LABEL_318:
                      v209 = v404;
                      v398.m128i_i64[1] = v199 | 4;
                      v398.m128i_i64[0] = v361;
                      if ( v404 == (__m128i *)v405[0] )
                      {
                        ++v194;
                        sub_F38BA0((const __m128i **)&v403, v404, &v398);
                        if ( v193 == v194 )
                          goto LABEL_303;
                      }
                      else
                      {
                        if ( v404 )
                        {
                          *v404 = _mm_loadu_si128(&v398);
                          v209 = v404;
                        }
                        ++v194;
                        v404 = (__m128i *)&v209[1];
                        if ( v193 == v194 )
                        {
LABEL_303:
                          v181 = v352;
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
            sub_B43D60((_QWORD *)(v358 - 24));
            sub_B43C20((__int64)&v398, v361);
            v201 = v398.m128i_i64[0];
            v202 = v398.m128i_u16[4];
            v204 = sub_BD2C40(72, 1u);
            if ( v204 )
              sub_B4C8F0((__int64)v204, v190, 1u, v201, v202);
            sub_FFB3D0((__int64)&v455, (unsigned __int64 *)v403, v404 - v403, v203, v205, v206);
            if ( !v433.m128i_i8[4] )
              _libc_free(v431);
            v77 = (__int64)v403;
            if ( v403 )
              j_j___libc_free_0((unsigned __int64)v403);
          }
          if ( (unsigned int)v381 > 0x40 )
          {
            v77 = (__int64)v380;
            if ( v380 )
              j_j___libc_free_0_0((unsigned __int64)v380);
          }
          v207 = v181 + 14;
          if ( v343 == v181 + 4 )
            break;
          v181 += 14;
          LODWORD(v381) = *((_DWORD *)v207 + 2);
          if ( (unsigned int)v381 <= 0x40 )
            goto LABEL_277;
        }
      }
    }
    v443 = 0;
    v430 = &v432;
    v144 = 32LL * (unsigned int)v379;
    v431 = 0x400000000LL;
    v450 = 0x400000000LL;
    v452 = v454;
    v403 = (__m128i *)v405;
    v404 = (__m128i *)0x1000000000LL;
    v444 = 0;
    v445 = 0;
    v446 = 0;
    v447 = 0;
    v448 = 0;
    v449 = (__int64 *)v451;
    v453 = 0;
    v454[0] = 0;
    v454[1] = 1;
    v353 = (__int64 *)((char *)v378 + v144);
    if ( v378 != (const void **)((char *)v378 + v144) )
    {
      for ( k = (__int64 *)v378; v353 != k; k += 4 )
      {
        v211 = k[2];
        v145 = k[1];
        v212 = *k;
        v213 = v211 - v145;
        v214 = *(_QWORD *)(*k + 40);
        if ( v211 == v145 )
        {
          v217 = 0;
          v216 = 0;
        }
        else
        {
          v362 = *k;
          if ( v213 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_529;
          v77 = k[2] - v145;
          v215 = sub_22077B0(v213);
          v145 = k[1];
          v212 = v362;
          v216 = (__int64 *)v215;
          v211 = k[2];
          v217 = v211 - v145;
        }
        if ( v145 != v211 )
        {
          v77 = (__int64)v216;
          v363 = v212;
          memmove(v216, (const void *)v145, v217);
          v212 = v363;
        }
        v218 = *(_QWORD *)(v212 + 16);
        v219 = (unsigned int)v404;
        if ( v218 )
        {
          v364 = v216;
          v220 = v212;
          while ( 1 )
          {
            v144 = *(_QWORD *)(v218 + 24);
            if ( *(_BYTE *)v144 != 84 )
              break;
            v221 = *(_QWORD *)(v144 - 8);
            v144 = 32LL * *(unsigned int *)(v144 + 72) + 8LL * (unsigned int)((v218 - v221) >> 5);
            if ( v214 != *(_QWORD *)(v221 + v144) )
            {
              v222 = v219 + 1;
              if ( v219 + 1 <= (unsigned __int64)HIDWORD(v404) )
              {
LABEL_346:
                v144 = (__int64)v403;
                v403->m128i_i64[v219] = v218;
                v219 = (unsigned int)((_DWORD)v404 + 1);
                LODWORD(v404) = (_DWORD)v404 + 1;
                goto LABEL_347;
              }
LABEL_351:
              v77 = (__int64)&v403;
              sub_C8D5F0((__int64)&v403, v405, v222, 8u, v142, v212);
              v219 = (unsigned int)v404;
              goto LABEL_346;
            }
LABEL_347:
            v218 = *(_QWORD *)(v218 + 8);
            if ( !v218 )
            {
              v212 = v220;
              v216 = v364;
              goto LABEL_353;
            }
          }
          if ( v214 == *(_QWORD *)(v144 + 40) )
            goto LABEL_347;
          v222 = v219 + 1;
          if ( v219 + 1 <= (unsigned __int64)HIDWORD(v404) )
            goto LABEL_346;
          goto LABEL_351;
        }
LABEL_353:
        if ( (_DWORD)v219 )
        {
          v260 = *(_QWORD *)(v212 + 8);
          v344 = v212;
          v365 = (__int64 *)((char *)v216 + v217);
          v261 = sub_BD5D20(v212);
          v77 = (__int64)&v430;
          v263 = (unsigned int)sub_2A58E70(&v430, v261, v262, v260);
          sub_2A59200(&v430, v263, v214, v344);
          if ( (__int64 *)((char *)v216 + v217) != v216 )
          {
            v345 = v216;
            do
            {
              v264 = *v216;
              v77 = (__int64)&v430;
              ++v216;
              sub_2A59200(&v430, (unsigned int)v263, *(_QWORD *)(v264 + 40), v264);
            }
            while ( v365 != v216 );
            v265 = (unsigned int)v404;
            v216 = v345;
            if ( !(_DWORD)v404 )
              goto LABEL_354;
            goto LABEL_424;
          }
          while ( 1 )
          {
            v265 = (unsigned int)v404;
            if ( !(_DWORD)v404 )
              break;
LABEL_424:
            v77 = (__int64)&v430;
            v266 = v403->m128i_i64[v265 - 1];
            LODWORD(v404) = v265 - 1;
            sub_2A58C00(&v430, (unsigned int)v263, v266);
          }
        }
LABEL_354:
        if ( v216 )
        {
          v77 = (__int64)v216;
          j_j___libc_free_0((unsigned __int64)v216);
        }
      }
    }
    sub_2A59A40(&v430, v420.m128i_i64[0], 0);
    if ( v403 != (__m128i *)v405 )
      _libc_free((unsigned __int64)v403);
    v223 = v449;
    v224 = &v449[(unsigned int)v450];
    if ( v449 != v224 )
    {
      for ( m = (unsigned __int64)v449; ; m = (unsigned __int64)v449 )
      {
        v226 = *v223;
        v227 = (unsigned int)((__int64)((__int64)v223 - m) >> 3) >> 7;
        v228 = 4096LL << v227;
        if ( v227 >= 0x1E )
          v228 = 0x40000000000LL;
        ++v223;
        sub_C7D6A0(v226, v228, 16);
        if ( v224 == v223 )
          break;
      }
    }
    v229 = v452;
    v230 = &v452[2 * (unsigned int)v453];
    if ( v452 != v230 )
    {
      do
      {
        v231 = v229[1];
        v232 = *v229;
        v229 += 2;
        sub_C7D6A0(v232, v231, 16);
      }
      while ( v230 != v229 );
      v230 = v452;
    }
    if ( v230 != v454 )
      _libc_free((unsigned __int64)v230);
    if ( v449 != (__int64 *)v451 )
      _libc_free((unsigned __int64)v449);
    sub_C7D6A0(v444, 24LL * v446, 8);
    v233 = (unsigned __int64)v430;
    v234 = (unsigned __int64)&v430[13 * (unsigned int)v431];
    if ( v430 != (__int64 *)v234 )
    {
      do
      {
        v234 -= 104LL;
        v235 = *(_QWORD *)(v234 + 32);
        if ( v235 != v234 + 48 )
          _libc_free(v235);
        sub_C7D6A0(*(_QWORD *)(v234 + 8), 16LL * *(unsigned int *)(v234 + 24), 8);
      }
      while ( v233 != v234 );
      v234 = (unsigned __int64)v430;
    }
    if ( (__int64 *)v234 != &v432 )
      _libc_free(v234);
    v236 = v410;
    if ( BYTE4(v412) )
    {
      v359 = &v410[HIDWORD(v411)];
      if ( v410 == v359 )
        goto LABEL_384;
    }
    else
    {
      v359 = &v410[(unsigned int)v411];
      if ( v410 == v359 )
      {
LABEL_475:
        _libc_free((unsigned __int64)v410);
        goto LABEL_384;
      }
    }
    while ( 1 )
    {
      v237 = *v236;
      if ( (unsigned __int64)*v236 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v359 == ++v236 )
        goto LABEL_383;
    }
    v354 = v236;
    if ( v359 == v236 )
    {
LABEL_383:
      if ( !BYTE4(v412) )
        goto LABEL_475;
      goto LABEL_384;
    }
    do
    {
      v267 = *(_QWORD *)(v237 + 16);
      if ( v267 )
      {
        while ( (unsigned __int8)(**(_BYTE **)(v267 + 24) - 30) > 0xAu )
        {
          v267 = *(_QWORD *)(v267 + 8);
          if ( !v267 )
            goto LABEL_457;
        }
        v268 = *(_QWORD *)(v237 + 56);
LABEL_431:
        if ( !v268 )
          goto LABEL_570;
        if ( *(_BYTE *)(v268 - 24) != 84 )
          goto LABEL_471;
        v430 = 0;
        v431 = 0;
        v432 = 0;
        v269 = 32LL * *(unsigned int *)(v268 + 48);
        v270 = v269 + 8LL * (*(_DWORD *)(v268 - 20) & 0x7FFFFFF);
        v271 = (__int64 *)(*(_QWORD *)(v268 - 32) + v269);
        v272 = (__int64 *)(*(_QWORD *)(v268 - 32) + v270);
        if ( v272 == v271 )
          goto LABEL_455;
        v273 = v271;
        while ( 1 )
        {
          v398.m128i_i64[0] = *v273;
          v403 = (__m128i *)v398.m128i_i64[0];
          v276 = *(_QWORD *)(v237 + 16);
          if ( v276 )
          {
            while ( (unsigned __int8)(**(_BYTE **)(v276 + 24) - 30) > 0xAu )
            {
              v276 = *(_QWORD *)(v276 + 8);
              if ( !v276 )
              {
                if ( sub_2765A40(0, 0, (__int64 *)&v403) )
                  goto LABEL_436;
                goto LABEL_441;
              }
            }
          }
          if ( sub_2765A40(v276, 0, (__int64 *)&v403) )
            goto LABEL_436;
LABEL_441:
          if ( (__int64 *)v432 == v275 )
          {
            sub_9319A0((__int64)&v430, v275, &v398);
            v275 = (__int64 *)v431;
LABEL_436:
            if ( v272 == ++v273 )
              goto LABEL_445;
          }
          else
          {
            if ( v275 )
            {
              *v275 = v274;
              v275 = (__int64 *)v431;
            }
            ++v275;
            ++v273;
            v431 = (unsigned __int64)v275;
            if ( v272 == v273 )
            {
LABEL_445:
              v277 = (unsigned __int64)v275;
              if ( v430 != v275 )
              {
                v278 = v430;
                do
                {
                  if ( (*(_DWORD *)(v268 - 20) & 0x7FFFFFF) != 0 )
                  {
                    v279 = 0;
                    while ( 1 )
                    {
                      v280 = v279;
                      if ( *v278 == *(_QWORD *)(*(_QWORD *)(v268 - 32) + 32LL * *(unsigned int *)(v268 + 48) + 8 * v279) )
                        break;
                      if ( (*(_DWORD *)(v268 - 20) & 0x7FFFFFF) == (_DWORD)++v279 )
                        goto LABEL_476;
                    }
                  }
                  else
                  {
LABEL_476:
                    v280 = -1;
                  }
                  ++v278;
                  sub_B48BF0(v268 - 24, v280, 1);
                }
                while ( (__int64 *)v277 != v278 );
                v277 = (unsigned __int64)v430;
              }
              if ( v277 )
                j_j___libc_free_0(v277);
LABEL_455:
              v268 = *(_QWORD *)(v268 + 8);
              goto LABEL_431;
            }
          }
        }
      }
LABEL_457:
      v430 = 0;
      v431 = 0;
      v432 = 0;
      v281 = *(_QWORD *)(v237 + 56);
      if ( !v281 )
        goto LABEL_330;
      v282 = 0;
      while ( *(_BYTE *)(v281 - 24) == 84 )
      {
        v403 = (__m128i *)(v281 - 24);
        if ( v282 == (__int64 *)v432 )
        {
          sub_249A9D0((__int64)&v430, v282, &v403);
        }
        else
        {
          if ( v282 )
          {
            *v282 = v281 - 24;
            v282 = (__int64 *)v431;
          }
          v431 = (unsigned __int64)(v282 + 1);
        }
        v281 = *(_QWORD *)(v281 + 8);
        if ( !v281 )
          goto LABEL_330;
        v282 = (__int64 *)v431;
      }
      v283 = (unsigned __int64)v430;
      v284 = v430;
      if ( v282 != v430 )
      {
        do
        {
          v285 = *v284++;
          v286 = sub_ACADE0(*(__int64 ***)(v285 + 8));
          sub_BD84D0(v285, v286);
          sub_B43D60((_QWORD *)v285);
        }
        while ( v282 != v284 );
        v283 = (unsigned __int64)v430;
      }
      if ( v283 )
        j_j___libc_free_0(v283);
LABEL_471:
      v287 = v354 + 1;
      if ( v354 + 1 == v359 )
        break;
      while ( 1 )
      {
        v237 = *v287;
        if ( (unsigned __int64)*v287 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v359 == ++v287 )
          goto LABEL_474;
      }
      v354 = v287;
    }
    while ( v359 != v287 );
LABEL_474:
    if ( !BYTE4(v412) )
      goto LABEL_475;
LABEL_384:
    v238 = v378;
    v239 = &v378[4 * (unsigned int)v379];
    if ( v378 != v239 )
    {
      do
      {
        v240 = (unsigned __int64)*(v239 - 3);
        v239 -= 4;
        if ( v240 )
          j_j___libc_free_0(v240);
      }
      while ( v238 != v239 );
      v239 = v378;
    }
    if ( v239 != &v380 )
      _libc_free((unsigned __int64)v239);
    sub_C7D6A0((__int64)v375, 16LL * v377, 8);
    v241 = v373;
    if ( v373 )
    {
      v242 = (_QWORD *)v371;
      v243 = (_QWORD *)(v371 + 32LL * v373);
      do
      {
        if ( *v242 != -8192 && *v242 != -4096 )
        {
          v244 = v242[2];
          v245 = v242[1];
          if ( v244 != v245 )
          {
            do
            {
              if ( *(_DWORD *)(v245 + 16) > 0x40u )
              {
                v246 = *(_QWORD *)(v245 + 8);
                if ( v246 )
                  j_j___libc_free_0_0(v246);
              }
              v245 += 24LL;
            }
            while ( v244 != v245 );
            v245 = v242[1];
          }
          if ( v245 )
            j_j___libc_free_0(v245);
        }
        v242 += 4;
      }
      while ( v243 != v242 );
      v241 = v373;
    }
    v247 = 32 * v241;
    sub_C7D6A0(v371, 32 * v241, 8);
    sub_FFCE90((__int64)&v455, v247, v248, v249, v250, v251);
    sub_FFD870((__int64)&v455, v247, v252, v253, v254, v255);
    sub_FFBC40((__int64)&v455, v247);
    v256 = v482;
    v257 = v481;
    if ( v482 != v481 )
    {
      do
      {
        v258 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v257[7];
        *v257 = &unk_49E5048;
        if ( v258 )
          v258(v257 + 5, v257 + 5, 3);
        *v257 = &unk_49DB368;
        v259 = v257[3];
        LOBYTE(v141) = v259 != -4096;
        LOBYTE(v144) = v259 != 0;
        if ( ((v259 != 0) & (unsigned __int8)v141) != 0 && v259 != -8192 )
          sub_BD60C0(v257 + 1);
        v257 += 9;
      }
      while ( v256 != v257 );
      v257 = v481;
    }
    if ( v257 )
      j_j___libc_free_0((unsigned __int64)v257);
    if ( !v478 )
      _libc_free((unsigned __int64)v475);
    v77 = (__int64)v455;
    if ( v455 != &v457 )
      _libc_free((unsigned __int64)v455);
LABEL_233:
    v157 = v428;
    v158 = v427;
    *(_BYTE *)a1 = 1;
    if ( v157 != v158 )
    {
      do
      {
        if ( *(_DWORD *)(v158 + 88) > 0x40u )
        {
          v159 = *(_QWORD *)(v158 + 80);
          if ( v159 )
            j_j___libc_free_0_0(v159);
        }
        v77 = v158;
        v158 += 112LL;
        sub_2767770((unsigned __int64 *)v77);
      }
      while ( v157 != v158 );
      v158 = v427;
    }
    if ( v158 )
    {
      v77 = v158;
      j_j___libc_free_0(v158);
    }
    if ( !v425 )
    {
      v77 = v424;
      _libc_free(v424);
    }
    v160 = v394;
    v161 = v393;
    if ( v394 != v393 )
    {
      do
      {
        if ( *(_DWORD *)(v161 + 88) > 0x40u )
        {
          v162 = *(_QWORD *)(v161 + 80);
          if ( v162 )
            j_j___libc_free_0_0(v162);
        }
        v77 = v161;
        v161 += 112LL;
        sub_2767770((unsigned __int64 *)v77);
      }
      while ( v160 != v161 );
      v161 = v393;
    }
    if ( v161 )
    {
      v77 = v161;
      j_j___libc_free_0(v161);
    }
    v355 += 9;
  }
  while ( v336 != v355 );
  v341 = 1;
  v163 = BYTE4(v417);
LABEL_253:
  if ( !v163 )
    _libc_free((unsigned __int64)v415);
LABEL_109:
  v82 = v406;
  v83 = (unsigned __int64)&v406[9 * (unsigned int)v407];
  if ( v406 != (const void **)v83 )
  {
    do
    {
      v84 = *(_QWORD *)(v83 - 40);
      v85 = *(_QWORD *)(v83 - 32);
      v83 -= 72LL;
      v86 = v84;
      if ( v85 != v84 )
      {
        do
        {
          if ( *(_DWORD *)(v86 + 88) > 0x40u )
          {
            v87 = *(_QWORD *)(v86 + 80);
            if ( v87 )
              j_j___libc_free_0_0(v87);
          }
          v88 = (unsigned __int64 *)v86;
          v86 += 112LL;
          sub_2767770(v88);
        }
        while ( v85 != v86 );
        v84 = *(_QWORD *)(v83 + 32);
      }
      if ( v84 )
        j_j___libc_free_0(v84);
    }
    while ( v82 != (const void **)v83 );
    v83 = (unsigned __int64)v406;
  }
  if ( (_BYTE *)v83 != v408 )
    _libc_free(v83);
  return v341;
}
