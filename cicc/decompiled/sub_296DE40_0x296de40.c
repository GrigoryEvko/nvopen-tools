// Function: sub_296DE40
// Address: 0x296de40
//
__int64 __fastcall sub_296DE40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8,
        __int64 a9)
{
  __int64 v9; // rcx
  __int64 *v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rsi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // r12
  _BYTE *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  _DWORD *v26; // r13
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // r13
  unsigned int v31; // r14d
  __int64 v32; // rbx
  _BYTE *v33; // r13
  __int64 v34; // rax
  unsigned __int64 *v35; // rax
  unsigned __int64 v36; // r12
  int v38; // eax
  unsigned __int64 v39; // rsi
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned __int64 v45; // rbx
  __int64 v46; // rbx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // r8
  unsigned __int64 v54; // rax
  __int64 v55; // r15
  __int64 v56; // rax
  __int64 v57; // r12
  unsigned int **v58; // r15
  unsigned int *v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  _QWORD *v69; // r15
  _QWORD *v70; // r12
  void (__fastcall *v71)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v72; // rax
  char v73; // r13
  unsigned __int64 v74; // rcx
  __int64 *v75; // rdx
  __int64 v76; // rax
  __int64 *v77; // rbx
  __int64 v78; // rax
  __int64 *v79; // r12
  signed __int64 v80; // rax
  __int64 *v81; // rsi
  _QWORD *v82; // rdi
  unsigned __int64 v83; // rdx
  unsigned __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rdx
  unsigned int v87; // eax
  __int64 v88; // rax
  __int64 v89; // rdx
  unsigned __int64 v90; // rcx
  __int64 v91; // rax
  __int64 v92; // rbx
  __int64 v93; // r14
  unsigned int v94; // r12d
  int v95; // esi
  __int64 v96; // r8
  __int64 v97; // rdi
  int v98; // esi
  __int64 *v99; // rax
  __int64 v100; // r10
  unsigned __int64 v101; // rax
  _BYTE *v102; // rdi
  __int64 v103; // r15
  unsigned __int64 v104; // rax
  __int64 v105; // rsi
  char v106; // al
  __int64 v107; // rcx
  __int64 v108; // rdx
  __int64 v109; // rdx
  __int64 v110; // rcx
  char v111; // al
  _BYTE *j; // rax
  const char *v113; // r15
  const __m128i *p_src; // rdx
  __int64 v115; // rax
  char *v116; // r12
  unsigned __int64 v117; // r9
  __int64 v118; // rax
  __m128i *v119; // rax
  __int64 v120; // rdx
  int v121; // ebx
  __int64 v122; // r14
  _QWORD *v123; // r13
  __int64 v124; // r12
  char *v125; // r9
  __int64 *v126; // rbx
  unsigned __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // r9
  unsigned __int64 v130; // rbx
  const char *v131; // rdx
  __int64 v132; // rax
  char *v133; // r14
  signed __int64 v134; // r10
  __int64 v135; // r15
  unsigned __int64 v136; // rdx
  unsigned __int64 v137; // rcx
  __int64 v138; // r8
  __int64 v139; // rcx
  unsigned int v140; // eax
  __int64 v141; // rax
  __int64 v142; // r9
  __int64 v143; // rax
  unsigned int *v144; // rbx
  _QWORD *v145; // rax
  __int64 v146; // rdx
  __int64 v147; // rdx
  __int64 v148; // rcx
  __int64 v149; // r8
  __int64 v150; // r9
  unsigned __int8 *v151; // r12
  __int64 v152; // rax
  unsigned __int64 v153; // rax
  __int64 v154; // rax
  _QWORD *v155; // rbx
  __int64 v156; // r12
  __int64 v157; // rax
  unsigned __int64 *v158; // rax
  unsigned __int64 v159; // r15
  unsigned __int64 v160; // r12
  _QWORD *v161; // rax
  __int64 v162; // rdx
  __int64 v163; // rcx
  __int64 v164; // r8
  __int64 v165; // r9
  unsigned __int64 v166; // rbx
  __m128i v167; // xmm7
  __m128i v168; // xmm0
  __m128i v169; // xmm7
  __int64 v170; // rax
  __int64 v171; // r12
  __int64 v172; // rbx
  __int64 v173; // r15
  unsigned __int64 v174; // r13
  __int64 v175; // rdx
  __int64 v176; // rax
  __int64 v177; // rdx
  unsigned __int64 v178; // rax
  __int64 v179; // r9
  __int64 v180; // rbx
  __int64 v181; // rdx
  __int64 v182; // rax
  __int64 v183; // r9
  unsigned int *v184; // r12
  unsigned int *v185; // rbx
  __int64 v186; // rdx
  unsigned int v187; // esi
  __int64 v188; // rdx
  __int64 v189; // r9
  __int64 v190; // r10
  _QWORD *v191; // rax
  __int64 v192; // r12
  __int64 v193; // r9
  unsigned int *v194; // r15
  unsigned int *k; // rbx
  __int64 v196; // rdx
  unsigned int v197; // esi
  __int64 v198; // r13
  __int64 v199; // rsi
  __int64 v200; // rax
  __int64 v201; // rax
  __m128i v202; // xmm0
  unsigned __int64 v203; // r12
  __int64 v204; // rax
  __m128i v205; // xmm7
  __m128i v206; // xmm0
  __m128i v207; // xmm7
  __int64 v208; // rcx
  __int64 v209; // r8
  __int64 v210; // r9
  __int64 v211; // rsi
  unsigned __int8 *v212; // rsi
  __int64 v213; // r12
  __int64 v214; // r8
  unsigned __int64 v215; // r15
  unsigned __int8 *v216; // rdx
  unsigned __int8 *v217; // rdi
  __int64 *v218; // rax
  __int64 v219; // rax
  __int64 v220; // r9
  int v221; // ecx
  __int64 v222; // rsi
  int v223; // ecx
  unsigned int v224; // edx
  __int64 *v225; // rax
  __int64 *v226; // rsi
  __int64 v227; // rdx
  __int64 v228; // rcx
  __int64 v229; // r8
  __int64 v230; // r9
  __int64 v231; // rdx
  __int64 v232; // rcx
  __int64 v233; // r8
  __int64 v234; // r9
  _QWORD *v235; // r15
  _QWORD *v236; // r13
  void (__fastcall *v237)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v238; // rax
  _QWORD *v239; // r13
  unsigned int v240; // edx
  __int64 v241; // rsi
  _QWORD *v242; // rbx
  unsigned __int64 v243; // rdi
  unsigned __int64 v244; // rax
  int v245; // eax
  __int64 v246; // rdx
  _QWORD *v247; // r12
  _QWORD *v248; // rax
  _QWORD *v249; // r15
  unsigned __int64 v250; // rbx
  __int64 v251; // r14
  __int32 v252; // esi
  __int64 v253; // r13
  __int32 v254; // r14d
  __int64 v255; // r12
  void *v256; // rax
  __m128i v257; // xmm1
  __m128i v258; // xmm2
  __m128i v259; // xmm3
  void *v260; // rax
  __int64 v261; // rax
  unsigned __int64 *v262; // rax
  unsigned __int64 v263; // r15
  __int64 v264; // rbx
  __int64 v265; // r13
  __int64 v266; // r15
  __int64 v267; // rax
  __int64 v268; // r8
  unsigned __int64 v269; // rdx
  __int64 v270; // rcx
  unsigned __int64 v271; // r9
  void **v272; // rsi
  int v273; // edi
  unsigned __int64 v274; // rdx
  char *v275; // r15
  __int64 *v276; // rax
  __int64 v277; // rax
  unsigned int v278; // eax
  __int64 v279; // r12
  __int64 *v280; // rax
  int v281; // eax
  int v282; // edi
  unsigned int v283; // r12d
  bool v284; // al
  unsigned __int64 v285; // rdx
  void *v286; // rdi
  size_t v287; // r9
  size_t v288; // r11
  char *v289; // r15
  unsigned __int64 v290; // rdx
  unsigned __int64 v291; // rax
  unsigned __int64 v292; // rsi
  int v293; // esi
  char *v294; // r12
  int v295; // eax
  char *v296; // rax
  unsigned __int64 v297; // r8
  const char *v298; // rsi
  _BYTE *v299; // rax
  unsigned int v300; // r12d
  int v301; // eax
  unsigned int v302; // ebx
  unsigned int v303; // r12d
  unsigned __int8 *v304; // rax
  int v305; // eax
  __int64 v306; // [rsp+8h] [rbp-758h]
  __int64 v307; // [rsp+10h] [rbp-750h]
  __int64 v308; // [rsp+18h] [rbp-748h]
  unsigned __int8 v309; // [rsp+30h] [rbp-730h]
  unsigned int v310; // [rsp+30h] [rbp-730h]
  int v311; // [rsp+38h] [rbp-728h]
  unsigned __int64 v312; // [rsp+48h] [rbp-718h]
  __int64 v313; // [rsp+48h] [rbp-718h]
  __int64 v314; // [rsp+48h] [rbp-718h]
  __int64 v315; // [rsp+50h] [rbp-710h]
  __int64 v316; // [rsp+50h] [rbp-710h]
  __int64 v317; // [rsp+50h] [rbp-710h]
  __int16 v318; // [rsp+50h] [rbp-710h]
  __int64 v319; // [rsp+50h] [rbp-710h]
  __int64 v320; // [rsp+50h] [rbp-710h]
  char v321; // [rsp+58h] [rbp-708h]
  unsigned __int64 v322; // [rsp+58h] [rbp-708h]
  __int64 v323; // [rsp+58h] [rbp-708h]
  _QWORD *v324; // [rsp+58h] [rbp-708h]
  _QWORD *v325; // [rsp+58h] [rbp-708h]
  size_t v326; // [rsp+58h] [rbp-708h]
  size_t v327; // [rsp+58h] [rbp-708h]
  __int64 v328; // [rsp+58h] [rbp-708h]
  __int64 v329; // [rsp+58h] [rbp-708h]
  __int64 v330; // [rsp+60h] [rbp-700h]
  __int64 **v331; // [rsp+68h] [rbp-6F8h]
  char v332; // [rsp+68h] [rbp-6F8h]
  __int64 v333; // [rsp+68h] [rbp-6F8h]
  bool v336; // [rsp+90h] [rbp-6D0h]
  unsigned int v337; // [rsp+90h] [rbp-6D0h]
  __int64 v338; // [rsp+90h] [rbp-6D0h]
  _QWORD *v339; // [rsp+90h] [rbp-6D0h]
  __int64 v340; // [rsp+90h] [rbp-6D0h]
  _QWORD *v341; // [rsp+90h] [rbp-6D0h]
  __int64 v342; // [rsp+90h] [rbp-6D0h]
  __int64 v343; // [rsp+90h] [rbp-6D0h]
  unsigned int v344; // [rsp+90h] [rbp-6D0h]
  __int64 *v345; // [rsp+98h] [rbp-6C8h]
  _BYTE *v346; // [rsp+98h] [rbp-6C8h]
  __int64 v347; // [rsp+98h] [rbp-6C8h]
  __int64 v348; // [rsp+98h] [rbp-6C8h]
  __int64 v349; // [rsp+A0h] [rbp-6C0h]
  _BYTE *v350; // [rsp+A0h] [rbp-6C0h]
  _QWORD *v351; // [rsp+A0h] [rbp-6C0h]
  __int64 v352; // [rsp+A0h] [rbp-6C0h]
  __int64 v353; // [rsp+A0h] [rbp-6C0h]
  __int64 v354; // [rsp+A0h] [rbp-6C0h]
  __int64 v355; // [rsp+A0h] [rbp-6C0h]
  signed __int64 v356; // [rsp+A0h] [rbp-6C0h]
  __int64 v357; // [rsp+A0h] [rbp-6C0h]
  signed __int64 v358; // [rsp+A0h] [rbp-6C0h]
  __int64 *i; // [rsp+A8h] [rbp-6B8h]
  __int64 v360; // [rsp+A8h] [rbp-6B8h]
  signed __int64 v361; // [rsp+A8h] [rbp-6B8h]
  _QWORD *v364; // [rsp+E0h] [rbp-680h]
  __int64 v365; // [rsp+E8h] [rbp-678h] BYREF
  __m128i v366; // [rsp+F0h] [rbp-670h] BYREF
  __int64 v367; // [rsp+100h] [rbp-660h]
  __m128i v368; // [rsp+108h] [rbp-658h] BYREF
  __m128i v369; // [rsp+118h] [rbp-648h] BYREF
  __int64 v370; // [rsp+128h] [rbp-638h]
  _BYTE *v371; // [rsp+130h] [rbp-630h] BYREF
  unsigned __int64 v372; // [rsp+138h] [rbp-628h] BYREF
  __m128i v373; // [rsp+140h] [rbp-620h] BYREF
  __int64 v374; // [rsp+150h] [rbp-610h]
  __m128i v375; // [rsp+158h] [rbp-608h]
  __m128i v376; // [rsp+168h] [rbp-5F8h]
  __int64 v377; // [rsp+178h] [rbp-5E8h]
  char *v378; // [rsp+180h] [rbp-5E0h] BYREF
  __int64 v379; // [rsp+188h] [rbp-5D8h]
  unsigned __int64 v380[2]; // [rsp+190h] [rbp-5D0h] BYREF
  __int64 v381; // [rsp+1A0h] [rbp-5C0h]
  unsigned __int64 v382; // [rsp+1A8h] [rbp-5B8h]
  __int64 v383; // [rsp+1B0h] [rbp-5B0h]
  unsigned __int64 v384; // [rsp+1B8h] [rbp-5A8h]
  __int64 v385; // [rsp+1C0h] [rbp-5A0h]
  unsigned __int64 v386; // [rsp+1C8h] [rbp-598h]
  void *dest; // [rsp+1D0h] [rbp-590h] BYREF
  __int64 v388; // [rsp+1D8h] [rbp-588h]
  _BYTE v389[48]; // [rsp+1E0h] [rbp-580h] BYREF
  __int64 v390; // [rsp+210h] [rbp-550h]
  char v391; // [rsp+218h] [rbp-548h]
  __int64 v392; // [rsp+220h] [rbp-540h]
  const char *v393; // [rsp+230h] [rbp-530h] BYREF
  unsigned __int64 v394; // [rsp+238h] [rbp-528h] BYREF
  __m128i v395; // [rsp+240h] [rbp-520h] BYREF
  __int64 v396; // [rsp+250h] [rbp-510h]
  __m128i v397; // [rsp+258h] [rbp-508h] BYREF
  __m128i v398; // [rsp+268h] [rbp-4F8h] BYREF
  __int64 v399; // [rsp+278h] [rbp-4E8h]
  unsigned __int8 v400; // [rsp+280h] [rbp-4E0h]
  unsigned int *v401; // [rsp+290h] [rbp-4D0h] BYREF
  __int64 v402; // [rsp+298h] [rbp-4C8h] BYREF
  __int64 v403; // [rsp+2A0h] [rbp-4C0h] BYREF
  unsigned int v404; // [rsp+2A8h] [rbp-4B8h]
  char v405; // [rsp+2B0h] [rbp-4B0h]
  char v406; // [rsp+2B1h] [rbp-4AFh]
  __int64 v407; // [rsp+2C0h] [rbp-4A0h]
  __int64 v408; // [rsp+2C8h] [rbp-498h]
  __int64 v409; // [rsp+2D0h] [rbp-490h]
  __int64 v410; // [rsp+2D8h] [rbp-488h]
  void **v411; // [rsp+2E0h] [rbp-480h]
  void **v412; // [rsp+2E8h] [rbp-478h]
  __int64 v413; // [rsp+2F0h] [rbp-470h]
  int v414; // [rsp+2F8h] [rbp-468h]
  __int16 v415; // [rsp+2FCh] [rbp-464h]
  char v416; // [rsp+2FEh] [rbp-462h]
  __int64 v417; // [rsp+300h] [rbp-460h]
  __int64 v418; // [rsp+308h] [rbp-458h]
  void *v419; // [rsp+310h] [rbp-450h] BYREF
  void *v420; // [rsp+318h] [rbp-448h] BYREF
  __int64 v421; // [rsp+320h] [rbp-440h] BYREF
  __int64 v422; // [rsp+328h] [rbp-438h]
  _BYTE v423[320]; // [rsp+330h] [rbp-430h] BYREF
  void *src; // [rsp+470h] [rbp-2F0h] BYREF
  __int64 v425; // [rsp+478h] [rbp-2E8h] BYREF
  __m128i v426; // [rsp+480h] [rbp-2E0h] BYREF
  __int64 v427; // [rsp+490h] [rbp-2D0h]
  __m128i v428; // [rsp+498h] [rbp-2C8h] BYREF
  __m128i v429; // [rsp+4A8h] [rbp-2B8h] BYREF
  __int64 v430; // [rsp+4B8h] [rbp-2A8h]
  __int64 v431; // [rsp+4C0h] [rbp-2A0h]
  void *v432; // [rsp+4C8h] [rbp-298h]
  __int64 v433; // [rsp+4D0h] [rbp-290h]
  __int64 v434; // [rsp+4D8h] [rbp-288h]
  __int64 v435; // [rsp+4E0h] [rbp-280h]
  unsigned int v436; // [rsp+4E8h] [rbp-278h]
  __int64 v437; // [rsp+680h] [rbp-E0h]
  __int64 v438; // [rsp+688h] [rbp-D8h]
  __int64 v439; // [rsp+690h] [rbp-D0h]
  __int64 v440; // [rsp+698h] [rbp-C8h]
  char v441; // [rsp+6A0h] [rbp-C0h]
  __int64 v442; // [rsp+6A8h] [rbp-B8h]
  _BYTE *v443; // [rsp+6B0h] [rbp-B0h]
  __int64 v444; // [rsp+6B8h] [rbp-A8h]
  int v445; // [rsp+6C0h] [rbp-A0h]
  char v446; // [rsp+6C4h] [rbp-9Ch]
  _BYTE v447[64]; // [rsp+6C8h] [rbp-98h] BYREF
  __int16 v448; // [rsp+708h] [rbp-58h]
  _QWORD *v449; // [rsp+710h] [rbp-50h]
  _QWORD *v450; // [rsp+718h] [rbp-48h]
  __int64 v451; // [rsp+720h] [rbp-40h]

  v422 = 0x400000000LL;
  dest = v389;
  v9 = 0x600000000LL;
  v315 = a5;
  v331 = (__int64 **)a6;
  v421 = (__int64)v423;
  v388 = 0x600000000LL;
  v390 = 0;
  v391 = 1;
  v392 = 0;
  v393 = (const char *)a1;
  v394 = (unsigned __int64)&v421;
  v395.m128i_i64[0] = a3;
  v336 = (_BYTE)qword_5005EC8
      && (v76 = sub_B6AC80(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 72LL) + 40LL), 153)) != 0
      && *(_QWORD *)(v76 + 16) != 0;
  v10 = *(__int64 **)(a1 + 32);
  for ( i = *(__int64 **)(a1 + 40); i != v10; ++v10 )
  {
    v11 = *v10;
    v9 = *(unsigned int *)(a3 + 24);
    v12 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)v9 )
    {
      v9 = (unsigned int)(v9 - 1);
      v13 = v9 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v11 == *v14 )
      {
LABEL_8:
        v9 = a1;
        if ( a1 == v14[1] )
        {
          if ( *(_QWORD *)(v11 + 56) != v11 + 48 )
          {
            v345 = v10;
            v16 = v11 + 48;
            v17 = *(_QWORD *)(v11 + 56);
            v349 = v11;
            while ( 1 )
            {
              if ( !v17 )
                BUG();
              if ( *(_BYTE *)(v17 - 24) == 86 )
                break;
              if ( v336
                && sub_D222C0(v17 - 24)
                && (v19 = sub_2958930(*(_BYTE **)(v17 - 24 - 32LL * (*(_DWORD *)(v17 - 20) & 0x7FFFFFF))), *v19 > 0x15u)
                && (v312 = (unsigned __int64)v19, (unsigned __int8)sub_D48480(a1, (__int64)v19, v20, v21)) )
              {
                src = (void *)(v17 - 24);
                v425 = v312 & 0xFFFFFFFFFFFFFFFBLL;
                LOBYTE(v427) = 0;
                LOBYTE(v430) = 0;
                sub_2958F30((__int64)&v421, (unsigned __int64)&src, v312 & 0xFFFFFFFFFFFFFFFBLL, v22, v23, v24);
                sub_295C970(&v425);
                v17 = *(_QWORD *)(v17 + 8);
                if ( v16 == v17 )
                {
LABEL_20:
                  v11 = v349;
                  v10 = v345;
                  goto LABEL_21;
                }
              }
              else
              {
LABEL_12:
                v17 = *(_QWORD *)(v17 + 8);
                if ( v16 == v17 )
                  goto LABEL_20;
              }
            }
            v18 = *(_QWORD *)(v17 - 120);
            if ( sub_BCAC40(*(_QWORD *)(v18 + 8), 1) && !sub_BCAC40(*(_QWORD *)(v17 - 16), 1) )
              sub_295CA00((__int64 *)&v393, v17 - 24, (_BYTE *)v18);
            goto LABEL_12;
          }
LABEL_21:
          v26 = (_DWORD *)sub_986580(v11);
          if ( *(_BYTE *)v26 == 32 )
          {
            v27 = (__int64 *)*((_QWORD *)v26 - 1);
            if ( *(_BYTE *)*v27 > 0x15u && (unsigned __int8)sub_D48480(a1, *v27, v25, v9) && !sub_AA5780(v11) )
            {
              v29 = **((_QWORD **)v26 - 1);
              src = v26;
              LOBYTE(v427) = 0;
              LOBYTE(v430) = 0;
              v425 = v29 & 0xFFFFFFFFFFFFFFFBLL;
              sub_2958F30((__int64)&v421, (unsigned __int64)&src, v28, v9, a5, a6);
              sub_295C970(&v425);
            }
          }
          else if ( *(_BYTE *)v26 == 31 && (v26[1] & 0x7FFFFFF) == 3 && *((_QWORD *)v26 - 4) != *((_QWORD *)v26 - 8) )
          {
            sub_295CA00((__int64 *)&v393, (__int64)v26, *((_BYTE **)v26 - 12));
          }
        }
      }
      else
      {
        v38 = 1;
        while ( v15 != -4096 )
        {
          a5 = (unsigned int)(v38 + 1);
          v13 = v9 & (v38 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v11 == *v14 )
            goto LABEL_8;
          v38 = a5;
        }
      }
    }
  }
  if ( !a8 )
    goto LABEL_30;
  v346 = sub_D49780(a1, "llvm.loop.unswitch.partial.disable", 0x22u, v9, a5, a6);
  if ( v346 )
    goto LABEL_30;
  v77 = (__int64 *)v421;
  v78 = 80LL * (unsigned int)v422;
  v79 = (__int64 *)(v421 + v78);
  v80 = 0xCCCCCCCCCCCCCCCDLL * (v78 >> 4);
  if ( !(v80 >> 2) )
  {
LABEL_310:
    switch ( v80 )
    {
      case 2LL:
        v244 = sub_986580(**(_QWORD **)(a1 + 32));
        break;
      case 3LL:
        v244 = sub_986580(**(_QWORD **)(a1 + 32));
        if ( *v77 == v244 )
          goto LABEL_160;
        v77 += 10;
        break;
      case 1LL:
        v244 = sub_986580(**(_QWORD **)(a1 + 32));
        goto LABEL_314;
      default:
        goto LABEL_161;
    }
    if ( *v77 == v244 )
      goto LABEL_160;
    v77 += 10;
LABEL_314:
    if ( v244 != *v77 )
      goto LABEL_161;
    goto LABEL_160;
  }
  a5 = 0;
  v81 = (__int64 *)(v421 + 320 * (v80 >> 2));
  v82 = (_QWORD *)(**(_QWORD **)(a1 + 32) + 48LL);
  v83 = *v82 & 0xFFFFFFFFFFFFFFF8LL;
  a6 = v83 - 24;
  while ( 1 )
  {
    v9 = *v77;
    if ( v82 == (_QWORD *)v83 )
    {
      v84 = 0;
    }
    else
    {
      if ( !v83 )
        BUG();
      v84 = 0;
      if ( (unsigned int)*(unsigned __int8 *)(v83 - 24) - 30 < 0xB )
        v84 = v83 - 24;
    }
    if ( v9 == v84 )
      break;
    if ( v77[10] == v84 )
    {
      v77 += 10;
      break;
    }
    if ( v77[20] == v84 )
    {
      v77 += 20;
      break;
    }
    if ( v77[30] == v84 )
    {
      v77 += 30;
      break;
    }
    v77 += 40;
    if ( v81 == v77 )
    {
      v80 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v79 - (char *)v77) >> 4);
      goto LABEL_310;
    }
  }
LABEL_160:
  if ( v79 != v77 )
  {
LABEL_30:
    v346 = 0;
    goto LABEL_31;
  }
LABEL_161:
  sub_F71D40(&src, a1, qword_5005D08, *a8, v315, a6);
  if ( (_BYTE)v432 )
  {
    v120 = (unsigned int)v425;
    v121 = v425;
    if ( (unsigned int)v425 <= (unsigned __int64)(unsigned int)v388 )
    {
      if ( (_DWORD)v425 )
      {
        memmove(dest, src, 8LL * (unsigned int)v425);
        v120 = (unsigned int)v425;
      }
      v123 = src;
      v124 = 8 * v120;
      LODWORD(v388) = v121;
      v125 = (char *)src + 8 * v120;
    }
    else
    {
      if ( (unsigned int)v425 > (unsigned __int64)HIDWORD(v388) )
      {
        v122 = 0;
        LODWORD(v388) = 0;
        sub_C8D5F0((__int64)&dest, v389, (unsigned int)v425, 8u, a5, a6);
        v120 = (unsigned int)v425;
      }
      else
      {
        v122 = 8LL * (unsigned int)v388;
      }
      v123 = src;
      v124 = 8 * v120;
      v125 = (char *)src + v122;
      if ( (char *)src + v122 != (char *)src + 8 * v120 )
      {
        memcpy((char *)dest + v122, (char *)src + v122, v124 - v122);
        v123 = src;
        v124 = 8LL * (unsigned int)v425;
        v125 = (char *)src + v124;
      }
      LODWORD(v388) = v121;
    }
    v360 = (__int64)v125;
    v390 = v429.m128i_i64[1];
    v391 = v430;
    v392 = v431;
    v126 = *(__int64 **)(a1 + 32);
    v127 = sub_986580(*v126);
    v378 = 0;
    v346 = (_BYTE *)v127;
    if ( v123 != (_QWORD *)v360 )
    {
      if ( (_QWORD *)v360 == v123 + 1 )
      {
        v378 = (char *)(*v123 & 0xFFFFFFFFFFFFFFFBLL);
      }
      else
      {
        v128 = sub_22077B0(0x30u);
        v129 = v360;
        v130 = v128 & 0xFFFFFFFFFFFFFFF8LL;
        v131 = (const char *)(v128 | 4);
        if ( v128 )
        {
          *(_QWORD *)v128 = v128 + 16;
          *(_QWORD *)(v128 + 8) = 0x400000000LL;
        }
        v132 = *(unsigned int *)(v130 + 8);
        v133 = *(char **)v130;
        v378 = (char *)v131;
        v134 = 8 * v132;
        v135 = v124 >> 3;
        v136 = v132 + (v124 >> 3);
        v137 = *(unsigned int *)(v130 + 12);
        v138 = (__int64)&v133[8 * v132];
        if ( v133 == (char *)v138 )
        {
          if ( v137 < v136 )
          {
            sub_C8D5F0(v130, (const void *)(v130 + 16), v136, 8u, v138, v360);
            v132 = *(unsigned int *)(v130 + 8);
            v138 = *(_QWORD *)v130 + 8 * v132;
          }
          v285 = 0;
          if ( v124 )
          {
            do
            {
              *(_QWORD *)(v138 + v285) = v123[v285 / 8];
              v285 += 8LL;
            }
            while ( v124 != v285 );
            LODWORD(v132) = *(_DWORD *)(v130 + 8);
          }
          *(_DWORD *)(v130 + 8) = v135 + v132;
          v126 = *(__int64 **)(a1 + 32);
        }
        else
        {
          if ( v137 < v136 )
          {
            sub_C8D5F0(v130, (const void *)(v130 + 16), v136, 8u, v138, v360);
            v132 = *(unsigned int *)(v130 + 8);
            v133 = *(char **)v130;
            v129 = v360;
            v134 = 8 * v132;
            v138 = *(_QWORD *)v130 + 8 * v132;
          }
          v139 = v134 >> 3;
          if ( v124 <= (unsigned __int64)v134 )
          {
            v286 = (void *)v138;
            v287 = v134 - v124;
            v288 = v124;
            v289 = &v133[v134 - v124];
            v290 = (v124 >> 3) + v132;
            if ( v290 > *(unsigned int *)(v130 + 12) )
            {
              v327 = v134 - v124;
              v343 = v138;
              v358 = v134;
              sub_C8D5F0(v130, (const void *)(v130 + 16), v290, 8u, v138, v287);
              v132 = *(unsigned int *)(v130 + 8);
              v288 = v124;
              v287 = v327;
              v138 = v343;
              v134 = v358;
              v286 = (void *)(*(_QWORD *)v130 + 8 * v132);
            }
            if ( v289 != (char *)v138 )
            {
              v326 = v287;
              v342 = v138;
              v356 = v134;
              memmove(v286, v289, v288);
              LODWORD(v132) = *(_DWORD *)(v130 + 8);
              v287 = v326;
              v138 = v342;
              v134 = v356;
            }
            *(_DWORD *)(v130 + 8) = (v124 >> 3) + v132;
            if ( v133 != v289 )
              memmove((void *)(v138 + v124 - v134), v133, v287);
            v291 = 0;
            if ( v124 )
            {
              do
              {
                *(_QWORD *)&v133[v291] = v123[v291 / 8];
                v291 += 8LL;
              }
              while ( v124 != v291 );
            }
          }
          else
          {
            v140 = v135 + v132;
            *(_DWORD *)(v130 + 8) = v140;
            if ( v133 != (char *)v138 )
            {
              v323 = v134 >> 3;
              v338 = v138;
              v352 = v129;
              v361 = v134;
              memcpy(&v133[8 * v140 - v134], v133, v134);
              v139 = v323;
              v138 = v338;
              v129 = v352;
              v134 = v361;
            }
            if ( v134 )
            {
              v141 = 0;
              do
              {
                *(_QWORD *)&v133[8 * v141] = v123[v141];
                ++v141;
              }
              while ( v139 != v141 );
              v123 = (_QWORD *)((char *)v123 + v134);
            }
            v142 = v129 - (_QWORD)v123;
            v143 = 0;
            if ( v142 > 0 )
            {
              do
              {
                *(_QWORD *)(v138 + 8 * v143) = v123[v143];
                ++v143;
              }
              while ( (v142 >> 3) - v143 > 0 );
            }
          }
          v126 = *(__int64 **)(a1 + 32);
        }
      }
    }
    v144 = (unsigned int *)sub_986580(*v126);
    v145 = sub_295C9D0(&v378);
    v401 = v144;
    sub_295C880((unsigned __int64 *)&v402, v145, v146);
    v405 = 0;
    LOBYTE(v410) = 0;
    sub_2958F30((__int64)&v421, (unsigned __int64)&v401, v147, v148, v149, v150);
    sub_295C970(&v402);
    sub_295C970((__int64 *)&v378);
    if ( (_BYTE)v432 && src != &v426 )
      _libc_free((unsigned __int64)src);
  }
LABEL_31:
  v30 = (__int64 *)sub_D49780(a1, "llvm.loop.unswitch.injection.disable", 0x24u, v9, a5, a6);
  if ( !v30 && (_BYTE)qword_5005B48 )
  {
    v85 = **(_QWORD **)(a1 + 32);
    if ( v85 )
    {
      v86 = (unsigned int)(*(_DWORD *)(v85 + 44) + 1);
      v87 = *(_DWORD *)(v85 + 44) + 1;
    }
    else
    {
      v86 = 0;
      v87 = 0;
    }
    if ( v87 < *(_DWORD *)(a2 + 32) )
    {
      if ( *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v86) )
      {
        v88 = sub_D47930(a1);
        if ( v88 )
        {
          v401 = 0;
          v402 = 0;
          v403 = 0;
          v404 = 0;
          v91 = (unsigned int)(*(_DWORD *)(v88 + 44) + 1);
          if ( (unsigned int)v91 < *(_DWORD *)(a2 + 32) )
          {
            v89 = *(_QWORD *)(a2 + 24);
            v30 = *(__int64 **)(v89 + 8 * v91);
          }
          v92 = a3;
          v93 = a1 + 56;
          while ( 1 )
          {
            v94 = sub_B19060(v93, *v30, v89, v90);
            if ( !(_BYTE)v94 )
              break;
            v95 = *(_DWORD *)(v92 + 24);
            v96 = *(_QWORD *)(v92 + 8);
            v371 = 0;
            v97 = *v30;
            if ( !v95 )
              goto LABEL_130;
            v98 = v95 - 1;
            v89 = v98 & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
            v99 = (__int64 *)(v96 + 16 * v89);
            v100 = *v99;
            if ( v97 != *v99 )
            {
              v245 = 1;
              while ( v100 != -4096 )
              {
                v90 = (unsigned int)(v245 + 1);
                v89 = v98 & (unsigned int)(v245 + v89);
                v99 = (__int64 *)(v96 + 16LL * (unsigned int)v89);
                v100 = *v99;
                if ( v97 == *v99 )
                  goto LABEL_134;
                v245 = v90;
              }
              goto LABEL_130;
            }
LABEL_134:
            v90 = a1;
            if ( a1 != v99[1] )
              goto LABEL_130;
            v101 = sub_986580(v97);
            v351 = (_QWORD *)v101;
            if ( *(_BYTE *)v101 != 31 )
              goto LABEL_130;
            if ( (*(_DWORD *)(v101 + 4) & 0x7FFFFFF) != 3 )
              goto LABEL_130;
            v102 = *(_BYTE **)(v101 - 96);
            if ( *v102 != 82 )
              goto LABEL_130;
            if ( !*((_QWORD *)v102 - 8) )
              goto LABEL_130;
            v371 = (_BYTE *)*((_QWORD *)v102 - 8);
            v103 = *((_QWORD *)v102 - 4);
            if ( !v103 )
              goto LABEL_130;
            v104 = sub_B53900((__int64)v102);
            v337 = v104;
            v89 = v104;
            v105 = *(v351 - 4);
            v317 = v105;
            if ( !v105 )
              goto LABEL_130;
            v313 = *(v351 - 8);
            if ( !v313 || *(_BYTE *)(*((_QWORD *)v371 + 1) + 8LL) != 12 )
              goto LABEL_130;
            v322 = v104;
            v106 = sub_B19060(v93, v105, v104, v90);
            v108 = v322;
            if ( !v106 )
            {
              v278 = sub_B52870(v322);
              v107 = v313;
              v337 = v278;
              v317 = v313;
              v313 = v105;
            }
            if ( (unsigned __int8)sub_D48480(a1, (__int64)v371, v108, v107) )
            {
              v337 = sub_B52F50(v337);
              v277 = (__int64)v371;
              v371 = (_BYTE *)v103;
              v103 = v277;
            }
            if ( v337 == 39 && *(_BYTE *)v103 <= 0x15u )
            {
              if ( sub_AC30F0(v103) )
                goto LABEL_380;
              if ( *(_BYTE *)v103 == 17 )
              {
                v283 = *(_DWORD *)(v103 + 32);
                if ( v283 <= 0x40 )
                  v284 = *(_QWORD *)(v103 + 24) == 0;
                else
                  v284 = v283 == (unsigned int)sub_C444A0(v103 + 24);
                goto LABEL_398;
              }
              v328 = *(_QWORD *)(v103 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v328 + 8) - 17 > 1 )
                goto LABEL_148;
              v299 = sub_AD7630(v103, 0, v109);
              v109 = 0;
              if ( v299 && *v299 == 17 )
              {
                v300 = *((_DWORD *)v299 + 8);
                if ( v300 <= 0x40 )
                  v284 = *((_QWORD *)v299 + 3) == 0;
                else
                  v284 = v300 == (unsigned int)sub_C444A0((__int64)(v299 + 24));
LABEL_398:
                if ( !v284 )
                  goto LABEL_148;
LABEL_380:
                LODWORD(v425) = *(_DWORD *)(*(_QWORD *)(v103 + 8) + 8LL) >> 8;
                v279 = 1LL << ((unsigned __int8)v425 - 1);
                if ( (unsigned int)v425 <= 0x40 )
                {
                  src = 0;
                  goto LABEL_382;
                }
                v344 = v425 - 1;
                sub_C43690((__int64)&src, 0, 0);
                if ( (unsigned int)v425 <= 0x40 )
LABEL_382:
                  src = (void *)(v279 | (unsigned __int64)src);
                else
                  *((_QWORD *)src + (v344 >> 6)) |= v279;
                v280 = (__int64 *)sub_BD5C60(v103);
                v103 = sub_ACCFD0(v280, (__int64)&src);
                if ( (unsigned int)v425 > 0x40 && src )
                  j_j___libc_free_0_0((unsigned __int64)src);
                v337 = 36;
                goto LABEL_148;
              }
              if ( *(_BYTE *)(v328 + 8) == 17 )
              {
                v301 = *(_DWORD *)(v328 + 32);
                v329 = v92;
                v302 = v94;
                v303 = 0;
                v311 = v301;
                while ( v311 != v303 )
                {
                  v309 = v109;
                  v304 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)v103, v303);
                  if ( !v304 )
                    goto LABEL_451;
                  v110 = *v304;
                  v109 = v309;
                  if ( (_BYTE)v110 != 13 )
                  {
                    if ( (_BYTE)v110 != 17 )
                      goto LABEL_451;
                    v109 = *((unsigned int *)v304 + 8);
                    if ( (unsigned int)v109 <= 0x40 )
                    {
                      if ( *((_QWORD *)v304 + 3) )
                      {
LABEL_451:
                        v92 = v329;
                        goto LABEL_148;
                      }
                    }
                    else
                    {
                      v310 = *((_DWORD *)v304 + 8);
                      v305 = sub_C444A0((__int64)(v304 + 24));
                      v109 = v310;
                      if ( v310 != v305 )
                        goto LABEL_451;
                    }
                    v109 = v302;
                  }
                  ++v303;
                }
                v92 = v329;
                if ( (_BYTE)v109 )
                  goto LABEL_380;
              }
            }
LABEL_148:
            if ( !(unsigned __int8)sub_D48480(a1, (__int64)v371, v109, v110) )
            {
              v111 = sub_D48480(a1, v103, v89, v90);
              if ( v337 == 36
                && v111 == 1
                && (unsigned __int8)sub_B19060(v93, v317, v89, v90)
                && !(unsigned __int8)sub_B19060(v93, v313, v89, v90)
                && **(_QWORD **)(a1 + 32) != v317
                && (unsigned __int8)sub_295C520((__int64)v351, v317) )
              {
                v425 = v103;
                v426.m128i_i64[0] = v317;
                src = v351;
                for ( j = v371; *j == 68; v371 = j )
                  j = (_BYTE *)*((_QWORD *)j - 4);
                if ( (unsigned __int8)sub_2957590((__int64)&v401, (__int64 *)&v371, &v378) )
                {
                  v113 = v378;
                  p_src = (const __m128i *)&src;
                  v115 = *((unsigned int *)v378 + 4);
                  v90 = *((unsigned int *)v378 + 5);
                  v116 = v378 + 8;
                  v117 = v115 + 1;
                  v118 = 24 * v115;
                  if ( v90 < v117 )
                  {
                    v297 = *((_QWORD *)v378 + 1);
                    v298 = v378 + 24;
                    if ( v297 > (unsigned __int64)&src
                      || (v357 = *((_QWORD *)v378 + 1), (unsigned __int64)&src >= v297 + v118) )
                    {
                      sub_C8D5F0((__int64)(v378 + 8), v298, v117, 0x18u, v297, v117);
                      p_src = (const __m128i *)&src;
                      v118 = 24LL * *((unsigned int *)v113 + 4);
                    }
                    else
                    {
                      sub_C8D5F0((__int64)(v378 + 8), v298, v117, 0x18u, v297, v117);
                      v90 = (unsigned __int64)&src - v357;
                      v118 = 24LL * *((unsigned int *)v113 + 4);
                      p_src = (const __m128i *)((char *)&src + *((_QWORD *)v113 + 1) - v357);
                    }
                  }
LABEL_158:
                  v119 = (__m128i *)(*(_QWORD *)v116 + v118);
                  *v119 = _mm_loadu_si128(p_src);
                  v89 = p_src[1].m128i_i64[0];
                  v119[1].m128i_i64[0] = v89;
                  ++*((_DWORD *)v116 + 2);
                  goto LABEL_130;
                }
                v293 = v404;
                v294 = v378;
                v401 = (unsigned int *)((char *)v401 + 1);
                v295 = v403 + 1;
                v393 = v378;
                if ( 4 * ((int)v403 + 1) >= 3 * v404 )
                {
                  v293 = 2 * v404;
                }
                else
                {
                  v90 = v404 >> 3;
                  if ( v404 - HIDWORD(v403) - v295 > (unsigned int)v90 )
                  {
LABEL_420:
                    LODWORD(v403) = v295;
                    if ( *(_QWORD *)v294 != -4096 )
                      --HIDWORD(v403);
                    p_src = (const __m128i *)&src;
                    *(_QWORD *)v294 = v371;
                    v296 = v294 + 24;
                    v116 = v294 + 8;
                    *(_QWORD *)v116 = v296;
                    *((_QWORD *)v116 + 1) = 0x400000000LL;
                    v118 = 0;
                    goto LABEL_158;
                  }
                }
                sub_295B650((__int64)&v401, v293);
                sub_2957590((__int64)&v401, (__int64 *)&v371, &v393);
                v294 = (char *)v393;
                v295 = v403 + 1;
                goto LABEL_420;
              }
            }
LABEL_130:
            v30 = (__int64 *)v30[1];
          }
          v239 = (_QWORD *)v402;
          v240 = v404;
          v241 = 15LL * v404;
          if ( (_DWORD)v403 )
          {
            v247 = (_QWORD *)(v402 + v241 * 8);
            if ( v402 != v402 + v241 * 8 )
            {
              v248 = (_QWORD *)v402;
              while ( *v248 == -4096 || *v248 == -8192 )
              {
                v248 += 15;
                if ( v247 == v248 )
                  goto LABEL_300;
              }
              if ( v248 != v247 )
              {
                v249 = v248;
                while ( 1 )
                {
                  v250 = *((unsigned int *)v249 + 4);
                  if ( v250 <= 1 )
                    goto LABEL_352;
                  v251 = v249[1];
                  v252 = sub_B531B0(0x24u);
                  v253 = v251 + 24;
                  v355 = v251 + 24 * v250;
                  if ( v251 + 24 == v355 )
                    goto LABEL_352;
                  v341 = v247;
                  v254 = v252;
                  v255 = v253;
                  v325 = v249;
                  do
                  {
                    v264 = *(_QWORD *)(v255 + 8);
                    v265 = *(_QWORD *)(v255 - 16);
                    v266 = *(_QWORD *)(v255 - 8);
                    src = *(void **)(v255 - 24);
                    v267 = sub_22077B0(0x30u);
                    if ( v267 )
                    {
                      *(_QWORD *)(v267 + 16) = v264;
                      *(_QWORD *)v267 = v267 + 16;
                      *(_QWORD *)(v267 + 24) = v265;
                      *(_QWORD *)(v267 + 8) = 0x400000002LL;
                    }
                    v269 = (unsigned int)v422;
                    v261 = v267 | 4;
                    LOBYTE(v427) = 0;
                    v425 = v261;
                    v270 = v421;
                    v271 = (unsigned int)v422 + 1LL;
                    v428.m128i_i32[0] = v254;
                    v272 = &src;
                    v273 = v422;
                    v428.m128i_i64[1] = v264;
                    v429.m128i_i64[0] = v265;
                    v429.m128i_i64[1] = v266;
                    LOBYTE(v430) = 1;
                    if ( v271 > HIDWORD(v422) )
                    {
                      v320 = v261;
                      if ( v421 > (unsigned __int64)&src
                        || (v269 = v421 + 80LL * (unsigned int)v422, (unsigned __int64)&src >= v269) )
                      {
                        sub_2958E00((__int64)&v421, (unsigned int)v422 + 1LL, v269, v421, v268, v271);
                        v269 = (unsigned int)v422;
                        v270 = v421;
                        v272 = &src;
                        v261 = v320;
                        v273 = v422;
                      }
                      else
                      {
                        v275 = (char *)&src - v421;
                        sub_2958E00((__int64)&v421, (unsigned int)v422 + 1LL, v269, v421, v268, v271);
                        v270 = v421;
                        v269 = (unsigned int)v422;
                        v261 = v320;
                        v272 = (void **)&v275[v421];
                        v273 = v422;
                      }
                    }
                    v274 = v270 + 80 * v269;
                    if ( v274 )
                    {
                      *(_QWORD *)v274 = *v272;
                      v256 = v272[1];
                      v272[1] = 0;
                      *(_QWORD *)(v274 + 8) = v256;
                      v257 = _mm_loadu_si128((const __m128i *)v272 + 1);
                      v258 = _mm_loadu_si128((const __m128i *)(v272 + 5));
                      *(_QWORD *)(v274 + 32) = v272[4];
                      v259 = _mm_loadu_si128((const __m128i *)(v272 + 7));
                      v260 = v272[9];
                      *(__m128i *)(v274 + 16) = v257;
                      *(__m128i *)(v274 + 40) = v258;
                      *(_QWORD *)(v274 + 72) = v260;
                      *(__m128i *)(v274 + 56) = v259;
                      v261 = v425;
                      LODWORD(v422) = v422 + 1;
                      if ( !v425 )
                        goto LABEL_345;
                    }
                    else
                    {
                      LODWORD(v422) = v273 + 1;
                    }
                    if ( (v261 & 4) != 0 )
                    {
                      v262 = (unsigned __int64 *)(v261 & 0xFFFFFFFFFFFFFFF8LL);
                      v263 = (unsigned __int64)v262;
                      if ( v262 )
                      {
                        if ( (unsigned __int64 *)*v262 != v262 + 2 )
                          _libc_free(*v262);
                        j_j___libc_free_0(v263);
                      }
                    }
LABEL_345:
                    v255 += 24;
                  }
                  while ( v255 != v355 );
                  v247 = v341;
                  v249 = v325;
LABEL_352:
                  v249 += 15;
                  if ( v249 != v247 )
                  {
                    while ( *v249 == -8192 || *v249 == -4096 )
                    {
                      v249 += 15;
                      if ( v247 == v249 )
                        goto LABEL_356;
                    }
                    if ( v247 != v249 )
                      continue;
                  }
LABEL_356:
                  v239 = (_QWORD *)v402;
                  v240 = v404;
                  v241 = 15LL * v404;
                  break;
                }
              }
            }
          }
LABEL_300:
          if ( v240 )
          {
            v242 = &v239[v241];
            do
            {
              if ( *v239 != -8192 && *v239 != -4096 )
              {
                v243 = v239[1];
                if ( (_QWORD *)v243 != v239 + 3 )
                  _libc_free(v243);
              }
              v239 += 15;
            }
            while ( v242 != v239 );
            v239 = (_QWORD *)v402;
            v241 = 15LL * v404;
          }
          sub_C7D6A0((__int64)v239, v241 * 8, 8);
        }
      }
    }
  }
  v31 = 0;
  if ( (_DWORD)v422 )
  {
    sub_2969390(&v393, v421, (unsigned int)v422, a1, a2, a3, a4, v331, (__int64)&dest);
    v31 = v400;
    if ( v400 )
    {
      v364 = v393;
      v365 = v394;
      if ( v394 )
      {
        if ( (v394 & 4) != 0 )
        {
          v39 = v394 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v394 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v40 = (_QWORD *)sub_22077B0(0x30u);
            v45 = (unsigned __int64)v40;
            if ( v40 )
            {
              *v40 = v40 + 2;
              v40[1] = 0x400000000LL;
              if ( *(_DWORD *)(v39 + 8) )
                sub_29577D0((__int64)v40, v39, v41, v42, v43, v44);
            }
            v365 = v45 | 4;
          }
        }
      }
      v367 = v396;
      v46 = (int)qword_50064E8;
      v366 = _mm_load_si128(&v395);
      v370 = v399;
      v368 = _mm_loadu_si128(&v397);
      v369 = _mm_loadu_si128(&v398);
      if ( a7
        && (unsigned int)sub_DCF980(a7, (char *)a1)
        && ((unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll.full", 0x15u, v47, v48, v49)
         || (unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll.enable", 0x17u, v208, v209, v210))
        && *(_BYTE *)v364 == 32
        && (unsigned int)qword_5006248 >= ((*((_DWORD *)v364 + 1) & 0x7FFFFFFu) >> 1) - 1 )
      {
        v46 = (int)qword_5006328;
      }
      if ( v366.m128i_i32[2] )
      {
        if ( v366.m128i_i32[2] < 0 )
        {
LABEL_67:
          v332 = 0;
          if ( !(_BYTE)v370 )
            goto LABEL_68;
          v371 = v364;
          v372 = v365;
          if ( v365 )
          {
            if ( (v365 & 4) != 0 )
            {
              v160 = v365 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v365 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                v161 = (_QWORD *)sub_22077B0(0x30u);
                v166 = (unsigned __int64)v161;
                if ( v161 )
                {
                  *v161 = v161 + 2;
                  v161[1] = 0x400000000LL;
                  if ( *(_DWORD *)(v160 + 8) )
                    sub_29577D0((__int64)v161, v160, v162, v163, v164, v165);
                }
                v372 = v166 | 4;
              }
            }
          }
          v167 = _mm_load_si128(&v366);
          v168 = _mm_loadu_si128(&v369);
          v374 = v367;
          v373 = v167;
          v169 = _mm_loadu_si128(&v368);
          v377 = v370;
          v375 = v169;
          v376 = v168;
          v170 = sub_D4B130(a1);
          v171 = v375.m128i_i64[1];
          v353 = v170;
          v333 = v376.m128i_i64[1];
          v172 = v376.m128i_i64[0];
          v318 = v375.m128i_i16[0];
          v324 = v371;
          v330 = *((_QWORD *)v371 + 5);
          v314 = *((_QWORD *)v371 - 4);
          if ( v376.m128i_i64[1] == v314 )
            v314 = *((_QWORD *)v371 - 8);
          v173 = sub_AA48A0(v330);
          v174 = sub_986580(v353);
          v410 = sub_BD5C60(v174);
          v411 = &v419;
          v412 = &v420;
          v401 = (unsigned int *)&v403;
          v419 = &unk_49DA100;
          v402 = 0x200000000LL;
          v415 = 512;
          v420 = &unk_49DA0B0;
          v413 = 0;
          v414 = 0;
          v416 = 7;
          v417 = 0;
          v418 = 0;
          v407 = 0;
          v408 = 0;
          LOWORD(v409) = 0;
          sub_D5F1F0((__int64)&v401, v174);
          v175 = *(_QWORD *)(v171 + 8);
          v176 = *(_QWORD *)(v172 + 8);
          if ( v175 != v176 )
          {
            if ( *(_DWORD *)(v175 + 8) >> 8 >= *(_DWORD *)(v176 + 8) >> 8 )
            {
              src = (void *)sub_BD5D20(v172);
              LOWORD(v427) = 773;
              v425 = v246;
              v426.m128i_i64[0] = (__int64)".wide";
              v172 = sub_A82F30(&v401, v172, *(_QWORD *)(v171 + 8), (__int64)&src, 0);
            }
            else
            {
              src = (void *)sub_BD5D20(v171);
              LOWORD(v427) = 773;
              v425 = v177;
              v426.m128i_i64[0] = (__int64)".wide";
              v171 = sub_A82F30(&v401, v171, *(_QWORD *)(v172 + 8), (__int64)&src, 0);
            }
          }
          v178 = sub_986580(v353);
          src = "injected.cond";
          LOWORD(v427) = 259;
          v319 = sub_B52500(53, v318, v171, v172, (__int64)&src, v179, v178 + 24, 0);
          v180 = *(_QWORD *)(v330 + 72);
          src = (void *)sub_BD5D20(v330);
          LOWORD(v427) = 773;
          v425 = v181;
          v426.m128i_i64[0] = (__int64)".check";
          v182 = sub_22077B0(0x50u);
          v354 = v182;
          if ( v182 )
            sub_AA4D50(v182, v173, (__int64)&src, v180, v333);
          sub_D5F1F0((__int64)&v401, (__int64)v324);
          LOWORD(v427) = 257;
          v339 = sub_BD2C40(72, 3u);
          if ( v339 )
            sub_B4C9A0((__int64)v339, v333, v354, v319, 3u, v183, 0, 0);
          (*((void (__fastcall **)(void **, _QWORD *, void **, __int64, __int64))*v412 + 2))(
            v412,
            v339,
            &src,
            v408,
            v409);
          v184 = v401;
          v185 = &v401[4 * (unsigned int)v402];
          if ( v401 != v185 )
          {
            do
            {
              v186 = *((_QWORD *)v184 + 1);
              v187 = *v184;
              v184 += 4;
              sub_B99FD0((__int64)v339, v187, v186);
            }
            while ( v185 != v184 );
          }
          LOWORD(v409) = 0;
          v407 = v354;
          v408 = v354 + 48;
          v188 = *(v324 - 8);
          v189 = *(v324 - 4);
          v190 = *(v324 - 12);
          LOWORD(v427) = 257;
          v306 = v188;
          v307 = v189;
          v308 = v190;
          v191 = sub_BD2C40(72, 3u);
          v192 = (__int64)v191;
          v193 = v307;
          if ( v191 )
            sub_B4C9A0((__int64)v191, v307, v306, v308, 3u, v307, 0, 0);
          (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64, __int64))*v412 + 2))(
            v412,
            v192,
            &src,
            v408,
            v409,
            v193);
          v194 = &v401[4 * (unsigned int)v402];
          for ( k = v401; v194 != k; k += 4 )
          {
            v196 = *((_QWORD *)k + 1);
            v197 = *k;
            sub_B99FD0(v192, v197, v196);
          }
          sub_B43D60(v324);
          if ( *(_QWORD *)(v333 + 56) != v333 + 48 )
          {
            v198 = *(_QWORD *)(v333 + 56);
            do
            {
              if ( !v198 )
                BUG();
              if ( *(_BYTE *)(v198 - 24) != 84 )
                break;
              v199 = *(_QWORD *)(v198 - 32);
              v200 = 0x1FFFFFFFE0LL;
              if ( (*(_DWORD *)(v198 - 20) & 0x7FFFFFF) != 0 )
              {
                v201 = 0;
                do
                {
                  if ( v330 == *(_QWORD *)(v199 + 32LL * *(unsigned int *)(v198 + 48) + 8 * v201) )
                  {
                    v200 = 32 * v201;
                    goto LABEL_239;
                  }
                  ++v201;
                }
                while ( (*(_DWORD *)(v198 - 20) & 0x7FFFFFF) != (_DWORD)v201 );
                v200 = 0x1FFFFFFFE0LL;
              }
LABEL_239:
              sub_F0A850(v198 - 24, *(_QWORD *)(v199 + v200), v354);
              v198 = *(_QWORD *)(v198 + 8);
            }
            while ( v333 + 48 != v198 );
          }
          sub_AA5D60(v314, v330, v354);
          v380[0] = v330;
          v380[1] = v354 & 0xFFFFFFFFFFFFFFFBLL;
          v384 = v314 & 0xFFFFFFFFFFFFFFFBLL;
          v386 = v314 & 0xFFFFFFFFFFFFFFFBLL | 4;
          v382 = v333 & 0xFFFFFFFFFFFFFFFBLL;
          v379 = 0x400000004LL;
          v378 = (char *)v380;
          v381 = v354;
          v383 = v354;
          v385 = v330;
          sub_B26290((__int64)&src, v380, 4, 1u);
          sub_B24D40(a2, (__int64)&src, 0);
          sub_B1A8B0((__int64)&src, (__int64)&src);
          if ( a8 )
            sub_D75690(a8, (unsigned __int64 *)v378, (unsigned int)v379, a2, 0);
          sub_D4F330((__int64 *)a1, v354, a3);
          v202 = _mm_load_si128(&v373);
          LOBYTE(v430) = 0;
          src = v339;
          v427 = v374;
          v425 = v319 & 0xFFFFFFFFFFFFFFFBLL;
          v426 = v202;
          if ( v378 != (char *)v380 )
            _libc_free((unsigned __int64)v378);
          nullsub_61();
          v419 = &unk_49DA100;
          nullsub_63();
          if ( v401 != (unsigned int *)&v403 )
            _libc_free((unsigned __int64)v401);
          v364 = src;
          if ( (v425 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v425 & 4) != 0 && !*(_DWORD *)((v425 & 0xFFFFFFFFFFFFFFF8LL) + 8) )
          {
            if ( ((v365 >> 2) & 1) != 0 )
            {
              if ( v365 && ((v365 >> 2) & 1) != 0 && (v365 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                *(_DWORD *)((v365 & 0xFFFFFFFFFFFFFFF8LL) + 8) = 0;
            }
            else
            {
              v365 = 0;
            }
            goto LABEL_257;
          }
          if ( v365 )
          {
            if ( (v365 & 4) != 0 )
            {
              v203 = v365 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v365 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( ((v425 >> 2) & 1) == 0 )
                {
                  *(_DWORD *)(v203 + 8) = 0;
                  if ( (v425 & 4) != 0 )
                    v292 = **(_QWORD **)(v425 & 0xFFFFFFFFFFFFFFF8LL);
                  else
                    v292 = v425 & 0xFFFFFFFFFFFFFFF8LL;
                  sub_94F890(v203, v292);
                  v425 = 0;
                  goto LABEL_257;
                }
                if ( *(_QWORD *)v203 != v203 + 16 )
                  _libc_free(*(_QWORD *)v203);
                j_j___libc_free_0(v203);
              }
            }
          }
          v204 = v425;
          v425 = 0;
          v365 = v204;
LABEL_257:
          v205 = _mm_load_si128(&v426);
          v206 = _mm_loadu_si128(&v428);
          v367 = v427;
          v366 = v205;
          v207 = _mm_loadu_si128(&v429);
          v370 = v430;
          v368 = v206;
          v369 = v207;
          sub_295C970(&v425);
          sub_295C970((__int64 *)&v372);
          v332 = 1;
LABEL_68:
          if ( v364 != (_QWORD *)v346 )
            LODWORD(v388) = 0;
          if ( *(_BYTE *)v364 == 86 )
          {
            v50 = sub_D4B130(a1);
            v51 = sub_986580(v50);
            v321 = sub_98ED60((unsigned __int8 *)*(v364 - 12), a4, v51, a2, 0) ^ 1;
            v52 = v364[5];
            v448 = 0;
            v53 = 0;
            v347 = v52;
            src = &v426;
            v425 = 0x1000000000LL;
            v437 = 0;
            v438 = 0;
            v439 = a2;
            v440 = 0;
            v441 = 0;
            v442 = 0;
            v443 = v447;
            v444 = 8;
            v445 = 0;
            v446 = 1;
            v449 = 0;
            v450 = 0;
            v451 = 0;
            if ( (*((_BYTE *)v364 + 7) & 0x20) != 0 )
              v53 = sub_B91C10((__int64)v364, 2);
            sub_F38250(*(v364 - 12), v364 + 3, 0, 0, v53, (__int64)&src, a3, 0);
            v54 = sub_986580(v347);
            v350 = (_BYTE *)v54;
            v316 = *(_QWORD *)(v54 - 32);
            if ( a8 )
              sub_D6DEB0((__int64)a8, v347, *(_QWORD *)(v54 - 64), (__int64)v364);
            v406 = 1;
            v401 = (unsigned int *)"unswitched.select";
            v405 = 3;
            v55 = v364[1];
            v56 = sub_BD2DA0(80);
            v57 = v56;
            if ( v56 )
            {
              sub_B44260(v56, v55, 55, 0x8000000u, (__int64)(v364 + 3), 0);
              *(_DWORD *)(v57 + 72) = 2;
              sub_BD6B50((unsigned __int8 *)v57, (const char **)&v401);
              sub_BD2A10(v57, *(_DWORD *)(v57 + 72), 1);
            }
            v58 = (unsigned int **)(v57 + 48);
            sub_F0A850(v57, *(v364 - 8), v316);
            sub_F0A850(v57, *(v364 - 4), v347);
            v59 = (unsigned int *)v364[6];
            v401 = v59;
            if ( v59 )
            {
              sub_B96E90((__int64)&v401, (__int64)v59, 1);
              if ( v58 == &v401 )
              {
                if ( v401 )
                  sub_B91220((__int64)&v401, (__int64)v401);
                goto LABEL_81;
              }
              v211 = *(_QWORD *)(v57 + 48);
              if ( !v211 )
              {
LABEL_265:
                v212 = (unsigned __int8 *)v401;
                *(_QWORD *)(v57 + 48) = v401;
                if ( v212 )
                  sub_B976B0((__int64)&v401, v212, v57 + 48);
                goto LABEL_81;
              }
            }
            else if ( v58 == &v401 || (v211 = *(_QWORD *)(v57 + 48)) == 0 )
            {
LABEL_81:
              v60 = v57;
              sub_BD84D0((__int64)v364, v57);
              sub_B43D60(v364);
              if ( a8 && byte_4F8F8E8[0] )
              {
                v60 = 0;
                nullsub_390();
              }
              sub_FFCE90((__int64)&src, v60, v61, v62, v63, v64);
              sub_FFD870((__int64)&src, v60, v65, v66, v67, v68);
              sub_FFBC40((__int64)&src, v60);
              v69 = v450;
              v70 = v449;
              if ( v450 != v449 )
              {
                do
                {
                  v71 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v70[7];
                  *v70 = &unk_49E5048;
                  if ( v71 )
                    v71(v70 + 5, v70 + 5, 3);
                  *v70 = &unk_49DB368;
                  v72 = v70[3];
                  if ( v72 != 0 && v72 != -4096 && v72 != -8192 )
                    sub_BD60C0(v70 + 1);
                  v70 += 9;
                }
                while ( v69 != v70 );
                v70 = v449;
              }
              if ( v70 )
                j_j___libc_free_0((unsigned __int64)v70);
              if ( !v446 )
                _libc_free((unsigned __int64)v443);
              if ( src != &v426 )
                _libc_free((unsigned __int64)src);
              v73 = v321;
LABEL_99:
              v74 = v365 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v365 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( (v365 & 4) != 0 )
                {
                  v75 = *(__int64 **)v74;
                  v74 = *(unsigned int *)(v74 + 8);
                }
                else
                {
                  v74 = 1;
                  v75 = &v365;
                }
              }
              else
              {
                v75 = 0;
              }
              sub_296ADF0(a1, (__int64)v350, v75, v74, (__int64)&dest, a2, a3, a4, (__int64)a7, a8, a9, v73, v332);
LABEL_119:
              sub_295C970(&v365);
              if ( v400 )
              {
                v400 = 0;
                sub_295C970((__int64 *)&v394);
              }
              goto LABEL_33;
            }
            sub_B91220(v57 + 48, v211);
            goto LABEL_265;
          }
          if ( !sub_D222C0((__int64)v364) )
          {
            v350 = v364;
LABEL_190:
            v73 = 0;
            if ( (_BYTE)qword_5005C28 )
            {
              v425 = 0;
              v426 = 0u;
              src = &unk_4A32950;
              LODWORD(v427) = 0;
              v428.m128i_i8[0] = 0;
              v428.m128i_i64[1] = (__int64)&unk_4A32730;
              v429 = 0u;
              v430 = 0;
              LODWORD(v431) = 0;
              v433 = 0;
              v434 = 0;
              v435 = 0;
              v436 = 0;
              v432 = &unk_4A32758;
              sub_31034C0(&src, a1);
              if ( !(unsigned __int8)sub_31035E0(&src, v350, a2, a1) )
              {
                if ( *v350 == 31 )
                  v151 = sub_2958930(*((_BYTE **)v350 - 12));
                else
                  v151 = sub_2958930(**((_BYTE ***)v350 - 1));
                v152 = sub_D4B130(a1);
                v153 = sub_986580(v152);
                v73 = sub_98ED60(v151, a4, v153, a2, 0) ^ 1;
              }
              src = &unk_4A32950;
              v432 = &unk_4A20C88;
              sub_C7D6A0(v434, 16LL * v436, 8);
              v428.m128i_i64[1] = (__int64)&unk_4A20C88;
              sub_C7D6A0(v429.m128i_i64[1], 16LL * (unsigned int)v431, 8);
              src = &unk_4A21008;
              v154 = (unsigned int)v427;
              if ( (_DWORD)v427 )
              {
                v155 = (_QWORD *)v426.m128i_i64[0];
                v156 = v426.m128i_i64[0] + 16LL * (unsigned int)v427;
                do
                {
                  if ( *v155 != -8192 && *v155 != -4096 )
                  {
                    v157 = v155[1];
                    if ( v157 )
                    {
                      if ( (v157 & 4) != 0 )
                      {
                        v158 = (unsigned __int64 *)(v157 & 0xFFFFFFFFFFFFFFF8LL);
                        v159 = (unsigned __int64)v158;
                        if ( v158 )
                        {
                          if ( (unsigned __int64 *)*v158 != v158 + 2 )
                            _libc_free(*v158);
                          j_j___libc_free_0(v159);
                        }
                      }
                    }
                  }
                  v155 += 2;
                }
                while ( (_QWORD *)v156 != v155 );
                v154 = (unsigned int)v427;
              }
              sub_C7D6A0(v426.m128i_i64[0], 16 * v154, 8);
              v350 = v364;
            }
            goto LABEL_99;
          }
          v401 = (unsigned int *)&v403;
          v402 = 0x400000000LL;
          v213 = v364[5];
          if ( a8 && byte_4F8F8E8[0] )
            nullsub_390();
          v214 = 0;
          v437 = 0;
          src = &v426;
          v425 = 0x1000000000LL;
          v438 = 0;
          v439 = a2;
          v440 = 0;
          v441 = 0;
          v442 = 0;
          v443 = v447;
          v444 = 8;
          v445 = 0;
          v446 = 1;
          v448 = 0;
          v449 = 0;
          v450 = 0;
          v451 = 0;
          if ( (*((_BYTE *)v364 + 7) & 0x20) != 0 )
            v214 = sub_B91C10((__int64)v364, 2);
          v215 = sub_F38250(v364[-4 * (*((_DWORD *)v364 + 1) & 0x7FFFFFF)], v364 + 3, 0, 1, v214, (__int64)&src, a3, 0);
          v350 = (_BYTE *)sub_986580(v213);
          sub_B4CC70((__int64)v350);
          v216 = (unsigned __int8 *)*((_QWORD *)v350 - 4);
          v378 = "guarded";
          v340 = (__int64)v216;
          LOWORD(v381) = 259;
          sub_BD6B50(v216, (const char **)&v378);
          v217 = (unsigned __int8 *)*((_QWORD *)v350 - 8);
          v378 = "deopt";
          LOWORD(v381) = 259;
          sub_BD6B50(v217, (const char **)&v378);
          if ( !a8 )
          {
            sub_B444E0(v364, v215 + 24, 0);
            v276 = (__int64 *)sub_BD5C60((__int64)v364);
            v226 = (__int64 *)sub_ACD720(v276);
            sub_AC2B30((__int64)&v364[-4 * (*((_DWORD *)v364 + 1) & 0x7FFFFFF)], (__int64)v226);
LABEL_278:
            if ( unk_4F876C8 )
            {
              v226 = (__int64 *)a2;
              sub_D50AF0(a3);
            }
            sub_FFCE90((__int64)&src, (__int64)v226, v227, v228, v229, v230);
            sub_FFD870((__int64)&src, (__int64)v226, v231, v232, v233, v234);
            sub_FFBC40((__int64)&src, (__int64)v226);
            v235 = v450;
            v236 = v449;
            if ( v450 != v449 )
            {
              do
              {
                v237 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v236[7];
                *v236 = &unk_49E5048;
                if ( v237 )
                  v237(v236 + 5, v236 + 5, 3);
                *v236 = &unk_49DB368;
                v238 = v236[3];
                if ( v238 != 0 && v238 != -4096 && v238 != -8192 )
                  sub_BD60C0(v236 + 1);
                v236 += 9;
              }
              while ( v235 != v236 );
              v236 = v449;
            }
            if ( v236 )
              j_j___libc_free_0((unsigned __int64)v236);
            if ( !v446 )
              _libc_free((unsigned __int64)v443);
            if ( src != &v426 )
              _libc_free((unsigned __int64)src);
            v364 = v350;
            goto LABEL_190;
          }
          v348 = *((_QWORD *)v350 - 8);
          sub_D6DEB0((__int64)a8, v213, v340, (__int64)v364);
          sub_B444E0(v364, v215 + 24, 0);
          v218 = (__int64 *)sub_BD5C60((__int64)v364);
          v219 = sub_ACD720(v218);
          sub_AC2B30((__int64)&v364[-4 * (*((_DWORD *)v364 + 1) & 0x7FFFFFF)], v219);
          v221 = *(_DWORD *)(*a8 + 56);
          v222 = *(_QWORD *)(*a8 + 40);
          if ( v221 )
          {
            v223 = v221 - 1;
            v224 = v223 & (((unsigned int)v364 >> 9) ^ ((unsigned int)v364 >> 4));
            v225 = (__int64 *)(v222 + 16LL * v224);
            v220 = *v225;
            if ( v364 == (_QWORD *)*v225 )
            {
LABEL_275:
              v226 = (__int64 *)v225[1];
              goto LABEL_276;
            }
            v281 = 1;
            while ( v220 != -4096 )
            {
              v282 = v281 + 1;
              v224 = v223 & (v281 + v224);
              v225 = (__int64 *)(v222 + 16LL * v224);
              v220 = *v225;
              if ( v364 == (_QWORD *)*v225 )
                goto LABEL_275;
              v281 = v282;
            }
          }
          v226 = 0;
LABEL_276:
          sub_D75590(a8, v226, v348, 2, v348, v220);
          if ( byte_4F8F8E8[0] )
          {
            v226 = 0;
            nullsub_390();
          }
          goto LABEL_278;
        }
      }
      else if ( v46 > v366.m128i_i64[0] )
      {
        goto LABEL_67;
      }
      v31 = 0;
      goto LABEL_119;
    }
  }
LABEL_33:
  if ( dest != v389 )
    _libc_free((unsigned __int64)dest);
  v32 = v421;
  v33 = (_BYTE *)(v421 + 80LL * (unsigned int)v422);
  if ( (_BYTE *)v421 != v33 )
  {
    do
    {
      v34 = *((_QWORD *)v33 - 9);
      v33 -= 80;
      if ( v34 )
      {
        if ( (v34 & 4) != 0 )
        {
          v35 = (unsigned __int64 *)(v34 & 0xFFFFFFFFFFFFFFF8LL);
          v36 = (unsigned __int64)v35;
          if ( v35 )
          {
            if ( (unsigned __int64 *)*v35 != v35 + 2 )
              _libc_free(*v35);
            j_j___libc_free_0(v36);
          }
        }
      }
    }
    while ( (_BYTE *)v32 != v33 );
    v33 = (_BYTE *)v421;
  }
  if ( v33 != v423 )
    _libc_free((unsigned __int64)v33);
  return v31;
}
