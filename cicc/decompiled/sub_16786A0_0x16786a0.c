// Function: sub_16786A0
// Address: 0x16786a0
//
__int64 *__fastcall sub_16786A0(
        __int64 *a1,
        __int64 a2,
        __int64 **a3,
        char **a4,
        __int64 a5,
        __m128i *a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        char a15,
        char a16)
{
  void (__fastcall *v18)(__m128i *, __m128i *, __int64); // rcx
  __int64 v19; // rax
  __m128i v20; // xmm1
  __int64 v22; // rsi
  __m128i v23; // xmm0
  __m128i v24; // xmm2
  __int64 *v25; // rsi
  __int64 **v26; // rdx
  __int64 v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  char v30; // cl
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  char v33; // cl
  double v34; // xmm4_8
  double v35; // xmm5_8
  _QWORD *v36; // r14
  _QWORD *v37; // rbx
  __int64 v38; // rsi
  _QWORD *v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rdx
  char **v42; // r12
  char *v43; // rsi
  _QWORD *v44; // rsi
  __int64 v45; // r12
  __int64 v46; // rax
  _QWORD *v47; // rbx
  _QWORD *v48; // r13
  __int64 v49; // rsi
  _QWORD *v50; // rax
  _QWORD *v51; // r12
  _QWORD *v52; // r13
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  _QWORD *v56; // r12
  _QWORD *v57; // r13
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 *v61; // r12
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 **v65; // rbx
  __int64 *v66; // rax
  size_t v67; // rdx
  _QWORD *v68; // rcx
  __int64 **v69; // rbx
  __m128i *v70; // rdi
  __m128i *v71; // rax
  size_t v72; // rsi
  __int64 *v73; // rcx
  __int64 *v74; // rdi
  _QWORD *v75; // rdx
  __int64 *v76; // rdx
  __int64 *v77; // rax
  __int64 *v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rsi
  __int64 v81; // r8
  __int64 v82; // rax
  _QWORD *v83; // rdx
  unsigned int v84; // eax
  __int64 v85; // rdx
  unsigned int v86; // edi
  _QWORD *v87; // rcx
  __int64 v88; // r10
  __int64 v89; // rax
  __int64 *v90; // rdi
  __int64 *v91; // rbx
  __int64 **v92; // r12
  const void *v93; // rsi
  size_t v94; // rdx
  __int64 v95; // r12
  _QWORD *v96; // rax
  __int64 v97; // rdx
  int v98; // r13d
  unsigned int v99; // r14d
  unsigned int v100; // esi
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // r14
  unsigned int v104; // r12d
  int v105; // ebx
  unsigned int v106; // esi
  __int64 v107; // r13
  __int64 v108; // rsi
  unsigned int v109; // edi
  _QWORD *v110; // rcx
  __int64 v111; // r10
  int v112; // r13d
  unsigned int j; // r14d
  __int64 v114; // rax
  __int64 v115; // rsi
  __int64 v116; // rax
  int v117; // ecx
  int v118; // r11d
  size_t v119; // rdx
  _QWORD *v120; // rbx
  _QWORD *v121; // r12
  __int64 v122; // rsi
  _QWORD *v123; // rbx
  _QWORD *v124; // r12
  __int64 v125; // rsi
  __int64 **v126; // r12
  __int64 v127; // rax
  __int64 *v128; // r12
  __int64 **v129; // rbx
  __int64 v130; // rcx
  __m128i *v131; // rax
  __int64 v132; // rcx
  __m128i *v133; // rax
  __int64 v134; // rcx
  __m128i *v135; // rax
  __int64 v136; // rcx
  __m128i *v137; // rax
  __int64 v138; // rcx
  __m128i *v139; // rax
  __int64 *v140; // rcx
  __m128i *v141; // rax
  __int64 v142; // rcx
  __m128i *v143; // rax
  __int64 v144; // rbx
  __m128i *v145; // rax
  unsigned int v146; // r12d
  int v147; // eax
  int v148; // r13d
  __int64 v149; // r14
  __int64 v150; // rcx
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 v153; // rbx
  __int64 v154; // rax
  __int64 v155; // rdx
  bool v156; // cc
  _QWORD *v157; // rdx
  int v158; // eax
  unsigned int v159; // r13d
  int v160; // r15d
  __int64 v161; // rax
  unsigned int v162; // esi
  __int64 v163; // rcx
  __int64 v164; // r8
  __int64 v165; // rdi
  __m128i *v166; // rdx
  unsigned int v167; // edx
  __int64 v168; // r9
  __int64 v169; // r10
  _QWORD *v170; // r9
  int v171; // r8d
  __int64 v172; // r9
  _QWORD *v173; // rcx
  __int64 **v174; // r12
  __int64 *v175; // rax
  _BYTE *v176; // rsi
  __m128i *v177; // rdi
  __m128i *v178; // rax
  size_t v179; // rsi
  __int64 *v180; // rcx
  __int64 *v181; // rdi
  __int64 *v182; // r14
  __int64 **v183; // rbx
  __int64 v184; // rcx
  __m128i *v185; // rax
  __int64 v186; // rcx
  __m128i *v187; // rax
  __int64 v188; // rcx
  __m128i *v189; // rax
  __int64 v190; // rcx
  __m128i *v191; // rax
  __int64 v192; // rcx
  __m128i *v193; // rax
  __int64 v194; // rcx
  __m128i *v195; // rax
  __int64 v196; // rcx
  __m128i *v197; // rax
  int v198; // eax
  __int64 v199; // r12
  int v200; // ebx
  __int64 *v201; // rax
  _BYTE *v202; // rdi
  _BYTE *v203; // rdx
  unsigned int v204; // r13d
  unsigned int v205; // r8d
  __int64 *v206; // rcx
  __int64 v207; // r9
  __int64 v208; // rax
  __int64 v209; // r12
  __int32 v210; // edx
  unsigned int v211; // ecx
  __int64 *v212; // rax
  __int64 v213; // r8
  __m128i v214; // rax
  char v215; // al
  _QWORD *v216; // rdx
  __int64 v217; // rsi
  __int64 v218; // rdx
  _QWORD *v219; // rax
  _QWORD *v220; // rdx
  __int64 **v221; // rcx
  __int64 v222; // rdx
  __int64 v223; // r12
  __int64 v224; // rsi
  char v225; // al
  __int64 v226; // rdx
  __m128i *v227; // rdx
  __int64 v228; // r12
  __int64 v229; // r8
  unsigned __int64 v230; // rsi
  __int64 v231; // rax
  __int64 v232; // rcx
  __int64 v233; // rdx
  __int64 v234; // r12
  _QWORD *v235; // rbx
  unsigned __int64 v236; // r9
  _QWORD *v237; // rdx
  _QWORD *v238; // r12
  __int64 v239; // rcx
  __int64 v240; // r12
  _QWORD *v241; // rbx
  __int64 v242; // r8
  __int64 *v243; // rsi
  _QWORD *v244; // rdx
  _QWORD *v245; // r12
  __int64 v246; // rsi
  __m128i v247; // rax
  char v248; // al
  __m128i *v249; // rdx
  __m128i v250; // rax
  char v251; // al
  unsigned int v252; // eax
  __int64 v253; // rdx
  int v254; // ecx
  int v255; // r11d
  int v256; // r10d
  __int64 v257; // r9
  __int32 v258; // edx
  int v259; // r9d
  __int64 *v260; // rbx
  _QWORD *m128i_i64; // rdx
  __int64 *v262; // rsi
  __int64 v263; // rcx
  __int64 v264; // rcx
  __m128i *v265; // rax
  __int64 v266; // rcx
  __int64 v267; // rdi
  int v268; // r9d
  __int64 v269; // rsi
  int v270; // r8d
  __int64 v271; // r15
  __int64 v272; // rcx
  __int64 v273; // rsi
  _BYTE *v274; // rsi
  __int64 *v275; // rcx
  unsigned int v276; // r13d
  int v277; // esi
  __int64 v278; // rdi
  __m128i v279; // rax
  int v280; // ebx
  int v281; // edi
  __int64 *v282; // rsi
  __int64 v283; // [rsp+0h] [rbp-950h]
  __int64 v284; // [rsp+0h] [rbp-950h]
  __int64 v285; // [rsp+8h] [rbp-948h]
  __int64 v286; // [rsp+8h] [rbp-948h]
  __int64 v287; // [rsp+8h] [rbp-948h]
  __int64 v288; // [rsp+10h] [rbp-940h]
  int v289; // [rsp+10h] [rbp-940h]
  int v292; // [rsp+58h] [rbp-8F8h]
  __int64 *v293; // [rsp+60h] [rbp-8F0h]
  __int64 v294; // [rsp+60h] [rbp-8F0h]
  int v295; // [rsp+60h] [rbp-8F0h]
  char **v296; // [rsp+68h] [rbp-8E8h]
  __int64 *v297; // [rsp+68h] [rbp-8E8h]
  _BYTE *v298; // [rsp+70h] [rbp-8E0h] BYREF
  __int64 v299; // [rsp+78h] [rbp-8D8h]
  _QWORD v300[2]; // [rsp+80h] [rbp-8D0h] BYREF
  __int64 v301[2]; // [rsp+90h] [rbp-8C0h] BYREF
  _QWORD v302[2]; // [rsp+A0h] [rbp-8B0h] BYREF
  _QWORD v303[2]; // [rsp+B0h] [rbp-8A0h] BYREF
  __int64 v304; // [rsp+C0h] [rbp-890h] BYREF
  __m128i *v305; // [rsp+D0h] [rbp-880h] BYREF
  __int64 v306; // [rsp+D8h] [rbp-878h]
  __m128i v307; // [rsp+E0h] [rbp-870h] BYREF
  __m128i *v308; // [rsp+F0h] [rbp-860h] BYREF
  __int64 v309; // [rsp+F8h] [rbp-858h]
  __m128i v310; // [rsp+100h] [rbp-850h] BYREF
  __m128i *v311; // [rsp+110h] [rbp-840h] BYREF
  __int64 v312; // [rsp+118h] [rbp-838h]
  __m128i v313; // [rsp+120h] [rbp-830h] BYREF
  __m128i *v314; // [rsp+130h] [rbp-820h] BYREF
  __int64 v315; // [rsp+138h] [rbp-818h]
  __m128i v316; // [rsp+140h] [rbp-810h] BYREF
  __m128i *v317; // [rsp+150h] [rbp-800h] BYREF
  __int64 v318; // [rsp+158h] [rbp-7F8h]
  __m128i v319; // [rsp+160h] [rbp-7F0h] BYREF
  __m128i *v320; // [rsp+170h] [rbp-7E0h] BYREF
  __int64 v321; // [rsp+178h] [rbp-7D8h]
  __m128i v322; // [rsp+180h] [rbp-7D0h] BYREF
  __m128i v323; // [rsp+190h] [rbp-7C0h] BYREF
  __int64 v324; // [rsp+1A0h] [rbp-7B0h] BYREF
  __m128i v325; // [rsp+1B0h] [rbp-7A0h] BYREF
  __m128i v326; // [rsp+1C0h] [rbp-790h] BYREF
  __m128i v327; // [rsp+1D0h] [rbp-780h] BYREF
  __m128i v328; // [rsp+1E0h] [rbp-770h] BYREF
  __m128i *v329; // [rsp+1F0h] [rbp-760h] BYREF
  __int64 v330; // [rsp+1F8h] [rbp-758h]
  __m128i v331; // [rsp+200h] [rbp-750h] BYREF
  __m128i v332; // [rsp+210h] [rbp-740h] BYREF
  __m128i v333; // [rsp+220h] [rbp-730h] BYREF
  __int64 *v334; // [rsp+230h] [rbp-720h]
  __m128i *v335; // [rsp+240h] [rbp-710h] BYREF
  __int64 *v336; // [rsp+248h] [rbp-708h]
  __m128i v337; // [rsp+250h] [rbp-700h] BYREF
  __m128i **v338; // [rsp+260h] [rbp-6F0h]
  __m128i **v339; // [rsp+268h] [rbp-6E8h]
  _QWORD v340[2]; // [rsp+270h] [rbp-6E0h] BYREF
  __int64 v341; // [rsp+280h] [rbp-6D0h] BYREF
  int v342; // [rsp+290h] [rbp-6C0h]
  _QWORD v343[2]; // [rsp+2B0h] [rbp-6A0h] BYREF
  __int64 v344; // [rsp+2C0h] [rbp-690h] BYREF
  __m128i v345; // [rsp+2F0h] [rbp-660h] BYREF
  __m128i v346; // [rsp+300h] [rbp-650h] BYREF
  __int64 v347; // [rsp+310h] [rbp-640h]
  _BYTE *v348; // [rsp+380h] [rbp-5D0h] BYREF
  __int64 v349; // [rsp+388h] [rbp-5C8h]
  _BYTE v350[128]; // [rsp+390h] [rbp-5C0h] BYREF
  size_t n[2]; // [rsp+410h] [rbp-540h] BYREF
  __m128i v352; // [rsp+420h] [rbp-530h] BYREF
  __int64 i; // [rsp+430h] [rbp-520h]
  __int64 *v354; // [rsp+4A0h] [rbp-4B0h] BYREF
  __int64 v355; // [rsp+4A8h] [rbp-4A8h]
  _BYTE v356[128]; // [rsp+4B0h] [rbp-4A0h] BYREF
  __int64 **v357; // [rsp+530h] [rbp-420h] BYREF
  __int64 *v358; // [rsp+538h] [rbp-418h]
  __m128i v359; // [rsp+540h] [rbp-410h] BYREF
  void (__fastcall *v360)(__m128i *, __m128i *, __int64); // [rsp+550h] [rbp-400h]
  __int64 v361; // [rsp+558h] [rbp-3F8h]
  _QWORD v362[2]; // [rsp+560h] [rbp-3F0h] BYREF
  __int64 v363; // [rsp+570h] [rbp-3E0h]
  __int64 v364; // [rsp+578h] [rbp-3D8h]
  int v365; // [rsp+580h] [rbp-3D0h]
  _BYTE *v366; // [rsp+588h] [rbp-3C8h]
  __int64 v367; // [rsp+590h] [rbp-3C0h]
  _BYTE v368[128]; // [rsp+598h] [rbp-3B8h] BYREF
  _BYTE *v369; // [rsp+618h] [rbp-338h]
  __int64 v370; // [rsp+620h] [rbp-330h]
  _BYTE v371[128]; // [rsp+628h] [rbp-328h] BYREF
  _BYTE *v372; // [rsp+6A8h] [rbp-2A8h]
  __int64 v373; // [rsp+6B0h] [rbp-2A0h]
  _BYTE v374[128]; // [rsp+6B8h] [rbp-298h] BYREF
  __int64 v375; // [rsp+738h] [rbp-218h]
  _BYTE *v376; // [rsp+740h] [rbp-210h]
  _BYTE *v377; // [rsp+748h] [rbp-208h]
  __int64 v378; // [rsp+750h] [rbp-200h]
  int v379; // [rsp+758h] [rbp-1F8h]
  _BYTE v380[128]; // [rsp+760h] [rbp-1F0h] BYREF
  __int64 v381; // [rsp+7E0h] [rbp-170h]
  char v382; // [rsp+7E8h] [rbp-168h]
  _QWORD v383[2]; // [rsp+7F0h] [rbp-160h] BYREF
  _QWORD v384[2]; // [rsp+800h] [rbp-150h] BYREF
  __int64 v385; // [rsp+810h] [rbp-140h]
  __int64 v386; // [rsp+818h] [rbp-138h] BYREF
  _QWORD *v387; // [rsp+820h] [rbp-130h]
  __int64 v388; // [rsp+828h] [rbp-128h]
  unsigned int v389; // [rsp+830h] [rbp-120h]
  __int64 v390; // [rsp+838h] [rbp-118h]
  _QWORD *v391; // [rsp+840h] [rbp-110h]
  __int64 v392; // [rsp+848h] [rbp-108h]
  unsigned int v393; // [rsp+850h] [rbp-100h]
  char v394; // [rsp+858h] [rbp-F8h]
  char v395; // [rsp+861h] [rbp-EFh]
  __int64 v396; // [rsp+868h] [rbp-E8h] BYREF
  _QWORD *v397; // [rsp+870h] [rbp-E0h]
  __int64 v398; // [rsp+878h] [rbp-D8h]
  unsigned int v399; // [rsp+880h] [rbp-D0h]
  _QWORD *v400; // [rsp+890h] [rbp-C0h]
  unsigned int v401; // [rsp+8A0h] [rbp-B0h]
  char v402; // [rsp+8A8h] [rbp-A8h]
  char v403; // [rsp+8B1h] [rbp-9Fh]
  __int64 v404; // [rsp+8B8h] [rbp-98h]
  __int64 v405; // [rsp+8C0h] [rbp-90h]
  __int64 v406; // [rsp+8C8h] [rbp-88h]
  __int64 v407; // [rsp+8D0h] [rbp-80h]
  unsigned __int64 v408; // [rsp+8D8h] [rbp-78h]
  __int64 *v409; // [rsp+8E0h] [rbp-70h]
  __int64 v410; // [rsp+8E8h] [rbp-68h]
  char v411; // [rsp+8F0h] [rbp-60h]
  char v412; // [rsp+8F1h] [rbp-5Fh]
  __int64 v413; // [rsp+8F8h] [rbp-58h] BYREF
  char v414; // [rsp+900h] [rbp-50h]
  char v415[8]; // [rsp+908h] [rbp-48h] BYREF
  int v416; // [rsp+910h] [rbp-40h]

  v18 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a6[1].m128i_i64[0];
  v19 = a6[1].m128i_i64[1];
  v20 = _mm_load_si128(&v345);
  v22 = v346.m128i_i64[1];
  a6[1].m128i_i64[0] = 0;
  v23 = _mm_loadu_si128(a6);
  *a6 = v20;
  v24 = _mm_load_si128(&v359);
  a6[1].m128i_i64[1] = v22;
  v25 = *a3;
  *a3 = 0;
  v26 = *(__int64 ***)a2;
  v358 = v25;
  v357 = v26;
  v27 = v361;
  v361 = v19;
  v362[0] = off_49EE4A0;
  v360 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v18;
  v346.m128i_i64[1] = v27;
  v345 = v24;
  v359 = v23;
  v346.m128i_i64[0] = 0;
  v362[1] = 0;
  v366 = v368;
  v367 = 0x1000000000LL;
  v370 = 0x1000000000LL;
  v373 = 0x1000000000LL;
  v376 = v380;
  v377 = v380;
  v381 = a2 + 8;
  v382 = a16;
  v383[0] = off_49EE4F8;
  v369 = v371;
  v384[0] = off_49EE518;
  v372 = v374;
  v383[1] = &v357;
  v384[1] = &v357;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v375 = 0;
  v378 = 16;
  v379 = 0;
  v385 = a2 + 72;
  v386 = 0;
  v389 = 128;
  v28 = (_QWORD *)sub_22077B0(0x2000);
  v388 = 0;
  v387 = v28;
  n[1] = 2;
  v29 = &v28[8 * (unsigned __int64)v389];
  n[0] = (size_t)&unk_49E6B50;
  v352.m128i_i64[0] = 0;
  v352.m128i_i64[1] = -8;
  for ( i = 0; v29 != v28; v28 += 8 )
  {
    if ( v28 )
    {
      v30 = n[1];
      v28[2] = 0;
      v28[3] = -8;
      *v28 = &unk_49E6B50;
      v28[1] = v30 & 6;
      v28[4] = i;
    }
  }
  v394 = 0;
  v395 = 1;
  v396 = 0;
  v399 = 128;
  v31 = (_QWORD *)sub_22077B0(0x2000);
  v398 = 0;
  v397 = v31;
  n[1] = 2;
  v32 = &v31[8 * (unsigned __int64)v399];
  n[0] = (size_t)&unk_49E6B50;
  v352.m128i_i64[0] = 0;
  v352.m128i_i64[1] = -8;
  for ( i = 0; v32 != v31; v31 += 8 )
  {
    if ( v31 )
    {
      v33 = n[1];
      v31[2] = 0;
      v31[3] = -8;
      *v31 = &unk_49E6B50;
      v31[1] = v33 & 6;
      v31[4] = i;
    }
  }
  v411 = a15;
  v402 = 0;
  v403 = 1;
  v404 = 0;
  v405 = 0;
  v406 = 0;
  v407 = 0;
  v408 = 0;
  v409 = 0;
  v410 = 0;
  v412 = 0;
  v414 = 0;
  sub_1B75040(v415, &v386, 6, v362, v383);
  v416 = sub_1B751F0(v415, &v396, v384);
  if ( v394 )
  {
    if ( v393 )
    {
      v36 = v391;
      v296 = a4;
      v37 = &v391[2 * v393];
      do
      {
        if ( *v36 != -4 && *v36 != -8 )
        {
          v38 = v36[1];
          if ( v38 )
            sub_161E7C0((__int64)(v36 + 1), v38);
        }
        v36 += 2;
      }
      while ( v37 != v36 );
      a4 = v296;
    }
    j___libc_free_0(v391);
    ++v390;
    v39 = *(_QWORD **)(a2 + 80);
    v40 = *(_DWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 80) = 0;
    ++*(_QWORD *)(a2 + 72);
    v391 = v39;
    v41 = *(_QWORD *)(a2 + 88);
    v393 = v40;
    v392 = v41;
    *(_QWORD *)(a2 + 88) = 0;
    *(_DWORD *)(a2 + 96) = 0;
  }
  else
  {
    v394 = 1;
    v390 = 1;
    v83 = *(_QWORD **)(a2 + 80);
    v84 = *(_DWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 80) = 0;
    ++*(_QWORD *)(a2 + 72);
    v391 = v83;
    v85 = *(_QWORD *)(a2 + 88);
    v393 = v84;
    v392 = v85;
    *(_QWORD *)(a2 + 88) = 0;
    *(_DWORD *)(a2 + 96) = 0;
  }
  v42 = &a4[a5];
  while ( v42 != a4 )
  {
    v43 = *a4++;
    sub_1671660((__int64)&v357, v43);
  }
  if ( a15 )
    sub_1671B40(
      (__int64)&v357,
      *(double *)v23.m128i_i64,
      *(double *)v20.m128i_i64,
      *(double *)v24.m128i_i64,
      a10,
      v34,
      v35,
      a13,
      a14);
  if ( v346.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v346.m128i_i64[0])(&v345, &v345, 3);
  v44 = (_QWORD *)v358[21];
  if ( v44 )
  {
    (*(void (__fastcall **)(size_t *))(*v44 + 32LL))(n);
    if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = n[0] & 0xFFFFFFFFFFFFFFFELL | 1;
      goto LABEL_28;
    }
  }
  if ( !*(_QWORD *)(sub_1632FA0((__int64)v357) + 200) )
  {
    v126 = v357;
    v127 = sub_1632FA0((__int64)v358);
    sub_1632B40((__int64)v126, v127);
  }
  v63 = sub_1632FA0((__int64)v357);
  v64 = sub_1632FA0((__int64)v358);
  if ( !(unsigned __int8)sub_15A8140(v64, v63) )
  {
    v128 = v358;
    v129 = v357;
    sub_8FD6D0((__int64)&v323, "Linking two modules of different data layouts: '", v358 + 22);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v323.m128i_i64[1]) <= 5 )
      goto LABEL_486;
    v131 = (__m128i *)sub_2241490(&v323, "' is '", 6, v130);
    v325.m128i_i64[0] = (__int64)&v326;
    if ( (__m128i *)v131->m128i_i64[0] == &v131[1] )
    {
      v326 = _mm_loadu_si128(v131 + 1);
    }
    else
    {
      v325.m128i_i64[0] = v131->m128i_i64[0];
      v326.m128i_i64[0] = v131[1].m128i_i64[0];
    }
    v325.m128i_i64[1] = v131->m128i_i64[1];
    v132 = v325.m128i_i64[1];
    v131->m128i_i64[0] = (__int64)v131[1].m128i_i64;
    v131->m128i_i64[1] = 0;
    v131[1].m128i_i8[0] = 0;
    v133 = (__m128i *)sub_2241490(&v325, v128[59], v128[60], v132);
    v327.m128i_i64[0] = (__int64)&v328;
    if ( (__m128i *)v133->m128i_i64[0] == &v133[1] )
    {
      v328 = _mm_loadu_si128(v133 + 1);
    }
    else
    {
      v327.m128i_i64[0] = v133->m128i_i64[0];
      v328.m128i_i64[0] = v133[1].m128i_i64[0];
    }
    v134 = v133->m128i_i64[1];
    v327.m128i_i64[1] = v134;
    v133->m128i_i64[0] = (__int64)v133[1].m128i_i64;
    v133->m128i_i64[1] = 0;
    v133[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v327.m128i_i64[1]) <= 0xA )
      goto LABEL_486;
    v135 = (__m128i *)sub_2241490(&v327, "' whereas '", 11, v134);
    v329 = &v331;
    if ( (__m128i *)v135->m128i_i64[0] == &v135[1] )
    {
      v331 = _mm_loadu_si128(v135 + 1);
    }
    else
    {
      v329 = (__m128i *)v135->m128i_i64[0];
      v331.m128i_i64[0] = v135[1].m128i_i64[0];
    }
    v330 = v135->m128i_i64[1];
    v136 = v330;
    v135->m128i_i64[0] = (__int64)v135[1].m128i_i64;
    v135->m128i_i64[1] = 0;
    v135[1].m128i_i8[0] = 0;
    v137 = (__m128i *)sub_2241490(&v329, v129[22], v129[23], v136);
    v332.m128i_i64[0] = (__int64)&v333;
    if ( (__m128i *)v137->m128i_i64[0] == &v137[1] )
    {
      v333 = _mm_loadu_si128(v137 + 1);
    }
    else
    {
      v332.m128i_i64[0] = v137->m128i_i64[0];
      v333.m128i_i64[0] = v137[1].m128i_i64[0];
    }
    v138 = v137->m128i_i64[1];
    v332.m128i_i64[1] = v138;
    v137->m128i_i64[0] = (__int64)v137[1].m128i_i64;
    v137->m128i_i64[1] = 0;
    v137[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v332.m128i_i64[1]) <= 5 )
      goto LABEL_486;
    v139 = (__m128i *)sub_2241490(&v332, "' is '", 6, v138);
    v335 = &v337;
    if ( (__m128i *)v139->m128i_i64[0] == &v139[1] )
    {
      v337 = _mm_loadu_si128(v139 + 1);
    }
    else
    {
      v335 = (__m128i *)v139->m128i_i64[0];
      v337.m128i_i64[0] = v139[1].m128i_i64[0];
    }
    v336 = (__int64 *)v139->m128i_i64[1];
    v140 = v336;
    v139->m128i_i64[0] = (__int64)v139[1].m128i_i64;
    v139->m128i_i64[1] = 0;
    v139[1].m128i_i8[0] = 0;
    v141 = (__m128i *)sub_2241490(&v335, v129[59], v129[60], v140);
    v345.m128i_i64[0] = (__int64)&v346;
    if ( (__m128i *)v141->m128i_i64[0] == &v141[1] )
    {
      v346 = _mm_loadu_si128(v141 + 1);
    }
    else
    {
      v345.m128i_i64[0] = v141->m128i_i64[0];
      v346.m128i_i64[0] = v141[1].m128i_i64[0];
    }
    v142 = v141->m128i_i64[1];
    v345.m128i_i64[1] = v142;
    v141->m128i_i64[0] = (__int64)v141[1].m128i_i64;
    v141->m128i_i64[1] = 0;
    v141[1].m128i_i8[0] = 0;
    if ( v345.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL || v345.m128i_i64[1] == 4611686018427387902LL )
      goto LABEL_486;
    v143 = (__m128i *)sub_2241490(&v345, "'\n", 2, v142);
    n[0] = (size_t)&v352;
    if ( (__m128i *)v143->m128i_i64[0] == &v143[1] )
    {
      v352 = _mm_loadu_si128(v143 + 1);
    }
    else
    {
      n[0] = v143->m128i_i64[0];
      v352.m128i_i64[0] = v143[1].m128i_i64[0];
    }
    n[1] = v143->m128i_u64[1];
    v143->m128i_i64[0] = (__int64)v143[1].m128i_i64;
    v143->m128i_i64[1] = 0;
    v143[1].m128i_i8[0] = 0;
    LOWORD(v341) = 260;
    v340[0] = n;
    v144 = *v358;
    sub_1670450((__int64)v343, 1, (__int64)v340);
    sub_16027F0(v144, (__int64)v343);
    if ( (__m128i *)n[0] != &v352 )
      j_j___libc_free_0(n[0], v352.m128i_i64[0] + 1);
    if ( (__m128i *)v345.m128i_i64[0] != &v346 )
      j_j___libc_free_0(v345.m128i_i64[0], v346.m128i_i64[0] + 1);
    if ( v335 != &v337 )
      j_j___libc_free_0(v335, v337.m128i_i64[0] + 1);
    if ( (__m128i *)v332.m128i_i64[0] != &v333 )
      j_j___libc_free_0(v332.m128i_i64[0], v333.m128i_i64[0] + 1);
    if ( v329 != &v331 )
      j_j___libc_free_0(v329, v331.m128i_i64[0] + 1);
    if ( (__m128i *)v327.m128i_i64[0] != &v328 )
      j_j___libc_free_0(v327.m128i_i64[0], v328.m128i_i64[0] + 1);
    if ( (__m128i *)v325.m128i_i64[0] != &v326 )
      j_j___libc_free_0(v325.m128i_i64[0], v326.m128i_i64[0] + 1);
    if ( (__int64 *)v323.m128i_i64[0] != &v324 )
      j_j___libc_free_0(v323.m128i_i64[0], v324 + 1);
  }
  v65 = v357;
  v66 = v358;
  v67 = (size_t)v357[31];
  if ( !v67 )
  {
    v68 = (_QWORD *)v358[31];
    if ( v68 )
    {
      v176 = (_BYTE *)v358[30];
      if ( v176 )
      {
        n[0] = (size_t)&v352;
        sub_1670060((__int64 *)n, v176, (__int64)v68 + (_QWORD)v176);
        v177 = (__m128i *)v65[30];
        v178 = v177;
        if ( (__m128i *)n[0] != &v352 )
        {
          v179 = n[1];
          v180 = (__int64 *)v352.m128i_i64[0];
          if ( v177 == (__m128i *)(v65 + 32) )
          {
            v65[30] = (__int64 *)n[0];
            v65[31] = (__int64 *)v179;
            v65[32] = v180;
          }
          else
          {
            v181 = v65[32];
            v65[30] = (__int64 *)n[0];
            v65[31] = (__int64 *)v179;
            v65[32] = v180;
            if ( v178 )
            {
              n[0] = (size_t)v178;
              v352.m128i_i64[0] = (__int64)v181;
LABEL_250:
              n[1] = 0;
              v178->m128i_i8[0] = 0;
              if ( (__m128i *)n[0] != &v352 )
                j_j___libc_free_0(n[0], v352.m128i_i64[0] + 1);
              v66 = v358;
              goto LABEL_96;
            }
          }
          n[0] = (size_t)&v352;
          v178 = &v352;
          goto LABEL_250;
        }
        v67 = n[1];
        if ( n[1] )
        {
          if ( n[1] == 1 )
            v177->m128i_i8[0] = v352.m128i_i8[0];
          else
            memcpy(v177, &v352, n[1]);
          v67 = n[1];
          v177 = (__m128i *)v65[30];
        }
      }
      else
      {
        v352.m128i_i8[0] = 0;
        n[0] = (size_t)&v352;
        v177 = (__m128i *)v357[30];
      }
      v65[31] = (__int64 *)v67;
      v177->m128i_i8[v67] = 0;
      v178 = (__m128i *)n[0];
      goto LABEL_250;
    }
  }
LABEL_96:
  v352.m128i_i16[0] = 260;
  n[0] = (size_t)(v66 + 30);
  sub_16E1010(v340);
  v352.m128i_i16[0] = 260;
  n[0] = (size_t)(v357 + 30);
  sub_16E1010(v343);
  if ( v358[31] && !(unsigned __int8)sub_16E2B00(v340, v343) )
  {
    v182 = v358;
    v183 = v357;
    sub_8FD6D0((__int64)v303, "Linking two modules of different target triples: ", v358 + 22);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v303[1]) <= 5 )
      goto LABEL_486;
    v185 = (__m128i *)sub_2241490(v303, "' is '", 6, v184);
    v305 = &v307;
    if ( (__m128i *)v185->m128i_i64[0] == &v185[1] )
    {
      v307 = _mm_loadu_si128(v185 + 1);
    }
    else
    {
      v305 = (__m128i *)v185->m128i_i64[0];
      v307.m128i_i64[0] = v185[1].m128i_i64[0];
    }
    v306 = v185->m128i_i64[1];
    v186 = v306;
    v185->m128i_i64[0] = (__int64)v185[1].m128i_i64;
    v185->m128i_i64[1] = 0;
    v185[1].m128i_i8[0] = 0;
    v187 = (__m128i *)sub_2241490(&v305, v182[30], v182[31], v186);
    v308 = &v310;
    if ( (__m128i *)v187->m128i_i64[0] == &v187[1] )
    {
      v310 = _mm_loadu_si128(v187 + 1);
    }
    else
    {
      v308 = (__m128i *)v187->m128i_i64[0];
      v310.m128i_i64[0] = v187[1].m128i_i64[0];
    }
    v188 = v187->m128i_i64[1];
    v309 = v188;
    v187->m128i_i64[0] = (__int64)v187[1].m128i_i64;
    v187->m128i_i64[1] = 0;
    v187[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v309) <= 0xA )
      goto LABEL_486;
    v189 = (__m128i *)sub_2241490(&v308, "' whereas '", 11, v188);
    v311 = &v313;
    if ( (__m128i *)v189->m128i_i64[0] == &v189[1] )
    {
      v313 = _mm_loadu_si128(v189 + 1);
    }
    else
    {
      v311 = (__m128i *)v189->m128i_i64[0];
      v313.m128i_i64[0] = v189[1].m128i_i64[0];
    }
    v312 = v189->m128i_i64[1];
    v190 = v312;
    v189->m128i_i64[0] = (__int64)v189[1].m128i_i64;
    v189->m128i_i64[1] = 0;
    v189[1].m128i_i8[0] = 0;
    v191 = (__m128i *)sub_2241490(&v311, v183[22], v183[23], v190);
    v314 = &v316;
    if ( (__m128i *)v191->m128i_i64[0] == &v191[1] )
    {
      v316 = _mm_loadu_si128(v191 + 1);
    }
    else
    {
      v314 = (__m128i *)v191->m128i_i64[0];
      v316.m128i_i64[0] = v191[1].m128i_i64[0];
    }
    v192 = v191->m128i_i64[1];
    v315 = v192;
    v191->m128i_i64[0] = (__int64)v191[1].m128i_i64;
    v191->m128i_i64[1] = 0;
    v191[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v315) <= 5 )
      goto LABEL_486;
    v193 = (__m128i *)sub_2241490(&v314, "' is '", 6, v192);
    v317 = &v319;
    if ( (__m128i *)v193->m128i_i64[0] == &v193[1] )
    {
      v319 = _mm_loadu_si128(v193 + 1);
    }
    else
    {
      v317 = (__m128i *)v193->m128i_i64[0];
      v319.m128i_i64[0] = v193[1].m128i_i64[0];
    }
    v318 = v193->m128i_i64[1];
    v194 = v318;
    v193->m128i_i64[0] = (__int64)v193[1].m128i_i64;
    v193->m128i_i64[1] = 0;
    v193[1].m128i_i8[0] = 0;
    v195 = (__m128i *)sub_2241490(&v317, v183[30], v183[31], v194);
    v320 = &v322;
    if ( (__m128i *)v195->m128i_i64[0] == &v195[1] )
    {
      v322 = _mm_loadu_si128(v195 + 1);
    }
    else
    {
      v320 = (__m128i *)v195->m128i_i64[0];
      v322.m128i_i64[0] = v195[1].m128i_i64[0];
    }
    v196 = v195->m128i_i64[1];
    v321 = v196;
    v195->m128i_i64[0] = (__int64)v195[1].m128i_i64;
    v195->m128i_i64[1] = 0;
    v195[1].m128i_i8[0] = 0;
    if ( v321 == 0x3FFFFFFFFFFFFFFFLL || v321 == 4611686018427387902LL )
      goto LABEL_486;
    v197 = (__m128i *)sub_2241490(&v320, "'\n", 2, v196);
    n[0] = (size_t)&v352;
    if ( (__m128i *)v197->m128i_i64[0] == &v197[1] )
    {
      v352 = _mm_loadu_si128(v197 + 1);
    }
    else
    {
      n[0] = v197->m128i_i64[0];
      v352.m128i_i64[0] = v197[1].m128i_i64[0];
    }
    n[1] = v197->m128i_u64[1];
    v197->m128i_i64[0] = (__int64)v197[1].m128i_i64;
    v197->m128i_i64[1] = 0;
    v197[1].m128i_i8[0] = 0;
    v337.m128i_i16[0] = 260;
    v335 = (__m128i *)n;
    v283 = *v358;
    sub_1670450((__int64)&v345, 1, (__int64)&v335);
    sub_16027F0(v283, (__int64)&v345);
    if ( (__m128i *)n[0] != &v352 )
      j_j___libc_free_0(n[0], v352.m128i_i64[0] + 1);
    if ( v320 != &v322 )
      j_j___libc_free_0(v320, v322.m128i_i64[0] + 1);
    if ( v317 != &v319 )
      j_j___libc_free_0(v317, v319.m128i_i64[0] + 1);
    if ( v314 != &v316 )
      j_j___libc_free_0(v314, v316.m128i_i64[0] + 1);
    if ( v311 != &v313 )
      j_j___libc_free_0(v311, v313.m128i_i64[0] + 1);
    if ( v308 != &v310 )
      j_j___libc_free_0(v308, v310.m128i_i64[0] + 1);
    if ( v305 != &v307 )
      j_j___libc_free_0(v305, v307.m128i_i64[0] + 1);
    if ( (__int64 *)v303[0] != &v304 )
      j_j___libc_free_0(v303[0], v304 + 1);
  }
  v69 = v357;
  sub_16E2C10(&v345, v340, v343);
  if ( !v345.m128i_i64[0] )
  {
    v352.m128i_i8[0] = 0;
    v119 = 0;
    n[0] = (size_t)&v352;
    v70 = (__m128i *)v69[30];
LABEL_157:
    v69[31] = (__int64 *)v119;
    v70->m128i_i8[v119] = 0;
    v71 = (__m128i *)n[0];
    goto LABEL_102;
  }
  n[0] = (size_t)&v352;
  sub_1670060((__int64 *)n, v345.m128i_i64[0], v345.m128i_i64[0] + v345.m128i_i64[1]);
  v70 = (__m128i *)v69[30];
  v71 = v70;
  if ( (__m128i *)n[0] == &v352 )
  {
    v119 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
        v70->m128i_i8[0] = v352.m128i_i8[0];
      else
        memcpy(v70, &v352, n[1]);
      v119 = n[1];
      v70 = (__m128i *)v69[30];
    }
    goto LABEL_157;
  }
  v72 = n[1];
  v73 = (__int64 *)v352.m128i_i64[0];
  if ( v70 == (__m128i *)(v69 + 32) )
  {
    v69[30] = (__int64 *)n[0];
    v69[31] = (__int64 *)v72;
    v69[32] = v73;
  }
  else
  {
    v74 = v69[32];
    v69[30] = (__int64 *)n[0];
    v69[31] = (__int64 *)v72;
    v69[32] = v73;
    if ( v71 )
    {
      n[0] = (size_t)v71;
      v352.m128i_i64[0] = (__int64)v74;
      goto LABEL_102;
    }
  }
  n[0] = (size_t)&v352;
  v71 = &v352;
LABEL_102:
  n[1] = 0;
  v71->m128i_i8[0] = 0;
  if ( (__m128i *)n[0] != &v352 )
    j_j___libc_free_0(n[0], v352.m128i_i64[0] + 1);
  if ( (__m128i *)v345.m128i_i64[0] != &v346 )
    j_j___libc_free_0(v345.m128i_i64[0], v346.m128i_i64[0] + 1);
  if ( v411 )
    goto LABEL_108;
  v75 = (_QWORD *)v358[12];
  if ( !v75 )
    goto LABEL_108;
  if ( (unsigned int)(v342 - 29) <= 1 )
  {
    sub_8FD6D0((__int64)&v298, ".text\n.balign 2\n.thumb\n", v358 + 11);
  }
  else if ( (unsigned int)(v342 - 1) > 1 )
  {
    v274 = (_BYTE *)v358[11];
    v298 = v300;
    sub_1670170((__int64 *)&v298, v274, (__int64)v75 + (_QWORD)v274);
  }
  else
  {
    sub_8FD6D0((__int64)&v298, ".text\n.balign 4\n.arm\n", v358 + 11);
  }
  v174 = v357;
  v175 = v357[12];
  if ( v175 )
  {
    v262 = v357[11];
    v301[0] = (__int64)v302;
    sub_1670170(v301, v262, (__int64)v175 + (_QWORD)v262);
    if ( v301[1] != 0x3FFFFFFFFFFFFFFFLL )
    {
      sub_2241490(v301, "\n", 1, v263);
      v265 = (__m128i *)sub_2241490(v301, v298, v299, v264);
      n[0] = (size_t)&v352;
      if ( (__m128i *)v265->m128i_i64[0] == &v265[1] )
      {
        v352 = _mm_loadu_si128(v265 + 1);
      }
      else
      {
        n[0] = v265->m128i_i64[0];
        v352.m128i_i64[0] = v265[1].m128i_i64[0];
      }
      n[1] = v265->m128i_u64[1];
      v265->m128i_i64[0] = (__int64)v265[1].m128i_i64;
      v265->m128i_i64[1] = 0;
      v265[1].m128i_i8[0] = 0;
      sub_16702C0(v174, (_BYTE *)n[0], n[1]);
      if ( (__m128i *)n[0] != &v352 )
        j_j___libc_free_0(n[0], v352.m128i_i64[0] + 1);
      if ( (_QWORD *)v301[0] != v302 )
        j_j___libc_free_0(v301[0], v302[0] + 1LL);
      goto LABEL_243;
    }
LABEL_486:
    sub_4262D8((__int64)"basic_string::append");
  }
  sub_16702C0(v357, v298, v299);
LABEL_243:
  if ( v298 != (_BYTE *)v300 )
    j_j___libc_free_0(v298, v300[0] + 1LL);
LABEL_108:
  sub_16755D0((__int64)&v357);
  v76 = v409;
  if ( v409 != (__int64 *)v408 )
  {
    v77 = v409 - 1;
    if ( v408 >= (unsigned __int64)(v409 - 1) )
      goto LABEL_115;
    v78 = (__int64 *)v408;
    do
    {
      v79 = *v78;
      v80 = *v77;
      ++v78;
      --v77;
      *(v78 - 1) = v80;
      v77[1] = v79;
    }
    while ( v77 > v78 );
LABEL_112:
    v76 = v409;
    while ( (__int64 *)v408 != v76 )
    {
      while ( 1 )
      {
        v77 = v76 - 1;
LABEL_115:
        v81 = *(v76 - 1);
        v409 = v77;
        v76 = v77;
        if ( !v389 )
          break;
        v86 = (v389 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
        v87 = &v387[8 * (unsigned __int64)v86];
        v88 = v87[3];
        if ( v88 != v81 )
        {
          v117 = 1;
          while ( v88 != -8 )
          {
            v118 = v117 + 1;
            v86 = (v389 - 1) & (v117 + v86);
            v87 = &v387[8 * (unsigned __int64)v86];
            v88 = v87[3];
            if ( v81 == v88 )
              goto LABEL_121;
            v117 = v118;
          }
          break;
        }
LABEL_121:
        if ( v87 == &v387[8 * (unsigned __int64)v389] )
          break;
        if ( (__int64 *)v408 == v77 )
          goto LABEL_123;
      }
      if ( !v399 )
        goto LABEL_117;
      v109 = (v399 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
      v110 = &v397[8 * (unsigned __int64)v109];
      v111 = v110[3];
      if ( v111 != v81 )
      {
        v254 = 1;
        while ( v111 != -8 )
        {
          v255 = v254 + 1;
          v109 = (v399 - 1) & (v254 + v109);
          v110 = &v397[8 * (unsigned __int64)v109];
          v111 = v110[3];
          if ( v81 == v111 )
            goto LABEL_141;
          v254 = v255;
        }
LABEL_117:
        sub_1B79660(v415, v81, v77);
        if ( v414 )
        {
          v82 = v413;
          v413 = 0;
          *a1 = v82 | 1;
          goto LABEL_136;
        }
        goto LABEL_112;
      }
LABEL_141:
      if ( v110 == &v397[8 * (unsigned __int64)v399] )
        goto LABEL_117;
    }
  }
LABEL_123:
  v412 = 1;
  sub_1B795C0(v415, 8);
  v89 = sub_16327A0((__int64)v358);
  v90 = v358;
  v297 = (__int64 *)v89;
  v91 = (__int64 *)v358[10];
  v293 = v358 + 9;
  if ( v358 + 9 == v91 )
    goto LABEL_131;
  do
  {
    while ( 1 )
    {
      if ( v91 != v297 )
      {
        v92 = v357;
        v93 = (const void *)sub_161F640((__int64)v91);
        v95 = sub_1632440((__int64)v92, v93, v94);
        v96 = (_QWORD *)sub_161F640((__int64)v91);
        if ( v97 != 16 || *v96 ^ 0x6E6E612E6D76766ELL | v96[1] ^ 0x736E6F697461746FLL )
        {
          v98 = sub_161F520((__int64)v91);
          if ( v98 )
          {
            v99 = 0;
            do
            {
              v100 = v99++;
              v101 = sub_161F530((__int64)v91, v100);
              v102 = sub_1B79620(v415, v101);
              sub_1623CA0(v95, v102);
            }
            while ( v98 != v99 );
          }
          goto LABEL_129;
        }
        v112 = sub_161F520((__int64)v91);
        if ( v112 )
          break;
      }
LABEL_129:
      v91 = (__int64 *)v91[1];
      if ( v293 == v91 )
        goto LABEL_130;
    }
    for ( j = 0; j != v112; ++j )
    {
      v114 = sub_161F530((__int64)v91, j);
      v115 = sub_1B79620(v415, v114);
      v116 = *(unsigned int *)(v115 + 8);
      if ( !(_DWORD)v116 || *(_QWORD *)(v115 - 8 * v116) )
        sub_1623CA0(v95, v115);
    }
    v91 = (__int64 *)v91[1];
  }
  while ( v293 != v91 );
LABEL_130:
  v90 = v358;
LABEL_131:
  v103 = sub_16327A0((__int64)v90);
  if ( !v103 )
  {
LABEL_135:
    *a1 = 1;
    goto LABEL_136;
  }
  v308 = (__m128i *)sub_16329D0((__int64)v357);
  v104 = sub_161F520((__int64)v308);
  if ( !v104 )
  {
    v105 = sub_161F520(v103);
    if ( v105 )
    {
      do
      {
        v106 = v104;
        v107 = (__int64)v308;
        ++v104;
        v108 = sub_161F530(v103, v106);
        sub_1623CA0(v107, v108);
      }
      while ( v105 != v104 );
    }
    goto LABEL_135;
  }
  v329 = 0;
  v145 = &v346;
  v330 = 0;
  v331.m128i_i64[0] = 0;
  v331.m128i_i32[2] = 0;
  v345.m128i_i64[0] = 0;
  v345.m128i_i64[1] = 1;
  do
  {
    v145->m128i_i64[0] = -8;
    v145 = (__m128i *)((char *)v145 + 8);
  }
  while ( v145 != (__m128i *)&v348 );
  v146 = 0;
  v348 = v350;
  v349 = 0x1000000000LL;
  v147 = sub_161F520((__int64)v308);
  if ( v147 )
  {
    v294 = v103;
    v148 = v147;
    while ( 1 )
    {
      while ( 1 )
      {
        v153 = sub_161F530((__int64)v308, v146);
        v154 = *(unsigned int *)(v153 + 8);
        v155 = *(_QWORD *)(*(_QWORD *)(v153 - 8 * v154) + 136LL);
        v156 = *(_DWORD *)(v155 + 32) <= 0x40u;
        v157 = *(_QWORD **)(v155 + 24);
        if ( !v156 )
          v157 = (_QWORD *)*v157;
        if ( v157 != (_QWORD *)3 )
          break;
        ++v146;
        n[0] = *(_QWORD *)(v153 + 8 * (2 - v154));
        sub_1677C60((__int64)&v345, n);
        if ( v148 == v146 )
        {
LABEL_222:
          v103 = v294;
          goto LABEL_223;
        }
      }
      v149 = *(_QWORD *)(v153 + 8 * (1 - v154));
      if ( !v331.m128i_i32[2] )
        break;
      LODWORD(v150) = (v331.m128i_i32[2] - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
      v151 = v330 + 24LL * (unsigned int)v150;
      v152 = *(_QWORD *)v151;
      if ( v149 != *(_QWORD *)v151 )
      {
        v256 = 1;
        v257 = 0;
        while ( v152 != -8 )
        {
          if ( !v257 && v152 == -16 )
            v257 = v151;
          v150 = (v331.m128i_i32[2] - 1) & (unsigned int)(v150 + v256);
          v151 = v330 + 24 * v150;
          v152 = *(_QWORD *)v151;
          if ( v149 == *(_QWORD *)v151 )
            goto LABEL_217;
          ++v256;
        }
        if ( v257 )
          v151 = v257;
        v329 = (__m128i *)((char *)v329 + 1);
        v258 = v331.m128i_i32[0] + 1;
        if ( 4 * (v331.m128i_i32[0] + 1) < (unsigned int)(3 * v331.m128i_i32[2]) )
        {
          if ( v331.m128i_i32[2] - v331.m128i_i32[1] - v258 <= (unsigned __int32)v331.m128i_i32[2] >> 3 )
          {
            sub_1672CC0((__int64)&v329, v331.m128i_i32[2]);
            if ( !v331.m128i_i32[2] )
              goto LABEL_502;
            v270 = 1;
            LODWORD(v271) = (v331.m128i_i32[2] - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
            v258 = v331.m128i_i32[0] + 1;
            v272 = 0;
            v151 = v330 + 24LL * (unsigned int)v271;
            v273 = *(_QWORD *)v151;
            if ( v149 != *(_QWORD *)v151 )
            {
              while ( v273 != -8 )
              {
                if ( !v272 && v273 == -16 )
                  v272 = v151;
                v271 = (v331.m128i_i32[2] - 1) & (unsigned int)(v271 + v270);
                v151 = v330 + 24 * v271;
                v273 = *(_QWORD *)v151;
                if ( v149 == *(_QWORD *)v151 )
                  goto LABEL_378;
                ++v270;
              }
              if ( v272 )
                v151 = v272;
            }
          }
          goto LABEL_378;
        }
LABEL_410:
        sub_1672CC0((__int64)&v329, 2 * v331.m128i_i32[2]);
        if ( !v331.m128i_i32[2] )
          goto LABEL_502;
        v258 = v331.m128i_i32[0] + 1;
        LODWORD(v266) = (v331.m128i_i32[2] - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
        v151 = v330 + 24LL * (unsigned int)v266;
        v267 = *(_QWORD *)v151;
        if ( v149 != *(_QWORD *)v151 )
        {
          v268 = 1;
          v269 = 0;
          while ( v267 != -8 )
          {
            if ( !v269 && v267 == -16 )
              v269 = v151;
            v266 = (v331.m128i_i32[2] - 1) & (unsigned int)(v266 + v268);
            v151 = v330 + 24 * v266;
            v267 = *(_QWORD *)v151;
            if ( v149 == *(_QWORD *)v151 )
              goto LABEL_378;
            ++v268;
          }
          if ( v269 )
            v151 = v269;
        }
LABEL_378:
        v331.m128i_i32[0] = v258;
        if ( *(_QWORD *)v151 != -8 )
          --v331.m128i_i32[1];
        *(_QWORD *)v151 = v149;
        *(_QWORD *)(v151 + 8) = 0;
        *(_DWORD *)(v151 + 16) = 0;
      }
LABEL_217:
      *(_DWORD *)(v151 + 16) = v146++;
      *(_QWORD *)(v151 + 8) = v153;
      if ( v148 == v146 )
        goto LABEL_222;
    }
    v329 = (__m128i *)((char *)v329 + 1);
    goto LABEL_410;
  }
LABEL_223:
  v158 = sub_161F520(v103);
  if ( v158 )
  {
    v159 = 0;
    v160 = v158;
    while ( 1 )
    {
      v161 = sub_161F530(v103, v159);
      v162 = v331.m128i_u32[2];
      v311 = (__m128i *)v161;
      v163 = *(unsigned int *)(v161 + 8);
      v164 = *(_QWORD *)(*(_QWORD *)(v161 - 8 * v163) + 136LL);
      v165 = *(_QWORD *)(v161 + 8 * (1 - v163));
      v166 = 0;
      v314 = (__m128i *)v165;
      if ( v331.m128i_i32[2] )
      {
        v167 = (v331.m128i_i32[2] - 1) & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
        v168 = v330 + 24LL * v167;
        v169 = *(_QWORD *)v168;
        if ( v165 == *(_QWORD *)v168 )
        {
LABEL_227:
          v166 = *(__m128i **)(v168 + 8);
          v162 = *(_DWORD *)(v168 + 16);
        }
        else
        {
          v259 = 1;
          while ( v169 != -8 )
          {
            v280 = v259 + 1;
            v167 = (v331.m128i_i32[2] - 1) & (v167 + v259);
            v168 = v330 + 24LL * v167;
            v169 = *(_QWORD *)v168;
            if ( v165 == *(_QWORD *)v168 )
              goto LABEL_227;
            v259 = v280;
          }
          v162 = 0;
          v166 = 0;
        }
      }
      v317 = v166;
      LODWORD(v305) = v162;
      v170 = *(_QWORD **)(v164 + 24);
      if ( *(_DWORD *)(v164 + 32) > 0x40u )
        v170 = (_QWORD *)*v170;
      v171 = (int)v170;
      if ( (_DWORD)v170 == 3 )
        break;
      if ( !v166 )
      {
        v198 = sub_161F520((__int64)v308);
        v199 = (__int64)v311;
        v200 = v198;
        v201 = sub_1672E90((__int64)&v329, (__int64 *)&v314);
        v201[1] = v199;
        *((_DWORD *)v201 + 4) = v200;
LABEL_289:
        sub_1623CA0((__int64)v308, (__int64)v311);
        goto LABEL_290;
      }
      v172 = *(_QWORD *)(v166->m128i_i64[-v166->m128i_u32[2]] + 136);
      v173 = *(_QWORD **)(v172 + 24);
      if ( *(_DWORD *)(v172 + 32) > 0x40u )
        v173 = (_QWORD *)*v173;
      v333.m128i_i64[0] = (__int64)&v311;
      v332.m128i_i64[0] = (__int64)&v308;
      v332.m128i_i64[1] = (__int64)&v305;
      v333.m128i_i64[1] = (__int64)&v329;
      v334 = (__int64 *)&v314;
      if ( (_DWORD)v173 == 4 )
      {
        if ( v171 == 4
          && *(_QWORD *)(v161 + 8 * (2LL - *(unsigned int *)(v161 + 8))) != v166->m128i_i64[2LL - v166->m128i_u32[2]] )
        {
          v335 = (__m128i *)"': IDs have conflicting override values";
          v337.m128i_i16[0] = 259;
          v250.m128i_i64[0] = sub_161E970(v165);
          v325 = v250;
          v327.m128i_i64[0] = (__int64)"linking module flags '";
          v327.m128i_i64[1] = (__int64)&v325;
          v251 = v337.m128i_i8[0];
          v328.m128i_i16[0] = 1283;
          if ( !v337.m128i_i8[0] )
            goto LABEL_357;
          if ( v337.m128i_i8[0] != 1 )
          {
LABEL_393:
            m128i_i64 = v335->m128i_i64;
            if ( v337.m128i_i8[1] != 1 )
            {
              m128i_i64 = &v335;
              v251 = 2;
            }
            n[1] = (size_t)m128i_i64;
            n[0] = (size_t)&v327;
            v352.m128i_i8[0] = 2;
            v352.m128i_i8[1] = v251;
            goto LABEL_358;
          }
LABEL_445:
          *(__m128i *)n = _mm_load_si128(&v327);
          v352.m128i_i64[0] = v328.m128i_i64[0];
LABEL_358:
          v252 = sub_16BCA90();
          sub_1670110(a1, (__int64)n, v252, v253);
          v202 = v348;
          goto LABEL_359;
        }
      }
      else
      {
        if ( v171 != 4 )
        {
          if ( v171 == (_DWORD)v173 )
          {
            v336 = (__int64 *)&v314;
            v335 = (__m128i *)&v317;
            v337.m128i_i64[0] = (__int64)&v357;
            v337.m128i_i64[1] = (__int64)&v308;
            v338 = &v305;
            v339 = &v329;
            switch ( v171 )
            {
              case 1:
                if ( *(_QWORD *)(v161 + 8 * (2LL - *(unsigned int *)(v161 + 8))) == v166->m128i_i64[2LL - v166->m128i_u32[2]] )
                  goto LABEL_290;
                v327.m128i_i64[0] = (__int64)"': IDs have conflicting values";
                v328.m128i_i16[0] = 259;
                v247.m128i_i64[0] = sub_161E970(v165);
                v323 = v247;
                v325.m128i_i64[0] = (__int64)"linking module flags '";
                v325.m128i_i64[1] = (__int64)&v323;
                v248 = v328.m128i_i8[0];
                v326.m128i_i16[0] = 1283;
                if ( !v328.m128i_i8[0] )
                  goto LABEL_357;
                if ( v328.m128i_i8[0] == 1 )
                {
                  *(__m128i *)n = _mm_load_si128(&v325);
                  v352.m128i_i64[0] = v326.m128i_i64[0];
                }
                else
                {
                  v249 = (__m128i *)v327.m128i_i64[0];
                  if ( v328.m128i_i8[1] != 1 )
                  {
                    v249 = &v327;
                    v248 = 2;
                  }
                  n[1] = (size_t)v249;
                  n[0] = (size_t)&v325;
                  v352.m128i_i8[0] = 2;
                  v352.m128i_i8[1] = v248;
                }
                break;
              case 2:
                if ( *(_QWORD *)(v161 + 8 * (2LL - *(unsigned int *)(v161 + 8))) != v166->m128i_i64[2LL - v166->m128i_u32[2]] )
                {
                  v325.m128i_i64[0] = (__int64)"': IDs have conflicting values";
                  v326.m128i_i16[0] = 259;
                  v320 = (__m128i *)sub_161E970(v165);
                  v323.m128i_i64[0] = (__int64)"linking module flags '";
                  v323.m128i_i64[1] = (__int64)&v320;
                  v225 = v326.m128i_i8[0];
                  v321 = v226;
                  LOWORD(v324) = 1283;
                  if ( v326.m128i_i8[0] )
                  {
                    if ( v326.m128i_i8[0] == 1 )
                    {
                      v327 = _mm_load_si128(&v323);
                      v328.m128i_i64[0] = v324;
                    }
                    else
                    {
                      v227 = (__m128i *)v325.m128i_i64[0];
                      if ( v326.m128i_i8[1] != 1 )
                      {
                        v227 = &v325;
                        v225 = 2;
                      }
                      v327.m128i_i64[1] = (__int64)v227;
                      v327.m128i_i64[0] = (__int64)&v323;
                      v328.m128i_i8[0] = 2;
                      v328.m128i_i8[1] = v225;
                    }
                  }
                  else
                  {
                    v328.m128i_i16[0] = 256;
                  }
                  v228 = *v358;
                  sub_1670450((__int64)n, 1, (__int64)&v327);
                  sub_16027F0(v228, (__int64)n);
                }
                goto LABEL_290;
              case 5:
                v229 = v166->m128i_i64[2LL - v166->m128i_u32[2]];
                v230 = 8;
                v231 = *(_QWORD *)(v161 + 8 * (2LL - *(unsigned int *)(v161 + 8)));
                n[0] = (size_t)&v352;
                v232 = 0;
                n[1] = 0x800000000LL;
                v233 = *(unsigned int *)(v229 + 8);
                if ( (unsigned int)(*(_DWORD *)(v231 + 8) + *(_DWORD *)(v229 + 8)) > 8uLL )
                {
                  v285 = v231;
                  v288 = v229;
                  sub_16CD150(n, &v352, (unsigned int)(*(_DWORD *)(v231 + 8) + *(_DWORD *)(v229 + 8)), 8);
                  v229 = v288;
                  v232 = LODWORD(n[1]);
                  v231 = v285;
                  v233 = *(unsigned int *)(v288 + 8);
                  v230 = HIDWORD(n[1]) - (unsigned __int64)LODWORD(n[1]);
                }
                v234 = v233;
                v235 = (_QWORD *)(v229 - 8 * v233);
                v236 = (8 * v233) >> 3;
                if ( v230 < v236 )
                {
                  v284 = v231;
                  v287 = v229;
                  v289 = (8 * v233) >> 3;
                  sub_16CD150(n, &v352, v232 + v236, 8);
                  v232 = LODWORD(n[1]);
                  v231 = v284;
                  v229 = v287;
                  LODWORD(v236) = v289;
                }
                v237 = (_QWORD *)(n[0] + 8 * v232);
                if ( (_QWORD *)v229 != v235 )
                {
                  v238 = &v237[v234];
                  do
                  {
                    if ( v237 )
                      *v237 = *v235;
                    ++v237;
                    ++v235;
                  }
                  while ( v237 != v238 );
                  LODWORD(v232) = n[1];
                }
                LODWORD(n[1]) = v236 + v232;
                v239 = (unsigned int)(v236 + v232);
                v240 = 8LL * *(unsigned int *)(v231 + 8);
                v241 = (_QWORD *)(v231 - v240);
                v242 = v240 >> 3;
                if ( v240 >> 3 > (unsigned __int64)HIDWORD(n[1]) - v239 )
                {
                  v286 = v231;
                  sub_16CD150(n, &v352, v242 + v239, 8);
                  v239 = LODWORD(n[1]);
                  v231 = v286;
                  v242 = v240 >> 3;
                }
                v243 = (__int64 *)n[0];
                v244 = (_QWORD *)(n[0] + 8 * v239);
                if ( (_QWORD *)v231 != v241 )
                {
                  v245 = &v244[(unsigned __int64)v240 / 8];
                  do
                  {
                    if ( v244 )
                      *v244 = *v241;
                    ++v244;
                    ++v241;
                  }
                  while ( v244 != v245 );
                  v243 = (__int64 *)n[0];
                  LODWORD(v239) = n[1];
                }
                LODWORD(n[1]) = v242 + v239;
                v246 = sub_1627350(*v357, v243, (__int64 *)(unsigned int)(v242 + v239), 0, 1);
                sub_1673340((__int64)&v335, v246);
                if ( (__m128i *)n[0] != &v352 )
                  _libc_free(n[0]);
                goto LABEL_290;
              case 6:
                v221 = (__int64 **)&v352;
                n[0] = 0;
                n[1] = 1;
                do
                  *v221++ = (__int64 *)-4LL;
                while ( v221 != &v354 );
                v355 = 0x1000000000LL;
                v354 = (__int64 *)v356;
                v222 = v166->m128i_i64[2LL - v166->m128i_u32[2]];
                v223 = *(_QWORD *)(v161 + 8 * (2LL - *(unsigned int *)(v161 + 8)));
                sub_1678340((__int64)n, (__int64 *)(v222 - 8LL * *(unsigned int *)(v222 + 8)), (__int64 *)v222);
                sub_1678340((__int64)n, (__int64 *)(v223 - 8LL * *(unsigned int *)(v223 + 8)), (__int64 *)v223);
                v224 = sub_1627350(*v357, v354, (__int64 *)(unsigned int)v355, 0, 1);
                sub_1673340((__int64)&v335, v224);
                if ( v354 != (__int64 *)v356 )
                  _libc_free((unsigned __int64)v354);
                if ( (n[1] & 1) == 0 )
                  j___libc_free_0(v352.m128i_i64[0]);
                goto LABEL_290;
              case 7:
                v217 = *(_QWORD *)(v166->m128i_i64[2LL - v166->m128i_u32[2]] + 136);
                v218 = *(_QWORD *)(*(_QWORD *)(v161 + 8 * (2LL - *(unsigned int *)(v161 + 8))) + 136LL);
                v219 = *(_QWORD **)(v218 + 24);
                if ( *(_DWORD *)(v218 + 32) > 0x40u )
                  v219 = (_QWORD *)*v219;
                v220 = *(_QWORD **)(v217 + 24);
                if ( *(_DWORD *)(v217 + 32) > 0x40u )
                  v220 = (_QWORD *)*v220;
                if ( v220 < v219 )
                  sub_16730E0((__int64)&v332);
                goto LABEL_290;
              default:
                goto LABEL_290;
            }
            goto LABEL_358;
          }
          v335 = (__m128i *)"': IDs have conflicting behaviors";
          v337.m128i_i16[0] = 259;
          v279.m128i_i64[0] = sub_161E970(v165);
          v325 = v279;
          v327.m128i_i64[0] = (__int64)"linking module flags '";
          v327.m128i_i64[1] = (__int64)&v325;
          v251 = v337.m128i_i8[0];
          v328.m128i_i16[0] = 1283;
          if ( v337.m128i_i8[0] )
          {
            if ( v337.m128i_i8[0] != 1 )
              goto LABEL_393;
            goto LABEL_445;
          }
LABEL_357:
          v352.m128i_i16[0] = 256;
          goto LABEL_358;
        }
        sub_1623BA0((__int64)v308, v162, v161);
        v260 = (__int64 *)v333.m128i_i64[0];
        sub_1672E90(v333.m128i_i64[1], v334)[1] = *v260;
      }
LABEL_290:
      if ( v160 == ++v159 )
        goto LABEL_291;
    }
    n[0] = *(_QWORD *)(v161 + 8 * (2 - v163));
    if ( !(unsigned __int8)sub_1677C60((__int64)&v345, n) )
      goto LABEL_290;
    goto LABEL_289;
  }
LABEL_291:
  v202 = v348;
  if ( !(_DWORD)v349 )
    goto LABEL_448;
  v203 = v348;
  v295 = v331.m128i_i32[2] - 1;
  while ( 2 )
  {
    v209 = *(_QWORD *)(*(_QWORD *)v203 - 8LL * *(unsigned int *)(*(_QWORD *)v203 + 8LL));
    if ( !v331.m128i_i32[2] )
    {
      v329 = (__m128i *)((char *)v329 + 1);
      goto LABEL_299;
    }
    v204 = ((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4);
    v205 = v204 & v295;
    v206 = (__int64 *)(v330 + 24LL * (v204 & v295));
    v207 = *v206;
    if ( v209 != *v206 )
    {
      v292 = 1;
      v212 = 0;
      while ( v207 != -8 )
      {
        if ( !v212 && v207 == -16 )
          v212 = v206;
        v205 = v295 & (v292 + v205);
        v206 = (__int64 *)(v330 + 24LL * v205);
        v207 = *v206;
        if ( v209 == *v206 )
          goto LABEL_294;
        ++v292;
      }
      if ( !v212 )
        v212 = v206;
      v329 = (__m128i *)((char *)v329 + 1);
      v210 = v331.m128i_i32[0] + 1;
      if ( 4 * (v331.m128i_i32[0] + 1) >= (unsigned int)(3 * v331.m128i_i32[2]) )
      {
LABEL_299:
        sub_1672CC0((__int64)&v329, 2 * v331.m128i_i32[2]);
        if ( !v331.m128i_i32[2] )
          goto LABEL_502;
        v210 = v331.m128i_i32[0] + 1;
        v211 = (v331.m128i_i32[2] - 1) & (((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4));
        v212 = (__int64 *)(v330 + 24LL * v211);
        v213 = *v212;
        if ( v209 != *v212 )
        {
          v281 = 1;
          v282 = 0;
          while ( v213 != -8 )
          {
            if ( !v282 && v213 == -16 )
              v282 = v212;
            v211 = (v331.m128i_i32[2] - 1) & (v281 + v211);
            v212 = (__int64 *)(v330 + 24LL * v211);
            v213 = *v212;
            if ( v209 == *v212 )
              goto LABEL_301;
            ++v281;
          }
          if ( v282 )
            v212 = v282;
        }
      }
      else if ( v331.m128i_i32[2] - v331.m128i_i32[1] - v210 <= (unsigned __int32)v331.m128i_i32[2] >> 3 )
      {
        sub_1672CC0((__int64)&v329, v331.m128i_i32[2]);
        if ( v331.m128i_i32[2] )
        {
          v275 = 0;
          v276 = (v331.m128i_i32[2] - 1) & v204;
          v277 = 1;
          v210 = v331.m128i_i32[0] + 1;
          v212 = (__int64 *)(v330 + 24LL * v276);
          v278 = *v212;
          if ( v209 != *v212 )
          {
            while ( v278 != -8 )
            {
              if ( v278 == -16 && !v275 )
                v275 = v212;
              v276 = (v331.m128i_i32[2] - 1) & (v277 + v276);
              v212 = (__int64 *)(v330 + 24LL * v276);
              v278 = *v212;
              if ( v209 == *v212 )
                goto LABEL_301;
              ++v277;
            }
            if ( v275 )
              v212 = v275;
          }
          goto LABEL_301;
        }
LABEL_502:
        ++v331.m128i_i32[0];
        BUG();
      }
LABEL_301:
      v331.m128i_i32[0] = v210;
      if ( *v212 != -8 )
        --v331.m128i_i32[1];
      *v212 = v209;
      v212[1] = 0;
      *((_DWORD *)v212 + 4) = 0;
LABEL_304:
      v335 = (__m128i *)"': does not have the required value";
      v337.m128i_i16[0] = 259;
      v214.m128i_i64[0] = sub_161E970(v209);
      v327 = v214;
      v332.m128i_i64[0] = (__int64)"linking module flags '";
      v332.m128i_i64[1] = (__int64)&v327;
      v215 = v337.m128i_i8[0];
      v333.m128i_i16[0] = 1283;
      if ( v337.m128i_i8[0] )
      {
        if ( v337.m128i_i8[0] == 1 )
        {
          *(__m128i *)n = _mm_load_si128(&v332);
          v352.m128i_i64[0] = v333.m128i_i64[0];
        }
        else
        {
          v216 = v335->m128i_i64;
          if ( v337.m128i_i8[1] != 1 )
          {
            v216 = &v335;
            v215 = 2;
          }
          n[1] = (size_t)v216;
          n[0] = (size_t)&v332;
          v352.m128i_i8[0] = 2;
          v352.m128i_i8[1] = v215;
        }
        goto LABEL_358;
      }
      goto LABEL_357;
    }
LABEL_294:
    v208 = v206[1];
    if ( !v208
      || *(_QWORD *)(*(_QWORD *)v203 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)v203 + 8LL))) != *(_QWORD *)(v208 + 8 * (2LL - *(unsigned int *)(v208 + 8))) )
    {
      goto LABEL_304;
    }
    v203 += 8;
    if ( &v348[8 * (unsigned int)(v349 - 1) + 8] != v203 )
      continue;
    break;
  }
  v202 = v348;
LABEL_448:
  *a1 = 1;
LABEL_359:
  if ( v202 != v350 )
    _libc_free((unsigned __int64)v202);
  if ( (v345.m128i_i8[8] & 1) == 0 )
    j___libc_free_0(v346.m128i_i64[0]);
  j___libc_free_0(v330);
LABEL_136:
  if ( (__int64 *)v343[0] != &v344 )
    j_j___libc_free_0(v343[0], v344 + 1);
  if ( (__int64 *)v340[0] != &v341 )
    j_j___libc_free_0(v340[0], v341 + 1);
LABEL_28:
  sub_16052B0(*(__int64 ***)a2);
  v45 = v385;
  v46 = *(unsigned int *)(v385 + 24);
  if ( (_DWORD)v46 )
  {
    v47 = *(_QWORD **)(v385 + 8);
    v48 = &v47[2 * v46];
    do
    {
      if ( *v47 != -8 && *v47 != -4 )
      {
        v49 = v47[1];
        if ( v49 )
          sub_161E7C0((__int64)(v47 + 1), v49);
      }
      v47 += 2;
    }
    while ( v48 != v47 );
  }
  j___libc_free_0(*(_QWORD *)(v45 + 8));
  ++*(_QWORD *)v45;
  *(_QWORD *)(v45 + 16) = 0;
  *(_QWORD *)(v45 + 8) = 0;
  *(_DWORD *)(v45 + 24) = 0;
  ++v390;
  v50 = *(_QWORD **)(v45 + 8);
  *(_QWORD *)(v45 + 8) = v391;
  v391 = v50;
  LODWORD(v50) = *(_DWORD *)(v45 + 16);
  *(_DWORD *)(v45 + 16) = v392;
  LODWORD(v392) = (_DWORD)v50;
  LODWORD(v50) = *(_DWORD *)(v45 + 20);
  *(_DWORD *)(v45 + 20) = HIDWORD(v392);
  HIDWORD(v392) = (_DWORD)v50;
  LODWORD(v50) = *(_DWORD *)(v45 + 24);
  *(_DWORD *)(v45 + 24) = v393;
  v393 = (unsigned int)v50;
  sub_1B75110(v415);
  if ( v414 && ((v413 & 1) != 0 || (v413 & 0xFFFFFFFFFFFFFFFELL) != 0) )
    sub_16BCAE0(&v413);
  if ( v408 )
    j_j___libc_free_0(v408, v410 - v408);
  j___libc_free_0(v405);
  if ( v402 )
  {
    if ( v401 )
    {
      v123 = v400;
      v124 = &v400[2 * v401];
      do
      {
        if ( *v123 != -4 && *v123 != -8 )
        {
          v125 = v123[1];
          if ( v125 )
            sub_161E7C0((__int64)(v123 + 1), v125);
        }
        v123 += 2;
      }
      while ( v124 != v123 );
    }
    j___libc_free_0(v400);
  }
  if ( v399 )
  {
    v51 = v397;
    v345.m128i_i64[1] = 2;
    v346.m128i_i64[0] = 0;
    v52 = &v397[8 * (unsigned __int64)v399];
    v346.m128i_i64[1] = -8;
    v345.m128i_i64[0] = (__int64)&unk_49E6B50;
    v53 = -8;
    v347 = 0;
    n[1] = 2;
    v352.m128i_i64[0] = 0;
    v352.m128i_i64[1] = -16;
    n[0] = (size_t)&unk_49E6B50;
    i = 0;
    while ( 1 )
    {
      v54 = v51[3];
      if ( v53 != v54 )
      {
        v53 = v352.m128i_i64[1];
        if ( v54 != v352.m128i_i64[1] )
        {
          v55 = v51[7];
          if ( v55 != 0 && v55 != -8 && v55 != -16 )
          {
            sub_1649B30(v51 + 5);
            v54 = v51[3];
          }
          v53 = v54;
        }
      }
      *v51 = &unk_49EE2B0;
      if ( v53 != -8 && v53 != 0 && v53 != -16 )
        sub_1649B30(v51 + 1);
      v51 += 8;
      if ( v52 == v51 )
        break;
      v53 = v346.m128i_i64[1];
    }
    n[0] = (size_t)&unk_49EE2B0;
    if ( v352.m128i_i64[1] != 0 && v352.m128i_i64[1] != -8 && v352.m128i_i64[1] != -16 )
      sub_1649B30(&n[1]);
    v345.m128i_i64[0] = (__int64)&unk_49EE2B0;
    if ( v346.m128i_i64[1] != 0 && v346.m128i_i64[1] != -8 && v346.m128i_i64[1] != -16 )
      sub_1649B30(&v345.m128i_i64[1]);
  }
  j___libc_free_0(v397);
  if ( v394 )
  {
    if ( v393 )
    {
      v120 = v391;
      v121 = &v391[2 * v393];
      do
      {
        if ( *v120 != -4 && *v120 != -8 )
        {
          v122 = v120[1];
          if ( v122 )
            sub_161E7C0((__int64)(v120 + 1), v122);
        }
        v120 += 2;
      }
      while ( v121 != v120 );
    }
    j___libc_free_0(v391);
  }
  if ( v389 )
  {
    v56 = v387;
    v345.m128i_i64[1] = 2;
    v346.m128i_i64[0] = 0;
    v57 = &v387[8 * (unsigned __int64)v389];
    v346.m128i_i64[1] = -8;
    v345.m128i_i64[0] = (__int64)&unk_49E6B50;
    v58 = -8;
    v347 = 0;
    n[1] = 2;
    v352.m128i_i64[0] = 0;
    v352.m128i_i64[1] = -16;
    n[0] = (size_t)&unk_49E6B50;
    i = 0;
    while ( 1 )
    {
      v59 = v56[3];
      if ( v58 != v59 )
      {
        v58 = v352.m128i_i64[1];
        if ( v59 != v352.m128i_i64[1] )
        {
          v60 = v56[7];
          if ( v60 != 0 && v60 != -8 && v60 != -16 )
          {
            sub_1649B30(v56 + 5);
            v59 = v56[3];
          }
          v58 = v59;
        }
      }
      *v56 = &unk_49EE2B0;
      if ( v58 != 0 && v58 != -8 && v58 != -16 )
        sub_1649B30(v56 + 1);
      v56 += 8;
      if ( v57 == v56 )
        break;
      v58 = v346.m128i_i64[1];
    }
    n[0] = (size_t)&unk_49EE2B0;
    if ( v352.m128i_i64[1] != 0 && v352.m128i_i64[1] != -8 && v352.m128i_i64[1] != -16 )
      sub_1649B30(&n[1]);
    v345.m128i_i64[0] = (__int64)&unk_49EE2B0;
    if ( v346.m128i_i64[1] != 0 && v346.m128i_i64[1] != -8 && v346.m128i_i64[1] != -16 )
      sub_1649B30(&v345.m128i_i64[1]);
  }
  j___libc_free_0(v387);
  v362[0] = off_49EE4A0;
  if ( v377 != v376 )
    _libc_free((unsigned __int64)v377);
  if ( v372 != v374 )
    _libc_free((unsigned __int64)v372);
  if ( v369 != v371 )
    _libc_free((unsigned __int64)v369);
  if ( v366 != v368 )
    _libc_free((unsigned __int64)v366);
  j___libc_free_0(v363);
  if ( v360 )
    v360(&v359, &v359, 3);
  v61 = v358;
  if ( v358 )
  {
    sub_1633490((_QWORD **)v358);
    j_j___libc_free_0(v61, 736);
  }
  return a1;
}
