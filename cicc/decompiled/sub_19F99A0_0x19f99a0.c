// Function: sub_19F99A0
// Address: 0x19f99a0
//
__int64 __fastcall sub_19F99A0(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        __m128 a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 *v11; // rdi
  _QWORD *v12; // rax
  int v13; // r8d
  int v14; // r9d
  unsigned int v15; // eax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  __m128i v19; // rax
  unsigned __int64 v20; // rax
  __m128i *v21; // rsi
  __int64 **v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  const __m128i *v25; // rdi
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 *v28; // rsi
  __m128 *v29; // rdx
  const __m128i *v30; // rax
  __int64 *v31; // rax
  unsigned __int64 v32; // rbx
  __int64 v33; // rax
  __m128 *v34; // rcx
  __m128 *v35; // rdx
  const __m128i *v36; // rax
  const __m128i *v37; // rax
  __int64 **v38; // rax
  char *v39; // rax
  __int64 v40; // r12
  __int64 v41; // r14
  int v42; // ebx
  __int64 v43; // r8
  unsigned int v44; // ecx
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rsi
  unsigned int v52; // ecx
  __int64 *v53; // rdx
  __int64 v54; // r8
  unsigned int v55; // esi
  int v56; // ecx
  __int64 v57; // r11
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // rdi
  unsigned int v62; // ecx
  __int64 *v63; // rdx
  __int64 v64; // r8
  __int64 v65; // rax
  __int64 *v66; // r12
  __int64 v67; // rbx
  unsigned __int64 v68; // rax
  __int64 **v69; // rbx
  __int64 *v70; // r12
  unsigned int v71; // r13d
  unsigned int v72; // r8d
  __int64 v73; // rdi
  unsigned int v74; // r9d
  __int64 v75; // rax
  __int64 *v76; // r10
  unsigned int v77; // r10d
  unsigned int v78; // ecx
  __int64 v79; // rax
  __int64 v80; // r11
  __int64 v81; // rdx
  unsigned int v82; // esi
  __int64 **v83; // r14
  __int64 v84; // rcx
  __int64 *v85; // rdi
  int v86; // edx
  __int64 v87; // r9
  int v88; // eax
  int v89; // eax
  int v90; // r11d
  int v91; // eax
  __int64 v92; // rbx
  __m128i v93; // rax
  unsigned __int64 v94; // rbx
  __int64 v95; // rax
  unsigned __int64 v96; // rcx
  unsigned __int64 v97; // rdx
  const __m128i *v98; // rax
  __int8 v99; // r8
  __int64 **v100; // r8
  unsigned __int64 v101; // rbx
  __int64 v102; // rax
  unsigned __int64 v103; // rcx
  unsigned __int64 v104; // rdx
  __int64 **v105; // rax
  char v106; // di
  unsigned __int64 v107; // rax
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rsi
  __int64 v110; // rdi
  __int64 v111; // rax
  __int64 v112; // rbx
  unsigned int v113; // r15d
  int v114; // r8d
  int v115; // r9d
  unsigned int v116; // eax
  __int64 *v117; // rdx
  __int64 v118; // rbx
  int v119; // r9d
  __int64 v120; // r8
  unsigned int v121; // edx
  __int64 v122; // rax
  __int64 v123; // rdi
  unsigned int v124; // eax
  __int64 *v125; // rdx
  __int64 v126; // r12
  char v127; // al
  __int64 *v128; // rcx
  unsigned int v129; // esi
  int v130; // edi
  char v131; // di
  char v132; // al
  bool v133; // al
  char v134; // r8
  __int64 v135; // rax
  int v136; // edi
  unsigned int v137; // esi
  int v138; // edx
  __int64 *v139; // rdx
  unsigned int v140; // esi
  __int64 v141; // rdi
  unsigned int v142; // edx
  __int64 *v143; // rax
  __int64 v144; // rcx
  unsigned __int64 v145; // rbx
  __int64 v146; // r13
  __int64 *v147; // rax
  char v148; // dl
  __int64 v149; // r14
  __int64 *v150; // rax
  __int64 *v151; // rcx
  __int64 *i; // rsi
  unsigned int v153; // esi
  __int64 v154; // rdx
  __int64 v155; // r11
  unsigned int v156; // r9d
  __int64 *v157; // rax
  __int64 v158; // rdi
  int v159; // r11d
  __int64 v160; // r10
  int v161; // edi
  __int64 *v162; // r8
  int v163; // edi
  int v164; // edi
  int v165; // edx
  int v166; // r10d
  int v167; // r11d
  __int64 v168; // r10
  int v169; // edi
  int v170; // edx
  int v171; // r11d
  __int64 *v172; // r9
  int v173; // ebx
  int v174; // edx
  __int64 v175; // rbx
  __int64 v176; // rax
  int v177; // ecx
  int v178; // r8d
  int v179; // r9d
  __int64 v180; // rbx
  unsigned int v181; // eax
  unsigned int v182; // eax
  __int64 v183; // rsi
  __int64 v184; // rax
  __int64 v185; // rsi
  __m128i v186; // rax
  __int64 v187; // rcx
  double v188; // xmm4_8
  double v189; // xmm5_8
  int v190; // edi
  __int64 v191; // r8
  __int64 v192; // rax
  __int64 v195; // rdx
  __int64 v196; // rax
  int v197; // r9d
  __int64 v198; // rbx
  unsigned int v199; // r8d
  unsigned int v200; // edx
  _QWORD *v201; // rdi
  _QWORD *v202; // rax
  __int64 v203; // rdx
  __int64 v204; // rax
  unsigned int v207; // r8d
  __int64 v208; // r12
  _QWORD *v209; // rdx
  _QWORD *v210; // rax
  _QWORD *v211; // rbx
  __int64 v212; // rax
  int v213; // edx
  int v214; // ecx
  unsigned int v215; // edi
  __int64 v216; // rdx
  __int64 v217; // r8
  int v218; // eax
  __int64 v219; // rax
  __int64 v220; // rbx
  __int64 *v221; // rax
  __int64 *v222; // rbx
  __int64 v223; // r14
  __int64 *v224; // r15
  __int64 v225; // r13
  __int64 *v226; // r12
  __int64 v227; // rcx
  __int64 v228; // rax
  int v229; // ecx
  __int64 v230; // rdi
  int v231; // ecx
  unsigned int v232; // edx
  __int64 *v233; // rax
  __int64 v234; // r10
  __int64 v235; // rax
  __int64 v236; // rdx
  int v237; // esi
  __int64 v238; // rcx
  int v239; // esi
  __int64 v240; // r10
  unsigned int v241; // edi
  __int64 *v242; // rcx
  __int64 v243; // r9
  int v244; // ecx
  unsigned int v245; // eax
  __int64 v246; // rdx
  int v247; // ecx
  unsigned __int64 v248; // r11
  unsigned __int64 v249; // r11
  unsigned int v252; // r12d
  _QWORD *v253; // rax
  __int64 v254; // rdx
  __int64 v255; // rbx
  __int64 v256; // r14
  __int64 v257; // rax
  __int64 v258; // rdx
  __int64 v259; // rcx
  int v260; // r8d
  int v261; // r9d
  double v262; // xmm4_8
  double v263; // xmm5_8
  __int64 v264; // r14
  __int64 *v265; // r15
  _QWORD *v266; // rbx
  _QWORD *v267; // r13
  unsigned __int64 v268; // r12
  bool v269; // zf
  __int64 v270; // rax
  double v271; // xmm4_8
  double v272; // xmm5_8
  _QWORD *v273; // rax
  __int64 **v274; // rbx
  __int64 v275; // r12
  __int64 **v276; // rax
  __int64 v277; // rdx
  __int64 v278; // rcx
  __int64 v279; // rbx
  unsigned __int64 v280; // r13
  _QWORD *v281; // rdi
  _QWORD *v282; // r12
  _QWORD *v283; // rax
  __int64 v284; // rbx
  __int64 v285; // rdx
  __int64 v286; // r13
  __int64 v287; // rdx
  __int64 v288; // rdx
  __int64 v289; // rcx
  int v290; // eax
  int v291; // r8d
  __int64 v292; // r10
  __int64 v293; // r14
  char v294; // r13
  __int64 v295; // r12
  _DWORD *v296; // rax
  __int64 v297; // r14
  int v298; // ebx
  char v299; // r8
  __int64 v300; // rax
  int v301; // eax
  char v302; // r8
  int v303; // esi
  int v304; // edx
  unsigned int v305; // esi
  __int64 v306; // rdx
  _QWORD *v307; // rax
  __int64 v308; // r13
  int v309; // ecx
  int v310; // ecx
  unsigned int v311; // edx
  __int64 *v312; // rax
  __int64 v313; // r9
  unsigned int v314; // eax
  char v315; // r10
  __int64 *v316; // r8
  __int64 v317; // r9
  __int64 v318; // rdi
  unsigned int v319; // edx
  unsigned int j; // eax
  int v321; // r8d
  __int64 v322; // rax
  int v323; // edx
  int v324; // r9d
  __int64 v326; // rax
  __int64 v327; // r14
  double v328; // xmm4_8
  double v329; // xmm5_8
  __int64 v330; // rax
  __int64 v331; // rdx
  __int64 v332; // rax
  __int64 *v333; // rax
  __int64 v334; // rdx
  __int64 v335; // rcx
  double v336; // xmm4_8
  double v337; // xmm5_8
  __int64 v338; // rax
  __int64 v339; // rax
  _QWORD *v340; // rdx
  int v341; // eax
  int v342; // r8d
  unsigned __int64 v343; // rdx
  __int64 v344; // rax
  __int64 v345; // r15
  __int64 v346; // r14
  __int64 v347; // rdx
  int v348; // r10d
  __int64 *v349; // rax
  __int64 *v350; // rbx
  int v351; // ecx
  __int64 v352; // rsi
  int v353; // ecx
  unsigned int v354; // edx
  __int64 **v355; // rax
  __int64 *v356; // r8
  int v357; // ecx
  __int64 v358; // rsi
  int v359; // ecx
  unsigned int v360; // edx
  __int64 *v361; // rax
  __int64 v362; // r8
  int v363; // eax
  int v364; // edi
  int v365; // eax
  int v366; // edi
  __int64 *v367; // [rsp+20h] [rbp-540h]
  __int64 v368; // [rsp+28h] [rbp-538h]
  __int64 v369; // [rsp+28h] [rbp-538h]
  __int64 v370; // [rsp+30h] [rbp-530h]
  __int64 v371; // [rsp+30h] [rbp-530h]
  int v372; // [rsp+50h] [rbp-510h]
  __int64 *v373; // [rsp+50h] [rbp-510h]
  __int64 v374; // [rsp+58h] [rbp-508h]
  __int64 *v375; // [rsp+58h] [rbp-508h]
  int v376; // [rsp+58h] [rbp-508h]
  __int64 *v377; // [rsp+58h] [rbp-508h]
  __int64 v378; // [rsp+58h] [rbp-508h]
  __int64 v379; // [rsp+60h] [rbp-500h]
  __int64 v380; // [rsp+60h] [rbp-500h]
  __int64 v381; // [rsp+60h] [rbp-500h]
  char *v382; // [rsp+68h] [rbp-4F8h]
  __int64 v383; // [rsp+68h] [rbp-4F8h]
  __int64 v384; // [rsp+68h] [rbp-4F8h]
  __int64 v385; // [rsp+68h] [rbp-4F8h]
  __int64 **v386; // [rsp+70h] [rbp-4F0h]
  unsigned int v387; // [rsp+70h] [rbp-4F0h]
  unsigned int v388; // [rsp+70h] [rbp-4F0h]
  __int64 v389; // [rsp+70h] [rbp-4F0h]
  __int64 v390; // [rsp+78h] [rbp-4E8h]
  __int64 v391; // [rsp+88h] [rbp-4D8h]
  __int64 v392; // [rsp+88h] [rbp-4D8h]
  __int64 v393; // [rsp+88h] [rbp-4D8h]
  __int64 v394; // [rsp+88h] [rbp-4D8h]
  __int64 v395; // [rsp+88h] [rbp-4D8h]
  __int64 v396; // [rsp+88h] [rbp-4D8h]
  __int64 v397; // [rsp+90h] [rbp-4D0h] BYREF
  __int64 v398; // [rsp+98h] [rbp-4C8h]
  __int64 v399; // [rsp+A0h] [rbp-4C0h]
  __int64 v400; // [rsp+B0h] [rbp-4B0h] BYREF
  __int64 *v401; // [rsp+B8h] [rbp-4A8h]
  __int64 *v402; // [rsp+C0h] [rbp-4A0h]
  unsigned int v403; // [rsp+C8h] [rbp-498h]
  unsigned int v404; // [rsp+CCh] [rbp-494h]
  int v405; // [rsp+D0h] [rbp-490h]
  char v406[64]; // [rsp+D8h] [rbp-488h] BYREF
  unsigned __int64 v407; // [rsp+118h] [rbp-448h] BYREF
  unsigned __int64 v408; // [rsp+120h] [rbp-440h]
  unsigned __int64 v409; // [rsp+128h] [rbp-438h]
  _QWORD v410[5]; // [rsp+130h] [rbp-430h] BYREF
  char v411[64]; // [rsp+158h] [rbp-408h] BYREF
  unsigned __int64 v412; // [rsp+198h] [rbp-3C8h]
  unsigned __int64 v413; // [rsp+1A0h] [rbp-3C0h]
  unsigned __int64 v414; // [rsp+1A8h] [rbp-3B8h]
  __int64 v415; // [rsp+1B0h] [rbp-3B0h] BYREF
  _BYTE *v416; // [rsp+1B8h] [rbp-3A8h]
  _BYTE *v417; // [rsp+1C0h] [rbp-3A0h]
  __int64 v418; // [rsp+1C8h] [rbp-398h]
  int v419; // [rsp+1D0h] [rbp-390h]
  _BYTE v420[64]; // [rsp+1D8h] [rbp-388h] BYREF
  const __m128i *v421; // [rsp+218h] [rbp-348h] BYREF
  __m128i *v422; // [rsp+220h] [rbp-340h]
  const __m128i *v423; // [rsp+228h] [rbp-338h]
  __int64 v424[16]; // [rsp+230h] [rbp-330h] BYREF
  __int64 v425[5]; // [rsp+2B0h] [rbp-2B0h] BYREF
  char v426[64]; // [rsp+2D8h] [rbp-288h] BYREF
  __m128 *v427; // [rsp+318h] [rbp-248h]
  __m128 *v428; // [rsp+320h] [rbp-240h]
  char *v429; // [rsp+328h] [rbp-238h]
  __int64 *v430[16]; // [rsp+330h] [rbp-230h] BYREF
  __m128i v431; // [rsp+3B0h] [rbp-1B0h] BYREF
  __int64 *v432; // [rsp+3C0h] [rbp-1A0h]
  __int64 v433; // [rsp+3C8h] [rbp-198h]
  __int64 *v434; // [rsp+3D0h] [rbp-190h]
  _BYTE v435[64]; // [rsp+3D8h] [rbp-188h] BYREF
  __int64 *v436; // [rsp+418h] [rbp-148h] BYREF
  __int64 *v437; // [rsp+420h] [rbp-140h]
  __int64 *v438; // [rsp+428h] [rbp-138h]
  __m128i v439; // [rsp+430h] [rbp-130h] BYREF
  __int64 *v440; // [rsp+440h] [rbp-120h]
  __int64 k; // [rsp+448h] [rbp-118h]
  __int64 *v442; // [rsp+450h] [rbp-110h]
  _QWORD v443[8]; // [rsp+458h] [rbp-108h] BYREF
  const __m128i *v444; // [rsp+498h] [rbp-C8h]
  __int64 **v445; // [rsp+4A0h] [rbp-C0h]
  char *v446; // [rsp+4A8h] [rbp-B8h]
  _QWORD v447[13]; // [rsp+4B0h] [rbp-B0h] BYREF
  __int64 **v448; // [rsp+518h] [rbp-48h]
  __int64 **v449; // [rsp+520h] [rbp-40h]

  v9 = a1;
  v439.m128i_i32[0] = dword_4FB3CA8;
  v10 = sub_16BAF20();
  if ( *((_BYTE *)sub_16BB850(v10, v439.m128i_i32) + 32) )
  {
    v431.m128i_i32[0] = dword_4FB3CA8;
    v175 = sub_16BAF20();
    if ( (unsigned __int8)sub_19E57A0(v175, v431.m128i_i32, &v439) )
      v176 = *(_QWORD *)(v439.m128i_i64[0] + 8);
    else
      v176 = *(_QWORD *)(*(_QWORD *)(v175 + 8) + 72LL * *(unsigned int *)(v175 + 24) + 8);
    *(_QWORD *)(a1 + 2800) = v176;
  }
  v11 = *(__int64 **)(a1 + 32);
  *(_DWORD *)(v9 + 1392) = *(_QWORD *)(*(_QWORD *)v9 + 96LL);
  *(_QWORD *)(v9 + 40) = sub_1423BA0(v11);
  v367 = (__int64 *)(v9 + 64);
  v12 = (_QWORD *)sub_145CDC0(0x18u, (__int64 *)(v9 + 64));
  if ( v12 )
  {
    v12[2] = 0;
    v12[1] = 0xFFFFFFFD00000003LL;
    *v12 = &unk_49F4C90;
  }
  *(_QWORD *)(v9 + 2088) = v12;
  v370 = v9 + 2424;
  v15 = *(_DWORD *)(v9 + 2432);
  if ( v15 >= *(_DWORD *)(v9 + 2436) )
  {
    sub_16CD150(v370, (const void *)(v9 + 2440), 0, 8, v13, v14);
    v15 = *(_DWORD *)(v9 + 2432);
  }
  v16 = (_QWORD *)(*(_QWORD *)(v9 + 2424) + 8LL * v15);
  if ( v16 )
  {
    *v16 = 0;
    v15 = *(_DWORD *)(v9 + 2432);
  }
  v397 = 0;
  *(_DWORD *)(v9 + 2432) = v15 + 1;
  v17 = *(_QWORD *)v9;
  v398 = 0;
  v18 = *(_QWORD *)(v17 + 80);
  v399 = 0;
  v415 = 0;
  v418 = 8;
  if ( v18 )
    v18 -= 24;
  memset(v424, 0, sizeof(v424));
  LODWORD(v424[3]) = 8;
  v424[1] = (__int64)&v424[5];
  v424[2] = (__int64)&v424[5];
  v416 = v420;
  v417 = v420;
  v419 = 0;
  v421 = 0;
  v422 = 0;
  v423 = 0;
  v19.m128i_i64[0] = (__int64)sub_1412190((__int64)&v415, v18);
  if ( v417 == v416 )
    v19.m128i_i64[1] = (__int64)&v417[8 * HIDWORD(v418)];
  else
    v19.m128i_i64[1] = (__int64)&v417[8 * (unsigned int)v418];
  v439 = v19;
  sub_19E4730((__int64)&v439);
  v20 = sub_157EBA0(v18);
  v439.m128i_i64[0] = v18;
  v21 = v422;
  v439.m128i_i64[1] = v20;
  LODWORD(v440) = 0;
  if ( v422 == v423 )
  {
    sub_13FDF40(&v421, v422, &v439);
  }
  else
  {
    if ( v422 )
    {
      a4 = (__m128)_mm_loadu_si128(&v439);
      *v422 = (__m128i)a4;
      v21[1].m128i_i64[0] = (__int64)v440;
      v21 = v422;
    }
    v422 = (__m128i *)((char *)v21 + 24);
  }
  sub_13FE0F0((__int64)&v415);
  v22 = &v430[5];
  sub_16CCCB0(v430, (__int64)&v430[5], (__int64)v424);
  v24 = v424[14];
  v25 = (const __m128i *)v424[13];
  memset(&v430[13], 0, 24);
  v26 = v424[14] - v424[13];
  if ( v424[14] == v424[13] )
  {
    v26 = 0;
    v28 = 0;
  }
  else
  {
    if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_530;
    v27 = sub_22077B0(v424[14] - v424[13]);
    v24 = v424[14];
    v25 = (const __m128i *)v424[13];
    v28 = (__int64 *)v27;
  }
  v430[13] = v28;
  v430[14] = v28;
  v430[15] = (__int64 *)((char *)v28 + v26);
  if ( (const __m128i *)v24 != v25 )
  {
    v29 = (__m128 *)v28;
    v30 = v25;
    do
    {
      if ( v29 )
      {
        a2 = (__m128)_mm_loadu_si128(v30);
        *v29 = a2;
        v29[1].m128_u64[0] = v30[1].m128i_u64[0];
      }
      v30 = (const __m128i *)((char *)v30 + 24);
      v29 = (__m128 *)((char *)v29 + 24);
    }
    while ( v30 != (const __m128i *)v24 );
    v28 += ((unsigned __int64)((char *)&v30[-2].m128i_u64[1] - (char *)v25) >> 3) + 3;
  }
  v430[14] = v28;
  sub_16CCEE0(&v431, (__int64)v435, 8, (__int64)v430);
  v31 = v430[13];
  memset(&v430[13], 0, 24);
  v436 = v31;
  v437 = v430[14];
  v438 = v430[15];
  sub_16CCCB0(v425, (__int64)v426, (__int64)&v415);
  v22 = (__int64 **)v422;
  v25 = v421;
  v427 = 0;
  v428 = 0;
  v429 = 0;
  v32 = (char *)v422 - (char *)v421;
  if ( v422 == v421 )
  {
    v34 = 0;
  }
  else
  {
    if ( v32 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_530;
    v33 = sub_22077B0((char *)v422 - (char *)v421);
    v22 = (__int64 **)v422;
    v25 = v421;
    v34 = (__m128 *)v33;
  }
  v427 = v34;
  v428 = v34;
  v429 = (char *)v34 + v32;
  if ( v22 != (__int64 **)v25 )
  {
    v35 = v34;
    v36 = v25;
    do
    {
      if ( v35 )
      {
        a3 = _mm_loadu_si128(v36);
        *v35 = (__m128)a3;
        v35[1].m128_u64[0] = v36[1].m128i_u64[0];
      }
      v36 = (const __m128i *)((char *)v36 + 24);
      v35 = (__m128 *)((char *)v35 + 24);
    }
    while ( v36 != (const __m128i *)v22 );
    v34 = (__m128 *)((char *)v34 + 8 * ((unsigned __int64)((char *)&v36[-2].m128i_u64[1] - (char *)v25) >> 3) + 24);
  }
  v428 = v34;
  sub_16CCEE0(&v439, (__int64)v443, 8, (__int64)v425);
  v37 = (const __m128i *)v427;
  v427 = 0;
  v444 = v37;
  v38 = (__int64 **)v428;
  v428 = 0;
  v445 = v38;
  v39 = v429;
  v429 = 0;
  v446 = v39;
  sub_19380F0((__int64)&v439, (__int64)&v431, (__int64)&v397);
  sub_19E7650(&v439);
  sub_19E7650(v425);
  sub_19E7650(&v431);
  sub_19E7650(v430);
  sub_19E7650(&v415);
  sub_19E7650(v424);
  v40 = v398;
  v41 = v397;
  if ( v398 == v397 )
  {
    v57 = *(_QWORD *)(v9 + 8);
    goto LABEL_96;
  }
  v42 = 0;
  v379 = v9 + 1400;
  do
  {
    while ( 1 )
    {
      v47 = *(_QWORD *)(v9 + 8);
      v48 = 0;
      v49 = *(unsigned int *)(v47 + 48);
      if ( !(_DWORD)v49 )
        goto LABEL_42;
      v50 = *(_QWORD *)(v40 - 8);
      v51 = *(_QWORD *)(v47 + 32);
      v52 = (v49 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v53 = (__int64 *)(v51 + 16LL * v52);
      v54 = *v53;
      if ( v50 == *v53 )
      {
LABEL_40:
        if ( v53 != (__int64 *)(v51 + 16 * v49) )
        {
          v48 = v53[1];
          goto LABEL_42;
        }
      }
      else
      {
        v170 = 1;
        while ( v54 != -8 )
        {
          v348 = v170 + 1;
          v52 = (v49 - 1) & (v170 + v52);
          v53 = (__int64 *)(v51 + 16LL * v52);
          v54 = *v53;
          if ( v50 == *v53 )
            goto LABEL_40;
          v170 = v348;
        }
      }
      v48 = 0;
LABEL_42:
      v55 = *(_DWORD *)(v9 + 1424);
      v431.m128i_i64[0] = v48;
      ++v42;
      if ( !v55 )
      {
        ++*(_QWORD *)(v9 + 1400);
LABEL_44:
        v55 *= 2;
LABEL_45:
        sub_19F5120(v9 + 1400, v55);
        sub_19E6B80(v9 + 1400, v431.m128i_i64, &v439);
        v45 = v439.m128i_i64[0];
        v48 = v431.m128i_i64[0];
        v56 = *(_DWORD *)(v9 + 1416) + 1;
        goto LABEL_46;
      }
      v43 = *(_QWORD *)(v9 + 1408);
      v44 = (v55 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v45 = v43 + 16LL * v44;
      v46 = *(_QWORD *)v45;
      if ( v48 != *(_QWORD *)v45 )
        break;
LABEL_37:
      v40 -= 8;
      *(_DWORD *)(v45 + 8) = v42;
      if ( v41 == v40 )
        goto LABEL_49;
    }
    v167 = 1;
    v168 = 0;
    while ( v46 != -8 )
    {
      if ( !v168 && v46 == -16 )
        v168 = v45;
      v44 = (v55 - 1) & (v167 + v44);
      v45 = v43 + 16LL * v44;
      v46 = *(_QWORD *)v45;
      if ( v48 == *(_QWORD *)v45 )
        goto LABEL_37;
      ++v167;
    }
    v169 = *(_DWORD *)(v9 + 1416);
    if ( v168 )
      v45 = v168;
    ++*(_QWORD *)(v9 + 1400);
    v56 = v169 + 1;
    if ( 4 * (v169 + 1) >= 3 * v55 )
      goto LABEL_44;
    if ( v55 - *(_DWORD *)(v9 + 1420) - v56 <= v55 >> 3 )
      goto LABEL_45;
LABEL_46:
    *(_DWORD *)(v9 + 1416) = v56;
    if ( *(_QWORD *)v45 != -8 )
      --*(_DWORD *)(v9 + 1420);
    v40 -= 8;
    *(_DWORD *)(v45 + 8) = 0;
    *(_QWORD *)v45 = v48;
    *(_DWORD *)(v45 + 8) = v42;
  }
  while ( v41 != v40 );
LABEL_49:
  v57 = *(_QWORD *)(v9 + 8);
  v374 = v397;
  if ( v397 != v398 )
  {
    v391 = v398;
    v58 = v9;
    do
    {
      v59 = *(unsigned int *)(v57 + 48);
      if ( !(_DWORD)v59 )
        goto LABEL_555;
      v60 = *(_QWORD *)(v57 + 32);
      v61 = *(_QWORD *)(v391 - 8);
      v62 = (v59 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v63 = (__int64 *)(v60 + 16LL * v62);
      v64 = *v63;
      if ( v61 != *v63 )
      {
        v165 = 1;
        while ( v64 != -8 )
        {
          v166 = v165 + 1;
          v62 = (v59 - 1) & (v165 + v62);
          v63 = (__int64 *)(v60 + 16LL * v62);
          v64 = *v63;
          if ( v61 == *v63 )
            goto LABEL_54;
          v165 = v166;
        }
LABEL_555:
        BUG();
      }
LABEL_54:
      if ( v63 == (__int64 *)(v60 + 16 * v59) )
        goto LABEL_555;
      v65 = v63[1];
      v66 = *(__int64 **)(v65 + 24);
      v382 = *(char **)(v65 + 32);
      v67 = v382 - (char *)v66;
      if ( (unsigned __int64)(v382 - (char *)v66) > 8 && v66 != (__int64 *)v382 )
      {
        _BitScanReverse64(&v68, v67 >> 3);
        sub_19F5A00(v66, v382, 2LL * (int)(63 - (v68 ^ 0x3F)), v58);
        if ( v67 > 128 )
        {
          sub_19F6B20(v66, v66 + 16, v58);
          v386 = (__int64 **)(v66 + 16);
          if ( v382 == (char *)(v66 + 16) )
            goto LABEL_79;
          while ( 1 )
          {
            v69 = v386;
            v70 = *v386;
            v71 = ((unsigned int)*v386 >> 9) ^ ((unsigned int)*v386 >> 4);
            while ( 1 )
            {
              v81 = (__int64)*(v69 - 1);
              v82 = *(_DWORD *)(v58 + 1424);
              v83 = v69;
              v430[0] = v70;
              v431.m128i_i64[0] = v81;
              if ( !v82 )
              {
                ++*(_QWORD *)(v58 + 1400);
LABEL_67:
                v82 *= 2;
                goto LABEL_68;
              }
              v72 = v82 - 1;
              v73 = *(_QWORD *)(v58 + 1408);
              v74 = (v82 - 1) & v71;
              v75 = v73 + 16LL * v74;
              v76 = *(__int64 **)v75;
              if ( v70 == *(__int64 **)v75 )
              {
LABEL_61:
                v77 = *(_DWORD *)(v75 + 8);
                goto LABEL_62;
              }
              v90 = 1;
              v84 = 0;
              while ( v76 != (__int64 *)-8LL )
              {
                if ( v84 || v76 != (__int64 *)-16LL )
                  v75 = v84;
                v74 = v72 & (v90 + v74);
                v76 = *(__int64 **)(v73 + 16LL * v74);
                if ( v70 == v76 )
                {
                  v75 = v73 + 16LL * v74;
                  goto LABEL_61;
                }
                ++v90;
                v84 = v75;
                v75 = v73 + 16LL * v74;
              }
              if ( !v84 )
                v84 = v75;
              v91 = *(_DWORD *)(v58 + 1416);
              ++*(_QWORD *)(v58 + 1400);
              v86 = v91 + 1;
              if ( 4 * (v91 + 1) >= 3 * v82 )
                goto LABEL_67;
              v85 = v70;
              if ( v82 - *(_DWORD *)(v58 + 1420) - v86 > v82 >> 3 )
                goto LABEL_69;
LABEL_68:
              sub_19F5120(v379, v82);
              sub_19E6B80(v379, (__int64 *)v430, &v439);
              v84 = v439.m128i_i64[0];
              v85 = v430[0];
              v86 = *(_DWORD *)(v58 + 1416) + 1;
LABEL_69:
              *(_DWORD *)(v58 + 1416) = v86;
              if ( *(_QWORD *)v84 != -8 )
                --*(_DWORD *)(v58 + 1420);
              *(_QWORD *)v84 = v85;
              *(_DWORD *)(v84 + 8) = 0;
              v82 = *(_DWORD *)(v58 + 1424);
              if ( !v82 )
              {
                ++*(_QWORD *)(v58 + 1400);
LABEL_73:
                v82 *= 2;
                goto LABEL_74;
              }
              v73 = *(_QWORD *)(v58 + 1408);
              v81 = v431.m128i_i64[0];
              v72 = v82 - 1;
              v77 = 0;
LABEL_62:
              v78 = v72 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
              v79 = v73 + 16LL * v78;
              v80 = *(_QWORD *)v79;
              if ( v81 != *(_QWORD *)v79 )
                break;
LABEL_63:
              --v69;
              if ( *(_DWORD *)(v79 + 8) <= v77 )
                goto LABEL_78;
              v69[1] = *v69;
            }
            v372 = 1;
            v87 = 0;
            while ( v80 != -8 )
            {
              if ( v80 != -16 || v87 )
                v79 = v87;
              v78 = v72 & (v372 + v78);
              v80 = *(_QWORD *)(v73 + 16LL * v78);
              if ( v80 == v81 )
              {
                v79 = v73 + 16LL * v78;
                goto LABEL_63;
              }
              ++v372;
              v87 = v79;
              v79 = v73 + 16LL * v78;
            }
            if ( !v87 )
              v87 = v79;
            v89 = *(_DWORD *)(v58 + 1416);
            ++*(_QWORD *)(v58 + 1400);
            v88 = v89 + 1;
            if ( 4 * v88 >= 3 * v82 )
              goto LABEL_73;
            if ( v82 - (v88 + *(_DWORD *)(v58 + 1420)) > v82 >> 3 )
              goto LABEL_75;
LABEL_74:
            sub_19F5120(v379, v82);
            sub_19E6B80(v379, v431.m128i_i64, &v439);
            v87 = v439.m128i_i64[0];
            v81 = v431.m128i_i64[0];
            v88 = *(_DWORD *)(v58 + 1416) + 1;
LABEL_75:
            *(_DWORD *)(v58 + 1416) = v88;
            if ( *(_QWORD *)v87 != -8 )
              --*(_DWORD *)(v58 + 1420);
            *(_QWORD *)v87 = v81;
            *(_DWORD *)(v87 + 8) = 0;
LABEL_78:
            ++v386;
            *v83 = v70;
            if ( v382 == (char *)v386 )
            {
LABEL_79:
              v57 = *(_QWORD *)(v58 + 8);
              goto LABEL_51;
            }
          }
        }
        sub_19F6B20(v66, (__int64 *)v382, v58);
        v57 = *(_QWORD *)(v58 + 8);
      }
LABEL_51:
      v391 -= 8;
    }
    while ( v374 != v391 );
    v9 = v58;
  }
LABEL_96:
  v92 = *(_QWORD *)(v57 + 56);
  v431.m128i_i64[0] = 0;
  memset(v430, 0, sizeof(v430));
  LODWORD(v430[3]) = 8;
  v430[1] = (__int64 *)&v430[5];
  v430[2] = (__int64 *)&v430[5];
  v433 = 8;
  v431.m128i_i64[1] = (__int64)v435;
  v432 = (__int64 *)v435;
  LODWORD(v434) = 0;
  v436 = 0;
  v437 = 0;
  v438 = 0;
  v93.m128i_i64[0] = (__int64)sub_1412190((__int64)&v431, v92);
  if ( v432 == (__int64 *)v431.m128i_i64[1] )
    v93.m128i_i64[1] = (__int64)&v432[HIDWORD(v433)];
  else
    v93.m128i_i64[1] = (__int64)&v432[(unsigned int)v433];
  v439 = v93;
  sub_19E4730((__int64)&v439);
  v439.m128i_i64[0] = v92;
  LOBYTE(v440) = 0;
  sub_13B8390((unsigned __int64 *)&v436, (__int64)&v439);
  sub_13BA6D0(&v439, &v431, v430);
  sub_19E4F00(&v431);
  sub_19E4F00(v430);
  sub_16CCCB0(&v400, (__int64)v406, (__int64)&v439);
  v22 = v445;
  v25 = v444;
  v407 = 0;
  v408 = 0;
  v409 = 0;
  v94 = (char *)v445 - (char *)v444;
  if ( v445 != (__int64 **)v444 )
  {
    if ( v94 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v95 = sub_22077B0((char *)v445 - (char *)v444);
      v22 = v445;
      v25 = v444;
      v96 = v95;
      goto LABEL_101;
    }
LABEL_530:
    sub_4261EA(v25, v22, v23);
  }
  v94 = 0;
  v96 = 0;
LABEL_101:
  v407 = v96;
  v408 = v96;
  v409 = v96 + v94;
  if ( v22 != (__int64 **)v25 )
  {
    v97 = v96;
    v98 = v25;
    do
    {
      if ( v97 )
      {
        *(_QWORD *)v97 = v98->m128i_i64[0];
        v99 = v98[1].m128i_i8[0];
        *(_BYTE *)(v97 + 16) = v99;
        if ( v99 )
          *(_QWORD *)(v97 + 8) = v98->m128i_i64[1];
      }
      v98 = (const __m128i *)((char *)v98 + 24);
      v97 += 24LL;
    }
    while ( v98 != (const __m128i *)v22 );
    v96 += 8 * ((unsigned __int64)((char *)&v98[-2].m128i_u64[1] - (char *)v25) >> 3) + 24;
  }
  v408 = v96;
  v25 = (const __m128i *)v410;
  sub_16CCCB0(v410, (__int64)v411, (__int64)v447);
  v22 = v449;
  v100 = v448;
  v412 = 0;
  v413 = 0;
  v414 = 0;
  v101 = (char *)v449 - (char *)v448;
  if ( v449 == v448 )
  {
    v103 = 0;
  }
  else
  {
    if ( v101 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_530;
    v102 = sub_22077B0((char *)v449 - (char *)v448);
    v22 = v449;
    v100 = v448;
    v103 = v102;
  }
  v412 = v103;
  v413 = v103;
  v414 = v103 + v101;
  if ( v22 == v100 )
  {
    v107 = v103;
  }
  else
  {
    v104 = v103;
    v105 = v100;
    do
    {
      if ( v104 )
      {
        *(_QWORD *)v104 = *v105;
        v106 = *((_BYTE *)v105 + 16);
        *(_BYTE *)(v104 + 16) = v106;
        if ( v106 )
          *(_QWORD *)(v104 + 8) = v105[1];
      }
      v105 += 3;
      v104 += 24LL;
    }
    while ( v105 != v22 );
    v107 = v103 + 8 * ((unsigned __int64)((char *)(v105 - 3) - (char *)v100) >> 3) + 24;
  }
  v413 = v107;
  v108 = v407;
  v368 = v9 + 2360;
  v109 = v408;
  v387 = 1;
  v383 = v9 + 2392;
  while ( 2 )
  {
    if ( v109 - v108 != v107 - v103 )
    {
LABEL_120:
      v110 = *(_QWORD *)(v9 + 32);
      v380 = **(_QWORD **)(v109 - 24);
      v425[0] = v380;
      v111 = sub_14228C0(v110, v380);
      v112 = v111;
      v113 = v387;
      if ( v111 )
      {
        v431.m128i_i64[0] = v111;
        v113 = v387 + 1;
        *(_DWORD *)(sub_19EC730(v383, v431.m128i_i64) + 8) = v387;
        v116 = *(_DWORD *)(v9 + 2432);
        if ( v116 >= *(_DWORD *)(v9 + 2436) )
        {
          sub_16CD150(v370, (const void *)(v9 + 2440), 0, 8, v114, v115);
          v116 = *(_DWORD *)(v9 + 2432);
        }
        v117 = (__int64 *)(*(_QWORD *)(v9 + 2424) + 8LL * v116);
        if ( v117 )
        {
          *v117 = v112;
          v116 = *(_DWORD *)(v9 + 2432);
        }
        *(_DWORD *)(v9 + 2432) = v116 + 1;
      }
      v118 = *(_QWORD *)(v425[0] + 48);
      v392 = v425[0] + 40;
      if ( v118 == v425[0] + 40 )
      {
LABEL_160:
        v140 = *(_DWORD *)(v9 + 2384);
        v431.m128i_i64[0] = v380;
        v431.m128i_i64[1] = __PAIR64__(v113, v387);
        if ( v140 )
        {
          v141 = *(_QWORD *)(v9 + 2368);
          v142 = (v140 - 1) & (((unsigned int)v380 >> 9) ^ ((unsigned int)v380 >> 4));
          v143 = (__int64 *)(v141 + 16LL * v142);
          v144 = *v143;
          if ( v380 == *v143 )
            goto LABEL_162;
          v171 = 1;
          v172 = 0;
          while ( v144 != -8 )
          {
            if ( !v172 && v144 == -16 )
              v172 = v143;
            v142 = (v140 - 1) & (v171 + v142);
            v143 = (__int64 *)(v141 + 16LL * v142);
            v144 = *v143;
            if ( v380 == *v143 )
              goto LABEL_162;
            ++v171;
          }
          v173 = *(_DWORD *)(v9 + 2376);
          if ( v172 )
            v143 = v172;
          ++*(_QWORD *)(v9 + 2360);
          v174 = v173 + 1;
          if ( 4 * (v173 + 1) < 3 * v140 )
          {
            if ( v140 - *(_DWORD *)(v9 + 2380) - v174 > v140 >> 3 )
            {
LABEL_228:
              *(_DWORD *)(v9 + 2376) = v174;
              if ( *v143 != -8 )
                --*(_DWORD *)(v9 + 2380);
              *v143 = v380;
              v143[1] = v431.m128i_i64[1];
LABEL_162:
              v393 = v9;
              v145 = v408;
              do
              {
                v146 = *(_QWORD *)(v145 - 24);
                if ( !*(_BYTE *)(v145 - 8) )
                {
                  v147 = *(__int64 **)(v146 + 24);
                  *(_BYTE *)(v145 - 8) = 1;
                  *(_QWORD *)(v145 - 16) = v147;
                  goto LABEL_167;
                }
LABEL_166:
                while ( 1 )
                {
                  v147 = *(__int64 **)(v145 - 16);
LABEL_167:
                  if ( v147 == *(__int64 **)(v146 + 32) )
                    break;
                  *(_QWORD *)(v145 - 16) = v147 + 1;
                  v149 = *v147;
                  v150 = v401;
                  if ( v402 == v401 )
                  {
                    v151 = &v401[v404];
                    if ( v401 != v151 )
                    {
                      for ( i = 0; ; i = v150++ )
                      {
                        while ( 1 )
                        {
                          if ( v149 == *v150 )
                            goto LABEL_166;
                          if ( *v150 == -2 )
                            break;
                          if ( v151 == ++v150 )
                          {
                            if ( !i )
                              goto LABEL_184;
                            v9 = v393;
                            goto LABEL_176;
                          }
                        }
                        if ( v150 + 1 == v151 )
                        {
                          v9 = v393;
                          i = v150;
LABEL_176:
                          *i = v149;
                          --v405;
                          ++v400;
                          goto LABEL_177;
                        }
                      }
                    }
LABEL_184:
                    if ( v404 < v403 )
                    {
                      v9 = v393;
                      ++v404;
                      *v151 = v149;
                      ++v400;
LABEL_177:
                      v431.m128i_i64[0] = v149;
                      LOBYTE(v432) = 0;
                      sub_13B8390(&v407, (__int64)&v431);
                      v108 = v407;
                      v109 = v408;
                      goto LABEL_178;
                    }
                  }
                  sub_16CCBA0((__int64)&v400, v149);
                  if ( v148 )
                  {
                    v9 = v393;
                    goto LABEL_177;
                  }
                }
                v408 -= 24LL;
                v108 = v407;
                v145 = v408;
              }
              while ( v408 != v407 );
              v9 = v393;
              v109 = v407;
LABEL_178:
              v387 = v113;
              v103 = v412;
              v107 = v413;
              continue;
            }
LABEL_238:
            sub_19F7080(v368, v140);
            sub_19E7410(v368, v431.m128i_i64, v430);
            v143 = v430[0];
            v380 = v431.m128i_i64[0];
            v174 = *(_DWORD *)(v9 + 2376) + 1;
            goto LABEL_228;
          }
        }
        else
        {
          ++*(_QWORD *)(v9 + 2360);
        }
        v140 *= 2;
        goto LABEL_238;
      }
      while ( 2 )
      {
        while ( 2 )
        {
          v126 = v118 - 24;
          if ( !v118 )
            v126 = 0;
          v127 = sub_1AE9990(v126, *(_QWORD *)(v9 + 16));
          v128 = (__int64 *)v126;
          if ( v127 )
          {
            v430[0] = (__int64 *)v126;
            v134 = sub_154CC80(v383, (__int64 *)v430, &v431);
            v135 = v431.m128i_i64[0];
            if ( !v134 )
            {
              v136 = *(_DWORD *)(v9 + 2408);
              v137 = *(_DWORD *)(v9 + 2416);
              ++*(_QWORD *)(v9 + 2392);
              v138 = v136 + 1;
              if ( 4 * (v136 + 1) >= 3 * v137 )
              {
                v137 *= 2;
              }
              else if ( v137 - *(_DWORD *)(v9 + 2412) - v138 > v137 >> 3 )
              {
                goto LABEL_156;
              }
              sub_1542080(v383, v137);
              sub_154CC80(v383, (__int64 *)v430, &v431);
              v135 = v431.m128i_i64[0];
              v138 = *(_DWORD *)(v9 + 2408) + 1;
LABEL_156:
              *(_DWORD *)(v9 + 2408) = v138;
              if ( *(_QWORD *)v135 != -8 )
                --*(_DWORD *)(v9 + 2412);
              v139 = v430[0];
              *(_DWORD *)(v135 + 8) = 0;
              *(_QWORD *)v135 = v139;
            }
            *(_DWORD *)(v135 + 8) = 0;
            sub_165A590((__int64)&v431, v9 + 2696, v126);
            v118 = *(_QWORD *)(v118 + 8);
            if ( v392 == v118 )
              goto LABEL_160;
            continue;
          }
          break;
        }
        if ( *(_BYTE *)(v126 + 16) != 77 )
        {
LABEL_138:
          v129 = *(_DWORD *)(v9 + 2416);
          v430[0] = (__int64 *)v126;
          if ( !v129 )
          {
            ++*(_QWORD *)(v9 + 2392);
            goto LABEL_140;
          }
          v119 = v129 - 1;
          v120 = *(_QWORD *)(v9 + 2400);
          v121 = (v129 - 1) & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
          v122 = v120 + 16LL * v121;
          v123 = *(_QWORD *)v122;
          if ( *(_QWORD *)v122 != v126 )
          {
            v159 = 1;
            v160 = 0;
            while ( v123 != -8 )
            {
              if ( v160 || v123 != -16 )
                v122 = v160;
              v121 = v119 & (v159 + v121);
              v123 = *(_QWORD *)(v120 + 16LL * v121);
              if ( v123 == v126 )
              {
                v122 = v120 + 16LL * v121;
                goto LABEL_129;
              }
              ++v159;
              v160 = v122;
              v122 = v120 + 16LL * v121;
            }
            v161 = *(_DWORD *)(v9 + 2408);
            if ( v160 )
              v122 = v160;
            ++*(_QWORD *)(v9 + 2392);
            v130 = v161 + 1;
            if ( 4 * v130 >= 3 * v129 )
            {
LABEL_140:
              v129 *= 2;
            }
            else
            {
              LODWORD(v120) = v129 >> 3;
              if ( v129 - *(_DWORD *)(v9 + 2412) - v130 > v129 >> 3 )
                goto LABEL_142;
            }
            sub_1542080(v383, v129);
            sub_154CC80(v383, (__int64 *)v430, &v431);
            v122 = v431.m128i_i64[0];
            v128 = v430[0];
            v130 = *(_DWORD *)(v9 + 2408) + 1;
LABEL_142:
            *(_DWORD *)(v9 + 2408) = v130;
            if ( *(_QWORD *)v122 != -8 )
              --*(_DWORD *)(v9 + 2412);
            *(_QWORD *)v122 = v128;
            *(_DWORD *)(v122 + 8) = 0;
          }
LABEL_129:
          *(_DWORD *)(v122 + 8) = v113;
          v124 = *(_DWORD *)(v9 + 2432);
          if ( v124 >= *(_DWORD *)(v9 + 2436) )
          {
            sub_16CD150(v370, (const void *)(v9 + 2440), 0, 8, v120, v119);
            v124 = *(_DWORD *)(v9 + 2432);
          }
          v125 = (__int64 *)(*(_QWORD *)(v9 + 2424) + 8LL * v124);
          if ( v125 )
          {
            *v125 = v126;
            v124 = *(_DWORD *)(v9 + 2432);
          }
          ++v113;
          *(_DWORD *)(v9 + 2432) = v124 + 1;
          v118 = *(_QWORD *)(v118 + 8);
          if ( v392 == v118 )
            goto LABEL_160;
          continue;
        }
        break;
      }
      v153 = *(_DWORD *)(v9 + 1888);
      if ( v153 )
      {
        v154 = v425[0];
        v155 = *(_QWORD *)(v9 + 1872);
        v156 = (v153 - 1) & ((LODWORD(v425[0]) >> 9) ^ (LODWORD(v425[0]) >> 4));
        v157 = (__int64 *)(v155 + 40LL * v156);
        v158 = *v157;
        if ( v425[0] == *v157 )
        {
LABEL_181:
          v375 = v128;
          sub_1369D60(v157 + 1, v113);
          v128 = v375;
          goto LABEL_138;
        }
        v376 = 1;
        v162 = 0;
        while ( v158 != -8 )
        {
          if ( v158 == -16 && !v162 )
            v162 = v157;
          v156 = (v153 - 1) & (v376 + v156);
          v157 = (__int64 *)(v155 + 40LL * v156);
          v158 = *v157;
          if ( v425[0] == *v157 )
            goto LABEL_181;
          ++v376;
        }
        v163 = *(_DWORD *)(v9 + 1880);
        if ( v162 )
          v157 = v162;
        ++*(_QWORD *)(v9 + 1864);
        v164 = v163 + 1;
        if ( 4 * v164 < 3 * v153 )
        {
          if ( v153 - *(_DWORD *)(v9 + 1884) - v164 > v153 >> 3 )
          {
LABEL_204:
            *(_DWORD *)(v9 + 1880) = v164;
            if ( *v157 != -8 )
              --*(_DWORD *)(v9 + 1884);
            *v157 = v154;
            v157[3] = (__int64)(v157 + 2);
            v157[2] = (__int64)(v157 + 2);
            v157[4] = 0;
            v157[1] = (__int64)(v157 + 2);
            goto LABEL_181;
          }
          v377 = (__int64 *)v126;
LABEL_233:
          sub_19F17B0(v9 + 1864, v153);
          sub_19EB4D0(v9 + 1864, v425, &v431);
          v157 = (__int64 *)v431.m128i_i64[0];
          v154 = v425[0];
          v128 = v377;
          v164 = *(_DWORD *)(v9 + 1880) + 1;
          goto LABEL_204;
        }
      }
      else
      {
        ++*(_QWORD *)(v9 + 1864);
      }
      v377 = (__int64 *)v126;
      v153 *= 2;
      goto LABEL_233;
    }
    break;
  }
  while ( v108 != v109 )
  {
    if ( *(_QWORD *)v108 != *(_QWORD *)v103 )
      goto LABEL_120;
    v131 = *(_BYTE *)(v108 + 16);
    v132 = *(_BYTE *)(v103 + 16);
    if ( v131 && v132 )
      v133 = *(_QWORD *)(v108 + 8) == *(_QWORD *)(v103 + 8);
    else
      v133 = v131 == v132;
    if ( !v133 )
      goto LABEL_120;
    v108 += 24LL;
    v103 += 24LL;
  }
  sub_19E4F00(v410);
  sub_19E4F00(&v400);
  sub_19E4F00(v447);
  sub_19E4F00(&v439);
  sub_19EB580(v9, *(_QWORD *)v9);
  v180 = *(_QWORD *)(v9 + 2344);
  if ( v387 > (unsigned __int64)(v180 << 6) )
  {
    v343 = (v387 + 63) >> 6;
    v344 = 2 * v180;
    if ( v343 >= 2 * v180 )
      v344 = (v387 + 63) >> 6;
    v345 = 8 * v344;
    v396 = v344;
    v346 = (__int64)realloc(*(_QWORD *)(v9 + 2336), 8 * v344, v343, v177, v178, v179);
    if ( !v346 && (v345 || (v346 = malloc(1u)) == 0) )
      sub_16BD1C0("Allocation failed", 1u);
    *(_QWORD *)(v9 + 2336) = v346;
    *(_QWORD *)(v9 + 2344) = v396;
    sub_13A4C60(v9 + 2336, 0);
    v347 = *(_QWORD *)(v9 + 2344) - (unsigned int)v180;
    if ( v347 )
      memset((void *)(*(_QWORD *)(v9 + 2336) + 8LL * (unsigned int)v180), 0, 8 * v347);
  }
  v181 = *(_DWORD *)(v9 + 2352);
  if ( v387 > v181 )
  {
    sub_13A4C60(v9 + 2336, 0);
    v181 = *(_DWORD *)(v9 + 2352);
  }
  *(_DWORD *)(v9 + 2352) = v387;
  if ( v387 < v181 )
    sub_13A4C60(v9 + 2336, 0);
  if ( v387 )
  {
    v182 = sub_1454B60(4 * v387 / 3 + 1);
    ++*(_QWORD *)(v9 + 2056);
    if ( *(_DWORD *)(v9 + 2080) < v182 )
      sub_19E3E70(v9 + 2056, v182);
  }
  else
  {
    ++*(_QWORD *)(v9 + 2056);
  }
  v183 = *(_QWORD *)(*(_QWORD *)v9 + 80LL);
  if ( v183 )
    v183 -= 24;
  v184 = sub_19E56C0(v368, v183);
  sub_19E1950((_QWORD *)(v9 + 2336), v184, HIDWORD(v184));
  v185 = *(_QWORD *)(*(_QWORD *)v9 + 80LL);
  if ( v185 )
    v185 -= 24;
  v369 = v9 + 2232;
  v186.m128i_i64[0] = (__int64)sub_1412190(v9 + 2232, v185);
  v186.m128i_i64[1] = *(_QWORD *)(v9 + 2248);
  if ( v186.m128i_i64[1] == *(_QWORD *)(v9 + 2240) )
    v187 = *(unsigned int *)(v9 + 2260);
  else
    v187 = *(unsigned int *)(v9 + 2256);
  v186.m128i_i64[1] += 8 * v187;
  v439 = v186;
  sub_19E4730((__int64)&v439);
  v190 = *(_DWORD *)(v9 + 2352);
  if ( !v190 )
    goto LABEL_344;
  v191 = (unsigned int)(v190 - 1) >> 6;
  v192 = 0;
  while ( 1 )
  {
    _RDX = *(_QWORD *)(*(_QWORD *)(v9 + 2336) + 8 * v192);
    v185 = (unsigned int)v192;
    if ( v192 == v191 )
      break;
    if ( _RDX )
      goto LABEL_282;
    if ( ++v192 == (_DWORD)v191 + 1 )
      goto LABEL_344;
  }
  _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v190;
  if ( !_RDX )
    goto LABEL_344;
LABEL_282:
  __asm { tzcnt   rdx, rdx }
  v185 = (unsigned int)((_DWORD)v192 << 6);
  v195 = (unsigned int)(v185 + _RDX);
  if ( (_DWORD)v195 == -1 )
    goto LABEL_344;
  v185 = *(_QWORD *)(*(_QWORD *)(v9 + 2424) + 8 * v195);
  v196 = sub_19E73A0(v9, v185);
  v197 = *(_DWORD *)(v9 + 2352);
  v198 = v196;
  v199 = v197 + 63;
  do
  {
    do
    {
LABEL_284:
      v200 = v199 >> 6;
      if ( !(v199 >> 6) )
        goto LABEL_344;
LABEL_285:
      v201 = *(_QWORD **)(v9 + 2336);
      v202 = v201;
      v203 = (__int64)&v201[v200];
      while ( !*v202 )
      {
        if ( ++v202 == (_QWORD *)v203 )
          goto LABEL_344;
      }
    }
    while ( !v197 );
    v204 = 0;
    v185 = (unsigned int)(v197 - 1) >> 6;
    while ( 1 )
    {
      _RDX = v201[v204];
      if ( (_DWORD)v185 == (_DWORD)v204 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v197) & v201[v204];
      if ( _RDX )
        break;
      if ( (_DWORD)v185 + 1 == ++v204 )
        goto LABEL_284;
    }
    __asm { tzcnt   rdx, rdx }
    v388 = ((_DWORD)v204 << 6) + _RDX;
  }
  while ( v388 == -1 );
  v394 = v9;
  while ( 2 )
  {
    v207 = v388;
    if ( !v388 )
    {
      *v201 &= ~1uLL;
      v197 = *(_DWORD *)(v394 + 2352);
      if ( v197 == 1 )
      {
        v9 = v394;
        v199 = 64;
        goto LABEL_284;
      }
      LOBYTE(v245) = 1;
      v185 = (unsigned int)(v197 - 1) >> 6;
      goto LABEL_333;
    }
    v185 = v394;
    v208 = *(_QWORD *)(*(_QWORD *)(v394 + 2424) + 8LL * v388);
    if ( *(_BYTE *)(v208 + 16) <= 0x17u )
    {
      if ( *(_BYTE *)(v208 + 16) != 23 )
        BUG();
      v381 = *(_QWORD *)(v208 + 64);
    }
    else
    {
      v381 = *(_QWORD *)(v208 + 40);
      if ( !v381 )
      {
        v309 = *(_DWORD *)(v394 + 1696);
        if ( v309 )
        {
          v310 = v309 - 1;
          v185 = *(_QWORD *)(v394 + 1680);
          v311 = v310 & (((unsigned int)v208 >> 9) ^ ((unsigned int)v208 >> 4));
          v312 = (__int64 *)(v185 + 16LL * v311);
          v313 = *v312;
          if ( v208 == *v312 )
          {
LABEL_438:
            v381 = v312[1];
          }
          else
          {
            v341 = 1;
            while ( v313 != -8 )
            {
              v342 = v341 + 1;
              v311 = v310 & (v341 + v311);
              v312 = (__int64 *)(v185 + 16LL * v311);
              v313 = *v312;
              if ( v208 == *v312 )
                goto LABEL_438;
              v341 = v342;
            }
          }
        }
      }
    }
    if ( v198 == v381 )
    {
LABEL_308:
      v201[v388 >> 6] &= ~(1LL << v388);
      v218 = *(unsigned __int8 *)(v208 + 16);
      if ( (_BYTE)v218 != 23 )
      {
        if ( (unsigned int)(v218 - 25) > 9 )
        {
          v439.m128i_i64[0] = 0;
          k = 2;
          v439.m128i_i64[1] = (__int64)v443;
          v440 = v443;
          LODWORD(v442) = 0;
          v307 = sub_19EEAC0(v394, v208);
          v308 = (__int64)v307;
          if ( v307 )
          {
            if ( (unsigned int)(*((_DWORD *)v307 + 2) - 1) > 1 )
            {
              v330 = *(_QWORD *)(v394 + 1552);
              v331 = v330 == *(_QWORD *)(v394 + 1544) ? *(unsigned int *)(v394 + 1564) : *(unsigned int *)(v394 + 1560);
              v431.m128i_i64[0] = v330 + 8 * v331;
              v431.m128i_i64[1] = v431.m128i_i64[0];
              sub_19E4730((__int64)&v431);
              v332 = *(_QWORD *)(v394 + 1536);
              v432 = (__int64 *)(v394 + 1536);
              v433 = v332;
              v333 = sub_15CC2D0(v394 + 1536, v208);
              v334 = *(_QWORD *)(v394 + 1552);
              v335 = v334 == *(_QWORD *)(v394 + 1544) ? *(unsigned int *)(v394 + 1564) : *(unsigned int *)(v394 + 1560);
              v430[0] = v333;
              v430[1] = (__int64 *)(v334 + 8 * v335);
              sub_19E4730((__int64)v430);
              if ( v430[0] != (__int64 *)v431.m128i_i64[0] )
              {
                v338 = sub_19F3410(
                         v394,
                         v208,
                         (__int64)&v439,
                         a2,
                         a3,
                         *(double *)a4.m128_u64,
                         *(double *)a5.m128i_i64,
                         v336,
                         v337,
                         a8,
                         a9);
                if ( v338 )
                {
                  v308 = v338;
                }
                else
                {
                  v349 = (__int64 *)sub_19E5730(v394 + 1704, v208);
                  v350 = v349;
                  if ( v349 )
                  {
                    v430[0] = v349;
                    if ( (unsigned __int8)sub_154CC80(v394 + 2392, (__int64 *)v430, &v431) )
                    {
                      *(_QWORD *)v431.m128i_i64[0] = -16;
                      --*(_DWORD *)(v394 + 2408);
                      ++*(_DWORD *)(v394 + 2412);
                    }
                    v351 = *(_DWORD *)(v394 + 1696);
                    if ( v351 )
                    {
                      v352 = *(_QWORD *)(v394 + 1680);
                      v353 = v351 - 1;
                      v354 = v353 & (((unsigned int)v350 >> 9) ^ ((unsigned int)v350 >> 4));
                      v355 = (__int64 **)(v352 + 16LL * v354);
                      v356 = *v355;
                      if ( v350 == *v355 )
                      {
LABEL_537:
                        *v355 = (__int64 *)-16LL;
                        --*(_DWORD *)(v394 + 1688);
                        ++*(_DWORD *)(v394 + 1692);
                      }
                      else
                      {
                        v363 = 1;
                        while ( v356 != (__int64 *)-8LL )
                        {
                          v364 = v363 + 1;
                          v354 = v353 & (v363 + v354);
                          v355 = (__int64 **)(v352 + 16LL * v354);
                          v356 = *v355;
                          if ( v350 == *v355 )
                            goto LABEL_537;
                          v363 = v364;
                        }
                      }
                    }
                    v357 = *(_DWORD *)(v394 + 1728);
                    if ( v357 )
                    {
                      v358 = *(_QWORD *)(v394 + 1712);
                      v359 = v357 - 1;
                      v360 = v359 & (((unsigned int)v208 >> 9) ^ ((unsigned int)v208 >> 4));
                      v361 = (__int64 *)(v358 + 16LL * v360);
                      v362 = *v361;
                      if ( *v361 == v208 )
                      {
LABEL_540:
                        *v361 = -16;
                        --*(_DWORD *)(v394 + 1720);
                        ++*(_DWORD *)(v394 + 1724);
                      }
                      else
                      {
                        v365 = 1;
                        while ( v362 != -8 )
                        {
                          v366 = v365 + 1;
                          v360 = v359 & (v365 + v360);
                          v361 = (__int64 *)(v358 + 16LL * v360);
                          v362 = *v361;
                          if ( v208 == *v361 )
                            goto LABEL_540;
                          v365 = v366;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          else
          {
            v339 = sub_145CDC0(0x20u, v367);
            v308 = v339;
            if ( v339 )
            {
              *(_DWORD *)(v339 + 8) = 4;
              *(_QWORD *)(v339 + 16) = 0;
              *(_QWORD *)(v339 + 24) = v208;
              *(_QWORD *)v339 = &unk_49F4D50;
            }
            *(_DWORD *)(v339 + 12) = *(unsigned __int8 *)(v208 + 16) - 24;
          }
          v185 = v208;
          sub_19F02E0(v394, v208, v308);
          if ( v440 != (__int64 *)v439.m128i_i64[1] )
            _libc_free((unsigned __int64)v440);
        }
        else
        {
          if ( *(_BYTE *)(*(_QWORD *)v208 + 8LL) )
          {
            v322 = sub_145CDC0(0x20u, v367);
            if ( v322 )
            {
              *(_DWORD *)(v322 + 8) = 4;
              *(_QWORD *)(v322 + 16) = 0;
              *(_QWORD *)(v322 + 24) = v208;
              *(_QWORD *)v322 = &unk_49F4D50;
            }
            *(_DWORD *)(v322 + 12) = *(unsigned __int8 *)(v208 + 16) - 24;
            sub_19F02E0(v394, v208, v322);
            v287 = *(_QWORD *)(v208 + 40);
            if ( (unsigned int)*(unsigned __int8 *)(v208 + 16) - 25 >= 0xA )
              v208 = 0;
          }
          else
          {
            v287 = *(_QWORD *)(v208 + 40);
          }
          v185 = v208;
          sub_19F1F00(v394, v208, v287);
        }
        goto LABEL_384;
      }
      v424[0] = v208;
      v425[0] = *(_QWORD *)(v208 + 64);
      v219 = 24LL * (*(_DWORD *)(v208 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v208 + 23) & 0x40) != 0 )
      {
        v220 = *(_QWORD *)(v208 - 8);
        v208 = v220 + v219;
      }
      else
      {
        v220 = v208 - v219;
      }
      v439.m128i_i64[0] = v208;
      v439.m128i_i64[1] = v208;
      v440 = v424;
      k = v394;
      v442 = v425;
      sub_19E9060((__int64)&v439);
      v431.m128i_i64[0] = v220;
      v431.m128i_i64[1] = v208;
      v432 = v424;
      v433 = v394;
      v434 = v425;
      sub_19E9060((__int64)&v431);
      v384 = v439.m128i_i64[0];
      if ( v431.m128i_i64[0] == v439.m128i_i64[0] )
      {
        v185 = v424[0];
        if ( !(unsigned __int8)sub_19E7150(v394, v424[0], *(_QWORD *)(v394 + 1432)) )
          goto LABEL_384;
        goto LABEL_415;
      }
      v440 = v432;
      v442 = v434;
      v443[0] = v394;
      v439 = v431;
      k = v433;
      v371 = v394 + 1960;
      v430[0] = *(__int64 **)v431.m128i_i64[0];
      if ( !(unsigned __int8)sub_19E11D0(v394 + 1960, (__int64 *)v430, &v431) )
        goto LABEL_554;
      v221 = *(__int64 **)(*(_QWORD *)(v431.m128i_i64[0] + 8) + 40LL);
      v439.m128i_i64[0] += 24;
      v373 = v221;
      sub_19E9060((__int64)&v439);
      v223 = v439.m128i_i64[1];
      v222 = (__int64 *)v439.m128i_i64[0];
      v224 = v440;
      v378 = v443[0];
      v225 = k;
      v226 = v442;
      while ( 1 )
      {
        if ( (__int64 *)v384 == v222 )
          goto LABEL_416;
        v229 = *(_DWORD *)(v378 + 1984);
        if ( !v229 )
          goto LABEL_554;
        v230 = *(_QWORD *)(v378 + 1968);
        v231 = v229 - 1;
        v232 = v231 & (((unsigned int)*v222 >> 9) ^ ((unsigned int)*v222 >> 4));
        v233 = (__int64 *)(v230 + 16LL * v232);
        v234 = *v233;
        if ( *v222 != *v233 )
        {
          v290 = 1;
          if ( v234 == -8 )
            goto LABEL_554;
          while ( 1 )
          {
            v291 = v290 + 1;
            v232 = v231 & (v290 + v232);
            v233 = (__int64 *)(v230 + 16LL * v232);
            v292 = *v233;
            if ( *v222 == *v233 )
              break;
            v290 = v291;
            if ( v292 == -8 )
              goto LABEL_554;
          }
        }
        if ( v373 != *(__int64 **)(v233[1] + 40) )
          break;
        while ( 1 )
        {
          v222 += 3;
          if ( (__int64 *)v223 == v222 )
            break;
          v235 = *v222;
          v236 = *v224;
          if ( *v222 != *v224 )
          {
            v237 = *(_DWORD *)(v225 + 1984);
            v238 = 0;
            if ( v237 )
            {
              v239 = v237 - 1;
              v240 = *(_QWORD *)(v225 + 1968);
              v241 = v239 & (((unsigned int)v235 >> 9) ^ ((unsigned int)v235 >> 4));
              v242 = (__int64 *)(v240 + 16LL * v241);
              v243 = *v242;
              if ( v235 == *v242 )
              {
LABEL_324:
                v238 = v242[1];
              }
              else
              {
                v244 = 1;
                while ( v243 != -8 )
                {
                  v321 = v244 + 1;
                  v241 = v239 & (v244 + v241);
                  v242 = (__int64 *)(v240 + 16LL * v241);
                  v243 = *v242;
                  if ( v235 == *v242 )
                    goto LABEL_324;
                  v244 = v321;
                }
                v238 = 0;
              }
            }
            if ( *(_QWORD *)(v225 + 1432) != v238 )
            {
              v227 = (*(_BYTE *)(v236 + 23) & 0x40) != 0
                   ? *(_QWORD *)(v236 - 8)
                   : v236 - 24LL * (*(_DWORD *)(v236 + 20) & 0xFFFFFFF);
              v228 = *v226;
              v431.m128i_i64[0] = *(_QWORD *)(v227
                                            + 0xFFFFFFFD55555558LL * (unsigned int)(((__int64)v222 - v227) >> 3)
                                            + 24LL * *(unsigned int *)(v236 + 76)
                                            + 8);
              v431.m128i_i64[1] = v228;
              if ( (unsigned __int8)sub_19E8F30(v225 + 2200, v431.m128i_i64, v430) )
                break;
            }
          }
        }
      }
      if ( (__int64 *)v384 == v222 )
      {
LABEL_416:
        v295 = 0;
        v430[0] = v373;
        if ( (unsigned __int8)sub_19E11D0(v371, (__int64 *)v430, &v431) )
          v295 = *(_QWORD *)(v431.m128i_i64[0] + 8);
        v297 = v394 + 1992;
        v430[0] = (__int64 *)v424[0];
        if ( !(unsigned __int8)sub_19E1280(v394 + 1992, (__int64 *)v430, &v431) )
        {
          v301 = 0;
          if ( (__int64 *)v384 != v222 )
          {
LABEL_431:
            v298 = 3;
            v294 = v301 != 3;
LABEL_421:
            v302 = sub_19E1280(v297, v424, &v431);
            v300 = v431.m128i_i64[0];
            if ( v302 )
              goto LABEL_427;
LABEL_422:
            v303 = *(_DWORD *)(v394 + 2008);
            ++*(_QWORD *)(v394 + 1992);
            v304 = v303 + 1;
            v305 = *(_DWORD *)(v394 + 2016);
            if ( 4 * v304 >= 3 * v305 )
            {
              v305 *= 2;
            }
            else if ( v305 - *(_DWORD *)(v394 + 2012) - v304 > v305 >> 3 )
            {
LABEL_424:
              *(_DWORD *)(v394 + 2008) = v304;
              if ( *(_QWORD *)v300 != -8 )
                --*(_DWORD *)(v394 + 2012);
              v306 = v424[0];
              *(_DWORD *)(v300 + 8) = 0;
              *(_QWORD *)v300 = v306;
LABEL_427:
              *(_DWORD *)(v300 + 8) = v298;
              v185 = v424[0];
              if ( !(unsigned __int8)sub_19E7150(v394, v424[0], v295) && !v294 )
                goto LABEL_384;
LABEL_415:
              v185 = v424[0];
              sub_19E5C50(v394, v424[0]);
              goto LABEL_384;
            }
            sub_19E4430(v297, v305);
            sub_19E1280(v297, v424, &v431);
            v300 = v431.m128i_i64[0];
            v304 = *(_DWORD *)(v394 + 2008) + 1;
            goto LABEL_424;
          }
LABEL_420:
          v298 = 2;
          v294 = v301 != 2;
          goto LABEL_421;
        }
      }
      else
      {
        v293 = v424[0];
        v430[0] = (__int64 *)v424[0];
        v294 = sub_19E11D0(v371, (__int64 *)v430, &v431);
        if ( !v294 )
LABEL_554:
          BUG();
        v295 = *(_QWORD *)(v431.m128i_i64[0] + 8);
        if ( v293 != *(_QWORD *)(v295 + 40) )
        {
          v296 = sub_19E13F0(v394, 0, 0);
          *((_QWORD *)v296 + 5) = v293;
          v295 = (__int64)v296;
        }
        v297 = v394 + 1992;
        v430[0] = (__int64 *)v424[0];
        if ( !(unsigned __int8)sub_19E1280(v394 + 1992, (__int64 *)v430, &v431) )
        {
          v298 = 3;
          v299 = sub_19E1280(v297, v424, &v431);
          v300 = v431.m128i_i64[0];
          if ( v299 )
          {
            *(_DWORD *)(v431.m128i_i64[0] + 8) = 3;
            sub_19E7150(v394, v424[0], v295);
            goto LABEL_415;
          }
          goto LABEL_422;
        }
      }
      v301 = *(_DWORD *)(v431.m128i_i64[0] + 8);
      if ( (__int64 *)v384 != v222 )
        goto LABEL_431;
      goto LABEL_420;
    }
    v209 = *(_QWORD **)(v394 + 2248);
    v210 = *(_QWORD **)(v394 + 2240);
    if ( v209 == v210 )
    {
      v211 = &v210[*(unsigned int *)(v394 + 2260)];
      if ( v210 == v211 )
      {
        v340 = *(_QWORD **)(v394 + 2240);
      }
      else
      {
        do
        {
          if ( v381 == *v210 )
            break;
          ++v210;
        }
        while ( v211 != v210 );
        v340 = v211;
      }
    }
    else
    {
      v185 = v381;
      v211 = &v209[*(unsigned int *)(v394 + 2256)];
      v210 = sub_16CC9F0(v369, v381);
      if ( v381 == *v210 )
      {
        v288 = *(_QWORD *)(v394 + 2248);
        if ( v288 == *(_QWORD *)(v394 + 2240) )
          v289 = *(unsigned int *)(v394 + 2260);
        else
          v289 = *(unsigned int *)(v394 + 2256);
        v340 = (_QWORD *)(v288 + 8 * v289);
      }
      else
      {
        v212 = *(_QWORD *)(v394 + 2248);
        if ( v212 != *(_QWORD *)(v394 + 2240) )
        {
          v210 = (_QWORD *)(v212 + 8LL * *(unsigned int *)(v394 + 2256));
          goto LABEL_304;
        }
        v210 = (_QWORD *)(v212 + 8LL * *(unsigned int *)(v394 + 2260));
        v340 = v210;
      }
    }
    while ( v340 != v210 && *v210 >= 0xFFFFFFFFFFFFFFFELL )
      ++v210;
LABEL_304:
    v213 = *(_DWORD *)(v394 + 2384);
    if ( !v213 )
    {
LABEL_387:
      if ( v210 == v211 )
        goto LABEL_384;
LABEL_307:
      v201 = *(_QWORD **)(v394 + 2336);
      goto LABEL_308;
    }
    v185 = *(_QWORD *)(v394 + 2368);
    v214 = v213 - 1;
    v215 = (v213 - 1) & (((unsigned int)v381 >> 9) ^ ((unsigned int)v381 >> 4));
    v216 = v185 + 16LL * v215;
    v217 = *(_QWORD *)v216;
    if ( *(_QWORD *)v216 != v381 )
    {
      v323 = 1;
      while ( v217 != -8 )
      {
        v324 = v323 + 1;
        v215 = v214 & (v323 + v215);
        v216 = v185 + 16LL * v215;
        v217 = *(_QWORD *)v216;
        if ( v381 == *(_QWORD *)v216 )
          goto LABEL_306;
        v323 = v324;
      }
      goto LABEL_387;
    }
LABEL_306:
    if ( v210 != v211 )
      goto LABEL_307;
    v314 = *(_DWORD *)(v216 + 8);
    v185 = *(unsigned int *)(v216 + 12);
    if ( v314 != (_DWORD)v185 )
    {
      v315 = v314 & 0x3F;
      v316 = (__int64 *)(*(_QWORD *)(v394 + 2336) + 8LL * (v314 >> 6));
      v317 = *v316;
      v318 = 1LL << v185;
      if ( v314 >> 6 == (unsigned int)v185 >> 6 )
      {
        *v316 = v317 & ~(v318 - (1LL << v315));
      }
      else
      {
        *v316 = v317 & ~(-1LL << v315);
        v319 = (v314 + 63) & 0xFFFFFFC0;
        for ( j = v319 + 64; j <= (unsigned int)v185; j += 64 )
        {
          *(_QWORD *)(*(_QWORD *)(v394 + 2336) + 8LL * ((j - 64) >> 6)) = 0;
          v319 = j;
        }
        if ( v319 < (unsigned int)v185 )
          *(_QWORD *)(*(_QWORD *)(v394 + 2336) + 8LL * (v319 >> 6)) &= -v318;
      }
    }
LABEL_384:
    v197 = *(_DWORD *)(v394 + 2352);
    v245 = v388 + 1;
    if ( v197 == v388 + 1 )
    {
      v9 = v394;
      v198 = v381;
      v199 = v388 + 64;
      goto LABEL_284;
    }
    v207 = v245 >> 6;
    v185 = (unsigned int)(v197 - 1) >> 6;
    if ( v245 >> 6 > (unsigned int)v185 )
    {
      v9 = v394;
      v198 = v381;
      v199 = v197 + 63;
      goto LABEL_284;
    }
    v198 = v381;
LABEL_333:
    v246 = v207;
    v247 = 64 - (v245 & 0x3F);
    v248 = 0xFFFFFFFFFFFFFFFFLL >> v247;
    if ( v247 == 64 )
      v248 = 0;
    v201 = *(_QWORD **)(v394 + 2336);
    v249 = ~v248;
    while ( 1 )
    {
      _RAX = v201[v246];
      if ( v207 == (_DWORD)v246 )
        _RAX = v249 & v201[v246];
      if ( (_DWORD)v246 == (_DWORD)v185 )
        _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v197;
      if ( _RAX )
        break;
      if ( (unsigned int)v185 < (unsigned int)++v246 )
        goto LABEL_343;
    }
    __asm { tzcnt   rax, rax }
    v388 = ((_DWORD)v246 << 6) + _RAX;
    if ( v388 != -1 )
      continue;
    break;
  }
LABEL_343:
  v199 = v197 + 63;
  v9 = v394;
  v200 = (unsigned int)(v197 + 63) >> 6;
  if ( v200 )
    goto LABEL_285;
LABEL_344:
  v252 = sub_19F7210(v9, a2, a3, a4, a5, v188, v189, a8, a9);
  v253 = *(_QWORD **)(v9 + 2712);
  if ( v253 == *(_QWORD **)(v9 + 2704) )
    v254 = *(unsigned int *)(v9 + 2724);
  else
    v254 = *(unsigned int *)(v9 + 2720);
  v255 = (__int64)&v253[v254];
  v439.m128i_i64[0] = *(_QWORD *)(v9 + 2712);
  v439.m128i_i64[1] = v255;
  if ( v253 != (_QWORD *)v255 )
  {
    while ( (unsigned __int64)(*v253 + 2LL) <= 1 )
    {
      v439.m128i_i64[0] = (__int64)++v253;
      if ( (_QWORD *)v255 == v253 )
        goto LABEL_349;
    }
    v326 = v439.m128i_i64[0];
    v440 = (__int64 *)(v9 + 2696);
    for ( k = *(_QWORD *)(v9 + 2696); v439.m128i_i64[0] != v255; v326 = v439.m128i_i64[0] )
    {
      v327 = *(_QWORD *)v326;
      if ( *(_QWORD *)(*(_QWORD *)v326 + 8LL) )
      {
        v185 = sub_1599EF0(*(__int64 ***)v327);
        sub_164D160(
          v327,
          v185,
          a2,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128_u64,
          *(double *)a5.m128i_i64,
          v328,
          v329,
          a8,
          a9);
      }
      if ( *(_QWORD *)(v327 + 40) )
        sub_15F20C0((_QWORD *)v327);
      v439.m128i_i64[0] += 8;
      sub_19E4730((__int64)&v439);
    }
  }
LABEL_349:
  v256 = *(_QWORD *)v9;
  v440 = (__int64 *)v9;
  v439.m128i_i64[0] = v256 + 72;
  v439.m128i_i64[1] = v256 + 72;
  sub_19E25D0(v439.m128i_i64);
  v257 = *(_QWORD *)(v256 + 80);
  v431.m128i_i64[1] = v256 + 72;
  v432 = (__int64 *)v9;
  v431.m128i_i64[0] = v257;
  sub_19E25D0(v431.m128i_i64);
  v385 = v9;
  v264 = v431.m128i_i64[0];
  v265 = v432;
  v395 = v431.m128i_i64[1];
  v389 = v439.m128i_i64[0];
  if ( v439.m128i_i64[0] == v431.m128i_i64[0] )
    goto LABEL_379;
LABEL_350:
  while ( 2 )
  {
    if ( !v264 )
      goto LABEL_554;
    v390 = v264 - 24;
    v266 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v264 + 16) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v266 != (_QWORD *)(v264 + 16) )
    {
      do
      {
        v267 = v266;
        v268 = *v266 & 0xFFFFFFFFFFFFFFF8LL;
        v269 = *(v266 - 2) == 0;
        v266 = (_QWORD *)v268;
        if ( !v269 )
        {
          v270 = sub_1599EF0((__int64 **)*(v267 - 3));
          sub_164D160(
            (__int64)(v267 - 3),
            v270,
            a2,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128_u64,
            *(double *)a5.m128i_i64,
            v271,
            v272,
            a8,
            a9);
        }
        if ( *((_BYTE *)v267 - 8) != 88 )
          sub_15F20C0(v267 - 3);
      }
      while ( v268 != v264 + 16 );
    }
    v273 = (_QWORD *)sub_157E9C0(v390);
    v274 = (__int64 **)sub_1643330(v273);
    v275 = sub_1599EF0(v274);
    v276 = (__int64 **)sub_1647190((__int64 *)v274, 0);
    v279 = sub_15A06D0(v276, 0, v277, v278);
    v185 = 2;
    v280 = sub_157EBA0(v390);
    v281 = sub_1648A60(64, 2u);
    if ( v281 )
    {
      v185 = v275;
      sub_15F9660((__int64)v281, v275, v279, v280);
    }
    v264 = *(_QWORD *)(v264 + 8);
    if ( v264 != v395 )
    {
      v259 = v265[281];
      do
      {
        v284 = v264 - 24;
        if ( !v264 )
          v284 = 0;
        v283 = (_QWORD *)v265[280];
        if ( (_QWORD *)v259 == v283 )
        {
          v285 = *((unsigned int *)v265 + 565);
          v282 = (_QWORD *)(v259 + 8 * v285);
          if ( (_QWORD *)v259 == v282 )
          {
            v258 = v259;
          }
          else
          {
            do
            {
              if ( v284 == *v283 )
                break;
              ++v283;
            }
            while ( v282 != v283 );
            v258 = v259 + 8 * v285;
          }
        }
        else
        {
          v185 = v284;
          v282 = (_QWORD *)(v259 + 8LL * *((unsigned int *)v265 + 564));
          v283 = sub_16CC9F0((__int64)(v265 + 279), v284);
          if ( v284 == *v283 )
          {
            v259 = v265[281];
            if ( v259 == v265[280] )
              v258 = v259 + 8LL * *((unsigned int *)v265 + 565);
            else
              v258 = v259 + 8LL * *((unsigned int *)v265 + 564);
          }
          else
          {
            v259 = v265[281];
            if ( v259 != v265[280] )
            {
              v283 = (_QWORD *)(v259 + 8LL * *((unsigned int *)v265 + 564));
LABEL_364:
              if ( v282 == v283 )
                goto LABEL_378;
              goto LABEL_365;
            }
            v258 = v259 + 8LL * *((unsigned int *)v265 + 565);
            v283 = (_QWORD *)v258;
          }
        }
        if ( v283 == (_QWORD *)v258 )
          goto LABEL_364;
        do
        {
          if ( *v283 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_364;
          ++v283;
        }
        while ( (_QWORD *)v258 != v283 );
        if ( v282 == v283 )
        {
LABEL_378:
          v252 = 1;
          if ( v389 == v264 )
          {
LABEL_379:
            v286 = v385;
            goto LABEL_465;
          }
          goto LABEL_350;
        }
LABEL_365:
        v264 = *(_QWORD *)(v264 + 8);
      }
      while ( v395 != v264 );
    }
    if ( v389 != v264 )
      continue;
    break;
  }
  v286 = v385;
  v252 = 1;
LABEL_465:
  sub_19E9650(
    v286,
    a2,
    *(double *)a3.m128i_i64,
    *(double *)a4.m128_u64,
    *(double *)a5.m128i_i64,
    v262,
    v263,
    a8,
    a9,
    v185,
    v258,
    v259,
    v260,
    v261);
  if ( v397 )
    j_j___libc_free_0(v397, v399 - v397);
  return v252;
}
