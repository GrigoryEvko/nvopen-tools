// Function: sub_1A45930
// Address: 0x1a45930
//
__int64 __fastcall sub_1A45930(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        __m128i a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 v11; // rsi
  int v12; // r9d
  double v13; // xmm4_8
  double v14; // xmm5_8
  __int64 v15; // rax
  _QWORD *v16; // r14
  _QWORD *v17; // r12
  __int64 v18; // rax
  char v20; // al
  _QWORD *v21; // rdx
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r8
  int v28; // r9d
  unsigned __int64 v29; // rax
  unsigned __int8 *v30; // rsi
  unsigned __int64 *v31; // rax
  int v32; // r8d
  int v33; // r9d
  double v34; // xmm4_8
  double v35; // xmm5_8
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // rdx
  unsigned __int8 *m; // rdx
  __int64 v39; // rbx
  __m128i *v40; // rdx
  char v41; // al
  _BYTE *v42; // rax
  __int64 **v43; // rdx
  __int64 v44; // r14
  unsigned __int8 *v45; // r13
  __int64 v46; // rax
  __m128i v47; // rax
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // r8
  int v52; // r9d
  __int64 v53; // rax
  unsigned __int8 *v54; // rsi
  unsigned __int64 *v55; // rax
  __int64 v56; // r8
  int v57; // r9d
  _QWORD *v58; // rax
  int v59; // r8d
  int v60; // r9d
  double v61; // xmm4_8
  double v62; // xmm5_8
  unsigned __int64 **v63; // rax
  unsigned __int64 **v64; // rdx
  unsigned __int64 **n; // rdx
  __int64 v66; // rbx
  __m128i *v67; // rdx
  char v68; // al
  _BYTE *v69; // r13
  _BYTE *v70; // r8
  _QWORD *v71; // r14
  __int16 v72; // r15
  _QWORD *v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // r8
  int v77; // r9d
  int v78; // r9d
  double v79; // xmm4_8
  double v80; // xmm5_8
  __int32 v81; // r8d
  unsigned __int64 **v82; // rax
  unsigned __int64 **v83; // rdx
  unsigned __int64 **kk; // rdx
  __int64 v85; // r12
  signed int v86; // esi
  __int64 v87; // rax
  _QWORD *v88; // r13
  __int64 v89; // rax
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 v92; // r8
  int v93; // r9d
  __int64 v94; // rax
  unsigned __int8 *v95; // rsi
  unsigned __int64 *v96; // rax
  __int64 v97; // r8
  int v98; // r9d
  _QWORD *v99; // rax
  int v100; // r8d
  int v101; // r9d
  unsigned __int64 **v102; // rax
  unsigned __int64 **v103; // rdx
  unsigned __int64 **ii; // rdx
  __int64 v105; // rbx
  __m128i *v106; // rdx
  char v107; // al
  _BYTE *v108; // r13
  _BYTE *v109; // r8
  _QWORD *v110; // r14
  __int16 v111; // r15
  _QWORD *v112; // r12
  __int64 v113; // rdx
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 v116; // rbx
  __int64 v117; // rax
  __int64 v118; // r8
  int v119; // r9d
  unsigned __int64 v120; // rax
  unsigned __int8 *v121; // rsi
  int v122; // r8d
  int v123; // r9d
  unsigned __int8 *v124; // rdx
  unsigned __int8 *v125; // rax
  unsigned __int8 *i; // rdx
  __int64 v127; // r13
  __m128i *v128; // rdx
  char v129; // al
  unsigned __int64 v130; // rbx
  unsigned __int8 *v131; // r14
  _QWORD *v132; // rax
  _QWORD *v133; // r12
  unsigned __int64 v134; // rsi
  __int64 v135; // rax
  __int64 v136; // rsi
  __int64 v137; // rdx
  unsigned __int8 *v138; // rsi
  __m128i v139; // rax
  __int64 v140; // rax
  __int64 v141; // rbx
  __int64 v142; // rax
  __int64 v143; // r8
  __int64 v144; // r9
  double v145; // xmm4_8
  double v146; // xmm5_8
  unsigned __int64 v147; // rax
  unsigned __int8 *v148; // rsi
  unsigned __int8 *v149; // rdx
  unsigned __int8 *v150; // rax
  unsigned __int8 *jj; // rdx
  __int64 v152; // rbx
  __m128i *v153; // rdx
  char v154; // al
  __int64 v155; // rsi
  unsigned __int8 *v156; // r14
  __int64 v157; // rax
  __int64 v158; // r15
  __int64 v159; // r13
  __int64 v160; // rsi
  __int64 v161; // rax
  __int64 v162; // rsi
  __int64 v163; // rdx
  unsigned __int8 *v164; // rsi
  __m128i v165; // rax
  __int64 v166; // rax
  __int64 v167; // r13
  __int64 v168; // rax
  __int64 v169; // r8
  int v170; // r9d
  unsigned __int64 v171; // rax
  unsigned __int8 *v172; // rsi
  unsigned __int64 *v173; // rax
  int v174; // r8d
  int v175; // r9d
  unsigned __int8 *v176; // rax
  unsigned __int8 *v177; // rdx
  unsigned __int8 *k; // rdx
  __int64 v179; // rbx
  __m128i *v180; // rdx
  char v181; // al
  _BYTE *v182; // rax
  __int64 **v183; // rdx
  __int64 v184; // r14
  unsigned __int8 *v185; // r13
  __int64 v186; // rax
  __m128i v187; // rax
  __int64 *v188; // r13
  __int64 v189; // rax
  __int64 v190; // rax
  __int64 v191; // rbx
  __int64 v192; // rax
  __int64 v193; // r8
  int v194; // r9d
  __int64 v195; // rax
  unsigned __int8 *v196; // rsi
  __int64 v197; // r8
  int v198; // r9d
  int v199; // r8d
  int v200; // r9d
  unsigned __int64 **v201; // rdx
  unsigned __int64 **v202; // rax
  unsigned __int64 **j; // rdx
  __int64 v204; // r13
  unsigned __int64 v205; // rbx
  _BYTE *v206; // r15
  _QWORD *v207; // r12
  _QWORD *v208; // rax
  _QWORD *v209; // r14
  unsigned __int64 v210; // rsi
  __int64 v211; // rax
  __int64 v212; // rsi
  __int64 v213; // rdx
  unsigned __int8 *v214; // rsi
  unsigned __int64 v215; // rdi
  __int64 v216; // r13
  __int64 **v217; // rbx
  _QWORD *v218; // r15
  __int64 v219; // rax
  __int64 v220; // r14
  __int64 *v221; // rbx
  __int64 v222; // r12
  _QWORD *v223; // rax
  unsigned __int8 *v224; // rsi
  __int64 v225; // rax
  __int64 v226; // rbx
  __m128i *v227; // rdx
  char v228; // al
  __int64 v229; // rax
  __int64 v230; // r10
  __int64 v231; // rax
  __m128i v232; // rax
  _QWORD *v233; // rax
  _QWORD *v234; // r14
  unsigned __int64 *v235; // r12
  __int64 v236; // rax
  unsigned __int64 v237; // rcx
  __int64 v238; // rsi
  __int64 v239; // rdx
  unsigned __int8 *v240; // rsi
  double v241; // xmm4_8
  double v242; // xmm5_8
  __int64 *v243; // r13
  _QWORD *v244; // rdx
  _QWORD *v245; // rdx
  __int64 v246; // r13
  __int64 v247; // rbx
  __int64 v248; // rcx
  __int64 v249; // r14
  __int64 v250; // rdx
  _BYTE *v251; // rcx
  int v252; // eax
  __int64 v253; // rax
  int v254; // edx
  __int64 v255; // rdx
  _QWORD *v256; // rax
  __int64 v257; // rsi
  unsigned __int64 v258; // rdx
  __int64 v259; // rdx
  __int64 v260; // rdx
  __int64 v261; // rsi
  __int64 v262; // rsi
  __int64 v263; // rax
  __int64 v264; // rsi
  __int64 v265; // rdx
  unsigned __int8 *v266; // rsi
  _QWORD *v267; // rax
  _BYTE *v268; // r8
  _QWORD **v269; // rax
  __int64 *v270; // rax
  __int64 v271; // rax
  __int64 v272; // r8
  int v273; // r13d
  unsigned __int64 *v274; // r13
  __int64 v275; // rax
  unsigned __int64 v276; // rcx
  __int64 v277; // rsi
  unsigned __int8 *v278; // rsi
  __int64 v279; // rsi
  __int64 v280; // rax
  __int64 v281; // rsi
  __int64 v282; // rdx
  unsigned __int8 *v283; // rsi
  _QWORD *v284; // rax
  _BYTE *v285; // r8
  _QWORD **v286; // rax
  __int64 *v287; // rax
  __int64 v288; // rax
  __int64 v289; // r8
  unsigned __int64 *v290; // r13
  __int64 v291; // rax
  unsigned __int64 v292; // rcx
  __int64 v293; // rsi
  unsigned __int8 *v294; // rsi
  __int64 v295; // rax
  unsigned __int8 *v296; // rsi
  unsigned __int8 *v297; // rsi
  __int64 v298; // [rsp+18h] [rbp-328h]
  __int64 v299; // [rsp+18h] [rbp-328h]
  _QWORD *v300; // [rsp+20h] [rbp-320h]
  _QWORD *v301; // [rsp+20h] [rbp-320h]
  __int64 v302; // [rsp+20h] [rbp-320h]
  _QWORD *v303; // [rsp+20h] [rbp-320h]
  __int64 v304; // [rsp+20h] [rbp-320h]
  unsigned __int8 v305; // [rsp+2Fh] [rbp-311h]
  _QWORD *v306; // [rsp+38h] [rbp-308h]
  _QWORD *v307; // [rsp+48h] [rbp-2F8h]
  __int64 v308; // [rsp+48h] [rbp-2F8h]
  _BYTE *v309; // [rsp+48h] [rbp-2F8h]
  __int64 v310; // [rsp+48h] [rbp-2F8h]
  _BYTE *v311; // [rsp+48h] [rbp-2F8h]
  __int64 v312; // [rsp+48h] [rbp-2F8h]
  unsigned int v313; // [rsp+50h] [rbp-2F0h]
  _QWORD *v314; // [rsp+50h] [rbp-2F0h]
  __int64 v315; // [rsp+58h] [rbp-2E8h]
  __int64 v316; // [rsp+58h] [rbp-2E8h]
  __int64 v317; // [rsp+58h] [rbp-2E8h]
  __int64 *v318; // [rsp+58h] [rbp-2E8h]
  __int64 *v319; // [rsp+58h] [rbp-2E8h]
  __int64 v320; // [rsp+60h] [rbp-2E0h]
  __int64 v321; // [rsp+68h] [rbp-2D8h]
  _QWORD *v322; // [rsp+70h] [rbp-2D0h]
  _QWORD *v323; // [rsp+70h] [rbp-2D0h]
  __int64 v324; // [rsp+70h] [rbp-2D0h]
  __int64 v325; // [rsp+70h] [rbp-2D0h]
  __int64 v326; // [rsp+78h] [rbp-2C8h]
  __int64 v327; // [rsp+78h] [rbp-2C8h]
  __int64 v328; // [rsp+78h] [rbp-2C8h]
  _BYTE *v329; // [rsp+78h] [rbp-2C8h]
  unsigned __int64 *v330; // [rsp+78h] [rbp-2C8h]
  unsigned int v331; // [rsp+78h] [rbp-2C8h]
  __int64 v332; // [rsp+78h] [rbp-2C8h]
  _BYTE *v333; // [rsp+78h] [rbp-2C8h]
  __int64 *v334; // [rsp+80h] [rbp-2C0h]
  __int64 v335; // [rsp+88h] [rbp-2B8h]
  __int64 *v336; // [rsp+88h] [rbp-2B8h]
  __int64 v337; // [rsp+88h] [rbp-2B8h]
  _QWORD *v338; // [rsp+88h] [rbp-2B8h]
  __int64 v339; // [rsp+90h] [rbp-2B0h]
  __int64 v340; // [rsp+90h] [rbp-2B0h]
  __int64 v341; // [rsp+98h] [rbp-2A8h]
  _BYTE *v342; // [rsp+98h] [rbp-2A8h]
  unsigned __int64 *v343; // [rsp+98h] [rbp-2A8h]
  __int64 v344; // [rsp+98h] [rbp-2A8h]
  unsigned __int64 v345; // [rsp+98h] [rbp-2A8h]
  _QWORD *v346; // [rsp+A0h] [rbp-2A0h]
  __int64 v347; // [rsp+A0h] [rbp-2A0h]
  __int64 *v349; // [rsp+A8h] [rbp-298h]
  unsigned __int8 *v350; // [rsp+B8h] [rbp-288h] BYREF
  const char *v351; // [rsp+C0h] [rbp-280h] BYREF
  __int64 v352; // [rsp+C8h] [rbp-278h]
  __int64 v353; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v354; // [rsp+D8h] [rbp-268h]
  __int64 v355; // [rsp+E0h] [rbp-260h]
  __m128i v356; // [rsp+F0h] [rbp-250h] BYREF
  unsigned __int64 *v357; // [rsp+100h] [rbp-240h]
  __m128i v358; // [rsp+110h] [rbp-230h] BYREF
  unsigned __int64 *v359; // [rsp+120h] [rbp-220h]
  __m128i v360; // [rsp+130h] [rbp-210h] BYREF
  unsigned __int64 *v361; // [rsp+140h] [rbp-200h]
  __m128i v362; // [rsp+150h] [rbp-1F0h] BYREF
  unsigned __int64 *v363; // [rsp+160h] [rbp-1E0h]
  __int64 v364; // [rsp+168h] [rbp-1D8h]
  __m128i v365; // [rsp+170h] [rbp-1D0h] BYREF
  unsigned __int64 *v366; // [rsp+180h] [rbp-1C0h]
  __int64 v367; // [rsp+188h] [rbp-1B8h]
  __int64 v368; // [rsp+190h] [rbp-1B0h]
  int v369; // [rsp+198h] [rbp-1A8h]
  __int64 v370; // [rsp+1A0h] [rbp-1A0h]
  __int64 v371; // [rsp+1A8h] [rbp-198h]
  __m128 v372; // [rsp+1C0h] [rbp-180h] BYREF
  unsigned __int64 *v373; // [rsp+1D0h] [rbp-170h] BYREF
  __int64 v374; // [rsp+1D8h] [rbp-168h]
  __int64 v375; // [rsp+1E0h] [rbp-160h]
  int v376; // [rsp+1E8h] [rbp-158h]
  __int64 v377; // [rsp+1F0h] [rbp-150h]
  __int64 v378; // [rsp+1F8h] [rbp-148h]
  unsigned __int8 *v379; // [rsp+210h] [rbp-130h] BYREF
  __int64 v380; // [rsp+218h] [rbp-128h]
  _WORD v381[12]; // [rsp+220h] [rbp-120h] BYREF
  _BYTE *v382; // [rsp+238h] [rbp-108h]
  _BYTE v383[64]; // [rsp+248h] [rbp-F8h] BYREF
  unsigned int v384; // [rsp+288h] [rbp-B8h]
  unsigned __int8 *v385; // [rsp+290h] [rbp-B0h] BYREF
  __int64 v386; // [rsp+298h] [rbp-A8h]
  unsigned __int64 *v387; // [rsp+2A0h] [rbp-A0h]
  _QWORD *v388; // [rsp+2A8h] [rbp-98h]
  __int64 v389; // [rsp+2B0h] [rbp-90h]
  _QWORD *v390; // [rsp+2B8h] [rbp-88h]
  __int64 v391; // [rsp+2C0h] [rbp-80h]
  _QWORD v392[15]; // [rsp+2C8h] [rbp-78h] BYREF

  v10 = a1;
  v305 = sub_1636880(a1, a2);
  if ( v305 )
    return 0;
  v11 = *(_QWORD *)(a2 + 80);
  v353 = 0;
  v354 = 0;
  v355 = 0;
  if ( v11 )
    v11 -= 24;
  sub_1A45460((__int64)&v353, v11);
  v339 = v354;
  v320 = v353;
  if ( v354 != v353 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(v339 - 8);
      v16 = *(_QWORD **)(v15 + 48);
      v346 = (_QWORD *)(v15 + 40);
      if ( v16 == (_QWORD *)(v15 + 40) )
        goto LABEL_13;
      do
      {
        if ( !v16 )
          BUG();
        v17 = v16 - 3;
        switch ( *((_BYTE *)v16 - 8) )
        {
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x21:
          case 0x22:
          case 0x35:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x49:
          case 0x4A:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x56:
          case 0x57:
          case 0x58:
            goto LABEL_11;
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
            v385 = (unsigned __int8 *)(v16 - 3);
            v20 = sub_1A41720(
                    a1,
                    v16 - 3,
                    (__int64 *)&v385,
                    a3,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v13,
                    v14,
                    *(double *)a9.m128i_i64,
                    a10);
            goto LABEL_22;
          case 0x36:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            if ( !*(_BYTE *)(a1 + 516) )
              goto LABEL_11;
            if ( sub_15F32D0((__int64)(v16 - 3)) )
              goto LABEL_11;
            if ( (*((_BYTE *)v16 - 6) & 1) != 0 )
              goto LABEL_11;
            v365 = 0u;
            v366 = 0;
            v367 = 0;
            v114 = sub_15F2050((__int64)(v16 - 3));
            v115 = sub_1632FA0(v114);
            if ( !(unsigned __int8)sub_1A40240(
                                     *(v16 - 3),
                                     1 << (*((unsigned __int16 *)v16 - 3) >> 1) >> 1,
                                     v365.m128i_i64,
                                     v115) )
              goto LABEL_11;
            v116 = *(_QWORD *)(v365.m128i_i64[0] + 32);
            v117 = sub_16498A0((__int64)(v16 - 3));
            v372.m128_u64[0] = 0;
            v374 = v117;
            v375 = 0;
            v376 = 0;
            v377 = 0;
            v378 = 0;
            v120 = v16[2];
            v373 = v16;
            v372.m128_u64[1] = v120;
            v121 = (unsigned __int8 *)v16[3];
            v385 = v121;
            if ( v121 )
            {
              sub_1623A60((__int64)&v385, (__int64)v121, 2);
              if ( v372.m128_u64[0] )
                sub_161E7C0((__int64)&v372, v372.m128_i64[0]);
              v372.m128_u64[0] = (unsigned __int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v372);
            }
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), *(v16 - 6), v118, v119);
            v379 = (unsigned __int8 *)v381;
            v380 = 0x800000000LL;
            v324 = (unsigned int)v116;
            if ( (_DWORD)v116 )
            {
              v124 = (unsigned __int8 *)v381;
              v125 = (unsigned __int8 *)v381;
              if ( (unsigned int)v116 > 8uLL )
              {
                sub_16CD150((__int64)&v379, v381, (unsigned int)v116, 8, v122, v123);
                v124 = v379;
                v125 = &v379[8 * (unsigned int)v380];
              }
              for ( i = &v124[8 * (unsigned int)v116]; i != v125; v125 += 8 )
              {
                if ( v125 )
                  *(_QWORD *)v125 = 0;
              }
              LODWORD(v380) = v116;
            }
            if ( (_DWORD)v116 )
            {
              v306 = v16;
              v127 = 0;
              v316 = (__int64)(v16 - 3);
              do
              {
                v360.m128i_i32[0] = v127;
                LOWORD(v361) = 265;
                v139.m128i_i64[0] = (__int64)sub_1649960(v316);
                v356 = v139;
                v358.m128i_i64[0] = (__int64)&v356;
                LOWORD(v359) = 773;
                v358.m128i_i64[1] = (__int64)".i";
                v129 = (char)v361;
                if ( (_BYTE)v361 )
                {
                  if ( (_BYTE)v361 == 1 )
                  {
                    a9 = _mm_loadu_si128(&v358);
                    v362 = a9;
                    v363 = v359;
                  }
                  else
                  {
                    v128 = (__m128i *)v360.m128i_i64[0];
                    if ( BYTE1(v361) != 1 )
                    {
                      v128 = &v360;
                      v129 = 2;
                    }
                    v362.m128i_i64[1] = (__int64)v128;
                    LOBYTE(v363) = 2;
                    v362.m128i_i64[0] = (__int64)&v358;
                    BYTE1(v363) = v129;
                  }
                }
                else
                {
                  LOWORD(v363) = 256;
                }
                v130 = -(__int64)((unsigned __int64)v366 | (v127 * v367)) & ((unsigned __int64)v366 | (v127 * v367));
                v329 = sub_1A3F820((__int64 *)&v385, v127);
                v131 = &v379[8 * v127];
                v132 = sub_1648A60(64, 1u);
                v133 = v132;
                if ( v132 )
                  sub_15F9210((__int64)v132, *(_QWORD *)(*(_QWORD *)v329 + 24LL), (__int64)v329, 0, 0, 0);
                if ( v372.m128_u64[1] )
                {
                  v330 = v373;
                  sub_157E9D0(v372.m128_u64[1] + 40, (__int64)v133);
                  v134 = *v330;
                  v135 = v133[3] & 7LL;
                  v133[4] = v330;
                  v134 &= 0xFFFFFFFFFFFFFFF8LL;
                  v133[3] = v134 | v135;
                  *(_QWORD *)(v134 + 8) = v133 + 3;
                  *v330 = *v330 & 7 | (unsigned __int64)(v133 + 3);
                }
                sub_164B780((__int64)v133, v362.m128i_i64);
                if ( v372.m128_u64[0] )
                {
                  v351 = (const char *)v372.m128_u64[0];
                  sub_1623A60((__int64)&v351, v372.m128_i64[0], 2);
                  v136 = v133[6];
                  v137 = (__int64)(v133 + 6);
                  if ( v136 )
                  {
                    sub_161E7C0((__int64)(v133 + 6), v136);
                    v137 = (__int64)(v133 + 6);
                  }
                  v138 = (unsigned __int8 *)v351;
                  v133[6] = v351;
                  if ( v138 )
                    sub_1623210((__int64)&v351, v138, v137);
                }
                ++v127;
                sub_15F8F50((__int64)v133, v130);
                *(_QWORD *)v131 = v133;
              }
              while ( v324 != v127 );
              v16 = v306;
              v17 = (_QWORD *)v316;
            }
            goto LABEL_359;
          case 0x37:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            if ( !*(_BYTE *)(a1 + 516) )
              goto LABEL_11;
            if ( sub_15F32D0((__int64)(v16 - 3)) )
              goto LABEL_11;
            if ( (*((_BYTE *)v16 - 6) & 1) != 0 )
              goto LABEL_11;
            v362 = 0u;
            v363 = 0;
            v364 = 0;
            v188 = (__int64 *)*(v16 - 9);
            v189 = sub_15F2050((__int64)(v16 - 3));
            v190 = sub_1632FA0(v189);
            if ( !(unsigned __int8)sub_1A40240(
                                     *v188,
                                     1 << (*((unsigned __int16 *)v16 - 3) >> 1) >> 1,
                                     v362.m128i_i64,
                                     v190) )
              goto LABEL_11;
            v191 = *(_QWORD *)(v362.m128i_i64[0] + 32);
            v192 = sub_16498A0((__int64)(v16 - 3));
            v365.m128i_i64[0] = 0;
            v367 = v192;
            v368 = 0;
            v369 = 0;
            v370 = 0;
            v371 = 0;
            v195 = v16[2];
            v366 = v16;
            v365.m128i_i64[1] = v195;
            v196 = (unsigned __int8 *)v16[3];
            v385 = v196;
            if ( v196 )
            {
              sub_1623A60((__int64)&v385, (__int64)v196, 2);
              if ( v365.m128i_i64[0] )
                sub_161E7C0((__int64)&v365, v365.m128i_i64[0]);
              v365.m128i_i64[0] = (__int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v365);
            }
            sub_1A41500((__int64)&v379, (_QWORD *)a1, (__int64)(v16 - 3), *(v16 - 6), v193, v194);
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), (unsigned __int64)v188, v197, v198);
            v372.m128_u64[0] = (unsigned __int64)&v373;
            v372.m128_u64[1] = 0x800000000LL;
            v337 = (unsigned int)v191;
            if ( (_DWORD)v191 )
            {
              v201 = &v373;
              v202 = &v373;
              if ( (unsigned int)v191 > 8uLL )
              {
                sub_16CD150((__int64)&v372, &v373, (unsigned int)v191, 8, v199, v200);
                v201 = (unsigned __int64 **)v372.m128_u64[0];
                v202 = (unsigned __int64 **)(v372.m128_u64[0] + 8LL * v372.m128_u32[2]);
              }
              for ( j = &v201[(unsigned int)v191]; j != v202; ++v202 )
              {
                if ( v202 )
                  *v202 = 0;
              }
              v372.m128_i32[2] = v191;
            }
            v204 = 0;
            if ( (_DWORD)v191 )
            {
              v314 = v16;
              v307 = v16 - 3;
              do
              {
                v205 = -(__int64)((unsigned __int64)v363 | (v204 * v364)) & ((unsigned __int64)v363 | (v204 * v364));
                v342 = sub_1A3F820((__int64 *)&v379, v204);
                v206 = sub_1A3F820((__int64 *)&v385, v204);
                LOWORD(v361) = 257;
                v207 = (_QWORD *)(v372.m128_u64[0] + 8 * v204);
                v208 = sub_1648A60(64, 2u);
                v209 = v208;
                if ( v208 )
                  sub_15F9650((__int64)v208, (__int64)v206, (__int64)v342, 0, 0);
                if ( v365.m128i_i64[1] )
                {
                  v343 = v366;
                  sub_157E9D0(v365.m128i_i64[1] + 40, (__int64)v209);
                  v210 = *v343;
                  v211 = v209[3] & 7LL;
                  v209[4] = v343;
                  v210 &= 0xFFFFFFFFFFFFFFF8LL;
                  v209[3] = v210 | v211;
                  *(_QWORD *)(v210 + 8) = v209 + 3;
                  *v343 = *v343 & 7 | (unsigned __int64)(v209 + 3);
                }
                sub_164B780((__int64)v209, v360.m128i_i64);
                if ( v365.m128i_i64[0] )
                {
                  v358.m128i_i64[0] = v365.m128i_i64[0];
                  sub_1623A60((__int64)&v358, v365.m128i_i64[0], 2);
                  v212 = v209[6];
                  v213 = (__int64)(v209 + 6);
                  if ( v212 )
                  {
                    sub_161E7C0((__int64)(v209 + 6), v212);
                    v213 = (__int64)(v209 + 6);
                  }
                  v214 = (unsigned __int8 *)v358.m128i_i64[0];
                  v209[6] = v358.m128i_i64[0];
                  if ( v214 )
                    sub_1623210((__int64)&v358, v214, v213);
                }
                ++v204;
                sub_15F9450((__int64)v209, v205);
                *v207 = v209;
              }
              while ( v337 != v204 );
              v16 = v314;
              v17 = v307;
            }
            sub_1A3F630(a1, (__int64)v17, (__int64)&v372);
            v215 = v372.m128_u64[0];
            if ( (unsigned __int64 **)v372.m128_u64[0] == &v373 )
              goto LABEL_388;
            goto LABEL_387;
          case 0x38:
            v20 = sub_1A41D90(
                    a1,
                    v16 - 3,
                    a3,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v13,
                    v14,
                    *(double *)a9.m128i_i64,
                    a10);
            goto LABEL_22;
          case 0x3C:
          case 0x3D:
          case 0x3E:
          case 0x41:
          case 0x42:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x48:
            v23 = sub_1A428B0(
                    a1,
                    (unsigned __int64)(v16 - 3),
                    a3,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v13,
                    v14,
                    *(double *)a9.m128i_i64,
                    a10);
            v21 = (_QWORD *)v16[1];
            if ( !v23 )
              goto LABEL_26;
            goto LABEL_23;
          case 0x3F:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            v166 = *(v16 - 3);
            if ( *(_BYTE *)(v166 + 8) != 16 )
              goto LABEL_11;
            v167 = *(_QWORD *)(v166 + 32);
            v168 = sub_16498A0((__int64)(v16 - 3));
            v372.m128_u64[0] = 0;
            v374 = v168;
            v375 = 0;
            v376 = 0;
            v377 = 0;
            v378 = 0;
            v171 = v16[2];
            v373 = v16;
            v372.m128_u64[1] = v171;
            v172 = (unsigned __int8 *)v16[3];
            v385 = v172;
            if ( v172 )
            {
              sub_1623A60((__int64)&v385, (__int64)v172, 2);
              if ( v372.m128_u64[0] )
                sub_161E7C0((__int64)&v372, v372.m128_i64[0]);
              v372.m128_u64[0] = (unsigned __int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v372);
            }
            if ( (*((_BYTE *)v16 - 1) & 0x40) != 0 )
              v173 = (unsigned __int64 *)*(v16 - 4);
            else
              v173 = &v17[-3 * (*((_DWORD *)v16 - 1) & 0xFFFFFFF)];
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), *v173, v169, v170);
            v380 = 0x800000000LL;
            v176 = (unsigned __int8 *)v381;
            v177 = (unsigned __int8 *)v381;
            v379 = (unsigned __int8 *)v381;
            v332 = (unsigned int)v167;
            if ( (_DWORD)v167 )
            {
              if ( (unsigned int)v167 > 8uLL )
              {
                sub_16CD150((__int64)&v379, v381, (unsigned int)v167, 8, v174, v175);
                v177 = v379;
                v176 = &v379[8 * (unsigned int)v380];
              }
              for ( k = &v177[8 * (unsigned int)v167]; k != v176; v176 += 8 )
              {
                if ( v176 )
                  *(_QWORD *)v176 = 0;
              }
              LODWORD(v380) = v167;
            }
            v179 = 0;
            if ( !(_DWORD)v167 )
              goto LABEL_359;
            v322 = v16;
            do
            {
              v360.m128i_i32[0] = v179;
              LOWORD(v361) = 265;
              v187.m128i_i64[0] = (__int64)sub_1649960((__int64)v17);
              v356 = v187;
              LOWORD(v359) = 773;
              v358.m128i_i64[0] = (__int64)&v356;
              v358.m128i_i64[1] = (__int64)".i";
              v181 = (char)v361;
              if ( (_BYTE)v361 )
              {
                if ( (_BYTE)v361 == 1 )
                {
                  a4 = _mm_loadu_si128(&v358);
                  v362 = a4;
                  v363 = v359;
                }
                else
                {
                  v180 = (__m128i *)v360.m128i_i64[0];
                  if ( BYTE1(v361) != 1 )
                  {
                    v180 = &v360;
                    v181 = 2;
                  }
                  v362.m128i_i64[1] = (__int64)v180;
                  LOBYTE(v363) = 2;
                  v362.m128i_i64[0] = (__int64)&v358;
                  BYTE1(v363) = v181;
                }
              }
              else
              {
                LOWORD(v363) = 256;
              }
              v182 = sub_1A3F820((__int64 *)&v385, v179);
              v183 = 0;
              v184 = (__int64)v182;
              v185 = &v379[8 * v179];
              v186 = *(v322 - 3);
              if ( *(_BYTE *)(v186 + 8) == 16 )
                v183 = *(__int64 ***)(v186 + 24);
              if ( v183 != *(__int64 ***)v184 )
              {
                if ( *(_BYTE *)(v184 + 16) > 0x10u )
                {
                  LOWORD(v366) = 257;
                  v184 = sub_15FDBD0(39, v184, (__int64)v183, (__int64)&v365, 0);
                  if ( v372.m128_u64[1] )
                  {
                    v319 = (__int64 *)v373;
                    sub_157E9D0(v372.m128_u64[1] + 40, v184);
                    v279 = *v319;
                    v280 = *(_QWORD *)(v184 + 24) & 7LL;
                    *(_QWORD *)(v184 + 32) = v319;
                    v279 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v184 + 24) = v279 | v280;
                    *(_QWORD *)(v279 + 8) = v184 + 24;
                    *v319 = *v319 & 7 | (v184 + 24);
                  }
                  sub_164B780(v184, v362.m128i_i64);
                  if ( v372.m128_u64[0] )
                  {
                    v351 = (const char *)v372.m128_u64[0];
                    sub_1623A60((__int64)&v351, v372.m128_i64[0], 2);
                    v281 = *(_QWORD *)(v184 + 48);
                    v282 = v184 + 48;
                    if ( v281 )
                    {
                      sub_161E7C0(v184 + 48, v281);
                      v282 = v184 + 48;
                    }
                    v283 = (unsigned __int8 *)v351;
                    *(_QWORD *)(v184 + 48) = v351;
                    if ( v283 )
                      sub_1623210((__int64)&v351, v283, v282);
                  }
                }
                else
                {
                  v184 = sub_15A46C0(39, (__int64 ***)v184, v183, 0);
                }
              }
              *(_QWORD *)v185 = v184;
              ++v179;
            }
            while ( v332 != v179 );
            goto LABEL_358;
          case 0x40:
            v12 = *(_DWORD *)(a1 + 496);
            if ( v12 && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            v24 = *(v16 - 3);
            if ( *(_BYTE *)(v24 + 8) != 16 )
              goto LABEL_11;
            v25 = *(_QWORD *)(v24 + 32);
            v26 = sub_16498A0((__int64)(v16 - 3));
            v372.m128_u64[0] = 0;
            v374 = v26;
            v375 = 0;
            v376 = 0;
            v377 = 0;
            v378 = 0;
            v29 = v16[2];
            v373 = v16;
            v372.m128_u64[1] = v29;
            v30 = (unsigned __int8 *)v16[3];
            v385 = v30;
            if ( v30 )
            {
              sub_1623A60((__int64)&v385, (__int64)v30, 2);
              if ( v372.m128_u64[0] )
                sub_161E7C0((__int64)&v372, v372.m128_i64[0]);
              v372.m128_u64[0] = (unsigned __int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v372);
            }
            if ( (*((_BYTE *)v16 - 1) & 0x40) != 0 )
              v31 = (unsigned __int64 *)*(v16 - 4);
            else
              v31 = &v17[-3 * (*((_DWORD *)v16 - 1) & 0xFFFFFFF)];
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), *v31, v27, v28);
            v380 = 0x800000000LL;
            v36 = (unsigned __int8 *)v381;
            v37 = (unsigned __int8 *)v381;
            v379 = (unsigned __int8 *)v381;
            v326 = (unsigned int)v25;
            if ( (_DWORD)v25 )
            {
              if ( (unsigned int)v25 > 8uLL )
              {
                sub_16CD150((__int64)&v379, v381, (unsigned int)v25, 8, v32, v33);
                v37 = v379;
                v36 = &v379[8 * (unsigned int)v380];
              }
              for ( m = &v37[8 * (unsigned int)v25]; m != v36; v36 += 8 )
              {
                if ( v36 )
                  *(_QWORD *)v36 = 0;
              }
              LODWORD(v380) = v25;
            }
            v39 = 0;
            if ( !(_DWORD)v25 )
              goto LABEL_359;
            v322 = v16;
            do
            {
              v360.m128i_i32[0] = v39;
              LOWORD(v361) = 265;
              v47.m128i_i64[0] = (__int64)sub_1649960((__int64)v17);
              v356 = v47;
              v358.m128i_i64[0] = (__int64)&v356;
              LOWORD(v359) = 773;
              v358.m128i_i64[1] = (__int64)".i";
              v41 = (char)v361;
              if ( (_BYTE)v361 )
              {
                if ( (_BYTE)v361 == 1 )
                {
                  a5 = _mm_loadu_si128(&v358);
                  v362 = a5;
                  v363 = v359;
                }
                else
                {
                  v40 = (__m128i *)v360.m128i_i64[0];
                  if ( BYTE1(v361) != 1 )
                  {
                    v40 = &v360;
                    v41 = 2;
                  }
                  v362.m128i_i64[1] = (__int64)v40;
                  LOBYTE(v363) = 2;
                  v362.m128i_i64[0] = (__int64)&v358;
                  BYTE1(v363) = v41;
                }
              }
              else
              {
                LOWORD(v363) = 256;
              }
              v42 = sub_1A3F820((__int64 *)&v385, v39);
              v43 = 0;
              v44 = (__int64)v42;
              v45 = &v379[8 * v39];
              v46 = *(v322 - 3);
              if ( *(_BYTE *)(v46 + 8) == 16 )
                v43 = *(__int64 ***)(v46 + 24);
              if ( v43 != *(__int64 ***)v44 )
              {
                if ( *(_BYTE *)(v44 + 16) > 0x10u )
                {
                  LOWORD(v366) = 257;
                  v44 = sub_15FDBD0(40, v44, (__int64)v43, (__int64)&v365, 0);
                  if ( v372.m128_u64[1] )
                  {
                    v318 = (__int64 *)v373;
                    sub_157E9D0(v372.m128_u64[1] + 40, v44);
                    v262 = *v318;
                    v263 = *(_QWORD *)(v44 + 24) & 7LL;
                    *(_QWORD *)(v44 + 32) = v318;
                    v262 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v44 + 24) = v262 | v263;
                    *(_QWORD *)(v262 + 8) = v44 + 24;
                    *v318 = *v318 & 7 | (v44 + 24);
                  }
                  sub_164B780(v44, v362.m128i_i64);
                  if ( v372.m128_u64[0] )
                  {
                    v351 = (const char *)v372.m128_u64[0];
                    sub_1623A60((__int64)&v351, v372.m128_i64[0], 2);
                    v264 = *(_QWORD *)(v44 + 48);
                    v265 = v44 + 48;
                    if ( v264 )
                    {
                      sub_161E7C0(v44 + 48, v264);
                      v265 = v44 + 48;
                    }
                    v266 = (unsigned __int8 *)v351;
                    *(_QWORD *)(v44 + 48) = v351;
                    if ( v266 )
                      sub_1623210((__int64)&v351, v266, v265);
                  }
                }
                else
                {
                  v44 = sub_15A46C0(40, (__int64 ***)v44, v43, 0);
                }
              }
              *(_QWORD *)v45 = v44;
              ++v39;
            }
            while ( v326 != v39 );
LABEL_358:
            v16 = v322;
LABEL_359:
            sub_1A41120(
              a1,
              (unsigned __int64)v17,
              &v379,
              a3,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v34,
              v35,
              *(double *)a9.m128i_i64,
              a10);
            if ( v379 != (unsigned __int8 *)v381 )
              _libc_free((unsigned __int64)v379);
            if ( v390 != v392 )
              _libc_free((unsigned __int64)v390);
            v261 = v372.m128_u64[0];
            if ( !v372.m128_u64[0] )
              goto LABEL_117;
LABEL_354:
            sub_161E7C0((__int64)&v372, v261);
            v21 = (_QWORD *)v16[1];
            goto LABEL_23;
          case 0x47:
            v20 = sub_1A42DC0(a1, v16 - 3, a3, a4, a5, a6, v13, v14, *(double *)a9.m128i_i64, a10);
            goto LABEL_22;
          case 0x4B:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            v48 = *(v16 - 3);
            if ( *(_BYTE *)(v48 + 8) != 16 )
              goto LABEL_11;
            v49 = *(_QWORD *)(v48 + 32);
            v50 = sub_16498A0((__int64)(v16 - 3));
            v365.m128i_i64[0] = 0;
            v367 = v50;
            v368 = 0;
            v369 = 0;
            v370 = 0;
            v371 = 0;
            v53 = v16[2];
            v366 = v16;
            v365.m128i_i64[1] = v53;
            v54 = (unsigned __int8 *)v16[3];
            v385 = v54;
            if ( v54 )
            {
              sub_1623A60((__int64)&v385, (__int64)v54, 2);
              if ( v365.m128i_i64[0] )
                sub_161E7C0((__int64)&v365, v365.m128i_i64[0]);
              v365.m128i_i64[0] = (__int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v365);
            }
            if ( (*((_BYTE *)v16 - 1) & 0x40) != 0 )
              v55 = (unsigned __int64 *)*(v16 - 4);
            else
              v55 = &v17[-3 * (*((_DWORD *)v16 - 1) & 0xFFFFFFF)];
            sub_1A41500((__int64)&v379, (_QWORD *)a1, (__int64)(v16 - 3), *v55, v51, v52);
            if ( (*((_BYTE *)v16 - 1) & 0x40) != 0 )
              v58 = (_QWORD *)*(v16 - 4);
            else
              v58 = &v17[-3 * (*((_DWORD *)v16 - 1) & 0xFFFFFFF)];
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), v58[3], v56, v57);
            v372.m128_u64[1] = 0x800000000LL;
            v63 = &v373;
            v64 = &v373;
            v372.m128_u64[0] = (unsigned __int64)&v373;
            v327 = (unsigned int)v49;
            if ( (_DWORD)v49 )
            {
              if ( (unsigned int)v49 > 8uLL )
              {
                sub_16CD150((__int64)&v372, &v373, (unsigned int)v49, 8, v59, v60);
                v64 = (unsigned __int64 **)v372.m128_u64[0];
                v63 = (unsigned __int64 **)(v372.m128_u64[0] + 8LL * v372.m128_u32[2]);
              }
              for ( n = &v64[(unsigned int)v49]; n != v63; ++v63 )
              {
                if ( v63 )
                  *v63 = 0;
              }
              v372.m128_i32[2] = v49;
            }
            v66 = 0;
            if ( !(_DWORD)v49 )
              goto LABEL_386;
            v323 = v16;
            v315 = (__int64)(v16 - 3);
            do
            {
              v358.m128i_i32[0] = v66;
              LOWORD(v359) = 265;
              v351 = sub_1649960(v315);
              LOWORD(v357) = 773;
              v352 = v74;
              v356.m128i_i64[0] = (__int64)&v351;
              v356.m128i_i64[1] = (__int64)".i";
              v68 = (char)v359;
              if ( (_BYTE)v359 )
              {
                if ( (_BYTE)v359 == 1 )
                {
                  a6 = _mm_loadu_si128(&v356);
                  v360 = a6;
                  v361 = v357;
                }
                else
                {
                  v67 = (__m128i *)v358.m128i_i64[0];
                  if ( BYTE1(v359) != 1 )
                  {
                    v67 = &v358;
                    v68 = 2;
                  }
                  v360.m128i_i64[1] = (__int64)v67;
                  LOBYTE(v361) = 2;
                  v360.m128i_i64[0] = (__int64)&v356;
                  BYTE1(v361) = v68;
                }
              }
              else
              {
                LOWORD(v361) = 256;
              }
              v69 = sub_1A3F820((__int64 *)&v385, v66);
              v70 = sub_1A3F820((__int64 *)&v379, v66);
              v71 = (_QWORD *)(v372.m128_u64[0] + 8 * v66);
              v72 = *((_WORD *)v323 - 3) & 0x7FFF;
              if ( v70[16] > 0x10u || v69[16] > 0x10u )
              {
                v311 = v70;
                LOWORD(v363) = 257;
                v284 = sub_1648A60(56, 2u);
                v285 = v311;
                v73 = v284;
                if ( v284 )
                {
                  v312 = (__int64)v284;
                  v286 = *(_QWORD ***)v285;
                  if ( *(_BYTE *)(*(_QWORD *)v285 + 8LL) == 16 )
                  {
                    v299 = (__int64)v285;
                    v303 = v286[4];
                    v287 = (__int64 *)sub_1643320(*v286);
                    v288 = (__int64)sub_16463B0(v287, (unsigned int)v303);
                    v289 = v299;
                  }
                  else
                  {
                    v304 = (__int64)v285;
                    v288 = sub_1643320(*v286);
                    v289 = v304;
                  }
                  sub_15FEC10((__int64)v73, v288, 51, v72, v289, (__int64)v69, (__int64)&v362, 0);
                }
                else
                {
                  v312 = 0;
                }
                if ( v365.m128i_i64[1] )
                {
                  v290 = v366;
                  sub_157E9D0(v365.m128i_i64[1] + 40, (__int64)v73);
                  v291 = v73[3];
                  v292 = *v290;
                  v73[4] = v290;
                  v292 &= 0xFFFFFFFFFFFFFFF8LL;
                  v73[3] = v292 | v291 & 7;
                  *(_QWORD *)(v292 + 8) = v73 + 3;
                  *v290 = *v290 & 7 | (unsigned __int64)(v73 + 3);
                }
                sub_164B780(v312, v360.m128i_i64);
                if ( v365.m128i_i64[0] )
                {
                  v350 = (unsigned __int8 *)v365.m128i_i64[0];
                  sub_1623A60((__int64)&v350, v365.m128i_i64[0], 2);
                  v293 = v73[6];
                  if ( v293 )
                    sub_161E7C0((__int64)(v73 + 6), v293);
                  v294 = v350;
                  v73[6] = v350;
                  if ( v294 )
                    sub_1623210((__int64)&v350, v294, (__int64)(v73 + 6));
                }
              }
              else
              {
                v73 = (_QWORD *)sub_15A37B0(*((_WORD *)v323 - 3) & 0x7FFF, v70, v69, 0);
              }
              *v71 = v73;
              ++v66;
            }
            while ( v327 != v66 );
            goto LABEL_385;
          case 0x4C:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            v89 = *(v16 - 3);
            if ( *(_BYTE *)(v89 + 8) != 16 )
              goto LABEL_11;
            v90 = *(_QWORD *)(v89 + 32);
            v91 = sub_16498A0((__int64)(v16 - 3));
            v365.m128i_i64[0] = 0;
            v367 = v91;
            v368 = 0;
            v369 = 0;
            v370 = 0;
            v371 = 0;
            v94 = v16[2];
            v366 = v16;
            v365.m128i_i64[1] = v94;
            v95 = (unsigned __int8 *)v16[3];
            v385 = v95;
            if ( v95 )
            {
              sub_1623A60((__int64)&v385, (__int64)v95, 2);
              if ( v365.m128i_i64[0] )
                sub_161E7C0((__int64)&v365, v365.m128i_i64[0]);
              v365.m128i_i64[0] = (__int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v365);
            }
            if ( (*((_BYTE *)v16 - 1) & 0x40) != 0 )
              v96 = (unsigned __int64 *)*(v16 - 4);
            else
              v96 = &v17[-3 * (*((_DWORD *)v16 - 1) & 0xFFFFFFF)];
            sub_1A41500((__int64)&v379, (_QWORD *)a1, (__int64)(v16 - 3), *v96, v92, v93);
            if ( (*((_BYTE *)v16 - 1) & 0x40) != 0 )
              v99 = (_QWORD *)*(v16 - 4);
            else
              v99 = &v17[-3 * (*((_DWORD *)v16 - 1) & 0xFFFFFFF)];
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), v99[3], v97, v98);
            v372.m128_u64[1] = 0x800000000LL;
            v102 = &v373;
            v103 = &v373;
            v372.m128_u64[0] = (unsigned __int64)&v373;
            v328 = (unsigned int)v90;
            if ( (_DWORD)v90 )
            {
              if ( (unsigned int)v90 > 8uLL )
              {
                sub_16CD150((__int64)&v372, &v373, (unsigned int)v90, 8, v100, v101);
                v103 = (unsigned __int64 **)v372.m128_u64[0];
                v102 = (unsigned __int64 **)(v372.m128_u64[0] + 8LL * v372.m128_u32[2]);
              }
              for ( ii = &v103[(unsigned int)v90]; ii != v102; ++v102 )
              {
                if ( v102 )
                  *v102 = 0;
              }
              v372.m128_i32[2] = v90;
            }
            if ( !(_DWORD)v90 )
              goto LABEL_386;
            v323 = v16;
            v105 = 0;
            v315 = (__int64)(v16 - 3);
            do
            {
LABEL_148:
              v358.m128i_i32[0] = v105;
              LOWORD(v359) = 265;
              v351 = sub_1649960(v315);
              v352 = v113;
              v356.m128i_i64[0] = (__int64)&v351;
              LOWORD(v357) = 773;
              v356.m128i_i64[1] = (__int64)".i";
              v107 = (char)v359;
              if ( (_BYTE)v359 )
              {
                if ( (_BYTE)v359 == 1 )
                {
                  v360 = _mm_loadu_si128(&v356);
                  v361 = v357;
                }
                else
                {
                  v106 = (__m128i *)v358.m128i_i64[0];
                  if ( BYTE1(v359) != 1 )
                  {
                    v106 = &v358;
                    v107 = 2;
                  }
                  v360.m128i_i64[1] = (__int64)v106;
                  LOBYTE(v361) = 2;
                  v360.m128i_i64[0] = (__int64)&v356;
                  BYTE1(v361) = v107;
                }
              }
              else
              {
                LOWORD(v361) = 256;
              }
              v108 = sub_1A3F820((__int64 *)&v385, v105);
              v109 = sub_1A3F820((__int64 *)&v379, v105);
              v110 = (_QWORD *)(v372.m128_u64[0] + 8 * v105);
              v111 = *((_WORD *)v323 - 3) & 0x7FFF;
              if ( v109[16] <= 0x10u && v108[16] <= 0x10u )
              {
                v112 = (_QWORD *)sub_15A37B0(*((_WORD *)v323 - 3) & 0x7FFF, v109, v108, 0);
LABEL_147:
                *v110 = v112;
                if ( v328 == ++v105 )
                  break;
                goto LABEL_148;
              }
              v309 = v109;
              LOWORD(v363) = 257;
              v267 = sub_1648A60(56, 2u);
              v268 = v309;
              v112 = v267;
              if ( v267 )
              {
                v310 = (__int64)v267;
                v269 = *(_QWORD ***)v268;
                if ( *(_BYTE *)(*(_QWORD *)v268 + 8LL) == 16 )
                {
                  v298 = (__int64)v268;
                  v301 = v269[4];
                  v270 = (__int64 *)sub_1643320(*v269);
                  v271 = (__int64)sub_16463B0(v270, (unsigned int)v301);
                  v272 = v298;
                }
                else
                {
                  v302 = (__int64)v268;
                  v271 = sub_1643320(*v269);
                  v272 = v302;
                }
                sub_15FEC10((__int64)v112, v271, 52, v111, v272, (__int64)v108, (__int64)&v362, 0);
              }
              else
              {
                v310 = 0;
              }
              v273 = v369;
              if ( v368 )
                sub_1625C10((__int64)v112, 3, v368);
              sub_15F2440((__int64)v112, v273);
              if ( v365.m128i_i64[1] )
              {
                v274 = v366;
                sub_157E9D0(v365.m128i_i64[1] + 40, (__int64)v112);
                v275 = v112[3];
                v276 = *v274;
                v112[4] = v274;
                v276 &= 0xFFFFFFFFFFFFFFF8LL;
                v112[3] = v276 | v275 & 7;
                *(_QWORD *)(v276 + 8) = v112 + 3;
                *v274 = *v274 & 7 | (unsigned __int64)(v112 + 3);
              }
              sub_164B780(v310, v360.m128i_i64);
              if ( !v365.m128i_i64[0] )
                goto LABEL_147;
              v350 = (unsigned __int8 *)v365.m128i_i64[0];
              sub_1623A60((__int64)&v350, v365.m128i_i64[0], 2);
              v277 = v112[6];
              if ( v277 )
                sub_161E7C0((__int64)(v112 + 6), v277);
              v278 = v350;
              v112[6] = v350;
              if ( !v278 )
                goto LABEL_147;
              ++v105;
              sub_1623210((__int64)&v350, v278, (__int64)(v112 + 6));
              *v110 = v112;
            }
            while ( v328 != v105 );
LABEL_385:
            v16 = v323;
            v17 = (_QWORD *)v315;
LABEL_386:
            sub_1A41120(
              a1,
              (unsigned __int64)v17,
              &v372,
              a3,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v61,
              v62,
              *(double *)a9.m128i_i64,
              a10);
            v215 = v372.m128_u64[0];
            if ( (unsigned __int64 **)v372.m128_u64[0] != &v373 )
LABEL_387:
              _libc_free(v215);
LABEL_388:
            if ( v390 != v392 )
              _libc_free((unsigned __int64)v390);
            if ( v382 != v383 )
              _libc_free((unsigned __int64)v382);
            if ( !v365.m128i_i64[0] )
              goto LABEL_117;
            sub_161E7C0((__int64)&v365, v365.m128i_i64[0]);
            v21 = (_QWORD *)v16[1];
            goto LABEL_23;
          case 0x4D:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3)) )
              goto LABEL_11;
            v140 = *(v16 - 3);
            v325 = v140;
            if ( *(_BYTE *)(v140 + 8) != 16 )
              goto LABEL_11;
            v141 = *(_QWORD *)(v140 + 32);
            v313 = v141;
            v142 = sub_16498A0((__int64)(v16 - 3));
            v372.m128_u64[0] = 0;
            v374 = v142;
            v375 = 0;
            v376 = 0;
            v377 = 0;
            v378 = 0;
            v147 = v16[2];
            v373 = v16;
            v372.m128_u64[1] = v147;
            v148 = (unsigned __int8 *)v16[3];
            v385 = v148;
            if ( v148 )
            {
              sub_1623A60((__int64)&v385, (__int64)v148, 2);
              if ( v372.m128_u64[0] )
                sub_161E7C0((__int64)&v372, v372.m128_i64[0]);
              v372.m128_u64[0] = (unsigned __int64)v385;
              if ( v385 )
                sub_1623210((__int64)&v385, v385, (__int64)&v372);
            }
            v379 = (unsigned __int8 *)v381;
            v380 = 0x800000000LL;
            v317 = (unsigned int)v141;
            if ( (_DWORD)v141 )
            {
              v149 = (unsigned __int8 *)v381;
              v150 = (unsigned __int8 *)v381;
              if ( (unsigned int)v141 > 8uLL )
              {
                sub_16CD150((__int64)&v379, v381, (unsigned int)v141, 8, v143, v144);
                v149 = v379;
                v150 = &v379[8 * (unsigned int)v380];
              }
              for ( jj = &v149[8 * (unsigned int)v141]; jj != v150; v150 += 8 )
              {
                if ( v150 )
                  *(_QWORD *)v150 = 0;
              }
              LODWORD(v380) = v141;
            }
            v331 = *((_DWORD *)v16 - 1) & 0xFFFFFFF;
            if ( (_DWORD)v141 )
            {
              v300 = v16;
              v152 = 0;
              do
              {
                v362.m128i_i32[0] = v152;
                LOWORD(v363) = 265;
                v165.m128i_i64[0] = (__int64)sub_1649960((__int64)v17);
                LOWORD(v361) = 773;
                v358 = v165;
                v360.m128i_i64[0] = (__int64)&v358;
                v360.m128i_i64[1] = (__int64)".i";
                v154 = (char)v363;
                if ( (_BYTE)v363 )
                {
                  if ( (_BYTE)v363 == 1 )
                  {
                    v365 = _mm_loadu_si128(&v360);
                    v366 = v361;
                  }
                  else
                  {
                    v153 = (__m128i *)v362.m128i_i64[0];
                    if ( BYTE1(v363) != 1 )
                    {
                      v153 = &v362;
                      v154 = 2;
                    }
                    v365.m128i_i64[1] = (__int64)v153;
                    LOBYTE(v366) = 2;
                    v365.m128i_i64[0] = (__int64)&v360;
                    BYTE1(v366) = v154;
                  }
                }
                else
                {
                  LOWORD(v366) = 256;
                }
                v155 = *(_QWORD *)(v325 + 24);
                LOWORD(v387) = 257;
                v156 = &v379[8 * v152];
                v157 = sub_1648B60(64);
                v158 = v157;
                if ( v157 )
                {
                  v159 = v157;
                  sub_15F1EA0(v157, v155, 53, 0, 0, 0);
                  *(_DWORD *)(v158 + 56) = v331;
                  sub_164B780(v158, (__int64 *)&v385);
                  sub_1648880(v158, *(_DWORD *)(v158 + 56), 1);
                }
                else
                {
                  v159 = 0;
                }
                if ( v372.m128_u64[1] )
                {
                  v336 = (__int64 *)v373;
                  sub_157E9D0(v372.m128_u64[1] + 40, v158);
                  v160 = *v336;
                  v161 = *(_QWORD *)(v158 + 24) & 7LL;
                  *(_QWORD *)(v158 + 32) = v336;
                  v160 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v158 + 24) = v160 | v161;
                  *(_QWORD *)(v160 + 8) = v158 + 24;
                  *v336 = *v336 & 7 | (v158 + 24);
                }
                sub_164B780(v159, v365.m128i_i64);
                if ( v372.m128_u64[0] )
                {
                  v356.m128i_i64[0] = v372.m128_u64[0];
                  sub_1623A60((__int64)&v356, v372.m128_i64[0], 2);
                  v162 = *(_QWORD *)(v158 + 48);
                  v163 = v158 + 48;
                  if ( v162 )
                  {
                    sub_161E7C0(v158 + 48, v162);
                    v163 = v158 + 48;
                  }
                  v164 = (unsigned __int8 *)v356.m128i_i64[0];
                  *(_QWORD *)(v158 + 48) = v356.m128i_i64[0];
                  if ( v164 )
                    sub_1623210((__int64)&v356, v164, v163);
                }
                *(_QWORD *)v156 = v158;
                ++v152;
              }
              while ( v152 != v317 );
              v16 = v300;
            }
            if ( v331 )
            {
              v338 = v16;
              v345 = 0;
              v308 = 8LL * v331;
              do
              {
                if ( (*((_BYTE *)v338 - 1) & 0x40) != 0 )
                  v244 = (_QWORD *)*(v338 - 4);
                else
                  v244 = &v17[-3 * (*((_DWORD *)v338 - 1) & 0xFFFFFFF)];
                sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)v17, v244[3 * v345 / 8], v143, v144);
                if ( (*((_BYTE *)v338 - 1) & 0x40) != 0 )
                  v245 = (_QWORD *)*(v338 - 4);
                else
                  v245 = &v17[-3 * (*((_DWORD *)v338 - 1) & 0xFFFFFFF)];
                v246 = 0;
                v143 = v313;
                v247 = v245[3 * *((unsigned int *)v338 + 8) + 1 + v345 / 8];
                if ( v313 )
                {
                  do
                  {
                    v249 = *(_QWORD *)&v379[8 * v246];
                    v251 = sub_1A3F820((__int64 *)&v385, v246);
                    v252 = *(_DWORD *)(v249 + 20) & 0xFFFFFFF;
                    if ( v252 == *(_DWORD *)(v249 + 56) )
                    {
                      v333 = v251;
                      sub_15F55D0(v249, (unsigned int)v246, v250, (__int64)v251, v143, v144);
                      v251 = v333;
                      v252 = *(_DWORD *)(v249 + 20) & 0xFFFFFFF;
                    }
                    v253 = (v252 + 1) & 0xFFFFFFF;
                    v254 = v253 | *(_DWORD *)(v249 + 20) & 0xF0000000;
                    *(_DWORD *)(v249 + 20) = v254;
                    if ( (v254 & 0x40000000) != 0 )
                      v255 = *(_QWORD *)(v249 - 8);
                    else
                      v255 = v249 - 24 * v253;
                    v256 = (_QWORD *)(v255 + 24LL * (unsigned int)(v253 - 1));
                    if ( *v256 )
                    {
                      v257 = v256[1];
                      v258 = v256[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v258 = v257;
                      if ( v257 )
                        *(_QWORD *)(v257 + 16) = *(_QWORD *)(v257 + 16) & 3LL | v258;
                    }
                    *v256 = v251;
                    if ( v251 )
                    {
                      v259 = *((_QWORD *)v251 + 1);
                      v256[1] = v259;
                      if ( v259 )
                      {
                        v143 = (__int64)(v256 + 1);
                        *(_QWORD *)(v259 + 16) = (unsigned __int64)(v256 + 1) | *(_QWORD *)(v259 + 16) & 3LL;
                      }
                      v256[2] = (unsigned __int64)(v251 + 8) | v256[2] & 3LL;
                      *((_QWORD *)v251 + 1) = v256;
                    }
                    v260 = *(_DWORD *)(v249 + 20) & 0xFFFFFFF;
                    if ( (*(_BYTE *)(v249 + 23) & 0x40) != 0 )
                      v248 = *(_QWORD *)(v249 - 8);
                    else
                      v248 = v249 - 24 * v260;
                    ++v246;
                    *(_QWORD *)(v248 + 8LL * (unsigned int)(v260 - 1) + 24LL * *(unsigned int *)(v249 + 56) + 8) = v247;
                  }
                  while ( v317 != v246 );
                }
                if ( v390 != v392 )
                  _libc_free((unsigned __int64)v390);
                v345 += 8LL;
              }
              while ( v308 != v345 );
              v16 = v338;
            }
            sub_1A41120(
              a1,
              (unsigned __int64)v17,
              &v379,
              a3,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              *(double *)a6.m128i_i64,
              v145,
              v146,
              *(double *)a9.m128i_i64,
              a10);
            if ( v379 != (unsigned __int8 *)v381 )
              _libc_free((unsigned __int64)v379);
            v261 = v372.m128_u64[0];
            if ( v372.m128_u64[0] )
              goto LABEL_354;
            goto LABEL_117;
          case 0x4E:
            v20 = sub_1A44940(
                    a1,
                    (__int64)(v16 - 3),
                    a3,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v13,
                    v14,
                    *(double *)a9.m128i_i64,
                    a10);
            goto LABEL_22;
          case 0x4F:
            v20 = sub_1A43DA0(
                    a1,
                    v16 - 3,
                    a3,
                    a4,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v13,
                    v14,
                    *(double *)a9.m128i_i64,
                    a10);
LABEL_22:
            v21 = (_QWORD *)v16[1];
            if ( v20 )
              goto LABEL_23;
LABEL_26:
            v16 = v21;
            continue;
          case 0x55:
            if ( *(_DWORD *)(a1 + 496) && !(unsigned __int8)sub_1A3F5B0(a1, (__int64)(v16 - 3))
              || (v75 = *(v16 - 3), (v341 = v75) == 0) )
            {
LABEL_11:
              v16 = (_QWORD *)v16[1];
              continue;
            }
            v335 = *(_QWORD *)(v75 + 32);
            sub_1A41500((__int64)&v379, (_QWORD *)a1, (__int64)(v16 - 3), *(v16 - 12), v335, v12);
            sub_1A41500((__int64)&v385, (_QWORD *)a1, (__int64)(v16 - 3), *(v16 - 9), v76, v77);
            v81 = v335;
            v372.m128_u64[1] = 0x800000000LL;
            v82 = &v373;
            v83 = &v373;
            v372.m128_u64[0] = (unsigned __int64)&v373;
            if ( (_DWORD)v335 )
            {
              if ( (unsigned int)v335 > 8uLL )
              {
                sub_16CD150((__int64)&v372, &v373, (unsigned int)v335, 8, v335, v78);
                v83 = (unsigned __int64 **)v372.m128_u64[0];
                v81 = v335;
                v82 = (unsigned __int64 **)(v372.m128_u64[0] + 8LL * v372.m128_u32[2]);
              }
              for ( kk = &v83[(unsigned int)v335]; kk != v82; ++v82 )
              {
                if ( v82 )
                  *v82 = 0;
              }
              v372.m128_i32[2] = v81;
            }
            if ( !v81 )
              goto LABEL_111;
            v85 = 0;
            break;
        }
        do
        {
          while ( 1 )
          {
            v86 = sub_15FA9D0(*(v16 - 6), v85);
            v87 = 8 * v85;
            if ( v86 < 0 )
            {
              v243 = (__int64 *)(v372.m128_u64[0] + v87);
              *v243 = sub_1599EF0(*(__int64 ***)(v341 + 24));
              goto LABEL_106;
            }
            v88 = (_QWORD *)(v372.m128_u64[0] + v87);
            if ( v384 <= v86 )
              break;
            *v88 = sub_1A3F820((__int64 *)&v379, v86);
LABEL_106:
            if ( (unsigned int)v335 == ++v85 )
              goto LABEL_110;
          }
          ++v85;
          *v88 = sub_1A3F820((__int64 *)&v385, v86 - v384);
        }
        while ( (unsigned int)v335 != v85 );
LABEL_110:
        v17 = v16 - 3;
LABEL_111:
        sub_1A41120(
          a1,
          (unsigned __int64)v17,
          &v372,
          a3,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          v79,
          v80,
          *(double *)a9.m128i_i64,
          a10);
        if ( (unsigned __int64 **)v372.m128_u64[0] != &v373 )
          _libc_free(v372.m128_u64[0]);
        if ( v390 != v392 )
          _libc_free((unsigned __int64)v390);
        if ( v382 != v383 )
          _libc_free((unsigned __int64)v382);
LABEL_117:
        v21 = (_QWORD *)v16[1];
LABEL_23:
        v22 = *(v16 - 3);
        v16 = v21;
        if ( !*(_BYTE *)(v22 + 8) )
          sub_15F20C0(v17);
      }
      while ( v346 != v16 );
LABEL_13:
      v339 -= 8;
      if ( v320 == v339 )
      {
        v10 = a1;
        break;
      }
    }
  }
  v18 = *(unsigned int *)(v10 + 216);
  if ( !(_DWORD)v18 )
  {
    if ( *(_QWORD *)(v10 + 200) )
      goto LABEL_17;
    goto LABEL_18;
  }
  v321 = v10;
  v349 = *(__int64 **)(v10 + 208);
  v334 = &v349[2 * v18];
  do
  {
    v216 = *v349;
    if ( !*(_QWORD *)(*v349 + 8) )
      goto LABEL_288;
    v217 = *(__int64 ***)v216;
    v218 = (_QWORD *)v349[1];
    v219 = sub_1599EF0(*(__int64 ***)v216);
    v220 = *(_QWORD *)(v216 + 40);
    v221 = v217[4];
    v222 = v219;
    v223 = (_QWORD *)sub_16498A0(v216);
    v391 = 0;
    v385 = 0;
    v388 = v223;
    v389 = 0;
    LODWORD(v390) = 0;
    v392[0] = 0;
    v386 = *(_QWORD *)(v216 + 40);
    v387 = (unsigned __int64 *)(v216 + 24);
    v224 = *(unsigned __int8 **)(v216 + 48);
    v379 = v224;
    if ( v224 )
    {
      sub_1623A60((__int64)&v379, (__int64)v224, 2);
      if ( v385 )
        sub_161E7C0((__int64)&v385, (__int64)v385);
      v385 = v379;
      if ( v379 )
        sub_1623210((__int64)&v379, v379, (__int64)&v385);
    }
    if ( *(_BYTE *)(v216 + 16) == 77 )
    {
      v295 = sub_157EE30(v220);
      if ( !v295 )
      {
        v386 = v220;
        v387 = 0;
        BUG();
      }
      v386 = v220;
      v387 = (unsigned __int64 *)v295;
      if ( v295 != v220 + 40 )
      {
        v296 = *(unsigned __int8 **)(v295 + 24);
        v379 = v296;
        if ( v296 )
        {
          sub_1623A60((__int64)&v379, (__int64)v296, 2);
          v297 = v385;
          if ( v385 )
LABEL_419:
            sub_161E7C0((__int64)&v385, (__int64)v297);
          v385 = v379;
          if ( v379 )
            sub_1623210((__int64)&v379, v379, (__int64)&v385);
          goto LABEL_296;
        }
        v297 = v385;
        if ( v385 )
          goto LABEL_419;
      }
    }
LABEL_296:
    if ( (_DWORD)v221 )
    {
      v225 = (unsigned int)v221;
      v226 = 0;
      v340 = v225;
      do
      {
        while ( 1 )
        {
          v365.m128i_i32[0] = v226;
          LOWORD(v366) = 265;
          v232.m128i_i64[0] = (__int64)sub_1649960(v216);
          v360 = v232;
          v362.m128i_i64[0] = (__int64)&v360;
          v362.m128i_i64[1] = (__int64)".upto";
          v228 = (char)v366;
          LOWORD(v363) = 773;
          if ( (_BYTE)v366 )
          {
            if ( (_BYTE)v366 == 1 )
            {
              a3 = (__m128)_mm_loadu_si128(&v362);
              v372 = a3;
              v373 = v363;
            }
            else
            {
              v227 = (__m128i *)v365.m128i_i64[0];
              if ( BYTE1(v366) != 1 )
              {
                v227 = &v365;
                v228 = 2;
              }
              v372.m128_u64[1] = (unsigned __int64)v227;
              LOBYTE(v373) = 2;
              v372.m128_u64[0] = (unsigned __int64)&v362;
              BYTE1(v373) = v228;
            }
          }
          else
          {
            LOWORD(v373) = 256;
          }
          v229 = sub_1643350(v388);
          v230 = sub_159C470(v229, v226, 0);
          if ( *(_BYTE *)(v222 + 16) > 0x10u
            || *(_BYTE *)(*(_QWORD *)(*v218 + 8 * v226) + 16LL) > 0x10u
            || *(_BYTE *)(v230 + 16) > 0x10u )
          {
            break;
          }
          v231 = sub_15A3890((__int64 *)v222, *(_QWORD *)(*v218 + 8 * v226++), v230, 0);
          v222 = v231;
          if ( v226 == v340 )
            goto LABEL_318;
        }
        v344 = v230;
        v347 = *(_QWORD *)(*v218 + 8 * v226);
        v381[0] = 257;
        v233 = sub_1648A60(56, 3u);
        v234 = v233;
        if ( v233 )
          sub_15FA480((__int64)v233, (__int64 *)v222, v347, v344, (__int64)&v379, 0);
        if ( v386 )
        {
          v235 = v387;
          sub_157E9D0(v386 + 40, (__int64)v234);
          v236 = v234[3];
          v237 = *v235;
          v234[4] = v235;
          v237 &= 0xFFFFFFFFFFFFFFF8LL;
          v234[3] = v237 | v236 & 7;
          *(_QWORD *)(v237 + 8) = v234 + 3;
          *v235 = *v235 & 7 | (unsigned __int64)(v234 + 3);
        }
        sub_164B780((__int64)v234, (__int64 *)&v372);
        if ( v385 )
        {
          v358.m128i_i64[0] = (__int64)v385;
          sub_1623A60((__int64)&v358, (__int64)v385, 2);
          v238 = v234[6];
          v239 = (__int64)(v234 + 6);
          if ( v238 )
          {
            sub_161E7C0((__int64)(v234 + 6), v238);
            v239 = (__int64)(v234 + 6);
          }
          v240 = (unsigned __int8 *)v358.m128i_i64[0];
          v234[6] = v358.m128i_i64[0];
          if ( v240 )
            sub_1623210((__int64)&v358, v240, v239);
        }
        v222 = (__int64)v234;
        ++v226;
      }
      while ( v226 != v340 );
    }
LABEL_318:
    sub_164B7C0(v222, v216);
    sub_164D160(
      v216,
      v222,
      a3,
      *(double *)a4.m128i_i64,
      *(double *)a5.m128i_i64,
      *(double *)a6.m128i_i64,
      v241,
      v242,
      *(double *)a9.m128i_i64,
      a10);
    if ( v385 )
      sub_161E7C0((__int64)&v385, (__int64)v385);
LABEL_288:
    sub_15F20C0((_QWORD *)v216);
    v349 += 2;
  }
  while ( v334 != v349 );
  v10 = v321;
LABEL_17:
  *(_DWORD *)(v10 + 216) = 0;
  sub_1A3F1B0(*(_QWORD **)(v10 + 176));
  *(_QWORD *)(v10 + 176) = 0;
  *(_QWORD *)(v10 + 184) = v10 + 168;
  *(_QWORD *)(v10 + 192) = v10 + 168;
  *(_QWORD *)(v10 + 200) = 0;
  v305 = 1;
LABEL_18:
  if ( v353 )
    j_j___libc_free_0(v353, v355 - v353);
  return v305;
}
