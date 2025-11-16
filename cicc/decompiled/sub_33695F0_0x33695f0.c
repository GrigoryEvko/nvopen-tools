// Function: sub_33695F0
// Address: 0x33695f0
//
void __fastcall sub_33695F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __m128i *a5,
        unsigned int a6,
        unsigned __int16 a7,
        _BYTE *a8,
        __int64 a9,
        int a10)
{
  __int64 v10; // r13
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  unsigned __int16 v17; // r12
  __int64 v18; // rcx
  bool v19; // al
  __m128i si128; // xmm1
  unsigned __int64 v21; // r11
  __int64 v22; // r12
  unsigned __int16 v23; // ax
  unsigned __int16 *v24; // rsi
  __int64 (__fastcall *v25)(__int64, __int64, __int64, unsigned int, unsigned __int64, __int64 *, unsigned int *, unsigned __int16 *); // rax
  unsigned int v26; // eax
  unsigned __int16 v27; // bx
  unsigned int v28; // r13d
  bool v29; // cl
  bool v30; // r8
  __int64 v31; // rax
  char v32; // dl
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  char v36; // al
  int v37; // r9d
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rdx
  char v41; // al
  unsigned int v42; // eax
  int v43; // r9d
  __int16 v44; // ax
  __int64 v45; // rdx
  __int64 v46; // r8
  unsigned int v47; // edx
  __int64 v48; // rax
  __int16 v49; // ax
  __int64 v50; // rdx
  __int64 v51; // r8
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rcx
  unsigned __int64 *v55; // rdx
  __int64 v56; // rdx
  char v57; // al
  __int64 v58; // rdx
  __int64 v59; // rdx
  char v60; // al
  __int64 v61; // rbx
  unsigned __int16 v62; // ax
  __int64 v63; // r9
  __int64 v64; // rdx
  __int64 v65; // rdx
  char v66; // al
  unsigned int v67; // eax
  int v68; // r9d
  int v69; // eax
  int v70; // edx
  int v71; // r8d
  int v72; // esi
  __int64 v73; // rax
  __int64 v74; // rdx
  unsigned int v75; // r10d
  unsigned int v76; // r9d
  __int64 v77; // rax
  __int64 v78; // r13
  __int64 v79; // r14
  _QWORD *v80; // rdi
  __int64 v81; // rax
  __int32 v82; // r11d
  __int64 v83; // rdx
  __int64 v84; // r8
  __m128i *v85; // rbx
  __m128i *v86; // r12
  __int128 v87; // rax
  int v88; // r9d
  __int64 v89; // rax
  __int64 v90; // rdx
  __int128 v91; // rax
  int v92; // r9d
  __int64 v93; // rax
  bool v94; // zf
  __int32 v95; // r9d
  __int64 v96; // rdx
  __int16 v97; // r11
  int v98; // eax
  __int64 v99; // rax
  __int64 v100; // rdx
  int v101; // r9d
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rax
  const __m128i *v105; // rdx
  unsigned __int64 v106; // rax
  __int64 v107; // rcx
  __m128i v108; // xmm0
  bool v109; // al
  unsigned __int64 v110; // r8
  char v111; // cl
  bool v112; // al
  __int64 v113; // rdx
  __int64 *v114; // rax
  __int64 v115; // rsi
  unsigned int v116; // r12d
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 v119; // r9
  unsigned __int16 v120; // bx
  unsigned __int16 v121; // r13
  __int64 v122; // r12
  const char *v123; // rax
  const char *v124; // rdx
  unsigned int v125; // ebx
  const char *i; // rdx
  unsigned int v127; // r12d
  int v128; // esi
  __int128 v129; // rax
  int v130; // r9d
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rdi
  __int64 v134; // rdx
  char *v135; // rax
  __int128 v136; // rax
  int v137; // r9d
  char *v138; // r9
  __int64 v139; // rdx
  const char *v140; // rdi
  int v141; // r13d
  int v142; // r12d
  unsigned int k; // ebx
  int v144; // ecx
  __int64 v145; // rax
  int v146; // ecx
  int v147; // edx
  int v148; // esi
  unsigned int v149; // edx
  __int64 v150; // rax
  unsigned __int16 v151; // ax
  __int64 v152; // rdx
  __int64 v153; // rax
  unsigned int v154; // ecx
  unsigned int v155; // eax
  unsigned int v156; // r12d
  __int128 v157; // rax
  int v158; // r9d
  int v159; // eax
  const __m128i *v160; // rbx
  int v161; // edx
  int v162; // r9d
  __int16 v163; // ax
  __int64 v164; // rdx
  __int64 v165; // r8
  int v166; // edx
  __int64 v167; // rdi
  int v168; // r9d
  int v169; // edx
  __int32 v170; // ecx
  __int64 j; // r12
  __int64 v172; // rdx
  __int64 v173; // rcx
  unsigned __int16 v174; // ax
  __int64 v175; // rdx
  __int64 v176; // r12
  __int64 v177; // rdx
  __int64 v178; // rax
  __int64 v179; // rdx
  _WORD *v180; // rdx
  _WORD *v181; // r12
  unsigned __int16 v182; // bx
  unsigned __int16 v183; // ax
  __int64 v184; // rdx
  __int64 v185; // rbx
  __int64 v186; // rax
  unsigned int v187; // edx
  __int32 v188; // esi
  unsigned __int64 v189; // r12
  __int64 v190; // rdx
  char v191; // r13
  __int64 v192; // rdx
  __int64 v193; // rax
  unsigned __int64 v194; // rdx
  unsigned __int16 *v195; // rdx
  __int64 v196; // r8
  __int64 v197; // rsi
  __int64 v198; // r13
  unsigned __int16 v199; // ax
  __int64 *v200; // r12
  unsigned int v201; // ebx
  __int16 v202; // ax
  __int64 v203; // r9
  unsigned __int64 v204; // r8
  unsigned int v205; // edx
  unsigned __int64 *v206; // rdx
  unsigned int v207; // edx
  unsigned __int16 v208; // ax
  __m128i *v209; // rax
  __m128i v210; // xmm0
  __int64 v211; // rcx
  __int64 v212; // rdx
  __int64 v213; // rax
  __int64 v214; // rdx
  char v215; // cl
  __int64 v216; // rbx
  __int64 v217; // rsi
  __int64 v218; // rdx
  __int64 v219; // rcx
  __int64 v220; // r8
  __int64 v221; // rdx
  unsigned __int64 *v222; // rcx
  __int16 v223; // ax
  unsigned __int64 *v224; // rdx
  unsigned __int64 *v225; // rdi
  __int64 v226; // rdx
  __int64 v227; // rbx
  int v228; // edx
  unsigned int v229; // esi
  __int64 v230; // rax
  __int64 v231; // rdx
  __int64 v232; // rsi
  __int64 v233; // rax
  __int64 v234; // rdx
  __int64 v235; // rsi
  unsigned __int64 v236; // rdx
  int v237; // edx
  unsigned int v238; // eax
  __int64 v239; // rdx
  int v240; // r9d
  unsigned int v241; // edx
  __int64 v242; // r8
  __int64 v243; // rax
  int v244; // ecx
  int v245; // edx
  int v246; // esi
  int v247; // edx
  unsigned __int64 v248; // rdx
  __int64 v249; // rdx
  __int64 v250; // rax
  __int128 v251; // rax
  int v252; // r9d
  int v253; // edx
  bool v254; // al
  __int64 v255; // rdx
  __int64 v256; // rcx
  unsigned int v257; // eax
  __int64 v258; // rdx
  unsigned int v259; // eax
  __int64 v260; // rdx
  unsigned int v261; // eax
  unsigned __int64 *v262; // rdx
  unsigned __int64 *v263; // rdx
  unsigned __int64 *v264; // rcx
  __int16 v265; // ax
  unsigned __int64 *v266; // rdx
  __int64 v267; // rdx
  __int64 v268; // rax
  __int64 v269; // rdx
  __int64 v270; // rax
  __int64 v271; // rdx
  __int64 v272; // rax
  __int64 v273; // rdx
  __int128 v274; // [rsp-20h] [rbp-310h]
  __int128 v275; // [rsp-20h] [rbp-310h]
  __int16 v276; // [rsp+Ah] [rbp-2E6h]
  __int64 v277; // [rsp+10h] [rbp-2E0h]
  unsigned __int8 v278; // [rsp+10h] [rbp-2E0h]
  __int16 v279; // [rsp+12h] [rbp-2DEh]
  bool v280; // [rsp+18h] [rbp-2D8h]
  __int64 *v281; // [rsp+18h] [rbp-2D8h]
  char v282; // [rsp+18h] [rbp-2D8h]
  __int16 v283; // [rsp+1Ah] [rbp-2D6h]
  int v284; // [rsp+20h] [rbp-2D0h]
  unsigned __int16 v285; // [rsp+20h] [rbp-2D0h]
  __int64 v286; // [rsp+28h] [rbp-2C8h]
  unsigned int v287; // [rsp+28h] [rbp-2C8h]
  unsigned int v288; // [rsp+28h] [rbp-2C8h]
  unsigned int v290; // [rsp+34h] [rbp-2BCh]
  unsigned int v291; // [rsp+34h] [rbp-2BCh]
  __int64 v292; // [rsp+38h] [rbp-2B8h]
  int v293; // [rsp+38h] [rbp-2B8h]
  __int64 v295; // [rsp+48h] [rbp-2A8h]
  unsigned int v296; // [rsp+48h] [rbp-2A8h]
  __int64 v297; // [rsp+48h] [rbp-2A8h]
  char v298; // [rsp+48h] [rbp-2A8h]
  unsigned __int128 v299; // [rsp+50h] [rbp-2A0h] BYREF
  __m128i v300; // [rsp+60h] [rbp-290h]
  unsigned __int64 *v301; // [rsp+70h] [rbp-280h]
  __int64 v302; // [rsp+78h] [rbp-278h]
  __m128i v303; // [rsp+80h] [rbp-270h]
  __int64 v304; // [rsp+90h] [rbp-260h]
  __int64 v305; // [rsp+98h] [rbp-258h]
  __int64 v306; // [rsp+A0h] [rbp-250h]
  __int64 v307; // [rsp+A8h] [rbp-248h]
  __int64 v308; // [rsp+B0h] [rbp-240h]
  __int64 v309; // [rsp+B8h] [rbp-238h]
  __int64 v310; // [rsp+C0h] [rbp-230h]
  __int64 v311; // [rsp+C8h] [rbp-228h]
  __int64 v312; // [rsp+D0h] [rbp-220h]
  __int64 v313; // [rsp+D8h] [rbp-218h]
  __m128i v314; // [rsp+E0h] [rbp-210h]
  __int64 v315; // [rsp+F0h] [rbp-200h]
  __int64 v316; // [rsp+F8h] [rbp-1F8h]
  __int64 v317; // [rsp+100h] [rbp-1F0h]
  __int64 v318; // [rsp+108h] [rbp-1E8h]
  __int64 v319; // [rsp+110h] [rbp-1E0h]
  __int64 v320; // [rsp+118h] [rbp-1D8h]
  __int64 v321; // [rsp+120h] [rbp-1D0h]
  __int64 v322; // [rsp+128h] [rbp-1C8h]
  unsigned __int16 v323; // [rsp+132h] [rbp-1BEh] BYREF
  unsigned int v324; // [rsp+134h] [rbp-1BCh] BYREF
  __int64 v325; // [rsp+138h] [rbp-1B8h]
  unsigned __int64 v326; // [rsp+140h] [rbp-1B0h]
  unsigned __int64 v327; // [rsp+148h] [rbp-1A8h]
  unsigned int v328; // [rsp+150h] [rbp-1A0h] BYREF
  __int64 v329; // [rsp+158h] [rbp-198h]
  unsigned __int64 v330; // [rsp+160h] [rbp-190h] BYREF
  unsigned __int64 v331; // [rsp+168h] [rbp-188h]
  __int64 v332; // [rsp+170h] [rbp-180h] BYREF
  __int64 v333; // [rsp+178h] [rbp-178h]
  __int64 v334; // [rsp+180h] [rbp-170h] BYREF
  __int64 v335; // [rsp+188h] [rbp-168h]
  _QWORD v336[6]; // [rsp+190h] [rbp-160h] BYREF
  __int64 v337; // [rsp+1C0h] [rbp-130h]
  __int64 v338; // [rsp+1C8h] [rbp-128h]
  __int64 v339; // [rsp+1D0h] [rbp-120h] BYREF
  __int64 v340; // [rsp+1D8h] [rbp-118h]
  __int64 v341; // [rsp+1E0h] [rbp-110h] BYREF
  __int64 v342; // [rsp+1E8h] [rbp-108h]
  __int64 v343; // [rsp+1F0h] [rbp-100h]
  __int64 v344; // [rsp+1F8h] [rbp-F8h]
  __int64 v345; // [rsp+200h] [rbp-F0h] BYREF
  unsigned __int64 *v346; // [rsp+208h] [rbp-E8h]
  __int64 v347; // [rsp+210h] [rbp-E0h] BYREF
  unsigned __int64 *v348; // [rsp+218h] [rbp-D8h]
  unsigned __int64 v349; // [rsp+220h] [rbp-D0h]
  __int64 v350; // [rsp+228h] [rbp-C8h]
  const char *v351; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v352; // [rsp+238h] [rbp-B8h]
  _BYTE v353[176]; // [rsp+240h] [rbp-B0h] BYREF

  v12 = a2;
  v13 = *(_QWORD *)(a1 + 16);
  v299 = __PAIR128__(a4, a3);
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 2272LL);
  v302 = a3;
  LODWORD(v301) = a4;
  if ( v14 != sub_3364F60
    && ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD, _QWORD, __m128i *, _QWORD, _QWORD, __int64))v14)(
         v13,
         a1,
         a2,
         v299,
         *((_QWORD *)&v299 + 1),
         a5,
         a6,
         a7,
         a9) )
  {
    return;
  }
  v295 = (unsigned int)v301;
  v15 = 16LL * (unsigned int)v301;
  v16 = v15 + *(_QWORD *)(v302 + 48);
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOWORD(v328) = v17;
  v329 = v18;
  if ( !v17 )
  {
    v292 = v18;
    v300.m128i_i64[0] = v16;
    v19 = sub_30070B0((__int64)&v328);
    v16 = v300.m128i_i64[0];
    v18 = v292;
    if ( !v19 )
      goto LABEL_4;
LABEL_11:
    si128 = _mm_load_si128((const __m128i *)&v299);
    v21 = *(_QWORD *)(v16 + 8);
    v325 = a9;
    v22 = *(_QWORD *)(a1 + 16);
    v300 = si128;
    v23 = *(_WORD *)v16;
    v331 = v21;
    LOWORD(v330) = v23;
    if ( a6 != 1 )
    {
      v24 = *(unsigned __int16 **)(a1 + 64);
      LOWORD(v332) = 0;
      v333 = 0;
      v323 = 0;
      if ( BYTE4(a9) )
      {
        v25 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int, unsigned __int64, __int64 *, unsigned int *, unsigned __int16 *))(*(_QWORD *)v22 + 600LL);
        if ( v25 == sub_2FE9890 )
        {
          v26 = sub_2FE8D10(v22, (__int64)v24, v330, v331, &v332, &v324, &v323);
        }
        else
        {
          v26 = ((__int64 (__fastcall *)(__int64, unsigned __int16 *, _QWORD, _QWORD, unsigned __int64))v25)(
                  v22,
                  v24,
                  (unsigned int)a9,
                  (unsigned int)v330,
                  v21);
          v24 = &v323;
        }
        v287 = v26;
      }
      else
      {
        v287 = sub_2FE8D10(v22, (__int64)v24, (unsigned int)v330, v21, &v332, &v324, &v323);
      }
      v27 = v332;
      v28 = v324;
      if ( (_WORD)v332 )
      {
        if ( (unsigned __int16)(v332 - 17) <= 0xD3u )
        {
          v29 = (unsigned __int16)(v332 - 176) <= 0x34u;
          v30 = v29;
          v28 = word_4456340[(unsigned __int16)v332 - 1] * v324;
        }
        else
        {
          v29 = 0;
          v30 = 0;
        }
        if ( (unsigned __int16)(v332 - 17) <= 0xD3u )
        {
          v277 = 0;
          v27 = word_4456580[(unsigned __int16)v332 - 1];
          goto LABEL_107;
        }
      }
      else
      {
        v109 = sub_30070B0((__int64)&v332);
        LOBYTE(v110) = 0;
        v111 = v109;
        if ( v109 )
        {
          v326 = sub_3007240((__int64)&v332);
          v28 *= (_DWORD)v326;
          v110 = HIDWORD(v326);
          v111 = BYTE4(v326);
        }
        v278 = v110;
        v280 = v111;
        v112 = sub_30070B0((__int64)&v332);
        v29 = v280;
        v30 = v278;
        if ( v112 )
        {
          v151 = sub_3009970((__int64)&v332, (__int64)v24, v113, v280, v278);
          v29 = v280;
          v30 = v278;
          v277 = v152;
          v27 = v151;
LABEL_107:
          v114 = *(__int64 **)(a1 + 64);
          v115 = v28;
          LODWORD(v351) = v28;
          v116 = v27;
          BYTE4(v351) = v30;
          v281 = v114;
          if ( v29 )
            v118 = (unsigned int)sub_2D43AD0(v27, v28);
          else
            v118 = (unsigned int)sub_2D43050(v27, v28);
          v120 = v118;
          if ( (_WORD)v118 )
          {
            v121 = v330;
            LOWORD(v334) = v118;
            v335 = 0;
            if ( (_WORD)v118 == (_WORD)v330 )
              goto LABEL_111;
            goto LABEL_235;
          }
          v115 = v116;
          v174 = sub_3009450(v281, v116, v277, (__int64)v351, v118, v119);
          v121 = v330;
          LOWORD(v334) = v174;
          v335 = v175;
          if ( (_WORD)v330 == v174 )
          {
            if ( (_WORD)v330 || v331 == v175 )
            {
LABEL_111:
              v122 = v324;
              v123 = v353;
              v124 = v353;
              v351 = v353;
              v125 = v324;
              v352 = 0x800000000LL;
              if ( v324 )
              {
                if ( v324 > 8uLL )
                {
                  sub_C8D5F0((__int64)&v351, v353, v324, 0x10u, v118, v119);
                  v124 = v351;
                  v123 = &v351[16 * (unsigned int)v352];
                }
                for ( i = &v124[16 * v122]; i != v123; v123 += 16 )
                {
                  if ( v123 )
                  {
                    *(_QWORD *)v123 = 0;
                    *((_DWORD *)v123 + 2) = 0;
                  }
                }
                v127 = v324;
                LODWORD(v352) = v125;
                if ( v324 )
                {
                  v127 = 0;
                  v297 = (unsigned int)v301;
                  do
                  {
                    if ( (_WORD)v332 )
                    {
                      if ( (unsigned __int16)(v332 - 17) > 0xD3u )
                        goto LABEL_127;
                      v128 = word_4456340[(unsigned __int16)v332 - 1];
                    }
                    else
                    {
                      if ( !sub_30070B0((__int64)&v332) )
                      {
LABEL_127:
                        *(_QWORD *)&v299 = v127;
                        *(_QWORD *)&v136 = sub_3400EE0(a1, v127, v12, 0, v118);
                        v300.m128i_i64[1] = v297 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                        *((_QWORD *)&v275 + 1) = v300.m128i_i64[1];
                        v300.m128i_i64[0] = v302;
                        *(_QWORD *)&v275 = v302;
                        v315 = sub_3406EB0(a1, 158, v12, v332, v333, v137, v275, v136);
                        v138 = (char *)&v351[16 * v299];
                        v316 = v139;
                        *(_QWORD *)v138 = v315;
                        *((_DWORD *)v138 + 2) = v316;
                        goto LABEL_123;
                      }
                      v336[0] = sub_3007240((__int64)&v332);
                      v128 = v336[0];
                    }
                    *(_QWORD *)&v129 = sub_3400EE0(a1, v127 * v128, v12, 0, v118);
                    v300.m128i_i64[1] = v297 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                    *((_QWORD *)&v274 + 1) = v300.m128i_i64[1];
                    v300.m128i_i64[0] = v302;
                    *(_QWORD *)&v274 = v302;
                    v131 = sub_3406EB0(a1, 161, v12, v332, v333, v130, v274, v129);
                    v133 = v132;
                    v134 = v131;
                    v135 = (char *)&v351[16 * v127];
                    v317 = v134;
                    v318 = v133;
                    *(_QWORD *)v135 = v134;
                    *((_DWORD *)v135 + 2) = v318;
LABEL_123:
                    ++v127;
                  }
                  while ( v324 != v127 );
                }
                v140 = v351;
              }
              else
              {
                v140 = v353;
                v127 = 0;
              }
              if ( v287 == v127 )
              {
                if ( v287 )
                {
                  for ( j = 0; ; j += 16 )
                  {
                    v172 = *(_QWORD *)&v140[j];
                    v173 = *(_QWORD *)&v140[j + 8];
                    LODWORD(v325) = a9;
                    BYTE4(v325) = BYTE4(a9);
                    sub_33695F0(a1, v12, v172, v173, (_DWORD)a5 + j, 1, a7, (__int64)a8, v325, 215);
                    v140 = v351;
                    if ( j == 16LL * (v287 - 1) )
                      break;
                  }
                }
              }
              else if ( v287 )
              {
                v141 = v287 / v127;
                if ( v324 )
                {
                  v142 = 0;
                  for ( k = 0; k != v324; ++k )
                  {
                    v144 = v142;
                    v142 += v141;
                    LODWORD(v325) = a9;
                    BYTE4(v325) = BYTE4(a9);
                    v145 = k;
                    sub_33695F0(
                      a1,
                      v12,
                      *(_QWORD *)&v140[16 * v145],
                      *(_QWORD *)&v140[16 * v145 + 8],
                      (_DWORD)a5 + 16 * v144,
                      v141,
                      a7,
                      (__int64)a8,
                      v325,
                      215);
                    v140 = v351;
                  }
                }
              }
              if ( v140 != v353 )
                _libc_free((unsigned __int64)v140);
              return;
            }
          }
          else if ( v174 )
          {
            v120 = v174;
LABEL_235:
            if ( v120 == 1 || (unsigned __int16)(v120 - 504) <= 7u )
              goto LABEL_375;
            v176 = *(_QWORD *)&byte_444C4A0[16 * v120 - 16];
            v282 = byte_444C4A0[16 * v120 - 8];
LABEL_187:
            if ( v121 )
            {
              if ( v121 == 1 || (unsigned __int16)(v121 - 504) <= 7u )
                goto LABEL_375;
              v178 = *(_QWORD *)&byte_444C4A0[16 * v121 - 16];
              v179 = (unsigned __int8)byte_444C4A0[16 * v121 - 8];
            }
            else
            {
              v178 = sub_3007260((__int64)&v330);
              v336[4] = v178;
              v336[5] = v179;
              v179 = (unsigned __int8)v179;
            }
            if ( v178 == v176 && (_BYTE)v179 == v282 )
            {
              v302 = sub_33FAF80(a1, 234, v12, v334, v335, v119);
              LODWORD(v301) = v207;
              v300.m128i_i64[0] = v302;
              v300.m128i_i64[1] = v207 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
              goto LABEL_111;
            }
            if ( v120 )
            {
              v180 = word_4456580;
              v181 = 0;
              v182 = word_4456580[v120 - 1];
            }
            else
            {
              v208 = sub_3009970((__int64)&v334, v115, v179, v117, v118);
              v121 = v330;
              v182 = v208;
              v181 = v180;
            }
            LOWORD(v336[0]) = v182;
            v336[1] = v181;
            if ( v121 )
            {
              v183 = word_4456580[v121 - 1];
              v184 = 0;
            }
            else
            {
              v183 = sub_3009970((__int64)&v330, v115, (__int64)v180, v117, v118);
            }
            if ( v183 == v182 )
            {
              if ( v183 || v181 == (_WORD *)v184 )
                goto LABEL_196;
              v352 = v184;
              LOWORD(v351) = 0;
            }
            else
            {
              LOWORD(v351) = v183;
              v352 = v184;
              if ( v183 )
              {
                if ( v183 == 1 || (unsigned __int16)(v183 - 504) <= 7u )
                  goto LABEL_375;
                v189 = *(_QWORD *)&byte_444C4A0[16 * v183 - 16];
                v191 = byte_444C4A0[16 * v183 - 8];
LABEL_206:
                if ( v182 )
                {
                  if ( v182 == 1 || (unsigned __int16)(v182 - 504) <= 7u )
                    goto LABEL_375;
                  v194 = *(_QWORD *)&byte_444C4A0[16 * v182 - 16];
                  LOBYTE(v193) = byte_444C4A0[16 * v182 - 8];
                }
                else
                {
                  v117 = sub_3007260((__int64)v336);
                  v193 = v192;
                  v339 = v117;
                  v194 = v117;
                  v340 = v193;
                }
                if ( ((_BYTE)v193 || !v191) && v189 < v194 )
                {
                  if ( (_WORD)v330 )
                  {
                    v195 = word_4456340;
                    LOBYTE(v117) = (unsigned __int16)(v330 - 176) <= 0x34u;
                    v196 = (unsigned int)v117;
                    v197 = word_4456340[(unsigned __int16)v330 - 1];
                  }
                  else
                  {
                    v327 = sub_3007240((__int64)&v330);
                    v197 = v327;
                    v196 = HIDWORD(v327);
                    v117 = HIDWORD(v327);
                  }
                  if ( (_WORD)v334 )
                  {
                    v198 = 0;
                    v199 = word_4456580[(unsigned __int16)v334 - 1];
                  }
                  else
                  {
                    v298 = v196;
                    LODWORD(v301) = v197;
                    LOBYTE(v302) = v117;
                    v199 = sub_3009970((__int64)&v334, v197, (__int64)v195, v117, v196);
                    LOBYTE(v196) = v298;
                    LODWORD(v197) = (_DWORD)v301;
                    LOBYTE(v117) = v302;
                    v198 = v249;
                  }
                  LODWORD(v351) = v197;
                  v200 = *(__int64 **)(a1 + 64);
                  v201 = v199;
                  BYTE4(v351) = v196;
                  if ( (_BYTE)v117 )
                    v202 = sub_2D43AD0(v199, v197);
                  else
                    v202 = sub_2D43050(v199, v197);
                  v204 = 0;
                  if ( !v202 )
                  {
                    v202 = sub_3009450(v200, v201, v198, (__int64)v351, 0, v203);
                    v204 = v248;
                  }
                  LOWORD(v330) = v202;
                  v331 = v204;
                  v185 = sub_33FAF80(a1, 215, v12, v330, v204, v203);
                  LODWORD(v301) = v205;
                  v295 = v205;
                  goto LABEL_197;
                }
LABEL_196:
                v185 = v302;
LABEL_197:
                v300.m128i_i64[0] = v185;
                v300.m128i_i64[1] = v295 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                v186 = sub_33686D0(a1, v185, v300.m128i_i64[1], v12, (unsigned int)v334, v335);
                v302 = v186;
                if ( v186 )
                {
                  v300.m128i_i64[0] = v186;
                  LODWORD(v301) = v187;
                  v300.m128i_i64[1] = v187 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                }
                else
                {
                  v302 = v185;
                }
                goto LABEL_111;
              }
            }
            v341 = sub_3007260((__int64)&v351);
            v189 = v341;
            v342 = v190;
            v191 = v190;
            goto LABEL_206;
          }
          v337 = sub_3007260((__int64)&v334);
          v176 = v337;
          v338 = v177;
          v282 = v177;
          goto LABEL_187;
        }
      }
      v277 = v333;
      goto LABEL_107;
    }
    v340 = 0;
    LOWORD(v339) = a7;
    if ( v23 == a7 && (!v21 || a7) )
    {
LABEL_203:
      v188 = (int)v301;
      a5->m128i_i64[0] = v302;
      a5->m128i_i32[2] = v188;
      return;
    }
    if ( v23 )
    {
      if ( v23 == 1 || (unsigned __int16)(v23 - 504) <= 7u )
        goto LABEL_375;
      v214 = *(_QWORD *)&byte_444C4A0[16 * v23 - 16];
      v215 = byte_444C4A0[16 * v23 - 8];
    }
    else
    {
      v211 = sub_3007260((__int64)&v330);
      v213 = v212;
      v336[2] = v211;
      v214 = v211;
      v336[3] = v213;
      v215 = v213;
    }
    if ( a7 <= 1u || (unsigned __int16)(a7 - 504) <= 7u )
      goto LABEL_375;
    v216 = a7 - 1;
    if ( v214 == *(_QWORD *)&byte_444C4A0[16 * v216] && byte_444C4A0[16 * v216 + 8] == v215 )
    {
      v244 = a7;
      v245 = a2;
      v246 = 234;
LABEL_310:
      v302 = sub_33FAF80(a1, v246, v245, v244, 0, a6);
      LODWORD(v301) = v247;
      goto LABEL_203;
    }
    v217 = v299;
    v302 = sub_33686D0(a1, v299, *((__int64 *)&v299 + 1), v12, a7, 0);
    LODWORD(v301) = v218;
    if ( v302 )
      goto LABEL_203;
    if ( (unsigned __int16)(a7 - 17) <= 0xD3u )
    {
      if ( (_WORD)v339 )
      {
        v263 = (unsigned __int64 *)word_4456580;
        v264 = 0;
        v217 = (unsigned __int16)word_4456580[(unsigned __int16)v339 - 1];
      }
      else
      {
        v217 = (unsigned int)sub_3009970((__int64)&v339, v217, v218, v219, v220);
        v264 = v263;
      }
      LOWORD(v345) = v217;
      v346 = v264;
      if ( (_WORD)v330 )
      {
        v265 = word_4456580[(unsigned __int16)v330 - 1];
        v266 = 0;
      }
      else
      {
        v301 = v264;
        v265 = sub_3009970((__int64)&v330, v217, (__int64)v263, (__int64)v264, v220);
        v217 = (unsigned int)v217;
        v264 = v301;
      }
      if ( v265 == (_WORD)v217 && (v265 || v266 == v264)
        || ((LOWORD(v347) = v265,
             v348 = v266,
             v351 = (const char *)sub_2D5B750((unsigned __int16 *)&v347),
             v352 = v267,
             v268 = sub_2D5B750((unsigned __int16 *)&v345),
             v350 = v269,
             v349 = v268,
             (_BYTE)v269)
         || !(_BYTE)v352)
        && v349 >= (unsigned __int64)v351 )
      {
        v351 = (const char *)sub_3281590((__int64)&v330);
        v349 = sub_3281590((__int64)&v339);
        if ( (_DWORD)v351 == (_DWORD)v349 && BYTE4(v349) == BYTE4(v351) )
        {
          v236 = *((_QWORD *)&v299 + 1);
          v235 = v299;
          goto LABEL_282;
        }
      }
    }
    v221 = (unsigned __int16)v339;
    if ( (_WORD)v339 )
    {
      if ( (unsigned __int16)(v339 - 17) > 0xD3u )
        goto LABEL_268;
    }
    else
    {
      LODWORD(v301) = 0;
      v254 = sub_30070B0((__int64)&v339);
      v221 = (unsigned int)v301;
      if ( !v254 )
        goto LABEL_268;
    }
    if ( (_WORD)v330 )
    {
      v217 = (unsigned __int16)word_4456580[(unsigned __int16)v330 - 1];
      v222 = 0;
    }
    else
    {
      v261 = sub_3009970((__int64)&v330, v217, v221, v219, v220);
      v222 = v262;
      v217 = v261;
      v221 = (unsigned __int16)v339;
    }
    if ( (_WORD)v221 )
    {
      v223 = word_4456580[(unsigned __int16)v221 - 1];
      v224 = 0;
    }
    else
    {
      v301 = v222;
      v223 = sub_3009970((__int64)&v339, v217, v221, (__int64)v222, v220);
      v217 = (unsigned int)v217;
      v222 = v301;
    }
    if ( v223 != (_WORD)v217 || !v223 && v222 != v224 )
    {
      v217 = v22;
      sub_2FE6CC0((__int64)&v351, v22, *(_QWORD *)(a1 + 64), v330, v331);
      if ( (_BYTE)v351 == 7 )
      {
        LODWORD(v332) = word_4456340[v216];
        BYTE4(v332) = (unsigned __int16)(a7 - 176) <= 0x34u;
        v257 = sub_3281170(&v330, v22, v255, v256, v220);
        v259 = sub_327FD70(*(__int64 **)(a1 + 64), v257, v258, v332);
        v235 = sub_33686D0(a1, v299, *((__int64 *)&v299 + 1), v12, v259, v260);
LABEL_282:
        v302 = sub_33FAFB0(a1, v235, v236, v12, a7, 0);
        LODWORD(v301) = v237;
        goto LABEL_203;
      }
    }
LABEL_268:
    if ( (_WORD)v330 )
    {
      if ( (unsigned __int16)(v330 - 176) <= 0x34u || word_4456340[(unsigned __int16)v330 - 1] != 1 )
        goto LABEL_271;
      if ( (unsigned __int16)(v330 - 10) > 6u
        && (unsigned __int16)(v330 - 126) > 0x31u
        && (unsigned __int16)(v330 - 208) > 0x14u )
      {
LABEL_327:
        if ( (unsigned __int16)(a7 - 10) > 6u
          && (unsigned __int16)(a7 - 126) > 0x31u
          && (unsigned __int16)(a7 - 208) > 0x14u )
        {
          *(_QWORD *)&v251 = sub_3400EE0(a1, 0, v12, 0, v220);
          v302 = sub_3406EB0(a1, 158, v12, a7, 0, v252, v299, v251);
          LODWORD(v301) = v253;
          goto LABEL_203;
        }
        LOWORD(v270) = sub_3281100((unsigned __int16 *)&v330, v217);
        v272 = sub_33FB890(a1, v270, v271, v299, *((_QWORD *)&v299 + 1));
        v246 = 233;
        v320 = v273;
        v300.m128i_i64[0] = v272;
        v319 = v272;
        v244 = a7;
        v245 = v12;
        v300.m128i_i64[1] = (unsigned int)v320 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        goto LABEL_310;
      }
    }
    else
    {
      v301 = &v330;
      v250 = sub_3007240((__int64)&v330);
      v225 = v301;
      v334 = v250;
      if ( BYTE4(v250) || (_DWORD)v250 != 1 )
        goto LABEL_272;
      if ( !(unsigned __int8)sub_3007030((__int64)v301) )
        goto LABEL_327;
    }
    if ( (unsigned __int16)(a7 - 2) <= 7u
      || (unsigned __int16)(a7 - 17) <= 0x6Cu
      || (unsigned __int16)(a7 - 176) <= 0x1Fu )
    {
LABEL_271:
      v225 = &v330;
LABEL_272:
      v230 = sub_2D5B750((unsigned __int16 *)v225);
      v227 = v226;
      v228 = v230;
      v351 = (const char *)v230;
      v229 = v230;
      v352 = v227;
      LOWORD(v230) = 2;
      if ( v228 != 1 )
      {
        LOWORD(v230) = 3;
        if ( v228 != 2 )
        {
          LOWORD(v230) = 4;
          if ( v228 != 4 )
          {
            LOWORD(v230) = 5;
            if ( v228 != 8 )
            {
              switch ( v228 )
              {
                case 16:
                  LOWORD(v230) = 6;
                  break;
                case 32:
                  LOWORD(v230) = 7;
                  break;
                case 64:
                  LOWORD(v230) = 8;
                  break;
                case 128:
                  LOWORD(v230) = 9;
                  break;
                default:
                  v230 = sub_3007020(*(_QWORD **)(a1 + 64), v229);
                  v286 = v230;
                  v302 = v231;
                  break;
              }
            }
          }
        }
      }
      v232 = v286;
      LOWORD(v232) = v230;
      v233 = sub_33FB890(a1, v232, v302, v299, *((_QWORD *)&v299 + 1));
      v322 = v234;
      v300.m128i_i64[0] = v233;
      v235 = v233;
      v321 = v233;
      v236 = (unsigned int)v234 | v300.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      goto LABEL_282;
    }
    goto LABEL_327;
  }
  if ( (unsigned __int16)(v17 - 17) <= 0xD3u )
    goto LABEL_11;
LABEL_4:
  if ( !a6 )
    return;
  if ( v17 == a7 )
  {
    if ( !v18 || v17 )
    {
      a5->m128i_i64[0] = v302;
      a5->m128i_i32[2] = (int)v301;
      return;
    }
LABEL_375:
    BUG();
  }
  if ( a7 <= 1u || (unsigned __int16)(a7 - 504) <= 7u )
    goto LABEL_375;
  v31 = 16LL * (a7 - 1);
  v32 = byte_444C4A0[v31 + 8];
  v33 = *(_QWORD *)&byte_444C4A0[v31];
  LOBYTE(v352) = v32;
  v351 = (const char *)v33;
  v284 = sub_CA1930(&v351);
  v293 = v284;
  v290 = v284 * a6;
  v300.m128i_i64[0] = v284 * a6;
  if ( (_WORD)v328 )
  {
    if ( (_WORD)v328 == 1 || (unsigned __int16)(v328 - 504) <= 7u )
      goto LABEL_375;
    v48 = 16LL * ((unsigned __int16)v328 - 1);
    v35 = *(_QWORD *)&byte_444C4A0[v48];
    v36 = byte_444C4A0[v48 + 8];
  }
  else
  {
    v343 = sub_3007260((__int64)&v328);
    v344 = v34;
    v35 = v343;
    v36 = v344;
  }
  v351 = (const char *)v35;
  LOBYTE(v352) = v36;
  if ( v300.m128i_i64[0] <= (unsigned __int64)sub_CA1930(&v351) )
  {
    if ( (_WORD)v328 )
    {
      if ( (_WORD)v328 == 1 || (unsigned __int16)(v328 - 504) <= 7u )
        goto LABEL_375;
      v150 = 16LL * ((unsigned __int16)v328 - 1);
      v56 = *(_QWORD *)&byte_444C4A0[v150];
      v57 = byte_444C4A0[v150 + 8];
    }
    else
    {
      v347 = sub_3007260((__int64)&v328);
      v348 = v55;
      v56 = v347;
      v57 = (char)v348;
    }
    v351 = (const char *)v56;
    LOBYTE(v352) = v57;
    if ( sub_CA1930(&v351) == v284 )
    {
      v146 = a7;
LABEL_225:
      v147 = a2;
      v148 = 234;
      goto LABEL_141;
    }
    if ( (_WORD)v328 )
    {
      if ( (_WORD)v328 == 1 || (unsigned __int16)(v328 - 504) <= 7u )
        goto LABEL_375;
      v243 = 16LL * ((unsigned __int16)v328 - 1);
      v59 = *(_QWORD *)&byte_444C4A0[v243];
      v60 = byte_444C4A0[v243 + 8];
    }
    else
    {
      v349 = sub_3007260((__int64)&v328);
      v350 = v58;
      v59 = v349;
      v60 = v350;
    }
    v351 = (const char *)v59;
    LOBYTE(v352) = v60;
    if ( v300.m128i_i64[0] >= (unsigned __int64)sub_CA1930(&v351) )
      goto LABEL_70;
    v238 = sub_327FC40(*(_QWORD **)(a1 + 64), v290);
    v329 = v239;
    v328 = v238;
    v242 = sub_33FAF80(a1, 216, a2, v238, v239, v240);
    LODWORD(v301) = v241;
    v302 = v242;
    if ( a7 != 261 )
    {
      v15 = 16LL * v241;
      goto LABEL_70;
    }
    *(_QWORD *)&v299 = v302;
    *((_QWORD *)&v299 + 1) = v241 | *((_QWORD *)&v299 + 1) & 0xFFFFFFFF00000000LL;
LABEL_227:
    v146 = 261;
    goto LABEL_225;
  }
  if ( (unsigned __int16)(a7 - 10) <= 6u
    || (unsigned __int16)(a7 - 126) <= 0x31u
    || (unsigned __int16)(a7 - 208) <= 0x14u )
  {
    if ( (_WORD)v328 )
    {
      if ( (unsigned __int16)(v328 - 10) > 6u
        && (unsigned __int16)(v328 - 126) > 0x31u
        && (unsigned __int16)(v328 - 208) > 0x14u )
      {
        goto LABEL_32;
      }
    }
    else if ( !(unsigned __int8)sub_3007030((__int64)&v328) )
    {
      goto LABEL_53;
    }
    v146 = a7;
    v147 = a2;
    v148 = 233;
LABEL_141:
    v302 = sub_33FAF80(a1, v148, v147, v146, 0, v37);
    LODWORD(v301) = v149;
    v15 = 16LL * v149;
    goto LABEL_70;
  }
  if ( (_WORD)v328 )
  {
    if ( (unsigned __int16)(v328 - 126) <= 0x31u || (unsigned __int16)(v328 - 10) <= 6u )
    {
      v38 = (unsigned __int16)v328;
      if ( (_WORD)v328 == 1 )
        goto LABEL_375;
LABEL_34:
      if ( (unsigned __int16)(v328 - 504) <= 7u )
        goto LABEL_375;
      v39 = 16LL * (v38 - 1);
      v40 = *(_QWORD *)&byte_444C4A0[v39];
      v41 = byte_444C4A0[v39 + 8];
LABEL_36:
      v351 = (const char *)v40;
      LOBYTE(v352) = v41;
      v42 = sub_CA1930(&v351);
      switch ( v42 )
      {
        case 1u:
          v44 = 2;
          break;
        case 2u:
          v44 = 3;
          break;
        case 4u:
          v44 = 4;
          break;
        case 8u:
          v44 = 5;
          break;
        case 0x10u:
          v44 = 6;
          break;
        case 0x20u:
          v44 = 7;
          break;
        case 0x40u:
          v44 = 8;
          break;
        case 0x80u:
          v44 = 9;
          break;
        default:
          v44 = sub_3007020(*(_QWORD **)(a1 + 64), v42);
          v46 = v45;
LABEL_45:
          LOWORD(v328) = v44;
          v329 = v46;
          v302 = sub_33FAF80(a1, 234, a2, v328, v46, v43);
          v295 = v47;
          goto LABEL_53;
      }
      v46 = 0;
      goto LABEL_45;
    }
LABEL_32:
    if ( (unsigned __int16)(v328 - 208) > 0x14u )
      goto LABEL_53;
    v38 = (unsigned __int16)v328;
    goto LABEL_34;
  }
  if ( (unsigned __int8)sub_3007030((__int64)&v328) )
  {
    v345 = sub_3007260((__int64)&v328);
    v346 = v206;
    v40 = v345;
    v41 = (char)v346;
    goto LABEL_36;
  }
LABEL_53:
  switch ( v290 )
  {
    case 1u:
      v49 = 2;
      break;
    case 2u:
      v49 = 3;
      break;
    case 4u:
      v49 = 4;
      break;
    case 8u:
      v49 = 5;
      break;
    case 0x10u:
      v49 = 6;
      break;
    case 0x20u:
      v49 = 7;
      break;
    case 0x40u:
      v49 = 8;
      break;
    case 0x80u:
      v49 = 9;
      break;
    default:
      v49 = sub_3007020(*(_QWORD **)(a1 + 64), v290);
      v51 = v50;
      goto LABEL_62;
  }
  v51 = 0;
LABEL_62:
  LOWORD(v328) = v49;
  v329 = v51;
  *(_QWORD *)&v299 = v302;
  *((_QWORD *)&v299 + 1) = v295 | *((_QWORD *)&v299 + 1) & 0xFFFFFFFF00000000LL;
  v52 = sub_33FAF80(a1, a10, a2, v328, v51, v37);
  v54 = v302;
  v302 = v52;
  LODWORD(v301) = v53;
  if ( a7 == 261 )
  {
    *(_QWORD *)&v299 = v302;
    *((_QWORD *)&v299 + 1) = v53 | *((_QWORD *)&v299 + 1) & 0xFFFFFFFF00000000LL;
    goto LABEL_227;
  }
  v15 = 16LL * v53;
LABEL_70:
  v61 = *(_QWORD *)(v302 + 48) + v15;
  v62 = *(_WORD *)v61;
  v63 = *(_QWORD *)(v61 + 8);
  LOWORD(v328) = *(_WORD *)v61;
  v329 = v63;
  if ( a6 != 1 )
  {
    v288 = a6;
    if ( ((a6 - 1) & a6) == 0 )
      goto LABEL_72;
    _BitScanReverse(&v154, a6);
    v155 = 0x80000000 >> (v154 ^ 0x1F);
    v156 = v155 * v284;
    v288 = v155;
    v300.m128i_i32[0] = a6 - v155;
    *(_QWORD *)&v157 = sub_3400E40(a1, v155 * v284, v328, v63, v12);
    *(_QWORD *)&v299 = v302;
    *((_QWORD *)&v299 + 1) = (unsigned int)v301 | *((_QWORD *)&v299 + 1) & 0xFFFFFFFF00000000LL;
    v159 = sub_3406EB0(a1, 192, v12, v328, v329, v158, __PAIR128__(*((unsigned __int64 *)&v299 + 1), v302), v157);
    v160 = &a5[v288];
    sub_33695F0(a1, v12, v159, v161, (_DWORD)a5 + 16 * v288, v300.m128i_i32[0], a7, (__int64)a8, a9, 215);
    if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1 + 40)) )
    {
      v209 = &a5[a6];
      if ( v160 != v209 )
      {
        while ( v160 < --v209 )
        {
          v210 = _mm_loadu_si128(v160++);
          v160[-1].m128i_i64[0] = v209->m128i_i64[0];
          v160[-1].m128i_i32[2] = v209->m128i_i32[2];
          v314 = v210;
          v209->m128i_i64[0] = v210.m128i_i64[0];
          v209->m128i_i32[2] = v314.m128i_i32[2];
        }
      }
    }
    switch ( v156 )
    {
      case 1u:
        v163 = 2;
        break;
      case 2u:
        v163 = 3;
        break;
      case 4u:
        v163 = 4;
        break;
      case 8u:
        v163 = 5;
        break;
      case 0x10u:
        v163 = 6;
        break;
      case 0x20u:
        v163 = 7;
        break;
      case 0x40u:
        v163 = 8;
        break;
      case 0x80u:
        v163 = 9;
        break;
      default:
        v163 = sub_3007020(*(_QWORD **)(a1 + 64), v156);
        v165 = v164;
LABEL_172:
        LOWORD(v328) = v163;
        v329 = v165;
        v302 = sub_33FAF80(a1, 216, v12, v328, v165, v162);
        v62 = v328;
        LODWORD(v301) = v166;
LABEL_72:
        if ( v62 )
        {
          if ( v62 == 1 || (unsigned __int16)(v62 - 504) <= 7u )
            goto LABEL_375;
          v153 = 16LL * (v62 - 1);
          v65 = *(_QWORD *)&byte_444C4A0[v153];
          v66 = byte_444C4A0[v153 + 8];
        }
        else
        {
          v351 = (const char *)sub_3007260((__int64)&v328);
          v352 = v64;
          v65 = (__int64)v351;
          v66 = v352;
        }
        v341 = v65;
        LOBYTE(v342) = v66;
        v67 = sub_CA1930(&v341);
        switch ( v67 )
        {
          case 1u:
            LOWORD(v69) = 2;
            break;
          case 2u:
            LOWORD(v69) = 3;
            break;
          case 4u:
            LOWORD(v69) = 4;
            break;
          case 8u:
            LOWORD(v69) = 5;
            break;
          case 0x10u:
            LOWORD(v69) = 6;
            break;
          case 0x20u:
            LOWORD(v69) = 7;
            break;
          case 0x40u:
            LOWORD(v69) = 8;
            break;
          case 0x80u:
            LOWORD(v69) = 9;
            break;
          default:
            v69 = sub_3007020(*(_QWORD **)(a1 + 64), v67);
            v276 = HIWORD(v69);
            v71 = v70;
LABEL_83:
            HIWORD(v72) = v276;
            LOWORD(v72) = v69;
            *(_QWORD *)&v299 = v302;
            *((_QWORD *)&v299 + 1) = (unsigned int)v301 | *((_QWORD *)&v299 + 1) & 0xFFFFFFFF00000000LL;
            v73 = sub_33FAF80(a1, 234, v12, v72, v71, v68);
            v313 = v74;
            v312 = v73;
            a5->m128i_i64[0] = v73;
            a5->m128i_i32[2] = v313;
            v296 = v288;
            while ( 1 )
            {
              v291 = v296;
              v75 = (v296 * v293) >> 1;
              v76 = 0;
              v296 >>= 1;
              v285 = 9 * (v75 == 128);
              v77 = v10;
              v78 = v12;
              v79 = v77;
              do
              {
                while ( 1 )
                {
                  switch ( v75 )
                  {
                    case 1u:
                      v82 = 2;
LABEL_150:
                      v84 = 0;
                      goto LABEL_95;
                    case 2u:
                      v82 = 3;
                      goto LABEL_150;
                    case 4u:
                      v82 = 4;
                      goto LABEL_150;
                    case 8u:
                      v82 = 5;
                      goto LABEL_150;
                    case 0x10u:
                      v82 = 6;
                      goto LABEL_150;
                    case 0x20u:
                      v82 = 7;
                      goto LABEL_150;
                    case 0x40u:
                      v82 = 8;
                      goto LABEL_150;
                  }
                  if ( v285 )
                  {
                    v82 = v285;
                    goto LABEL_150;
                  }
                  v80 = *(_QWORD **)(a1 + 64);
                  v300.m128i_i32[0] = v76;
                  LODWORD(v302) = v75;
                  v81 = sub_3007020(v80, v75);
                  v76 = v300.m128i_i32[0];
                  v75 = v302;
                  v79 = v81;
                  v82 = v81;
                  v84 = v83;
LABEL_95:
                  LOWORD(v79) = v82;
                  LODWORD(v301) = v75;
                  v85 = &a5[v76];
                  LODWORD(v299) = v82;
                  v300.m128i_i32[0] = v76;
                  v86 = &a5[v296 + v76];
                  v302 = v84;
                  *(_QWORD *)&v87 = sub_3400D50(a1, 1, v78, 0);
                  v89 = sub_3406EB0(a1, 53, v78, v79, v302, v88, (__int128)*v85, v87);
                  v311 = v90;
                  v310 = v89;
                  v86->m128i_i64[0] = v89;
                  v86->m128i_i32[2] = v311;
                  *(_QWORD *)&v91 = sub_3400D50(a1, 0, v78, 0);
                  v93 = sub_3406EB0(a1, 53, v78, v79, v302, v92, (__int128)*v85, v91);
                  v75 = (unsigned int)v301;
                  v94 = v293 == (_DWORD)v301;
                  v308 = v93;
                  v95 = v300.m128i_i32[0];
                  v309 = v96;
                  v97 = v299;
                  v85->m128i_i64[0] = v93;
                  v85->m128i_i32[2] = v309;
                  if ( v94 && a7 != v97 )
                    break;
                  v76 = v291 + v95;
                  if ( v288 <= v76 )
                    goto LABEL_98;
                }
                HIWORD(v98) = v279;
                LOWORD(v98) = a7;
                LODWORD(v302) = v75;
                v99 = sub_33FAF80(a1, 234, v78, v98, 0, v95);
                v307 = v100;
                v306 = v99;
                v85->m128i_i64[0] = v99;
                v85->m128i_i32[2] = v307;
                WORD1(v99) = v283;
                LOWORD(v99) = a7;
                v102 = sub_33FAF80(a1, 234, v78, v99, 0, v101);
                v75 = v302;
                v304 = v102;
                v76 = v291 + v300.m128i_i32[0];
                v305 = v103;
                v86->m128i_i64[0] = v102;
                v86->m128i_i32[2] = v305;
              }
              while ( v288 > v76 );
LABEL_98:
              v104 = v79;
              v12 = v78;
              v10 = v104;
              if ( v296 == 1 )
              {
                if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1 + 40)) )
                {
                  v105 = a5;
                  v106 = (unsigned __int64)&a5[a6 - 1];
                  if ( (unsigned __int64)a5 < v106 )
                  {
                    do
                    {
                      v107 = *(_QWORD *)v106;
                      v108 = _mm_loadu_si128(v105);
                      v106 -= 16LL;
                      ++v105;
                      v105[-1].m128i_i64[0] = v107;
                      v105[-1].m128i_i32[2] = *(_DWORD *)(v106 + 24);
                      v303 = v108;
                      *(_QWORD *)(v106 + 16) = v108.m128i_i64[0];
                      *(_DWORD *)(v106 + 24) = v303.m128i_i32[2];
                    }
                    while ( (unsigned __int64)v105 < v106 );
                  }
                }
                return;
              }
            }
        }
        v71 = 0;
        goto LABEL_83;
    }
    v165 = 0;
    goto LABEL_172;
  }
  if ( v62 != a7 )
  {
    v167 = *(_QWORD *)(a1 + 64);
    v351 = "scalar-to-vector conversion failed";
    v353[17] = 1;
    v353[16] = 3;
    sub_33681A0(v167, a8, (__int64)&v351, v54);
    *(_QWORD *)&v299 = v302;
    v302 = sub_33FAF80(a1, 234, v12, a7, 0, v168);
    LODWORD(v301) = v169;
  }
  v170 = (int)v301;
  a5->m128i_i64[0] = v302;
  a5->m128i_i32[2] = v170;
}
