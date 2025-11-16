// Function: sub_E49E40
// Address: 0xe49e40
//
__int64 __fastcall sub_E49E40(__int64 *a1, __int64 a2, unsigned int a3, __m128i *a4)
{
  __m128i v5; // xmm0
  void (__fastcall *v6)(__m128i *, __m128i *, __int64); // r8
  __m128i v7; // xmm1
  void (__fastcall *v8)(__m128i *, __int64, __int64 *); // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rbx
  __int64 v15; // rcx
  _QWORD *v16; // r12
  unsigned int v17; // eax
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r14
  const void *v23; // r15
  size_t v24; // rdx
  size_t v25; // r13
  int v26; // eax
  int v27; // eax
  unsigned int v28; // r8d
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned int v31; // r14d
  int v32; // r13d
  _QWORD *v33; // rax
  __int64 v34; // rcx
  _QWORD *v35; // rdx
  _QWORD *v36; // rdi
  _QWORD *v37; // rax
  const void *v38; // r14
  size_t v39; // rdx
  size_t v40; // r12
  int v41; // eax
  int v42; // eax
  __int64 v43; // rdx
  _QWORD *v44; // rax
  unsigned int v45; // ecx
  __int64 *v46; // rax
  _QWORD *v47; // rdx
  int v48; // r10d
  __int64 *v49; // r9
  int v50; // edx
  __int64 v51; // r13
  __int64 v52; // rdi
  __int64 v53; // r13
  __int64 v54; // rdi
  __int64 v55; // r13
  __int64 v56; // rdi
  _QWORD **v57; // rax
  _QWORD *v58; // rbx
  _QWORD **v59; // r13
  __int64 v60; // r12
  int v61; // r10d
  _QWORD *v62; // r9
  unsigned int v63; // ecx
  _QWORD *v64; // rax
  __int64 v65; // rdx
  _BYTE *v66; // rsi
  _BYTE *v67; // rcx
  __int64 v68; // rdi
  _QWORD *v69; // rbx
  _QWORD **v70; // r13
  __int64 v71; // r12
  int v72; // r10d
  _QWORD *v73; // r9
  unsigned int v74; // ecx
  _QWORD *v75; // rax
  __int64 v76; // rdx
  _BYTE *v77; // rsi
  _BYTE *v78; // rcx
  __int64 v79; // rdi
  _QWORD *v80; // rbx
  _QWORD **v81; // r14
  __int64 v82; // rax
  __int64 v83; // r12
  int v84; // r10d
  __int64 *v85; // r9
  unsigned int v86; // r15d
  unsigned int v87; // ecx
  __int64 *v88; // rax
  __int64 v89; // rdx
  _BYTE *v90; // rsi
  _BYTE *v91; // rdx
  __int64 v92; // rdi
  _QWORD **v93; // r13
  _QWORD *v94; // r15
  __int64 v95; // rsi
  unsigned int v96; // r12d
  unsigned int v97; // eax
  _QWORD *v98; // rbx
  _QWORD *v99; // r13
  __int64 v100; // rdi
  __m128i *v101; // rsi
  __int64 v102; // r8
  __int64 v103; // r13
  __int64 v104; // rbx
  _QWORD *v105; // rdi
  __int64 v106; // rsi
  __int64 v107; // rdx
  __int64 v108; // rcx
  _QWORD **v109; // r13
  _QWORD *v111; // r13
  int v112; // edi
  __int64 *v113; // rdx
  _QWORD *v114; // rcx
  int v115; // r10d
  int v116; // edx
  unsigned int v117; // ecx
  int v118; // edx
  __int64 v119; // r10
  int v120; // edi
  int v121; // r8d
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 v125; // rdx
  unsigned __int64 v126; // rax
  _QWORD *v127; // r15
  _QWORD **v128; // r13
  int v129; // edx
  int v130; // edx
  int v131; // edx
  _QWORD **v132; // r14
  _QWORD *v133; // r12
  unsigned int v134; // r13d
  __int64 v135; // r11
  __int64 *v136; // rax
  __int64 v137; // rdx
  __int64 v138; // rdi
  __int64 v139; // rax
  __int64 v140; // rbx
  unsigned int v141; // ecx
  int v142; // edx
  __int64 v143; // r8
  _QWORD *v144; // r15
  _QWORD **v145; // r13
  __int64 *v146; // r15
  __int64 *v147; // r14
  _QWORD *v148; // r15
  _QWORD **i; // r13
  __int64 v150; // rdx
  _QWORD *v151; // rax
  __int64 v152; // rdx
  __int64 v153; // r8
  __int64 v154; // r13
  _QWORD *v155; // rbx
  __int64 v156; // rdx
  __int64 v157; // rcx
  _QWORD *v158; // rax
  __int64 v159; // rbx
  __m128i *v160; // r13
  __int64 v161; // rdi
  char v162; // al
  unsigned int v163; // eax
  __int64 *v164; // rdi
  __int64 v165; // r10
  int v166; // edi
  int v167; // r11d
  _QWORD *v168; // rcx
  unsigned int v169; // r14d
  _QWORD *v170; // r9
  _QWORD *v171; // r9
  __int64 v172; // r14
  int v173; // ecx
  __int64 v174; // rsi
  __int64 v175; // rcx
  __int64 v176; // r8
  int v177; // edi
  _QWORD *v178; // rsi
  __int64 v179; // rcx
  __int64 v180; // r8
  int v181; // edi
  __int64 *v182; // rsi
  _QWORD *v183; // r9
  __int64 v184; // r14
  int v185; // ecx
  __int64 v186; // rsi
  __int64 v187; // rcx
  __int64 v188; // r8
  int v189; // edi
  _QWORD *v190; // rsi
  __int64 *v191; // r9
  __int64 v192; // r15
  int v193; // ecx
  __int64 v194; // rsi
  __int64 *v195; // r12
  char v196; // cl
  __int64 v197; // r11
  _QWORD *v198; // rdx
  __int64 v199; // r9
  char v200; // r8
  _QWORD *v201; // rax
  __int64 v202; // r15
  __int64 v203; // rdx
  __int64 v204; // rcx
  _BYTE *v205; // r14
  __int64 v206; // rdi
  __m128i v207; // rax
  char v208; // al
  __int64 *v209; // rdx
  __int64 v210; // r14
  int v211; // ecx
  __int64 *v212; // rdi
  __int64 *v213; // rdi
  unsigned int v214; // r13d
  int v215; // ecx
  __int64 v216; // rax
  int v217; // r8d
  __int64 **v218; // rax
  __int64 *v219; // r13
  __int64 *v220; // r12
  __int64 v221; // rdi
  __int64 v222; // r14
  const char *v223; // rax
  unsigned __int64 v224; // rdx
  __int64 v225; // rax
  unsigned int v226; // eax
  const char *v227; // rax
  int v228; // ecx
  __int64 v229; // r9
  int v230; // edi
  unsigned int v231; // ecx
  __int64 v232; // r9
  int v233; // edi
  __int64 *v234; // rcx
  int v235; // r14d
  __int64 v236; // r8
  __m128i v237; // xmm3
  __int64 *v238; // r12
  const char *v239; // r15
  size_t v240; // rdx
  size_t v241; // rbx
  int v242; // eax
  unsigned int v243; // r13d
  __int64 v244; // rax
  _QWORD *v245; // rcx
  __int64 v246; // r14
  __int64 v247; // rbx
  __m128i *v248; // rdi
  _QWORD **v249; // r15
  unsigned __int64 v250; // rdi
  __int64 *v251; // rax
  __int64 *v252; // rcx
  __int64 *v253; // r13
  __int64 v254; // rax
  __int64 *v255; // rcx
  unsigned int v256; // r14d
  __int64 v257; // r8
  int v258; // edi
  __int64 v259; // [rsp-10h] [rbp-2A0h]
  __int64 v260; // [rsp-8h] [rbp-298h]
  __int64 v261; // [rsp+0h] [rbp-290h]
  _QWORD **v262; // [rsp+8h] [rbp-288h]
  _QWORD *v263; // [rsp+8h] [rbp-288h]
  char v264; // [rsp+10h] [rbp-280h]
  unsigned __int64 v265; // [rsp+18h] [rbp-278h]
  char v266; // [rsp+18h] [rbp-278h]
  __int64 v267; // [rsp+20h] [rbp-270h]
  __int64 v268; // [rsp+20h] [rbp-270h]
  __int64 v269; // [rsp+20h] [rbp-270h]
  _QWORD *v270; // [rsp+20h] [rbp-270h]
  __int64 v271; // [rsp+28h] [rbp-268h]
  _QWORD *v272; // [rsp+38h] [rbp-258h]
  __int64 v273; // [rsp+40h] [rbp-250h]
  unsigned int v274; // [rsp+48h] [rbp-248h]
  unsigned int v275; // [rsp+48h] [rbp-248h]
  __int64 v276; // [rsp+48h] [rbp-248h]
  __int64 v277; // [rsp+48h] [rbp-248h]
  unsigned int v278; // [rsp+48h] [rbp-248h]
  __int64 *v279; // [rsp+48h] [rbp-248h]
  unsigned int v280; // [rsp+48h] [rbp-248h]
  __int64 *v281; // [rsp+48h] [rbp-248h]
  __int64 *v282; // [rsp+48h] [rbp-248h]
  int v283; // [rsp+50h] [rbp-240h]
  char v284; // [rsp+50h] [rbp-240h]
  char v285; // [rsp+50h] [rbp-240h]
  __int64 v286; // [rsp+50h] [rbp-240h]
  __int64 *v287; // [rsp+58h] [rbp-238h]
  unsigned __int8 v288; // [rsp+67h] [rbp-229h] BYREF
  __int64 v289; // [rsp+68h] [rbp-228h] BYREF
  __int64 v290; // [rsp+70h] [rbp-220h] BYREF
  __int64 v291; // [rsp+78h] [rbp-218h] BYREF
  __int64 v292; // [rsp+80h] [rbp-210h] BYREF
  __int64 v293; // [rsp+88h] [rbp-208h] BYREF
  __int64 *v294; // [rsp+90h] [rbp-200h] BYREF
  __int64 v295; // [rsp+98h] [rbp-1F8h]
  __int64 v296[4]; // [rsp+A0h] [rbp-1F0h] BYREF
  __int64 v297; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v298; // [rsp+C8h] [rbp-1C8h]
  __int64 v299; // [rsp+D0h] [rbp-1C0h]
  __int64 v300; // [rsp+D8h] [rbp-1B8h]
  __int64 v301; // [rsp+E0h] [rbp-1B0h] BYREF
  __int64 *v302; // [rsp+E8h] [rbp-1A8h]
  __int64 v303; // [rsp+F0h] [rbp-1A0h]
  __int64 v304; // [rsp+F8h] [rbp-198h]
  __m128i v305; // [rsp+100h] [rbp-190h] BYREF
  __m128i v306; // [rsp+110h] [rbp-180h] BYREF
  __int64 v307; // [rsp+120h] [rbp-170h]
  __int64 v308; // [rsp+130h] [rbp-160h] BYREF
  __int64 v309; // [rsp+138h] [rbp-158h]
  const void *v310; // [rsp+140h] [rbp-150h]
  size_t v311; // [rsp+148h] [rbp-148h]
  __int16 v312; // [rsp+150h] [rbp-140h]
  __m128i v313; // [rsp+160h] [rbp-130h] BYREF
  __m128i v314; // [rsp+170h] [rbp-120h] BYREF
  __int64 v315; // [rsp+180h] [rbp-110h]
  __int64 *v316; // [rsp+1A0h] [rbp-F0h] BYREF
  _QWORD **v317; // [rsp+1A8h] [rbp-E8h]
  __int64 v318; // [rsp+1B0h] [rbp-E0h] BYREF
  __int64 v319; // [rsp+1B8h] [rbp-D8h]
  __int64 v320; // [rsp+1C0h] [rbp-D0h]
  __int64 v321; // [rsp+1C8h] [rbp-C8h]
  __int64 *v322; // [rsp+1D0h] [rbp-C0h]
  __int64 v323; // [rsp+1D8h] [rbp-B8h]
  unsigned int v324; // [rsp+1E0h] [rbp-B0h] BYREF
  __int64 v325; // [rsp+1E8h] [rbp-A8h] BYREF
  __int64 v326; // [rsp+1F0h] [rbp-A0h]
  __int64 v327; // [rsp+1F8h] [rbp-98h]
  __m128i v328; // [rsp+200h] [rbp-90h] BYREF
  void (__fastcall *v329)(__m128i *, __m128i *, __int64); // [rsp+210h] [rbp-80h]
  void (__fastcall *v330)(__m128i *, __int64, __int64 *); // [rsp+218h] [rbp-78h]
  __int64 v331; // [rsp+220h] [rbp-70h] BYREF
  __int64 v332; // [rsp+228h] [rbp-68h]
  __int64 v333; // [rsp+230h] [rbp-60h]
  unsigned int v334; // [rsp+238h] [rbp-58h]
  __int64 v335; // [rsp+240h] [rbp-50h] BYREF
  _QWORD *v336; // [rsp+248h] [rbp-48h]
  __int64 v337; // [rsp+250h] [rbp-40h]
  unsigned int v338; // [rsp+258h] [rbp-38h]

  v5 = _mm_loadu_si128(a4);
  v6 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a4[1].m128i_i64[0];
  a4[1].m128i_i64[0] = 0;
  v7 = _mm_loadu_si128(&v313);
  v8 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a4[1].m128i_i64[1];
  a4[1].m128i_i64[1] = 0;
  v316 = a1;
  *a4 = v7;
  v9 = *(_QWORD *)a2;
  v322 = (__int64 *)&v324;
  v327 = 0x800000000LL;
  *(_QWORD *)a2 = 0;
  v317 = (_QWORD **)v9;
  v318 = 0;
  v319 = 0;
  v320 = 0;
  v321 = 0;
  v323 = 0;
  v324 = a3;
  v325 = 0;
  v326 = 0;
  v329 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v6;
  v313 = v5;
  v328 = v5;
  v330 = (void (__fastcall *)(__m128i *, __int64, __int64 *))v8;
  v10 = *a1;
  v332 = 0;
  v333 = 0;
  v334 = 0;
  v335 = 0;
  v336 = 0;
  v337 = 0;
  v338 = 0;
  v273 = v10;
  v11 = *(_DWORD *)(v9 + 136);
  v331 = 0;
  v297 = 0;
  v298 = 0;
  v299 = 0;
  v300 = 0;
  v301 = 0;
  v302 = 0;
  v303 = 0;
  v304 = 0;
  if ( v11 )
  {
    a2 = *(_QWORD *)(v9 + 128);
    if ( *(_QWORD *)a2 && *(_QWORD *)a2 != -8 )
    {
      v14 = *(__int64 **)(v9 + 128);
    }
    else
    {
      v12 = (__int64 *)(a2 + 8);
      do
      {
        do
        {
          v13 = *v12;
          v14 = v12++;
        }
        while ( v13 == -8 );
      }
      while ( !v13 );
    }
    v287 = (__int64 *)(a2 + 8LL * v11);
    if ( v287 != v14 )
    {
      v15 = *v14;
      v16 = (_QWORD *)(*v14 + 8);
LABEL_17:
      while ( 1 )
      {
        v274 = *(_DWORD *)(v15 + 16);
        v22 = *v316;
        v23 = (const void *)sub_AA8810(v16);
        v25 = v24;
        v26 = sub_C92610();
        v27 = sub_C92860((__int64 *)(v22 + 128), v23, v25, v26);
        v28 = v274;
        if ( v27 != -1 )
        {
          v29 = *(_QWORD *)(v22 + 128);
          v30 = v29 + 8LL * v27;
          if ( v30 != v29 + 8LL * *(unsigned int *)(v22 + 136) )
            break;
        }
        v283 = 1;
        a2 = v334;
        if ( !v334 )
        {
LABEL_130:
          ++v331;
          goto LABEL_131;
        }
LABEL_24:
        v32 = 1;
        v33 = 0;
        LODWORD(v34) = (a2 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v35 = (_QWORD *)(v332 + 16LL * (unsigned int)v34);
        v36 = (_QWORD *)*v35;
        if ( v16 == (_QWORD *)*v35 )
        {
LABEL_25:
          v37 = v35 + 1;
          goto LABEL_26;
        }
        while ( v36 != (_QWORD *)-4096LL )
        {
          if ( !v33 && v36 == (_QWORD *)-8192LL )
            v33 = v35;
          v34 = ((_DWORD)a2 - 1) & (unsigned int)(v34 + v32);
          v35 = (_QWORD *)(v332 + 16 * v34);
          v36 = (_QWORD *)*v35;
          if ( v16 == (_QWORD *)*v35 )
            goto LABEL_25;
          ++v32;
        }
        if ( !v33 )
          v33 = v35;
        ++v331;
        v118 = v333 + 1;
        if ( 4 * ((int)v333 + 1) < (unsigned int)(3 * a2) )
        {
          if ( (int)a2 - HIDWORD(v333) - v118 <= (unsigned int)a2 >> 3 )
          {
            v278 = v28;
            sub_E47D80((__int64)&v331, a2);
            if ( !v334 )
            {
LABEL_569:
              LODWORD(v333) = v333 + 1;
              BUG();
            }
            v168 = 0;
            v169 = (v334 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v28 = v278;
            v118 = v333 + 1;
            a2 = 1;
            v33 = (_QWORD *)(v332 + 16LL * v169);
            v170 = (_QWORD *)*v33;
            if ( v16 != (_QWORD *)*v33 )
            {
              while ( v170 != (_QWORD *)-4096LL )
              {
                if ( v170 == (_QWORD *)-8192LL && !v168 )
                  v168 = v33;
                v169 = (v334 - 1) & (a2 + v169);
                v33 = (_QWORD *)(v332 + 16LL * v169);
                v170 = (_QWORD *)*v33;
                if ( v16 == (_QWORD *)*v33 )
                  goto LABEL_236;
                a2 = (unsigned int)(a2 + 1);
              }
              if ( v168 )
                v33 = v168;
            }
          }
          goto LABEL_236;
        }
LABEL_131:
        v275 = v28;
        sub_E47D80((__int64)&v331, 2 * a2);
        if ( !v334 )
          goto LABEL_569;
        a2 = (unsigned int)v333;
        v28 = v275;
        v117 = (v334 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v118 = v333 + 1;
        v33 = (_QWORD *)(v332 + 16LL * v117);
        v119 = *v33;
        if ( v16 != (_QWORD *)*v33 )
        {
          v120 = 1;
          a2 = 0;
          while ( v119 != -4096 )
          {
            if ( v119 == -8192 && !a2 )
              a2 = (__int64)v33;
            v117 = (v334 - 1) & (v120 + v117);
            v33 = (_QWORD *)(v332 + 16LL * v117);
            v119 = *v33;
            if ( v16 == (_QWORD *)*v33 )
              goto LABEL_236;
            ++v120;
          }
          if ( a2 )
            v33 = (_QWORD *)a2;
        }
LABEL_236:
        LODWORD(v333) = v118;
        if ( *v33 != -4096 )
          --HIDWORD(v333);
        *v33 = v16;
        v37 = v33 + 1;
        *v37 = 0;
LABEL_26:
        *(_DWORD *)v37 = v28;
        *((_DWORD *)v37 + 1) = v283;
        if ( v283 )
        {
          if ( v283 != 1 )
            goto LABEL_10;
          v38 = (const void *)sub_AA8810(v16);
          v40 = v39;
          v41 = sub_C92610();
          a2 = (__int64)v38;
          v42 = sub_C92860((__int64 *)(v273 + 128), v38, v40, v41);
          if ( v42 == -1 )
            goto LABEL_10;
          v43 = *(_QWORD *)(v273 + 128);
          v44 = (_QWORD *)(v43 + 8LL * v42);
          if ( v44 == (_QWORD *)(v43 + 8LL * *(unsigned int *)(v273 + 136)) )
            goto LABEL_10;
          a2 = (unsigned int)v300;
          v16 = (_QWORD *)(*v44 + 8LL);
          if ( (_DWORD)v300 )
          {
            v45 = (v300 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v46 = (__int64 *)(v298 + 8LL * v45);
            v47 = (_QWORD *)*v46;
            if ( v16 == (_QWORD *)*v46 )
              goto LABEL_10;
            v48 = 1;
            v49 = 0;
            while ( v47 != (_QWORD *)-4096LL )
            {
              if ( v47 == (_QWORD *)-8192LL && !v49 )
                v49 = v46;
              v45 = (v300 - 1) & (v48 + v45);
              v46 = (__int64 *)(v298 + 8LL * v45);
              v47 = (_QWORD *)*v46;
              if ( v16 == (_QWORD *)*v46 )
                goto LABEL_10;
              ++v48;
            }
            if ( v49 )
              v46 = v49;
            ++v297;
            v50 = v299 + 1;
            if ( 4 * ((int)v299 + 1) < (unsigned int)(3 * v300) )
            {
              if ( (int)v300 - HIDWORD(v299) - v50 > (unsigned int)v300 >> 3 )
              {
LABEL_38:
                LODWORD(v299) = v50;
                if ( *v46 != -4096 )
                  --HIDWORD(v299);
                goto LABEL_128;
              }
              sub_A683D0((__int64)&v297, v300);
              if ( (_DWORD)v300 )
              {
                v255 = 0;
                v256 = (v300 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
                v50 = v299 + 1;
                a2 = 1;
                v46 = (__int64 *)(v298 + 8LL * v256);
                v257 = *v46;
                if ( v16 != (_QWORD *)*v46 )
                {
                  while ( v257 != -4096 )
                  {
                    if ( !v255 && v257 == -8192 )
                      v255 = v46;
                    v256 = (v300 - 1) & (a2 + v256);
                    v46 = (__int64 *)(v298 + 8LL * v256);
                    v257 = *v46;
                    if ( v16 == (_QWORD *)*v46 )
                      goto LABEL_38;
                    a2 = (unsigned int)(a2 + 1);
                  }
                  if ( v255 )
                    v46 = v255;
                }
                goto LABEL_38;
              }
LABEL_574:
              LODWORD(v299) = v299 + 1;
              BUG();
            }
          }
          else
          {
            ++v297;
          }
          sub_A683D0((__int64)&v297, 2 * v300);
          if ( (_DWORD)v300 )
          {
            a2 = (unsigned int)v299;
            v50 = v299 + 1;
            v231 = (v300 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v46 = (__int64 *)(v298 + 8LL * v231);
            v232 = *v46;
            if ( v16 != (_QWORD *)*v46 )
            {
              v233 = 1;
              a2 = 0;
              while ( v232 != -4096 )
              {
                if ( v232 == -8192 && !a2 )
                  a2 = (__int64)v46;
                v231 = (v300 - 1) & (v233 + v231);
                v46 = (__int64 *)(v298 + 8LL * v231);
                v232 = *v46;
                if ( v16 == (_QWORD *)*v46 )
                  goto LABEL_38;
                ++v233;
              }
              if ( a2 )
                v46 = (__int64 *)a2;
            }
            goto LABEL_38;
          }
          goto LABEL_574;
        }
        a2 = (unsigned int)v304;
        if ( !(_DWORD)v304 )
        {
          ++v301;
          goto LABEL_393;
        }
        v112 = (v304 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v113 = &v302[v112];
        v114 = (_QWORD *)*v113;
        if ( v16 == (_QWORD *)*v113 )
          goto LABEL_10;
        v115 = 1;
        v46 = 0;
        while ( 1 )
        {
          if ( v114 == (_QWORD *)-4096LL )
          {
            if ( !v46 )
              v46 = v113;
            ++v301;
            v116 = v303 + 1;
            if ( 4 * ((int)v303 + 1) < (unsigned int)(3 * v304) )
            {
              if ( (int)v304 - HIDWORD(v303) - v116 > (unsigned int)v304 >> 3 )
                goto LABEL_126;
              sub_A683D0((__int64)&v301, v304);
              if ( (_DWORD)v304 )
              {
                v234 = 0;
                v235 = (v304 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
                v116 = v303 + 1;
                a2 = 1;
                v46 = &v302[v235];
                v236 = *v46;
                if ( v16 != (_QWORD *)*v46 )
                {
                  while ( v236 != -4096 )
                  {
                    if ( v236 == -8192 && !v234 )
                      v234 = v46;
                    v235 = (v304 - 1) & (a2 + v235);
                    v46 = &v302[v235];
                    v236 = *v46;
                    if ( v16 == (_QWORD *)*v46 )
                      goto LABEL_126;
                    a2 = (unsigned int)(a2 + 1);
                  }
                  if ( v234 )
                    v46 = v234;
                }
                goto LABEL_126;
              }
LABEL_573:
              LODWORD(v303) = v303 + 1;
              BUG();
            }
LABEL_393:
            sub_A683D0((__int64)&v301, 2 * v304);
            if ( !(_DWORD)v304 )
              goto LABEL_573;
            a2 = (unsigned int)v303;
            v116 = v303 + 1;
            v228 = (v304 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v46 = &v302[v228];
            v229 = *v46;
            if ( v16 != (_QWORD *)*v46 )
            {
              v230 = 1;
              a2 = 0;
              while ( v229 != -4096 )
              {
                if ( !a2 && v229 == -8192 )
                  a2 = (__int64)v46;
                v228 = (v304 - 1) & (v230 + v228);
                v46 = &v302[v228];
                v229 = *v46;
                if ( v16 == (_QWORD *)*v46 )
                  goto LABEL_126;
                ++v230;
              }
              if ( a2 )
                v46 = (__int64 *)a2;
            }
LABEL_126:
            LODWORD(v303) = v116;
            if ( *v46 != -4096 )
              --HIDWORD(v303);
LABEL_128:
            *v46 = (__int64)v16;
            goto LABEL_10;
          }
          if ( v114 == (_QWORD *)-8192LL && !v46 )
            v46 = v113;
          v112 = (v304 - 1) & (v115 + v112);
          v113 = &v302[v112];
          v114 = (_QWORD *)*v113;
          if ( v16 == (_QWORD *)*v113 )
            break;
          ++v115;
        }
        do
        {
LABEL_10:
          v19 = v14[1];
          if ( v19 != -8 && v19 )
          {
            ++v14;
          }
          else
          {
            v20 = v14 + 2;
            do
            {
              do
              {
                v21 = *v20;
                v14 = v20++;
              }
              while ( !v21 );
            }
            while ( v21 == -8 );
          }
          if ( v287 == v14 )
            goto LABEL_40;
          v15 = *v14;
          a2 = v332;
          v16 = (_QWORD *)(*v14 + 8);
          if ( !v334 )
            goto LABEL_17;
          v17 = (v334 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v18 = *(_QWORD **)(v332 + 16LL * v17);
        }
        while ( v16 == v18 );
        v121 = 1;
        while ( v18 != (_QWORD *)-4096LL )
        {
          v17 = (v334 - 1) & (v121 + v17);
          v18 = *(_QWORD **)(v332 + 16LL * v17);
          if ( v16 == v18 )
            goto LABEL_10;
          ++v121;
        }
      }
      v31 = *(_DWORD *)(*(_QWORD *)v30 + 16LL);
      if ( ((v31 | v274) & 0xFFFFFFFD) == 0 )
      {
        if ( v31 != 2 && v274 != 2 )
          goto LABEL_22;
        v31 = 2;
        goto LABEL_144;
      }
      if ( v274 != v31 )
      {
        v310 = v23;
        v312 = 1283;
        v308 = (__int64)"Linking COMDATs named '";
        v313.m128i_i64[0] = (__int64)&v308;
        v314.m128i_i64[0] = (__int64)"': invalid selection kinds!";
        LOWORD(v315) = 770;
        v311 = v25;
        v111 = *v317;
        goto LABEL_116;
      }
      if ( v31 == 3 )
      {
        v283 = 2;
      }
      else
      {
        if ( v31 <= 3 )
        {
          if ( !v31 )
          {
LABEL_22:
            v283 = 0;
            v28 = 0;
            goto LABEL_23;
          }
LABEL_144:
          v276 = *v316;
          if ( (unsigned __int8)sub_E48140((__int64)&v316, *v316, (__int64)v23, v25, (__int64 *)&v294)
            || (unsigned __int8)sub_E48140((__int64)&v316, (__int64)v317, (__int64)v23, v25, v296) )
          {
            goto LABEL_117;
          }
          v262 = v317 + 39;
          v277 = v276 + 312;
          v267 = v294[3];
          v284 = sub_AE5020(v277, v267);
          v122 = sub_9208B0(v277, v267);
          v313.m128i_i64[1] = v123;
          v313.m128i_i64[0] = ((1LL << v284) + ((unsigned __int64)(v122 + 7) >> 3) - 1) >> v284 << v284;
          v265 = sub_CA1930(&v313);
          v268 = *(_QWORD *)(v296[0] + 24);
          v285 = sub_AE5020((__int64)v262, v268);
          v124 = sub_9208B0((__int64)v262, v268);
          v313.m128i_i64[1] = v125;
          v313.m128i_i64[0] = ((1LL << v285) + ((unsigned __int64)(v124 + 7) >> 3) - 1) >> v285 << v285;
          v126 = sub_CA1930(&v313);
          switch ( v31 )
          {
            case 1u:
              if ( *(_QWORD *)(v296[0] - 32) != *(v294 - 4) )
              {
                v311 = v25;
                v308 = (__int64)"Linking COMDATs named '";
                v313.m128i_i64[0] = (__int64)&v308;
                v227 = "': ExactMatch violated!";
                v312 = 1283;
                v310 = v23;
LABEL_390:
                v314.m128i_i64[0] = (__int64)v227;
                LOWORD(v315) = 770;
                v111 = *v317;
LABEL_116:
                sub_1061A30(&v305, 0, &v313);
                sub_B6EB20((__int64)v111, (__int64)&v305);
LABEL_117:
                v96 = 1;
                goto LABEL_92;
              }
              break;
            case 2u:
              v283 = v265 < v126;
              goto LABEL_151;
            case 4u:
              if ( v265 != v126 )
              {
                v310 = v23;
                v312 = 1283;
                v311 = v25;
                v308 = (__int64)"Linking COMDATs named '";
                v313.m128i_i64[0] = (__int64)&v308;
                v227 = "': SameSize violated!";
                goto LABEL_390;
              }
              break;
            default:
              BUG();
          }
          v283 = 0;
LABEL_151:
          v28 = v31;
          goto LABEL_23;
        }
        if ( v274 == 4 )
          goto LABEL_144;
      }
LABEL_23:
      a2 = v334;
      if ( !v334 )
        goto LABEL_130;
      goto LABEL_24;
    }
  }
LABEL_40:
  v51 = *(_QWORD *)(v273 + 48);
  while ( v273 + 40 != v51 )
  {
    v52 = v51;
    v51 = *(_QWORD *)(v51 + 8);
    a2 = (__int64)&v297;
    sub_E47BC0(v52 - 48, (__int64)&v297);
  }
  v53 = *(_QWORD *)(v273 + 16);
  while ( v273 + 8 != v53 )
  {
    v54 = v53;
    v53 = *(_QWORD *)(v53 + 8);
    a2 = (__int64)&v297;
    sub_E47BC0(v54 - 56, (__int64)&v297);
  }
  v55 = *(_QWORD *)(v273 + 32);
  while ( v273 + 24 != v55 )
  {
    v56 = v55;
    v55 = *(_QWORD *)(v55 + 8);
    a2 = (__int64)&v297;
    sub_E47BC0(v56 - 56, (__int64)&v297);
  }
  if ( !(_DWORD)v303 )
    goto LABEL_47;
  v308 = 0;
  v309 = 0;
  v310 = 0;
  v132 = v317 + 5;
  v311 = 0;
  v133 = v317[6];
  if ( v317 + 5 == v133 )
    goto LABEL_247;
  do
  {
    while ( 1 )
    {
      v138 = (__int64)(v133 - 6);
      if ( !v133 )
        v138 = 0;
      v139 = sub_B325F0(v138);
      v140 = v139;
      if ( v139 )
      {
        if ( *(_QWORD *)(v139 + 48) )
        {
          a2 = (unsigned int)v311;
          if ( !(_DWORD)v311 )
          {
            ++v308;
LABEL_219:
            sub_9E6990((__int64)&v308, 2 * v311);
            if ( !(_DWORD)v311 )
              goto LABEL_577;
            a2 = (unsigned int)v310;
            v141 = (v311 - 1) & (((unsigned int)v140 >> 9) ^ ((unsigned int)v140 >> 4));
            v142 = (_DWORD)v310 + 1;
            v136 = (__int64 *)(v309 + 8LL * v141);
            v143 = *v136;
            if ( v140 != *v136 )
            {
              v258 = 1;
              a2 = 0;
              while ( v143 != -4096 )
              {
                if ( v143 == -8192 && !a2 )
                  a2 = (__int64)v136;
                v141 = (v311 - 1) & (v258 + v141);
                v136 = (__int64 *)(v309 + 8LL * v141);
                v143 = *v136;
                if ( v140 == *v136 )
                  goto LABEL_221;
                ++v258;
              }
              if ( a2 )
                v136 = (__int64 *)a2;
            }
            goto LABEL_221;
          }
          v134 = ((unsigned int)v139 >> 9) ^ ((unsigned int)v139 >> 4);
          LODWORD(v135) = (v311 - 1) & v134;
          v136 = (__int64 *)(v309 + 8LL * (unsigned int)v135);
          v137 = *v136;
          if ( v140 != *v136 )
            break;
        }
      }
LABEL_212:
      v133 = (_QWORD *)v133[1];
      if ( v132 == v133 )
        goto LABEL_224;
    }
    v211 = 1;
    v212 = 0;
    while ( v137 != -4096 )
    {
      if ( !v212 && v137 == -8192 )
        v212 = v136;
      v135 = ((_DWORD)v311 - 1) & (unsigned int)(v135 + v211);
      v136 = (__int64 *)(v309 + 8 * v135);
      v137 = *v136;
      if ( v140 == *v136 )
        goto LABEL_212;
      ++v211;
    }
    if ( v212 )
      v136 = v212;
    ++v308;
    v142 = (_DWORD)v310 + 1;
    if ( 4 * ((int)v310 + 1) >= (unsigned int)(3 * v311) )
      goto LABEL_219;
    if ( (int)v311 - HIDWORD(v310) - v142 <= (unsigned int)v311 >> 3 )
    {
      sub_9E6990((__int64)&v308, v311);
      if ( !(_DWORD)v311 )
      {
LABEL_577:
        LODWORD(v310) = (_DWORD)v310 + 1;
        BUG();
      }
      v213 = 0;
      v214 = (v311 - 1) & v134;
      v142 = (_DWORD)v310 + 1;
      v215 = 1;
      v136 = (__int64 *)(v309 + 8LL * v214);
      a2 = *v136;
      if ( v140 != *v136 )
      {
        while ( a2 != -4096 )
        {
          if ( !v213 && a2 == -8192 )
            v213 = v136;
          v214 = (v311 - 1) & (v215 + v214);
          v136 = (__int64 *)(v309 + 8LL * v214);
          a2 = *v136;
          if ( v140 == *v136 )
            goto LABEL_221;
          ++v215;
        }
        if ( v213 )
          v136 = v213;
      }
    }
LABEL_221:
    LODWORD(v310) = v142;
    if ( *v136 != -4096 )
      --HIDWORD(v310);
    *v136 = v140;
    v133 = (_QWORD *)v133[1];
  }
  while ( v132 != v133 );
LABEL_224:
  if ( !(_DWORD)v303 )
    goto LABEL_225;
LABEL_247:
  v146 = v302;
  v147 = &v302[(unsigned int)v304];
  if ( v302 != v147 )
  {
    while ( *v146 == -8192 || *v146 == -4096 )
    {
      if ( v147 == ++v146 )
        goto LABEL_225;
    }
    if ( v147 != v146 )
    {
LABEL_261:
      v150 = *v146;
      v313.m128i_i64[0] = (__int64)&v314;
      v313.m128i_i64[1] = 0x600000000LL;
      v151 = *(_QWORD **)(v150 + 24);
      if ( *(_BYTE *)(v150 + 44) )
        v152 = *(unsigned int *)(v150 + 36);
      else
        v152 = *(unsigned int *)(v150 + 32);
      v153 = (__int64)&v151[v152];
      if ( v151 == (_QWORD *)v153 )
        goto LABEL_266;
      while ( 1 )
      {
        v154 = *v151;
        v155 = v151;
        if ( *v151 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( (_QWORD *)v153 == ++v151 )
          goto LABEL_266;
      }
      if ( (_QWORD *)v153 == v151 )
        goto LABEL_266;
      v156 = 0;
      v157 = *(_BYTE *)(v154 + 32) & 0xF;
      if ( (_BYTE)v157 == 8 )
        goto LABEL_287;
      while ( 1 )
      {
        do
        {
LABEL_275:
          v158 = v155 + 1;
          if ( v155 + 1 == (_QWORD *)v153 )
            goto LABEL_278;
          while ( 1 )
          {
            v154 = *v158;
            v155 = v158;
            if ( *v158 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( (_QWORD *)v153 == ++v158 )
              goto LABEL_278;
          }
          if ( (_QWORD *)v153 == v158 )
          {
LABEL_278:
            v159 = v313.m128i_i64[0];
            v160 = (__m128i *)(v313.m128i_i64[0] + 8LL * (unsigned int)v156);
            if ( (__m128i *)v313.m128i_i64[0] != v160 )
            {
              do
              {
                v161 = *(_QWORD *)v159;
                v162 = *(_BYTE *)(*(_QWORD *)v159 + 32LL) & 0xF0 | 1;
                *(_BYTE *)(*(_QWORD *)v159 + 32LL) = v162;
                if ( (v162 & 0x30) != 0 )
                  *(_BYTE *)(v161 + 33) |= 0x40u;
                a2 = 0;
                v159 += 8;
                sub_B2F990(v161, 0, v156, v157);
              }
              while ( v160 != (__m128i *)v159 );
              v160 = (__m128i *)v313.m128i_i64[0];
            }
            if ( v160 != &v314 )
              _libc_free(v160, a2);
LABEL_266:
            if ( ++v146 == v147 )
              goto LABEL_225;
            while ( *v146 == -4096 || *v146 == -8192 )
            {
              if ( v147 == ++v146 )
                goto LABEL_225;
            }
            if ( v147 == v146 )
              goto LABEL_225;
            goto LABEL_261;
          }
          v157 = *(_BYTE *)(v154 + 32) & 0xF;
        }
        while ( (_BYTE)v157 != 8 );
LABEL_287:
        v157 = (unsigned int)v311;
        a2 = v309;
        if ( !(_DWORD)v311 )
          goto LABEL_290;
        v163 = (v311 - 1) & (((unsigned int)v154 >> 9) ^ ((unsigned int)v154 >> 4));
        v164 = (__int64 *)(v309 + 8LL * v163);
        v165 = *v164;
        if ( v154 != *v164 )
          break;
LABEL_289:
        if ( v164 == (__int64 *)(v309 + 8LL * (unsigned int)v311) )
          goto LABEL_290;
      }
      v166 = 1;
      while ( v165 != -4096 )
      {
        v167 = v166 + 1;
        v163 = (v311 - 1) & (v163 + v166);
        v164 = (__int64 *)(v309 + 8LL * v163);
        v165 = *v164;
        if ( v154 == *v164 )
          goto LABEL_289;
        v166 = v167;
      }
LABEL_290:
      if ( v156 + 1 > (unsigned __int64)v313.m128i_u32[3] )
      {
        a2 = (__int64)&v314;
        v286 = v153;
        sub_C8D5F0((__int64)&v313, &v314, v156 + 1, 8u, v153, v156 + 1);
        v156 = v313.m128i_u32[2];
        v153 = v286;
      }
      *(_QWORD *)(v313.m128i_i64[0] + 8 * v156) = v154;
      v156 = (unsigned int)++v313.m128i_i32[2];
      goto LABEL_275;
    }
  }
LABEL_225:
  sub_C7D6A0(v309, 8LL * (unsigned int)v311, 8);
LABEL_47:
  v57 = v317;
  v58 = v317[2];
  v59 = v317 + 1;
  if ( v317 + 1 == v58 )
    goto LABEL_59;
  while ( 2 )
  {
    if ( !v58 )
      BUG();
    if ( (*(_BYTE *)(v58 - 3) & 0xFu) - 2 <= 1 )
    {
      v60 = *(v58 - 1);
      if ( v60 )
      {
        if ( v338 )
        {
          v61 = 1;
          v62 = 0;
          v63 = (v338 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v64 = &v336[4 * v63];
          v65 = *v64;
          if ( v60 == *v64 )
          {
LABEL_53:
            v66 = (_BYTE *)v64[2];
            v67 = (_BYTE *)v64[3];
            v68 = (__int64)(v64 + 1);
            v313.m128i_i64[0] = (__int64)(v58 - 7);
            if ( v67 != v66 )
            {
              if ( v66 )
              {
                *(_QWORD *)v66 = v58 - 7;
                v66 = (_BYTE *)v64[2];
              }
              v64[2] = v66 + 8;
              goto LABEL_57;
            }
LABEL_208:
            sub_E48660(v68, v66, &v313);
            goto LABEL_57;
          }
          while ( v65 != -4096 )
          {
            if ( v65 == -8192 && !v62 )
              v62 = v64;
            v63 = (v338 - 1) & (v61 + v63);
            v64 = &v336[4 * v63];
            v65 = *v64;
            if ( v60 == *v64 )
              goto LABEL_53;
            ++v61;
          }
          if ( v62 )
            v64 = v62;
          ++v335;
          v131 = v337 + 1;
          if ( 4 * ((int)v337 + 1) < 3 * v338 )
          {
            if ( v338 - HIDWORD(v337) - v131 <= v338 >> 3 )
            {
              sub_E487F0((__int64)&v335, v338);
              if ( !v338 )
                goto LABEL_570;
              v183 = 0;
              LODWORD(v184) = (v338 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v131 = v337 + 1;
              v185 = 1;
              v64 = &v336[4 * (unsigned int)v184];
              v186 = *v64;
              if ( v60 != *v64 )
              {
                while ( v186 != -4096 )
                {
                  if ( v186 == -8192 && !v183 )
                    v183 = v64;
                  v184 = (v338 - 1) & ((_DWORD)v184 + v185);
                  v64 = &v336[4 * v184];
                  v186 = *v64;
                  if ( v60 == *v64 )
                    goto LABEL_205;
                  ++v185;
                }
                if ( v183 )
                  v64 = v183;
              }
            }
LABEL_205:
            LODWORD(v337) = v131;
            if ( *v64 != -4096 )
              --HIDWORD(v337);
            *v64 = v60;
            v68 = (__int64)(v64 + 1);
            v66 = 0;
            v64[1] = 0;
            v64[2] = 0;
            v64[3] = 0;
            v313.m128i_i64[0] = (__int64)(v58 - 7);
            goto LABEL_208;
          }
        }
        else
        {
          ++v335;
        }
        sub_E487F0((__int64)&v335, 2 * v338);
        if ( !v338 )
        {
LABEL_570:
          LODWORD(v337) = v337 + 1;
          BUG();
        }
        LODWORD(v187) = (v338 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
        v131 = v337 + 1;
        v64 = &v336[4 * (unsigned int)v187];
        v188 = *v64;
        if ( v60 != *v64 )
        {
          v189 = 1;
          v190 = 0;
          while ( v188 != -4096 )
          {
            if ( !v190 && v188 == -8192 )
              v190 = v64;
            v187 = (v338 - 1) & ((_DWORD)v187 + v189);
            v64 = &v336[4 * v187];
            v188 = *v64;
            if ( v60 == *v64 )
              goto LABEL_205;
            ++v189;
          }
          if ( v190 )
            v64 = v190;
        }
        goto LABEL_205;
      }
    }
LABEL_57:
    v58 = (_QWORD *)v58[1];
    if ( v59 != v58 )
      continue;
    break;
  }
  v57 = v317;
LABEL_59:
  v69 = v57[4];
  v70 = v57 + 3;
  if ( v69 != v57 + 3 )
  {
    while ( 2 )
    {
      if ( !v69 )
        BUG();
      if ( (*(_BYTE *)(v69 - 3) & 0xFu) - 2 > 1 || (v71 = *(v69 - 1)) == 0 )
      {
LABEL_69:
        v69 = (_QWORD *)v69[1];
        if ( v70 == v69 )
        {
          v57 = v317;
          goto LABEL_71;
        }
        continue;
      }
      break;
    }
    if ( v338 )
    {
      v72 = 1;
      v73 = 0;
      v74 = (v338 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
      v75 = &v336[4 * v74];
      v76 = *v75;
      if ( v71 == *v75 )
      {
LABEL_65:
        v77 = (_BYTE *)v75[2];
        v78 = (_BYTE *)v75[3];
        v79 = (__int64)(v75 + 1);
        v313.m128i_i64[0] = (__int64)(v69 - 7);
        if ( v78 != v77 )
        {
          if ( v77 )
          {
            *(_QWORD *)v77 = v69 - 7;
            v77 = (_BYTE *)v75[2];
          }
          v75[2] = v77 + 8;
          goto LABEL_69;
        }
LABEL_194:
        sub_E48660(v79, v77, &v313);
        goto LABEL_69;
      }
      while ( v76 != -4096 )
      {
        if ( v76 == -8192 && !v73 )
          v73 = v75;
        v74 = (v338 - 1) & (v72 + v74);
        v75 = &v336[4 * v74];
        v76 = *v75;
        if ( v71 == *v75 )
          goto LABEL_65;
        ++v72;
      }
      if ( v73 )
        v75 = v73;
      ++v335;
      v130 = v337 + 1;
      if ( 4 * ((int)v337 + 1) < 3 * v338 )
      {
        if ( v338 - HIDWORD(v337) - v130 <= v338 >> 3 )
        {
          sub_E487F0((__int64)&v335, v338);
          if ( !v338 )
            goto LABEL_575;
          v171 = 0;
          LODWORD(v172) = (v338 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
          v130 = v337 + 1;
          v173 = 1;
          v75 = &v336[4 * (unsigned int)v172];
          v174 = *v75;
          if ( v71 != *v75 )
          {
            while ( v174 != -4096 )
            {
              if ( !v171 && v174 == -8192 )
                v171 = v75;
              v172 = (v338 - 1) & ((_DWORD)v172 + v173);
              v75 = &v336[4 * v172];
              v174 = *v75;
              if ( v71 == *v75 )
                goto LABEL_191;
              ++v173;
            }
            if ( v171 )
              v75 = v171;
          }
        }
LABEL_191:
        LODWORD(v337) = v130;
        if ( *v75 != -4096 )
          --HIDWORD(v337);
        *v75 = v71;
        v79 = (__int64)(v75 + 1);
        v77 = 0;
        v75[1] = 0;
        v75[2] = 0;
        v75[3] = 0;
        v313.m128i_i64[0] = (__int64)(v69 - 7);
        goto LABEL_194;
      }
    }
    else
    {
      ++v335;
    }
    sub_E487F0((__int64)&v335, 2 * v338);
    if ( !v338 )
    {
LABEL_575:
      LODWORD(v337) = v337 + 1;
      BUG();
    }
    LODWORD(v175) = (v338 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
    v130 = v337 + 1;
    v75 = &v336[4 * (unsigned int)v175];
    v176 = *v75;
    if ( v71 != *v75 )
    {
      v177 = 1;
      v178 = 0;
      while ( v176 != -4096 )
      {
        if ( !v178 && v176 == -8192 )
          v178 = v75;
        v175 = (v338 - 1) & ((_DWORD)v175 + v177);
        v75 = &v336[4 * v175];
        v176 = *v75;
        if ( v71 == *v75 )
          goto LABEL_191;
        ++v177;
      }
      if ( v178 )
        v75 = v178;
    }
    goto LABEL_191;
  }
LABEL_71:
  v80 = v57[6];
  v81 = v57 + 5;
  if ( v57 + 5 == v80 )
    goto LABEL_83;
  while ( 2 )
  {
    if ( !v80 )
      BUG();
    if ( (*(_BYTE *)(v80 - 2) & 0xFu) - 2 <= 1 )
    {
      v82 = sub_B326A0((__int64)(v80 - 6));
      v83 = v82;
      if ( v82 )
      {
        if ( v338 )
        {
          v84 = 1;
          v85 = 0;
          v86 = ((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4);
          v87 = (v338 - 1) & v86;
          v88 = &v336[4 * v87];
          v89 = *v88;
          if ( v83 == *v88 )
          {
LABEL_77:
            v90 = (_BYTE *)v88[2];
            v91 = (_BYTE *)v88[3];
            v92 = (__int64)(v88 + 1);
            v313.m128i_i64[0] = (__int64)(v80 - 6);
            if ( v91 != v90 )
            {
              if ( v90 )
              {
                *(_QWORD *)v90 = v80 - 6;
                v90 = (_BYTE *)v88[2];
              }
              v88[2] = (__int64)(v90 + 8);
              goto LABEL_81;
            }
LABEL_180:
            sub_E48660(v92, v90, &v313);
            goto LABEL_81;
          }
          while ( v89 != -4096 )
          {
            if ( v89 == -8192 && !v85 )
              v85 = v88;
            v87 = (v338 - 1) & (v84 + v87);
            v88 = &v336[4 * v87];
            v89 = *v88;
            if ( v83 == *v88 )
              goto LABEL_77;
            ++v84;
          }
          if ( v85 )
            v88 = v85;
          ++v335;
          v129 = v337 + 1;
          if ( 4 * ((int)v337 + 1) < 3 * v338 )
          {
            if ( v338 - HIDWORD(v337) - v129 <= v338 >> 3 )
            {
              sub_E487F0((__int64)&v335, v338);
              if ( !v338 )
              {
LABEL_576:
                LODWORD(v337) = v337 + 1;
                BUG();
              }
              v191 = 0;
              LODWORD(v192) = (v338 - 1) & v86;
              v129 = v337 + 1;
              v193 = 1;
              v88 = &v336[4 * (unsigned int)v192];
              v194 = *v88;
              if ( v83 != *v88 )
              {
                while ( v194 != -4096 )
                {
                  if ( !v191 && v194 == -8192 )
                    v191 = v88;
                  v192 = (v338 - 1) & ((_DWORD)v192 + v193);
                  v88 = &v336[4 * v192];
                  v194 = *v88;
                  if ( v83 == *v88 )
                    goto LABEL_177;
                  ++v193;
                }
                if ( v191 )
                  v88 = v191;
              }
            }
            goto LABEL_177;
          }
        }
        else
        {
          ++v335;
        }
        sub_E487F0((__int64)&v335, 2 * v338);
        if ( !v338 )
          goto LABEL_576;
        LODWORD(v179) = (v338 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
        v129 = v337 + 1;
        v88 = &v336[4 * (unsigned int)v179];
        v180 = *v88;
        if ( v83 != *v88 )
        {
          v181 = 1;
          v182 = 0;
          while ( v180 != -4096 )
          {
            if ( !v182 && v180 == -8192 )
              v182 = v88;
            v179 = (v338 - 1) & ((_DWORD)v179 + v181);
            v88 = &v336[4 * v179];
            v180 = *v88;
            if ( v83 == *v88 )
              goto LABEL_177;
            ++v181;
          }
          if ( v182 )
            v88 = v182;
        }
LABEL_177:
        LODWORD(v337) = v129;
        if ( *v88 != -4096 )
          --HIDWORD(v337);
        *v88 = v83;
        v92 = (__int64)(v88 + 1);
        v90 = 0;
        v88[1] = 0;
        v88[2] = 0;
        v88[3] = 0;
        v313.m128i_i64[0] = (__int64)(v80 - 6);
        goto LABEL_180;
      }
    }
LABEL_81:
    v80 = (_QWORD *)v80[1];
    if ( v81 != v80 )
      continue;
    break;
  }
  v57 = v317;
LABEL_83:
  v93 = v57 + 1;
  v295 = 0;
  v294 = v296;
  v94 = v57[2];
  if ( v57 + 1 != v94 )
  {
    do
    {
      v95 = (__int64)(v94 - 7);
      if ( !v94 )
        v95 = 0;
      if ( (unsigned __int8)sub_E496B0((__int64)&v316, v95, (__int64)&v294) )
        goto LABEL_89;
      v94 = (_QWORD *)v94[1];
    }
    while ( v93 != v94 );
    v57 = v317;
  }
  v127 = v57[4];
  v128 = v57 + 3;
  if ( v127 != v57 + 3 )
  {
    do
    {
      v95 = (__int64)(v127 - 7);
      if ( !v127 )
        v95 = 0;
      if ( (unsigned __int8)sub_E496B0((__int64)&v316, v95, (__int64)&v294) )
        goto LABEL_89;
      v127 = (_QWORD *)v127[1];
    }
    while ( v128 != v127 );
    v57 = v317;
  }
  v144 = v57[6];
  v145 = v57 + 5;
  if ( v57 + 5 != v144 )
  {
    do
    {
      v95 = (__int64)(v144 - 6);
      if ( !v144 )
        v95 = 0;
      if ( (unsigned __int8)sub_E496B0((__int64)&v316, v95, (__int64)&v294) )
        goto LABEL_89;
      v144 = (_QWORD *)v144[1];
    }
    while ( v145 != v144 );
    v57 = v317;
  }
  v148 = v57[8];
  for ( i = v57 + 7; i != v148; v148 = (_QWORD *)v148[1] )
  {
    v95 = (__int64)(v148 - 7);
    if ( !v148 )
      v95 = 0;
    if ( (unsigned __int8)sub_E496B0((__int64)&v316, v95, (__int64)&v294) )
      goto LABEL_89;
  }
  v195 = v294;
  v279 = &v294[(unsigned int)v295];
  if ( v294 != v279 )
  {
    do
    {
      v205 = (_BYTE *)*v195;
      if ( *(_BYTE *)*v195 == 3 )
      {
        v196 = v205[80];
        v197 = *((_QWORD *)v205 + 5);
        v198 = (_QWORD *)*((_QWORD *)v205 + 3);
        v199 = *((_QWORD *)v205 - 4);
        v200 = v205[32] & 0xF;
        BYTE4(v308) = 0;
        v266 = v200;
        v261 = v197;
        v263 = v198;
        v264 = v196 & 1;
        v269 = v199;
        LOWORD(v315) = 257;
        v201 = sub_BD2C40(88, unk_3F0FAE8);
        v202 = (__int64)v201;
        if ( v201 )
          sub_B30000((__int64)v201, v261, v263, v264, v266, v269, (__int64)&v313, 0, 0, v308, 0);
        sub_B32030(v202, (__int64)v205);
        v203 = *(_WORD *)(v202 + 32) & 0xBCC0 | 0x4008u;
        *(_WORD *)(v202 + 32) = v203;
        sub_B2F990(v202, *((_QWORD *)v205 + 6), v203, v204);
        if ( *((_QWORD *)v205 + 5) != *v316 )
        {
          v313.m128i_i64[0] = v202;
          sub_E49430((__int64)&v318, v313.m128i_i64);
        }
      }
      else
      {
        v206 = *v195;
        v308 = (__int64)"': non-variables in comdat nodeduplicate are not handled";
        v312 = 259;
        v207.m128i_i64[0] = (__int64)sub_BD5D20(v206);
        v306 = v207;
        v208 = v312;
        LOWORD(v307) = 1283;
        v305.m128i_i64[0] = (__int64)"linking '";
        if ( (_BYTE)v312 )
        {
          if ( (_BYTE)v312 == 1 )
          {
            v237 = _mm_loadu_si128(&v306);
            v313 = _mm_loadu_si128(&v305);
            v315 = v307;
            v314 = v237;
          }
          else
          {
            if ( HIBYTE(v312) == 1 )
            {
              v209 = (__int64 *)v308;
              v271 = v309;
            }
            else
            {
              v209 = &v308;
              v208 = 2;
            }
            v314.m128i_i64[0] = (__int64)v209;
            LOBYTE(v315) = 2;
            v313.m128i_i64[0] = (__int64)&v305;
            BYTE1(v315) = v208;
            v314.m128i_i64[1] = v271;
          }
        }
        else
        {
          LOWORD(v315) = 256;
        }
        v210 = (__int64)*v317;
        sub_1061A30(v296, 0, &v313);
        sub_B6EB20(v210, (__int64)v296);
      }
      ++v195;
    }
    while ( v279 != v195 );
  }
  v216 = 0;
  v217 = 0;
  v280 = 0;
  if ( (_DWORD)v323 )
  {
    while ( 1 )
    {
      v308 = sub_B326A0(v322[v216]);
      if ( v308 )
      {
        v218 = (__int64 **)sub_E48A20((__int64)&v335, &v308);
        v219 = v218[1];
        v220 = *v218;
        if ( *v218 != v219 )
          break;
      }
LABEL_374:
      ++v280;
      v217 = v323;
      v216 = v280;
      if ( (unsigned int)v323 <= v280 )
      {
        if ( !v329 )
          goto LABEL_431;
        v238 = v322;
        v281 = &v322[(unsigned int)v323];
        if ( v281 == v322 )
          goto LABEL_431;
        while ( 1 )
        {
          v239 = sub_BD5D20(*v238);
          v241 = v240;
          v242 = sub_C92610();
          v243 = sub_C92740((__int64)&v325, v239, v241, v242);
          v272 = (_QWORD *)(v325 + 8LL * v243);
          if ( !*v272 )
            goto LABEL_427;
          if ( *v272 == -8 )
            break;
LABEL_423:
          if ( v281 == ++v238 )
          {
            v217 = v323;
            goto LABEL_431;
          }
        }
        LODWORD(v327) = v327 - 1;
LABEL_427:
        v244 = sub_C7D670(v241 + 9, 8);
        v245 = (_QWORD *)v244;
        if ( v241 )
        {
          v270 = (_QWORD *)v244;
          memcpy((void *)(v244 + 8), v239, v241);
          v245 = v270;
        }
        *((_BYTE *)v245 + v241 + 8) = 0;
        *v245 = v241;
        *v272 = v245;
        ++HIDWORD(v326);
        sub_C929D0(&v325, v243);
        goto LABEL_423;
      }
    }
    while ( 1 )
    {
      v221 = *v220;
      v313.m128i_i64[0] = v221;
      if ( (*(_BYTE *)(v221 + 7) & 0x10) == 0 )
        break;
      if ( ((*(_BYTE *)(v221 + 32) + 9) & 0xFu) <= 1 )
        break;
      v222 = *v316;
      v223 = sub_BD5D20(v221);
      v225 = sub_BA8B30(v222, (__int64)v223, v224);
      if ( !v225 || (*(_BYTE *)(v225 + 32) & 0xFu) - 7 <= 1 )
        break;
      v305.m128i_i8[0] = 1;
      if ( (v324 & 1) != 0 )
        goto LABEL_379;
      v95 = (__int64)&v305;
      v226 = sub_E48260((__int64)&v316, (bool *)v305.m128i_i8, v225, v313.m128i_i64[0]);
      if ( (_BYTE)v226 )
      {
        v96 = v226;
        goto LABEL_90;
      }
      if ( v305.m128i_i8[0] )
        goto LABEL_379;
LABEL_380:
      if ( v219 == ++v220 )
        goto LABEL_374;
    }
    v305.m128i_i8[0] = 1;
LABEL_379:
    sub_E49430((__int64)&v318, v313.m128i_i64);
    goto LABEL_380;
  }
LABEL_431:
  v95 = (__int64)v316;
  v314.m128i_i64[1] = (__int64)&off_4C5D168 + 2;
  v288 = 0;
  v308 = (__int64)v317;
  v313.m128i_i64[0] = (__int64)&v316;
  v317 = 0;
  sub_106AB30(
    (unsigned int)&v289,
    (_DWORD)v316,
    (unsigned int)&v308,
    (_DWORD)v322,
    v217,
    (unsigned int)&v313,
    0,
    (v324 >> 2) & 1);
  v246 = v308;
  if ( v308 )
  {
    sub_BA9C10((_QWORD **)v308, v95, v259, v260);
    v95 = 880;
    j_j___libc_free_0(v246, 880);
  }
  if ( (v314.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v247 = (v314.m128i_i64[1] >> 1) & 1;
    if ( (v314.m128i_i8[8] & 4) != 0 )
    {
      v248 = &v313;
      if ( !(_BYTE)v247 )
        v248 = (__m128i *)v313.m128i_i64[0];
      (*(void (__fastcall **)(__m128i *))((v314.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16))(v248);
    }
    if ( !(_BYTE)v247 )
    {
      v95 = v313.m128i_i64[1];
      sub_C7D6A0(v313.m128i_i64[0], v313.m128i_i64[1], v314.m128i_i64[0]);
    }
  }
  v249 = (_QWORD **)(v289 & 0xFFFFFFFFFFFFFFFELL);
  if ( (v289 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v250 = v289 & 0xFFFFFFFFFFFFFFFELL;
    v289 = 0;
    v290 = 0;
    v95 = (__int64)&unk_4F84052;
    v313.m128i_i64[0] = v273;
    v313.m128i_i64[1] = (__int64)&v288;
    v291 = 0;
    if ( ((unsigned __int8 (__fastcall *)(unsigned __int64, void *))(*v249)[6])(v250, &unk_4F84052) )
    {
      v251 = v249[2];
      v252 = v249[1];
      v292 = 1;
      v282 = v251;
      if ( v252 != v251 )
      {
        v253 = v252;
        do
        {
          v296[0] = *v253;
          *v253 = 0;
          sub_E47F60(v305.m128i_i64, v296, (__int64)&v313);
          v254 = v292;
          v95 = (__int64)&v293;
          v292 = 0;
          v293 = v254 | 1;
          sub_9CDB40((unsigned __int64 *)&v308, (unsigned __int64 *)&v293, (unsigned __int64 *)&v305);
          if ( (v292 & 1) != 0 || (v292 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v292, (__int64)&v293);
          v292 |= v308 | 1;
          if ( (v293 & 1) != 0 || (v293 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v293, (__int64)&v293);
          if ( (v305.m128i_i8[0] & 1) != 0 || (v305.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_C63C30(&v305, (__int64)&v293);
          if ( v296[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v296[0] + 8LL))(v296[0]);
          ++v253;
        }
        while ( v282 != v253 );
      }
      v305.m128i_i64[0] = v292 | 1;
      ((void (__fastcall *)(_QWORD **))(*v249)[1])(v249);
    }
    else
    {
      v95 = (__int64)&v308;
      v308 = (__int64)v249;
      sub_E47F60(v305.m128i_i64, &v308, (__int64)&v313);
      if ( v308 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v308 + 8LL))(v308);
    }
    if ( (v305.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v291 & 1) != 0 || (v291 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v291, v95);
    if ( (v290 & 1) != 0 || (v290 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v290, v95);
    if ( (v289 & 1) != 0 || (v289 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v289, v95);
  }
  v96 = v288;
  if ( v288 )
  {
LABEL_89:
    v96 = 1;
    goto LABEL_90;
  }
  if ( v329 )
  {
    v95 = v273;
    v330(&v328, v273, &v325);
  }
LABEL_90:
  if ( v294 != v296 )
    _libc_free(v294, v95);
LABEL_92:
  sub_C7D6A0((__int64)v302, 8LL * (unsigned int)v304, 8);
  sub_C7D6A0(v298, 8LL * (unsigned int)v300, 8);
  v97 = v338;
  if ( v338 )
  {
    v98 = v336;
    v99 = &v336[4 * v338];
    do
    {
      if ( *v98 != -8192 && *v98 != -4096 )
      {
        v100 = v98[1];
        if ( v100 )
          j_j___libc_free_0(v100, v98[3] - v100);
      }
      v98 += 4;
    }
    while ( v99 != v98 );
    v97 = v338;
  }
  sub_C7D6A0((__int64)v336, 32LL * v97, 8);
  v101 = (__m128i *)(16LL * v334);
  sub_C7D6A0(v332, (__int64)v101, 8);
  if ( v329 )
  {
    v101 = &v328;
    v329(&v328, &v328, 3);
  }
  v102 = v325;
  if ( HIDWORD(v326) && (_DWORD)v326 )
  {
    v103 = 8LL * (unsigned int)v326;
    v104 = 0;
    do
    {
      v105 = *(_QWORD **)(v102 + v104);
      if ( v105 != (_QWORD *)-8LL && v105 )
      {
        v101 = (__m128i *)(*v105 + 9LL);
        sub_C7D6A0((__int64)v105, (__int64)v101, 8);
        v102 = v325;
      }
      v104 += 8;
    }
    while ( v104 != v103 );
  }
  _libc_free(v102, v101);
  if ( v322 != (__int64 *)&v324 )
    _libc_free(v322, v101);
  v106 = 8LL * (unsigned int)v321;
  sub_C7D6A0(v319, v106, 8);
  v109 = v317;
  if ( v317 )
  {
    sub_BA9C10(v317, v106, v107, v108);
    j_j___libc_free_0(v109, 880);
  }
  return v96;
}
