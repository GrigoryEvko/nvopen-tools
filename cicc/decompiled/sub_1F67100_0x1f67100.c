// Function: sub_1F67100
// Address: 0x1f67100
//
__int64 **__fastcall sub_1F67100(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 **result; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // r8
  _QWORD *v15; // rdx
  char v16; // cl
  __int64 *v17; // r14
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 *v20; // rax
  _BYTE *v21; // r9
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned int v24; // esi
  int v25; // r9d
  int v26; // r9d
  __int64 v27; // r10
  int v28; // ecx
  unsigned int v29; // esi
  __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rax
  _QWORD *v36; // r9
  int v37; // ecx
  unsigned int v38; // edx
  _QWORD *v39; // rbx
  __int64 v40; // rsi
  int v41; // edx
  __int64 v42; // rdx
  unsigned __int64 *v43; // rdi
  __int64 *v44; // rdx
  __int64 v45; // rax
  const __m128i *v46; // rsi
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 *v49; // rsi
  unsigned int v50; // esi
  __int64 v51; // rax
  __int64 v52; // r10
  __int64 v53; // rdx
  __int64 *v54; // r12
  __int64 v55; // r9
  __int64 v56; // rax
  unsigned __int64 v57; // r15
  __int64 v58; // rax
  int v59; // r9d
  __int64 v60; // rdx
  unsigned __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 *v64; // rdx
  __int64 *v65; // rcx
  __int64 *v66; // r12
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 *v70; // rax
  __int64 *v71; // rax
  __int64 *v72; // rax
  __int64 *v73; // rcx
  unsigned int v74; // esi
  __int64 v75; // r10
  __int64 v76; // r9
  __int64 *v77; // rax
  __int64 v78; // rdi
  __int64 v79; // r10
  unsigned __int64 v80; // rdx
  __int64 *v81; // rcx
  __int64 v82; // rsi
  __int64 *v83; // r9
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 *v86; // rdx
  __int64 v87; // rbx
  unsigned __int64 v88; // rbx
  __int64 v89; // rdx
  __int64 v90; // r12
  __int64 *v91; // rax
  __int64 v92; // r15
  __int64 m; // r12
  __int64 v94; // r13
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // r8
  int v98; // r9d
  double v99; // xmm4_8
  double v100; // xmm5_8
  const __m128i *v101; // r15
  const __m128i *v102; // r13
  __int64 v103; // rax
  __int64 v104; // rbx
  __int64 n; // r14
  _QWORD *v106; // rax
  __int64 v107; // r12
  unsigned __int64 v108; // rax
  int v109; // r8d
  int v110; // r9d
  __int64 v111; // rdx
  __int64 *v112; // rdx
  __int64 *v113; // rsi
  __int64 v114; // rax
  __int64 v115; // rdi
  unsigned __int64 v116; // rcx
  __int64 v117; // rcx
  __int64 v118; // rcx
  __int64 v119; // rax
  const __m128i *v120; // r13
  __int64 v121; // r14
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // r12
  __int64 v125; // rbx
  __int64 v126; // rbx
  __int64 v127; // rdx
  __int64 v128; // r8
  int v129; // edi
  _QWORD *v130; // rcx
  int v131; // ecx
  unsigned int v132; // edx
  __int64 v133; // rsi
  __int64 v134; // rax
  __int64 v135; // rdx
  __int64 v136; // r12
  __int64 v137; // rbx
  __int64 v138; // rbx
  unsigned __int64 v139; // rdi
  __int64 v140; // r12
  unsigned int ii; // r15d
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rdx
  __int64 v145; // r14
  __int64 v146; // rbx
  unsigned int v147; // eax
  int v148; // edx
  __int64 v149; // r9
  __int64 v150; // rdi
  __int64 v151; // rsi
  __int64 v152; // rcx
  __int64 v153; // r8
  __int64 v154; // rcx
  __int64 v155; // r13
  __int64 v156; // r10
  __int64 v157; // rdx
  unsigned int v158; // ecx
  int v159; // eax
  __int64 v160; // rdx
  _QWORD *v161; // rax
  __int64 v162; // rcx
  unsigned __int64 v163; // rdx
  __int64 v164; // rdx
  __int64 v165; // rdx
  __int64 v166; // rcx
  __int64 v167; // rax
  _QWORD *v168; // r14
  _QWORD *v169; // rbx
  __int64 v170; // rdx
  __int64 v171; // rax
  __int64 v172; // rdx
  __int64 v173; // rax
  int v174; // edi
  __int64 *v175; // rcx
  int v176; // edi
  int v177; // edx
  __int64 v178; // rax
  __int64 v179; // rsi
  __int64 *v180; // rdx
  int v181; // r9d
  int v182; // r9d
  __int64 v183; // r11
  __int64 v184; // rcx
  __int64 v185; // r10
  int v186; // edi
  __int64 *v187; // rsi
  int v188; // r9d
  int v189; // r9d
  __int64 v190; // r10
  unsigned int v191; // ecx
  int v192; // edx
  __int64 v193; // r12
  int v194; // edi
  __int64 *v195; // rsi
  int v196; // ecx
  __int64 *v197; // rdx
  int v198; // edi
  int v199; // r9d
  int v200; // r9d
  __int64 v201; // r10
  __int64 *v202; // rcx
  unsigned int v203; // r15d
  int v204; // esi
  __int64 v205; // rdi
  _QWORD *v206; // rbx
  _QWORD *v207; // r12
  __int64 v208; // rsi
  _QWORD *v209; // r13
  unsigned __int64 v210; // rax
  _QWORD *v211; // r14
  __int64 v212; // rax
  __int64 v213; // rax
  __int64 v214; // rdi
  __int64 v215; // rbx
  int v216; // r8d
  __int64 v217; // rdi
  unsigned int v218; // ecx
  _QWORD *v219; // rax
  __int64 v220; // rdx
  __int64 v221; // rcx
  unsigned __int64 v222; // rdx
  __int64 v223; // rax
  _QWORD *v224; // rax
  _QWORD *v225; // r9
  unsigned int v226; // esi
  __int64 v227; // r14
  int v228; // r8d
  __int64 v229; // rcx
  int v230; // edx
  __int64 v231; // r11
  _QWORD *v232; // rax
  __int64 v233; // rax
  int v234; // r10d
  int v235; // edi
  int v236; // r8d
  _QWORD *v237; // rcx
  __int64 v238; // r15
  int v239; // esi
  __int64 v240; // rdi
  char *v241; // rax
  size_t v242; // rdx
  int jj; // eax
  __int64 *v244; // rsi
  _QWORD *v245; // r14
  _QWORD *v246; // rbx
  __int64 v247; // rdx
  __int64 v248; // rax
  __int64 v249; // rdx
  int v250; // r9d
  int v251; // r9d
  __int64 v252; // r11
  int v253; // edi
  __int64 v254; // rcx
  __int64 v255; // r10
  __int64 *v256; // rdi
  int v257; // edi
  int v258; // r9d
  int v259; // r9d
  __int64 *v260; // r11
  __int64 v261; // r10
  int v262; // edi
  unsigned int v263; // esi
  __int64 v264; // r8
  __int64 *v265; // rax
  _QWORD *v266; // rbx
  _QWORD *v267; // r12
  __int64 v268; // rsi
  int v269; // edi
  _QWORD *v270; // rsi
  int v271; // edi
  unsigned int v272; // [rsp+0h] [rbp-240h]
  __int64 **v273; // [rsp+8h] [rbp-238h]
  const __m128i *v274; // [rsp+18h] [rbp-228h]
  __int64 v276; // [rsp+28h] [rbp-218h]
  __int64 **i; // [rsp+30h] [rbp-210h]
  const __m128i *v278; // [rsp+38h] [rbp-208h]
  _QWORD *v279; // [rsp+38h] [rbp-208h]
  unsigned __int64 v280; // [rsp+38h] [rbp-208h]
  _QWORD *v282; // [rsp+48h] [rbp-1F8h]
  const __m128i *v283; // [rsp+48h] [rbp-1F8h]
  unsigned __int64 v284; // [rsp+48h] [rbp-1F8h]
  __int64 *v285; // [rsp+48h] [rbp-1F8h]
  __int64 *v286; // [rsp+50h] [rbp-1F0h]
  __int64 v287; // [rsp+50h] [rbp-1F0h]
  __int64 *v288; // [rsp+50h] [rbp-1F0h]
  int v289; // [rsp+50h] [rbp-1F0h]
  unsigned __int64 v290; // [rsp+50h] [rbp-1F0h]
  __int64 v291; // [rsp+58h] [rbp-1E8h]
  const __m128i *v292; // [rsp+58h] [rbp-1E8h]
  __int64 *k; // [rsp+58h] [rbp-1E8h]
  const __m128i *v294; // [rsp+58h] [rbp-1E8h]
  __int64 v295; // [rsp+58h] [rbp-1E8h]
  __int64 v296; // [rsp+58h] [rbp-1E8h]
  __int64 v297; // [rsp+60h] [rbp-1E0h] BYREF
  __int64 v298; // [rsp+68h] [rbp-1D8h] BYREF
  const __m128i *v299; // [rsp+70h] [rbp-1D0h] BYREF
  const __m128i *v300; // [rsp+78h] [rbp-1C8h]
  const __m128i *v301; // [rsp+80h] [rbp-1C0h]
  __int64 *v302; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v303; // [rsp+98h] [rbp-1A8h]
  __int64 *v304; // [rsp+A0h] [rbp-1A0h]
  __int64 *v305; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v306; // [rsp+B8h] [rbp-188h]
  _BYTE v307[16]; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v308; // [rsp+D0h] [rbp-170h] BYREF
  unsigned __int64 v309[2]; // [rsp+D8h] [rbp-168h] BYREF
  __int64 v310; // [rsp+E8h] [rbp-158h]
  const char *v311; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v312; // [rsp+F8h] [rbp-148h] BYREF
  __int64 v313; // [rsp+100h] [rbp-140h]
  __int64 v314; // [rsp+108h] [rbp-138h]
  __int64 v315; // [rsp+110h] [rbp-130h]
  __int64 v316; // [rsp+130h] [rbp-110h] BYREF
  _QWORD *v317; // [rsp+138h] [rbp-108h]
  __int64 v318; // [rsp+140h] [rbp-100h]
  unsigned int v319; // [rsp+148h] [rbp-F8h]
  _QWORD *v320; // [rsp+158h] [rbp-E8h]
  unsigned int v321; // [rsp+168h] [rbp-D8h]
  char v322; // [rsp+170h] [rbp-D0h]
  char v323; // [rsp+179h] [rbp-C7h]
  const char *v324; // [rsp+180h] [rbp-C0h] BYREF
  __int64 v325; // [rsp+188h] [rbp-B8h] BYREF
  __int64 v326; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v327; // [rsp+198h] [rbp-A8h]
  __int64 *j; // [rsp+1A0h] [rbp-A0h]

  result = *(__int64 ***)(a1 + 232);
  v273 = *(__int64 ***)(a1 + 240);
  if ( result != v273 )
  {
    for ( i = result + 1; ; i += 4 )
    {
      v11 = (__int64)*(i - 1);
      v12 = *(_QWORD *)(a2 + 80);
      v297 = v11;
      if ( v12 )
        v12 -= 24;
      if ( v11 == v12 )
      {
        v265 = (__int64 *)sub_15E0530(a2);
        v298 = sub_1594470(v265);
      }
      else
      {
        v298 = sub_157ED20(v11);
      }
      v299 = 0;
      v300 = 0;
      v301 = 0;
      v316 = 0;
      v319 = 128;
      v13 = (_QWORD *)sub_22077B0(0x2000);
      v318 = 0;
      v317 = v13;
      v325 = 2;
      v15 = &v13[8 * (unsigned __int64)v319];
      v324 = (const char *)&unk_49E6B50;
      v326 = 0;
      v327 = -8;
      for ( j = 0; v15 != v13; v13 += 8 )
      {
        if ( v13 )
        {
          v16 = v325;
          v13[2] = 0;
          v13[3] = -8;
          *v13 = &unk_49E6B50;
          v13[1] = v16 & 6;
          v13[4] = j;
        }
      }
      v322 = 0;
      v323 = 1;
      v17 = *i;
      v286 = i[1];
      if ( *i != v286 )
      {
        v291 = a1 + 168;
        while ( 1 )
        {
          v23 = *v17;
          v24 = *(_DWORD *)(a1 + 192);
          v305 = (__int64 *)*v17;
          if ( !v24 )
            break;
          v18 = *(_QWORD *)(a1 + 176);
          v19 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v20 = (__int64 *)(v18 + 16LL * v19);
          v21 = (_BYTE *)*v20;
          if ( v23 != *v20 )
          {
            LODWORD(v14) = 1;
            v256 = 0;
            while ( v21 != (_BYTE *)-8LL )
            {
              if ( v21 == (_BYTE *)-16LL && !v256 )
                v256 = v20;
              v19 = (v24 - 1) & ((_DWORD)v14 + v19);
              v20 = (__int64 *)(v18 + 16LL * v19);
              v21 = (_BYTE *)*v20;
              if ( v23 == *v20 )
                goto LABEL_14;
              LODWORD(v14) = (_DWORD)v14 + 1;
            }
            if ( v256 )
              v20 = v256;
            v257 = *(_DWORD *)(a1 + 184);
            ++*(_QWORD *)(a1 + 168);
            v28 = v257 + 1;
            if ( 4 * (v257 + 1) < 3 * v24 )
            {
              if ( v24 - *(_DWORD *)(a1 + 188) - v28 <= v24 >> 3 )
              {
                sub_14DDDA0(v291, v24);
                v258 = *(_DWORD *)(a1 + 192);
                if ( !v258 )
                {
LABEL_541:
                  ++*(_DWORD *)(a1 + 184);
                  BUG();
                }
                v23 = (__int64)v305;
                v259 = v258 - 1;
                v260 = 0;
                v261 = *(_QWORD *)(a1 + 176);
                v28 = *(_DWORD *)(a1 + 184) + 1;
                v262 = 1;
                v263 = v259 & (((unsigned int)v305 >> 9) ^ ((unsigned int)v305 >> 4));
                v20 = (__int64 *)(v261 + 16LL * v263);
                v264 = *v20;
                if ( v305 != (__int64 *)*v20 )
                {
                  while ( v264 != -8 )
                  {
                    if ( v264 == -16 && !v260 )
                      v260 = v20;
                    v263 = v259 & (v262 + v263);
                    v20 = (__int64 *)(v261 + 16LL * v263);
                    v264 = *v20;
                    if ( v305 == (__int64 *)*v20 )
                      goto LABEL_22;
                    ++v262;
                  }
LABEL_440:
                  if ( v260 )
                    v20 = v260;
                }
              }
LABEL_22:
              *(_DWORD *)(a1 + 184) = v28;
              if ( *v20 != -8 )
                --*(_DWORD *)(a1 + 188);
              *v20 = v23;
              v20[1] = 0;
              goto LABEL_25;
            }
LABEL_20:
            sub_14DDDA0(v291, 2 * v24);
            v25 = *(_DWORD *)(a1 + 192);
            if ( !v25 )
              goto LABEL_541;
            v23 = (__int64)v305;
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 176);
            v28 = *(_DWORD *)(a1 + 184) + 1;
            v29 = v26 & (((unsigned int)v305 >> 9) ^ ((unsigned int)v305 >> 4));
            v20 = (__int64 *)(v27 + 16LL * v29);
            v30 = *v20;
            if ( v305 != (__int64 *)*v20 )
            {
              v271 = 1;
              v260 = 0;
              while ( v30 != -8 )
              {
                if ( v30 == -16 && !v260 )
                  v260 = v20;
                v29 = v26 & (v271 + v29);
                v20 = (__int64 *)(v27 + 16LL * v29);
                v30 = *v20;
                if ( v305 == (__int64 *)*v20 )
                  goto LABEL_22;
                ++v271;
              }
              goto LABEL_440;
            }
            goto LABEL_22;
          }
LABEL_14:
          v22 = v20[1];
          if ( (v22 & 0xFFFFFFFFFFFFFFF8LL) != 0
            && ((v22 & 4) == 0 || *(_DWORD *)((v22 & 0xFFFFFFFFFFFFFFF8LL) + 8) == 1) )
          {
            goto LABEL_17;
          }
LABEL_25:
          v311 = sub_1649960(v297);
          v312 = v31;
          LOWORD(v326) = 1283;
          v324 = ".for.";
          v325 = (__int64)&v311;
          v308 = sub_1AB5760((__int64)v305, (__int64)&v316, (__int64 *)&v324, 0, 0, 0);
          v32 = v305[4];
          if ( v32 == v305[7] + 72 || !v32 )
            v33 = 0;
          else
            v33 = v32 - 24;
          sub_157FA80(v308, a2, v33);
          v325 = 2;
          v326 = 0;
          v34 = v308;
          v327 = (__int64)v305;
          if ( v305 != 0 && v305 + 1 != 0 && v305 != (__int64 *)-16LL )
            sub_164C220((__int64)&v325);
          j = &v316;
          v324 = (const char *)&unk_49E6B50;
          if ( !v319 )
          {
            ++v316;
            goto LABEL_33;
          }
          v35 = v327;
          v127 = (v319 - 1) & (((unsigned int)v327 >> 9) ^ ((unsigned int)v327 >> 4));
          v39 = &v317[8 * v127];
          v128 = v39[3];
          if ( v327 != v128 )
          {
            v129 = 1;
            v130 = 0;
            while ( v128 != -8 )
            {
              if ( v128 == -16 && !v130 )
                v130 = v39;
              LODWORD(v127) = (v319 - 1) & (v129 + v127);
              v39 = &v317[8 * (unsigned __int64)(unsigned int)v127];
              v128 = v39[3];
              if ( v327 == v128 )
                goto LABEL_46;
              ++v129;
            }
            if ( v130 )
              v39 = v130;
            ++v316;
            v41 = v318 + 1;
            if ( 4 * ((int)v318 + 1) >= 3 * v319 )
            {
LABEL_33:
              sub_12E48B0((__int64)&v316, 2 * v319);
              if ( !v319 )
                goto LABEL_430;
              v35 = v327;
              v36 = 0;
              v37 = 1;
              v38 = (v319 - 1) & (((unsigned int)v327 >> 9) ^ ((unsigned int)v327 >> 4));
              v39 = &v317[8 * (unsigned __int64)v38];
              v40 = v39[3];
              if ( v327 != v40 )
              {
                while ( v40 != -8 )
                {
                  if ( v40 == -16 && !v36 )
                    v36 = v39;
                  v38 = (v319 - 1) & (v37 + v38);
                  v39 = &v317[8 * (unsigned __int64)v38];
                  v40 = v39[3];
                  if ( v327 == v40 )
                    goto LABEL_35;
                  ++v37;
                }
                goto LABEL_166;
              }
LABEL_35:
              v41 = v318 + 1;
            }
            else if ( v319 - HIDWORD(v318) - v41 <= v319 >> 3 )
            {
              sub_12E48B0((__int64)&v316, v319);
              if ( v319 )
              {
                v35 = v327;
                v36 = 0;
                v131 = 1;
                v132 = (v319 - 1) & (((unsigned int)v327 >> 9) ^ ((unsigned int)v327 >> 4));
                v39 = &v317[8 * (unsigned __int64)v132];
                v133 = v39[3];
                if ( v327 != v133 )
                {
                  while ( v133 != -8 )
                  {
                    if ( v133 == -16 && !v36 )
                      v36 = v39;
                    v132 = (v319 - 1) & (v131 + v132);
                    v39 = &v317[8 * (unsigned __int64)v132];
                    v133 = v39[3];
                    if ( v327 == v133 )
                      goto LABEL_35;
                    ++v131;
                  }
LABEL_166:
                  if ( v36 )
                    v39 = v36;
                }
                goto LABEL_35;
              }
LABEL_430:
              v35 = v327;
              v39 = 0;
              goto LABEL_35;
            }
            LODWORD(v318) = v41;
            v42 = v39[3];
            v43 = v39 + 1;
            if ( v42 == -8 )
            {
              if ( v35 != -8 )
                goto LABEL_41;
            }
            else
            {
              --HIDWORD(v318);
              if ( v42 != v35 )
              {
                if ( v42 && v42 != -16 )
                {
                  sub_1649B30(v43);
                  v35 = v327;
                  v43 = v39 + 1;
                }
LABEL_41:
                v39[3] = v35;
                if ( v35 != 0 && v35 != -8 && v35 != -16 )
                  sub_1649AC0(v43, v325 & 0xFFFFFFFFFFFFFFF8LL);
                v35 = v327;
              }
            }
            v44 = j;
            v39[5] = 6;
            v39[6] = 0;
            v39[4] = v44;
            v39[7] = 0;
          }
LABEL_46:
          v14 = v39 + 5;
          v324 = (const char *)&unk_49EE2B0;
          if ( v35 != -8 && v35 != 0 && v35 != -16 )
          {
            sub_1649B30(&v325);
            v14 = v39 + 5;
          }
          v45 = v39[7];
          if ( v34 != v45 )
          {
            if ( v45 != 0 && v45 != -8 && v45 != -16 )
            {
              v282 = v14;
              sub_1649B30(v14);
              v14 = v282;
            }
            v39[7] = v34;
            if ( v34 != 0 && v34 != -8 && v34 != -16 )
              sub_164C220((__int64)v14);
          }
          v46 = v300;
          if ( v300 == v301 )
          {
            sub_1F60D40(&v299, v300, &v305, &v308);
LABEL_17:
            if ( v286 == ++v17 )
              goto LABEL_60;
          }
          else
          {
            if ( v300 )
            {
              v300->m128i_i64[0] = (__int64)v305;
              v46->m128i_i64[1] = v308;
              v46 = v300;
            }
            ++v17;
            v300 = v46 + 1;
            if ( v286 == v17 )
              goto LABEL_60;
          }
        }
        ++*(_QWORD *)(a1 + 168);
        goto LABEL_20;
      }
LABEL_60:
      v283 = v300;
      if ( v299 != v300 )
        break;
      if ( v322 )
      {
        if ( v321 )
        {
          v266 = v320;
          v267 = &v320[2 * v321];
          do
          {
            if ( *v266 != -8 && *v266 != -4 )
            {
              v268 = v266[1];
              if ( v268 )
                sub_161E7C0((__int64)(v266 + 1), v268);
            }
            v266 += 2;
          }
          while ( v267 != v266 );
        }
        j___libc_free_0(v320);
      }
      if ( v319 )
      {
        v245 = v317;
        v312 = 2;
        v313 = 0;
        v246 = &v317[8 * (unsigned __int64)v319];
        v314 = -8;
        v311 = (const char *)&unk_49E6B50;
        v324 = (const char *)&unk_49E6B50;
        v247 = -8;
        v315 = 0;
        v325 = 2;
        v326 = 0;
        v327 = -16;
        j = 0;
        while ( 1 )
        {
          v248 = v245[3];
          if ( v247 != v248 && v248 != v327 )
          {
            v249 = v245[7];
            if ( v249 != -8 && v249 != 0 && v249 != -16 )
            {
              sub_1649B30(v245 + 5);
              v248 = v245[3];
            }
          }
          *v245 = &unk_49EE2B0;
          if ( v248 != 0 && v248 != -8 && v248 != -16 )
            sub_1649B30(v245 + 1);
          v245 += 8;
          if ( v246 == v245 )
            break;
          v247 = v314;
        }
        v324 = (const char *)&unk_49EE2B0;
        v173 = v327;
        if ( v327 != -8 && v327 != 0 )
        {
LABEL_239:
          if ( v173 != -16 )
            sub_1649B30(&v325);
        }
LABEL_241:
        v311 = (const char *)&unk_49EE2B0;
        if ( v314 != 0 && v314 != -8 && v314 != -16 )
          sub_1649B30(&v312);
      }
LABEL_244:
      j___libc_free_0(v317);
      if ( v299 )
        j_j___libc_free_0(v299, (char *)v301 - (char *)v299);
      result = i + 3;
      if ( v273 == i + 3 )
        return result;
    }
    v292 = v299;
    v276 = a1 + 168;
    while ( 1 )
    {
      v47 = v292->m128i_i64[0];
      v48 = v292->m128i_i64[1];
      v324 = (const char *)v48;
      v49 = i[1];
      if ( v49 == i[2] )
      {
        sub_1292090((__int64)i, v49, &v324);
        v50 = *(_DWORD *)(a1 + 192);
        if ( !v50 )
          goto LABEL_278;
      }
      else
      {
        if ( v49 )
        {
          *v49 = v48;
          v49 = i[1];
        }
        i[1] = v49 + 1;
        v50 = *(_DWORD *)(a1 + 192);
        if ( !v50 )
        {
LABEL_278:
          ++*(_QWORD *)(a1 + 168);
          goto LABEL_279;
        }
      }
      v51 = (__int64)v324;
      v52 = *(_QWORD *)(a1 + 176);
      v53 = (v50 - 1) & (((unsigned int)v324 >> 9) ^ ((unsigned int)v324 >> 4));
      v54 = (__int64 *)(v52 + 16 * v53);
      v55 = *v54;
      if ( v324 != (const char *)*v54 )
        break;
LABEL_67:
      v287 = v297;
      v56 = v54[1];
      v57 = v56 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v56 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (v56 & 4) == 0 )
        {
          v58 = sub_22077B0(48);
          if ( v58 )
          {
            *(_QWORD *)v58 = v58 + 16;
            *(_QWORD *)(v58 + 8) = 0x400000000LL;
          }
          v60 = v58;
          v61 = v58 & 0xFFFFFFFFFFFFFFF8LL;
          v54[1] = v60 | 4;
          v62 = *(unsigned int *)(v61 + 8);
          if ( (unsigned int)v62 >= *(_DWORD *)(v61 + 12) )
          {
            v280 = v61;
            sub_16CD150(v61, (const void *)(v61 + 16), 0, 8, (int)v14, v59);
            v61 = v280;
            v62 = *(unsigned int *)(v280 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v61 + 8 * v62) = v57;
          ++*(_DWORD *)(v61 + 8);
          v55 = v54[1];
          v57 = v55 & 0xFFFFFFFFFFFFFFF8LL;
        }
        v63 = *(unsigned int *)(v57 + 8);
        if ( (unsigned int)v63 >= *(_DWORD *)(v57 + 12) )
        {
          sub_16CD150(v57, (const void *)(v57 + 16), 0, 8, (int)v14, v55);
          v63 = *(unsigned int *)(v57 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v57 + 8 * v63) = v287;
        ++*(_DWORD *)(v57 + 8);
        goto LABEL_77;
      }
LABEL_266:
      v54[1] = v287;
LABEL_77:
      v64 = i[1];
      v65 = *i;
      v66 = v64;
      v67 = (char *)v64 - (char *)*i;
      v68 = v67 >> 5;
      v69 = v67 >> 3;
      if ( v68 <= 0 )
        goto LABEL_289;
      v70 = &v65[4 * v68];
      do
      {
        if ( v47 == *v65 )
          goto LABEL_84;
        if ( v47 == v65[1] )
        {
          ++v65;
          goto LABEL_84;
        }
        if ( v47 == v65[2] )
        {
          v65 += 2;
          goto LABEL_84;
        }
        if ( v47 == v65[3] )
        {
          v65 += 3;
          goto LABEL_84;
        }
        v65 += 4;
      }
      while ( v70 != v65 );
      v69 = v64 - v65;
LABEL_289:
      if ( v69 != 2 )
      {
        if ( v69 != 3 )
        {
          if ( v69 == 1 )
          {
            if ( v47 != *v65 )
              goto LABEL_293;
LABEL_84:
            if ( v65 != v64 )
            {
              v71 = v65 + 1;
              if ( v64 == v65 + 1 )
                goto LABEL_91;
              do
              {
                if ( v47 != *v71 )
                  *v65++ = *v71;
                ++v71;
              }
              while ( v64 != v71 );
              if ( v64 != v65 )
              {
                v64 = i[1];
LABEL_91:
                if ( v66 != v64 )
                {
                  v72 = (__int64 *)memmove(v65, v66, (char *)v64 - (char *)v66);
                  v64 = i[1];
                  v65 = v72;
                }
                v73 = (__int64 *)((char *)v65 + (char *)v64 - (char *)v66);
                if ( v73 != v64 )
                  i[1] = v73;
              }
            }
          }
          v74 = *(_DWORD *)(a1 + 192);
          if ( !v74 )
            goto LABEL_294;
          goto LABEL_96;
        }
        if ( v47 == *v65 )
          goto LABEL_84;
        ++v65;
      }
      if ( v47 == *v65 )
        goto LABEL_84;
      if ( v47 == *++v65 )
        goto LABEL_84;
LABEL_293:
      v74 = *(_DWORD *)(a1 + 192);
      if ( !v74 )
      {
LABEL_294:
        ++*(_QWORD *)(a1 + 168);
        goto LABEL_295;
      }
LABEL_96:
      v75 = *(_QWORD *)(a1 + 176);
      LODWORD(v76) = (v74 - 1) & (((unsigned int)v47 >> 4) ^ ((unsigned int)v47 >> 9));
      v77 = (__int64 *)(v75 + 16LL * (unsigned int)v76);
      v78 = *v77;
      if ( v47 != *v77 )
      {
        v196 = 1;
        v197 = 0;
        while ( v78 != -8 )
        {
          if ( !v197 && v78 == -16 )
            v197 = v77;
          LODWORD(v14) = v196 + 1;
          v76 = (v74 - 1) & ((_DWORD)v76 + v196);
          v77 = (__int64 *)(v75 + 16 * v76);
          v78 = *v77;
          if ( v47 == *v77 )
            goto LABEL_97;
          ++v196;
        }
        v198 = *(_DWORD *)(a1 + 184);
        if ( v197 )
          v77 = v197;
        ++*(_QWORD *)(a1 + 168);
        v192 = v198 + 1;
        if ( 4 * (v198 + 1) >= 3 * v74 )
        {
LABEL_295:
          sub_14DDDA0(v276, 2 * v74);
          v188 = *(_DWORD *)(a1 + 192);
          if ( !v188 )
            goto LABEL_536;
          v189 = v188 - 1;
          v190 = *(_QWORD *)(a1 + 176);
          v191 = v189 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v192 = *(_DWORD *)(a1 + 184) + 1;
          v77 = (__int64 *)(v190 + 16LL * v191);
          v193 = *v77;
          if ( v47 != *v77 )
          {
            v194 = 1;
            v195 = 0;
            while ( v193 != -8 )
            {
              if ( v193 == -16 && !v195 )
                v195 = v77;
              LODWORD(v14) = v194 + 1;
              v191 = v189 & (v194 + v191);
              v77 = (__int64 *)(v190 + 16LL * v191);
              v193 = *v77;
              if ( v47 == *v77 )
                goto LABEL_319;
              ++v194;
            }
            if ( v195 )
              v77 = v195;
          }
        }
        else if ( v74 - *(_DWORD *)(a1 + 188) - v192 <= v74 >> 3 )
        {
          sub_14DDDA0(v276, v74);
          v199 = *(_DWORD *)(a1 + 192);
          if ( !v199 )
          {
LABEL_536:
            ++*(_DWORD *)(a1 + 184);
            BUG();
          }
          v200 = v199 - 1;
          v201 = *(_QWORD *)(a1 + 176);
          v202 = 0;
          v203 = v200 & (((unsigned int)v47 >> 4) ^ ((unsigned int)v47 >> 9));
          v192 = *(_DWORD *)(a1 + 184) + 1;
          v204 = 1;
          v77 = (__int64 *)(v201 + 16LL * v203);
          v205 = *v77;
          if ( v47 != *v77 )
          {
            while ( v205 != -8 )
            {
              if ( !v202 && v205 == -16 )
                v202 = v77;
              LODWORD(v14) = v204 + 1;
              v203 = v200 & (v204 + v203);
              v77 = (__int64 *)(v201 + 16LL * v203);
              v205 = *v77;
              if ( v47 == *v77 )
                goto LABEL_319;
              ++v204;
            }
            if ( v202 )
              v77 = v202;
          }
        }
LABEL_319:
        *(_DWORD *)(a1 + 184) = v192;
        if ( *v77 != -8 )
          --*(_DWORD *)(a1 + 188);
        *v77 = v47;
        v83 = v77 + 1;
        v77[1] = 0;
        goto LABEL_322;
      }
LABEL_97:
      v79 = v77[1];
      v80 = v79 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v79 & 4) != 0 )
      {
        v81 = *(__int64 **)v80;
        v82 = 8LL * *(unsigned int *)(v80 + 8);
        v83 = (__int64 *)(*(_QWORD *)v80 + v82);
        v84 = v82 >> 3;
        v85 = v82 >> 5;
        if ( !v85 )
          goto LABEL_268;
        do
        {
          if ( *v81 == v297 )
            goto LABEL_105;
          if ( v297 == v81[1] )
          {
            ++v81;
            goto LABEL_105;
          }
          if ( v297 == v81[2] )
          {
            v81 += 2;
            goto LABEL_105;
          }
          if ( v297 == v81[3] )
          {
            v81 += 3;
            goto LABEL_105;
          }
          v81 += 4;
          --v85;
        }
        while ( v85 );
        v84 = v83 - v81;
LABEL_268:
        v179 = v297;
        if ( v84 == 2 )
        {
LABEL_269:
          if ( *v81 != v297 )
          {
            v180 = v81 + 1;
            v81 = v83;
            goto LABEL_271;
          }
          goto LABEL_105;
        }
        if ( v84 == 3 )
        {
          if ( *v81 != v297 )
          {
            ++v81;
            goto LABEL_269;
          }
          goto LABEL_105;
        }
        if ( v84 != 1 )
          goto LABEL_322;
      }
      else
      {
        if ( !v80 )
        {
          v83 = v77 + 1;
LABEL_322:
          v81 = v83;
          goto LABEL_272;
        }
        v83 = v77 + 2;
        v81 = v77 + 1;
      }
      v180 = v81;
      v179 = v297;
      v81 = v83;
LABEL_271:
      if ( v179 != *v180 )
        goto LABEL_272;
      v81 = v180;
LABEL_105:
      if ( v81 != v83 )
      {
        v86 = v81 + 1;
        if ( v81 + 1 != v83 )
        {
          do
          {
            if ( *v86 != v297 )
              *v81++ = *v86;
            ++v86;
          }
          while ( v86 != v83 );
          v79 = v77[1];
        }
        v87 = v79;
        if ( (v79 & 4) == 0 )
        {
          if ( v81 != v83 && v77 + 1 == v81 )
            v77[1] = 0;
          goto LABEL_116;
        }
LABEL_112:
        v88 = v87 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v88 )
        {
          v89 = *(_QWORD *)v88;
          v90 = *(_QWORD *)v88 + 8LL * *(unsigned int *)(v88 + 8) - (_QWORD)v83;
          if ( v83 != (__int64 *)(*(_QWORD *)v88 + 8LL * *(unsigned int *)(v88 + 8)) )
          {
            v91 = (__int64 *)memmove(v81, v83, *(_QWORD *)v88 + 8LL * *(unsigned int *)(v88 + 8) - (_QWORD)v83);
            v89 = *(_QWORD *)v88;
            v81 = v91;
          }
          *(_DWORD *)(v88 + 8) = ((__int64)v81 + v90 - v89) >> 3;
        }
        goto LABEL_116;
      }
LABEL_272:
      v87 = v77[1];
      if ( (v87 & 4) != 0 )
        goto LABEL_112;
LABEL_116:
      if ( v283 == ++v292 )
      {
        v288 = i[1];
        for ( k = *i; v288 != k; ++k )
        {
          v92 = *(_QWORD *)(*k + 48);
          for ( m = *k + 40; m != v92; v92 = *(_QWORD *)(v92 + 8) )
          {
            v94 = v92 - 24;
            if ( !v92 )
              v94 = 0;
            sub_1B75040((__int64 *)&v324, (__int64)&v316, 3, 0, 0);
            sub_1B79630((__int64 *)&v324, v94, v95, v96, v97, v98, a3, a4, a5, a6, v99, v100, a9, a10);
            sub_1B75110((__int64 *)&v324);
          }
        }
        v101 = v300;
        v102 = v299;
        v305 = (__int64 *)v307;
        v306 = 0x200000000LL;
        if ( v300 == v299 )
        {
          v302 = &v298;
          v303 = a1;
          v304 = &v297;
          goto LABEL_222;
        }
LABEL_124:
        while ( 2 )
        {
          v103 = v102->m128i_i64[0];
          v104 = v102->m128i_i64[1];
          LODWORD(v306) = 0;
          n = *(_QWORD *)(v103 + 8);
          if ( !n )
            goto LABEL_145;
          while ( 1 )
          {
            v106 = sub_1648700(n);
            if ( (unsigned __int8)(*((_BYTE *)v106 + 16) - 25) <= 9u )
              break;
            n = *(_QWORD *)(n + 8);
            if ( !n )
            {
              if ( v101 == ++v102 )
                goto LABEL_146;
              goto LABEL_124;
            }
          }
          v107 = 0;
LABEL_130:
          v108 = sub_157EBA0(v106[5]);
          if ( *(_BYTE *)(v108 + 16) != 33 )
            goto LABEL_128;
          v111 = *(_QWORD *)(*(_QWORD *)(v108 - 48) - 24LL);
          if ( (*(_BYTE *)(v111 + 23) & 0x40) != 0 )
          {
            if ( v298 != **(_QWORD **)(v111 - 8) )
              goto LABEL_128;
          }
          else if ( v298 != *(_QWORD *)(v111 - 24LL * (*(_DWORD *)(v111 + 20) & 0xFFFFFFF)) )
          {
            goto LABEL_128;
          }
          if ( (unsigned int)v107 >= HIDWORD(v306) )
          {
            v290 = v108;
            sub_16CD150((__int64)&v305, v307, 0, 8, v109, v110);
            v107 = (unsigned int)v306;
            v108 = v290;
          }
          v305[v107] = v108;
          v107 = (unsigned int)(v306 + 1);
          LODWORD(v306) = v306 + 1;
          for ( n = *(_QWORD *)(n + 8); n; n = *(_QWORD *)(n + 8) )
          {
            v106 = sub_1648700(n);
            if ( (unsigned __int8)(*((_BYTE *)v106 + 16) - 25) <= 9u )
              goto LABEL_130;
LABEL_128:
            ;
          }
          v112 = v305;
          v113 = &v305[(unsigned int)v107];
          if ( v113 != v305 )
          {
            do
            {
              v114 = *v112;
              if ( *(_QWORD *)(*v112 - 24) )
              {
                v115 = *(_QWORD *)(v114 - 16);
                v116 = *(_QWORD *)(v114 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v116 = v115;
                if ( v115 )
                  *(_QWORD *)(v115 + 16) = *(_QWORD *)(v115 + 16) & 3LL | v116;
              }
              *(_QWORD *)(v114 - 24) = v104;
              if ( v104 )
              {
                v117 = *(_QWORD *)(v104 + 8);
                *(_QWORD *)(v114 - 16) = v117;
                if ( v117 )
                  *(_QWORD *)(v117 + 16) = (v114 - 16) | *(_QWORD *)(v117 + 16) & 3LL;
                v118 = *(_QWORD *)(v114 - 8);
                v119 = v114 - 24;
                *(_QWORD *)(v119 + 16) = (v104 + 8) | v118 & 3;
                *(_QWORD *)(v104 + 8) = v119;
              }
              ++v112;
            }
            while ( v113 != v112 );
          }
LABEL_145:
          if ( v101 != ++v102 )
            continue;
          break;
        }
LABEL_146:
        v120 = v299;
        v302 = &v298;
        v294 = v300;
        v303 = a1;
        v304 = &v297;
        if ( v299 != v300 )
        {
          do
          {
            v121 = v120->m128i_i64[1];
            v122 = sub_157F280(v120->m128i_i64[0]);
            v124 = v123;
            v125 = v122;
            if ( v122 != v123 )
            {
              while ( 1 )
              {
                sub_1F61350((__int64)&v302, v125, 1);
                if ( !v125 )
                  goto LABEL_155;
                v126 = *(_QWORD *)(v125 + 32);
                if ( !v126 )
                  BUG();
                if ( *(_BYTE *)(v126 - 8) != 77 )
                  break;
                v125 = v126 - 24;
                if ( v124 == v125 )
                  goto LABEL_176;
              }
              if ( v124 )
              {
                sub_1F61350((__int64)&v302, 0, 1);
LABEL_155:
                BUG();
              }
            }
LABEL_176:
            v134 = sub_157F280(v121);
            v136 = v135;
            v137 = v134;
            if ( v134 != v135 )
            {
              while ( 1 )
              {
                sub_1F61350((__int64)&v302, v137, 0);
                if ( !v137 )
                  goto LABEL_184;
                v138 = *(_QWORD *)(v137 + 32);
                if ( !v138 )
                  BUG();
                if ( *(_BYTE *)(v138 - 8) != 77 )
                  break;
                v137 = v138 - 24;
                if ( v136 == v137 )
                  goto LABEL_185;
              }
              if ( v136 )
              {
                sub_1F61350((__int64)&v302, 0, 0);
LABEL_184:
                BUG();
              }
            }
LABEL_185:
            ++v120;
          }
          while ( v294 != v120 );
          v274 = v300;
          if ( v300 != v299 )
          {
            v278 = v299;
            do
            {
              v295 = v278->m128i_i64[1];
              v139 = sub_157EBA0(v295);
              if ( v139 )
              {
                v140 = v278->m128i_i64[0];
                v289 = sub_15F4D60(v139);
                v284 = sub_157EBA0(v295);
                if ( v289 )
                {
                  for ( ii = 0; ii != v289; ++ii )
                  {
                    v142 = sub_15F4DF0(v284, ii);
                    v143 = sub_157F280(v142);
                    v145 = v144;
                    v146 = v143;
                    while ( v145 != v146 )
                    {
                      v147 = *(_DWORD *)(v146 + 20) & 0xFFFFFFF;
                      v148 = v147;
                      if ( !v147 )
                        break;
                      v149 = *(_BYTE *)(v146 + 23) & 0x40;
                      v150 = *(unsigned int *)(v146 + 56);
                      v151 = 24 * v150 + 8;
                      v152 = 0;
                      while ( 1 )
                      {
                        v153 = v146 - 24LL * v147;
                        if ( (_BYTE)v149 )
                          v153 = *(_QWORD *)(v146 - 8);
                        if ( v140 == *(_QWORD *)(v153 + v151) )
                          break;
                        ++v152;
                        v151 += 8;
                        if ( v147 == (_DWORD)v152 )
                          goto LABEL_220;
                      }
                      v154 = 3 * v152;
                      v155 = *(_QWORD *)(v153 + 8 * v154);
                      if ( *(_BYTE *)(v155 + 16) > 0x17u )
                      {
                        v154 = v319;
                        if ( v319 )
                        {
                          v153 = (__int64)v317;
                          v149 = (v319 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
                          v151 = (__int64)&v317[8 * v149];
                          v156 = *(_QWORD *)(v151 + 24);
                          if ( v156 == v155 )
                          {
LABEL_201:
                            v154 = (__int64)&v317[8 * (unsigned __int64)v319];
                            if ( v151 != v154 )
                              v155 = *(_QWORD *)(v151 + 56);
                          }
                          else
                          {
                            v151 = 1;
                            while ( v156 != -8 )
                            {
                              v149 = (v319 - 1) & ((_DWORD)v151 + (_DWORD)v149);
                              v272 = v151 + 1;
                              v151 = (__int64)&v317[8 * (unsigned __int64)(unsigned int)v149];
                              v156 = *(_QWORD *)(v151 + 24);
                              if ( v155 == v156 )
                                goto LABEL_201;
                              v151 = v272;
                            }
                          }
                        }
                      }
                      if ( v147 == (_DWORD)v150 )
                      {
                        sub_15F55D0(v146, v151, v147, v154, v153, v149);
                        v148 = *(_DWORD *)(v146 + 20) & 0xFFFFFFF;
                      }
                      v157 = (v148 + 1) & 0xFFFFFFF;
                      v158 = v157 - 1;
                      v159 = v157 | *(_DWORD *)(v146 + 20) & 0xF0000000;
                      *(_DWORD *)(v146 + 20) = v159;
                      if ( (v159 & 0x40000000) != 0 )
                        v160 = *(_QWORD *)(v146 - 8);
                      else
                        v160 = v146 - 24 * v157;
                      v161 = (_QWORD *)(v160 + 24LL * v158);
                      if ( *v161 )
                      {
                        v162 = v161[1];
                        v163 = v161[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v163 = v162;
                        if ( v162 )
                          *(_QWORD *)(v162 + 16) = *(_QWORD *)(v162 + 16) & 3LL | v163;
                      }
                      *v161 = v155;
                      if ( v155 )
                      {
                        v164 = *(_QWORD *)(v155 + 8);
                        v161[1] = v164;
                        if ( v164 )
                          *(_QWORD *)(v164 + 16) = (unsigned __int64)(v161 + 1) | *(_QWORD *)(v164 + 16) & 3LL;
                        v161[2] = (v155 + 8) | v161[2] & 3LL;
                        *(_QWORD *)(v155 + 8) = v161;
                      }
                      v165 = *(_DWORD *)(v146 + 20) & 0xFFFFFFF;
                      if ( (*(_BYTE *)(v146 + 23) & 0x40) != 0 )
                        v166 = *(_QWORD *)(v146 - 8);
                      else
                        v166 = v146 - 24 * v165;
                      *(_QWORD *)(v166 + 8LL * (unsigned int)(v165 - 1) + 24LL * *(unsigned int *)(v146 + 56) + 8) = v295;
                      v167 = *(_QWORD *)(v146 + 32);
                      if ( !v167 )
                        BUG();
                      v146 = 0;
                      if ( *(_BYTE *)(v167 - 8) == 77 )
                        v146 = v167 - 24;
                    }
LABEL_220:
                    ;
                  }
                }
              }
              ++v278;
            }
            while ( v274 != v278 );
          }
        }
LABEL_222:
        if ( (_DWORD)v318 )
        {
          v209 = v317;
          v210 = (unsigned __int64)v319 << 6;
          v211 = (_QWORD *)((char *)v317 + v210);
          if ( v317 != (_QWORD *)((char *)v317 + v210) )
          {
            while ( 1 )
            {
              v212 = v209[3];
              if ( v212 != -16 && v212 != -8 )
                break;
              v209 += 8;
              if ( v211 == v209 )
                goto LABEL_223;
            }
            if ( v209 != v211 )
            {
              v279 = v211;
              while ( 1 )
              {
                v213 = v209[3];
                v309[0] = 6;
                v309[1] = 0;
                v308 = v213;
                v214 = v209[7];
                v285 = (__int64 *)v213;
                v296 = v214;
                v310 = v214;
                if ( v214 == -8 || v214 == 0 || v214 == -16 )
                {
                  v324 = (const char *)&v326;
                  v325 = 0x1000000000LL;
                  if ( *(_BYTE *)(v213 + 16) <= 0x17u )
                    goto LABEL_371;
                }
                else
                {
                  sub_1649AC0(v309, v209[5] & 0xFFFFFFFFFFFFFFF8LL);
                  v296 = v310;
                  v285 = (__int64 *)v308;
                  v324 = (const char *)&v326;
                  v325 = 0x1000000000LL;
                  if ( *(_BYTE *)(v308 + 16) <= 0x17u )
                    goto LABEL_382;
                }
                v215 = v285[1];
                if ( !v215 )
                  goto LABEL_381;
                do
                {
                  v224 = sub_1648700(v215);
                  v226 = *(_DWORD *)(a1 + 192);
                  v227 = v224[5];
                  if ( !v226 )
                  {
                    ++*(_QWORD *)(a1 + 168);
                    goto LABEL_358;
                  }
                  v216 = v226 - 1;
                  v217 = *(_QWORD *)(a1 + 176);
                  v218 = (v226 - 1) & (((unsigned int)v227 >> 9) ^ ((unsigned int)v227 >> 4));
                  v219 = (_QWORD *)(v217 + 16LL * v218);
                  v220 = *v219;
                  if ( v227 != *v219 )
                  {
                    v234 = 1;
                    v225 = 0;
                    while ( v220 != -8 )
                    {
                      if ( !v225 && v220 == -16 )
                        v225 = v219;
                      v218 = v216 & (v234 + v218);
                      v219 = (_QWORD *)(v217 + 16LL * v218);
                      v220 = *v219;
                      if ( v227 == *v219 )
                        goto LABEL_350;
                      ++v234;
                    }
                    v235 = *(_DWORD *)(a1 + 184);
                    if ( v225 )
                      v219 = v225;
                    ++*(_QWORD *)(a1 + 168);
                    v230 = v235 + 1;
                    if ( 4 * (v235 + 1) >= 3 * v226 )
                    {
LABEL_358:
                      sub_14DDDA0(v276, 2 * v226);
                      v228 = *(_DWORD *)(a1 + 192);
                      if ( !v228 )
                        goto LABEL_540;
                      v216 = v228 - 1;
                      v225 = *(_QWORD **)(a1 + 176);
                      LODWORD(v229) = v216 & (((unsigned int)v227 >> 9) ^ ((unsigned int)v227 >> 4));
                      v230 = *(_DWORD *)(a1 + 184) + 1;
                      v219 = &v225[2 * (unsigned int)v229];
                      v231 = *v219;
                      if ( v227 != *v219 )
                      {
                        v269 = 1;
                        v270 = 0;
                        while ( v231 != -8 )
                        {
                          if ( v231 == -16 && !v270 )
                            v270 = v219;
                          v229 = v216 & (unsigned int)(v229 + v269);
                          v219 = &v225[2 * v229];
                          v231 = *v219;
                          if ( v227 == *v219 )
                            goto LABEL_360;
                          ++v269;
                        }
                        if ( v270 )
                          v219 = v270;
                      }
                    }
                    else if ( v226 - *(_DWORD *)(a1 + 188) - v230 <= v226 >> 3 )
                    {
                      sub_14DDDA0(v276, v226);
                      v236 = *(_DWORD *)(a1 + 192);
                      if ( !v236 )
                      {
LABEL_540:
                        ++*(_DWORD *)(a1 + 184);
                        BUG();
                      }
                      v216 = v236 - 1;
                      v225 = *(_QWORD **)(a1 + 176);
                      v237 = 0;
                      LODWORD(v238) = v216 & (((unsigned int)v227 >> 9) ^ ((unsigned int)v227 >> 4));
                      v230 = *(_DWORD *)(a1 + 184) + 1;
                      v239 = 1;
                      v219 = &v225[2 * (unsigned int)v238];
                      v240 = *v219;
                      if ( v227 != *v219 )
                      {
                        while ( v240 != -8 )
                        {
                          if ( v240 == -16 && !v237 )
                            v237 = v219;
                          v238 = v216 & (unsigned int)(v238 + v239);
                          v219 = &v225[2 * v238];
                          v240 = *v219;
                          if ( v227 == *v219 )
                            goto LABEL_360;
                          ++v239;
                        }
                        if ( v237 )
                          v219 = v237;
                      }
                    }
LABEL_360:
                    *(_DWORD *)(a1 + 184) = v230;
                    if ( *v219 != -8 )
                      --*(_DWORD *)(a1 + 188);
                    *v219 = v227;
                    v219[1] = 0;
LABEL_363:
                    v232 = v219 + 1;
                    goto LABEL_364;
                  }
LABEL_350:
                  v221 = ((__int64)v219[1] >> 2) & 1;
                  v222 = v219[1] & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v222 )
                  {
                    if ( !(_BYTE)v221 )
                      goto LABEL_363;
                    if ( *(_DWORD *)(v222 + 8) > 1u )
                    {
                      v223 = (unsigned int)v325;
                      if ( (unsigned int)v325 >= HIDWORD(v325) )
                        goto LABEL_366;
                      goto LABEL_354;
                    }
                  }
                  else
                  {
                    if ( !(_BYTE)v221 )
                      goto LABEL_363;
                    v222 = 0;
                  }
                  v232 = *(_QWORD **)v222;
LABEL_364:
                  if ( *v232 != v297 )
                  {
                    v223 = (unsigned int)v325;
                    if ( (unsigned int)v325 >= HIDWORD(v325) )
                    {
LABEL_366:
                      sub_16CD150((__int64)&v324, &v326, 0, 8, v216, (int)v225);
                      v223 = (unsigned int)v325;
                    }
LABEL_354:
                    *(_QWORD *)&v324[8 * v223] = v215;
                    LODWORD(v325) = v325 + 1;
                  }
                  v215 = *(_QWORD *)(v215 + 8);
                }
                while ( v215 );
                if ( (_DWORD)v325 )
                {
                  sub_1B3B830((__int64)&v311, 0);
                  v241 = (char *)sub_1649960((__int64)v285);
                  sub_1B3B8C0((__int64 *)&v311, *v285, v241, v242);
                  sub_1B3BE00((__int64 *)&v311, v285[5], (__int64)v285);
                  sub_1B3BE00((__int64 *)&v311, *(_QWORD *)(v296 + 40), v296);
                  for ( jj = v325; (_DWORD)v325; jj = v325 )
                  {
                    v244 = *(__int64 **)&v324[8 * jj - 8];
                    LODWORD(v325) = jj - 1;
                    sub_1B42200((__int64 *)&v311, v244);
                  }
                  sub_1B3B860((__int64 *)&v311);
                  if ( v324 != (const char *)&v326 )
                    _libc_free((unsigned __int64)v324);
                  if ( v310 != 0 && v310 != -8 && v310 != -16 )
                    goto LABEL_384;
                }
                else
                {
                  if ( v324 != (const char *)&v326 )
                    _libc_free((unsigned __int64)v324);
LABEL_381:
                  v296 = v310;
LABEL_382:
                  if ( v296 != 0 && v296 != -8 && v296 != -16 )
LABEL_384:
                    sub_1649B30(v309);
                }
LABEL_371:
                v209 += 8;
                if ( v209 != v279 )
                {
                  while ( 1 )
                  {
                    v233 = v209[3];
                    if ( v233 != -8 && v233 != -16 )
                      break;
                    v209 += 8;
                    if ( v279 == v209 )
                      goto LABEL_223;
                  }
                  if ( v279 != v209 )
                    continue;
                }
                break;
              }
            }
          }
        }
LABEL_223:
        if ( v305 != (__int64 *)v307 )
          _libc_free((unsigned __int64)v305);
        if ( v322 )
        {
          if ( v321 )
          {
            v206 = v320;
            v207 = &v320[2 * v321];
            do
            {
              if ( *v206 != -8 && *v206 != -4 )
              {
                v208 = v206[1];
                if ( v208 )
                  sub_161E7C0((__int64)(v206 + 1), v208);
              }
              v206 += 2;
            }
            while ( v207 != v206 );
          }
          j___libc_free_0(v320);
        }
        if ( v319 )
        {
          v168 = v317;
          v312 = 2;
          v313 = 0;
          v169 = &v317[8 * (unsigned __int64)v319];
          v314 = -8;
          v311 = (const char *)&unk_49E6B50;
          v324 = (const char *)&unk_49E6B50;
          v170 = -8;
          v315 = 0;
          v325 = 2;
          v326 = 0;
          v327 = -16;
          j = 0;
          while ( 1 )
          {
            v171 = v168[3];
            if ( v170 != v171 && v171 != v327 )
            {
              v172 = v168[7];
              if ( v172 != -8 && v172 != 0 && v172 != -16 )
              {
                sub_1649B30(v168 + 5);
                v171 = v168[3];
              }
            }
            *v168 = &unk_49EE2B0;
            if ( v171 != 0 && v171 != -8 && v171 != -16 )
              sub_1649B30(v168 + 1);
            v168 += 8;
            if ( v169 == v168 )
              break;
            v170 = v314;
          }
          v324 = (const char *)&unk_49EE2B0;
          v173 = v327;
          if ( v327 != 0 && v327 != -8 )
            goto LABEL_239;
          goto LABEL_241;
        }
        goto LABEL_244;
      }
    }
    v174 = 1;
    v175 = 0;
    while ( v55 != -8 )
    {
      if ( !v175 && v55 == -16 )
        v175 = v54;
      LODWORD(v14) = v174 + 1;
      v53 = (v50 - 1) & ((_DWORD)v53 + v174);
      v54 = (__int64 *)(v52 + 16 * v53);
      v55 = *v54;
      if ( v324 == (const char *)*v54 )
        goto LABEL_67;
      ++v174;
    }
    v176 = *(_DWORD *)(a1 + 184);
    if ( v175 )
      v54 = v175;
    ++*(_QWORD *)(a1 + 168);
    v177 = v176 + 1;
    if ( 4 * (v176 + 1) >= 3 * v50 )
    {
LABEL_279:
      sub_14DDDA0(v276, 2 * v50);
      v181 = *(_DWORD *)(a1 + 192);
      if ( !v181 )
        goto LABEL_539;
      v51 = (__int64)v324;
      v182 = v181 - 1;
      v183 = *(_QWORD *)(a1 + 176);
      v177 = *(_DWORD *)(a1 + 184) + 1;
      LODWORD(v184) = v182 & (((unsigned int)v324 >> 9) ^ ((unsigned int)v324 >> 4));
      v54 = (__int64 *)(v183 + 16LL * (unsigned int)v184);
      v185 = *v54;
      if ( v324 == (const char *)*v54 )
        goto LABEL_263;
      v186 = 1;
      v187 = 0;
      while ( v185 != -8 )
      {
        if ( !v187 && v185 == -16 )
          v187 = v54;
        LODWORD(v14) = v186 + 1;
        v184 = v182 & (unsigned int)(v184 + v186);
        v54 = (__int64 *)(v183 + 16 * v184);
        v185 = *v54;
        if ( v324 == (const char *)*v54 )
          goto LABEL_263;
        ++v186;
      }
    }
    else
    {
      if ( v50 - *(_DWORD *)(a1 + 188) - v177 > v50 >> 3 )
        goto LABEL_263;
      sub_14DDDA0(v276, v50);
      v250 = *(_DWORD *)(a1 + 192);
      if ( !v250 )
      {
LABEL_539:
        ++*(_DWORD *)(a1 + 184);
        BUG();
      }
      v51 = (__int64)v324;
      v251 = v250 - 1;
      v252 = *(_QWORD *)(a1 + 176);
      v187 = 0;
      v177 = *(_DWORD *)(a1 + 184) + 1;
      v253 = 1;
      LODWORD(v254) = v251 & (((unsigned int)v324 >> 9) ^ ((unsigned int)v324 >> 4));
      v54 = (__int64 *)(v252 + 16LL * (unsigned int)v254);
      v255 = *v54;
      if ( v324 == (const char *)*v54 )
        goto LABEL_263;
      while ( v255 != -8 )
      {
        if ( !v187 && v255 == -16 )
          v187 = v54;
        LODWORD(v14) = v253 + 1;
        v254 = v251 & (unsigned int)(v254 + v253);
        v54 = (__int64 *)(v252 + 16 * v254);
        v255 = *v54;
        if ( v324 == (const char *)*v54 )
          goto LABEL_263;
        ++v253;
      }
    }
    if ( v187 )
      v54 = v187;
LABEL_263:
    *(_DWORD *)(a1 + 184) = v177;
    if ( *v54 != -8 )
      --*(_DWORD *)(a1 + 188);
    *v54 = v51;
    v178 = v297;
    v54[1] = 0;
    v287 = v178;
    goto LABEL_266;
  }
  return result;
}
