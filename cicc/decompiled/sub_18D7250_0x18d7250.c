// Function: sub_18D7250
// Address: 0x18d7250
//
__int64 __fastcall sub_18D7250(
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
  double v10; // xmm4_8
  double v11; // xmm5_8
  __int64 v12; // rbx
  __int64 i; // rdi
  int v14; // edi
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned __int8 v17; // al
  int v18; // eax
  unsigned __int8 v19; // al
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r12
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r8
  __int64 *v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rdx
  __int64 v30; // r15
  _WORD *v31; // rdi
  __int64 v32; // r14
  unsigned int v33; // edx
  _QWORD *v34; // rax
  __int64 v35; // r8
  __int64 *v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // r15
  __int64 *v39; // rax
  __int64 *v40; // r12
  unsigned int v41; // esi
  _QWORD *v42; // r13
  int v43; // ecx
  __int64 v44; // r9
  _QWORD *v45; // rdx
  _QWORD *v46; // rax
  _QWORD *v47; // r14
  __int64 v48; // rax
  __int64 *v49; // rax
  char v50; // dl
  __int64 *v51; // rdx
  __int64 *v52; // r8
  int v53; // r9d
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  int v57; // eax
  __int64 v58; // r12
  __int64 v59; // r13
  unsigned int v60; // r13d
  _QWORD *v61; // rbx
  _QWORD *v62; // r14
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // rdi
  _QWORD *v65; // r15
  _QWORD *v66; // r12
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // rdi
  _QWORD *v69; // r15
  _QWORD *v70; // r12
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rdi
  __int64 *v73; // rbx
  __int64 *v74; // r12
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // rdi
  _QWORD *v77; // rbx
  _QWORD *v78; // r12
  unsigned __int64 v79; // rdi
  unsigned __int64 v80; // rdi
  __int64 v82; // rax
  __int64 *v83; // rbx
  __int64 *v84; // rax
  __int64 *v85; // rdx
  __int64 *v86; // rdi
  __int64 *v87; // rsi
  __int64 v88; // rdx
  __int64 v89; // rsi
  __int64 *v90; // rax
  __int64 v91; // rdx
  __int64 *v92; // rbx
  __int64 v93; // rsi
  __int64 *v94; // r13
  __int64 *v95; // rax
  __int64 *v96; // r12
  __int64 v97; // rbx
  char v98; // dl
  __int64 *v99; // rdx
  __int64 v100; // rax
  __int64 v101; // rdx
  unsigned __int64 v102; // rax
  __int64 *v103; // rax
  __int64 *v104; // rdi
  int v105; // edx
  _QWORD *v106; // rdx
  __int64 v107; // rdi
  int v108; // edi
  char v109; // al
  __int64 v110; // rax
  __int64 v111; // r12
  unsigned __int8 v112; // al
  int v113; // eax
  _QWORD ***v114; // r15
  __int64 *v115; // rax
  int v116; // r8d
  __int64 v117; // r9
  __int64 *v118; // rax
  __int64 v119; // r13
  __int64 *v120; // rax
  __int64 *v121; // rax
  __int64 *v122; // r13
  __int64 v123; // r12
  __int64 *v124; // rbx
  __int64 *v125; // rdx
  __int64 *v126; // rbx
  __int64 v127; // r13
  __int64 *v128; // r12
  int v129; // edi
  unsigned int v130; // eax
  _QWORD *v131; // r14
  unsigned __int64 v132; // rdi
  unsigned __int64 v133; // rdi
  __int64 v134; // rax
  __int64 *v135; // rax
  unsigned int v136; // ecx
  __int64 *v137; // rdx
  __int64 v138; // rdi
  __int64 v139; // rax
  __int64 *v140; // rax
  __int64 v141; // r15
  __int64 v142; // r12
  _QWORD *v143; // rax
  __int64 v144; // r13
  __int64 v145; // rbx
  _QWORD *v146; // rax
  __int64 v147; // r12
  __int64 v148; // r11
  __int64 v149; // r13
  int v150; // esi
  __int64 *v151; // rax
  __int64 v152; // rax
  bool v153; // zf
  __int64 *v154; // rax
  __int64 *j; // rdx
  __int64 *v156; // r13
  __int64 *v157; // rax
  __int64 v158; // r12
  __int64 *v159; // rax
  __int64 v160; // rax
  __int64 v161; // rax
  int v162; // eax
  __int64 v163; // r12
  _QWORD *v164; // rax
  __int64 v165; // r15
  __int64 v166; // rbx
  _QWORD *v167; // rax
  __int64 v168; // r12
  __int64 v169; // r11
  __int64 *v170; // rax
  __int64 *v171; // rax
  __int64 *v172; // r15
  __int64 *v173; // rax
  __int64 v174; // r12
  __int64 v175; // rax
  __int64 v176; // rdi
  __int64 v177; // rax
  int v178; // edx
  __int64 v179; // rdi
  __int64 *v180; // r13
  __int64 v181; // r15
  unsigned int v182; // edx
  __int64 *v183; // rax
  __int64 v184; // r10
  __int64 *v185; // rbx
  _QWORD *v186; // rdx
  _QWORD *v187; // rax
  _QWORD *v188; // r8
  __int64 v189; // rax
  __int64 *v190; // rax
  char v191; // dl
  __int64 *v192; // rdx
  int v193; // r8d
  __int64 *v194; // r9
  __int64 v195; // rax
  __int64 v196; // rdx
  unsigned __int64 v197; // rax
  __int64 v198; // rax
  __int64 *v199; // rax
  __int64 *v200; // rsi
  __int64 *v201; // rcx
  _QWORD *v202; // rdx
  __int64 v203; // rdx
  __int64 v204; // rcx
  __int64 *v205; // rax
  __int64 v206; // rdx
  __int64 v207; // rcx
  __int64 *v208; // rbx
  __int64 *v209; // rax
  __int64 *v210; // r12
  char v211; // dl
  __int64 *v212; // rdx
  __int64 v213; // rax
  __int64 v214; // rdx
  unsigned __int64 v215; // rax
  __int64 *v216; // rax
  __int64 *v217; // rdi
  __int64 *v218; // rsi
  int v219; // eax
  int v220; // r8d
  __int64 v221; // r10
  int v222; // eax
  int v223; // r9d
  int v224; // r9d
  __int64 v225; // [rsp+8h] [rbp-348h]
  __int64 *v226; // [rsp+18h] [rbp-338h]
  _QWORD ***v227; // [rsp+20h] [rbp-330h]
  unsigned __int8 v228; // [rsp+2Dh] [rbp-323h]
  bool v229; // [rsp+2Eh] [rbp-322h]
  char v230; // [rsp+2Fh] [rbp-321h]
  __int64 *v231; // [rsp+30h] [rbp-320h]
  char v232; // [rsp+38h] [rbp-318h]
  char v233; // [rsp+39h] [rbp-317h]
  char v234; // [rsp+39h] [rbp-317h]
  char v235; // [rsp+3Ah] [rbp-316h]
  char v236; // [rsp+3Ah] [rbp-316h]
  char v237; // [rsp+3Bh] [rbp-315h]
  int v238; // [rsp+3Ch] [rbp-314h]
  __int64 *v239; // [rsp+48h] [rbp-308h]
  __int64 *v240; // [rsp+48h] [rbp-308h]
  __int64 v241; // [rsp+50h] [rbp-300h]
  __int64 v242; // [rsp+50h] [rbp-300h]
  __int64 *v243; // [rsp+58h] [rbp-2F8h]
  int v244; // [rsp+60h] [rbp-2F0h]
  __int64 *v245; // [rsp+60h] [rbp-2F0h]
  __int64 *v246; // [rsp+60h] [rbp-2F0h]
  char v247; // [rsp+68h] [rbp-2E8h]
  int v248; // [rsp+70h] [rbp-2E0h]
  _QWORD *v249; // [rsp+70h] [rbp-2E0h]
  __int64 v250; // [rsp+70h] [rbp-2E0h]
  __int64 v251; // [rsp+70h] [rbp-2E0h]
  _QWORD *v252; // [rsp+70h] [rbp-2E0h]
  char v254; // [rsp+A0h] [rbp-2B0h]
  __int64 v255; // [rsp+A0h] [rbp-2B0h]
  __int64 v256; // [rsp+A0h] [rbp-2B0h]
  __int64 *v257; // [rsp+B0h] [rbp-2A0h]
  _QWORD **v258; // [rsp+B0h] [rbp-2A0h]
  _QWORD *v259; // [rsp+B0h] [rbp-2A0h]
  __int64 *v260; // [rsp+B0h] [rbp-2A0h]
  __int64 *v261; // [rsp+B8h] [rbp-298h]
  __int64 *v262; // [rsp+B8h] [rbp-298h]
  __int64 *v263; // [rsp+B8h] [rbp-298h]
  __int64 *v264; // [rsp+B8h] [rbp-298h]
  __int64 v265; // [rsp+C8h] [rbp-288h] BYREF
  __int64 v266; // [rsp+D0h] [rbp-280h] BYREF
  _QWORD *v267; // [rsp+D8h] [rbp-278h]
  __int64 v268; // [rsp+E0h] [rbp-270h]
  unsigned int v269; // [rsp+E8h] [rbp-268h]
  __int64 v270; // [rsp+F0h] [rbp-260h] BYREF
  _QWORD *v271; // [rsp+F8h] [rbp-258h]
  __int64 v272; // [rsp+100h] [rbp-250h]
  unsigned int v273; // [rsp+108h] [rbp-248h]
  __int64 *v274; // [rsp+110h] [rbp-240h] BYREF
  __int64 v275; // [rsp+118h] [rbp-238h]
  _QWORD v276[4]; // [rsp+120h] [rbp-230h] BYREF
  _WORD *v277; // [rsp+140h] [rbp-210h] BYREF
  __int64 v278; // [rsp+148h] [rbp-208h]
  _WORD v279[16]; // [rsp+150h] [rbp-200h] BYREF
  __int64 v280; // [rsp+170h] [rbp-1E0h] BYREF
  __int64 v281; // [rsp+178h] [rbp-1D8h]
  __int64 v282; // [rsp+180h] [rbp-1D0h]
  unsigned int v283; // [rsp+188h] [rbp-1C8h]
  __int64 *v284; // [rsp+190h] [rbp-1C0h]
  __int64 *v285; // [rsp+198h] [rbp-1B8h]
  __int64 v286; // [rsp+1A0h] [rbp-1B0h]
  _BYTE *v287; // [rsp+1B0h] [rbp-1A0h] BYREF
  __int64 v288; // [rsp+1B8h] [rbp-198h]
  _BYTE v289[64]; // [rsp+1C0h] [rbp-190h] BYREF
  __int16 v290; // [rsp+200h] [rbp-150h]
  __int64 v291; // [rsp+208h] [rbp-148h]
  __int64 v292; // [rsp+210h] [rbp-140h] BYREF
  __int64 *v293; // [rsp+218h] [rbp-138h]
  __int64 *v294; // [rsp+220h] [rbp-130h]
  __int64 v295; // [rsp+228h] [rbp-128h]
  int v296; // [rsp+230h] [rbp-120h]
  _BYTE v297[16]; // [rsp+238h] [rbp-118h] BYREF
  __int64 v298; // [rsp+248h] [rbp-108h] BYREF
  __int64 *v299; // [rsp+250h] [rbp-100h]
  __int64 *v300; // [rsp+258h] [rbp-F8h]
  __int64 v301; // [rsp+260h] [rbp-F0h]
  int v302; // [rsp+268h] [rbp-E8h]
  _BYTE v303[32]; // [rsp+270h] [rbp-E0h] BYREF
  __int16 v304; // [rsp+290h] [rbp-C0h]
  __int64 v305; // [rsp+298h] [rbp-B8h]
  __int64 v306; // [rsp+2A0h] [rbp-B0h] BYREF
  __int64 *v307; // [rsp+2A8h] [rbp-A8h]
  __int64 *v308; // [rsp+2B0h] [rbp-A0h]
  __int64 v309; // [rsp+2B8h] [rbp-98h]
  int v310; // [rsp+2C0h] [rbp-90h]
  _BYTE v311[16]; // [rsp+2C8h] [rbp-88h] BYREF
  __int64 v312; // [rsp+2D8h] [rbp-78h] BYREF
  __int64 *v313; // [rsp+2E0h] [rbp-70h]
  __int64 *v314; // [rsp+2E8h] [rbp-68h]
  __int64 v315; // [rsp+2F0h] [rbp-60h]
  int v316; // [rsp+2F8h] [rbp-58h]
  _BYTE v317[80]; // [rsp+300h] [rbp-50h] BYREF

  v266 = 0;
  v267 = 0;
  v268 = 0;
  v269 = 0;
  v280 = 0;
  v281 = 0;
  v282 = 0;
  v283 = 0;
  v284 = 0;
  v285 = 0;
  v286 = 0;
  v270 = 0;
  v271 = 0;
  v272 = 0;
  v273 = 0;
  v228 = sub_18D3D80(a1, a2, (__int64)&v270, (__int64)&v280, (__int64)&v266);
  v287 = v289;
  v288 = 0x800000000LL;
  v231 = v285;
  if ( v284 == v285 )
  {
    v60 = 0;
    goto LABEL_76;
  }
  v243 = v284;
  v229 = 0;
  do
  {
    v12 = *v243;
    if ( !*v243 )
      goto LABEL_70;
    for ( i = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
          ;
          i = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)) )
    {
      v15 = sub_1649C60(i);
      v14 = 23;
      v16 = v15;
      v17 = *(_BYTE *)(v15 + 16);
      if ( v17 > 0x17u )
      {
        if ( v17 != 78 )
        {
          v14 = 2 * (v17 != 29) + 21;
          goto LABEL_6;
        }
        v14 = 21;
        if ( !*(_BYTE *)(*(_QWORD *)(v16 - 24) + 16LL) )
          break;
      }
LABEL_6:
      if ( !(unsigned __int8)sub_1439C90(v14) )
        goto LABEL_12;
LABEL_7:
      ;
    }
    v18 = sub_1438F00(*(_QWORD *)(v16 - 24));
    if ( (unsigned __int8)sub_1439C90(v18) )
      goto LABEL_7;
LABEL_12:
    v19 = *(_BYTE *)(v16 + 16);
    v247 = 1;
    if ( v19 <= 0x10u )
      goto LABEL_15;
    if ( v19 != 54 )
    {
      v247 = v19 == 53;
      goto LABEL_15;
    }
    v107 = *(_QWORD *)(v16 - 24);
    while ( 2 )
    {
      v110 = sub_1649C60(v107);
      v108 = 23;
      v111 = v110;
      v112 = *(_BYTE *)(v110 + 16);
      if ( v112 <= 0x17u )
        goto LABEL_205;
      if ( v112 != 78 )
      {
        v108 = 2 * (v112 != 29) + 21;
LABEL_205:
        v109 = sub_1439C90(v108);
        if ( !v109 )
          goto LABEL_211;
LABEL_206:
        v107 = *(_QWORD *)(v111 - 24LL * (*(_DWORD *)(v111 + 20) & 0xFFFFFFF));
        continue;
      }
      break;
    }
    v108 = 21;
    if ( *(_BYTE *)(*(_QWORD *)(v111 - 24) + 16LL) )
      goto LABEL_205;
    v113 = sub_1438F00(*(_QWORD *)(v111 - 24));
    v109 = sub_1439C90(v113);
    if ( v109 )
      goto LABEL_206;
LABEL_211:
    v247 = v109;
    if ( *(_BYTE *)(v111 + 16) == 3 )
      v247 = *(_BYTE *)(v111 + 80) & 1;
LABEL_15:
    v20 = v276;
    v293 = (__int64 *)v297;
    v294 = (__int64 *)v297;
    v299 = (__int64 *)v303;
    v300 = (__int64 *)v303;
    v307 = (__int64 *)v311;
    v308 = (__int64 *)v311;
    v313 = (__int64 *)v317;
    v314 = (__int64 *)v317;
    v275 = 0x400000001LL;
    v21 = 1;
    v290 = 0;
    v291 = 0;
    v292 = 0;
    v295 = 2;
    v296 = 0;
    v298 = 0;
    v301 = 2;
    v302 = 0;
    v303[16] = 0;
    v304 = 0;
    v305 = 0;
    v306 = 0;
    v309 = 2;
    v310 = 0;
    v312 = 0;
    v315 = 2;
    v316 = 0;
    v317[16] = 0;
    v274 = v276;
    v276[0] = v12;
    v237 = 0;
    v254 = 1;
    v238 = 0;
    v248 = 0;
    v244 = 0;
    v230 = 1;
    v232 = 1;
    v227 = (_QWORD ***)v16;
LABEL_16:
    v278 = 0x400000000LL;
    v277 = v279;
    v226 = &v20[v21];
    if ( v226 == v20 )
    {
      v114 = v227;
      goto LABEL_214;
    }
    v239 = v20;
    while ( 2 )
    {
      v22 = *v239;
      if ( !v283 )
        goto LABEL_199;
      v23 = (v283 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v24 = (__int64 *)(v281 + 16LL * v23);
      v25 = *v24;
      if ( v22 != *v24 )
      {
        v105 = 1;
        while ( v25 != -8 )
        {
          v224 = v105 + 1;
          v23 = (v283 - 1) & (v105 + v23);
          v24 = (__int64 *)(v281 + 16LL * v23);
          v25 = *v24;
          if ( v22 == *v24 )
            goto LABEL_20;
          v105 = v224;
        }
LABEL_199:
        v26 = v285;
        goto LABEL_22;
      }
LABEL_20:
      if ( v24 == (__int64 *)(v281 + 16LL * v283) )
        goto LABEL_199;
      v26 = &v284[18 * v24[1]];
LABEL_22:
      v235 = *((_BYTE *)v26 + 8);
      v233 = *((_BYTE *)v26 + 136);
      v27 = (__int64 *)v26[5];
      if ( v27 == (__int64 *)v26[4] )
        v28 = *((unsigned int *)v26 + 13);
      else
        v28 = *((unsigned int *)v26 + 12);
      v29 = &v27[v28];
      v261 = v29;
      if ( v27 == v29 )
        goto LABEL_27;
      while ( 1 )
      {
        v30 = *v27;
        if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v29 == ++v27 )
          goto LABEL_27;
      }
      if ( v29 == v27 )
      {
LABEL_27:
        ++v239;
        v232 &= v235;
        v237 |= v233;
        if ( v226 != v239 )
          continue;
        v31 = v277;
        LODWORD(v275) = 0;
        if ( !(_DWORD)v278 )
          goto LABEL_405;
        v262 = (__int64 *)v277;
        v240 = (__int64 *)&v277[4 * (unsigned int)v278];
        while ( 2 )
        {
          v32 = *v262;
          if ( v269 )
          {
            v33 = (v269 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v34 = &v267[18 * v33];
            v35 = *v34;
            if ( v32 == *v34 )
            {
LABEL_32:
              v36 = (__int64 *)v34[5];
              v236 = *((_BYTE *)v34 + 8);
              v234 = *((_BYTE *)v34 + 136);
              if ( v36 == (__int64 *)v34[4] )
                v37 = *((unsigned int *)v34 + 13);
              else
                v37 = *((unsigned int *)v34 + 12);
              v38 = &v36[v37];
              if ( v36 != v38 )
              {
                v39 = v36;
                while ( 1 )
                {
                  v40 = v39;
                  if ( (unsigned __int64)*v39 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v38 == ++v39 )
                    goto LABEL_38;
                }
                if ( v38 != v39 )
                {
                  v180 = v38;
                  v181 = *v39;
                  do
                  {
                    if ( !v283 )
                      goto LABEL_58;
                    v182 = (v283 - 1) & (((unsigned int)v181 >> 9) ^ ((unsigned int)v181 >> 4));
                    v183 = (__int64 *)(v281 + 16LL * v182);
                    v184 = *v183;
                    if ( v181 != *v183 )
                    {
                      v219 = 1;
                      if ( v184 == -8 )
                        goto LABEL_58;
                      while ( 1 )
                      {
                        v220 = v219 + 1;
                        v182 = (v283 - 1) & (v219 + v182);
                        v183 = (__int64 *)(v281 + 16LL * v182);
                        v221 = *v183;
                        if ( v181 == *v183 )
                          break;
                        v219 = v220;
                        if ( v221 == -8 )
                          goto LABEL_58;
                      }
                    }
                    if ( v183 == (__int64 *)(v281 + 16LL * v283) )
                      goto LABEL_58;
                    v185 = &v284[18 * v183[1]];
                    if ( v285 == v185 )
                      goto LABEL_58;
                    v186 = (_QWORD *)v185[5];
                    v187 = (_QWORD *)v185[4];
                    if ( v186 == v187 )
                    {
                      v202 = &v187[*((unsigned int *)v185 + 13)];
                      if ( v187 == v202 )
                      {
                        v188 = (_QWORD *)v185[4];
                      }
                      else
                      {
                        do
                        {
                          if ( v32 == *v187 )
                            break;
                          ++v187;
                        }
                        while ( v202 != v187 );
                        v188 = v202;
                      }
                    }
                    else
                    {
                      v259 = &v186[*((unsigned int *)v185 + 12)];
                      v187 = sub_16CC9F0((__int64)(v185 + 3), v32);
                      v188 = v259;
                      if ( v32 == *v187 )
                      {
                        v203 = v185[5];
                        if ( v203 == v185[4] )
                          v204 = *((unsigned int *)v185 + 13);
                        else
                          v204 = *((unsigned int *)v185 + 12);
                        v202 = (_QWORD *)(v203 + 8 * v204);
                      }
                      else
                      {
                        v189 = v185[5];
                        if ( v189 != v185[4] )
                        {
                          v187 = (_QWORD *)(v189 + 8LL * *((unsigned int *)v185 + 12));
                          goto LABEL_328;
                        }
                        v202 = (_QWORD *)(v189 + 8LL * *((unsigned int *)v185 + 13));
                        v187 = v202;
                      }
                    }
                    while ( v202 != v187 && *v187 >= 0xFFFFFFFFFFFFFFFELL )
                      ++v187;
LABEL_328:
                    if ( v188 == v187 )
                      goto LABEL_58;
                    v190 = v293;
                    if ( v294 != v293 )
                      goto LABEL_330;
                    v200 = &v293[HIDWORD(v295)];
                    if ( v293 == v200 )
                    {
LABEL_401:
                      if ( HIDWORD(v295) >= (unsigned int)v295 )
                      {
LABEL_330:
                        sub_16CCBA0((__int64)&v292, v181);
                        if ( !v191 )
                          goto LABEL_339;
                      }
                      else
                      {
                        ++HIDWORD(v295);
                        *v200 = v181;
                        ++v292;
                      }
LABEL_331:
                      v265 = *(_QWORD *)(v181 + 40);
                      v192 = sub_18CDCF0((__int64)&v270, &v265);
                      v195 = *((unsigned int *)v192 + 2);
                      if ( (_DWORD)v195 == -1 )
                        goto LABEL_58;
                      v196 = *((unsigned int *)v192 + 3);
                      if ( (_DWORD)v196 == -1 )
                        goto LABEL_58;
                      v197 = v196 * v195;
                      if ( HIDWORD(v197) || v197 == 0xFFFFFFFF )
                        goto LABEL_58;
                      v244 += v197;
                      if ( !v247 )
                      {
                        v205 = (__int64 *)v185[12];
                        v206 = v205 == (__int64 *)v185[11] ? *((unsigned int *)v185 + 27) : *((unsigned int *)v185 + 26);
                        v194 = &v205[v206];
                        if ( v205 != v194 )
                        {
                          while ( 1 )
                          {
                            v207 = *v205;
                            v208 = v205;
                            if ( (unsigned __int64)*v205 < 0xFFFFFFFFFFFFFFFELL )
                              break;
                            if ( v194 == ++v205 )
                              goto LABEL_336;
                          }
                          if ( v205 != v194 )
                          {
                            v260 = v40;
                            v209 = v299;
                            v210 = v194;
                            if ( v300 == v299 )
                            {
LABEL_386:
                              v217 = &v209[HIDWORD(v301)];
                              v193 = HIDWORD(v301);
                              if ( v209 != v217 )
                              {
                                v218 = 0;
                                do
                                {
                                  if ( *v209 == v207 )
                                    goto LABEL_380;
                                  if ( *v209 == -2 )
                                    v218 = v209;
                                  ++v209;
                                }
                                while ( v217 != v209 );
                                if ( v218 )
                                {
                                  *v218 = v207;
                                  --v302;
                                  ++v298;
                                  goto LABEL_375;
                                }
                              }
                              if ( HIDWORD(v301) < (unsigned int)v301 )
                              {
                                ++HIDWORD(v301);
                                *v217 = v207;
                                ++v298;
                                goto LABEL_375;
                              }
                            }
                            while ( 1 )
                            {
                              v225 = v207;
                              sub_16CCBA0((__int64)&v298, v207);
                              v207 = v225;
                              if ( v211 )
                              {
LABEL_375:
                                v265 = *(_QWORD *)(v207 + 40);
                                v212 = sub_18CDCF0((__int64)&v270, &v265);
                                v213 = *((unsigned int *)v212 + 2);
                                if ( (_DWORD)v213 == -1 )
                                  goto LABEL_58;
                                v214 = *((unsigned int *)v212 + 3);
                                if ( (_DWORD)v214 == -1 )
                                  goto LABEL_58;
                                v215 = v214 * v213;
                                if ( HIDWORD(v215) || v215 == 0xFFFFFFFF )
                                  goto LABEL_58;
                                v248 += v215;
                                v238 += v215;
                              }
LABEL_380:
                              v216 = v208 + 1;
                              if ( v208 + 1 == v210 )
                                break;
                              while ( 1 )
                              {
                                v207 = *v216;
                                v208 = v216;
                                if ( (unsigned __int64)*v216 < 0xFFFFFFFFFFFFFFFELL )
                                  break;
                                if ( v210 == ++v216 )
                                  goto LABEL_383;
                              }
                              if ( v210 == v216 )
                                break;
                              v209 = v299;
                              if ( v300 == v299 )
                                goto LABEL_386;
                            }
LABEL_383:
                            v40 = v260;
                          }
                        }
                      }
LABEL_336:
                      v198 = (unsigned int)v275;
                      if ( (unsigned int)v275 >= HIDWORD(v275) )
                      {
                        sub_16CD150((__int64)&v274, v276, 0, 8, v193, (int)v194);
                        v198 = (unsigned int)v275;
                      }
                      v274[v198] = v181;
                      LODWORD(v275) = v275 + 1;
                      goto LABEL_339;
                    }
                    v201 = 0;
                    while ( v181 != *v190 )
                    {
                      if ( *v190 == -2 )
                        v201 = v190;
                      if ( v200 == ++v190 )
                      {
                        if ( !v201 )
                          goto LABEL_401;
                        *v201 = v181;
                        --v296;
                        ++v292;
                        goto LABEL_331;
                      }
                    }
LABEL_339:
                    v199 = v40 + 1;
                    if ( v40 + 1 == v180 )
                      break;
                    v181 = *v199;
                    for ( ++v40; (unsigned __int64)*v199 >= 0xFFFFFFFFFFFFFFFELL; v40 = v199 )
                    {
                      if ( v180 == ++v199 )
                        goto LABEL_38;
                      v181 = *v199;
                    }
                  }
                  while ( v180 != v40 );
                }
              }
LABEL_38:
              ++v262;
              v230 &= v236;
              v237 |= v234;
              if ( v240 != v262 )
                continue;
              v21 = (unsigned int)v275;
              v31 = v277;
              if ( (_DWORD)v275 )
              {
                if ( v277 != v279 )
                {
                  _libc_free((unsigned __int64)v277);
                  v21 = (unsigned int)v275;
                }
                v20 = v274;
                goto LABEL_16;
              }
LABEL_405:
              v114 = v227;
              if ( v31 != v279 )
                _libc_free((unsigned __int64)v31);
LABEL_214:
              if ( v274 != v276 )
                _libc_free((unsigned __int64)v274);
              if ( v232 && v230 )
              {
                sub_18CE100((__int64)&v298);
                sub_18CE100((__int64)&v312);
                v238 = 0;
              }
              else if ( v248 || (HIDWORD(v301) != v302 || HIDWORD(v315) != v316) && v237 )
              {
                goto LABEL_62;
              }
              if ( v244 )
                goto LABEL_62;
              *(_BYTE *)(a1 + 153) = 1;
              v229 = v238 == 0;
              v258 = *v114;
              v115 = (__int64 *)sub_1643330(**v114);
              v256 = sub_1646BA0(v115, 0);
              v118 = v314;
              if ( v314 == v313 )
                v263 = &v314[HIDWORD(v315)];
              else
                v263 = &v314[(unsigned int)v315];
              if ( v314 == v263 )
                goto LABEL_225;
              while ( 1 )
              {
                v119 = *v118;
                if ( (unsigned __int64)*v118 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v263 == ++v118 )
                  goto LABEL_225;
              }
              if ( v263 == v118 )
              {
LABEL_225:
                v120 = v300;
                if ( v300 == v299 )
                {
LABEL_304:
                  v264 = &v120[HIDWORD(v301)];
LABEL_227:
                  if ( v120 != v264 )
                  {
                    while ( 1 )
                    {
                      v117 = *v120;
                      if ( (unsigned __int64)*v120 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v264 == ++v120 )
                        goto LABEL_230;
                    }
                    if ( v264 != v120 )
                    {
                      v245 = v120;
                      v241 = (__int64)v114;
                      v141 = *v120;
                      do
                      {
                        v142 = v241;
                        if ( v258 != (_QWORD **)v256 )
                        {
                          v279[0] = 257;
                          v143 = sub_1648A60(56, 1u);
                          v142 = (__int64)v143;
                          if ( v143 )
                            sub_15FD590((__int64)v143, v241, v256, (__int64)&v277, v141);
                        }
                        v265 = v142;
                        v144 = *(_QWORD *)(a1 + 248);
                        if ( !v144 )
                        {
                          v156 = **(__int64 ***)(a1 + 232);
                          v157 = (__int64 *)sub_1643330(v156);
                          v274 = (__int64 *)sub_1646BA0(v157, 0);
                          v277 = 0;
                          v158 = sub_1563AB0((__int64 *)&v277, v156, -1, 30);
                          v159 = (__int64 *)sub_1643270(v156);
                          v160 = sub_1644EA0(v159, &v274, 1, 0);
                          v161 = sub_1632080(*(_QWORD *)(a1 + 232), (__int64)"objc_release", 12, v160, v158);
                          *(_QWORD *)(a1 + 248) = v161;
                          v144 = v161;
                        }
                        v279[0] = 257;
                        v145 = *(_QWORD *)(*(_QWORD *)v144 + 24LL);
                        v146 = sub_1648AB0(72, 2u, 0);
                        v147 = (__int64)v146;
                        if ( v146 )
                        {
                          v249 = v146;
                          sub_15F1EA0((__int64)v146, **(_QWORD **)(v145 + 16), 54, (__int64)(v146 - 6), 2, v141);
                          *(_QWORD *)(v147 + 56) = 0;
                          sub_15F5B40(v147, v145, v144, &v265, 1, (__int64)&v277, 0, 0);
                          v148 = (__int64)v249;
                        }
                        else
                        {
                          v148 = 0;
                        }
                        v149 = v305;
                        if ( v305 )
                        {
                          if ( *(_BYTE *)(a1 + 324) )
                          {
                            v150 = *(_DWORD *)(a1 + 320);
                          }
                          else
                          {
                            v251 = v148;
                            v162 = sub_1602B80(**(__int64 ***)(a1 + 312), "clang.imprecise_release", 0x17u);
                            v148 = v251;
                            v150 = v162;
                            if ( *(_BYTE *)(a1 + 324) )
                            {
                              *(_DWORD *)(a1 + 320) = v162;
                            }
                            else
                            {
                              *(_DWORD *)(a1 + 320) = v162;
                              *(_BYTE *)(a1 + 324) = 1;
                            }
                          }
                          v250 = v148;
                          sub_1625C10(v148, v150, v149);
                          v148 = v250;
                        }
                        v277 = *(_WORD **)(v147 + 56);
                        v151 = (__int64 *)sub_16498A0(v148);
                        v152 = sub_1563AB0((__int64 *)&v277, v151, -1, 30);
                        v153 = HIBYTE(v304) == 0;
                        v277 = (_WORD *)v152;
                        *(_QWORD *)(v147 + 56) = v152;
                        if ( !v153 )
                          *(_WORD *)(v147 + 18) = *(_WORD *)(v147 + 18) & 0xFFFC | 1;
                        v154 = v245 + 1;
                        if ( v245 + 1 == v264 )
                          break;
                        v141 = *v154;
                        for ( j = v245 + 1; (unsigned __int64)*v154 >= 0xFFFFFFFFFFFFFFFELL; j = v154 )
                        {
                          if ( v264 == ++v154 )
                            goto LABEL_230;
                          v141 = *v154;
                        }
                        v245 = j;
                      }
                      while ( v264 != j );
                    }
                  }
LABEL_230:
                  v121 = v294;
                  if ( v294 == v293 )
                    v122 = &v294[HIDWORD(v295)];
                  else
                    v122 = &v294[(unsigned int)v295];
                  if ( v294 != v122 )
                  {
                    while ( 1 )
                    {
                      v123 = *v121;
                      v124 = v121;
                      if ( (unsigned __int64)*v121 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v122 == ++v121 )
                        goto LABEL_235;
                    }
LABEL_256:
                    if ( v122 == v124 )
                      goto LABEL_235;
                    if ( v283 )
                    {
                      v116 = v283 - 1;
                      v136 = (v283 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
                      v137 = (__int64 *)(v281 + 16LL * v136);
                      v138 = *v137;
                      if ( v123 == *v137 )
                        goto LABEL_259;
                      v178 = 1;
                      if ( v138 != -8 )
                      {
                        while ( 1 )
                        {
                          LODWORD(v117) = v178 + 1;
                          v136 = v116 & (v178 + v136);
                          v137 = (__int64 *)(v281 + 16LL * v136);
                          v179 = *v137;
                          if ( *v137 == v123 )
                            break;
                          v178 = v117;
                          if ( v179 == -8 )
                            goto LABEL_261;
                        }
LABEL_259:
                        if ( v137 != (__int64 *)(v281 + 16LL * v283) )
                        {
                          v284[18 * v137[1]] = 0;
                          *v137 = -16;
                          LODWORD(v282) = v282 - 1;
                          ++HIDWORD(v282);
                        }
                      }
                    }
LABEL_261:
                    v139 = (unsigned int)v288;
                    if ( (unsigned int)v288 >= HIDWORD(v288) )
                    {
                      sub_16CD150((__int64)&v287, v289, 0, 8, v116, v117);
                      v139 = (unsigned int)v288;
                    }
                    *(_QWORD *)&v287[8 * v139] = v123;
                    v140 = v124 + 1;
                    LODWORD(v288) = v288 + 1;
                    if ( v124 + 1 == v122 )
                      goto LABEL_235;
                    while ( 1 )
                    {
                      v123 = *v140;
                      v124 = v140;
                      if ( (unsigned __int64)*v140 < 0xFFFFFFFFFFFFFFFELL )
                        goto LABEL_256;
                      if ( v122 == ++v140 )
                      {
                        v125 = v308;
                        if ( v308 != v307 )
                          goto LABEL_236;
                        goto LABEL_267;
                      }
                    }
                  }
LABEL_235:
                  v125 = v308;
                  if ( v308 == v307 )
LABEL_267:
                    v126 = &v125[HIDWORD(v309)];
                  else
LABEL_236:
                    v126 = &v125[(unsigned int)v309];
                  if ( v125 != v126 )
                  {
                    while ( 1 )
                    {
                      v127 = *v125;
                      v128 = v125;
                      if ( (unsigned __int64)*v125 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v126 == ++v125 )
                        goto LABEL_62;
                    }
LABEL_241:
                    if ( v128 == v126 )
                      goto LABEL_62;
                    if ( v269 )
                    {
                      v129 = 1;
                      v130 = (v269 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
                      v131 = &v267[18 * v130];
                      if ( *v131 == v127 )
                        goto LABEL_244;
                      if ( *v131 != -8 )
                      {
                        while ( 1 )
                        {
                          v116 = v129 + 1;
                          v130 = (v269 - 1) & (v129 + v130);
                          v131 = &v267[18 * v130];
                          if ( *v131 == v127 )
                            break;
                          ++v129;
                          if ( *v131 == -8 )
                            goto LABEL_249;
                        }
LABEL_244:
                        v132 = v131[12];
                        if ( v132 != v131[11] )
                          _libc_free(v132);
                        v133 = v131[5];
                        if ( v133 != v131[4] )
                          _libc_free(v133);
                        *v131 = -16;
                        LODWORD(v268) = v268 - 1;
                        ++HIDWORD(v268);
                      }
                    }
LABEL_249:
                    v134 = (unsigned int)v288;
                    if ( (unsigned int)v288 >= HIDWORD(v288) )
                    {
                      sub_16CD150((__int64)&v287, v289, 0, 8, v116, v117);
                      v134 = (unsigned int)v288;
                    }
                    *(_QWORD *)&v287[8 * v134] = v127;
                    v135 = v128 + 1;
                    LODWORD(v288) = v288 + 1;
                    if ( v128 + 1 == v126 )
                      goto LABEL_62;
                    while ( 1 )
                    {
                      v127 = *v135;
                      v128 = v135;
                      if ( (unsigned __int64)*v135 < 0xFFFFFFFFFFFFFFFELL )
                        goto LABEL_241;
                      if ( v126 == ++v135 )
                        goto LABEL_62;
                    }
                  }
                  goto LABEL_62;
                }
              }
              else
              {
                v242 = (__int64)v114;
                v246 = v118;
                do
                {
                  v163 = v242;
                  if ( v258 != (_QWORD **)v256 )
                  {
                    v279[0] = 257;
                    v164 = sub_1648A60(56, 1u);
                    v163 = (__int64)v164;
                    if ( v164 )
                      sub_15FD590((__int64)v164, v242, v256, (__int64)&v277, v119);
                  }
                  v265 = v163;
                  v165 = *(_QWORD *)(a1 + 256);
                  if ( !v165 )
                  {
                    v172 = **(__int64 ***)(a1 + 232);
                    v173 = (__int64 *)sub_1643330(v172);
                    v274 = (__int64 *)sub_1646BA0(v173, 0);
                    v174 = sub_1644EA0(v274, &v274, 1, 0);
                    v277 = 0;
                    v175 = sub_1563AB0((__int64 *)&v277, v172, -1, 30);
                    v176 = *(_QWORD *)(a1 + 232);
                    v277 = (_WORD *)v175;
                    v177 = sub_1632080(v176, (__int64)"objc_retain", 11, v174, v175);
                    *(_QWORD *)(a1 + 256) = v177;
                    v165 = v177;
                  }
                  v279[0] = 257;
                  v166 = *(_QWORD *)(*(_QWORD *)v165 + 24LL);
                  v167 = sub_1648AB0(72, 2u, 0);
                  v168 = (__int64)v167;
                  if ( v167 )
                  {
                    v252 = v167;
                    sub_15F1EA0((__int64)v167, **(_QWORD **)(v166 + 16), 54, (__int64)(v167 - 6), 2, v119);
                    *(_QWORD *)(v168 + 56) = 0;
                    sub_15F5B40(v168, v166, v165, &v265, 1, (__int64)&v277, 0, 0);
                    v169 = (__int64)v252;
                  }
                  else
                  {
                    v169 = 0;
                  }
                  v277 = *(_WORD **)(v168 + 56);
                  v170 = (__int64 *)sub_16498A0(v169);
                  v277 = (_WORD *)sub_1563AB0((__int64 *)&v277, v170, -1, 30);
                  *(_QWORD *)(v168 + 56) = v277;
                  *(_WORD *)(v168 + 18) = *(_WORD *)(v168 + 18) & 0xFFFC | 1;
                  v171 = v246 + 1;
                  if ( v246 + 1 == v263 )
                    break;
                  while ( 1 )
                  {
                    v119 = *v171;
                    if ( (unsigned __int64)*v171 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v263 == ++v171 )
                      goto LABEL_303;
                  }
                  v246 = v171;
                }
                while ( v263 != v171 );
LABEL_303:
                v114 = (_QWORD ***)v242;
                v120 = v300;
                if ( v300 == v299 )
                  goto LABEL_304;
              }
              v264 = &v120[(unsigned int)v301];
              goto LABEL_227;
            }
            v222 = 1;
            while ( v35 != -8 )
            {
              v223 = v222 + 1;
              v33 = (v269 - 1) & (v222 + v33);
              v34 = &v267[18 * v33];
              v35 = *v34;
              if ( v32 == *v34 )
                goto LABEL_32;
              v222 = v223;
            }
          }
          break;
        }
        v34 = &v267[18 * v269];
        goto LABEL_32;
      }
      break;
    }
    v257 = v27;
    while ( 1 )
    {
      if ( !v269 )
        goto LABEL_58;
      v41 = (v269 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v42 = &v267[18 * v41];
      v43 = 1;
      v44 = *v42;
      if ( *v42 != v30 )
        break;
LABEL_47:
      if ( v42 == &v267[18 * v269] )
        goto LABEL_58;
      v45 = (_QWORD *)v42[5];
      v46 = (_QWORD *)v42[4];
      if ( v45 == v46 )
      {
        v47 = &v46[*((unsigned int *)v42 + 13)];
        if ( v46 == v47 )
        {
          v106 = (_QWORD *)v42[4];
        }
        else
        {
          do
          {
            if ( v22 == *v46 )
              break;
            ++v46;
          }
          while ( v47 != v46 );
          v106 = v47;
        }
      }
      else
      {
        v47 = &v45[*((unsigned int *)v42 + 12)];
        v46 = sub_16CC9F0((__int64)(v42 + 3), v22);
        if ( v22 == *v46 )
        {
          v88 = v42[5];
          if ( v88 == v42[4] )
            v89 = *((unsigned int *)v42 + 13);
          else
            v89 = *((unsigned int *)v42 + 12);
          v106 = (_QWORD *)(v88 + 8 * v89);
        }
        else
        {
          v48 = v42[5];
          if ( v48 != v42[4] )
          {
            v46 = (_QWORD *)(v48 + 8LL * *((unsigned int *)v42 + 12));
            goto LABEL_52;
          }
          v46 = (_QWORD *)(v48 + 8LL * *((unsigned int *)v42 + 13));
          v106 = v46;
        }
      }
      while ( v106 != v46 && *v46 >= 0xFFFFFFFFFFFFFFFELL )
        ++v46;
LABEL_52:
      if ( v47 == v46 )
        goto LABEL_58;
      v49 = v307;
      if ( v308 == v307 )
      {
        v86 = &v307[HIDWORD(v309)];
        if ( v307 != v86 )
        {
          v87 = 0;
          do
          {
            if ( *v49 == v30 )
              goto LABEL_129;
            if ( *v49 == -2 )
              v87 = v49;
            ++v49;
          }
          while ( v86 != v49 );
          if ( v87 )
          {
            *v87 = v30;
            --v310;
            ++v306;
            goto LABEL_55;
          }
        }
        if ( HIDWORD(v309) < (unsigned int)v309 )
        {
          ++HIDWORD(v309);
          *v86 = v30;
          ++v306;
          goto LABEL_55;
        }
      }
      sub_16CCBA0((__int64)&v306, v30);
      if ( !v50 )
        goto LABEL_129;
LABEL_55:
      v265 = *(_QWORD *)(v30 + 40);
      v51 = sub_18CDCF0((__int64)&v270, &v265);
      v54 = *((unsigned int *)v51 + 2);
      if ( (_DWORD)v54 == -1 )
        goto LABEL_58;
      v55 = *((unsigned int *)v51 + 3);
      if ( (_DWORD)v55 == -1 )
        goto LABEL_58;
      v56 = v55 * v54;
      if ( HIDWORD(v56) || v56 == 0xFFFFFFFF )
        goto LABEL_58;
      v244 -= v56;
      if ( v254 )
      {
        v305 = v42[2];
        HIBYTE(v304) = *((_BYTE *)v42 + 9);
LABEL_126:
        if ( v247 )
          goto LABEL_127;
        goto LABEL_159;
      }
      if ( v305 != v42[2] )
        v305 = 0;
      if ( HIBYTE(v304) == *((_BYTE *)v42 + 9) )
        goto LABEL_126;
      HIBYTE(v304) = 0;
      if ( v247 )
        goto LABEL_127;
LABEL_159:
      v90 = (__int64 *)v42[12];
      if ( v90 == (__int64 *)v42[11] )
        v91 = *((unsigned int *)v42 + 27);
      else
        v91 = *((unsigned int *)v42 + 26);
      v92 = &v90[v91];
      if ( v90 == v92 )
        goto LABEL_127;
      while ( 1 )
      {
        v93 = *v90;
        v94 = v90;
        if ( (unsigned __int64)*v90 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v92 == ++v90 )
          goto LABEL_127;
      }
      if ( v92 == v90 )
      {
LABEL_127:
        v82 = (unsigned int)v278;
        if ( (unsigned int)v278 >= HIDWORD(v278) )
          goto LABEL_177;
        goto LABEL_128;
      }
      v255 = v22;
      v95 = v313;
      v96 = v92;
      v97 = v93;
      if ( v314 != v313 )
        goto LABEL_167;
      while ( 2 )
      {
        v52 = &v95[HIDWORD(v315)];
        v53 = HIDWORD(v315);
        if ( v95 == v52 )
          goto LABEL_189;
        v104 = 0;
        do
        {
          if ( v97 == *v95 )
            goto LABEL_173;
          if ( *v95 == -2 )
            v104 = v95;
          ++v95;
        }
        while ( v52 != v95 );
        if ( !v104 )
        {
LABEL_189:
          if ( HIDWORD(v315) >= (unsigned int)v315 )
            goto LABEL_167;
          ++HIDWORD(v315);
          *v52 = v97;
          ++v312;
        }
        else
        {
          *v104 = v97;
          --v316;
          ++v312;
        }
LABEL_168:
        v265 = *(_QWORD *)(v97 + 40);
        v99 = sub_18CDCF0((__int64)&v270, &v265);
        v100 = *((unsigned int *)v99 + 2);
        if ( (_DWORD)v100 == -1 )
          goto LABEL_58;
        v101 = *((unsigned int *)v99 + 3);
        if ( (_DWORD)v101 == -1 )
          goto LABEL_58;
        v102 = v101 * v100;
        if ( HIDWORD(v102) || v102 == 0xFFFFFFFF )
          goto LABEL_58;
        v248 -= v102;
LABEL_173:
        v103 = v94 + 1;
        if ( v94 + 1 != v96 )
        {
          while ( 1 )
          {
            v97 = *v103;
            v94 = v103;
            if ( (unsigned __int64)*v103 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v96 == ++v103 )
              goto LABEL_176;
          }
          if ( v96 != v103 )
          {
            v95 = v313;
            if ( v314 == v313 )
              continue;
LABEL_167:
            sub_16CCBA0((__int64)&v312, v97);
            if ( !v98 )
              goto LABEL_173;
            goto LABEL_168;
          }
        }
        break;
      }
LABEL_176:
      v22 = v255;
      v82 = (unsigned int)v278;
      if ( (unsigned int)v278 >= HIDWORD(v278) )
      {
LABEL_177:
        sub_16CD150((__int64)&v277, v279, 0, 8, (int)v52, v53);
        v82 = (unsigned int)v278;
      }
LABEL_128:
      v254 = 0;
      *(_QWORD *)&v277[4 * v82] = v30;
      LODWORD(v278) = v278 + 1;
LABEL_129:
      v83 = v257 + 1;
      if ( v257 + 1 == v261 )
        goto LABEL_27;
      v84 = v257 + 1;
      v30 = *v83;
      v85 = v257 + 1;
      if ( (unsigned __int64)*v83 >= 0xFFFFFFFFFFFFFFFELL )
      {
        do
        {
          if ( v261 == ++v84 )
            goto LABEL_27;
          v30 = *v84;
          v85 = v84;
        }
        while ( (unsigned __int64)*v84 >= 0xFFFFFFFFFFFFFFFELL );
      }
      v257 = v85;
      if ( v261 == v85 )
        goto LABEL_27;
    }
    while ( v44 != -8 )
    {
      v41 = (v269 - 1) & (v43 + v41);
      v42 = &v267[18 * v41];
      v44 = *v42;
      if ( *v42 == v30 )
        goto LABEL_47;
      ++v43;
    }
LABEL_58:
    if ( v277 != v279 )
      _libc_free((unsigned __int64)v277);
    if ( v274 != v276 )
      _libc_free((unsigned __int64)v274);
LABEL_62:
    if ( v314 != v313 )
      _libc_free((unsigned __int64)v314);
    if ( v308 != v307 )
      _libc_free((unsigned __int64)v308);
    if ( v300 != v299 )
      _libc_free((unsigned __int64)v300);
    if ( v294 != v293 )
      _libc_free((unsigned __int64)v294);
LABEL_70:
    v243 += 18;
  }
  while ( v231 != v243 );
  while ( 1 )
  {
    v57 = v288;
    if ( !(_DWORD)v288 )
      break;
    while ( 1 )
    {
      v58 = *(_QWORD *)&v287[8 * v57 - 8];
      LODWORD(v288) = v57 - 1;
      v59 = *(_QWORD *)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF));
      if ( !*(_QWORD *)(v58 + 8) )
        break;
      sub_164D160(v58, *(_QWORD *)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF)), a3, a4, a5, a6, v10, v11, a9, a10);
      sub_15F20C0((_QWORD *)v58);
      v57 = v288;
      if ( !(_DWORD)v288 )
        goto LABEL_74;
    }
    sub_15F20C0((_QWORD *)v58);
    sub_1AEB370(v59, 0);
  }
LABEL_74:
  v60 = v228;
  LOBYTE(v60) = v229 & v228;
  if ( v287 != v289 )
    _libc_free((unsigned __int64)v287);
LABEL_76:
  if ( v273 )
  {
    v61 = v271;
    v62 = &v271[24 * v273];
    do
    {
      if ( *v61 != -16 && *v61 != -8 )
      {
        v63 = v61[20];
        if ( (_QWORD *)v63 != v61 + 22 )
          _libc_free(v63);
        v64 = v61[16];
        if ( (_QWORD *)v64 != v61 + 18 )
          _libc_free(v64);
        v65 = (_QWORD *)v61[14];
        v66 = (_QWORD *)v61[13];
        if ( v65 != v66 )
        {
          do
          {
            v67 = v66[13];
            if ( v67 != v66[12] )
              _libc_free(v67);
            v68 = v66[6];
            if ( v68 != v66[5] )
              _libc_free(v68);
            v66 += 19;
          }
          while ( v65 != v66 );
          v66 = (_QWORD *)v61[13];
        }
        if ( v66 )
          j_j___libc_free_0(v66, v61[15] - (_QWORD)v66);
        j___libc_free_0(v61[10]);
        v69 = (_QWORD *)v61[7];
        v70 = (_QWORD *)v61[6];
        if ( v69 != v70 )
        {
          do
          {
            v71 = v70[13];
            if ( v71 != v70[12] )
              _libc_free(v71);
            v72 = v70[6];
            if ( v72 != v70[5] )
              _libc_free(v72);
            v70 += 19;
          }
          while ( v69 != v70 );
          v70 = (_QWORD *)v61[6];
        }
        if ( v70 )
          j_j___libc_free_0(v70, v61[8] - (_QWORD)v70);
        j___libc_free_0(v61[3]);
      }
      v61 += 24;
    }
    while ( v62 != v61 );
  }
  j___libc_free_0(v271);
  v73 = v285;
  v74 = v284;
  if ( v285 != v284 )
  {
    do
    {
      v75 = v74[12];
      if ( v75 != v74[11] )
        _libc_free(v75);
      v76 = v74[5];
      if ( v76 != v74[4] )
        _libc_free(v76);
      v74 += 18;
    }
    while ( v73 != v74 );
    v74 = v284;
  }
  if ( v74 )
    j_j___libc_free_0(v74, v286 - (_QWORD)v74);
  j___libc_free_0(v281);
  if ( v269 )
  {
    v77 = v267;
    v78 = &v267[18 * v269];
    do
    {
      if ( *v77 != -16 && *v77 != -8 )
      {
        v79 = v77[12];
        if ( v79 != v77[11] )
          _libc_free(v79);
        v80 = v77[5];
        if ( v80 != v77[4] )
          _libc_free(v80);
      }
      v77 += 18;
    }
    while ( v78 != v77 );
  }
  j___libc_free_0(v267);
  return v60;
}
