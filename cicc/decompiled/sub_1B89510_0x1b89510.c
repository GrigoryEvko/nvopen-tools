// Function: sub_1B89510
// Address: 0x1b89510
//
__int64 __fastcall sub_1B89510(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rcx
  __int64 **v12; // rdx
  __int64 *v13; // rax
  __int64 **v14; // r10
  __int64 *v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // r8d
  __int64 v19; // r9
  __int64 *v20; // r13
  __int64 v21; // r8
  char v23; // cl
  __int64 *v24; // rax
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v27; // rax
  unsigned __int64 v28; // r12
  __int64 *v29; // rbx
  __int64 v30; // r14
  unsigned int v31; // esi
  __int64 v32; // rcx
  int v33; // eax
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rdi
  int v39; // ecx
  __int64 v40; // rdx
  unsigned __int64 v41; // r15
  int v42; // eax
  __int64 v43; // rdx
  _QWORD *v44; // rax
  _QWORD *i; // rdx
  __int64 v46; // r13
  __int64 v47; // rax
  unsigned __int64 v48; // r13
  __int64 v49; // r15
  __int64 v50; // r12
  int v51; // r8d
  char v52; // r9
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // eax
  __int64 v56; // rdx
  _QWORD *v57; // rax
  _QWORD *m; // rdx
  unsigned __int64 v59; // r12
  __int64 v60; // r14
  _BOOL4 v61; // r13d
  double v62; // xmm4_8
  double v63; // xmm5_8
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  __int64 *v66; // rsi
  unsigned int v67; // eax
  _DWORD *v68; // rsi
  _DWORD *v69; // rsi
  __int64 *v70; // r12
  int v71; // r8d
  char v72; // r9
  __int64 v73; // rax
  char v74; // cl
  __int64 v75; // rax
  int v76; // eax
  char v77; // al
  _QWORD **v78; // rax
  _QWORD **v79; // r12
  _QWORD *v80; // rdi
  _QWORD *v81; // r13
  bool v82; // dl
  _QWORD *v83; // rbx
  unsigned int v84; // ecx
  _QWORD *v85; // rdi
  unsigned int v86; // eax
  int v87; // eax
  unsigned __int64 v88; // rax
  unsigned __int64 v89; // rax
  int v90; // ebx
  __int64 v91; // r12
  _QWORD *v92; // rax
  __int64 v93; // rdx
  _QWORD *n; // rdx
  unsigned int v95; // ecx
  _QWORD *v96; // rdi
  unsigned int v97; // eax
  int v98; // eax
  unsigned __int64 v99; // rax
  unsigned __int64 v100; // rax
  int v101; // ebx
  __int64 v102; // r12
  _QWORD *v103; // rax
  __int64 v104; // rdx
  _QWORD *v105; // rdx
  char *v106; // rdx
  _QWORD *v107; // r12
  unsigned int v108; // ebx
  __int64 v109; // rax
  __int64 v110; // rbx
  char *v111; // r12
  char *v112; // rsi
  unsigned __int64 v113; // rax
  char *v114; // rdi
  _DWORD *v115; // rdx
  int v116; // eax
  unsigned int v117; // edx
  unsigned int v118; // esi
  _QWORD *v119; // rdx
  _QWORD *v120; // rax
  __int64 v121; // rax
  unsigned int v122; // edx
  _QWORD *v123; // rdi
  unsigned int v124; // eax
  __int64 v125; // rax
  unsigned __int64 v126; // rax
  unsigned __int64 v127; // rax
  int v128; // ebx
  __int64 v129; // r12
  _QWORD *v130; // rax
  __int64 v131; // rdx
  unsigned int v132; // esi
  _QWORD *ii; // rdx
  __int64 *v134; // r13
  __int64 v135; // rax
  __int64 *v136; // r12
  __int64 v137; // rbx
  __int64 v138; // rsi
  __int64 v139; // rax
  __int64 v140; // rsi
  __int64 v141; // r15
  __int64 v142; // rax
  unsigned int v143; // edi
  __int64 v144; // rax
  char v145; // cl
  char v146; // si
  __int64 v147; // rdx
  __int64 v148; // r10
  __int64 v149; // rbx
  __int64 v150; // rdx
  __int64 v151; // rdi
  __int64 v152; // r11
  __int64 v153; // rdi
  __int64 v154; // rdx
  __int64 v155; // rsi
  __int64 v156; // rcx
  __int64 v157; // rax
  _QWORD *v158; // rcx
  _QWORD *v159; // rbx
  int v160; // ebx
  _QWORD *v161; // rsi
  _QWORD *v162; // rax
  __int64 v163; // rcx
  _QWORD *v164; // r10
  __int64 v165; // rcx
  unsigned __int64 v166; // rdx
  __int64 v167; // rax
  __int64 v168; // rax
  int v169; // r8d
  __int64 v170; // r9
  __int64 v171; // r10
  __int64 v172; // rax
  __int64 v173; // rax
  __int64 *v174; // rcx
  _QWORD *v175; // r10
  unsigned __int64 v176; // rdx
  __int64 v177; // rax
  __int64 *v178; // rbx
  __int64 v179; // rax
  char *v180; // rbx
  char *v181; // rdi
  _QWORD *v182; // rax
  _QWORD *v183; // rax
  _QWORD *v184; // rax
  __int64 v185; // rcx
  int v186; // ecx
  __int64 *v187; // rax
  _QWORD *v188; // rdi
  unsigned int v189; // eax
  int v190; // eax
  unsigned __int64 v191; // rax
  unsigned __int64 v192; // rax
  int v193; // ebx
  __int64 v194; // r12
  __int64 v195; // rax
  __int64 v196; // r12
  unsigned __int64 v197; // r13
  unsigned __int64 v198; // rax
  unsigned int v199; // esi
  _QWORD *v200; // rax
  _QWORD *k; // rdx
  __int64 v202; // rax
  __int64 v203; // r9
  _QWORD *v204; // r10
  __int64 v205; // rax
  _QWORD *v206; // r9
  unsigned __int64 v207; // rdx
  __int32 v208; // eax
  _QWORD *v209; // rcx
  __int64 v210; // rax
  __int64 v211; // rax
  __int64 v212; // rax
  __int64 v213; // rax
  __int64 *v214; // r13
  int v215; // r8d
  char v216; // r9
  __int64 v217; // rax
  __int64 *v218; // rbx
  __int64 v219; // rax
  __int64 v220; // rax
  _QWORD *v221; // rax
  __int64 v222; // [rsp+0h] [rbp-9C0h]
  _QWORD *v223; // [rsp+8h] [rbp-9B8h]
  _QWORD *v224; // [rsp+8h] [rbp-9B8h]
  _QWORD *v225; // [rsp+8h] [rbp-9B8h]
  __int64 v226; // [rsp+10h] [rbp-9B0h]
  __int64 v227; // [rsp+10h] [rbp-9B0h]
  unsigned __int64 v228; // [rsp+10h] [rbp-9B0h]
  __int64 v229; // [rsp+10h] [rbp-9B0h]
  _QWORD *v230; // [rsp+10h] [rbp-9B0h]
  _QWORD *v231; // [rsp+10h] [rbp-9B0h]
  unsigned __int64 v232; // [rsp+18h] [rbp-9A8h]
  __int64 v233; // [rsp+28h] [rbp-998h]
  __int64 v234; // [rsp+30h] [rbp-990h]
  __int64 v235; // [rsp+38h] [rbp-988h]
  _DWORD *v236; // [rsp+40h] [rbp-980h]
  __int64 v237; // [rsp+40h] [rbp-980h]
  __int64 v238; // [rsp+50h] [rbp-970h]
  __int64 v239; // [rsp+58h] [rbp-968h]
  unsigned int v240; // [rsp+60h] [rbp-960h]
  unsigned int v241; // [rsp+64h] [rbp-95Ch]
  unsigned __int8 v242; // [rsp+7Bh] [rbp-945h]
  unsigned int v243; // [rsp+7Ch] [rbp-944h]
  __int64 v244; // [rsp+88h] [rbp-938h]
  _QWORD *v245; // [rsp+90h] [rbp-930h]
  __int64 v246; // [rsp+90h] [rbp-930h]
  __int64 v247; // [rsp+90h] [rbp-930h]
  unsigned __int64 v248; // [rsp+90h] [rbp-930h]
  int v249; // [rsp+98h] [rbp-928h]
  unsigned int v250; // [rsp+98h] [rbp-928h]
  __int64 v251; // [rsp+98h] [rbp-928h]
  __int64 v252; // [rsp+98h] [rbp-928h]
  int v253; // [rsp+A8h] [rbp-918h]
  __int64 *v254; // [rsp+A8h] [rbp-918h]
  int v255; // [rsp+B0h] [rbp-910h]
  char v256; // [rsp+B0h] [rbp-910h]
  __int64 v257; // [rsp+B0h] [rbp-910h]
  int v258; // [rsp+B8h] [rbp-908h]
  __int64 v259; // [rsp+B8h] [rbp-908h]
  __int64 v260; // [rsp+B8h] [rbp-908h]
  int v261; // [rsp+B8h] [rbp-908h]
  int *v262; // [rsp+C0h] [rbp-900h]
  __int64 v263; // [rsp+C0h] [rbp-900h]
  _QWORD *v264; // [rsp+C0h] [rbp-900h]
  __int64 v265; // [rsp+C0h] [rbp-900h]
  char **src; // [rsp+C8h] [rbp-8F8h]
  int *srca; // [rsp+C8h] [rbp-8F8h]
  char *srcb; // [rsp+C8h] [rbp-8F8h]
  unsigned int srcc; // [rsp+C8h] [rbp-8F8h]
  char *srcd; // [rsp+C8h] [rbp-8F8h]
  void *srce; // [rsp+C8h] [rbp-8F8h]
  int v272; // [rsp+DCh] [rbp-8E4h] BYREF
  __m128i v273; // [rsp+E0h] [rbp-8E0h] BYREF
  unsigned __int64 *v274; // [rsp+F0h] [rbp-8D0h]
  __int64 v275; // [rsp+F8h] [rbp-8C8h]
  __m128i v276; // [rsp+100h] [rbp-8C0h] BYREF
  unsigned __int64 *v277; // [rsp+110h] [rbp-8B0h]
  __int64 v278; // [rsp+118h] [rbp-8A8h]
  __m128i v279; // [rsp+120h] [rbp-8A0h] BYREF
  unsigned __int64 *v280; // [rsp+130h] [rbp-890h]
  __int64 v281; // [rsp+138h] [rbp-888h]
  __m128i v282; // [rsp+140h] [rbp-880h] BYREF
  unsigned __int64 v283; // [rsp+160h] [rbp-860h] BYREF
  __int64 v284; // [rsp+168h] [rbp-858h]
  __int64 v285; // [rsp+170h] [rbp-850h] BYREF
  unsigned int v286; // [rsp+178h] [rbp-848h]
  _DWORD *v287; // [rsp+1B0h] [rbp-810h] BYREF
  __int64 v288; // [rsp+1B8h] [rbp-808h]
  _BYTE v289[128]; // [rsp+1C0h] [rbp-800h] BYREF
  __int64 *v290; // [rsp+240h] [rbp-780h] BYREF
  __int64 v291; // [rsp+248h] [rbp-778h]
  _BYTE v292[128]; // [rsp+250h] [rbp-770h] BYREF
  __m128i v293; // [rsp+2D0h] [rbp-6F0h] BYREF
  unsigned __int64 *v294; // [rsp+2E0h] [rbp-6E0h] BYREF
  __int64 v295; // [rsp+2E8h] [rbp-6D8h]
  int v296; // [rsp+2F0h] [rbp-6D0h]
  _BYTE v297[136]; // [rsp+2F8h] [rbp-6C8h] BYREF
  _BYTE *v298; // [rsp+380h] [rbp-640h] BYREF
  __int64 v299; // [rsp+388h] [rbp-638h]
  _BYTE v300[512]; // [rsp+390h] [rbp-630h] BYREF
  char *j; // [rsp+590h] [rbp-430h] BYREF
  __int64 v302; // [rsp+598h] [rbp-428h]
  _BYTE v303[1056]; // [rsp+5A0h] [rbp-420h] BYREF

  v244 = *(_QWORD *)(a2 + 32);
  v235 = *(_QWORD *)(a2 + 40);
  if ( v244 != v235 )
  {
    v242 = 0;
    while ( 1 )
    {
      v11 = *(unsigned int *)(v244 + 16);
      v241 = v11;
      if ( (unsigned int)v11 <= 1 || (unsigned int)v11 > dword_4FB7B00 )
        goto LABEL_9;
      v12 = *(__int64 ***)(v244 + 8);
      v13 = *v12;
      v14 = &v12[v11];
      v15 = (__int64 *)(v12 + 1);
      v16 = **v12;
      v17 = *(unsigned __int8 *)(v16 + 8);
      v18 = *(unsigned __int8 *)(v16 + 8);
      LOBYTE(v19) = (_BYTE)v17 == 16;
      while ( 1 )
      {
        LOBYTE(v18) = (_BYTE)v18 == 16;
        if ( (_BYTE)v19 != (_BYTE)v18 )
          goto LABEL_9;
        v20 = v15;
        if ( v14 == (__int64 **)v15 )
          break;
        v21 = *v15++;
        v18 = *(unsigned __int8 *)(*(_QWORD *)v21 + 8LL);
      }
      if ( *(_DWORD *)(a1 + 120) > 2u )
        goto LABEL_24;
      v23 = *((_BYTE *)v13 + 16);
      if ( v23 != 54 )
        break;
LABEL_17:
      v25 = *(_QWORD *)(a1 + 40);
      v26 = 1;
      while ( 2 )
      {
        switch ( v17 )
        {
          case 0LL:
          case 8LL:
          case 10LL:
          case 12LL:
          case 16LL:
            v195 = *(_QWORD *)(v16 + 32);
            v16 = *(_QWORD *)(v16 + 24);
            v26 *= v195;
            v17 = *(unsigned __int8 *)(v16 + 8);
            continue;
          case 1LL:
            v27 = 16;
            goto LABEL_20;
          case 2LL:
            v27 = 32;
            goto LABEL_20;
          case 3LL:
          case 9LL:
            v27 = 64;
            goto LABEL_20;
          case 4LL:
            v27 = 80;
            goto LABEL_20;
          case 5LL:
          case 6LL:
            v27 = 128;
            goto LABEL_20;
          case 7LL:
            v199 = 0;
            goto LABEL_309;
          case 11LL:
            v27 = *(_DWORD *)(v16 + 8) >> 8;
            goto LABEL_20;
          case 13LL:
            v198 = *(_QWORD *)sub_15A9930(*(_QWORD *)(a1 + 40), v16);
            goto LABEL_302;
          case 14LL:
            v196 = *(_QWORD *)(v16 + 32);
            srce = *(void **)(v16 + 24);
            v197 = (unsigned int)sub_15A9FE0(*(_QWORD *)(a1 + 40), (__int64)srce);
            v198 = v197 * v196 * ((v197 + ((unsigned __int64)(sub_127FA20(v25, (__int64)srce) + 7) >> 3) - 1) / v197);
LABEL_302:
            v27 = 8 * v198;
            v12 = *(__int64 ***)(v244 + 8);
            v20 = (__int64 *)&v12[*(unsigned int *)(v244 + 16)];
            break;
          case 15LL:
            v199 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_309:
            v27 = 8 * (unsigned int)sub_15A9520(*(_QWORD *)(a1 + 40), v199);
            v12 = *(__int64 ***)(v244 + 8);
            v20 = (__int64 *)&v12[*(unsigned int *)(v244 + 16)];
            break;
        }
        break;
      }
LABEL_20:
      v28 = (unsigned __int64)(v27 * v26) >> 3;
      if ( v12 != (__int64 **)v20 )
      {
        v29 = (__int64 *)v12;
        while ( (unsigned int)v28 >= (unsigned int)sub_1B7C680(a1, *v29) )
        {
          if ( v20 == ++v29 )
            goto LABEL_9;
        }
LABEL_24:
        v30 = a1;
        v243 = 0;
        v234 = a1 + 128;
        v238 = a1 + 168;
        while ( 2 )
        {
          v31 = *(_DWORD *)(v30 + 152);
          v32 = *(_QWORD *)(v30 + 128) + 1LL;
          v33 = *(_DWORD *)(v30 + 144);
          *(_QWORD *)(v30 + 128) = v32;
          if ( !v33 )
          {
            LODWORD(v19) = *(_DWORD *)(v30 + 148);
            if ( (_DWORD)v19 )
            {
              if ( v31 <= 0x40 )
                goto LABEL_28;
              j___libc_free_0(*(_QWORD *)(v30 + 136));
              ++*(_QWORD *)(v30 + 128);
              *(_QWORD *)(v30 + 136) = 0;
              *(_QWORD *)(v30 + 144) = 0;
              *(_DWORD *)(v30 + 152) = 0;
LABEL_186:
              sub_1B864A0(v234, 128);
LABEL_33:
              v36 = *(_QWORD *)(v30 + 176);
              while ( v36 )
              {
                v37 = v36;
                sub_1B7DEF0(*(_QWORD *)(v36 + 24));
                v36 = *(_QWORD *)(v36 + 16);
                if ( *(_DWORD *)(v37 + 48) > 0x40u )
                {
                  v38 = *(_QWORD *)(v37 + 40);
                  if ( v38 )
                    j_j___libc_free_0_0(v38);
                }
                j_j___libc_free_0(v37, 64);
              }
              v39 = 64;
              *(_QWORD *)(v30 + 176) = 0;
              *(_QWORD *)(v30 + 200) = 0;
              *(_QWORD *)(v30 + 184) = v238;
              *(_QWORD *)(v30 + 192) = v238;
              *(_BYTE *)(v30 + 124) = 1;
              if ( v241 - v243 <= 0x40 )
                v39 = v241 - v243;
              v40 = *(_QWORD *)(v244 + 8);
              v240 = v39;
              ++*(_QWORD *)(v30 + 240);
              v41 = v40 + 8LL * v243;
              v298 = v300;
              v299 = 0x4000000000LL;
              v42 = *(_DWORD *)(v30 + 256);
              if ( v42 )
              {
                v95 = 4 * v42;
                v43 = *(unsigned int *)(v30 + 264);
                if ( (unsigned int)(4 * v42) < 0x40 )
                  v95 = 64;
                if ( (unsigned int)v43 <= v95 )
                {
LABEL_43:
                  v44 = *(_QWORD **)(v30 + 248);
                  for ( i = &v44[v43]; i != v44; ++v44 )
                    *v44 = -8;
                  *(_QWORD *)(v30 + 256) = 0;
                  goto LABEL_46;
                }
                v96 = *(_QWORD **)(v30 + 248);
                v97 = v42 - 1;
                if ( v97 )
                {
                  _BitScanReverse(&v97, v97);
                  v98 = 1 << (33 - (v97 ^ 0x1F));
                  if ( v98 < 64 )
                    v98 = 64;
                  if ( (_DWORD)v43 == v98 )
                  {
                    *(_QWORD *)(v30 + 256) = 0;
                    v183 = &v96[v43];
                    do
                    {
                      if ( v96 )
                        *v96 = -8;
                      ++v96;
                    }
                    while ( v183 != v96 );
                    goto LABEL_46;
                  }
                  v99 = (4 * v98 / 3u + 1) | ((unsigned __int64)(4 * v98 / 3u + 1) >> 1);
                  v100 = ((v99 | (v99 >> 2)) >> 4)
                       | v99
                       | (v99 >> 2)
                       | ((((v99 | (v99 >> 2)) >> 4) | v99 | (v99 >> 2)) >> 8);
                  v101 = (v100 | (v100 >> 16)) + 1;
                  v102 = 8 * ((v100 | (v100 >> 16)) + 1);
                }
                else
                {
                  v102 = 1024;
                  v101 = 128;
                }
                j___libc_free_0(v96);
                *(_DWORD *)(v30 + 264) = v101;
                v103 = (_QWORD *)sub_22077B0(v102);
                v104 = *(unsigned int *)(v30 + 264);
                *(_QWORD *)(v30 + 256) = 0;
                *(_QWORD *)(v30 + 248) = v103;
                v105 = &v103[v104];
                if ( v103 == v105 )
                  goto LABEL_46;
                do
                {
                  if ( v103 )
                    *v103 = -8;
                  ++v103;
                }
                while ( v105 != v103 );
                if ( *(_BYTE *)(*(_QWORD *)v41 + 16LL) == 54 )
                  goto LABEL_151;
LABEL_47:
                v283 = (unsigned __int64)&v285;
                v284 = 0x1000000000LL;
                v288 = 0x1000000000LL;
                v287 = v289;
                v249 = v240;
                goto LABEL_48;
              }
              v18 = *(_DWORD *)(v30 + 260);
              if ( v18 )
              {
                v43 = *(unsigned int *)(v30 + 264);
                if ( (unsigned int)v43 <= 0x40 )
                  goto LABEL_43;
                j___libc_free_0(*(_QWORD *)(v30 + 248));
                *(_QWORD *)(v30 + 248) = 0;
                *(_QWORD *)(v30 + 256) = 0;
                *(_DWORD *)(v30 + 264) = 0;
              }
LABEL_46:
              if ( *(_BYTE *)(*(_QWORD *)v41 + 16LL) != 54 )
                goto LABEL_47;
LABEL_151:
              if ( !byte_4FB7BE0 )
                goto LABEL_47;
              v106 = v303;
              v107 = (_QWORD *)v41;
              v108 = 0;
              v302 = 0x4000000000LL;
              v109 = 0;
              for ( j = v303; ; v106 = j )
              {
                *(_QWORD *)&v106[8 * v109] = *v107;
                ++v108;
                v109 = (unsigned int)(v302 + 1);
                LODWORD(v302) = v302 + 1;
                if ( v240 <= (unsigned __int64)v108 )
                  break;
                v107 = (_QWORD *)(v41 + 8LL * v108);
                if ( HIDWORD(v302) <= (unsigned int)v109 )
                {
                  sub_16CD150((__int64)&j, v303, 0, 8, v18, (unsigned __int8)v19);
                  v109 = (unsigned int)v302;
                }
              }
              v110 = 8 * v109;
              v283 = 0;
              v284 = 0;
              v285 = 0;
              v111 = &j[8 * v109];
              v286 = 0;
              if ( j == v111 )
              {
                v283 = 1;
                goto LABEL_191;
              }
              v112 = &j[8 * v109];
              srcb = j;
              v273.m128i_i64[1] = 0;
              v273.m128i_i64[0] = (__int64)sub_1B8B710;
              _BitScanReverse64(&v113, v110 >> 3);
              v274 = &v283;
              v275 = v30;
              sub_1B7E1A0(j, v112, 2LL * (int)(63 - (v113 ^ 0x3F)), &v273);
              v114 = srcb;
              if ( (unsigned __int64)v110 <= 0x80 )
              {
                v276.m128i_i64[0] = (__int64)sub_1B8B710;
                v276.m128i_i64[1] = 0;
                v277 = &v283;
                v278 = v30;
                sub_1B7D8B0(srcb, v111, &v276);
                goto LABEL_160;
              }
              v279.m128i_i64[0] = (__int64)sub_1B8B710;
              srcd = srcb + 128;
              v279.m128i_i64[1] = 0;
              v280 = &v283;
              v281 = v30;
              sub_1B7D8B0(v114, srcd, &v279);
              if ( v111 == srcd )
              {
LABEL_160:
                v115 = (_DWORD *)(v283 + 1);
                v116 = v285;
              }
              else
              {
                v180 = srcd;
                do
                {
                  v181 = v180;
                  v180 += 8;
                  v282.m128i_i64[0] = (__int64)sub_1B8B710;
                  v282.m128i_i64[1] = 0;
                  a3 = _mm_loadu_si128(&v282);
                  v294 = &v283;
                  v295 = v30;
                  v293 = a3;
                  sub_1B7D770(v181, (char **)&v293);
                }
                while ( v111 != v180 );
                v115 = (_DWORD *)(v283 + 1);
                v116 = v285;
              }
              v283 = (unsigned __int64)v115;
              if ( v116 )
              {
                v117 = 4 * v116;
                v118 = v286;
                if ( (unsigned int)(4 * v116) < 0x40 )
                  v117 = 64;
                if ( v286 <= v117 )
                  goto LABEL_165;
                v188 = (_QWORD *)v284;
                v189 = v116 - 1;
                if ( !v189 )
                {
                  v194 = 2048;
                  v193 = 128;
                  goto LABEL_312;
                }
                _BitScanReverse(&v189, v189);
                v190 = 1 << (33 - (v189 ^ 0x1F));
                if ( v190 < 64 )
                  v190 = 64;
                if ( v286 == v190 )
                {
                  v285 = 0;
                  v221 = (_QWORD *)(v284 + 16LL * v286);
                  do
                  {
                    if ( v188 )
                      *v188 = -8;
                    v188 += 2;
                  }
                  while ( v221 != v188 );
                }
                else
                {
                  v191 = (4 * v190 / 3u + 1) | ((unsigned __int64)(4 * v190 / 3u + 1) >> 1);
                  v192 = ((v191 | (v191 >> 2)) >> 4)
                       | v191
                       | (v191 >> 2)
                       | ((((v191 | (v191 >> 2)) >> 4) | v191 | (v191 >> 2)) >> 8);
                  v193 = (v192 | (v192 >> 16)) + 1;
                  v194 = 16 * ((v192 | (v192 >> 16)) + 1);
LABEL_312:
                  j___libc_free_0(v284);
                  v286 = v193;
                  v200 = (_QWORD *)sub_22077B0(v194);
                  v285 = 0;
                  v284 = (__int64)v200;
                  for ( k = &v200[2 * v286]; k != v200; v200 += 2 )
                  {
                    if ( v200 )
                      *v200 = -8;
                  }
                }
LABEL_191:
                v121 = (unsigned int)v302;
                if ( !(_DWORD)v302 )
                  goto LABEL_168;
LABEL_192:
                v232 = v41;
                v134 = 0;
                v233 = 8 * v121;
                v257 = 0;
                srcc = 0;
                while ( 2 )
                {
                  v136 = *(__int64 **)&j[v257];
                  if ( !v134 )
                  {
                    v135 = (unsigned int)v299;
                    if ( HIDWORD(v299) <= (unsigned int)v299 )
                      goto LABEL_201;
LABEL_196:
                    v134 = v136;
                    *(_QWORD *)&v298[8 * v135] = v136;
                    LODWORD(v299) = v299 + 1;
                    if ( srcc > 2 )
                    {
                      srcc = 0;
                      v134 = 0;
                    }
LABEL_198:
                    v257 += 8;
                    if ( v233 == v257 )
                    {
                      v41 = v232;
                      goto LABEL_168;
                    }
                    continue;
                  }
                  break;
                }
                if ( sub_1B88260(v30, (__int64)v134, *(_QWORD *)&j[v257], a3, a4) )
                {
                  ++srcc;
                  v135 = (unsigned int)v299;
                  goto LABEL_195;
                }
                if ( (*(_BYTE *)(*v134 + 8) == 16) != (*(_BYTE *)(*v136 + 8) == 16) )
                  goto LABEL_203;
                v137 = sub_127FA20(*(_QWORD *)(v30 + 40), *v134);
                if ( (unsigned __int64)(v137 + 7) >> 3 != (unsigned __int64)(sub_127FA20(*(_QWORD *)(v30 + 40), *v136)
                                                                           + 7) >> 3 )
                  goto LABEL_203;
                v138 = *v134;
                if ( *(_BYTE *)(*v134 + 8) == 16 )
                  v138 = **(_QWORD **)(v138 + 16);
                v139 = sub_127FA20(*(_QWORD *)(v30 + 40), v138);
                v140 = *v136;
                if ( *(_BYTE *)(*v136 + 8) == 16 )
                  v140 = **(_QWORD **)(v140 + 16);
                if ( (unsigned __int64)(v139 + 7) >> 3 != (unsigned __int64)(sub_127FA20(*(_QWORD *)(v30 + 40), v140) + 7) >> 3 )
                  goto LABEL_203;
                v141 = sub_1B7F330(v30, (__int64)v134);
                v142 = sub_1B7F330(v30, (__int64)v136);
                v19 = v142;
                if ( !v141 )
                  goto LABEL_203;
                if ( !v142 )
                  goto LABEL_203;
                v143 = *(_DWORD *)(v141 + 20) & 0xFFFFFFF;
                v144 = *(_DWORD *)(v142 + 20) & 0xFFFFFFF;
                v250 = v143;
                if ( v143 != (_DWORD)v144 )
                  goto LABEL_203;
                v245 = (_QWORD *)v19;
                v145 = *(_BYTE *)(v19 + 23) & 0x40;
                v146 = *(_BYTE *)(v141 + 23) & 0x40;
                v18 = v143 - 1;
                if ( v143 != 1 )
                {
                  v147 = 24LL * v143;
                  v148 = v141 - v147;
                  v149 = v19 - v147;
                  v150 = 0;
                  v263 = 8 * (3LL * (v143 - 2) + 3);
                  do
                  {
                    v151 = v148;
                    if ( v146 )
                      v151 = *(_QWORD *)(v141 - 8);
                    v152 = *(_QWORD *)(v151 + v150);
                    v153 = v149;
                    if ( v145 )
                      v153 = *(_QWORD *)(v19 - 8);
                    if ( v152 != *(_QWORD *)(v153 + v150) )
                      goto LABEL_203;
                    v150 += 24;
                  }
                  while ( v263 != v150 );
                }
                if ( v146 )
                  v154 = *(_QWORD *)(v141 - 8);
                else
                  v154 = v141 - 24LL * v250;
                v155 = 0;
                v239 = *(_QWORD *)(v154 + 24LL * v18);
                if ( *(_BYTE *)(v239 + 16) == 13 )
                  v155 = *(_QWORD *)(v154 + 24LL * v18);
                v156 = v145 ? *(_QWORD *)(v19 - 8) : v19 - 24 * v144;
                v157 = *(_QWORD *)(v156 + 24LL * v18);
                if ( *(_BYTE *)(v157 + 16) != 13 || !v155 )
                  goto LABEL_203;
                v158 = *(_QWORD **)(v155 + 24);
                if ( *(_DWORD *)(v155 + 32) > 0x40u )
                  v158 = (_QWORD *)*v158;
                v159 = *(_QWORD **)(v157 + 24);
                v264 = v158;
                if ( *(_DWORD *)(v157 + 32) > 0x40u )
                  v159 = (_QWORD *)*v159;
                v160 = (_DWORD)v159 - (_DWORD)v158;
                if ( (unsigned int)(v160 - 2) > 1 )
                  goto LABEL_203;
                v161 = v289;
                v162 = (_QWORD *)(v154 + 24);
                v287 = v289;
                v163 = 24LL * v250;
                v164 = (_QWORD *)(v154 + v163);
                v165 = v163 - 24;
                v288 = 0x1000000000LL;
                v166 = 0xAAAAAAAAAAAAAAABLL * (v165 >> 3);
                if ( (unsigned __int64)v165 > 0x180 )
                {
                  v225 = v162;
                  v231 = v164;
                  v237 = v19;
                  v261 = -1431655765 * (v165 >> 3);
                  sub_16CD150((__int64)&v287, v289, v166, 8, v18, (unsigned __int8)v19);
                  v162 = v225;
                  v164 = v231;
                  v19 = v237;
                  LODWORD(v166) = v261;
                  v161 = &v287[2 * (unsigned int)v288];
                }
                for ( ; v164 != v162; ++v161 )
                {
                  if ( v161 )
                    *v161 = *v162;
                  v162 += 3;
                }
                v226 = v19;
                v236 = v287;
                LODWORD(v288) = v288 + v166;
                v259 = (unsigned int)v288;
                v167 = sub_16348C0(v141);
                v168 = sub_15F9F50(v167, (__int64)v236, v259);
                v170 = v226;
                v260 = v168;
                if ( (*(_BYTE *)(v226 + 23) & 0x40) != 0 )
                {
                  v171 = *(_QWORD *)(v226 - 8);
                  v172 = 24LL * (*(_DWORD *)(v226 + 20) & 0xFFFFFFF);
                  v245 = (_QWORD *)(v171 + v172);
                }
                else
                {
                  v172 = 24LL * (*(_DWORD *)(v226 + 20) & 0xFFFFFFF);
                  v171 = v226 - v172;
                }
                v173 = v172 - 24;
                v174 = (__int64 *)v292;
                v175 = (_QWORD *)(v171 + 24);
                v291 = 0x1000000000LL;
                v290 = (__int64 *)v292;
                v176 = 0xAAAAAAAAAAAAAAABLL * (v173 >> 3);
                if ( (unsigned __int64)v173 > 0x180 )
                {
                  v222 = v226;
                  v223 = v175;
                  v228 = 0xAAAAAAAAAAAAAAABLL * (v173 >> 3);
                  sub_16CD150((__int64)&v290, v292, v176, 8, v169, (unsigned __int8)v170);
                  v170 = v222;
                  v175 = v223;
                  LODWORD(v176) = v228;
                  v174 = &v290[(unsigned int)v291];
                }
                for ( ; v175 != v245; ++v174 )
                {
                  if ( v174 )
                    *v174 = *v175;
                  v175 += 3;
                }
                v227 = (__int64)v290;
                LODWORD(v291) = v176 + v291;
                v246 = (unsigned int)v291;
                v177 = sub_16348C0(v170);
                if ( v260 == sub_15F9F50(v177, v227, v246) && v260 == *v134 )
                {
                  v202 = 24LL * (*(_DWORD *)(v141 + 20) & 0xFFFFFFF);
                  v203 = v141 - v202;
                  if ( (*(_BYTE *)(v141 + 23) & 0x40) != 0 )
                    v203 = *(_QWORD *)(v141 - 8);
                  v204 = (_QWORD *)(v203 + v202);
                  v205 = v202 - 24;
                  v206 = (_QWORD *)(v203 + 24);
                  v293.m128i_i64[0] = (__int64)&v294;
                  v293.m128i_i64[1] = 0x1000000000LL;
                  v207 = 0xAAAAAAAAAAAAAAABLL * (v205 >> 3);
                  if ( (unsigned __int64)v205 > 0x180 )
                  {
                    v224 = v206;
                    v230 = v204;
                    v248 = 0xAAAAAAAAAAAAAAABLL * (v205 >> 3);
                    sub_16CD150((__int64)&v293, &v294, v207, 8, v18, (unsigned __int8)v206);
                    v206 = v224;
                    v204 = v230;
                    LODWORD(v207) = v248;
                  }
                  v208 = v293.m128i_i32[2];
                  v209 = (_QWORD *)(v293.m128i_i64[0] + 8LL * v293.m128i_u32[2]);
                  if ( v204 != v206 )
                  {
                    do
                    {
                      if ( v209 )
                        *v209 = *v206;
                      v206 += 3;
                      ++v209;
                    }
                    while ( v204 != v206 );
                    v208 = v293.m128i_i32[2];
                  }
                  v293.m128i_i32[2] = v207 + v208;
                  v210 = sub_15A0680(*(_QWORD *)v239, (__int64)v264 + 1, 0);
                  *(_QWORD *)(v293.m128i_i64[0] + 8LL * (v250 - 2)) = v210;
                  v251 = v250 - 2;
                  v229 = v293.m128i_i64[0];
                  v247 = v293.m128i_u32[2];
                  v211 = sub_16348C0(v141);
                  if ( v260 == sub_15F9F50(v211, v229, v247) )
                  {
                    if ( v160 == 3 )
                    {
                      v212 = sub_15A0680(*(_QWORD *)v239, (__int64)v264 + 2, 0);
                      *(_QWORD *)(v293.m128i_i64[0] + 8 * v251) = v212;
                      v252 = v293.m128i_i64[0];
                      v265 = v293.m128i_u32[2];
                      v213 = sub_16348C0(v141);
                      if ( v260 != sub_15F9F50(v213, v252, v265) )
                        v160 = 0;
                    }
                    else
                    {
                      v160 = 2;
                    }
                  }
                  else
                  {
                    v160 = 0;
                  }
                  if ( (unsigned __int64 **)v293.m128i_i64[0] != &v294 )
                    _libc_free(v293.m128i_u64[0]);
                }
                else
                {
                  v160 = 0;
                }
                if ( v290 != (__int64 *)v292 )
                  _libc_free((unsigned __int64)v290);
                if ( v287 != (_DWORD *)v289 )
                  _libc_free((unsigned __int64)v287);
                if ( v160 == 2 )
                {
                  v178 = sub_1B889F0(v30, (__int64)v136);
                  sub_1C30170(v178, 0);
                  v179 = (unsigned int)v299;
                  if ( (unsigned int)v299 >= HIDWORD(v299) )
                  {
                    sub_16CD150((__int64)&v298, v300, 0, 8, v18, (unsigned __int8)v19);
                    v179 = (unsigned int)v299;
                  }
                  srcc += 2;
                  *(_QWORD *)&v298[8 * v179] = v178;
                  v135 = (unsigned int)(v299 + 1);
                  LODWORD(v299) = v299 + 1;
LABEL_195:
                  if ( HIDWORD(v299) <= (unsigned int)v135 )
                    goto LABEL_201;
                  goto LABEL_196;
                }
                if ( v160 == 3 && !srcc )
                {
                  v214 = sub_1B889F0(v30, (__int64)v136);
                  sub_1C30170(v214, 0);
                  v217 = (unsigned int)v299;
                  if ( (unsigned int)v299 >= HIDWORD(v299) )
                  {
                    sub_16CD150((__int64)&v298, v300, 0, 8, v215, v216);
                    v217 = (unsigned int)v299;
                  }
                  *(_QWORD *)&v298[8 * v217] = v214;
                  LODWORD(v299) = v299 + 1;
                  v218 = sub_1B889F0(v30, (__int64)v214);
                  sub_1C30170(v218, 0);
                  v219 = (unsigned int)v299;
                  if ( (unsigned int)v299 >= HIDWORD(v299) )
                  {
                    sub_16CD150((__int64)&v298, v300, 0, 8, v18, (unsigned __int8)v19);
                    v219 = (unsigned int)v299;
                  }
                  *(_QWORD *)&v298[8 * v219] = v218;
                  v220 = (unsigned int)(v299 + 1);
                  LODWORD(v299) = v220;
                  if ( HIDWORD(v299) > (unsigned int)v220 )
                  {
                    v134 = 0;
                    srcc = 0;
                    *(_QWORD *)&v298[8 * v220] = v136;
                    LODWORD(v299) = v299 + 1;
                    goto LABEL_198;
                  }
                  srcc = 3;
                }
                else
                {
LABEL_203:
                  if ( (unsigned int)v299 < HIDWORD(v299) )
                  {
                    v134 = v136;
                    srcc = 0;
                    *(_QWORD *)&v298[8 * (unsigned int)v299] = v136;
                    LODWORD(v299) = v299 + 1;
                    goto LABEL_198;
                  }
                  srcc = 0;
                }
LABEL_201:
                sub_16CD150((__int64)&v298, v300, 0, 8, v18, (unsigned __int8)v19);
                v135 = (unsigned int)v299;
                goto LABEL_196;
              }
              if ( !HIDWORD(v285) )
                goto LABEL_191;
              v118 = v286;
              if ( v286 > 0x40 )
              {
                j___libc_free_0(v284);
                v284 = 0;
                v285 = 0;
                v286 = 0;
                goto LABEL_191;
              }
LABEL_165:
              v119 = (_QWORD *)v284;
              v120 = (_QWORD *)(v284 + 16LL * v118);
              if ( (_QWORD *)v284 != v120 )
              {
                do
                {
                  *v119 = -8;
                  v119 += 2;
                }
                while ( v120 != v119 );
              }
              v121 = (unsigned int)v302;
              v285 = 0;
              if ( (_DWORD)v302 )
                goto LABEL_192;
LABEL_168:
              j___libc_free_0(v284);
              if ( j != v303 )
                _libc_free((unsigned __int64)j);
              if ( v240 >= (unsigned __int64)(unsigned int)v299 )
                goto LABEL_47;
              v249 = v299;
              v284 = 0x1000000000LL;
              v41 = (unsigned __int64)v298;
              v283 = (unsigned __int64)&v285;
              v287 = v289;
              v288 = 0x1000000000LL;
              if ( (int)v299 > 0 )
              {
LABEL_48:
                v46 = 0;
                src = &j;
                do
                {
                  v253 = v46;
                  v258 = v46;
                  *(_DWORD *)src = -1;
                  v47 = v46;
                  v255 = v46;
                  v48 = v41;
                  v49 = v249 - 1;
                  v50 = v47;
                  do
                  {
                    if ( (_DWORD)v49 != (_DWORD)v50
                      && sub_1B88260(v30, *(_QWORD *)(v48 + 8 * v50), *(_QWORD *)(v48 + 8 * v49), a3, a4)
                      && (*(_DWORD *)src == -1
                       || (int)abs32(*(_DWORD *)src - v258) >= (int)abs32(*(_DWORD *)src - v49) && (int)v49 >= v255) )
                    {
                      v53 = (unsigned int)v288;
                      if ( (unsigned int)v288 >= HIDWORD(v288) )
                      {
                        sub_16CD150((__int64)&v287, v289, 0, 4, v51, v52);
                        v53 = (unsigned int)v288;
                      }
                      v287[v53] = v49;
                      v54 = (unsigned int)v284;
                      LODWORD(v288) = v288 + 1;
                      if ( (unsigned int)v284 >= HIDWORD(v284) )
                      {
                        sub_16CD150((__int64)&v283, &v285, 0, 4, v51, v52);
                        v54 = (unsigned int)v284;
                      }
                      *(_DWORD *)(v283 + 4 * v54) = v253;
                      LODWORD(v284) = v284 + 1;
                      *(_DWORD *)src = v49;
                    }
                    --v49;
                  }
                  while ( (_DWORD)v49 != -1 );
                  v41 = v48;
                  src = (char **)((char *)src + 4);
                  v46 = v50 + 1;
                }
                while ( v249 > (int)v50 + 1 );
              }
              v55 = *(_DWORD *)(v30 + 144);
              ++*(_QWORD *)(v30 + 128);
              if ( v55 )
              {
                v84 = 4 * v55;
                v56 = *(unsigned int *)(v30 + 152);
                if ( (unsigned int)(4 * v55) < 0x40 )
                  v84 = 64;
                if ( (unsigned int)v56 <= v84 )
                {
LABEL_65:
                  v57 = *(_QWORD **)(v30 + 136);
                  for ( m = &v57[2 * v56]; m != v57; v57 += 2 )
                    *v57 = -8;
                  *(_QWORD *)(v30 + 144) = 0;
                  goto LABEL_68;
                }
                v85 = *(_QWORD **)(v30 + 136);
                v86 = v55 - 1;
                if ( v86 )
                {
                  _BitScanReverse(&v86, v86);
                  v87 = 1 << (33 - (v86 ^ 0x1F));
                  if ( v87 < 64 )
                    v87 = 64;
                  if ( (_DWORD)v56 == v87 )
                  {
                    *(_QWORD *)(v30 + 144) = 0;
                    v184 = &v85[2 * (unsigned int)v56];
                    do
                    {
                      if ( v85 )
                        *v85 = -8;
                      v85 += 2;
                    }
                    while ( v184 != v85 );
                    goto LABEL_68;
                  }
                  v88 = (4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1);
                  v89 = ((v88 | (v88 >> 2)) >> 4)
                      | v88
                      | (v88 >> 2)
                      | ((((v88 | (v88 >> 2)) >> 4) | v88 | (v88 >> 2)) >> 8);
                  v90 = (v89 | (v89 >> 16)) + 1;
                  v91 = 16 * ((v89 | (v89 >> 16)) + 1);
                }
                else
                {
                  v91 = 2048;
                  v90 = 128;
                }
                j___libc_free_0(v85);
                *(_DWORD *)(v30 + 152) = v90;
                v92 = (_QWORD *)sub_22077B0(v91);
                v93 = *(unsigned int *)(v30 + 152);
                *(_QWORD *)(v30 + 144) = 0;
                *(_QWORD *)(v30 + 136) = v92;
                for ( n = &v92[2 * v93]; n != v92; v92 += 2 )
                {
                  if ( v92 )
                    *v92 = -8;
                }
                goto LABEL_68;
              }
              if ( *(_DWORD *)(v30 + 148) )
              {
                v56 = *(unsigned int *)(v30 + 152);
                if ( (unsigned int)v56 <= 0x40 )
                  goto LABEL_65;
                j___libc_free_0(*(_QWORD *)(v30 + 136));
                *(_QWORD *)(v30 + 136) = 0;
                *(_QWORD *)(v30 + 144) = 0;
                *(_DWORD *)(v30 + 152) = 0;
              }
LABEL_68:
              sub_1B7DEF0(*(_QWORD *)(v30 + 176));
              *(_BYTE *)(v30 + 124) = 0;
              v59 = v283;
              *(_QWORD *)(v30 + 176) = 0;
              *(_QWORD *)(v30 + 184) = v238;
              *(_QWORD *)(v30 + 192) = v238;
              *(_QWORD *)(v30 + 200) = 0;
              v293.m128i_i64[1] = (__int64)v297;
              v294 = (unsigned __int64 *)v297;
              v293.m128i_i64[0] = 0;
              v295 = 16;
              v296 = 0;
              v262 = (int *)(v59 + 4LL * (unsigned int)v284);
              if ( (int *)v59 != v262 )
              {
                srca = (int *)v59;
                v256 = 0;
                v254 = (__int64 *)v30;
                do
                {
                  v60 = *srca;
                  v61 = sub_13A0E30((__int64)&v293, *(_QWORD *)(v41 + 8 * v60));
                  if ( !v61 )
                  {
                    if ( (_DWORD)v288 )
                    {
                      v64 = 0;
                      while ( 1 )
                      {
                        while ( (_DWORD)v60 != v287[v64] )
                        {
                          v64 = (unsigned int)(v61 + 1);
                          v61 = v64;
                          if ( (unsigned int)v288 <= (unsigned int)v64 )
                            goto LABEL_78;
                        }
                        if ( !sub_13A0E30((__int64)&v293, *(_QWORD *)(v41 + 8LL * *(int *)(v283 + 4 * v64))) )
                          break;
                        v64 = (unsigned int)(v61 + 1);
                        v61 = v64;
                        if ( (unsigned int)v288 <= (unsigned int)v64 )
                          goto LABEL_78;
                      }
                    }
                    else
                    {
LABEL_78:
                      v272 = v60;
                      v65 = 0;
                      v66 = (__int64 *)v292;
                      v290 = (__int64 *)v292;
                      v291 = 0x1000000000LL;
                      if ( (_DWORD)v60 != -1 )
                      {
                        do
                        {
                          v68 = &v287[(unsigned int)v288];
                          if ( v68 == sub_1B7D0F0(v287, (__int64)v68, &v272)
                            && (v69 = (_DWORD *)(v283 + 4LL * (unsigned int)v284),
                                v69 == sub_1B7D0F0((_DWORD *)v283, (__int64)v69, &v272))
                            || (v70 = (__int64 *)(v41 + 8 * v60), sub_13A0E30((__int64)&v293, *v70)) )
                          {
                            v66 = v290;
                            v65 = (unsigned int)v291;
                            goto LABEL_87;
                          }
                          v73 = (unsigned int)v291;
                          if ( (unsigned int)v291 >= HIDWORD(v291) )
                          {
                            sub_16CD150((__int64)&v290, v292, 0, 8, v71, v72);
                            v73 = (unsigned int)v291;
                          }
                          v290[v73] = *v70;
                          v60 = *((int *)&j + v60);
                          v67 = v291 + 1;
                          LODWORD(v291) = v291 + 1;
                          v272 = v60;
                        }
                        while ( (_DWORD)v60 != -1 );
                        v66 = v290;
                        v65 = v67;
                      }
LABEL_87:
                      v74 = *(_BYTE *)(*v66 + 16);
                      if ( v74 == 54
                        || v74 == 78
                        && (v75 = *(_QWORD *)(*v66 - 24), !*(_BYTE *)(v75 + 16))
                        && ((v76 = *(_DWORD *)(v75 + 36), v76 == 4085) || v76 == 4057) )
                      {
                        v77 = sub_1B83220(
                                v254,
                                v66,
                                v65,
                                (__int64)&v293,
                                (__m128)a3,
                                *(double *)a4.m128i_i64,
                                a5,
                                a6,
                                v62,
                                v63,
                                a9,
                                a10);
                      }
                      else
                      {
                        v77 = sub_1B84C00(v254, v66, v65, (__int64)&v293);
                      }
                      v256 |= v77;
                      if ( v290 != (__int64 *)v292 )
                        _libc_free((unsigned __int64)v290);
                    }
                  }
                  ++srca;
                }
                while ( v262 != srca );
                v30 = (__int64)v254;
                v242 |= v256;
                if ( v294 != (unsigned __int64 *)v293.m128i_i64[1] )
                  _libc_free((unsigned __int64)v294);
              }
              if ( v287 != (_DWORD *)v289 )
                _libc_free((unsigned __int64)v287);
              if ( (__int64 *)v283 != &v285 )
                _libc_free(v283);
LABEL_101:
              if ( !*(_DWORD *)(v30 + 256) )
                goto LABEL_102;
              v78 = *(_QWORD ***)(v30 + 248);
              v79 = &v78[*(unsigned int *)(v30 + 264)];
              if ( v78 == v79 )
                goto LABEL_102;
              while ( 1 )
              {
                v80 = *v78;
                v81 = v78;
                v82 = *v78 + 2 == 0 || *v78 + 1 == 0;
                if ( !v82 )
                  break;
                if ( v79 == ++v78 )
                  goto LABEL_102;
              }
              if ( v78 == v79 )
              {
LABEL_102:
                if ( v298 != v300 )
                  _libc_free((unsigned __int64)v298);
                v243 += 64;
                if ( v241 <= v243 )
                {
                  a1 = v30;
                  goto LABEL_9;
                }
                continue;
              }
              while ( 2 )
              {
                v83 = v81 + 1;
                if ( v81 + 1 == v79 )
                {
LABEL_115:
                  if ( v80[1] )
                    goto LABEL_121;
                }
                else
                {
                  while ( *v83 == -8 || *v83 == -16 )
                  {
                    if ( v79 == ++v83 )
                      goto LABEL_115;
                  }
                  if ( v80[1] )
                  {
                    v81 = v83;
                    goto LABEL_117;
                  }
                }
                sub_15F20C0(v80);
                *v81 = -16;
                v82 = 1;
                v81 = v83;
                --*(_DWORD *)(v30 + 256);
                ++*(_DWORD *)(v30 + 260);
LABEL_117:
                if ( v83 == v79 )
                {
LABEL_121:
                  if ( !v82 )
                    goto LABEL_102;
                  goto LABEL_101;
                }
                v80 = (_QWORD *)*v83;
                continue;
              }
            }
LABEL_32:
            *(_QWORD *)(v30 + 128) = v32 + 1;
            if ( v31 <= 0x7F )
              goto LABEL_186;
            goto LABEL_33;
          }
          break;
        }
        v122 = 4 * v33;
        if ( (unsigned int)(4 * v33) < 0x40 )
          v122 = 64;
        if ( v122 >= v31 )
        {
LABEL_28:
          v34 = *(_QWORD **)(v30 + 136);
          v35 = &v34[2 * v31];
          if ( v34 != v35 )
          {
            do
            {
              *v34 = -8;
              v34 += 2;
            }
            while ( v35 != v34 );
            v32 = *(_QWORD *)(v30 + 128);
            v31 = *(_DWORD *)(v30 + 152);
          }
          *(_QWORD *)(v30 + 144) = 0;
          goto LABEL_32;
        }
        v123 = *(_QWORD **)(v30 + 136);
        v124 = v33 - 1;
        if ( v124 )
        {
          _BitScanReverse(&v124, v124);
          v125 = (unsigned int)(1 << (33 - (v124 ^ 0x1F)));
          if ( (int)v125 < 64 )
            v125 = 64;
          if ( (_DWORD)v125 == v31 )
          {
            *(_QWORD *)(v30 + 144) = 0;
            v182 = &v123[2 * v125];
            do
            {
              if ( v123 )
                *v123 = -8;
              v123 += 2;
            }
            while ( v182 != v123 );
            v32 = *(_QWORD *)(v30 + 128);
            v31 = *(_DWORD *)(v30 + 152);
            goto LABEL_32;
          }
          v126 = (4 * (int)v125 / 3u + 1) | ((unsigned __int64)(4 * (int)v125 / 3u + 1) >> 1);
          v127 = ((v126 | (v126 >> 2)) >> 4)
               | v126
               | (v126 >> 2)
               | ((((v126 | (v126 >> 2)) >> 4) | v126 | (v126 >> 2)) >> 8);
          v128 = (v127 | (v127 >> 16)) + 1;
          v129 = 16 * ((v127 | (v127 >> 16)) + 1);
        }
        else
        {
          v129 = 2048;
          v128 = 128;
        }
        j___libc_free_0(v123);
        *(_DWORD *)(v30 + 152) = v128;
        v130 = (_QWORD *)sub_22077B0(v129);
        v131 = *(unsigned int *)(v30 + 152);
        *(_QWORD *)(v30 + 144) = 0;
        *(_QWORD *)(v30 + 136) = v130;
        v132 = v131;
        for ( ii = &v130[2 * v131]; ii != v130; v130 += 2 )
        {
          if ( v130 )
            *v130 = -8;
        }
        ++*(_QWORD *)(v30 + 128);
        if ( v132 <= 0x7F )
          goto LABEL_186;
        goto LABEL_33;
      }
LABEL_9:
      v244 += 88;
      if ( v235 == v244 )
        return v242;
    }
    if ( v23 == 78 )
    {
      v185 = *(v13 - 3);
      if ( !*(_BYTE *)(v185 + 16) )
      {
        v186 = *(_DWORD *)(v185 + 36);
        if ( v186 == 4085 || v186 == 4057 )
          goto LABEL_17;
      }
    }
    else if ( v23 == 55 )
    {
      v24 = (__int64 *)*(v13 - 6);
LABEL_16:
      v16 = *v24;
      v17 = *(unsigned __int8 *)(*v24 + 8);
      goto LABEL_17;
    }
    if ( (*((_BYTE *)v13 + 23) & 0x40) != 0 )
      v187 = (__int64 *)*(v13 - 1);
    else
      v187 = &v13[-3 * (*((_DWORD *)v13 + 5) & 0xFFFFFFF)];
    v24 = (__int64 *)v187[3];
    goto LABEL_16;
  }
  return 0;
}
