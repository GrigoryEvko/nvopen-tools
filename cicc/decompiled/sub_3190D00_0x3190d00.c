// Function: sub_3190D00
// Address: 0x3190d00
//
__int64 __fastcall sub_3190D00(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  char v3; // r13
  char *v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 *v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r15
  char v10; // cl
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // r11d
  __int64 v14; // rsi
  unsigned int i; // ecx
  unsigned int *v16; // r15
  unsigned int v17; // ecx
  __int64 v18; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  unsigned __int64 v23; // rax
  int v24; // r15d
  __int64 v25; // r9
  unsigned int *v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned int *v29; // r14
  unsigned int *v30; // rcx
  unsigned int v31; // esi
  unsigned int *v32; // rdx
  unsigned int *v33; // rax
  _BYTE *v34; // rax
  _QWORD *v35; // rdi
  __int64 *v36; // rsi
  char v37; // al
  __int64 v38; // r14
  _QWORD *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdi
  __int64 v46; // r14
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r14
  unsigned int *v52; // rdx
  int v53; // esi
  unsigned int *v54; // rcx
  unsigned int *v55; // rax
  __int64 v56; // rsi
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r14
  unsigned int *v60; // rdx
  int v61; // esi
  unsigned int *v62; // rcx
  unsigned int *v63; // rax
  char *v64; // rax
  unsigned __int8 **v65; // rdx
  unsigned __int8 *v66; // rsi
  char v67; // al
  unsigned __int8 *v68; // r14
  __int64 v69; // rax
  __int64 (__fastcall *v70)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // r14
  __int64 v74; // rdx
  unsigned int *v75; // r13
  unsigned int *v76; // rbx
  __int64 v77; // rdx
  unsigned int v78; // esi
  __int64 v79; // rdx
  char v80; // al
  char v81; // dl
  unsigned __int8 *v82; // rax
  char v83; // r15
  __int64 v84; // r14
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rsi
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // r14
  unsigned int *v91; // rdx
  int v92; // esi
  unsigned int *v93; // rcx
  unsigned int *v94; // rax
  __int64 v95; // rsi
  int v96; // eax
  unsigned __int8 *v97; // r14
  __int64 (__fastcall *v98)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v99; // r15
  _BYTE *v100; // rax
  __int64 v101; // r15
  _QWORD *v102; // rax
  __int64 v103; // r14
  unsigned int *v104; // rbx
  unsigned int *v105; // r15
  __int64 v106; // rdx
  unsigned int v107; // esi
  unsigned int *v108; // rdi
  char v109; // al
  __int64 v110; // rax
  unsigned int v111; // esi
  int v112; // eax
  int v113; // eax
  __int64 v114; // rax
  __int64 v115; // r12
  __int64 v116; // r12
  __int64 v117; // r14
  unsigned int *v118; // rsi
  __int64 v119; // r8
  __int64 v120; // r9
  unsigned int *v121; // r14
  unsigned int *v122; // rdx
  int v123; // esi
  unsigned int *v124; // rcx
  unsigned int *v125; // rax
  __int64 **v126; // r15
  __int64 v127; // r10
  __int64 (__fastcall *v128)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 (__fastcall *v129)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v130; // rax
  unsigned __int8 *v131; // r14
  __int64 v132; // rax
  __int64 (__fastcall *v133)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v134; // r15
  __int64 v135; // rax
  __int64 **v136; // r10
  __int64 (__fastcall *v137)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v138; // rax
  __int64 **v139; // r10
  __int64 (__fastcall *v140)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v141; // rax
  __int64 (__fastcall *v142)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v143; // r9
  __int64 (__fastcall *v144)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v145; // rdx
  __int64 v146; // r13
  unsigned int *v147; // rbx
  unsigned int *v148; // r14
  __int64 v149; // rdx
  unsigned int v150; // esi
  unsigned int *v151; // rbx
  unsigned int *v152; // r14
  __int64 v153; // rdx
  unsigned int v154; // esi
  unsigned int *v155; // rbx
  unsigned int *v156; // r14
  __int64 v157; // rdx
  unsigned int v158; // esi
  __int64 (__fastcall *v159)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v160; // r15
  __int64 v161; // rdx
  __int64 v162; // r13
  unsigned int *v163; // r15
  unsigned int *v164; // rbx
  __int64 v165; // rdx
  unsigned int v166; // esi
  _QWORD *v167; // rax
  unsigned int *v168; // rbx
  unsigned int *v169; // r13
  __int64 v170; // rdx
  unsigned int v171; // esi
  _QWORD *v172; // rax
  __int64 v173; // rdx
  unsigned int *v174; // r13
  unsigned int *v175; // rbx
  __int64 v176; // rdx
  unsigned int v177; // esi
  unsigned int *v178; // rbx
  unsigned int *v179; // r15
  __int64 v180; // rdx
  unsigned int v181; // esi
  unsigned int *v182; // rbx
  unsigned int *v183; // r15
  __int64 v184; // rdx
  unsigned int v185; // esi
  __int64 v186; // rdx
  __int64 v187; // rbx
  unsigned int *v188; // r12
  unsigned int *v189; // r14
  __int64 v190; // rdx
  unsigned int v191; // esi
  __int64 v192; // rax
  __int64 v193; // rax
  __int64 v194; // rax
  unsigned int *v195; // rbx
  __int64 v196; // rax
  unsigned int *v197; // r12
  __int64 v198; // r14
  __int64 v199; // rdx
  unsigned int v200; // esi
  __int64 v201; // r15
  _QWORD *v202; // rdi
  __int64 v203; // rax
  __int64 v204; // rax
  __int64 v205; // rax
  __int64 v206; // rdx
  __int64 v207; // rax
  _QWORD *v208; // rax
  __int64 v209; // r14
  __int64 v210; // rdx
  __int64 v211; // rax
  unsigned int *v212; // r12
  unsigned int *v213; // rbx
  __int64 v214; // r15
  __int64 v215; // rdx
  unsigned int v216; // esi
  unsigned int *v217; // rcx
  unsigned __int64 v218; // rdi
  unsigned int *v219; // rdx
  unsigned int *v220; // rdx
  unsigned __int64 v221; // rdi
  unsigned int *v222; // rdx
  unsigned __int64 v223; // rdi
  unsigned int *v224; // rdx
  __int64 v225; // [rsp+8h] [rbp-368h]
  __int64 v226; // [rsp+10h] [rbp-360h]
  __int64 v227; // [rsp+10h] [rbp-360h]
  int v228; // [rsp+10h] [rbp-360h]
  __int64 v229; // [rsp+10h] [rbp-360h]
  __int64 v230; // [rsp+20h] [rbp-350h]
  char v231; // [rsp+20h] [rbp-350h]
  unsigned __int8 *v232; // [rsp+30h] [rbp-340h]
  char v233; // [rsp+30h] [rbp-340h]
  __int64 v234; // [rsp+30h] [rbp-340h]
  __int64 v235; // [rsp+30h] [rbp-340h]
  char v236; // [rsp+30h] [rbp-340h]
  __int64 v237; // [rsp+30h] [rbp-340h]
  __int64 v238; // [rsp+30h] [rbp-340h]
  __int64 v239; // [rsp+38h] [rbp-338h]
  __int64 v240; // [rsp+38h] [rbp-338h]
  __int64 v241; // [rsp+58h] [rbp-318h]
  int v242; // [rsp+60h] [rbp-310h]
  unsigned __int8 *v243; // [rsp+60h] [rbp-310h]
  __int64 v244; // [rsp+60h] [rbp-310h]
  __int64 v245; // [rsp+70h] [rbp-300h]
  __int64 v246; // [rsp+70h] [rbp-300h]
  __int64 v247; // [rsp+70h] [rbp-300h]
  __int64 v248; // [rsp+70h] [rbp-300h]
  __int64 v249; // [rsp+78h] [rbp-2F8h]
  unsigned __int64 v250; // [rsp+80h] [rbp-2F0h]
  __int64 v251; // [rsp+80h] [rbp-2F0h]
  __int64 **v252; // [rsp+80h] [rbp-2F0h]
  __int64 **v253; // [rsp+80h] [rbp-2F0h]
  __int64 v254; // [rsp+80h] [rbp-2F0h]
  __int64 v255; // [rsp+80h] [rbp-2F0h]
  __int64 v256; // [rsp+80h] [rbp-2F0h]
  __int64 v257; // [rsp+80h] [rbp-2F0h]
  __int64 v258; // [rsp+80h] [rbp-2F0h]
  __int64 **v259; // [rsp+80h] [rbp-2F0h]
  __int64 v260; // [rsp+80h] [rbp-2F0h]
  __int64 v261; // [rsp+80h] [rbp-2F0h]
  unsigned __int64 v262; // [rsp+90h] [rbp-2E0h]
  unsigned __int8 *v263; // [rsp+90h] [rbp-2E0h]
  __int64 v264; // [rsp+90h] [rbp-2E0h]
  __int64 v265; // [rsp+90h] [rbp-2E0h]
  unsigned __int64 v266; // [rsp+90h] [rbp-2E0h]
  __int64 **v267; // [rsp+90h] [rbp-2E0h]
  __int64 v268; // [rsp+90h] [rbp-2E0h]
  __int64 v269; // [rsp+90h] [rbp-2E0h]
  char v270; // [rsp+90h] [rbp-2E0h]
  char v271; // [rsp+90h] [rbp-2E0h]
  __int64 v272; // [rsp+90h] [rbp-2E0h]
  __int64 **v273; // [rsp+90h] [rbp-2E0h]
  __int64 v274; // [rsp+90h] [rbp-2E0h]
  __int64 v275; // [rsp+90h] [rbp-2E0h]
  __int64 v276[4]; // [rsp+C0h] [rbp-2B0h] BYREF
  __int64 v277; // [rsp+E0h] [rbp-290h] BYREF
  __int64 v278; // [rsp+E8h] [rbp-288h]
  __int64 v279; // [rsp+F0h] [rbp-280h]
  __int64 v280; // [rsp+100h] [rbp-270h] BYREF
  __int64 v281; // [rsp+108h] [rbp-268h]
  unsigned __int64 v282; // [rsp+110h] [rbp-260h]
  __int16 v283; // [rsp+120h] [rbp-250h]
  __int64 v284[4]; // [rsp+130h] [rbp-240h] BYREF
  __int16 v285; // [rsp+150h] [rbp-220h]
  unsigned __int8 v286; // [rsp+160h] [rbp-210h]
  _QWORD v287[2]; // [rsp+168h] [rbp-208h] BYREF
  __int64 v288; // [rsp+178h] [rbp-1F8h]
  _QWORD v289[2]; // [rsp+180h] [rbp-1F0h] BYREF
  __int64 v290; // [rsp+190h] [rbp-1E0h]
  __int64 v291; // [rsp+1A0h] [rbp-1D0h] BYREF
  char *v292; // [rsp+1A8h] [rbp-1C8h]
  __int64 v293; // [rsp+1B0h] [rbp-1C0h]
  int v294; // [rsp+1B8h] [rbp-1B8h]
  char v295; // [rsp+1BCh] [rbp-1B4h]
  char v296; // [rsp+1C0h] [rbp-1B0h] BYREF
  unsigned int *v297; // [rsp+1E0h] [rbp-190h] BYREF
  char *v298; // [rsp+1E8h] [rbp-188h]
  __int64 v299; // [rsp+1F0h] [rbp-180h]
  int v300; // [rsp+1F8h] [rbp-178h]
  char v301; // [rsp+1FCh] [rbp-174h]
  char v302; // [rsp+200h] [rbp-170h] BYREF
  unsigned int *v303; // [rsp+220h] [rbp-150h] BYREF
  unsigned int v304; // [rsp+228h] [rbp-148h]
  unsigned int v305; // [rsp+22Ch] [rbp-144h]
  _BYTE v306[16]; // [rsp+230h] [rbp-140h] BYREF
  __int16 v307; // [rsp+240h] [rbp-130h]
  __int64 v308; // [rsp+258h] [rbp-118h]
  __int64 v309; // [rsp+260h] [rbp-110h]
  __int64 v310; // [rsp+278h] [rbp-F8h]
  void *v311; // [rsp+2A0h] [rbp-D0h]
  unsigned int *v312; // [rsp+2B0h] [rbp-C0h] BYREF
  unsigned __int64 v313; // [rsp+2B8h] [rbp-B8h] BYREF
  __int64 v314; // [rsp+2C0h] [rbp-B0h] BYREF
  __int64 v315; // [rsp+2C8h] [rbp-A8h]
  unsigned __int64 v316[2]; // [rsp+2D0h] [rbp-A0h] BYREF
  __int64 v317; // [rsp+2E0h] [rbp-90h]
  __m128i v318; // [rsp+2E8h] [rbp-88h] BYREF
  __int64 v319; // [rsp+2F8h] [rbp-78h]
  void **v320; // [rsp+300h] [rbp-70h]
  _QWORD *v321; // [rsp+308h] [rbp-68h]
  __int64 v322; // [rsp+310h] [rbp-60h]
  int v323; // [rsp+318h] [rbp-58h]
  __int16 v324; // [rsp+31Ch] [rbp-54h]
  char v325; // [rsp+31Eh] [rbp-52h]
  __int64 v326; // [rsp+320h] [rbp-50h]
  __int64 v327; // [rsp+328h] [rbp-48h]
  void *v328; // [rsp+330h] [rbp-40h] BYREF
  _QWORD v329[7]; // [rsp+338h] [rbp-38h] BYREF

  v3 = *(_BYTE *)a1;
  if ( !*(_BYTE *)a1 )
    return 0;
  v4 = *(char **)(a1 + 8);
  v5 = a1;
  v6 = a2;
  if ( (v4[7] & 0x40) != 0 )
    v7 = (__int64 *)*((_QWORD *)v4 - 1);
  else
    v7 = (__int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v8 = *v7;
  v9 = v7[4];
  v10 = *v4;
  v287[0] = 0;
  v287[1] = 0;
  v288 = v8;
  v286 = v10 == 52 || v10 == 49;
  if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
    sub_BD73F0((__int64)v287);
  v290 = v9;
  v289[0] = 0;
  v289[1] = 0;
  if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
    sub_BD73F0((__int64)v289);
  v11 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v11 )
  {
    v12 = v290;
    v13 = 1;
    v14 = *(_QWORD *)(a2 + 8);
    for ( i = (v11 - 1) & (v286 ^ v290 ^ v288); ; i = (v11 - 1) & v17 )
    {
      v16 = (unsigned int *)(v14 + 72LL * i);
      if ( v286 == *(_BYTE *)v16 && v288 == *((_QWORD *)v16 + 3) && v290 == *((_QWORD *)v16 + 6) )
        break;
      if ( !*(_BYTE *)v16 && !*((_QWORD *)v16 + 3) && !*((_QWORD *)v16 + 6) )
        goto LABEL_32;
      v17 = v13 + i;
      ++v13;
    }
    if ( v16 != (unsigned int *)(v14 + 72 * v11) )
      goto LABEL_19;
  }
LABEL_32:
  v20 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(v20 + 7) & 0x40) != 0 )
    v21 = *(_QWORD *)(v20 - 8);
  else
    v21 = v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF);
  v22 = *(unsigned __int8 **)v21;
  v23 = *(_QWORD *)(v21 + 32);
  v291 = 0;
  v262 = v23;
  v250 = (unsigned __int64)v22;
  v292 = &v296;
  v293 = 4;
  v294 = 0;
  v295 = 1;
  v24 = sub_318F620(a1, v22, (__int64)&v291);
  if ( v24 != 2 )
  {
    v297 = 0;
    v298 = &v302;
    v299 = 4;
    v300 = 0;
    v301 = 1;
    v242 = sub_318F620(a1, (unsigned __int8 *)v262, (__int64)&v297);
    if ( v242 == 2 )
    {
LABEL_133:
      v3 = 0;
      goto LABEL_134;
    }
    if ( v24 | v242 )
    {
      if ( *(_BYTE *)v262 == 17
        || *(_BYTE *)v262 == 78
        && *(_QWORD *)(v262 + 40) == *(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL)
        && **(_BYTE **)(v262 - 32) == 17 )
      {
        goto LABEL_133;
      }
      sub_2412230((__int64)&v303, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 24) + 48LL, 0, 0, v25, 0, 0);
      v26 = *(unsigned int **)(*(_QWORD *)(a1 + 8) + 48LL);
      v312 = v26;
      if ( v26 && (sub_B96E90((__int64)&v312, (__int64)v26, 1), (v29 = v312) != 0) )
      {
        v30 = v303;
        v31 = v304;
        v32 = &v303[4 * v304];
        if ( v303 != v32 )
        {
          v33 = v303;
          while ( *v33 )
          {
            v33 += 4;
            if ( v32 == v33 )
              goto LABEL_235;
          }
          *((_QWORD *)v33 + 1) = v312;
LABEL_50:
          sub_B91220((__int64)&v312, (__int64)v29);
LABEL_51:
          v34 = *(_BYTE **)(a1 + 8);
          v35 = *(_QWORD **)(a1 + 24);
          v36 = (__int64 *)(v34 + 24);
          if ( !v24 )
          {
            v37 = *v34;
            if ( v37 != 49 && v37 != 52 && !*(_BYTE *)(v5 + 32) )
            {
              LOWORD(v316[0]) = 257;
              v201 = sub_AA8550(v35, v36, 0, (__int64)&v312, 0);
              v202 = (_QWORD *)((*(_QWORD *)(*(_QWORD *)(v5 + 24) + 48LL) & 0xFFFFFFFFFFFFFFF8LL) - 24);
              if ( (*(_QWORD *)(*(_QWORD *)(v5 + 24) + 48LL) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                v202 = 0;
              sub_B43D60(v202);
              v203 = *(_QWORD *)(v5 + 24);
              v281 = 0;
              v280 = v203;
              v204 = *(_QWORD *)(v5 + 8);
              v282 = 0;
              v281 = sub_AD64C0(*(_QWORD *)(v204 + 8), 0, 0);
              v282 = v250;
              sub_3190580(v284, v5, v201);
              v205 = sub_318FE00(v5, v284, &v280, v201);
              v241 = v206;
              LOWORD(v316[0]) = 257;
              v244 = v205;
              v207 = sub_92B530(&v303, 0x23u, v250, (_BYTE *)v262, (__int64)&v312);
              LOWORD(v316[0]) = 257;
              v248 = v207;
              v261 = v284[0];
              v208 = sub_BD2C40(72, 3u);
              v209 = (__int64)v208;
              if ( v208 )
                sub_B4C9A0((__int64)v208, v261, v201, v248, 3u, v261, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, unsigned int **, __int64, __int64))(*(_QWORD *)v310 + 16LL))(
                v310,
                v209,
                &v312,
                v308,
                v309);
              v210 = 4LL * v304;
              if ( v303 != &v303[v210] )
              {
                v211 = v6;
                v275 = v5;
                v212 = v303;
                v213 = &v303[v210];
                v214 = v211;
                do
                {
                  v215 = *((_QWORD *)v212 + 1);
                  v216 = *v212;
                  v212 += 4;
                  sub_B99FD0(v209, v216, v215);
                }
                while ( v213 != v212 );
                v5 = v275;
                v6 = v214;
              }
              v2 = v244;
              v249 = v241;
LABEL_129:
              nullsub_61();
              v311 = &unk_49DA100;
              nullsub_63();
              v108 = v303;
              if ( v303 == (unsigned int *)v306 )
                goto LABEL_134;
              goto LABEL_130;
            }
          }
          LOWORD(v316[0]) = 257;
          v38 = sub_AA8550(v35, v36, 0, (__int64)&v312, 0);
          v239 = v38;
          v39 = (_QWORD *)((*(_QWORD *)(*(_QWORD *)(v5 + 24) + 48LL) & 0xFFFFFFFFFFFFFFF8LL) - 24);
          if ( (*(_QWORD *)(*(_QWORD *)(v5 + 24) + 48LL) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            v39 = 0;
          sub_B43D60(v39);
          sub_3190580(v276, v5, v38);
          v40 = *(_QWORD *)(v5 + 24);
          v278 = 0;
          v279 = 0;
          v41 = *(_QWORD *)(v40 + 72);
          LOWORD(v316[0]) = 257;
          v245 = v41;
          v42 = sub_B2BE50(v41);
          v43 = sub_22077B0(0x50u);
          v44 = v245;
          v45 = v43;
          if ( v43 )
          {
            v246 = v43;
            sub_AA4D50(v43, v42, (__int64)&v312, v44, v239);
            v45 = v246;
          }
          v46 = *(_QWORD *)(v45 + 56);
          v277 = v45;
          v47 = sub_AA48A0(v45);
          v325 = 7;
          v319 = v47;
          v320 = &v328;
          v321 = v329;
          v324 = 512;
          v328 = &unk_49DA100;
          v312 = (unsigned int *)&v314;
          v313 = 0x200000000LL;
          v329[0] = &unk_49DA0B0;
          v318.m128i_i16[4] = 1;
          v322 = 0;
          v323 = 0;
          v326 = 0;
          v327 = 0;
          v317 = v45;
          v318.m128i_i64[0] = v46;
          if ( v46 == v45 + 48 )
          {
LABEL_70:
            v56 = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 48LL);
            v284[0] = v56;
            if ( v56 && (sub_B96E90((__int64)v284, v56, 1), (v59 = v284[0]) != 0) )
            {
              v60 = v312;
              v61 = v313;
              v62 = &v312[4 * (unsigned int)v313];
              if ( v312 != v62 )
              {
                v63 = v312;
                while ( *v63 )
                {
                  v63 += 4;
                  if ( v62 == v63 )
                    goto LABEL_244;
                }
                *((_QWORD *)v63 + 1) = v284[0];
                goto LABEL_77;
              }
LABEL_244:
              if ( (unsigned int)v313 >= (unsigned __int64)HIDWORD(v313) )
              {
                v218 = (unsigned int)v313 + 1LL;
                if ( HIDWORD(v313) < v218 )
                {
                  sub_C8D5F0((__int64)&v312, &v314, v218, 0x10u, v57, v58);
                  v60 = v312;
                }
                v219 = &v60[4 * (unsigned int)v313];
                *(_QWORD *)v219 = 0;
                *((_QWORD *)v219 + 1) = v59;
                v59 = v284[0];
                LODWORD(v313) = v313 + 1;
              }
              else
              {
                if ( v62 )
                {
                  *v62 = 0;
                  *((_QWORD *)v62 + 1) = v59;
                  v61 = v313;
                  v59 = v284[0];
                }
                LODWORD(v313) = v61 + 1;
              }
            }
            else
            {
              sub_93FB40((__int64)&v312, 0);
              v59 = v284[0];
            }
            if ( !v59 )
            {
LABEL_78:
              v64 = *(char **)(v5 + 8);
              if ( (v64[7] & 0x40) != 0 )
                v65 = (unsigned __int8 **)*((_QWORD *)v64 - 1);
              else
                v65 = (unsigned __int8 **)&v64[-32 * (*((_DWORD *)v64 + 1) & 0x7FFFFFF)];
              v66 = v65[4];
              v67 = *v64;
              v68 = *v65;
              v232 = v66;
              if ( v67 != 52 && v67 != 49 )
              {
                v285 = 257;
                v69 = sub_3122580((__int64 *)&v312, v68, v66, (__int64)v284, 0);
                v283 = 257;
                v278 = v69;
                v70 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v320 + 2);
                if ( v70 == sub_9202E0 )
                {
                  if ( *v68 > 0x15u || *v66 > 0x15u )
                  {
LABEL_323:
                    v285 = 257;
                    v237 = sub_B504D0(22, (__int64)v68, (__int64)v66, (__int64)v284, 0, 0);
                    (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                      v321,
                      v237,
                      &v280,
                      v318.m128i_i64[0],
                      v318.m128i_i64[1]);
                    v186 = 4LL * (unsigned int)v313;
                    v71 = v237;
                    if ( v312 != &v312[v186] )
                    {
                      v238 = v5;
                      v187 = v71;
                      v229 = v6;
                      v188 = v312;
                      v189 = &v312[v186];
                      do
                      {
                        v190 = *((_QWORD *)v188 + 1);
                        v191 = *v188;
                        v188 += 4;
                        sub_B99FD0(v187, v191, v190);
                      }
                      while ( v189 != v188 );
                      v71 = v187;
                      v6 = v229;
                      v5 = v238;
                    }
                    goto LABEL_88;
                  }
                  if ( (unsigned __int8)sub_AC47B0(22) )
                    v71 = sub_AD5570(22, (__int64)v68, v66, 0, 0);
                  else
                    v71 = sub_AABE40(0x16u, v68, v66);
                }
                else
                {
                  v71 = v70((__int64)v320, 22u, v68, v66);
                }
                if ( v71 )
                  goto LABEL_88;
                goto LABEL_323;
              }
              v283 = 257;
              v142 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8))*((_QWORD *)*v320 + 3);
              if ( v142 == sub_920250 )
              {
                if ( *v68 > 0x15u || *v66 > 0x15u )
                {
LABEL_290:
                  v285 = 257;
                  v230 = sub_B504D0(20, (__int64)v68, (__int64)v66, (__int64)v284, 0, 0);
                  (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                    v321,
                    v230,
                    &v280,
                    v318.m128i_i64[0],
                    v318.m128i_i64[1]);
                  v143 = v230;
                  v161 = 4LL * (unsigned int)v313;
                  if ( v312 != &v312[v161] )
                  {
                    v231 = v3;
                    v162 = v143;
                    v228 = v24;
                    v163 = &v312[v161];
                    v225 = v5;
                    v164 = v312;
                    do
                    {
                      v165 = *((_QWORD *)v164 + 1);
                      v166 = *v164;
                      v164 += 4;
                      sub_B99FD0(v162, v166, v165);
                    }
                    while ( v163 != v164 );
                    v143 = v162;
                    v24 = v228;
                    v3 = v231;
                    v5 = v225;
                  }
LABEL_225:
                  v278 = v143;
                  v283 = 257;
                  v144 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v320 + 2);
                  if ( v144 == sub_9202E0 )
                  {
                    if ( *v68 > 0x15u || *v232 > 0x15u )
                    {
LABEL_231:
                      v285 = 257;
                      v235 = sub_B504D0(23, (__int64)v68, (__int64)v232, (__int64)v284, 0, 0);
                      (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                        v321,
                        v235,
                        &v280,
                        v318.m128i_i64[0],
                        v318.m128i_i64[1]);
                      v145 = 4LL * (unsigned int)v313;
                      v71 = v235;
                      if ( v312 != &v312[v145] )
                      {
                        v236 = v3;
                        v146 = v71;
                        v227 = v5;
                        v147 = v312;
                        v148 = &v312[v145];
                        do
                        {
                          v149 = *((_QWORD *)v147 + 1);
                          v150 = *v147;
                          v147 += 4;
                          sub_B99FD0(v146, v150, v149);
                        }
                        while ( v148 != v147 );
                        v71 = v146;
                        v5 = v227;
                        v3 = v236;
                      }
LABEL_88:
                      v279 = v71;
                      v285 = 257;
                      v72 = sub_BD2C40(72, 1u);
                      v73 = (__int64)v72;
                      if ( v72 )
                        sub_B4C8F0((__int64)v72, v239, 1u, 0, 0);
                      (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                        v321,
                        v73,
                        v284,
                        v318.m128i_i64[0],
                        v318.m128i_i64[1]);
                      v74 = 4LL * (unsigned int)v313;
                      if ( v312 != &v312[v74] )
                      {
                        v233 = v3;
                        v75 = &v312[v74];
                        v226 = v5;
                        v76 = v312;
                        do
                        {
                          v77 = *((_QWORD *)v76 + 1);
                          v78 = *v76;
                          v76 += 4;
                          sub_B99FD0(v73, v78, v77);
                        }
                        while ( v75 != v76 );
                        v3 = v233;
                        v5 = v226;
                      }
                      nullsub_61();
                      v328 = &unk_49DA100;
                      nullsub_63();
                      if ( v312 != (unsigned int *)&v314 )
                        _libc_free((unsigned __int64)v312);
                      v240 = sub_318FE00(v5, v276, &v277, v239);
                      v234 = v79;
                      v80 = 0;
                      if ( v242 )
                        v80 = v3;
                      v81 = v80;
                      v82 = 0;
                      if ( v242 )
                        v82 = (unsigned __int8 *)v262;
                      v263 = v82;
                      if ( v24 )
                      {
                        v83 = v81 & (v250 != 0);
                      }
                      else
                      {
                        v250 = 0;
                        v83 = 0;
                      }
                      v84 = *(_QWORD *)(v5 + 24);
                      v85 = sub_AA48A0(v84);
                      v323 = 0;
                      v319 = v85;
                      v320 = &v328;
                      v312 = (unsigned int *)&v314;
                      v321 = v329;
                      v324 = 512;
                      v313 = 0x200000000LL;
                      v322 = 0;
                      v328 = &unk_49DA100;
                      v325 = 7;
                      v326 = 0;
                      v329[0] = &unk_49DA0B0;
                      v318.m128i_i64[0] = v84 + 48;
                      v318.m128i_i16[4] = 0;
                      v86 = *(_QWORD *)(v5 + 8);
                      v327 = 0;
                      v317 = v84;
                      v87 = *(_QWORD *)(v86 + 48);
                      v284[0] = v87;
                      if ( v87 && (sub_B96E90((__int64)v284, v87, 1), (v90 = v284[0]) != 0) )
                      {
                        v91 = v312;
                        v92 = v313;
                        v93 = &v312[4 * (unsigned int)v313];
                        if ( v312 != v93 )
                        {
                          v94 = v312;
                          while ( *v94 )
                          {
                            v94 += 4;
                            if ( v93 == v94 )
                              goto LABEL_251;
                          }
                          *((_QWORD *)v94 + 1) = v284[0];
                          goto LABEL_109;
                        }
LABEL_251:
                        if ( (unsigned int)v313 >= (unsigned __int64)HIDWORD(v313) )
                        {
                          if ( HIDWORD(v313) < (unsigned __int64)(unsigned int)v313 + 1 )
                          {
                            sub_C8D5F0((__int64)&v312, &v314, (unsigned int)v313 + 1LL, 0x10u, v88, v89);
                            v91 = v312;
                          }
                          v220 = &v91[4 * (unsigned int)v313];
                          *(_QWORD *)v220 = 0;
                          *((_QWORD *)v220 + 1) = v90;
                          v90 = v284[0];
                          LODWORD(v313) = v313 + 1;
                        }
                        else
                        {
                          if ( v93 )
                          {
                            *v93 = 0;
                            *((_QWORD *)v93 + 1) = v90;
                            v92 = v313;
                            v90 = v284[0];
                          }
                          LODWORD(v313) = v92 + 1;
                        }
                      }
                      else
                      {
                        sub_93FB40((__int64)&v312, 0);
                        v90 = v284[0];
                      }
                      if ( !v90 )
                      {
LABEL_110:
                        if ( !v83 )
                        {
                          v95 = (__int64)v263;
                          if ( v250 )
                            v95 = v250;
                          v251 = v95;
LABEL_114:
                          v96 = *(_DWORD *)(*(_QWORD *)(v5 + 16) + 8LL);
                          v283 = 257;
                          v97 = (unsigned __int8 *)sub_AD64C0(
                                                     *(_QWORD *)(v251 + 8),
                                                     ~(0xFFFFFFFFFFFFFFFFLL >> (64 - BYTE1(v96))),
                                                     0);
                          v98 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v320 + 2);
                          if ( v98 == sub_9202E0 )
                          {
                            if ( *(_BYTE *)v251 > 0x15u || *v97 > 0x15u )
                              goto LABEL_261;
                            if ( (unsigned __int8)sub_AC47B0(28) )
                              v99 = sub_AD5570(28, v251, v97, 0, 0);
                            else
                              v99 = sub_AABE40(0x1Cu, (unsigned __int8 *)v251, v97);
                          }
                          else
                          {
                            v99 = v98((__int64)v320, 28u, (_BYTE *)v251, v97);
                          }
                          if ( v99 )
                          {
LABEL_120:
                            v100 = (_BYTE *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(v5 + 8) + 8LL), 0, 1u);
                            v285 = 257;
                            v264 = sub_92B530(&v312, 0x20u, v99, v100, (__int64)v284);
                            nullsub_61();
                            v328 = &unk_49DA100;
                            nullsub_63();
                            if ( v312 != (unsigned int *)&v314 )
                              _libc_free((unsigned __int64)v312);
                            LOWORD(v316[0]) = 257;
                            v101 = v276[0];
                            v247 = v277;
                            v102 = sub_BD2C40(72, 3u);
                            v103 = (__int64)v102;
                            if ( v102 )
                              sub_B4C9A0((__int64)v102, v101, v247, v264, 3u, 0, 0, 0);
                            (*(void (__fastcall **)(__int64, __int64, unsigned int **, __int64, __int64))(*(_QWORD *)v310 + 16LL))(
                              v310,
                              v103,
                              &v312,
                              v308,
                              v309);
                            if ( v303 != &v303[4 * v304] )
                            {
                              v265 = v5;
                              v104 = v303;
                              v105 = &v303[4 * v304];
                              do
                              {
                                v106 = *((_QWORD *)v104 + 1);
                                v107 = *v104;
                                v104 += 4;
                                sub_B99FD0(v103, v107, v106);
                              }
                              while ( v105 != v104 );
                              v5 = v265;
                            }
                            v2 = v240;
                            v249 = v234;
                            goto LABEL_129;
                          }
LABEL_261:
                          v285 = 257;
                          v99 = sub_B504D0(28, v251, (__int64)v97, (__int64)v284, 0, 0);
                          (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                            v321,
                            v99,
                            &v280,
                            v318.m128i_i64[0],
                            v318.m128i_i64[1]);
                          if ( v312 != &v312[4 * (unsigned int)v313] )
                          {
                            v268 = v5;
                            v151 = v312;
                            v152 = &v312[4 * (unsigned int)v313];
                            do
                            {
                              v153 = *((_QWORD *)v151 + 1);
                              v154 = *v151;
                              v151 += 4;
                              sub_B99FD0(v99, v154, v153);
                            }
                            while ( v152 != v151 );
                            v5 = v268;
                          }
                          goto LABEL_120;
                        }
                        v283 = 257;
                        v159 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v320 + 2);
                        if ( v159 == sub_9202E0 )
                        {
                          if ( *(_BYTE *)v250 > 0x15u || *v263 > 0x15u )
                            goto LABEL_336;
                          if ( (unsigned __int8)sub_AC47B0(29) )
                            v160 = sub_AD5570(29, v250, v263, 0, 0);
                          else
                            v160 = sub_AABE40(0x1Du, (unsigned __int8 *)v250, v263);
                        }
                        else
                        {
                          v160 = v159((__int64)v320, 29u, (_BYTE *)v250, v263);
                        }
                        if ( v160 )
                        {
LABEL_282:
                          v251 = v160;
                          goto LABEL_114;
                        }
LABEL_336:
                        v285 = 257;
                        v160 = sub_B504D0(29, v250, (__int64)v263, (__int64)v284, 0, 0);
                        (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                          v321,
                          v160,
                          &v280,
                          v318.m128i_i64[0],
                          v318.m128i_i64[1]);
                        if ( v312 != &v312[4 * (unsigned int)v313] )
                        {
                          v274 = v5;
                          v195 = &v312[4 * (unsigned int)v313];
                          v196 = v6;
                          v197 = v312;
                          v198 = v196;
                          do
                          {
                            v199 = *((_QWORD *)v197 + 1);
                            v200 = *v197;
                            v197 += 4;
                            sub_B99FD0(v160, v200, v199);
                          }
                          while ( v195 != v197 );
                          v5 = v274;
                          v6 = v198;
                        }
                        goto LABEL_282;
                      }
LABEL_109:
                      sub_B91220((__int64)v284, v90);
                      goto LABEL_110;
                    }
                    if ( (unsigned __int8)sub_AC47B0(23) )
                      v71 = sub_AD5570(23, (__int64)v68, v232, 0, 0);
                    else
                      v71 = sub_AABE40(0x17u, v68, v232);
                  }
                  else
                  {
                    v71 = v144((__int64)v320, 23u, v68, v232);
                  }
                  if ( v71 )
                    goto LABEL_88;
                  goto LABEL_231;
                }
                if ( (unsigned __int8)sub_AC47B0(20) )
                  v143 = sub_AD5570(20, (__int64)v68, v66, 0, 0);
                else
                  v143 = sub_AABE40(0x14u, v68, v66);
              }
              else
              {
                v143 = v142((__int64)v320, 20u, v68, v66, 0);
              }
              if ( v143 )
                goto LABEL_225;
              goto LABEL_290;
            }
LABEL_77:
            sub_B91220((__int64)v284, v59);
            goto LABEL_78;
          }
          if ( v46 )
            v46 -= 24;
          v48 = *(_QWORD *)sub_B46C60(v46);
          v284[0] = v48;
          if ( v48 && (sub_B96E90((__int64)v284, v48, 1), (v51 = v284[0]) != 0) )
          {
            v52 = v312;
            v53 = v313;
            v54 = &v312[4 * (unsigned int)v313];
            if ( v312 != v54 )
            {
              v55 = v312;
              while ( *v55 )
              {
                v55 += 4;
                if ( v54 == v55 )
                  goto LABEL_283;
              }
              *((_QWORD *)v55 + 1) = v284[0];
LABEL_69:
              sub_B91220((__int64)v284, v51);
              goto LABEL_70;
            }
LABEL_283:
            if ( (unsigned int)v313 >= (unsigned __int64)HIDWORD(v313) )
            {
              v223 = (unsigned int)v313 + 1LL;
              if ( HIDWORD(v313) < v223 )
              {
                sub_C8D5F0((__int64)&v312, &v314, v223, 0x10u, v49, v50);
                v52 = v312;
              }
              v224 = &v52[4 * (unsigned int)v313];
              *(_QWORD *)v224 = 0;
              *((_QWORD *)v224 + 1) = v51;
              v51 = v284[0];
              LODWORD(v313) = v313 + 1;
            }
            else
            {
              if ( v54 )
              {
                *v54 = 0;
                *((_QWORD *)v54 + 1) = v51;
                v53 = v313;
                v51 = v284[0];
              }
              LODWORD(v313) = v53 + 1;
            }
          }
          else
          {
            sub_93FB40((__int64)&v312, 0);
            v51 = v284[0];
          }
          if ( !v51 )
            goto LABEL_70;
          goto LABEL_69;
        }
LABEL_235:
        if ( v304 >= (unsigned __int64)v305 )
        {
          if ( v305 < (unsigned __int64)v304 + 1 )
          {
            sub_C8D5F0((__int64)&v303, v306, v304 + 1LL, 0x10u, v27, v28);
            v30 = v303;
          }
          v217 = &v30[4 * v304];
          *(_QWORD *)v217 = 0;
          *((_QWORD *)v217 + 1) = v29;
          v29 = v312;
          ++v304;
        }
        else
        {
          if ( v32 )
          {
            *v32 = 0;
            *((_QWORD *)v32 + 1) = v29;
            v31 = v304;
            v29 = v312;
          }
          v304 = v31 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v303, 0);
        v29 = v312;
      }
      if ( !v29 )
        goto LABEL_51;
      goto LABEL_50;
    }
    v117 = *(_QWORD *)(a1 + 8);
    v319 = sub_BD5C60(v117);
    v320 = &v328;
    v321 = v329;
    v312 = (unsigned int *)&v314;
    v328 = &unk_49DA100;
    v324 = 512;
    v313 = 0x200000000LL;
    v317 = 0;
    v318.m128i_i64[0] = 0;
    v322 = 0;
    v323 = 0;
    v325 = 7;
    v326 = 0;
    v327 = 0;
    v318.m128i_i16[4] = 0;
    v329[0] = &unk_49DA0B0;
    v317 = *(_QWORD *)(v117 + 40);
    v318.m128i_i64[0] = v117 + 24;
    v118 = *(unsigned int **)sub_B46C60(v117);
    v303 = v118;
    if ( v118 && (sub_B96E90((__int64)&v303, (__int64)v118, 1), (v121 = v303) != 0) )
    {
      v122 = v312;
      v123 = v313;
      v124 = &v312[4 * (unsigned int)v313];
      if ( v312 != v124 )
      {
        v125 = v312;
        while ( 1 )
        {
          v119 = *v125;
          if ( !(_DWORD)v119 )
            break;
          v125 += 4;
          if ( v124 == v125 )
            goto LABEL_265;
        }
        *((_QWORD *)v125 + 1) = v303;
        goto LABEL_181;
      }
LABEL_265:
      if ( (unsigned int)v313 >= (unsigned __int64)HIDWORD(v313) )
      {
        v221 = (unsigned int)v313 + 1LL;
        if ( HIDWORD(v313) < v221 )
        {
          sub_C8D5F0((__int64)&v312, &v314, v221, 0x10u, v119, v120);
          v122 = v312;
        }
        v222 = &v122[4 * (unsigned int)v313];
        *(_QWORD *)v222 = 0;
        *((_QWORD *)v222 + 1) = v121;
        v121 = v303;
        LODWORD(v313) = v313 + 1;
      }
      else
      {
        if ( v124 )
        {
          *v124 = 0;
          *((_QWORD *)v124 + 1) = v121;
          v123 = v313;
          v121 = v303;
        }
        LODWORD(v313) = v123 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v312, 0);
      v121 = v303;
    }
    if ( !v121 )
    {
LABEL_182:
      v126 = *(__int64 ***)(v5 + 16);
      v285 = 257;
      v127 = *(_QWORD *)(v250 + 8);
      if ( v126 == (__int64 **)v127 )
      {
        v243 = (unsigned __int8 *)v250;
LABEL_189:
        v285 = 257;
        if ( *(_QWORD *)(v262 + 8) == v127 )
        {
          v131 = (unsigned __int8 *)v262;
          goto LABEL_196;
        }
        v129 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v320 + 15);
        if ( v129 == sub_920130 )
        {
          if ( *(_BYTE *)v262 > 0x15u )
          {
LABEL_309:
            v307 = 257;
            v131 = (unsigned __int8 *)sub_B51D30(38, v262, v127, (__int64)&v303, 0, 0);
            (*(void (__fastcall **)(_QWORD *, unsigned __int8 *, __int64 *, __int64, __int64))(*v321 + 16LL))(
              v321,
              v131,
              v284,
              v318.m128i_i64[0],
              v318.m128i_i64[1]);
            if ( v312 != &v312[4 * (unsigned int)v313] )
            {
              v272 = v5;
              v178 = v312;
              v179 = &v312[4 * (unsigned int)v313];
              do
              {
                v180 = *((_QWORD *)v178 + 1);
                v181 = *v178;
                v178 += 4;
                sub_B99FD0((__int64)v131, v181, v180);
              }
              while ( v179 != v178 );
              v5 = v272;
            }
LABEL_196:
            v307 = 257;
            v132 = sub_3122580((__int64 *)&v312, v243, v131, (__int64)&v303, 0);
            v285 = 257;
            v266 = v132;
            v133 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v320 + 2);
            if ( v133 == sub_9202E0 )
            {
              if ( *v243 > 0x15u || *v131 > 0x15u )
              {
LABEL_272:
                v307 = 257;
                v134 = sub_B504D0(22, (__int64)v243, (__int64)v131, (__int64)&v303, 0, 0);
                (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                  v321,
                  v134,
                  v284,
                  v318.m128i_i64[0],
                  v318.m128i_i64[1]);
                if ( v312 != &v312[4 * (unsigned int)v313] )
                {
                  v254 = v5;
                  v155 = v312;
                  v156 = &v312[4 * (unsigned int)v313];
                  do
                  {
                    v157 = *((_QWORD *)v155 + 1);
                    v158 = *v155;
                    v155 += 4;
                    sub_B99FD0(v134, v158, v157);
                  }
                  while ( v156 != v155 );
                  v5 = v254;
                }
LABEL_202:
                v135 = *(_QWORD *)(v5 + 8);
                v285 = 257;
                v136 = *(__int64 ***)(v135 + 8);
                if ( v136 == *(__int64 ***)(v266 + 8) )
                {
                  v2 = v266;
LABEL_210:
                  v285 = 257;
                  v139 = *(__int64 ***)(v135 + 8);
                  if ( v139 == *(__int64 ***)(v134 + 8) )
                  {
                    v249 = v134;
                    goto LABEL_217;
                  }
                  v140 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v320 + 15);
                  if ( v140 == sub_920130 )
                  {
                    if ( *(_BYTE *)v134 > 0x15u )
                    {
LABEL_297:
                      v269 = (__int64)v139;
                      v307 = 257;
                      v167 = sub_BD2C40(72, 1u);
                      v249 = (__int64)v167;
                      if ( v167 )
                        sub_B515B0((__int64)v167, v134, v269, (__int64)&v303, 0, 0);
                      (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                        v321,
                        v249,
                        v284,
                        v318.m128i_i64[0],
                        v318.m128i_i64[1]);
                      if ( v312 != &v312[4 * (unsigned int)v313] )
                      {
                        v255 = v5;
                        v168 = v312;
                        v270 = v3;
                        v169 = &v312[4 * (unsigned int)v313];
                        do
                        {
                          v170 = *((_QWORD *)v168 + 1);
                          v171 = *v168;
                          v168 += 4;
                          sub_B99FD0(v249, v171, v170);
                        }
                        while ( v169 != v168 );
                        v3 = v270;
                        v5 = v255;
                      }
LABEL_217:
                      nullsub_61();
                      v328 = &unk_49DA100;
                      nullsub_63();
                      v108 = v312;
                      if ( v312 == (unsigned int *)&v314 )
                      {
LABEL_134:
                        if ( !v301 )
                          _libc_free((unsigned __int64)v298);
                        if ( !v295 )
                          _libc_free((unsigned __int64)v292);
                        if ( !v3 )
                          goto LABEL_37;
                        v313 = 0;
                        v314 = 0;
                        LOBYTE(v312) = v286;
                        v315 = v288;
                        if ( v288 != -4096 && v288 != 0 && v288 != -8192 )
                          sub_BD6050(&v313, v287[0] & 0xFFFFFFFFFFFFFFF8LL);
                        v316[0] = 0;
                        v316[1] = 0;
                        v317 = v290;
                        if ( v290 != 0 && v290 != -4096 && v290 != -8192 )
                          sub_BD6050(v316, v289[0] & 0xFFFFFFFFFFFFFFF8LL);
                        v318.m128i_i64[0] = v2;
                        v318.m128i_i64[1] = v249;
                        v109 = sub_318EE30(v6, (unsigned __int8 *)&v312, &v297);
                        v16 = v297;
                        if ( v109 )
                        {
                          v110 = v317;
LABEL_147:
                          if ( v110 != 0 && v110 != -4096 && v110 != -8192 )
                            sub_BD60C0(v316);
                          if ( v315 != 0 && v315 != -4096 && v315 != -8192 )
                            sub_BD60C0(&v313);
                          v12 = v290;
LABEL_19:
                          if ( (unsigned __int8)(**(_BYTE **)(v5 + 8) - 48) > 1u )
                            v18 = *((_QWORD *)v16 + 8);
                          else
                            v18 = *((_QWORD *)v16 + 7);
                          goto LABEL_21;
                        }
                        v111 = *(_DWORD *)(v6 + 24);
                        v112 = *(_DWORD *)(v6 + 16);
                        v303 = v297;
                        ++*(_QWORD *)v6;
                        v113 = v112 + 1;
                        if ( 4 * v113 >= 3 * v111 )
                        {
                          v111 *= 2;
                        }
                        else if ( v111 - *(_DWORD *)(v6 + 20) - v113 > v111 >> 3 )
                        {
                          goto LABEL_156;
                        }
                        sub_318EF00(v6, v111);
                        sub_318EE30(v6, (unsigned __int8 *)&v312, &v303);
                        v16 = v303;
                        v113 = *(_DWORD *)(v6 + 16) + 1;
LABEL_156:
                        *(_DWORD *)(v6 + 16) = v113;
                        if ( *(_BYTE *)v16 || *((_QWORD *)v16 + 3) || *((_QWORD *)v16 + 6) )
                          --*(_DWORD *)(v6 + 20);
                        *(_BYTE *)v16 = (_BYTE)v312;
                        v114 = *((_QWORD *)v16 + 3);
                        v115 = v315;
                        if ( v315 != v114 )
                        {
                          if ( v114 != 0 && v114 != -4096 && v114 != -8192 )
                            sub_BD60C0((_QWORD *)v16 + 1);
                          *((_QWORD *)v16 + 3) = v115;
                          if ( v115 != 0 && v115 != -4096 && v115 != -8192 )
                            sub_BD73F0((__int64)(v16 + 2));
                        }
                        v116 = v317;
                        v110 = *((_QWORD *)v16 + 6);
                        if ( v317 != v110 )
                        {
                          if ( v110 != 0 && v110 != -4096 && v110 != -8192 )
                            sub_BD60C0((_QWORD *)v16 + 4);
                          *((_QWORD *)v16 + 6) = v116;
                          if ( v116 != 0 && v116 != -4096 && v116 != -8192 )
                            sub_BD73F0((__int64)(v16 + 8));
                          v110 = v317;
                        }
                        *(__m128i *)(v16 + 14) = _mm_loadu_si128(&v318);
                        goto LABEL_147;
                      }
LABEL_130:
                      _libc_free((unsigned __int64)v108);
                      goto LABEL_134;
                    }
                    v267 = v139;
                    if ( (unsigned __int8)sub_AC4810(0x27u) )
                      v141 = sub_ADAB70(39, v134, v267, 0);
                    else
                      v141 = sub_AA93C0(0x27u, v134, (__int64)v267);
                    v139 = v267;
                    v249 = v141;
                  }
                  else
                  {
                    v273 = v139;
                    v192 = v140((__int64)v320, 39u, (_BYTE *)v134, (__int64)v139);
                    v139 = v273;
                    v249 = v192;
                  }
                  if ( v249 )
                    goto LABEL_217;
                  goto LABEL_297;
                }
                v137 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v320 + 15);
                if ( v137 == sub_920130 )
                {
                  if ( *(_BYTE *)v266 > 0x15u )
                    goto LABEL_303;
                  v253 = v136;
                  if ( (unsigned __int8)sub_AC4810(0x27u) )
                    v138 = sub_ADAB70(39, v266, v253, 0);
                  else
                    v138 = sub_AA93C0(0x27u, v266, (__int64)v253);
                  v136 = v253;
                  v2 = v138;
                }
                else
                {
                  v259 = v136;
                  v193 = v137((__int64)v320, 39u, (_BYTE *)v266, (__int64)v136);
                  v136 = v259;
                  v2 = v193;
                }
                if ( v2 )
                {
LABEL_209:
                  v135 = *(_QWORD *)(v5 + 8);
                  goto LABEL_210;
                }
LABEL_303:
                v256 = (__int64)v136;
                v307 = 257;
                v172 = sub_BD2C40(72, 1u);
                v2 = (__int64)v172;
                if ( v172 )
                  sub_B515B0((__int64)v172, v266, v256, (__int64)&v303, 0, 0);
                (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v321 + 16LL))(
                  v321,
                  v2,
                  v284,
                  v318.m128i_i64[0],
                  v318.m128i_i64[1]);
                v173 = 4LL * (unsigned int)v313;
                if ( v312 != &v312[v173] )
                {
                  v271 = v3;
                  v174 = &v312[v173];
                  v257 = v5;
                  v175 = v312;
                  do
                  {
                    v176 = *((_QWORD *)v175 + 1);
                    v177 = *v175;
                    v175 += 4;
                    sub_B99FD0(v2, v177, v176);
                  }
                  while ( v174 != v175 );
                  v3 = v271;
                  v5 = v257;
                }
                goto LABEL_209;
              }
              if ( (unsigned __int8)sub_AC47B0(22) )
                v134 = sub_AD5570(22, (__int64)v243, v131, 0, 0);
              else
                v134 = sub_AABE40(0x16u, v243, v131);
            }
            else
            {
              v134 = v133((__int64)v320, 22u, v243, v131);
            }
            if ( v134 )
              goto LABEL_202;
            goto LABEL_272;
          }
          v252 = (__int64 **)v127;
          if ( (unsigned __int8)sub_AC4810(0x26u) )
            v130 = sub_ADAB70(38, v262, v252, 0);
          else
            v130 = sub_AA93C0(0x26u, v262, (__int64)v252);
          v127 = (__int64)v252;
          v131 = (unsigned __int8 *)v130;
        }
        else
        {
          v260 = v127;
          v194 = v129((__int64)v320, 38u, (_BYTE *)v262, v127);
          v127 = v260;
          v131 = (unsigned __int8 *)v194;
        }
        if ( v131 )
          goto LABEL_196;
        goto LABEL_309;
      }
      v128 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v320 + 15);
      if ( v128 == sub_920130 )
      {
        if ( *(_BYTE *)v250 > 0x15u )
          goto LABEL_313;
        if ( (unsigned __int8)sub_AC4810(0x26u) )
          v243 = (unsigned __int8 *)sub_ADAB70(38, v250, v126, 0);
        else
          v243 = (unsigned __int8 *)sub_AA93C0(0x26u, v250, (__int64)v126);
      }
      else
      {
        v243 = (unsigned __int8 *)v128((__int64)v320, 38u, (_BYTE *)v250, (__int64)v126);
      }
      if ( v243 )
      {
LABEL_188:
        v127 = *(_QWORD *)(v5 + 16);
        goto LABEL_189;
      }
LABEL_313:
      v307 = 257;
      v243 = (unsigned __int8 *)sub_B51D30(38, v250, (__int64)v126, (__int64)&v303, 0, 0);
      (*(void (__fastcall **)(_QWORD *, unsigned __int8 *, __int64 *, __int64, __int64))(*v321 + 16LL))(
        v321,
        v243,
        v284,
        v318.m128i_i64[0],
        v318.m128i_i64[1]);
      if ( v312 != &v312[4 * (unsigned int)v313] )
      {
        v258 = v5;
        v182 = v312;
        v183 = &v312[4 * (unsigned int)v313];
        do
        {
          v184 = *((_QWORD *)v182 + 1);
          v185 = *v182;
          v182 += 4;
          sub_B99FD0((__int64)v243, v185, v184);
        }
        while ( v183 != v182 );
        v5 = v258;
      }
      goto LABEL_188;
    }
LABEL_181:
    sub_B91220((__int64)&v303, (__int64)v121);
    goto LABEL_182;
  }
  if ( !v295 )
    _libc_free((unsigned __int64)v292);
LABEL_37:
  v12 = v290;
  v18 = 0;
LABEL_21:
  if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
    sub_BD60C0(v289);
  if ( v288 != 0 && v288 != -4096 && v288 != -8192 )
    sub_BD60C0(v287);
  return v18;
}
