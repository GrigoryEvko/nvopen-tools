// Function: sub_1A7F3A0
// Address: 0x1a7f3a0
//
__int64 __fastcall sub_1A7F3A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r12
  int v12; // ebx
  __int64 *v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  char v17; // cl
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r12
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdi
  unsigned __int8 v29; // al
  bool v30; // zf
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // r14
  __int64 v35; // r12
  int v36; // r15d
  unsigned __int64 v37; // r10
  _QWORD *v38; // rbx
  __int64 v39; // r13
  unsigned int v40; // r12d
  unsigned __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rbx
  __int64 v50; // r14
  unsigned __int64 v51; // r15
  unsigned __int64 v52; // rax
  _QWORD *v53; // r10
  _QWORD *j; // rax
  __int64 v55; // rdx
  unsigned int v56; // edi
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rsi
  __int64 v60; // rcx
  __int64 v61; // rdi
  __int64 v62; // rsi
  __int64 v63; // rax
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // rax
  __int64 v68; // rcx
  unsigned __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // rbx
  __int64 v74; // rax
  double v75; // xmm4_8
  double v76; // xmm5_8
  _QWORD *v77; // rax
  _QWORD *v78; // r14
  unsigned __int64 v79; // r15
  __int64 v80; // r13
  int v81; // ebx
  __int64 v82; // rax
  int v83; // edx
  __int64 v84; // r12
  __int64 *v85; // rbx
  __int64 v86; // r15
  __int64 v87; // rdx
  unsigned int v88; // esi
  int v89; // r11d
  __int64 v90; // rax
  _QWORD *v91; // r12
  int v92; // ecx
  __int64 v93; // rcx
  unsigned __int64 *v94; // rdi
  __int64 *v95; // rcx
  __int64 v96; // r12
  unsigned int v97; // esi
  __int64 v98; // r8
  __int64 v99; // r9
  int v100; // eax
  __int64 v101; // rcx
  _QWORD *v102; // rax
  int v103; // edi
  __int64 v104; // rsi
  unsigned __int64 *v105; // rdi
  __int64 *v106; // rsi
  __int64 v107; // rsi
  int v108; // eax
  __int64 v109; // rax
  int v110; // ecx
  __int64 v111; // rcx
  __int64 *v112; // rax
  __int64 v113; // rdi
  unsigned __int64 v114; // rcx
  __int64 v115; // rcx
  __int64 v116; // rcx
  __int64 v117; // rsi
  __int64 v118; // r12
  __int64 v119; // rsi
  double v120; // xmm4_8
  double v121; // xmm5_8
  __int64 *v122; // rbx
  unsigned int v123; // eax
  _QWORD *v124; // r14
  unsigned __int64 v125; // r15
  __int64 v126; // rax
  _QWORD *v127; // r15
  __int64 v128; // rdx
  __int64 v129; // rax
  __int64 result; // rax
  unsigned __int64 v131; // rax
  _QWORD *v132; // rbx
  _QWORD *v133; // r15
  int v134; // r12d
  unsigned __int64 v135; // r13
  __int64 v136; // r8
  unsigned int v137; // ecx
  __int64 v138; // rdi
  _QWORD *v139; // r10
  int v140; // edi
  int v141; // r11d
  int v142; // r11d
  __int64 v143; // r9
  int v144; // edi
  _QWORD *v145; // rsi
  __int64 v146; // rcx
  __int64 v147; // r8
  unsigned int v148; // edi
  int v149; // r11d
  int v150; // edi
  int v151; // eax
  __int64 v152; // r11
  _QWORD *v153; // rdi
  unsigned int v154; // esi
  char v155; // bl
  __int64 v156; // rax
  char v157; // dl
  __int64 v158; // r13
  __int64 v159; // rax
  __int64 v160; // r14
  unsigned __int64 v161; // r12
  __int64 v162; // rax
  unsigned int v163; // eax
  _QWORD *v164; // r12
  _QWORD *v165; // r14
  __int64 v166; // rsi
  __int64 v167; // r14
  int v168; // r8d
  int v169; // r9d
  __int64 v170; // r12
  __int64 v171; // rbx
  signed __int64 v172; // rbx
  __int64 *v173; // r12
  _QWORD *v174; // rax
  int v175; // r12d
  __int64 v176; // rax
  _QWORD *v177; // rax
  __int64 *v178; // rbx
  unsigned int v179; // edx
  _QWORD *v180; // r15
  unsigned __int64 v181; // r12
  __int64 v182; // rdx
  _QWORD *v183; // r12
  __int64 v184; // rcx
  __int64 v185; // rdx
  unsigned int v186; // eax
  _QWORD *v187; // r12
  _QWORD *v188; // r13
  __int64 v189; // rsi
  __int64 v190; // r11
  unsigned int v191; // esi
  int v192; // r11d
  __int64 v193; // r9
  int v194; // edi
  __int64 v195; // rcx
  __int64 v196; // r8
  char v198; // [rsp+Fh] [rbp-1C1h]
  __int64 v199; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v200; // [rsp+18h] [rbp-1B8h]
  __int64 *v201; // [rsp+20h] [rbp-1B0h]
  _QWORD *v202; // [rsp+30h] [rbp-1A0h]
  unsigned __int64 v203; // [rsp+38h] [rbp-198h]
  _QWORD *v204; // [rsp+38h] [rbp-198h]
  __int64 v205; // [rsp+40h] [rbp-190h]
  unsigned __int64 *v206; // [rsp+40h] [rbp-190h]
  _QWORD *v207; // [rsp+40h] [rbp-190h]
  _QWORD *v208; // [rsp+40h] [rbp-190h]
  __int64 v209; // [rsp+48h] [rbp-188h]
  unsigned __int64 v210; // [rsp+48h] [rbp-188h]
  __int64 v211; // [rsp+50h] [rbp-180h]
  _QWORD *v212; // [rsp+50h] [rbp-180h]
  __int64 v213; // [rsp+50h] [rbp-180h]
  __int64 v214; // [rsp+58h] [rbp-178h]
  unsigned int v215; // [rsp+68h] [rbp-168h]
  __int64 v216; // [rsp+70h] [rbp-160h]
  __int64 v218; // [rsp+80h] [rbp-150h]
  __int64 v219; // [rsp+88h] [rbp-148h]
  char v220; // [rsp+88h] [rbp-148h]
  __int64 v221; // [rsp+90h] [rbp-140h]
  int v222; // [rsp+90h] [rbp-140h]
  _QWORD *v223; // [rsp+90h] [rbp-140h]
  __int64 v224; // [rsp+98h] [rbp-138h] BYREF
  __int64 v225; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v226; // [rsp+A8h] [rbp-128h] BYREF
  __int64 v227; // [rsp+B0h] [rbp-120h]
  __int64 v228; // [rsp+B8h] [rbp-118h]
  __int64 v229; // [rsp+C0h] [rbp-110h]
  __int64 *v230; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v231; // [rsp+D8h] [rbp-F8h] BYREF
  __int64 v232; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v233; // [rsp+E8h] [rbp-E8h]
  __int64 *i; // [rsp+F0h] [rbp-E0h]
  __int64 v235[2]; // [rsp+100h] [rbp-D0h] BYREF
  char v236; // [rsp+110h] [rbp-C0h]
  char v237; // [rsp+111h] [rbp-BFh]
  _BYTE v238[80]; // [rsp+150h] [rbp-80h] BYREF
  _BYTE v239[48]; // [rsp+1A0h] [rbp-30h] BYREF

  v224 = a1;
  v201 = (__int64 *)(a1 & 0xFFFFFFFFFFFFFFF8LL);
  v214 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40);
  v198 = (a1 >> 2) & 1;
  if ( ((a1 >> 2) & 1) != 0 && (*(_WORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 3) == 2 )
  {
    v218 = 0;
  }
  else
  {
    v218 = v201[1];
    if ( v218 )
    {
      v237 = 1;
      v235[0] = (__int64)"phi.call";
      v236 = 3;
      v11 = *v201;
      v12 = *(_DWORD *)(a2 + 8);
      v218 = sub_1648B60(64);
      if ( v218 )
      {
        sub_15F1EA0(v218, v11, 53, 0, 0, 0);
        *(_DWORD *)(v218 + 56) = v12;
        sub_164B780(v218, v235);
        sub_1648880(v218, *(_DWORD *)(v218 + 56), 1);
      }
      v198 = 0;
    }
    else
    {
      v198 = 0;
    }
  }
  v13 = v235;
  do
  {
    *v13 = 0;
    *((_DWORD *)v13 + 6) = 128;
    v14 = (_QWORD *)sub_22077B0(0x2000);
    v15 = *((unsigned int *)v13 + 6);
    *((_DWORD *)v13 + 4) = 0;
    v13[1] = (__int64)v14;
    *((_DWORD *)v13 + 5) = 0;
    v16 = &v14[8 * v15];
    v231 = 2;
    v232 = 0;
    v233 = -8;
    v230 = (__int64 *)&unk_49E6B50;
    for ( i = 0; v16 != v14; v14 += 8 )
    {
      if ( v14 )
      {
        v17 = v231;
        v14[2] = 0;
        v14[3] = -8;
        *v14 = &unk_49E6B50;
        v14[1] = v17 & 6;
        v14[4] = i;
      }
    }
    *((_BYTE *)v13 + 64) = 0;
    v13 += 10;
    *((_BYTE *)v13 - 7) = 1;
  }
  while ( v239 != (_BYTE *)v13 );
  v18 = 0;
  v19 = 0;
  v215 = 0;
  if ( *(_DWORD *)(a2 + 8) )
  {
    while ( 1 )
    {
      v20 = 56 * v18;
      v21 = *(_QWORD *)(*(_QWORD *)a2 + 56 * v18);
      v22 = &v235[10 * v19];
      v23 = v201[4];
      if ( v23 )
        v23 -= 24;
      v24 = sub_1AB5340(v214, v21, v23, v22, a3);
      v25 = *(_QWORD *)(sub_157EBA0(v24) + 24) & 0xFFFFFFFFFFFFFFF8LL;
      v200 = v25;
      if ( !v25 )
        goto LABEL_336;
      v219 = 0;
      v28 = v25 - 24;
      v29 = *(_BYTE *)(v25 - 8);
      v205 = v28;
      if ( v29 > 0x17u )
      {
        if ( v29 == 78 )
        {
          v219 = v28 | 4;
        }
        else
        {
          v30 = v29 == 29;
          v31 = 0;
          if ( v30 )
            v31 = v28;
          v219 = v31;
        }
      }
      v32 = *(_QWORD *)a2 + v20;
      v33 = *(_QWORD *)(v32 + 8);
      v221 = v33 + 16LL * *(unsigned int *)(v32 + 16);
      if ( v33 != v221 )
        break;
LABEL_42:
      v44 = sub_157F280(v214);
      v49 = v45;
      v50 = v44;
      v51 = v219 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v45 != v44 )
      {
        while ( 1 )
        {
          v52 = sub_1389B50(&v224);
          v47 = 0;
          v53 = (_QWORD *)v52;
          v45 = 24LL * (*(_DWORD *)((v224 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
          for ( j = (_QWORD *)((v224 & 0xFFFFFFFFFFFFFFF8LL) - v45); v53 != j; *(_QWORD *)(v46 + 8) = v45 )
          {
            while ( 1 )
            {
              if ( v50 == *j )
              {
                v55 = 0x17FFFFFFE8LL;
                v48 = *(_BYTE *)(v50 + 23) & 0x40;
                v56 = *(_DWORD *)(v50 + 20) & 0xFFFFFFF;
                if ( v56 )
                {
                  v57 = 24LL * *(unsigned int *)(v50 + 56) + 8;
                  v58 = 0;
                  do
                  {
                    v59 = v50 - 24LL * v56;
                    if ( (_BYTE)v48 )
                      v59 = *(_QWORD *)(v50 - 8);
                    if ( v24 == *(_QWORD *)(v59 + v57) )
                    {
                      v55 = 24 * v58;
                      goto LABEL_54;
                    }
                    ++v58;
                    v57 += 8;
                  }
                  while ( v56 != (_DWORD)v58 );
                  v55 = 0x17FFFFFFE8LL;
                }
LABEL_54:
                if ( (_BYTE)v48 )
                  v60 = *(_QWORD *)(v50 - 8);
                else
                  v60 = v50 - 24LL * v56;
                v46 = *(_QWORD *)(v60 + v55);
                if ( (*(_BYTE *)(v51 + 23) & 0x40) != 0 )
                  v21 = *(_QWORD *)(v51 - 8);
                else
                  v21 = v51 - 24LL * (*(_DWORD *)(v51 + 20) & 0xFFFFFFF);
                v45 = v21 + 24LL * (unsigned int)v47;
                if ( *(_QWORD *)v45 )
                {
                  v61 = *(_QWORD *)(v45 + 8);
                  v21 = *(_QWORD *)(v45 + 16) & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v21 = v61;
                  if ( v61 )
                  {
                    v48 = *(_QWORD *)(v61 + 16) & 3LL;
                    v21 |= v48;
                    *(_QWORD *)(v61 + 16) = v21;
                  }
                }
                *(_QWORD *)v45 = v46;
                if ( v46 )
                  break;
              }
              j += 3;
              v47 = (unsigned int)(v47 + 1);
              if ( v53 == j )
                goto LABEL_65;
            }
            v62 = *(_QWORD *)(v46 + 8);
            v48 = v46 + 8;
            *(_QWORD *)(v45 + 8) = v62;
            if ( v62 )
              *(_QWORD *)(v62 + 16) = (v45 + 8) | *(_QWORD *)(v62 + 16) & 3LL;
            j += 3;
            v47 = (unsigned int)(v47 + 1);
            v21 = v48 | *(_QWORD *)(v45 + 16) & 3LL;
            *(_QWORD *)(v45 + 16) = v21;
          }
LABEL_65:
          if ( !v50 )
            BUG();
          v63 = *(_QWORD *)(v50 + 32);
          if ( !v63 )
            break;
          v50 = 0;
          if ( *(_BYTE *)(v63 - 8) == 77 )
            v50 = v63 - 24;
          if ( v49 == v50 )
            goto LABEL_70;
        }
LABEL_336:
        BUG();
      }
LABEL_70:
      if ( v218 )
      {
        v64 = *(_DWORD *)(v218 + 20) & 0xFFFFFFF;
        if ( v64 == *(_DWORD *)(v218 + 56) )
        {
          sub_15F55D0(v218, v21, v45, v46, v47, v48);
          v64 = *(_DWORD *)(v218 + 20) & 0xFFFFFFF;
        }
        v65 = (v64 + 1) & 0xFFFFFFF;
        v222 = *(_DWORD *)(v218 + 20);
        *(_DWORD *)(v218 + 20) = v65 | v222 & 0xF0000000;
        if ( v65 & 0x40000000 | v222 & 0x40000000 )
          v66 = *(_QWORD *)(v218 - 8);
        else
          v66 = v218 - 24 * v65;
        v67 = (__int64 *)(v66 + 24LL * (unsigned int)(v65 - 1));
        if ( *v67 )
        {
          v68 = v67[1];
          v69 = v67[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v69 = v68;
          if ( v68 )
            *(_QWORD *)(v68 + 16) = *(_QWORD *)(v68 + 16) & 3LL | v69;
        }
        *v67 = v205;
        v70 = *(_QWORD *)(v200 - 16);
        v67[1] = v70;
        if ( v70 )
          *(_QWORD *)(v70 + 16) = (unsigned __int64)(v67 + 1) | *(_QWORD *)(v70 + 16) & 3LL;
        v67[2] = (v200 - 16) | v67[2] & 3;
        *(_QWORD *)(v200 - 16) = v67;
        v71 = *(_DWORD *)(v218 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v218 + 23) & 0x40) != 0 )
          v72 = *(_QWORD *)(v218 - 8);
        else
          v72 = v218 - 24 * v71;
        *(_QWORD *)(v72 + 8LL * (unsigned int)(v71 - 1) + 24LL * *(unsigned int *)(v218 + 56) + 8) = v24;
      }
      if ( v198 )
      {
        v155 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v24 + 56) + 24LL) + 16LL) + 8LL);
        v156 = v201[4];
        if ( !v156 )
          BUG();
        v157 = *(_BYTE *)(v156 - 8);
        v158 = v156 - 24;
        if ( v157 == 71 )
        {
          v159 = *(_QWORD *)(v156 + 8);
          if ( !v159 )
            goto LABEL_336;
          v160 = v159 - 24;
          if ( *(_BYTE *)(v159 - 8) != 25 )
            v160 = 0;
          v161 = sub_157EBA0(v24);
          v205 = sub_1A7ED40(v158, v161, v205);
        }
        else
        {
          if ( v157 == 25 )
            v160 = v156 - 24;
          else
            v160 = 0;
          v161 = sub_157EBA0(v24);
        }
        v162 = 0;
        if ( v155 )
          v162 = v205;
        sub_1A7ED40(v160, v161, v162);
      }
      v19 = ++v215;
      v18 = v215;
      if ( *(_DWORD *)(a2 + 8) <= v215 )
        goto LABEL_85;
    }
    v199 = v24;
    while ( 1 )
    {
      v34 = *(_QWORD *)(*(_QWORD *)v33 - 48LL);
      v35 = *(_QWORD *)(*(_QWORD *)v33 - 24LL);
      if ( *(_DWORD *)(v33 + 8) != 32 )
        break;
      v36 = 0;
      v225 = v219;
      v37 = sub_1389B50(&v225);
      v38 = (_QWORD *)((v225 & 0xFFFFFFFFFFFFFFF8LL)
                     - 24LL * (*(_DWORD *)((v225 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      v211 = v35 + 8;
      if ( (_QWORD *)v37 == v38 )
      {
LABEL_25:
        v33 += 16;
        if ( v221 == v33 )
          goto LABEL_41;
      }
      else
      {
        v209 = v33;
        v39 = v35;
        do
        {
          while ( 1 )
          {
            v40 = v36++;
            if ( v34 == *v38 )
            {
              v202 = (_QWORD *)v37;
              v203 = v225 & 0xFFFFFFFFFFFFFFF8LL;
              v230 = *(__int64 **)((v225 & 0xFFFFFFFFFFFFFFF8LL) + 56);
              v21 = sub_16498A0(v225 & 0xFFFFFFFFFFFFFFF8LL);
              *(_QWORD *)(v203 + 56) = sub_1563C10((__int64 *)&v230, (__int64 *)v21, v36, 32);
              v37 = (unsigned __int64)v202;
              v41 = v225 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)((v225 & 0xFFFFFFFFFFFFFFF8LL) + 23) & 0x40) != 0 )
              {
                v26 = *(_QWORD *)(v41 - 8);
              }
              else
              {
                v27 = 24LL * (*(_DWORD *)(v41 + 20) & 0xFFFFFFF);
                v26 = v41 - v27;
              }
              v42 = (_QWORD *)(v26 + 24LL * v40);
              if ( *v42 )
              {
                v27 = v42[1];
                v26 = v42[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v26 = v27;
                if ( v27 )
                {
                  v21 = *(_QWORD *)(v27 + 16) & 3LL;
                  v26 |= v21;
                  *(_QWORD *)(v27 + 16) = v26;
                }
              }
              *v42 = v39;
              if ( v39 )
                break;
            }
            v38 += 3;
            if ( (_QWORD *)v37 == v38 )
              goto LABEL_40;
          }
          v43 = *(_QWORD *)(v39 + 8);
          v42[1] = v43;
          if ( v43 )
          {
            v21 = (__int64)(v42 + 1);
            v27 = (unsigned __int64)(v42 + 1) | *(_QWORD *)(v43 + 16) & 3LL;
            *(_QWORD *)(v43 + 16) = v27;
          }
          v38 += 3;
          v26 = v211 | v42[2] & 3LL;
          v42[2] = v26;
          *(_QWORD *)(v39 + 8) = v42;
        }
        while ( v202 != v38 );
LABEL_40:
        v33 = v209 + 16;
        if ( v221 == v209 + 16 )
        {
LABEL_41:
          v24 = v199;
          goto LABEL_42;
        }
      }
    }
    if ( *(_BYTE *)(*(_QWORD *)v35 + 8LL) == 15 && sub_1593BB0(*(_QWORD *)(*(_QWORD *)v33 - 24LL), v21, v26, v27) )
    {
      v225 = v219;
      v131 = sub_1389B50(&v225);
      v26 = 0;
      v132 = (_QWORD *)((v225 & 0xFFFFFFFFFFFFFFF8LL)
                      - 24LL * (*(_DWORD *)((v225 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      if ( (_QWORD *)v131 != v132 )
      {
        v213 = v33;
        v133 = (_QWORD *)v131;
        v134 = 0;
        do
        {
          while ( 1 )
          {
            ++v134;
            if ( v34 == *v132 )
              break;
            v132 += 3;
            if ( v133 == v132 )
              goto LABEL_185;
          }
          v132 += 3;
          v135 = v225 & 0xFFFFFFFFFFFFFFF8LL;
          v230 = *(__int64 **)((v225 & 0xFFFFFFFFFFFFFFF8LL) + 56);
          v21 = sub_16498A0(v225 & 0xFFFFFFFFFFFFFFF8LL);
          *(_QWORD *)(v135 + 56) = sub_1563AB0((__int64 *)&v230, (__int64 *)v21, v134, 32);
        }
        while ( v133 != v132 );
LABEL_185:
        v33 = v213;
      }
    }
    goto LABEL_25;
  }
LABEL_85:
  if ( !v198 )
  {
    v73 = *(_QWORD *)(v214 + 48);
    v74 = v73 - 24;
    if ( !v73 )
      v74 = 0;
    v212 = (_QWORD *)v74;
    if ( v218 )
    {
      sub_15F2120(v218, v74);
      sub_164D160((__int64)v201, v218, a4, a5, a6, a7, v75, v76, a10, a11);
    }
    v223 = v201 + 3;
    if ( v201 + 3 == (__int64 *)(v214 + 40) )
      goto LABEL_155;
LABEL_93:
    v77 = v223;
    v78 = v223 - 3;
    v79 = *v223 & 0xFFFFFFFFFFFFFFF8LL;
    v30 = *(v223 - 2) == 0;
    v223 = (_QWORD *)v79;
    if ( v30 )
      goto LABEL_154;
    if ( *((_BYTE *)v77 - 8) == 77 )
      goto LABEL_92;
    LOWORD(v232) = 257;
    v80 = *(v77 - 3);
    v81 = *(_DWORD *)(a2 + 8);
    v82 = sub_1648B60(64);
    v84 = v82;
    if ( v82 )
    {
      v216 = v82;
      sub_15F1EA0(v82, v80, 53, 0, 0, 0);
      *(_DWORD *)(v84 + 56) = v81;
      sub_164B780(v84, (__int64 *)&v230);
      v82 = (__int64)sub_1648880(v84, *(_DWORD *)(v84 + 56), 1);
    }
    else
    {
      v216 = 0;
    }
    v85 = v235;
    v210 = v79;
    v86 = v84;
    LOBYTE(v83) = v78 + 1 != 0;
    LOBYTE(v82) = v78 + 2 != 0;
    v87 = (unsigned int)v82 & v83;
    v220 = v87;
    while ( 1 )
    {
      v233 = (__int64)v78;
      v231 = 2;
      v232 = 0;
      if ( v220 )
        sub_164C220((__int64)&v231);
      v88 = *((_DWORD *)v85 + 6);
      i = v85;
      v230 = (__int64 *)&unk_49E6B50;
      if ( !v88 )
        break;
      v90 = v233;
      v136 = v85[1];
      v137 = (v88 - 1) & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
      v91 = (_QWORD *)(v136 + ((unsigned __int64)v137 << 6));
      v138 = v91[3];
      if ( v233 == v138 )
        goto LABEL_115;
      v87 = 1;
      v139 = 0;
      while ( v138 != -8 )
      {
        if ( v138 == -16 && !v139 )
          v139 = v91;
        v137 = (v88 - 1) & (v87 + v137);
        v91 = (_QWORD *)(v136 + ((unsigned __int64)v137 << 6));
        v138 = v91[3];
        if ( v233 == v138 )
          goto LABEL_115;
        v87 = (unsigned int)(v87 + 1);
      }
      v140 = *((_DWORD *)v85 + 4);
      if ( v139 )
        v91 = v139;
      ++*v85;
      v92 = v140 + 1;
      if ( 4 * (v140 + 1) >= 3 * v88 )
        goto LABEL_102;
      if ( v88 - *((_DWORD *)v85 + 5) - v92 <= v88 >> 3 )
      {
        sub_12E48B0((__int64)v85, v88);
        v141 = *((_DWORD *)v85 + 6);
        if ( v141 )
        {
          v90 = v233;
          v142 = v141 - 1;
          v143 = v85[1];
          v144 = 1;
          v145 = 0;
          LODWORD(v146) = v142 & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
          v91 = (_QWORD *)(v143 + ((unsigned __int64)(unsigned int)v146 << 6));
          v147 = v91[3];
          if ( v233 != v147 )
          {
            while ( v147 != -8 )
            {
              if ( !v145 && v147 == -16 )
                v145 = v91;
              v87 = (unsigned int)(v144 + 1);
              v146 = v142 & (unsigned int)(v146 + v144);
              v91 = (_QWORD *)(v143 + (v146 << 6));
              v147 = v91[3];
              if ( v233 == v147 )
                goto LABEL_104;
              ++v144;
            }
LABEL_309:
            if ( v145 )
              v91 = v145;
          }
LABEL_104:
          v92 = *((_DWORD *)v85 + 4) + 1;
          goto LABEL_105;
        }
LABEL_103:
        v90 = v233;
        v91 = 0;
        goto LABEL_104;
      }
LABEL_105:
      *((_DWORD *)v85 + 4) = v92;
      if ( v91[3] == -8 )
      {
        v94 = v91 + 1;
        if ( v90 != -8 )
          goto LABEL_110;
      }
      else
      {
        --*((_DWORD *)v85 + 5);
        v93 = v91[3];
        if ( v90 != v93 )
        {
          v94 = v91 + 1;
          if ( v93 != -8 && v93 != 0 && v93 != -16 )
          {
            sub_1649B30(v94);
            v90 = v233;
          }
LABEL_110:
          v91[3] = v90;
          if ( v90 != 0 && v90 != -8 && v90 != -16 )
            sub_1649AC0(v94, v231 & 0xFFFFFFFFFFFFFFF8LL);
          v90 = v233;
        }
      }
      v95 = i;
      v91[5] = 6;
      v91[6] = 0;
      v91[4] = v95;
      v91[7] = 0;
LABEL_115:
      v230 = (__int64 *)&unk_49EE2B0;
      if ( v90 != 0 && v90 != -8 && v90 != -16 )
        sub_1649B30(&v231);
      v96 = *(_QWORD *)(v91[7] + 40LL);
      v233 = (__int64)v78;
      v231 = 2;
      v232 = 0;
      if ( v220 )
        sub_164C220((__int64)&v231);
      v97 = *((_DWORD *)v85 + 6);
      i = v85;
      v230 = (__int64 *)&unk_49E6B50;
      if ( !v97 )
      {
        ++*v85;
        goto LABEL_122;
      }
      v101 = v233;
      v99 = v85[1];
      v148 = (v97 - 1) & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
      v102 = (_QWORD *)(v99 + ((unsigned __int64)v148 << 6));
      v98 = v102[3];
      if ( v233 != v98 )
      {
        v149 = 1;
        v87 = 0;
        while ( v98 != -8 )
        {
          if ( !v87 && v98 == -16 )
            v87 = (__int64)v102;
          v148 = (v97 - 1) & (v149 + v148);
          v102 = (_QWORD *)(v99 + ((unsigned __int64)v148 << 6));
          v98 = v102[3];
          if ( v233 == v98 )
            goto LABEL_134;
          ++v149;
        }
        v150 = *((_DWORD *)v85 + 4);
        if ( v87 )
          v102 = (_QWORD *)v87;
        ++*v85;
        v103 = v150 + 1;
        if ( 4 * v103 >= 3 * v97 )
        {
LABEL_122:
          sub_12E48B0((__int64)v85, 2 * v97);
          v100 = *((_DWORD *)v85 + 6);
          if ( !v100 )
            goto LABEL_123;
          v101 = v233;
          v87 = (unsigned int)(v100 - 1);
          v190 = v85[1];
          v153 = 0;
          v98 = 1;
          v191 = v87 & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
          v102 = (_QWORD *)(v190 + ((unsigned __int64)v191 << 6));
          v99 = v102[3];
          if ( v233 != v99 )
          {
            while ( v99 != -8 )
            {
              if ( v99 == -16 && !v153 )
                v153 = v102;
              v191 = v87 & (v98 + v191);
              v102 = (_QWORD *)(v190 + ((unsigned __int64)v191 << 6));
              v99 = v102[3];
              if ( v233 == v99 )
                goto LABEL_124;
              v98 = (unsigned int)(v98 + 1);
            }
            goto LABEL_217;
          }
LABEL_124:
          v103 = *((_DWORD *)v85 + 4) + 1;
        }
        else
        {
          v98 = v97 - *((_DWORD *)v85 + 5) - v103;
          v99 = v97 >> 3;
          if ( (unsigned int)v98 <= (unsigned int)v99 )
          {
            sub_12E48B0((__int64)v85, v97);
            v151 = *((_DWORD *)v85 + 6);
            if ( v151 )
            {
              v101 = v233;
              v87 = (unsigned int)(v151 - 1);
              v152 = v85[1];
              v153 = 0;
              v98 = 1;
              v154 = v87 & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
              v102 = (_QWORD *)(v152 + ((unsigned __int64)v154 << 6));
              v99 = v102[3];
              if ( v99 != v233 )
              {
                while ( v99 != -8 )
                {
                  if ( !v153 && v99 == -16 )
                    v153 = v102;
                  v154 = v87 & (v98 + v154);
                  v102 = (_QWORD *)(v152 + ((unsigned __int64)v154 << 6));
                  v99 = v102[3];
                  if ( v233 == v99 )
                    goto LABEL_124;
                  v98 = (unsigned int)(v98 + 1);
                }
LABEL_217:
                if ( v153 )
                  v102 = v153;
              }
              goto LABEL_124;
            }
LABEL_123:
            v101 = v233;
            v102 = 0;
            goto LABEL_124;
          }
        }
        *((_DWORD *)v85 + 4) = v103;
        if ( v102[3] == -8 )
        {
          v105 = v102 + 1;
          if ( v101 != -8 )
            goto LABEL_130;
        }
        else
        {
          --*((_DWORD *)v85 + 5);
          v104 = v102[3];
          if ( v101 != v104 )
          {
            v105 = v102 + 1;
            LOBYTE(v99) = v104 != -8;
            if ( ((v104 != 0) & (unsigned __int8)v99) != 0 && v104 != -16 )
            {
              v204 = v102;
              v206 = v102 + 1;
              sub_1649B30(v105);
              v101 = v233;
              v102 = v204;
              v105 = v206;
            }
LABEL_130:
            v102[3] = v101;
            LOBYTE(v98) = v101 != -8;
            if ( ((v101 != 0) & (unsigned __int8)v98) == 0 || v101 == -16 )
            {
              v101 = v233;
            }
            else
            {
              v207 = v102;
              sub_1649AC0(v105, v231 & 0xFFFFFFFFFFFFFFF8LL);
              v101 = v233;
              v102 = v207;
            }
          }
        }
        v106 = i;
        v102[5] = 6;
        v102[6] = 0;
        v102[4] = v106;
        v102[7] = 0;
      }
LABEL_134:
      v230 = (__int64 *)&unk_49EE2B0;
      if ( v101 != -8 && v101 != 0 && v101 != -16 )
      {
        v208 = v102;
        sub_1649B30(&v231);
        v102 = v208;
      }
      v107 = v102[7];
      v108 = *(_DWORD *)(v86 + 20) & 0xFFFFFFF;
      if ( v108 == *(_DWORD *)(v86 + 56) )
      {
        sub_15F55D0(v86, v107, v87, v101, v98, v99);
        v108 = *(_DWORD *)(v86 + 20) & 0xFFFFFFF;
      }
      v109 = (v108 + 1) & 0xFFFFFFF;
      v110 = v109 | *(_DWORD *)(v86 + 20) & 0xF0000000;
      *(_DWORD *)(v86 + 20) = v110;
      if ( (v110 & 0x40000000) != 0 )
        v111 = *(_QWORD *)(v86 - 8);
      else
        v111 = v216 - 24 * v109;
      v112 = (__int64 *)(v111 + 24LL * (unsigned int)(v109 - 1));
      if ( *v112 )
      {
        v113 = v112[1];
        v114 = v112[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v114 = v113;
        if ( v113 )
          *(_QWORD *)(v113 + 16) = *(_QWORD *)(v113 + 16) & 3LL | v114;
      }
      *v112 = v107;
      if ( v107 )
      {
        v115 = *(_QWORD *)(v107 + 8);
        v112[1] = v115;
        if ( v115 )
          *(_QWORD *)(v115 + 16) = (unsigned __int64)(v112 + 1) | *(_QWORD *)(v115 + 16) & 3LL;
        v112[2] = (v107 + 8) | v112[2] & 3;
        *(_QWORD *)(v107 + 8) = v112;
      }
      v116 = *(_DWORD *)(v86 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v86 + 23) & 0x40) != 0 )
        v117 = *(_QWORD *)(v86 - 8);
      else
        v117 = v216 - 24 * v116;
      v85 += 10;
      *(_QWORD *)(v117 + 8LL * (unsigned int)(v116 - 1) + 24LL * *(unsigned int *)(v86 + 56) + 8) = v96;
      if ( v85 == (__int64 *)v239 )
      {
        v118 = v86;
        v79 = v210;
        v119 = *(_QWORD *)(v214 + 48);
        if ( v119 )
          v119 -= 24;
        sub_15F2120(v216, v119);
        sub_164D160((__int64)v78, v118, a4, a5, a6, a7, v120, v121, a10, a11);
LABEL_154:
        sub_15F20C0(v78);
        if ( v212 == v78 )
        {
LABEL_155:
          v122 = (__int64 *)v239;
          do
          {
            v122 -= 10;
            if ( *((_BYTE *)v122 + 64) )
            {
              v163 = *((_DWORD *)v122 + 14);
              if ( v163 )
              {
                v164 = (_QWORD *)v122[5];
                v165 = &v164[2 * v163];
                do
                {
                  if ( *v164 != -8 && *v164 != -4 )
                  {
                    v166 = v164[1];
                    if ( v166 )
                      sub_161E7C0((__int64)(v164 + 1), v166);
                  }
                  v164 += 2;
                }
                while ( v165 != v164 );
              }
              j___libc_free_0(v122[5]);
            }
            v123 = *((_DWORD *)v122 + 6);
            if ( v123 )
            {
              v124 = (_QWORD *)v122[1];
              v226 = 2;
              v125 = (unsigned __int64)v123 << 6;
              v227 = 0;
              v126 = -8;
              v228 = -8;
              v127 = (_QWORD *)((char *)v124 + v125);
              v225 = (__int64)&unk_49E6B50;
              v229 = 0;
              v231 = 2;
              v232 = 0;
              v233 = -16;
              v230 = (__int64 *)&unk_49E6B50;
              i = 0;
              while ( 1 )
              {
                v128 = v124[3];
                if ( v126 != v128 )
                {
                  v126 = v233;
                  if ( v128 != v233 )
                  {
                    v129 = v124[7];
                    if ( v129 != -8 && v129 != 0 && v129 != -16 )
                    {
                      sub_1649B30(v124 + 5);
                      v128 = v124[3];
                    }
                    v126 = v128;
                  }
                }
                *v124 = &unk_49EE2B0;
                if ( v126 != -8 && v126 != 0 && v126 != -16 )
                  sub_1649B30(v124 + 1);
                v124 += 8;
                if ( v127 == v124 )
                  break;
                v126 = v228;
              }
              v230 = (__int64 *)&unk_49EE2B0;
              if ( v233 != 0 && v233 != -8 && v233 != -16 )
                sub_1649B30(&v231);
              v225 = (__int64)&unk_49EE2B0;
              if ( v228 != 0 && v228 != -8 && v228 != -16 )
                sub_1649B30(&v226);
            }
            result = j___libc_free_0(v122[1]);
          }
          while ( v122 != v235 );
          return result;
        }
LABEL_92:
        if ( v79 == v214 + 40 )
          goto LABEL_155;
        goto LABEL_93;
      }
    }
    ++*v85;
LABEL_102:
    sub_12E48B0((__int64)v85, 2 * v88);
    v89 = *((_DWORD *)v85 + 6);
    if ( v89 )
    {
      v90 = v233;
      v192 = v89 - 1;
      v193 = v85[1];
      v194 = 1;
      v145 = 0;
      LODWORD(v195) = v192 & (((unsigned int)v233 >> 9) ^ ((unsigned int)v233 >> 4));
      v91 = (_QWORD *)(v193 + ((unsigned __int64)(unsigned int)v195 << 6));
      v196 = v91[3];
      if ( v233 != v196 )
      {
        while ( v196 != -8 )
        {
          if ( !v145 && v196 == -16 )
            v145 = v91;
          v87 = (unsigned int)(v194 + 1);
          v195 = v192 & (unsigned int)(v195 + v194);
          v91 = (_QWORD *)(v193 + (v195 << 6));
          v196 = v91[3];
          if ( v233 == v196 )
            goto LABEL_104;
          ++v194;
        }
        goto LABEL_309;
      }
      goto LABEL_104;
    }
    goto LABEL_103;
  }
  v167 = *(_QWORD *)(v214 + 8);
  if ( v167 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v167) + 16) - 25) > 9u )
    {
      v167 = *(_QWORD *)(v167 + 8);
      if ( !v167 )
        goto LABEL_299;
    }
    v170 = v167;
    v171 = 0;
    v230 = &v232;
    v231 = 0x200000000LL;
    while ( 1 )
    {
      v170 = *(_QWORD *)(v170 + 8);
      if ( !v170 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v170) + 16) - 25) <= 9u )
      {
        v170 = *(_QWORD *)(v170 + 8);
        ++v171;
        if ( !v170 )
          goto LABEL_256;
      }
    }
LABEL_256:
    v172 = v171 + 1;
    v173 = &v232;
    if ( v172 > 2 )
    {
      sub_16CD150((__int64)&v230, &v232, v172, 8, v168, v169);
      v173 = &v230[(unsigned int)v231];
    }
    v174 = sub_1648700(v167);
LABEL_261:
    if ( v173 )
      *v173 = v174[5];
    while ( 1 )
    {
      v167 = *(_QWORD *)(v167 + 8);
      if ( !v167 )
        break;
      v174 = sub_1648700(v167);
      if ( (unsigned __int8)(*((_BYTE *)v174 + 16) - 25) <= 9u )
      {
        ++v173;
        goto LABEL_261;
      }
    }
    v175 = 0;
    v176 = 0;
    for ( LODWORD(v231) = v231 + v172; (unsigned int)v176 < (unsigned int)v231; v175 = v176 )
    {
      v177 = (_QWORD *)sub_157EBA0(v230[v176]);
      sub_15F20C0(v177);
      v176 = (unsigned int)(v175 + 1);
    }
  }
  else
  {
LABEL_299:
    v230 = &v232;
    v231 = 0x200000000LL;
  }
  sub_157F980(v214);
  if ( v230 != &v232 )
    _libc_free((unsigned __int64)v230);
  v178 = (__int64 *)v238;
  if ( v238[64] )
    goto LABEL_290;
  while ( 1 )
  {
    v179 = *((_DWORD *)v178 + 6);
    if ( v179 )
    {
      v180 = (_QWORD *)v178[1];
      v226 = 2;
      v181 = (unsigned __int64)v179 << 6;
      v225 = (__int64)&unk_49E6B50;
      v182 = -8;
      v183 = (_QWORD *)((char *)v180 + v181);
      v227 = 0;
      v228 = -8;
      v229 = 0;
      v231 = 2;
      v232 = 0;
      v233 = -16;
      v230 = (__int64 *)&unk_49E6B50;
      i = 0;
      while ( 1 )
      {
        v184 = v180[3];
        if ( v184 != v182 )
        {
          v182 = v233;
          if ( v184 != v233 )
          {
            v185 = v180[7];
            if ( v185 != 0 && v185 != -8 && v185 != -16 )
            {
              sub_1649B30(v180 + 5);
              v184 = v180[3];
            }
            v182 = v184;
          }
        }
        *v180 = &unk_49EE2B0;
        if ( v182 != -8 && v182 != 0 && v182 != -16 )
          sub_1649B30(v180 + 1);
        v180 += 8;
        if ( v183 == v180 )
          break;
        v182 = v228;
      }
      v230 = (__int64 *)&unk_49EE2B0;
      if ( v233 != -8 && v233 != 0 && v233 != -16 )
        sub_1649B30(&v231);
      v225 = (__int64)&unk_49EE2B0;
      if ( v228 != -8 && v228 != 0 && v228 != -16 )
        sub_1649B30(&v226);
    }
    result = j___libc_free_0(v178[1]);
    if ( v178 == v235 )
      break;
    v178 -= 10;
    if ( *((_BYTE *)v178 + 64) )
    {
LABEL_290:
      v186 = *((_DWORD *)v178 + 14);
      if ( v186 )
      {
        v187 = (_QWORD *)v178[5];
        v188 = &v187[2 * v186];
        do
        {
          if ( *v187 != -4 && *v187 != -8 )
          {
            v189 = v187[1];
            if ( v189 )
              sub_161E7C0((__int64)(v187 + 1), v189);
          }
          v187 += 2;
        }
        while ( v188 != v187 );
      }
      j___libc_free_0(v178[5]);
    }
  }
  return result;
}
