// Function: sub_EBF510
// Address: 0xebf510
//
__int64 __fastcall sub_EBF510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  int i; // eax
  __int64 v5; // rax
  unsigned int v6; // edx
  int v7; // eax
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rbx
  _BYTE *v11; // r12
  size_t v12; // r13
  int v13; // eax
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r8
  void (*v18)(void); // rax
  unsigned int v19; // r13d
  __int64 (*v21)(); // rax
  int v22; // eax
  char v23; // al
  __m128i v24; // xmm3
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 (*v27)(); // rax
  void *v28; // r15
  _BYTE *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r9
  void *v32; // rdx
  void *v33; // r8
  void *v34; // rbx
  __int64 v35; // rdi
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // r15
  bool v41; // zf
  __int64 v42; // rbx
  __int64 v43; // r12
  __int64 v44; // r13
  size_t v45; // r12
  const void *v46; // r14
  bool v47; // al
  _BYTE *v48; // rdi
  void (*v49)(); // rax
  __int64 v50; // rdi
  void (*v51)(); // rax
  __int64 v52; // rax
  void *v53; // rdx
  void *v54; // rax
  char *v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *v61; // r15
  __int64 v62; // rax
  void (*v63)(); // rdx
  __int64 (__fastcall *v64)(__int64 *, _BYTE *); // rbx
  __m128i v65; // xmm2
  int v66; // eax
  unsigned int v67; // r8d
  int v68; // ebx
  void *v69; // r15
  const void *v70; // rbx
  int v71; // eax
  int v72; // eax
  __int64 v73; // rcx
  __int64 v74; // r9
  __int64 v75; // rdx
  __int64 *v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdi
  __int64 (__fastcall *v79)(__int64, void *, void *, __int64, _QWORD); // rax
  __int64 (*v80)(); // rax
  const void *v81; // r15
  __int64 v82; // rbx
  int v83; // eax
  int v84; // eax
  __int64 v85; // r9
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 **v88; // rax
  __int64 *v89; // rbx
  __int64 v90; // r13
  __int64 v91; // rsi
  __int64 v92; // rcx
  unsigned __int64 v93; // rax
  __int64 v94; // r8
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 *v97; // r15
  __int64 *v98; // r14
  __int64 v99; // rbx
  __int64 v100; // r12
  __int64 v101; // rdi
  char v102; // al
  __int64 v103; // rbx
  __int64 v104; // rax
  __int64 v105; // r15
  unsigned int v106; // eax
  __int64 v107; // r8
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // rdi
  __int64 v111; // rax
  int v112; // edx
  __int64 v113; // rax
  __int64 v114; // r14
  unsigned __int64 v115; // rdx
  unsigned __int64 v116; // r15
  unsigned __int64 v117; // rsi
  int v118; // ecx
  unsigned __int64 v119; // rax
  __int64 v120; // r12
  __m128i *v121; // rsi
  __int64 v122; // rdx
  unsigned int v123; // eax
  __int64 v124; // r8
  __int64 v125; // r9
  _QWORD *v126; // rax
  __int64 v127; // rbx
  __int64 v128; // rdx
  __int64 v129; // rax
  __int64 v130; // rax
  _BYTE *v131; // rsi
  __int64 v132; // rax
  __int64 *v133; // rbx
  _QWORD *v134; // rax
  _QWORD *v135; // rax
  unsigned __int64 v136; // rbx
  _QWORD *v137; // rdx
  __int64 v138; // rax
  __int64 v139; // rbx
  __int64 v140; // rax
  __int64 v141; // r8
  __int64 v142; // r9
  _DWORD *v143; // rcx
  __int64 v144; // rax
  __int64 v145; // rdx
  unsigned int v146; // r14d
  __int64 v147; // r13
  _DWORD *v148; // rbx
  unsigned __int64 v149; // r12
  __m128i v150; // xmm1
  bool v151; // cc
  __int64 v152; // rdi
  __int64 v153; // rax
  __int64 v154; // rax
  _DWORD *v155; // rax
  __int64 v156; // rdi
  __int64 v157; // rcx
  __int64 v158; // r8
  __int64 v159; // r9
  unsigned __int64 v160; // rsi
  __int64 v161; // rax
  __int64 v162; // r15
  unsigned int v163; // eax
  __int64 v164; // r9
  unsigned __int64 v165; // rax
  __int64 v166; // rbx
  const void *v167; // rcx
  unsigned __int64 v168; // rdx
  unsigned __int64 v169; // r14
  unsigned __int64 v170; // rsi
  unsigned __int64 v171; // rax
  unsigned __int64 v172; // rdx
  unsigned __int64 v173; // rax
  __int64 v174; // rcx
  _DWORD *v175; // rsi
  _DWORD *v176; // rdi
  __int64 v177; // rsi
  __int64 v178; // rsi
  _DWORD *v179; // rdx
  unsigned __int64 v180; // rax
  _DWORD *v181; // rbx
  unsigned __int64 v182; // r15
  __m128i v183; // xmm5
  __int64 v184; // rdi
  __int64 v185; // rax
  __int64 v186; // rax
  _DWORD *v187; // rax
  __int64 v188; // rdi
  __int64 v189; // rcx
  __int64 v190; // r8
  __int64 v191; // r9
  const void *v192; // rsi
  unsigned __int64 v193; // rdx
  unsigned __int64 v194; // rax
  __int64 v195; // rcx
  _DWORD *v196; // rsi
  _DWORD *v197; // rdi
  const void *v198; // rsi
  void (*v199)(void); // rax
  __int64 v200; // r13
  __int64 v201; // rbx
  const void *v202; // r12
  void *v203; // rbx
  bool v204; // al
  const void *v205; // r12
  void *v206; // rbx
  bool v207; // dl
  __int64 v208; // rdi
  void (*v209)(); // rax
  void *v210; // rax
  void *v211; // rax
  void *v212; // rax
  void *v213; // rax
  __int64 v214; // [rsp-10h] [rbp-2D0h]
  unsigned int v216; // [rsp+8h] [rbp-2B8h]
  __int64 v217; // [rsp+8h] [rbp-2B8h]
  __int64 v218; // [rsp+8h] [rbp-2B8h]
  __int64 v219; // [rsp+10h] [rbp-2B0h]
  __int64 v220; // [rsp+10h] [rbp-2B0h]
  void *v221; // [rsp+10h] [rbp-2B0h]
  const void *v222; // [rsp+18h] [rbp-2A8h]
  unsigned __int8 v223; // [rsp+18h] [rbp-2A8h]
  int v224; // [rsp+18h] [rbp-2A8h]
  unsigned int v225; // [rsp+18h] [rbp-2A8h]
  unsigned int v226; // [rsp+18h] [rbp-2A8h]
  unsigned int v227; // [rsp+18h] [rbp-2A8h]
  void *v228; // [rsp+18h] [rbp-2A8h]
  __int64 v229; // [rsp+18h] [rbp-2A8h]
  unsigned int v230; // [rsp+18h] [rbp-2A8h]
  void *v231; // [rsp+18h] [rbp-2A8h]
  int v232; // [rsp+18h] [rbp-2A8h]
  int v233; // [rsp+18h] [rbp-2A8h]
  int v234; // [rsp+18h] [rbp-2A8h]
  __int64 v235; // [rsp+18h] [rbp-2A8h]
  __int64 v236; // [rsp+18h] [rbp-2A8h]
  __int64 v237; // [rsp+18h] [rbp-2A8h]
  _BYTE *v238; // [rsp+18h] [rbp-2A8h]
  __int64 v239; // [rsp+28h] [rbp-298h] BYREF
  __int64 v240; // [rsp+30h] [rbp-290h] BYREF
  __int64 v241; // [rsp+38h] [rbp-288h] BYREF
  void *s1[2]; // [rsp+40h] [rbp-280h] BYREF
  __int64 *v243; // [rsp+50h] [rbp-270h] BYREF
  __int64 *v244; // [rsp+58h] [rbp-268h]
  __int64 v245; // [rsp+60h] [rbp-260h]
  int v246; // [rsp+70h] [rbp-250h] BYREF
  __m128i v247; // [rsp+78h] [rbp-248h] BYREF
  const void *v248; // [rsp+88h] [rbp-238h] BYREF
  unsigned int v249; // [rsp+90h] [rbp-230h]
  __m128i v250; // [rsp+A0h] [rbp-220h] BYREF
  _QWORD v251[2]; // [rsp+B0h] [rbp-210h] BYREF
  __int16 v252; // [rsp+C0h] [rbp-200h]
  __m128i v253; // [rsp+D0h] [rbp-1F0h] BYREF
  __int64 v254; // [rsp+E0h] [rbp-1E0h]
  __int64 v255; // [rsp+E8h] [rbp-1D8h]
  __int64 v256; // [rsp+F0h] [rbp-1D0h]
  __int64 v257; // [rsp+F8h] [rbp-1C8h]
  _BYTE *v258; // [rsp+100h] [rbp-1C0h]
  _BYTE s2[24]; // [rsp+110h] [rbp-1B0h] BYREF
  const void *v260; // [rsp+128h] [rbp-198h] BYREF
  _BYTE *v261; // [rsp+130h] [rbp-190h]
  _BYTE *v262; // [rsp+138h] [rbp-188h]
  unsigned __int64 v263; // [rsp+140h] [rbp-180h]
  __int64 v264; // [rsp+148h] [rbp-178h]
  __int64 v265; // [rsp+150h] [rbp-170h] BYREF
  __int64 v266; // [rsp+158h] [rbp-168h]
  __int64 *v267; // [rsp+160h] [rbp-160h] BYREF
  __int64 v268; // [rsp+168h] [rbp-158h]
  __int64 v269; // [rsp+170h] [rbp-150h] BYREF
  __int64 v270; // [rsp+178h] [rbp-148h]
  __int64 (__fastcall **v271)(); // [rsp+180h] [rbp-140h] BYREF
  char v272; // [rsp+188h] [rbp-138h]
  __int64 v273; // [rsp+258h] [rbp-68h]
  __int16 v274; // [rsp+260h] [rbp-60h]
  __int64 v275; // [rsp+268h] [rbp-58h]
  __int64 v276; // [rsp+270h] [rbp-50h]
  __int64 v277; // [rsp+278h] [rbp-48h]
  __int64 v278; // [rsp+280h] [rbp-40h]

  v3 = a1;
  for ( i = **(_DWORD **)(a1 + 48); i == 11; i = **(_DWORD **)(a1 + 48) )
    sub_EABFE0(a1);
  if ( i != 9 )
  {
    v5 = sub_ECD7B0(a1);
    v246 = *(_DWORD *)v5;
    v6 = *(_DWORD *)(v5 + 32);
    v247 = _mm_loadu_si128((const __m128i *)(v5 + 8));
    v249 = v6;
    if ( v6 > 0x40 )
      sub_C43780((__int64)&v248, (const void **)(v5 + 24));
    else
      v248 = *(const void **)(v5 + 24);
    s1[0] = 0;
    v239 = sub_ECD6A0(&v246);
    s1[1] = 0;
    *(_QWORD *)(a1 + 280) = sub_ECD6A0(&v246);
    v7 = **(_DWORD **)(a1 + 48);
    switch ( v7 )
    {
      case 8:
        v19 = sub_EB0E30(a1, v239, *(_QWORD *)(a1 + 368) == *(_QWORD *)(a1 + 376));
        goto LABEL_62;
      case 4:
        v25 = sub_ECD7B0(a1);
        if ( *(_DWORD *)(v25 + 32) <= 0x40u )
          v10 = *(_QWORD *)(v25 + 24);
        else
          v10 = **(_QWORD **)(v25 + 24);
        if ( v10 >= 0 )
        {
          v52 = sub_ECD7B0(a1);
          v53 = *(void **)(v52 + 16);
          v54 = *(void **)(v52 + 8);
          s1[1] = v53;
          s1[0] = v54;
          sub_EABFE0(a1);
          if ( **(_DWORD **)(a1 + 48) == 10 || *(_BYTE *)(a1 + 313) )
            goto LABEL_18;
        }
        else if ( *(_BYTE *)(a1 + 313) )
        {
          s1[1] = 0;
          s1[0] = (void *)byte_3F871B3;
          goto LABEL_18;
        }
        sub_EABFE0(a1);
        *(_QWORD *)s2 = "unexpected token at start of statement";
        LOWORD(v261) = 259;
        v19 = sub_ECDA70(a1, v239, s2, 0, 0);
        goto LABEL_62;
      case 25:
        v10 = -1;
        sub_EABFE0(a1);
        s1[1] = (void *)1;
        s1[0] = ".";
        goto LABEL_18;
      case 21:
        v10 = -1;
        sub_EABFE0(a1);
        s1[1] = (void *)1;
        s1[0] = "{";
        goto LABEL_18;
      case 22:
        v10 = -1;
        sub_EABFE0(a1);
        s1[1] = (void *)1;
        s1[0] = "}";
        goto LABEL_18;
      case 24:
        v21 = *(__int64 (**)())(**(_QWORD **)(a1 + 8) + 152LL);
        if ( v21 == sub_EA21D0 )
          goto LABEL_14;
        if ( (unsigned __int8)v21() )
        {
          v10 = -1;
          sub_EABFE0(a1);
          s1[1] = (void *)1;
          s1[0] = "*";
          goto LABEL_18;
        }
        v7 = **(_DWORD **)(a1 + 48);
        break;
    }
    if ( v7 == 46 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = *(__int64 (**)())(*(_QWORD *)v8 + 160LL);
      if ( v9 != sub_EA21E0 )
      {
        if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v9)(v8, 46) )
        {
          v10 = -1;
          sub_EABFE0(v3);
          s1[1] = (void *)1;
          s1[0] = "@";
          goto LABEL_18;
        }
      }
    }
LABEL_14:
    if ( (unsigned __int8)sub_EB61F0(v3, (__int64 *)s1) )
    {
      if ( !*(_BYTE *)(v3 + 313) )
      {
        sub_EABFE0(v3);
        *(_QWORD *)s2 = "unexpected token at start of statement";
        LOWORD(v261) = 259;
        v19 = sub_ECDA70(v3, v239, s2, 0, 0);
        goto LABEL_62;
      }
      s1[1] = 0;
      s1[0] = (void *)byte_3F871B3;
    }
    v10 = -1;
LABEL_18:
    v11 = s2;
    sub_C93130((__int64 *)s2, (__int64)s1);
    v12 = *(_QWORD *)&s2[8];
    v222 = *(const void **)s2;
    v13 = sub_C92610();
    v14 = sub_C92860((__int64 *)(v3 + 872), v222, v12, v13);
    v15 = *(_QWORD *)(v3 + 872);
    if ( v14 == -1 )
      v16 = v15 + 8LL * *(unsigned int *)(v3 + 880);
    else
      v16 = v15 + 8LL * v14;
    if ( *(_BYTE **)s2 != &s2[16] )
    {
      j_j___libc_free_0(*(_QWORD *)s2, *(_QWORD *)&s2[16] + 1LL);
      v15 = *(_QWORD *)(v3 + 872);
    }
    if ( v16 != v15 + 8LL * *(unsigned int *)(v3 + 880) )
    {
      v17 = *(unsigned int *)(*(_QWORD *)v16 + 8LL);
      switch ( (int)v17 )
      {
        case 'R':
        case 'S':
        case 'T':
        case 'U':
        case 'V':
        case 'W':
        case 'X':
          v55 = *(char **)(v3 + 328);
          if ( v55 == *(char **)(v3 + 336) )
          {
            v233 = *(_DWORD *)(*(_QWORD *)v16 + 8LL);
            sub_EA9230((char **)(v3 + 320), v55, (_QWORD *)(v3 + 308));
            LODWORD(v17) = v233;
          }
          else
          {
            if ( v55 )
            {
              *(_QWORD *)v55 = *(_QWORD *)(v3 + 308);
              v55 = *(char **)(v3 + 328);
            }
            *(_QWORD *)(v3 + 328) = v55 + 8;
          }
          v41 = *(_BYTE *)(v3 + 313) == 0;
          *(_DWORD *)(v3 + 308) = 1;
          if ( !v41 )
            goto LABEL_72;
          v224 = v17;
          if ( (unsigned __int8)sub_EAC8B0(v3, s2) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
          {
            v19 = 1;
          }
          else
          {
            v56 = *(_QWORD *)s2;
            switch ( v224 )
            {
              case 'S':
                v56 = *(_QWORD *)s2 == 0;
                *(_QWORD *)s2 = v56;
                break;
              case 'T':
                v56 = *(_QWORD *)s2 >= 0LL;
                *(_QWORD *)s2 = v56;
                break;
              case 'U':
                v56 = *(_QWORD *)s2 > 0LL;
                *(_QWORD *)s2 = v56;
                break;
              case 'V':
                v56 = *(_QWORD *)s2 <= 0LL;
                *(_QWORD *)s2 = v56;
                break;
              case 'W':
                v56 = *(_QWORD *)s2 >> 63;
                *(_QWORD *)s2 >>= 63;
                break;
              default:
                break;
            }
            *(_BYTE *)(v3 + 312) = v56 != 0;
            *(_BYTE *)(v3 + 313) = v56 == 0;
          }
          break;
        case 'Y':
          v19 = sub_EB5D60(v3, 1);
          break;
        case 'Z':
          v19 = sub_EB5D60(v3, 0);
          break;
        case '[':
          v19 = sub_EB5E30(v3, 1);
          break;
        case '\\':
          v19 = sub_EB4010(v3, 1);
          break;
        case ']':
          v19 = sub_EB5E30(v3, 0);
          break;
        case '^':
          v19 = sub_EB4010(v3, 0);
          break;
        case '_':
          v19 = sub_EBB1F0(v3, 1);
          break;
        case '`':
        case 'a':
          v19 = sub_EBB1F0(v3, 0);
          break;
        case 'b':
          if ( (unsigned int)(*(_DWORD *)(v3 + 308) - 1) > 1 )
          {
            *(_QWORD *)s2 = "Encountered a .elseif that doesn't follow an .if or  an .elseif";
            LOWORD(v261) = 259;
            v19 = sub_ECDA70(v3, v239, s2, 0, 0);
          }
          else
          {
            *(_DWORD *)(v3 + 308) = 2;
            v60 = *(_QWORD *)(v3 + 328);
            if ( v60 != *(_QWORD *)(v3 + 320) && *(_BYTE *)(v60 - 3) || *(_BYTE *)(v3 + 312) )
            {
              *(_BYTE *)(v3 + 313) = 1;
              v19 = 0;
              sub_EB4E00(v3);
            }
            else if ( (unsigned __int8)sub_EAC8B0(v3, s2) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
            {
              v19 = 1;
            }
            else
            {
              v41 = *(_QWORD *)s2 == 0;
              *(_BYTE *)(v3 + 312) = *(_QWORD *)s2 != 0;
              *(_BYTE *)(v3 + 313) = v41;
            }
          }
          break;
        case 'c':
          v19 = sub_ECE000(v3);
          if ( !(_BYTE)v19 )
          {
            if ( (unsigned int)(*(_DWORD *)(v3 + 308) - 1) > 1 )
            {
              *(_QWORD *)s2 = "Encountered a .else that doesn't follow  an .if or an .elseif";
              LOWORD(v261) = 259;
              v19 = sub_ECDA70(v3, v239, s2, 0, 0);
            }
            else
            {
              *(_DWORD *)(v3 + 308) = 3;
              v59 = *(_QWORD *)(v3 + 328);
              if ( v59 != *(_QWORD *)(v3 + 320) && *(_BYTE *)(v59 - 3) || *(_BYTE *)(v3 + 312) )
              {
                *(_BYTE *)(v3 + 313) = 1;
              }
              else
              {
                *(_BYTE *)(v3 + 313) = 0;
                v19 = 0;
              }
            }
          }
          break;
        case 'd':
          v19 = sub_ECE000(v3);
          if ( !(_BYTE)v19 )
          {
            if ( !*(_DWORD *)(v3 + 308) || (v57 = *(_QWORD *)(v3 + 328), v57 == *(_QWORD *)(v3 + 320)) )
            {
              *(_QWORD *)s2 = "Encountered a .endif that doesn't follow an .if or .else";
              LOWORD(v261) = 259;
              v19 = sub_ECDA70(v3, v239, s2, 0, 0);
            }
            else
            {
              v58 = *(_QWORD *)(v57 - 8);
              *(_QWORD *)(v3 + 328) = v57 - 8;
              *(_QWORD *)(v3 + 308) = v58;
            }
          }
          break;
        default:
          goto LABEL_38;
      }
      goto LABEL_62;
    }
    v17 = 0;
LABEL_38:
    v19 = *(unsigned __int8 *)(v3 + 313);
    if ( (_BYTE)v19 )
    {
LABEL_72:
      v19 = 0;
      sub_EB4E00(v3);
      goto LABEL_62;
    }
    v22 = **(_DWORD **)(v3 + 48);
    if ( v22 == 10 )
    {
      v26 = *(_QWORD *)(v3 + 8);
      v27 = *(__int64 (**)())(*(_QWORD *)v26 + 144LL);
      if ( v27 == sub_EA21C0 || (v225 = v17, ((unsigned __int8 (__fastcall *)(__int64, int *))v27)(v26, &v246)) )
      {
        if ( *(_BYTE *)(v3 + 869) || !(unsigned __int8)sub_EA2540(v3) )
        {
          sub_EABFE0(v3);
          v28 = s1[1];
          v29 = s1[0];
          if ( s1[1] == (void *)1 && *(_BYTE *)s1[0] == 46 )
          {
            *(_QWORD *)s2 = "invalid use of pseudo-symbol '.' as a label";
            LOWORD(v261) = 259;
            v19 = sub_ECDA70(v3, v239, s2, 0, 0);
          }
          else
          {
            if ( v10 == -1 )
            {
              if ( *(_BYTE *)(v3 + 869) && a3 )
              {
                v30 = (*(__int64 (__fastcall **)(__int64, void *, void *, _QWORD, __int64, __int64))(*(_QWORD *)a3 + 24LL))(
                        a3,
                        s1[0],
                        s1[1],
                        *(_QWORD *)(v3 + 248),
                        v239,
                        1);
                v33 = v32;
                v34 = (void *)v30;
                v29 = (_BYTE *)v30;
                v28 = v33;
                v35 = *(_QWORD *)(a2 + 88);
                v36 = *(unsigned int *)(v35 + 8);
                if ( v36 >= *(unsigned int *)(v35 + 12) )
                {
                  v221 = v33;
                  v261 = v29;
                  v238 = v29;
                  v262 = v33;
                  *(_DWORD *)s2 = 7;
                  *(_QWORD *)&s2[8] = v239;
                  *(_DWORD *)&s2[16] = s1[1];
                  s2[20] = 0;
                  v260 = 0;
                  LOBYTE(v263) = 0;
                  v264 = 0;
                  v265 = 0;
                  v266 = 0;
                  v267 = 0;
                  v268 = 0;
                  v269 = 0;
                  v270 = 0;
                  LODWORD(v271) = 1;
                  v272 = 0;
                  sub_EA9A80(v35, (const __m128i *)s2, (__int64)v29, (__int64)s1[1], (__int64)v33, v31);
                  v29 = v238;
                  v33 = v221;
                }
                else
                {
                  v37 = *(_QWORD *)v35 + (v36 << 7);
                  if ( v37 )
                  {
                    v38 = v239;
                    *(_DWORD *)(v37 + 16) = s1[1];
                    *(_DWORD *)v37 = 7;
                    *(_QWORD *)(v37 + 8) = v38;
                    *(_BYTE *)(v37 + 20) = 0;
                    *(_QWORD *)(v37 + 24) = 0;
                    *(_BYTE *)(v37 + 48) = 0;
                    *(_QWORD *)(v37 + 56) = 0;
                    *(_QWORD *)(v37 + 64) = 0;
                    *(_QWORD *)(v37 + 72) = 0;
                    *(_QWORD *)(v37 + 80) = 0;
                    *(_QWORD *)(v37 + 88) = 0;
                    *(_QWORD *)(v37 + 96) = 0;
                    *(_QWORD *)(v37 + 104) = 0;
                    *(_DWORD *)(v37 + 112) = 1;
                    *(_BYTE *)(v37 + 120) = 0;
                    *(_QWORD *)(v37 + 32) = v29;
                    *(_QWORD *)(v37 + 40) = v33;
                  }
                  ++*(_DWORD *)(v35 + 8);
                }
                s1[0] = v34;
                s1[1] = v33;
              }
              v39 = *(_QWORD *)(v3 + 224);
              *(_QWORD *)&s2[8] = v28;
              LOWORD(v261) = 261;
              *(_QWORD *)s2 = v29;
              v40 = sub_E6C460(v39, (const char **)s2);
            }
            else
            {
              v40 = sub_E70DC0(*(_QWORD *)(v3 + 224), v10);
            }
            if ( *(_DWORD *)sub_ECD7B0(v3) == 38 )
            {
              v139 = v3 + 48;
              v140 = sub_EABDC0(v3);
              v143 = *(_DWORD **)(v3 + 48);
              v236 = v140;
              v144 = *(unsigned int *)(v3 + 56);
              v220 = v145;
              *(_BYTE *)(v3 + 155) = *v143 == 9;
              if ( (unsigned __int64)(40 * v144) > 0x28 )
              {
                v218 = v3;
                v146 = v19;
                v147 = v139;
                v148 = v143 + 10;
                v149 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v144 - 40) >> 3);
                do
                {
                  v150 = _mm_loadu_si128((const __m128i *)(v148 + 2));
                  v151 = *(v148 - 2) <= 0x40u;
                  *(v148 - 10) = *v148;
                  *((__m128i *)v148 - 2) = v150;
                  if ( !v151 )
                  {
                    v152 = *((_QWORD *)v148 - 2);
                    if ( v152 )
                      j_j___libc_free_0_0(v152);
                  }
                  v153 = *((_QWORD *)v148 + 3);
                  v148 += 10;
                  *((_QWORD *)v148 - 7) = v153;
                  LODWORD(v153) = *(v148 - 2);
                  *(v148 - 2) = 0;
                  *(v148 - 12) = v153;
                  --v149;
                }
                while ( v149 );
                v139 = v147;
                v19 = v146;
                v3 = v218;
                v143 = *(_DWORD **)(v218 + 48);
              }
              v154 = (unsigned int)(*(_DWORD *)(v3 + 56) - 1);
              *(_DWORD *)(v3 + 56) = v154;
              v155 = &v143[10 * v154];
              if ( v155[8] > 0x40u )
              {
                v156 = *((_QWORD *)v155 + 3);
                if ( v156 )
                  j_j___libc_free_0_0(v156);
              }
              if ( !*(_DWORD *)(v3 + 56) )
              {
                sub_1097F60(s2, v3 + 40);
                sub_EAA0A0(v139, *(_QWORD *)(v3 + 48), (unsigned __int64)s2, v157, v158, v159);
                if ( (unsigned int)v261 > 0x40 )
                {
                  if ( v260 )
                    j_j___libc_free_0_0(v260);
                }
              }
              v160 = *(_QWORD *)(v3 + 48);
              *(_BYTE *)(v3 + 155) = 0;
              *(_QWORD *)&s2[8] = v236;
              *(_DWORD *)s2 = 9;
              *(_QWORD *)&s2[16] = v220;
              LODWORD(v261) = 64;
              v260 = 0;
              sub_EAA0A0(v139, v160, (unsigned __int64)s2, (__int64)v143, v141, v142);
              if ( (unsigned int)v261 > 0x40 && v260 )
                j_j___libc_free_0_0(v260);
            }
            if ( *(_DWORD *)sub_ECD7B0(v3) == 9 )
              sub_EABFE0(v3);
            if ( *(_BYTE *)(*(_QWORD *)(v3 + 240) + 18LL)
              && *(_BYTE *)(v3 + 296)
              && (*(_BYTE *)(v40 + 8) & 0x20) != 0
              && (*(_BYTE *)(v40 + 13) & 2) == 0 )
            {
              v177 = *(_QWORD *)(v3 + 280);
              v253.m128i_i64[0] = (__int64)"non-private labels cannot appear between .cfi_startproc / .cfi_endproc pairs";
              LOWORD(v256) = 259;
              v19 = sub_ECDA70(v3, v177, &v253, 0, 0);
              if ( (_BYTE)v19 )
              {
                v178 = *(_QWORD *)(v3 + 288);
                *(_QWORD *)s2 = "previous .cfi_startproc was here";
                LOWORD(v261) = 259;
                v19 = sub_ECDA70(v3, v178, s2, 0, 0);
              }
            }
            else
            {
              v41 = *(_QWORD *)(v3 + 856) == 0;
              *(__m128i *)s2 = _mm_loadu_si128((const __m128i *)s1);
              if ( v41 )
              {
                v42 = *(_QWORD *)(v3 + 768);
                v43 = v42 + 16LL * *(unsigned int *)(v3 + 776);
                if ( v42 != v43 )
                {
                  v223 = v19;
                  v44 = v42 + 16LL * *(unsigned int *)(v3 + 776);
                  v219 = v3;
                  v45 = *(_QWORD *)&s2[8];
                  v46 = *(const void **)s2;
                  do
                  {
                    if ( v45 == *(_QWORD *)(v42 + 8) && (!v45 || !memcmp(*(const void **)v42, v46, v45)) )
                      break;
                    v42 += 16;
                  }
                  while ( v44 != v42 );
                  v43 = v44;
                  v3 = v219;
                  v19 = v223;
                }
                v47 = v43 != v42;
              }
              else
              {
                v47 = v3 + 824 != sub_EA96F0(v3 + 816, (__int64)s2);
              }
              if ( !v47 )
              {
                v48 = *(_BYTE **)(v3 + 8);
                v49 = *(void (**)())(*(_QWORD *)v48 + 176LL);
                if ( v49 != nullsub_369 )
                {
                  ((void (__fastcall *)(_BYTE *, __int64, __int64))v49)(v48, v40, v239);
                  v48 = *(_BYTE **)(v3 + 8);
                }
                if ( !v48[64] )
                  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v3 + 232) + 208LL))(
                    *(_QWORD *)(v3 + 232),
                    v40,
                    v239);
                if ( (unsigned __int8)sub_EAA750(v3) )
                  sub_E787B0(v40, *(_QWORD **)(v3 + 232), *(__int64 **)(v3 + 248), (unsigned __int64 *)&v239);
                v50 = *(_QWORD *)(v3 + 8);
                v51 = *(void (**)())(*(_QWORD *)v50 + 184LL);
                if ( v51 != nullsub_370 )
                {
                  v19 = 0;
                  ((void (__fastcall *)(__int64, __int64))v51)(v50, v40);
                }
              }
            }
          }
          goto LABEL_62;
        }
LABEL_59:
        v19 = 1;
LABEL_62:
        if ( v249 > 0x40 && v248 )
          j_j___libc_free_0_0(v248);
        return v19;
      }
      v17 = v225;
      v22 = **(_DWORD **)(v3 + 48);
    }
    if ( v22 == 28 )
    {
      v80 = *(__int64 (**)())(**(_QWORD **)(v3 + 8) + 136LL);
      if ( v80 == sub_EA21B0 || (v230 = v17, v102 = v80(), v17 = v230, v102) )
      {
        sub_EABFE0(v3);
        v19 = sub_EA98B0(v3, (const char *)s1[0], (size_t)s1[1], 2);
        goto LABEL_62;
      }
    }
    if ( (*(_BYTE *)(v3 + 472) & 1) != 0 )
    {
      v81 = s1[0];
      v216 = v17;
      v82 = *(_QWORD *)(v3 + 224);
      v228 = s1[1];
      v83 = sub_C92610();
      v84 = sub_C92860((__int64 *)(v82 + 2384), v81, (size_t)v228, v83);
      v17 = v216;
      if ( v84 != -1 )
      {
        v86 = *(_QWORD *)(v82 + 2384);
        v87 = *(unsigned int *)(v82 + 2392);
        v88 = (__int64 **)(v86 + 8LL * v84);
        if ( v88 != (__int64 **)(v86 + 8 * v87) )
        {
          v89 = *v88;
          v90 = LODWORD(qword_4F8A3A8[8]);
          if ( LODWORD(qword_4F8A3A8[8]) == (__int64)(*(_QWORD *)(v3 + 376) - *(_QWORD *)(v3 + 368)) >> 3 )
          {
            sub_222DF20(&v271);
            v273 = 0;
            v274 = 0;
            v275 = 0;
            v271 = off_4A06798;
            v276 = 0;
            v277 = 0;
            v278 = 0;
            *(_QWORD *)s2 = qword_4A071C8;
            *(_QWORD *)&s2[qword_4A071C8[-3]] = &unk_4A071F0;
            sub_222DD70(&s2[*(_QWORD *)(*(_QWORD *)s2 - 24LL)], 0);
            *(_QWORD *)&s2[16] = 0;
            v260 = 0;
            v261 = 0;
            *(_QWORD *)s2 = off_4A07238;
            v262 = 0;
            v263 = 0;
            v271 = off_4A07260;
            v264 = 0;
            *(_QWORD *)&s2[8] = off_4A07480;
            sub_220A990(&v265);
            LODWORD(v266) = 16;
            LOBYTE(v269) = 0;
            *(_QWORD *)&s2[8] = off_4A07080;
            v267 = &v269;
            v268 = 0;
            sub_222DD70(&v271, &s2[8]);
            sub_223E0D0(s2, "macros cannot be nested more than ", 34);
            v120 = sub_223E760(s2, v90);
            sub_223E0D0(v120, " levels deep.", 13);
            sub_223E0D0(v120, " Use -asm-macro-max-nesting-depth to increase this limit.", 57);
            v250.m128i_i64[1] = 0;
            v250.m128i_i64[0] = (__int64)v251;
            LOBYTE(v251[0]) = 0;
            if ( v263 )
            {
              if ( v263 <= (unsigned __int64)v261 )
                sub_2241130(&v250, 0, 0, v262, v261 - v262);
              else
                sub_2241130(&v250, 0, 0, v262, v263 - (_QWORD)v262);
            }
            else
            {
              sub_2240AE0(&v250, &v267);
            }
            v121 = &v253;
            LOWORD(v256) = 260;
            v253.m128i_i64[0] = (__int64)&v250;
            v19 = sub_ECE0E0(v3, &v253, 0, 0);
            if ( (_QWORD *)v250.m128i_i64[0] != v251 )
            {
              v121 = (__m128i *)(v251[0] + 1LL);
              j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
            }
            *(_QWORD *)s2 = off_4A07238;
            v271 = off_4A07260;
            *(_QWORD *)&s2[8] = off_4A07080;
            if ( v267 != &v269 )
            {
              v121 = (__m128i *)(v269 + 1);
              j_j___libc_free_0(v267, v269 + 1);
            }
            *(_QWORD *)&s2[8] = off_4A07480;
            sub_2209150(&v265, v121, v122);
            *(_QWORD *)s2 = qword_4A071C8;
            *(_QWORD *)&s2[qword_4A071C8[-3]] = &unk_4A071F0;
            v271 = off_4A06798;
            sub_222E050(&v271);
            goto LABEL_62;
          }
          v243 = 0;
          v229 = v239;
          v244 = 0;
          v245 = 0;
          v19 = sub_EBC8F0(v3, v89 + 1, &v243, v87, v216, v85);
          if ( (_BYTE)v19 )
            goto LABEL_180;
          *(_QWORD *)s2 = &v260;
          v257 = 0x100000000LL;
          *(_QWORD *)&s2[8] = 0;
          *(_QWORD *)&s2[16] = 256;
          v253.m128i_i64[0] = (__int64)&unk_49DD288;
          v253.m128i_i64[1] = 2;
          v254 = 0;
          v255 = 0;
          v256 = 0;
          v258 = s2;
          sub_CB5980((__int64)&v253, 0, 0, 0);
          v91 = v89[6];
          v92 = v89[5];
          if ( *(_BYTE *)(v3 + 868) && v91 == v92 )
          {
            v94 = 0;
            v93 = 0xAAAAAAAAAAAAAAABLL * (v244 - v243);
          }
          else
          {
            v93 = 0xAAAAAAAAAAAAAAABLL * (v244 - v243);
            v94 = v93;
            if ( 0xAAAAAAAAAAAAAAABLL * ((v91 - v92) >> 4) != v93 )
            {
              v250.m128i_i64[0] = (__int64)"Wrong number of arguments";
              v252 = 259;
              v95 = sub_ECD7B0(v3);
              v96 = sub_ECD6A0(v95);
              v19 = sub_ECDA70(v3, v96, &v250, 0, 0);
LABEL_178:
              v253.m128i_i64[0] = (__int64)&unk_49DD388;
              sub_CB5840((__int64)&v253);
              if ( *(const void ***)s2 != &v260 )
                _libc_free(*(_QWORD *)s2, v96);
LABEL_180:
              v97 = v244;
              v98 = v243;
              if ( v244 != v243 )
              {
                do
                {
                  v99 = v98[1];
                  v100 = *v98;
                  if ( v99 != *v98 )
                  {
                    do
                    {
                      if ( *(_DWORD *)(v100 + 32) > 0x40u )
                      {
                        v101 = *(_QWORD *)(v100 + 24);
                        if ( v101 )
                          j_j___libc_free_0_0(v101);
                      }
                      v100 += 40;
                    }
                    while ( v99 != v100 );
                    v100 = *v98;
                  }
                  if ( v100 )
                    j_j___libc_free_0(v100, v98[2] - v100);
                  v98 += 3;
                }
                while ( v97 != v98 );
                v98 = v243;
              }
              if ( v98 )
                j_j___libc_free_0(v98, v245 - (_QWORD)v98);
              goto LABEL_62;
            }
          }
          v123 = sub_EA4200(v3, v253.m128i_i32, (__int64)(v89 + 1), v92, v94, 1, (__int64)v243, v93);
          v96 = v214;
          v19 = v123;
          if ( !(_BYTE)v123 )
          {
            v126 = (_QWORD *)v256;
            if ( (unsigned __int64)(v255 - v256) <= 9 )
            {
              sub_CB6200((__int64)&v253, ".endmacro\n", 0xAu);
            }
            else
            {
              *(_WORD *)(v256 + 8) = 2671;
              *v126 = 0x7263616D646E652ELL;
              v256 += 10;
            }
            v250.m128i_i64[0] = (__int64)"<instantiation>";
            v252 = 259;
            sub_C7DE20(&v240, *(const void **)v258, *((_QWORD *)v258 + 1), (__int64)&v250, v124, v125);
            v217 = sub_ECD7B0(v3);
            v127 = (__int64)(*(_QWORD *)(v3 + 328) - *(_QWORD *)(v3 + 320)) >> 3;
            v128 = sub_22077B0(32);
            if ( v128 )
            {
              v129 = v229;
              v235 = v128;
              *(_QWORD *)v128 = v129;
              *(_DWORD *)(v128 + 8) = *(_DWORD *)(v3 + 304);
              v130 = sub_ECD6A0(v217);
              v128 = v235;
              *(_QWORD *)(v235 + 16) = v130;
              *(_QWORD *)(v235 + 24) = v127;
            }
            v241 = v128;
            v131 = *(_BYTE **)(v3 + 376);
            if ( v131 == *(_BYTE **)(v3 + 384) )
            {
              sub_EA27B0(v3 + 368, v131, &v241);
            }
            else
            {
              if ( v131 )
                *(_QWORD *)v131 = v128;
              *(_QWORD *)(v3 + 376) += 8LL;
            }
            v132 = v240;
            v133 = *(__int64 **)(v3 + 248);
            v250.m128i_i64[1] = 0;
            ++*(_DWORD *)(v3 + 476);
            v250.m128i_i64[0] = v132;
            v251[0] = 0;
            v134 = (_QWORD *)v133[1];
            v240 = 0;
            if ( v134 == (_QWORD *)v133[2] )
            {
              sub_C12520(v133, (__int64)v134, (__int64)&v250);
              v135 = (_QWORD *)v133[1];
            }
            else
            {
              if ( v134 )
              {
                sub_C8EDF0(v134, &v250);
                v134 = (_QWORD *)v133[1];
              }
              v135 = v134 + 3;
              v133[1] = (__int64)v135;
            }
            v136 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v135 - *v133) >> 3);
            sub_C8EE20(v250.m128i_i64);
            v137 = *(_QWORD **)(v3 + 248);
            *(_DWORD *)(v3 + 304) = v136;
            v138 = *(_QWORD *)(*v137 + 24LL * (unsigned int)(v136 - 1));
            v96 = *(_QWORD *)(v138 + 8);
            sub_1095BD0(v3 + 40, v96, *(_QWORD *)(v138 + 16) - v96, 0, 1);
            sub_EABFE0(v3);
            if ( v240 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v240 + 8LL))(v240);
          }
          goto LABEL_178;
        }
      }
    }
    if ( s1[1] )
    {
      if ( *(_BYTE *)s1[0] != 46 )
      {
        if ( !*(_BYTE *)(v3 + 869) )
          goto LABEL_58;
        if ( s1[1] == (void *)5 )
        {
          if ( *(_DWORD *)s1[0] == 1768777055 && *((_BYTE *)s1[0] + 4) == 116
            || *(_DWORD *)s1[0] == 1229800799 && *((_BYTE *)s1[0] + 4) == 84 )
          {
            goto LABEL_206;
          }
          if ( *(_DWORD *)s1[0] == 1734962273 && *((_BYTE *)s1[0] + 4) == 110
            || *(_DWORD *)s1[0] == 1195985985 && *((_BYTE *)s1[0] + 4) == 78 )
          {
            v237 = v239;
            v161 = sub_ECD690(v3 + 40);
            *(_QWORD *)s2 = 0;
            v162 = v161;
            LOBYTE(v163) = sub_EAC4D0(v3, v253.m128i_i64, (__int64)s2);
            v19 = v163;
            if ( !(_BYTE)v163 )
            {
              if ( *(_BYTE *)v253.m128i_i64[0] == 1 )
              {
                v165 = *(_QWORD *)(v253.m128i_i64[0] + 16);
                if ( !v165 || (v165 & (v165 - 1)) != 0 )
                {
                  *(_QWORD *)s2 = "literal value not a power of two greater then zero";
                  LOWORD(v261) = 259;
                  v19 = sub_ECDA70(v3, v162, s2, 0, 0);
                }
                else
                {
                  _BitScanReverse64(&v165, v165);
                  v166 = *(_QWORD *)(a2 + 88);
                  v167 = (const void *)(int)(63 - (v165 ^ 0x3F));
                  v168 = *(unsigned int *)(v166 + 8);
                  v169 = *(_QWORD *)v166;
                  v170 = *(unsigned int *)(v166 + 12);
                  v171 = *(_QWORD *)v166 + (v168 << 7);
                  if ( v168 >= v170 )
                  {
                    v193 = v168 + 1;
                    *(_DWORD *)s2 = 0;
                    *(_QWORD *)&s2[8] = v239;
                    *(_DWORD *)&s2[16] = 5;
                    s2[20] = 0;
                    v260 = v167;
                    v261 = 0;
                    v262 = 0;
                    LOBYTE(v263) = 0;
                    v264 = 0;
                    v265 = 0;
                    v266 = 0;
                    v267 = 0;
                    v268 = 0;
                    v269 = 0;
                    v270 = 0;
                    LODWORD(v271) = 1;
                    v272 = 0;
                    if ( v170 < v193 )
                    {
                      v198 = (const void *)(v166 + 16);
                      if ( v169 > (unsigned __int64)s2 || v171 <= (unsigned __int64)s2 )
                      {
                        sub_C8D5F0(v166, v198, v193, 0x80u, v239, v164);
                        v194 = *(_QWORD *)v166;
                      }
                      else
                      {
                        sub_C8D5F0(v166, v198, v193, 0x80u, v239, v164);
                        v194 = *(_QWORD *)v166;
                        v11 = &s2[*(_QWORD *)v166 - v169];
                      }
                    }
                    else
                    {
                      v194 = *(_QWORD *)v166;
                    }
                    v195 = 32;
                    v196 = v11;
                    v197 = (_DWORD *)(((unsigned __int64)*(unsigned int *)(v166 + 8) << 7) + v194);
                    while ( v195 )
                    {
                      *v197++ = *v196++;
                      --v195;
                    }
                    ++*(_DWORD *)(v166 + 8);
                  }
                  else
                  {
                    if ( v171 )
                    {
                      *(_DWORD *)v171 = 0;
                      *(_QWORD *)(v171 + 8) = v237;
                      *(_DWORD *)(v171 + 16) = 5;
                      *(_BYTE *)(v171 + 20) = 0;
                      *(_QWORD *)(v171 + 24) = v167;
                      *(_QWORD *)(v171 + 32) = 0;
                      *(_QWORD *)(v171 + 40) = 0;
                      *(_BYTE *)(v171 + 48) = 0;
                      *(_QWORD *)(v171 + 56) = 0;
                      *(_QWORD *)(v171 + 64) = 0;
                      *(_QWORD *)(v171 + 72) = 0;
                      *(_QWORD *)(v171 + 80) = 0;
                      *(_QWORD *)(v171 + 88) = 0;
                      *(_QWORD *)(v171 + 96) = 0;
                      *(_QWORD *)(v171 + 104) = 0;
                      *(_DWORD *)(v171 + 112) = 1;
                      *(_BYTE *)(v171 + 120) = 0;
                    }
                    ++*(_DWORD *)(v166 + 8);
                  }
                }
              }
              else
              {
                *(_QWORD *)s2 = "unexpected expression in align";
                LOWORD(v261) = 259;
                v19 = sub_ECDA70(v3, v162, s2, 0, 0);
              }
            }
            goto LABEL_62;
          }
        }
        else if ( s1[1] == (void *)6 )
        {
          if ( *(_DWORD *)s1[0] == 1835360095 && *((_WORD *)s1[0] + 2) == 29801
            || *(_DWORD *)s1[0] == 1296392031 && *((_WORD *)s1[0] + 2) == 21577 )
          {
LABEL_206:
            v231 = s1[1];
            v103 = v239;
            v104 = sub_ECD690(v3 + 40);
            *(_QWORD *)s2 = 0;
            v105 = v104;
            LOBYTE(v106) = sub_EAC4D0(v3, v253.m128i_i64, (__int64)s2);
            v19 = v106;
            if ( !(_BYTE)v106 )
            {
              if ( *(_BYTE *)v253.m128i_i64[0] == 1 )
              {
                v108 = *(_QWORD *)(v253.m128i_i64[0] + 16);
                if ( (v108 & 0xFFFFFFFFFFFFFF00LL) == 0 || v108 == (char)v108 )
                {
                  v114 = *(_QWORD *)(a2 + 88);
                  v115 = *(unsigned int *)(v114 + 8);
                  v116 = *(_QWORD *)v114;
                  v117 = *(unsigned int *)(v114 + 12);
                  v118 = *(_DWORD *)(v114 + 8);
                  v119 = *(_QWORD *)v114 + (v115 << 7);
                  if ( v115 >= v117 )
                  {
                    v172 = v115 + 1;
                    *(_QWORD *)&s2[8] = v239;
                    *(_DWORD *)s2 = 2;
                    *(_DWORD *)&s2[16] = (_DWORD)v231;
                    s2[20] = 0;
                    v260 = 0;
                    v261 = 0;
                    v262 = 0;
                    LOBYTE(v263) = 0;
                    v264 = 0;
                    v265 = 0;
                    v266 = 0;
                    v267 = 0;
                    v268 = 0;
                    v269 = 0;
                    v270 = 0;
                    LODWORD(v271) = 1;
                    v272 = 0;
                    if ( v117 < v172 )
                    {
                      v192 = (const void *)(v114 + 16);
                      if ( v116 > (unsigned __int64)s2 || v119 <= (unsigned __int64)s2 )
                      {
                        sub_C8D5F0(v114, v192, v172, 0x80u, v107, (__int64)v231);
                        v173 = *(_QWORD *)v114;
                      }
                      else
                      {
                        sub_C8D5F0(v114, v192, v172, 0x80u, v107, (__int64)v231);
                        v173 = *(_QWORD *)v114;
                        v11 = &s2[*(_QWORD *)v114 - v116];
                      }
                    }
                    else
                    {
                      v173 = *(_QWORD *)v114;
                    }
                    v174 = 32;
                    v175 = v11;
                    v176 = (_DWORD *)(((unsigned __int64)*(unsigned int *)(v114 + 8) << 7) + v173);
                    while ( v174 )
                    {
                      *v176++ = *v175++;
                      --v174;
                    }
                    ++*(_DWORD *)(v114 + 8);
                  }
                  else
                  {
                    if ( v119 )
                    {
                      *(_DWORD *)v119 = 2;
                      *(_QWORD *)(v119 + 8) = v103;
                      *(_DWORD *)(v119 + 16) = (_DWORD)v231;
                      *(_BYTE *)(v119 + 20) = 0;
                      *(_QWORD *)(v119 + 24) = 0;
                      *(_QWORD *)(v119 + 32) = 0;
                      *(_QWORD *)(v119 + 40) = 0;
                      *(_BYTE *)(v119 + 48) = 0;
                      *(_QWORD *)(v119 + 56) = 0;
                      *(_QWORD *)(v119 + 64) = 0;
                      *(_QWORD *)(v119 + 72) = 0;
                      *(_QWORD *)(v119 + 80) = 0;
                      *(_QWORD *)(v119 + 88) = 0;
                      *(_QWORD *)(v119 + 96) = 0;
                      *(_QWORD *)(v119 + 104) = 0;
                      *(_DWORD *)(v119 + 112) = 1;
                      *(_BYTE *)(v119 + 120) = 0;
                      v118 = *(_DWORD *)(v114 + 8);
                    }
                    *(_DWORD *)(v114 + 8) = v118 + 1;
                  }
                }
                else
                {
                  *(_QWORD *)s2 = "literal value out of range for directive";
                  LOWORD(v261) = 259;
                  v19 = sub_ECDA70(v3, v105, s2, 0, 0);
                }
              }
              else
              {
                *(_QWORD *)s2 = "unexpected expression in _emit";
                LOWORD(v261) = 259;
                v19 = sub_ECDA70(v3, v105, s2, 0, 0);
              }
            }
            goto LABEL_62;
          }
        }
        else if ( s1[1] == (void *)4 && (*(_DWORD *)s1[0] == 1852143205 || *(_DWORD *)s1[0] == 1313166917) )
        {
          v109 = v239;
          v110 = *(_QWORD *)(a2 + 88);
          v111 = *(unsigned int *)(v110 + 8);
          v112 = v111;
          if ( *(_DWORD *)(v110 + 12) <= (unsigned int)v111 )
          {
            *(_DWORD *)s2 = 1;
            *(_QWORD *)&s2[8] = v239;
            *(_DWORD *)&s2[16] = 4;
            s2[20] = 0;
            v260 = 0;
            v261 = 0;
            v262 = 0;
            LOBYTE(v263) = 0;
            v264 = 0;
            v265 = 0;
            v266 = 0;
            v267 = 0;
            v268 = 0;
            v269 = 0;
            v270 = 0;
            LODWORD(v271) = 1;
            v272 = 0;
            sub_EA9A80(v110, (const __m128i *)s2, v111, v239, v17, 4);
            v23 = *(_BYTE *)(v3 + 869);
          }
          else
          {
            v113 = *(_QWORD *)v110 + (v111 << 7);
            if ( v113 )
            {
              *(_DWORD *)v113 = 1;
              *(_QWORD *)(v113 + 8) = v109;
              *(_DWORD *)(v113 + 16) = 4;
              *(_BYTE *)(v113 + 20) = 0;
              *(_QWORD *)(v113 + 24) = 0;
              *(_QWORD *)(v113 + 32) = 0;
              *(_QWORD *)(v113 + 40) = 0;
              *(_BYTE *)(v113 + 48) = 0;
              *(_QWORD *)(v113 + 56) = 0;
              *(_QWORD *)(v113 + 64) = 0;
              *(_QWORD *)(v113 + 72) = 0;
              *(_QWORD *)(v113 + 80) = 0;
              *(_QWORD *)(v113 + 88) = 0;
              *(_QWORD *)(v113 + 96) = 0;
              *(_QWORD *)(v113 + 104) = 0;
              *(_DWORD *)(v113 + 112) = 1;
              *(_BYTE *)(v113 + 120) = 0;
              v112 = *(_DWORD *)(v110 + 8);
            }
            *(_DWORD *)(v110 + 8) = v112 + 1;
            v23 = *(_BYTE *)(v3 + 869);
          }
LABEL_51:
          if ( v23 )
            goto LABEL_52;
LABEL_58:
          if ( !(unsigned __int8)sub_EA2540(v3) )
          {
LABEL_52:
            v24 = _mm_loadu_si128(&v247);
            *(_DWORD *)s2 = v246;
            *(__m128i *)&s2[8] = v24;
            LODWORD(v261) = v249;
            if ( v249 > 0x40 )
              sub_C43780((__int64)&v260, &v248);
            else
              v260 = v248;
            v19 = sub_EAA8B0(v3, a2, (__int64)s1[0], (__int64)s1[1], (int *)s2, v239);
            if ( (unsigned int)v261 > 0x40 && v260 )
              j_j___libc_free_0_0(v260);
            goto LABEL_62;
          }
          goto LABEL_59;
        }
LABEL_50:
        v23 = *(_BYTE *)(v3 + 869);
        goto LABEL_51;
      }
      if ( s1[1] != (void *)1 )
      {
        v61 = *(__int64 **)(v3 + 8);
        v62 = *v61;
        v63 = *(void (**)())(*v61 + 192);
        if ( v63 != nullsub_371 )
        {
          v234 = v17;
          ((void (__fastcall *)(__int64 *, _QWORD))v63)(v61, *(_QWORD *)(v3 + 232));
          v61 = *(__int64 **)(v3 + 8);
          LODWORD(v17) = v234;
          v62 = *v61;
        }
        v64 = *(__int64 (__fastcall **)(__int64 *, _BYTE *))(v62 + 72);
        v65 = _mm_loadu_si128(&v247);
        *(_DWORD *)s2 = v246;
        *(__m128i *)&s2[8] = v65;
        LODWORD(v261) = v249;
        if ( v249 > 0x40 )
        {
          v232 = v17;
          sub_C43780((__int64)&v260, &v248);
          LODWORD(v17) = v232;
        }
        else
        {
          v260 = v248;
        }
        v226 = v17;
        v66 = v64(v61, s2);
        v67 = v226;
        v68 = v66;
        if ( (unsigned int)v261 > 0x40 && v260 )
        {
          j_j___libc_free_0_0(v260);
          v67 = v226;
        }
        if ( v68 == 1 )
        {
LABEL_240:
          v19 = 1;
        }
        else
        {
          v227 = v67;
          if ( v68 )
          {
            v69 = s1[1];
            v70 = s1[0];
            v71 = sub_C92610();
            v72 = sub_C92860((__int64 *)(v3 + 344), v70, (size_t)v69, v71);
            if ( v72 == -1
              || (v75 = *(_QWORD *)(v3 + 344),
                  v73 = *(unsigned int *)(v3 + 352),
                  v76 = (__int64 *)(v75 + 8LL * v72),
                  v76 == (__int64 *)(v75 + 8 * v73))
              || (v77 = *v76,
                  v78 = *(_QWORD *)(v77 + 8),
                  v79 = *(__int64 (__fastcall **)(__int64, void *, void *, __int64, _QWORD))(v77 + 16),
                  !v78) )
            {
              switch ( v227 )
              {
                case 1u:
                case 2u:
                  v19 = sub_EBB3E0(v3, 0);
                  break;
                case 3u:
                  v19 = sub_EBB3E0(v3, 1);
                  break;
                case 4u:
                  v253.m128i_i8[0] = 0;
                  *(_QWORD *)s2 = v3;
                  *(_QWORD *)&s2[8] = &v253;
                  v19 = sub_ECE300(v3, sub_EAEB40, s2, 1);
                  break;
                case 5u:
                case 6u:
                  v253.m128i_i8[0] = 1;
                  *(_QWORD *)s2 = v3;
                  *(_QWORD *)&s2[8] = &v253;
                  v19 = sub_ECE300(v3, sub_EAEB40, s2, 1);
                  break;
                case 7u:
                case 0x14u:
                  v19 = sub_EA2630(v3, 1);
                  break;
                case 8u:
                case 0xAu:
                case 0xBu:
                case 0x12u:
                case 0x18u:
                  v19 = sub_EA2630(v3, 2);
                  break;
                case 9u:
                  v19 = sub_EB0A00((_QWORD *)v3, v239);
                  break;
                case 0xCu:
                case 0xDu:
                case 0xEu:
                case 0x16u:
                  v19 = sub_EA2630(v3, 4);
                  break;
                case 0xFu:
                case 0x10u:
                  v19 = sub_EA2630(v3, 8);
                  break;
                case 0x11u:
                  *(_QWORD *)s2 = v3;
                  v19 = sub_ECE300(v3, sub_EAFCF0, s2, 1);
                  break;
                case 0x13u:
                  v19 = sub_EA2630(v3, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 224) + 152LL) + 8LL));
                  break;
                case 0x15u:
                case 0x2Bu:
                  v213 = sub_C33320();
                  *(_QWORD *)s2 = v3;
                  *(_QWORD *)&s2[8] = v213;
                  v19 = sub_ECE300(v3, sub_EBC0E0, s2, 1);
                  break;
                case 0x17u:
                case 0x29u:
                case 0x2Au:
                  v212 = sub_C33310();
                  *(_QWORD *)s2 = v3;
                  *(_QWORD *)&s2[8] = v212;
                  v19 = sub_ECE300(v3, sub_EBC0E0, s2, 1);
                  break;
                case 0x19u:
                case 0x20u:
                case 0x4Du:
                case 0x4Eu:
                  v253.m128i_i64[0] = (__int64)" not currently supported for this target";
                  LOWORD(v256) = 259;
                  v250 = *(__m128i *)s1;
                  v252 = 261;
                  sub_9C6370((__m128i *)s2, &v250, &v253, v73, v227, v74);
                  v19 = sub_ECE0E0(v3, s2, 0, 0);
                  break;
                case 0x1Au:
                case 0x1Fu:
                  v19 = sub_EADE50(v3, (__int64)s1[0], (__int64)s1[1], 2u);
                  break;
                case 0x1Bu:
                  v19 = sub_EADE50(v3, (__int64)s1[0], (__int64)s1[1], 1u);
                  break;
                case 0x1Cu:
                  v211 = sub_C33320();
                  v19 = sub_EBC1D0(v3, (__int64)s1[0], (__int64)s1[1], (__int64)v211);
                  break;
                case 0x1Du:
                  v19 = sub_EADE50(v3, (__int64)s1[0], (__int64)s1[1], 4u);
                  break;
                case 0x1Eu:
                  v210 = sub_C33310();
                  v19 = sub_EBC1D0(v3, (__int64)s1[0], (__int64)s1[1], (__int64)v210);
                  break;
                case 0x21u:
                case 0x27u:
                  v19 = sub_EAD790(v3, (__int64)s1[0], (__int64)s1[1], 2u);
                  break;
                case 0x22u:
                  v19 = sub_EAD790(v3, (__int64)s1[0], (__int64)s1[1], 1u);
                  break;
                case 0x23u:
                  v19 = sub_EAD790(v3, (__int64)s1[0], (__int64)s1[1], 8u);
                  break;
                case 0x24u:
                case 0x26u:
                  v19 = sub_EAD790(v3, (__int64)s1[0], (__int64)s1[1], 4u);
                  break;
                case 0x25u:
                case 0x28u:
                  v19 = sub_EAD790(v3, (__int64)s1[0], (__int64)s1[1], 0xCu);
                  break;
                case 0x2Cu:
                  v19 = sub_EAC980(v3, *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 + 224) + 152LL) + 260LL) ^ 1u, 1u);
                  break;
                case 0x2Du:
                  v19 = sub_EAC980(v3, *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 + 224) + 152LL) + 260LL) ^ 1u, 4u);
                  break;
                case 0x2Eu:
                  v19 = sub_EAC980(v3, 0, 1u);
                  break;
                case 0x2Fu:
                  v19 = sub_EAC980(v3, 0, 2u);
                  break;
                case 0x30u:
                  v19 = sub_EAC980(v3, 0, 4u);
                  break;
                case 0x31u:
                  v19 = sub_EAC980(v3, 1u, 1u);
                  break;
                case 0x32u:
                  v19 = sub_EAC980(v3, 1u, 2u);
                  break;
                case 0x33u:
                  v19 = sub_EAC980(v3, 1u, 4u);
                  break;
                case 0x34u:
                  v19 = sub_EADAF0(v3);
                  break;
                case 0x35u:
                  v19 = sub_EADBC0(v3);
                  break;
                case 0x36u:
                  if ( *(_QWORD *)(v3 + 376) == *(_QWORD *)(v3 + 368) )
                  {
                    *(_QWORD *)s2 = "unmatched '.endr' directive";
                    LOWORD(v261) = 259;
                    v19 = sub_ECE0E0(v3, s2, 0, 0);
                  }
                  else
                  {
                    sub_EAFEB0(v3);
                  }
                  break;
                case 0x37u:
                  v19 = sub_EAD1B0(v3);
                  break;
                case 0x38u:
                  v19 = sub_EB7950(v3);
                  break;
                case 0x39u:
                  if ( !*(_BYTE *)(v3 + 869) && (unsigned __int8)sub_EA2540(v3) )
                    goto LABEL_240;
                  v19 = sub_ECE000(v3);
                  if ( (_BYTE)v19 )
                    goto LABEL_240;
                  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v3 + 232) + 1256LL))(*(_QWORD *)(v3 + 232));
                  break;
                case 0x3Au:
                  v19 = sub_EAFDE0(v3);
                  break;
                case 0x3Bu:
                  sub_EB4E00(v3);
                  break;
                case 0x3Cu:
                case 0x3Du:
                  v19 = sub_EA2510(v3, 9);
                  break;
                case 0x3Eu:
                  v19 = sub_EA2510(v3, 16);
                  break;
                case 0x3Fu:
                  v19 = sub_EA2510(v3, 18);
                  break;
                case 0x40u:
                  v19 = sub_EA2510(v3, 19);
                  break;
                case 0x41u:
                  v19 = sub_EA2510(v3, 21);
                  break;
                case 0x42u:
                  v19 = sub_EA2510(v3, 23);
                  break;
                case 0x43u:
                  v19 = sub_EA2510(v3, 25);
                  break;
                case 0x44u:
                  v19 = sub_EA2510(v3, 26);
                  break;
                case 0x45u:
                  v19 = sub_EA2510(v3, 27);
                  break;
                case 0x46u:
                  v19 = sub_EA2510(v3, 1);
                  break;
                case 0x47u:
                case 0x48u:
                  v19 = sub_EB7570(v3, 0);
                  break;
                case 0x49u:
                  v19 = sub_EB7570(v3, 1u);
                  break;
                case 0x4Au:
                  v19 = sub_EABF30(v3, v239);
                  break;
                case 0x4Bu:
                  v19 = sub_EAEC30(v3);
                  break;
                case 0x4Cu:
                  v19 = sub_EAEF50((_QWORD *)v3);
                  break;
                case 0x4Fu:
                  v19 = sub_EB4AE0(v3, v239, (__int64)s1[0], (__int64)s1[1]);
                  break;
                case 0x50u:
                  v19 = sub_EBD6F0(v3, v239);
                  break;
                case 0x51u:
                  v19 = sub_EBDA90(v3, v239);
                  break;
                case 0x65u:
                case 0x66u:
                  v19 = sub_EAE0B0(v3);
                  break;
                case 0x67u:
                  v19 = sub_EB7A90(v3, v239);
                  break;
                case 0x68u:
                  sub_ECE2A0(v3, 4);
                  v19 = sub_ECE000(v3);
                  break;
                case 0x69u:
                  v19 = sub_EB00F0(v3);
                  break;
                case 0x6Au:
                  v19 = sub_EB8840(v3);
                  break;
                case 0x6Bu:
                  *(_QWORD *)s2 = "unsupported directive '.stabs'";
                  LOWORD(v261) = 259;
                  v19 = sub_ECE0E0(v3, s2, 0, 0);
                  break;
                case 0x6Cu:
                  v19 = sub_EAF4F0(v3);
                  break;
                case 0x6Du:
                  v19 = sub_EA2700(v3);
                  break;
                case 0x6Eu:
                  v19 = sub_EB03C0(v3);
                  break;
                case 0x6Fu:
                  v19 = sub_EB0680(v3);
                  break;
                case 0x70u:
                  v19 = sub_EB8AB0(v3);
                  break;
                case 0x71u:
                  v19 = sub_EB8CC0(v3);
                  break;
                case 0x72u:
                  v19 = sub_EB8FB0(v3);
                  break;
                case 0x73u:
                  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v3 + 232) + 800LL))(*(_QWORD *)(v3 + 232));
                  break;
                case 0x74u:
                  v19 = sub_EAFA30(v3);
                  break;
                case 0x75u:
                  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v3 + 232) + 808LL))(*(_QWORD *)(v3 + 232));
                  break;
                case 0x76u:
                  *(_QWORD *)s2 = "expected integer";
                  LOWORD(v261) = 259;
                  if ( (unsigned __int8)sub_ECE130(v3, &v253, s2) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                  {
                    v19 = 1;
                  }
                  else
                  {
                    v208 = *(_QWORD *)(v3 + 232);
                    v209 = *(void (**)())(*(_QWORD *)v208 + 816LL);
                    if ( v209 != nullsub_111 )
                      ((void (__fastcall *)(__int64, _QWORD))v209)(v208, v253.m128i_u32[0]);
                  }
                  break;
                case 0x77u:
                  v19 = sub_EB95A0(v3);
                  break;
                case 0x78u:
                  v19 = sub_EB9680(v3);
                  break;
                case 0x79u:
                  v19 = sub_EB97E0(v3);
                  break;
                case 0x7Au:
                  if ( *(_BYTE *)(v3 + 296) )
                    *(_BYTE *)(v3 + 296) = 0;
                  v19 = sub_ECE000(v3);
                  if ( (_BYTE)v19 )
                    goto LABEL_240;
                  sub_E99540(*(_DWORD **)(v3 + 232));
                  break;
                case 0x7Bu:
                  v19 = sub_EAD3D0(v3, v239);
                  break;
                case 0x7Cu:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAC8B0(v3, s2) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 232) + 872LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2,
                      v239);
                  break;
                case 0x7Du:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAC8B0(v3, s2) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 232) + 960LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2,
                      v239);
                  break;
                case 0x7Eu:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAD290(v3, s2, v239) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 232) + 880LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2,
                      v239);
                  break;
                case 0x7Fu:
                  v19 = sub_EAD480(v3, v239);
                  break;
                case 0x80u:
                  v19 = sub_EAD580(v3, v239);
                  break;
                case 0x81u:
                  v19 = sub_EAD630(v3, v239);
                  break;
                case 0x82u:
                  v19 = sub_EB98F0(v3, 1);
                  break;
                case 0x83u:
                  v19 = sub_EB98F0(v3, 0);
                  break;
                case 0x84u:
                  v19 = sub_ECE000(v3);
                  if ( !(_BYTE)v19 )
                    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v3 + 232) + 920LL))(
                      *(_QWORD *)(v3 + 232),
                      v239);
                  break;
                case 0x85u:
                  v19 = sub_ECE000(v3);
                  if ( !(_BYTE)v19 )
                    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v3 + 232) + 928LL))(
                      *(_QWORD *)(v3 + 232),
                      v239);
                  break;
                case 0x86u:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAD290(v3, s2, v239) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 232) + 936LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2,
                      v239);
                  break;
                case 0x87u:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAD290(v3, s2, v239) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 232) + 944LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2,
                      v239);
                  break;
                case 0x88u:
                  v19 = sub_EB0850(v3, v239);
                  break;
                case 0x89u:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAD290(v3, s2, v239) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(v3 + 232) + 976LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2);
                  break;
                case 0x8Au:
                  v19 = sub_ECE000(v3);
                  if ( (_BYTE)v19 )
                    goto LABEL_240;
                  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v3 + 232) + 992LL))(*(_QWORD *)(v3 + 232));
                  break;
                case 0x8Bu:
                  *(_QWORD *)s2 = 0;
                  if ( (unsigned __int8)sub_EAD290(v3, s2, v239) || (v19 = sub_ECE000(v3), (_BYTE)v19) )
                    v19 = 1;
                  else
                    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v3 + 232) + 1000LL))(
                      *(_QWORD *)(v3 + 232),
                      *(_QWORD *)s2,
                      v239);
                  break;
                case 0x8Cu:
                  v19 = sub_EAD320(v3, v239);
                  break;
                case 0x8Du:
                  v19 = sub_ECE000(v3);
                  if ( !(_BYTE)v19 )
                    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v3 + 232) + 1016LL))(
                      *(_QWORD *)(v3 + 232),
                      v239);
                  break;
                case 0x8Eu:
                  v19 = sub_EB9A90(v3);
                  break;
                case 0x90u:
                  v19 = sub_EAD6E0(v3, v239);
                  break;
                case 0x91u:
                case 0x92u:
                  v205 = s1[0];
                  v206 = s1[1];
                  v19 = sub_ECE000(v3);
                  if ( !(_BYTE)v19 )
                  {
                    v207 = 0;
                    if ( v206 == (void *)10 )
                      v207 = memcmp(v205, ".macros_on", 0xAu) == 0;
                    *(_BYTE *)(v3 + 472) = v207 | *(_BYTE *)(v3 + 472) & 0xFE;
                  }
                  break;
                case 0x93u:
                case 0x94u:
                  v202 = s1[0];
                  v203 = s1[1];
                  v19 = sub_ECE000(v3);
                  if ( !(_BYTE)v19 )
                  {
                    v204 = 0;
                    if ( v203 == (void *)9 )
                      v204 = memcmp(v202, ".altmacro", 9u) == 0;
                    *(_BYTE *)(v3 + 871) = v204;
                  }
                  break;
                case 0x95u:
                  v19 = sub_EBE070(v3, v239);
                  break;
                case 0x96u:
                  v19 = sub_EAFF30((_QWORD *)v3, (__int64)s1[0], (__int64)s1[1]);
                  break;
                case 0x97u:
                case 0x98u:
                  v19 = sub_EB0040(v3, (__int64)s1[0], (__int64)s1[1]);
                  break;
                case 0x99u:
                  v19 = sub_EB9B50(v3, v239);
                  break;
                case 0x9Au:
                  v19 = sub_EA25E0(v3, 1);
                  break;
                case 0x9Bu:
                  v19 = sub_EA25E0(v3, 0);
                  break;
                case 0x9Cu:
                  v19 = sub_EB5B20(v3, v239, 0);
                  break;
                case 0x9Du:
                  v19 = sub_EB5B20(v3, v239, 1);
                  break;
                case 0x9Eu:
                  v19 = sub_EB5C40(v3, v239);
                  break;
                case 0x9Fu:
                  v19 = sub_EB0CE0(v3, v239);
                  break;
                case 0xA0u:
                  if ( (unsigned __int8)sub_ECE000(v3) )
                    goto LABEL_240;
                  v199 = *(void (**)(void))(**(_QWORD **)(v3 + 232) + 1208LL);
                  if ( v199 != nullsub_113 )
                  {
                    v199();
                    v19 = 0;
                  }
                  break;
                case 0xA1u:
                  v19 = sub_EB9E70(v3);
                  break;
                case 0xA2u:
                  v19 = sub_EB9F40(v3);
                  break;
                case 0xA3u:
                  *(_QWORD *)s2 = v3;
                  v200 = *(_QWORD *)(v3 + 832);
                  *(_DWORD *)(v3 + 776) = 0;
                  while ( v200 )
                  {
                    sub_EA2BF0(*(_QWORD *)(v200 + 24));
                    v201 = *(_QWORD *)(v200 + 16);
                    j_j___libc_free_0(v200, 48);
                    v200 = v201;
                  }
                  *(_QWORD *)(v3 + 840) = v3 + 824;
                  *(_QWORD *)(v3 + 832) = 0;
                  *(_QWORD *)(v3 + 848) = v3 + 824;
                  *(_QWORD *)(v3 + 856) = 0;
                  v19 = sub_ECE300(v3, sub_EBA420, s2, 1);
                  break;
                case 0xA4u:
                  v19 = sub_EBB3E0(v3, 3);
                  break;
                case 0xA6u:
                  v19 = sub_EA2510(v3, 29);
                  break;
                case 0xA7u:
                  if ( (unsigned __int8)sub_ECE000(v3) )
                    goto LABEL_240;
                  while ( 1 )
                  {
                    v179 = *(_DWORD **)(v3 + 48);
                    if ( !*v179 )
                      break;
                    v180 = *(unsigned int *)(v3 + 56);
                    v181 = v179 + 10;
                    *(_BYTE *)(v3 + 155) = *v179 == 9;
                    v180 *= 40LL;
                    v182 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v180 - 40) >> 3);
                    if ( v180 > 0x28 )
                    {
                      do
                      {
                        v183 = _mm_loadu_si128((const __m128i *)(v181 + 2));
                        v151 = *(v181 - 2) <= 0x40u;
                        *(v181 - 10) = *v181;
                        *((__m128i *)v181 - 2) = v183;
                        if ( !v151 )
                        {
                          v184 = *((_QWORD *)v181 - 2);
                          if ( v184 )
                            j_j___libc_free_0_0(v184);
                        }
                        v185 = *((_QWORD *)v181 + 3);
                        v181 += 10;
                        *((_QWORD *)v181 - 7) = v185;
                        LODWORD(v185) = *(v181 - 2);
                        *(v181 - 2) = 0;
                        *(v181 - 12) = v185;
                        --v182;
                      }
                      while ( v182 );
                      v179 = *(_DWORD **)(v3 + 48);
                    }
                    v186 = (unsigned int)(*(_DWORD *)(v3 + 56) - 1);
                    *(_DWORD *)(v3 + 56) = v186;
                    v187 = &v179[10 * v186];
                    if ( v187[8] > 0x40u )
                    {
                      v188 = *((_QWORD *)v187 + 3);
                      if ( v188 )
                        j_j___libc_free_0_0(v188);
                    }
                    if ( !*(_DWORD *)(v3 + 56) )
                    {
                      sub_1097F60(s2, v3 + 40);
                      sub_EAA0A0(v3 + 48, *(_QWORD *)(v3 + 48), (unsigned __int64)s2, v189, v190, v191);
                      if ( (unsigned int)v261 > 0x40 )
                      {
                        if ( v260 )
                          j_j___libc_free_0_0(v260);
                      }
                    }
                  }
                  break;
                default:
                  *(_QWORD *)s2 = "unknown directive";
                  LOWORD(v261) = 259;
                  v19 = sub_ECDA70(v3, v239, s2, 0, 0);
                  break;
              }
            }
            else
            {
              v19 = v79(v78, s1[0], s1[1], v239, v227);
            }
          }
        }
        goto LABEL_62;
      }
    }
    if ( !*(_BYTE *)(v3 + 869) )
      goto LABEL_58;
    goto LABEL_50;
  }
  if ( !*(_QWORD *)(sub_ECD7B0(a1) + 16)
    || **(_BYTE **)(sub_ECD7B0(a1) + 8) == 13
    || **(_BYTE **)(sub_ECD7B0(a1) + 8) == 10 )
  {
    v18 = *(void (**)(void))(**(_QWORD **)(a1 + 232) + 160LL);
    if ( v18 != nullsub_99 )
      v18();
  }
  v19 = 0;
  sub_EABFE0(a1);
  return v19;
}
