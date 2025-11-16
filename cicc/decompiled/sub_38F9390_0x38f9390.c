// Function: sub_38F9390
// Address: 0x38f9390
//
__int64 __fastcall sub_38F9390(__int64 a1, __int64 a2, __int64 a3, __m128 a4, double a5, double a6)
{
  __int64 v6; // r13
  int i; // eax
  __int64 v9; // rax
  __m128 v10; // xmm1
  unsigned int v11; // edx
  __int64 v12; // rdx
  unsigned int v13; // ecx
  int v14; // eax
  __int64 (*v15)(); // rax
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // ebx
  void (*v24)(void); // rax
  unsigned int v25; // r15d
  int v27; // eax
  __int64 (*v28)(); // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // r12
  __int64 v33; // r15
  unsigned __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // r10
  __int64 (__fastcall *v39)(__int64, __int64, __int64, __int64, __int64, __int64); // rbx
  void *v40; // r15
  __int64 (__fastcall *v41)(__int64, __int64 *, void *, void *, __int64, __int64); // rbx
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 (*v44)(); // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  int v47; // r9d
  __int64 v48; // rbx
  __int32 v49; // r12d
  __int64 v50; // r14
  __int64 v51; // r8
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdi
  _BYTE *v56; // r12
  __int64 v57; // rdi
  void (*v58)(); // rax
  char *v59; // rsi
  bool v60; // zf
  unsigned __int64 v61; // rax
  __int32 v62; // r14d
  unsigned int v63; // eax
  int v64; // r8d
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r14
  int v71; // eax
  __int64 v72; // rdx
  _QWORD *v73; // rax
  _QWORD *v74; // r12
  __int64 v75; // rbx
  size_t v76; // r15
  unsigned __int8 *v77; // r14
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // r14
  __int64 v83; // rax
  int v84; // edx
  _BYTE *v85; // rsi
  void *v86; // rax
  __int64 *v87; // r14
  __int64 v88; // rax
  __int64 v89; // rax
  unsigned __int64 v90; // r14
  _QWORD *v91; // rdx
  __int64 v92; // rax
  unsigned __int64 *v93; // r13
  unsigned __int64 *v94; // r14
  unsigned __int64 v95; // rbx
  unsigned __int64 v96; // r12
  unsigned __int64 v97; // rdi
  __int64 v98; // rdi
  void (*v99)(); // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // r8
  __int64 (__fastcall *v103)(__int64, __m128i **); // r12
  unsigned int v104; // r12d
  __int64 v105; // rax
  __m128i *v106; // rax
  __m128i si128; // xmm0
  __int64 v108; // r8
  int v109; // r15d
  __int64 *v110; // rdi
  __int64 v111; // rdx
  __m128i *v112; // rax
  __int64 v113; // rax
  __int64 v114; // rbx
  unsigned int v115; // eax
  int v116; // r9d
  unsigned __int64 v117; // r8
  __int64 v118; // rbx
  __int64 v119; // rax
  unsigned __int64 v120; // rdx
  __int64 v121; // r8
  __int64 v122; // rbx
  unsigned int v123; // edx
  __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rsi
  int v128; // eax
  __int64 v129; // rdx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rdx
  __int64 *v133; // rax
  __int64 v134; // rax
  __int64 v135; // rdi
  __int64 (__fastcall *v136)(__int64, __int64, __int64, __int64); // rax
  __int64 *v137; // r15
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 v140; // rbx
  unsigned int v141; // edx
  __int64 v142; // rax
  __int64 v143; // rbx
  __int64 v144; // rax
  _DWORD *v145; // rcx
  unsigned __int64 v146; // rax
  __int64 v147; // rdx
  unsigned __int64 v148; // r14
  __int64 v149; // r13
  _DWORD *v150; // rbx
  __m128i v151; // xmm3
  bool v152; // cc
  unsigned __int64 v153; // rdi
  __int64 v154; // rax
  __int64 v155; // rax
  _DWORD *v156; // rax
  unsigned __int64 v157; // rdi
  unsigned __int64 v158; // rsi
  __int64 v159; // rdx
  __int64 v160; // rsi
  __int64 v161; // rax
  int v162; // eax
  __int64 v163; // r9
  int v164; // edi
  int v165; // ecx
  unsigned int v166; // eax
  __int64 v167; // rdx
  unsigned __int64 **v168; // rax
  __int64 *v169; // rdi
  unsigned int v170; // r15d
  __int64 v171; // rcx
  __int64 v172; // rsi
  void (__fastcall *v173)(__m128i *, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __m128i **, _QWORD); // rax
  _DWORD *v174; // rbx
  __int64 v175; // r12
  _DWORD *v176; // rbx
  signed __int64 v177; // r12
  __m128i v178; // xmm7
  unsigned __int64 v179; // rdi
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // rax
  unsigned __int64 v183; // rdi
  unsigned __int8 v184; // al
  __int64 v185; // rdi
  __int64 v186; // rax
  __int64 v187; // rdi
  __int64 v188; // rsi
  void (*v189)(); // rax
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // r8
  __int64 v193; // r9
  __int64 v194; // rdx
  __int64 v195; // rcx
  __int64 v196; // r8
  _QWORD *v197; // rax
  __int64 v198; // r13
  void *v199; // rdi
  unsigned __int64 v200; // rax
  size_t v201; // r12
  void *v202; // rsi
  _BYTE *v203; // rax
  __int64 v204; // rcx
  unsigned __int64 v205; // rax
  __int64 v206; // rsi
  unsigned __int64 v207; // rdx
  _DWORD *v208; // rax
  __m128i v209; // xmm5
  bool v210; // dl
  __int64 v211; // rdi
  void (*v212)(); // rax
  __int64 v213; // r8
  __int64 v214; // r9
  void *v215; // rax
  void *v216; // rax
  void *v217; // rax
  void *v218; // rax
  unsigned int v219; // eax
  __int64 v220; // [rsp+8h] [rbp-2A8h]
  __int64 v221; // [rsp+8h] [rbp-2A8h]
  __int64 v222; // [rsp+8h] [rbp-2A8h]
  __int64 v223; // [rsp+10h] [rbp-2A0h]
  void *v224; // [rsp+10h] [rbp-2A0h]
  __int64 v225; // [rsp+10h] [rbp-2A0h]
  __int64 v226; // [rsp+10h] [rbp-2A0h]
  __int64 v227; // [rsp+10h] [rbp-2A0h]
  unsigned __int64 v228; // [rsp+10h] [rbp-2A0h]
  __int64 v230; // [rsp+18h] [rbp-298h]
  unsigned __int8 v231; // [rsp+18h] [rbp-298h]
  __int64 v232; // [rsp+18h] [rbp-298h]
  __int64 v233; // [rsp+18h] [rbp-298h]
  __int64 v234; // [rsp+18h] [rbp-298h]
  __int64 v235; // [rsp+18h] [rbp-298h]
  __int64 v236; // [rsp+18h] [rbp-298h]
  __int64 v237; // [rsp+18h] [rbp-298h]
  __int64 v238; // [rsp+18h] [rbp-298h]
  __int64 v239; // [rsp+20h] [rbp-290h] BYREF
  __int64 v240; // [rsp+28h] [rbp-288h] BYREF
  __m128i v241; // [rsp+30h] [rbp-280h] BYREF
  _QWORD v242[2]; // [rsp+40h] [rbp-270h] BYREF
  unsigned __int64 v243[2]; // [rsp+50h] [rbp-260h] BYREF
  unsigned __int64 *v244; // [rsp+60h] [rbp-250h] BYREF
  unsigned __int64 *v245; // [rsp+68h] [rbp-248h]
  __int64 v246; // [rsp+70h] [rbp-240h]
  void *s1[2]; // [rsp+80h] [rbp-230h] BYREF
  __int64 v248; // [rsp+90h] [rbp-220h] BYREF
  int v249; // [rsp+A0h] [rbp-210h] BYREF
  __m128i v250; // [rsp+A8h] [rbp-208h] BYREF
  unsigned __int64 v251; // [rsp+B8h] [rbp-1F8h] BYREF
  unsigned int v252; // [rsp+C0h] [rbp-1F0h]
  __m128i v253; // [rsp+D0h] [rbp-1E0h] BYREF
  __m128i *v254; // [rsp+E0h] [rbp-1D0h] BYREF
  __m128i *v255; // [rsp+E8h] [rbp-1C8h]
  int v256; // [rsp+F0h] [rbp-1C0h]
  __m128i **v257; // [rsp+F8h] [rbp-1B8h]
  __m128i *v258; // [rsp+100h] [rbp-1B0h] BYREF
  __m128i v259; // [rsp+108h] [rbp-1A8h] BYREF
  unsigned __int64 v260; // [rsp+118h] [rbp-198h] BYREF
  unsigned __int64 v261; // [rsp+120h] [rbp-190h]
  _BYTE *v262; // [rsp+128h] [rbp-188h]
  unsigned __int64 v263; // [rsp+130h] [rbp-180h]
  __int64 v264; // [rsp+138h] [rbp-178h]
  volatile signed __int32 *v265; // [rsp+140h] [rbp-170h] BYREF
  int v266; // [rsp+148h] [rbp-168h]
  unsigned __int64 v267[2]; // [rsp+150h] [rbp-160h] BYREF
  _BYTE v268[16]; // [rsp+160h] [rbp-150h] BYREF
  _QWORD v269[28]; // [rsp+170h] [rbp-140h] BYREF
  __int16 v270; // [rsp+250h] [rbp-60h]
  __int64 v271; // [rsp+258h] [rbp-58h]
  __int64 v272; // [rsp+260h] [rbp-50h]
  __int64 v273; // [rsp+268h] [rbp-48h]
  __int64 v274; // [rsp+270h] [rbp-40h]

  v6 = a1;
  for ( i = **(_DWORD **)(a1 + 152); i == 11; i = **(_DWORD **)(a1 + 152) )
    sub_38EB180(a1);
  if ( i != 9 )
  {
    v9 = sub_3909460(a1);
    v10 = (__m128)_mm_loadu_si128((const __m128i *)(v9 + 8));
    v249 = *(_DWORD *)v9;
    v11 = *(_DWORD *)(v9 + 32);
    v250 = (__m128i)v10;
    v252 = v11;
    if ( v11 > 0x40 )
      sub_16A4FD0((__int64)&v251, (const void **)(v9 + 24));
    else
      v251 = *(_QWORD *)(v9 + 24);
    v241 = 0u;
    v239 = sub_39092A0(&v249);
    v14 = **(_DWORD **)(a1 + 152);
    if ( v14 == 8 )
    {
      sub_38EB180(a1);
      v29 = sub_3909460(a1);
      if ( *(_DWORD *)(v29 + 32) <= 0x40u )
        v30 = *(_QWORD *)(v29 + 24);
      else
        v30 = **(_QWORD **)(v29 + 24);
      sub_38EB180(a1);
      v31 = sub_3909460(a1);
      v32 = *(_QWORD *)(v31 + 16);
      v33 = *(_QWORD *)(v31 + 8);
      sub_38EB180(a1);
      if ( v32 )
      {
        v34 = v32 - 2;
        if ( v34 <= --v32 )
          v32 = v34;
        ++v33;
      }
      v35 = *(_DWORD *)(a1 + 376);
      *(_QWORD *)(a1 + 584) = v239;
      *(_QWORD *)(a1 + 568) = v32;
      *(_QWORD *)(a1 + 576) = v30;
      *(_DWORD *)(a1 + 592) = v35;
      *(_QWORD *)(a1 + 560) = v33;
      v25 = 0;
      goto LABEL_43;
    }
    if ( v14 != 4 )
    {
      switch ( v14 )
      {
        case 24:
          sub_38EB180(a1);
          v241.m128i_i64[1] = 1;
          v241.m128i_i64[0] = (__int64)".";
          v223 = -1;
          break;
        case 21:
          sub_38EB180(a1);
          v241.m128i_i64[1] = 1;
          v241.m128i_i64[0] = (__int64)"{";
          v223 = -1;
          break;
        case 22:
          sub_38EB180(a1);
          v241.m128i_i64[1] = 1;
          v241.m128i_i64[0] = (__int64)"}";
          v223 = -1;
          break;
        default:
          if ( v14 == 23
            && (v15 = *(__int64 (**)())(**(_QWORD **)(a1 + 8) + 144LL), v15 != sub_38E2A00)
            && (unsigned __int8)v15() )
          {
            sub_38EB180(a1);
            v241.m128i_i64[1] = 1;
            v241.m128i_i64[0] = (__int64)"*";
            v223 = -1;
          }
          else
          {
            if ( (unsigned __int8)sub_38F0EE0(a1, v241.m128i_i64, v12, v13) )
            {
              if ( !*(_BYTE *)(a1 + 385) )
              {
                sub_38EB180(a1);
                v258 = (__m128i *)"unexpected token at start of statement";
                v259.m128i_i16[4] = 259;
                v25 = sub_3909790(a1, v239, &v258, 0, 0);
                goto LABEL_43;
              }
              v241.m128i_i64[1] = 0;
              v241.m128i_i64[0] = (__int64)byte_3F871B3;
            }
            v223 = -1;
          }
          break;
      }
      goto LABEL_15;
    }
    v36 = sub_3909460(a1);
    if ( *(_DWORD *)(v36 + 32) <= 0x40u )
      v37 = *(_QWORD *)(v36 + 24);
    else
      v37 = **(_QWORD **)(v36 + 24);
    v223 = v37;
    if ( v37 >= 0 )
    {
      v241 = *(__m128i *)(sub_3909460(a1) + 8);
      sub_38EB180(a1);
      if ( **(_DWORD **)(a1 + 152) == 10 || *(_BYTE *)(a1 + 385) )
      {
LABEL_15:
        v16 = sub_16D1B30((__int64 *)(a1 + 848), (unsigned __int8 *)v241.m128i_i64[0], v241.m128i_u64[1]);
        if ( v16 != -1 )
        {
          v21 = *(_QWORD *)(a1 + 848);
          v18 = *(unsigned int *)(a1 + 856);
          v22 = v21 + 8LL * v16;
          v17 = v21 + 8 * v18;
          if ( v22 != v17 )
          {
            v23 = *(_DWORD *)(*(_QWORD *)v22 + 8LL);
            switch ( v23 )
            {
              case 'Q':
              case 'R':
              case 'S':
              case 'T':
              case 'U':
              case 'V':
              case 'W':
                v59 = *(char **)(a1 + 400);
                if ( v59 == *(char **)(a1 + 408) )
                {
                  sub_38E9AD0((unsigned __int64 *)(a1 + 392), v59, (_QWORD *)(a1 + 380));
                }
                else
                {
                  if ( v59 )
                  {
                    *(_QWORD *)v59 = *(_QWORD *)(a1 + 380);
                    v59 = *(char **)(a1 + 400);
                  }
                  *(_QWORD *)(a1 + 400) = v59 + 8;
                }
                v60 = *(_BYTE *)(a1 + 385) == 0;
                *(_DWORD *)(a1 + 380) = 1;
                if ( !v60 )
                  goto LABEL_69;
                if ( (unsigned __int8)sub_38EB9C0(a1, &v253)
                  || (v258 = (__m128i *)"unexpected token in '.if' directive",
                      v259.m128i_i16[4] = 259,
                      v25 = sub_3909E20(a1, 9, &v258),
                      (_BYTE)v25) )
                {
                  v25 = 1;
                }
                else
                {
                  v61 = v253.m128i_i64[0];
                  switch ( v23 )
                  {
                    case 'R':
                      v61 = v253.m128i_i64[0] == 0;
                      v253.m128i_i64[0] = v61;
                      break;
                    case 'S':
                      v61 = v253.m128i_i64[0] >= 0;
                      v253.m128i_i64[0] = v61;
                      break;
                    case 'T':
                      v61 = v253.m128i_i64[0] > 0;
                      v253.m128i_i64[0] = v61;
                      break;
                    case 'U':
                      v61 = v253.m128i_i64[0] <= 0;
                      v253.m128i_i64[0] = v61;
                      break;
                    case 'V':
                      v61 = (unsigned __int64)v253.m128i_i64[0] >> 63;
                      v253.m128i_i64[0] = (unsigned __int64)v253.m128i_i64[0] >> 63;
                      break;
                    default:
                      break;
                  }
                  *(_BYTE *)(a1 + 384) = v61 != 0;
                  *(_BYTE *)(a1 + 385) = v61 == 0;
                }
                break;
              case 'X':
                v25 = sub_38F09C0(a1, 1);
                break;
              case 'Y':
                v25 = sub_38F09C0(a1, 0);
                break;
              case 'Z':
                v25 = sub_38F0AB0(a1, 1);
                break;
              case '[':
                v25 = sub_38EF650(a1, 1, v17, v18, v19, v20);
                break;
              case '\\':
                v25 = sub_38F0AB0(a1, 0);
                break;
              case ']':
                v25 = sub_38EF650(a1, 0, v17, v18, v19, v20);
                break;
              case '^':
                v25 = sub_38F4A90(a1, 1, v17, v18);
                break;
              case '_':
              case '`':
                v25 = sub_38F4A90(a1, 0, v17, v18);
                break;
              case 'a':
                if ( (unsigned int)(*(_DWORD *)(a1 + 380) - 1) > 1 )
                {
                  v258 = (__m128i *)"Encountered a .elseif that doesn't follow an .if or  an .elseif";
                  v259.m128i_i16[4] = 259;
                  v25 = sub_3909790(a1, v239, &v258, 0, 0);
                }
                else
                {
                  *(_DWORD *)(a1 + 380) = 2;
                  v67 = *(_QWORD *)(a1 + 400);
                  if ( v67 != *(_QWORD *)(a1 + 392) && *(_BYTE *)(v67 - 3) || *(_BYTE *)(a1 + 384) )
                  {
                    *(_BYTE *)(a1 + 385) = 1;
                    v25 = 0;
                    sub_38F0630(a1);
                  }
                  else if ( (unsigned __int8)sub_38EB9C0(a1, &v253)
                         || (v258 = (__m128i *)"unexpected token in '.elseif' directive",
                             v259.m128i_i16[4] = 259,
                             v25 = sub_3909E20(a1, 9, &v258),
                             (_BYTE)v25) )
                  {
                    v25 = 1;
                  }
                  else
                  {
                    v60 = v253.m128i_i64[0] == 0;
                    *(_BYTE *)(a1 + 384) = v253.m128i_i64[0] != 0;
                    *(_BYTE *)(a1 + 385) = v60;
                  }
                }
                break;
              case 'b':
                v258 = (__m128i *)"unexpected token in '.else' directive";
                v259.m128i_i16[4] = 259;
                v25 = sub_3909E20(a1, 9, &v258);
                if ( !(_BYTE)v25 )
                {
                  if ( (unsigned int)(*(_DWORD *)(a1 + 380) - 1) > 1 )
                  {
                    v258 = (__m128i *)"Encountered a .else that doesn't follow  an .if or an .elseif";
                    v259.m128i_i16[4] = 259;
                    v25 = sub_3909790(a1, v239, &v258, 0, 0);
                  }
                  else
                  {
                    *(_DWORD *)(a1 + 380) = 3;
                    v66 = *(_QWORD *)(a1 + 400);
                    if ( v66 != *(_QWORD *)(a1 + 392) && *(_BYTE *)(v66 - 3) || *(_BYTE *)(a1 + 384) )
                    {
                      *(_BYTE *)(a1 + 385) = 1;
                    }
                    else
                    {
                      *(_BYTE *)(a1 + 385) = 0;
                      v25 = 0;
                    }
                  }
                }
                break;
              case 'c':
                v258 = (__m128i *)"unexpected token in '.endif' directive";
                v259.m128i_i16[4] = 259;
                v25 = sub_3909E20(a1, 9, &v258);
                if ( !(_BYTE)v25 )
                {
                  if ( !*(_DWORD *)(a1 + 380) || (v68 = *(_QWORD *)(a1 + 400), v68 == *(_QWORD *)(a1 + 392)) )
                  {
                    v258 = (__m128i *)"Encountered a .endif that doesn't follow an .if or .else";
                    v259.m128i_i16[4] = 259;
                    v25 = sub_3909790(a1, v239, &v258, 0, 0);
                  }
                  else
                  {
                    v69 = *(_QWORD *)(v68 - 8);
                    *(_QWORD *)(a1 + 400) = v68 - 8;
                    *(_QWORD *)(a1 + 380) = v69;
                  }
                }
                break;
              default:
                goto LABEL_31;
            }
            goto LABEL_43;
          }
        }
        v23 = 0;
LABEL_31:
        v25 = *(unsigned __int8 *)(a1 + 385);
        if ( (_BYTE)v25 )
        {
LABEL_69:
          v25 = 0;
          sub_38F0630(a1);
        }
        else
        {
          v27 = **(_DWORD **)(a1 + 152);
          if ( v27 == 10 )
          {
            v43 = *(_QWORD *)(a1 + 8);
            v44 = *(__int64 (**)())(*(_QWORD *)v43 + 136LL);
            if ( v44 == sub_38E29F0 || ((unsigned __int8 (__fastcall *)(__int64, int *))v44)(v43, &v249) )
            {
              if ( *(_BYTE *)(v6 + 845) || !(unsigned __int8)sub_38E36C0(v6) )
              {
                sub_38EB180(v6);
                if ( v241.m128i_i64[1] == 1 && *(_BYTE *)v241.m128i_i64[0] == 46 )
                {
                  v258 = (__m128i *)"invalid use of pseudo-symbol '.' as a label";
                  v259.m128i_i16[4] = 259;
                  v25 = sub_3909790(v6, v239, &v258, 0, 0);
                }
                else
                {
                  if ( v223 == -1 )
                  {
                    if ( *(_BYTE *)(v6 + 845) && a3 )
                    {
                      v45 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)a3 + 24LL))(
                              a3,
                              v241.m128i_i64[0],
                              v241.m128i_i64[1],
                              *(_QWORD *)(v6 + 344),
                              v239,
                              1);
                      v48 = *(_QWORD *)(a2 + 88);
                      v49 = v241.m128i_i32[2];
                      v50 = v45;
                      v51 = v46;
                      v52 = *(unsigned int *)(v48 + 8);
                      if ( (unsigned int)v52 >= *(_DWORD *)(v48 + 12) )
                      {
                        v238 = v46;
                        sub_16CD150(v48, (const void *)(v48 + 16), 0, 104, v46, v47);
                        v52 = *(unsigned int *)(v48 + 8);
                        v51 = v238;
                      }
                      v53 = *(_QWORD *)v48 + 104 * v52;
                      if ( v53 )
                      {
                        v54 = v239;
                        *(_DWORD *)(v53 + 16) = v49;
                        *(_DWORD *)v53 = 6;
                        *(_QWORD *)(v53 + 8) = v54;
                        *(_QWORD *)(v53 + 24) = 0;
                        *(_BYTE *)(v53 + 48) = 0;
                        *(_QWORD *)(v53 + 56) = 0;
                        *(_QWORD *)(v53 + 64) = 0;
                        *(_QWORD *)(v53 + 72) = 0;
                        *(_QWORD *)(v53 + 80) = 0;
                        *(_QWORD *)(v53 + 88) = 0;
                        *(_DWORD *)(v53 + 96) = 1;
                        *(_QWORD *)(v53 + 32) = v50;
                        *(_QWORD *)(v53 + 40) = v51;
                      }
                      ++*(_DWORD *)(v48 + 8);
                      v241.m128i_i64[0] = v50;
                      v241.m128i_i64[1] = v51;
                    }
                    v55 = *(_QWORD *)(v6 + 320);
                    v259.m128i_i16[4] = 261;
                    v258 = &v241;
                    v56 = (_BYTE *)sub_38BF510(v55, (__int64)&v258);
                  }
                  else
                  {
                    v56 = (_BYTE *)sub_38C4ED0(*(_QWORD *)(v6 + 320), v223);
                  }
                  if ( *(_DWORD *)sub_3909460(v6) == 37 )
                  {
                    v143 = v6 + 152;
                    v144 = sub_38EAF10(v6);
                    v145 = *(_DWORD **)(v6 + 152);
                    v237 = v144;
                    v146 = *(unsigned int *)(v6 + 160);
                    v227 = v147;
                    *(_BYTE *)(v6 + 258) = *v145 == 9;
                    v146 *= 40LL;
                    v148 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v146 - 40) >> 3);
                    if ( v146 > 0x28 )
                    {
                      v222 = v6;
                      v149 = v6 + 152;
                      v150 = v145;
                      do
                      {
                        v151 = _mm_loadu_si128((const __m128i *)v150 + 3);
                        v152 = v150[8] <= 0x40u;
                        *v150 = v150[10];
                        *(__m128i *)(v150 + 2) = v151;
                        if ( !v152 )
                        {
                          v153 = *((_QWORD *)v150 + 3);
                          if ( v153 )
                            j_j___libc_free_0_0(v153);
                        }
                        v154 = *((_QWORD *)v150 + 8);
                        v150 += 10;
                        *((_QWORD *)v150 - 2) = v154;
                        LODWORD(v154) = v150[8];
                        v150[8] = 0;
                        *(v150 - 2) = v154;
                        --v148;
                      }
                      while ( v148 );
                      v143 = v149;
                      v6 = v222;
                      v145 = *(_DWORD **)(v222 + 152);
                    }
                    v155 = (unsigned int)(*(_DWORD *)(v6 + 160) - 1);
                    *(_DWORD *)(v6 + 160) = v155;
                    v156 = &v145[10 * v155];
                    if ( v156[8] > 0x40u )
                    {
                      v157 = *((_QWORD *)v156 + 3);
                      if ( v157 )
                        j_j___libc_free_0_0(v157);
                    }
                    if ( !*(_DWORD *)(v6 + 160) )
                    {
                      sub_392C2E0(&v258, v6 + 144);
                      sub_38E90E0(v143, *(_QWORD *)(v6 + 152), (unsigned __int64)&v258);
                      if ( (unsigned int)v261 > 0x40 )
                      {
                        if ( v260 )
                          j_j___libc_free_0_0(v260);
                      }
                    }
                    v158 = *(_QWORD *)(v6 + 152);
                    *(_BYTE *)(v6 + 258) = 0;
                    v259.m128i_i64[0] = v237;
                    LODWORD(v258) = 9;
                    v259.m128i_i64[1] = v227;
                    LODWORD(v261) = 64;
                    v260 = 0;
                    sub_38E90E0(v143, v158, (unsigned __int64)&v258);
                    if ( (unsigned int)v261 > 0x40 && v260 )
                      j_j___libc_free_0_0(v260);
                  }
                  if ( *(_DWORD *)sub_3909460(v6) == 9 )
                    sub_38EB180(v6);
                  if ( !*(_BYTE *)(*(_QWORD *)(v6 + 8) + 32LL) )
                    (*(void (__fastcall **)(_QWORD, _BYTE *, __int64))(**(_QWORD **)(v6 + 328) + 176LL))(
                      *(_QWORD *)(v6 + 328),
                      v56,
                      v239);
                  if ( (unsigned __int8)sub_38E9F70(v6) )
                    sub_38C8610(v56, *(_QWORD *)(v6 + 328), *(__int64 **)(v6 + 344), (unsigned __int64 *)&v239);
                  v57 = *(_QWORD *)(v6 + 8);
                  v58 = *(void (**)())(*(_QWORD *)v57 + 160LL);
                  if ( v58 != nullsub_1960 )
                    ((void (__fastcall *)(__int64, _BYTE *))v58)(v57, v56);
                }
                goto LABEL_43;
              }
LABEL_222:
              v25 = 1;
              goto LABEL_43;
            }
LABEL_52:
            if ( (*(_BYTE *)(v6 + 552) & 1) != 0 )
            {
              v70 = *(_QWORD *)(v6 + 320);
              v71 = sub_16D1B30((__int64 *)(v70 + 1488), (unsigned __int8 *)v241.m128i_i64[0], v241.m128i_u64[1]);
              if ( v71 != -1 )
              {
                v72 = *(_QWORD *)(v70 + 1488);
                v73 = (_QWORD *)(v72 + 8LL * v71);
                v17 = v72 + 8LL * *(unsigned int *)(v70 + 1496);
                if ( v73 != (_QWORD *)v17 )
                {
                  v74 = (_QWORD *)*v73;
                  v75 = (unsigned int)dword_5052CA0;
                  if ( dword_5052CA0 == (__int64)(*(_QWORD *)(v6 + 456) - *(_QWORD *)(v6 + 448)) >> 3 )
                  {
                    sub_222DF20((__int64)v269);
                    v269[27] = 0;
                    v271 = 0;
                    v272 = 0;
                    v269[0] = off_4A06798;
                    v270 = 0;
                    v273 = 0;
                    v274 = 0;
                    v258 = (__m128i *)qword_4A071C8;
                    *(__m128i **)((char *)&v258 + qword_4A071C8[-3]) = (__m128i *)&unk_4A071F0;
                    sub_222DD70((__int64)&v258 + v258[-2].m128i_i64[1], 0);
                    v259.m128i_i64[1] = 0;
                    v260 = 0;
                    v261 = 0;
                    v258 = (__m128i *)off_4A07238;
                    v262 = 0;
                    v263 = 0;
                    v269[0] = off_4A07260;
                    v264 = 0;
                    v259.m128i_i64[0] = (__int64)off_4A07480;
                    sub_220A990(&v265);
                    v266 = 16;
                    v268[0] = 0;
                    v259.m128i_i64[0] = (__int64)off_4A07080;
                    v267[0] = (unsigned __int64)v268;
                    v267[1] = 0;
                    sub_222DD70((__int64)v269, (__int64)&v259);
                    sub_223E0D0((__int64 *)&v258, "macros cannot be nested more than ", 34);
                    v137 = sub_223E760((__int64 *)&v258, v75);
                    sub_223E0D0(v137, " levels deep.", 13);
                    sub_223E0D0(v137, " Use -asm-macro-max-nesting-depth to increase this limit.", 57);
                    v253.m128i_i64[1] = 0;
                    v253.m128i_i64[0] = (__int64)&v254;
                    LOBYTE(v254) = 0;
                    if ( v263 )
                    {
                      if ( v263 <= v261 )
                        sub_2241130((unsigned __int64 *)&v253, 0, 0, v262, v261 - (_QWORD)v262);
                      else
                        sub_2241130((unsigned __int64 *)&v253, 0, 0, v262, v263 - (_QWORD)v262);
                    }
                    else
                    {
                      sub_2240AE0((unsigned __int64 *)&v253, v267);
                    }
                    LOWORD(v248) = 260;
                    s1[0] = &v253;
                    v25 = sub_3909CF0(v6, s1, 0, 0, v138, v139);
                    if ( (__m128i **)v253.m128i_i64[0] != &v254 )
                      j_j___libc_free_0(v253.m128i_u64[0]);
                    v258 = (__m128i *)off_4A07238;
                    v269[0] = off_4A07260;
                    v259.m128i_i64[0] = (__int64)off_4A07080;
                    if ( (_BYTE *)v267[0] != v268 )
                      j_j___libc_free_0(v267[0]);
                    v259.m128i_i64[0] = (__int64)off_4A07480;
                    sub_2209150(&v265);
                    v258 = (__m128i *)qword_4A071C8;
                    *(__m128i **)((char *)&v258 + qword_4A071C8[-3]) = (__m128i *)&unk_4A071F0;
                    v269[0] = off_4A06798;
                    sub_222E050((__int64)v269);
                  }
                  else
                  {
                    v244 = 0;
                    v233 = v239;
                    v245 = 0;
                    v246 = 0;
                    v25 = sub_38F6810(v6, (__int64)(v74 + 1), (__int64 *)&v244, v18, v19, v20);
                    if ( !(_BYTE)v25 )
                    {
                      v258 = (__m128i *)&v259.m128i_u64[1];
                      v259.m128i_i64[0] = 0x10000000000LL;
                      v76 = v74[4];
                      v77 = (unsigned __int8 *)v74[3];
                      v253.m128i_i64[0] = (__int64)&unk_49EFC48;
                      v257 = &v258;
                      v256 = 1;
                      v255 = 0;
                      v254 = 0;
                      v253.m128i_i64[1] = 0;
                      sub_16E7A40((__int64)&v253, 0, 0, 0);
                      v78 = sub_3909460(v6);
                      v79 = sub_39092A0(v78);
                      v25 = sub_38E48B0(
                              v6,
                              (__int64)&v253,
                              v77,
                              v76,
                              v74[5],
                              0xAAAAAAAAAAAAAAABLL * ((__int64)(v74[6] - v74[5]) >> 4),
                              (__int64)v244,
                              0xAAAAAAAAAAAAAAABLL * (v245 - v244),
                              1,
                              v79);
                      if ( !(_BYTE)v25 )
                      {
                        v80 = (__int64 *)v255;
                        if ( (unsigned __int64)((char *)v254 - (char *)v255) <= 9 )
                        {
                          sub_16E7EE0((__int64)&v253, ".endmacro\n", 0xAu);
                        }
                        else
                        {
                          v255->m128i_i16[4] = 2671;
                          *v80 = 0x7263616D646E652ELL;
                          v255 = (__m128i *)((char *)v255 + 10);
                        }
                        s1[0] = "<instantiation>";
                        LOWORD(v248) = 259;
                        sub_16C28C0(v242, *v257, *((unsigned int *)v257 + 2), (__int64)s1);
                        v81 = sub_3909460(v6);
                        v82 = sub_39092A0(v81);
                        v220 = (__int64)(*(_QWORD *)(v6 + 400) - *(_QWORD *)(v6 + 392)) >> 3;
                        v83 = sub_22077B0(0x20u);
                        if ( v83 )
                        {
                          v84 = *(_DWORD *)(v6 + 376);
                          *(_QWORD *)(v83 + 16) = v82;
                          *(_QWORD *)v83 = v233;
                          *(_DWORD *)(v83 + 8) = v84;
                          *(_QWORD *)(v83 + 24) = v220;
                        }
                        v243[0] = v83;
                        v85 = *(_BYTE **)(v6 + 456);
                        if ( v85 == *(_BYTE **)(v6 + 464) )
                        {
                          sub_38E2F40(v6 + 448, v85, v243);
                        }
                        else
                        {
                          if ( v85 )
                            *(_QWORD *)v85 = v83;
                          *(_QWORD *)(v6 + 456) += 8LL;
                        }
                        v86 = (void *)v242[0];
                        v87 = *(__int64 **)(v6 + 344);
                        v242[0] = 0;
                        ++*(_DWORD *)(v6 + 556);
                        s1[1] = 0;
                        s1[0] = v86;
                        v248 = 0;
                        v88 = v87[1];
                        if ( v88 == v87[2] )
                        {
                          sub_168C7C0(v87, v87[1], (__int64)s1);
                          v89 = v87[1];
                        }
                        else
                        {
                          if ( v88 )
                          {
                            sub_16CE2D0((_QWORD *)v87[1], s1);
                            v88 = v87[1];
                          }
                          v89 = v88 + 24;
                          v87[1] = v89;
                        }
                        v90 = 0xAAAAAAAAAAAAAAABLL * ((v89 - *v87) >> 3);
                        sub_16CE300((__int64 *)s1);
                        v91 = *(_QWORD **)(v6 + 344);
                        *(_DWORD *)(v6 + 376) = v90;
                        v92 = *(_QWORD *)(*v91 + 24LL * (unsigned int)(v90 - 1));
                        sub_392A730(v6 + 144, *(_QWORD *)(v92 + 8), *(_QWORD *)(v92 + 16) - *(_QWORD *)(v92 + 8), 0);
                        sub_38EB180(v6);
                        if ( v242[0] )
                          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v242[0] + 8LL))(v242[0]);
                      }
                      v253.m128i_i64[0] = (__int64)&unk_49EFD28;
                      sub_16E7960((__int64)&v253);
                      if ( v258 != (__m128i *)&v259.m128i_u64[1] )
                        _libc_free((unsigned __int64)v258);
                    }
                    v93 = v245;
                    v94 = v244;
                    if ( v245 != v244 )
                    {
                      do
                      {
                        v95 = v94[1];
                        v96 = *v94;
                        if ( v95 != *v94 )
                        {
                          do
                          {
                            if ( *(_DWORD *)(v96 + 32) > 0x40u )
                            {
                              v97 = *(_QWORD *)(v96 + 24);
                              if ( v97 )
                                j_j___libc_free_0_0(v97);
                            }
                            v96 += 40LL;
                          }
                          while ( v95 != v96 );
                          v96 = *v94;
                        }
                        if ( v96 )
                          j_j___libc_free_0(v96);
                        v94 += 3;
                      }
                      while ( v93 != v94 );
                      v94 = v244;
                    }
                    if ( v94 )
                      j_j___libc_free_0((unsigned __int64)v94);
                  }
                  goto LABEL_43;
                }
              }
            }
            if ( *(_BYTE *)v241.m128i_i64[0] == 46 )
            {
              if ( v241.m128i_i64[1] != 1 )
              {
                v98 = *(_QWORD *)(v6 + 8);
                v99 = *(void (**)())(*(_QWORD *)v98 + 168LL);
                if ( v99 != nullsub_1961 )
                  ((void (__fastcall *)(__int64, _QWORD))v99)(v98, *(_QWORD *)(v6 + 328));
                v100 = sub_3909460(v6);
                v101 = sub_39092A0(v100);
                v102 = *(_QWORD *)(v6 + 8);
                v234 = v101;
                v103 = *(__int64 (__fastcall **)(__int64, __m128i **))(*(_QWORD *)v102 + 64LL);
                v259 = _mm_loadu_si128(&v250);
                LODWORD(v258) = v249;
                LODWORD(v261) = v252;
                if ( v252 > 0x40 )
                {
                  v226 = v102;
                  sub_16A4FD0((__int64)&v260, (const void **)&v251);
                  v102 = v226;
                }
                else
                {
                  v260 = v251;
                }
                v104 = v103(v102, &v258);
                if ( (unsigned int)v261 > 0x40 && v260 )
                  j_j___libc_free_0_0(v260);
                if ( *(_DWORD *)(v6 + 32) )
                {
LABEL_232:
                  v25 = 1;
                }
                else if ( (_BYTE)v104 )
                {
                  v105 = sub_3909460(v6);
                  if ( v234 == sub_39092A0(v105) )
                  {
                    v126 = sub_3909460(v6);
                    if ( v234 == sub_39092A0(v126) )
                    {
                      v127 = v241.m128i_i64[0];
                      v128 = sub_16D1B30((__int64 *)(v6 + 416), (unsigned __int8 *)v241.m128i_i64[0], v241.m128i_u64[1]);
                      if ( v128 == -1
                        || (v132 = *(_QWORD *)(v6 + 416),
                            v127 = *(unsigned int *)(v6 + 424),
                            v133 = (__int64 *)(v132 + 8LL * v128),
                            v129 = v132 + 8 * v127,
                            v133 == (__int64 *)v129)
                        || (v134 = *v133,
                            v135 = *(_QWORD *)(v134 + 8),
                            v136 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(v134 + 16),
                            !v135) )
                      {
                        v171 = (unsigned int)(v23 - 1);
                        switch ( v23 )
                        {
                          case 1:
                          case 2:
                            v25 = sub_38F2190(v6, v241.m128i_i64[0], v241.m128i_i64[1], 1u);
                            goto LABEL_43;
                          case 3:
                            v25 = sub_38F2190(v6, v241.m128i_i64[0], v241.m128i_i64[1], 0);
                            goto LABEL_43;
                          case 4:
                            v25 = sub_38E46E0(v6, v241.m128i_i64[0], v241.m128i_i64[1], 0);
                            goto LABEL_43;
                          case 5:
                          case 6:
                            v25 = sub_38E46E0(v6, v241.m128i_i64[0], v241.m128i_i64[1], 1);
                            goto LABEL_43;
                          case 7:
                          case 20:
                            v25 = sub_38E4780(v6, v241.m128i_i64[0], v241.m128i_i64[1], 1);
                            goto LABEL_43;
                          case 8:
                          case 10:
                          case 11:
                          case 18:
                          case 24:
                            v25 = sub_38E4780(v6, v241.m128i_i64[0], v241.m128i_i64[1], 2);
                            goto LABEL_43;
                          case 9:
                            LOBYTE(v219) = sub_38EF2C0((_QWORD *)v6, v239);
                            v25 = v219;
                            goto LABEL_43;
                          case 12:
                          case 13:
                          case 14:
                          case 22:
                            v25 = sub_38E4780(v6, v241.m128i_i64[0], v241.m128i_i64[1], 4);
                            goto LABEL_43;
                          case 15:
                          case 16:
                            v25 = sub_38E4780(v6, v241.m128i_i64[0], v241.m128i_i64[1], 8);
                            goto LABEL_43;
                          case 17:
                            v25 = sub_38E4650(v6, v241.m128i_i64[0], v241.m128i_i64[1]);
                            goto LABEL_43;
                          case 19:
                            v25 = sub_38E4780(
                                    v6,
                                    v241.m128i_i64[0],
                                    v241.m128i_i64[1],
                                    *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v6 + 320) + 16LL) + 8LL));
                            goto LABEL_43;
                          case 21:
                          case 43:
                            v218 = sub_1698280();
                            v25 = sub_38E4820(v6, v241.m128i_i64[0], v241.m128i_i64[1], (__int64)v218);
                            goto LABEL_43;
                          case 23:
                          case 41:
                          case 42:
                            v217 = sub_1698270();
                            v25 = sub_38E4820(v6, v241.m128i_i64[0], v241.m128i_i64[1], (__int64)v217);
                            goto LABEL_43;
                          case 25:
                          case 32:
                          case 76:
                          case 77:
                            v253.m128i_i64[0] = (__int64)" not currently supported for this target";
                            s1[0] = &v241;
                            LOWORD(v254) = 259;
                            LOWORD(v248) = 261;
                            sub_14EC200((__m128i *)&v258, (const __m128i *)s1, &v253);
                            v25 = sub_3909CF0(v6, &v258, 0, 0, v213, v214);
                            goto LABEL_43;
                          case 26:
                          case 31:
                            v25 = sub_38ECA70(v6, v241.m128i_i64[0], v241.m128i_i64[1], 2u);
                            goto LABEL_43;
                          case 27:
                            v25 = sub_38ECA70(v6, v241.m128i_i64[0], v241.m128i_i64[1], 1u);
                            goto LABEL_43;
                          case 28:
                            v216 = sub_1698280();
                            v25 = sub_38F5D90(v6, v241.m128i_i64[0], v241.m128i_i64[1], (__int64)v216, a4, v10, a6);
                            goto LABEL_43;
                          case 29:
                            v25 = sub_38ECA70(v6, v241.m128i_i64[0], v241.m128i_i64[1], 4u);
                            goto LABEL_43;
                          case 30:
                            v215 = sub_1698270();
                            v25 = sub_38F5D90(v6, v241.m128i_i64[0], v241.m128i_i64[1], (__int64)v215, a4, v10, a6);
                            goto LABEL_43;
                          case 33:
                          case 39:
                            v25 = sub_38EC290(v6, v241.m128i_i64[0], v241.m128i_i64[1], 2u);
                            goto LABEL_43;
                          case 34:
                            v25 = sub_38EC290(v6, v241.m128i_i64[0], v241.m128i_i64[1], 1u);
                            goto LABEL_43;
                          case 35:
                            v25 = sub_38EC290(v6, v241.m128i_i64[0], v241.m128i_i64[1], 8u);
                            goto LABEL_43;
                          case 36:
                          case 38:
                            v25 = sub_38EC290(v6, v241.m128i_i64[0], v241.m128i_i64[1], 4u);
                            goto LABEL_43;
                          case 37:
                          case 40:
                            v25 = sub_38EC290(v6, v241.m128i_i64[0], v241.m128i_i64[1], 0xCu);
                            goto LABEL_43;
                          case 44:
                            v25 = sub_38EBA90(
                                    v6,
                                    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 320) + 16LL) + 283LL) ^ 1u,
                                    1u);
                            goto LABEL_43;
                          case 45:
                            v25 = sub_38EBA90(
                                    v6,
                                    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 320) + 16LL) + 283LL) ^ 1u,
                                    4u);
                            goto LABEL_43;
                          case 46:
                            v25 = sub_38EBA90(v6, 0, 1u);
                            goto LABEL_43;
                          case 47:
                            v25 = sub_38EBA90(v6, 0, 2u);
                            goto LABEL_43;
                          case 48:
                            v25 = sub_38EBA90(v6, 0, 4u);
                            goto LABEL_43;
                          case 49:
                            v25 = sub_38EBA90(v6, 1, 1u);
                            goto LABEL_43;
                          case 50:
                            v25 = sub_38EBA90(v6, 1, 2u);
                            goto LABEL_43;
                          case 51:
                            v25 = sub_38EBA90(v6, 1, 4u);
                            goto LABEL_43;
                          case 52:
                            v25 = sub_38EC550(v6);
                            goto LABEL_43;
                          case 53:
                            v25 = sub_38EC670(v6);
                            goto LABEL_43;
                          case 54:
                            if ( *(_QWORD *)(v6 + 456) == *(_QWORD *)(v6 + 448) )
                            {
                              v258 = (__m128i *)"unmatched '.endr' directive";
                              v259.m128i_i16[4] = 259;
                              v25 = sub_3909CF0(v6, &v258, 0, 0, v130, v131);
                            }
                            else
                            {
                              sub_38EE710(v6);
                            }
                            goto LABEL_43;
                          case 55:
                            v25 = sub_38EBE90(v6);
                            goto LABEL_43;
                          case 56:
                            v25 = sub_38F27B0(v6);
                            goto LABEL_43;
                          case 57:
                            if ( !*(_BYTE *)(v6 + 845) && (unsigned __int8)sub_38E36C0(v6)
                              || (v258 = (__m128i *)"unexpected token in '.bundle_unlock' directive",
                                  v259.m128i_i16[4] = 259,
                                  v25 = sub_3909E20(v6, 9, &v258),
                                  (_BYTE)v25) )
                            {
                              v25 = v104;
                            }
                            else
                            {
                              (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 1040LL))(*(_QWORD *)(v6 + 328));
                            }
                            goto LABEL_43;
                          case 58:
                            v25 = sub_38EE610(v6);
                            goto LABEL_43;
                          case 59:
                            goto LABEL_375;
                          case 60:
                          case 61:
                            v25 = sub_38E2ED0(v6, 8);
                            goto LABEL_43;
                          case 62:
                            v25 = sub_38E2ED0(v6, 12);
                            goto LABEL_43;
                          case 63:
                            v25 = sub_38E2ED0(v6, 14);
                            goto LABEL_43;
                          case 64:
                            v25 = sub_38E2ED0(v6, 15);
                            goto LABEL_43;
                          case 65:
                            v25 = sub_38E2ED0(v6, 17);
                            goto LABEL_43;
                          case 66:
                            v25 = sub_38E2ED0(v6, 19);
                            goto LABEL_43;
                          case 67:
                            v25 = sub_38E2ED0(v6, 21);
                            goto LABEL_43;
                          case 68:
                            v25 = sub_38E2ED0(v6, 22);
                            goto LABEL_43;
                          case 69:
                            v25 = sub_38E2ED0(v6, 23);
                            goto LABEL_43;
                          case 70:
                          case 71:
                            v25 = sub_38F23C0(v6, 0);
                            goto LABEL_43;
                          case 72:
                            v25 = sub_38F23C0(v6, 1);
                            goto LABEL_43;
                          case 73:
                            v25 = sub_38EB0A0(v6);
                            goto LABEL_43;
                          case 74:
                            v25 = sub_38ED6B0(v6);
                            goto LABEL_43;
                          case 75:
                            v25 = sub_38ED9A0(v6);
                            goto LABEL_43;
                          case 78:
                            v25 = sub_38F02C0(v6, v239, v241.m128i_i64[0], v241.m128i_i64[1]);
                            goto LABEL_43;
                          case 79:
                            v25 = sub_38F7640(v6, v239, v129, v171);
                            goto LABEL_43;
                          case 80:
                            v25 = sub_38F7A10(v6, v239, v129, v171);
                            goto LABEL_43;
                          case 100:
                          case 101:
                            v25 = sub_38EC920(v6, v241.m128i_i64[0], v241.m128i_i64[1]);
                            goto LABEL_43;
                          case 102:
                            v25 = sub_38F2910(v6, v239);
                            goto LABEL_43;
                          case 103:
                            if ( **(_DWORD **)(v6 + 152) != 4
                              || (v258 = (__m128i *)"unexpected token in '.line' directive",
                                  v259.m128i_i16[4] = 259,
                                  v25 = sub_3909D40(v6, &v253, &v258),
                                  !(_BYTE)v25) )
                            {
                              v258 = (__m128i *)"unexpected token in '.line' directive";
                              v259.m128i_i16[4] = 259;
                              v25 = sub_3909E20(v6, 9, &v258);
                            }
                            goto LABEL_43;
                          case 104:
                            v25 = sub_38EE960(v6);
                            goto LABEL_43;
                          case 105:
                            v258 = (__m128i *)"unsupported directive '.stabs'";
                            v259.m128i_i16[4] = 259;
                            v25 = sub_3909CF0(v6, &v258, 0, 0, v130, v131);
                            goto LABEL_43;
                          case 106:
                            v25 = sub_38EDF20(v6);
                            goto LABEL_43;
                          case 107:
                            v25 = sub_38E3290(v6);
                            goto LABEL_43;
                          case 108:
                            v25 = sub_38EEC20(v6);
                            goto LABEL_43;
                          case 109:
                            v25 = sub_38EEF40(v6);
                            goto LABEL_43;
                          case 110:
                            v25 = sub_38F3A80(v6);
                            goto LABEL_43;
                          case 111:
                            v25 = sub_38F3C90(v6);
                            goto LABEL_43;
                          case 112:
                            v25 = sub_38F3F90(v6);
                            goto LABEL_43;
                          case 113:
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 656LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 114:
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 664LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 115:
                            v258 = (__m128i *)"expected identifier in directive";
                            v259.m128i_i16[4] = 259;
                            if ( (unsigned __int8)sub_3909D40(v6, &v253, &v258)
                              || (v258 = (__m128i *)"Expected End of Statement",
                                  v259.m128i_i16[4] = 259,
                                  v25 = sub_3909E20(v6, 9, &v258),
                                  (_BYTE)v25) )
                            {
                              v25 = v104;
                            }
                            else
                            {
                              v211 = *(_QWORD *)(v6 + 328);
                              v212 = *(void (**)())(*(_QWORD *)v211 + 672LL);
                              if ( v212 != nullsub_593 )
                                ((void (__fastcall *)(__int64, _QWORD))v212)(v211, v253.m128i_u32[0]);
                            }
                            goto LABEL_43;
                          case 116:
                            v25 = sub_38F4240(v6);
                            goto LABEL_43;
                          case 117:
                            v25 = sub_38F4360(v6, v127, v129, v171);
                            goto LABEL_43;
                          case 118:
                            v25 = sub_38F4550(v6);
                            goto LABEL_43;
                          case 119:
                            sub_38DD230(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 120:
                            v25 = sub_38EC0B0(v6, v239);
                            goto LABEL_43;
                          case 121:
                            v258 = 0;
                            v25 = sub_38EB9C0(v6, &v258);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 728LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 122:
                            v258 = 0;
                            v25 = sub_38EB9C0(v6, &v258);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 808LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 123:
                            v258 = 0;
                            v25 = sub_38EBF60(v6, &v258, v239);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 736LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 124:
                            v25 = sub_38EC150(v6, v239);
                            goto LABEL_43;
                          case 125:
                            v25 = sub_38EC1F0(v6, v239);
                            goto LABEL_43;
                          case 126:
                            v25 = sub_38F4680(v6, 1);
                            goto LABEL_43;
                          case 127:
                            v25 = sub_38F4680(v6, 0);
                            goto LABEL_43;
                          case 128:
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 768LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 129:
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 776LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 130:
                            v258 = 0;
                            v25 = sub_38EBF60(v6, &v258, v239);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 784LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 131:
                            v258 = 0;
                            v25 = sub_38EBF60(v6, &v258, v239);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 792LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 132:
                            v25 = sub_38EF120(v6);
                            goto LABEL_43;
                          case 133:
                            v258 = 0;
                            v25 = sub_38EBF60(v6, &v258, v239);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 824LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 134:
                            v258 = (__m128i *)"unexpected token in '.cfi_signal_frame'";
                            v259.m128i_i16[4] = 259;
                            v25 = sub_3909E20(v6, 9, &v258);
                            if ( (_BYTE)v25 )
                              goto LABEL_232;
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 840LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 135:
                            v258 = 0;
                            v25 = sub_38EBF60(v6, &v258, v239);
                            if ( !(_BYTE)v25 )
                              (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(v6 + 328) + 848LL))(
                                *(_QWORD *)(v6 + 328),
                                v258);
                            goto LABEL_43;
                          case 136:
                            v25 = sub_38EC000(v6, v239);
                            goto LABEL_43;
                          case 137:
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 864LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 138:
                          case 139:
                            v253.m128i_i64[0] = (__int64)"unexpected token in '";
                            v209 = _mm_loadu_si128(&v241);
                            v253.m128i_i64[1] = (__int64)s1;
                            v258 = &v253;
                            LOWORD(v254) = 1283;
                            v259.m128i_i64[0] = (__int64)"' directive";
                            v259.m128i_i16[4] = 770;
                            *(__m128i *)s1 = v209;
                            v25 = sub_3909E20(v6, 9, &v258);
                            if ( !(_BYTE)v25 )
                            {
                              v210 = 0;
                              if ( s1[1] == (void *)10 )
                                v210 = memcmp(s1[0], ".macros_on", 0xAu) == 0;
                              *(_BYTE *)(v6 + 552) = v210 | *(_BYTE *)(v6 + 552) & 0xFE;
                            }
                            goto LABEL_43;
                          case 140:
                          case 141:
                            v208 = *(_DWORD **)(v6 + 152);
                            *(__m128i *)s1 = _mm_loadu_si128(&v241);
                            if ( *v208 == 9 )
                            {
                              *(_BYTE *)(v6 + 272) = s1[1] == (void *)9 && !memcmp(s1[0], ".altmacro", 9u);
                            }
                            else
                            {
                              v253.m128i_i64[0] = (__int64)"unexpected token in '";
                              v253.m128i_i64[1] = (__int64)s1;
                              v258 = &v253;
                              LOWORD(v254) = 1283;
                              v259.m128i_i64[0] = (__int64)"' directive";
                              v259.m128i_i16[4] = 770;
                              v25 = sub_3909CF0(v6, &v258, 0, 0, v130, v131);
                            }
                            goto LABEL_43;
                          case 142:
                            v25 = sub_38F7FC0(v6, v239, v129, v171);
                            goto LABEL_43;
                          case 143:
                            v25 = sub_38EE770((_QWORD *)v6, v241.m128i_i64[0], v241.m128i_i64[1]);
                            goto LABEL_43;
                          case 144:
                          case 145:
                            v25 = sub_38EE8B0((_QWORD *)v6, v241.m128i_i64[0], v241.m128i_i64[1], v171, v130, v131);
                            goto LABEL_43;
                          case 146:
                            v25 = sub_38F4810(v6, v239);
                            goto LABEL_43;
                          case 147:
                            v25 = sub_38E3760(v6, 1);
                            goto LABEL_43;
                          case 148:
                            v25 = sub_38E3760(v6, 0);
                            goto LABEL_43;
                          case 149:
                            v25 = sub_38F0890(v6, v239, 0, v171, v130, v131);
                            goto LABEL_43;
                          case 150:
                            v25 = sub_38F0890(v6, v239, 1, v171, v130, v131);
                            goto LABEL_43;
                          case 151:
                            v191 = *(_QWORD *)(v6 + 400);
                            if ( v191 != *(_QWORD *)(v6 + 392) && *(_BYTE *)(v191 - 3) )
                            {
LABEL_375:
                              sub_38F0630(v6);
                            }
                            else
                            {
                              v253.m128i_i64[1] = 41;
                              v253.m128i_i64[0] = (__int64)".warning directive invoked in source file";
                              if ( (unsigned __int8)sub_3909EB0(v6, 9) )
                                goto LABEL_372;
                              if ( **(_DWORD **)(v6 + 152) != 3 )
                              {
                                v258 = (__m128i *)".warning argument must be a string";
                                v259.m128i_i16[4] = 259;
                                v25 = sub_3909CF0(v6, &v258, 0, 0, v192, v193);
                                goto LABEL_43;
                              }
                              v204 = sub_3909460(v6);
                              v205 = *(_QWORD *)(v204 + 16);
                              if ( v205 )
                              {
                                v206 = 1;
                                v207 = v205 - 1;
                                if ( v205 == 1 )
                                  v207 = 1;
                              }
                              else
                              {
                                v206 = 0;
                                v207 = -1;
                              }
                              if ( v205 > v207 )
                                v205 = v207;
                              v253.m128i_i64[0] = v206 + *(_QWORD *)(v204 + 8);
                              v253.m128i_i64[1] = v205 - v206;
                              sub_38EB180(v6);
                              v258 = (__m128i *)"expected end of statement in '.warning' directive";
                              v259.m128i_i16[4] = 259;
                              v25 = sub_3909E20(v6, 9, &v258);
                              if ( !(_BYTE)v25 )
                              {
LABEL_372:
                                v259.m128i_i16[4] = 261;
                                v258 = &v253;
                                v25 = sub_38E4170((_QWORD *)v6, v239, (__int64)&v258, 0, 0);
                              }
                            }
                            break;
                          case 152:
                            v190 = sub_3909460(v6);
                            LODWORD(v258) = *(_DWORD *)v190;
                            v259 = _mm_loadu_si128((const __m128i *)(v190 + 8));
                            LODWORD(v261) = *(_DWORD *)(v190 + 32);
                            if ( (unsigned int)v261 > 0x40 )
                              sub_16A4FD0((__int64)&v260, (const void **)(v190 + 24));
                            else
                              v260 = *(_QWORD *)(v190 + 24);
                            sub_38EB180(v6);
                            if ( (_DWORD)v258 == 3 && *(_BYTE *)v259.m128i_i64[0] == 34 )
                            {
                              v253.m128i_i64[0] = (__int64)"expected end of statement";
                              LOWORD(v254) = 259;
                              v25 = sub_3909E20(v6, 9, &v253);
                              if ( !(_BYTE)v25 )
                              {
                                v197 = sub_16E8C20(v6, 9, v194, v195, v196);
                                v198 = (__int64)v197;
                                if ( v259.m128i_i64[1] )
                                {
                                  v199 = (void *)v197[3];
                                  v200 = v259.m128i_i64[1] - 1;
                                  if ( v259.m128i_i64[1] == 1 )
                                    v200 = 1;
                                  if ( v200 > v259.m128i_i64[1] )
                                    v200 = v259.m128i_u64[1];
                                  v201 = v200 - 1;
                                  v202 = (void *)(v259.m128i_i64[0] + 1);
                                  if ( *(_QWORD *)(v198 + 16) - (_QWORD)v199 >= v200 - 1 )
                                  {
                                    if ( v200 != 1 )
                                    {
                                      memcpy(v199, v202, v201);
                                      *(_QWORD *)(v198 + 24) += v201;
                                    }
                                  }
                                  else
                                  {
                                    v198 = sub_16E7EE0(v198, (char *)v202, v201);
                                  }
                                }
                                v203 = *(_BYTE **)(v198 + 24);
                                if ( (unsigned __int64)v203 >= *(_QWORD *)(v198 + 16) )
                                {
                                  sub_16E7DE0(v198, 10);
                                }
                                else
                                {
                                  *(_QWORD *)(v198 + 24) = v203 + 1;
                                  *v203 = 10;
                                }
                              }
                            }
                            else
                            {
                              v253.m128i_i64[0] = (__int64)"expected double quoted string after .print";
                              LOWORD(v254) = 259;
                              v25 = sub_3909790(v6, v239, &v253, 0, 0);
                            }
                            if ( (unsigned int)v261 > 0x40 && v260 )
                              j_j___libc_free_0_0(v260);
                            goto LABEL_43;
                          case 153:
                            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 328) + 1000LL))(*(_QWORD *)(v6 + 328));
                            goto LABEL_43;
                          case 154:
                            v253 = 0u;
                            v258 = (__m128i *)"expected identifier in '.addrsig_sym' directive";
                            v259.m128i_i16[4] = 259;
                            v184 = sub_38F0EE0(v6, v253.m128i_i64, v129, v171);
                            v25 = sub_3909CB0(v6, v184, &v258);
                            if ( !(_BYTE)v25 )
                            {
                              v185 = *(_QWORD *)(v6 + 320);
                              v259.m128i_i16[4] = 261;
                              v258 = &v253;
                              v186 = sub_38BF510(v185, (__int64)&v258);
                              v187 = *(_QWORD *)(v6 + 328);
                              v188 = v186;
                              v189 = *(void (**)())(*(_QWORD *)v187 + 1008LL);
                              if ( v189 != nullsub_596 )
                                ((void (__fastcall *)(__int64, __int64))v189)(v187, v188);
                            }
                            goto LABEL_43;
                          case 155:
                            v258 = (__m128i *)"unexpected token in '.end' directive";
                            v259.m128i_i16[4] = 259;
                            v25 = sub_3909E20(v6, 9, &v258);
                            if ( (_BYTE)v25 )
                              goto LABEL_232;
                            while ( 1 )
                            {
                              v174 = *(_DWORD **)(v6 + 152);
                              if ( !*v174 )
                                break;
                              v175 = *(unsigned int *)(v6 + 160);
                              *(_BYTE *)(v6 + 258) = *v174 == 9;
                              v176 = v174 + 10;
                              v177 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v175 - 40) >> 3);
                              while ( v177 > 0 )
                              {
                                v178 = _mm_loadu_si128((const __m128i *)(v176 + 2));
                                v152 = *(v176 - 2) <= 0x40u;
                                *(v176 - 10) = *v176;
                                *((__m128i *)v176 - 2) = v178;
                                if ( !v152 )
                                {
                                  v179 = *((_QWORD *)v176 - 2);
                                  if ( v179 )
                                    j_j___libc_free_0_0(v179);
                                }
                                v180 = *((_QWORD *)v176 + 3);
                                --v177;
                                v176 += 10;
                                *((_QWORD *)v176 - 7) = v180;
                                LODWORD(v180) = *(v176 - 2);
                                *(v176 - 2) = 0;
                                *(v176 - 12) = v180;
                              }
                              v181 = (unsigned int)(*(_DWORD *)(v6 + 160) - 1);
                              *(_DWORD *)(v6 + 160) = v181;
                              v182 = *(_QWORD *)(v6 + 152) + 40 * v181;
                              if ( *(_DWORD *)(v182 + 32) > 0x40u )
                              {
                                v183 = *(_QWORD *)(v182 + 24);
                                if ( v183 )
                                  j_j___libc_free_0_0(v183);
                              }
                              if ( !*(_DWORD *)(v6 + 160) )
                              {
                                sub_392C2E0(&v258, v6 + 144);
                                sub_38E90E0(v6 + 152, *(_QWORD *)(v6 + 152), (unsigned __int64)&v258);
                                if ( (unsigned int)v261 > 0x40 )
                                {
                                  if ( v260 )
                                    j_j___libc_free_0_0(v260);
                                }
                              }
                            }
                            goto LABEL_43;
                          default:
                            v258 = (__m128i *)"unknown directive";
                            v259.m128i_i16[4] = 259;
                            v25 = sub_3909790(v6, v239, &v258, 0, 0);
                            goto LABEL_43;
                        }
                      }
                      else
                      {
                        v25 = v136(v135, v241.m128i_i64[0], v241.m128i_i64[1], v239);
                      }
                    }
                  }
                  else
                  {
                    v25 = v104;
                  }
                }
                goto LABEL_43;
              }
              if ( !*(_BYTE *)(v6 + 845) )
                goto LABEL_55;
            }
            else
            {
              if ( !*(_BYTE *)(v6 + 845) )
                goto LABEL_55;
              v62 = v241.m128i_i32[2];
              if ( v241.m128i_i64[1] != 5 )
              {
                if ( v241.m128i_i64[1] == 6 )
                {
                  if ( (*(_DWORD *)v241.m128i_i64[0] != 1835360095 || *(_WORD *)(v241.m128i_i64[0] + 4) != 29801)
                    && (*(_DWORD *)v241.m128i_i64[0] != 1296392031 || *(_WORD *)(v241.m128i_i64[0] + 4) != 21577) )
                  {
                    goto LABEL_56;
                  }
LABEL_110:
                  v232 = v239;
                  v225 = sub_3909290(v6 + 144);
                  v258 = 0;
                  LOBYTE(v63) = sub_38EB6A0(v6, v253.m128i_i64, (__int64)&v258);
                  v25 = v63;
                  if ( !(_BYTE)v63 )
                  {
                    if ( *(_DWORD *)v253.m128i_i64[0] == 1 )
                    {
                      v65 = *(_QWORD *)(v253.m128i_i64[0] + 16);
                      if ( (v65 & 0xFFFFFFFFFFFFFF00LL) == 0 || v65 == (char)v65 )
                      {
                        v140 = *(_QWORD *)(a2 + 88);
                        v141 = *(_DWORD *)(v140 + 8);
                        if ( v141 >= *(_DWORD *)(v140 + 12) )
                        {
                          sub_16CD150(*(_QWORD *)(a2 + 88), (const void *)(v140 + 16), 0, 104, v64, v225);
                          v141 = *(_DWORD *)(v140 + 8);
                        }
                        v142 = *(_QWORD *)v140 + 104LL * v141;
                        if ( v142 )
                        {
                          *(_DWORD *)v142 = 2;
                          *(_DWORD *)(v142 + 16) = v62;
                          *(_QWORD *)(v142 + 8) = v232;
                          *(_QWORD *)(v142 + 24) = 0;
                          *(_QWORD *)(v142 + 32) = 0;
                          *(_QWORD *)(v142 + 40) = 0;
                          *(_BYTE *)(v142 + 48) = 0;
                          *(_QWORD *)(v142 + 56) = 0;
                          *(_QWORD *)(v142 + 64) = 0;
                          *(_QWORD *)(v142 + 72) = 0;
                          *(_QWORD *)(v142 + 80) = 0;
                          *(_QWORD *)(v142 + 88) = 0;
                          *(_DWORD *)(v142 + 96) = 1;
                          v141 = *(_DWORD *)(v140 + 8);
                        }
                        *(_DWORD *)(v140 + 8) = v141 + 1;
                      }
                      else
                      {
                        v258 = (__m128i *)"literal value out of range for directive";
                        v259.m128i_i16[4] = 259;
                        v25 = sub_3909790(v6, v225, &v258, 0, 0);
                      }
                    }
                    else
                    {
                      v258 = (__m128i *)"unexpected expression in _emit";
                      v259.m128i_i16[4] = 259;
                      v25 = sub_3909790(v6, v225, &v258, 0, 0);
                    }
                  }
                  goto LABEL_43;
                }
                if ( v241.m128i_i64[1] != 4
                  || *(_DWORD *)v241.m128i_i64[0] != 1852143205 && *(_DWORD *)v241.m128i_i64[0] != 1313166917 )
                {
                  goto LABEL_56;
                }
                v122 = *(_QWORD *)(a2 + 88);
                v123 = *(_DWORD *)(v122 + 8);
                if ( v123 >= *(_DWORD *)(v122 + 12) )
                {
                  sub_16CD150(*(_QWORD *)(a2 + 88), (const void *)(v122 + 16), 0, 104, v19, v20);
                  v123 = *(_DWORD *)(v122 + 8);
                }
                v18 = 13LL * v123;
                v124 = *(_QWORD *)v122 + 104LL * v123;
                if ( v124 )
                {
                  v125 = v239;
                  *(_DWORD *)(v124 + 16) = 4;
                  *(_DWORD *)v124 = 1;
                  *(_QWORD *)(v124 + 8) = v125;
                  *(_QWORD *)(v124 + 24) = 0;
                  *(_QWORD *)(v124 + 32) = 0;
                  *(_QWORD *)(v124 + 40) = 0;
                  *(_BYTE *)(v124 + 48) = 0;
                  *(_QWORD *)(v124 + 56) = 0;
                  *(_QWORD *)(v124 + 64) = 0;
                  *(_QWORD *)(v124 + 72) = 0;
                  *(_QWORD *)(v124 + 80) = 0;
                  *(_QWORD *)(v124 + 88) = 0;
                  *(_DWORD *)(v124 + 96) = 1;
                  v123 = *(_DWORD *)(v122 + 8);
                }
                v17 = v123 + 1;
                *(_DWORD *)(v122 + 8) = v17;
                if ( *(_BYTE *)(v6 + 845) )
                  goto LABEL_56;
LABEL_55:
                if ( !(unsigned __int8)sub_38E36C0(v6) )
                  goto LABEL_56;
                goto LABEL_222;
              }
              if ( *(_DWORD *)v241.m128i_i64[0] == 1768777055 && *(_BYTE *)(v241.m128i_i64[0] + 4) == 116
                || *(_DWORD *)v241.m128i_i64[0] == 1229800799 && *(_BYTE *)(v241.m128i_i64[0] + 4) == 84 )
              {
                goto LABEL_110;
              }
              if ( *(_DWORD *)v241.m128i_i64[0] == 1734962273 && *(_BYTE *)(v241.m128i_i64[0] + 4) == 110
                || *(_DWORD *)v241.m128i_i64[0] == 1195985985 && *(_BYTE *)(v241.m128i_i64[0] + 4) == 78 )
              {
                v235 = v239;
                v113 = sub_3909290(v6 + 144);
                v258 = 0;
                v114 = v113;
                LOBYTE(v115) = sub_38EB6A0(v6, v253.m128i_i64, (__int64)&v258);
                v25 = v115;
                if ( !(_BYTE)v115 )
                {
                  if ( *(_DWORD *)v253.m128i_i64[0] == 1 )
                  {
                    v117 = *(_QWORD *)(v253.m128i_i64[0] + 16);
                    if ( !v117 || (v117 & (v117 - 1)) != 0 )
                    {
                      v258 = (__m128i *)"literal value not a power of two greater then zero";
                      v259.m128i_i16[4] = 259;
                      v25 = sub_3909790(v6, v114, &v258, 0, 0);
                    }
                    else
                    {
                      v118 = *(_QWORD *)(a2 + 88);
                      if ( *(_DWORD *)(v118 + 8) >= *(_DWORD *)(v118 + 12) )
                      {
                        v228 = *(_QWORD *)(v253.m128i_i64[0] + 16);
                        sub_16CD150(*(_QWORD *)(a2 + 88), (const void *)(v118 + 16), 0, 104, v117, v116);
                        v117 = v228;
                      }
                      v119 = *(_QWORD *)v118 + 104LL * *(unsigned int *)(v118 + 8);
                      if ( v119 )
                      {
                        _BitScanReverse64(&v120, v117);
                        *(_DWORD *)v119 = 0;
                        *(_DWORD *)(v119 + 16) = 5;
                        *(_QWORD *)(v119 + 8) = v235;
                        *(_QWORD *)(v119 + 32) = 0;
                        *(_QWORD *)(v119 + 24) = 63 - ((unsigned int)v120 ^ 0x3F);
                        *(_QWORD *)(v119 + 40) = 0;
                        *(_BYTE *)(v119 + 48) = 0;
                        *(_QWORD *)(v119 + 56) = 0;
                        *(_QWORD *)(v119 + 64) = 0;
                        *(_QWORD *)(v119 + 72) = 0;
                        *(_QWORD *)(v119 + 80) = 0;
                        *(_QWORD *)(v119 + 88) = 0;
                        *(_DWORD *)(v119 + 96) = 1;
                      }
                      ++*(_DWORD *)(v118 + 8);
                    }
                  }
                  else
                  {
                    v258 = (__m128i *)"unexpected expression in align";
                    v259.m128i_i16[4] = 259;
                    v25 = sub_3909790(v6, v114, &v258, 0, 0);
                  }
                }
                goto LABEL_43;
              }
            }
LABEL_56:
            sub_16D2060(s1, &v241, v17, v18, v19);
            v38 = *(_QWORD *)(v6 + 8);
            v240 = *(_QWORD *)(a2 + 88);
            v39 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v38 + 56LL);
            v259 = _mm_loadu_si128(&v250);
            LODWORD(v258) = v249;
            LODWORD(v261) = v252;
            if ( v252 > 0x40 )
            {
              v236 = v38;
              sub_16A4FD0((__int64)&v260, (const void **)&v251);
              v38 = v236;
            }
            else
            {
              v260 = v251;
            }
            v40 = s1[1];
            if ( v39 == sub_38E2D80 )
            {
              v224 = s1[0];
              v230 = v38;
              v41 = *(__int64 (__fastcall **)(__int64, __int64 *, void *, void *, __int64, __int64))(*(_QWORD *)v38 + 48LL);
              v42 = sub_39092A0(&v258);
              v231 = v41(v230, &v240, v224, v40, v42, a2);
            }
            else
            {
              v231 = v39(v38, (__int64)&v240, (__int64)s1[0], (__int64)s1[1], (__int64)&v258, a2);
            }
            if ( (unsigned int)v261 > 0x40 && v260 )
              j_j___libc_free_0_0(v260);
            *(_BYTE *)(a2 + 84) = v231;
            if ( (*(_BYTE *)(v6 + 16) & 1) != 0 )
            {
              v258 = (__m128i *)&v259.m128i_u64[1];
              v259.m128i_i64[0] = 0x10000000000LL;
              v256 = 1;
              v255 = 0;
              v253.m128i_i64[0] = (__int64)&unk_49EFC48;
              v254 = 0;
              v253.m128i_i64[1] = 0;
              v257 = &v258;
              sub_16E7A40((__int64)&v253, 0, 0, 0);
              v106 = v255;
              if ( (unsigned __int64)((char *)v254 - (char *)v255) <= 0x14 )
              {
                sub_16E7EE0((__int64)&v253, "parsed instruction: [", 0x15u);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_3F85330);
                v255[1].m128i_i32[0] = 540700271;
                v106[1].m128i_i8[4] = 91;
                *v106 = si128;
                v255 = (__m128i *)((char *)v255 + 21);
              }
              v108 = 0;
              v109 = 0;
              if ( *(_DWORD *)(a2 + 8) )
              {
                while ( 1 )
                {
                  (*(void (__fastcall **)(_QWORD, __m128i *))(**(_QWORD **)(*(_QWORD *)a2 + 8 * v108) + 112LL))(
                    *(_QWORD *)(*(_QWORD *)a2 + 8 * v108),
                    &v253);
                  v108 = (unsigned int)(v109 + 1);
                  v109 = v108;
                  if ( *(_DWORD *)(a2 + 8) == (_DWORD)v108 )
                    break;
                  if ( (_DWORD)v108 )
                  {
                    if ( (unsigned __int64)((char *)v254 - (char *)v255) <= 1 )
                    {
                      v221 = v108;
                      sub_16E7EE0((__int64)&v253, ", ", 2u);
                      v108 = v221;
                    }
                    else
                    {
                      v255->m128i_i16[0] = 8236;
                      v255 = (__m128i *)((char *)v255 + 2);
                    }
                  }
                }
              }
              if ( v254 == v255 )
              {
                sub_16E7EE0((__int64)&v253, "]", 1u);
              }
              else
              {
                v255->m128i_i8[0] = 93;
                v255 = (__m128i *)((char *)v255 + 1);
              }
              v110 = *(__int64 **)(v6 + 344);
              v111 = *((unsigned int *)v257 + 2);
              v112 = *v257;
              LOWORD(v246) = 261;
              v242[0] = v112;
              v242[1] = v111;
              v244 = v242;
              v243[0] = 0;
              v243[1] = 0;
              sub_16D14E0(v110, v239, 3, (__int64)&v244, v243, 1, 0, 0, 1u);
              v253.m128i_i64[0] = (__int64)&unk_49EFD28;
              sub_16E7960((__int64)&v253);
              if ( v258 != (__m128i *)&v259.m128i_u64[1] )
                _libc_free((unsigned __int64)v258);
            }
            v25 = v231;
            LOBYTE(v25) = (*(_DWORD *)(v6 + 32) != 0) | v231;
            if ( !(_BYTE)v25 )
            {
              if ( (unsigned __int8)sub_38E9F70(v6) )
              {
                v121 = *(_QWORD *)(v6 + 328);
                v159 = *(_QWORD *)(v6 + 320);
                v160 = 0;
                v161 = *(unsigned int *)(v121 + 120);
                if ( (_DWORD)v161 )
                  v160 = *(_QWORD *)(*(_QWORD *)(v121 + 112) + 32 * v161 - 32);
                v162 = *(_DWORD *)(v159 + 1072);
                if ( v162 )
                {
                  v163 = *(_QWORD *)(v159 + 1056);
                  v164 = v162 - 1;
                  v165 = 1;
                  v166 = (v162 - 1) & (((unsigned int)v160 >> 9) ^ ((unsigned int)v160 >> 4));
                  v167 = *(_QWORD *)(v163 + 8LL * v166);
                  if ( v167 == v160 )
                  {
LABEL_299:
                    v168 = *(unsigned __int64 ***)(v6 + 448);
                    v169 = *(__int64 **)(v6 + 344);
                    if ( *(unsigned __int64 ***)(v6 + 456) == v168 )
                      v170 = sub_16CFA40(v169, v239, *(_DWORD *)(v6 + 376));
                    else
                      v170 = sub_16CFA40(v169, **v168, *((_DWORD *)*v168 + 2));
                    if ( *(_QWORD *)(v6 + 568) )
                    {
                      v172 = *(_QWORD *)(v6 + 328);
                      v173 = *(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __m128i **, _QWORD))(*(_QWORD *)v172 + 568LL);
                      v259.m128i_i8[8] = 0;
                      v173(&v253, v172, 0, 0, 0, 0, *(_QWORD *)(v6 + 560), *(_QWORD *)(v6 + 568), &v258, 0);
                      *(_DWORD *)(*(_QWORD *)(v6 + 320) + 1044LL) = v253.m128i_i32[0];
                      v170 += *(_DWORD *)(v6 + 576)
                            + ~(unsigned int)sub_16CFA40(
                                               *(__int64 **)(v6 + 344),
                                               *(_QWORD *)(v6 + 584),
                                               *(_DWORD *)(v6 + 592));
                    }
                    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v6 + 328) + 584LL))(
                      *(_QWORD *)(v6 + 328),
                      *(unsigned int *)(*(_QWORD *)(v6 + 320) + 1044LL),
                      v170,
                      0,
                      1,
                      0,
                      0,
                      0,
                      0);
                    v121 = *(_QWORD *)(v6 + 328);
                  }
                  else
                  {
                    while ( v167 != -8 )
                    {
                      v166 = v164 & (v165 + v166);
                      v167 = *(_QWORD *)(v163 + 8LL * v166);
                      if ( v167 == v160 )
                        goto LABEL_299;
                      ++v165;
                    }
                  }
                }
              }
              else
              {
                v121 = *(_QWORD *)(v6 + 328);
              }
              v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __m128i **, _QWORD))(**(_QWORD **)(v6 + 8) + 72LL))(
                      *(_QWORD *)(v6 + 8),
                      v239,
                      a2 + 80,
                      a2,
                      v121,
                      &v258,
                      *(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 32LL));
            }
            if ( s1[0] != &v248 )
              j_j___libc_free_0((unsigned __int64)s1[0]);
            goto LABEL_43;
          }
          if ( v27 != 27 )
            goto LABEL_52;
          v28 = *(__int64 (**)())(**(_QWORD **)(a1 + 8) + 128LL);
          if ( v28 != sub_38E29E0 && !(unsigned __int8)v28() )
            goto LABEL_52;
          sub_38EB180(a1);
          v25 = sub_38E8800(a1, v241.m128i_i64[0], v241.m128i_i64[1], 1, 0);
        }
LABEL_43:
        if ( v252 > 0x40 && v251 )
          j_j___libc_free_0_0(v251);
        return v25;
      }
    }
    else if ( *(_BYTE *)(a1 + 385) )
    {
      v241.m128i_i64[1] = 0;
      v241.m128i_i64[0] = (__int64)byte_3F871B3;
      goto LABEL_15;
    }
    sub_38EB180(a1);
    v258 = (__m128i *)"unexpected token at start of statement";
    v259.m128i_i16[4] = 259;
    v25 = sub_3909790(a1, v239, &v258, 0, 0);
    goto LABEL_43;
  }
  if ( !*(_QWORD *)(sub_3909460(a1) + 16)
    || **(_BYTE **)(sub_3909460(a1) + 8) == 13
    || **(_BYTE **)(sub_3909460(a1) + 8) == 10 )
  {
    v24 = *(void (**)(void))(**(_QWORD **)(a1 + 328) + 144LL);
    if ( v24 != nullsub_581 )
      v24();
  }
  v25 = 0;
  sub_38EB180(a1);
  return v25;
}
