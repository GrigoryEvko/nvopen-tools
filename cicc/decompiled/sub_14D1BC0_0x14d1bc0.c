// Function: sub_14D1BC0
// Address: 0x14d1bc0
//
__int64 __fastcall sub_14D1BC0(
        char *s1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v9; // r14
  unsigned int v10; // ebx
  __int64 v11; // r12
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // r12
  _QWORD *v16; // r12
  char v17; // al
  __int64 v18; // rdx
  double v19; // xmm0_8
  __int64 v20; // rcx
  char v21; // al
  __int64 v22; // rdi
  __int64 v23; // rdx
  double v24; // xmm0_8
  double v25; // xmm2_8
  double v26; // xmm1_8
  char v27; // al
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  char v31; // al
  double v32; // xmm0_8
  _QWORD *v33; // rsi
  double (__fastcall *v34)(double); // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // r12
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  _QWORD *v53; // r12
  _QWORD *v54; // r13
  __int64 v55; // rsi
  __int64 v56; // rdi
  _QWORD *v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  size_t v62; // rdx
  int *v63; // rax
  int *v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // r12
  __int64 v68; // rax
  unsigned __int64 v69; // rdi
  __int64 v70; // r15
  __int64 v71; // r12
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rbx
  __int64 v75; // rax
  unsigned int v76; // eax
  unsigned __int64 v77; // rsi
  __int64 v78; // rdx
  unsigned __int64 v79; // rdi
  unsigned int v80; // eax
  char v82; // al
  _QWORD *v83; // rax
  __int64 v84; // r13
  unsigned int v85; // ebx
  unsigned int v86; // ebx
  bool v87; // al
  unsigned int v88; // ebx
  _QWORD *v89; // rbx
  _QWORD *v90; // r12
  _QWORD *v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // r12
  __int64 v94; // rdi
  __int64 v95; // rbx
  _QWORD *v96; // rbx
  int v97; // eax
  __int64 v98; // rax
  __int64 v99; // rcx
  __int64 v100; // rdi
  __int64 v101; // rdx
  char v102; // si
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rsi
  __int64 v106; // rbx
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // rax
  __int64 v110; // rsi
  __int64 v111; // rsi
  __int64 v112; // rsi
  __int64 v113; // rsi
  _BYTE *v114; // rax
  __int64 v116; // rdi
  float v117; // xmm0_4
  __int64 v118; // rdi
  float v119; // xmm0_4
  __int64 *v120; // rdi
  float v121; // xmm2_4
  float v122; // xmm0_4
  float v123; // xmm0_4
  unsigned int v124; // edx
  unsigned __int64 v125; // rax
  int v126; // ecx
  _QWORD *v127; // rax
  float v128; // xmm0_4
  float v129; // xmm0_4
  __int64 v130; // rax
  _QWORD *v131; // rsi
  __int64 v132; // rbx
  __int64 *v133; // rax
  bool v134; // al
  __int64 v135; // rcx
  __int64 v136; // rax
  __int64 v137; // rax
  unsigned int v138; // ebx
  bool v139; // al
  unsigned int v140; // ebx
  bool v141; // al
  _QWORD *v143; // rbx
  size_t v144; // rdx
  char *v145; // rsi
  _QWORD *v146; // rsi
  double v147; // xmm0_8
  double (*v148)(double, double); // rdi
  unsigned __int64 v149; // rax
  int v150; // edx
  __int64 v151; // r13
  __int64 v152; // r15
  __int64 v153; // r12
  __int64 v154; // rdx
  __int64 v155; // rcx
  __int64 v156; // r8
  __int64 v157; // r15
  __int64 i; // r12
  char v159; // al
  char *v160; // r15
  __int64 v161; // [rsp+0h] [rbp-C0h]
  __int64 v162; // [rsp+8h] [rbp-B8h]
  float v163; // [rsp+8h] [rbp-B8h]
  __int64 v164; // [rsp+8h] [rbp-B8h]
  __int64 v165; // [rsp+8h] [rbp-B8h]
  double ya; // [rsp+10h] [rbp-B0h]
  __int64 yb; // [rsp+10h] [rbp-B0h]
  double yc; // [rsp+10h] [rbp-B0h]
  __int64 yd; // [rsp+10h] [rbp-B0h]
  float yf; // [rsp+10h] [rbp-B0h]
  double ye; // [rsp+10h] [rbp-B0h]
  __int64 yg; // [rsp+10h] [rbp-B0h]
  __int64 yh; // [rsp+10h] [rbp-B0h]
  __int64 yi; // [rsp+10h] [rbp-B0h]
  __int64 xa; // [rsp+18h] [rbp-A8h]
  __int64 xb; // [rsp+18h] [rbp-A8h]
  double xc; // [rsp+18h] [rbp-A8h]
  double xd; // [rsp+18h] [rbp-A8h]
  double xk; // [rsp+18h] [rbp-A8h]
  __int64 xe; // [rsp+18h] [rbp-A8h]
  __int64 xf; // [rsp+18h] [rbp-A8h]
  __int64 xg; // [rsp+18h] [rbp-A8h]
  __int64 xh; // [rsp+18h] [rbp-A8h]
  double xl; // [rsp+18h] [rbp-A8h]
  __int64 xi; // [rsp+18h] [rbp-A8h]
  double xj; // [rsp+18h] [rbp-A8h]
  char v189; // [rsp+2Fh] [rbp-91h] BYREF
  _BYTE v190[32]; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v191; // [rsp+50h] [rbp-70h] BYREF
  __int64 v192; // [rsp+58h] [rbp-68h] BYREF
  __int64 v193; // [rsp+60h] [rbp-60h]
  unsigned __int64 v194; // [rsp+70h] [rbp-50h] BYREF
  __int64 v195; // [rsp+78h] [rbp-48h] BYREF
  __int64 v196; // [rsp+80h] [rbp-40h]

  v9 = (_QWORD *)a4;
  v10 = a3;
  if ( a6 == 1 )
  {
    v14 = *a5;
    v27 = *(_BYTE *)(*a5 + 16);
    switch ( v27 )
    {
      case 9:
        if ( (_DWORD)a3 != 30 )
        {
          if ( (unsigned int)(a3 - 5) <= 1 || (_DWORD)a3 == 115 || (_DWORD)a3 == 203 || (_DWORD)a3 == 3660 )
            return v14;
          return 0;
        }
        return sub_15A06D0(v9);
      case 15:
        if ( (_DWORD)a3 == 203 || (_DWORD)a3 == 115 || (_DWORD)a3 == 3660 )
        {
          v36 = *(_QWORD *)((a8 & 0xFFFFFFFFFFFFFFF8LL) + 40);
          if ( v36 )
          {
            v37 = *(_QWORD *)(v36 + 56);
            if ( v37 )
            {
              v38 = *(_QWORD *)v14;
              if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 16 )
                v38 = **(_QWORD **)(v38 + 16);
              if ( !(unsigned __int8)sub_15E4690(v37, *(_DWORD *)(v38 + 8) >> 8) )
                return *a5;
            }
          }
        }
        return 0;
      case 14:
        if ( (_DWORD)a3 == 12 )
        {
          v104 = sub_16982C0(s1, a2, a3, a8);
          v105 = v14 + 32;
          v106 = v104;
          if ( *(_QWORD *)(v14 + 32) == v104 )
            sub_169C6E0(&v195, v105);
          else
            sub_16986C0(&v195, v105);
          v190[0] = 0;
          v109 = sub_1698260(&v195, v105, v107, v108);
          sub_16A3360(&v194, v109, 0, v190);
          if ( v195 == v106 )
            sub_169D930(&v191, &v195);
          else
            sub_169D7E0(&v191, &v195);
          v14 = sub_159C0E0(*v9, &v191);
          if ( (unsigned int)v192 > 0x40 && v191 )
            j_j___libc_free_0_0(v191);
          goto LABEL_80;
        }
        if ( (unsigned __int8)(*(_BYTE *)(a4 + 8) - 1) > 2u )
          return 0;
        yb = *(_QWORD *)(v14 + 32);
        v28 = sub_16982C0(s1, a2, a3, a8);
        switch ( v10 )
        {
          case 0xBCu:
            v110 = v14 + 32;
            xe = v28;
            if ( v28 == yb )
              sub_169C6E0(&v195, v110);
            else
              sub_16986C0(&v195, v110);
            v45 = 4;
            if ( v195 == xe )
              goto LABEL_220;
            break;
          case 0x61u:
            v44 = v14 + 32;
            xb = v28;
            if ( v28 == yb )
              sub_169C6E0(&v195, v44);
            else
              sub_16986C0(&v195, v44);
            v45 = 2;
            if ( v195 == xb )
              goto LABEL_220;
            break;
          case 8u:
            v112 = v14 + 32;
            xg = v28;
            if ( v28 == yb )
              sub_169C6E0(&v195, v112);
            else
              sub_16986C0(&v195, v112);
            v45 = 1;
            if ( v195 == xg )
              goto LABEL_220;
            break;
          case 0xCEu:
            v113 = v14 + 32;
            xh = v28;
            if ( v28 == yb )
              sub_169C6E0(&v195, v113);
            else
              sub_16986C0(&v195, v113);
            v45 = 3;
            if ( v195 == xh )
              goto LABEL_220;
            break;
          case 0xBBu:
          case 0x8Cu:
            v111 = v14 + 32;
            xf = v28;
            if ( v28 == yb )
              sub_169C6E0(&v195, v111);
            else
              sub_16986C0(&v195, v111);
            v45 = 0;
            if ( v195 == xf )
            {
LABEL_220:
              sub_169EBA0(&v195, v45);
              goto LABEL_79;
            }
            break;
          default:
            if ( v28 == yb )
            {
              v31 = *(_BYTE *)(*(_QWORD *)(v14 + 40) + 26LL) & 7;
              if ( v31 == 1 )
                return 0;
            }
            else
            {
              v31 = *(_BYTE *)(v14 + 50) & 7;
              if ( v31 == 1 )
                return 0;
            }
            if ( !v31 )
              return 0;
            v32 = sub_14D1620((_QWORD *)v14, (__int64)a2, v29, v30);
            if ( v10 > 0xEB5 )
            {
              if ( v10 > 0xF55 )
              {
                if ( v10 == 4413 )
                  goto LABEL_41;
                if ( v10 <= 0x113D )
                {
                  if ( v10 - 4073 <= 1 )
                    goto LABEL_247;
                }
                else if ( v10 - 4475 <= 9 && ((1LL << ((unsigned __int8)v10 - 123)) & 0x309) != 0 )
                {
                  goto LABEL_226;
                }
                goto LABEL_238;
              }
              if ( v10 > 0xF53 )
                goto LABEL_379;
              if ( v10 > 0xF1F )
              {
                if ( v10 - 3911 <= 1 )
                  goto LABEL_224;
                goto LABEL_238;
              }
              if ( v10 <= 0xF1D )
              {
                if ( v10 != 3787 )
                  goto LABEL_238;
                goto LABEL_233;
              }
            }
            else
            {
              if ( v10 > 0xEB3 )
              {
LABEL_372:
                v33 = v9;
                v34 = (double (__fastcall *)(double))sub_14D1410;
                return sub_14D19F0(v34, v33, v32);
              }
              if ( v10 == 122 )
              {
LABEL_382:
                v33 = v9;
                v34 = j_log;
                return sub_14D19F0(v34, v33, v32);
              }
              if ( v10 > 0x7A )
              {
                if ( v10 != 124 )
                {
                  if ( v10 <= 0x7C )
                  {
LABEL_227:
                    v33 = v9;
                    v34 = j_log10;
                    return sub_14D19F0(v34, v33, v32);
                  }
                  if ( v10 == 194 )
                  {
LABEL_41:
                    v33 = v9;
                    v34 = j_sin;
                    return sub_14D19F0(v34, v33, v32);
                  }
                  if ( v10 == 196 )
                  {
LABEL_226:
                    v33 = v9;
                    v34 = sub_14D1470;
                    return sub_14D19F0(v34, v33, v32);
                  }
                  goto LABEL_238;
                }
LABEL_247:
                v33 = v9;
                v34 = j_log2;
                return sub_14D19F0(v34, v33, v32);
              }
              if ( v10 != 55 )
              {
                if ( v10 > 0x37 )
                {
                  if ( v10 == 96 )
                  {
LABEL_224:
                    v33 = v9;
                    v34 = (double (__fastcall *)(double))sub_14D1280;
                    return sub_14D19F0(v34, v33, v32);
                  }
LABEL_238:
                  if ( !a7 )
                    return 0;
                  if ( *s1 != 95 )
                  {
                    switch ( *s1 )
                    {
                      case 'a':
                        goto LABEL_459;
                      case 'c':
                        goto LABEL_454;
                      case 'e':
                        goto LABEL_476;
                      case 'f':
                        goto LABEL_471;
                      case 'l':
                        goto LABEL_436;
                      case 'r':
                        goto LABEL_432;
                      case 's':
                        goto LABEL_446;
                      case 't':
                        goto LABEL_465;
                      default:
                        return 0;
                    }
                  }
                  if ( (unsigned __int64)a2 > 2 && s1[1] == 95 )
                  {
                    switch ( s1[2] )
                    {
                      case 'a':
LABEL_459:
                        if ( a2 == (char *)4 )
                        {
                          switch ( *(_DWORD *)s1 )
                          {
                            case 0x736F6361:
                              goto LABEL_498;
                            case 0x6E697361:
                              goto LABEL_464;
                            case 0x6E617461:
                              goto LABEL_506;
                          }
                          return 0;
                        }
                        if ( a2 != (char *)5 )
                        {
                          if ( a2 == (char *)13 )
                          {
                            if ( !memcmp(s1, "__acos_finite", 0xDu) )
                              goto LABEL_498;
                            if ( memcmp(s1, "__asin_finite", 0xDu) )
                              return 0;
                          }
                          else
                          {
                            if ( a2 != (char *)14 )
                              return 0;
                            if ( !memcmp(s1, "__acosf_finite", 0xEu) )
                              goto LABEL_498;
                            if ( memcmp(s1, "__asinf_finite", 0xEu) )
                              return 0;
                          }
                          goto LABEL_464;
                        }
                        if ( !memcmp(s1, "acosf", 5u) )
                          goto LABEL_498;
                        if ( !memcmp(s1, "asinf", 5u) )
                          goto LABEL_464;
                        if ( memcmp(s1, "atanf", 5u) )
                          return 0;
                        goto LABEL_506;
                      case 'c':
LABEL_454:
                        if ( a2 == (char *)4 )
                        {
                          switch ( *(_DWORD *)s1 )
                          {
                            case 0x6C696563:
                              goto LABEL_372;
                            case 0x66736F63:
                              goto LABEL_233;
                            case 0x68736F63:
                              goto LABEL_551;
                          }
                        }
                        else
                        {
                          if ( a2 != (char *)5 )
                          {
                            if ( a2 == (char *)3 )
                            {
                              if ( !memcmp(s1, "cos", 3u) )
                                goto LABEL_233;
                              return 0;
                            }
                            if ( a2 == (char *)13 )
                            {
                              if ( !memcmp(s1, "__cosh_finite", 0xDu) )
                                goto LABEL_551;
                              return 0;
                            }
                            if ( a2 != (char *)14 || memcmp(s1, "__coshf_finite", 0xEu) )
                              return 0;
                            goto LABEL_551;
                          }
                          if ( !memcmp(s1, "ceilf", 5u) )
                            goto LABEL_372;
                          if ( !memcmp(s1, "coshf", 5u) )
                            goto LABEL_551;
                        }
                        return 0;
                      case 'e':
LABEL_476:
                        if ( a2 == (char *)3 )
                        {
                          if ( !memcmp(s1, "exp", 3u) )
                            goto LABEL_236;
                          return 0;
                        }
                        if ( a2 != (char *)4 )
                        {
                          if ( a2 == (char *)12 )
                          {
                            if ( !memcmp(s1, "__exp_finite", 0xCu) )
                              goto LABEL_236;
                            return 0;
                          }
                          if ( a2 == (char *)13 )
                          {
                            if ( !memcmp(s1, "__expf_finite", 0xDu) )
                              goto LABEL_236;
                            if ( memcmp(s1, "__exp2_finite", 0xDu) )
                              return 0;
                          }
                          else
                          {
                            if ( a2 == (char *)5 )
                            {
                              if ( !memcmp(s1, "exp2f", 5u) )
                                return sub_14D1A80(j_pow, v9, 2.0, v32);
                              return 0;
                            }
                            if ( a2 != (char *)14 || memcmp(s1, "__exp2f_finite", 0xEu) )
                              return 0;
                          }
                          return sub_14D1A80(j_pow, v9, 2.0, v32);
                        }
                        if ( *(_DWORD *)s1 == 1718646885 )
                          goto LABEL_236;
                        if ( *(_DWORD *)s1 == 846231653 )
                          return sub_14D1A80(j_pow, v9, 2.0, v32);
                        return 0;
                      case 'f':
LABEL_471:
                        if ( a2 == (char *)4 )
                        {
                          if ( *(_DWORD *)s1 == 1935827302 )
                            goto LABEL_224;
                          return 0;
                        }
                        if ( a2 == (char *)5 )
                        {
                          if ( !memcmp(s1, "fabsf", 5u) )
                            goto LABEL_224;
                          if ( memcmp(s1, "floor", 5u) )
                            return 0;
                        }
                        else if ( a2 != (char *)6 || memcmp(s1, "floorf", 6u) )
                        {
                          return 0;
                        }
                        goto LABEL_379;
                      case 'l':
LABEL_436:
                        if ( a2 != (char *)3 )
                        {
                          if ( a2 == (char *)4 )
                          {
                            if ( *(_DWORD *)s1 == 1718054764 && v32 > 0.0 )
                              goto LABEL_382;
                          }
                          else if ( a2 == (char *)12 )
                          {
                            if ( !memcmp(s1, "__log_finite", 0xCu) && v32 > 0.0 )
                              goto LABEL_382;
                          }
                          else if ( a2 == (char *)13 )
                          {
                            if ( !memcmp(s1, "__logf_finite", 0xDu) && v32 > 0.0 )
                              goto LABEL_382;
                          }
                          else if ( a2 == (char *)5 )
                          {
                            if ( !memcmp(s1, "log10", 5u) && v32 > 0.0 )
                              goto LABEL_227;
                          }
                          else if ( a2 == (char *)6 )
                          {
                            if ( !memcmp(s1, "log10f", 6u) && v32 > 0.0 )
                              goto LABEL_227;
                          }
                          else if ( a2 == (char *)14 )
                          {
                            if ( !memcmp(s1, "__log10_finite", 0xEu) && v32 > 0.0 )
                              goto LABEL_227;
                          }
                          else if ( a2 == (char *)15 && !memcmp(s1, "__log10f_finite", 0xFu) && v32 > 0.0 )
                          {
                            goto LABEL_227;
                          }
                          return 0;
                        }
                        if ( memcmp(s1, "log", 3u) || v32 <= 0.0 )
                          return 0;
                        goto LABEL_382;
                      case 'r':
LABEL_432:
                        if ( a2 == (char *)5 )
                        {
                          if ( memcmp(s1, "round", 5u) )
                            return 0;
                        }
                        else if ( a2 != (char *)6 || memcmp(s1, "roundf", 6u) )
                        {
                          return 0;
                        }
                        v33 = v9;
                        v34 = j__round;
                        return sub_14D19F0(v34, v33, v32);
                      case 's':
LABEL_446:
                        if ( a2 == (char *)3 )
                        {
                          if ( !memcmp(s1, "sin", 3u) )
                            goto LABEL_41;
                          return 0;
                        }
                        if ( a2 == (char *)4 )
                        {
                          if ( *(_DWORD *)s1 == 1718511987 )
                            goto LABEL_41;
                          if ( *(_DWORD *)s1 == 1752066419 )
                            goto LABEL_451;
                          if ( *(_DWORD *)s1 == 1953657203 && v32 >= 0.0 )
                            goto LABEL_226;
                        }
                        else
                        {
                          if ( a2 != (char *)5 )
                          {
                            if ( a2 == (char *)13 )
                            {
                              if ( memcmp(s1, "__sinh_finite", 0xDu) )
                                return 0;
                            }
                            else if ( a2 != (char *)14 || memcmp(s1, "__sinhf_finite", 0xEu) )
                            {
                              return 0;
                            }
                            goto LABEL_451;
                          }
                          if ( !memcmp(s1, "sinhf", 5u) )
                            goto LABEL_451;
                          if ( !memcmp(s1, "sqrtf", 5u) && v32 >= 0.0 )
                            goto LABEL_226;
                        }
                        return 0;
                      case 't':
LABEL_465:
                        if ( a2 == (char *)3 )
                        {
                          if ( !memcmp(s1, "tan", 3u) )
                            goto LABEL_586;
                          return 0;
                        }
                        if ( a2 != (char *)4 )
                        {
                          if ( a2 != (char *)5 || memcmp(s1, "tanhf", 5u) )
                            return 0;
                          goto LABEL_469;
                        }
                        if ( *(_DWORD *)s1 == 1718509940 )
                          goto LABEL_586;
                        if ( *(_DWORD *)s1 == 1752064372 )
                          goto LABEL_469;
                        return 0;
                      default:
                        return 0;
                    }
                  }
                  if ( s1[1] != 90 || (unsigned __int64)a2 <= 6 )
                    return 0;
                  v159 = s1[2];
                  if ( v159 == 51 )
                  {
                    switch ( s1[3] )
                    {
                      case 'c':
                        if ( a2 == (char *)7 && (*(_DWORD *)(s1 + 3) == 1718841187 || *(_DWORD *)(s1 + 3) == 1685286755) )
                          goto LABEL_233;
                        return 0;
                      case 'e':
                        if ( a2 == (char *)7 && (*(_DWORD *)(s1 + 3) == 1718646885 || *(_DWORD *)(s1 + 3) == 1685092453) )
                          goto LABEL_236;
                        return 0;
                      case 'l':
                        if ( a2 == (char *)7 && (*(_DWORD *)(s1 + 3) == 1718054764 || *(_DWORD *)(s1 + 3) == 1684500332) )
                          goto LABEL_382;
                        return 0;
                      case 's':
                        if ( a2 == (char *)7 && (*(_DWORD *)(s1 + 3) == 1718511987 || *(_DWORD *)(s1 + 3) == 1684957555) )
                          goto LABEL_41;
                        return 0;
                      case 't':
                        if ( a2 != (char *)7 || *(_DWORD *)(s1 + 3) != 1718509940 && *(_DWORD *)(s1 + 3) != 1684955508 )
                          return 0;
LABEL_586:
                        v33 = v9;
                        v34 = j_tan;
                        break;
                      default:
                        return 0;
                    }
                    return sub_14D19F0(v34, v33, v32);
                  }
                  if ( v159 == 52 )
                  {
                    switch ( s1[3] )
                    {
                      case 'a':
                        if ( a2 != (char *)8 )
                          return 0;
                        if ( !memcmp(s1 + 3, "acosf", 5u) || !memcmp(s1 + 3, "acosd", 5u) )
                        {
LABEL_498:
                          v33 = v9;
                          v34 = j_acos;
                          return sub_14D19F0(v34, v33, v32);
                        }
                        if ( !memcmp(s1 + 3, "asinf", 5u) || !memcmp(s1 + 3, "asind", 5u) )
                        {
LABEL_464:
                          v33 = v9;
                          v34 = j_asin;
                          return sub_14D19F0(v34, v33, v32);
                        }
                        if ( memcmp(s1 + 3, "atanf", 5u) && memcmp(s1 + 3, "atand", 5u) )
                          return 0;
LABEL_506:
                        v33 = v9;
                        v34 = j_atan;
                        return sub_14D19F0(v34, v33, v32);
                      case 'c':
                        if ( a2 != (char *)8 )
                          return 0;
                        if ( !memcmp(s1 + 3, "ceilf", 5u) || !memcmp(s1 + 3, "ceild", 5u) )
                          goto LABEL_372;
                        if ( memcmp(s1 + 3, "coshf", 5u) && memcmp(s1 + 3, "coshd", 5u) )
                          return 0;
LABEL_551:
                        v33 = v9;
                        v34 = j_cosh;
                        return sub_14D19F0(v34, v33, v32);
                      case 'e':
                        if ( a2 != (char *)8 || memcmp(s1 + 3, "exp2f", 5u) && memcmp(s1 + 3, "exp2d", 5u) )
                          return 0;
                        return sub_14D1A80(j_pow, v9, 2.0, v32);
                      case 'f':
                        if ( a2 == (char *)8 && (!memcmp(s1 + 3, "fabsf", 5u) || !memcmp(s1 + 3, "fabsd", 5u)) )
                          goto LABEL_224;
                        return 0;
                      case 's':
                        if ( a2 != (char *)8 )
                          return 0;
                        if ( !memcmp(s1 + 3, "sinhf", 5u) || !memcmp(s1 + 3, "sinhd", 5u) )
                        {
LABEL_451:
                          v33 = v9;
                          v34 = j_sinh;
                          return sub_14D19F0(v34, v33, v32);
                        }
                        if ( memcmp(s1 + 3, "sqrtf", 5u) && memcmp(s1 + 3, "sqrtd", 5u) || v32 < 0.0 )
                          return 0;
                        goto LABEL_226;
                      case 't':
                        if ( a2 != (char *)8 || memcmp(s1 + 3, "tanhf", 5u) && memcmp(s1 + 3, "tanhd", 5u) )
                          return 0;
LABEL_469:
                        v33 = v9;
                        v34 = j_tanh;
                        return sub_14D19F0(v34, v33, v32);
                      default:
                        return 0;
                    }
                  }
                  if ( v159 != 53 || a2 != (char *)9 )
                    return 0;
                  v160 = s1 + 3;
                  if ( memcmp(s1 + 3, "floorf", 6u) && memcmp(v160, "floord", 6u) )
                  {
                    if ( memcmp(v160, "log10f", 6u) && memcmp(v160, "log10d", 6u) )
                      return 0;
                    goto LABEL_227;
                  }
LABEL_379:
                  v33 = v9;
                  v34 = (double (__fastcall *)(double))sub_14D13B0;
                  return sub_14D19F0(v34, v33, v32);
                }
                if ( v10 != 30 )
                {
                  if ( v10 == 54 )
                  {
LABEL_236:
                    v33 = v9;
                    v34 = j_exp;
                    return sub_14D19F0(v34, v33, v32);
                  }
                  goto LABEL_238;
                }
LABEL_233:
                v33 = v9;
                v34 = j_cos;
                return sub_14D19F0(v34, v33, v32);
              }
            }
            v33 = v9;
            v34 = j_exp2;
            return sub_14D19F0(v34, v33, v32);
        }
        sub_169D440(&v195, v45);
LABEL_79:
        v14 = sub_159CCF0(*v9, &v194);
LABEL_80:
        sub_127D120(&v195);
        return v14;
      case 13:
        if ( (_DWORD)a3 != 32 )
        {
          if ( (unsigned int)a3 <= 0x20 )
          {
            switch ( (_DWORD)a3 )
            {
              case 6:
                sub_16A85B0(&v194, v14 + 24, a3, a8, a5);
                break;
              case 0xB:
                v70 = v14 + 24;
                v71 = sub_1698260(s1, a2, a3, a8);
                v74 = sub_16982C0(s1, a2, v72, v73);
                if ( v71 == v74 )
                  sub_169D060(&v195, v71, v70);
                else
                  sub_169D050(&v195, v71, v70);
                v75 = *((unsigned __int8 *)v9 + 8);
                v190[0] = 0;
                switch ( v75 )
                {
                  case 0LL:
                  case 6LL:
                    v71 = v74;
                    break;
                  case 1LL:
                    break;
                  case 2LL:
                    v71 = sub_1698270(&v195, v71);
                    break;
                  case 3LL:
                    v71 = sub_1698280(&v195);
                    break;
                  case 4LL:
                    v71 = sub_16982A0();
                    break;
                  case 5LL:
                    v71 = sub_1698290();
                    break;
                }
                sub_16A3360(&v194, v71, 0, v190);
                goto LABEL_79;
              case 5:
                sub_16A8270(&v194, v14 + 24, a3, a8, a5);
                break;
              default:
                return 0;
            }
            v14 = sub_159C0E0(*v9, &v194);
            if ( (unsigned int)v195 <= 0x40 )
              return v14;
            v69 = v194;
            if ( !v194 )
              return v14;
LABEL_111:
            j_j___libc_free_0_0(v69);
            return v14;
          }
          if ( (unsigned int)a3 <= 0xEC3 )
          {
            if ( (unsigned int)a3 > 0xEC1 )
            {
              v124 = *(_DWORD *)(v14 + 32);
              if ( v124 > 0x40 )
              {
                _RSI = (unsigned int)sub_16A57B0(v14 + 24);
              }
              else
              {
                v125 = *(_QWORD *)(v14 + 24);
                v126 = 64;
                if ( v125 )
                {
                  _BitScanReverse64(&v125, v125);
                  v126 = v125 ^ 0x3F;
                }
                _RSI = v124 + v126 - 64;
              }
              return sub_15A0680(v9, _RSI, 0);
            }
            if ( (unsigned int)(a3 - 3637) > 2 )
              return 0;
            v76 = *(_DWORD *)(v14 + 32);
            v77 = *(_QWORD *)(v14 + 24);
            v78 = 1LL << ((unsigned __int8)v76 - 1);
            if ( v76 > 0x40 )
            {
              if ( (*(_QWORD *)(v77 + 8LL * ((v76 - 1) >> 6)) & v78) != 0 )
              {
                LODWORD(v195) = *(_DWORD *)(v14 + 32);
                sub_16A4FD0(&v194, v14 + 24);
                LOBYTE(v76) = v195;
                if ( (unsigned int)v195 > 0x40 )
                {
                  sub_16A8F40(&v194);
                  goto LABEL_143;
                }
                v79 = v194;
LABEL_142:
                v194 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v76) & ~v79;
LABEL_143:
                sub_16A7400(&v194);
                LODWORD(v192) = v195;
                v191 = v194;
                goto LABEL_144;
              }
              LODWORD(v192) = *(_DWORD *)(v14 + 32);
              sub_16A4FD0(&v191, v14 + 24);
            }
            else
            {
              v79 = *(_QWORD *)(v14 + 24);
              if ( (v78 & v77) != 0 )
              {
                LODWORD(v195) = *(_DWORD *)(v14 + 32);
                goto LABEL_142;
              }
              LODWORD(v192) = *(_DWORD *)(v14 + 32);
              v191 = v77;
            }
LABEL_144:
            v14 = sub_159C0E0(*v9, &v191);
            if ( (unsigned int)v192 <= 0x40 )
              return v14;
LABEL_110:
            v69 = v191;
            if ( !v191 )
              return v14;
            goto LABEL_111;
          }
          if ( (unsigned int)(a3 - 4230) > 1 )
            return 0;
        }
        if ( *(_DWORD *)(v14 + 32) > 0x40u )
          v80 = sub_16A5940(v14 + 24);
        else
          v80 = sub_39FAC40(*(_QWORD *)(v14 + 24));
        goto LABEL_149;
    }
    if ( (v27 & 0xFB) != 8 )
      return 0;
    if ( (unsigned int)a3 > 0x1CA7 )
    {
      if ( (unsigned int)(a3 - 7339) > 1 )
        return 0;
    }
    else
    {
      if ( (unsigned int)a3 > 0x1CA5 )
      {
LABEL_200:
        v98 = sub_15A0A60(*a5, 0);
        if ( !v98 || *(_BYTE *)(v98 + 16) != 14 )
          return 0;
        v100 = v98 + 24;
        v101 = (__int64)v9;
        v102 = 0;
        return sub_14D1500(v100, v102, v101, v99);
      }
      if ( (unsigned int)a3 <= 0x1C82 )
      {
        if ( (unsigned int)a3 <= 0x1C80 )
          return 0;
        goto LABEL_200;
      }
      if ( (unsigned int)(a3 - 7301) > 1 )
        return 0;
    }
    v103 = sub_15A0A60(*a5, 0);
    if ( !v103 || *(_BYTE *)(v103 + 16) != 14 )
      return 0;
    v100 = v103 + 24;
    v101 = (__int64)v9;
    v102 = 1;
    return sub_14D1500(v100, v102, v101, v99);
  }
  if ( a6 != 2 )
  {
    if ( a6 != 3 )
      return 0;
    v11 = *a5;
    if ( *(_BYTE *)(*a5 + 16) != 14 )
      return 0;
    v12 = a5[1];
    if ( *(_BYTE *)(v12 + 16) != 14 )
      return 0;
    v13 = a5[2];
    if ( *(_BYTE *)(v13 + 16) != 14 || (unsigned int)(a3 - 99) > 1 )
      return 0;
    v39 = sub_16982C0(s1, a2, a3, a8);
    v40 = v11 + 32;
    xa = v39;
    if ( *(_QWORD *)(v11 + 32) == v39 )
      sub_169C6E0(&v192, v40);
    else
      sub_16986C0(&v192, v40);
    v41 = v192;
    if ( v192 == xa )
    {
      v43 = sub_169F930(&v192, v12 + 32, v13 + 32, 0);
    }
    else
    {
      v42 = sub_1698270(&v192, v40);
      if ( v41 == v42 )
      {
        yd = v42;
        v114 = (_BYTE *)sub_16D40F0(qword_4FBB490);
        if ( v114 ? *v114 : LOBYTE(qword_4FBB490[2]) )
        {
          v116 = v13 + 32;
          if ( xa == *(_QWORD *)(v13 + 32) )
            v116 = *(_QWORD *)(v13 + 40) + 8LL;
          v117 = sub_169D890(v116);
          v118 = v12 + 32;
          if ( xa == *(_QWORD *)(v12 + 32) )
            v118 = *(_QWORD *)(v12 + 40) + 8LL;
          v163 = v117;
          v119 = sub_169D890(v118);
          v120 = &v192;
          v121 = v163;
          if ( xa == v192 )
            v120 = (__int64 *)(v193 + 8);
          v164 = yd;
          yf = v119;
          v122 = sub_169D890(v120);
          v123 = sub_1C40E60(&v189, 1, 1, v122, yf, v121);
          if ( (unsigned int)sub_1C40EE0(&v189) )
          {
            if ( xa == v192 )
              sub_169CAA0(&v192, 0, 0, 0, v164, v123);
            else
              sub_16986F0(&v192, 0, 0, 0, v164, v123);
            v14 = 0;
            goto LABEL_74;
          }
          sub_169D3B0(v190, v123);
          sub_169E320(&v195, v190, v164);
          sub_1698460(v190);
          if ( xa == v192 )
          {
            if ( xa == v195 )
            {
              v157 = v193;
              if ( v193 )
              {
                for ( i = v193 + 32LL * *(_QWORD *)(v193 - 8); v157 != i; sub_127D120((_QWORD *)(i + 8)) )
                  i -= 32;
                j_j_j___libc_free_0_0(v157 - 8);
              }
              goto LABEL_413;
            }
          }
          else if ( xa != v195 )
          {
            sub_16983E0(&v192, &v195);
            goto LABEL_278;
          }
          sub_127D120(&v192);
          if ( xa != v195 )
          {
            sub_1698450(&v192, &v195);
            goto LABEL_278;
          }
LABEL_413:
          sub_169C7E0(&v192, &v195);
LABEL_278:
          if ( xa == v195 )
          {
            v151 = v196;
            if ( v196 )
            {
              v152 = v196 + 32LL * *(_QWORD *)(v196 - 8);
              if ( v196 != v152 )
              {
                do
                {
                  v152 -= 32;
                  if ( xa == *(_QWORD *)(v152 + 8) )
                  {
                    v153 = *(_QWORD *)(v152 + 16);
                    if ( v153 )
                    {
                      v154 = v153 + 32LL * *(_QWORD *)(v153 - 8);
                      while ( v153 != v154 )
                      {
                        v154 -= 32;
                        if ( xa == *(_QWORD *)(v154 + 8) )
                        {
                          v155 = *(_QWORD *)(v154 + 16);
                          if ( v155 )
                          {
                            v156 = v155 + 32LL * *(_QWORD *)(v155 - 8);
                            if ( v155 != v156 )
                            {
                              do
                              {
                                v161 = v155;
                                v165 = v154;
                                yh = v156 - 32;
                                sub_127D120((_QWORD *)(v156 - 24));
                                v156 = yh;
                                v155 = v161;
                                v154 = v165;
                              }
                              while ( v161 != yh );
                            }
                            yi = v154;
                            j_j_j___libc_free_0_0(v155 - 8);
                            v154 = yi;
                          }
                        }
                        else
                        {
                          yg = v154;
                          sub_1698460(v154 + 8);
                          v154 = yg;
                        }
                      }
                      j_j_j___libc_free_0_0(v153 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v152 + 8);
                  }
                }
                while ( v151 != v152 );
              }
              j_j_j___libc_free_0_0(v151 - 8);
            }
          }
          else
          {
            sub_1698460(&v195);
          }
          goto LABEL_73;
        }
      }
      v43 = sub_169DD30(&v192, v12 + 32, v13 + 32, 0);
    }
    v14 = 0;
    if ( v43 == 1 )
    {
LABEL_74:
      sub_127D120(&v192);
      return v14;
    }
LABEL_73:
    v14 = sub_159CCF0(*v9, &v191);
    goto LABEL_74;
  }
  v16 = (_QWORD *)*a5;
  v17 = *(_BYTE *)(*a5 + 16);
  if ( v17 == 14 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a4 + 8) - 1) > 2u )
      return 0;
    v19 = sub_14D1620(v16, (__int64)a2, a3, a8);
    v20 = a5[1];
    v21 = *(_BYTE *)(v20 + 16);
    if ( v21 != 14 )
    {
      if ( v10 != 147 || v21 != 13 )
        return 0;
      v82 = *((_BYTE *)v9 + 8);
      if ( v82 == 1 || v82 == 2 )
      {
        v127 = *(_QWORD **)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) > 0x40u )
          v127 = (_QWORD *)*v127;
        v128 = v19;
        xl = pow(v128, (double)(int)v127);
        v84 = sub_1698270(v16, a2);
        v129 = xl;
        sub_169D3B0(&v191, v129);
      }
      else
      {
        if ( v82 != 3 )
          return 0;
        v83 = *(_QWORD **)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) > 0x40u )
          v83 = (_QWORD *)*v83;
        xk = pow(v19, (double)(int)v83);
        v84 = sub_1698280(v16);
        sub_169D3F0(&v191, xk);
      }
      sub_169E320(&v195, &v191, v84);
      sub_1698460(&v191);
      v14 = sub_159CCF0(*v9, &v194);
      sub_127D120(&v195);
      return v14;
    }
    ya = v19;
    if ( *v16 != *(_QWORD *)v20 )
      return 0;
    v22 = v20;
    v162 = v20;
    v24 = sub_14D1620((_QWORD *)v20, (__int64)a2, v18, v20);
    v25 = ya;
    v26 = v24;
    switch ( v10 )
    {
      case 0x92u:
        goto LABEL_105;
      case 0xDu:
        xi = v162;
        v130 = sub_16982C0(v22, a2, v23, v162);
        v131 = v16 + 4;
        v132 = v130;
        if ( v16[4] == v130 )
        {
          xi = v162;
          sub_169C6E0(&v195, v131);
        }
        else
        {
          sub_16986C0(&v195, v131);
        }
        v133 = &v195;
        if ( v132 == v195 )
          v133 = (__int64 *)(v196 + 8);
        v134 = (*((_BYTE *)v133 + 18) & 8) != 0;
        if ( v132 == *(_QWORD *)(xi + 32) )
          v135 = *(_QWORD *)(xi + 40) + 8LL;
        else
          v135 = xi + 32;
        if ( ((*(_BYTE *)(v135 + 18) & 8) != 0) != v134 )
        {
          if ( v132 == v195 )
            sub_169C8D0(&v195, v24, v24, ya);
          else
            sub_1699490(&v195, v24, v24, ya);
        }
        goto LABEL_79;
      case 0x8Bu:
        v46 = v16[4];
        v136 = sub_16982C0(v22, a2, v23, v162);
        v48 = v162;
        v49 = v136;
        v137 = (__int64)(v16 + 4);
        if ( v46 == v49 )
          v137 = v16[5] + 8LL;
        v51 = *(_QWORD *)(v162 + 32);
        if ( (*(_BYTE *)(v137 + 18) & 7) != 1 )
        {
          if ( v49 == v51 )
LABEL_307:
            v61 = *(_QWORD *)(v48 + 40) + 8LL;
          else
LABEL_96:
            v61 = v48 + 32;
          if ( (*(_BYTE *)(v61 + 18) & 7) != 1 )
          {
            v53 = v16 + 3;
            v54 = (_QWORD *)(v48 + 24);
            v55 = (__int64)v53;
            v56 = v48 + 24;
LABEL_88:
            if ( (unsigned int)sub_14A9E40(v56, v55) )
              v54 = v53;
            v57 = v54 + 1;
            if ( v49 != v54[1] )
              goto LABEL_91;
LABEL_291:
            sub_169C6E0(&v195, v57);
            goto LABEL_79;
          }
          goto LABEL_290;
        }
        break;
      case 0x84u:
LABEL_81:
        v46 = v16[4];
        v47 = sub_16982C0(v22, a2, v23, v162);
        v48 = v162;
        v49 = v47;
        v50 = (__int64)(v16 + 4);
        if ( v46 == v49 )
          v50 = v16[5] + 8LL;
        v51 = *(_QWORD *)(v162 + 32);
        if ( (*(_BYTE *)(v50 + 18) & 7) != 1 )
        {
          v52 = v162 + 32;
          if ( v49 == v51 )
            v52 = *(_QWORD *)(v162 + 40) + 8LL;
          if ( (*(_BYTE *)(v52 + 18) & 7) != 1 )
          {
            v53 = v16 + 3;
            v54 = (_QWORD *)(v162 + 24);
            v55 = v162 + 24;
            v56 = (__int64)v53;
            goto LABEL_88;
          }
LABEL_290:
          v57 = v16 + 4;
          if ( v46 == v49 )
            goto LABEL_291;
LABEL_91:
          sub_16986C0(&v195, v57);
          goto LABEL_79;
        }
        break;
      default:
        if ( !a7 )
          return 0;
        switch ( v10 )
        {
          case 0xF68u:
          case 0xF6Au:
          case 0xF6Cu:
            goto LABEL_81;
          case 0xF6Eu:
          case 0xF70u:
          case 0xF72u:
            v46 = v16[4];
            v58 = sub_16982C0(v22, a2, v23, v162);
            v48 = v162;
            v49 = v58;
            v59 = (__int64)(v16 + 4);
            if ( v46 == v49 )
              v59 = v16[5] + 8LL;
            v60 = *(_QWORD *)(v162 + 32);
            if ( (*(_BYTE *)(v59 + 18) & 7) != 1 )
            {
              if ( v60 == v49 )
                goto LABEL_307;
              goto LABEL_96;
            }
            v57 = (_QWORD *)(v162 + 32);
            if ( v60 != v49 )
              goto LABEL_91;
            goto LABEL_291;
          default:
            if ( a2 == (char *)3 )
            {
              if ( *(_WORD *)s1 != 28528 || s1[2] != 119 )
                return 0;
              break;
            }
            if ( a2 != (char *)4 )
            {
              if ( a2 == (char *)12 )
              {
                yc = v24;
                v62 = 12;
                a2 = "__pow_finite";
                xc = v25;
LABEL_104:
                v25 = xc;
                v26 = yc;
                if ( memcmp(s1, a2, v62) )
                  return 0;
                break;
              }
              if ( a2 == (char *)13 )
              {
                yc = v24;
                v62 = 13;
                a2 = "__powf_finite";
                xc = v25;
                goto LABEL_104;
              }
              if ( a2 == (char *)5 )
              {
                v26 = v24;
                if ( memcmp(s1, "fmodf", 5u) )
                {
                  ye = v24;
                  v144 = 5;
                  v145 = "atan2";
                  xj = v25;
                  goto LABEL_352;
                }
              }
              else
              {
                if ( a2 == (char *)6 )
                {
                  ye = v24;
                  v144 = 6;
                  v145 = "atan2f";
                  xj = v25;
                  goto LABEL_352;
                }
                if ( a2 == (char *)14 )
                {
                  ye = v24;
                  v144 = 14;
                  v145 = "__atan2_finite";
                  xj = v25;
                  goto LABEL_352;
                }
                if ( a2 == (char *)15 )
                {
                  ye = v24;
                  v144 = 15;
                  v145 = "__atan2f_finite";
                  xj = v25;
                  goto LABEL_352;
                }
                if ( s1[1] != 90 || (unsigned __int64)a2 <= 6 )
                  return 0;
                if ( a2 == (char *)8 )
                {
                  if ( *(_QWORD *)s1 != 0x6666776F70335A5FLL && *(_QWORD *)s1 != 0x6464776F70335A5FLL )
                    return 0;
                  v146 = v9;
                  v147 = ya;
                  v148 = j_pow;
                  return sub_14D1A80(v148, v146, v147, v26);
                }
                if ( a2 != (char *)9 )
                {
                  if ( a2 != (char *)10 )
                    return 0;
                  ye = v24;
                  xj = v25;
                  v26 = v24;
                  if ( !memcmp(s1, "_Z5atan2ff", 0xAu) )
                    goto LABEL_353;
                  v144 = 10;
                  v145 = "_Z5atan2dd";
LABEL_352:
                  v25 = xj;
                  v26 = ye;
                  if ( memcmp(s1, v145, v144) )
                    return 0;
LABEL_353:
                  v146 = v9;
                  v147 = v25;
                  v148 = j_atan2;
                  return sub_14D1A80(v148, v146, v147, v26);
                }
                v26 = v24;
                if ( memcmp(s1, "_Z4fmodff", 9u) )
                {
                  v26 = v24;
                  if ( memcmp(s1, "_Z4fmoddd", 9u) )
                    return 0;
                }
              }
LABEL_357:
              v146 = v9;
              v147 = ya;
              v148 = j_fmod;
              return sub_14D1A80(v148, v146, v147, v26);
            }
            if ( *(_DWORD *)s1 != 1719103344 )
            {
              if ( *(_DWORD *)s1 != 1685024102 )
                return 0;
              goto LABEL_357;
            }
            break;
        }
LABEL_105:
        feclearexcept(61);
        v63 = __errno_location();
        *v63 = 0;
        v64 = v63;
        xd = pow(v25, v26);
        if ( (unsigned int)(*v64 - 33) > 1 && !fetestexcept(29) )
          return sub_14D17B0(v9, (__int64)a2, xd);
        feclearexcept(61);
        *v64 = 0;
        return 0;
    }
    v57 = (_QWORD *)(v48 + 32);
    if ( v49 == v51 )
      goto LABEL_291;
    goto LABEL_91;
  }
  if ( v17 != 13 )
    return 0;
  v35 = a5[1];
  if ( *(_BYTE *)(v35 + 16) != 13 )
    return 0;
  if ( (unsigned int)a3 <= 0x1013 )
  {
    if ( (unsigned int)a3 <= 0x1011 )
    {
      if ( (_DWORD)a3 == 33 )
      {
        v138 = *(_DWORD *)(v35 + 32);
        if ( v138 <= 0x40 )
          v139 = *(_QWORD *)(v35 + 24) == 1;
        else
          v139 = v138 - 1 == (unsigned int)sub_16A57B0(v35 + 24);
        v140 = *((_DWORD *)v16 + 8);
        if ( !v139 || (v140 <= 0x40 ? (v141 = v16[3] == 0) : (v141 = (unsigned int)sub_16A57B0(v16 + 3) == v140), !v141) )
        {
          if ( v140 > 0x40 )
          {
            _RSI = (unsigned int)sub_16A58A0(v16 + 3);
          }
          else
          {
            _RAX = v16[3];
            __asm { tzcnt   rsi, rax }
            if ( !_RAX )
              LODWORD(_RSI) = 64;
            if ( v140 < (unsigned int)_RSI )
              LODWORD(_RSI) = v140;
            _RSI = (unsigned int)_RSI;
          }
          return sub_15A0680(v9, _RSI, 0);
        }
      }
      else
      {
        if ( (unsigned int)a3 > 0x21 )
        {
          if ( (unsigned int)a3 <= 0xD3 )
          {
            if ( (unsigned int)a3 <= 0xBC || ((1LL << ((unsigned __int8)a3 + 67)) & 0x700241) == 0 )
              return 0;
            LODWORD(v192) = 1;
            v191 = 0;
            switch ( (int)a3 )
            {
              case 189:
                sub_16A7290(&v194, v16 + 3, v35 + 24, v190);
                break;
              case 195:
                sub_16AA420(&v194, v16 + 3, v35 + 24, v190);
                break;
              case 198:
                sub_16A7620(&v194, v16 + 3, v35 + 24, v190);
                break;
              case 209:
                sub_16A99B0(&v194, v16 + 3, v35 + 24, v190);
                break;
              case 210:
                sub_16AA580(&v194, v16 + 3, v35 + 24, v190);
                break;
              case 211:
                sub_16A9930(&v194, v16 + 3, v35 + 24, v190);
                break;
              default:
                ++*(_DWORD *)(a4 + 16);
                BUG();
            }
            sub_14D15E0((__int64 *)&v191, (__int64 *)&v194);
            sub_135E100((__int64 *)&v194);
            v65 = sub_159C0E0(*v9, &v191);
            v66 = *v9;
            v67 = v190[0];
            v194 = v65;
            v68 = sub_1643320(v66);
            v195 = sub_159C470(v68, v67, 0);
            v14 = sub_159F090(v9, &v194, 2);
            if ( (unsigned int)v192 <= 0x40 )
              return v14;
            goto LABEL_110;
          }
          if ( (_DWORD)a3 != 3811 )
            return 0;
          v85 = *(_DWORD *)(v35 + 32);
          if ( v85 <= 0x40 )
          {
            if ( *(_QWORD *)(v35 + 24) )
              return 0;
          }
          else if ( v85 != (unsigned int)sub_16A57B0(v35 + 24) )
          {
            return 0;
          }
          return sub_15A06D0(v9);
        }
        if ( (_DWORD)a3 != 31 )
          return 0;
        v86 = *(_DWORD *)(v35 + 32);
        if ( v86 <= 0x40 )
          v87 = *(_QWORD *)(v35 + 24) == 1;
        else
          v87 = v86 - 1 == (unsigned int)sub_16A57B0(v35 + 24);
        v88 = *((_DWORD *)v16 + 8);
        if ( !v87 )
        {
          if ( v88 > 0x40 )
            goto LABEL_170;
          goto LABEL_367;
        }
        if ( v88 > 0x40 )
        {
          if ( (unsigned int)sub_16A57B0(v16 + 3) != v88 )
          {
LABEL_170:
            v80 = sub_16A57B0(v16 + 3);
LABEL_149:
            _RSI = v80;
            return sub_15A0680(v9, _RSI, 0);
          }
          return sub_1599EF0(v9);
        }
        if ( v16[3] )
        {
LABEL_367:
          v149 = v16[3];
          v150 = 64;
          if ( v149 )
          {
            _BitScanReverse64(&v149, v149);
            v150 = v149 ^ 0x3F;
          }
          _RSI = v88 + v150 - 64;
          return sub_15A0680(v9, _RSI, 0);
        }
      }
      return sub_1599EF0(v9);
    }
    v143 = (_QWORD *)(v35 + 24);
    v90 = v16 + 3;
    if ( (int)sub_16AEA10(v90, v35 + 24) <= 0 )
      v90 = v143;
    return sub_159C0E0(*v9, v90);
  }
  if ( (unsigned int)a3 <= 0x104C )
  {
    if ( (unsigned int)a3 > 0x104A )
    {
      v96 = (_QWORD *)(v35 + 24);
      v90 = v16 + 3;
      v97 = sub_16A9900(v90, v35 + 24);
    }
    else
    {
      if ( (unsigned int)a3 <= 0x1017 )
      {
        if ( (unsigned int)a3 <= 0x1015 )
          return 0;
        v89 = (_QWORD *)(v35 + 24);
        v90 = v16 + 3;
        if ( (int)sub_16A9900(v90, v35 + 24) <= 0 )
          v90 = v89;
        return sub_159C0E0(*v9, v90);
      }
      if ( (unsigned int)(a3 - 4167) > 1 )
        return 0;
      v96 = (_QWORD *)(v35 + 24);
      v90 = v16 + 3;
      v97 = sub_16AEA10(v90, v35 + 24);
    }
    if ( v97 >= 0 )
      v90 = v96;
    return sub_159C0E0(*v9, v90);
  }
  if ( (_DWORD)a3 != 5293 && (_DWORD)a3 != 5300 )
    return 0;
  v91 = (_QWORD *)v16[3];
  if ( *((_DWORD *)v16 + 8) > 0x40u )
    v91 = (_QWORD *)*v91;
  if ( (unsigned int)v91 > 2 || *(_DWORD *)(v35 + 32) != 1 )
    return 0;
  if ( v10 != 5300 )
  {
    if ( !*(_QWORD *)(v35 + 24) )
    {
      if ( *(_BYTE *)(a4 + 8) != 13 )
        BUG();
      v92 = *(__int64 **)(a4 + 16);
      v93 = *v92;
      v94 = v92[1];
      if ( (_DWORD)v91 == 2 )
        v95 = sub_15A0600(v94);
      else
        v95 = sub_15A0640(v94);
      v195 = v95;
      v194 = sub_15A0680(v93, 0, 0);
      return sub_159F090(v9, &v194, 2);
    }
    return 0;
  }
  if ( *(_BYTE *)(a4 + 8) != 11 )
    v9 = 0;
  if ( (_DWORD)v91 == 2 )
    return sub_15A0600(v9);
  else
    return sub_159C470(v9, *(_QWORD *)(v35 + 24), 0);
}
