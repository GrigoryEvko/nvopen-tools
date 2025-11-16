// Function: sub_21F6F00
// Address: 0x21f6f00
//
__int64 __fastcall sub_21F6F00(
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
  _QWORD *v11; // rbx
  _QWORD *v12; // rdx
  int v13; // r8d
  int v14; // r9d
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 *v21; // rsi
  __int64 *v22; // rcx
  _QWORD *v23; // r15
  __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  int v26; // r8d
  int v27; // r9d
  unsigned __int8 v28; // cl
  __int64 v29; // r14
  unsigned __int64 *v30; // r13
  unsigned __int64 *v31; // r12
  unsigned __int64 v32; // rdx
  __int64 i; // r13
  __int64 v34; // rdi
  unsigned __int64 *v35; // rdi
  unsigned int v36; // eax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // r15
  __int64 v42; // rbx
  char v43; // al
  __int64 v44; // rax
  _BYTE *v45; // rsi
  unsigned __int64 v46; // rdi
  __int64 result; // rax
  __int64 v48; // r15
  char v49; // dl
  __int64 *v50; // rax
  __int64 *v51; // rsi
  __int64 *v52; // rcx
  __int64 v53; // rax
  unsigned __int64 v54; // r15
  unsigned __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r14
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // r14
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // rax
  __int64 v71; // rax
  unsigned int v72; // r15d
  __int64 v73; // rax
  __int64 v74; // rsi
  char v75; // al
  unsigned __int64 *v76; // rsi
  unsigned __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r14
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r14
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r13
  __int64 v92; // rax
  unsigned __int64 *v93; // rsi
  unsigned int v94; // ecx
  __int64 v95; // rax
  __int64 v96; // rsi
  char v97; // al
  unsigned __int64 *v98; // rsi
  __int64 v99; // rsi
  unsigned __int64 *v100; // r14
  unsigned __int64 *v101; // r13
  double v102; // xmm4_8
  double v103; // xmm5_8
  unsigned __int64 v104; // r8
  __int64 v105; // r14
  unsigned __int64 *v106; // r13
  double v107; // xmm4_8
  double v108; // xmm5_8
  __int64 j; // r14
  void *v110; // rdi
  _QWORD *v111; // r13
  unsigned int v112; // r14d
  __int64 v113; // r15
  char v114; // dl
  __int64 *v115; // rax
  __int64 *v116; // rsi
  __int64 *v117; // rcx
  __int64 v118; // rax
  __int64 v119; // [rsp+8h] [rbp-318h]
  unsigned int v120; // [rsp+10h] [rbp-310h]
  _QWORD *v122; // [rsp+28h] [rbp-2F8h]
  __int64 v123; // [rsp+28h] [rbp-2F8h]
  unsigned __int64 v124; // [rsp+38h] [rbp-2E8h] BYREF
  __int64 v125; // [rsp+40h] [rbp-2E0h] BYREF
  _BYTE *v126; // [rsp+48h] [rbp-2D8h]
  _BYTE *v127; // [rsp+50h] [rbp-2D0h]
  unsigned __int64 *v128; // [rsp+60h] [rbp-2C0h] BYREF
  unsigned __int64 *v129; // [rsp+68h] [rbp-2B8h]
  unsigned __int64 *v130; // [rsp+70h] [rbp-2B0h]
  unsigned __int64 *v131; // [rsp+80h] [rbp-2A0h] BYREF
  unsigned __int64 *v132; // [rsp+88h] [rbp-298h] BYREF
  unsigned __int64 *v133; // [rsp+90h] [rbp-290h]
  unsigned __int64 **v134; // [rsp+98h] [rbp-288h]
  unsigned __int64 **v135; // [rsp+A0h] [rbp-280h]
  __int64 v136; // [rsp+A8h] [rbp-278h]
  _QWORD *v137; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v138; // [rsp+B8h] [rbp-268h]
  _QWORD v139[32]; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v140; // [rsp+1C0h] [rbp-160h] BYREF
  __int64 *v141; // [rsp+1C8h] [rbp-158h] BYREF
  void *s; // [rsp+1D0h] [rbp-150h]
  __int128 v143; // [rsp+1D8h] [rbp-148h]
  _QWORD v144[39]; // [rsp+1E8h] [rbp-138h] BYREF

  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, a2);
    v11 = *(_QWORD **)(a2 + 88);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      sub_15E08E0(a2, a2);
    v12 = *(_QWORD **)(a2 + 88);
  }
  else
  {
    v11 = *(_QWORD **)(a2 + 88);
    v12 = v11;
  }
  v122 = &v12[5 * *(_QWORD *)(a2 + 96)];
  if ( v122 != v11 )
  {
    while ( 1 )
    {
      while ( *(_BYTE *)(*v11 + 8LL) != 15 || !(unsigned __int8)sub_15E04B0((__int64)v11) )
      {
LABEL_5:
        v11 += 5;
        if ( v11 == v122 )
          goto LABEL_49;
      }
      v140 = 0;
      v137 = v139;
      v138 = 0x2000000000LL;
      v17 = v144;
      v141 = v144;
      v18 = v144;
      s = v144;
      *(_QWORD *)&v143 = 32;
      DWORD2(v143) = 0;
      v128 = 0;
      v129 = 0;
      v130 = 0;
      v19 = v11[1];
      if ( v19 )
        break;
      v111 = v139;
      v112 = 0;
LABEL_25:
      v23 = &v111[v112];
      while ( 2 )
      {
        if ( !v112 )
        {
          LODWORD(v132) = 0;
          v133 = 0;
          v134 = &v132;
          v135 = &v132;
          v136 = 0;
          if ( v128 == v129 )
          {
            v34 = 0;
          }
          else
          {
            v29 = a1;
            v30 = v129;
            v31 = v128;
            do
            {
              v32 = *v31++;
              sub_21F3A20(*(unsigned __int8 **)(v29 + 216), a2, v32, &v131, a3, a4, a5, a6, v15, v16, a9, a10);
            }
            while ( v30 != v31 );
            a1 = v29;
            for ( i = (__int64)v134; (unsigned __int64 **)i != &v132; i = sub_220EF30(i) )
              sub_15F20C0(*(_QWORD **)(i + 32));
            v34 = (__int64)v133;
          }
          sub_21F25B0(v34);
          v133 = 0;
          v134 = &v132;
          v135 = &v132;
          v136 = 0;
          sub_21F25B0(0);
          goto LABEL_35;
        }
        v24 = *(v23 - 1);
        LODWORD(v138) = --v112;
        v25 = (unsigned __int64)sub_1648700(v24);
        v28 = *(_BYTE *)(v25 + 16);
        switch ( v28 )
        {
          case 0x19u:
          case 0x4Bu:
            --v23;
            continue;
          case 0x1Du:
          case 0x4Eu:
            if ( v28 <= 0x17u )
            {
              v54 = 0;
              v55 = 0;
LABEL_99:
              v56 = *(_QWORD *)(v54 - 72);
              if ( *(_BYTE *)(v56 + 16) || (v54 = v55 & 0xFFFFFFFFFFFFFFF8LL, *(_DWORD *)(v56 + 36) != 3660) )
              {
                if ( (unsigned __int8)sub_1560260((_QWORD *)(v54 + 56), -1, 36) )
                  goto LABEL_95;
                if ( *(char *)(v54 + 23) >= 0 )
                  goto LABEL_264;
                v57 = sub_1648A40(v54);
                v59 = v57 + v58;
                v60 = 0;
                if ( *(char *)(v54 + 23) < 0 )
                  v60 = sub_1648A40(v54);
                if ( !(unsigned int)((v59 - v60) >> 4) )
                {
LABEL_264:
                  v61 = *(_QWORD *)(v54 - 72);
                  if ( !*(_BYTE *)(v61 + 16) )
                  {
                    v131 = *(unsigned __int64 **)(v61 + 112);
                    if ( (unsigned __int8)sub_1560260(&v131, -1, 36) )
                      goto LABEL_95;
                  }
                }
                if ( (unsigned __int8)sub_1560260((_QWORD *)(v54 + 56), -1, 36) )
                  goto LABEL_95;
                if ( *(char *)(v54 + 23) >= 0 )
                  goto LABEL_265;
                v62 = sub_1648A40(v54);
                v64 = v62 + v63;
                v65 = 0;
                if ( *(char *)(v54 + 23) < 0 )
                  v65 = sub_1648A40(v54);
                if ( !(unsigned int)((v64 - v65) >> 4) )
                {
LABEL_265:
                  v66 = *(_QWORD *)(v54 - 72);
                  if ( !*(_BYTE *)(v66 + 16) )
                  {
                    v131 = *(unsigned __int64 **)(v66 + 112);
                    if ( (unsigned __int8)sub_1560260(&v131, -1, 36) )
                      goto LABEL_95;
                  }
                }
                if ( (unsigned __int8)sub_1560260((_QWORD *)(v54 + 56), -1, 37) )
                  goto LABEL_95;
                if ( *(char *)(v54 + 23) < 0 )
                {
                  v67 = sub_1648A40(v54);
                  v69 = v67 + v68;
                  v70 = *(char *)(v54 + 23) >= 0 ? 0LL : sub_1648A40(v54);
                  if ( v70 != v69 )
                  {
                    while ( *(_DWORD *)(*(_QWORD *)v70 + 8LL) <= 1u )
                    {
                      v70 += 16;
                      if ( v69 == v70 )
                        goto LABEL_122;
                    }
                    goto LABEL_35;
                  }
                }
LABEL_122:
                v71 = *(_QWORD *)(v54 - 72);
                if ( *(_BYTE *)(v71 + 16) )
                  goto LABEL_35;
                goto LABEL_123;
              }
              goto LABEL_226;
            }
            if ( v28 == 78 )
            {
              v77 = v25 | 4;
              v55 = v25 | 4;
            }
            else
            {
              v54 = 0;
              v55 = 0;
              if ( v28 != 29 )
                goto LABEL_99;
              v77 = v25 & 0xFFFFFFFFFFFFFFFBLL;
              v55 = v25 & 0xFFFFFFFFFFFFFFFBLL;
            }
            v54 = v77 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v77 & 4) == 0 )
              goto LABEL_99;
            v78 = *(_QWORD *)(v54 - 24);
            if ( !*(_BYTE *)(v78 + 16) && *(_DWORD *)(v78 + 36) == 3660 )
            {
LABEL_226:
              v113 = *(_QWORD *)(v25 + 8);
              if ( v113 )
              {
                while ( 1 )
                {
                  v115 = v141;
                  if ( s == v141 )
                  {
                    v116 = &v141[DWORD1(v143)];
                    if ( v141 != v116 )
                    {
                      v117 = 0;
                      while ( *v115 != v113 )
                      {
                        if ( *v115 == -2 )
                          v117 = v115;
                        if ( v116 == ++v115 )
                        {
                          if ( !v117 )
                            goto LABEL_244;
                          *v117 = v113;
                          --DWORD2(v143);
                          ++v140;
                          goto LABEL_239;
                        }
                      }
                      goto LABEL_229;
                    }
LABEL_244:
                    if ( DWORD1(v143) < (unsigned int)v143 )
                      break;
                  }
                  sub_16CCBA0((__int64)&v140, v113);
                  if ( v114 )
                    goto LABEL_239;
LABEL_229:
                  v113 = *(_QWORD *)(v113 + 8);
                  if ( !v113 )
                    goto LABEL_95;
                }
                ++DWORD1(v143);
                *v116 = v113;
                ++v140;
LABEL_239:
                v118 = (unsigned int)v138;
                if ( (unsigned int)v138 >= HIDWORD(v138) )
                {
                  sub_16CD150((__int64)&v137, v139, 0, 8, v26, v27);
                  v118 = (unsigned int)v138;
                }
                v137[v118] = v113;
                LODWORD(v138) = v138 + 1;
                goto LABEL_229;
              }
              goto LABEL_25;
            }
            if ( (unsigned __int8)sub_1560260((_QWORD *)(v54 + 56), -1, 36) )
              goto LABEL_95;
            if ( *(char *)(v54 + 23) >= 0 )
              goto LABEL_266;
            v79 = sub_1648A40(v54);
            v81 = v79 + v80;
            v82 = 0;
            if ( *(char *)(v54 + 23) < 0 )
              v82 = sub_1648A40(v54);
            if ( !(unsigned int)((v81 - v82) >> 4) )
            {
LABEL_266:
              v83 = *(_QWORD *)(v54 - 24);
              if ( !*(_BYTE *)(v83 + 16) )
              {
                v131 = *(unsigned __int64 **)(v83 + 112);
                if ( (unsigned __int8)sub_1560260(&v131, -1, 36) )
                  goto LABEL_95;
              }
            }
            if ( (unsigned __int8)sub_1560260((_QWORD *)(v54 + 56), -1, 36) )
              goto LABEL_95;
            if ( *(char *)(v54 + 23) >= 0 )
              goto LABEL_267;
            v84 = sub_1648A40(v54);
            v86 = v84 + v85;
            v87 = 0;
            if ( *(char *)(v54 + 23) < 0 )
              v87 = sub_1648A40(v54);
            if ( !(unsigned int)((v86 - v87) >> 4) )
            {
LABEL_267:
              v88 = *(_QWORD *)(v54 - 24);
              if ( !*(_BYTE *)(v88 + 16) )
              {
                v131 = *(unsigned __int64 **)(v88 + 112);
                if ( (unsigned __int8)sub_1560260(&v131, -1, 36) )
                  goto LABEL_95;
              }
            }
            if ( (unsigned __int8)sub_1560260((_QWORD *)(v54 + 56), -1, 37) )
              goto LABEL_95;
            if ( *(char *)(v54 + 23) >= 0
              || ((v89 = sub_1648A40(v54), v91 = v89 + v90, *(char *)(v54 + 23) >= 0)
                ? (v92 = 0)
                : (v92 = sub_1648A40(v54)),
                  v92 == v91) )
            {
LABEL_164:
              v71 = *(_QWORD *)(v54 - 24);
              if ( *(_BYTE *)(v71 + 16) )
                goto LABEL_35;
LABEL_123:
              v131 = *(unsigned __int64 **)(v71 + 112);
              if ( !(unsigned __int8)sub_1560260(&v131, -1, 37) )
                goto LABEL_35;
LABEL_95:
              v111 = v137;
              v112 = v138;
              goto LABEL_25;
            }
            while ( *(_DWORD *)(*(_QWORD *)v92 + 8LL) <= 1u )
            {
              v92 += 16;
              if ( v91 == v92 )
                goto LABEL_164;
            }
LABEL_35:
            v35 = v128;
            if ( v128 != v129 )
              v129 = v128;
            ++v140;
            LODWORD(v138) = 0;
            if ( s == v141 )
              goto LABEL_42;
            v36 = 4 * (DWORD1(v143) - DWORD2(v143));
            if ( v36 < 0x20 )
              v36 = 32;
            if ( v36 >= (unsigned int)v143 )
            {
              memset(s, -1, 8LL * (unsigned int)v143);
              v35 = v128;
LABEL_42:
              *(_QWORD *)((char *)&v143 + 4) = 0;
              goto LABEL_43;
            }
            sub_16CC920((__int64)&v140);
            v35 = v128;
LABEL_43:
            if ( v35 )
              j_j___libc_free_0(v35, (char *)v130 - (char *)v35);
            if ( s != v141 )
              _libc_free((unsigned __int64)s);
            if ( v137 == v139 )
              goto LABEL_5;
            _libc_free((unsigned __int64)v137);
            v11 += 5;
            if ( v11 == v122 )
              goto LABEL_49;
            break;
          case 0x36u:
            v131 = (unsigned __int64 *)v25;
            v72 = *(unsigned __int16 *)(v25 + 18);
            if ( (v72 & 1) != 0 )
              goto LABEL_25;
            v73 = **(_QWORD **)(v25 - 24);
            if ( *(_BYTE *)(v73 + 8) != 15 || *(_DWORD *)(v73 + 8) >> 8 != 1 )
              goto LABEL_25;
            v74 = *(_QWORD *)v25;
            v75 = *(_BYTE *)(*(_QWORD *)v25 + 8LL);
            if ( v75 == 11 )
            {
              v119 = *(_QWORD *)v25;
              if ( (unsigned int)sub_1643030(*(_QWORD *)v25) > 0x40 )
                goto LABEL_25;
            }
            else
            {
              if ( (unsigned __int8)(v75 - 1) <= 5u || v75 == 13 )
              {
LABEL_132:
                if ( 1 << (v72 >> 1) >> 1 >= (unsigned int)sub_15A9FE0(*(_QWORD *)(a1 + 216), v74) )
                {
                  v76 = v129;
                  if ( v129 == v130 )
                  {
                    sub_14147F0((__int64)&v128, v129, &v131);
                  }
                  else
                  {
                    if ( v129 )
                    {
                      *v129 = (unsigned __int64)v131;
                      v76 = v129;
                    }
                    v129 = v76 + 1;
                  }
                }
                goto LABEL_95;
              }
              if ( v75 != 16 )
              {
                if ( v75 != 15 )
                  goto LABEL_25;
                goto LABEL_132;
              }
              v119 = *(_QWORD *)v25;
              if ( (unsigned int)sub_1643030(*(_QWORD *)(v74 + 24)) <= 7 )
                goto LABEL_25;
            }
            v74 = v119;
            goto LABEL_132;
          case 0x38u:
          case 0x47u:
          case 0x48u:
          case 0x4Du:
          case 0x4Fu:
            v48 = *(_QWORD *)(v25 + 8);
            if ( !v48 )
              goto LABEL_25;
            while ( 1 )
            {
              v50 = v141;
              if ( s != v141 )
                break;
              v51 = &v141[DWORD1(v143)];
              if ( v141 != v51 )
              {
                v52 = 0;
                while ( *v50 != v48 )
                {
                  if ( *v50 == -2 )
                    v52 = v50;
                  if ( v51 == ++v50 )
                  {
                    if ( !v52 )
                      goto LABEL_199;
                    *v52 = v48;
                    --DWORD2(v143);
                    ++v140;
                    goto LABEL_93;
                  }
                }
LABEL_83:
                v48 = *(_QWORD *)(v48 + 8);
                if ( !v48 )
                  goto LABEL_95;
                continue;
              }
LABEL_199:
              if ( DWORD1(v143) >= (unsigned int)v143 )
                break;
              ++DWORD1(v143);
              *v51 = v48;
              v53 = (unsigned int)v138;
              ++v140;
              if ( (unsigned int)v138 >= HIDWORD(v138) )
                goto LABEL_201;
LABEL_94:
              v137[v53] = v48;
              LODWORD(v138) = v138 + 1;
              v48 = *(_QWORD *)(v48 + 8);
              if ( !v48 )
                goto LABEL_95;
            }
            sub_16CCBA0((__int64)&v140, v48);
            if ( !v49 )
              goto LABEL_83;
LABEL_93:
            v53 = (unsigned int)v138;
            if ( (unsigned int)v138 < HIDWORD(v138) )
              goto LABEL_94;
LABEL_201:
            sub_16CD150((__int64)&v137, v139, 0, 8, v26, v27);
            v53 = (unsigned int)v138;
            goto LABEL_94;
          default:
            goto LABEL_35;
        }
        break;
      }
    }
    while ( 1 )
    {
      if ( v17 != v18 )
        goto LABEL_10;
      v21 = &v17[DWORD1(v143)];
      if ( v17 != v21 )
      {
        v22 = 0;
        do
        {
          if ( *v17 == v19 )
            goto LABEL_11;
          if ( *v17 == -2 )
            v22 = v17;
          ++v17;
        }
        while ( v21 != v17 );
        if ( v22 )
        {
          *v22 = v19;
          v20 = (unsigned int)v138;
          --DWORD2(v143);
          ++v140;
          if ( (unsigned int)v138 < HIDWORD(v138) )
            goto LABEL_12;
          goto LABEL_23;
        }
      }
      if ( DWORD1(v143) < (unsigned int)v143 )
      {
        ++DWORD1(v143);
        *v21 = v19;
        ++v140;
      }
      else
      {
LABEL_10:
        sub_16CCBA0((__int64)&v140, v19);
      }
LABEL_11:
      v20 = (unsigned int)v138;
      if ( (unsigned int)v138 < HIDWORD(v138) )
        goto LABEL_12;
LABEL_23:
      sub_16CD150((__int64)&v137, v139, 0, 8, v13, v14);
      v20 = (unsigned int)v138;
LABEL_12:
      v137[v20] = v19;
      v112 = v138 + 1;
      LODWORD(v138) = v138 + 1;
      v19 = *(_QWORD *)(v19 + 8);
      if ( !v19 )
      {
        v111 = v137;
        goto LABEL_25;
      }
      v18 = (__int64 *)s;
      v17 = v141;
    }
  }
LABEL_49:
  v126 = 0;
  v37 = *(__int64 **)(a1 + 8);
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v137 = 0;
  v138 = 0;
  v139[0] = 0;
  v125 = 0;
  v38 = *v37;
  v39 = v37[1];
  if ( v38 == v39 )
LABEL_261:
    BUG();
  while ( *(_UNKNOWN **)v38 != &unk_4F9A488 )
  {
    v38 += 16;
    if ( v39 == v38 )
      goto LABEL_261;
  }
  *(_QWORD *)(a1 + 208) = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(
                                        *(_QWORD *)(v38 + 8),
                                        &unk_4F9A488)
                                    + 160);
  v40 = *(_QWORD *)(a2 + 80);
  if ( v40 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v40 )
        BUG();
      v41 = *(_QWORD *)(v40 + 24);
      v42 = v40 + 16;
      if ( v41 != v40 + 16 )
        break;
LABEL_69:
      v40 = *(_QWORD *)(v40 + 8);
      if ( a2 + 72 == v40 )
        goto LABEL_70;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v41 )
          BUG();
        v43 = *(_BYTE *)(v41 - 8);
        if ( v43 == 78 )
        {
          v124 = v41 - 24;
          if ( (unsigned __int8)sub_21F2100(v41 - 24) )
          {
            v93 = v129;
            if ( v129 == v130 )
            {
              sub_187FFB0((__int64)&v128, v129, &v124);
            }
            else
            {
              if ( v129 )
              {
                *v129 = v124;
                v93 = v129;
              }
              v129 = v93 + 1;
            }
          }
          goto LABEL_58;
        }
        v124 = 0;
        v140 = 0;
        if ( v43 != 55 )
          break;
        v44 = *(_QWORD *)(v41 - 48);
        v140 = v44;
LABEL_63:
        if ( !v44 || *(_DWORD *)(*(_QWORD *)v44 + 8LL) > 0x1FFu )
          goto LABEL_58;
        v45 = v126;
        if ( v126 == v127 )
        {
          sub_1287830((__int64)&v125, v126, &v140);
          goto LABEL_58;
        }
        if ( v126 )
        {
          *(_QWORD *)v126 = v44;
          v45 = v126;
        }
        v126 = v45 + 8;
        v41 = *(_QWORD *)(v41 + 8);
        if ( v42 == v41 )
          goto LABEL_69;
      }
      switch ( v43 )
      {
        case ';':
          v44 = *(_QWORD *)(v41 - 72);
          v140 = v44;
          goto LABEL_63;
        case ':':
          v44 = *(_QWORD *)(v41 - 96);
          v140 = v44;
          goto LABEL_63;
        case '6':
          v140 = v41 - 24;
          v94 = *(unsigned __int16 *)(v41 - 6);
          if ( (v94 & 1) == 0 )
          {
            v95 = **(_QWORD **)(v41 - 48);
            if ( *(_BYTE *)(v95 + 8) == 15 && *(_DWORD *)(v95 + 8) >> 8 == 1 )
            {
              v96 = *(_QWORD *)(v41 - 24);
              v97 = *(_BYTE *)(v96 + 8);
              if ( v97 == 11 )
              {
                v120 = *(unsigned __int16 *)(v41 - 6);
                v123 = *(_QWORD *)(v41 - 24);
                if ( (unsigned int)sub_1643030(v123) <= 0x40 )
                  goto LABEL_191;
              }
              else
              {
                if ( (unsigned __int8)(v97 - 1) <= 5u || v97 == 13 )
                  goto LABEL_182;
                if ( v97 == 16 )
                {
                  v120 = *(unsigned __int16 *)(v41 - 6);
                  v123 = *(_QWORD *)(v41 - 24);
                  if ( (unsigned int)sub_1643030(*(_QWORD *)(v96 + 24)) > 7 )
                  {
LABEL_191:
                    v96 = v123;
                    v94 = v120;
LABEL_182:
                    if ( 1 << (v94 >> 1) >> 1 >= (unsigned int)sub_15A9FE0(*(_QWORD *)(a1 + 216), v96) )
                    {
                      if ( (unsigned __int8)sub_21F6E60((_QWORD *)(a1 + 160), v140, *(_QWORD *)(a1 + 208)) )
                      {
                        v98 = v132;
                        if ( v132 == v133 )
                        {
                          sub_14147F0((__int64)&v131, v132, &v140);
                        }
                        else
                        {
                          if ( v132 )
                          {
                            *v132 = v140;
                            v98 = v132;
                          }
                          v132 = v98 + 1;
                        }
                      }
                      else
                      {
                        v99 = v138;
                        if ( v138 == v139[0] )
                        {
                          sub_14147F0((__int64)&v137, (_BYTE *)v138, &v140);
                        }
                        else
                        {
                          if ( v138 )
                          {
                            *(_QWORD *)v138 = v140;
                            v99 = v138;
                          }
                          v138 = v99 + 8;
                        }
                      }
                    }
                  }
                }
                else if ( v97 == 15 )
                {
                  goto LABEL_182;
                }
              }
            }
          }
          break;
      }
LABEL_58:
      v41 = *(_QWORD *)(v41 + 8);
      if ( v42 == v41 )
        goto LABEL_69;
    }
  }
LABEL_70:
  if ( v129 == v128 )
  {
    v100 = v132;
    LODWORD(v141) = 0;
    s = 0;
    *(_QWORD *)&v143 = &v141;
    *((_QWORD *)&v143 + 1) = &v141;
    v144[0] = 0;
    if ( v131 == v132 )
    {
      v104 = (unsigned __int64)v137;
      v105 = v138;
      v110 = 0;
      if ( (_QWORD *)v138 == v137 )
      {
LABEL_218:
        sub_21F25B0((__int64)v110);
        s = 0;
        *(_QWORD *)&v143 = &v141;
        *((_QWORD *)&v143 + 1) = &v141;
        v144[0] = 0;
        sub_21F25B0(0);
        goto LABEL_71;
      }
    }
    else
    {
      v101 = v131;
      do
      {
        if ( !(unsigned __int8)sub_21F6DD0(a1, &v125, *v101) )
          sub_21F3A20(*(unsigned __int8 **)(a1 + 216), a2, *v101, &v140, a3, a4, a5, a6, v102, v103, a9, a10);
        ++v101;
      }
      while ( v100 != v101 );
      v104 = (unsigned __int64)v137;
      v105 = v138;
      if ( v137 == (_QWORD *)v138 )
        goto LABEL_215;
    }
    v106 = (unsigned __int64 *)v104;
    do
    {
      if ( !(unsigned __int8)sub_21F6DD0(a1, &v125, *v106) )
        sub_21F3A20(*(unsigned __int8 **)(a1 + 216), a2, *v106, &v140, a3, a4, a5, a6, v107, v108, a9, a10);
      ++v106;
    }
    while ( (unsigned __int64 *)v105 != v106 );
LABEL_215:
    for ( j = v143; (__int64 **)j != &v141; j = sub_220EF30(j) )
      sub_15F20C0(*(_QWORD **)(j + 32));
    v110 = s;
    goto LABEL_218;
  }
LABEL_71:
  sub_21F2490(*(_QWORD **)(a1 + 176));
  v46 = (unsigned __int64)v137;
  result = a1 + 168;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = a1 + 168;
  *(_QWORD *)(a1 + 192) = a1 + 168;
  *(_QWORD *)(a1 + 200) = 0;
  if ( v46 )
    result = j_j___libc_free_0(v46, v139[0] - v46);
  if ( v131 )
    result = j_j___libc_free_0(v131, (char *)v133 - (char *)v131);
  if ( v128 )
    result = j_j___libc_free_0(v128, (char *)v130 - (char *)v128);
  if ( v125 )
    return j_j___libc_free_0(v125, &v127[-v125]);
  return result;
}
