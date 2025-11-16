// Function: sub_11D0CC0
// Address: 0x11d0cc0
//
__int64 __fastcall sub_11D0CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v9; // dl
  __int64 v10; // rcx
  int v11; // esi
  unsigned int v12; // eax
  _QWORD *v13; // rbx
  __int64 v14; // rdi
  unsigned int v15; // r12d
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned int v20; // edi
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  const void *v24; // r13
  unsigned __int64 v25; // rbx
  __int64 v26; // rsi
  _BYTE *v27; // rdi
  unsigned int v28; // edx
  __int64 v29; // r12
  __int64 v30; // r15
  _QWORD *v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  unsigned int v35; // eax
  __int64 *v36; // rax
  __int64 v37; // rbx
  _QWORD *v38; // rax
  _QWORD *v39; // rcx
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdi
  __int64 v43; // rdx
  _QWORD *v44; // rdx
  _QWORD *v45; // r12
  __int64 v46; // r8
  __int64 v47; // r15
  _BYTE *v48; // r14
  int v49; // ecx
  __int64 v50; // r13
  __int64 v51; // rsi
  int v52; // ecx
  unsigned int v53; // edx
  __int64 *v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rbx
  __int64 v57; // rax
  _QWORD *v58; // r15
  __int64 v59; // r12
  __int64 v60; // r13
  __int64 v61; // r9
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  _QWORD *v66; // r12
  __int64 v67; // rsi
  int v68; // r11d
  unsigned int v69; // edx
  __int64 *v70; // rcx
  __int64 v71; // rdi
  int v72; // eax
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  int v78; // eax
  int v79; // r9d
  _BYTE *v80; // rdi
  __int64 *v81; // r13
  __int64 *v82; // rbx
  unsigned int v83; // eax
  __int64 *v84; // rdi
  __int64 v85; // rcx
  unsigned int v86; // edx
  __int64 *v87; // r10
  __int64 v88; // rdi
  int v89; // eax
  __int64 *v90; // rcx
  int v91; // r12d
  int v92; // r11d
  __int64 v93; // r8
  int v94; // edi
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdi
  int v98; // edx
  __int64 v99; // rax
  __int64 v100; // rsi
  int v101; // ecx
  _QWORD *v102; // r8
  unsigned int v103; // edx
  __int64 v104; // rdi
  int v105; // edi
  int v106; // edx
  unsigned int v107; // edx
  __int64 v108; // rdi
  __int64 v109; // rcx
  int v110; // edx
  unsigned int v111; // r13d
  __int64 v112; // rcx
  unsigned int v113; // r12d
  int v114; // ecx
  _QWORD *v115; // rdx
  unsigned int v116; // r11d
  __int64 v117; // [rsp+10h] [rbp-130h]
  __int64 v118; // [rsp+18h] [rbp-128h]
  __int64 v119; // [rsp+18h] [rbp-128h]
  __int64 v120; // [rsp+18h] [rbp-128h]
  __int64 v121; // [rsp+18h] [rbp-128h]
  __int64 v122; // [rsp+18h] [rbp-128h]
  __int64 v123; // [rsp+20h] [rbp-120h]
  __int64 v124; // [rsp+20h] [rbp-120h]
  __int64 v125; // [rsp+20h] [rbp-120h]
  __int64 v126; // [rsp+20h] [rbp-120h]
  __int64 v128; // [rsp+38h] [rbp-108h]
  __int64 v129; // [rsp+38h] [rbp-108h]
  __int64 v130; // [rsp+38h] [rbp-108h]
  __int64 v131; // [rsp+40h] [rbp-100h]
  __int64 v132; // [rsp+40h] [rbp-100h]
  __int64 v133; // [rsp+40h] [rbp-100h]
  __int64 v134; // [rsp+48h] [rbp-F8h]
  _BYTE *v135; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v136; // [rsp+58h] [rbp-E8h]
  _BYTE v137[64]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v138; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v139; // [rsp+A8h] [rbp-98h]
  __int64 v140; // [rsp+B0h] [rbp-90h]
  __int64 v141; // [rsp+B8h] [rbp-88h]
  _BYTE *v142; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v143; // [rsp+C8h] [rbp-78h]
  _BYTE v144[112]; // [rsp+D0h] [rbp-70h] BYREF

  v134 = a5;
  v9 = *(_BYTE *)(a5 + 8) & 1;
  if ( v9 )
  {
    v10 = a5 + 16;
    v11 = 3;
  }
  else
  {
    v17 = *(unsigned int *)(a5 + 24);
    v10 = *(_QWORD *)(a5 + 16);
    if ( !(_DWORD)v17 )
    {
      v13 = 0;
      ++*(_QWORD *)a5;
      v18 = *(_DWORD *)(a5 + 8);
      v19 = (v18 >> 1) + 1;
      goto LABEL_9;
    }
    v11 = v17 - 1;
  }
  v12 = v11 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v13 = (_QWORD *)(v10 + 32LL * v12);
  v14 = *v13;
  if ( a1 == *v13 )
    goto LABEL_4;
  v91 = 1;
  a5 = 0;
  while ( 1 )
  {
    if ( v14 == -4096 )
    {
      if ( a5 )
        v13 = (_QWORD *)a5;
      ++*(_QWORD *)v134;
      v18 = *(_DWORD *)(v134 + 8);
      v19 = (v18 >> 1) + 1;
      if ( v9 )
      {
        v20 = 12;
        v17 = 4;
LABEL_10:
        v21 = (unsigned int)(4 * v19);
        if ( (unsigned int)v21 < v20 )
        {
          v22 = (unsigned int)(v17 - *(_DWORD *)(v134 + 12) - v19);
          v23 = (unsigned int)v17 >> 3;
          if ( (unsigned int)v22 > (unsigned int)v23 )
            goto LABEL_12;
          v130 = a3;
          v133 = a2;
          sub_11CE470(v134, (_QWORD *)v17, v22, v23, a5, a6);
          a2 = v133;
          a3 = v130;
          if ( (*(_BYTE *)(v134 + 8) & 1) != 0 )
          {
            v97 = v134 + 16;
            v98 = 3;
            goto LABEL_145;
          }
          v106 = *(_DWORD *)(v134 + 24);
          v97 = *(_QWORD *)(v134 + 16);
          if ( v106 )
          {
            v98 = v106 - 1;
LABEL_145:
            v99 = v98 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v13 = (_QWORD *)(v97 + 32 * v99);
            v100 = *v13;
            if ( a1 != *v13 )
            {
              v101 = 1;
              v102 = 0;
              while ( v100 != -4096 )
              {
                if ( v100 == -8192 && !v102 )
                  v102 = v13;
                LODWORD(v99) = v98 & (v101 + v99);
                v13 = (_QWORD *)(v97 + 32LL * (unsigned int)v99);
                v100 = *v13;
                if ( a1 == *v13 )
                  goto LABEL_142;
                ++v101;
              }
              if ( v102 )
                v13 = v102;
            }
            goto LABEL_142;
          }
LABEL_223:
          *(_DWORD *)(v134 + 8) = (2 * (*(_DWORD *)(v134 + 8) >> 1) + 2) | *(_DWORD *)(v134 + 8) & 1;
          BUG();
        }
        v129 = a3;
        v132 = a2;
        sub_11CE470(v134, (_QWORD *)(unsigned int)(2 * v17), v21, v19, a5, a6);
        a2 = v132;
        a3 = v129;
        if ( (*(_BYTE *)(v134 + 8) & 1) != 0 )
        {
          v93 = v134 + 16;
          v94 = 3;
        }
        else
        {
          v105 = *(_DWORD *)(v134 + 24);
          v93 = *(_QWORD *)(v134 + 16);
          if ( !v105 )
            goto LABEL_223;
          v94 = v105 - 1;
        }
        v95 = v94 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v13 = (_QWORD *)(v93 + 32 * v95);
        v96 = *v13;
        if ( a1 != *v13 )
        {
          v114 = 1;
          v115 = 0;
          while ( v96 != -4096 )
          {
            if ( !v115 && v96 == -8192 )
              v115 = v13;
            LODWORD(v95) = v94 & (v114 + v95);
            v13 = (_QWORD *)(v93 + 32LL * (unsigned int)v95);
            v96 = *v13;
            if ( a1 == *v13 )
              goto LABEL_142;
            ++v114;
          }
          if ( v115 )
          {
            v13 = v115;
            v18 = *(_DWORD *)(v134 + 8);
LABEL_12:
            *(_DWORD *)(v134 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v13 != -4096 )
              --*(_DWORD *)(v134 + 12);
            *v13 = a1;
            v13[1] = v13 + 3;
            v15 = 0;
            v13[2] = 0x100000000LL;
            v128 = a3;
            v131 = a2;
            sub_D472F0(a1, (__int64)(v13 + 1));
            a3 = v128;
            a2 = v131;
            if ( !*((_DWORD *)v13 + 4) )
              return v15;
LABEL_15:
            v140 = 0;
            v142 = v144;
            v141 = 0;
            v143 = 0x800000000LL;
            v139 = 0;
            v24 = (const void *)v13[1];
            v25 = *((unsigned int *)v13 + 4);
            v26 = (__int64)v137;
            v138 = 0;
            v135 = v137;
            v136 = 0x800000000LL;
            if ( v25 > 8 )
            {
              v120 = a3;
              v125 = a2;
              sub_C8D5F0((__int64)&v135, v137, v25, 8u, a5, a6);
              a2 = v125;
              a3 = v120;
              v80 = &v135[8 * (unsigned int)v136];
            }
            else
            {
              if ( !(8 * v25) )
              {
                LODWORD(v136) = v25;
                v27 = v137;
                v28 = v25;
                if ( !(_DWORD)v25 )
                  goto LABEL_43;
LABEL_18:
                v29 = a2;
                v30 = a3;
                while ( 1 )
                {
                  v31 = *(_QWORD **)(a1 + 32);
                  v32 = v28--;
                  v33 = *(_QWORD *)&v27[8 * v32 - 8];
                  LODWORD(v136) = v28;
                  if ( v33 != *v31 )
                    break;
LABEL_39:
                  if ( !v28 )
                  {
                    a2 = v29;
                    a3 = v30;
                    goto LABEL_41;
                  }
                }
                if ( v33 )
                {
                  v34 = (unsigned int)(*(_DWORD *)(v33 + 44) + 1);
                  v35 = *(_DWORD *)(v33 + 44) + 1;
                }
                else
                {
                  v34 = 0;
                  v35 = 0;
                }
                if ( v35 >= *(_DWORD *)(v29 + 32) )
                  BUG();
                v36 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v29 + 24) + 8 * v34) + 8LL);
                v37 = *v36;
                if ( *(_BYTE *)(a1 + 84) )
                {
                  v38 = *(_QWORD **)(a1 + 64);
                  v39 = &v38[*(unsigned int *)(a1 + 76)];
                  if ( v38 == v39 )
                    goto LABEL_39;
                  while ( v37 != *v38 )
                  {
                    if ( v39 == ++v38 )
                      goto LABEL_39;
                  }
                }
                else
                {
                  v26 = *v36;
                  if ( !sub_C8CA60(a1 + 56, *v36) )
                  {
                    v28 = v136;
                    v27 = v135;
                    goto LABEL_39;
                  }
                }
                if ( !(_DWORD)v140 )
                {
                  v40 = v142;
                  v41 = 8LL * (unsigned int)v143;
                  v26 = (__int64)&v142[v41];
                  v42 = v41 >> 3;
                  v43 = v41 >> 5;
                  if ( !v43 )
                    goto LABEL_85;
                  v44 = &v142[32 * v43];
                  do
                  {
                    if ( v37 == *v40 )
                      goto LABEL_36;
                    if ( v37 == v40[1] )
                    {
                      ++v40;
                      goto LABEL_36;
                    }
                    if ( v37 == v40[2] )
                    {
                      v40 += 2;
                      goto LABEL_36;
                    }
                    if ( v37 == v40[3] )
                    {
                      v40 += 3;
                      goto LABEL_36;
                    }
                    v40 += 4;
                  }
                  while ( v44 != v40 );
                  v42 = (v26 - (__int64)v40) >> 3;
LABEL_85:
                  if ( v42 != 2 )
                  {
                    if ( v42 != 3 )
                    {
                      if ( v42 != 1 )
                        goto LABEL_88;
LABEL_105:
                      if ( v37 == *v40 )
                      {
LABEL_36:
                        if ( (_QWORD *)v26 == v40 )
                          goto LABEL_88;
                        goto LABEL_37;
                      }
LABEL_88:
                      if ( (unsigned __int64)(unsigned int)v143 + 1 > HIDWORD(v143) )
                      {
                        sub_C8D5F0((__int64)&v142, v144, (unsigned int)v143 + 1LL, 8u, a5, a6);
                        v26 = (__int64)&v142[8 * (unsigned int)v143];
                      }
                      *(_QWORD *)v26 = v37;
                      v75 = (unsigned int)(v143 + 1);
                      LODWORD(v143) = v75;
                      if ( (unsigned int)v75 <= 8 )
                      {
LABEL_91:
                        v76 = (unsigned int)v136;
                        v77 = (unsigned int)v136 + 1LL;
                        if ( v77 > HIDWORD(v136) )
                        {
                          v26 = (__int64)v137;
                          sub_C8D5F0((__int64)&v135, v137, v77, 8u, a5, a6);
                          v76 = (unsigned int)v136;
                        }
                        *(_QWORD *)&v135[8 * v76] = v37;
                        v28 = v136 + 1;
                        LODWORD(v136) = v136 + 1;
                        goto LABEL_38;
                      }
                      v81 = (__int64 *)v142;
                      v122 = v37;
                      v82 = (__int64 *)&v142[8 * v75];
                      while ( 1 )
                      {
                        v26 = (unsigned int)v141;
                        if ( !(_DWORD)v141 )
                          break;
                        a6 = (unsigned int)(v141 - 1);
                        a5 = v139;
                        v83 = a6 & (((unsigned int)*v81 >> 9) ^ ((unsigned int)*v81 >> 4));
                        v84 = (__int64 *)(v139 + 8LL * v83);
                        v85 = *v84;
                        if ( *v84 != *v81 )
                        {
                          v92 = 1;
                          v87 = 0;
                          while ( v85 != -4096 )
                          {
                            if ( v85 != -8192 || v87 )
                              v84 = v87;
                            v83 = a6 & (v92 + v83);
                            v85 = *(_QWORD *)(v139 + 8LL * v83);
                            if ( *v81 == v85 )
                              goto LABEL_112;
                            ++v92;
                            v87 = v84;
                            v84 = (__int64 *)(v139 + 8LL * v83);
                          }
                          if ( !v87 )
                            v87 = v84;
                          ++v138;
                          v89 = v140 + 1;
                          if ( 4 * ((int)v140 + 1) < (unsigned int)(3 * v141) )
                          {
                            if ( (int)v141 - HIDWORD(v140) - v89 <= (unsigned int)v141 >> 3 )
                            {
                              sub_CF28B0((__int64)&v138, v141);
                              if ( !(_DWORD)v141 )
                              {
LABEL_222:
                                LODWORD(v140) = v140 + 1;
                                BUG();
                              }
                              a5 = *v81;
                              v26 = 1;
                              v90 = 0;
                              v103 = (v141 - 1) & (((unsigned int)*v81 >> 9) ^ ((unsigned int)*v81 >> 4));
                              v87 = (__int64 *)(v139 + 8LL * v103);
                              v104 = *v87;
                              v89 = v140 + 1;
                              if ( *v87 != *v81 )
                              {
                                while ( v104 != -4096 )
                                {
                                  if ( !v90 && v104 == -8192 )
                                    v90 = v87;
                                  a6 = (unsigned int)(v26 + 1);
                                  v26 = ((_DWORD)v141 - 1) & (v103 + (unsigned int)v26);
                                  v87 = (__int64 *)(v139 + 8 * v26);
                                  v103 = v26;
                                  v104 = *v87;
                                  if ( a5 == *v87 )
                                    goto LABEL_136;
                                  v26 = (unsigned int)a6;
                                }
LABEL_119:
                                if ( v90 )
                                  v87 = v90;
                              }
                            }
LABEL_136:
                            LODWORD(v140) = v89;
                            if ( *v87 != -4096 )
                              --HIDWORD(v140);
                            *v87 = *v81;
                            goto LABEL_112;
                          }
LABEL_115:
                          v26 = (unsigned int)(2 * v141);
                          sub_CF28B0((__int64)&v138, v26);
                          if ( !(_DWORD)v141 )
                            goto LABEL_222;
                          a5 = *v81;
                          v86 = (v141 - 1) & (((unsigned int)*v81 >> 9) ^ ((unsigned int)*v81 >> 4));
                          v87 = (__int64 *)(v139 + 8LL * v86);
                          v88 = *v87;
                          v89 = v140 + 1;
                          if ( *v87 != *v81 )
                          {
                            v26 = 1;
                            v90 = 0;
                            while ( v88 != -4096 )
                            {
                              if ( v88 == -8192 && !v90 )
                                v90 = v87;
                              a6 = (unsigned int)(v26 + 1);
                              v26 = ((_DWORD)v141 - 1) & (v86 + (unsigned int)v26);
                              v87 = (__int64 *)(v139 + 8 * v26);
                              v86 = v26;
                              v88 = *v87;
                              if ( a5 == *v87 )
                                goto LABEL_136;
                              v26 = (unsigned int)a6;
                            }
                            goto LABEL_119;
                          }
                          goto LABEL_136;
                        }
LABEL_112:
                        if ( v82 == ++v81 )
                        {
                          v37 = v122;
                          goto LABEL_91;
                        }
                      }
                      ++v138;
                      goto LABEL_115;
                    }
                    if ( v37 == *v40 )
                      goto LABEL_36;
                    ++v40;
                  }
                  if ( v37 != *v40 )
                  {
                    ++v40;
                    goto LABEL_105;
                  }
                  goto LABEL_36;
                }
                v26 = (unsigned int)v141;
                if ( (_DWORD)v141 )
                {
                  a6 = v139;
                  v68 = 1;
                  a5 = 0;
                  v69 = (v141 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
                  v70 = (__int64 *)(v139 + 8LL * v69);
                  v71 = *v70;
                  if ( v37 == *v70 )
                  {
LABEL_37:
                    v28 = v136;
LABEL_38:
                    v27 = v135;
                    goto LABEL_39;
                  }
                  while ( v71 != -4096 )
                  {
                    if ( v71 == -8192 && !a5 )
                      a5 = (__int64)v70;
                    v69 = (v141 - 1) & (v68 + v69);
                    v70 = (__int64 *)(v139 + 8LL * v69);
                    v71 = *v70;
                    if ( v37 == *v70 )
                      goto LABEL_37;
                    ++v68;
                  }
                  if ( !a5 )
                    a5 = (__int64)v70;
                  v72 = v140 + 1;
                  ++v138;
                  if ( 4 * ((int)v140 + 1) < (unsigned int)(3 * v141) )
                  {
                    if ( (int)v141 - HIDWORD(v140) - v72 <= (unsigned int)v141 >> 3 )
                    {
                      sub_CF28B0((__int64)&v138, v141);
                      if ( !(_DWORD)v141 )
                      {
LABEL_224:
                        LODWORD(v140) = v140 + 1;
                        BUG();
                      }
                      a6 = v139;
                      v110 = 1;
                      v26 = 0;
                      v111 = (v141 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
                      a5 = v139 + 8LL * v111;
                      v112 = *(_QWORD *)a5;
                      v72 = v140 + 1;
                      if ( v37 != *(_QWORD *)a5 )
                      {
                        while ( v112 != -4096 )
                        {
                          if ( !v26 && v112 == -8192 )
                            v26 = a5;
                          v111 = (v141 - 1) & (v110 + v111);
                          a5 = v139 + 8LL * v111;
                          v112 = *(_QWORD *)a5;
                          if ( v37 == *(_QWORD *)a5 )
                            goto LABEL_79;
                          ++v110;
                        }
                        if ( v26 )
                          a5 = v26;
                      }
                    }
                    goto LABEL_79;
                  }
                }
                else
                {
                  ++v138;
                }
                v26 = (unsigned int)(2 * v141);
                sub_CF28B0((__int64)&v138, v26);
                if ( !(_DWORD)v141 )
                  goto LABEL_224;
                a6 = (unsigned int)(v141 - 1);
                v107 = a6 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
                a5 = v139 + 8LL * v107;
                v108 = *(_QWORD *)a5;
                v72 = v140 + 1;
                if ( *(_QWORD *)a5 != v37 )
                {
                  v26 = 1;
                  v109 = 0;
                  while ( v108 != -4096 )
                  {
                    if ( v108 == -8192 && !v109 )
                      v109 = a5;
                    v116 = v26 + 1;
                    v107 = a6 & (v26 + v107);
                    v26 = v107;
                    a5 = v139 + 8LL * v107;
                    v108 = *(_QWORD *)a5;
                    if ( v37 == *(_QWORD *)a5 )
                      goto LABEL_79;
                    v26 = v116;
                  }
                  if ( v109 )
                    a5 = v109;
                }
LABEL_79:
                LODWORD(v140) = v72;
                if ( *(_QWORD *)a5 != -4096 )
                  --HIDWORD(v140);
                *(_QWORD *)a5 = v37;
                v73 = (unsigned int)v143;
                v74 = (unsigned int)v143 + 1LL;
                if ( v74 > HIDWORD(v143) )
                {
                  v26 = (__int64)v144;
                  sub_C8D5F0((__int64)&v142, v144, v74, 8u, a5, a6);
                  v73 = (unsigned int)v143;
                }
                *(_QWORD *)&v142[8 * v73] = v37;
                LODWORD(v143) = v143 + 1;
                goto LABEL_91;
              }
              v80 = v137;
            }
            v26 = (__int64)v24;
            v121 = a3;
            v126 = a2;
            memcpy(v80, v24, 8 * v25);
            v27 = v135;
            a2 = v126;
            a3 = v121;
            LODWORD(v136) = v136 + v25;
            v28 = v136;
            if ( !(_DWORD)v136 )
            {
LABEL_41:
              if ( v27 != v137 )
              {
                v118 = a3;
                v123 = a2;
                _libc_free(v27, v26);
                a3 = v118;
                a2 = v123;
              }
LABEL_43:
              v45 = v142;
              v46 = a1;
              v47 = a3;
              v135 = v137;
              v136 = 0x800000000LL;
              v48 = &v142[8 * (unsigned int)v143];
              if ( v142 == v48 )
              {
LABEL_64:
                v67 = a2;
                v15 = sub_11CE990((__int64)&v135, a2, a3, a4, 0, 0, v134);
                if ( v135 != v137 )
                  _libc_free(v135, v67);
                if ( v142 != v144 )
                  _libc_free(v142, v67);
                sub_C7D6A0(v139, 8LL * (unsigned int)v141, 8);
                return v15;
              }
LABEL_46:
              while ( 2 )
              {
                v49 = *(_DWORD *)(v47 + 24);
                v50 = *v45;
                v51 = *(_QWORD *)(v47 + 8);
                if ( v49 )
                {
                  v52 = v49 - 1;
                  v53 = v52 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
                  v54 = (__int64 *)(v51 + 16LL * v53);
                  v55 = *v54;
                  if ( v50 == *v54 )
                  {
LABEL_48:
                    if ( v46 == v54[1] )
                    {
                      v56 = *(_QWORD *)(v50 + 56);
                      if ( v50 + 48 != v56 )
                      {
                        v57 = v47;
                        v58 = v45;
                        v59 = *v45;
                        v60 = v50 + 48;
                        v61 = v57;
                        while ( 1 )
                        {
                          if ( !v56 )
                            BUG();
                          v64 = *(_QWORD *)(v56 - 8);
                          if ( !v64 )
                            goto LABEL_55;
                          if ( *(_QWORD *)(v64 + 8)
                            || (v65 = *(_QWORD *)(v64 + 24), v59 != *(_QWORD *)(v65 + 40))
                            || *(_BYTE *)v65 == 84 )
                          {
                            if ( *(_BYTE *)(*(_QWORD *)(v56 - 16) + 8LL) != 11 )
                            {
                              v62 = (unsigned int)v136;
                              v63 = (unsigned int)v136 + 1LL;
                              if ( v63 > HIDWORD(v136) )
                              {
                                v117 = v61;
                                v119 = a2;
                                v124 = v46;
                                sub_C8D5F0((__int64)&v135, v137, v63, 8u, v46, v61);
                                v62 = (unsigned int)v136;
                                v61 = v117;
                                a2 = v119;
                                v46 = v124;
                              }
                              *(_QWORD *)&v135[8 * v62] = v56 - 24;
                              LODWORD(v136) = v136 + 1;
                            }
LABEL_55:
                            v56 = *(_QWORD *)(v56 + 8);
                            if ( v60 == v56 )
                              goto LABEL_62;
                          }
                          else
                          {
                            v56 = *(_QWORD *)(v56 + 8);
                            if ( v60 == v56 )
                            {
LABEL_62:
                              v66 = v58;
                              v47 = v61;
                              v45 = v66 + 1;
                              if ( v48 == (_BYTE *)v45 )
                              {
LABEL_63:
                                a3 = v47;
                                goto LABEL_64;
                              }
                              goto LABEL_46;
                            }
                          }
                        }
                      }
                    }
                  }
                  else
                  {
                    v78 = 1;
                    while ( v55 != -4096 )
                    {
                      v79 = v78 + 1;
                      v53 = v52 & (v78 + v53);
                      v54 = (__int64 *)(v51 + 16LL * v53);
                      v55 = *v54;
                      if ( v50 == *v54 )
                        goto LABEL_48;
                      v78 = v79;
                    }
                  }
                }
                if ( v48 == (_BYTE *)++v45 )
                  goto LABEL_63;
                continue;
              }
            }
            goto LABEL_18;
          }
        }
LABEL_142:
        v18 = *(_DWORD *)(v134 + 8);
        goto LABEL_12;
      }
      v17 = *(unsigned int *)(v134 + 24);
LABEL_9:
      v20 = 3 * v17;
      goto LABEL_10;
    }
    if ( a5 || v14 != -8192 )
      v13 = (_QWORD *)a5;
    a5 = (unsigned int)(v91 + 1);
    v113 = v12 + v91;
    v12 = v11 & v113;
    a6 = v10 + 32LL * (v11 & v113);
    v14 = *(_QWORD *)a6;
    if ( a1 == *(_QWORD *)a6 )
      break;
    v91 = a5;
    a5 = (__int64)v13;
    v13 = (_QWORD *)a6;
  }
  v13 = (_QWORD *)(v10 + 32LL * (v11 & v113));
LABEL_4:
  v15 = 0;
  if ( *((_DWORD *)v13 + 4) )
    goto LABEL_15;
  return v15;
}
