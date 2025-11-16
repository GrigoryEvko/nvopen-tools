// Function: sub_28739E0
// Address: 0x28739e0
//
__int64 __fastcall sub_28739E0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rcx
  _QWORD *v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // rdi
  _QWORD *v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r12
  _QWORD *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // esi
  unsigned int v31; // ecx
  __int64 *v32; // rdi
  __int64 *v33; // rdx
  int v34; // eax
  __int64 v35; // rax
  unsigned int v36; // edi
  __int64 *v37; // rdx
  __int64 *v38; // r11
  int v39; // eax
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  _QWORD **v42; // rbx
  unsigned __int64 v43; // rax
  unsigned int v44; // r12d
  __int64 v45; // r13
  _QWORD *v46; // r12
  _QWORD *v47; // r14
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // rax
  unsigned __int64 v57; // rax
  __int64 v58; // r13
  __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // r13
  __int64 v62; // rdi
  unsigned int v63; // r12d
  unsigned __int64 v64; // rax
  unsigned int v65; // ecx
  __int64 v66; // rdx
  __int64 v67; // rdi
  unsigned int v68; // r12d
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // rax
  __int64 v72; // rdx
  unsigned int v73; // eax
  __int64 v74; // rcx
  _QWORD *v75; // rax
  __int64 v76; // rax
  __int64 v78; // rax
  __int64 *v79; // r12
  __int64 *v80; // rbx
  unsigned int v81; // eax
  __int64 *v82; // rdi
  __int64 v83; // rcx
  unsigned int v84; // esi
  int v85; // eax
  int v86; // r10d
  __int64 v87; // r11
  unsigned int v88; // edx
  __int64 v89; // rdi
  int v90; // eax
  __int64 v91; // rax
  int v92; // r11d
  int v93; // eax
  int v94; // eax
  int v95; // r10d
  int v96; // esi
  __int64 v97; // rcx
  __int64 v98; // r11
  unsigned int v99; // edx
  __int64 v100; // rdi
  int v101; // esi
  int v102; // r11d
  int v103; // r11d
  __int64 v104; // rcx
  int v105; // edi
  __int64 *v106; // rsi
  int v107; // r10d
  int v108; // r10d
  int v109; // esi
  __int64 *v110; // rcx
  __int64 v111; // r11
  __int64 v112; // rdi
  __int64 v113; // rcx
  __int64 v114; // r11
  int v115; // edi
  __int64 *v116; // rsi
  int v117; // esi
  __int64 v118; // rcx
  __int64 v119; // rdi
  int v120; // [rsp+8h] [rbp-F8h]
  int v121; // [rsp+10h] [rbp-F0h]
  __int64 v122; // [rsp+10h] [rbp-F0h]
  unsigned int v123; // [rsp+28h] [rbp-D8h]
  int v124; // [rsp+28h] [rbp-D8h]
  _QWORD *v125; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v126; // [rsp+28h] [rbp-D8h]
  unsigned int v127; // [rsp+28h] [rbp-D8h]
  _QWORD *v128; // [rsp+30h] [rbp-D0h]
  __int64 v129; // [rsp+38h] [rbp-C8h]
  __int64 v130; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v131; // [rsp+38h] [rbp-C8h]
  __int64 v132; // [rsp+48h] [rbp-B8h] BYREF
  _BYTE *v133; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v134; // [rsp+58h] [rbp-A8h]
  _BYTE v135[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v136; // [rsp+80h] [rbp-80h] BYREF
  __int64 v137; // [rsp+88h] [rbp-78h]
  __int64 v138; // [rsp+90h] [rbp-70h]
  __int64 v139; // [rsp+98h] [rbp-68h]
  _QWORD *v140; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v141; // [rsp+A8h] [rbp-58h]
  _BYTE v142[80]; // [rsp+B0h] [rbp-50h] BYREF

  v2 = *(_QWORD *)a1;
  v140 = v142;
  v3 = *(_QWORD *)(v2 + 208);
  v141 = 0x400000000LL;
  v134 = 0x400000000LL;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v133 = v135;
  v129 = v2 + 200;
  if ( v3 == v2 + 200 )
  {
    if ( *(_DWORD *)(a1 + 1280) != 1 )
      goto LABEL_92;
LABEL_123:
    sub_28662B0(a1 + 1240);
    *(_DWORD *)(a1 + 1280) = 0;
    goto LABEL_90;
  }
  v4 = v3;
  while ( 1 )
  {
    v5 = v4 - 32;
    if ( !v4 )
      v5 = 0;
    v6 = sub_22AD250(v2, v5);
    v7 = v6;
    if ( v6 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = sub_D95540(v6);
      v12 = sub_D97090(v8, v9);
      v13 = *(_DWORD *)(a1 + 1256);
      if ( !v13 )
      {
        v14 = *(unsigned int *)(a1 + 1280);
        v15 = *(_QWORD **)(a1 + 1272);
        v16 = &v15[v14];
        v17 = (8 * v14) >> 3;
        if ( !((8 * v14) >> 5) )
          goto LABEL_96;
        v18 = &v15[4 * ((8 * v14) >> 5)];
        do
        {
          if ( v12 == *v15 )
            goto LABEL_16;
          if ( v12 == v15[1] )
          {
            ++v15;
            goto LABEL_16;
          }
          if ( v12 == v15[2] )
          {
            v15 += 2;
            goto LABEL_16;
          }
          if ( v12 == v15[3] )
          {
            v15 += 3;
            goto LABEL_16;
          }
          v15 += 4;
        }
        while ( v18 != v15 );
        v17 = v16 - v15;
LABEL_96:
        if ( v17 != 2 )
        {
          if ( v17 != 3 )
          {
            if ( v17 != 1 )
              goto LABEL_99;
LABEL_118:
            if ( v12 != *v15 )
            {
LABEL_99:
              if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1284) )
              {
                sub_C8D5F0(a1 + 1272, (const void *)(a1 + 1288), v14 + 1, 8u, v10, v11);
                v16 = (__int64 *)(*(_QWORD *)(a1 + 1272) + 8LL * *(unsigned int *)(a1 + 1280));
              }
              *v16 = v12;
              v78 = (unsigned int)(*(_DWORD *)(a1 + 1280) + 1);
              *(_DWORD *)(a1 + 1280) = v78;
              if ( (unsigned int)v78 > 4 )
              {
                v126 = v7;
                v79 = *(__int64 **)(a1 + 1272);
                v122 = a1 + 1240;
                v80 = &v79[v78];
                while ( 1 )
                {
                  v84 = *(_DWORD *)(a1 + 1264);
                  if ( !v84 )
                    break;
                  v11 = *(_QWORD *)(a1 + 1248);
                  v81 = (v84 - 1) & (((unsigned int)*v79 >> 9) ^ ((unsigned int)*v79 >> 4));
                  v82 = (__int64 *)(v11 + 8LL * v81);
                  v83 = *v82;
                  if ( *v79 == *v82 )
                  {
LABEL_104:
                    if ( v80 == ++v79 )
                      goto LABEL_112;
                  }
                  else
                  {
                    v92 = 1;
                    v10 = 0;
                    while ( v83 != -4096 )
                    {
                      if ( v10 || v83 != -8192 )
                        v82 = (__int64 *)v10;
                      v10 = (unsigned int)(v92 + 1);
                      v81 = (v84 - 1) & (v92 + v81);
                      v83 = *(_QWORD *)(v11 + 8LL * v81);
                      if ( *v79 == v83 )
                        goto LABEL_104;
                      ++v92;
                      v10 = (__int64)v82;
                      v82 = (__int64 *)(v11 + 8LL * v81);
                    }
                    v93 = *(_DWORD *)(a1 + 1256);
                    if ( !v10 )
                      v10 = (__int64)v82;
                    ++*(_QWORD *)(a1 + 1240);
                    v90 = v93 + 1;
                    if ( 4 * v90 < 3 * v84 )
                    {
                      if ( v84 - *(_DWORD *)(a1 + 1260) - v90 > v84 >> 3 )
                        goto LABEL_109;
                      sub_BCFDB0(v122, v84);
                      v94 = *(_DWORD *)(a1 + 1264);
                      if ( !v94 )
                      {
LABEL_213:
                        ++*(_DWORD *)(a1 + 1256);
                        BUG();
                      }
                      v11 = *v79;
                      v95 = v94 - 1;
                      v96 = 1;
                      v97 = 0;
                      v98 = *(_QWORD *)(a1 + 1248);
                      v99 = (v94 - 1) & (((unsigned int)*v79 >> 9) ^ ((unsigned int)*v79 >> 4));
                      v10 = v98 + 8LL * v99;
                      v100 = *(_QWORD *)v10;
                      v90 = *(_DWORD *)(a1 + 1256) + 1;
                      if ( *v79 == *(_QWORD *)v10 )
                        goto LABEL_109;
                      while ( v100 != -4096 )
                      {
                        if ( v100 == -8192 && !v97 )
                          v97 = v10;
                        v99 = v95 & (v96 + v99);
                        v10 = v98 + 8LL * v99;
                        v100 = *(_QWORD *)v10;
                        if ( v11 == *(_QWORD *)v10 )
                          goto LABEL_109;
                        ++v96;
                      }
                      goto LABEL_133;
                    }
LABEL_107:
                    sub_BCFDB0(v122, 2 * v84);
                    v85 = *(_DWORD *)(a1 + 1264);
                    if ( !v85 )
                      goto LABEL_213;
                    v11 = *v79;
                    v86 = v85 - 1;
                    v87 = *(_QWORD *)(a1 + 1248);
                    v88 = (v85 - 1) & (((unsigned int)*v79 >> 9) ^ ((unsigned int)*v79 >> 4));
                    v10 = v87 + 8LL * v88;
                    v89 = *(_QWORD *)v10;
                    v90 = *(_DWORD *)(a1 + 1256) + 1;
                    if ( *v79 == *(_QWORD *)v10 )
                      goto LABEL_109;
                    v101 = 1;
                    v97 = 0;
                    while ( v89 != -4096 )
                    {
                      if ( v89 == -8192 && !v97 )
                        v97 = v10;
                      v88 = v86 & (v101 + v88);
                      v10 = v87 + 8LL * v88;
                      v89 = *(_QWORD *)v10;
                      if ( v11 == *(_QWORD *)v10 )
                        goto LABEL_109;
                      ++v101;
                    }
LABEL_133:
                    if ( v97 )
                      v10 = v97;
LABEL_109:
                    *(_DWORD *)(a1 + 1256) = v90;
                    if ( *(_QWORD *)v10 != -4096 )
                      --*(_DWORD *)(a1 + 1260);
                    v91 = *v79++;
                    *(_QWORD *)v10 = v91;
                    if ( v80 == v79 )
                    {
LABEL_112:
                      v7 = v126;
                      goto LABEL_17;
                    }
                  }
                }
                ++*(_QWORD *)(a1 + 1240);
                goto LABEL_107;
              }
              goto LABEL_17;
            }
LABEL_16:
            if ( v16 != v15 )
              goto LABEL_17;
            goto LABEL_99;
          }
          if ( v12 == *v15 )
            goto LABEL_16;
          ++v15;
        }
        if ( v12 != *v15 )
        {
          ++v15;
          goto LABEL_118;
        }
        goto LABEL_16;
      }
      v30 = *(_DWORD *)(a1 + 1264);
      if ( v30 )
      {
        v11 = *(_QWORD *)(a1 + 1248);
        v123 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
        v31 = (v30 - 1) & v123;
        v32 = (__int64 *)(v11 + 8LL * v31);
        v10 = *v32;
        if ( v12 == *v32 )
          goto LABEL_17;
        v121 = 1;
        v33 = 0;
        v120 = *(_DWORD *)(a1 + 1256);
        while ( v10 != -4096 )
        {
          if ( v10 == -8192 && !v33 )
            v33 = v32;
          v31 = (v30 - 1) & (v121 + v31);
          v32 = (__int64 *)(v11 + 8LL * v31);
          v10 = *v32;
          if ( v12 == *v32 )
            goto LABEL_17;
          ++v121;
        }
        if ( !v33 )
          v33 = v32;
        ++*(_QWORD *)(a1 + 1240);
        v34 = v13 + 1;
        if ( 4 * (v120 + 1) < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(a1 + 1260) - v34 > v30 >> 3 )
          {
LABEL_40:
            *(_DWORD *)(a1 + 1256) = v34;
            if ( *v33 != -4096 )
              --*(_DWORD *)(a1 + 1260);
            *v33 = v12;
            v35 = *(unsigned int *)(a1 + 1280);
            if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1284) )
            {
              sub_C8D5F0(a1 + 1272, (const void *)(a1 + 1288), v35 + 1, 8u, v10, v11);
              v35 = *(unsigned int *)(a1 + 1280);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 1272) + 8 * v35) = v12;
            ++*(_DWORD *)(a1 + 1280);
LABEL_17:
            v19 = (unsigned int)v134;
            v20 = (unsigned int)v134 + 1LL;
            if ( v20 > HIDWORD(v134) )
            {
              sub_C8D5F0((__int64)&v133, v135, v20, 8u, v10, v11);
              v19 = (unsigned int)v134;
            }
            *(_QWORD *)&v133[8 * v19] = v7;
            LODWORD(v19) = v134 + 1;
            LODWORD(v134) = v134 + 1;
            while ( 1 )
            {
              v23 = (unsigned int)v19;
              v19 = (unsigned int)(v19 - 1);
              v24 = *(_QWORD *)&v133[8 * v23 - 8];
              LODWORD(v134) = v19;
              v25 = *(unsigned __int16 *)(v24 + 24);
              if ( (_WORD)v25 == 8 )
                break;
              if ( (_WORD)v25 == 5 )
              {
                sub_28555C0(
                  (__int64)&v133,
                  &v133[8 * v19],
                  *(char **)(v24 + 32),
                  (char *)(*(_QWORD *)(v24 + 32) + 8LL * *(_QWORD *)(v24 + 40)));
                LODWORD(v19) = v134;
              }
LABEL_24:
              if ( !(_DWORD)v19 )
                goto LABEL_3;
            }
            v21 = *(_QWORD *)(v24 + 48);
            if ( *(_QWORD *)(a1 + 56) != v21 )
            {
LABEL_21:
              v22 = **(_QWORD **)(v24 + 32);
              if ( v19 + 1 > (unsigned __int64)HIDWORD(v134) )
              {
                sub_C8D5F0((__int64)&v133, v135, v19 + 1, 8u, v10, v11);
                v19 = (unsigned int)v134;
              }
              *(_QWORD *)&v133[8 * v19] = v22;
              LODWORD(v19) = v134 + 1;
              LODWORD(v134) = v134 + 1;
              goto LABEL_24;
            }
            v132 = sub_D33D80((_QWORD *)v24, *(_QWORD *)(a1 + 8), v25, v21, v10);
            v26 = v132;
            if ( !(_DWORD)v138 )
            {
              v27 = &v140[(unsigned int)v141];
              if ( v27 == sub_284FC00(v140, (__int64)v27, &v132) )
                sub_2871A60((__int64)&v136, v26, v28, v29, v10, v11);
              goto LABEL_31;
            }
            if ( (_DWORD)v139 )
            {
              v11 = v137;
              v36 = (v139 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
              v37 = (__int64 *)(v137 + 8LL * v36);
              v10 = *v37;
              if ( v132 == *v37 )
              {
LABEL_31:
                v19 = (unsigned int)v134;
                goto LABEL_21;
              }
              v124 = 1;
              v38 = 0;
              while ( v10 != -4096 )
              {
                if ( v10 == -8192 && !v38 )
                  v38 = v37;
                v36 = (v139 - 1) & (v124 + v36);
                v37 = (__int64 *)(v137 + 8LL * v36);
                v10 = *v37;
                if ( v132 == *v37 )
                  goto LABEL_31;
                ++v124;
              }
              if ( v38 )
                v37 = v38;
              ++v136;
              v39 = v138 + 1;
              if ( 4 * ((int)v138 + 1) < (unsigned int)(3 * v139) )
              {
                v10 = (unsigned int)v139 >> 3;
                if ( (int)v139 - HIDWORD(v138) - v39 <= (unsigned int)v10 )
                {
                  v127 = ((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4);
                  sub_2871610((__int64)&v136, v139);
                  if ( !(_DWORD)v139 )
                  {
LABEL_212:
                    LODWORD(v138) = v138 + 1;
                    BUG();
                  }
                  v11 = (unsigned int)(v139 - 1);
                  v117 = 1;
                  v10 = 0;
                  LODWORD(v118) = v11 & v127;
                  v37 = (__int64 *)(v137 + 8LL * ((unsigned int)v11 & v127));
                  v119 = *v37;
                  v39 = v138 + 1;
                  if ( v26 != *v37 )
                  {
                    while ( v119 != -4096 )
                    {
                      if ( v119 == -8192 && !v10 )
                        v10 = (__int64)v37;
                      v118 = (unsigned int)v11 & ((_DWORD)v118 + v117);
                      v37 = (__int64 *)(v137 + 8 * v118);
                      v119 = *v37;
                      if ( v26 == *v37 )
                        goto LABEL_53;
                      ++v117;
                    }
                    if ( v10 )
                      v37 = (__int64 *)v10;
                  }
                }
                goto LABEL_53;
              }
            }
            else
            {
              ++v136;
            }
            sub_2871610((__int64)&v136, 2 * v139);
            if ( !(_DWORD)v139 )
              goto LABEL_212;
            v10 = (unsigned int)(v139 - 1);
            v11 = v137;
            LODWORD(v113) = v10 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v37 = (__int64 *)(v137 + 8LL * (unsigned int)v113);
            v114 = *v37;
            v39 = v138 + 1;
            if ( v26 != *v37 )
            {
              v115 = 1;
              v116 = 0;
              while ( v114 != -4096 )
              {
                if ( !v116 && v114 == -8192 )
                  v116 = v37;
                v113 = (unsigned int)v10 & ((_DWORD)v113 + v115);
                v37 = (__int64 *)(v137 + 8 * v113);
                v114 = *v37;
                if ( v26 == *v37 )
                  goto LABEL_53;
                ++v115;
              }
              if ( v116 )
                v37 = v116;
            }
LABEL_53:
            LODWORD(v138) = v39;
            if ( *v37 != -4096 )
              --HIDWORD(v138);
            *v37 = v26;
            v40 = (unsigned int)v141;
            v41 = (unsigned int)v141 + 1LL;
            if ( v41 > HIDWORD(v141) )
            {
              sub_C8D5F0((__int64)&v140, v142, v41, 8u, v10, v11);
              v40 = (unsigned int)v141;
            }
            v140[v40] = v26;
            LODWORD(v141) = v141 + 1;
            goto LABEL_31;
          }
          sub_BCFDB0(a1 + 1240, v30);
          v107 = *(_DWORD *)(a1 + 1264);
          if ( v107 )
          {
            v108 = v107 - 1;
            v109 = 1;
            v110 = 0;
            v10 = *(_QWORD *)(a1 + 1248);
            LODWORD(v111) = v108 & v123;
            v33 = (__int64 *)(v10 + 8LL * (v108 & v123));
            v112 = *v33;
            v34 = *(_DWORD *)(a1 + 1256) + 1;
            if ( v12 != *v33 )
            {
              while ( v112 != -4096 )
              {
                if ( !v110 && v112 == -8192 )
                  v110 = v33;
                v11 = (unsigned int)(v109 + 1);
                v111 = v108 & (unsigned int)(v111 + v109);
                v33 = (__int64 *)(v10 + 8 * v111);
                v112 = *v33;
                if ( v12 == *v33 )
                  goto LABEL_40;
                ++v109;
              }
              if ( v110 )
                v33 = v110;
            }
            goto LABEL_40;
          }
LABEL_211:
          ++*(_DWORD *)(a1 + 1256);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 1240);
      }
      sub_BCFDB0(a1 + 1240, 2 * v30);
      v102 = *(_DWORD *)(a1 + 1264);
      if ( v102 )
      {
        v103 = v102 - 1;
        v11 = *(_QWORD *)(a1 + 1248);
        LODWORD(v104) = v103 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v33 = (__int64 *)(v11 + 8LL * (unsigned int)v104);
        v10 = *v33;
        v34 = *(_DWORD *)(a1 + 1256) + 1;
        if ( v12 != *v33 )
        {
          v105 = 1;
          v106 = 0;
          while ( v10 != -4096 )
          {
            if ( !v106 && v10 == -8192 )
              v106 = v33;
            v104 = v103 & (unsigned int)(v104 + v105);
            v33 = (__int64 *)(v11 + 8 * v104);
            v10 = *v33;
            if ( v12 == *v33 )
              goto LABEL_40;
            ++v105;
          }
          if ( v106 )
            v33 = v106;
        }
        goto LABEL_40;
      }
      goto LABEL_211;
    }
LABEL_3:
    v4 = *(_QWORD *)(v4 + 8);
    if ( v129 == v4 )
      break;
    v2 = *(_QWORD *)a1;
  }
  v128 = v140;
  v125 = &v140[(unsigned int)v141];
  if ( v125 == v140 )
    goto LABEL_89;
LABEL_59:
  v42 = (_QWORD **)++v128;
  if ( v125 != v128 )
  {
    while ( 1 )
    {
      v45 = *(_QWORD *)(a1 + 8);
      v46 = *v42;
      v47 = (_QWORD *)*(v128 - 1);
      v48 = sub_D95540((__int64)v47);
      v49 = sub_D97050(v45, v48);
      v50 = *(_QWORD *)(a1 + 8);
      v130 = v49;
      v51 = sub_D95540((__int64)v46);
      if ( v130 != sub_D97050(v50, v51) )
      {
        v52 = *(_QWORD *)(a1 + 8);
        v53 = sub_D95540((__int64)v47);
        v54 = sub_D97050(v52, v53);
        v55 = *(_QWORD *)(a1 + 8);
        v131 = v54;
        v56 = sub_D95540((__int64)v46);
        v57 = sub_D97050(v55, v56);
        v58 = *(_QWORD *)(a1 + 8);
        if ( v131 <= v57 )
        {
          v76 = sub_D95540((__int64)v46);
          v47 = sub_DC5000(v58, (__int64)v47, v76, 0);
        }
        else
        {
          v59 = sub_D95540((__int64)v47);
          v46 = sub_DC5000(v58, (__int64)v46, v59, 0);
        }
      }
      v60 = sub_285DD00((__int64)v46, (__int64)v47, *(__int64 **)(a1 + 8), 1);
      v61 = (__int64)v60;
      if ( !v60 || *((_WORD *)v60 + 12) )
      {
        v75 = sub_285DD00((__int64)v47, (__int64)v46, *(__int64 **)(a1 + 8), 1);
        v61 = (__int64)v75;
        if ( !v75 || *((_WORD *)v75 + 12) )
          goto LABEL_66;
      }
      v62 = *(_QWORD *)(v61 + 32);
      v63 = *(_DWORD *)(v62 + 32);
      v64 = *(_QWORD *)(v62 + 24);
      v65 = v63 - 1;
      v66 = 1LL << ((unsigned __int8)v63 - 1);
      if ( v63 <= 0x40 )
        break;
      v67 = v62 + 24;
      v68 = v63 + 1;
      if ( (*(_QWORD *)(v64 + 8LL * (v65 >> 6)) & v66) == 0 )
      {
        v44 = v68 - sub_C444A0(v67);
LABEL_65:
        if ( v44 <= 0x40 )
          goto LABEL_75;
        goto LABEL_66;
      }
      if ( v68 - (unsigned int)sub_C44500(v67) <= 0x40 )
        goto LABEL_75;
LABEL_66:
      if ( ++v42 == v125 )
        goto LABEL_59;
    }
    if ( (v66 & v64) != 0 )
    {
      if ( !v63 )
        goto LABEL_75;
      v43 = ~(v64 << (64 - (unsigned __int8)v63));
      if ( v43 )
      {
        _BitScanReverse64(&v43, v43);
        v44 = v63 + 1 - (v43 ^ 0x3F);
      }
      else
      {
        v44 = v63 - 63;
      }
    }
    else
    {
      if ( !v64 )
      {
LABEL_75:
        if ( !sub_D968A0(v61) )
        {
          v71 = *(_QWORD *)(v61 + 32);
          v72 = *(_QWORD *)(v71 + 24);
          v73 = *(_DWORD *)(v71 + 32);
          if ( v73 > 0x40 )
          {
            v74 = *(_QWORD *)v72;
          }
          else
          {
            v74 = 0;
            if ( v73 )
            {
              v72 = v72 << (64 - (unsigned __int8)v73) >> (64 - (unsigned __int8)v73);
              v74 = v72;
            }
          }
          v132 = v74;
          sub_2872920(a1 + 968, &v132, v72, v74, v69, v70);
        }
        goto LABEL_66;
      }
      _BitScanReverse64(&v64, v64);
      v44 = 65 - (v64 ^ 0x3F);
    }
    goto LABEL_65;
  }
LABEL_89:
  if ( *(_DWORD *)(a1 + 1280) == 1 )
    goto LABEL_123;
LABEL_90:
  if ( v133 != v135 )
    _libc_free((unsigned __int64)v133);
LABEL_92:
  if ( v140 != (_QWORD *)v142 )
    _libc_free((unsigned __int64)v140);
  return sub_C7D6A0(v137, 8LL * (unsigned int)v139, 8);
}
