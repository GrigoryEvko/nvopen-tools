// Function: sub_3094B40
// Address: 0x3094b40
//
__int64 __fastcall sub_3094B40(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rbx
  unsigned int v17; // eax
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r12
  unsigned int v25; // r13d
  unsigned int v26; // edi
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rax
  unsigned int v30; // esi
  unsigned int v31; // r14d
  int v32; // r11d
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // rdx
  unsigned int *v36; // rax
  __int64 v37; // r8
  __int64 v38; // rsi
  __int64 v39; // rbx
  __int64 v40; // rbx
  __int64 v41; // r14
  __int64 *v42; // rbx
  __int64 *v43; // r13
  __int64 v44; // rdi
  __int64 v46; // rax
  int *v47; // rdx
  int v48; // eax
  int *v49; // rax
  int *v50; // rsi
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // ecx
  __int64 v57; // rsi
  __int64 v58; // rdi
  int v59; // ecx
  unsigned int v60; // edx
  __int64 *v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rcx
  int *v65; // rdx
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  int v68; // edx
  __int64 v69; // r11
  int i; // r10d
  __int64 v71; // r13
  int v72; // r10d
  __int64 *v73; // rax
  unsigned int v74; // edi
  __int64 *v75; // rdx
  __int64 v76; // rcx
  __int64 *v77; // rax
  __int64 *v78; // r15
  __int64 v79; // rdx
  __int64 v80; // rax
  unsigned int v81; // ecx
  __int64 v82; // rax
  int v83; // r10d
  int v84; // r10d
  __int64 *v85; // r9
  int v86; // edx
  __int64 *v87; // r12
  __int64 *v88; // rax
  __int64 v89; // rdi
  __int64 *v90; // rbx
  int v91; // esi
  unsigned int v92; // r12d
  __int64 v93; // rdi
  __int64 v94; // rax
  unsigned int v95; // r13d
  __int64 v96; // rsi
  int v97; // ecx
  __int64 *v98; // rax
  int v99; // ecx
  unsigned int v100; // r13d
  __int64 v101; // rsi
  int v102; // edx
  __int64 v103; // rcx
  __int64 v104; // r8
  int v105; // edi
  __int64 *v106; // rsi
  int v107; // esi
  __int64 v108; // r15
  __int64 *v109; // rcx
  __int64 v110; // r8
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // rax
  signed __int64 v115; // rdx
  __int64 v116; // rsi
  __int64 v117; // rax
  unsigned __int64 v118; // rdx
  __int64 v119; // rdx
  __int64 v120; // rdi
  __int64 v121; // rdx
  __int64 v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rdx
  __int64 v125; // rdx
  __int64 v126; // rdx
  __int64 v127; // rdx
  __int64 v128; // rdx
  __int64 v129; // rsi
  __int64 v130; // rdx
  __int64 v131; // rdx
  __int64 v132; // rsi
  __int64 v133; // rdx
  __int64 v134; // rdx
  __int64 v135; // rsi
  __int64 v136; // rdx
  int v137; // eax
  unsigned int v138; // r12d
  __int64 *v139; // [rsp+0h] [rbp-110h]
  char v140; // [rsp+Fh] [rbp-101h]
  __int64 *v141; // [rsp+10h] [rbp-100h]
  __int64 v142; // [rsp+20h] [rbp-F0h]
  __int64 v143; // [rsp+28h] [rbp-E8h]
  __int64 v144; // [rsp+30h] [rbp-E0h]
  _QWORD *v145; // [rsp+38h] [rbp-D8h]
  __int64 v146; // [rsp+40h] [rbp-D0h]
  __int64 v147; // [rsp+48h] [rbp-C8h]
  __int64 v148; // [rsp+48h] [rbp-C8h]
  __int64 v149; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v150; // [rsp+58h] [rbp-B8h]
  __int64 v151; // [rsp+60h] [rbp-B0h]
  unsigned int v152; // [rsp+68h] [rbp-A8h]
  __int64 v153; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v154; // [rsp+78h] [rbp-98h]
  __int64 v155; // [rsp+80h] [rbp-90h]
  unsigned int v156; // [rsp+88h] [rbp-88h]
  __int64 *v157; // [rsp+90h] [rbp-80h] BYREF
  __int64 v158; // [rsp+98h] [rbp-78h]
  _BYTE v159[112]; // [rsp+A0h] [rbp-70h] BYREF

  v2 = a1;
  v140 = sub_BB98D0(a1, *a2);
  if ( v140 )
    return 0;
  a1[25] = a2;
  v3 = a2[2];
  a1[26] = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 128LL))(v3);
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  v5 = (__int64 *)a1[1];
  a1[27] = v4;
  a1[28] = *(_QWORD *)(a1[25] + 32LL);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_327:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_50208AC )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_327;
  }
  a1[29] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(
             *(_QWORD *)(v6 + 8),
             &unk_50208AC)
         + 200;
  v8 = a1[25];
  v9 = *(_QWORD *)(v8 + 328);
  v157 = (__int64 *)v159;
  v158 = 0x600000000LL;
  v10 = *(_QWORD *)(v9 + 56);
  v11 = v9 + 48;
  v146 = v9;
  if ( v10 == v9 + 48 )
  {
    v138 = 0;
    goto LABEL_41;
  }
  do
  {
    while ( 1 )
    {
      if ( !(unsigned int)sub_30941A0(v10) )
      {
        if ( !v10 )
          BUG();
        goto LABEL_11;
      }
      v46 = *(_QWORD *)(v10 + 48);
      v47 = (int *)(v46 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v46 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_61;
      v48 = v46 & 7;
      if ( !v48 )
      {
        *(_QWORD *)(v10 + 48) = v47;
        v49 = (int *)(v10 + 48);
        v50 = (int *)(v10 + 56);
        goto LABEL_60;
      }
      if ( v48 != 3 )
        goto LABEL_61;
      v49 = v47 + 4;
      v62 = 8LL * *v47;
      v50 = &v47[(unsigned __int64)v62 / 4 + 4];
      v63 = v62 >> 5;
      v64 = v62 >> 3;
      if ( v63 <= 0 )
      {
LABEL_257:
        if ( v64 != 2 )
        {
          if ( v64 != 3 )
          {
            if ( v64 != 1 )
              goto LABEL_61;
            goto LABEL_60;
          }
          if ( *(_BYTE *)(*(_QWORD *)v49 + 84LL) )
            goto LABEL_84;
          v49 += 2;
        }
        if ( *(_BYTE *)(*(_QWORD *)v49 + 84LL) )
          goto LABEL_84;
        v49 += 2;
LABEL_60:
        if ( !*(_BYTE *)(*(_QWORD *)v49 + 84LL) )
          goto LABEL_61;
        goto LABEL_84;
      }
      v65 = &v49[8 * v63];
      while ( !*(_BYTE *)(*(_QWORD *)v49 + 84LL) )
      {
        if ( *(_BYTE *)(*((_QWORD *)v49 + 1) + 84LL) )
        {
          v49 += 2;
          break;
        }
        if ( *(_BYTE *)(*((_QWORD *)v49 + 2) + 84LL) )
        {
          v49 += 4;
          break;
        }
        if ( *(_BYTE *)(*((_QWORD *)v49 + 3) + 84LL) )
        {
          v49 += 6;
          break;
        }
        v49 += 8;
        if ( v65 == v49 )
        {
          v64 = ((char *)v50 - (char *)v49) >> 3;
          goto LABEL_257;
        }
      }
LABEL_84:
      if ( v49 != v50 )
        goto LABEL_85;
LABEL_61:
      if ( !(_BYTE)qword_502D4E8 )
        goto LABEL_11;
      v51 = *(_QWORD *)(v10 + 32);
      v52 = v2[28];
      v53 = *(unsigned int *)(v51 + 8);
      v54 = (int)v53 < 0
          ? *(_QWORD *)(*(_QWORD *)(v52 + 56) + 16 * (v53 & 0x7FFFFFFF) + 8)
          : *(_QWORD *)(*(_QWORD *)(v52 + 304) + 8 * v53);
      if ( !v54 )
        goto LABEL_11;
      if ( (*(_BYTE *)(v54 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v54 = *(_QWORD *)(v54 + 32);
          if ( !v54 )
            break;
          if ( (*(_BYTE *)(v54 + 3) & 0x10) == 0 )
            goto LABEL_68;
        }
      }
      else
      {
        v54 = *(_QWORD *)(v54 + 32);
        if ( !v54 )
        {
LABEL_69:
          v55 = v2[29];
          v56 = *(_DWORD *)(v55 + 24);
          v57 = *(_QWORD *)(*(_QWORD *)(v51 + 16) + 24LL);
          v58 = *(_QWORD *)(v55 + 8);
          if ( v56 )
          {
            v59 = v56 - 1;
            v60 = v59 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v61 = (__int64 *)(v58 + 16LL * v60);
            v13 = *v61;
            if ( v57 == *v61 )
            {
LABEL_71:
              if ( v61[1] )
                goto LABEL_11;
            }
            else
            {
              v137 = 1;
              while ( v13 != -4096 )
              {
                v12 = (unsigned int)(v137 + 1);
                v60 = v59 & (v137 + v60);
                v61 = (__int64 *)(v58 + 16LL * v60);
                v13 = *v61;
                if ( v57 == *v61 )
                  goto LABEL_71;
                v137 = v12;
              }
            }
          }
LABEL_85:
          v66 = (unsigned int)v158;
          v67 = (unsigned int)v158 + 1LL;
          if ( v67 > HIDWORD(v158) )
          {
            sub_C8D5F0((__int64)&v157, v159, v67, 8u, v12, v13);
            v66 = (unsigned int)v158;
          }
          v157[v66] = v10;
          LODWORD(v158) = v158 + 1;
          goto LABEL_11;
        }
        while ( (*(_BYTE *)(v54 + 3) & 0x10) != 0 )
        {
LABEL_68:
          v54 = *(_QWORD *)(v54 + 32);
          if ( !v54 )
            goto LABEL_69;
        }
      }
LABEL_11:
      if ( (*(_BYTE *)v10 & 4) == 0 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v11 == v10 )
        goto LABEL_13;
    }
    while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
      v10 = *(_QWORD *)(v10 + 8);
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v11 != v10 );
LABEL_13:
  v14 = v157;
  if ( (_DWORD)v158 )
  {
    v141 = v157;
    v139 = &v157[(unsigned int)v158];
    v145 = v2;
    do
    {
      v15 = *v141;
      v16 = *(_QWORD *)(*v141 + 32);
      v143 = *v141;
      v17 = sub_2E88FE0(*v141);
      v147 = *(_QWORD *)(v15 + 32);
      v142 = v16 + 40LL * v17;
      if ( v142 != v147 )
      {
        while ( 1 )
        {
          v20 = *(_DWORD *)(v147 + 8);
          v21 = v145[28];
          v149 = 0;
          v150 = 0;
          v151 = 0;
          v152 = 0;
          v153 = 0;
          v154 = 0;
          v155 = 0;
          v156 = 0;
          if ( v20 < 0 )
            v22 = *(_QWORD *)(*(_QWORD *)(v21 + 56) + 16LL * (v20 & 0x7FFFFFFF) + 8);
          else
            v22 = *(_QWORD *)(*(_QWORD *)(v21 + 304) + 8LL * (unsigned int)v20);
          if ( !v22 )
            goto LABEL_90;
          if ( (*(_BYTE *)(v22 + 3) & 0x10) != 0 )
            break;
LABEL_20:
          v144 = 16LL * (v20 & 0x7FFFFFFF);
LABEL_21:
          v23 = *(_QWORD *)(v22 + 16);
          v24 = *(_QWORD *)(v23 + 24);
          if ( !*(_WORD *)(v23 + 68) || *(_WORD *)(v23 + 68) == 68 )
            v24 = *(_QWORD *)(*(_QWORD *)(v23 + 32)
                            + 40LL * (-858993459 * (unsigned int)((v22 - *(_QWORD *)(v23 + 32)) >> 3) + 1)
                            + 24);
          if ( v146 != v24 )
          {
            if ( v152 )
            {
              v18 = v152 - 1;
              v25 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
              v26 = v18 & v25;
              v27 = (__int64 *)(v150 + 16LL * ((unsigned int)v18 & v25));
              v28 = *v27;
              if ( *v27 == v24 )
              {
LABEL_27:
                v29 = v27[1];
                goto LABEL_28;
              }
              v19 = *v27;
              LODWORD(v69) = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              for ( i = 1; ; ++i )
              {
                if ( v19 == -4096 )
                  goto LABEL_109;
                v69 = (unsigned int)v18 & ((_DWORD)v69 + i);
                v19 = *(_QWORD *)(v150 + 16 * v69);
                if ( v19 == v24 )
                  break;
              }
              v84 = 1;
              v85 = 0;
              while ( v28 != -4096 )
              {
                if ( v28 == -8192 && !v85 )
                  v85 = v27;
                v26 = v18 & (v84 + v26);
                v27 = (__int64 *)(v150 + 16LL * v26);
                v28 = *v27;
                if ( *v27 == v24 )
                  goto LABEL_27;
                ++v84;
              }
              if ( !v85 )
                v85 = v27;
              ++v149;
              v86 = v151 + 1;
              if ( 4 * ((int)v151 + 1) >= 3 * v152 )
              {
                sub_3094780((__int64)&v149, 2 * v152);
                if ( v152 )
                {
                  v95 = (v152 - 1) & v25;
                  v86 = v151 + 1;
                  v85 = (__int64 *)(v150 + 16LL * v95);
                  v96 = *v85;
                  if ( *v85 == v24 )
                    goto LABEL_128;
                  v97 = 1;
                  v98 = 0;
                  while ( v96 != -4096 )
                  {
                    if ( v96 == -8192 && !v98 )
                      v98 = v85;
                    v95 = (v152 - 1) & (v97 + v95);
                    v85 = (__int64 *)(v150 + 16LL * v95);
                    v96 = *v85;
                    if ( *v85 == v24 )
                      goto LABEL_128;
                    ++v97;
                  }
LABEL_155:
                  if ( v98 )
                    v85 = v98;
                  goto LABEL_128;
                }
              }
              else
              {
                if ( v152 - HIDWORD(v151) - v86 > v152 >> 3 )
                {
LABEL_128:
                  LODWORD(v151) = v86;
                  if ( *v85 != -4096 )
                    --HIDWORD(v151);
                  *v85 = v24;
                  v29 = 0;
                  v85[1] = 0;
LABEL_28:
                  v30 = v156;
                  v31 = *(_DWORD *)(*(_QWORD *)(v29 + 32) + 8LL);
                  if ( v156 )
                  {
LABEL_29:
                    v18 = v30 - 1;
                    v32 = 1;
                    v19 = 0;
                    v33 = v18 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                    v34 = &v154[2 * v33];
                    v35 = *v34;
                    if ( *v34 == v22 )
                    {
LABEL_30:
                      v36 = (unsigned int *)(v34 + 1);
LABEL_31:
                      *v36 = v31;
                      goto LABEL_33;
                    }
                    while ( v35 != -4096 )
                    {
                      if ( !v19 && v35 == -8192 )
                        v19 = (__int64)v34;
                      v33 = v18 & (v32 + v33);
                      v34 = &v154[2 * v33];
                      v35 = *v34;
                      if ( v22 == *v34 )
                        goto LABEL_30;
                      ++v32;
                    }
                    if ( !v19 )
                      v19 = (__int64)v34;
                    ++v153;
                    v68 = v155 + 1;
                    if ( 4 * ((int)v155 + 1) < 3 * v30 )
                    {
                      if ( v30 - HIDWORD(v155) - v68 > v30 >> 3 )
                      {
LABEL_104:
                        LODWORD(v155) = v68;
                        if ( *(_QWORD *)v19 != -4096 )
                          --HIDWORD(v155);
                        *(_QWORD *)v19 = v22;
                        v36 = (unsigned int *)(v19 + 8);
                        *(_DWORD *)(v19 + 8) = 0;
                        goto LABEL_31;
                      }
                      sub_3094960((__int64)&v153, v30);
                      if ( v156 )
                      {
                        v18 = 1;
                        v92 = (v156 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                        v68 = v155 + 1;
                        v93 = 0;
                        v19 = (__int64)&v154[2 * v92];
                        v94 = *(_QWORD *)v19;
                        if ( v22 != *(_QWORD *)v19 )
                        {
                          while ( v94 != -4096 )
                          {
                            if ( !v93 && v94 == -8192 )
                              v93 = v19;
                            v92 = (v156 - 1) & (v18 + v92);
                            v19 = (__int64)&v154[2 * v92];
                            v94 = *(_QWORD *)v19;
                            if ( v22 == *(_QWORD *)v19 )
                              goto LABEL_104;
                            v18 = (unsigned int)(v18 + 1);
                          }
                          if ( v93 )
                            v19 = v93;
                        }
                        goto LABEL_104;
                      }
LABEL_325:
                      LODWORD(v155) = v155 + 1;
                      BUG();
                    }
LABEL_114:
                    sub_3094960((__int64)&v153, 2 * v30);
                    if ( v156 )
                    {
                      v68 = v155 + 1;
                      v81 = (v156 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                      v19 = (__int64)&v154[2 * v81];
                      v82 = *(_QWORD *)v19;
                      if ( v22 != *(_QWORD *)v19 )
                      {
                        v83 = 1;
                        v18 = 0;
                        while ( v82 != -4096 )
                        {
                          if ( v82 == -8192 && !v18 )
                            v18 = v19;
                          v81 = (v156 - 1) & (v83 + v81);
                          v19 = (__int64)&v154[2 * v81];
                          v82 = *(_QWORD *)v19;
                          if ( v22 == *(_QWORD *)v19 )
                            goto LABEL_104;
                          ++v83;
                        }
                        if ( v18 )
                          v19 = v18;
                      }
                      goto LABEL_104;
                    }
                    goto LABEL_325;
                  }
LABEL_113:
                  ++v153;
                  goto LABEL_114;
                }
                sub_3094780((__int64)&v149, v152);
                if ( v152 )
                {
                  v99 = 1;
                  v100 = (v152 - 1) & v25;
                  v86 = v151 + 1;
                  v98 = 0;
                  v85 = (__int64 *)(v150 + 16LL * v100);
                  v101 = *v85;
                  if ( *v85 == v24 )
                    goto LABEL_128;
                  while ( v101 != -4096 )
                  {
                    if ( v101 == -8192 && !v98 )
                      v98 = v85;
                    v100 = (v152 - 1) & (v99 + v100);
                    v85 = (__int64 *)(v150 + 16LL * v100);
                    v101 = *v85;
                    if ( *v85 == v24 )
                      goto LABEL_128;
                    ++v99;
                  }
                  goto LABEL_155;
                }
              }
              LODWORD(v151) = v151 + 1;
              BUG();
            }
LABEL_109:
            v31 = sub_2EC06C0(
                    v145[28],
                    *(_QWORD *)(*(_QWORD *)(v145[28] + 56LL) + v144) & 0xFFFFFFFFFFFFFFF8LL,
                    byte_3F871B3,
                    0,
                    v18,
                    v19);
            v71 = (__int64)sub_2E7B2C0(*(_QWORD **)(v24 + 32), v143);
            sub_2E8A790(v71, *(_DWORD *)(*(_QWORD *)(v71 + 32) + 8LL), v31, 0, (_QWORD *)v145[27]);
            if ( v152 )
            {
              v72 = 1;
              v73 = 0;
              v74 = (v152 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v75 = (__int64 *)(v150 + 16LL * v74);
              v76 = *v75;
              if ( *v75 == v24 )
              {
LABEL_111:
                v77 = v75 + 1;
                goto LABEL_112;
              }
              while ( v76 != -4096 )
              {
                if ( !v73 && v76 == -8192 )
                  v73 = v75;
                v74 = (v152 - 1) & (v72 + v74);
                v75 = (__int64 *)(v150 + 16LL * v74);
                v76 = *v75;
                if ( *v75 == v24 )
                  goto LABEL_111;
                ++v72;
              }
              if ( !v73 )
                v73 = v75;
              ++v149;
              v102 = v151 + 1;
              if ( 4 * ((int)v151 + 1) < 3 * v152 )
              {
                if ( v152 - HIDWORD(v151) - v102 > v152 >> 3 )
                {
LABEL_176:
                  LODWORD(v151) = v102;
                  if ( *v73 != -4096 )
                    --HIDWORD(v151);
                  *v73 = v24;
                  v77 = v73 + 1;
                  *v77 = 0;
LABEL_112:
                  *v77 = v71;
                  v78 = (__int64 *)sub_2E311E0(v24);
                  sub_2E31040((__int64 *)(v24 + 40), v71);
                  v79 = *v78;
                  v80 = *(_QWORD *)v71;
                  *(_QWORD *)(v71 + 8) = v78;
                  v79 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)v71 = v79 | v80 & 7;
                  *(_QWORD *)(v79 + 8) = v71;
                  *v78 = *v78 & 7 | v71;
                  v30 = v156;
                  if ( v156 )
                    goto LABEL_29;
                  goto LABEL_113;
                }
                sub_3094780((__int64)&v149, v152);
                if ( v152 )
                {
                  v107 = 1;
                  LODWORD(v108) = (v152 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
                  v102 = v151 + 1;
                  v109 = 0;
                  v73 = (__int64 *)(v150 + 16LL * (unsigned int)v108);
                  v110 = *v73;
                  if ( v24 != *v73 )
                  {
                    while ( v110 != -4096 )
                    {
                      if ( v110 == -8192 && !v109 )
                        v109 = v73;
                      v108 = (v152 - 1) & ((_DWORD)v108 + v107);
                      v73 = (__int64 *)(v150 + 16 * v108);
                      v110 = *v73;
                      if ( *v73 == v24 )
                        goto LABEL_176;
                      ++v107;
                    }
                    if ( v109 )
                      v73 = v109;
                  }
                  goto LABEL_176;
                }
LABEL_324:
                LODWORD(v151) = v151 + 1;
                BUG();
              }
            }
            else
            {
              ++v149;
            }
            sub_3094780((__int64)&v149, 2 * v152);
            if ( v152 )
            {
              v102 = v151 + 1;
              LODWORD(v103) = (v152 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v73 = (__int64 *)(v150 + 16LL * (unsigned int)v103);
              v104 = *v73;
              if ( *v73 != v24 )
              {
                v105 = 1;
                v106 = 0;
                while ( v104 != -4096 )
                {
                  if ( !v106 && v104 == -8192 )
                    v106 = v73;
                  v103 = (v152 - 1) & ((_DWORD)v103 + v105);
                  v73 = (__int64 *)(v150 + 16 * v103);
                  v104 = *v73;
                  if ( *v73 == v24 )
                    goto LABEL_176;
                  ++v105;
                }
                if ( v106 )
                  v73 = v106;
              }
              goto LABEL_176;
            }
            goto LABEL_324;
          }
LABEL_33:
          while ( 1 )
          {
            v22 = *(_QWORD *)(v22 + 32);
            if ( !v22 )
              break;
            if ( (*(_BYTE *)(v22 + 3) & 0x10) == 0 )
              goto LABEL_21;
          }
          v37 = (__int64)v154;
          v38 = 2LL * v156;
          if ( (_DWORD)v155 )
          {
            v87 = &v154[v38];
            if ( &v154[v38] != v154 )
            {
              v88 = v154;
              while ( 1 )
              {
                v89 = *v88;
                v90 = v88;
                if ( *v88 != -4096 && v89 != -8192 )
                  break;
                v88 += 2;
                if ( v87 == v88 )
                  goto LABEL_35;
              }
              if ( v87 != v88 )
              {
                do
                {
                  v91 = *((_DWORD *)v90 + 2);
                  v90 += 2;
                  sub_2EAB0C0(v89, v91);
                  if ( v90 == v87 )
                    break;
                  while ( 1 )
                  {
                    v89 = *v90;
                    if ( *v90 != -8192 && v89 != -4096 )
                      break;
                    v90 += 2;
                    if ( v87 == v90 )
                      goto LABEL_142;
                  }
                }
                while ( v87 != v90 );
LABEL_142:
                v37 = (__int64)v154;
                v38 = 2LL * v156;
              }
            }
          }
LABEL_35:
          sub_C7D6A0(v37, v38 * 8, 8);
          sub_C7D6A0(v150, 16LL * v152, 8);
          v147 += 40;
          if ( v147 == v142 )
            goto LABEL_36;
        }
        while ( 1 )
        {
          v22 = *(_QWORD *)(v22 + 32);
          if ( !v22 )
            break;
          if ( (*(_BYTE *)(v22 + 3) & 0x10) == 0 )
            goto LABEL_20;
        }
LABEL_90:
        v37 = 0;
        v38 = 0;
        goto LABEL_35;
      }
LABEL_36:
      ++v141;
    }
    while ( v139 != v141 );
    v2 = v145;
    v14 = v157;
    v138 = 1;
  }
  else
  {
    v138 = 0;
  }
  if ( v14 != (__int64 *)v159 )
    _libc_free((unsigned __int64)v14);
  v8 = v2[25];
LABEL_41:
  v39 = *(_QWORD *)(v8 + 328);
  v157 = (__int64 *)v159;
  v40 = v39 + 48;
  v158 = 0x800000000LL;
  v41 = *(_QWORD *)(v40 + 8);
  if ( v41 == v40 )
    return v138;
  do
  {
    while ( 1 )
    {
      if ( !(unsigned int)sub_30941A0(v41) )
      {
        if ( !v41 )
          BUG();
        goto LABEL_46;
      }
      v148 = *(_QWORD *)(v41 + 32);
      v111 = v148 + 40LL * (unsigned int)sub_2E88FE0(v41);
      v114 = *(_QWORD *)(v41 + 32);
      v115 = 0xCCCCCCCCCCCCCCCDLL * ((v111 - v114) >> 3);
      if ( v115 >> 2 > 0 )
      {
        v116 = v114 + 160 * (v115 >> 2);
        while ( !*(_BYTE *)v114 )
        {
          v119 = *(unsigned int *)(v114 + 8);
          v120 = v2[28];
          if ( (int)v119 < 0 )
          {
            v121 = *(_QWORD *)(*(_QWORD *)(v120 + 56) + 16 * (v119 & 0x7FFFFFFF) + 8);
          }
          else
          {
            v112 = *(_QWORD *)(v120 + 304);
            v121 = *(_QWORD *)(v112 + 8 * v119);
          }
          if ( v121 )
          {
            if ( (*(_BYTE *)(v121 + 3) & 0x10) == 0 )
              break;
            while ( 1 )
            {
              v121 = *(_QWORD *)(v121 + 32);
              if ( !v121 )
                break;
              if ( (*(_BYTE *)(v121 + 3) & 0x10) == 0 )
                goto LABEL_196;
            }
          }
          v112 = v114 + 40;
          if ( *(_BYTE *)(v114 + 40) )
            goto LABEL_208;
          v122 = *(unsigned int *)(v114 + 48);
          if ( (int)v122 < 0 )
          {
            v123 = *(_QWORD *)(*(_QWORD *)(v120 + 56) + 16 * (v122 & 0x7FFFFFFF) + 8);
          }
          else
          {
            v113 = *(_QWORD *)(v120 + 304);
            v123 = *(_QWORD *)(v113 + 8 * v122);
          }
          if ( v123 )
          {
            if ( (*(_BYTE *)(v123 + 3) & 0x10) == 0 )
              goto LABEL_208;
            while ( 1 )
            {
              v123 = *(_QWORD *)(v123 + 32);
              if ( !v123 )
                break;
              if ( (*(_BYTE *)(v123 + 3) & 0x10) == 0 )
                goto LABEL_208;
            }
          }
          v112 = v114 + 80;
          if ( *(_BYTE *)(v114 + 80) )
            goto LABEL_208;
          v124 = *(unsigned int *)(v114 + 88);
          if ( (int)v124 < 0 )
          {
            v125 = *(_QWORD *)(*(_QWORD *)(v120 + 56) + 16 * (v124 & 0x7FFFFFFF) + 8);
          }
          else
          {
            v113 = *(_QWORD *)(v120 + 304);
            v125 = *(_QWORD *)(v113 + 8 * v124);
          }
          if ( v125 )
          {
            if ( (*(_BYTE *)(v125 + 3) & 0x10) == 0 )
              goto LABEL_208;
            while ( 1 )
            {
              v125 = *(_QWORD *)(v125 + 32);
              if ( !v125 )
                break;
              if ( (*(_BYTE *)(v125 + 3) & 0x10) == 0 )
                goto LABEL_208;
            }
          }
          v112 = v114 + 120;
          if ( *(_BYTE *)(v114 + 120) )
          {
LABEL_208:
            v114 = v112;
            break;
          }
          v126 = *(unsigned int *)(v114 + 128);
          if ( (int)v126 < 0 )
            v127 = *(_QWORD *)(*(_QWORD *)(v120 + 56) + 16 * (v126 & 0x7FFFFFFF) + 8);
          else
            v127 = *(_QWORD *)(*(_QWORD *)(v120 + 304) + 8 * v126);
          if ( v127 )
          {
            if ( (*(_BYTE *)(v127 + 3) & 0x10) == 0 )
              goto LABEL_208;
            while ( 1 )
            {
              v127 = *(_QWORD *)(v127 + 32);
              if ( !v127 )
                break;
              if ( (*(_BYTE *)(v127 + 3) & 0x10) == 0 )
                goto LABEL_208;
            }
          }
          v114 += 160;
          if ( v116 == v114 )
          {
            v115 = 0xCCCCCCCCCCCCCCCDLL * ((v111 - v114) >> 3);
            goto LABEL_234;
          }
        }
LABEL_196:
        if ( v111 != v114 )
          goto LABEL_46;
        goto LABEL_197;
      }
LABEL_234:
      if ( v115 != 2 )
      {
        if ( v115 != 3 )
        {
          if ( v115 != 1 )
            goto LABEL_197;
          goto LABEL_237;
        }
        if ( *(_BYTE *)v114 )
          goto LABEL_196;
        v131 = *(unsigned int *)(v114 + 8);
        v132 = v2[28];
        if ( (int)v131 < 0 )
          v133 = *(_QWORD *)(*(_QWORD *)(v132 + 56) + 16 * (v131 & 0x7FFFFFFF) + 8);
        else
          v133 = *(_QWORD *)(*(_QWORD *)(v132 + 304) + 8 * v131);
        if ( v133 )
        {
          if ( (*(_BYTE *)(v133 + 3) & 0x10) == 0 )
            goto LABEL_196;
          while ( 1 )
          {
            v133 = *(_QWORD *)(v133 + 32);
            if ( !v133 )
              break;
            if ( (*(_BYTE *)(v133 + 3) & 0x10) == 0 )
              goto LABEL_196;
          }
        }
        v114 += 40;
      }
      if ( *(_BYTE *)v114 )
        goto LABEL_196;
      v134 = *(unsigned int *)(v114 + 8);
      v135 = v2[28];
      if ( (int)v134 < 0 )
        v136 = *(_QWORD *)(*(_QWORD *)(v135 + 56) + 16 * (v134 & 0x7FFFFFFF) + 8);
      else
        v136 = *(_QWORD *)(*(_QWORD *)(v135 + 304) + 8 * v134);
      if ( v136 )
      {
        if ( (*(_BYTE *)(v136 + 3) & 0x10) == 0 )
          goto LABEL_196;
        while ( 1 )
        {
          v136 = *(_QWORD *)(v136 + 32);
          if ( !v136 )
            break;
          if ( (*(_BYTE *)(v136 + 3) & 0x10) == 0 )
            goto LABEL_196;
        }
      }
      v114 += 40;
LABEL_237:
      if ( *(_BYTE *)v114 )
        goto LABEL_196;
      v128 = *(unsigned int *)(v114 + 8);
      v129 = v2[28];
      if ( (int)v128 < 0 )
        v130 = *(_QWORD *)(*(_QWORD *)(v129 + 56) + 16 * (v128 & 0x7FFFFFFF) + 8);
      else
        v130 = *(_QWORD *)(*(_QWORD *)(v129 + 304) + 8 * v128);
      if ( v130 )
      {
        if ( (*(_BYTE *)(v130 + 3) & 0x10) == 0 )
          goto LABEL_196;
        while ( 1 )
        {
          v130 = *(_QWORD *)(v130 + 32);
          if ( !v130 )
            break;
          if ( (*(_BYTE *)(v130 + 3) & 0x10) == 0 )
            goto LABEL_196;
        }
      }
LABEL_197:
      v117 = (unsigned int)v158;
      v118 = (unsigned int)v158 + 1LL;
      if ( v118 > HIDWORD(v158) )
      {
        sub_C8D5F0((__int64)&v157, v159, v118, 8u, v112, v113);
        v117 = (unsigned int)v158;
      }
      v140 = 1;
      v157[v117] = v41;
      LODWORD(v158) = v158 + 1;
LABEL_46:
      if ( (*(_BYTE *)v41 & 4) == 0 )
        break;
      v41 = *(_QWORD *)(v41 + 8);
      if ( v40 == v41 )
        goto LABEL_48;
    }
    while ( (*(_BYTE *)(v41 + 44) & 8) != 0 )
      v41 = *(_QWORD *)(v41 + 8);
    v41 = *(_QWORD *)(v41 + 8);
  }
  while ( v40 != v41 );
LABEL_48:
  v42 = v157;
  LOBYTE(v138) = v140 | v138;
  v43 = &v157[(unsigned int)v158];
  if ( v157 != v43 )
  {
    do
    {
      v44 = *v42++;
      sub_2E88E20(v44);
    }
    while ( v43 != v42 );
    v43 = v157;
  }
  if ( v43 != (__int64 *)v159 )
    _libc_free((unsigned __int64)v43);
  return v138;
}
