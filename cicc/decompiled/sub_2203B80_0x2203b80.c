// Function: sub_2203B80
// Address: 0x2203b80
//
__int64 __fastcall sub_2203B80(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r13
  __int64 v11; // rbx
  int v12; // r8d
  __int64 v13; // r9
  _QWORD *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rdx
  _QWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // rcx
  int v27; // edi
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 *v33; // rdi
  __int64 v34; // r14
  __int64 v35; // rbx
  unsigned int v36; // eax
  __int64 v37; // r8
  __int64 *v38; // r9
  int v39; // eax
  __int64 v40; // rcx
  __int64 *v41; // r12
  __int64 v42; // rax
  __int64 v43; // rbx
  unsigned int v44; // r13d
  __int64 v45; // rcx
  __int64 *v46; // rax
  __int64 *v47; // rdx
  __int64 *v48; // rax
  unsigned int v49; // esi
  unsigned int v50; // r14d
  unsigned int v51; // ecx
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rbx
  __int64 v56; // r14
  __int64 *v57; // rbx
  __int64 *v58; // r13
  __int64 v59; // rdi
  __int64 v61; // r11
  int i; // r10d
  __int64 v63; // r13
  unsigned int v64; // ecx
  __int64 *v65; // rax
  __int64 v66; // rdi
  __int64 *v67; // r15
  __int64 v68; // rcx
  __int64 v69; // rax
  unsigned int v70; // edx
  int v71; // ecx
  __int64 v72; // rsi
  int v73; // r10d
  int v74; // r11d
  int v75; // edx
  int v76; // r10d
  __int64 *v77; // r13
  __int64 *v78; // rax
  __int64 v79; // rdi
  __int64 *v80; // rbx
  int v81; // esi
  __int64 *v82; // rdx
  unsigned int v83; // ebx
  int v84; // esi
  __int64 v85; // rdi
  int v86; // r15d
  __int64 *v87; // r11
  int v88; // ecx
  __int64 v89; // rdx
  __int64 v90; // r8
  int v91; // edi
  __int64 *v92; // rsi
  unsigned int v93; // r13d
  __int64 v94; // rsi
  int v95; // ecx
  __int64 *v96; // rax
  int v97; // ecx
  unsigned int v98; // r13d
  __int64 v99; // rdi
  int v100; // edi
  __int64 v101; // rdx
  __int64 v102; // r8
  __int64 v103; // rcx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // rax
  signed __int64 v107; // rdx
  __int64 v108; // rsi
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rdi
  __int64 v112; // rdx
  __int64 v113; // rdx
  __int64 v114; // rdx
  __int64 v115; // rdx
  __int64 v116; // rdx
  __int64 v117; // rdx
  __int64 v118; // rdx
  __int64 v119; // rdx
  __int64 v120; // rsi
  __int64 v121; // rdx
  __int64 **v122; // r10
  __int64 v123; // rdx
  __int64 v124; // rdx
  __int64 v125; // rsi
  __int64 v126; // rdx
  __int64 v127; // rdx
  __int64 v128; // rsi
  __int64 v129; // rdx
  int v130; // eax
  unsigned int v131; // r12d
  __int64 *v132; // [rsp+0h] [rbp-110h]
  char v133; // [rsp+Fh] [rbp-101h]
  __int64 *v134; // [rsp+10h] [rbp-100h]
  __int64 v135; // [rsp+20h] [rbp-F0h]
  __int64 v136; // [rsp+28h] [rbp-E8h]
  __int64 v137; // [rsp+30h] [rbp-E0h]
  __int64 v138; // [rsp+38h] [rbp-D8h]
  _QWORD *v139; // [rsp+40h] [rbp-D0h]
  __int64 v140; // [rsp+48h] [rbp-C8h]
  __int64 v141; // [rsp+48h] [rbp-C8h]
  __int64 v142; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v143; // [rsp+58h] [rbp-B8h]
  __int64 v144; // [rsp+60h] [rbp-B0h]
  unsigned int v145; // [rsp+68h] [rbp-A8h]
  __int64 v146; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v147; // [rsp+78h] [rbp-98h]
  __int64 v148; // [rsp+80h] [rbp-90h]
  unsigned int v149; // [rsp+88h] [rbp-88h]
  __int64 *v150; // [rsp+90h] [rbp-80h] BYREF
  __int64 v151; // [rsp+98h] [rbp-78h]
  _BYTE v152[112]; // [rsp+A0h] [rbp-70h] BYREF

  v2 = a1;
  v133 = sub_1636880((__int64)a1, *a2);
  if ( v133 )
    return 0;
  a1[29] = a2;
  v3 = a2[2];
  a1[30] = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 40LL))(v3);
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 112LL))(v3);
  v5 = (__int64 *)a1[1];
  a1[31] = v4;
  a1[32] = *(_QWORD *)(a1[29] + 40LL);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_321:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4FC6A0C )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_321;
  }
  a1[33] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(
             *(_QWORD *)(v6 + 8),
             &unk_4FC6A0C);
  v8 = a1[29];
  v9 = *(_QWORD *)(v8 + 328);
  v150 = (__int64 *)v152;
  v151 = 0x600000000LL;
  v10 = *(_QWORD *)(v9 + 32);
  v11 = v9 + 24;
  v137 = v9;
  if ( v10 == v9 + 24 )
  {
    v131 = 0;
    goto LABEL_64;
  }
  do
  {
    while ( 1 )
    {
      if ( !(unsigned int)sub_22030E0(v10) )
      {
        if ( !v10 )
          BUG();
        goto LABEL_9;
      }
      v14 = *(_QWORD **)(v10 + 56);
      v15 = *(_QWORD *)(v10 + 32);
      v16 = 8LL * *(unsigned __int8 *)(v10 + 49);
      v17 = &v14[(unsigned __int64)v16 / 8];
      v18 = v16 >> 3;
      v19 = v16 >> 5;
      if ( v19 )
      {
        v20 = &v14[4 * v19];
        while ( !*(_BYTE *)(*v14 + 76LL) )
        {
          if ( *(_BYTE *)(v14[1] + 76LL) )
          {
            ++v14;
            break;
          }
          if ( *(_BYTE *)(v14[2] + 76LL) )
          {
            v14 += 2;
            break;
          }
          if ( *(_BYTE *)(v14[3] + 76LL) )
          {
            v14 += 3;
            break;
          }
          v14 += 4;
          if ( v20 == v14 )
          {
            v18 = v17 - v14;
            goto LABEL_81;
          }
        }
LABEL_19:
        if ( v17 != v14 )
        {
LABEL_31:
          v32 = (unsigned int)v151;
          if ( (unsigned int)v151 >= HIDWORD(v151) )
          {
            sub_16CD150((__int64)&v150, v152, 0, 8, v12, v13);
            v32 = (unsigned int)v151;
          }
          v150[v32] = v10;
          LODWORD(v151) = v151 + 1;
          goto LABEL_9;
        }
        goto LABEL_20;
      }
LABEL_81:
      if ( v18 != 2 )
      {
        if ( v18 != 3 )
        {
          if ( v18 == 1 && *(_BYTE *)(*v14 + 76LL) )
            goto LABEL_19;
          goto LABEL_20;
        }
        if ( *(_BYTE *)(*v14 + 76LL) )
          goto LABEL_19;
        ++v14;
      }
      if ( *(_BYTE *)(*v14 + 76LL) )
        goto LABEL_19;
      v123 = v14[1];
      ++v14;
      if ( *(_BYTE *)(v123 + 76) )
        goto LABEL_19;
LABEL_20:
      if ( byte_4FD43C0 )
      {
        v21 = *(unsigned int *)(v15 + 8);
        v22 = v2[32];
        v23 = (int)v21 < 0
            ? *(_QWORD *)(*(_QWORD *)(v22 + 24) + 16 * (v21 & 0x7FFFFFFF) + 8)
            : *(_QWORD *)(*(_QWORD *)(v22 + 272) + 8 * v21);
        if ( v23 )
        {
          if ( (*(_BYTE *)(v23 + 3) & 0x10) == 0 )
            goto LABEL_27;
          while ( 1 )
          {
            v23 = *(_QWORD *)(v23 + 32);
            if ( !v23 )
              break;
            if ( (*(_BYTE *)(v23 + 3) & 0x10) == 0 )
            {
LABEL_27:
              while ( 1 )
              {
                v23 = *(_QWORD *)(v23 + 32);
                if ( !v23 )
                  break;
                if ( (*(_BYTE *)(v23 + 3) & 0x10) == 0 )
                  goto LABEL_9;
              }
              v24 = v2[33];
              v25 = *(_DWORD *)(v24 + 256);
              if ( v25 )
              {
                v26 = *(_QWORD *)(v15 + 16);
                v27 = v25 - 1;
                v28 = *(_QWORD *)(v24 + 240);
                v29 = *(_QWORD *)(v26 + 24);
                v30 = (v25 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v31 = (__int64 *)(v28 + 16LL * v30);
                v13 = *v31;
                if ( v29 == *v31 )
                {
LABEL_30:
                  if ( v31[1] )
                    break;
                }
                else
                {
                  v130 = 1;
                  while ( v13 != -8 )
                  {
                    v12 = v130 + 1;
                    v30 = v27 & (v130 + v30);
                    v31 = (__int64 *)(v28 + 16LL * v30);
                    v13 = *v31;
                    if ( v29 == *v31 )
                      goto LABEL_30;
                    v130 = v12;
                  }
                }
              }
              goto LABEL_31;
            }
          }
        }
      }
LABEL_9:
      if ( (*(_BYTE *)v10 & 4) == 0 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v11 == v10 )
        goto LABEL_37;
    }
    while ( (*(_BYTE *)(v10 + 46) & 8) != 0 )
      v10 = *(_QWORD *)(v10 + 8);
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v11 != v10 );
LABEL_37:
  v33 = v150;
  if ( (_DWORD)v151 )
  {
    v134 = v150;
    v132 = &v150[(unsigned int)v151];
    v139 = v2;
    do
    {
      v34 = *v134;
      v35 = *(_QWORD *)(*v134 + 32);
      v136 = *v134;
      v36 = sub_1E163A0(*v134);
      v140 = *(_QWORD *)(v34 + 32);
      v135 = v35 + 40LL * v36;
      while ( v140 != v135 )
      {
        v39 = *(_DWORD *)(v140 + 8);
        v40 = v139[32];
        v142 = 0;
        v143 = 0;
        v144 = 0;
        v145 = 0;
        v146 = 0;
        v147 = 0;
        v148 = 0;
        v149 = 0;
        if ( v39 < 0 )
          v41 = *(__int64 **)(*(_QWORD *)(v40 + 24) + 16LL * (v39 & 0x7FFFFFFF) + 8);
        else
          v41 = *(__int64 **)(*(_QWORD *)(v40 + 272) + 8LL * (unsigned int)v39);
        if ( v41 )
        {
          if ( (*((_BYTE *)v41 + 3) & 0x10) != 0 )
          {
            while ( 1 )
            {
              v41 = (__int64 *)v41[4];
              if ( !v41 )
                break;
              if ( (*((_BYTE *)v41 + 3) & 0x10) == 0 )
                goto LABEL_44;
            }
          }
          else
          {
LABEL_44:
            v138 = 16LL * (v39 & 0x7FFFFFFF);
LABEL_45:
            v42 = v41[2];
            v43 = *(_QWORD *)(v42 + 24);
            if ( !**(_WORD **)(v42 + 16) || **(_WORD **)(v42 + 16) == 45 )
              v43 = *(_QWORD *)(*(_QWORD *)(v42 + 32)
                              + 40LL * (-858993459 * (unsigned int)(((__int64)v41 - *(_QWORD *)(v42 + 32)) >> 3) + 1)
                              + 24);
            if ( v137 != v43 )
            {
              if ( v145 )
              {
                v37 = v145 - 1;
                v44 = ((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4);
                LODWORD(v45) = v37 & v44;
                v46 = (__int64 *)(v143 + 16LL * ((unsigned int)v37 & v44));
                v47 = (__int64 *)*v46;
                if ( *v46 == v43 )
                {
                  v48 = (__int64 *)v46[1];
                  goto LABEL_52;
                }
                v38 = (__int64 *)*v46;
                LODWORD(v61) = v37 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                for ( i = 1; ; ++i )
                {
                  if ( v38 == (__int64 *)-8LL )
                    goto LABEL_94;
                  v61 = (unsigned int)v37 & ((_DWORD)v61 + i);
                  v38 = *(__int64 **)(v143 + 16 * v61);
                  if ( v38 == (__int64 *)v43 )
                    break;
                }
                v74 = 1;
                v38 = 0;
                while ( v47 != (__int64 *)-8LL )
                {
                  if ( v47 != (__int64 *)-16LL || v38 )
                    v46 = v38;
                  LODWORD(v38) = v74 + 1;
                  v45 = (unsigned int)v37 & ((_DWORD)v45 + v74);
                  v122 = (__int64 **)(v143 + 16 * v45);
                  v47 = *v122;
                  if ( *v122 == (__int64 *)v43 )
                  {
                    v48 = v122[1];
                    goto LABEL_52;
                  }
                  ++v74;
                  v38 = v46;
                  v46 = (__int64 *)(v143 + 16 * v45);
                }
                if ( !v38 )
                  v38 = v46;
                ++v142;
                v75 = v144 + 1;
                if ( 4 * ((int)v144 + 1) >= 3 * v145 )
                {
                  sub_2203800((__int64)&v142, 2 * v145);
                  if ( v145 )
                  {
                    v93 = (v145 - 1) & v44;
                    v75 = v144 + 1;
                    v38 = (__int64 *)(v143 + 16LL * v93);
                    v94 = *v38;
                    if ( *v38 == v43 )
                      goto LABEL_113;
                    v95 = 1;
                    v96 = 0;
                    while ( v94 != -8 )
                    {
                      if ( v94 == -16 && !v96 )
                        v96 = v38;
                      v93 = (v145 - 1) & (v95 + v93);
                      v38 = (__int64 *)(v143 + 16LL * v93);
                      v94 = *v38;
                      if ( *v38 == v43 )
                        goto LABEL_113;
                      ++v95;
                    }
LABEL_165:
                    if ( v96 )
                      v38 = v96;
                    goto LABEL_113;
                  }
                }
                else
                {
                  if ( v145 - HIDWORD(v144) - v75 > v145 >> 3 )
                  {
LABEL_113:
                    LODWORD(v144) = v75;
                    if ( *v38 != -8 )
                      --HIDWORD(v144);
                    *v38 = v43;
                    v48 = 0;
                    v38[1] = 0;
LABEL_52:
                    v49 = v149;
                    v50 = *(_DWORD *)(v48[4] + 8);
                    if ( v149 )
                    {
LABEL_53:
                      v37 = v49 - 1;
                      v51 = v37 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                      v52 = &v147[2 * v51];
                      v53 = *v52;
                      if ( v41 == (__int64 *)*v52 )
                      {
LABEL_54:
                        *((_DWORD *)v52 + 2) = v50;
                        goto LABEL_56;
                      }
                      v76 = 1;
                      v38 = 0;
                      while ( v53 != -8 )
                      {
                        if ( v53 == -16 && !v38 )
                          v38 = v52;
                        v51 = v37 & (v76 + v51);
                        v52 = &v147[2 * v51];
                        v53 = *v52;
                        if ( v41 == (__int64 *)*v52 )
                          goto LABEL_54;
                        ++v76;
                      }
                      if ( v38 )
                        v52 = v38;
                      ++v146;
                      v71 = v148 + 1;
                      if ( 4 * ((int)v148 + 1) < 3 * v49 )
                      {
                        if ( v49 - HIDWORD(v148) - v71 > v49 >> 3 )
                        {
LABEL_122:
                          LODWORD(v148) = v71;
                          if ( *v52 != -8 )
                            --HIDWORD(v148);
                          *v52 = (__int64)v41;
                          *((_DWORD *)v52 + 2) = 0;
                          goto LABEL_54;
                        }
                        sub_22039C0((__int64)&v146, v49);
                        if ( v149 )
                        {
                          LODWORD(v38) = v149 - 1;
                          v37 = (__int64)v147;
                          v82 = 0;
                          v83 = (v149 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                          v71 = v148 + 1;
                          v84 = 1;
                          v52 = &v147[2 * v83];
                          v85 = *v52;
                          if ( v41 != (__int64 *)*v52 )
                          {
                            while ( v85 != -8 )
                            {
                              if ( !v82 && v85 == -16 )
                                v82 = v52;
                              v83 = (unsigned int)v38 & (v84 + v83);
                              v52 = &v147[2 * v83];
                              v85 = *v52;
                              if ( v41 == (__int64 *)*v52 )
                                goto LABEL_122;
                              ++v84;
                            }
                            if ( v82 )
                              v52 = v82;
                          }
                          goto LABEL_122;
                        }
LABEL_318:
                        LODWORD(v148) = v148 + 1;
                        BUG();
                      }
LABEL_98:
                      sub_22039C0((__int64)&v146, 2 * v49);
                      if ( v149 )
                      {
                        v37 = v149 - 1;
                        v70 = v37 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                        v71 = v148 + 1;
                        v52 = &v147[2 * v70];
                        v72 = *v52;
                        if ( (__int64 *)*v52 != v41 )
                        {
                          v73 = 1;
                          v38 = 0;
                          while ( v72 != -8 )
                          {
                            if ( !v38 && v72 == -16 )
                              v38 = v52;
                            v70 = v37 & (v73 + v70);
                            v52 = &v147[2 * v70];
                            v72 = *v52;
                            if ( v41 == (__int64 *)*v52 )
                              goto LABEL_122;
                            ++v73;
                          }
                          if ( v38 )
                            v52 = v38;
                        }
                        goto LABEL_122;
                      }
                      goto LABEL_318;
                    }
LABEL_97:
                    ++v146;
                    goto LABEL_98;
                  }
                  sub_2203800((__int64)&v142, v145);
                  if ( v145 )
                  {
                    v97 = 1;
                    v98 = (v145 - 1) & v44;
                    v75 = v144 + 1;
                    v96 = 0;
                    v38 = (__int64 *)(v143 + 16LL * v98);
                    v99 = *v38;
                    if ( *v38 == v43 )
                      goto LABEL_113;
                    while ( v99 != -8 )
                    {
                      if ( v99 == -16 && !v96 )
                        v96 = v38;
                      v98 = (v145 - 1) & (v97 + v98);
                      v38 = (__int64 *)(v143 + 16LL * v98);
                      v99 = *v38;
                      if ( *v38 == v43 )
                        goto LABEL_113;
                      ++v97;
                    }
                    goto LABEL_165;
                  }
                }
                LODWORD(v144) = v144 + 1;
                BUG();
              }
LABEL_94:
              v50 = sub_1E6B9A0(
                      v139[32],
                      *(_QWORD *)(*(_QWORD *)(v139[32] + 24LL) + v138) & 0xFFFFFFFFFFFFFFF8LL,
                      (unsigned __int8 *)byte_3F871B3,
                      0,
                      v37,
                      (int)v38);
              v63 = (__int64)sub_1E0B7C0(*(_QWORD *)(v43 + 56), v136);
              sub_1E17170(v63, *(_DWORD *)(*(_QWORD *)(v63 + 32) + 8LL), v50, 0, v139[31]);
              if ( v145 )
              {
                v64 = (v145 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v65 = (__int64 *)(v143 + 16LL * v64);
                v66 = *v65;
                if ( *v65 == v43 )
                  goto LABEL_96;
                v86 = 1;
                v87 = 0;
                while ( v66 != -8 )
                {
                  if ( v87 || v66 != -16 )
                    v65 = v87;
                  v64 = (v145 - 1) & (v86 + v64);
                  v66 = *(_QWORD *)(v143 + 16LL * v64);
                  if ( v66 == v43 )
                  {
                    v65 = (__int64 *)(v143 + 16LL * v64);
                    goto LABEL_96;
                  }
                  ++v86;
                  v87 = v65;
                  v65 = (__int64 *)(v143 + 16LL * v64);
                }
                if ( v87 )
                  v65 = v87;
                ++v142;
                v88 = v144 + 1;
                if ( 4 * ((int)v144 + 1) < 3 * v145 )
                {
                  if ( v145 - HIDWORD(v144) - v88 > v145 >> 3 )
                  {
LABEL_150:
                    LODWORD(v144) = v88;
                    if ( *v65 != -8 )
                      --HIDWORD(v144);
                    *v65 = v43;
                    v65[1] = 0;
LABEL_96:
                    v65[1] = v63;
                    v67 = (__int64 *)sub_1DD5D10(v43);
                    sub_1DD5BA0((__int64 *)(v43 + 16), v63);
                    v68 = *v67;
                    v69 = *(_QWORD *)v63;
                    *(_QWORD *)(v63 + 8) = v67;
                    v68 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)v63 = v68 | v69 & 7;
                    *(_QWORD *)(v68 + 8) = v63;
                    *v67 = *v67 & 7 | v63;
                    v49 = v149;
                    if ( v149 )
                      goto LABEL_53;
                    goto LABEL_97;
                  }
                  sub_2203800((__int64)&v142, v145);
                  if ( v145 )
                  {
                    v100 = 1;
                    LODWORD(v101) = (v145 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                    v88 = v144 + 1;
                    v92 = 0;
                    v65 = (__int64 *)(v143 + 16LL * (unsigned int)v101);
                    v102 = *v65;
                    if ( *v65 == v43 )
                      goto LABEL_150;
                    while ( v102 != -8 )
                    {
                      if ( v102 == -16 && !v92 )
                        v92 = v65;
                      v101 = (v145 - 1) & ((_DWORD)v101 + v100);
                      v65 = (__int64 *)(v143 + 16 * v101);
                      v102 = *v65;
                      if ( *v65 == v43 )
                        goto LABEL_150;
                      ++v100;
                    }
                    goto LABEL_158;
                  }
                  goto LABEL_319;
                }
              }
              else
              {
                ++v142;
              }
              sub_2203800((__int64)&v142, 2 * v145);
              if ( v145 )
              {
                LODWORD(v89) = (v145 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v88 = v144 + 1;
                v65 = (__int64 *)(v143 + 16LL * (unsigned int)v89);
                v90 = *v65;
                if ( *v65 == v43 )
                  goto LABEL_150;
                v91 = 1;
                v92 = 0;
                while ( v90 != -8 )
                {
                  if ( v90 == -16 && !v92 )
                    v92 = v65;
                  v89 = (v145 - 1) & ((_DWORD)v89 + v91);
                  v65 = (__int64 *)(v143 + 16 * v89);
                  v90 = *v65;
                  if ( *v65 == v43 )
                    goto LABEL_150;
                  ++v91;
                }
LABEL_158:
                if ( v92 )
                  v65 = v92;
                goto LABEL_150;
              }
LABEL_319:
              LODWORD(v144) = v144 + 1;
              BUG();
            }
LABEL_56:
            while ( 1 )
            {
              v41 = (__int64 *)v41[4];
              if ( !v41 )
                break;
              if ( (*((_BYTE *)v41 + 3) & 0x10) == 0 )
                goto LABEL_45;
            }
            v41 = v147;
            if ( (_DWORD)v148 )
            {
              v77 = &v147[2 * v149];
              if ( v77 != v147 )
              {
                v78 = v147;
                while ( 1 )
                {
                  v79 = *v78;
                  v80 = v78;
                  if ( *v78 != -8 && v79 != -16 )
                    break;
                  v78 += 2;
                  if ( v77 == v78 )
                    goto LABEL_58;
                }
                if ( v77 != v78 )
                {
                  do
                  {
                    v81 = *((_DWORD *)v80 + 2);
                    v80 += 2;
                    sub_1E310D0(v79, v81);
                    if ( v80 == v77 )
                      break;
                    while ( 1 )
                    {
                      v79 = *v80;
                      if ( *v80 != -16 && v79 != -8 )
                        break;
                      v80 += 2;
                      if ( v77 == v80 )
                        goto LABEL_135;
                    }
                  }
                  while ( v77 != v80 );
LABEL_135:
                  v41 = v147;
                }
              }
            }
          }
        }
LABEL_58:
        j___libc_free_0(v41);
        j___libc_free_0(v143);
        v140 += 40;
      }
      ++v134;
    }
    while ( v132 != v134 );
    v2 = v139;
    v33 = v150;
    v131 = 1;
  }
  else
  {
    v131 = 0;
  }
  if ( v33 != (__int64 *)v152 )
    _libc_free((unsigned __int64)v33);
  v8 = v2[29];
LABEL_64:
  v54 = *(_QWORD *)(v8 + 328);
  v150 = (__int64 *)v152;
  v55 = v54 + 24;
  v151 = 0x800000000LL;
  v56 = *(_QWORD *)(v55 + 8);
  if ( v56 == v55 )
    return v131;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( !(unsigned int)sub_22030E0(v56) )
      {
        if ( !v56 )
          BUG();
        goto LABEL_69;
      }
      v141 = *(_QWORD *)(v56 + 32);
      v103 = v141 + 40LL * (unsigned int)sub_1E163A0(v56);
      v106 = *(_QWORD *)(v56 + 32);
      v107 = 0xCCCCCCCCCCCCCCCDLL * ((v103 - v106) >> 3);
      if ( v107 >> 2 > 0 )
      {
        v108 = v106 + 160 * (v107 >> 2);
        while ( !*(_BYTE *)v106 )
        {
          v110 = *(unsigned int *)(v106 + 8);
          v111 = v2[32];
          if ( (int)v110 < 0 )
          {
            v112 = *(_QWORD *)(*(_QWORD *)(v111 + 24) + 16 * (v110 & 0x7FFFFFFF) + 8);
          }
          else
          {
            v104 = *(_QWORD *)(v111 + 272);
            v112 = *(_QWORD *)(v104 + 8 * v110);
          }
          if ( v112 )
          {
            if ( (*(_BYTE *)(v112 + 3) & 0x10) == 0 )
              goto LABEL_187;
            while ( 1 )
            {
              v112 = *(_QWORD *)(v112 + 32);
              if ( !v112 )
                break;
              if ( (*(_BYTE *)(v112 + 3) & 0x10) == 0 )
                goto LABEL_187;
            }
          }
          v104 = v106 + 40;
          if ( *(_BYTE *)(v106 + 40) )
            goto LABEL_199;
          v113 = *(unsigned int *)(v106 + 48);
          if ( (int)v113 < 0 )
          {
            v114 = *(_QWORD *)(*(_QWORD *)(v111 + 24) + 16 * (v113 & 0x7FFFFFFF) + 8);
          }
          else
          {
            v105 = *(_QWORD *)(v111 + 272);
            v114 = *(_QWORD *)(v105 + 8 * v113);
          }
          if ( v114 )
          {
            if ( (*(_BYTE *)(v114 + 3) & 0x10) == 0 )
              goto LABEL_199;
            while ( 1 )
            {
              v114 = *(_QWORD *)(v114 + 32);
              if ( !v114 )
                break;
              if ( (*(_BYTE *)(v114 + 3) & 0x10) == 0 )
                goto LABEL_199;
            }
          }
          v104 = v106 + 80;
          if ( *(_BYTE *)(v106 + 80) )
            goto LABEL_199;
          v115 = *(unsigned int *)(v106 + 88);
          if ( (int)v115 < 0 )
          {
            v116 = *(_QWORD *)(*(_QWORD *)(v111 + 24) + 16 * (v115 & 0x7FFFFFFF) + 8);
          }
          else
          {
            v105 = *(_QWORD *)(v111 + 272);
            v116 = *(_QWORD *)(v105 + 8 * v115);
          }
          if ( v116 )
          {
            if ( (*(_BYTE *)(v116 + 3) & 0x10) == 0 )
              goto LABEL_199;
            while ( 1 )
            {
              v116 = *(_QWORD *)(v116 + 32);
              if ( !v116 )
                break;
              if ( (*(_BYTE *)(v116 + 3) & 0x10) == 0 )
                goto LABEL_199;
            }
          }
          v104 = v106 + 120;
          if ( *(_BYTE *)(v106 + 120) )
          {
LABEL_199:
            v106 = v104;
            goto LABEL_187;
          }
          v117 = *(unsigned int *)(v106 + 128);
          if ( (int)v117 < 0 )
            v118 = *(_QWORD *)(*(_QWORD *)(v111 + 24) + 16 * (v117 & 0x7FFFFFFF) + 8);
          else
            v118 = *(_QWORD *)(*(_QWORD *)(v111 + 272) + 8 * v117);
          if ( v118 )
          {
            if ( (*(_BYTE *)(v118 + 3) & 0x10) == 0 )
              goto LABEL_199;
            while ( 1 )
            {
              v118 = *(_QWORD *)(v118 + 32);
              if ( !v118 )
                break;
              if ( (*(_BYTE *)(v118 + 3) & 0x10) == 0 )
                goto LABEL_199;
            }
          }
          v106 += 160;
          if ( v106 == v108 )
          {
            v107 = 0xCCCCCCCCCCCCCCCDLL * ((v103 - v106) >> 3);
            goto LABEL_225;
          }
        }
        goto LABEL_187;
      }
LABEL_225:
      switch ( v107 )
      {
        case 2LL:
          goto LABEL_274;
        case 3LL:
          if ( *(_BYTE *)v106 )
            goto LABEL_187;
          v124 = *(unsigned int *)(v106 + 8);
          v125 = v2[32];
          if ( (int)v124 < 0 )
            v126 = *(_QWORD *)(*(_QWORD *)(v125 + 24) + 16 * (v124 & 0x7FFFFFFF) + 8);
          else
            v126 = *(_QWORD *)(*(_QWORD *)(v125 + 272) + 8 * v124);
          if ( v126 )
          {
            if ( (*(_BYTE *)(v126 + 3) & 0x10) == 0 )
              goto LABEL_187;
            while ( 1 )
            {
              v126 = *(_QWORD *)(v126 + 32);
              if ( !v126 )
                break;
              if ( (*(_BYTE *)(v126 + 3) & 0x10) == 0 )
                goto LABEL_187;
            }
          }
          v106 += 40;
LABEL_274:
          if ( *(_BYTE *)v106 )
            goto LABEL_187;
          v127 = *(unsigned int *)(v106 + 8);
          v128 = v2[32];
          if ( (int)v127 < 0 )
            v129 = *(_QWORD *)(*(_QWORD *)(v128 + 24) + 16 * (v127 & 0x7FFFFFFF) + 8);
          else
            v129 = *(_QWORD *)(*(_QWORD *)(v128 + 272) + 8 * v127);
          if ( v129 )
          {
            if ( (*(_BYTE *)(v129 + 3) & 0x10) == 0 )
              goto LABEL_187;
            while ( 1 )
            {
              v129 = *(_QWORD *)(v129 + 32);
              if ( !v129 )
                break;
              if ( (*(_BYTE *)(v129 + 3) & 0x10) == 0 )
                goto LABEL_187;
            }
          }
          v106 += 40;
LABEL_228:
          if ( !*(_BYTE *)v106 )
          {
            v119 = *(unsigned int *)(v106 + 8);
            v120 = v2[32];
            if ( (int)v119 < 0 )
              v121 = *(_QWORD *)(*(_QWORD *)(v120 + 24) + 16 * (v119 & 0x7FFFFFFF) + 8);
            else
              v121 = *(_QWORD *)(*(_QWORD *)(v120 + 272) + 8 * v119);
            if ( !v121 )
              break;
            while ( (*(_BYTE *)(v121 + 3) & 0x10) != 0 )
            {
              v121 = *(_QWORD *)(v121 + 32);
              if ( !v121 )
                goto LABEL_188;
            }
          }
LABEL_187:
          if ( v103 != v106 )
            goto LABEL_69;
          break;
        case 1LL:
          goto LABEL_228;
      }
LABEL_188:
      v109 = (unsigned int)v151;
      if ( (unsigned int)v151 >= HIDWORD(v151) )
      {
        sub_16CD150((__int64)&v150, v152, 0, 8, v104, v105);
        v109 = (unsigned int)v151;
      }
      v133 = 1;
      v150[v109] = v56;
      LODWORD(v151) = v151 + 1;
LABEL_69:
      if ( (*(_BYTE *)v56 & 4) != 0 )
      {
        v56 = *(_QWORD *)(v56 + 8);
        if ( v55 == v56 )
          goto LABEL_71;
        continue;
      }
      break;
    }
    while ( (*(_BYTE *)(v56 + 46) & 8) != 0 )
      v56 = *(_QWORD *)(v56 + 8);
    v56 = *(_QWORD *)(v56 + 8);
    if ( v55 != v56 )
      continue;
    break;
  }
LABEL_71:
  v57 = v150;
  LOBYTE(v131) = v133 | v131;
  v58 = &v150[(unsigned int)v151];
  if ( v150 != v58 )
  {
    do
    {
      v59 = *v57++;
      sub_1E16240(v59);
    }
    while ( v58 != v57 );
    v58 = v150;
  }
  if ( v58 != (__int64 *)v152 )
    _libc_free((unsigned __int64)v58);
  return v131;
}
