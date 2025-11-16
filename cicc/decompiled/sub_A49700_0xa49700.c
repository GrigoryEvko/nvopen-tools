// Function: sub_A49700
// Address: 0xa49700
//
__int64 __fastcall sub_A49700(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int8 *v4; // rbx
  unsigned __int8 *v5; // r13
  char *v6; // rax
  unsigned __int8 *v7; // rdi
  char *v8; // rax
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int8 **v13; // rbx
  unsigned __int8 **v14; // r14
  unsigned __int8 v15; // al
  _BYTE *v16; // rsi
  _BYTE *v17; // rsi
  __int64 v18; // r15
  unsigned int v19; // esi
  __int64 v20; // r10
  unsigned int v21; // r8d
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // r14
  _BYTE *v25; // rsi
  __int64 v26; // rax
  __int64 result; // rax
  __int64 **v28; // r15
  __int64 i; // r12
  __int64 v30; // r10
  unsigned __int8 *v31; // rbx
  unsigned __int8 *v32; // r13
  __int64 **v33; // r8
  __int64 v34; // r9
  __int64 *v35; // r15
  unsigned __int8 *v36; // r14
  unsigned __int8 *v37; // r13
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // r10
  _QWORD *v45; // rbx
  _BYTE *v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  char *v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 v54; // rbx
  __int64 v55; // r8
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 *v58; // rbx
  __int64 *v59; // r13
  __int64 v60; // rdx
  __int64 *v61; // r13
  __int64 *v62; // rbx
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  _QWORD *v66; // rax
  _QWORD *v67; // r9
  __int64 **v68; // rdi
  __int64 v69; // r15
  __int64 v70; // r13
  _BYTE *v71; // rbx
  __int64 v72; // rdx
  __int64 v73; // r8
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  _QWORD *v78; // rax
  _QWORD *v79; // r9
  __int64 v80; // r8
  _QWORD *v81; // rbx
  __int64 v82; // rax
  __int64 **v83; // rdi
  _QWORD *v84; // r13
  __int64 v85; // r15
  __int64 v86; // r9
  _BYTE *v87; // r12
  __int64 v88; // rax
  unsigned __int64 v89; // rdx
  int v90; // ecx
  __int64 *v91; // rdx
  int v92; // edi
  int v93; // edx
  int v94; // edi
  int v95; // edi
  __int64 v96; // r8
  __int64 *v97; // r9
  unsigned int v98; // r14d
  int v99; // ecx
  __int64 v100; // rsi
  int v101; // r8d
  int v102; // r8d
  __int64 v103; // r10
  unsigned int v104; // ecx
  __int64 v105; // r9
  int v106; // edi
  __int64 *v107; // rsi
  __int64 v108; // [rsp+0h] [rbp-120h]
  __int64 v109; // [rsp+8h] [rbp-118h]
  __int64 v111; // [rsp+20h] [rbp-100h]
  _QWORD *v112; // [rsp+30h] [rbp-F0h]
  _QWORD *v113; // [rsp+30h] [rbp-F0h]
  __int64 v114; // [rsp+30h] [rbp-F0h]
  __int64 v115; // [rsp+38h] [rbp-E8h]
  _QWORD *v116; // [rsp+38h] [rbp-E8h]
  __int64 v117; // [rsp+38h] [rbp-E8h]
  __int64 v118; // [rsp+38h] [rbp-E8h]
  __int64 **v119; // [rsp+38h] [rbp-E8h]
  __int64 v120; // [rsp+40h] [rbp-E0h]
  __int64 **v121; // [rsp+40h] [rbp-E0h]
  __int64 v122; // [rsp+40h] [rbp-E0h]
  __int64 **v123; // [rsp+40h] [rbp-E0h]
  __int64 v124; // [rsp+40h] [rbp-E0h]
  __int64 v125; // [rsp+40h] [rbp-E0h]
  __int64 v126; // [rsp+40h] [rbp-E0h]
  __int64 v127; // [rsp+40h] [rbp-E0h]
  __int64 v128; // [rsp+48h] [rbp-D8h]
  unsigned __int8 *v129; // [rsp+48h] [rbp-D8h]
  __int64 *v130; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v131; // [rsp+58h] [rbp-C8h]
  _BYTE v132[64]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 *v133; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v134; // [rsp+A8h] [rbp-78h]
  _BYTE v135[112]; // [rsp+B0h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112);
  *(_DWORD *)(a1 + 504) = 0;
  *(_DWORD *)(a1 + 536) = v3 >> 4;
  sub_A3F730(a1, a2);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2);
    v4 = *(unsigned __int8 **)(a2 + 96);
    v5 = &v4[40 * *(_QWORD *)(a2 + 104)];
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2);
      v4 = *(unsigned __int8 **)(a2 + 96);
    }
  }
  else
  {
    v4 = *(unsigned __int8 **)(a2 + 96);
    v5 = &v4[40 * *(_QWORD *)(a2 + 104)];
  }
  while ( v5 != v4 )
  {
    while ( 1 )
    {
      sub_A45280(a1, v4);
      if ( !(unsigned __int8)sub_B2D670(v4, 81) )
        break;
      v6 = (char *)sub_B2BD20(v4);
      sub_A44BF0(a1, v6);
LABEL_6:
      v4 += 40;
      if ( v5 == v4 )
        goto LABEL_10;
    }
    if ( !(unsigned __int8)sub_B2D670(v4, 85) )
    {
      if ( (unsigned __int8)sub_B2D670(v4, 80) )
      {
        v49 = (char *)sub_B2BD40(v4);
        sub_A44BF0(a1, v49);
      }
      goto LABEL_6;
    }
    v7 = v4;
    v4 += 40;
    v8 = (char *)sub_B2BD30(v7);
    sub_A44BF0(a1, v8);
  }
LABEL_10:
  v9 = (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 4;
  *(_DWORD *)(a1 + 548) = v9;
  v10 = v9;
  v109 = a2 + 72;
  v120 = *(_QWORD *)(a2 + 80);
  if ( v120 == a2 + 72 )
    goto LABEL_35;
  do
  {
    if ( !v120 )
      BUG();
    v11 = *(_QWORD *)(v120 + 32);
    v128 = v120 - 24;
    if ( v11 != v120 + 24 )
    {
      while ( 1 )
      {
        if ( !v11 )
          BUG();
        v12 = 4LL * (*(_DWORD *)(v11 - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v11 - 17) & 0x40) != 0 )
        {
          v13 = *(unsigned __int8 ***)(v11 - 32);
          v14 = &v13[v12];
        }
        else
        {
          v14 = (unsigned __int8 **)(v11 - 24);
          v13 = (unsigned __int8 **)(v11 - 24 - v12 * 8);
        }
        if ( v13 != v14 )
          break;
LABEL_24:
        if ( *(_BYTE *)(v11 - 24) == 92 )
          sub_A45280(a1, *(unsigned __int8 **)(v11 + 80));
        v11 = *(_QWORD *)(v11 + 8);
        if ( v120 + 24 == v11 )
          goto LABEL_27;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v15 = **v13;
          if ( v15 <= 0x15u )
            break;
          if ( v15 == 25 )
            goto LABEL_19;
LABEL_20:
          v13 += 4;
          if ( v14 == v13 )
            goto LABEL_24;
        }
        if ( v15 > 3u )
        {
LABEL_19:
          sub_A45280(a1, *v13);
          goto LABEL_20;
        }
        v13 += 4;
        if ( v14 == v13 )
          goto LABEL_24;
      }
    }
LABEL_27:
    v16 = *(_BYTE **)(a1 + 520);
    v133 = (__int64 *)(v120 - 24);
    if ( v16 == *(_BYTE **)(a1 + 528) )
    {
      sub_A413F0(a1 + 512, v16, &v133);
      v17 = *(_BYTE **)(a1 + 520);
    }
    else
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = v128;
        v16 = *(_BYTE **)(a1 + 520);
      }
      v17 = v16 + 8;
      *(_QWORD *)(a1 + 520) = v17;
    }
    v18 = (__int64)&v17[-*(_QWORD *)(a1 + 512)] >> 3;
    v19 = *(_DWORD *)(a1 + 104);
    if ( !v19 )
    {
      ++*(_QWORD *)(a1 + 80);
LABEL_140:
      sub_A429D0(a1 + 80, 2 * v19);
      v101 = *(_DWORD *)(a1 + 104);
      if ( !v101 )
      {
LABEL_164:
        ++*(_DWORD *)(a1 + 96);
        BUG();
      }
      v102 = v101 - 1;
      v103 = *(_QWORD *)(a1 + 88);
      v93 = *(_DWORD *)(a1 + 96) + 1;
      v104 = v102 & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
      v22 = (__int64 *)(v103 + 16LL * v104);
      v105 = *v22;
      if ( v128 != *v22 )
      {
        v106 = 1;
        v107 = 0;
        while ( v105 != -4096 )
        {
          if ( !v107 && v105 == -8192 )
            v107 = v22;
          v104 = v102 & (v106 + v104);
          v22 = (__int64 *)(v103 + 16LL * v104);
          v105 = *v22;
          if ( v128 == *v22 )
            goto LABEL_130;
          ++v106;
        }
        if ( v107 )
          v22 = v107;
      }
      goto LABEL_130;
    }
    v20 = *(_QWORD *)(a1 + 88);
    v21 = (v19 - 1) & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
    v22 = (__int64 *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( v128 == *v22 )
      goto LABEL_33;
    v90 = 1;
    v91 = 0;
    while ( v23 != -4096 )
    {
      if ( v23 == -8192 && !v91 )
        v91 = v22;
      v21 = (v19 - 1) & (v90 + v21);
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v128 == *v22 )
        goto LABEL_33;
      ++v90;
    }
    v92 = *(_DWORD *)(a1 + 96);
    if ( v91 )
      v22 = v91;
    ++*(_QWORD *)(a1 + 80);
    v93 = v92 + 1;
    if ( 4 * (v92 + 1) >= 3 * v19 )
      goto LABEL_140;
    if ( v19 - *(_DWORD *)(a1 + 100) - v93 <= v19 >> 3 )
    {
      sub_A429D0(a1 + 80, v19);
      v94 = *(_DWORD *)(a1 + 104);
      if ( !v94 )
        goto LABEL_164;
      v95 = v94 - 1;
      v96 = *(_QWORD *)(a1 + 88);
      v97 = 0;
      v98 = v95 & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
      v93 = *(_DWORD *)(a1 + 96) + 1;
      v99 = 1;
      v22 = (__int64 *)(v96 + 16LL * v98);
      v100 = *v22;
      if ( v128 != *v22 )
      {
        while ( v100 != -4096 )
        {
          if ( !v97 && v100 == -8192 )
            v97 = v22;
          v98 = v95 & (v99 + v98);
          v22 = (__int64 *)(v96 + 16LL * v98);
          v100 = *v22;
          if ( v128 == *v22 )
            goto LABEL_130;
          ++v99;
        }
        if ( v97 )
          v22 = v97;
      }
    }
LABEL_130:
    *(_DWORD *)(a1 + 96) = v93;
    if ( *v22 != -4096 )
      --*(_DWORD *)(a1 + 100);
    *((_DWORD *)v22 + 2) = 0;
    *v22 = v128;
LABEL_33:
    *((_DWORD *)v22 + 2) = v18;
    v120 = *(_QWORD *)(v120 + 8);
  }
  while ( v109 != v120 );
  v10 = *(_DWORD *)(a1 + 548);
  v9 = (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 4;
LABEL_35:
  v24 = (__int64 *)v132;
  sub_A42BB0(a1, v10, v9);
  v25 = *(_BYTE **)(a2 + 120);
  sub_A47220(a1, (__int64)v25);
  v26 = (__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 4;
  v130 = (__int64 *)v132;
  *(_DWORD *)(a1 + 552) = v26;
  v131 = 0x800000000LL;
  v134 = 0x800000000LL;
  result = *(_QWORD *)(a2 + 80);
  v133 = (__int64 *)v135;
  v108 = result;
  if ( v109 != result )
  {
    v111 = a1;
    v28 = &v130;
    do
    {
      if ( !v108 )
        BUG();
      for ( i = *(_QWORD *)(v108 + 32); v108 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        v129 = (unsigned __int8 *)(i - 24);
        v30 = 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(i - 17) & 0x40) != 0 )
        {
          v31 = *(unsigned __int8 **)(i - 32);
          v32 = &v31[v30];
        }
        else
        {
          v32 = (unsigned __int8 *)(i - 24);
          v31 = &v129[-v30];
        }
        if ( v31 != v32 )
        {
          v33 = v28;
          v34 = i;
          v35 = v24;
          v36 = v32;
          v37 = v31;
          do
          {
            if ( **(_BYTE **)v37 == 24 )
            {
              v40 = *(_QWORD *)(*(_QWORD *)v37 + 24LL);
              if ( v40 )
              {
                if ( *(_BYTE *)v40 == 2 )
                {
                  v38 = (unsigned int)v131;
                  v39 = (unsigned int)v131 + 1LL;
                  if ( v39 > HIDWORD(v131) )
                  {
                    v25 = v35;
                    v117 = v34;
                    v123 = v33;
                    sub_C8D5F0(v33, v35, v39, 8);
                    v38 = (unsigned int)v131;
                    v34 = v117;
                    v33 = v123;
                  }
                  v130[v38] = v40;
                  LODWORD(v131) = v131 + 1;
                }
                else if ( *(_BYTE *)v40 == 4 )
                {
                  v41 = (unsigned int)v134;
                  v42 = (unsigned int)v134 + 1LL;
                  if ( v42 > HIDWORD(v134) )
                  {
                    v25 = v135;
                    v119 = v33;
                    v125 = v34;
                    sub_C8D5F0(&v133, v135, v42, 8);
                    v41 = (unsigned int)v134;
                    v33 = v119;
                    v34 = v125;
                  }
                  v133[v41] = v40;
                  LODWORD(v134) = v134 + 1;
                  v43 = *(_QWORD **)(v40 + 136);
                  v44 = &v43[*(unsigned int *)(v40 + 144)];
                  if ( v43 != v44 )
                  {
                    v45 = *(_QWORD **)(v40 + 136);
                    do
                    {
                      v46 = (_BYTE *)*v45;
                      if ( *(_BYTE *)*v45 == 2 )
                      {
                        v47 = (unsigned int)v131;
                        v48 = (unsigned int)v131 + 1LL;
                        if ( v48 > HIDWORD(v131) )
                        {
                          v25 = v35;
                          v112 = v44;
                          v115 = v34;
                          v121 = v33;
                          sub_C8D5F0(v33, v35, v48, 8);
                          v47 = (unsigned int)v131;
                          v44 = v112;
                          v34 = v115;
                          v33 = v121;
                        }
                        v130[v47] = (__int64)v46;
                        LODWORD(v131) = v131 + 1;
                      }
                      ++v45;
                    }
                    while ( v44 != v45 );
                  }
                }
              }
            }
            v37 += 32;
          }
          while ( v36 != v37 );
          v24 = v35;
          i = v34;
          v28 = v33;
        }
        v50 = *(_QWORD *)(i + 40);
        if ( v50 )
        {
          v51 = sub_B14240(v50);
          v53 = v52;
          v54 = v51;
          if ( v52 != v51 )
          {
            while ( *(_BYTE *)(v54 + 32) )
            {
              v54 = *(_QWORD *)(v54 + 8);
              if ( v52 == v54 )
                goto LABEL_79;
            }
LABEL_69:
            if ( v53 != v54 )
            {
              v55 = *(_QWORD *)(v54 + 40);
              if ( !v55 )
                goto LABEL_75;
              if ( *(_BYTE *)v55 == 2 )
              {
                v56 = (unsigned int)v131;
                v57 = (unsigned int)v131 + 1LL;
                if ( v57 > HIDWORD(v131) )
                {
                  v25 = v24;
                  v122 = *(_QWORD *)(v54 + 40);
                  sub_C8D5F0(v28, v24, v57, 8);
                  v56 = (unsigned int)v131;
                  v55 = v122;
                }
                v130[v56] = v55;
                LODWORD(v131) = v131 + 1;
LABEL_75:
                if ( *(_BYTE *)(v54 + 64) != 2 )
                  goto LABEL_78;
LABEL_105:
                v73 = *(_QWORD *)(v54 + 48);
                if ( v73 )
                {
                  if ( *(_BYTE *)v73 == 2 )
                  {
                    v74 = (unsigned int)v131;
                    v75 = (unsigned int)v131 + 1LL;
                    if ( v75 > HIDWORD(v131) )
                    {
                      v25 = v24;
                      v126 = *(_QWORD *)(v54 + 48);
                      sub_C8D5F0(v28, v24, v75, 8);
                      v74 = (unsigned int)v131;
                      v73 = v126;
                    }
                    v130[v74] = v73;
                    LODWORD(v131) = v131 + 1;
                  }
                  else if ( *(_BYTE *)v73 == 4 )
                  {
                    v76 = (unsigned int)v134;
                    v77 = (unsigned int)v134 + 1LL;
                    if ( v77 > HIDWORD(v134) )
                    {
                      v25 = v135;
                      v127 = *(_QWORD *)(v54 + 48);
                      sub_C8D5F0(&v133, v135, v77, 8);
                      v76 = (unsigned int)v134;
                      v73 = v127;
                    }
                    v133[v76] = v73;
                    LODWORD(v134) = v134 + 1;
                    v78 = *(_QWORD **)(v73 + 136);
                    v79 = &v78[*(unsigned int *)(v73 + 144)];
                    if ( v78 != v79 )
                    {
                      v80 = v54;
                      v81 = v78;
                      v82 = v53;
                      v83 = v28;
                      v84 = v79;
                      v85 = i;
                      v86 = v82;
                      do
                      {
                        v87 = (_BYTE *)*v81;
                        if ( *(_BYTE *)*v81 == 2 )
                        {
                          v88 = (unsigned int)v131;
                          v89 = (unsigned int)v131 + 1LL;
                          if ( v89 > HIDWORD(v131) )
                          {
                            v25 = v24;
                            v114 = v86;
                            v118 = v80;
                            sub_C8D5F0(v83, v24, v89, 8);
                            v88 = (unsigned int)v131;
                            v86 = v114;
                            v80 = v118;
                          }
                          v130[v88] = (__int64)v87;
                          LODWORD(v131) = v131 + 1;
                        }
                        ++v81;
                      }
                      while ( v84 != v81 );
                      i = v85;
                      v54 = v80;
                      v53 = v86;
                      v28 = v83;
                    }
                  }
                }
                goto LABEL_78;
              }
              if ( *(_BYTE *)v55 != 4 )
                goto LABEL_75;
              v64 = (unsigned int)v134;
              v65 = (unsigned int)v134 + 1LL;
              if ( v65 > HIDWORD(v134) )
              {
                v25 = v135;
                v124 = *(_QWORD *)(v54 + 40);
                sub_C8D5F0(&v133, v135, v65, 8);
                v64 = (unsigned int)v134;
                v55 = v124;
              }
              v133[v64] = v55;
              LODWORD(v134) = v134 + 1;
              v66 = *(_QWORD **)(v55 + 136);
              v67 = &v66[*(unsigned int *)(v55 + 144)];
              if ( v66 == v67 )
                goto LABEL_75;
              v68 = v28;
              v69 = v53;
              v70 = v54;
              do
              {
                v71 = (_BYTE *)*v66;
                if ( *(_BYTE *)*v66 == 2 )
                {
                  v72 = (unsigned int)v131;
                  if ( (unsigned __int64)(unsigned int)v131 + 1 > HIDWORD(v131) )
                  {
                    v25 = v24;
                    v113 = v66;
                    v116 = v67;
                    sub_C8D5F0(v68, v24, (unsigned int)v131 + 1LL, 8);
                    v72 = (unsigned int)v131;
                    v66 = v113;
                    v67 = v116;
                  }
                  v130[v72] = (__int64)v71;
                  LODWORD(v131) = v131 + 1;
                }
                ++v66;
              }
              while ( v67 != v66 );
              v54 = v70;
              v53 = v69;
              v28 = v68;
              if ( *(_BYTE *)(v54 + 64) == 2 )
                goto LABEL_105;
LABEL_78:
              while ( 1 )
              {
                v54 = *(_QWORD *)(v54 + 8);
                if ( v53 == v54 )
                  break;
                if ( !*(_BYTE *)(v54 + 32) )
                  goto LABEL_69;
              }
            }
          }
        }
LABEL_79:
        if ( *(_BYTE *)(*(_QWORD *)(i - 16) + 8LL) != 7 )
        {
          v25 = v129;
          sub_A45280(v111, v129);
        }
      }
      v108 = *(_QWORD *)(v108 + 8);
    }
    while ( v109 != v108 );
    v58 = v130;
    v59 = &v130[(unsigned int)v131];
    if ( v59 != v130 )
    {
      do
      {
        v60 = *v58;
        v25 = (_BYTE *)a2;
        ++v58;
        sub_A46D60(v111, a2, v60);
      }
      while ( v59 != v58 );
    }
    result = (__int64)v133;
    v61 = &v133[(unsigned int)v134];
    if ( v133 != v61 )
    {
      v62 = v133;
      do
      {
        v63 = *v62;
        v25 = (_BYTE *)a2;
        ++v62;
        result = sub_A46A50(v111, a2, v63);
      }
      while ( v61 != v62 );
      v61 = v133;
    }
    if ( v61 != (__int64 *)v135 )
      result = _libc_free(v61, v25);
  }
  if ( v130 != v24 )
    return _libc_free(v130, v25);
  return result;
}
