// Function: sub_34E0DF0
// Address: 0x34e0df0
//
__int64 __fastcall sub_34E0DF0(__int64 *a1, __int64 *a2, __int64 a3, unsigned __int64 a4, int a5, __int64 *a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  _QWORD *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // r9
  int v13; // r10d
  _QWORD *v14; // rdx
  __int64 v15; // r8
  _QWORD *v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // r12d
  __int64 v19; // r12
  int v20; // eax
  _QWORD *v21; // rdi
  unsigned __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rbx
  _DWORD *v25; // r14
  __int64 *v26; // r13
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  __int64 i; // r15
  __int16 v31; // ax
  __int64 v33; // r8
  __int64 v34; // r9
  int v35; // eax
  int v36; // eax
  int v37; // eax
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  unsigned int v40; // ecx
  __int64 v41; // r12
  __int64 v42; // rbx
  unsigned int v43; // esi
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  unsigned int v46; // r14d
  __int64 v47; // rcx
  __int64 v48; // rbx
  __int64 v49; // rdx
  __int64 v50; // r15
  __int64 v51; // r13
  int v52; // r14d
  unsigned __int64 v53; // r12
  unsigned int v54; // eax
  __int64 v55; // rsi
  _QWORD *v56; // r12
  unsigned int v57; // r15d
  __int64 v58; // r14
  unsigned __int16 ***v59; // rbx
  __int64 v60; // r12
  __int64 v61; // rdx
  int v62; // eax
  int v63; // r14d
  __int64 v64; // rbx
  int v65; // r10d
  _QWORD *v66; // rdx
  unsigned int v67; // edi
  _QWORD *v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // r15
  unsigned int v72; // r8d
  __int64 v73; // rdx
  __int64 v74; // rax
  __int16 v75; // ax
  __int64 v76; // rdi
  __int64 v77; // rax
  int v78; // eax
  __int64 v79; // rcx
  __int64 v80; // r8
  int v81; // edi
  _QWORD *v82; // rsi
  int v83; // esi
  __int64 v84; // r15
  _QWORD *v85; // rcx
  __int64 v86; // rdi
  unsigned int v87; // r14d
  __int64 v88; // rcx
  __int64 v89; // rax
  __int64 j; // rdi
  __int64 v91; // rsi
  int v92; // r10d
  unsigned int v93; // r10d
  unsigned __int64 v94; // [rsp+8h] [rbp-108h]
  __int64 v95; // [rsp+10h] [rbp-100h]
  unsigned int v96; // [rsp+1Ch] [rbp-F4h]
  __int64 v97; // [rsp+20h] [rbp-F0h]
  _QWORD *v99; // [rsp+50h] [rbp-C0h]
  __int64 *v100; // [rsp+58h] [rbp-B8h]
  unsigned int v101; // [rsp+58h] [rbp-B8h]
  unsigned int v102; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v103; // [rsp+60h] [rbp-B0h]
  unsigned __int64 v105; // [rsp+68h] [rbp-A8h]
  __int64 v106; // [rsp+68h] [rbp-A8h]
  __int64 v107; // [rsp+70h] [rbp-A0h]
  unsigned int v108; // [rsp+70h] [rbp-A0h]
  __int64 v109; // [rsp+70h] [rbp-A0h]
  unsigned __int64 v110; // [rsp+78h] [rbp-98h]
  unsigned int v112; // [rsp+88h] [rbp-88h]
  int v114; // [rsp+8Ch] [rbp-84h]
  unsigned int v115; // [rsp+9Ch] [rbp-74h] BYREF
  _BYTE *v116; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v117; // [rsp+A8h] [rbp-68h]
  _BYTE v118[16]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v119; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v120; // [rsp+C8h] [rbp-48h]
  __int64 v121; // [rsp+D0h] [rbp-40h]
  unsigned int v122; // [rsp+D8h] [rbp-38h]

  v6 = a2[1];
  v7 = *a2;
  if ( *a2 == v6 )
  {
    return 0;
  }
  else
  {
    v122 = 0;
    v9 = 0;
    v119 = 0;
    v10 = 0;
    v11 = 0;
    v120 = 0;
    v121 = 0;
    v110 = a4;
    while ( 1 )
    {
      v19 = *(_QWORD *)v7;
      if ( !(_DWORD)v9 )
      {
        ++v119;
LABEL_15:
        sub_34E0C10((__int64)&v119, 2 * (_DWORD)v9);
        if ( !v122 )
          goto LABEL_200;
        v9 = (_QWORD *)(v122 - 1);
        v15 = v120;
        v10 = (unsigned int)v9 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v20 = v121 + 1;
        v14 = (_QWORD *)(v120 + 16 * v10);
        v21 = (_QWORD *)*v14;
        if ( v19 != *v14 )
        {
          v92 = 1;
          v12 = 0;
          while ( v21 != (_QWORD *)-4096LL )
          {
            if ( !v12 && v21 == (_QWORD *)-8192LL )
              v12 = (__int64)v14;
            v10 = (unsigned int)v9 & ((_DWORD)v10 + v92);
            v14 = (_QWORD *)(v120 + 16 * v10);
            v21 = (_QWORD *)*v14;
            if ( v19 == *v14 )
              goto LABEL_17;
            ++v92;
          }
          if ( v12 )
            v14 = (_QWORD *)v12;
        }
        goto LABEL_17;
      }
      v12 = (unsigned int)((_DWORD)v9 - 1);
      v13 = 1;
      v14 = 0;
      v15 = (unsigned int)v12 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v16 = (_QWORD *)(v10 + 16 * v15);
      v17 = *v16;
      if ( v19 != *v16 )
        break;
LABEL_4:
      v16[1] = v7;
      if ( v11 )
        goto LABEL_5;
LABEL_20:
      v11 = v7;
      v7 += 256;
      if ( v6 == v7 )
        goto LABEL_21;
LABEL_12:
      v10 = v120;
      v9 = (_QWORD *)v122;
    }
    while ( v17 != -4096 )
    {
      if ( v17 == -8192 && !v14 )
        v14 = v16;
      v15 = (unsigned int)v12 & (v13 + (_DWORD)v15);
      v16 = (_QWORD *)(v10 + 16LL * (unsigned int)v15);
      v17 = *v16;
      if ( v19 == *v16 )
        goto LABEL_4;
      ++v13;
    }
    if ( !v14 )
      v14 = v16;
    ++v119;
    v20 = v121 + 1;
    if ( 4 * ((int)v121 + 1) >= (unsigned int)(3 * (_DWORD)v9) )
      goto LABEL_15;
    v10 = (unsigned int)v9 >> 3;
    if ( (int)v9 - (v20 + HIDWORD(v121)) <= (unsigned int)v10 )
    {
      sub_34E0C10((__int64)&v119, (int)v9);
      if ( !v122 )
      {
LABEL_200:
        LODWORD(v121) = v121 + 1;
        BUG();
      }
      v10 = v122 - 1;
      v12 = 1;
      v57 = v10 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v15 = 0;
      v20 = v121 + 1;
      v14 = (_QWORD *)(v120 + 16LL * v57);
      v9 = (_QWORD *)*v14;
      if ( v19 != *v14 )
      {
        while ( v9 != (_QWORD *)-4096LL )
        {
          if ( v9 == (_QWORD *)-8192LL && !v15 )
            v15 = (__int64)v14;
          v93 = v12 + 1;
          v12 = (unsigned int)v10 & (v57 + (_DWORD)v12);
          v57 = v12;
          v14 = (_QWORD *)(v120 + 16LL * (unsigned int)v12);
          v9 = (_QWORD *)*v14;
          if ( v19 == *v14 )
            goto LABEL_17;
          v12 = v93;
        }
        if ( v15 )
          v14 = (_QWORD *)v15;
      }
    }
LABEL_17:
    LODWORD(v121) = v20;
    if ( *v14 != -4096 )
      --HIDWORD(v121);
    *v14 = v19;
    v14[1] = 0;
    v14[1] = v7;
    if ( !v11 )
      goto LABEL_20;
LABEL_5:
    if ( (*(_BYTE *)(v7 + 254) & 1) == 0 )
      sub_2F8F5D0(v7, v9, (__int64)v14, v10, v15, v12);
    v18 = *(_DWORD *)(v7 + 240) + *(unsigned __int16 *)(v7 + 252);
    if ( (*(_BYTE *)(v11 + 254) & 1) == 0 )
      sub_2F8F5D0(v11, v9, (__int64)v14, v10, v15, v12);
    if ( v18 > *(_DWORD *)(v11 + 240) + (unsigned int)*(unsigned __int16 *)(v11 + 252) )
      v11 = v7;
    v7 += 256;
    if ( v6 != v7 )
      goto LABEL_12;
LABEL_21:
    v22 = v110;
    v107 = *(_QWORD *)v11;
    v23 = *(unsigned int *)(a1[4] + 16);
    v24 = 4 * v23;
    if ( *(_DWORD *)(a1[4] + 16) )
    {
      v103 = sub_22077B0(4 * v23);
      v25 = (_DWORD *)v103;
      do
      {
        if ( v25 )
          *v25 = 0;
        ++v25;
      }
      while ( (_DWORD *)(v24 + v103) != v25 );
      v114 = a5 - 1;
      if ( v110 != a3 )
      {
LABEL_27:
        v112 = 0;
        v99 = (_QWORD *)v11;
        v26 = a1;
LABEL_28:
        v27 = (_QWORD *)(*(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL);
        v28 = v27;
        if ( !v27 )
          BUG();
        v22 = *(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL;
        v29 = *v27;
        if ( (v29 & 4) == 0 && (*((_BYTE *)v28 + 44) & 4) != 0 )
        {
          for ( i = v29; ; i = *(_QWORD *)v22 )
          {
            v22 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v22 + 44) & 4) == 0 )
              break;
          }
        }
        v31 = *(_WORD *)(v22 + 68);
        if ( (unsigned __int16)(v31 - 14) <= 4u || v31 == 7 )
          goto LABEL_36;
        v115 = 0;
        if ( v22 == v107 )
        {
          v47 = 0;
          v48 = v99[5];
          v49 = v48 + 16LL * *((unsigned int *)v99 + 12);
          if ( v48 == v49 )
            goto LABEL_104;
          v105 = v22;
          v50 = 0;
          v100 = v26;
          v51 = v48 + 16LL * *((unsigned int *)v99 + 12);
          do
          {
            while ( 1 )
            {
              v52 = *(_DWORD *)(v48 + 12);
              v53 = *(_QWORD *)v48 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v53 + 254) & 1) == 0 )
              {
                v108 = v47;
                sub_2F8F5D0(*(_QWORD *)v48 & 0xFFFFFFFFFFFFFFF8LL, 0, v49, v47, v15, v12);
                v47 = v108;
              }
              v54 = v52 + *(_DWORD *)(v53 + 240);
              if ( v54 <= (unsigned int)v47 )
                break;
              v50 = v48;
              v48 += 16;
              v47 = v54;
              if ( v51 == v48 )
                goto LABEL_80;
            }
            if ( v54 == (_DWORD)v47 && ((*(__int64 *)v48 >> 1) & 3) == 1 )
              v50 = v48;
            v48 += 16;
          }
          while ( v51 != v48 );
LABEL_80:
          v55 = v50;
          v26 = v100;
          v22 = v105;
          if ( !v55 )
          {
LABEL_104:
            v107 = 0;
            v99 = 0;
          }
          else
          {
            v56 = (_QWORD *)(*(_QWORD *)v55 & 0xFFFFFFFFFFFFFFF8LL);
            if ( ((*(__int64 *)v55 >> 1) & 3) == 1 )
            {
              v88 = v100[2];
              v115 = *(_DWORD *)(v55 + 8);
              v87 = v115;
              v109 = v88;
              if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)v88 + 16LL)
                                                                                         + 200LL))(*(_QWORD *)(*(_QWORD *)v88 + 16LL))
                                                     + 248)
                                         + 16LL)
                             + v87)
                || (*(_QWORD *)(*(_QWORD *)(v109 + 384) + 8LL * (v87 >> 6)) & (1LL << v87)) != 0
                || (*(_QWORD *)(v100[30] + 8LL * (v115 >> 6)) & (1LL << v115)) != 0 )
              {
LABEL_171:
                v115 = 0;
              }
              else
              {
                v89 = v99[5];
                for ( j = v89 + 16LL * *((unsigned int *)v99 + 12); j != v89; v89 += 16 )
                {
                  v91 = (*(__int64 *)v89 >> 1) & 3;
                  if ( v56 == (_QWORD *)(*(_QWORD *)v89 & 0xFFFFFFFFFFFFFFF8LL) )
                  {
                    if ( v91 != 1 || *(_DWORD *)(v89 + 8) != v115 )
                      goto LABEL_171;
                  }
                  else if ( !v91 && v115 == *(_DWORD *)(v89 + 8) )
                  {
                    goto LABEL_171;
                  }
                }
              }
            }
            v99 = v56;
            v107 = *v56;
          }
        }
        sub_34E00F0(v26, v22);
        v116 = v118;
        v117 = 0x200000000LL;
        v35 = *(_DWORD *)(v22 + 44);
        if ( (v35 & 4) != 0 || (v35 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v22 + 16) + 24LL) & 0x80u) != 0LL )
            goto LABEL_45;
        }
        else if ( sub_2E88A90(v22, 128, 1) )
        {
          goto LABEL_45;
        }
        v36 = *(_DWORD *)(v22 + 44);
        if ( (v36 & 4) != 0 || (v36 & 8) == 0 )
          v37 = *(_DWORD *)(*(_QWORD *)(v22 + 16) + 28LL) & 1;
        else
          LOBYTE(v37) = sub_2E88A90(v22, 0x100000000LL, 1);
        if ( (_BYTE)v37 )
          goto LABEL_45;
        v38 = v26[3];
        v39 = *(__int64 (**)())(*(_QWORD *)v38 + 920LL);
        if ( v39 != sub_2DB1B30 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v39)(v38, v22) )
            goto LABEL_45;
        }
        v40 = v115;
        if ( !v115 )
          goto LABEL_46;
        v41 = *(_QWORD *)(v22 + 32);
        v42 = v41 + 40LL * (*(_DWORD *)(v22 + 40) & 0xFFFFFF);
        if ( v41 != v42 )
        {
          v43 = v115;
          do
          {
            if ( !*(_BYTE *)v41 )
            {
              v46 = *(_DWORD *)(v41 + 8);
              if ( v46 )
              {
                if ( (*(_BYTE *)(v41 + 3) & 0x10) != 0 )
                  goto LABEL_56;
                if ( v46 == v43 )
                  goto LABEL_45;
                if ( v43 - 1 <= 0x3FFFFFFE && v46 - 1 <= 0x3FFFFFFE )
                {
                  if ( (unsigned __int8)sub_E92070(v26[4], v43, v46) )
                    goto LABEL_45;
                  v43 = v115;
                  if ( (*(_BYTE *)(v41 + 3) & 0x10) != 0 )
                  {
LABEL_56:
                    if ( v46 != v43 )
                    {
                      v44 = (unsigned int)v117;
                      v45 = (unsigned int)v117 + 1LL;
                      if ( v45 > HIDWORD(v117) )
                      {
                        sub_C8D5F0((__int64)&v116, v118, v45, 4u, v33, v34);
                        v44 = (unsigned int)v117;
                      }
                      *(_DWORD *)&v116[4 * v44] = v46;
                      v43 = v115;
                      LODWORD(v117) = v117 + 1;
                    }
                  }
                }
              }
            }
            v41 += 40;
          }
          while ( v42 != v41 );
          v40 = v43;
          if ( !v43 )
            goto LABEL_46;
        }
        v58 = v40;
        v59 = *(unsigned __int16 ****)(v26[15] + 8LL * v40);
        if ( v59 == (unsigned __int16 ***)-1LL )
        {
LABEL_45:
          v115 = 0;
          goto LABEL_46;
        }
        v101 = v40;
        v60 = sub_34E04E0((__int64)(v26 + 18), &v115);
        v106 = v61;
        v62 = sub_34DFE70(v26, v60, v61, v101, *(_DWORD *)(v103 + 4 * v58), v59, (__int64)&v116);
        v102 = v62;
        if ( !v62 )
          goto LABEL_46;
        if ( v106 == v60 )
          goto LABEL_127;
        v94 = v22;
        v63 = v62;
        while ( 1 )
        {
          sub_2EAB0C0(*(_QWORD *)(v60 + 40), v63);
          v64 = *(_QWORD *)(*(_QWORD *)(v60 + 40) + 16LL);
          if ( v122 )
          {
            v65 = 1;
            v66 = 0;
            v67 = (v122 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
            v68 = (_QWORD *)(v120 + 16LL * v67);
            v69 = *v68;
            if ( v64 == *v68 )
            {
LABEL_112:
              if ( v68[1] )
              {
                v70 = *a6;
                v71 = a6[1];
                v72 = v115;
                if ( *a6 != v71 )
                {
                  v73 = 0;
                  while ( 1 )
                  {
                    while ( 1 )
                    {
                      v74 = *(_QWORD *)(v71 - 8);
                      if ( v64 != v74 && v73 != v74 )
                      {
                        if ( v73 )
                          goto LABEL_125;
                        goto LABEL_116;
                      }
                      v73 = *(_QWORD *)(v71 - 16);
                      v75 = *(_WORD *)(v73 + 68);
                      if ( v75 != 14 )
                        break;
LABEL_122:
                      v76 = *(_QWORD *)(v73 + 32);
                      if ( *(_BYTE *)v76 || v72 != *(_DWORD *)(v76 + 8) )
                        goto LABEL_116;
LABEL_124:
                      v95 = v70;
                      v71 -= 16;
                      v96 = v72;
                      v97 = v73;
                      sub_2EAB0C0(v76, v63);
                      v70 = v95;
                      v73 = v97;
                      v72 = v96;
                      if ( v95 == v71 )
                        goto LABEL_125;
                    }
                    if ( v75 != 15 )
                    {
                      if ( v75 != 17 )
                        BUG();
                      goto LABEL_122;
                    }
                    v77 = *(_QWORD *)(v73 + 32);
                    if ( !*(_BYTE *)(v77 + 80) )
                    {
                      v76 = v77 + 80;
                      if ( v72 == *(_DWORD *)(v77 + 88) )
                        goto LABEL_124;
                    }
LABEL_116:
                    v71 -= 16;
                    if ( v70 == v71 )
                      goto LABEL_125;
                  }
                }
              }
              goto LABEL_125;
            }
            while ( v69 != -4096 )
            {
              if ( !v66 && v69 == -8192 )
                v66 = v68;
              v67 = (v122 - 1) & (v65 + v67);
              v68 = (_QWORD *)(v120 + 16LL * v67);
              v69 = *v68;
              if ( v64 == *v68 )
                goto LABEL_112;
              ++v65;
            }
            if ( !v66 )
              v66 = v68;
            ++v119;
            v78 = v121 + 1;
            if ( 4 * ((int)v121 + 1) < 3 * v122 )
            {
              if ( v122 - HIDWORD(v121) - v78 <= v122 >> 3 )
              {
                sub_34E0C10((__int64)&v119, v122);
                if ( !v122 )
                {
LABEL_201:
                  LODWORD(v121) = v121 + 1;
                  BUG();
                }
                v83 = 1;
                LODWORD(v84) = (v122 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                v85 = 0;
                v78 = v121 + 1;
                v66 = (_QWORD *)(v120 + 16LL * (unsigned int)v84);
                v86 = *v66;
                if ( v64 != *v66 )
                {
                  while ( v86 != -4096 )
                  {
                    if ( !v85 && v86 == -8192 )
                      v85 = v66;
                    v84 = (v122 - 1) & ((_DWORD)v84 + v83);
                    v66 = (_QWORD *)(v120 + 16 * v84);
                    v86 = *v66;
                    if ( v64 == *v66 )
                      goto LABEL_141;
                    ++v83;
                  }
                  if ( v85 )
                    v66 = v85;
                }
              }
              goto LABEL_141;
            }
          }
          else
          {
            ++v119;
          }
          sub_34E0C10((__int64)&v119, 2 * v122);
          if ( !v122 )
            goto LABEL_201;
          LODWORD(v79) = (v122 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
          v78 = v121 + 1;
          v66 = (_QWORD *)(v120 + 16LL * (unsigned int)v79);
          v80 = *v66;
          if ( v64 != *v66 )
          {
            v81 = 1;
            v82 = 0;
            while ( v80 != -4096 )
            {
              if ( v80 == -8192 && !v82 )
                v82 = v66;
              v79 = (v122 - 1) & ((_DWORD)v79 + v81);
              v66 = (_QWORD *)(v120 + 16 * v79);
              v80 = *v66;
              if ( v64 == *v66 )
                goto LABEL_141;
              ++v81;
            }
            if ( v82 )
              v66 = v82;
          }
LABEL_141:
          LODWORD(v121) = v78;
          if ( *v66 != -4096 )
            --HIDWORD(v121);
          *v66 = v64;
          v66[1] = 0;
LABEL_125:
          v60 = sub_220EEE0(v60);
          if ( v106 == v60 )
          {
            v22 = v94;
LABEL_127:
            *(_QWORD *)(v26[15] + 8LL * v102) = *(_QWORD *)(v26[15] + 8LL * v115);
            *(_DWORD *)(v26[27] + 4LL * v102) = *(_DWORD *)(v26[27] + 4LL * v115);
            *(_DWORD *)(v26[24] + 4LL * v102) = *(_DWORD *)(v26[24] + 4LL * v115);
            *(_QWORD *)(v26[15] + 8LL * v115) = 0;
            *(_DWORD *)(v26[27] + 4LL * v115) = *(_DWORD *)(v26[24] + 4LL * v115);
            *(_DWORD *)(v26[24] + 4LL * v115) = -1;
            sub_34E0580(v26 + 18, &v115);
            ++v112;
            *(_DWORD *)(v103 + 4LL * v115) = v102;
LABEL_46:
            sub_34E0650((__int64)v26, v22, v114);
            if ( v116 != v118 )
              _libc_free((unsigned __int64)v116);
LABEL_36:
            --v114;
            if ( v22 == a3 )
              goto LABEL_37;
            goto LABEL_28;
          }
        }
      }
      v112 = 0;
LABEL_37:
      if ( v103 )
        j_j___libc_free_0(v103);
    }
    else
    {
      v103 = 0;
      v114 = a5 - 1;
      if ( a3 != v110 )
        goto LABEL_27;
      v112 = 0;
    }
    sub_C7D6A0(v120, 16LL * v122, 8);
  }
  return v112;
}
