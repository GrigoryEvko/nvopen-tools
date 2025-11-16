// Function: sub_20E74E0
// Address: 0x20e74e0
//
__int64 __fastcall sub_20E74E0(_QWORD *a1, __int64 *a2, __int64 a3, unsigned __int64 a4, int a5, __int64 *a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v10; // r12
  _QWORD *v11; // rsi
  __int64 v12; // r15
  __int64 *v13; // r13
  __int64 v14; // rdx
  _QWORD *v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r8
  _QWORD *v18; // rax
  __int64 v19; // rdi
  unsigned int v20; // r14d
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rdi
  __int64 v24; // rax
  size_t v25; // r12
  unsigned __int64 v26; // r13
  int v27; // r14d
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rax
  __int64 i; // r13
  __int16 v32; // ax
  _WORD *v33; // r8
  __int16 v34; // ax
  __int16 v36; // ax
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  unsigned int v40; // r15d
  __int64 v41; // rax
  int v42; // edi
  __int64 v43; // rbx
  int v44; // r15d
  unsigned __int16 *v45; // r9
  _BYTE **v46; // r10
  __int64 v47; // r14
  __int64 v48; // rax
  __int64 v49; // rax
  int v50; // r12d
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rcx
  unsigned int v54; // esi
  __int16 v55; // ax
  _WORD *v56; // rsi
  unsigned __int16 v57; // ax
  _WORD *v58; // rcx
  unsigned int v59; // esi
  int v60; // edx
  unsigned int k; // ecx
  bool v62; // cf
  int v63; // r11d
  _QWORD *v64; // r10
  int v65; // r11d
  __int64 v66; // rdi
  __int64 v67; // rsi
  __int64 v68; // rcx
  __int64 v69; // rbx
  __int64 v70; // rdx
  __int64 v71; // r14
  unsigned int v72; // r13d
  int v73; // r15d
  unsigned __int64 v74; // r12
  unsigned int v75; // eax
  unsigned __int64 v76; // r12
  unsigned int v77; // r15d
  __int64 v78; // rcx
  __int64 (*v79)(); // rax
  int v80; // r11d
  __int64 **v81; // rbx
  __int64 v82; // r12
  __int64 v83; // rdx
  __int64 v84; // r15
  __int64 v85; // rbx
  unsigned int v86; // edi
  _QWORD *v87; // rax
  __int64 v88; // r8
  __int64 v89; // r8
  __int64 v90; // rax
  unsigned int v91; // r9d
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // rdi
  int v95; // ecx
  int v96; // r11d
  _QWORD *v97; // rcx
  int v98; // eax
  unsigned int v99; // edx
  __int64 v100; // r8
  int v101; // edi
  _QWORD *v102; // rsi
  int v103; // edi
  unsigned int v104; // edx
  __int64 v105; // r8
  __int64 v106; // rax
  __int64 j; // rsi
  __int64 v108; // rdi
  __int64 v109; // [rsp+18h] [rbp-108h]
  __int64 v110; // [rsp+20h] [rbp-100h]
  __int64 v111; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v112; // [rsp+48h] [rbp-D8h]
  __int64 v113; // [rsp+48h] [rbp-D8h]
  unsigned int v114; // [rsp+48h] [rbp-D8h]
  char *v115; // [rsp+50h] [rbp-D0h]
  int v117; // [rsp+60h] [rbp-C0h]
  unsigned int v118; // [rsp+60h] [rbp-C0h]
  _BYTE **v119; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v121; // [rsp+68h] [rbp-B8h]
  unsigned int v123; // [rsp+70h] [rbp-B0h]
  _DWORD *v124; // [rsp+78h] [rbp-A8h]
  __int64 v125; // [rsp+80h] [rbp-A0h]
  __int64 v126; // [rsp+80h] [rbp-A0h]
  __int64 v127; // [rsp+88h] [rbp-98h]
  unsigned int v130; // [rsp+ACh] [rbp-74h] BYREF
  _BYTE *v131; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v132; // [rsp+B8h] [rbp-68h]
  _BYTE v133[16]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v134; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v135; // [rsp+D8h] [rbp-48h]
  __int64 v136; // [rsp+E0h] [rbp-40h]
  unsigned int v137; // [rsp+E8h] [rbp-38h]

  v6 = a2[1];
  v7 = *a2;
  if ( *a2 == v6 )
    return 0;
  v134 = 0;
  v135 = 0;
  v8 = 0xF0F0F0F0F0F0F0F1LL * ((v6 - v7) >> 4);
  v136 = 0;
  v137 = 0;
  if ( !(_DWORD)v8 )
    BUG();
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = a2;
  v14 = 0;
  v127 = 272LL * (unsigned int)(v8 - 1);
  while ( 1 )
  {
    v21 = v10 + v7;
    v22 = *(_QWORD *)(v21 + 8);
    if ( !(_DWORD)v11 )
    {
      ++v134;
      goto LABEL_16;
    }
    LODWORD(v15) = (_DWORD)v11 - 1;
    v16 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
    v17 = ((_DWORD)v11 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v18 = (_QWORD *)(v14 + 16 * v17);
    v19 = *v18;
    if ( v22 != *v18 )
      break;
LABEL_5:
    v18[1] = v21;
    if ( v12 )
      goto LABEL_6;
LABEL_21:
    v12 = v21;
    if ( v127 == v10 )
      goto LABEL_22;
LABEL_13:
    v7 = *v13;
    v14 = v135;
    v10 += 272;
    v11 = (_QWORD *)v137;
  }
  v63 = 1;
  v64 = 0;
  while ( v19 != -8 )
  {
    if ( v19 == -16 && !v64 )
      v64 = v18;
    v17 = (unsigned int)v15 & (v63 + (_DWORD)v17);
    v18 = (_QWORD *)(v14 + 16LL * (unsigned int)v17);
    v19 = *v18;
    if ( v22 == *v18 )
      goto LABEL_5;
    ++v63;
  }
  if ( v64 )
    v18 = v64;
  ++v134;
  v14 = (unsigned int)(v136 + 1);
  if ( 4 * (int)v14 >= (unsigned int)(3 * (_DWORD)v11) )
  {
LABEL_16:
    sub_20E7320((__int64)&v134, 2 * (_DWORD)v11);
    if ( !v137 )
      goto LABEL_195;
    v11 = (_QWORD *)(v137 - 1);
    v17 = v135;
    v16 = (unsigned int)v11 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v14 = (unsigned int)(v136 + 1);
    v18 = (_QWORD *)(v135 + 16 * v16);
    v23 = *v18;
    if ( v22 != *v18 )
    {
      v80 = 1;
      v15 = 0;
      while ( v23 != -8 )
      {
        if ( !v15 && v23 == -16 )
          v15 = v18;
        v16 = (unsigned int)v11 & (v80 + (_DWORD)v16);
        v18 = (_QWORD *)(v135 + 16LL * (unsigned int)v16);
        v23 = *v18;
        if ( v22 == *v18 )
          goto LABEL_18;
        ++v80;
      }
LABEL_113:
      if ( v15 )
        v18 = v15;
      goto LABEL_18;
    }
    goto LABEL_18;
  }
  v17 = (unsigned int)((_DWORD)v11 - (v14 + HIDWORD(v136)));
  if ( (unsigned int)v17 <= (unsigned int)v11 >> 3 )
  {
    sub_20E7320((__int64)&v134, (int)v11);
    if ( !v137 )
    {
LABEL_195:
      LODWORD(v136) = v136 + 1;
      BUG();
    }
    v11 = (_QWORD *)(v137 - 1);
    v17 = v135;
    v15 = 0;
    v65 = 1;
    v16 = (unsigned int)v11 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v14 = (unsigned int)(v136 + 1);
    v18 = (_QWORD *)(v135 + 16 * v16);
    v66 = *v18;
    if ( v22 != *v18 )
    {
      while ( v66 != -8 )
      {
        if ( !v15 && v66 == -16 )
          v15 = v18;
        v16 = (unsigned int)v11 & (v65 + (_DWORD)v16);
        v18 = (_QWORD *)(v135 + 16LL * (unsigned int)v16);
        v66 = *v18;
        if ( v22 == *v18 )
          goto LABEL_18;
        ++v65;
      }
      goto LABEL_113;
    }
  }
LABEL_18:
  LODWORD(v136) = v14;
  if ( *v18 != -8 )
    --HIDWORD(v136);
  v18[1] = 0;
  *v18 = v22;
  v18[1] = v21;
  if ( !v12 )
    goto LABEL_21;
LABEL_6:
  if ( (*(_BYTE *)(v21 + 236) & 1) == 0 )
    sub_1F01DD0(v21, v11, v14, v16, v17, (int)v15);
  v20 = *(_DWORD *)(v21 + 240) + *(unsigned __int16 *)(v21 + 226);
  if ( (*(_BYTE *)(v12 + 236) & 1) == 0 )
    sub_1F01DD0(v12, v11, v14, v16, v17, (int)v15);
  if ( v20 > *(_DWORD *)(v12 + 240) + (unsigned int)*(unsigned __int16 *)(v12 + 226) )
    v12 = v21;
  if ( v127 != v10 )
    goto LABEL_13;
LABEL_22:
  v125 = *(_QWORD *)(v12 + 8);
  v24 = a1[4];
  if ( !*(_DWORD *)(v24 + 16) )
  {
    v26 = a4;
    v115 = 0;
    v124 = 0;
    v27 = a5 - 1;
    if ( a4 == a3 )
    {
      v123 = 0;
      goto LABEL_43;
    }
LABEL_24:
    v121 = v12;
    v123 = 0;
    while ( 1 )
    {
      v28 = (_QWORD *)(*(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL);
      v29 = v28;
      if ( !v28 )
        BUG();
      v26 = *(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL;
      v30 = *v28;
      if ( (v30 & 4) == 0 && (*((_BYTE *)v29 + 46) & 4) != 0 )
      {
        for ( i = v30; ; i = *(_QWORD *)v26 )
        {
          v26 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v26 + 46) & 4) == 0 )
            break;
        }
      }
      v32 = **(_WORD **)(v26 + 16);
      if ( (unsigned __int16)(v32 - 12) > 1u && v32 != 6 )
      {
        v130 = 0;
        if ( v26 == v125 )
        {
          v67 = 0;
          v68 = 0;
          v69 = *(_QWORD *)(v121 + 32);
          v70 = v69 + 16LL * *(unsigned int *)(v121 + 40);
          if ( v69 == v70 )
            goto LABEL_102;
          v117 = v27;
          v71 = v69 + 16LL * *(unsigned int *)(v121 + 40);
          v112 = v26;
          v72 = 0;
          do
          {
            while ( 1 )
            {
              v73 = *(_DWORD *)(v69 + 12);
              v74 = *(_QWORD *)v69 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v74 + 236) & 1) == 0 )
                sub_1F01DD0(*(_QWORD *)v69 & 0xFFFFFFFFFFFFFFF8LL, (_QWORD *)v67, v70, v68, v17, (int)v15);
              v75 = v73 + *(_DWORD *)(v74 + 240);
              if ( v72 >= v75 )
                break;
              v67 = v69;
              v69 += 16;
              v72 = v73 + *(_DWORD *)(v74 + 240);
              if ( v71 == v69 )
                goto LABEL_97;
            }
            if ( v72 == v75 && ((*(__int64 *)v69 >> 1) & 3) == 1 )
              v67 = v69;
            v69 += 16;
          }
          while ( v71 != v69 );
LABEL_97:
          v27 = v117;
          v26 = v112;
          if ( !v67 )
          {
LABEL_102:
            v125 = 0;
            v121 = 0;
          }
          else
          {
            v76 = *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
            if ( ((*(__int64 *)v67 >> 1) & 3) == 1 )
            {
              v77 = *(_DWORD *)(v67 + 8);
              v78 = a1[2];
              v130 = v77;
              v126 = v78;
              v79 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)v78 + 16LL) + 112LL);
              if ( v79 == sub_1D00B10 )
                BUG();
              if ( !*(_BYTE *)(*(_QWORD *)(v79() + 232) + 8LL * v77 + 4)
                || (*(_QWORD *)(*(_QWORD *)(v126 + 304) + 8LL * (v77 >> 6)) & (1LL << v77)) != 0
                || (*(_QWORD *)(a1[24] + 8LL * (v130 >> 6)) & (1LL << v130)) != 0 )
              {
LABEL_110:
                v130 = 0;
              }
              else
              {
                v106 = *(_QWORD *)(v121 + 32);
                for ( j = v106 + 16LL * *(unsigned int *)(v121 + 40); v106 != j; v106 += 16 )
                {
                  v108 = (*(__int64 *)v106 >> 1) & 3;
                  if ( v76 == (*(_QWORD *)v106 & 0xFFFFFFFFFFFFFFF8LL) )
                  {
                    if ( v108 != 1 || v130 != *(_DWORD *)(v106 + 8) )
                      goto LABEL_110;
                  }
                  else if ( !v108 && v130 == *(_DWORD *)(v106 + 8) )
                  {
                    goto LABEL_110;
                  }
                }
              }
            }
            v121 = v76;
            v125 = *(_QWORD *)(v76 + 8);
          }
        }
        sub_20E6570((__int64)a1, v26);
        v131 = v133;
        v132 = 0x200000000LL;
        v34 = *(_WORD *)(v26 + 46);
        if ( (v34 & 4) != 0 || (v34 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v26 + 16) + 8LL) & 0x10LL) != 0 )
            goto LABEL_37;
        }
        else if ( sub_1E15D00(v26, 0x10u, 1) )
        {
          goto LABEL_37;
        }
        v36 = *(_WORD *)(v26 + 46);
        if ( (v36 & 4) == 0 && (v36 & 8) != 0 )
          LOBYTE(v37) = sub_1E15D00(v26, 0x10000000u, 1);
        else
          v37 = (*(_QWORD *)(*(_QWORD *)(v26 + 16) + 8LL) >> 28) & 1LL;
        if ( (_BYTE)v37
          || (v38 = a1[3], v39 = *(__int64 (**)())(*(_QWORD *)v38 + 656LL), v39 != sub_1D918C0)
          && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v39)(v38, v26) )
        {
LABEL_37:
          v130 = 0;
        }
        else
        {
          v40 = v130;
          if ( v130 )
          {
            v41 = *(unsigned int *)(v26 + 40);
            if ( !(_DWORD)v41 )
              goto LABEL_117;
            v42 = v130;
            v43 = 0;
            v44 = v27;
            LODWORD(v45) = 40 * v41;
            v46 = &v131;
            v47 = 40 * v41;
            do
            {
              v49 = v43 + *(_QWORD *)(v26 + 32);
              if ( !*(_BYTE *)v49 )
              {
                v50 = *(_DWORD *)(v49 + 8);
                if ( v50 )
                {
                  if ( (*(_BYTE *)(v49 + 3) & 0x10) != 0 )
                  {
                    if ( v50 != v42 )
                    {
                      v48 = (unsigned int)v132;
                      if ( (unsigned int)v132 >= HIDWORD(v132) )
                      {
                        v119 = v46;
                        sub_16CD150((__int64)v46, v133, 0, 4, (int)v33, (int)v45);
                        v48 = (unsigned int)v132;
                        v46 = v119;
                      }
                      *(_DWORD *)&v131[4 * v48] = v50;
                      v42 = v130;
                      LODWORD(v132) = v132 + 1;
                    }
                  }
                  else
                  {
                    if ( v50 == v42 )
                      goto LABEL_72;
                    if ( v42 >= 0 && v50 >= 0 )
                    {
                      v51 = a1[4];
                      v52 = *(_QWORD *)(v51 + 8);
                      v53 = *(_QWORD *)(v51 + 56);
                      v54 = *(_DWORD *)(v52 + 24LL * (unsigned int)v42 + 16);
                      v55 = v42 * (v54 & 0xF);
                      v56 = (_WORD *)(v53 + 2LL * (v54 >> 4));
                      v57 = *v56 + v55;
                      v33 = v56 + 1;
                      LODWORD(v56) = *(_DWORD *)(v52 + 24LL * (unsigned int)v50 + 16);
                      v60 = v50 * ((unsigned __int8)v56 & 0xF);
                      v58 = (_WORD *)(v53 + 2LL * ((unsigned int)v56 >> 4));
                      v59 = v57;
                      LOWORD(v60) = *v58 + v60;
                      v45 = v58 + 1;
                      for ( k = (unsigned __int16)v60; ; k = (unsigned __int16)v60 )
                      {
                        v62 = v59 < k;
                        if ( v59 == k )
                          break;
                        while ( v62 )
                        {
                          v57 += *v33;
                          if ( !*v33 )
                            goto LABEL_60;
                          v59 = v57;
                          ++v33;
                          v62 = v57 < k;
                          if ( v57 == k )
                            goto LABEL_72;
                        }
                        v95 = *v45;
                        if ( !(_WORD)v95 )
                          goto LABEL_60;
                        v60 += v95;
                        ++v45;
                      }
LABEL_72:
                      v27 = v44;
                      goto LABEL_37;
                    }
                  }
                }
              }
LABEL_60:
              v43 += 40;
            }
            while ( v47 != v43 );
            v27 = v44;
            v40 = v42;
            if ( v42 )
            {
LABEL_117:
              v81 = *(__int64 ***)(a1[9] + 8LL * v40);
              if ( v81 == (__int64 **)-1LL )
                goto LABEL_37;
              v82 = sub_20E6AF0((__int64)(a1 + 12), &v130);
              v113 = v83;
              v118 = sub_20E62E0(a1, v82, v83, v40, v124[v40], v81, (__int64)&v131);
              if ( v118 )
              {
                if ( v82 != v113 )
                {
                  v84 = v113;
                  while ( 1 )
                  {
                    sub_1E310D0(*(_QWORD *)(v82 + 40), v118);
                    v85 = *(_QWORD *)(*(_QWORD *)(v82 + 40) + 16LL);
                    if ( !v137 )
                      break;
                    v86 = (v137 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                    v87 = (_QWORD *)(v135 + 16LL * v86);
                    v88 = *v87;
                    if ( v85 != *v87 )
                    {
                      v96 = 1;
                      v97 = 0;
                      while ( v88 != -8 )
                      {
                        if ( !v97 && v88 == -16 )
                          v97 = v87;
                        v86 = (v137 - 1) & (v96 + v86);
                        v87 = (_QWORD *)(v135 + 16LL * v86);
                        v88 = *v87;
                        if ( v85 == *v87 )
                          goto LABEL_123;
                        ++v96;
                      }
                      if ( !v97 )
                        v97 = v87;
                      ++v134;
                      v98 = v136 + 1;
                      if ( 4 * ((int)v136 + 1) < 3 * v137 )
                      {
                        if ( v137 - HIDWORD(v136) - v98 <= v137 >> 3 )
                        {
                          sub_20E7320((__int64)&v134, v137);
                          if ( !v137 )
                          {
LABEL_192:
                            LODWORD(v136) = v136 + 1;
                            BUG();
                          }
                          v102 = 0;
                          v103 = 1;
                          v104 = (v137 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                          v98 = v136 + 1;
                          v97 = (_QWORD *)(v135 + 16LL * v104);
                          v105 = *v97;
                          if ( v85 != *v97 )
                          {
                            while ( v105 != -8 )
                            {
                              if ( !v102 && v105 == -16 )
                                v102 = v97;
                              v104 = (v137 - 1) & (v103 + v104);
                              v97 = (_QWORD *)(v135 + 16LL * v104);
                              v105 = *v97;
                              if ( v85 == *v97 )
                                goto LABEL_143;
                              ++v103;
                            }
                            goto LABEL_151;
                          }
                        }
                        goto LABEL_143;
                      }
LABEL_147:
                      sub_20E7320((__int64)&v134, 2 * v137);
                      if ( !v137 )
                        goto LABEL_192;
                      v99 = (v137 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                      v98 = v136 + 1;
                      v97 = (_QWORD *)(v135 + 16LL * v99);
                      v100 = *v97;
                      if ( v85 != *v97 )
                      {
                        v101 = 1;
                        v102 = 0;
                        while ( v100 != -8 )
                        {
                          if ( !v102 && v100 == -16 )
                            v102 = v97;
                          v99 = (v137 - 1) & (v101 + v99);
                          v97 = (_QWORD *)(v135 + 16LL * v99);
                          v100 = *v97;
                          if ( v85 == *v97 )
                            goto LABEL_143;
                          ++v101;
                        }
LABEL_151:
                        if ( v102 )
                          v97 = v102;
                      }
LABEL_143:
                      LODWORD(v136) = v98;
                      if ( *v97 != -8 )
                        --HIDWORD(v136);
                      *v97 = v85;
                      v97[1] = 0;
                      goto LABEL_133;
                    }
LABEL_123:
                    if ( v87[1] )
                    {
                      v89 = *a6;
                      v90 = a6[1];
                      v91 = v130;
                      if ( *a6 != v90 )
                      {
                        v92 = 0;
                        while ( 1 )
                        {
                          v93 = *(_QWORD *)(v90 - 8);
                          if ( v85 != v93 && v93 != v92 )
                            break;
                          v92 = *(_QWORD *)(v90 - 16);
                          v94 = *(_QWORD *)(v92 + 32);
                          if ( *(_BYTE *)v94 || v91 != *(_DWORD *)(v94 + 8) )
                          {
LABEL_127:
                            v90 -= 16;
                            if ( v89 == v90 )
                              goto LABEL_133;
                          }
                          else
                          {
                            v109 = *(_QWORD *)(v90 - 16);
                            v110 = v89;
                            v111 = v90;
                            v114 = v91;
                            sub_1E310D0(v94, v118);
                            v89 = v110;
                            v91 = v114;
                            v92 = v109;
                            v90 = v111 - 16;
                            if ( v110 == v111 - 16 )
                              goto LABEL_133;
                          }
                        }
                        if ( v92 )
                          goto LABEL_133;
                        goto LABEL_127;
                      }
                    }
LABEL_133:
                    v82 = sub_220EEE0(v82);
                    if ( v84 == v82 )
                      goto LABEL_134;
                  }
                  ++v134;
                  goto LABEL_147;
                }
LABEL_134:
                *(_QWORD *)(a1[9] + 8LL * v118) = *(_QWORD *)(a1[9] + 8LL * v130);
                *(_DWORD *)(a1[21] + 4LL * v118) = *(_DWORD *)(a1[21] + 4LL * v130);
                *(_DWORD *)(a1[18] + 4LL * v118) = *(_DWORD *)(a1[18] + 4LL * v130);
                *(_QWORD *)(a1[9] + 8LL * v130) = 0;
                *(_DWORD *)(a1[21] + 4LL * v130) = *(_DWORD *)(a1[18] + 4LL * v130);
                *(_DWORD *)(a1[18] + 4LL * v130) = -1;
                sub_20E6B90(a1 + 12, &v130);
                ++v123;
                v124[v130] = v118;
              }
            }
          }
        }
        sub_20E6C60((__int64)a1, v26, v27);
        if ( v131 != v133 )
          _libc_free((unsigned __int64)v131);
      }
      --v27;
      if ( v26 == a3 )
      {
        if ( v124 )
          goto LABEL_42;
        goto LABEL_43;
      }
    }
  }
  v25 = 4LL * *(unsigned int *)(v24 + 16);
  v124 = (_DWORD *)sub_22077B0(v25);
  v115 = (char *)&v124[v25 / 4];
  memset(v124, 0, v25);
  v26 = a4;
  v27 = a5 - 1;
  if ( a3 != a4 )
    goto LABEL_24;
  v123 = 0;
LABEL_42:
  j_j___libc_free_0(v124, v115 - (char *)v124);
LABEL_43:
  j___libc_free_0(v135);
  return v123;
}
