// Function: sub_2A4E920
// Address: 0x2a4e920
//
__int64 __fastcall sub_2A4E920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 *v7; // rdx
  __int64 i; // rax
  __int64 v9; // r14
  _QWORD *v10; // rax
  char v11; // dl
  char v12; // al
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 *v15; // r13
  __int64 v16; // r15
  _BYTE *v17; // r13
  unsigned int v18; // esi
  __int64 v19; // r9
  __int64 v20; // r8
  int v21; // r11d
  _QWORD *v22; // rdx
  unsigned int v23; // edi
  _QWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rbx
  int v27; // r12d
  __int64 v28; // rax
  int v29; // r12d
  __int64 v30; // r15
  const char *v31; // rsi
  const char **v32; // r12
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  __int64 v35; // rcx
  int v36; // eax
  __int64 v37; // rdi
  int v38; // r12d
  __int64 v39; // r11
  unsigned __int8 **v40; // rbx
  unsigned __int8 **v41; // r14
  __int64 v42; // r15
  int v43; // r10d
  __int64 *v44; // rdx
  unsigned int v45; // edi
  __int64 *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r12
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  char *v51; // rax
  unsigned __int8 *v52; // r12
  unsigned __int8 v53; // al
  __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // r12
  unsigned __int64 v57; // rdx
  char *v58; // rax
  int v59; // r10d
  _QWORD *v60; // rdx
  unsigned int v61; // ebx
  unsigned int v62; // edi
  _QWORD *v63; // rax
  __int64 v64; // rcx
  __int64 v65; // r15
  const char *v66; // rdi
  const char *v67; // r14
  const char *v68; // rbx
  int v69; // eax
  unsigned int v70; // edx
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // r12
  __int64 v75; // r13
  int v76; // eax
  __int64 v77; // rbx
  int v78; // r10d
  _QWORD *v79; // rdx
  unsigned int v80; // edi
  _QWORD *v81; // rax
  __int64 v82; // rcx
  __int64 v83; // r12
  int v85; // ecx
  int v86; // eax
  unsigned int v87; // eax
  int v88; // r10d
  unsigned int v89; // r13d
  _BYTE *v90; // rsi
  unsigned int v91; // ecx
  __int64 v92; // rdi
  int v93; // ebx
  _QWORD *v94; // rdi
  __int64 v95; // r12
  int v96; // r11d
  __int64 v97; // rsi
  _QWORD *v98; // rdi
  unsigned int v99; // ebx
  int v100; // r10d
  __int64 v101; // rsi
  int v102; // eax
  unsigned int v103; // ecx
  __int64 v104; // r8
  int v105; // edi
  _QWORD *v106; // rsi
  _QWORD *v107; // rcx
  unsigned int v108; // r12d
  int v109; // esi
  __int64 v110; // rdi
  unsigned int v111; // r10d
  __int64 v113; // [rsp+20h] [rbp-3B0h]
  __int64 v114; // [rsp+20h] [rbp-3B0h]
  __int64 v115; // [rsp+20h] [rbp-3B0h]
  __int64 v116; // [rsp+20h] [rbp-3B0h]
  unsigned __int64 v117; // [rsp+28h] [rbp-3A8h]
  _BYTE *v118; // [rsp+38h] [rbp-398h]
  __int64 v119; // [rsp+40h] [rbp-390h] BYREF
  __int64 v120; // [rsp+48h] [rbp-388h]
  __int64 v121; // [rsp+50h] [rbp-380h]
  unsigned int v122; // [rsp+58h] [rbp-378h]
  _BYTE *v123; // [rsp+60h] [rbp-370h] BYREF
  __int64 v124; // [rsp+68h] [rbp-368h]
  _BYTE v125[256]; // [rsp+70h] [rbp-360h] BYREF
  __int64 *v126; // [rsp+170h] [rbp-260h] BYREF
  __int64 v127; // [rsp+178h] [rbp-258h]
  _QWORD v128[32]; // [rsp+180h] [rbp-250h] BYREF
  char *v129; // [rsp+280h] [rbp-150h] BYREF
  unsigned __int64 v130; // [rsp+288h] [rbp-148h]
  __int64 v131; // [rsp+290h] [rbp-140h] BYREF
  int v132; // [rsp+298h] [rbp-138h]
  unsigned __int8 v133; // [rsp+29Ch] [rbp-134h]
  _BYTE v134[304]; // [rsp+2A0h] [rbp-130h] BYREF

  v6 = 1;
  v7 = v128;
  v130 = (unsigned __int64)v134;
  v123 = v125;
  v124 = 0x2000000000LL;
  i = *(_QWORD *)(a1 - 64);
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v126 = v128;
  v129 = 0;
  v131 = 32;
  v132 = 0;
  v133 = 1;
  v128[0] = i;
  v127 = 0x2000000001LL;
  LODWORD(i) = 1;
  while ( 1 )
  {
    v9 = v7[(unsigned int)i - 1];
    LODWORD(v127) = i - 1;
    if ( !(_BYTE)v6 )
      goto LABEL_9;
    v10 = (_QWORD *)v130;
    v7 = (__int64 *)(v130 + 8LL * HIDWORD(v131));
    if ( (__int64 *)v130 != v7 )
    {
      while ( v9 != *v10 )
      {
        if ( v7 == ++v10 )
          goto LABEL_50;
      }
LABEL_7:
      LODWORD(i) = v127;
      if ( !(_DWORD)v127 )
        break;
      goto LABEL_8;
    }
LABEL_50:
    if ( HIDWORD(v131) < (unsigned int)v131 )
    {
      ++HIDWORD(v131);
      *v7 = v9;
      v6 = v133;
      ++v129;
    }
    else
    {
LABEL_9:
      sub_C8CC70((__int64)&v129, v9, (__int64)v7, v6, a5, a6);
      v6 = v133;
      if ( !v11 )
        goto LABEL_7;
    }
    v12 = *(_BYTE *)v9;
    if ( *(_BYTE *)v9 <= 0x1Cu )
      goto LABEL_244;
    if ( v12 == 61 )
      goto LABEL_7;
    if ( v12 != 84 )
LABEL_244:
      BUG();
    v13 = (unsigned int)v124;
    v14 = (unsigned int)v124 + 1LL;
    if ( v14 > HIDWORD(v124) )
    {
      sub_C8D5F0((__int64)&v123, v125, v14, 8u, a5, a6);
      v13 = (unsigned int)v124;
    }
    *(_QWORD *)&v123[8 * v13] = v9;
    LODWORD(v124) = v124 + 1;
    if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
    {
      v15 = *(__int64 **)(v9 - 8);
      v9 = (__int64)&v15[4 * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)];
    }
    else
    {
      v15 = (__int64 *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
    }
    for ( i = (unsigned int)v127; (__int64 *)v9 != v15; LODWORD(v127) = v127 + 1 )
    {
      v16 = *v15;
      if ( i + 1 > (unsigned __int64)HIDWORD(v127) )
      {
        sub_C8D5F0((__int64)&v126, v128, i + 1, 8u, a5, a6);
        i = (unsigned int)v127;
      }
      v15 += 4;
      v126[i] = v16;
      i = (unsigned int)(v127 + 1);
    }
    v6 = v133;
    if ( !(_DWORD)i )
      break;
LABEL_8:
    v7 = v126;
  }
  if ( !(_BYTE)v6 )
    _libc_free(v130);
  if ( v126 != v128 )
    _libc_free((unsigned __int64)v126);
  v117 = (unsigned __int64)v123;
  v17 = &v123[8 * (unsigned int)v124];
  if ( v123 == v17 )
    goto LABEL_101;
  v113 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
  while ( 2 )
  {
    while ( 2 )
    {
      v26 = *((_QWORD *)v17 - 1);
      v134[1] = 1;
      v129 = "NewPHI";
      v134[0] = 3;
      v27 = *(_DWORD *)(v26 + 4);
      v28 = sub_BD2DA0(80);
      v29 = v27 & 0x7FFFFFF;
      v30 = v28;
      if ( v28 )
      {
        sub_B44260(v28, v113, 55, 0x8000000u, v26 + 24, 0);
        *(_DWORD *)(v30 + 72) = v29;
        sub_BD6B50((unsigned __int8 *)v30, (const char **)&v129);
        sub_BD2A10(v30, *(_DWORD *)(v30 + 72), 1);
      }
      v31 = *(const char **)(v26 + 48);
      v32 = (const char **)(v30 + 48);
      v129 = (char *)v31;
      if ( !v31 )
      {
        if ( v32 != (const char **)&v129 )
        {
          v33 = *(_QWORD *)(v30 + 48);
          if ( v33 )
          {
LABEL_39:
            sub_B91220(v30 + 48, v33);
            goto LABEL_40;
          }
        }
        goto LABEL_31;
      }
      sub_B96E90((__int64)&v129, (__int64)v31, 1);
      if ( v32 == (const char **)&v129 )
      {
        if ( v129 )
          sub_B91220((__int64)&v129, (__int64)v129);
        goto LABEL_31;
      }
      v33 = *(_QWORD *)(v30 + 48);
      if ( v33 )
        goto LABEL_39;
LABEL_40:
      v34 = (unsigned __int8 *)v129;
      *(_QWORD *)(v30 + 48) = v129;
      if ( !v34 )
      {
LABEL_31:
        v18 = v122;
        if ( !v122 )
          break;
        goto LABEL_32;
      }
      sub_B976B0((__int64)&v129, v34, v30 + 48);
      v18 = v122;
      if ( v122 )
      {
LABEL_32:
        v19 = v18 - 1;
        v20 = v120;
        v21 = 1;
        v22 = 0;
        v23 = v19 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v24 = (_QWORD *)(v120 + 16LL * v23);
        v25 = *v24;
        if ( v26 == *v24 )
          goto LABEL_33;
        while ( 1 )
        {
          if ( v25 == -4096 )
          {
            if ( !v22 )
              v22 = v24;
            ++v119;
            v36 = v121 + 1;
            if ( 4 * ((int)v121 + 1) >= 3 * v18 )
              goto LABEL_43;
            if ( v18 - HIDWORD(v121) - v36 <= v18 >> 3 )
            {
              sub_2A4E740((__int64)&v119, v18);
              if ( !v122 )
              {
LABEL_245:
                LODWORD(v121) = v121 + 1;
                BUG();
              }
              v20 = v120;
              v94 = 0;
              LODWORD(v95) = (v122 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
              v96 = 1;
              v36 = v121 + 1;
              v22 = (_QWORD *)(v120 + 16LL * (unsigned int)v95);
              v97 = *v22;
              if ( v26 != *v22 )
              {
                while ( v97 != -4096 )
                {
                  if ( !v94 && v97 == -8192 )
                    v94 = v22;
                  v19 = (unsigned int)(v96 + 1);
                  v95 = (v122 - 1) & ((_DWORD)v95 + v96);
                  v22 = (_QWORD *)(v120 + 16 * v95);
                  v97 = *v22;
                  if ( v26 == *v22 )
                    goto LABEL_59;
                  ++v96;
                }
                if ( v94 )
                  v22 = v94;
              }
            }
            goto LABEL_59;
          }
          if ( v25 != -8192 || v22 )
            v24 = v22;
          v23 = v19 & (v21 + v23);
          v25 = *(_QWORD *)(v120 + 16LL * v23);
          if ( v26 == v25 )
            break;
          ++v21;
          v22 = v24;
          v24 = (_QWORD *)(v120 + 16LL * v23);
        }
        v24 = (_QWORD *)(v120 + 16LL * v23);
LABEL_33:
        v17 -= 8;
        v24[1] = v30;
        if ( (_BYTE *)v117 == v17 )
          goto LABEL_62;
        continue;
      }
      break;
    }
    ++v119;
LABEL_43:
    sub_2A4E740((__int64)&v119, 2 * v18);
    if ( !v122 )
      goto LABEL_245;
    v19 = v120;
    LODWORD(v35) = (v122 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v36 = v121 + 1;
    v22 = (_QWORD *)(v120 + 16LL * (unsigned int)v35);
    v37 = *v22;
    if ( v26 != *v22 )
    {
      v38 = 1;
      v20 = 0;
      while ( v37 != -4096 )
      {
        if ( !v20 && v37 == -8192 )
          v20 = (__int64)v22;
        v35 = (v122 - 1) & ((_DWORD)v35 + v38);
        v22 = (_QWORD *)(v120 + 16 * v35);
        v37 = *v22;
        if ( v26 == *v22 )
          goto LABEL_59;
        ++v38;
      }
      if ( v20 )
        v22 = (_QWORD *)v20;
    }
LABEL_59:
    LODWORD(v121) = v36;
    if ( *v22 != -4096 )
      --HIDWORD(v121);
    *v22 = v26;
    v17 -= 8;
    v22[1] = 0;
    v22[1] = v30;
    if ( (_BYTE *)v117 != v17 )
      continue;
    break;
  }
LABEL_62:
  v118 = &v123[8 * (unsigned int)v124];
  if ( (_BYTE *)v117 != v118 )
  {
    while ( 1 )
    {
      v39 = *((_QWORD *)v118 - 1);
      v129 = (char *)&v131;
      v130 = 0x400000000LL;
      if ( (*(_BYTE *)(v39 + 7) & 0x40) != 0 )
      {
        v40 = *(unsigned __int8 ***)(v39 - 8);
        v41 = &v40[4 * (*(_DWORD *)(v39 + 4) & 0x7FFFFFF)];
      }
      else
      {
        v41 = (unsigned __int8 **)v39;
        v40 = (unsigned __int8 **)(v39 - 32LL * (*(_DWORD *)(v39 + 4) & 0x7FFFFFF));
      }
      if ( v40 != v41 )
      {
        v114 = v39;
        while ( 1 )
        {
          while ( 1 )
          {
            v52 = *v40;
            v53 = **v40;
            if ( v53 <= 0x1Cu )
LABEL_241:
              BUG();
            if ( v53 != 61 )
              break;
            v54 = (unsigned int)v130;
            v55 = *((_QWORD *)v52 + 5);
            v56 = *((_QWORD *)v52 - 4);
            v57 = (unsigned int)v130 + 1LL;
            if ( v57 > HIDWORD(v130) )
            {
              sub_C8D5F0((__int64)&v129, &v131, v57, 0x10u, v20, v19);
              v54 = (unsigned int)v130;
            }
            v40 += 4;
            v58 = &v129[16 * v54];
            *(_QWORD *)v58 = v56;
            *((_QWORD *)v58 + 1) = v55;
            LODWORD(v130) = v130 + 1;
            if ( v41 == v40 )
              goto LABEL_79;
          }
          if ( v53 != 84 )
            goto LABEL_241;
          v42 = *((_QWORD *)v52 + 5);
          v19 = v120;
          if ( !v122 )
            break;
          v20 = v122 - 1;
          v43 = 1;
          v44 = 0;
          v45 = v20 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
          v46 = (__int64 *)(v120 + 16LL * v45);
          v47 = *v46;
          if ( v52 != (unsigned __int8 *)*v46 )
          {
            while ( v47 != -4096 )
            {
              if ( v47 == -8192 && !v44 )
                v44 = v46;
              v45 = v20 & (v43 + v45);
              v46 = (__int64 *)(v120 + 16LL * v45);
              v47 = *v46;
              if ( v52 == (unsigned __int8 *)*v46 )
                goto LABEL_70;
              ++v43;
            }
            if ( !v44 )
              v44 = v46;
            ++v119;
            v85 = v121 + 1;
            if ( 4 * ((int)v121 + 1) < 3 * v122 )
            {
              if ( v122 - HIDWORD(v121) - v85 <= v122 >> 3 )
              {
                sub_2A4E740((__int64)&v119, v122);
                if ( !v122 )
                {
LABEL_240:
                  LODWORD(v121) = v121 + 1;
                  BUG();
                }
                v20 = 0;
                v89 = (v122 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
                v19 = 1;
                v85 = v121 + 1;
                v44 = (__int64 *)(v120 + 16LL * v89);
                v90 = (_BYTE *)*v44;
                if ( v52 != (unsigned __int8 *)*v44 )
                {
                  while ( v90 != (_BYTE *)-4096LL )
                  {
                    if ( !v20 && v90 == (_BYTE *)-8192LL )
                      v20 = (__int64)v44;
                    v111 = v19 + 1;
                    v19 = (v122 - 1) & (v89 + (_DWORD)v19);
                    v89 = v19;
                    v44 = (__int64 *)(v120 + 16LL * (unsigned int)v19);
                    v90 = (_BYTE *)*v44;
                    if ( v52 == (unsigned __int8 *)*v44 )
                      goto LABEL_117;
                    v19 = v111;
                  }
                  if ( v20 )
                    v44 = (__int64 *)v20;
                }
              }
              goto LABEL_117;
            }
LABEL_135:
            sub_2A4E740((__int64)&v119, 2 * v122);
            if ( !v122 )
              goto LABEL_240;
            v87 = (v122 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
            v85 = v121 + 1;
            v44 = (__int64 *)(v120 + 16LL * v87);
            v20 = *v44;
            if ( v52 != (unsigned __int8 *)*v44 )
            {
              v88 = 1;
              v19 = 0;
              while ( v20 != -4096 )
              {
                if ( !v19 && v20 == -8192 )
                  v19 = (__int64)v44;
                v87 = (v122 - 1) & (v88 + v87);
                v44 = (__int64 *)(v120 + 16LL * v87);
                v20 = *v44;
                if ( v52 == (unsigned __int8 *)*v44 )
                  goto LABEL_117;
                ++v88;
              }
              if ( v19 )
                v44 = (__int64 *)v19;
            }
LABEL_117:
            LODWORD(v121) = v85;
            if ( *v44 != -4096 )
              --HIDWORD(v121);
            *v44 = (__int64)v52;
            v48 = 0;
            v44[1] = 0;
            goto LABEL_71;
          }
LABEL_70:
          v48 = v46[1];
LABEL_71:
          v49 = (unsigned int)v130;
          v50 = (unsigned int)v130 + 1LL;
          if ( v50 > HIDWORD(v130) )
          {
            sub_C8D5F0((__int64)&v129, &v131, v50, 0x10u, v20, v19);
            v49 = (unsigned int)v130;
          }
          v40 += 4;
          v51 = &v129[16 * v49];
          *(_QWORD *)v51 = v48;
          *((_QWORD *)v51 + 1) = v42;
          LODWORD(v130) = v130 + 1;
          if ( v41 == v40 )
          {
LABEL_79:
            v39 = v114;
            goto LABEL_80;
          }
        }
        ++v119;
        goto LABEL_135;
      }
LABEL_80:
      if ( !v122 )
        break;
      v19 = v122 - 1;
      v59 = 1;
      v20 = v120;
      v60 = 0;
      v61 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
      v62 = v19 & v61;
      v63 = (_QWORD *)(v120 + 16LL * ((unsigned int)v19 & v61));
      v64 = *v63;
      if ( v39 != *v63 )
      {
        while ( v64 != -4096 )
        {
          if ( v64 == -8192 && !v60 )
            v60 = v63;
          v62 = v19 & (v59 + v62);
          v63 = (_QWORD *)(v120 + 16LL * v62);
          v64 = *v63;
          if ( v39 == *v63 )
            goto LABEL_82;
          ++v59;
        }
        if ( !v60 )
          v60 = v63;
        ++v119;
        v86 = v121 + 1;
        if ( 4 * ((int)v121 + 1) < 3 * v122 )
        {
          if ( v122 - HIDWORD(v121) - v86 <= v122 >> 3 )
          {
            v116 = v39;
            sub_2A4E740((__int64)&v119, v122);
            if ( !v122 )
            {
LABEL_242:
              LODWORD(v121) = v121 + 1;
              BUG();
            }
            v20 = v120;
            v98 = 0;
            v99 = (v122 - 1) & v61;
            v39 = v116;
            v100 = 1;
            v86 = v121 + 1;
            v60 = (_QWORD *)(v120 + 16LL * v99);
            v101 = *v60;
            if ( v116 != *v60 )
            {
              while ( v101 != -4096 )
              {
                if ( v101 == -8192 && !v98 )
                  v98 = v60;
                v19 = (unsigned int)(v100 + 1);
                v99 = (v122 - 1) & (v100 + v99);
                v60 = (_QWORD *)(v120 + 16LL * v99);
                v101 = *v60;
                if ( v116 == *v60 )
                  goto LABEL_130;
                ++v100;
              }
              if ( v98 )
                v60 = v98;
            }
          }
          goto LABEL_130;
        }
LABEL_149:
        v115 = v39;
        sub_2A4E740((__int64)&v119, 2 * v122);
        if ( !v122 )
          goto LABEL_242;
        v39 = v115;
        v19 = v120;
        v91 = (v122 - 1) & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
        v86 = v121 + 1;
        v60 = (_QWORD *)(v120 + 16LL * v91);
        v92 = *v60;
        if ( v115 != *v60 )
        {
          v93 = 1;
          v20 = 0;
          while ( v92 != -4096 )
          {
            if ( v92 == -8192 && !v20 )
              v20 = (__int64)v60;
            v91 = (v122 - 1) & (v93 + v91);
            v60 = (_QWORD *)(v120 + 16LL * v91);
            v92 = *v60;
            if ( v115 == *v60 )
              goto LABEL_130;
            ++v93;
          }
          if ( v20 )
            v60 = (_QWORD *)v20;
        }
LABEL_130:
        LODWORD(v121) = v86;
        if ( *v60 != -4096 )
          --HIDWORD(v121);
        *v60 = v39;
        v65 = 0;
        v60[1] = 0;
        goto LABEL_83;
      }
LABEL_82:
      v65 = v63[1];
LABEL_83:
      v66 = v129;
      v67 = &v129[16 * (unsigned int)v130];
      if ( v67 != v129 )
      {
        v68 = v129;
        do
        {
          v74 = *(_QWORD *)v68;
          v75 = *((_QWORD *)v68 + 1);
          v76 = *(_DWORD *)(v65 + 4) & 0x7FFFFFF;
          if ( v76 == *(_DWORD *)(v65 + 72) )
          {
            sub_B48D90(v65);
            v76 = *(_DWORD *)(v65 + 4) & 0x7FFFFFF;
          }
          v69 = (v76 + 1) & 0x7FFFFFF;
          v70 = v69 | *(_DWORD *)(v65 + 4) & 0xF8000000;
          v71 = *(_QWORD *)(v65 - 8) + 32LL * (unsigned int)(v69 - 1);
          *(_DWORD *)(v65 + 4) = v70;
          if ( *(_QWORD *)v71 )
          {
            v72 = *(_QWORD *)(v71 + 8);
            **(_QWORD **)(v71 + 16) = v72;
            if ( v72 )
              *(_QWORD *)(v72 + 16) = *(_QWORD *)(v71 + 16);
          }
          *(_QWORD *)v71 = v74;
          if ( v74 )
          {
            v73 = *(_QWORD *)(v74 + 16);
            *(_QWORD *)(v71 + 8) = v73;
            if ( v73 )
              *(_QWORD *)(v73 + 16) = v71 + 8;
            *(_QWORD *)(v71 + 16) = v74 + 16;
            *(_QWORD *)(v74 + 16) = v71;
          }
          v68 += 16;
          *(_QWORD *)(*(_QWORD *)(v65 - 8)
                    + 32LL * *(unsigned int *)(v65 + 72)
                    + 8LL * ((*(_DWORD *)(v65 + 4) & 0x7FFFFFFu) - 1)) = v75;
        }
        while ( v67 != v68 );
        v66 = v129;
      }
      if ( v66 != (const char *)&v131 )
        _libc_free((unsigned __int64)v66);
      v118 -= 8;
      if ( (_BYTE *)v117 == v118 )
        goto LABEL_101;
    }
    ++v119;
    goto LABEL_149;
  }
LABEL_101:
  v77 = *(_QWORD *)(a1 - 64);
  if ( !v122 )
  {
    ++v119;
    goto LABEL_187;
  }
  v78 = 1;
  v79 = 0;
  v80 = (v122 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
  v81 = (_QWORD *)(v120 + 16LL * v80);
  v82 = *v81;
  if ( v77 == *v81 )
  {
LABEL_103:
    v83 = v81[1];
    goto LABEL_104;
  }
  while ( v82 != -4096 )
  {
    if ( v82 == -8192 && !v79 )
      v79 = v81;
    v80 = (v122 - 1) & (v78 + v80);
    v81 = (_QWORD *)(v120 + 16LL * v80);
    v82 = *v81;
    if ( v77 == *v81 )
      goto LABEL_103;
    ++v78;
  }
  if ( !v79 )
    v79 = v81;
  ++v119;
  v102 = v121 + 1;
  if ( 4 * ((int)v121 + 1) >= 3 * v122 )
  {
LABEL_187:
    sub_2A4E740((__int64)&v119, 2 * v122);
    if ( v122 )
    {
      v103 = (v122 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
      v102 = v121 + 1;
      v79 = (_QWORD *)(v120 + 16LL * v103);
      v104 = *v79;
      if ( v77 != *v79 )
      {
        v105 = 1;
        v106 = 0;
        while ( v104 != -4096 )
        {
          if ( !v106 && v104 == -8192 )
            v106 = v79;
          v103 = (v122 - 1) & (v105 + v103);
          v79 = (_QWORD *)(v120 + 16LL * v103);
          v104 = *v79;
          if ( v77 == *v79 )
            goto LABEL_182;
          ++v105;
        }
        if ( v106 )
          v79 = v106;
      }
      goto LABEL_182;
    }
    goto LABEL_243;
  }
  if ( v122 - HIDWORD(v121) - v102 <= v122 >> 3 )
  {
    sub_2A4E740((__int64)&v119, v122);
    if ( v122 )
    {
      v107 = 0;
      v108 = (v122 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
      v109 = 1;
      v102 = v121 + 1;
      v79 = (_QWORD *)(v120 + 16LL * v108);
      v110 = *v79;
      if ( v77 != *v79 )
      {
        while ( v110 != -4096 )
        {
          if ( !v107 && v110 == -8192 )
            v107 = v79;
          v108 = (v122 - 1) & (v109 + v108);
          v79 = (_QWORD *)(v120 + 16LL * v108);
          v110 = *v79;
          if ( v77 == *v79 )
            goto LABEL_182;
          ++v109;
        }
        if ( v107 )
          v79 = v107;
      }
      goto LABEL_182;
    }
LABEL_243:
    LODWORD(v121) = v121 + 1;
    BUG();
  }
LABEL_182:
  LODWORD(v121) = v102;
  if ( *v79 != -4096 )
    --HIDWORD(v121);
  *v79 = v77;
  v83 = 0;
  v79[1] = 0;
LABEL_104:
  if ( v123 != v125 )
    _libc_free((unsigned __int64)v123);
  sub_C7D6A0(v120, 16LL * v122, 8);
  return v83;
}
