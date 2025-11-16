// Function: sub_2C89F30
// Address: 0x2c89f30
//
bool __fastcall sub_2C89F30(unsigned __int8 ***a1, __int64 a2)
{
  unsigned __int8 ***v2; // rbx
  int v3; // r13d
  unsigned __int8 **v4; // r15
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int8 *v7; // r14
  unsigned __int8 v8; // al
  bool result; // al
  _DWORD *v10; // r14
  __int64 v11; // r8
  _BYTE *v12; // r15
  _DWORD *v13; // r10
  char v14; // al
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int8 *v20; // rax
  char v21; // al
  int v22; // r15d
  unsigned int v23; // edi
  char v24; // si
  unsigned int v25; // ecx
  unsigned int i; // edx
  unsigned __int8 *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // eax
  _BYTE *v31; // rax
  _BYTE *v32; // rdx
  __int64 v33; // r13
  unsigned __int8 ***v34; // r12
  _DWORD *v35; // rdx
  _BYTE *v36; // r9
  bool v37; // dl
  char v38; // bl
  int v39; // eax
  _BYTE *v40; // r9
  bool v41; // dl
  int v42; // esi
  char v43; // r14
  bool v44; // r13
  int v45; // edx
  unsigned __int8 ***v46; // r15
  _BYTE *v47; // r12
  char v48; // r9
  unsigned int v49; // ebx
  unsigned __int8 *v50; // rax
  unsigned __int8 *v51; // rax
  __int64 v52; // rdi
  char v53; // al
  char v54; // al
  char v55; // r10
  bool v56; // dl
  char v57; // al
  int v58; // r15d
  __int64 *v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  bool v62; // al
  __int64 v63; // r15
  __int64 v64; // r14
  unsigned int v65; // esi
  int v66; // r8d
  _QWORD *v67; // rcx
  unsigned int k; // edi
  _QWORD *v69; // rax
  __int64 v70; // r9
  int v71; // edi
  unsigned int ii; // esi
  unsigned __int8 *v73; // rdi
  __int64 v74; // r8
  __int64 v75; // rax
  unsigned __int8 *v76; // rax
  __int64 v77; // rdx
  bool v78; // di
  char v79; // r8
  __int64 v80; // rcx
  __int64 v81; // rax
  bool v82; // al
  int v83; // esi
  int v84; // esi
  int v85; // edi
  _QWORD *v86; // rcx
  unsigned int m; // edx
  __int64 v88; // r8
  unsigned int v89; // r10d
  __int64 jj; // rcx
  _BYTE *v91; // r14
  unsigned int v92; // esi
  _QWORD *v93; // r15
  int v94; // edi
  unsigned int kk; // edx
  _QWORD *v96; // rax
  _BYTE *v97; // r8
  int v98; // edx
  int v99; // ecx
  int v100; // ecx
  int v101; // esi
  _QWORD *v102; // r11
  unsigned int mm; // edx
  __int64 v104; // rdi
  int v105; // ecx
  int v106; // ecx
  int v107; // esi
  unsigned int nn; // edx
  __int64 v109; // rdi
  unsigned int v110; // edx
  unsigned int v111; // edx
  unsigned int v112; // edx
  unsigned int v113; // edi
  int v114; // esi
  int v115; // esi
  int v116; // edi
  unsigned int n; // edx
  __int64 v118; // r9
  unsigned int v119; // edx
  __int64 v120; // rax
  __int64 v121; // rdx
  unsigned int v122; // edx
  int v123; // eax
  unsigned int j; // eax
  unsigned __int8 *v125; // rcx
  __int64 v126; // rcx
  int v127; // [rsp+4h] [rbp-9Ch]
  int v128; // [rsp+8h] [rbp-98h]
  int v129; // [rsp+8h] [rbp-98h]
  _DWORD *v130; // [rsp+8h] [rbp-98h]
  _DWORD *v131; // [rsp+10h] [rbp-90h]
  unsigned __int8 v132; // [rsp+10h] [rbp-90h]
  unsigned __int8 v133; // [rsp+18h] [rbp-88h]
  bool v134; // [rsp+18h] [rbp-88h]
  __int64 v135; // [rsp+18h] [rbp-88h]
  __int64 v136; // [rsp+18h] [rbp-88h]
  _BYTE *v137; // [rsp+20h] [rbp-80h]
  _BYTE *v138; // [rsp+20h] [rbp-80h]
  __int64 v139; // [rsp+28h] [rbp-78h]
  __int64 v140; // [rsp+38h] [rbp-68h]
  __int64 v141; // [rsp+38h] [rbp-68h]
  unsigned __int8 **v143; // [rsp+48h] [rbp-58h]
  int v144; // [rsp+48h] [rbp-58h]
  bool v145; // [rsp+48h] [rbp-58h]
  unsigned __int8 **v146; // [rsp+48h] [rbp-58h]
  unsigned int v147; // [rsp+48h] [rbp-58h]
  _QWORD *v148; // [rsp+50h] [rbp-50h] BYREF
  __int64 v149; // [rsp+58h] [rbp-48h]
  _QWORD v150[8]; // [rsp+60h] [rbp-40h] BYREF

  v2 = a1;
  v3 = *((_DWORD *)a1 + 2);
  v143 = *a1;
  if ( v3 )
  {
    v4 = *a1;
    v5 = (__int64)&(*a1)[(unsigned int)(v3 - 1) + 1];
    do
    {
      while ( 1 )
      {
        v7 = *v4;
        v8 = **v4;
        if ( v8 == 84 )
          return 0;
        if ( v8 != 85 )
          break;
        v6 = *((_QWORD *)v7 - 4);
        if ( *(_BYTE *)v6 )
        {
          if ( *(_BYTE *)v6 == 25 && *(_BYTE *)(v6 + 96) )
            return 0;
        }
        else if ( *(_QWORD *)(v6 + 24) == *((_QWORD *)v7 + 10) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
        {
          v30 = *(_DWORD *)(v6 + 36);
          if ( v30 == 9067 )
            return 0;
          if ( v30 == 9145 )
            return 0;
        }
LABEL_5:
        if ( (unsigned __int8 **)v5 == ++v4 )
          goto LABEL_19;
      }
      if ( v8 == 61 || v8 == 62 )
      {
        result = sub_B46500(*v4);
        if ( result )
          return 0;
        if ( (v7[2] & 1) != 0 )
          return result;
        goto LABEL_5;
      }
      if ( v8 == 93 || v8 == 66 )
        return 0;
      ++v4;
    }
    while ( (unsigned __int8 **)v5 != v4 );
  }
LABEL_19:
  v10 = *v143;
  v11 = **v143;
  v144 = *((_DWORD *)*v143 + 1) & 0x7FFFFFF;
  v140 = *((_QWORD *)v10 + 1);
  if ( (_BYTE)v11 != 85 )
  {
    v13 = 0;
    goto LABEL_22;
  }
  v12 = (_BYTE *)*((_QWORD *)v10 - 4);
  if ( *v12 )
    return 0;
  v13 = v10;
  if ( (unsigned int)qword_5011A28 <= 1 )
  {
    if ( !(unsigned __int8)sub_B2F6B0(*((_QWORD *)v10 - 4))
      && !(unsigned __int8)sub_B2D610((__int64)v12, 27)
      && ((_DWORD)qword_5011A28 || (unsigned __int8)sub_B2D610((__int64)v12, 41)) )
    {
      v62 = sub_B2DCC0((__int64)v12);
      v11 = 85;
      if ( v62 )
      {
        v3 = *((_DWORD *)a1 + 2);
        v13 = v10;
        goto LABEL_22;
      }
      if ( (unsigned __int8)sub_B2DCE0((__int64)v12) )
      {
        v3 = *((_DWORD *)a1 + 2);
        v11 = 85;
        v13 = v10;
        goto LABEL_22;
      }
    }
    return 0;
  }
LABEL_22:
  v148 = v150;
  v149 = 0x200000000LL;
  v14 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 != 61 && v14 != 62 )
  {
    v139 = 0;
    if ( v3 == 1 )
    {
      if ( !v144 )
      {
        if ( v14 == 94 )
          goto LABEL_43;
        goto LABEL_111;
      }
      goto LABEL_71;
    }
    goto LABEL_25;
  }
  v139 = *((_QWORD *)v10 - 4);
  if ( !v139 )
  {
    if ( v3 == 1 )
    {
      if ( !v144 )
        goto LABEL_111;
      goto LABEL_71;
    }
    goto LABEL_25;
  }
  LODWORD(v149) = 1;
  v150[0] = v139;
  if ( v3 != 1 )
  {
LABEL_25:
    v15 = 1;
    do
    {
      v16 = (unsigned int)v15;
      v17 = (__int64)(*v2)[(unsigned int)v15];
      if ( (*(_DWORD *)(v17 + 4) & 0x7FFFFFF) != v144 || v140 != *(_QWORD *)(v17 + 8) || (_BYTE)v11 != *(_BYTE *)v17 )
        goto LABEL_50;
      if ( (_BYTE)v11 == 61 || (_BYTE)v11 == 62 )
      {
        v18 = (unsigned int)v149;
        v19 = *(_QWORD *)(v17 - 32);
        if ( (unsigned __int64)(unsigned int)v149 + 1 > HIDWORD(v149) )
        {
          v127 = v15;
          v130 = v13;
          v132 = v11;
          v136 = *(_QWORD *)(v17 - 32);
          sub_C8D5F0((__int64)&v148, v150, (unsigned int)v149 + 1LL, 8u, v11, v15);
          v18 = (unsigned int)v149;
          LODWORD(v15) = v127;
          v13 = v130;
          v11 = v132;
          v19 = v136;
        }
        v148[v18] = v19;
        LODWORD(v149) = v149 + 1;
      }
      if ( v13 )
      {
        v20 = (*v2)[v16];
        if ( *v20 != 85 || *((_QWORD *)v13 - 4) != *((_QWORD *)v20 - 4) )
          goto LABEL_50;
      }
      v21 = *(_BYTE *)v10;
      if ( (unsigned __int8)(*(_BYTE *)v10 - 82) <= 1u )
      {
        if ( (unsigned __int8)(*(_BYTE *)v17 - 82) > 1u )
          goto LABEL_50;
        v22 = *((_WORD *)v10 + 1) & 0x3F;
        v23 = *(_WORD *)(v17 + 2) & 0x3F;
        if ( v22 == v23 )
          goto LABEL_39;
        v128 = v15;
        v131 = v13;
        v133 = v11;
        if ( v22 != (unsigned int)sub_B52F50(v23) )
          goto LABEL_50;
        sub_B53070(v17);
        LODWORD(v15) = v128;
        v13 = v131;
        v11 = v133;
        *(_WORD *)(v17 + 2) = *(_WORD *)(v17 + 2) & 0xFFC0 | *((_WORD *)v10 + 1) & 0x3F;
        v21 = *(_BYTE *)v10;
      }
      if ( v21 == 90 )
      {
        if ( *(_BYTE *)v17 != 90 )
          goto LABEL_50;
        v31 = (_BYTE *)*((_QWORD *)v10 - 4);
        if ( *v31 <= 0x15u )
        {
          v32 = *(_BYTE **)(v17 - 32);
          if ( *v32 <= 0x15u && v31 != v32 )
            goto LABEL_50;
        }
      }
      else if ( v21 == 63 && (*(_BYTE *)v17 != 63 || *((_QWORD *)v10 + 9) != *(_QWORD *)(v17 + 72)) )
      {
        goto LABEL_50;
      }
LABEL_39:
      v15 = (unsigned int)(v15 + 1);
    }
    while ( (_DWORD)v15 != v3 );
  }
  if ( !v144 )
    goto LABEL_41;
LABEL_71:
  v33 = 0;
  v34 = v2;
  do
  {
    if ( (*((_BYTE *)v10 + 7) & 0x40) != 0 )
      v35 = (_DWORD *)*((_QWORD *)v10 - 1);
    else
      v35 = &v10[-8 * (v10[1] & 0x7FFFFFF)];
    v36 = *(_BYTE **)&v35[8 * v33];
    v141 = 32 * v33;
    if ( *v36 <= 0x15u )
    {
      v38 = 1;
      v37 = 0;
    }
    else
    {
      v37 = *v36 == 60;
      v38 = 0;
    }
    v134 = v37;
    v137 = v36;
    v39 = sub_BD3960((__int64)v36);
    v40 = v137;
    v41 = v134;
    v42 = v39;
    if ( *((_DWORD *)v34 + 2) == 1 )
      goto LABEL_145;
    v138 = v10;
    v135 = v33;
    v43 = 1;
    v44 = v41;
    v45 = *((_DWORD *)v34 + 2);
    v46 = v34;
    v47 = v40;
    v48 = v38;
    v49 = 1;
    do
    {
      v50 = (*v46)[v49];
      if ( (v50[7] & 0x40) != 0 )
        v51 = (unsigned __int8 *)*((_QWORD *)v50 - 1);
      else
        v51 = &v50[-32 * (*((_DWORD *)v50 + 1) & 0x7FFFFFF)];
      v52 = *(_QWORD *)&v51[v141];
      if ( v47 != (_BYTE *)v52 )
      {
        if ( *(_QWORD *)(v52 + 8) != *((_QWORD *)v47 + 1) )
          goto LABEL_50;
        v43 = 0;
      }
      v53 = *(_BYTE *)v52;
      if ( *(_BYTE *)v52 <= 0x15u )
      {
        if ( v44 )
          goto LABEL_50;
      }
      else
      {
        if ( v44 )
        {
          if ( v53 != 60 )
            goto LABEL_50;
          v129 = v45;
          v54 = sub_BD3610(v52, v42);
          v45 = v129;
          if ( !v54 )
            goto LABEL_50;
        }
        else if ( v53 == 60 )
        {
          goto LABEL_50;
        }
        v48 = 0;
      }
      ++v49;
    }
    while ( v45 != v49 );
    v55 = v43;
    v56 = v44;
    v10 = v138;
    v33 = v135;
    v34 = v46;
    if ( v55 )
      goto LABEL_145;
    v57 = *v138;
    if ( v135 != 2 )
    {
      if ( v57 != 63 )
        goto LABEL_92;
      if ( !(_DWORD)v135 )
      {
        if ( !v56 )
          goto LABEL_50;
LABEL_151:
        if ( !(_DWORD)v135 )
          goto LABEL_92;
      }
      if ( v48 )
        goto LABEL_50;
      goto LABEL_92;
    }
    if ( v57 == 92 )
      goto LABEL_50;
    if ( v57 == 63 )
      goto LABEL_151;
LABEL_92:
    v58 = *((_DWORD *)v46 + 2);
    if ( v58 )
    {
      v59 = (__int64 *)*v34;
      do
      {
        v61 = *v59;
        v60 = (*(_BYTE *)(*v59 + 7) & 0x40) != 0
            ? *(_QWORD *)(v61 - 8)
            : v61 - 32LL * (*(_DWORD *)(v61 + 4) & 0x7FFFFFF);
        if ( (unsigned __int8)(**(_BYTE **)(v60 + v141) - 23) <= 5u )
          goto LABEL_50;
        ++v59;
      }
      while ( &(*v34)[(unsigned int)(v58 - 1) + 1] != (unsigned __int8 **)v59 );
    }
LABEL_145:
    ++v33;
  }
  while ( v144 != (_DWORD)v33 );
  v2 = v34;
LABEL_41:
  v24 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 == 94 )
  {
    v3 = *((_DWORD *)v2 + 2);
LABEL_43:
    v25 = v10[20];
    for ( i = 1; v3 != i; ++i )
    {
      v27 = (*v2)[i];
      if ( v25 != *((_DWORD *)v27 + 20) )
        goto LABEL_50;
      v28 = *((_QWORD *)v27 + 9);
      if ( v25 )
      {
        v29 = 0;
        while ( *(_DWORD *)(v28 + 4 * v29) == *(_DWORD *)(*((_QWORD *)v10 + 9) + 4 * v29) )
        {
          if ( v25 <= (unsigned int)++v29 )
            goto LABEL_144;
        }
        goto LABEL_50;
      }
LABEL_144:
      ;
    }
    v24 = 94;
  }
  if ( !v139 )
    goto LABEL_239;
  v77 = 0;
  v78 = 0;
  v79 = 1;
  while ( (_DWORD)v149 != (_DWORD)v77 )
  {
    v80 = v148[v77];
    if ( v80 != v139 )
      v79 = 0;
    if ( *(_BYTE *)v80 == 63 )
    {
      v81 = *(_QWORD *)(*(_QWORD *)(v80 - 32LL * (*(_DWORD *)(v80 + 4) & 0x7FFFFFF)) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v81 + 8) - 17 <= 1 )
        v81 = **(_QWORD **)(v81 + 16);
      v82 = *(_DWORD *)(v81 + 8) >> 8 == 5 || *(_DWORD *)(v81 + 8) >> 8 == 0;
      if ( v82 )
        v78 = v82;
    }
    ++v77;
  }
  if ( v79 )
  {
LABEL_239:
    v3 = *((_DWORD *)v2 + 2);
    v146 = *v2;
    goto LABEL_112;
  }
  if ( v78 )
  {
LABEL_50:
    result = 0;
    goto LABEL_51;
  }
  if ( v24 != 62 )
    goto LABEL_239;
  v120 = *(_QWORD *)(v139 + 8);
  if ( *(_BYTE *)(v120 + 8) != 14 )
    v120 = 0;
  v121 = *((_QWORD *)v10 - 8);
  if ( *(_BYTE *)(*(_QWORD *)(v121 + 8) + 8LL) != 15 )
    goto LABEL_239;
  v3 = *((_DWORD *)v2 + 2);
  v123 = *(_DWORD *)(v120 + 8) >> 8;
  if ( v123 != 5 && v123 )
  {
LABEL_111:
    v146 = *v2;
    goto LABEL_112;
  }
  v146 = *v2;
  for ( j = 1; j != v3; ++j )
  {
    v125 = v146[j];
    if ( *v125 != 62 )
      BUG();
    v126 = *((_QWORD *)v125 - 8);
    if ( v121 != v126 || !v126 )
      goto LABEL_50;
  }
LABEL_112:
  v63 = *((_QWORD *)*v146 + 2);
  while ( 2 )
  {
    if ( v63 )
    {
      v64 = *(_QWORD *)(v63 + 24);
      if ( *(_BYTE *)v64 == 84 )
      {
        v65 = *(_DWORD *)(a2 + 24);
        if ( v65 )
        {
          v66 = 1;
          v67 = 0;
          for ( k = (v65 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4)); ; k = (v65 - 1) & v113 )
          {
            v69 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * k);
            v70 = *v69;
            if ( v64 == *v69 )
              goto LABEL_113;
            if ( v70 == -4096 )
              break;
            if ( v67 || v70 != -8192 )
              v69 = v67;
            v113 = v66 + k;
            v67 = v69;
            ++v66;
          }
          v71 = *(_DWORD *)(a2 + 16);
          if ( v67 )
            v69 = v67;
          ++*(_QWORD *)a2;
          if ( 4 * (v71 + 1) < 3 * v65 )
          {
            if ( v65 - *(_DWORD *)(a2 + 20) - (v71 + 1) <= v65 >> 3 )
            {
              sub_110B120(a2, v65);
              v83 = *(_DWORD *)(a2 + 24);
              if ( !v83 )
                goto LABEL_253;
              v84 = v83 - 1;
              v85 = 1;
              v86 = 0;
              for ( m = v84 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4)); ; m = v84 & v122 )
              {
                v69 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * m);
                v88 = *v69;
                if ( v64 == *v69 )
                  break;
                if ( v88 == -4096 )
                  goto LABEL_174;
                if ( v88 != -8192 || v86 )
                  v69 = v86;
                v122 = v85 + m;
                v86 = v69;
                ++v85;
              }
            }
            goto LABEL_124;
          }
        }
        else
        {
          ++*(_QWORD *)a2;
        }
        sub_110B120(a2, 2 * v65);
        v114 = *(_DWORD *)(a2 + 24);
        if ( !v114 )
        {
LABEL_253:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v115 = v114 - 1;
        v116 = 1;
        v86 = 0;
        for ( n = v115 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4)); ; n = v115 & v119 )
        {
          v69 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * n);
          v118 = *v69;
          if ( v64 == *v69 )
            break;
          if ( v118 == -4096 )
          {
LABEL_174:
            if ( v86 )
              v69 = v86;
            break;
          }
          if ( v118 != -8192 || v86 )
            v69 = v86;
          v119 = v116 + n;
          v86 = v69;
          ++v116;
        }
LABEL_124:
        ++*(_DWORD *)(a2 + 16);
        if ( *v69 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v69 = v64;
        for ( ii = 1; v3 != ii; ++ii )
        {
          v73 = (*v2)[ii];
          if ( (*(_DWORD *)(v64 + 4) & 0x7FFFFFF) == 0 )
            goto LABEL_50;
          v74 = *(_QWORD *)(v64 - 8);
          v75 = 0;
          while ( *((_QWORD *)v73 + 5) != *(_QWORD *)(v74 + 32LL * *(unsigned int *)(v64 + 72) + 8 * v75) )
          {
            if ( (*(_DWORD *)(v64 + 4) & 0x7FFFFFF) == (_DWORD)++v75 )
              goto LABEL_50;
          }
          v76 = *(unsigned __int8 **)(v74 + 32 * v75);
          if ( v73 != v76 || !v76 )
            goto LABEL_50;
        }
      }
LABEL_113:
      v63 = *(_QWORD *)(v63 + 8);
      continue;
    }
    break;
  }
  v89 = 1;
LABEL_178:
  if ( v3 == v89 )
  {
    result = 1;
  }
  else
  {
    for ( jj = *((_QWORD *)(*v2)[v89] + 2); ; jj = *(_QWORD *)(jj + 8) )
    {
      if ( !jj )
      {
        ++v89;
        goto LABEL_178;
      }
      v91 = *(_BYTE **)(jj + 24);
      if ( *v91 == 84 )
        break;
LABEL_180:
      ;
    }
    v92 = *(_DWORD *)(a2 + 24);
    if ( v92 )
    {
      v93 = 0;
      v94 = 1;
      v147 = ((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4);
      for ( kk = (v92 - 1) & v147; ; kk = (v92 - 1) & v112 )
      {
        v96 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * kk);
        v97 = (_BYTE *)*v96;
        if ( v91 == (_BYTE *)*v96 )
          goto LABEL_180;
        if ( v97 == (_BYTE *)-4096LL )
          break;
        if ( v93 || v97 != (_BYTE *)-8192LL )
          v96 = v93;
        v112 = v94 + kk;
        v93 = v96;
        ++v94;
      }
      if ( v93 )
        v96 = v93;
      ++*(_QWORD *)a2;
      v98 = *(_DWORD *)(a2 + 16) + 1;
      if ( 4 * v98 < 3 * v92 )
      {
        if ( v92 - (*(_DWORD *)(a2 + 20) + v98) > v92 >> 3 )
          goto LABEL_191;
        sub_110B120(a2, v92);
        v99 = *(_DWORD *)(a2 + 24);
        if ( v99 )
        {
          v100 = v99 - 1;
          v101 = 1;
          v102 = 0;
          for ( mm = v100 & v147; ; mm = v100 & v111 )
          {
            v96 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * mm);
            v104 = *v96;
            if ( v91 == (_BYTE *)*v96 )
              break;
            if ( v104 == -4096 )
              goto LABEL_198;
            if ( v102 || v104 != -8192 )
              v96 = v102;
            v111 = v101 + mm;
            v102 = v96;
            ++v101;
          }
          goto LABEL_191;
        }
LABEL_254:
        ++*(_DWORD *)(a2 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_110B120(a2, 2 * v92);
    v105 = *(_DWORD *)(a2 + 24);
    if ( !v105 )
      goto LABEL_254;
    v106 = v105 - 1;
    v102 = 0;
    v107 = 1;
    for ( nn = v106 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4)); ; nn = v106 & v110 )
    {
      v96 = (_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * nn);
      v109 = *v96;
      if ( v91 == (_BYTE *)*v96 )
        break;
      if ( v109 == -4096 )
      {
LABEL_198:
        if ( v102 )
          v96 = v102;
        break;
      }
      if ( v102 || v109 != -8192 )
        v96 = v102;
      v110 = v107 + nn;
      v102 = v96;
      ++v107;
    }
LABEL_191:
    ++*(_DWORD *)(a2 + 16);
    if ( *v96 != -4096 )
      --*(_DWORD *)(a2 + 20);
    *v96 = v91;
    result = 0;
  }
LABEL_51:
  if ( v148 != v150 )
  {
    v145 = result;
    _libc_free((unsigned __int64)v148);
    return v145;
  }
  return result;
}
