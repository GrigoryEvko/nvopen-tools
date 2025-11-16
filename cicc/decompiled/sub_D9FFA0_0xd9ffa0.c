// Function: sub_D9FFA0
// Address: 0xd9ffa0
//
__int64 __fastcall sub_D9FFA0(__int64 a1, __int64 a2)
{
  int v3; // r15d
  unsigned int v4; // eax
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // r13
  int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *j; // rdx
  int v12; // eax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *m; // rdx
  int v16; // eax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *ii; // rdx
  int v20; // r15d
  unsigned int v21; // eax
  __int64 v22; // r12
  __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // rdi
  int v26; // r14d
  unsigned int v27; // eax
  _QWORD *v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rsi
  _QWORD *v31; // r13
  _QWORD *v32; // rdi
  int v33; // ecx
  unsigned int v34; // edx
  __int64 v35; // rdi
  __int64 v36; // r9
  __int64 v37; // rsi
  __int64 v38; // rax
  int v39; // r14d
  __int64 result; // rax
  __int64 *v41; // r12
  __int64 v42; // rsi
  __int64 *v43; // r13
  __int64 *v44; // rdi
  unsigned int v45; // ecx
  unsigned int v46; // eax
  _QWORD *v47; // rdi
  int v48; // r12d
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rdi
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _QWORD *k; // rdx
  unsigned int v54; // ecx
  unsigned int v55; // eax
  _QWORD *v56; // rdi
  int v57; // r12d
  _QWORD *v58; // rax
  unsigned int v59; // ecx
  unsigned int v60; // eax
  _QWORD *v61; // rdi
  int v62; // r12d
  _QWORD *v63; // rax
  __int64 *v64; // rdi
  __int64 v65; // rdi
  _QWORD *v66; // rdi
  int v67; // edx
  _QWORD *v68; // rdi
  __int64 v69; // r12
  unsigned int v70; // eax
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // rax
  __int64 v73; // rdx
  __int64 i1; // rdx
  int v75; // edx
  __int64 v76; // r12
  unsigned int v77; // eax
  _QWORD *v78; // rdi
  unsigned __int64 v79; // rdx
  unsigned __int64 v80; // rax
  _QWORD *v81; // rax
  __int64 v82; // rdx
  _QWORD *kk; // rdx
  int v84; // edx
  __int64 v85; // r12
  unsigned int v86; // eax
  _QWORD *v87; // rdi
  unsigned __int64 v88; // rdx
  unsigned __int64 v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // rdx
  _QWORD *i; // rdx
  unsigned int v93; // edx
  int v94; // r12d
  unsigned int v95; // eax
  _QWORD *v96; // rdi
  unsigned __int64 v97; // rax
  unsigned __int64 v98; // rdi
  _QWORD *v99; // rax
  __int64 v100; // rdx
  _QWORD *mm; // rdx
  unsigned int v102; // ecx
  unsigned int v103; // edx
  int v104; // r12d
  unsigned __int64 v105; // rax
  unsigned __int64 v106; // rdi
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 nn; // rdx
  unsigned __int64 v110; // rax
  unsigned __int64 v111; // rdi
  _QWORD *v112; // rax
  __int64 v113; // rdx
  _QWORD *jj; // rdx
  unsigned __int64 v115; // rax
  unsigned __int64 v116; // rdi
  _QWORD *v117; // rax
  __int64 v118; // rdx
  _QWORD *n; // rdx
  _QWORD *v120; // rax
  _QWORD *v121; // rax
  _QWORD *v122; // rax
  _QWORD *v123; // rax

  sub_D9CC70(a1 + 648);
  sub_D9CC70(a1 + 680);
  v3 = *(_DWORD *)(a1 + 728);
  ++*(_QWORD *)(a1 + 712);
  if ( !v3 && !*(_DWORD *)(a1 + 732) )
    goto LABEL_15;
  v4 = 4 * v3;
  v5 = *(_QWORD *)(a1 + 720);
  v6 = 72LL * *(unsigned int *)(a1 + 736);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v4 = 64;
  v7 = v5 + v6;
  if ( *(_DWORD *)(a1 + 736) <= v4 )
  {
    for ( ; v5 != v7; v5 += 72 )
    {
      if ( *(_QWORD *)v5 != -4096 )
      {
        if ( *(_QWORD *)v5 != -8192 && !*(_BYTE *)(v5 + 36) )
          _libc_free(*(_QWORD *)(v5 + 16), a2);
        *(_QWORD *)v5 = -4096;
      }
    }
    goto LABEL_14;
  }
  do
  {
    while ( 1 )
    {
      if ( *(_QWORD *)v5 == -4096 || *(_QWORD *)v5 == -8192 )
        goto LABEL_125;
      if ( *(_BYTE *)(v5 + 36) )
        break;
      _libc_free(*(_QWORD *)(v5 + 16), a2);
LABEL_125:
      v5 += 72;
      if ( v5 == v7 )
        goto LABEL_173;
    }
    v5 += 72;
  }
  while ( v5 != v7 );
LABEL_173:
  v84 = *(_DWORD *)(a1 + 736);
  if ( !v3 )
  {
    if ( v84 )
    {
      a2 = v6;
      sub_C7D6A0(*(_QWORD *)(a1 + 720), v6, 8);
      *(_QWORD *)(a1 + 720) = 0;
      *(_QWORD *)(a1 + 728) = 0;
      *(_DWORD *)(a1 + 736) = 0;
      goto LABEL_15;
    }
LABEL_14:
    *(_QWORD *)(a1 + 728) = 0;
    goto LABEL_15;
  }
  v85 = 64;
  if ( v3 != 1 )
  {
    _BitScanReverse(&v86, v3 - 1);
    v85 = (unsigned int)(1 << (33 - (v86 ^ 0x1F)));
    if ( (int)v85 < 64 )
      v85 = 64;
  }
  v87 = *(_QWORD **)(a1 + 720);
  if ( (_DWORD)v85 == v84 )
  {
    *(_QWORD *)(a1 + 728) = 0;
    v121 = &v87[9 * v85];
    do
    {
      if ( v87 )
        *v87 = -4096;
      v87 += 9;
    }
    while ( v121 != v87 );
  }
  else
  {
    sub_C7D6A0((__int64)v87, v6, 8);
    a2 = 8;
    v88 = ((((((((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v85 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v85 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v85 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v85 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 16;
    v89 = (v88
         | (((((((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v85 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v85 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v85 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v85 / 3u + 1) | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v85 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v85 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 736) = v89;
    v90 = (_QWORD *)sub_C7D670(72 * v89, 8);
    v91 = *(unsigned int *)(a1 + 736);
    *(_QWORD *)(a1 + 728) = 0;
    *(_QWORD *)(a1 + 720) = v90;
    for ( i = &v90[9 * v91]; i != v90; v90 += 9 )
    {
      if ( v90 )
        *v90 = -4096;
    }
  }
LABEL_15:
  v8 = *(_DWORD *)(a1 + 888);
  ++*(_QWORD *)(a1 + 872);
  if ( v8 )
  {
    v45 = 4 * v8;
    a2 = 64;
    v9 = *(unsigned int *)(a1 + 896);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v45 = 64;
    if ( (unsigned int)v9 <= v45 )
    {
LABEL_18:
      v10 = *(_QWORD **)(a1 + 880);
      for ( j = &v10[2 * v9]; j != v10; v10 += 2 )
        *v10 = -4096;
      *(_QWORD *)(a1 + 888) = 0;
      goto LABEL_21;
    }
    v46 = v8 - 1;
    if ( v46 )
    {
      _BitScanReverse(&v46, v46);
      v47 = *(_QWORD **)(a1 + 880);
      v48 = 1 << (33 - (v46 ^ 0x1F));
      if ( v48 < 64 )
        v48 = 64;
      if ( (_DWORD)v9 == v48 )
      {
        *(_QWORD *)(a1 + 888) = 0;
        v123 = &v47[2 * (unsigned int)v9];
        do
        {
          if ( v47 )
            *v47 = -4096;
          v47 += 2;
        }
        while ( v123 != v47 );
        goto LABEL_21;
      }
    }
    else
    {
      v47 = *(_QWORD **)(a1 + 880);
      v48 = 64;
    }
    sub_C7D6A0((__int64)v47, 16LL * (unsigned int)v9, 8);
    a2 = 8;
    v49 = ((((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
             | (4 * v48 / 3u + 1)
             | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
         | (4 * v48 / 3u + 1)
         | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 16;
    v50 = (v49
         | (((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
             | (4 * v48 / 3u + 1)
             | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
         | (4 * v48 / 3u + 1)
         | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 896) = v50;
    v51 = (_QWORD *)sub_C7D670(16 * v50, 8);
    v52 = *(unsigned int *)(a1 + 896);
    *(_QWORD *)(a1 + 888) = 0;
    *(_QWORD *)(a1 + 880) = v51;
    for ( k = &v51[2 * v52]; k != v51; v51 += 2 )
    {
      if ( v51 )
        *v51 = -4096;
    }
    goto LABEL_21;
  }
  if ( *(_DWORD *)(a1 + 892) )
  {
    v9 = *(unsigned int *)(a1 + 896);
    if ( (unsigned int)v9 <= 0x40 )
      goto LABEL_18;
    a2 = 16LL * (unsigned int)v9;
    sub_C7D6A0(*(_QWORD *)(a1 + 880), a2, 8);
    *(_QWORD *)(a1 + 880) = 0;
    *(_QWORD *)(a1 + 888) = 0;
    *(_DWORD *)(a1 + 896) = 0;
  }
LABEL_21:
  v12 = *(_DWORD *)(a1 + 760);
  ++*(_QWORD *)(a1 + 744);
  if ( v12 )
  {
    v59 = 4 * v12;
    a2 = 64;
    v13 = *(unsigned int *)(a1 + 768);
    if ( (unsigned int)(4 * v12) < 0x40 )
      v59 = 64;
    if ( (unsigned int)v13 <= v59 )
    {
LABEL_24:
      v14 = *(_QWORD **)(a1 + 752);
      for ( m = &v14[2 * v13]; m != v14; v14 += 2 )
        *v14 = -4096;
      *(_QWORD *)(a1 + 760) = 0;
      goto LABEL_27;
    }
    v60 = v12 - 1;
    if ( v60 )
    {
      _BitScanReverse(&v60, v60);
      v61 = *(_QWORD **)(a1 + 752);
      v62 = 1 << (33 - (v60 ^ 0x1F));
      if ( v62 < 64 )
        v62 = 64;
      if ( v62 == (_DWORD)v13 )
      {
        *(_QWORD *)(a1 + 760) = 0;
        v63 = &v61[2 * (unsigned int)v62];
        do
        {
          if ( v61 )
            *v61 = -4096;
          v61 += 2;
        }
        while ( v63 != v61 );
        goto LABEL_27;
      }
    }
    else
    {
      v61 = *(_QWORD **)(a1 + 752);
      v62 = 64;
    }
    sub_C7D6A0((__int64)v61, 16LL * (unsigned int)v13, 8);
    a2 = 8;
    v115 = ((((((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
              | (4 * v62 / 3u + 1)
              | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
            | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
            | (4 * v62 / 3u + 1)
            | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
            | (4 * v62 / 3u + 1)
            | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
          | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
          | (4 * v62 / 3u + 1)
          | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 16;
    v116 = (v115
          | (((((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
              | (4 * v62 / 3u + 1)
              | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
            | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
            | (4 * v62 / 3u + 1)
            | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
            | (4 * v62 / 3u + 1)
            | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
          | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
          | (4 * v62 / 3u + 1)
          | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(a1 + 768) = v116;
    v117 = (_QWORD *)sub_C7D670(16 * v116, 8);
    v118 = *(unsigned int *)(a1 + 768);
    *(_QWORD *)(a1 + 760) = 0;
    *(_QWORD *)(a1 + 752) = v117;
    for ( n = &v117[2 * v118]; n != v117; v117 += 2 )
    {
      if ( v117 )
        *v117 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 764) )
  {
    v13 = *(unsigned int *)(a1 + 768);
    if ( (unsigned int)v13 <= 0x40 )
      goto LABEL_24;
    a2 = 16LL * (unsigned int)v13;
    sub_C7D6A0(*(_QWORD *)(a1 + 752), a2, 8);
    *(_QWORD *)(a1 + 752) = 0;
    *(_QWORD *)(a1 + 760) = 0;
    *(_DWORD *)(a1 + 768) = 0;
  }
LABEL_27:
  sub_D9FB70(a1 + 128);
  sub_D9D010(a1 + 776, a2);
  sub_D9D010(a1 + 808, a2);
  sub_D9D270(a1 + 840, a2);
  sub_D9D4C0(a1 + 904, a2);
  sub_D9DC00(a1 + 968);
  sub_D9DC00(a1 + 1000);
  sub_D9DE90(a1 + 96, a2);
  v16 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  if ( v16 )
  {
    v54 = 4 * v16;
    v17 = *(unsigned int *)(a1 + 88);
    if ( (unsigned int)(4 * v16) < 0x40 )
      v54 = 64;
    if ( (unsigned int)v17 <= v54 )
    {
LABEL_30:
      v18 = *(_QWORD **)(a1 + 72);
      for ( ii = &v18[2 * v17]; ii != v18; v18 += 2 )
        *v18 = -4096;
      *(_QWORD *)(a1 + 80) = 0;
      goto LABEL_33;
    }
    v55 = v16 - 1;
    if ( v55 )
    {
      _BitScanReverse(&v55, v55);
      v56 = *(_QWORD **)(a1 + 72);
      v57 = 1 << (33 - (v55 ^ 0x1F));
      if ( v57 < 64 )
        v57 = 64;
      if ( (_DWORD)v17 == v57 )
      {
        *(_QWORD *)(a1 + 80) = 0;
        v58 = &v56[2 * (unsigned int)v17];
        do
        {
          if ( v56 )
            *v56 = -4096;
          v56 += 2;
        }
        while ( v58 != v56 );
        goto LABEL_33;
      }
    }
    else
    {
      v56 = *(_QWORD **)(a1 + 72);
      v57 = 64;
    }
    sub_C7D6A0((__int64)v56, 16LL * (unsigned int)v17, 8);
    v110 = ((((((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
              | (4 * v57 / 3u + 1)
              | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
            | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
            | (4 * v57 / 3u + 1)
            | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
            | (4 * v57 / 3u + 1)
            | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
          | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
          | (4 * v57 / 3u + 1)
          | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 16;
    v111 = (v110
          | (((((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
              | (4 * v57 / 3u + 1)
              | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
            | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
            | (4 * v57 / 3u + 1)
            | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
            | (4 * v57 / 3u + 1)
            | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
          | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
          | (4 * v57 / 3u + 1)
          | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(a1 + 88) = v111;
    v112 = (_QWORD *)sub_C7D670(16 * v111, 8);
    v113 = *(unsigned int *)(a1 + 88);
    *(_QWORD *)(a1 + 80) = 0;
    *(_QWORD *)(a1 + 72) = v112;
    for ( jj = &v112[2 * v113]; jj != v112; v112 += 2 )
    {
      if ( v112 )
        *v112 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 84) )
  {
    v17 = *(unsigned int *)(a1 + 88);
    if ( (unsigned int)v17 <= 0x40 )
      goto LABEL_30;
    sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * (unsigned int)v17, 8);
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 80) = 0;
    *(_DWORD *)(a1 + 88) = 0;
  }
LABEL_33:
  v20 = *(_DWORD *)(a1 + 632);
  ++*(_QWORD *)(a1 + 616);
  if ( !v20 && !*(_DWORD *)(a1 + 636) )
    goto LABEL_47;
  v21 = 4 * v20;
  v22 = *(_QWORD *)(a1 + 624);
  v23 = 24LL * *(unsigned int *)(a1 + 640);
  if ( (unsigned int)(4 * v20) < 0x40 )
    v21 = 64;
  v24 = v22 + v23;
  if ( *(_DWORD *)(a1 + 640) <= v21 )
  {
    for ( ; v22 != v24; v22 += 24 )
    {
      if ( *(_QWORD *)v22 != -4096 )
      {
        if ( *(_QWORD *)v22 != -8192 && *(_DWORD *)(v22 + 16) > 0x40u )
        {
          v25 = *(_QWORD *)(v22 + 8);
          if ( v25 )
            j_j___libc_free_0_0(v25);
        }
        *(_QWORD *)v22 = -4096;
      }
    }
LABEL_46:
    *(_QWORD *)(a1 + 632) = 0;
    goto LABEL_47;
  }
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_QWORD *)v22 == -4096 )
      {
LABEL_140:
        v22 += 24;
        if ( v22 == v24 )
          goto LABEL_163;
        continue;
      }
      break;
    }
    if ( *(_QWORD *)v22 != -8192 )
    {
      if ( *(_DWORD *)(v22 + 16) > 0x40u )
      {
        v65 = *(_QWORD *)(v22 + 8);
        if ( v65 )
          j_j___libc_free_0_0(v65);
      }
      goto LABEL_140;
    }
    v22 += 24;
    if ( v22 != v24 )
      continue;
    break;
  }
LABEL_163:
  v75 = *(_DWORD *)(a1 + 640);
  if ( !v20 )
  {
    if ( v75 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 624), v23, 8);
      *(_QWORD *)(a1 + 624) = 0;
      *(_QWORD *)(a1 + 632) = 0;
      *(_DWORD *)(a1 + 640) = 0;
      goto LABEL_47;
    }
    goto LABEL_46;
  }
  v76 = 64;
  if ( v20 != 1 )
  {
    _BitScanReverse(&v77, v20 - 1);
    v76 = (unsigned int)(1 << (33 - (v77 ^ 0x1F)));
    if ( (int)v76 < 64 )
      v76 = 64;
  }
  v78 = *(_QWORD **)(a1 + 624);
  if ( (_DWORD)v76 == v75 )
  {
    *(_QWORD *)(a1 + 632) = 0;
    v120 = &v78[3 * v76];
    do
    {
      if ( v78 )
        *v78 = -4096;
      v78 += 3;
    }
    while ( v120 != v78 );
  }
  else
  {
    sub_C7D6A0((__int64)v78, v23, 8);
    v79 = ((((((((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v76 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v76 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v76 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v76 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 16;
    v80 = (v79
         | (((((((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v76 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v76 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v76 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v76 / 3u + 1) | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v76 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v76 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 640) = v80;
    v81 = (_QWORD *)sub_C7D670(24 * v80, 8);
    v82 = *(unsigned int *)(a1 + 640);
    *(_QWORD *)(a1 + 632) = 0;
    *(_QWORD *)(a1 + 624) = v81;
    for ( kk = &v81[3 * v82]; kk != v81; v81 += 3 )
    {
      if ( v81 )
        *v81 = -4096;
    }
  }
LABEL_47:
  v26 = *(_DWORD *)(a1 + 1208);
  ++*(_QWORD *)(a1 + 1192);
  if ( !v26 && !*(_DWORD *)(a1 + 1212) )
    goto LABEL_63;
  v27 = 4 * v26;
  v28 = *(_QWORD **)(a1 + 1200);
  v29 = *(unsigned int *)(a1 + 1216);
  v30 = v29 << 6;
  if ( (unsigned int)(4 * v26) < 0x40 )
    v27 = 64;
  v31 = (_QWORD *)((char *)v28 + v30);
  if ( (unsigned int)v29 <= v27 )
  {
    while ( 1 )
    {
      if ( v28 == v31 )
        goto LABEL_62;
      if ( *v28 != -4096 )
        break;
      if ( v28[1] != -4096 )
        goto LABEL_54;
LABEL_57:
      v28 += 8;
    }
    if ( *v28 != -8192 || v28[1] != -8192 )
    {
LABEL_54:
      v32 = (_QWORD *)v28[3];
      if ( v32 != v28 + 5 )
        _libc_free(v32, v30);
    }
    *v28 = -4096;
    v28[1] = -4096;
    goto LABEL_57;
  }
  while ( 2 )
  {
    if ( *v28 == -4096 )
    {
      if ( v28[1] == -4096 )
        goto LABEL_148;
    }
    else if ( *v28 == -8192 && v28[1] == -8192 )
    {
      goto LABEL_148;
    }
    v66 = (_QWORD *)v28[3];
    if ( v66 != v28 + 5 )
      _libc_free(v66, v30);
LABEL_148:
    v28 += 8;
    if ( v28 != v31 )
      continue;
    break;
  }
  v93 = *(_DWORD *)(a1 + 1216);
  if ( !v26 )
  {
    if ( v93 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1200), v30, 8);
      *(_QWORD *)(a1 + 1200) = 0;
      *(_QWORD *)(a1 + 1208) = 0;
      *(_DWORD *)(a1 + 1216) = 0;
      goto LABEL_63;
    }
LABEL_62:
    *(_QWORD *)(a1 + 1208) = 0;
    goto LABEL_63;
  }
  v94 = 64;
  if ( v26 != 1 )
  {
    _BitScanReverse(&v95, v26 - 1);
    v94 = 1 << (33 - (v95 ^ 0x1F));
    if ( v94 < 64 )
      v94 = 64;
  }
  v96 = *(_QWORD **)(a1 + 1200);
  if ( v93 == v94 )
  {
    *(_QWORD *)(a1 + 1208) = 0;
    v122 = &v96[8 * (unsigned __int64)v93];
    do
    {
      if ( v96 )
      {
        *v96 = -4096;
        v96[1] = -4096;
      }
      v96 += 8;
    }
    while ( v122 != v96 );
  }
  else
  {
    sub_C7D6A0((__int64)v96, v30, 8);
    v97 = ((((((((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
             | (4 * v94 / 3u + 1)
             | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 4)
           | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
           | (4 * v94 / 3u + 1)
           | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
           | (4 * v94 / 3u + 1)
           | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 4)
         | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
         | (4 * v94 / 3u + 1)
         | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 16;
    v98 = (v97
         | (((((((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
             | (4 * v94 / 3u + 1)
             | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 4)
           | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
           | (4 * v94 / 3u + 1)
           | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
           | (4 * v94 / 3u + 1)
           | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 4)
         | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
         | (4 * v94 / 3u + 1)
         | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 1216) = v98;
    v99 = (_QWORD *)sub_C7D670(v98 << 6, 8);
    v100 = *(unsigned int *)(a1 + 1216);
    *(_QWORD *)(a1 + 1208) = 0;
    *(_QWORD *)(a1 + 1200) = v99;
    for ( mm = &v99[8 * v100]; mm != v99; v99 += 8 )
    {
      if ( v99 )
      {
        *v99 = -4096;
        v99[1] = -4096;
      }
    }
  }
LABEL_63:
  v33 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  if ( v33 || *(_DWORD *)(a1 + 180) )
  {
    v34 = 4 * v33;
    v35 = *(_QWORD *)(a1 + 168);
    v36 = *(unsigned int *)(a1 + 184);
    v37 = 32 * v36;
    if ( (unsigned int)(4 * v33) < 0x40 )
      v34 = 64;
    v38 = v35 + v37;
    if ( (unsigned int)v36 <= v34 )
    {
      for ( ; v35 != v38; *(_WORD *)(v35 - 16) = 0 )
      {
        *(_QWORD *)v35 = 0;
        v35 += 32;
        *(_QWORD *)(v35 - 24) = 0;
      }
      *(_QWORD *)(a1 + 176) = 0;
      goto LABEL_71;
    }
    if ( !v33 )
    {
      sub_C7D6A0(v35, v37, 8);
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      goto LABEL_71;
    }
    v102 = v33 - 1;
    if ( !v102 )
    {
      v104 = 64;
      goto LABEL_201;
    }
    _BitScanReverse(&v103, v102);
    v104 = 1 << (33 - (v103 ^ 0x1F));
    if ( v104 < 64 )
      v104 = 64;
    if ( (_DWORD)v36 == v104 )
    {
      *(_QWORD *)(a1 + 176) = 0;
      do
      {
        if ( v35 )
        {
          *(_QWORD *)v35 = 0;
          *(_QWORD *)(v35 + 8) = 0;
          *(_WORD *)(v35 + 16) = 0;
        }
        v35 += 32;
      }
      while ( v38 != v35 );
    }
    else
    {
LABEL_201:
      sub_C7D6A0(v35, v37, 8);
      v105 = ((((((((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
                | (4 * v104 / 3u + 1)
                | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 4)
              | (((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
              | (4 * v104 / 3u + 1)
              | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 8)
            | (((((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
              | (4 * v104 / 3u + 1)
              | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 4)
            | (((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
            | (4 * v104 / 3u + 1)
            | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 16;
      v106 = (v105
            | (((((((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
                | (4 * v104 / 3u + 1)
                | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 4)
              | (((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
              | (4 * v104 / 3u + 1)
              | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 8)
            | (((((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
              | (4 * v104 / 3u + 1)
              | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 4)
            | (((4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1)) >> 2)
            | (4 * v104 / 3u + 1)
            | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1))
           + 1;
      *(_DWORD *)(a1 + 184) = v106;
      v107 = sub_C7D670(32 * v106, 8);
      v108 = *(unsigned int *)(a1 + 184);
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 168) = v107;
      for ( nn = v107 + 32 * v108; nn != v107; v107 += 32 )
      {
        if ( v107 )
        {
          *(_QWORD *)v107 = 0;
          *(_QWORD *)(v107 + 8) = 0;
          *(_WORD *)(v107 + 16) = 0;
        }
      }
    }
  }
LABEL_71:
  v39 = *(_DWORD *)(a1 + 208);
  ++*(_QWORD *)(a1 + 192);
  if ( v39 || (result = *(unsigned int *)(a1 + 212), (_DWORD)result) )
  {
    result = (unsigned int)(4 * v39);
    v41 = *(__int64 **)(a1 + 200);
    v42 = 9LL * *(unsigned int *)(a1 + 216);
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v43 = &v41[9 * *(unsigned int *)(a1 + 216)];
    if ( *(_DWORD *)(a1 + 216) <= (unsigned int)result )
    {
      while ( v41 != v43 )
      {
        result = *v41;
        if ( *v41 != -4096 )
        {
          if ( result != -8192 )
          {
            v44 = (__int64 *)v41[1];
            result = (__int64)(v41 + 3);
            if ( v44 != v41 + 3 )
              result = _libc_free(v44, v42);
          }
          *v41 = -4096;
        }
        v41 += 9;
      }
    }
    else
    {
      do
      {
        result = *v41;
        if ( *v41 != -8192 && result != -4096 )
        {
          v64 = (__int64 *)v41[1];
          result = (__int64)(v41 + 3);
          if ( v64 != v41 + 3 )
            result = _libc_free(v64, v42);
        }
        v41 += 9;
      }
      while ( v41 != v43 );
      v67 = *(_DWORD *)(a1 + 216);
      v68 = *(_QWORD **)(a1 + 200);
      if ( v39 )
      {
        v69 = 64;
        if ( v39 != 1 )
        {
          _BitScanReverse(&v70, v39 - 1);
          v69 = (unsigned int)(1 << (33 - (v70 ^ 0x1F)));
          if ( (int)v69 < 64 )
            v69 = 64;
        }
        if ( (_DWORD)v69 == v67 )
        {
          *(_QWORD *)(a1 + 208) = 0;
          result = (__int64)&v68[9 * v69];
          do
          {
            if ( v68 )
              *v68 = -4096;
            v68 += 9;
          }
          while ( (_QWORD *)result != v68 );
        }
        else
        {
          sub_C7D6A0((__int64)v68, 8 * v42, 8);
          v71 = ((((((((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v69 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v69 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v69 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v69 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 16;
          v72 = (v71
               | (((((((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v69 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v69 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v69 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v69 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 216) = v72;
          result = sub_C7D670(72 * v72, 8);
          v73 = *(unsigned int *)(a1 + 216);
          *(_QWORD *)(a1 + 208) = 0;
          *(_QWORD *)(a1 + 200) = result;
          for ( i1 = result + 72 * v73; i1 != result; result += 72 )
          {
            if ( result )
              *(_QWORD *)result = -4096;
          }
        }
        return result;
      }
      if ( v67 )
      {
        result = sub_C7D6A0((__int64)v68, 8 * v42, 8);
        *(_QWORD *)(a1 + 200) = 0;
        *(_QWORD *)(a1 + 208) = 0;
        *(_DWORD *)(a1 + 216) = 0;
        return result;
      }
    }
    *(_QWORD *)(a1 + 208) = 0;
  }
  return result;
}
