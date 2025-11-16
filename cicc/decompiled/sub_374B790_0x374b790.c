// Function: sub_374B790
// Address: 0x374b790
//
__int64 __fastcall sub_374B790(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _DWORD *v8; // rax
  _DWORD *k; // rdx
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *ii; // rdx
  __int64 v14; // r13
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  int v18; // eax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *jj; // rdx
  int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdx
  _DWORD *v25; // rax
  _DWORD *mm; // rdx
  int v27; // eax
  __int64 v28; // rdx
  size_t v29; // rdx
  int v30; // r15d
  __int64 v31; // r12
  unsigned int v32; // eax
  __int64 v33; // r14
  __int64 v34; // r13
  int v35; // eax
  __int64 result; // rax
  __int64 v37; // rdx
  __int64 i3; // rdx
  unsigned int v39; // eax
  __int64 v40; // rdx
  unsigned int v41; // eax
  __int64 v42; // rdx
  unsigned int v43; // ecx
  unsigned int v44; // eax
  _QWORD *v45; // rdi
  int v46; // r12d
  unsigned int v47; // ecx
  unsigned int v48; // eax
  _DWORD *v49; // rdi
  __int64 v50; // r12
  _DWORD *v51; // rax
  unsigned int v52; // ecx
  unsigned int v53; // eax
  _DWORD *v54; // rdi
  __int64 v55; // r12
  _DWORD *v56; // rax
  unsigned int v57; // ecx
  unsigned int v58; // eax
  _QWORD *v59; // rdi
  int v60; // r12d
  _QWORD *v61; // rax
  unsigned int v62; // ecx
  unsigned int v63; // eax
  _QWORD *v64; // rdi
  int v65; // r12d
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdi
  _QWORD *v68; // rax
  __int64 v69; // rdx
  _QWORD *n; // rdx
  unsigned int v71; // ecx
  unsigned int v72; // eax
  _DWORD *v73; // rdi
  int v74; // r12d
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rdi
  _DWORD *v77; // rax
  __int64 v78; // rdx
  _DWORD *m; // rdx
  unsigned int v80; // ecx
  unsigned int v81; // eax
  _QWORD *v82; // rdi
  int v83; // r12d
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // rdi
  _QWORD *v86; // rax
  __int64 v87; // rdx
  _QWORD *j; // rdx
  int v89; // edx
  __int64 v90; // r12
  unsigned int v91; // r15d
  unsigned int v92; // eax
  _QWORD *v93; // rdi
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // rax
  _QWORD *v96; // rax
  __int64 v97; // rdx
  _QWORD *i2; // rdx
  unsigned __int64 v99; // rax
  unsigned __int64 v100; // rdi
  __int64 v101; // rdx
  __int64 i4; // rdx
  unsigned __int64 v103; // rdx
  unsigned __int64 v104; // rax
  _DWORD *v105; // rax
  __int64 v106; // rdx
  _DWORD *nn; // rdx
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rax
  _DWORD *v110; // rax
  __int64 v111; // rdx
  _DWORD *i1; // rdx
  unsigned __int64 v113; // rax
  unsigned __int64 v114; // rdi
  _QWORD *v115; // rax
  __int64 v116; // rdx
  _QWORD *kk; // rdx
  _QWORD *v118; // rax
  _QWORD *v119; // rax
  _DWORD *v120; // rax
  _QWORD *v121; // rax

  v2 = *(_DWORD *)(a1 + 136);
  ++*(_QWORD *)(a1 + 120);
  *(_DWORD *)(a1 + 64) = 0;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 140) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 144);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 136) = 0;
      *(_DWORD *)(a1 + 144) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v80 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 144);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v80 = 64;
  if ( (unsigned int)v3 <= v80 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 128);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 136) = 0;
    goto LABEL_7;
  }
  v81 = v2 - 1;
  if ( !v81 )
  {
    v82 = *(_QWORD **)(a1 + 128);
    v83 = 64;
LABEL_155:
    sub_C7D6A0((__int64)v82, 16LL * (unsigned int)v3, 8);
    v84 = ((((((((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
             | (4 * v83 / 3u + 1)
             | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 4)
           | (((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
           | (4 * v83 / 3u + 1)
           | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
           | (4 * v83 / 3u + 1)
           | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 4)
         | (((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
         | (4 * v83 / 3u + 1)
         | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 16;
    v85 = (v84
         | (((((((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
             | (4 * v83 / 3u + 1)
             | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 4)
           | (((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
           | (4 * v83 / 3u + 1)
           | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
           | (4 * v83 / 3u + 1)
           | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 4)
         | (((4 * v83 / 3u + 1) | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1)) >> 2)
         | (4 * v83 / 3u + 1)
         | ((unsigned __int64)(4 * v83 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 144) = v85;
    v86 = (_QWORD *)sub_C7D670(16 * v85, 8);
    v87 = *(unsigned int *)(a1 + 144);
    *(_QWORD *)(a1 + 136) = 0;
    *(_QWORD *)(a1 + 128) = v86;
    for ( j = &v86[2 * v87]; j != v86; v86 += 2 )
    {
      if ( v86 )
        *v86 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v81, v81);
  v82 = *(_QWORD **)(a1 + 128);
  v83 = 1 << (33 - (v81 ^ 0x1F));
  if ( v83 < 64 )
    v83 = 64;
  if ( v83 != (_DWORD)v3 )
    goto LABEL_155;
  *(_QWORD *)(a1 + 136) = 0;
  v119 = &v82[2 * (unsigned int)v83];
  do
  {
    if ( v82 )
      *v82 = -4096;
    v82 += 2;
  }
  while ( v119 != v82 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 168);
  ++*(_QWORD *)(a1 + 152);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 172) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 176);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 160), 16LL * (unsigned int)v7, 8);
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 168) = 0;
      *(_DWORD *)(a1 + 176) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v71 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 176);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v71 = 64;
  if ( v71 >= (unsigned int)v7 )
  {
LABEL_10:
    v8 = *(_DWORD **)(a1 + 160);
    for ( k = &v8[4 * v7]; k != v8; v8 += 4 )
      *v8 = -1;
    *(_QWORD *)(a1 + 168) = 0;
    goto LABEL_13;
  }
  v72 = v6 - 1;
  if ( !v72 )
  {
    v73 = *(_DWORD **)(a1 + 160);
    v74 = 64;
LABEL_143:
    sub_C7D6A0((__int64)v73, 16LL * (unsigned int)v7, 8);
    v75 = ((((((((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
             | (4 * v74 / 3u + 1)
             | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 4)
           | (((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
           | (4 * v74 / 3u + 1)
           | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
           | (4 * v74 / 3u + 1)
           | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 4)
         | (((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
         | (4 * v74 / 3u + 1)
         | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 16;
    v76 = (v75
         | (((((((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
             | (4 * v74 / 3u + 1)
             | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 4)
           | (((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
           | (4 * v74 / 3u + 1)
           | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
           | (4 * v74 / 3u + 1)
           | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 4)
         | (((4 * v74 / 3u + 1) | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1)) >> 2)
         | (4 * v74 / 3u + 1)
         | ((unsigned __int64)(4 * v74 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 176) = v76;
    v77 = (_DWORD *)sub_C7D670(16 * v76, 8);
    v78 = *(unsigned int *)(a1 + 176);
    *(_QWORD *)(a1 + 168) = 0;
    *(_QWORD *)(a1 + 160) = v77;
    for ( m = &v77[4 * v78]; m != v77; v77 += 4 )
    {
      if ( v77 )
        *v77 = -1;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v72, v72);
  v73 = *(_DWORD **)(a1 + 160);
  v74 = 1 << (33 - (v72 ^ 0x1F));
  if ( v74 < 64 )
    v74 = 64;
  if ( v74 != (_DWORD)v7 )
    goto LABEL_143;
  *(_QWORD *)(a1 + 168) = 0;
  v120 = &v73[4 * v74];
  do
  {
    if ( v73 )
      *v73 = -1;
    v73 += 4;
  }
  while ( v120 != v73 );
LABEL_13:
  v10 = *(_DWORD *)(a1 + 264);
  ++*(_QWORD *)(a1 + 248);
  if ( v10 )
  {
    v62 = 4 * v10;
    v11 = *(unsigned int *)(a1 + 272);
    if ( (unsigned int)(4 * v10) < 0x40 )
      v62 = 64;
    if ( (unsigned int)v11 <= v62 )
      goto LABEL_16;
    v63 = v10 - 1;
    if ( v63 )
    {
      _BitScanReverse(&v63, v63);
      v64 = *(_QWORD **)(a1 + 256);
      v65 = 1 << (33 - (v63 ^ 0x1F));
      if ( v65 < 64 )
        v65 = 64;
      if ( (_DWORD)v11 == v65 )
      {
        *(_QWORD *)(a1 + 264) = 0;
        v121 = &v64[2 * (unsigned int)v11];
        do
        {
          if ( v64 )
            *v64 = -4096;
          v64 += 2;
        }
        while ( v121 != v64 );
        goto LABEL_19;
      }
    }
    else
    {
      v64 = *(_QWORD **)(a1 + 256);
      v65 = 64;
    }
    sub_C7D6A0((__int64)v64, 16LL * (unsigned int)v11, 8);
    v66 = ((((((((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
             | (4 * v65 / 3u + 1)
             | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 4)
           | (((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
           | (4 * v65 / 3u + 1)
           | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
           | (4 * v65 / 3u + 1)
           | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 4)
         | (((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
         | (4 * v65 / 3u + 1)
         | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 16;
    v67 = (v66
         | (((((((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
             | (4 * v65 / 3u + 1)
             | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 4)
           | (((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
           | (4 * v65 / 3u + 1)
           | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
           | (4 * v65 / 3u + 1)
           | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 4)
         | (((4 * v65 / 3u + 1) | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)) >> 2)
         | (4 * v65 / 3u + 1)
         | ((unsigned __int64)(4 * v65 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 272) = v67;
    v68 = (_QWORD *)sub_C7D670(16 * v67, 8);
    v69 = *(unsigned int *)(a1 + 272);
    *(_QWORD *)(a1 + 264) = 0;
    *(_QWORD *)(a1 + 256) = v68;
    for ( n = &v68[2 * v69]; n != v68; v68 += 2 )
    {
      if ( v68 )
        *v68 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 268) )
  {
    v11 = *(unsigned int *)(a1 + 272);
    if ( (unsigned int)v11 <= 0x40 )
    {
LABEL_16:
      v12 = *(_QWORD **)(a1 + 256);
      for ( ii = &v12[2 * v11]; ii != v12; v12 += 2 )
        *v12 = -4096;
      *(_QWORD *)(a1 + 264) = 0;
      goto LABEL_19;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 256), 16LL * (unsigned int)v11, 8);
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 264) = 0;
    *(_DWORD *)(a1 + 272) = 0;
  }
LABEL_19:
  v14 = *(_QWORD *)(a1 + 1088);
  v15 = v14 + 40LL * *(unsigned int *)(a1 + 1096);
  while ( v14 != v15 )
  {
    while ( 1 )
    {
      v15 -= 40;
      if ( *(_DWORD *)(v15 + 32) > 0x40u )
      {
        v16 = *(_QWORD *)(v15 + 24);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      if ( *(_DWORD *)(v15 + 16) <= 0x40u )
        break;
      v17 = *(_QWORD *)(v15 + 8);
      if ( !v17 )
        break;
      j_j___libc_free_0_0(v17);
      if ( v14 == v15 )
        goto LABEL_27;
    }
  }
LABEL_27:
  v18 = *(_DWORD *)(a1 + 296);
  ++*(_QWORD *)(a1 + 280);
  *(_DWORD *)(a1 + 1096) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 400) = 0;
  if ( !v18 )
  {
    if ( !*(_DWORD *)(a1 + 300) )
      goto LABEL_33;
    v19 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)v19 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 288), 16LL * (unsigned int)v19, 8);
      *(_QWORD *)(a1 + 288) = 0;
      *(_QWORD *)(a1 + 296) = 0;
      *(_DWORD *)(a1 + 304) = 0;
      goto LABEL_33;
    }
    goto LABEL_30;
  }
  v57 = 4 * v18;
  v19 = *(unsigned int *)(a1 + 304);
  if ( (unsigned int)(4 * v18) < 0x40 )
    v57 = 64;
  if ( (unsigned int)v19 <= v57 )
  {
LABEL_30:
    v20 = *(_QWORD **)(a1 + 288);
    for ( jj = &v20[2 * v19]; jj != v20; v20 += 2 )
      *v20 = -4096;
    *(_QWORD *)(a1 + 296) = 0;
    goto LABEL_33;
  }
  v58 = v18 - 1;
  if ( v58 )
  {
    _BitScanReverse(&v58, v58);
    v59 = *(_QWORD **)(a1 + 288);
    v60 = 1 << (33 - (v58 ^ 0x1F));
    if ( v60 < 64 )
      v60 = 64;
    if ( v60 == (_DWORD)v19 )
    {
      *(_QWORD *)(a1 + 296) = 0;
      v61 = &v59[2 * (unsigned int)v60];
      do
      {
        if ( v59 )
          *v59 = -4096;
        v59 += 2;
      }
      while ( v61 != v59 );
      goto LABEL_33;
    }
  }
  else
  {
    v59 = *(_QWORD **)(a1 + 288);
    v60 = 64;
  }
  sub_C7D6A0((__int64)v59, 16LL * (unsigned int)v19, 8);
  v113 = ((((((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
            | (4 * v60 / 3u + 1)
            | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
          | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
          | (4 * v60 / 3u + 1)
          | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
          | (4 * v60 / 3u + 1)
          | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
        | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
        | (4 * v60 / 3u + 1)
        | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 16;
  v114 = (v113
        | (((((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
            | (4 * v60 / 3u + 1)
            | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
          | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
          | (4 * v60 / 3u + 1)
          | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
          | (4 * v60 / 3u + 1)
          | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
        | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
        | (4 * v60 / 3u + 1)
        | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 304) = v114;
  v115 = (_QWORD *)sub_C7D670(16 * v114, 8);
  v116 = *(unsigned int *)(a1 + 304);
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 288) = v115;
  for ( kk = &v115[2 * v116]; kk != v115; v115 += 2 )
  {
    if ( v115 )
      *v115 = -4096;
  }
LABEL_33:
  v22 = *(_DWORD *)(a1 + 480);
  ++*(_QWORD *)(a1 + 464);
  if ( !v22 )
  {
    v23 = *(unsigned int *)(a1 + 484);
    if ( !(_DWORD)v23 )
      goto LABEL_39;
    v24 = *(unsigned int *)(a1 + 488);
    if ( (unsigned int)v24 > 0x40 )
    {
      v23 = 8 * v24;
      sub_C7D6A0(*(_QWORD *)(a1 + 472), 8 * v24, 4);
      *(_QWORD *)(a1 + 472) = 0;
      *(_QWORD *)(a1 + 480) = 0;
      *(_DWORD *)(a1 + 488) = 0;
      goto LABEL_39;
    }
    goto LABEL_36;
  }
  v52 = 4 * v22;
  v23 = 64;
  v24 = *(unsigned int *)(a1 + 488);
  if ( (unsigned int)(4 * v22) < 0x40 )
    v52 = 64;
  if ( v52 >= (unsigned int)v24 )
  {
LABEL_36:
    v25 = *(_DWORD **)(a1 + 472);
    for ( mm = &v25[2 * v24]; mm != v25; v25 += 2 )
      *v25 = -1;
    *(_QWORD *)(a1 + 480) = 0;
    goto LABEL_39;
  }
  v53 = v22 - 1;
  if ( v53 )
  {
    _BitScanReverse(&v53, v53);
    v54 = *(_DWORD **)(a1 + 472);
    v55 = (unsigned int)(1 << (33 - (v53 ^ 0x1F)));
    if ( (int)v55 < 64 )
      v55 = 64;
    if ( (_DWORD)v55 == (_DWORD)v24 )
    {
      *(_QWORD *)(a1 + 480) = 0;
      v56 = &v54[2 * v55];
      do
      {
        if ( v54 )
          *v54 = -1;
        v54 += 2;
      }
      while ( v56 != v54 );
      goto LABEL_39;
    }
  }
  else
  {
    v54 = *(_DWORD **)(a1 + 472);
    LODWORD(v55) = 64;
  }
  sub_C7D6A0((__int64)v54, 8 * v24, 4);
  v23 = 4;
  v103 = ((((((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v55 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v55 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 8)
        | (((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v55 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
        | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v55 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 16;
  v104 = (v103
        | (((((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v55 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v55 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 8)
        | (((((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v55 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 4)
        | (((4 * (int)v55 / 3u + 1) | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v55 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v55 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 488) = v104;
  v105 = (_DWORD *)sub_C7D670(8 * v104, 4);
  v106 = *(unsigned int *)(a1 + 488);
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 472) = v105;
  for ( nn = &v105[2 * v106]; nn != v105; v105 += 2 )
  {
    if ( v105 )
      *v105 = -1;
  }
LABEL_39:
  v27 = *(_DWORD *)(a1 + 512);
  ++*(_QWORD *)(a1 + 496);
  if ( !v27 )
  {
    if ( !*(_DWORD *)(a1 + 516) )
      goto LABEL_45;
    v28 = *(unsigned int *)(a1 + 520);
    if ( (unsigned int)v28 > 0x40 )
    {
      v23 = 4 * v28;
      sub_C7D6A0(*(_QWORD *)(a1 + 504), 4 * v28, 4);
      *(_QWORD *)(a1 + 504) = 0;
      *(_QWORD *)(a1 + 512) = 0;
      *(_DWORD *)(a1 + 520) = 0;
      goto LABEL_45;
    }
    goto LABEL_42;
  }
  v47 = 4 * v27;
  v23 = 64;
  v28 = *(unsigned int *)(a1 + 520);
  if ( (unsigned int)(4 * v27) < 0x40 )
    v47 = 64;
  if ( (unsigned int)v28 <= v47 )
  {
LABEL_42:
    v29 = 4 * v28;
    if ( v29 )
    {
      v23 = 255;
      memset(*(void **)(a1 + 504), 255, v29);
    }
    *(_QWORD *)(a1 + 512) = 0;
    goto LABEL_45;
  }
  v48 = v27 - 1;
  if ( v48 )
  {
    _BitScanReverse(&v48, v48);
    v49 = *(_DWORD **)(a1 + 504);
    v50 = (unsigned int)(1 << (33 - (v48 ^ 0x1F)));
    if ( (int)v50 < 64 )
      v50 = 64;
    if ( (_DWORD)v50 == (_DWORD)v28 )
    {
      *(_QWORD *)(a1 + 512) = 0;
      v51 = &v49[v50];
      do
      {
        if ( v49 )
          *v49 = -1;
        ++v49;
      }
      while ( v51 != v49 );
      goto LABEL_45;
    }
  }
  else
  {
    v49 = *(_DWORD **)(a1 + 504);
    LODWORD(v50) = 64;
  }
  sub_C7D6A0((__int64)v49, 4 * v28, 4);
  v23 = 4;
  v108 = ((((((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v50 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v50 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 8)
        | (((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v50 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
        | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v50 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 16;
  v109 = (v108
        | (((((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v50 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v50 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 8)
        | (((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v50 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
        | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v50 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 520) = v109;
  v110 = (_DWORD *)sub_C7D670(4 * v109, 4);
  v111 = *(unsigned int *)(a1 + 520);
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 504) = v110;
  for ( i1 = &v110[v111]; i1 != v110; ++v110 )
  {
    if ( v110 )
      *v110 = -1;
  }
LABEL_45:
  v30 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  *(_DWORD *)(a1 + 536) = 0;
  if ( !v30 && !*(_DWORD *)(a1 + 236) )
    goto LABEL_58;
  v31 = *(_QWORD *)(a1 + 224);
  v32 = 4 * v30;
  v33 = 40LL * *(unsigned int *)(a1 + 240);
  if ( (unsigned int)(4 * v30) < 0x40 )
    v32 = 64;
  v34 = v31 + v33;
  if ( *(_DWORD *)(a1 + 240) <= v32 )
  {
    while ( v31 != v34 )
    {
      if ( *(_QWORD *)v31 != -4096 )
      {
        if ( *(_QWORD *)v31 != -8192 )
        {
          v23 = 16LL * *(unsigned int *)(v31 + 32);
          sub_C7D6A0(*(_QWORD *)(v31 + 16), v23, 8);
        }
        *(_QWORD *)v31 = -4096;
      }
      v31 += 40;
    }
    goto LABEL_57;
  }
  do
  {
    if ( *(_QWORD *)v31 != -8192 && *(_QWORD *)v31 != -4096 )
    {
      v23 = 16LL * *(unsigned int *)(v31 + 32);
      sub_C7D6A0(*(_QWORD *)(v31 + 16), v23, 8);
    }
    v31 += 40;
  }
  while ( v31 != v34 );
  v89 = *(_DWORD *)(a1 + 240);
  if ( !v30 )
  {
    if ( v89 )
    {
      v23 = v33;
      sub_C7D6A0(*(_QWORD *)(a1 + 224), v33, 8);
      *(_QWORD *)(a1 + 224) = 0;
      *(_QWORD *)(a1 + 232) = 0;
      *(_DWORD *)(a1 + 240) = 0;
      goto LABEL_58;
    }
LABEL_57:
    *(_QWORD *)(a1 + 232) = 0;
    goto LABEL_58;
  }
  v90 = 64;
  v91 = v30 - 1;
  if ( v91 )
  {
    _BitScanReverse(&v92, v91);
    v90 = (unsigned int)(1 << (33 - (v92 ^ 0x1F)));
    if ( (int)v90 < 64 )
      v90 = 64;
  }
  v93 = *(_QWORD **)(a1 + 224);
  if ( (_DWORD)v90 == v89 )
  {
    *(_QWORD *)(a1 + 232) = 0;
    v118 = &v93[5 * v90];
    do
    {
      if ( v93 )
        *v93 = -4096;
      v93 += 5;
    }
    while ( v118 != v93 );
  }
  else
  {
    sub_C7D6A0((__int64)v93, v33, 8);
    v23 = 8;
    v94 = ((((((((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v90 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v90 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v90 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v90 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 16;
    v95 = (v94
         | (((((((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v90 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v90 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v90 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v90 / 3u + 1) | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v90 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v90 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 240) = v95;
    v96 = (_QWORD *)sub_C7D670(40 * v95, 8);
    v97 = *(unsigned int *)(a1 + 240);
    *(_QWORD *)(a1 + 232) = 0;
    *(_QWORD *)(a1 + 224) = v96;
    for ( i2 = &v96[5 * v97]; i2 != v96; v96 += 5 )
    {
      if ( v96 )
        *v96 = -4096;
    }
  }
LABEL_58:
  v35 = *(_DWORD *)(a1 + 776);
  ++*(_QWORD *)(a1 + 760);
  if ( !v35 )
  {
    result = *(unsigned int *)(a1 + 780);
    if ( !(_DWORD)result )
      goto LABEL_64;
    v37 = *(unsigned int *)(a1 + 784);
    if ( (unsigned int)v37 > 0x40 )
    {
      v23 = 16LL * (unsigned int)v37;
      result = sub_C7D6A0(*(_QWORD *)(a1 + 768), v23, 8);
      *(_QWORD *)(a1 + 768) = 0;
      *(_QWORD *)(a1 + 776) = 0;
      *(_DWORD *)(a1 + 784) = 0;
      goto LABEL_64;
    }
    goto LABEL_61;
  }
  v43 = 4 * v35;
  v23 = 64;
  v37 = *(unsigned int *)(a1 + 784);
  if ( (unsigned int)(4 * v35) < 0x40 )
    v43 = 64;
  if ( v43 >= (unsigned int)v37 )
  {
LABEL_61:
    result = *(_QWORD *)(a1 + 768);
    for ( i3 = result + 16 * v37; i3 != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 776) = 0;
    goto LABEL_64;
  }
  v44 = v35 - 1;
  if ( v44 )
  {
    _BitScanReverse(&v44, v44);
    v45 = *(_QWORD **)(a1 + 768);
    v46 = 1 << (33 - (v44 ^ 0x1F));
    if ( v46 < 64 )
      v46 = 64;
    if ( v46 == (_DWORD)v37 )
    {
      *(_QWORD *)(a1 + 776) = 0;
      result = (__int64)&v45[2 * (unsigned int)v46];
      do
      {
        if ( v45 )
          *v45 = -4096;
        v45 += 2;
      }
      while ( (_QWORD *)result != v45 );
      goto LABEL_64;
    }
  }
  else
  {
    v45 = *(_QWORD **)(a1 + 768);
    v46 = 64;
  }
  sub_C7D6A0((__int64)v45, 16LL * (unsigned int)v37, 8);
  v23 = 8;
  v99 = ((((((((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
           | (4 * v46 / 3u + 1)
           | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 4)
         | (((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
         | (4 * v46 / 3u + 1)
         | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
         | (4 * v46 / 3u + 1)
         | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 4)
       | (((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
       | (4 * v46 / 3u + 1)
       | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 16;
  v100 = (v99
        | (((((((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
            | (4 * v46 / 3u + 1)
            | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 4)
          | (((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
          | (4 * v46 / 3u + 1)
          | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
          | (4 * v46 / 3u + 1)
          | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 4)
        | (((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
        | (4 * v46 / 3u + 1)
        | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 784) = v100;
  result = sub_C7D670(16 * v100, 8);
  v101 = *(unsigned int *)(a1 + 784);
  *(_QWORD *)(a1 + 776) = 0;
  *(_QWORD *)(a1 + 768) = result;
  for ( i4 = result + 16 * v101; i4 != result; result += 16 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
LABEL_64:
  ++*(_QWORD *)(a1 + 896);
  if ( *(_BYTE *)(a1 + 924) )
  {
LABEL_69:
    *(_QWORD *)(a1 + 916) = 0;
    goto LABEL_70;
  }
  v39 = 4 * (*(_DWORD *)(a1 + 916) - *(_DWORD *)(a1 + 920));
  v40 = *(unsigned int *)(a1 + 912);
  if ( v39 < 0x20 )
    v39 = 32;
  if ( (unsigned int)v40 <= v39 )
  {
    v23 = 0xFFFFFFFFLL;
    result = (__int64)memset(*(void **)(a1 + 904), -1, 8 * v40);
    goto LABEL_69;
  }
  result = (__int64)sub_C8C990(a1 + 896, v23);
LABEL_70:
  ++*(_QWORD *)(a1 + 992);
  if ( *(_BYTE *)(a1 + 1020) )
  {
LABEL_75:
    *(_QWORD *)(a1 + 1012) = 0;
    return result;
  }
  v41 = 4 * (*(_DWORD *)(a1 + 1012) - *(_DWORD *)(a1 + 1016));
  v42 = *(unsigned int *)(a1 + 1008);
  if ( v41 < 0x20 )
    v41 = 32;
  if ( (unsigned int)v42 <= v41 )
  {
    result = (__int64)memset(*(void **)(a1 + 1000), -1, 8 * v42);
    goto LABEL_75;
  }
  return (__int64)sub_C8C990(a1 + 992, v23);
}
