// Function: sub_31FA4C0
// Address: 0x31fa4c0
//
__int64 __fastcall sub_31FA4C0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *k; // rdx
  _QWORD *v10; // rdi
  unsigned __int64 *v11; // r13
  unsigned __int64 *v12; // r14
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // r13
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r12
  int v17; // eax
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *n; // rdx
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *jj; // rdx
  int v25; // r15d
  unsigned int v26; // eax
  _QWORD *v27; // r12
  __int64 v28; // rdx
  __int64 v29; // r14
  _QWORD *v30; // r13
  unsigned __int64 *v31; // r14
  int v32; // eax
  __int64 result; // rax
  __int64 v34; // rdx
  __int64 nn; // rdx
  unsigned int v36; // ecx
  unsigned int v37; // eax
  _QWORD *v38; // rdi
  int v39; // r12d
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 i1; // rdx
  unsigned int v44; // ecx
  unsigned int v45; // eax
  _QWORD *v46; // rdi
  int v47; // r12d
  _QWORD *v48; // rax
  unsigned int v49; // ecx
  unsigned int v50; // eax
  _QWORD *v51; // rdi
  int v52; // r12d
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rdx
  _QWORD *ii; // rdx
  unsigned int v58; // ecx
  unsigned int v59; // eax
  _QWORD *v60; // rdi
  int v61; // r12d
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rdi
  _QWORD *v64; // rax
  __int64 v65; // rdx
  _QWORD *m; // rdx
  unsigned int v67; // ecx
  unsigned int v68; // eax
  _QWORD *v69; // rdi
  int v70; // r12d
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // rax
  _QWORD *v73; // rax
  __int64 v74; // rdx
  _QWORD *j; // rdx
  unsigned __int64 *v76; // r8
  unsigned int v77; // edx
  int v78; // r12d
  unsigned int v79; // r15d
  unsigned int v80; // eax
  _QWORD *v81; // rdi
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rdi
  _QWORD *v84; // rax
  __int64 v85; // rdx
  _QWORD *mm; // rdx
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // rdi
  _QWORD *v89; // rax
  __int64 v90; // rdx
  _QWORD *kk; // rdx
  _QWORD *v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // rax
  _QWORD *v95; // rax
  unsigned __int64 *v96; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 1120);
  ++*(_QWORD *)(a1 + 1104);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 1124) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 1128);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1112), 24 * v3, 8);
      *(_QWORD *)(a1 + 1112) = 0;
      *(_QWORD *)(a1 + 1120) = 0;
      *(_DWORD *)(a1 + 1128) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v67 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 1128);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v67 = 64;
  if ( (unsigned int)v3 <= v67 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 1112);
    for ( i = &v4[3 * v3]; i != v4; *(v4 - 2) = 0 )
    {
      *v4 = -1;
      v4 += 3;
    }
    *(_QWORD *)(a1 + 1120) = 0;
    goto LABEL_7;
  }
  v68 = v2 - 1;
  if ( !v68 )
  {
    v69 = *(_QWORD **)(a1 + 1112);
    v70 = 64;
LABEL_114:
    sub_C7D6A0((__int64)v69, 24 * v3, 8);
    v71 = ((((((((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
             | (4 * v70 / 3u + 1)
             | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 4)
           | (((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
           | (4 * v70 / 3u + 1)
           | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
           | (4 * v70 / 3u + 1)
           | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 4)
         | (((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
         | (4 * v70 / 3u + 1)
         | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 16;
    v72 = (v71
         | (((((((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
             | (4 * v70 / 3u + 1)
             | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 4)
           | (((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
           | (4 * v70 / 3u + 1)
           | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
           | (4 * v70 / 3u + 1)
           | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 4)
         | (((4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1)) >> 2)
         | (4 * v70 / 3u + 1)
         | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 1128) = v72;
    v73 = (_QWORD *)sub_C7D670(24 * v72, 8);
    v74 = *(unsigned int *)(a1 + 1128);
    *(_QWORD *)(a1 + 1120) = 0;
    *(_QWORD *)(a1 + 1112) = v73;
    for ( j = &v73[3 * v74]; j != v73; v73 += 3 )
    {
      if ( v73 )
      {
        *v73 = -1;
        v73[1] = 0;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v68, v68);
  v69 = *(_QWORD **)(a1 + 1112);
  v70 = 1 << (33 - (v68 ^ 0x1F));
  if ( v70 < 64 )
    v70 = 64;
  if ( (_DWORD)v3 != v70 )
    goto LABEL_114;
  *(_QWORD *)(a1 + 1120) = 0;
  v93 = &v69[3 * v3];
  do
  {
    if ( v69 )
    {
      *v69 = -1;
      v69[1] = 0;
    }
    v69 += 3;
  }
  while ( v93 != v69 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 1072);
  ++*(_QWORD *)(a1 + 1056);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 1076) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 1080);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1064), 16LL * (unsigned int)v7, 8);
      *(_QWORD *)(a1 + 1064) = 0;
      *(_QWORD *)(a1 + 1072) = 0;
      *(_DWORD *)(a1 + 1080) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v58 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 1080);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v58 = 64;
  if ( v58 >= (unsigned int)v7 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 1064);
    for ( k = &v8[2 * v7]; k != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 1072) = 0;
    goto LABEL_13;
  }
  v59 = v6 - 1;
  if ( !v59 )
  {
    v60 = *(_QWORD **)(a1 + 1064);
    v61 = 64;
LABEL_102:
    sub_C7D6A0((__int64)v60, 16LL * (unsigned int)v7, 8);
    v62 = ((((((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
             | (4 * v61 / 3u + 1)
             | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
           | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
           | (4 * v61 / 3u + 1)
           | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
           | (4 * v61 / 3u + 1)
           | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
         | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
         | (4 * v61 / 3u + 1)
         | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 16;
    v63 = (v62
         | (((((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
             | (4 * v61 / 3u + 1)
             | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
           | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
           | (4 * v61 / 3u + 1)
           | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
           | (4 * v61 / 3u + 1)
           | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
         | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
         | (4 * v61 / 3u + 1)
         | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 1080) = v63;
    v64 = (_QWORD *)sub_C7D670(16 * v63, 8);
    v65 = *(unsigned int *)(a1 + 1080);
    *(_QWORD *)(a1 + 1072) = 0;
    *(_QWORD *)(a1 + 1064) = v64;
    for ( m = &v64[2 * v65]; m != v64; v64 += 2 )
    {
      if ( v64 )
        *v64 = -4096;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v59, v59);
  v60 = *(_QWORD **)(a1 + 1064);
  v61 = 1 << (33 - (v59 ^ 0x1F));
  if ( v61 < 64 )
    v61 = 64;
  if ( v61 != (_DWORD)v7 )
    goto LABEL_102;
  *(_QWORD *)(a1 + 1072) = 0;
  v95 = &v60[2 * (unsigned int)v61];
  do
  {
    if ( v60 )
      *v60 = -4096;
    v60 += 2;
  }
  while ( v95 != v60 );
LABEL_13:
  sub_31F9F70(*(_QWORD *)(a1 + 1088), *(_QWORD *)(a1 + 1088) + 16LL * *(unsigned int *)(a1 + 1096));
  v10 = *(_QWORD **)(a1 + 1408);
  *(_DWORD *)(a1 + 1096) = 0;
  sub_31F5340(v10);
  v11 = *(unsigned __int64 **)(a1 + 1344);
  v12 = *(unsigned __int64 **)(a1 + 1352);
  *(_QWORD *)(a1 + 1408) = 0;
  *(_QWORD *)(a1 + 1416) = a1 + 1400;
  *(_QWORD *)(a1 + 1424) = a1 + 1400;
  *(_QWORD *)(a1 + 1432) = 0;
  if ( v11 != v12 )
  {
    v13 = v11;
    do
    {
      if ( (unsigned __int64 *)*v13 != v13 + 2 )
        j_j___libc_free_0(*v13);
      v13 += 5;
    }
    while ( v12 != v13 );
    *(_QWORD *)(a1 + 1352) = v11;
  }
  v14 = *(unsigned __int64 **)(a1 + 1368);
  v15 = *(unsigned __int64 **)(a1 + 1376);
  if ( v14 != v15 )
  {
    v16 = *(unsigned __int64 **)(a1 + 1368);
    do
    {
      if ( (unsigned __int64 *)*v16 != v16 + 2 )
        j_j___libc_free_0(*v16);
      v16 += 5;
    }
    while ( v15 != v16 );
    *(_QWORD *)(a1 + 1376) = v14;
  }
  v17 = *(_DWORD *)(a1 + 1232);
  ++*(_QWORD *)(a1 + 1216);
  if ( !v17 )
  {
    if ( !*(_DWORD *)(a1 + 1236) )
      goto LABEL_31;
    v18 = *(unsigned int *)(a1 + 1240);
    if ( (unsigned int)v18 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1224), 24 * v18, 8);
      *(_QWORD *)(a1 + 1224) = 0;
      *(_QWORD *)(a1 + 1232) = 0;
      *(_DWORD *)(a1 + 1240) = 0;
      goto LABEL_31;
    }
    goto LABEL_28;
  }
  v49 = 4 * v17;
  v18 = *(unsigned int *)(a1 + 1240);
  if ( (unsigned int)(4 * v17) < 0x40 )
    v49 = 64;
  if ( (unsigned int)v18 <= v49 )
  {
LABEL_28:
    v19 = *(_QWORD **)(a1 + 1224);
    for ( n = &v19[3 * v18]; n != v19; *(v19 - 2) = -4096 )
    {
      *v19 = -4096;
      v19 += 3;
    }
    *(_QWORD *)(a1 + 1232) = 0;
    goto LABEL_31;
  }
  v50 = v17 - 1;
  if ( !v50 )
  {
    v51 = *(_QWORD **)(a1 + 1224);
    v52 = 64;
LABEL_90:
    sub_C7D6A0((__int64)v51, 24 * v18, 8);
    v53 = ((((((((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
             | (4 * v52 / 3u + 1)
             | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 4)
           | (((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
           | (4 * v52 / 3u + 1)
           | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
           | (4 * v52 / 3u + 1)
           | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 4)
         | (((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
         | (4 * v52 / 3u + 1)
         | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 16;
    v54 = (v53
         | (((((((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
             | (4 * v52 / 3u + 1)
             | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 4)
           | (((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
           | (4 * v52 / 3u + 1)
           | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
           | (4 * v52 / 3u + 1)
           | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 4)
         | (((4 * v52 / 3u + 1) | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1)) >> 2)
         | (4 * v52 / 3u + 1)
         | ((unsigned __int64)(4 * v52 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 1240) = v54;
    v55 = (_QWORD *)sub_C7D670(24 * v54, 8);
    v56 = *(unsigned int *)(a1 + 1240);
    *(_QWORD *)(a1 + 1232) = 0;
    *(_QWORD *)(a1 + 1224) = v55;
    for ( ii = &v55[3 * v56]; ii != v55; v55 += 3 )
    {
      if ( v55 )
      {
        *v55 = -4096;
        v55[1] = -4096;
      }
    }
    goto LABEL_31;
  }
  _BitScanReverse(&v50, v50);
  v51 = *(_QWORD **)(a1 + 1224);
  v52 = 1 << (33 - (v50 ^ 0x1F));
  if ( v52 < 64 )
    v52 = 64;
  if ( (_DWORD)v18 != v52 )
    goto LABEL_90;
  *(_QWORD *)(a1 + 1232) = 0;
  v94 = &v51[3 * v18];
  do
  {
    if ( v51 )
    {
      *v51 = -4096;
      v51[1] = -4096;
    }
    v51 += 3;
  }
  while ( v94 != v51 );
LABEL_31:
  v21 = *(_DWORD *)(a1 + 1264);
  ++*(_QWORD *)(a1 + 1248);
  if ( !v21 )
  {
    if ( !*(_DWORD *)(a1 + 1268) )
      goto LABEL_37;
    v22 = *(unsigned int *)(a1 + 1272);
    if ( (unsigned int)v22 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1256), 16LL * (unsigned int)v22, 8);
      *(_QWORD *)(a1 + 1256) = 0;
      *(_QWORD *)(a1 + 1264) = 0;
      *(_DWORD *)(a1 + 1272) = 0;
      goto LABEL_37;
    }
    goto LABEL_34;
  }
  v44 = 4 * v21;
  v22 = *(unsigned int *)(a1 + 1272);
  if ( (unsigned int)(4 * v21) < 0x40 )
    v44 = 64;
  if ( (unsigned int)v22 <= v44 )
  {
LABEL_34:
    v23 = *(_QWORD **)(a1 + 1256);
    for ( jj = &v23[2 * v22]; jj != v23; v23 += 2 )
      *v23 = -4096;
    *(_QWORD *)(a1 + 1264) = 0;
    goto LABEL_37;
  }
  v45 = v21 - 1;
  if ( v45 )
  {
    _BitScanReverse(&v45, v45);
    v46 = *(_QWORD **)(a1 + 1256);
    v47 = 1 << (33 - (v45 ^ 0x1F));
    if ( v47 < 64 )
      v47 = 64;
    if ( (_DWORD)v22 == v47 )
    {
      *(_QWORD *)(a1 + 1264) = 0;
      v48 = &v46[2 * (unsigned int)v22];
      do
      {
        if ( v46 )
          *v46 = -4096;
        v46 += 2;
      }
      while ( v48 != v46 );
      goto LABEL_37;
    }
  }
  else
  {
    v46 = *(_QWORD **)(a1 + 1256);
    v47 = 64;
  }
  sub_C7D6A0((__int64)v46, 16LL * (unsigned int)v22, 8);
  v87 = ((((((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
         | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
       | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
       | (4 * v47 / 3u + 1)
       | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 16;
  v88 = (v87
       | (((((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
         | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
       | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
       | (4 * v47 / 3u + 1)
       | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 1272) = v88;
  v89 = (_QWORD *)sub_C7D670(16 * v88, 8);
  v90 = *(unsigned int *)(a1 + 1272);
  *(_QWORD *)(a1 + 1264) = 0;
  *(_QWORD *)(a1 + 1256) = v89;
  for ( kk = &v89[2 * v90]; kk != v89; v89 += 2 )
  {
    if ( v89 )
      *v89 = -4096;
  }
LABEL_37:
  v25 = *(_DWORD *)(a1 + 888);
  ++*(_QWORD *)(a1 + 872);
  if ( !v25 && !*(_DWORD *)(a1 + 892) )
    goto LABEL_52;
  v26 = 4 * v25;
  v27 = *(_QWORD **)(a1 + 880);
  v28 = *(unsigned int *)(a1 + 896);
  v29 = 16 * v28;
  if ( (unsigned int)(4 * v25) < 0x40 )
    v26 = 64;
  v30 = &v27[(unsigned __int64)v29 / 8];
  if ( (unsigned int)v28 <= v26 )
  {
    for ( ; v27 != v30; v27 += 2 )
    {
      if ( *v27 != -4096 )
      {
        if ( *v27 != -8192 )
        {
          v31 = (unsigned __int64 *)v27[1];
          if ( v31 )
          {
            if ( (unsigned __int64 *)*v31 != v31 + 2 )
              _libc_free(*v31);
            j_j___libc_free_0((unsigned __int64)v31);
          }
        }
        *v27 = -4096;
      }
    }
    goto LABEL_51;
  }
  do
  {
    while ( *v27 == -4096 )
    {
LABEL_123:
      v27 += 2;
      if ( v27 == v30 )
        goto LABEL_127;
    }
    if ( *v27 != -8192 )
    {
      v76 = (unsigned __int64 *)v27[1];
      if ( v76 )
      {
        if ( (unsigned __int64 *)*v76 != v76 + 2 )
        {
          v96 = (unsigned __int64 *)v27[1];
          _libc_free(*v76);
          v76 = v96;
        }
        j_j___libc_free_0((unsigned __int64)v76);
      }
      goto LABEL_123;
    }
    v27 += 2;
  }
  while ( v27 != v30 );
LABEL_127:
  v77 = *(_DWORD *)(a1 + 896);
  if ( !v25 )
  {
    if ( v77 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 880), v29, 8);
      *(_QWORD *)(a1 + 880) = 0;
      *(_QWORD *)(a1 + 888) = 0;
      *(_DWORD *)(a1 + 896) = 0;
      goto LABEL_52;
    }
LABEL_51:
    *(_QWORD *)(a1 + 888) = 0;
    goto LABEL_52;
  }
  v78 = 64;
  v79 = v25 - 1;
  if ( v79 )
  {
    _BitScanReverse(&v80, v79);
    v78 = 1 << (33 - (v80 ^ 0x1F));
    if ( v78 < 64 )
      v78 = 64;
  }
  v81 = *(_QWORD **)(a1 + 880);
  if ( v77 == v78 )
  {
    *(_QWORD *)(a1 + 888) = 0;
    v92 = &v81[2 * v77];
    do
    {
      if ( v81 )
        *v81 = -4096;
      v81 += 2;
    }
    while ( v92 != v81 );
  }
  else
  {
    sub_C7D6A0((__int64)v81, v29, 8);
    v82 = ((((((((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
             | (4 * v78 / 3u + 1)
             | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 4)
           | (((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
           | (4 * v78 / 3u + 1)
           | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
           | (4 * v78 / 3u + 1)
           | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 4)
         | (((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
         | (4 * v78 / 3u + 1)
         | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 16;
    v83 = (v82
         | (((((((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
             | (4 * v78 / 3u + 1)
             | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 4)
           | (((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
           | (4 * v78 / 3u + 1)
           | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
           | (4 * v78 / 3u + 1)
           | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 4)
         | (((4 * v78 / 3u + 1) | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1)) >> 2)
         | (4 * v78 / 3u + 1)
         | ((unsigned __int64)(4 * v78 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 896) = v83;
    v84 = (_QWORD *)sub_C7D670(16 * v83, 8);
    v85 = *(unsigned int *)(a1 + 896);
    *(_QWORD *)(a1 + 888) = 0;
    *(_QWORD *)(a1 + 880) = v84;
    for ( mm = &v84[2 * v85]; mm != v84; v84 += 2 )
    {
      if ( v84 )
        *v84 = -4096;
    }
  }
LABEL_52:
  v32 = *(_DWORD *)(a1 + 824);
  ++*(_QWORD *)(a1 + 808);
  if ( v32 )
  {
    v36 = 4 * v32;
    v34 = *(unsigned int *)(a1 + 832);
    if ( (unsigned int)(4 * v32) < 0x40 )
      v36 = 64;
    if ( v36 >= (unsigned int)v34 )
    {
LABEL_55:
      result = *(_QWORD *)(a1 + 816);
      for ( nn = result + 16 * v34; nn != result; result += 16 )
        *(_QWORD *)result = -4096;
      *(_QWORD *)(a1 + 824) = 0;
      return result;
    }
    v37 = v32 - 1;
    if ( v37 )
    {
      _BitScanReverse(&v37, v37);
      v38 = *(_QWORD **)(a1 + 816);
      v39 = 1 << (33 - (v37 ^ 0x1F));
      if ( v39 < 64 )
        v39 = 64;
      if ( v39 == (_DWORD)v34 )
      {
        *(_QWORD *)(a1 + 824) = 0;
        result = (__int64)&v38[2 * (unsigned int)v39];
        do
        {
          if ( v38 )
            *v38 = -4096;
          v38 += 2;
        }
        while ( (_QWORD *)result != v38 );
        return result;
      }
    }
    else
    {
      v38 = *(_QWORD **)(a1 + 816);
      v39 = 64;
    }
    sub_C7D6A0((__int64)v38, 16LL * (unsigned int)v34, 8);
    v40 = ((((((((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
             | (4 * v39 / 3u + 1)
             | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 4)
           | (((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
           | (4 * v39 / 3u + 1)
           | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
           | (4 * v39 / 3u + 1)
           | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 4)
         | (((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
         | (4 * v39 / 3u + 1)
         | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 16;
    v41 = (v40
         | (((((((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
             | (4 * v39 / 3u + 1)
             | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 4)
           | (((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
           | (4 * v39 / 3u + 1)
           | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
           | (4 * v39 / 3u + 1)
           | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 4)
         | (((4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1)) >> 2)
         | (4 * v39 / 3u + 1)
         | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 832) = v41;
    result = sub_C7D670(16 * v41, 8);
    v42 = *(unsigned int *)(a1 + 832);
    *(_QWORD *)(a1 + 824) = 0;
    *(_QWORD *)(a1 + 816) = result;
    for ( i1 = result + 16 * v42; i1 != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    return result;
  }
  result = *(unsigned int *)(a1 + 828);
  if ( (_DWORD)result )
  {
    v34 = *(unsigned int *)(a1 + 832);
    if ( (unsigned int)v34 <= 0x40 )
      goto LABEL_55;
    result = sub_C7D6A0(*(_QWORD *)(a1 + 816), 16LL * (unsigned int)v34, 8);
    *(_QWORD *)(a1 + 816) = 0;
    *(_QWORD *)(a1 + 824) = 0;
    *(_DWORD *)(a1 + 832) = 0;
  }
  return result;
}
