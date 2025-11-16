// Function: sub_278EBA0
// Address: 0x278eba0
//
__int64 __fastcall sub_278EBA0(__int64 a1, __int64 a2, __int64 i, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  _QWORD *v8; // rax
  int v9; // r14d
  __int64 v10; // rdi
  __int64 v11; // r13
  _DWORD *v12; // rbx
  _DWORD *v13; // r13
  __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // rdx
  _DWORD *v18; // rax
  _DWORD *j; // rdx
  int v20; // eax
  __int64 v21; // rdx
  _DWORD *v22; // rax
  _DWORD *m; // rdx
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // rbx
  unsigned __int64 v27; // rdi
  __int64 result; // rax
  unsigned int v29; // ecx
  unsigned int v30; // eax
  __int64 v31; // rdi
  int v32; // ebx
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 n; // rdx
  unsigned int v38; // ecx
  unsigned int v39; // eax
  _DWORD *v40; // rdi
  int v41; // ebx
  _DWORD *v42; // rax
  unsigned int v43; // eax
  _QWORD *v44; // rdi
  int v45; // eax
  int v46; // ebx
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdi
  _QWORD *v49; // rax
  __int64 v50; // rdx
  unsigned int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned int v56; // eax
  unsigned int v57; // kr00_4
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // r13
  __int64 v65; // rbx
  __int64 v66; // r13
  __int64 v67; // rcx
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rdi
  _DWORD *v70; // rax
  __int64 v71; // rdx
  _DWORD *k; // rdx
  __int64 v73; // rbx
  __int64 v74; // r13
  __int64 v75; // rax
  _QWORD *v76; // rax
  _BYTE *v77; // [rsp+10h] [rbp-60h] BYREF
  __int64 v78; // [rsp+18h] [rbp-58h]
  _BYTE v79[16]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v80; // [rsp+30h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v7 )
  {
    a6 = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)a6 )
      goto LABEL_7;
    i = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)i > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * (unsigned int)i, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  a4 = (unsigned int)(4 * v7);
  i = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)a4 < 0x40 )
    a4 = 64;
  if ( (unsigned int)i <= (unsigned int)a4 )
  {
LABEL_4:
    v8 = *(_QWORD **)(a1 + 8);
    for ( i = (__int64)&v8[2 * i]; (_QWORD *)i != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v43 = v7 - 1;
  if ( !v43 )
  {
    v44 = *(_QWORD **)(a1 + 8);
    v46 = 64;
LABEL_66:
    sub_C7D6A0((__int64)v44, 16LL * (unsigned int)i, 8);
    v47 = ((((((((4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1)) >> 2)
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
    v48 = (v47
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
    *(_DWORD *)(a1 + 24) = v48;
    v49 = (_QWORD *)sub_C7D670(16 * v48, 8);
    v50 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v49;
    for ( i = (__int64)&v49[2 * v50]; (_QWORD *)i != v49; v49 += 2 )
    {
      if ( v49 )
        *v49 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v43, v43);
  v44 = *(_QWORD **)(a1 + 8);
  v45 = v43 ^ 0x1F;
  a4 = (unsigned int)(33 - v45);
  v46 = 1 << (33 - v45);
  if ( v46 < 64 )
    v46 = 64;
  if ( (_DWORD)i != v46 )
    goto LABEL_66;
  *(_QWORD *)(a1 + 16) = 0;
  v76 = &v44[2 * (unsigned int)i];
  do
  {
    if ( v44 )
      *v44 = -4096;
    v44 += 2;
  }
  while ( v76 != v44 );
LABEL_7:
  v9 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v10 = a1 + 32;
  if ( v9 )
  {
    v51 = 4 * v9;
    v11 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)(4 * v9) < 0x40 )
      v51 = 64;
    if ( v51 >= (unsigned int)v11 )
      goto LABEL_10;
    sub_278E480(v10);
    v56 = v9 - 1;
    v9 = 64;
    if ( v56 )
    {
      _BitScanReverse(&v56, v56);
      v53 = 33 - (v56 ^ 0x1F);
      v9 = 1 << (33 - (v56 ^ 0x1F));
      if ( v9 < 64 )
        v9 = 64;
    }
  }
  else
  {
    a5 = *(unsigned int *)(a1 + 52);
    if ( !(_DWORD)a5 )
      goto LABEL_14;
    v11 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v11 <= 0x40 )
    {
LABEL_10:
      v12 = *(_DWORD **)(a1 + 40);
      v13 = &v12[16 * v11];
      v77 = v79;
      v78 = 0x400000000LL;
      v80 = 0;
      if ( v12 == v13 )
      {
        *(_QWORD *)(a1 + 48) = 0;
      }
      else
      {
        do
        {
          *v12 = -1;
          v14 = (__int64)(v12 + 4);
          v12 += 16;
          *((_BYTE *)v12 - 60) = 0;
          *((_QWORD *)v12 - 7) = 0;
          sub_2789770(v14, (__int64)&v77, i, a4, a5, a6);
          *((_QWORD *)v12 - 2) = v80;
        }
        while ( v13 != v12 );
        *(_QWORD *)(a1 + 48) = 0;
        v15 = (unsigned __int64)v77;
        if ( v77 != v79 )
          goto LABEL_13;
      }
      goto LABEL_14;
    }
    sub_278E480(v10);
  }
  if ( *(_DWORD *)(a1 + 56) == v9 )
  {
    v73 = *(_QWORD *)(a1 + 40);
    v77 = v79;
    v74 = v73 + ((unsigned __int64)(unsigned int)v9 << 6);
    *(_QWORD *)(a1 + 48) = 0;
    v78 = 0x400000000LL;
    v80 = 0;
    if ( v73 == v74 )
      goto LABEL_14;
    do
    {
      if ( v73 )
      {
        *(_DWORD *)v73 = -1;
        *(_BYTE *)(v73 + 4) = 0;
        *(_DWORD *)(v73 + 24) = 0;
        *(_QWORD *)(v73 + 8) = 0;
        *(_QWORD *)(v73 + 16) = v73 + 32;
        *(_DWORD *)(v73 + 28) = 4;
        if ( (_DWORD)v78 )
          sub_2789770(v73 + 16, (__int64)&v77, v52, v53, v54, v55);
        *(_QWORD *)(v73 + 48) = v80;
      }
      v73 += 64;
    }
    while ( v74 != v73 );
    goto LABEL_98;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), (unsigned __int64)(unsigned int)v11 << 6, 8);
  if ( !v9 )
  {
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 48) = 0;
    *(_DWORD *)(a1 + 56) = 0;
    goto LABEL_14;
  }
  v57 = 4 * v9;
  v58 = ((((((((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
           | (v57 / 3 + 1)
           | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 4)
         | (((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
         | (v57 / 3 + 1)
         | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 8)
       | (((((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
         | (v57 / 3 + 1)
         | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 4)
       | (((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
       | (v57 / 3 + 1)
       | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 16;
  v59 = (v58
       | (((((((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
           | (v57 / 3 + 1)
           | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 4)
         | (((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
         | (v57 / 3 + 1)
         | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 8)
       | (((((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
         | (v57 / 3 + 1)
         | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 4)
       | (((v57 / 3 + 1) | ((unsigned __int64)(v57 / 3 + 1) >> 1)) >> 2)
       | (v57 / 3 + 1)
       | ((unsigned __int64)(v57 / 3 + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 56) = v59;
  v60 = sub_C7D670(v59 << 6, 8);
  v64 = *(unsigned int *)(a1 + 56);
  *(_QWORD *)(a1 + 48) = 0;
  v65 = v60;
  *(_QWORD *)(a1 + 40) = v60;
  v66 = v60 + (v64 << 6);
  v77 = v79;
  v78 = 0x400000000LL;
  v80 = 0;
  if ( v60 != v66 )
  {
    do
    {
      if ( v65 )
      {
        v67 = (unsigned int)v78;
        *(_DWORD *)(v65 + 24) = 0;
        *(_DWORD *)(v65 + 28) = 4;
        *(_DWORD *)v65 = -1;
        *(_BYTE *)(v65 + 4) = 0;
        *(_QWORD *)(v65 + 8) = 0;
        *(_QWORD *)(v65 + 16) = v65 + 32;
        if ( (_DWORD)v67 )
          sub_2789770(v65 + 16, (__int64)&v77, v61, v67, v62, v63);
        *(_QWORD *)(v65 + 48) = v80;
      }
      v65 += 64;
    }
    while ( v66 != v65 );
LABEL_98:
    v15 = (unsigned __int64)v77;
    if ( v77 != v79 )
LABEL_13:
      _libc_free(v15);
  }
LABEL_14:
  v16 = *(_DWORD *)(a1 + 136);
  ++*(_QWORD *)(a1 + 120);
  if ( !v16 )
  {
    if ( !*(_DWORD *)(a1 + 140) )
      goto LABEL_20;
    v17 = *(unsigned int *)(a1 + 144);
    if ( (unsigned int)v17 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * (unsigned int)v17, 8);
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 136) = 0;
      *(_DWORD *)(a1 + 144) = 0;
      goto LABEL_20;
    }
    goto LABEL_17;
  }
  v38 = 4 * v16;
  v17 = *(unsigned int *)(a1 + 144);
  if ( (unsigned int)(4 * v16) < 0x40 )
    v38 = 64;
  if ( (unsigned int)v17 <= v38 )
  {
LABEL_17:
    v18 = *(_DWORD **)(a1 + 128);
    for ( j = &v18[4 * v17]; j != v18; v18 += 4 )
      *v18 = -1;
    *(_QWORD *)(a1 + 136) = 0;
    goto LABEL_20;
  }
  v39 = v16 - 1;
  if ( v39 )
  {
    _BitScanReverse(&v39, v39);
    v40 = *(_DWORD **)(a1 + 128);
    v41 = 1 << (33 - (v39 ^ 0x1F));
    if ( v41 < 64 )
      v41 = 64;
    if ( (_DWORD)v17 == v41 )
    {
      *(_QWORD *)(a1 + 136) = 0;
      v42 = &v40[4 * (unsigned int)v17];
      do
      {
        if ( v40 )
          *v40 = -1;
        v40 += 4;
      }
      while ( v42 != v40 );
      goto LABEL_20;
    }
  }
  else
  {
    v40 = *(_DWORD **)(a1 + 128);
    v41 = 64;
  }
  sub_C7D6A0((__int64)v40, 16LL * (unsigned int)v17, 8);
  v68 = ((((((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
           | (4 * v41 / 3u + 1)
           | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
         | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
         | (4 * v41 / 3u + 1)
         | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
         | (4 * v41 / 3u + 1)
         | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
       | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
       | (4 * v41 / 3u + 1)
       | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 16;
  v69 = (v68
       | (((((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
           | (4 * v41 / 3u + 1)
           | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
         | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
         | (4 * v41 / 3u + 1)
         | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
         | (4 * v41 / 3u + 1)
         | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4)
       | (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
       | (4 * v41 / 3u + 1)
       | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 144) = v69;
  v70 = (_DWORD *)sub_C7D670(16 * v69, 8);
  v71 = *(unsigned int *)(a1 + 144);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 128) = v70;
  for ( k = &v70[4 * v71]; k != v70; v70 += 4 )
  {
    if ( v70 )
      *v70 = -1;
  }
LABEL_20:
  v20 = *(_DWORD *)(a1 + 168);
  ++*(_QWORD *)(a1 + 152);
  if ( !v20 )
  {
    if ( !*(_DWORD *)(a1 + 172) )
      goto LABEL_26;
    v21 = *(unsigned int *)(a1 + 176);
    if ( (unsigned int)v21 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 160), 24 * v21, 8);
      *(_QWORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 168) = 0;
      *(_DWORD *)(a1 + 176) = 0;
      goto LABEL_26;
    }
    goto LABEL_23;
  }
  v29 = 4 * v20;
  v21 = *(unsigned int *)(a1 + 176);
  if ( (unsigned int)(4 * v20) < 0x40 )
    v29 = 64;
  if ( v29 >= (unsigned int)v21 )
  {
LABEL_23:
    v22 = *(_DWORD **)(a1 + 160);
    for ( m = &v22[6 * v21]; m != v22; *((_QWORD *)v22 - 2) = -4096 )
    {
      *v22 = -1;
      v22 += 6;
    }
    *(_QWORD *)(a1 + 168) = 0;
    goto LABEL_26;
  }
  v30 = v20 - 1;
  if ( !v30 )
  {
    v31 = *(_QWORD *)(a1 + 160);
    v32 = 64;
LABEL_42:
    sub_C7D6A0(v31, 24 * v21, 8);
    v33 = ((((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 16;
    v34 = (v33
         | (((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 176) = v34;
    v35 = sub_C7D670(24 * v34, 8);
    v36 = *(unsigned int *)(a1 + 176);
    *(_QWORD *)(a1 + 168) = 0;
    *(_QWORD *)(a1 + 160) = v35;
    for ( n = v35 + 24 * v36; n != v35; v35 += 24 )
    {
      if ( v35 )
      {
        *(_DWORD *)v35 = -1;
        *(_QWORD *)(v35 + 8) = -4096;
      }
    }
    goto LABEL_26;
  }
  _BitScanReverse(&v30, v30);
  v31 = *(_QWORD *)(a1 + 160);
  v32 = 1 << (33 - (v30 ^ 0x1F));
  if ( v32 < 64 )
    v32 = 64;
  if ( (_DWORD)v21 != v32 )
    goto LABEL_42;
  *(_QWORD *)(a1 + 168) = 0;
  v75 = v31 + 24 * v21;
  do
  {
    if ( v31 )
    {
      *(_DWORD *)v31 = -1;
      *(_QWORD *)(v31 + 8) = -4096;
    }
    v31 += 24;
  }
  while ( v75 != v31 );
LABEL_26:
  v24 = *(_QWORD *)(a1 + 72);
  v25 = *(_QWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 208) = 1;
  if ( v24 != v25 )
  {
    v26 = v24;
    do
    {
      v27 = *(_QWORD *)(v26 + 16);
      if ( v27 != v26 + 32 )
        _libc_free(v27);
      v26 += 56;
    }
    while ( v25 != v26 );
    *(_QWORD *)(a1 + 80) = v24;
  }
  result = *(_QWORD *)(a1 + 96);
  if ( result != *(_QWORD *)(a1 + 104) )
    *(_QWORD *)(a1 + 104) = result;
  *(_DWORD *)(a1 + 64) = 0;
  return result;
}
