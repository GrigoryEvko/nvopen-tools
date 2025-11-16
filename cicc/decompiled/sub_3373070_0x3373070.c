// Function: sub_3373070
// Address: 0x3373070
//
__int64 __fastcall sub_3373070(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *k; // rdx
  unsigned int v11; // ecx
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  __int64 v14; // r12
  _QWORD *v15; // rax
  unsigned int v16; // ecx
  unsigned int v17; // eax
  _QWORD *v18; // rdi
  int v19; // r12d
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *m; // rdx
  _QWORD *v30; // rax

  v1 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 24 * v3, 8);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v16 = 4 * v1;
  v3 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v16 = 64;
  if ( v16 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 16);
    for ( i = &v4[3 * v3]; i != v4; v4 += 3 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
  v17 = v1 - 1;
  if ( !v17 )
  {
    v18 = *(_QWORD **)(a1 + 16);
    v19 = 64;
LABEL_33:
    sub_C7D6A0((__int64)v18, 24 * v3, 8);
    v20 = ((((((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
             | (4 * v19 / 3u + 1)
             | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
           | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
           | (4 * v19 / 3u + 1)
           | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
           | (4 * v19 / 3u + 1)
           | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
         | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
         | (4 * v19 / 3u + 1)
         | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 16;
    v21 = (v20
         | (((((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
             | (4 * v19 / 3u + 1)
             | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
           | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
           | (4 * v19 / 3u + 1)
           | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
           | (4 * v19 / 3u + 1)
           | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
         | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
         | (4 * v19 / 3u + 1)
         | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 32) = v21;
    v22 = (_QWORD *)sub_C7D670(24 * v21, 8);
    v23 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v22;
    for ( j = &v22[3 * v23]; j != v22; v22 += 3 )
    {
      if ( v22 )
        *v22 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v17, v17);
  v18 = *(_QWORD **)(a1 + 16);
  v19 = 1 << (33 - (v17 ^ 0x1F));
  if ( v19 < 64 )
    v19 = 64;
  if ( (_DWORD)v3 != v19 )
    goto LABEL_33;
  *(_QWORD *)(a1 + 24) = 0;
  v30 = &v18[3 * v3];
  do
  {
    if ( v18 )
      *v18 = -4096;
    v18 += 3;
  }
  while ( v30 != v18 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 60) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 48), 24 * v7, 8);
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v11 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v11 = 64;
  if ( v11 >= (unsigned int)v7 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 48);
    for ( k = &v8[3 * v7]; k != v8; v8 += 3 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_13;
  }
  v12 = v6 - 1;
  if ( v12 )
  {
    _BitScanReverse(&v12, v12);
    v13 = *(_QWORD **)(a1 + 48);
    v14 = (unsigned int)(1 << (33 - (v12 ^ 0x1F)));
    if ( (int)v14 < 64 )
      v14 = 64;
    if ( (_DWORD)v14 == (_DWORD)v7 )
    {
      *(_QWORD *)(a1 + 56) = 0;
      v15 = &v13[3 * v14];
      do
      {
        if ( v13 )
          *v13 = -4096;
        v13 += 3;
      }
      while ( v15 != v13 );
      goto LABEL_13;
    }
  }
  else
  {
    v13 = *(_QWORD **)(a1 + 48);
    LODWORD(v14) = 64;
  }
  sub_C7D6A0((__int64)v13, 24 * v7, 8);
  v25 = ((((((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v14 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v14 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v14 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v14 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 16;
  v26 = (v25
       | (((((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v14 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v14 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v14 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v14 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 64) = v26;
  v27 = (_QWORD *)sub_C7D670(24 * v26, 8);
  v28 = *(unsigned int *)(a1 + 64);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 48) = v27;
  for ( m = &v27[3 * v28]; m != v27; v27 += 3 )
  {
    if ( v27 )
      *v27 = -4096;
  }
LABEL_13:
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 424) = 0;
  *(_DWORD *)(a1 + 568) = 0;
  *(_DWORD *)(a1 + 712) = 0;
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 1016) = 0;
  *(_DWORD *)(a1 + 848) = 1;
  return sub_3433F40(a1 + 272);
}
