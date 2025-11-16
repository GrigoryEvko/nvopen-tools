// Function: sub_20515F0
// Address: 0x20515f0
//
__int64 __fastcall sub_20515F0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *k; // rdx
  unsigned int v11; // ecx
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  int v17; // r13d
  __int64 v18; // r12
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *m; // rdx
  unsigned int v22; // ecx
  _QWORD *v23; // rdi
  unsigned int v24; // eax
  int v25; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  int v28; // r13d
  __int64 v29; // r12
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *j; // rdx
  _QWORD *v33; // rax
  _QWORD *v34; // rax

  v2 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v22 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v22 = 64;
  if ( v22 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 16);
    for ( i = &v4[3 * v3]; i != v4; v4 += 3 )
      *v4 = -8;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
  v23 = *(_QWORD **)(a1 + 16);
  v24 = v2 - 1;
  if ( !v24 )
  {
    v29 = 3072;
    v28 = 128;
LABEL_35:
    j___libc_free_0(v23);
    *(_DWORD *)(a1 + 32) = v28;
    v30 = (_QWORD *)sub_22077B0(v29);
    v31 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v30;
    for ( j = &v30[3 * v31]; j != v30; v30 += 3 )
    {
      if ( v30 )
        *v30 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v24, v24);
  v25 = 1 << (33 - (v24 ^ 0x1F));
  if ( v25 < 64 )
    v25 = 64;
  if ( (_DWORD)v3 != v25 )
  {
    v26 = (4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1);
    v27 = ((((v26 >> 2) | v26 | (((v26 >> 2) | v26) >> 4)) >> 8)
         | (v26 >> 2)
         | v26
         | (((v26 >> 2) | v26) >> 4)
         | (((((v26 >> 2) | v26 | (((v26 >> 2) | v26) >> 4)) >> 8) | (v26 >> 2) | v26 | (((v26 >> 2) | v26) >> 4)) >> 16))
        + 1;
    v28 = v27;
    v29 = 24 * v27;
    goto LABEL_35;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v33 = &v23[3 * v3];
  do
  {
    if ( v23 )
      *v23 = -8;
    v23 += 3;
  }
  while ( v33 != v23 );
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
      j___libc_free_0(*(_QWORD *)(a1 + 48));
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
  if ( (unsigned int)v7 <= v11 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 48);
    for ( k = &v8[3 * v7]; k != v8; v8 += 3 )
      *v8 = -8;
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_13;
  }
  v12 = *(_QWORD **)(a1 + 48);
  v13 = v6 - 1;
  if ( !v13 )
  {
    v18 = 3072;
    v17 = 128;
LABEL_22:
    j___libc_free_0(v12);
    *(_DWORD *)(a1 + 64) = v17;
    v19 = (_QWORD *)sub_22077B0(v18);
    v20 = *(unsigned int *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 48) = v19;
    for ( m = &v19[3 * v20]; m != v19; v19 += 3 )
    {
      if ( v19 )
        *v19 = -8;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v13, v13);
  v14 = 1 << (33 - (v13 ^ 0x1F));
  if ( v14 < 64 )
    v14 = 64;
  if ( (_DWORD)v7 != v14 )
  {
    v15 = (4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1);
    v16 = ((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8)
         | (v15 >> 2)
         | v15
         | (((v15 >> 2) | v15) >> 4)
         | (((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8) | (v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 16))
        + 1;
    v17 = v16;
    v18 = 24 * v16;
    goto LABEL_22;
  }
  *(_QWORD *)(a1 + 56) = 0;
  v34 = &v12[3 * v7];
  do
  {
    if ( v12 )
      *v12 = -8;
    v12 += 3;
  }
  while ( v34 != v12 );
LABEL_13:
  *(_DWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 400) = 0;
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 760) = 0;
  *(_DWORD *)(a1 + 536) = 1;
  return sub_2098790(a1 + 248);
}
