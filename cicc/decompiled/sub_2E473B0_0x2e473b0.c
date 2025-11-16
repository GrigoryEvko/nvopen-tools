// Function: sub_2E473B0
// Address: 0x2e473b0
//
void __fastcall sub_2E473B0(__int64 a1)
{
  int v2; // r15d
  __int64 v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdx
  int v8; // ebx
  unsigned int v9; // r15d
  unsigned int v10; // eax
  _QWORD *v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rcx
  _QWORD *i; // rdx
  _QWORD *v17; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = 4 * v2;
  v5 = 56LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v4 = 64;
  v6 = v3 + v5;
  if ( *(_DWORD *)(a1 + 24) <= v4 )
  {
    for ( ; v3 != v6; v3 += 56 )
    {
      if ( *(_QWORD *)v3 != -4096 )
      {
        if ( *(_QWORD *)v3 != -8192 && !*(_BYTE *)(v3 + 36) )
          _libc_free(*(_QWORD *)(v3 + 16));
        *(_QWORD *)v3 = -4096;
      }
    }
LABEL_14:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  do
  {
    while ( 1 )
    {
      if ( *(_QWORD *)v3 == -8192 || *(_QWORD *)v3 == -4096 )
        goto LABEL_17;
      if ( *(_BYTE *)(v3 + 36) )
        break;
      _libc_free(*(_QWORD *)(v3 + 16));
LABEL_17:
      v3 += 56;
      if ( v3 == v6 )
        goto LABEL_22;
    }
    v3 += 56;
  }
  while ( v3 != v6 );
LABEL_22:
  v7 = *(unsigned int *)(a1 + 24);
  if ( !v2 )
  {
    if ( (_DWORD)v7 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_14;
  }
  v8 = 64;
  v9 = v2 - 1;
  if ( v9 )
  {
    _BitScanReverse(&v10, v9);
    v8 = 1 << (33 - (v10 ^ 0x1F));
    if ( v8 < 64 )
      v8 = 64;
  }
  v11 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v7 == v8 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v17 = &v11[7 * v7];
    do
    {
      if ( v11 )
        *v11 = -4096;
      v11 += 7;
    }
    while ( v17 != v11 );
  }
  else
  {
    sub_C7D6A0((__int64)v11, v5, 8);
    v12 = ((((((((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
             | (4 * v8 / 3u + 1)
             | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 4)
           | (((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
           | (4 * v8 / 3u + 1)
           | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
           | (4 * v8 / 3u + 1)
           | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 4)
         | (((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
         | (4 * v8 / 3u + 1)
         | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 16;
    v13 = (v12
         | (((((((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
             | (4 * v8 / 3u + 1)
             | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 4)
           | (((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
           | (4 * v8 / 3u + 1)
           | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
           | (4 * v8 / 3u + 1)
           | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 4)
         | (((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
         | (4 * v8 / 3u + 1)
         | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v13;
    v14 = (_QWORD *)sub_C7D670(56 * v13, 8);
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v14;
    for ( i = &v14[7 * v15]; i != v14; v14 += 7 )
    {
      if ( v14 )
        *v14 = -4096;
    }
  }
}
