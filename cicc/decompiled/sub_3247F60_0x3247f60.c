// Function: sub_3247F60
// Address: 0x3247f60
//
void __fastcall sub_3247F60(__int64 a1)
{
  unsigned __int64 v1; // r9
  __int64 v3; // rsi
  __int64 i; // r10
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  _QWORD *v9; // rcx
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *k; // rdx
  __int64 v14; // r13
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // r12d
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *j; // rdx
  _QWORD *v26; // rax

  v1 = (a1 + 8) & 0xFFFFFFFFFFFFFFFBLL;
  v3 = *(_QWORD *)(a1 + 376);
  for ( i = v3 + 88LL * *(unsigned int *)(a1 + 384); i != v3; v3 += 88 )
  {
    while ( 1 )
    {
      v5 = *(unsigned int *)(v3 + 16);
      if ( (_DWORD)v5 )
        break;
      v3 += 88;
      if ( i == v3 )
        goto LABEL_10;
    }
    v6 = 8 * v5;
    v7 = 0;
    do
    {
      v8 = *(_QWORD *)(*(_QWORD *)(v3 + 8) + v7);
      *(_QWORD *)(v8 + 40) = v1;
      v9 = *(_QWORD **)(a1 + 40);
      if ( v9 )
      {
        *(_QWORD *)v8 = *v9;
        **(_QWORD **)(a1 + 40) = v8 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v7 += 8;
      *(_QWORD *)(a1 + 40) = v8;
    }
    while ( v6 != v7 );
    *(_DWORD *)(v3 + 16) = 0;
  }
LABEL_10:
  v10 = *(_DWORD *)(a1 + 360);
  ++*(_QWORD *)(a1 + 344);
  if ( v10 )
  {
    v17 = 4 * v10;
    v11 = *(unsigned int *)(a1 + 368);
    if ( (unsigned int)(4 * v10) < 0x40 )
      v17 = 64;
    if ( (unsigned int)v11 <= v17 )
      goto LABEL_13;
    v18 = v10 - 1;
    if ( v18 )
    {
      _BitScanReverse(&v18, v18);
      v19 = *(_QWORD **)(a1 + 352);
      v20 = 1 << (33 - (v18 ^ 0x1F));
      if ( v20 < 64 )
        v20 = 64;
      if ( (_DWORD)v11 == v20 )
      {
        *(_QWORD *)(a1 + 360) = 0;
        v26 = &v19[2 * (unsigned int)v11];
        do
        {
          if ( v19 )
            *v19 = -4096;
          v19 += 2;
        }
        while ( v26 != v19 );
        goto LABEL_16;
      }
    }
    else
    {
      v19 = *(_QWORD **)(a1 + 352);
      v20 = 64;
    }
    sub_C7D6A0((__int64)v19, 16LL * *(unsigned int *)(a1 + 368), 8);
    v21 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
    v22 = (v21
         | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 368) = v22;
    v23 = (_QWORD *)sub_C7D670(16 * v22, 8);
    v24 = *(unsigned int *)(a1 + 368);
    *(_QWORD *)(a1 + 360) = 0;
    *(_QWORD *)(a1 + 352) = v23;
    for ( j = &v23[2 * v24]; j != v23; v23 += 2 )
    {
      if ( v23 )
        *v23 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 364) )
  {
    v11 = *(unsigned int *)(a1 + 368);
    if ( (unsigned int)v11 <= 0x40 )
    {
LABEL_13:
      v12 = *(_QWORD **)(a1 + 352);
      for ( k = &v12[2 * v11]; k != v12; v12 += 2 )
        *v12 = -4096;
      *(_QWORD *)(a1 + 360) = 0;
      goto LABEL_16;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 352), 16LL * *(unsigned int *)(a1 + 368), 8);
    *(_QWORD *)(a1 + 352) = 0;
    *(_QWORD *)(a1 + 360) = 0;
    *(_DWORD *)(a1 + 368) = 0;
  }
LABEL_16:
  v14 = *(_QWORD *)(a1 + 376);
  v15 = v14 + 88LL * *(unsigned int *)(a1 + 384);
  while ( v14 != v15 )
  {
    while ( 1 )
    {
      v15 -= 88;
      v16 = *(_QWORD *)(v15 + 8);
      if ( v16 == v15 + 24 )
        break;
      _libc_free(v16);
      if ( v14 == v15 )
        goto LABEL_20;
    }
  }
LABEL_20:
  *(_DWORD *)(a1 + 384) = 0;
}
