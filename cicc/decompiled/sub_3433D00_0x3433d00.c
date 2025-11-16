// Function: sub_3433D00
// Address: 0x3433d00
//
unsigned __int64 __fastcall sub_3433D00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  unsigned __int64 v11; // r13
  unsigned int v13; // eax
  __int64 v14; // rdi
  int v15; // eax
  int v16; // r13d
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 j; // rdx
  __int64 v22; // rax

  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v8 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v8 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * (unsigned int)v8, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  a4 = (unsigned int)(4 * v6);
  v8 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)a4 < 0x40 )
    a4 = 64;
  if ( (unsigned int)v8 <= (unsigned int)a4 )
  {
LABEL_4:
    v9 = *(_QWORD **)(a1 + 8);
    for ( i = &v9[4 * v8]; i != v9; *((_DWORD *)v9 - 6) = -1 )
    {
      *v9 = 0;
      v9 += 4;
    }
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v13 = v6 - 1;
  if ( !v13 )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v16 = 64;
LABEL_20:
    sub_C7D6A0(v14, 32LL * (unsigned int)v8, 8);
    v17 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 16;
    v18 = (v17
         | (((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
             | (4 * v16 / 3u + 1)
             | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v18;
    v19 = sub_C7D670(32 * v18, 8);
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v19;
    for ( j = v19 + 32 * v20; j != v19; v19 += 32 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = 0;
        *(_DWORD *)(v19 + 8) = -1;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v13, v13);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = v13 ^ 0x1F;
  a4 = (unsigned int)(33 - v15);
  v16 = 1 << (33 - v15);
  if ( v16 < 64 )
    v16 = 64;
  if ( (_DWORD)v8 != v16 )
    goto LABEL_20;
  *(_QWORD *)(a1 + 16) = 0;
  v22 = v14 + 32LL * (unsigned int)v8;
  do
  {
    if ( v14 )
    {
      *(_QWORD *)v14 = 0;
      *(_DWORD *)(v14 + 8) = -1;
    }
    v14 += 32;
  }
  while ( v22 != v14 );
LABEL_7:
  v11 = *(_QWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 40) = 0;
  if ( (v11 & 1) == 0 && v11 )
  {
    if ( *(_QWORD *)v11 != v11 + 16 )
      _libc_free(*(_QWORD *)v11);
    j_j___libc_free_0(v11);
  }
  *(_QWORD *)(a1 + 32) = 1;
  return sub_228BF90((unsigned __int64 *)(a1 + 32), *(_DWORD *)(*(_QWORD *)(a2 + 960) + 536LL), 0, a4, a5, a6);
}
