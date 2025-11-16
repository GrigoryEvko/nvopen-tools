// Function: sub_3433F40
// Address: 0x3433f40
//
void __fastcall sub_3433F40(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  unsigned __int64 v6; // r12
  unsigned int v7; // ecx
  unsigned int v8; // eax
  __int64 v9; // rdi
  int v10; // r12d
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 j; // rdx
  __int64 v16; // rax

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v7 = 4 * v1;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v3 <= v7 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 8);
    for ( i = &v4[4 * v3]; i != v4; *((_DWORD *)v4 - 6) = -1 )
    {
      *v4 = 0;
      v4 += 4;
    }
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v8 = v1 - 1;
  if ( !v8 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v10 = 64;
LABEL_20:
    sub_C7D6A0(v9, 32LL * (unsigned int)v3, 8);
    v11 = ((((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 16;
    v12 = (v11
         | (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v12;
    v13 = sub_C7D670(32 * v12, 8);
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v13;
    for ( j = v13 + 32 * v14; j != v13; v13 += 32 )
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = 0;
        *(_DWORD *)(v13 + 8) = -1;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v8, v8);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = 1 << (33 - (v8 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v3 != v10 )
    goto LABEL_20;
  *(_QWORD *)(a1 + 16) = 0;
  v16 = v9 + 32LL * (unsigned int)v3;
  do
  {
    if ( v9 )
    {
      *(_QWORD *)v9 = 0;
      *(_DWORD *)(v9 + 8) = -1;
    }
    v9 += 32;
  }
  while ( v16 != v9 );
LABEL_7:
  v6 = *(_QWORD *)(a1 + 32);
  if ( (v6 & 1) == 0 && v6 )
  {
    if ( *(_QWORD *)v6 != v6 + 16 )
      _libc_free(*(_QWORD *)v6);
    j_j___libc_free_0(v6);
  }
  *(_QWORD *)(a1 + 32) = 1;
}
