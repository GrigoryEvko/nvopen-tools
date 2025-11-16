// Function: sub_E856C0
// Address: 0xe856c0
//
__int64 __fastcall sub_E856C0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  unsigned int v7; // ecx
  unsigned int v8; // eax
  _QWORD *v9; // rdi
  int v10; // ebx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *j; // rdx
  _QWORD *v16; // rax

  v2 = *(_DWORD *)(a1 + 464);
  ++*(_QWORD *)(a1 + 448);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 468) )
      return sub_E8D690(a1);
    v3 = *(unsigned int *)(a1 + 472);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 456), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 456) = 0;
      *(_QWORD *)(a1 + 464) = 0;
      *(_DWORD *)(a1 + 472) = 0;
      return sub_E8D690(a1);
    }
    goto LABEL_4;
  }
  v7 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 472);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 456);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 464) = 0;
    return sub_E8D690(a1);
  }
  v8 = v2 - 1;
  if ( !v8 )
  {
    v9 = *(_QWORD **)(a1 + 456);
    v10 = 64;
LABEL_15:
    sub_C7D6A0((__int64)v9, 16LL * (unsigned int)v3, 8);
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
    *(_DWORD *)(a1 + 472) = v12;
    v13 = (_QWORD *)sub_C7D670(16 * v12, 8);
    v14 = *(unsigned int *)(a1 + 472);
    *(_QWORD *)(a1 + 464) = 0;
    *(_QWORD *)(a1 + 456) = v13;
    for ( j = &v13[2 * v14]; j != v13; v13 += 2 )
    {
      if ( v13 )
        *v13 = -4096;
    }
    return sub_E8D690(a1);
  }
  _BitScanReverse(&v8, v8);
  v9 = *(_QWORD **)(a1 + 456);
  v10 = 1 << (33 - (v8 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( v10 != (_DWORD)v3 )
    goto LABEL_15;
  *(_QWORD *)(a1 + 464) = 0;
  v16 = &v9[2 * (unsigned int)v10];
  do
  {
    if ( v9 )
      *v9 = -4096;
    v9 += 2;
  }
  while ( v16 != v9 );
  return sub_E8D690(a1);
}
