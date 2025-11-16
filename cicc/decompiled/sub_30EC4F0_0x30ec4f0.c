// Function: sub_30EC4F0
// Address: 0x30ec4f0
//
__int64 __fastcall sub_30EC4F0(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  unsigned int v6; // ecx
  unsigned int v7; // eax
  _QWORD *v8; // rdi
  int v9; // r12d
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 j; // rdx

  v1 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v1 )
  {
    result = *(unsigned int *)(a1 + 28);
    if ( !(_DWORD)result )
      return result;
    v4 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v4 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v4, 8);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      return result;
    }
    goto LABEL_4;
  }
  v6 = 4 * v1;
  v4 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v6 = 64;
  if ( v6 >= (unsigned int)v4 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 16);
    for ( i = result + 16 * v4; i != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 24) = 0;
    return result;
  }
  v7 = v1 - 1;
  if ( !v7 )
  {
    v8 = *(_QWORD **)(a1 + 16);
    v9 = 64;
LABEL_15:
    sub_C7D6A0((__int64)v8, 16LL * (unsigned int)v4, 8);
    v10 = ((((((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
             | (4 * v9 / 3u + 1)
             | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4)
           | (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
           | (4 * v9 / 3u + 1)
           | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
           | (4 * v9 / 3u + 1)
           | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4)
         | (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
         | (4 * v9 / 3u + 1)
         | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 16;
    v11 = (v10
         | (((((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
             | (4 * v9 / 3u + 1)
             | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4)
           | (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
           | (4 * v9 / 3u + 1)
           | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
           | (4 * v9 / 3u + 1)
           | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4)
         | (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
         | (4 * v9 / 3u + 1)
         | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 32) = v11;
    result = sub_C7D670(16 * v11, 8);
    v12 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = result;
    for ( j = result + 16 * v12; j != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    return result;
  }
  _BitScanReverse(&v7, v7);
  v8 = *(_QWORD **)(a1 + 16);
  v9 = 1 << (33 - (v7 ^ 0x1F));
  if ( v9 < 64 )
    v9 = 64;
  if ( v9 != (_DWORD)v4 )
    goto LABEL_15;
  *(_QWORD *)(a1 + 24) = 0;
  result = (__int64)&v8[2 * (unsigned int)v9];
  do
  {
    if ( v8 )
      *v8 = -4096;
    v8 += 2;
  }
  while ( (_QWORD *)result != v8 );
  return result;
}
