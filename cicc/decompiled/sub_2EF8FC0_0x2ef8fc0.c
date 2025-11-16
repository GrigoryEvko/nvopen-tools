// Function: sub_2EF8FC0
// Address: 0x2ef8fc0
//
__int64 __fastcall sub_2EF8FC0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  size_t v4; // rdx
  void *v5; // rdi
  unsigned int v6; // ecx
  unsigned int v7; // eax
  _DWORD *v8; // rdi
  __int64 v9; // r12
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !(_DWORD)result )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 4 * v3, 4);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_4;
  }
  v6 = 4 * result;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * result) < 0x40 )
    v6 = 64;
  if ( v6 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = 4 * v3;
    v5 = *(void **)(a1 + 8);
    if ( v4 )
      result = (__int64)memset(v5, 255, v4);
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v7 = result - 1;
  if ( !v7 )
  {
    v8 = *(_DWORD **)(a1 + 8);
    LODWORD(v9) = 64;
LABEL_15:
    sub_C7D6A0((__int64)v8, 4 * v3, 4);
    v10 = ((((((((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v9 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v9 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v9 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v9 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 16;
    v11 = (v10
         | (((((((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v9 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v9 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v9 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v9 / 3u + 1) | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v9 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v9 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v11;
    result = sub_C7D670(4 * v11, 4);
    v12 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 4 * v12; i != result; result += 4 )
    {
      if ( result )
        *(_DWORD *)result = -1;
    }
    return result;
  }
  _BitScanReverse(&v7, v7);
  v8 = *(_DWORD **)(a1 + 8);
  v9 = (unsigned int)(1 << (33 - (v7 ^ 0x1F)));
  if ( (int)v9 < 64 )
    v9 = 64;
  if ( (_DWORD)v9 != (_DWORD)v3 )
    goto LABEL_15;
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v8[v9];
  do
  {
    if ( v8 )
      *v8 = -1;
    ++v8;
  }
  while ( (_DWORD *)result != v8 );
  return result;
}
