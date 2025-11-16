// Function: sub_2989CE0
// Address: 0x2989ce0
//
__int64 __fastcall sub_2989CE0(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdx
  int v8; // ebx
  unsigned int v9; // r15d
  unsigned int v10; // eax
  _QWORD *v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 i; // rdx

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(_QWORD *)(a1 + 8);
  result = (unsigned int)(4 * v2);
  v5 = 40LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v6 = v4 + v5;
  if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
  {
    while ( v4 != v6 )
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -4096 )
      {
        if ( result != -8192 )
          result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 32LL * *(unsigned int *)(v4 + 32), 8);
        *(_QWORD *)v4 = -4096;
      }
      v4 += 40;
    }
LABEL_13:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    result = *(_QWORD *)v4;
    if ( *(_QWORD *)v4 != -8192 && result != -4096 )
      result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 32LL * *(unsigned int *)(v4 + 32), 8);
    v4 += 40;
  }
  while ( v4 != v6 );
  v7 = *(unsigned int *)(a1 + 24);
  if ( !v2 )
  {
    if ( (_DWORD)v7 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_13;
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
    result = (__int64)&v11[5 * v7];
    do
    {
      if ( v11 )
        *v11 = -4096;
      v11 += 5;
    }
    while ( (_QWORD *)result != v11 );
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
    result = sub_C7D670(40 * v13, 8);
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 40 * v14; i != result; result += 40 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  return result;
}
