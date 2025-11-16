// Function: sub_1A6DA40
// Address: 0x1a6da40
//
__int64 __fastcall sub_1A6DA40(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 *v6; // r13
  int v7; // edx
  __int64 v8; // rbx
  unsigned int v9; // r14d
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
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = &v4[5 * v5];
  result = (unsigned int)(4 * v2);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    while ( v4 != v6 )
    {
      result = *v4;
      if ( *v4 != -8 )
      {
        if ( result != -16 )
          result = j___libc_free_0(v4[2]);
        *v4 = -8;
      }
      v4 += 5;
    }
LABEL_13:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    result = *v4;
    if ( *v4 != -16 && result != -8 )
      result = j___libc_free_0(v4[2]);
    v4 += 5;
  }
  while ( v4 != v6 );
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v7 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
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
    v8 = (unsigned int)(1 << (33 - (v10 ^ 0x1F)));
    if ( (int)v8 < 64 )
      v8 = 64;
  }
  v11 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v8 == v7 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v11[5 * v8];
    do
    {
      if ( v11 )
        *v11 = -8;
      v11 += 5;
    }
    while ( (_QWORD *)result != v11 );
  }
  else
  {
    j___libc_free_0(v11);
    v12 = ((((((((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v8 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v8 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v8 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v8 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 16;
    v13 = (v12
         | (((((((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v8 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v8 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v8 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v8 / 3u + 1) | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v8 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v8 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v13;
    result = sub_22077B0(40 * v13);
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 40 * v14; i != result; result += 40 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
  }
  return result;
}
