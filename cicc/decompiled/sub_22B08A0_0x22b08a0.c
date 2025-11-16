// Function: sub_22B08A0
// Address: 0x22b08a0
//
__int64 __fastcall sub_22B08A0(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  unsigned int *v4; // rbx
  __int64 v5; // r14
  unsigned int *v6; // r13
  int v7; // edx
  __int64 v8; // rbx
  unsigned int v9; // r15d
  unsigned int v10; // eax
  _DWORD *v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 i; // rdx

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v4 = *(unsigned int **)(a1 + 8);
    result = (unsigned int)(4 * v2);
    v5 = 40LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v6 = &v4[(unsigned __int64)v5 / 4];
    if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
    {
      for ( ; v4 != v6; v4 += 10 )
      {
        result = *v4;
        if ( (_DWORD)result != -1 )
        {
          if ( (_DWORD)result != -2 )
            result = sub_C7D6A0(*((_QWORD *)v4 + 2), 4LL * v4[8], 4);
          *v4 = -1;
        }
      }
LABEL_12:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    do
    {
      if ( *v4 <= 0xFFFFFFFD )
        result = sub_C7D6A0(*((_QWORD *)v4 + 2), 4LL * v4[8], 4);
      v4 += 10;
    }
    while ( v6 != v4 );
    v7 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
    {
      if ( v7 )
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return result;
      }
      goto LABEL_12;
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
    v11 = *(_DWORD **)(a1 + 8);
    if ( (_DWORD)v8 == v7 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v11[10 * v8];
      do
      {
        if ( v11 )
          *v11 = -1;
        v11 += 10;
      }
      while ( (_DWORD *)result != v11 );
    }
    else
    {
      sub_C7D6A0((__int64)v11, v5, 8);
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
      result = sub_C7D670(40 * v13, 8);
      v14 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 40 * v14; i != result; result += 40 )
      {
        if ( result )
          *(_DWORD *)result = -1;
      }
    }
  }
  return result;
}
