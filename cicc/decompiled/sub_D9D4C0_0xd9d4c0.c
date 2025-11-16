// Function: sub_D9D4C0
// Address: 0xd9d4c0
//
__int64 __fastcall sub_D9D4C0(__int64 a1, __int64 a2)
{
  int v3; // r15d
  __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v6; // r14
  __int64 *v7; // r13
  __int64 *v8; // rdi
  __int64 *v9; // rdi
  __int64 v10; // rdx
  int v11; // ebx
  unsigned int v12; // r15d
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 i; // rdx

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v3 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v5 = *(__int64 **)(a1 + 8);
    result = (unsigned int)(4 * v3);
    v6 = 40LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v7 = &v5[(unsigned __int64)v6 / 8];
    if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
    {
      for ( ; v5 != v7; v5 += 5 )
      {
        result = *v5;
        if ( *v5 != -4096 )
        {
          if ( result != -8192 )
          {
            v8 = (__int64 *)v5[1];
            result = (__int64)(v5 + 3);
            if ( v8 != v5 + 3 )
              result = _libc_free(v8, a2);
          }
          *v5 = -4096;
        }
      }
LABEL_13:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    do
    {
      result = *v5;
      if ( *v5 != -8192 && result != -4096 )
      {
        v9 = (__int64 *)v5[1];
        result = (__int64)(v5 + 3);
        if ( v9 != v5 + 3 )
          result = _libc_free(v9, a2);
      }
      v5 += 5;
    }
    while ( v5 != v7 );
    v10 = *(unsigned int *)(a1 + 24);
    if ( !v3 )
    {
      if ( (_DWORD)v10 )
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v6, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return result;
      }
      goto LABEL_13;
    }
    v11 = 64;
    v12 = v3 - 1;
    if ( v12 )
    {
      _BitScanReverse(&v13, v12);
      v11 = 1 << (33 - (v13 ^ 0x1F));
      if ( v11 < 64 )
        v11 = 64;
    }
    v14 = *(_QWORD **)(a1 + 8);
    if ( (_DWORD)v10 == v11 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v14[5 * v10];
      do
      {
        if ( v14 )
          *v14 = -4096;
        v14 += 5;
      }
      while ( (_QWORD *)result != v14 );
    }
    else
    {
      sub_C7D6A0((__int64)v14, v6, 8);
      v15 = ((((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
             | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 16;
      v16 = (v15
           | (((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
             | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v16;
      result = sub_C7D670(40 * v16, 8);
      v17 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 40 * v17; i != result; result += 40 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
    }
  }
  return result;
}
