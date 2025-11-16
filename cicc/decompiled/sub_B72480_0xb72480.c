// Function: sub_B72480
// Address: 0xb72480
//
__int64 __fastcall sub_B72480(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 *v7; // r14
  __int64 v8; // r13
  __int64 v9; // rdi
  unsigned int v10; // edx
  int v11; // ebx
  unsigned int v12; // r15d
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 i; // rdx
  __int64 v19; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v4 = *(__int64 **)(a1 + 8);
    result = (unsigned int)(4 * v2);
    v5 = *(unsigned int *)(a1 + 24);
    v6 = 16 * v5;
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v7 = &v4[(unsigned __int64)v6 / 8];
    if ( (unsigned int)v5 <= (unsigned int)result )
    {
      for ( ; v4 != v7; v4 += 2 )
      {
        result = *v4;
        if ( *v4 != -4096 )
        {
          if ( result != -8192 )
          {
            v8 = v4[1];
            if ( v8 )
            {
              sub_BD7260(v4[1]);
              result = sub_BD2DD0(v8);
            }
          }
          *v4 = -4096;
        }
      }
LABEL_13:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    do
    {
      result = *v4;
      if ( *v4 != -8192 && result != -4096 )
      {
        v9 = v4[1];
        if ( v9 )
        {
          v19 = v4[1];
          sub_BD7260(v9);
          result = sub_BD2DD0(v19);
        }
      }
      v4 += 2;
    }
    while ( v4 != v7 );
    v10 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
    {
      if ( v10 )
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
    v12 = v2 - 1;
    if ( v12 )
    {
      _BitScanReverse(&v13, v12);
      v11 = 1 << (33 - (v13 ^ 0x1F));
      if ( v11 < 64 )
        v11 = 64;
    }
    v14 = *(_QWORD **)(a1 + 8);
    if ( v10 == v11 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v14[2 * v10];
      do
      {
        if ( v14 )
          *v14 = -4096;
        v14 += 2;
      }
      while ( (_QWORD *)result != v14 );
    }
    else
    {
      sub_C7D6A0(v14, v6, 8);
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
      result = sub_C7D670(16 * v16, 8);
      v17 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 16 * v17; i != result; result += 16 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
    }
  }
  return result;
}
