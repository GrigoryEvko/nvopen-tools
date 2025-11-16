// Function: sub_BEDCB0
// Address: 0xbedcb0
//
__int64 __fastcall sub_BEDCB0(__int64 a1, __int64 a2)
{
  int v3; // r15d
  __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 *v8; // r14
  __int64 v9; // r12
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

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  result = (unsigned int)(4 * v3);
  v5 = *(__int64 **)(a1 + 8);
  v6 = *(unsigned int *)(a1 + 24);
  v7 = 16 * v6;
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v8 = &v5[(unsigned __int64)v7 / 8];
  if ( (unsigned int)v6 <= (unsigned int)result )
  {
    for ( ; v5 != v8; v5 += 2 )
    {
      result = *v5;
      if ( *v5 != -4096 )
      {
        if ( result != -8192 )
        {
          result = v5[1];
          if ( result )
          {
            if ( (result & 4) != 0 )
            {
              result &= 0xFFFFFFFFFFFFFFF8LL;
              v9 = result;
              if ( result )
              {
                if ( *(_QWORD *)result != result + 16 )
                  _libc_free(*(_QWORD *)result, a2);
                a2 = 48;
                result = j_j___libc_free_0(v9, 48);
              }
            }
          }
        }
        *v5 = -4096;
      }
    }
LABEL_17:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *v5;
      if ( *v5 != -8192 )
        break;
LABEL_25:
      v5 += 2;
      if ( v5 == v8 )
        goto LABEL_29;
    }
    if ( result != -4096 )
    {
      result = v5[1];
      if ( result )
      {
        if ( (result & 4) != 0 )
        {
          result &= 0xFFFFFFFFFFFFFFF8LL;
          if ( result )
          {
            if ( *(_QWORD *)result != result + 16 )
            {
              v19 = result;
              _libc_free(*(_QWORD *)result, a2);
              result = v19;
            }
            a2 = 48;
            result = j_j___libc_free_0(result, 48);
          }
        }
      }
      goto LABEL_25;
    }
    v5 += 2;
  }
  while ( v5 != v8 );
LABEL_29:
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    if ( v10 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v7, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_17;
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
    sub_C7D6A0(v14, v7, 8);
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
  return result;
}
