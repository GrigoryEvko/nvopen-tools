// Function: sub_D9DC00
// Address: 0xd9dc00
//
__int64 __fastcall sub_D9DC00(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  int v11; // edx
  __int64 v12; // rbx
  unsigned int v13; // r15d
  unsigned int v14; // eax
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
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
    for ( ; v4 != v6; v4 += 40 )
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -4096 )
      {
        if ( result != -8192 )
        {
          if ( *(_DWORD *)(v4 + 32) > 0x40u )
          {
            v7 = *(_QWORD *)(v4 + 24);
            if ( v7 )
              result = j_j___libc_free_0_0(v7);
          }
          if ( *(_DWORD *)(v4 + 16) > 0x40u )
          {
            v8 = *(_QWORD *)(v4 + 8);
            if ( v8 )
              result = j_j___libc_free_0_0(v8);
          }
        }
        *(_QWORD *)v4 = -4096;
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
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -8192 )
        break;
LABEL_25:
      v4 += 40;
      if ( v6 == v4 )
        goto LABEL_29;
    }
    if ( result != -4096 )
    {
      if ( *(_DWORD *)(v4 + 32) > 0x40u )
      {
        v9 = *(_QWORD *)(v4 + 24);
        if ( v9 )
          result = j_j___libc_free_0_0(v9);
      }
      if ( *(_DWORD *)(v4 + 16) > 0x40u )
      {
        v10 = *(_QWORD *)(v4 + 8);
        if ( v10 )
          result = j_j___libc_free_0_0(v10);
      }
      goto LABEL_25;
    }
    v4 += 40;
  }
  while ( v6 != v4 );
LABEL_29:
  v11 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v11 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_17;
  }
  v12 = 64;
  v13 = v2 - 1;
  if ( v13 )
  {
    _BitScanReverse(&v14, v13);
    v12 = (unsigned int)(1 << (33 - (v14 ^ 0x1F)));
    if ( (int)v12 < 64 )
      v12 = 64;
  }
  v15 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v12 == v11 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v15[5 * v12];
    do
    {
      if ( v15 )
        *v15 = -4096;
      v15 += 5;
    }
    while ( (_QWORD *)result != v15 );
  }
  else
  {
    sub_C7D6A0((__int64)v15, v5, 8);
    v16 = ((((((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v12 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v12 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v12 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v12 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 16;
    v17 = (v16
         | (((((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v12 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v12 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v12 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v12 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v17;
    result = sub_C7D670(40 * v17, 8);
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 40 * v18; i != result; result += 40 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  return result;
}
