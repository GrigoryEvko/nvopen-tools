// Function: sub_1A6DC40
// Address: 0x1a6dc40
//
__int64 __fastcall sub_1A6DC40(__int64 a1)
{
  int v2; // ebx
  __int64 result; // rax
  __int64 *v4; // r15
  __int64 v5; // rcx
  __int64 *v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // r12
  unsigned __int64 v11; // rdi
  int v12; // esi
  int v13; // r12d
  unsigned int v14; // eax
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 i; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(__int64 **)(a1 + 8);
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = &v4[8 * v5];
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    for ( ; v4 != v6; v4 += 8 )
    {
      result = *v4;
      if ( *v4 != -8 )
      {
        if ( result != -16 )
        {
          v7 = v4[6];
          v8 = v4[5];
          if ( v7 != v8 )
          {
            do
            {
              v9 = *(_QWORD *)(v8 + 8);
              if ( v9 != v8 + 24 )
                _libc_free(v9);
              v8 += 56;
            }
            while ( v7 != v8 );
            v8 = v4[5];
          }
          if ( v8 )
            j_j___libc_free_0(v8, v4[7] - v8);
          result = j___libc_free_0(v4[2]);
        }
        *v4 = -8;
      }
    }
LABEL_19:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *v4;
      if ( *v4 != -16 )
        break;
LABEL_29:
      v4 += 8;
      if ( v4 == v6 )
        goto LABEL_33;
    }
    if ( result != -8 )
    {
      v10 = v4[5];
      v20 = v4[6];
      if ( v20 != v10 )
      {
        do
        {
          v11 = *(_QWORD *)(v10 + 8);
          if ( v11 != v10 + 24 )
            _libc_free(v11);
          v10 += 56;
        }
        while ( v20 != v10 );
        v10 = v4[5];
      }
      if ( v10 )
        j_j___libc_free_0(v10, v4[7] - v10);
      result = j___libc_free_0(v4[2]);
      goto LABEL_29;
    }
    v4 += 8;
  }
  while ( v4 != v6 );
LABEL_33:
  v12 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v12 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_19;
  }
  v13 = 64;
  if ( v2 != 1 )
  {
    _BitScanReverse(&v14, v2 - 1);
    v13 = 1 << (33 - (v14 ^ 0x1F));
    if ( v13 < 64 )
      v13 = 64;
  }
  v15 = *(_QWORD **)(a1 + 8);
  if ( v13 == v12 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v15[8 * (unsigned __int64)(unsigned int)v13];
    do
    {
      if ( v15 )
        *v15 = -8;
      v15 += 8;
    }
    while ( (_QWORD *)result != v15 );
  }
  else
  {
    j___libc_free_0(v15);
    v16 = ((((((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
             | (4 * v13 / 3u + 1)
             | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
           | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
         | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
         | (4 * v13 / 3u + 1)
         | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 16;
    v17 = (v16
         | (((((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
             | (4 * v13 / 3u + 1)
             | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
           | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
         | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
         | (4 * v13 / 3u + 1)
         | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v17;
    result = sub_22077B0(v17 << 6);
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + (v18 << 6); i != result; result += 64 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
  }
  return result;
}
