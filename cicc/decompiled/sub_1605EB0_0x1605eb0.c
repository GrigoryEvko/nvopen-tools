// Function: sub_1605EB0
// Address: 0x1605eb0
//
__int64 __fastcall sub_1605EB0(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 *v6; // r12
  __int64 v7; // r14
  __int64 v8; // r15
  int v9; // edx
  int v10; // ebx
  unsigned int v11; // r14d
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
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
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = &v4[2 * v5];
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
        {
          v7 = v4[1];
          if ( v7 )
          {
            sub_164BE60(v4[1]);
            result = sub_1648B90(v7);
          }
        }
        *v4 = -8;
      }
      v4 += 2;
    }
LABEL_14:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    result = *v4;
    if ( *v4 != -16 && result != -8 )
    {
      v8 = v4[1];
      if ( v8 )
      {
        sub_164BE60(v4[1]);
        result = sub_1648B90(v8);
      }
    }
    v4 += 2;
  }
  while ( v4 != v6 );
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v9 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_14;
  }
  v10 = 64;
  v11 = v2 - 1;
  if ( v11 )
  {
    _BitScanReverse(&v12, v11);
    v10 = 1 << (33 - (v12 ^ 0x1F));
    if ( v10 < 64 )
      v10 = 64;
  }
  v13 = *(_QWORD **)(a1 + 8);
  if ( v10 == v9 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v13[2 * (unsigned int)v10];
    do
    {
      if ( v13 )
        *v13 = -8;
      v13 += 2;
    }
    while ( (_QWORD *)result != v13 );
  }
  else
  {
    j___libc_free_0(v13);
    v14 = ((((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 16;
    v15 = (v14
         | (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v15;
    result = sub_22077B0(16 * v15);
    v16 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 16 * v16; i != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
  }
  return result;
}
