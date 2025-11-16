// Function: sub_37F1EF0
// Address: 0x37f1ef0
//
__int64 __fastcall sub_37F1EF0(__int64 a1)
{
  int v2; // ecx
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // r12
  int v11; // edx
  __int64 v12; // rbx
  unsigned int v13; // ecx
  unsigned int v14; // eax
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  __int64 i; // rdx
  int v20; // [rsp+Ch] [rbp-34h]
  int v21; // [rsp+Ch] [rbp-34h]

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
  v5 = 56LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v6 = v4 + v5;
  if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
  {
    if ( v6 != v4 )
    {
      do
      {
        result = *(_QWORD *)v4;
        v7 = v4 + 56;
        if ( *(_QWORD *)v4 != -4096 )
        {
          if ( result != -8192 )
          {
            v8 = *(_QWORD *)(v4 + 40);
            if ( v8 != v7 )
              _libc_free(v8);
            result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 8LL * *(unsigned int *)(v4 + 32), 8);
          }
          *(_QWORD *)v4 = -4096;
        }
        v4 += 56;
      }
      while ( v7 != v6 );
    }
LABEL_14:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *(_QWORD *)v4;
      v10 = v4 + 56;
      if ( *(_QWORD *)v4 != -8192 )
        break;
LABEL_19:
      v4 += 56;
      if ( v6 == v10 )
        goto LABEL_23;
    }
    if ( result != -4096 )
    {
      v9 = *(_QWORD *)(v4 + 40);
      if ( v9 != v10 )
      {
        v20 = v2;
        _libc_free(v9);
        v2 = v20;
      }
      v21 = v2;
      result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 8LL * *(unsigned int *)(v4 + 32), 8);
      v2 = v21;
      goto LABEL_19;
    }
    v4 += 56;
  }
  while ( v6 != v10 );
LABEL_23:
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
    goto LABEL_14;
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
    result = (__int64)&v15[7 * v12];
    do
    {
      if ( v15 )
        *v15 = -4096;
      v15 += 7;
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
    result = sub_C7D670(56 * v17, 8);
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 56 * v18; i != result; result += 56 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  return result;
}
