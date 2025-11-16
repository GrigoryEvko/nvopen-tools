// Function: sub_1E1E220
// Address: 0x1e1e220
//
__int64 __fastcall sub_1E1E220(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  unsigned int *v4; // rbx
  __int64 v5; // rdx
  unsigned int *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rdi
  int v9; // edx
  int v10; // ebx
  unsigned int v11; // r14d
  unsigned int v12; // eax
  _DWORD *v13; // rdi
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
  v4 = *(unsigned int **)(a1 + 8);
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = &v4[8 * v5];
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    while ( v6 != v4 )
    {
      result = *v4;
      if ( (_DWORD)result != -1 )
      {
        if ( (_DWORD)result != -2 )
        {
          v7 = *((_QWORD *)v4 + 1);
          if ( v7 )
            result = j_j___libc_free_0(v7, *((_QWORD *)v4 + 3) - v7);
        }
        *v4 = -1;
      }
      v4 += 8;
    }
    goto LABEL_14;
  }
  do
  {
    if ( *v4 <= 0xFFFFFFFD )
    {
      v8 = *((_QWORD *)v4 + 1);
      if ( v8 )
        result = j_j___libc_free_0(v8, *((_QWORD *)v4 + 3) - v8);
    }
    v4 += 8;
  }
  while ( v6 != v4 );
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v9 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_34;
    }
LABEL_14:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
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
  v13 = *(_DWORD **)(a1 + 8);
  if ( v9 != v10 )
  {
    j___libc_free_0(v13);
    result = sub_1454B60(4 * v10 / 3u + 1);
    *(_DWORD *)(a1 + 24) = result;
    if ( (_DWORD)result )
    {
      result = sub_22077B0(32LL * (unsigned int)result);
      v14 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 32 * v14; i != result; result += 32 )
      {
        if ( result )
          *(_DWORD *)result = -1;
      }
      return result;
    }
LABEL_34:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v13[8 * v9];
  do
  {
    if ( v13 )
      *v13 = -1;
    v13 += 8;
  }
  while ( (_DWORD *)result != v13 );
  return result;
}
