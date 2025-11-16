// Function: sub_195EB50
// Address: 0x195eb50
//
__int64 __fastcall sub_195EB50(__int64 a1)
{
  __int64 result; // rax
  int v3; // r13d
  _QWORD *v4; // rbx
  _QWORD *v5; // r14
  unsigned __int64 v6; // rdi
  int v7; // ebx
  unsigned int v8; // r13d
  _QWORD *v9; // rdi
  unsigned int v10; // ebx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 24);
  v3 = *(_DWORD *)(a1 + 16);
  if ( !(_DWORD)result )
  {
    if ( !v3 )
      goto LABEL_29;
    v8 = v3 - 1;
    if ( !v8 )
    {
      v9 = *(_QWORD **)(a1 + 8);
      v7 = 64;
LABEL_16:
      j___libc_free_0(v9);
      v10 = 4 * v7 / 3u;
      v11 = ((((((((((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2)
                 | (v10 + 1)
                 | ((unsigned __int64)(v10 + 1) >> 1)) >> 4)
               | (((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2)
               | (v10 + 1)
               | ((unsigned __int64)(v10 + 1) >> 1)) >> 8)
             | (((((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2) | (v10 + 1)
                                                                         | ((unsigned __int64)(v10 + 1) >> 1)) >> 4)
             | (((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2)
             | (v10 + 1)
             | ((unsigned __int64)(v10 + 1) >> 1)) >> 16)
           | (((((((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2) | (v10 + 1)
                                                                         | ((unsigned __int64)(v10 + 1) >> 1)) >> 4)
             | (((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2)
             | (v10 + 1)
             | ((unsigned __int64)(v10 + 1) >> 1)) >> 8)
           | (((((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2) | (v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 4)
           | (((v10 + 1) | ((unsigned __int64)(v10 + 1) >> 1)) >> 2)
           | (v10 + 1)
           | ((unsigned __int64)(v10 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v11;
      result = sub_22077B0(40 * v11);
      v12 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 40 * v12; i != result; result += 40 )
      {
        if ( result )
          *(_QWORD *)result = -8;
      }
      return result;
    }
    goto LABEL_13;
  }
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[5 * result];
  do
  {
    if ( *v4 != -8 && *v4 != -16 )
    {
      v6 = v4[1];
      if ( (_QWORD *)v6 != v4 + 3 )
        _libc_free(v6);
    }
    v4 += 5;
  }
  while ( v5 != v4 );
  result = *(unsigned int *)(a1 + 24);
  if ( !v3 )
  {
    if ( (_DWORD)result )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
LABEL_29:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v7 = 64;
  v8 = v3 - 1;
  if ( v8 )
  {
LABEL_13:
    _BitScanReverse(&v8, v8);
    v7 = 1 << (33 - (v8 ^ 0x1F));
    if ( v7 < 64 )
      v7 = 64;
  }
  v9 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)result != v7 )
    goto LABEL_16;
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v9[5 * result];
  do
  {
    if ( v9 )
      *v9 = -8;
    v9 += 5;
  }
  while ( (_QWORD *)result != v9 );
  return result;
}
