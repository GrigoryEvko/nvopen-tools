// Function: sub_1FDED60
// Address: 0x1fded60
//
__int64 __fastcall sub_1FDED60(__int64 a1)
{
  __int64 result; // rax
  int v3; // r13d
  _QWORD *v4; // rbx
  _QWORD *v5; // r14
  int v6; // ebx
  unsigned int v7; // r13d
  _QWORD *v8; // rdi
  unsigned int v9; // ebx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 24);
  v3 = *(_DWORD *)(a1 + 16);
  if ( !(_DWORD)result )
  {
    if ( !v3 )
      goto LABEL_28;
    v7 = v3 - 1;
    if ( !v7 )
    {
      v8 = *(_QWORD **)(a1 + 8);
      v6 = 64;
LABEL_15:
      j___libc_free_0(v8);
      v9 = 4 * v6 / 3u;
      v10 = ((((((((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
               | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
               | (v9 + 1)
               | ((unsigned __int64)(v9 + 1) >> 1)) >> 8)
             | (((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
             | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
             | (v9 + 1)
             | ((unsigned __int64)(v9 + 1) >> 1)) >> 16)
           | (((((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
             | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
             | (v9 + 1)
             | ((unsigned __int64)(v9 + 1) >> 1)) >> 8)
           | (((((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2) | (v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 4)
           | (((v9 + 1) | ((unsigned __int64)(v9 + 1) >> 1)) >> 2)
           | (v9 + 1)
           | ((unsigned __int64)(v9 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v10;
      result = sub_22077B0(72 * v10);
      v11 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 72 * v11; i != result; result += 72 )
      {
        if ( result )
          *(_QWORD *)result = -8;
      }
      return result;
    }
    goto LABEL_12;
  }
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[9 * result];
  do
  {
    if ( *v4 != -8 && *v4 != -16 )
    {
      j___libc_free_0(v4[6]);
      j___libc_free_0(v4[2]);
    }
    v4 += 9;
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
LABEL_28:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v6 = 64;
  v7 = v3 - 1;
  if ( v7 )
  {
LABEL_12:
    _BitScanReverse(&v7, v7);
    v6 = 1 << (33 - (v7 ^ 0x1F));
    if ( v6 < 64 )
      v6 = 64;
  }
  v8 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)result != v6 )
    goto LABEL_15;
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v8[9 * result];
  do
  {
    if ( v8 )
      *v8 = -8;
    v8 += 9;
  }
  while ( (_QWORD *)result != v8 );
  return result;
}
