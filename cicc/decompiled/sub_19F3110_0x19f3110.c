// Function: sub_19F3110
// Address: 0x19f3110
//
_QWORD *__fastcall sub_19F3110(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 *v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // rsi
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  _QWORD *j; // rdx
  __int64 *v15; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0((unsigned __int64)v5 << 6);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[8 * v3];
    for ( i = &result[8 * v7]; i != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        if ( *v10 != 0x7FFFFFFF0LL && *v10 != -8 )
        {
          sub_19EB1D0(a1, v10, &v15);
          v11 = v15;
          *v15 = *v10;
          sub_16CCEE0(v11 + 1, (__int64)(v11 + 6), 2, (__int64)(v10 + 1));
          ++*(_DWORD *)(a1 + 16);
          v12 = v10[3];
          if ( v12 != v10[2] )
            _libc_free(v12);
        }
        v10 += 8;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v13]; j != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
