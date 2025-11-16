// Function: sub_C8C190
// Address: 0xc8c190
//
__int64 __fastcall sub_C8C190(volatile __int64 *a1, __int64 a2)
{
  volatile __int64 *v3; // r12
  __int64 v4; // rdi
  volatile __int64 *v5; // r14
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // rdi

  v3 = (volatile __int64 *)_InterlockedExchange64(a1 + 1, 0);
  if ( v3 )
  {
    v5 = (volatile __int64 *)_InterlockedExchange64(v3 + 1, 0);
    if ( v5 )
    {
      v8 = _InterlockedExchange64(v5 + 1, 0);
      if ( v8 )
      {
        sub_C8C190();
        a2 = 16;
        j_j___libc_free_0(v8, 16);
      }
      v9 = _InterlockedExchange64(v5, 0);
      if ( v9 )
        _libc_free(v9, a2);
      a2 = 16;
      j_j___libc_free_0(v5, 16);
    }
    v6 = _InterlockedExchange64(v3, 0);
    if ( v6 )
      _libc_free(v6, a2);
    a2 = 16;
    result = j_j___libc_free_0(v3, 16);
  }
  v4 = _InterlockedExchange64(a1, 0);
  if ( v4 )
    return _libc_free(v4, a2);
  return result;
}
