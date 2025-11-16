// Function: sub_16CC390
// Address: 0x16cc390
//
void __fastcall sub_16CC390(volatile __int64 *a1)
{
  volatile __int64 *v2; // r12
  unsigned __int64 v3; // rdi
  volatile __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v2 = (volatile __int64 *)_InterlockedExchange64(a1 + 1, 0);
  if ( v2 )
  {
    v4 = (volatile __int64 *)_InterlockedExchange64(v2 + 1, 0);
    if ( v4 )
    {
      v6 = _InterlockedExchange64(v4 + 1, 0);
      if ( v6 )
      {
        sub_16CC390();
        j_j___libc_free_0(v6, 16);
      }
      v7 = _InterlockedExchange64(v4, 0);
      if ( v7 )
        _libc_free(v7);
      j_j___libc_free_0(v4, 16);
    }
    v5 = _InterlockedExchange64(v2, 0);
    if ( v5 )
      _libc_free(v5);
    j_j___libc_free_0(v2, 16);
  }
  v3 = _InterlockedExchange64(a1, 0);
  if ( v3 )
    _libc_free(v3);
}
