// Function: sub_16CC450
// Address: 0x16cc450
//
__int64 __fastcall sub_16CC450(__int64 a1)
{
  volatile __int64 *v2; // r13
  __int64 result; // rax
  volatile __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  volatile __int64 *v6; // r15
  unsigned __int64 v7; // rdi
  volatile __int64 *v8; // rbx
  unsigned __int64 v9; // rdi
  volatile __int64 *v10; // r8
  unsigned __int64 v11; // rdi
  volatile __int64 *v12; // rdi
  unsigned __int64 v13; // rdi
  volatile __int64 *v14; // [rsp-48h] [rbp-48h]
  volatile __int64 *v15; // [rsp-40h] [rbp-40h]

  if ( a1 )
  {
    v2 = (volatile __int64 *)_InterlockedExchange64(&qword_4FA1088, 0);
    if ( v2 )
    {
      v4 = (volatile __int64 *)_InterlockedExchange64(v2 + 1, 0);
      if ( v4 )
      {
        v6 = (volatile __int64 *)_InterlockedExchange64(v4 + 1, 0);
        if ( v6 )
        {
          v8 = (volatile __int64 *)_InterlockedExchange64(v6 + 1, 0);
          if ( v8 )
          {
            v10 = (volatile __int64 *)_InterlockedExchange64(v8 + 1, 0);
            if ( v10 )
            {
              v12 = (volatile __int64 *)_InterlockedExchange64(v10 + 1, 0);
              if ( v12 )
              {
                v14 = v10;
                sub_16CC390(v12);
                j_j___libc_free_0(v12, 16);
                v10 = v14;
              }
              v13 = _InterlockedExchange64(v10, 0);
              if ( v13 )
              {
                v15 = v10;
                _libc_free(v13);
                v10 = v15;
              }
              j_j___libc_free_0(v10, 16);
            }
            v11 = _InterlockedExchange64(v8, 0);
            if ( v11 )
              _libc_free(v11);
            j_j___libc_free_0(v8, 16);
          }
          v9 = _InterlockedExchange64(v6, 0);
          if ( v9 )
            _libc_free(v9);
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
    return j_j___libc_free_0(a1, 1);
  }
  return result;
}
