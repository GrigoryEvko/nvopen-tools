// Function: sub_C8C250
// Address: 0xc8c250
//
__int64 __fastcall sub_C8C250(__int64 a1, __int64 a2)
{
  volatile __int64 *v3; // r13
  __int64 result; // rax
  volatile __int64 *v5; // r14
  __int64 v6; // rdi
  volatile __int64 *v7; // r15
  __int64 v8; // rdi
  volatile __int64 *v9; // rbx
  __int64 v10; // rdi
  volatile __int64 *v11; // r8
  __int64 v12; // rdi
  volatile __int64 *v13; // rdi
  __int64 v14; // rdi
  volatile __int64 *v15; // [rsp-48h] [rbp-48h]
  volatile __int64 *v16; // [rsp-40h] [rbp-40h]

  if ( a1 )
  {
    v3 = (volatile __int64 *)_InterlockedExchange64(&qword_4F84BA8, 0);
    if ( v3 )
    {
      v5 = (volatile __int64 *)_InterlockedExchange64(v3 + 1, 0);
      if ( v5 )
      {
        v7 = (volatile __int64 *)_InterlockedExchange64(v5 + 1, 0);
        if ( v7 )
        {
          v9 = (volatile __int64 *)_InterlockedExchange64(v7 + 1, 0);
          if ( v9 )
          {
            v11 = (volatile __int64 *)_InterlockedExchange64(v9 + 1, 0);
            if ( v11 )
            {
              v13 = (volatile __int64 *)_InterlockedExchange64(v11 + 1, 0);
              if ( v13 )
              {
                v15 = v11;
                sub_C8C190(v13, a2);
                a2 = 16;
                j_j___libc_free_0(v13, 16);
                v11 = v15;
              }
              v14 = _InterlockedExchange64(v11, 0);
              if ( v14 )
              {
                v16 = v11;
                _libc_free(v14, a2);
                v11 = v16;
              }
              a2 = 16;
              j_j___libc_free_0(v11, 16);
            }
            v12 = _InterlockedExchange64(v9, 0);
            if ( v12 )
              _libc_free(v12, a2);
            a2 = 16;
            j_j___libc_free_0(v9, 16);
          }
          v10 = _InterlockedExchange64(v7, 0);
          if ( v10 )
            _libc_free(v10, a2);
          a2 = 16;
          j_j___libc_free_0(v7, 16);
        }
        v8 = _InterlockedExchange64(v5, 0);
        if ( v8 )
          _libc_free(v8, a2);
        a2 = 16;
        j_j___libc_free_0(v5, 16);
      }
      v6 = _InterlockedExchange64(v3, 0);
      if ( v6 )
        _libc_free(v6, a2);
      j_j___libc_free_0(v3, 16);
    }
    return j_j___libc_free_0(a1, 1);
  }
  return result;
}
