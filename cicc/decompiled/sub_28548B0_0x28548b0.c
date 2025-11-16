// Function: sub_28548B0
// Address: 0x28548b0
//
void __fastcall sub_28548B0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 **v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  __int64 v6; // rbx
  unsigned __int64 *v7; // r14
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  _QWORD *v10; // rbx
  _QWORD *v11; // r12
  __int64 v12; // rax

  v1 = *a1;
  if ( *a1 )
  {
    v2 = *(unsigned __int64 ***)(v1 + 120);
    v3 = (unsigned __int64)&v2[*(unsigned int *)(v1 + 128)];
    if ( v2 != (unsigned __int64 **)v3 )
    {
      do
      {
        v4 = *v2;
        *v2 = 0;
        if ( v4 )
        {
          v5 = v4[8];
          if ( (unsigned __int64 *)v5 != v4 + 10 )
            _libc_free(v5);
          if ( (unsigned __int64 *)*v4 != v4 + 2 )
            _libc_free(*v4);
          j_j___libc_free_0((unsigned __int64)v4);
        }
        ++v2;
      }
      while ( (unsigned __int64 **)v3 != v2 );
      v6 = *(_QWORD *)(v1 + 120);
      v3 = v6 + 8LL * *(unsigned int *)(v1 + 128);
      if ( v6 != v3 )
      {
        do
        {
          v7 = *(unsigned __int64 **)(v3 - 8);
          v3 -= 8LL;
          if ( v7 )
          {
            v8 = v7[8];
            if ( (unsigned __int64 *)v8 != v7 + 10 )
              _libc_free(v8);
            if ( (unsigned __int64 *)*v7 != v7 + 2 )
              _libc_free(*v7);
            j_j___libc_free_0((unsigned __int64)v7);
          }
        }
        while ( v6 != v3 );
        v3 = *(_QWORD *)(v1 + 120);
      }
    }
    if ( v3 != v1 + 136 )
      _libc_free(v3);
    v9 = *(_QWORD *)(v1 + 88);
    if ( v9 != v1 + 104 )
      _libc_free(v9);
    v10 = *(_QWORD **)(v1 + 24);
    v11 = &v10[3 * *(unsigned int *)(v1 + 32)];
    if ( v10 != v11 )
    {
      do
      {
        v12 = *(v11 - 1);
        v11 -= 3;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD60C0(v11);
      }
      while ( v10 != v11 );
      v11 = *(_QWORD **)(v1 + 24);
    }
    if ( v11 != (_QWORD *)(v1 + 40) )
      _libc_free((unsigned __int64)v11);
    j_j___libc_free_0(v1);
  }
}
