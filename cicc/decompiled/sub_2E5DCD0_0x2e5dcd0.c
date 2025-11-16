// Function: sub_2E5DCD0
// Address: 0x2e5dcd0
//
void __fastcall sub_2E5DCD0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rbx
  _QWORD *v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 *v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v18; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v19; // [rsp+18h] [rbp-38h]

  v1 = a1 + 192;
  v2 = *(_QWORD *)(a1 + 176);
  if ( v2 != v1 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 88);
  if ( v3 != a1 + 104 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 8LL * *(unsigned int *)(a1 + 80), 8);
  v17 = *(unsigned __int64 **)(a1 + 40);
  v18 = *(unsigned __int64 **)(a1 + 32);
  if ( v17 != v18 )
  {
    do
    {
      v4 = *v18;
      if ( *v18 )
      {
        v5 = *(_QWORD *)(v4 + 176);
        if ( v5 != v4 + 192 )
          _libc_free(v5);
        v6 = *(_QWORD *)(v4 + 88);
        if ( v6 != v4 + 104 )
          _libc_free(v6);
        sub_C7D6A0(*(_QWORD *)(v4 + 64), 8LL * *(unsigned int *)(v4 + 80), 8);
        v7 = *(unsigned __int64 **)(v4 + 32);
        v19 = *(unsigned __int64 **)(v4 + 40);
        if ( v19 != v7 )
        {
          do
          {
            v8 = *v7;
            if ( *v7 )
            {
              v9 = *(_QWORD *)(v8 + 176);
              if ( v9 != v8 + 192 )
                _libc_free(v9);
              v10 = *(_QWORD *)(v8 + 88);
              if ( v10 != v8 + 104 )
                _libc_free(v10);
              sub_C7D6A0(*(_QWORD *)(v8 + 64), 8LL * *(unsigned int *)(v8 + 80), 8);
              v11 = *(_QWORD **)(v8 + 40);
              v12 = *(_QWORD **)(v8 + 32);
              if ( v11 != v12 )
              {
                do
                {
                  if ( *v12 )
                    sub_2E5DCD0();
                  ++v12;
                }
                while ( v11 != v12 );
                v12 = *(_QWORD **)(v8 + 32);
              }
              if ( v12 )
                j_j___libc_free_0((unsigned __int64)v12);
              v13 = *(_QWORD *)(v8 + 8);
              if ( v13 != v8 + 24 )
                _libc_free(v13);
              j_j___libc_free_0(v8);
            }
            ++v7;
          }
          while ( v19 != v7 );
          v7 = *(unsigned __int64 **)(v4 + 32);
        }
        if ( v7 )
          j_j___libc_free_0((unsigned __int64)v7);
        v14 = *(_QWORD *)(v4 + 8);
        if ( v14 != v4 + 24 )
          _libc_free(v14);
        j_j___libc_free_0(v4);
      }
      ++v18;
    }
    while ( v17 != v18 );
    v18 = *(unsigned __int64 **)(a1 + 32);
  }
  if ( v18 )
    j_j___libc_free_0((unsigned __int64)v18);
  v15 = *(_QWORD *)(a1 + 8);
  if ( v15 != a1 + 24 )
    _libc_free(v15);
  j_j___libc_free_0(a1);
}
