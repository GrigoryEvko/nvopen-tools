// Function: sub_2399F40
// Address: 0x2399f40
//
void __fastcall sub_2399F40(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // r14
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 *i; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v20; // [rsp+18h] [rbp-38h]

  for ( i = a1; a2 != i; ++i )
  {
    v2 = *i;
    v18 = *i;
    if ( *i )
    {
      v3 = *(_QWORD *)(v2 + 176);
      if ( v3 != v2 + 192 )
        _libc_free(v3);
      v4 = *(_QWORD *)(v18 + 88);
      if ( v4 != v18 + 104 )
        _libc_free(v4);
      sub_C7D6A0(*(_QWORD *)(v18 + 64), 8LL * *(unsigned int *)(v18 + 80), 8);
      v5 = *(unsigned __int64 **)(v18 + 32);
      v20 = *(unsigned __int64 **)(v18 + 40);
      if ( v20 != v5 )
      {
        do
        {
          v6 = *v5;
          if ( *v5 )
          {
            v7 = *(_QWORD *)(v6 + 176);
            if ( v7 != v6 + 192 )
              _libc_free(v7);
            v8 = *(_QWORD *)(v6 + 88);
            if ( v8 != v6 + 104 )
              _libc_free(v8);
            sub_C7D6A0(*(_QWORD *)(v6 + 64), 8LL * *(unsigned int *)(v6 + 80), 8);
            v9 = *(unsigned __int64 **)(v6 + 40);
            v10 = *(unsigned __int64 **)(v6 + 32);
            if ( v9 != v10 )
            {
              do
              {
                v11 = *v10;
                if ( *v10 )
                {
                  v12 = *(_QWORD *)(v11 + 176);
                  if ( v12 != v11 + 192 )
                    _libc_free(v12);
                  v13 = *(_QWORD *)(v11 + 88);
                  if ( v13 != v11 + 104 )
                    _libc_free(v13);
                  sub_C7D6A0(*(_QWORD *)(v11 + 64), 8LL * *(unsigned int *)(v11 + 80), 8);
                  sub_2399F40(*(_QWORD *)(v11 + 32), *(_QWORD *)(v11 + 40));
                  v14 = *(_QWORD *)(v11 + 32);
                  if ( v14 )
                    j_j___libc_free_0(v14);
                  v15 = *(_QWORD *)(v11 + 8);
                  if ( v15 != v11 + 24 )
                    _libc_free(v15);
                  j_j___libc_free_0(v11);
                }
                ++v10;
              }
              while ( v9 != v10 );
              v10 = *(unsigned __int64 **)(v6 + 32);
            }
            if ( v10 )
              j_j___libc_free_0((unsigned __int64)v10);
            v16 = *(_QWORD *)(v6 + 8);
            if ( v16 != v6 + 24 )
              _libc_free(v16);
            j_j___libc_free_0(v6);
          }
          ++v5;
        }
        while ( v20 != v5 );
        v5 = *(unsigned __int64 **)(v18 + 32);
      }
      if ( v5 )
        j_j___libc_free_0((unsigned __int64)v5);
      v17 = *(_QWORD *)(v18 + 8);
      if ( v17 != v18 + 24 )
        _libc_free(v17);
      j_j___libc_free_0(v18);
    }
  }
}
