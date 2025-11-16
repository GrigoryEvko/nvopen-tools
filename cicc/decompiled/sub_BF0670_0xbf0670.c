// Function: sub_BF0670
// Address: 0xbf0670
//
void __fastcall sub_BF0670(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 *v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 *i; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 *v24; // [rsp+10h] [rbp-40h]
  __int64 *v25; // [rsp+18h] [rbp-38h]

  v24 = a1;
  for ( i = (__int64 *)a2; i != v24; ++v24 )
  {
    v2 = *v24;
    v23 = *v24;
    if ( *v24 )
    {
      v3 = *(_QWORD *)(v2 + 176);
      if ( v3 != v2 + 192 )
        _libc_free(v3, a2);
      v4 = *(_QWORD *)(v23 + 88);
      if ( v4 != v23 + 104 )
        _libc_free(v4, a2);
      v5 = 8LL * *(unsigned int *)(v23 + 80);
      sub_C7D6A0(*(_QWORD *)(v23 + 64), v5, 8);
      v6 = *(__int64 **)(v23 + 32);
      v25 = *(__int64 **)(v23 + 40);
      if ( v25 != v6 )
      {
        do
        {
          v7 = *v6;
          if ( *v6 )
          {
            v8 = *(_QWORD *)(v7 + 176);
            if ( v8 != v7 + 192 )
              _libc_free(v8, v5);
            v9 = *(_QWORD *)(v7 + 88);
            if ( v9 != v7 + 104 )
              _libc_free(v9, v5);
            v10 = 8LL * *(unsigned int *)(v7 + 80);
            sub_C7D6A0(*(_QWORD *)(v7 + 64), v10, 8);
            v11 = *(__int64 **)(v7 + 40);
            v12 = *(__int64 **)(v7 + 32);
            if ( v11 != v12 )
            {
              do
              {
                v13 = *v12;
                if ( *v12 )
                {
                  v14 = *(_QWORD *)(v13 + 176);
                  if ( v14 != v13 + 192 )
                    _libc_free(v14, v10);
                  v15 = *(_QWORD *)(v13 + 88);
                  if ( v15 != v13 + 104 )
                    _libc_free(v15, v10);
                  sub_C7D6A0(*(_QWORD *)(v13 + 64), 8LL * *(unsigned int *)(v13 + 80), 8);
                  v16 = *(_QWORD *)(v13 + 40);
                  sub_BF0670(*(_QWORD *)(v13 + 32), v16);
                  v17 = *(_QWORD *)(v13 + 32);
                  if ( v17 )
                  {
                    v16 = *(_QWORD *)(v13 + 48) - v17;
                    j_j___libc_free_0(v17, v16);
                  }
                  v18 = *(_QWORD *)(v13 + 8);
                  if ( v18 != v13 + 24 )
                    _libc_free(v18, v16);
                  v10 = 224;
                  j_j___libc_free_0(v13, 224);
                }
                ++v12;
              }
              while ( v11 != v12 );
              v12 = *(__int64 **)(v7 + 32);
            }
            if ( v12 )
            {
              v10 = *(_QWORD *)(v7 + 48) - (_QWORD)v12;
              j_j___libc_free_0(v12, v10);
            }
            v19 = *(_QWORD *)(v7 + 8);
            if ( v19 != v7 + 24 )
              _libc_free(v19, v10);
            v5 = 224;
            j_j___libc_free_0(v7, 224);
          }
          ++v6;
        }
        while ( v25 != v6 );
        v6 = *(__int64 **)(v23 + 32);
      }
      if ( v6 )
      {
        v20 = *(_QWORD *)(v23 + 48);
        v5 = v20 - (_QWORD)v6;
        j_j___libc_free_0(v6, v20 - (_QWORD)v6);
      }
      v21 = *(_QWORD *)(v23 + 8);
      if ( v21 != v23 + 24 )
        _libc_free(v21, v5);
      a2 = 224;
      j_j___libc_free_0(v23, 224);
    }
  }
}
