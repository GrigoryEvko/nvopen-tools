// Function: sub_E38110
// Address: 0xe38110
//
__int64 __fastcall sub_E38110(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 *v10; // r12
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rsi
  _QWORD *v15; // rbx
  _QWORD *v16; // r14
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdi
  __int64 *v23; // [rsp+8h] [rbp-48h]
  __int64 *v24; // [rsp+10h] [rbp-40h]
  __int64 *v25; // [rsp+18h] [rbp-38h]

  v2 = a1 + 192;
  v3 = *(_QWORD *)(a1 + 176);
  if ( v3 != v2 )
    _libc_free(v3, a2);
  v4 = *(_QWORD *)(a1 + 88);
  if ( v4 != a1 + 104 )
    _libc_free(v4, a2);
  v5 = 8LL * *(unsigned int *)(a1 + 80);
  sub_C7D6A0(*(_QWORD *)(a1 + 64), v5, 8);
  v23 = *(__int64 **)(a1 + 40);
  v24 = *(__int64 **)(a1 + 32);
  if ( v23 != v24 )
  {
    do
    {
      v6 = *v24;
      if ( *v24 )
      {
        v7 = *(_QWORD *)(v6 + 176);
        if ( v7 != v6 + 192 )
          _libc_free(v7, v5);
        v8 = *(_QWORD *)(v6 + 88);
        if ( v8 != v6 + 104 )
          _libc_free(v8, v5);
        v9 = 8LL * *(unsigned int *)(v6 + 80);
        sub_C7D6A0(*(_QWORD *)(v6 + 64), v9, 8);
        v10 = *(__int64 **)(v6 + 32);
        v25 = *(__int64 **)(v6 + 40);
        if ( v25 != v10 )
        {
          do
          {
            v11 = *v10;
            if ( *v10 )
            {
              v12 = *(_QWORD *)(v11 + 176);
              if ( v12 != v11 + 192 )
                _libc_free(v12, v9);
              v13 = *(_QWORD *)(v11 + 88);
              if ( v13 != v11 + 104 )
                _libc_free(v13, v9);
              v14 = 8LL * *(unsigned int *)(v11 + 80);
              sub_C7D6A0(*(_QWORD *)(v11 + 64), v14, 8);
              v15 = *(_QWORD **)(v11 + 40);
              v16 = *(_QWORD **)(v11 + 32);
              if ( v15 != v16 )
              {
                do
                {
                  if ( *v16 )
                    sub_E38110();
                  ++v16;
                }
                while ( v15 != v16 );
                v16 = *(_QWORD **)(v11 + 32);
              }
              if ( v16 )
              {
                v14 = *(_QWORD *)(v11 + 48) - (_QWORD)v16;
                j_j___libc_free_0(v16, v14);
              }
              v17 = *(_QWORD *)(v11 + 8);
              if ( v17 != v11 + 24 )
                _libc_free(v17, v14);
              v9 = 224;
              j_j___libc_free_0(v11, 224);
            }
            ++v10;
          }
          while ( v25 != v10 );
          v10 = *(__int64 **)(v6 + 32);
        }
        if ( v10 )
        {
          v9 = *(_QWORD *)(v6 + 48) - (_QWORD)v10;
          j_j___libc_free_0(v10, v9);
        }
        v18 = *(_QWORD *)(v6 + 8);
        if ( v18 != v6 + 24 )
          _libc_free(v18, v9);
        v5 = 224;
        j_j___libc_free_0(v6, 224);
      }
      ++v24;
    }
    while ( v23 != v24 );
    v24 = *(__int64 **)(a1 + 32);
  }
  if ( v24 )
  {
    v19 = *(_QWORD *)(a1 + 48);
    v5 = v19 - (_QWORD)v24;
    j_j___libc_free_0(v24, v19 - (_QWORD)v24);
  }
  v20 = *(_QWORD *)(a1 + 8);
  if ( v20 != a1 + 24 )
    _libc_free(v20, v5);
  return j_j___libc_free_0(a1, 224);
}
