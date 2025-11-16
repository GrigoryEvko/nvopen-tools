// Function: sub_17B7F60
// Address: 0x17b7f60
//
void *__fastcall sub_17B7F60(__int64 a1)
{
  __int64 v1; // rsi
  unsigned __int64 v2; // r13
  __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r15
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // r14
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = off_49F0128;
  v1 = *(_QWORD *)(a1 + 232);
  v2 = v1 + 8LL * *(unsigned int *)(a1 + 240);
  if ( v1 != v2 )
  {
    do
    {
      v3 = *(_QWORD *)(v2 - 8);
      v2 -= 8LL;
      if ( v3 )
      {
        v4 = *(_QWORD *)(v3 + 112);
        if ( v4 != v3 + 128 )
          _libc_free(v4);
        v5 = *(_QWORD *)(v3 + 80);
        if ( *(_DWORD *)(v3 + 92) )
        {
          v6 = *(unsigned int *)(v3 + 88);
          if ( (_DWORD)v6 )
          {
            v7 = 8 * v6;
            v8 = 0;
            do
            {
              v9 = *(_QWORD *)(v5 + v8);
              if ( v9 != -8 && v9 )
              {
                v10 = *(_QWORD *)(v9 + 32);
                if ( v10 != v9 + 48 )
                  _libc_free(v10);
                _libc_free(v9);
                v5 = *(_QWORD *)(v3 + 80);
              }
              v8 += 8;
            }
            while ( v7 != v8 );
          }
        }
        _libc_free(v5);
        v11 = *(unsigned int *)(v3 + 56);
        if ( (_DWORD)v11 )
        {
          v12 = *(_QWORD *)(v3 + 40);
          v24 = v12 + 104 * v11;
          do
          {
            if ( *(_QWORD *)v12 != -16 && *(_QWORD *)v12 != -8 )
            {
              v13 = *(_QWORD *)(v12 + 56);
              if ( v13 != v12 + 72 )
                _libc_free(v13);
              v14 = *(_QWORD *)(v12 + 24);
              if ( *(_DWORD *)(v12 + 36) )
              {
                v15 = *(unsigned int *)(v12 + 32);
                if ( (_DWORD)v15 )
                {
                  v16 = 8 * v15;
                  v17 = 0;
                  do
                  {
                    v18 = *(_QWORD *)(v14 + v17);
                    if ( v18 != -8 && v18 )
                    {
                      v19 = *(_QWORD *)(v18 + 32);
                      if ( v19 != v18 + 48 )
                      {
                        v23 = v18;
                        _libc_free(v19);
                        v18 = v23;
                      }
                      _libc_free(v18);
                      v14 = *(_QWORD *)(v12 + 24);
                    }
                    v17 += 8;
                  }
                  while ( v16 != v17 );
                }
              }
              _libc_free(v14);
            }
            v12 += 104;
          }
          while ( v24 != v12 );
        }
        j___libc_free_0(*(_QWORD *)(v3 + 40));
        j_j___libc_free_0(v3, 160);
      }
    }
    while ( v1 != v2 );
    v2 = *(_QWORD *)(a1 + 232);
  }
  if ( v2 != a1 + 248 )
    _libc_free(v2);
  v20 = *(_QWORD *)(a1 + 176);
  if ( v20 != a1 + 192 )
    _libc_free(v20);
  return sub_1636790((_QWORD *)a1);
}
