// Function: sub_17B8160
// Address: 0x17b8160
//
__int64 __fastcall sub_17B8160(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  unsigned __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r14
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // r15
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 v17; // rbx
  __int64 v18; // r14
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 232);
  v2 = *(unsigned int *)(a1 + 240);
  *(_QWORD *)a1 = off_49F0128;
  v3 = v1 + 8 * v2;
  if ( v1 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
      {
        v5 = *(_QWORD *)(v4 + 112);
        if ( v5 != v4 + 128 )
          _libc_free(v5);
        v6 = *(_QWORD *)(v4 + 80);
        if ( *(_DWORD *)(v4 + 92) )
        {
          v7 = *(unsigned int *)(v4 + 88);
          if ( (_DWORD)v7 )
          {
            v8 = 8 * v7;
            v9 = 0;
            do
            {
              v10 = *(_QWORD *)(v6 + v9);
              if ( v10 != -8 && v10 )
              {
                v11 = *(_QWORD *)(v10 + 32);
                if ( v11 != v10 + 48 )
                  _libc_free(v11);
                _libc_free(v10);
                v6 = *(_QWORD *)(v4 + 80);
              }
              v9 += 8;
            }
            while ( v8 != v9 );
          }
        }
        _libc_free(v6);
        v12 = *(unsigned int *)(v4 + 56);
        if ( (_DWORD)v12 )
        {
          v13 = *(_QWORD *)(v4 + 40);
          v25 = v13 + 104 * v12;
          do
          {
            if ( *(_QWORD *)v13 != -16 && *(_QWORD *)v13 != -8 )
            {
              v14 = *(_QWORD *)(v13 + 56);
              if ( v14 != v13 + 72 )
                _libc_free(v14);
              v15 = *(_QWORD *)(v13 + 24);
              if ( *(_DWORD *)(v13 + 36) )
              {
                v16 = *(unsigned int *)(v13 + 32);
                if ( (_DWORD)v16 )
                {
                  v17 = 8 * v16;
                  v18 = 0;
                  do
                  {
                    v19 = *(_QWORD *)(v15 + v18);
                    if ( v19 != -8 && v19 )
                    {
                      v20 = *(_QWORD *)(v19 + 32);
                      if ( v20 != v19 + 48 )
                      {
                        v24 = v19;
                        _libc_free(v20);
                        v19 = v24;
                      }
                      _libc_free(v19);
                      v15 = *(_QWORD *)(v13 + 24);
                    }
                    v18 += 8;
                  }
                  while ( v17 != v18 );
                }
              }
              _libc_free(v15);
            }
            v13 += 104;
          }
          while ( v25 != v13 );
        }
        j___libc_free_0(*(_QWORD *)(v4 + 40));
        j_j___libc_free_0(v4, 160);
      }
    }
    while ( v1 != v3 );
    v3 = *(_QWORD *)(a1 + 232);
  }
  if ( v3 != a1 + 248 )
    _libc_free(v3);
  v21 = *(_QWORD *)(a1 + 176);
  if ( v21 != a1 + 192 )
    _libc_free(v21);
  sub_1636790((_QWORD *)a1);
  return j_j___libc_free_0(a1, 376);
}
