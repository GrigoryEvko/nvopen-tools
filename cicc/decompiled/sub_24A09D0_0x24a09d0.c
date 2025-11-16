// Function: sub_24A09D0
// Address: 0x24a09d0
//
void __fastcall sub_24A09D0(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r15
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r9
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rbx
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rbx
  unsigned __int64 v14; // rdi
  _QWORD *v15; // rbx
  unsigned __int64 v16; // rdi
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rbx
  unsigned __int64 v20; // rdi
  _QWORD *v21; // rbx
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rbx
  unsigned __int64 v24; // [rsp+8h] [rbp-58h]
  _QWORD *v25; // [rsp+10h] [rbp-50h]
  _QWORD *v26; // [rsp+18h] [rbp-48h]
  _QWORD *v27; // [rsp+20h] [rbp-40h]
  _QWORD *v28; // [rsp+28h] [rbp-38h]

  v25 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v28 = (_QWORD *)v25[3];
      if ( v28 )
      {
        while ( 1 )
        {
          v27 = (_QWORD *)v28[3];
          if ( v27 )
          {
            while ( 1 )
            {
              v26 = (_QWORD *)v27[3];
              if ( v26 )
              {
                while ( 1 )
                {
                  v1 = (_QWORD *)v26[3];
                  if ( v1 )
                  {
                    while ( 1 )
                    {
                      v2 = (_QWORD *)v1[3];
                      if ( v2 )
                      {
                        while ( 1 )
                        {
                          v3 = (_QWORD *)v2[3];
                          if ( v3 )
                          {
                            while ( 1 )
                            {
                              v4 = (_QWORD *)v3[3];
                              if ( v4 )
                              {
                                while ( 1 )
                                {
                                  v5 = v4[3];
                                  while ( v5 )
                                  {
                                    sub_24A09D0(*(_QWORD *)(v5 + 24));
                                    v6 = v5;
                                    v5 = *(_QWORD *)(v5 + 16);
                                    v7 = *(_QWORD *)(v6 + 40);
                                    if ( v7 != v6 + 56 )
                                    {
                                      v24 = v6;
                                      _libc_free(v7);
                                      v6 = v24;
                                    }
                                    j_j___libc_free_0(v6);
                                  }
                                  v8 = v4[5];
                                  v9 = (_QWORD *)v4[2];
                                  if ( (_QWORD *)v8 != v4 + 7 )
                                    _libc_free(v8);
                                  j_j___libc_free_0((unsigned __int64)v4);
                                  if ( !v9 )
                                    break;
                                  v4 = v9;
                                }
                              }
                              v16 = v3[5];
                              v17 = (_QWORD *)v3[2];
                              if ( (_QWORD *)v16 != v3 + 7 )
                                _libc_free(v16);
                              j_j___libc_free_0((unsigned __int64)v3);
                              if ( !v17 )
                                break;
                              v3 = v17;
                            }
                          }
                          v12 = v2[5];
                          v13 = (_QWORD *)v2[2];
                          if ( (_QWORD *)v12 != v2 + 7 )
                            _libc_free(v12);
                          j_j___libc_free_0((unsigned __int64)v2);
                          if ( !v13 )
                            break;
                          v2 = v13;
                        }
                      }
                      v10 = v1[5];
                      v11 = (_QWORD *)v1[2];
                      if ( (_QWORD *)v10 != v1 + 7 )
                        _libc_free(v10);
                      j_j___libc_free_0((unsigned __int64)v1);
                      if ( !v11 )
                        break;
                      v1 = v11;
                    }
                  }
                  v14 = v26[5];
                  v15 = (_QWORD *)v26[2];
                  if ( (_QWORD *)v14 != v26 + 7 )
                    _libc_free(v14);
                  j_j___libc_free_0((unsigned __int64)v26);
                  if ( !v15 )
                    break;
                  v26 = v15;
                }
              }
              v18 = v27[5];
              v19 = (_QWORD *)v27[2];
              if ( (_QWORD *)v18 != v27 + 7 )
                _libc_free(v18);
              j_j___libc_free_0((unsigned __int64)v27);
              if ( !v19 )
                break;
              v27 = v19;
            }
          }
          v20 = v28[5];
          v21 = (_QWORD *)v28[2];
          if ( (_QWORD *)v20 != v28 + 7 )
            _libc_free(v20);
          j_j___libc_free_0((unsigned __int64)v28);
          if ( !v21 )
            break;
          v28 = v21;
        }
      }
      v22 = v25[5];
      v23 = (_QWORD *)v25[2];
      if ( (_QWORD *)v22 != v25 + 7 )
        _libc_free(v22);
      j_j___libc_free_0((unsigned __int64)v25);
      if ( !v23 )
        break;
      v25 = v23;
    }
  }
}
