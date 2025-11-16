// Function: sub_11DA7F0
// Address: 0x11da7f0
//
__int64 __fastcall sub_11DA7F0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 v8; // rdi
  _QWORD *v9; // rdi
  _QWORD *v10; // rbx
  _QWORD *v11; // rdi
  _QWORD *v12; // rbx
  _QWORD *v13; // rdi
  _QWORD *v14; // rbx
  _QWORD *v15; // rdi
  _QWORD *v16; // rbx
  _QWORD *v17; // rdi
  _QWORD *v18; // rbx
  _QWORD *v19; // rdi
  _QWORD *v20; // rbx
  _QWORD *v21; // rdi
  _QWORD *v22; // rbx
  _QWORD *v23; // rdi
  _QWORD *v24; // rbx
  __int64 result; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  _QWORD *v27; // [rsp+10h] [rbp-50h]
  _QWORD *v28; // [rsp+18h] [rbp-48h]
  _QWORD *v29; // [rsp+20h] [rbp-40h]
  _QWORD *v30; // [rsp+28h] [rbp-38h]

  v27 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v30 = (_QWORD *)v27[3];
      if ( v30 )
      {
        while ( 1 )
        {
          v29 = (_QWORD *)v30[3];
          if ( v29 )
          {
            while ( 1 )
            {
              v28 = (_QWORD *)v29[3];
              if ( v28 )
              {
                while ( 1 )
                {
                  v2 = (_QWORD *)v28[3];
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
                              v5 = (_QWORD *)v4[3];
                              if ( v5 )
                              {
                                while ( 1 )
                                {
                                  v6 = v5[3];
                                  while ( v6 )
                                  {
                                    sub_11DA7F0(*(_QWORD *)(v6 + 24));
                                    v7 = v6;
                                    v6 = *(_QWORD *)(v6 + 16);
                                    v8 = *(_QWORD *)(v7 + 32);
                                    if ( v8 != v7 + 56 )
                                    {
                                      v26 = v7;
                                      _libc_free(v8, a2);
                                      v7 = v26;
                                    }
                                    a2 = 88;
                                    j_j___libc_free_0(v7, 88);
                                  }
                                  v9 = (_QWORD *)v5[4];
                                  v10 = (_QWORD *)v5[2];
                                  if ( v9 != v5 + 7 )
                                    _libc_free(v9, a2);
                                  a2 = 88;
                                  j_j___libc_free_0(v5, 88);
                                  if ( !v10 )
                                    break;
                                  v5 = v10;
                                }
                              }
                              v17 = (_QWORD *)v4[4];
                              v18 = (_QWORD *)v4[2];
                              if ( v17 != v4 + 7 )
                                _libc_free(v17, a2);
                              a2 = 88;
                              j_j___libc_free_0(v4, 88);
                              if ( !v18 )
                                break;
                              v4 = v18;
                            }
                          }
                          v13 = (_QWORD *)v3[4];
                          v14 = (_QWORD *)v3[2];
                          if ( v13 != v3 + 7 )
                            _libc_free(v13, a2);
                          a2 = 88;
                          j_j___libc_free_0(v3, 88);
                          if ( !v14 )
                            break;
                          v3 = v14;
                        }
                      }
                      v11 = (_QWORD *)v2[4];
                      v12 = (_QWORD *)v2[2];
                      if ( v11 != v2 + 7 )
                        _libc_free(v11, a2);
                      a2 = 88;
                      j_j___libc_free_0(v2, 88);
                      if ( !v12 )
                        break;
                      v2 = v12;
                    }
                  }
                  v15 = (_QWORD *)v28[4];
                  v16 = (_QWORD *)v28[2];
                  if ( v15 != v28 + 7 )
                    _libc_free(v15, a2);
                  a2 = 88;
                  j_j___libc_free_0(v28, 88);
                  if ( !v16 )
                    break;
                  v28 = v16;
                }
              }
              v19 = (_QWORD *)v29[4];
              v20 = (_QWORD *)v29[2];
              if ( v19 != v29 + 7 )
                _libc_free(v19, a2);
              a2 = 88;
              j_j___libc_free_0(v29, 88);
              if ( !v20 )
                break;
              v29 = v20;
            }
          }
          v21 = (_QWORD *)v30[4];
          v22 = (_QWORD *)v30[2];
          if ( v21 != v30 + 7 )
            _libc_free(v21, a2);
          a2 = 88;
          j_j___libc_free_0(v30, 88);
          if ( !v22 )
            break;
          v30 = v22;
        }
      }
      v23 = (_QWORD *)v27[4];
      v24 = (_QWORD *)v27[2];
      if ( v23 != v27 + 7 )
        _libc_free(v23, a2);
      a2 = 88;
      result = j_j___libc_free_0(v27, 88);
      if ( !v24 )
        break;
      v27 = v24;
    }
  }
  return result;
}
