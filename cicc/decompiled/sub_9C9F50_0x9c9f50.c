// Function: sub_9C9F50
// Address: 0x9c9f50
//
__int64 __fastcall sub_9C9F50(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rdi
  _QWORD *v10; // rdi
  _QWORD *v11; // rbx
  _QWORD *v12; // rdi
  _QWORD *v13; // rbx
  _QWORD *v14; // rdi
  _QWORD *v15; // rbx
  _QWORD *v16; // rdi
  _QWORD *v17; // rbx
  _QWORD *v18; // rdi
  _QWORD *v19; // rbx
  _QWORD *v20; // rdi
  _QWORD *v21; // rbx
  _QWORD *v22; // rdi
  _QWORD *v23; // rbx
  _QWORD *v24; // rdi
  _QWORD *v25; // rbx
  __int64 result; // rax
  __int64 v27; // [rsp+0h] [rbp-60h]
  _QWORD *v29; // [rsp+10h] [rbp-50h]
  _QWORD *v30; // [rsp+18h] [rbp-48h]
  _QWORD *v31; // [rsp+20h] [rbp-40h]
  _QWORD *v32; // [rsp+28h] [rbp-38h]

  v29 = (_QWORD *)a2;
  if ( a2 )
  {
    while ( 1 )
    {
      v32 = (_QWORD *)v29[3];
      if ( v32 )
      {
        while ( 1 )
        {
          v31 = (_QWORD *)v32[3];
          if ( v31 )
          {
            while ( 1 )
            {
              v30 = (_QWORD *)v31[3];
              if ( v30 )
              {
                while ( 1 )
                {
                  v2 = (_QWORD *)v30[3];
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
                                    v7 = *(_QWORD *)(v6 + 24);
                                    sub_9C9F50(a1, v7);
                                    v8 = v6;
                                    v6 = *(_QWORD *)(v6 + 16);
                                    v9 = *(_QWORD *)(v8 + 32);
                                    if ( v9 != v8 + 56 )
                                    {
                                      v27 = v8;
                                      _libc_free(v9, v7);
                                      v8 = v27;
                                    }
                                    a2 = 88;
                                    j_j___libc_free_0(v8, 88);
                                  }
                                  v10 = (_QWORD *)v5[4];
                                  v11 = (_QWORD *)v5[2];
                                  if ( v10 != v5 + 7 )
                                    _libc_free(v10, a2);
                                  a2 = 88;
                                  j_j___libc_free_0(v5, 88);
                                  if ( !v11 )
                                    break;
                                  v5 = v11;
                                }
                              }
                              v18 = (_QWORD *)v4[4];
                              v19 = (_QWORD *)v4[2];
                              if ( v18 != v4 + 7 )
                                _libc_free(v18, a2);
                              a2 = 88;
                              j_j___libc_free_0(v4, 88);
                              if ( !v19 )
                                break;
                              v4 = v19;
                            }
                          }
                          v14 = (_QWORD *)v3[4];
                          v15 = (_QWORD *)v3[2];
                          if ( v14 != v3 + 7 )
                            _libc_free(v14, a2);
                          a2 = 88;
                          j_j___libc_free_0(v3, 88);
                          if ( !v15 )
                            break;
                          v3 = v15;
                        }
                      }
                      v12 = (_QWORD *)v2[4];
                      v13 = (_QWORD *)v2[2];
                      if ( v12 != v2 + 7 )
                        _libc_free(v12, a2);
                      a2 = 88;
                      j_j___libc_free_0(v2, 88);
                      if ( !v13 )
                        break;
                      v2 = v13;
                    }
                  }
                  v16 = (_QWORD *)v30[4];
                  v17 = (_QWORD *)v30[2];
                  if ( v16 != v30 + 7 )
                    _libc_free(v16, a2);
                  a2 = 88;
                  j_j___libc_free_0(v30, 88);
                  if ( !v17 )
                    break;
                  v30 = v17;
                }
              }
              v20 = (_QWORD *)v31[4];
              v21 = (_QWORD *)v31[2];
              if ( v20 != v31 + 7 )
                _libc_free(v20, a2);
              a2 = 88;
              j_j___libc_free_0(v31, 88);
              if ( !v21 )
                break;
              v31 = v21;
            }
          }
          v22 = (_QWORD *)v32[4];
          v23 = (_QWORD *)v32[2];
          if ( v22 != v32 + 7 )
            _libc_free(v22, a2);
          a2 = 88;
          j_j___libc_free_0(v32, 88);
          if ( !v23 )
            break;
          v32 = v23;
        }
      }
      v24 = (_QWORD *)v29[4];
      v25 = (_QWORD *)v29[2];
      if ( v24 != v29 + 7 )
        _libc_free(v24, a2);
      a2 = 88;
      result = j_j___libc_free_0(v29, 88);
      if ( !v25 )
        break;
      v29 = v25;
    }
  }
  return result;
}
