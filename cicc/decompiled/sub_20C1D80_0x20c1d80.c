// Function: sub_20C1D80
// Address: 0x20c1d80
//
void __fastcall sub_20C1D80(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r12
  _QWORD *v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // r12
  _QWORD *v9; // r12
  _QWORD *v10; // rbx
  _QWORD *v11; // rbx
  _QWORD *v12; // rbx
  __int64 v13; // [rsp+8h] [rbp-58h]
  _QWORD *v14; // [rsp+10h] [rbp-50h]
  _QWORD *v15; // [rsp+18h] [rbp-48h]
  _QWORD *v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]
  _QWORD *v18; // [rsp+28h] [rbp-38h]
  _QWORD *v19; // [rsp+28h] [rbp-38h]

  v14 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v16 = (_QWORD *)v14[3];
      if ( v16 )
      {
        while ( 1 )
        {
          v15 = (_QWORD *)v16[3];
          if ( v15 )
          {
            while ( 1 )
            {
              v1 = (_QWORD *)v15[3];
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
                              v5 = (_QWORD *)v4[3];
                              if ( v5 )
                              {
                                while ( 1 )
                                {
                                  v6 = v5[3];
                                  if ( v6 )
                                  {
                                    do
                                    {
                                      v17 = v6;
                                      sub_20C1D80(*(_QWORD *)(v6 + 24));
                                      v13 = *(_QWORD *)(v17 + 16);
                                      _libc_free(*(_QWORD *)(v17 + 40));
                                      j_j___libc_free_0(v17, 64);
                                      v6 = v13;
                                    }
                                    while ( v13 );
                                  }
                                  v18 = (_QWORD *)v5[2];
                                  _libc_free(v5[5]);
                                  j_j___libc_free_0(v5, 64);
                                  if ( !v18 )
                                    break;
                                  v5 = v18;
                                }
                              }
                              v19 = (_QWORD *)v4[2];
                              _libc_free(v4[5]);
                              j_j___libc_free_0(v4, 64);
                              if ( !v19 )
                                break;
                              v4 = v19;
                            }
                          }
                          v8 = (_QWORD *)v3[2];
                          _libc_free(v3[5]);
                          j_j___libc_free_0(v3, 64);
                          if ( !v8 )
                            break;
                          v3 = v8;
                        }
                      }
                      v7 = (_QWORD *)v2[2];
                      _libc_free(v2[5]);
                      j_j___libc_free_0(v2, 64);
                      if ( !v7 )
                        break;
                      v2 = v7;
                    }
                  }
                  v9 = (_QWORD *)v1[2];
                  _libc_free(v1[5]);
                  j_j___libc_free_0(v1, 64);
                  if ( !v9 )
                    break;
                  v1 = v9;
                }
              }
              v10 = (_QWORD *)v15[2];
              _libc_free(v15[5]);
              j_j___libc_free_0(v15, 64);
              if ( !v10 )
                break;
              v15 = v10;
            }
          }
          v11 = (_QWORD *)v16[2];
          _libc_free(v16[5]);
          j_j___libc_free_0(v16, 64);
          if ( !v11 )
            break;
          v16 = v11;
        }
      }
      v12 = (_QWORD *)v14[2];
      _libc_free(v14[5]);
      j_j___libc_free_0(v14, 64);
      if ( !v12 )
        break;
      v14 = v12;
    }
  }
}
