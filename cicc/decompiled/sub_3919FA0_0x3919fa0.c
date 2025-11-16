// Function: sub_3919FA0
// Address: 0x3919fa0
//
void __fastcall sub_3919FA0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r9
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  _QWORD *v10; // rbx
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rbx
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rbx
  unsigned __int64 v15; // rdi
  _QWORD *v16; // rbx
  unsigned __int64 v17; // rdi
  _QWORD *v18; // rbx
  unsigned __int64 v19; // rdi
  _QWORD *v20; // rbx
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rbx
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rbx
  unsigned __int64 v25; // [rsp+0h] [rbp-60h]
  _QWORD *v27; // [rsp+10h] [rbp-50h]
  _QWORD *v28; // [rsp+18h] [rbp-48h]
  _QWORD *v29; // [rsp+20h] [rbp-40h]
  _QWORD *v30; // [rsp+28h] [rbp-38h]

  v27 = a2;
  if ( a2 )
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
                                    sub_3919FA0(a1, *(_QWORD *)(v6 + 24));
                                    v7 = v6;
                                    v6 = *(_QWORD *)(v6 + 16);
                                    v8 = *(_QWORD *)(v7 + 48);
                                    if ( v8 )
                                    {
                                      v25 = v7;
                                      j_j___libc_free_0(v8);
                                      v7 = v25;
                                    }
                                    j_j___libc_free_0(v7);
                                  }
                                  v9 = v5[6];
                                  v10 = (_QWORD *)v5[2];
                                  if ( v9 )
                                    j_j___libc_free_0(v9);
                                  j_j___libc_free_0((unsigned __int64)v5);
                                  if ( !v10 )
                                    break;
                                  v5 = v10;
                                }
                              }
                              v17 = v4[6];
                              v18 = (_QWORD *)v4[2];
                              if ( v17 )
                                j_j___libc_free_0(v17);
                              j_j___libc_free_0((unsigned __int64)v4);
                              if ( !v18 )
                                break;
                              v4 = v18;
                            }
                          }
                          v13 = v3[6];
                          v14 = (_QWORD *)v3[2];
                          if ( v13 )
                            j_j___libc_free_0(v13);
                          j_j___libc_free_0((unsigned __int64)v3);
                          if ( !v14 )
                            break;
                          v3 = v14;
                        }
                      }
                      v11 = v2[6];
                      v12 = (_QWORD *)v2[2];
                      if ( v11 )
                        j_j___libc_free_0(v11);
                      j_j___libc_free_0((unsigned __int64)v2);
                      if ( !v12 )
                        break;
                      v2 = v12;
                    }
                  }
                  v15 = v28[6];
                  v16 = (_QWORD *)v28[2];
                  if ( v15 )
                    j_j___libc_free_0(v15);
                  j_j___libc_free_0((unsigned __int64)v28);
                  if ( !v16 )
                    break;
                  v28 = v16;
                }
              }
              v19 = v29[6];
              v20 = (_QWORD *)v29[2];
              if ( v19 )
                j_j___libc_free_0(v19);
              j_j___libc_free_0((unsigned __int64)v29);
              if ( !v20 )
                break;
              v29 = v20;
            }
          }
          v21 = v30[6];
          v22 = (_QWORD *)v30[2];
          if ( v21 )
            j_j___libc_free_0(v21);
          j_j___libc_free_0((unsigned __int64)v30);
          if ( !v22 )
            break;
          v30 = v22;
        }
      }
      v23 = v27[6];
      v24 = (_QWORD *)v27[2];
      if ( v23 )
        j_j___libc_free_0(v23);
      j_j___libc_free_0((unsigned __int64)v27);
      if ( !v24 )
        break;
      v27 = v24;
    }
  }
}
