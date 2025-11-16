// Function: sub_C600E0
// Address: 0xc600e0
//
void __fastcall sub_C600E0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  _QWORD *v6; // rax
  _QWORD *v7; // r9
  _QWORD *v8; // rax
  _QWORD *v9; // rdi
  _QWORD *v10; // rdi
  _QWORD *v11; // rdi
  _QWORD *v12; // r13
  _QWORD *v13; // rdi
  _QWORD *v14; // r14
  _QWORD *v15; // rdi
  _QWORD *v16; // r12
  _QWORD *v17; // rdi
  _QWORD *v18; // r15
  _QWORD *v19; // rdi
  _QWORD *v20; // rbx
  _QWORD *v21; // rdi
  _QWORD *v22; // rbx
  _QWORD *v23; // rdi
  _QWORD *v24; // rbx
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  _QWORD *v26; // [rsp+10h] [rbp-50h]
  _QWORD *v27; // [rsp+18h] [rbp-48h]
  _QWORD *v28; // [rsp+20h] [rbp-40h]
  _QWORD *v29; // [rsp+28h] [rbp-38h]
  _QWORD *v30; // [rsp+28h] [rbp-38h]
  _QWORD *v31; // [rsp+28h] [rbp-38h]

  v26 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v28 = (_QWORD *)v26[3];
      if ( v28 )
      {
        while ( 1 )
        {
          v27 = (_QWORD *)v28[3];
          if ( v27 )
          {
            while ( 1 )
            {
              v1 = (_QWORD *)v27[3];
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
                                  v6 = (_QWORD *)v5[3];
                                  if ( v6 )
                                  {
                                    do
                                    {
                                      v29 = v6;
                                      sub_C600E0(v6[3]);
                                      v7 = v29;
                                      v8 = (_QWORD *)v29[2];
                                      v9 = (_QWORD *)v29[4];
                                      if ( v9 != v29 + 6 )
                                      {
                                        v25 = (_QWORD *)v29[2];
                                        j_j___libc_free_0(v9, v29[6] + 1LL);
                                        v8 = v25;
                                        v7 = v29;
                                      }
                                      v30 = v8;
                                      j_j___libc_free_0(v7, 72);
                                      v6 = v30;
                                    }
                                    while ( v30 );
                                  }
                                  v10 = (_QWORD *)v5[4];
                                  v31 = (_QWORD *)v5[2];
                                  if ( v10 != v5 + 6 )
                                    j_j___libc_free_0(v10, v5[6] + 1LL);
                                  j_j___libc_free_0(v5, 72);
                                  if ( !v31 )
                                    break;
                                  v5 = v31;
                                }
                              }
                              v17 = (_QWORD *)v4[4];
                              v18 = (_QWORD *)v4[2];
                              if ( v17 != v4 + 6 )
                                j_j___libc_free_0(v17, v4[6] + 1LL);
                              j_j___libc_free_0(v4, 72);
                              if ( !v18 )
                                break;
                              v4 = v18;
                            }
                          }
                          v13 = (_QWORD *)v3[4];
                          v14 = (_QWORD *)v3[2];
                          if ( v13 != v3 + 6 )
                            j_j___libc_free_0(v13, v3[6] + 1LL);
                          j_j___libc_free_0(v3, 72);
                          if ( !v14 )
                            break;
                          v3 = v14;
                        }
                      }
                      v11 = (_QWORD *)v2[4];
                      v12 = (_QWORD *)v2[2];
                      if ( v11 != v2 + 6 )
                        j_j___libc_free_0(v11, v2[6] + 1LL);
                      j_j___libc_free_0(v2, 72);
                      if ( !v12 )
                        break;
                      v2 = v12;
                    }
                  }
                  v15 = (_QWORD *)v1[4];
                  v16 = (_QWORD *)v1[2];
                  if ( v15 != v1 + 6 )
                    j_j___libc_free_0(v15, v1[6] + 1LL);
                  j_j___libc_free_0(v1, 72);
                  if ( !v16 )
                    break;
                  v1 = v16;
                }
              }
              v19 = (_QWORD *)v27[4];
              v20 = (_QWORD *)v27[2];
              if ( v19 != v27 + 6 )
                j_j___libc_free_0(v19, v27[6] + 1LL);
              j_j___libc_free_0(v27, 72);
              if ( !v20 )
                break;
              v27 = v20;
            }
          }
          v21 = (_QWORD *)v28[4];
          v22 = (_QWORD *)v28[2];
          if ( v21 != v28 + 6 )
            j_j___libc_free_0(v21, v28[6] + 1LL);
          j_j___libc_free_0(v28, 72);
          if ( !v22 )
            break;
          v28 = v22;
        }
      }
      v23 = (_QWORD *)v26[4];
      v24 = (_QWORD *)v26[2];
      if ( v23 != v26 + 6 )
        j_j___libc_free_0(v23, v26[6] + 1LL);
      j_j___libc_free_0(v26, 72);
      if ( !v24 )
        break;
      v26 = v24;
    }
  }
}
