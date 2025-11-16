// Function: sub_D690E0
// Address: 0xd690e0
//
void __fastcall sub_D690E0(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // rax
  _QWORD *v13; // rbx
  __int64 v14; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rax
  _QWORD *v17; // rbx
  __int64 v18; // rax
  _QWORD *v19; // rbx
  __int64 v20; // rax
  _QWORD *v21; // rbx
  _QWORD *v22; // rbx
  __int64 v23; // [rsp+8h] [rbp-58h]
  _QWORD *v24; // [rsp+10h] [rbp-50h]
  _QWORD *v25; // [rsp+18h] [rbp-48h]
  _QWORD *v26; // [rsp+20h] [rbp-40h]
  _QWORD *v27; // [rsp+28h] [rbp-38h]

  v24 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v27 = (_QWORD *)v24[3];
      if ( v27 )
      {
        while ( 1 )
        {
          v26 = (_QWORD *)v27[3];
          if ( v26 )
          {
            while ( 1 )
            {
              v25 = (_QWORD *)v26[3];
              if ( v25 )
              {
                while ( 1 )
                {
                  v1 = (_QWORD *)v25[3];
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
                                    sub_D690E0(*(_QWORD *)(v5 + 24));
                                    v6 = v5;
                                    v5 = *(_QWORD *)(v5 + 16);
                                    v7 = *(_QWORD *)(v6 + 48);
                                    if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
                                    {
                                      v23 = v6;
                                      sub_BD60C0((_QWORD *)(v6 + 32));
                                      v6 = v23;
                                    }
                                    j_j___libc_free_0(v6, 56);
                                  }
                                  v8 = v4[6];
                                  v9 = (_QWORD *)v4[2];
                                  if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
                                    sub_BD60C0(v4 + 4);
                                  j_j___libc_free_0(v4, 56);
                                  if ( !v9 )
                                    break;
                                  v4 = v9;
                                }
                              }
                              v16 = v3[6];
                              v17 = (_QWORD *)v3[2];
                              if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
                                sub_BD60C0(v3 + 4);
                              j_j___libc_free_0(v3, 56);
                              if ( !v17 )
                                break;
                              v3 = v17;
                            }
                          }
                          v12 = v2[6];
                          v13 = (_QWORD *)v2[2];
                          if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
                            sub_BD60C0(v2 + 4);
                          j_j___libc_free_0(v2, 56);
                          if ( !v13 )
                            break;
                          v2 = v13;
                        }
                      }
                      v10 = v1[6];
                      v11 = (_QWORD *)v1[2];
                      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
                        sub_BD60C0(v1 + 4);
                      j_j___libc_free_0(v1, 56);
                      if ( !v11 )
                        break;
                      v1 = v11;
                    }
                  }
                  v14 = v25[6];
                  v15 = (_QWORD *)v25[2];
                  if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
                    sub_BD60C0(v25 + 4);
                  j_j___libc_free_0(v25, 56);
                  if ( !v15 )
                    break;
                  v25 = v15;
                }
              }
              v18 = v26[6];
              v19 = (_QWORD *)v26[2];
              if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
                sub_BD60C0(v26 + 4);
              j_j___libc_free_0(v26, 56);
              if ( !v19 )
                break;
              v26 = v19;
            }
          }
          v20 = v27[6];
          v21 = (_QWORD *)v27[2];
          if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
            sub_BD60C0(v27 + 4);
          j_j___libc_free_0(v27, 56);
          if ( !v21 )
            break;
          v27 = v21;
        }
      }
      v22 = (_QWORD *)v24[2];
      sub_D68D70(v24 + 4);
      j_j___libc_free_0(v24, 56);
      if ( !v22 )
        break;
      v24 = v22;
    }
  }
}
