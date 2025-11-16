// Function: sub_1C97640
// Address: 0x1c97640
//
void __fastcall sub_1C97640(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r15
  _QWORD *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  _QWORD *v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // r15
  _QWORD *v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // r13
  _QWORD *v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // rdi
  __int64 v24; // rbx
  _QWORD *v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rbx
  _QWORD *v28; // r12
  __int64 v29; // rdi
  __int64 v30; // rbx
  _QWORD *v31; // r12
  __int64 v32; // rdi
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  _QWORD *v41; // [rsp+10h] [rbp-50h]
  _QWORD *v42; // [rsp+10h] [rbp-50h]
  _QWORD *v43; // [rsp+18h] [rbp-48h]
  _QWORD *v44; // [rsp+20h] [rbp-40h]
  _QWORD *v45; // [rsp+28h] [rbp-38h]

  v43 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v45 = (_QWORD *)v43[3];
      if ( v45 )
      {
        while ( 1 )
        {
          v44 = (_QWORD *)v45[3];
          if ( v44 )
          {
            while ( 1 )
            {
              v1 = (_QWORD *)v44[3];
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
                                      v37 = v6;
                                      sub_1C97640(*(_QWORD *)(v6 + 24));
                                      v7 = v37;
                                      v8 = *(_QWORD *)(v37 + 16);
                                      v9 = *(_QWORD *)(v37 + 64);
                                      if ( v9 )
                                      {
                                        do
                                        {
                                          v33 = v7;
                                          v34 = v8;
                                          v38 = v9;
                                          sub_1C97470(*(_QWORD *)(v9 + 24));
                                          v10 = v38;
                                          v39 = *(_QWORD *)(v38 + 16);
                                          j_j___libc_free_0(v10, 40);
                                          v9 = v39;
                                          v8 = v34;
                                          v7 = v33;
                                        }
                                        while ( v39 );
                                      }
                                      v40 = v8;
                                      j_j___libc_free_0(v7, 96);
                                      v6 = v40;
                                    }
                                    while ( v40 );
                                  }
                                  v41 = (_QWORD *)v5[2];
                                  v11 = v5[8];
                                  if ( v11 )
                                  {
                                    do
                                    {
                                      v35 = v11;
                                      sub_1C97470(*(_QWORD *)(v11 + 24));
                                      v12 = v35;
                                      v36 = *(_QWORD *)(v35 + 16);
                                      j_j___libc_free_0(v12, 40);
                                      v11 = v36;
                                    }
                                    while ( v36 );
                                  }
                                  j_j___libc_free_0(v5, 96);
                                  if ( !v41 )
                                    break;
                                  v5 = v41;
                                }
                              }
                              v22 = v4[8];
                              v42 = (_QWORD *)v4[2];
                              while ( v22 )
                              {
                                sub_1C97470(*(_QWORD *)(v22 + 24));
                                v23 = v22;
                                v22 = *(_QWORD *)(v22 + 16);
                                j_j___libc_free_0(v23, 40);
                              }
                              j_j___libc_free_0(v4, 96);
                              if ( !v42 )
                                break;
                              v4 = v42;
                            }
                          }
                          v16 = v3[8];
                          v17 = (_QWORD *)v3[2];
                          while ( v16 )
                          {
                            sub_1C97470(*(_QWORD *)(v16 + 24));
                            v18 = v16;
                            v16 = *(_QWORD *)(v16 + 16);
                            j_j___libc_free_0(v18, 40);
                          }
                          j_j___libc_free_0(v3, 96);
                          if ( !v17 )
                            break;
                          v3 = v17;
                        }
                      }
                      v13 = v2[8];
                      v14 = (_QWORD *)v2[2];
                      while ( v13 )
                      {
                        sub_1C97470(*(_QWORD *)(v13 + 24));
                        v15 = v13;
                        v13 = *(_QWORD *)(v13 + 16);
                        j_j___libc_free_0(v15, 40);
                      }
                      j_j___libc_free_0(v2, 96);
                      if ( !v14 )
                        break;
                      v2 = v14;
                    }
                  }
                  v19 = v1[8];
                  v20 = (_QWORD *)v1[2];
                  while ( v19 )
                  {
                    sub_1C97470(*(_QWORD *)(v19 + 24));
                    v21 = v19;
                    v19 = *(_QWORD *)(v19 + 16);
                    j_j___libc_free_0(v21, 40);
                  }
                  j_j___libc_free_0(v1, 96);
                  if ( !v20 )
                    break;
                  v1 = v20;
                }
              }
              v24 = v44[8];
              v25 = (_QWORD *)v44[2];
              while ( v24 )
              {
                sub_1C97470(*(_QWORD *)(v24 + 24));
                v26 = v24;
                v24 = *(_QWORD *)(v24 + 16);
                j_j___libc_free_0(v26, 40);
              }
              j_j___libc_free_0(v44, 96);
              if ( !v25 )
                break;
              v44 = v25;
            }
          }
          v27 = v45[8];
          v28 = (_QWORD *)v45[2];
          while ( v27 )
          {
            sub_1C97470(*(_QWORD *)(v27 + 24));
            v29 = v27;
            v27 = *(_QWORD *)(v27 + 16);
            j_j___libc_free_0(v29, 40);
          }
          j_j___libc_free_0(v45, 96);
          if ( !v28 )
            break;
          v45 = v28;
        }
      }
      v30 = v43[8];
      v31 = (_QWORD *)v43[2];
      while ( v30 )
      {
        sub_1C97470(*(_QWORD *)(v30 + 24));
        v32 = v30;
        v30 = *(_QWORD *)(v30 + 16);
        j_j___libc_free_0(v32, 40);
      }
      j_j___libc_free_0(v43, 96);
      if ( !v31 )
        break;
      v43 = v31;
    }
  }
}
