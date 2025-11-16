// Function: sub_3121850
// Address: 0x3121850
//
void __fastcall sub_3121850(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  _QWORD *v6; // rax
  unsigned __int64 v7; // r9
  _QWORD *v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // r14
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // r12
  unsigned __int64 v21; // rdi
  _QWORD *v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rbx
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  _QWORD *v29; // rbx
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // rbx
  unsigned __int64 v33; // rdi
  _QWORD *v34; // [rsp+8h] [rbp-58h]
  _QWORD *v35; // [rsp+8h] [rbp-58h]
  _QWORD *v36; // [rsp+10h] [rbp-50h]
  _QWORD *v37; // [rsp+18h] [rbp-48h]
  _QWORD *v38; // [rsp+20h] [rbp-40h]
  _QWORD *v39; // [rsp+28h] [rbp-38h]
  unsigned __int64 v40; // [rsp+28h] [rbp-38h]
  _QWORD *v41; // [rsp+28h] [rbp-38h]
  _QWORD *v42; // [rsp+28h] [rbp-38h]

  v36 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v38 = (_QWORD *)v36[3];
      if ( v38 )
      {
        while ( 1 )
        {
          v37 = (_QWORD *)v38[3];
          if ( v37 )
          {
            while ( 1 )
            {
              v1 = (_QWORD *)v37[3];
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
                                      v39 = v6;
                                      sub_3121850(v6[3]);
                                      v7 = (unsigned __int64)v39;
                                      v8 = (_QWORD *)v39[2];
                                      v9 = v39[12];
                                      if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
                                      {
                                        v34 = (_QWORD *)v39[2];
                                        sub_BD60C0(v39 + 10);
                                        v8 = v34;
                                        v7 = (unsigned __int64)v39;
                                      }
                                      v10 = *(_QWORD *)(v7 + 32);
                                      if ( v10 != v7 + 48 )
                                      {
                                        v35 = v8;
                                        v40 = v7;
                                        j_j___libc_free_0(v10);
                                        v8 = v35;
                                        v7 = v40;
                                      }
                                      v41 = v8;
                                      j_j___libc_free_0(v7);
                                      v6 = v41;
                                    }
                                    while ( v41 );
                                  }
                                  v42 = (_QWORD *)v5[2];
                                  v11 = v5[12];
                                  if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
                                    sub_BD60C0(v5 + 10);
                                  v12 = v5[4];
                                  if ( (_QWORD *)v12 != v5 + 6 )
                                    j_j___libc_free_0(v12);
                                  j_j___libc_free_0((unsigned __int64)v5);
                                  if ( !v42 )
                                    break;
                                  v5 = v42;
                                }
                              }
                              v22 = (_QWORD *)v4[2];
                              v23 = v4[12];
                              if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
                                sub_BD60C0(v4 + 10);
                              v24 = v4[4];
                              if ( (_QWORD *)v24 != v4 + 6 )
                                j_j___libc_free_0(v24);
                              j_j___libc_free_0((unsigned __int64)v4);
                              if ( !v22 )
                                break;
                              v4 = v22;
                            }
                          }
                          v16 = v3[12];
                          v17 = (_QWORD *)v3[2];
                          if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
                            sub_BD60C0(v3 + 10);
                          v18 = v3[4];
                          if ( (_QWORD *)v18 != v3 + 6 )
                            j_j___libc_free_0(v18);
                          j_j___libc_free_0((unsigned __int64)v3);
                          if ( !v17 )
                            break;
                          v3 = v17;
                        }
                      }
                      v13 = v2[12];
                      v14 = (_QWORD *)v2[2];
                      if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
                        sub_BD60C0(v2 + 10);
                      v15 = v2[4];
                      if ( (_QWORD *)v15 != v2 + 6 )
                        j_j___libc_free_0(v15);
                      j_j___libc_free_0((unsigned __int64)v2);
                      if ( !v14 )
                        break;
                      v2 = v14;
                    }
                  }
                  v19 = v1[12];
                  v20 = (_QWORD *)v1[2];
                  if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
                    sub_BD60C0(v1 + 10);
                  v21 = v1[4];
                  if ( (_QWORD *)v21 != v1 + 6 )
                    j_j___libc_free_0(v21);
                  j_j___libc_free_0((unsigned __int64)v1);
                  if ( !v20 )
                    break;
                  v1 = v20;
                }
              }
              v25 = v37[12];
              v26 = (_QWORD *)v37[2];
              if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
                sub_BD60C0(v37 + 10);
              v27 = v37[4];
              if ( (_QWORD *)v27 != v37 + 6 )
                j_j___libc_free_0(v27);
              j_j___libc_free_0((unsigned __int64)v37);
              if ( !v26 )
                break;
              v37 = v26;
            }
          }
          v28 = v38[12];
          v29 = (_QWORD *)v38[2];
          if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
            sub_BD60C0(v38 + 10);
          v30 = v38[4];
          if ( (_QWORD *)v30 != v38 + 6 )
            j_j___libc_free_0(v30);
          j_j___libc_free_0((unsigned __int64)v38);
          if ( !v29 )
            break;
          v38 = v29;
        }
      }
      v31 = v36[12];
      v32 = (_QWORD *)v36[2];
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v36 + 10);
      v33 = v36[4];
      if ( (_QWORD *)v33 != v36 + 6 )
        j_j___libc_free_0(v33);
      j_j___libc_free_0((unsigned __int64)v36);
      if ( !v32 )
        break;
      v36 = v32;
    }
  }
}
