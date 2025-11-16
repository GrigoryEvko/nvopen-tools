// Function: sub_105DD10
// Address: 0x105dd10
//
__int64 __fastcall sub_105DD10(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r12
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  _QWORD *v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdi
  __int64 v15; // rdi
  _QWORD *v16; // rbx
  __int64 v17; // rdi
  _QWORD *v18; // rbx
  __int64 v19; // rdi
  _QWORD *v20; // rbx
  __int64 v21; // rdi
  _QWORD *v22; // rbx
  __int64 v23; // rdi
  _QWORD *v24; // rbx
  __int64 v25; // rdi
  _QWORD *v26; // rbx
  __int64 v27; // rdi
  _QWORD *v28; // rbx
  __int64 v29; // rdi
  _QWORD *v30; // rbx
  __int64 result; // rax
  __int64 v32; // [rsp+8h] [rbp-58h]
  _QWORD *v33; // [rsp+10h] [rbp-50h]
  _QWORD *v34; // [rsp+18h] [rbp-48h]
  _QWORD *v35; // [rsp+20h] [rbp-40h]
  _QWORD *v36; // [rsp+28h] [rbp-38h]

  v33 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v36 = (_QWORD *)v33[3];
      if ( v36 )
      {
        while ( 1 )
        {
          v35 = (_QWORD *)v36[3];
          if ( v35 )
          {
            while ( 1 )
            {
              v34 = (_QWORD *)v35[3];
              if ( v34 )
              {
                while ( 1 )
                {
                  v5 = (_QWORD *)v34[3];
                  if ( v5 )
                  {
                    while ( 1 )
                    {
                      v6 = (_QWORD *)v5[3];
                      if ( v6 )
                      {
                        while ( 1 )
                        {
                          v7 = (_QWORD *)v6[3];
                          if ( v7 )
                          {
                            while ( 1 )
                            {
                              v8 = (_QWORD *)v7[3];
                              if ( v8 )
                              {
                                while ( 1 )
                                {
                                  v9 = v8[3];
                                  while ( v9 )
                                  {
                                    sub_105DD10(*(_QWORD *)(v9 + 24));
                                    v13 = v9;
                                    v9 = *(_QWORD *)(v9 + 16);
                                    v14 = *(_QWORD *)(v13 + 40);
                                    if ( v14 )
                                    {
                                      v32 = v13;
                                      sub_BA65D0(v14, a2, v10, v11, v12);
                                      v13 = v32;
                                    }
                                    a2 = 56;
                                    j_j___libc_free_0(v13, 56);
                                  }
                                  v15 = v8[5];
                                  v16 = (_QWORD *)v8[2];
                                  if ( v15 )
                                    sub_BA65D0(v15, a2, a3, a4, a5);
                                  a2 = 56;
                                  j_j___libc_free_0(v8, 56);
                                  if ( !v16 )
                                    break;
                                  v8 = v16;
                                }
                              }
                              v23 = v7[5];
                              v24 = (_QWORD *)v7[2];
                              if ( v23 )
                                sub_BA65D0(v23, a2, a3, a4, a5);
                              a2 = 56;
                              j_j___libc_free_0(v7, 56);
                              if ( !v24 )
                                break;
                              v7 = v24;
                            }
                          }
                          v19 = v6[5];
                          v20 = (_QWORD *)v6[2];
                          if ( v19 )
                            sub_BA65D0(v19, a2, a3, a4, a5);
                          a2 = 56;
                          j_j___libc_free_0(v6, 56);
                          if ( !v20 )
                            break;
                          v6 = v20;
                        }
                      }
                      v17 = v5[5];
                      v18 = (_QWORD *)v5[2];
                      if ( v17 )
                        sub_BA65D0(v17, a2, a3, a4, a5);
                      a2 = 56;
                      j_j___libc_free_0(v5, 56);
                      if ( !v18 )
                        break;
                      v5 = v18;
                    }
                  }
                  v21 = v34[5];
                  v22 = (_QWORD *)v34[2];
                  if ( v21 )
                    sub_BA65D0(v21, a2, a3, a4, a5);
                  a2 = 56;
                  j_j___libc_free_0(v34, 56);
                  if ( !v22 )
                    break;
                  v34 = v22;
                }
              }
              v25 = v35[5];
              v26 = (_QWORD *)v35[2];
              if ( v25 )
                sub_BA65D0(v25, a2, a3, a4, a5);
              a2 = 56;
              j_j___libc_free_0(v35, 56);
              if ( !v26 )
                break;
              v35 = v26;
            }
          }
          v27 = v36[5];
          v28 = (_QWORD *)v36[2];
          if ( v27 )
            sub_BA65D0(v27, a2, a3, a4, a5);
          a2 = 56;
          j_j___libc_free_0(v36, 56);
          if ( !v28 )
            break;
          v36 = v28;
        }
      }
      v29 = v33[5];
      v30 = (_QWORD *)v33[2];
      if ( v29 )
        sub_BA65D0(v29, a2, a3, a4, a5);
      a2 = 56;
      result = j_j___libc_free_0(v33, 56);
      if ( !v30 )
        break;
      v33 = v30;
    }
  }
  return result;
}
