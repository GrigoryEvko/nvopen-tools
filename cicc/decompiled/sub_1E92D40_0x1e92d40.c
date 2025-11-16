// Function: sub_1E92D40
// Address: 0x1e92d40
//
__int64 __fastcall sub_1E92D40(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r9
  __int64 v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // rbx
  __int64 v10; // rdi
  _QWORD *v11; // rbx
  __int64 v12; // rdi
  _QWORD *v13; // rbx
  __int64 v14; // rdi
  _QWORD *v15; // rbx
  __int64 v16; // rdi
  _QWORD *v17; // rbx
  __int64 v18; // rdi
  _QWORD *v19; // rbx
  __int64 v20; // rdi
  _QWORD *v21; // rbx
  __int64 v22; // rdi
  _QWORD *v23; // rbx
  __int64 result; // rax
  __int64 v25; // [rsp+8h] [rbp-58h]
  _QWORD *v26; // [rsp+10h] [rbp-50h]
  _QWORD *v27; // [rsp+18h] [rbp-48h]
  _QWORD *v28; // [rsp+20h] [rbp-40h]
  _QWORD *v29; // [rsp+28h] [rbp-38h]

  v26 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v29 = (_QWORD *)v26[3];
      if ( v29 )
      {
        while ( 1 )
        {
          v28 = (_QWORD *)v29[3];
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
                                  v5 = v4[3];
                                  while ( v5 )
                                  {
                                    sub_1E92D40(*(_QWORD *)(v5 + 24));
                                    v6 = v5;
                                    v5 = *(_QWORD *)(v5 + 16);
                                    v7 = *(_QWORD *)(v6 + 40);
                                    if ( v7 )
                                    {
                                      v25 = v6;
                                      j_j___libc_free_0(v7, *(_QWORD *)(v6 + 56) - v7);
                                      v6 = v25;
                                    }
                                    j_j___libc_free_0(v6, 64);
                                  }
                                  v8 = v4[5];
                                  v9 = (_QWORD *)v4[2];
                                  if ( v8 )
                                    j_j___libc_free_0(v8, v4[7] - v8);
                                  j_j___libc_free_0(v4, 64);
                                  if ( !v9 )
                                    break;
                                  v4 = v9;
                                }
                              }
                              v16 = v3[5];
                              v17 = (_QWORD *)v3[2];
                              if ( v16 )
                                j_j___libc_free_0(v16, v3[7] - v16);
                              j_j___libc_free_0(v3, 64);
                              if ( !v17 )
                                break;
                              v3 = v17;
                            }
                          }
                          v12 = v2[5];
                          v13 = (_QWORD *)v2[2];
                          if ( v12 )
                            j_j___libc_free_0(v12, v2[7] - v12);
                          j_j___libc_free_0(v2, 64);
                          if ( !v13 )
                            break;
                          v2 = v13;
                        }
                      }
                      v10 = v1[5];
                      v11 = (_QWORD *)v1[2];
                      if ( v10 )
                        j_j___libc_free_0(v10, v1[7] - v10);
                      j_j___libc_free_0(v1, 64);
                      if ( !v11 )
                        break;
                      v1 = v11;
                    }
                  }
                  v14 = v27[5];
                  v15 = (_QWORD *)v27[2];
                  if ( v14 )
                    j_j___libc_free_0(v14, v27[7] - v14);
                  j_j___libc_free_0(v27, 64);
                  if ( !v15 )
                    break;
                  v27 = v15;
                }
              }
              v18 = v28[5];
              v19 = (_QWORD *)v28[2];
              if ( v18 )
                j_j___libc_free_0(v18, v28[7] - v18);
              j_j___libc_free_0(v28, 64);
              if ( !v19 )
                break;
              v28 = v19;
            }
          }
          v20 = v29[5];
          v21 = (_QWORD *)v29[2];
          if ( v20 )
            j_j___libc_free_0(v20, v29[7] - v20);
          j_j___libc_free_0(v29, 64);
          if ( !v21 )
            break;
          v29 = v21;
        }
      }
      v22 = v26[5];
      v23 = (_QWORD *)v26[2];
      if ( v22 )
        j_j___libc_free_0(v22, v26[7] - v22);
      result = j_j___libc_free_0(v26, 64);
      if ( !v23 )
        break;
      v26 = v23;
    }
  }
  return result;
}
