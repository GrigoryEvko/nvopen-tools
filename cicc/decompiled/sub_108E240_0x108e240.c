// Function: sub_108E240
// Address: 0x108e240
//
__int64 __fastcall sub_108E240(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // rdi
  _QWORD *v10; // rbx
  __int64 v11; // rdi
  _QWORD *v12; // rbx
  __int64 v13; // rdi
  _QWORD *v14; // rbx
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
  __int64 result; // rax
  __int64 v26; // [rsp+0h] [rbp-60h]
  _QWORD *v28; // [rsp+10h] [rbp-50h]
  _QWORD *v29; // [rsp+18h] [rbp-48h]
  _QWORD *v30; // [rsp+20h] [rbp-40h]
  _QWORD *v31; // [rsp+28h] [rbp-38h]

  v28 = a2;
  if ( a2 )
  {
    while ( 1 )
    {
      v31 = (_QWORD *)v28[3];
      if ( v31 )
      {
        while ( 1 )
        {
          v30 = (_QWORD *)v31[3];
          if ( v30 )
          {
            while ( 1 )
            {
              v29 = (_QWORD *)v30[3];
              if ( v29 )
              {
                while ( 1 )
                {
                  v2 = (_QWORD *)v29[3];
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
                                    sub_108E240(a1, *(_QWORD *)(v6 + 24));
                                    v7 = v6;
                                    v6 = *(_QWORD *)(v6 + 16);
                                    v8 = *(_QWORD *)(v7 + 64);
                                    if ( v8 )
                                    {
                                      v26 = v7;
                                      j_j___libc_free_0(v8, *(_QWORD *)(v7 + 80) - v8);
                                      v7 = v26;
                                    }
                                    j_j___libc_free_0(v7, 88);
                                  }
                                  v9 = v5[8];
                                  v10 = (_QWORD *)v5[2];
                                  if ( v9 )
                                    j_j___libc_free_0(v9, v5[10] - v9);
                                  j_j___libc_free_0(v5, 88);
                                  if ( !v10 )
                                    break;
                                  v5 = v10;
                                }
                              }
                              v17 = v4[8];
                              v18 = (_QWORD *)v4[2];
                              if ( v17 )
                                j_j___libc_free_0(v17, v4[10] - v17);
                              j_j___libc_free_0(v4, 88);
                              if ( !v18 )
                                break;
                              v4 = v18;
                            }
                          }
                          v13 = v3[8];
                          v14 = (_QWORD *)v3[2];
                          if ( v13 )
                            j_j___libc_free_0(v13, v3[10] - v13);
                          j_j___libc_free_0(v3, 88);
                          if ( !v14 )
                            break;
                          v3 = v14;
                        }
                      }
                      v11 = v2[8];
                      v12 = (_QWORD *)v2[2];
                      if ( v11 )
                        j_j___libc_free_0(v11, v2[10] - v11);
                      j_j___libc_free_0(v2, 88);
                      if ( !v12 )
                        break;
                      v2 = v12;
                    }
                  }
                  v15 = v29[8];
                  v16 = (_QWORD *)v29[2];
                  if ( v15 )
                    j_j___libc_free_0(v15, v29[10] - v15);
                  j_j___libc_free_0(v29, 88);
                  if ( !v16 )
                    break;
                  v29 = v16;
                }
              }
              v19 = v30[8];
              v20 = (_QWORD *)v30[2];
              if ( v19 )
                j_j___libc_free_0(v19, v30[10] - v19);
              j_j___libc_free_0(v30, 88);
              if ( !v20 )
                break;
              v30 = v20;
            }
          }
          v21 = v31[8];
          v22 = (_QWORD *)v31[2];
          if ( v21 )
            j_j___libc_free_0(v21, v31[10] - v21);
          j_j___libc_free_0(v31, 88);
          if ( !v22 )
            break;
          v31 = v22;
        }
      }
      v23 = v28[8];
      v24 = (_QWORD *)v28[2];
      if ( v23 )
        j_j___libc_free_0(v23, v28[10] - v23);
      result = j_j___libc_free_0(v28, 88);
      if ( !v24 )
        break;
      v28 = v24;
    }
  }
  return result;
}
