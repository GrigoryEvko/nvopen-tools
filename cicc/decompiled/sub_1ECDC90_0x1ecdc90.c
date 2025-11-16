// Function: sub_1ECDC90
// Address: 0x1ecdc90
//
__int64 __fastcall sub_1ECDC90(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // r10
  __int64 i; // rbx
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 v15; // rbx
  __int64 result; // rax
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+28h] [rbp-38h]

  v20 = a2;
  if ( a2 )
  {
    while ( 1 )
    {
      v22 = *(_QWORD *)(v20 + 24);
      if ( v22 )
      {
        while ( 1 )
        {
          v2 = *(_QWORD *)(v22 + 24);
          if ( v2 )
          {
            while ( 1 )
            {
              v21 = *(_QWORD *)(v2 + 24);
              if ( v21 )
              {
                while ( 1 )
                {
                  v3 = *(_QWORD *)(v21 + 24);
                  if ( v3 )
                  {
                    while ( 1 )
                    {
                      v4 = *(_QWORD *)(v3 + 24);
                      if ( v4 )
                      {
                        while ( 1 )
                        {
                          v5 = *(_QWORD *)(v4 + 24);
                          if ( v5 )
                          {
                            while ( 1 )
                            {
                              v6 = *(_QWORD *)(v5 + 24);
                              if ( v6 )
                              {
                                while ( 1 )
                                {
                                  for ( i = *(_QWORD *)(v6 + 24); i; v6 = v17 )
                                  {
                                    v17 = v6;
                                    sub_1ECDC90(a1, *(_QWORD *)(i + 24));
                                    v8 = i;
                                    i = *(_QWORD *)(i + 16);
                                    j_j___libc_free_0(v8, 56);
                                  }
                                  v9 = *(_QWORD *)(v6 + 16);
                                  j_j___libc_free_0(v6, 56);
                                  if ( !v9 )
                                    break;
                                  v6 = v9;
                                }
                              }
                              v18 = *(_QWORD *)(v5 + 16);
                              j_j___libc_free_0(v5, 56);
                              if ( !v18 )
                                break;
                              v5 = v18;
                            }
                          }
                          v11 = *(_QWORD *)(v4 + 16);
                          j_j___libc_free_0(v4, 56);
                          if ( !v11 )
                            break;
                          v4 = v11;
                        }
                      }
                      v10 = *(_QWORD *)(v3 + 16);
                      j_j___libc_free_0(v3, 56);
                      if ( !v10 )
                        break;
                      v3 = v10;
                    }
                  }
                  v12 = *(_QWORD *)(v21 + 16);
                  j_j___libc_free_0(v21, 56);
                  if ( !v12 )
                    break;
                  v21 = v12;
                }
              }
              v13 = *(_QWORD *)(v2 + 16);
              j_j___libc_free_0(v2, 56);
              if ( !v13 )
                break;
              v2 = v13;
            }
          }
          v14 = *(_QWORD *)(v22 + 16);
          j_j___libc_free_0(v22, 56);
          if ( !v14 )
            break;
          v22 = v14;
        }
      }
      v15 = *(_QWORD *)(v20 + 16);
      result = j_j___libc_free_0(v20, 56);
      if ( !v15 )
        break;
      v20 = v15;
    }
  }
  return result;
}
