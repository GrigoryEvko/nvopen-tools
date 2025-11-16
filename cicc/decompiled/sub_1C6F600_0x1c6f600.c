// Function: sub_1C6F600
// Address: 0x1c6f600
//
__int64 __fastcall sub_1C6F600(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r10
  __int64 i; // rbx
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 result; // rax
  __int64 v16; // [rsp+0h] [rbp-50h]
  __int64 v17; // [rsp+0h] [rbp-50h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v18 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(v18 + 24);
      if ( v20 )
      {
        while ( 1 )
        {
          v1 = *(_QWORD *)(v20 + 24);
          if ( v1 )
          {
            while ( 1 )
            {
              v19 = *(_QWORD *)(v1 + 24);
              if ( v19 )
              {
                while ( 1 )
                {
                  v2 = *(_QWORD *)(v19 + 24);
                  if ( v2 )
                  {
                    while ( 1 )
                    {
                      v3 = *(_QWORD *)(v2 + 24);
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
                                  for ( i = *(_QWORD *)(v5 + 24); i; v5 = v16 )
                                  {
                                    v16 = v5;
                                    sub_1C6F600(*(_QWORD *)(i + 24));
                                    v7 = i;
                                    i = *(_QWORD *)(i + 16);
                                    j_j___libc_free_0(v7, 40);
                                  }
                                  v8 = *(_QWORD *)(v5 + 16);
                                  j_j___libc_free_0(v5, 40);
                                  if ( !v8 )
                                    break;
                                  v5 = v8;
                                }
                              }
                              v17 = *(_QWORD *)(v4 + 16);
                              j_j___libc_free_0(v4, 40);
                              if ( !v17 )
                                break;
                              v4 = v17;
                            }
                          }
                          v10 = *(_QWORD *)(v3 + 16);
                          j_j___libc_free_0(v3, 40);
                          if ( !v10 )
                            break;
                          v3 = v10;
                        }
                      }
                      v9 = *(_QWORD *)(v2 + 16);
                      j_j___libc_free_0(v2, 40);
                      if ( !v9 )
                        break;
                      v2 = v9;
                    }
                  }
                  v11 = *(_QWORD *)(v19 + 16);
                  j_j___libc_free_0(v19, 40);
                  if ( !v11 )
                    break;
                  v19 = v11;
                }
              }
              v12 = *(_QWORD *)(v1 + 16);
              j_j___libc_free_0(v1, 40);
              if ( !v12 )
                break;
              v1 = v12;
            }
          }
          v13 = *(_QWORD *)(v20 + 16);
          j_j___libc_free_0(v20, 40);
          if ( !v13 )
            break;
          v20 = v13;
        }
      }
      v14 = *(_QWORD *)(v18 + 16);
      result = j_j___libc_free_0(v18, 40);
      if ( !v14 )
        break;
      v18 = v14;
    }
  }
  return result;
}
