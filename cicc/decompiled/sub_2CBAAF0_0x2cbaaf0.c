// Function: sub_2CBAAF0
// Address: 0x2cbaaf0
//
void __fastcall sub_2CBAAF0(unsigned __int64 a1)
{
  unsigned __int64 v1; // r15
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // r10
  unsigned __int64 i; // rbx
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // [rsp+0h] [rbp-50h]
  unsigned __int64 v16; // [rsp+0h] [rbp-50h]
  unsigned __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18; // [rsp+10h] [rbp-40h]
  unsigned __int64 v19; // [rsp+18h] [rbp-38h]

  v17 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v17 + 24);
      if ( v19 )
      {
        while ( 1 )
        {
          v1 = *(_QWORD *)(v19 + 24);
          if ( v1 )
          {
            while ( 1 )
            {
              v18 = *(_QWORD *)(v1 + 24);
              if ( v18 )
              {
                while ( 1 )
                {
                  v2 = *(_QWORD *)(v18 + 24);
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
                                  for ( i = *(_QWORD *)(v5 + 24); i; v5 = v15 )
                                  {
                                    v15 = v5;
                                    sub_2CBAAF0(*(_QWORD *)(i + 24));
                                    v7 = i;
                                    i = *(_QWORD *)(i + 16);
                                    j_j___libc_free_0(v7);
                                  }
                                  v8 = *(_QWORD *)(v5 + 16);
                                  j_j___libc_free_0(v5);
                                  if ( !v8 )
                                    break;
                                  v5 = v8;
                                }
                              }
                              v16 = *(_QWORD *)(v4 + 16);
                              j_j___libc_free_0(v4);
                              if ( !v16 )
                                break;
                              v4 = v16;
                            }
                          }
                          v10 = *(_QWORD *)(v3 + 16);
                          j_j___libc_free_0(v3);
                          if ( !v10 )
                            break;
                          v3 = v10;
                        }
                      }
                      v9 = *(_QWORD *)(v2 + 16);
                      j_j___libc_free_0(v2);
                      if ( !v9 )
                        break;
                      v2 = v9;
                    }
                  }
                  v11 = *(_QWORD *)(v18 + 16);
                  j_j___libc_free_0(v18);
                  if ( !v11 )
                    break;
                  v18 = v11;
                }
              }
              v12 = *(_QWORD *)(v1 + 16);
              j_j___libc_free_0(v1);
              if ( !v12 )
                break;
              v1 = v12;
            }
          }
          v13 = *(_QWORD *)(v19 + 16);
          j_j___libc_free_0(v19);
          if ( !v13 )
            break;
          v19 = v13;
        }
      }
      v14 = *(_QWORD *)(v17 + 16);
      j_j___libc_free_0(v17);
      if ( !v14 )
        break;
      v17 = v14;
    }
  }
}
