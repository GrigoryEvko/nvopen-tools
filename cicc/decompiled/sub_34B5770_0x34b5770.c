// Function: sub_34B5770
// Address: 0x34b5770
//
void __fastcall sub_34B5770(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // r10
  unsigned __int64 i; // rbx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v17; // [rsp+8h] [rbp-58h]
  unsigned __int64 v19; // [rsp+18h] [rbp-48h]
  unsigned __int64 v20; // [rsp+20h] [rbp-40h]
  unsigned __int64 v21; // [rsp+28h] [rbp-38h]

  v19 = a2;
  if ( a2 )
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(v19 + 24);
      if ( v21 )
      {
        while ( 1 )
        {
          v2 = *(_QWORD *)(v21 + 24);
          if ( v2 )
          {
            while ( 1 )
            {
              v20 = *(_QWORD *)(v2 + 24);
              if ( v20 )
              {
                while ( 1 )
                {
                  v3 = *(_QWORD *)(v20 + 24);
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
                                  for ( i = *(_QWORD *)(v6 + 24); i; v6 = v16 )
                                  {
                                    v16 = v6;
                                    sub_34B5770(a1, *(_QWORD *)(i + 24));
                                    v8 = i;
                                    i = *(_QWORD *)(i + 16);
                                    j_j___libc_free_0(v8);
                                  }
                                  v9 = *(_QWORD *)(v6 + 16);
                                  j_j___libc_free_0(v6);
                                  if ( !v9 )
                                    break;
                                  v6 = v9;
                                }
                              }
                              v17 = *(_QWORD *)(v5 + 16);
                              j_j___libc_free_0(v5);
                              if ( !v17 )
                                break;
                              v5 = v17;
                            }
                          }
                          v11 = *(_QWORD *)(v4 + 16);
                          j_j___libc_free_0(v4);
                          if ( !v11 )
                            break;
                          v4 = v11;
                        }
                      }
                      v10 = *(_QWORD *)(v3 + 16);
                      j_j___libc_free_0(v3);
                      if ( !v10 )
                        break;
                      v3 = v10;
                    }
                  }
                  v12 = *(_QWORD *)(v20 + 16);
                  j_j___libc_free_0(v20);
                  if ( !v12 )
                    break;
                  v20 = v12;
                }
              }
              v13 = *(_QWORD *)(v2 + 16);
              j_j___libc_free_0(v2);
              if ( !v13 )
                break;
              v2 = v13;
            }
          }
          v14 = *(_QWORD *)(v21 + 16);
          j_j___libc_free_0(v21);
          if ( !v14 )
            break;
          v21 = v14;
        }
      }
      v15 = *(_QWORD *)(v19 + 16);
      j_j___libc_free_0(v19);
      if ( !v15 )
        break;
      v19 = v15;
    }
  }
}
