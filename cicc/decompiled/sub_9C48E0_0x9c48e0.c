// Function: sub_9C48E0
// Address: 0x9c48e0
//
void __fastcall sub_9C48E0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // rbx
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+28h] [rbp-38h]
  __int64 v19; // [rsp+28h] [rbp-38h]

  v14 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v14 + 24);
      if ( v16 )
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(v16 + 24);
          if ( v15 )
          {
            while ( 1 )
            {
              v1 = *(_QWORD *)(v15 + 24);
              if ( v1 )
              {
                while ( 1 )
                {
                  v2 = *(_QWORD *)(v1 + 24);
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
                                  v6 = *(_QWORD *)(v5 + 24);
                                  if ( v6 )
                                  {
                                    do
                                    {
                                      v17 = v6;
                                      sub_9C48E0(*(_QWORD *)(v6 + 24));
                                      v13 = *(_QWORD *)(v17 + 16);
                                      sub_9C4830(*(_QWORD **)(v17 + 112));
                                      j_j___libc_free_0(v17, 144);
                                      v6 = v13;
                                    }
                                    while ( v13 );
                                  }
                                  v18 = *(_QWORD *)(v5 + 16);
                                  sub_9C4830(*(_QWORD **)(v5 + 112));
                                  j_j___libc_free_0(v5, 144);
                                  if ( !v18 )
                                    break;
                                  v5 = v18;
                                }
                              }
                              v19 = *(_QWORD *)(v4 + 16);
                              sub_9C4830(*(_QWORD **)(v4 + 112));
                              j_j___libc_free_0(v4, 144);
                              if ( !v19 )
                                break;
                              v4 = v19;
                            }
                          }
                          v8 = *(_QWORD *)(v3 + 16);
                          sub_9C4830(*(_QWORD **)(v3 + 112));
                          j_j___libc_free_0(v3, 144);
                          if ( !v8 )
                            break;
                          v3 = v8;
                        }
                      }
                      v7 = *(_QWORD *)(v2 + 16);
                      sub_9C4830(*(_QWORD **)(v2 + 112));
                      j_j___libc_free_0(v2, 144);
                      if ( !v7 )
                        break;
                      v2 = v7;
                    }
                  }
                  v9 = *(_QWORD *)(v1 + 16);
                  sub_9C4830(*(_QWORD **)(v1 + 112));
                  j_j___libc_free_0(v1, 144);
                  if ( !v9 )
                    break;
                  v1 = v9;
                }
              }
              v10 = *(_QWORD *)(v15 + 16);
              sub_9C4830(*(_QWORD **)(v15 + 112));
              j_j___libc_free_0(v15, 144);
              if ( !v10 )
                break;
              v15 = v10;
            }
          }
          v11 = *(_QWORD *)(v16 + 16);
          sub_9C4830(*(_QWORD **)(v16 + 112));
          j_j___libc_free_0(v16, 144);
          if ( !v11 )
            break;
          v16 = v11;
        }
      }
      v12 = *(_QWORD *)(v14 + 16);
      sub_9C4830(*(_QWORD **)(v14 + 112));
      j_j___libc_free_0(v14, 144);
      if ( !v12 )
        break;
      v14 = v12;
    }
  }
}
