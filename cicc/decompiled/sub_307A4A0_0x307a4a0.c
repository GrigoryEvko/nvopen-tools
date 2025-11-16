// Function: sub_307A4A0
// Address: 0x307a4a0
//
void __fastcall sub_307A4A0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rbx
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // [rsp+8h] [rbp-58h]
  unsigned __int64 v14; // [rsp+10h] [rbp-50h]
  unsigned __int64 v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 v16; // [rsp+20h] [rbp-40h]
  unsigned __int64 v17; // [rsp+28h] [rbp-38h]
  unsigned __int64 v18; // [rsp+28h] [rbp-38h]
  unsigned __int64 v19; // [rsp+28h] [rbp-38h]

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
                                      sub_307A4A0(*(_QWORD *)(v6 + 24));
                                      v13 = *(_QWORD *)(v17 + 16);
                                      sub_307A420(*(_QWORD **)(v17 + 56));
                                      j_j___libc_free_0(v17);
                                      v6 = v13;
                                    }
                                    while ( v13 );
                                  }
                                  v18 = *(_QWORD *)(v5 + 16);
                                  sub_307A420(*(_QWORD **)(v5 + 56));
                                  j_j___libc_free_0(v5);
                                  if ( !v18 )
                                    break;
                                  v5 = v18;
                                }
                              }
                              v19 = *(_QWORD *)(v4 + 16);
                              sub_307A420(*(_QWORD **)(v4 + 56));
                              j_j___libc_free_0(v4);
                              if ( !v19 )
                                break;
                              v4 = v19;
                            }
                          }
                          v8 = *(_QWORD *)(v3 + 16);
                          sub_307A420(*(_QWORD **)(v3 + 56));
                          j_j___libc_free_0(v3);
                          if ( !v8 )
                            break;
                          v3 = v8;
                        }
                      }
                      v7 = *(_QWORD *)(v2 + 16);
                      sub_307A420(*(_QWORD **)(v2 + 56));
                      j_j___libc_free_0(v2);
                      if ( !v7 )
                        break;
                      v2 = v7;
                    }
                  }
                  v9 = *(_QWORD *)(v1 + 16);
                  sub_307A420(*(_QWORD **)(v1 + 56));
                  j_j___libc_free_0(v1);
                  if ( !v9 )
                    break;
                  v1 = v9;
                }
              }
              v10 = *(_QWORD *)(v15 + 16);
              sub_307A420(*(_QWORD **)(v15 + 56));
              j_j___libc_free_0(v15);
              if ( !v10 )
                break;
              v15 = v10;
            }
          }
          v11 = *(_QWORD *)(v16 + 16);
          sub_307A420(*(_QWORD **)(v16 + 56));
          j_j___libc_free_0(v16);
          if ( !v11 )
            break;
          v16 = v11;
        }
      }
      v12 = *(_QWORD *)(v14 + 16);
      sub_307A420(*(_QWORD **)(v14 + 56));
      j_j___libc_free_0(v14);
      if ( !v12 )
        break;
      v14 = v12;
    }
  }
}
