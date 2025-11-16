// Function: sub_2C91AF0
// Address: 0x2c91af0
//
void __fastcall sub_2C91AF0(unsigned __int64 a1)
{
  unsigned __int64 v1; // r15
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // [rsp+8h] [rbp-58h]
  unsigned __int64 v12; // [rsp+10h] [rbp-50h]
  unsigned __int64 v13; // [rsp+18h] [rbp-48h]
  unsigned __int64 v14; // [rsp+20h] [rbp-40h]
  unsigned __int64 v15; // [rsp+28h] [rbp-38h]
  unsigned __int64 v16; // [rsp+28h] [rbp-38h]
  unsigned __int64 v17; // [rsp+28h] [rbp-38h]
  unsigned __int64 v18; // [rsp+28h] [rbp-38h]
  unsigned __int64 v19; // [rsp+28h] [rbp-38h]

  v12 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v12 + 24);
      if ( v14 )
      {
        while ( 1 )
        {
          v1 = *(_QWORD *)(v14 + 24);
          if ( v1 )
          {
            while ( 1 )
            {
              v13 = *(_QWORD *)(v1 + 24);
              if ( v13 )
              {
                while ( 1 )
                {
                  v2 = *(_QWORD *)(v13 + 24);
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
                                      v15 = v6;
                                      sub_2C91AF0(*(_QWORD *)(v6 + 24));
                                      v11 = *(_QWORD *)(v15 + 16);
                                      sub_C7D6A0(*(_QWORD *)(v15 + 48), 16LL * *(unsigned int *)(v15 + 64), 8);
                                      j_j___libc_free_0(v15);
                                      v6 = v11;
                                    }
                                    while ( v11 );
                                  }
                                  v16 = *(_QWORD *)(v5 + 16);
                                  sub_C7D6A0(*(_QWORD *)(v5 + 48), 16LL * *(unsigned int *)(v5 + 64), 8);
                                  j_j___libc_free_0(v5);
                                  if ( !v16 )
                                    break;
                                  v5 = v16;
                                }
                              }
                              v19 = *(_QWORD *)(v4 + 16);
                              sub_C7D6A0(*(_QWORD *)(v4 + 48), 16LL * *(unsigned int *)(v4 + 64), 8);
                              j_j___libc_free_0(v4);
                              if ( !v19 )
                                break;
                              v4 = v19;
                            }
                          }
                          v18 = *(_QWORD *)(v3 + 16);
                          sub_C7D6A0(*(_QWORD *)(v3 + 48), 16LL * *(unsigned int *)(v3 + 64), 8);
                          j_j___libc_free_0(v3);
                          if ( !v18 )
                            break;
                          v3 = v18;
                        }
                      }
                      v17 = *(_QWORD *)(v2 + 16);
                      sub_C7D6A0(*(_QWORD *)(v2 + 48), 16LL * *(unsigned int *)(v2 + 64), 8);
                      j_j___libc_free_0(v2);
                      if ( !v17 )
                        break;
                      v2 = v17;
                    }
                  }
                  v7 = *(_QWORD *)(v13 + 16);
                  sub_C7D6A0(*(_QWORD *)(v13 + 48), 16LL * *(unsigned int *)(v13 + 64), 8);
                  j_j___libc_free_0(v13);
                  if ( !v7 )
                    break;
                  v13 = v7;
                }
              }
              v8 = *(_QWORD *)(v1 + 16);
              sub_C7D6A0(*(_QWORD *)(v1 + 48), 16LL * *(unsigned int *)(v1 + 64), 8);
              j_j___libc_free_0(v1);
              if ( !v8 )
                break;
              v1 = v8;
            }
          }
          v9 = *(_QWORD *)(v14 + 16);
          sub_C7D6A0(*(_QWORD *)(v14 + 48), 16LL * *(unsigned int *)(v14 + 64), 8);
          j_j___libc_free_0(v14);
          if ( !v9 )
            break;
          v14 = v9;
        }
      }
      v10 = *(_QWORD *)(v12 + 16);
      sub_C7D6A0(*(_QWORD *)(v12 + 48), 16LL * *(unsigned int *)(v12 + 64), 8);
      j_j___libc_free_0(v12);
      if ( !v10 )
        break;
      v12 = v10;
    }
  }
}
