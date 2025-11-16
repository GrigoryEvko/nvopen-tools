// Function: sub_1B7DEF0
// Address: 0x1b7def0
//
void __fastcall sub_1B7DEF0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // rdi
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+28h] [rbp-38h]

  v25 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v28 = *(_QWORD *)(v25 + 24);
      if ( v28 )
      {
        while ( 1 )
        {
          v27 = *(_QWORD *)(v28 + 24);
          if ( v27 )
          {
            while ( 1 )
            {
              v26 = *(_QWORD *)(v27 + 24);
              if ( v26 )
              {
                while ( 1 )
                {
                  v1 = *(_QWORD *)(v26 + 24);
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
                                  while ( v5 )
                                  {
                                    sub_1B7DEF0(*(_QWORD *)(v5 + 24));
                                    v6 = v5;
                                    v24 = v5;
                                    v5 = *(_QWORD *)(v5 + 16);
                                    if ( *(_DWORD *)(v6 + 48) > 0x40u )
                                    {
                                      v7 = *(_QWORD *)(v6 + 40);
                                      if ( v7 )
                                        j_j___libc_free_0_0(v7);
                                    }
                                    j_j___libc_free_0(v24, 64);
                                  }
                                  v8 = *(_QWORD *)(v4 + 16);
                                  if ( *(_DWORD *)(v4 + 48) > 0x40u )
                                  {
                                    v9 = *(_QWORD *)(v4 + 40);
                                    if ( v9 )
                                      j_j___libc_free_0_0(v9);
                                  }
                                  j_j___libc_free_0(v4, 64);
                                  if ( !v8 )
                                    break;
                                  v4 = v8;
                                }
                              }
                              v16 = *(_QWORD *)(v3 + 16);
                              if ( *(_DWORD *)(v3 + 48) > 0x40u )
                              {
                                v17 = *(_QWORD *)(v3 + 40);
                                if ( v17 )
                                  j_j___libc_free_0_0(v17);
                              }
                              j_j___libc_free_0(v3, 64);
                              if ( !v16 )
                                break;
                              v3 = v16;
                            }
                          }
                          v12 = *(_QWORD *)(v2 + 16);
                          if ( *(_DWORD *)(v2 + 48) > 0x40u )
                          {
                            v13 = *(_QWORD *)(v2 + 40);
                            if ( v13 )
                              j_j___libc_free_0_0(v13);
                          }
                          j_j___libc_free_0(v2, 64);
                          if ( !v12 )
                            break;
                          v2 = v12;
                        }
                      }
                      v10 = *(_QWORD *)(v1 + 16);
                      if ( *(_DWORD *)(v1 + 48) > 0x40u )
                      {
                        v11 = *(_QWORD *)(v1 + 40);
                        if ( v11 )
                          j_j___libc_free_0_0(v11);
                      }
                      j_j___libc_free_0(v1, 64);
                      if ( !v10 )
                        break;
                      v1 = v10;
                    }
                  }
                  v14 = *(_QWORD *)(v26 + 16);
                  if ( *(_DWORD *)(v26 + 48) > 0x40u )
                  {
                    v15 = *(_QWORD *)(v26 + 40);
                    if ( v15 )
                      j_j___libc_free_0_0(v15);
                  }
                  j_j___libc_free_0(v26, 64);
                  if ( !v14 )
                    break;
                  v26 = v14;
                }
              }
              v18 = *(_QWORD *)(v27 + 16);
              if ( *(_DWORD *)(v27 + 48) > 0x40u )
              {
                v19 = *(_QWORD *)(v27 + 40);
                if ( v19 )
                  j_j___libc_free_0_0(v19);
              }
              j_j___libc_free_0(v27, 64);
              if ( !v18 )
                break;
              v27 = v18;
            }
          }
          v20 = *(_QWORD *)(v28 + 16);
          if ( *(_DWORD *)(v28 + 48) > 0x40u )
          {
            v21 = *(_QWORD *)(v28 + 40);
            if ( v21 )
              j_j___libc_free_0_0(v21);
          }
          j_j___libc_free_0(v28, 64);
          if ( !v20 )
            break;
          v28 = v20;
        }
      }
      v22 = *(_QWORD *)(v25 + 16);
      if ( *(_DWORD *)(v25 + 48) > 0x40u )
      {
        v23 = *(_QWORD *)(v25 + 40);
        if ( v23 )
          j_j___libc_free_0_0(v23);
      }
      j_j___libc_free_0(v25, 64);
      if ( !v22 )
        break;
      v25 = v22;
    }
  }
}
