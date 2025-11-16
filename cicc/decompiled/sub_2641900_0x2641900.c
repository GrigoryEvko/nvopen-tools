// Function: sub_2641900
// Address: 0x2641900
//
void __fastcall sub_2641900(unsigned __int64 a1)
{
  unsigned __int64 v1; // r15
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r8
  unsigned __int64 i; // rbx
  unsigned __int64 v7; // r9
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // [rsp+8h] [rbp-58h]
  unsigned __int64 v18; // [rsp+10h] [rbp-50h]
  unsigned __int64 v19; // [rsp+10h] [rbp-50h]
  unsigned __int64 v20; // [rsp+10h] [rbp-50h]
  unsigned __int64 v21; // [rsp+10h] [rbp-50h]
  unsigned __int64 v22; // [rsp+18h] [rbp-48h]
  unsigned __int64 v23; // [rsp+20h] [rbp-40h]
  unsigned __int64 v24; // [rsp+28h] [rbp-38h]

  v22 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(v22 + 24);
      if ( v24 )
      {
        while ( 1 )
        {
          v23 = *(_QWORD *)(v24 + 24);
          if ( v23 )
          {
            while ( 1 )
            {
              v1 = *(_QWORD *)(v23 + 24);
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
                                  for ( i = *(_QWORD *)(v5 + 24); i; v5 = v19 )
                                  {
                                    v18 = v5;
                                    sub_2641900(*(_QWORD *)(i + 24));
                                    v7 = i;
                                    v8 = v18;
                                    i = *(_QWORD *)(i + 16);
                                    if ( !*(_BYTE *)(v7 + 68) )
                                    {
                                      v17 = v18;
                                      v20 = v7;
                                      _libc_free(*(_QWORD *)(v7 + 48));
                                      v8 = v17;
                                      v7 = v20;
                                    }
                                    v19 = v8;
                                    j_j___libc_free_0(v7);
                                  }
                                  v9 = *(_QWORD *)(v5 + 16);
                                  if ( !*(_BYTE *)(v5 + 68) )
                                  {
                                    v21 = v5;
                                    _libc_free(*(_QWORD *)(v5 + 48));
                                    v5 = v21;
                                  }
                                  j_j___libc_free_0(v5);
                                  if ( !v9 )
                                    break;
                                  v5 = v9;
                                }
                              }
                              v13 = *(_QWORD *)(v4 + 16);
                              if ( !*(_BYTE *)(v4 + 68) )
                                _libc_free(*(_QWORD *)(v4 + 48));
                              j_j___libc_free_0(v4);
                              if ( !v13 )
                                break;
                              v4 = v13;
                            }
                          }
                          v11 = *(_QWORD *)(v3 + 16);
                          if ( !*(_BYTE *)(v3 + 68) )
                            _libc_free(*(_QWORD *)(v3 + 48));
                          j_j___libc_free_0(v3);
                          if ( !v11 )
                            break;
                          v3 = v11;
                        }
                      }
                      v10 = *(_QWORD *)(v2 + 16);
                      if ( !*(_BYTE *)(v2 + 68) )
                        _libc_free(*(_QWORD *)(v2 + 48));
                      j_j___libc_free_0(v2);
                      if ( !v10 )
                        break;
                      v2 = v10;
                    }
                  }
                  v12 = *(_QWORD *)(v1 + 16);
                  if ( !*(_BYTE *)(v1 + 68) )
                    _libc_free(*(_QWORD *)(v1 + 48));
                  j_j___libc_free_0(v1);
                  if ( !v12 )
                    break;
                  v1 = v12;
                }
              }
              v14 = *(_QWORD *)(v23 + 16);
              if ( !*(_BYTE *)(v23 + 68) )
                _libc_free(*(_QWORD *)(v23 + 48));
              j_j___libc_free_0(v23);
              if ( !v14 )
                break;
              v23 = v14;
            }
          }
          v15 = *(_QWORD *)(v24 + 16);
          if ( !*(_BYTE *)(v24 + 68) )
            _libc_free(*(_QWORD *)(v24 + 48));
          j_j___libc_free_0(v24);
          if ( !v15 )
            break;
          v24 = v15;
        }
      }
      v16 = *(_QWORD *)(v22 + 16);
      if ( !*(_BYTE *)(v22 + 68) )
        _libc_free(*(_QWORD *)(v22 + 48));
      j_j___libc_free_0(v22);
      if ( !v16 )
        break;
      v22 = v16;
    }
  }
}
