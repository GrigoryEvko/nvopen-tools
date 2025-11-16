// Function: sub_31D63F0
// Address: 0x31d63f0
//
__int64 __fastcall sub_31D63F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rbx
  int v9; // r14d
  __int64 v10; // rdx
  int v11; // r15d
  __int64 v12; // rcx
  __int64 v13; // r13
  int v14; // r12d
  _BYTE *v15; // rdi
  int v16; // eax
  __int64 i; // [rsp+0h] [rbp-80h]
  int v19; // [rsp+8h] [rbp-78h]
  int v20; // [rsp+Ch] [rbp-74h]
  __int64 v21; // [rsp+10h] [rbp-70h]
  unsigned int v22; // [rsp+18h] [rbp-68h]
  int v23; // [rsp+1Ch] [rbp-64h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  int v27; // [rsp+38h] [rbp-48h]
  int v28; // [rsp+3Ch] [rbp-44h]
  __int64 v29; // [rsp+40h] [rbp-40h]
  __int64 v30; // [rsp+48h] [rbp-38h]

  if ( a1 )
  {
    v22 = 1;
    if ( *(_BYTE *)a1 != 3 )
    {
      v22 = 0;
      for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v1 = *(_QWORD *)(i + 24);
        if ( *(_BYTE *)v1 <= 0x15u )
        {
          if ( *(_BYTE *)v1 == 3 )
          {
            ++v22;
          }
          else
          {
            v21 = *(_QWORD *)(v1 + 16);
            if ( v21 )
            {
              v19 = 0;
              do
              {
                v2 = *(_QWORD *)(v21 + 24);
                if ( *(_BYTE *)v2 <= 0x15u )
                {
                  if ( *(_BYTE *)v2 == 3 )
                  {
                    ++v19;
                  }
                  else
                  {
                    v24 = *(_QWORD *)(v2 + 16);
                    if ( v24 )
                    {
                      v20 = 0;
                      do
                      {
                        v3 = *(_QWORD *)(v24 + 24);
                        if ( *(_BYTE *)v3 <= 0x15u )
                        {
                          if ( *(_BYTE *)v3 == 3 )
                          {
                            ++v20;
                          }
                          else
                          {
                            v25 = *(_QWORD *)(v3 + 16);
                            if ( v25 )
                            {
                              v23 = 0;
                              do
                              {
                                v4 = *(_QWORD *)(v25 + 24);
                                if ( *(_BYTE *)v4 <= 0x15u )
                                {
                                  if ( *(_BYTE *)v4 == 3 )
                                  {
                                    ++v23;
                                  }
                                  else
                                  {
                                    v29 = *(_QWORD *)(v4 + 16);
                                    if ( v29 )
                                    {
                                      v27 = 0;
                                      do
                                      {
                                        v5 = *(_QWORD *)(v29 + 24);
                                        if ( *(_BYTE *)v5 <= 0x15u )
                                        {
                                          if ( *(_BYTE *)v5 == 3 )
                                          {
                                            ++v27;
                                          }
                                          else
                                          {
                                            v6 = *(_QWORD *)(v5 + 16);
                                            if ( v6 )
                                            {
                                              v28 = 0;
                                              do
                                              {
                                                v7 = *(_QWORD *)(v6 + 24);
                                                if ( *(_BYTE *)v7 <= 0x15u )
                                                {
                                                  if ( *(_BYTE *)v7 == 3 )
                                                  {
                                                    ++v28;
                                                  }
                                                  else
                                                  {
                                                    v8 = *(_QWORD *)(v7 + 16);
                                                    if ( v8 )
                                                    {
                                                      v26 = v6;
                                                      v9 = 0;
                                                      do
                                                      {
                                                        v10 = *(_QWORD *)(v8 + 24);
                                                        if ( *(_BYTE *)v10 <= 0x15u )
                                                        {
                                                          if ( *(_BYTE *)v10 == 3 )
                                                          {
                                                            ++v9;
                                                          }
                                                          else if ( *(_QWORD *)(v10 + 16) )
                                                          {
                                                            v30 = *(_QWORD *)(v10 + 16);
                                                            v11 = 0;
                                                            do
                                                            {
                                                              v12 = *(_QWORD *)(v30 + 24);
                                                              if ( *(_BYTE *)v12 <= 0x15u )
                                                              {
                                                                if ( *(_BYTE *)v12 == 3 )
                                                                {
                                                                  ++v11;
                                                                }
                                                                else
                                                                {
                                                                  v13 = *(_QWORD *)(v12 + 16);
                                                                  if ( v13 )
                                                                  {
                                                                    v14 = 0;
                                                                    do
                                                                    {
                                                                      v15 = *(_BYTE **)(v13 + 24);
                                                                      if ( *v15 >= 0x16u )
                                                                        v15 = 0;
                                                                      v16 = sub_31D63F0(v15);
                                                                      v13 = *(_QWORD *)(v13 + 8);
                                                                      v14 += v16;
                                                                    }
                                                                    while ( v13 );
                                                                    v11 += v14;
                                                                  }
                                                                }
                                                              }
                                                              v30 = *(_QWORD *)(v30 + 8);
                                                            }
                                                            while ( v30 );
                                                            v9 += v11;
                                                          }
                                                        }
                                                        v8 = *(_QWORD *)(v8 + 8);
                                                      }
                                                      while ( v8 );
                                                      v28 += v9;
                                                      v6 = v26;
                                                    }
                                                  }
                                                }
                                                v6 = *(_QWORD *)(v6 + 8);
                                              }
                                              while ( v6 );
                                              v27 += v28;
                                            }
                                          }
                                        }
                                        v29 = *(_QWORD *)(v29 + 8);
                                      }
                                      while ( v29 );
                                      v23 += v27;
                                    }
                                  }
                                }
                                v25 = *(_QWORD *)(v25 + 8);
                              }
                              while ( v25 );
                              v20 += v23;
                            }
                          }
                        }
                        v24 = *(_QWORD *)(v24 + 8);
                      }
                      while ( v24 );
                      v19 += v20;
                    }
                  }
                }
                v21 = *(_QWORD *)(v21 + 8);
              }
              while ( v21 );
              v22 += v19;
            }
          }
        }
      }
    }
  }
  else
  {
    return 0;
  }
  return v22;
}
