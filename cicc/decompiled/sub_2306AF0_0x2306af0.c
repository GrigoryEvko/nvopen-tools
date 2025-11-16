// Function: sub_2306AF0
// Address: 0x2306af0
//
void __fastcall sub_2306AF0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r15
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // r13
  _QWORD *v11; // rax
  _QWORD *v12; // r15
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // r14
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 *v18; // rbx
  __int64 *v19; // r13
  __int64 v20; // rsi
  _QWORD *v21; // [rsp-A8h] [rbp-A8h]
  _QWORD *v22; // [rsp-A0h] [rbp-A0h]
  _QWORD *v23; // [rsp-98h] [rbp-98h]
  _QWORD *v24; // [rsp-90h] [rbp-90h]
  _QWORD *v25; // [rsp-88h] [rbp-88h]
  _QWORD *v26; // [rsp-80h] [rbp-80h]
  _QWORD *v27; // [rsp-78h] [rbp-78h]
  _QWORD *v28; // [rsp-70h] [rbp-70h]
  _QWORD *v29; // [rsp-68h] [rbp-68h]
  _QWORD *v30; // [rsp-60h] [rbp-60h]
  _QWORD *v31; // [rsp-58h] [rbp-58h]
  _QWORD *i; // [rsp-50h] [rbp-50h]
  _QWORD *v33; // [rsp-48h] [rbp-48h]
  _QWORD *v34; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v2 = a1;
    v3 = (_QWORD *)a2[6];
    v4 = (_QWORD *)a2[5];
    a2[2] = a1;
    v23 = v3;
    if ( v4 != v3 )
    {
      v25 = v4;
      do
      {
        v5 = (_QWORD *)*v25;
        if ( *v25 )
        {
          v5[2] = v2;
          v22 = (_QWORD *)v5[6];
          if ( (_QWORD *)v5[5] != v22 )
          {
            v26 = (_QWORD *)v5[5];
            do
            {
              v6 = (_QWORD *)*v26;
              if ( *v26 )
              {
                v6[2] = v2;
                v21 = (_QWORD *)v6[6];
                if ( (_QWORD *)v6[5] != v21 )
                {
                  v27 = (_QWORD *)v6[5];
                  do
                  {
                    v7 = (_QWORD *)*v27;
                    if ( *v27 )
                    {
                      v7[2] = v2;
                      v24 = (_QWORD *)v7[6];
                      if ( (_QWORD *)v7[5] != v24 )
                      {
                        v28 = (_QWORD *)v7[5];
                        do
                        {
                          v8 = (_QWORD *)*v28;
                          if ( *v28 )
                          {
                            v8[2] = v2;
                            v29 = (_QWORD *)v8[6];
                            if ( (_QWORD *)v8[5] != v29 )
                            {
                              v34 = (_QWORD *)v8[5];
                              do
                              {
                                v9 = (_QWORD *)*v34;
                                if ( *v34 )
                                {
                                  v9[2] = v2;
                                  v30 = (_QWORD *)v9[6];
                                  if ( (_QWORD *)v9[5] != v30 )
                                  {
                                    v33 = (_QWORD *)v9[5];
                                    v10 = v2;
                                    do
                                    {
                                      v11 = (_QWORD *)*v33;
                                      if ( *v33 )
                                      {
                                        v11[2] = v10;
                                        v12 = (_QWORD *)v11[5];
                                        for ( i = (_QWORD *)v11[6]; i != v12; ++v12 )
                                        {
                                          v13 = (_QWORD *)*v12;
                                          if ( *v12 )
                                          {
                                            v14 = (_QWORD *)v13[5];
                                            v15 = (_QWORD *)v13[6];
                                            v13[2] = v10;
                                            if ( v14 != v15 )
                                            {
                                              v31 = v12;
                                              v16 = v10;
                                              do
                                              {
                                                v17 = (_QWORD *)*v14;
                                                if ( *v14 )
                                                {
                                                  v18 = (__int64 *)v17[5];
                                                  v19 = (__int64 *)v17[6];
                                                  v17[2] = v16;
                                                  while ( v19 != v18 )
                                                  {
                                                    v20 = *v18++;
                                                    sub_2306AF0(v16, v20);
                                                  }
                                                }
                                                ++v14;
                                              }
                                              while ( v15 != v14 );
                                              v10 = v16;
                                              v12 = v31;
                                            }
                                          }
                                        }
                                      }
                                      ++v33;
                                    }
                                    while ( v30 != v33 );
                                    v2 = v10;
                                  }
                                }
                                ++v34;
                              }
                              while ( v29 != v34 );
                            }
                          }
                          ++v28;
                        }
                        while ( v24 != v28 );
                      }
                    }
                    ++v27;
                  }
                  while ( v21 != v27 );
                }
              }
              ++v26;
            }
            while ( v22 != v26 );
          }
        }
        ++v25;
      }
      while ( v23 != v25 );
    }
  }
}
