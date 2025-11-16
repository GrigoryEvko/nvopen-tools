// Function: sub_2852CB0
// Address: 0x2852cb0
//
__int64 __fastcall sub_2852CB0(__int64 a1)
{
  __int64 v1; // r14
  unsigned int v2; // r15d
  __int64 v3; // r13
  unsigned int v4; // edx
  unsigned int v5; // r15d
  __int64 v6; // rbx
  _QWORD *v7; // r13
  _QWORD *v8; // r14
  unsigned int v9; // ebx
  _QWORD *v10; // r12
  unsigned int v11; // r14d
  unsigned int v12; // eax
  unsigned int v13; // r12d
  unsigned int v14; // r12d
  unsigned int v15; // ebx
  unsigned int v16; // ebx
  unsigned int v17; // r9d
  unsigned int v18; // r12d
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 v24; // [rsp+8h] [rbp-C8h]
  __int64 v25; // [rsp+10h] [rbp-C0h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  __int64 v27; // [rsp+20h] [rbp-B0h]
  unsigned int v28; // [rsp+28h] [rbp-A8h]
  unsigned int v29; // [rsp+2Ch] [rbp-A4h]
  __int64 v30; // [rsp+30h] [rbp-A0h]
  __int64 v31; // [rsp+38h] [rbp-98h]
  __int64 v32; // [rsp+40h] [rbp-90h]
  unsigned int v33; // [rsp+48h] [rbp-88h]
  unsigned int v34; // [rsp+4Ch] [rbp-84h]
  __int64 v35; // [rsp+50h] [rbp-80h]
  __int64 v36; // [rsp+58h] [rbp-78h]
  __int64 v37; // [rsp+60h] [rbp-70h]
  __int64 v38; // [rsp+68h] [rbp-68h]
  __int64 v39; // [rsp+70h] [rbp-60h]
  unsigned int v40; // [rsp+78h] [rbp-58h]
  unsigned int v41; // [rsp+7Ch] [rbp-54h]
  __int64 v42; // [rsp+80h] [rbp-50h]
  __int64 v43; // [rsp+88h] [rbp-48h]
  __int64 v44; // [rsp+90h] [rbp-40h]
  __int64 v45; // [rsp+98h] [rbp-38h]

  v26 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) == v26 )
    return 1;
  v32 = *(_QWORD *)(a1 + 8);
  v33 = 0;
  do
  {
    v25 = *(_QWORD *)(*(_QWORD *)v32 + 16LL);
    if ( v25 == *(_QWORD *)(*(_QWORD *)v32 + 8LL) )
    {
      v22 = 1;
    }
    else
    {
      v31 = *(_QWORD *)(*(_QWORD *)v32 + 8LL);
      v29 = 0;
      do
      {
        v24 = *(_QWORD *)(*(_QWORD *)v31 + 16LL);
        if ( v24 == *(_QWORD *)(*(_QWORD *)v31 + 8LL) )
        {
          v21 = 1;
        }
        else
        {
          v30 = *(_QWORD *)(*(_QWORD *)v31 + 8LL);
          v28 = 0;
          do
          {
            v27 = *(_QWORD *)(*(_QWORD *)v30 + 16LL);
            if ( v27 == *(_QWORD *)(*(_QWORD *)v30 + 8LL) )
            {
              v20 = 1;
            }
            else
            {
              v35 = *(_QWORD *)(*(_QWORD *)v30 + 8LL);
              v34 = 0;
              do
              {
                v36 = *(_QWORD *)(*(_QWORD *)v35 + 16LL);
                if ( v36 == *(_QWORD *)(*(_QWORD *)v35 + 8LL) )
                {
                  v19 = 1;
                }
                else
                {
                  v40 = 0;
                  v1 = *(_QWORD *)(*(_QWORD *)v35 + 8LL);
                  do
                  {
                    v37 = *(_QWORD *)(*(_QWORD *)v1 + 16LL);
                    if ( v37 == *(_QWORD *)(*(_QWORD *)v1 + 8LL) )
                    {
                      v18 = 1;
                    }
                    else
                    {
                      v43 = *(_QWORD *)(*(_QWORD *)v1 + 8LL);
                      v41 = 0;
                      v38 = v1;
                      do
                      {
                        v39 = *(_QWORD *)(*(_QWORD *)v43 + 16LL);
                        if ( v39 == *(_QWORD *)(*(_QWORD *)v43 + 8LL) )
                        {
                          v17 = 1;
                        }
                        else
                        {
                          v2 = 0;
                          v3 = *(_QWORD *)(*(_QWORD *)v43 + 8LL);
                          do
                          {
                            v42 = *(_QWORD *)(*(_QWORD *)v3 + 16LL);
                            if ( v42 == *(_QWORD *)(*(_QWORD *)v3 + 8LL) )
                            {
                              v16 = 1;
                            }
                            else
                            {
                              v45 = v3;
                              v4 = v2;
                              v5 = 0;
                              v6 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
                              do
                              {
                                v7 = *(_QWORD **)(*(_QWORD *)v6 + 16LL);
                                v8 = *(_QWORD **)(*(_QWORD *)v6 + 8LL);
                                if ( v7 == v8 )
                                {
                                  v14 = 1;
                                }
                                else
                                {
                                  v44 = v6;
                                  v9 = 0;
                                  v10 = v8;
                                  v11 = v4;
                                  do
                                  {
                                    v12 = sub_2852CB0(*v10);
                                    if ( v9 < v12 )
                                      v9 = v12;
                                    ++v10;
                                  }
                                  while ( v7 != v10 );
                                  v13 = v9;
                                  v6 = v44;
                                  v4 = v11;
                                  v14 = v13 + 1;
                                }
                                if ( v5 < v14 )
                                  v5 = v14;
                                v6 += 8;
                              }
                              while ( v42 != v6 );
                              v15 = v5;
                              v3 = v45;
                              v2 = v4;
                              v16 = v15 + 1;
                            }
                            if ( v2 < v16 )
                              v2 = v16;
                            v3 += 8;
                          }
                          while ( v39 != v3 );
                          v17 = v2 + 1;
                        }
                        if ( v41 >= v17 )
                          v17 = v41;
                        v43 += 8;
                        v41 = v17;
                      }
                      while ( v37 != v43 );
                      v1 = v38;
                      v18 = v17 + 1;
                    }
                    if ( v40 >= v18 )
                      v18 = v40;
                    v1 += 8;
                    v40 = v18;
                  }
                  while ( v36 != v1 );
                  v19 = v18 + 1;
                }
                if ( v34 >= v19 )
                  v19 = v34;
                v35 += 8;
                v34 = v19;
              }
              while ( v27 != v35 );
              v20 = v19 + 1;
            }
            if ( v28 >= v20 )
              v20 = v28;
            v30 += 8;
            v28 = v20;
          }
          while ( v24 != v30 );
          v21 = v20 + 1;
        }
        if ( v29 >= v21 )
          v21 = v29;
        v31 += 8;
        v29 = v21;
      }
      while ( v25 != v31 );
      v22 = v21 + 1;
    }
    if ( v33 >= v22 )
      v22 = v33;
    v32 += 8;
    v33 = v22;
  }
  while ( v26 != v32 );
  return v22 + 1;
}
