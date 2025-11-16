// Function: sub_1A92A70
// Address: 0x1a92a70
//
void __fastcall sub_1A92A70(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 *v4; // r14
  __int64 v5; // r13
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 v12; // r12
  __int64 *v13; // r15
  __int64 v14; // rbx
  __int64 v15; // r12
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // [rsp+8h] [rbp-E8h]
  _QWORD *v20; // [rsp+10h] [rbp-E0h]
  __int64 v21; // [rsp+20h] [rbp-D0h]
  __int64 v22; // [rsp+28h] [rbp-C8h]
  __int64 *v23; // [rsp+30h] [rbp-C0h]
  __int64 v24; // [rsp+38h] [rbp-B8h]
  __int64 *v25; // [rsp+40h] [rbp-B0h]
  __int64 v26; // [rsp+48h] [rbp-A8h]
  __int64 *v27; // [rsp+50h] [rbp-A0h]
  __int64 v28; // [rsp+58h] [rbp-98h]
  __int64 *v29; // [rsp+60h] [rbp-90h]
  __int64 *v30; // [rsp+68h] [rbp-88h]
  __int64 v31; // [rsp+70h] [rbp-80h]
  __int64 *v32; // [rsp+78h] [rbp-78h]
  __int64 *v33; // [rsp+80h] [rbp-70h]
  __int64 *v34; // [rsp+88h] [rbp-68h]
  __int64 *v35; // [rsp+90h] [rbp-60h]
  __int64 *v36; // [rsp+98h] [rbp-58h]
  __int64 *v37; // [rsp+A0h] [rbp-50h]
  __int64 v38; // [rsp+A8h] [rbp-48h]
  __int64 *v39; // [rsp+B0h] [rbp-40h]
  __int64 *v40; // [rsp+B8h] [rbp-38h]

  v29 = *(__int64 **)(a2 + 16);
  if ( *(__int64 **)(a2 + 8) != v29 )
  {
    v32 = *(__int64 **)(a2 + 8);
    do
    {
      v28 = *v32;
      v27 = *(__int64 **)(*v32 + 16);
      if ( *(__int64 **)(*v32 + 8) != v27 )
      {
        v33 = *(__int64 **)(*v32 + 8);
        do
        {
          v26 = *v33;
          v25 = *(__int64 **)(*v33 + 16);
          if ( *(__int64 **)(*v33 + 8) != v25 )
          {
            v34 = *(__int64 **)(*v33 + 8);
            do
            {
              v24 = *v34;
              v23 = *(__int64 **)(*v34 + 16);
              if ( *(__int64 **)(*v34 + 8) != v23 )
              {
                v35 = *(__int64 **)(*v34 + 8);
                v3 = a1;
                do
                {
                  v22 = *v35;
                  v37 = *(__int64 **)(*v35 + 16);
                  if ( *(__int64 **)(*v35 + 8) != v37 )
                  {
                    v4 = *(__int64 **)(*v35 + 8);
                    do
                    {
                      v5 = *v4;
                      v40 = *(__int64 **)(*v4 + 16);
                      if ( *(__int64 **)(*v4 + 8) != v40 )
                      {
                        v39 = v4;
                        v6 = *(__int64 **)(*v4 + 8);
                        v38 = *v4;
                        v7 = v3;
                        do
                        {
                          v8 = *v6;
                          v30 = *(__int64 **)(*v6 + 16);
                          if ( *(__int64 **)(*v6 + 8) != v30 )
                          {
                            v9 = *(__int64 **)(*v6 + 8);
                            v10 = v7;
                            v11 = v6;
                            v12 = v10;
                            do
                            {
                              v21 = *v9;
                              v36 = *(__int64 **)(*v9 + 16);
                              if ( *(__int64 **)(*v9 + 8) != v36 )
                              {
                                v31 = v8;
                                v13 = *(__int64 **)(*v9 + 8);
                                v14 = v12;
                                do
                                {
                                  v15 = *v13;
                                  v16 = *(_QWORD **)(*v13 + 8);
                                  v17 = *(_QWORD **)(*v13 + 16);
                                  if ( v16 != v17 )
                                  {
                                    do
                                    {
                                      v19 = v17;
                                      v20 = v16;
                                      sub_1A92A70(v14, *v16);
                                      v17 = v19;
                                      v16 = v20 + 1;
                                    }
                                    while ( v19 != v20 + 1 );
                                  }
                                  ++v13;
                                  sub_1A922B0(v14, v15);
                                }
                                while ( v36 != v13 );
                                v12 = v14;
                                v8 = v31;
                              }
                              ++v9;
                              sub_1A922B0(v12, v21);
                            }
                            while ( v30 != v9 );
                            v18 = v12;
                            v6 = v11;
                            v7 = v18;
                          }
                          ++v6;
                          sub_1A922B0(v7, v8);
                        }
                        while ( v40 != v6 );
                        v3 = v7;
                        v4 = v39;
                        v5 = v38;
                      }
                      ++v4;
                      sub_1A922B0(v3, v5);
                    }
                    while ( v37 != v4 );
                  }
                  sub_1A922B0(v3, v22);
                  ++v35;
                }
                while ( v23 != v35 );
                a1 = v3;
              }
              sub_1A922B0(a1, v24);
              ++v34;
            }
            while ( v25 != v34 );
          }
          sub_1A922B0(a1, v26);
          ++v33;
        }
        while ( v27 != v33 );
      }
      sub_1A922B0(a1, v28);
      ++v32;
    }
    while ( v29 != v32 );
  }
  sub_1A922B0(a1, a2);
}
