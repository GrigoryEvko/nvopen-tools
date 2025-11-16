// Function: sub_28E6DF0
// Address: 0x28e6df0
//
void __fastcall sub_28E6DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 *v12; // r14
  __int64 v13; // r13
  __int64 v14; // rcx
  __int64 *v15; // r12
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 *v19; // r14
  __int64 v20; // rax
  __int64 *v21; // r13
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 *v24; // r15
  __int64 v25; // rbx
  __int64 v26; // r12
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-E8h]
  _QWORD *v31; // [rsp+10h] [rbp-E0h]
  __int64 v32; // [rsp+20h] [rbp-D0h]
  __int64 v33; // [rsp+28h] [rbp-C8h]
  __int64 *v34; // [rsp+30h] [rbp-C0h]
  __int64 v35; // [rsp+38h] [rbp-B8h]
  __int64 *v36; // [rsp+40h] [rbp-B0h]
  __int64 v37; // [rsp+48h] [rbp-A8h]
  __int64 v38; // [rsp+50h] [rbp-A0h]
  __int64 v39; // [rsp+58h] [rbp-98h]
  __int64 v40; // [rsp+60h] [rbp-90h]
  __int64 v41; // [rsp+68h] [rbp-88h]
  __int64 v42; // [rsp+70h] [rbp-80h]
  __int64 *v43; // [rsp+78h] [rbp-78h]
  __int64 *v44; // [rsp+80h] [rbp-70h]
  __int64 *v45; // [rsp+88h] [rbp-68h]
  __int64 *v46; // [rsp+90h] [rbp-60h]
  __int64 v47; // [rsp+98h] [rbp-58h]
  __int64 *v48; // [rsp+A0h] [rbp-50h]
  __int64 v49; // [rsp+A8h] [rbp-48h]
  __int64 *v50; // [rsp+B0h] [rbp-40h]
  __int64 v51; // [rsp+B8h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 16);
  v40 = v7;
  if ( *(_QWORD *)(a2 + 8) != v7 )
  {
    v43 = *(__int64 **)(a2 + 8);
    do
    {
      v8 = *(_QWORD *)(*v43 + 16);
      v39 = *v43;
      v38 = v8;
      if ( *(_QWORD *)(*v43 + 8) != v8 )
      {
        v44 = *(__int64 **)(*v43 + 8);
        do
        {
          v9 = *v44;
          v37 = *v44;
          v36 = *(__int64 **)(*v44 + 16);
          if ( *(__int64 **)(*v44 + 8) != v36 )
          {
            v45 = *(__int64 **)(*v44 + 8);
            do
            {
              v10 = *v45;
              v35 = *v45;
              v34 = *(__int64 **)(*v45 + 16);
              if ( *(__int64 **)(*v45 + 8) != v34 )
              {
                v46 = *(__int64 **)(*v45 + 8);
                v11 = a1;
                do
                {
                  v33 = *v46;
                  v48 = *(__int64 **)(*v46 + 16);
                  if ( *(__int64 **)(*v46 + 8) != v48 )
                  {
                    v12 = *(__int64 **)(*v46 + 8);
                    do
                    {
                      v13 = *v12;
                      v14 = *(_QWORD *)(*v12 + 16);
                      v51 = v14;
                      if ( *(_QWORD *)(*v12 + 8) != v14 )
                      {
                        v50 = v12;
                        v15 = *(__int64 **)(*v12 + 8);
                        v49 = *v12;
                        v16 = v11;
                        do
                        {
                          v17 = *v15;
                          v18 = *(_QWORD *)(*v15 + 16);
                          v41 = v18;
                          if ( *(_QWORD *)(*v15 + 8) != v18 )
                          {
                            v19 = *(__int64 **)(*v15 + 8);
                            v20 = v16;
                            v21 = v15;
                            v22 = v20;
                            do
                            {
                              v23 = *(_QWORD *)(*v19 + 16);
                              v32 = *v19;
                              v47 = v23;
                              if ( *(_QWORD *)(*v19 + 8) != v23 )
                              {
                                v42 = v17;
                                v24 = *(__int64 **)(*v19 + 8);
                                v25 = v22;
                                do
                                {
                                  v26 = *v24;
                                  v27 = *(_QWORD **)(*v24 + 8);
                                  v28 = *(_QWORD *)(*v24 + 16);
                                  if ( v27 != (_QWORD *)v28 )
                                  {
                                    do
                                    {
                                      v30 = v28;
                                      v31 = v27;
                                      sub_28E6DF0(v25, *v27);
                                      v28 = v30;
                                      v27 = v31 + 1;
                                    }
                                    while ( (_QWORD *)v30 != v31 + 1 );
                                  }
                                  ++v24;
                                  sub_28E6650(v25, v26, v28, v23, a5, a6);
                                }
                                while ( (__int64 *)v47 != v24 );
                                v22 = v25;
                                v17 = v42;
                              }
                              ++v19;
                              sub_28E6650(v22, v32, v18, v23, a5, a6);
                            }
                            while ( (__int64 *)v41 != v19 );
                            v29 = v22;
                            v15 = v21;
                            v16 = v29;
                          }
                          ++v15;
                          sub_28E6650(v16, v17, v18, v14, a5, a6);
                        }
                        while ( (__int64 *)v51 != v15 );
                        v11 = v16;
                        v12 = v50;
                        v13 = v49;
                      }
                      ++v12;
                      sub_28E6650(v11, v13, v10, v14, a5, a6);
                    }
                    while ( v48 != v12 );
                  }
                  sub_28E6650(v11, v33, v10, v9, a5, a6);
                  ++v46;
                }
                while ( v34 != v46 );
                a1 = v11;
              }
              sub_28E6650(a1, v35, v10, v9, a5, a6);
              ++v45;
            }
            while ( v36 != v45 );
          }
          sub_28E6650(a1, v37, v8, v9, a5, a6);
          ++v44;
        }
        while ( (__int64 *)v38 != v44 );
      }
      sub_28E6650(a1, v39, v8, v7, a5, a6);
      ++v43;
    }
    while ( (__int64 *)v40 != v43 );
  }
  sub_28E6650(a1, a2, a3, v7, a5, a6);
}
