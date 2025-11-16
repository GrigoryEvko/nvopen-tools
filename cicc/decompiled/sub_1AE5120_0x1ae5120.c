// Function: sub_1AE5120
// Address: 0x1ae5120
//
char __fastcall sub_1AE5120(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 *v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // r12
  __int64 v22; // r13
  char v23; // r15
  __int64 *v24; // r14
  __int64 *v25; // r8
  __int64 v26; // rdi
  char v27; // al
  char v28; // al
  __int64 v29; // rax
  __int64 *v31; // [rsp+8h] [rbp-108h]
  __int64 v33; // [rsp+18h] [rbp-F8h]
  __int64 v34; // [rsp+20h] [rbp-F0h]
  __int64 *v35; // [rsp+28h] [rbp-E8h]
  __int64 v36; // [rsp+30h] [rbp-E0h]
  __int64 *v37; // [rsp+38h] [rbp-D8h]
  __int64 v38; // [rsp+40h] [rbp-D0h]
  __int64 *v39; // [rsp+48h] [rbp-C8h]
  __int64 v40; // [rsp+50h] [rbp-C0h]
  __int64 *v41; // [rsp+58h] [rbp-B8h]
  __int64 *v42; // [rsp+60h] [rbp-B0h]
  __int64 v43; // [rsp+68h] [rbp-A8h]
  __int64 *v44; // [rsp+70h] [rbp-A0h]
  __int64 *v45; // [rsp+78h] [rbp-98h]
  __int64 *v46; // [rsp+80h] [rbp-90h]
  __int64 *v47; // [rsp+88h] [rbp-88h]
  __int64 *v48; // [rsp+90h] [rbp-80h]
  __int64 *v49; // [rsp+98h] [rbp-78h]
  __int64 v50; // [rsp+A0h] [rbp-70h]
  __int64 *v51; // [rsp+A8h] [rbp-68h]
  __int64 *v52; // [rsp+B0h] [rbp-60h]
  __int64 v53; // [rsp+B8h] [rbp-58h]
  __int64 v54; // [rsp+B8h] [rbp-58h]
  __int64 v55; // [rsp+C0h] [rbp-50h]
  __int64 v56; // [rsp+C0h] [rbp-50h]
  __int64 v57; // [rsp+C0h] [rbp-50h]
  char v58; // [rsp+C8h] [rbp-48h]
  char v59; // [rsp+C9h] [rbp-47h]
  char v60; // [rsp+CAh] [rbp-46h]
  char v61; // [rsp+CBh] [rbp-45h]
  char v62; // [rsp+CCh] [rbp-44h]
  char v63; // [rsp+CDh] [rbp-43h]
  char v64; // [rsp+CEh] [rbp-42h]
  char v65; // [rsp+CFh] [rbp-41h]
  __int64 *v66; // [rsp+D0h] [rbp-40h]
  __int64 *v67; // [rsp+D8h] [rbp-38h]

  v61 = 0;
  v41 = *(__int64 **)(a1 + 16);
  if ( *(__int64 **)(a1 + 8) != v41 )
  {
    v44 = *(__int64 **)(a1 + 8);
    do
    {
      v60 = 0;
      v40 = *v44;
      v39 = *(__int64 **)(*v44 + 16);
      if ( *(__int64 **)(*v44 + 8) != v39 )
      {
        v49 = *(__int64 **)(*v44 + 8);
        do
        {
          v59 = 0;
          v38 = *v49;
          v37 = *(__int64 **)(*v49 + 16);
          if ( *(__int64 **)(*v49 + 8) != v37 )
          {
            v48 = *(__int64 **)(*v49 + 8);
            do
            {
              v58 = 0;
              v36 = *v48;
              v35 = *(__int64 **)(*v48 + 16);
              if ( *(__int64 **)(*v48 + 8) != v35 )
              {
                v45 = *(__int64 **)(*v48 + 8);
                v7 = a2;
                v8 = a3;
                do
                {
                  v64 = 0;
                  v34 = *v45;
                  v47 = *(__int64 **)(*v45 + 16);
                  if ( *(__int64 **)(*v45 + 8) != v47 )
                  {
                    v67 = *(__int64 **)(*v45 + 8);
                    v9 = v8;
                    v10 = v7;
                    do
                    {
                      v65 = 0;
                      v11 = *v67;
                      v51 = *(__int64 **)(*v67 + 16);
                      if ( *(__int64 **)(*v67 + 8) != v51 )
                      {
                        v66 = *(__int64 **)(*v67 + 8);
                        v12 = a4;
                        v13 = v9;
                        v14 = v10;
                        v50 = *v67;
                        do
                        {
                          v62 = 0;
                          v43 = *v66;
                          v42 = *(__int64 **)(*v66 + 16);
                          if ( *(__int64 **)(*v66 + 8) != v42 )
                          {
                            v52 = *(__int64 **)(*v66 + 8);
                            v15 = v13;
                            v16 = v12;
                            v17 = v14;
                            v18 = v15;
                            do
                            {
                              v63 = 0;
                              v33 = *v52;
                              v19 = *(__int64 **)(*v52 + 8);
                              v46 = *(__int64 **)(*v52 + 16);
                              if ( v19 != v46 )
                              {
                                v20 = v18;
                                v21 = v17;
                                do
                                {
                                  v22 = *v19;
                                  v23 = 0;
                                  v24 = *(__int64 **)(*v19 + 8);
                                  v25 = *(__int64 **)(*v19 + 16);
                                  if ( v24 != v25 )
                                  {
                                    do
                                    {
                                      v26 = *v24;
                                      v53 = v16;
                                      ++v24;
                                      v31 = v25;
                                      v55 = v20;
                                      v27 = sub_1AE5120(v26, v21, v20, v16);
                                      v25 = v31;
                                      v20 = v55;
                                      v23 |= v27;
                                      v16 = v53;
                                    }
                                    while ( v31 != v24 );
                                  }
                                  v54 = v16;
                                  ++v19;
                                  v56 = v20;
                                  v28 = sub_1AE4880(v22, v21, v20, v16);
                                  v20 = v56;
                                  v16 = v54;
                                  v63 |= v28 | v23;
                                }
                                while ( v46 != v19 );
                                v17 = v21;
                                v18 = v56;
                              }
                              v57 = v16;
                              ++v52;
                              v62 |= v63 | sub_1AE4880(v33, v17, v18, v16);
                              v16 = v57;
                            }
                            while ( v42 != v52 );
                            v29 = v18;
                            v12 = v57;
                            v14 = v17;
                            v13 = v29;
                          }
                          ++v66;
                          v65 |= v62 | sub_1AE4880(v43, v14, v13, v12);
                        }
                        while ( v51 != v66 );
                        v11 = v50;
                        v10 = v14;
                        v9 = v13;
                        a4 = v12;
                      }
                      ++v67;
                      v64 |= v65 | sub_1AE4880(v11, v10, v9, a4);
                    }
                    while ( v47 != v67 );
                    v7 = v10;
                    v8 = v9;
                  }
                  ++v45;
                  v58 |= v64 | sub_1AE4880(v34, v7, v8, a4);
                }
                while ( v35 != v45 );
                a3 = v8;
                a2 = v7;
              }
              ++v48;
              v59 |= v58 | sub_1AE4880(v36, a2, a3, a4);
            }
            while ( v37 != v48 );
          }
          ++v49;
          v60 |= v59 | sub_1AE4880(v38, a2, a3, a4);
        }
        while ( v39 != v49 );
      }
      ++v44;
      v61 |= v60 | sub_1AE4880(v40, a2, a3, a4);
    }
    while ( v41 != v44 );
  }
  return v61 | sub_1AE4880(a1, a2, a3, a4);
}
