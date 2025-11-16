// Function: sub_11D1D20
// Address: 0x11d1d20
//
char __fastcall sub_11D1D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // r14
  __int64 v25; // r13
  __int64 v26; // r15
  __int64 v27; // r8
  __int64 v28; // r15
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // rcx
  __int64 *v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // r12
  char v35; // r14
  __int64 *v36; // r13
  __int64 v37; // r9
  __int64 v38; // rdi
  char v39; // al
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 *v47; // [rsp+0h] [rbp-110h]
  __int64 v49; // [rsp+10h] [rbp-100h]
  __int64 v50; // [rsp+18h] [rbp-F8h]
  __int64 *v51; // [rsp+20h] [rbp-F0h]
  __int64 v52; // [rsp+28h] [rbp-E8h]
  __int64 *v53; // [rsp+30h] [rbp-E0h]
  __int64 v54; // [rsp+38h] [rbp-D8h]
  __int64 *v55; // [rsp+40h] [rbp-D0h]
  __int64 v56; // [rsp+48h] [rbp-C8h]
  __int64 *v57; // [rsp+50h] [rbp-C0h]
  __int64 *v58; // [rsp+58h] [rbp-B8h]
  __int64 v59; // [rsp+60h] [rbp-B0h]
  __int64 *v60; // [rsp+68h] [rbp-A8h]
  __int64 *v61; // [rsp+70h] [rbp-A0h]
  __int64 *v62; // [rsp+78h] [rbp-98h]
  __int64 *v63; // [rsp+80h] [rbp-90h]
  __int64 *v64; // [rsp+88h] [rbp-88h]
  __int64 *v65; // [rsp+90h] [rbp-80h]
  __int64 *v66; // [rsp+98h] [rbp-78h]
  __int64 *v67; // [rsp+A0h] [rbp-70h]
  __int64 v68; // [rsp+A8h] [rbp-68h]
  __int64 v69; // [rsp+A8h] [rbp-68h]
  __int64 v70; // [rsp+B0h] [rbp-60h]
  __int64 v71; // [rsp+B0h] [rbp-60h]
  __int64 v72; // [rsp+B0h] [rbp-60h]
  __int64 v73; // [rsp+B8h] [rbp-58h]
  __int64 v74; // [rsp+B8h] [rbp-58h]
  __int64 v75; // [rsp+B8h] [rbp-58h]
  __int64 v76; // [rsp+B8h] [rbp-58h]
  __int64 v77; // [rsp+C0h] [rbp-50h]
  char v78; // [rsp+C8h] [rbp-48h]
  char v79; // [rsp+C9h] [rbp-47h]
  char v80; // [rsp+CAh] [rbp-46h]
  char v81; // [rsp+CBh] [rbp-45h]
  char v82; // [rsp+CCh] [rbp-44h]
  char v83; // [rsp+CDh] [rbp-43h]
  char v84; // [rsp+CEh] [rbp-42h]
  char v85; // [rsp+CFh] [rbp-41h]
  __int64 *v86; // [rsp+D0h] [rbp-40h]
  __int64 *v87; // [rsp+D8h] [rbp-38h]

  v6 = a2;
  v7 = a4;
  v8 = a5;
  v81 = 0;
  v57 = *(__int64 **)(a1 + 16);
  if ( *(__int64 **)(a1 + 8) != v57 )
  {
    v60 = *(__int64 **)(a1 + 8);
    v10 = a2;
    v11 = a4;
    v12 = a5;
    do
    {
      v80 = 0;
      v56 = *v60;
      v55 = *(__int64 **)(*v60 + 16);
      if ( *(__int64 **)(*v60 + 8) != v55 )
      {
        v65 = *(__int64 **)(*v60 + 8);
        v13 = v12;
        v14 = v10;
        v15 = v13;
        do
        {
          v79 = 0;
          v54 = *v65;
          v53 = *(__int64 **)(*v65 + 16);
          if ( *(__int64 **)(*v65 + 8) != v53 )
          {
            v64 = *(__int64 **)(*v65 + 8);
            v16 = v15;
            v17 = v14;
            v18 = v16;
            do
            {
              v78 = 0;
              v52 = *v64;
              v51 = *(__int64 **)(*v64 + 16);
              if ( *(__int64 **)(*v64 + 8) != v51 )
              {
                v61 = *(__int64 **)(*v64 + 8);
                v19 = v18;
                v20 = a3;
                v21 = v19;
                do
                {
                  v84 = 0;
                  v50 = *v61;
                  v63 = *(__int64 **)(*v61 + 16);
                  if ( *(__int64 **)(*v61 + 8) != v63 )
                  {
                    v87 = *(__int64 **)(*v61 + 8);
                    v22 = v21;
                    v23 = v20;
                    v24 = v17;
                    v25 = v11;
                    v26 = v22;
                    do
                    {
                      v85 = 0;
                      v77 = *v87;
                      v66 = *(__int64 **)(*v87 + 16);
                      if ( *(__int64 **)(*v87 + 8) != v66 )
                      {
                        v86 = *(__int64 **)(*v87 + 8);
                        v27 = v26;
                        v28 = v24;
                        v29 = v23;
                        v30 = v25;
                        do
                        {
                          v82 = 0;
                          v59 = *v86;
                          v58 = *(__int64 **)(*v86 + 16);
                          if ( *(__int64 **)(*v86 + 8) != v58 )
                          {
                            v67 = *(__int64 **)(*v86 + 8);
                            v31 = v30;
                            do
                            {
                              v83 = 0;
                              v49 = *v67;
                              v32 = *(__int64 **)(*v67 + 8);
                              v62 = *(__int64 **)(*v67 + 16);
                              if ( v32 != v62 )
                              {
                                v33 = v29;
                                do
                                {
                                  v34 = *v32;
                                  v35 = 0;
                                  v36 = *(__int64 **)(*v32 + 8);
                                  v37 = *(_QWORD *)(*v32 + 16);
                                  if ( v36 != (__int64 *)v37 )
                                  {
                                    do
                                    {
                                      v38 = *v36;
                                      v68 = v27;
                                      ++v36;
                                      v47 = (__int64 *)v37;
                                      v70 = v31;
                                      v73 = v33;
                                      v39 = sub_11D1D20(v38, v28);
                                      v37 = (__int64)v47;
                                      v33 = v73;
                                      v35 |= v39;
                                      v31 = v70;
                                      v27 = v68;
                                    }
                                    while ( v47 != v36 );
                                  }
                                  v69 = v27;
                                  ++v32;
                                  v71 = v31;
                                  v74 = v33;
                                  v40 = sub_11D0CC0(v34, v28, v33, v31, v27, v37);
                                  v33 = v74;
                                  v31 = v71;
                                  v83 |= v40 | v35;
                                  v27 = v69;
                                }
                                while ( v62 != v32 );
                                v29 = v74;
                              }
                              v72 = v27;
                              v75 = v31;
                              ++v67;
                              v82 |= v83 | sub_11D0CC0(v49, v28, v29, v31, v27, a6);
                              v31 = v75;
                              v27 = v72;
                            }
                            while ( v58 != v67 );
                            v30 = v75;
                          }
                          v76 = v27;
                          ++v86;
                          v85 |= v82 | sub_11D0CC0(v59, v28, v29, v30, v27, a6);
                          v27 = v76;
                        }
                        while ( v66 != v86 );
                        v25 = v30;
                        v23 = v29;
                        v24 = v28;
                        v26 = v76;
                      }
                      ++v87;
                      v84 |= v85 | sub_11D0CC0(v77, v24, v23, v25, v26, a6);
                    }
                    while ( v63 != v87 );
                    v41 = v26;
                    v11 = v25;
                    v17 = v24;
                    v20 = v23;
                    v21 = v41;
                  }
                  ++v61;
                  v78 |= v84 | sub_11D0CC0(v50, v17, v20, v11, v21, a6);
                }
                while ( v51 != v61 );
                v42 = v21;
                a3 = v20;
                v18 = v42;
              }
              ++v64;
              v79 |= v78 | sub_11D0CC0(v52, v17, a3, v11, v18, a6);
            }
            while ( v53 != v64 );
            v43 = v18;
            v14 = v17;
            v15 = v43;
          }
          ++v65;
          v80 |= v79 | sub_11D0CC0(v54, v14, a3, v11, v15, a6);
        }
        while ( v55 != v65 );
        v44 = v15;
        v10 = v14;
        v12 = v44;
      }
      ++v60;
      v81 |= v80 | sub_11D0CC0(v56, v10, a3, v11, v12, a6);
    }
    while ( v57 != v60 );
    v45 = v12;
    v7 = v11;
    v6 = v10;
    v8 = v45;
  }
  return v81 | sub_11D0CC0(a1, v6, a3, v7, v8, a6);
}
