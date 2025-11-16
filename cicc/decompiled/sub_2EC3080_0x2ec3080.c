// Function: sub_2EC3080
// Address: 0x2ec3080
//
void __fastcall sub_2EC3080(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r15
  _QWORD *v5; // rbx
  _QWORD *v6; // r8
  unsigned __int64 v7; // rax
  _QWORD *v8; // r8
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rax
  _QWORD *v11; // rdx
  unsigned __int64 v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // r15
  _QWORD *v16; // rax
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // rbx
  _QWORD *v22; // r14
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rdi
  _QWORD *v25; // rdi
  _QWORD *v26; // rax
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rbx
  _QWORD *v31; // r12
  unsigned __int64 v32; // rdi
  _QWORD *v33; // rbx
  _QWORD *v34; // r12
  unsigned __int64 v35; // rdi
  _QWORD *v36; // [rsp+8h] [rbp-68h]
  unsigned __int64 v37; // [rsp+10h] [rbp-60h]
  _QWORD *v38; // [rsp+10h] [rbp-60h]
  _QWORD *v39; // [rsp+18h] [rbp-58h]
  _QWORD *v40; // [rsp+18h] [rbp-58h]
  _QWORD *v41; // [rsp+18h] [rbp-58h]
  _QWORD *v42; // [rsp+20h] [rbp-50h]
  unsigned __int64 v43; // [rsp+20h] [rbp-50h]
  unsigned __int64 v44; // [rsp+20h] [rbp-50h]
  _QWORD *v45; // [rsp+20h] [rbp-50h]
  _QWORD *v46; // [rsp+20h] [rbp-50h]
  _QWORD *v47; // [rsp+20h] [rbp-50h]
  unsigned __int64 v48; // [rsp+28h] [rbp-48h]
  _QWORD *v49; // [rsp+28h] [rbp-48h]
  _QWORD *v50; // [rsp+28h] [rbp-48h]
  _QWORD *v51; // [rsp+28h] [rbp-48h]
  _QWORD *v52; // [rsp+28h] [rbp-48h]
  _QWORD *v53; // [rsp+28h] [rbp-48h]
  _QWORD *v54; // [rsp+28h] [rbp-48h]
  _QWORD *v55; // [rsp+30h] [rbp-40h]
  _QWORD *v56; // [rsp+38h] [rbp-38h]

  v55 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v56 = (_QWORD *)v55[3];
      if ( v56 )
      {
        while ( 1 )
        {
          v1 = (_QWORD *)v56[3];
          if ( v1 )
          {
            while ( 1 )
            {
              v2 = (_QWORD *)v1[3];
              if ( v2 )
              {
                while ( 1 )
                {
                  v3 = (_QWORD *)v2[3];
                  if ( v3 )
                  {
                    while ( 1 )
                    {
                      v4 = (_QWORD *)v3[3];
                      if ( v4 )
                      {
                        while ( 1 )
                        {
                          v5 = (_QWORD *)v4[3];
                          if ( v5 )
                          {
                            while ( 1 )
                            {
                              v6 = (_QWORD *)v5[3];
                              if ( v6 )
                              {
                                while ( 1 )
                                {
                                  v7 = v6[3];
                                  if ( v7 )
                                  {
                                    do
                                    {
                                      v42 = v6;
                                      v48 = v7;
                                      sub_2EC3080(*(_QWORD *)(v7 + 24));
                                      v8 = v42;
                                      v9 = v48;
                                      v10 = *(_QWORD *)(v48 + 16);
                                      v11 = *(_QWORD **)(v48 + 40);
                                      v12 = v48 + 40;
                                      if ( v11 != (_QWORD *)(v48 + 40) )
                                      {
                                        do
                                        {
                                          v36 = (_QWORD *)v12;
                                          v37 = v10;
                                          v39 = v8;
                                          v43 = v9;
                                          v49 = (_QWORD *)*v11;
                                          j_j___libc_free_0((unsigned __int64)v11);
                                          v11 = v49;
                                          v12 = (unsigned __int64)v36;
                                          v9 = v43;
                                          v8 = v39;
                                          v10 = v37;
                                        }
                                        while ( v49 != v36 );
                                      }
                                      v44 = v10;
                                      v50 = v8;
                                      j_j___libc_free_0(v9);
                                      v7 = v44;
                                      v6 = v50;
                                    }
                                    while ( v44 );
                                  }
                                  v13 = v6 + 5;
                                  v51 = (_QWORD *)v6[2];
                                  v14 = (_QWORD *)v6[5];
                                  if ( v14 != v6 + 5 )
                                  {
                                    do
                                    {
                                      v38 = v13;
                                      v40 = v6;
                                      v45 = (_QWORD *)*v14;
                                      j_j___libc_free_0((unsigned __int64)v14);
                                      v14 = v45;
                                      v13 = v38;
                                      v6 = v40;
                                    }
                                    while ( v45 != v38 );
                                  }
                                  j_j___libc_free_0((unsigned __int64)v6);
                                  if ( !v51 )
                                    break;
                                  v6 = v51;
                                }
                              }
                              v25 = (_QWORD *)v5[5];
                              v54 = (_QWORD *)v5[2];
                              v26 = v5 + 5;
                              if ( v25 != v5 + 5 )
                              {
                                while ( 1 )
                                {
                                  v41 = v26;
                                  v47 = (_QWORD *)*v25;
                                  j_j___libc_free_0((unsigned __int64)v25);
                                  v26 = v41;
                                  if ( v47 == v41 )
                                    break;
                                  v25 = v47;
                                }
                              }
                              j_j___libc_free_0((unsigned __int64)v5);
                              if ( !v54 )
                                break;
                              v5 = v54;
                            }
                          }
                          v19 = (_QWORD *)v4[5];
                          v20 = v4 + 5;
                          v21 = (_QWORD *)v4[2];
                          if ( v19 != v4 + 5 )
                          {
                            do
                            {
                              v46 = v20;
                              v53 = (_QWORD *)*v19;
                              j_j___libc_free_0((unsigned __int64)v19);
                              v19 = v53;
                              v20 = v46;
                            }
                            while ( v53 != v46 );
                          }
                          j_j___libc_free_0((unsigned __int64)v4);
                          if ( !v21 )
                            break;
                          v4 = v21;
                        }
                      }
                      v15 = (_QWORD *)v3[5];
                      v16 = v3 + 5;
                      v17 = (_QWORD *)v3[2];
                      if ( v15 != v3 + 5 )
                      {
                        do
                        {
                          v18 = (unsigned __int64)v15;
                          v52 = v16;
                          v15 = (_QWORD *)*v15;
                          j_j___libc_free_0(v18);
                          v16 = v52;
                        }
                        while ( v15 != v52 );
                      }
                      j_j___libc_free_0((unsigned __int64)v3);
                      if ( !v17 )
                        break;
                      v3 = v17;
                    }
                  }
                  v22 = (_QWORD *)v2[5];
                  v23 = (_QWORD *)v2[2];
                  while ( v22 != v2 + 5 )
                  {
                    v24 = (unsigned __int64)v22;
                    v22 = (_QWORD *)*v22;
                    j_j___libc_free_0(v24);
                  }
                  j_j___libc_free_0((unsigned __int64)v2);
                  if ( !v23 )
                    break;
                  v2 = v23;
                }
              }
              v27 = (_QWORD *)v1[5];
              v28 = (_QWORD *)v1[2];
              while ( v27 != v1 + 5 )
              {
                v29 = (unsigned __int64)v27;
                v27 = (_QWORD *)*v27;
                j_j___libc_free_0(v29);
              }
              j_j___libc_free_0((unsigned __int64)v1);
              if ( !v28 )
                break;
              v1 = v28;
            }
          }
          v30 = (_QWORD *)v56[5];
          v31 = (_QWORD *)v56[2];
          while ( v30 != v56 + 5 )
          {
            v32 = (unsigned __int64)v30;
            v30 = (_QWORD *)*v30;
            j_j___libc_free_0(v32);
          }
          j_j___libc_free_0((unsigned __int64)v56);
          if ( !v31 )
            break;
          v56 = v31;
        }
      }
      v33 = (_QWORD *)v55[5];
      v34 = (_QWORD *)v55[2];
      while ( v33 != v55 + 5 )
      {
        v35 = (unsigned __int64)v33;
        v33 = (_QWORD *)*v33;
        j_j___libc_free_0(v35);
      }
      j_j___libc_free_0((unsigned __int64)v55);
      if ( !v34 )
        break;
      v55 = v34;
    }
  }
}
