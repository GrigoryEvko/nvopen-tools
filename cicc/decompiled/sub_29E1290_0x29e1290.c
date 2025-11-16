// Function: sub_29E1290
// Address: 0x29e1290
//
void __fastcall sub_29E1290(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  _QWORD *v5; // r15
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  _QWORD *v8; // r12
  __int64 v9; // r15
  _QWORD *v10; // rbx
  __int64 v11; // r13
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v40; // [rsp+8h] [rbp-D8h]
  _QWORD *v41; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v43; // [rsp+20h] [rbp-C0h]
  __int64 v44; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v45; // [rsp+30h] [rbp-B0h]
  __int64 v46; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v47; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v48; // [rsp+48h] [rbp-98h]
  __int64 v49; // [rsp+50h] [rbp-90h]
  unsigned __int64 v50; // [rsp+58h] [rbp-88h]
  __int64 v51; // [rsp+60h] [rbp-80h]
  unsigned __int64 v52; // [rsp+68h] [rbp-78h]
  _QWORD *v53; // [rsp+70h] [rbp-70h]
  unsigned __int64 v54; // [rsp+78h] [rbp-68h]
  _QWORD *v55; // [rsp+80h] [rbp-60h]
  _QWORD *v56; // [rsp+88h] [rbp-58h]
  unsigned __int64 i; // [rsp+90h] [rbp-50h]
  __int64 v58; // [rsp+98h] [rbp-48h]
  __int64 v59; // [rsp+A0h] [rbp-40h]
  _QWORD *v60; // [rsp+A8h] [rbp-38h]

  if ( a1 )
  {
    v49 = a1[7];
    v41 = a1 + 5;
    if ( (_QWORD *)v49 != a1 + 5 )
    {
      do
      {
        v1 = *(_QWORD *)(v49 + 40);
        v48 = v1;
        if ( v1 )
        {
          v40 = v1 + 40;
          v46 = *(_QWORD *)(v1 + 56);
          if ( v46 != v1 + 40 )
          {
            do
            {
              v2 = *(_QWORD *)(v46 + 40);
              v47 = v2;
              if ( v2 )
              {
                v39 = v2 + 40;
                v44 = *(_QWORD *)(v2 + 56);
                if ( v44 != v2 + 40 )
                {
                  do
                  {
                    v3 = *(_QWORD *)(v44 + 40);
                    v45 = v3;
                    if ( v3 )
                    {
                      v43 = v3 + 40;
                      v51 = *(_QWORD *)(v3 + 56);
                      if ( v51 != v3 + 40 )
                      {
                        do
                        {
                          v4 = *(_QWORD *)(v51 + 40);
                          v52 = v4;
                          if ( v4 )
                          {
                            v54 = v4 + 40;
                            v59 = *(_QWORD *)(v4 + 56);
                            if ( v59 != v4 + 40 )
                            {
                              do
                              {
                                v5 = *(_QWORD **)(v59 + 40);
                                if ( v5 )
                                {
                                  v55 = v5 + 5;
                                  v58 = v5[7];
                                  if ( (_QWORD *)v58 != v5 + 5 )
                                  {
                                    v53 = *(_QWORD **)(v59 + 40);
                                    do
                                    {
                                      v6 = *(_QWORD *)(v58 + 40);
                                      v50 = v6;
                                      if ( v6 )
                                      {
                                        v7 = *(_QWORD *)(v6 + 56);
                                        for ( i = v6 + 40; i != v7; v7 = sub_220EEE0(v7) )
                                        {
                                          v8 = *(_QWORD **)(v7 + 40);
                                          if ( v8 )
                                          {
                                            v9 = v8[7];
                                            v60 = v8 + 5;
                                            if ( (_QWORD *)v9 != v8 + 5 )
                                            {
                                              v56 = *(_QWORD **)(v7 + 40);
                                              do
                                              {
                                                v10 = *(_QWORD **)(v9 + 40);
                                                if ( v10 )
                                                {
                                                  v11 = v10[7];
                                                  if ( (_QWORD *)v11 != v10 + 5 )
                                                  {
                                                    do
                                                    {
                                                      sub_29E1290(*(_QWORD *)(v11 + 40));
                                                      v11 = sub_220EEE0(v11);
                                                    }
                                                    while ( v10 + 5 != (_QWORD *)v11 );
                                                  }
                                                  v12 = v10[6];
                                                  if ( v12 )
                                                  {
                                                    while ( 1 )
                                                    {
                                                      sub_29E06B0(*(_QWORD *)(v12 + 24));
                                                      v13 = *(_QWORD *)(v12 + 16);
                                                      j_j___libc_free_0(v12);
                                                      if ( !v13 )
                                                        break;
                                                      v12 = v13;
                                                    }
                                                  }
                                                  v14 = v10[1];
                                                  if ( v14 )
                                                    j_j___libc_free_0(v14);
                                                  j_j___libc_free_0((unsigned __int64)v10);
                                                }
                                                v9 = sub_220EEE0(v9);
                                              }
                                              while ( v60 != (_QWORD *)v9 );
                                              v8 = v56;
                                            }
                                            v15 = v8[6];
                                            if ( v15 )
                                            {
                                              while ( 1 )
                                              {
                                                sub_29E06B0(*(_QWORD *)(v15 + 24));
                                                v16 = *(_QWORD *)(v15 + 16);
                                                j_j___libc_free_0(v15);
                                                if ( !v16 )
                                                  break;
                                                v15 = v16;
                                              }
                                            }
                                            v17 = v8[1];
                                            if ( v17 )
                                              j_j___libc_free_0(v17);
                                            j_j___libc_free_0((unsigned __int64)v8);
                                          }
                                        }
                                        v18 = *(_QWORD *)(v50 + 48);
                                        while ( v18 )
                                        {
                                          sub_29E06B0(*(_QWORD *)(v18 + 24));
                                          v19 = v18;
                                          v18 = *(_QWORD *)(v18 + 16);
                                          j_j___libc_free_0(v19);
                                        }
                                        v20 = *(_QWORD *)(v50 + 8);
                                        if ( v20 )
                                          j_j___libc_free_0(v20);
                                        j_j___libc_free_0(v50);
                                      }
                                      v58 = sub_220EEE0(v58);
                                    }
                                    while ( v55 != (_QWORD *)v58 );
                                    v5 = v53;
                                  }
                                  v21 = v5[6];
                                  while ( v21 )
                                  {
                                    sub_29E06B0(*(_QWORD *)(v21 + 24));
                                    v22 = v21;
                                    v21 = *(_QWORD *)(v21 + 16);
                                    j_j___libc_free_0(v22);
                                  }
                                  v23 = v5[1];
                                  if ( v23 )
                                    j_j___libc_free_0(v23);
                                  j_j___libc_free_0((unsigned __int64)v5);
                                }
                                v59 = sub_220EEE0(v59);
                              }
                              while ( v54 != v59 );
                            }
                            v24 = *(_QWORD *)(v52 + 48);
                            while ( v24 )
                            {
                              sub_29E06B0(*(_QWORD *)(v24 + 24));
                              v25 = v24;
                              v24 = *(_QWORD *)(v24 + 16);
                              j_j___libc_free_0(v25);
                            }
                            v26 = *(_QWORD *)(v52 + 8);
                            if ( v26 )
                              j_j___libc_free_0(v26);
                            j_j___libc_free_0(v52);
                          }
                          v51 = sub_220EEE0(v51);
                        }
                        while ( v43 != v51 );
                      }
                      v27 = *(_QWORD *)(v45 + 48);
                      while ( v27 )
                      {
                        sub_29E06B0(*(_QWORD *)(v27 + 24));
                        v28 = v27;
                        v27 = *(_QWORD *)(v27 + 16);
                        j_j___libc_free_0(v28);
                      }
                      v29 = *(_QWORD *)(v45 + 8);
                      if ( v29 )
                        j_j___libc_free_0(v29);
                      j_j___libc_free_0(v45);
                    }
                    v44 = sub_220EEE0(v44);
                  }
                  while ( v39 != v44 );
                }
                v30 = *(_QWORD *)(v47 + 48);
                while ( v30 )
                {
                  sub_29E06B0(*(_QWORD *)(v30 + 24));
                  v31 = v30;
                  v30 = *(_QWORD *)(v30 + 16);
                  j_j___libc_free_0(v31);
                }
                v32 = *(_QWORD *)(v47 + 8);
                if ( v32 )
                  j_j___libc_free_0(v32);
                j_j___libc_free_0(v47);
              }
              v46 = sub_220EEE0(v46);
            }
            while ( v40 != v46 );
          }
          v33 = *(_QWORD *)(v48 + 48);
          while ( v33 )
          {
            sub_29E06B0(*(_QWORD *)(v33 + 24));
            v34 = v33;
            v33 = *(_QWORD *)(v33 + 16);
            j_j___libc_free_0(v34);
          }
          v35 = *(_QWORD *)(v48 + 8);
          if ( v35 )
            j_j___libc_free_0(v35);
          j_j___libc_free_0(v48);
        }
        v49 = sub_220EEE0(v49);
      }
      while ( v41 != (_QWORD *)v49 );
    }
    v36 = a1[6];
    while ( v36 )
    {
      sub_29E06B0(*(_QWORD *)(v36 + 24));
      v37 = v36;
      v36 = *(_QWORD *)(v36 + 16);
      j_j___libc_free_0(v37);
    }
    v38 = a1[1];
    if ( v38 )
      j_j___libc_free_0(v38);
    j_j___libc_free_0((unsigned __int64)a1);
  }
}
