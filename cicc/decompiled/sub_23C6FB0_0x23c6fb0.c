// Function: sub_23C6FB0
// Address: 0x23c6fb0
//
void __fastcall sub_23C6FB0(__int64 a1)
{
  int v2; // r9d
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r12
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r15
  __int64 v22; // r13
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rax
  unsigned __int64 *v25; // rbx
  unsigned __int64 *v26; // r12
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // rbx
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // r15
  __int64 v35; // r14
  unsigned __int64 v36; // rdi
  __int64 v38; // [rsp+8h] [rbp-A8h]
  __int64 v39; // [rsp+10h] [rbp-A0h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+30h] [rbp-80h]
  __int64 v44; // [rsp+38h] [rbp-78h]
  __int64 v45; // [rsp+40h] [rbp-70h]
  __int64 v46; // [rsp+48h] [rbp-68h]
  __int64 v47; // [rsp+50h] [rbp-60h]
  __int64 v48; // [rsp+58h] [rbp-58h]
  __int64 v49; // [rsp+58h] [rbp-58h]
  __int64 v50; // [rsp+60h] [rbp-50h]
  __int64 v51; // [rsp+68h] [rbp-48h]
  __int64 v52; // [rsp+68h] [rbp-48h]
  __int64 v53; // [rsp+70h] [rbp-40h]
  unsigned __int64 v54; // [rsp+70h] [rbp-40h]
  __int64 v55; // [rsp+78h] [rbp-38h]
  __int64 v56; // [rsp+78h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 12);
  v3 = *(_QWORD *)a1;
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v4 )
    {
      v50 = 0;
      v40 = 8 * v4;
      do
      {
        v5 = *(_QWORD *)(v3 + v50);
        v44 = v5;
        if ( v5 != -8 && v5 )
        {
          v6 = *(_QWORD *)(v5 + 16);
          v38 = *(_QWORD *)v5 + 41LL;
          if ( *(_DWORD *)(v5 + 28) )
          {
            v7 = *(unsigned int *)(v5 + 24);
            if ( (_DWORD)v7 )
            {
              v51 = 0;
              v41 = 8 * v7;
              do
              {
                v8 = *(_QWORD *)(v6 + v51);
                v47 = v8;
                if ( v8 != -8 && v8 )
                {
                  v9 = *(_QWORD *)(v8 + 8);
                  v39 = *(_QWORD *)v8 + 33LL;
                  if ( *(_DWORD *)(v8 + 20) )
                  {
                    v10 = *(unsigned int *)(v8 + 16);
                    if ( (_DWORD)v10 )
                    {
                      v53 = 0;
                      v45 = 8 * v10;
                      do
                      {
                        v11 = *(_QWORD *)(v9 + v53);
                        if ( v11 && v11 != -8 )
                        {
                          v12 = *(unsigned __int64 **)(v11 + 40);
                          v13 = *(unsigned __int64 **)(v11 + 32);
                          v42 = *(_QWORD *)v11 + 57LL;
                          if ( v12 != v13 )
                          {
                            do
                            {
                              v14 = *v13;
                              if ( *v13 )
                              {
                                sub_C88FF0((_QWORD *)*v13);
                                j_j___libc_free_0(v14);
                              }
                              v13 += 2;
                            }
                            while ( v12 != v13 );
                            v13 = *(unsigned __int64 **)(v11 + 32);
                          }
                          if ( v13 )
                            j_j___libc_free_0((unsigned __int64)v13);
                          v15 = *(_QWORD *)(v11 + 8);
                          if ( *(_DWORD *)(v11 + 20) )
                          {
                            v16 = *(unsigned int *)(v11 + 16);
                            if ( (_DWORD)v16 )
                            {
                              v55 = 0;
                              v48 = 8 * v16;
                              v43 = v11;
                              do
                              {
                                v17 = *(_QWORD *)(v15 + v55);
                                if ( v17 != -8 && v17 )
                                {
                                  v18 = *(_QWORD *)(v17 + 24);
                                  v46 = *(_QWORD *)v17 + 89LL;
                                  v19 = v18 + 40LL * *(unsigned int *)(v17 + 32);
                                  if ( v18 != v19 )
                                  {
                                    do
                                    {
                                      v19 -= 40LL;
                                      v20 = *(_QWORD *)(v19 + 16);
                                      if ( v20 != v19 + 40 )
                                        _libc_free(v20);
                                      v21 = *(_QWORD *)v19;
                                      v22 = *(_QWORD *)v19 + 80LL * *(unsigned int *)(v19 + 8);
                                      if ( *(_QWORD *)v19 != v22 )
                                      {
                                        do
                                        {
                                          v22 -= 80;
                                          v23 = *(_QWORD *)(v22 + 8);
                                          if ( v23 != v22 + 24 )
                                            _libc_free(v23);
                                        }
                                        while ( v21 != v22 );
                                        v21 = *(_QWORD *)v19;
                                      }
                                      if ( v21 != v19 + 16 )
                                        _libc_free(v21);
                                    }
                                    while ( v18 != v19 );
                                    v19 = *(_QWORD *)(v17 + 24);
                                  }
                                  if ( v19 != v17 + 40 )
                                    _libc_free(v19);
                                  sub_C7D6A0(v17, v46, 8);
                                  v15 = *(_QWORD *)(v43 + 8);
                                }
                                v55 += 8;
                              }
                              while ( v48 != v55 );
                              v11 = v43;
                            }
                          }
                          _libc_free(v15);
                          sub_C7D6A0(v11, v42, 8);
                          v9 = *(_QWORD *)(v47 + 8);
                        }
                        v53 += 8;
                      }
                      while ( v45 != v53 );
                    }
                  }
                  _libc_free(v9);
                  sub_C7D6A0(v47, v39, 8);
                  v6 = *(_QWORD *)(v44 + 16);
                }
                v51 += 8;
              }
              while ( v41 != v51 );
            }
          }
          _libc_free(v6);
          v24 = *(_QWORD *)(v44 + 8);
          v54 = v24;
          if ( v24 )
          {
            v25 = *(unsigned __int64 **)(v24 + 32);
            v26 = *(unsigned __int64 **)(v24 + 24);
            if ( v25 != v26 )
            {
              do
              {
                v27 = *v26;
                if ( *v26 )
                {
                  sub_C88FF0((_QWORD *)*v26);
                  j_j___libc_free_0(v27);
                }
                v26 += 2;
              }
              while ( v25 != v26 );
              v26 = *(unsigned __int64 **)(v54 + 24);
            }
            if ( v26 )
              j_j___libc_free_0((unsigned __int64)v26);
            v28 = *(_QWORD *)v54;
            if ( *(_DWORD *)(v54 + 12) )
            {
              v29 = *(unsigned int *)(v54 + 8);
              if ( (_DWORD)v29 )
              {
                v56 = 0;
                v52 = 8 * v29;
                do
                {
                  v30 = *(_QWORD *)(v28 + v56);
                  if ( v30 != -8 && v30 )
                  {
                    v31 = *(_QWORD *)(v30 + 24);
                    v49 = *(_QWORD *)v30 + 89LL;
                    v32 = v31 + 40LL * *(unsigned int *)(v30 + 32);
                    if ( v31 != v32 )
                    {
                      do
                      {
                        v32 -= 40LL;
                        v33 = *(_QWORD *)(v32 + 16);
                        if ( v33 != v32 + 40 )
                          _libc_free(v33);
                        v34 = *(_QWORD *)v32;
                        v35 = *(_QWORD *)v32 + 80LL * *(unsigned int *)(v32 + 8);
                        if ( *(_QWORD *)v32 != v35 )
                        {
                          do
                          {
                            v35 -= 80;
                            v36 = *(_QWORD *)(v35 + 8);
                            if ( v36 != v35 + 24 )
                              _libc_free(v36);
                          }
                          while ( v34 != v35 );
                          v34 = *(_QWORD *)v32;
                        }
                        if ( v34 != v32 + 16 )
                          _libc_free(v34);
                      }
                      while ( v31 != v32 );
                      v32 = *(_QWORD *)(v30 + 24);
                    }
                    if ( v32 != v30 + 40 )
                      _libc_free(v32);
                    sub_C7D6A0(v30, v49, 8);
                    v28 = *(_QWORD *)v54;
                  }
                  v56 += 8;
                }
                while ( v52 != v56 );
              }
            }
            _libc_free(v28);
            j_j___libc_free_0(v54);
          }
          sub_C7D6A0(v44, v38, 8);
          v3 = *(_QWORD *)a1;
        }
        v50 += 8;
      }
      while ( v40 != v50 );
    }
  }
  _libc_free(v3);
}
