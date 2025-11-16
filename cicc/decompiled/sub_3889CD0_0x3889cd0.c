// Function: sub_3889CD0
// Address: 0x3889cd0
//
void __fastcall sub_3889CD0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rbx
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 i; // r14
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // r12
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 j; // r14
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rbx
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  __int64 v53; // r12
  __int64 v54; // rax
  __int64 k; // r13
  __int64 v56; // r15
  __int64 v57; // rax
  __int64 v58; // rdx
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rdi
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 m; // r13
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // r12
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  unsigned __int64 v71; // rdi
  __int64 v72; // rbx
  __int64 v73; // rax
  __int64 n; // r13
  __int64 v75; // [rsp+0h] [rbp-70h]
  unsigned __int64 v76; // [rsp+8h] [rbp-68h]
  __int64 v77; // [rsp+8h] [rbp-68h]
  _QWORD *v78; // [rsp+10h] [rbp-60h]
  unsigned __int64 v79; // [rsp+10h] [rbp-60h]
  __int64 v80; // [rsp+10h] [rbp-60h]
  unsigned __int64 v81; // [rsp+18h] [rbp-58h]
  __int64 v82; // [rsp+18h] [rbp-58h]
  __int64 v83; // [rsp+18h] [rbp-58h]
  __int64 v84; // [rsp+18h] [rbp-58h]
  unsigned __int64 v85; // [rsp+20h] [rbp-50h]
  unsigned __int64 v86; // [rsp+28h] [rbp-48h]
  unsigned __int64 v87; // [rsp+30h] [rbp-40h]
  void *v88; // [rsp+38h] [rbp-38h]

  v85 = a1;
  if ( a1 )
  {
    v88 = sub_16982C0();
    while ( 1 )
    {
      v87 = *(_QWORD *)(v85 + 24);
      if ( v87 )
      {
        while ( 1 )
        {
          v86 = *(_QWORD *)(v87 + 24);
          if ( v86 )
          {
            while ( 1 )
            {
              v1 = *(_QWORD *)(v86 + 24);
              if ( v1 )
              {
                while ( 1 )
                {
                  v2 = *(_QWORD *)(v1 + 24);
                  if ( v2 )
                  {
                    while ( 1 )
                    {
                      v3 = *(_QWORD *)(v2 + 24);
                      if ( v3 )
                      {
                        while ( 1 )
                        {
                          v4 = *(_QWORD *)(v3 + 24);
                          if ( v4 )
                          {
                            do
                            {
                              v5 = *(_QWORD *)(v4 + 24);
                              if ( v5 )
                              {
                                do
                                {
                                  v78 = *(_QWORD **)(v5 + 24);
                                  while ( v78 )
                                  {
                                    sub_3889CD0(v78[3]);
                                    v81 = (unsigned __int64)v78;
                                    v6 = v78[23];
                                    v78 = (_QWORD *)v78[2];
                                    if ( v6 )
                                      j_j___libc_free_0_0(v6);
                                    if ( *(void **)(v81 + 152) == v88 )
                                    {
                                      v31 = *(_QWORD *)(v81 + 160);
                                      if ( v31 )
                                      {
                                        v32 = 32LL * *(_QWORD *)(v31 - 8);
                                        v33 = v31 + v32;
                                        if ( v31 != v31 + v32 )
                                        {
                                          do
                                          {
                                            v75 = v31;
                                            v77 = v33 - 32;
                                            sub_127D120((_QWORD *)(v33 - 24));
                                            v33 = v77;
                                            v31 = v75;
                                          }
                                          while ( v75 != v77 );
                                        }
                                        j_j_j___libc_free_0_0(v31 - 8);
                                      }
                                    }
                                    else
                                    {
                                      sub_1698460(v81 + 152);
                                    }
                                    if ( *(_DWORD *)(v81 + 136) > 0x40u )
                                    {
                                      v7 = *(_QWORD *)(v81 + 128);
                                      if ( v7 )
                                        j_j___libc_free_0_0(v7);
                                    }
                                    v8 = *(_QWORD *)(v81 + 96);
                                    if ( v8 != v81 + 112 )
                                      j_j___libc_free_0(v8);
                                    v9 = *(_QWORD *)(v81 + 64);
                                    if ( v9 != v81 + 80 )
                                      j_j___libc_free_0(v9);
                                    j_j___libc_free_0(v81);
                                  }
                                  v10 = *(_QWORD *)(v5 + 184);
                                  v76 = *(_QWORD *)(v5 + 16);
                                  if ( v10 )
                                    j_j___libc_free_0_0(v10);
                                  if ( *(void **)(v5 + 152) == v88 )
                                  {
                                    v28 = *(_QWORD *)(v5 + 160);
                                    if ( v28 )
                                    {
                                      v29 = 32LL * *(_QWORD *)(v28 - 8);
                                      v30 = v28 + v29;
                                      if ( v28 != v28 + v29 )
                                      {
                                        do
                                        {
                                          v80 = v28;
                                          v82 = v30 - 32;
                                          sub_127D120((_QWORD *)(v30 - 24));
                                          v30 = v82;
                                          v28 = v80;
                                        }
                                        while ( v80 != v82 );
                                      }
                                      j_j_j___libc_free_0_0(v28 - 8);
                                    }
                                  }
                                  else
                                  {
                                    sub_1698460(v5 + 152);
                                  }
                                  if ( *(_DWORD *)(v5 + 136) > 0x40u )
                                  {
                                    v11 = *(_QWORD *)(v5 + 128);
                                    if ( v11 )
                                      j_j___libc_free_0_0(v11);
                                  }
                                  v12 = *(_QWORD *)(v5 + 96);
                                  if ( v12 != v5 + 112 )
                                    j_j___libc_free_0(v12);
                                  v13 = *(_QWORD *)(v5 + 64);
                                  if ( v13 != v5 + 80 )
                                    j_j___libc_free_0(v13);
                                  j_j___libc_free_0(v5);
                                  v5 = v76;
                                }
                                while ( v76 );
                              }
                              v14 = *(_QWORD *)(v4 + 184);
                              v79 = *(_QWORD *)(v4 + 16);
                              if ( v14 )
                                j_j___libc_free_0_0(v14);
                              if ( *(void **)(v4 + 152) == v88 )
                              {
                                v34 = *(_QWORD *)(v4 + 160);
                                if ( v34 )
                                {
                                  v35 = 32LL * *(_QWORD *)(v34 - 8);
                                  v36 = v34 + v35;
                                  if ( v34 != v34 + v35 )
                                  {
                                    do
                                    {
                                      v83 = v36 - 32;
                                      sub_127D120((_QWORD *)(v36 - 24));
                                      v36 = v83;
                                    }
                                    while ( v34 != v83 );
                                  }
                                  j_j_j___libc_free_0_0(v34 - 8);
                                }
                              }
                              else
                              {
                                sub_1698460(v4 + 152);
                              }
                              if ( *(_DWORD *)(v4 + 136) > 0x40u )
                              {
                                v15 = *(_QWORD *)(v4 + 128);
                                if ( v15 )
                                  j_j___libc_free_0_0(v15);
                              }
                              v16 = *(_QWORD *)(v4 + 96);
                              if ( v16 != v4 + 112 )
                                j_j___libc_free_0(v16);
                              v17 = *(_QWORD *)(v4 + 64);
                              if ( v17 != v4 + 80 )
                                j_j___libc_free_0(v17);
                              j_j___libc_free_0(v4);
                              v4 = v79;
                            }
                            while ( v79 );
                          }
                          v18 = *(_QWORD *)(v3 + 184);
                          v19 = *(_QWORD *)(v3 + 16);
                          if ( v18 )
                            j_j___libc_free_0_0(v18);
                          if ( *(void **)(v3 + 152) == v88 )
                          {
                            v56 = *(_QWORD *)(v3 + 160);
                            if ( v56 )
                            {
                              v57 = 32LL * *(_QWORD *)(v56 - 8);
                              v58 = v56 + v57;
                              if ( v56 != v56 + v57 )
                              {
                                do
                                {
                                  v84 = v58 - 32;
                                  sub_127D120((_QWORD *)(v58 - 24));
                                  v58 = v84;
                                }
                                while ( v56 != v84 );
                              }
                              j_j_j___libc_free_0_0(v56 - 8);
                            }
                          }
                          else
                          {
                            sub_1698460(v3 + 152);
                          }
                          if ( *(_DWORD *)(v3 + 136) > 0x40u )
                          {
                            v20 = *(_QWORD *)(v3 + 128);
                            if ( v20 )
                              j_j___libc_free_0_0(v20);
                          }
                          v21 = *(_QWORD *)(v3 + 96);
                          if ( v21 != v3 + 112 )
                            j_j___libc_free_0(v21);
                          v22 = *(_QWORD *)(v3 + 64);
                          if ( v22 != v3 + 80 )
                            j_j___libc_free_0(v22);
                          j_j___libc_free_0(v3);
                          if ( !v19 )
                            break;
                          v3 = v19;
                        }
                      }
                      v23 = *(_QWORD *)(v2 + 184);
                      v24 = *(_QWORD *)(v2 + 16);
                      if ( v23 )
                        j_j___libc_free_0_0(v23);
                      if ( *(void **)(v2 + 152) == v88 )
                      {
                        v37 = *(_QWORD *)(v2 + 160);
                        if ( v37 )
                        {
                          v38 = 32LL * *(_QWORD *)(v37 - 8);
                          for ( i = v37 + v38; v37 != i; sub_127D120((_QWORD *)(i + 8)) )
                            i -= 32;
                          j_j_j___libc_free_0_0(v37 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v2 + 152);
                      }
                      if ( *(_DWORD *)(v2 + 136) > 0x40u )
                      {
                        v25 = *(_QWORD *)(v2 + 128);
                        if ( v25 )
                          j_j___libc_free_0_0(v25);
                      }
                      v26 = *(_QWORD *)(v2 + 96);
                      if ( v26 != v2 + 112 )
                        j_j___libc_free_0(v26);
                      v27 = *(_QWORD *)(v2 + 64);
                      if ( v27 != v2 + 80 )
                        j_j___libc_free_0(v27);
                      j_j___libc_free_0(v2);
                      if ( !v24 )
                        break;
                      v2 = v24;
                    }
                  }
                  v40 = *(_QWORD *)(v1 + 184);
                  v41 = *(_QWORD *)(v1 + 16);
                  if ( v40 )
                    j_j___libc_free_0_0(v40);
                  if ( *(void **)(v1 + 152) == v88 )
                  {
                    v45 = *(_QWORD *)(v1 + 160);
                    if ( v45 )
                    {
                      v46 = 32LL * *(_QWORD *)(v45 - 8);
                      for ( j = v45 + v46; v45 != j; sub_127D120((_QWORD *)(j + 8)) )
                        j -= 32;
                      j_j_j___libc_free_0_0(v45 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v1 + 152);
                  }
                  if ( *(_DWORD *)(v1 + 136) > 0x40u )
                  {
                    v42 = *(_QWORD *)(v1 + 128);
                    if ( v42 )
                      j_j___libc_free_0_0(v42);
                  }
                  v43 = *(_QWORD *)(v1 + 96);
                  if ( v43 != v1 + 112 )
                    j_j___libc_free_0(v43);
                  v44 = *(_QWORD *)(v1 + 64);
                  if ( v44 != v1 + 80 )
                    j_j___libc_free_0(v44);
                  j_j___libc_free_0(v1);
                  if ( !v41 )
                    break;
                  v1 = v41;
                }
              }
              v48 = *(_QWORD *)(v86 + 184);
              v49 = *(_QWORD *)(v86 + 16);
              if ( v48 )
                j_j___libc_free_0_0(v48);
              if ( *(void **)(v86 + 152) == v88 )
              {
                v53 = *(_QWORD *)(v86 + 160);
                if ( v53 )
                {
                  v54 = 32LL * *(_QWORD *)(v53 - 8);
                  for ( k = v53 + v54; v53 != k; sub_127D120((_QWORD *)(k + 8)) )
                    k -= 32;
                  j_j_j___libc_free_0_0(v53 - 8);
                }
              }
              else
              {
                sub_1698460(v86 + 152);
              }
              if ( *(_DWORD *)(v86 + 136) > 0x40u )
              {
                v50 = *(_QWORD *)(v86 + 128);
                if ( v50 )
                  j_j___libc_free_0_0(v50);
              }
              v51 = *(_QWORD *)(v86 + 96);
              if ( v51 != v86 + 112 )
                j_j___libc_free_0(v51);
              v52 = *(_QWORD *)(v86 + 64);
              if ( v52 != v86 + 80 )
                j_j___libc_free_0(v52);
              j_j___libc_free_0(v86);
              if ( !v49 )
                break;
              v86 = v49;
            }
          }
          v59 = *(_QWORD *)(v87 + 184);
          v60 = *(_QWORD *)(v87 + 16);
          if ( v59 )
            j_j___libc_free_0_0(v59);
          if ( *(void **)(v87 + 152) == v88 )
          {
            v64 = *(_QWORD *)(v87 + 160);
            if ( v64 )
            {
              v65 = 32LL * *(_QWORD *)(v64 - 8);
              for ( m = v64 + v65; v64 != m; sub_127D120((_QWORD *)(m + 8)) )
                m -= 32;
              j_j_j___libc_free_0_0(v64 - 8);
            }
          }
          else
          {
            sub_1698460(v87 + 152);
          }
          if ( *(_DWORD *)(v87 + 136) > 0x40u )
          {
            v61 = *(_QWORD *)(v87 + 128);
            if ( v61 )
              j_j___libc_free_0_0(v61);
          }
          v62 = *(_QWORD *)(v87 + 96);
          if ( v62 != v87 + 112 )
            j_j___libc_free_0(v62);
          v63 = *(_QWORD *)(v87 + 64);
          if ( v63 != v87 + 80 )
            j_j___libc_free_0(v63);
          j_j___libc_free_0(v87);
          if ( !v60 )
            break;
          v87 = v60;
        }
      }
      v67 = *(_QWORD *)(v85 + 184);
      v68 = *(_QWORD *)(v85 + 16);
      if ( v67 )
        j_j___libc_free_0_0(v67);
      if ( *(void **)(v85 + 152) == v88 )
      {
        v72 = *(_QWORD *)(v85 + 160);
        if ( v72 )
        {
          v73 = 32LL * *(_QWORD *)(v72 - 8);
          for ( n = v72 + v73; v72 != n; sub_127D120((_QWORD *)(n + 8)) )
            n -= 32;
          j_j_j___libc_free_0_0(v72 - 8);
        }
      }
      else
      {
        sub_1698460(v85 + 152);
      }
      if ( *(_DWORD *)(v85 + 136) > 0x40u )
      {
        v69 = *(_QWORD *)(v85 + 128);
        if ( v69 )
          j_j___libc_free_0_0(v69);
      }
      v70 = *(_QWORD *)(v85 + 96);
      if ( v70 != v85 + 112 )
        j_j___libc_free_0(v70);
      v71 = *(_QWORD *)(v85 + 64);
      if ( v71 != v85 + 80 )
        j_j___libc_free_0(v71);
      j_j___libc_free_0(v85);
      if ( !v68 )
        break;
      v85 = v68;
    }
  }
}
