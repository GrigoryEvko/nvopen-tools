// Function: sub_1209010
// Address: 0x1209010
//
void __fastcall sub_1209010(__int64 a1)
{
  void *v1; // rbx
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rdi
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  void **v27; // rcx
  void **v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdi
  void **v34; // rcx
  void **v35; // r12
  __int64 v36; // rdx
  __int64 v37; // rax
  void **v38; // r15
  __int64 v39; // rdi
  __int64 v40; // r12
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rax
  void **v46; // r14
  __int64 v47; // rdi
  __int64 v48; // r12
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // rdi
  __int64 v52; // r15
  __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rax
  void **v58; // r14
  __int64 v59; // rdi
  __int64 v60; // r12
  __int64 v61; // rdi
  __int64 v62; // rdi
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // rax
  void **v66; // r14
  __int64 v67; // rdi
  __int64 v68; // r13
  __int64 v69; // rdi
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // rdx
  __int64 v73; // rax
  void **v74; // r12
  __int64 v75; // [rsp+8h] [rbp-68h]
  _QWORD *v76; // [rsp+10h] [rbp-60h]
  __int64 v77; // [rsp+10h] [rbp-60h]
  _QWORD *v78; // [rsp+18h] [rbp-58h]
  __int64 v79; // [rsp+18h] [rbp-58h]
  __int64 v80; // [rsp+18h] [rbp-58h]
  __int64 v81; // [rsp+20h] [rbp-50h]
  __int64 v82; // [rsp+28h] [rbp-48h]
  __int64 v83; // [rsp+30h] [rbp-40h]
  __int64 v84; // [rsp+38h] [rbp-38h]

  v81 = a1;
  if ( a1 )
  {
    v1 = sub_C33340();
    while ( 1 )
    {
      v84 = *(_QWORD *)(v81 + 24);
      if ( v84 )
      {
        while ( 1 )
        {
          v83 = *(_QWORD *)(v84 + 24);
          if ( v83 )
          {
            while ( 1 )
            {
              v82 = *(_QWORD *)(v83 + 24);
              if ( v82 )
              {
                while ( 1 )
                {
                  v2 = *(_QWORD *)(v82 + 24);
                  if ( v2 )
                  {
                    while ( 1 )
                    {
                      v3 = *(_QWORD *)(v2 + 24);
                      if ( v3 )
                      {
                        do
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
                                  v76 = *(_QWORD **)(v5 + 24);
                                  while ( v76 )
                                  {
                                    sub_1209010(v76[3]);
                                    v6 = v76[22];
                                    v78 = v76;
                                    v76 = (_QWORD *)v76[2];
                                    if ( v6 )
                                      j_j___libc_free_0_0(v6);
                                    if ( (void *)v78[18] == v1 )
                                    {
                                      v29 = v78[19];
                                      if ( v29 )
                                      {
                                        v30 = 24LL * *(_QWORD *)(v29 - 8);
                                        v31 = v29 + v30;
                                        if ( v29 != v29 + v30 )
                                        {
                                          while ( 1 )
                                          {
                                            v75 = v31 - 24;
                                            v33 = v31 - 24;
                                            if ( *(void **)(v31 - 24) == v1 )
                                            {
                                              sub_969EE0(v33);
                                              v32 = v75;
                                              if ( v78[19] == v75 )
                                              {
LABEL_86:
                                                v31 = v32;
                                                break;
                                              }
                                            }
                                            else
                                            {
                                              sub_C338F0(v33);
                                              v32 = v75;
                                              if ( v78[19] == v75 )
                                                goto LABEL_86;
                                            }
                                            v31 = v32;
                                          }
                                        }
                                        j_j_j___libc_free_0_0(v31 - 8);
                                      }
                                    }
                                    else
                                    {
                                      sub_C338F0((__int64)(v78 + 18));
                                    }
                                    if ( *((_DWORD *)v78 + 34) > 0x40u )
                                    {
                                      v7 = v78[16];
                                      if ( v7 )
                                        j_j___libc_free_0_0(v7);
                                    }
                                    v8 = (_QWORD *)v78[12];
                                    if ( v8 != v78 + 14 )
                                      j_j___libc_free_0(v8, v78[14] + 1LL);
                                    v9 = (_QWORD *)v78[8];
                                    if ( v9 != v78 + 10 )
                                      j_j___libc_free_0(v9, v78[10] + 1LL);
                                    j_j___libc_free_0(v78, 200);
                                  }
                                  v10 = *(_QWORD *)(v5 + 176);
                                  v79 = *(_QWORD *)(v5 + 16);
                                  if ( v10 )
                                    j_j___libc_free_0_0(v10);
                                  if ( *(void **)(v5 + 144) == v1 )
                                  {
                                    v27 = *(void ***)(v5 + 152);
                                    if ( v27 )
                                    {
                                      v28 = &v27[3 * (_QWORD)*(v27 - 1)];
                                      if ( v27 != v28 )
                                      {
                                        do
                                        {
                                          while ( 1 )
                                          {
                                            v28 -= 3;
                                            if ( *v28 == v1 )
                                              break;
                                            sub_C338F0((__int64)v28);
                                            if ( *(void ***)(v5 + 152) == v28 )
                                              goto LABEL_78;
                                          }
                                          sub_969EE0((__int64)v28);
                                        }
                                        while ( *(void ***)(v5 + 152) != v28 );
                                      }
LABEL_78:
                                      j_j_j___libc_free_0_0(v28 - 1);
                                    }
                                  }
                                  else
                                  {
                                    sub_C338F0(v5 + 144);
                                  }
                                  if ( *(_DWORD *)(v5 + 136) > 0x40u )
                                  {
                                    v11 = *(_QWORD *)(v5 + 128);
                                    if ( v11 )
                                      j_j___libc_free_0_0(v11);
                                  }
                                  v12 = *(_QWORD *)(v5 + 96);
                                  if ( v12 != v5 + 112 )
                                    j_j___libc_free_0(v12, *(_QWORD *)(v5 + 112) + 1LL);
                                  v13 = *(_QWORD *)(v5 + 64);
                                  if ( v13 != v5 + 80 )
                                    j_j___libc_free_0(v13, *(_QWORD *)(v5 + 80) + 1LL);
                                  j_j___libc_free_0(v5, 200);
                                  v5 = v79;
                                }
                                while ( v79 );
                              }
                              v14 = *(_QWORD *)(v4 + 176);
                              v77 = *(_QWORD *)(v4 + 16);
                              if ( v14 )
                                j_j___libc_free_0_0(v14);
                              if ( *(void **)(v4 + 144) == v1 )
                              {
                                v34 = *(void ***)(v4 + 152);
                                if ( v34 )
                                {
                                  v35 = &v34[3 * (_QWORD)*(v34 - 1)];
                                  if ( v34 != v35 )
                                  {
                                    do
                                    {
                                      while ( 1 )
                                      {
                                        v35 -= 3;
                                        if ( *v35 == v1 )
                                          break;
                                        sub_C338F0((__int64)v35);
                                        if ( *(void ***)(v4 + 152) == v35 )
                                          goto LABEL_94;
                                      }
                                      sub_969EE0((__int64)v35);
                                    }
                                    while ( *(void ***)(v4 + 152) != v35 );
                                  }
LABEL_94:
                                  j_j_j___libc_free_0_0(v35 - 1);
                                }
                              }
                              else
                              {
                                sub_C338F0(v4 + 144);
                              }
                              if ( *(_DWORD *)(v4 + 136) > 0x40u )
                              {
                                v15 = *(_QWORD *)(v4 + 128);
                                if ( v15 )
                                  j_j___libc_free_0_0(v15);
                              }
                              v16 = *(_QWORD *)(v4 + 96);
                              if ( v16 != v4 + 112 )
                                j_j___libc_free_0(v16, *(_QWORD *)(v4 + 112) + 1LL);
                              v17 = *(_QWORD *)(v4 + 64);
                              if ( v17 != v4 + 80 )
                                j_j___libc_free_0(v17, *(_QWORD *)(v4 + 80) + 1LL);
                              j_j___libc_free_0(v4, 200);
                              v4 = v77;
                            }
                            while ( v77 );
                          }
                          v18 = *(_QWORD *)(v3 + 176);
                          v80 = *(_QWORD *)(v3 + 16);
                          if ( v18 )
                            j_j___libc_free_0_0(v18);
                          if ( *(void **)(v3 + 144) == v1 )
                          {
                            v52 = *(_QWORD *)(v3 + 152);
                            if ( v52 )
                            {
                              v53 = 24LL * *(_QWORD *)(v52 - 8);
                              v54 = v52 + v53;
                              if ( v52 != v52 + v53 )
                              {
                                while ( 1 )
                                {
                                  v52 = v54 - 24;
                                  v55 = v54 - 24;
                                  if ( *(void **)(v54 - 24) == v1 )
                                  {
                                    sub_969EE0(v55);
                                    if ( *(_QWORD *)(v3 + 152) == v52 )
                                      break;
                                  }
                                  else
                                  {
                                    sub_C338F0(v55);
                                    if ( *(_QWORD *)(v3 + 152) == v52 )
                                      break;
                                  }
                                  v54 -= 24;
                                }
                              }
                              j_j_j___libc_free_0_0(v52 - 8);
                            }
                          }
                          else
                          {
                            sub_C338F0(v3 + 144);
                          }
                          if ( *(_DWORD *)(v3 + 136) > 0x40u )
                          {
                            v19 = *(_QWORD *)(v3 + 128);
                            if ( v19 )
                              j_j___libc_free_0_0(v19);
                          }
                          v20 = *(_QWORD *)(v3 + 96);
                          if ( v20 != v3 + 112 )
                            j_j___libc_free_0(v20, *(_QWORD *)(v3 + 112) + 1LL);
                          v21 = *(_QWORD *)(v3 + 64);
                          if ( v21 != v3 + 80 )
                            j_j___libc_free_0(v21, *(_QWORD *)(v3 + 80) + 1LL);
                          j_j___libc_free_0(v3, 200);
                          v3 = v80;
                        }
                        while ( v80 );
                      }
                      v22 = *(_QWORD *)(v2 + 176);
                      v23 = *(_QWORD *)(v2 + 16);
                      if ( v22 )
                        j_j___libc_free_0_0(v22);
                      if ( *(void **)(v2 + 144) == v1 )
                      {
                        v36 = *(_QWORD *)(v2 + 152);
                        if ( v36 )
                        {
                          v37 = 24LL * *(_QWORD *)(v36 - 8);
                          v38 = (void **)(v36 + v37);
                          if ( v36 != v36 + v37 )
                          {
                            do
                            {
                              while ( 1 )
                              {
                                v38 -= 3;
                                if ( *v38 == v1 )
                                  break;
                                sub_C338F0((__int64)v38);
                                if ( *(void ***)(v2 + 152) == v38 )
                                  goto LABEL_101;
                              }
                              sub_969EE0((__int64)v38);
                            }
                            while ( *(void ***)(v2 + 152) != v38 );
                          }
LABEL_101:
                          j_j_j___libc_free_0_0(v38 - 1);
                        }
                      }
                      else
                      {
                        sub_C338F0(v2 + 144);
                      }
                      if ( *(_DWORD *)(v2 + 136) > 0x40u )
                      {
                        v24 = *(_QWORD *)(v2 + 128);
                        if ( v24 )
                          j_j___libc_free_0_0(v24);
                      }
                      v25 = *(_QWORD *)(v2 + 96);
                      if ( v25 != v2 + 112 )
                        j_j___libc_free_0(v25, *(_QWORD *)(v2 + 112) + 1LL);
                      v26 = *(_QWORD *)(v2 + 64);
                      if ( v26 != v2 + 80 )
                        j_j___libc_free_0(v26, *(_QWORD *)(v2 + 80) + 1LL);
                      j_j___libc_free_0(v2, 200);
                      if ( !v23 )
                        break;
                      v2 = v23;
                    }
                  }
                  v39 = *(_QWORD *)(v82 + 176);
                  v40 = *(_QWORD *)(v82 + 16);
                  if ( v39 )
                    j_j___libc_free_0_0(v39);
                  if ( *(void **)(v82 + 144) == v1 )
                  {
                    v44 = *(_QWORD *)(v82 + 152);
                    if ( v44 )
                    {
                      v45 = 24LL * *(_QWORD *)(v44 - 8);
                      v46 = (void **)(v44 + v45);
                      if ( v44 != v44 + v45 )
                      {
                        do
                        {
                          while ( 1 )
                          {
                            v46 -= 3;
                            if ( *v46 == v1 )
                              break;
                            sub_C338F0((__int64)v46);
                            if ( *(void ***)(v82 + 152) == v46 )
                              goto LABEL_121;
                          }
                          sub_969EE0((__int64)v46);
                        }
                        while ( *(void ***)(v82 + 152) != v46 );
                      }
LABEL_121:
                      j_j_j___libc_free_0_0(v46 - 1);
                    }
                  }
                  else
                  {
                    sub_C338F0(v82 + 144);
                  }
                  if ( *(_DWORD *)(v82 + 136) > 0x40u )
                  {
                    v41 = *(_QWORD *)(v82 + 128);
                    if ( v41 )
                      j_j___libc_free_0_0(v41);
                  }
                  v42 = *(_QWORD *)(v82 + 96);
                  if ( v42 != v82 + 112 )
                    j_j___libc_free_0(v42, *(_QWORD *)(v82 + 112) + 1LL);
                  v43 = *(_QWORD *)(v82 + 64);
                  if ( v43 != v82 + 80 )
                    j_j___libc_free_0(v43, *(_QWORD *)(v82 + 80) + 1LL);
                  j_j___libc_free_0(v82, 200);
                  if ( !v40 )
                    break;
                  v82 = v40;
                }
              }
              v47 = *(_QWORD *)(v83 + 176);
              v48 = *(_QWORD *)(v83 + 16);
              if ( v47 )
                j_j___libc_free_0_0(v47);
              if ( *(void **)(v83 + 144) == v1 )
              {
                v56 = *(_QWORD *)(v83 + 152);
                if ( v56 )
                {
                  v57 = 24LL * *(_QWORD *)(v56 - 8);
                  v58 = (void **)(v56 + v57);
                  if ( v56 != v56 + v57 )
                  {
                    do
                    {
                      while ( 1 )
                      {
                        v58 -= 3;
                        if ( *v58 == v1 )
                          break;
                        sub_C338F0((__int64)v58);
                        if ( *(void ***)(v83 + 152) == v58 )
                          goto LABEL_149;
                      }
                      sub_969EE0((__int64)v58);
                    }
                    while ( *(void ***)(v83 + 152) != v58 );
                  }
LABEL_149:
                  j_j_j___libc_free_0_0(v58 - 1);
                }
              }
              else
              {
                sub_C338F0(v83 + 144);
              }
              if ( *(_DWORD *)(v83 + 136) > 0x40u )
              {
                v49 = *(_QWORD *)(v83 + 128);
                if ( v49 )
                  j_j___libc_free_0_0(v49);
              }
              v50 = *(_QWORD *)(v83 + 96);
              if ( v50 != v83 + 112 )
                j_j___libc_free_0(v50, *(_QWORD *)(v83 + 112) + 1LL);
              v51 = *(_QWORD *)(v83 + 64);
              if ( v51 != v83 + 80 )
                j_j___libc_free_0(v51, *(_QWORD *)(v83 + 80) + 1LL);
              j_j___libc_free_0(v83, 200);
              if ( !v48 )
                break;
              v83 = v48;
            }
          }
          v59 = *(_QWORD *)(v84 + 176);
          v60 = *(_QWORD *)(v84 + 16);
          if ( v59 )
            j_j___libc_free_0_0(v59);
          if ( *(void **)(v84 + 144) == v1 )
          {
            v64 = *(_QWORD *)(v84 + 152);
            if ( v64 )
            {
              v65 = 24LL * *(_QWORD *)(v64 - 8);
              v66 = (void **)(v64 + v65);
              if ( v64 != v64 + v65 )
              {
                do
                {
                  while ( 1 )
                  {
                    v66 -= 3;
                    if ( *v66 == v1 )
                      break;
                    sub_C338F0((__int64)v66);
                    if ( *(void ***)(v84 + 152) == v66 )
                      goto LABEL_169;
                  }
                  sub_969EE0((__int64)v66);
                }
                while ( *(void ***)(v84 + 152) != v66 );
              }
LABEL_169:
              j_j_j___libc_free_0_0(v66 - 1);
            }
          }
          else
          {
            sub_C338F0(v84 + 144);
          }
          if ( *(_DWORD *)(v84 + 136) > 0x40u )
          {
            v61 = *(_QWORD *)(v84 + 128);
            if ( v61 )
              j_j___libc_free_0_0(v61);
          }
          v62 = *(_QWORD *)(v84 + 96);
          if ( v62 != v84 + 112 )
            j_j___libc_free_0(v62, *(_QWORD *)(v84 + 112) + 1LL);
          v63 = *(_QWORD *)(v84 + 64);
          if ( v63 != v84 + 80 )
            j_j___libc_free_0(v63, *(_QWORD *)(v84 + 80) + 1LL);
          j_j___libc_free_0(v84, 200);
          if ( !v60 )
            break;
          v84 = v60;
        }
      }
      v67 = *(_QWORD *)(v81 + 176);
      v68 = *(_QWORD *)(v81 + 16);
      if ( v67 )
        j_j___libc_free_0_0(v67);
      if ( *(void **)(v81 + 144) == v1 )
      {
        v72 = *(_QWORD *)(v81 + 152);
        if ( v72 )
        {
          v73 = 24LL * *(_QWORD *)(v72 - 8);
          v74 = (void **)(v72 + v73);
          if ( v72 != v72 + v73 )
          {
            do
            {
              while ( 1 )
              {
                v74 -= 3;
                if ( *v74 == v1 )
                  break;
                sub_C338F0((__int64)v74);
                if ( *(void ***)(v81 + 152) == v74 )
                  goto LABEL_189;
              }
              sub_969EE0((__int64)v74);
            }
            while ( *(void ***)(v81 + 152) != v74 );
          }
LABEL_189:
          j_j_j___libc_free_0_0(v74 - 1);
        }
      }
      else
      {
        sub_C338F0(v81 + 144);
      }
      if ( *(_DWORD *)(v81 + 136) > 0x40u )
      {
        v69 = *(_QWORD *)(v81 + 128);
        if ( v69 )
          j_j___libc_free_0_0(v69);
      }
      v70 = *(_QWORD *)(v81 + 96);
      if ( v70 != v81 + 112 )
        j_j___libc_free_0(v70, *(_QWORD *)(v81 + 112) + 1LL);
      v71 = *(_QWORD *)(v81 + 64);
      if ( v71 != v81 + 80 )
        j_j___libc_free_0(v71, *(_QWORD *)(v81 + 80) + 1LL);
      j_j___libc_free_0(v81, 200);
      if ( !v68 )
        break;
      v81 = v68;
    }
  }
}
