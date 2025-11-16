// Function: sub_19222E0
// Address: 0x19222e0
//
void __fastcall sub_19222E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int8 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned __int8 *v19; // r14
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // r13
  unsigned __int8 *v23; // r14
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r14
  unsigned __int64 v28; // rax
  __int64 v29; // r15
  __int64 *v30; // rbx
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 *v33; // r13
  __int64 v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __int64 *v37; // r12
  __int64 v38; // rbx
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  __int64 *v41; // r12
  __int64 v42; // rbx
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 *v45; // r12
  __int64 v46; // r13
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  __int64 *v49; // rbx
  __int64 v50; // r12
  __int64 v51; // rax
  __int64 v52; // [rsp+8h] [rbp-F8h]
  unsigned __int8 *v53; // [rsp+10h] [rbp-F0h]
  __int64 v54; // [rsp+18h] [rbp-E8h]
  __int64 v55; // [rsp+28h] [rbp-D8h]
  __int64 v56; // [rsp+30h] [rbp-D0h]
  __int64 v57; // [rsp+30h] [rbp-D0h]
  __int64 v58; // [rsp+38h] [rbp-C8h]
  __int64 v59; // [rsp+40h] [rbp-C0h]
  __int64 v60; // [rsp+48h] [rbp-B8h]
  __int64 v61; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v62; // [rsp+58h] [rbp-A8h]
  __int64 v63; // [rsp+60h] [rbp-A0h]
  unsigned __int8 *v64; // [rsp+68h] [rbp-98h]
  unsigned __int8 *v65; // [rsp+70h] [rbp-90h]
  __int64 v66; // [rsp+78h] [rbp-88h]
  __int64 v67; // [rsp+80h] [rbp-80h]
  __int64 v70; // [rsp+98h] [rbp-68h]
  __int64 v71; // [rsp+A0h] [rbp-60h]
  __int64 v72; // [rsp+B0h] [rbp-50h]
  __int64 v73; // [rsp+B8h] [rbp-48h]
  __int64 v74; // [rsp+C0h] [rbp-40h]
  __int64 v75; // [rsp+C8h] [rbp-38h]

  v65 = (unsigned __int8 *)sub_15F4880(a5);
  v7 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
  if ( v7 )
  {
    v75 = 0;
    v8 = a3;
    v63 = 24LL * v7;
    do
    {
      if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
        v9 = *(_QWORD *)(a5 - 8);
      else
        v9 = a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF);
      v10 = *(_QWORD *)(v9 + v75);
      v70 = v10;
      if ( *(_BYTE *)(v10 + 16) > 0x17u
        && !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v10 + 40), v8)
        && *(_BYTE *)(v10 + 16) == 56 )
      {
        v64 = (unsigned __int8 *)sub_15F4880(v10);
        v11 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        if ( v11 )
        {
          v74 = 0;
          v73 = v8;
          v61 = 24LL * v11;
          do
          {
            if ( (*(_BYTE *)(v70 + 23) & 0x40) != 0 )
              v12 = *(_QWORD *)(v70 - 8);
            else
              v12 = v70 - 24LL * (*(_DWORD *)(v70 + 20) & 0xFFFFFFF);
            v13 = *(_QWORD *)(v12 + v74);
            if ( *(_BYTE *)(v13 + 16) > 0x17u
              && !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v13 + 40), v73)
              && *(_BYTE *)(v13 + 16) == 56 )
            {
              v62 = (unsigned __int8 *)sub_15F4880(v13);
              if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) != 0 )
              {
                v72 = 0;
                v67 = v13;
                v60 = 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
                do
                {
                  if ( (*(_BYTE *)(v67 + 23) & 0x40) != 0 )
                    v14 = *(_QWORD *)(v67 - 8);
                  else
                    v14 = v67 - 24LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF);
                  v15 = *(_QWORD *)(v14 + v72);
                  if ( *(_BYTE *)(v15 + 16) > 0x17u
                    && !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v15 + 40), v73)
                    && *(_BYTE *)(v15 + 16) == 56 )
                  {
                    v16 = (unsigned __int8 *)sub_15F4880(v15);
                    if ( (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) != 0 )
                    {
                      v71 = 0;
                      v66 = v15;
                      v55 = (__int64)v16;
                      v59 = 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
                      do
                      {
                        if ( (*(_BYTE *)(v66 + 23) & 0x40) != 0 )
                          v17 = *(_QWORD *)(v66 - 8);
                        else
                          v17 = v66 - 24LL * (*(_DWORD *)(v66 + 20) & 0xFFFFFFF);
                        v18 = *(_QWORD *)(v17 + v71);
                        if ( *(_BYTE *)(v18 + 16) > 0x17u
                          && !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v18 + 40), v73)
                          && *(_BYTE *)(v18 + 16) == 56 )
                        {
                          v19 = (unsigned __int8 *)sub_15F4880(v18);
                          if ( (*(_DWORD *)(v18 + 20) & 0xFFFFFFF) != 0 )
                          {
                            v54 = (__int64)v19;
                            v20 = 0;
                            v58 = 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
                            do
                            {
                              if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
                                v21 = *(_QWORD *)(v18 - 8);
                              else
                                v21 = v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
                              v22 = *(_QWORD *)(v21 + v20);
                              if ( *(_BYTE *)(v22 + 16) > 0x17u
                                && !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v22 + 40), v73)
                                && *(_BYTE *)(v22 + 16) == 56 )
                              {
                                v23 = (unsigned __int8 *)sub_15F4880(v22);
                                if ( (*(_DWORD *)(v22 + 20) & 0xFFFFFFF) != 0 )
                                {
                                  v53 = v23;
                                  v52 = v18;
                                  v24 = v22;
                                  v56 = 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
                                  v25 = 0;
                                  do
                                  {
                                    if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
                                      v26 = *(_QWORD *)(v24 - 8);
                                    else
                                      v26 = v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF);
                                    v27 = *(_QWORD *)(v26 + v25);
                                    if ( *(_BYTE *)(v27 + 16) > 0x17u
                                      && !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v27 + 40), v73)
                                      && *(_BYTE *)(v27 + 16) == 56 )
                                    {
                                      sub_19222E0(a1, v53, v73, a4, v27);
                                    }
                                    v25 += 24;
                                  }
                                  while ( v56 != v25 );
                                  v22 = v24;
                                  v23 = v53;
                                  v18 = v52;
                                }
                                v28 = sub_157EBA0(v73);
                                sub_15F2120((__int64)v23, v28);
                                sub_1624960((__int64)v23, 0, 0);
                                if ( *(_QWORD *)a4 != *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8) )
                                {
                                  v57 = v20;
                                  v29 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
                                  v30 = *(__int64 **)a4;
                                  do
                                  {
                                    v31 = *v30++;
                                    sub_15F2780(v23, *(_QWORD *)(v31 - 24));
                                  }
                                  while ( (__int64 *)v29 != v30 );
                                  v20 = v57;
                                }
                                sub_1648780(v54, v22, (__int64)v23);
                              }
                              v20 += 24;
                            }
                            while ( v58 != v20 );
                            v19 = (unsigned __int8 *)v54;
                          }
                          v32 = sub_157EBA0(v73);
                          sub_15F2120((__int64)v19, v32);
                          sub_1624960((__int64)v19, 0, 0);
                          v33 = *(__int64 **)a4;
                          v34 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
                          if ( *(_QWORD *)a4 != v34 )
                          {
                            do
                            {
                              v35 = *v33++;
                              sub_15F2780(v19, *(_QWORD *)(v35 - 24));
                            }
                            while ( (__int64 *)v34 != v33 );
                          }
                          sub_1648780(v55, v18, (__int64)v19);
                        }
                        v71 += 24;
                      }
                      while ( v59 != v71 );
                      v15 = v66;
                      v16 = (unsigned __int8 *)v55;
                    }
                    v36 = sub_157EBA0(v73);
                    sub_15F2120((__int64)v16, v36);
                    sub_1624960((__int64)v16, 0, 0);
                    v37 = *(__int64 **)a4;
                    v38 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
                    if ( *(_QWORD *)a4 != v38 )
                    {
                      do
                      {
                        v39 = *v37++;
                        sub_15F2780(v16, *(_QWORD *)(v39 - 24));
                      }
                      while ( (__int64 *)v38 != v37 );
                    }
                    sub_1648780((__int64)v62, v15, (__int64)v16);
                  }
                  v72 += 24;
                }
                while ( v60 != v72 );
                v13 = v67;
              }
              v40 = sub_157EBA0(v73);
              sub_15F2120((__int64)v62, v40);
              sub_1624960((__int64)v62, 0, 0);
              v41 = *(__int64 **)a4;
              v42 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
              if ( *(_QWORD *)a4 != v42 )
              {
                do
                {
                  v43 = *v41++;
                  sub_15F2780(v62, *(_QWORD *)(v43 - 24));
                }
                while ( (__int64 *)v42 != v41 );
              }
              sub_1648780((__int64)v64, v13, (__int64)v62);
            }
            v74 += 24;
          }
          while ( v61 != v74 );
          v8 = v73;
        }
        v44 = sub_157EBA0(v8);
        sub_15F2120((__int64)v64, v44);
        sub_1624960((__int64)v64, 0, 0);
        v45 = *(__int64 **)a4;
        v46 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
        if ( *(_QWORD *)a4 != v46 )
        {
          do
          {
            v47 = *v45++;
            sub_15F2780(v64, *(_QWORD *)(v47 - 24));
          }
          while ( (__int64 *)v46 != v45 );
        }
        sub_1648780((__int64)v65, v70, (__int64)v64);
      }
      v75 += 24;
    }
    while ( v63 != v75 );
    a3 = v8;
  }
  v48 = sub_157EBA0(a3);
  sub_15F2120((__int64)v65, v48);
  sub_1624960((__int64)v65, 0, 0);
  v49 = *(__int64 **)a4;
  v50 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( *(_QWORD *)a4 != v50 )
  {
    do
    {
      v51 = *v49++;
      sub_15F2780(v65, *(_QWORD *)(v51 - 24));
    }
    while ( (__int64 *)v50 != v49 );
  }
  sub_1648780(a2, a5, (__int64)v65);
}
