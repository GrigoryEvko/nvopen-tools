// Function: sub_3886120
// Address: 0x3886120
//
__int64 __fastcall sub_3886120(__int64 a1)
{
  __int64 v2; // r12
  int v3; // edi
  char v4; // r14
  unsigned __int8 *v5; // r15
  int v6; // edi
  unsigned __int8 *v7; // r13
  void *v8; // r14
  void *v9; // rax
  void *v10; // r13
  void *v12; // r14
  void *v13; // rax
  void *v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  void *v24; // r12
  void *v25; // r14
  void *v26; // rax
  void *v27; // r13
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  void *v52; // r14
  void *v53; // rax
  void *v54; // r13
  __int64 v55; // rbx
  __int64 v56; // r15
  __int64 v57; // r12
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // [rsp+0h] [rbp-80h]
  __int64 v61; // [rsp+0h] [rbp-80h]
  __int64 v62; // [rsp+0h] [rbp-80h]
  __int64 v63; // [rsp+0h] [rbp-80h]
  __int64 v64; // [rsp+0h] [rbp-80h]
  __int64 v65; // [rsp+8h] [rbp-78h]
  __int64 v66; // [rsp+8h] [rbp-78h]
  __int64 v67; // [rsp+8h] [rbp-78h]
  __int64 v68; // [rsp+8h] [rbp-78h]
  __int64 v69; // [rsp+8h] [rbp-78h]
  __int64 v70[2]; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v71; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v72; // [rsp+28h] [rbp-58h]
  void *v73; // [rsp+38h] [rbp-48h] BYREF
  __int64 v74; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)a1 = v2 + 2;
  v3 = *(unsigned __int8 *)(v2 + 2);
  if ( (unsigned __int8)(v3 - 75) > 2u )
  {
    v4 = 74;
    if ( (_BYTE)v3 != 72 )
    {
      v5 = (unsigned __int8 *)(v2 + 3);
      if ( isxdigit(v3) )
        goto LABEL_4;
LABEL_18:
      *(_QWORD *)a1 = v2 + 1;
      return 1;
    }
  }
  *(_QWORD *)a1 = v2 + 3;
  v5 = (unsigned __int8 *)(v2 + 4);
  v4 = *(_BYTE *)(v2 + 2);
  if ( !isxdigit(*(unsigned __int8 *)(v2 + 3)) )
    goto LABEL_18;
  do
  {
LABEL_4:
    *(_QWORD *)a1 = v5;
    v6 = *v5;
    v7 = v5++;
  }
  while ( isxdigit(v6) );
  if ( v4 == 74 )
  {
    v72 = 64;
    v71 = sub_3885DE0(a1, (_BYTE *)(v2 + 2), v7);
    v52 = sub_1698280();
    v53 = sub_16982C0();
    v54 = v53;
    if ( v52 == v53 )
      sub_169D060(&v73, (__int64)v53, (__int64 *)&v71);
    else
      sub_169D050((__int64)&v73, v52, (__int64 *)&v71);
    sub_3880CD0((void **)(a1 + 120), &v73);
    if ( v73 != v54 )
      goto LABEL_12;
    v15 = v74;
    if ( !v74 )
      goto LABEL_13;
    v55 = v74 + 32LL * *(_QWORD *)(v74 - 8);
    if ( v74 != v55 )
    {
      do
      {
        v55 -= 32;
        if ( v54 == *(void **)(v55 + 8) )
        {
          v56 = *(_QWORD *)(v55 + 16);
          if ( v56 )
          {
            v57 = v56 + 32LL * *(_QWORD *)(v56 - 8);
            while ( v56 != v57 )
            {
              v57 -= 32;
              if ( v54 == *(void **)(v57 + 8) )
              {
                v58 = *(_QWORD *)(v57 + 16);
                if ( v58 )
                {
                  v59 = v58 + 32LL * *(_QWORD *)(v58 - 8);
                  if ( v58 != v59 )
                  {
                    do
                    {
                      v64 = v58;
                      v69 = v59 - 32;
                      sub_127D120((_QWORD *)(v59 - 24));
                      v59 = v69;
                      v58 = v64;
                    }
                    while ( v64 != v69 );
                  }
                  j_j_j___libc_free_0_0(v58 - 8);
                }
              }
              else
              {
                sub_1698460(v57 + 8);
              }
            }
            j_j_j___libc_free_0_0(v56 - 8);
          }
        }
        else
        {
          sub_1698460(v55 + 8);
        }
      }
      while ( v15 != v55 );
    }
    goto LABEL_59;
  }
  if ( v4 == 76 )
  {
    sub_3885FD0(a1, (_BYTE *)(v2 + 3), v7, v70);
    sub_16A50F0((__int64)&v71, 128, v70, 2u);
    v25 = sub_1698290();
    v26 = sub_16982C0();
    v27 = v26;
    if ( v25 == v26 )
      sub_169D060(&v73, (__int64)v26, (__int64 *)&v71);
    else
      sub_169D050((__int64)&v73, v25, (__int64 *)&v71);
    sub_3880CD0((void **)(a1 + 120), &v73);
    if ( v73 != v27 )
      goto LABEL_12;
    v15 = v74;
    if ( !v74 )
      goto LABEL_13;
    v28 = 32LL * *(_QWORD *)(v74 - 8);
    v29 = v74 + v28;
    if ( v74 != v74 + v28 )
    {
      do
      {
        v29 -= 32;
        if ( v27 == *(void **)(v29 + 8) )
        {
          v30 = *(_QWORD *)(v29 + 16);
          if ( v30 )
          {
            v31 = 32LL * *(_QWORD *)(v30 - 8);
            v32 = v30 + v31;
            while ( v30 != v32 )
            {
              v32 -= 32;
              if ( v27 == *(void **)(v32 + 8) )
              {
                v33 = *(_QWORD *)(v32 + 16);
                if ( v33 )
                {
                  v34 = 32LL * *(_QWORD *)(v33 - 8);
                  v35 = v33 + v34;
                  if ( v33 != v33 + v34 )
                  {
                    do
                    {
                      v61 = v33;
                      v66 = v35 - 32;
                      sub_127D120((_QWORD *)(v35 - 24));
                      v35 = v66;
                      v33 = v61;
                    }
                    while ( v61 != v66 );
                  }
                  j_j_j___libc_free_0_0(v33 - 8);
                }
              }
              else
              {
                sub_1698460(v32 + 8);
              }
            }
            j_j_j___libc_free_0_0(v30 - 8);
          }
        }
        else
        {
          sub_1698460(v29 + 8);
        }
      }
      while ( v15 != v29 );
    }
    goto LABEL_59;
  }
  if ( v4 <= 76 )
  {
    if ( v4 == 72 )
    {
      v72 = 16;
      v71 = (unsigned __int16)sub_3885DE0(a1, (_BYTE *)(v2 + 3), v7);
      v8 = sub_1698260();
      v9 = sub_16982C0();
      v10 = v9;
      if ( v8 == v9 )
        sub_169D060(&v73, (__int64)v9, (__int64 *)&v71);
      else
        sub_169D050((__int64)&v73, v8, (__int64 *)&v71);
      sub_3880CD0((void **)(a1 + 120), &v73);
      if ( v73 != v10 )
        goto LABEL_12;
      v15 = v74;
      if ( !v74 )
        goto LABEL_13;
      v44 = 32LL * *(_QWORD *)(v74 - 8);
      v45 = v74 + v44;
      if ( v74 != v74 + v44 )
      {
        do
        {
          v45 -= 32;
          if ( v10 == *(void **)(v45 + 8) )
          {
            v46 = *(_QWORD *)(v45 + 16);
            if ( v46 )
            {
              v47 = 32LL * *(_QWORD *)(v46 - 8);
              v48 = v46 + v47;
              while ( v46 != v48 )
              {
                v48 -= 32;
                if ( v10 == *(void **)(v48 + 8) )
                {
                  v49 = *(_QWORD *)(v48 + 16);
                  if ( v49 )
                  {
                    v50 = 32LL * *(_QWORD *)(v49 - 8);
                    v51 = v49 + v50;
                    if ( v49 != v49 + v50 )
                    {
                      do
                      {
                        v63 = v49;
                        v68 = v51 - 32;
                        sub_127D120((_QWORD *)(v51 - 24));
                        v51 = v68;
                        v49 = v63;
                      }
                      while ( v63 != v68 );
                    }
                    j_j_j___libc_free_0_0(v49 - 8);
                  }
                }
                else
                {
                  sub_1698460(v48 + 8);
                }
              }
              j_j_j___libc_free_0_0(v46 - 8);
            }
          }
          else
          {
            sub_1698460(v45 + 8);
          }
        }
        while ( v15 != v45 );
      }
    }
    else
    {
      sub_3885E90(a1, (_BYTE *)(v2 + 3), v7, v70);
      sub_16A50F0((__int64)&v71, 80, v70, 2u);
      v12 = sub_16982A0();
      v13 = sub_16982C0();
      v14 = v13;
      if ( v12 == v13 )
        sub_169D060(&v73, (__int64)v13, (__int64 *)&v71);
      else
        sub_169D050((__int64)&v73, v12, (__int64 *)&v71);
      sub_3880CD0((void **)(a1 + 120), &v73);
      if ( v73 != v14 )
        goto LABEL_12;
      v15 = v74;
      if ( !v74 )
        goto LABEL_13;
      v16 = 32LL * *(_QWORD *)(v74 - 8);
      v17 = v74 + v16;
      if ( v74 != v74 + v16 )
      {
        do
        {
          v17 -= 32;
          if ( v14 == *(void **)(v17 + 8) )
          {
            v18 = *(_QWORD *)(v17 + 16);
            if ( v18 )
            {
              v19 = 32LL * *(_QWORD *)(v18 - 8);
              v20 = v18 + v19;
              while ( v18 != v20 )
              {
                v20 -= 32;
                if ( v14 == *(void **)(v20 + 8) )
                {
                  v21 = *(_QWORD *)(v20 + 16);
                  if ( v21 )
                  {
                    v22 = 32LL * *(_QWORD *)(v21 - 8);
                    v23 = v21 + v22;
                    if ( v21 != v21 + v22 )
                    {
                      do
                      {
                        v60 = v21;
                        v65 = v23 - 32;
                        sub_127D120((_QWORD *)(v23 - 24));
                        v23 = v65;
                        v21 = v60;
                      }
                      while ( v60 != v65 );
                    }
                    j_j_j___libc_free_0_0(v21 - 8);
                  }
                }
                else
                {
                  sub_1698460(v20 + 8);
                }
              }
              j_j_j___libc_free_0_0(v18 - 8);
            }
          }
          else
          {
            sub_1698460(v17 + 8);
          }
        }
        while ( v15 != v17 );
      }
    }
LABEL_59:
    j_j_j___libc_free_0_0(v15 - 8);
    goto LABEL_13;
  }
  sub_3885FD0(a1, (_BYTE *)(v2 + 3), v7, v70);
  sub_16A50F0((__int64)&v71, 128, v70, 2u);
  v24 = sub_16982C0();
  sub_169D060(&v73, (__int64)v24, (__int64 *)&v71);
  sub_3880CD0((void **)(a1 + 120), &v73);
  if ( v24 == v73 )
  {
    v15 = v74;
    if ( !v74 )
      goto LABEL_13;
    v36 = 32LL * *(_QWORD *)(v74 - 8);
    v37 = v74 + v36;
    if ( v74 != v74 + v36 )
    {
      do
      {
        v37 -= 32;
        if ( v24 == *(void **)(v37 + 8) )
        {
          v38 = *(_QWORD *)(v37 + 16);
          if ( v38 )
          {
            v39 = 32LL * *(_QWORD *)(v38 - 8);
            v40 = v38 + v39;
            while ( v38 != v40 )
            {
              v40 -= 32;
              if ( v24 == *(void **)(v40 + 8) )
              {
                v41 = *(_QWORD *)(v40 + 16);
                if ( v41 )
                {
                  v42 = 32LL * *(_QWORD *)(v41 - 8);
                  v43 = v41 + v42;
                  if ( v41 != v41 + v42 )
                  {
                    do
                    {
                      v62 = v41;
                      v67 = v43 - 32;
                      sub_127D120((_QWORD *)(v43 - 24));
                      v43 = v67;
                      v41 = v62;
                    }
                    while ( v62 != v67 );
                  }
                  j_j_j___libc_free_0_0(v41 - 8);
                }
              }
              else
              {
                sub_1698460(v40 + 8);
              }
            }
            j_j_j___libc_free_0_0(v38 - 8);
          }
        }
        else
        {
          sub_1698460(v37 + 8);
        }
      }
      while ( v15 != v37 );
    }
    goto LABEL_59;
  }
LABEL_12:
  sub_1698460((__int64)&v73);
LABEL_13:
  if ( v72 > 0x40 )
  {
    if ( v71 )
      j_j___libc_free_0_0(v71);
  }
  return 389;
}
