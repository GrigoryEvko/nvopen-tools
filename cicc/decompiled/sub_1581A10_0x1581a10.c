// Function: sub_1581A10
// Address: 0x1581a10
//
__int64 __fastcall sub_1581A10(__int64 a1, __int64 a2, unsigned int a3, float a4)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // r13
  float v10; // xmm0_4
  __int64 *v11; // r12
  __int64 v12; // rdi
  float v13; // xmm0_4
  __int64 v14; // r8
  float v15; // xmm0_4
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rdi
  float v24; // xmm0_4
  __int64 v25; // rdi
  float v26; // xmm0_4
  __int64 v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r14
  __int64 v31; // rbx
  __int64 v32; // r15
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  float v37; // xmm0_4
  __int64 v38; // rdi
  float v39; // xmm0_4
  __int64 v40; // r15
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rbx
  __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdi
  float v49; // xmm0_4
  __int64 v50; // rdi
  float v51; // xmm0_4
  __int64 v52; // r15
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rbx
  __int64 v56; // r15
  __int64 v57; // r12
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rbx
  __int64 v61; // r15
  __int64 v62; // r12
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // [rsp+0h] [rbp-90h]
  __int64 v66; // [rsp+0h] [rbp-90h]
  __int64 v67; // [rsp+0h] [rbp-90h]
  __int64 v68; // [rsp+0h] [rbp-90h]
  float v69; // [rsp+8h] [rbp-88h]
  float v70; // [rsp+8h] [rbp-88h]
  float v71; // [rsp+8h] [rbp-88h]
  float v72; // [rsp+8h] [rbp-88h]
  __int64 v73; // [rsp+8h] [rbp-88h]
  float v74; // [rsp+8h] [rbp-88h]
  float v75; // [rsp+8h] [rbp-88h]
  __int64 v76; // [rsp+8h] [rbp-88h]
  float v77; // [rsp+8h] [rbp-88h]
  float v78; // [rsp+8h] [rbp-88h]
  __int64 v79; // [rsp+8h] [rbp-88h]
  __int64 v80; // [rsp+8h] [rbp-88h]
  char v81; // [rsp+1Fh] [rbp-71h] BYREF
  _BYTE v82[40]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v83; // [rsp+48h] [rbp-48h] BYREF
  __int64 v84; // [rsp+50h] [rbp-40h]

  v5 = a1;
  if ( (unsigned __int8)((__int64 (*)(void))sub_169DE70)() || (a1 = a2, (unsigned __int8)sub_169DE70(a2)) )
  {
    v20 = sub_16982C0(a1, a2, v6, v7);
    v22 = v5 + 8;
    if ( *(_QWORD *)(v5 + 8) == v20 )
      sub_169CAA0(v22, 0, 0, 0, v21, a4);
    else
      sub_16986F0(v22, 0, 0, 0, v21, a4);
    return 1;
  }
  if ( a3 == 2 )
  {
    v48 = a2 + 8;
    v9 = sub_16982C0(a2, a2, v6, v7);
    if ( *(_QWORD *)(a2 + 8) == v9 )
      v48 = *(_QWORD *)(a2 + 16) + 8LL;
    v49 = sub_169D890(v48);
    v11 = (__int64 *)(v5 + 8);
    v50 = v5 + 8;
    if ( *(_QWORD *)(v5 + 8) == v9 )
      v50 = *(_QWORD *)(v5 + 16) + 8LL;
    v77 = v49;
    v51 = sub_169D890(v50);
    v78 = sub_1C40E50(&v81, 1, 1, v51, v77);
    v15 = v78;
    if ( (unsigned int)sub_1C40EE0(&v81) )
      goto LABEL_40;
    v52 = sub_1698270(&v81, 1);
    sub_169D3B0(v82, v78);
    sub_169E320(&v83, v82, v52);
    sub_1698460(v82);
    sub_1581840(v11, &v83, v53, v54);
    if ( v9 != v83 )
      goto LABEL_12;
    v30 = v84;
    if ( !v84 )
      return 0;
    v55 = v84 + 32LL * *(_QWORD *)(v84 - 8);
    if ( v84 != v55 )
    {
      do
      {
        v55 -= 32;
        if ( v9 == *(_QWORD *)(v55 + 8) )
        {
          v56 = *(_QWORD *)(v55 + 16);
          if ( v56 )
          {
            v57 = v56 + 32LL * *(_QWORD *)(v56 - 8);
            while ( v56 != v57 )
            {
              v57 -= 32;
              if ( v9 == *(_QWORD *)(v57 + 8) )
              {
                v58 = *(_QWORD *)(v57 + 16);
                if ( v58 )
                {
                  v59 = v58 + 32LL * *(_QWORD *)(v58 - 8);
                  if ( v58 != v59 )
                  {
                    do
                    {
                      v67 = v58;
                      v79 = v59 - 32;
                      sub_127D120((_QWORD *)(v59 - 24));
                      v59 = v79;
                      v58 = v67;
                    }
                    while ( v67 != v79 );
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
      while ( v30 != v55 );
    }
LABEL_104:
    j_j_j___libc_free_0_0(v30 - 8);
    return 0;
  }
  if ( a3 > 2 )
  {
    if ( a3 != 3 )
      return 1;
    v23 = a2 + 8;
    v9 = sub_16982C0(a2, a2, v6, v7);
    if ( *(_QWORD *)(a2 + 8) == v9 )
      v23 = *(_QWORD *)(a2 + 16) + 8LL;
    v24 = sub_169D890(v23);
    v11 = (__int64 *)(v5 + 8);
    v25 = v5 + 8;
    if ( *(_QWORD *)(v5 + 8) == v9 )
      v25 = *(_QWORD *)(v5 + 16) + 8LL;
    v71 = v24;
    v26 = sub_169D890(v25);
    v72 = sub_1C40E40(&v81, 1, 1, v26, v71);
    v15 = v72;
    if ( (unsigned int)sub_1C40EE0(&v81) )
      goto LABEL_40;
    v27 = sub_1698270(&v81, 1);
    sub_169D3B0(v82, v72);
    sub_169E320(&v83, v82, v27);
    sub_1698460(v82);
    sub_1581840(v11, &v83, v28, v29);
    if ( v9 != v83 )
      goto LABEL_12;
    v30 = v84;
    if ( !v84 )
      return 0;
    v31 = v84 + 32LL * *(_QWORD *)(v84 - 8);
    if ( v84 != v31 )
    {
      do
      {
        v31 -= 32;
        if ( v9 == *(_QWORD *)(v31 + 8) )
        {
          v32 = *(_QWORD *)(v31 + 16);
          if ( v32 )
          {
            v33 = v32 + 32LL * *(_QWORD *)(v32 - 8);
            while ( v32 != v33 )
            {
              v33 -= 32;
              if ( v9 == *(_QWORD *)(v33 + 8) )
              {
                v34 = *(_QWORD *)(v33 + 16);
                if ( v34 )
                {
                  v35 = v34 + 32LL * *(_QWORD *)(v34 - 8);
                  if ( v34 != v35 )
                  {
                    do
                    {
                      v65 = v34;
                      v73 = v35 - 32;
                      sub_127D120((_QWORD *)(v35 - 24));
                      v35 = v73;
                      v34 = v65;
                    }
                    while ( v65 != v73 );
                  }
                  j_j_j___libc_free_0_0(v34 - 8);
                }
              }
              else
              {
                sub_1698460(v33 + 8);
              }
            }
            j_j_j___libc_free_0_0(v32 - 8);
          }
        }
        else
        {
          sub_1698460(v31 + 8);
        }
      }
      while ( v30 != v31 );
    }
    goto LABEL_104;
  }
  if ( a3 )
  {
    v8 = a2 + 8;
    v9 = sub_16982C0(a2, a2, v6, v7);
    if ( *(_QWORD *)(a2 + 8) == v9 )
      v8 = *(_QWORD *)(a2 + 16) + 8LL;
    v10 = sub_169D890(v8);
    v11 = (__int64 *)(v5 + 8);
    v12 = v5 + 8;
    if ( *(_QWORD *)(v5 + 8) == v9 )
      v12 = *(_QWORD *)(v5 + 16) + 8LL;
    v69 = v10;
    v13 = sub_169D890(v12);
    v70 = sub_1C40E40(&v81, 1, 1, v13, v69);
    v15 = v70;
    if ( !(unsigned int)sub_1C40EE0(&v81) )
    {
      v16 = sub_1698270(&v81, 1);
      sub_169D3B0(v82, v70);
      sub_169E320(&v83, v82, v16);
      sub_1698460(v82);
      sub_1581840(v11, &v83, v17, v18);
      if ( v9 != v83 )
      {
LABEL_12:
        sub_1698460(&v83);
        return 0;
      }
      v30 = v84;
      if ( !v84 )
        return 0;
      v60 = v84 + 32LL * *(_QWORD *)(v84 - 8);
      if ( v84 != v60 )
      {
        do
        {
          v60 -= 32;
          if ( v9 == *(_QWORD *)(v60 + 8) )
          {
            v61 = *(_QWORD *)(v60 + 16);
            if ( v61 )
            {
              v62 = v61 + 32LL * *(_QWORD *)(v61 - 8);
              while ( v61 != v62 )
              {
                v62 -= 32;
                if ( v9 == *(_QWORD *)(v62 + 8) )
                {
                  v63 = *(_QWORD *)(v62 + 16);
                  if ( v63 )
                  {
                    v64 = v63 + 32LL * *(_QWORD *)(v63 - 8);
                    if ( v63 != v64 )
                    {
                      do
                      {
                        v68 = v63;
                        v80 = v64 - 32;
                        sub_127D120((_QWORD *)(v64 - 24));
                        v64 = v80;
                        v63 = v68;
                      }
                      while ( v68 != v80 );
                    }
                    j_j_j___libc_free_0_0(v63 - 8);
                  }
                }
                else
                {
                  sub_1698460(v62 + 8);
                }
              }
              j_j_j___libc_free_0_0(v61 - 8);
            }
          }
          else
          {
            sub_1698460(v60 + 8);
          }
        }
        while ( v30 != v60 );
      }
      goto LABEL_104;
    }
    goto LABEL_40;
  }
  v36 = a2 + 8;
  v9 = sub_16982C0(a2, a2, v6, v7);
  if ( *(_QWORD *)(a2 + 8) == v9 )
    v36 = *(_QWORD *)(a2 + 16) + 8LL;
  v37 = sub_169D890(v36);
  v11 = (__int64 *)(v5 + 8);
  v38 = v5 + 8;
  if ( *(_QWORD *)(v5 + 8) == v9 )
    v38 = *(_QWORD *)(v5 + 16) + 8LL;
  v74 = v37;
  v39 = sub_169D890(v38);
  v75 = sub_1C40E30(&v81, 1, 1, v39, v74);
  v15 = v75;
  if ( !(unsigned int)sub_1C40EE0(&v81) )
  {
    v40 = sub_1698270(&v81, 1);
    sub_169D3B0(v82, v75);
    sub_169E320(&v83, v82, v40);
    sub_1698460(v82);
    sub_1581840(v11, &v83, v41, v42);
    if ( v9 != v83 )
      goto LABEL_12;
    v30 = v84;
    if ( !v84 )
      return 0;
    v43 = v84 + 32LL * *(_QWORD *)(v84 - 8);
    if ( v84 != v43 )
    {
      do
      {
        v43 -= 32;
        if ( v9 == *(_QWORD *)(v43 + 8) )
        {
          v44 = *(_QWORD *)(v43 + 16);
          if ( v44 )
          {
            v45 = v44 + 32LL * *(_QWORD *)(v44 - 8);
            while ( v44 != v45 )
            {
              v45 -= 32;
              if ( v9 == *(_QWORD *)(v45 + 8) )
              {
                v46 = *(_QWORD *)(v45 + 16);
                if ( v46 )
                {
                  v47 = v46 + 32LL * *(_QWORD *)(v46 - 8);
                  if ( v46 != v47 )
                  {
                    do
                    {
                      v66 = v46;
                      v76 = v47 - 32;
                      sub_127D120((_QWORD *)(v47 - 24));
                      v47 = v76;
                      v46 = v66;
                    }
                    while ( v66 != v76 );
                  }
                  j_j_j___libc_free_0_0(v46 - 8);
                }
              }
              else
              {
                sub_1698460(v45 + 8);
              }
            }
            j_j_j___libc_free_0_0(v44 - 8);
          }
        }
        else
        {
          sub_1698460(v43 + 8);
        }
      }
      while ( v30 != v43 );
    }
    goto LABEL_104;
  }
LABEL_40:
  if ( v9 != *(_QWORD *)(v5 + 8) )
  {
    sub_16986F0(v11, 0, 0, 0, v14, v15);
    return 1;
  }
  sub_169CAA0(v11, 0, 0, 0, v14, v15);
  return 1;
}
