// Function: sub_210CB60
// Address: 0x210cb60
//
signed __int64 __fastcall sub_210CB60(__int64 **a1, __int64 **a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 *v5; // r13
  __int64 **v6; // r12
  size_t v7; // rdx
  const char *v8; // rax
  size_t v9; // rdx
  size_t v10; // rcx
  size_t v11; // r13
  int v12; // eax
  __int64 *v13; // r13
  size_t v14; // rdx
  const char *v15; // rax
  size_t v16; // rdx
  size_t v17; // rcx
  size_t v18; // r13
  int v19; // eax
  __int64 *v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // r12
  __int64 **v23; // r15
  int v24; // eax
  size_t v25; // rcx
  const char *v26; // r13
  size_t v27; // rdx
  size_t v28; // r14
  size_t v29; // rdx
  const char *v30; // rdi
  __int64 *v31; // rax
  __int64 *v32; // r12
  const char *v33; // r13
  size_t v34; // rdx
  size_t v35; // r14
  size_t v36; // rdx
  const char *v37; // rdi
  size_t v38; // r12
  int v39; // eax
  __int64 *v40; // rax
  __int64 *v41; // r13
  size_t v42; // rdx
  const char *v43; // rax
  size_t v44; // rdx
  size_t v45; // rcx
  size_t v46; // r13
  int v47; // eax
  __int64 v48; // r12
  __int64 v49; // r14
  __int64 *v50; // rcx
  __int64 *v51; // r14
  const char *v52; // rax
  size_t v53; // rdx
  size_t v54; // r13
  const char *v55; // rsi
  size_t v56; // rdx
  const char *v57; // rdi
  size_t v58; // rcx
  int v59; // eax
  __int64 *v60; // rax
  __int64 *v61; // r13
  const char *v62; // rax
  size_t v63; // rdx
  size_t v64; // r12
  const char *v65; // rsi
  size_t v66; // rdx
  const char *v67; // rdi
  size_t v68; // r13
  int v69; // eax
  __int64 **v70; // [rsp+0h] [rbp-60h]
  __int64 v71; // [rsp+8h] [rbp-58h]
  __int64 **v72; // [rsp+10h] [rbp-50h]
  size_t v73; // [rsp+18h] [rbp-48h]
  size_t v74; // [rsp+18h] [rbp-48h]
  const char *s2; // [rsp+20h] [rbp-40h]
  const char *s2a; // [rsp+20h] [rbp-40h]
  __int64 **s2b; // [rsp+20h] [rbp-40h]
  const char *s2c; // [rsp+20h] [rbp-40h]
  size_t n; // [rsp+28h] [rbp-38h]
  size_t na; // [rsp+28h] [rbp-38h]
  __int64 **nb; // [rsp+28h] [rbp-38h]
  size_t nc; // [rsp+28h] [rbp-38h]
  size_t nd; // [rsp+28h] [rbp-38h]
  size_t ne; // [rsp+28h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v72 = a2;
  v71 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    s2b = a2;
    goto LABEL_49;
  }
  v70 = a1 + 1;
  while ( 2 )
  {
    v5 = a1[1];
    --v71;
    v6 = &a1[result >> 4];
    s2 = sub_1649960(**v6);
    n = v7;
    v8 = sub_1649960(*v5);
    v10 = n;
    v11 = v9;
    if ( v9 > n )
    {
      if ( !n )
        goto LABEL_42;
      v12 = memcmp(v8, s2, n);
      v10 = n;
      if ( !v12 )
        goto LABEL_8;
    }
    else if ( !v9 || (v12 = memcmp(v8, s2, v9), v10 = n, !v12) )
    {
      if ( v11 == v10 )
        goto LABEL_42;
LABEL_8:
      if ( v11 < v10 )
        goto LABEL_9;
LABEL_42:
      v41 = a1[1];
      s2c = sub_1649960(**(v72 - 1));
      nc = v42;
      v43 = sub_1649960(*v41);
      v45 = nc;
      v46 = v44;
      if ( nc < v44 )
      {
        if ( !nc )
          goto LABEL_56;
        v47 = memcmp(v43, s2c, nc);
        v45 = nc;
        if ( !v47 )
          goto LABEL_46;
      }
      else if ( !v44 || (v47 = memcmp(v43, s2c, v44), v45 = nc, !v47) )
      {
        if ( v45 == v46 )
          goto LABEL_56;
LABEL_46:
        if ( v45 > v46 )
          goto LABEL_47;
LABEL_56:
        v51 = *v6;
        v52 = sub_1649960(**(v72 - 1));
        v54 = v53;
        v55 = v52;
        v57 = sub_1649960(*v51);
        v58 = v56;
        if ( v56 > v54 )
        {
          if ( !v54 )
            goto LABEL_14;
          ne = v56;
          v59 = memcmp(v57, v55, v54);
          v58 = ne;
          if ( !v59 )
          {
LABEL_60:
            if ( v58 >= v54 )
              goto LABEL_14;
            goto LABEL_61;
          }
        }
        else if ( !v56 || (nd = v56, v59 = memcmp(v57, v55, v56), v58 = nd, !v59) )
        {
          if ( v58 == v54 )
            goto LABEL_14;
          goto LABEL_60;
        }
        if ( v59 >= 0 )
          goto LABEL_14;
        goto LABEL_61;
      }
      if ( v47 < 0 )
        goto LABEL_47;
      goto LABEL_56;
    }
    if ( v12 >= 0 )
      goto LABEL_42;
LABEL_9:
    v13 = *v6;
    s2a = sub_1649960(**(v72 - 1));
    na = v14;
    v15 = sub_1649960(*v13);
    v17 = na;
    v18 = v16;
    if ( v16 > na )
    {
      if ( !na )
        goto LABEL_65;
      v19 = memcmp(v15, s2a, na);
      v17 = na;
      if ( !v19 )
        goto LABEL_13;
LABEL_64:
      if ( v19 < 0 )
        goto LABEL_14;
LABEL_65:
      v61 = a1[1];
      v62 = sub_1649960(**(v72 - 1));
      v64 = v63;
      v65 = v62;
      v67 = sub_1649960(*v61);
      v68 = v66;
      if ( v64 < v66 )
      {
        if ( !v64 )
          goto LABEL_47;
        v69 = memcmp(v67, v65, v64);
        if ( !v69 )
        {
LABEL_69:
          if ( v64 <= v68 )
            goto LABEL_47;
LABEL_61:
          v60 = *a1;
          *a1 = *(v72 - 1);
          *(v72 - 1) = v60;
          v21 = *a1;
          v22 = a1[1];
          goto LABEL_15;
        }
      }
      else if ( !v66 || (v69 = memcmp(v67, v65, v66)) == 0 )
      {
        if ( v64 != v68 )
          goto LABEL_69;
LABEL_47:
        v22 = *a1;
        v21 = a1[1];
        a1[1] = *a1;
        *a1 = v21;
        goto LABEL_15;
      }
      if ( v69 < 0 )
        goto LABEL_61;
      goto LABEL_47;
    }
    if ( v16 )
    {
      v19 = memcmp(v15, s2a, v16);
      v17 = na;
      if ( v19 )
        goto LABEL_64;
    }
    if ( v18 == v17 )
      goto LABEL_65;
LABEL_13:
    if ( v18 >= v17 )
      goto LABEL_65;
LABEL_14:
    v20 = *a1;
    *a1 = *v6;
    *v6 = v20;
    v21 = *a1;
    v22 = a1[1];
LABEL_15:
    v23 = v72;
    for ( nb = v70; ; ++nb )
    {
      s2b = nb;
      v26 = sub_1649960(*v21);
      v28 = v27;
      v30 = sub_1649960(*v22);
      v25 = v29;
      if ( v29 > v28 )
        break;
      if ( v29 )
      {
        v73 = v29;
        v24 = memcmp(v30, v26, v29);
        v25 = v73;
        if ( v24 )
          goto LABEL_24;
      }
      if ( v25 == v28 )
        goto LABEL_25;
LABEL_19:
      if ( v25 >= v28 )
        goto LABEL_25;
LABEL_20:
      v21 = *a1;
      v22 = nb[1];
    }
    if ( !v28 )
      goto LABEL_25;
    v74 = v29;
    v24 = memcmp(v30, v26, v28);
    v25 = v74;
    if ( !v24 )
      goto LABEL_19;
LABEL_24:
    if ( v24 < 0 )
      goto LABEL_20;
    do
    {
LABEL_25:
      while ( 1 )
      {
        v31 = *(v23 - 1);
        v32 = *a1;
        --v23;
        v33 = sub_1649960(*v31);
        v35 = v34;
        v37 = sub_1649960(*v32);
        v38 = v36;
        if ( v36 > v35 )
          break;
        if ( v36 )
        {
          v39 = memcmp(v37, v33, v36);
          if ( v39 )
            goto LABEL_34;
        }
        if ( v38 == v35 )
          goto LABEL_30;
LABEL_29:
        if ( v38 >= v35 )
          goto LABEL_30;
      }
      if ( !v35 )
      {
LABEL_30:
        if ( nb >= v23 )
          goto LABEL_36;
LABEL_31:
        v40 = *nb;
        *nb = *v23;
        *v23 = v40;
        goto LABEL_20;
      }
      v39 = memcmp(v37, v33, v35);
      if ( !v39 )
        goto LABEL_29;
LABEL_34:
      ;
    }
    while ( v39 < 0 );
    if ( nb < v23 )
      goto LABEL_31;
LABEL_36:
    sub_210CB60(nb, v72, v71);
    result = (char *)nb - (char *)a1;
    if ( (char *)nb - (char *)a1 > 128 )
    {
      if ( v71 )
      {
        v72 = nb;
        continue;
      }
LABEL_49:
      v48 = result >> 3;
      v49 = ((result >> 3) - 2) >> 1;
      sub_210C6E0((__int64)a1, v49, result >> 3, a1[v49]);
      do
      {
        --v49;
        sub_210C6E0((__int64)a1, v49, v48, a1[v49]);
      }
      while ( v49 );
      do
      {
        v50 = *--s2b;
        *s2b = *a1;
        result = (signed __int64)sub_210C6E0((__int64)a1, 0, s2b - a1, v50);
      }
      while ( (char *)s2b - (char *)a1 > 8 );
    }
    return result;
  }
}
