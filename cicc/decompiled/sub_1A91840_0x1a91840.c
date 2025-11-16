// Function: sub_1A91840
// Address: 0x1a91840
//
__int64 __fastcall sub_1A91840(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r13
  char *v6; // r12
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // r14
  const char *v10; // rsi
  size_t v11; // rdx
  const char *v12; // rdi
  size_t v13; // r13
  int v14; // eax
  __int64 v15; // r13
  const char *v16; // rax
  size_t v17; // rdx
  size_t v18; // r14
  const char *v19; // rsi
  size_t v20; // rdx
  const char *v21; // rdi
  size_t v22; // r13
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r12
  int v27; // eax
  size_t v28; // rcx
  const char *v29; // r14
  size_t v30; // rdx
  size_t v31; // r15
  size_t v32; // rdx
  const char *v33; // rdi
  char *v34; // r14
  __int64 v35; // rax
  __int64 v36; // r13
  const char *v37; // r15
  size_t v38; // rdx
  size_t v39; // r12
  size_t v40; // rdx
  const char *v41; // rdi
  size_t v42; // r13
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // r13
  const char *v46; // rax
  size_t v47; // rdx
  size_t v48; // r14
  const char *v49; // rsi
  size_t v50; // rdx
  const char *v51; // rdi
  size_t v52; // r13
  int v53; // eax
  __int64 v54; // r13
  __int64 i; // r12
  __int64 *v56; // r14
  __int64 v57; // rcx
  __int64 v58; // r12
  __int64 v59; // r14
  const char *v60; // rax
  size_t v61; // rdx
  size_t v62; // r13
  const char *v63; // rsi
  size_t v64; // rdx
  const char *v65; // rdi
  size_t v66; // r14
  int v67; // eax
  __int64 v68; // rax
  __int64 v69; // r13
  const char *v70; // r14
  size_t v71; // rdx
  size_t v72; // r12
  size_t v73; // rdx
  const char *v74; // rdi
  size_t v75; // r13
  int v76; // eax
  char *v77; // [rsp+8h] [rbp-68h]
  __int64 v78; // [rsp+10h] [rbp-60h]
  char *v79; // [rsp+18h] [rbp-58h]
  size_t v80; // [rsp+20h] [rbp-50h]
  size_t v81; // [rsp+20h] [rbp-50h]
  char *v82; // [rsp+28h] [rbp-48h]
  char *v83; // [rsp+30h] [rbp-40h]
  char *s2; // [rsp+38h] [rbp-38h]

  result = a2 - a1;
  v78 = a3;
  v79 = a2;
  if ( a2 - a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v82 = a2;
    goto LABEL_50;
  }
  v77 = a1 + 8;
  while ( 2 )
  {
    v5 = *((_QWORD *)a1 + 1);
    --v78;
    v6 = &a1[8 * ((__int64)(((v79 - a1) >> 3) + ((unsigned __int64)(v79 - a1) >> 63)) >> 1)];
    v7 = sub_1649960(*(_QWORD *)(*(_QWORD *)v6 + 40LL));
    v9 = v8;
    v10 = v7;
    v12 = sub_1649960(*(_QWORD *)(v5 + 40));
    v13 = v11;
    if ( v11 > v9 )
    {
      if ( !v9 )
        goto LABEL_43;
      v14 = memcmp(v12, v10, v9);
      if ( !v14 )
        goto LABEL_8;
    }
    else if ( !v11 || (v14 = memcmp(v12, v10, v11)) == 0 )
    {
      if ( v13 == v9 )
        goto LABEL_43;
LABEL_8:
      if ( v13 < v9 )
        goto LABEL_9;
LABEL_43:
      v45 = *((_QWORD *)a1 + 1);
      v46 = sub_1649960(*(_QWORD *)(*((_QWORD *)v79 - 1) + 40LL));
      v48 = v47;
      v49 = v46;
      v51 = sub_1649960(*(_QWORD *)(v45 + 40));
      v52 = v50;
      if ( v50 > v48 )
      {
        if ( !v48 )
          goto LABEL_59;
        v53 = memcmp(v51, v49, v48);
        if ( !v53 )
          goto LABEL_47;
      }
      else if ( !v50 || (v53 = memcmp(v51, v49, v50)) == 0 )
      {
        if ( v52 == v48 )
          goto LABEL_59;
LABEL_47:
        if ( v52 < v48 )
          goto LABEL_48;
LABEL_59:
        v59 = *(_QWORD *)v6;
        v60 = sub_1649960(*(_QWORD *)(*((_QWORD *)v79 - 1) + 40LL));
        v62 = v61;
        v63 = v60;
        v65 = sub_1649960(*(_QWORD *)(v59 + 40));
        v66 = v64;
        if ( v62 < v64 )
        {
          if ( !v62 )
            goto LABEL_14;
          v67 = memcmp(v65, v63, v62);
          if ( !v67 )
          {
LABEL_63:
            if ( v62 <= v66 )
              goto LABEL_14;
            goto LABEL_64;
          }
        }
        else if ( !v64 || (v67 = memcmp(v65, v63, v64)) == 0 )
        {
          if ( v62 == v66 )
            goto LABEL_14;
          goto LABEL_63;
        }
        if ( v67 >= 0 )
          goto LABEL_14;
        goto LABEL_64;
      }
      if ( v53 < 0 )
        goto LABEL_48;
      goto LABEL_59;
    }
    if ( v14 >= 0 )
      goto LABEL_43;
LABEL_9:
    v15 = *(_QWORD *)v6;
    v16 = sub_1649960(*(_QWORD *)(*((_QWORD *)v79 - 1) + 40LL));
    v18 = v17;
    v19 = v16;
    v21 = sub_1649960(*(_QWORD *)(v15 + 40));
    v22 = v20;
    if ( v20 > v18 )
    {
      if ( !v18 )
        goto LABEL_68;
      v23 = memcmp(v21, v19, v18);
      if ( !v23 )
        goto LABEL_13;
LABEL_67:
      if ( v23 < 0 )
        goto LABEL_14;
LABEL_68:
      v69 = *((_QWORD *)a1 + 1);
      v70 = sub_1649960(*(_QWORD *)(*((_QWORD *)v79 - 1) + 40LL));
      v72 = v71;
      v74 = sub_1649960(*(_QWORD *)(v69 + 40));
      v75 = v73;
      if ( v73 > v72 )
      {
        if ( !v72 )
          goto LABEL_48;
        v76 = memcmp(v74, v70, v72);
        if ( !v76 )
        {
LABEL_72:
          if ( v75 >= v72 )
            goto LABEL_48;
LABEL_64:
          v68 = *(_QWORD *)a1;
          *(_QWORD *)a1 = *((_QWORD *)v79 - 1);
          *((_QWORD *)v79 - 1) = v68;
          v25 = *(_QWORD *)a1;
          v26 = *((_QWORD *)a1 + 1);
          goto LABEL_15;
        }
      }
      else if ( !v73 || (v76 = memcmp(v74, v70, v73)) == 0 )
      {
        if ( v75 != v72 )
          goto LABEL_72;
LABEL_48:
        v26 = *(_QWORD *)a1;
        v25 = *((_QWORD *)a1 + 1);
        *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
        *(_QWORD *)a1 = v25;
        goto LABEL_15;
      }
      if ( v76 < 0 )
        goto LABEL_64;
      goto LABEL_48;
    }
    if ( v20 )
    {
      v23 = memcmp(v21, v19, v20);
      if ( v23 )
        goto LABEL_67;
    }
    if ( v22 == v18 )
      goto LABEL_68;
LABEL_13:
    if ( v22 >= v18 )
      goto LABEL_68;
LABEL_14:
    v24 = *(_QWORD *)a1;
    *(_QWORD *)a1 = *(_QWORD *)v6;
    *(_QWORD *)v6 = v24;
    v25 = *(_QWORD *)a1;
    v26 = *((_QWORD *)a1 + 1);
LABEL_15:
    v83 = v77;
    s2 = v79;
    while ( 1 )
    {
      v82 = v83;
      v29 = sub_1649960(*(_QWORD *)(v25 + 40));
      v31 = v30;
      v33 = sub_1649960(*(_QWORD *)(v26 + 40));
      v28 = v32;
      if ( v32 > v31 )
        break;
      if ( v32 )
      {
        v80 = v32;
        v27 = memcmp(v33, v29, v32);
        v28 = v80;
        if ( v27 )
          goto LABEL_24;
      }
      if ( v28 == v31 )
        goto LABEL_25;
LABEL_19:
      if ( v28 >= v31 )
        goto LABEL_25;
LABEL_20:
      v25 = *(_QWORD *)a1;
      v26 = *((_QWORD *)v83 + 1);
      v83 += 8;
    }
    if ( !v31 )
      goto LABEL_25;
    v81 = v32;
    v27 = memcmp(v33, v29, v31);
    v28 = v81;
    if ( !v27 )
      goto LABEL_19;
LABEL_24:
    if ( v27 < 0 )
      goto LABEL_20;
LABEL_25:
    v34 = s2;
    do
    {
      while ( 1 )
      {
        v35 = *((_QWORD *)v34 - 1);
        v36 = *(_QWORD *)a1;
        v34 -= 8;
        s2 = v34;
        v37 = sub_1649960(*(_QWORD *)(v35 + 40));
        v39 = v38;
        v41 = sub_1649960(*(_QWORD *)(v36 + 40));
        v42 = v40;
        if ( v40 <= v39 )
          break;
        if ( !v39 )
        {
LABEL_31:
          if ( v83 >= v34 )
            goto LABEL_37;
LABEL_32:
          v44 = *(_QWORD *)v83;
          *(_QWORD *)v83 = *(_QWORD *)v34;
          *(_QWORD *)v34 = v44;
          goto LABEL_20;
        }
        v43 = memcmp(v41, v37, v39);
        if ( v43 )
          goto LABEL_35;
LABEL_30:
        if ( v42 >= v39 )
          goto LABEL_31;
      }
      if ( !v40 || (v43 = memcmp(v41, v37, v40)) == 0 )
      {
        if ( v42 == v39 )
          goto LABEL_31;
        goto LABEL_30;
      }
LABEL_35:
      ;
    }
    while ( v43 < 0 );
    if ( v83 < v34 )
      goto LABEL_32;
LABEL_37:
    sub_1A91840(v83, v79, v78);
    result = v83 - a1;
    if ( v83 - a1 > 128 )
    {
      if ( v78 )
      {
        v79 = v83;
        continue;
      }
LABEL_50:
      v54 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_1A91360((__int64)a1, i, v54, *(_QWORD *)&a1[8 * i]);
        if ( !i )
          break;
      }
      v56 = (__int64 *)(v82 - 8);
      do
      {
        v57 = *v56;
        v58 = (char *)v56-- - a1;
        v56[1] = *(_QWORD *)a1;
        result = sub_1A91360((__int64)a1, 0, v58 >> 3, v57);
      }
      while ( v58 > 8 );
    }
    return result;
  }
}
