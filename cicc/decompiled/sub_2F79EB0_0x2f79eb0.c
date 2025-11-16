// Function: sub_2F79EB0
// Address: 0x2f79eb0
//
signed __int64 __fastcall sub_2F79EB0(__int64 **a1, __int64 **a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 *v4; // r14
  __int64 **v5; // r13
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const char *v9; // r15
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r14
  bool v13; // cc
  size_t v14; // rdx
  int v15; // eax
  __int64 *v16; // r14
  const char *v17; // rax
  size_t v18; // rdx
  size_t v19; // rbx
  const char *v20; // r15
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r14
  size_t v24; // rdx
  int v25; // eax
  __int64 *v26; // r13
  const char *v27; // rax
  size_t v28; // rdx
  size_t v29; // rbx
  const char *v30; // r14
  const char *v31; // rax
  size_t v32; // rdx
  size_t v33; // r13
  size_t v34; // rdx
  int v35; // eax
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // r14
  __int64 *v39; // r14
  const char *v40; // rax
  size_t v41; // rdx
  size_t v42; // rbx
  const char *v43; // r15
  const char *v44; // rax
  size_t v45; // rdx
  size_t v46; // r14
  size_t v47; // rdx
  int v48; // eax
  __int64 *v49; // rbx
  const char *v50; // rax
  size_t v51; // rdx
  size_t v52; // r14
  const char *v53; // r15
  const char *v54; // rax
  size_t v55; // rdx
  size_t v56; // rbx
  int v57; // eax
  __int64 *v58; // rax
  __int64 **v59; // r15
  const char *v60; // rax
  size_t v61; // rdx
  size_t v62; // rbx
  const char *v63; // rax
  size_t v64; // rdx
  size_t v65; // r14
  int v66; // eax
  __int64 *v67; // rax
  __int64 *v68; // r14
  const char *v69; // rax
  size_t v70; // rdx
  size_t v71; // rbx
  const char *v72; // r13
  const char *v73; // rax
  size_t v74; // rdx
  size_t v75; // r14
  size_t v76; // rdx
  int v77; // eax
  __int64 *v78; // rax
  __int64 v79; // rbx
  __int64 v80; // r13
  __int64 *v81; // rcx
  __int64 v82; // [rsp+8h] [rbp-58h]
  __int64 **v83; // [rsp+10h] [rbp-50h]
  __int64 **v84; // [rsp+18h] [rbp-48h]
  const char *s2; // [rsp+20h] [rbp-40h]
  __int64 **i; // [rsp+28h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v83 = a2;
  v82 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v84 = a2;
    goto LABEL_62;
  }
  while ( 2 )
  {
    v4 = a1[1];
    --v82;
    v5 = &a1[result >> 4];
    v6 = sub_BD5D20(**v5);
    v8 = v7;
    v9 = v6;
    v10 = sub_BD5D20(*v4);
    v12 = v11;
    v13 = v11 <= v8;
    v14 = v8;
    if ( v13 )
      v14 = v12;
    if ( v14 )
    {
      v15 = memcmp(v10, v9, v14);
      if ( v15 )
      {
        if ( v15 < 0 )
          goto LABEL_9;
LABEL_23:
        v39 = a1[1];
        v40 = sub_BD5D20(**(v83 - 1));
        v42 = v41;
        v43 = v40;
        v44 = sub_BD5D20(*v39);
        v46 = v45;
        v13 = v45 <= v42;
        v47 = v42;
        if ( v13 )
          v47 = v46;
        if ( v47 && (v48 = memcmp(v44, v43, v47)) != 0 )
        {
          if ( v48 < 0 )
            goto LABEL_60;
        }
        else if ( v46 != v42 && v46 < v42 )
        {
          goto LABEL_60;
        }
        v49 = *v5;
        v50 = sub_BD5D20(**(v83 - 1));
        v52 = v51;
        v53 = v50;
        v54 = sub_BD5D20(*v49);
        v56 = v55;
        if ( v52 <= v55 )
          v55 = v52;
        if ( v55 && (v57 = memcmp(v54, v53, v55)) != 0 )
        {
          if ( v57 < 0 )
            goto LABEL_21;
        }
        else if ( v52 != v56 && v52 > v56 )
        {
          goto LABEL_21;
        }
        goto LABEL_35;
      }
    }
    if ( v12 == v8 || v12 >= v8 )
      goto LABEL_23;
LABEL_9:
    v16 = *v5;
    v17 = sub_BD5D20(**(v83 - 1));
    v19 = v18;
    v20 = v17;
    v21 = sub_BD5D20(*v16);
    v23 = v22;
    v13 = v22 <= v19;
    v24 = v19;
    if ( v13 )
      v24 = v23;
    if ( !v24 || (v25 = memcmp(v21, v20, v24)) == 0 )
    {
      if ( v23 == v19 || v23 >= v19 )
        goto LABEL_15;
LABEL_35:
      v58 = *a1;
      *a1 = *v5;
      *v5 = v58;
      v37 = *a1;
      v38 = a1[1];
      goto LABEL_36;
    }
    if ( v25 < 0 )
      goto LABEL_35;
LABEL_15:
    v26 = a1[1];
    v27 = sub_BD5D20(**(v83 - 1));
    v29 = v28;
    v30 = v27;
    v31 = sub_BD5D20(*v26);
    v33 = v32;
    v13 = v32 <= v29;
    v34 = v29;
    if ( v13 )
      v34 = v33;
    if ( !v34 || (v35 = memcmp(v31, v30, v34)) == 0 )
    {
      if ( v33 != v29 && v33 < v29 )
        goto LABEL_21;
LABEL_60:
      v38 = *a1;
      v37 = a1[1];
      a1[1] = *a1;
      *a1 = v37;
      goto LABEL_36;
    }
    if ( v35 >= 0 )
      goto LABEL_60;
LABEL_21:
    v36 = *a1;
    *a1 = *(v83 - 1);
    *(v83 - 1) = v36;
    v37 = *a1;
    v38 = a1[1];
LABEL_36:
    v59 = v83;
    for ( i = a1 + 1; ; ++i )
    {
      v84 = i;
      v60 = sub_BD5D20(*v37);
      v62 = v61;
      s2 = v60;
      v63 = sub_BD5D20(*v38);
      v65 = v64;
      if ( v62 <= v64 )
        v64 = v62;
      if ( !v64 )
        break;
      v66 = memcmp(v63, s2, v64);
      if ( !v66 )
        break;
      if ( v66 >= 0 )
        goto LABEL_44;
LABEL_37:
      v37 = *a1;
      v38 = i[1];
    }
    if ( v62 != v65 && v62 > v65 )
      goto LABEL_37;
    do
    {
      while ( 1 )
      {
LABEL_44:
        v67 = *(v59 - 1);
        v68 = *a1;
        --v59;
        v69 = sub_BD5D20(*v67);
        v71 = v70;
        v72 = v69;
        v73 = sub_BD5D20(*v68);
        v75 = v74;
        v13 = v74 <= v71;
        v76 = v71;
        if ( v13 )
          v76 = v75;
        if ( v76 )
        {
          v77 = memcmp(v73, v72, v76);
          if ( v77 )
            break;
        }
        if ( v75 == v71 || v75 >= v71 )
        {
          if ( i >= v59 )
            goto LABEL_54;
LABEL_51:
          v78 = *i;
          *i = *v59;
          *v59 = v78;
          goto LABEL_37;
        }
      }
    }
    while ( v77 < 0 );
    if ( i < v59 )
      goto LABEL_51;
LABEL_54:
    sub_2F79EB0(i, v83, v82);
    result = (char *)i - (char *)a1;
    if ( (char *)i - (char *)a1 > 128 )
    {
      if ( v82 )
      {
        v83 = i;
        continue;
      }
LABEL_62:
      v79 = result >> 3;
      v80 = ((result >> 3) - 2) >> 1;
      sub_2F79CB0((__int64)a1, v80, result >> 3, a1[v80]);
      do
      {
        --v80;
        sub_2F79CB0((__int64)a1, v80, v79, a1[v80]);
      }
      while ( v80 );
      do
      {
        v81 = *--v84;
        *v84 = *a1;
        result = (signed __int64)sub_2F79CB0((__int64)a1, 0, v84 - a1, v81);
      }
      while ( (char *)v84 - (char *)a1 > 8 );
    }
    return result;
  }
}
