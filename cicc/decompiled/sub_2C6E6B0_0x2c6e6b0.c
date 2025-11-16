// Function: sub_2C6E6B0
// Address: 0x2c6e6b0
//
__int64 __fastcall sub_2C6E6B0(__int64 *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r14
  __int64 *v6; // r13
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // rbx
  const char *v10; // r15
  const char *v11; // rax
  size_t v12; // rdx
  size_t v13; // r14
  bool v14; // cc
  size_t v15; // rdx
  int v16; // eax
  __int64 v17; // r14
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // rbx
  const char *v21; // r15
  const char *v22; // rax
  size_t v23; // rdx
  size_t v24; // r14
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // r13
  const char *v28; // rax
  size_t v29; // rdx
  size_t v30; // rbx
  const char *v31; // r14
  const char *v32; // rax
  size_t v33; // rdx
  size_t v34; // r13
  size_t v35; // rdx
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // r15
  __int64 v40; // r14
  const char *v41; // rax
  size_t v42; // rdx
  size_t v43; // rbx
  const char *v44; // r15
  const char *v45; // rax
  size_t v46; // rdx
  size_t v47; // r14
  size_t v48; // rdx
  int v49; // eax
  __int64 v50; // rbx
  const char *v51; // rax
  size_t v52; // rdx
  size_t v53; // r14
  const char *v54; // r15
  const char *v55; // rax
  size_t v56; // rdx
  size_t v57; // rbx
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rax
  const char *v61; // rax
  size_t v62; // rdx
  size_t v63; // rbx
  const char *v64; // rax
  size_t v65; // rdx
  size_t v66; // r15
  size_t v67; // rdx
  int v68; // eax
  char *v69; // r15
  __int64 v70; // r13
  __int64 v71; // rdi
  const char *v72; // rax
  size_t v73; // rdx
  size_t v74; // rbx
  const char *v75; // r14
  const char *v76; // rax
  size_t v77; // rdx
  size_t v78; // r13
  int v79; // eax
  __int64 v80; // r13
  __int64 i; // rbx
  __int64 *v82; // r15
  __int64 v83; // rcx
  __int64 v84; // rbx
  char *v85; // [rsp+8h] [rbp-68h]
  __int64 v86; // [rsp+10h] [rbp-60h]
  char *v87; // [rsp+18h] [rbp-58h]
  char *v88; // [rsp+20h] [rbp-50h]
  const char *s2; // [rsp+28h] [rbp-48h]
  char *v90; // [rsp+30h] [rbp-40h]
  char *v91; // [rsp+38h] [rbp-38h]

  result = a2 - (char *)a1;
  v86 = a3;
  v87 = a2;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v88 = a2;
    goto LABEL_64;
  }
  v85 = (char *)(a1 + 1);
  while ( 2 )
  {
    v5 = a1[1];
    --v86;
    v6 = &a1[(__int64)(((v87 - (char *)a1) >> 3) + ((unsigned __int64)(v87 - (char *)a1) >> 63)) >> 1];
    v7 = sub_BD5D20(*v6);
    v9 = v8;
    v10 = v7;
    v11 = sub_BD5D20(v5);
    v13 = v12;
    v14 = v12 <= v9;
    v15 = v9;
    if ( v14 )
      v15 = v13;
    if ( v15 )
    {
      v16 = memcmp(v11, v10, v15);
      if ( v16 )
      {
        if ( v16 < 0 )
          goto LABEL_10;
LABEL_24:
        v40 = a1[1];
        v41 = sub_BD5D20(*((_QWORD *)v87 - 1));
        v43 = v42;
        v44 = v41;
        v45 = sub_BD5D20(v40);
        v47 = v46;
        v14 = v46 <= v43;
        v48 = v43;
        if ( v14 )
          v48 = v47;
        if ( v48 && (v49 = memcmp(v45, v44, v48)) != 0 )
        {
          if ( v49 < 0 )
            goto LABEL_62;
        }
        else if ( v47 != v43 && v47 < v43 )
        {
          goto LABEL_62;
        }
        v50 = *v6;
        v51 = sub_BD5D20(*((_QWORD *)v87 - 1));
        v53 = v52;
        v54 = v51;
        v55 = sub_BD5D20(v50);
        v57 = v56;
        if ( v53 <= v56 )
          v56 = v53;
        if ( v56 && (v58 = memcmp(v55, v54, v56)) != 0 )
        {
          if ( v58 < 0 )
            goto LABEL_22;
        }
        else if ( v53 != v57 && v53 > v57 )
        {
          goto LABEL_22;
        }
        goto LABEL_36;
      }
    }
    if ( v13 == v9 || v13 >= v9 )
      goto LABEL_24;
LABEL_10:
    v17 = *v6;
    v18 = sub_BD5D20(*((_QWORD *)v87 - 1));
    v20 = v19;
    v21 = v18;
    v22 = sub_BD5D20(v17);
    v24 = v23;
    v14 = v23 <= v20;
    v25 = v20;
    if ( v14 )
      v25 = v24;
    if ( !v25 || (v26 = memcmp(v22, v21, v25)) == 0 )
    {
      if ( v24 == v20 || v24 >= v20 )
        goto LABEL_16;
LABEL_36:
      v59 = *a1;
      *a1 = *v6;
      *v6 = v59;
      v38 = *a1;
      v39 = a1[1];
      goto LABEL_37;
    }
    if ( v26 < 0 )
      goto LABEL_36;
LABEL_16:
    v27 = a1[1];
    v28 = sub_BD5D20(*((_QWORD *)v87 - 1));
    v30 = v29;
    v31 = v28;
    v32 = sub_BD5D20(v27);
    v34 = v33;
    v14 = v33 <= v30;
    v35 = v30;
    if ( v14 )
      v35 = v34;
    if ( !v35 || (v36 = memcmp(v32, v31, v35)) == 0 )
    {
      if ( v34 != v30 && v34 < v30 )
        goto LABEL_22;
LABEL_62:
      v39 = *a1;
      v38 = a1[1];
      a1[1] = *a1;
      *a1 = v38;
      goto LABEL_37;
    }
    if ( v36 >= 0 )
      goto LABEL_62;
LABEL_22:
    v37 = *a1;
    *a1 = *((_QWORD *)v87 - 1);
    *((_QWORD *)v87 - 1) = v37;
    v38 = *a1;
    v39 = a1[1];
LABEL_37:
    v90 = v85;
    v91 = v87;
    while ( 1 )
    {
      v88 = v90;
      v61 = sub_BD5D20(v38);
      v63 = v62;
      s2 = v61;
      v64 = sub_BD5D20(v39);
      v66 = v65;
      v14 = v65 <= v63;
      v67 = v63;
      if ( v14 )
        v67 = v66;
      if ( !v67 )
        break;
      v68 = memcmp(v64, s2, v67);
      if ( !v68 )
        break;
      if ( v68 >= 0 )
        goto LABEL_49;
LABEL_42:
      v38 = *a1;
      v39 = *((_QWORD *)v90 + 1);
      v90 += 8;
    }
    if ( v66 != v63 && v66 < v63 )
      goto LABEL_42;
LABEL_49:
    v69 = v91;
    do
    {
      while ( 1 )
      {
        v70 = *a1;
        v71 = *((_QWORD *)v69 - 1);
        v69 -= 8;
        v91 = v69;
        v72 = sub_BD5D20(v71);
        v74 = v73;
        v75 = v72;
        v76 = sub_BD5D20(v70);
        v78 = v77;
        if ( v74 <= v77 )
          v77 = v74;
        if ( v77 )
        {
          v79 = memcmp(v76, v75, v77);
          if ( v79 )
            break;
        }
        if ( v74 == v78 || v74 <= v78 )
        {
          if ( v90 >= v69 )
            goto LABEL_56;
LABEL_41:
          v60 = *(_QWORD *)v90;
          *(_QWORD *)v90 = *(_QWORD *)v69;
          *(_QWORD *)v69 = v60;
          goto LABEL_42;
        }
      }
    }
    while ( v79 < 0 );
    if ( v90 < v69 )
      goto LABEL_41;
LABEL_56:
    sub_2C6E6B0(v90, v87, v86);
    result = v90 - (char *)a1;
    if ( v90 - (char *)a1 > 128 )
    {
      if ( v86 )
      {
        v87 = v90;
        continue;
      }
LABEL_64:
      v80 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_2C6E300((__int64)a1, i, v80, a1[i]);
        if ( !i )
          break;
      }
      v82 = (__int64 *)(v88 - 8);
      do
      {
        v83 = *v82;
        v84 = (char *)v82-- - (char *)a1;
        v82[1] = *a1;
        result = sub_2C6E300((__int64)a1, 0, v84 >> 3, v83);
      }
      while ( v84 > 8 );
    }
    return result;
  }
}
