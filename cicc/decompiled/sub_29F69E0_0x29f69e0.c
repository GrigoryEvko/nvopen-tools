// Function: sub_29F69E0
// Address: 0x29f69e0
//
void __fastcall sub_29F69E0(char *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // rbx
  char *v6; // rbx
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // r13
  const char *v10; // r15
  const char *v11; // rax
  size_t v12; // rdx
  size_t v13; // r14
  bool v14; // cc
  size_t v15; // rdx
  int v16; // eax
  const char *v17; // rax
  size_t v18; // rdx
  size_t v19; // r13
  const char *v20; // r15
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r14
  size_t v24; // rdx
  int v25; // eax
  const char *v26; // rax
  size_t v27; // rdx
  size_t v28; // rbx
  const char *v29; // r14
  const char *v30; // rax
  size_t v31; // rdx
  size_t v32; // r13
  size_t v33; // rdx
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdi
  const char *v39; // rax
  size_t v40; // rdx
  size_t v41; // r13
  const char *v42; // r15
  const char *v43; // rax
  size_t v44; // rdx
  size_t v45; // r14
  size_t v46; // rdx
  int v47; // eax
  const char *v48; // rax
  size_t v49; // rdx
  size_t v50; // r14
  const char *v51; // r15
  const char *v52; // rax
  size_t v53; // rdx
  size_t v54; // r13
  int v55; // eax
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 *v59; // r15
  const char *v60; // rax
  size_t v61; // rdx
  size_t v62; // rbx
  const char *v63; // r14
  const char *v64; // rax
  size_t v65; // rdx
  size_t v66; // rcx
  int v67; // eax
  __int64 v68; // rdi
  const char *v69; // rax
  size_t v70; // rdx
  size_t v71; // rbx
  const char *v72; // r13
  const char *v73; // rax
  size_t v74; // rdx
  size_t v75; // r14
  size_t v76; // rdx
  int v77; // eax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rbx
  __int64 v84; // r13
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 *v87; // [rsp+0h] [rbp-60h]
  __int64 v88; // [rsp+8h] [rbp-58h]
  __int64 *v89; // [rsp+10h] [rbp-50h]
  __int64 *v90; // [rsp+18h] [rbp-48h]
  size_t v91; // [rsp+20h] [rbp-40h]
  __int64 *i; // [rsp+28h] [rbp-38h]

  v89 = a2;
  v3 = (char *)a2 - a1;
  v88 = a3;
  if ( v3 <= 256 )
    return;
  v5 = v3;
  if ( !a3 )
  {
    v90 = v89;
    goto LABEL_63;
  }
  v87 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    --v88;
    v6 = &a1[16 * (v5 >> 5)];
    v7 = sub_BD5D20(*((_QWORD *)v6 + 1));
    v9 = v8;
    v10 = v7;
    v11 = sub_BD5D20(*((_QWORD *)a1 + 3));
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
        v39 = sub_BD5D20(*(v89 - 1));
        v41 = v40;
        v42 = v39;
        v43 = sub_BD5D20(*((_QWORD *)a1 + 3));
        v45 = v44;
        v14 = v44 <= v41;
        v46 = v41;
        if ( v14 )
          v46 = v45;
        if ( v46 && (v47 = memcmp(v43, v42, v46)) != 0 )
        {
          if ( v47 < 0 )
            goto LABEL_61;
        }
        else if ( v45 != v41 && v45 < v41 )
        {
          goto LABEL_61;
        }
        v48 = sub_BD5D20(*(v89 - 1));
        v50 = v49;
        v51 = v48;
        v52 = sub_BD5D20(*((_QWORD *)v6 + 1));
        v54 = v53;
        if ( v50 <= v53 )
          v53 = v50;
        if ( v53 && (v55 = memcmp(v52, v51, v53)) != 0 )
        {
          if ( v55 < 0 )
            goto LABEL_22;
        }
        else if ( v50 != v54 && v50 > v54 )
        {
          goto LABEL_22;
        }
        goto LABEL_36;
      }
    }
    if ( v13 == v9 || v13 >= v9 )
      goto LABEL_24;
LABEL_10:
    v17 = sub_BD5D20(*(v89 - 1));
    v19 = v18;
    v20 = v17;
    v21 = sub_BD5D20(*((_QWORD *)v6 + 1));
    v23 = v22;
    v14 = v22 <= v19;
    v24 = v19;
    if ( v14 )
      v24 = v23;
    if ( !v24 || (v25 = memcmp(v21, v20, v24)) == 0 )
    {
      if ( v23 == v19 || v23 >= v19 )
        goto LABEL_16;
LABEL_36:
      v56 = *(_QWORD *)a1;
      *(_QWORD *)a1 = *(_QWORD *)v6;
      v57 = *((_QWORD *)v6 + 1);
      *(_QWORD *)v6 = v56;
      v58 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = v57;
      *((_QWORD *)v6 + 1) = v58;
      v38 = *((_QWORD *)a1 + 1);
      goto LABEL_37;
    }
    if ( v25 < 0 )
      goto LABEL_36;
LABEL_16:
    v26 = sub_BD5D20(*(v89 - 1));
    v28 = v27;
    v29 = v26;
    v30 = sub_BD5D20(*((_QWORD *)a1 + 3));
    v32 = v31;
    v14 = v31 <= v28;
    v33 = v28;
    if ( v14 )
      v33 = v32;
    if ( !v33 || (v34 = memcmp(v30, v29, v33)) == 0 )
    {
      if ( v32 != v28 && v32 < v28 )
        goto LABEL_22;
LABEL_61:
      v81 = *((_QWORD *)a1 + 2);
      v38 = *((_QWORD *)a1 + 3);
      *((_QWORD *)a1 + 2) = *(_QWORD *)a1;
      v82 = *((_QWORD *)a1 + 1);
      *(_QWORD *)a1 = v81;
      *((_QWORD *)a1 + 1) = v38;
      *((_QWORD *)a1 + 3) = v82;
      goto LABEL_37;
    }
    if ( v34 >= 0 )
      goto LABEL_61;
LABEL_22:
    v35 = *(_QWORD *)a1;
    *(_QWORD *)a1 = *(v89 - 2);
    v36 = *(v89 - 1);
    *(v89 - 2) = v35;
    v37 = *((_QWORD *)a1 + 1);
    *((_QWORD *)a1 + 1) = v36;
    *(v89 - 1) = v37;
    v38 = *((_QWORD *)a1 + 1);
LABEL_37:
    v59 = v89;
    for ( i = v87; ; i += 2 )
    {
      v90 = i;
      v60 = sub_BD5D20(v38);
      v62 = v61;
      v63 = v60;
      v64 = sub_BD5D20(i[1]);
      v66 = v65;
      if ( v62 <= v65 )
        v65 = v62;
      if ( !v65 )
        break;
      v91 = v66;
      v67 = memcmp(v64, v63, v65);
      v66 = v91;
      if ( !v67 )
        break;
      if ( v67 >= 0 )
        goto LABEL_45;
LABEL_38:
      v38 = *((_QWORD *)a1 + 1);
    }
    if ( v62 != v66 && v62 > v66 )
      goto LABEL_38;
    do
    {
      while ( 1 )
      {
LABEL_45:
        v68 = *(v59 - 1);
        v59 -= 2;
        v69 = sub_BD5D20(v68);
        v71 = v70;
        v72 = v69;
        v73 = sub_BD5D20(*((_QWORD *)a1 + 1));
        v75 = v74;
        v14 = v74 <= v71;
        v76 = v71;
        if ( v14 )
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
            goto LABEL_55;
LABEL_52:
          v78 = *i;
          *i = *v59;
          v79 = v59[1];
          *v59 = v78;
          v80 = i[1];
          i[1] = v79;
          v59[1] = v80;
          goto LABEL_38;
        }
      }
    }
    while ( v77 < 0 );
    if ( i < v59 )
      goto LABEL_52;
LABEL_55:
    v5 = (char *)i - a1;
    sub_29F69E0(i, v89, v88);
    if ( (char *)i - a1 > 256 )
    {
      if ( v88 )
      {
        v89 = i;
        continue;
      }
LABEL_63:
      v83 = v5 >> 4;
      v84 = (v83 - 2) >> 1;
      sub_29F4F20((__int64)a1, v84, v83, *(_QWORD *)&a1[16 * v84], *(_QWORD *)&a1[16 * v84 + 8]);
      do
      {
        --v84;
        sub_29F4F20((__int64)a1, v84, v83, *(_QWORD *)&a1[16 * v84], *(_QWORD *)&a1[16 * v84 + 8]);
      }
      while ( v84 );
      do
      {
        v90 -= 2;
        v85 = *v90;
        v86 = v90[1];
        *v90 = *(_QWORD *)a1;
        v90[1] = *((_QWORD *)a1 + 1);
        sub_29F4F20((__int64)a1, 0, ((char *)v90 - a1) >> 4, v85, v86);
      }
      while ( (char *)v90 - a1 > 16 );
    }
    break;
  }
}
