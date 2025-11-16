// Function: sub_28E6000
// Address: 0x28e6000
//
__int64 __fastcall sub_28E6000(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  char *v5; // r13
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
  __int64 v16; // r14
  const char *v17; // rax
  size_t v18; // rdx
  size_t v19; // rbx
  const char *v20; // r15
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r14
  size_t v24; // rdx
  int v25; // eax
  __int64 v26; // r13
  const char *v27; // rax
  size_t v28; // rdx
  size_t v29; // rbx
  const char *v30; // r14
  const char *v31; // rax
  size_t v32; // rdx
  size_t v33; // r13
  size_t v34; // rdx
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // r14
  const char *v40; // rax
  size_t v41; // rdx
  size_t v42; // rbx
  const char *v43; // r15
  const char *v44; // rax
  size_t v45; // rdx
  size_t v46; // r14
  size_t v47; // rdx
  int v48; // eax
  __int64 v49; // rbx
  const char *v50; // rax
  size_t v51; // rdx
  size_t v52; // r14
  const char *v53; // r15
  const char *v54; // rax
  size_t v55; // rdx
  size_t v56; // rbx
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  const char *v60; // rax
  size_t v61; // rdx
  size_t v62; // rbx
  const char *v63; // rax
  size_t v64; // rdx
  size_t v65; // r15
  size_t v66; // rdx
  int v67; // eax
  char *v68; // r15
  __int64 v69; // rax
  __int64 v70; // r13
  const char *v71; // rax
  size_t v72; // rdx
  size_t v73; // rbx
  const char *v74; // r14
  const char *v75; // rax
  size_t v76; // rdx
  size_t v77; // r13
  int v78; // eax
  __int64 v79; // r13
  __int64 i; // rbx
  __int64 *v81; // r15
  __int64 v82; // rcx
  __int64 v83; // rbx
  __int64 v84; // [rsp+10h] [rbp-60h]
  char *v85; // [rsp+18h] [rbp-58h]
  char *v86; // [rsp+20h] [rbp-50h]
  const char *s2; // [rsp+28h] [rbp-48h]
  char *v88; // [rsp+30h] [rbp-40h]
  char *v89; // [rsp+38h] [rbp-38h]

  result = a2 - a1;
  v84 = a3;
  v85 = a2;
  if ( a2 - a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v86 = a2;
    goto LABEL_63;
  }
  while ( 2 )
  {
    v4 = *((_QWORD *)a1 + 1);
    --v84;
    v5 = &a1[8 * ((__int64)(((v85 - a1) >> 3) + ((unsigned __int64)(v85 - a1) >> 63)) >> 1)];
    v6 = sub_BD5D20(*(_QWORD *)(*(_QWORD *)v5 + 40LL));
    v8 = v7;
    v9 = v6;
    v10 = sub_BD5D20(*(_QWORD *)(v4 + 40));
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
        v39 = *((_QWORD *)a1 + 1);
        v40 = sub_BD5D20(*(_QWORD *)(*((_QWORD *)v85 - 1) + 40LL));
        v42 = v41;
        v43 = v40;
        v44 = sub_BD5D20(*(_QWORD *)(v39 + 40));
        v46 = v45;
        v13 = v45 <= v42;
        v47 = v42;
        if ( v13 )
          v47 = v46;
        if ( v47 && (v48 = memcmp(v44, v43, v47)) != 0 )
        {
          if ( v48 < 0 )
            goto LABEL_61;
        }
        else if ( v46 != v42 && v46 < v42 )
        {
          goto LABEL_61;
        }
        v49 = *(_QWORD *)v5;
        v50 = sub_BD5D20(*(_QWORD *)(*((_QWORD *)v85 - 1) + 40LL));
        v52 = v51;
        v53 = v50;
        v54 = sub_BD5D20(*(_QWORD *)(v49 + 40));
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
    v16 = *(_QWORD *)v5;
    v17 = sub_BD5D20(*(_QWORD *)(*((_QWORD *)v85 - 1) + 40LL));
    v19 = v18;
    v20 = v17;
    v21 = sub_BD5D20(*(_QWORD *)(v16 + 40));
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
      v58 = *(_QWORD *)a1;
      *(_QWORD *)a1 = *(_QWORD *)v5;
      *(_QWORD *)v5 = v58;
      v37 = *(_QWORD *)a1;
      v38 = *((_QWORD *)a1 + 1);
      goto LABEL_36;
    }
    if ( v25 < 0 )
      goto LABEL_35;
LABEL_15:
    v26 = *((_QWORD *)a1 + 1);
    v27 = sub_BD5D20(*(_QWORD *)(*((_QWORD *)v85 - 1) + 40LL));
    v29 = v28;
    v30 = v27;
    v31 = sub_BD5D20(*(_QWORD *)(v26 + 40));
    v33 = v32;
    v13 = v32 <= v29;
    v34 = v29;
    if ( v13 )
      v34 = v33;
    if ( !v34 || (v35 = memcmp(v31, v30, v34)) == 0 )
    {
      if ( v33 != v29 && v33 < v29 )
        goto LABEL_21;
LABEL_61:
      v38 = *(_QWORD *)a1;
      v37 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
      *(_QWORD *)a1 = v37;
      goto LABEL_36;
    }
    if ( v35 >= 0 )
      goto LABEL_61;
LABEL_21:
    v36 = *(_QWORD *)a1;
    *(_QWORD *)a1 = *((_QWORD *)v85 - 1);
    *((_QWORD *)v85 - 1) = v36;
    v37 = *(_QWORD *)a1;
    v38 = *((_QWORD *)a1 + 1);
LABEL_36:
    v88 = a1 + 8;
    v89 = v85;
    while ( 1 )
    {
      v86 = v88;
      v60 = sub_BD5D20(*(_QWORD *)(v37 + 40));
      v62 = v61;
      s2 = v60;
      v63 = sub_BD5D20(*(_QWORD *)(v38 + 40));
      v65 = v64;
      v13 = v64 <= v62;
      v66 = v62;
      if ( v13 )
        v66 = v65;
      if ( !v66 )
        break;
      v67 = memcmp(v63, s2, v66);
      if ( !v67 )
        break;
      if ( v67 >= 0 )
        goto LABEL_48;
LABEL_41:
      v37 = *(_QWORD *)a1;
      v38 = *((_QWORD *)v88 + 1);
      v88 += 8;
    }
    if ( v65 != v62 && v65 < v62 )
      goto LABEL_41;
LABEL_48:
    v68 = v89;
    do
    {
      while ( 1 )
      {
        v69 = *((_QWORD *)v68 - 1);
        v70 = *(_QWORD *)a1;
        v68 -= 8;
        v89 = v68;
        v71 = sub_BD5D20(*(_QWORD *)(v69 + 40));
        v73 = v72;
        v74 = v71;
        v75 = sub_BD5D20(*(_QWORD *)(v70 + 40));
        v77 = v76;
        if ( v73 <= v76 )
          v76 = v73;
        if ( v76 )
        {
          v78 = memcmp(v75, v74, v76);
          if ( v78 )
            break;
        }
        if ( v73 == v77 || v73 <= v77 )
        {
          if ( v88 >= v68 )
            goto LABEL_55;
LABEL_40:
          v59 = *(_QWORD *)v88;
          *(_QWORD *)v88 = *(_QWORD *)v68;
          *(_QWORD *)v68 = v59;
          goto LABEL_41;
        }
      }
    }
    while ( v78 < 0 );
    if ( v88 < v68 )
      goto LABEL_40;
LABEL_55:
    sub_28E6000(v88, v85, v84);
    result = v88 - a1;
    if ( v88 - a1 > 128 )
    {
      if ( v84 )
      {
        v85 = v88;
        continue;
      }
LABEL_63:
      v79 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_28E5C30((__int64)a1, i, v79, *(_QWORD *)&a1[8 * i]);
        if ( !i )
          break;
      }
      v81 = (__int64 *)(v86 - 8);
      do
      {
        v82 = *v81;
        v83 = (char *)v81-- - a1;
        v81[1] = *(_QWORD *)a1;
        result = sub_28E5C30((__int64)a1, 0, v83 >> 3, v82);
      }
      while ( v83 > 8 );
    }
    return result;
  }
}
