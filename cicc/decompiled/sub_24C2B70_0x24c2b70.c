// Function: sub_24C2B70
// Address: 0x24c2b70
//
signed __int64 __fastcall sub_24C2B70(_QWORD *a1, char *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 v6; // r10
  unsigned int v7; // ebx
  __int64 v8; // r15
  unsigned int v9; // r13d
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rcx
  __int64 v13; // r14
  unsigned int v14; // r13d
  __int64 v15; // r8
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  __int64 *v18; // rax
  __int64 v19; // r14
  unsigned int v20; // ebx
  __int64 *v21; // r13
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // rax
  __int64 v24; // r15
  unsigned int v25; // r12d
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  unsigned int v28; // ebx
  int v29; // eax
  unsigned int v30; // ebx
  __int64 v31; // rsi
  int v32; // eax
  int v33; // eax
  unsigned int v34; // edx
  int v35; // eax
  int v36; // eax
  __int64 v37; // rcx
  int v38; // eax
  unsigned int v39; // ebx
  __int64 v40; // r14
  unsigned __int64 v41; // rax
  _QWORD *v42; // rsi
  __int64 v43; // rdi
  int v44; // eax
  unsigned int v45; // edi
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // r12
  int v48; // eax
  unsigned __int64 v49; // rax
  bool v50; // cc
  int v51; // eax
  __int64 v52; // rbx
  __int64 v53; // r12
  __int64 *v54; // rbx
  __int64 v55; // rcx
  unsigned __int64 v56; // rcx
  int v57; // eax
  unsigned int v58; // r13d
  unsigned __int64 v59; // rax
  __int64 *v60; // rax
  int v61; // eax
  unsigned int v62; // edi
  int v63; // eax
  unsigned int v64; // ebx
  __int64 *v65; // [rsp+8h] [rbp-78h]
  __int64 v66; // [rsp+10h] [rbp-70h]
  __int64 *v67; // [rsp+18h] [rbp-68h]
  __int64 *v69; // [rsp+28h] [rbp-58h]
  unsigned __int64 v70; // [rsp+28h] [rbp-58h]
  __int64 v71; // [rsp+28h] [rbp-58h]
  unsigned __int64 v72; // [rsp+28h] [rbp-58h]
  unsigned __int64 v73; // [rsp+28h] [rbp-58h]
  __int64 v74; // [rsp+30h] [rbp-50h]
  __int64 v75; // [rsp+30h] [rbp-50h]
  __int64 v76; // [rsp+30h] [rbp-50h]
  __int64 v77; // [rsp+30h] [rbp-50h]
  unsigned __int64 v78; // [rsp+30h] [rbp-50h]
  unsigned __int64 v79; // [rsp+30h] [rbp-50h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  __int64 v81; // [rsp+38h] [rbp-48h]
  __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 v83; // [rsp+38h] [rbp-48h]
  __int64 v84; // [rsp+38h] [rbp-48h]
  __int64 v85; // [rsp+38h] [rbp-48h]
  __int64 v86; // [rsp+38h] [rbp-48h]
  __int64 v87; // [rsp+38h] [rbp-48h]
  __int64 *v88; // [rsp+40h] [rbp-40h]
  __int64 *i; // [rsp+40h] [rbp-40h]
  unsigned __int64 v90; // [rsp+40h] [rbp-40h]
  __int64 v91; // [rsp+48h] [rbp-38h]
  unsigned int v92; // [rsp+48h] [rbp-38h]
  unsigned __int64 v93; // [rsp+48h] [rbp-38h]
  unsigned __int64 v94; // [rsp+48h] [rbp-38h]
  unsigned __int64 *v95; // [rsp+48h] [rbp-38h]
  __int64 v96; // [rsp+48h] [rbp-38h]

  result = a2 - (char *)a1;
  v67 = (__int64 *)a2;
  v66 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v69 = (__int64 *)a2;
    goto LABEL_62;
  }
  v65 = a1 + 2;
  while ( 2 )
  {
    --v66;
    v4 = &a1[result >> 4];
    v5 = a1[1];
    v6 = *v4;
    v88 = v4;
    v7 = *(_DWORD *)(v5 + 32);
    v8 = v5 + 24;
    v9 = *(_DWORD *)(*v4 + 32);
    v91 = *v4 + 24;
    if ( v7 > 0x40 )
    {
      v82 = *v4;
      v36 = sub_C444A0(v5 + 24);
      v6 = v82;
      if ( v7 - v36 > 0x40 )
      {
        v37 = *(v67 - 1);
LABEL_41:
        v75 = v6;
        v83 = v37;
        v38 = sub_C444A0(v5 + 24);
        v37 = v83;
        v6 = v75;
        v10 = -1;
        if ( v7 - v38 <= 0x40 )
        {
          v39 = *(_DWORD *)(v83 + 32);
          v40 = v83 + 24;
          v10 = **(_QWORD **)(v5 + 24);
          if ( v39 <= 0x40 )
            goto LABEL_43;
          goto LABEL_47;
        }
LABEL_42:
        v39 = *(_DWORD *)(v37 + 32);
        v40 = v37 + 24;
        if ( v39 <= 0x40 )
        {
LABEL_43:
          v41 = *(_QWORD *)(v37 + 24);
          goto LABEL_44;
        }
LABEL_47:
        v70 = v10;
        v76 = v6;
        v84 = v37;
        v44 = sub_C444A0(v40);
        v37 = v84;
        v6 = v76;
        v45 = v39 - v44;
        v10 = v70;
        v41 = -1;
        if ( v45 <= 0x40 )
        {
          v42 = a1;
          v46 = **(_QWORD **)(v84 + 24);
          v43 = *a1;
          v81 = *a1;
          if ( v46 > v70 )
            goto LABEL_45;
          goto LABEL_49;
        }
LABEL_44:
        v42 = a1;
        v43 = *a1;
        v81 = *a1;
        if ( v41 > v10 )
        {
LABEL_45:
          *v42 = v5;
          v42[1] = v43;
          v19 = *(v67 - 1);
          goto LABEL_12;
        }
LABEL_49:
        if ( v9 <= 0x40 )
        {
          v47 = *(_QWORD *)(v6 + 24);
          if ( v39 <= 0x40 )
          {
LABEL_53:
            v49 = *(_QWORD *)(v37 + 24);
            goto LABEL_54;
          }
        }
        else
        {
          v71 = v6;
          v47 = -1;
          v77 = v37;
          v48 = sub_C444A0(v91);
          v37 = v77;
          v6 = v71;
          if ( v9 - v48 <= 0x40 )
            v47 = **(_QWORD **)(v71 + 24);
          if ( v39 <= 0x40 )
            goto LABEL_53;
        }
        v80 = v6;
        v96 = v37;
        v63 = sub_C444A0(v40);
        v37 = v96;
        v6 = v80;
        v64 = v39 - v63;
        v49 = -1;
        if ( v64 <= 0x40 )
        {
          v18 = a1;
          if ( **(_QWORD **)(v96 + 24) > v47 )
            goto LABEL_55;
          goto LABEL_83;
        }
LABEL_54:
        v50 = v49 <= v47;
        v18 = a1;
        if ( !v50 )
        {
LABEL_55:
          v19 = v81;
          *v18 = v37;
          *(v67 - 1) = v81;
          v5 = *v18;
          v8 = *v18 + 24;
          v81 = v18[1];
          goto LABEL_12;
        }
LABEL_83:
        *v18 = v6;
        *v88 = v81;
        goto LABEL_11;
      }
      v10 = *(_QWORD *)(v5 + 24);
      v11 = *(_QWORD *)v10;
      if ( v9 <= 0x40 )
        goto LABEL_6;
    }
    else
    {
      v10 = *(_QWORD *)(v5 + 24);
      v11 = v10;
      if ( v9 <= 0x40 )
      {
LABEL_6:
        v12 = *(_QWORD *)(v6 + 24);
        v13 = *(v67 - 1);
        if ( v11 < v12 )
          goto LABEL_7;
LABEL_59:
        v37 = v13;
        if ( v7 <= 0x40 )
          goto LABEL_42;
        goto LABEL_41;
      }
    }
    v72 = v10;
    v78 = v11;
    v85 = v6;
    v51 = sub_C444A0(v91);
    v6 = v85;
    v10 = v72;
    v13 = *(v67 - 1);
    if ( v9 - v51 <= 0x40 )
    {
      v12 = **(_QWORD **)(v85 + 24);
      if ( v78 >= v12 )
      {
        v37 = *(v67 - 1);
        if ( v7 <= 0x40 )
          goto LABEL_42;
        goto LABEL_41;
      }
    }
    else
    {
      v12 = -1;
      if ( v78 == -1 )
        goto LABEL_59;
    }
LABEL_7:
    v14 = *(_DWORD *)(v13 + 32);
    v15 = v13 + 24;
    if ( v14 > 0x40 )
    {
      v73 = v10;
      v79 = v12;
      v87 = v6;
      v61 = sub_C444A0(v13 + 24);
      v15 = v13 + 24;
      v6 = v87;
      v62 = v14 - v61;
      v12 = v79;
      v10 = v73;
      v16 = -1;
      if ( v62 <= 0x40 )
        v16 = **(_QWORD **)(v13 + 24);
    }
    else
    {
      v16 = *(_QWORD *)(v13 + 24);
    }
    v17 = *a1;
    if ( v16 > v12 )
    {
      *a1 = v6;
      v18 = a1;
      *v88 = v17;
LABEL_11:
      v5 = *v18;
      v81 = v18[1];
      v8 = *v18 + 24;
      v19 = *(v67 - 1);
      goto LABEL_12;
    }
    v95 = (unsigned __int64 *)v10;
    v56 = v10;
    if ( v7 > 0x40 )
    {
      v86 = v15;
      v57 = sub_C444A0(v5 + 24);
      v15 = v86;
      v56 = -1;
      if ( v7 - v57 <= 0x40 )
        v56 = *v95;
    }
    if ( v14 <= 0x40 )
    {
      v60 = a1;
      if ( *(_QWORD *)(v13 + 24) > v56 )
      {
LABEL_74:
        *v60 = v13;
        v19 = v17;
        *(v67 - 1) = v17;
        v5 = *v60;
        v8 = *v60 + 24;
        v81 = v60[1];
        goto LABEL_12;
      }
    }
    else
    {
      v90 = v56;
      v58 = v14 - sub_C444A0(v15);
      v59 = -1;
      if ( v58 <= 0x40 )
        v59 = **(_QWORD **)(v13 + 24);
      v50 = v59 <= v90;
      v60 = a1;
      if ( !v50 )
        goto LABEL_74;
    }
    *v60 = v5;
    v60[1] = v17;
    v81 = v17;
    v19 = *(v67 - 1);
LABEL_12:
    v20 = *(_DWORD *)(v5 + 32);
    v21 = v67;
    for ( i = v65; ; ++i )
    {
      v69 = i - 1;
      v92 = *(_DWORD *)(v81 + 32);
      if ( v92 > 0x40 )
      {
        v35 = sub_C444A0(v81 + 24);
        v22 = -1;
        if ( v92 - v35 <= 0x40 )
          v22 = **(_QWORD **)(v81 + 24);
      }
      else
      {
        v22 = *(_QWORD *)(v81 + 24);
      }
      if ( v20 > 0x40 )
      {
        v94 = v22;
        v33 = sub_C444A0(v8);
        v22 = v94;
        v34 = v20 - v33;
        v23 = -1;
        if ( v34 <= 0x40 )
          v23 = **(_QWORD **)(v5 + 24);
      }
      else
      {
        v23 = *(_QWORD *)(v5 + 24);
      }
      if ( v23 <= v22 )
        break;
LABEL_29:
      v31 = *i;
      v81 = v31;
    }
    v74 = v8;
    --v21;
    v24 = v5;
    v25 = v20;
    while ( 1 )
    {
      if ( v25 > 0x40 )
      {
        v32 = sub_C444A0(v74);
        v27 = -1;
        if ( v25 - v32 <= 0x40 )
          v27 = **(_QWORD **)(v24 + 24);
      }
      else
      {
        v27 = *(_QWORD *)(v24 + 24);
      }
      v28 = *(_DWORD *)(v19 + 32);
      if ( v28 <= 0x40 )
      {
        v26 = *(_QWORD *)(v19 + 24);
        goto LABEL_20;
      }
      v93 = v27;
      v29 = sub_C444A0(v19 + 24);
      v27 = v93;
      v30 = v28 - v29;
      v26 = -1;
      if ( v30 <= 0x40 )
        break;
LABEL_20:
      if ( v26 <= v27 )
        goto LABEL_27;
LABEL_21:
      v19 = *--v21;
    }
    if ( **(_QWORD **)(v19 + 24) > v93 )
      goto LABEL_21;
LABEL_27:
    if ( v21 > v69 )
    {
      *(i - 1) = v19;
      v19 = *(v21 - 1);
      *v21 = v81;
      v5 = *a1;
      v20 = *(_DWORD *)(*a1 + 32LL);
      v8 = *a1 + 24LL;
      goto LABEL_29;
    }
    sub_24C2B70(v69, v67, v66);
    result = (char *)v69 - (char *)a1;
    if ( (char *)v69 - (char *)a1 > 128 )
    {
      if ( v66 )
      {
        v67 = i - 1;
        continue;
      }
LABEL_62:
      v52 = result >> 3;
      v53 = ((result >> 3) - 2) >> 1;
      sub_24C26D0((__int64)a1, v53, result >> 3, a1[v53]);
      do
      {
        --v53;
        sub_24C26D0((__int64)a1, v53, v52, a1[v53]);
      }
      while ( v53 );
      v54 = v69;
      do
      {
        v55 = *--v54;
        *v54 = *a1;
        result = sub_24C26D0((__int64)a1, 0, v54 - a1, v55);
      }
      while ( (char *)v54 - (char *)a1 > 8 );
    }
    return result;
  }
}
