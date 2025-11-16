// Function: sub_2C4F0E0
// Address: 0x2c4f0e0
//
void __fastcall sub_2C4F0E0(unsigned int *a1, unsigned int *a2, __int64 **a3, _BYTE **a4)
{
  unsigned int *v4; // rbx
  _BYTE *v7; // r14
  __int64 v8; // r12
  unsigned int v9; // r15d
  __int64 v10; // r13
  unsigned __int8 *v11; // rdx
  __int64 *v12; // rcx
  char *v13; // r15
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // r15
  unsigned int *v17; // r8
  __int64 v18; // rbx
  unsigned int *v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // ecx
  __int64 v22; // r8
  _BYTE *v23; // r13
  __int64 v24; // r12
  unsigned int *v25; // rcx
  int v26; // r14d
  unsigned __int8 *v27; // rdx
  __int64 *v28; // rdi
  char *v29; // r14
  char v30; // si
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  _QWORD *v34; // rsi
  _QWORD *v35; // rdi
  __int64 v36; // r9
  char **v37; // rax
  char **v38; // r9
  __int64 *v39; // rax
  __int64 *v40; // rax
  bool v41; // zf
  __int64 v42; // rdi
  char **v43; // rsi
  char **v44; // rdi
  __int64 v45; // rdi
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 *v50; // rax
  unsigned int *v51; // [rsp+8h] [rbp-78h]
  _BYTE **v52; // [rsp+10h] [rbp-70h]
  _BYTE **v53; // [rsp+18h] [rbp-68h]
  __int64 **v54; // [rsp+18h] [rbp-68h]
  __int64 **v55; // [rsp+20h] [rbp-60h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+28h] [rbp-58h]
  __int64 v58; // [rsp+28h] [rbp-58h]
  unsigned int *v59; // [rsp+30h] [rbp-50h]
  _BYTE **v60; // [rsp+30h] [rbp-50h]
  _BYTE **v61; // [rsp+30h] [rbp-50h]
  unsigned int v63; // [rsp+40h] [rbp-40h]
  __int64 **v64; // [rsp+40h] [rbp-40h]
  __int64 **v65; // [rsp+40h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v4 = a1 + 2;
  if ( a2 == a1 + 2 )
    return;
LABEL_3:
  v7 = *a4;
  v8 = *v4;
  v9 = *v4;
  v10 = *a1;
  if ( **a4 != 92 )
    goto LABEL_11;
  v11 = (unsigned __int8 *)*((_QWORD *)v7 - 4);
  v12 = *a3;
  if ( (unsigned int)*v11 - 12 > 1 )
    goto LABEL_6;
  v13 = (char *)*((_QWORD *)v7 - 8);
  v14 = *v13;
  if ( *v13 != 92 )
    goto LABEL_6;
  v42 = *v12;
  if ( !*(_BYTE *)(*v12 + 28) )
  {
    v60 = a4;
    v64 = a3;
    v49 = sub_C8CA60(v42, *((_QWORD *)v7 - 8));
    a3 = v64;
    a4 = v60;
    v41 = v49 == 0;
    v14 = *v7;
    v12 = *v64;
    if ( v41 )
    {
LABEL_61:
      v15 = *((_QWORD *)v7 + 9);
      LODWORD(v8) = *(_DWORD *)(v15 + 4 * v8);
      goto LABEL_52;
    }
LABEL_51:
    v15 = *((_QWORD *)v7 + 9);
    LODWORD(v8) = *(_DWORD *)(*((_QWORD *)v13 + 9) + 4LL * *(unsigned int *)(v15 + 4 * v8));
LABEL_52:
    if ( v14 != 92 )
      goto LABEL_10;
    v11 = (unsigned __int8 *)*((_QWORD *)v7 - 4);
    goto LABEL_7;
  }
  v43 = *(char ***)(v42 + 8);
  v44 = &v43[*(unsigned int *)(v42 + 20)];
  if ( v43 != v44 )
  {
    while ( v13 != *v43 )
    {
      if ( v44 == ++v43 )
        goto LABEL_61;
    }
    goto LABEL_51;
  }
LABEL_6:
  v15 = *((_QWORD *)v7 + 9);
  LODWORD(v8) = *(_DWORD *)(v15 + 4 * v8);
LABEL_7:
  if ( (unsigned int)*v11 - 12 > 1 )
    goto LABEL_9;
  v16 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v16 != 92 )
    goto LABEL_9;
  v45 = *v12;
  if ( *(_BYTE *)(*v12 + 28) )
  {
    v46 = *(_QWORD **)(v45 + 8);
    v47 = &v46[*(unsigned int *)(v45 + 20)];
    if ( v46 == v47 )
    {
LABEL_9:
      LODWORD(v10) = *(_DWORD *)(v15 + 4 * v10);
LABEL_10:
      v9 = *v4;
      goto LABEL_11;
    }
    while ( v16 != *v46 )
    {
      if ( v47 == ++v46 )
        goto LABEL_9;
    }
  }
  else
  {
    v61 = a4;
    v65 = a3;
    v50 = sub_C8CA60(v45, *((_QWORD *)v7 - 8));
    v15 = *((_QWORD *)v7 + 9);
    a3 = v65;
    a4 = v61;
    if ( !v50 )
      goto LABEL_9;
  }
  v48 = *(_QWORD *)(v16 + 72);
  v9 = *v4;
  LODWORD(v10) = *(_DWORD *)(v48 + 4LL * *(unsigned int *)(v15 + 4 * v10));
LABEL_11:
  v17 = v4 + 2;
  v63 = v4[1];
  if ( (int)v10 <= (int)v8 )
  {
    v59 = v4 + 2;
    v22 = 4LL * v9;
    while ( 1 )
    {
      v23 = *a4;
      v24 = *(v4 - 2);
      v25 = v4;
      v26 = v9;
      if ( **a4 != 92 )
        goto LABEL_25;
      v27 = (unsigned __int8 *)*((_QWORD *)v23 - 4);
      v28 = *a3;
      if ( (unsigned int)*v27 - 12 > 1 || (v29 = (char *)*((_QWORD *)v23 - 8), v30 = *v29, *v29 != 92) )
      {
LABEL_21:
        v31 = *((_QWORD *)v23 + 9);
        v26 = *(_DWORD *)(v31 + v22);
        goto LABEL_22;
      }
      v36 = *v28;
      if ( *(_BYTE *)(*v28 + 28) )
      {
        v37 = *(char ***)(v36 + 8);
        v38 = &v37[*(unsigned int *)(v36 + 20)];
        if ( v37 == v38 )
          goto LABEL_21;
        while ( v29 != *v37 )
        {
          if ( v38 == ++v37 )
            goto LABEL_43;
        }
      }
      else
      {
        v53 = a4;
        v55 = a3;
        v57 = v22;
        v39 = sub_C8CA60(*v28, *((_QWORD *)v23 - 8));
        a3 = v55;
        v22 = v57;
        a4 = v53;
        v25 = v4;
        v28 = *v55;
        v30 = *v23;
        if ( !v39 )
        {
LABEL_43:
          v31 = *((_QWORD *)v23 + 9);
          v26 = *(_DWORD *)(v31 + v22);
          goto LABEL_40;
        }
      }
      v31 = *((_QWORD *)v23 + 9);
      v26 = *(_DWORD *)(*((_QWORD *)v29 + 9) + 4LL * *(unsigned int *)(v31 + v22));
LABEL_40:
      if ( v30 != 92 )
        goto LABEL_25;
      v27 = (unsigned __int8 *)*((_QWORD *)v23 - 4);
LABEL_22:
      if ( (unsigned int)*v27 - 12 > 1 || (v32 = *((_QWORD *)v23 - 8), *(_BYTE *)v32 != 92) )
      {
LABEL_24:
        LODWORD(v24) = *(_DWORD *)(v31 + 4 * v24);
        goto LABEL_25;
      }
      v33 = *v28;
      if ( *(_BYTE *)(v33 + 28) )
      {
        v34 = *(_QWORD **)(v33 + 8);
        v35 = &v34[*(unsigned int *)(v33 + 20)];
        if ( v34 == v35 )
          goto LABEL_24;
        while ( v32 != *v34 )
        {
          if ( v35 == ++v34 )
            goto LABEL_24;
        }
        LODWORD(v24) = *(_DWORD *)(*(_QWORD *)(v32 + 72) + 4LL * *(unsigned int *)(v31 + 4 * v24));
      }
      else
      {
        v51 = v25;
        v52 = a4;
        v54 = a3;
        v56 = v22;
        v58 = *((_QWORD *)v23 - 8);
        v40 = sub_C8CA60(v33, v58);
        v22 = v56;
        v41 = v40 == 0;
        a3 = v54;
        a4 = v52;
        v25 = v51;
        v31 = *((_QWORD *)v23 + 9);
        if ( v41 )
          goto LABEL_24;
        LODWORD(v24) = *(_DWORD *)(*(_QWORD *)(v58 + 72) + 4LL * *(unsigned int *)(v31 + 4 * v24));
      }
LABEL_25:
      v4 -= 2;
      if ( (int)v24 <= v26 )
      {
        v17 = v59;
        *v25 = v9;
        v25[1] = v63;
        if ( a2 == v59 )
          return;
LABEL_15:
        v4 = v17;
        goto LABEL_3;
      }
      v4[2] = *v4;
      v4[3] = v4[1];
    }
  }
  v18 = (char *)v4 - (char *)a1;
  v19 = v17;
  v20 = v18 >> 3;
  if ( v18 > 0 )
  {
    do
    {
      v21 = *(v19 - 4);
      v19 -= 2;
      *v19 = v21;
      v19[1] = *(v19 - 1);
      --v20;
    }
    while ( v20 );
  }
  *a1 = v9;
  a1[1] = v63;
  if ( a2 != v17 )
    goto LABEL_15;
}
