// Function: sub_2C50850
// Address: 0x2c50850
//
void __fastcall sub_2C50850(
        unsigned int *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 **a7,
        _BYTE **a8)
{
  __int64 v8; // r12
  unsigned int *v9; // r9
  __int64 v10; // rbx
  unsigned int *v11; // r11
  __int64 v12; // r13
  unsigned int *v13; // r15
  unsigned int *v14; // rax
  char *v15; // r11
  int v16; // r9d
  char *v17; // r10
  __int64 v18; // r14
  unsigned int *v19; // r10
  unsigned int *v20; // rax
  char *v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rbx
  _BYTE *v24; // r13
  unsigned __int8 *v25; // rsi
  __int64 *v26; // rdi
  char *v27; // r14
  char v28; // r8
  __int64 v29; // rcx
  __int64 v30; // r14
  unsigned int v31; // eax
  unsigned int v32; // ecx
  unsigned int v33; // eax
  __int64 v34; // rdi
  _QWORD *v35; // rax
  _QWORD *v36; // rsi
  __int64 v37; // r9
  char **v38; // rax
  char **v39; // rcx
  __int64 *v40; // rax
  __int64 *v41; // rax
  unsigned int *v43; // [rsp+8h] [rbp-58h]
  unsigned int *v44; // [rsp+8h] [rbp-58h]
  int v45; // [rsp+10h] [rbp-50h]
  char *v46; // [rsp+10h] [rbp-50h]
  unsigned int *v47; // [rsp+10h] [rbp-50h]
  unsigned int *v48; // [rsp+18h] [rbp-48h]
  int v49; // [rsp+18h] [rbp-48h]
  unsigned int *v50; // [rsp+18h] [rbp-48h]
  char *v51; // [rsp+18h] [rbp-48h]
  char *v52; // [rsp+20h] [rbp-40h]
  unsigned int *v53; // [rsp+28h] [rbp-38h]
  unsigned int *v54; // [rsp+28h] [rbp-38h]

  if ( !a4 )
    return;
  v8 = a5;
  if ( !a5 )
    return;
  v9 = a1;
  v10 = a4;
  v11 = a2;
  if ( a5 + a4 == 2 )
  {
    v19 = a2;
    v21 = (char *)a1;
LABEL_12:
    v22 = *v19;
    v23 = *(unsigned int *)v21;
    v24 = *a8;
    if ( **a8 != 92 )
      goto LABEL_19;
    v25 = (unsigned __int8 *)*((_QWORD *)v24 - 4);
    v26 = *a7;
    if ( (unsigned int)*v25 - 12 > 1 )
      goto LABEL_15;
    v27 = (char *)*((_QWORD *)v24 - 8);
    v28 = *v27;
    if ( *v27 != 92 )
      goto LABEL_15;
    v37 = *v26;
    if ( *(_BYTE *)(*v26 + 28) )
    {
      v38 = *(char ***)(v37 + 8);
      v39 = &v38[*(unsigned int *)(v37 + 20)];
      if ( v38 == v39 )
      {
LABEL_15:
        v29 = *((_QWORD *)v24 + 9);
        LODWORD(v22) = *(_DWORD *)(v29 + 4 * v22);
        goto LABEL_16;
      }
      while ( v27 != *v38 )
      {
        if ( v39 == ++v38 )
          goto LABEL_38;
      }
    }
    else
    {
      v51 = v21;
      v53 = v19;
      v40 = sub_C8CA60(*v26, *((_QWORD *)v24 - 8));
      v19 = v53;
      v21 = v51;
      v28 = *v24;
      v26 = *a7;
      if ( !v40 )
      {
LABEL_38:
        v29 = *((_QWORD *)v24 + 9);
        LODWORD(v22) = *(_DWORD *)(v29 + 4 * v22);
        goto LABEL_35;
      }
    }
    v29 = *((_QWORD *)v24 + 9);
    LODWORD(v22) = *(_DWORD *)(*((_QWORD *)v27 + 9) + 4LL * *(unsigned int *)(v29 + 4 * v22));
LABEL_35:
    if ( v28 != 92 )
      goto LABEL_19;
    v25 = (unsigned __int8 *)*((_QWORD *)v24 - 4);
LABEL_16:
    if ( (unsigned int)*v25 - 12 > 1 )
      goto LABEL_18;
    v30 = *((_QWORD *)v24 - 8);
    if ( *(_BYTE *)v30 != 92 )
      goto LABEL_18;
    v34 = *v26;
    if ( *(_BYTE *)(v34 + 28) )
    {
      v35 = *(_QWORD **)(v34 + 8);
      v36 = &v35[*(unsigned int *)(v34 + 20)];
      if ( v35 == v36 )
      {
LABEL_18:
        LODWORD(v23) = *(_DWORD *)(v29 + 4 * v23);
        goto LABEL_19;
      }
      while ( v30 != *v35 )
      {
        if ( v36 == ++v35 )
          goto LABEL_18;
      }
    }
    else
    {
      v52 = v21;
      v54 = v19;
      v41 = sub_C8CA60(v34, *((_QWORD *)v24 - 8));
      v29 = *((_QWORD *)v24 + 9);
      v19 = v54;
      v21 = v52;
      if ( !v41 )
        goto LABEL_18;
    }
    LODWORD(v23) = *(_DWORD *)(*(_QWORD *)(v30 + 72) + 4LL * *(unsigned int *)(v29 + 4 * v23));
LABEL_19:
    if ( (int)v22 < (int)v23 )
    {
      v31 = *(_DWORD *)v21;
      *(_DWORD *)v21 = *v19;
      v32 = v19[1];
      *v19 = v31;
      v33 = *((_DWORD *)v21 + 1);
      *((_DWORD *)v21 + 1) = v32;
      v19[1] = v33;
    }
    return;
  }
  if ( a4 <= a5 )
    goto LABEL_10;
LABEL_5:
  v45 = (int)v9;
  v48 = v11;
  v12 = v10 / 2;
  v13 = &v9[2 * (v10 / 2)];
  v14 = sub_2C4E5D0(v11, a3, v13, a7, a8);
  v15 = (char *)v48;
  v16 = v45;
  v17 = (char *)v14;
  v18 = ((char *)v14 - (char *)v48) >> 3;
  while ( 1 )
  {
    v49 = v16;
    v43 = (unsigned int *)v17;
    v8 -= v18;
    v46 = sub_2C4CF50((char *)v13, v15, v17);
    sub_2C50850(v49, (_DWORD)v13, (_DWORD)v46, v12, v18, v49, (__int64)a7, (__int64)a8);
    v10 -= v12;
    if ( !v10 )
      break;
    v19 = v43;
    if ( !v8 )
      break;
    if ( v8 + v10 == 2 )
    {
      v21 = v46;
      goto LABEL_12;
    }
    v11 = v43;
    v9 = (unsigned int *)v46;
    if ( v10 > v8 )
      goto LABEL_5;
LABEL_10:
    v47 = v11;
    v50 = v9;
    v18 = v8 / 2;
    v44 = &v11[2 * (v8 / 2)];
    v20 = sub_2C4E390(v9, (__int64)v11, v44, a7, a8);
    v16 = (int)v50;
    v17 = (char *)v44;
    v15 = (char *)v47;
    v13 = v20;
    v12 = ((char *)v20 - (char *)v50) >> 3;
  }
}
