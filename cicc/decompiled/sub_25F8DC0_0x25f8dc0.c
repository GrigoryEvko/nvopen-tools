// Function: sub_25F8DC0
// Address: 0x25f8dc0
//
void __fastcall sub_25F8DC0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 *v10; // r14
  __int64 *v11; // r13
  __int64 v12; // rbx
  __int64 *v13; // rax
  __int64 v14; // r9
  char *v15; // r10
  char *v16; // r11
  __int64 v17; // r15
  char *v18; // r13
  char *v19; // rax
  __int64 v20; // rcx
  int v21; // edi
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  bool v25; // of
  signed __int64 v26; // rax
  signed __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rax
  int v30; // r8d
  __int64 v31; // rax
  bool v32; // cc
  __int64 v33; // [rsp-58h] [rbp-58h]
  __int64 v34; // [rsp-50h] [rbp-50h]
  char *v35; // [rsp-50h] [rbp-50h]
  __int64 *v36; // [rsp-50h] [rbp-50h]
  __int64 *v37; // [rsp-48h] [rbp-48h]
  char *v38; // [rsp-48h] [rbp-48h]
  __int64 v39; // [rsp-48h] [rbp-48h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  __int64 v41; // [rsp-30h] [rbp-30h]
  __int64 v42; // [rsp-20h] [rbp-20h]
  __int64 v43; // [rsp-18h] [rbp-18h]
  __int64 v44; // [rsp-10h] [rbp-10h]

  while ( 1 )
  {
    if ( !a4 )
      return;
    v44 = v8;
    v43 = v7;
    v42 = v6;
    v9 = a5;
    v41 = v5;
    if ( !a5 )
      return;
    v10 = a1;
    v11 = a2;
    v12 = a4;
    if ( a4 + a5 == 2 )
      break;
    if ( a4 > a5 )
    {
      v39 = a3;
      v17 = a4 / 2;
      v36 = &a1[a4 / 2];
      v19 = (char *)sub_25F7EB0(a2, a3, v36);
      v14 = v39;
      v16 = (char *)v36;
      v15 = v19;
      v40 = (v19 - (char *)a2) >> 3;
    }
    else
    {
      v34 = a3;
      v40 = a5 / 2;
      v37 = &a2[a5 / 2];
      v13 = sub_25F7DB0(a1, (__int64)a2, v37);
      v14 = v34;
      v15 = (char *)v37;
      v16 = (char *)v13;
      v17 = v13 - a1;
    }
    v33 = v14;
    v35 = v15;
    v38 = v16;
    v18 = sub_25F8C00(v16, (char *)a2, v15);
    sub_25F8DC0(a1, v38, v18, v17, v40);
    a4 = v12 - v17;
    a1 = (__int64 *)v18;
    v5 = v41;
    a5 = v9 - v40;
    a3 = v33;
    a2 = (__int64 *)v35;
    v6 = v42;
    v7 = v43;
    v8 = v44;
  }
  v20 = *a1;
  v21 = 1;
  v22 = *a2;
  v23 = *(_QWORD *)(v20 + 296);
  v24 = *(_QWORD *)(v20 + 280);
  if ( *(_DWORD *)(v20 + 304) != 1 )
    v21 = *(_DWORD *)(v20 + 288);
  v25 = __OFSUB__(v24, v23);
  v26 = v24 - v23;
  if ( v25 )
  {
    v32 = v23 <= 0;
    v27 = 0x7FFFFFFFFFFFFFFFLL;
    if ( !v32 )
      v27 = 0x8000000000000000LL;
  }
  else
  {
    v27 = v26;
  }
  v28 = *(_QWORD *)(v22 + 296);
  v29 = *(_QWORD *)(v22 + 280);
  v30 = 1;
  if ( *(_DWORD *)(v22 + 304) != 1 )
    v30 = *(_DWORD *)(v22 + 288);
  v25 = __OFSUB__(v29, v28);
  v31 = v29 - v28;
  if ( !v25 )
    goto LABEL_15;
  if ( v28 <= 0 )
  {
    v31 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_15:
    if ( v30 != v21 )
      goto LABEL_16;
    if ( v31 > v27 )
    {
LABEL_17:
      *v10 = v22;
      *v11 = v20;
      return;
    }
    return;
  }
  if ( v30 != v21 )
  {
LABEL_16:
    if ( v21 >= v30 )
      return;
    goto LABEL_17;
  }
}
