// Function: sub_12D5220
// Address: 0x12d5220
//
bool __fastcall sub_12D5220(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // eax
  __int64 v23; // rax
  bool result; // al
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // rax
  unsigned int v31; // r14d
  unsigned int v32; // r13d
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  __int64 *v37; // rax
  __int64 v38; // rax
  unsigned int v39; // edx
  unsigned int v40; // ecx
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rdx
  int v44; // eax
  __int64 v45; // rax
  unsigned int v46; // r13d
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rax
  unsigned int v52; // eax
  __int64 v53; // [rsp+0h] [rbp-40h]
  __int64 v54; // [rsp+8h] [rbp-38h]

  v4 = *a1;
  v5 = sub_1649960(a2);
  v53 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
  v7 = sub_16D1B30(v4, v5, v6);
  if ( v7 == -1 )
    v54 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
  else
    v54 = *(_QWORD *)v4 + 8LL * v7;
  v8 = *a1;
  v9 = sub_1649960(a3);
  v10 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
  v12 = sub_16D1B30(v8, v9, v11);
  if ( v12 == -1 )
  {
    if ( v10 != *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8) )
      goto LABEL_5;
  }
  else if ( v10 != *(_QWORD *)v8 + 8LL * v12 )
  {
LABEL_5:
    if ( v54 == v53 )
      v13 = a1[1];
    else
      v13 = *a1;
    v14 = sub_1649960(a2);
    v16 = sub_16D1B30(v13, v14, v15);
    if ( v16 == -1 || (v17 = *(_QWORD *)v13 + 8LL * v16, v17 == *(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8)) )
      v18 = 0;
    else
      v18 = *(_DWORD *)(*(_QWORD *)v17 + 8LL);
    v19 = *a1;
    v20 = sub_1649960(a3);
    v22 = sub_16D1B30(v19, v20, v21);
    result = v22 != -1
          && (v23 = *(_QWORD *)v19 + 8LL * v22, v23 != *(_QWORD *)v19 + 8LL * *(unsigned int *)(v19 + 8))
          && *(_DWORD *)(*(_QWORD *)v23 + 8LL) > v18;
    return result;
  }
  if ( v54 == v53 )
  {
    v25 = a1[1];
    v26 = sub_1649960(a2);
    v28 = sub_16D1B30(v25, v26, v27);
    if ( v28 == -1
      || (v29 = (__int64 *)(*(_QWORD *)v25 + 8LL * v28),
          v29 == (__int64 *)(*(_QWORD *)v25 + 8LL * *(unsigned int *)(v25 + 8))) )
    {
      v32 = 0;
      v31 = 0;
    }
    else
    {
      v30 = *v29;
      v31 = *(_DWORD *)(v30 + 8);
      v32 = *(_DWORD *)(v30 + 12);
    }
    v33 = a1[1];
    v34 = sub_1649960(a3);
    v36 = sub_16D1B30(v33, v34, v35);
    if ( v36 == -1
      || (v37 = (__int64 *)(*(_QWORD *)v33 + 8LL * v36),
          v37 == (__int64 *)(*(_QWORD *)v33 + 8LL * *(unsigned int *)(v33 + 8))) )
    {
      v40 = 0;
      v39 = 0;
    }
    else
    {
      v38 = *v37;
      v39 = *(_DWORD *)(v38 + 8);
      v40 = *(_DWORD *)(v38 + 12);
    }
    result = v39 > v31;
    if ( v39 == v31 )
      return v40 > v32;
  }
  else
  {
    v41 = *a1;
    v42 = sub_1649960(a2);
    v44 = sub_16D1B30(v41, v42, v43);
    if ( v44 == -1 || (v45 = *(_QWORD *)v41 + 8LL * v44, v45 == *(_QWORD *)v41 + 8LL * *(unsigned int *)(v41 + 8)) )
      v46 = 0;
    else
      v46 = *(_DWORD *)(*(_QWORD *)v45 + 8LL);
    v47 = a1[1];
    v48 = sub_1649960(a3);
    v50 = sub_16D1B30(v47, v48, v49);
    if ( v50 == -1 || (v51 = *(_QWORD *)v47 + 8LL * v50, v51 == *(_QWORD *)v47 + 8LL * *(unsigned int *)(v47 + 8)) )
      v52 = 0;
    else
      v52 = *(_DWORD *)(*(_QWORD *)v51 + 8LL);
    return v46 <= v52;
  }
  return result;
}
