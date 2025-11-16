// Function: sub_2261010
// Address: 0x2261010
//
bool __fastcall sub_2261010(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  const char *v5; // rax
  size_t v6; // rdx
  size_t v7; // r14
  int v8; // eax
  int v9; // eax
  __int64 v10; // r14
  const char *v11; // r13
  size_t v12; // rdx
  int v13; // eax
  int v14; // eax
  __int64 v15; // r14
  const char *v16; // r15
  size_t v17; // rdx
  size_t v18; // r13
  int v19; // eax
  int v20; // eax
  __int64 v21; // rax
  unsigned int v22; // r13d
  __int64 v23; // rbx
  const char *v24; // r14
  size_t v25; // rdx
  size_t v26; // r12
  int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  bool result; // al
  __int64 v31; // r14
  const char *v32; // r15
  size_t v33; // rdx
  size_t v34; // r13
  int v35; // eax
  int v36; // eax
  __int64 *v37; // rax
  __int64 v38; // rax
  unsigned int v39; // r14d
  unsigned int v40; // r13d
  __int64 v41; // rbx
  const char *v42; // r15
  size_t v43; // rdx
  size_t v44; // r12
  int v45; // eax
  int v46; // eax
  __int64 *v47; // rax
  __int64 v48; // rax
  unsigned int v49; // edx
  unsigned int v50; // ecx
  __int64 v51; // r14
  const char *v52; // r15
  size_t v53; // rdx
  size_t v54; // r13
  int v55; // eax
  int v56; // eax
  __int64 v57; // rax
  unsigned int v58; // r13d
  __int64 v59; // rbx
  const char *v60; // r14
  size_t v61; // rdx
  size_t v62; // r12
  int v63; // eax
  int v64; // eax
  __int64 v65; // rax
  unsigned int v66; // eax
  __int64 v67; // [rsp+0h] [rbp-50h]
  __int64 v68; // [rsp+8h] [rbp-48h]
  size_t v69; // [rsp+10h] [rbp-40h]
  const char *v70; // [rsp+18h] [rbp-38h]
  __int64 v71; // [rsp+18h] [rbp-38h]

  v4 = *a1;
  v5 = sub_BD5D20(a2);
  v7 = v6;
  v70 = v5;
  v67 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
  v8 = sub_C92610();
  v9 = sub_C92860((__int64 *)v4, v70, v7, v8);
  if ( v9 == -1 )
    v68 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
  else
    v68 = *(_QWORD *)v4 + 8LL * v9;
  v10 = *a1;
  v11 = sub_BD5D20(a3);
  v69 = v12;
  v71 = *(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8);
  v13 = sub_C92610();
  v14 = sub_C92860((__int64 *)v10, v11, v69, v13);
  if ( v14 == -1 )
  {
    if ( v71 != *(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8) )
      goto LABEL_5;
  }
  else if ( v71 != *(_QWORD *)v10 + 8LL * v14 )
  {
LABEL_5:
    if ( v68 == v67 )
      v15 = a1[1];
    else
      v15 = *a1;
    v16 = sub_BD5D20(a2);
    v18 = v17;
    v19 = sub_C92610();
    v20 = sub_C92860((__int64 *)v15, v16, v18, v19);
    if ( v20 == -1 || (v21 = *(_QWORD *)v15 + 8LL * v20, v21 == *(_QWORD *)v15 + 8LL * *(unsigned int *)(v15 + 8)) )
      v22 = 0;
    else
      v22 = *(_DWORD *)(*(_QWORD *)v21 + 8LL);
    v23 = *a1;
    v24 = sub_BD5D20(a3);
    v26 = v25;
    v27 = sub_C92610();
    v28 = sub_C92860((__int64 *)v23, v24, v26, v27);
    result = v28 != -1
          && (v29 = *(_QWORD *)v23 + 8LL * v28, v29 != *(_QWORD *)v23 + 8LL * *(unsigned int *)(v23 + 8))
          && *(_DWORD *)(*(_QWORD *)v29 + 8LL) > v22;
    return result;
  }
  if ( v68 == v67 )
  {
    v31 = a1[1];
    v32 = sub_BD5D20(a2);
    v34 = v33;
    v35 = sub_C92610();
    v36 = sub_C92860((__int64 *)v31, v32, v34, v35);
    if ( v36 == -1
      || (v37 = (__int64 *)(*(_QWORD *)v31 + 8LL * v36),
          v37 == (__int64 *)(*(_QWORD *)v31 + 8LL * *(unsigned int *)(v31 + 8))) )
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
    v41 = a1[1];
    v42 = sub_BD5D20(a3);
    v44 = v43;
    v45 = sub_C92610();
    v46 = sub_C92860((__int64 *)v41, v42, v44, v45);
    if ( v46 == -1
      || (v47 = (__int64 *)(*(_QWORD *)v41 + 8LL * v46),
          v47 == (__int64 *)(*(_QWORD *)v41 + 8LL * *(unsigned int *)(v41 + 8))) )
    {
      v50 = 0;
      v49 = 0;
    }
    else
    {
      v48 = *v47;
      v49 = *(_DWORD *)(v48 + 8);
      v50 = *(_DWORD *)(v48 + 12);
    }
    result = v49 > v39;
    if ( v49 == v39 )
      return v50 > v40;
  }
  else
  {
    v51 = *a1;
    v52 = sub_BD5D20(a2);
    v54 = v53;
    v55 = sub_C92610();
    v56 = sub_C92860((__int64 *)v51, v52, v54, v55);
    if ( v56 == -1 || (v57 = *(_QWORD *)v51 + 8LL * v56, v57 == *(_QWORD *)v51 + 8LL * *(unsigned int *)(v51 + 8)) )
      v58 = 0;
    else
      v58 = *(_DWORD *)(*(_QWORD *)v57 + 8LL);
    v59 = a1[1];
    v60 = sub_BD5D20(a3);
    v62 = v61;
    v63 = sub_C92610();
    v64 = sub_C92860((__int64 *)v59, v60, v62, v63);
    if ( v64 == -1 || (v65 = *(_QWORD *)v59 + 8LL * v64, v65 == *(_QWORD *)v59 + 8LL * *(unsigned int *)(v59 + 8)) )
      v66 = 0;
    else
      v66 = *(_DWORD *)(*(_QWORD *)v65 + 8LL);
    return v58 <= v66;
  }
  return result;
}
