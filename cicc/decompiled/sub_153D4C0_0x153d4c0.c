// Function: sub_153D4C0
// Address: 0x153d4c0
//
char __fastcall sub_153D4C0(__int64 *a1, __int64 a2, __int64 a3)
{
  char result; // al
  __int64 v7; // r15
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 *v13; // rax
  __int64 v14; // r8
  unsigned int v15; // r14d
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // r8
  unsigned int v23; // edx
  unsigned int v24; // eax
  unsigned int v25; // ebx
  int v26; // eax
  unsigned int v27; // ecx
  unsigned int v28; // ecx
  int v29; // eax
  __int64 v30; // rax
  __int64 *v31; // rax
  unsigned int v32; // ebx
  int v33; // r9d
  int v34; // r9d
  int v35; // [rsp-44h] [rbp-44h]
  int v36; // [rsp-44h] [rbp-44h]
  __int64 v37; // [rsp-40h] [rbp-40h]
  __int64 v38; // [rsp-40h] [rbp-40h]

  result = 0;
  if ( a2 == a3 )
    return result;
  v7 = *a1;
  v8 = *(_DWORD *)(*a1 + 24);
  if ( !v8 )
  {
LABEL_12:
    if ( !*(_BYTE *)a1[2] )
    {
      v32 = sub_1648720(a2);
      return v32 < (unsigned int)sub_1648720(a3);
    }
LABEL_13:
    v25 = sub_1648720(a2);
    return v25 > (unsigned int)sub_1648720(a3);
  }
  v9 = sub_1648700(a2);
  v10 = *(_QWORD *)(v7 + 8);
  v11 = v9;
  v12 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( *v13 != v11 )
  {
    v29 = 1;
    while ( v14 != -8 )
    {
      v33 = v29 + 1;
      v12 = (v8 - 1) & (v29 + v12);
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v11 == *v13 )
        goto LABEL_4;
      v29 = v33;
    }
    v36 = v8 - 1;
    v15 = 0;
    v38 = *(_QWORD *)(v7 + 8);
    v30 = sub_1648700(a3);
    v17 = v36;
    v18 = v38;
    v19 = v30;
    v20 = v36 & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
    v31 = (__int64 *)(v38 + 16LL * v20);
    v22 = *v31;
    if ( v19 == *v31 )
    {
      v23 = *((_DWORD *)v31 + 2);
      v15 = 0;
      goto LABEL_6;
    }
LABEL_14:
    v26 = 1;
    while ( v22 != -8 )
    {
      v34 = v26 + 1;
      v20 = v17 & (v26 + v20);
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( v19 == *v21 )
        goto LABEL_5;
      v26 = v34;
    }
    v23 = 0;
    goto LABEL_17;
  }
LABEL_4:
  v35 = v8 - 1;
  v15 = *((_DWORD *)v13 + 2);
  v37 = *(_QWORD *)(v7 + 8);
  v16 = sub_1648700(a3);
  v17 = v35;
  v18 = v37;
  v19 = v16;
  v20 = v35 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v21 = (__int64 *)(v37 + 16LL * v20);
  v22 = *v21;
  if ( *v21 != v19 )
    goto LABEL_14;
LABEL_5:
  v23 = *((_DWORD *)v21 + 2);
  v24 = *(_DWORD *)(v7 + 36);
  if ( v24 >= v15 )
  {
    v28 = *(_DWORD *)(v7 + 32);
    if ( v24 >= v23 && v28 < v23 && v28 < v15 )
      return v15 < v23;
  }
LABEL_6:
  if ( v23 > v15 )
  {
    result = 0;
    if ( v23 <= *(_DWORD *)a1[1] )
      return *(_BYTE *)a1[2] ^ 1;
    return result;
  }
LABEL_17:
  v27 = *(_DWORD *)a1[1];
  if ( v23 >= v15 )
  {
    if ( v27 < v15 )
      goto LABEL_13;
    goto LABEL_12;
  }
  result = 1;
  if ( v15 <= v27 )
    return *(_BYTE *)a1[2];
  return result;
}
