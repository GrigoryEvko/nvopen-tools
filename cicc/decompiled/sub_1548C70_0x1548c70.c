// Function: sub_1548C70
// Address: 0x1548c70
//
char __fastcall sub_1548C70(__int64 *a1, __int64 a2, __int64 a3)
{
  char result; // al
  __int64 v7; // r15
  int v8; // r14d
  int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  unsigned int v17; // edx
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdi
  unsigned int v22; // ecx
  char *v23; // rsi
  unsigned int v24; // ebx
  int v25; // eax
  char v26; // si
  unsigned int v27; // ebx
  int v28; // eax
  __int64 v29; // rax
  int v30; // edi
  int v31; // r8d
  unsigned int v32; // [rsp-3Ch] [rbp-3Ch]

  result = 0;
  if ( a2 == a3 )
    return result;
  v7 = *a1;
  v8 = *(_DWORD *)(*a1 + 24);
  if ( !v8 )
  {
    if ( *(_BYTE *)a1[1] )
    {
LABEL_21:
      v27 = sub_1648720(a2);
      return v27 < (unsigned int)sub_1648720(a3);
    }
LABEL_11:
    v24 = sub_1648720(a2);
    return v24 > (unsigned int)sub_1648720(a3);
  }
  v9 = v8 - 1;
  v10 = sub_1648700(a2);
  v11 = *(_QWORD *)(v7 + 8);
  v12 = v10;
  v13 = v9 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( v12 == *v14 )
  {
LABEL_4:
    v32 = *((_DWORD *)v14 + 2);
    v16 = sub_1648700(a3);
    v17 = v32;
    v18 = v16;
  }
  else
  {
    v28 = 1;
    while ( v15 != -8 )
    {
      v30 = v28 + 1;
      v13 = v9 & (v28 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( v12 == *v14 )
        goto LABEL_4;
      v28 = v30;
    }
    v29 = sub_1648700(a3);
    v17 = 0;
    v18 = v29;
  }
  v19 = v9 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v20 = (__int64 *)(v11 + 16LL * v19);
  v21 = *v20;
  if ( *v20 == v18 )
  {
LABEL_6:
    v22 = *((_DWORD *)v20 + 2);
    v23 = (char *)a1[1];
    result = *v23;
    if ( v22 > v17 )
    {
      if ( result )
        return *(_DWORD *)a1[2] >= v22;
      return result;
    }
  }
  else
  {
    v25 = 1;
    while ( v21 != -8 )
    {
      v31 = v25 + 1;
      v19 = v9 & (v25 + v19);
      v20 = (__int64 *)(v11 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == v18 )
        goto LABEL_6;
      v25 = v31;
    }
    v23 = (char *)a1[1];
    v22 = 0;
  }
  v26 = *v23;
  if ( v22 >= v17 )
  {
    if ( v26 && *(_DWORD *)a1[2] >= v17 )
      goto LABEL_21;
    goto LABEL_11;
  }
  result = 1;
  if ( v26 )
    return *(_DWORD *)a1[2] < v17;
  return result;
}
