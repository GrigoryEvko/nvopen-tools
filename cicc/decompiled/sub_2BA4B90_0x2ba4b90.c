// Function: sub_2BA4B90
// Address: 0x2ba4b90
//
void __fastcall sub_2BA4B90(__int64 a1, _QWORD *a2, __int64 a3, char *a4, unsigned int a5)
{
  __int64 v7; // rbx
  int v8; // eax
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // rax
  int v16; // ecx
  int v17; // edx
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rdi
  int v21; // ecx
  unsigned int v22; // eax
  __int64 *v23; // rdx
  __int64 v24; // rsi
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rsi
  _QWORD *v28; // rax
  int v29; // r9d
  int v30; // eax
  int v31; // r8d
  int v32; // edx
  int v33; // r8d
  __int64 v34; // [rsp-48h] [rbp-48h] BYREF
  __int64 v35; // [rsp-40h] [rbp-40h] BYREF

  if ( *a4 == 84 )
    return;
  v7 = (__int64)a4;
  if ( (unsigned __int8)sub_2B15E10(a4, (__int64)a2, a3, (__int64)a4, a5)
    || a3
    && (&a2[a3] == sub_2B0BF30(a2, (__int64)&a2[a3], (unsigned __int8 (__fastcall *)(_QWORD))sub_2B099C0)
     || sub_2B0D880(a2, a3, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B16010)) )
  {
    return;
  }
  if ( (unsigned __int8)sub_2B14730(v7) )
    v7 = *sub_2B0BF30(a2, (__int64)&a2[a3], (unsigned __int8 (__fastcall *)(_QWORD))sub_2B14730);
  if ( *(_BYTE *)v7 <= 0x1Cu
    || *(_QWORD *)a1 != *(_QWORD *)(v7 + 40)
    || (v8 = *(_DWORD *)(a1 + 104), v9 = *(_QWORD *)(a1 + 88), !v8) )
  {
LABEL_41:
    BUG();
  }
  v10 = v8 - 1;
  v11 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v7 != *v12 )
  {
    v30 = 1;
    while ( v13 != -4096 )
    {
      v31 = v30 + 1;
      v11 = v10 & (v30 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v7 == *v12 )
        goto LABEL_11;
      v30 = v31;
    }
    goto LABEL_41;
  }
LABEL_11:
  v14 = v12[1];
  if ( !v14 || *(_DWORD *)(v14 + 136) != *(_DWORD *)(a1 + 204) )
    goto LABEL_41;
  v34 = v12[1];
  v15 = v14;
  v16 = 0;
  do
  {
    v17 = *(_DWORD *)(v15 + 148);
    if ( v17 == -1 )
      goto LABEL_16;
    v15 = *(_QWORD *)(v15 + 24);
    v16 += v17;
  }
  while ( v15 );
  if ( v16 || *(_BYTE *)(v14 + 152) || (v19 = *(_DWORD *)(a1 + 136), v20 = *(_QWORD *)(a1 + 120), !v19) )
  {
LABEL_16:
    v35 = v14;
    goto LABEL_17;
  }
  v21 = v19 - 1;
  v22 = (v19 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v23 = (__int64 *)(v20 + 8LL * v22);
  v24 = *v23;
  if ( v14 != *v23 )
  {
    v32 = 1;
    while ( v24 != -4096 )
    {
      v33 = v32 + 1;
      v22 = v21 & (v32 + v22);
      v23 = (__int64 *)(v20 + 8LL * v22);
      v24 = *v23;
      if ( v14 == *v23 )
        goto LABEL_29;
      v32 = v33;
    }
    goto LABEL_16;
  }
LABEL_29:
  *v23 = -8192;
  v25 = *(_QWORD **)(a1 + 144);
  --*(_DWORD *)(a1 + 128);
  v26 = *(unsigned int *)(a1 + 152);
  ++*(_DWORD *)(a1 + 132);
  v27 = (__int64)&v25[v26];
  v28 = sub_2B0B870(v25, v27, &v34);
  if ( v28 + 1 == (_QWORD *)v27 )
  {
    *(_DWORD *)(a1 + 152) = v29 - 1;
    goto LABEL_16;
  }
  memmove(v28, v28 + 1, v27 - (_QWORD)(v28 + 1));
  v14 = v34;
  --*(_DWORD *)(a1 + 152);
  v35 = v14;
  if ( !v14 )
    return;
  do
  {
LABEL_17:
    v18 = v14;
    *(_QWORD *)(v14 + 16) = v14;
    v14 = *(_QWORD *)(v14 + 24);
    *(_QWORD *)(v18 + 24) = 0;
    *(_QWORD *)(v18 + 8) = 0;
    if ( !*(_DWORD *)(v18 + 148) )
      sub_2BA3420(a1 + 112, &v35);
    v35 = v14;
  }
  while ( v14 );
}
