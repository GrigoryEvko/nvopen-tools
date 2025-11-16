// Function: sub_30D8A00
// Address: 0x30d8a00
//
bool __fastcall sub_30D8A00(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4)
{
  __int64 v4; // rbx
  __int64 *v7; // r15
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r12
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  __int64 *v15; // rax
  unsigned int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // edi
  _QWORD *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  _QWORD *v22; // rcx
  unsigned int v23; // r14d
  int v24; // eax
  int v25; // edx
  int v26; // eax
  int v27; // r9d
  __int64 v28; // r10
  unsigned int v29; // eax
  __int64 v30; // r8
  int v31; // edi
  _QWORD *v32; // rsi
  int v33; // eax
  int v34; // r8d
  __int64 v35; // rdi
  _QWORD *v36; // r9
  unsigned int v37; // r10d
  int v38; // eax
  __int64 v39; // rsi
  int v40; // [rsp+Ch] [rbp-44h]
  _QWORD *v41; // [rsp+10h] [rbp-40h]
  _QWORD *v42; // [rsp+10h] [rbp-40h]

  v4 = a1;
  if ( a2 == a1 )
    return a2 == v4;
  v7 = a3;
  while ( 1 )
  {
    v8 = *v7;
    v9 = *(_QWORD *)(v4 + 24);
    v10 = *a4;
    v11 = *(_QWORD *)(v9 + 40);
    if ( !*(_BYTE *)(*v7 + 292) )
      break;
    v12 = *(_QWORD **)(v8 + 272);
    v13 = &v12[*(unsigned int *)(v8 + 284)];
    if ( v12 == v13 )
      goto LABEL_14;
    while ( v11 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_14;
    }
    do
LABEL_8:
      v4 = *(_QWORD *)(v4 + 8);
    while ( v4 && (unsigned __int8)(**(_BYTE **)(v4 + 24) - 30) > 0xAu );
    if ( a2 == v4 )
      return a2 == v4;
  }
  v41 = a4;
  v15 = sub_C8CA60(v8 + 264, *(_QWORD *)(v9 + 40));
  a4 = v41;
  if ( v15 )
    goto LABEL_8;
  v8 = *v7;
LABEL_14:
  v16 = *(_DWORD *)(v8 + 448);
  if ( !v16 )
  {
    ++*(_QWORD *)(v8 + 424);
    goto LABEL_29;
  }
  v17 = *(_QWORD *)(v8 + 432);
  v18 = (v16 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v19 = (_QWORD *)(v17 + 16LL * v18);
  v20 = *v19;
  if ( v11 == *v19 )
  {
LABEL_16:
    v21 = v19[1];
    if ( v10 == v21 || !v21 )
      return a2 == v4;
    goto LABEL_8;
  }
  v40 = 1;
  v42 = 0;
  while ( v20 != -4096 )
  {
    if ( !v42 )
    {
      if ( v20 != -8192 )
        v19 = 0;
      v42 = v19;
    }
    v18 = (v16 - 1) & (v40 + v18);
    v19 = (_QWORD *)(v17 + 16LL * v18);
    v20 = *v19;
    if ( v11 == *v19 )
      goto LABEL_16;
    ++v40;
  }
  v22 = v42;
  v23 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
  if ( !v42 )
    v22 = v19;
  v24 = *(_DWORD *)(v8 + 440);
  ++*(_QWORD *)(v8 + 424);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) < 3 * v16 )
  {
    if ( v16 - *(_DWORD *)(v8 + 444) - v25 > v16 >> 3 )
      goto LABEL_25;
    sub_22E02D0(v8 + 424, v16);
    v33 = *(_DWORD *)(v8 + 448);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(v8 + 432);
      v36 = 0;
      v37 = v34 & v23;
      v25 = *(_DWORD *)(v8 + 440) + 1;
      v38 = 1;
      v22 = (_QWORD *)(v35 + 16LL * (v34 & v23));
      v39 = *v22;
      if ( v11 != *v22 )
      {
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v36 )
            v36 = v22;
          v37 = v34 & (v38 + v37);
          v22 = (_QWORD *)(v35 + 16LL * v37);
          v39 = *v22;
          if ( v11 == *v22 )
            goto LABEL_25;
          ++v38;
        }
        if ( v36 )
          v22 = v36;
      }
      goto LABEL_25;
    }
LABEL_59:
    ++*(_DWORD *)(v8 + 440);
    BUG();
  }
LABEL_29:
  sub_22E02D0(v8 + 424, 2 * v16);
  v26 = *(_DWORD *)(v8 + 448);
  if ( !v26 )
    goto LABEL_59;
  v27 = v26 - 1;
  v28 = *(_QWORD *)(v8 + 432);
  v29 = (v26 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v25 = *(_DWORD *)(v8 + 440) + 1;
  v22 = (_QWORD *)(v28 + 16LL * v29);
  v30 = *v22;
  if ( v11 != *v22 )
  {
    v31 = 1;
    v32 = 0;
    while ( v30 != -4096 )
    {
      if ( !v32 && v30 == -8192 )
        v32 = v22;
      v29 = v27 & (v31 + v29);
      v22 = (_QWORD *)(v28 + 16LL * v29);
      v30 = *v22;
      if ( v11 == *v22 )
        goto LABEL_25;
      ++v31;
    }
    if ( v32 )
      v22 = v32;
  }
LABEL_25:
  *(_DWORD *)(v8 + 440) = v25;
  if ( *v22 != -4096 )
    --*(_DWORD *)(v8 + 444);
  *v22 = v11;
  v22[1] = 0;
  return a2 == v4;
}
