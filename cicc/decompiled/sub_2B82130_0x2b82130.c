// Function: sub_2B82130
// Address: 0x2b82130
//
__int64 __fastcall sub_2B82130(__int64 a1, int *a2, _DWORD *a3, int *a4)
{
  int v8; // r13d
  unsigned int v9; // esi
  __int64 v10; // r9
  int *v11; // rdx
  int v12; // r11d
  unsigned int v13; // edi
  int *v14; // rax
  int v15; // ecx
  int v17; // eax
  int v18; // ecx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // rcx
  int *v23; // rdx
  __int64 v24; // rax
  int v25; // eax
  int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // eax
  int v29; // edi
  int v30; // r10d
  __int64 v31; // rax
  int v32; // eax
  int v33; // eax
  __int64 v34; // rdi
  int v35; // r10d
  unsigned int v36; // r8d
  int v37; // esi
  int v38; // [rsp+Ch] [rbp-44h]
  __int64 v39; // [rsp+14h] [rbp-3Ch]
  int v40; // [rsp+1Ch] [rbp-34h]

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 0;
  v12 = 1;
  v13 = (v9 - 1) & (37 * v8);
  v14 = (int *)(v10 + 8LL * v13);
  v15 = *v14;
  if ( v8 == *v14 )
    return *(_QWORD *)(a1 + 32) + 12LL * (unsigned int)v14[1];
  while ( v15 != -1 )
  {
    if ( v15 == -2 && !v11 )
      v11 = v14;
    v13 = (v9 - 1) & (v12 + v13);
    v14 = (int *)(v10 + 8LL * v13);
    v15 = *v14;
    if ( v8 == *v14 )
      return *(_QWORD *)(a1 + 32) + 12LL * (unsigned int)v14[1];
    ++v12;
  }
  if ( !v11 )
    v11 = v14;
  v17 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v9 )
  {
LABEL_22:
    sub_A09770(a1, 2 * v9);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = (v25 - 1) & (37 * v8);
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v11 = (int *)(v27 + 8LL * v28);
      v29 = *v11;
      if ( v8 == *v11 )
        goto LABEL_14;
      v30 = 1;
      v10 = 0;
      while ( v29 != -1 )
      {
        if ( !v10 && v29 == -2 )
          v10 = (__int64)v11;
        v28 = v26 & (v30 + v28);
        v11 = (int *)(v27 + 8LL * v28);
        v29 = *v11;
        if ( v8 == *v11 )
          goto LABEL_14;
        ++v30;
      }
LABEL_26:
      if ( v10 )
        v11 = (int *)v10;
      goto LABEL_14;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v18 <= v9 >> 3 )
  {
    v38 = 37 * v8;
    sub_A09770(a1, v9);
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v10 = 0;
      v35 = 1;
      v36 = v33 & v38;
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v11 = (int *)(v34 + 8LL * (v33 & (unsigned int)v38));
      v37 = *v11;
      if ( v8 == *v11 )
        goto LABEL_14;
      while ( v37 != -1 )
      {
        if ( !v10 && v37 == -2 )
          v10 = (__int64)v11;
        v36 = v33 & (v35 + v36);
        v11 = (int *)(v34 + 8LL * v36);
        v37 = *v11;
        if ( v8 == *v11 )
          goto LABEL_14;
        ++v35;
      }
      goto LABEL_26;
    }
    goto LABEL_45;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v11 != -1 )
    --*(_DWORD *)(a1 + 20);
  v11[1] = 0;
  *v11 = v8;
  v11[1] = *(_DWORD *)(a1 + 40);
  v19 = *(unsigned int *)(a1 + 40);
  v20 = *(unsigned int *)(a1 + 44);
  v21 = *(_DWORD *)(a1 + 40);
  if ( v19 >= v20 )
  {
    HIDWORD(v39) = *a3;
    LODWORD(v39) = *a2;
    v40 = *a4;
    if ( v20 < v19 + 1 )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v19 + 1, 0xCu, v19 + 1, v10);
      v19 = *(unsigned int *)(a1 + 40);
    }
    v31 = *(_QWORD *)(a1 + 32) + 12 * v19;
    *(_QWORD *)v31 = v39;
    *(_DWORD *)(v31 + 8) = v40;
    v22 = *(_QWORD *)(a1 + 32);
    v24 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v24;
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = (int *)(v22 + 12 * v19);
    if ( v23 )
    {
      *v23 = *a2;
      v23[1] = *a3;
      v23[2] = *a4;
      v21 = *(_DWORD *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
    }
    v24 = (unsigned int)(v21 + 1);
    *(_DWORD *)(a1 + 40) = v24;
  }
  return v22 + 12 * v24 - 12;
}
