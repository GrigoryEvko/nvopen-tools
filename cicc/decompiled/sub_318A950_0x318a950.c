// Function: sub_318A950
// Address: 0x318a950
//
__int64 __fastcall sub_318A950(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rdi
  _QWORD *v12; // r10
  int v13; // r15d
  unsigned int v14; // edx
  _QWORD *v15; // rcx
  __int64 v16; // r11
  int v17; // edi
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // ecx
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // edx
  __int64 v25; // r9
  int v26; // r14d
  _QWORD *v27; // r11
  int v28; // ecx
  int v29; // esi
  __int64 v30; // r8
  int v31; // r14d
  unsigned int v32; // edx
  __int64 v33; // r9
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    *(_QWORD *)a2 = v9 + 1;
    goto LABEL_18;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 0;
  v13 = 1;
  v14 = (v8 - 1) & (((0xBF58476D1CE4E5B9LL * *a3) >> 31) ^ (484763065 * *(_DWORD *)a3));
  v15 = (_QWORD *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
  {
LABEL_3:
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = v9;
    *(_QWORD *)(result + 16) = v15;
    *(_QWORD *)(result + 24) = 16 * v8 + v11;
    *(_BYTE *)(result + 32) = 0;
    return result;
  }
  while ( v16 != -1 )
  {
    if ( !v12 && v16 == -2 )
      v12 = v15;
    v14 = (v8 - 1) & (v13 + v14);
    v15 = (_QWORD *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
    ++v13;
  }
  v17 = *(_DWORD *)(a2 + 16);
  if ( !v12 )
    v12 = v15;
  v18 = v17 + 1;
  *(_QWORD *)a2 = v9 + 1;
  if ( 4 * (v17 + 1) >= (unsigned int)(3 * v8) )
  {
LABEL_18:
    v34 = result;
    sub_318A760(a2, 2 * v8);
    v21 = *(_DWORD *)(a2 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a2 + 8);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v34;
      v24 = v22 & (((0xBF58476D1CE4E5B9LL * *a3) >> 31) ^ (484763065 * *(_DWORD *)a3));
      v12 = (_QWORD *)(v23 + 16LL * v24);
      v25 = *v12;
      if ( *v12 == *a3 )
        goto LABEL_14;
      v26 = 1;
      v27 = 0;
      while ( v25 != -1 )
      {
        if ( !v27 && v25 == -2 )
          v27 = v12;
        v24 = v22 & (v26 + v24);
        v12 = (_QWORD *)(v23 + 16LL * v24);
        v25 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v26;
      }
LABEL_22:
      if ( v27 )
        v12 = v27;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v18 <= (unsigned int)v8 >> 3 )
  {
    v35 = result;
    sub_318A760(a2, v8);
    v28 = *(_DWORD *)(a2 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a2 + 8);
      v27 = 0;
      v31 = 1;
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v35;
      v32 = v29 & (((0xBF58476D1CE4E5B9LL * *a3) >> 31) ^ (484763065 * *(_DWORD *)a3));
      v12 = (_QWORD *)(v30 + 16LL * v32);
      v33 = *v12;
      if ( *a3 == *v12 )
        goto LABEL_14;
      while ( v33 != -1 )
      {
        if ( v33 == -2 && !v27 )
          v27 = v12;
        v32 = v29 & (v31 + v32);
        v12 = (_QWORD *)(v30 + 16LL * v32);
        v33 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v31;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v12 != -1 )
    --*(_DWORD *)(a2 + 20);
  v19 = *a3;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v12;
  *v12 = v19;
  LODWORD(v19) = *a4;
  *(_BYTE *)(result + 32) = 1;
  *((_DWORD *)v12 + 2) = v19;
  v20 = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(result + 8) = *(_QWORD *)a2;
  *(_QWORD *)(result + 24) = v20;
  return result;
}
