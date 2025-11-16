// Function: sub_141AAC0
// Address: 0x141aac0
//
__int64 __fastcall sub_141AAC0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rdi
  _QWORD *v12; // r10
  int v13; // r15d
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // r11
  int v17; // edi
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // edx
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // edx
  __int64 v26; // r9
  int v27; // r14d
  _QWORD *v28; // r11
  int v29; // edx
  int v30; // r8d
  __int64 v31; // rdi
  int v32; // r14d
  unsigned int v33; // edx
  __int64 v34; // r9
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

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
  v14 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
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
  while ( v16 != -8 )
  {
    if ( !v12 && v16 == -16 )
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
    v35 = result;
    sub_141A900(a2, 2 * v8);
    v22 = *(_DWORD *)(a2 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a2 + 8);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v35;
      v25 = (v22 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v12 = (_QWORD *)(v24 + 16LL * v25);
      v26 = *v12;
      if ( *v12 == *a3 )
        goto LABEL_14;
      v27 = 1;
      v28 = 0;
      while ( v26 != -8 )
      {
        if ( !v28 && v26 == -16 )
          v28 = v12;
        v25 = v23 & (v27 + v25);
        v12 = (_QWORD *)(v24 + 16LL * v25);
        v26 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v27;
      }
LABEL_22:
      if ( v28 )
        v12 = v28;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v18 <= (unsigned int)v8 >> 3 )
  {
    v36 = result;
    sub_141A900(a2, v8);
    v29 = *(_DWORD *)(a2 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a2 + 8);
      v28 = 0;
      v32 = 1;
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v36;
      v33 = (v29 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v12 = (_QWORD *)(v31 + 16LL * v33);
      v34 = *v12;
      if ( *a3 == *v12 )
        goto LABEL_14;
      while ( v34 != -8 )
      {
        if ( v34 == -16 && !v28 )
          v28 = v12;
        v33 = v30 & (v32 + v33);
        v12 = (_QWORD *)(v31 + 16LL * v33);
        v34 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v32;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v12 != -8 )
    --*(_DWORD *)(a2 + 20);
  v19 = *a3;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v12;
  *v12 = v19;
  v20 = *a4;
  *(_BYTE *)(result + 32) = 1;
  v12[1] = v20;
  v21 = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(result + 8) = *(_QWORD *)a2;
  *(_QWORD *)(result + 24) = v21;
  return result;
}
