// Function: sub_2F2D4D0
// Address: 0x2f2d4d0
//
__int64 __fastcall sub_2F2D4D0(__int64 a1, __int64 a2, int *a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  int v10; // edi
  __int64 v11; // rcx
  int v12; // r15d
  _DWORD *v13; // r10
  unsigned int v14; // r8d
  _DWORD *v15; // rdx
  int v16; // r11d
  int v17; // edi
  int v18; // ecx
  int v19; // edx
  __int64 v20; // rdx
  __int64 v21; // rdx
  int v22; // edx
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // edx
  int v26; // r8d
  int v27; // r14d
  _DWORD *v28; // r11
  int v29; // edx
  int v30; // edx
  __int64 v31; // r9
  int v32; // r14d
  unsigned int v33; // r8d
  int v34; // edi
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
  v12 = 1;
  v13 = 0;
  v14 = (v8 - 1) & (37 * *a3);
  v15 = (_DWORD *)(v11 + 16LL * v14);
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
    if ( !v13 && v16 == -2 )
      v13 = v15;
    v14 = (v8 - 1) & (v12 + v14);
    v15 = (_DWORD *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
    ++v12;
  }
  v17 = *(_DWORD *)(a2 + 16);
  if ( !v13 )
    v13 = v15;
  v18 = v17 + 1;
  *(_QWORD *)a2 = v9 + 1;
  if ( 4 * (v17 + 1) >= (unsigned int)(3 * v8) )
  {
LABEL_18:
    v35 = result;
    sub_2F2D2F0(a2, 2 * v8);
    v22 = *(_DWORD *)(a2 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a2 + 8);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v35;
      v25 = (v22 - 1) & (37 * *a3);
      v13 = (_DWORD *)(v24 + 16LL * (v23 & (unsigned int)(37 * *a3)));
      v26 = *v13;
      if ( *v13 == *a3 )
        goto LABEL_14;
      v27 = 1;
      v28 = 0;
      while ( v26 != -1 )
      {
        if ( !v28 && v26 == -2 )
          v28 = v13;
        v25 = v23 & (v27 + v25);
        v13 = (_DWORD *)(v24 + 16LL * v25);
        v26 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_14;
        ++v27;
      }
LABEL_22:
      if ( v28 )
        v13 = v28;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v18 <= (unsigned int)v8 >> 3 )
  {
    v36 = result;
    sub_2F2D2F0(a2, v8);
    v29 = *(_DWORD *)(a2 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a2 + 8);
      v28 = 0;
      v32 = 1;
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v36;
      v33 = v30 & (37 * *a3);
      v13 = (_DWORD *)(v31 + 16LL * v33);
      v34 = *v13;
      if ( *a3 == *v13 )
        goto LABEL_14;
      while ( v34 != -1 )
      {
        if ( v34 == -2 && !v28 )
          v28 = v13;
        v33 = v30 & (v32 + v33);
        v13 = (_DWORD *)(v31 + 16LL * v33);
        v34 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_14;
        ++v32;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v13 != -1 )
    --*(_DWORD *)(a2 + 20);
  v19 = *a3;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v13;
  *v13 = v19;
  v20 = *a4;
  *(_BYTE *)(result + 32) = 1;
  *((_QWORD *)v13 + 1) = v20;
  v21 = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(result + 8) = *(_QWORD *)a2;
  *(_QWORD *)(result + 24) = v21;
  return result;
}
