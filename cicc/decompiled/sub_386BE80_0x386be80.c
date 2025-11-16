// Function: sub_386BE80
// Address: 0x386be80
//
__int64 __fastcall sub_386BE80(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rdi
  _QWORD *v12; // r14
  int v13; // r15d
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // r10
  int v17; // edi
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  int v24; // edx
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // edx
  __int64 v28; // r9
  int v29; // r11d
  _QWORD *v30; // r10
  int v31; // edx
  int v32; // r8d
  __int64 v33; // rdi
  int v34; // r11d
  unsigned int v35; // edx
  __int64 v36; // r9
  __int64 v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    *(_QWORD *)a2 = v9 + 1;
    goto LABEL_21;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 0;
  v13 = 1;
  v14 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v15 = (_QWORD *)(v11 + 32LL * v14);
  v16 = *v15;
  if ( *v15 == v10 )
  {
LABEL_3:
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = v9;
    *(_QWORD *)(result + 16) = v15;
    *(_QWORD *)(result + 24) = 32 * v8 + v11;
    *(_BYTE *)(result + 32) = 0;
    return result;
  }
  while ( v16 != -8 )
  {
    if ( !v12 && v16 == -16 )
      v12 = v15;
    v14 = (v8 - 1) & (v13 + v14);
    v15 = (_QWORD *)(v11 + 32LL * v14);
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
LABEL_21:
    v38 = result;
    sub_386BC40(a2, 2 * v8);
    v24 = *(_DWORD *)(a2 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a2 + 8);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v38;
      v27 = (v24 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v12 = (_QWORD *)(v26 + 32LL * v27);
      v28 = *v12;
      if ( *a3 == *v12 )
        goto LABEL_14;
      v29 = 1;
      v30 = 0;
      while ( v28 != -8 )
      {
        if ( !v30 && v28 == -16 )
          v30 = v12;
        v27 = v25 & (v29 + v27);
        v12 = (_QWORD *)(v26 + 32LL * v27);
        v28 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v29;
      }
LABEL_25:
      if ( v30 )
        v12 = v30;
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v18 <= (unsigned int)v8 >> 3 )
  {
    v39 = result;
    sub_386BC40(a2, v8);
    v31 = *(_DWORD *)(a2 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a2 + 8);
      v30 = 0;
      v34 = 1;
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v39;
      v35 = (v31 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v12 = (_QWORD *)(v33 + 32LL * v35);
      v36 = *v12;
      if ( *a3 == *v12 )
        goto LABEL_14;
      while ( v36 != -8 )
      {
        if ( v36 == -16 && !v30 )
          v30 = v12;
        v35 = v32 & (v34 + v35);
        v12 = (_QWORD *)(v33 + 32LL * v35);
        v36 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v34;
      }
      goto LABEL_25;
    }
    goto LABEL_41;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v12 != -8 )
    --*(_DWORD *)(a2 + 20);
  v19 = *a3;
  v12[1] = 6;
  v12[2] = 0;
  *v12 = v19;
  v20 = a4[2];
  v12[3] = v20;
  if ( v20 != -8 && v20 != 0 && v20 != -16 )
  {
    v37 = result;
    sub_1649AC0(v12 + 1, *a4 & 0xFFFFFFFFFFFFFFF8LL);
    result = v37;
  }
  v21 = *(unsigned int *)(a2 + 24);
  v22 = *(_QWORD *)a2;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v12;
  v23 = *(_QWORD *)(a2 + 8) + 32 * v21;
  *(_QWORD *)(result + 8) = v22;
  *(_QWORD *)(result + 24) = v23;
  *(_BYTE *)(result + 32) = 1;
  return result;
}
