// Function: sub_11B4BB0
// Address: 0x11b4bb0
//
__int64 __fastcall sub_11B4BB0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // rdi
  _QWORD *v10; // r11
  int v11; // r15d
  unsigned int v12; // ecx
  _QWORD *v13; // rdx
  __int64 v14; // r14
  int v15; // edi
  int v16; // ecx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rdx
  int v21; // edx
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // edx
  __int64 v25; // r9
  int v26; // r13d
  _QWORD *v27; // r10
  int v28; // edx
  int v29; // esi
  __int64 v30; // r8
  int v31; // r13d
  unsigned int v32; // edx
  __int64 v33; // r9
  __int64 v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  result = a1;
  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
    goto LABEL_18;
  }
  v8 = *a3;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v6 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v13 = (_QWORD *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = v7;
    *(_QWORD *)(result + 16) = v13;
    *(_QWORD *)(result + 24) = v9 + 16 * v6;
    *(_BYTE *)(result + 32) = 0;
    return result;
  }
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v6 - 1) & (v11 + v12);
    v13 = (_QWORD *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v11;
  }
  v15 = *(_DWORD *)(a2 + 16);
  if ( !v10 )
    v10 = v13;
  v16 = v15 + 1;
  *(_QWORD *)a2 = v7 + 1;
  if ( 4 * (v15 + 1) >= (unsigned int)(3 * v6) )
  {
LABEL_18:
    v34 = result;
    sub_11B49D0(a2, 2 * v6);
    v21 = *(_DWORD *)(a2 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a2 + 8);
      v16 = *(_DWORD *)(a2 + 16) + 1;
      result = v34;
      v24 = (v21 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v10 = (_QWORD *)(v23 + 16LL * v24);
      v25 = *v10;
      if ( *v10 == *a3 )
        goto LABEL_14;
      v26 = 1;
      v27 = 0;
      while ( v25 != -4096 )
      {
        if ( !v27 && v25 == -8192 )
          v27 = v10;
        v24 = v22 & (v26 + v24);
        v10 = (_QWORD *)(v23 + 16LL * v24);
        v25 = *v10;
        if ( *a3 == *v10 )
          goto LABEL_14;
        ++v26;
      }
LABEL_22:
      if ( v27 )
        v10 = v27;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v16 <= (unsigned int)v6 >> 3 )
  {
    v35 = result;
    sub_11B49D0(a2, v6);
    v28 = *(_DWORD *)(a2 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a2 + 8);
      v27 = 0;
      v31 = 1;
      v16 = *(_DWORD *)(a2 + 16) + 1;
      result = v35;
      v32 = (v28 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v10 = (_QWORD *)(v30 + 16LL * v32);
      v33 = *v10;
      if ( *a3 == *v10 )
        goto LABEL_14;
      while ( v33 != -4096 )
      {
        if ( v33 == -8192 && !v27 )
          v27 = v10;
        v32 = v29 & (v31 + v32);
        v10 = (_QWORD *)(v30 + 16LL * v32);
        v33 = *v10;
        if ( *a3 == *v10 )
          goto LABEL_14;
        ++v31;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v17 = *a3;
  v10[1] = 0;
  *(_QWORD *)result = a2;
  *v10 = v17;
  v18 = *(unsigned int *)(a2 + 24);
  v19 = *(_QWORD *)a2;
  *(_QWORD *)(result + 16) = v10;
  v20 = *(_QWORD *)(a2 + 8) + 16 * v18;
  *(_BYTE *)(result + 32) = 1;
  *(_QWORD *)(result + 8) = v19;
  *(_QWORD *)(result + 24) = v20;
  return result;
}
