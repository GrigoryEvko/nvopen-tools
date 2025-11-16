// Function: sub_1F35DB0
// Address: 0x1f35db0
//
__int64 __fastcall sub_1F35DB0(__int64 a1, __int64 a2, int *a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  int v10; // edx
  __int64 v11; // r8
  int v12; // r15d
  _DWORD *v13; // r10
  unsigned int v14; // ecx
  _DWORD *v15; // rdi
  int v16; // r11d
  int v17; // ecx
  int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edx
  int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // edx
  int v25; // r8d
  int v26; // r14d
  _DWORD *v27; // r11
  int v28; // edx
  int v29; // edx
  __int64 v30; // r9
  int v31; // r14d
  unsigned int v32; // r8d
  int v33; // edi
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
  v12 = 1;
  v13 = 0;
  v14 = (v8 - 1) & (37 * v10);
  v15 = (_DWORD *)(v11 + 12LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
  {
LABEL_3:
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = v9;
    *(_QWORD *)(result + 16) = v15;
    *(_QWORD *)(result + 24) = v11 + 12 * v8;
    *(_BYTE *)(result + 32) = 0;
    return result;
  }
  while ( v16 != -1 )
  {
    if ( !v13 && v16 == -2 )
      v13 = v15;
    v14 = (v8 - 1) & (v12 + v14);
    v15 = (_DWORD *)(v11 + 12LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
    ++v12;
  }
  if ( !v13 )
    v13 = v15;
  v17 = *(_DWORD *)(a2 + 16) + 1;
  *(_QWORD *)a2 = v9 + 1;
  if ( 4 * v17 >= (unsigned int)(3 * v8) )
  {
LABEL_18:
    v34 = result;
    sub_1F35BF0(a2, 2 * v8);
    v21 = *(_DWORD *)(a2 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a2 + 8);
      v24 = (v21 - 1) & (37 * *a3);
      v13 = (_DWORD *)(v23 + 12LL * (v22 & (unsigned int)(37 * *a3)));
      v17 = *(_DWORD *)(a2 + 16) + 1;
      result = v34;
      v25 = *v13;
      if ( *v13 == *a3 )
        goto LABEL_14;
      v26 = 1;
      v27 = 0;
      while ( v25 != -1 )
      {
        if ( !v27 && v25 == -2 )
          v27 = v13;
        v24 = v22 & (v26 + v24);
        v13 = (_DWORD *)(v23 + 12LL * v24);
        v25 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_14;
        ++v26;
      }
LABEL_22:
      if ( v27 )
        v13 = v27;
      goto LABEL_14;
    }
LABEL_38:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v17 <= (unsigned int)v8 >> 3 )
  {
    v35 = result;
    sub_1F35BF0(a2, v8);
    v28 = *(_DWORD *)(a2 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a2 + 8);
      v27 = 0;
      v31 = 1;
      v32 = v29 & (37 * *a3);
      v13 = (_DWORD *)(v30 + 12LL * v32);
      v17 = *(_DWORD *)(a2 + 16) + 1;
      result = v35;
      v33 = *v13;
      if ( *a3 == *v13 )
        goto LABEL_14;
      while ( v33 != -1 )
      {
        if ( v33 == -2 && !v27 )
          v27 = v13;
        v32 = v29 & (v31 + v32);
        v13 = (_DWORD *)(v30 + 12LL * v32);
        v33 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_14;
        ++v31;
      }
      goto LABEL_22;
    }
    goto LABEL_38;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v17;
  if ( *v13 != -1 )
    --*(_DWORD *)(a2 + 20);
  v18 = *a3;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v13;
  *v13 = v18;
  v19 = *a4;
  *(_BYTE *)(result + 32) = 1;
  *(_QWORD *)(v13 + 1) = v19;
  v20 = *(_QWORD *)a2;
  *(_QWORD *)(result + 24) = *(_QWORD *)(a2 + 8) + 12LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(result + 8) = v20;
  return result;
}
