// Function: sub_2F5B510
// Address: 0x2f5b510
//
__int64 __fastcall sub_2F5B510(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // r10
  __int64 *v10; // r13
  int v11; // r14d
  unsigned int v12; // edx
  __int64 *v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdi
  char v16; // si
  int v17; // edi
  int v18; // edi
  int v19; // edx
  int v20; // esi
  __int64 v21; // r10
  unsigned int v22; // edx
  __int64 v23; // r9
  int v24; // r13d
  __int64 *v25; // r11
  int v26; // edx
  int v27; // esi
  __int64 v28; // r10
  int v29; // r13d
  unsigned int v30; // edx
  __int64 v31; // r9
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]

  result = a1;
  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
    goto LABEL_19;
  }
  v8 = *a3;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v6 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v13 = (__int64 *)(v9 + 8LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    v15 = v9 + 8 * v6;
    v16 = 0;
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v6 - 1) & (v11 + v12);
    v13 = (__int64 *)(v9 + 8LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v11;
  }
  v17 = *(_DWORD *)(a2 + 16);
  if ( v10 )
    v13 = v10;
  *(_QWORD *)a2 = v7 + 1;
  v18 = v17 + 1;
  if ( 4 * v18 >= (unsigned int)(3 * v6) )
  {
LABEL_19:
    v32 = result;
    sub_2F5B340(a2, 2 * v6);
    v19 = *(_DWORD *)(a2 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a2 + 8);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v32;
      v22 = (v19 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v13 = (__int64 *)(v21 + 8LL * v22);
      v23 = *v13;
      if ( *v13 == *a3 )
        goto LABEL_15;
      v24 = 1;
      v25 = 0;
      while ( v23 != -4096 )
      {
        if ( !v25 && v23 == -8192 )
          v25 = v13;
        v22 = v20 & (v24 + v22);
        v13 = (__int64 *)(v21 + 8LL * v22);
        v23 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_15;
        ++v24;
      }
LABEL_23:
      if ( v25 )
        v13 = v25;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v18 <= (unsigned int)v6 >> 3 )
  {
    v33 = result;
    sub_2F5B340(a2, v6);
    v26 = *(_DWORD *)(a2 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a2 + 8);
      v25 = 0;
      v29 = 1;
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v33;
      v30 = (v26 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v13 = (__int64 *)(v28 + 8LL * v30);
      v31 = *v13;
      if ( *a3 == *v13 )
        goto LABEL_15;
      while ( v31 != -4096 )
      {
        if ( v31 == -8192 && !v25 )
          v25 = v13;
        v30 = v27 & (v29 + v30);
        v13 = (__int64 *)(v28 + 8LL * v30);
        v31 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_15;
        ++v29;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v13 = *a3;
  v7 = *(_QWORD *)a2;
  v15 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
  v16 = 1;
LABEL_4:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v7;
  *(_QWORD *)(result + 16) = v13;
  *(_QWORD *)(result + 24) = v15;
  *(_BYTE *)(result + 32) = v16;
  return result;
}
