// Function: sub_D7AC80
// Address: 0xd7ac80
//
__int64 __fastcall sub_D7AC80(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r10
  __int64 *v10; // r13
  int v11; // r14d
  unsigned int v12; // edx
  __int64 *v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rsi
  char v16; // dl
  int v17; // edi
  int v18; // ecx
  int v19; // esi
  __int64 v20; // r10
  unsigned int v21; // edx
  __int64 v22; // r9
  int v23; // r13d
  __int64 *v24; // r11
  int v25; // ecx
  int v26; // esi
  __int64 v27; // r10
  int v28; // r13d
  unsigned int v29; // edx
  __int64 v30; // r9
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]

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
  v12 = (v6 - 1) & (((0xBF58476D1CE4E5B9LL * *a3) >> 31) ^ (484763065 * *(_DWORD *)a3));
  v13 = (__int64 *)(v9 + 8LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    v15 = v9 + 8 * v6;
    v16 = 0;
    goto LABEL_4;
  }
  while ( v14 != -1 )
  {
    if ( !v10 && v14 == -2 )
      v10 = v13;
    v12 = (v6 - 1) & (v11 + v12);
    v13 = (__int64 *)(v9 + 8LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v11;
  }
  if ( v10 )
    v13 = v10;
  *(_QWORD *)a2 = v7 + 1;
  v17 = *(_DWORD *)(a2 + 16) + 1;
  if ( 4 * v17 >= (unsigned int)(3 * v6) )
  {
LABEL_19:
    v31 = result;
    sub_A32210(a2, 2 * v6);
    v18 = *(_DWORD *)(a2 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a2 + 8);
      v17 = *(_DWORD *)(a2 + 16) + 1;
      result = v31;
      v21 = (v18 - 1) & (((0xBF58476D1CE4E5B9LL * *a3) >> 31) ^ (484763065 * *(_DWORD *)a3));
      v13 = (__int64 *)(v20 + 8LL * v21);
      v22 = *v13;
      if ( *v13 == *a3 )
        goto LABEL_15;
      v23 = 1;
      v24 = 0;
      while ( v22 != -1 )
      {
        if ( !v24 && v22 == -2 )
          v24 = v13;
        v21 = v19 & (v23 + v21);
        v13 = (__int64 *)(v20 + 8LL * v21);
        v22 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_15;
        ++v23;
      }
LABEL_23:
      if ( v24 )
        v13 = v24;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v17 <= (unsigned int)v6 >> 3 )
  {
    v32 = result;
    sub_A32210(a2, v6);
    v25 = *(_DWORD *)(a2 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a2 + 8);
      v24 = 0;
      v28 = 1;
      v17 = *(_DWORD *)(a2 + 16) + 1;
      result = v32;
      v29 = (v25 - 1) & (((0xBF58476D1CE4E5B9LL * *a3) >> 31) ^ (484763065 * *(_DWORD *)a3));
      v13 = (__int64 *)(v27 + 8LL * v29);
      v30 = *v13;
      if ( *a3 == *v13 )
        goto LABEL_15;
      while ( v30 != -1 )
      {
        if ( v30 == -2 && !v24 )
          v24 = v13;
        v29 = v26 & (v28 + v29);
        v13 = (__int64 *)(v27 + 8LL * v29);
        v30 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_15;
        ++v28;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v17;
  if ( *v13 != -1 )
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
