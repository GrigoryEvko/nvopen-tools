// Function: sub_25D0260
// Address: 0x25d0260
//
__int64 __fastcall sub_25D0260(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned int v8; // esi
  int v9; // eax
  int v10; // r12d
  __int64 v11; // rdi
  int *v12; // r10
  int v13; // r11d
  int v14; // r13d
  unsigned int v15; // ecx
  int *v16; // rdx
  int v17; // eax
  int v19; // eax
  int v20; // edx
  int v21; // eax
  __int64 v22; // rsi
  int v23; // r12d
  int v24; // eax
  unsigned int v25; // edx
  int *v26; // rcx
  int v27; // edi
  int v28; // ecx
  int v29; // r8d
  int v30; // eax
  int v31; // ecx
  __int64 v32; // rdi
  unsigned int v33; // eax
  int v34; // esi
  int v35; // r9d
  int *v36; // r8
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  int v40; // r8d
  unsigned int v41; // r13d
  int *v42; // rdi
  int v43; // ecx
  __int64 v44; // [rsp+8h] [rbp-48h] BYREF
  __m128i v45; // [rsp+10h] [rbp-40h] BYREF
  __int64 v46; // [rsp+20h] [rbp-30h]

  v5 = *(_QWORD *)a1;
  v45.m128i_i64[0] = a2;
  v45.m128i_i64[1] = a3;
  v6 = *(unsigned int *)(v5 + 40);
  v46 = a4;
  v44 = v6;
  v7 = sub_25D0030(v5, &v45, &v44);
  v8 = *(_DWORD *)(a1 + 32);
  v9 = *(_DWORD *)(v7 + 24);
  v10 = 2 * v9;
  if ( v8 )
  {
    v11 = *(_QWORD *)(a1 + 16);
    v12 = 0;
    v13 = 1;
    v14 = 74 * v9;
    v15 = (v8 - 1) & (74 * v9);
    v16 = (int *)(v11 + 4LL * v15);
    v17 = *v16;
    if ( v10 == *v16 )
      return 0;
    while ( v17 != -1 )
    {
      if ( v12 || v17 != -2 )
        v16 = v12;
      v15 = (v8 - 1) & (v13 + v15);
      v17 = *(_DWORD *)(v11 + 4LL * v15);
      if ( v10 == v17 )
        return 0;
      ++v13;
      v12 = v16;
      v16 = (int *)(v11 + 4LL * v15);
    }
    v19 = *(_DWORD *)(a1 + 24);
    if ( !v12 )
      v12 = v16;
    ++*(_QWORD *)(a1 + 8);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 28) - v20 > v8 >> 3 )
        goto LABEL_13;
      sub_A08C50(a1 + 8, v8);
      v37 = *(_DWORD *)(a1 + 32);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 16);
        v40 = 1;
        v41 = v38 & v14;
        v12 = (int *)(v39 + 4LL * v41);
        v20 = *(_DWORD *)(a1 + 24) + 1;
        v42 = 0;
        v43 = *v12;
        if ( v10 != *v12 )
        {
          while ( v43 != -1 )
          {
            if ( v43 == -2 && !v42 )
              v42 = v12;
            v41 = v38 & (v40 + v41);
            v12 = (int *)(v39 + 4LL * v41);
            v43 = *v12;
            if ( v10 == *v12 )
              goto LABEL_13;
            ++v40;
          }
          if ( v42 )
            v12 = v42;
        }
        goto LABEL_13;
      }
LABEL_48:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 8);
  }
  sub_A08C50(a1 + 8, 2 * v8);
  v30 = *(_DWORD *)(a1 + 32);
  if ( !v30 )
    goto LABEL_48;
  v31 = v30 - 1;
  v32 = *(_QWORD *)(a1 + 16);
  v33 = (v30 - 1) & (37 * v10);
  v12 = (int *)(v32 + 4LL * v33);
  v34 = *v12;
  v20 = *(_DWORD *)(a1 + 24) + 1;
  if ( v10 != *v12 )
  {
    v35 = 1;
    v36 = 0;
    while ( v34 != -1 )
    {
      if ( v34 == -2 && !v36 )
        v36 = v12;
      v33 = v31 & (v35 + v33);
      v12 = (int *)(v32 + 4LL * v33);
      v34 = *v12;
      if ( v10 == *v12 )
        goto LABEL_13;
      ++v35;
    }
    if ( v36 )
      v12 = v36;
  }
LABEL_13:
  *(_DWORD *)(a1 + 24) = v20;
  if ( *v12 != -1 )
    --*(_DWORD *)(a1 + 28);
  *v12 = v10;
  v21 = *(_DWORD *)(a1 + 32);
  v22 = *(_QWORD *)(a1 + 16);
  if ( v21 )
  {
    v23 = v10 | 1;
    v24 = v21 - 1;
    v25 = v24 & (37 * v23);
    v26 = (int *)(v22 + 4LL * v25);
    v27 = *v26;
    if ( v23 == *v26 )
    {
LABEL_17:
      *v26 = -2;
      --*(_DWORD *)(a1 + 24);
      ++*(_DWORD *)(a1 + 28);
      return 2;
    }
    v28 = 1;
    while ( v27 != -1 )
    {
      v29 = v28 + 1;
      v25 = v24 & (v28 + v25);
      v26 = (int *)(v22 + 4LL * v25);
      v27 = *v26;
      if ( v23 == *v26 )
        goto LABEL_17;
      v28 = v29;
    }
  }
  return 1;
}
