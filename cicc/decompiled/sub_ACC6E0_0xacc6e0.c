// Function: sub_ACC6E0
// Address: 0xacc6e0
//
__int64 __fastcall sub_ACC6E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r14d
  __int64 v9; // r8
  __int64 *v10; // rdx
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r10
  __int64 *v14; // rbx
  __int64 result; // rax
  int v16; // eax
  int v17; // ecx
  int v18; // eax
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rdi
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  __int64 *v28; // r8
  unsigned int v29; // r13d
  int v30; // r9d
  __int64 v31; // rsi
  __int64 v32; // [rsp+8h] [rbp-28h]

  v4 = sub_BD5C60(a1, a2, a3);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(*(_QWORD *)v4 + 2048LL);
  v7 = *(_QWORD *)v4 + 2024LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 2024);
    goto LABEL_22;
  }
  v8 = 1;
  v9 = *(_QWORD *)(v5 + 2032);
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a1 )
  {
LABEL_3:
    v14 = v12 + 1;
    result = v12[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = (v6 - 1) & (v8 + v11);
    v12 = (__int64 *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a1 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v10 )
    v10 = v12;
  v16 = *(_DWORD *)(v5 + 2040);
  ++*(_QWORD *)(v5 + 2024);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v6 )
  {
LABEL_22:
    sub_ACC500(v7, 2 * v6);
    v18 = *(_DWORD *)(v5 + 2048);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v5 + 2032);
      v21 = (v18 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v17 = *(_DWORD *)(v5 + 2040) + 1;
      v10 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v10;
      if ( *v10 != a1 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v10;
          v21 = v19 & (v23 + v21);
          v10 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v10;
          if ( *v10 == a1 )
            goto LABEL_15;
          ++v23;
        }
        if ( v24 )
          v10 = v24;
      }
      goto LABEL_15;
    }
    goto LABEL_45;
  }
  if ( v6 - *(_DWORD *)(v5 + 2044) - v17 <= v6 >> 3 )
  {
    sub_ACC500(v7, v6);
    v25 = *(_DWORD *)(v5 + 2048);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v5 + 2032);
      v28 = 0;
      v29 = v26 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v30 = 1;
      v17 = *(_DWORD *)(v5 + 2040) + 1;
      v10 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v10;
      if ( *v10 != a1 )
      {
        while ( v31 != -4096 )
        {
          if ( !v28 && v31 == -8192 )
            v28 = v10;
          v29 = v26 & (v30 + v29);
          v10 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v10;
          if ( *v10 == a1 )
            goto LABEL_15;
          ++v30;
        }
        if ( v28 )
          v10 = v28;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(v5 + 2040);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v5 + 2040) = v17;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v5 + 2044);
  *v10 = a1;
  v14 = v10 + 1;
  v10[1] = 0;
LABEL_18:
  result = sub_BD2C40(24, unk_3F2899C);
  if ( result )
  {
    v32 = result;
    sub_AC4180(result, a1);
    result = v32;
  }
  *v14 = result;
  return result;
}
