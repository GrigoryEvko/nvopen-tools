// Function: sub_D440B0
// Address: 0xd440b0
//
__int64 __fastcall sub_D440B0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  int v6; // r9d
  __int64 *v7; // r13
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  int v12; // eax
  int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 *v17; // rbx
  __int64 v18; // rcx
  __int64 *v19; // rdi
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rsi
  int v25; // r9d
  __int64 *v26; // r8
  int v27; // eax
  int v28; // eax
  __int64 v29; // rsi
  int v30; // r8d
  unsigned int v31; // r14d
  __int64 *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( a2 == *v9 )
    return v9[1];
  while ( v10 != -4096 )
  {
    if ( !v7 && v10 == -8192 )
      v7 = v9;
    v8 = (v4 - 1) & (v6 + v8);
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      return v9[1];
    ++v6;
  }
  if ( !v7 )
    v7 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_22:
    sub_D3FE80(a1, 2 * v4);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v7 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v7;
      if ( a2 != *v7 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -4096 )
        {
          if ( !v26 && v24 == -8192 )
            v26 = v7;
          v23 = v21 & (v25 + v23);
          v7 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v7;
          if ( a2 == *v7 )
            goto LABEL_15;
          ++v25;
        }
        if ( v26 )
          v7 = v26;
      }
      goto LABEL_15;
    }
    goto LABEL_45;
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_D3FE80(a1, v4);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 8);
      v30 = 1;
      v31 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = 0;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v7 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v7;
      if ( a2 != *v7 )
      {
        while ( v33 != -4096 )
        {
          if ( !v32 && v33 == -8192 )
            v32 = v7;
          v31 = v28 & (v30 + v31);
          v7 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v7;
          if ( a2 == *v7 )
            goto LABEL_15;
          ++v30;
        }
        if ( v32 )
          v7 = v32;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = a2;
  v7[1] = 0;
  v14 = *(_QWORD *)(a1 + 64);
  v15 = *(_QWORD *)(a1 + 56);
  v16 = *(_QWORD *)(a1 + 48);
  v34 = *(_QWORD *)(a1 + 40);
  v35 = *(_QWORD *)(a1 + 32);
  v17 = *(__int64 **)(a1 + 72);
  v36 = v14;
  result = sub_22077B0(152);
  if ( result )
  {
    v18 = v36;
    v37 = result;
    sub_D43BB0(result, a2, v35, v18, v17, v34, v16, v15);
    result = v37;
  }
  v19 = (__int64 *)v7[1];
  v7[1] = result;
  if ( v19 )
  {
    sub_D33160(v19);
    return v7[1];
  }
  return result;
}
