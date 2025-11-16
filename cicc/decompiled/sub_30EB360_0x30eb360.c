// Function: sub_30EB360
// Address: 0x30eb360
//
__int64 __fastcall sub_30EB360(__int64 a1)
{
  unsigned int v2; // esi
  __int64 v3; // r12
  unsigned int v4; // edx
  __int64 v5; // r9
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // rdi
  int v10; // eax
  int v11; // r8d
  __int64 v12; // r14
  __int64 *v13; // r15
  __int64 *v14; // r8
  int v15; // eax
  int v16; // edx
  int v17; // edx
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r9d
  unsigned int v29; // r13d
  __int64 *v30; // rdi
  __int64 v31; // rcx
  int v32; // [rsp+8h] [rbp-38h]
  unsigned int v33; // [rsp+Ch] [rbp-34h]

  sub_30E3440(a1);
  v2 = *(_DWORD *)(a1 + 208);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 16))-- - 8);
  if ( !v2 )
  {
    ++*(_QWORD *)(a1 + 184);
    goto LABEL_21;
  }
  v4 = v2 - 1;
  v5 = *(_QWORD *)(a1 + 192);
  v6 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( v3 == *v7 )
    goto LABEL_3;
  v33 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v12 = *v7;
  v13 = (__int64 *)(v5 + 16LL * (v4 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4))));
  v14 = 0;
  v32 = 1;
  while ( v12 != -4096 )
  {
    if ( v14 || v12 != -8192 )
      v13 = v14;
    v33 = v4 & (v33 + v32);
    v12 = *(_QWORD *)(v5 + 16LL * v33);
    if ( v3 == v12 )
      goto LABEL_3;
    ++v32;
    v14 = v13;
    v13 = (__int64 *)(v5 + 16LL * v33);
  }
  v15 = *(_DWORD *)(a1 + 200);
  if ( !v14 )
    v14 = v13;
  ++*(_QWORD *)(a1 + 184);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v2 )
  {
LABEL_21:
    sub_30E3EE0(a1 + 184, 2 * v2);
    v18 = *(_DWORD *)(a1 + 208);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 192);
      v21 = (v18 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v16 = *(_DWORD *)(a1 + 200) + 1;
      v14 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v14;
      if ( v3 != *v14 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( v22 == -8192 && !v24 )
            v24 = v14;
          v21 = v19 & (v23 + v21);
          v14 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v14;
          if ( v3 == *v14 )
            goto LABEL_16;
          ++v23;
        }
        if ( v24 )
          v14 = v24;
      }
      goto LABEL_16;
    }
    goto LABEL_49;
  }
  if ( v2 - *(_DWORD *)(a1 + 204) - v16 <= v2 >> 3 )
  {
    sub_30E3EE0(a1 + 184, v2);
    v25 = *(_DWORD *)(a1 + 208);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 192);
      v28 = 1;
      v29 = v26 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v16 = *(_DWORD *)(a1 + 200) + 1;
      v30 = 0;
      v14 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v14;
      if ( v3 != *v14 )
      {
        while ( v31 != -4096 )
        {
          if ( v31 == -8192 && !v30 )
            v30 = v14;
          v29 = v26 & (v28 + v29);
          v14 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v14;
          if ( v3 == *v14 )
            goto LABEL_16;
          ++v28;
        }
        if ( v30 )
          v14 = v30;
      }
      goto LABEL_16;
    }
LABEL_49:
    ++*(_DWORD *)(a1 + 200);
    BUG();
  }
LABEL_16:
  *(_DWORD *)(a1 + 200) = v16;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a1 + 204);
  *v14 = v3;
  *((_DWORD *)v14 + 2) = 0;
  v17 = *(_DWORD *)(a1 + 208);
  if ( !v17 )
    return v3;
  v4 = v17 - 1;
  v5 = *(_QWORD *)(a1 + 192);
  v6 = v4 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
LABEL_3:
  if ( v3 == v8 )
  {
LABEL_4:
    *v7 = -8192;
    --*(_DWORD *)(a1 + 200);
    ++*(_DWORD *)(a1 + 204);
  }
  else
  {
    v10 = 1;
    while ( v8 != -4096 )
    {
      v11 = v10 + 1;
      v6 = v4 & (v10 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( v3 == *v7 )
        goto LABEL_4;
      v10 = v11;
    }
  }
  return v3;
}
