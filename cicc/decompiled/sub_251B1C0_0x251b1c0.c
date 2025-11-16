// Function: sub_251B1C0
// Address: 0x251b1c0
//
__int64 __fastcall sub_251B1C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r14d
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  _QWORD *v12; // rbx
  __int64 result; // rax
  int v14; // eax
  int v15; // ecx
  _QWORD *v16; // rdx
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rdi
  int v22; // r10d
  __int64 *v23; // r9
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  __int64 *v27; // r8
  unsigned int v28; // ebx
  int v29; // r9d
  __int64 v30; // rsi

  v4 = a1 + 8;
  v5 = *(_DWORD *)(a1 + 32);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_22;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    v12 = v10 + 1;
    result = v10[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_22:
    sub_25132F0(v4, 2 * v5);
    v17 = *(_DWORD *)(a1 + 32);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 16);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 24) + 1;
      v8 = (__int64 *)(v19 + 16LL * v20);
      v21 = *v8;
      if ( a2 != *v8 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v8;
          v20 = v18 & (v22 + v20);
          v8 = (__int64 *)(v19 + 16LL * v20);
          v21 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v22;
        }
        if ( v23 )
          v8 = v23;
      }
      goto LABEL_15;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 28) - v15 <= v5 >> 3 )
  {
    sub_25132F0(v4, v5);
    v24 = *(_DWORD *)(a1 + 32);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 16);
      v27 = 0;
      v28 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = 1;
      v15 = *(_DWORD *)(a1 + 24) + 1;
      v8 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v8;
      if ( a2 != *v8 )
      {
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v27 )
            v27 = v8;
          v28 = v25 & (v29 + v28);
          v8 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v29;
        }
        if ( v27 )
          v8 = v27;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 24) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v8 = a2;
  v12 = v8 + 1;
  v8[1] = 0;
LABEL_18:
  v16 = (_QWORD *)sub_A777F0(0x78u, *(__int64 **)(a1 + 112));
  if ( v16 )
  {
    memset(v16, 0, 0x78u);
    v16[4] = v16 + 6;
    v16[5] = 0x800000000LL;
  }
  *v12 = v16;
  sub_251A430(a1, a2, (__int64)v16);
  return *v12;
}
