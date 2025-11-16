// Function: sub_2AFF8B0
// Address: 0x2aff8b0
//
unsigned __int64 __fastcall sub_2AFF8B0(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 *v8; // r10
  int v9; // r14d
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r11
  unsigned __int64 v13; // rcx
  unsigned __int64 i; // rsi
  int v15; // edx
  __int64 v16; // r8
  int v17; // edi
  __int64 v18; // r9
  int v19; // ecx
  unsigned int v20; // esi
  __int64 v21; // rdx
  int v22; // r13d
  __int64 *v23; // r11
  int v24; // edi
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // r8
  int v28; // edi
  __int64 v29; // r9
  int v30; // r13d
  unsigned int v31; // esi
  __int64 v32; // rdx
  unsigned __int64 v33; // [rsp+8h] [rbp-28h]
  unsigned __int64 v34; // [rsp+8h] [rbp-28h]

  result = a2;
  if ( a2
    && (a2 & (a2 - 1)) != 0
    && (v13 = ((((((a2 | (a2 >> 1)) >> 2)
                | a2
                | (a2 >> 1)
                | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)
                | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1) | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)) >> 8)) >> 16)
              | ((a2 | (a2 >> 1)) >> 2)
              | a2
              | (a2 >> 1)
              | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)
              | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1) | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)) >> 8)
              | ((((((a2 | (a2 >> 1)) >> 2)
                  | a2
                  | (a2 >> 1)
                  | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)
                  | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1) | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)) >> 8)) >> 16)
                | ((a2 | (a2 >> 1)) >> 2)
                | a2
                | (a2 >> 1)
                | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)
                | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1) | ((((a2 | (a2 >> 1)) >> 2) | a2 | (a2 >> 1)) >> 4)) >> 8)) >> 32))
             + 1) >> 1) != 0 )
  {
    for ( i = a2 % v13; ; i = result % i )
    {
      result = v13;
      if ( !i )
        break;
      v13 = i;
    }
    v4 = *a1;
    v5 = *(_DWORD *)(*a1 + 24);
    if ( !v5 )
      goto LABEL_11;
  }
  else
  {
    v4 = *a1;
    v5 = *(_DWORD *)(*a1 + 24);
    if ( !v5 )
    {
LABEL_11:
      ++*(_QWORD *)v4;
      goto LABEL_12;
    }
  }
  v6 = a1[1];
  v7 = *(_QWORD *)(v4 + 8);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)a1[1] >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (__int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( v6 == *v11 )
  {
LABEL_5:
    v11[1] = result;
    return result;
  }
  while ( v12 != -4096 )
  {
    if ( !v8 && v12 == -8192 )
      v8 = v11;
    v10 = (v5 - 1) & (v9 + v10);
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
      goto LABEL_5;
    ++v9;
  }
  v24 = *(_DWORD *)(v4 + 16);
  if ( !v8 )
    v8 = v11;
  ++*(_QWORD *)v4;
  v19 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v5 )
  {
LABEL_12:
    v33 = result;
    sub_2AFF6D0(v4, 2 * v5);
    v15 = *(_DWORD *)(v4 + 24);
    if ( v15 )
    {
      v16 = a1[1];
      v17 = v15 - 1;
      v18 = *(_QWORD *)(v4 + 8);
      v19 = *(_DWORD *)(v4 + 16) + 1;
      result = v33;
      v20 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v8 = (__int64 *)(v18 + 16LL * v20);
      v21 = *v8;
      if ( *v8 != v16 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v8;
          v20 = v17 & (v22 + v20);
          v8 = (__int64 *)(v18 + 16LL * v20);
          v21 = *v8;
          if ( v16 == *v8 )
            goto LABEL_29;
          ++v22;
        }
LABEL_16:
        if ( v23 )
          v8 = v23;
        goto LABEL_29;
      }
      goto LABEL_29;
    }
LABEL_45:
    ++*(_DWORD *)(v4 + 16);
    BUG();
  }
  if ( v5 - *(_DWORD *)(v4 + 20) - v19 > v5 >> 3 )
    goto LABEL_29;
  v34 = result;
  sub_2AFF6D0(v4, v5);
  v26 = *(_DWORD *)(v4 + 24);
  if ( !v26 )
    goto LABEL_45;
  v27 = a1[1];
  v28 = v26 - 1;
  v29 = *(_QWORD *)(v4 + 8);
  v23 = 0;
  v30 = 1;
  v19 = *(_DWORD *)(v4 + 16) + 1;
  result = v34;
  v31 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v8 = (__int64 *)(v29 + 16LL * v31);
  v32 = *v8;
  if ( v27 != *v8 )
  {
    while ( v32 != -4096 )
    {
      if ( !v23 && v32 == -8192 )
        v23 = v8;
      v31 = v28 & (v30 + v31);
      v8 = (__int64 *)(v29 + 16LL * v31);
      v32 = *v8;
      if ( v27 == *v8 )
        goto LABEL_29;
      ++v30;
    }
    goto LABEL_16;
  }
LABEL_29:
  *(_DWORD *)(v4 + 16) = v19;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v4 + 20);
  v25 = a1[1];
  v8[1] = 0;
  *v8 = v25;
  v8[1] = result;
  return result;
}
