// Function: sub_1628DA0
// Address: 0x1628da0
//
__int64 __fastcall sub_1628DA0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // r14d
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  int v15; // r10d
  __int64 *v16; // r15
  int v17; // eax
  int v18; // edx
  __int64 v19; // r13
  __int64 v20; // rax
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 *v27; // r8
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  __int64 *v31; // rdi
  unsigned int v32; // r14d
  int v33; // r8d
  __int64 v34; // rcx

  v3 = sub_1628D40(a1, a2);
  v4 = *a1;
  v5 = v3;
  v6 = *(_DWORD *)(*a1 + 456);
  v7 = *a1 + 432;
  if ( !v6 )
  {
    ++*(_QWORD *)(v4 + 432);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(v4 + 440);
  v9 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v10 = (v6 - 1) & v9;
  v11 = (__int64 *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v5 != *v11 )
  {
    v15 = 1;
    v16 = 0;
    while ( v12 != -4 )
    {
      if ( !v16 && v12 == -8 )
        v16 = v11;
      v10 = (v6 - 1) & (v15 + v10);
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v5 == *v11 )
        goto LABEL_3;
      ++v15;
    }
    if ( !v16 )
      v16 = v11;
    v17 = *(_DWORD *)(v4 + 448);
    ++*(_QWORD *)(v4 + 432);
    v18 = v17 + 1;
    if ( 4 * (v17 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v4 + 452) - v18 > v6 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(v4 + 448) = v18;
        if ( *v16 != -4 )
          --*(_DWORD *)(v4 + 452);
        *v16 = v5;
        v16[1] = 0;
        goto LABEL_14;
      }
      sub_16228F0(v7, v6);
      v28 = *(_DWORD *)(v4 + 456);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(v4 + 440);
        v31 = 0;
        v32 = v29 & v9;
        v33 = 1;
        v18 = *(_DWORD *)(v4 + 448) + 1;
        v16 = (__int64 *)(v30 + 16LL * v32);
        v34 = *v16;
        if ( v5 != *v16 )
        {
          while ( v34 != -4 )
          {
            if ( !v31 && v34 == -8 )
              v31 = v16;
            v32 = v29 & (v33 + v32);
            v16 = (__int64 *)(v30 + 16LL * v32);
            v34 = *v16;
            if ( v5 == *v16 )
              goto LABEL_11;
            ++v33;
          }
          if ( v31 )
            v16 = v31;
        }
        goto LABEL_11;
      }
LABEL_47:
      ++*(_DWORD *)(v4 + 448);
      BUG();
    }
LABEL_18:
    sub_16228F0(v7, 2 * v6);
    v21 = *(_DWORD *)(v4 + 456);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v4 + 440);
      v24 = (v21 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v18 = *(_DWORD *)(v4 + 448) + 1;
      v16 = (__int64 *)(v23 + 16LL * v24);
      v25 = *v16;
      if ( v5 != *v16 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4 )
        {
          if ( !v27 && v25 == -8 )
            v27 = v16;
          v24 = v22 & (v26 + v24);
          v16 = (__int64 *)(v23 + 16LL * v24);
          v25 = *v16;
          if ( v5 == *v16 )
            goto LABEL_11;
          ++v26;
        }
        if ( v27 )
          v16 = v27;
      }
      goto LABEL_11;
    }
    goto LABEL_47;
  }
LABEL_3:
  v13 = v11[1];
  if ( v13 )
    return v13;
  v16 = v11;
LABEL_14:
  v19 = sub_16432C0(a1);
  v20 = sub_22077B0(32);
  v13 = v20;
  if ( v20 )
    sub_1623AE0(v20, v19, v5);
  v16[1] = v13;
  return v13;
}
