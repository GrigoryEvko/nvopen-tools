// Function: sub_16290E0
// Address: 0x16290e0
//
__int64 __fastcall sub_16290E0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  int v7; // eax
  __int64 v8; // rsi
  int v9; // ecx
  __int64 v10; // rdi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  unsigned int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // edi
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 result; // rax
  int v21; // eax
  int v22; // r9d
  int v23; // r10d
  __int64 *v24; // rdx
  int v25; // eax
  int v26; // ecx
  int v27; // eax
  int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // eax
  __int64 v31; // rdi
  int v32; // r10d
  __int64 *v33; // r9
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  __int64 *v37; // r8
  unsigned int v38; // r15d
  int v39; // r9d
  __int64 v40; // rsi
  __int64 *v41; // [rsp+8h] [rbp-38h]

  v3 = (__int64 *)sub_16498A0(a1);
  v4 = sub_1628D40(v3, a2);
  v5 = *v3;
  v6 = v4;
  v7 = *(_DWORD *)(v5 + 456);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 24);
    v9 = v7 - 1;
    v10 = *(_QWORD *)(v5 + 440);
    v11 = (v7 - 1) & (((unsigned int)*(_QWORD *)(a1 + 24) >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_3:
      *v12 = -8;
      --*(_DWORD *)(v5 + 448);
      ++*(_DWORD *)(v5 + 452);
    }
    else
    {
      v21 = 1;
      while ( v13 != -4 )
      {
        v22 = v21 + 1;
        v11 = v9 & (v21 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v8 == *v12 )
          goto LABEL_3;
        v21 = v22;
      }
    }
  }
  sub_161E810(a1);
  *(_QWORD *)(a1 + 24) = 0;
  v14 = *(_DWORD *)(v5 + 456);
  if ( !v14 )
  {
    ++*(_QWORD *)(v5 + 432);
    goto LABEL_23;
  }
  v15 = *(_QWORD *)(v5 + 440);
  v16 = (v14 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v17 = (__int64 *)(v15 + 16LL * v16);
  v18 = *v17;
  if ( v6 != *v17 )
  {
    v23 = 1;
    v24 = 0;
    while ( v18 != -4 )
    {
      if ( !v24 && v18 == -8 )
        v24 = v17;
      v16 = (v14 - 1) & (v23 + v16);
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v6 == *v17 )
        goto LABEL_6;
      ++v23;
    }
    if ( !v24 )
      v24 = v17;
    v25 = *(_DWORD *)(v5 + 448);
    ++*(_QWORD *)(v5 + 432);
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v14 )
    {
      if ( v14 - *(_DWORD *)(v5 + 452) - v26 > v14 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(v5 + 448) = v26;
        if ( *v24 != -4 )
          --*(_DWORD *)(v5 + 452);
        *v24 = v6;
        v24[1] = 0;
        goto LABEL_21;
      }
      sub_16228F0(v5 + 432, v14);
      v34 = *(_DWORD *)(v5 + 456);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(v5 + 440);
        v37 = 0;
        v38 = v35 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v39 = 1;
        v26 = *(_DWORD *)(v5 + 448) + 1;
        v24 = (__int64 *)(v36 + 16LL * v38);
        v40 = *v24;
        if ( v6 != *v24 )
        {
          while ( v40 != -4 )
          {
            if ( v40 == -8 && !v37 )
              v37 = v24;
            v38 = v35 & (v39 + v38);
            v24 = (__int64 *)(v36 + 16LL * v38);
            v40 = *v24;
            if ( v6 == *v24 )
              goto LABEL_18;
            ++v39;
          }
          if ( v37 )
            v24 = v37;
        }
        goto LABEL_18;
      }
LABEL_52:
      ++*(_DWORD *)(v5 + 448);
      BUG();
    }
LABEL_23:
    sub_16228F0(v5 + 432, 2 * v14);
    v27 = *(_DWORD *)(v5 + 456);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v5 + 440);
      v30 = (v27 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v26 = *(_DWORD *)(v5 + 448) + 1;
      v24 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v24;
      if ( v6 != *v24 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -4 )
        {
          if ( v31 == -8 && !v33 )
            v33 = v24;
          v30 = v28 & (v32 + v30);
          v24 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v24;
          if ( v6 == *v24 )
            goto LABEL_18;
          ++v32;
        }
        if ( v33 )
          v24 = v33;
      }
      goto LABEL_18;
    }
    goto LABEL_52;
  }
LABEL_6:
  v19 = v17[1];
  if ( v19 )
  {
    sub_164D160(a1, v19);
    sub_161E830(a1);
    return j_j___libc_free_0(a1, 32);
  }
  v24 = v17;
LABEL_21:
  *(_QWORD *)(a1 + 24) = v6;
  v41 = v24;
  result = sub_1623AC0(a1);
  v41[1] = a1;
  return result;
}
