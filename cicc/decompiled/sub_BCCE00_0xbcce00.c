// Function: sub_BCCE00
// Address: 0xbcce00
//
__int64 __fastcall sub_BCCE00(_QWORD *a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r10d
  unsigned int *v9; // r14
  __int64 v10; // r8
  unsigned int v11; // ecx
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 *v14; // r14
  int v15; // eax
  int v16; // edx
  int v17; // edx
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned int v22; // esi
  int v23; // r9d
  unsigned int *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  unsigned int *v29; // rdi
  unsigned int v30; // r15d
  unsigned int v31; // ecx

  if ( a2 == 32 )
    return sub_BCB2D0(a1);
  if ( a2 > 0x20 )
  {
    if ( a2 == 64 )
      return sub_BCB2E0(a1);
    if ( a2 == 128 )
      return sub_BCB2F0(a1);
  }
  else
  {
    switch ( a2 )
    {
      case 8u:
        return sub_BCB2B0(a1);
      case 0x10u:
        return sub_BCB2C0(a1);
      case 1u:
        return sub_BCB2A0(a1);
    }
  }
  v5 = *a1;
  v6 = *(_DWORD *)(*a1 + 2896LL);
  v7 = *a1 + 2872LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 2872);
    goto LABEL_35;
  }
  v8 = 1;
  v9 = 0;
  v10 = *(_QWORD *)(v5 + 2880);
  v11 = (v6 - 1) & (37 * a2);
  v12 = v10 + 16LL * v11;
  v13 = *(_DWORD *)v12;
  if ( *(_DWORD *)v12 == a2 )
  {
LABEL_16:
    v14 = (__int64 *)(v12 + 8);
    result = *(_QWORD *)(v12 + 8);
    if ( result )
      return result;
    goto LABEL_31;
  }
  while ( v13 != -1 )
  {
    if ( !v9 && v13 == -2 )
      v9 = (unsigned int *)v12;
    v11 = (v6 - 1) & (v8 + v11);
    v12 = v10 + 16LL * v11;
    v13 = *(_DWORD *)v12;
    if ( *(_DWORD *)v12 == a2 )
      goto LABEL_16;
    ++v8;
  }
  if ( !v9 )
    v9 = (unsigned int *)v12;
  v15 = *(_DWORD *)(v5 + 2888);
  ++*(_QWORD *)(v5 + 2872);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v6 )
  {
LABEL_35:
    sub_BCCC20(v7, 2 * v6);
    v18 = *(_DWORD *)(v5 + 2896);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v5 + 2880);
      v21 = (v18 - 1) & (37 * a2);
      v16 = *(_DWORD *)(v5 + 2888) + 1;
      v9 = (unsigned int *)(v20 + 16LL * v21);
      v22 = *v9;
      if ( *v9 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -1 )
        {
          if ( !v24 && v22 == -2 )
            v24 = v9;
          v21 = v19 & (v23 + v21);
          v9 = (unsigned int *)(v20 + 16LL * v21);
          v22 = *v9;
          if ( *v9 == a2 )
            goto LABEL_28;
          ++v23;
        }
        if ( v24 )
          v9 = v24;
      }
      goto LABEL_28;
    }
    goto LABEL_58;
  }
  if ( v6 - *(_DWORD *)(v5 + 2892) - v16 <= v6 >> 3 )
  {
    sub_BCCC20(v7, v6);
    v25 = *(_DWORD *)(v5 + 2896);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v5 + 2880);
      v28 = 1;
      v29 = 0;
      v30 = v26 & (37 * a2);
      v16 = *(_DWORD *)(v5 + 2888) + 1;
      v9 = (unsigned int *)(v27 + 16LL * v30);
      v31 = *v9;
      if ( *v9 != a2 )
      {
        while ( v31 != -1 )
        {
          if ( !v29 && v31 == -2 )
            v29 = v9;
          v30 = v26 & (v28 + v30);
          v9 = (unsigned int *)(v27 + 16LL * v30);
          v31 = *v9;
          if ( *v9 == a2 )
            goto LABEL_28;
          ++v28;
        }
        if ( v29 )
          v9 = v29;
      }
      goto LABEL_28;
    }
LABEL_58:
    ++*(_DWORD *)(v5 + 2888);
    BUG();
  }
LABEL_28:
  *(_DWORD *)(v5 + 2888) = v16;
  if ( *v9 != -1 )
    --*(_DWORD *)(v5 + 2892);
  *v9 = a2;
  v14 = (__int64 *)(v9 + 2);
  *v14 = 0;
  v5 = *a1;
LABEL_31:
  result = sub_A777F0(0x18u, (__int64 *)(v5 + 2640));
  if ( result )
  {
    *(_BYTE *)(result + 8) = 12;
    v17 = *(unsigned __int8 *)(result + 8);
    *(_QWORD *)result = a1;
    *(_DWORD *)(result + 12) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_DWORD *)(result + 8) = v17 | (a2 << 8);
  }
  *v14 = result;
  return result;
}
