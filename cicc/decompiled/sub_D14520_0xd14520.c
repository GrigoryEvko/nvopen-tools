// Function: sub_D14520
// Address: 0xd14520
//
__int64 __fastcall sub_D14520(unsigned __int8 *a1, __int64 a2)
{
  char v4; // cl
  __int64 v5; // rdi
  int v6; // esi
  unsigned int v7; // edx
  __int64 v8; // rax
  unsigned __int8 *v9; // r8
  __int64 result; // rax
  unsigned int v11; // esi
  int v12; // r9d
  __int64 v13; // r13
  unsigned int v14; // eax
  int v15; // edx
  unsigned int v16; // edi
  __int64 v17; // rsi
  int v18; // edx
  unsigned int v19; // eax
  unsigned __int8 *v20; // rcx
  __int64 v21; // rcx
  int v22; // edx
  unsigned int v23; // eax
  unsigned __int8 *v24; // rsi
  int v25; // r8d
  __int64 v26; // rdi
  int v27; // edx
  int v28; // edx
  int v29; // r8d

  if ( !a2 )
  {
    if ( (unsigned __int8)sub_CF70D0(a1) )
      return (unsigned int)sub_D13FA0((__int64)a1, 0, 0) ^ 1;
    return 0;
  }
  v4 = *(_BYTE *)(a2 + 8) & 1;
  if ( v4 )
  {
    v5 = a2 + 16;
    v6 = 7;
    v7 = ((unsigned __int8)((unsigned int)a1 >> 4) ^ (unsigned __int8)((unsigned int)a1 >> 9)) & 7;
    v8 = v5 + 16LL * (((unsigned __int8)((unsigned int)a1 >> 4) ^ (unsigned __int8)((unsigned int)a1 >> 9)) & 7);
    v9 = *(unsigned __int8 **)v8;
    if ( a1 == *(unsigned __int8 **)v8 )
      return *(unsigned __int8 *)(v8 + 8);
  }
  else
  {
    v11 = *(_DWORD *)(a2 + 24);
    if ( !v11 )
    {
      v14 = *(_DWORD *)(a2 + 8);
      ++*(_QWORD *)a2;
      v13 = 0;
      v15 = (v14 >> 1) + 1;
      goto LABEL_17;
    }
    v6 = v11 - 1;
    v5 = *(_QWORD *)(a2 + 16);
    v7 = v6 & (((unsigned int)a1 >> 4) ^ ((unsigned int)a1 >> 9));
    v8 = v5 + 16LL * v7;
    v9 = *(unsigned __int8 **)v8;
    if ( a1 == *(unsigned __int8 **)v8 )
      return *(unsigned __int8 *)(v8 + 8);
  }
  v12 = 1;
  v13 = 0;
  while ( v9 != (unsigned __int8 *)-4096LL )
  {
    if ( !v13 && v9 == (unsigned __int8 *)-8192LL )
      v13 = v8;
    v7 = v6 & (v12 + v7);
    v8 = v5 + 16LL * v7;
    v9 = *(unsigned __int8 **)v8;
    if ( a1 == *(unsigned __int8 **)v8 )
      return *(unsigned __int8 *)(v8 + 8);
    ++v12;
  }
  if ( !v13 )
    v13 = v8;
  v14 = *(_DWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v15 = (v14 >> 1) + 1;
  if ( v4 )
  {
    v16 = 24;
    v11 = 8;
    goto LABEL_18;
  }
  v11 = *(_DWORD *)(a2 + 24);
LABEL_17:
  v16 = 3 * v11;
LABEL_18:
  if ( 4 * v15 < v16 )
  {
    if ( v11 - *(_DWORD *)(a2 + 12) - v15 > v11 >> 3 )
      goto LABEL_20;
    sub_D140E0(a2, v11);
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v21 = a2 + 16;
      v22 = 7;
      goto LABEL_35;
    }
    v28 = *(_DWORD *)(a2 + 24);
    v21 = *(_QWORD *)(a2 + 16);
    if ( v28 )
    {
      v22 = v28 - 1;
LABEL_35:
      v23 = v22 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = v21 + 16LL * v23;
      v24 = *(unsigned __int8 **)v13;
      if ( a1 != *(unsigned __int8 **)v13 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != (unsigned __int8 *)-4096LL )
        {
          if ( !v26 && v24 == (unsigned __int8 *)-8192LL )
            v26 = v13;
          v23 = v22 & (v25 + v23);
          v13 = v21 + 16LL * v23;
          v24 = *(unsigned __int8 **)v13;
          if ( a1 == *(unsigned __int8 **)v13 )
            goto LABEL_32;
          ++v25;
        }
LABEL_38:
        if ( v26 )
          v13 = v26;
        goto LABEL_32;
      }
      goto LABEL_32;
    }
LABEL_57:
    *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
    BUG();
  }
  sub_D140E0(a2, 2 * v11);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v17 = a2 + 16;
    v18 = 7;
  }
  else
  {
    v27 = *(_DWORD *)(a2 + 24);
    v17 = *(_QWORD *)(a2 + 16);
    if ( !v27 )
      goto LABEL_57;
    v18 = v27 - 1;
  }
  v19 = v18 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v13 = v17 + 16LL * v19;
  v20 = *(unsigned __int8 **)v13;
  if ( a1 != *(unsigned __int8 **)v13 )
  {
    v29 = 1;
    v26 = 0;
    while ( v20 != (unsigned __int8 *)-4096LL )
    {
      if ( !v26 && v20 == (unsigned __int8 *)-8192LL )
        v26 = v13;
      v19 = v18 & (v29 + v19);
      v13 = v17 + 16LL * v19;
      v20 = *(unsigned __int8 **)v13;
      if ( a1 == *(unsigned __int8 **)v13 )
        goto LABEL_32;
      ++v29;
    }
    goto LABEL_38;
  }
LABEL_32:
  v14 = *(_DWORD *)(a2 + 8);
LABEL_20:
  *(_DWORD *)(a2 + 8) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *(_QWORD *)v13 != -4096 )
    --*(_DWORD *)(a2 + 12);
  *(_QWORD *)v13 = a1;
  *(_BYTE *)(v13 + 8) = 0;
  if ( !(unsigned __int8)sub_CF70D0(a1) )
    return 0;
  result = (unsigned int)sub_D13FA0((__int64)a1, 0, 0) ^ 1;
  *(_BYTE *)(v13 + 8) = result;
  return result;
}
