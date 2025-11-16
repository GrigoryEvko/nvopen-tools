// Function: sub_390D660
// Address: 0x390d660
//
unsigned __int64 __fastcall sub_390D660(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rdx
  __int64 v6; // r13
  unsigned int v7; // esi
  __int64 v8; // rdi
  __int64 v9; // r8
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  int v16; // r11d
  _QWORD *v17; // r10
  int v18; // ecx
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // edx
  __int64 v24; // rdi
  int v25; // r10d
  _QWORD *v26; // r9
  int v27; // eax
  int v28; // edx
  __int64 v29; // rdi
  _QWORD *v30; // r8
  unsigned int v31; // r14d
  int v32; // r9d
  __int64 v33; // rsi

  v4 = *(_QWORD *)(a2 + 24);
  if ( a2 == *(_QWORD *)(v4 + 104) || (v5 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    *(_QWORD *)(a2 + 40) = 0;
  }
  else
  {
    v6 = *(_QWORD *)(v5 + 40);
    *(_QWORD *)(a2 + 40) = sub_390B580(*(_QWORD *)a1, (_QWORD *)a1, v5) + v6;
    v4 = *(_QWORD *)(a2 + 24);
  }
  v7 = *(_DWORD *)(a1 + 176);
  v8 = a1 + 152;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_24;
  }
  v9 = *(_QWORD *)(a1 + 160);
  v10 = (v7 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = (_QWORD *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( v4 == *v11 )
    goto LABEL_6;
  v16 = 1;
  v17 = 0;
  while ( v12 != -8 )
  {
    if ( v12 == -16 && !v17 )
      v17 = v11;
    v10 = (v7 - 1) & (v16 + v10);
    v11 = (_QWORD *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == v4 )
      goto LABEL_6;
    ++v16;
  }
  v18 = *(_DWORD *)(a1 + 168);
  if ( v17 )
    v11 = v17;
  ++*(_QWORD *)(a1 + 152);
  v19 = v18 + 1;
  if ( 4 * v19 >= 3 * v7 )
  {
LABEL_24:
    sub_38CFAA0(v8, 2 * v7);
    v20 = *(_DWORD *)(a1 + 176);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 160);
      v23 = (v20 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v19 = *(_DWORD *)(a1 + 168) + 1;
      v11 = (_QWORD *)(v22 + 16LL * v23);
      v24 = *v11;
      if ( *v11 != v4 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( !v26 && v24 == -16 )
            v26 = v11;
          v23 = v21 & (v25 + v23);
          v11 = (_QWORD *)(v22 + 16LL * v23);
          v24 = *v11;
          if ( *v11 == v4 )
            goto LABEL_20;
          ++v25;
        }
        if ( v26 )
          v11 = v26;
      }
      goto LABEL_20;
    }
    goto LABEL_54;
  }
  if ( v7 - *(_DWORD *)(a1 + 172) - v19 <= v7 >> 3 )
  {
    sub_38CFAA0(v8, v7);
    v27 = *(_DWORD *)(a1 + 176);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 160);
      v30 = 0;
      v31 = (v27 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v32 = 1;
      v19 = *(_DWORD *)(a1 + 168) + 1;
      v11 = (_QWORD *)(v29 + 16LL * v31);
      v33 = *v11;
      if ( *v11 != v4 )
      {
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v30 )
            v30 = v11;
          v31 = v28 & (v32 + v31);
          v11 = (_QWORD *)(v29 + 16LL * v31);
          v33 = *v11;
          if ( v4 == *v11 )
            goto LABEL_20;
          ++v32;
        }
        if ( v30 )
          v11 = v30;
      }
      goto LABEL_20;
    }
LABEL_54:
    ++*(_DWORD *)(a1 + 168);
    BUG();
  }
LABEL_20:
  *(_DWORD *)(a1 + 168) = v19;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 172);
  *v11 = v4;
  v11[1] = 0;
LABEL_6:
  v11[1] = a2;
  result = *(unsigned int *)(*(_QWORD *)a1 + 480LL);
  if ( (_DWORD)result && *(_BYTE *)(a2 + 17) )
  {
    v14 = sub_390B580(*(_QWORD *)a1, (_QWORD *)a1, a2);
    v15 = *(_QWORD *)a1;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 484LL) & 1) == 0 && *(unsigned int *)(v15 + 480) < v14 )
      sub_16BD130("Fragment can't be larger than a bundle size", 1u);
    result = sub_38CF6F0(v15, a2, *(_QWORD *)(a2 + 40), v14);
    if ( result > 0xFF )
      sub_16BD130("Padding cannot exceed 255 bytes", 1u);
    *(_QWORD *)(a2 + 40) += result;
    *(_BYTE *)(a2 + 49) = result;
  }
  return result;
}
