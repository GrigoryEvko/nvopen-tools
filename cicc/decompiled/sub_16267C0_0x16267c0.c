// Function: sub_16267C0
// Address: 0x16267c0
//
unsigned __int64 __fastcall sub_16267C0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r9
  __int64 v10; // rcx
  unsigned int v11; // r15d
  unsigned int v12; // edx
  unsigned int *v13; // rdi
  __int64 v14; // rax
  int v16; // r11d
  unsigned int *v17; // r10
  int v18; // eax
  int v19; // edx
  int v20; // eax
  int v21; // ecx
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rsi
  int v25; // r10d
  unsigned int *v26; // r9
  int v27; // eax
  int v28; // eax
  __int64 v29; // rsi
  int v30; // r9d
  unsigned int *v31; // r8
  unsigned int v32; // r15d
  __int64 v33; // rcx

  if ( (*(_DWORD *)(a1 + 32) & 0x100000) == 0 )
    *(_DWORD *)(a1 + 32) = (((*(_DWORD *)(a1 + 32) >> 15) & 0xFFFFFFDF | 0x20) << 15) | *(_DWORD *)(a1 + 32) & 0x7FFF;
  v6 = sub_16498A0(a1);
  v7 = *(_QWORD *)v6;
  v8 = *(_DWORD *)(*(_QWORD *)v6 + 2760LL);
  v9 = *(_QWORD *)v6 + 2736LL;
  if ( !v8 )
  {
    ++*(_QWORD *)(v7 + 2736);
    goto LABEL_16;
  }
  v10 = *(_QWORD *)(v7 + 2744);
  v11 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v12 = (v8 - 1) & v11;
  v13 = (unsigned int *)(v10 + 40LL * v12);
  v14 = *(_QWORD *)v13;
  if ( a1 == *(_QWORD *)v13 )
    return sub_1623C00(v13 + 2, a2, a3);
  v16 = 1;
  v17 = 0;
  while ( v14 != -8 )
  {
    if ( !v17 && v14 == -16 )
      v17 = v13;
    v12 = (v8 - 1) & (v16 + v12);
    v13 = (unsigned int *)(v10 + 40LL * v12);
    v14 = *(_QWORD *)v13;
    if ( a1 == *(_QWORD *)v13 )
      return sub_1623C00(v13 + 2, a2, a3);
    ++v16;
  }
  v18 = *(_DWORD *)(v7 + 2752);
  if ( v17 )
    v13 = v17;
  ++*(_QWORD *)(v7 + 2736);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v8 )
  {
LABEL_16:
    sub_16261B0(v9, 2 * v8);
    v20 = *(_DWORD *)(v7 + 2760);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v7 + 2744);
      v23 = (v20 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = (unsigned int *)(v22 + 40LL * v23);
      v24 = *(_QWORD *)v13;
      v19 = *(_DWORD *)(v7 + 2752) + 1;
      if ( a1 != *(_QWORD *)v13 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( !v26 && v24 == -16 )
            v26 = v13;
          v23 = v21 & (v25 + v23);
          v13 = (unsigned int *)(v22 + 40LL * v23);
          v24 = *(_QWORD *)v13;
          if ( a1 == *(_QWORD *)v13 )
            goto LABEL_12;
          ++v25;
        }
        if ( v26 )
          v13 = v26;
      }
      goto LABEL_12;
    }
    goto LABEL_44;
  }
  if ( v8 - *(_DWORD *)(v7 + 2756) - v19 <= v8 >> 3 )
  {
    sub_16261B0(v9, v8);
    v27 = *(_DWORD *)(v7 + 2760);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v7 + 2744);
      v30 = 1;
      v31 = 0;
      v32 = v28 & v11;
      v13 = (unsigned int *)(v29 + 40LL * v32);
      v33 = *(_QWORD *)v13;
      v19 = *(_DWORD *)(v7 + 2752) + 1;
      if ( a1 != *(_QWORD *)v13 )
      {
        while ( v33 != -8 )
        {
          if ( !v31 && v33 == -16 )
            v31 = v13;
          v32 = v28 & (v30 + v32);
          v13 = (unsigned int *)(v29 + 40LL * v32);
          v33 = *(_QWORD *)v13;
          if ( a1 == *(_QWORD *)v13 )
            goto LABEL_12;
          ++v30;
        }
        if ( v31 )
          v13 = v31;
      }
      goto LABEL_12;
    }
LABEL_44:
    ++*(_DWORD *)(v7 + 2752);
    BUG();
  }
LABEL_12:
  *(_DWORD *)(v7 + 2752) = v19;
  if ( *(_QWORD *)v13 != -8 )
    --*(_DWORD *)(v7 + 2756);
  *(_QWORD *)v13 = a1;
  *((_QWORD *)v13 + 1) = v13 + 6;
  *((_QWORD *)v13 + 2) = 0x100000000LL;
  return sub_1623C00(v13 + 2, a2, a3);
}
