// Function: sub_2EECD10
// Address: 0x2eecd10
//
__int64 __fastcall sub_2EECD10(unsigned int *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  int v8; // eax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rdi
  int v14; // r11d
  __int64 *v15; // rdx
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // r9
  int v20; // eax
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // eax
  int v30; // eax
  __int64 v31; // rdi
  int v32; // r9d
  unsigned int v33; // r14d
  __int64 *v34; // r8
  __int64 v35; // rsi

  v6 = *(_QWORD *)a1;
  v8 = *(unsigned __int16 *)(*(_QWORD *)a1 + 68LL);
  if ( (_WORD)v8 )
  {
    v9 = (unsigned int)(v8 - 9);
    if ( (unsigned __int16)v9 > 0x3Bu || (v10 = 0x800000000000C09LL, !_bittest64(&v10, v9)) )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v6 + 16) + 24LL) & 0x10) == 0 )
      {
        v11 = sub_2FF8170(a5, v6, a1[2], a2, a1[3]);
        v6 = *(_QWORD *)a1;
        a3 += v11;
      }
    }
  }
  v12 = *(_DWORD *)(a4 + 24);
  if ( !v12 )
  {
    ++*(_QWORD *)a4;
    goto LABEL_25;
  }
  v13 = *(_QWORD *)(a4 + 8);
  v14 = 1;
  v15 = 0;
  v16 = (v12 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v17 = (__int64 *)(v13 + 16LL * v16);
  v18 = *v17;
  if ( v6 == *v17 )
  {
LABEL_8:
    if ( a3 > *((_DWORD *)v17 + 2) )
      *((_DWORD *)v17 + 2) = a3;
    return 0;
  }
  while ( v18 != -4096 )
  {
    if ( !v15 && v18 == -8192 )
      v15 = v17;
    v16 = (v12 - 1) & (v14 + v16);
    v17 = (__int64 *)(v13 + 16LL * v16);
    v18 = *v17;
    if ( *v17 == v6 )
      goto LABEL_8;
    ++v14;
  }
  if ( !v15 )
    v15 = v17;
  v20 = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v12 )
  {
LABEL_25:
    sub_2EECB30(a4, 2 * v12);
    v22 = *(_DWORD *)(a4 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a4 + 8);
      v25 = (v22 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v21 = *(_DWORD *)(a4 + 16) + 1;
      v15 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v15;
      if ( *v15 != v6 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( !v28 && v26 == -8192 )
            v28 = v15;
          v25 = v23 & (v27 + v25);
          v15 = (__int64 *)(v24 + 16LL * v25);
          v26 = *v15;
          if ( *v15 == v6 )
            goto LABEL_21;
          ++v27;
        }
        if ( v28 )
          v15 = v28;
      }
      goto LABEL_21;
    }
    goto LABEL_48;
  }
  if ( v12 - *(_DWORD *)(a4 + 20) - v21 <= v12 >> 3 )
  {
    sub_2EECB30(a4, v12);
    v29 = *(_DWORD *)(a4 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a4 + 8);
      v32 = 1;
      v33 = v30 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v34 = 0;
      v21 = *(_DWORD *)(a4 + 16) + 1;
      v15 = (__int64 *)(v31 + 16LL * v33);
      v35 = *v15;
      if ( *v15 != v6 )
      {
        while ( v35 != -4096 )
        {
          if ( !v34 && v35 == -8192 )
            v34 = v15;
          v33 = v30 & (v32 + v33);
          v15 = (__int64 *)(v31 + 16LL * v33);
          v35 = *v15;
          if ( v6 == *v15 )
            goto LABEL_21;
          ++v32;
        }
        if ( v34 )
          v15 = v34;
      }
      goto LABEL_21;
    }
LABEL_48:
    ++*(_DWORD *)(a4 + 16);
    BUG();
  }
LABEL_21:
  *(_DWORD *)(a4 + 16) = v21;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a4 + 20);
  *v15 = v6;
  *((_DWORD *)v15 + 2) = a3;
  return 1;
}
