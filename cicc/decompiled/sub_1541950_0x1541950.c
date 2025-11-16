// Function: sub_1541950
// Address: 0x1541950
//
void __fastcall sub_1541950(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __int64 a6, __int64 a7)
{
  unsigned int v11; // esi
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // r10d
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rdi
  int v23; // eax
  int v24; // esi
  __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 v27; // r8
  int v28; // r10d
  int v29; // eax
  int v30; // edx
  __int64 v31; // rdi
  __int64 v32; // r8
  unsigned int v33; // r12d
  __int64 v34; // rsi

  v11 = *(_DWORD *)(a3 + 24);
  if ( !v11 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_28;
  }
  v12 = *(_QWORD *)(a3 + 8);
  v13 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v14 = v12 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( a1 != *(_QWORD *)v14 )
  {
    v16 = 1;
    a7 = 0;
    while ( v15 != -8 )
    {
      if ( v15 == -16 && !a7 )
        a7 = v14;
      v13 = (v11 - 1) & (v16 + v13);
      v14 = v12 + 16LL * v13;
      v15 = *(_QWORD *)v14;
      if ( a1 == *(_QWORD *)v14 )
        goto LABEL_3;
      ++v16;
    }
    v17 = *(_DWORD *)(a3 + 16);
    if ( a7 )
      v14 = a7;
    ++*(_QWORD *)a3;
    v18 = v17 + 1;
    if ( 4 * v18 < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(a3 + 20) - v18 > v11 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a3 + 16) = v18;
        if ( *(_QWORD *)v14 != -8 )
          --*(_DWORD *)(a3 + 20);
        *(_QWORD *)v14 = a1;
        *(_DWORD *)(v14 + 8) = 0;
        *(_BYTE *)(v14 + 12) = 0;
        goto LABEL_14;
      }
      sub_1541430(a3, v11);
      v29 = *(_DWORD *)(a3 + 24);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a3 + 8);
        v32 = 0;
        v33 = (v29 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        a7 = 1;
        v18 = *(_DWORD *)(a3 + 16) + 1;
        v14 = v31 + 16LL * v33;
        v34 = *(_QWORD *)v14;
        if ( a1 != *(_QWORD *)v14 )
        {
          while ( v34 != -8 )
          {
            if ( v34 == -16 && !v32 )
              v32 = v14;
            v33 = v30 & (a7 + v33);
            v14 = v31 + 16LL * v33;
            v34 = *(_QWORD *)v14;
            if ( a1 == *(_QWORD *)v14 )
              goto LABEL_11;
            a7 = (unsigned int)(a7 + 1);
          }
          if ( v32 )
            v14 = v32;
        }
        goto LABEL_11;
      }
LABEL_56:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_28:
    sub_1541430(a3, 2 * v11);
    v23 = *(_DWORD *)(a3 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a3 + 8);
      v26 = (v23 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v18 = *(_DWORD *)(a3 + 16) + 1;
      v14 = v25 + 16LL * v26;
      v27 = *(_QWORD *)v14;
      if ( a1 != *(_QWORD *)v14 )
      {
        v28 = 1;
        a7 = 0;
        while ( v27 != -8 )
        {
          if ( !a7 && v27 == -16 )
            a7 = v14;
          v26 = v24 & (v28 + v26);
          v14 = v25 + 16LL * v26;
          v27 = *(_QWORD *)v14;
          if ( a1 == *(_QWORD *)v14 )
            goto LABEL_11;
          ++v28;
        }
        if ( a7 )
          v14 = a7;
      }
      goto LABEL_11;
    }
    goto LABEL_56;
  }
LABEL_3:
  if ( *(_BYTE *)(v14 + 12) )
    return;
LABEL_14:
  *(_BYTE *)(v14 + 12) = 1;
  v19 = *(_QWORD *)(a1 + 8);
  if ( v19 && *(_QWORD *)(v19 + 8) )
  {
    sub_153F240(a1, a2, *(_DWORD *)(v14 + 8), a3, a4, a7, a5);
    if ( *(_BYTE *)(a1 + 16) > 0x10u )
      return;
  }
  else if ( *(_BYTE *)(a1 + 16) > 0x10u )
  {
    return;
  }
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
  {
    v20 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v21 = *(_QWORD *)(a1 - 8);
      a1 = v21 + v20;
    }
    else
    {
      v21 = a1 - v20;
    }
    v22 = *(_QWORD *)v21;
    if ( *(_BYTE *)(*(_QWORD *)v21 + 16LL) <= 0x10u )
      goto LABEL_22;
    while ( 1 )
    {
      v21 += 24;
      if ( a1 == v21 )
        break;
      v22 = *(_QWORD *)v21;
      if ( *(_BYTE *)(*(_QWORD *)v21 + 16LL) <= 0x10u )
LABEL_22:
        sub_1541950(v22, a2, a3, a4);
    }
  }
}
