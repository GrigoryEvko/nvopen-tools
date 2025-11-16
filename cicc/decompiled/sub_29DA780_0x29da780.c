// Function: sub_29DA780
// Address: 0x29da780
//
__int64 __fastcall sub_29DA780(__int64 *a1, unsigned __int8 *a2, unsigned __int8 *a3, _BYTE *a4)
{
  unsigned int v7; // r12d
  __int64 v9; // rbx
  unsigned __int8 *v10; // rdx
  __int64 v11; // rdx
  unsigned __int8 *v12; // rcx
  unsigned int v13; // eax
  char v14; // al
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rbx
  __int64 v18; // r13
  __int64 i; // r14
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rax
  __int16 v24; // si
  int v25; // edx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rsi
  __int64 v28; // r12
  __int64 v29; // rsi
  unsigned __int64 v30; // rbx
  __int64 j; // rcx
  unsigned __int64 v32; // rbx
  __int64 k; // rcx
  __int64 v34; // rbx
  __int64 v35; // rcx
  __int64 v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+18h] [rbp-38h]
  __int64 v41; // [rsp+18h] [rbp-38h]
  __int64 v42; // [rsp+18h] [rbp-38h]
  __int64 v43; // [rsp+18h] [rbp-38h]
  int v44; // [rsp+18h] [rbp-38h]

  *a4 = 1;
  v7 = sub_29DA390((__int64)a1, (__int64)a2, (__int64)a3);
  if ( v7 )
    return v7;
  v7 = sub_29D7CF0((__int64)a1, (unsigned int)*a2 - 29, (unsigned int)*a3 - 29);
  if ( v7 )
    return v7;
  if ( *a2 == 63 )
  {
    *a4 = 0;
    v7 = sub_29DA390(
           (__int64)a1,
           *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)],
           *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)]);
    if ( !v7 )
      return sub_29DA520(a1, (__int64)a2, (__int64)a3);
    return v7;
  }
  v7 = sub_29D7CF0((__int64)a1, *((_DWORD *)a2 + 1) & 0x7FFFFFF, *((_DWORD *)a3 + 1) & 0x7FFFFFF);
  if ( v7 )
    return v7;
  v7 = sub_29D81B0(a1, *((_QWORD *)a2 + 1), *((_QWORD *)a3 + 1));
  if ( v7 )
    return v7;
  v7 = sub_29D7CF0((__int64)a1, a2[1] >> 1, a3[1] >> 1);
  if ( v7 )
    return v7;
  v40 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v9 = 0;
  if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) != 0 )
  {
    do
    {
      v10 = (a3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a3 - 1) : &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
      v11 = *(_QWORD *)(*(_QWORD *)&v10[v9] + 8LL);
      v12 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v13 = sub_29D81B0(a1, *(_QWORD *)(*(_QWORD *)&v12[v9] + 8LL), v11);
      if ( v13 )
        return v13;
      v9 += 32;
    }
    while ( v40 != v9 );
  }
  v14 = *a2;
  if ( *a2 != 60 )
  {
    if ( v14 == 61 )
    {
      v7 = sub_29D7CF0((__int64)a1, *((_WORD *)a2 + 1) & 1, *((_WORD *)a3 + 1) & 1);
      if ( v7 )
        return v7;
      _BitScanReverse64(&v20, 1LL << (*((_WORD *)a3 + 1) >> 1));
      _BitScanReverse64(&v21, 1LL << (*((_WORD *)a2 + 1) >> 1));
      v7 = sub_29D7D10((__int64)a1, 63 - ((unsigned __int8)v21 ^ 0x3Fu), 63 - ((unsigned __int8)v20 ^ 0x3Fu));
      if ( v7 )
        return v7;
      v7 = sub_29D7D40((__int64)a1, (*((_WORD *)a2 + 1) >> 7) & 7, (*((_WORD *)a3 + 1) >> 7) & 7);
      if ( v7 )
        return v7;
      v7 = sub_29D7CF0((__int64)a1, a2[72], a3[72]);
      if ( v7 )
        return v7;
      return sub_29DA230(a1, (__int64)a2, (__int64)a3);
    }
    if ( v14 == 62 )
    {
      v7 = sub_29D7CF0((__int64)a1, *((_WORD *)a2 + 1) & 1, *((_WORD *)a3 + 1) & 1);
      if ( v7 )
        return v7;
      _BitScanReverse64(&v22, 1LL << (*((_WORD *)a3 + 1) >> 1));
      _BitScanReverse64(&v23, 1LL << (*((_WORD *)a2 + 1) >> 1));
      v7 = sub_29D7D10((__int64)a1, 63 - ((unsigned __int8)v23 ^ 0x3Fu), 63 - ((unsigned __int8)v22 ^ 0x3Fu));
      if ( v7 )
        return v7;
      v24 = *((_WORD *)a2 + 1) >> 7;
      v25 = (*((_WORD *)a3 + 1) >> 7) & 7;
      goto LABEL_41;
    }
    if ( (unsigned __int8)(v14 - 82) <= 1u )
    {
      v26 = *((_WORD *)a3 + 1) & 0x3F;
      v27 = *((_WORD *)a2 + 1) & 0x3F;
      return sub_29D7CF0((__int64)a1, v27, v26);
    }
    if ( (unsigned __int8)(v14 - 34) <= 0x33u )
    {
      if ( ((0x8000000000041uLL >> (v14 - 34)) & 1) != 0 )
      {
        v7 = sub_29D7CF0((__int64)a1, (*((_WORD *)a2 + 1) >> 2) & 0x3FF, (*((_WORD *)a3 + 1) >> 2) & 0x3FF);
        if ( v7 )
          return v7;
        v7 = sub_29D84C0(a1, *((_QWORD *)a2 + 9), *((_QWORD *)a3 + 9));
        if ( v7 )
          return v7;
        v7 = sub_29D7FD0((__int64)a1, (__int64)a2, (__int64)a3);
        if ( v7 )
          return v7;
        if ( *a2 == 85 )
        {
          v7 = sub_29D7CF0((__int64)a1, *((_WORD *)a2 + 1) & 3, *((_WORD *)a3 + 1) & 3);
          if ( v7 )
            return v7;
        }
        if ( (a3[7] & 0x20) != 0 )
          v28 = sub_B91C10((__int64)a3, 4);
        else
          v28 = 0;
        if ( (a2[7] & 0x20) != 0 )
          v29 = sub_B91C10((__int64)a2, 4);
        else
          v29 = 0;
        return sub_29DA080(a1, v29, v28);
      }
    }
    else
    {
      if ( v14 == 94 )
      {
        v17 = *((unsigned int *)a2 + 20);
        v18 = *((_QWORD *)a3 + 9);
        v41 = *((_QWORD *)a2 + 9);
        v7 = sub_29D7CF0((__int64)a1, v17, *((unsigned int *)a3 + 20));
        if ( !v7 )
        {
          for ( i = 0; v17 != i; ++i )
          {
            v13 = sub_29D7CF0((__int64)a1, *(unsigned int *)(v41 + 4 * i), *(unsigned int *)(v18 + 4 * i));
            if ( v13 )
              return v13;
          }
        }
        return v7;
      }
      if ( v14 == 93 )
      {
        v30 = *((unsigned int *)a2 + 20);
        v42 = *((_QWORD *)a2 + 9);
        v38 = *((_QWORD *)a3 + 9);
        v13 = sub_29D7CF0((__int64)a1, v30, *((unsigned int *)a3 + 20));
        if ( v13 )
          return v13;
        for ( j = 0; v30 != j; j = v36 + 1 )
        {
          v36 = j;
          v13 = sub_29D7CF0((__int64)a1, *(unsigned int *)(v42 + 4 * j), *(unsigned int *)(v38 + 4 * j));
          if ( v13 )
            return v13;
        }
        v14 = *a2;
      }
    }
    switch ( v14 )
    {
      case '@':
        v24 = *((_WORD *)a2 + 1);
        v25 = *((_WORD *)a3 + 1) & 7;
LABEL_41:
        v7 = sub_29D7D40((__int64)a1, v24 & 7, v25);
        if ( v7 )
          return v7;
        v26 = a3[72];
        v27 = a2[72];
        return sub_29D7CF0((__int64)a1, v27, v26);
      case 'A':
        v7 = sub_29D7CF0((__int64)a1, *((_WORD *)a2 + 1) & 1, *((_WORD *)a3 + 1) & 1);
        if ( v7 )
          return v7;
        v7 = sub_29D7CF0((__int64)a1, (*((_WORD *)a2 + 1) >> 1) & 1, (*((_WORD *)a3 + 1) >> 1) & 1);
        if ( v7 )
          return v7;
        v7 = sub_29D7D40((__int64)a1, (*((_WORD *)a2 + 1) >> 2) & 7, (*((_WORD *)a3 + 1) >> 2) & 7);
        if ( v7 )
          return v7;
        v24 = *((_WORD *)a2 + 1) >> 5;
        v25 = (*((_WORD *)a3 + 1) >> 5) & 7;
        goto LABEL_41;
      case 'B':
        v7 = sub_29D7CF0((__int64)a1, (*((_WORD *)a2 + 1) >> 4) & 0x1F, (*((_WORD *)a3 + 1) >> 4) & 0x1F);
        if ( v7 )
          return v7;
        v7 = sub_29D7CF0((__int64)a1, *((_WORD *)a2 + 1) & 1, *((_WORD *)a3 + 1) & 1);
        if ( v7 )
          return v7;
        v24 = *((_WORD *)a2 + 1) >> 1;
        v25 = (*((_WORD *)a3 + 1) >> 1) & 7;
        goto LABEL_41;
    }
    if ( v14 != 92 )
    {
LABEL_83:
      if ( *a2 == 84 )
      {
        v34 = 0;
        v44 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
        while ( v44 != (_DWORD)v34 )
        {
          v35 = 8 * v34++;
          v13 = sub_29DA390(
                  (__int64)a1,
                  *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + v35),
                  *(_QWORD *)(*((_QWORD *)a3 - 1) + 32LL * *((unsigned int *)a3 + 18) + v35));
          if ( v13 )
            return v13;
        }
      }
      return v7;
    }
    v32 = *((unsigned int *)a2 + 20);
    v43 = *((_QWORD *)a2 + 9);
    v39 = *((_QWORD *)a3 + 9);
    v13 = sub_29D7CF0((__int64)a1, v32, *((unsigned int *)a3 + 20));
    if ( !v13 )
    {
      for ( k = 0; v32 != k; k = v37 + 1 )
      {
        v37 = k;
        v13 = sub_29D7CF0((__int64)a1, *(int *)(v43 + 4 * k), *(int *)(v39 + 4 * k));
        if ( v13 )
          return v13;
      }
      goto LABEL_83;
    }
    return v13;
  }
  v7 = sub_29D81B0(a1, *((_QWORD *)a2 + 9), *((_QWORD *)a3 + 9));
  if ( v7 )
    return v7;
  _BitScanReverse64(&v15, 1LL << *((_WORD *)a3 + 1));
  _BitScanReverse64(&v16, 1LL << *((_WORD *)a2 + 1));
  return sub_29D7D10((__int64)a1, 63 - ((unsigned __int8)v16 ^ 0x3Fu), 63 - ((unsigned __int8)v15 ^ 0x3Fu));
}
