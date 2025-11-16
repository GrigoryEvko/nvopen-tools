// Function: sub_13AD000
// Address: 0x13ad000
//
__int64 __fastcall sub_13AD000(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        unsigned int *a7,
        __int64 a8)
{
  int v8; // r14d
  unsigned int v11; // esi
  unsigned int i; // ecx
  __int64 v13; // r12
  unsigned int v14; // r15d
  unsigned int v15; // edx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi

  v8 = (int)a6;
  v11 = *(_DWORD *)(a1 + 32);
  if ( a2 <= v11 )
  {
    for ( i = a2; i <= v11; ++i )
    {
      if ( (*(_BYTE *)a6 & 1) != 0 )
      {
        if ( ((((*a6 >> 1) & ~(-1LL << (*a6 >> 58))) >> i) & 1) != 0 )
          goto LABEL_7;
      }
      else if ( ((*(_QWORD *)(*(_QWORD *)*a6 + 8LL * (i >> 6)) >> i) & 1) != 0 )
      {
LABEL_7:
        v13 = i;
        if ( *a7 < i )
        {
          *a7 = i;
          sub_13AC8A0(a1, a3, a4, a5, i);
          sub_13ACA70(a1, a3, a4, a5, v13);
          sub_13AC760(a1, a3, a4, a5, v13);
        }
        v14 = 0;
        if ( (unsigned __int8)sub_13ACF70(a1, 1, v13, a5, a8) )
          v14 = sub_13AD000(a1, (int)v13 + 1, a3, a4, a5, v8, (__int64)a7, a8);
        if ( (unsigned __int8)sub_13ACF70(a1, 2, v13, a5, a8) )
          v14 += sub_13AD000(a1, (int)v13 + 1, a3, a4, a5, v8, (__int64)a7, a8);
        if ( (unsigned __int8)sub_13ACF70(a1, 4, v13, a5, a8) )
          v14 += sub_13AD000(a1, (int)v13 + 1, a3, a4, a5, v8, (__int64)a7, a8);
        *(_BYTE *)(a5 + 144 * v13 + 136) = 7;
        return v14;
      }
    }
  }
  if ( v11 )
  {
    v15 = 1;
    do
    {
      v17 = *a6;
      if ( (*a6 & 1) != 0 )
        v16 = (((v17 >> 1) & ~(-1LL << (*a6 >> 58))) >> v15) & 1;
      else
        v16 = (*(_QWORD *)(*(_QWORD *)v17 + 8LL * (v15 >> 6)) >> v15) & 1LL;
      if ( (_BYTE)v16 )
        *(_BYTE *)(a5 + 144LL * v15 + 137) |= *(_BYTE *)(a5 + 144LL * v15 + 136);
      ++v15;
    }
    while ( *(_DWORD *)(a1 + 32) >= v15 );
  }
  return 1;
}
