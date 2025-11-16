// Function: sub_DC1810
// Address: 0xdc1810
//
__int64 __fastcall sub_DC1810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // esi
  __int64 v10; // rdx

  v5 = (*(unsigned __int16 *)(a1 + 28) >> 1) & 2;
  if ( (*(_WORD *)(a1 + 28) & 5 | 2) != (*(_WORD *)(a1 + 28) & 7) )
    return v5;
  v7 = sub_D33D80((_QWORD *)a1, a2, *(_WORD *)(a1 + 28) & 7 | 2u, a4, a5);
  if ( *(_WORD *)(v7 + 24) )
    return v5;
  v8 = *(_QWORD *)(v7 + 32);
  v9 = *(_DWORD *)(v8 + 32);
  v10 = *(_QWORD *)(v8 + 24);
  if ( v9 > 0x40 )
    v10 = *(_QWORD *)(v10 + 8LL * ((v9 - 1) >> 6));
  if ( (v10 & (1LL << ((unsigned __int8)v9 - 1))) == 0 )
    v5 |= 1u;
  return v5;
}
