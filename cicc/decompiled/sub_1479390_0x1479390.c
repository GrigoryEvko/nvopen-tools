// Function: sub_1479390
// Address: 0x1479390
//
__int64 __fastcall sub_1479390(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // esi
  __int64 v7; // rdx

  v2 = (*(unsigned __int16 *)(a1 + 26) >> 1) & 2;
  if ( (*(_WORD *)(a1 + 26) & 5 | 2) != (*(_WORD *)(a1 + 26) & 7) )
    return v2;
  v4 = sub_13A5BC0((_QWORD *)a1, a2);
  if ( *(_WORD *)(v4 + 24) )
    return v2;
  v5 = *(_QWORD *)(v4 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  v7 = *(_QWORD *)(v5 + 24);
  if ( v6 > 0x40 )
    v7 = *(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6));
  if ( (v7 & (1LL << ((unsigned __int8)v6 - 1))) == 0 )
    v2 |= 1u;
  return v2;
}
