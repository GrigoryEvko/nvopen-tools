// Function: sub_F91210
// Address: 0xf91210
//
__int64 __fastcall sub_F91210(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  __int64 v6; // rdx
  unsigned __int8 v7; // cl

  if ( *a1 == 85 && *(_BYTE *)a2 == 85 && ((*(_WORD *)(a2 + 2) & 3) == 2) != ((*((_WORD *)a1 + 1) & 3) == 2) )
    return 0;
  if ( !(unsigned __int8)sub_DFA890(a3) )
    return 0;
  v4 = sub_DFA890(a3);
  if ( !(_BYTE)v4 )
    return 0;
  if ( (unsigned __int8)(*a1 - 34) <= 0x33u )
  {
    v6 = 0x8000000000041LL;
    if ( _bittest64(&v6, (unsigned int)*a1 - 34) )
    {
      if ( (unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 32)
        || (unsigned __int8)sub_B49560((__int64)a1, 32)
        || (unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 6)
        || (unsigned __int8)sub_B49560((__int64)a1, 6) )
      {
        return 0;
      }
    }
  }
  v7 = *(_BYTE *)a2 - 34;
  if ( v7 <= 0x33u )
  {
    v4 = ((0x8000000000041uLL >> v7) & 1) == 0;
    if ( ((0x8000000000041uLL >> v7) & 1) != 0 )
    {
      if ( !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 32)
        && !(unsigned __int8)sub_B49560(a2, 32)
        && !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 6) )
      {
        return (unsigned int)sub_B49560(a2, 6) ^ 1;
      }
      return 0;
    }
  }
  return v4;
}
