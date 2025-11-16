// Function: sub_38D7300
// Address: 0x38d7300
//
__int64 __fastcall sub_38D7300(__int64 a1, __int64 a2, unsigned int a3)
{
  _WORD *v5; // rsi

  v5 = (_WORD *)(*(_QWORD *)(a1 + 40) + 14LL * a3);
  if ( (*v5 & 0x3FFF) == 0x3FFF )
    return 0;
  else
    return sub_38D72B0(a2, (__int64)v5);
}
