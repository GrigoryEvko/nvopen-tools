// Function: sub_760760
// Address: 0x760760
//
unsigned __int64 __fastcall sub_760760(__int64 a1, char a2, __int64 a3, int a4)
{
  unsigned __int64 result; // rax

  if ( a1 != a3 )
    result = sub_72A270(a1, a2);
  if ( (*(_BYTE *)(a3 + 88) & 8) != 0 )
  {
    *(_BYTE *)(a1 + 88) &= ~8u;
    result = sub_760370((_QWORD *)a1, a2);
    if ( a4 )
      return sub_75C030(a1);
  }
  return result;
}
