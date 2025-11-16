// Function: sub_134B160
// Address: 0x134b160
//
int __fastcall sub_134B160(__int64 a1, __int64 a2, __int64 a3)
{
  int result; // eax

  sub_134AB20(a1, a3, a2 + 68120, 1);
  sub_134AB20(a1, a3, a2 + 80, 2);
  sub_134AB20(a1, a3, a2 + 19520, 3);
  sub_134AB20(a1, a3, a2 + 38960, 4);
  sub_134AB20(a1, a3, a2 + 58672, 5);
  result = sub_134AB20(a1, a3, a2 + 60456, 6);
  if ( *(_BYTE *)(a2 + 17) )
  {
    sub_134AB20(a1, a3, a2 + 62448, 9);
    sub_134AB20(a1, a3, a2 + 62560, 10);
    return sub_130EE60(a1, a2 + 62264, a3 + 704);
  }
  return result;
}
