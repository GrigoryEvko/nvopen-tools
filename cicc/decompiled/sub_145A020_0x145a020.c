// Function: sub_145A020
// Address: 0x145a020
//
__int64 __fastcall sub_145A020(__int64 a1)
{
  __int64 result; // rax

  sub_1459590(*(_QWORD *)(a1 + 64), a1 + 32);
  sub_16BDCA0(*(_QWORD *)(a1 + 64) + 816LL, a1 + 32);
  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    if ( result != -8 && result != -16 )
      result = sub_1649B30(a1 + 8);
    *(_QWORD *)(a1 + 24) = 0;
  }
  return result;
}
