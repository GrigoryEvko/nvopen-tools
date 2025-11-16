// Function: sub_1383800
// Address: 0x1383800
//
__int64 __fastcall sub_1383800(__int64 a1)
{
  __int64 result; // rax

  sub_13836B0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24));
  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    if ( result != -8 && result != -16 )
      result = sub_1649B30(a1 + 8);
    *(_QWORD *)(a1 + 24) = 0;
  }
  return result;
}
