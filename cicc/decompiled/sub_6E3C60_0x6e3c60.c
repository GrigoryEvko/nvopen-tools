// Function: sub_6E3C60
// Address: 0x6e3c60
//
__int64 __fastcall sub_6E3C60(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 result; // rax

  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    if ( *(_BYTE *)(a1 + 24) )
    {
      sub_6E3AC0(a1, a2, 0, a3);
      result = *(_QWORD *)(a1 + 80);
      *(_QWORD *)(result + 376) = a4;
    }
  }
  return result;
}
