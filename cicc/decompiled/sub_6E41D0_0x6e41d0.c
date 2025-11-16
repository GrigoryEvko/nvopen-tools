// Function: sub_6E41D0
// Address: 0x6e41d0
//
__int64 __fastcall sub_6E41D0(__int64 a1, __int64 a2, int a3, __int64 *a4, _QWORD *a5, __int64 a6)
{
  __int64 result; // rax

  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    result = sub_6E3FE0(a2, a3, a1);
    if ( result )
      return sub_6E3CB0(result, a4, a5, a6);
  }
  return result;
}
