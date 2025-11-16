// Function: sub_6E2220
// Address: 0x6e2220
//
__int64 __fastcall sub_6E2220(__int64 a1)
{
  __int64 result; // rax

  if ( *(_QWORD *)(a1 + 328) )
  {
    result = qword_4D03C50;
    *(_QWORD *)(qword_4D03C50 + 136LL) = a1 + 328;
  }
  return result;
}
