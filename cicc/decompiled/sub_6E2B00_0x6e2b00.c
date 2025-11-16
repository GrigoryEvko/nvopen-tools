// Function: sub_6E2B00
// Address: 0x6e2b00
//
__int64 __fastcall sub_6E2B00(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(qword_4D03C50 + 48LL);
  if ( result )
  {
    if ( *(_QWORD *)(result + 24) )
      return sub_6E2A90(a1, a2);
  }
  return result;
}
