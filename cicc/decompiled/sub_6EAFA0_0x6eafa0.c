// Function: sub_6EAFA0
// Address: 0x6eafa0
//
__int64 __fastcall sub_6EAFA0(unsigned __int8 a1)
{
  __int64 result; // rax

  result = sub_725A70(a1);
  if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) != 0 )
    *(_BYTE *)(result + 49) |= 4u;
  return result;
}
