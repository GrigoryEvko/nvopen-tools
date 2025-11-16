// Function: sub_13409B0
// Address: 0x13409b0
//
__int64 __fastcall sub_13409B0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  sub_133DFA0(a1);
  a1[2] = 0;
  result = sub_130AF40((__int64)(a1 + 3));
  if ( !(_BYTE)result )
    a1[17] = a2;
  return result;
}
