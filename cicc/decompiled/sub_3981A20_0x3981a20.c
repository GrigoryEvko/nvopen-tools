// Function: sub_3981A20
// Address: 0x3981a20
//
__int64 __fastcall sub_3981A20(unsigned __int16 *a1, __int64 a2)
{
  __int64 result; // rax

  sub_16BD430(a2, *a1);
  result = sub_16BD430(a2, a1[1]);
  if ( a1[1] == 33 )
    return sub_16BD4D0(a2, *((_QWORD *)a1 + 1));
  return result;
}
