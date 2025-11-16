// Function: sub_7AF1D0
// Address: 0x7af1d0
//
__int64 __fastcall sub_7AF1D0(unsigned __int64 a1)
{
  __int64 result; // rax

  for ( result = qword_4F08580[(a1 >> 3) - 7993 * (a1 / 0xF9C8)];
        *(_QWORD *)(result + 16) != a1;
        result = *(_QWORD *)(result + 8) )
  {
    ;
  }
  return result;
}
