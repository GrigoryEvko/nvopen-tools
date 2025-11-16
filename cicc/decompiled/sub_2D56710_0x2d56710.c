// Function: sub_2D56710
// Address: 0x2d56710
//
__int64 __fastcall sub_2D56710(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(result + 8) = *(_QWORD *)(a1 + 16);
  return result;
}
