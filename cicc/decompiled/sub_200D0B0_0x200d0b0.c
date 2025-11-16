// Function: sub_200D0B0
// Address: 0x200d0b0
//
__int64 __fastcall sub_200D0B0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 664) = *(_QWORD *)(a1 + 8);
  return result;
}
