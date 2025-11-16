// Function: sub_1D46090
// Address: 0x1d46090
//
__int64 __fastcall sub_1D46090(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 664) = *(_QWORD *)(a1 + 8);
  return result;
}
