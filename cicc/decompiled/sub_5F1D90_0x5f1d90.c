// Function: sub_5F1D90
// Address: 0x5f1d90
//
__int64 __fastcall sub_5F1D90(__int64 a1)
{
  __int64 result; // rax

  result = sub_725E60();
  *(_BYTE *)result |= 9u;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
