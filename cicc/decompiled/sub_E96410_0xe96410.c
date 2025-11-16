// Function: sub_E96410
// Address: 0xe96410
//
__int64 __fastcall sub_E96410(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(result + 8 * a2) = a3;
  return result;
}
