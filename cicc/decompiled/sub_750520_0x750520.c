// Function: sub_750520
// Address: 0x750520
//
__int64 __fastcall sub_750520(__int64 a1)
{
  __int64 result; // rax

  result = **(_QWORD **)(a1 + 128);
  *(_QWORD *)(a1 + 128) = result;
  return result;
}
