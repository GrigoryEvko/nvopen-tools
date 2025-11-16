// Function: sub_750500
// Address: 0x750500
//
__int64 __fastcall sub_750500(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a2 + 128);
  *a1 = result;
  *(_QWORD *)(a2 + 128) = a1;
  return result;
}
