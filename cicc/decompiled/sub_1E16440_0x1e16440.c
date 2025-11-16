// Function: sub_1E16440
// Address: 0x1e16440
//
__int64 __fastcall sub_1E16440(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  *(_WORD *)(a1 + 46) &= ~8u;
  *(_WORD *)(result + 46) &= ~4u;
  return result;
}
