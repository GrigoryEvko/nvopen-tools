// Function: sub_23DCD00
// Address: 0x23dcd00
//
__int64 __fastcall sub_23DCD00(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)a1 = "___asan_gen_";
  *(_QWORD *)(a1 + 16) = a2;
  *(_WORD *)(a1 + 32) = 1283;
  return a1;
}
