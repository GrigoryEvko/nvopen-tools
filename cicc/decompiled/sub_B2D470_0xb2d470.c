// Function: sub_B2D470
// Address: 0xb2d470
//
unsigned __int64 __fastcall sub_B2D470(__int64 a1, int a2)
{
  __int64 *v2; // rax
  unsigned __int64 result; // rax

  v2 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7B980((__int64 *)(a1 + 120), v2, -1, a2);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
