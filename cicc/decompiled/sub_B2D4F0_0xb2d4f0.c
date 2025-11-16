// Function: sub_B2D4F0
// Address: 0xb2d4f0
//
unsigned __int64 __fastcall sub_B2D4F0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  unsigned __int64 result; // rax

  v2 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7A440((__int64 *)(a1 + 120), v2, -1, a2);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
