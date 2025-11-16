// Function: sub_B2D9A0
// Address: 0xb2d9a0
//
unsigned __int64 __fastcall sub_B2D9A0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  unsigned __int64 result; // rax

  v2 = (_QWORD *)sub_B2BE50(a1);
  result = sub_A7B4D0((__int64 *)(a1 + 120), v2, a2);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
