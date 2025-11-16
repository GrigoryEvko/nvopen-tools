// Function: sub_B2D4A0
// Address: 0xb2d4a0
//
unsigned __int64 __fastcall sub_B2D4A0(__int64 a1, const void *a2, size_t a3)
{
  __int64 *v4; // rax
  unsigned __int64 result; // rax

  v4 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7A340((__int64 *)(a1 + 120), v4, -1, a2, a3);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
