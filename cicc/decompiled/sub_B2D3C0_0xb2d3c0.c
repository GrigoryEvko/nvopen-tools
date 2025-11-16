// Function: sub_B2D3C0
// Address: 0xb2d3c0
//
unsigned __int64 __fastcall sub_B2D3C0(__int64 a1, int a2, int a3)
{
  __int64 *v4; // rax
  unsigned __int64 result; // rax

  v4 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7A090((__int64 *)(a1 + 120), v4, a2 + 1, a3);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
