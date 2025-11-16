// Function: sub_B2CD30
// Address: 0xb2cd30
//
unsigned __int64 __fastcall sub_B2CD30(__int64 a1, int a2)
{
  __int64 *v2; // rax
  unsigned __int64 result; // rax

  v2 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7A090((__int64 *)(a1 + 120), v2, -1, a2);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
