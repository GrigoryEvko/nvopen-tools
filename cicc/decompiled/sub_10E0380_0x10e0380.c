// Function: sub_10E0380
// Address: 0x10e0380
//
unsigned __int64 __fastcall sub_10E0380(__int64 a1, int a2)
{
  __int64 *v2; // rax
  unsigned __int64 result; // rax

  v2 = (__int64 *)sub_BD5C60(a1);
  result = sub_A7A090((__int64 *)(a1 + 72), v2, 0, a2);
  *(_QWORD *)(a1 + 72) = result;
  return result;
}
