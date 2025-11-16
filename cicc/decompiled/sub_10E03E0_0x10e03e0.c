// Function: sub_10E03E0
// Address: 0x10e03e0
//
unsigned __int64 __fastcall sub_10E03E0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  unsigned __int64 result; // rax

  v2 = (_QWORD *)sub_BD5C60(a1);
  result = sub_A7B4D0((__int64 *)(a1 + 72), v2, a2);
  *(_QWORD *)(a1 + 72) = result;
  return result;
}
