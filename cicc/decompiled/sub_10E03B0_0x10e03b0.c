// Function: sub_10E03B0
// Address: 0x10e03b0
//
unsigned __int64 __fastcall sub_10E03B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  unsigned __int64 result; // rax

  v2 = (__int64 *)sub_BD5C60(a1);
  result = sub_A7B440((__int64 *)(a1 + 72), v2, 0, a2);
  *(_QWORD *)(a1 + 72) = result;
  return result;
}
