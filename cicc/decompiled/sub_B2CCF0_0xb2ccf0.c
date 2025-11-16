// Function: sub_B2CCF0
// Address: 0xb2ccf0
//
unsigned __int64 __fastcall sub_B2CCF0(__int64 a1, int a2, __int64 a3)
{
  __int64 *v4; // rax
  unsigned __int64 result; // rax

  v4 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7B440((__int64 *)(a1 + 120), v4, a2, a3);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
