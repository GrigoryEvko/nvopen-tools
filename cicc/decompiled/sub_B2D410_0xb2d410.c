// Function: sub_B2D410
// Address: 0xb2d410
//
unsigned __int64 __fastcall sub_B2D410(__int64 a1, int a2, __int64 a3)
{
  __int64 *v4; // rax
  unsigned __int64 result; // rax
  _DWORD v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  v6[0] = a2;
  v4 = (__int64 *)sub_B2BE50(a1);
  result = sub_A7B660((__int64 *)(a1 + 120), v4, v6, 1, a3);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
