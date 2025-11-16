// Function: sub_B49DB0
// Address: 0xb49db0
//
unsigned __int64 __fastcall sub_B49DB0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 *v4; // rax
  unsigned __int64 result; // rax

  v2 = (__int64 *)sub_BD5C60(a1, a2);
  v3 = sub_A77AB0(v2, a2);
  v4 = (__int64 *)sub_BD5C60(a1, (unsigned int)a2);
  result = sub_A7B440((__int64 *)(a1 + 72), v4, -1, v3);
  *(_QWORD *)(a1 + 72) = result;
  return result;
}
