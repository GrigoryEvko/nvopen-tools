// Function: sub_7E2230
// Address: 0x7e2230
//
_BYTE *__fastcall sub_7E2230(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  _BYTE *result; // rax

  sub_7313A0((__int64)a1, a2, a3, a4, a5, a6);
  v6 = sub_8D67C0(*a1);
  result = sub_73DBF0(0x15u, v6, (__int64)a1);
  result[27] |= 2u;
  return result;
}
