// Function: sub_161F4C0
// Address: 0x161f4c0
//
_QWORD *__fastcall sub_161F4C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *result; // rax

  v3 = a1 + 16;
  *(_QWORD *)(v3 - 16) = 0;
  *(_QWORD *)(v3 - 8) = 0;
  sub_16E2FC0(v3, a2);
  *(_QWORD *)(a1 + 48) = 0;
  result = (_QWORD *)sub_22077B0(48);
  if ( result )
  {
    *result = result + 2;
    result[1] = 0x400000000LL;
  }
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
