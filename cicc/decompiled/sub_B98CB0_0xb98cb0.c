// Function: sub_B98CB0
// Address: 0xb98cb0
//
__int64 __fastcall sub_B98CB0(__int64 a1, unsigned __int8 **a2, _QWORD *a3)
{
  __int64 v3; // r13
  _QWORD *v4; // rbx
  __int64 v6; // rax

  v3 = ((__int64)a2 - a1) >> 3;
  v4 = a3;
  if ( a2 && (unsigned int)**a2 - 1 <= 1 && !a3 )
  {
    v6 = sub_ACADE0(*(__int64 ***)(*((_QWORD *)*a2 + 17) + 8LL));
    v4 = sub_B98A20(v6, (__int64)a2);
  }
  sub_B91340(a1, v3);
  *(_QWORD *)(a1 + 8 * v3) = v4;
  return sub_B96F50(a1, v3);
}
