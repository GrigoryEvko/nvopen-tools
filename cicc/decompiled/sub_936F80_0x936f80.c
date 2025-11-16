// Function: sub_936F80
// Address: 0x936f80
//
__int64 __fastcall sub_936F80(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // r13
  _QWORD *v4; // r15
  _QWORD *v5; // r14

  result = *(_QWORD *)(a2 + 72);
  v3 = *(_QWORD *)(result + 8);
  if ( (*(_BYTE *)(result + 24) & 2) != 0 )
    v3 = *(_QWORD *)result;
  if ( v3 )
  {
    v4 = (_QWORD *)sub_945CA0(a1, "constexpr_if.body", 0, 0);
    v5 = (_QWORD *)sub_945CA0(a1, "constexpr_if.end", 0, 0);
    sub_92FEA0((__int64)a1, v4, 0);
    sub_9363D0(a1, v3);
    sub_92FD90((__int64)a1, (__int64)v5);
    return sub_92FEA0((__int64)a1, v5, 1);
  }
  return result;
}
