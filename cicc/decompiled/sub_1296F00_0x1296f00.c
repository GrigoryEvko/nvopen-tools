// Function: sub_1296F00
// Address: 0x1296f00
//
__int64 __fastcall sub_1296F00(__int64 *a1, __int64 a2)
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
    v4 = (_QWORD *)sub_12A4D50(a1, "constexpr_if.body", 0, 0);
    v5 = (_QWORD *)sub_12A4D50(a1, "constexpr_if.end", 0, 0);
    sub_1290AF0(a1, v4, 0);
    sub_1296350(a1, v3);
    sub_12909B0(a1, (__int64)v5);
    return sub_1290AF0(a1, v5, 1);
  }
  return result;
}
