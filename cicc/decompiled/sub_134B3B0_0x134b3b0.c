// Function: sub_134B3B0
// Address: 0x134b3b0
//
__int64 __fastcall sub_134B3B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  __int64 result; // rax

  v3 = (_QWORD *)(a2 + 24LL * *(unsigned __int8 *)(a3 + 16));
  ++*v3;
  v3[1] += *(_QWORD *)(a3 + 104);
  v3[2] += *(_QWORD *)(a3 + 176) - *(_QWORD *)(a3 + 104);
  ++a1[129];
  a1[130] += *(_QWORD *)(a3 + 104);
  result = *(_QWORD *)(a3 + 176) - *(_QWORD *)(a3 + 104);
  a1[131] += result;
  return result;
}
