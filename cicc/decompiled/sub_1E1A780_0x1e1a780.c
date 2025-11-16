// Function: sub_1E1A780
// Address: 0x1e1a780
//
__int64 __fastcall sub_1E1A780(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax

  v2 = sub_1E16520(a1);
  sub_1E313E0(*(_QWORD *)(a1 + 32), a2);
  sub_1E313C0(*(_QWORD *)(a1 + 32) + 40LL, 0);
  result = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(result + 144) = v2;
  return result;
}
