// Function: sub_3244F20
// Address: 0x3244f20
//
__int64 __fastcall sub_3244F20(__int64 *a1, __int64 a2)
{
  int v2; // r12d
  int v3; // eax

  v2 = sub_31DF740(*a1);
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 56LL))(a2);
  return sub_3244EC0(a1, a2 + 8, v2 + v3);
}
