// Function: sub_1E6BD20
// Address: 0x1e6bd20
//
__int64 __fastcall sub_1E6BD20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d

  v5 = a5;
  sub_1F03430(a1, a2, a3, a4, a5);
  return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 2120) + 24LL))(
           *(_QWORD *)(a1 + 2120),
           a3,
           a4,
           v5);
}
