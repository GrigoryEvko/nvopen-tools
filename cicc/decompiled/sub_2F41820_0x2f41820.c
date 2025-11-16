// Function: sub_2F41820
// Address: 0x2f41820
//
__int64 __fastcall sub_2F41820(__int64 a1, __int64 a2, unsigned int a3, unsigned __int16 a4)
{
  __int64 v4; // r13
  unsigned int v6; // eax

  v4 = a3;
  v6 = sub_2F41760((__int64 *)a1, a3);
  return (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD, _QWORD, unsigned __int64, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 24) + 568LL))(
           *(_QWORD *)(a1 + 24),
           *(_QWORD *)(a1 + 384),
           a2,
           a4,
           v6,
           *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL) + 16 * (v4 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
           *(_QWORD *)(a1 + 16),
           v4,
           0);
}
