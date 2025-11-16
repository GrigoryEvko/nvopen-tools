// Function: sub_3982290
// Address: 0x3982290
//
__int64 __fastcall sub_3982290(_QWORD *a1, __int64 a2, __int16 a3)
{
  __int64 (__fastcall *v3)(__int64, _QWORD, _QWORD); // r13
  unsigned int v4; // eax

  v3 = *(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 344LL);
  v4 = sub_3982260((__int64)a1, a2, a3);
  return v3(a2, *a1, v4);
}
