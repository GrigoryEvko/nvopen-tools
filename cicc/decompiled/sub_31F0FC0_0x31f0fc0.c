// Function: sub_31F0FC0
// Address: 0x31f0fc0
//
__int64 __fastcall sub_31F0FC0(_QWORD *a1, __int64 a2, char a3)
{
  __int64 v4; // r13
  __int64 (__fastcall *v5)(__int64, __int64, _QWORD); // rbx
  unsigned int v6; // eax

  if ( (a3 & 7) == 1 )
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))(*a1 + 424LL))(a1, a2, 0, 0);
  v4 = a1[28];
  v5 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v4 + 536LL);
  v6 = sub_31F0C50((__int64)a1, a3);
  return v5(v4, a2, v6);
}
