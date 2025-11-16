// Function: sub_31F0CC0
// Address: 0x31f0cc0
//
__int64 __fastcall sub_31F0CC0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned __int8 *v7; // r14
  __int64 v9; // r14
  __int64 (__fastcall *v10)(__int64, _QWORD, _QWORD); // r15
  unsigned int v11; // eax

  if ( a2 )
  {
    v4 = sub_31DA6B0((__int64)a1);
    v5 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v4 + 136LL))(
           v4,
           a2,
           a3,
           a1[25],
           a1[30],
           a1[28]);
    v6 = a1[28];
    v7 = (unsigned __int8 *)v5;
    sub_31F0C50((__int64)a1, a3);
    return sub_E9A5B0(v6, v7);
  }
  else
  {
    v9 = a1[28];
    v10 = *(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 536LL);
    v11 = sub_31F0C50((__int64)a1, a3);
    return v10(v9, 0, v11);
  }
}
