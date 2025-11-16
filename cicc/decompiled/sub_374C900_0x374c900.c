// Function: sub_374C900
// Address: 0x374c900
//
__int64 __fastcall sub_374C900(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64, unsigned __int16); // rax
  __int64 v9; // rsi

  v6 = *(_QWORD *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v7 + 552LL);
  if ( v8 == sub_2EC09E0 )
    v9 = *(_QWORD *)(v7 + 8LL * (unsigned __int16)a2 + 112);
  else
    v9 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v8)(v7, a2, a3);
  return sub_2EC06C0(v6, v9, byte_3F871B3, 0, a5, a6);
}
