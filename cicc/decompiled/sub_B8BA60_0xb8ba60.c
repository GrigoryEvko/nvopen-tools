// Function: sub_B8BA60
// Address: 0xb8ba60
//
__int64 __fastcall sub_B8BA60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // rsi
  __int64 (__fastcall *v6)(__int64, __int64, __int64, __int64, unsigned __int64); // rax

  v5 = *(_QWORD *)(a2 + 24);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v5 + 32LL);
  if ( v6 == sub_B8BA50 )
    sub_B8B9D0(a1, v5 - 176, a3, a4, a5);
  else
    ((void (__fastcall *)(__int64))v6)(a1);
  return a1;
}
