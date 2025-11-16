// Function: sub_16CFD20
// Address: 0x16cfd20
//
__int64 (__fastcall *__fastcall sub_16CFD20(__int64 *a1, _QWORD *a2, __int64 a3, unsigned __int8 a4))(__int64 a1)
{
  __int64 (__fastcall *v5)(__int64, __int64); // rax
  unsigned __int64 v8; // rsi
  unsigned int v9; // r14d
  int v10; // eax

  v5 = (__int64 (__fastcall *)(__int64, __int64))a1[6];
  if ( v5 )
    return (__int64 (__fastcall *)(__int64))v5(a3, a1[7]);
  v8 = *(_QWORD *)(a3 + 8);
  v9 = a4;
  if ( v8 )
  {
    v10 = sub_16CE270(a1, v8);
    sub_16CFB30(a1, *(_QWORD *)(*a1 + 24LL * (unsigned int)(v10 - 1) + 16), (__int64)a2);
  }
  return sub_16CE370(a3, 0, a2, v9, 1);
}
