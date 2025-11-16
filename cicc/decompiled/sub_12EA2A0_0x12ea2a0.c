// Function: sub_12EA2A0
// Address: 0x12ea2a0
//
__int64 __fastcall sub_12EA2A0(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, _QWORD); // rbx
  __int64 v4; // rsi

  v3 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v4 = sub_190BB10(*(unsigned __int8 *)(a1 + 14), *(unsigned __int8 *)(a1 + 15));
  return v3(a2, v4, 0);
}
