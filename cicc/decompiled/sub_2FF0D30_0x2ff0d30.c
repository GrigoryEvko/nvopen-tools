// Function: sub_2FF0D30
// Address: 0x2ff0d30
//
__int64 __fastcall sub_2FF0D30(__int64 a1)
{
  __int64 v1; // r12
  __int64 (__fastcall *v2)(__int64, __int64, _QWORD); // rbx
  __int64 v3; // rsi

  v1 = *(_QWORD *)(a1 + 176);
  v2 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v1 + 16LL);
  v3 = sub_352E240();
  return v2(v1, v3, 0);
}
