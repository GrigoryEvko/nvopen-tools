// Function: sub_31F0F00
// Address: 0x31f0f00
//
__int64 __fastcall sub_31F0F00(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 (__fastcall *v3)(__int64, __int64, _QWORD); // rbx
  unsigned int v4; // eax

  v2 = *(_QWORD *)(a1 + 224);
  v3 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 536LL);
  v4 = sub_31DF6B0(a1);
  return v3(v2, a2, v4);
}
