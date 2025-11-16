// Function: sub_37E6CB0
// Address: 0x37e6cb0
//
__int64 __fastcall sub_37E6CB0(_QWORD *a1, __int64 a2, int *a3)
{
  unsigned int v4; // eax
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 (__fastcall *v7)(__int64, _QWORD, __int64); // r15
  __int64 v8; // rdx

  v4 = sub_37E6C00((__int64)a1, a2);
  v5 = a1[65];
  v6 = *(_QWORD *)(a2 + 24);
  v7 = *(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v5 + 320LL);
  if ( a3[63] == *(_DWORD *)(v6 + 252) && *(_DWORD *)(v6 + 256) == a3[64] )
    v8 = *(unsigned int *)(a1[25] + 8LL * a3[6]) - (unsigned __int64)v4;
  else
    v8 = sub_23CF1D0(a1[66]);
  return v7(v5, *(unsigned __int16 *)(a2 + 68), v8);
}
