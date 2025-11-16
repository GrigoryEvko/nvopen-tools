// Function: sub_1E6B9A0
// Address: 0x1e6b9a0
//
__int64 __fastcall sub_1E6B9A0(size_t a1, __int64 a2, unsigned __int8 *a3, size_t a4, __int64 a5, int a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rdi

  v6 = sub_1E6B1C0(a1, a3, a4, a4, a5, a6);
  *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (v6 & 0x7FFFFFFF)) = a2;
  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 )
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v7 + 24LL))(v7, v6);
  return v6;
}
