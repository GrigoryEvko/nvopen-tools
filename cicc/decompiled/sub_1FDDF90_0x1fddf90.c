// Function: sub_1FDDF90
// Address: 0x1fddf90
//
__int64 __fastcall sub_1FDDF90(__int64 a1, unsigned __int8 a2)
{
  size_t v2; // r13
  __int64 (*v3)(); // rax
  __int64 v4; // rdi
  __int64 v5; // r8
  int v6; // r9d
  __int64 (__fastcall *v7)(__int64, unsigned __int8); // rax
  __int64 v8; // rsi

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 16LL) + 56LL);
  if ( v3 == sub_1D12D20 )
    BUG();
  v4 = v3();
  v7 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v4 + 288LL);
  if ( v7 == sub_1D45FB0 )
    v8 = *(_QWORD *)(v4 + 8LL * a2 + 120);
  else
    v8 = v7(v4, a2);
  return sub_1E6B9A0(v2, v8, (unsigned __int8 *)byte_3F871B3, 0, v5, v6);
}
