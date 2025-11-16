// Function: sub_30E2450
// Address: 0x30e2450
//
void __fastcall sub_30E2450(__int64 a1)
{
  void (__fastcall *v2)(__int64, __int64, __int64); // rax
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = off_49D8B98;
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 16LL * *(unsigned int *)(a1 + 240), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 192), 16LL * *(unsigned int *)(a1 + 208), 8);
  v2 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 168);
  if ( v2 )
    v2(a1 + 152, a1 + 152, 3);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 != a1 + 24 )
    _libc_free(v3);
}
