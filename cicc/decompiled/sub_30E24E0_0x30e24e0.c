// Function: sub_30E24E0
// Address: 0x30e24e0
//
void __fastcall sub_30E24E0(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = off_49D8B18;
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 16LL * *(unsigned int *)(a1 + 240), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 192), 16LL * *(unsigned int *)(a1 + 208), 8);
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 168);
  if ( v2 )
    v2(a1 + 152, a1 + 152, 3);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 != a1 + 24 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
