// Function: sub_261AF60
// Address: 0x261af60
//
void __fastcall sub_261AF60(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *(_QWORD *)a1 = &unk_4A1F9E0;
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 592);
  if ( v2 )
    v2(a1 + 576, a1 + 576, 3);
  v3 = *(_QWORD *)(a1 + 176);
  if ( v3 != a1 + 192 )
    _libc_free(v3);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104));
  v4 = *(_QWORD *)(a1 + 72);
  if ( v4 != a1 + 88 )
    _libc_free(v4);
  j_j___libc_free_0(a1);
}
