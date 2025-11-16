// Function: sub_226AF50
// Address: 0x226af50
//
void __fastcall sub_226AF50(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = &unk_4A08488;
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 168);
  if ( v2 )
    v2(a1 + 152, a1 + 152, 3);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104));
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != a1 + 88 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
