// Function: sub_2F3C530
// Address: 0x2f3c530
//
void __fastcall sub_2F3C530(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = &unk_4A2AA68;
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 192);
  if ( v2 )
    v2(a1 + 176, a1 + 176, 3);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104));
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != a1 + 88 )
    _libc_free(v3);
  j_j___libc_free_0(a1);
}
