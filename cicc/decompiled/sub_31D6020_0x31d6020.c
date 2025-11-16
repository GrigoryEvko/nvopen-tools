// Function: sub_31D6020
// Address: 0x31d6020
//
void __fastcall sub_31D6020(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = &unk_4A34FB8;
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 600);
  if ( v2 )
    v2(a1 + 584, a1 + 584, 3);
  v3 = *(_QWORD *)(a1 + 184);
  if ( v3 != a1 + 200 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 144);
  if ( v4 )
    j_j___libc_free_0(v4);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104));
  v5 = *(_QWORD *)(a1 + 72);
  if ( v5 != a1 + 88 )
    _libc_free(v5);
  j_j___libc_free_0(a1);
}
