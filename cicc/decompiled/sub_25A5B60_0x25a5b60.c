// Function: sub_25A5B60
// Address: 0x25a5b60
//
void __fastcall sub_25A5B60(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  if ( !*(_BYTE *)(a1 + 132) )
    _libc_free(*(_QWORD *)(a1 + 112));
  v2 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)a1 = &unk_4A1EF50;
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 )
    j_j___libc_free_0(v4);
  j_j___libc_free_0(a1);
}
