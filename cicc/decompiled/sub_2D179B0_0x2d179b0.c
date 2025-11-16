// Function: sub_2D179B0
// Address: 0x2d179b0
//
void __fastcall sub_2D179B0(unsigned __int64 a1)
{
  void (__fastcall *v2)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  *(_QWORD *)a1 = &unk_4A25DD0;
  v2 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 240);
  if ( v2 )
    v2(a1 + 224, a1 + 224, 3);
  v3 = *(_QWORD *)(a1 + 192);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 160);
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD *)(a1 + 136);
  if ( v5 )
    j_j___libc_free_0(v5);
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104));
  v6 = *(_QWORD *)(a1 + 72);
  if ( v6 != a1 + 88 )
    _libc_free(v6);
  j_j___libc_free_0(a1);
}
