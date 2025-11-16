// Function: sub_254A1C0
// Address: 0x254a1c0
//
void __fastcall sub_254A1C0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *(_QWORD *)a1 = off_4A1E558;
  *(_QWORD *)(a1 + 88) = &unk_4A1E5E0;
  v2 = *(_QWORD *)(a1 + 248);
  if ( v2 != a1 + 264 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 8LL * *(unsigned int *)(a1 + 240), 8);
  v3 = *(_QWORD *)(a1 + 168);
  if ( v3 != a1 + 184 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 144), 8LL * *(unsigned int *)(a1 + 160), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 16LL * *(unsigned int *)(a1 + 128), 8);
  v4 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v4 != a1 + 56 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
