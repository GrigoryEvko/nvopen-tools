// Function: sub_37E6E00
// Address: 0x37e6e00
//
__int64 __fastcall sub_37E6E00(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  *(_QWORD *)a1 = off_4A3D538;
  v2 = *(_QWORD *)(a1 + 488);
  if ( v2 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 448);
  if ( v3 != a1 + 472 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 432);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 96);
    if ( v5 != v4 + 112 )
      _libc_free(v5);
    v6 = *(_QWORD *)(v4 + 40);
    if ( v6 != v4 + 56 )
      _libc_free(v6);
    j_j___libc_free_0(v4);
  }
  if ( (*(_BYTE *)(a1 + 360) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 368), 16LL * *(unsigned int *)(a1 + 376), 8);
  v7 = *(_QWORD *)(a1 + 200);
  if ( v7 != a1 + 216 )
    _libc_free(v7);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
