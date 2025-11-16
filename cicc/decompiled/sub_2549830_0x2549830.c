// Function: sub_2549830
// Address: 0x2549830
//
__int64 __fastcall sub_2549830(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *(_QWORD *)(a1 - 88) = off_4A1E558;
  *(_QWORD *)a1 = &unk_4A1E5E0;
  v2 = *(_QWORD *)(a1 + 160);
  if ( v2 != a1 + 176 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 8LL * *(unsigned int *)(a1 + 152), 8);
  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 != a1 + 96 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 8LL * *(unsigned int *)(a1 + 72), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  v4 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v4 != a1 - 32 )
    _libc_free(v4);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
