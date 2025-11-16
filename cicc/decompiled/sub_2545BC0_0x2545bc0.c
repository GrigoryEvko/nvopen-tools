// Function: sub_2545BC0
// Address: 0x2545bc0
//
__int64 __fastcall sub_2545BC0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = &unk_4A1D118;
  *(_QWORD *)(a1 + 88) = &unk_4A1D1C0;
  *(_QWORD *)(a1 + 104) = &unk_4A1D220;
  v2 = *(_QWORD *)(a1 + 152);
  if ( v2 != a1 + 168 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 8LL * *(unsigned int *)(a1 + 144), 8);
  v3 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v3 != a1 + 56 )
    _libc_free(v3);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
