// Function: sub_2544550
// Address: 0x2544550
//
void __fastcall sub_2544550(unsigned __int64 a1)
{
  _QWORD *v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = (_QWORD *)(a1 + 88);
  *v2 = &unk_4A17510;
  *(v2 - 11) = &unk_4A17458;
  sub_253BF50((__int64)v2);
  v3 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v3 != a1 + 56 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
