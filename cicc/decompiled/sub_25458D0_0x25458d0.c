// Function: sub_25458D0
// Address: 0x25458d0
//
__int64 __fastcall sub_25458D0(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 80);
  *(_QWORD *)(a1 - 88) = &unk_4A17358;
  *(_QWORD *)a1 = &unk_4A17318;
  sub_C7D6A0(*(_QWORD *)(a1 + 64), v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  v3 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v3 != a1 - 32 )
    _libc_free(v3);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
