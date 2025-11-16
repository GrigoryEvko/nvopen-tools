// Function: sub_2398430
// Address: 0x2398430
//
void __fastcall sub_2398430(unsigned __int64 a1)
{
  __int64 v2; // rdi

  v2 = a1 + 16;
  *(_QWORD *)(v2 - 16) = &unk_4A0AB38;
  sub_2398130(v2);
  if ( (*(_BYTE *)(a1 + 24) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 32), 16LL * *(unsigned int *)(a1 + 40), 8);
  j_j___libc_free_0(a1);
}
