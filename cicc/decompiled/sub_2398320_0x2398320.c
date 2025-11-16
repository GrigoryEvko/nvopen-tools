// Function: sub_2398320
// Address: 0x2398320
//
void __fastcall sub_2398320(__int64 a1)
{
  __int64 v2; // rdi

  v2 = a1 + 16;
  *(_QWORD *)(v2 - 16) = &unk_4A0ABB0;
  sub_2398130(v2);
  if ( (*(_BYTE *)(a1 + 24) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 32), 16LL * *(unsigned int *)(a1 + 40), 8);
}
