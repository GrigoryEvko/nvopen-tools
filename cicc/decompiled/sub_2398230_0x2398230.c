// Function: sub_2398230
// Address: 0x2398230
//
void __fastcall sub_2398230(__int64 a1)
{
  __int64 v2; // rdi

  v2 = a1 + 16;
  *(_QWORD *)(v2 - 16) = &unk_4A0AB88;
  sub_2398130(v2);
  if ( (*(_BYTE *)(a1 + 24) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 32), 16LL * *(unsigned int *)(a1 + 40), 8);
}
