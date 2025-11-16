// Function: sub_7EB800
// Address: 0x7eb800
//
void __fastcall sub_7EB800(__int64 a1, __m128i *a2)
{
  if ( unk_4D03F90 || (*(_BYTE *)(a1 - 8) & 1) == 0 )
  {
    sub_7EB190(a1, a2);
  }
  else
  {
    if ( *(_BYTE *)(a1 + 173) == 6 && *(_BYTE *)(a1 + 176) == 5 )
      sub_7DC650(*(_QWORD *)(a1 + 184));
    if ( !*(_QWORD *)(a1 - 16) )
      sub_729FB0(a1, 2, qword_4D03FF0);
  }
}
