// Function: sub_7E0190
// Address: 0x7e0190
//
void __fastcall sub_7E0190(__int64 a1)
{
  char v1; // al
  const __m128i *v3; // rdi

  if ( (*(_BYTE *)(a1 + 25) & 1) == 0 )
  {
    v1 = *(_BYTE *)(a1 + 24);
    if ( v1 == 1 )
    {
      if ( *(_BYTE *)(a1 + 56) == 9 )
        return;
    }
    else if ( v1 == 4 )
    {
      return;
    }
    v3 = *(const __m128i **)a1;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 140LL) & 0xFB) == 8 )
    {
      if ( (unsigned int)sub_8D4C10(v3, dword_4F077C4 != 2) )
        *(_QWORD *)a1 = sub_73D4C0(*(const __m128i **)a1, dword_4F077C4 == 2);
    }
  }
}
