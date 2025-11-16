// Function: sub_7E6FC0
// Address: 0x7e6fc0
//
void __fastcall sub_7E6FC0(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 72);
  if ( *(_BYTE *)(v1 + 24) == 1 && *(_BYTE *)(v1 + 56) == 5 )
  {
    if ( (unsigned int)sub_8D2600(*(_QWORD *)v1) )
      sub_730620(v1, *(const __m128i **)(v1 + 72));
  }
}
