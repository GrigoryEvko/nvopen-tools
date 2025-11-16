// Function: sub_65C210
// Address: 0x65c210
//
void __fastcall sub_65C210(__int64 a1)
{
  __int64 v1; // r12
  const __m128i *v3; // rdi
  __int64 v4; // r12

  v1 = *(_QWORD *)(a1 + 352);
  if ( v1 )
  {
    if ( *(char *)(a1 + 121) < 0 )
      sub_65C1C0(a1);
    if ( *(_BYTE *)(v1 + 16) == 53 )
    {
      v3 = *(const __m128i **)(a1 + 200);
      v4 = *(_QWORD *)(v1 + 24);
      if ( v3 || (v3 = *(const __m128i **)(a1 + 184)) != 0 )
        *(_QWORD *)(v4 + 48) = sub_5CF190(v3);
      if ( *(_BYTE *)(a1 + 268) )
      {
        *(_BYTE *)(v4 + 58) |= 1u;
        *(_BYTE *)(v4 + 56) = *(_BYTE *)(a1 + 268);
      }
    }
  }
}
