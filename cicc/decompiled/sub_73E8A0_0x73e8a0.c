// Function: sub_73E8A0
// Address: 0x73e8a0
//
_BYTE *__fastcall sub_73E8A0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // r12

  v2 = sub_73DE50(a1, a2);
  v2[25] &= ~1u;
  v3 = v2;
  *(_QWORD *)v2 = sub_73D720(*(const __m128i **)v2);
  return v3;
}
