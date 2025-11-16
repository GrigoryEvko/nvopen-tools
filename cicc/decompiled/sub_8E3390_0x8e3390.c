// Function: sub_8E3390
// Address: 0x8e3390
//
__m128i *__fastcall sub_8E3390(__int64 a1)
{
  _QWORD *v2; // r13
  __int64 v3; // rax

  if ( *(_BYTE *)(a1 + 140) != 8 || (*(_BYTE *)(a1 + 169) & 0x10) == 0 )
    return sub_8DC200(a1, (unsigned int (__fastcall *)(__m128i *, _QWORD, __m128i **))sub_8E3410, 0);
  v2 = sub_7259C0(8);
  v3 = sub_8E3390(*(_QWORD *)(a1 + 160));
  *((_BYTE *)v2 + 169) |= 3u;
  v2[20] = v3;
  sub_8D6090((__int64)v2);
  return (__m128i *)v2;
}
