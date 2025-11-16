// Function: sub_7F53E0
// Address: 0x7f53e0
//
_BYTE *__fastcall sub_7F53E0(__int64 a1)
{
  const __m128i *v1; // r13
  int v3; // r14d
  __m128i *v4; // rax
  __m128i *v5; // rax
  __int64 v6; // rax

  v1 = *(const __m128i **)a1;
  if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
  {
    if ( (v1[8].m128i_i8[12] & 0xFB) != 8 )
      return (_BYTE *)a1;
  }
  else
  {
    v1 = (const __m128i *)sub_8D46C0(*(_QWORD *)a1);
    if ( (v1[8].m128i_i8[12] & 0xFB) != 8 )
      return (_BYTE *)a1;
  }
  if ( (sub_8D4C10(v1, dword_4F077C4 != 2) & 1) == 0 || (unsigned int)sub_8D3410(v1) )
    return (_BYTE *)a1;
  v3 = 0;
  if ( (v1[8].m128i_i8[12] & 0xFB) == 8 )
    v3 = sub_8D4C10(v1, dword_4F077C4 != 2) & 0xFFFFFFFE;
  v4 = sub_73D4C0(v1, dword_4F077C4 == 2);
  v5 = sub_73C570(v4, v3);
  if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
    return sub_73DC90((_QWORD *)a1, (__int64)v5);
  v6 = sub_72D2E0(v5);
  return sub_73E110(a1, v6);
}
