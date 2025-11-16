// Function: sub_7E7C20
// Address: 0x7e7c20
//
__m128i *__fastcall sub_7E7C20(__int64 a1, __int64 a2, int a3, int a4)
{
  char v7; // al
  __m128i *v8; // r13

  if ( a2 )
  {
    v7 = *(_BYTE *)(a2 + 28);
    if ( ((v7 - 15) & 0xFD) != 0 && v7 != 2 )
      a3 = 1;
  }
  else if ( !qword_4F04C50 )
  {
    a3 = 1;
  }
  v8 = sub_7E20D0(a1, a3);
  sub_7E7A90((__int64)v8, a2, a4);
  return v8;
}
