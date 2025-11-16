// Function: sub_8992B0
// Address: 0x8992b0
//
__m128i *__fastcall sub_8992B0(__int64 a1)
{
  __m128i *v1; // rax
  __m128i *v2; // r12

  v1 = sub_727240();
  *(_QWORD *)(a1 + 88) = v1;
  v2 = v1;
  sub_88EF90(a1);
  return v2;
}
