// Function: sub_18A7A00
// Address: 0x18a7a00
//
__m128i *sub_18A7A00()
{
  _BYTE *v0; // r14
  __int64 v1; // r13
  __m128i *v2; // rax
  __m128i *v3; // r12

  v0 = (_BYTE *)qword_4FAD640;
  v1 = qword_4FAD648;
  v2 = (__m128i *)sub_22077B0(1448);
  v3 = v2;
  if ( v2 )
    sub_18A75E0(v2, v0, v1);
  return v3;
}
