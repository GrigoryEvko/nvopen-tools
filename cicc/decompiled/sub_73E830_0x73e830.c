// Function: sub_73E830
// Address: 0x73e830
//
_QWORD *__fastcall sub_73E830(__int64 a1)
{
  _QWORD *v1; // r12
  __m128i *v2; // rax

  v1 = sub_726700(3);
  v2 = sub_73D720(*(const __m128i **)(a1 + 120));
  v1[7] = a1;
  *v1 = v2;
  return v1;
}
