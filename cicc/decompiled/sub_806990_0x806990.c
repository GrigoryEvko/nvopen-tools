// Function: sub_806990
// Address: 0x806990
//
__m128i *__fastcall sub_806990(__int64 a1)
{
  __m128i *v1; // r12
  _BYTE *v2; // rax
  __int64 v3; // r13
  __int64 v4; // rax

  v1 = sub_726410();
  v2 = sub_726B30(7);
  *((_QWORD *)v2 + 9) = v1;
  v3 = (__int64)v2;
  v1[5].m128i_i8[8] |= 4u;
  v1[8].m128i_i64[0] = (__int64)v2;
  sub_730430((__int64)v1);
  v4 = qword_4D03F68[2];
  if ( v4 && *(_QWORD *)(qword_4F07288 + 88) == v4 )
    v4 = 0;
  *(_QWORD *)(v3 + 80) = v4;
  sub_7E6810(v3, a1, 1);
  return v1;
}
