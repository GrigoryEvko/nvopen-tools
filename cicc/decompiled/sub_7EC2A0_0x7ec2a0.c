// Function: sub_7EC2A0
// Address: 0x7ec2a0
//
const __m128i *__fastcall sub_7EC2A0(const __m128i *a1, __int64 a2)
{
  if ( (a1[1].m128i_i8[9] & 3) != 0 )
  {
    sub_7E0590((__int64)a1);
    return (const __m128i *)sub_73E1B0((__int64)a1, a2);
  }
  else if ( (unsigned int)sub_8D2E30(a1->m128i_i64[0]) )
  {
    return a1;
  }
  else
  {
    return sub_7EC130(a1);
  }
}
