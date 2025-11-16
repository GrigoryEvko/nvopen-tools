// Function: sub_23BB830
// Address: 0x23bb830
//
void __fastcall sub_23BB830(__m128i **a1, _QWORD *a2)
{
  __m128i *v3; // rdi
  __int64 v4; // rax
  _BYTE *v5; // rsi

  v3 = a1[1];
  if ( v3 == a1[2] )
  {
    sub_125BCB0(a1, v3, (__int64)a2);
  }
  else
  {
    if ( v3 )
    {
      v4 = a2[1];
      v5 = (_BYTE *)*a2;
      v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
      sub_23AE760(v3->m128i_i64, v5, (__int64)&v5[v4]);
      v3 = a1[1];
    }
    a1[1] = v3 + 2;
  }
}
