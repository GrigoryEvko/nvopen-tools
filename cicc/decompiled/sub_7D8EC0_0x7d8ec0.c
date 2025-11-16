// Function: sub_7D8EC0
// Address: 0x7d8ec0
//
void __fastcall sub_7D8EC0(_QWORD *a1)
{
  __m128i *v1; // r14
  __int64 v2; // r13
  const __m128i *v3; // rax
  __int64 v4; // rax
  int v5[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( (unsigned int)sub_8D2B20(*a1) )
  {
    sub_7D8CF0((__m128i *)a1[7]);
  }
  else if ( (unsigned int)sub_8D2B50(*a1) )
  {
    v1 = (__m128i *)a1[7];
    v2 = v1[10].m128i_i64[0];
    if ( !v2 )
    {
      v4 = sub_7E7C20(*a1, *(_QWORD *)(qword_4F04C68[0] + 184LL), 0, 0);
      *(_BYTE *)(v4 + 177) = 1;
      v2 = v4;
      if ( (v1[-1].m128i_i8[8] & 1) == 0 )
      {
        sub_7296C0(v5);
        v1 = sub_740540(v1, 1, 0);
        sub_729730(v5[0]);
      }
      *(_QWORD *)(v2 + 184) = v1;
      sub_7EB800(v1);
      v1[10].m128i_i64[0] = v2;
    }
    v3 = (const __m128i *)sub_73E830(v2);
    sub_730620((__int64)a1, v3);
  }
}
