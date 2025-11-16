// Function: sub_39F0F10
// Address: 0x39f0f10
//
void __fastcall sub_39F0F10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __m128i v6; // [rsp+0h] [rbp-20h] BYREF
  __int64 v7; // [rsp+10h] [rbp-10h]

  v4 = *(_QWORD *)(a1 + 264);
  v6.m128i_i64[0] = a2;
  v6.m128i_i64[1] = a3;
  v5 = *(_QWORD *)(v4 + 2112);
  v7 = a4;
  if ( v5 == *(_QWORD *)(v4 + 2120) )
  {
    sub_39F0D70(v4 + 2104, (_BYTE *)v5, &v6);
  }
  else
  {
    if ( v5 )
    {
      *(__m128i *)v5 = _mm_loadu_si128(&v6);
      *(_QWORD *)(v5 + 16) = v7;
      v5 = *(_QWORD *)(v4 + 2112);
    }
    *(_QWORD *)(v4 + 2112) = v5 + 24;
  }
}
