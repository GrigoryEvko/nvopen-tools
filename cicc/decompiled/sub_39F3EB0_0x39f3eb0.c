// Function: sub_39F3EB0
// Address: 0x39f3eb0
//
void __fastcall sub_39F3EB0(__int64 a1, __int32 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rsi
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+10h] [rbp-30h]

  v3 = sub_38BFA60(*(_QWORD *)(a1 + 8), 1);
  sub_39F3BB0(a1, v3, 0);
  v4 = *(_QWORD *)(a1 + 264);
  v7 = 0;
  v6 = 0;
  v5 = *(_QWORD *)(v4 + 112);
  v6.m128i_i32[0] = a2;
  v6.m128i_i64[1] = v3;
  if ( v5 == *(_QWORD *)(v4 + 120) )
  {
    sub_39F2660(v4 + 104, (_BYTE *)v5, &v6);
  }
  else
  {
    if ( v5 )
    {
      *(__m128i *)v5 = _mm_loadu_si128(&v6);
      *(_QWORD *)(v5 + 16) = v7;
      v5 = *(_QWORD *)(v4 + 112);
    }
    *(_QWORD *)(v4 + 112) = v5 + 24;
  }
}
