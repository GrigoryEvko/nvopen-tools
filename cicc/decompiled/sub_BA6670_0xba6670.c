// Function: sub_BA6670
// Address: 0xba6670
//
__m128i *__fastcall sub_BA6670(const __m128i *a1, __int64 a2)
{
  __m128i *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __m128i *v7; // r13
  __int64 v8; // rdi

  v3 = (__m128i *)sub_BA3380(a1);
  v7 = v3;
  if ( a1 == v3 )
  {
    sub_B96FD0((__int64)v3, a2);
    return v7;
  }
  else
  {
    v8 = a1->m128i_i64[1];
    if ( (v8 & 4) != 0 )
    {
      a2 = (__int64)v3;
      sub_BA6110((const __m128i *)(v8 & 0xFFFFFFFFFFFFFFF8LL), v3->m128i_i64);
    }
    sub_B97380((__int64)a1, a2, v4, v5, v6);
    return v7;
  }
}
