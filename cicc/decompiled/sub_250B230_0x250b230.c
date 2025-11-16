// Function: sub_250B230
// Address: 0x250b230
//
__m128i *__fastcall sub_250B230(__m128i *a1, __int64 a2)
{
  char v3; // al
  __int64 *(__fastcall *v4)(__int64 *); // rax
  __m128i *v5; // rax
  __int64 v6; // rcx
  __int64 *v7; // rdi
  size_t v9; // [rsp+8h] [rbp-68h] BYREF
  __m128i *v10; // [rsp+10h] [rbp-60h] BYREF
  size_t v11; // [rsp+18h] [rbp-58h]
  _QWORD v12[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v13[2]; // [rsp+30h] [rbp-40h] BYREF
  __int64 v14; // [rsp+40h] [rbp-30h] BYREF

  v3 = sub_2509800((_QWORD *)(*(_QWORD *)a2 + 72LL));
  sub_2509010(v13, v3);
  v4 = *(__int64 *(__fastcall **)(__int64 *))(**(_QWORD **)a2 + 72LL);
  if ( v4 == sub_2508BA0 )
  {
    v9 = 16;
    v10 = (__m128i *)v12;
    v10 = (__m128i *)sub_22409D0((__int64)&v10, &v9, 0);
    v12[0] = v9;
    *v10 = _mm_load_si128((const __m128i *)&xmmword_4389C50);
    v11 = v9;
    v10->m128i_i8[v9] = 0;
  }
  else
  {
    v4((__int64 *)&v10);
  }
  v5 = (__m128i *)sub_2241130((unsigned __int64 *)v13, 0, 0, v10, v11);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    a1[1] = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v5->m128i_i64[0];
    a1[1].m128i_i64[0] = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_i64[1];
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v7 = (__int64 *)v10;
  v5->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v6;
  v5[1].m128i_i8[0] = 0;
  if ( v7 != v12 )
    j_j___libc_free_0((unsigned __int64)v7);
  if ( (__int64 *)v13[0] != &v14 )
    j_j___libc_free_0(v13[0]);
  return a1;
}
