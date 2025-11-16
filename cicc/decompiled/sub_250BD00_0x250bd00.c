// Function: sub_250BD00
// Address: 0x250bd00
//
__m128i *__fastcall sub_250BD00(__m128i *a1, __int64 a2)
{
  char v3; // al
  __int64 *(__fastcall *v4)(__int64 *); // rax
  __int64 v5; // rax
  __m128i *v6; // rax
  __int64 v7; // rcx
  _QWORD *v8; // rdi
  size_t v10; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v11; // [rsp+10h] [rbp-60h] BYREF
  size_t v12; // [rsp+18h] [rbp-58h]
  _QWORD v13[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v14[2]; // [rsp+30h] [rbp-40h] BYREF
  __int64 v15; // [rsp+40h] [rbp-30h] BYREF

  v3 = sub_2509800((_QWORD *)(*(_QWORD *)a2 + 72LL));
  sub_2509010(v14, v3);
  v4 = *(__int64 *(__fastcall **)(__int64 *))(**(_QWORD **)a2 + 72LL);
  if ( v4 == sub_2508D10 )
  {
    v10 = 18;
    v11 = v13;
    v5 = sub_22409D0((__int64)&v11, &v10, 0);
    v11 = (_BYTE *)v5;
    v13[0] = v10;
    *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_4389C70);
    *(_WORD *)(v5 + 16) = 28518;
    v12 = v10;
    v11[v10] = 0;
  }
  else
  {
    v4((__int64 *)&v11);
  }
  v6 = (__m128i *)sub_2241130((unsigned __int64 *)v14, 0, 0, v11, v12);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    a1[1] = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v6->m128i_i64[0];
    a1[1].m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v8 = v11;
  v6->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v7;
  v6[1].m128i_i8[0] = 0;
  if ( v8 != v13 )
    j_j___libc_free_0((unsigned __int64)v8);
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0]);
  return a1;
}
