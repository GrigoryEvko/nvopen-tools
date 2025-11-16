// Function: sub_B2EBE0
// Address: 0xb2ebe0
//
__int64 __fastcall sub_B2EBE0(__int64 a1, __m128i *a2)
{
  __int64 v2; // rax
  __m128i *v3; // rdx
  __int64 v4; // rdx
  __int64 result; // rax
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  _OWORD v7[3]; // [rsp+10h] [rbp-30h] BYREF

  sub_B2E700(a1, 14, a2->m128i_i64[1] != 0);
  v2 = sub_B2BE50(a1);
  v3 = (__m128i *)a2->m128i_i64[0];
  v6[0] = v7;
  if ( v3 == &a2[1] )
  {
    v7[0] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v6[0] = v3;
    *(_QWORD *)&v7[0] = a2[1].m128i_i64[0];
  }
  v4 = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  a2->m128i_i64[1] = 0;
  a2[1].m128i_i8[0] = 0;
  v6[1] = v4;
  result = sub_B70050(v2, a1, v6);
  if ( (_OWORD *)v6[0] != v7 )
    return j_j___libc_free_0(v6[0], *(_QWORD *)&v7[0] + 1LL);
  return result;
}
