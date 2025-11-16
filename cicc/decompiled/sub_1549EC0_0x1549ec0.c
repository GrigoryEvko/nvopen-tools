// Function: sub_1549EC0
// Address: 0x1549ec0
//
__int64 __fastcall sub_1549EC0(__int64 a1, int a2)
{
  __int64 v4; // rcx
  __m128i *v5; // rax
  __int64 v6; // rcx
  __int64 *v7; // rdi
  _QWORD v8[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+10h] [rbp-20h] BYREF

  if ( !a2 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  sub_1549CA0((char *)v8, a2);
  if ( v8[1] == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v5 = (__m128i *)sub_2241490(v8, " ", 1, v4);
  *(_QWORD *)a1 = a1 + 16;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    *(_QWORD *)a1 = v5->m128i_i64[0];
    *(_QWORD *)(a1 + 16) = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_i64[1];
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v7 = (__int64 *)v8[0];
  v5->m128i_i64[1] = 0;
  *(_QWORD *)(a1 + 8) = v6;
  v5[1].m128i_i8[0] = 0;
  if ( v7 == &v9 )
    return a1;
  j_j___libc_free_0(v7, v9 + 1);
  return a1;
}
