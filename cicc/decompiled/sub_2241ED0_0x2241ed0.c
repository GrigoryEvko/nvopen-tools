// Function: sub_2241ED0
// Address: 0x2241ed0
//
__int64 (__fastcall **__fastcall sub_2241ED0(_QWORD *a1, __int64 a2, __int64 a3, char *a4))()
{
  __m128i *v6; // rax
  unsigned __int64 v7; // rcx
  size_t v8; // rax
  __m128i *v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v12[2]; // [rsp+0h] [rbp-88h] BYREF
  __m128i v13; // [rsp+10h] [rbp-78h] BYREF
  unsigned __int64 v14[2]; // [rsp+20h] [rbp-68h] BYREF
  __m128i v15; // [rsp+30h] [rbp-58h] BYREF
  unsigned __int64 v16[2]; // [rsp+40h] [rbp-48h] BYREF
  char v17; // [rsp+50h] [rbp-38h] BYREF

  (*(void (__fastcall **)(unsigned __int64 *, __int64, _QWORD))(*(_QWORD *)a3 + 32LL))(v16, a3, (unsigned int)a2);
  v6 = (__m128i *)sub_2241130(v16, 0, 0, ": ", 2u);
  v14[0] = (unsigned __int64)&v15;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v15 = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v14[0] = v6->m128i_i64[0];
    v15.m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_u64[1];
  v6[1].m128i_i8[0] = 0;
  v14[1] = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  v8 = strlen(a4);
  v9 = (__m128i *)sub_2241130(v14, 0, 0, a4, v8);
  v12[0] = (unsigned __int64)&v13;
  if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
  {
    v13 = _mm_loadu_si128(v9 + 1);
  }
  else
  {
    v12[0] = v9->m128i_i64[0];
    v13.m128i_i64[0] = v9[1].m128i_i64[0];
  }
  v10 = v9->m128i_u64[1];
  v9[1].m128i_i8[0] = 0;
  v12[1] = v10;
  v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
  v9->m128i_i64[1] = 0;
  sub_2223570((__int64)a1, v12);
  if ( (__m128i *)v12[0] != &v13 )
    j___libc_free_0(v12[0]);
  if ( (__m128i *)v14[0] != &v15 )
    j___libc_free_0(v14[0]);
  if ( (char *)v16[0] != &v17 )
    j___libc_free_0(v16[0]);
  a1[2] = a2;
  a1[3] = a3;
  *a1 = off_4A07678;
  return off_4A07678;
}
