// Function: sub_250B4A0
// Address: 0x250b4a0
//
__m128i *__fastcall sub_250B4A0(__m128i *a1, __int64 a2)
{
  char v3; // al
  __int64 (__fastcall *v4)(__int64); // rax
  size_t v5; // r8
  char *v6; // rcx
  __m128i *v7; // rax
  __int64 v8; // rcx
  char *v9; // rdi
  char *v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+8h] [rbp-58h]
  char v13[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15; // [rsp+30h] [rbp-30h] BYREF

  v3 = sub_2509800((_QWORD *)(*(_QWORD *)a2 + 72LL));
  sub_2509010(v14, v3);
  v4 = *(__int64 (__fastcall **)(__int64))(**(_QWORD **)a2 + 72LL);
  if ( v4 == sub_25086E0 )
  {
    v5 = 9;
    v11 = v13;
    v6 = v13;
    strcpy(v13, "AANonNull");
    v12 = 9;
  }
  else
  {
    v4((__int64)&v11);
    v6 = v11;
    v5 = v12;
  }
  v7 = (__m128i *)sub_2241130((unsigned __int64 *)v14, 0, 0, v6, v5);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    a1[1] = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v7->m128i_i64[0];
    a1[1].m128i_i64[0] = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_i64[1];
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v9 = v11;
  v7->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v8;
  v7[1].m128i_i8[0] = 0;
  if ( v9 != v13 )
    j_j___libc_free_0((unsigned __int64)v9);
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0]);
  return a1;
}
