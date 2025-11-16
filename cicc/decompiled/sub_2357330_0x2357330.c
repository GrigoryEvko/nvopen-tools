// Function: sub_2357330
// Address: 0x2357330
//
void __fastcall sub_2357330(unsigned __int64 *a1, __m128i *a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // [rsp+8h] [rbp-48h] BYREF
  __m128i *v9; // [rsp+10h] [rbp-40h]
  __int64 v10; // [rsp+18h] [rbp-38h]
  __m128i v11; // [rsp+20h] [rbp-30h] BYREF
  __int64 v12; // [rsp+30h] [rbp-20h]

  v2 = a2->m128i_i64[0];
  v9 = &v11;
  if ( (__m128i *)v2 == &a2[1] )
  {
    v11 = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v9 = (__m128i *)v2;
    v11.m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v3 = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v4 = a2[2].m128i_i64[0];
  a2->m128i_i64[1] = 0;
  a2[1].m128i_i8[0] = 0;
  a2[2].m128i_i64[0] = 0;
  v10 = v3;
  v12 = v4;
  v5 = sub_22077B0(0x30u);
  if ( v5 )
  {
    *(_QWORD *)v5 = &unk_4A0E938;
    *(_QWORD *)(v5 + 8) = v5 + 24;
    if ( v9 == &v11 )
    {
      *(__m128i *)(v5 + 24) = _mm_load_si128(&v11);
    }
    else
    {
      *(_QWORD *)(v5 + 8) = v9;
      *(_QWORD *)(v5 + 24) = v11.m128i_i64[0];
    }
    v6 = v10;
    v9 = &v11;
    v10 = 0;
    *(_QWORD *)(v5 + 16) = v6;
    v11.m128i_i8[0] = 0;
    *(_QWORD *)(v5 + 40) = v12;
    v12 = 0;
  }
  v8 = v5;
  sub_2356EF0(a1, (unsigned __int64 *)&v8);
  if ( v8 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  v7 = v12;
  if ( v12 && !_InterlockedSub((volatile signed __int32 *)(v12 + 8), 1u) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  if ( v9 != &v11 )
    j_j___libc_free_0((unsigned __int64)v9);
}
