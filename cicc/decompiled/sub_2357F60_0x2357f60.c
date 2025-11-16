// Function: sub_2357F60
// Address: 0x2357f60
//
void __fastcall sub_2357F60(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rdx
  __int16 v5; // ax
  __m128i *v6; // rax
  __int64 v7; // rdx
  __m128i *v8; // [rsp+8h] [rbp-48h] BYREF
  __int64 v9; // [rsp+10h] [rbp-40h]
  __m128i *v10; // [rsp+18h] [rbp-38h]
  __int64 v11; // [rsp+20h] [rbp-30h]
  __m128i v12; // [rsp+28h] [rbp-28h] BYREF
  __int16 v13; // [rsp+38h] [rbp-18h]

  v2 = *(_QWORD *)a2;
  v3 = *(_QWORD *)(a2 + 8);
  v10 = &v12;
  v9 = v2;
  if ( v3 == a2 + 24 )
  {
    v12 = _mm_loadu_si128((const __m128i *)(a2 + 24));
  }
  else
  {
    v10 = (__m128i *)v3;
    v12.m128i_i64[0] = *(_QWORD *)(a2 + 24);
  }
  v4 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = a2 + 24;
  *(_QWORD *)(a2 + 16) = 0;
  v5 = *(_WORD *)(a2 + 40);
  *(_BYTE *)(a2 + 24) = 0;
  v11 = v4;
  v13 = v5;
  v6 = (__m128i *)sub_22077B0(0x38u);
  if ( v6 )
  {
    v6->m128i_i64[0] = (__int64)&unk_4A08AE8;
    v6->m128i_i64[1] = v9;
    v6[1].m128i_i64[0] = (__int64)v6[2].m128i_i64;
    if ( v10 == &v12 )
    {
      v6[2] = _mm_loadu_si128(&v12);
    }
    else
    {
      v6[1].m128i_i64[0] = (__int64)v10;
      v6[2].m128i_i64[0] = v12.m128i_i64[0];
    }
    v7 = v11;
    v10 = &v12;
    v11 = 0;
    v6[1].m128i_i64[1] = v7;
    v12.m128i_i8[0] = 0;
    v6[3].m128i_i16[0] = v13;
  }
  v8 = v6;
  sub_2356EF0(a1, (unsigned __int64 *)&v8);
  if ( v8 )
    (*(void (__fastcall **)(__m128i *))(v8->m128i_i64[0] + 8))(v8);
  if ( v10 != &v12 )
    j_j___libc_free_0((unsigned __int64)v10);
}
