// Function: sub_235E050
// Address: 0x235e050
//
void __fastcall sub_235E050(unsigned __int64 *a1, __m128i *a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // [rsp+8h] [rbp-38h] BYREF
  __m128i *v7; // [rsp+10h] [rbp-30h]
  __int64 v8; // [rsp+18h] [rbp-28h]
  __m128i v9[2]; // [rsp+20h] [rbp-20h] BYREF

  v2 = a2->m128i_i64[0];
  v7 = v9;
  if ( (__m128i *)v2 == &a2[1] )
  {
    v9[0] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v7 = (__m128i *)v2;
    v9[0].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v3 = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  a2->m128i_i64[1] = 0;
  a2[1].m128i_i8[0] = 0;
  v8 = v3;
  v4 = sub_22077B0(0x28u);
  if ( v4 )
  {
    *(_QWORD *)v4 = &unk_4A155B8;
    *(_QWORD *)(v4 + 8) = v4 + 24;
    if ( v7 == v9 )
    {
      *(__m128i *)(v4 + 24) = _mm_load_si128(v9);
    }
    else
    {
      *(_QWORD *)(v4 + 8) = v7;
      *(_QWORD *)(v4 + 24) = v9[0].m128i_i64[0];
    }
    v5 = v8;
    v7 = v9;
    v8 = 0;
    *(_QWORD *)(v4 + 16) = v5;
    v9[0].m128i_i8[0] = 0;
  }
  v6 = v4;
  sub_235DE40(a1, (unsigned __int64 *)&v6);
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  if ( v7 != v9 )
    j_j___libc_free_0((unsigned __int64)v7);
}
