// Function: sub_2358080
// Address: 0x2358080
//
void __fastcall sub_2358080(unsigned __int64 *a1, __m128i *a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rdx
  unsigned __int64 v4; // rdx
  __int8 v5; // al
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-78h] BYREF
  __m128i *v14; // [rsp+10h] [rbp-70h]
  __int64 v15; // [rsp+18h] [rbp-68h]
  __m128i v16; // [rsp+20h] [rbp-60h] BYREF
  __m128i *v17; // [rsp+30h] [rbp-50h]
  __int64 v18; // [rsp+38h] [rbp-48h]
  __m128i v19; // [rsp+40h] [rbp-40h] BYREF
  __int8 v20; // [rsp+50h] [rbp-30h]
  __int64 v21; // [rsp+58h] [rbp-28h]

  v2 = a2->m128i_i64[0];
  v14 = &v16;
  if ( (__m128i *)v2 == &a2[1] )
  {
    v16 = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v14 = (__m128i *)v2;
    v16.m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v3 = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  a2->m128i_i64[1] = 0;
  v15 = v3;
  v4 = a2[2].m128i_u64[0];
  a2[1].m128i_i8[0] = 0;
  v17 = &v19;
  if ( (__m128i *)v4 == &a2[3] )
  {
    v19 = _mm_loadu_si128(a2 + 3);
  }
  else
  {
    v17 = (__m128i *)v4;
    v19.m128i_i64[0] = a2[3].m128i_i64[0];
  }
  a2[2].m128i_i64[0] = (__int64)a2[3].m128i_i64;
  v5 = a2[4].m128i_i8[0];
  v6 = a2[2].m128i_i64[1];
  a2[3].m128i_i8[0] = 0;
  v20 = v5;
  v7 = a2[4].m128i_i64[1];
  a2[2].m128i_i64[1] = 0;
  a2[4].m128i_i64[1] = 0;
  v18 = v6;
  v21 = v7;
  v8 = sub_22077B0(0x58u);
  if ( v8 )
  {
    *(_QWORD *)v8 = &unk_4A0DB78;
    *(_QWORD *)(v8 + 8) = v8 + 24;
    if ( v14 == &v16 )
    {
      *(__m128i *)(v8 + 24) = _mm_load_si128(&v16);
    }
    else
    {
      *(_QWORD *)(v8 + 8) = v14;
      *(_QWORD *)(v8 + 24) = v16.m128i_i64[0];
    }
    v9 = v15;
    v14 = &v16;
    v15 = 0;
    *(_QWORD *)(v8 + 16) = v9;
    *(_QWORD *)(v8 + 40) = v8 + 56;
    v16.m128i_i8[0] = 0;
    if ( v17 == &v19 )
    {
      *(__m128i *)(v8 + 56) = _mm_load_si128(&v19);
    }
    else
    {
      *(_QWORD *)(v8 + 40) = v17;
      *(_QWORD *)(v8 + 56) = v19.m128i_i64[0];
    }
    v10 = v18;
    v17 = &v19;
    v18 = 0;
    *(_QWORD *)(v8 + 48) = v10;
    v19.m128i_i8[0] = 0;
    *(_BYTE *)(v8 + 72) = v20;
    v11 = v21;
    v21 = 0;
    *(_QWORD *)(v8 + 80) = v11;
  }
  v13 = v8;
  sub_2356EF0(a1, (unsigned __int64 *)&v13);
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
  v12 = v21;
  if ( v21 && !_InterlockedSub((volatile signed __int32 *)(v21 + 8), 1u) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  if ( v17 != &v19 )
    j_j___libc_free_0((unsigned __int64)v17);
  if ( v14 != &v16 )
    j_j___libc_free_0((unsigned __int64)v14);
}
