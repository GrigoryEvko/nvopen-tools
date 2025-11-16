// Function: sub_2358560
// Address: 0x2358560
//
void __fastcall sub_2358560(unsigned __int64 *a1, __int16 *a2)
{
  __int16 v2; // ax
  unsigned __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  __m128i *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __m128i *v11; // [rsp+8h] [rbp-78h] BYREF
  __int16 v12; // [rsp+10h] [rbp-70h]
  int v13; // [rsp+12h] [rbp-6Eh]
  __int16 v14; // [rsp+16h] [rbp-6Ah]
  __m128i *v15; // [rsp+18h] [rbp-68h]
  __int64 v16; // [rsp+20h] [rbp-60h]
  __m128i v17; // [rsp+28h] [rbp-58h] BYREF
  __m128i *v18; // [rsp+38h] [rbp-48h]
  __int64 v19; // [rsp+40h] [rbp-40h]
  __m128i v20[3]; // [rsp+48h] [rbp-38h] BYREF

  v2 = *a2;
  v3 = *((_QWORD *)a2 + 1);
  v15 = &v17;
  v12 = v2;
  v13 = *(_DWORD *)(a2 + 1);
  v14 = a2[3];
  if ( (__int16 *)v3 == a2 + 12 )
  {
    v17 = _mm_loadu_si128((const __m128i *)(a2 + 12));
  }
  else
  {
    v15 = (__m128i *)v3;
    v17.m128i_i64[0] = *((_QWORD *)a2 + 3);
  }
  v4 = *((_QWORD *)a2 + 2);
  *((_QWORD *)a2 + 1) = a2 + 12;
  *((_QWORD *)a2 + 2) = 0;
  v16 = v4;
  v5 = *((_QWORD *)a2 + 5);
  *((_BYTE *)a2 + 24) = 0;
  v18 = v20;
  if ( (__int16 *)v5 == a2 + 28 )
  {
    v20[0] = _mm_loadu_si128((const __m128i *)(a2 + 28));
  }
  else
  {
    v18 = (__m128i *)v5;
    v20[0].m128i_i64[0] = *((_QWORD *)a2 + 7);
  }
  v6 = *((_QWORD *)a2 + 6);
  *((_QWORD *)a2 + 5) = a2 + 28;
  *((_QWORD *)a2 + 6) = 0;
  *((_BYTE *)a2 + 56) = 0;
  v19 = v6;
  v7 = sub_22077B0(0x50u);
  v8 = (__m128i *)v7;
  if ( v7 )
  {
    *(_QWORD *)v7 = &unk_4A0D4F8;
    *(_WORD *)(v7 + 8) = v12;
    *(_DWORD *)(v7 + 10) = v13;
    *(_WORD *)(v7 + 14) = v14;
    *(_QWORD *)(v7 + 16) = v7 + 32;
    if ( v15 == &v17 )
    {
      *(__m128i *)(v7 + 32) = _mm_loadu_si128(&v17);
    }
    else
    {
      *(_QWORD *)(v7 + 16) = v15;
      *(_QWORD *)(v7 + 32) = v17.m128i_i64[0];
    }
    v9 = v16;
    v15 = &v17;
    v16 = 0;
    v8[1].m128i_i64[1] = v9;
    v8[3].m128i_i64[0] = (__int64)v8[4].m128i_i64;
    v17.m128i_i8[0] = 0;
    if ( v18 == v20 )
    {
      v8[4] = _mm_loadu_si128(v20);
    }
    else
    {
      v8[3].m128i_i64[0] = (__int64)v18;
      v8[4].m128i_i64[0] = v20[0].m128i_i64[0];
    }
    v10 = v19;
    v18 = v20;
    v19 = 0;
    v8[3].m128i_i64[1] = v10;
    v20[0].m128i_i8[0] = 0;
  }
  v11 = v8;
  sub_2356EF0(a1, (unsigned __int64 *)&v11);
  if ( v11 )
    (*(void (__fastcall **)(__m128i *))(v11->m128i_i64[0] + 8))(v11);
  if ( v18 != v20 )
    j_j___libc_free_0((unsigned __int64)v18);
  if ( v15 != &v17 )
    j_j___libc_free_0((unsigned __int64)v15);
}
