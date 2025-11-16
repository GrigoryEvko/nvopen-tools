// Function: sub_2356560
// Address: 0x2356560
//
__int64 __fastcall sub_2356560(__int64 a1, __int64 a2, char a3, char a4, char a5)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rdx
  __m128i *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v16; // [rsp+8h] [rbp-68h] BYREF
  __int64 v17; // [rsp+10h] [rbp-60h]
  __m128i *v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+20h] [rbp-50h]
  __m128i v20[4]; // [rsp+28h] [rbp-48h] BYREF

  v8 = *(_QWORD *)a2;
  v9 = *(_QWORD *)(a2 + 8);
  v18 = v20;
  v17 = v8;
  if ( v9 == a2 + 24 )
  {
    v20[0] = _mm_loadu_si128((const __m128i *)(a2 + 24));
  }
  else
  {
    v18 = (__m128i *)v9;
    v20[0].m128i_i64[0] = *(_QWORD *)(a2 + 24);
  }
  v10 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = a2 + 24;
  *(_QWORD *)(a2 + 16) = 0;
  *(_BYTE *)(a2 + 24) = 0;
  v19 = v10;
  v11 = (__m128i *)sub_22077B0(0x30u);
  if ( v11 )
  {
    v11->m128i_i64[0] = (__int64)&unk_4A122F8;
    v11->m128i_i64[1] = v17;
    v11[1].m128i_i64[0] = (__int64)v11[2].m128i_i64;
    if ( v18 == v20 )
    {
      v11[2] = _mm_loadu_si128(v20);
    }
    else
    {
      v11[1].m128i_i64[0] = (__int64)v18;
      v11[2].m128i_i64[0] = v20[0].m128i_i64[0];
    }
    v12 = v19;
    v18 = v20;
    v19 = 0;
    v11[1].m128i_i64[1] = v12;
    v20[0].m128i_i8[0] = 0;
  }
  *(_BYTE *)(a1 + 50) = a5;
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = a3;
  *(_BYTE *)(a1 + 49) = a4;
  *(_BYTE *)(a1 + 51) = 0;
  v13 = (_QWORD *)sub_22077B0(0x10u);
  if ( v13 )
    *v13 = &unk_4A0B640;
  v16 = v13;
  sub_2353900((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)&v16);
  if ( v16 )
    (*(void (__fastcall **)(_QWORD *))(*v16 + 8LL))(v16);
  v14 = (_QWORD *)sub_22077B0(0x10u);
  if ( v14 )
    *v14 = &unk_4A0B680;
  v16 = v14;
  sub_2353900((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)&v16);
  if ( v16 )
    (*(void (__fastcall **)(_QWORD *))(*v16 + 8LL))(v16);
  if ( v18 != v20 )
    j_j___libc_free_0((unsigned __int64)v18);
  return a1;
}
