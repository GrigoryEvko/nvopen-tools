// Function: sub_ED2E50
// Address: 0xed2e50
//
__m128i *__fastcall sub_ED2E50(__m128i *a1, __int64 *a2)
{
  __int64 v2; // rax
  int v3; // esi
  __int64 v4; // rax
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  char v11; // al
  __m128i v12; // xmm4
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-78h] BYREF
  _BYTE v15[24]; // [rsp+10h] [rbp-70h] BYREF
  __int128 v16; // [rsp+28h] [rbp-58h] BYREF
  __int128 v17; // [rsp+38h] [rbp-48h] BYREF
  __int128 v18; // [rsp+48h] [rbp-38h]

  v2 = *a2;
  *(_OWORD *)&v15[8] = 0;
  v16 = 0;
  *(_QWORD *)v15 = v2;
  v17 = 0;
  v18 = 0;
  if ( v2 != 0x8169666F72706CFFLL )
  {
    v3 = 3;
LABEL_3:
    sub_ED07D0(&v14, v3);
    v4 = v14;
    a1[4].m128i_i8[8] |= 3u;
    a1->m128i_i64[0] = v4 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  *(_QWORD *)&v15[8] = a2[1];
  if ( (unsigned __int64)sub_ED2E40((__int64)v15) > 0xC )
  {
    v3 = 5;
    goto LABEL_3;
  }
  v6 = a2 + 5;
  v16 = *(_OWORD *)(a2 + 3);
  if ( (unsigned __int64)sub_ED2E40((__int64)v15) > 7 )
  {
    v6 = a2 + 6;
    *(_QWORD *)&v17 = a2[5];
  }
  if ( (unsigned __int64)sub_ED2E40((__int64)v15) > 8 )
  {
    v7 = *v6++;
    *((_QWORD *)&v17 + 1) = v7;
  }
  if ( (unsigned __int64)sub_ED2E40((__int64)v15) > 9 )
  {
    v8 = *v6++;
    *(_QWORD *)&v18 = v8;
  }
  if ( (unsigned __int64)sub_ED2E40((__int64)v15) > 0xB )
    *((_QWORD *)&v18 + 1) = *v6;
  v9 = _mm_loadu_si128((const __m128i *)&v15[16]);
  v10 = _mm_loadu_si128((const __m128i *)((char *)&v16 + 8));
  v11 = a1[4].m128i_i8[8] & 0xFC;
  v12 = _mm_loadu_si128((const __m128i *)((char *)&v17 + 8));
  *a1 = _mm_loadu_si128((const __m128i *)v15);
  a1[1] = v9;
  a1[4].m128i_i8[8] = v11 | 2;
  v13 = *((_QWORD *)&v18 + 1);
  a1[2] = v10;
  a1[4].m128i_i64[0] = v13;
  a1[3] = v12;
  return a1;
}
