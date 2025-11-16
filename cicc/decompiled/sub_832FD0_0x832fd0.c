// Function: sub_832FD0
// Address: 0x832fd0
//
__int64 __fastcall sub_832FD0(__int64 a1, __int64 a2, __m128i *a3)
{
  __m128i *v4; // r13
  __int8 v5; // al
  _BYTE *v6; // rbx
  __int64 v7; // rax
  __int8 v8; // al
  _BYTE *v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // r13
  _QWORD *v12; // rax
  __int64 *v13; // rax
  _BYTE *v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-198h]
  _OWORD v17[9]; // [rsp+10h] [rbp-190h] BYREF
  __m128i v18; // [rsp+A0h] [rbp-100h]
  __m128i v19; // [rsp+B0h] [rbp-F0h]
  __m128i v20; // [rsp+C0h] [rbp-E0h]
  __m128i v21; // [rsp+D0h] [rbp-D0h]
  __m128i v22; // [rsp+E0h] [rbp-C0h]
  __m128i v23; // [rsp+F0h] [rbp-B0h]
  __m128i v24; // [rsp+100h] [rbp-A0h]
  __m128i v25; // [rsp+110h] [rbp-90h]
  __m128i v26; // [rsp+120h] [rbp-80h]
  __m128i v27; // [rsp+130h] [rbp-70h]
  __m128i v28; // [rsp+140h] [rbp-60h]
  __m128i v29; // [rsp+150h] [rbp-50h]
  __m128i v30; // [rsp+160h] [rbp-40h]

  v4 = sub_73D720(*(const __m128i **)(a2 + 120));
  sub_6FA3A0(a3, a2);
  sub_6FC3F0((__int64)v4, a3, 1u);
  v17[0] = _mm_loadu_si128(a3);
  v5 = a3[1].m128i_i8[0];
  v17[1] = _mm_loadu_si128(a3 + 1);
  v17[2] = _mm_loadu_si128(a3 + 2);
  v17[3] = _mm_loadu_si128(a3 + 3);
  v17[4] = _mm_loadu_si128(a3 + 4);
  v17[5] = _mm_loadu_si128(a3 + 5);
  v17[6] = _mm_loadu_si128(a3 + 6);
  v17[7] = _mm_loadu_si128(a3 + 7);
  v17[8] = _mm_loadu_si128(a3 + 8);
  if ( v5 == 2 )
  {
    v18 = _mm_loadu_si128(a3 + 9);
    v19 = _mm_loadu_si128(a3 + 10);
    v20 = _mm_loadu_si128(a3 + 11);
    v21 = _mm_loadu_si128(a3 + 12);
    v22 = _mm_loadu_si128(a3 + 13);
    v23 = _mm_loadu_si128(a3 + 14);
    v24 = _mm_loadu_si128(a3 + 15);
    v25 = _mm_loadu_si128(a3 + 16);
    v26 = _mm_loadu_si128(a3 + 17);
    v27 = _mm_loadu_si128(a3 + 18);
    v28 = _mm_loadu_si128(a3 + 19);
    v29 = _mm_loadu_si128(a3 + 20);
    v30 = _mm_loadu_si128(a3 + 21);
  }
  else if ( v5 == 5 || v5 == 1 )
  {
    v18.m128i_i64[0] = a3[9].m128i_i64[0];
  }
  v6 = sub_724D50(13);
  v7 = sub_72CBE0();
  v6[176] |= 1u;
  *((_QWORD *)v6 + 16) = v7;
  *((_QWORD *)v6 + 23) = a2;
  v8 = a3[1].m128i_i8[0];
  if ( v8 == 1 )
  {
    v16 = sub_725A70(3u);
    v16[7] = a3[9].m128i_i64[0];
    v15 = sub_724D50(9);
    *((_QWORD *)v15 + 16) = v4;
    v9 = v15;
    *((_QWORD *)v15 + 22) = v16;
  }
  else
  {
    if ( v8 != 2 )
      sub_721090();
    v9 = sub_724D50(a3[19].m128i_i8[13]);
    sub_6F4950(a3, (__int64)v9);
  }
  *((_QWORD *)v6 + 15) = v9;
  v10 = sub_724D50(10);
  v10[16] = a1;
  v11 = v10;
  v10[22] = v6;
  v10[23] = v9;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
  {
    sub_6E6A50((__int64)v10, (__int64)a3);
  }
  else
  {
    v12 = sub_725A70(6u);
    v12[7] = v11;
    v13 = (__int64 *)sub_6EC670(a1, (__int64)v12, 0, 0);
    sub_6E70E0(v13, (__int64)a3);
    sub_6E26D0(2, (__int64)a3);
  }
  return sub_6E4BC0((__int64)a3, (__int64)v17);
}
