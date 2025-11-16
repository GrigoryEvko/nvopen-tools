// Function: sub_82F8F0
// Address: 0x82f8f0
//
__int64 __fastcall sub_82F8F0(const __m128i *a1, int a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v7; // al
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // r14
  int v14; // r13d
  const __m128i *v15; // rdi
  const __m128i *v16; // rax
  __int64 *v17; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  const __m128i *v24; // [rsp+0h] [rbp-1A0h]
  __int8 v25; // [rsp+Bh] [rbp-195h]
  _BOOL4 v26; // [rsp+Ch] [rbp-194h]
  _OWORD v27[9]; // [rsp+10h] [rbp-190h] BYREF
  __m128i v28; // [rsp+A0h] [rbp-100h]
  __m128i v29; // [rsp+B0h] [rbp-F0h]
  __m128i v30; // [rsp+C0h] [rbp-E0h]
  __m128i v31; // [rsp+D0h] [rbp-D0h]
  __m128i v32; // [rsp+E0h] [rbp-C0h]
  __m128i v33; // [rsp+F0h] [rbp-B0h]
  __m128i v34; // [rsp+100h] [rbp-A0h]
  __m128i v35; // [rsp+110h] [rbp-90h]
  __m128i v36; // [rsp+120h] [rbp-80h]
  __m128i v37; // [rsp+130h] [rbp-70h]
  __m128i v38; // [rsp+140h] [rbp-60h]
  __m128i v39; // [rsp+150h] [rbp-50h]
  __m128i v40; // [rsp+160h] [rbp-40h]

  v25 = a3[1].m128i_i8[1];
  v27[0] = _mm_loadu_si128(a3);
  v27[1] = _mm_loadu_si128(a3 + 1);
  v7 = a3[1].m128i_i8[0];
  v27[2] = _mm_loadu_si128(a3 + 2);
  v27[3] = _mm_loadu_si128(a3 + 3);
  v27[4] = _mm_loadu_si128(a3 + 4);
  v27[5] = _mm_loadu_si128(a3 + 5);
  v27[6] = _mm_loadu_si128(a3 + 6);
  v27[7] = _mm_loadu_si128(a3 + 7);
  v27[8] = _mm_loadu_si128(a3 + 8);
  if ( v7 == 2 )
  {
    v28 = _mm_loadu_si128(a3 + 9);
    v29 = _mm_loadu_si128(a3 + 10);
    v30 = _mm_loadu_si128(a3 + 11);
    v31 = _mm_loadu_si128(a3 + 12);
    v32 = _mm_loadu_si128(a3 + 13);
    v33 = _mm_loadu_si128(a3 + 14);
    v34 = _mm_loadu_si128(a3 + 15);
    v35 = _mm_loadu_si128(a3 + 16);
    v36 = _mm_loadu_si128(a3 + 17);
    v37 = _mm_loadu_si128(a3 + 18);
    v38 = _mm_loadu_si128(a3 + 19);
    v39 = _mm_loadu_si128(a3 + 20);
    v40 = _mm_loadu_si128(a3 + 21);
  }
  else if ( v7 == 5 || v7 == 1 )
  {
    v28.m128i_i64[0] = a3[9].m128i_i64[0];
  }
  v8 = sub_6F6F40(a1, 0, (__int64)a3, a4, a5, a6);
  v13 = (__int64 *)sub_6F6F40(a3, 0, v9, v10, v11, v12);
  if ( (dword_4F04C44 != -1
     || (v19 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v19 + 6) & 6) != 0)
     || *(_BYTE *)(v19 + 4) == 12)
    && *(_BYTE *)(v8 + 24) == 2 )
  {
    if ( (v13[3] & 0x3FF) != 2 )
    {
LABEL_7:
      v26 = 0;
      v14 = 0;
      goto LABEL_8;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v8 + 56) + 173LL) == 12 || *(_BYTE *)(v13[7] + 173) == 12 )
    {
      v26 = 1;
      v14 = 0;
      goto LABEL_8;
    }
  }
  else if ( (v13[3] & 0x3FF) != 2 || *(_BYTE *)(v13[7] + 173) == 12 )
  {
    goto LABEL_7;
  }
  if ( !sub_6ED2E0() )
    goto LABEL_7;
  v26 = sub_827180(a2, (__int64 *)v8);
  if ( v26 )
    goto LABEL_7;
  sub_6E6A50(v13[7], (__int64)a3);
  if ( dword_4F07270[0] == dword_4F073B8[0] && qword_4F04C50 || !*(_BYTE *)(qword_4D03C50 + 16LL) )
    goto LABEL_17;
  v14 = 1;
LABEL_8:
  v24 = (const __m128i *)*v13;
  if ( !(unsigned int)sub_8D2310(*v13) )
    goto LABEL_15;
  v15 = v24;
  if ( v24[8].m128i_i8[12] == 12 )
  {
    v16 = v24;
    do
      v16 = (const __m128i *)v16[10].m128i_i64[0];
    while ( v16[8].m128i_i8[12] == 12 );
    if ( *(_QWORD *)(v16[10].m128i_i64[1] + 40) )
    {
      do
        v15 = (const __m128i *)v15[10].m128i_i64[0];
      while ( v15[8].m128i_i8[12] == 12 );
      goto LABEL_14;
    }
  }
  else if ( *(_QWORD *)(v24[10].m128i_i64[1] + 40) )
  {
LABEL_14:
    sub_73F430(v15, 1);
  }
LABEL_15:
  *(_QWORD *)(v8 + 16) = v13;
  v17 = (__int64 *)sub_73DBF0(100 - ((a2 == 0) - 1), *v13, v8);
  sub_730580((__int64)v13, (__int64)v17);
  if ( v14 )
  {
    a3[19].m128i_i8[10] &= ~0x10u;
    a3[18].m128i_i64[0] = (__int64)v17;
  }
  else
  {
    sub_6E70E0(v17, (__int64)a3);
    a3[1].m128i_i8[1] = v25;
    if ( v26 )
      sub_6F4B70(a3, (__int64)a3, v20, v21, v22, v23);
    sub_6E4EE0((__int64)a3, (__int64)v27);
  }
LABEL_17:
  a3[1].m128i_i8[2] &= ~1u;
  return sub_6E26D0(2, (__int64)a3);
}
