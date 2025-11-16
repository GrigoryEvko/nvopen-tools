// Function: sub_695B00
// Address: 0x695b00
//
unsigned int *__fastcall sub_695B00(const __m128i *a1, int a2, __int64 a3)
{
  _BOOL4 v3; // r13d
  _QWORD *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int8 v16; // al
  unsigned int *result; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp-10h] [rbp-1A0h]
  __int64 v23; // [rsp-8h] [rbp-198h]
  __m128i v24; // [rsp+0h] [rbp-190h] BYREF
  __m128i v25; // [rsp+10h] [rbp-180h]
  __m128i v26; // [rsp+20h] [rbp-170h]
  __m128i v27; // [rsp+30h] [rbp-160h]
  __m128i v28; // [rsp+40h] [rbp-150h]
  __m128i v29; // [rsp+50h] [rbp-140h]
  __m128i v30; // [rsp+60h] [rbp-130h]
  __m128i v31; // [rsp+70h] [rbp-120h]
  __m128i v32; // [rsp+80h] [rbp-110h]
  __int64 v33; // [rsp+90h] [rbp-100h]

  v3 = 1;
  if ( dword_4F04C44 == -1 && (a1[1].m128i_i8[4] & 8) == 0 )
  {
    v3 = 0;
    if ( a1[1].m128i_i8[0] == 1 )
    {
      v18 = a1[9].m128i_i64[0];
      if ( *(_BYTE *)(v18 + 24) == 3 )
        v3 = (*(_BYTE *)(*(_QWORD *)(v18 + 56) + 172LL) & 8) != 0;
    }
  }
  sub_843C40((_DWORD)a1, a2, 0, 0, 1, 4, 458);
  v7 = v22;
  v8 = v23;
  if ( a1[1].m128i_i8[0] != 2 )
    sub_6E6B60(a1, 1);
  if ( dword_4F077BC )
  {
    if ( !(_DWORD)qword_4F077B4 && a1[1].m128i_i8[0] != 2 )
    {
      if ( dword_4F04C44 != -1
        || (v5 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
      {
        v10 = a3;
        sub_6F4910(a1, a3, 0, v6, v7, v8);
        v16 = a1[1].m128i_i8[0];
        if ( v16 == 2 )
          goto LABEL_10;
        goto LABEL_21;
      }
    }
  }
  v9 = unk_4F04C50;
  if ( !unk_4F04C50 )
  {
    v5 = (_QWORD *)qword_4F04C68[0];
    v19 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v19 + 4) == 14 )
    {
      v20 = *(int *)(v19 + 452);
      if ( (_DWORD)v20 != -1 )
      {
        v21 = *(int *)(qword_4F04C68[0] + 776 * v20 + 400);
        if ( (_DWORD)v21 != -1 )
          v9 = *(_QWORD *)(qword_4F04C68[0] + 776 * v21 + 184);
      }
    }
  }
  v10 = a3;
  sub_6F4950(a1, a3, v5, v6, v7, v8);
  v11 = *(_QWORD *)(a3 + 144);
  sub_6959C0(a3, a3);
  if ( v11 && (*(_BYTE *)(v11 - 8) & 1) == 0 && (*(_BYTE *)(qword_4D03C50 + 21LL) & 0x40) == 0 && v9 )
  {
    v10 = 8;
    sub_72D910(v11, 8, a3);
  }
  v16 = a1[1].m128i_i8[0];
  if ( v16 != 2 )
  {
LABEL_21:
    v24 = _mm_loadu_si128(a1);
    v25 = _mm_loadu_si128(a1 + 1);
    v26 = _mm_loadu_si128(a1 + 2);
    v27 = _mm_loadu_si128(a1 + 3);
    v28 = _mm_loadu_si128(a1 + 4);
    v29 = _mm_loadu_si128(a1 + 5);
    v30 = _mm_loadu_si128(a1 + 6);
    v31 = _mm_loadu_si128(a1 + 7);
    v32 = _mm_loadu_si128(a1 + 8);
    if ( v16 == 1 || v16 == 5 )
      v33 = a1[9].m128i_i64[0];
    sub_6E6A50(a3, a1);
    v10 = (__int64)&v24;
    sub_6E4BC0(a1, &v24);
  }
LABEL_10:
  sub_72A1A0(
    a3,
    v10,
    v12,
    v13,
    v14,
    v15,
    v24.m128i_i64[0],
    v24.m128i_i64[1],
    v25.m128i_i64[0],
    v25.m128i_i64[1],
    v26.m128i_i64[0],
    v26.m128i_i64[1],
    v27.m128i_i64[0],
    v27.m128i_i64[1],
    v28.m128i_i64[0],
    v28.m128i_i64[1],
    v29.m128i_i64[0],
    v29.m128i_i64[1],
    v30.m128i_i64[0],
    v30.m128i_i64[1],
    v31.m128i_i64[0],
    v31.m128i_i64[1],
    v32.m128i_i64[0],
    v32.m128i_i64[1],
    v33);
  result = &dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 > 201102 || (result = &dword_4F07774, dword_4F07774) )
    {
      result = &dword_4F077BC;
      if ( !dword_4F077BC
        || (result = (unsigned int *)(unsigned int)qword_4F077B4, (_DWORD)qword_4F077B4)
        || (result = (unsigned int *)&qword_4F077A8, qword_4F077A8 > 0xEA5Fu) )
      {
        *(_BYTE *)(a3 + 169) |= 4u;
      }
    }
  }
  if ( !v3 )
    *(_QWORD *)(a3 + 144) = 0;
  return result;
}
