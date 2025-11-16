// Function: sub_6E6B60
// Address: 0x6e6b60
//
__int64 __fastcall sub_6E6B60(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v7; // al
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdi
  __int8 v17; // al
  __int64 v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-180h] BYREF
  __int64 v20; // [rsp+8h] [rbp-178h] BYREF
  __m128i v21; // [rsp+10h] [rbp-170h] BYREF
  __m128i v22; // [rsp+20h] [rbp-160h]
  __m128i v23; // [rsp+30h] [rbp-150h]
  __m128i v24; // [rsp+40h] [rbp-140h]
  __m128i v25; // [rsp+50h] [rbp-130h]
  __m128i v26; // [rsp+60h] [rbp-120h]
  __m128i v27; // [rsp+70h] [rbp-110h]
  __m128i v28; // [rsp+80h] [rbp-100h]
  __m128i v29; // [rsp+90h] [rbp-F0h]
  __m128i v30; // [rsp+A0h] [rbp-E0h]
  __m128i v31; // [rsp+B0h] [rbp-D0h]
  __m128i v32; // [rsp+C0h] [rbp-C0h]
  __m128i v33; // [rsp+D0h] [rbp-B0h]
  __m128i v34; // [rsp+E0h] [rbp-A0h]
  __m128i v35; // [rsp+F0h] [rbp-90h]
  __m128i v36; // [rsp+100h] [rbp-80h]
  __m128i v37; // [rsp+110h] [rbp-70h]
  __m128i v38; // [rsp+120h] [rbp-60h]
  __m128i v39; // [rsp+130h] [rbp-50h]
  __m128i v40; // [rsp+140h] [rbp-40h]
  __m128i v41; // [rsp+150h] [rbp-30h]
  __m128i v42; // [rsp+160h] [rbp-20h]

  v19 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( !word_4D04898 )
    goto LABEL_34;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
  {
    if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0 )
      goto LABEL_34;
    if ( dword_4F04C44 != -1 )
      goto LABEL_34;
    v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v9 + 6) & 6) != 0 || *(_BYTE *)(v9 + 4) == 12 )
      goto LABEL_34;
  }
  if ( !(_DWORD)a2 && !(unsigned int)sub_6DEB30(a1, v19) )
    goto LABEL_34;
  if ( a1[1].m128i_i16[0] != 513 )
    return sub_724E30(&v19);
  if ( (unsigned int)sub_719770(a1[9].m128i_i64[0], v19, (unsigned int)a2, 0) )
  {
    v21 = _mm_loadu_si128(a1);
    v22 = _mm_loadu_si128(a1 + 1);
    v7 = a1[1].m128i_i8[0];
    v23 = _mm_loadu_si128(a1 + 2);
    v24 = _mm_loadu_si128(a1 + 3);
    v25 = _mm_loadu_si128(a1 + 4);
    v26 = _mm_loadu_si128(a1 + 5);
    v27 = _mm_loadu_si128(a1 + 6);
    v28 = _mm_loadu_si128(a1 + 7);
    v29 = _mm_loadu_si128(a1 + 8);
    if ( v7 == 2 )
    {
      v30 = _mm_loadu_si128(a1 + 9);
      v31 = _mm_loadu_si128(a1 + 10);
      v32 = _mm_loadu_si128(a1 + 11);
      v33 = _mm_loadu_si128(a1 + 12);
      v34 = _mm_loadu_si128(a1 + 13);
      v35 = _mm_loadu_si128(a1 + 14);
      v36 = _mm_loadu_si128(a1 + 15);
      v37 = _mm_loadu_si128(a1 + 16);
      v38 = _mm_loadu_si128(a1 + 17);
      v39 = _mm_loadu_si128(a1 + 18);
      v40 = _mm_loadu_si128(a1 + 19);
      v41 = _mm_loadu_si128(a1 + 20);
      v42 = _mm_loadu_si128(a1 + 21);
    }
    else if ( v7 == 5 || v7 == 1 )
    {
      v30.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    sub_6E6A50(v19, (__int64)a1);
    sub_6E4BC0((__int64)a1, (__int64)&v21);
  }
  else
  {
LABEL_34:
    if ( a1[1].m128i_i16[0] == 513 )
    {
      v10 = a1->m128i_i64[0];
      if ( (unsigned int)sub_8D2E30(a1->m128i_i64[0]) )
      {
        v15 = sub_724DC0(v10, v19, v11, v12, v13, v14);
        v16 = a1[9].m128i_i64[0];
        v20 = v15;
        if ( (unsigned int)sub_717520(v16, v15, 1) )
        {
          v21 = _mm_loadu_si128(a1);
          v17 = a1[1].m128i_i8[0];
          v22 = _mm_loadu_si128(a1 + 1);
          v23 = _mm_loadu_si128(a1 + 2);
          v24 = _mm_loadu_si128(a1 + 3);
          v25 = _mm_loadu_si128(a1 + 4);
          v26 = _mm_loadu_si128(a1 + 5);
          v27 = _mm_loadu_si128(a1 + 6);
          v28 = _mm_loadu_si128(a1 + 7);
          v29 = _mm_loadu_si128(a1 + 8);
          if ( v17 == 2 )
          {
            v30 = _mm_loadu_si128(a1 + 9);
            v31 = _mm_loadu_si128(a1 + 10);
            v32 = _mm_loadu_si128(a1 + 11);
            v33 = _mm_loadu_si128(a1 + 12);
            v34 = _mm_loadu_si128(a1 + 13);
            v35 = _mm_loadu_si128(a1 + 14);
            v36 = _mm_loadu_si128(a1 + 15);
            v37 = _mm_loadu_si128(a1 + 16);
            v38 = _mm_loadu_si128(a1 + 17);
            v39 = _mm_loadu_si128(a1 + 18);
            v40 = _mm_loadu_si128(a1 + 19);
            v41 = _mm_loadu_si128(a1 + 20);
            v42 = _mm_loadu_si128(a1 + 21);
          }
          else if ( v17 == 5 || v17 == 1 )
          {
            v30.m128i_i64[0] = a1[9].m128i_i64[0];
          }
          v18 = v20;
          if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
            *(_QWORD *)(v20 + 144) = a1[9].m128i_i64[0];
          sub_6E6A50(v18, (__int64)a1);
          sub_6E4BC0((__int64)a1, (__int64)&v21);
        }
        sub_724E30(&v20);
      }
    }
  }
  return sub_724E30(&v19);
}
