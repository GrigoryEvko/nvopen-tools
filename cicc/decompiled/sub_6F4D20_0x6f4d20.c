// Function: sub_6F4D20
// Address: 0x6f4d20
//
__int64 __fastcall sub_6F4D20(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int8 v19; // al
  __int64 v20; // rdi
  _DWORD *v21; // r15
  __int64 v22; // [rsp+8h] [rbp-1A8h] BYREF
  __m128i v23; // [rsp+10h] [rbp-1A0h] BYREF
  __m128i v24; // [rsp+20h] [rbp-190h] BYREF
  __m128i v25; // [rsp+30h] [rbp-180h]
  __m128i v26; // [rsp+40h] [rbp-170h]
  __m128i v27; // [rsp+50h] [rbp-160h]
  __m128i v28; // [rsp+60h] [rbp-150h]
  __m128i v29; // [rsp+70h] [rbp-140h]
  __m128i v30; // [rsp+80h] [rbp-130h]
  __m128i v31; // [rsp+90h] [rbp-120h]
  __m128i v32; // [rsp+A0h] [rbp-110h]
  __m128i v33; // [rsp+B0h] [rbp-100h]
  __m128i v34; // [rsp+C0h] [rbp-F0h]
  __m128i v35; // [rsp+D0h] [rbp-E0h]
  __m128i v36; // [rsp+E0h] [rbp-D0h]
  __m128i v37; // [rsp+F0h] [rbp-C0h]
  __m128i v38; // [rsp+100h] [rbp-B0h]
  __m128i v39; // [rsp+110h] [rbp-A0h]
  __m128i v40; // [rsp+120h] [rbp-90h]
  __m128i v41; // [rsp+130h] [rbp-80h]
  __m128i v42; // [rsp+140h] [rbp-70h]
  __m128i v43; // [rsp+150h] [rbp-60h]
  __m128i v44; // [rsp+160h] [rbp-50h]
  __m128i v45; // [rsp+170h] [rbp-40h]

  v6 = a3;
  v8 = a2;
  v9 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v10 = v6;
  v11 = a1[9].m128i_i64[0];
  v22 = v9;
  v23 = 0u;
  if ( (unsigned int)sub_7A30C0(v11, v6, 0, v9) )
  {
    v24 = _mm_loadu_si128(a1);
    v19 = a1[1].m128i_i8[0];
    v25 = _mm_loadu_si128(a1 + 1);
    v26 = _mm_loadu_si128(a1 + 2);
    v27 = _mm_loadu_si128(a1 + 3);
    v28 = _mm_loadu_si128(a1 + 4);
    v29 = _mm_loadu_si128(a1 + 5);
    v30 = _mm_loadu_si128(a1 + 6);
    v31 = _mm_loadu_si128(a1 + 7);
    v32 = _mm_loadu_si128(a1 + 8);
    if ( v19 == 2 )
    {
      v33 = _mm_loadu_si128(a1 + 9);
      v34 = _mm_loadu_si128(a1 + 10);
      v35 = _mm_loadu_si128(a1 + 11);
      v36 = _mm_loadu_si128(a1 + 12);
      v37 = _mm_loadu_si128(a1 + 13);
      v38 = _mm_loadu_si128(a1 + 14);
      v39 = _mm_loadu_si128(a1 + 15);
      v40 = _mm_loadu_si128(a1 + 16);
      v41 = _mm_loadu_si128(a1 + 17);
      v42 = _mm_loadu_si128(a1 + 18);
      v43 = _mm_loadu_si128(a1 + 19);
      v44 = _mm_loadu_si128(a1 + 20);
      v45 = _mm_loadu_si128(a1 + 21);
    }
    else if ( v19 == 5 || v19 == 1 )
    {
      v33.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    v20 = v22;
    if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
      *(_QWORD *)(v22 + 144) = 0;
    sub_6E6A50(v20, (__int64)a1);
    if ( v25.m128i_i8[1] == 1 && !sub_6ED0A0((__int64)&v24) && (unsigned int)sub_8D32E0(*(_QWORD *)(v22 + 128)) )
    {
      a1[1].m128i_i8[1] = v25.m128i_i8[1];
      a1->m128i_i64[0] = v24.m128i_i64[0];
    }
    v12 = 1;
    sub_6E4BC0((__int64)a1, (__int64)&v24);
  }
  else
  {
    v12 = 0;
    if ( (dword_4F04C44 != -1
       || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 6) & 0x12) != 0)
       || (*(_BYTE *)(v13 + 6) & 4) != 0 && (*(_BYTE *)(v13 + 12) & 0x10) == 0)
      && (unsigned int)sub_696840((__int64)a1) )
    {
      v12 = 1;
      sub_6F4B70(a1, v10, v15, v16, v17, v18);
    }
    else if ( v8 )
    {
      if ( (unsigned int)sub_6E5430() )
      {
        v21 = sub_67D9D0(0x1Cu, &a1[4].m128i_i32[1]);
        sub_67E370((__int64)v21, &v23);
        sub_685910((__int64)v21, (FILE *)&v23);
      }
      sub_6E6260(a1);
    }
    else
    {
      v12 = 0;
    }
  }
  sub_67E3D0(&v23);
  sub_724E30(&v22);
  return v12;
}
