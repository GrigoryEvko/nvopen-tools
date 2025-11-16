// Function: sub_305EA20
// Address: 0x305ea20
//
void __fastcall sub_305EA20(__int64 a1)
{
  _QWORD *v1; // rax
  __m128i *v2; // rax
  __int64 v3; // rax
  __m128i si128; // xmm0
  __int64 v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // rax
  __m128i v8; // xmm0
  unsigned __int64 v9; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *m128i_i64; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v11; // [rsp+18h] [rbp-38h]
  _QWORD v12[6]; // [rsp+20h] [rbp-30h] BYREF

  v1 = (_QWORD *)sub_3096A30();
  sub_2FF0E80(a1, v1, 0);
  if ( sub_2FF12A0(a1, &unk_5026410, 0) )
  {
    m128i_i64 = v12;
    v9 = 32;
    v2 = (__m128i *)sub_22409D0((__int64)&m128i_i64, &v9, 0);
    m128i_i64 = v2->m128i_i64;
    v12[0] = v9;
    *v2 = _mm_load_si128((const __m128i *)&xmmword_44C9F50);
    v2[1] = _mm_load_si128((const __m128i *)&xmmword_44C9F60);
    v11 = v9;
    *((_BYTE *)m128i_i64 + v9) = 0;
    sub_2FF0D00(a1, (__int64)&m128i_i64);
    if ( m128i_i64 != v12 )
      j_j___libc_free_0((unsigned __int64)m128i_i64);
  }
  sub_2FF12A0(a1, &unk_5022610, 0);
  sub_2FF12A0(a1, &unk_5025D0C, 1u);
  sub_2FF12A0(a1, &unk_501EB24, 0);
  sub_2FF12A0(a1, &unk_501CF4C, 0);
  m128i_i64 = v12;
  v9 = 22;
  v3 = sub_22409D0((__int64)&m128i_i64, &v9, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_44C9F70);
  m128i_i64 = (_QWORD *)v3;
  v12[0] = v9;
  *(_DWORD *)(v3 + 16) = 1634738245;
  *(_WORD *)(v3 + 20) = 29555;
  *(__m128i *)v3 = si128;
  v11 = v9;
  *((_BYTE *)m128i_i64 + v9) = 0;
  sub_2FF0D00(a1, (__int64)&m128i_i64);
  if ( m128i_i64 != v12 )
    j_j___libc_free_0((unsigned __int64)m128i_i64);
  sub_2FF12A0(a1, &unk_50201E8, 0);
  sub_2FF12A0(a1, &unk_501F54C, 0);
  if ( (_BYTE)qword_502C9C8 )
    sub_2FF12A0(a1, &unk_5021D2C, 0);
  m128i_i64 = v12;
  v9 = 42;
  v5 = sub_22409D0((__int64)&m128i_i64, &v9, 0);
  m128i_i64 = (_QWORD *)v5;
  v12[0] = v9;
  *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_44C9F80);
  v6 = _mm_load_si128((const __m128i *)&xmmword_44C9F90);
  qmemcpy((void *)(v5 + 32), "ing passes", 10);
  *(__m128i *)(v5 + 16) = v6;
  v11 = v9;
  *((_BYTE *)m128i_i64 + v9) = 0;
  sub_2FF0D00(a1, (__int64)&m128i_i64);
  if ( m128i_i64 != v12 )
    j_j___libc_free_0((unsigned __int64)m128i_i64);
  sub_2FF12A0(a1, &unk_50226F4, 0);
  m128i_i64 = v12;
  v9 = 40;
  v7 = sub_22409D0((__int64)&m128i_i64, &v9, 0);
  m128i_i64 = (_QWORD *)v7;
  v12[0] = v9;
  *(__m128i *)v7 = _mm_load_si128((const __m128i *)&xmmword_44C9FA0);
  v8 = _mm_load_si128((const __m128i *)&xmmword_44C9FB0);
  *(_QWORD *)(v7 + 32) = 0x73736170206E6F69LL;
  *(__m128i *)(v7 + 16) = v8;
  v11 = v9;
  *((_BYTE *)m128i_i64 + v9) = 0;
  sub_2FF0D00(a1, (__int64)&m128i_i64);
  if ( m128i_i64 != v12 )
    j_j___libc_free_0((unsigned __int64)m128i_i64);
}
