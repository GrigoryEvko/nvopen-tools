// Function: sub_7161E0
// Address: 0x7161e0
//
__int64 __fastcall sub_7161E0(const __m128i *a1, __int64 a2, __int64 a3, __m128i *a4, __int64 a5, __int64 a6)
{
  __int8 v9; // al
  __int64 *v10; // r15
  __m128i *v11; // rdi
  bool v12; // zf
  __int8 v13; // al
  _QWORD *v14; // rax
  unsigned int v15; // r12d
  _BOOL4 v17; // [rsp+Ch] [rbp-44h] BYREF
  __m128i *v18; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19; // [rsp+18h] [rbp-38h] BYREF

  v18 = (__m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  sub_72A510(a1, a4);
  v9 = a1[10].m128i_i8[13];
  if ( v9 )
  {
    if ( v9 == 12 )
    {
      v15 = 0;
      goto LABEL_14;
    }
    v10 = &v19;
    v11 = v18;
    v12 = a4[10].m128i_i8[13] == 6;
    v19 = 0;
    if ( !v12 )
      v10 = 0;
    if ( v9 == 1 )
    {
      *v18 = _mm_loadu_si128(a1);
      v11[1] = _mm_loadu_si128(a1 + 1);
      v11[2] = _mm_loadu_si128(a1 + 2);
      v11[3] = _mm_loadu_si128(a1 + 3);
      v11[4] = _mm_loadu_si128(a1 + 4);
      v11[5] = _mm_loadu_si128(a1 + 5);
      v11[6] = _mm_loadu_si128(a1 + 6);
      v11[7] = _mm_loadu_si128(a1 + 7);
      v11[8] = _mm_loadu_si128(a1 + 8);
      v11[9] = _mm_loadu_si128(a1 + 9);
      v11[10] = _mm_loadu_si128(a1 + 10);
      v11[11] = _mm_loadu_si128(a1 + 11);
      v11[12] = _mm_loadu_si128(a1 + 12);
    }
    else
    {
      if ( v9 != 6 )
        goto LABEL_18;
      sub_72BAF0(v18, a1[12].m128i_i64[0], unk_4F06A60);
    }
    sub_70CFA0((__int64)v18, a2, v10, &v17);
    v11 = v18;
    v13 = a4[10].m128i_i8[13];
    if ( v13 == 1 )
    {
      *a4 = _mm_loadu_si128(v18);
      a4[1] = _mm_loadu_si128(v11 + 1);
      a4[2] = _mm_loadu_si128(v11 + 2);
      a4[3] = _mm_loadu_si128(v11 + 3);
      a4[4] = _mm_loadu_si128(v11 + 4);
      a4[5] = _mm_loadu_si128(v11 + 5);
      a4[6] = _mm_loadu_si128(v11 + 6);
      a4[7] = _mm_loadu_si128(v11 + 7);
      a4[8] = _mm_loadu_si128(v11 + 8);
      a4[9] = _mm_loadu_si128(v11 + 9);
      a4[10] = _mm_loadu_si128(v11 + 10);
      a4[11] = _mm_loadu_si128(v11 + 11);
      a4[12] = _mm_loadu_si128(v11 + 12);
LABEL_11:
      a4[10].m128i_i8[8] |= 8u;
      v12 = v19 == 0;
      a4[8].m128i_i64[0] = a3;
      if ( !v12 )
      {
        v14 = (_QWORD *)sub_77F6E0(a4);
        *v14 = v19;
      }
      goto LABEL_13;
    }
    if ( v13 == 6 )
    {
      a4[12].m128i_i64[0] = sub_620FA0((__int64)v18, &v17);
      goto LABEL_11;
    }
LABEL_18:
    sub_721090(v11);
  }
LABEL_13:
  v15 = 1;
LABEL_14:
  sub_724E30(&v18);
  return v15;
}
