// Function: sub_16D79D0
// Address: 0x16d79d0
//
__int64 __fastcall sub_16D79D0(double *a1, __int64 a2, __int64 a3)
{
  double v4; // xmm0_8
  double v5; // xmm0_8
  double v6; // xmm0_8
  double v7; // xmm0_8
  __m128i *v8; // rdx
  __m128i v9; // xmm0
  _WORD *v10; // rdx
  __int64 v11; // rax
  __int64 result; // rax
  double v13; // xmm2_8
  double v14; // rax
  __m128i *v15; // rdx
  __m128i v16; // xmm0
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  __m128i *v19; // rdx
  __m128i si128; // xmm0
  double v21; // xmm4_8
  double v22; // xmm4_8
  double v23; // xmm1_8
  void *v24; // [rsp+10h] [rbp-40h] BYREF
  const char *v25; // [rsp+18h] [rbp-38h]
  double v26; // [rsp+20h] [rbp-30h]
  double v27; // [rsp+28h] [rbp-28h]

  v4 = *(double *)(a2 + 8);
  if ( v4 != 0.0 )
  {
    if ( v4 >= 0.0000001 )
    {
      v21 = a1[1];
      v25 = "  %7.4f (%5.1f%%)";
      v27 = v21;
      v24 = &unk_49EF688;
      v26 = 100.0 * v21 / v4;
      sub_16E8450(a3, &v24);
    }
    else
    {
      v19 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v19 <= 0x11u )
      {
        sub_16E7EE0(a3, "        -----     ", 18);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
        v19[1].m128i_i16[0] = 8224;
        *v19 = si128;
        *(_QWORD *)(a3 + 24) += 18LL;
      }
    }
  }
  v5 = *(double *)(a2 + 16);
  if ( v5 != 0.0 )
  {
    if ( v5 >= 0.0000001 )
    {
      v22 = a1[2];
      v25 = "  %7.4f (%5.1f%%)";
      v27 = v22;
      v24 = &unk_49EF688;
      v26 = 100.0 * v22 / v5;
      sub_16E8450(a3, &v24);
      v5 = *(double *)(a2 + 16);
    }
    else
    {
      v17 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v17 <= 0x11u )
      {
        sub_16E7EE0(a3, "        -----     ", 18);
      }
      else
      {
        v18 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
        v17[1].m128i_i16[0] = 8224;
        *v17 = v18;
        *(_QWORD *)(a3 + 24) += 18LL;
      }
      v5 = *(double *)(a2 + 16);
    }
  }
  v6 = v5 + *(double *)(a2 + 8);
  if ( v6 != 0.0 )
  {
    if ( v6 >= 0.0000001 )
    {
      v23 = a1[2] + a1[1];
      v25 = "  %7.4f (%5.1f%%)";
      v27 = v23;
      v24 = &unk_49EF688;
      v26 = 100.0 * v23 / v6;
      sub_16E8450(a3, &v24);
    }
    else
    {
      v15 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v15 <= 0x11u )
      {
        sub_16E7EE0(a3, "        -----     ", 18);
      }
      else
      {
        v16 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
        v15[1].m128i_i16[0] = 8224;
        *v15 = v16;
        *(_QWORD *)(a3 + 24) += 18LL;
      }
    }
  }
  v7 = *(double *)a2;
  if ( *(double *)a2 >= 0.0000001 )
  {
    v13 = *a1;
    v25 = "  %7.4f (%5.1f%%)";
    v27 = v13;
    v24 = &unk_49EF688;
    v26 = 100.0 * v13 / v7;
    sub_16E8450(a3, &v24);
    v10 = *(_WORD **)(a3 + 24);
  }
  else
  {
    v8 = *(__m128i **)(a3 + 24);
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v8 > 0x11u )
    {
      v9 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
      v8[1].m128i_i16[0] = 8224;
      *v8 = v9;
      v10 = (_WORD *)(*(_QWORD *)(a3 + 24) + 18LL);
      v11 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)(a3 + 24) = v10;
      if ( (unsigned __int64)(v11 - (_QWORD)v10) <= 1 )
        goto LABEL_7;
      goto LABEL_11;
    }
    sub_16E7EE0(a3, "        -----     ", 18);
    v10 = *(_WORD **)(a3 + 24);
  }
  if ( *(_QWORD *)(a3 + 16) - (_QWORD)v10 <= 1u )
  {
LABEL_7:
    result = sub_16E7EE0(a3, "  ", 2);
    if ( !*(_QWORD *)(a2 + 24) )
      return result;
LABEL_12:
    v14 = a1[3];
    v25 = "%9ld  ";
    v24 = &unk_49EEAD0;
    v26 = v14;
    return sub_16E8450(a3, &v24);
  }
LABEL_11:
  result = 8224;
  *v10 = 8224;
  *(_QWORD *)(a3 + 24) += 2LL;
  if ( *(_QWORD *)(a2 + 24) )
    goto LABEL_12;
  return result;
}
