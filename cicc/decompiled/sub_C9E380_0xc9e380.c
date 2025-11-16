// Function: sub_C9E380
// Address: 0xc9e380
//
__int64 __fastcall sub_C9E380(double *a1, __int64 a2, __int64 a3)
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
  double v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  __m128i *v18; // rdx
  __m128i v19; // xmm0
  __m128i *v20; // rdx
  __m128i si128; // xmm0
  double v22; // xmm4_8
  double v23; // xmm4_8
  double v24; // xmm1_8
  void *v25; // [rsp+10h] [rbp-40h] BYREF
  const char *v26; // [rsp+18h] [rbp-38h]
  double v27; // [rsp+20h] [rbp-30h]
  double v28; // [rsp+28h] [rbp-28h]

  v4 = *(double *)(a2 + 8);
  if ( v4 != 0.0 )
  {
    if ( v4 >= 0.0000001 )
    {
      v22 = a1[1];
      v26 = "  %7.4f (%5.1f%%)";
      v28 = v22;
      v25 = &unk_49DCB78;
      v27 = 100.0 * v22 / v4;
      sub_CB6620(a3, &v25);
    }
    else
    {
      v20 = *(__m128i **)(a3 + 32);
      if ( *(_QWORD *)(a3 + 24) - (_QWORD)v20 <= 0x11u )
      {
        sub_CB6200(a3, "        -----     ", 18);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
        v20[1].m128i_i16[0] = 8224;
        *v20 = si128;
        *(_QWORD *)(a3 + 32) += 18LL;
      }
    }
  }
  v5 = *(double *)(a2 + 16);
  if ( v5 != 0.0 )
  {
    if ( v5 >= 0.0000001 )
    {
      v23 = a1[2];
      v26 = "  %7.4f (%5.1f%%)";
      v28 = v23;
      v25 = &unk_49DCB78;
      v27 = 100.0 * v23 / v5;
      sub_CB6620(a3, &v25);
      v5 = *(double *)(a2 + 16);
    }
    else
    {
      v18 = *(__m128i **)(a3 + 32);
      if ( *(_QWORD *)(a3 + 24) - (_QWORD)v18 <= 0x11u )
      {
        sub_CB6200(a3, "        -----     ", 18);
      }
      else
      {
        v19 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
        v18[1].m128i_i16[0] = 8224;
        *v18 = v19;
        *(_QWORD *)(a3 + 32) += 18LL;
      }
      v5 = *(double *)(a2 + 16);
    }
  }
  v6 = v5 + *(double *)(a2 + 8);
  if ( v6 != 0.0 )
  {
    if ( v6 >= 0.0000001 )
    {
      v24 = a1[2] + a1[1];
      v26 = "  %7.4f (%5.1f%%)";
      v28 = v24;
      v25 = &unk_49DCB78;
      v27 = 100.0 * v24 / v6;
      sub_CB6620(a3, &v25);
    }
    else
    {
      v16 = *(__m128i **)(a3 + 32);
      if ( *(_QWORD *)(a3 + 24) - (_QWORD)v16 <= 0x11u )
      {
        sub_CB6200(a3, "        -----     ", 18);
      }
      else
      {
        v17 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
        v16[1].m128i_i16[0] = 8224;
        *v16 = v17;
        *(_QWORD *)(a3 + 32) += 18LL;
      }
    }
  }
  v7 = *(double *)a2;
  if ( *(double *)a2 >= 0.0000001 )
  {
    v13 = *a1;
    v26 = "  %7.4f (%5.1f%%)";
    v28 = v13;
    v25 = &unk_49DCB78;
    v27 = 100.0 * v13 / v7;
    sub_CB6620(a3, &v25);
    v10 = *(_WORD **)(a3 + 32);
  }
  else
  {
    v8 = *(__m128i **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v8 > 0x11u )
    {
      v9 = _mm_load_si128((const __m128i *)&xmmword_3F67A30);
      v8[1].m128i_i16[0] = 8224;
      *v8 = v9;
      v10 = (_WORD *)(*(_QWORD *)(a3 + 32) + 18LL);
      v11 = *(_QWORD *)(a3 + 24);
      *(_QWORD *)(a3 + 32) = v10;
      if ( (unsigned __int64)(v11 - (_QWORD)v10) <= 1 )
        goto LABEL_7;
      goto LABEL_12;
    }
    sub_CB6200(a3, "        -----     ", 18);
    v10 = *(_WORD **)(a3 + 32);
  }
  if ( *(_QWORD *)(a3 + 24) - (_QWORD)v10 <= 1u )
  {
LABEL_7:
    result = sub_CB6200(a3, "  ", 2);
    if ( !*(_QWORD *)(a2 + 24) )
      goto LABEL_8;
    goto LABEL_13;
  }
LABEL_12:
  result = 8224;
  *v10 = 8224;
  *(_QWORD *)(a3 + 32) += 2LL;
  if ( !*(_QWORD *)(a2 + 24) )
  {
LABEL_8:
    if ( !*(_QWORD *)(a2 + 32) )
      return result;
LABEL_14:
    v15 = a1[4];
    v26 = "%9ld  ";
    v25 = &unk_49DBEF0;
    v27 = v15;
    return sub_CB6620(a3, &v25);
  }
LABEL_13:
  v14 = a1[3];
  v26 = "%9ld  ";
  v25 = &unk_49DBEF0;
  v27 = v14;
  result = sub_CB6620(a3, &v25);
  if ( *(_QWORD *)(a2 + 32) )
    goto LABEL_14;
  return result;
}
