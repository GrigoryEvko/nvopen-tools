// Function: sub_1636540
// Address: 0x1636540
//
__int64 __fastcall sub_1636540(const char *a1, const char *a2)
{
  const char *v2; // r13
  __m128i *v3; // rdx
  __int64 v4; // r12
  __m128i si128; // xmm0
  __m128i v6; // xmm0
  const char *(__fastcall *v7)(__int64, __int64); // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  const char *v11; // rsi
  size_t v12; // r13
  _BYTE *v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  size_t v18; // rdx

  v2 = a1;
  v3 = (__m128i *)*((_QWORD *)a2 + 3);
  v4 = (__int64)a2;
  if ( *((_QWORD *)a2 + 2) - (_QWORD)v3 <= 0x26u )
  {
    a1 = a2;
    a2 = "Pass::print not implemented for pass: '";
    v4 = sub_16E7EE0(a1, "Pass::print not implemented for pass: '", 39);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F560A0);
    v3[2].m128i_i32[0] = 1936941424;
    v3[2].m128i_i16[2] = 8250;
    *v3 = si128;
    v6 = _mm_load_si128((const __m128i *)&xmmword_3F560B0);
    v3[2].m128i_i8[6] = 39;
    v3[1] = v6;
    *((_QWORD *)a2 + 3) += 39LL;
  }
  v7 = *(const char *(__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 16LL);
  if ( v7 == sub_1635FB0 )
  {
    v8 = *((_QWORD *)v2 + 2);
    v9 = sub_163A1D0(a1, a2);
    v10 = sub_163A340(v9, v8);
    if ( !v10 )
    {
      v13 = *(_BYTE **)(v4 + 24);
      v12 = 43;
      v11 = "Unnamed pass: implement Pass::getPassName()";
      if ( *(_QWORD *)(v4 + 16) - (_QWORD)v13 > 0x2Au )
        goto LABEL_11;
      goto LABEL_13;
    }
    v11 = *(const char **)v10;
    v12 = *(_QWORD *)(v10 + 8);
  }
  else
  {
    v11 = (const char *)((__int64 (__fastcall *)(const char *))v7)(v2);
    v12 = v18;
  }
  v13 = *(_BYTE **)(v4 + 24);
  v14 = *(_QWORD *)(v4 + 16) - (_QWORD)v13;
  if ( v14 >= v12 )
  {
    if ( !v12 )
      goto LABEL_8;
LABEL_11:
    memcpy(v13, v11, v12);
    v13 = (_BYTE *)(v12 + *(_QWORD *)(v4 + 24));
    v16 = *(_QWORD *)(v4 + 16) - (_QWORD)v13;
    *(_QWORD *)(v4 + 24) = v13;
    if ( v16 > 2 )
      goto LABEL_9;
    return sub_16E7EE0(v4, "'!\n", 3);
  }
LABEL_13:
  v17 = sub_16E7EE0(v4, v11, v12);
  v13 = *(_BYTE **)(v17 + 24);
  v4 = v17;
  v14 = *(_QWORD *)(v17 + 16) - (_QWORD)v13;
LABEL_8:
  if ( v14 > 2 )
  {
LABEL_9:
    v13[2] = 10;
    *(_WORD *)v13 = 8487;
    *(_QWORD *)(v4 + 24) += 3LL;
    return 8487;
  }
  return sub_16E7EE0(v4, "'!\n", 3);
}
