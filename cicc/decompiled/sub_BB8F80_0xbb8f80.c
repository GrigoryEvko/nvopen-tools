// Function: sub_BB8F80
// Address: 0xbb8f80
//
__int64 __fastcall sub_BB8F80(const char *a1, const char *a2)
{
  const char *v2; // r13
  __m128i *v3; // rdx
  __int64 v4; // r12
  __m128i si128; // xmm0
  __m128i v6; // xmm0
  const char *(__fastcall *v7)(__int64, __int64); // rax
  pthread_rwlock_t *v8; // rax
  __int64 v9; // rax
  const char *v10; // rsi
  size_t v11; // r13
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  size_t v17; // rdx

  v2 = a1;
  v3 = (__m128i *)*((_QWORD *)a2 + 4);
  v4 = (__int64)a2;
  if ( *((_QWORD *)a2 + 3) - (_QWORD)v3 <= 0x26u )
  {
    a1 = a2;
    a2 = "Pass::print not implemented for pass: '";
    v4 = sub_CB6200(a1, "Pass::print not implemented for pass: '", 39);
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
    *((_QWORD *)a2 + 4) += 39LL;
  }
  v7 = *(const char *(__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 16LL);
  if ( v7 == sub_BB8680 )
  {
    v8 = (pthread_rwlock_t *)sub_BC2B00(a1, a2);
    v9 = sub_BC2C30(v8);
    if ( !v9 )
    {
      v12 = *(_BYTE **)(v4 + 32);
      v11 = 43;
      v10 = "Unnamed pass: implement Pass::getPassName()";
      if ( *(_QWORD *)(v4 + 24) - (_QWORD)v12 > 0x2Au )
        goto LABEL_11;
      goto LABEL_13;
    }
    v10 = *(const char **)v9;
    v11 = *(_QWORD *)(v9 + 8);
  }
  else
  {
    v10 = (const char *)((__int64 (__fastcall *)(const char *))v7)(v2);
    v11 = v17;
  }
  v12 = *(_BYTE **)(v4 + 32);
  v13 = *(_QWORD *)(v4 + 24) - (_QWORD)v12;
  if ( v13 >= v11 )
  {
    if ( !v11 )
      goto LABEL_8;
LABEL_11:
    memcpy(v12, v10, v11);
    v12 = (_BYTE *)(v11 + *(_QWORD *)(v4 + 32));
    v15 = *(_QWORD *)(v4 + 24) - (_QWORD)v12;
    *(_QWORD *)(v4 + 32) = v12;
    if ( v15 > 2 )
      goto LABEL_9;
    return sub_CB6200(v4, "'!\n", 3);
  }
LABEL_13:
  v16 = sub_CB6200(v4, v10, v11);
  v12 = *(_BYTE **)(v16 + 32);
  v4 = v16;
  v13 = *(_QWORD *)(v16 + 24) - (_QWORD)v12;
LABEL_8:
  if ( v13 > 2 )
  {
LABEL_9:
    v12[2] = 10;
    *(_WORD *)v12 = 8487;
    *(_QWORD *)(v4 + 32) += 3LL;
    return 8487;
  }
  return sub_CB6200(v4, "'!\n", 3);
}
