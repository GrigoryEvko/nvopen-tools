// Function: sub_F038C0
// Address: 0xf038c0
//
unsigned __int64 __fastcall sub_F038C0(
        unsigned int *a1,
        __int64 a2,
        __int32 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // r15
  __m128i *v8; // rax
  __m128i *v9; // r12
  __int64 v10; // r14
  __int32 v11; // r13d
  __int64 v12; // rdx
  unsigned __int64 v13; // r14
  __int64 v14; // rcx
  __m128i *v15; // rsi
  unsigned __int64 v16; // rdx
  unsigned __int64 result; // rax
  __m128i *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // r13

  v6 = HIDWORD(a4);
  v8 = *(__m128i **)a1;
  v8->m128i_i64[0] = a2;
  v8->m128i_i32[2] = a3;
  v8->m128i_i32[3] = a4;
  v9 = (__m128i *)(*(_QWORD *)a1 + 16LL);
  v10 = *(_QWORD *)(**(_QWORD **)a1 + 8LL * *(unsigned int *)(*(_QWORD *)a1 + 12LL));
  v11 = (v10 & 0x3F) + 1;
  v12 = a1[2];
  v13 = v10 & 0xFFFFFFFFFFFFFFC0LL;
  v14 = 16 * v12;
  v15 = (__m128i *)(*(_QWORD *)a1 + 16 * v12);
  if ( v9 != v15 )
  {
    v16 = v12 + 1;
    result = *(_QWORD *)a1 + v14 - 16;
    if ( v16 <= a1[3]
      || (sub_C8D5F0((__int64)a1, a1 + 4, v16, 0x10u, a5, a6),
          v18 = *(__m128i **)a1,
          v19 = a1[2],
          v20 = 16 * v19,
          v9 = (__m128i *)(*(_QWORD *)a1 + 16LL),
          result = *(_QWORD *)a1 + 16 * v19 - 16,
          (v15 = (__m128i *)(16 * v19 + *(_QWORD *)a1)) != 0) )
    {
      *v15 = _mm_loadu_si128((const __m128i *)result);
      v18 = *(__m128i **)a1;
      v19 = a1[2];
      v20 = 16 * v19;
      result = *(_QWORD *)a1 + 16 * v19 - 16;
      if ( (__m128i *)result == v9 )
      {
LABEL_5:
        a1[2] = v19 + 1;
        v9->m128i_i64[0] = v13;
        v9->m128i_i32[2] = v11;
        v9->m128i_i32[3] = v6;
        return result;
      }
    }
    else if ( (__m128i *)result == v9 )
    {
      goto LABEL_5;
    }
    result = (unsigned __int64)memmove(&v18->m128i_i8[v20 - (result - (_QWORD)v9)], v9, result - (_QWORD)v9);
    LODWORD(v19) = a1[2];
    goto LABEL_5;
  }
  result = a1[3];
  v21 = v12 + 1;
  v22 = (v6 << 32) | v11 & 0x7F;
  if ( v21 > result )
  {
    result = sub_C8D5F0((__int64)a1, a1 + 4, v21, 0x10u, a5, a6);
    v15 = (__m128i *)(*(_QWORD *)a1 + 16LL * a1[2]);
  }
  v15->m128i_i64[0] = v13;
  v15->m128i_i64[1] = v22;
  ++a1[2];
  return result;
}
