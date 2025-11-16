// Function: ctor_620_0
// Address: 0x58b6c0
//
int ctor_620_0()
{
  __m128i *v0; // rax
  int v2; // [rsp+4h] [rbp-1BCh] BYREF
  int *v3; // [rsp+8h] [rbp-1B8h] BYREF
  const char *v4; // [rsp+10h] [rbp-1B0h] BYREF
  __int64 v5; // [rsp+18h] [rbp-1A8h]
  _BYTE v6[160]; // [rsp+20h] [rbp-1A0h] BYREF
  __m128i v7; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v8; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v9; // [rsp+E0h] [rbp-E0h] BYREF
  __m128i v10; // [rsp+F0h] [rbp-D0h] BYREF
  __m128i v11; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v12; // [rsp+110h] [rbp-B0h] BYREF
  __m128i v13; // [rsp+120h] [rbp-A0h] BYREF
  __m128i v14; // [rsp+130h] [rbp-90h] BYREF
  __m128i v15; // [rsp+140h] [rbp-80h] BYREF
  __m128i v16; // [rsp+150h] [rbp-70h] BYREF
  __m128i v17; // [rsp+160h] [rbp-60h] BYREF
  __m128i v18; // [rsp+170h] [rbp-50h] BYREF
  __int64 v19; // [rsp+180h] [rbp-40h]

  v7.m128i_i64[0] = (__int64)"throughput";
  v8.m128i_i64[1] = (__int64)"Reciprocal throughput";
  v9.m128i_i64[1] = (__int64)"latency";
  v11.m128i_i64[0] = (__int64)"Instruction latency";
  v12.m128i_i64[0] = (__int64)"code-size";
  v13.m128i_i64[1] = (__int64)"Code size";
  v14.m128i_i64[1] = (__int64)"size-latency";
  v16.m128i_i64[0] = (__int64)"Code size and latency";
  v17.m128i_i64[0] = (__int64)"all";
  v18.m128i_i64[1] = (__int64)"Print all cost kinds";
  v7.m128i_i64[1] = 10;
  v8.m128i_i32[0] = 0;
  v9.m128i_i64[0] = 21;
  v10.m128i_i64[0] = 7;
  v10.m128i_i32[2] = 1;
  v11.m128i_i64[1] = 19;
  v12.m128i_i64[1] = 9;
  v13.m128i_i32[0] = 2;
  v14.m128i_i64[0] = 9;
  v15.m128i_i64[0] = 12;
  v15.m128i_i32[2] = 3;
  v16.m128i_i64[1] = 21;
  v17.m128i_i64[1] = 3;
  v18.m128i_i32[0] = 4;
  v19 = 20;
  v4 = v6;
  v5 = 0x400000000LL;
  sub_C8D5F0(&v4, v6, 5, 40);
  v0 = (__m128i *)&v4[40 * (unsigned int)v5];
  *v0 = _mm_loadu_si128(&v7);
  v0[1] = _mm_loadu_si128(&v8);
  v0[2] = _mm_loadu_si128(&v9);
  v0[3] = _mm_loadu_si128(&v10);
  v0[4] = _mm_loadu_si128(&v11);
  v0[5] = _mm_loadu_si128(&v12);
  v0[6] = _mm_loadu_si128(&v13);
  v0[7] = _mm_loadu_si128(&v14);
  v0[8] = _mm_loadu_si128(&v15);
  v0[9] = _mm_loadu_si128(&v16);
  v0[10] = _mm_loadu_si128(&v17);
  v0[11] = _mm_loadu_si128(&v18);
  v0[12].m128i_i64[0] = v19;
  LODWORD(v5) = v5 + 5;
  v2 = 0;
  v3 = &v2;
  v7.m128i_i64[0] = (__int64)"Target cost kind";
  v7.m128i_i64[1] = 16;
  sub_30AFA40(&unk_502E780, "cost-kind", &v7, &v3, &v4);
  if ( v4 != v6 )
    _libc_free(v4, "cost-kind");
  __cxa_atexit(sub_30AF090, &unk_502E780, &qword_4A427C0);
  v7.m128i_i64[0] = (__int64)&v8;
  v8.m128i_i64[0] = (__int64)"instruction-cost";
  v9.m128i_i64[1] = (__int64)"Use TargetTransformInfo::getInstructionCost";
  v10.m128i_i64[1] = (__int64)"intrinsic-cost";
  v12.m128i_i64[0] = (__int64)"Use TargetTransformInfo::getIntrinsicInstrCost";
  v13.m128i_i64[0] = (__int64)"type-based-intrinsic-cost";
  v14.m128i_i64[1] = (__int64)"Calculate the intrinsic cost based only on argument types";
  v7.m128i_i64[1] = 0x400000003LL;
  v8.m128i_i64[1] = 16;
  v9.m128i_i32[0] = 0;
  v10.m128i_i64[0] = 43;
  v11.m128i_i64[0] = 14;
  v11.m128i_i32[2] = 1;
  v12.m128i_i64[1] = 46;
  v13.m128i_i64[1] = 25;
  v14.m128i_i32[0] = 2;
  v15.m128i_i64[0] = 57;
  v2 = 0;
  v3 = &v2;
  v4 = "Costing strategy for intrinsic instructions";
  v5 = 43;
  sub_30AFE70(&unk_502E520, "intrinsic-cost-strategy", &v4, &v3, &v7);
  if ( (__m128i *)v7.m128i_i64[0] != &v8 )
    _libc_free(v7.m128i_i64[0], "intrinsic-cost-strategy");
  return __cxa_atexit(sub_30AF000, &unk_502E520, &qword_4A427C0);
}
