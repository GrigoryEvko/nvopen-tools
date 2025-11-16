// Function: ctor_452_0
// Address: 0x541c20
//
int ctor_452_0()
{
  __m128i *v0; // rax
  __m128i v1; // xmm1
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v17; // [rsp+8h] [rbp-1C8h]
  __int64 v18; // [rsp+8h] [rbp-1C8h]
  int v19; // [rsp+10h] [rbp-1C0h] BYREF
  int v20; // [rsp+14h] [rbp-1BCh] BYREF
  int *v21; // [rsp+18h] [rbp-1B8h] BYREF
  _BYTE *v22; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-1A8h]
  _BYTE v24[160]; // [rsp+30h] [rbp-1A0h] BYREF
  __m128i v25; // [rsp+D0h] [rbp-100h] BYREF
  __m128i v26; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v27; // [rsp+F0h] [rbp-E0h] BYREF
  __m128i v28; // [rsp+100h] [rbp-D0h] BYREF
  __m128i v29; // [rsp+110h] [rbp-C0h] BYREF
  __m128i v30; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v31; // [rsp+130h] [rbp-A0h] BYREF
  __m128i v32; // [rsp+140h] [rbp-90h] BYREF
  __m128i v33; // [rsp+150h] [rbp-80h] BYREF
  __m128i v34; // [rsp+160h] [rbp-70h] BYREF
  __m128i v35; // [rsp+170h] [rbp-60h] BYREF
  __m128i v36; // [rsp+180h] [rbp-50h] BYREF
  __int64 v37; // [rsp+190h] [rbp-40h]

  v25.m128i_i64[0] = (__int64)"never";
  v26.m128i_i64[1] = (__int64)"never replace exit value";
  v27.m128i_i64[1] = (__int64)"cheap";
  v29.m128i_i64[0] = (__int64)"only replace exit value when the cost is cheap";
  v30.m128i_i64[0] = (__int64)"unusedindvarinloop";
  v31.m128i_i64[1] = (__int64)"only replace exit value when it is an unused induction variable in the loop and has cheap "
                              "replacement cost";
  v32.m128i_i64[1] = (__int64)"noharduse";
  v34.m128i_i64[0] = (__int64)"only replace exit values when loop def likely dead";
  v35.m128i_i64[0] = (__int64)"always";
  v36.m128i_i64[1] = (__int64)"always replace exit value whenever possible";
  v25.m128i_i64[1] = 5;
  v26.m128i_i32[0] = 0;
  v27.m128i_i64[0] = 24;
  v28.m128i_i64[0] = 5;
  v28.m128i_i32[2] = 1;
  v29.m128i_i64[1] = 46;
  v30.m128i_i64[1] = 18;
  v31.m128i_i32[0] = 3;
  v32.m128i_i64[0] = 106;
  v33.m128i_i64[0] = 9;
  v33.m128i_i32[2] = 2;
  v34.m128i_i64[1] = 50;
  v35.m128i_i64[1] = 6;
  v36.m128i_i32[0] = 4;
  v37 = 43;
  v22 = v24;
  v23 = 0x400000000LL;
  sub_C8D5F0(&v22, v24, 5, 40);
  v0 = (__m128i *)&v22[40 * (unsigned int)v23];
  *v0 = _mm_loadu_si128(&v25);
  v1 = _mm_loadu_si128(&v26);
  LODWORD(v23) = v23 + 5;
  v0[1] = v1;
  v0[2] = _mm_loadu_si128(&v27);
  v0[3] = _mm_loadu_si128(&v28);
  v0[4] = _mm_loadu_si128(&v29);
  v0[5] = _mm_loadu_si128(&v30);
  v0[6] = _mm_loadu_si128(&v31);
  v0[7] = _mm_loadu_si128(&v32);
  v0[8] = _mm_loadu_si128(&v33);
  v0[9] = _mm_loadu_si128(&v34);
  v0[10] = _mm_loadu_si128(&v35);
  v0[11] = _mm_loadu_si128(&v36);
  v0[12].m128i_i64[0] = v37;
  v25.m128i_i64[0] = (__int64)"Choose the strategy to replace exit value in IndVarSimplify";
  v25.m128i_i64[1] = 59;
  v20 = 1;
  v21 = &v20;
  v19 = 1;
  sub_27CA330(&unk_4FFCA60, "replexitval", &v19, &v21, &v25, &v22);
  if ( v22 != v24 )
    _libc_free(v22, "replexitval");
  __cxa_atexit(sub_27C0630, &unk_4FFCA60, &qword_4A427C0);
  qword_4FFC980 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFC9D0 = 0x100000000LL;
  dword_4FFC98C &= 0x8000u;
  word_4FFC990 = 0;
  qword_4FFC998 = 0;
  qword_4FFC9A0 = 0;
  dword_4FFC988 = v2;
  qword_4FFC9A8 = 0;
  qword_4FFC9B0 = 0;
  qword_4FFC9B8 = 0;
  qword_4FFC9C0 = 0;
  qword_4FFC9C8 = (__int64)&unk_4FFC9D8;
  qword_4FFC9E0 = 0;
  qword_4FFC9E8 = (__int64)&unk_4FFCA00;
  qword_4FFC9F0 = 1;
  dword_4FFC9F8 = 0;
  byte_4FFC9FC = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_4FFC9D0;
  v5 = (unsigned int)qword_4FFC9D0 + 1LL;
  if ( v5 > HIDWORD(qword_4FFC9D0) )
  {
    sub_C8D5F0((char *)&unk_4FFC9D8 - 16, &unk_4FFC9D8, v5, 8);
    v4 = (unsigned int)qword_4FFC9D0;
  }
  *(_QWORD *)(qword_4FFC9C8 + 8 * v4) = v3;
  qword_4FFCA10 = (__int64)&unk_49D9748;
  qword_4FFC980 = (__int64)&unk_49DC090;
  qword_4FFCA20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFC9D0) = qword_4FFC9D0 + 1;
  qword_4FFCA40 = (__int64)nullsub_23;
  qword_4FFCA08 = 0;
  qword_4FFCA38 = (__int64)sub_984030;
  qword_4FFCA18 = 0;
  sub_C53080(&qword_4FFC980, "indvars-post-increment-ranges", 29);
  LOWORD(qword_4FFCA18) = 257;
  LOBYTE(qword_4FFCA08) = 1;
  qword_4FFC9B0 = 61;
  LOBYTE(dword_4FFC98C) = dword_4FFC98C & 0x9F | 0x20;
  qword_4FFC9A8 = (__int64)"Use post increment control-dependent ranges in IndVarSimplify";
  sub_C53130(&qword_4FFC980);
  __cxa_atexit(sub_984900, &qword_4FFC980, &qword_4A427C0);
  qword_4FFC8A0 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFC8F0 = 0x100000000LL;
  dword_4FFC8AC &= 0x8000u;
  qword_4FFC8E8 = (__int64)&unk_4FFC8F8;
  word_4FFC8B0 = 0;
  qword_4FFC8B8 = 0;
  dword_4FFC8A8 = v6;
  qword_4FFC8C0 = 0;
  qword_4FFC8C8 = 0;
  qword_4FFC8D0 = 0;
  qword_4FFC8D8 = 0;
  qword_4FFC8E0 = 0;
  qword_4FFC900 = 0;
  qword_4FFC908 = (__int64)&unk_4FFC920;
  qword_4FFC910 = 1;
  dword_4FFC918 = 0;
  byte_4FFC91C = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FFC8F0;
  if ( (unsigned __int64)(unsigned int)qword_4FFC8F0 + 1 > HIDWORD(qword_4FFC8F0) )
  {
    v17 = v7;
    sub_C8D5F0((char *)&unk_4FFC8F8 - 16, &unk_4FFC8F8, (unsigned int)qword_4FFC8F0 + 1LL, 8);
    v8 = (unsigned int)qword_4FFC8F0;
    v7 = v17;
  }
  *(_QWORD *)(qword_4FFC8E8 + 8 * v8) = v7;
  qword_4FFC930 = (__int64)&unk_49D9748;
  qword_4FFC8A0 = (__int64)&unk_49DC090;
  qword_4FFC940 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFC8F0) = qword_4FFC8F0 + 1;
  qword_4FFC960 = (__int64)nullsub_23;
  qword_4FFC928 = 0;
  qword_4FFC958 = (__int64)sub_984030;
  qword_4FFC938 = 0;
  sub_C53080(&qword_4FFC8A0, "disable-lftr", 12);
  LOWORD(qword_4FFC938) = 256;
  LOBYTE(qword_4FFC928) = 0;
  qword_4FFC8D0 = 49;
  LOBYTE(dword_4FFC8AC) = dword_4FFC8AC & 0x9F | 0x20;
  qword_4FFC8C8 = (__int64)"Disable Linear Function Test Replace optimization";
  sub_C53130(&qword_4FFC8A0);
  __cxa_atexit(sub_984900, &qword_4FFC8A0, &qword_4A427C0);
  qword_4FFC7C0 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFC810 = 0x100000000LL;
  dword_4FFC7CC &= 0x8000u;
  qword_4FFC808 = (__int64)&unk_4FFC818;
  word_4FFC7D0 = 0;
  qword_4FFC7D8 = 0;
  dword_4FFC7C8 = v9;
  qword_4FFC7E0 = 0;
  qword_4FFC7E8 = 0;
  qword_4FFC7F0 = 0;
  qword_4FFC7F8 = 0;
  qword_4FFC800 = 0;
  qword_4FFC820 = 0;
  qword_4FFC828 = (__int64)&unk_4FFC840;
  qword_4FFC830 = 1;
  dword_4FFC838 = 0;
  byte_4FFC83C = 1;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_4FFC810;
  if ( (unsigned __int64)(unsigned int)qword_4FFC810 + 1 > HIDWORD(qword_4FFC810) )
  {
    v18 = v10;
    sub_C8D5F0((char *)&unk_4FFC818 - 16, &unk_4FFC818, (unsigned int)qword_4FFC810 + 1LL, 8);
    v11 = (unsigned int)qword_4FFC810;
    v10 = v18;
  }
  *(_QWORD *)(qword_4FFC808 + 8 * v11) = v10;
  qword_4FFC850 = (__int64)&unk_49D9748;
  qword_4FFC7C0 = (__int64)&unk_49DC090;
  qword_4FFC860 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFC810) = qword_4FFC810 + 1;
  qword_4FFC880 = (__int64)nullsub_23;
  qword_4FFC848 = 0;
  qword_4FFC878 = (__int64)sub_984030;
  qword_4FFC858 = 0;
  sub_C53080(&qword_4FFC7C0, "indvars-predicate-loops", 23);
  LOWORD(qword_4FFC858) = 257;
  LOBYTE(qword_4FFC848) = 1;
  qword_4FFC7F0 = 39;
  LOBYTE(dword_4FFC7CC) = dword_4FFC7CC & 0x9F | 0x20;
  qword_4FFC7E8 = (__int64)"Predicate conditions in read only loops";
  sub_C53130(&qword_4FFC7C0);
  __cxa_atexit(sub_984900, &qword_4FFC7C0, &qword_4A427C0);
  qword_4FFC6E0 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFC730 = 0x100000000LL;
  dword_4FFC6EC &= 0x8000u;
  word_4FFC6F0 = 0;
  qword_4FFC728 = (__int64)&unk_4FFC738;
  qword_4FFC6F8 = 0;
  dword_4FFC6E8 = v12;
  qword_4FFC700 = 0;
  qword_4FFC708 = 0;
  qword_4FFC710 = 0;
  qword_4FFC718 = 0;
  qword_4FFC720 = 0;
  qword_4FFC740 = 0;
  qword_4FFC748 = (__int64)&unk_4FFC760;
  qword_4FFC750 = 1;
  dword_4FFC758 = 0;
  byte_4FFC75C = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4FFC730;
  v15 = (unsigned int)qword_4FFC730 + 1LL;
  if ( v15 > HIDWORD(qword_4FFC730) )
  {
    sub_C8D5F0((char *)&unk_4FFC738 - 16, &unk_4FFC738, v15, 8);
    v14 = (unsigned int)qword_4FFC730;
  }
  *(_QWORD *)(qword_4FFC728 + 8 * v14) = v13;
  qword_4FFC770 = (__int64)&unk_49D9748;
  qword_4FFC6E0 = (__int64)&unk_49DC090;
  qword_4FFC780 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFC730) = qword_4FFC730 + 1;
  qword_4FFC7A0 = (__int64)nullsub_23;
  qword_4FFC768 = 0;
  qword_4FFC798 = (__int64)sub_984030;
  qword_4FFC778 = 0;
  sub_C53080(&qword_4FFC6E0, "indvars-widen-indvars", 21);
  LOBYTE(qword_4FFC768) = 1;
  qword_4FFC710 = 45;
  LOBYTE(dword_4FFC6EC) = dword_4FFC6EC & 0x9F | 0x20;
  LOWORD(qword_4FFC778) = 257;
  qword_4FFC708 = (__int64)"Allow widening of indvars to eliminate s/zext";
  sub_C53130(&qword_4FFC6E0);
  return __cxa_atexit(sub_984900, &qword_4FFC6E0, &qword_4A427C0);
}
