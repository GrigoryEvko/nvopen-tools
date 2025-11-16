// Function: ctor_032
// Address: 0x48aa00
//
int ctor_032()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __m128i v12; // xmm0
  void (__fastcall *v13)(__m128i *, __m128i *, __int64); // rax
  _BYTE v15[16]; // [rsp+0h] [rbp-A0h] BYREF
  void (__fastcall *v16)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-90h]
  __int64 (__fastcall *v17)(); // [rsp+18h] [rbp-88h]
  __m128i v18; // [rsp+20h] [rbp-80h] BYREF
  void (__fastcall *v19)(__m128i *, __m128i *, __int64); // [rsp+30h] [rbp-70h]
  __int64 v20; // [rsp+38h] [rbp-68h]
  __m128i v21; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v22)(__m128i *, __m128i *, __int64); // [rsp+50h] [rbp-50h]
  __int64 (__fastcall *v23)(); // [rsp+58h] [rbp-48h]
  char v24; // [rsp+60h] [rbp-40h]
  char v25; // [rsp+61h] [rbp-3Fh]

  qword_4F82620 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8269C = 1;
  qword_4F82670 = 0x100000000LL;
  dword_4F8262C &= 0x8000u;
  qword_4F82638 = 0;
  qword_4F82640 = 0;
  qword_4F82648 = 0;
  dword_4F82628 = v0;
  word_4F82630 = 0;
  qword_4F82650 = 0;
  qword_4F82658 = 0;
  qword_4F82660 = 0;
  qword_4F82668 = (__int64)&unk_4F82678;
  qword_4F82680 = 0;
  qword_4F82688 = (__int64)&unk_4F826A0;
  qword_4F82690 = 1;
  dword_4F82698 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F82670;
  v3 = (unsigned int)qword_4F82670 + 1LL;
  if ( v3 > HIDWORD(qword_4F82670) )
  {
    sub_C8D5F0((char *)&unk_4F82678 - 16, &unk_4F82678, v3, 8);
    v2 = (unsigned int)qword_4F82670;
  }
  *(_QWORD *)(qword_4F82668 + 8 * v2) = v1;
  LODWORD(qword_4F82670) = qword_4F82670 + 1;
  byte_4F826B9 = 0;
  qword_4F826B0 = (__int64)&unk_49D9748;
  qword_4F826A8 = 0;
  qword_4F82620 = (__int64)&unk_49D9AD8;
  qword_4F826C0 = (__int64)&unk_49DC1D0;
  qword_4F826E0 = (__int64)nullsub_39;
  qword_4F826D8 = (__int64)sub_AA4180;
  sub_C53080(&qword_4F82620, "time-passes", 11);
  if ( qword_4F826A8 )
  {
    v4 = sub_CEADF0();
    v25 = 1;
    v21.m128i_i64[0] = (__int64)"cl::location(x) specified more than once!";
    v24 = 3;
    sub_C53280(&qword_4F82620, &v21, 0, 0, v4);
  }
  else
  {
    byte_4F826B9 = 1;
    qword_4F826A8 = (__int64)byte_4F826E9;
    byte_4F826B8 = byte_4F826E9[0];
  }
  qword_4F82650 = 54;
  LOBYTE(dword_4F8262C) = dword_4F8262C & 0x9F | 0x20;
  qword_4F82648 = (__int64)"Time each pass, printing elapsed time for each on exit";
  sub_C53130(&qword_4F82620);
  __cxa_atexit(sub_AA4490, &qword_4F82620, &qword_4A427C0);
  v16 = 0;
  v23 = sub_BC3500;
  v22 = (void (__fastcall *)(__m128i *, __m128i *, __int64))sub_BC3510;
  sub_BC3510(v15, &v21, 2);
  v17 = v23;
  v16 = v22;
  sub_A17130(&v21);
  qword_4F82540 = (__int64)&unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8254C &= 0x8000u;
  word_4F82550 = 0;
  qword_4F82590 = 0x100000000LL;
  qword_4F82558 = 0;
  qword_4F82560 = 0;
  qword_4F82568 = 0;
  dword_4F82548 = v5;
  qword_4F82570 = 0;
  qword_4F82578 = 0;
  qword_4F82580 = 0;
  qword_4F82588 = (__int64)&unk_4F82598;
  qword_4F825A0 = 0;
  qword_4F825A8 = (__int64)&unk_4F825C0;
  qword_4F825B0 = 1;
  dword_4F825B8 = 0;
  byte_4F825BC = 1;
  v6 = sub_C57470();
  v7 = (unsigned int)qword_4F82590;
  v8 = (unsigned int)qword_4F82590 + 1LL;
  if ( v8 > HIDWORD(qword_4F82590) )
  {
    sub_C8D5F0((char *)&unk_4F82598 - 16, &unk_4F82598, v8, 8);
    v7 = (unsigned int)qword_4F82590;
  }
  *(_QWORD *)(qword_4F82588 + 8 * v7) = v6;
  LODWORD(qword_4F82590) = qword_4F82590 + 1;
  byte_4F825D9 = 0;
  qword_4F825D0 = (__int64)&unk_49D9748;
  qword_4F825C8 = 0;
  qword_4F82540 = (__int64)&unk_49D9AD8;
  qword_4F825E0 = (__int64)&unk_49DC1D0;
  qword_4F82600 = (__int64)nullsub_39;
  qword_4F825F8 = (__int64)sub_AA4180;
  sub_C53080(&qword_4F82540, "time-passes-per-run", 19);
  if ( qword_4F825C8 )
  {
    v9 = sub_CEADF0();
    v25 = 1;
    v21.m128i_i64[0] = (__int64)"cl::location(x) specified more than once!";
    v24 = 3;
    sub_C53280(&qword_4F82540, &v21, 0, 0, v9);
  }
  else
  {
    byte_4F825D9 = 1;
    qword_4F825C8 = (__int64)&unk_4F826E8;
    byte_4F825D8 = unk_4F826E8;
  }
  v19 = 0;
  qword_4F82570 = 62;
  LOBYTE(dword_4F8254C) = dword_4F8254C & 0x9F | 0x20;
  qword_4F82568 = (__int64)"Time each pass run, printing elapsed time for each run on exit";
  if ( v16 && (v16(&v18, (__m128i *)v15, 2), v22 = 0, v20 = (__int64)v17, (v19 = v16) != 0) )
  {
    v16(&v21, &v18, 2);
    v10 = v20;
    v11 = (__int64)v19;
  }
  else
  {
    v10 = (__int64)v23;
    v11 = 0;
  }
  v12 = _mm_loadu_si128(&v21);
  v13 = (void (__fastcall *)(__m128i *, _BYTE *, __int64))qword_4F825F8;
  qword_4F825F8 = v11;
  v21 = _mm_loadu_si128((const __m128i *)&xmmword_4F825E8);
  v22 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v13;
  v23 = (__int64 (__fastcall *)())qword_4F82600;
  qword_4F82600 = v10;
  xmmword_4F825E8 = (__int128)v12;
  if ( v13 )
    v13(&v21, &v21, 3);
  if ( v19 )
    v19(&v18, &v18, 3);
  sub_C53130(&qword_4F82540);
  sub_A17130(v15);
  return __cxa_atexit(sub_AA4490, &qword_4F82540, &qword_4A427C0);
}
