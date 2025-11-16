// Function: ctor_056_0
// Address: 0x492190
//
int ctor_056_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  const char *v4; // rsi
  __int64 v5; // rax
  const char **v6; // r13
  const char **v7; // r14
  const char *v8; // rsi
  const __m128i *v9; // r10
  const char *v10; // rcx
  const char *v11; // r9
  const char *v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // esi
  __m128i *v16; // rdx
  __m128i v17; // xmm1
  int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v27; // rdx
  const char *v28; // [rsp+8h] [rbp-128h]
  const char *v29; // [rsp+8h] [rbp-128h]
  const char *v30; // [rsp+10h] [rbp-120h]
  const char *v31; // [rsp+10h] [rbp-120h]
  __int64 v32; // [rsp+18h] [rbp-118h]
  _QWORD v33[4]; // [rsp+20h] [rbp-110h] BYREF
  void *v34; // [rsp+40h] [rbp-F0h]
  int v35; // [rsp+48h] [rbp-E8h]
  char v36; // [rsp+4Ch] [rbp-E4h]
  const char **v37; // [rsp+50h] [rbp-E0h]
  __int64 v38; // [rsp+58h] [rbp-D8h]
  _QWORD v39[2]; // [rsp+60h] [rbp-D0h] BYREF
  int v40; // [rsp+70h] [rbp-C0h]
  const char *v41; // [rsp+78h] [rbp-B8h]
  __int64 v42; // [rsp+80h] [rbp-B0h]
  const char *v43; // [rsp+88h] [rbp-A8h]
  __int64 v44; // [rsp+90h] [rbp-A0h]
  int v45; // [rsp+98h] [rbp-98h]
  const char *v46; // [rsp+A0h] [rbp-90h]
  __int64 v47; // [rsp+A8h] [rbp-88h]
  char *v48; // [rsp+B0h] [rbp-80h]
  __int64 v49; // [rsp+B8h] [rbp-78h]
  int v50; // [rsp+C0h] [rbp-70h]
  const char *v51; // [rsp+C8h] [rbp-68h]
  __int64 v52; // [rsp+D0h] [rbp-60h]

  v39[0] = "none";
  v41 = "None.";
  v43 = "all-non-critical";
  v46 = "All non-critical edges.";
  v48 = "all";
  v51 = "All edges.";
  v38 = 0x400000003LL;
  v37 = (const char **)v39;
  v39[1] = 4;
  v40 = 0;
  v42 = 5;
  v44 = 16;
  v45 = 1;
  v47 = 23;
  v49 = 3;
  v50 = 2;
  v52 = 10;
  qword_4F87A00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F87A50 = 0x100000000LL;
  dword_4F87A0C &= 0x8000u;
  word_4F87A10 = 0;
  qword_4F87A18 = 0;
  qword_4F87A20 = 0;
  dword_4F87A08 = v0;
  qword_4F87A28 = 0;
  qword_4F87A30 = 0;
  qword_4F87A38 = 0;
  qword_4F87A40 = 0;
  qword_4F87A48 = (__int64)&unk_4F87A58;
  qword_4F87A60 = 0;
  qword_4F87A68 = (__int64)&unk_4F87A80;
  qword_4F87A70 = 1;
  dword_4F87A78 = 0;
  byte_4F87A7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F87A50;
  v3 = (unsigned int)qword_4F87A50 + 1LL;
  if ( v3 > HIDWORD(qword_4F87A50) )
  {
    sub_C8D5F0((char *)&unk_4F87A58 - 16, &unk_4F87A58, v3, 8);
    v2 = (unsigned int)qword_4F87A50;
  }
  v4 = "force-summary-edges-cold";
  *(_QWORD *)(qword_4F87A48 + 8 * v2) = v1;
  LODWORD(qword_4F87A50) = qword_4F87A50 + 1;
  byte_4F87A9C = 0;
  qword_4F87A90 = (__int64)&unk_49DE3A8;
  qword_4F87A88 = 0;
  qword_4F87AA8 = (__int64)&qword_4F87A00;
  qword_4F87A00 = (__int64)&unk_49DE418;
  qword_4F87AA0 = (__int64)&unk_49DE3C8;
  qword_4F87AB0 = (__int64)&unk_4F87AC0;
  qword_4F87AB8 = 0x800000000LL;
  qword_4F87C58 = (__int64)nullsub_189;
  qword_4F87C50 = (__int64)sub_D75BE0;
  sub_C53080(&qword_4F87A00, "force-summary-edges-cold", 24);
  LOBYTE(dword_4F87A0C) = dword_4F87A0C & 0x9F | 0x20;
  if ( qword_4F87A88 )
  {
    v5 = sub_CEADF0();
    v4 = (const char *)v33;
    v33[0] = "cl::location(x) specified more than once!";
    LOWORD(v34) = 259;
    sub_C53280(&qword_4F87A00, v33, 0, 0, v5);
  }
  else
  {
    byte_4F87A9C = 1;
    qword_4F87A88 = (__int64)&dword_4F87C60;
    dword_4F87A98 = dword_4F87C60;
  }
  qword_4F87A30 = 47;
  qword_4F87A28 = (__int64)"Force all edges in the function summary to cold";
  v6 = &v37[5 * (unsigned int)v38];
  v7 = v37;
  while ( v6 != v7 )
  {
    v8 = v7[3];
    v9 = (const __m128i *)v33;
    v10 = v7[4];
    v11 = *v7;
    v12 = v7[1];
    v35 = *((_DWORD *)v7 + 4);
    v13 = (unsigned int)qword_4F87AB8;
    v33[2] = v8;
    v33[3] = v10;
    v14 = qword_4F87AB0;
    v33[0] = v11;
    v15 = qword_4F87AB8;
    v33[1] = v12;
    v34 = &unk_49DE3A8;
    v36 = 1;
    if ( (unsigned __int64)(unsigned int)qword_4F87AB8 + 1 > HIDWORD(qword_4F87AB8) )
    {
      if ( qword_4F87AB0 > (unsigned __int64)v33 )
      {
        v29 = v12;
        v31 = v11;
        sub_D80F80(&qword_4F87AB0, (unsigned int)qword_4F87AB8 + 1LL, (unsigned int)qword_4F87AB8, qword_4F87AB0);
        v13 = (unsigned int)qword_4F87AB8;
        v14 = qword_4F87AB0;
        v9 = (const __m128i *)v33;
        v11 = v31;
        v12 = v29;
        v15 = qword_4F87AB8;
      }
      else
      {
        v28 = v12;
        v30 = v11;
        v27 = qword_4F87AB0 + 48LL * (unsigned int)qword_4F87AB8;
        if ( (unsigned __int64)v33 < v27 )
        {
          v32 = qword_4F87AB0;
          sub_D80F80(&qword_4F87AB0, (unsigned int)qword_4F87AB8 + 1LL, v27, qword_4F87AB0);
          v13 = (unsigned int)qword_4F87AB8;
          v12 = v28;
          v11 = v30;
          v14 = qword_4F87AB0;
          v15 = qword_4F87AB8;
          v9 = (const __m128i *)((char *)v33 + qword_4F87AB0 - v32);
        }
        else
        {
          sub_D80F80(&qword_4F87AB0, (unsigned int)qword_4F87AB8 + 1LL, v27, qword_4F87AB0);
          v13 = (unsigned int)qword_4F87AB8;
          v14 = qword_4F87AB0;
          v11 = v30;
          v12 = v28;
          v9 = (const __m128i *)v33;
          v15 = qword_4F87AB8;
        }
      }
    }
    v16 = (__m128i *)(v14 + 48 * v13);
    if ( v16 )
    {
      v17 = _mm_loadu_si128(v9 + 1);
      *v16 = _mm_loadu_si128(v9);
      v16[1] = v17;
      v16[2].m128i_i32[2] = v9[2].m128i_i32[2];
      v14 = v9[2].m128i_u8[12];
      v16[2].m128i_i64[0] = (__int64)&unk_49DE3A8;
      v16[2].m128i_i8[12] = v14;
      v15 = qword_4F87AB8;
    }
    v7 += 5;
    LODWORD(qword_4F87AB8) = v15 + 1;
    v4 = v11;
    sub_C52F90(qword_4F87AA8, v11, v12, v14);
  }
  sub_C53130(&qword_4F87A00);
  if ( v37 != v39 )
    _libc_free(v37, v4);
  __cxa_atexit(sub_D76EA0, &qword_4F87A00, &qword_4A427C0);
  qword_4F87900 = (__int64)&unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F87950 = 0x100000000LL;
  word_4F87910 = 0;
  dword_4F8790C &= 0x8000u;
  qword_4F87918 = 0;
  qword_4F87920 = 0;
  dword_4F87908 = v18;
  qword_4F87928 = 0;
  qword_4F87930 = 0;
  qword_4F87938 = 0;
  qword_4F87940 = 0;
  qword_4F87948 = (__int64)&unk_4F87958;
  qword_4F87960 = 0;
  qword_4F87968 = (__int64)&unk_4F87980;
  qword_4F87970 = 1;
  dword_4F87978 = 0;
  byte_4F8797C = 1;
  v19 = sub_C57470();
  v20 = (unsigned int)qword_4F87950;
  v21 = (unsigned int)qword_4F87950 + 1LL;
  if ( v21 > HIDWORD(qword_4F87950) )
  {
    sub_C8D5F0((char *)&unk_4F87958 - 16, &unk_4F87958, v21, 8);
    v20 = (unsigned int)qword_4F87950;
  }
  *(_QWORD *)(qword_4F87948 + 8 * v20) = v19;
  qword_4F87988 = (__int64)&byte_4F87998;
  qword_4F879B0 = (__int64)&byte_4F879C0;
  LODWORD(qword_4F87950) = qword_4F87950 + 1;
  qword_4F87990 = 0;
  qword_4F879A8 = (__int64)&unk_49DC130;
  byte_4F87998 = 0;
  byte_4F879C0 = 0;
  qword_4F87900 = (__int64)&unk_49DC010;
  qword_4F879B8 = 0;
  byte_4F879D0 = 0;
  qword_4F879D8 = (__int64)&unk_49DC350;
  qword_4F879F8 = (__int64)nullsub_92;
  qword_4F879F0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4F87900, "module-summary-dot-file", 23);
  qword_4F87940 = 8;
  qword_4F87930 = 42;
  LOBYTE(dword_4F8790C) = dword_4F8790C & 0x9F | 0x20;
  qword_4F87938 = (__int64)"filename";
  qword_4F87928 = (__int64)"File to emit dot graph of new summary into";
  sub_C53130(&qword_4F87900);
  __cxa_atexit(sub_BC5A40, &qword_4F87900, &qword_4A427C0);
  qword_4F87820 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8789C = 1;
  qword_4F87870 = 0x100000000LL;
  dword_4F8782C &= 0x8000u;
  qword_4F87838 = 0;
  qword_4F87840 = 0;
  qword_4F87848 = 0;
  dword_4F87828 = v22;
  word_4F87830 = 0;
  qword_4F87850 = 0;
  qword_4F87858 = 0;
  qword_4F87860 = 0;
  qword_4F87868 = (__int64)&unk_4F87878;
  qword_4F87880 = 0;
  qword_4F87888 = (__int64)&unk_4F878A0;
  qword_4F87890 = 1;
  dword_4F87898 = 0;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_4F87870;
  v25 = (unsigned int)qword_4F87870 + 1LL;
  if ( v25 > HIDWORD(qword_4F87870) )
  {
    sub_C8D5F0((char *)&unk_4F87878 - 16, &unk_4F87878, v25, 8);
    v24 = (unsigned int)qword_4F87870;
  }
  *(_QWORD *)(qword_4F87868 + 8 * v24) = v23;
  LODWORD(qword_4F87870) = qword_4F87870 + 1;
  qword_4F878A8 = 0;
  qword_4F878B0 = (__int64)&unk_49D9748;
  qword_4F878B8 = 0;
  qword_4F87820 = (__int64)&unk_49DC090;
  qword_4F878C0 = (__int64)&unk_49DC1D0;
  qword_4F878E0 = (__int64)nullsub_23;
  qword_4F878D8 = (__int64)sub_984030;
  sub_C53080(&qword_4F87820, "enable-memprof-indirect-call-support", 36);
  LOBYTE(qword_4F878A8) = 0;
  LOWORD(qword_4F878B8) = 256;
  qword_4F87850 = 65;
  LOBYTE(dword_4F8782C) = dword_4F8782C & 0x9F | 0x20;
  qword_4F87848 = (__int64)"Enable MemProf support for summarizing and cloning indirect calls";
  sub_C53130(&qword_4F87820);
  return __cxa_atexit(sub_984900, &qword_4F87820, &qword_4A427C0);
}
