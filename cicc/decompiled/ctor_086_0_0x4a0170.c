// Function: ctor_086_0
// Address: 0x4a0170
//
int ctor_086_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  const char *v4; // rsi
  _QWORD *v5; // r14
  _QWORD *v6; // r13
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rdx
  const char *v10; // r15
  __int64 v11; // r10
  __int64 v12; // rax
  const __m128i *v13; // rcx
  int v14; // edx
  __int64 v15; // rax
  __m128i v16; // xmm1
  __int8 v17; // dl
  int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // rax
  __m128i *v22; // rax
  void *v23; // rdi
  __int64 v24; // rsi
  const __m128i *v25; // rdx
  __int64 v26; // rcx
  __int8 v27; // di
  int v28; // eax
  int v29; // [rsp+8h] [rbp-148h]
  unsigned __int64 v30; // [rsp+10h] [rbp-140h]
  char v31; // [rsp+1Fh] [rbp-131h]
  __int64 v32; // [rsp+20h] [rbp-130h]
  __int64 v33; // [rsp+38h] [rbp-118h] BYREF
  _QWORD v34[5]; // [rsp+40h] [rbp-110h] BYREF
  int v35; // [rsp+68h] [rbp-E8h]
  char v36; // [rsp+6Ch] [rbp-E4h]
  _QWORD *v37; // [rsp+70h] [rbp-E0h]
  __int64 v38; // [rsp+78h] [rbp-D8h]
  _QWORD v39[2]; // [rsp+80h] [rbp-D0h] BYREF
  int v40; // [rsp+90h] [rbp-C0h]
  const char *v41; // [rsp+98h] [rbp-B8h]
  __int64 v42; // [rsp+A0h] [rbp-B0h]
  const char *v43; // [rsp+A8h] [rbp-A8h]
  __int64 v44; // [rsp+B0h] [rbp-A0h]
  int v45; // [rsp+B8h] [rbp-98h]
  char *v46; // [rsp+C0h] [rbp-90h]
  __int64 v47; // [rsp+C8h] [rbp-88h]
  const char *v48; // [rsp+D0h] [rbp-80h]
  __int64 v49; // [rsp+D8h] [rbp-78h]
  int v50; // [rsp+E0h] [rbp-70h]
  char *v51; // [rsp+E8h] [rbp-68h]
  __int64 v52; // [rsp+F0h] [rbp-60h]

  v39[0] = "Default";
  v41 = "Default for platform";
  v43 = "Enable";
  v46 = "Enabled";
  v48 = "Disable";
  v51 = "Disabled";
  v38 = 0x400000003LL;
  v37 = v39;
  v39[1] = 7;
  v40 = 0;
  v42 = 20;
  v44 = 6;
  v45 = 1;
  v47 = 7;
  v49 = 7;
  v50 = 2;
  v52 = 8;
  qword_4F8FEC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8FF3C = 1;
  qword_4F8FF10 = 0x100000000LL;
  dword_4F8FECC &= 0x8000u;
  qword_4F8FED8 = 0;
  qword_4F8FEE0 = 0;
  qword_4F8FEE8 = 0;
  dword_4F8FEC8 = v0;
  word_4F8FED0 = 0;
  qword_4F8FEF0 = 0;
  qword_4F8FEF8 = 0;
  qword_4F8FF00 = 0;
  qword_4F8FF08 = (__int64)&unk_4F8FF18;
  qword_4F8FF20 = 0;
  qword_4F8FF28 = (__int64)&unk_4F8FF40;
  qword_4F8FF30 = 1;
  dword_4F8FF38 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8FF10;
  v3 = (unsigned int)qword_4F8FF10 + 1LL;
  if ( v3 > HIDWORD(qword_4F8FF10) )
  {
    sub_C8D5F0((char *)&unk_4F8FF18 - 16, &unk_4F8FF18, v3, 8);
    v2 = (unsigned int)qword_4F8FF10;
  }
  v4 = "dwarf-extended-loc";
  *(_QWORD *)(qword_4F8FF08 + 8 * v2) = v1;
  qword_4F8FEC0 = (__int64)&off_49E5F88;
  qword_4F8FF60 = (__int64)&off_49E5F38;
  qword_4F8FF70 = (__int64)&unk_4F8FF80;
  qword_4F8FF78 = 0x800000000LL;
  qword_4F90118 = (__int64)nullsub_395;
  qword_4F90110 = (__int64)sub_106E290;
  LODWORD(qword_4F8FF10) = qword_4F8FF10 + 1;
  qword_4F8FF48 = 0;
  qword_4F8FF58 = 0;
  qword_4F8FF50 = (__int64)&off_49E5F18;
  qword_4F8FF68 = (__int64)&qword_4F8FEC0;
  sub_C53080(&qword_4F8FEC0, "dwarf-extended-loc", 18);
  qword_4F8FEF0 = 58;
  v5 = v37;
  LOBYTE(dword_4F8FECC) = dword_4F8FECC & 0x9F | 0x20;
  qword_4F8FEE8 = (__int64)"Disable emission of the extended flags in .loc directives.";
  v6 = &v37[5 * (unsigned int)v38];
  if ( v37 != v6 )
  {
    do
    {
      v7 = *((_DWORD *)v5 + 4);
      v8 = v5[3];
      v34[4] = &off_49E5F18;
      v9 = v5[4];
      v10 = (const char *)*v5;
      v36 = 1;
      v11 = v5[1];
      v35 = v7;
      v12 = (unsigned int)qword_4F8FF78;
      v34[2] = v8;
      v13 = (const __m128i *)v34;
      v34[3] = v9;
      v34[0] = v10;
      v14 = qword_4F8FF78;
      v34[1] = v11;
      if ( (unsigned __int64)(unsigned int)qword_4F8FF78 + 1 > HIDWORD(qword_4F8FF78) )
      {
        if ( qword_4F8FF70 > (unsigned __int64)v34
          || (unsigned __int64)v34 >= qword_4F8FF70 + 48 * (unsigned __int64)(unsigned int)qword_4F8FF78 )
        {
          v30 = -1;
          v31 = 0;
        }
        else
        {
          v31 = 1;
          v30 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v34 - qword_4F8FF70) >> 4);
        }
        v32 = v11;
        v22 = (__m128i *)sub_C8D7D0(
                           (char *)&unk_4F8FF80 - 16,
                           &unk_4F8FF80,
                           (unsigned int)qword_4F8FF78 + 1LL,
                           48,
                           &v33);
        v23 = (void *)qword_4F8FF70;
        v11 = v32;
        v24 = (__int64)v22;
        v25 = (const __m128i *)qword_4F8FF70;
        v26 = qword_4F8FF70 + 48LL * (unsigned int)qword_4F8FF78;
        if ( qword_4F8FF70 != v26 )
        {
          do
          {
            if ( v22 )
            {
              *v22 = _mm_loadu_si128(v25);
              v22[1] = _mm_loadu_si128(v25 + 1);
              v22[2].m128i_i32[2] = v25[2].m128i_i32[2];
              v27 = v25[2].m128i_i8[12];
              v22[2].m128i_i64[0] = (__int64)&off_49E5F18;
              v22[2].m128i_i8[12] = v27;
            }
            v25 += 3;
            v22 += 3;
          }
          while ( (const __m128i *)v26 != v25 );
          v23 = (void *)qword_4F8FF70;
        }
        v28 = v33;
        if ( v23 != &unk_4F8FF80 )
        {
          v29 = v33;
          _libc_free(v23, v24);
          v11 = v32;
          v28 = v29;
        }
        HIDWORD(qword_4F8FF78) = v28;
        v12 = (unsigned int)qword_4F8FF78;
        v13 = (const __m128i *)v34;
        qword_4F8FF70 = v24;
        v14 = qword_4F8FF78;
        if ( v31 )
          v13 = (const __m128i *)(v24 + 48 * v30);
      }
      v15 = qword_4F8FF70 + 48 * v12;
      if ( v15 )
      {
        v16 = _mm_loadu_si128(v13 + 1);
        *(__m128i *)v15 = _mm_loadu_si128(v13);
        *(__m128i *)(v15 + 16) = v16;
        *(_DWORD *)(v15 + 40) = v13[2].m128i_i32[2];
        v17 = v13[2].m128i_i8[12];
        *(_QWORD *)(v15 + 32) = &off_49E5F18;
        *(_BYTE *)(v15 + 44) = v17;
        v14 = qword_4F8FF78;
      }
      v4 = v10;
      v5 += 5;
      LODWORD(qword_4F8FF78) = v14 + 1;
      sub_C52F90(qword_4F8FF68, v10, v11, v13);
    }
    while ( v6 != v5 );
  }
  LODWORD(qword_4F8FF48) = 0;
  BYTE4(qword_4F8FF58) = 1;
  LODWORD(qword_4F8FF58) = 0;
  sub_C53130(&qword_4F8FEC0);
  if ( v37 != v39 )
    _libc_free(v37, v4);
  __cxa_atexit(sub_106E980, &qword_4F8FEC0, &qword_4A427C0);
  qword_4F8FDE0 = &unk_49DC150;
  v18 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8FDEC = word_4F8FDEC & 0x8000;
  unk_4F8FDF0 = 0;
  qword_4F8FE28[1] = 0x100000000LL;
  unk_4F8FDE8 = v18;
  unk_4F8FDF8 = 0;
  unk_4F8FE00 = 0;
  unk_4F8FE08 = 0;
  unk_4F8FE10 = 0;
  unk_4F8FE18 = 0;
  unk_4F8FE20 = 0;
  qword_4F8FE28[0] = &qword_4F8FE28[2];
  qword_4F8FE28[3] = 0;
  qword_4F8FE28[4] = &qword_4F8FE28[7];
  qword_4F8FE28[5] = 1;
  LODWORD(qword_4F8FE28[6]) = 0;
  BYTE4(qword_4F8FE28[6]) = 1;
  v19 = sub_C57470();
  v20 = LODWORD(qword_4F8FE28[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8FE28[1]) + 1 > HIDWORD(qword_4F8FE28[1]) )
  {
    sub_C8D5F0(qword_4F8FE28, &qword_4F8FE28[2], LODWORD(qword_4F8FE28[1]) + 1LL, 8);
    v20 = LODWORD(qword_4F8FE28[1]);
  }
  *(_QWORD *)(qword_4F8FE28[0] + 8 * v20) = v19;
  ++LODWORD(qword_4F8FE28[1]);
  qword_4F8FE28[8] = 0;
  qword_4F8FE28[9] = &unk_49DC110;
  qword_4F8FE28[10] = 0;
  qword_4F8FDE0 = &unk_49D97F0;
  qword_4F8FE28[11] = &unk_49DC200;
  qword_4F8FE28[15] = nullsub_26;
  qword_4F8FE28[14] = sub_9C26D0;
  sub_C53080(&qword_4F8FDE0, "use-leb128-directives", 21);
  unk_4F8FE10 = 67;
  LODWORD(qword_4F8FE28[8]) = 0;
  BYTE4(qword_4F8FE28[10]) = 1;
  LODWORD(qword_4F8FE28[10]) = 0;
  LOBYTE(word_4F8FDEC) = word_4F8FDEC & 0x9F | 0x20;
  unk_4F8FE08 = "Disable the usage of LEB128 directives, and generate .byte instead.";
  sub_C53130(&qword_4F8FDE0);
  return __cxa_atexit(sub_9C44F0, &qword_4F8FDE0, &qword_4A427C0);
}
