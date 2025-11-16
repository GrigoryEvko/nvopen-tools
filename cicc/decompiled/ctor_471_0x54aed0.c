// Function: ctor_471
// Address: 0x54aed0
//
int ctor_471()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_5001BE0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5001C30 = 0x100000000LL;
  dword_5001BEC &= 0x8000u;
  word_5001BF0 = 0;
  qword_5001BF8 = 0;
  qword_5001C00 = 0;
  dword_5001BE8 = v0;
  qword_5001C08 = 0;
  qword_5001C10 = 0;
  qword_5001C18 = 0;
  qword_5001C20 = 0;
  qword_5001C28 = (__int64)&unk_5001C38;
  qword_5001C40 = 0;
  qword_5001C48 = (__int64)&unk_5001C60;
  qword_5001C50 = 1;
  dword_5001C58 = 0;
  byte_5001C5C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5001C30;
  v3 = (unsigned int)qword_5001C30 + 1LL;
  if ( v3 > HIDWORD(qword_5001C30) )
  {
    sub_C8D5F0((char *)&unk_5001C38 - 16, &unk_5001C38, v3, 8);
    v2 = (unsigned int)qword_5001C30;
  }
  *(_QWORD *)(qword_5001C28 + 8 * v2) = v1;
  LODWORD(qword_5001C30) = qword_5001C30 + 1;
  qword_5001C68 = 0;
  qword_5001C70 = (__int64)&unk_49D9748;
  qword_5001C78 = 0;
  qword_5001BE0 = (__int64)&unk_49DC090;
  qword_5001C80 = (__int64)&unk_49DC1D0;
  qword_5001CA0 = (__int64)nullsub_23;
  qword_5001C98 = (__int64)sub_984030;
  sub_C53080(&qword_5001BE0, "allow-unroll-and-jam", 20);
  qword_5001C10 = 37;
  LOBYTE(dword_5001BEC) = dword_5001BEC & 0x9F | 0x20;
  qword_5001C08 = (__int64)"Allows loops to be unroll-and-jammed.";
  sub_C53130(&qword_5001BE0);
  __cxa_atexit(sub_984900, &qword_5001BE0, &qword_4A427C0);
  qword_5001B00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5001B50 = 0x100000000LL;
  word_5001B10 = 0;
  dword_5001B0C &= 0x8000u;
  qword_5001B18 = 0;
  qword_5001B20 = 0;
  dword_5001B08 = v4;
  qword_5001B28 = 0;
  qword_5001B30 = 0;
  qword_5001B38 = 0;
  qword_5001B40 = 0;
  qword_5001B48 = (__int64)&unk_5001B58;
  qword_5001B60 = 0;
  qword_5001B68 = (__int64)&unk_5001B80;
  qword_5001B70 = 1;
  dword_5001B78 = 0;
  byte_5001B7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5001B50;
  v7 = (unsigned int)qword_5001B50 + 1LL;
  if ( v7 > HIDWORD(qword_5001B50) )
  {
    sub_C8D5F0((char *)&unk_5001B58 - 16, &unk_5001B58, v7, 8);
    v6 = (unsigned int)qword_5001B50;
  }
  *(_QWORD *)(qword_5001B48 + 8 * v6) = v5;
  qword_5001B90 = (__int64)&unk_49D9728;
  qword_5001B00 = (__int64)&unk_49DBF10;
  qword_5001BA0 = (__int64)&unk_49DC290;
  LODWORD(qword_5001B50) = qword_5001B50 + 1;
  qword_5001BC0 = (__int64)nullsub_24;
  qword_5001B88 = 0;
  qword_5001BB8 = (__int64)sub_984050;
  qword_5001B98 = 0;
  sub_C53080(&qword_5001B00, "unroll-and-jam-count", 20);
  qword_5001B30 = 113;
  LOBYTE(dword_5001B0C) = dword_5001B0C & 0x9F | 0x20;
  qword_5001B28 = (__int64)"Use this unroll count for all loops including those with unroll_and_jam_count pragma values, "
                           "for testing purposes";
  sub_C53130(&qword_5001B00);
  __cxa_atexit(sub_984970, &qword_5001B00, &qword_4A427C0);
  qword_5001A20 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5001A9C = 1;
  qword_5001A70 = 0x100000000LL;
  dword_5001A2C &= 0x8000u;
  qword_5001A68 = (__int64)&unk_5001A78;
  qword_5001A38 = 0;
  qword_5001A40 = 0;
  dword_5001A28 = v8;
  word_5001A30 = 0;
  qword_5001A48 = 0;
  qword_5001A50 = 0;
  qword_5001A58 = 0;
  qword_5001A60 = 0;
  qword_5001A80 = 0;
  qword_5001A88 = (__int64)&unk_5001AA0;
  qword_5001A90 = 1;
  dword_5001A98 = 0;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_5001A70;
  if ( (unsigned __int64)(unsigned int)qword_5001A70 + 1 > HIDWORD(qword_5001A70) )
  {
    v16 = v9;
    sub_C8D5F0((char *)&unk_5001A78 - 16, &unk_5001A78, (unsigned int)qword_5001A70 + 1LL, 8);
    v10 = (unsigned int)qword_5001A70;
    v9 = v16;
  }
  *(_QWORD *)(qword_5001A68 + 8 * v10) = v9;
  qword_5001AB0 = (__int64)&unk_49D9728;
  qword_5001A20 = (__int64)&unk_49DBF10;
  qword_5001AC0 = (__int64)&unk_49DC290;
  LODWORD(qword_5001A70) = qword_5001A70 + 1;
  qword_5001AE0 = (__int64)nullsub_24;
  qword_5001AA8 = 0;
  qword_5001AD8 = (__int64)sub_984050;
  qword_5001AB8 = 0;
  sub_C53080(&qword_5001A20, "unroll-and-jam-threshold", 24);
  LODWORD(qword_5001AA8) = 60;
  BYTE4(qword_5001AB8) = 1;
  LODWORD(qword_5001AB8) = 60;
  qword_5001A50 = 58;
  LOBYTE(dword_5001A2C) = dword_5001A2C & 0x9F | 0x20;
  qword_5001A48 = (__int64)"Threshold to use for inner loop when doing unroll and jam.";
  sub_C53130(&qword_5001A20);
  __cxa_atexit(sub_984970, &qword_5001A20, &qword_4A427C0);
  qword_5001940 = (__int64)&unk_49DC150;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500194C &= 0x8000u;
  word_5001950 = 0;
  qword_5001990 = 0x100000000LL;
  qword_5001988 = (__int64)&unk_5001998;
  qword_5001958 = 0;
  qword_5001960 = 0;
  dword_5001948 = v11;
  qword_5001968 = 0;
  qword_5001970 = 0;
  qword_5001978 = 0;
  qword_5001980 = 0;
  qword_50019A0 = 0;
  qword_50019A8 = (__int64)&unk_50019C0;
  qword_50019B0 = 1;
  dword_50019B8 = 0;
  byte_50019BC = 1;
  v12 = sub_C57470();
  v13 = (unsigned int)qword_5001990;
  v14 = (unsigned int)qword_5001990 + 1LL;
  if ( v14 > HIDWORD(qword_5001990) )
  {
    sub_C8D5F0((char *)&unk_5001998 - 16, &unk_5001998, v14, 8);
    v13 = (unsigned int)qword_5001990;
  }
  *(_QWORD *)(qword_5001988 + 8 * v13) = v12;
  qword_50019D0 = (__int64)&unk_49D9728;
  qword_5001940 = (__int64)&unk_49DBF10;
  qword_50019E0 = (__int64)&unk_49DC290;
  LODWORD(qword_5001990) = qword_5001990 + 1;
  qword_5001A00 = (__int64)nullsub_24;
  qword_50019C8 = 0;
  qword_50019F8 = (__int64)sub_984050;
  qword_50019D8 = 0;
  sub_C53080(&qword_5001940, "pragma-unroll-and-jam-threshold", 31);
  LODWORD(qword_50019C8) = 1024;
  BYTE4(qword_50019D8) = 1;
  LODWORD(qword_50019D8) = 1024;
  qword_5001970 = 82;
  LOBYTE(dword_500194C) = dword_500194C & 0x9F | 0x20;
  qword_5001968 = (__int64)"Unrolled size limit for loops with an unroll_and_jam(full) or unroll_count pragma.";
  sub_C53130(&qword_5001940);
  return __cxa_atexit(sub_984970, &qword_5001940, &qword_4A427C0);
}
