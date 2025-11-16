// Function: ctor_534
// Address: 0x569570
//
int ctor_534()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  qword_5014B80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5014BD0 = 0x100000000LL;
  dword_5014B8C &= 0x8000u;
  word_5014B90 = 0;
  qword_5014B98 = 0;
  qword_5014BA0 = 0;
  dword_5014B88 = v0;
  qword_5014BA8 = 0;
  qword_5014BB0 = 0;
  qword_5014BB8 = 0;
  qword_5014BC0 = 0;
  qword_5014BC8 = (__int64)&unk_5014BD8;
  qword_5014BE0 = 0;
  qword_5014BE8 = (__int64)&unk_5014C00;
  qword_5014BF0 = 1;
  dword_5014BF8 = 0;
  byte_5014BFC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5014BD0;
  v3 = (unsigned int)qword_5014BD0 + 1LL;
  if ( v3 > HIDWORD(qword_5014BD0) )
  {
    sub_C8D5F0((char *)&unk_5014BD8 - 16, &unk_5014BD8, v3, 8);
    v2 = (unsigned int)qword_5014BD0;
  }
  *(_QWORD *)(qword_5014BC8 + 8 * v2) = v1;
  qword_5014C10 = (__int64)&unk_49D9748;
  qword_5014B80 = (__int64)&unk_49DC090;
  qword_5014C20 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5014BD0) = qword_5014BD0 + 1;
  qword_5014C40 = (__int64)nullsub_23;
  qword_5014C08 = 0;
  qword_5014C38 = (__int64)sub_984030;
  qword_5014C18 = 0;
  sub_C53080(&qword_5014B80, "allow-restrict-in-struct", 24);
  LOBYTE(qword_5014C08) = 0;
  LOWORD(qword_5014C18) = 256;
  qword_5014BA8 = (__int64)"Allows __restrict__ keyword in struct.";
  qword_5014BB0 = 38;
  sub_C53130(&qword_5014B80);
  __cxa_atexit(sub_984900, &qword_5014B80, &qword_4A427C0);
  qword_5014AA0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5014AF0 = 0x100000000LL;
  dword_5014AAC &= 0x8000u;
  word_5014AB0 = 0;
  qword_5014AE8 = (__int64)&unk_5014AF8;
  qword_5014AB8 = 0;
  dword_5014AA8 = v4;
  qword_5014AC0 = 0;
  qword_5014AC8 = 0;
  qword_5014AD0 = 0;
  qword_5014AD8 = 0;
  qword_5014AE0 = 0;
  qword_5014B00 = 0;
  qword_5014B08 = (__int64)&unk_5014B20;
  qword_5014B10 = 1;
  dword_5014B18 = 0;
  byte_5014B1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5014AF0;
  if ( (unsigned __int64)(unsigned int)qword_5014AF0 + 1 > HIDWORD(qword_5014AF0) )
  {
    v15 = v5;
    sub_C8D5F0((char *)&unk_5014AF8 - 16, &unk_5014AF8, (unsigned int)qword_5014AF0 + 1LL, 8);
    v6 = (unsigned int)qword_5014AF0;
    v5 = v15;
  }
  *(_QWORD *)(qword_5014AE8 + 8 * v6) = v5;
  qword_5014B30 = (__int64)&unk_49D9748;
  qword_5014AA0 = (__int64)&unk_49DC090;
  qword_5014B40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5014AF0) = qword_5014AF0 + 1;
  qword_5014B60 = (__int64)nullsub_23;
  qword_5014B28 = 0;
  qword_5014B58 = (__int64)sub_984030;
  qword_5014B38 = 0;
  sub_C53080(&qword_5014AA0, "apply-multi-level-restrict", 26);
  LOWORD(qword_5014B38) = 256;
  qword_5014AC8 = (__int64)"Apply __restrict__ to all pointer levels.";
  LOBYTE(qword_5014B28) = 0;
  qword_5014AD0 = 41;
  sub_C53130(&qword_5014AA0);
  __cxa_atexit(sub_984900, &qword_5014AA0, &qword_4A427C0);
  qword_50149C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_5014A3C = 1;
  word_50149D0 = 0;
  qword_5014A10 = 0x100000000LL;
  dword_50149CC &= 0x8000u;
  qword_5014A08 = (__int64)&unk_5014A18;
  qword_50149D8 = 0;
  dword_50149C8 = v7;
  qword_50149E0 = 0;
  qword_50149E8 = 0;
  qword_50149F0 = 0;
  qword_50149F8 = 0;
  qword_5014A00 = 0;
  qword_5014A20 = 0;
  qword_5014A28 = (__int64)&unk_5014A40;
  qword_5014A30 = 1;
  dword_5014A38 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5014A10;
  if ( (unsigned __int64)(unsigned int)qword_5014A10 + 1 > HIDWORD(qword_5014A10) )
  {
    v16 = v8;
    sub_C8D5F0((char *)&unk_5014A18 - 16, &unk_5014A18, (unsigned int)qword_5014A10 + 1LL, 8);
    v9 = (unsigned int)qword_5014A10;
    v8 = v16;
  }
  *(_QWORD *)(qword_5014A08 + 8 * v9) = v8;
  LODWORD(qword_5014A10) = qword_5014A10 + 1;
  qword_5014A48 = 0;
  qword_5014A50 = (__int64)&unk_49DA090;
  qword_5014A58 = 0;
  qword_50149C0 = (__int64)&unk_49DBF90;
  qword_5014A60 = (__int64)&unk_49DC230;
  qword_5014A80 = (__int64)nullsub_58;
  qword_5014A78 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_50149C0, "process-restrict", 16);
  LODWORD(qword_5014A48) = 1;
  BYTE4(qword_5014A58) = 1;
  LODWORD(qword_5014A58) = 1;
  qword_50149F0 = 29;
  LOBYTE(dword_50149CC) = dword_50149CC & 0x9F | 0x20;
  qword_50149E8 = (__int64)"Process __restrict__ keyword.";
  sub_C53130(&qword_50149C0);
  __cxa_atexit(sub_B2B680, &qword_50149C0, &qword_4A427C0);
  qword_50148E0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501495C = 1;
  qword_5014930 = 0x100000000LL;
  dword_50148EC &= 0x8000u;
  qword_5014928 = (__int64)&unk_5014938;
  qword_50148F8 = 0;
  qword_5014900 = 0;
  dword_50148E8 = v10;
  word_50148F0 = 0;
  qword_5014908 = 0;
  qword_5014910 = 0;
  qword_5014918 = 0;
  qword_5014920 = 0;
  qword_5014940 = 0;
  qword_5014948 = (__int64)&unk_5014960;
  qword_5014950 = 1;
  dword_5014958 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5014930;
  v13 = (unsigned int)qword_5014930 + 1LL;
  if ( v13 > HIDWORD(qword_5014930) )
  {
    sub_C8D5F0((char *)&unk_5014938 - 16, &unk_5014938, v13, 8);
    v12 = (unsigned int)qword_5014930;
  }
  *(_QWORD *)(qword_5014928 + 8 * v12) = v11;
  qword_5014970 = (__int64)&unk_49D9748;
  qword_50148E0 = (__int64)&unk_49DC090;
  qword_5014980 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5014930) = qword_5014930 + 1;
  qword_50149A0 = (__int64)nullsub_23;
  qword_5014968 = 0;
  qword_5014998 = (__int64)sub_984030;
  qword_5014978 = 0;
  sub_C53080(&qword_50148E0, "dump-process-restrict", 21);
  LOBYTE(qword_5014968) = 0;
  LOWORD(qword_5014978) = 256;
  qword_5014910 = 48;
  LOBYTE(dword_50148EC) = dword_50148EC & 0x9F | 0x20;
  qword_5014908 = (__int64)"Dump debug messages during DebugProcessRestrict.";
  sub_C53130(&qword_50148E0);
  return __cxa_atexit(sub_984900, &qword_50148E0, &qword_4A427C0);
}
