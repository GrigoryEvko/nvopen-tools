// Function: ctor_693
// Address: 0x5a71a0
//
int __fastcall ctor_693(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx

  qword_5040B00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5040B50 = 0x100000000LL;
  dword_5040B0C &= 0x8000u;
  word_5040B10 = 0;
  qword_5040B18 = 0;
  qword_5040B20 = 0;
  dword_5040B08 = v4;
  qword_5040B28 = 0;
  qword_5040B30 = 0;
  qword_5040B38 = 0;
  qword_5040B40 = 0;
  qword_5040B48 = (__int64)&unk_5040B58;
  qword_5040B60 = 0;
  qword_5040B68 = (__int64)&unk_5040B80;
  qword_5040B70 = 1;
  dword_5040B78 = 0;
  byte_5040B7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040B50;
  v7 = (unsigned int)qword_5040B50 + 1LL;
  if ( v7 > HIDWORD(qword_5040B50) )
  {
    sub_C8D5F0((char *)&unk_5040B58 - 16, &unk_5040B58, v7, 8);
    v6 = (unsigned int)qword_5040B50;
  }
  *(_QWORD *)(qword_5040B48 + 8 * v6) = v5;
  LODWORD(qword_5040B50) = qword_5040B50 + 1;
  qword_5040B88 = 0;
  qword_5040B90 = (__int64)&unk_49D9728;
  qword_5040B98 = 0;
  qword_5040B00 = (__int64)&unk_49DBF10;
  qword_5040BA0 = (__int64)&unk_49DC290;
  qword_5040BC0 = (__int64)nullsub_24;
  qword_5040BB8 = (__int64)sub_984050;
  sub_C53080(&qword_5040B00, "nvptx-traverse-address-aliasing-limit", 37);
  qword_5040B30 = 55;
  LODWORD(qword_5040B88) = 100;
  BYTE4(qword_5040B98) = 1;
  LODWORD(qword_5040B98) = 100;
  LOBYTE(dword_5040B0C) = dword_5040B0C & 0x9F | 0x20;
  qword_5040B28 = (__int64)"Depth limit for finding address space through traversal";
  sub_C53130(&qword_5040B00);
  __cxa_atexit(sub_984970, &qword_5040B00, &qword_4A427C0);
  qword_5040A20 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5040B00, v8, v9), 1u);
  qword_5040A70 = 0x100000000LL;
  dword_5040A2C &= 0x8000u;
  word_5040A30 = 0;
  qword_5040A38 = 0;
  qword_5040A40 = 0;
  dword_5040A28 = v10;
  qword_5040A48 = 0;
  qword_5040A50 = 0;
  qword_5040A58 = 0;
  qword_5040A60 = 0;
  qword_5040A68 = (__int64)&unk_5040A78;
  qword_5040A80 = 0;
  qword_5040A88 = (__int64)&unk_5040AA0;
  qword_5040A90 = 1;
  dword_5040A98 = 0;
  byte_5040A9C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5040A70;
  v13 = (unsigned int)qword_5040A70 + 1LL;
  if ( v13 > HIDWORD(qword_5040A70) )
  {
    sub_C8D5F0((char *)&unk_5040A78 - 16, &unk_5040A78, v13, 8);
    v12 = (unsigned int)qword_5040A70;
  }
  *(_QWORD *)(qword_5040A68 + 8 * v12) = v11;
  qword_5040AB0 = (__int64)&unk_49D9748;
  qword_5040A20 = (__int64)&unk_49DC090;
  LODWORD(qword_5040A70) = qword_5040A70 + 1;
  qword_5040AA8 = 0;
  qword_5040AC0 = (__int64)&unk_49DC1D0;
  qword_5040AB8 = 0;
  qword_5040AE0 = (__int64)nullsub_23;
  qword_5040AD8 = (__int64)sub_984030;
  sub_C53080(&qword_5040A20, "strict-aliasing", 15);
  LOWORD(qword_5040AB8) = 257;
  LOBYTE(qword_5040AA8) = 1;
  qword_5040A50 = 27;
  LOBYTE(dword_5040A2C) = dword_5040A2C & 0x9F | 0x20;
  qword_5040A48 = (__int64)"Datatype based strict alias";
  sub_C53130(&qword_5040A20);
  __cxa_atexit(sub_984900, &qword_5040A20, &qword_4A427C0);
  qword_5040940 = (__int64)&unk_49DC150;
  v16 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5040A20, v14, v15), 1u);
  qword_5040990 = 0x100000000LL;
  word_5040950 = 0;
  dword_504094C &= 0x8000u;
  qword_5040958 = 0;
  qword_5040960 = 0;
  dword_5040948 = v16;
  qword_5040968 = 0;
  qword_5040970 = 0;
  qword_5040978 = 0;
  qword_5040980 = 0;
  qword_5040988 = (__int64)&unk_5040998;
  qword_50409A0 = 0;
  qword_50409A8 = (__int64)&unk_50409C0;
  qword_50409B0 = 1;
  dword_50409B8 = 0;
  byte_50409BC = 1;
  v17 = sub_C57470();
  v18 = (unsigned int)qword_5040990;
  v19 = (unsigned int)qword_5040990 + 1LL;
  if ( v19 > HIDWORD(qword_5040990) )
  {
    sub_C8D5F0((char *)&unk_5040998 - 16, &unk_5040998, v19, 8);
    v18 = (unsigned int)qword_5040990;
  }
  *(_QWORD *)(qword_5040988 + 8 * v18) = v17;
  qword_50409D0 = (__int64)&unk_49D9748;
  qword_5040940 = (__int64)&unk_49DC090;
  LODWORD(qword_5040990) = qword_5040990 + 1;
  qword_50409C8 = 0;
  qword_50409E0 = (__int64)&unk_49DC1D0;
  qword_50409D8 = 0;
  qword_5040A00 = (__int64)nullsub_23;
  qword_50409F8 = (__int64)sub_984030;
  sub_C53080(&qword_5040940, "nvptxaa-relax-fences", 20);
  qword_5040970 = 37;
  LOBYTE(qword_50409C8) = 1;
  LOBYTE(dword_504094C) = dword_504094C & 0x9F | 0x20;
  qword_5040968 = (__int64)"Enable ordering relaxation for fences";
  LOWORD(qword_50409D8) = 257;
  sub_C53130(&qword_5040940);
  return __cxa_atexit(sub_984900, &qword_5040940, &qword_4A427C0);
}
