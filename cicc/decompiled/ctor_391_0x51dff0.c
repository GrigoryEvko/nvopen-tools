// Function: ctor_391
// Address: 0x51dff0
//
int ctor_391()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4FDF640 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDF690 = 0x100000000LL;
  word_4FDF650 = 0;
  dword_4FDF64C &= 0x8000u;
  qword_4FDF658 = 0;
  qword_4FDF660 = 0;
  dword_4FDF648 = v0;
  qword_4FDF668 = 0;
  qword_4FDF670 = 0;
  qword_4FDF678 = 0;
  qword_4FDF680 = 0;
  qword_4FDF688 = (__int64)&unk_4FDF698;
  qword_4FDF6A0 = 0;
  qword_4FDF6A8 = (__int64)&unk_4FDF6C0;
  qword_4FDF6B0 = 1;
  dword_4FDF6B8 = 0;
  byte_4FDF6BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FDF690;
  v3 = (unsigned int)qword_4FDF690 + 1LL;
  if ( v3 > HIDWORD(qword_4FDF690) )
  {
    sub_C8D5F0((char *)&unk_4FDF698 - 16, &unk_4FDF698, v3, 8);
    v2 = (unsigned int)qword_4FDF690;
  }
  *(_QWORD *)(qword_4FDF688 + 8 * v2) = v1;
  qword_4FDF6D0 = (__int64)&unk_49D9728;
  qword_4FDF640 = (__int64)&unk_49DBF10;
  qword_4FDF6E0 = (__int64)&unk_49DC290;
  LODWORD(qword_4FDF690) = qword_4FDF690 + 1;
  qword_4FDF700 = (__int64)nullsub_24;
  qword_4FDF6C8 = 0;
  qword_4FDF6F8 = (__int64)sub_984050;
  qword_4FDF6D8 = 0;
  sub_C53080(&qword_4FDF640, "aggressive-instcombine-max-scan-instrs", 38);
  LODWORD(qword_4FDF6C8) = 64;
  BYTE4(qword_4FDF6D8) = 1;
  LODWORD(qword_4FDF6D8) = 64;
  qword_4FDF670 = 62;
  LOBYTE(dword_4FDF64C) = dword_4FDF64C & 0x9F | 0x20;
  qword_4FDF668 = (__int64)"Max number of instructions to scan for aggressive instcombine.";
  sub_C53130(&qword_4FDF640);
  __cxa_atexit(sub_984970, &qword_4FDF640, &qword_4A427C0);
  qword_4FDF560 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FDF5DC = 1;
  qword_4FDF5B0 = 0x100000000LL;
  dword_4FDF56C &= 0x8000u;
  qword_4FDF5A8 = (__int64)&unk_4FDF5B8;
  qword_4FDF578 = 0;
  qword_4FDF580 = 0;
  dword_4FDF568 = v4;
  word_4FDF570 = 0;
  qword_4FDF588 = 0;
  qword_4FDF590 = 0;
  qword_4FDF598 = 0;
  qword_4FDF5A0 = 0;
  qword_4FDF5C0 = 0;
  qword_4FDF5C8 = (__int64)&unk_4FDF5E0;
  qword_4FDF5D0 = 1;
  dword_4FDF5D8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FDF5B0;
  if ( (unsigned __int64)(unsigned int)qword_4FDF5B0 + 1 > HIDWORD(qword_4FDF5B0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FDF5B8 - 16, &unk_4FDF5B8, (unsigned int)qword_4FDF5B0 + 1LL, 8);
    v6 = (unsigned int)qword_4FDF5B0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FDF5A8 + 8 * v6) = v5;
  qword_4FDF5F0 = (__int64)&unk_49D9728;
  qword_4FDF560 = (__int64)&unk_49DBF10;
  qword_4FDF600 = (__int64)&unk_49DC290;
  LODWORD(qword_4FDF5B0) = qword_4FDF5B0 + 1;
  qword_4FDF620 = (__int64)nullsub_24;
  qword_4FDF5E8 = 0;
  qword_4FDF618 = (__int64)sub_984050;
  qword_4FDF5F8 = 0;
  sub_C53080(&qword_4FDF560, "strncmp-inline-threshold", 24);
  LODWORD(qword_4FDF5E8) = 3;
  BYTE4(qword_4FDF5F8) = 1;
  LODWORD(qword_4FDF5F8) = 3;
  qword_4FDF590 = 116;
  LOBYTE(dword_4FDF56C) = dword_4FDF56C & 0x9F | 0x20;
  qword_4FDF588 = (__int64)"The maximum length of a constant string for a builtin string cmp call eligible for inlining. "
                           "The default value is 3.";
  sub_C53130(&qword_4FDF560);
  __cxa_atexit(sub_984970, &qword_4FDF560, &qword_4A427C0);
  qword_4FDF480 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FDF48C &= 0x8000u;
  word_4FDF490 = 0;
  qword_4FDF4D0 = 0x100000000LL;
  qword_4FDF4C8 = (__int64)&unk_4FDF4D8;
  qword_4FDF498 = 0;
  qword_4FDF4A0 = 0;
  dword_4FDF488 = v7;
  qword_4FDF4A8 = 0;
  qword_4FDF4B0 = 0;
  qword_4FDF4B8 = 0;
  qword_4FDF4C0 = 0;
  qword_4FDF4E0 = 0;
  qword_4FDF4E8 = (__int64)&unk_4FDF500;
  qword_4FDF4F0 = 1;
  dword_4FDF4F8 = 0;
  byte_4FDF4FC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FDF4D0;
  v10 = (unsigned int)qword_4FDF4D0 + 1LL;
  if ( v10 > HIDWORD(qword_4FDF4D0) )
  {
    sub_C8D5F0((char *)&unk_4FDF4D8 - 16, &unk_4FDF4D8, v10, 8);
    v9 = (unsigned int)qword_4FDF4D0;
  }
  *(_QWORD *)(qword_4FDF4C8 + 8 * v9) = v8;
  qword_4FDF510 = (__int64)&unk_49D9728;
  qword_4FDF480 = (__int64)&unk_49DBF10;
  qword_4FDF520 = (__int64)&unk_49DC290;
  LODWORD(qword_4FDF4D0) = qword_4FDF4D0 + 1;
  qword_4FDF540 = (__int64)nullsub_24;
  qword_4FDF508 = 0;
  qword_4FDF538 = (__int64)sub_984050;
  qword_4FDF518 = 0;
  sub_C53080(&qword_4FDF480, "memchr-inline-threshold", 23);
  LODWORD(qword_4FDF508) = 3;
  BYTE4(qword_4FDF518) = 1;
  LODWORD(qword_4FDF518) = 3;
  qword_4FDF4B0 = 64;
  LOBYTE(dword_4FDF48C) = dword_4FDF48C & 0x9F | 0x20;
  qword_4FDF4A8 = (__int64)"The maximum length of a constant string to inline a memchr call.";
  sub_C53130(&qword_4FDF480);
  return __cxa_atexit(sub_984970, &qword_4FDF480, &qword_4A427C0);
}
