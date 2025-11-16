// Function: ctor_418
// Address: 0x5311b0
//
int ctor_418()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4FF0240 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  word_4FF0250 = 0;
  qword_4FF0258 = 0;
  qword_4FF0260 = 0;
  dword_4FF024C = dword_4FF024C & 0x8000 | 1;
  qword_4FF0290 = 0x100000000LL;
  dword_4FF0248 = v0;
  qword_4FF0268 = 0;
  qword_4FF0270 = 0;
  qword_4FF0278 = 0;
  qword_4FF0280 = 0;
  qword_4FF0288 = (__int64)&unk_4FF0298;
  qword_4FF02A0 = 0;
  qword_4FF02A8 = (__int64)&unk_4FF02C0;
  qword_4FF02B0 = 1;
  dword_4FF02B8 = 0;
  byte_4FF02BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF0290;
  v3 = (unsigned int)qword_4FF0290 + 1LL;
  if ( v3 > HIDWORD(qword_4FF0290) )
  {
    sub_C8D5F0((char *)&unk_4FF0298 - 16, &unk_4FF0298, v3, 8);
    v2 = (unsigned int)qword_4FF0290;
  }
  *(_QWORD *)(qword_4FF0288 + 8 * v2) = v1;
  qword_4FF0240 = (__int64)&unk_49DAD08;
  LODWORD(qword_4FF0290) = qword_4FF0290 + 1;
  qword_4FF0318 = (__int64)&unk_49DC350;
  qword_4FF02C8 = 0;
  qword_4FF0338 = (__int64)nullsub_81;
  qword_4FF02D0 = 0;
  qword_4FF0330 = (__int64)sub_BB8600;
  qword_4FF02D8 = 0;
  qword_4FF02E0 = 0;
  qword_4FF02E8 = 0;
  qword_4FF02F0 = 0;
  byte_4FF02F8 = 0;
  qword_4FF0300 = 0;
  qword_4FF0308 = 0;
  qword_4FF0310 = 0;
  sub_C53080(&qword_4FF0240, "force-attribute", 15);
  qword_4FF0270 = 306;
  LOBYTE(dword_4FF024C) = dword_4FF024C & 0x9F | 0x20;
  qword_4FF0268 = (__int64)"Add an attribute to a function. This can be a pair of 'function-name:attribute-name', to appl"
                           "y an attribute to a specific function. For example -force-attribute=foo:noinline. Specifying "
                           "only an attribute will apply the attribute to every function in the module. This option can b"
                           "e specified multiple times.";
  sub_C53130(&qword_4FF0240);
  __cxa_atexit(sub_BB89D0, &qword_4FF0240, &qword_4A427C0);
  qword_4FF0140 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF0158 = 0;
  qword_4FF0160 = 0;
  qword_4FF0168 = 0;
  qword_4FF0170 = 0;
  dword_4FF014C = dword_4FF014C & 0x8000 | 1;
  qword_4FF0190 = 0x100000000LL;
  dword_4FF0148 = v4;
  word_4FF0150 = 0;
  qword_4FF0178 = 0;
  qword_4FF0180 = 0;
  qword_4FF0188 = (__int64)&unk_4FF0198;
  qword_4FF01A0 = 0;
  qword_4FF01A8 = (__int64)&unk_4FF01C0;
  qword_4FF01B0 = 1;
  dword_4FF01B8 = 0;
  byte_4FF01BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF0190;
  if ( (unsigned __int64)(unsigned int)qword_4FF0190 + 1 > HIDWORD(qword_4FF0190) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FF0198 - 16, &unk_4FF0198, (unsigned int)qword_4FF0190 + 1LL, 8);
    v6 = (unsigned int)qword_4FF0190;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FF0188 + 8 * v6) = v5;
  qword_4FF0140 = (__int64)&unk_49DAD08;
  qword_4FF0218 = (__int64)&unk_49DC350;
  LODWORD(qword_4FF0190) = qword_4FF0190 + 1;
  qword_4FF0238 = (__int64)nullsub_81;
  qword_4FF01C8 = 0;
  qword_4FF0230 = (__int64)sub_BB8600;
  qword_4FF01D0 = 0;
  qword_4FF01D8 = 0;
  qword_4FF01E0 = 0;
  qword_4FF01E8 = 0;
  qword_4FF01F0 = 0;
  byte_4FF01F8 = 0;
  qword_4FF0200 = 0;
  qword_4FF0208 = 0;
  qword_4FF0210 = 0;
  sub_C53080(&qword_4FF0140, "force-remove-attribute", 22);
  qword_4FF0170 = 322;
  LOBYTE(dword_4FF014C) = dword_4FF014C & 0x9F | 0x20;
  qword_4FF0168 = (__int64)"Remove an attribute from a function. This can be a pair of 'function-name:attribute-name' to "
                           "remove an attribute from a specific function. For example -force-remove-attribute=foo:noinlin"
                           "e. Specifying only an attribute will remove the attribute from all functions in the module. T"
                           "his option can be specified multiple times.";
  sub_C53130(&qword_4FF0140);
  __cxa_atexit(sub_BB89D0, &qword_4FF0140, &qword_4A427C0);
  qword_4FF0040 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF004C &= 0x8000u;
  word_4FF0050 = 0;
  qword_4FF0090 = 0x100000000LL;
  qword_4FF0058 = 0;
  qword_4FF0060 = 0;
  qword_4FF0068 = 0;
  dword_4FF0048 = v7;
  qword_4FF0070 = 0;
  qword_4FF0078 = 0;
  qword_4FF0080 = 0;
  qword_4FF0088 = (__int64)&unk_4FF0098;
  qword_4FF00A0 = 0;
  qword_4FF00A8 = (__int64)&unk_4FF00C0;
  qword_4FF00B0 = 1;
  dword_4FF00B8 = 0;
  byte_4FF00BC = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FF0090;
  v10 = (unsigned int)qword_4FF0090 + 1LL;
  if ( v10 > HIDWORD(qword_4FF0090) )
  {
    sub_C8D5F0((char *)&unk_4FF0098 - 16, &unk_4FF0098, v10, 8);
    v9 = (unsigned int)qword_4FF0090;
  }
  *(_QWORD *)(qword_4FF0088 + 8 * v9) = v8;
  qword_4FF00C8 = (__int64)&byte_4FF00D8;
  qword_4FF00F0 = (__int64)&byte_4FF0100;
  LODWORD(qword_4FF0090) = qword_4FF0090 + 1;
  qword_4FF00D0 = 0;
  qword_4FF00E8 = (__int64)&unk_49DC130;
  byte_4FF00D8 = 0;
  byte_4FF0100 = 0;
  qword_4FF0040 = (__int64)&unk_49DC010;
  qword_4FF0118 = (__int64)&unk_49DC350;
  qword_4FF00F8 = 0;
  qword_4FF0138 = (__int64)nullsub_92;
  byte_4FF0110 = 0;
  qword_4FF0130 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FF0040, "forceattrs-csv-path", 19);
  qword_4FF0070 = 126;
  LOBYTE(dword_4FF004C) = dword_4FF004C & 0x9F | 0x20;
  qword_4FF0068 = (__int64)"Path to CSV file containing lines of function names and attributes to add to them in the form"
                           " of `f1,attr1` or `f2,attr2=str`.";
  sub_C53130(&qword_4FF0040);
  return __cxa_atexit(sub_BC5A40, &qword_4FF0040, &qword_4A427C0);
}
