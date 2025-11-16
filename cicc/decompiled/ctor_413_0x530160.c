// Function: ctor_413
// Address: 0x530160
//
int ctor_413()
{
  __int64 v0; // r14
  __int64 v1; // rax
  _QWORD v3[2]; // [rsp+0h] [rbp-80h] BYREF
  char v4; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v6[80]; // [rsp+30h] [rbp-50h] BYREF

  v0 = sub_C60B10();
  v5[0] = v6;
  sub_2539970(v5, "How many AAs should be initialized");
  v3[0] = &v4;
  sub_2539970(v3, "num-abstract-attributes");
  sub_CF9810(v0, v3, v5);
  sub_2240A30(v3);
  sub_2240A30(v5);
  sub_D95050(&qword_4FEF960, 0, 0);
  qword_4FEF9E8 = 0;
  qword_4FEF9F8 = 0;
  qword_4FEF9F0 = (__int64)&unk_49D9748;
  qword_4FEF960 = (__int64)&unk_49DC090;
  qword_4FEFA00 = (__int64)&unk_49DC1D0;
  qword_4FEFA20 = (__int64)nullsub_23;
  qword_4FEFA18 = (__int64)sub_984030;
  sub_C53080(&qword_4FEF960, "attributor-manifest-internal", 28);
  qword_4FEF990 = 47;
  LOBYTE(qword_4FEF9E8) = 0;
  byte_4FEF96C = byte_4FEF96C & 0x9F | 0x20;
  qword_4FEF988 = (__int64)"Manifest Attributor internal string attributes.";
  LOWORD(qword_4FEF9F8) = 256;
  sub_C53130(&qword_4FEF960);
  __cxa_atexit(sub_984900, &qword_4FEF960, &qword_4A427C0);
  sub_D95050(&qword_4FEF880, 0, 0);
  qword_4FEF908 = 0;
  qword_4FEF918 = 0;
  qword_4FEF910 = (__int64)&unk_49DA090;
  qword_4FEF880 = (__int64)&unk_49DBF90;
  qword_4FEF920 = (__int64)&unk_49DC230;
  qword_4FEF940 = (__int64)nullsub_58;
  qword_4FEF938 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FEF880, "max-heap-to-stack-size", 22);
  LODWORD(qword_4FEF908) = 128;
  BYTE4(qword_4FEF918) = 1;
  LODWORD(qword_4FEF918) = 128;
  byte_4FEF88C = byte_4FEF88C & 0x9F | 0x20;
  sub_C53130(&qword_4FEF880);
  __cxa_atexit(sub_B2B680, &qword_4FEF880, &qword_4A427C0);
  sub_D95050(&qword_4FEF7A0, 0, 0);
  byte_4FEF83C = 0;
  qword_4FEF828 = 0;
  qword_4FEF830 = (__int64)&unk_49D9728;
  qword_4FEF7A0 = (__int64)&unk_49DDF20;
  qword_4FEF840 = (__int64)&unk_49DC290;
  qword_4FEF860 = (__int64)nullsub_186;
  qword_4FEF858 = (__int64)sub_D320E0;
  sub_C53080(&qword_4FEF7A0, "attributor-max-potential-values", 31);
  qword_4FEF7D0 = 67;
  byte_4FEF7AC = byte_4FEF7AC & 0x9F | 0x20;
  qword_4FEF7C8 = (__int64)"Maximum number of potential values to be tracked for each position.";
  if ( qword_4FEF828 )
  {
    v1 = sub_CEADF0();
    v6[17] = 1;
    v5[0] = "cl::location(x) specified more than once!";
    v6[16] = 3;
    sub_C53280(&qword_4FEF7A0, v5, 0, 0, v1);
  }
  else
  {
    qword_4FEF828 = (__int64)&unk_4FEF868;
  }
  *(_DWORD *)qword_4FEF828 = 7;
  byte_4FEF83C = 1;
  dword_4FEF838 = 7;
  sub_C53130(&qword_4FEF7A0);
  __cxa_atexit(sub_D32600, &qword_4FEF7A0, &qword_4A427C0);
  sub_D95050(&qword_4FEF6C0, 0, 0);
  qword_4FEF750 = (__int64)&unk_49DA090;
  qword_4FEF6C0 = (__int64)&unk_49DBF90;
  qword_4FEF760 = (__int64)&unk_49DC230;
  qword_4FEF748 = 0;
  qword_4FEF780 = (__int64)nullsub_58;
  qword_4FEF758 = 0;
  qword_4FEF778 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FEF6C0, "attributor-max-potential-values-iterations", 42);
  qword_4FEF6F0 = 66;
  LODWORD(qword_4FEF748) = 64;
  BYTE4(qword_4FEF758) = 1;
  LODWORD(qword_4FEF758) = 64;
  byte_4FEF6CC = byte_4FEF6CC & 0x9F | 0x20;
  qword_4FEF6E8 = (__int64)"Maximum number of iterations we keep dismantling potential values.";
  sub_C53130(&qword_4FEF6C0);
  return __cxa_atexit(sub_B2B680, &qword_4FEF6C0, &qword_4A427C0);
}
