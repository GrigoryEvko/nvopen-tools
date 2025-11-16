// Function: ctor_543
// Address: 0x56be20
//
int ctor_543()
{
  sub_D95050(&qword_5016880, 0, 0);
  qword_5016908 = 0;
  qword_5016918 = 0;
  qword_5016910 = (__int64)&unk_49D9728;
  qword_5016880 = (__int64)&unk_49DBF10;
  qword_5016920 = (__int64)&unk_49DC290;
  qword_5016940 = (__int64)nullsub_24;
  qword_5016938 = (__int64)sub_984050;
  sub_C53080(&qword_5016880, "debug-ata-max-blocks", 20);
  LODWORD(qword_5016908) = 10000;
  qword_50168A8 = (__int64)"Maximum num basic blocks before debug info dropped";
  BYTE4(qword_5016918) = 1;
  LODWORD(qword_5016918) = 10000;
  qword_50168B0 = 50;
  byte_501688C = byte_501688C & 0x9F | 0x20;
  sub_C53130(&qword_5016880);
  __cxa_atexit(sub_984970, &qword_5016880, &qword_4A427C0);
  sub_D95050(&qword_50167A0, 0, 0);
  qword_5016828 = 0;
  qword_5016838 = 0;
  qword_5016830 = (__int64)&unk_49D9748;
  qword_50167A0 = (__int64)&unk_49DC090;
  qword_5016840 = (__int64)&unk_49DC1D0;
  qword_5016860 = (__int64)nullsub_23;
  qword_5016858 = (__int64)sub_984030;
  sub_C53080(&qword_50167A0, "mem-loc-frag-fill", 17);
  LOBYTE(qword_5016828) = 1;
  LOWORD(qword_5016838) = 257;
  byte_50167AC = byte_50167AC & 0x9F | 0x20;
  sub_C53130(&qword_50167A0);
  __cxa_atexit(sub_984900, &qword_50167A0, &qword_4A427C0);
  sub_D95050(&qword_50166C0, 0, 0);
  qword_5016750 = (__int64)&unk_49D9748;
  qword_5016780 = (__int64)nullsub_23;
  qword_50166C0 = (__int64)&unk_49DC090;
  qword_5016760 = (__int64)&unk_49DC1D0;
  qword_5016778 = (__int64)sub_984030;
  qword_5016748 = 0;
  qword_5016758 = 0;
  sub_C53080(&qword_50166C0, "print-debug-ata", 15);
  LOWORD(qword_5016758) = 256;
  LOBYTE(qword_5016748) = 0;
  byte_50166CC = byte_50166CC & 0x9F | 0x20;
  sub_C53130(&qword_50166C0);
  __cxa_atexit(sub_984900, &qword_50166C0, &qword_4A427C0);
  sub_D95050(&qword_50165E0, 0, 0);
  qword_5016668 = 0;
  qword_5016678 = 0;
  qword_5016670 = (__int64)&unk_49DC110;
  qword_50165E0 = (__int64)&unk_49D97F0;
  qword_5016680 = (__int64)&unk_49DC200;
  qword_50166A0 = (__int64)nullsub_26;
  qword_5016698 = (__int64)sub_9C26D0;
  sub_C53080(&qword_50165E0, "debug-ata-coalesce-frags", 24);
  byte_50165EC = byte_50165EC & 0x9F | 0x20;
  sub_C53130(&qword_50165E0);
  return __cxa_atexit(sub_9C44F0, &qword_50165E0, &qword_4A427C0);
}
