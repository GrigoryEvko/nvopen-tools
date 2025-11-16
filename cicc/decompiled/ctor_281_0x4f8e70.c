// Function: ctor_281
// Address: 0x4f8e70
//
int ctor_281()
{
  qword_4FC0A08 = (__int64)"fast";
  qword_4FC0A18 = (__int64)"Fast suboptimal list scheduling";
  qword_4FC0A00 = 0;
  qword_4FC0A28 = (__int64)sub_1CFBF70;
  qword_4FC0A10 = 4;
  qword_4FC0A20 = 31;
  sub_1E40390(&unk_4FC1B10, &qword_4FC0A00);
  __cxa_atexit(sub_1CFC0C0, &qword_4FC0A00, &qword_4A427C0);
  qword_4FC09C8 = (__int64)"linearize";
  qword_4FC09D8 = (__int64)"Linearize DAG, no scheduling";
  qword_4FC09C0 = 0;
  qword_4FC09D0 = 9;
  qword_4FC09E0 = 28;
  qword_4FC09E8 = (__int64)sub_1CFC020;
  sub_1E40390(&unk_4FC1B10, &qword_4FC09C0);
  return __cxa_atexit(sub_1CFC0C0, &qword_4FC09C0, &qword_4A427C0);
}
