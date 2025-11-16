// Function: ctor_428
// Address: 0x536730
//
int ctor_428()
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
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx

  qword_4FF43A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF43F0 = 0x100000000LL;
  dword_4FF43AC &= 0x8000u;
  word_4FF43B0 = 0;
  qword_4FF43B8 = 0;
  qword_4FF43C0 = 0;
  dword_4FF43A8 = v0;
  qword_4FF43C8 = 0;
  qword_4FF43D0 = 0;
  qword_4FF43D8 = 0;
  qword_4FF43E0 = 0;
  qword_4FF43E8 = (__int64)&unk_4FF43F8;
  qword_4FF4400 = 0;
  qword_4FF4408 = (__int64)&unk_4FF4420;
  qword_4FF4410 = 1;
  dword_4FF4418 = 0;
  byte_4FF441C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF43F0;
  v3 = (unsigned int)qword_4FF43F0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF43F0) )
  {
    sub_C8D5F0((char *)&unk_4FF43F8 - 16, &unk_4FF43F8, v3, 8);
    v2 = (unsigned int)qword_4FF43F0;
  }
  *(_QWORD *)(qword_4FF43E8 + 8 * v2) = v1;
  LODWORD(qword_4FF43F0) = qword_4FF43F0 + 1;
  qword_4FF4428 = 0;
  qword_4FF4430 = (__int64)&unk_49D9728;
  qword_4FF4438 = 0;
  qword_4FF43A0 = (__int64)&unk_49DBF10;
  qword_4FF4440 = (__int64)&unk_49DC290;
  qword_4FF4460 = (__int64)nullsub_24;
  qword_4FF4458 = (__int64)sub_984050;
  sub_C53080(&qword_4FF43A0, "mergefunc-verify", 16);
  qword_4FF43D0 = 153;
  qword_4FF43C8 = (__int64)"How many functions in a module could be used for MergeFunctions to pass a basic correctness c"
                           "heck. '0' disables this check. Works only with '-debug' key.";
  LODWORD(qword_4FF4428) = 0;
  BYTE4(qword_4FF4438) = 1;
  LODWORD(qword_4FF4438) = 0;
  LOBYTE(dword_4FF43AC) = dword_4FF43AC & 0x9F | 0x20;
  sub_C53130(&qword_4FF43A0);
  __cxa_atexit(sub_984970, &qword_4FF43A0, &qword_4A427C0);
  qword_4FF42C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF4310 = 0x100000000LL;
  dword_4FF42CC &= 0x8000u;
  word_4FF42D0 = 0;
  qword_4FF42D8 = 0;
  qword_4FF42E0 = 0;
  dword_4FF42C8 = v4;
  qword_4FF42E8 = 0;
  qword_4FF42F0 = 0;
  qword_4FF42F8 = 0;
  qword_4FF4300 = 0;
  qword_4FF4308 = (__int64)&unk_4FF4318;
  qword_4FF4320 = 0;
  qword_4FF4328 = (__int64)&unk_4FF4340;
  qword_4FF4330 = 1;
  dword_4FF4338 = 0;
  byte_4FF433C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF4310;
  v7 = (unsigned int)qword_4FF4310 + 1LL;
  if ( v7 > HIDWORD(qword_4FF4310) )
  {
    sub_C8D5F0((char *)&unk_4FF4318 - 16, &unk_4FF4318, v7, 8);
    v6 = (unsigned int)qword_4FF4310;
  }
  *(_QWORD *)(qword_4FF4308 + 8 * v6) = v5;
  qword_4FF4350 = (__int64)&unk_49D9748;
  qword_4FF42C0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF4310) = qword_4FF4310 + 1;
  qword_4FF4348 = 0;
  qword_4FF4360 = (__int64)&unk_49DC1D0;
  qword_4FF4358 = 0;
  qword_4FF4380 = (__int64)nullsub_23;
  qword_4FF4378 = (__int64)sub_984030;
  sub_C53080(&qword_4FF42C0, "mergefunc-preserve-debug-info", 29);
  LOWORD(qword_4FF4358) = 256;
  LOBYTE(qword_4FF4348) = 0;
  qword_4FF42F0 = 69;
  LOBYTE(dword_4FF42CC) = dword_4FF42CC & 0x9F | 0x20;
  qword_4FF42E8 = (__int64)"Preserve debug info in thunk when mergefunc transformations are made.";
  sub_C53130(&qword_4FF42C0);
  __cxa_atexit(sub_984900, &qword_4FF42C0, &qword_4A427C0);
  qword_4FF41E0 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF4230 = 0x100000000LL;
  word_4FF41F0 = 0;
  dword_4FF41EC &= 0x8000u;
  qword_4FF41F8 = 0;
  qword_4FF4200 = 0;
  dword_4FF41E8 = v8;
  qword_4FF4208 = 0;
  qword_4FF4210 = 0;
  qword_4FF4218 = 0;
  qword_4FF4220 = 0;
  qword_4FF4228 = (__int64)&unk_4FF4238;
  qword_4FF4240 = 0;
  qword_4FF4248 = (__int64)&unk_4FF4260;
  qword_4FF4250 = 1;
  dword_4FF4258 = 0;
  byte_4FF425C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FF4230;
  v11 = (unsigned int)qword_4FF4230 + 1LL;
  if ( v11 > HIDWORD(qword_4FF4230) )
  {
    sub_C8D5F0((char *)&unk_4FF4238 - 16, &unk_4FF4238, v11, 8);
    v10 = (unsigned int)qword_4FF4230;
  }
  *(_QWORD *)(qword_4FF4228 + 8 * v10) = v9;
  qword_4FF4270 = (__int64)&unk_49D9748;
  qword_4FF41E0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF4230) = qword_4FF4230 + 1;
  qword_4FF4268 = 0;
  qword_4FF4280 = (__int64)&unk_49DC1D0;
  qword_4FF4278 = 0;
  qword_4FF42A0 = (__int64)nullsub_23;
  qword_4FF4298 = (__int64)sub_984030;
  sub_C53080(&qword_4FF41E0, "mergefunc-use-aliases", 21);
  LOBYTE(qword_4FF4268) = 0;
  qword_4FF4210 = 33;
  LOBYTE(dword_4FF41EC) = dword_4FF41EC & 0x9F | 0x20;
  LOWORD(qword_4FF4278) = 256;
  qword_4FF4208 = (__int64)"Allow mergefunc to create aliases";
  sub_C53130(&qword_4FF41E0);
  return __cxa_atexit(sub_984900, &qword_4FF41E0, &qword_4A427C0);
}
