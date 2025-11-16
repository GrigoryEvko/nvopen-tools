// Function: ctor_034
// Address: 0x48c880
//
int ctor_034()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_4F83600 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8360C = word_4F8360C & 0x8000;
  unk_4F83608 = v0;
  qword_4F83648[1] = 0x100000000LL;
  unk_4F83610 = 0;
  unk_4F83618 = 0;
  unk_4F83620 = 0;
  unk_4F83628 = 0;
  unk_4F83630 = 0;
  unk_4F83638 = 0;
  unk_4F83640 = 0;
  qword_4F83648[0] = &qword_4F83648[2];
  qword_4F83648[3] = 0;
  qword_4F83648[4] = &qword_4F83648[7];
  qword_4F83648[5] = 1;
  LODWORD(qword_4F83648[6]) = 0;
  BYTE4(qword_4F83648[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F83648[1]);
  if ( (unsigned __int64)LODWORD(qword_4F83648[1]) + 1 > HIDWORD(qword_4F83648[1]) )
  {
    sub_C8D5F0(qword_4F83648, &qword_4F83648[2], LODWORD(qword_4F83648[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F83648[1]);
  }
  *(_QWORD *)(qword_4F83648[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F83648[1]);
  qword_4F83648[8] = 0;
  qword_4F83648[9] = &unk_49D9748;
  qword_4F83648[10] = 0;
  qword_4F83600 = &unk_49DC090;
  qword_4F83648[11] = &unk_49DC1D0;
  qword_4F83648[15] = nullsub_23;
  qword_4F83648[14] = sub_984030;
  sub_C53080(&qword_4F83600, "use-dereferenceable-at-point-semantics", 38);
  LOBYTE(qword_4F83648[8]) = 0;
  unk_4F83630 = 60;
  LOBYTE(word_4F8360C) = word_4F8360C & 0x9F | 0x20;
  LOWORD(qword_4F83648[10]) = 256;
  unk_4F83628 = "Deref attributes and metadata infer facts at definition only";
  sub_C53130(&qword_4F83600);
  return __cxa_atexit(sub_984900, &qword_4F83600, &qword_4A427C0);
}
