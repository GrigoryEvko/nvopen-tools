// Function: ctor_236
// Address: 0x4ec170
//
int ctor_236()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4FB6600 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB660C &= 0xF000u;
  qword_4FB6648 = (__int64)qword_4FA01C0;
  qword_4FB6658 = (__int64)&unk_4FB6678;
  qword_4FB6660 = (__int64)&unk_4FB6678;
  qword_4FB6610 = 0;
  dword_4FB6608 = v0;
  qword_4FB66A8 = (__int64)&unk_49E74E8;
  qword_4FB6618 = 0;
  qword_4FB6620 = 0;
  qword_4FB6600 = (__int64)&unk_49EAB58;
  qword_4FB6628 = 0;
  qword_4FB6630 = 0;
  qword_4FB6638 = 0;
  qword_4FB6640 = 0;
  qword_4FB6650 = 0;
  qword_4FB6668 = 4;
  dword_4FB6670 = 0;
  byte_4FB6698 = 0;
  qword_4FB66A0 = 0;
  byte_4FB66B1 = 0;
  qword_4FB66B8 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB6600, "verify-loop-lcssa", 17);
  if ( qword_4FB66A0 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4FB6600, &v3, 0, 0, v1);
  }
  else
  {
    byte_4FB66B1 = 1;
    qword_4FB66A0 = (__int64)&byte_4FB66C8;
    byte_4FB66B0 = byte_4FB66C8;
  }
  qword_4FB6630 = 39;
  LOBYTE(word_4FB660C) = word_4FB660C & 0x9F | 0x20;
  qword_4FB6628 = (__int64)"Verify loop lcssa form (time consuming)";
  sub_16B88A0(&qword_4FB6600);
  return __cxa_atexit(sub_13F9A70, &qword_4FB6600, &qword_4A427C0);
}
