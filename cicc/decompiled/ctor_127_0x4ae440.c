// Function: ctor_127
// Address: 0x4ae440
//
int ctor_127()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4F99BE0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F99BEC &= 0xF000u;
  qword_4F99C28 = (__int64)&unk_4FA01C0;
  qword_4F99C38 = (__int64)&unk_4F99C58;
  qword_4F99C40 = (__int64)&unk_4F99C58;
  qword_4F99BF0 = 0;
  dword_4F99BE8 = v0;
  qword_4F99C88 = (__int64)&unk_49E74E8;
  qword_4F99BF8 = 0;
  qword_4F99C00 = 0;
  qword_4F99BE0 = (__int64)&unk_49EAB58;
  qword_4F99C08 = 0;
  qword_4F99C10 = 0;
  qword_4F99C98 = (__int64)&unk_49EEDB0;
  qword_4F99C18 = 0;
  qword_4F99C20 = 0;
  qword_4F99C30 = 0;
  qword_4F99C48 = 4;
  dword_4F99C50 = 0;
  byte_4F99C78 = 0;
  qword_4F99C80 = 0;
  byte_4F99C91 = 0;
  sub_16B8280(&qword_4F99BE0, "enable-objc-arc-opts", 20);
  qword_4F99C10 = 36;
  qword_4F99C08 = (__int64)"enable/disable all ARC Optimizations";
  if ( qword_4F99C80 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4F99BE0, &v3, 0, 0, v1);
  }
  else
  {
    qword_4F99C80 = (__int64)&unk_4F99CA8;
  }
  *(_BYTE *)qword_4F99C80 = 1;
  unk_4F99C90 = 257;
  LOBYTE(word_4F99BEC) = word_4F99BEC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F99BE0);
  return __cxa_atexit(sub_13F9A70, &qword_4F99BE0, &qword_4A427C0);
}
